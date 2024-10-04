import Mathlib

namespace probability_at_most_3_heads_l148_148437

def binom (n k : ℕ) : ℕ := Nat.choose n k

theorem probability_at_most_3_heads :
  let total_outcomes := 2^10
  let successful_outcomes := binom 10 0 + binom 10 1 + binom 10 2 + binom 10 3
  (successful_outcomes : ℚ) / total_outcomes = 11 / 64 := 
by 
  let total_outcomes := 2 ^ 10
  let successful_outcomes := binom 10 0 + binom 10 1 + binom 10 2 + binom 10 3
  have h : (successful_outcomes : ℚ) / total_outcomes = 11 / 64 := sorry
  exact h

end probability_at_most_3_heads_l148_148437


namespace probability_at_most_three_heads_10_coins_l148_148356

open Nat

noncomputable def probability_at_most_three_heads (n : ℕ) (k : ℕ) : ℚ :=
  (∑ i in Finset.range (k + 1), (Nat.choose n i : ℚ)) / (2 ^ n : ℚ)

theorem probability_at_most_three_heads_10_coins :
  probability_at_most_three_heads 10 3 = 11 / 64 :=
by
  sorry

end probability_at_most_three_heads_10_coins_l148_148356


namespace tangent_line_at_point_2_tangent_lines_parallel_to_5x_plus_3_l148_148721

noncomputable def f (x : ℝ) : ℝ := x^3 - x

theorem tangent_line_at_point_2 :
  let slope := (3 * (2 : ℝ)^2 - 1)
  let y₂ := f 2
  let equation := λ x y : ℝ, slope * x - y - (slope * 2 - y₂) = 0
  (equation 2 (f 2)) = 11 * 2 - (f 2) - 16 :=
by
  sorry

theorem tangent_lines_parallel_to_5x_plus_3 :
  let x₀₁ := Real.sqrt 2
  let x₀₂ := -Real.sqrt 2
  let slope₀ := 5
  let tangent_point₁ := (x₀₁, f x₀₁)
  let tangent_point₂ := (x₀₂, f x₀₂)
  let equation₁ := λ x y : ℝ, slope₀ * x - y - (slope₀ * x₀₁ - (f x₀₁)) = 0
  let equation₂ := λ x y : ℝ, slope₀ * x - y - (slope₀ * x₀₂ - (f x₀₂)) = 0
  (equation₁ x₀₁ (f x₀₁) = 5 * x₀₁ - (f x₀₁) - 4 * Real.sqrt 2)
  ∧ (equation₂ x₀₂ (f x₀₂) = 5 * x₀₂ - (f x₀₂) + 4 * Real.sqrt 2) :=
by
  sorry

end tangent_line_at_point_2_tangent_lines_parallel_to_5x_plus_3_l148_148721


namespace max_x_coordinate_of_cos_2theta_l148_148055

noncomputable def maximum_x_coordinate (theta : ℝ) : ℝ :=
  let u := cos theta
  let x := 4 * u ^ 3 - 2 * u
  x

theorem max_x_coordinate_of_cos_2theta :
  ∃ (theta : ℝ), maximum_x_coordinate theta = (4 * Real.sqrt 6 / 9) :=
by
  sorry

end max_x_coordinate_of_cos_2theta_l148_148055


namespace sum_distances_XA_XB_XC_l148_148888

-- Define the lengths of the sides of the triangle
def AB : ℝ := 15
def BC : ℝ := 18
def AC : ℝ := 21

-- Define the ratios in which points D, E, F divide the sides
def ratio_AD_DB : ℝ := 1 / 3
def ratio_BE_EC : ℝ := 2 / 3
def ratio_CF_FA : ℝ := 1 / 3

-- Prove that XA + XB + XC equals the given value under the given conditions
theorem sum_distances_XA_XB_XC 
  (D E F X : Type) 
  (h1 : X ≠ E) 
  (h2 : (∀ s t : ℝ, s / t = ratio_AD_DB ∧ s + t = AB) → (D = s)) 
  (h3 : (∀ s t : ℝ, s / t = ratio_BE_EC ∧ s + t = BC) → (E = s)) 
  (h4 : (∀ s t : ℝ, s / t = ratio_CF_FA ∧ s + t = AC) → (F = s)) 
  (h5 : is_circumcircle Π T1 T2 : Type, T1 = (BDE) ∧ T2 = (CEF))
  (R_BDE : ℝ) 
  (R_CEF : ℝ) 
  (h6 : R_BDE = (10 * 12 * 6) / (4 * 32))
  (h7 : R_CEF = (7 * 18 * 12) / (4 * 30.3)) :
  XA + XB + XC = 36.21 :=
by
  sorry

end sum_distances_XA_XB_XC_l148_148888


namespace direction_vector_of_line_and_cosine_angle_l148_148780

-- Definitions from the conditions in the original problem
variables (x y z : ℝ) (α : ℝ → ℝ → ℝ → Prop)

def plane_eq (α : ℝ → ℝ → ℝ → Prop) : Prop :=
  ∀ x y z, α x y z ↔ 3*x + y - z = 5

def line_eq : (ℝ × ℝ × ℝ) → Prop :=
  λ (p : ℝ × ℝ × ℝ), p.1 = p.2/2 ∧ p.1 = -p.3

axiom P : ℝ × ℝ × ℝ
axiom alpha : ℝ → ℝ → ℝ → Prop

noncomputable def normal_vector_plane : ℝ × ℝ × ℝ := (3, 1, -1)

-- The required direction vector of the line
def direction_vector_line : ℝ × ℝ × ℝ := (1, 2, -1)

-- To prove that the cosine of the angle between this line and plane
noncomputable def cos_angle : ℝ :=
  let m := normal_vector_plane in
  let n := direction_vector_line in
  let dot_product := m.1 * n.1 + m.2 * n.2 + m.3 * n.3 in
  let n_magnitude := real.sqrt (m.1^2 + m.2^2 + m.3^2) in
  let l_magnitude := real.sqrt (n.1^2 + n.2^2 + n.3^2) in
  dot_product / (n_magnitude * l_magnitude)

theorem direction_vector_of_line_and_cosine_angle :
  direction_vector_line = (1, 2, -1) ∧ cos_angle = real.sqrt 55 / 11 :=
sorry

end direction_vector_of_line_and_cosine_angle_l148_148780


namespace probability_of_at_most_3_heads_l148_148409

-- Definitions and conditions
def num_coins : ℕ := 10
def at_most_3_heads_probability : ℚ := 11 / 64

-- Statement of the problem
theorem probability_of_at_most_3_heads (n : ℕ) (p : ℚ) (h1 : n = num_coins) (h2 : p = at_most_3_heads_probability) :
  p = (1 + 10 + 45 + 120 : ℕ) / (2 ^ 10 : ℕ) := by
  sorry

end probability_of_at_most_3_heads_l148_148409


namespace operation_example_result_l148_148139

def myOperation (A B : ℕ) : ℕ := (A^2 + B^2) / 3

theorem operation_example_result : myOperation (myOperation 6 3) 9 = 102 := by
  sorry

end operation_example_result_l148_148139


namespace probability_at_most_3_heads_l148_148431

def binom (n k : ℕ) : ℕ := Nat.choose n k

theorem probability_at_most_3_heads :
  let total_outcomes := 2^10
  let successful_outcomes := binom 10 0 + binom 10 1 + binom 10 2 + binom 10 3
  (successful_outcomes : ℚ) / total_outcomes = 11 / 64 := 
by 
  let total_outcomes := 2 ^ 10
  let successful_outcomes := binom 10 0 + binom 10 1 + binom 10 2 + binom 10 3
  have h : (successful_outcomes : ℚ) / total_outcomes = 11 / 64 := sorry
  exact h

end probability_at_most_3_heads_l148_148431


namespace average_weight_l148_148187

theorem average_weight (Ishmael Ponce Jalen : ℝ) 
  (h1 : Ishmael = Ponce + 20) 
  (h2 : Ponce = Jalen - 10) 
  (h3 : Jalen = 160) : 
  (Ishmael + Ponce + Jalen) / 3 = 160 := 
by 
  sorry

end average_weight_l148_148187


namespace area_convex_polygon_lt_pi_l148_148073

theorem area_convex_polygon_lt_pi (T : Type) [convex_polygon T] :
  (∀ (P : vertex T), ∃ (Q : perimeter T), bisects_area P Q ∧ length PQ ≤ 2) →
  area T < π :=
begin
  sorry
end

end area_convex_polygon_lt_pi_l148_148073


namespace probability_at_most_3_heads_10_coins_l148_148425

noncomputable def coin_flip_prob_at_most_3_heads : ℚ :=
  let total_outcomes := 2^10
  let favorable_outcomes := Nat.binomial 10 0 + Nat.binomial 10 1 + Nat.binomial 10 2 + Nat.binomial 10 3
  favorable_outcomes / total_outcomes

theorem probability_at_most_3_heads_10_coins : coin_flip_prob_at_most_3_heads = 11 / 64 :=
by
  -- Definitions of let values are integrated directly in the theorem body
  let total_outcomes := 2^10
  let favorable_outcomes := Nat.binomial 10 0 + Nat.binomial 10 1 + Nat.binomial 10 2 + Nat.binomial 10 3
  show favorable_outcomes / total_outcomes = 11 / 64
  sorry

end probability_at_most_3_heads_10_coins_l148_148425


namespace distance_lima_caracas_l148_148990

-- Definitions of the coordinates in the complex plane
def caracas : ℂ := 0
def buenos_aires : ℂ := 3200 * complex.I
def lima : ℂ := 960 + 1280 * complex.I

-- Theorem statement for the distance from Lima to Caracas
theorem distance_lima_caracas :
  complex.abs (lima - caracas) = 1600 :=
by
  -- We state the proof steps and conditions here, but end with "sorry"
  sorry

end distance_lima_caracas_l148_148990


namespace average_visitors_per_day_in_month_of_30_days_l148_148329

theorem average_visitors_per_day_in_month_of_30_days :
  let total_days := 30
  let sundays := 5
  let other_days := 25
  let sunday_visitors := 510
  let other_day_visitors := 240
  ∃ avg_visitors_per_day : ℕ,
    avg_visitors_per_day = 
      (sundays * sunday_visitors + other_days * other_day_visitors) / total_days :=
begin
  sorry
end

end average_visitors_per_day_in_month_of_30_days_l148_148329


namespace three_players_same_number_of_flips_l148_148885

noncomputable def biased_coin_outcome (heads_prob tails_prob : ℚ) (n : ℕ) : ℚ :=
  (tails_prob)^(n - 1) * heads_prob

theorem three_players_same_number_of_flips :
  let heads_prob : ℚ := 1 / 3
  let tails_prob : ℚ := 2 / 3
  let P (n : ℕ) : ℚ := (biased_coin_outcome heads_prob tails_prob n)^3
  (∑ n in filter (λ n, n ≥ 1) (range ∞), P n) = 1 / 19 :=
by sorry

end three_players_same_number_of_flips_l148_148885


namespace trajectory_of_M_l148_148704

-- Define Point
structure Point where
  x : ℝ
  y : ℝ

-- Define points A and B
def A : Point := { x := -1, y := 0 }
def B : Point := { x := 1, y := 0 }

-- Define the slope calculation function
noncomputable def slope (p1 p2 : Point) : ℝ :=
  if p1.x = p2.x then 0 else (p2.y - p1.y) / (p2.x - p1.x)

-- The main theorem statement
theorem trajectory_of_M
  (M : Point)
  (hM : M.y ≠ 0)
  (h_ratio : slope A M / slope B M = 3) : M.x = -2 ∧ M ≠ { x := -2, y := 0 } :=
by
  sorry

end trajectory_of_M_l148_148704


namespace sum_alternating_powers_of_neg1_l148_148293

theorem sum_alternating_powers_of_neg1 : 
  ∑ k in (Finset.range 2011).map (Nat.succ), (-1 : ℤ) ^ k = -1 := by
  sorry

end sum_alternating_powers_of_neg1_l148_148293


namespace angle_between_OA_OB_range_of_λ_l148_148736

-- Question (1) conditions
def OA (λ α : ℝ) : ℝ × ℝ := (λ * Real.cos α, λ * Real.sin α)
def OB (β : ℝ) : ℝ × ℝ := (-Real.sin β, Real.cos β)

-- Question (2) condition
def BA (λ α β : ℝ) : ℝ × ℝ := (λ * Real.cos α + Real.sin β, λ * Real.sin α - Real.cos β)

-- Definitions of vectors for specific α and β
noncomputable def OA1 := OA 1 (Real.pi / 2)
noncomputable def OB1 := OB (Real.pi / 3)

-- Theorem stating the angle calculation for question (1)
theorem angle_between_OA_OB :
  ∠ (OA1) (OB1) = Real.pi / 3 :=
sorry

-- Theorem for question (2)
theorem range_of_λ (α β : ℝ) (h : α - β = Real.pi / 2) :
  ∀ λ : ℝ, (λ^2 - 2*λ - 3 ≥ 0) ↔ (λ ≤ -1 ∨ λ ≥ 3) :=
sorry

end angle_between_OA_OB_range_of_λ_l148_148736


namespace probability_heads_at_most_three_out_of_ten_coins_l148_148545

theorem probability_heads_at_most_three_out_of_ten_coins :
  let total_outcomes := 2^10 in
  let favorable_outcomes := Nat.choose 10 0 + Nat.choose 10 1 + Nat.choose 10 2 + Nat.choose 10 3 in
  favorable_outcomes / total_outcomes = 11 / 64 :=
by
  sorry

end probability_heads_at_most_three_out_of_ten_coins_l148_148545


namespace sally_bread_consumption_l148_148834

theorem sally_bread_consumption :
  (2 * 2) + (1 * 2) = 6 :=
by
  sorry

end sally_bread_consumption_l148_148834


namespace focal_length_of_given_ellipse_l148_148670

noncomputable def focal_length_of_ellipse (a b : ℝ) : ℝ :=
  let c := real.sqrt (a^2 - b^2)
  2 * c

theorem focal_length_of_given_ellipse : focal_length_of_ellipse 3 (real.sqrt 8) = 2 :=
by
  -- Introduce variables a and b
  let a := 3
  let b := real.sqrt 8

  -- Define c using the given relationship
  let c := real.sqrt (a^2 - b^2)

  -- Calculate the focal length
  have h : focal_length_of_ellipse a b = 2 * c, by rfl

  -- Prove that c = 1
  have hc : c = 1, by
    unfold c
    have ha : a^2 = 9, by norm_num
    have hb : b^2 = 8, by norm_num
    simp [ha, hb]
    exact rfl

  -- Conclude that the focal length is 2
  rw hc at h
  simp at h
  exact h

end focal_length_of_given_ellipse_l148_148670


namespace remaining_pieces_l148_148039

variables (sh_s : Nat) (sh_f : Nat) (sh_r : Nat)
variables (sh_pairs_s : Nat) (sh_pairs_f : Nat) (sh_pairs_r : Nat)
variables (total : Nat)

def conditions :=
  sh_s = 20 ∧
  sh_f = 12 ∧
  sh_r = 20 - 12 ∧
  sh_pairs_s = 8 ∧
  sh_pairs_f = 5 ∧
  sh_pairs_r = 8 - 5 ∧
  total = sh_r + sh_pairs_r

theorem remaining_pieces : conditions → total = 11 :=
by intro h; cases h with _ h'; cases h' with _ h''; cases h'' with _ h''';
   cases h''' with _ h''''; cases h'''' with _ h''''' ; cases h''''' with _ _ ;
   sorry

end remaining_pieces_l148_148039


namespace probability_at_most_three_heads_10_coins_l148_148359

open Nat

noncomputable def probability_at_most_three_heads (n : ℕ) (k : ℕ) : ℚ :=
  (∑ i in Finset.range (k + 1), (Nat.choose n i : ℚ)) / (2 ^ n : ℚ)

theorem probability_at_most_three_heads_10_coins :
  probability_at_most_three_heads 10 3 = 11 / 64 :=
by
  sorry

end probability_at_most_three_heads_10_coins_l148_148359


namespace probability_at_most_three_heads_10_coins_l148_148361

open Nat

noncomputable def probability_at_most_three_heads (n : ℕ) (k : ℕ) : ℚ :=
  (∑ i in Finset.range (k + 1), (Nat.choose n i : ℚ)) / (2 ^ n : ℚ)

theorem probability_at_most_three_heads_10_coins :
  probability_at_most_three_heads 10 3 = 11 / 64 :=
by
  sorry

end probability_at_most_three_heads_10_coins_l148_148361


namespace probability_heads_at_most_3_l148_148539

theorem probability_heads_at_most_3 (num_coins : ℕ) (num_heads_at_most : ℕ) (probability : ℚ):
  num_coins = 10 → 
  num_heads_at_most = 3 → 
  probability = (binomial 10 0 + binomial 10 1 + binomial 10 2 + binomial 10 3) / (2^10) →
  probability = (11 / 64) := 
by
  intros hnum_coins hnum_heads_at_most hprobability
  sorry

end probability_heads_at_most_3_l148_148539


namespace a_7_eq_64_l148_148697

-- Define the problem conditions using variables in Lean
variable {a : ℕ → ℝ} -- defining the sequence as a function from natural numbers to reals
variable {q : ℝ}  -- common ratio

-- The sequence is geometric
axiom geom_seq (n : ℕ) : a (n + 1) = a n * q

-- Conditions given in the problem
axiom condition1 : a 1 + a 2 = 3
axiom condition2 : a 2 + a 3 = 6

-- Target statement to prove
theorem a_7_eq_64 : a 7 = 64 := 
sorry

end a_7_eq_64_l148_148697


namespace scatter_plot_b_value_l148_148164

theorem scatter_plot_b_value
  (x y : Fin 6 → ℝ)
  (h_curve : ∀ i, y i = b * (x i) ^ 2 - 1)
  (h_sum_x : (∑ i, x i) = 11)
  (h_sum_y : (∑ i, y i) = 13)
  (h_sum_x_squared : (∑ i, (x i) ^ 2) = 21) : 
  b = 19 / 21 :=
by
  sorry

end scatter_plot_b_value_l148_148164


namespace probability_heads_at_most_3_of_10_coins_flipped_l148_148379

theorem probability_heads_at_most_3_of_10_coins_flipped :
  (∑ k in finset.range 4, nat.choose 10 k) / 2^10 = 11 / 64 :=
sorry

end probability_heads_at_most_3_of_10_coins_flipped_l148_148379


namespace probability_nickel_l148_148926

-- Define the values and worths
def V_d : ℝ := 5.00
def w_d : ℝ := 0.10
def V_n : ℝ := 2.50
def w_n : ℝ := 0.05
def V_p : ℝ := 1.00
def w_p : ℝ := 0.01

-- Define the number of coins based on the given values and worths
def num_dimes : ℝ := V_d / w_d
def num_nickels : ℝ := V_n / w_n
def num_pennies : ℝ := V_p / w_p

-- Define the total number of coins
def total_coins : ℝ := num_dimes + num_nickels + num_pennies

-- Define the probability of selecting a nickel
def P_nickel : ℝ := num_nickels / total_coins

-- The statement we need to prove
theorem probability_nickel : P_nickel = 1 / 4 := by
  sorry

end probability_nickel_l148_148926


namespace geometric_sequence_relation_l148_148273

variables {a : ℕ → ℝ} {q : ℝ}
variables {m n p : ℕ}

-- Conditions
def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = q * a n

def are_in_geometric_sequence (a : ℕ → ℝ) (m n p : ℕ) : Prop :=
  a n ^ 2 = a m * a p

-- Theorem
theorem geometric_sequence_relation (h_geom : is_geometric_sequence a q) (h_order : are_in_geometric_sequence a m n p) (hq_ne_one : q ≠ 1) :
  2 * n = m + p :=
sorry

end geometric_sequence_relation_l148_148273


namespace odd_and_even_inter_empty_l148_148347

-- Define the set of odd numbers
def odd_numbers : Set ℤ := {x | ∃ k : ℤ, x = 2 * k + 1}

-- Define the set of even numbers
def even_numbers : Set ℤ := {x | ∃ k : ℤ, x = 2 * k}

-- The theorem stating that the intersection of odd numbers and even numbers is empty
theorem odd_and_even_inter_empty : odd_numbers ∩ even_numbers = ∅ :=
by
  -- placeholder for the proof
  sorry

end odd_and_even_inter_empty_l148_148347


namespace vans_capacity_l148_148128

def students : ℕ := 33
def adults : ℕ := 9
def vans : ℕ := 6

def total_people : ℕ := students + adults
def people_per_van : ℕ := total_people / vans

theorem vans_capacity : people_per_van = 7 := by
  sorry

end vans_capacity_l148_148128


namespace count_chocolate_bars_l148_148224

theorem count_chocolate_bars 
    (chewing_gums : ℕ) 
    (candies : ℕ) 
    (total_treats : ℕ) 
    (chocolate_bars : ℕ)
    (h_chewing_gums : chewing_gums = 60)
    (h_candies : candies = 40)
    (h_total_treats : total_treats = 155) :
  chocolate_bars = total_treats - (chewing_gums + candies) :=
by
  -- Injecting given conditions
  rw [h_chewing_gums, h_candies, h_total_treats]
  -- Perform arithmetic calculation
  show 155 - (60 + 40) = 55
  sorry

end count_chocolate_bars_l148_148224


namespace fourth_term_of_geometric_sequence_is_320_l148_148941

noncomputable def geometric_sequence_first_term : ℕ := 5
noncomputable def geometric_sequence_fifth_term : ℕ := 1280
noncomputable def common_ratio := (geometric_sequence_fifth_term / geometric_sequence_first_term)^(1/4 : ℝ)

-- Now, we state the theorem to be proved
theorem fourth_term_of_geometric_sequence_is_320 :
  let r := (geometric_sequence_fifth_term / geometric_sequence_first_term)^((1:ℝ)/4)
  5 * r^3 = 320 :=
by
  sorry

end fourth_term_of_geometric_sequence_is_320_l148_148941


namespace distance_between_midpoints_l148_148822

-- Define the problem constants and conditions
variables {a b c d m n : ℝ}
def M := (m, n)
def A := (a, b)
def B := (c, d)
def M' := (m - 3.5, n + 2.5)
def distance (p q : ℝ × ℝ) := sqrt ((q.1 - p.1)^2 + (q.2 - p.2)^2)

-- The initial midpoint condition
axiom mid_eq_init : ∀ (a b c d : ℝ), (m, n) = ( (a + c) / 2, (b + d) / 2 )

-- The new midpoint calculation, after movements
axiom mid_eq_new : ∀ (a b c d : ℝ), (m - 3.5, n + 2.5) = ( (a + 5 + c - 12) / 2, (b + 10 + d - 5) / 2)

-- Prove the distance between the original midpoint M and new midpoint M' equals sqrt(18.5)
theorem distance_between_midpoints : distance M M' = sqrt 18.5 := sorry

end distance_between_midpoints_l148_148822


namespace hexagon_area_equilateral_triangle_l148_148763

theorem hexagon_area_equilateral_triangle :
  let side_length := 10,
      XA := 4,
      XB := 6,
      YC := 6,
      YD := 4,
      ZE := 4,
      ZF := 6 in
  let area_XYZ := (side_length^2 * Real.sqrt 3) / 4,
      area_XFA := (1 / 2) * XA * (XA * Real.sqrt 3),
      area_YCB := (1 / 2) * (side_length - XA) * XA * (Real.sqrt 3 / 2),
      area_ZED := (1 / 2) * XA * (XA * Real.sqrt 3 / 2),
      area_ABCDEF := area_XYZ - area_XFA - area_YCB - area_ZED in
  area_ABCDEF = Real.sqrt 768 :=
sorry

end hexagon_area_equilateral_triangle_l148_148763


namespace first_term_of_series_l148_148604

noncomputable def geometric_series_sum (a : ℝ) (r : ℝ) : ℝ :=
  a / (1 - r)

theorem first_term_of_series (a : ℝ) (r : ℝ) (sum : ℝ) (h_r : r = -1/3) (h_sum : sum = 9) :
  a = 12 :=
by
  have h : geometric_series_sum a r = sum := by sorry
  rw [h_r] at h
  simp at h
  rw [← h_sum] at h
  sorry

end first_term_of_series_l148_148604


namespace correct_number_can_be_expressed_l148_148905

-- Definition of the property that a number can be expressed as the sum of 150 consecutive positive integers.
def isSumOf150ConsecutiveIntegers (n : ℕ) : Prop :=
  ∃ a : ℕ, n = 150 * a + 11175

-- List of the given numbers.
def givenNumbers : List ℕ :=
  [3410775, 2245600, 1257925, 1725225, 4146950]

-- Specify the correct answer from the given conditions.
def correctAnswer := 1725225

-- The main theorem stating that the correct answer is the sum of 150 consecutive positive integers.
theorem correct_number_can_be_expressed : isSumOf150ConsecutiveIntegers correctAnswer :=
begin
  sorry
end

end correct_number_can_be_expressed_l148_148905


namespace find_CF_l148_148173

open_locale classical

variables (A B C D E F : Point)
variables (BD : Line) (perp_to_BD : ∀ (X : Point), X ∈ BD → Line.perpendicular BD (line_through A X))
variables [line A E BD] [line C F BD]
variables (BE AE DF : ℝ)
variables [BE = 4] [AE = 6] [DF = 9]

theorem find_CF (h1 : ∠ ABC = 90 ) 
                (h2 : ∠ BCD = 90 )
                (h3 : E ∈ BD)
                (h4 : F ∈ BD)
                (h5 : AE ⟂ BD)
                (h6 : CF ⟂ BD)
                (h7 : BE = 4)
                (h8 : AE = 6)
                (h9 : DF = 9) : CF = 13.5 :=
by
  sorry

end find_CF_l148_148173


namespace probability_at_most_three_heads_10_coins_l148_148357

open Nat

noncomputable def probability_at_most_three_heads (n : ℕ) (k : ℕ) : ℚ :=
  (∑ i in Finset.range (k + 1), (Nat.choose n i : ℚ)) / (2 ^ n : ℚ)

theorem probability_at_most_three_heads_10_coins :
  probability_at_most_three_heads 10 3 = 11 / 64 :=
by
  sorry

end probability_at_most_three_heads_10_coins_l148_148357


namespace exists_n_consecutive_composite_l148_148827

theorem exists_n_consecutive_composite (n : ℕ) :
  ∃ (seq : Fin (n+1) → ℕ), (∀ i, seq i > 1 ∧ ¬ Nat.Prime (seq i)) ∧
                          ∀ k, seq k = (n+1)! + (2 + k) := 
by
  sorry

end exists_n_consecutive_composite_l148_148827


namespace find_XY_sum_in_base10_l148_148654

def base8_addition_step1 (X : ℕ) : Prop :=
  X + 5 = 9

def base8_addition_step2 (Y X : ℕ) : Prop :=
  Y + 3 = X

theorem find_XY_sum_in_base10 (X Y : ℕ) (h1 : base8_addition_step1 X) (h2 : base8_addition_step2 Y X) :
  X + Y = 5 :=
by
  sorry

end find_XY_sum_in_base10_l148_148654


namespace prove_m_eq_n_l148_148799

variable (m n : ℕ)

noncomputable def p := m + n + 1

theorem prove_m_eq_n 
  (is_prime : Prime p) 
  (divides : p ∣ 2 * (m^2 + n^2) - 1) : 
  m = n :=
by
  sorry

end prove_m_eq_n_l148_148799


namespace length_of_platform_l148_148591

variables (l a t : ℝ)

-- Define the length of the train, constant acceleration, and time
-- Conditions:
-- 1. Train length is l
-- 2. Constant acceleration is a
-- 3. Passes a pole in time t

-- Define the statement to prove
theorem length_of_platform (h1 : l = 0.5 * a * t^2) 
                           (h2 : 6 * t > 0):
  let P := 18 * l - l in
  P = 17 * l :=
by {
  sorry
}

end length_of_platform_l148_148591


namespace negation_of_diagonals_equal_l148_148324

def Rectangle : Type := sorry -- Let's assume there exists a type Rectangle
def diagonals_equal (r : Rectangle) : Prop := sorry -- Assume a function that checks if diagonals are equal

theorem negation_of_diagonals_equal :
  ¬(∀ r : Rectangle, diagonals_equal r) ↔ ∃ r : Rectangle, ¬diagonals_equal r :=
by
  sorry

end negation_of_diagonals_equal_l148_148324


namespace pattern_paths_are_18_l148_148346

def placement : list (list (option char)) := 
  [[none, none, none, none, none, none, none, none, some 'C', none, none, none, none, none],  
   [none, none, none, none, none, none, some 'C', some 'O', some 'C', none, none, none, none, none],
   [none, none, none, none, none, some 'C', some 'O', some 'N', some 'O', some 'C', none, none, none, none],
   [none, none, none, none, some 'C', some 'O', some 'N', some 'T', some 'N', some 'O', some 'C', none, none, none],
   [none, none, none, some 'C', some 'R', some 'A', some 'T', some 'T', some 'E', some 'R', some 'N', some 'O', some 'C'],
   [none, none, some 'P', some 'A', some 'T', some 'T', some 'E', some 'R', some 'N', some 'A', some 'T', some 'T', some 'E', some 'R'],
   [some 'P', some 'A', some 'T', some 'T', some 'E', some 'R', some 'N', some 'A', some 'T', some 'T', some 'E', some 'R', some 'N', some 'A']]

-- Define the movement rules: horizontal, vertical, diagonal
def is_adjacent (pos1 pos2 : (ℕ × ℕ)) : Prop :=
  (pos1.1 = pos2.1 ∧ (pos1.2 = pos2.2 + 1 ∨ pos1.2 + 1 = pos2.2)) ∨
  (pos1.2 = pos2.2 ∧ (pos1.1 = pos2.1 + 1 ∨ pos1.1 + 1 = pos2.1)) ∨
  (pos1.1 = pos2.1 + 1 ∨ pos1.1 + 1 = pos2.1) ∧ (pos1.2 = pos2.2 + 1 ∨ pos1.2 + 1 = pos2.2)

-- The word to be spelled
def word := "PATTERN".to_list

-- Function to list out all valid paths (skipping the implementation, just defining)
noncomputable def count_paths (grid : list (list (option char))) (word : list char) : ℕ :=
  sorry

-- Prove the number of valid paths
theorem pattern_paths_are_18 : count_paths placement word = 18 :=
by sorry

end pattern_paths_are_18_l148_148346


namespace number_and_sum_of_possible_values_of_f_5_l148_148209

noncomputable def f : ℝ → ℝ := sorry

lemma problem_conditions (x y : ℝ) : f (f x - y) = f x + f (f y - f (-x)) + 2 * x := sorry

theorem number_and_sum_of_possible_values_of_f_5 : 
  let n := 1 in let s := -10 in n * s = -10 :=
by
  sorry

end number_and_sum_of_possible_values_of_f_5_l148_148209


namespace prob_heads_at_most_3_out_of_10_flips_l148_148462

def coin_flips := 10
def max_heads := 3
def total_outcomes := 2^coin_flips
def favorable_outcomes := (nat.choose coin_flips 0) +
                          (nat.choose coin_flips 1) +
                          (nat.choose coin_flips 2) +
                          (nat.choose coin_flips 3)

theorem prob_heads_at_most_3_out_of_10_flips :
  (favorable_outcomes : ℚ) / total_outcomes = 11 / 64 :=
by
  sorry

end prob_heads_at_most_3_out_of_10_flips_l148_148462


namespace find_exponential_function_l148_148756

theorem find_exponential_function (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (h3 : a^(-1) = 2) :
  ∀ x : ℝ, (a^x = (1/2)^x) :=
by {
  sorry
}

end find_exponential_function_l148_148756


namespace relationship_among_f_values_l148_148089

variable (f : ℝ → ℝ)
variable (h_even : ∀ x : ℝ, f x = f (-x))
variable (h_decreasing : ∀ x₁ x₂ : ℝ, 0 ≤ x₁ → 0 ≤ x₂ → x₁ ≠ x₂ → (x₁ - x₂) * (f x₁ - f x₂) < 0)

theorem relationship_among_f_values (h₀ : 0 < 2) (h₁ : 2 < 3) :
  f 0 > f (-2) ∧ f (-2) > f 3 :=
by
  sorry

end relationship_among_f_values_l148_148089


namespace probability_at_most_3_heads_l148_148440

def binom (n k : ℕ) : ℕ := Nat.choose n k

theorem probability_at_most_3_heads :
  let total_outcomes := 2^10
  let successful_outcomes := binom 10 0 + binom 10 1 + binom 10 2 + binom 10 3
  (successful_outcomes : ℚ) / total_outcomes = 11 / 64 := 
by 
  let total_outcomes := 2 ^ 10
  let successful_outcomes := binom 10 0 + binom 10 1 + binom 10 2 + binom 10 3
  have h : (successful_outcomes : ℚ) / total_outcomes = 11 / 64 := sorry
  exact h

end probability_at_most_3_heads_l148_148440


namespace solve_for_x_l148_148841

theorem solve_for_x (x : ℝ) (h : 3^(32^x) = 32^(3^x)) : 
  x = real.log 3 5 / 4 :=
by
  sorry

end solve_for_x_l148_148841


namespace largest_constant_C_l148_148053

theorem largest_constant_C :
  ∃ C, C = 2 / Real.sqrt 3 ∧ ∀ (x y z : ℝ), x^2 + y^2 + z^2 + 1 ≥ C * (x + y + z) := sorry

end largest_constant_C_l148_148053


namespace luke_piles_coins_l148_148814

theorem luke_piles_coins (x : ℕ) (h_total_piles : 10 = 5 + 5) (h_total_coins : 10 * x = 30) :
  x = 3 :=
by
  sorry

end luke_piles_coins_l148_148814


namespace fraction_eggs_given_to_Sofia_l148_148969

variables (m : ℕ) -- Number of eggs Mia has
def Sofia_eggs := 3 * m
def Pablo_eggs := 4 * Sofia_eggs
def Lucas_eggs := 0

theorem fraction_eggs_given_to_Sofia (h1 : Pablo_eggs = 12 * m) :
  (1 : ℚ) / (12 : ℚ) = 1 / 12 := by sorry

end fraction_eggs_given_to_Sofia_l148_148969


namespace probability_fully_lit_l148_148847

-- define the conditions of the problem
def characters : List String := ["K", "y", "o", "t", "o", " ", "G", "r", "a", "n", "d", " ", "H", "o", "t", "e", "l"]

-- define the length of the sequence
def length_sequence : ℕ := characters.length

-- theorem stating the probability of seeing the fully lit sign
theorem probability_fully_lit : (1 / length_sequence) = 1 / 5 :=
by
  -- The proof is omitted
  sorry

end probability_fully_lit_l148_148847


namespace domain_of_f_range_of_f_l148_148109

-- Define the function f(x)
def f (a x : ℝ) : ℝ := Real.sqrt ((1 - a^2) * x^2 + 3 * (1 - a) * x + 6)

-- Define a predicate for the domain of f being ℝ
def domain_is_ℝ (a : ℝ) : Prop :=
  (a = 1) ∨ 
  (a = -1 → false) ∧
  (-1 < a ∧ a < 1 ∧ (a - 1) * (11 * a + 5) ≤ 0)

-- Define a predicate for the range of f being [0, +∞)
def range_is_nonnnegative (a : ℝ) : Prop :=
  (-1 < a ∧ a < 1 ∧ (a - 1) * (11 * a + 5) ≥ 0) ∨ 
  (a = -1)

-- Theorem stating the range of a when domain of f is ℝ
theorem domain_of_f (a : ℝ) : domain_is_ℝ a ↔ - (5 / 11) ≤ a ∧ a ≤ 1 :=
by sorry

-- Theorem stating the range of a when range of f is [0, +∞)
theorem range_of_f (a : ℝ) : range_is_nonnnegative a ↔ -1 ≤ a ∧ a ≤ -(5 / 11) :=
by sorry

end domain_of_f_range_of_f_l148_148109


namespace Billy_weight_is_159_l148_148616

def Carl_weight : ℕ := 145
def Brad_weight : ℕ := Carl_weight + 5
def Billy_weight : ℕ := Brad_weight + 9

theorem Billy_weight_is_159 : Billy_weight = 159 := by
  sorry

end Billy_weight_is_159_l148_148616


namespace circle_center_eq_l148_148933

theorem circle_center_eq :
  ∃ (a b : ℝ), ∀ (x y : ℝ),
    (0, 1) ∈ (a, b) → (2, 4) ∈ (a, b) ∧ tangent (y = x^2) (2, 4) ∧
    let center := (a, b) in
    center = (-16/5, 53/10) :=
begin
  sorry
end

end circle_center_eq_l148_148933


namespace angle_PQR_is_90_l148_148770

theorem angle_PQR_is_90 
  (R S P Q : Type) 
  (line_RSP : ∀ x, x = R ∨ x = S ∨ x = P)
  (angle_QSP : ∠QSP = 70)
  (isosceles_PSQ : PQ = SQ) :
  ∠PQR = 90 :=
sorry

end angle_PQR_is_90_l148_148770


namespace probability_at_most_3_heads_l148_148507

-- Definitions and conditions
def coin_flips : ℕ := 10
def success_prob : ℚ := 1/2
def max_heads : ℕ := 3

-- Binomial Probability Computation
noncomputable def probability_of_at_most_3_heads : ℚ :=
  (∑ k in finset.range (max_heads + 1), nat.choose coin_flips k) / (2^coin_flips)

-- The statement to prove
theorem probability_at_most_3_heads :
  probability_of_at_most_3_heads = 11/64 := by
  sorry

end probability_at_most_3_heads_l148_148507


namespace probability_heads_at_most_3_l148_148534

theorem probability_heads_at_most_3 (num_coins : ℕ) (num_heads_at_most : ℕ) (probability : ℚ):
  num_coins = 10 → 
  num_heads_at_most = 3 → 
  probability = (binomial 10 0 + binomial 10 1 + binomial 10 2 + binomial 10 3) / (2^10) →
  probability = (11 / 64) := 
by
  intros hnum_coins hnum_heads_at_most hprobability
  sorry

end probability_heads_at_most_3_l148_148534


namespace similarity_PQR_ABC_l148_148122

-- Definitions to represent the triangles and their properties
universe u
variables {α : Type u} [MetricSpace α]

structure Triangle (α : Type u) :=
(vert1 vert2 vert3 : α)

variables (ABC KMN PQR : Triangle α)

-- Function to represent the median of a triangle
def median (T : Triangle α) : α → α := sorry

-- Given conditions
variables (AA1 BB1 CC1 : α)
variables (KK1 MM1 NN1 : α)

-- Defining the triangles through their medians
axiom medians_ABC : median ABC vert1 = AA1 ∧ median ABC vert2 = BB1 ∧ median ABC vert3 = CC1
axiom triangle_KMN : KMN.vert1 = AA1 ∧ KMN.vert2 = BB1 ∧ KMN.vert3 = CC1

axiom medians_KMN : median KMN vert1 = KK1 ∧ median KMN vert2 = MM1 ∧ median KMN vert3 = NN1
axiom triangle_PQR : PQR.vert1 = KK1 ∧ PQR.vert2 = MM1 ∧ PQR.vert3 = NN1

-- The theorem to prove similarity and the similarity ratio
theorem similarity_PQR_ABC : (∃ k : ℝ, k = 3/4 ∧ similar PQR ABC k) :=
by
  sorry

end similarity_PQR_ABC_l148_148122


namespace circle_ratio_l148_148920

theorem circle_ratio :
  ∀ (O1 O2 : Type) [Circle O1] [Circle O2] 
    (B C A E F H G D : Point)
    (BC_diameter_O1 : Diameter B C O1)
    (tangent_C_O1_A : Tangent C O1 A O2)
    (AB_intersect_O1_E : IntersectLineCircle A B O1 E)
    (CE_extend_intersect_O2_F : ExtendLineIntersectCircle C E O2 F)
    (H_on_AF : OnSegment H A F)
    (HE_extend_intersect_O1_G : ExtendLineIntersectCircle H E O1 G)
    (BG_extend_meet_AC_D : ExtendLineMeet B G (Extension A C) D),
  (AH_ratio_HF_AC_ratio_CD : ratio (Segment A H) (Segment H F)
                                 (Segment A C) (Segment C D)) :=
begin
  sorry
end

end circle_ratio_l148_148920


namespace slope_of_line_determined_by_solutions_l148_148303

theorem slope_of_line_determined_by_solutions :
  (∀ (x₁ y₁ x₂ y₂ : ℝ), (4 / x₁ + 6 / y₁ = 0) ∧ (4 / x₂ + 6 / y₂ = 0) →
    (y₂ - y₁) / (x₂ - x₁) = -3 / 2) :=
sorry

end slope_of_line_determined_by_solutions_l148_148303


namespace sample_variance_proof_l148_148272

variable a1 : ℤ
variable a2 : ℤ
variable a3 : ℤ
variable a4 : ℤ
variable a5 : ℤ

-- Hypotheses
def mean_eq_one (x1 x2 x3 x4 x5 : ℤ) : Prop := (x1 + x2 + x3 + x4 + x5) / 5 = 1
def sample_variance_eq (x1 x2 x3 x4 x5 : ℤ) (mean : ℤ) : ℚ := 
  (1 / 5 : ℚ) * ((x1 - mean)^2 + (x2 - mean)^2 + (x3 - mean)^2 + (x4 - mean)^2 + (x5 - mean)^2)

-- Theorem
theorem sample_variance_proof :
  (mean_eq_one a1 a2 a3 a4 a5) → (a1 = 0) → (a2 = 1) → (a3 = 2) → (a4 = 3) → (a5 = -1) → 
  sample_variance_eq a1 a2 a3 a4 a5 1 = 2 := 
by
  sorry

end sample_variance_proof_l148_148272


namespace probability_at_most_3_heads_10_flips_l148_148516

theorem probability_at_most_3_heads_10_flips : 
  let num_heads_0 := nat.choose 10 0,
      num_heads_1 := nat.choose 10 1,
      num_heads_2 := nat.choose 10 2,
      num_heads_3 := nat.choose 10 3,
      total_outcomes := 2^10 in
  (num_heads_0 + num_heads_1 + num_heads_2 + num_heads_3) / total_outcomes = 11 / 64 :=
by
  sorry

end probability_at_most_3_heads_10_flips_l148_148516


namespace area_of_enclosed_region_l148_148666

noncomputable def enclosed_area : ℝ :=
  let d1 := 180 - 90
  let d2 := 40 - (-40)
  (1/2) * d1 * d2

theorem area_of_enclosed_region :
  (∑ (x y : ℝ), if |x - 120| + |y| = |x / 3| then 1 else 0) = 3600 :=
begin
  sorry
end

end area_of_enclosed_region_l148_148666


namespace period_of_f_g_is_translated_f_g_is_odd_l148_148726

-- Definition of the function f(x)
def f (x : ℝ) : ℝ := 2 * sin (x - π / 3)

-- (1) Prove that the period of f(x) is 2π
theorem period_of_f : ∃ T > 0, ∀ x, f(x + T) = f(x) := 
  ∃ (2 * π), sorry

-- Definition of the function g(x) obtained by translating f(x) to the left by π/3 units
def g (x : ℝ) : ℝ := 2 * sin x

-- Prove that g(x) is 2 * sin x
theorem g_is_translated_f : ∀ x, g(x) = 2 * sin x := 
  by sorry

-- (2) Prove that g(x) is an odd function
theorem g_is_odd : ∀ x, g(-x) = -g(x) := 
  by sorry

end period_of_f_g_is_translated_f_g_is_odd_l148_148726


namespace john_not_stronger_l148_148167

-- Define the alcohol content percentages
def beer_pct : ℝ := 0.05
def liqueur_pct : ℝ := 0.1
def vodka_pct : ℝ := 0.4
def whiskey_pct : ℝ := 0.5

-- Define the mixture amounts for Ivan
def ivan_vodka : ℝ := 400
def ivan_beer : ℝ := 100

-- Define the mixture amounts for John
def john_liqueur : ℝ := 400
def john_whiskey : ℝ := 100

-- Calculate the total amount of alcohol in Ivan's cocktail
def ivan_total_alcohol : ℝ := (ivan_vodka * vodka_pct) + (ivan_beer * beer_pct)

-- Calculate the total amount of alcohol in John’s cocktail
def john_total_alcohol : ℝ := (john_liqueur * liqueur_pct) + (john_whiskey * whiskey_pct)

-- The proof statement
theorem john_not_stronger : ivan_total_alcohol > john_total_alcohol := by
  -- Using the previously defined values
  have ivan_total : ivan_total_alcohol = 165 := by
    simp [ivan_total_alcohol, ivan_vodka, vodka_pct, ivan_beer, beer_pct]
  have john_total : john_total_alcohol = 90 := by
    simp [john_total_alcohol, john_liqueur, liqueur_pct, john_whiskey, whiskey_pct]
  rw [ivan_total, john_total]
  linarith

end john_not_stronger_l148_148167


namespace largest_number_people_l148_148942

theorem largest_number_people (n : ℕ) (h : n = 60) : ∃ k, k = 40 ∧
  ∀ arrangement : list ℕ, 
    arrangement.length = n → 
    (∀ i, arrangement.nth i = some 1 → 
      (arrangement.nth ((i + 1) % n) = some 1 ↔ arrangement.nth ((i - 1 + n) % n) = some 0)) :=
by
  sorry

end largest_number_people_l148_148942


namespace probability_of_at_most_3_heads_l148_148400

-- Definitions and conditions
def num_coins : ℕ := 10
def at_most_3_heads_probability : ℚ := 11 / 64

-- Statement of the problem
theorem probability_of_at_most_3_heads (n : ℕ) (p : ℚ) (h1 : n = num_coins) (h2 : p = at_most_3_heads_probability) :
  p = (1 + 10 + 45 + 120 : ℕ) / (2 ^ 10 : ℕ) := by
  sorry

end probability_of_at_most_3_heads_l148_148400


namespace solve_length_of_AB_l148_148772

noncomputable def length_of_AB 
  (isosceles_ABC : ∀ (A B C : ℝ), (A = C) → (A ≠ B ∧ B ≠ C)) 
  (isosceles_CBD : ∀ (C B D : ℝ), (C = D) → (C ≠ B ∧ B ≠ D)) 
  (perimeter_CBD : ∀ (C B D : ℝ), B + C + D = 25)
  (perimeter_ABC : ∀ (A B C : ℝ), A + B + C = 26)
  (BD_length : ∀ (B D : ℝ), B + D = 9) 
  : ℝ :=
let B := 8 in let C := 8 in let A := 10 in A

theorem solve_length_of_AB 
  (isosceles_ABC : ∀ (A B C : ℝ), (A = C) → (A ≠ B ∧ B ≠ C)) 
  (isosceles_CBD : ∀ (C B D : ℝ), (C = D) → (C ≠ B ∧ B ≠ D)) 
  (perimeter_CBD : ∀ (C B D : ℝ), B + C + D = 25)
  (perimeter_ABC : ∀ (A B C : ℝ), A + B + C = 26)
  (BD_length : ∀ (B D : ℝ), B + D = 9) 
  : length_of_AB isosceles_ABC isosceles_CBD perimeter_CBD perimeter_ABC BD_length = 10 :=
by
  sorry

end solve_length_of_AB_l148_148772


namespace sin_convex_intervals_cos_convex_intervals_l148_148900

noncomputable def is_convex_on (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
∀ (x y ∈ s) (θ ∈ Icc (0:ℝ) 1), f (θ*x + (1-θ)*y) ≤ θ*f x + (1-θ)*y

noncomputable def is_concave_on (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
∀ (x y ∈ s) (θ ∈ Icc (0:ℝ) 1), f (θ*x + (1-θ)*y) ≥ θ*f x + (1-θ)*y

theorem sin_convex_intervals : 
  (∀ k : ℤ, is_convex_on sin (set.Icc ((2*k+1):ℝ)*real.pi ((2*k+2):ℝ)*real.pi)) ∧
  (∀ k : ℤ, is_concave_on sin (set.Icc ((2*k):ℝ)*real.pi ((2*k+1):ℝ)*real.pi)) := sorry

theorem cos_convex_intervals : 
  (∀ k : ℤ, is_convex_on cos (set.Icc ((4*k+1):ℝ)*real.pi/2 ((4*k+3):ℝ)*real.pi/2)) ∧
  (∀ k : ℤ, is_concave_on cos (set.Icc ((4*k-1):ℝ)*real.pi /2 ((4*k+1):ℝ)*real.pi/2)) := sorry

end sin_convex_intervals_cos_convex_intervals_l148_148900


namespace language_spoken_by_at_least_200_people_l148_148879

theorem language_spoken_by_at_least_200_people :
  ∀ (n : ℕ) (s : ℕ → set ℕ),
    n = 1985 →
    (∀ i, |(s i)| ≤ 5) →
    (∀ i j k, i ≠ j → j ≠ k → i ≠ k → ∃ l, l ∈ (s i) ∧ l ∈ (s j) ∧ l ∈ (s k)) →
    ∃ l, ∃ cnt, cnt ≥ 200 ∧ ∃ i, l ∈ (s i) ∧ cnt = (Finset.card (Finset.filter (λ i, l ∈ (s i)) (Finset.range n))) :=
by
  sorry

end language_spoken_by_at_least_200_people_l148_148879


namespace train_length_l148_148288

theorem train_length (L : ℝ) 
  (equal_length : ∀ (A B : ℝ), A = B → L = A)
  (same_direction : ∀ (dir1 dir2 : ℤ), dir1 = 1 → dir2 = 1)
  (speed_faster : ℝ := 50) (speed_slower : ℝ := 36)
  (time_to_pass : ℝ := 36)
  (relative_speed := speed_faster - speed_slower)
  (relative_speed_km_per_sec := relative_speed / 3600)
  (distance_covered := relative_speed_km_per_sec * time_to_pass)
  (total_distance := distance_covered)
  (length_per_train := total_distance / 2)
  (length_in_meters := length_per_train * 1000): 
  L = 70 := 
by 
  sorry

end train_length_l148_148288


namespace purely_imaginary_complex_iff_l148_148719

theorem purely_imaginary_complex_iff (m : ℝ) :
  (m + 2 = 0) → (m = -2) :=
by
  sorry

end purely_imaginary_complex_iff_l148_148719


namespace probability_of_at_most_3_heads_l148_148478

-- Definitions based on conditions
def binom : ℕ → ℕ → ℕ
| 0, 0       => 1
| 0, (k + 1) => 0
| (n + 1), 0 => 1
| (n + 1), (k + 1) => binom n k + binom n (k + 1)

def favorable_outcomes : ℕ :=
binom 10 0 + binom 10 1 + binom 10 2 + binom 10 3

def total_outcomes : ℕ := 2 ^ 10

def probability (favorable : ℕ) (total : ℕ) : ℚ :=
favorable / total

-- Theorem statement
theorem probability_of_at_most_3_heads : 
  probability favorable_outcomes total_outcomes = 11 / 64 :=
sorry

end probability_of_at_most_3_heads_l148_148478


namespace probability_at_most_three_heads_10_coins_l148_148358

open Nat

noncomputable def probability_at_most_three_heads (n : ℕ) (k : ℕ) : ℚ :=
  (∑ i in Finset.range (k + 1), (Nat.choose n i : ℚ)) / (2 ^ n : ℚ)

theorem probability_at_most_three_heads_10_coins :
  probability_at_most_three_heads 10 3 = 11 / 64 :=
by
  sorry

end probability_at_most_three_heads_10_coins_l148_148358


namespace range_of_a_l148_148992

noncomputable def f (x : ℝ) : ℝ := 
  if x ∈ -1..1 then x^2 - 2*x
  else if (x - 4 * int.floor (x / 4)) ∈ -1..1 then (x - 4 * int.floor (x / 4))^2 - 2*(x - 4 * int.floor (x / 4))
  else 0

def I_k (k : ℤ) : set ℝ :=
  {x : ℝ | 4*k-1 ≤ x ∧ x ≤ 4*k+3}

def g (a : ℝ) (x : ℝ) : ℝ := log a x

theorem range_of_a (a : ℝ) (k : ℤ) (hk : 1 ≤ k) :
  (∃ x y ∈ I_k k, x ≠ y ∧ f x = g a x ∧ f y = g a y) ↔ 
  (a ∈ set.Ici (sqrt (4 * k + 3)) ∨ a ∈ set.Ico 0 (1 / (4 * k + 1))) :=
sorry

end range_of_a_l148_148992


namespace inscribed_squares_ratio_l148_148590

theorem inscribed_squares_ratio (a b c x y : ℝ) (h_tri : a^2 + b^2 = c^2)
  (h_square1 : ∃ x, ∃ y, (a - x) * (b - x) / (1 + x * (a + b - x)) = 1)
  (h_square2 : ∃ x, ∃ y, (x * (a + b - x)) / (1 + x * (c - x)) = 1) :
  x / y = 144 / 221 := 
  sorry

end inscribed_squares_ratio_l148_148590


namespace infinite_common_elements_l148_148028

noncomputable def a_seq : ℕ → ℤ
| 0     := 2
| 1     := 14
| (n+2) := 14 * a_seq (n + 1) + a_seq n

noncomputable def b_seq : ℕ → ℤ
| 0     := 2
| 1     := 14
| (n+2) := 6 * b_seq (n + 1) - b_seq n

theorem infinite_common_elements :
  ∃ᶠ n in at_top, a_seq n = b_seq n :=
sorry

end infinite_common_elements_l148_148028


namespace original_price_four_pack_l148_148682

theorem original_price_four_pack (price_with_rush: ℝ) (increase_rate: ℝ) (num_packs: ℕ):
  price_with_rush = 13 → increase_rate = 0.30 → num_packs = 4 → num_packs * (price_with_rush / (1 + increase_rate)) = 40 :=
by
  intros h_price h_rate h_packs
  rw [h_price, h_rate, h_packs]
  sorry

end original_price_four_pack_l148_148682


namespace sin_beta_value_l148_148700

noncomputable theory

open Real

variables (α β : ℝ)
variables (h1 : 0 < α ∧ α < π / 2)
variables (h2 : 0 < β ∧ β < π / 2)
variables (h3 : cos α = 2 * sqrt 5 / 5)
variables (h4 : sin (α - β) = -3 / 5)

theorem sin_beta_value : sin β = 2 * sqrt 5 / 5 :=
sorry

end sin_beta_value_l148_148700


namespace preston_charges_5_dollars_l148_148825

def cost_per_sandwich (x : Real) : Prop :=
  let number_of_sandwiches := 18
  let delivery_fee := 20
  let tip_percentage := 0.10
  let total_received := 121
  let total_cost := number_of_sandwiches * x + delivery_fee
  let tip := tip_percentage * total_cost
  let final_amount := total_cost + tip
  final_amount = total_received

theorem preston_charges_5_dollars :
  ∀ x : Real, cost_per_sandwich x → x = 5 :=
by
  intros x h
  sorry

end preston_charges_5_dollars_l148_148825


namespace bronson_profit_l148_148621

theorem bronson_profit :
  let cost_per_bushel := 12
  let apples_per_bushel := 48
  let selling_price_per_apple := 0.40
  let apples_sold := 100
  let cost_per_apple := cost_per_bushel / apples_per_bushel
  let profit_per_apple := selling_price_per_apple - cost_per_apple
  let total_profit := profit_per_apple * apples_sold
  in
  total_profit = 15 := 
by 
  sorry

end bronson_profit_l148_148621


namespace solve_system_of_equations_l148_148241

theorem solve_system_of_equations :
  ∃ (x y z : ℝ), x + y + z = 11 ∧ x^2 + 2 * y^2 + 3 * z^2 = 66 ∧ x = 6 ∧ y = 3 ∧ z = 2 :=
by
  use 6, 3, 2
  simp
  done

end solve_system_of_equations_l148_148241


namespace distributions_ex_variances_l148_148155

noncomputable def EX (p : ℝ) : ℝ := p
noncomputable def DX (p : ℝ) : ℝ := p * (1 - p)
noncomputable def EY (n : ℕ) (p : ℝ) : ℝ := n * p
noncomputable def DY (n : ℕ) (p : ℝ) : ℝ := n * p * (1 - p)

theorem distributions_ex_variances :
  let X_bern := 0.7 in
  let Y_binom_n := 10 in
  let Y_binom_p := 0.8 in
  EX X_bern = 0.7 ∧ DX X_bern = 0.21 ∧ EY Y_binom_n Y_binom_p = 8 ∧ DY Y_binom_n Y_binom_p = 1.6 :=
by
  sorry

end distributions_ex_variances_l148_148155


namespace probability_at_most_three_heads_10_coins_l148_148354

open Nat

noncomputable def probability_at_most_three_heads (n : ℕ) (k : ℕ) : ℚ :=
  (∑ i in Finset.range (k + 1), (Nat.choose n i : ℚ)) / (2 ^ n : ℚ)

theorem probability_at_most_three_heads_10_coins :
  probability_at_most_three_heads 10 3 = 11 / 64 :=
by
  sorry

end probability_at_most_three_heads_10_coins_l148_148354


namespace range_of_x0_on_line_and_circle_l148_148072

def circle (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 1
def line (x y : ℝ) : Prop := x + y + 1 = 0

theorem range_of_x0_on_line_and_circle {x_0 : ℝ} (H : ∃ y, line x_0 y) :
  -1 ≤ x_0 ∧ x_0 ≤ 2 :=
sorry

end range_of_x0_on_line_and_circle_l148_148072


namespace circle_center_proof_l148_148931

open Classical

noncomputable def center_of_circle (circle : ℝ × ℝ → Prop) : ℝ × ℝ :=
  let center := (-16/5, 53/10)
  center

theorem circle_center_proof :
  ∃ center : ℝ × ℝ, 
    (circle center) ∧ 
    (center_of_circle circle = (-16/5, 53/10)) :=
by
  sorry

end circle_center_proof_l148_148931


namespace retailer_actual_profit_is_27_5_percent_l148_148965
open Real

variables (c : ℝ)

def marked_price (c : ℝ) := 1.5 * c
def discount (c : ℝ) := 0.15 * (marked_price c)
def selling_price (c : ℝ) := (marked_price c) - (discount c)
def actual_profit (c : ℝ) := (selling_price c) - c
def actual_profit_percentage (c : ℝ) := (actual_profit c / c) * 100

theorem retailer_actual_profit_is_27_5_percent:
  actual_profit_percentage c = 27.5 :=
by 
  sorry

end retailer_actual_profit_is_27_5_percent_l148_148965


namespace probability_of_at_most_3_heads_l148_148464

-- Definitions based on conditions
def binom : ℕ → ℕ → ℕ
| 0, 0       => 1
| 0, (k + 1) => 0
| (n + 1), 0 => 1
| (n + 1), (k + 1) => binom n k + binom n (k + 1)

def favorable_outcomes : ℕ :=
binom 10 0 + binom 10 1 + binom 10 2 + binom 10 3

def total_outcomes : ℕ := 2 ^ 10

def probability (favorable : ℕ) (total : ℕ) : ℚ :=
favorable / total

-- Theorem statement
theorem probability_of_at_most_3_heads : 
  probability favorable_outcomes total_outcomes = 11 / 64 :=
sorry

end probability_of_at_most_3_heads_l148_148464


namespace probability_10_coins_at_most_3_heads_l148_148391

def probability_at_most_3_heads (n : ℕ) : ℚ :=
  let total_outcomes := 2^n
  let favorable_outcomes := (Nat.choose n 0) + (Nat.choose n 1) + (Nat.choose n 2) + (Nat.choose n 3)
  favorable_outcomes / total_outcomes

theorem probability_10_coins_at_most_3_heads : probability_at_most_3_heads 10 = 11 / 64 :=
by
  sorry

end probability_10_coins_at_most_3_heads_l148_148391


namespace prob_heads_at_most_3_out_of_10_flips_l148_148450

def coin_flips := 10
def max_heads := 3
def total_outcomes := 2^coin_flips
def favorable_outcomes := (nat.choose coin_flips 0) +
                          (nat.choose coin_flips 1) +
                          (nat.choose coin_flips 2) +
                          (nat.choose coin_flips 3)

theorem prob_heads_at_most_3_out_of_10_flips :
  (favorable_outcomes : ℚ) / total_outcomes = 11 / 64 :=
by
  sorry

end prob_heads_at_most_3_out_of_10_flips_l148_148450


namespace marble_arrangement_remainder_l148_148651

-- Definitions based on conditions
def num_green : ℕ := 5
def arrangement_condition (arr : list char) : Prop :=
  let same_color_pairs := list.filter (λ (p : char × char), p.1 = p.2) (list.zip arr (list.tail arr)) in
  let diff_color_pairs := list.filter (λ (p : char × char), p.1 ≠ p.2) (list.zip arr (list.tail arr)) in
  2 * list.length diff_color_pairs = list.length same_color_pairs

def ways_to_arrange (m : ℕ) : ℕ :=
  let total_marbles := m + num_green in
  if arrangement_condition then nat.choose (m + num_green - 1) (num_green - 1) + 1 else 0

-- Theorem statement
theorem marble_arrangement_remainder (m : ℕ) :
  m = 15 → ((ways_to_arrange m) % 1000) = 3 :=
by
  sorry

end marble_arrangement_remainder_l148_148651


namespace Darcy_remaining_clothes_l148_148034

/--
Darcy initially has 20 shirts and 8 pairs of shorts.
He folds 12 of the shirts and 5 of the pairs of shorts.
We want to prove that the total number of remaining pieces of clothing Darcy has to fold is 11.
-/
theorem Darcy_remaining_clothes
  (initial_shirts : Nat)
  (initial_shorts : Nat)
  (folded_shirts : Nat)
  (folded_shorts : Nat)
  (remaining_shirts : Nat)
  (remaining_shorts : Nat)
  (total_remaining : Nat) :
  initial_shirts = 20 → initial_shorts = 8 →
  folded_shirts = 12 → folded_shorts = 5 →
  remaining_shirts = initial_shirts - folded_shirts →
  remaining_shorts = initial_shorts - folded_shorts →
  total_remaining = remaining_shirts + remaining_shorts →
  total_remaining = 11 := by
  sorry

end Darcy_remaining_clothes_l148_148034


namespace AD_eq_BC_l148_148252

open EuclideanGeometry

variables {A B C D E O : Point}
variables (h_cyclic : Cyclic A B C D)
variables (h_diag_intersect : Line A C ∩ Line B D = {E})
variables (h_circumcenter_O : Circumcenter A B C D = O)
variables (M N P : Point)
variables (h_mid_AB : midpoint A B = M)
variables (h_mid_CD : midpoint C D = N)
variables (h_mid_OE : midpoint O E = P)
variables (h_collinear : Collinear M N P)

theorem AD_eq_BC : distance A D = distance B C :=
sorry

end AD_eq_BC_l148_148252


namespace fourth_graders_bought_more_markers_l148_148967

-- Define the conditions
def cost_per_marker : ℕ := 20
def total_payment_fifth_graders : ℕ := 180
def total_payment_fourth_graders : ℕ := 200

-- Compute the number of markers bought by fifth and fourth graders
def markers_bought_by_fifth_graders : ℕ := total_payment_fifth_graders / cost_per_marker
def markers_bought_by_fourth_graders : ℕ := total_payment_fourth_graders / cost_per_marker

-- Statement to prove
theorem fourth_graders_bought_more_markers : 
  markers_bought_by_fourth_graders - markers_bought_by_fifth_graders = 1 := by
  sorry

end fourth_graders_bought_more_markers_l148_148967


namespace order_of_values_l148_148687

noncomputable def a : ℝ := (1 / 5) ^ 2
noncomputable def b : ℝ := 2 ^ (1 / 5)
noncomputable def c : ℝ := Real.log (1 / 5) / Real.log 2  -- change of base from log base 2 to natural log

theorem order_of_values : c < a ∧ a < b :=
by
  sorry

end order_of_values_l148_148687


namespace probability_at_most_3_heads_l148_148446

def binom (n k : ℕ) : ℕ := Nat.choose n k

theorem probability_at_most_3_heads :
  let total_outcomes := 2^10
  let successful_outcomes := binom 10 0 + binom 10 1 + binom 10 2 + binom 10 3
  (successful_outcomes : ℚ) / total_outcomes = 11 / 64 := 
by 
  let total_outcomes := 2 ^ 10
  let successful_outcomes := binom 10 0 + binom 10 1 + binom 10 2 + binom 10 3
  have h : (successful_outcomes : ℚ) / total_outcomes = 11 / 64 := sorry
  exact h

end probability_at_most_3_heads_l148_148446


namespace measure_angle_4_l148_148643

theorem measure_angle_4 (m1 m2 m3 m5 m6 m4 : ℝ) 
  (h1 : m1 = 82) 
  (h2 : m2 = 34) 
  (h3 : m3 = 19) 
  (h4 : m5 = m6 + 10) 
  (h5 : m1 + m2 + m3 + m5 + m6 = 180)
  (h6 : m4 + m5 + m6 = 180) : 
  m4 = 135 :=
by
  -- Placeholder for the full proof, omitted due to instructions
  sorry

end measure_angle_4_l148_148643


namespace angle_C_45_degrees_l148_148096

variable (R a b : ℝ)
variable (A B C : ℝ)
variable (sin : ℝ → ℝ)

axiom sin_squared_eq (x : ℝ) : sin x * sin x = sin x ^ 2

theorem angle_C_45_degrees 
  (h1 : ∀ (x y z : ℝ), x = 2 * R * (sin A ^ 2 - sin z ^ 2) → y = (Real.sqrt 2 * a - b) * sin B)
  (h2 : 2 * R * (sin A ^ 2 - sin C ^ 2) = (Real.sqrt 2 * a - b) * sin B) : 
  C = Real.arccos (√2 / 2) :=
sorry

end angle_C_45_degrees_l148_148096


namespace sum_consecutive_naturals_l148_148874

theorem sum_consecutive_naturals {
  (a : ℕ) (n : ℕ) :
    1 ≤ a ∧ a ≤ 9 ∧ (∃ n : ℕ, (n * (n + 1)) / 2 = 11 * a) 
  → (n = 10 ∨ n = 11) :=
by
  sorry

end sum_consecutive_naturals_l148_148874


namespace interest_rate_of_first_investment_l148_148817

-- Conditions
variables (inheritance total : ℝ) (r : ℝ)
variables (invested_at_r invested_at_8 : ℝ)
variables (total_interest yearly_interest : ℝ)

-- Initialization of the conditions
def inheritance_initial : Prop := inheritance = 12000
def invested_initial : Prop := invested_at_r = 5000
def remainder_initial : Prop := invested_at_8 = inheritance - invested_at_r
def yearly_interest_initial : Prop := yearly_interest = 860
def interest_calculation : Prop := total = (invested_at_r * r) + (invested_at_8 * 0.08)

-- Theorem to prove
theorem interest_rate_of_first_investment :
    inheritance_initial →
    invested_initial →
    remainder_initial →
    yearly_interest_initial →
    interest_calculation →
    r = 0.06 :=
by
  sorry

end interest_rate_of_first_investment_l148_148817


namespace probability_heads_at_most_3_l148_148531

theorem probability_heads_at_most_3 (num_coins : ℕ) (num_heads_at_most : ℕ) (probability : ℚ):
  num_coins = 10 → 
  num_heads_at_most = 3 → 
  probability = (binomial 10 0 + binomial 10 1 + binomial 10 2 + binomial 10 3) / (2^10) →
  probability = (11 / 64) := 
by
  intros hnum_coins hnum_heads_at_most hprobability
  sorry

end probability_heads_at_most_3_l148_148531


namespace Darcy_remaining_clothes_l148_148035

/--
Darcy initially has 20 shirts and 8 pairs of shorts.
He folds 12 of the shirts and 5 of the pairs of shorts.
We want to prove that the total number of remaining pieces of clothing Darcy has to fold is 11.
-/
theorem Darcy_remaining_clothes
  (initial_shirts : Nat)
  (initial_shorts : Nat)
  (folded_shirts : Nat)
  (folded_shorts : Nat)
  (remaining_shirts : Nat)
  (remaining_shorts : Nat)
  (total_remaining : Nat) :
  initial_shirts = 20 → initial_shorts = 8 →
  folded_shirts = 12 → folded_shorts = 5 →
  remaining_shirts = initial_shirts - folded_shirts →
  remaining_shorts = initial_shorts - folded_shorts →
  total_remaining = remaining_shirts + remaining_shorts →
  total_remaining = 11 := by
  sorry

end Darcy_remaining_clothes_l148_148035


namespace angle_m_n_is_135_degrees_l148_148738

open Real

def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2
def magnitude (v : ℝ × ℝ) : ℝ := sqrt (v.1 ^ 2 + v.2 ^ 2)
def cos_angle (v w : ℝ × ℝ) : ℝ := dot_product v w / (magnitude v * magnitude w)
def angle_radians (v w : ℝ × ℝ) : ℝ := arccos (cos_angle v w)
def angle_degrees (v w : ℝ × ℝ) : ℝ := (angle_radians v w) * (180 / π)

def vector_a : ℝ × ℝ := (3, 4)
def vector_b : ℝ × ℝ := (9, 12)
def vector_c : ℝ × ℝ := (4, -3)
def vector_m : ℝ × ℝ := (2 * 3 - 9, 2 * 4 - 12)
def vector_n : ℝ × ℝ := (3 + 4, 4 + (-3))

theorem angle_m_n_is_135_degrees :
  angle_degrees vector_m vector_n = 135 := by
  sorry

end angle_m_n_is_135_degrees_l148_148738


namespace verify_graphical_method_l148_148918

variable {R : Type} [LinearOrderedField R]

/-- Statement of the mentioned conditions -/
def poly (a b c d x : R) : R := a * x^3 + b * x^2 + c * x + d

/-- The main theorem stating the graphical method validity -/
theorem verify_graphical_method (a b c d x0 EJ : R) (h : 0 < a) (h1 : 0 < b) (h2 : 0 < c) (h3 : 0 < d) (h4 : 0 < x0) (h5 : x0 < 1)
: EJ = poly a b c d x0 := by sorry

end verify_graphical_method_l148_148918


namespace numAscendingNumbersBetween400And600_l148_148581

-- Definition of ascending number
def isAscending (n : ℕ) : Prop :=
  ∀ i j, i < j → (n / 10^i % 10) < (n / 10^j % 10)

-- Ascending numbers between 400 and 600
def ascendingNumbersBetween400And600 :=
  {n | 400 ≤ n ∧ n < 600 ∧ isAscending n}

theorem numAscendingNumbersBetween400And600 :
  (ascendingNumbersBetween400And600.toFinset.card) = 16 :=
by sorry

end numAscendingNumbersBetween400And600_l148_148581


namespace min_x_squared_plus_y_squared_l148_148912

theorem min_x_squared_plus_y_squared (x y : ℝ) (h : (x + 5) * (y - 5) = 0) :
  x^2 + y^2 ≥ 50 :=
by
  sorry

end min_x_squared_plus_y_squared_l148_148912


namespace range_of_f_l148_148674

noncomputable def f (x : ℝ) : ℝ :=
  (Real.arcsin (x / 3))^2 + π * Real.arccos (x / 3) + (π^2 / 4) * (x^2 + 4 * x + 3)

theorem range_of_f : 
  ∀ x : ℝ, x ∈ Icc (-3) 3 → 
    ∃ y ∈ Icc (5 * π^2 / 4) (53 * π^2 / 12), f x = y := 
sorry

end range_of_f_l148_148674


namespace probability_10_coins_at_most_3_heads_l148_148385

def probability_at_most_3_heads (n : ℕ) : ℚ :=
  let total_outcomes := 2^n
  let favorable_outcomes := (Nat.choose n 0) + (Nat.choose n 1) + (Nat.choose n 2) + (Nat.choose n 3)
  favorable_outcomes / total_outcomes

theorem probability_10_coins_at_most_3_heads : probability_at_most_3_heads 10 = 11 / 64 :=
by
  sorry

end probability_10_coins_at_most_3_heads_l148_148385


namespace total_age_of_wines_l148_148247

theorem total_age_of_wines (age_carlo_rosi : ℕ) (age_franzia : ℕ) (age_twin_valley : ℕ) 
    (h1 : age_carlo_rosi = 40) (h2 : age_franzia = 3 * age_carlo_rosi) (h3 : age_carlo_rosi = 4 * age_twin_valley) : 
    age_franzia + age_carlo_rosi + age_twin_valley = 170 := 
by
    sorry

end total_age_of_wines_l148_148247


namespace triangle_division_into_nine_equal_parts_l148_148229

theorem triangle_division_into_nine_equal_parts (A B C D E F G H I : Type) 
  (triangle_ABC : ∀ (d e f : A), d = A ∧ e = B ∧ f = C)
  (divide_segments : ∀ (x y : A) (p q : ℕ) (segment_len : ℝ), p = 3 * segment_len ∧ q = 3 * segment_len) :
  ∃ (parts : list (A × A × A)), parts.length = 9 ∧ ∀ (triangle ∈ parts), congruent (triangle, some_triangle) :=
sorry

end triangle_division_into_nine_equal_parts_l148_148229


namespace probability_heads_at_most_3_of_10_coins_flipped_l148_148382

theorem probability_heads_at_most_3_of_10_coins_flipped :
  (∑ k in finset.range 4, nat.choose 10 k) / 2^10 = 11 / 64 :=
sorry

end probability_heads_at_most_3_of_10_coins_flipped_l148_148382


namespace sum_of_base_7_digits_of_999_l148_148312

theorem sum_of_base_7_digits_of_999 : 
  let n := 999 
  let base := 7 
  let base7_rep := [2, 6, 2, 5]
  list.sum base7_rep = 15 :=
by
  let n := 999
  let base := 7
  let base7_rep := [2, 6, 2, 5]
  have h : list.sum base7_rep = 2 + 6 + 2 + 5 := sorry
  have sum_is_15 : 2 + 6 + 2 + 5 = 15 := sorry
  show list.sum base7_rep = 15, from h.trans sum_is_15

end sum_of_base_7_digits_of_999_l148_148312


namespace probability_at_most_3_heads_l148_148497

-- Definitions and conditions
def coin_flips : ℕ := 10
def success_prob : ℚ := 1/2
def max_heads : ℕ := 3

-- Binomial Probability Computation
noncomputable def probability_of_at_most_3_heads : ℚ :=
  (∑ k in finset.range (max_heads + 1), nat.choose coin_flips k) / (2^coin_flips)

-- The statement to prove
theorem probability_at_most_3_heads :
  probability_of_at_most_3_heads = 11/64 := by
  sorry

end probability_at_most_3_heads_l148_148497


namespace hyperbola_equation_l148_148253

theorem hyperbola_equation
  (passes_through : ∃ (x y : ℝ), (x = 2 ∧ y = 0))
  (asymptotes : ∀ (x y c : ℝ), (x ≠ 0 → y/x = ±1 → y = x ∨ y = -x)) :
  ∃ (k : ℝ), (k ≠ 0 ∧ ∀ x y : ℝ, x^2 - y^2 = k → x^2 / 4 - y^2 / 4 = 1) := 
sorry

end hyperbola_equation_l148_148253


namespace tangent_line_at_x1_f_nonnegative_iff_l148_148115

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := (x-1) * Real.log x - m * (x+1)

noncomputable def f' (x : ℝ) (m : ℝ) : ℝ := Real.log x + (x-1) / x - m

theorem tangent_line_at_x1 (m : ℝ) (h : m = 1) :
  ∀ x y : ℝ, f x 1 = y → (x = 1) → x + y + 1 = 0 :=
sorry

theorem f_nonnegative_iff (m : ℝ) :
  (∀ x : ℝ, 0 < x → f x m ≥ 0) ↔ m ≤ 0 :=
sorry

end tangent_line_at_x1_f_nonnegative_iff_l148_148115


namespace probability_same_district_l148_148161

section PublicRentalHousing

-- Define the districts
inductive District
| A
| B
| C
| D

-- Define the applicants
inductive Applicant
| Jia
| Yi
| Bing

open District

-- Define the problem
def total_application_scenarios : ℕ := 4 * 4

theorem probability_same_district :
  (∃ (d : District), (d 甲 = d 乙)) / total_application_scenarios = 1 / 4 :=
sorry

end PublicRentalHousing

end probability_same_district_l148_148161


namespace right_triangle_AB_is_approximately_8point3_l148_148168

noncomputable def tan_deg (θ : ℝ) : ℝ := Real.tan (θ * Real.pi / 180)

theorem right_triangle_AB_is_approximately_8point3 :
  ∀ (A B C : Type) (angle_A : ℝ) (angle_B : ℝ) (BC AB : ℝ),
  angle_A = 40 ∧ angle_B = 90 ∧ BC = 7 →
  AB = 7 / tan_deg 40 →
  abs (AB - 8.3) < 0.1 :=
by
  intros A B C angle_A angle_B BC AB h_cond h_AB
  sorry

end right_triangle_AB_is_approximately_8point3_l148_148168


namespace probability_of_at_most_3_heads_out_of_10_l148_148487
open_locale big_operators
open nat

theorem probability_of_at_most_3_heads_out_of_10 (n : ℕ) (p : ℕ) (h : 0 < n) (hp : p = 2) :
  ∑ k in finset.range (4), nat.choose n k = 176 → nat.pow p n = 1024 → (∑ k in finset.range (4), nat.choose n k) / (nat.pow p n) = 11 / 64 :=
by { intro h₁ h₂, 
     have : (∑ k in finset.range 4, nat.choose n k) / (nat.pow p n) = 176 / 1024,
     { rw [h₁, h₂] },
     rw this,
     norm_num }

end probability_of_at_most_3_heads_out_of_10_l148_148487


namespace number_of_eight_letter_good_words_l148_148044

def is_good_word (word : List Char) : Prop :=
  ∀ i, i < word.length - 1 →
    ((word[i] = 'A' → word[i+1] ≠ 'B') ∧
    (word[i] = 'B' → word[i+1] ≠ 'C') ∧
    (word[i] = 'C' → word[i+1] ≠ 'A') ∧
    (word[i] = 'D' → word[i+1] ≠ 'A'))

def num_good_words (n : Nat) : Nat :=
  if n = 0 then 1 else 4 * 2^(n-1)

theorem number_of_eight_letter_good_words : 
  num_good_words 8 = 512 :=
by
  simp [num_good_words]

end number_of_eight_letter_good_words_l148_148044


namespace z_leg_time_l148_148917

theorem z_leg_time (t : ℕ) (h1 : 58) (h2 : (58 + t) / 2 = 42) : t = 26 := 
by
  sorry

end z_leg_time_l148_148917


namespace arrange_plants_in_a_row_l148_148013

-- Definitions for the conditions
def basil_plants : ℕ := 5 -- Number of basil plants
def tomato_plants : ℕ := 4 -- Number of tomato plants

-- Theorem statement asserting the number of ways to arrange the plants
theorem arrange_plants_in_a_row : 
  let total_items := basil_plants + 1,
      ways_to_arrange_total_items := Nat.factorial total_items,
      ways_to_arrange_tomato_group := Nat.factorial tomato_plants in
  (ways_to_arrange_total_items * ways_to_arrange_tomato_group) = 17280 := 
by
  sorry

end arrange_plants_in_a_row_l148_148013


namespace probability_of_at_most_3_heads_out_of_10_l148_148486
open_locale big_operators
open nat

theorem probability_of_at_most_3_heads_out_of_10 (n : ℕ) (p : ℕ) (h : 0 < n) (hp : p = 2) :
  ∑ k in finset.range (4), nat.choose n k = 176 → nat.pow p n = 1024 → (∑ k in finset.range (4), nat.choose n k) / (nat.pow p n) = 11 / 64 :=
by { intro h₁ h₂, 
     have : (∑ k in finset.range 4, nat.choose n k) / (nat.pow p n) = 176 / 1024,
     { rw [h₁, h₂] },
     rw this,
     norm_num }

end probability_of_at_most_3_heads_out_of_10_l148_148486


namespace probability_at_most_3_heads_l148_148505

-- Definitions and conditions
def coin_flips : ℕ := 10
def success_prob : ℚ := 1/2
def max_heads : ℕ := 3

-- Binomial Probability Computation
noncomputable def probability_of_at_most_3_heads : ℚ :=
  (∑ k in finset.range (max_heads + 1), nat.choose coin_flips k) / (2^coin_flips)

-- The statement to prove
theorem probability_at_most_3_heads :
  probability_of_at_most_3_heads = 11/64 := by
  sorry

end probability_at_most_3_heads_l148_148505


namespace probability_at_most_3_heads_10_flips_l148_148520

theorem probability_at_most_3_heads_10_flips : 
  let num_heads_0 := nat.choose 10 0,
      num_heads_1 := nat.choose 10 1,
      num_heads_2 := nat.choose 10 2,
      num_heads_3 := nat.choose 10 3,
      total_outcomes := 2^10 in
  (num_heads_0 + num_heads_1 + num_heads_2 + num_heads_3) / total_outcomes = 11 / 64 :=
by
  sorry

end probability_at_most_3_heads_10_flips_l148_148520


namespace find_magnitude_of_vec_a_l148_148739

-- Define the vectors a and b in the conditions
def vec_a (t : ℝ) : ℝ × ℝ := (t - 2, 3)
def vec_b : ℝ × ℝ := (3, -1)

-- Define the parallel condition 
def parallel_condition (t : ℝ) :=
  let a_plus_2b := (t - 2 + 6, 3 - 2)
  in a_plus_2b.1 * vec_b.2 = a_plus_2b.2 * vec_b.1

-- Define the magnitude of vec_a
def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1 ^ 2 + v.2 ^ 2)

-- The final statement to prove
theorem find_magnitude_of_vec_a : 
  ∃ t : ℝ, parallel_condition t ∧ magnitude (vec_a t) = 3 * Real.sqrt 10 :=
by
  sorry

end find_magnitude_of_vec_a_l148_148739


namespace sum_of_a_and_b_l148_148083

theorem sum_of_a_and_b {a b : ℝ} (h : a^2 + b^2 + (a*b)^2 = 4*a*b - 1) : a + b = 2 ∨ a + b = -2 :=
sorry

end sum_of_a_and_b_l148_148083


namespace parabola_equation_given_focus_l148_148150

open Real

def parabola_focus (p : ℝ) : ℝ × ℝ := (p / 2, 0)

def hyperbola_left_focus : ℝ × ℝ := (-2, 0)

theorem parabola_equation_given_focus :
  ∀ (p : ℝ), parabola_focus p = hyperbola_left_focus → (∀ x y : ℝ, y^2 = 2 * p * x ↔ y^2 = -8 * x) :=
by
  intro p hp
  have hp' : p = -4 := sorry -- (This would be the part where the proof step solving for p = -4 would be completed)
  intro x y
  split
  { intro h
    rw [hp'] at h
    simp at h
    exact h
  }
  { intro h
    rw [hp']
    simp
    exact h
  }

end parabola_equation_given_focus_l148_148150


namespace find_x_given_inverse_relationship_l148_148877

theorem find_x_given_inverse_relationship :
  ∀ (x y: ℝ), (0 < x ∧ 0 < y) ∧ ((x^3 * y = 64) ↔ (x = 2 ∧ y = 8)) ∧ (y = 500) →
  x = 2 / 5 :=
by
  intros x y h
  sorry

end find_x_given_inverse_relationship_l148_148877


namespace measure_of_AB_l148_148169

-- Define the conditions
variables {a b : ℝ} -- Assume a and b are real numbers
variables {A B C D E : Type} -- Points A, B, C, D, E in Type (they could be points in the plane)
variable [geometry : EuclideanGeometry] -- Assuming Euclidean geometry structure

-- Define segments AB and CD
variables (AB CD AD : geometry.Segment) -- assuming AB, CD, and AD are geometric segments
variable (angle_B angle_D : ℝ) -- angles at points B and D

-- Define the conditions based on the problem statement
axiom AB_parallel_CD : geometry.Parallel AB CD
axiom angle_D_eq_3angle_B : angle_D = 3 * angle_B
axiom AD_length : geometry.length AD = 2 * a
axiom CD_length : geometry.length CD = 3 * b

-- The theorem to prove
theorem measure_of_AB : geometry.length AB = 3 * b :=
by sorry -- proof to be provided

end measure_of_AB_l148_148169


namespace emily_seeds_start_with_l148_148997

-- Define the conditions as hypotheses
variables (big_garden_seeds : ℕ) (small_gardens : ℕ) (seeds_per_small_garden : ℕ)

-- Conditions: Emily planted 29 seeds in the big garden and 4 seeds in each of her 3 small gardens.
def emily_conditions := big_garden_seeds = 29 ∧ small_gardens = 3 ∧ seeds_per_small_garden = 4

-- Define the statement to prove the total number of seeds Emily started with
theorem emily_seeds_start_with (h : emily_conditions big_garden_seeds small_gardens seeds_per_small_garden) : 
(big_garden_seeds + small_gardens * seeds_per_small_garden) = 41 :=
by
  -- Assuming the proof follows logically from conditions
  sorry

end emily_seeds_start_with_l148_148997


namespace probability_of_at_most_3_heads_l148_148412

-- Definitions and conditions
def num_coins : ℕ := 10
def at_most_3_heads_probability : ℚ := 11 / 64

-- Statement of the problem
theorem probability_of_at_most_3_heads (n : ℕ) (p : ℚ) (h1 : n = num_coins) (h2 : p = at_most_3_heads_probability) :
  p = (1 + 10 + 45 + 120 : ℕ) / (2 ^ 10 : ℕ) := by
  sorry

end probability_of_at_most_3_heads_l148_148412


namespace program_arrangements_l148_148280

/-- Given 5 programs, if A, B, and C appear in a specific order, then the number of different
    arrangements is 20. -/
theorem program_arrangements (A B C A_order : ℕ) : 
  (A + B + C + A_order = 5) → 
  (A_order = 3) → 
  (B = 1) → 
  (C = 1) → 
  (A = 1) → 
  (A * B * C * A_order = 1) :=
  by sorry

end program_arrangements_l148_148280


namespace find_n_l148_148671

theorem find_n : ∃ n : ℕ, 0 ≤ n ∧ n ≤ 9 ∧ n ≡ -4229 [MOD 10] ∧ n = 1 := by
  sorry

end find_n_l148_148671


namespace probability_at_most_3_heads_l148_148504

-- Definitions and conditions
def coin_flips : ℕ := 10
def success_prob : ℚ := 1/2
def max_heads : ℕ := 3

-- Binomial Probability Computation
noncomputable def probability_of_at_most_3_heads : ℚ :=
  (∑ k in finset.range (max_heads + 1), nat.choose coin_flips k) / (2^coin_flips)

-- The statement to prove
theorem probability_at_most_3_heads :
  probability_of_at_most_3_heads = 11/64 := by
  sorry

end probability_at_most_3_heads_l148_148504


namespace ratio_of_candy_bar_to_caramel_l148_148864

noncomputable def price_of_caramel : ℝ := 3
noncomputable def price_of_candy_bar (k : ℝ) : ℝ := k * price_of_caramel
noncomputable def price_of_cotton_candy (C : ℝ) : ℝ := 2 * C 

theorem ratio_of_candy_bar_to_caramel (k : ℝ) (C CC : ℝ) :
  C = price_of_candy_bar k →
  CC = price_of_cotton_candy C →
  6 * C + 3 * price_of_caramel + CC = 57 →
  C / price_of_caramel = 2 :=
by
  sorry

end ratio_of_candy_bar_to_caramel_l148_148864


namespace crates_on_tuesday_is_6_l148_148225

-- Define the conditions
def crates_bought_on_tuesday (T : ℕ) : Prop :=
  let total_crates := T + 5 in
  let crates_after_giving := total_crates - 2 in
  let total_eggs := crates_after_giving * 30 in
  total_eggs = 270

-- Define the statement to prove
theorem crates_on_tuesday_is_6 : ∃ T : ℕ, crates_bought_on_tuesday T ∧ T = 6 :=
by
  exists 6
  unfold crates_bought_on_tuesday
  simp
  sorry

end crates_on_tuesday_is_6_l148_148225


namespace gcd_102_238_l148_148260

theorem gcd_102_238 : Nat.gcd 102 238 = 34 :=
by
  -- Given conditions as part of proof structure
  have h1 : 238 = 102 * 2 + 34 := by rfl
  have h2 : 102 = 34 * 3 := by rfl
  sorry

end gcd_102_238_l148_148260


namespace external_angle_bisector_excenter_l148_148970

theorem external_angle_bisector_excenter {O A B C M S : Type}
  (circle : O) (center : O → A)
  (radius : O → ℝ) (radius_eq : ∀ {x : O} (p : A), radius (center x) = radius p)
  (arc_midpoint : A) (S_intersection : A) :
  (∃ A B, (circle = center O) ∧ 
          (radius (center A) = radius (center B)) ∧ 
          (arc_midpoint = M) ∧ 
          (S_intersection = S) ∧ 
          (AM bisector_line (∠ CAB) external) ∧ 
          (BM bisector_line (∠ CBA) external)) →
  (¬ {C : A ↔ M is_excenter_of_triangle Δ ABC}) := sorry

end external_angle_bisector_excenter_l148_148970


namespace arithmetic_sequence_divisibility_l148_148657

theorem arithmetic_sequence_divisibility (a d : ℕ) (h: ∀ n : ℕ, (∏ i in Finset.range n, (a + i * d)) ∣ (∏ i in Finset.range n, (a + (n + i) * d))) :
  ∃ k : ℕ, ∀ n : ℕ, a + n * d = k * (n + 1) :=
begin
  sorry
end

end arithmetic_sequence_divisibility_l148_148657


namespace piglet_balloons_l148_148063

theorem piglet_balloons (n w o total_balloons: ℕ) (H1: w = 2 * n) (H2: o = 4 * n) (H3: n + w + o = total_balloons) (H4: total_balloons = 44) : n - (7 * n - total_balloons) = 2 :=
by
  sorry

end piglet_balloons_l148_148063


namespace cone_volume_from_half_sector_l148_148578

theorem cone_volume_from_half_sector (R : ℝ) (V : ℝ) : 
  R = 6 →
  V = (1/3) * Real.pi * (R / 2)^2 * (R * Real.sqrt 3) →
  V = 9 * Real.pi * Real.sqrt 3 := by sorry

end cone_volume_from_half_sector_l148_148578


namespace number_of_integer_values_l148_148625

theorem number_of_integer_values (n : ℤ) :
  let expr := 3200 * (4 / 5) ^ n in
  (∃ k : ℤ, expr = k) →
  3200 = 2^6 * 5^2 →
  (finset.card ({n | ∃ k : ℤ, 3200 * (4 / 5) ^ n = k}) = 6) :=
by
  sorry

end number_of_integer_values_l148_148625


namespace kahina_chessboard_impossible_l148_148797

theorem kahina_chessboard_impossible :
  ∀ (board : Fin 8 → Fin 8 → Prop), 
    (∀ r c, board r c = false) →
    (∀ r c, (r < 6 ∧ c < 8 → 
             (board (r + 1) c = ¬ board (r + 1) c ∧ 
              board (r + 2) c = ¬ board (r + 2) c ∧ 
              board (r + 3) c = ¬ board (r + 3) c)) ∨
            (r < 8 ∧ c < 6 → 
             (board r (c + 1) = ¬ board r (c + 1) ∧ 
              board r (c + 2) = ¬ board r (c + 2) ∧ 
              board r (c + 3) = ¬ board r (c + 3)))) →
    ¬ ∀ r c, board r c = true :=
begin
  sorry
end

end kahina_chessboard_impossible_l148_148797


namespace cosine_solutions_l148_148051

theorem cosine_solutions (n : ℝ) (h : 0 ≤ n ∧ n ≤ 360) :
  (cos n = cos 145) → (n = 145 ∨ n = 215) :=
by
  sorry

end cosine_solutions_l148_148051


namespace max_participants_win_at_least_three_matches_l148_148567

theorem max_participants_win_at_least_three_matches (n : ℕ) (h : n = 200) : 
  ∃ k : ℕ, k = 66 ∧ ∀ m : ℕ, (k * 3 ≤ m) ∧ (m ≤ 199) → k ≤ m / 3 := 
by
  sorry

end max_participants_win_at_least_three_matches_l148_148567


namespace statement_B_statement_C_statement_D_l148_148069

variables (a b : ℝ)

-- Condition: a > 0
axiom a_pos : a > 0

-- Condition: e^a + ln b = 1
axiom eq1 : Real.exp a + Real.log b = 1

-- Statement B: a + ln b < 0
theorem statement_B : a + Real.log b < 0 :=
  sorry

-- Statement C: e^a + b > 2
theorem statement_C : Real.exp a + b > 2 :=
  sorry

-- Statement D: a + b > 1
theorem statement_D : a + b > 1 :=
  sorry

end statement_B_statement_C_statement_D_l148_148069


namespace greatest_value_of_q_sub_r_l148_148860

-- Definitions and conditions as given in the problem statement
def q : ℕ := 44
def r : ℕ := 15
def n : ℕ := 1027
def d : ℕ := 23

-- Statement of the theorem to be proved
theorem greatest_value_of_q_sub_r : 
  ∃ (q r : ℕ), n = d * q + r ∧ 1 ≤ q ∧ 1 ≤ r ∧ q - r = 29 := 
by
  use q
  use r
  have h1 : n = d * q + r := by norm_num
  have h2 : 1 ≤ q := by norm_num
  have h3 : 1 ≤ r := by norm_num
  have h4 : q - r = 29 := by norm_num
  tauto

end greatest_value_of_q_sub_r_l148_148860


namespace complex_number_proof_l148_148806

def w : ℂ := sorry

-- Given conditions
axiom modulus_w : complex.abs w = 7
axiom on_line_y_equals_x : ∃ x : ℝ, w = x * (1 + complex.I)

-- Proof statements
theorem complex_number_proof : 
  (w * complex.conj w = 49) ∧ 
  (∃ x : ℝ, w = x * (1 + complex.I) ∧ real.abs x = 7 / real.sqrt 2) :=
begin
  split,
  { sorry },
  { obtain ⟨x, hx⟩ := on_line_y_equals_x,
    use x,
    split,
    { exact hx },
    { sorry }
  }
end

end complex_number_proof_l148_148806


namespace probability_heads_at_most_three_out_of_ten_coins_l148_148554

theorem probability_heads_at_most_three_out_of_ten_coins :
  let total_outcomes := 2^10 in
  let favorable_outcomes := Nat.choose 10 0 + Nat.choose 10 1 + Nat.choose 10 2 + Nat.choose 10 3 in
  favorable_outcomes / total_outcomes = 11 / 64 :=
by
  sorry

end probability_heads_at_most_three_out_of_ten_coins_l148_148554


namespace exists_consecutive_composite_l148_148829

theorem exists_consecutive_composite (n : ℕ) : 
  ∃ (a : ℕ → ℕ), 
    (∀ i : ℕ, 2 ≤ i ∧ i ≤ n + 1 → a i = (n+1)! + i ∧ ∀ j : ℕ, 2 ≤ j → j ∣ a i ∧ j ∤ 1 ∧ j ∤ a i → a i ≠ j ∧ a i ≠ ((n+1)! + i) ∧ a i > 1) := 
sorry

end exists_consecutive_composite_l148_148829


namespace gcd_lcm_identity_l148_148802

variables {n m k : ℕ}

/-- Given positive integers n, m, and k such that n divides lcm(m, k) 
    and m divides lcm(n, k), we prove that n * gcd(m, k) = m * gcd(n, k). -/
theorem gcd_lcm_identity (n_pos : 0 < n) (m_pos : 0 < m) (k_pos : 0 < k) 
  (h1 : n ∣ Nat.lcm m k) (h2 : m ∣ Nat.lcm n k) :
  n * Nat.gcd m k = m * Nat.gcd n k :=
sorry

end gcd_lcm_identity_l148_148802


namespace probability_at_most_3_heads_l148_148434

def binom (n k : ℕ) : ℕ := Nat.choose n k

theorem probability_at_most_3_heads :
  let total_outcomes := 2^10
  let successful_outcomes := binom 10 0 + binom 10 1 + binom 10 2 + binom 10 3
  (successful_outcomes : ℚ) / total_outcomes = 11 / 64 := 
by 
  let total_outcomes := 2 ^ 10
  let successful_outcomes := binom 10 0 + binom 10 1 + binom 10 2 + binom 10 3
  have h : (successful_outcomes : ℚ) / total_outcomes = 11 / 64 := sorry
  exact h

end probability_at_most_3_heads_l148_148434


namespace complex_coordinates_l148_148148

theorem complex_coordinates (z : ℂ) (h : complex.I * z = 2 - 4 * complex.I) : z = -4 - 2 * complex.I := by 
  sorry

end complex_coordinates_l148_148148


namespace calculate_spelling_problems_l148_148067

-- Conditions
def math_problems : Nat := 36
def spelling_problems : Nat -- unknown
def rate_problems_per_hour : Nat := 8
def total_hours : Nat := 8

-- Total problems solved
def total_problems : Nat := rate_problems_per_hour * total_hours

-- Proof statement
theorem calculate_spelling_problems (h : math_problems + spelling_problems = total_problems) : spelling_problems = 28 :=
by
  sorry

end calculate_spelling_problems_l148_148067


namespace probability_heads_at_most_3_l148_148538

theorem probability_heads_at_most_3 (num_coins : ℕ) (num_heads_at_most : ℕ) (probability : ℚ):
  num_coins = 10 → 
  num_heads_at_most = 3 → 
  probability = (binomial 10 0 + binomial 10 1 + binomial 10 2 + binomial 10 3) / (2^10) →
  probability = (11 / 64) := 
by
  intros hnum_coins hnum_heads_at_most hprobability
  sorry

end probability_heads_at_most_3_l148_148538


namespace probability_at_most_3_heads_10_coins_l148_148427

noncomputable def coin_flip_prob_at_most_3_heads : ℚ :=
  let total_outcomes := 2^10
  let favorable_outcomes := Nat.binomial 10 0 + Nat.binomial 10 1 + Nat.binomial 10 2 + Nat.binomial 10 3
  favorable_outcomes / total_outcomes

theorem probability_at_most_3_heads_10_coins : coin_flip_prob_at_most_3_heads = 11 / 64 :=
by
  -- Definitions of let values are integrated directly in the theorem body
  let total_outcomes := 2^10
  let favorable_outcomes := Nat.binomial 10 0 + Nat.binomial 10 1 + Nat.binomial 10 2 + Nat.binomial 10 3
  show favorable_outcomes / total_outcomes = 11 / 64
  sorry

end probability_at_most_3_heads_10_coins_l148_148427


namespace no_convex_27gon_with_distinct_integer_angles_l148_148648

noncomputable def sum_of_interior_angles (n : ℕ) : ℕ :=
  (n - 2) * 180

def is_convex (n : ℕ) (angles : Fin n → ℕ) : Prop :=
  ∀ i, angles i < 180

def all_distinct (n : ℕ) (angles : Fin n → ℕ) : Prop :=
  ∀ i j, i ≠ j → angles i ≠ angles j

def sum_is_correct (n : ℕ) (angles : Fin n → ℕ) : Prop :=
  Finset.sum (Finset.univ : Finset (Fin n)) angles = sum_of_interior_angles n

theorem no_convex_27gon_with_distinct_integer_angles :
  ¬ ∃ (angles : Fin 27 → ℕ), is_convex 27 angles ∧ all_distinct 27 angles ∧ sum_is_correct 27 angles :=
by
  sorry

end no_convex_27gon_with_distinct_integer_angles_l148_148648


namespace asha_final_amount_is_correct_l148_148966

noncomputable def total_savings : ℝ :=
  let brother := 25 * 1.183
  let father := 50 * 1.329
  let mother := 35
  let granny := 8000 * 0.009
  let cousin := 60 * 0.193
  let savings := 105
  in brother + father + mother + granny + cousin + savings

noncomputable def conversion_fee (amount : ℝ) : ℝ :=
  0.015 * amount

noncomputable def remaining_after_conversion : ℝ :=
  total_savings - conversion_fee total_savings

noncomputable def remaining_after_transportation_and_gift_wrapping : ℝ :=
  remaining_after_conversion - 12 - 7.5

noncomputable def amount_spent : ℝ :=
  0.75 * remaining_after_transportation_and_gift_wrapping

noncomputable def sales_tax (amount : ℝ) : ℝ :=
  0.08 * amount

noncomputable def remaining_after_purchases : ℝ :=
  remaining_after_transportation_and_gift_wrapping - amount_spent

noncomputable def final_amount_after_all_expenses : ℝ :=
  remaining_after_transportation_and_gift_wrapping - (amount_spent - sales_tax amount_spent)

theorem asha_final_amount_is_correct : final_amount_after_all_expenses = 73.82773125 :=
 by sorry

end asha_final_amount_is_correct_l148_148966


namespace non_determining_condition_for_equilateral_triangle_l148_148783
-- Import Mathlib to include broad mathematical libraries:

-- Define the theorem
theorem non_determining_condition_for_equilateral_triangle
  (A B C D : Type)
  (angle_A_eq_60 : ∠ A B C = 60)
  (AD_perp_BC : A - C - D = 90) :
  ∃ (B' C' : Type), ∠ A B' C' ≠ 60 ∧ A - B - C = A - B' - C' :=
by
  sorry

end non_determining_condition_for_equilateral_triangle_l148_148783


namespace lcm_5_6_10_18_l148_148297

theorem lcm_5_6_10_18 : Nat.lcm 5 6 10 18 = 90 :=
  by sorry

end lcm_5_6_10_18_l148_148297


namespace M_greater_or_equal_N_l148_148337

-- Define the sequence of 0's and 1's as a list of integers for simplicity
def sequence : List ℤ := ...

-- Define M and N
def M (seq : List ℤ) : ℕ := 
  seq.pairs_count_even_between  -- This is a placeholder for the actual calculation

def N (seq : List ℤ) : ℕ := 
  seq.pairs_count_odd_between  -- This is a placeholder for the actual calculation

theorem M_greater_or_equal_N (seq : List ℤ) : M seq ≥ N seq := 
  sorry  -- Proof to be completed

end M_greater_or_equal_N_l148_148337


namespace Josh_wallet_total_l148_148796

theorem Josh_wallet_total (wallet_money : ℝ) (investment : ℝ) (increase_percentage : ℝ) (increase_amount : ℝ) (final_investment_value : ℝ) :
  wallet_money = 300 →
  investment = 2000 →
  increase_percentage = 0.30 →
  increase_amount = investment * increase_percentage →
  final_investment_value = investment + increase_amount →
  wallet_money + final_investment_value = 2900 :=
begin
  intros h1 h2 h3 h4 h5,
  rw [h1, h2, h3, h4, h5],
  sorry
end

end Josh_wallet_total_l148_148796


namespace smallest_number_of_coins_l148_148291

theorem smallest_number_of_coins (p n d q h: ℕ) (total: ℕ) 
  (coin_value: ℕ → ℕ)
  (h_p: coin_value 1 = 1) 
  (h_n: coin_value 5 = 5) 
  (h_d: coin_value 10 = 10) 
  (h_q: coin_value 25 = 25) 
  (h_h: coin_value 50 = 50)
  (total_def: total = p * (coin_value 1) + n * (coin_value 5) +
                     d * (coin_value 10) + q * (coin_value 25) + 
                     h * (coin_value 50))
  (h_total: total = 100): 
  p + n + d + q + h = 3 :=
by
  sorry

end smallest_number_of_coins_l148_148291


namespace max_value_f_l148_148071

-- Definitions for the conditions and functions.
def f (m : ℝ) (θ : ℝ) : ℝ := (Real.sin θ)^2 + m * (Real.cos θ)

-- The maximum value g(m) we want to prove.
def g (m : ℝ) : ℝ := m

-- The maximum function theorem under the given conditions.
theorem max_value_f (m : ℝ) (h : m > 2) : ∃ θ : ℝ, f m θ = g m :=
by
  sorry

end max_value_f_l148_148071


namespace nine_digit_number_l148_148283

open Nat

theorem nine_digit_number 
  (a1 a2 a3 b1 b2 b3 : ℕ) 
  (n : ℕ) 
  (h1 : n = a1 * 10^8 + a2 * 10^7 + a3 * 10^6 + b1 * 10^5 + b2 * 10^4 + b3 * 10^3 + a1 * 10^2 + a2 * 10 + a3)
  (h2 : b1 * 10^2 + b2 * 10 + b3 = 2 * (a1 * 10^2 + a2 * 10 + a3))
  (h3 : a1 ≠ 0)
  (h4 : ∃ p1 p2 p3 p4 p5 : ℕ, prime p1 ∧ prime p2 ∧ prime p3 ∧ prime p4 ∧ prime p5 ∧ m = p1 * p2 * p3 * p4 * p5 ∧ n = m ^ 2) :
  n = 100200100 ∨ n = 225450225 :=
sorry

end nine_digit_number_l148_148283


namespace function_property_l148_148151

theorem function_property 
  (f : ℝ → ℝ) 
  (hf : ∀ x, x ≠ 0 → f (1 - 2 * x) = (1 - x^2) / (x^2)) 
  : 
  (f (1 / 2) = 15) ∧
  (∀ x, x ≠ 1 → f (x) = 4 / (x - 1)^2 - 1) ∧
  (∀ x, x ≠ 0 → x ≠ 1 → f (1 / x) = 4 * x^2 / (x - 1)^2 - 1) :=
by {
  sorry
}

end function_property_l148_148151


namespace probability_heads_at_most_3_l148_148533

theorem probability_heads_at_most_3 (num_coins : ℕ) (num_heads_at_most : ℕ) (probability : ℚ):
  num_coins = 10 → 
  num_heads_at_most = 3 → 
  probability = (binomial 10 0 + binomial 10 1 + binomial 10 2 + binomial 10 3) / (2^10) →
  probability = (11 / 64) := 
by
  intros hnum_coins hnum_heads_at_most hprobability
  sorry

end probability_heads_at_most_3_l148_148533


namespace vasya_correct_l148_148821

theorem vasya_correct (n : ℕ) (x : Fin n → ℕ) 
    (h1 : (∏ i in Finset.range n, x i) = 10) :
    (1024 = 2^10) ∧ ((2:ℤ) = 2) →   -- We take these two just for ensuring completeness.
    ∏ i in Finset.range n, x i = 10 :=
by
  sorry

end vasya_correct_l148_148821


namespace number_of_plants_to_achieve_20_yuan_l148_148576

-- Define the function representing the profit per pot
def profit_per_pot (n : ℕ) : ℝ :=
  n * (5 - 0.5 * (n - 3))

-- State the theorem to be proved
theorem number_of_plants_to_achieve_20_yuan :
  ∃ n : ℕ, (n ≤ 6) ∧ (profit_per_pot n = 20) ∧ (3 ≤ n) :=
begin
  use 5,
  split,
  { -- Verifying the number of plants per pot does not exceed 6
    norm_num },
  split,
  { -- Verifying the profit per pot equals 20 yuan with 5 plants per pot
    show profit_per_pot 5 = 20,
    norm_num },
  { -- Verifying the number of plants is at least 3
    norm_num }
end

end number_of_plants_to_achieve_20_yuan_l148_148576


namespace probability_increasing_function_l148_148070

def is_increasing (a b : ℝ) := 0 < b / a ∧ b / a < 1

theorem probability_increasing_function :
  let a_vals := {1, 3, 5}
  let b_vals := {2, 4, 8}
  let valid_pairs := { (a, b) | a ∈ a_vals ∧ b ∈ b_vals ∧ is_increasing a b }
  valid_pairs.size = 3 / 9 :=
by
  sorry

end probability_increasing_function_l148_148070


namespace muffin_half_as_expensive_as_banana_l148_148246

-- Define Susie's expenditure in terms of muffin cost (m) and banana cost (b)
def susie_expenditure (m b : ℝ) : ℝ := 5 * m + 2 * b

-- Define Calvin's expenditure as three times Susie's expenditure
def calvin_expenditure_via_susie (m b : ℝ) : ℝ := 3 * (susie_expenditure m b)

-- Define Calvin's direct expenditure on muffins and bananas
def calvin_direct_expenditure (m b : ℝ) : ℝ := 3 * m + 12 * b

-- Formulate the theorem stating the relationship between muffin and banana costs
theorem muffin_half_as_expensive_as_banana (m b : ℝ) 
  (h₁ : susie_expenditure m b = 5 * m + 2 * b)
  (h₂ : calvin_expenditure_via_susie m b = calvin_direct_expenditure m b) : 
  m = (1/2) * b := 
by {
  -- These conditions automatically fulfill the given problem requirements.
  sorry
}

end muffin_half_as_expensive_as_banana_l148_148246


namespace arithmetic_series_sum_l148_148059

theorem arithmetic_series_sum :
  let a1 := 40
  let an := 60
  let d := 1 / 7
  let n := 141
  2 * (an - a1) * d + a1 = an ∧
  n = ((an - a1) / d) + 1 →
  (n * (a1 + an)) / 2 = 7050 :=
begin
  sorry
end

end arithmetic_series_sum_l148_148059


namespace jack_total_dollars_l148_148190

-- Constants
def initial_dollars : ℝ := 45
def euro_amount : ℝ := 36
def yen_amount : ℝ := 1350
def ruble_amount : ℝ := 1500
def euro_to_dollar : ℝ := 2
def yen_to_dollar : ℝ := 0.009
def ruble_to_dollar : ℝ := 0.013
def transaction_fee_rate : ℝ := 0.01
def spending_rate : ℝ := 0.1

-- Convert each foreign currency to dollars
def euros_to_dollars : ℝ := euro_amount * euro_to_dollar
def yen_to_dollars : ℝ := yen_amount * yen_to_dollar
def rubles_to_dollars : ℝ := ruble_amount * ruble_to_dollar

-- Calculate transaction fees for each currency conversion
def euros_fee : ℝ := euros_to_dollars * transaction_fee_rate
def yen_fee : ℝ := yen_to_dollars * transaction_fee_rate
def rubles_fee : ℝ := rubles_to_dollars * transaction_fee_rate

-- Subtract transaction fees from the converted amounts
def euros_after_fee : ℝ := euros_to_dollars - euros_fee
def yen_after_fee : ℝ := yen_to_dollars - yen_fee
def rubles_after_fee : ℝ := rubles_to_dollars - rubles_fee

-- Calculate total dollars after conversion and fees
def total_dollars_before_spending : ℝ := initial_dollars + euros_after_fee + yen_after_fee + rubles_after_fee

-- Calculate 10% expenditure
def spending_amount : ℝ := total_dollars_before_spending * spending_rate

-- Calculate final amount after spending
def final_amount : ℝ := total_dollars_before_spending - spending_amount

theorem jack_total_dollars : final_amount = 132.85 := by
  sorry

end jack_total_dollars_l148_148190


namespace domain_g_l148_148714

-- Definitions and conditions
def f : ℝ → ℝ := sorry  -- Abstract function f with domain [-8, 1]

-- g function definition
def g (x : ℝ) : ℝ := f (2*x + 1) / (x + 2)

-- Domain condition for f
def domain_f := {x : ℝ | -8 ≤ x ∧ x ≤ 1}

-- Condition that must hold:
theorem domain_g : {x : ℝ | (x ∈ (-9/2 : ℝ)..0) ∧ x ≠ -2} = {x : ℝ | (-9/2 < x ∧ x < -2) ∨ (-2 < x ∧ x ≤ 0)} :=
by
  sorry

end domain_g_l148_148714


namespace Josh_wallet_total_l148_148795

theorem Josh_wallet_total (wallet_money : ℝ) (investment : ℝ) (increase_percentage : ℝ) (increase_amount : ℝ) (final_investment_value : ℝ) :
  wallet_money = 300 →
  investment = 2000 →
  increase_percentage = 0.30 →
  increase_amount = investment * increase_percentage →
  final_investment_value = investment + increase_amount →
  wallet_money + final_investment_value = 2900 :=
begin
  intros h1 h2 h3 h4 h5,
  rw [h1, h2, h3, h4, h5],
  sorry
end

end Josh_wallet_total_l148_148795


namespace probability_at_most_3_heads_10_coins_l148_148423

noncomputable def coin_flip_prob_at_most_3_heads : ℚ :=
  let total_outcomes := 2^10
  let favorable_outcomes := Nat.binomial 10 0 + Nat.binomial 10 1 + Nat.binomial 10 2 + Nat.binomial 10 3
  favorable_outcomes / total_outcomes

theorem probability_at_most_3_heads_10_coins : coin_flip_prob_at_most_3_heads = 11 / 64 :=
by
  -- Definitions of let values are integrated directly in the theorem body
  let total_outcomes := 2^10
  let favorable_outcomes := Nat.binomial 10 0 + Nat.binomial 10 1 + Nat.binomial 10 2 + Nat.binomial 10 3
  show favorable_outcomes / total_outcomes = 11 / 64
  sorry

end probability_at_most_3_heads_10_coins_l148_148423


namespace binary_to_decimal_l148_148030

theorem binary_to_decimal :
  1 * 2^8 + 0 * 2^7 + 1 * 2^6 + 1 * 2^5 + 1 * 2^4 + 1 * 2^3 + 0 * 2^2 + 1 * 2^1 + 1 * 2^0 = 379 :=
by
  sorry

end binary_to_decimal_l148_148030


namespace paintings_in_four_weeks_l148_148957

def weekly_hours := 30
def hours_per_painting := 3
def weeks := 4

theorem paintings_in_four_weeks (w_hours : ℕ) (h_per_painting : ℕ) (n_weeks : ℕ) (result : ℕ) :
  w_hours = weekly_hours →
  h_per_painting = hours_per_painting →
  n_weeks = weeks →
  result = (w_hours / h_per_painting) * n_weeks →
  result = 40 :=
by
  intros
  sorry

end paintings_in_four_weeks_l148_148957


namespace sum_max_min_n_l148_148571

theorem sum_max_min_n (A B : Finset ℕ) (hAUB : (A ∪ B).card = 48) (hA : A.card = 24) (hB : B.card = 30) :
  let n := (A ∩ B).card in n + min 24 30 = 30 :=
by
  sorry

end sum_max_min_n_l148_148571


namespace analytical_expression_monotonic_increase_l148_148110

noncomputable def f (x : ℝ) : ℝ := 3 * Real.sin(2 * x + (Real.pi / 3))

theorem analytical_expression :
  ∃ (A ω θ : ℝ), (A > 0) ∧ (ω > 0) ∧ (|θ| < Real.pi / 2) ∧ 
  (∀ x, f x = A * Real.sin(ω * x + θ)) ∧ 
  (ω = 2) ∧ (θ = Real.pi / 3) := 
sorry

theorem monotonic_increase :
  ∀ x, (0 ≤ x ∧ x ≤ Real.pi) →
  ((0 ≤ x ∧ x ≤ Real.pi / 12) ∨ (7 * Real.pi / 12 ≤ x ∧ x ≤ Real.pi)) :=
sorry

end analytical_expression_monotonic_increase_l148_148110


namespace value_of_f_6_l148_148332

noncomputable def f : ℤ → ℤ 
| n := if n = 4 then 12 else f (n - 1) - n

theorem value_of_f_6 : f 6 = 1 := by
  sorry

end value_of_f_6_l148_148332


namespace number_of_true_propositions_l148_148956

-- Given conditions and definitions of the propositions
def prop1 : Prop := (∫ x in 0..(π / 2), Real.cos x) = 1
def prop2 : Prop := ¬(∀ m, (m = -2) → ((m + 2) * (m - 2) + m * (m + 2) = 0 ↔ m = -2))
def prop3 : Prop := ¬(∀ ξ, ξ ~ (Gaussian 0 σ) → (P(-2 ≤ ξ ∧ ξ ≤ 0) = 0.4 → P(ξ > 2) = 0.2))
def prop4 : Prop := let C1 := circle (-1, 0) 1 in
                     let C2 := circle (0, -1) (sqrt 2) in
                     ∃ tangents, count tangents = 2

-- Statement of the problem
theorem number_of_true_propositions : (count_true [prop1, prop2, prop3, prop4] = 2) :=
sorry

end number_of_true_propositions_l148_148956


namespace solution_set_of_inequality_l148_148275

theorem solution_set_of_inequality :
  { x : ℝ | (x - 1) / x ≥ 2 } = { x : ℝ | -1 ≤ x ∧ x < 0 } :=
by
  sorry

end solution_set_of_inequality_l148_148275


namespace f_prime_at_1_l148_148693

-- Define the function f(x)
def f (x : ℝ) : ℝ := Real.exp (-x) + Real.exp x

-- Define the first derivative of the function f(x)
def f' (x : ℝ) : ℝ := - (Real.exp (-x)) + Real.exp x

-- State the theorem that f'(1) = e - 1/e
theorem f_prime_at_1 : f' 1 = Real.exp 1 - Real.exp (-1) :=
by
  sorry

end f_prime_at_1_l148_148693


namespace square_of_1031_l148_148982

theorem square_of_1031 : 1031 ^ 2 = 1060961 := by
  calc
    1031 ^ 2 = (1000 + 31) ^ 2       : by sorry
           ... = 1000 ^ 2 + 2 * 1000 * 31 + 31 ^ 2 : by sorry
           ... = 1000000 + 62000 + 961 : by sorry
           ... = 1060961 : by sorry

end square_of_1031_l148_148982


namespace probability_at_most_3_heads_10_coins_l148_148428

noncomputable def coin_flip_prob_at_most_3_heads : ℚ :=
  let total_outcomes := 2^10
  let favorable_outcomes := Nat.binomial 10 0 + Nat.binomial 10 1 + Nat.binomial 10 2 + Nat.binomial 10 3
  favorable_outcomes / total_outcomes

theorem probability_at_most_3_heads_10_coins : coin_flip_prob_at_most_3_heads = 11 / 64 :=
by
  -- Definitions of let values are integrated directly in the theorem body
  let total_outcomes := 2^10
  let favorable_outcomes := Nat.binomial 10 0 + Nat.binomial 10 1 + Nat.binomial 10 2 + Nat.binomial 10 3
  show favorable_outcomes / total_outcomes = 11 / 64
  sorry

end probability_at_most_3_heads_10_coins_l148_148428


namespace stations_equation_l148_148744

theorem stations_equation (x : ℕ) (h : x * (x - 1) = 1482) : true :=
by
  sorry

end stations_equation_l148_148744


namespace tan_ratio_sum_l148_148212

variables (x y : ℝ)

-- condition 1
axiom condition1 : (sin x / cos y) + (sin y / cos x) = 1
-- condition 2
axiom condition2 : (cos x / sin y) + (cos y / sin x) = 7

theorem tan_ratio_sum :
  (tan x / tan y) + (tan y / tan x) = 138 / 15 := by
  sorry

end tan_ratio_sum_l148_148212


namespace fixed_fee_rental_l148_148226

theorem fixed_fee_rental (F C h : ℕ) (hC : C = F + 7 * h) (hC80 : C = 80) (hh9 : h = 9) : F = 17 :=
by
  sorry

end fixed_fee_rental_l148_148226


namespace sum_of_real_solutions_l148_148676

theorem sum_of_real_solutions:
  (∃ (s : ℝ), ∀ x : ℝ, 
    (x - 3) / (x^2 + 6 * x + 2) = (x - 6) / (x^2 - 12 * x) → 
    s = 106 / 9) :=
  sorry

end sum_of_real_solutions_l148_148676


namespace min_value_expression_l148_148208

open Real

theorem min_value_expression (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_condition : a * b * c = 1) :
  a^2 + 8 * a * b + 32 * b^2 + 24 * b * c + 8 * c^2 ≥ 36 :=
by
  sorry

end min_value_expression_l148_148208


namespace probability_at_most_3_heads_l148_148501

-- Definitions and conditions
def coin_flips : ℕ := 10
def success_prob : ℚ := 1/2
def max_heads : ℕ := 3

-- Binomial Probability Computation
noncomputable def probability_of_at_most_3_heads : ℚ :=
  (∑ k in finset.range (max_heads + 1), nat.choose coin_flips k) / (2^coin_flips)

-- The statement to prove
theorem probability_at_most_3_heads :
  probability_of_at_most_3_heads = 11/64 := by
  sorry

end probability_at_most_3_heads_l148_148501


namespace bronson_total_profit_l148_148619

def cost_per_bushel := 12
def apples_per_bushel := 48
def selling_price_per_apple := 0.40
def number_of_apples_sold := 100

theorem bronson_total_profit : 
  let cost_per_apple := cost_per_bushel / apples_per_bushel in
  let profit_per_apple := selling_price_per_apple - cost_per_apple in
  let total_profit := profit_per_apple * number_of_apples_sold in
  total_profit = 15 :=
by
  sorry

end bronson_total_profit_l148_148619


namespace probability_of_at_most_3_heads_l148_148414

-- Definitions and conditions
def num_coins : ℕ := 10
def at_most_3_heads_probability : ℚ := 11 / 64

-- Statement of the problem
theorem probability_of_at_most_3_heads (n : ℕ) (p : ℚ) (h1 : n = num_coins) (h2 : p = at_most_3_heads_probability) :
  p = (1 + 10 + 45 + 120 : ℕ) / (2 ^ 10 : ℕ) := by
  sorry

end probability_of_at_most_3_heads_l148_148414


namespace find_line_eq_l148_148107

variables {a b c : ℝ} (h_c : c > 0) {x y : ℝ}

def ellipse (x y a b : ℝ) : Prop :=
  (x^2)/(a^2) + (y^2)/(b^2) = 1

def line_through_focus (x y c : ℝ) : Prop :=
  y = x + c ∨ y = -x + c

theorem find_line_eq (h1 : ellipse x y a b)
                     (h2 : line_through_focus x y c)
                     (h_perpendicular : ⊥)
                     (h_focus : x = -c ∧ y = 0)
                     (h_intersect_AB : ∃ A B : ℝ, A = (x, y) ∧ B = (x, -y))
                     (h_intersect_CD : ∃ C D : ℝ, C = (x, y) ∧ D = (0, y)) :
                     y = x + c ∨ y = -x + c :=
by
  sorry

end find_line_eq_l148_148107


namespace ellipse_sum_l148_148985

-- Define the givens
def h : ℤ := -3
def k : ℤ := 5
def a : ℤ := 7
def b : ℤ := 4

-- State the theorem to be proven
theorem ellipse_sum : h + k + a + b = 13 := by
  sorry

end ellipse_sum_l148_148985


namespace John_pays_more_than_Jane_l148_148193

theorem John_pays_more_than_Jane : 
  let original_price := 24.00000000000002
  let discount_rate := 0.10
  let tip_rate := 0.15
  let discount := discount_rate * original_price
  let discounted_price := original_price - discount
  let john_tip := tip_rate * original_price
  let jane_tip := tip_rate * discounted_price
  let john_total := discounted_price + john_tip
  let jane_total := discounted_price + jane_tip
  john_total - jane_total = 0.3600000000000003 :=
by
  sorry

end John_pays_more_than_Jane_l148_148193


namespace average_weight_of_three_l148_148184

theorem average_weight_of_three (Ishmael Ponce Jalen : ℕ) 
  (h1 : Jalen = 160) 
  (h2 : Ponce = Jalen - 10) 
  (h3 : Ishmael = Ponce + 20) : 
  (Ishmael + Ponce + Jalen) / 3 = 160 := 
sorry

end average_weight_of_three_l148_148184


namespace probability_at_most_3_heads_l148_148441

def binom (n k : ℕ) : ℕ := Nat.choose n k

theorem probability_at_most_3_heads :
  let total_outcomes := 2^10
  let successful_outcomes := binom 10 0 + binom 10 1 + binom 10 2 + binom 10 3
  (successful_outcomes : ℚ) / total_outcomes = 11 / 64 := 
by 
  let total_outcomes := 2 ^ 10
  let successful_outcomes := binom 10 0 + binom 10 1 + binom 10 2 + binom 10 3
  have h : (successful_outcomes : ℚ) / total_outcomes = 11 / 64 := sorry
  exact h

end probability_at_most_3_heads_l148_148441


namespace two_pow_divides_a_seq_l148_148963

open Nat

noncomputable def a_seq : ℕ → ℕ
| 0     := 0
| 1     := 1
| (n+2) := 2 * a_seq (n+1) + a_seq n

theorem two_pow_divides_a_seq (n k : ℕ) : 
  (2^k ∣ a_seq n) ↔ (2^k ∣ n) :=
by
  sorry

end two_pow_divides_a_seq_l148_148963


namespace area_of_triangle_l148_148883

-- Step 1: Define the first line
def line1 := λ (x: ℝ), (1/3) * x + 8/3

-- Step 2: Define the second line
def line2 := λ (x: ℝ), 3 * x

-- Step 3: Define the third line equation implicitly
def line3 := λ (x y: ℝ), x + y = 9

-- Step 4: Define the intersection points of the lines
def intersection1 := (19 / 4, 27 / 4)
def intersection2 := (9 / 4, 27 / 4)
def point3 := (1, 3)  -- Given common point (1, 3)

-- Step 5: Prove the area of the triangle formed by these three points is 15/16
theorem area_of_triangle : 
  let area (a b c : (ℝ × ℝ)) := 
    (1 / 2) * abs (a.1 * (b.2 - c.2) + b.1 * (c.2 - a.2) + c.1 * (a.2 - b.2)) in
  area point3 intersection1 intersection2 = 15 / 16 :=
sorry

end area_of_triangle_l148_148883


namespace remaining_pieces_l148_148040

variables (sh_s : Nat) (sh_f : Nat) (sh_r : Nat)
variables (sh_pairs_s : Nat) (sh_pairs_f : Nat) (sh_pairs_r : Nat)
variables (total : Nat)

def conditions :=
  sh_s = 20 ∧
  sh_f = 12 ∧
  sh_r = 20 - 12 ∧
  sh_pairs_s = 8 ∧
  sh_pairs_f = 5 ∧
  sh_pairs_r = 8 - 5 ∧
  total = sh_r + sh_pairs_r

theorem remaining_pieces : conditions → total = 11 :=
by intro h; cases h with _ h'; cases h' with _ h''; cases h'' with _ h''';
   cases h''' with _ h''''; cases h'''' with _ h''''' ; cases h''''' with _ _ ;
   sorry

end remaining_pieces_l148_148040


namespace remaining_pieces_l148_148041

variables (sh_s : Nat) (sh_f : Nat) (sh_r : Nat)
variables (sh_pairs_s : Nat) (sh_pairs_f : Nat) (sh_pairs_r : Nat)
variables (total : Nat)

def conditions :=
  sh_s = 20 ∧
  sh_f = 12 ∧
  sh_r = 20 - 12 ∧
  sh_pairs_s = 8 ∧
  sh_pairs_f = 5 ∧
  sh_pairs_r = 8 - 5 ∧
  total = sh_r + sh_pairs_r

theorem remaining_pieces : conditions → total = 11 :=
by intro h; cases h with _ h'; cases h' with _ h''; cases h'' with _ h''';
   cases h''' with _ h''''; cases h'''' with _ h''''' ; cases h''''' with _ _ ;
   sorry

end remaining_pieces_l148_148041


namespace two_x_minus_six_y_equals_neg21_l148_148221

-- Define points A and B
structure Point where
  x : ℝ
  y : ℝ

def A : Point := { x := 20, y := 9 }
def B : Point := { x := 4, y := 6 }

-- Define midpoint calculation
def midpoint (P Q : Point) : Point :=
  { x := (P.x + Q.x) / 2, y := (P.y + Q.y) / 2 }

-- Define point C as the midpoint of A and B
def C : Point := midpoint A B

-- Theorem to prove the required expression equals -21
theorem two_x_minus_six_y_equals_neg21 : 2 * C.x - 6 * C.y = -21 := by
  sorry

end two_x_minus_six_y_equals_neg21_l148_148221


namespace ln_sqrt_lt_sqrt_minus_one_exp_x_minus_one_gt_x_f_gt_sqrt_plus_log_div_2_l148_148725

variables {x t : ℝ}

theorem ln_sqrt_lt_sqrt_minus_one (hx : x > 1) : log (sqrt x) < sqrt x - 1 :=
sorry

theorem exp_x_minus_one_gt_x (hx : x > 1) : exp (x - 1) > x :=
sorry

theorem f_gt_sqrt_plus_log_div_2 (hx : x > 1) (ht : t > -1) : 
  (x + t) / (x - 1) * exp (x - 1) > sqrt x * (1 + 1/2 * log x) :=
sorry

end ln_sqrt_lt_sqrt_minus_one_exp_x_minus_one_gt_x_f_gt_sqrt_plus_log_div_2_l148_148725


namespace tangent_length_possible_values_l148_148683

theorem tangent_length_possible_values :
  ∃ (t : ℤ), 
    (∀ m : ℤ, 1 ≤ m ∧ m ≤ 9 → t = Int.sqrt (m * (10 - m))) ∧
    ((∃ m : ℤ, t = 4 ∧ 1 ≤ m ∧ m ≤ 9) ∧ (∃ m : ℤ, t = 5 ∧ 1 ≤ m ∧ m ≤ 9)) ∧ 
    ((•the number of integer solutions for t == length of the tangent is exactly 2•)) := sorry

end tangent_length_possible_values_l148_148683


namespace proof_of_multiplication_distance_l148_148078

-- Define the properties of the triangle and the extension point
def PE : ℝ := 3
def PF : ℝ := 5
def EF : ℝ := 7
def PA : ℝ := 1.5

noncomputable def circumcenter_distance_mul_two : ℝ :=
  2 * (sqrt ((PE / 2) * (PE / 2) + (PF / 2) * (PF / 2)) / 2.5)

theorem proof_of_multiplication_distance : circumcenter_distance_mul_two = 5 := by
  sorry

end proof_of_multiplication_distance_l148_148078


namespace probability_10_coins_at_most_3_heads_l148_148393

def probability_at_most_3_heads (n : ℕ) : ℚ :=
  let total_outcomes := 2^n
  let favorable_outcomes := (Nat.choose n 0) + (Nat.choose n 1) + (Nat.choose n 2) + (Nat.choose n 3)
  favorable_outcomes / total_outcomes

theorem probability_10_coins_at_most_3_heads : probability_at_most_3_heads 10 = 11 / 64 :=
by
  sorry

end probability_10_coins_at_most_3_heads_l148_148393


namespace area_triangle_DEF_l148_148583

-- Define the areas of the smaller triangles
def u1_area : ℕ := 16
def u2_area : ℕ := 25
def u3_area : ℕ := 36

-- Define the area of triangle DEF as a variable 
def area_DEF (A B C : Triangle) (Q : Point) (u1 u2 u3 : Triangle) : ℕ :=
if (is_interior Q A B C) ∧
   (is_parallel_line_through Q (side_AB A B C) (side_AB u1 u2 u3)) ∧
   (is_parallel_line_through Q (side_AC A B C) (side_AC u1 u2 u3)) ∧
   (is_parallel_line_through Q (side_BC A B C) (side_BC u1 u2 u3)) ∧
   (area u1 = u1_area) ∧
   (area u2 = u2_area) ∧
   (area u3 = u3_area) 
then
  225 
else 
  0

-- Theorem stating the required proof problem
theorem area_triangle_DEF (A B C : Triangle) (Q : Point) (u1 u2 u3 : Triangle) :
  (is_interior Q A B C) → 
  (is_parallel_line_through Q (side_AB A B C) (side_AB u1 u2 u3)) →
  (is_parallel_line_through Q (side_AC A B C) (side_AC u1 u2 u3)) →
  (is_parallel_line_through Q (side_BC A B C) (side_BC u1 u2 u3)) →
  (area u1 = u1_area) →
  (area u2 = u2_area) →
  (area u3 = u3_area) →
  area_DEF A B C Q u1 u2 u3 = 225 :=
by sorry

end area_triangle_DEF_l148_148583


namespace tangent_line_at_0_1_l148_148668

noncomputable def curve (x : ℝ) : ℝ := Real.cos x - x / 2

def tangent_line_eq (x y : ℝ) : Prop := x + 2 * y = 2

theorem tangent_line_at_0_1 : tangent_line_eq 0 1 :=
by
  -- Definition of the curve
  let y := curve
  -- Derivative of the curve at x = 0
  have derivative_at_0 : deriv y 0 = -1 / 2 :=
    sorry
  -- Equation of the tangent line
  let t_l := λ x y : ℝ, y - 1 + 1 / 2 * x = 0
  -- Simplify to standard form x + 2 * y = 2
  show tangent_line_eq 0 1

end tangent_line_at_0_1_l148_148668


namespace average_words_per_hour_l148_148582

theorem average_words_per_hour :
  let words := 60000
  let total_hours := 100
  let break_hours := 15
  let writing_hours := total_hours - break_hours
  let average_words_per_hour := words / writing_hours
  (average_words_per_hour.toNat == 705) :=
by
  let words := 60000
  let total_hours := 100
  let break_hours := 15
  let writing_hours := total_hours - break_hours
  let average_words_per_hour := words / writing_hours
  show words / writing_hours == 705 from sorry

end average_words_per_hour_l148_148582


namespace correct_operation_l148_148321

theorem correct_operation (x : ℝ) : (x^2) * (x^4) = x^6 :=
  sorry

end correct_operation_l148_148321


namespace abcd_zero_l148_148830

theorem abcd_zero (a b c d : ℝ) (h1 : a + b + c + d = 0) (h2 : ab + ac + bc + bd + ad + cd = 0) : 
  a = 0 ∧ b = 0 ∧ c = 0 ∧ d = 0 :=
sorry

end abcd_zero_l148_148830


namespace tree_height_increase_l148_148915

theorem tree_height_increase (h : ℚ) (initial_height : ℚ) : initial_height = 4 →
    (∀ n : ℕ, 0 ≤ n ∧ n ≤ 6 → initial_height + n * h) →
    (initial_height + 6 * h = (initial_height + 4 * h) + (1 / 7) * (initial_height + 4 * h)) →
    h = 2 / 5 :=
by
  intros h initial_height initial_height_eq cond height_relation
  sorry

end tree_height_increase_l148_148915


namespace lcm_5_6_10_18_l148_148296

theorem lcm_5_6_10_18 : Nat.lcm 5 6 10 18 = 90 :=
  by sorry

end lcm_5_6_10_18_l148_148296


namespace disc_partition_impossible_l148_148627

-- Definition of the disc of radius 1
def disc : set (ℝ × ℝ) :=
  { p | (p.1)^2 + (p.2)^2 ≤ 1 }

-- The main problem statement
theorem disc_partition_impossible :
  ¬(∃ (A B C : set (ℝ × ℝ)),
    (A ∪ B ∪ C = disc) ∧
    (∀ (p q : ℝ × ℝ), p ∈ A ∧ q ∈ A ∧ dist p q = 1 → false) ∧
    (∀ (p q : ℝ × ℝ), p ∈ B ∧ q ∈ B ∧ dist p q = 1 → false) ∧
    (∀ (p q : ℝ × ℝ), p ∈ C ∧ q ∈ C ∧ dist p q = 1 → false))
:=
by sorry

end disc_partition_impossible_l148_148627


namespace max_dead_souls_l148_148977

theorem max_dead_souls :
  ∃ m, (∀ (distribution : List ℕ), 
    distribution.length = 3 ∧ distribution.sum = 1001 →
    ∃ N (hN : 1 ≤ N ∧ N ≤ 1001)
      (additional_nuts : ℕ),  
      1 ≤ additional_nuts ∧ additional_nuts ≤ 71 ∧
      (∃ (used_boxes : List ℕ), 
        (used_boxes.sum = N ∧ used_boxes.length ≥ 1 ∧ used_boxes.length ≤ 3))) ∧ 
  (∀ (strategy : ℕ → List ℕ), ∀ N, 1 ≤ N ∧ N ≤ 1001 →
    strategy N ⊆ [0,1,2,3] → additional_nuts = 71) := 
  sorry

end max_dead_souls_l148_148977


namespace triangle_ratio_l148_148180

variables {Point : Type} [InnerProductSpace ℝ Point]

-- Define points A, B, C, M, N, K
variables (A B C M N K : Point)

-- Define vector notations
noncomputable def vecAB := B - A
noncomputable def vecAC := C - A

-- Define the conditions
def median_BM (M: Point) : Prop := M = (B + C) / 2
def median_BN (N : Point) : Prop := N = (A + B + M) / 3
def median_NK (K : Point) : Prop := K = (B + N + C) / 3
def perpendicular_NK_BM : Prop := inner ((K - N) : RealEuclideanSpace ℝ) ((M - B) : RealEuclideanSpace ℝ) = 0

-- Prove the result
theorem triangle_ratio (h1 : median_BM M) (h2 : median_BN N) (h3 : median_NK K) (h4 : perpendicular_NK_BM) :
  (dist A B) / (dist A C) = 1 / 2 := sorry

end triangle_ratio_l148_148180


namespace bret_spends_77_dollars_l148_148618

def num_people : ℕ := 4
def main_meal_cost : ℝ := 12.0
def num_appetizers : ℕ := 2
def appetizer_cost : ℝ := 6.0
def tip_rate : ℝ := 0.20
def rush_order_fee : ℝ := 5.0

def total_cost (num_people : ℕ) (main_meal_cost : ℝ) (num_appetizers : ℕ) (appetizer_cost : ℝ) (tip_rate : ℝ) (rush_order_fee : ℝ) : ℝ :=
  let main_meal_total := num_people * main_meal_cost
  let appetizer_total := num_appetizers * appetizer_cost
  let subtotal := main_meal_total + appetizer_total
  let tip := tip_rate * subtotal
  subtotal + tip + rush_order_fee

theorem bret_spends_77_dollars :
  total_cost num_people main_meal_cost num_appetizers appetizer_cost tip_rate rush_order_fee = 77.0 :=
by
  sorry

end bret_spends_77_dollars_l148_148618


namespace binary_101_to_decimal_l148_148640

-- Defining the question and translating it to Lean definitions
def binary_to_decimal (b : list ℕ) : ℕ :=
  b.reverse.enum.foldr (λ ⟨i, bit⟩ acc, acc + bit * (2 ^ i)) 0

def binary_101 := [1, 0, 1]

-- Stating the theorem to be proved
theorem binary_101_to_decimal :
  binary_to_decimal binary_101 = 5 :=
sorry

end binary_101_to_decimal_l148_148640


namespace count_orders_l148_148240
noncomputable def count_valid_orders (n : ℕ) : ℕ :=
  factorial (n / 2) * factorial ((n + 1) / 2)

theorem count_orders (n : ℕ) : 
  ∃! count_valid_orders n, (∀ (k : ℕ) (a_k : fin n → fin n), 
  (1 ≤ k ≤ n) → 
  (a_k.val + k) % 2 = 0) → 
  ∃! (orderings : set (list (fin n))), 
  orderings.count = count_valid_orders n :=
sorry

end count_orders_l148_148240


namespace part1_part2_part3_max_part3_min_l148_148097

noncomputable def f : ℝ → ℝ := sorry

-- Given Conditions
axiom f_add (x y : ℝ) : f (x + y) = f x + f y
axiom f_neg (x : ℝ) : x > 0 → f x < 0
axiom f_one : f 1 = -2

-- Prove that f(0) = 0
theorem part1 : f 0 = 0 := sorry

-- Prove that f(x) is an odd function
theorem part2 : ∀ x : ℝ, f (-x) = -f x := sorry

-- Prove the maximum and minimum values of f(x) on [-3,3]
theorem part3_max : f (-3) = 6 := sorry
theorem part3_min : f 3 = -6 := sorry

end part1_part2_part3_max_part3_min_l148_148097


namespace regular_hexagon_interior_angle_deg_l148_148851

theorem regular_hexagon_interior_angle_deg (n : ℕ) (h1 : n = 6) :
  let sum_of_interior_angles : ℕ := (n - 2) * 180
  let each_angle : ℕ := sum_of_interior_angles / n
  each_angle = 120 := by
  sorry

end regular_hexagon_interior_angle_deg_l148_148851


namespace chocolate_division_l148_148023

theorem chocolate_division (total_chocolate : ℚ) (piles : ℚ) (num_piles : ℚ) (brother_piles : ℚ) :
  total_chocolate = 75 / 7 →
  piles = total_chocolate / 5 →
  brother_piles = 2 * piles →
  brother_piles = 30 / 7 :=
begin
  intros,
  sorry
end

end chocolate_division_l148_148023


namespace arithmetic_sequence_divisibility_l148_148658

theorem arithmetic_sequence_divisibility (a d : ℕ) (h: ∀ n : ℕ, (∏ i in Finset.range n, (a + i * d)) ∣ (∏ i in Finset.range n, (a + (n + i) * d))) :
  ∃ k : ℕ, ∀ n : ℕ, a + n * d = k * (n + 1) :=
begin
  sorry
end

end arithmetic_sequence_divisibility_l148_148658


namespace total_valid_colorings_l148_148994

-- Define a function for the valid colorings
def valid_colorings (colors : Fin 9 → Fin 3) (edges : Finset (Fin 9 × Fin 9)) : Prop :=
  ∀ ⦃i j⦄, (i, j) ∈ edges → colors i ≠ colors j

-- Consider vertices of the three connected triangles
def vertices : Fin 9 := sorry -- Vertex positions in the nine-dot figure

-- Define edges of the three connected triangles
def edges : Finset (Fin 9 × Fin 9) := sorry -- Edges among the nine dots representing the figure

-- The sets of vertices for the three triangles
def first_triangle : Finset (Fin 9) := sorry
def second_triangle : Finset (Fin 9) := sorry
def third_triangle : Finset (Fin 9) := sorry

-- The total number of valid colorings
theorem total_valid_colorings (colors : Fin 9 → Fin 3) (h_valid : valid_colorings colors edges) : 
  ∃ n, n = 12 :=
by
  -- Here you would provide the proof steps, but we're skipping it with sorry
  sorry

end total_valid_colorings_l148_148994


namespace count_lines_passing_through_point_intersecting_parabola_at_one_point_l148_148100

noncomputable def point := (0, 2)
noncomputable def parabola (x y : ℝ) : Prop := y^2 = 8 * x

theorem count_lines_passing_through_point_intersecting_parabola_at_one_point :
  let lines := {line | ∃ k : ℝ, (λ x, k * x + 2 = line) ∨ (λ y, y = 2 = line) ∨ (λ x, x = 0 = line)} in 
  (∃! line ∈ lines, ∃ x y : ℝ, (line x y) ∧ (parabola x y)) ∧
  (card lines = 3) := sorry

end count_lines_passing_through_point_intersecting_parabola_at_one_point_l148_148100


namespace find_number_of_shorts_l148_148609

def price_of_shorts : ℕ := 7
def price_of_shoes : ℕ := 20
def total_spent : ℕ := 75

-- We represent the price of 4 tops as a variable
variable (T : ℕ)

theorem find_number_of_shorts (S : ℕ) (h : 7 * S + 4 * T + 20 = 75) : S = 7 :=
by
  sorry

end find_number_of_shorts_l148_148609


namespace min_value_and_points_on_curve_l148_148730

-- Define the parametric equation of line L
def line_param (t : ℝ) : ℝ × ℝ := (1 + t, 2 + sqrt 3 * t)

-- Define the Cartesian equation of line L
def line_eq_lhs (x y : ℝ) : ℝ := sqrt 3 * x - y - sqrt 3 + 2

-- Define the Cartesian equation of curve C
def curve_C_eq (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define the transformed equation of curve C'
def curve_C'_eq (x y : ℝ) : Prop := (x^2 / 4) + y^2 = 1

-- Define the quadratic form whose minimum we seek on curve C'
def quadratic_form (x y : ℝ) : ℝ := x^2 - sqrt 3 * x * y + 2 * y^2

-- Define the points we need to prove where the minimum occurs
def point_M1 : ℝ × ℝ := (1, sqrt 3 / 2)
def point_M2 : ℝ × ℝ := (-1, -sqrt 3 / 2)

theorem min_value_and_points_on_curve : 
  (∃ x y, curve_C'_eq x y ∧ quadratic_form x y = 1 ∧ (x, y) = point_M1 ∨ (x, y) = point_M2) ∧
  ∀ x y, curve_C'_eq x y → quadratic_form x y ≥ 1 :=
by
  sorry

end min_value_and_points_on_curve_l148_148730


namespace max_value_2cosx_3sinx_l148_148642

open Real 

theorem max_value_2cosx_3sinx : ∀ x : ℝ, 2 * cos x + 3 * sin x ≤ sqrt 13 :=
by sorry

end max_value_2cosx_3sinx_l148_148642


namespace james_speed_downhill_l148_148191

theorem james_speed_downhill (T1 T2 v : ℝ) (h1 : T1 = 20 / v) (h2 : T2 = 12 / 3 + 1) (h3 : T1 = T2 - 1) : v = 5 :=
by
  -- Declare variables
  have hT2 : T2 = 5 := by linarith
  have hT1 : T1 = 4 := by linarith
  have hv : v = 20 / 4 := by sorry
  linarith

#exit

end james_speed_downhill_l148_148191


namespace sum_of_base_7_digits_of_999_l148_148311

theorem sum_of_base_7_digits_of_999 : 
  let n := 999 
  let base := 7 
  let base7_rep := [2, 6, 2, 5]
  list.sum base7_rep = 15 :=
by
  let n := 999
  let base := 7
  let base7_rep := [2, 6, 2, 5]
  have h : list.sum base7_rep = 2 + 6 + 2 + 5 := sorry
  have sum_is_15 : 2 + 6 + 2 + 5 = 15 := sorry
  show list.sum base7_rep = 15, from h.trans sum_is_15

end sum_of_base_7_digits_of_999_l148_148311


namespace table_tennis_championship_max_winners_l148_148559

theorem table_tennis_championship_max_winners 
  (n : ℕ) 
  (knockout_tournament : ∀ (players : ℕ), players = 200) 
  (eliminations_per_match : ℕ := 1) 
  (matches_needed_to_win_3_times : ℕ := 3) : 
  ∃ k : ℕ, k = 66 ∧ k * matches_needed_to_win_3_times ≤ (n - 1) :=
begin
  sorry
end

end table_tennis_championship_max_winners_l148_148559


namespace weight_of_pants_l148_148887

def weight_socks := 2
def weight_underwear := 4
def weight_shirt := 5
def weight_shorts := 8
def total_allowed := 50

def weight_total (num_shirts num_shorts num_socks num_underwear : Nat) :=
  num_shirts * weight_shirt + num_shorts * weight_shorts + num_socks * weight_socks + num_underwear * weight_underwear

def items_in_wash := weight_total 2 1 3 4

theorem weight_of_pants :
  let weight_pants := total_allowed - items_in_wash
  weight_pants = 10 :=
by
  sorry

end weight_of_pants_l148_148887


namespace width_of_first_tv_is_24_l148_148815

-- Define the conditions
def height_first_tv := 16
def cost_first_tv := 672
def width_new_tv := 48
def height_new_tv := 32
def cost_new_tv := 1152
def cost_per_sq_inch_diff := 1

-- Define the width of the first TV
def width_first_tv := 24

-- Define the areas
def area_first_tv (W : ℕ) := W * height_first_tv
def area_new_tv := width_new_tv * height_new_tv

-- Define the cost per square inch
def cost_per_sq_inch_first_tv (W : ℕ) := cost_first_tv / area_first_tv W
def cost_per_sq_inch_new_tv := cost_new_tv / area_new_tv

-- The proof statement
theorem width_of_first_tv_is_24 :
  cost_per_sq_inch_first_tv width_first_tv = cost_per_sq_inch_new_tv + cost_per_sq_inch_diff
  := by
    unfold cost_per_sq_inch_first_tv
    unfold area_first_tv
    unfold cost_per_sq_inch_new_tv
    unfold area_new_tv
    sorry -- proof to be filled in

end width_of_first_tv_is_24_l148_148815


namespace hexagon_perimeter_l148_148858

def side_length : ℕ := 10
def num_sides : ℕ := 6

theorem hexagon_perimeter : num_sides * side_length = 60 := by
  sorry

end hexagon_perimeter_l148_148858


namespace probability_heads_at_most_3_l148_148530

theorem probability_heads_at_most_3 (num_coins : ℕ) (num_heads_at_most : ℕ) (probability : ℚ):
  num_coins = 10 → 
  num_heads_at_most = 3 → 
  probability = (binomial 10 0 + binomial 10 1 + binomial 10 2 + binomial 10 3) / (2^10) →
  probability = (11 / 64) := 
by
  intros hnum_coins hnum_heads_at_most hprobability
  sorry

end probability_heads_at_most_3_l148_148530


namespace range_of_m_l148_148731

open Real

def p (x : ℝ) : Prop := x^2 - 3 * x - 10 > 0
def q (x m : ℝ) : Prop := x > m^2 - m + 3
def neg_p (x : ℝ) : Prop := ¬ p x
def neg_q (x m : ℝ) : Prop := ¬ q x m

theorem range_of_m (m : ℝ) : (∀ x : ℝ, neg_p x ↔ neg_q x m) → m ∈ Iic (-1) ∪ Ici 2 :=
by
  sorry

end range_of_m_l148_148731


namespace power_function_value_at_minus_two_l148_148715

-- Define the power function assumption and points
variable (f : ℝ → ℝ)
variable (hf : f (1 / 2) = 8)

-- Prove that the given condition implies the required result
theorem power_function_value_at_minus_two : f (-2) = -1 / 8 := 
by {
  -- proof to be filled here
  sorry
}

end power_function_value_at_minus_two_l148_148715


namespace johns_exam_l148_148196

theorem johns_exam (total_problems mechanics_problems thermodynamics_problems optics_problems: ℕ)
  (mechanics_percent thermodynamics_percent optics_percent pass_percent: ℝ)
  (correct_mechanics correct_thermodynamics correct_optics correctly_answered passing_grade: ℕ) :

  (total_problems = 90) ∧
  (mechanics_problems = 20) ∧ 
  (thermodynamics_problems = 40) ∧
  (optics_problems = 30) ∧ 
  (mechanics_percent = 0.8) ∧ 
  (thermodynamics_percent = 0.5) ∧ 
  (optics_percent = 0.7) ∧ 
  (pass_percent = 0.65) ∧ 
  (correct_mechanics = (mechanics_percent * mechanics_problems : ℝ)) ∧
  (correct_thermodynamics = (thermodynamics_percent * thermodynamics_problems : ℝ)) ∧
  (correct_optics = (optics_percent * optics_problems : ℝ)) ∧ 
  (correctly_answered = correct_mechanics + correct_thermodynamics + correct_optics) ∧
  (passing_grade = (pass_percent * total_problems : ℝ).ceil.toNat) →
  passing_grade - correctly_answered = 2 := sorry

end johns_exam_l148_148196


namespace sum_sequence_formula_l148_148020

noncomputable def sequence_term (n : ℕ) : ℕ :=
  8 * 10^n + (10^n - 1) / 9 + 9 * (10^(n-1) - 1) / 9 * 10^(n-1)

noncomputable def sum_sequence (n : ℕ) : ℕ :=
  ∑ i in range n, sequence_term (i + 1)

theorem sum_sequence_formula (n : ℕ) : sum_sequence n = 10^(n+1) - 9n - 10 :=
begin
  sorry
end

end sum_sequence_formula_l148_148020


namespace rhombus_count_l148_148773

-- Define the grid and its properties.
def small_triangle_grid : Type := { t : ℕ // t = 25 }

-- Define the concept of rhombuses formed by two adjacent triangles.
def rhombuses (g : small_triangle_grid) : ℕ :=
  let triangles := g.val in  -- Number of small triangles in the grid
  let internal_edges := 30 in  -- Total count of internal edges forming rhombuses
  internal_edges

-- The main statement: Prove that the number of rhombuses formed is 30.
theorem rhombus_count (g : small_triangle_grid) : rhombuses g = 30 :=
by
  sorry

end rhombus_count_l148_148773


namespace seq_a3_eq_1_l148_148871

theorem seq_a3_eq_1 (a : ℕ → ℤ) (h₁ : ∀ n ≥ 1, a (n + 1) = a n - 3) (h₂ : a 1 = 7) : a 3 = 1 :=
by
  sorry

end seq_a3_eq_1_l148_148871


namespace smaller_circle_radius_l148_148934

theorem smaller_circle_radius
  (A1 A2 : ℝ)
  (h1 : ∀ r, A1 = π * r^2)
  (h2 : (A1 + A2) = π * 5^2)
  (h3 : ∃ x, A1 = x ∧ A2 = (25 * π - x))
  (h4 : 2 * x = A1 + 25 * π - A1) :
  ∃ r, A1 = π * r^2 ∧ r = 5 * real.sqrt(3) / 3 :=
by
  sorry

end smaller_circle_radius_l148_148934


namespace plane_equation_l148_148024

theorem plane_equation (x y z : ℝ)
  (h₁ : ∃ t : ℝ, x = 2 * t + 1 ∧ y = -3 * t ∧ z = 3 - t)
  (h₂ : ∃ (t₁ t₂ : ℝ), 4 * t₁ + 5 * t₂ - 3 = 0 ∧ 2 * t₁ + t₂ + 2 * t₂ = 0) : 
  2*x - y + 7*z - 23 = 0 :=
sorry

end plane_equation_l148_148024


namespace discriminant_pos_chord_length_12_min_chord_length_l148_148123

noncomputable def parabola (m : ℝ) (x : ℝ) : ℝ :=
  x^2 - (m^2 + 4) * x - 2 * m^2 - 12

theorem discriminant_pos (m : ℝ) : 
  let a := 1
  let b := -(m^2 + 4)
  let c := -2 * m^2 - 12
  (b^2 - 4 * a * c) > 0 :=
by
  have a_pos : 1 > 0 := by linarith
  have delta := ((m^2 + 4)^2 - 4 * 1 * (-2 * m^2 - 12))
  have delta_pos : delta = (m^2 + 8)^2 := by ring
  linarith [delta_pos]

theorem chord_length_12 (m : ℝ) : 
  let L := m^2 + 8
  L = 12 ↔ m = 2 ∨ m = -2 :=
by
  have L_def : L = m^2 + 8 := rfl
  rw L_def at *
  split
  { intro h
    have h := eq_sub_of_add_eq h
    have h₁ : m^2 = 4 := by linarith
    rw eq_comm at h₁
    exact eq_or_eq_neg_of_sq_eq_sq m.m 2 h₁ },
  { intro h
    cases h,
    { rw h, exact rfl },
    { rw h, exact rfl } }

theorem min_chord_length : 
  let L := m^2 + 8
  ∀ (m : ℝ), m = 0 → L = 8 :=
by
  intro m h
  have L_min := m^2 + 8
  rw h at L_min
  simp only [zero_sq] at L_min
  exact rfl

end discriminant_pos_chord_length_12_min_chord_length_l148_148123


namespace max_participants_win_at_least_three_matches_l148_148566

theorem max_participants_win_at_least_three_matches (n : ℕ) (h : n = 200) : 
  ∃ k : ℕ, k = 66 ∧ ∀ m : ℕ, (k * 3 ≤ m) ∧ (m ≤ 199) → k ≤ m / 3 := 
by
  sorry

end max_participants_win_at_least_three_matches_l148_148566


namespace probability_heads_at_most_three_out_of_ten_coins_l148_148557

theorem probability_heads_at_most_three_out_of_ten_coins :
  let total_outcomes := 2^10 in
  let favorable_outcomes := Nat.choose 10 0 + Nat.choose 10 1 + Nat.choose 10 2 + Nat.choose 10 3 in
  favorable_outcomes / total_outcomes = 11 / 64 :=
by
  sorry

end probability_heads_at_most_three_out_of_ten_coins_l148_148557


namespace max_area_of_sector_l148_148101

/-- Given that the perimeter of a sector of a circle is 30 centimeters,
    prove that the maximum area of the sector is 225/4 square centimeters. -/
theorem max_area_of_sector (l R : ℝ) (h : l + 2 * R = 30) :
  ∃ Rₘax, Rₘax = 15 / 2 ∧ (∃ Smax, Smax = 225 / 4 ∧ ∀ R', (l' : ℝ) (hl' : l' + 2 * R' = 30), 
  let S' := (1 / 2) * l' * R' in S' ≤ Smax) :=
sorry

end max_area_of_sector_l148_148101


namespace indistinguishable_balls_boxes_l148_148133

theorem indistinguishable_balls_boxes :
  ∃ p : Finset (Finset ℕ), 
  p.card = 9 ∧
  ∀ x ∈ p, 
    Multiset.sum x = 6 ∧
    Multiset.card x ≤ 4 :=
begin
  sorry
end

end indistinguishable_balls_boxes_l148_148133


namespace probability_of_at_most_3_heads_out_of_10_l148_148490
open_locale big_operators
open nat

theorem probability_of_at_most_3_heads_out_of_10 (n : ℕ) (p : ℕ) (h : 0 < n) (hp : p = 2) :
  ∑ k in finset.range (4), nat.choose n k = 176 → nat.pow p n = 1024 → (∑ k in finset.range (4), nat.choose n k) / (nat.pow p n) = 11 / 64 :=
by { intro h₁ h₂, 
     have : (∑ k in finset.range 4, nat.choose n k) / (nat.pow p n) = 176 / 1024,
     { rw [h₁, h₂] },
     rw this,
     norm_num }

end probability_of_at_most_3_heads_out_of_10_l148_148490


namespace num_trapezoids_in_20_sided_polygon_l148_148076

-- Given a regular 20-sided polygon, prove the number of sets of four vertices forming trapezoids.
theorem num_trapezoids_in_20_sided_polygon : 
  let n : ℕ := 20 in
  let regular_polygon (n : ℕ) := true in -- placeholder for the property of a regular n-sided polygon; can be refined
  ∃ (sets_of_four : {s : set (fin n) // s.card = 4}), 
  regular_polygon n → ∃! S : finset (set (fin n)), S.card = 720 := 
by {
  sorry 
}

end num_trapezoids_in_20_sided_polygon_l148_148076


namespace probability_10_coins_at_most_3_heads_l148_148398

def probability_at_most_3_heads (n : ℕ) : ℚ :=
  let total_outcomes := 2^n
  let favorable_outcomes := (Nat.choose n 0) + (Nat.choose n 1) + (Nat.choose n 2) + (Nat.choose n 3)
  favorable_outcomes / total_outcomes

theorem probability_10_coins_at_most_3_heads : probability_at_most_3_heads 10 = 11 / 64 :=
by
  sorry

end probability_10_coins_at_most_3_heads_l148_148398


namespace probability_at_most_3_heads_l148_148442

def binom (n k : ℕ) : ℕ := Nat.choose n k

theorem probability_at_most_3_heads :
  let total_outcomes := 2^10
  let successful_outcomes := binom 10 0 + binom 10 1 + binom 10 2 + binom 10 3
  (successful_outcomes : ℚ) / total_outcomes = 11 / 64 := 
by 
  let total_outcomes := 2 ^ 10
  let successful_outcomes := binom 10 0 + binom 10 1 + binom 10 2 + binom 10 3
  have h : (successful_outcomes : ℚ) / total_outcomes = 11 / 64 := sorry
  exact h

end probability_at_most_3_heads_l148_148442


namespace total_earnings_l148_148327

theorem total_earnings (x y : ℝ) (h : 20 * x * y = 18 * x * y + 150) : 
  18 * x * y + 20 * x * y + 20 * x * y = 4350 :=
by sorry

end total_earnings_l148_148327


namespace probability_heads_at_most_3_of_10_coins_flipped_l148_148376

theorem probability_heads_at_most_3_of_10_coins_flipped :
  (∑ k in finset.range 4, nat.choose 10 k) / 2^10 = 11 / 64 :=
sorry

end probability_heads_at_most_3_of_10_coins_flipped_l148_148376


namespace intersecting_line_circle_condition_l148_148153

theorem intersecting_line_circle_condition {a b : ℝ} (h : ∃ x y : ℝ, x^2 + y^2 = 1 ∧ x / a + y / b = 1) :
  (1 / a ^ 2) + (1 / b ^ 2) ≥ 1 :=
sorry

end intersecting_line_circle_condition_l148_148153


namespace probability_heads_at_most_three_out_of_ten_coins_l148_148555

theorem probability_heads_at_most_three_out_of_ten_coins :
  let total_outcomes := 2^10 in
  let favorable_outcomes := Nat.choose 10 0 + Nat.choose 10 1 + Nat.choose 10 2 + Nat.choose 10 3 in
  favorable_outcomes / total_outcomes = 11 / 64 :=
by
  sorry

end probability_heads_at_most_three_out_of_ten_coins_l148_148555


namespace probability_at_most_3_heads_l148_148445

def binom (n k : ℕ) : ℕ := Nat.choose n k

theorem probability_at_most_3_heads :
  let total_outcomes := 2^10
  let successful_outcomes := binom 10 0 + binom 10 1 + binom 10 2 + binom 10 3
  (successful_outcomes : ℚ) / total_outcomes = 11 / 64 := 
by 
  let total_outcomes := 2 ^ 10
  let successful_outcomes := binom 10 0 + binom 10 1 + binom 10 2 + binom 10 3
  have h : (successful_outcomes : ℚ) / total_outcomes = 11 / 64 := sorry
  exact h

end probability_at_most_3_heads_l148_148445


namespace greatest_k_for_200k_divides_100_factorial_l148_148916

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem greatest_k_for_200k_divides_100_factorial :
  let x := factorial 100
  let k_max := 12
  ∃ k : ℕ, y = 200 ^ k ∧ y ∣ x ∧ k = k_max :=
sorry

end greatest_k_for_200k_divides_100_factorial_l148_148916


namespace number_line_point_B_l148_148734

theorem number_line_point_B (A B : ℝ) (AB : ℝ) (h1 : AB = 4 * Real.sqrt 2) (h2 : A = 3 * Real.sqrt 2) :
  B = -Real.sqrt 2 ∨ B = 7 * Real.sqrt 2 :=
sorry

end number_line_point_B_l148_148734


namespace probability_at_most_3_heads_10_flips_l148_148518

theorem probability_at_most_3_heads_10_flips : 
  let num_heads_0 := nat.choose 10 0,
      num_heads_1 := nat.choose 10 1,
      num_heads_2 := nat.choose 10 2,
      num_heads_3 := nat.choose 10 3,
      total_outcomes := 2^10 in
  (num_heads_0 + num_heads_1 + num_heads_2 + num_heads_3) / total_outcomes = 11 / 64 :=
by
  sorry

end probability_at_most_3_heads_10_flips_l148_148518


namespace probability_at_most_3_heads_10_coins_l148_148424

noncomputable def coin_flip_prob_at_most_3_heads : ℚ :=
  let total_outcomes := 2^10
  let favorable_outcomes := Nat.binomial 10 0 + Nat.binomial 10 1 + Nat.binomial 10 2 + Nat.binomial 10 3
  favorable_outcomes / total_outcomes

theorem probability_at_most_3_heads_10_coins : coin_flip_prob_at_most_3_heads = 11 / 64 :=
by
  -- Definitions of let values are integrated directly in the theorem body
  let total_outcomes := 2^10
  let favorable_outcomes := Nat.binomial 10 0 + Nat.binomial 10 1 + Nat.binomial 10 2 + Nat.binomial 10 3
  show favorable_outcomes / total_outcomes = 11 / 64
  sorry

end probability_at_most_3_heads_10_coins_l148_148424


namespace cross_section_area_l148_148988

def a := Real  -- the side of the base

def BK : Real := (3 * a) / 4
def BD : Real := a / 2
def BQ : Real := a / 4
def BG : Real := a / 6

def S_ABC : Real := a^2 * Real.sqrt 3 / 4 -- Area of base triangle

-- Given area of the projection trapezoid
def S_np : Real := S_ABC * (3 / 4 * 1 / 2 - 1 / 4 * 1 / 6) / 3

def cos_alpha : Real := 1 / Real.sqrt 3

-- Final calculation of the area of the cross-section
def S_ce4 : Real := S_np / cos_alpha

theorem cross_section_area : S_ce4 = 14 := by
  sorry

end cross_section_area_l148_148988


namespace probability_of_at_most_3_heads_l148_148408

-- Definitions and conditions
def num_coins : ℕ := 10
def at_most_3_heads_probability : ℚ := 11 / 64

-- Statement of the problem
theorem probability_of_at_most_3_heads (n : ℕ) (p : ℚ) (h1 : n = num_coins) (h2 : p = at_most_3_heads_probability) :
  p = (1 + 10 + 45 + 120 : ℕ) / (2 ^ 10 : ℕ) := by
  sorry

end probability_of_at_most_3_heads_l148_148408


namespace probability_heads_at_most_3_l148_148535

theorem probability_heads_at_most_3 (num_coins : ℕ) (num_heads_at_most : ℕ) (probability : ℚ):
  num_coins = 10 → 
  num_heads_at_most = 3 → 
  probability = (binomial 10 0 + binomial 10 1 + binomial 10 2 + binomial 10 3) / (2^10) →
  probability = (11 / 64) := 
by
  intros hnum_coins hnum_heads_at_most hprobability
  sorry

end probability_heads_at_most_3_l148_148535


namespace sufficient_but_not_necessary_l148_148710

theorem sufficient_but_not_necessary (x : ℝ) : (x = -1 → x^2 - 5 * x - 6 = 0) ∧ (∃ y : ℝ, y ≠ -1 ∧ y^2 - 5 * y - 6 = 0) :=
by
  sorry

end sufficient_but_not_necessary_l148_148710


namespace infinite_points_in_any_circle_l148_148800

variable {Point : Type} [MetricSpace Point]

def H (P : Point) : Prop := sorry

lemma reflection_condition (P Q R S : Point) [MetricSpace Point] 
  (hP : H P) (hQ : H Q) (hR : H R) : H S :=
sorry

lemma epsilon_circle_exists (ε : ℝ) (h : ε > 0) : 
  ∃ C : Set Point, Metric.circular C ε ∧ (infinite {P : Point | P ∈ C ∧ H P ∧ ¬ collinear P}) :=
sorry

theorem infinite_points_in_any_circle (C : Set Point) (C_is_circle : Metric.circular C (radius_of C)) :
  infinite {P : Point | P ∈ C ∧ H P} :=
sorry

end infinite_points_in_any_circle_l148_148800


namespace football_team_gain_l148_148939

theorem football_team_gain (G : ℤ) :
  (-5 + G = 2) → (G = 7) :=
by
  intro h
  sorry

end football_team_gain_l148_148939


namespace nuts_in_boxes_l148_148882

theorem nuts_in_boxes (x y : ℕ) (H1 : 1.1 * x = 1.3 * y) (H2 : x = y + 80) :
  x = 520 ∧ (1.1 * x : ℝ) = 572 ∧ y = 440 :=
by {
  sorry
}

end nuts_in_boxes_l148_148882


namespace average_book_width_l148_148195

-- Define the widths of the books as given in the problem conditions
def widths : List ℝ := [3, 7.5, 1.25, 0.75, 4, 12]

-- Define the number of books from the problem conditions
def num_books : ℝ := 6

-- We prove that the average width of the books is equal to 4.75
theorem average_book_width : (widths.sum / num_books) = 4.75 :=
by
  sorry

end average_book_width_l148_148195


namespace total_hours_worked_l148_148002

theorem total_hours_worked (Amber_hours : ℕ) (h_Amber : Amber_hours = 12) 
  (Armand_hours : ℕ) (h_Armand : Armand_hours = Amber_hours / 3)
  (Ella_hours : ℕ) (h_Ella : Ella_hours = Amber_hours * 2) : 
  Amber_hours + Armand_hours + Ella_hours = 40 :=
by
  rw [h_Amber, h_Armand, h_Ella]
  norm_num
  sorry

end total_hours_worked_l148_148002


namespace function_properties_l148_148114

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := x^2 + a*x + b

theorem function_properties (a b : ℝ) (h : (a - 1) ^ 2 - 4 * b < 0) : 
  (∀ x : ℝ, f x a b > x) ∧ (∀ x : ℝ, f (f x a b) a b > x) ∧ (a + b > 0) :=
by
  sorry

end function_properties_l148_148114


namespace average_weight_l148_148186

theorem average_weight (Ishmael Ponce Jalen : ℝ) 
  (h1 : Ishmael = Ponce + 20) 
  (h2 : Ponce = Jalen - 10) 
  (h3 : Jalen = 160) : 
  (Ishmael + Ponce + Jalen) / 3 = 160 := 
by 
  sorry

end average_weight_l148_148186


namespace probability_heads_at_most_three_out_of_ten_coins_l148_148550

theorem probability_heads_at_most_three_out_of_ten_coins :
  let total_outcomes := 2^10 in
  let favorable_outcomes := Nat.choose 10 0 + Nat.choose 10 1 + Nat.choose 10 2 + Nat.choose 10 3 in
  favorable_outcomes / total_outcomes = 11 / 64 :=
by
  sorry

end probability_heads_at_most_three_out_of_ten_coins_l148_148550


namespace evaluate_expression_at_neg2_l148_148998

def expression (x : ℤ) : ℤ :=
  (3 + x * (3 + x) - 3^2) / (x - 3 + x^2)

theorem evaluate_expression_at_neg2 : expression (-2) = 8 :=
by
  have h1 : expression (-2) = (3 + (-2) * (3 + (-2)) - 9) / ((-2) - 3 + 4), by rfl
  have h2 : (3 + (-2) * (3 + (-2)) - 9) = -8, by norm_num
  have h3 : ((-2) - 3 + 4) = -1, by norm_num
  rw [h1, h2, h3]
  norm_num

end evaluate_expression_at_neg2_l148_148998


namespace find_first_day_speed_l148_148927

theorem find_first_day_speed (t : ℝ) (d : ℝ) (v : ℝ) (h1 : d = 2.5) 
  (h2 : v * (t - 7/60) = d) (h3 : 10 * (t - 8/60) = d) : v = 9.375 :=
by {
  -- Proof omitted for brevity
  sorry
}

end find_first_day_speed_l148_148927


namespace total_hours_worked_l148_148001

theorem total_hours_worked (Amber_hours : ℕ) (h_Amber : Amber_hours = 12) 
  (Armand_hours : ℕ) (h_Armand : Armand_hours = Amber_hours / 3)
  (Ella_hours : ℕ) (h_Ella : Ella_hours = Amber_hours * 2) : 
  Amber_hours + Armand_hours + Ella_hours = 40 :=
by
  rw [h_Amber, h_Armand, h_Ella]
  norm_num
  sorry

end total_hours_worked_l148_148001


namespace decimal_to_base8_conversion_l148_148176

-- Define the base and the number in decimal.
def base : ℕ := 8
def decimal_number : ℕ := 127

-- Define the expected representation in base 8.
def expected_base8_representation : ℕ := 177

-- Theorem stating that conversion of 127 in base 10 to base 8 yields 177
theorem decimal_to_base8_conversion : Nat.ofDigits base (Nat.digits base decimal_number) = expected_base8_representation := 
by
  sorry

end decimal_to_base8_conversion_l148_148176


namespace largest_possible_expression_value_l148_148751

-- Definition of the conditions.
def distinct_digits (X Y Z : ℕ) : Prop := X ≠ Y ∧ Y ≠ Z ∧ X ≠ Z ∧ X < 10 ∧ Y < 10 ∧ Z < 10

-- The main theorem statement.
theorem largest_possible_expression_value : ∀ (X Y Z : ℕ), distinct_digits X Y Z → 
  (100 * X + 10 * Y + Z - 10 * Z - Y - X) ≤ 900 :=
by
  sorry

end largest_possible_expression_value_l148_148751


namespace probability_heads_at_most_3_l148_148528

theorem probability_heads_at_most_3 (num_coins : ℕ) (num_heads_at_most : ℕ) (probability : ℚ):
  num_coins = 10 → 
  num_heads_at_most = 3 → 
  probability = (binomial 10 0 + binomial 10 1 + binomial 10 2 + binomial 10 3) / (2^10) →
  probability = (11 / 64) := 
by
  intros hnum_coins hnum_heads_at_most hprobability
  sorry

end probability_heads_at_most_3_l148_148528


namespace num_pos_four_digit_integers_l148_148130

theorem num_pos_four_digit_integers : 
  ∃ (n : ℕ), n = (Nat.factorial 4) / ((Nat.factorial 3) * (Nat.factorial 1)) ∧ n = 4 := 
by
  sorry

end num_pos_four_digit_integers_l148_148130


namespace no_tetrahedron_with_obtuse_angle_at_edge_l148_148787

-- Define vertices of the tetrahedron
variables {A B C D : Type}

-- Define an edge length ordering for a tetrahedron
def edge_lengths (tetrahedron : {A B C D}) : list ℝ := sorry

-- Define the property that an angle in a triangle is obtuse
def is_obtuse (angle : ℝ) : Prop :=
  angle > 90

-- Define the property of a tetrahedron such that each pair of adjacent triangular faces has an obtuse angle sharing an edge
def has_obtuse_angle_at_edge (tetrahedron : {A B C D}) (edge : (ℝ, ℝ)) : Prop := sorry

theorem no_tetrahedron_with_obtuse_angle_at_edge :
  ∀ (tetrahedron : {A B C D}),
    ¬ (∀ edge ∈ edge_lengths tetrahedron, has_obtuse_angle_at_edge tetrahedron edge) :=
by sorry

end no_tetrahedron_with_obtuse_angle_at_edge_l148_148787


namespace sum_geometric_series_l148_148245

theorem sum_geometric_series (x : ℂ) (h1 : x ^ 2023 - 3 * x + 2 = 0) (h2 : x ≠ 1) : 
  x ^ 2022 + x ^ 2021 + ... + x + 1 = 3 :=
sorry

end sum_geometric_series_l148_148245


namespace probability_heads_at_most_three_out_of_ten_coins_l148_148548

theorem probability_heads_at_most_three_out_of_ten_coins :
  let total_outcomes := 2^10 in
  let favorable_outcomes := Nat.choose 10 0 + Nat.choose 10 1 + Nat.choose 10 2 + Nat.choose 10 3 in
  favorable_outcomes / total_outcomes = 11 / 64 :=
by
  sorry

end probability_heads_at_most_three_out_of_ten_coins_l148_148548


namespace find_f_5_l148_148856

noncomputable def f : ℝ → ℝ :=
λ x, 2 * x^2 + x - 3 * (1 - x) -- This is just a placeholder; the real function would solve the equation

theorem find_f_5 (x : ℝ) (h : ∀ x, f x + 3 * f (1 - x) = 2 * x^2 + x) :
  f 5 = 29 / 8 :=
by
  sorry

end find_f_5_l148_148856


namespace construction_company_l148_148574

/-- A construction company has two engineering teams, Team A and Team B.
  Team A can improve 15 meters per day, and Team B can improve 10 meters per day.
  They took a total of 25 days to complete the task.
  The total length of the riverbank road completed is 300 meters.
  Team A's daily cost is 0.6 million yuan and Team B's daily cost is 0.8 million yuan.
  The total cost must be below 18 million yuan. -/
theorem construction_company
  (x y m : ℕ)
  (hx : x + y = 300)
  (ht : x / 15 + y / 10 = 25)
  (h_cost : 0.6 * m + 0.8 * (300 - 15 * m) / 10 ≤ 18) :
  x = 150 ∧ y = 150 ∧ m ≥ 10 :=
by
  sorry

end construction_company_l148_148574


namespace max_radius_of_sphere_l148_148784

-- Define key setup, points, and conditions.
def unit_cube_side_length : ℝ := 1

def diagonal_AC1 : ℝ × ℝ × ℝ := (1, 0, 0)

-- Define the function that represents the maximum radius of the sphere.
def max_radius_sphere_tangent_to_diagonal_cube (side_length : ℝ) : ℝ :=
  (4 - real.sqrt 6) / 5

-- The theorem stating the maximum radius of a sphere tangent to the diagonal AC1 of a cube with side length 1.
theorem max_radius_of_sphere (radius : ℝ) 
  (h1 : radius = max_radius_sphere_tangent_to_diagonal_cube unit_cube_side_length) :
  radius = (4 - real.sqrt 6) / 5 := by
  sorry

end max_radius_of_sphere_l148_148784


namespace solve_frac_eq_l148_148027

theorem solve_frac_eq (x : ℝ) (h : 3 - 5 / x + 2 / (x^2) = 0) : 
  ∃ y : ℝ, (y = 3 / x ∧ (y = 9 / 2 ∨ y = 3)) :=
sorry

end solve_frac_eq_l148_148027


namespace probability_at_most_3_heads_l148_148498

-- Definitions and conditions
def coin_flips : ℕ := 10
def success_prob : ℚ := 1/2
def max_heads : ℕ := 3

-- Binomial Probability Computation
noncomputable def probability_of_at_most_3_heads : ℚ :=
  (∑ k in finset.range (max_heads + 1), nat.choose coin_flips k) / (2^coin_flips)

-- The statement to prove
theorem probability_at_most_3_heads :
  probability_of_at_most_3_heads = 11/64 := by
  sorry

end probability_at_most_3_heads_l148_148498


namespace geometric_locus_l148_148339

noncomputable def locus_type (z z1 z2 : ℂ) (λ : ℝ) : String :=
  if λ = 1 then "straight line" else "circle"

theorem geometric_locus (z z1 z2 : ℂ) (λ : ℝ) (h : λ > 0) :
  (λ = 1 → locus_type z z1 z2 λ = "straight line") ∧ (λ ≠ 1 → locus_type z z1 z2 λ = "circle") :=
by
  split 
  { intro hλ
    rw [locus_type, if_pos hλ]
    refl }
  { intro hλ
    rw [locus_type, if_neg hλ]
    refl }
  sorry

end geometric_locus_l148_148339


namespace people_on_train_after_third_stop_l148_148285

variable (initial_people : ℕ) (off_1 boarded_1 off_2 boarded_2 off_3 boarded_3 : ℕ)

def people_after_first_stop (initial : ℕ) (off_1 boarded_1 : ℕ) : ℕ :=
  initial - off_1 + boarded_1

def people_after_second_stop (first_stop : ℕ) (off_2 boarded_2 : ℕ) : ℕ :=
  first_stop - off_2 + boarded_2

def people_after_third_stop (second_stop : ℕ) (off_3 boarded_3 : ℕ) : ℕ :=
  second_stop - off_3 + boarded_3

theorem people_on_train_after_third_stop :
  people_after_third_stop (people_after_second_stop (people_after_first_stop initial_people off_1 boarded_1) off_2 boarded_2) off_3 boarded_3 = 42 :=
  by
    have initial_people := 48
    have off_1 := 12
    have boarded_1 := 7
    have off_2 := 15
    have boarded_2 := 9
    have off_3 := 6
    have boarded_3 := 11
    sorry

end people_on_train_after_third_stop_l148_148285


namespace inequality_solution_l148_148842

theorem inequality_solution (x : ℝ) :
  (x ∈ set.Ioo (⊥ : ℝ) (-2) ∪ set.Ioo (-2) 3 ∪ set.Ioo 3 5) ↔
  ((x - 5) / ((x - 3) * (x + 2)) < 0) :=
by
  sorry

end inequality_solution_l148_148842


namespace probability_of_at_most_3_heads_l148_148413

-- Definitions and conditions
def num_coins : ℕ := 10
def at_most_3_heads_probability : ℚ := 11 / 64

-- Statement of the problem
theorem probability_of_at_most_3_heads (n : ℕ) (p : ℚ) (h1 : n = num_coins) (h2 : p = at_most_3_heads_probability) :
  p = (1 + 10 + 45 + 120 : ℕ) / (2 ^ 10 : ℕ) := by
  sorry

end probability_of_at_most_3_heads_l148_148413


namespace alex_wins_if_picks_two_l148_148595

theorem alex_wins_if_picks_two (matches_left : ℕ) (alex_picks bob_picks : ℕ) :
  matches_left = 30 →
  1 ≤ alex_picks ∧ alex_picks ≤ 6 →
  1 ≤ bob_picks ∧ bob_picks ≤ 6 →
  alex_picks = 2 →
  (∀ n, (n % 7 ≠ 0) → ¬ (∃ k, matches_left - k ≤ 0 ∧ (matches_left - k) % 7 = 0)) :=
by sorry

end alex_wins_if_picks_two_l148_148595


namespace artist_paintings_in_four_weeks_l148_148959

theorem artist_paintings_in_four_weeks (hours_per_week : ℕ) (hours_per_painting : ℕ) (weeks : ℕ) (total_paintings : ℕ) :
  hours_per_week = 30 → hours_per_painting = 3 → weeks = 4 → total_paintings = ((hours_per_week / hours_per_painting) * weeks) → total_paintings = 40 :=
by
  intros h_week h_painting h_weeks h_total
  rw [h_week, h_painting, h_weeks]
  norm_num
  exact h_total

end artist_paintings_in_four_weeks_l148_148959


namespace probability_of_at_most_3_heads_l148_148402

-- Definitions and conditions
def num_coins : ℕ := 10
def at_most_3_heads_probability : ℚ := 11 / 64

-- Statement of the problem
theorem probability_of_at_most_3_heads (n : ℕ) (p : ℚ) (h1 : n = num_coins) (h2 : p = at_most_3_heads_probability) :
  p = (1 + 10 + 45 + 120 : ℕ) / (2 ^ 10 : ℕ) := by
  sorry

end probability_of_at_most_3_heads_l148_148402


namespace find_m_n_l148_148135

-- Given definition of like terms
def like_terms (a b : ℝ) (n1 n2 m1 m2 : ℤ) : Prop :=
  n1 = n2 ∧ m1 = m2

-- Variables
variables {x y : ℝ} {m n : ℤ}

-- Definitions based on problem conditions
def expr1 : ℝ := 2 * x^(n + 2) * y^3
def expr2 : ℝ := -3 * x^3 * y^(2 * m - 1)

-- Proof problem
theorem find_m_n (h : like_terms expr1 expr2 (n + 2) 3 3 (2 * m - 1)) : m = 2 ∧ n = 1 :=
  sorry

end find_m_n_l148_148135


namespace affine_transformation_representation_l148_148228

-- Define transformations and properties
variable (Point : Type) [AffineSpace Point]
variable (Vector : Type) [AddCommGroup Vector] [Module ℝ Vector]

variable (L : Point → Point)
variable (O A : Point)
variable (T : Point → Point)
variable (H : Point → Point)

-- Define affine transformation and dilation (compression)
def affine_transformation (L : Point → Point) := 
  ∀ O : Point, ∃ T : Point → Point, ∃ H : Point → Point, 
    (T ∘ L = L) ∧ (H ∘ T ∘ L = H ∘ L) ∧
    ∀ triangle : Point × Point × Point, 
      is_similar_triangle (H ∘ T ∘ L <$> triangle)

-- The theorem statement
theorem affine_transformation_representation :
  affine_transformation L :=
by
  -- Definitions and setup
  -- T is the translation by the vector \(\overrightarrow{L(O)O}\)
  sorry

end affine_transformation_representation_l148_148228


namespace domain_of_f_l148_148868

noncomputable def f (x : ℝ) : ℝ := (11 / (sqrt (3 - x)) + sqrt (abs x))

theorem domain_of_f (x : ℝ) : x < 3 ↔ ∃ y : ℝ, y = (11 / (sqrt (3 - x)) + sqrt (abs x)) :=
by
  -- Conditions that make the function meaningful
  -- (3 - x > 0)
  have h1 : 3 - x > 0 ↔ x < 3 := by
    sorry,
  -- (|x| >= 0) is always true
  have h2 : |x| ≥ 0 := by
    sorry,
  sorry

end domain_of_f_l148_148868


namespace x_cubed_inverse_cubed_l148_148711

theorem x_cubed_inverse_cubed (x : ℝ) (hx : x + 1/x = 3) : x^3 + 1/x^3 = 18 :=
by
  sorry

end x_cubed_inverse_cubed_l148_148711


namespace probability_heads_at_most_3_of_10_coins_flipped_l148_148381

theorem probability_heads_at_most_3_of_10_coins_flipped :
  (∑ k in finset.range 4, nat.choose 10 k) / 2^10 = 11 / 64 :=
sorry

end probability_heads_at_most_3_of_10_coins_flipped_l148_148381


namespace circle_radius_9_l148_148679

theorem circle_radius_9 (k : ℝ) : 
  (∀ x y : ℝ, 2 * x^2 + 20 * x + 3 * y^2 + 18 * y - k = 81) → 
  (k = 94) :=
by
  sorry

end circle_radius_9_l148_148679


namespace spinner_divisibility_by_3_probability_l148_148995

-- Definitions based on the conditions
def outcomes := {1, 2, 4}

def is_divisible_by_3 (n : ℕ) : Prop := n % 3 = 0

-- Main statement to be proven
theorem spinner_divisibility_by_3_probability :
  let numbers := {a * 100 + b * 10 + c | a b c : ℕ, a ∈ outcomes, b ∈ outcomes, c ∈ outcomes} in
  let div_by_3_count := {n ∈ numbers | is_divisible_by_3 (n % 1000)}.card in
  div_by_3_count = 6 / 27 ∧ (6 / 27) = (2 / 9) :=
begin
  sorry,
end

end spinner_divisibility_by_3_probability_l148_148995


namespace passengers_landed_on_time_l148_148197

theorem passengers_landed_on_time (total_passengers : ℕ) (passengers_late : ℕ) :
  total_passengers = 14720 ∧ passengers_late = 213 → total_passengers - passengers_late = 14507 :=
by
  intro h
  obtain ⟨ht, hl⟩ := h
  rw [ht, hl]
  sorry

end passengers_landed_on_time_l148_148197


namespace max_participants_win_at_least_three_matches_l148_148565

theorem max_participants_win_at_least_three_matches (n : ℕ) (h : n = 200) : 
  ∃ k : ℕ, k = 66 ∧ ∀ m : ℕ, (k * 3 ≤ m) ∧ (m ≤ 199) → k ≤ m / 3 := 
by
  sorry

end max_participants_win_at_least_three_matches_l148_148565


namespace slope_of_line_determined_by_solutions_l148_148302

theorem slope_of_line_determined_by_solutions :
  (∀ (x₁ y₁ x₂ y₂ : ℝ), (4 / x₁ + 6 / y₁ = 0) ∧ (4 / x₂ + 6 / y₂ = 0) →
    (y₂ - y₁) / (x₂ - x₁) = -3 / 2) :=
sorry

end slope_of_line_determined_by_solutions_l148_148302


namespace area_of_triangle_AQB_l148_148951

noncomputable def side_length := 8
noncomputable def point_Q := (4, 4)
noncomputable def point_A := (0, 0)
noncomputable def point_B := (8, 0)
noncomputable def point_C := (4, 2)
noncomputable def point_D := (8, 8)
noncomputable def point_F := (0, 8)

-- Definitions from the problem
def qa_length := real.sqrt ((point_Q.1 - point_A.1)^2 + (point_Q.2 - point_A.2)^2)
def qb_length := real.sqrt ((point_Q.1 - point_B.1)^2 + (point_Q.2 - point_B.2)^2)
def qc_length := real.sqrt ((point_Q.1 - point_C.1)^2 + (point_Q.2 - point_C.2)^2)

-- Theorem statement translating the problem to Lean
theorem area_of_triangle_AQB : qa_length = qb_length ∧ qb_length = qc_length ∧ qc_length = qa_length →
  (point_C.2 - point_F.2) * (point_C.2 - point_A.2) = 0 →
  real.sqrt ((point_A.1 - point_B.1)^2 + (3 : real)^2) / 2 * (real.sqrt ((point_A.1 - point_B.1)^2 + (3 : real)^2)) = 12 := 
by
  sorry

end area_of_triangle_AQB_l148_148951


namespace square_of_1031_l148_148983

theorem square_of_1031 : 1031 ^ 2 = 1060961 := by
  calc
    1031 ^ 2 = (1000 + 31) ^ 2       : by sorry
           ... = 1000 ^ 2 + 2 * 1000 * 31 + 31 ^ 2 : by sorry
           ... = 1000000 + 62000 + 961 : by sorry
           ... = 1060961 : by sorry

end square_of_1031_l148_148983


namespace number_of_white_jelly_beans_l148_148791

def number_of_red_per_bag : ℕ := 24
def total_red_and_white_guess : ℕ := 126
def number_of_bags : ℕ := 3

theorem number_of_white_jelly_beans :
  let total_red_jelly_beans := number_of_red_per_bag * number_of_bags in
  let number_of_white_jelly_beans := total_red_and_white_guess - total_red_jelly_beans in
  number_of_white_jelly_beans = 54 :=
by
  sorry

end number_of_white_jelly_beans_l148_148791


namespace reflection_of_v_over_u_l148_148675

def v : ℝ × ℝ :=
⟨2, 5⟩

def u : ℝ × ℝ :=
⟨3, 1⟩

def normalize (x : ℝ × ℝ) : ℝ × ℝ :=
let norm := real.sqrt (x.1 * x.1 + x.2 * x.2)
in (x.1 / norm, x.2 / norm)

def projection (a b : ℝ × ℝ) : ℝ × ℝ :=
let dot_product := a.1 * b.1 + a.2 * b.2
let norm := b.1 * b.1 + b.2 * b.2
let scalar := dot_product / norm
in (scalar * b.1, scalar * b.2)

def reflection (a b : ℝ × ℝ) : ℝ × ℝ :=
let p := projection a (normalize b)
in (2 * p.1 - a.1, 2 * p.2 - a.2)

theorem reflection_of_v_over_u :
  reflection v u = ⟨4.6, -2.8⟩ :=
by
  sorry

end reflection_of_v_over_u_l148_148675


namespace count_rational_numbers_l148_148696

theorem count_rational_numbers :
  let n := 20.factorial in
  {f : ℚ // 0 < f ∧ f < 1 ∧ ∃ (a b : ℕ), f = a / b ∧ a * b = n ∧ Nat.gcd a b = 1}.to_finset.card = 128 := 
by
  sorry

end count_rational_numbers_l148_148696


namespace hyperbola_eccentricity_l148_148093

theorem hyperbola_eccentricity (F B : Point) (a b c e : ℝ)
  (h1 : FB_perpendicular_asymptote F B)
  (h2 : equation_perpendicular_condition a b c)
  : e = (1 + sqrt 5) / 2 := 
  sorry

end hyperbola_eccentricity_l148_148093


namespace average_weight_l148_148188

theorem average_weight (Ishmael Ponce Jalen : ℝ) 
  (h1 : Ishmael = Ponce + 20) 
  (h2 : Ponce = Jalen - 10) 
  (h3 : Jalen = 160) : 
  (Ishmael + Ponce + Jalen) / 3 = 160 := 
by 
  sorry

end average_weight_l148_148188


namespace rectangle_perimeter_l148_148774

-- Definitions and assumptions
variables (outer_square_area inner_square_area : ℝ) (rectangles_identical : Prop)

-- Given conditions
def outer_square_area_condition : Prop := outer_square_area = 9
def inner_square_area_condition : Prop := inner_square_area = 1
def rectangles_identical_condition : Prop := rectangles_identical

-- The main theorem to prove
theorem rectangle_perimeter (h_outer : outer_square_area_condition outer_square_area)
                            (h_inner : inner_square_area_condition inner_square_area)
                            (h_rectangles : rectangles_identical_condition rectangles_identical) :
  ∃ perimeter : ℝ, perimeter = 6 :=
by
  sorry

end rectangle_perimeter_l148_148774


namespace probability_at_most_3_heads_l148_148433

def binom (n k : ℕ) : ℕ := Nat.choose n k

theorem probability_at_most_3_heads :
  let total_outcomes := 2^10
  let successful_outcomes := binom 10 0 + binom 10 1 + binom 10 2 + binom 10 3
  (successful_outcomes : ℚ) / total_outcomes = 11 / 64 := 
by 
  let total_outcomes := 2 ^ 10
  let successful_outcomes := binom 10 0 + binom 10 1 + binom 10 2 + binom 10 3
  have h : (successful_outcomes : ℚ) / total_outcomes = 11 / 64 := sorry
  exact h

end probability_at_most_3_heads_l148_148433


namespace lcm_5_6_10_18_l148_148299

theorem lcm_5_6_10_18 : Nat.lcm (5 :: 6 :: 10 :: 18 :: []) = 90 := 
by
  sorry

end lcm_5_6_10_18_l148_148299


namespace could_be_green_l148_148968

noncomputable def traffic_light_color (time : ℕ) : Type :=
  if time % 60 < 30 then "red"
  else if time % 60 < 55 then "green"
  else "yellow"

theorem could_be_green : ∃ t : ℕ, traffic_light_color t = "green" := by
  intros
  use 30
  simp [traffic_light_color]
  sorry

end could_be_green_l148_148968


namespace average_weight_of_three_l148_148183

theorem average_weight_of_three (Ishmael Ponce Jalen : ℕ) 
  (h1 : Jalen = 160) 
  (h2 : Ponce = Jalen - 10) 
  (h3 : Ishmael = Ponce + 20) : 
  (Ishmael + Ponce + Jalen) / 3 = 160 := 
sorry

end average_weight_of_three_l148_148183


namespace probability_10_coins_at_most_3_heads_l148_148390

def probability_at_most_3_heads (n : ℕ) : ℚ :=
  let total_outcomes := 2^n
  let favorable_outcomes := (Nat.choose n 0) + (Nat.choose n 1) + (Nat.choose n 2) + (Nat.choose n 3)
  favorable_outcomes / total_outcomes

theorem probability_10_coins_at_most_3_heads : probability_at_most_3_heads 10 = 11 / 64 :=
by
  sorry

end probability_10_coins_at_most_3_heads_l148_148390


namespace number_and_sum_of_g_25_l148_148219

noncomputable def g : ℕ → ℕ := sorry

axiom g_condition : ∀ (a b : ℕ), 2 * g (a^2 + b^2) = (g a + 1)^2 + (g b + 1)^2

theorem number_and_sum_of_g_25 :
  let n := 1 in
  let s := 18.5 in
  n * s = 18.5 := by
  sorry

end number_and_sum_of_g_25_l148_148219


namespace mixed_bag_total_weight_l148_148631

variable (P : ℝ) -- The weight of the Peruvian beans.

-- Assumptions based on the conditions.
def colombian_cost_per_pound : ℝ := 5.50
def peruvian_cost_per_pound : ℝ := 4.25
def mixed_cost_per_pound : ℝ := 4.60
def colombian_weight : ℝ := 28.8

-- The equation reflecting the given problem conditions.
axiom mixed_cost_equation :
  colombian_cost_per_pound * colombian_weight + peruvian_cost_per_pound * P = mixed_cost_per_pound * (colombian_weight + P)

-- The proof problem is to show that if we solve for P and add it to the colombian_weight, we get 102.8.
theorem mixed_bag_total_weight :
  P = 74 → colombian_weight + P = 102.8 :=
by {
  intro h,
  rw h,
  exact (colombian_weight + 74) = 102.8, -- Assuming this step as a given for now
  sorry
}

end mixed_bag_total_weight_l148_148631


namespace slope_of_line_determined_by_solutions_l148_148301

theorem slope_of_line_determined_by_solutions :
  (∀ (x₁ y₁ x₂ y₂ : ℝ), (4 / x₁ + 6 / y₁ = 0) ∧ (4 / x₂ + 6 / y₂ = 0) →
    (y₂ - y₁) / (x₂ - x₁) = -3 / 2) :=
sorry

end slope_of_line_determined_by_solutions_l148_148301


namespace tangential_quadrilateral_exists_l148_148996

theorem tangential_quadrilateral_exists
  (A B C D : Point)
  (convex_quad : ConvexQuadrilateral A B C D)
  (circle_omega : Circle)
  (intersects_sides : ∀ side ∈ sides_of_quadrilateral A B C D, ∃ p1 p2 ∈ circle_omega, equal_chord p1 p2)
  (equal_chords : ∀ side ∈ sides_of_quadrilateral A B C D, 
    ∃ chord1 chord2, chord1.length = chord2.length ∧ chord1 ∈ circle_omega ∧ chord2 ∈ circle_omega) :
  ∃ inscribed_circle : Circle, inscribed_circle < side_of_quadrilateral A B C D :=
sorry

end tangential_quadrilateral_exists_l148_148996


namespace probability_at_most_3_heads_10_coins_l148_148430

noncomputable def coin_flip_prob_at_most_3_heads : ℚ :=
  let total_outcomes := 2^10
  let favorable_outcomes := Nat.binomial 10 0 + Nat.binomial 10 1 + Nat.binomial 10 2 + Nat.binomial 10 3
  favorable_outcomes / total_outcomes

theorem probability_at_most_3_heads_10_coins : coin_flip_prob_at_most_3_heads = 11 / 64 :=
by
  -- Definitions of let values are integrated directly in the theorem body
  let total_outcomes := 2^10
  let favorable_outcomes := Nat.binomial 10 0 + Nat.binomial 10 1 + Nat.binomial 10 2 + Nat.binomial 10 3
  show favorable_outcomes / total_outcomes = 11 / 64
  sorry

end probability_at_most_3_heads_10_coins_l148_148430


namespace parallel_vectors_l148_148686

variable (x : ℝ)

def a := (1, 2)
def b := (2 * x, -3)
def is_parallel (u v : ℝ × ℝ) : Prop :=
  u.1 * v.2 = u.2 * v.1

theorem parallel_vectors (h : is_parallel a b) : x = -3 / 4 :=
by
  sorry

end parallel_vectors_l148_148686


namespace determine_a_range_l148_148728

noncomputable def f (a x : ℝ) : ℝ := |real.exp x + a / real.exp x|

theorem determine_a_range (a : ℝ) :
  (∀ x ∈ Icc (0:ℝ) (1:ℝ), (differentiable ℝ (f a x)) → ∀ (h : differentiable ℝ (f a x)), (fderiv ℝ (f a) x) ≥ 0) ↔ a ∈ Icc (-1:ℝ) (1:ℝ) :=
sorry

end determine_a_range_l148_148728


namespace meaningful_fraction_l148_148256

theorem meaningful_fraction (x : ℝ) : (x + 1 ≠ 0) ↔ (x ≠ -1) :=
by
  sorry

end meaningful_fraction_l148_148256


namespace probability_10_coins_at_most_3_heads_l148_148384

def probability_at_most_3_heads (n : ℕ) : ℚ :=
  let total_outcomes := 2^n
  let favorable_outcomes := (Nat.choose n 0) + (Nat.choose n 1) + (Nat.choose n 2) + (Nat.choose n 3)
  favorable_outcomes / total_outcomes

theorem probability_10_coins_at_most_3_heads : probability_at_most_3_heads 10 = 11 / 64 :=
by
  sorry

end probability_10_coins_at_most_3_heads_l148_148384


namespace minimum_value_x_plus_4y_l148_148694

variable {x y : ℝ}

-- Define the conditions
axiom hx : x > 0
axiom hy : y > 0
axiom hxy : x + y = 2 * x * y

-- Define the proof statement
theorem minimum_value_x_plus_4y (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x + y = 2 * x * y) : x + 4 * y = 9 / 2 := sorry

end minimum_value_x_plus_4y_l148_148694


namespace probability_at_most_3_heads_l148_148510

-- Definitions and conditions
def coin_flips : ℕ := 10
def success_prob : ℚ := 1/2
def max_heads : ℕ := 3

-- Binomial Probability Computation
noncomputable def probability_of_at_most_3_heads : ℚ :=
  (∑ k in finset.range (max_heads + 1), nat.choose coin_flips k) / (2^coin_flips)

-- The statement to prove
theorem probability_at_most_3_heads :
  probability_of_at_most_3_heads = 11/64 := by
  sorry

end probability_at_most_3_heads_l148_148510


namespace divisibility_of_sum_l148_148647

theorem divisibility_of_sum (A B : ℕ) (h : ∀ k : ℕ, k ∈ {1, 2, 3, ..., 65} → (A + B) % k = 0) :
  ¬ (A + B) % 67 = 0 :=
by
  sorry

end divisibility_of_sum_l148_148647


namespace adam_cat_food_vs_dog_food_l148_148954

def cat_packages := 15
def dog_packages := 10
def cans_per_cat_package := 12
def cans_per_dog_package := 8

theorem adam_cat_food_vs_dog_food:
  cat_packages * cans_per_cat_package - dog_packages * cans_per_dog_package = 100 :=
by
  sorry

end adam_cat_food_vs_dog_food_l148_148954


namespace probability_at_most_three_heads_10_coins_l148_148366

open Nat

noncomputable def probability_at_most_three_heads (n : ℕ) (k : ℕ) : ℚ :=
  (∑ i in Finset.range (k + 1), (Nat.choose n i : ℚ)) / (2 ^ n : ℚ)

theorem probability_at_most_three_heads_10_coins :
  probability_at_most_three_heads 10 3 = 11 / 64 :=
by
  sorry

end probability_at_most_three_heads_10_coins_l148_148366


namespace probability_at_most_3_heads_10_flips_l148_148512

theorem probability_at_most_3_heads_10_flips : 
  let num_heads_0 := nat.choose 10 0,
      num_heads_1 := nat.choose 10 1,
      num_heads_2 := nat.choose 10 2,
      num_heads_3 := nat.choose 10 3,
      total_outcomes := 2^10 in
  (num_heads_0 + num_heads_1 + num_heads_2 + num_heads_3) / total_outcomes = 11 / 64 :=
by
  sorry

end probability_at_most_3_heads_10_flips_l148_148512


namespace probability_of_6_heads_in_10_flips_l148_148753

theorem probability_of_6_heads_in_10_flips :
  let total_outcomes := 2 ^ 10
  let favorable_outcomes := Nat.choose 10 6
  let probability := (favorable_outcomes : ℚ) / (total_outcomes : ℚ)
  probability = 210 / 1024 :=
by
  sorry

end probability_of_6_heads_in_10_flips_l148_148753


namespace mike_gave_12_pears_l148_148789

variable (P M K N : ℕ)

def initial_pears := 46
def pears_given_to_keith := 47
def pears_left := 11

theorem mike_gave_12_pears (M : ℕ) : 
  initial_pears - pears_given_to_keith + M = pears_left → M = 12 :=
by
  intro h
  sorry

end mike_gave_12_pears_l148_148789


namespace count_valid_numbers_l148_148131

theorem count_valid_numbers : 
  let P (n : ℕ) := n >= 1000 ∧ n <= 3000 ∧ (n % 10 = (n / 10) % 10 + (n / 100) % 10 + (n / 1000)) in
  (Σ' n, P n).to_finset.card = 109 := 
sorry

end count_valid_numbers_l148_148131


namespace max_participants_won_at_least_three_matches_l148_148564

theorem max_participants_won_at_least_three_matches :
  ∀ (n : ℕ), n = 200 → ∃ k : ℕ, k ≤ 66 ∧ ∀ p : ℕ, (p ≥ k ∧ p > 66) → false := by
  sorry

end max_participants_won_at_least_three_matches_l148_148564


namespace geometric_sequence_problem_l148_148174

variable {a : ℕ → ℝ}

theorem geometric_sequence_problem (h1 : a 5 * a 7 = 2) (h2 : a 2 + a 10 = 3) : 
  (a 12 / a 4 = 1 / 2) ∨ (a 12 / a 4 = 2) := 
sorry

end geometric_sequence_problem_l148_148174


namespace probability_heads_at_most_3_l148_148537

theorem probability_heads_at_most_3 (num_coins : ℕ) (num_heads_at_most : ℕ) (probability : ℚ):
  num_coins = 10 → 
  num_heads_at_most = 3 → 
  probability = (binomial 10 0 + binomial 10 1 + binomial 10 2 + binomial 10 3) / (2^10) →
  probability = (11 / 64) := 
by
  intros hnum_coins hnum_heads_at_most hprobability
  sorry

end probability_heads_at_most_3_l148_148537


namespace length_ST_is_8_l148_148181

-- Given conditions
variables (P Q R S T : Type) [metric_space P]
variables (angle_R_eq_90 : ∠R = 90)
variables (PR QR PQ : ℝ) [PR_eq_9 : PR = 9] [QR_eq_12 : QR = 12]
variables (PTS_eq_90 : ∠PTS = 90)
variables (ST_eq_6 : ST = 6)
variables (S_on_PQ : S ∈ PQ)
variables (T_on_QR : T ∈ QR)

-- The goal is to prove ST = 8
theorem length_ST_is_8 : ST = 8 := 
sorry -- proof omitted

end length_ST_is_8_l148_148181


namespace DY_bisects_ZDB_l148_148031

-- Define the necessary geometrical objects and properties
variable {Ω : Type*} [circle Ω]
variable {A B C D X Y Z : point Ω}
variable {BC_CD : dist B C = dist C D}
variable {AD_lt_AB : dist A D < dist A B}
variable {circumcircle_BCX : ∃ Y ≠ B, onCircle Y (circumcircle B C X) ∧ onSegment Y A B}
variable {CY_meets_Z : ∃ Z ≠ C, onRay (C, Y) Z ∧ onCircle Z Ω}

-- Main theorem stating the conclusion
theorem DY_bisects_ZDB
  (convex_ABCD : convexQuadrilateral A B C D)
  (inscribed_in_Ω : inscribedIn A B C D Ω)
  (diagonals_meet_at_X : intersects AC BD = intersects (lineThrough A C) (lineThrough B D) X)
  (Y_circumcircle_BCX : ∀ {Y}, circumcircle B C X Y → Y ≠ B → Y ∈ segment A B)
  (CY_intersects_Ω_at_Z : ∀ {Z}, Z ≠ C → onRay (C, Y) Z → onCircle Z Ω) 
: bisects (DY) (∠ Z D B) := 
sorry

end DY_bisects_ZDB_l148_148031


namespace minimum_value_2_l148_148141

noncomputable def minimum_value (x y : ℝ) : ℝ := 2 * x + 3 * y ^ 2

theorem minimum_value_2 (x y : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (h : x + 2 * y = 1) : minimum_value x y = 2 :=
sorry

end minimum_value_2_l148_148141


namespace smallest_n_for_f_greater_than_20_l148_148804

noncomputable def f (n : ℕ) : ℕ :=
  Inf {k : ℕ | k.factorial % n = 0}

def is_multiple_of_25 (n : ℕ) : Prop :=
  ∃ r : ℕ, n = 25 * r

theorem smallest_n_for_f_greater_than_20 (n : ℕ) :
  is_multiple_of_25 n → f(n) > 20 → n = 575 :=
by
  intros h1 h2
  sorry

end smallest_n_for_f_greater_than_20_l148_148804


namespace min_value_am_gm_sequence_l148_148698

-- Define the conditions and the expression that need to be proven
theorem min_value_am_gm_sequence :
  ∃ m n : ℕ, (∃ (a : ℕ → ℝ), 
  (∀ n, a n > 0) ∧ 
  (a 7 = a 6 + 2 * a 5) ∧ 
  (∃ m n, sqrt(a m * a n) = 4 * a 1) ∧ 
  m + n = 6) → 
  (1 / m : ℝ) + (5 / n) = 2 :=
sorry

end min_value_am_gm_sequence_l148_148698


namespace probability_of_at_most_3_heads_l148_148404

-- Definitions and conditions
def num_coins : ℕ := 10
def at_most_3_heads_probability : ℚ := 11 / 64

-- Statement of the problem
theorem probability_of_at_most_3_heads (n : ℕ) (p : ℚ) (h1 : n = num_coins) (h2 : p = at_most_3_heads_probability) :
  p = (1 + 10 + 45 + 120 : ℕ) / (2 ^ 10 : ℕ) := by
  sorry

end probability_of_at_most_3_heads_l148_148404


namespace service_charge_percentage_is_correct_l148_148948

-- Define the conditions
def orderAmount : ℝ := 450
def totalAmountPaid : ℝ := 468
def serviceCharge : ℝ := totalAmountPaid - orderAmount

-- Define the target percentage
def expectedServiceChargePercentage : ℝ := 4.0

-- Proof statement: the service charge percentage is expectedServiceChargePercentage
theorem service_charge_percentage_is_correct : 
  (serviceCharge / orderAmount) * 100 = expectedServiceChargePercentage :=
by
  sorry

end service_charge_percentage_is_correct_l148_148948


namespace calculate_fish_in_pond_l148_148569

noncomputable def April1_sample_size : ℕ := 80 -- Number of fish tagged and released on April 1.
noncomputable def August1_sample_size : ℕ := 100 -- Number of fish caught in the sample on August 1.
noncomputable def tagged_August1 : ℕ := 4 -- Number of tagged fish found in the August 1 sample.
noncomputable def April_to_August_loss : ℝ := 0.30 -- Fraction of original tagged fish lost by August 1.
noncomputable def August_fish_origin_April : ℝ := 0.50 -- Fraction of August 1 fish that were in the pond since April 1.

theorem calculate_fish_in_pond (April1_sample_size = 80) (August1_sample_size = 100) (tagged_August1 = 4)
  (April_to_August_loss = 0.30) (August_fish_origin_April = 0.50) :
  ∃ (x : ℕ), x = 1000 :=
by
  sorry

end calculate_fish_in_pond_l148_148569


namespace PD_perpendicular_to_BC_l148_148015

-- Definitions using the conditions
variables {A B C D I1 I2 O1 O2 P : Type} [AffineSpace ℝ Type]
variables [IncidenceGeometry A B C D I1 I2 O1 O2 P]

-- Proof problem: Given the triangle and the specified points and lines, prove the perpendicularity
theorem PD_perpendicular_to_BC : ∀ (triangle : Triangle A B C) 
  (D_on_BC : D ∈ segment B C)
  (I1_incenter_ABD : is_incenter I1 (Triangle A B D))
  (I2_incenter_ACD : is_incenter I2 (Triangle A C D))
  (O1_circumcenter_AI1D : is_circumcenter O1 (Triangle A I1 D))
  (O2_circumcenter_AI2D : is_circumcenter O2 (Triangle A I2 D))
  (P_intersection : P ∈ intersection (line I1 O2) (line I2 O1)), 
  perpendicular (line P D) (line B C) :=
begin
  sorry
end

end PD_perpendicular_to_BC_l148_148015


namespace central_angle_correct_l148_148867

-- Given a circle with radius r and an arc length of (3/2)r, the central angle
def central_angle (r : ℝ) (arc_length : ℝ) : ℝ := arc_length / r

-- Given: r and arc_length = (3/2)r
-- Prove: central_angle r ((3/2) * r) = (3/2)rad
theorem central_angle_correct (r : ℝ) (h: r > 0) : central_angle r ((3 : ℝ) / 2 * r) = (3 : ℝ) / 2 :=
by
  sorry

end central_angle_correct_l148_148867


namespace f_deriv_ineq_g_monotonicity_l148_148112

noncomputable def f (x : ℝ) : ℝ := x - 2 / x - Math.log x

noncomputable def f' (x : ℝ) : ℝ := 1 + 2 / (x * x) - 1 / x

theorem f_deriv_ineq (x : ℝ) (h : x > 0) : f' x < 2 ↔ x > 1 :=
by
  sorry

noncomputable def g (x : ℝ) : ℝ := f x - 4 * x

noncomputable def g' (x : ℝ) : ℝ := -3 + 2 / (x * x) - 1 / x

theorem g_monotonicity (x : ℝ) (h : x > 0) : 
  (g' x > 0 ↔ 0 < x ∧ x < (2 / 3)) ∧
  (g' x < 0 ↔ x > (2 / 3)) :=
by
  sorry

end f_deriv_ineq_g_monotonicity_l148_148112


namespace sam_distance_when_meet_l148_148068

noncomputable def distance_sam_walks
  (initial_distance : ℝ)
  (fred_speed : ℝ)
  (sam_speed : ℝ)
  : ℝ :=
let t := initial_distance / (fred_speed + sam_speed) in sam_speed * t

theorem sam_distance_when_meet :
  distance_sam_walks 75 4 6 = 45 :=
by
  sorry

end sam_distance_when_meet_l148_148068


namespace age_problem_l148_148945

theorem age_problem (S F : ℕ) (h1 : F = S + 27) (h2 : F + 2 = 2 * (S + 2)) :
  S = 25 := by
  sorry

end age_problem_l148_148945


namespace compare_magnitudes_l148_148644

theorem compare_magnitudes :
  let a := Real.log 2
  let b := 5^(-1/2 : ℤ)
  let c := ∫ x in 0..(Real.pi / 2), (1/2 : ℝ) * Real.cos x
  a > c ∧ c > b := by
sorry

end compare_magnitudes_l148_148644


namespace max_participants_won_at_least_three_matches_l148_148563

theorem max_participants_won_at_least_three_matches :
  ∀ (n : ℕ), n = 200 → ∃ k : ℕ, k ≤ 66 ∧ ∀ p : ℕ, (p ≥ k ∧ p > 66) → false := by
  sorry

end max_participants_won_at_least_three_matches_l148_148563


namespace triangulation_exactly_two_internal_triangles_l148_148809

theorem triangulation_exactly_two_internal_triangles (n : ℕ) (h₁ : n > 7) :
  ∃ P : convex_polygon n, 
  (∀ (d : ℕ), d = n - 3 → non_intersecting_diagonals d P → divides_into_triangles d P) ∧ 
  (∃ (T : set (triangle P)), ∀ (t : triangle P), internal_triangle t ↔ t ∈ T) →
  (number_of_internal_triangles T = 2) →
  number_of_dissections_with_two_internal_triangles P = n * (nat.choose (n-4) 4) * 2^(n-9) := 
begin
  sorry
end

end triangulation_exactly_two_internal_triangles_l148_148809


namespace cost_per_tshirt_l148_148021
-- Import necessary libraries

-- Define the given conditions
def t_shirts : ℕ := 20
def total_cost : ℝ := 199

-- Define the target proof statement
theorem cost_per_tshirt : (total_cost / t_shirts) = 9.95 := 
sorry

end cost_per_tshirt_l148_148021


namespace probability_at_most_3_heads_10_flips_l148_148514

theorem probability_at_most_3_heads_10_flips : 
  let num_heads_0 := nat.choose 10 0,
      num_heads_1 := nat.choose 10 1,
      num_heads_2 := nat.choose 10 2,
      num_heads_3 := nat.choose 10 3,
      total_outcomes := 2^10 in
  (num_heads_0 + num_heads_1 + num_heads_2 + num_heads_3) / total_outcomes = 11 / 64 :=
by
  sorry

end probability_at_most_3_heads_10_flips_l148_148514


namespace min_cos_beta_l148_148086

open Real

theorem min_cos_beta (α β : ℝ)
  (hα : 0 < α ∧ α < π / 2)
  (hβ : 0 < β ∧ β < π / 2)
  (h_eq : sin (2 * α + β) = (3 / 2) * sin β) :
  cos β = sqrt 5 / 3 := 
sorry

end min_cos_beta_l148_148086


namespace probability_of_at_most_3_heads_out_of_10_l148_148485
open_locale big_operators
open nat

theorem probability_of_at_most_3_heads_out_of_10 (n : ℕ) (p : ℕ) (h : 0 < n) (hp : p = 2) :
  ∑ k in finset.range (4), nat.choose n k = 176 → nat.pow p n = 1024 → (∑ k in finset.range (4), nat.choose n k) / (nat.pow p n) = 11 / 64 :=
by { intro h₁ h₂, 
     have : (∑ k in finset.range 4, nat.choose n k) / (nat.pow p n) = 176 / 1024,
     { rw [h₁, h₂] },
     rw this,
     norm_num }

end probability_of_at_most_3_heads_out_of_10_l148_148485


namespace find_valid_ns_l148_148665

theorem find_valid_ns (n : ℕ) (h1 : n > 1) (h2 : ∃ k : ℕ, k^2 = (n^2 + 7 * n + 136) / (n-1)) : n = 5 ∨ n = 37 :=
sorry

end find_valid_ns_l148_148665


namespace sum_S_n_gt_inequality_l148_148775

noncomputable def a_n : ℕ → ℝ
| n => (n + 2) ^ 2 / 4

def S_n (n : ℕ) : ℝ := ∑ i in range n, 1 / a_n (i + 1)

theorem sum_S_n_gt_inequality (n : ℕ) (hn : n > 0) : 
  S_n n > 4 * n / (3 * (n + 3)) :=
sorry

end sum_S_n_gt_inequality_l148_148775


namespace probability_at_most_three_heads_10_coins_l148_148364

open Nat

noncomputable def probability_at_most_three_heads (n : ℕ) (k : ℕ) : ℚ :=
  (∑ i in Finset.range (k + 1), (Nat.choose n i : ℚ)) / (2 ^ n : ℚ)

theorem probability_at_most_three_heads_10_coins :
  probability_at_most_three_heads 10 3 = 11 / 64 :=
by
  sorry

end probability_at_most_three_heads_10_coins_l148_148364


namespace external_similarity_point_is_A_l148_148262

-- Define the entities
variable (O1 O2 A B C : Point)
variable (circle1 : Circle)
variable (circle2 : Circle)
variable (O1_center : O1 ∈ circle1)
variable (O2_center : O2 ∈ circle2)
variable (A_intersection : A ∈ circle1 ∧ A ∈ circle2)
variable (B_diameter1 : EuclideanDistance A B = EuclideanDistance O1 A * 2)
variable (C_diameter2 : EuclideanDistance A C = EuclideanDistance O2 A * 2)

-- Define the claim
def external_similarity_point : Prop :=
  A = similarity_point circle1 circle2

theorem external_similarity_point_is_A :
  external_similarity_point O1 O2 A B C circle1 circle2 O1_center O2_center A_intersection B_diameter1 C_diameter2 :=
by
  sorry

end external_similarity_point_is_A_l148_148262


namespace sum_of_base_7_digits_of_999_l148_148310

theorem sum_of_base_7_digits_of_999 : 
  let n := 999 
  let base := 7 
  let base7_rep := [2, 6, 2, 5]
  list.sum base7_rep = 15 :=
by
  let n := 999
  let base := 7
  let base7_rep := [2, 6, 2, 5]
  have h : list.sum base7_rep = 2 + 6 + 2 + 5 := sorry
  have sum_is_15 : 2 + 6 + 2 + 5 = 15 := sorry
  show list.sum base7_rep = 15, from h.trans sum_is_15

end sum_of_base_7_digits_of_999_l148_148310


namespace incenter_distance_less_than_vertices_l148_148289

variable {α : Type*} [metric_space α]

abbrev Point (α : Type*) := α

variables (A B C D I1 I2 : Point α)

def incircle_center (A B C : Point α) : Point α := I1
def incircle_center' (A D C : Point α) : Point α := I2

theorem incenter_distance_less_than_vertices
  (h_shared_side : dist A C = dist A C)
  (h_incenter_ABC : I1 = incircle_center A B C)
  (h_incenter_ADC : I2 = incircle_center' A D C)
  : dist I1 I2 < dist B D :=
sorry

end incenter_distance_less_than_vertices_l148_148289


namespace prob_heads_at_most_3_out_of_10_flips_l148_148452

def coin_flips := 10
def max_heads := 3
def total_outcomes := 2^coin_flips
def favorable_outcomes := (nat.choose coin_flips 0) +
                          (nat.choose coin_flips 1) +
                          (nat.choose coin_flips 2) +
                          (nat.choose coin_flips 3)

theorem prob_heads_at_most_3_out_of_10_flips :
  (favorable_outcomes : ℚ) / total_outcomes = 11 / 64 :=
by
  sorry

end prob_heads_at_most_3_out_of_10_flips_l148_148452


namespace part1_part2_l148_148102

-- Definitions based on initial problem conditions
def a (n : ℕ) : ℤ := 2 * n + 1
def S (n : ℕ) : ℤ := n * (2 * (1 + n) + n - 1) / 2  -- Sum of the first n terms of an arithmetic series, simplified using known general form

-- Given conditions
def condition1 : Prop := a 3 = 7
def condition2 : Prop := 2 * (a 7 + 4) = S 1 + S 5

-- Proof goals
theorem part1 : condition1 ∧ condition2 → ∀ n, a n = 2 * n + 1 := sorry

noncomputable def b (n : ℕ) : ℝ := (a n : ℝ) * Real.sin (↑(a n) * Real.pi / 2)

theorem part2 (n : ℕ) (h : condition1 ∧ condition2) : 
  let T : ℕ → ℝ := λ n, (Finset.range (n + 1)).sum (λ i, b i)
  in T n = if ∃ (k : ℕ), n = 2 * k - 1 then -(↑n + 2) else n := sorry

end part1_part2_l148_148102


namespace problem_equivalence_l148_148213

variable {x y z w : ℝ}

theorem problem_equivalence (h : (x - y) * (z - w) / ((y - z) * (w - x)) = 3 / 7) :
  (x - z) * (y - w) / ((x - y) * (z - w)) = -4 / 3 := 
sorry

end problem_equivalence_l148_148213


namespace simplify_expression_l148_148238

theorem simplify_expression (a b : ℤ) : 
  (17 * a + 45 * b) + (15 * a + 36 * b) - (12 * a + 42 * b) - 3 * (2 * a + 3 * b) = 14 * a + 30 * b :=
by
  sorry

end simplify_expression_l148_148238


namespace max_sum_of_first_n_terms_l148_148701

variable {a : ℕ → ℝ} -- Define sequence a with index ℕ and real values
variable {d : ℝ}      -- Common difference for the arithmetic sequence

-- Conditions and question are formulated into the theorem statement
theorem max_sum_of_first_n_terms (h_arith : ∀ n, a (n + 1) = a n + d)
  (h_diff_neg : d < 0)
  (h_a4_eq_a12 : (a 4)^2 = (a 12)^2) :
  n = 7 ∨ n = 8 := 
sorry

end max_sum_of_first_n_terms_l148_148701


namespace clerical_staff_percentage_l148_148819

def employees_initial := 4500
def clerical_fraction := 5/12
def reduction_fraction := 1/4
def new_non_clerical := 50

def clerical_initial := clerical_fraction * employees_initial
def reduction := reduction_fraction * clerical_initial
def clerical_after_reduction := clerical_initial - reduction
def total_after_changes := employees_initial - reduction + new_non_clerical

def percentage_clerical := (clerical_after_reduction / total_after_changes) * 100

theorem clerical_staff_percentage : percentage_clerical ≈ 34.46 := by
  sorry

end clerical_staff_percentage_l148_148819


namespace smallest_possible_sum_of_face_l148_148266

theorem smallest_possible_sum_of_face (n : ℕ) (h: (∀ {a b c d : ℕ}, a + b + c ≥ 10 ∧ a + b + d ≥ 10 ∧ a + c + d ≥ 10 ∧ b + c + d ≥ 10)) : 
  ∃ a b c d : ℕ, a + b + c + d = 16 ∧ (a = 2 ∨ a = 3 ∨ a = 5 ∨ a = 6) ∧ (b = 2 ∨ b = 3 ∨ b = 5 ∨ b = 6) ∧ (c = 2 ∨ c = 3 ∨ c = 5 ∨ c = 6) ∧ (d = 2 ∨ d = 3 ∨ d = 5 ∨ d = 6) :=
begin
  sorry
end

end smallest_possible_sum_of_face_l148_148266


namespace bottles_in_one_bag_l148_148923

theorem bottles_in_one_bag (total_bottles : ℕ) (cartons bags_per_carton : ℕ)
  (h1 : total_bottles = 180)
  (h2 : cartons = 3)
  (h3 : bags_per_carton = 4) :
  total_bottles / cartons / bags_per_carton = 15 :=
by sorry

end bottles_in_one_bag_l148_148923


namespace longest_proper_sequence_D40_l148_148045

structure Domino (i j : ℕ) [Distinct : i ≠ j]

def D40 : Set (ℕ × ℕ) := { p | p.1 ≠ p.2 ∧ 1 ≤ p.1 ∧ p.1 ≤ 40 ∧ 1 ≤ p.2 ∧ p.2 ≤ 40 }

def is_proper_sequence (seq : List (ℕ × ℕ)) : Prop :=
  match seq with
  | []                => True
  | [d]               => True
  | (d1 :: d2 :: ds) =>
    (∀ i j, (i, j) ∈ (d1 :: d2 :: ds) → (j, i) ∉ (d1 :: d2 :: ds)) ∧ -- no (i, j) and (j, i) both appear
    List.chain (λ x y => x.2 = y.1) d1 (d2 :: ds) ∧ -- first coordinate of each after first == second coordinate of preceding
    is_proper_sequence (d2 :: ds) -- recursive check

noncomputable def longest_proper_sequence_length : ℕ :=
  let seqs : List (List (ℕ × ℕ)) := { seq | is_proper_sequence seq ∧ ∀ d ∈ seq, (d.1, d.2) ∈ D40 }
  seqs.foldr (λ seq acc => max acc seq.length) 0

theorem longest_proper_sequence_D40 : longest_proper_sequence_length = 723 := sorry

end longest_proper_sequence_D40_l148_148045


namespace probability_heads_at_most_3_of_10_coins_flipped_l148_148377

theorem probability_heads_at_most_3_of_10_coins_flipped :
  (∑ k in finset.range 4, nat.choose 10 k) / 2^10 = 11 / 64 :=
sorry

end probability_heads_at_most_3_of_10_coins_flipped_l148_148377


namespace sum_of_base7_digits_of_999_l148_148314

theorem sum_of_base7_digits_of_999 : 
  let base7_representation := 2 * 7^3 + 6 * 7^2 + 2 * 7^1 + 5
  in (let digits_sum := 2 + 6 + 2 + 5 in digits_sum = 15) := sorry

end sum_of_base7_digits_of_999_l148_148314


namespace least_n_for_zeroes_l148_148054

-- Define a function to count the number of factors of a prime p in n!
def factors_of (n p : ℕ) : ℕ :=
  if p = 0 ∨ p = 1 then 0
  else
    let rec aux m :=
      if m = 0 then 0
      else m / p + aux (m / p)
    aux n

theorem least_n_for_zeroes (n : ℕ) :
  (factors_of (2 * n) 5 - 2 * factors_of n 5 ≥ 4) → n ≥ 313 := sorry

end least_n_for_zeroes_l148_148054


namespace probability_heads_at_most_3_of_10_coins_flipped_l148_148374

theorem probability_heads_at_most_3_of_10_coins_flipped :
  (∑ k in finset.range 4, nat.choose 10 k) / 2^10 = 11 / 64 :=
sorry

end probability_heads_at_most_3_of_10_coins_flipped_l148_148374


namespace cylinder_height_in_sphere_l148_148586

noncomputable def height_of_cylinder (r R : ℝ) : ℝ :=
  2 * Real.sqrt (R ^ 2 - r ^ 2)

theorem cylinder_height_in_sphere :
  height_of_cylinder 3 6 = 6 * Real.sqrt 3 :=
by
  sorry

end cylinder_height_in_sphere_l148_148586


namespace probability_at_most_3_heads_10_coins_l148_148421

noncomputable def coin_flip_prob_at_most_3_heads : ℚ :=
  let total_outcomes := 2^10
  let favorable_outcomes := Nat.binomial 10 0 + Nat.binomial 10 1 + Nat.binomial 10 2 + Nat.binomial 10 3
  favorable_outcomes / total_outcomes

theorem probability_at_most_3_heads_10_coins : coin_flip_prob_at_most_3_heads = 11 / 64 :=
by
  -- Definitions of let values are integrated directly in the theorem body
  let total_outcomes := 2^10
  let favorable_outcomes := Nat.binomial 10 0 + Nat.binomial 10 1 + Nat.binomial 10 2 + Nat.binomial 10 3
  show favorable_outcomes / total_outcomes = 11 / 64
  sorry

end probability_at_most_3_heads_10_coins_l148_148421


namespace polar_equations_and_max_OB_OA_l148_148767

-- Definitions of the curves and conditions
def C1 := {p : ℝ × ℝ | p.1 + p.2 = 4}
def C2 := {p : ℝ × ℝ | ∃ θ : ℝ, p.1 = 1 + cos θ ∧ p.2 = sin θ}

def ray_l (α : ℝ) := {p : ℝ × ℝ | ∃ ρ : ℝ, ρ > 0 ∧ p = (ρ * cos α, ρ * sin α)}

-- Statement of the proof problem

theorem polar_equations_and_max_OB_OA :
  (∀ ρ θ, (ρ * (cos θ + sin θ) = 4) ↔ (∃ x y, (x, y) ∈ C1 ∧ ρ = sqrt(x^2 + y^2) ∧ tan θ = y / x)) ∧
  (∀ ρ θ, (ρ = 2 * cos θ) ↔ (∃ x y, (x, y) ∈ C2 ∧ ρ = sqrt(x^2 + y^2) ∧ tan θ = y / x)) ∧
  (∀ α : ℝ, ∃ OA OB : ℝ, OA > 0 ∧ OB > 0 ∧ 
    (∀ A ∈ ray_l α, A ∈ C1 → OA = sqrt(A.1^2 + A.2^2)) ∧ 
    (∀ B ∈ ray_l α, B ∈ C2 → OB = sqrt(B.1^2 + B.2^2)) ∧ 
    (max_val : ℝ, max_val = OB / OA → max_val = 1/4 * (sqrt 2 + 1))) :=
by
  sorry

end polar_equations_and_max_OB_OA_l148_148767


namespace no_return_to_initial_l148_148635

theorem no_return_to_initial (a b c d: ℝ) (h1: 0 < a) (h2: 0 < b) (h3: 0 < c) (h4: 0 < d) (h5: a ≠ 1) (h6: b ≠ 1) (h7: c ≠ 1) (h8: d ≠ 1) :
  ∀ m: ℕ, (a ≠ (λ (t: ℕ), (a*b^t, b*c^t, c*d^t, d*(a*b*c)^t) )m ) :=
begin
  sorry
end

end no_return_to_initial_l148_148635


namespace symmetric_point_xy_plane_l148_148177

theorem symmetric_point_xy_plane (P : ℝ × ℝ × ℝ) (h : P = (1, 1, -2)) :
  ∃ Q : ℝ × ℝ × ℝ, Q = (1, 1, 2) ∧ 
    (Q.1 = P.1) ∧ (Q.2 = P.2) ∧ (Q.3 = -P.3) :=
by {
  use (1, 1, 2),
  refine ⟨rfl, ⟨_, _, _⟩⟩,
  all_goals {rfl},
  sorry
}

end symmetric_point_xy_plane_l148_148177


namespace artist_paintings_in_four_weeks_l148_148960

theorem artist_paintings_in_four_weeks (hours_per_week : ℕ) (hours_per_painting : ℕ) (weeks : ℕ) (total_paintings : ℕ) :
  hours_per_week = 30 → hours_per_painting = 3 → weeks = 4 → total_paintings = ((hours_per_week / hours_per_painting) * weeks) → total_paintings = 40 :=
by
  intros h_week h_painting h_weeks h_total
  rw [h_week, h_painting, h_weeks]
  norm_num
  exact h_total

end artist_paintings_in_four_weeks_l148_148960


namespace probability_of_at_most_3_heads_out_of_10_l148_148481
open_locale big_operators
open nat

theorem probability_of_at_most_3_heads_out_of_10 (n : ℕ) (p : ℕ) (h : 0 < n) (hp : p = 2) :
  ∑ k in finset.range (4), nat.choose n k = 176 → nat.pow p n = 1024 → (∑ k in finset.range (4), nat.choose n k) / (nat.pow p n) = 11 / 64 :=
by { intro h₁ h₂, 
     have : (∑ k in finset.range 4, nat.choose n k) / (nat.pow p n) = 176 / 1024,
     { rw [h₁, h₂] },
     rw this,
     norm_num }

end probability_of_at_most_3_heads_out_of_10_l148_148481


namespace Nancy_picked_35_limes_l148_148681

-- Define the quantities
def FredLimes := 36
def AlyssaLimes := 32
def TotalLimes := 103

-- The quantity of limes Nancy picked
def NancyLimes : ℕ := TotalLimes - (FredLimes + AlyssaLimes)

-- The proof problem
theorem Nancy_picked_35_limes : NancyLimes = 35 :=
by
  dsimp[NancyLimes] 
  rw [FredLimes, AlyssaLimes, TotalLimes]
  -- sorry: proof ommitted
  sorry

end Nancy_picked_35_limes_l148_148681


namespace every_pos_int_representation_l148_148230

theorem every_pos_int_representation:
  ∀ n: ℕ, n > 0 →
    ∃ (k: ℕ) (m n: Fin k → ℕ),
      (∀ i j, i < j → m i > m j) ∧
      (∀ i j, i < j → n i < n j) ∧ 
      (∀ i, 0 ≤ n i) ∧ 
      (∃ s : Fin k → ℕ, n = ∑ i, 3^(m i) * 2^(n i)) :=
sorry

end every_pos_int_representation_l148_148230


namespace max_profit_at_max_price_l148_148928

-- Definitions based on the given problem's conditions
def cost_price : ℝ := 30
def profit_margin : ℝ := 0.5
def max_price : ℝ := cost_price * (1 + profit_margin)
def min_price : ℝ := 35
def base_sales : ℝ := 350
def sales_decrease_per_price_increase : ℝ := 50
def price_increase_step : ℝ := 5

-- Profit function based on the conditions
def profit (x : ℝ) : ℝ := (-10 * x^2 + 1000 * x - 21000)

-- Maximum profit and corresponding price
theorem max_profit_at_max_price :
  ∀ x, min_price ≤ x ∧ x ≤ max_price →
  profit x ≤ profit max_price ∧ profit max_price = 3750 :=
by sorry

end max_profit_at_max_price_l148_148928


namespace parallelism_perpendicularity_l148_148607

theorem parallelism_perpendicularity :
  (∀ (l₁ l₂ l₃ : Line), (Perpendicular l₁ l₃ ∧ Perpendicular l₂ l₃ → ¬ Parallel l₁ l₂)) ∧
  (∀ (l₁ l₂ : Line) (P : Plane), (Perpendicular l₁ P ∧ Perpendicular l₂ P → Parallel l₁ l₂)) ∧
  (∀ (P₁ P₂ : Plane) (l : Line), (Perpendicular P₁ l ∧ Perpendicular P₂ l → Parallel P₁ P₂)) ∧
  (∀ (P₁ P₂ P₃ : Plane), (Perpendicular P₁ P₃ ∧ Perpendicular P₂ P₃ → ¬ Parallel P₁ P₂)) :=
by
  sorry

end parallelism_perpendicularity_l148_148607


namespace number_of_integers_in_range_of_f_l148_148672

noncomputable def f (x : ℝ) : ℝ := 2 * Real.cos (2 * x) + 2 * Real.cos x - 2019

theorem number_of_integers_in_range_of_f : 
  (set.range f).count (λ y, y ∈ set.Ico (-2021.25) (-2015)) = 7 := sorry

end number_of_integers_in_range_of_f_l148_148672


namespace sqrt_fraction_value_l148_148317

theorem sqrt_fraction_value (a b c d : Nat) (h : a = 2 ∧ b = 0 ∧ c = 2 ∧ d = 3) : 
  Real.sqrt (2023 / (a + b + c + d)) = 17 := by
  sorry

end sqrt_fraction_value_l148_148317


namespace range_of_f_l148_148047

-- Define the function and conditions
def f (x : ℝ) : ℝ := 1 / (x^2 - 4 * x - 2)
def g (x : ℝ) : ℝ := x^2 - 4 * x - 2

-- State the main theorem
theorem range_of_f :
  (∀ (x : ℝ), g(x) ≠ 0 → f(x) ∈ ((Set.Iic (-1/6)) ∪ (Set.Ioi 0))) :=
sorry

end range_of_f_l148_148047


namespace no_integer_solutions_for_quadratic_l148_148231

theorem no_integer_solutions_for_quadratic :
  ¬ ∃ x y : ℤ, x^2 + 3 * x * y - 2 * y^2 = 122 :=
begin
  sorry
end

end no_integer_solutions_for_quadratic_l148_148231


namespace correct_statements_l148_148103

variables {n : ℕ}
noncomputable def S (n : ℕ) : ℝ := (n + 1) / n
noncomputable def T (n : ℕ) : ℝ := (n + 1)
noncomputable def a (n : ℕ) : ℝ := if n = 1 then 2 else (-(1:ℝ)) / (n * (n - 1))

theorem correct_statements (n : ℕ) (hn : n ≠ 0) :
  (S n + T n = S n * T n) ∧ (a 1 = 2) ∧ (∀ n, ∃ d, ∀ m, T (n + m) - T n = m * d) ∧ (S n = (n + 1) / n) :=
by
  sorry

end correct_statements_l148_148103


namespace probability_at_most_3_heads_10_coins_l148_148418

noncomputable def coin_flip_prob_at_most_3_heads : ℚ :=
  let total_outcomes := 2^10
  let favorable_outcomes := Nat.binomial 10 0 + Nat.binomial 10 1 + Nat.binomial 10 2 + Nat.binomial 10 3
  favorable_outcomes / total_outcomes

theorem probability_at_most_3_heads_10_coins : coin_flip_prob_at_most_3_heads = 11 / 64 :=
by
  -- Definitions of let values are integrated directly in the theorem body
  let total_outcomes := 2^10
  let favorable_outcomes := Nat.binomial 10 0 + Nat.binomial 10 1 + Nat.binomial 10 2 + Nat.binomial 10 3
  show favorable_outcomes / total_outcomes = 11 / 64
  sorry

end probability_at_most_3_heads_10_coins_l148_148418


namespace probability_of_at_most_3_heads_out_of_10_l148_148484
open_locale big_operators
open nat

theorem probability_of_at_most_3_heads_out_of_10 (n : ℕ) (p : ℕ) (h : 0 < n) (hp : p = 2) :
  ∑ k in finset.range (4), nat.choose n k = 176 → nat.pow p n = 1024 → (∑ k in finset.range (4), nat.choose n k) / (nat.pow p n) = 11 / 64 :=
by { intro h₁ h₂, 
     have : (∑ k in finset.range 4, nat.choose n k) / (nat.pow p n) = 176 / 1024,
     { rw [h₁, h₂] },
     rw this,
     norm_num }

end probability_of_at_most_3_heads_out_of_10_l148_148484


namespace arithmetic_seq_div_l148_148660

open Nat

theorem arithmetic_seq_div (a d : ℕ) (h : ∀ n : ℕ, 
  ∏ i in range n, (a + i*d) ∣ ∏ i in range n, (a + (i+n)*d)) :
  ∃ k : ℕ, ∀ n : ℕ, a + n*d = k*(n+1) := sorry

end arithmetic_seq_div_l148_148660


namespace age_of_youngest_child_l148_148580

theorem age_of_youngest_child 
  (cost_mother : ℝ) (cost_per_year : ℝ) (total_cost : ℝ)
  (triplet_count : ℕ) (multiple_choice : list ℕ)
  (all_triplet_age_valid : ∀ t, t ∈ multiple_choice → 15 - 3 * t ∈ multiple_choice ∨ (15 - 3 * t > 0 ∧ (15 - 3 * t ≠ t)))
  : 15 - 3 * 4 = 3 :=
by
  let cost_mother := 6.95
  let cost_per_year := 0.55
  let total_cost := 15.25
  let triplet_count := 3
  let multiple_choice := [1, 2, 3, 4, 5]
  sorry -- Proof will go here

end age_of_youngest_child_l148_148580


namespace count_ordered_pairs_satisfying_equation_l148_148747

theorem count_ordered_pairs_satisfying_equation :
  (λ (num_pairs : ℕ), num_pairs = 2) :=
by
  sorry

end count_ordered_pairs_satisfying_equation_l148_148747


namespace probability_of_at_most_3_heads_out_of_10_l148_148479
open_locale big_operators
open nat

theorem probability_of_at_most_3_heads_out_of_10 (n : ℕ) (p : ℕ) (h : 0 < n) (hp : p = 2) :
  ∑ k in finset.range (4), nat.choose n k = 176 → nat.pow p n = 1024 → (∑ k in finset.range (4), nat.choose n k) / (nat.pow p n) = 11 / 64 :=
by { intro h₁ h₂, 
     have : (∑ k in finset.range 4, nat.choose n k) / (nat.pow p n) = 176 / 1024,
     { rw [h₁, h₂] },
     rw this,
     norm_num }

end probability_of_at_most_3_heads_out_of_10_l148_148479


namespace largest_number_2013_l148_148875

theorem largest_number_2013 (x y : ℕ) (h1 : x + y = 2013)
    (h2 : y = 5 * (x / 100 + 1)) : max x y = 1913 := by
  sorry

end largest_number_2013_l148_148875


namespace probability_at_most_3_heads_l148_148439

def binom (n k : ℕ) : ℕ := Nat.choose n k

theorem probability_at_most_3_heads :
  let total_outcomes := 2^10
  let successful_outcomes := binom 10 0 + binom 10 1 + binom 10 2 + binom 10 3
  (successful_outcomes : ℚ) / total_outcomes = 11 / 64 := 
by 
  let total_outcomes := 2 ^ 10
  let successful_outcomes := binom 10 0 + binom 10 1 + binom 10 2 + binom 10 3
  have h : (successful_outcomes : ℚ) / total_outcomes = 11 / 64 := sorry
  exact h

end probability_at_most_3_heads_l148_148439


namespace number_of_ordered_pairs_l148_148124

open Finset

noncomputable def count_pairs (U : Finset ℕ) :=
  (U.powerset.filter (λ A, A.nonempty)).sum (λ A, 
    (U \ A).powerset.filter (λ B, B.nonempty ∧ disjoint A B)).card

theorem number_of_ordered_pairs : count_pairs (finset.range 5) = 50 := 
begin
  -- Proof will be provided here
  sorry
end

end number_of_ordered_pairs_l148_148124


namespace probability_heads_at_most_three_out_of_ten_coins_l148_148547

theorem probability_heads_at_most_three_out_of_ten_coins :
  let total_outcomes := 2^10 in
  let favorable_outcomes := Nat.choose 10 0 + Nat.choose 10 1 + Nat.choose 10 2 + Nat.choose 10 3 in
  favorable_outcomes / total_outcomes = 11 / 64 :=
by
  sorry

end probability_heads_at_most_three_out_of_ten_coins_l148_148547


namespace parabola_tangent_intersection_length_l148_148220

theorem parabola_tangent_intersection_length :
    let C := λ x y, y^2 = 6 * x
    let F := (3 / 2, 0)
    ∀ A B : ℝ × ℝ,
      C A.1 A.2 ∧ C B.1 B.2 →
      let l₁ := λ x y, A.2 * y = 3 * (x + A.1)
      let l₂ := λ x y, B.2 * y = 3 * (x + B.1)
      ∃ P : ℝ × ℝ,
        (l₁ P.1 P.2 ∧ l₂ P.1 P.2 ∧ dist P F = 2 * (sqrt 3)) →
        dist A B = 8 := sorry

end parabola_tangent_intersection_length_l148_148220


namespace probability_heads_at_most_3_l148_148529

theorem probability_heads_at_most_3 (num_coins : ℕ) (num_heads_at_most : ℕ) (probability : ℚ):
  num_coins = 10 → 
  num_heads_at_most = 3 → 
  probability = (binomial 10 0 + binomial 10 1 + binomial 10 2 + binomial 10 3) / (2^10) →
  probability = (11 / 64) := 
by
  intros hnum_coins hnum_heads_at_most hprobability
  sorry

end probability_heads_at_most_3_l148_148529


namespace probability_more_red_balls_drawn_probability_distribution_and_expected_value_l148_148925

noncomputable theory
open_locale big_operators

-- Define the initial conditions of the problem
def num_red_balls := 6
def num_white_balls := 4
def num_draws := 4

-- Question 1: Prove the probability that the number of red balls drawn is greater than the number of white balls drawn is 19/42.
theorem probability_more_red_balls_drawn : 
  (∑ (r in finset.range (num_red_balls + 1)), if r > num_draws - r then (nat.choose num_red_balls r) * (nat.choose num_white_balls (num_draws - r)) else 0) / (nat.choose (num_red_balls + num_white_balls) num_draws) = 19 / 42 :=
sorry

-- Define the random variable X representing the total score
def X (r : ℕ) : ℕ := 2 * r + (num_draws - r)

-- Question 2: Prove the probability distribution of X and its expected value.
theorem probability_distribution_and_expected_value :
  (finset.sum (finset.range (num_red_balls + 1))
    (λ r, (nat.choose num_red_balls r) * (nat.choose num_white_balls (num_draws - r)) / (nat.choose (num_red_balls + num_white_balls) num_draws) * X r)) =
    (1 / 210 * 4 + 4 / 35 * 5 + 3 / 7 * 6 + 8 / 21 * 7 + 1 / 14 * 8) :=
sorry

end probability_more_red_balls_drawn_probability_distribution_and_expected_value_l148_148925


namespace arrangement_ways_l148_148006

-- Defining the conditions
def num_basil_plants : Nat := 5
def num_tomato_plants : Nat := 4
def num_total_units : Nat := num_basil_plants + 1

-- Proof statement
theorem arrangement_ways : (num_total_units.factorial) * (num_tomato_plants.factorial) = 17280 := by
  sorry

end arrangement_ways_l148_148006


namespace calculate_rolls_of_toilet_paper_l148_148653

-- Definitions based on the problem conditions
def seconds_per_egg := 15
def minutes_per_roll := 30
def total_cleaning_minutes := 225
def number_of_eggs := 60
def time_per_minute := 60

-- Calculation of the time spent on eggs in minutes
def egg_cleaning_minutes := (number_of_eggs * seconds_per_egg) / time_per_minute

-- Total cleaning time minus time spent on eggs
def remaining_cleaning_minutes := total_cleaning_minutes - egg_cleaning_minutes

-- Verify the number of rolls of toilet paper cleaned up
def rolls_of_toilet_paper := remaining_cleaning_minutes / minutes_per_roll

-- Theorem statement to be proved
theorem calculate_rolls_of_toilet_paper : rolls_of_toilet_paper = 7 := by
  sorry

end calculate_rolls_of_toilet_paper_l148_148653


namespace geometric_locus_l148_148338

noncomputable def locus_type (z z1 z2 : ℂ) (λ : ℝ) : String :=
  if λ = 1 then "straight line" else "circle"

theorem geometric_locus (z z1 z2 : ℂ) (λ : ℝ) (h : λ > 0) :
  (λ = 1 → locus_type z z1 z2 λ = "straight line") ∧ (λ ≠ 1 → locus_type z z1 z2 λ = "circle") :=
by
  split 
  { intro hλ
    rw [locus_type, if_pos hλ]
    refl }
  { intro hλ
    rw [locus_type, if_neg hλ]
    refl }
  sorry

end geometric_locus_l148_148338


namespace probability_of_at_most_3_heads_l148_148407

-- Definitions and conditions
def num_coins : ℕ := 10
def at_most_3_heads_probability : ℚ := 11 / 64

-- Statement of the problem
theorem probability_of_at_most_3_heads (n : ℕ) (p : ℚ) (h1 : n = num_coins) (h2 : p = at_most_3_heads_probability) :
  p = (1 + 10 + 45 + 120 : ℕ) / (2 ^ 10 : ℕ) := by
  sorry

end probability_of_at_most_3_heads_l148_148407


namespace geometric_locus_l148_148340

open Complex

theorem geometric_locus 
  (z z1 z2 : ℂ) (λ : ℝ) (hλ : λ > 0) :
  (λ = 1 → ∃ (line : Set ℂ), z ∈ line) ∧ 
  (λ ≠ 1 → ∃ (circle : Set ℂ), z ∈ circle) :=
  sorry

end geometric_locus_l148_148340


namespace probability_heads_at_most_3_of_10_coins_flipped_l148_148378

theorem probability_heads_at_most_3_of_10_coins_flipped :
  (∑ k in finset.range 4, nat.choose 10 k) / 2^10 = 11 / 64 :=
sorry

end probability_heads_at_most_3_of_10_coins_flipped_l148_148378


namespace arrangements_of_doctors_and_nurses_l148_148227

theorem arrangements_of_doctors_and_nurses (docs nurses schools : ℕ) 
  (h_docs : docs = 3) (h_nurses : nurses = 4) (h_schools : schools = 3) 
  (h_min_one_doc : True) (h_min_one_nurse : True) : 
  ∃ (arrangements : ℕ), arrangements = 216 := 
by
  have docs_arrangements : ℕ := Nat.factorial docs / Nat.factorial (docs - schools)
  have nurses_pairing : ℕ := Nat.choose nurses 2
  have nurses_arrangements : ℕ := docs_arrangements
  have total_arrangements : ℕ := docs_arrangements * nurses_pairing * nurses_arrangements
  have h : docs_arrangements = 6 := by sorry
  have h' : nurses_pairing = 6 := by sorry
  have h'' : nurses_arrangements = 6 := by sorry
  exists total_arrangements
  show total_arrangements = 216 by
    rw [h, h', h'']
    simp [total_arrangements]
    sorry

end arrangements_of_doctors_and_nurses_l148_148227


namespace difference_between_extrema_l148_148113

noncomputable def f (x a b : ℝ) : ℝ := x^3 + 3 * a * x^2 + 3 * b * x

theorem difference_between_extrema (a b : ℝ)
  (h1 : 3 * (2 : ℝ)^2 + 6 * a * (2 : ℝ) + 3 * b = 0)
  (h2 : 3 * (1 : ℝ)^2 + 6 * a * (1 : ℝ) + 3 * b = -3) :
  f 0 a b - f 2 a b = 4 :=
by
  sorry

end difference_between_extrema_l148_148113


namespace cos_B_proof_area_ABC_proof_l148_148758

variables {A B C a b c : ℝ}

-- Conditions
noncomputable def condition_b := (b = 4)
noncomputable def condition_c := (c = 6)
noncomputable def condition_C := (C = 2 * B)

-- Question 1: Prove cos B = 3/4
theorem cos_B_proof (h1 : condition_b) (h2 : condition_c) (h3 : condition_C) : cos B = 3/4 := by
  sorry

-- Question 2: Prove the area of triangle ABC is 15√7/4
theorem area_ABC_proof (h1 : condition_b) (h2 : condition_c) (h3 : condition_C) : 
  let S := (1/2) * b * c * (sin (A)) in S = 15 * real.sqrt 7 / 4 := by
  sorry

end cos_B_proof_area_ABC_proof_l148_148758


namespace find_probability_l148_148718

theorem find_probability (σ : ℝ)
  (hξ : ∀ x, (NormalDist.pdf 2 σ^2).cdf x = (cdf (NormalDist.mk 2 σ^2)) x)
  (hzero_prob : P(λ ξ, ξ < 0) = 0.08) :
  P(λ ξ, 0 < ξ ∧ ξ < 2) = 0.42 := 
sorry

end find_probability_l148_148718


namespace number_in_both_l148_148757

def x : Set ℤ := sorry  -- x is a set of integers
def y : Set ℤ := sorry  -- y is a set of integers
def sym_diff (A B : Set ℤ) : Set ℤ := {z | (z ∈ A ∧ z ∉ B) ∨ (z ∈ B ∧ z ∉ A)}  -- Symmetric difference definition

-- The conditions
axiom h₁ : Finset.card (x.to_finset) = 8
axiom h₂ : Finset.card (y.to_finset) = 18
axiom h₃ : Finset.card ((sym_diff x y).to_finset) = 14

-- The statement to be proved
theorem number_in_both : Finset.card ((x ∩ y).to_finset) = 6 :=
by
  sorry

end number_in_both_l148_148757


namespace cos2_alpha_add_sin2_alpha_eq_eight_over_five_l148_148768

theorem cos2_alpha_add_sin2_alpha_eq_eight_over_five (x y : ℝ) (r : ℝ) (α : ℝ) 
(hx : x = 2) 
(hy : y = 1)
(hr : r = Real.sqrt (x^2 + y^2))
(hcos : Real.cos α = x / r)
(hsin : Real.sin α = y / r) :
  Real.cos α ^ 2 + Real.sin (2 * α) = 8 / 5 :=
sorry

end cos2_alpha_add_sin2_alpha_eq_eight_over_five_l148_148768


namespace circle_diameter_segments_l148_148160

theorem circle_diameter_segments (r : ℝ) (h₁ : r = 7) 
  (h₂ : ∀ O A B C D K H : E, r = dist O A ∧ r = dist O B ∧ 
    dist AB C = 12 ∧ CD ⊥ AB ∧ CD ∋ O ∧ AB ∋ O ∧ CH ∋ K ∧ AB ∋ K ∧ 
    ∃ p₁ p₂ : ℝ, dist A K = p₁ ∧ dist B K = p₂) : 
  (∃ p₁ p₂ : ℝ, p₁ = sqrt 13 ∧ p₂ = 14 - sqrt 13) :=
begin
  sorry
end

end circle_diameter_segments_l148_148160


namespace arrangement_count_l148_148011

theorem arrangement_count (basil_plants tomato_plants : ℕ) (b : basil_plants = 5) (t : tomato_plants = 4) : 
  (Nat.factorial (basil_plants + 1) * Nat.factorial tomato_plants) = 17280 :=
by
  rw [b, t] 
  exact Eq.refl 17280

end arrangement_count_l148_148011


namespace range_f_range_a_l148_148729

-- Part (1) statement
theorem range_f (x : ℝ) (h : 1 ≤ x ∧ x ≤ 3) :
  ∃ (y : ℝ), y ∈ Icc (-1/3) (-1/4) ∧ y = x / (x^2 - 8*x + 4) :=
sorry

-- Part (2) statement
theorem range_a (x1 x2 : ℝ) (h1 : 1 ≤ x1 ∧ x1 ≤ 3) (h2 : 0 ≤ x2 ∧ x2 ≤ 1) :
  ∃ (a : ℝ), 
  (g : ℝ → ℝ := λ x => x^2 + 2*a*x - a^2) ∧ 
  (fx : ℝ → ℝ := λ x => x / (x^2 - 8*x + 4)) ∧ 
  g(x2) * fx(x1) = 1 ∧ 
  (a ∈ Icc (-√3) (1 - √6) ∨ a ∈ Icc 2 (1 + √5)) :=
sorry

end range_f_range_a_l148_148729


namespace sally_bread_consumption_l148_148835

theorem sally_bread_consumption :
  (2 * 2) + (1 * 2) = 6 :=
by
  sorry

end sally_bread_consumption_l148_148835


namespace bob_total_candies_l148_148971

def total_chewing_gums : ℕ := 45
def total_chocolate_bars : ℕ := 60
def total_assorted_candies : ℕ := 45

def chewing_gums_ratio_sam_bob := (2, 3)
def chocolate_bars_ratio_sam_bob := (3, 1)
def assorted_candies_ratio_sam_bob := (1, 1)

def bob_additional_reward := 5
def bob_reward_condition := 20

theorem bob_total_candies : 
  let _
      total_chewing_gums := total_chewing_gums,
      total_chocolate_bars := total_chocolate_bars,
      total_assorted_candies := total_assorted_candies,
      (chewing_gums_ratio_sam, chewing_gums_ratio_bob) := chewing_gums_ratio_sam_bob,
      (chocolate_bars_ratio_sam, chocolate_bars_ratio_bob) := chocolate_bars_ratio_sam_bob,
      (assorted_candies_ratio_sam, assorted_candies_ratio_bob) := assorted_candies_ratio_sam_bob,
      bob_chewing_gums := (total_chewing_gums * chewing_gums_ratio_bob) / (chewing_gums_ratio_sam + chewing_gums_ratio_bob),
      bob_chocolate_bars := (total_chocolate_bars * chocolate_bars_ratio_bob) / (chocolate_bars_ratio_sam + chocolate_bars_ratio_bob),
      bob_assorted_candies := (total_assorted_candies * assorted_candies_ratio_bob) / (assorted_candies_ratio_sam + assorted_candies_ratio_bob),
      bob_reward := if bob_chocolate_bars >= bob_reward_condition then bob_additional_reward else 0,
      bob_total := bob_chewing_gums + bob_chocolate_bars + bob_assorted_candies + bob_reward
  in bob_total = 64 :=
by sorry

end bob_total_candies_l148_148971


namespace greatest_n_l148_148331

theorem greatest_n (n : ℕ) (a : ℕ → ℕ) : 
  (a 1 = 1) ∧ (a n = 2020) ∧ (∀ i : ℕ, (2 ≤ i ∧ i ≤ n) → (a i - a (i - 1) = -2 ∨ a i - a (i - 1) = 3)) → n = 2019 :=
begin
  sorry
end

end greatest_n_l148_148331


namespace maximum_q_minus_r_l148_148862

theorem maximum_q_minus_r : 
  ∀ q r : ℕ, (1027 = 23 * q + r) ∧ (q > 0) ∧ (r > 0) → q - r ≤ 29 := 
by
  sorry

end maximum_q_minus_r_l148_148862


namespace sum_largest_smallest_digits_l148_148326

theorem sum_largest_smallest_digits : 
  let digits := {0, 2, 4, 6}
  let largest_number := 642
  let smallest_number := 204
  largest_number + smallest_number = 846 :=
by
  -- Define the digits and their properties
  let digits := {0, 2, 4, 6}
  let largest_number := 642
  let smallest_number := 204
  -- Sum the largest and smallest numbers and prove the result
  show 642 + 204 = 846
  sorry

end sum_largest_smallest_digits_l148_148326


namespace operation_result_l148_148042

def operation (a b : ℤ) : ℤ := a * (b + 2) + a * b

theorem operation_result : operation 3 (-1) = 0 :=
by
  sorry

end operation_result_l148_148042


namespace walt_total_interest_l148_148891

noncomputable def interest_8_percent (P_8 R_8 : ℝ) : ℝ :=
  P_8 * R_8

noncomputable def remaining_amount (P_total P_8 : ℝ) : ℝ :=
  P_total - P_8

noncomputable def interest_9_percent (P_9 R_9 : ℝ) : ℝ :=
  P_9 * R_9

noncomputable def total_interest (I_8 I_9 : ℝ) : ℝ :=
  I_8 + I_9

theorem walt_total_interest :
  let P_8 := 4000
  let R_8 := 0.08
  let P_total := 9000
  let R_9 := 0.09
  let I_8 := interest_8_percent P_8 R_8
  let P_9 := remaining_amount P_total P_8
  let I_9 := interest_9_percent P_9 R_9
  let I_total := total_interest I_8 I_9
  I_total = 770 := 
by
  sorry

end walt_total_interest_l148_148891


namespace area_shaded_region_l148_148017

theorem area_shaded_region (a1 a2 b1 b2 : ℝ) 
  (ha1 : a1 = 8) (ha2 : a2 = 5) 
  (hb1 : b1 = 6) (hb2 : b2 = 2) : 
  let rect_area := (a1 + b1) * (a2 + b2),
      tri_a_area := (1 / 2) * a1 * a2,
      tri_b_area := (1 / 2) * b1 * b2,
      total_tri_area := tri_a_area + tri_b_area,
      shaded_area := rect_area - total_tri_area in
  shaded_area = 74 :=
by
  sorry

end area_shaded_region_l148_148017


namespace ordered_triple_solution_satisfies_conditions_l148_148057

noncomputable def solution := (130 / 161, 76 / 23, 3)

theorem ordered_triple_solution_satisfies_conditions :
  let (x, y, z) := solution in
  7 * x - 3 * y + 2 * z = 4 ∧ 
  4 * y - x - 5 * z = -3 ∧ 
  3 * x + 2 * y - z = 7 :=
by
  let (x, y, z) := solution
  have h1 : 7 * x - 3 * y + 2 * z = 4 := sorry
  have h2 : 4 * y - x - 5 * z = -3 := sorry
  have h3 : 3 * x + 2 * y - z = 7 := sorry
  exact ⟨h1, h2, h3⟩

end ordered_triple_solution_satisfies_conditions_l148_148057


namespace chandler_bike_purchase_l148_148628

theorem chandler_bike_purchase :
  ∀ (cost_bike grandparents aunt cousin weekly_earnings : ℕ),
    cost_bike = 600 →
    grandparents = 60 →
    aunt = 40 →
    cousin = 20 →
    weekly_earnings = 20 →
    (∃ x : ℕ, 120 + 20 * x = 600 ∧ x = 24) :=
by
  intros cost_bike grandparents aunt cousin weekly_earnings
  intros h_cost_bike h_grandparents h_aunt h_cousin h_weekly_earnings
  use 24
  simp [h_cost_bike, h_grandparents, h_aunt, h_cousin, h_weekly_earnings]
  sorry

end chandler_bike_purchase_l148_148628


namespace inverse_of_f_128_l148_148691

noncomputable def f : ℕ → ℕ :=
  sorry -- The definition of f is noncomputable based on the conditions provided

axiom f_at_4 : f 4 = 2
axiom f_doubling : ∀ x, f (2 * x) = 2 * f x

theorem inverse_of_f_128 : f⁻¹ 128 = 256 :=
by
  sorry

end inverse_of_f_128_l148_148691


namespace biased_coin_prob_three_heads_l148_148899

def prob_heads := 1/3

theorem biased_coin_prob_three_heads : prob_heads^3 = 1/27 :=
by
  sorry

end biased_coin_prob_three_heads_l148_148899


namespace calculate_expression_l148_148623

theorem calculate_expression : 2^345 + 9^5 / 9^3 = 2^345 + 81 :=
by
  have exp_div : 9^5 / 9^3 = 81 := by
    rw [←pow_sub (by norm_num : (9 : ℝ) ≠ 0), (show 5 - 3 = 2 by norm_num)]
    norm_num
  rw exp_div
  rfl

end calculate_expression_l148_148623


namespace probability_x_lt_y_lt_3x_l148_148584

noncomputable theory
open Real Set MeasureTheory

-- Define the two random points on the interval [0, 1]
def point_on_unit_interval : MeasureTheory.MeasureSpace (ℝ → ℝ) :=
  MeasureTheory.MeasureSpace.mk volume

/--
Given two random points \( x \) and \( y \) chosen independently from the interval \([0,1]\),
prove that the probability that \( x < y < 3x \) is \( \frac{1}{3} \).
-/
theorem probability_x_lt_y_lt_3x :
  ∀ (x y : ℝ), 0 ≤ x ∧ x ≤ 1 ∧ 0 ≤ y ∧ y ≤ 1 ∧ x < y ∧ y < 3 * x →  measure_space.measure.to_real (volume) = 1/3 :=
sorry

end probability_x_lt_y_lt_3x_l148_148584


namespace diamond_expression_evaluation_l148_148645

def diamond (a b : ℚ) : ℚ := a - (1 / b)

theorem diamond_expression_evaluation :
  ((diamond (diamond 2 3) 4) - (diamond 2 (diamond 3 4))) = -29 / 132 :=
by {
    sorry
}

end diamond_expression_evaluation_l148_148645


namespace domain_f_l148_148852

noncomputable def f (x : ℝ) : ℝ := sqrt (x - 1) / (x - 3)

theorem domain_f :
  ∀ x : ℝ, (x ≥ 1 ∧ x ≠ 3) ↔ (x ∈ (set.Ico 1 3) ∨ x ∈ (set.Ioi 3)) :=
by
  sorry

end domain_f_l148_148852


namespace probability_of_at_most_3_heads_l148_148399

-- Definitions and conditions
def num_coins : ℕ := 10
def at_most_3_heads_probability : ℚ := 11 / 64

-- Statement of the problem
theorem probability_of_at_most_3_heads (n : ℕ) (p : ℚ) (h1 : n = num_coins) (h2 : p = at_most_3_heads_probability) :
  p = (1 + 10 + 45 + 120 : ℕ) / (2 ^ 10 : ℕ) := by
  sorry

end probability_of_at_most_3_heads_l148_148399


namespace probability_of_at_most_3_heads_out_of_10_l148_148489
open_locale big_operators
open nat

theorem probability_of_at_most_3_heads_out_of_10 (n : ℕ) (p : ℕ) (h : 0 < n) (hp : p = 2) :
  ∑ k in finset.range (4), nat.choose n k = 176 → nat.pow p n = 1024 → (∑ k in finset.range (4), nat.choose n k) / (nat.pow p n) = 11 / 64 :=
by { intro h₁ h₂, 
     have : (∑ k in finset.range 4, nat.choose n k) / (nat.pow p n) = 176 / 1024,
     { rw [h₁, h₂] },
     rw this,
     norm_num }

end probability_of_at_most_3_heads_out_of_10_l148_148489


namespace mc_area_correct_l148_148287

variables {A M C V U : Type*}
variables [EuclideanSpace A M C V U]

def is_isosceles (AM AC : ℝ) : Prop := AM = AC
def medians_perpendicular (MV CU : ℝ) : Prop := MV = CU ∧ MV = 15 ∧ ⟪MV, CU⟫ = 0

noncomputable def area_triangle_MC (MV CU : ℝ) (hMV : MV = CU) (hCM : medians_perpendicular MV CU) : ℝ :=
  4 * (1/2 * MV * CU)

theorem mc_area_correct (AM AC MV CU : ℝ) (h : is_isosceles AM AC) (h' : medians_perpendicular MV CU) :
  area_triangle_MC MV CU (h'.1) h' = 450 :=
sorry

end mc_area_correct_l148_148287


namespace coefficient_of_x31_is_148_l148_148984

theorem coefficient_of_x31_is_148 (x : ℝ) : 
  (coeff ((1 - x ^ 31) * (1 - x ^ 13) ^ 2 / (1 - x) ^ 3) 31 = 148) :=
sorry

end coefficient_of_x31_is_148_l148_148984


namespace circle_coloring_l148_148808

noncomputable theory

def colorable (R : set (ℝ × ℝ)) (d : ℝ) : Prop :=
∃ (c : (ℝ × ℝ) → ℕ), ∀ (x y : ℝ × ℝ), (x ∈ R ∧ y ∈ R ∧ c x = c y) → dist x y < d

theorem circle_coloring (R : set (ℝ × ℝ)) (hR : ∀ p, p ∈ R ↔ (p.1)^2 + (p.2)^2 ≤ 1) :
  ∀ d, (∀ (c : (ℝ × ℝ) → ℕ), ∃ (x y : ℝ × ℝ), x ∈ R ∧ y ∈ R ∧ c x = c y ∧ dist x y ≥ d) ↔ d = sqrt 3 :=
by
  sorry

end circle_coloring_l148_148808


namespace max_integer_value_l148_148752

theorem max_integer_value (x : ℝ) : 
  ∃ m : ℤ, ∀ (x : ℝ), (3 * x^2 + 9 * x + 17) / (3 * x^2 + 9 * x + 7) ≤ m ∧ m = 41 :=
by sorry

end max_integer_value_l148_148752


namespace number_of_numbers_l148_148846

theorem number_of_numbers 
  (avg : ℚ) (avg1 : ℚ) (avg2 : ℚ) (avg3 : ℚ)
  (h_avg : avg = 4.60) 
  (h_avg1 : avg1 = 3.4) 
  (h_avg2 : avg2 = 3.8) 
  (h_avg3 : avg3 = 6.6) 
  (h_sum_eq : 2 * avg1 + 2 * avg2 + 2 * avg3 = 27.6) : 
  (27.6 / avg = 6) := 
  by sorry

end number_of_numbers_l148_148846


namespace prob_heads_at_most_3_out_of_10_flips_l148_148456

def coin_flips := 10
def max_heads := 3
def total_outcomes := 2^coin_flips
def favorable_outcomes := (nat.choose coin_flips 0) +
                          (nat.choose coin_flips 1) +
                          (nat.choose coin_flips 2) +
                          (nat.choose coin_flips 3)

theorem prob_heads_at_most_3_out_of_10_flips :
  (favorable_outcomes : ℚ) / total_outcomes = 11 / 64 :=
by
  sorry

end prob_heads_at_most_3_out_of_10_flips_l148_148456


namespace to_ellipse_area_correct_l148_148759

/-
Define the endpoints of the major axis of the ellipse.
-/
def p1 : ℝ × ℝ := (-5, 3)
def p2 : ℝ × ℝ := (15, 3)

/-
Define the point through which the ellipse passes.
-/
def pointOnEllipse : ℝ × ℝ := (10, 10)

/-
Calculate the midpoint as the center of the ellipse.
-/
def center : ℝ × ℝ := ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

/-
Calculate the semi-major axis length.
-/
def a : ℝ := (Real.abs (p2.1 - p1.1)) / 2

/-
Calculate b^2 from the given ellipse equation and point.
-/
def bSquare : ℝ := (49 * 4) / 3

/-
The area of the ellipse is given by the formula π * a * b, where b = sqrt(bSquare).
-/
noncomputable def area : ℝ := Real.pi * a * Real.sqrt bSquare

/-
Theorem to prove that the calculated area is equal to the given correct answer.
-/
theorem ellipse_area_correct : area = (140 * Real.pi / 3) * Real.sqrt 3 := sorry

end to_ellipse_area_correct_l148_148759


namespace minimum_value_of_expr_l148_148680

def expr (x : ℝ) : ℝ :=
  1 + (Real.cos ((π * Real.sin (2 * x)) / Real.sqrt 3))^2 + 
      (Real.sin (2 * Real.sqrt 3 * π * Real.cos x))^2

theorem minimum_value_of_expr :
  ∀ x, x ∈ { x | ∃ k : ℤ, x = π * k + π / 6 ∨ x = π * k - π / 6} ∧
  expr x = 1 :=
begin
  sorry
end

end minimum_value_of_expr_l148_148680


namespace vectors_are_perpendicular_l148_148125

variable (a b : ℝ × ℝ)
variable (a_value : a = (-2, 1))
variable (b_value : b = (-1, 3))

theorem vectors_are_perpendicular :
  let c := (a.1 - b.1, a.2 - b.2) in
  a.1 * c.1 + a.2 * c.2 = 0 :=
by
  rw [a_value, b_value]
  let c := (-1, -2)
  have h1 : (-2:ℝ) * (-1) = 2 := by norm_num
  have h2 : (1:ℝ) * (-2) = -2 := by norm_num
  rw [mul_add, h1, h2]
  norm_num
  sorry

end vectors_are_perpendicular_l148_148125


namespace exists_K7_subgraph_l148_148278

theorem exists_K7_subgraph (G : SimpleGraph (Fin 2023)) (h_regular : ∀ v : Fin 2023, G.degree v = 1686)
  (h_symmetric : ∀ v w : Fin 2023, (G.adj v w ↔ G.adj w v)) :
  ∃ H : SimpleGraph (Fin 7), H.complete :=
by sorry

end exists_K7_subgraph_l148_148278


namespace trajectory_of_P_is_parabola_l148_148703

theorem trajectory_of_P_is_parabola 
  (M : ℝ × ℝ) (l : ℝ → Prop) (B : ℝ × ℝ)
  (hM : M = (1, 0))
  (hl : ∀ x : ℝ, l x ↔ x = -1)
  (hB : l (fst B))
  (P : ℝ × ℝ) :
  ∃ f : ℝ → ℝ, ∀ t : ℝ, (P = (t, f t)) ∧ 
  (dist P M = dist P B) → 
  ∀ y : ℝ, ∃ x : ℝ, P = (x, y) ∧ y = (x - 1) * (x + 1) :=
begin
  sorry
end

end trajectory_of_P_is_parabola_l148_148703


namespace probability_at_most_3_heads_10_coins_l148_148416

noncomputable def coin_flip_prob_at_most_3_heads : ℚ :=
  let total_outcomes := 2^10
  let favorable_outcomes := Nat.binomial 10 0 + Nat.binomial 10 1 + Nat.binomial 10 2 + Nat.binomial 10 3
  favorable_outcomes / total_outcomes

theorem probability_at_most_3_heads_10_coins : coin_flip_prob_at_most_3_heads = 11 / 64 :=
by
  -- Definitions of let values are integrated directly in the theorem body
  let total_outcomes := 2^10
  let favorable_outcomes := Nat.binomial 10 0 + Nat.binomial 10 1 + Nat.binomial 10 2 + Nat.binomial 10 3
  show favorable_outcomes / total_outcomes = 11 / 64
  sorry

end probability_at_most_3_heads_10_coins_l148_148416


namespace circle_cut_by_parabolas_l148_148785

theorem circle_cut_by_parabolas (n : ℕ) (h : n = 10) : 
  2 * n ^ 2 + 1 = 201 :=
by
  sorry

end circle_cut_by_parabolas_l148_148785


namespace systematic_sampling_8th_group_l148_148592

-- Define the conditions
def total_employees : ℕ := 200
def sample_size : ℕ := 40
def grouping_interval : ℕ := 5
def N5 : ℕ := 23

-- Define the hypothesis that number drawn from the 5th group gives N5
def first_group_number := l : ℕ
axiom group5_eq_23 : first_group_number + 4 * grouping_interval = N5

-- Formulate the theorem for the 8th group
theorem systematic_sampling_8th_group : first_group_number + 7 * grouping_interval = 38 :=
by
  sorry

end systematic_sampling_8th_group_l148_148592


namespace find_larger_integer_l148_148336

-- Defining the problem statement with the given conditions
theorem find_larger_integer (x : ℕ) (h : (x + 6) * 2 = 4 * x) : 4 * x = 24 :=
sorry

end find_larger_integer_l148_148336


namespace probability_heads_at_most_3_l148_148532

theorem probability_heads_at_most_3 (num_coins : ℕ) (num_heads_at_most : ℕ) (probability : ℚ):
  num_coins = 10 → 
  num_heads_at_most = 3 → 
  probability = (binomial 10 0 + binomial 10 1 + binomial 10 2 + binomial 10 3) / (2^10) →
  probability = (11 / 64) := 
by
  intros hnum_coins hnum_heads_at_most hprobability
  sorry

end probability_heads_at_most_3_l148_148532


namespace maximum_q_minus_r_l148_148863

theorem maximum_q_minus_r : 
  ∀ q r : ℕ, (1027 = 23 * q + r) ∧ (q > 0) ∧ (r > 0) → q - r ≤ 29 := 
by
  sorry

end maximum_q_minus_r_l148_148863


namespace line_intersects_circle_l148_148104

noncomputable def circleC : set (ℝ × ℝ) :=
{ p | let (x, y) := p in x^2 + y^2 - 4 * x = 0 }

noncomputable def lineL (k : ℝ) : set (ℝ × ℝ) :=
{ p | let (x, y) := p in y = k * x - 3 * k + 1 }

theorem line_intersects_circle (k : ℝ) :
  ∃ (p : ℝ × ℝ), p ∈ circleC ∧ p ∈ lineL k :=
sorry

end line_intersects_circle_l148_148104


namespace probability_10_coins_at_most_3_heads_l148_148383

def probability_at_most_3_heads (n : ℕ) : ℚ :=
  let total_outcomes := 2^n
  let favorable_outcomes := (Nat.choose n 0) + (Nat.choose n 1) + (Nat.choose n 2) + (Nat.choose n 3)
  favorable_outcomes / total_outcomes

theorem probability_10_coins_at_most_3_heads : probability_at_most_3_heads 10 = 11 / 64 :=
by
  sorry

end probability_10_coins_at_most_3_heads_l148_148383


namespace sin_cos_identity_l148_148350

theorem sin_cos_identity (x : ℝ) (h : log 5 (sin x) + log 5 (cos x) = -1) :
  abs (sin x ^ 2 * cos x + cos x ^ 2 * sin x) = real.sqrt 35 / 25 :=
by sorry

end sin_cos_identity_l148_148350


namespace probability_at_most_3_heads_10_coins_l148_148417

noncomputable def coin_flip_prob_at_most_3_heads : ℚ :=
  let total_outcomes := 2^10
  let favorable_outcomes := Nat.binomial 10 0 + Nat.binomial 10 1 + Nat.binomial 10 2 + Nat.binomial 10 3
  favorable_outcomes / total_outcomes

theorem probability_at_most_3_heads_10_coins : coin_flip_prob_at_most_3_heads = 11 / 64 :=
by
  -- Definitions of let values are integrated directly in the theorem body
  let total_outcomes := 2^10
  let favorable_outcomes := Nat.binomial 10 0 + Nat.binomial 10 1 + Nat.binomial 10 2 + Nat.binomial 10 3
  show favorable_outcomes / total_outcomes = 11 / 64
  sorry

end probability_at_most_3_heads_10_coins_l148_148417


namespace min_s8_value_l148_148202

theorem min_s8_value
  (p : ℕ) (hp : Nat.Prime p) (h_p_eq : p = 9001)
  (A B : Finset (ZMod p)) 
  (s : Fin 9 → ℕ) (hs : ∀ i, 2 ≤ s i) 
  (T : Fin 9 → Finset (ZMod p)) (hT : ∀ i, T i.card = s i)
  (H7 : Finset.card ((Finset.range 7).bind (λ i, T i)) < p) 
  (H8 : (Finset.range 8).bind (λ i, T i) = Finset.univ) 
  : s 8 = 2 := by
  sorry

end min_s8_value_l148_148202


namespace backyard_max_area_l148_148235

def max_area_backyard (fencing_length : ℕ) : ℕ :=
  let l w : ℕ := sorry in
  let area (w : ℕ) := (fencing_length - 2 * w) * w in
  -2 * (w - 90)^2 + 16200

theorem backyard_max_area (fencing_length : ℕ) (h : fencing_length = 360) :
  max_area_backyard fencing_length = 16200 := sorry

end backyard_max_area_l148_148235


namespace crabapple_recipients_sequence_count_l148_148816

/--
Mrs. Crabapple teaches a class of 13 students and holds class sessions five times a week. 
She chooses a different student at each meeting to receive a crabapple, ensuring that no student 
gets picked more than once in a week. Prove that the number of different sequences of crabapple 
recipients possible in one week is 154440.
-/
theorem crabapple_recipients_sequence_count : 
    ∃ sequences : ℕ, sequences = 13 * 12 * 11 * 10 * 9 ∧ sequences = 154440 :=
by 
  use 13 * 12 * 11 * 10 * 9
  simp
  sorry

end crabapple_recipients_sequence_count_l148_148816


namespace tangent_line_at_A_trajectory_of_midpoint_l148_148720

-- Definitions
def curve_c (x y : ℝ) : Prop :=
  y^2 = 2 * x - 4

def tangent_line (x y : ℝ) : Prop :=
  x - real.sqrt 2 * y - 1 = 0

def midpoint_trajectory (x y : ℝ) : Prop :=
  y^2 = x ∧ x > 4

-- Proof statements
theorem tangent_line_at_A :
  tangent_line 3 (real.sqrt 2) :=
sorry

theorem trajectory_of_midpoint :
  ∀ (k : ℝ), (0 < k) →
  let x := 1 / (k^2),
      y := 1 / k in
  midpoint_trajectory x y :=
sorry

end tangent_line_at_A_trajectory_of_midpoint_l148_148720


namespace player_winning_strategy_l148_148080

-- Define the game conditions
def Sn (n : ℕ) : Type := Equiv.Perm (Fin n)

def game_condition (n : ℕ) : Prop :=
  n > 1 ∧ (∀ G : Set (Sn n), ∃ x : Sn n, x ∈ G → G ≠ (Set.univ : Set (Sn n)))

-- Statement of the proof problem
theorem player_winning_strategy (n : ℕ) (hn : n > 1) : 
  ((n = 2 ∨ n = 3) → (∃ strategyA : Sn n → (Sn n → Prop), ∀ x : Sn n, strategyA x x)) ∧ 
  ((n ≥ 4 ∧ n % 2 = 1) → (∃ strategyB : Sn n → (Sn n → Prop), ∀ x : Sn n, strategyB x x)) :=
by
  sorry

end player_winning_strategy_l148_148080


namespace rent_for_each_room_l148_148594

theorem rent_for_each_room (x : ℝ) (ha : 4800 / x = 4200 / (x - 30)) (hx : x = 240) :
  x = 240 ∧ (x - 30) = 210 :=
by
  sorry

end rent_for_each_room_l148_148594


namespace entrance_fee_increase_l148_148853

theorem entrance_fee_increase
  (entrance_fee_under_18 : ℕ)
  (rides_cost : ℕ)
  (num_rides : ℕ)
  (total_spent : ℕ)
  (total_cost_twins : ℕ)
  (total_ride_cost_twins : ℕ)
  (amount_spent_joe : ℕ)
  (total_ride_cost_joe : ℕ)
  (joe_entrance_fee : ℕ)
  (increase : ℕ)
  (percentage_increase : ℕ)
  (h1 : entrance_fee_under_18 = 5)
  (h2 : rides_cost = 50) -- representing $0.50 as 50 cents to maintain integer calculations
  (h3 : num_rides = 3)
  (h4 : total_spent = 2050) -- representing $20.5 as 2050 cents
  (h5 : total_cost_twins = 1300) -- combining entrance fees and cost of rides for the twins in cents
  (h6 : total_ride_cost_twins = 300) -- cost of rides for twins in cents
  (h7 : amount_spent_joe = 750) -- representing $7.5 as 750 cents
  (h8 : total_ride_cost_joe = 150) -- cost of rides for Joe in cents
  (h9 : joe_entrance_fee = 600) -- representing $6 as 600 cents
  (h10 : increase = 100) -- increase in entrance fee in cents
  (h11 : percentage_increase = 20) :
  percentage_increase = ((increase * 100) / entrance_fee_under_18) :=
sorry

end entrance_fee_increase_l148_148853


namespace fencing_cost_correct_l148_148624

def major_axis : ℝ := 30
def minor_axis : ℝ := 20
def cost_per_meter : ℝ := 5

noncomputable def semi_major_axis : ℝ := major_axis / 2
noncomputable def semi_minor_axis : ℝ := minor_axis / 2

noncomputable def ramanujan_perimeter (a b : ℝ) : ℝ :=
  Real.pi * (3*(a + b) - real.sqrt((3*a + b) * (a + 3*b)))

noncomputable def estimated_perimeter : ℝ :=
  ramanujan_perimeter semi_major_axis semi_minor_axis

noncomputable def total_cost : ℝ :=
  estimated_perimeter * cost_per_meter

theorem fencing_cost_correct : total_cost = 396.90 :=
by
  sorry

end fencing_cost_correct_l148_148624


namespace edge_length_of_cubic_block_l148_148074

-- Define the conditions: base area and height of the copper pentagonal prism
def base_area (A : ℝ) : Prop := A = 16
def height (h : ℝ) : Prop := h = 4

-- Define the volume of the copper pentagonal prism
def volume_prism (V : ℝ) (A h : ℝ) : Prop := V = A * h

-- Define the edge length of the resulting cubic copper block
def edge_length (a : ℝ) (V : ℝ) : Prop := a^3 = V

-- Proof statement that the edge length of the resulting cubic copper block is 4 cm
theorem edge_length_of_cubic_block : ∀ (A h V a: ℝ), base_area A → height h → volume_prism V A h → edge_length a V → a = 4 :=
by
  intro A h V a A_base_area A_height V_prism a_edge_length
  sorry

end edge_length_of_cubic_block_l148_148074


namespace probability_heads_at_most_3_of_10_coins_flipped_l148_148371

theorem probability_heads_at_most_3_of_10_coins_flipped :
  (∑ k in finset.range 4, nat.choose 10 k) / 2^10 = 11 / 64 :=
sorry

end probability_heads_at_most_3_of_10_coins_flipped_l148_148371


namespace sum_in_range_l148_148019

open Real

def mix1 := 3 + 3/8
def mix2 := 4 + 2/5
def mix3 := 6 + 1/11
def mixed_sum := mix1 + mix2 + mix3

theorem sum_in_range : mixed_sum > 13 ∧ mixed_sum < 14 :=
by
  -- Since we are just providing the statement, we leave the proof as a placeholder.
  sorry

end sum_in_range_l148_148019


namespace identify_universal_proposition_l148_148600

-- Definitions of four propositions using conditions from the problem
def prop_A : Prop := ∃ x : ℝ, irrational x
def prop_B : Prop := ∃ n : ℤ, ¬ (n % 3 = 0)
def prop_C : Prop := ∀ f : ℝ → ℝ, (even_function f → symmetric_about_y_axis f)
def prop_D : Prop := ∃ T : Triangle, ¬ (right_triangle T)

-- Defining what is to be proven: prop_C is the universal proposition among the given propositions
theorem identify_universal_proposition :
  prop_C ∧ ¬ prop_A ∧ ¬ prop_B ∧ ¬ prop_D := 
sorry

end identify_universal_proposition_l148_148600


namespace probability_at_most_3_heads_l148_148436

def binom (n k : ℕ) : ℕ := Nat.choose n k

theorem probability_at_most_3_heads :
  let total_outcomes := 2^10
  let successful_outcomes := binom 10 0 + binom 10 1 + binom 10 2 + binom 10 3
  (successful_outcomes : ℚ) / total_outcomes = 11 / 64 := 
by 
  let total_outcomes := 2 ^ 10
  let successful_outcomes := binom 10 0 + binom 10 1 + binom 10 2 + binom 10 3
  have h : (successful_outcomes : ℚ) / total_outcomes = 11 / 64 := sorry
  exact h

end probability_at_most_3_heads_l148_148436


namespace vector_magnitude_proof_l148_148777

variables (a b c : ℝ × ℝ)
variables (λ μ : ℝ)

def magnitude (v : ℝ × ℝ) : ℝ := real.sqrt (v.1^2 + v.2^2)

def condition1 : Prop := ∥a∥ = 1 ∧ ∥b∥ = 1
def condition2 : Prop := magnitude c = 2 * real.sqrt 3
def condition3 : Prop := c = (λ, μ)

theorem vector_magnitude_proof
  (h1 : condition1)
  (h2 : condition2)
  (h3 : condition3) :
  λ^2 + μ^2 = 12 :=
by
  sorry

end vector_magnitude_proof_l148_148777


namespace fraction_value_l148_148981

theorem fraction_value :
  (12^4 + 400) * (24^4 + 400) * (36^4 + 400) * (48^4 + 400) * (60^4 + 400) / 
  ((6^4 + 400) * (18^4 + 400) * (30^4 + 400) * (42^4 + 400) * (54^4 + 400)) = 244.375 :=
by
  -- The proof would be provided here, but we are skipping it as per the instructions.
  sorry

end fraction_value_l148_148981


namespace prob_heads_at_most_3_out_of_10_flips_l148_148460

def coin_flips := 10
def max_heads := 3
def total_outcomes := 2^coin_flips
def favorable_outcomes := (nat.choose coin_flips 0) +
                          (nat.choose coin_flips 1) +
                          (nat.choose coin_flips 2) +
                          (nat.choose coin_flips 3)

theorem prob_heads_at_most_3_out_of_10_flips :
  (favorable_outcomes : ℚ) / total_outcomes = 11 / 64 :=
by
  sorry

end prob_heads_at_most_3_out_of_10_flips_l148_148460


namespace total_hours_worked_l148_148000

theorem total_hours_worked (amber_hours : ℕ) (armand_hours : ℕ) (ella_hours : ℕ) 
(h_amber : amber_hours = 12) 
(h_armand : armand_hours = (1 / 3) * amber_hours) 
(h_ella : ella_hours = 2 * amber_hours) :
amber_hours + armand_hours + ella_hours = 40 :=
sorry

end total_hours_worked_l148_148000


namespace cost_formula_l148_148850

-- Given Conditions
def flat_fee := 5  -- flat service fee in cents
def first_kg_cost := 12  -- cost for the first kilogram in cents
def additional_kg_cost := 5  -- cost for each additional kilogram in cents

-- Integer weight in kilograms
variable (P : ℕ)

-- Total cost calculation proof problem
theorem cost_formula : ∃ C, C = flat_fee + first_kg_cost + additional_kg_cost * (P - 1) → C = 5 * P + 12 :=
by
  sorry

end cost_formula_l148_148850


namespace geometry_propositions_correct_answer_l148_148322

theorem geometry_propositions_correct_answer :
  (let A := "There is only one line passing through a point that is parallel to a given line."
       B := "The length of the perpendicular segment from a point outside a line to the line is called the distance from the point to the line."
       C := "Among all the line segments connecting two points, the shortest one is the line segment."
       D := "If two lines are parallel, the interior angles on the same side are equal."
       P_A := True
       P_B := True
       P_C := True
       P_D := False
   in P_C) :=
by {
  let A := "There is only one line passing through a point that is parallel to a given line."
  let B := "The length of the perpendicular segment from a point outside a line to the line is called the distance from the point to the line."
  let C := "Among all the line segments connecting two points, the shortest one is the line segment."
  let D := "If two lines are parallel, the interior angles on the same side are equal."
  let P_A := True
  let P_B := True
  let P_C := True
  let P_D := False
  exact P_C
}

end geometry_propositions_correct_answer_l148_148322


namespace all_terms_are_integers_l148_148962

variable {α : Type} [LinearOrderedField α]

def is_arithmetic_progression (a d : α) (u : ℕ → α) : Prop :=
∀ n, u n = a + n * d

def product_in_progression (u : ℕ → α) : Prop :=
∀ i j, i ≠ j → ∃ k, u i * u j = u k

theorem all_terms_are_integers {a d : ℕ} (u : ℕ → ℕ)
  (h1 : is_arithmetic_progression a d u)
  (h2 : product_in_progression u) :
  ∀ n, ∃ m, u n = m := 
sorry

end all_terms_are_integers_l148_148962


namespace seq_geometric_and_formula_sum_b_formula_c_lambda_lambda_gt_neg_3_l148_148077

noncomputable def S (n : ℕ) (a : ℕ → ℕ) : ℕ := 2 * a n - n

def is_geometric (S : ℕ → ℕ → ℕ) (a : ℕ → ℕ) :=
∀ n : ℕ, n > 0 → S n a = 2^(n+1) - 2

theorem seq_geometric_and_formula {a : ℕ → ℕ} :
  (∀ n : ℕ, n > 0 → S n a = 2 * (a n) - n) →
  ∀ n : ℕ, n > 0 → is_geometric S a :=
begin
  sorry
end

def b (n : ℕ) (a : ℕ → ℕ) := (2 * n + 1) * a n + 2 * n + 1

def sum_b (n : ℕ) (a : ℕ → ℕ) := 
  (2 * n - 1) * 2^(n+1) + 2

theorem sum_b_formula {a : ℕ → ℕ} (n : ℕ) :
  (∀ n : ℕ, n > 0 → S n a = 2 * (a n) - n) →
  ∀ n : ℕ, n > 0 → sum_b n a = (2 * n - 1) * 2^(n+1) + 2 :=
begin
  sorry
end

def c (n : ℕ) (a : ℕ → ℕ) (λ : ℤ) := 
  3^n + λ * (2^n)

theorem c_lambda_lambda_gt_neg_3 {a : ℕ → ℕ} (λ : ℤ) :
  (∀ n : ℕ, n > 0 → S n a = 2 * (a n) - n) →
  λ > -3 →
  ∀ n : ℕ, n > 0 → c(n + 1) a λ > c n a λ :=
begin
  sorry
end

end seq_geometric_and_formula_sum_b_formula_c_lambda_lambda_gt_neg_3_l148_148077


namespace probability_10_coins_at_most_3_heads_l148_148394

def probability_at_most_3_heads (n : ℕ) : ℚ :=
  let total_outcomes := 2^n
  let favorable_outcomes := (Nat.choose n 0) + (Nat.choose n 1) + (Nat.choose n 2) + (Nat.choose n 3)
  favorable_outcomes / total_outcomes

theorem probability_10_coins_at_most_3_heads : probability_at_most_3_heads 10 = 11 / 64 :=
by
  sorry

end probability_10_coins_at_most_3_heads_l148_148394


namespace trajectory_eq_find_m_l148_148705

-- First problem: Trajectory equation
theorem trajectory_eq (P : ℝ × ℝ) (A B : ℝ × ℝ) :
  A = (1, 0) → B = (-1, 0) → 
  (dist P A) * (dist A B) = (dist P B) * (dist A B) → 
  P.snd ^ 2 = 4 * P.fst :=
by sorry

-- Second problem: Value of m
theorem find_m (P : ℝ × ℝ) (M N : ℝ × ℝ) (m : ℝ) :
  P.snd ^ 2 = 4 * P.fst → 
  M.snd = M.fst + m → 
  N.snd = N.fst + m →
  (M.fst - N.fst) * (M.snd - N.snd) + (N.snd - M.snd) * (N.fst - M.fst) = 0 →
  m ≠ 0 →
  m < 1 →
  m = -4 :=
by sorry

end trajectory_eq_find_m_l148_148705


namespace find_vector_at_t_0_l148_148944

def vec2 := ℝ × ℝ

def line_at_t (a d : vec2) (t : ℝ) : vec2 :=
  (a.1 + t * d.1, a.2 + t * d.2)

-- Given conditions
def vector_at_t_1 (v : vec2) : Prop :=
  v = (2, 3)

def vector_at_t_4 (v : vec2) : Prop :=
  v = (8, -5)

-- Prove that the vector at t = 0 is (0, 17/3)
theorem find_vector_at_t_0 (a d: vec2) (h1: line_at_t a d 1 = (2, 3)) (h4: line_at_t a d 4 = (8, -5)) :
  line_at_t a d 0 = (0, 17 / 3) :=
sorry

end find_vector_at_t_0_l148_148944


namespace find_circle_center_value_x_plus_y_l148_148061

theorem find_circle_center_value_x_plus_y : 
  ∀ (x y : ℝ), (x^2 + y^2 = 4 * x - 6 * y + 9) → 
    x + y = -1 :=
by
  intros x y h
  sorry

end find_circle_center_value_x_plus_y_l148_148061


namespace find_z_and_modulus_find_a_and_b_l148_148105

-- Define the complex number z
def z := (1 - complex.i)^2 + 1 + 3 * complex.i

-- Define the modulus of z
def z_modulus := complex.abs z

-- Define the quadratic equation conditions
def quadratic_cond (a b : ℝ) := (z^2 + complex.of_real a * z + complex.of_real b = 1 - complex.i)

theorem find_z_and_modulus : 
  z = 3 + 3 * complex.i ∧ z_modulus = 3 * real.sqrt 2 := by
sorry

theorem find_a_and_b :
  ∃ a b : ℝ, quadratic_cond a b ∧ a = -6 ∧ b = 10 := by
sorry

end find_z_and_modulus_find_a_and_b_l148_148105


namespace find_ax_plus_a_negx_l148_148708

theorem find_ax_plus_a_negx
  (a : ℝ) (x : ℝ)
  (h₁ : a > 0)
  (h₂ : a^(x/2) + a^(-x/2) = 5) :
  a^x + a^(-x) = 23 :=
by
  sorry

end find_ax_plus_a_negx_l148_148708


namespace arrangement_count_l148_148010

theorem arrangement_count (basil_plants tomato_plants : ℕ) (b : basil_plants = 5) (t : tomato_plants = 4) : 
  (Nat.factorial (basil_plants + 1) * Nat.factorial tomato_plants) = 17280 :=
by
  rw [b, t] 
  exact Eq.refl 17280

end arrangement_count_l148_148010


namespace geometric_progression_complex_l148_148812

theorem geometric_progression_complex (a b c m : ℂ) (r : ℂ) (hr : r ≠ 0) 
    (h1 : a = r) (h2 : b = r^2) (h3 : c = r^3) 
    (h4 : a / (1 - b) = m) (h5 : b / (1 - c) = m) (h6 : c / (1 - a) = m) : 
    ∃ m : ℂ, ∀ a b c : ℂ, ∃ r : ℂ, a = r ∧ b = r^2 ∧ c = r^3 
    ∧ r ≠ 0 
    ∧ (a / (1 - b) = m) 
    ∧ (b / (1 - c) = m) 
    ∧ (c / (1 - a) = m) := 
sorry

end geometric_progression_complex_l148_148812


namespace minimum_dominoes_to_crowd_l148_148919

theorem minimum_dominoes_to_crowd (n : ℕ) : n = 9 :=
by
  let board_size := 5
  let domino_size := (2, 1)
  have crowded_definition : ∀ (pieces : ℕ), pieces = 9 →
                                   (pieces * 2) = (board_size * board_size - 7) → true,
    from sorry
  exact sorry

end minimum_dominoes_to_crowd_l148_148919


namespace total_spent_on_clothing_l148_148792

def shorts_cost : ℝ := 15
def jacket_cost : ℝ := 14.82
def shirt_cost : ℝ := 12.51

theorem total_spent_on_clothing : shorts_cost + jacket_cost + shirt_cost = 42.33 := by
  -- Proof goes here.
  sorry

end total_spent_on_clothing_l148_148792


namespace train_A_traveled_distance_l148_148889

-- Defining the conditions
def distance : ℝ := 125
def time_A : ℝ := 12
def time_B : ℝ := 8

-- Calculating speeds
def speed_A : ℝ := distance / time_A
def speed_B : ℝ := distance / time_B

-- Defining the time when trains meet
def time_meeting : ℝ := distance / (speed_A + speed_B)

-- Proving Train A's traveled distance at the meeting time
theorem train_A_traveled_distance : speed_A * time_meeting = 50 := by
  sorry

end train_A_traveled_distance_l148_148889


namespace valid_dodecahedron_configurations_l148_148947

-- Conditions

def congruent_pentagons := 5
def cross_shape_formation := true
def sixth_pentagon(Position: ℕ) := 1 <= Position ∧ Position <= 12

-- Theorem Statement

theorem valid_dodecahedron_configurations : 
  congruent_pentagons = 5 ∧ cross_shape_formation ∧ (∀ (pos: ℕ), sixth_pentagon(pos)) → 
  (∃ (count: ℕ), count = 8) := 
by 
  sorry

end valid_dodecahedron_configurations_l148_148947


namespace measure_of_each_arc_l148_148848

theorem measure_of_each_arc (circle : Type) 
                            (arc : circle → ℝ) 
                            (h1 : ∀ A B C D E : circle, arc A = arc B 
                                ∧ arc B = arc C ∧ arc C = arc D ∧ arc D = arc E)
                            (total_measure : ∑ x, arc x = 360) : 
  arc = λ x, 72 :=
by
  sorry

end measure_of_each_arc_l148_148848


namespace probability_heads_at_most_3_of_10_coins_flipped_l148_148367

theorem probability_heads_at_most_3_of_10_coins_flipped :
  (∑ k in finset.range 4, nat.choose 10 k) / 2^10 = 11 / 64 :=
sorry

end probability_heads_at_most_3_of_10_coins_flipped_l148_148367


namespace exists_permutation_with_scrambling_l148_148811

-- Let's first define the concept of a permutation and the scrambling count
def is_permutation {α : Type*} (l : list α) : Prop :=
  (l.nodup ∧ l.perm (list.range l.length))

-- Define the scrambling count of a permutation
def scrambling {α : Type*} [linear_order α] (l : list α) : ℕ :=
(l.zip_with_index).filter (λ ⟨a_i, i⟩, l.drop (i + 1) |>.any (λ a_j, a_j > a_i)).length

-- Now state the main theorem
theorem exists_permutation_with_scrambling (n : ℕ) (k : ℕ) (h1 : 0 ≤ k) (h2 : k ≤ (nat.choose n 2)) :
  ∃ π : list ℕ, is_permutation π ∧ length π = n ∧ scrambling π = k :=
begin
  sorry
end

end exists_permutation_with_scrambling_l148_148811


namespace perpendicular_sufficient_not_necessary_l148_148348

theorem perpendicular_sufficient_not_necessary (m : ℝ) :
  m = -1 → (∀ (l1 l2 : ℝ → ℝ → ℝ), 
    l1 = λ x y, m * x + (2 * m - 1) * y + 1 ∧ 
      l2 = λ x y, 3 * x + m * y + 3 → 
        (∃ (slope1 slope2 : ℝ), slope1 * slope2 = -1) 
          ∧ ¬ (∀ (m : ℝ), (∃ (slope1 slope2 : ℝ), slope1 * slope2 = -1) → m = -1)) :=
by
  sorry

end perpendicular_sufficient_not_necessary_l148_148348


namespace solve_for_b_l148_148140

def is_imaginary (z : ℂ) : Prop := z.re = 0

theorem solve_for_b (b : ℝ) (i_is_imag_unit : ∀ (z : ℂ), i * z = z * i):
  is_imaginary (i * (b * i + 1)) → b = 0 :=
by
  sorry

end solve_for_b_l148_148140


namespace unique_four_letter_sequence_l148_148650

def alphabet_value (c : Char) : ℕ :=
  if 'A' <= c ∧ c <= 'Z' then (c.toNat - 'A'.toNat + 1) else 0

def sequence_product (s : String) : ℕ :=
  s.foldl (λ acc c => acc * alphabet_value c) 1

theorem unique_four_letter_sequence (s : String) :
  sequence_product "WXYZ" = sequence_product s → s = "WXYZ" :=
by
  sorry

end unique_four_letter_sequence_l148_148650


namespace value_of_x_l148_148156

theorem value_of_x (x : ℕ) (h : (multiset.range (7 :: 9 :: 6 :: x :: 8 :: 7 :: 5 :: [])) = 6) :
  x = 11 ∨ x = 3 :=
sorry

end value_of_x_l148_148156


namespace prove_area_A_prove_total_population_prove_elevation_difference_l148_148869

variables (Area_A Area_B : ℝ) (Dist_AB : ℝ) (Pop_Dens_A Pop_Dens_B : ℝ) (Pop_A Pop_B : ℕ) (Elev_Diff : ℝ) 

-- Given conditions
def area_B : ℝ := 200
def dist_between_A_and_B : ℝ := 25
def pop_density_A : ℝ := 50
def pop_density_B : ℝ := 75
def linear_gradient : ℝ := 500 / 10

-- Calculate the area east of plain A
def area_A := area_B - 50
#check (area_A = 150) -- Should be true

-- Calculate the population of both plains
def population_A := area_A * pop_density_A
def population_B := area_B * pop_density_B
def total_population := population_A + population_B
#check (total_population = 22500) -- Should be true

-- Calculate the elevation difference
def elevation_difference := linear_gradient * (dist_between_A_and_B / 10)
#check (elevation_difference = 125) -- Should be true

theorem prove_area_A : area_A = 150 := 
by {
  -- Proof skipped
  sorry
}

theorem prove_total_population : total_population = 22500 := 
by {
  -- Proof skipped
  sorry
}

theorem prove_elevation_difference : elevation_difference = 125 := 
by {
  -- Proof skipped
  sorry
}

end prove_area_A_prove_total_population_prove_elevation_difference_l148_148869


namespace prism_surface_area_l148_148154

theorem prism_surface_area (P : ℝ) (h : ℝ) (S : ℝ) (s: ℝ) 
  (hP : P = 4)
  (hh : h = 2) 
  (hs : s = 1) 
  (h_surf_top : S = s * s) 
  (h_lat : S = 8) : 
  S = 10 := 
sorry

end prism_surface_area_l148_148154


namespace dog_years_second_year_l148_148255

theorem dog_years_second_year (human_years : ℕ) :
  15 + human_years + 5 * 8 = 64 →
  human_years = 9 :=
by
  intro h
  sorry

end dog_years_second_year_l148_148255


namespace fraction_identity_l148_148082

theorem fraction_identity (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h1 : a^2 = b^2 + b * c) (h2 : b^2 = c^2 + a * c) : 
  (1 / c) = (1 / a) + (1 / b) :=
by 
  sorry

end fraction_identity_l148_148082


namespace common_chord_l148_148254

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 + x - 2*y - 20 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 = 25

-- The common chord is the line where both circle equations are satisfied
theorem common_chord (x y : ℝ) : circle1 x y ∧ circle2 x y → x - 2*y + 5 = 0 :=
sorry

end common_chord_l148_148254


namespace problem_1_problem_2_problem_3_l148_148234

-- Problem 1
theorem problem_1 (c d : ℕ) : 
  ∃ a b : ℕ, a + b * sqrt 3 = (c + d * sqrt 3)^2 ∧ a = c^2 + 3*d^2 ∧ b = 2*c*d := 
by
  sorry

-- Problem 2
theorem problem_2 (e f : ℕ) (h₁ : 7 - 4 * sqrt 3 = (e - f * sqrt 3)^2) :
  7 - 4 * sqrt 3 = (2 - sqrt 3)^2 := 
by
  sorry

-- Problem 3 
theorem problem_3 : 
  sqrt (7 + sqrt (21 - sqrt 80)) = 1 + sqrt 5 := 
by 
  sorry

end problem_1_problem_2_problem_3_l148_148234


namespace transformed_function_correct_l148_148259

def original_function(x : ℝ) : ℝ := Real.sin x

def translated_function(x : ℝ) : ℝ := original_function(x + Real.pi / 3)

def stretched_function(x : ℝ) : ℝ := translated_function(x / 2)

theorem transformed_function_correct :
    stretched_function(x) = Real.sin (x / 2 + Real.pi / 3) :=
  sorry

end transformed_function_correct_l148_148259


namespace series_sum_correct_l148_148633

-- Define the series S
def series_sum : ℕ → ℚ :=
λ n, (2 + n * 6) / (4^(101 - n))

-- Define the sum S
def S : ℚ :=
∑ n in finset.range 100, series_sum (n + 1)

-- Prove that S = 200
theorem series_sum_correct : S = 200 := by
  sorry

end series_sum_correct_l148_148633


namespace probability_at_most_three_heads_10_coins_l148_148353

open Nat

noncomputable def probability_at_most_three_heads (n : ℕ) (k : ℕ) : ℚ :=
  (∑ i in Finset.range (k + 1), (Nat.choose n i : ℚ)) / (2 ^ n : ℚ)

theorem probability_at_most_three_heads_10_coins :
  probability_at_most_three_heads 10 3 = 11 / 64 :=
by
  sorry

end probability_at_most_three_heads_10_coins_l148_148353


namespace parcel_total_weight_l148_148844

theorem parcel_total_weight (x y z : ℝ) 
  (h1 : x + y = 132) 
  (h2 : y + z = 146) 
  (h3 : z + x = 140) : 
  x + y + z = 209 :=
by
  sorry

end parcel_total_weight_l148_148844


namespace projection_theorem_l148_148673

noncomputable def projection_vector : ℝ × ℝ × ℝ → ℝ × ℝ × ℝ → ℝ × ℝ × ℝ
| (a₁, a₂, a₃), (b₁, b₂, b₃) :=
    let dot_vd := a₁ * b₁ + a₂ * b₂ + a₃ * b₃ in
    let dot_dd := b₁ * b₁ + b₂ * b₂ + b₃ * b₃ in
    let scale := dot_vd / dot_dd in
    (scale * b₁, scale * b₂, scale * b₃)

theorem projection_theorem :
  projection_vector (4, -1, 3) (1, -2, 2) = (4 / 3, -8 / 3, 8 / 3) :=
by
  sorry

end projection_theorem_l148_148673


namespace max_n_minus_m_l148_148108

noncomputable def f (x : ℝ) : ℝ := Real.log (x / 2) + 1 / 2

noncomputable def g (x : ℝ) : ℝ := Real.exp (x - 2)

theorem max_n_minus_m :
  ∀ m ∈ (set.univ : set ℝ), ∃ n ∈ set.Ioi (0 : ℝ), f m = g n → (n - m) ≤ Real.log 2 :=
by
  intro m hm
  use Real.exp (Real.log 1/2 + 2)
  intro h
  sorry

end max_n_minus_m_l148_148108


namespace probability_heads_at_most_three_out_of_ten_coins_l148_148552

theorem probability_heads_at_most_three_out_of_ten_coins :
  let total_outcomes := 2^10 in
  let favorable_outcomes := Nat.choose 10 0 + Nat.choose 10 1 + Nat.choose 10 2 + Nat.choose 10 3 in
  favorable_outcomes / total_outcomes = 11 / 64 :=
by
  sorry

end probability_heads_at_most_three_out_of_ten_coins_l148_148552


namespace probability_at_most_three_heads_10_coins_l148_148352

open Nat

noncomputable def probability_at_most_three_heads (n : ℕ) (k : ℕ) : ℚ :=
  (∑ i in Finset.range (k + 1), (Nat.choose n i : ℚ)) / (2 ^ n : ℚ)

theorem probability_at_most_three_heads_10_coins :
  probability_at_most_three_heads 10 3 = 11 / 64 :=
by
  sorry

end probability_at_most_three_heads_10_coins_l148_148352


namespace pow_log_sqrt_l148_148897

theorem pow_log_sqrt (a b c : ℝ) (h1 : a = 81) (h2 : b = 500) (h3 : c = 3) :
  ((a ^ (Real.log b / Real.log c)) ^ (1 / 2)) = 250000 :=
by
  sorry

end pow_log_sqrt_l148_148897


namespace probability_of_at_most_3_heads_l148_148403

-- Definitions and conditions
def num_coins : ℕ := 10
def at_most_3_heads_probability : ℚ := 11 / 64

-- Statement of the problem
theorem probability_of_at_most_3_heads (n : ℕ) (p : ℚ) (h1 : n = num_coins) (h2 : p = at_most_3_heads_probability) :
  p = (1 + 10 + 45 + 120 : ℕ) / (2 ^ 10 : ℕ) := by
  sorry

end probability_of_at_most_3_heads_l148_148403


namespace vector_parallel_angle_l148_148737

theorem vector_parallel_angle (α : ℝ) (h_acute : 0 < α ∧ α < π / 2)
  (h_parallel : (3 / 2 : ℝ) * (1 / 3 : ℝ) = (sin α) * (cos α)) : α = π / 4 := by
  sorry

end vector_parallel_angle_l148_148737


namespace rectangle_to_square_l148_148831

theorem rectangle_to_square (a b : ℝ) (h1 : b / 2 < a) (h2 : a < b) :
  ∃ (r : ℝ), r = Real.sqrt (a * b) ∧ 
    (∃ (cut1 cut2 : ℝ × ℝ), 
      cut1.1 = 0 ∧ cut1.2 = a ∧
      cut2.1 = b - r ∧ cut2.2 = r - a ∧
      ∀ t, t = (a * b) - (r ^ 2)) := sorry

end rectangle_to_square_l148_148831


namespace vector_CI_relationship_l148_148764

-- Definitions and conditions of the problem
variables (A B C I : Type)
variables [InnerProductSpace ℝ (Fin 2 → ℝ)] -- Using a 2D vector space over reals
variables {a b c : Fin 2 → ℝ} -- Vectors representing points A, B, and C

-- Given conditions
def is_right_triangle (A B C : Fin 2 → ℝ) : Prop :=
  (∥A - C∥ = 3) ∧ (∥B - C∥ = 4) ∧ ∠ ACB = 90°

def is_incenter (ABC : Fin 2 → ℝ) (I : Fin 2 → ℝ) : Prop := sorry -- Definition of incenter, to be formalized

-- The main statement
theorem vector_CI_relationship (A B C I : Fin 2 → ℝ)
  (h_right : is_right_triangle A B C)
  (h_incenter : is_incenter ({a, b, c} : Fin 2 → ℝ) I) :
  (I - C) = -(C - A) - (C - B) :=
begin
  sorry -- Proof to be filled in
end

end vector_CI_relationship_l148_148764


namespace parabola_vertex_l148_148849

-- Definition of the quadratic function representing the parabola
def parabola (x : ℝ) : ℝ := (3 * x - 1) ^ 2 + 2

-- Statement asserting the coordinates of the vertex of the given parabola
theorem parabola_vertex :
  ∃ h k : ℝ, ∀ x : ℝ, parabola x = 9 * (x - h) ^ 2 + k ∧ h = 1/3 ∧ k = 2 :=
by
  sorry

end parabola_vertex_l148_148849


namespace range_of_m_l148_148717

theorem range_of_m (m : ℝ) :
  (¬ ∃ x : ℝ, x^2 - 2*x + 2 = m) → m < 1 :=
begin
  sorry
end

end range_of_m_l148_148717


namespace polar_coordinates_of_P_l148_148267

theorem polar_coordinates_of_P :
  ∃ (ρ θ : ℝ), (ρ = 2 ∧ θ = -π / 3 ∧ (1, -sqrt 3) = (ρ * cos θ, ρ * sin θ)) :=
by
  use 2
  use -π / 3
  split
  { reflexivity }
  split
  { reflexivity }
  exact sorry

end polar_coordinates_of_P_l148_148267


namespace exists_monochromatic_tree_l148_148284

variables {V : Type*} [Fintype V]

def chromatic_number (G : simple_graph V) : ℕ := G.chromatic_number

def is_monochromatic_tree (G : simple_graph V) (C : set V) (color : edge_coloring G) : Prop :=
  G.is_tree C ∧ G.forall_edges_in_set C (λ e, color e = color ⟨C.some_spec e⟩)

theorem exists_monochromatic_tree
  (G : simple_graph V)
  (k : ℕ)
  (hG : chromatic_number G = k)
  (color : edge_coloring G (fin 2)) :
  ∃ (C : set V), C.card = k ∧ ∃ c : fin 2, is_monochromatic_tree G C (λ e, c) :=
begin
  sorry,
end

end exists_monochromatic_tree_l148_148284


namespace min_value_squared_sum_l148_148207

theorem min_value_squared_sum : 
  ∃ (a b c d e f g h : ℤ), 
  (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧ a ≠ h ∧ 
   b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧ b ≠ h ∧ 
   c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧ c ≠ h ∧ 
   d ≠ e ∧ d ≠ f ∧ d ≠ g ∧ d ≠ h ∧ 
   e ≠ f ∧ e ≠ g ∧ e ≠ h ∧ 
   f ≠ g ∧ f ≠ h ∧ 
   g ≠ h) ∧ 
  {a, b, c, d, e, f, g, h} = {-8, -6, -4, -1, 1, 3, 5, 14} ∧ 
  (a + b + c + d)^2 + (e + f + g + h)^2 = 8 := 
by
  sorry

end min_value_squared_sum_l148_148207


namespace time_after_10000_seconds_l148_148189

def time_add_seconds (h m s : Nat) (t : Nat) : (Nat × Nat × Nat) :=
  let total_seconds := h * 3600 + m * 60 + s + t
  let hours := (total_seconds / 3600) % 24
  let minutes := (total_seconds % 3600) / 60
  let seconds := (total_seconds % 3600) % 60
  (hours, minutes, seconds)

theorem time_after_10000_seconds :
  time_add_seconds 5 45 0 10000 = (8, 31, 40) :=
by
  sorry

end time_after_10000_seconds_l148_148189


namespace probability_10_coins_at_most_3_heads_l148_148397

def probability_at_most_3_heads (n : ℕ) : ℚ :=
  let total_outcomes := 2^n
  let favorable_outcomes := (Nat.choose n 0) + (Nat.choose n 1) + (Nat.choose n 2) + (Nat.choose n 3)
  favorable_outcomes / total_outcomes

theorem probability_10_coins_at_most_3_heads : probability_at_most_3_heads 10 = 11 / 64 :=
by
  sorry

end probability_10_coins_at_most_3_heads_l148_148397


namespace sally_bread_consumption_l148_148833

theorem sally_bread_consumption :
  (2 * 2) + (1 * 2) = 6 :=
by
  sorry

end sally_bread_consumption_l148_148833


namespace square_perimeter_inside_triangle_l148_148929

noncomputable def more_than_half_square_in_triangle (T : Triangle) (C : Circle) (S : Square) 
  (h1 : inscribed C T) (h2 : circumscribed S C) : Prop :=
  (perimeter_inside_triangle T S) > (perimeter S / 2)

-- Stating the main theorem
theorem square_perimeter_inside_triangle 
  (T : Triangle) (C : Circle) (S : Square)
  (h1 : inscribed C T) 
  (h2 : circumscribed S C) : 
  more_than_half_square_in_triangle T C S h1 h2 := 
begin
  sorry
end

end square_perimeter_inside_triangle_l148_148929


namespace probability_heads_at_most_3_l148_148536

theorem probability_heads_at_most_3 (num_coins : ℕ) (num_heads_at_most : ℕ) (probability : ℚ):
  num_coins = 10 → 
  num_heads_at_most = 3 → 
  probability = (binomial 10 0 + binomial 10 1 + binomial 10 2 + binomial 10 3) / (2^10) →
  probability = (11 / 64) := 
by
  intros hnum_coins hnum_heads_at_most hprobability
  sorry

end probability_heads_at_most_3_l148_148536


namespace probability_at_most_three_heads_10_coins_l148_148360

open Nat

noncomputable def probability_at_most_three_heads (n : ℕ) (k : ℕ) : ℚ :=
  (∑ i in Finset.range (k + 1), (Nat.choose n i : ℚ)) / (2 ^ n : ℚ)

theorem probability_at_most_three_heads_10_coins :
  probability_at_most_three_heads 10 3 = 11 / 64 :=
by
  sorry

end probability_at_most_three_heads_10_coins_l148_148360


namespace probability_heads_at_most_three_out_of_ten_coins_l148_148544

theorem probability_heads_at_most_three_out_of_ten_coins :
  let total_outcomes := 2^10 in
  let favorable_outcomes := Nat.choose 10 0 + Nat.choose 10 1 + Nat.choose 10 2 + Nat.choose 10 3 in
  favorable_outcomes / total_outcomes = 11 / 64 :=
by
  sorry

end probability_heads_at_most_three_out_of_ten_coins_l148_148544


namespace Darcy_remaining_clothes_l148_148033

/--
Darcy initially has 20 shirts and 8 pairs of shorts.
He folds 12 of the shirts and 5 of the pairs of shorts.
We want to prove that the total number of remaining pieces of clothing Darcy has to fold is 11.
-/
theorem Darcy_remaining_clothes
  (initial_shirts : Nat)
  (initial_shorts : Nat)
  (folded_shirts : Nat)
  (folded_shorts : Nat)
  (remaining_shirts : Nat)
  (remaining_shorts : Nat)
  (total_remaining : Nat) :
  initial_shirts = 20 → initial_shorts = 8 →
  folded_shirts = 12 → folded_shorts = 5 →
  remaining_shirts = initial_shirts - folded_shirts →
  remaining_shorts = initial_shorts - folded_shorts →
  total_remaining = remaining_shirts + remaining_shorts →
  total_remaining = 11 := by
  sorry

end Darcy_remaining_clothes_l148_148033


namespace Q_a_probability_l148_148066

noncomputable def Q (a : ℝ) : ℝ := sorry -- definition of Q(a)

theorem Q_a_probability (a : ℝ) (h : 0 ≤ a ∧ a ≤ 1) :
  Q(a) = 7 / 12 :=
sorry

end Q_a_probability_l148_148066


namespace figure_F10_squares_l148_148637

def num_squares (n : ℕ) : ℕ :=
  if n = 1 then 1
  else 1 + 3 * (n - 1) * n

theorem figure_F10_squares : num_squares 10 = 271 :=
by sorry

end figure_F10_squares_l148_148637


namespace problem_statement_l148_148203

open Real

-- Define the basic conditions for the parallelogram and projections
variables (A B C D P Q R S : ℝ×ℝ) (d m n p : ℕ)

-- Conditions based on the problem statement
def parallelogram_area := 18
def projection_PQ := 5
def projection_RS := 7

-- d is the square of the length of AC
def AC := d

-- The final goal based on the asked question and answer 
theorem problem_statement :
  ∃ (m n p : ℕ), p ∉ square_prime_divisors ∧ 
  AC^2 = m + n*sqrt p ∧ 
  m + n + p = 92 := sorry


end problem_statement_l148_148203


namespace probability_of_at_most_3_heads_l148_148470

-- Definitions based on conditions
def binom : ℕ → ℕ → ℕ
| 0, 0       => 1
| 0, (k + 1) => 0
| (n + 1), 0 => 1
| (n + 1), (k + 1) => binom n k + binom n (k + 1)

def favorable_outcomes : ℕ :=
binom 10 0 + binom 10 1 + binom 10 2 + binom 10 3

def total_outcomes : ℕ := 2 ^ 10

def probability (favorable : ℕ) (total : ℕ) : ℚ :=
favorable / total

-- Theorem statement
theorem probability_of_at_most_3_heads : 
  probability favorable_outcomes total_outcomes = 11 / 64 :=
sorry

end probability_of_at_most_3_heads_l148_148470


namespace lcm_230d_l148_148263

noncomputable def lcm_problem {d : ℕ} (h1 : d % 2 ≠ 0) : ℕ :=
  let n := 230
  in n * d

theorem lcm_230d (d : ℕ) (h1 : d % 2 ≠ 0) (h2 : ¬ (230 % 3 = 0)) :
  lcm 230 d = 230 * d :=
by
  sorry

end lcm_230d_l148_148263


namespace scientific_notation_of_4370000_l148_148649

theorem scientific_notation_of_4370000 :
  4370000 = 4.37 * 10^6 :=
sorry

end scientific_notation_of_4370000_l148_148649


namespace probability_at_most_3_heads_l148_148444

def binom (n k : ℕ) : ℕ := Nat.choose n k

theorem probability_at_most_3_heads :
  let total_outcomes := 2^10
  let successful_outcomes := binom 10 0 + binom 10 1 + binom 10 2 + binom 10 3
  (successful_outcomes : ℚ) / total_outcomes = 11 / 64 := 
by 
  let total_outcomes := 2 ^ 10
  let successful_outcomes := binom 10 0 + binom 10 1 + binom 10 2 + binom 10 3
  have h : (successful_outcomes : ℚ) / total_outcomes = 11 / 64 := sorry
  exact h

end probability_at_most_3_heads_l148_148444


namespace elberta_has_21_75_dollars_l148_148743

def granny_smith_amount : ℝ := 75
def anjou_amount : ℝ := granny_smith_amount / 4
def elberta_amount : ℝ := anjou_amount + 3

theorem elberta_has_21_75_dollars : elberta_amount = 21.75 := 
by 
  unfold anjou_amount elberta_amount granny_smith_amount
  sorry

end elberta_has_21_75_dollars_l148_148743


namespace probability_at_most_3_heads_10_coins_l148_148429

noncomputable def coin_flip_prob_at_most_3_heads : ℚ :=
  let total_outcomes := 2^10
  let favorable_outcomes := Nat.binomial 10 0 + Nat.binomial 10 1 + Nat.binomial 10 2 + Nat.binomial 10 3
  favorable_outcomes / total_outcomes

theorem probability_at_most_3_heads_10_coins : coin_flip_prob_at_most_3_heads = 11 / 64 :=
by
  -- Definitions of let values are integrated directly in the theorem body
  let total_outcomes := 2^10
  let favorable_outcomes := Nat.binomial 10 0 + Nat.binomial 10 1 + Nat.binomial 10 2 + Nat.binomial 10 3
  show favorable_outcomes / total_outcomes = 11 / 64
  sorry

end probability_at_most_3_heads_10_coins_l148_148429


namespace fraction_equality_l148_148904

theorem fraction_equality (a b c : ℝ) (hc : c ≠ 0) (h : a / c = b / c) : a = b := 
by
  sorry

end fraction_equality_l148_148904


namespace rational_numbers_classification_l148_148270

theorem rational_numbers_classification :
  ∀ (numbers : List ℚ) (fractions : List ℚ) (integers : List ℚ),
  numbers = [-8, 2.1, 1/9, 3, 0, -2.5, -11, -1] →
  fractions = [2.1, 1/9, -2.5] →
  integers = [-8, 3, 0, -11, -1] →
  (∀ x ∈ fractions, ∃ a b : ℤ, b ≠ 0 ∧ x = a / b ∧ ¬(∃ n : ℤ, x = n)) ∧
  (∀ y ∈ integers, ∃ n : ℤ, y = n) :=
by
  intros numbers fractions integers h_numbers h_fractions h_integers
  split
  sorry


end rational_numbers_classification_l148_148270


namespace vector_pointing_to_line_is_parallel_l148_148638

open Real

theorem vector_pointing_to_line_is_parallel :
  ∃ a b t, a = 5 * t + 3 ∧ b = 2 * t - 1 ∧ ∃ k : Real, (a, b) = (5 * k, 2 * k) :=
by
  use [-2.5, -1, -1.1]
  sorry

end vector_pointing_to_line_is_parallel_l148_148638


namespace probability_heads_at_most_3_of_10_coins_flipped_l148_148373

theorem probability_heads_at_most_3_of_10_coins_flipped :
  (∑ k in finset.range 4, nat.choose 10 k) / 2^10 = 11 / 64 :=
sorry

end probability_heads_at_most_3_of_10_coins_flipped_l148_148373


namespace probability_at_most_3_heads_10_flips_l148_148526

theorem probability_at_most_3_heads_10_flips : 
  let num_heads_0 := nat.choose 10 0,
      num_heads_1 := nat.choose 10 1,
      num_heads_2 := nat.choose 10 2,
      num_heads_3 := nat.choose 10 3,
      total_outcomes := 2^10 in
  (num_heads_0 + num_heads_1 + num_heads_2 + num_heads_3) / total_outcomes = 11 / 64 :=
by
  sorry

end probability_at_most_3_heads_10_flips_l148_148526


namespace ratio_of_areas_of_BCY_and_ACX_l148_148175

theorem ratio_of_areas_of_BCY_and_ACX (A B C X Y : Point) 
  (hAX : midpoint A X B)
  (hAY : midpoint A Y C)
  (hAB : distance A B = 35)
  (hBC : distance B C = 40)
  (hAC : distance A C = 45)
  (hBisect_CY : is_angle_bisector B C Y)
  (hBisect_CX : is_angle_bisector A C X) :
  (area (triangle B C Y)) / (area (triangle A C X)) = 
  (2480 * 8 * 45) / (105 * 7 * 8960 * 40) :=
  sorry

end ratio_of_areas_of_BCY_and_ACX_l148_148175


namespace chang_apple_problem_l148_148629

theorem chang_apple_problem 
  (A : ℝ)
  (h1 : 0.50 * A * 0.50 + 0.25 * A * 0.10 + 0.15 * A * 0.30 + 0.10 * A * 0.20 = 80)
  : A = 235 := 
sorry

end chang_apple_problem_l148_148629


namespace proof_parametric_to_cartesian_l148_148766

-- Define the parametric equation of curve C
def parametric_curve_C (θ : ℝ) : ℝ × ℝ :=
  (1 + 4 * Real.cos θ, -1 + 4 * Real.sin θ)

-- Define the polar equation of line l
def polar_line_l (ρ θ : ℝ) (m : ℝ) : Prop :=
  ρ = 2 * Real.sqrt 2 * m / Real.sin (θ + Real.pi / 4)

-- Convert polar to Cartesian equation: x + y - 4m = 0
def cartesian_line_l (x y m : ℝ) : Prop :=
  x + y - 4 * m = 0

-- Define the condition that the line intersects the curve at two points A and B with distance |AB| = 4
def line_intersects_curve_at_AB (A B : ℝ × ℝ) (m : ℝ) :=
  let d := 2 * Real.sqrt 3  -- distance from the center to the line
  in ((A.1 - 1)^2 + (A.2 + 1)^2 = 16) ∧ (A.1 + A.2 - 4 * m = 0) ∧
     ((B.1 - 1)^2 + (B.2 + 1)^2 = 16) ∧ (B.1 + B.2 - 4 * m = 0) ∧
     (Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2) = 4)

-- The theorem we need to prove:
theorem proof_parametric_to_cartesian (θ ρ m : ℝ) (A B x y : ℝ) :
  (parametric_curve_C θ = (x, y)) →
  polar_line_l ρ θ m →
  line_intersects_curve_at_AB (A, B) m →
  ((x-1)^2 + (y+1)^2 = 16) ∧ (x+y-4*m = 0) ∧ (m = Real.sqrt 6 / 2 ∨ m = -Real.sqrt 6 / 2) :=
by sorry

end proof_parametric_to_cartesian_l148_148766


namespace no_convolution_square_distrib_l148_148200

open MeasureTheory

variables {P : Measure ℝ} {p : ℝ → ℝ}
variables (c d : ℝ) (hx : 0 < c) (hy : c < d)

noncomputable def valid_density_function (p : ℝ → ℝ) : Prop :=
  ∀ x, (-1 ≤ x ∧ x ≤ 1) → c < p x ∧ p x < d

theorem no_convolution_square_distrib
  (symm : ∀ s, P s = P (-s))
  (abs_cont : ∀ s, Real.measure_theory.is_absolutely_continuous P volume)
  (bounded_density : ∀ x, ¬ (-1 ≤ x ∧ x ≤ 1) → p x = 0)
  (density_bound : valid_density_function p c d):
  ¬ ∃ Q : Measure ℝ, convolution Q Q = P :=
sorry

end no_convolution_square_distrib_l148_148200


namespace ellipse_equation_find_exists_fixed_point_dot_product_l148_148094

noncomputable theory
open_locale big_operators

-- Definition for the ellipse equation given conditions
def ellipse_eq (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

-- Definition for the constant c and b
def c := real.sqrt 3
def b := 1

-- Theorem part (I)
theorem ellipse_equation_find (a : ℝ) (h1 : a^2 = c^2 + b^2) :
  ellipse_eq a b x y ↔ (x^2 / 4 + y^2 = 1) :=
sorry

-- Definitions for vectors PE and QE dot product
def dot_product (P E Q : ℝ × ℝ) : ℝ :=
  (E.1 - P.1) * (E.1 - Q.1) + (E.2 - P.2) * (E.2 - Q.2)

-- Theorem part (II)
theorem exists_fixed_point_dot_product :
  ∃ m : ℝ, ∀ P Q : ℝ × ℝ, (P ≠ Q) →
  (∃ l : ℝ, line_through l (1, 0) P Q ∧ intersects l ellipse_eq) →
  (dot_product P (m, 0) Q = 33 / 64 ∧ m = 17 / 8) :=
sorry

end ellipse_equation_find_exists_fixed_point_dot_product_l148_148094


namespace real_part_of_fraction_l148_148214

-- Define the primary condition |z| = 2, with z being a complex number.
variable {x y : ℝ}

theorem real_part_of_fraction {z : ℂ} (hz : x ^ 2 + y ^ 2 = 4) :
  let z := x + y * complex.I in (complex.re (1 / (1 - z)) = (1 - x) / (5 - 2 * x)) :=
by
  sorry

end real_part_of_fraction_l148_148214


namespace range_of_a_l148_148702

variables (x a : ℝ)
def p : Prop := abs (x + 1) > 2
def q : Prop := abs x > a

theorem range_of_a (h : ¬ p → ¬ q) : a ≤ 1 :=
sorry

end range_of_a_l148_148702


namespace monotonic_intervals_f_greater_than_4_l148_148120

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := 4 * x + 1 / x + a * Real.log x

-- Proof Problem 1: Monotonic intervals
theorem monotonic_intervals (a : ℝ) : 
  let critical_x := (-a + Real.sqrt (a^2 + 16)) / 8 in
  (∀ x, 0 < x ∧ x < critical_x → deriv (λ x, f x a) x < 0) ∧ 
  (∀ x, critical_x < x → deriv (λ x, f x a) x > 0) :=
  sorry

-- Proof Problem 2: f(x) > 4 when -3 < a < 0
theorem f_greater_than_4 (a : ℝ) (h_a : -3 < a ∧ a < 0) (x : ℝ) (h_x : 0 < x) : 
  f x a > 4 :=
  sorry

end monotonic_intervals_f_greater_than_4_l148_148120


namespace domain_and_range_of_g_l148_148210

noncomputable def f (x : ℝ) : ℝ := x^2 / 3

noncomputable def g (x : ℝ) : ℝ := 2 - f (x^2 + 1)

theorem domain_and_range_of_g :
  (∀ x : ℝ, x ∈ (-real.sqrt 2 : ℝ) .. (real.sqrt 2 : ℝ) → g x ∈ [-1, 5/3]) ∧
  (∀ (a b c d : ℝ),
    a = -real.sqrt 2 ∧
    b = real.sqrt 2 ∧
    c = -1 ∧
    d = 5/3 → 
    (a, b, c, d) = (-real.sqrt 2, real.sqrt 2, -1, 5/3)) :=
by 
  sorry

end domain_and_range_of_g_l148_148210


namespace probability_of_at_most_3_heads_l148_148474

-- Definitions based on conditions
def binom : ℕ → ℕ → ℕ
| 0, 0       => 1
| 0, (k + 1) => 0
| (n + 1), 0 => 1
| (n + 1), (k + 1) => binom n k + binom n (k + 1)

def favorable_outcomes : ℕ :=
binom 10 0 + binom 10 1 + binom 10 2 + binom 10 3

def total_outcomes : ℕ := 2 ^ 10

def probability (favorable : ℕ) (total : ℕ) : ℚ :=
favorable / total

-- Theorem statement
theorem probability_of_at_most_3_heads : 
  probability favorable_outcomes total_outcomes = 11 / 64 :=
sorry

end probability_of_at_most_3_heads_l148_148474


namespace table_tennis_championship_max_winners_l148_148560

theorem table_tennis_championship_max_winners 
  (n : ℕ) 
  (knockout_tournament : ∀ (players : ℕ), players = 200) 
  (eliminations_per_match : ℕ := 1) 
  (matches_needed_to_win_3_times : ℕ := 3) : 
  ∃ k : ℕ, k = 66 ∧ k * matches_needed_to_win_3_times ≤ (n - 1) :=
begin
  sorry
end

end table_tennis_championship_max_winners_l148_148560


namespace probability_at_most_3_heads_10_flips_l148_148517

theorem probability_at_most_3_heads_10_flips : 
  let num_heads_0 := nat.choose 10 0,
      num_heads_1 := nat.choose 10 1,
      num_heads_2 := nat.choose 10 2,
      num_heads_3 := nat.choose 10 3,
      total_outcomes := 2^10 in
  (num_heads_0 + num_heads_1 + num_heads_2 + num_heads_3) / total_outcomes = 11 / 64 :=
by
  sorry

end probability_at_most_3_heads_10_flips_l148_148517


namespace slope_of_line_determined_by_solutions_l148_148306

theorem slope_of_line_determined_by_solutions :
  ∀ (x1 x2 y1 y2 : ℝ), 
  (4 / x1 + 6 / y1 = 0) → (4 / x2 + 6 / y2 = 0) →
  (y2 - y1) / (x2 - x1) = -3 / 2 :=
by
  intros x1 x2 y1 y2 h1 h2
  -- Proof steps go here
  sorry

end slope_of_line_determined_by_solutions_l148_148306


namespace second_discount_percentage_l148_148271

/-- 
  Given:
  - The listed price of Rs. 560.
  - The final sale price after successive discounts of 20% and another discount is Rs. 313.6.
  Prove:
  - The second discount percentage is 30%.
-/
theorem second_discount_percentage (list_price final_price : ℝ) (first_discount_percentage : ℝ) : 
  list_price = 560 → 
  final_price = 313.6 → 
  first_discount_percentage = 20 → 
  ∃ (second_discount_percentage : ℝ), second_discount_percentage = 30 :=
by
  sorry

end second_discount_percentage_l148_148271


namespace probability_of_at_most_3_heads_l148_148406

-- Definitions and conditions
def num_coins : ℕ := 10
def at_most_3_heads_probability : ℚ := 11 / 64

-- Statement of the problem
theorem probability_of_at_most_3_heads (n : ℕ) (p : ℚ) (h1 : n = num_coins) (h2 : p = at_most_3_heads_probability) :
  p = (1 + 10 + 45 + 120 : ℕ) / (2 ^ 10 : ℕ) := by
  sorry

end probability_of_at_most_3_heads_l148_148406


namespace keesha_total_cost_is_correct_l148_148798

noncomputable def hair_cost : ℝ := 
  let cost := 50.0 
  let discount := cost * 0.10 
  let discounted_cost := cost - discount 
  let tip := discounted_cost * 0.20 
  discounted_cost + tip

noncomputable def nails_cost : ℝ := 
  let manicure_cost := 30.0 
  let pedicure_cost := 35.0 * 0.50 
  let total_without_tip := manicure_cost + pedicure_cost 
  let tip := total_without_tip * 0.20 
  total_without_tip + tip

noncomputable def makeup_cost : ℝ := 
  let cost := 40.0 
  let tax := cost * 0.07 
  let total_without_tip := cost + tax 
  let tip := total_without_tip * 0.20 
  total_without_tip + tip

noncomputable def facial_cost : ℝ := 
  let cost := 60.0 
  let discount := cost * 0.15 
  let discounted_cost := cost - discount 
  let tip := discounted_cost * 0.20 
  discounted_cost + tip

noncomputable def total_cost : ℝ := 
  hair_cost + nails_cost + makeup_cost + facial_cost

theorem keesha_total_cost_is_correct : total_cost = 223.56 := by
  sorry

end keesha_total_cost_is_correct_l148_148798


namespace probability_of_at_most_3_heads_l148_148410

-- Definitions and conditions
def num_coins : ℕ := 10
def at_most_3_heads_probability : ℚ := 11 / 64

-- Statement of the problem
theorem probability_of_at_most_3_heads (n : ℕ) (p : ℚ) (h1 : n = num_coins) (h2 : p = at_most_3_heads_probability) :
  p = (1 + 10 + 45 + 120 : ℕ) / (2 ^ 10 : ℕ) := by
  sorry

end probability_of_at_most_3_heads_l148_148410


namespace unique_solution_of_sqrt_eq_l148_148832

theorem unique_solution_of_sqrt_eq (x : ℝ)
  (h1 : 0 ≤ x + 3)
  (h2 : 0 ≤ 3 * x - 2)
  (h3 : sqrt (x + 3) + sqrt (3 * x - 2) = 7) :
  x = 6 :=
sorry

end unique_solution_of_sqrt_eq_l148_148832


namespace probability_10_coins_at_most_3_heads_l148_148396

def probability_at_most_3_heads (n : ℕ) : ℚ :=
  let total_outcomes := 2^n
  let favorable_outcomes := (Nat.choose n 0) + (Nat.choose n 1) + (Nat.choose n 2) + (Nat.choose n 3)
  favorable_outcomes / total_outcomes

theorem probability_10_coins_at_most_3_heads : probability_at_most_3_heads 10 = 11 / 64 :=
by
  sorry

end probability_10_coins_at_most_3_heads_l148_148396


namespace variance_of_all_students_l148_148936

-- Definitions based on the given conditions
def total_students : Nat := 40
def male_students : Nat := 22
def female_students : Nat := 18
def avg_height_male : ℝ := 173
def avg_height_female : ℝ := 163
def variance_male : ℝ := 28
def variance_female : ℝ := 32

-- The statement to prove
theorem variance_of_all_students :
  let avg_height_all := (male_students * avg_height_male + female_students * avg_height_female) / total_students,
      sum_of_sq_male := male_students * variance_male,
      sum_of_sq_female := female_students * variance_female,
      adjustment_male := male_students * (avg_height_male - avg_height_all) ^ 2,
      adjustment_female := female_students * (avg_height_female - avg_height_all) ^ 2
  in (sum_of_sq_male + adjustment_male + sum_of_sq_female + adjustment_female) / total_students = 54.5875 :=
by
  sorry

end variance_of_all_students_l148_148936


namespace Wang_Hua_wins_l148_148277

/-- 
Given a game with 2002 marbles where two players, Zhang Wei and Wang Hua, 
take turns to draw 1, 2, or 3 marbles at a time, and the player who draws the last marble wins. 
Wang Hua goes first.
--/
theorem Wang_Hua_wins : ∀ (WangHua ZhangWei : ℕ → ℕ → ℕ), 
  ∃ (f : ℕ → ℕ → ℕ), 
    f 2002 0 = 2 ∧    -- Wang Hua should take 2 marbles at the start
    (∀ n m, n % 4 = 0 → ZhangWei n m ∈ {1, 2, 3} → WangHua (n - ZhangWei n m) m = 4 - ZhangWei n m) ∧  -- Wang Hua should take 4 - x if Zhang Wei takes x when remaining marbles are a multiple of 4
    (∀ n, n % 4 = 0 → ∃ m, m ≤ n ∧ WangHua (n - m) (m + 1) = 4 - m) :=   -- Ensuring the strategy holds for all multiples of 4
sorry

end Wang_Hua_wins_l148_148277


namespace prob_heads_at_most_3_out_of_10_flips_l148_148461

def coin_flips := 10
def max_heads := 3
def total_outcomes := 2^coin_flips
def favorable_outcomes := (nat.choose coin_flips 0) +
                          (nat.choose coin_flips 1) +
                          (nat.choose coin_flips 2) +
                          (nat.choose coin_flips 3)

theorem prob_heads_at_most_3_out_of_10_flips :
  (favorable_outcomes : ℚ) / total_outcomes = 11 / 64 :=
by
  sorry

end prob_heads_at_most_3_out_of_10_flips_l148_148461


namespace problem_l148_148597

-- Definitions of the concepts
def skew (a b : Line) : Prop := ¬∃(point : Point), point ∈ a ∧ point ∈ b
def intersect (a b : Line) : Prop := ∃(point : Point), point ∈ a ∧ point ∈ b
def parallel (a b : Line) : Prop := ∀(point₁ point₂ : Point), point₁ ∈ a → point₂ ∈ b → vector_point point₁ point₂ = 0
def perpendicular (a b : Line) : Prop := ∃(point₁ point₂ : Point), point₁ ∈ a ∧ point₂ ∈ b ∧ vector_point point₁ point₂ = 90

-- Definition of propositions
def P1 (a b c : Line) : Prop := skew a b ∧ skew b c → skew a c
def P2 (a b c : Line) : Prop := intersect a b ∧ intersect b c → intersect a c
def P3 (a b c : Line) : Prop := parallel a b ∧ parallel b c → parallel a c
def P4 (a b c : Line) : Prop := perpendicular a b ∧ perpendicular b c → perpendicular a c

-- Counting the number of true comparisons
def count_true_propositions (a b c : Line) : Nat :=
  (if P1 a b c then 1 else 0) +
  (if P2 a b c then 1 else 0) +
  (if P3 a b c then 1 else 0) +
  (if P4 a b c then 1 else 0)

-- Statement of the problem
theorem problem : ∀ (a b c : Line), count_true_propositions a b c = 1 := by
  sorry

end problem_l148_148597


namespace smallest_n_l148_148222

def reflect_angle (theta line_angle : ℝ) : ℝ :=
  2 * line_angle - theta

def R (theta : ℝ) : ℝ :=
  let theta₁ := reflect_angle theta (Real.pi / 50)
  reflect_angle theta₁ (Real.pi / 45)

def Rn (theta : ℝ) (n : ℕ) : ℝ :=
  Nat.iterate R n theta

def theta : ℝ :=
  Real.arctan (21 / 88)

theorem smallest_n (n : ℕ) : Rn theta n = R theta → n = 15 := sorry

end smallest_n_l148_148222


namespace probability_at_most_3_heads_10_flips_l148_148522

theorem probability_at_most_3_heads_10_flips : 
  let num_heads_0 := nat.choose 10 0,
      num_heads_1 := nat.choose 10 1,
      num_heads_2 := nat.choose 10 2,
      num_heads_3 := nat.choose 10 3,
      total_outcomes := 2^10 in
  (num_heads_0 + num_heads_1 + num_heads_2 + num_heads_3) / total_outcomes = 11 / 64 :=
by
  sorry

end probability_at_most_3_heads_10_flips_l148_148522


namespace remaining_pieces_to_fold_l148_148037

-- Define the initial counts of shirts and shorts
def initial_shirts : ℕ := 20
def initial_shorts : ℕ := 8

-- Define the counts of folded shirts and shorts
def folded_shirts : ℕ := 12
def folded_shorts : ℕ := 5

-- The target theorem to prove the remaining pieces of clothing to fold
theorem remaining_pieces_to_fold :
  initial_shirts + initial_shorts - (folded_shirts + folded_shorts) = 11 := 
by
  sorry

end remaining_pieces_to_fold_l148_148037


namespace constants_exist_l148_148788

theorem constants_exist (a b c : ℚ) :
  (∀ n : ℕ, 0 < n → ∑ k in finset.range n, k * (n^2 - k^2) = a * n^4 + b * n^2 + c) → 
  a = 1/4 ∧ b = -1/4 ∧ c = 0 :=
by
  sorry

end constants_exist_l148_148788


namespace gas_volume_ranking_l148_148343

theorem gas_volume_ranking (Russia_V: ℝ) (Non_West_V: ℝ) (West_V: ℝ)
  (h_russia: Russia_V = 302790.13)
  (h_non_west: Non_West_V = 26848.55)
  (h_west: West_V = 21428): Russia_V > Non_West_V ∧ Non_West_V > West_V :=
by
  have h1: Russia_V = 302790.13 := h_russia
  have h2: Non_West_V = 26848.55 := h_non_west
  have h3: West_V = 21428 := h_west
  sorry


end gas_volume_ranking_l148_148343


namespace trapezoid_EFGH_area_l148_148300

structure Point :=
  (x : ℝ)
  (y : ℝ)

def trapezoid_area (E F G H : Point) : ℝ :=
  let a := F.y - E.y
  let b := H.y - G.y
  let h := G.x - E.x
  (1 / 2) * (a + b) * h

theorem trapezoid_EFGH_area :
  let E := Point.mk 2 2
  let F := Point.mk 2 5
  let G := Point.mk 6 2
  let H := Point.mk 6 8
  trapezoid_area E F G H = 18 := 
by
  let E := Point.mk 2 2
  let F := Point.mk 2 5
  let G := Point.mk 6 2
  let H := Point.mk 6 8
  have h1 : trapezoid_area E F G H = 18 := by sorry
  exact h1

end trapezoid_EFGH_area_l148_148300


namespace problem1_simplify_problem2_value_l148_148632

theorem problem1_simplify (α : ℝ) : 
  (cos (α - π / 2)) / (sin (5 * π / 2 + α)) * (sin (α - 2 * π)) * (cos (2 * π - α)) = (sin α)^2 := 
sorry

theorem problem2_value (α : ℝ) (h : tan α = 2) : 
  (sin (2 * α)) / ((sin α)^2 + (sin α) * (cos α) - (cos (2 * α)) - 1) = 1 := 
sorry

end problem1_simplify_problem2_value_l148_148632


namespace pyramid_volume_l148_148163

noncomputable def area_of_equilateral_triangle (s : ℝ) := (sqrt 3 / 4) * s^2

noncomputable def area_of_pentagon_base (s : ℝ) := 5 * area_of_equilateral_triangle s

noncomputable def height_of_pyramid (s : ℝ) := (s * sqrt 3) / 2

noncomputable def volume_of_pyramid (base_area : ℝ) (height : ℝ) := (1 / 3) * base_area * height

theorem pyramid_volume {s : ℝ} (h : s = 10) :
  volume_of_pyramid (area_of_pentagon_base s) (height_of_pyramid s) = 625 :=
by
  sorry

end pyramid_volume_l148_148163


namespace probability_at_most_3_heads_l148_148509

-- Definitions and conditions
def coin_flips : ℕ := 10
def success_prob : ℚ := 1/2
def max_heads : ℕ := 3

-- Binomial Probability Computation
noncomputable def probability_of_at_most_3_heads : ℚ :=
  (∑ k in finset.range (max_heads + 1), nat.choose coin_flips k) / (2^coin_flips)

-- The statement to prove
theorem probability_at_most_3_heads :
  probability_of_at_most_3_heads = 11/64 := by
  sorry

end probability_at_most_3_heads_l148_148509


namespace syllogism_sequence_l148_148903

-- Definitions based on the conditions in a)
def Z1_Z2_not_comparable_in_size := ∀ Z1 Z2 : ℂ, complex.is_imaginary Z1 → complex.is_imaginary Z2 → ¬(Z1 < Z2 ∨ Z2 < Z1)
def Z1_Z2_are_imaginary := ∀ Z1 Z2 : ℂ, complex.is_imaginary Z1 ∧ complex.is_imaginary Z2
def imaginary_numbers_not_comparable := ∀ z : ℂ, complex.is_imaginary z → ¬(z < 0 ∨ 0 < z)

-- The theorem we need to prove
theorem syllogism_sequence : imaginary_numbers_not_comparable → 
                            Z1_Z2_are_imaginary → 
                            Z1_Z2_not_comparable_in_size :=
by
  intros h1 h2
  sorry

end syllogism_sequence_l148_148903


namespace gcd_lcm_sum_l148_148316

open Nat

theorem gcd_lcm_sum (a b : ℕ) (h1 : a = 45) (h2 : b = 4410) :
  (gcd a b + lcm a b) = 4455 := by
  sorry

end gcd_lcm_sum_l148_148316


namespace eval_expression_l148_148999

theorem eval_expression :
  (3^3 - 3) - (4^3 - 4) + (5^3 - 5) = 84 := 
by
  sorry

end eval_expression_l148_148999


namespace bingley_bracelets_left_l148_148617

theorem bingley_bracelets_left (bingley_initial : ℕ) (kelly_bracelets : ℕ) (fraction_given_by_kelly : ℕ)
  (bracelets_in_set : ℕ) (give_away_rule : ℕ → ℕ) (portion_given_to_sister : ℕ):
  bingley_initial = 5 ∧
  kelly_bracelets = 16 ∧
  fraction_given_by_kelly = 4 ∧
  bracelets_in_set = 3 ∧
  (∀ b, give_away_rule b = b / 2) ∧
  portion_given_to_sister = 3 → 
  let received_from_kelly := (kelly_bracelets / fraction_given_by_kelly) / bracelets_in_set in
  let total_after_receiving := bingley_initial + received_from_kelly in
  let remaining_after_give_away := total_after_receiving - give_away_rule received_from_kelly in
  let given_to_sister := remaining_after_give_away / portion_given_to_sister in
  let final_count := remaining_after_give_away - given_to_sister in
  final_count = 4 :=
begin
  intro h,
  rcases h with ⟨h1, h2, h3, h4, h5, h6⟩,
  have received_from_kelly := (h2 / h3) / h4,
  have total_after_receiving := h1 + received_from_kelly,
  have remaining_after_give_away := total_after_receiving - h5 received_from_kelly,
  have given_to_sister := remaining_after_give_away / h6,
  have final_count := remaining_after_give_away - given_to_sister,
  exact sorry
end

end bingley_bracelets_left_l148_148617


namespace find_coefficients_of_trig_function_l148_148727

theorem find_coefficients_of_trig_function
  (f : ℝ → ℝ)
  (a b : ℝ)
  (h_def : ∀ x, f x = 2 * a * real.sin (2 * x - real.pi / 3) + b)
  (h_domain : ∀ x, 0 ≤ x ∧ x ≤ real.pi / 2)
  (h_max : ∃ x, f x = 1)
  (h_min : ∃ x, f x = -5) :
  a = 12 - 6 * real.sqrt 3 ∧ b = -23 + 12 * real.sqrt 3 :=
sorry

end find_coefficients_of_trig_function_l148_148727


namespace equilateral_triangle_probability_at_least_one_distance_less_than_one_l148_148762

theorem equilateral_triangle_probability_at_least_one_distance_less_than_one :
  let ABC : Type := EquilateralTriangle 2
  ∃ P : Point ABC, (distance P ABC.A < 1) ∨ (distance P ABC.B < 1) ∨ (distance P ABC.C < 1) →
  probability (P in ABC | (distance P ABC.A < 1) ∨ (distance P ABC.B < 1) ∨ (distance P ABC.C < 1)) =
  (3 / 4) - (Real.pi / (2 * Real.sqrt 3)) :=
sorry

end equilateral_triangle_probability_at_least_one_distance_less_than_one_l148_148762


namespace smallest_percent_increase_from_2_to_3_l148_148678

def percent_increase (initial final : ℕ) : ℚ := 
  ((final - initial : ℕ) : ℚ) / (initial : ℕ) * 100

def value_at_question : ℕ → ℕ
| 1 => 100
| 2 => 200
| 3 => 300
| 4 => 500
| 5 => 1000
| 6 => 2000
| 7 => 4000
| 8 => 8000
| 9 => 16000
| 10 => 32000
| 11 => 64000
| 12 => 125000
| 13 => 250000
| 14 => 500000
| 15 => 1000000
| _ => 0  -- Default case for questions out of range

theorem smallest_percent_increase_from_2_to_3 :
  let p1 := percent_increase (value_at_question 1) (value_at_question 2)
  let p2 := percent_increase (value_at_question 2) (value_at_question 3)
  let p3 := percent_increase (value_at_question 3) (value_at_question 4)
  let p11 := percent_increase (value_at_question 11) (value_at_question 12)
  let p14 := percent_increase (value_at_question 14) (value_at_question 15)
  p2 < p1 ∧ p2 < p3 ∧ p2 < p11 ∧ p2 < p14 :=
by
  sorry

end smallest_percent_increase_from_2_to_3_l148_148678


namespace probability_at_most_3_heads_l148_148502

-- Definitions and conditions
def coin_flips : ℕ := 10
def success_prob : ℚ := 1/2
def max_heads : ℕ := 3

-- Binomial Probability Computation
noncomputable def probability_of_at_most_3_heads : ℚ :=
  (∑ k in finset.range (max_heads + 1), nat.choose coin_flips k) / (2^coin_flips)

-- The statement to prove
theorem probability_at_most_3_heads :
  probability_of_at_most_3_heads = 11/64 := by
  sorry

end probability_at_most_3_heads_l148_148502


namespace interval_of_decrease_l148_148857

def f (x : ℝ) : ℝ := x^2 - 2*x + 4

theorem interval_of_decrease : 
  {x : ℝ | ∃ y : ℝ, y ≤ 1 ∧ f(x) = y} = set.Iic 1 :=
by
  sorry

end interval_of_decrease_l148_148857


namespace maximize_cone_volume_l148_148292

-- Define the conditions of the problem
def R : ℝ := 1
def V_total (n : ℕ) : ℝ := (↑π / 3) * (real.sqrt (n^2 - 1) / n^2)

-- State the problem as a theorem
theorem maximize_cone_volume : ∃ n : ℕ, 2 ≤ n ∧ V_total 2 ≥ V_total n :=
by
  let n := 2
  have h : V_total 2 = (↑π / 3) * (real.sqrt (4 - 1) / 4) := by sorry
  have h2 : ∀ m ≥ 2, V_total m ≤ V_total 2 := by sorry
  use n
  split
  · sorry  -- Prove that n = 2 satisfies the condition
  · sorry  -- Prove that V_total 2 ≥ V_total n for all n ≥ 2

end maximize_cone_volume_l148_148292


namespace smallest_number_of_eggs_l148_148907

theorem smallest_number_of_eggs (c : ℕ) (h1 : 15 * c - 3 > 150) (h2 : c ≥ 11) : 15 * 11 - 3 = 162 :=
by {
  have h3: 15 * c > 153, from nat.mul_lt_mul_of_pos_left h2 (by norm_num), -- proving intermediate steps using existing conditions
  have h4: c ≥ 11, { sorry }, -- to complete the proof, we normally would show there's no smaller integer satisfying the conditions
  exact (by simp),
}

end smallest_number_of_eggs_l148_148907


namespace probability_at_most_three_heads_10_coins_l148_148355

open Nat

noncomputable def probability_at_most_three_heads (n : ℕ) (k : ℕ) : ℚ :=
  (∑ i in Finset.range (k + 1), (Nat.choose n i : ℚ)) / (2 ^ n : ℚ)

theorem probability_at_most_three_heads_10_coins :
  probability_at_most_three_heads 10 3 = 11 / 64 :=
by
  sorry

end probability_at_most_three_heads_10_coins_l148_148355


namespace slope_of_line_determined_by_solutions_l148_148305

theorem slope_of_line_determined_by_solutions :
  ∀ (x1 x2 y1 y2 : ℝ), 
  (4 / x1 + 6 / y1 = 0) → (4 / x2 + 6 / y2 = 0) →
  (y2 - y1) / (x2 - x1) = -3 / 2 :=
by
  intros x1 x2 y1 y2 h1 h2
  -- Proof steps go here
  sorry

end slope_of_line_determined_by_solutions_l148_148305


namespace f_inv_128_l148_148690

section proof_problem

variable {α : Type*} (f : α → ℝ)

-- Given conditions
axiom f_four : f 4 = 2
axiom f_two_x : ∀ x, f (2 * x) = 2 * f x

-- Main theorem stating that the inverse of f at 128 equals 256
theorem f_inv_128 : ∃ x, f x = 128 ∧ x = 256 :=
by
  use 256
  split
  sorry -- Proof goes here

end proof_problem

end f_inv_128_l148_148690


namespace bounded_sequence_l148_148274

theorem bounded_sequence (a : ℕ → ℕ) (h1 : a 1 = 2) (h2 : a 2 = 2)
  (h_rec : ∀ n : ℕ, a (n + 2) = (a (n + 1) + a n) / Nat.gcd (a n) (a (n + 1))) :
  ∃ M : ℕ, ∀ n : ℕ, a n ≤ M := 
sorry

end bounded_sequence_l148_148274


namespace most_likely_dissatisfied_passengers_expected_dissatisfied_passengers_variance_dissatisfied_passengers_l148_148878

-- Define the given conditions
variables (n : ℕ) (hn : n > 1)

-- Define the binomial distribution properties
noncomputable def P_dissatisfied (k : ℕ) : ℝ :=
  2 * (nat.choose (2 * n) (n - k)) * (1 / (2:ℝ)^(2 * n))

-- Task (a): The most likely number of dissatisfied passengers
theorem most_likely_dissatisfied_passengers : ∃ k, k = 1 :=
by sorry

-- Task (b): The expected number of dissatisfied passengers
theorem expected_dissatisfied_passengers : ∃ e, e = real.sqrt (n / real.pi) :=
by sorry

-- Task (c): The variance of the number of dissatisfied passengers
theorem variance_dissatisfied_passengers : ∃ d, d = 0.182 * n :=
by sorry

end most_likely_dissatisfied_passengers_expected_dissatisfied_passengers_variance_dissatisfied_passengers_l148_148878


namespace perpendicular_lines_sum_l148_148099

theorem perpendicular_lines_sum (a b : ℝ) :
  (∃ (x y : ℝ), 2 * x - 5 * y + b = 0 ∧ a * x + 4 * y - 2 = 0 ∧ x = 1 ∧ y = -2) ∧
  (-a / 4) * (2 / 5) = -1 →
  a + b = -2 :=
by
  sorry

end perpendicular_lines_sum_l148_148099


namespace composite_sum_of_powers_l148_148706

theorem composite_sum_of_powers (a b c d : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_pos_d : 0 < d) (h : a * b = c * d) : 
  ∃ x y : ℕ, 1 < x ∧ 1 < y ∧ a^2016 + b^2016 + c^2016 + d^2016 = x * y :=
by sorry

end composite_sum_of_powers_l148_148706


namespace distance_to_lighthouses_at_second_point_in_time_l148_148946

-- Define constants and conditions for the problem
def plane_speed : ℝ := 432 -- speed of the plane in km/h
def travel_time_minutes : ℝ := 5
def initial_distance_travelled : ℝ := (plane_speed / 60) * travel_time_minutes

-- Define angles based on directions
def angle_initial_to_final : ℝ := 22.5

-- Prove the distances using Lean
theorem distance_to_lighthouses_at_second_point_in_time 
  (speed : ℝ) (time : ℝ) (distance: ℝ) (angle: ℝ) 
  (initial_distance : distance = (speed / 60) * time) 
  : ∃ d1 d2, d1 = 86.9 ∧ d2 = 66.6 :=
by
  -- Provided statements and proofs would go here
  sorry

end distance_to_lighthouses_at_second_point_in_time_l148_148946


namespace probability_heads_at_most_three_out_of_ten_coins_l148_148551

theorem probability_heads_at_most_three_out_of_ten_coins :
  let total_outcomes := 2^10 in
  let favorable_outcomes := Nat.choose 10 0 + Nat.choose 10 1 + Nat.choose 10 2 + Nat.choose 10 3 in
  favorable_outcomes / total_outcomes = 11 / 64 :=
by
  sorry

end probability_heads_at_most_three_out_of_ten_coins_l148_148551


namespace smallest_number_of_eggs_l148_148908

theorem smallest_number_of_eggs (c : ℕ) (h1 : 15 * c - 3 > 150) : ∃ n, n = 15 * 11 - 3 :=
by
  use 162
  sorry

end smallest_number_of_eggs_l148_148908


namespace probability_at_most_3_heads_l148_148495

-- Definitions and conditions
def coin_flips : ℕ := 10
def success_prob : ℚ := 1/2
def max_heads : ℕ := 3

-- Binomial Probability Computation
noncomputable def probability_of_at_most_3_heads : ℚ :=
  (∑ k in finset.range (max_heads + 1), nat.choose coin_flips k) / (2^coin_flips)

-- The statement to prove
theorem probability_at_most_3_heads :
  probability_of_at_most_3_heads = 11/64 := by
  sorry

end probability_at_most_3_heads_l148_148495


namespace probability_of_at_most_3_heads_l148_148473

-- Definitions based on conditions
def binom : ℕ → ℕ → ℕ
| 0, 0       => 1
| 0, (k + 1) => 0
| (n + 1), 0 => 1
| (n + 1), (k + 1) => binom n k + binom n (k + 1)

def favorable_outcomes : ℕ :=
binom 10 0 + binom 10 1 + binom 10 2 + binom 10 3

def total_outcomes : ℕ := 2 ^ 10

def probability (favorable : ℕ) (total : ℕ) : ℚ :=
favorable / total

-- Theorem statement
theorem probability_of_at_most_3_heads : 
  probability favorable_outcomes total_outcomes = 11 / 64 :=
sorry

end probability_of_at_most_3_heads_l148_148473


namespace sum_of_digits_base_7_of_999_l148_148308

theorem sum_of_digits_base_7_of_999 : 
  (let n := 999;
       repr_base_7 := [2, 6, 2, 5];
       sum_digits := repr_base_7.foldl (λ x y, x + y) 0
   in sum_digits = 15) :=
by
  sorry

end sum_of_digits_base_7_of_999_l148_148308


namespace candy_bar_sales_l148_148236

def max_sales : ℕ := 24
def seth_sales (max_sales : ℕ) : ℕ := 3 * max_sales + 6
def emma_sales (seth_sales : ℕ) : ℕ := seth_sales / 2 + 5
def total_sales (seth_sales emma_sales : ℕ) : ℕ := seth_sales + emma_sales

theorem candy_bar_sales : total_sales (seth_sales max_sales) (emma_sales (seth_sales max_sales)) = 122 := by
  sorry

end candy_bar_sales_l148_148236


namespace smallest_number_greater_than_300_divided_by_25_has_remainder_24_l148_148568

theorem smallest_number_greater_than_300_divided_by_25_has_remainder_24 :
  ∃ x : ℕ, (x > 300) ∧ (x % 25 = 24) ∧ (x = 324) := by
  sorry

end smallest_number_greater_than_300_divided_by_25_has_remainder_24_l148_148568


namespace arithmetic_sequences_l148_148664

open Nat

theorem arithmetic_sequences
  (a : ℕ → ℕ)
  (h : ∀ n, (∏ i in range n, a i) ∣ (∏ i in range n, a (n + i))): 
  ∃ k, (∀ i, a i = k * (i + 1)) :=
by
  sorry

end arithmetic_sequences_l148_148664


namespace arrange_plants_in_a_row_l148_148012

-- Definitions for the conditions
def basil_plants : ℕ := 5 -- Number of basil plants
def tomato_plants : ℕ := 4 -- Number of tomato plants

-- Theorem statement asserting the number of ways to arrange the plants
theorem arrange_plants_in_a_row : 
  let total_items := basil_plants + 1,
      ways_to_arrange_total_items := Nat.factorial total_items,
      ways_to_arrange_tomato_group := Nat.factorial tomato_plants in
  (ways_to_arrange_total_items * ways_to_arrange_tomato_group) = 17280 := 
by
  sorry

end arrange_plants_in_a_row_l148_148012


namespace car_speed_l148_148328

def time80 := 1 / 80 * 3600   -- Time to travel 1 km at 80 km/hr in seconds
def time_v := time80 + 4      -- Car takes 4 seconds longer
noncomputable def v := 1 / (time_v / 3600)  -- Calculate speed v in km/hr

theorem car_speed : v = 3600 / 49 := 
by 
  unfold time80 time_v v
  simp
  sorry

end car_speed_l148_148328


namespace choosing_officers_l148_148938

noncomputable def total_ways_to_choose_officers (members : List String) (boys : ℕ) (girls : ℕ) : ℕ :=
  let total_members := boys + girls
  let president_choices := total_members
  let vice_president_choices := boys - 1 + girls - 1
  let remaining_members := total_members - 2
  president_choices * vice_president_choices * remaining_members

theorem choosing_officers (members : List String) (boys : ℕ) (girls : ℕ) :
  boys = 15 → girls = 15 → members.length = 30 → total_ways_to_choose_officers members boys girls = 11760 :=
by
  intros hboys hgirls htotal
  rw [hboys, hgirls]
  sorry

end choosing_officers_l148_148938


namespace probability_heads_at_most_three_out_of_ten_coins_l148_148549

theorem probability_heads_at_most_three_out_of_ten_coins :
  let total_outcomes := 2^10 in
  let favorable_outcomes := Nat.choose 10 0 + Nat.choose 10 1 + Nat.choose 10 2 + Nat.choose 10 3 in
  favorable_outcomes / total_outcomes = 11 / 64 :=
by
  sorry

end probability_heads_at_most_three_out_of_ten_coins_l148_148549


namespace allocation_schemes_165_l148_148881

theorem allocation_schemes_165 :
  let groups := 9 in
  let total_individuals := 12 in
  ∃ f : Fin groups → Fin total_individuals.succ,
    (∀ i : Fin groups, 1 ≤ f i) ∧ ∑ i, f i = total_individuals ↔ 165 := sorry

end allocation_schemes_165_l148_148881


namespace probability_at_most_3_heads_10_flips_l148_148524

theorem probability_at_most_3_heads_10_flips : 
  let num_heads_0 := nat.choose 10 0,
      num_heads_1 := nat.choose 10 1,
      num_heads_2 := nat.choose 10 2,
      num_heads_3 := nat.choose 10 3,
      total_outcomes := 2^10 in
  (num_heads_0 + num_heads_1 + num_heads_2 + num_heads_3) / total_outcomes = 11 / 64 :=
by
  sorry

end probability_at_most_3_heads_10_flips_l148_148524


namespace computer_multiplications_in_30_minutes_l148_148573

def multiplications_per_second : ℕ := 20000
def seconds_per_minute : ℕ := 60
def minutes : ℕ := 30
def total_seconds : ℕ := minutes * seconds_per_minute
def expected_multiplications : ℕ := 36000000

theorem computer_multiplications_in_30_minutes :
  multiplications_per_second * total_seconds = expected_multiplications :=
by
  sorry

end computer_multiplications_in_30_minutes_l148_148573


namespace license_plates_count_l148_148745

def num_consonants : Nat := 20
def num_vowels : Nat := 6
def num_digits : Nat := 10
def num_symbols : Nat := 3

theorem license_plates_count : 
  num_consonants * num_vowels * num_consonants * num_digits * num_symbols = 72000 :=
by 
  sorry

end license_plates_count_l148_148745


namespace phi_k_2001_l148_148043

-- Definitions given in the problem's condition
def phi_k (k n : ℕ) : ℕ := number of positive integers ≤ n / k that are relatively prime to n

-- Given condition (hint): phi(2003) = 2002
def phi (n : ℕ) : ℕ := Euler's Totient Function evaluated at n

axiom phi_2003_is_2002 : phi 2003 = 2002

-- Problem statement
theorem phi_k_2001 (n : ℕ) : phi_k 2001 (2002^2 - 1) = 1233 := by
  sorry

end phi_k_2001_l148_148043


namespace prob_heads_at_most_3_out_of_10_flips_l148_148451

def coin_flips := 10
def max_heads := 3
def total_outcomes := 2^coin_flips
def favorable_outcomes := (nat.choose coin_flips 0) +
                          (nat.choose coin_flips 1) +
                          (nat.choose coin_flips 2) +
                          (nat.choose coin_flips 3)

theorem prob_heads_at_most_3_out_of_10_flips :
  (favorable_outcomes : ℚ) / total_outcomes = 11 / 64 :=
by
  sorry

end prob_heads_at_most_3_out_of_10_flips_l148_148451


namespace prob_heads_at_most_3_out_of_10_flips_l148_148447

def coin_flips := 10
def max_heads := 3
def total_outcomes := 2^coin_flips
def favorable_outcomes := (nat.choose coin_flips 0) +
                          (nat.choose coin_flips 1) +
                          (nat.choose coin_flips 2) +
                          (nat.choose coin_flips 3)

theorem prob_heads_at_most_3_out_of_10_flips :
  (favorable_outcomes : ℚ) / total_outcomes = 11 / 64 :=
by
  sorry

end prob_heads_at_most_3_out_of_10_flips_l148_148447


namespace josh_total_money_l148_148793

-- Define the initial conditions
def initial_wallet : ℝ := 300
def initial_investment : ℝ := 2000
def stock_increase_rate : ℝ := 0.30

-- The expected total amount Josh will have after selling his stocks
def expected_total_amount : ℝ := 2900

-- Define the problem: that the total money in Josh's wallet after selling all stocks equals $2900
theorem josh_total_money :
  let increased_value := initial_investment * stock_increase_rate
  let new_investment := initial_investment + increased_value
  let total_money := new_investment + initial_wallet
  total_money = expected_total_amount :=
by
  sorry

end josh_total_money_l148_148793


namespace complement_union_l148_148206

def U : Set Int := {-2, -1, 0, 1, 2}

def A : Set Int := {-1, 2}

def B : Set Int := {-1, 0, 1}

theorem complement_union :
  (U \ B) ∪ A = {-2, -1, 2} :=
by
  sorry

end complement_union_l148_148206


namespace smallest_n_is_29_l148_148949

noncomputable def smallest_possible_n (r g b : ℕ) : ℕ :=
  Nat.lcm (Nat.lcm (10 * r) (16 * g)) (18 * b) / 25

theorem smallest_n_is_29 (r g b : ℕ) (h : 10 * r = 16 * g ∧ 16 * g = 18 * b) :
  smallest_possible_n r g b = 29 :=
by
  sorry

end smallest_n_is_29_l148_148949


namespace _l148_148596

-- Defining the persons involved
inductive Person
| Alice
| Bob
| Carla
| Derek
| Eric
deriving DecidableEq, Fintype

open Person

-- Function to check if a seating arrangement is valid
def validSeating (arrangement : List Person) : Prop :=
  arrangement.length = 5 ∧
  (∀ i, i < 4 → 
    (arrangement.nth i = some Alice → 
      arrangement.nth (i + 1) ≠ some Bob ∧ arrangement.nth (i + 1) ≠ some Carla)) ∧
  (∀ i, i < 4 → 
    (arrangement.nth i = some Derek →
      arrangement.nth (i + 1) ≠ some Eric)) ∧
  (∀ i, i < 4 → 
    (arrangement.nth i = some Bob →
      arrangement.nth (i + 1) ≠ some Derek))

-- Counting valid arrangements
def count_valid_arrangements : ℕ :=
  (Finset.univ.filter validSeating).card

-- Main theorem statement
lemma seating_arrangements_count : count_valid_arrangements = 2 := by
  sorry

end _l148_148596


namespace correct_option_is_d_l148_148599

theorem correct_option_is_d :
  let f1 := fun x => Real.sin (Real.abs x),
      f2 := fun x => Real.cos (Real.abs x),
      f3 := fun x => Real.abs (Real.cot x),
      f4 := fun x => Real.log (Real.abs (Real.sin x))
  in
  (Real.periodic f1 (2 * Real.pi) ∧ ¬Real.periodic f1 Real.pi) ∧
  (Real.periodic f2 (2 * Real.pi) ∧ ¬Real.periodic f2 Real.pi) ∧ 
  (¬Real.strictly_increasing_on f2 (Set.Ioo 0 (Real.pi / 2))) ∧
  (Real.periodic f3 Real.pi ∧ ¬Real.strictly_increasing_on f3 (Set.Ioo 0 (Real.pi / 2))) ∧
  (Real.periodic f4 Real.pi ∧ Real.strictly_increasing_on f4 (Set.Ioo 0 (Real.pi / 2)))

end correct_option_is_d_l148_148599


namespace angle_ratio_l148_148593

theorem angle_ratio (O A B C E : Point) (h1 : AcuteAngledTriangle A B C)
  (h2 : O.circle (A) ∧ O.circle (B) ∧ O.circle (C))
  (h3 : arc AB = 100 ∧ arc BC = 80)
  (h4 : InMinorArc E AC)
  (h5 : O.perpendicular_to AC) :
  ∠OBE.toDeg / ∠BAC.toDeg = 2 / 9 := sorry

end angle_ratio_l148_148593


namespace parametric_curve_segment_l148_148251

theorem parametric_curve_segment :
  ∀ (t : ℝ), -1 ≤ t ∧ t ≤ 1 →
  ∃ (x y : ℝ), x = 2 * t ∧ y = 2 ∧ (-2 ≤ x ∧ x ≤ 2) :=
by 
  intros t ht,
  use (2 * t),
  use 2,
  split,
  { exact rfl }, -- y = 2
  split;
    -- x ranges from -2 to 2
    linarith[ht.left, ht.right, show 2 > 0 by norm_num]

end parametric_curve_segment_l148_148251


namespace max_special_set_elements_l148_148215

-- Define the conditions for the set M
def is_special_set (M : set ℝ) : Prop :=
  ∀ x y z ∈ M, x ≠ y ∧ y ≠ z ∧ x ≠ z → (∃ a b ∈ {x, y, z}, a + b ∈ M)

-- State the main theorem
theorem max_special_set_elements :
  ∀ (M : set ℝ), is_special_set M → set.finite M → set.card M ≤ 7 :=
sorry

end max_special_set_elements_l148_148215


namespace minimum_number_of_distinct_complex_solutions_l148_148244

noncomputable def P : Polynomial ℝ := Polynomial.C 1 + Polynomial.X ^ 2
noncomputable def Q : Polynomial ℝ := Polynomial.C 2 - Polynomial.C 1 * Polynomial.X ^ 2 + Polynomial.X ^ 3
noncomputable def R : Polynomial ℝ := Polynomial.C 3 + Polynomial.X ^ 6

theorem minimum_number_of_distinct_complex_solutions :
  ∃ z : ℂ, P.eval z * Q.eval z = R.eval z :=
sorry

end minimum_number_of_distinct_complex_solutions_l148_148244


namespace earnings_per_word_l148_148585

-- Definitions based on conditions
def earnings_per_article : ℝ := 60
def number_of_articles : ℕ := 3
def hours : ℝ := 4
def words_per_minute : ℝ := 10
def earning_rate_per_hour : ℝ := 105

-- Theorem statement
theorem earnings_per_word (earnings_per_article : ℝ)
                          (number_of_articles : ℕ)
                          (hours : ℝ)
                          (words_per_minute : ℝ)
                          (earning_rate_per_hour : ℝ) :
  let total_earnings := earning_rate_per_hour * hours in
  let total_words := words_per_minute * 60 * hours in
  let article_earnings := earnings_per_article * number_of_articles in
  let word_earnings := total_earnings - article_earnings in
  let earnings_per_word := word_earnings / total_words in
  earnings_per_word = 0.10 := 
by
  sorry

end earnings_per_word_l148_148585


namespace problem_1_problem_2_l148_148088

theorem problem_1 (α : ℝ) (hα : Real.tan α = 2) :
  Real.tan (α + Real.pi / 4) = -3 :=
by
  sorry

theorem problem_2 (α : ℝ) (hα : Real.tan α = 2) :
  (6 * Real.sin α + Real.cos α) / (3 * Real.sin α - 2 * Real.cos α) = 13 / 4 :=
by
  sorry

end problem_1_problem_2_l148_148088


namespace probability_of_at_most_3_heads_l148_148476

-- Definitions based on conditions
def binom : ℕ → ℕ → ℕ
| 0, 0       => 1
| 0, (k + 1) => 0
| (n + 1), 0 => 1
| (n + 1), (k + 1) => binom n k + binom n (k + 1)

def favorable_outcomes : ℕ :=
binom 10 0 + binom 10 1 + binom 10 2 + binom 10 3

def total_outcomes : ℕ := 2 ^ 10

def probability (favorable : ℕ) (total : ℕ) : ℚ :=
favorable / total

-- Theorem statement
theorem probability_of_at_most_3_heads : 
  probability favorable_outcomes total_outcomes = 11 / 64 :=
sorry

end probability_of_at_most_3_heads_l148_148476


namespace poly_remainder_correct_l148_148894

noncomputable def poly_remainder : ℤ[X] :=
  (X - 1)^2007 % (X^2 - X + 1)

theorem poly_remainder_correct :
  poly_remainder = -X + 1 :=
by
  sorry

end poly_remainder_correct_l148_148894


namespace slope_of_FM_l148_148722

theorem slope_of_FM
  (a b : ℝ)
  (h1 : a > b)
  (h2 : b > 0)
  (c : ℝ)
  (h3 : c = (Real.sqrt 3 / 3) * a)
  (h4 : b^2 = (2 / 3) * a^2)
  (M : ℝ × ℝ)
  (h5 : (M.1^2 / a^2 + M.2^2 / b^2 = 1 ∧ M.1 > 0 ∧ M.2 > 0))
  (h6 : let k := M.2 / (M.1 - c) in (M.1^2 + ((M.2 / (M.1 - c)) * M.1 - c)^2 = b^2 / 4 ∧ c = Real.sqrt (1 + (M.2 / (M.1 - c))^2) * Real.sqrt (((M.2 / (M.1 - c)) * (2 * c) / (1 + (M.2 / (M.1 - c))^2))^2 - 4 * (M.1 * (M.2 / (M.1 - c)) * ((M.2 / (M.1 - c)) * c^2 - b^2 / 4) / (1 + (M.2 / (M.1 - c))^2))))
  : M.2 / (M.1 - c) = Real.sqrt 3 / 3 :=
begin
  sorry
end

end slope_of_FM_l148_148722


namespace closed_formula_for_f_l148_148940

noncomputable def f (x : ℝ) : ℝ := (x^3 - x^2 - 1) / (2 * x^2 - 2 * x)

theorem closed_formula_for_f :
  ∀ (x : ℝ), x ≠ 0 ∧ x ≠ 1 → f(x) + f(1 - 1 / x) = 1 + x :=
by
  intro x
  intro hx
  rw [f, f]
  -- Proof of equivalence is omitted but should follow from given functional equation details.
  sorry

end closed_formula_for_f_l148_148940


namespace andy_initial_cookies_l148_148608

def initial_cookies (andy_ate : ℕ) (brother_ate : ℕ) (team_sizes : ℕ) (cookies_taken_by_team : ℕ → ℕ) : ℕ :=
  andy_ate + brother_ate + (Finset.range team_sizes).sum cookies_taken_by_team

theorem andy_initial_cookies :
  initial_cookies 3 5 8 (λ n, 2 * n + 1) = 72 :=
by
  sorry

end andy_initial_cookies_l148_148608


namespace bronson_profit_l148_148622

theorem bronson_profit :
  let cost_per_bushel := 12
  let apples_per_bushel := 48
  let selling_price_per_apple := 0.40
  let apples_sold := 100
  let cost_per_apple := cost_per_bushel / apples_per_bushel
  let profit_per_apple := selling_price_per_apple - cost_per_apple
  let total_profit := profit_per_apple * apples_sold
  in
  total_profit = 15 := 
by 
  sorry

end bronson_profit_l148_148622


namespace allocation_scheme_count_l148_148062

theorem allocation_scheme_count :
  ∃ (volunteers : Finset ℕ) (projects : Finset ℕ), 
  volunteers.card = 5 ∧ projects.card = 4 ∧ 
  (∀ volunteer ∈ volunteers, ∃ project ∈ projects, true) ∧
  (∀ project ∈ projects, ∃ volunteer ∈ volunteers, true) ∧ 
  (sum (λ v, if ∃ p ∈ projects, true then 1 else 0) volunteers) = 240 :=
sorry

end allocation_scheme_count_l148_148062


namespace oranges_per_rupee_initial_rate_l148_148579

noncomputable def initial_selling_rate (r: ℝ) (l: ℝ) (g: ℝ) : ℝ :=
  let C := (1 / (r * g))
  0.92 * C

theorem oranges_per_rupee_initial_rate :
  let rate_gain_45 := 11.420689655172414
  let loss_8_percent := 0.92
  let gain_45_percent := 1.45
  (rate_gain_45 * gain_45_percent = 1.45) →
  initial_selling_rate rate_gain_45 loss_8_percent gain_45_percent = 1 / 18 :=
by
  intros
  have h1 : 1 / rate_gain_45 = 1 / 11.420689655172414 := by sorry
  have h2 : (1 / rate_gain_45) / gain_45_percent = 1 / (rate_gain_45 * gain_45_percent) := by sorry
  have h3 : initial_selling_rate rate_gain_45 loss_8_percent gain_45_percent = 0.92 * (1 / (rate_gain_45 * gain_45_percent)) := by sorry
  have h4 : 0.92 / (11.420689655172414 * 1.45) ≈ 0.05555555555555555 := by sorry
  show initial_selling_rate rate_gain_45 loss_8_percent gain_45_percent = 1 / 18 from sorry

end oranges_per_rupee_initial_rate_l148_148579


namespace mixed_doubles_selection_l148_148761

theorem mixed_doubles_selection (males females : ℕ) (hm : males = 5) (hf : females = 4) : 
  (males * females) = 20 :=
by
  rw [hm, hf]
  exact 5 * 4 = 20

end mixed_doubles_selection_l148_148761


namespace probability_at_most_3_heads_10_flips_l148_148511

theorem probability_at_most_3_heads_10_flips : 
  let num_heads_0 := nat.choose 10 0,
      num_heads_1 := nat.choose 10 1,
      num_heads_2 := nat.choose 10 2,
      num_heads_3 := nat.choose 10 3,
      total_outcomes := 2^10 in
  (num_heads_0 + num_heads_1 + num_heads_2 + num_heads_3) / total_outcomes = 11 / 64 :=
by
  sorry

end probability_at_most_3_heads_10_flips_l148_148511


namespace six_digit_even_count_not_adjacent_even_count_exactly_two_adjacent_even_count_odd_digits_ascending_count_l148_148601

-- Conditions: Digits 1, 2, 3, 4, 5, 6 without repetition
def digits : List Nat := [1, 2, 3, 4, 5, 6]

-- Definitions of conditions for generating numbers
def isSixDigit (n : Nat) : Prop := (10^5 ≤ n) ∧ (n < 10^6)
def isEven (n : Nat) : Prop := n % 2 = 0
def isEvenCombination (l : List Nat) : Prop := l.filter (λ x => isEven x).length = 3
def notAdjacent (l : List Nat) : Prop := ¬ List.any (List.zip l (List.tail l)) (λ x => (isEven x.fst) ∧ (isEven x.snd))
def exactlyTwoAdjacent (l : List Nat) : Prop := l.filter (λ x => isEven x).length = 2 ∧ List.any (List.zip l (List.tail l)) (λ x => (isEven x.fst) ∧ (isEven x.snd))
def oddAscending (l : List Nat) : Prop := l.filter (λ x => ¬ isEven x) = List.sort (l.filter (λ x => ¬ isEven x))

-- Proof problems equivalent to the given questions
theorem six_digit_even_count : (List.permutations digits).countp (λ l => isSixDigit (Nat.ofDigits 10 l) ∧ isEven (Nat.ofDigits 10 l)) = 360 := 
sorry

theorem not_adjacent_even_count : (List.permutations digits).countp (λ l => isSixDigit (Nat.ofDigits 10 l) ∧ isEvenCombination l ∧ notAdjacent l) = 144 := 
sorry

theorem exactly_two_adjacent_even_count : (List.permutations digits).countp (λ l => isSixDigit (Nat.ofDigits 10 l) ∧ exactlyTwoAdjacent l) = 432 := 
sorry

theorem odd_digits_ascending_count : (List.permutations digits).countp (λ l => isSixDigit (Nat.ofDigits 10 l) ∧ oddAscending l) = 120 := 
sorry

end six_digit_even_count_not_adjacent_even_count_exactly_two_adjacent_even_count_odd_digits_ascending_count_l148_148601


namespace series_sum_199_l148_148018

noncomputable def seriesSum : ℕ → ℤ
| 0       => 1
| (n + 1) => seriesSum n + (-1)^(n + 1) * (n + 2)

theorem series_sum_199 : seriesSum 199 = 100 := 
by
  sorry

end series_sum_199_l148_148018


namespace island_length_l148_148605

theorem island_length (area width : ℝ) (h_area : area = 50) (h_width : width = 5) : 
  area / width = 10 := 
by
  sorry

end island_length_l148_148605


namespace unique_positive_odd_integers_product_count_l148_148748

-- Define the predicate for a number being a positive odd integer less than 1000
def positive_odd_integer_less_than_1000 (n : ℕ) : Prop :=
  n < 1000 ∧ n % 2 = 1

-- Define the predicate for a number being a positive multiple of 5
def positive_multiple_of_five (m : ℕ) : Prop :=
  m > 0 ∧ m % 5 = 0

-- Define the main predicate for the problem
def unique_odd_number_product (n : ℕ) : Prop :=
  ∃ m k : ℕ, positive_multiple_of_five m ∧ positive_odd_integer_less_than_1000 k ∧ n = m * k

-- The theorem statement
theorem unique_positive_odd_integers_product_count : 
  (finset.filter unique_odd_number_product (finset.range 1000)).card = 20 :=
by
  sorry

end unique_positive_odd_integers_product_count_l148_148748


namespace Papi_Calot_plants_l148_148823

theorem Papi_Calot_plants :
  let initial_potatoes_plants := 10 * 25
  let initial_carrots_plants := 15 * 30
  let initial_onions_plants := 12 * 20
  let total_potato_plants := initial_potatoes_plants + 20
  let total_carrot_plants := initial_carrots_plants + 30
  let total_onion_plants := initial_onions_plants + 10
  total_potato_plants = 270 ∧
  total_carrot_plants = 480 ∧
  total_onion_plants = 250 := by
  sorry

end Papi_Calot_plants_l148_148823


namespace probability_of_at_most_3_heads_l148_148467

-- Definitions based on conditions
def binom : ℕ → ℕ → ℕ
| 0, 0       => 1
| 0, (k + 1) => 0
| (n + 1), 0 => 1
| (n + 1), (k + 1) => binom n k + binom n (k + 1)

def favorable_outcomes : ℕ :=
binom 10 0 + binom 10 1 + binom 10 2 + binom 10 3

def total_outcomes : ℕ := 2 ^ 10

def probability (favorable : ℕ) (total : ℕ) : ℚ :=
favorable / total

-- Theorem statement
theorem probability_of_at_most_3_heads : 
  probability favorable_outcomes total_outcomes = 11 / 64 :=
sorry

end probability_of_at_most_3_heads_l148_148467


namespace f_10_is_11_l148_148257

def f : ℕ → ℕ :=
sorry

axiom f_recurrence (x : ℕ) : f(x + 1) = f(x) + 1
axiom f_initial : f 0 = 1

theorem f_10_is_11 : f 10 = 11 :=
by
sory

end f_10_is_11_l148_148257


namespace probability_divisible_by_7_is_2_over_5_l148_148145

open Function

/-- If a five-digit number has digits summing to 44, prove that the probability of it being divisible by 7 is 2/5 --/
theorem probability_divisible_by_7_is_2_over_5:
  let numbers := [99998, 99989, 99899, 98999, 89999]
  have valid_numbers: ∀ n ∈ numbers, digits_sum n = 44 := by sorry
  let divisible_by_7 := [98999, 89999]
  ∃ valid_divisible_by_7 ∈ numbers, valid_divisible_by_7 % 7 = 0 ∧
    length divisible_by_7 / length numbers = 2 / 5 := by
begin
  -- sorry for the proof here, if any proof steps might be needed
  sorry
end

end probability_divisible_by_7_is_2_over_5_l148_148145


namespace beef_weight_after_processing_l148_148589

-- Problem definitions based on the conditions
variable (initial_weight : ℝ) (weight_lost_percentage : ℝ)
def weight_after_processing (initial_weight : ℝ) (weight_lost_percentage : ℝ) : ℝ :=
  (1 - weight_lost_percentage / 100) * initial_weight

-- Assertion that needs to be proved
theorem beef_weight_after_processing
  (h_initial : initial_weight = 800)
  (h_lost : weight_lost_percentage = 20) :
  weight_after_processing initial_weight weight_lost_percentage = 640 :=
by
  sorry

end beef_weight_after_processing_l148_148589


namespace triangle_to_square_impossible_l148_148786

theorem triangle_to_square_impossible 
  (AC BC : ℕ)
  (hAC : AC = 20000)
  (hBC : BC = 1 / 10000) :
  ¬(∃ (parts : Finset (Finset (ℝ × ℝ))), parts.card = 1000 ∧
    (∀ (p ∈ parts) (q ∈ parts), p ≠ q → interior p ∩ interior q = ∅) ∧
    (⋃₀ parts = (λ x y, y = 0 ∧ 0 ≤ x ∧ x ≤ AC) ∪ (λ x y, x = AC ∧ 0 ≤ y ∧ y ≤ BC)) ∧
    (∃ square : Finset (ℝ × ℝ), is_square square ∧ 
        (⋃₀ parts = ⋃₀ square))) :=
begin
  sorry
end

end triangle_to_square_impossible_l148_148786


namespace largest_number_2013_l148_148876

theorem largest_number_2013 (x y : ℕ) (h1 : x + y = 2013)
    (h2 : y = 5 * (x / 100 + 1)) : max x y = 1913 := by
  sorry

end largest_number_2013_l148_148876


namespace is_simple_random_sample_D_l148_148323

def sampling_method_A : Prop :=
  ∃ (n : ℕ), ∀ (postcards : list ℕ), postcards.length = 10^6 →
    ∃ (x ∈ postcards), x % 10000 = 2709

def sampling_method_B : Prop :=
  ∃ (t : ℕ), t = 30 ∧ ∃ (weights : list ℕ), ∀ i < weights.length, i % t = 0

def sampling_method_C : Prop :=
  ∃ (admin : ℕ) (teachers : ℕ) (logistics : ℕ), admin = 2 ∧ teachers = 14 ∧ logistics = 4

def sampling_method_D : Prop :=
  ∃ (n m : ℕ), n = 10 ∧ m = 3 ∧ ∃ (products : list ℕ), products.length = n →
    ∃ (selected : list ℕ), selected.length = m ∧ selected ⊆ products

theorem is_simple_random_sample_D : sampling_method_D :=
  sorry

end is_simple_random_sample_D_l148_148323


namespace range_of_a_for_increasing_function_l148_148258

noncomputable def y (x a : ℝ) : ℝ := x^3 + a*x^2 + x

theorem range_of_a_for_increasing_function :
  (∀ x : ℝ, deriv (y x a) x ≥ 0) ↔ (-Real.sqrt 3 ≤ a ∧ a ≤ Real.sqrt 3) :=
by
  sorry

end range_of_a_for_increasing_function_l148_148258


namespace problem_eq_995_l148_148978

theorem problem_eq_995 :
  (12^4 + 400) * (24^4 + 400) * (36^4 + 400) * (48^4 + 400) * (60^4 + 400) /
  ((6^4 + 400) * (18^4 + 400) * (30^4 + 400) * (42^4 + 400) * (54^4 + 400))
  = 995 := sorry

end problem_eq_995_l148_148978


namespace probability_of_at_most_3_heads_l148_148411

-- Definitions and conditions
def num_coins : ℕ := 10
def at_most_3_heads_probability : ℚ := 11 / 64

-- Statement of the problem
theorem probability_of_at_most_3_heads (n : ℕ) (p : ℚ) (h1 : n = num_coins) (h2 : p = at_most_3_heads_probability) :
  p = (1 + 10 + 45 + 120 : ℕ) / (2 ^ 10 : ℕ) := by
  sorry

end probability_of_at_most_3_heads_l148_148411


namespace sally_bread_consumption_l148_148838

/-
Given:
    1) Sally eats 2 sandwiches on Saturday
    2) Sally eats 1 sandwich on Sunday
    3) Each sandwich uses 2 pieces of bread

Prove:
    Sally eats 6 pieces of bread across Saturday and Sunday
-/
theorem sally_bread_consumption (sandwiches_sat : Nat) (sandwiches_sun : Nat) (bread_per_sandwich : Nat)
    (H1 : sandwiches_sat = 2) (H2 : sandwiches_sun = 1) (H3 : bread_per_sandwich = 2) :
    2 * bread_per_sandwich + 1 * bread_per_sandwich = 6 := by
  sorry

end sally_bread_consumption_l148_148838


namespace length_PR_correct_l148_148170

def length_PR (AB AD : ℝ) (area_cutoff : ℝ) : ℝ :=
  let x := sqrt (area_cutoff / 2)
  AB - 2 * x

theorem length_PR_correct (AB AD : ℝ) (area_cutoff : ℝ)
  (hAB : AB = 18) (hAD : AD = 12)
  (h_area_cutoff : area_cutoff = 180) :
  length_PR AB AD area_cutoff = 18 - 6 * sqrt 10 :=
by
  apply congrArg2
  sorry

end length_PR_correct_l148_148170


namespace magnitude_a_eq_3sqrt10_l148_148741

noncomputable def vector_a (t : ℝ) : ℝ × ℝ := (t - 2, 3)
def vector_b : ℝ × ℝ := (3, -1)

theorem magnitude_a_eq_3sqrt10 (t : ℝ) 
  (h : ∃ k : ℝ, ∀ (x y : ℝ), x = t + 4 → y = 1 → (x, y) = k • (3, -1)) :
  |(t - 2, 3)| = 3 * real.sqrt 10 :=
by sorry

end magnitude_a_eq_3sqrt10_l148_148741


namespace slope_of_line_determined_by_solutions_l148_148304

theorem slope_of_line_determined_by_solutions :
  ∀ (x1 x2 y1 y2 : ℝ), 
  (4 / x1 + 6 / y1 = 0) → (4 / x2 + 6 / y2 = 0) →
  (y2 - y1) / (x2 - x1) = -3 / 2 :=
by
  intros x1 x2 y1 y2 h1 h2
  -- Proof steps go here
  sorry

end slope_of_line_determined_by_solutions_l148_148304


namespace vector_ratio_l148_148205

theorem vector_ratio (A B P: Type) [AddCommGroup P] [Module ℝ P] (overrightarrow : A → P) (ratio : ℝ) (AP_PB : ratio = 4 /(1 + 4)) :
  ∃ t u : ℝ, (∀ p q : A, overrightarrow p = t * overrightarrow p + u * overrightarrow q) ∧ (t = 4/5 ∧ u = 1/5) :=
by
  use (4 / 5), (1 / 5)
  split
  { intros p q
    simp [ratio] }
  { exact ⟨rfl, rfl⟩ }
  sorry

end vector_ratio_l148_148205


namespace convert_568_to_base_8_l148_148029

theorem convert_568_to_base_8 : ∀ (n : ℕ), n = 568 → to_base_8 n = 1070 := by
  intro n hn
  rw hn
  sorry

end convert_568_to_base_8_l148_148029


namespace prob_heads_at_most_3_out_of_10_flips_l148_148458

def coin_flips := 10
def max_heads := 3
def total_outcomes := 2^coin_flips
def favorable_outcomes := (nat.choose coin_flips 0) +
                          (nat.choose coin_flips 1) +
                          (nat.choose coin_flips 2) +
                          (nat.choose coin_flips 3)

theorem prob_heads_at_most_3_out_of_10_flips :
  (favorable_outcomes : ℚ) / total_outcomes = 11 / 64 :=
by
  sorry

end prob_heads_at_most_3_out_of_10_flips_l148_148458


namespace sum_of_digits_base_7_of_999_l148_148309

theorem sum_of_digits_base_7_of_999 : 
  (let n := 999;
       repr_base_7 := [2, 6, 2, 5];
       sum_digits := repr_base_7.foldl (λ x y, x + y) 0
   in sum_digits = 15) :=
by
  sorry

end sum_of_digits_base_7_of_999_l148_148309


namespace irrational_number_among_list_l148_148901

def isRational (x : ℝ) : Prop :=
  ∃ (a b : ℤ), b ≠ 0 ∧ x = a / b

theorem irrational_number_among_list :
  ∀ x ∈ ({1, Real.sqrt 4, 2 / 3: ℝ, Real.cbrt 3} : set ℝ), ¬ isRational x ↔ x = Real.cbrt 3 :=
by
  sorry

end irrational_number_among_list_l148_148901


namespace angle_A_eq_find_a_l148_148179

variable {a b c A B C : ℝ}

-- The conditions in the problem
def Equation1 (a c : ℝ) (A C : ℝ) : Prop :=
  a * sin C - c * cos (A - π / 6) = 0

def Equation2 (a b c : ℝ) (A B : ℝ) : Prop :=
  2 * c * cos A = a * cos B + b * cos A

def Equation3 (a b c : ℝ) : Prop :=
  b^2 + c^2 - a^2 = b * c

variable {area : ℝ}

-- Prove angle A given any of the equations
theorem angle_A_eq (h : Equation1 a c A C ∨ Equation2 a b c A B ∨ Equation3 a b c) : A = π / 3 :=
sorry

-- Given specific conditions to find side a
theorem find_a (A_eq : A = π / 3) (b_eq : b = 6) (area_eq : area = 3 * sqrt 3)
  (h_area : 3 * sqrt 3 = 1 / 2 * 6 * c * sin (π / 3)) : a = 2 * sqrt 7 :=
sorry

end angle_A_eq_find_a_l148_148179


namespace arithmetic_sequence_divisibility_l148_148656

theorem arithmetic_sequence_divisibility (a d : ℕ) (h: ∀ n : ℕ, (∏ i in Finset.range n, (a + i * d)) ∣ (∏ i in Finset.range n, (a + (n + i) * d))) :
  ∃ k : ℕ, ∀ n : ℕ, a + n * d = k * (n + 1) :=
begin
  sorry
end

end arithmetic_sequence_divisibility_l148_148656


namespace find_a_values_l148_148046

theorem find_a_values (a n : ℕ) (h1 : 7 * a * n - 3 * n = 2020) :
    a = 68 ∨ a = 289 := sorry

end find_a_values_l148_148046


namespace f_functional_eq_l148_148922

noncomputable def f (x : ℝ) : ℝ := sorry

theorem f_functional_eq (f : ℝ → ℝ) (h : ∀ x y : ℝ, f(f(x) + y) = f(x + y) + 2*x*f(y) - 3*x*y - 2*x + 2) : f 1 = 3 :=
by
  -- Proof goes here
  sorry

end f_functional_eq_l148_148922


namespace veranda_area_l148_148914

theorem veranda_area
  (room_length : ℝ) (room_width : ℝ)
  (veranda_width : ℝ)
  (room_length = 17)
  (room_width = 12)
  (veranda_width = 2) :
  let total_length := room_length + 2 * veranda_width,
      total_width := room_width + 2 * veranda_width,
      total_area := total_length * total_width,
      room_area := room_length * room_width in
  total_area - room_area = 132 := by
  sorry

end veranda_area_l148_148914


namespace sum_of_digits_base_7_of_999_l148_148307

theorem sum_of_digits_base_7_of_999 : 
  (let n := 999;
       repr_base_7 := [2, 6, 2, 5];
       sum_digits := repr_base_7.foldl (λ x y, x + y) 0
   in sum_digits = 15) :=
by
  sorry

end sum_of_digits_base_7_of_999_l148_148307


namespace number_of_elements_begin_with_digit_one_l148_148216

-- Define T
def T := {k : ℕ | 0 ≤ k ∧ k ≤ 1000}

-- Define the property for the number of digits
def num_digits (n : ℕ) : ℕ := (n : ℚ).log10.floor + 1

-- Given conditions from the problem
def three_pow_1000_has_477_digits : Prop := num_digits (3 ^ 1000) = 477

-- Theorem statement
theorem number_of_elements_begin_with_digit_one : 
  three_pow_1000_has_477_digits →
  (∑ k in T, if (3^k).nat_abs.to_digits.base 10 ∧ 10^(num_digits (3^k) - 1) = 1 then 1 else 0 = 524) :=
by sorry

end number_of_elements_begin_with_digit_one_l148_148216


namespace train_crossing_time_l148_148577

noncomputable def time_to_cross_platform
  (speed_kmph : ℝ)
  (length_train : ℝ)
  (length_platform : ℝ) : ℝ :=
  let speed_ms := speed_kmph / 3.6
  let total_distance := length_train + length_platform
  total_distance / speed_ms

theorem train_crossing_time
  (speed_kmph : ℝ)
  (length_train : ℝ)
  (length_platform : ℝ)
  (h_speed : speed_kmph = 72)
  (h_train_length : length_train = 280.0416)
  (h_platform_length : length_platform = 240) :
  time_to_cross_platform speed_kmph length_train length_platform = 26.00208 := by
  sorry

end train_crossing_time_l148_148577


namespace prob_heads_at_most_3_out_of_10_flips_l148_148454

def coin_flips := 10
def max_heads := 3
def total_outcomes := 2^coin_flips
def favorable_outcomes := (nat.choose coin_flips 0) +
                          (nat.choose coin_flips 1) +
                          (nat.choose coin_flips 2) +
                          (nat.choose coin_flips 3)

theorem prob_heads_at_most_3_out_of_10_flips :
  (favorable_outcomes : ℚ) / total_outcomes = 11 / 64 :=
by
  sorry

end prob_heads_at_most_3_out_of_10_flips_l148_148454


namespace sum_of_angles_in_circumscribed_quadrilateral_l148_148570

theorem sum_of_angles_in_circumscribed_quadrilateral:
  ∀ (W X Y Z : Type) (circle : circle Type)
  (WXY_inscribed: angle W X Y = 50) (YZW_inscribed: angle Y Z W = 70),
  angle Y W Z + angle X Y Z = 120 := by
  sorry

end sum_of_angles_in_circumscribed_quadrilateral_l148_148570


namespace probability_heads_at_most_three_out_of_ten_coins_l148_148553

theorem probability_heads_at_most_three_out_of_ten_coins :
  let total_outcomes := 2^10 in
  let favorable_outcomes := Nat.choose 10 0 + Nat.choose 10 1 + Nat.choose 10 2 + Nat.choose 10 3 in
  favorable_outcomes / total_outcomes = 11 / 64 :=
by
  sorry

end probability_heads_at_most_three_out_of_ten_coins_l148_148553


namespace tangent_line_at_A_l148_148152

def f (x : ℝ) : ℝ := x ^ (1 / 2)

def tangent_line_equation (x y: ℝ) : Prop :=
  4 * x - 4 * y + 1 = 0

theorem tangent_line_at_A :
  tangent_line_equation (1/4) (f (1/4)) :=
by
  sorry

end tangent_line_at_A_l148_148152


namespace probability_at_most_3_heads_10_coins_l148_148419

noncomputable def coin_flip_prob_at_most_3_heads : ℚ :=
  let total_outcomes := 2^10
  let favorable_outcomes := Nat.binomial 10 0 + Nat.binomial 10 1 + Nat.binomial 10 2 + Nat.binomial 10 3
  favorable_outcomes / total_outcomes

theorem probability_at_most_3_heads_10_coins : coin_flip_prob_at_most_3_heads = 11 / 64 :=
by
  -- Definitions of let values are integrated directly in the theorem body
  let total_outcomes := 2^10
  let favorable_outcomes := Nat.binomial 10 0 + Nat.binomial 10 1 + Nat.binomial 10 2 + Nat.binomial 10 3
  show favorable_outcomes / total_outcomes = 11 / 64
  sorry

end probability_at_most_3_heads_10_coins_l148_148419


namespace reflection_through_plane_l148_148801

def normal_vector : ℝ × ℝ × ℝ := (2, -1, 1)

def reflection_matrix : Matrix (Fin 3) (Fin 3) ℝ :=
  ![
    [-2 / 3, 1 / 3, 5 / 3],
    [5 / 3, 4 / 3, -4 / 3],
    [-7 / 3, 10 / 3, 4 / 3]
  ]

theorem reflection_through_plane (w : Fin 3 → ℝ) : 
  reflection_matrix.mulVec w =
  sorry

end reflection_through_plane_l148_148801


namespace probability_10_coins_at_most_3_heads_l148_148387

def probability_at_most_3_heads (n : ℕ) : ℚ :=
  let total_outcomes := 2^n
  let favorable_outcomes := (Nat.choose n 0) + (Nat.choose n 1) + (Nat.choose n 2) + (Nat.choose n 3)
  favorable_outcomes / total_outcomes

theorem probability_10_coins_at_most_3_heads : probability_at_most_3_heads 10 = 11 / 64 :=
by
  sorry

end probability_10_coins_at_most_3_heads_l148_148387


namespace propositions_correctness_l148_148281

theorem propositions_correctness :
  let p₁ := (∀ A B, (¬ (corresponding_angles_equal A B) → ¬ (parallel A B)))
  let p₂ := (∃ α, (sin α = 1/2) ∧ (α ≠ 30))
  let p₃ := (∀ p q, (¬(p ∧ q) → (¬p ∨ ¬q)) ∧ (¬(p ∧ q) → ¬p ∧ ¬q))
  let p₄ := (∀ x : ℝ, ¬(∃ x₀, (x₀^2 + 2*x₀ + 2 ≤ 0)) → (x^2 + 2*x + 2 > 0))
   
  p₁ ∧ p₂ ∧ ¬p₃ ∧ p₄ :=
by
  intro p₁ p₂ p₃ p₄
  exact ⟨
    p₁, 
    p₂, 
    false.elim (p₃.2 p₃.1), 
    p₄ 
  ⟩

end propositions_correctness_l148_148281


namespace number_of_roots_l148_148211

noncomputable def g : ℝ → ℝ := sorry

theorem number_of_roots (h1 : ∀ x, g (3 + x) = g (3 - x))
                        (h2 : ∀ x, g (8 + x) = g (8 - x))
                        (h3 : g 0 = 0) :
  ∃ n ≥ 402, (range (λ m : ℤ, g (10 * m)) ++ range (λ m : ℤ, g (6 + 10 * m)))
    .filter (λ x, x ≥ -1000 ∧ x ≤ 1000) = 0 :=
sorry

end number_of_roots_l148_148211


namespace arrangement_count_l148_148009

theorem arrangement_count (basil_plants tomato_plants : ℕ) (b : basil_plants = 5) (t : tomato_plants = 4) : 
  (Nat.factorial (basil_plants + 1) * Nat.factorial tomato_plants) = 17280 :=
by
  rw [b, t] 
  exact Eq.refl 17280

end arrangement_count_l148_148009


namespace probability_of_at_most_3_heads_out_of_10_l148_148488
open_locale big_operators
open nat

theorem probability_of_at_most_3_heads_out_of_10 (n : ℕ) (p : ℕ) (h : 0 < n) (hp : p = 2) :
  ∑ k in finset.range (4), nat.choose n k = 176 → nat.pow p n = 1024 → (∑ k in finset.range (4), nat.choose n k) / (nat.pow p n) = 11 / 64 :=
by { intro h₁ h₂, 
     have : (∑ k in finset.range 4, nat.choose n k) / (nat.pow p n) = 176 / 1024,
     { rw [h₁, h₂] },
     rw this,
     norm_num }

end probability_of_at_most_3_heads_out_of_10_l148_148488


namespace Barbara_spent_46_22_on_different_goods_l148_148615

theorem Barbara_spent_46_22_on_different_goods :
  let tuna_cost := (5 * 2) -- Total cost of tuna
  let water_cost := (4 * 1.5) -- Total cost of water
  let total_before_discount := 56 / 0.9 -- Total before discount, derived from the final amount paid after discount
  let total_tuna_water_cost := 10 + 6 -- Total cost of tuna and water together
  let different_goods_cost := total_before_discount - total_tuna_water_cost
  different_goods_cost = 46.22 := 
sorry

end Barbara_spent_46_22_on_different_goods_l148_148615


namespace part1_part2_l148_148111

noncomputable def f (x : ℝ) : ℝ := 
  Real.sin (2 * x + Real.pi / 6) - 2 * (Real.cos x) ^ 2

theorem part1 : f (Real.pi / 6) = -1 / 2 := 
  by sorry

theorem part2 : ∃ x ∈ Icc (-Real.pi / 3) (Real.pi / 6), 
  ∀ y ∈ Icc (-Real.pi / 3) (Real.pi / 6), f y ≤ f x ∧ f x = 0 := 
  by sorry

end part1_part2_l148_148111


namespace problem_1_problem_2_l148_148048

theorem problem_1 (p x : ℝ) (h1 : |p| ≤ 2) (h2 : x^2 + p*x + 1 > 2*x + p) : x < -1 ∨ x > 3 :=
sorry

theorem problem_2 (p x : ℝ) (h1 : 2 ≤ x) (h2 : x ≤ 4) (h3 : x^2 + p*x + 1 > 2*x + p) : p > -1 :=
sorry

end problem_1_problem_2_l148_148048


namespace probability_heads_at_most_three_out_of_ten_coins_l148_148558

theorem probability_heads_at_most_three_out_of_ten_coins :
  let total_outcomes := 2^10 in
  let favorable_outcomes := Nat.choose 10 0 + Nat.choose 10 1 + Nat.choose 10 2 + Nat.choose 10 3 in
  favorable_outcomes / total_outcomes = 11 / 64 :=
by
  sorry

end probability_heads_at_most_three_out_of_ten_coins_l148_148558


namespace find_theta_even_fn_l148_148098

noncomputable def f (x θ : ℝ) := Real.sin (x + θ) + Real.cos (x + θ)

theorem find_theta_even_fn (θ : ℝ) (hθ: 0 ≤ θ ∧ θ ≤ π / 2) 
  (h: ∀ x : ℝ, f x θ = f (-x) θ) : θ = π / 4 :=
by sorry

end find_theta_even_fn_l148_148098


namespace probability_at_most_3_heads_l148_148432

def binom (n k : ℕ) : ℕ := Nat.choose n k

theorem probability_at_most_3_heads :
  let total_outcomes := 2^10
  let successful_outcomes := binom 10 0 + binom 10 1 + binom 10 2 + binom 10 3
  (successful_outcomes : ℚ) / total_outcomes = 11 / 64 := 
by 
  let total_outcomes := 2 ^ 10
  let successful_outcomes := binom 10 0 + binom 10 1 + binom 10 2 + binom 10 3
  have h : (successful_outcomes : ℚ) / total_outcomes = 11 / 64 := sorry
  exact h

end probability_at_most_3_heads_l148_148432


namespace problem_statement_l148_148723

noncomputable def f (a x : ℝ) : ℝ := Real.log (x^2 + 1) / Real.log a

theorem problem_statement (a : ℝ) (h_a : 1 < a) :
  (∀ x : ℝ, f a(-x) = f a(x)) ∧ (set.range (f a) = set.Ici 0) :=
by
  sorry

end problem_statement_l148_148723


namespace prove_angle_PMN_l148_148771

def angle_PQR : ℝ := 60  -- given angle PQR is 60 degrees
def PR_eq_RQ : Prop := PR = RQ  -- triangle PQR is isosceles with sides PR and RQ equal
def PM_eq_PN : Prop := PM = PN  -- PM equals PN
def angle_PMN : ℝ := 60  -- we need to prove angle PMN is 60 degrees

theorem prove_angle_PMN (h1 : angle_PQR = 60) (h2 : PR_eq_RQ) (h3 : PM_eq_PN) : angle_PMN = 60 :=
  sorry

end prove_angle_PMN_l148_148771


namespace sum_equation_l148_148974

theorem sum_equation : (∑ n in finset.range 2022, n * (2022 - n)) = 2021 * 1011 * 674 := 
sorry

end sum_equation_l148_148974


namespace value_of_power_l148_148688

theorem value_of_power (a : ℝ) (m n k : ℕ) (h1 : a ^ m = 2) (h2 : a ^ n = 4) (h3 : a ^ k = 32) : 
  a ^ (3 * m + 2 * n - k) = 4 := 
by sorry

end value_of_power_l148_148688


namespace minimum_sum_l148_148735

variables {A B P Q M : Point} {a b : ℝ} [positive_a : a > 0] [positive_b : b > 0]

-- Assume points A and B are on different sides of line PQ 
def different_sides (A B P Q : Point) : Prop := 
-- You would define the specifics of what different sides mean in your geometry library

-- Define the given conditions
def given_conditions (A B P Q M : Point) (a b : ℝ) : Prop :=
(A B P Q are different sides) ∧ (M on PQ) ∧ (a > 0) ∧ (b > 0) ∧
(∃ (α β : ℝ), (cos α / cos β) = (a / b) ∧ (angle A M P = α) ∧ (angle B M Q = β))

-- Define the statement that needs to be proved
theorem minimum_sum (A B P Q M : Point) (a b : ℝ) (h₁ : given_conditions A B P Q M a b) :
∀ (X : Point), X ≠ M → X on PQ → 
  ((AM / a + BM / b) < (AX / a + BX / b)) :=
by 
-- Here you can add your proof
sorry

end minimum_sum_l148_148735


namespace value_of_a_l148_148872

theorem value_of_a
  (h : { x : ℝ | -3 < x ∧ x < -1 ∨ 2 < x } = { x : ℝ | (x+a)/(x^2 + 4*x + 3) > 0}) : a = -2 :=
by sorry

end value_of_a_l148_148872


namespace arithmetic_seq_div_l148_148659

open Nat

theorem arithmetic_seq_div (a d : ℕ) (h : ∀ n : ℕ, 
  ∏ i in range n, (a + i*d) ∣ ∏ i in range n, (a + (i+n)*d)) :
  ∃ k : ℕ, ∀ n : ℕ, a + n*d = k*(n+1) := sorry

end arithmetic_seq_div_l148_148659


namespace probability_at_most_3_heads_l148_148508

-- Definitions and conditions
def coin_flips : ℕ := 10
def success_prob : ℚ := 1/2
def max_heads : ℕ := 3

-- Binomial Probability Computation
noncomputable def probability_of_at_most_3_heads : ℚ :=
  (∑ k in finset.range (max_heads + 1), nat.choose coin_flips k) / (2^coin_flips)

-- The statement to prove
theorem probability_at_most_3_heads :
  probability_of_at_most_3_heads = 11/64 := by
  sorry

end probability_at_most_3_heads_l148_148508


namespace part1_part2_l148_148724

open Real

noncomputable def f (x a : ℝ) : ℝ := x * exp (a * x) + x * cos x + 1

theorem part1 (x : ℝ) (hx : 0 ≤ x) : cos x ≥ 1 - (1 / 2) * x^2 := 
sorry

theorem part2 (a x : ℝ) (ha : 1 ≤ a) (hx : 0 ≤ x) : f x a ≥ (1 + sin x)^2 := 
sorry

end part1_part2_l148_148724


namespace prob_heads_at_most_3_out_of_10_flips_l148_148459

def coin_flips := 10
def max_heads := 3
def total_outcomes := 2^coin_flips
def favorable_outcomes := (nat.choose coin_flips 0) +
                          (nat.choose coin_flips 1) +
                          (nat.choose coin_flips 2) +
                          (nat.choose coin_flips 3)

theorem prob_heads_at_most_3_out_of_10_flips :
  (favorable_outcomes : ℚ) / total_outcomes = 11 / 64 :=
by
  sorry

end prob_heads_at_most_3_out_of_10_flips_l148_148459


namespace probability_of_at_most_3_heads_l148_148468

-- Definitions based on conditions
def binom : ℕ → ℕ → ℕ
| 0, 0       => 1
| 0, (k + 1) => 0
| (n + 1), 0 => 1
| (n + 1), (k + 1) => binom n k + binom n (k + 1)

def favorable_outcomes : ℕ :=
binom 10 0 + binom 10 1 + binom 10 2 + binom 10 3

def total_outcomes : ℕ := 2 ^ 10

def probability (favorable : ℕ) (total : ℕ) : ℚ :=
favorable / total

-- Theorem statement
theorem probability_of_at_most_3_heads : 
  probability favorable_outcomes total_outcomes = 11 / 64 :=
sorry

end probability_of_at_most_3_heads_l148_148468


namespace probability_of_at_most_3_heads_out_of_10_l148_148492
open_locale big_operators
open nat

theorem probability_of_at_most_3_heads_out_of_10 (n : ℕ) (p : ℕ) (h : 0 < n) (hp : p = 2) :
  ∑ k in finset.range (4), nat.choose n k = 176 → nat.pow p n = 1024 → (∑ k in finset.range (4), nat.choose n k) / (nat.pow p n) = 11 / 64 :=
by { intro h₁ h₂, 
     have : (∑ k in finset.range 4, nat.choose n k) / (nat.pow p n) = 176 / 1024,
     { rw [h₁, h₂] },
     rw this,
     norm_num }

end probability_of_at_most_3_heads_out_of_10_l148_148492


namespace more_newborn_elephants_than_baby_hippos_l148_148004

-- Define the given conditions
def initial_elephants := 20
def initial_hippos := 35
def female_frac := 5 / 7
def births_per_female_hippo := 5
def total_animals_after_birth := 315

-- Calculate the required values
def female_hippos := female_frac * initial_hippos
def baby_hippos := female_hippos * births_per_female_hippo
def total_animals_before_birth := initial_elephants + initial_hippos
def total_newborns := total_animals_after_birth - total_animals_before_birth
def newborn_elephants := total_newborns - baby_hippos

-- Define the proof statement
theorem more_newborn_elephants_than_baby_hippos :
  (newborn_elephants - baby_hippos) = 10 :=
by
  sorry

end more_newborn_elephants_than_baby_hippos_l148_148004


namespace length_AD_l148_148349

theorem length_AD (x : ℝ) (h1 : 0 ≤ x) (h2 : x ≤ 13) (h3 : x * (13 - x) = 36) : x = 4 ∨ x = 9 :=
by sorry

end length_AD_l148_148349


namespace expression_value_l148_148144

theorem expression_value (x : ℝ) (h : x = Real.sqrt (19 - 8 * Real.sqrt 3)) :
  (x ^ 4 - 6 * x ^ 3 - 2 * x ^ 2 + 18 * x + 23) / (x ^ 2 - 8 * x + 15) = 5 :=
by
  sorry

end expression_value_l148_148144


namespace probability_of_at_most_3_heads_l148_148477

-- Definitions based on conditions
def binom : ℕ → ℕ → ℕ
| 0, 0       => 1
| 0, (k + 1) => 0
| (n + 1), 0 => 1
| (n + 1), (k + 1) => binom n k + binom n (k + 1)

def favorable_outcomes : ℕ :=
binom 10 0 + binom 10 1 + binom 10 2 + binom 10 3

def total_outcomes : ℕ := 2 ^ 10

def probability (favorable : ℕ) (total : ℕ) : ℚ :=
favorable / total

-- Theorem statement
theorem probability_of_at_most_3_heads : 
  probability favorable_outcomes total_outcomes = 11 / 64 :=
sorry

end probability_of_at_most_3_heads_l148_148477


namespace point_on_fixed_circle_l148_148810

theorem point_on_fixed_circle 
  (r s : Set Point) (A : Point) 
  (h1 : Parallel r s) 
  (h2 : Equidistant A r s) 
  (B : Point) 
  (h3 : B ∈ r) 
  (C : Point) 
  (h4 : C ∈ s) 
  (h5 : ∠BAC = 90)
  (P : Point) 
  (h6 : FootPerpendicular A B C P) : 
  ∃ (circleCenter : Point) (radius : ℝ), radius > 0 ∧ 
    (∀ B', B' ∈ r → ∃ C' : Point, C' ∈ s ∧ ∠BAC = 90 → 
    ∃ P' : Point, FootPerpendicular A B' C' P' ∧ 
    dist circleCenter P' = radius) :=
sorry

end point_on_fixed_circle_l148_148810


namespace min_triangle_area_of_ABC_l148_148092

theorem min_triangle_area_of_ABC (a b c : ℝ) (B A C : ℝ) (BC : ℝ := b + c)
  (D : {D // D ∈ segment B C}) (AD : ℝ := sqrt 7) (BAC : ℝ := 60)
  (ratio : BD / DC = 2c / b) : 
  ∀ A B C, min_area (ABC_area) := 2 * sqrt 3 :=
begin
  sorry
end

end min_triangle_area_of_ABC_l148_148092


namespace arithmetic_sequence_groups_count_l148_148993

theorem arithmetic_sequence_groups_count :
  ∃ (groups : list (list ℕ)), 
    (∀ g ∈ groups, g.length = 3 ∧ ∃ d, (∀ i ∈ finset.range (g.length - 1), g[i + 1] - g[i] = d)) ∧
    (list.join groups).erase_dup = [1, 2, 3, 4, 5, 6, 7, 8, 9] ∧ 
    groups.length = 3 ∧ 
    ∃! g : list (list ℕ), g.length = 5 :=
sorry

end arithmetic_sequence_groups_count_l148_148993


namespace range_of_abc_l148_148217

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 ∧ x ≤ 9 then |real.log x / real.log 3 - 1|
  else if x > 9 then 4 - real.sqrt x
  else 0 -- edge case not covered by the problem definition

theorem range_of_abc (a b c : ℝ) (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c)
  (h_a : 0 < a ∧ a ≤ 3)
  (h_b : 3 < b ∧ b < 9)
  (h_c : 9 < c ∧ c < 16)
  (h_f : f(a) = f(b) ∧ f(b) = f(c)) :
  81 < a * b * c ∧ a * b * c < 144 := 
sorry

end range_of_abc_l148_148217


namespace limit_of_cubic_difference_l148_148972

open Real

theorem limit_of_cubic_difference (x : ℝ) : 
  ∀ x, (0 < |x - 1| → ∀ ε > 0, ∃ δ > 0, ∀ (x : ℝ), 0 < |x - 1| ∧ |x - 1| < δ → |(x^3 - 1) / (x - 1) - 3| < ε) :=
  sorry

end limit_of_cubic_difference_l148_148972


namespace relationship_y_l148_148095

theorem relationship_y (y1 y2 y3 : ℝ) (h1 : y1 = -4) (h2 : y2 = -2) (h3 : y3 = 4 / 3) :
  y1 < y2 ∧ y2 < y3 :=
by {
  sorry,
}

end relationship_y_l148_148095


namespace monotonic_increasing_interval_l148_148859

noncomputable def sqrt (x : ℝ) : ℝ := Real.sqrt x

theorem monotonic_increasing_interval :
  ∀ (x : ℝ), 0 ≤ x ∧ x ≤ 1 → ∀ (x1 x2 : ℝ), 0 ≤ x1 ∧ x1 ≤ 1 → 0 ≤ x2 ∧ x2 ≤ 1 → x1 ≤ x2 → sqrt (- x1 ^ 2 + 2 * x1) ≤ sqrt (- x2 ^ 2 + 2 * x2) :=
sorry

end monotonic_increasing_interval_l148_148859


namespace inverse_proportion_k_value_l148_148147

theorem inverse_proportion_k_value (k : ℝ) (h : k ≠ 0) :
  (1, 3) ∈ {p : ℝ × ℝ | p.2 = k / p.1} → k = 3 :=
by
  intro hpoint
  have h_eq : 3 = k / 1 := by rw [mem_set_of_eq] at hpoint; exact hpoint
  rw [div_one] at h_eq
  exact h_eq.symm


end inverse_proportion_k_value_l148_148147


namespace cubic_exponent_eq_zero_l148_148921

theorem cubic_exponent_eq_zero (n : ℕ) (a_n b_n c_n : ℤ) 
  (h : (1 + 4 * real.cbrt 2 - 4 * real.cbrt 4)^n = a_n + b_n * real.cbrt 2 + c_n * real.cbrt 4) :
  c_n = 0 → n = 0 :=
begin
  intro hcn,
  sorry  -- Proof omitted
end

end cubic_exponent_eq_zero_l148_148921


namespace probability_heads_at_most_three_out_of_ten_coins_l148_148546

theorem probability_heads_at_most_three_out_of_ten_coins :
  let total_outcomes := 2^10 in
  let favorable_outcomes := Nat.choose 10 0 + Nat.choose 10 1 + Nat.choose 10 2 + Nat.choose 10 3 in
  favorable_outcomes / total_outcomes = 11 / 64 :=
by
  sorry

end probability_heads_at_most_three_out_of_ten_coins_l148_148546


namespace isosceles_triangle_rotation_l148_148606

variables {A B C A₁ B₁ : Type} [AffineSpace ℝ A] [AffineSpace ℝ B] [AffineSpace ℝ C] [AffineSpace ℝ A₁] [AffineSpace ℝ B₁]
variables (A B C A₁ B₁ : AffinePoint ℝ)
variables (BC : Line ℝ) (h1 : Isosceles A B C) (h2 : RotatedAround A₁ C BC A) (h3 : B move_to B₁ A₁ B₂ side_of BC)
variables (θ : ℝ) 

theorem isosceles_triangle_rotation: 
  Isosceles A B C ∧ 
  RotatedAround A₁ C BC A ∧ 
  SameSide B₁ A BC →
  Paralle AB B₁C :=
by 
  sorry

end isosceles_triangle_rotation_l148_148606


namespace sally_bread_consumption_l148_148837

/-
Given:
    1) Sally eats 2 sandwiches on Saturday
    2) Sally eats 1 sandwich on Sunday
    3) Each sandwich uses 2 pieces of bread

Prove:
    Sally eats 6 pieces of bread across Saturday and Sunday
-/
theorem sally_bread_consumption (sandwiches_sat : Nat) (sandwiches_sun : Nat) (bread_per_sandwich : Nat)
    (H1 : sandwiches_sat = 2) (H2 : sandwiches_sun = 1) (H3 : bread_per_sandwich = 2) :
    2 * bread_per_sandwich + 1 * bread_per_sandwich = 6 := by
  sorry

end sally_bread_consumption_l148_148837


namespace sum_of_digits_greatest_prime_divisor_of_32766_l148_148261

theorem sum_of_digits_greatest_prime_divisor_of_32766 :
  ∃ p : ℕ, prime p ∧ p ∣ 32766 ∧ (∀ q : ℕ, prime q ∧ q ∣ 32766 → q ≤ p) ∧ (p.digits 10).sum = 8 :=
by sorry

end sum_of_digits_greatest_prime_divisor_of_32766_l148_148261


namespace simple_interest_rate_l148_148199

variable (P : ℝ) (A : ℝ) (T : ℝ)

theorem simple_interest_rate (h1 : P = 35) (h2 : A = 36.4) (h3 : T = 1) :
  (A - P) / (P * T) = 0.04 :=
by
  sorry

end simple_interest_rate_l148_148199


namespace arithmetic_sequences_l148_148662

open Nat

theorem arithmetic_sequences
  (a : ℕ → ℕ)
  (h : ∀ n, (∏ i in range n, a i) ∣ (∏ i in range n, a (n + i))): 
  ∃ k, (∀ i, a i = k * (i + 1)) :=
by
  sorry

end arithmetic_sequences_l148_148662


namespace confirm_per_capita_volumes_l148_148345

noncomputable def west_volume_per_capita := 21428
noncomputable def non_west_volume_per_capita := 26848.55
noncomputable def russia_volume_per_capita := 302790.13

theorem confirm_per_capita_volumes :
  west_volume_per_capita = 21428 ∧
  non_west_volume_per_capita = 26848.55 ∧
  russia_volume_per_capita = 302790.13 :=
by
  split; sorry
  split; sorry
  sorry

end confirm_per_capita_volumes_l148_148345


namespace probability_at_most_three_heads_10_coins_l148_148362

open Nat

noncomputable def probability_at_most_three_heads (n : ℕ) (k : ℕ) : ℚ :=
  (∑ i in Finset.range (k + 1), (Nat.choose n i : ℚ)) / (2 ^ n : ℚ)

theorem probability_at_most_three_heads_10_coins :
  probability_at_most_three_heads 10 3 = 11 / 64 :=
by
  sorry

end probability_at_most_three_heads_10_coins_l148_148362


namespace exists_consecutive_composite_l148_148828

theorem exists_consecutive_composite (n : ℕ) : 
  ∃ (a : ℕ → ℕ), 
    (∀ i : ℕ, 2 ≤ i ∧ i ≤ n + 1 → a i = (n+1)! + i ∧ ∀ j : ℕ, 2 ≤ j → j ∣ a i ∧ j ∤ 1 ∧ j ∤ a i → a i ≠ j ∧ a i ≠ ((n+1)! + i) ∧ a i > 1) := 
sorry

end exists_consecutive_composite_l148_148828


namespace g_eq_one_l148_148803

noncomputable def g (a b c : ℝ) : ℝ := 
  a / (a + b + c) + b / (b + c + a) + c / (c + a + b)

theorem g_eq_one (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  g(a, b, c) = 1 :=
by
  sorry

end g_eq_one_l148_148803


namespace inequality_abc_l148_148712

theorem inequality_abc (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a * b + b * c + c * a = 1) :
  (∑ i in [a, b, c], real.cbrt ((6 * i.elt) + 1 / i.elt)) ≤ 1 / (a * b * c) := sorry

end inequality_abc_l148_148712


namespace parallel_DM_BL_l148_148243

variables (A B C D E F K M L : Type) [IncidenceStructure A B C D E F K M L] 
variables (angle : A → A → A → ℝ)

-- Define all necessary conditions from the problem statement
variables (cyclic_ABCD : Cyclic A B C D)
variables (CD_eq_DA : distance C D = distance D A)
variables (E_on_AB : OnSegment E A B)
variables (F_on_BC : OnSegment F B C)
variables (angle_ADC_eq_2angle_EDF : angle A D C = 2 * angle E D F)
variables (DK_height : IsHeight D K)
variables (DM_median : IsMedian D M)
variables (L_symmetric_K_M : IsSymmetric L K M)

-- The theorem that we need to prove
theorem parallel_DM_BL : Parallel (LineThrough D M) (LineThrough B L) :=
sorry

end parallel_DM_BL_l148_148243


namespace probability_heads_at_most_3_l148_148541

theorem probability_heads_at_most_3 (num_coins : ℕ) (num_heads_at_most : ℕ) (probability : ℚ):
  num_coins = 10 → 
  num_heads_at_most = 3 → 
  probability = (binomial 10 0 + binomial 10 1 + binomial 10 2 + binomial 10 3) / (2^10) →
  probability = (11 / 64) := 
by
  intros hnum_coins hnum_heads_at_most hprobability
  sorry

end probability_heads_at_most_3_l148_148541


namespace probability_at_most_3_heads_10_coins_l148_148420

noncomputable def coin_flip_prob_at_most_3_heads : ℚ :=
  let total_outcomes := 2^10
  let favorable_outcomes := Nat.binomial 10 0 + Nat.binomial 10 1 + Nat.binomial 10 2 + Nat.binomial 10 3
  favorable_outcomes / total_outcomes

theorem probability_at_most_3_heads_10_coins : coin_flip_prob_at_most_3_heads = 11 / 64 :=
by
  -- Definitions of let values are integrated directly in the theorem body
  let total_outcomes := 2^10
  let favorable_outcomes := Nat.binomial 10 0 + Nat.binomial 10 1 + Nat.binomial 10 2 + Nat.binomial 10 3
  show favorable_outcomes / total_outcomes = 11 / 64
  sorry

end probability_at_most_3_heads_10_coins_l148_148420


namespace inverse_of_f_128_l148_148692

noncomputable def f : ℕ → ℕ :=
  sorry -- The definition of f is noncomputable based on the conditions provided

axiom f_at_4 : f 4 = 2
axiom f_doubling : ∀ x, f (2 * x) = 2 * f x

theorem inverse_of_f_128 : f⁻¹ 128 = 256 :=
by
  sorry

end inverse_of_f_128_l148_148692


namespace find_odd_number_l148_148065
open Nat

def is_odd (n : ℕ) : Prop := n % 2 = 1
def is_even (n : ℕ) : Prop := n % 2 = 0
def f (n : ℕ) : ℕ :=
  if is_odd n then 3 * n else (1 / 2 : ℚ) * n

theorem find_odd_number (n : ℕ) (h_odd : is_odd n) : f n * f 10 = 45 → n = 3 :=
by
  have h10 : f 10 = 5 := sorry
  sorry

end find_odd_number_l148_148065


namespace probability_10_coins_at_most_3_heads_l148_148389

def probability_at_most_3_heads (n : ℕ) : ℚ :=
  let total_outcomes := 2^n
  let favorable_outcomes := (Nat.choose n 0) + (Nat.choose n 1) + (Nat.choose n 2) + (Nat.choose n 3)
  favorable_outcomes / total_outcomes

theorem probability_10_coins_at_most_3_heads : probability_at_most_3_heads 10 = 11 / 64 :=
by
  sorry

end probability_10_coins_at_most_3_heads_l148_148389


namespace sally_bread_consumption_l148_148836

/-
Given:
    1) Sally eats 2 sandwiches on Saturday
    2) Sally eats 1 sandwich on Sunday
    3) Each sandwich uses 2 pieces of bread

Prove:
    Sally eats 6 pieces of bread across Saturday and Sunday
-/
theorem sally_bread_consumption (sandwiches_sat : Nat) (sandwiches_sun : Nat) (bread_per_sandwich : Nat)
    (H1 : sandwiches_sat = 2) (H2 : sandwiches_sun = 1) (H3 : bread_per_sandwich = 2) :
    2 * bread_per_sandwich + 1 * bread_per_sandwich = 6 := by
  sorry

end sally_bread_consumption_l148_148836


namespace min_value_expression_l148_148892

theorem min_value_expression (x : ℝ) : 
  ∃ y : ℝ, (y = (x+2)*(x+3)*(x+4)*(x+5) + 3033) ∧ y ≥ 3032 ∧ 
  (∀ z : ℝ, (z = (x+2)*(x+3)*(x+4)*(x+5) + 3033) → z ≥ 3032) := 
sorry

end min_value_expression_l148_148892


namespace sum_of_solutions_equation_l148_148269

open Real

theorem sum_of_solutions_equation :
  let eq := (fun x => (3 * x + 5) / (4 * x + 6) = (6 * x + 4) / (10 * x + 8))
  in (Σ ξ, eq ξ) = -11 / 3 :=
by
  sorry

end sum_of_solutions_equation_l148_148269


namespace find_c_l148_148669

theorem find_c (x y c : ℝ) (h : x = 5 * y) (h2 : 7 * x + 4 * y = 13 * c) : c = 3 * y :=
by
  sorry

end find_c_l148_148669


namespace right_triangle_cos_B_l148_148765

theorem right_triangle_cos_B (A B C : ℝ) (hC : C = 90) (hSinA : Real.sin A = 2 / 3) :
  Real.cos B = 2 / 3 :=
sorry

end right_triangle_cos_B_l148_148765


namespace like_terms_exponents_l148_148138

theorem like_terms_exponents (n m : ℕ) (h1 : n + 2 = 3) (h2 : 2 * m - 1 = 3) : n = 1 ∧ m = 2 :=
by sorry

end like_terms_exponents_l148_148138


namespace probability_heads_at_most_three_out_of_ten_coins_l148_148556

theorem probability_heads_at_most_three_out_of_ten_coins :
  let total_outcomes := 2^10 in
  let favorable_outcomes := Nat.choose 10 0 + Nat.choose 10 1 + Nat.choose 10 2 + Nat.choose 10 3 in
  favorable_outcomes / total_outcomes = 11 / 64 :=
by
  sorry

end probability_heads_at_most_three_out_of_ten_coins_l148_148556


namespace ap_square_sequel_l148_148182

theorem ap_square_sequel {a b c : ℝ} (h1 : a ≠ b ∧ b ≠ c ∧ a ≠ c)
                     (h2 : 2 * (b / (c + a)) = (a / (b + c)) + (c / (a + b))) :
  (a^2 + c^2 = 2 * b^2) :=
by
  sorry

end ap_square_sequel_l148_148182


namespace probability_at_most_3_heads_l148_148500

-- Definitions and conditions
def coin_flips : ℕ := 10
def success_prob : ℚ := 1/2
def max_heads : ℕ := 3

-- Binomial Probability Computation
noncomputable def probability_of_at_most_3_heads : ℚ :=
  (∑ k in finset.range (max_heads + 1), nat.choose coin_flips k) / (2^coin_flips)

-- The statement to prove
theorem probability_at_most_3_heads :
  probability_of_at_most_3_heads = 11/64 := by
  sorry

end probability_at_most_3_heads_l148_148500


namespace tangent_segment_ratio_l148_148943

-- Given circles k1 and k2 with radii r1 and r2
variables {k1 k2 : Type*} [metric_space k1] [metric_space k2] 
variables {r1 r2 : ℝ}

-- Chords AB in k1 and CD in k2 with equal lengths
variables {A B C D : k1} {x : ℝ}
variables (P : k1 → k2) (h_tangent_1 : ∀ P, tangent_to_circle P A k1)
variables (h_tangent_2 : ∀ P, tangent_to_circle P D k2)
variables (h_chord_1 : chord_length A B = x)
variables (h_chord_2 : chord_length C D = x)

-- Prove PA/PD = r1/r2
theorem tangent_segment_ratio (PA : ℝ) (PD : ℝ) :
  PA / PD = r1 / r2 :=
begin
  sorry
end

end tangent_segment_ratio_l148_148943


namespace problem1_problem2_l148_148126

noncomputable def vec_a (x : ℝ) : ℝ × ℝ := (Real.sin x, 0)
noncomputable def vec_b (x : ℝ) : ℝ × ℝ := (Real.cos x, Real.sin x)

noncomputable def f (x : ℝ) : ℝ :=
  let a := vec_a x
  let b := vec_b x
  (a.1 * b.1 + a.2 * b.2) + (a.1 * a.1 + a.2 * a.2)

theorem problem1 (k : ℤ) (x : ℝ) :
  x ∈ Set.Icc (k * Real.pi + 3 * Real.pi / 8) (k * Real.pi + 7 * Real.pi / 8) →
  Monotone.decreasingOn f fun y => y ∈ Set.Icc (k * Real.pi + 3 * Real.pi / 8) (k * Real.pi + 7 * Real.pi / 8) :=
sorry

theorem problem2 (A B : ℝ) (ABC : {A, B, C : ℝ // A + B + C = Real.pi}) :
  f (A / 2) = 1 →
  0 < f B ∧ f B ≤ (Real.sqrt 2 + 1) / 2 :=
sorry

end problem1_problem2_l148_148126


namespace repeating_decimal_sum_l148_148050

noncomputable def repeating_decimal_1 : ℚ := 137 / 999
noncomputable def repeating_decimal_2 : ℚ := 24 / 999

theorem repeating_decimal_sum : 
  0.\overline{137} + 0.\overline{024} = 161 / 999 := by sorry

end repeating_decimal_sum_l148_148050


namespace proof1_proof2a_proof2b_l148_148782

variable (A B C : ℝ)
variable (a b c : ℝ)
variable (S : ℝ)

-- Given conditions for Question 1
def condition1 := (a = 3 * Real.cos C ∧ b = 1)

-- Proof statement for Question 1
theorem proof1 : condition1 a b C → Real.tan C = 2 * Real.tan B :=
by sorry

-- Given conditions for Question 2a
def condition2a := (S = 1 / 2 * a * b * Real.sin C ∧ S = 1 / 2 * 3 * Real.cos C * 1 * Real.sin C)

-- Proof statement for Question 2a
theorem proof2a : condition2a a b C S → Real.cos (2 * B) = 3 / 5 :=
by sorry

-- Given conditions for Question 2b
def condition2b := (c = Real.sqrt 10 / 2)

-- Proof statement for Question 2b
theorem proof2b : condition1 a b C → condition2b c → Real.cos (2 * B) = 3 / 5 :=
by sorry

end proof1_proof2a_proof2b_l148_148782


namespace remainder_of_13_plus_x_mod_29_l148_148218

theorem remainder_of_13_plus_x_mod_29
  (x : ℕ)
  (hx : 8 * x ≡ 1 [MOD 29])
  (hp : 0 < x) : 
  (13 + x) % 29 = 18 :=
sorry

end remainder_of_13_plus_x_mod_29_l148_148218


namespace like_terms_exponents_l148_148137

theorem like_terms_exponents (n m : ℕ) (h1 : n + 2 = 3) (h2 : 2 * m - 1 = 3) : n = 1 ∧ m = 2 :=
by sorry

end like_terms_exponents_l148_148137


namespace empty_square_in_grid_of_15_points_l148_148159

theorem empty_square_in_grid_of_15_points : 
  ∀ (grid : fin 4 × fin 4 → Prop), (∃ (points : fin 15 → (fin 4 × fin 4)), ∀ (p : fin 4 × fin 4), p ∉ set.range points) -> ∃ (p : fin 4 × fin 4), p ∉ set.range points :=
by
  sorry

end empty_square_in_grid_of_15_points_l148_148159


namespace appropriate_sampling_method_l148_148880

-- Definitions and conditions
def total_products : ℕ := 40
def first_class_products : ℕ := 10
def second_class_products : ℕ := 25
def defective_products : ℕ := 5
def samples_needed : ℕ := 8

-- Theorem statement
theorem appropriate_sampling_method : 
  (first_class_products + second_class_products + defective_products = total_products) ∧ 
  (2 ≤ first_class_products ∧ 2 ≤ second_class_products ∧ 1 ≤ defective_products) → 
  "Stratified Sampling" = "The appropriate sampling method for quality analysis" :=
  sorry

end appropriate_sampling_method_l148_148880


namespace cosine_vertex_angle_l148_148713

theorem cosine_vertex_angle (a : ℝ) (h : cos a = 1 / 3) :
  cos (π - 2 * a) = 7 / 9 :=
sorry

end cosine_vertex_angle_l148_148713


namespace josh_total_money_l148_148794

-- Define the initial conditions
def initial_wallet : ℝ := 300
def initial_investment : ℝ := 2000
def stock_increase_rate : ℝ := 0.30

-- The expected total amount Josh will have after selling his stocks
def expected_total_amount : ℝ := 2900

-- Define the problem: that the total money in Josh's wallet after selling all stocks equals $2900
theorem josh_total_money :
  let increased_value := initial_investment * stock_increase_rate
  let new_investment := initial_investment + increased_value
  let total_money := new_investment + initial_wallet
  total_money = expected_total_amount :=
by
  sorry

end josh_total_money_l148_148794


namespace dragon_chain_length_l148_148575

theorem dragon_chain_length (a b c : ℕ) (h_prime : Prime c) (h_chain_length : 30 = a - √b) 
(h_castle_radius : 10 = radius) (h_dragon_height : 6 = height) (h_dragon_distance : 6 = distance) :
  a + b + c = 1533 :=
by
  -- Definitions from conditions:
  let a := 90
  let b := 1440
  let c := 3
  have h_prime : Prime c := Prime.intro rfl
  have h_chain_length : 30 = 90 - √1440 := rfl
  
  -- Assertion of the final equation
  sorry

end dragon_chain_length_l148_148575


namespace trig_identity_75_30_15_150_l148_148646

theorem trig_identity_75_30_15_150 :
  (Real.sin (75 * Real.pi / 180) * Real.cos (30 * Real.pi / 180) - 
   Real.sin (15 * Real.pi / 180) * Real.sin (150 * Real.pi / 180)) = 
  (Real.sqrt 2 / 2) :=
by
  -- Proof goes here
  sorry

end trig_identity_75_30_15_150_l148_148646


namespace probability_of_at_most_3_heads_l148_148463

-- Definitions based on conditions
def binom : ℕ → ℕ → ℕ
| 0, 0       => 1
| 0, (k + 1) => 0
| (n + 1), 0 => 1
| (n + 1), (k + 1) => binom n k + binom n (k + 1)

def favorable_outcomes : ℕ :=
binom 10 0 + binom 10 1 + binom 10 2 + binom 10 3

def total_outcomes : ℕ := 2 ^ 10

def probability (favorable : ℕ) (total : ℕ) : ℚ :=
favorable / total

-- Theorem statement
theorem probability_of_at_most_3_heads : 
  probability favorable_outcomes total_outcomes = 11 / 64 :=
sorry

end probability_of_at_most_3_heads_l148_148463


namespace probability_10_coins_at_most_3_heads_l148_148392

def probability_at_most_3_heads (n : ℕ) : ℚ :=
  let total_outcomes := 2^n
  let favorable_outcomes := (Nat.choose n 0) + (Nat.choose n 1) + (Nat.choose n 2) + (Nat.choose n 3)
  favorable_outcomes / total_outcomes

theorem probability_10_coins_at_most_3_heads : probability_at_most_3_heads 10 = 11 / 64 :=
by
  sorry

end probability_10_coins_at_most_3_heads_l148_148392


namespace arithmetic_seq_div_l148_148661

open Nat

theorem arithmetic_seq_div (a d : ℕ) (h : ∀ n : ℕ, 
  ∏ i in range n, (a + i*d) ∣ ∏ i in range n, (a + (i+n)*d)) :
  ∃ k : ℕ, ∀ n : ℕ, a + n*d = k*(n+1) := sorry

end arithmetic_seq_div_l148_148661


namespace probability_at_most_three_heads_10_coins_l148_148351

open Nat

noncomputable def probability_at_most_three_heads (n : ℕ) (k : ℕ) : ℚ :=
  (∑ i in Finset.range (k + 1), (Nat.choose n i : ℚ)) / (2 ^ n : ℚ)

theorem probability_at_most_three_heads_10_coins :
  probability_at_most_three_heads 10 3 = 11 / 64 :=
by
  sorry

end probability_at_most_three_heads_10_coins_l148_148351


namespace sum_a_k_proof_l148_148733

theorem sum_a_k_proof :
  let a : ℕ → ℝ := λ n, if n = 1 then 1 else (λ a (n : ℕ), (n * a) / (2 * (n - 1) + a)) (a (n - 1)) n
  in
  ((∑ k in Finset.range (2021 + 1).tail, (k / (a k))) = (2 : ℕ)^(2022) - 2023) := 
by
  sorry

end sum_a_k_proof_l148_148733


namespace abs_diff_m_n_l148_148805

variable (m n : ℝ)

theorem abs_diff_m_n (h1 : m * n = 6) (h2 : m + n = 7) (h3 : m^2 - n^2 = 13) : |m - n| = 13 / 7 :=
by
  sorry

end abs_diff_m_n_l148_148805


namespace probability_at_most_3_heads_10_coins_l148_148415

noncomputable def coin_flip_prob_at_most_3_heads : ℚ :=
  let total_outcomes := 2^10
  let favorable_outcomes := Nat.binomial 10 0 + Nat.binomial 10 1 + Nat.binomial 10 2 + Nat.binomial 10 3
  favorable_outcomes / total_outcomes

theorem probability_at_most_3_heads_10_coins : coin_flip_prob_at_most_3_heads = 11 / 64 :=
by
  -- Definitions of let values are integrated directly in the theorem body
  let total_outcomes := 2^10
  let favorable_outcomes := Nat.binomial 10 0 + Nat.binomial 10 1 + Nat.binomial 10 2 + Nat.binomial 10 3
  show favorable_outcomes / total_outcomes = 11 / 64
  sorry

end probability_at_most_3_heads_10_coins_l148_148415


namespace probability_of_at_most_3_heads_out_of_10_l148_148491
open_locale big_operators
open nat

theorem probability_of_at_most_3_heads_out_of_10 (n : ℕ) (p : ℕ) (h : 0 < n) (hp : p = 2) :
  ∑ k in finset.range (4), nat.choose n k = 176 → nat.pow p n = 1024 → (∑ k in finset.range (4), nat.choose n k) / (nat.pow p n) = 11 / 64 :=
by { intro h₁ h₂, 
     have : (∑ k in finset.range 4, nat.choose n k) / (nat.pow p n) = 176 / 1024,
     { rw [h₁, h₂] },
     rw this,
     norm_num }

end probability_of_at_most_3_heads_out_of_10_l148_148491


namespace arrangement_ways_l148_148007

-- Defining the conditions
def num_basil_plants : Nat := 5
def num_tomato_plants : Nat := 4
def num_total_units : Nat := num_basil_plants + 1

-- Proof statement
theorem arrangement_ways : (num_total_units.factorial) * (num_tomato_plants.factorial) = 17280 := by
  sorry

end arrangement_ways_l148_148007


namespace compare_exponent_inequality_l148_148084

theorem compare_exponent_inequality (a x y : ℝ) (h1 : 0 < a) (h2 : a < 1) (h3 : a^x < a^y) : x^3 > y^3 :=
sorry

end compare_exponent_inequality_l148_148084


namespace find_m_n_l148_148136

-- Given definition of like terms
def like_terms (a b : ℝ) (n1 n2 m1 m2 : ℤ) : Prop :=
  n1 = n2 ∧ m1 = m2

-- Variables
variables {x y : ℝ} {m n : ℤ}

-- Definitions based on problem conditions
def expr1 : ℝ := 2 * x^(n + 2) * y^3
def expr2 : ℝ := -3 * x^3 * y^(2 * m - 1)

-- Proof problem
theorem find_m_n (h : like_terms expr1 expr2 (n + 2) 3 3 (2 * m - 1)) : m = 2 ∧ n = 1 :=
  sorry

end find_m_n_l148_148136


namespace median_perpendicular_to_angle_bisector_l148_148178

noncomputable def triangle_ABC (A B C : Point) : Prop :=
  let AB := dist A B
  let AC := dist A C
  let BC := dist B C
  AB = AC ∧ AB = 2 * BC

theorem median_perpendicular_to_angle_bisector (A B C : Point) (h : triangle_ABC A B C) :
  let B₁ := midpoint A C
  let median := Line B B₁
  let bisector := angleBisector C A B
  isPerpendicular median bisector :=
sorry

end median_perpendicular_to_angle_bisector_l148_148178


namespace probability_at_most_3_heads_10_flips_l148_148519

theorem probability_at_most_3_heads_10_flips : 
  let num_heads_0 := nat.choose 10 0,
      num_heads_1 := nat.choose 10 1,
      num_heads_2 := nat.choose 10 2,
      num_heads_3 := nat.choose 10 3,
      total_outcomes := 2^10 in
  (num_heads_0 + num_heads_1 + num_heads_2 + num_heads_3) / total_outcomes = 11 / 64 :=
by
  sorry

end probability_at_most_3_heads_10_flips_l148_148519


namespace coefficient_in_expansion_l148_148172

noncomputable def coefficient_of_x2 : ℕ :=
  let general_term (r : ℕ) := (-2 : ℤ)^r * nat.choose 6 r * (x : ℤ)^(6 - 2 * r)
  if 6 - 2 * 2 = 2 then
    (-2 : ℤ)^2 * nat.choose 6 2
  else
    0

theorem coefficient_in_expansion : coefficient_of_x2 = 60 :=
by
  sorry

end coefficient_in_expansion_l148_148172


namespace probability_10_coins_at_most_3_heads_l148_148388

def probability_at_most_3_heads (n : ℕ) : ℚ :=
  let total_outcomes := 2^n
  let favorable_outcomes := (Nat.choose n 0) + (Nat.choose n 1) + (Nat.choose n 2) + (Nat.choose n 3)
  favorable_outcomes / total_outcomes

theorem probability_10_coins_at_most_3_heads : probability_at_most_3_heads 10 = 11 / 64 :=
by
  sorry

end probability_10_coins_at_most_3_heads_l148_148388


namespace change_in_y_is_zero_l148_148117

noncomputable def y (x : ℝ) : ℝ := sin (x) ^ 2 - 3

theorem change_in_y_is_zero (x : ℝ) : 
  y (x + π) = y x ∧ y (x - π) = y x :=
by
  sorry

end change_in_y_is_zero_l148_148117


namespace probability_at_most_three_heads_10_coins_l148_148365

open Nat

noncomputable def probability_at_most_three_heads (n : ℕ) (k : ℕ) : ℚ :=
  (∑ i in Finset.range (k + 1), (Nat.choose n i : ℚ)) / (2 ^ n : ℚ)

theorem probability_at_most_three_heads_10_coins :
  probability_at_most_three_heads 10 3 = 11 / 64 :=
by
  sorry

end probability_at_most_three_heads_10_coins_l148_148365


namespace speed_of_car_A_l148_148976

variable (V_A V_B T : ℕ)
variable (h1 : V_B = 35) (h2 : T = 10) (h3 : 2 * V_B * T = V_A * T)

theorem speed_of_car_A :
  V_A = 70 :=
by
  sorry

end speed_of_car_A_l148_148976


namespace epidemic_duration_l148_148602

theorem epidemic_duration
  (num_dwarfs : ℕ)
  (immune_first_day: ℕ → Prop)
  (sick_next_day: ℕ → ℕ → Prop) 
  (healthy: ℕ → Prop):
  (∃ d, immune_first_day d) → (∃ F, ∀ t, sick_next_day t (F t)) ∨ ¬(∃ d, immune_first_day d) → (∃ t, ¬ ∃ d, sick_next_day t d) :=
by
  sorry

end epidemic_duration_l148_148602


namespace probability_heads_at_most_3_of_10_coins_flipped_l148_148368

theorem probability_heads_at_most_3_of_10_coins_flipped :
  (∑ k in finset.range 4, nat.choose 10 k) / 2^10 = 11 / 64 :=
sorry

end probability_heads_at_most_3_of_10_coins_flipped_l148_148368


namespace circle_center_eq_l148_148932

theorem circle_center_eq :
  ∃ (a b : ℝ), ∀ (x y : ℝ),
    (0, 1) ∈ (a, b) → (2, 4) ∈ (a, b) ∧ tangent (y = x^2) (2, 4) ∧
    let center := (a, b) in
    center = (-16/5, 53/10) :=
begin
  sorry
end

end circle_center_eq_l148_148932


namespace total_caffeine_is_correct_l148_148194

def first_drink_caffeine := 250 -- milligrams
def first_drink_size := 12 -- ounces

def second_drink_caffeine_per_ounce := (first_drink_caffeine / first_drink_size) * 3
def second_drink_size := 8 -- ounces
def second_drink_caffeine := second_drink_caffeine_per_ounce * second_drink_size

def third_drink_concentration := 18 -- milligrams per milliliter
def third_drink_size := 150 -- milliliters
def third_drink_caffeine := third_drink_concentration * third_drink_size

def caffeine_pill_caffeine := first_drink_caffeine + second_drink_caffeine + third_drink_caffeine

def total_caffeine_consumed := first_drink_caffeine + second_drink_caffeine + third_drink_caffeine + caffeine_pill_caffeine

theorem total_caffeine_is_correct : total_caffeine_consumed = 6900 :=
by
  sorry

end total_caffeine_is_correct_l148_148194


namespace total_spots_l148_148807

-- Define the variables
variables (R C G S B : ℕ)

-- State the problem conditions
def conditions : Prop :=
  R = 46 ∧
  C = R / 2 - 5 ∧
  G = 5 * C ∧
  S = 3 * R ∧
  B = 2 * (G + S)

-- State the proof problem
theorem total_spots : conditions R C G S B → G + C + S + B = 702 :=
by
  intro h
  obtain ⟨hR, hC, hG, hS, hB⟩ := h
  -- The proof steps would go here
  sorry

end total_spots_l148_148807


namespace magnitude_a_eq_3sqrt10_l148_148742

noncomputable def vector_a (t : ℝ) : ℝ × ℝ := (t - 2, 3)
def vector_b : ℝ × ℝ := (3, -1)

theorem magnitude_a_eq_3sqrt10 (t : ℝ) 
  (h : ∃ k : ℝ, ∀ (x y : ℝ), x = t + 4 → y = 1 → (x, y) = k • (3, -1)) :
  |(t - 2, 3)| = 3 * real.sqrt 10 :=
by sorry

end magnitude_a_eq_3sqrt10_l148_148742


namespace frog_escape_prob_l148_148760

noncomputable def P : ℕ → ℚ
| 0 => 0
| 12 => 1
| (N + 1) => 
  if (N + 1 < 12) then 
    (N + 2) / 13 * P (N) + (1 - (N + 2) / 13) * P (N + 2)
  else 
    sorry
  
theorem frog_escape_prob : P 2 = -- correct answer here as computed in the solution
sorry

end frog_escape_prob_l148_148760


namespace find_angle_QPR_l148_148171

-- Define the angles and line segment
variables (R S Q T P : Type) 
variables (line_RT : R ≠ S)
variables (x : ℝ) 
variables (angle_PTQ : ℝ := 62)
variables (angle_RPS : ℝ := 34)

-- Hypothesis that PQ = PT, making triangle PQT isosceles
axiom eq_PQ_PT : ℝ

-- Conditions
axiom lie_on_RT : ∀ {R S Q T : Type}, R ≠ S 
axiom angle_PTQ_eq : angle_PTQ = 62
axiom angle_RPS_eq : angle_RPS = 34

-- Hypothesis that defines the problem structure
theorem find_angle_QPR : x = 11 := by
sorry

end find_angle_QPR_l148_148171


namespace min_one_over_a_plus_one_over_b_l148_148121

theorem min_one_over_a_plus_one_over_b (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h_line : 2*a - b*2 + 2 = 0) (h_center : (x + 1)^2 + (y - 2)^2 = 4) :
  min (1/a + 1/b) = 4 :=
by
  -- Conditions translated from the problem statement
  have h_center_def : (-1,2) = ⟨-1,2⟩, from sorry,
  have h1 : a + b = 1, from sorry,
  
  -- Intermediate steps which would be proved later
  have h2 : 2 + b/a + a/b ≥ 2 + 2, from sorry,
  
  -- Final proof
  show min (1/a + 1/b) = 4, from sorry

end min_one_over_a_plus_one_over_b_l148_148121


namespace brother_more_cars_l148_148886

constant Tommy_cars : Nat
constant Jessie_cars : Nat
constant Total_cars : Nat

axiom Tommy_has_3_cars : Tommy_cars = 3
axiom Jessie_has_3_cars : Jessie_cars = 3
axiom Total_cars_is_17 : Total_cars = 17

def Brother_cars : Nat := Total_cars - (Tommy_cars + Jessie_cars)
def More_than_Tommy_Jessie : Nat := Brother_cars - (Tommy_cars + Jessie_cars)

theorem brother_more_cars : More_than_Tommy_Jessie = 5 :=
  by
  sorry

end brother_more_cars_l148_148886


namespace probability_10_coins_at_most_3_heads_l148_148386

def probability_at_most_3_heads (n : ℕ) : ℚ :=
  let total_outcomes := 2^n
  let favorable_outcomes := (Nat.choose n 0) + (Nat.choose n 1) + (Nat.choose n 2) + (Nat.choose n 3)
  favorable_outcomes / total_outcomes

theorem probability_10_coins_at_most_3_heads : probability_at_most_3_heads 10 = 11 / 64 :=
by
  sorry

end probability_10_coins_at_most_3_heads_l148_148386


namespace probability_at_most_3_heads_l148_148503

-- Definitions and conditions
def coin_flips : ℕ := 10
def success_prob : ℚ := 1/2
def max_heads : ℕ := 3

-- Binomial Probability Computation
noncomputable def probability_of_at_most_3_heads : ℚ :=
  (∑ k in finset.range (max_heads + 1), nat.choose coin_flips k) / (2^coin_flips)

-- The statement to prove
theorem probability_at_most_3_heads :
  probability_of_at_most_3_heads = 11/64 := by
  sorry

end probability_at_most_3_heads_l148_148503


namespace probability_heads_at_most_3_of_10_coins_flipped_l148_148369

theorem probability_heads_at_most_3_of_10_coins_flipped :
  (∑ k in finset.range 4, nat.choose 10 k) / 2^10 = 11 / 64 :=
sorry

end probability_heads_at_most_3_of_10_coins_flipped_l148_148369


namespace max_participants_won_at_least_three_matches_l148_148562

theorem max_participants_won_at_least_three_matches :
  ∀ (n : ℕ), n = 200 → ∃ k : ℕ, k ≤ 66 ∧ ∀ p : ℕ, (p ≥ k ∧ p > 66) → false := by
  sorry

end max_participants_won_at_least_three_matches_l148_148562


namespace problem_eq_995_l148_148979

theorem problem_eq_995 :
  (12^4 + 400) * (24^4 + 400) * (36^4 + 400) * (48^4 + 400) * (60^4 + 400) /
  ((6^4 + 400) * (18^4 + 400) * (30^4 + 400) * (42^4 + 400) * (54^4 + 400))
  = 995 := sorry

end problem_eq_995_l148_148979


namespace probability_not_buy_l148_148865

-- Define the given probability of Sam buying a new book
def P_buy : ℚ := 5 / 8

-- Theorem statement: The probability that Sam will not buy a new book is 3 / 8
theorem probability_not_buy : 1 - P_buy = 3 / 8 :=
by
  -- Proof omitted
  sorry

end probability_not_buy_l148_148865


namespace probability_10_coins_at_most_3_heads_l148_148395

def probability_at_most_3_heads (n : ℕ) : ℚ :=
  let total_outcomes := 2^n
  let favorable_outcomes := (Nat.choose n 0) + (Nat.choose n 1) + (Nat.choose n 2) + (Nat.choose n 3)
  favorable_outcomes / total_outcomes

theorem probability_10_coins_at_most_3_heads : probability_at_most_3_heads 10 = 11 / 64 :=
by
  sorry

end probability_10_coins_at_most_3_heads_l148_148395


namespace find_strawberry_jelly_amount_l148_148840

noncomputable def strawberry_jelly (t b : ℕ) : ℕ := t - b

theorem find_strawberry_jelly_amount (h₁ : 6310 = 4518 + s) : s = 1792 := by
  sorry

end find_strawberry_jelly_amount_l148_148840


namespace axis_of_symmetry_parabola_eq_l148_148667

theorem axis_of_symmetry_parabola_eq : ∀ (x y p : ℝ), 
  y = -2 * x^2 → 
  (x^2 = -2 * p * y) → 
  (p = 1/4) →  
  (y = p / 2) → 
  y = 1 / 8 := by 
  intros x y p h1 h2 h3 h4
  sorry

end axis_of_symmetry_parabola_eq_l148_148667


namespace probability_of_at_most_3_heads_l148_148405

-- Definitions and conditions
def num_coins : ℕ := 10
def at_most_3_heads_probability : ℚ := 11 / 64

-- Statement of the problem
theorem probability_of_at_most_3_heads (n : ℕ) (p : ℚ) (h1 : n = num_coins) (h2 : p = at_most_3_heads_probability) :
  p = (1 + 10 + 45 + 120 : ℕ) / (2 ^ 10 : ℕ) := by
  sorry

end probability_of_at_most_3_heads_l148_148405


namespace value_of_sum_of_squares_l148_148134

theorem value_of_sum_of_squares (x y : ℝ) (h₁ : (x + y)^2 = 25) (h₂ : x * y = -6) : x^2 + y^2 = 37 :=
by
  sorry

end value_of_sum_of_squares_l148_148134


namespace probability_heads_at_most_3_of_10_coins_flipped_l148_148380

theorem probability_heads_at_most_3_of_10_coins_flipped :
  (∑ k in finset.range 4, nat.choose 10 k) / 2^10 = 11 / 64 :=
sorry

end probability_heads_at_most_3_of_10_coins_flipped_l148_148380


namespace probability_at_most_3_heads_10_flips_l148_148523

theorem probability_at_most_3_heads_10_flips : 
  let num_heads_0 := nat.choose 10 0,
      num_heads_1 := nat.choose 10 1,
      num_heads_2 := nat.choose 10 2,
      num_heads_3 := nat.choose 10 3,
      total_outcomes := 2^10 in
  (num_heads_0 + num_heads_1 + num_heads_2 + num_heads_3) / total_outcomes = 11 / 64 :=
by
  sorry

end probability_at_most_3_heads_10_flips_l148_148523


namespace remainder_abc_l148_148142

theorem remainder_abc (a b c : ℕ) 
  (h₀ : a < 9) (h₁ : b < 9) (h₂ : c < 9)
  (h₃ : (a + 3 * b + 2 * c) % 9 = 0)
  (h₄ : (2 * a + 2 * b + 3 * c) % 9 = 3)
  (h₅ : (3 * a + b + 2 * c) % 9 = 6) : 
  (a * b * c) % 9 = 0 := by
  sorry

end remainder_abc_l148_148142


namespace base7_to_decimal_146_l148_148639

theorem base7_to_decimal_146 : 
  146_7 = 83 := by
  -- Break down the base 7 number to its components:
  let a := 1 * 7^2 + 4 * 7^1 + 6 * 7^0
  
  -- Verify that it equals 83 in base 10:
  have h : a = 83
  sorry

  -- Finalize the theorem statement:
  exact h

end base7_to_decimal_146_l148_148639


namespace probability_of_at_most_3_heads_l148_148465

-- Definitions based on conditions
def binom : ℕ → ℕ → ℕ
| 0, 0       => 1
| 0, (k + 1) => 0
| (n + 1), 0 => 1
| (n + 1), (k + 1) => binom n k + binom n (k + 1)

def favorable_outcomes : ℕ :=
binom 10 0 + binom 10 1 + binom 10 2 + binom 10 3

def total_outcomes : ℕ := 2 ^ 10

def probability (favorable : ℕ) (total : ℕ) : ℚ :=
favorable / total

-- Theorem statement
theorem probability_of_at_most_3_heads : 
  probability favorable_outcomes total_outcomes = 11 / 64 :=
sorry

end probability_of_at_most_3_heads_l148_148465


namespace ABFK_can_be_inscribed_in_circle_l148_148884

open EuclideanGeometry

variables {A B C D K L F : Point}
variables {rect : Rectangle A B C D}
variables (h₁ : ∃ (p q : Line), Perpendicular p q ∧ on p B ∧ on q B)
variables (h₂ : ∃ K, on K AD)
variables (h₃ : ∃ L, on L (Extension CD))
variables (h₄ : IntersectLine KL AC = F)

theorem ABFK_can_be_inscribed_in_circle 
  (h₅ : ∃ ⦃A B C D : Point⦄, Rectangle A B C D)
  (h₆ : ∃ {K L : Point}, LineThrough B K ⊥ LineThrough B L)
  (h₇ : ∃ {K F : Point}, LineIntersect KL AC F)
  (h₈ : ∃ {A B K F : Point}, OppositeAnglesSum180 A B K F) :
  ∃ {Ω : Circle}, CyclicQuadrilateral A B K F :=
sorry

end ABFK_can_be_inscribed_in_circle_l148_148884


namespace probability_heads_at_most_3_of_10_coins_flipped_l148_148372

theorem probability_heads_at_most_3_of_10_coins_flipped :
  (∑ k in finset.range 4, nat.choose 10 k) / 2^10 = 11 / 64 :=
sorry

end probability_heads_at_most_3_of_10_coins_flipped_l148_148372


namespace q_sufficient_but_not_necessary_for_p_l148_148025

variable (x : ℝ)

def p : Prop := (x - 2) ^ 2 ≤ 1
def q : Prop := 2 / (x - 1) ≥ 1

theorem q_sufficient_but_not_necessary_for_p : 
  (∀ x, q x → p x) ∧ (∃ x, p x ∧ ¬ q x) := 
by
  sorry

end q_sufficient_but_not_necessary_for_p_l148_148025


namespace confirm_per_capita_volumes_l148_148344

noncomputable def west_volume_per_capita := 21428
noncomputable def non_west_volume_per_capita := 26848.55
noncomputable def russia_volume_per_capita := 302790.13

theorem confirm_per_capita_volumes :
  west_volume_per_capita = 21428 ∧
  non_west_volume_per_capita = 26848.55 ∧
  russia_volume_per_capita = 302790.13 :=
by
  split; sorry
  split; sorry
  sorry

end confirm_per_capita_volumes_l148_148344


namespace statements_correct_l148_148612

def best_decomposition(n : ℕ) : ℕ × ℕ :=
if h : ∃ (s t : ℕ), s * t = n ∧ s ≤ t ∧ ∀ p q, p * q = n → p ≤ q → |p - q| ≥ |s - t| 
then Classical.choose h else (1, n)

def F (n : ℕ) : ℚ :=
let ⟨p, q⟩ := best_decomposition n in
p / q

theorem statements_correct :
  F 2 = 1/2 ∧
  ¬ (F 24 = 3/8) ∧
  ¬ (F 27 = 3) ∧
  (∀ m : ℕ, F (m^2) = 1) :=
by
  -- Proof goes here
  sorry

end statements_correct_l148_148612


namespace sum_series_correct_l148_148973

-- Define the series term
def series_term (n : ℕ) : ℕ :=
  n * (1 - 1 / n)

-- Define the finite sum function
def sum_series (m n : ℕ) := (∑ k in Finset.range (n - (m - 1)), series_term (k + m))

theorem sum_series_correct : sum_series 2 15 = 91 := sorry

end sum_series_correct_l148_148973


namespace minimum_rotation_angle_of_square_l148_148950

theorem minimum_rotation_angle_of_square : 
  ∀ (angle : ℝ), (∃ n : ℕ, angle = 360 / n) ∧ (n ≥ 1) ∧ (n ≤ 4) → angle = 90 :=
by
  sorry

end minimum_rotation_angle_of_square_l148_148950


namespace probability_heads_at_most_3_l148_148527

theorem probability_heads_at_most_3 (num_coins : ℕ) (num_heads_at_most : ℕ) (probability : ℚ):
  num_coins = 10 → 
  num_heads_at_most = 3 → 
  probability = (binomial 10 0 + binomial 10 1 + binomial 10 2 + binomial 10 3) / (2^10) →
  probability = (11 / 64) := 
by
  intros hnum_coins hnum_heads_at_most hprobability
  sorry

end probability_heads_at_most_3_l148_148527


namespace hexagon_monochromatic_triangle_probability_l148_148636

theorem hexagon_monochromatic_triangle_probability :
  ∃ (p : ℚ), p = 1048575 / 1048576 ∧ 
    let edges := (∅ : set (fin 6 × fin 6)), 
    ∀ (e : fin 6 × fin 6) (h : e ∈ edges), 
      (random_color : ℕ → bool),
      (∀ (e : fin 6 × fin 6), ∈ edges → random_color e = tt ∨ random_color e = ff) ∧
      (∀ (e : fin 6 × fin 6), random_color e = tt ∨ random_color e = ff) → 
      ∃ (triangle : finset (fin 6 × fin 6)), 
      triangle.card = 3 ∧ 
      ∀ (e : fin 6 × fin 6), e ∈ triangle → random_color e = random_color (triangle.min’ sorry) := 
sorry

end hexagon_monochromatic_triangle_probability_l148_148636


namespace bronson_total_profit_l148_148620

def cost_per_bushel := 12
def apples_per_bushel := 48
def selling_price_per_apple := 0.40
def number_of_apples_sold := 100

theorem bronson_total_profit : 
  let cost_per_apple := cost_per_bushel / apples_per_bushel in
  let profit_per_apple := selling_price_per_apple - cost_per_apple in
  let total_profit := profit_per_apple * number_of_apples_sold in
  total_profit = 15 :=
by
  sorry

end bronson_total_profit_l148_148620


namespace central_cell_value_l148_148769

theorem central_cell_value (A B C D E F G H I : ℕ)
  (hDistinct : [A, B, C, D, E, F, G, H, I].nodup) 
  (hRange : ∀ x ∈ [A, B, C, D, E, F, G, H, I], 1 ≤ x ∧ x ≤ 9)
  (hConsecutiveAdj : ∀ (x : ℕ), x ∈ [1, 2, 3, 4, 5, 6, 7, 8, 9] →
                      (x+1 ∈ [A, B, C, D, E, F, G, H, I] →
                      ((x, x+1) ∈ [(A, B), (B, C), (C, F), (F, I), (I, H), (H, G), (G, D), (D, A), (A, D), (B, E), (C, F), (G, H), (H, E), (E, I)] ∨
                       (x, x+1) ∈ [(B, A), (C, B), (F, C), (I, F), (H, I), (G, H), (D, G), (A, D), (D, A), (E, B), (F, C), (H, G), (E, H), (I, E)])))
  (hCornersSum : A + C + G + I = 18):
  E = 7 :=
sorry

end central_cell_value_l148_148769


namespace prob_heads_at_most_3_out_of_10_flips_l148_148457

def coin_flips := 10
def max_heads := 3
def total_outcomes := 2^coin_flips
def favorable_outcomes := (nat.choose coin_flips 0) +
                          (nat.choose coin_flips 1) +
                          (nat.choose coin_flips 2) +
                          (nat.choose coin_flips 3)

theorem prob_heads_at_most_3_out_of_10_flips :
  (favorable_outcomes : ℚ) / total_outcomes = 11 / 64 :=
by
  sorry

end prob_heads_at_most_3_out_of_10_flips_l148_148457


namespace total_age_of_wines_l148_148248

theorem total_age_of_wines (age_carlo_rosi : ℕ) (age_franzia : ℕ) (age_twin_valley : ℕ) 
    (h1 : age_carlo_rosi = 40) (h2 : age_franzia = 3 * age_carlo_rosi) (h3 : age_carlo_rosi = 4 * age_twin_valley) : 
    age_franzia + age_carlo_rosi + age_twin_valley = 170 := 
by
    sorry

end total_age_of_wines_l148_148248


namespace probability_of_at_most_3_heads_l148_148472

-- Definitions based on conditions
def binom : ℕ → ℕ → ℕ
| 0, 0       => 1
| 0, (k + 1) => 0
| (n + 1), 0 => 1
| (n + 1), (k + 1) => binom n k + binom n (k + 1)

def favorable_outcomes : ℕ :=
binom 10 0 + binom 10 1 + binom 10 2 + binom 10 3

def total_outcomes : ℕ := 2 ^ 10

def probability (favorable : ℕ) (total : ℕ) : ℚ :=
favorable / total

-- Theorem statement
theorem probability_of_at_most_3_heads : 
  probability favorable_outcomes total_outcomes = 11 / 64 :=
sorry

end probability_of_at_most_3_heads_l148_148472


namespace prob_heads_at_most_3_out_of_10_flips_l148_148448

def coin_flips := 10
def max_heads := 3
def total_outcomes := 2^coin_flips
def favorable_outcomes := (nat.choose coin_flips 0) +
                          (nat.choose coin_flips 1) +
                          (nat.choose coin_flips 2) +
                          (nat.choose coin_flips 3)

theorem prob_heads_at_most_3_out_of_10_flips :
  (favorable_outcomes : ℚ) / total_outcomes = 11 / 64 :=
by
  sorry

end prob_heads_at_most_3_out_of_10_flips_l148_148448


namespace f_inv_128_l148_148689

section proof_problem

variable {α : Type*} (f : α → ℝ)

-- Given conditions
axiom f_four : f 4 = 2
axiom f_two_x : ∀ x, f (2 * x) = 2 * f x

-- Main theorem stating that the inverse of f at 128 equals 256
theorem f_inv_128 : ∃ x, f x = 128 ∧ x = 256 :=
by
  use 256
  split
  sorry -- Proof goes here

end proof_problem

end f_inv_128_l148_148689


namespace largest_four_digit_int_l148_148294

theorem largest_four_digit_int (n : ℤ) (h1 : 1000 ≤ n) (h2 : n ≤ 9999) (h3 : 45 * n ≡ 180 [MOD 315]) :
  n = 9993 :=
sorry

end largest_four_digit_int_l148_148294


namespace trajectory_sufficient_not_necessary_l148_148854

-- Define for any point P if its trajectory is y = |x|
def trajectory (P : ℝ × ℝ) : Prop :=
  P.2 = abs P.1

-- Define for any point P if its distances to the coordinate axes are equal
def equal_distances (P : ℝ × ℝ) : Prop :=
  abs P.1 = abs P.2

-- The main statement: prove that the trajectory is a sufficient but not necessary condition for equal_distances
theorem trajectory_sufficient_not_necessary (P : ℝ × ℝ) :
  trajectory P → equal_distances P ∧ ¬(equal_distances P → trajectory P) := 
sorry

end trajectory_sufficient_not_necessary_l148_148854


namespace simplify_expression_l148_148318

theorem simplify_expression :
  (- (1 : ℝ) / 27) ^ (-3 / 4) = 3 ^ (3 / 4) :=
sorry

end simplify_expression_l148_148318


namespace number_of_roots_ffx_eq_3_l148_148685

noncomputable def f (x : ℝ) : ℝ :=
if x ≤ 0 then 2 * x + 1 else abs (Real.log x)

theorem number_of_roots_ffx_eq_3 : (finset.filter (λ x : ℝ, f (f x) = 3) finset.univ).card = 5 := by
  sorry

end number_of_roots_ffx_eq_3_l148_148685


namespace find_abc_pairs_l148_148052

theorem find_abc_pairs :
  ∀ (a b c : ℕ), 1 < a ∧ a < b ∧ b < c ∧ (a-1)*(b-1)*(c-1) ∣ a*b*c - 1 → 
  (a = 2 ∧ b = 4 ∧ c = 8) ∨ (a = 3 ∧ b = 5 ∧ c = 15) :=
by
  -- Proof omitted
  sorry

end find_abc_pairs_l148_148052


namespace ab_value_l148_148754

-- Defining the conditions as Lean assumptions
theorem ab_value (a b c : ℝ) (h1 : a - b = 3) (h2 : a^2 + b^2 = 27) (h3 : a + b + c = 10) : a * b = 9 :=
by
  sorry

end ab_value_l148_148754


namespace fifth_eqn_nth_eqn_l148_148818

theorem fifth_eqn : 10 * 12 + 1 = 121 :=
by
  sorry

theorem nth_eqn (n : ℕ) : 2 * n * (2 * n + 2) + 1 = (2 * n + 1)^2 :=
by
  sorry

end fifth_eqn_nth_eqn_l148_148818


namespace gas_volume_ranking_l148_148342

theorem gas_volume_ranking (Russia_V: ℝ) (Non_West_V: ℝ) (West_V: ℝ)
  (h_russia: Russia_V = 302790.13)
  (h_non_west: Non_West_V = 26848.55)
  (h_west: West_V = 21428): Russia_V > Non_West_V ∧ Non_West_V > West_V :=
by
  have h1: Russia_V = 302790.13 := h_russia
  have h2: Non_West_V = 26848.55 := h_non_west
  have h3: West_V = 21428 := h_west
  sorry


end gas_volume_ranking_l148_148342


namespace tangent_y_intercept_range_l148_148953

theorem tangent_y_intercept_range :
  ∀ (x₀ : ℝ), (∃ y₀ : ℝ, y₀ = Real.exp x₀ ∧ (∃ m : ℝ, m = Real.exp x₀ ∧ ∃ b : ℝ, b = Real.exp x₀ * (1 - x₀) ∧ b < 0)) → x₀ > 1 := by
  sorry

end tangent_y_intercept_range_l148_148953


namespace circle_center_proof_l148_148930

open Classical

noncomputable def center_of_circle (circle : ℝ × ℝ → Prop) : ℝ × ℝ :=
  let center := (-16/5, 53/10)
  center

theorem circle_center_proof :
  ∃ center : ℝ × ℝ, 
    (circle center) ∧ 
    (center_of_circle circle = (-16/5, 53/10)) :=
by
  sorry

end circle_center_proof_l148_148930


namespace probability_at_most_3_heads_10_flips_l148_148513

theorem probability_at_most_3_heads_10_flips : 
  let num_heads_0 := nat.choose 10 0,
      num_heads_1 := nat.choose 10 1,
      num_heads_2 := nat.choose 10 2,
      num_heads_3 := nat.choose 10 3,
      total_outcomes := 2^10 in
  (num_heads_0 + num_heads_1 + num_heads_2 + num_heads_3) / total_outcomes = 11 / 64 :=
by
  sorry

end probability_at_most_3_heads_10_flips_l148_148513


namespace product_of_1011_in_base2_and_102_in_base3_l148_148626

theorem product_of_1011_in_base2_and_102_in_base3 :
  let bin_1011 := 1 * 2^3 + 0 * 2^2 + 1 * 2^1 + 1 * 2^0,
      ter_102 := 1 * 3^2 + 0 * 3^1 + 2 * 3^0
  in bin_1011 * ter_102 = 121 :=
by
  let bin_1011 := 1 * 2^3 + 0 * 2^2 + 1 * 2^1 + 1 * 2^0
  let ter_102 := 1 * 3^2 + 0 * 3^1 + 2 * 3^0
  have h1 : bin_1011 = 11 := by sorry
  have h2 : ter_102 = 11 := by sorry
  have h3 : 11 * 11 = 121 := by sorry
  exact h3

end product_of_1011_in_base2_and_102_in_base3_l148_148626


namespace leq_radius_square_sub_twice_ratio_squared_height_l148_148016

theorem leq_radius_square_sub_twice_ratio_squared_height 
  (A B C E F : Point) (N H : Point) (O : Point) (R : ℝ)
  (h1 : is_angle_bisector A E)
  (h2 : is_angle_bisector A F)
  (h3 : angle A B E = angle C B F)
  (h4 : perp E N A B)
  (h5 : perp F H B C)
  (h6 : circumradius (triangle A B C) = R)
  : ∥O - E∥^2 = R^2 - 2 * R * (∥E - N∥^2 / ∥F - H∥) := 
sorry

end leq_radius_square_sub_twice_ratio_squared_height_l148_148016


namespace net_salary_decrease_l148_148334

variables (S : ℝ)

def net_salary_change (S : ℝ) : ℝ :=
  let increased_salary := S + 0.25 * S in
  let final_salary := increased_salary - 0.25 * increased_salary in
  final_salary - S

theorem net_salary_decrease (S : ℝ) : net_salary_change S = -0.0625 * S :=
by
  -- The proof would be provided here, but we use sorry for now
  sorry

end net_salary_decrease_l148_148334


namespace isosceles_triangle_base_length_l148_148333

theorem isosceles_triangle_base_length :
  ∀ (equilateral_perimeter isosceles_perimeter : ℕ) (side_length base_length : ℕ),
  equilateral_perimeter = 60 →
  isosceles_perimeter = 65 →
  side_length = equilateral_perimeter / 3 →
  is_triangle_equilateral tri_1 side_length side_length side_length → 
  is_triangle_isosceles tri_2 side_length side_length base_length →
  base_length = 25 :=
-- hypotheses
assume equilateral_perimeter isosceles_perimeter side_length base_length,
assume h1 : equilateral_perimeter = 60,
assume h2 : isosceles_perimeter = 65,
assume h3 : side_length = equilateral_perimeter / 3,
assume h4 : is_triangle_equilateral tri_1 side_length side_length side_length,
assume h5 : is_triangle_isosceles tri_2 side_length side_length base_length,
-- conclusion (proof)
sorry

end isosceles_triangle_base_length_l148_148333


namespace problem_statement_l148_148079

open Nat

noncomputable def a_n (n : ℕ) : ℕ := 2 * n - 1

noncomputable def b_n (n : ℕ) : ℚ := 1 / ((2 * n - 1) * (2 * n + 1))

noncomputable def S_n (n : ℕ) : ℕ := n * n

noncomputable def T_n (n : ℕ) : ℚ := n / (2 * n + 1)

theorem problem_statement :
  (a_n 4 = 7 ∧ a_n 10 = 19) →
  (∀ n, S_n n = (n * (1 + 2 * n - 1)) / 2) →
  (∀ n, ∑ k in range n, b_n k = n / (2 * n + 1)) →
  ∀ n, (a_n n = 2 * n - 1) ∧ (S_n n = n * n) ∧ (T_n n = n / (2 * n + 1)) :=
by
  intro h_arithmetic h_sum_arith h_sum_bn
  sorry

end problem_statement_l148_148079


namespace area_of_triangle_QTU_l148_148264

theorem area_of_triangle_QTU (PQ PS : ℝ) (T U : ℝ → ℝ) (hPQ : PQ = 6) (hPS : PS = 4)
    (T_divides_PR : ∀ t, T t = t * (PQ^2 + PS^2)^0.5 / 4) :
    ∃ area_QTU : ℝ, area_QTU = 3 :=
by
  sorry

end area_of_triangle_QTU_l148_148264


namespace start_day_of_busy_schedule_l148_148192

def goal : Nat := 600
def days_in_september : Nat := 30
def busy_days : Nat := 4
def flight_day : Nat := 23
def flight_day_pages : Nat := 100
def reading_rate : Nat := 20
def required_reading_days : Nat := 25

theorem start_day_of_busy_schedule : Nat :=
  let remaining_pages := goal - flight_day_pages
  let reading_days := remaining_pages / reading_rate
  days_in_september - reading_days - 1 - busy_days

#eval start_day_of_busy_schedule = 26

end start_day_of_busy_schedule_l148_148192


namespace helen_remaining_chocolate_chip_cookies_l148_148129

-- Definitions based on the identified conditions
def helen_baked_yesterday_chocolate_chip : ℕ := 527
def cookies_baked_first_batch : ℕ := 372
def first_batch_ratio_chocolate_chip : ℕ := 3
def first_batch_ratio_raisin : ℕ := 1
def cookies_baked_second_batch : ℕ := 490
def second_batch_ratio_chocolate_chip : ℕ := 5
def second_batch_ratio_raisin : ℕ := 2
def given_away_chocolate_chip : ℕ := 57

-- Lean 4 statement for the mathematically equivalent proof problem
theorem helen_remaining_chocolate_chip_cookies :
  let chocolate_chip_first_batch := (first_batch_ratio_chocolate_chip * cookies_baked_first_batch) / (first_batch_ratio_chocolate_chip + first_batch_ratio_raisin),
      chocolate_chip_second_batch := (second_batch_ratio_chocolate_chip * cookies_baked_second_batch) / (second_batch_ratio_chocolate_chip + second_batch_ratio_raisin),
      total_chocolate_chip_baked := helen_baked_yesterday_chocolate_chip + chocolate_chip_first_batch + chocolate_chip_second_batch,
      remaining_chocolate_chip := total_chocolate_chip_baked - given_away_chocolate_chip
  in remaining_chocolate_chip = 1099 := by
  sorry

end helen_remaining_chocolate_chip_cookies_l148_148129


namespace square_area_error_l148_148603

-- Define the actual side and the erroneous side
def actual_side (s : ℝ) := s
def measured_side (s : ℝ) := 1.02 * s

-- Define the actual area and the erroneous area
def actual_area (s : ℝ) := s^2
def erroneous_area (s : ℝ) := (measured_side s)^2

-- Define the percentage error calculation
def percentage_error (s : ℝ) :=
  ((erroneous_area s - actual_area s) / actual_area s) * 100

-- Main theorem to prove
theorem square_area_error (s : ℝ) (h_positive : s > 0) :
  percentage_error s = 4.04 :=
by
  sorry

end square_area_error_l148_148603


namespace oleg_bought_bar_for_60_rubles_l148_148820

theorem oleg_bought_bar_for_60_rubles (n : ℕ) (h₁ : 96 = n * (1 + n / 100)) : n = 60 :=
by {
  sorry
}

end oleg_bought_bar_for_60_rubles_l148_148820


namespace probability_at_most_3_heads_l148_148443

def binom (n k : ℕ) : ℕ := Nat.choose n k

theorem probability_at_most_3_heads :
  let total_outcomes := 2^10
  let successful_outcomes := binom 10 0 + binom 10 1 + binom 10 2 + binom 10 3
  (successful_outcomes : ℚ) / total_outcomes = 11 / 64 := 
by 
  let total_outcomes := 2 ^ 10
  let successful_outcomes := binom 10 0 + binom 10 1 + binom 10 2 + binom 10 3
  have h : (successful_outcomes : ℚ) / total_outcomes = 11 / 64 := sorry
  exact h

end probability_at_most_3_heads_l148_148443


namespace construct_rectangle_l148_148987

structure Rectangle where
  side1 : ℝ
  side2 : ℝ
  diagonal : ℝ
  sum_diag_side : ℝ := side2 + diagonal

theorem construct_rectangle (b a d : ℝ) (r : Rectangle) :
  r.side2 = a ∧ r.side1 = b ∧ r.sum_diag_side = a + d :=
by
  sorry

end construct_rectangle_l148_148987


namespace exists_infinitely_many_composites_l148_148064

-- Define the function τ(a) which denotes the number of positive divisors of a
def tau (a : ℕ) : ℕ := (finset.range (a + 1)).filter (λ i, a % i = 0).card

-- Define the function f(n) = τ(n!) - τ((n-1)!)
def f (n : ℕ) : ℕ := tau (nat.factorial n) - tau (nat.factorial (n - 1))

-- The main theorem statement
theorem exists_infinitely_many_composites :
  ∃ᶠ (n : ℕ) in at_top, ¬nat.prime n ∧ ∀ (m : ℕ), m < n → f m < f n := 
sorry

end exists_infinitely_many_composites_l148_148064


namespace students_passed_both_tests_l148_148924

theorem students_passed_both_tests
    (total_students : ℕ)
    (passed_long_jump : ℕ)
    (passed_shot_put : ℕ)
    (failed_both : ℕ)
    (h_total : total_students = 50)
    (h_long_jump : passed_long_jump = 40)
    (h_shot_put : passed_shot_put = 31)
    (h_failed_both : failed_both = 4) : 
    (total_students - failed_both = passed_long_jump + passed_shot_put - 25) :=
by 
  sorry

end students_passed_both_tests_l148_148924


namespace largest_prime_factor_of_93_correct_largest_prime_factor_among_given_numbers_l148_148320

def is_prime (n : ℕ) : Prop := 
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def prime_factors (n : ℕ) : List ℕ := 
  List.filter is_prime (List.factors n)

def largest_prime_factor (n : ℕ) : ℕ :=
  List.maximum' (prime_factors n)

theorem largest_prime_factor_of_93_correct :
  largest_prime_factor 93 = 31 :=
by sorry

theorem largest_prime_factor_among_given_numbers :
  ∀ n ∈ [57, 63, 85, 93, 133], largest_prime_factor 93 ≥ largest_prime_factor n :=
by sorry

end largest_prime_factor_of_93_correct_largest_prime_factor_among_given_numbers_l148_148320


namespace table_tennis_championship_max_winners_l148_148561

theorem table_tennis_championship_max_winners 
  (n : ℕ) 
  (knockout_tournament : ∀ (players : ℕ), players = 200) 
  (eliminations_per_match : ℕ := 1) 
  (matches_needed_to_win_3_times : ℕ := 3) : 
  ∃ k : ℕ, k = 66 ∧ k * matches_needed_to_win_3_times ≤ (n - 1) :=
begin
  sorry
end

end table_tennis_championship_max_winners_l148_148561


namespace consecutive_cubes_perfect_square_l148_148032

theorem consecutive_cubes_perfect_square :
  ∃ n k : ℕ, (n + 1)^3 - n^3 = k^2 ∧ 
             (∀ m l : ℕ, (m + 1)^3 - m^3 = l^2 → n ≤ m) :=
sorry

end consecutive_cubes_perfect_square_l148_148032


namespace Sandy_fingernail_growth_rate_l148_148839

-- Definitions based on conditions
def Sandy_current_age : ℕ := 12
def Sandy_goal_age : ℕ := 32
def Sandy_current_fingernail_length : ℕ := 2
def world_record_fingernail_length : ℕ := 26

-- Goal: prove the rate of growth of Sandy's fingernails per month
theorem Sandy_fingernail_growth_rate :
  (world_record_fingernail_length - Sandy_current_fingernail_length) / ((Sandy_goal_age - Sandy_current_age) * 12) = 0.1 := 
begin
  sorry -- proof required
end

end Sandy_fingernail_growth_rate_l148_148839


namespace infinite_ns_with_property_l148_148843

theorem infinite_ns_with_property 
  (S : ℕ → set ℤ)
  (h1 : ∀ n, S n ⊆ {x | ∃ k, x = k → k ∈ ℤ}) 
  (h2 : ∀ n, ∃ a b : ℤ, S n = {a, b} ∧ a + b = n)
  (h3 : ∀ n m, S n ∩ S m = ∅ → n ≠ m) :
  ∃ᶠ n in at_top, ∃ a ∈ S n, a > (13 * n) / 7 :=
by
  sorry

end infinite_ns_with_property_l148_148843


namespace sum_digits_80_eights_80_fives_plus_80_ones_l148_148060

noncomputable def sum_of_digits (n : ℕ) : ℕ :=
n.digits 10 |>.sum

theorem sum_digits_80_eights_80_fives_plus_80_ones :
  let eights := (10^80 - 1) / 9 * 8
      fives := (10^80 - 1) / 9 * 5
      ones := (10^80 - 1) / 9
      result := sum_of_digits (eights * fives + ones)
  in result = 400 :=
by
  sorry

end sum_digits_80_eights_80_fives_plus_80_ones_l148_148060


namespace count_sums_of_cubes_lt_800_l148_148132

def is_cube (n : ℕ) : Prop := ∃ k : ℕ, k^3 = n

theorem count_sums_of_cubes_lt_800 :
  (finset.card 
    ((finset.range 9).product (finset.range 9)).filter 
    (λ ab, is_cube (ab.1 ^ 3 + ab.2 ^ 3) ∧ (ab.1 ^ 3 + ab.2 ^ 3) < 800)) = 34 :=
sorry

end count_sums_of_cubes_lt_800_l148_148132


namespace necessary_and_sufficient_condition_l148_148149

noncomputable def PointOnCurve (F : ℝ × ℝ → ℝ) (a b : ℝ) : Prop :=
  F (a, b) = 0

theorem necessary_and_sufficient_condition (F : ℝ × ℝ → ℝ) (a b : ℝ) :
  curve_condition : (∀ x y : ℝ, F (x, y) = 0 ↔ (x, y) = (a, b)) :=
begin
  sorry, -- Proof to be filled in.
end

end necessary_and_sufficient_condition_l148_148149


namespace probability_of_at_most_3_heads_out_of_10_l148_148493
open_locale big_operators
open nat

theorem probability_of_at_most_3_heads_out_of_10 (n : ℕ) (p : ℕ) (h : 0 < n) (hp : p = 2) :
  ∑ k in finset.range (4), nat.choose n k = 176 → nat.pow p n = 1024 → (∑ k in finset.range (4), nat.choose n k) / (nat.pow p n) = 11 / 64 :=
by { intro h₁ h₂, 
     have : (∑ k in finset.range 4, nat.choose n k) / (nat.pow p n) = 176 / 1024,
     { rw [h₁, h₂] },
     rw this,
     norm_num }

end probability_of_at_most_3_heads_out_of_10_l148_148493


namespace intersection_point_parabola_l148_148081

theorem intersection_point_parabola (n : ℤ) (h1 : 2 ≤ n) (x₀ y₀ : ℤ) 
(h2 : y₀ = x₀) (h3 : y₀^2 = n * x₀ - 1) (m : ℕ) (hm : 1 ≤ m) : 
∃ k : ℤ, 2 ≤ k ∧ k = (x₀^m + x₀^(-m)) := 
  sorry

end intersection_point_parabola_l148_148081


namespace five_times_of_g_at_7_l148_148143

def g (x : ℝ) : ℝ :=
  - (1 / x)

theorem five_times_of_g_at_7 : g (g (g (g (g 7)))) = - (1 / 7) :=
by
  sorry

end five_times_of_g_at_7_l148_148143


namespace smallest_number_of_eggs_l148_148909

theorem smallest_number_of_eggs (c : ℕ) (h1 : 15 * c - 3 > 150) : ∃ n, n = 15 * 11 - 3 :=
by
  use 162
  sorry

end smallest_number_of_eggs_l148_148909


namespace prob_heads_at_most_3_out_of_10_flips_l148_148455

def coin_flips := 10
def max_heads := 3
def total_outcomes := 2^coin_flips
def favorable_outcomes := (nat.choose coin_flips 0) +
                          (nat.choose coin_flips 1) +
                          (nat.choose coin_flips 2) +
                          (nat.choose coin_flips 3)

theorem prob_heads_at_most_3_out_of_10_flips :
  (favorable_outcomes : ℚ) / total_outcomes = 11 / 64 :=
by
  sorry

end prob_heads_at_most_3_out_of_10_flips_l148_148455


namespace find_m_l148_148813

def U : Set ℕ := {1, 2, 3, 4}
def compl_U_A : Set ℕ := {1, 4}

theorem find_m (m : ℕ) (A : Set ℕ) (hA : A = {x | x ^ 2 - 5 * x + m = 0 ∧ x ∈ U}) :
  compl_U_A = U \ A → m = 6 :=
by
  sorry

end find_m_l148_148813


namespace estimation_1_estimation_2_estimation_3_estimation_4_l148_148049

noncomputable def estimate_sum_1 : ℕ := 212 + 384
noncomputable def estimate_sum_2 : ℕ := 903 - 497
noncomputable def estimate_sum_3 : ℕ := 206 + 3060
noncomputable def estimate_sum_4 : ℕ := 523 + 386

theorem estimation_1 : estimate_sum_1 ≈ 600 := sorry
theorem estimation_2 : estimate_sum_2 ≈ 400 := sorry
theorem estimation_3 : estimate_sum_3 ≈ 3200 := sorry
theorem estimation_4 : estimate_sum_4 ≈ 900 := sorry

end estimation_1_estimation_2_estimation_3_estimation_4_l148_148049


namespace matrix_not_invertible_iff_l148_148677

-- Define the determinant of the given matrix
def det_matrix (x : ℝ) : ℝ := (2 * x) * (3 * x) - (5 * 4)

-- Define the predicate that the matrix is not invertible
def is_not_invertible (x : ℝ) : Prop := det_matrix x = 0

-- State the main theorem
theorem matrix_not_invertible_iff (x : ℝ) : is_not_invertible x ↔ (x = sqrt(30) / 3 ∨ x = -sqrt(30) / 3) :=
by
  sorry

end matrix_not_invertible_iff_l148_148677


namespace probability_at_most_3_heads_l148_148499

-- Definitions and conditions
def coin_flips : ℕ := 10
def success_prob : ℚ := 1/2
def max_heads : ℕ := 3

-- Binomial Probability Computation
noncomputable def probability_of_at_most_3_heads : ℚ :=
  (∑ k in finset.range (max_heads + 1), nat.choose coin_flips k) / (2^coin_flips)

-- The statement to prove
theorem probability_at_most_3_heads :
  probability_of_at_most_3_heads = 11/64 := by
  sorry

end probability_at_most_3_heads_l148_148499


namespace infinite_series_computation_l148_148634

theorem infinite_series_computation : 
  ∑' k : ℕ, (8^k) / ((2^k - 1) * (2^(k + 1) - 1)) = 4 :=
by
  sorry

end infinite_series_computation_l148_148634


namespace decimal_digits_of_fraction_l148_148893

noncomputable def fraction : ℚ := 987654321 / (2 ^ 30 * 5 ^ 2)

theorem decimal_digits_of_fraction :
  ∃ n ≥ 30, fraction = (987654321 / 10^2) / 2^28 := sorry

end decimal_digits_of_fraction_l148_148893


namespace arrangement_ways_l148_148008

-- Defining the conditions
def num_basil_plants : Nat := 5
def num_tomato_plants : Nat := 4
def num_total_units : Nat := num_basil_plants + 1

-- Proof statement
theorem arrangement_ways : (num_total_units.factorial) * (num_tomato_plants.factorial) = 17280 := by
  sorry

end arrangement_ways_l148_148008


namespace functional_relationship_point_not_on_graph_l148_148091

noncomputable def proportionalRelationship : Prop := 
  ∃ k : ℝ, ∀ x : ℝ, y = k * (x + 2)

theorem functional_relationship (y : ℝ) (x : ℝ) (h : y = -3 * (x + 2)) :
  y = -3x - 6 :=
by
  sorry

theorem point_not_on_graph (x y : ℝ) (hx : x = 7) (hy : y = -25) (h : y ≠ -3x - 6) : 
  ¬ (y = -3 * x - 6) :=
by
  sorry

end functional_relationship_point_not_on_graph_l148_148091


namespace probability_at_most_three_heads_10_coins_l148_148363

open Nat

noncomputable def probability_at_most_three_heads (n : ℕ) (k : ℕ) : ℚ :=
  (∑ i in Finset.range (k + 1), (Nat.choose n i : ℚ)) / (2 ^ n : ℚ)

theorem probability_at_most_three_heads_10_coins :
  probability_at_most_three_heads 10 3 = 11 / 64 :=
by
  sorry

end probability_at_most_three_heads_10_coins_l148_148363


namespace exponent_quotient_l148_148158

theorem exponent_quotient (x y : ℝ) (h : x * y = 1) : 
  ((5^(x + y))^2 / (5^(x - y))^2) = 5^(4 / x) :=
by 
  sorry

end exponent_quotient_l148_148158


namespace find_y_l148_148655

theorem find_y (y : ℝ) : log 16 (4 * y - 3) = 2 → y = 259 / 4 :=
by
  sorry

end find_y_l148_148655


namespace wine_age_problem_l148_148250

theorem wine_age_problem
  (carlo_rosi : ℕ)
  (franzia : ℕ)
  (twin_valley : ℕ)
  (h1 : franzia = 3 * carlo_rosi)
  (h2 : carlo_rosi = 4 * twin_valley)
  (h3 : carlo_rosi = 40) :
  franzia + carlo_rosi + twin_valley = 170 :=
by
  sorry

end wine_age_problem_l148_148250


namespace whole_numbers_between_sqrt_18_and_sqrt_98_l148_148749

theorem whole_numbers_between_sqrt_18_and_sqrt_98 :
  let lower_bound := Real.sqrt 18
  let upper_bound := Real.sqrt 98
  let smallest_whole_num := 5
  let largest_whole_num := 9
  (largest_whole_num - smallest_whole_num + 1) = 5 :=
by
  -- Introduce variables
  let lower_bound := Real.sqrt 18
  let upper_bound := Real.sqrt 98
  let smallest_whole_num := 5
  let largest_whole_num := 9
  -- Sorry indicates the proof steps are skipped
  sorry

end whole_numbers_between_sqrt_18_and_sqrt_98_l148_148749


namespace problem1_expr_eval_l148_148975

theorem problem1_expr_eval : 
  (1:ℤ) - (1:ℤ)^(2022:ℕ) - (3 * (2/3:ℚ)^2 - (8/3:ℚ) / ((-2)^3:ℤ)) = -8/3 :=
by
  sorry

end problem1_expr_eval_l148_148975


namespace tokensPiperUsed_l148_148223

-- Definitions based on conditions
def tokensPerToken := 15
def macyTokens := 11
def macyHits := 50
def piperHits := 55
def totalMissedPitches := 315

-- Theorem we want to prove
theorem tokensPiperUsed : 
  (macyTokens * tokensPerToken - macyHits + (tokensPerToken * piperTokens - piperHits) = totalMissedPitches) → piperTokens = 17 :=
begin
  sorry
end

end tokensPiperUsed_l148_148223


namespace find_other_root_l148_148709

theorem find_other_root (x y : ℚ) (h : 48 * x^2 - 77 * x + 21 = 0) (hx : x = 3 / 4) : y = 7 / 12 → 48 * y^2 - 77 * y + 21 = 0 := by
  sorry

end find_other_root_l148_148709


namespace probability_at_most_3_heads_10_coins_l148_148426

noncomputable def coin_flip_prob_at_most_3_heads : ℚ :=
  let total_outcomes := 2^10
  let favorable_outcomes := Nat.binomial 10 0 + Nat.binomial 10 1 + Nat.binomial 10 2 + Nat.binomial 10 3
  favorable_outcomes / total_outcomes

theorem probability_at_most_3_heads_10_coins : coin_flip_prob_at_most_3_heads = 11 / 64 :=
by
  -- Definitions of let values are integrated directly in the theorem body
  let total_outcomes := 2^10
  let favorable_outcomes := Nat.binomial 10 0 + Nat.binomial 10 1 + Nat.binomial 10 2 + Nat.binomial 10 3
  show favorable_outcomes / total_outcomes = 11 / 64
  sorry

end probability_at_most_3_heads_10_coins_l148_148426


namespace solve_fractional_equation_l148_148873

theorem solve_fractional_equation (x : ℝ) (h₀ : x ≠ 0) (h₁ : x ≠ 3) : (2 / (x - 3) = 3 / x) → x = 9 :=
by
  sorry

end solve_fractional_equation_l148_148873


namespace andrew_purchased_mangoes_l148_148005

variable (m : ℕ)

def cost_of_grapes := 8 * 70
def cost_of_mangoes (m : ℕ) := 55 * m
def total_cost (m : ℕ) := cost_of_grapes + cost_of_mangoes m

theorem andrew_purchased_mangoes :
  total_cost m = 1055 → m = 9 := by
  intros h_total_cost
  sorry

end andrew_purchased_mangoes_l148_148005


namespace zoey_holidays_in_a_year_l148_148910

-- Given conditions as definitions
def holidays_per_month : ℕ := 2
def months_in_a_year : ℕ := 12

-- Definition of the total holidays in a year
def total_holidays_in_year : ℕ := holidays_per_month * months_in_a_year

-- Proof statement
theorem zoey_holidays_in_a_year : total_holidays_in_year = 24 := 
by
  sorry

end zoey_holidays_in_a_year_l148_148910


namespace apples_discrepancy_l148_148290

-- Definitions of conditions
def first_day_earnings (apples1 apples2 apples_sold1 apples_sold2 krajsz1 krajsz2: ℕ) : ℕ := 
  (apples1 / apples_sold1) * krajsz1 + (apples2 / apples_sold2) * krajsz2

def second_day_avg_price (total_apples apples_sold krajsz : ℚ) : ℚ :=
  (total_apples / apples_sold) * krajsz

-- Given conditions for the problem
theorem apples_discrepancy :
  let apples := 30 in
  let apples_sold1 := 3 in
  let krajsz1 := 2 in
  let apples_sold2 := 2 in
  let krajsz2 := 2 in
  let apples_sold_second_day := 5 in
  let krajsz_second_day := 4 in
  let total_apples := 60 in
  let first_day_total_earnings := (first_day_earnings apples apples apples_sold1 apples_sold2 krajsz1 krajsz2) in
  let second_day_earnings := 48 in
  let avg_price_first_day := first_day_total_earnings / total_apples in
  let avg_price_second_day := (second_day_earnings : ℚ) / total_apples in
  avg_price_first_day - avg_price_second_day = 1/30 :=
by
  sorry

end apples_discrepancy_l148_148290


namespace original_price_of_sarees_l148_148870

theorem original_price_of_sarees (P : ℝ) (h1 : 0.95 * 0.80 * P = 133) : P = 175 :=
sorry

end original_price_of_sarees_l148_148870


namespace angle_B_tan_A_l148_148707

-- Definition of cosine rule, angle B and angle A for a triangle
namespace TriangleProblem

variables {a b c : ℝ} (A B C : ℝ)

-- Given condition for the triangle
axiom triangle_condition (h : a^2 + c^2 - b^2 = ac)

-- Prove part (I)
theorem angle_B (h : a^2 + c^2 - b^2 = ac) : B = Real.pi / 3 :=
  by
  sorry

-- Prove part (II)
theorem tan_A (h : a^2 + c^2 - b^2 = ac) (h1 : c = 3 * a) : (Real.tan A) = Real.sqrt 3 / 5 :=
  by
  sorry

end TriangleProblem

end angle_B_tan_A_l148_148707


namespace erica_took_224_l148_148652

open Rat

def erica_fraction : ℚ :=
  (2 / 7) + (3 / 8) + (5 / 12) + (3 / 5) + (7 / 24) + (11 / 40)

def erica_percentage : ℚ := (erica_fraction * 100)

theorem erica_took_224:
  erica_percentage ≈ 224.40 :=
sorry

end erica_took_224_l148_148652


namespace thief_is_A_l148_148162

def person : Type := {A, B, C, D}

def statement : person → Prop :=
  | A => ¬(thief A)
  | B => (thief C)
  | C => (thief D)
  | D => ¬(thief D)

-- Given conditions:
-- 1. Exactly one person is telling the truth.
-- 2. Exactly one person stole the jewelry.
def conditions : Prop :=
  (∃! p, statement p) ∧ (∃! q, thief q)

-- Goal: Prove that A is the thief.
theorem thief_is_A : conditions → thief A :=
by
  sorry

end thief_is_A_l148_148162


namespace fraction_of_married_men_l148_148614

theorem fraction_of_married_men (num_women : ℕ) (num_single_women : ℕ) (num_married_women : ℕ)
  (num_married_men : ℕ) (total_people : ℕ) 
  (h1 : num_single_women = num_women / 4) 
  (h2 : num_married_women = num_women - num_single_women)
  (h3 : num_married_men = num_married_women) 
  (h4 : total_people = num_women + num_married_men) :
  (num_married_men : ℚ) / (total_people : ℚ) = 3 / 7 := 
by 
  sorry

end fraction_of_married_men_l148_148614


namespace first_book_length_l148_148325

-- Statement of the problem
theorem first_book_length
  (x : ℕ) -- Number of pages in the first book
  (total_pages : ℕ)
  (days_in_two_weeks : ℕ)
  (pages_per_day : ℕ)
  (second_book_pages : ℕ := 100) :
  pages_per_day = 20 ∧ days_in_two_weeks = 14 ∧ total_pages = 280 ∧ total_pages = pages_per_day * days_in_two_weeks ∧ total_pages = x + second_book_pages → x = 180 :=
by
  sorry

end first_book_length_l148_148325


namespace total_spent_correct_l148_148790

def main_dish_jebb := 25
def appetizer_jebb := 12
def dessert_jebb := 7
def main_dish_friend := 22
def appetizer_friend := 10
def dessert_friend := 6

def total_food_cost_jebb := main_dish_jebb + appetizer_jebb + dessert_jebb
def total_food_cost_friend := main_dish_friend + appetizer_friend + dessert_friend

def combined_food_cost := total_food_cost_jebb + total_food_cost_friend

def service_fee (cost : ℝ) : ℝ :=
  if cost >= 70 then cost * 0.15
  else if cost >= 50 then cost * 0.12
  else if cost >= 30 then cost * 0.10
  else 0

def total_bill_before_tip := combined_food_cost + service_fee combined_food_cost

def tip := total_bill_before_tip * 0.18

def final_amount_spent := total_bill_before_tip + tip

theorem total_spent_correct : final_amount_spent = 111.27 := by
  sorry

end total_spent_correct_l148_148790


namespace translation_equivalence_l148_148319

-- Define the movements
inductive Movement
| swingingOnSwing : Movement
| rotatingBlades : Movement
| elevatorMovingUpwards : Movement
| rearWheelMovingBicycle : Movement

-- Define a predicate that determines if a movement is translation
def isTranslation : Movement → Prop
| Movement.swingingOnSwing := False
| Movement.rotatingBlades := False
| Movement.elevatorMovingUpwards := True
| Movement.rearWheelMovingBicycle := False

-- The theorem stating the proof problem
theorem translation_equivalence : 
  isTranslation Movement.elevatorMovingUpwards = True :=
by 
  -- use sorry to skip the proof
  sorry

end translation_equivalence_l148_148319


namespace remaining_pieces_to_fold_l148_148036

-- Define the initial counts of shirts and shorts
def initial_shirts : ℕ := 20
def initial_shorts : ℕ := 8

-- Define the counts of folded shirts and shorts
def folded_shirts : ℕ := 12
def folded_shorts : ℕ := 5

-- The target theorem to prove the remaining pieces of clothing to fold
theorem remaining_pieces_to_fold :
  initial_shirts + initial_shorts - (folded_shirts + folded_shorts) = 11 := 
by
  sorry

end remaining_pieces_to_fold_l148_148036


namespace P1_value_l148_148268

noncomputable def Q (x : Fin 7 → ℝ) : ℝ :=
  (∑ i in Finset.univ, x i)^2 + 2 * (∑ i in Finset.univ, (x i)^2)

theorem P1_value :
  ∃ (P : Fin 7 → (Fin 7 → ℝ) → ℝ),
    (Q = (λ x, ∑ i in Finset.univ, (P i x)^2)) →
    (P 0 (λ _, 1) = 3) :=
begin
  sorry
end

end P1_value_l148_148268


namespace proof_max_sin_MCN_l148_148075

noncomputable def max_sin_MCN {p x y : ℝ} (h_p : p > 0) (h_circle : (x - x)^2 + (y - p)^2 = x^2 + (y - p)^2) : ℝ :=
  1

theorem proof_max_sin_MCN (p : ℝ) (h_p : p > 0) (x y : ℝ)
  (h_circle : ∀ x y C, (C.1 - x)^2 + (C.2 - y)^2 = x^2 + (y - p)^2 ∧ C.1^2 = 2 * p * C.2 ∧ y = 0) :
  max_sin_MCN h_p h_circle = 1 := by
  sorry

end proof_max_sin_MCN_l148_148075


namespace quinary_to_binary_correct_l148_148989

def base5_to_decimal (n : ℕ) : ℕ :=
  match n with
  | 0 => 0
  | _ => (n % 10) + 5 * base5_to_decimal (n / 10)

def decimal_to_binary (n : ℕ) : ℕ :=
  match n with
  | 0 => 0
  | _ => (n % 2) + 10 * decimal_to_binary (n / 2)

theorem quinary_to_binary_correct :
  (decimal_to_binary (base5_to_decimal 324) = 1011001) :=
by
  sorry

end quinary_to_binary_correct_l148_148989


namespace competition_problem_l148_148165

theorem competition_problem (n : ℕ) (s : ℕ) (correct_first_12 : s = (12 * 13) / 2)
    (gain_708_if_last_12_correct : s + 708 = (n - 11) * (n + 12) / 2):
    n = 71 :=
by
  sorry

end competition_problem_l148_148165


namespace increasing_f_sum_x_ge_2_sqrt6_l148_148119

-- Problem 1: Increase property of f
theorem increasing_f (a : ℝ) (h1 : 1 < a) (h2 : a ≤ real.exp 1) :
  ∀ x : ℝ, 2 * x^2 - 3 * x + (1 / (x * real.log a)) ≥ 0 :=
sorry

-- Problem 2: g(x₁) + g(x₂) = 0 implies x₁ + x₂ ≥ 2 + √6
theorem sum_x_ge_2_sqrt6 (x1 x2 : ℝ) (h : x1 > 0) (h_1 : x2 > 0)
  (h2 : - (3 / 2) * x1^2 - 3 * real.log x1 + 6 * x1 + - (3 / 2) * x2^2 - 3 * real.log x2 + 6 * x2 = 0) :
  x1 + x2 ≥ 2 + real.sqrt 6 :=
sorry

end increasing_f_sum_x_ge_2_sqrt6_l148_148119


namespace paintings_in_four_weeks_l148_148958

def weekly_hours := 30
def hours_per_painting := 3
def weeks := 4

theorem paintings_in_four_weeks (w_hours : ℕ) (h_per_painting : ℕ) (n_weeks : ℕ) (result : ℕ) :
  w_hours = weekly_hours →
  h_per_painting = hours_per_painting →
  n_weeks = weeks →
  result = (w_hours / h_per_painting) * n_weeks →
  result = 40 :=
by
  intros
  sorry

end paintings_in_four_weeks_l148_148958


namespace geometric_locus_l148_148341

open Complex

theorem geometric_locus 
  (z z1 z2 : ℂ) (λ : ℝ) (hλ : λ > 0) :
  (λ = 1 → ∃ (line : Set ℂ), z ∈ line) ∧ 
  (λ ≠ 1 → ∃ (circle : Set ℂ), z ∈ circle) :=
  sorry

end geometric_locus_l148_148341


namespace classA_win_championship_expectation_of_X_is_0_9_l148_148779

noncomputable def classA_win_championship_probability : ℝ :=
  let pA := 0.4
  let pB := 0.5
  let pC := 0.8
  pA * pB * pC + (1 - pA) * pB * pC + pA * (1 - pB) * pC + pA * pB * (1 - pC)

theorem classA_win_championship : classA_win_championship_probability = 0.6 := 
sorry

def distribution_of_X : list (ℤ × ℝ) :=
  [ (-3, 0.4 * 0.5 * 0.8),
    (0, (1 - 0.4) * 0.5 * 0.8 + 0.4 * (1 - 0.5) * 0.2 + 0.4 * 0.5 * (1 - 0.8)),
    (3, (1 - 0.4) * 0.5 * 0.2 + (1 - 0.4) * 0.5 * 0.8 + 0.4 * (1 - 0.5) * 0.2),
    (6, (1 - 0.4) * (1 - 0.5) * 0.2) ]

noncomputable def expectation_of_X : ℝ :=
  ∑ (x, px) in distribution_of_X, x * px

theorem expectation_of_X_is_0_9 : expectation_of_X = 0.9 :=
sorry

end classA_win_championship_expectation_of_X_is_0_9_l148_148779


namespace probability_of_at_most_3_heads_out_of_10_l148_148483
open_locale big_operators
open nat

theorem probability_of_at_most_3_heads_out_of_10 (n : ℕ) (p : ℕ) (h : 0 < n) (hp : p = 2) :
  ∑ k in finset.range (4), nat.choose n k = 176 → nat.pow p n = 1024 → (∑ k in finset.range (4), nat.choose n k) / (nat.pow p n) = 11 / 64 :=
by { intro h₁ h₂, 
     have : (∑ k in finset.range 4, nat.choose n k) / (nat.pow p n) = 176 / 1024,
     { rw [h₁, h₂] },
     rw this,
     norm_num }

end probability_of_at_most_3_heads_out_of_10_l148_148483


namespace common_point_of_three_circles_exists_l148_148961

-- Definitions and conditions based on the problem statement
def equilateral_triangle (A B C : Point) : Prop := 
  dist A B = dist B C ∧ dist B C = dist C A

def is_inscribed_in_circle (pts : list Point) (Ω : Circle) : Prop := 
  ∀ p ∈ pts, is_on_circle p Ω

def is_circumscribed_around_circle (A B C : Point) (ω : Circle) : Prop := 
  ∃ P Q : Point, P ∈ line AC ∧ Q ∈ line AB ∧ is_tangent PQ ω

def circle_through_point_with_center (center point : Point) (Ω : Circle) : Prop := 
  is_on_circle point Ω ∧ Ω.center = center

-- Theorem statement
theorem common_point_of_three_circles_exists 
  (A B C P Q X : Point) 
  (Ω ω Ω_b Ω_c : Circle) 
  (h_eq_triangle : equilateral_triangle A B C)
  (h_inscribed_Ω : is_inscribed_in_circle [A, B, C] Ω)
  (h_circumscribed_ω : is_circumscribed_around_circle A B C ω)
  (h_tangent : is_tangent PQ ω)
  (h_Ω_b : circle_through_point_with_center P B Ω_b)
  (h_Ω_c : circle_through_point_with_center Q C Ω_c) :
  is_on_circle X Ω ∧ is_on_circle X Ω_b ∧ is_on_circle X Ω_c := 
sorry 

end common_point_of_three_circles_exists_l148_148961


namespace seq_formula_l148_148699

noncomputable def a : ℕ → ℚ
| 1       := 1
| 2       := 1
| (n + 3) := 1 - (a 1 + a 2 + (Finset.range (n + 1)).sum (λ k, a (k + 3))) / 4

theorem seq_formula (n : ℕ) : a (n + 1) = n.succ / (2 ^ n) := 
sorry

end seq_formula_l148_148699


namespace minimum_value_18_sqrt_3_minimum_value_at_x_3_l148_148056

noncomputable def f (x : ℝ) : ℝ :=
  x^2 + 12*x + 81 / x^3

theorem minimum_value_18_sqrt_3 (x : ℝ) (hx : x > 0) :
  f x ≥ 18 * Real.sqrt 3 :=
by
  sorry

theorem minimum_value_at_x_3 : f 3 = 18 * Real.sqrt 3 :=
by
  sorry

end minimum_value_18_sqrt_3_minimum_value_at_x_3_l148_148056


namespace count_valid_integers_l148_148746

def is_valid_digit_sequence (n : ℕ) : Prop :=
  let digits := Nat.digits 10 n
  n >= 3100 ∧ n < 3500 ∧
  digits.length = 4 ∧
  digits.nodup ∧
  digits = digits.sorted

theorem count_valid_integers : 
  (Finset.range 4000).filter is_valid_digit_sequence).card = 55 :=
by sorry

end count_valid_integers_l148_148746


namespace min_value_of_n_l148_148087

theorem min_value_of_n (n : ℕ) (x : ℕ → ℝ) (h₁ : ∀ i, i < n → abs (x i) < 1)
  (h₂ : ∑ i in Finset.range n, abs (x i) = 19 + abs (∑ i in Finset.range n, x i)) : n ≥ 20 :=
sorry

end min_value_of_n_l148_148087


namespace pipe_fill_time_l148_148824

variable (t : ℝ)

theorem pipe_fill_time (h1 : 0 < t) (h2 : 0 < t / 5) (h3 : (1 / t) + (5 / t) = 1 / 5) : t = 30 :=
by
  sorry

end pipe_fill_time_l148_148824


namespace probability_of_at_most_3_heads_out_of_10_l148_148480
open_locale big_operators
open nat

theorem probability_of_at_most_3_heads_out_of_10 (n : ℕ) (p : ℕ) (h : 0 < n) (hp : p = 2) :
  ∑ k in finset.range (4), nat.choose n k = 176 → nat.pow p n = 1024 → (∑ k in finset.range (4), nat.choose n k) / (nat.pow p n) = 11 / 64 :=
by { intro h₁ h₂, 
     have : (∑ k in finset.range 4, nat.choose n k) / (nat.pow p n) = 176 / 1024,
     { rw [h₁, h₂] },
     rw this,
     norm_num }

end probability_of_at_most_3_heads_out_of_10_l148_148480


namespace perfect_square_trinomial_l148_148750

theorem perfect_square_trinomial (m : ℤ) :
  (∃ a b : ℤ, 9 = a^2 ∧ 16 = b^2 ∧ 9x^2 + m * x * y + 16y^2 = (a * x + b * y)^2) ↔ m = 24 ∨ m = -24 :=
by
  intro h1
  sorry

end perfect_square_trinomial_l148_148750


namespace marks_chemistry_l148_148991

-- Definitions based on conditions
def marks_english : ℕ := 96
def marks_math : ℕ := 98
def marks_physics : ℕ := 99
def marks_biology : ℕ := 98
def average_marks : ℝ := 98.2
def num_subjects : ℕ := 5

-- Statement to prove
theorem marks_chemistry :
  ((marks_english + marks_math + marks_physics + marks_biology : ℕ) + (x : ℕ)) / num_subjects = average_marks →
  x = 100 :=
by
  sorry

end marks_chemistry_l148_148991


namespace general_formula_sequence_l148_148732

variable {a : ℕ → ℝ}
variable {S : ℕ → ℝ}

-- Conditions
axiom sn_condition (n : ℕ) : S n = (1 / 3) * a n + 2 / 3

-- Goal: Find the general formula for {a_n}
theorem general_formula_sequence :
  (∀ n, S n = (1 / 3) * a n + 2 / 3) →
  (∀ n, a n = (-1 / 2) ^ (n - 1)) :=
by
  assume h : ∀ n, S n = (1 / 3) * a n + 2 / 3
  sorry

end general_formula_sequence_l148_148732


namespace problem_l148_148755

theorem problem (h : (0.00027 : ℝ) = 27 / 100000) : (10^5 - 10^3) * 0.00027 = 26.73 := by
  sorry

end problem_l148_148755


namespace robert_time_to_complete_l148_148937

noncomputable def time_to_complete_semicircle_path (length_mile : ℝ) (width_feet : ℝ) (speed_mph : ℝ) (mile_to_feet : ℝ) : ℝ :=
  let diameter_mile := width_feet / mile_to_feet
  let radius_mile := diameter_mile / 2
  let circumference_mile := 2 * Real.pi * radius_mile
  let semicircle_length_mile := circumference_mile / 2
  semicircle_length_mile / speed_mph

theorem robert_time_to_complete :
  time_to_complete_semicircle_path 1 40 5 5280 = Real.pi / 10 :=
by
  sorry

end robert_time_to_complete_l148_148937


namespace sum_of_base7_digits_of_999_l148_148313

theorem sum_of_base7_digits_of_999 : 
  let base7_representation := 2 * 7^3 + 6 * 7^2 + 2 * 7^1 + 5
  in (let digits_sum := 2 + 6 + 2 + 5 in digits_sum = 15) := sorry

end sum_of_base7_digits_of_999_l148_148313


namespace probability_of_target_hit_l148_148716

theorem probability_of_target_hit  :
  let A_hits := 0.9
  let B_hits := 0.8
  ∃ (P_A P_B : ℝ), 
  P_A = A_hits ∧ P_B = B_hits ∧ 
  (∀ events_independent : Prop, 
   events_independent → P_A * P_B = (0.1) * (0.2)) →
  1 - (0.1 * 0.2) = 0.98
:= 
  sorry

end probability_of_target_hit_l148_148716


namespace fixed_point_sum_l148_148118

theorem fixed_point_sum (a : ℝ) (ha_pos : a > 0) (ha_ne_one : a ≠ 1) : 
  let m := 2 in 
  let n := 3 in 
  m + n = 5 := 
by
  sorry

end fixed_point_sum_l148_148118


namespace area_triangle_QPO_l148_148781

-- Definitions according to the problem conditions
variables {A B C D P Q O N M : Type} [MetricSpace A]
variable (trapezoid_ABCD : Trapezoid A B C D) 
variable (BC_bisected_by_DP_at_N : Bisects D P B C N)
variable (AD_bisected_by_CQ_at_M : Bisects C Q A D M)
variable (DP_meets_AB_at_P : Meets D P A B P)
variable (CQ_meets_AB_at_Q : Meets C Q A B Q)
variable (DP_and_CQ_intersect_at_O : Intersects D P C Q O)
variable (area_ABCD : ℝ)
variable (k : ℝ)
variable (area_trapezoid_ABCD_eq_k : area trapezoid_ABCD = k)

-- Statement of the theorem
theorem area_triangle_QPO :
  area (Triangle Q P O) = 9/8 * k :=
by
  -- Proof steps to be added here
  sorry

end area_triangle_QPO_l148_148781


namespace larger_square_area_l148_148613

theorem larger_square_area 
    (s₁ s₂ s₃ s₄ : ℕ) 
    (H1 : s₁ = 20) 
    (H2 : s₂ = 10) 
    (H3 : s₃ = 18) 
    (H4 : s₄ = 12) :
    (s₃ + s₄) > (s₁ + s₂) :=
by
  sorry

end larger_square_area_l148_148613


namespace probability_of_at_most_3_heads_l148_148475

-- Definitions based on conditions
def binom : ℕ → ℕ → ℕ
| 0, 0       => 1
| 0, (k + 1) => 0
| (n + 1), 0 => 1
| (n + 1), (k + 1) => binom n k + binom n (k + 1)

def favorable_outcomes : ℕ :=
binom 10 0 + binom 10 1 + binom 10 2 + binom 10 3

def total_outcomes : ℕ := 2 ^ 10

def probability (favorable : ℕ) (total : ℕ) : ℚ :=
favorable / total

-- Theorem statement
theorem probability_of_at_most_3_heads : 
  probability favorable_outcomes total_outcomes = 11 / 64 :=
sorry

end probability_of_at_most_3_heads_l148_148475


namespace pyramid_volume_l148_148778

noncomputable def volume_pyramid (SA SB SC : ℝ) (height : ℝ) : ℝ :=
  (1 / 3) * (1 / 2 * real.sqrt (SA^2 * SB^2 + SB^2 * SC^2 + SC^2 * SA^2)) * height

theorem pyramid_volume (SA SB SC : ℝ) :
  (∀ (P : ℝ), SA * SB * SC = P) →
  height = 2 * real.sqrt (102 / 55) →
  ∃ CAB : ℝ, CAB = real.arccos (1 / 6 * real.sqrt (17 / 2)) → 
  SA^2 + SB^2 - 5 * SC^2 = 60 →
  volume_pyramid SA SB SC height = 34 * real.sqrt 6 / 3 :=
begin
  intros h_prod h_height h_angle h_eq,
  -- The proof will go here
  sorry
end

end pyramid_volume_l148_148778


namespace prob_heads_at_most_3_out_of_10_flips_l148_148453

def coin_flips := 10
def max_heads := 3
def total_outcomes := 2^coin_flips
def favorable_outcomes := (nat.choose coin_flips 0) +
                          (nat.choose coin_flips 1) +
                          (nat.choose coin_flips 2) +
                          (nat.choose coin_flips 3)

theorem prob_heads_at_most_3_out_of_10_flips :
  (favorable_outcomes : ℚ) / total_outcomes = 11 / 64 :=
by
  sorry

end prob_heads_at_most_3_out_of_10_flips_l148_148453


namespace probability_at_most_3_heads_l148_148438

def binom (n k : ℕ) : ℕ := Nat.choose n k

theorem probability_at_most_3_heads :
  let total_outcomes := 2^10
  let successful_outcomes := binom 10 0 + binom 10 1 + binom 10 2 + binom 10 3
  (successful_outcomes : ℚ) / total_outcomes = 11 / 64 := 
by 
  let total_outcomes := 2 ^ 10
  let successful_outcomes := binom 10 0 + binom 10 1 + binom 10 2 + binom 10 3
  have h : (successful_outcomes : ℚ) / total_outcomes = 11 / 64 := sorry
  exact h

end probability_at_most_3_heads_l148_148438


namespace smallest_x_for_power_l148_148896

theorem smallest_x_for_power (x : ℕ) (h : x > 0) : 
  (1512 * x) ∈ {y | ∃ (n : ℕ), ∃ (k : ℕ), y = k^n} → x = 49 := by
  have h₁ : 1512 = 2^3 * 3^3 * 7 := sorry
  have h₂ : ∀ (x : ℕ), x > 0 → (1512 * x) ∈ {y | ∃ (n : ℕ), ∃ (k : ℕ), y = k^n} → ∃ (n k : ℕ), (1512 * x) = k^n := sorry
  have h₃ : ∀ (x : ℕ), x > 0 → (∃ (n k : ℕ), (1512 * x) = k^n) → (x = 49) := sorry
  exact h₃ x h (h₂ x h)

end smallest_x_for_power_l148_148896


namespace arrange_plants_in_a_row_l148_148014

-- Definitions for the conditions
def basil_plants : ℕ := 5 -- Number of basil plants
def tomato_plants : ℕ := 4 -- Number of tomato plants

-- Theorem statement asserting the number of ways to arrange the plants
theorem arrange_plants_in_a_row : 
  let total_items := basil_plants + 1,
      ways_to_arrange_total_items := Nat.factorial total_items,
      ways_to_arrange_tomato_group := Nat.factorial tomato_plants in
  (ways_to_arrange_total_items * ways_to_arrange_tomato_group) = 17280 := 
by
  sorry

end arrange_plants_in_a_row_l148_148014


namespace probability_of_at_most_3_heads_out_of_10_l148_148494
open_locale big_operators
open nat

theorem probability_of_at_most_3_heads_out_of_10 (n : ℕ) (p : ℕ) (h : 0 < n) (hp : p = 2) :
  ∑ k in finset.range (4), nat.choose n k = 176 → nat.pow p n = 1024 → (∑ k in finset.range (4), nat.choose n k) / (nat.pow p n) = 11 / 64 :=
by { intro h₁ h₂, 
     have : (∑ k in finset.range 4, nat.choose n k) / (nat.pow p n) = 176 / 1024,
     { rw [h₁, h₂] },
     rw this,
     norm_num }

end probability_of_at_most_3_heads_out_of_10_l148_148494


namespace translate_cosine_left_l148_148286

theorem translate_cosine_left 
  (f : ℝ → ℝ) (g : ℝ → ℝ)
  (h1 : ∀ x, f x = cos (2 * x - π / 3))
  (h2 : ∀ x, g x = f (x + π / 6)) :
  ∀ x, g x = cos (2 * x) := 
by 
  sorry

end translate_cosine_left_l148_148286


namespace area_of_ABM_l148_148588

open Real

variables {A B C M K : Type*}
variables [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited M] [Inhabited K]
variables (AC AB BC AK BK : ℝ)

noncomputable def right_triangle := 
  {A B C : Type* // ∃ (AC BC : ℝ), AC = 2 ∧ BC = sqrt 2 ∧ A ≠ B ∧ B ≠ C ∧ A ≠ C ∧ 
  (∃ (AB : ℝ), AB = sqrt (AC^2 + BC^2)) ∧ 
  ∃ (angleACB : ℝ), angleACB = 90}

noncomputable def ratio := 
  {AK AB : ℝ // ∃ (r : ℝ), r = 1/4 ∧ AK = AB * r}

noncomputable def area_triangle_ABC (AB AC BC : ℝ) : ℝ := 
  0.5 * AC * BC

noncomputable def CK :=
  sqrt ((sqrt 2)^2 + (3 * sqrt 6 / 4)^2 - 2 * (sqrt 2) * (3 * sqrt 6 / 4) * (1 / sqrt 3))

noncomputable def KM :=
  (frac 9 4) * (sqrt 2 / sqrt 19)

noncomputable def area_triangle_ABM := 
  1 / 2 * AB * AK * BK / CK

theorem area_of_ABM :
  ∀ (T : right_triangle) (R : ratio), 
    let (AC, BC) := T
    in let (AK_ratio, AB) := R
    in AC = 2 → BC = sqrt 2 → AK_ratio = 1 / 4 → AB = sqrt 6 →
    area_triangle_ABM AB AK BK CK = (9 / 19) * sqrt 2 := 
by 
  intros,
  sorry

end area_of_ABM_l148_148588


namespace prob_adjacent_abby_bridget_l148_148237

-- Definitions of the conditions
def total_students : ℕ := 7
def rows : ℕ := 3
def seats : ℕ := 7
def seats_first_two_rows : ℕ := 2
def seats_last_row : ℕ := 3

-- Abby and Bridget are specific students amongst the 7.
def students := { student : ℕ // student ≤ total_students }

-- We need to express our probability calculation in Lean.
theorem prob_adjacent_abby_bridget : 
  let total_arrangements := Nat.factorial total_students
  let row_pairs := 2 + 2 + 3
  let vertical_pairs := 4
  let abby_bridget_permutations := 2
  let remaining_students_arrangements := Nat.factorial 5
  in (abby_bridget_permutations * remaining_students_arrangements * (row_pairs + vertical_pairs)) / total_arrangements = 11 / 21 :=
by
  sorry

end prob_adjacent_abby_bridget_l148_148237


namespace fraction_value_l148_148980

theorem fraction_value :
  (12^4 + 400) * (24^4 + 400) * (36^4 + 400) * (48^4 + 400) * (60^4 + 400) / 
  ((6^4 + 400) * (18^4 + 400) * (30^4 + 400) * (42^4 + 400) * (54^4 + 400)) = 244.375 :=
by
  -- The proof would be provided here, but we are skipping it as per the instructions.
  sorry

end fraction_value_l148_148980


namespace wine_age_problem_l148_148249

theorem wine_age_problem
  (carlo_rosi : ℕ)
  (franzia : ℕ)
  (twin_valley : ℕ)
  (h1 : franzia = 3 * carlo_rosi)
  (h2 : carlo_rosi = 4 * twin_valley)
  (h3 : carlo_rosi = 40) :
  franzia + carlo_rosi + twin_valley = 170 :=
by
  sorry

end wine_age_problem_l148_148249


namespace speed_of_stream_l148_148335

theorem speed_of_stream (D v : ℝ) (h1 : ∀ D, D / (54 - v) = 2 * (D / (54 + v))) : v = 18 := 
sorry

end speed_of_stream_l148_148335


namespace zero_of_f_l148_148276

noncomputable def f (x : ℝ) : ℝ := 2^x - 8

theorem zero_of_f :
  ∃ x : ℝ, f x = 0 ∧ x = 3 := by
  use 3
  split
  . unfold f
    norm_num
  . rfl

end zero_of_f_l148_148276


namespace main_theorem_l148_148201

-- Define all necessary variables and assumptions
def proof_problem (k : ℕ+) (n : ℕ) (a : Fin (n - 1) → ℕ) (g : Fin (n - 1) → ℕ) : Prop :=
  k > 0 ∧ n = 2^k + 1 ∧
  (∀ i, g i ∈ {1..(n - 1)}) ∧
  (∀ i, a i ∈ {1..(n - 1)}) ∧
  (∀ i : Fin (n - 1), n ∣ (g i)^(a i) - (a (i + 1)))

-- Formalize the theorem
theorem main_theorem (k : ℕ+) (n : ℕ) (a : Fin (n - 1) → ℕ) (g : Fin (n - 1) → ℕ) :
  n = 2^k + 1 → (IsPrime n ↔ ∃ a g, proof_problem k n a g) :=
by
  sorry

end main_theorem_l148_148201


namespace base6_divisibility_l148_148166

theorem base6_divisibility (y : ℕ) (h : y ∈ {0, 1, 2, 3, 4, 5}) : 
  (6 * y + 578) % 13 = 0 ↔ y = 3 :=
sorry

end base6_divisibility_l148_148166


namespace prob_heads_at_most_3_out_of_10_flips_l148_148449

def coin_flips := 10
def max_heads := 3
def total_outcomes := 2^coin_flips
def favorable_outcomes := (nat.choose coin_flips 0) +
                          (nat.choose coin_flips 1) +
                          (nat.choose coin_flips 2) +
                          (nat.choose coin_flips 3)

theorem prob_heads_at_most_3_out_of_10_flips :
  (favorable_outcomes : ℚ) / total_outcomes = 11 / 64 :=
by
  sorry

end prob_heads_at_most_3_out_of_10_flips_l148_148449


namespace total_investment_sum_l148_148890

theorem total_investment_sum 
  (raghu_investment : ℝ) 
  (vishal_invested_10_more_than_trishul : ∀ (trishul_investment : ℝ), vishal_investment = 1.1 * trishul_investment)
  (trishul_invested_10_less_than_raghu : ∀ (raghu_investment : ℝ), trishul_investment = 0.9 * raghu_investment)
  (raghu_invested : raghu_investment = 2100) : 
  raghu_investment 
  + (trishul_investment (raghu_investment)) 
  + (vishal_investment (trishul_investment (raghu_investment))) = 6069 := 
begin
  -- since the proof is not required, we add sorry
  sorry
end

end total_investment_sum_l148_148890


namespace tissue_magnification_l148_148003

theorem tissue_magnification (d_image d_actual : ℝ) (h_image : d_image = 0.3) (h_actual : d_actual = 0.0003) :
  (d_image / d_actual) = 1000 :=
by
  sorry

end tissue_magnification_l148_148003


namespace least_number_to_subtract_l148_148898

theorem least_number_to_subtract (n : ℕ) (m : ℕ) : n = 10154 → m = 30 → n % m = 14 :=
by
  intros hn hm
  rw [hn, hm]
  norm_num
  sorry

end least_number_to_subtract_l148_148898


namespace slope_of_parallel_line_l148_148895

open Real

theorem slope_of_parallel_line (a b c : ℝ) (h : a = 3) (h2 : b = 6) (h3 : c = -12) : -1 / 2 = -1 / 2 :=
by
  -- Given a = 3, b = 6, and c = -12, the slope of the line parallel to ax + by = c is the same as the slope of the line
  have h_slope : -a / b = -1 / 2,
  { rw [h, h2], -- Substitute a = 3, b = 6
    norm_num, -- Simplify -3 / 6 to -1 / 2
  },
  -- Parallel lines have the same slope, so the slope of the line parallel to 3x + 6y = -12 is -1 / 2
  exact h_slope

end slope_of_parallel_line_l148_148895


namespace probability_at_most_3_heads_10_coins_l148_148422

noncomputable def coin_flip_prob_at_most_3_heads : ℚ :=
  let total_outcomes := 2^10
  let favorable_outcomes := Nat.binomial 10 0 + Nat.binomial 10 1 + Nat.binomial 10 2 + Nat.binomial 10 3
  favorable_outcomes / total_outcomes

theorem probability_at_most_3_heads_10_coins : coin_flip_prob_at_most_3_heads = 11 / 64 :=
by
  -- Definitions of let values are integrated directly in the theorem body
  let total_outcomes := 2^10
  let favorable_outcomes := Nat.binomial 10 0 + Nat.binomial 10 1 + Nat.binomial 10 2 + Nat.binomial 10 3
  show favorable_outcomes / total_outcomes = 11 / 64
  sorry

end probability_at_most_3_heads_10_coins_l148_148422


namespace max_expression_value_l148_148146

open Real

theorem max_expression_value (x y : ℝ)
  (h : x^2 + y^2 ≤ 5) : 
  ∃ (M : ℝ), M = 27 + 6 * sqrt 5 ∧ 
             ∀ (x y : ℝ), x^2 + y^2 ≤ 5 → 
                           3 * |(x + y)| + |(4 * y + 9)| + |(7 * y - 3 * x - 18)| ≤ M := 
begin
  use 27 + 6 * sqrt 5,
  intros x y h,
  sorry
end

end max_expression_value_l148_148146


namespace probability_at_most_3_heads_l148_148506

-- Definitions and conditions
def coin_flips : ℕ := 10
def success_prob : ℚ := 1/2
def max_heads : ℕ := 3

-- Binomial Probability Computation
noncomputable def probability_of_at_most_3_heads : ℚ :=
  (∑ k in finset.range (max_heads + 1), nat.choose coin_flips k) / (2^coin_flips)

-- The statement to prove
theorem probability_at_most_3_heads :
  probability_of_at_most_3_heads = 11/64 := by
  sorry

end probability_at_most_3_heads_l148_148506


namespace total_cost_l148_148913

noncomputable def C1 : ℝ := 990 / 1.10
noncomputable def C2 : ℝ := 990 / 0.90

theorem total_cost (SP : ℝ) (profit_rate loss_rate : ℝ) : SP = 990 ∧ profit_rate = 0.10 ∧ loss_rate = 0.10 →
  C1 + C2 = 2000 :=
by
  intro h
  -- Show the sum of C1 and C2 equals 2000
  sorry

end total_cost_l148_148913


namespace increase_diameter_of_cd_l148_148572

theorem increase_diameter_of_cd (initial_diameter : ℝ) (hole_diameter : ℝ) : ∀ (added_diameter : ℝ),
  initial_diameter = 5 ∧ hole_diameter = 1 ∧
  (capacity : ℝ) -> 
  capacity = (π * ((initial_diameter / 2) ^ 2 - (hole_diameter / 2) ^ 2)) ∧
  (new_capacity : ℝ -> capacity * 2 = new_capacity) ∧
  (new_radius : ℝ -> initial_diameter / 2 + added_diameter / 2 = new_radius ) ∧
  (new_area : ℝ -> π * ((new_radius) ^ 2 - (hole_diameter / 2) ^ 2) = new_capacity) -> 
  added_diameter = 2 := 
by sorry

end increase_diameter_of_cd_l148_148572


namespace findHyperbolaEquation_l148_148106

noncomputable def hyperbolaEquation (m a b : ℝ) (e : ℝ) : Prop :=
  (∀ x y : ℝ, (x^2 / (4 + m^2) + y^2 / m^2 = 1) → (x^2 / a^2 - y^2 / b^2 = 1)) ∧
  (a^2 + b^2 = 4) ∧ (e = 2) → (a = 1 ∧ b = real.sqrt 3)

theorem findHyperbolaEquation (m : ℝ) (a b e : ℝ):
  hyperbolaEquation m a b e :=
sorry

end findHyperbolaEquation_l148_148106


namespace lcm_5_6_10_18_l148_148298

theorem lcm_5_6_10_18 : Nat.lcm (5 :: 6 :: 10 :: 18 :: []) = 90 := 
by
  sorry

end lcm_5_6_10_18_l148_148298


namespace exists_n_consecutive_composite_l148_148826

theorem exists_n_consecutive_composite (n : ℕ) :
  ∃ (seq : Fin (n+1) → ℕ), (∀ i, seq i > 1 ∧ ¬ Nat.Prime (seq i)) ∧
                          ∀ k, seq k = (n+1)! + (2 + k) := 
by
  sorry

end exists_n_consecutive_composite_l148_148826


namespace cylinder_surface_area_l148_148587

-- Define the necessary variable conditions
def right_cylinder_height : ℝ := 8
def right_cylinder_radius : ℝ := 3

-- Define the formula for calculating the total surface area of the right cylinder
noncomputable def total_surface_area (r h : ℝ) : ℝ := 2 * π * r * h + 2 * π * r^2

-- State the theorem
theorem cylinder_surface_area : total_surface_area right_cylinder_radius right_cylinder_height = 66 * π :=
by
  sorry

end cylinder_surface_area_l148_148587


namespace sum_of_base7_digits_of_999_l148_148315

theorem sum_of_base7_digits_of_999 : 
  let base7_representation := 2 * 7^3 + 6 * 7^2 + 2 * 7^1 + 5
  in (let digits_sum := 2 + 6 + 2 + 5 in digits_sum = 15) := sorry

end sum_of_base7_digits_of_999_l148_148315


namespace principal_amount_l148_148157

theorem principal_amount (P R T SI : ℝ) (hR : R = 4) (hT : T = 5) (hSI : SI = P - 2240) 
    (h_formula : SI = (P * R * T) / 100) : P = 2800 :=
by 
  sorry

end principal_amount_l148_148157


namespace probability_of_at_most_3_heads_l148_148471

-- Definitions based on conditions
def binom : ℕ → ℕ → ℕ
| 0, 0       => 1
| 0, (k + 1) => 0
| (n + 1), 0 => 1
| (n + 1), (k + 1) => binom n k + binom n (k + 1)

def favorable_outcomes : ℕ :=
binom 10 0 + binom 10 1 + binom 10 2 + binom 10 3

def total_outcomes : ℕ := 2 ^ 10

def probability (favorable : ℕ) (total : ℕ) : ℚ :=
favorable / total

-- Theorem statement
theorem probability_of_at_most_3_heads : 
  probability favorable_outcomes total_outcomes = 11 / 64 :=
sorry

end probability_of_at_most_3_heads_l148_148471


namespace painting_cost_l148_148952

theorem painting_cost (east_first : ℕ) (east_diff : ℕ) (west_first : ℕ) (west_diff : ℕ) 
  (num_houses : ℕ) (cost_per_digit : ℕ) : 
  east_first = 5 → east_diff = 6 → west_first = 7 → west_diff = 6 → num_houses = 25 → cost_per_digit = 1 → 
  ∑ i in (range num_houses), digit_count (east_first + i * east_diff) * cost_per_digit + 
  ∑ i in (range num_houses), digit_count (west_first + i * west_diff) * cost_per_digit = 116 :=
begin
  sorry 
end

-- Helper function to count digits of a number.
def digit_count (n : ℕ) : ℕ :=
  if n = 0 then 1 else Nat.log10 n + 1

#check painting_cost

end painting_cost_l148_148952


namespace evaluate_f_at_neg_3_halves_l148_148090

-- Definitions
def odd_fn_period_2 (f : ℝ → ℝ) : Prop :=
  ∀ x, f(-x+1) = -f(x+1) ∧ f(x+2) = f(x)

-- Given function f definition within domain [-1, 0]
def f_on_domain (f : ℝ → ℝ) : Prop :=
  ∀ x, -1 ≤ x ∧ x ≤ 0 → f(x) = -2*x*(x+1)

-- Main proof problem
theorem evaluate_f_at_neg_3_halves (f : ℝ → ℝ) 
  (h1 : odd_fn_period_2 f)
  (h2 : f_on_domain f) :
  f (-3 / 2) = -1 / 2 :=
sorry

end evaluate_f_at_neg_3_halves_l148_148090


namespace trajectory_of_P_l148_148776

-- Definitions and conditions
def polar_curve_C1 (ρ θ : ℝ) : Prop := ρ * sin θ = 2
def segment_condition (ρ ρ1 : ℝ) : Prop := ρ * ρ1 = 4
def polar_to_rectangular (ρ θ : ℝ) : Prop := (ρ = 2 * sin θ) → ((x, y) : ℝ × ℝ) → (x^2 + (y - 1)^2 = 1 ∧ y ≠ 0)

-- Given answers
def rect_coord_eq_C2 : Prop := x^2 + (y - 1)^2 = 1 ∧ y ≠ 0

-- Parametric equations and slope
def parametric_eq_l (t α x y : ℝ) : Prop := x = t * cos α ∧ y = t * sin α ∧ (0 ≤ α < π)
def distance_cond_A (k : ℝ) : Prop := |(dist : ℝ) - (1 / sqrt(1 + k^2)) = (sqrt(3) / 2)|
def slope_cond_A (k : ℝ) : Prop := k = sqrt 3 ∨ k = -sqrt 3

-- Lean statement for the proof
theorem trajectory_of_P (ρ θ ρ1 x y : ℝ) (t α k : ℝ) :
  polar_curve_C1 ρ1 θ →
  segment_condition ρ ρ1 →
  polar_to_rectangular ρ θ →
  rect_coord_eq_C2 →
  parametric_eq_l t α x y →
  distance_cond_A k →
  slope_cond_A k := 
  sorry

end trajectory_of_P_l148_148776


namespace max_M_min_N_l148_148695

noncomputable def M (x y : ℝ) : ℝ := x / (2 * x + y) + y / (x + 2 * y)
noncomputable def N (x y : ℝ) : ℝ := x / (x + 2 * y) + y / (2 * x + y)

theorem max_M_min_N (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
  (∃ t : ℝ, (∀ x y, 0 < x → 0 < y → M x y ≤ t) ∧ (∀ x y, 0 < x → 0 < y → N x y ≥ t) ∧ t = 2 / 3) :=
sorry

end max_M_min_N_l148_148695


namespace tangent_segment_right_angle_l148_148232

-- Definition of the ellipse
def ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  (x^2 / a^2) + (y^2 / b^2) = 1

-- Equation of the tangent at a point (x0, y0) on the ellipse
def tangent_eq (a b x0 y0 x y : ℝ) : Prop :=
  (x0 * x / a^2) + (y0 * y / b^2) = 1

-- Intersection points of the tangent with the vertex tangents
def intersection_points (a b x0 y0 : ℝ) : (ℝ × ℝ) × (ℝ × ℝ) :=
  ((-a, (b^2 / y0) * (1 + (x0 / a))), (a, (b^2 / y0) * (1 - (x0 / a))))

-- Definition of perpendicular slopes
def perpendicular (m n : ℝ) : Prop :=
  m * n = -1

-- Prove the main theorem
theorem tangent_segment_right_angle (a b x0 y0 : ℝ) (h : ellipse a b x0 y0) :
  ∀ (F1 : ℝ × ℝ), 
  ∃ M N, M = (-a, (b^2 / y0) * (1 + (x0 / a))) ∧
         N = (a, (b^2 / y0) * (1 - (x0 / a))) ∧
         ∀ (m n : ℝ), 
         (m = slope (F1.1, F1.2) M) ∧ (n = slope (F1.1, F1.2) N) →
         perpendicular m n :=
begin
  sorry
end

end tangent_segment_right_angle_l148_148232


namespace leftmost_digit_exists_l148_148085

theorem leftmost_digit_exists (d : ℕ) (hd : 1 ≤ d ∧ d ≤ 9) : ∃ n : ℕ, 
(leftmost_digit (2^n) = d ∧ leftmost_digit (3^n) = d) :=
by
  sorry

end leftmost_digit_exists_l148_148085


namespace probability_at_most_3_heads_10_flips_l148_148521

theorem probability_at_most_3_heads_10_flips : 
  let num_heads_0 := nat.choose 10 0,
      num_heads_1 := nat.choose 10 1,
      num_heads_2 := nat.choose 10 2,
      num_heads_3 := nat.choose 10 3,
      total_outcomes := 2^10 in
  (num_heads_0 + num_heads_1 + num_heads_2 + num_heads_3) / total_outcomes = 11 / 64 :=
by
  sorry

end probability_at_most_3_heads_10_flips_l148_148521


namespace probability_heads_at_most_3_l148_148540

theorem probability_heads_at_most_3 (num_coins : ℕ) (num_heads_at_most : ℕ) (probability : ℚ):
  num_coins = 10 → 
  num_heads_at_most = 3 → 
  probability = (binomial 10 0 + binomial 10 1 + binomial 10 2 + binomial 10 3) / (2^10) →
  probability = (11 / 64) := 
by
  intros hnum_coins hnum_heads_at_most hprobability
  sorry

end probability_heads_at_most_3_l148_148540


namespace seventy_three_days_after_monday_is_thursday_l148_148022

def day_of_week : Nat → String
| 0 => "Monday"
| 1 => "Tuesday"
| 2 => "Wednesday"
| 3 => "Thursday"
| 4 => "Friday"
| 5 => "Saturday"
| _ => "Sunday"

theorem seventy_three_days_after_monday_is_thursday :
  day_of_week (73 % 7) = "Thursday" :=
by
  sorry

end seventy_three_days_after_monday_is_thursday_l148_148022


namespace min_sum_of_areas_l148_148641

noncomputable def minimum_rectangle_area_sum (l : ℝ) : ℝ :=
  let x := (27 / 52) * l
  in (3 / 104) * l^2

theorem min_sum_of_areas (l : ℝ) (h : l > 0) : 
  ∃ x : ℝ, 0 < x ∧ x < l ∧ 
  ( (1 / 18) * x ^ 2 + (3 / 50) * (l - x) ^ 2 ) = (3 / 104) * l ^ 2 :=
begin
  use (27 / 52) * l,
  split,
  { linarith },
  split,
  { linarith },
  { sorry }  -- Proof is omitted as per instructions.
end

end min_sum_of_areas_l148_148641


namespace equation_solution_l148_148610

theorem equation_solution :
  ∃ a b c d : ℤ, a > 0 ∧ (∀ x : ℝ, (64 * x^2 + 96 * x - 36) = (a * x + b)^2 + d) ∧ c = -36 ∧ a + b + c + d = -94 :=
by sorry

end equation_solution_l148_148610


namespace arithmetic_sequences_l148_148663

open Nat

theorem arithmetic_sequences
  (a : ℕ → ℕ)
  (h : ∀ n, (∏ i in range n, a i) ∣ (∏ i in range n, a (n + i))): 
  ∃ k, (∀ i, a i = k * (i + 1)) :=
by
  sorry

end arithmetic_sequences_l148_148663


namespace probability_of_at_most_3_heads_l148_148466

-- Definitions based on conditions
def binom : ℕ → ℕ → ℕ
| 0, 0       => 1
| 0, (k + 1) => 0
| (n + 1), 0 => 1
| (n + 1), (k + 1) => binom n k + binom n (k + 1)

def favorable_outcomes : ℕ :=
binom 10 0 + binom 10 1 + binom 10 2 + binom 10 3

def total_outcomes : ℕ := 2 ^ 10

def probability (favorable : ℕ) (total : ℕ) : ℚ :=
favorable / total

-- Theorem statement
theorem probability_of_at_most_3_heads : 
  probability favorable_outcomes total_outcomes = 11 / 64 :=
sorry

end probability_of_at_most_3_heads_l148_148466


namespace probability_heads_at_most_3_of_10_coins_flipped_l148_148370

theorem probability_heads_at_most_3_of_10_coins_flipped :
  (∑ k in finset.range 4, nat.choose 10 k) / 2^10 = 11 / 64 :=
sorry

end probability_heads_at_most_3_of_10_coins_flipped_l148_148370


namespace simplify_expression_l148_148239

theorem simplify_expression (x y : ℝ) : ((3 * x + 22) + (150 * y + 22)) = (3 * x + 150 * y + 44) :=
by
  sorry

end simplify_expression_l148_148239


namespace greatest_value_of_q_sub_r_l148_148861

-- Definitions and conditions as given in the problem statement
def q : ℕ := 44
def r : ℕ := 15
def n : ℕ := 1027
def d : ℕ := 23

-- Statement of the theorem to be proved
theorem greatest_value_of_q_sub_r : 
  ∃ (q r : ℕ), n = d * q + r ∧ 1 ≤ q ∧ 1 ≤ r ∧ q - r = 29 := 
by
  use q
  use r
  have h1 : n = d * q + r := by norm_num
  have h2 : 1 ≤ q := by norm_num
  have h3 : 1 ≤ r := by norm_num
  have h4 : q - r = 29 := by norm_num
  tauto

end greatest_value_of_q_sub_r_l148_148861


namespace probability_heads_at_most_3_of_10_coins_flipped_l148_148375

theorem probability_heads_at_most_3_of_10_coins_flipped :
  (∑ k in finset.range 4, nat.choose 10 k) / 2^10 = 11 / 64 :=
sorry

end probability_heads_at_most_3_of_10_coins_flipped_l148_148375


namespace probability_heads_at_most_3_l148_148542

theorem probability_heads_at_most_3 (num_coins : ℕ) (num_heads_at_most : ℕ) (probability : ℚ):
  num_coins = 10 → 
  num_heads_at_most = 3 → 
  probability = (binomial 10 0 + binomial 10 1 + binomial 10 2 + binomial 10 3) / (2^10) →
  probability = (11 / 64) := 
by
  intros hnum_coins hnum_heads_at_most hprobability
  sorry

end probability_heads_at_most_3_l148_148542


namespace irrational_number_among_list_l148_148955

theorem irrational_number_among_list :
  ¬ irrational 0.121221222 ∧
  ¬ irrational 0.6666 ∧  -- Note: 0.\overline{6} is represented as 0.6666
  ¬ irrational (11 / 2) ∧
  irrational (π / 2) :=
by {
  sorry
}

end irrational_number_among_list_l148_148955


namespace rectangle_area_is_44_l148_148026

-- Definitions (conditions)
def shaded_square_area : ℕ := 4
def side_of_shaded_square : ℕ := Int.sqrt shaded_square_area
def side_of_larger_square : ℕ := 3 * side_of_shaded_square
def large_square_area : ℕ := side_of_larger_square * side_of_larger_square
def non_shaded_small_square_area : ℕ := shaded_square_area
def rectangle_area : ℕ := shaded_square_area + non_shaded_small_square_area + large_square_area

-- Theorem (question == answer)
theorem rectangle_area_is_44 : rectangle_area = 44 := by
  sorry

end rectangle_area_is_44_l148_148026


namespace find_cost_price_l148_148330

variable (CP SP1 SP2 : ℝ)

theorem find_cost_price
    (h1 : SP1 = CP * 0.92)
    (h2 : SP2 = CP * 1.04)
    (h3 : SP2 = SP1 + 140) :
    CP = 1166.67 :=
by
  -- Proof would be filled here
  sorry

end find_cost_price_l148_148330


namespace tangent_line_at_x1_f_nonnegative_iff_l148_148116

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := (x-1) * Real.log x - m * (x+1)

noncomputable def f' (x : ℝ) (m : ℝ) : ℝ := Real.log x + (x-1) / x - m

theorem tangent_line_at_x1 (m : ℝ) (h : m = 1) :
  ∀ x y : ℝ, f x 1 = y → (x = 1) → x + y + 1 = 0 :=
sorry

theorem f_nonnegative_iff (m : ℝ) :
  (∀ x : ℝ, 0 < x → f x m ≥ 0) ↔ m ≤ 0 :=
sorry

end tangent_line_at_x1_f_nonnegative_iff_l148_148116


namespace foodWeightCalculation_l148_148282

open Nat

variable (numPieces : Nat)
variable (weightBowlWithFood weightEmptyBowl : Real)

-- Define the conditions that appear in step a
def totalWeightFood : Real :=
  weightBowlWithFood - weightEmptyBowl

def weightPerPiece : Real :=
  totalWeightFood / numPieces

-- The statement of the theorem we want to prove
theorem foodWeightCalculation
  (h1 : numPieces = 14)
  (h2 : weightBowlWithFood = 11.14)
  (h3 : weightEmptyBowl = 0.5) :
  weightPerPiece = 0.76 := 
by
  sorry

end foodWeightCalculation_l148_148282


namespace probability_of_at_most_3_heads_l148_148401

-- Definitions and conditions
def num_coins : ℕ := 10
def at_most_3_heads_probability : ℚ := 11 / 64

-- Statement of the problem
theorem probability_of_at_most_3_heads (n : ℕ) (p : ℚ) (h1 : n = num_coins) (h2 : p = at_most_3_heads_probability) :
  p = (1 + 10 + 45 + 120 : ℕ) / (2 ^ 10 : ℕ) := by
  sorry

end probability_of_at_most_3_heads_l148_148401


namespace smallest_k_l148_148964

-- Define the table dimensions
def table_dim := (100, 100)

-- Define the type for a cell position
structure Pos :=
  (x : ℕ) (y : ℕ)
  (h1 : x < table_dim.1)
  (h2 : y < table_dim.2)

-- Define a function for valid adjacent moves
def adjacent (p : Pos) : list Pos :=
  [⟨p.x + 1, p.y, sorry, p.h2⟩, ⟨p.x - 1, p.y, sorry, p.h2⟩, ⟨p.x, p.y + 1, p.h1, sorry⟩, ⟨p.x, p.y - 1, p.h1, sorry⟩]
  -- Note: Additional constraints should ensure movements stay within the table boundaries.

-- Define the property of hitting the tank
def can_hit (k : ℕ) : Prop :=
  k = 4 -- Mathematically encoding that hitting the tank for the smallest k is 4

theorem smallest_k (k : ℕ) : k = 4 ↔ can_hit k := 
by
sorry

end smallest_k_l148_148964


namespace probability_at_most_3_heads_l148_148496

-- Definitions and conditions
def coin_flips : ℕ := 10
def success_prob : ℚ := 1/2
def max_heads : ℕ := 3

-- Binomial Probability Computation
noncomputable def probability_of_at_most_3_heads : ℚ :=
  (∑ k in finset.range (max_heads + 1), nat.choose coin_flips k) / (2^coin_flips)

-- The statement to prove
theorem probability_at_most_3_heads :
  probability_of_at_most_3_heads = 11/64 := by
  sorry

end probability_at_most_3_heads_l148_148496


namespace max_diagonal_length_l148_148204

noncomputable def cyclic_quadrilateral_max_diagonal_length (EF FG GH EH : ℝ)
  (cyclic : Cyclic EF FG GH EH) (a_prod_diag_equals_two_areas : (product_diagonals EQ FH = 2 * area EFGH)) :
  Real := 
sqrt(153.67)

theorem max_diagonal_length (EF FG GH EH : ℝ)
  (hEF : EF = 7) (hFG : FG = 8) (hGH : GH = 9) (hEH : EH = 12) 
  (cyclic : Cyclic EF FG GH EH) 
  (a_prod_diag_equals_two_areas : (product_diagonals EQ FH = 2 * area EFGH)) :
  ∃ (EG : ℝ), EG = cyclic_quadrilateral_max_diagonal_length EF FG GH EH cyclic a_prod_diag_equals_two_areas :=
by
  exists sqrt(153.67)
  sorry

end max_diagonal_length_l148_148204


namespace history_paper_pages_l148_148242

theorem history_paper_pages (p d : ℕ) (h1 : p = 11) (h2 : d = 3) : p * d = 33 :=
by
  sorry

end history_paper_pages_l148_148242


namespace smallest_number_of_eggs_l148_148906

theorem smallest_number_of_eggs (c : ℕ) (h1 : 15 * c - 3 > 150) (h2 : c ≥ 11) : 15 * 11 - 3 = 162 :=
by {
  have h3: 15 * c > 153, from nat.mul_lt_mul_of_pos_left h2 (by norm_num), -- proving intermediate steps using existing conditions
  have h4: c ≥ 11, { sorry }, -- to complete the proof, we normally would show there's no smaller integer satisfying the conditions
  exact (by simp),
}

end smallest_number_of_eggs_l148_148906


namespace count_three_digit_even_numbers_l148_148684

-- Definitions based on conditions
def digits : Finset ℕ := {0, 1, 2, 3, 4, 5}
def is_even (n : ℕ) : Prop := n % 2 = 0
def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n < 1000

-- Problem statement in Lean 4
theorem count_three_digit_even_numbers :
  (Finset.filter (λ n, is_even n ∧ is_three_digit n ∧ (n.digits.nodup ∧ n.digits.all (λ d, d ∈ digits))) (Finset.range 1000)).card = 52 := 
sorry

end count_three_digit_even_numbers_l148_148684


namespace last_year_ticket_cost_l148_148198

theorem last_year_ticket_cost (this_year_cost : ℝ) (increase_percentage : ℝ) (last_year_cost : ℝ) :
  this_year_cost = last_year_cost * (1 + increase_percentage) ↔ last_year_cost = 85 :=
by
  let this_year_cost := 102
  let increase_percentage := 0.20
  sorry

end last_year_ticket_cost_l148_148198


namespace largest_y_value_l148_148295

theorem largest_y_value (y : ℝ) (h : 3*y^2 + 18*y - 90 = y*(y + 17)) : y ≤ 3 :=
by
  sorry

end largest_y_value_l148_148295


namespace ab_max_min_sum_l148_148127

-- Define the conditions
variables {a b : ℝ}
axiom h1 : a > 0
axiom h2 : b > 0
axiom h3 : a + 4 * b = 4

-- Problem (1)
theorem ab_max : ∀ a b : ℝ, (a > 0) ∧ (b > 0) ∧ (a + 4 * b = 4) → a * b ≤ 1 :=
by sorry

-- Problem (2)
theorem min_sum : ∀ a b : ℝ, (a > 0) ∧ (b > 0) ∧ (a + 4 * b = 4) → (1 / a) + (4 / b) ≥ 25 / 4 :=
by sorry

end ab_max_min_sum_l148_148127


namespace product_sum_inequality_l148_148233

theorem product_sum_inequality 
  (k n : ℕ) 
  (x : ℕ → ℝ) 
  (hx : ∀ i, 0 < x i) 
  (hk : 0 < k) 
  (hn : 0 < n) : 
  ∏ i in finset.range k, x i * ∑ i in finset.range k, (x i)^(n-1) ≤ ∑ i in finset.range k, (x i)^(n+k-1) := 
by 
  sorry

end product_sum_inequality_l148_148233


namespace sum_of_special_angles_in_pentagon_l148_148855

theorem sum_of_special_angles_in_pentagon 
  (A B C D E : Type) 
  [ConvexPentagon A B C D E] :
  ∃ (α β γ δ ε : ℝ), 
  α = ∠ DAC ∧ β = ∠ EBD ∧ γ = ∠ ACE ∧ δ = ∠ BDA ∧ ε = ∠ CEB → 
  α + β + γ + δ + ε = 180 := sorry

end sum_of_special_angles_in_pentagon_l148_148855


namespace find_magnitude_of_vec_a_l148_148740

-- Define the vectors a and b in the conditions
def vec_a (t : ℝ) : ℝ × ℝ := (t - 2, 3)
def vec_b : ℝ × ℝ := (3, -1)

-- Define the parallel condition 
def parallel_condition (t : ℝ) :=
  let a_plus_2b := (t - 2 + 6, 3 - 2)
  in a_plus_2b.1 * vec_b.2 = a_plus_2b.2 * vec_b.1

-- Define the magnitude of vec_a
def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1 ^ 2 + v.2 ^ 2)

-- The final statement to prove
theorem find_magnitude_of_vec_a : 
  ∃ t : ℝ, parallel_condition t ∧ magnitude (vec_a t) = 3 * Real.sqrt 10 :=
by
  sorry

end find_magnitude_of_vec_a_l148_148740


namespace remaining_pieces_to_fold_l148_148038

-- Define the initial counts of shirts and shorts
def initial_shirts : ℕ := 20
def initial_shorts : ℕ := 8

-- Define the counts of folded shirts and shorts
def folded_shirts : ℕ := 12
def folded_shorts : ℕ := 5

-- The target theorem to prove the remaining pieces of clothing to fold
theorem remaining_pieces_to_fold :
  initial_shirts + initial_shorts - (folded_shirts + folded_shorts) = 11 := 
by
  sorry

end remaining_pieces_to_fold_l148_148038


namespace length_of_train_correct_l148_148911

-- Definitions for the conditions
def speed_kmph : ℝ := 60              -- Speed in km/hr
def time_seconds : ℝ := 9             -- Time in seconds

-- Conversion factor
def kmph_to_mps (speed : ℝ) : ℝ := speed * (1000 / 3600)

-- Converted speed in m/s
def speed_mps : ℝ := kmph_to_mps speed_kmph

-- Compute length of the train
def length_of_train (speed : ℝ) (time : ℝ) : ℝ := speed * time

-- Statement to be proved
theorem length_of_train_correct :
  length_of_train speed_mps time_seconds = 150.03 :=
sorry

end length_of_train_correct_l148_148911


namespace probability_of_at_most_3_heads_l148_148469

-- Definitions based on conditions
def binom : ℕ → ℕ → ℕ
| 0, 0       => 1
| 0, (k + 1) => 0
| (n + 1), 0 => 1
| (n + 1), (k + 1) => binom n k + binom n (k + 1)

def favorable_outcomes : ℕ :=
binom 10 0 + binom 10 1 + binom 10 2 + binom 10 3

def total_outcomes : ℕ := 2 ^ 10

def probability (favorable : ℕ) (total : ℕ) : ℚ :=
favorable / total

-- Theorem statement
theorem probability_of_at_most_3_heads : 
  probability favorable_outcomes total_outcomes = 11 / 64 :=
sorry

end probability_of_at_most_3_heads_l148_148469


namespace smallest_non_palindromic_power_of_11_l148_148058

def is_palindrome (n : ℕ) : Prop :=
  Nat.digits 10 n = Nat.digits 10 n.reverse

noncomputable def smallest_n : ℕ := 5

theorem smallest_non_palindromic_power_of_11 : 
  ∃ n : ℕ, n = smallest_n ∧ ¬ is_palindrome (11^n) ∧ ∀ m < n, is_palindrome (11^m) :=
by
  sorry

end smallest_non_palindromic_power_of_11_l148_148058


namespace probability_at_most_3_heads_l148_148435

def binom (n k : ℕ) : ℕ := Nat.choose n k

theorem probability_at_most_3_heads :
  let total_outcomes := 2^10
  let successful_outcomes := binom 10 0 + binom 10 1 + binom 10 2 + binom 10 3
  (successful_outcomes : ℚ) / total_outcomes = 11 / 64 := 
by 
  let total_outcomes := 2 ^ 10
  let successful_outcomes := binom 10 0 + binom 10 1 + binom 10 2 + binom 10 3
  have h : (successful_outcomes : ℚ) / total_outcomes = 11 / 64 := sorry
  exact h

end probability_at_most_3_heads_l148_148435


namespace probability_at_most_3_heads_10_flips_l148_148515

theorem probability_at_most_3_heads_10_flips : 
  let num_heads_0 := nat.choose 10 0,
      num_heads_1 := nat.choose 10 1,
      num_heads_2 := nat.choose 10 2,
      num_heads_3 := nat.choose 10 3,
      total_outcomes := 2^10 in
  (num_heads_0 + num_heads_1 + num_heads_2 + num_heads_3) / total_outcomes = 11 / 64 :=
by
  sorry

end probability_at_most_3_heads_10_flips_l148_148515


namespace area_of_triangle_ABC_l148_148630

-- Define some preliminary structures
structure Point where
  x : ℝ
  y : ℝ

def dist (p1 p2 : Point) : ℝ :=
  (p1.x - p2.x)^2 + (p1.y - p2.y)^2

-- Define centers of circles
def A := Point.mk (5 * Real.sqrt 2 / 2) (-5 * Real.sqrt 2 / 2)
def B := Point.mk 0 0
def C := Point.mk 8 0

-- Define the function to calculate area of the triangle given its vertices
def triangle_area (A B C : Point) : ℝ :=
  1 / 2 * abs (A.x * (B.y - C.y) + B.x * (C.y - A.y) + C.x * (A.y - B.y))

-- The statement we need to prove
theorem area_of_triangle_ABC :
  triangle_area A B C = 10 * Real.sqrt 2 :=
by
  sorry

end area_of_triangle_ABC_l148_148630


namespace circumference_of_base_l148_148935

-- Definitions used for the problem
def radius : ℝ := 6
def sector_angle : ℝ := 300
def full_circle_angle : ℝ := 360

-- Ask for the circumference of the base of the cone formed by the sector
theorem circumference_of_base (r : ℝ) (theta_sector : ℝ) (theta_full : ℝ) :
  (theta_sector / theta_full) * (2 * π * r) = 10 * π :=
by
  sorry

end circumference_of_base_l148_148935


namespace final_invariant_number_l148_148265

theorem final_invariant_number (M : ℕ) :
  let initial_numbers := list.range' 1 50 in
  let final_number := (list.prod (initial_numbers.map (λ x, 2 * x + 1)) - 1) / 2 in
  (∀ a b, a ∈ list.range' 1 50 → b ∈ list.range' 1 50 → 
    M = (a + b + 2 * a * b)) →
  (M = final_number) :=
by
  sorry

end final_invariant_number_l148_148265


namespace correct_statement_1_l148_148598

theorem correct_statement_1
  (a : Type[Line])
  (α : Type[Plane])
  (h_intersect : intersects a α)
  (h_parallel_in_α : ∀ (l : Type[Line]), (l ∈ α → l ∥ a) → False) : True :=
sorry

end correct_statement_1_l148_148598


namespace product_of_two_numbers_l148_148866

theorem product_of_two_numbers (x y : ℝ) (h1 : x^2 + y^2 = 289) (h2 : x + y = 23) : x * y = 120 :=
sorry

end product_of_two_numbers_l148_148866


namespace find_x_from_arithmetic_mean_l148_148845

theorem find_x_from_arithmetic_mean (x : ℝ) 
  (h : (x + 10 + 18 + 3 * x + 16 + (x + 5) + (3 * x + 6)) / 6 = 25) : 
  x = 95 / 8 := by
  sorry

end find_x_from_arithmetic_mean_l148_148845


namespace ending_number_correct_l148_148279

-- Definitions:
def is_odd_unit_digit (n : ℕ) : Prop :=
  (n % 10 = 1) ∨ (n % 10 = 3) ∨ (n % 10 = 5) ∨ (n % 10 = 7) ∨ (n % 10 = 9)

def count_odd_unit_digits (start end_ : ℕ) : ℕ :=
  ((end_ - start) + 1).to_nat.filter (λ n, is_odd_unit_digit (start + n)).length

-- The math proof problem statement:
theorem ending_number_correct :
  count_odd_unit_digits 400 999 = 300 := 
sorry

end ending_number_correct_l148_148279


namespace probability_heads_at_most_three_out_of_ten_coins_l148_148543

theorem probability_heads_at_most_three_out_of_ten_coins :
  let total_outcomes := 2^10 in
  let favorable_outcomes := Nat.choose 10 0 + Nat.choose 10 1 + Nat.choose 10 2 + Nat.choose 10 3 in
  favorable_outcomes / total_outcomes = 11 / 64 :=
by
  sorry

end probability_heads_at_most_three_out_of_ten_coins_l148_148543


namespace prop_B_prop_D_l148_148902

variable {U : Type} (A B C : Set U)

-- Proposition B
theorem prop_B : (A ∩ B) ∪ C = (A ∪ C) ∩ (B ∪ C) :=
sorry

-- Proposition D
theorem prop_D : compl (compl A) = A :=
sorry

example : (prop_B A B C) ∧ (prop_D A) :=
by
  split
  · exact prop_B A B C
  · exact prop_D A
  sorry

end prop_B_prop_D_l148_148902


namespace average_weight_of_three_l148_148185

theorem average_weight_of_three (Ishmael Ponce Jalen : ℕ) 
  (h1 : Jalen = 160) 
  (h2 : Ponce = Jalen - 10) 
  (h3 : Ishmael = Ponce + 20) : 
  (Ishmael + Ponce + Jalen) / 3 = 160 := 
sorry

end average_weight_of_three_l148_148185


namespace f_is_odd_l148_148986

def f (x : ℝ) : ℝ := |x + 1| - |x - 1|

theorem f_is_odd : ∀ x, f (-x) = -f x := by
  sorry

end f_is_odd_l148_148986


namespace probability_at_most_3_heads_10_flips_l148_148525

theorem probability_at_most_3_heads_10_flips : 
  let num_heads_0 := nat.choose 10 0,
      num_heads_1 := nat.choose 10 1,
      num_heads_2 := nat.choose 10 2,
      num_heads_3 := nat.choose 10 3,
      total_outcomes := 2^10 in
  (num_heads_0 + num_heads_1 + num_heads_2 + num_heads_3) / total_outcomes = 11 / 64 :=
by
  sorry

end probability_at_most_3_heads_10_flips_l148_148525


namespace find_initial_alison_stamps_l148_148611

-- Define initial number of stamps Anna, Jeff, and Alison had
def initial_anna_stamps : ℕ := 37
def initial_jeff_stamps : ℕ := 31
def final_anna_stamps : ℕ := 50

-- Define the assumption that Alison gave Anna half of her stamps
def alison_gave_anna_half (a : ℕ) : Prop :=
  initial_anna_stamps + a / 2 = final_anna_stamps

-- Define the problem of finding the initial number of stamps Alison had
def alison_initial_stamps : ℕ := 26

theorem find_initial_alison_stamps :
  ∃ a : ℕ, alison_gave_anna_half a ∧ a = alison_initial_stamps :=
by
  sorry

end find_initial_alison_stamps_l148_148611


namespace probability_of_at_most_3_heads_out_of_10_l148_148482
open_locale big_operators
open nat

theorem probability_of_at_most_3_heads_out_of_10 (n : ℕ) (p : ℕ) (h : 0 < n) (hp : p = 2) :
  ∑ k in finset.range (4), nat.choose n k = 176 → nat.pow p n = 1024 → (∑ k in finset.range (4), nat.choose n k) / (nat.pow p n) = 11 / 64 :=
by { intro h₁ h₂, 
     have : (∑ k in finset.range 4, nat.choose n k) / (nat.pow p n) = 176 / 1024,
     { rw [h₁, h₂] },
     rw this,
     norm_num }

end probability_of_at_most_3_heads_out_of_10_l148_148482
