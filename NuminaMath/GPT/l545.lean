import Mathlib

namespace three_digit_integers_congruent_to_2_mod_4_l545_545940

theorem three_digit_integers_congruent_to_2_mod_4 : 
  {n : ℤ | 100 ≤ n ∧ n ≤ 999 ∧ n % 4 = 2}.card = 225 :=
by
  sorry

end three_digit_integers_congruent_to_2_mod_4_l545_545940


namespace greatest_value_q_minus_r_l545_545729

theorem greatest_value_q_minus_r {x y : ℕ} (hx : x < 10) (hy : y < 10) (hqr : 9 * (x - y) < 70) :
  9 * (x - y) = 63 :=
sorry

end greatest_value_q_minus_r_l545_545729


namespace Anna_needs_308_tulips_l545_545784

-- Define conditions as assertions or definitions
def number_of_eyes := 2
def red_tulips_per_eye := 8 
def number_of_eyebrows := 2
def purple_tulips_per_eyebrow := 5
def red_tulips_for_nose := 12
def red_tulips_for_smile := 18
def yellow_tulips_background := 9 * red_tulips_for_smile
def additional_purple_tulips_eyebrows := 4 * number_of_eyes * red_tulips_per_eye - number_of_eyebrows * purple_tulips_per_eyebrow
def yellow_tulips_for_nose := 3 * red_tulips_for_nose

-- Define total number of tulips for each color
def total_red_tulips := number_of_eyes * red_tulips_per_eye + red_tulips_for_nose + red_tulips_for_smile
def total_purple_tulips := number_of_eyebrows * purple_tulips_per_eyebrow + additional_purple_tulips_eyebrows
def total_yellow_tulips := yellow_tulips_background + yellow_tulips_for_nose

-- Define the total number of tulips
def total_tulips := total_red_tulips + total_purple_tulips + total_yellow_tulips

theorem Anna_needs_308_tulips :
  total_tulips = 308 :=
sorry

end Anna_needs_308_tulips_l545_545784


namespace combined_profit_percentage_correct_l545_545266

-- Definitions based on the conditions
noncomputable def profit_percentage_A := 30
noncomputable def discount_percentage_A := 10
noncomputable def profit_percentage_B := 24
noncomputable def discount_percentage_B := 15
noncomputable def profit_percentage_C := 40
noncomputable def discount_percentage_C := 20

-- Function to calculate selling price without discount
noncomputable def selling_price_without_discount (cost_price profit_percentage : ℝ) : ℝ :=
  cost_price * (1 + profit_percentage / 100)

-- Assume cost price for simplicity
noncomputable def cost_price : ℝ := 100

-- Calculations based on the conditions
noncomputable def selling_price_A := selling_price_without_discount cost_price profit_percentage_A
noncomputable def selling_price_B := selling_price_without_discount cost_price profit_percentage_B
noncomputable def selling_price_C := selling_price_without_discount cost_price profit_percentage_C

-- Calculate total cost price and the total selling price without any discount
noncomputable def total_cost_price := 3 * cost_price
noncomputable def total_selling_price_without_discount := selling_price_A + selling_price_B + selling_price_C

-- Combined profit
noncomputable def combined_profit := total_selling_price_without_discount - total_cost_price

-- Combined profit percentage
noncomputable def combined_profit_percentage := (combined_profit / total_cost_price) * 100

theorem combined_profit_percentage_correct :
  combined_profit_percentage = 31.33 :=
by
  sorry

end combined_profit_percentage_correct_l545_545266


namespace right_triangles_with_leg_2012_l545_545528

theorem right_triangles_with_leg_2012 :
  ∀ (a b c : ℕ), a = 2012 ∧ a ^ 2 + b ^ 2 = c ^ 2 → 
  (b = 253005 ∧ c = 253013) ∨ 
  (b = 506016 ∧ c = 506020) ∨ 
  (b = 1012035 ∧ c = 1012037) ∨ 
  (b = 1509 ∧ c = 2515) :=
by
  intros
  sorry

end right_triangles_with_leg_2012_l545_545528


namespace right_triangles_with_leg_2012_l545_545520

theorem right_triangles_with_leg_2012 :
  ∃ (a b c : ℕ), (a = 2012 ∧ (a^2 + b^2 = c^2 ∨ b^2 + a^2 = c^2)) ∧
  (b = 253005 ∧ c = 253013 ∨ b = 506016 ∧ c = 506020 ∨ b = 1012035 ∧ c = 1012037 ∨ b = 1509 ∧ c = 2515) :=
begin
  sorry
end

end right_triangles_with_leg_2012_l545_545520


namespace smallest_n_l545_545305

/--
Each of \( 2020 \) boxes in a line contains 2 red marbles, 
and for \( 1 \le k \le 2020 \), the box in the \( k \)-th 
position also contains \( k \) white marbles. 

Let \( Q(n) \) be the probability that James stops after 
drawing exactly \( n \) marbles. Prove that the smallest 
value of \( n \) for which \( Q(n) < \frac{1}{2020} \) 
is 31.
-/
theorem smallest_n (Q : ℕ → ℚ) (hQ : ∀ n, Q n = (2 : ℚ) / ((n + 1) * (n + 2)))
  : ∃ n, Q n < 1/2020 ∧ ∀ m < n, Q m ≥ 1/2020 := by
  sorry

end smallest_n_l545_545305


namespace probability_product_positive_of_independent_selection_l545_545208

theorem probability_product_positive_of_independent_selection :
  let I := set.Icc (-30 : ℝ) (15 : ℝ)
  let P := (λ (x y : ℝ), x ∈ I ∧ y ∈ I ∧ x * y > 0)
  (Prob { x : ℝ × ℝ | P x.1 x.2 } :
    ProbabilitySpace (I × I)) = 5 / 9 :=
by
  sorry

end probability_product_positive_of_independent_selection_l545_545208


namespace right_triangle_area_l545_545686

theorem right_triangle_area (a b c : ℝ) (h1 : a = 30) (h2 : c = 34) (h3 : a^2 + b^2 = c^2) :
  (1 / 2) * a * b = 240 :=
by
  sorry

end right_triangle_area_l545_545686


namespace number_of_frogs_is_two_l545_545517

-- Define the types and basic predicates
universe u
constant Amphibian : Type u
constant IsToad : Amphibian → Prop
constant IsFrog : Amphibian → Prop

variable (Brian Chris LeRoy Mike : Amphibian)

-- Define the conditions
axiom Brian_statement : (IsToad Brian ↔ IsFrog Mike)
axiom Chris_statement : IsFrog LeRoy
axiom LeRoy_statement : IsFrog Chris
axiom Mike_statement : (∃ x y, (x ≠ y) ∧ IsToad x ∧ IsToad y)

-- The conjecture to prove
theorem number_of_frogs_is_two
  (HBrian : IsFrog Brian ∨ IsToad Brian)
  (HChris : IsFrog Chris ∨ IsToad Chris)
  (HLeRoy : IsFrog LeRoy ∨ IsToad LeRoy)
  (HMike : IsFrog Mike ∨ IsToad Mike) :
  ∃ F1 F2, (F1 ≠ F2) ∧ (IsFrog F1) ∧ (IsFrog F2) ∧
  (F1 = Brian ∨ F1 = Chris ∨ F1 = LeRoy ∨ F1 = Mike) ∧
  (F2 = Brian ∨ F2 = Chris ∨ F2 = LeRoy ∨ F2 = Mike) ∧
  ∀ a, (a = Brian ∨ a = Chris ∨ a = LeRoy ∨ a = Mike) →
    (IsFrog a ∨ IsToad a).

end number_of_frogs_is_two_l545_545517


namespace wall_building_time_l545_545123

theorem wall_building_time (n t : ℕ) (h1 : n * t = 48) (h2 : n = 4) : t = 12 :=
by
  -- appropriate proof steps would go here
  sorry

end wall_building_time_l545_545123


namespace barycentric_identity_l545_545227

variables {V : Type*} [AddCommGroup V] [Module ℝ V]

noncomputable def barycentric (α β γ : ℝ) (a b c : V) : V := 
  α • a + β • b + γ • c

theorem barycentric_identity 
  (A B C X : V) 
  (α β γ : ℝ)
  (h : α + β + γ = 1)
  (hXA : X = barycentric α β γ A B C) :
  X - A = β • (B - A) + γ • (C - A) :=
by
  sorry

end barycentric_identity_l545_545227


namespace count_four_digit_numbers_divisible_by_5_and_ending_with_45_l545_545408

def is_four_digit_number (n : ℕ) : Prop := 1000 ≤ n ∧ n < 10000
def ends_with_45 (n : ℕ) : Prop := (n % 100) = 45 
def divisible_by_5 (n : ℕ) : Prop := (n % 5) = 0

theorem count_four_digit_numbers_divisible_by_5_and_ending_with_45 : 
  {n : ℕ | is_four_digit_number n ∧ ends_with_45 n ∧ divisible_by_5 n}.to_finset.card = 90 :=
sorry

end count_four_digit_numbers_divisible_by_5_and_ending_with_45_l545_545408


namespace find_height_to_AC_in_triangle_l545_545958

theorem find_height_to_AC_in_triangle
  (A B C : ℝ)
  (sin_B_eq_sqrt3_sin_A : sin B = sqrt 3 * sin A)
  (BC_eq_sqrt2 : BC = sqrt 2)
  (C_eq_pi_div_six : C = π / 6) :
  height_to_side_AC = sqrt 2 / 2 :=
by 
  sorry

end find_height_to_AC_in_triangle_l545_545958


namespace position_of_digit_5_in_decimal_of_fraction_l545_545949

theorem position_of_digit_5_in_decimal_of_fraction:
  let frac := 325 / 999 in
  let dec := "0.325325325..." in
  (∃ seq: ℕ → char, 
    (∀ n: ℕ, seq n = dec.get! (n % 3 + 1) ) ∧ 
    (seq 2 = '5')) :=
sorry

end position_of_digit_5_in_decimal_of_fraction_l545_545949


namespace simplify_expression_l545_545614

-- conditions
def condition1 (x : ℝ) : Prop := x < 0

-- expression simplification target
def simplified_expression (x : ℝ) : ℝ := (1 - x) * real.sqrt(-x)

noncomputable def original_expression (x : ℝ) : ℝ := real.sqrt(-x^3) - x * real.sqrt(-1 / x)

-- theorem statement
theorem simplify_expression (x : ℝ) (h : condition1 x) : original_expression x = simplified_expression x :=
by sorry

end simplify_expression_l545_545614


namespace tyler_meal_choices_l545_545215

-- Define the total number of different meals Tyler can choose given the conditions.
theorem tyler_meal_choices : 
    (3 * (Nat.choose 5 3) * 4 * 4 = 480) := 
by
    -- Using the built-in combination function and the fact that meat, dessert, and drink choices are directly multiplied.
    sorry

end tyler_meal_choices_l545_545215


namespace cost_of_nail_service_at_Gustran_l545_545917

noncomputable def nailCostGustran : ℤ := 
  let haircut_Gustran := 45
  let facial_Gustran := 22
  let total_Cost := 84
  total_Cost - (haircut_Gustran + facial_Gustran)

theorem cost_of_nail_service_at_Gustran : nailCostGustran = 17 := 
by {
  -- definitions from conditions
  let haircut_Gustran := 45
  let facial_Gustran := 22
  let total_Cost := 84
  -- solve for nail cost
  have h : nailCostGustran = total_Cost - (haircut_Gustran + facial_Gustran) := rfl
  rw [h]
  norm_num -- automatically simplifies the arithmetic expressions to show equality
  sorry
}

end cost_of_nail_service_at_Gustran_l545_545917


namespace silver_coins_change_l545_545508

-- Define the conditions
def condition1 : ℕ × ℕ := (20, 4) -- (20 silver coins, 4 gold coins change)
def condition2 : ℕ × ℕ := (15, 1) -- (15 silver coins, 1 gold coin change)
def cost_of_cloak_in_gold_coins : ℕ := 14

-- Define the theorem to be proven
theorem silver_coins_change (s1 g1 s2 g2 cloak_g : ℕ) (h1 : (s1, g1) = condition1) (h2 : (s2, g2) = condition2) :
  ∃ silver : ℕ, (silver = 10) :=
by {
  sorry
}

end silver_coins_change_l545_545508


namespace base_area_of_hemisphere_is_3_l545_545170

-- Given conditions
def surface_area_of_sphere (r : ℝ) : ℝ := 4 * Real.pi * r^2
def surface_area_of_hemisphere : ℝ := 9

-- Required to prove: The base area of the hemisphere is 3
theorem base_area_of_hemisphere_is_3 (r : ℝ) (h : surface_area_of_hemisphere = 9) :
  π * r^2 = 3 :=
sorry

end base_area_of_hemisphere_is_3_l545_545170


namespace solve_log_eq_l545_545030

theorem solve_log_eq (b x : ℝ) (hb_pos : b > 0) (hb_ne_one : b ≠ 1) (hx_ne_one : x ≠ 1)
  (h : log (x) / log (b^3) + log (b) / log (x^3) = 2) :
  x = b^(3 + 2 * Real.sqrt 2) ∨ x = b^(3 - 2 * Real.sqrt 2) := by
  sorry

end solve_log_eq_l545_545030


namespace britta_wins_if_n_is_prime_l545_545281

noncomputable def winning_strategy (n : ℕ) : Prop :=
  ∃ A B : set ℕ, 
    (∀ x ∈ A, x ∈ (set.range (λ i : ℕ, i + 1) ∩ (set.Ico 1 n \ₛ{n}) ∩ (∅))) ∧ 
    (∀ y ∈ B, y ∈ (set.range (λ i : ℕ, i + 1) ∩ (set.Ico 1 n \ₛ{n}) ∩ (∅))) ∧
    ∀ (x1 x2 : ℕ), x1 ≠ x2 → x1 ∈ A → x2 ∈ A → 
    ∃ (y1 y2 : ℕ), y1 ≠ y2 ∧ y1 ∈ B ∧ y2 ∈ B ∧ 
    (x1 * x2 * (x1 - y1) * (x2 - y2))^(nat.pred n / 2) % n = 1

theorem britta_wins_if_n_is_prime (n : ℕ) (h_odd : n % 2 = 1) (h_ge_5 : 5 ≤ n) : 
  winning_strategy n ↔ nat.prime n :=
sorry

end britta_wins_if_n_is_prime_l545_545281


namespace values_of_x_l545_545863

theorem values_of_x (x : ℕ) (h : Nat.choose 18 x = Nat.choose 18 (3 * x - 6)) : x = 3 ∨ x = 6 :=
by
  sorry

end values_of_x_l545_545863


namespace probability_of_event_A_l545_545145

axiom prob_event_A_and_B : ℝ
axiom prob_event_A_or_B : ℝ
axiom prob_event_B : ℝ

theorem probability_of_event_A :
  prob_event_A_and_B = 0.25 →
  prob_event_A_or_B = 0.8 →
  prob_event_B = 0.65 →
  let prob_event_A := prob_event_A_or_B + prob_event_A_and_B - prob_event_B in
  prob_event_A = 0.4 :=
by
  intros h1 h2 h3
  let prob_event_A := prob_event_A_or_B + prob_event_A_and_B - prob_event_B
  have h : prob_event_A = 0.4 := by
    rw [h2, h3, h1]
    sorry
  exact h

end probability_of_event_A_l545_545145


namespace range_of_a_l545_545363

noncomputable def f (x : ℝ) : ℝ := Real.exp (2 * x) + Real.exp (x + 2) - 2 * Real.exp 4
noncomputable def g (x : ℝ) (a : ℝ) : ℝ := x * x - 3 * a * Real.exp x
def A : Set ℝ := { x | f x = 0 }
def B (a : ℝ) : Set ℝ := { x | g x a = 0 }

theorem range_of_a (a : ℝ) :
  (∃ x₁ ∈ A, ∃ x₂ ∈ B a, |x₁ - x₂| < 1) →
  a ∈ Set.Ici (1 / (3 * Real.exp 1)) ∩ Set.Iic (4 / (3 * Real.exp 4)) :=
sorry

end range_of_a_l545_545363


namespace number_of_two_digit_multiples_of_seven_l545_545024

theorem number_of_two_digit_multiples_of_seven :
  let a := 14 in
  let l := 98 in
  let d := 7 in
  l - a >= 0 ∧ (l - a) % d = 0 →
  let n := (l - a) / d + 1 in
  n = 13 :=
by
  intros a l d h
  let n := (l - a) / d + 1
  have h1 : (l - a) / d + 1 = 13
  { sorry }
  exact h1

end number_of_two_digit_multiples_of_seven_l545_545024


namespace roots_reciprocal_sum_l545_545652

theorem roots_reciprocal_sum (x₁ x₂ : ℝ) 
    (h_roots : x₁ * x₁ + x₁ - 2 = 0 ∧ x₂ * x₂ + x₂ - 2 = 0):
    x₁ ≠ x₂ → (1 / x₁ + 1 / x₂ = 1 / 2) :=
by
  intro h_neq
  sorry

end roots_reciprocal_sum_l545_545652


namespace mod_product_2023_2024_2025_2026_l545_545804

theorem mod_product_2023_2024_2025_2026 :
  (2023 * 2024 * 2025 * 2026) % 7 = 0 :=
by
  have h2023 : 2023 % 7 = 6 := by norm_num
  have h2024 : 2024 % 7 = 0 := by norm_num
  have h2025 : 2025 % 7 = 1 := by norm_num
  have h2026 : 2026 % 7 = 2 := by norm_num
  calc
    (2023 * 2024 * 2025 * 2026) % 7
      = ((2023 % 7) * (2024 % 7) * (2025 % 7) * (2026 % 7)) % 7 : by rw [Nat.mul_mod, Nat.mul_mod, Nat.mul_mod, Nat.mul_mod]
  ... = (6 * 0 * 1 * 2) % 7 : by rw [h2023, h2024, h2025, h2026]
  ... = 0 % 7 : by norm_num
  ... = 0 : by norm_num

end mod_product_2023_2024_2025_2026_l545_545804


namespace base_area_of_hemisphere_is_3_l545_545168

-- Given conditions
def surface_area_of_sphere (r : ℝ) : ℝ := 4 * Real.pi * r^2
def surface_area_of_hemisphere : ℝ := 9

-- Required to prove: The base area of the hemisphere is 3
theorem base_area_of_hemisphere_is_3 (r : ℝ) (h : surface_area_of_hemisphere = 9) :
  π * r^2 = 3 :=
sorry

end base_area_of_hemisphere_is_3_l545_545168


namespace three_non_collinear_points_determine_plane_l545_545967

theorem three_non_collinear_points_determine_plane
  (P1 P2 P3 : Point)
  (h1 : P1 ≠ P2)
  (h2 : P2 ≠ P3)
  (h3 : P1 ≠ P3)
  (h_non_collinear : ¬ collinear P1 P2 P3) : 
  ∃ (plane : Plane), plane_contains_point plane P1 ∧ plane_contains_point plane P2 ∧ plane_contains_point plane P3 :=
by
  sorry

end three_non_collinear_points_determine_plane_l545_545967


namespace part1_max_value_part2_two_zeros_l545_545002

noncomputable def f (x : ℝ) : ℝ := Real.exp x * Real.cos x

theorem part1_max_value :
  ∃ x ∈ Set.Ioo (0:ℝ) (Real.pi / 2), f x = (Real.sqrt 2 / 2) * Real.exp (Real.pi / 4) ∧
  ∀ y ∈ Set.Ioo (0:ℝ) (Real.pi / 2), f y ≤ (Real.sqrt 2 / 2) * Real.exp (Real.pi / 4) :=
by
  sorry

def F (a x : ℝ) : ℝ := (a * f x) / Real.exp x - 1 / x

theorem part2_two_zeros (a : ℝ) (h : a > (4 * Real.sqrt 2) / Real.pi) :
  ∃! x₁ x₂ ∈ Set.Ioo (0:ℝ) (Real.pi / 2), (x₁ ≠ x₂) ∧ F a x₁ = 0 ∧ F a x₂ = 0 :=
by
  sorry

end part1_max_value_part2_two_zeros_l545_545002


namespace arithmetic_geometric_sequence_sum_min_l545_545359

theorem arithmetic_geometric_sequence_sum_min {
  A B C D : ℚ
  (h1 : 2 * B = A + C)        -- A, B, C form an arithmetic sequence
  (h2 : C^2 = B * D)          -- B, C, D form a geometric sequence
  (h3 : 2 * C = 3 * B)        -- C/B = 3/2
} : A + B + C + D = 21 :=
sorry

end arithmetic_geometric_sequence_sum_min_l545_545359


namespace trapezoid_height_l545_545141

-- Definitions for the conditions
def Trapezoid (midline height : ℕ) :=
  ∃ (x y : ℕ), (x + y = midline * 2) ∧ (midline = (x + y) / 2) 

-- Statement of the problem
theorem trapezoid_height {midline : ℕ} (height : ℕ) 
  (h1 : midline = 5)
  (h2 : ∃ r, ∃ x y : ℕ, (x + y = 10) ∧ (midline = (x + y) / 2) ∧ (x = 1)  ∧ (y = 8))
  (h3 : ∃ a1 a2 : ℕ, (a1 / a2 = 7 / 13)) :
  height = 4 :=
  sorry

end trapezoid_height_l545_545141


namespace slant_height_is_5_l545_545372

-- Given data
def base_radius : ℝ := 3
def lateral_surface_area : ℝ := 15 * Real.pi
def circumference_of_base (r : ℝ) : ℝ := 2 * Real.pi * r

-- Definition of the slant height based on the given problem
def slant_height (r a : ℝ) : ℝ := 2 * a / (Real.pi * r)

-- The theorem we want to prove
theorem slant_height_is_5 :
  slant_height base_radius lateral_surface_area = 5 := 
by
  sorry

end slant_height_is_5_l545_545372


namespace area_of_right_triangle_l545_545703

-- Given definitions
def leg_a : ℝ := 30
def hypotenuse_c : ℝ := 34

-- The theorem statement
theorem area_of_right_triangle : 
  ∀ (b : ℝ), b = real.sqrt (hypotenuse_c^2 - leg_a^2) → 
  let area := 1 / 2 * leg_a * b in
  area = 240 := 
by
  intro b
  intro h
  let area := 1 / 2 * leg_a * b
  sorry

end area_of_right_triangle_l545_545703


namespace least_four_digit_integer_l545_545221

def is_divisible (n d : ℕ) : Prop := d ≠ 0 ∧ n % d = 0

def are_all_digits_different (n : ℕ) : Prop :=
  let digits := [n / 1000 % 10, n / 100 % 10, n / 10 % 10, n % 10] in
  list.nodup digits

def is_valid_number (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000 ∧
  are_all_digits_different n ∧
  is_divisible n (n / 1000 % 10) ∧
  is_divisible n (n / 100 % 10) ∧
  is_divisible n (n / 10 % 10)

theorem least_four_digit_integer : ∃ n, is_valid_number n ∧ n = 1240 := sorry

end least_four_digit_integer_l545_545221


namespace log_estimate_lg2_l545_545825

theorem log_estimate_lg2 :
  (10 ^ 3 = 1000) →
  (10 ^ 4 = 10000) →
  (2 ^ 10 = 1024) →
  (2 ^ 11 = 2048) →
  (2 ^ 12 = 4096) →
  (2 ^ 13 = 8192) →
  (3 / 10 < real.log 2 / real.log 10 ∧ real.log 2 / real.log 10 < 4 / 13) :=
by {
  sorry
}

end log_estimate_lg2_l545_545825


namespace angle_between_a_b_l545_545374

section VectorProof

open Real -- For trigonometric functions and constants

variables (a b : ℝ × ℝ)
variables (θ : ℝ)

-- Defining the conditions
def is_unit_vector (v : ℝ × ℝ) : Prop :=
  v.1 ^ 2 + v.2 ^ 2 = 1

def is_perpendicular (v1 v2 : ℝ × ℝ) : Prop :=
  v1.1 * v2.1 + v1.2 * v2.2 = 0

def angle_between (v1 v2 : ℝ × ℝ) : ℝ :=
  Real.arccos ((v1.1 * v2.1 + v1.2 * v2.2) / (sqrt (v1.1 ^ 2 + v1.2 ^ 2) * sqrt (v2.1 ^ 2 + v2.2 ^ 2)))

-- Conditions extracted from part a)
axiom a_is_unit : is_unit_vector a
axiom b_def : b = (2, 2 * sqrt 3)
axiom perp_cond : is_perpendicular a (2 * a.1 + b.1, 2 * a.2 + b.2)

-- Proving the angle is 2π/3
theorem angle_between_a_b : angle_between a b = 2 * π / 3 :=
  sorry

end VectorProof

end angle_between_a_b_l545_545374


namespace ellipse_eq_first_ellipse_eq_second_l545_545838

-- Definitions per conditions
def p1 := (Real.sqrt 6, 1)
def p2 := (-Real.sqrt 3, -Real.sqrt 2)
def P := (3, 0)

-- First condition ellipse problem
theorem ellipse_eq_first :
  ∃ m n : ℝ, m > 0 ∧ n > 0 ∧ m ≠ n ∧ 6 * m + n = 1 ∧ 3 * m + 2 * n = 1 ∧ (x ^ 2) / 9 + (y ^ 2) / 3 = 1 :=
by sorry

-- Second condition ellipse problem
theorem ellipse_eq_second :
  (∃ a b : ℝ, a > b ∧ b > 0 ∧ a = 3 ∧ 2 * a = 3 * 2 * b ∧ (x ^ 2) / 9 + y ^ 2 = 1)
  ∨ (∃ a b : ℝ, a > b ∧ b > 0 ∧ b = 3 ∧ 2 * a = 3 * 2 * b ∧ (y ^ 2) / 81 + (x ^ 2) / 9 = 1) :=
by sorry

end ellipse_eq_first_ellipse_eq_second_l545_545838


namespace right_triangle_area_l545_545681

theorem right_triangle_area (leg1 hypotenuse : ℝ) (h1 : leg1 = 30) (h2 : hypotenuse = 34) (h3 : (leg1 ^ 2 + (sqrt (hypotenuse ^ 2 - leg1 ^ 2)) ^ 2 = hypotenuse ^ 2)) :
  (1 / 2) * leg1 * (sqrt (hypotenuse ^ 2 - leg1 ^ 2)) = 240 :=
by
  sorry

end right_triangle_area_l545_545681


namespace geometry_statements_correct_l545_545277

theorem geometry_statements_correct:
  (∀ l p, ∃! m, (m ∥ l ∧ p ∉ l) → m ∥ l)
  ∧ (∀ l p, ∃! m, (m ⊥ l ∧ p ∉ l) → m ⊥ l) 
  ∧ (∀ l m₁ m₂, (m₁ ⊥ l ∧ m₂ ⊥ l) → m₁ ∥ m₂) 
  ∧ (∀ l₁ l₂ l₃, (l₁ ∥ l₂ ∧ l₂ ∥ l₃) → l₁ ∥ l₃)
  ∧ (∀ l₁ l₂, (l₃ ∠ (l₁, l₂) = 180) → l₁ ∥ l₂)
  ∧ (∀ A B, distance A B = line_segment_length A B) := 
sorry

end geometry_statements_correct_l545_545277


namespace part_I_solution_set_part_II_inequality_l545_545898

def f (x a : ℝ) : ℝ := abs (x + a) + abs (x + 1 / a)

theorem part_I_solution_set (x : ℝ) : f x 2 > 3 ↔ x < -(11/4) ∨ x > 1/4 :=
by 
  sorry

theorem part_II_inequality (m a : ℝ) (h : a > 0) : f m a + f (-1 / m) a ≥ 4 :=
by 
  sorry

end part_I_solution_set_part_II_inequality_l545_545898


namespace range_of_a_l545_545908

-- Define the sets A and B
def A := {y : ℝ | ∃ x : ℝ, y = log (x) / log (2) ∧ x > 1}
def B := {y : ℝ | ∃ x : ℝ, y = 2 ^ x ∧ 0 < x ∧ x < 2}

-- Define the set C as the domain of the function f
def C (a : ℝ) := {x : ℝ | a < x ∧ x ≤ a + 1}

-- State the theorem including intersection and subset conditions
theorem range_of_a (a : ℝ) : (∀ x, x ∈ C a → x ∈ {y : ℝ | 1 < y ∧ y < 4}) → 1 ≤ a ∧ a ≤ 3 :=
by 
  sorry

end range_of_a_l545_545908


namespace diet_soda_bottles_l545_545752

-- Define the conditions and then state the problem
theorem diet_soda_bottles (R D : ℕ) (h1 : R = 67) (h2 : R = D + 58) : D = 9 :=
by
  -- The proof goes here
  sorry

end diet_soda_bottles_l545_545752


namespace smallest_good_number_exists_l545_545461

-- Mathlib is imported to bring in the necessary library for number theory and divisors

open_locale big_operators

theorem smallest_good_number_exists :
  ∃ n : ℕ, (∀ d : ℕ, d ∣ n → d > 0) ∧
            (fintype.card (finset.filter (∣ n) (finset.range n.succ)) = 8) ∧
            (finset.sum (finset.filter (∣ n) (finset.range n.succ)) id = 3240) ∧ 
            (∀ m : ℕ, (∀ d : ℕ, d ∣ m → d > 0) ∧
                      (fintype.card (finset.filter (∣ m) (finset.range m.succ)) = 8) ∧
                      (finset.sum (finset.filter (∣ m) (finset.range m.succ)) id = 3240) → n ≤ m) :=
begin
  -- Proof goes here
  sorry
end

end smallest_good_number_exists_l545_545461


namespace never_sunday_l545_545477

theorem never_sunday (n : ℕ) (days_in_month : ℕ → ℕ) (is_leap_year : Bool) : 
  (∀ (month : ℕ), 1 ≤ month ∧ month ≤ 12 → (days_in_month month = 28 ∨ days_in_month month = 29 ∨ days_in_month month = 30 ∨ days_in_month month = 31) ∧
  (∃ (k : ℕ), k < 7 ∧ ∀ (d : ℕ), d < days_in_month month → (d % 7 = k ↔ n ≠ d))) → n = 31 := 
by
  sorry

end never_sunday_l545_545477


namespace min_value_x_plus_y_l545_545668

theorem min_value_x_plus_y (x y : ℤ) (det : 3 < x * y ∧ x * y < 5) : x + y = -5 :=
sorry

end min_value_x_plus_y_l545_545668


namespace parkway_girls_not_playing_soccer_l545_545728

theorem parkway_girls_not_playing_soccer (total_students boys soccer_students : ℕ) 
    (percent_boys_playing_soccer : ℕ) 
    (h1 : total_students = 420)
    (h2 : boys = 312)
    (h3 : soccer_students = 250)
    (h4 : percent_boys_playing_soccer = 86) :
   (total_students - boys - (soccer_students - soccer_students * percent_boys_playing_soccer / 100)) = 73 :=
by sorry

end parkway_girls_not_playing_soccer_l545_545728


namespace magic_shop_change_l545_545490

theorem magic_shop_change :
  (∀ (cloak : Type), ∃ price_gold price_silver1 change_gold1 price_silver2 change_gold2, 
  price_silver1 = 20 ∧ change_gold1 = 4 ∧ 
  price_silver2 = 15 ∧ change_gold2 = 1 ∧ 
  price_gold = 14 ∧ 
  ∀ change_silver, 
    (20 - 4) * change_silver = 15 - 1 → -- Relation derived from the conditions
    (14 - (15 - 1) * change_silver / (20 - 4)) * change_silver = 10) := 
sorry

end magic_shop_change_l545_545490


namespace fencing_cost_proof_l545_545637

theorem fencing_cost_proof (L : ℝ) (B : ℝ) (c : ℝ) (total_cost : ℝ)
  (hL : L = 60) (hL_B : L = B + 20) (hc : c = 26.50) : 
  total_cost = 5300 :=
by
  sorry

end fencing_cost_proof_l545_545637


namespace minimum_orders_l545_545263

theorem minimum_orders (total_items : ℕ) (item_price : ℕ) (discount : ℕ → ℕ) (extra_discount : ℕ → ℕ) 
  (required_discount_threshold : ℕ) (total_amount : ℕ) (order_1 : ℕ) (order_2 : ℕ): 
  total_items = 42 →
  item_price = 48 →
  (∀ x, discount x = (x * 6) / 10) → 
  (∀ x, x > required_discount_threshold → extra_discount x = 100) →
  required_discount_threshold = 300 →
  total_amount = 1209.6 → 
  9 + 11 + 11 + 11 = total_items →
  (discount (item_price * order_1) = 288) →
  (discount (item_price * order_2) = 316.8) →
  (count_extra_discount : ℕ) = 3 → 
  (minimum_numbers_orders : ℕ) = 4.

end minimum_orders_l545_545263


namespace triangle_area_l545_545549

theorem triangle_area (A B C D : Type*)
  [geometry : euclidean_geometry A B C D] :
  (angle A = 120) → (AD ⊥ AC) → (AD = 2) → (∃ area, area = 8 * sqrt 3 / 3) :=
by
  intro h1 h2 h3
  -- Definitions and conditions
  have cond_angle : angle A = 120 := h1
  have cond_perpendicular : AD ⊥ AC := h2
  have cond_length : AD = 2 := h3
  sorry

end triangle_area_l545_545549


namespace part1_part2_l545_545244

namespace Problem

def A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 5}
def B (m : ℝ) : Set ℝ := {x | m + 1 ≤ x ∧ x ≤ 2 * m - 1}

theorem part1 (m : ℝ) : (B m ⊆ A) → (m ≤ 3) :=
by
  intro h
  sorry

theorem part2 (m : ℝ) : (A ∩ B m = ∅) → (m < 2 ∨ 4 < m) :=
by
  intro h
  sorry

end Problem

end part1_part2_l545_545244


namespace externally_tangent_internally_tangent_common_chord_and_length_l545_545389

-- Definitions of Circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 6*y - 1 = 0
def circle2 (x y : ℝ) (m : ℝ) : Prop := x^2 + y^2 - 10*x - 12*y + m = 0

-- Proof problem 1: Externally tangent
theorem externally_tangent (m : ℝ) : (∃ x y : ℝ, circle1 x y ∧ circle2 x y m) → m = 25 + 10 * Real.sqrt 11 :=
sorry

-- Proof problem 2: Internally tangent
theorem internally_tangent (m : ℝ) : (∃ x y : ℝ, circle1 x y ∧ circle2 x y m) → m = 25 - 10 * Real.sqrt 11 :=
sorry

-- Proof problem 3: Common chord and length when m = 45
theorem common_chord_and_length :
  (∃ x y : ℝ, circle2 x y 45) →
  (∃ l : ℝ, l = 4 * Real.sqrt 7 ∧ ∀ x y : ℝ, (circle1 x y ∧ circle2 x y 45) → (4*x + 3*y - 23 = 0)) :=
sorry

end externally_tangent_internally_tangent_common_chord_and_length_l545_545389


namespace count_three_digit_integers_with_4_and_without_6_l545_545925

def is_three_digit_integer (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

def contains_digit (n : ℕ) (d : ℕ) : Prop :=
  nat.digits 10 n |> list.any (λ x => x = d)

def does_not_contain_digit (n : ℕ) (d : ℕ) : Prop :=
  nat.digits 10 n |> list.all (λ x => x ≠ d)

theorem count_three_digit_integers_with_4_and_without_6 : 
  (nat.card {n : ℕ // is_three_digit_integer n ∧ contains_digit n 4 ∧ does_not_contain_digit n 6} = 200) :=
by
  sorry

end count_three_digit_integers_with_4_and_without_6_l545_545925


namespace find_a_add_b_l545_545129

noncomputable def a : ℤ := 8
noncomputable def b : ℤ := 3

theorem find_a_add_b :
  (√9801 - 99 = (√a - b)^3) ∧ (a > 0 ∧ b > 0) → a + b = 11 :=
by
  intro h
  cases h with h1 h2
  cases h2 with ha hb
  sorry

end find_a_add_b_l545_545129


namespace largest_among_a_b_c_l545_545338

theorem largest_among_a_b_c (x : ℝ) (h0 : 0 < x) (h1 : x < 1)
  (a : ℝ := 2 * Real.sqrt x) 
  (b : ℝ := 1 + x) 
  (c : ℝ := 1 / (1 - x)) : c > b ∧ b > a := by
  sorry

end largest_among_a_b_c_l545_545338


namespace find_a_values_l545_545857

theorem find_a_values (a b : ℝ) (h1 : 0 < b) (h2 : b < a + 1)
  (h3 : ∃ (s : Set ℤ), ∀ x ∈ s, (x - b)^2 > (a * x)^2 ∧ s.card = 3) :
  a = 3/2 ∨ a = 5/2 :=
sorry

end find_a_values_l545_545857


namespace find_curved_surface_area_l545_545730

def slant_height : ℝ := 14
def radius : ℝ := 12
noncomputable def pi_approx : ℝ := 3.14159
noncomputable def curved_surface_area (r l : ℝ) : ℝ := π * r * l

theorem find_curved_surface_area :
  let CSA := curved_surface_area radius slant_height
  in Real.floor (CSA * 100) / 100 = 528.01 :=
by
  sorry

end find_curved_surface_area_l545_545730


namespace squares_per_student_l545_545855

theorem squares_per_student :
  (∀ (bar_squares : ℕ) (gerald_bars : ℕ) (multiplier : ℕ) (students : ℕ), 
    bar_squares = 8 → gerald_bars = 7 → multiplier = 2 → students = 24 → 
    (gerald_bars + gerald_bars * multiplier) * bar_squares / students = 7) :=
by
  intros bar_squares gerald_bars multiplier students
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  norm_num
  sorry

end squares_per_student_l545_545855


namespace Problem_l545_545330

theorem Problem (N : ℕ) (hn : N = 16) :
  (Nat.choose N 5) = 2002 := 
by 
  rw [hn] 
  sorry

end Problem_l545_545330


namespace largest_B_at_45_l545_545787

def binomial_coeff (n k : ℕ) : ℕ := Nat.choose n k

def B (k : ℕ) : ℝ :=
  if k ≤ 500 then (binomial_coeff 500 k) * (0.1)^k else 0

theorem largest_B_at_45 : ∀ k : ℕ, 0 ≤ k ∧ k ≤ 500 → B k ≤ B 45 :=
by
  intros k hk
  sorry

end largest_B_at_45_l545_545787


namespace count_four_digit_numbers_divisible_by_5_ending_in_45_l545_545404

theorem count_four_digit_numbers_divisible_by_5_ending_in_45 : 
  ∃ (n : ℕ), n = 90 ∧ ∀ (x : ℕ), (1000 ≤ x ∧ x < 10000) ∧ (x % 100 = 45) → count x = n :=
sorry

end count_four_digit_numbers_divisible_by_5_ending_in_45_l545_545404


namespace b_squared_gt_4ac_l545_545551

theorem b_squared_gt_4ac (a b c : ℝ) (h : (a + b + c) * c < 0) : b^2 > 4 * a * c :=
by
  sorry

end b_squared_gt_4ac_l545_545551


namespace hypotenuse_length_l545_545050

variable (a b c : ℝ)

-- Given conditions
theorem hypotenuse_length (h1 : b = 3 * a) 
                          (h2 : a^2 + b^2 + c^2 = 500) 
                          (h3 : c^2 = a^2 + b^2) : 
                          c = 5 * Real.sqrt 10 := 
by 
  sorry

end hypotenuse_length_l545_545050


namespace quadratic_expression_value_l545_545176

theorem quadratic_expression_value (x₁ x₂ : ℝ) (h₁ : x₁^2 - 3 * x₁ + 1 = 0) (h₂ : x₂^2 - 3 * x₂ + 1 = 0) :
  x₁^2 + 3 * x₂ + x₁ * x₂ - 2 = 7 :=
by
  sorry

end quadratic_expression_value_l545_545176


namespace stones_in_pyramid_l545_545762

theorem stones_in_pyramid : 
  let a := 10
  let l := 2
  let d := -2
  let n := 5
  let sum := n / 2 * (a + l)
  sum = 30 :=
by
  let a := 10
  let l := 2
  let d := -2
  let n := 5
  have h_term : l = a + (n-1) * d, by rfl
  have h_n : n = 5, by sorry
  have h_sum : sum = n / 2 * (a + l), by rfl
  have h_result : sum = 30, by sorry
  exact h_result
 
end stones_in_pyramid_l545_545762


namespace sum_of_roots_is_k_over_5_l545_545996

noncomputable def sum_of_roots 
  (x1 x2 k d : ℝ) 
  (hx : x1 ≠ x2) 
  (h1 : 5 * x1^2 - k * x1 = d) 
  (h2 : 5 * x2^2 - k * x2 = d) : ℝ :=
x1 + x2

theorem sum_of_roots_is_k_over_5 
  {x1 x2 k d : ℝ} 
  (hx : x1 ≠ x2) 
  (h1 : 5 * x1^2 - k * x1 = d) 
  (h2 : 5 * x2^2 - k * x2 = d) : 
  sum_of_roots x1 x2 k d hx h1 h2 = k / 5 :=
sorry

end sum_of_roots_is_k_over_5_l545_545996


namespace volume_of_cylinder_l545_545626

-- Define the conditions
def straw_length : ℝ := 12
def protrude_min : ℝ := 2
def protrude_max : ℝ := 4
def height : ℝ := 8
def radius : ℝ := 6
def pi_approx : ℝ := 3.14

-- Define the proof goal
theorem volume_of_cylinder :
  let V := pi_approx * radius^2 * height in
  V = 226.08 :=
by
  sorry

end volume_of_cylinder_l545_545626


namespace mass_of_segment_AB_l545_545287

/-- Given two points A(-2,1,0) and B(-1,3,5), and a proportionality constant k,
    prove that the mass of the material segment AB, where the density at each
    point M on the segment is proportional to the distance from M to A with
    the proportionality coefficient k, is 15k. -/
theorem mass_of_segment_AB (k : ℝ) : 
  let A := (-2 : ℝ, 1 : ℝ, 0 : ℝ)
      B := (-1 : ℝ, 3 : ℝ, 5 : ℝ)
  in mass_segment_AB A B k = 15 * k :=
by
  -- Assuming this is defined somewhere in the imported Mathlib, or would need to be defined.
  sorry

end mass_of_segment_AB_l545_545287


namespace length_of_A_l545_545081

-- Define points A, B, and C as given in the problem
def A : ℝ × ℝ := (0, 10)
def B : ℝ × ℝ := (0, 15)
def C : ℝ × ℝ := (3, 9)

-- Define conditions explicitly
def on_line_y_eq_x (P : ℝ × ℝ) : Prop := P.1 = P.2
def intersects_at (A A' : ℝ × ℝ) (B B' : ℝ × ℝ) (C : ℝ × ℝ) : Prop :=
  ∃ k l : ℝ, A'.1 = A.1 + k * (C.1 - A.1) ∧ A'.2 = A.2 + k * (C.2 - A.2) ∧
             B'.1 = B.1 + l * (C.1 - B.1) ∧ B'.2 = B.2 + l * (C.2 - B.2)

-- Statement of the proof problem
theorem length_of_A'B' :
  ∃ A' B' : ℝ × ℝ, on_line_y_eq_x A' ∧ on_line_y_eq_x B' ∧ 
  intersects_at A A' B B' C ∧ (dist A' B' = 5 * real.sqrt 2) := by
  sorry

end length_of_A_l545_545081


namespace distinct_ways_to_place_digits_l545_545458

theorem distinct_ways_to_place_digits :
  let n := 4 -- number of digits
  let k := 5 -- number of boxes
  (k * (n!)) = 120 := by
  sorry

end distinct_ways_to_place_digits_l545_545458


namespace count_valid_numbers_l545_545395

-- Define what it means to be a four-digit number that ends in 45
def is_valid_number (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999 ∧ n % 100 = 45 ∧ n % 5 = 0

-- Define the set of valid two-digit prefixes
def valid_prefixes : set ℕ := {ab | 10 ≤ ab ∧ ab ≤ 99}

-- Define the set of four-digit numbers that end in 45 and are divisible by 5
def valid_numbers : set ℕ := {n | ∃ ab : ℕ, ab ∈ valid_prefixes ∧ n = ab * 100 + 45}

-- State the theorem
theorem count_valid_numbers : (finset.card (finset.filter is_valid_number (finset.range 10000)) = 90) :=
sorry

end count_valid_numbers_l545_545395


namespace mgo_production_l545_545841

structure Reaction :=
  (initial_moles_Mg : ℕ)
  (initial_moles_CO2 : ℕ)

axiom balanced_reaction : ∀ (r : Reaction), (2 : ℕ) * r.initial_moles_Mg / 2 = (2 : ℕ) * (r.initial_moles_CO2 + 1) / 1

theorem mgo_production (r : Reaction) (H : r.initial_moles_Mg = 2) (H2 : r.initial_moles_CO2 = 1) : r.initial_moles_Mg = 2 → r.initial_moles_CO2 = 1 → 2 = 2 :=
by simp *; sorry

end mgo_production_l545_545841


namespace competition_results_l545_545251

variables (x : ℝ) (freq1 freq3 freq4 freq5 freq2 : ℝ)

/-- Axiom: Given frequencies of groups and total frequency, determine the total number of participants and the probability of an excellent score -/
theorem competition_results :
  freq1 = 0.30 ∧
  freq3 = 0.15 ∧
  freq4 = 0.10 ∧
  freq5 = 0.05 ∧
  freq2 = 40 / x ∧
  (freq1 + freq2 + freq3 + freq4 + freq5 = 1) ∧
  (x * freq2 = 40) →
  x = 100 ∧ (freq4 + freq5 = 0.15) := sorry

end competition_results_l545_545251


namespace minimize_J_l545_545944

def H (p q : ℝ) : ℝ := -4 * p * q + 7 * p * (1 - q) + 5 * (1 - p) * q - 6 * (1 - p) * (1 - q)

def J (p : ℝ) : ℝ := 
  max (H p 0) (H p 1)

theorem minimize_J (p : ℝ) (hp : 0 ≤ p ∧ p ≤ 1) : 
  (∀ q : ℝ, 0 ≤ q ∧ q ≤ 1 → H p q ≤ J p) ∧  
  (∃ p_min : ℝ, 0 ≤ p_min ∧ p_min ≤ 1 ∧ ∀ p' : ℝ, 0 ≤ p' ∧ p' ≤ 1 → J p_min ≤ J p') :=
begin
  sorry
end

end minimize_J_l545_545944


namespace center_of_symmetry_value_at_15π_over_4_infinite_series_sum_eq_pi_squared_over_8_l545_545087

def f (x : ℝ) : ℝ :=
  π / 2 - (4 / π) * (cos x + (cos (3 * x)) / (3 ^ 2) + ∑'_n (cos ((2*n-1) * x) / (2*n-1)^2))

theorem center_of_symmetry : ∃ x y : ℝ, x = π / 2 ∧ y = π / 2 ∧
  (∀ x : ℝ, f (π - x) + f x = π) :=
sorry

theorem value_at_15π_over_4 : f (15 * π / 4) = π / 4 :=
sorry

theorem infinite_series_sum_eq_pi_squared_over_8 : 
  1 + ∑'_n (1 / ((2*n-1)^2)) = π^2 / 8 :=
sorry

end center_of_symmetry_value_at_15π_over_4_infinite_series_sum_eq_pi_squared_over_8_l545_545087


namespace compute_modulo_l545_545814

theorem compute_modulo :
  (2023 * 2024 * 2025 * 2026) % 7 = 0 := by
  sorry

end compute_modulo_l545_545814


namespace largest_of_five_consecutive_even_integers_l545_545160

theorem largest_of_five_consecutive_even_integers :
  (∃ n, 2 + 4 + ... + 50 = 5 * n - 20) →
  (∃ n, ∑ i in range 25, 2 * (i + 1) = 5 * n - 20) →
  (∃ n, n = 134) :=
begin
  sorry
end

end largest_of_five_consecutive_even_integers_l545_545160


namespace domain_of_f_equiv_l545_545624

def domain_of_f (x : ℝ) : Prop :=
  1 + x ≥ 0 ∧ 1 - x ≠ 0

theorem domain_of_f_equiv (x : ℝ) :
  domain_of_f x ↔ (x ∈ set.Ici (-1) ∧ x ≠ 1) :=
by
    sorry

end domain_of_f_equiv_l545_545624


namespace mean_of_all_students_is_76_l545_545592

-- Definitions
def M : ℝ := 84
def A : ℝ := 70
def ratio_m_a : ℝ := 3 / 4

-- Theorem statement
theorem mean_of_all_students_is_76 (m a : ℝ) (hm : m = ratio_m_a * a) : 
  ((63 * a) + 70 * a) / ((3 / 4 * a) + a) = 76 := 
sorry

end mean_of_all_students_is_76_l545_545592


namespace count_desired_property_l545_545930

-- Define the property that a number is a three-digit positive integer
def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000

-- Define the property that a number contains at least one specific digit
def contains_digit (d : ℕ) (n : ℕ) : Prop :=
  list.any (nat.digits 10 n) (λ m, m = d)

-- Define the property that a number does not contain a specific digit
def does_not_contain_digit (d : ℕ) (n : ℕ) : Prop :=
  ¬ list.any (nat.digits 10 n) (λ m, m = d)

-- Define the overall property for the desired number
def desired_property (n : ℕ) : Prop :=
  is_three_digit n ∧ contains_digit 4 n ∧ does_not_contain_digit 6 n

-- State the theorem to prove the count of numbers with the desired property
theorem count_desired_property : finset.card (finset.filter desired_property (finset.range 1000)) = 200 := by
  sorry

end count_desired_property_l545_545930


namespace most_appropriate_sampling_method_l545_545748

theorem most_appropriate_sampling_method 
  (has_large_customer_base : Prop)
  (significant_differences_in_service_evaluations_among_age_groups : Prop)
  (available_sampling_methods : List String)
  (simple_random_sampling := available_sampling_methods.head)
  (stratified_sampling := available_sampling_methods.nth 1)
  (systematic_sampling := available_sampling_methods.nth 2)
  (available_sampling_methods = [
     "simple random sampling", 
     "stratified sampling", 
     "systematic sampling"
  ]) :
  stratified_sampling = "stratified sampling" :=
by
  -- Proof skipped
  sorry

end most_appropriate_sampling_method_l545_545748


namespace sum_of_x_for_median_eq_mean_l545_545715

theorem sum_of_x_for_median_eq_mean : 
  let numbers := [5, 7, 10, 21]
  list.sum (filter (λ x, 
    let sorted := list.sort (<=) (x :: numbers)
    let mean := (43 + x) / 5
    (sorted.nth 2 = some 7 ∧ mean = 7) ∨ 
    (sorted.nth 2 = some 10 ∧ mean = 10) ∨
    (sorted.nth 2 = some x ∧ mean = x)
  ) (-8::7::[])) = -1 :=
by sorry

end sum_of_x_for_median_eq_mean_l545_545715


namespace product_fraction_sequence_l545_545796

theorem product_fraction_sequence : (∏ n in Finset.range 1000, (n + 5) / (n + 4)) = 251 := by
  sorry

end product_fraction_sequence_l545_545796


namespace circle_O₁_equation_sum_of_squares_constant_l545_545354

-- Given conditions
def circle_O (x y : ℝ) := x^2 + y^2 = 25
def center_O₁ (m : ℝ) : ℝ × ℝ := (m, 0) 
def intersect_point := (3, 4)
def is_intersection (x y : ℝ) := circle_O x y ∧ (x - intersect_point.1)^2 + (y - intersect_point.2)^2 = 0
def line_passing_P (k : ℝ) (x y : ℝ) := y - intersect_point.2 = k * (x - intersect_point.1)
def point_on_circle (circle : ℝ × ℝ → Prop) (x y : ℝ) := circle (x, y)
def distance_squared (A B : ℝ × ℝ) := (A.1 - B.1)^2 + (A.2 - B.2)^2

-- Problem statements
theorem circle_O₁_equation (k : ℝ) (m : ℝ) (x y : ℝ) (h : k = 1) (h_intersect: is_intersection 3 4)
  (h_BP_distance : distance_squared (3, 4) (x, y) = (7 * Real.sqrt 2)^2) : 
  (x - 14)^2 + y^2 = 137 := sorry

theorem sum_of_squares_constant (k m : ℝ) (h : k ≠ 0) (h_perpendicular : line_passing_P (-1/k) 3 4)
  (A B C D : ℝ × ℝ) (h_AB_distance : distance_squared A B = 4 * m^2 / (1 + k^2)) 
  (h_CD_distance : distance_squared C D = 4 * m^2 * k^2 / (1 + k^2)) : 
  distance_squared A B + distance_squared C D = 4 * m^2 := sorry

end circle_O₁_equation_sum_of_squares_constant_l545_545354


namespace equilateral_triangle_circumcircle_MB_MC_MA_l545_545986

-- Definitions and Hypotheses
variables (A B C M : Type) 
variables [metric_space A] [metric_space B] [metric_space C] [metric_space M]
variables (ABC : triangle A B C) (is_equilateral : equilateral ABC)
variables (circumcircle : circle) 
variables (M_on_circumcircle : M ∈ circumcircle)
variables (arc_condition : M is_on_arc_between B C excluding A)

-- Final statement
theorem equilateral_triangle_circumcircle_MB_MC_MA
  (h : equilateral ABC)
  (h1: M ∈ circumcircle)
  (h2 : M is_on_arc_between B C excluding A) :
  distance M B + distance M C = distance M A :=
sorry

end equilateral_triangle_circumcircle_MB_MC_MA_l545_545986


namespace hyperbolas_same_asymptotes_M_l545_545630

theorem hyperbolas_same_asymptotes_M :
  ∃ M, (∀ x y: ℝ, x^2 / 9 - y^2 / 16 = 1 → 
              (y = 4 / 3 * x ∨ y = - 4 / 3 * x) ↔
              (y^2 / 25 - x^2 / M = 1 → 
               (y = 5 / real.sqrt M * x ∨ y = - 5 / real.sqrt M * x))) →
  M = 225 / 16 :=
by sorry

end hyperbolas_same_asymptotes_M_l545_545630


namespace circumcircles_tangent_l545_545562

noncomputable def midpoint (P Q : Point) : Point := sorry
noncomputable def intersection (line1 line2 : Line) : Point := sorry
noncomputable def on_circumcircle (P Q R : Point) : Prop := sorry
noncomputable def tangent_circles (circle1 circle2 : Circle) : Prop := sorry
noncomputable def circumcircle (A B C : Point) : Circle := sorry

theorem circumcircles_tangent
  (A B C D M N E F X Y : Point)
  (c : on_circumcircle A B C D)
  (m1 : M = midpoint A B)
  (m2 : N = midpoint C D)
  (e : E = intersection (line AC) (line BD))
  (f : F = intersection (line AB) (line CD))
  (x : X = intersection (line MN) (line BD))
  (y : Y = intersection (line MN) (line AC)) :
  tangent_circles (circumcircle E X Y) (circumcircle F M N) := 
sorry

end circumcircles_tangent_l545_545562


namespace factor_expression_l545_545836

-- Define the variables
variables (x : ℝ)

-- State the theorem to prove
theorem factor_expression : 3 * x * (x + 1) + 7 * (x + 1) = (3 * x + 7) * (x + 1) :=
by
  sorry

end factor_expression_l545_545836


namespace silver_coins_change_l545_545505

-- Define the conditions
def condition1 : ℕ × ℕ := (20, 4) -- (20 silver coins, 4 gold coins change)
def condition2 : ℕ × ℕ := (15, 1) -- (15 silver coins, 1 gold coin change)
def cost_of_cloak_in_gold_coins : ℕ := 14

-- Define the theorem to be proven
theorem silver_coins_change (s1 g1 s2 g2 cloak_g : ℕ) (h1 : (s1, g1) = condition1) (h2 : (s2, g2) = condition2) :
  ∃ silver : ℕ, (silver = 10) :=
by {
  sorry
}

end silver_coins_change_l545_545505


namespace find_a_from_function_property_l545_545650

theorem find_a_from_function_property {a : ℝ} (h : ∀ (x : ℝ), (0 ≤ x → x ≤ 1 → ax ≤ 3) ∧ (0 ≤ x → x ≤ 1 → ax ≥ 3)) :
  a = 3 :=
sorry

end find_a_from_function_property_l545_545650


namespace number_of_two_digit_multiples_of_seven_l545_545025

theorem number_of_two_digit_multiples_of_seven :
  let a := 14 in
  let l := 98 in
  let d := 7 in
  l - a >= 0 ∧ (l - a) % d = 0 →
  let n := (l - a) / d + 1 in
  n = 13 :=
by
  intros a l d h
  let n := (l - a) / d + 1
  have h1 : (l - a) / d + 1 = 13
  { sorry }
  exact h1

end number_of_two_digit_multiples_of_seven_l545_545025


namespace michael_and_truck_never_meet_l545_545587

-- Definitions based on the conditions
def michael_position (t : ℝ) : ℝ := 4 * t
def truck_position (t : ℝ) : ℝ :=
  let cycle_time := 250 / 12 + 40 in
  let cycles := ⌊t / cycle_time⌋ in
  let remainder := t % cycle_time in
  cycles * 250 + if remainder < 250 / 12 then 12 * remainder else 250

theorem michael_and_truck_never_meet :
  ∀ t > 0, michael_position t ≠ truck_position t :=
by
  -- The proof steps would go here
  sorry

end michael_and_truck_never_meet_l545_545587


namespace right_triangle_area_l545_545699

/-- Given a right triangle with one leg of length 30 inches and a hypotenuse of 34 inches,
    the area of the triangle is 240 square inches. -/
theorem right_triangle_area (a b c : ℝ) (h1 : a = 30) (h2 : c = 34) (h3 : a^2 + b^2 = c^2) :
  (1/2) * a * b = 240 :=
by
  rw [h1, h2] at h3
  have hb : b = 16 := by
    rw [←h3]
    norm_num
  rw [h1, hb]
  norm_num
  sorry

end right_triangle_area_l545_545699


namespace tablespoons_in_half_cup_l545_545179

theorem tablespoons_in_half_cup
    (grains_per_cup : ℕ)
    (half_cup : ℕ)
    (tbsp_to_tsp : ℕ)
    (grains_per_tsp : ℕ)
    (h1 : grains_per_cup = 480)
    (h2 : half_cup = grains_per_cup / 2)
    (h3 : tbsp_to_tsp = 3)
    (h4 : grains_per_tsp = 10) :
    (half_cup / (tbsp_to_tsp * grains_per_tsp) = 8) :=
by
  sorry

end tablespoons_in_half_cup_l545_545179


namespace rectangle_area_l545_545183

theorem rectangle_area (shorter_side : ℝ) (num_rectangles : ℕ) 
(h1 : shorter_side = 5) 
(h2 : num_rectangles = 3) :
  let longer_side := 2 * shorter_side,
      total_length := longer_side + shorter_side,
      width := longer_side,
      area := total_length * width
  in area = 150 := sorry

end rectangle_area_l545_545183


namespace equivalence_of_curves_l545_545883

variable {X Y : Type} [TopologicalSpace X] [TopologicalSpace Y]

-- Given condition
variable (C : Set (X × Y))
variable (f : X × Y → Prop)

axiom condition : ∀ (x y : X × Y), f (x, y) → (x, y) ∈ C

-- Proof goal
theorem equivalence_of_curves :
  ∀ (x y : X × Y), (x, y) ∉ C → ¬ f (x, y) := by
  sorry

end equivalence_of_curves_l545_545883


namespace cloak_change_in_silver_l545_545483

theorem cloak_change_in_silver :
  (∀ c : ℤ, (20 = c + 4) → (15 = c + 1)) →
  (5 * g = 3) →
  14 * gold / exchange_rate = 10 := 
sorry

end cloak_change_in_silver_l545_545483


namespace count_four_digit_numbers_divisible_by_5_and_ending_with_45_l545_545405

def is_four_digit_number (n : ℕ) : Prop := 1000 ≤ n ∧ n < 10000
def ends_with_45 (n : ℕ) : Prop := (n % 100) = 45 
def divisible_by_5 (n : ℕ) : Prop := (n % 5) = 0

theorem count_four_digit_numbers_divisible_by_5_and_ending_with_45 : 
  {n : ℕ | is_four_digit_number n ∧ ends_with_45 n ∧ divisible_by_5 n}.to_finset.card = 90 :=
sorry

end count_four_digit_numbers_divisible_by_5_and_ending_with_45_l545_545405


namespace mod_product_2023_2024_2025_2026_l545_545803

theorem mod_product_2023_2024_2025_2026 :
  (2023 * 2024 * 2025 * 2026) % 7 = 0 :=
by
  have h2023 : 2023 % 7 = 6 := by norm_num
  have h2024 : 2024 % 7 = 0 := by norm_num
  have h2025 : 2025 % 7 = 1 := by norm_num
  have h2026 : 2026 % 7 = 2 := by norm_num
  calc
    (2023 * 2024 * 2025 * 2026) % 7
      = ((2023 % 7) * (2024 % 7) * (2025 % 7) * (2026 % 7)) % 7 : by rw [Nat.mul_mod, Nat.mul_mod, Nat.mul_mod, Nat.mul_mod]
  ... = (6 * 0 * 1 * 2) % 7 : by rw [h2023, h2024, h2025, h2026]
  ... = 0 % 7 : by norm_num
  ... = 0 : by norm_num

end mod_product_2023_2024_2025_2026_l545_545803


namespace ratio_of_numbers_l545_545181

theorem ratio_of_numbers :
  ∃ (S L : ℕ), S = 28 ∧ L = S + 16 ∧ L / S = 11 / 7 :=
by
  -- Definitions corresponding to the conditions
  existsi 28, 44,
  split, exact rfl, -- S = 28
  split, norm_num, -- L = S + 16
  norm_num -- L / S = 11 / 7

end ratio_of_numbers_l545_545181


namespace find_right_triangles_with_leg_2012_l545_545524

theorem find_right_triangles_with_leg_2012 :
  ∃ x y z : ℕ, x^2 + 2012^2 = y^2 ∧ x + y + z = 2012 + 253005 + 253013 ∨
  x^2 + 2012^2 = z^2 ∧ x + y + z = 2012 + 506016 + 506020 ∨
  x^2 + 2012^2 = y^2 ∧ x + y + z = 2012 + 1012035 + 1012037 ∨
  x^2 + 2012^2 = y^2 ∧ x + y + z = 2012 + 1509 + 2515 ∧
  y ≠ 2012 :=
begin
  sorry  -- The actual proof is omitted as specified.
end

end find_right_triangles_with_leg_2012_l545_545524


namespace eccentricity_of_ellipse_l545_545134

theorem eccentricity_of_ellipse : 
  ∀ (a b c e : ℝ), a^2 = 16 → b^2 = 8 → c^2 = a^2 - b^2 → e = c / a → e = (Real.sqrt 2) / 2 := 
by 
  intros a b c e ha hb hc he
  sorry

end eccentricity_of_ellipse_l545_545134


namespace a_sufficient_but_not_necessary_l545_545736

theorem a_sufficient_but_not_necessary (a : ℝ) : 
  (a = 1 → |a| = 1) ∧ (¬ (|a| = 1 → a = 1)) :=
by 
  sorry

end a_sufficient_but_not_necessary_l545_545736


namespace product_of_random_numbers_greater_zero_l545_545202

noncomputable def random_product_positive_probability : ℝ := 
  let interval_length := 45
  let neg_interval_length := 30
  let pos_interval_length := 15
  let prob_neg := (neg_interval_length : ℝ) / interval_length
  let prob_pos := (pos_interval_length : ℝ) / interval_length
  prob_pos * prob_pos + prob_neg * prob_neg

-- Prove that the probability that the product of two randomly selected numbers
-- from the interval [-30, 15] is greater than zero is 5/9.
theorem product_of_random_numbers_greater_zero : 
  random_product_positive_probability = 5 / 9 := by
  sorry

end product_of_random_numbers_greater_zero_l545_545202


namespace possible_m_values_l545_545899

noncomputable def f (x m : ℝ) := x^2 - m * x + 1

theorem possible_m_values (m : ℝ) :
  (∀ x ∈ set.Icc (3 : ℝ) (8 : ℝ), deriv (f x m) x ≥ 0) ↔ (m ≤ 6 ∨ m ≥ 16) :=
begin
  sorry,
end

end possible_m_values_l545_545899


namespace max_sum_value_l545_545875

noncomputable def max_sum_n (a : ℕ → ℝ) (d : ℝ) : ℕ :=
  if a_1 >= 0 then 1 else 2 -- Fictitious function definition to define solution placeholder

theorem max_sum_value (a : ℕ → ℝ) (d : ℝ)
  (ar_seq : ∀ n, a (n + 1) = a n + d)
  (d_neg : d < 0)
  (S : ℕ → ℝ := λ n, (n + 1) * a 0 + (n * (n + 1) / 2) * d)
  (S16_pos : S 16 > 0)
  (S17_neg : S 17 < 0) :
  max_sum_n a d = 8 :=
by
  sorry

end max_sum_value_l545_545875


namespace problem_1_problem_2_l545_545007

noncomputable theory

open Real

-- Define the quadratic function g(x)
def g (x : ℝ) (a b : ℝ) := a * x^2 - 2 * a * x + b + 1

-- Problem statement
theorem problem_1 (h1 : ∀ x : ℝ, 2 ≤ x ∧ x ≤ 3 → ∀ a > 0, ∃ b, g(2, a, b) = 1 ∧ g(3, a, b) = 4):
  ∃ a b, g(x, 1, 0) = x^2 - 2 * x + 1 := 
begin
  sorry -- proof for part (1) here
end

-- Define function f(x)
def f (x : ℝ) := g(x, 1, 0) / x

-- Problem statement for part (2)
theorem problem_2 (h2 : ∀ x : ℝ, x ∈ Icc (1/27) (1/3) → 
  ∀ t = log 3 x, f(t) - k * t ≥ 0):
  ∃ k, k ∈ { k : ℝ | k ≤ 0 ∨ k ≥ 16 / 9 } :=
begin
  sorry -- proof for part (2) here
end

end problem_1_problem_2_l545_545007


namespace projections_straight_lines_not_implies_original_straight_line_l545_545643

-- Definitions and conditions
def SpatialFigure (ℝ^3 : Type*) := set (ℝ^3)

def projection_onto_plane (fig : SpatialFigure ℝ) (plane : set ℝ) : set ℝ :=
{p ∈ plane | ∃ q ∈ fig, p = q}

def is_straight_line (line : set ℝ) : Prop :=
∀ p1 p2 p3 ∈ line, (p2 - p1) / (p3 - p1) = (p2 - p1)

-- The problem statement
theorem projections_straight_lines_not_implies_original_straight_line
  (fig : SpatialFigure ℝ)
  (plane1 plane2 : set ℝ)
  (h_int : ∃ p ∈ plane1, p ∈ plane2)
  (h_proj1 : is_straight_line (projection_onto_plane fig plane1))
  (h_proj2 : is_straight_line (projection_onto_plane fig plane2)) :
  ¬ is_straight_line fig :=
sorry

end projections_straight_lines_not_implies_original_straight_line_l545_545643


namespace cyclists_meet_at_starting_point_l545_545234

/--
Given a circular track of length 1200 meters, and three cyclists with speeds of 36 kmph, 54 kmph, and 72 kmph,
prove that all three cyclists will meet at the starting point for the first time after 4 minutes.
-/
theorem cyclists_meet_at_starting_point :
  let track_length := 1200
  let speed_a_kmph := 36
  let speed_b_kmph := 54
  let speed_c_kmph := 72
  
  let speed_a_m_per_min := speed_a_kmph * 1000 / 60
  let speed_b_m_per_min := speed_b_kmph * 1000 / 60
  let speed_c_m_per_min := speed_c_kmph * 1000 / 60
  
  let time_a := track_length / speed_a_m_per_min
  let time_b := track_length / speed_b_m_per_min
  let time_c := track_length / speed_c_m_per_min
  
  let lcm := (2 : ℚ)

  (time_a = 2) ∧ (time_b = 4 / 3) ∧ (time_c = 1) → 
  ∀ t, t = lcm * 3 → t = 12 / 3 → t = 4 :=
by
  sorry

end cyclists_meet_at_starting_point_l545_545234


namespace count_positive_slope_lattice_points_l545_545732

open BigOperators

theorem count_positive_slope_lattice_points :
  let points := {P : Fin 7 × Fin 7 | true}
  ∃ (S : Finset (Fin 7 × Fin 7)), ∀ (i j : Fin S.card), P_iP_j ∈ S → i < j → 
  (S \in points) →
  ( ∑ m in Finset.range 8, Nat.choose 7 m ^ 2 = Nat.choose 14 7 ) := sorry

end count_positive_slope_lattice_points_l545_545732


namespace perpendicular_vectors_k_value_l545_545915

theorem perpendicular_vectors_k_value (k : ℝ) :
  let a := (1 / 2, k)
  let b := (k - 1, 4)
  let dot_product := a.1 * b.1 + a.2 * b.2
  dot_product = 0 → k = 1 / 9 :=
by
  have ab_perp := (1 / 2) * (k - 1) + k * 4
  assume h : ab_perp = 0
  sorry

end perpendicular_vectors_k_value_l545_545915


namespace ratio_of_inscribed_triangle_area_l545_545965

theorem ratio_of_inscribed_triangle_area (ABC A1 B1 C1 : Triangle) (α : ℝ) :
  ABC.is_isosceles → 
  ABC.angle_at_base = α → 
  ABC.is_inscribed A1 B1 C1 → 
  (ABC.area_ratio A1 B1 C1) = (cos α * (1 - cos α)) :=
by
  sorry

end ratio_of_inscribed_triangle_area_l545_545965


namespace matrix_eigenvalues_and_eigenvectors_l545_545864

open Matrix

variables {R : Type*} [Field R]

def matrix_A : Matrix (Fin 2) (Fin 2) R :=
  ![![1, 2], ![2, 1]]

def matrix_A_inv : Matrix (Fin 2) (Fin 2) R :=
  ![![(-1 : R) / 3, (2 : R) / 3], ![(2 : R) / 3, (-1 : R) / 3]]

theorem matrix_eigenvalues_and_eigenvectors :
  let A := matrix_A in
  let A_inv := matrix_A_inv in
  eigenvalues A = {3, -1} ∧ 
  eigenvector A 3 = ![1, 1] ∧ 
  eigenvector A (-1) = ![1, -1] ∧ 
  A.det ≠ 0 ∧ 
  A⁻¹ = A_inv :=
by
  -- elaborate the proof, which is not required
  sorry

end matrix_eigenvalues_and_eigenvectors_l545_545864


namespace cistern_fill_time_l545_545196

-- Define the problem conditions
def pipe_p_fill_time : ℕ := 10
def pipe_q_fill_time : ℕ := 15
def joint_filling_time : ℕ := 2
def remaining_fill_time : ℕ := 10 -- This is the answer we need to prove

-- Prove that the remaining fill time is equal to 10 minutes
theorem cistern_fill_time :
  (joint_filling_time * (1 / pipe_p_fill_time + 1 / pipe_q_fill_time) + (remaining_fill_time / pipe_q_fill_time)) = 1 :=
sorry

end cistern_fill_time_l545_545196


namespace right_triangle_area_l545_545673

theorem right_triangle_area (leg1 leg2 hypotenuse : ℕ) (h_leg1 : leg1 = 30)
  (h_hypotenuse : hypotenuse = 34)
  (hypotenuse_sq : hypotenuse * hypotenuse = leg1 * leg1 + leg2 * leg2) :
  (1 / 2 : ℚ) * leg1 * leg2 = 240 := by
  sorry

end right_triangle_area_l545_545673


namespace matrix_addition_is_correct_l545_545326

-- Definitions of matrices A and B according to given conditions
def A : Matrix (Fin 4) (Fin 4) ℤ :=  
  ![![ 3,  0,  1,  4],
    ![ 1,  2,  0,  0],
    ![ 5, -3,  2,  1],
    ![ 0,  0, -1,  3]]

def B : Matrix (Fin 4) (Fin 4) ℤ :=
  ![![-5, -7,  3,  2],
    ![ 4, -9,  5, -2],
    ![ 8,  2, -3,  0],
    ![ 1,  1, -2, -4]]

-- The expected result matrix from the addition of A and B
def C : Matrix (Fin 4) (Fin 4) ℤ :=
  ![![-2, -7,  4,  6],
    ![ 5, -7,  5, -2],
    ![13, -1, -1,  1],
    ![ 1,  1, -3, -1]]

-- The statement that A + B equals C
theorem matrix_addition_is_correct : A + B = C :=
by 
  -- Here we would provide the proof steps.
  sorry

end matrix_addition_is_correct_l545_545326


namespace length_A_l545_545084

open Real

noncomputable def point := ℝ × ℝ

def A : point := (0, 10)
def B : point := (0, 15)
def C : point := (3, 9)
def is_on_line_y_eq_x (P : point) : Prop := P.1 = P.2
def length (P Q : point) := sqrt ((Q.1 - P.1)^2 + (Q.2 - P.2)^2)

theorem length_A'B'_is_correct (A' B' : point)
  (hA' : is_on_line_y_eq_x A')
  (hB' : is_on_line_y_eq_x B')
  (hA'_line : ∃ m b, C.2 = m * C.1 + b ∧ A'.2 = m * A'.1 + b ∧ A.2 = m * A.1 + b)
  (hB'_line : ∃ m b, C.2 = m * C.1 + b ∧ B'.2 = m * B'.1 + b ∧ B.2 = m * B.1 + b)
  : length A' B' = 2.5 * sqrt 2 :=
sorry

end length_A_l545_545084


namespace distinct_ways_to_place_digits_l545_545457

theorem distinct_ways_to_place_digits :
  let n := 4 -- number of digits
  let k := 5 -- number of boxes
  (k * (n!)) = 120 := by
  sorry

end distinct_ways_to_place_digits_l545_545457


namespace distinct_ways_to_place_digits_l545_545453

theorem distinct_ways_to_place_digits :
  let digits := {1, 2, 3, 4}
  let boxes := 5
  let empty_box := 1
  -- There are 5! permutations of the list [0, 1, 2, 3, 4]
  let total_digits := insert 0 digits
  -- Resulting in 120 ways to place these digits in 5 boxes
  nat.factorial boxes = 120 :=
by 
  sorry

end distinct_ways_to_place_digits_l545_545453


namespace volume_parallelepiped_l545_545441

variables (a b : ℝ^3)
variables (ha : ∥a∥ = 1) (hb : ∥b∥ = 1)
variables (angle_ab : real.angle a b = π / 4)

theorem volume_parallelepiped : 
  abs (a • ((a + b × a) × b)) = 1 / 2 :=
sorry

end volume_parallelepiped_l545_545441


namespace angle_of_inclination_l545_545036

theorem angle_of_inclination (A B : ℝ × ℝ) (hA : A = (2, 5)) (hB : B = (4, 3)) : 
  ∃ θ : ℝ, θ = (3 * Real.pi) / 4 ∧ (∃ k : ℝ, k = (A.2 - B.2) / (A.1 - B.1) ∧ Real.tan θ = k) :=
by
  sorry

end angle_of_inclination_l545_545036


namespace mass_of_plate_l545_545321

noncomputable def surface_density (x y : ℝ) : ℝ := x^2 / (x^2 + y^2)

def region_D (x y : ℝ) : Prop :=
  (y^2 - 4*y + x^2 = 0 ∨ y^2 - 8*y + x^2 = 0) ∧ 
  (y ≤ x / sqrt 3 ∧ x ≥ 0)

theorem mass_of_plate :
  ∫∫ (x y : ℝ) in (set_of (λ x y, region_D x y)), surface_density x y = π + 3*sqrt 3 / 8 := 
by
  sorry

end mass_of_plate_l545_545321


namespace right_triangle_area_l545_545688

theorem right_triangle_area (a b c : ℝ) (h1 : a = 30) (h2 : c = 34) (h3 : a^2 + b^2 = c^2) :
  (1 / 2) * a * b = 240 :=
by
  sorry

end right_triangle_area_l545_545688


namespace count_four_digit_numbers_divisible_by_5_ending_in_45_l545_545419

theorem count_four_digit_numbers_divisible_by_5_ending_in_45 : 
  ∃ n : ℕ, (∀ x : ℕ, x ≥ 1000 ∧ x < 10000 ∧ x % 100 = 45 → (x % 5 = 0 ∧ x % 100 = 45)) 
  ∧ (n = 90) :=
by
  sorry

end count_four_digit_numbers_divisible_by_5_ending_in_45_l545_545419


namespace cloak_change_14_gold_coins_l545_545516

def exchange_rate (silver gold : ℕ) : Prop :=
  ∃ c : ℕ, (20 - 4) * c = silver ∧ (15 - 1) * c = silver

def cloak_purchase (paid_gold received_silver : ℕ) : Prop :=
  let exchange_rate := (5 * 14) / 3 in
  received_silver = 2 * exchange_rate

theorem cloak_change_14_gold_coins :
  exchange_rate 16 3 →
  exchange_rate 14 1 →
  cloak_purchase 14 10 := sorry

end cloak_change_14_gold_coins_l545_545516


namespace triangle_lines_concur_l545_545536

noncomputable def acute_triangle (A B C : Type) (ABC : Triangle A B C) : Prop :=
  ∀ (P : Point A),
    isAltitude C P A B → 
    (let H := footOfAltitude C ABC in
     let K := intersection (altitude A P B) (parallelLine H BC) in
     isAngleBisector (B K A))

theorem triangle_lines_concur
  (A B C : Type) (ABC : acute_triangle A B C) 
  (H : footOfAltitude C ABC)
  (AH_eq_BC : length (A, H) = length (B, C)) :
  ∃ (K : Type), isIntersectionPoint (angleBisector B) (altitude A B) (parallelLine H BC) K :=
begin
  sorry
end

end triangle_lines_concur_l545_545536


namespace part1_part2_l545_545909

theorem part1 :
  ∀ m, (∀ x, -1 ≤ x ∧ x ≤ 1 → x^2 - x - m < 0) ↔ (m > 2) := 
by sorry

theorem part2 (B : set ℝ) (hb : B = {x | x > 2}) :
  ∀ a, a < 1 →
    (∀ x, (3 * a < x ∧ x < a + 2) → x ∈ B) →
    (a ≥ 2 / 3 ∧ a < 1) :=
by sorry

end part1_part2_l545_545909


namespace area_of_right_triangle_l545_545701

-- Given definitions
def leg_a : ℝ := 30
def hypotenuse_c : ℝ := 34

-- The theorem statement
theorem area_of_right_triangle : 
  ∀ (b : ℝ), b = real.sqrt (hypotenuse_c^2 - leg_a^2) → 
  let area := 1 / 2 * leg_a * b in
  area = 240 := 
by
  intro b
  intro h
  let area := 1 / 2 * leg_a * b
  sorry

end area_of_right_triangle_l545_545701


namespace locus_of_Q_is_ellipse_l545_545376

-- Definitions

def ellipse_C (x y : ℝ) : Prop := (x^2)/24 + (y^2)/16 = 1
def line_l (x y : ℝ) : Prop := (x/12) + (y/8) = 1
def O : ℝ × ℝ := (0, 0)

-- Points P, Q, R
variables (P Q R : ℝ × ℝ)

-- Definitions based on conditions
def on_line_l (P : ℝ × ℝ) : Prop := line_l P.1 P.2
def on_ellipse_C (R : ℝ × ℝ) : Prop := ellipse_C R.1 R.2
def on_ray_OP (Q : ℝ × ℝ) (P : ℝ × ℝ) : Prop := ∃ k : ℝ, k > 0 ∧ Q = (k * P.1, k * P.2)
def distance (A B : ℝ × ℝ) : ℝ := real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

-- Condition relating distances
def distances_condition (O P Q R : ℝ × ℝ) : Prop :=
  distance O Q * distance O P = distance O R ^ 2

-- Locus Equation
def locus_eq (x y : ℝ) : Prop := (x - 1)^2 / (5/2) + (y - 1)^2 / (5/3) = 1

-- Mathematical Equivalent Proof Problem in Lean 4
theorem locus_of_Q_is_ellipse :
  (∃ P : ℝ × ℝ, on_line_l P ∧ ∃ Q R : ℝ × ℝ,
    on_ray_OP Q P ∧
    on_ellipse_C R ∧
    distances_condition O P Q R) →
  locus_eq Q.1 Q.2 :=
sorry

end locus_of_Q_is_ellipse_l545_545376


namespace id_number_2520th_citizen_l545_545064

-- Define a 7-digit number as a permutation of digits 1 through 7
def is_permutation_of_1_to_7 (l : List ℕ) : Prop :=
  l.permutation [1, 2, 3, 4, 5, 6, 7]

-- Define the function to find the 2520th permutation in lexicographical order
noncomputable def find_2520th_permutation : List ℕ :=
  (List.permutations [1, 2, 3, 4, 5, 6, 7]).nthLe 2519 sorry

-- Main theorem statement
theorem id_number_2520th_citizen : 
  (find_2520th_permutation = [4, 3, 7, 6, 5, 2, 1]) :=
sorry

end id_number_2520th_citizen_l545_545064


namespace income_max_takehome_pay_l545_545476

theorem income_max_takehome_pay :
  ∃ x : ℝ, (∀ y : ℝ, 1000 * y - 5 * y^2 ≤ 1000 * x - 5 * x^2) ∧ x = 100 :=
by
  sorry

end income_max_takehome_pay_l545_545476


namespace distinct_hyperbolas_l545_545331

def binomial (m n : ℕ) : ℕ := Nat.factorial m / (Nat.factorial n * Nat.factorial (m - n))

theorem distinct_hyperbolas :
  (∃ s: Finset ℕ, s = {binomial m n | 1 ≤ n ∧ n ≤ m ∧ m ≤ 5 ∧ 1 < binomial m n} ∧ s.card = 6) :=
sorry

end distinct_hyperbolas_l545_545331


namespace surface_area_ratio_l545_545726

theorem surface_area_ratio (x : ℝ) (hx : x > 0) :
  let SA1 := 6 * (4 * x) ^ 2
  let SA2 := 6 * x ^ 2
  (SA1 / SA2) = 16 := by
  sorry

end surface_area_ratio_l545_545726


namespace count_desired_property_l545_545928

-- Define the property that a number is a three-digit positive integer
def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000

-- Define the property that a number contains at least one specific digit
def contains_digit (d : ℕ) (n : ℕ) : Prop :=
  list.any (nat.digits 10 n) (λ m, m = d)

-- Define the property that a number does not contain a specific digit
def does_not_contain_digit (d : ℕ) (n : ℕ) : Prop :=
  ¬ list.any (nat.digits 10 n) (λ m, m = d)

-- Define the overall property for the desired number
def desired_property (n : ℕ) : Prop :=
  is_three_digit n ∧ contains_digit 4 n ∧ does_not_contain_digit 6 n

-- State the theorem to prove the count of numbers with the desired property
theorem count_desired_property : finset.card (finset.filter desired_property (finset.range 1000)) = 200 := by
  sorry

end count_desired_property_l545_545928


namespace song_distribution_l545_545279

-- Definitions for the problem

-- Define the sets for songs liked by pairs and individuals
def AB : Set ℕ := {s | ∀ (songs : Finset ℕ), s ∈ songs ∧ ¬∃ j, j ∈ songs \ {s}}
def BC : Set ℕ := {s | ∀ (songs : Finset ℕ), s ∈ songs ∧ ¬∃ a, a ∈ songs \ {s}}
def CA : Set ℕ := {s | ∀ (songs : Finset ℕ), s ∈ songs ∧ ¬∃ b, b ∈ songs \ {s}}
def A : Set ℕ := {s | ∀ (songs : Finset ℕ), s ∈ songs \ {s}}
def B : Set ℕ := {s | ∀ (songs : Finset ℕ), s ∈ songs \ {s}}
def C : Set ℕ := {s | ∀ (songs : Finset ℕ), s ∈ songs \ {s}}

-- Main statement
theorem song_distribution : α ∈ (Finset ℕ) → AB α ∈ {s | ∀ j ∈ α, j ≠ α } ∧ 
                            BC α ∈ {s | ∀ j ∈ α, j ≠ α } ∧ 
                            CA α ∈ {s | ∀ j ∈ α, j ≠ α } ∧ 
                            (A ∪ B ∪ C).card = 1 → 
                            (AB ∪ BC ∪ CA ∪ A ∪ B ∪ C).card = 5 → 
                            (number_of_ways α AB BC CA A B C) = 300
:= sorry

end song_distribution_l545_545279


namespace problem_part1_problem_part2_l545_545003

noncomputable def A := {x : ℝ | -1 / 2 ≤ x ∧ x ≤ 2}

noncomputable def f (x : ℝ) : ℝ := log ((2 * x^2 - 2 * x + 2) / (x^2 + 1))

theorem problem_part1 :
  (∃ b c : ℝ, ∀ x : ℝ, (x ∈ A) ↔ (b * x^2 + 3 * x + c ≥ 0)) → b = -2 ∧ c = 2 :=
begin
  sorry
end

theorem problem_part2 (b c : ℝ) (h : b = -2 ∧ c = 2) :
  ∃ M : set ℝ, M = { m : ℝ | 0 ≤ m ∧ m ≤ log (14 / 5) } :=
begin
  have ha : ∀ x : ℝ, x ∈ A → f(x) = log ((2 * x^2 - 2 * x + 2) / (x^2 + 1)), {
    assume x hx,
    unfold f,
  },
  use { m : ℝ | 0 ≤ m ∧ m ≤ log (14 / 5) },
  sorry
end

end problem_part1_problem_part2_l545_545003


namespace number_of_correct_analogies_l545_545278

theorem number_of_correct_analogies :
  let analogy1 := (focal_length_hyperbola_twice_real_axis :
                   ∀ {a c : ℝ}, 2 * c = 2 * a → 2) = -- This is a condition
                  (focal_length_ellipse_half_major_axis :
                   ∀ {a c : ℝ}, 2 * c = 1 / 2 * 2 * a → 1 / 2)
              .mpl := -- Given that analogy1 holds
  let analogy2 := (sum_of_first_3_terms_arith_seq :
                   ∀ {a1 a2 a3 : ℝ}, a1 + a2 + a3 = 1 →
                    a2 = 1 / 3) = -- This is a condition
                  (product_of_first_3_terms_geom_seq :
                   ∀ {a1 a2 a3 : ℝ}, a1 * a2 * a3 = 1 →
                    a2 = 1)
              .mpl := -- Given that analogy2 holds
  let analogy3 := (ratio_of_sides_equilateral_triangles :
                   ∀ {a b : ℝ}, a / b = 1 / 2 →
                    a^2 / b^2 = 1 / 4) = -- This is a condition
                  (ratio_of_edges_tetrahedrons :
                   ∀ {a b : ℝ}, a / b = 1 / 2 →
                    a^3 / b^3 = 1 / 8)
              .mpl := -- Given that analogy3 holds
  analogy1 ∧ analogy2 ∧ analogy3 → 3 = 3 := -- Prove that all analogies are correct

by {
  sorry
}

end number_of_correct_analogies_l545_545278


namespace find_FC_l545_545347

noncomputable def DC : ℝ := 12
noncomputable def CB : ℝ := 9
noncomputable def AD : ℝ := 31.5
noncomputable def AB : ℝ := 1/3 * AD
noncomputable def ED : ℝ := 3/4 * AD
noncomputable def CA : ℝ := CB + AB

theorem find_FC (DC_eq : DC = 12) (CB_eq : CB = 9) (AB_eq : AB = 1/3 * AD) 
                (ED_eq : ED = 3/4 * AD) (AD_eq : AD = 31.5) : 
  let CA := CB + AB in
  FC = (ED * CA) / AD := 
  sorry

end find_FC_l545_545347


namespace true_statements_l545_545184

theorem true_statements (a b m : ℝ) (a_pos : a > 0) (b_pos : b > 0)
  (h1 : ¬ (a = 0 ∧ b = 0) ↔ a^2 + b^2 ≠ 0)
  (h2 : m = 1/2 → ((m + 2) * (m + 2) + 3*m^2 - (m - 2) * 3 * m ≠ 0))
  (h3 : (1 * b = 2 * a) ∧ (sqrt (1 + (b^2 / a^2)) = sqrt 5)) :
  { n : ℕ // n = 2 ∧ (n = 3 ∨ n = 1) } :=
by
  have statement1 := h1
  have statement2 := (m = 1/2 ∨ m = -2) → (m + 2) * x + 3 * m * y_and_one  = 0
  have statement3 := h3
  sorry  -- Implementation of the proofs for these checks are left as exercise.

end true_statements_l545_545184


namespace smallest_multiple_of_3_l545_545152

theorem smallest_multiple_of_3 (a : ℕ) (h : ∀ i j : ℕ, i < 6 → j < 6 → 3 * (a + i) = 3 * (a + 10 + j) → a = 50) : 3 * a = 150 :=
by
  sorry

end smallest_multiple_of_3_l545_545152


namespace a4_value_l545_545465

variable (a : ℕ → ℤ)
variable (S : ℕ → ℤ)

-- Condition: The sum of the first n terms of the sequence {a_n} is S_n = n^2 - 1
axiom sum_of_sequence (n : ℕ) : S n = n^2 - 1

-- We need to prove that a_4 = 7
theorem a4_value : a 4 = S 4 - S 3 :=
by 
  -- Proof goes here
  sorry

end a4_value_l545_545465


namespace dad_sent_amount_l545_545798

-- Define the problem conditions
variables (D : ℝ) (current_balance original_balance total_sent : ℝ)

-- Assign the given values
def original_balance := 12
def current_balance := 87
def total_sent := current_balance - original_balance

-- Define the conditions based on the problem statement
def dad_sent := D
def mom_sent := 2 * D
def total_parent_sent := dad_sent + mom_sent

-- Lean statement to prove the amount sent by dad
theorem dad_sent_amount : total_sent = 75 → total_parent_sent = 75 → dad_sent = 25 :=
by
  intros ht1 ht2
  rw [total_sent, total_parent_sent, dad_sent, mom_sent] at ht1 ht2 -- substitute definitions
  sorry -- Placeholder for the actual proof

end dad_sent_amount_l545_545798


namespace count_valid_numbers_l545_545396

-- Define what it means to be a four-digit number that ends in 45
def is_valid_number (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999 ∧ n % 100 = 45 ∧ n % 5 = 0

-- Define the set of valid two-digit prefixes
def valid_prefixes : set ℕ := {ab | 10 ≤ ab ∧ ab ≤ 99}

-- Define the set of four-digit numbers that end in 45 and are divisible by 5
def valid_numbers : set ℕ := {n | ∃ ab : ℕ, ab ∈ valid_prefixes ∧ n = ab * 100 + 45}

-- State the theorem
theorem count_valid_numbers : (finset.card (finset.filter is_valid_number (finset.range 10000)) = 90) :=
sorry

end count_valid_numbers_l545_545396


namespace max_sum_clock_digits_l545_545751

theorem max_sum_clock_digits : ∃ t : ℕ, 0 ≤ t ∧ t < 24 ∧ 
  (∃ h1 h2 m1 m2 : ℕ, t = h1 * 10 + h2 + m1 * 10 + m2 ∧ 
   (0 ≤ h1 ∧ h1 ≤ 2) ∧ (0 ≤ h2 ∧ h2 ≤ 9) ∧ (0 ≤ m1 ∧ m1 ≤ 5) ∧ (0 ≤ m2 ∧ m2 ≤ 9) ∧ 
   h1 + h2 + m1 + m2 = 24) := sorry

end max_sum_clock_digits_l545_545751


namespace length_of_A_l545_545082

-- Define points A, B, and C as given in the problem
def A : ℝ × ℝ := (0, 10)
def B : ℝ × ℝ := (0, 15)
def C : ℝ × ℝ := (3, 9)

-- Define conditions explicitly
def on_line_y_eq_x (P : ℝ × ℝ) : Prop := P.1 = P.2
def intersects_at (A A' : ℝ × ℝ) (B B' : ℝ × ℝ) (C : ℝ × ℝ) : Prop :=
  ∃ k l : ℝ, A'.1 = A.1 + k * (C.1 - A.1) ∧ A'.2 = A.2 + k * (C.2 - A.2) ∧
             B'.1 = B.1 + l * (C.1 - B.1) ∧ B'.2 = B.2 + l * (C.2 - B.2)

-- Statement of the proof problem
theorem length_of_A'B' :
  ∃ A' B' : ℝ × ℝ, on_line_y_eq_x A' ∧ on_line_y_eq_x B' ∧ 
  intersects_at A A' B B' C ∧ (dist A' B' = 5 * real.sqrt 2) := by
  sorry

end length_of_A_l545_545082


namespace dot_product_theorem_l545_545017

open Real

namespace VectorProof

-- Define the vectors m and n
def m := (2, 5)
def n (t : ℝ) := (-5, t)

-- Define the condition that m is perpendicular to n
def perpendicular (t : ℝ) : Prop := (2 * -5) + (5 * t) = 0

-- Function to calculate the dot product
def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

-- Define the vectors m+n and m-2n
def vector_add (t : ℝ) : ℝ × ℝ := (m.1 + (n t).1, m.2 + (n t).2)
def vector_sub (t : ℝ) : ℝ × ℝ := (m.1 - 2 * (n t).1, m.2 - 2 * (n t).2)

-- The theorem to prove
theorem dot_product_theorem : ∀ (t : ℝ), perpendicular t → dot_product (vector_add t) (vector_sub t) = -29 :=
by
  intros t ht
  sorry

end VectorProof

end dot_product_theorem_l545_545017


namespace column_of_2023_is_C_l545_545821

/-- The sequence of columns repeats every 8 numbers as: [B, C, D, E, D, C, B, A] 
    We need to determine where the integer 2023 will be placed in this sequence. -/
def find_column (n : ℕ) : char :=
  let sequence : List char := ['B', 'C', 'D', 'E', 'D', 'C', 'B', 'A']
  sequence.get! ((n - 2) % 8)

theorem column_of_2023_is_C : find_column 2023 = 'C' :=
sorry

end column_of_2023_is_C_l545_545821


namespace count_integers_between_200_and_300_same_remainder_even_l545_545824

theorem count_integers_between_200_and_300_same_remainder_even :
  {n : ℤ | 200 < n ∧ n < 300 ∧ (∃ a b r : ℤ, n = 7 * a + r ∧ n = 9 * b + r ∧ even n)}.to_finset.card = 4 :=
by
  sorry

end count_integers_between_200_and_300_same_remainder_even_l545_545824


namespace normal_distribution_probability_l545_545100

theorem normal_distribution_probability
  (X : ℝ → ℝ) (μ : ℝ) (σ : ℝ) 
  (hX : ∀ x, X x ∼ Normal μ σ)
  (hμ : μ = 3)
  (hP4 : P (λ x, X x ≤ 4) = 0.84) :
  P (λ x, 2 < X x ∧ X x < 4) = 0.68 := 
sorry

end normal_distribution_probability_l545_545100


namespace point_in_fourth_quadrant_l545_545449

theorem point_in_fourth_quadrant (x y : ℝ) 
  (h_eq : y = sqrt (x-3) + sqrt (6-2*x) - 4)
  (h_x_ge_3 : x ≥ 3)
  (h_x_le_3 : x ≤ 3) : 
  x = 3 ∧ y = -4 ∧ (x > 0 ∧ y < 0) :=
by
  sorry

end point_in_fourth_quadrant_l545_545449


namespace find_sin_pairs_l545_545324

theorem find_sin_pairs : 
∃ n : ℕ, n = 63 ∧ 
  ∀ (m n : ℕ), 1 ≤ m ∧ m < n ∧ n ≤ 30 ∧ 
  ∃ x : ℝ, sin (m * x) + sin (n * x) = 2 → 
  true :=
begin
  sorry,
end

end find_sin_pairs_l545_545324


namespace linear_polynomials_exist_l545_545077

open Int

noncomputable def polynomial (x : ℤ) : Type := ℤ

def finds_linear_polynomial (P : polynomial ℤ) (seq : ℕ → ℤ) : Prop :=
∀ (a : ℕ → ℤ) (h_distinct : ∀ i j, i ≠ j → a i ≠ a j),
∃ (i j k : ℕ), i < j ∧ ∑ l in finset.range (j - i + 1), a (i + l) = P k

theorem linear_polynomials_exist {a b : ℤ} (P : polynomial ℤ) (seq : ℕ → ℤ) :
(finds_linear_polynomial (λ x, a * x + b) seq) ↔ (b % a = 0) :=
sorry

end linear_polynomials_exist_l545_545077


namespace find_d_value_l545_545096

/-- Let d be an odd prime number. If 89 - (d+3)^2 is the square of an integer, then d = 5. -/
theorem find_d_value (d : ℕ) (h₁ : Nat.Prime d) (h₂ : Odd d) (h₃ : ∃ m : ℤ, 89 - (d + 3)^2 = m^2) : d = 5 := 
by
  sorry

end find_d_value_l545_545096


namespace sum_x_coords_Q3_eq_2500_l545_545741

noncomputable def Q1_vertices {n : ℕ} (h : n = 120) : list ℝ := sorry

def sum_x_coords (vertices : list ℝ) : ℝ :=
  vertices.sum

theorem sum_x_coords_Q3_eq_2500 (vertices_Q1 : list ℝ) (h1 : vertices_Q1.length = 120)
  (h2 : sum_x_coords vertices_Q1 = 2500) : 
  sum_x_coords (map (λ v, (v + v)/2) vertices_Q1) = 2500 :=
by
  -- Proof needed
  sorry

end sum_x_coords_Q3_eq_2500_l545_545741


namespace angle_BQP_eq_angle_DAQ_l545_545622

-- Define the geometric context of the problem and the conditions
structure Trapezoid (α : Type*) :=
(A B C D P Q : α)
(BC_parallel_AD : BC IsParallel AD)
(diagonals_intersect_at_P : ∃ P, P ∈ AC ∧ P ∈ BD)
(PQ_separates_CD : separates CD P Q)
(angle_AQD_eq_angle_CQB : ∠AQD = ∠CQB)

open Trapezoid

-- The main theorem to be proven
theorem angle_BQP_eq_angle_DAQ {α : Type*} [EuclideanSpace α] (trapezoid : Trapezoid α) : 
  ∠BQP = ∠DAQ :=
by
  -- The proof goes here
  sorry

end angle_BQP_eq_angle_DAQ_l545_545622


namespace brittany_second_test_grade_l545_545793

theorem brittany_second_test_grade
  (first_test_grade second_test_grade : ℕ) 
  (average_after_second : ℕ)
  (h1 : first_test_grade = 78)
  (h2 : average_after_second = 81) 
  (h3 : (first_test_grade + second_test_grade) / 2 = average_after_second) :
  second_test_grade = 84 :=
by
  sorry

end brittany_second_test_grade_l545_545793


namespace acute_triangle_cosine_inequality_l545_545984

theorem acute_triangle_cosine_inequality (α β γ : ℝ) (hα : 0 < α ∧ α < π / 2)
(hβ : 0 < β ∧ β < π / 2) (hγ : 0 < γ ∧ γ < π / 2) (hαβγ : α + β + γ = π) :
  (cos (β - γ) / cos α) + (cos (γ - α) / cos β) + (cos (α - β) / cos γ) ≥ (3 / 2) :=
sorry

end acute_triangle_cosine_inequality_l545_545984


namespace area_of_triangle_BQW_l545_545973

-- Define the initial conditions

structure Rectangle where
  AB CD : ℝ
  AZ WC : ℝ
  ZW : ℝ

def area_of_trapezoid (rect : Rectangle) : ℝ := 
  1/2 * (rect.ZW + rect.CD) * (rect.AZ + rect.WC)

def midpoint (a b : ℝ × ℝ) : ℝ × ℝ := ((a.1 + b.1) / 2, (a.2 + b.2) / 2)

theorem area_of_triangle_BQW (rect: Rectangle) (AB_value CD_value: rect.AB = 16) (AZ_WC_value: rect.AZ = 8) (area_trap: area_of_trapezoid rect = 160) : 
    area_of_trapezoid rect / 2 = 96 :=
  by sorry

end area_of_triangle_BQW_l545_545973


namespace eventually_all_ones_l545_545867

theorem eventually_all_ones (N : ℕ) (k : ℕ) (hN : N = 2^k)
  (a : fin N → ℤ) (h_a : ∀ i, a i = 1 ∨ a i = -1) :
  ∃ m, ∀ i, (iterate (λ b, (λ j, b j * b ((j + 1) % N))) m a) i = 1 := 
  sorry

end eventually_all_ones_l545_545867


namespace count_valid_numbers_l545_545393

-- Define what it means to be a four-digit number that ends in 45
def is_valid_number (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999 ∧ n % 100 = 45 ∧ n % 5 = 0

-- Define the set of valid two-digit prefixes
def valid_prefixes : set ℕ := {ab | 10 ≤ ab ∧ ab ≤ 99}

-- Define the set of four-digit numbers that end in 45 and are divisible by 5
def valid_numbers : set ℕ := {n | ∃ ab : ℕ, ab ∈ valid_prefixes ∧ n = ab * 100 + 45}

-- State the theorem
theorem count_valid_numbers : (finset.card (finset.filter is_valid_number (finset.range 10000)) = 90) :=
sorry

end count_valid_numbers_l545_545393


namespace sum_of_edges_96_l545_545653

noncomputable def volume (a r : ℝ) : ℝ := 
  (a / r) * a * (a * r)

noncomputable def surface_area (a r : ℝ) : ℝ := 
  2 * ((a^2) / r + a^2 + a^2 * r)

noncomputable def sum_of_edges (a r : ℝ) : ℝ := 
  4 * ((a / r) + a + (a * r))

theorem sum_of_edges_96 :
  (∃ (a r : ℝ), volume a r = 512 ∧ surface_area a r = 384 ∧ sum_of_edges a r = 96) :=
by
  have a := 8
  have r := 1
  have h_volume : volume a r = 512 := sorry
  have h_surface_area : surface_area a r = 384 := sorry
  have h_sum_of_edges : sum_of_edges a r = 96 := sorry
  exact ⟨a, r, h_volume, h_surface_area, h_sum_of_edges⟩

end sum_of_edges_96_l545_545653


namespace max_books_l545_545956

theorem max_books (price_per_book available_money : ℕ) (h1 : price_per_book = 15) (h2 : available_money = 200) :
  ∃ n : ℕ, n = 13 ∧ n ≤ available_money / price_per_book :=
by {
  sorry
}

end max_books_l545_545956


namespace total_pages_in_book_l545_545942

theorem total_pages_in_book (x : ℕ) : 
  (∃ x, 
    let remaining_after_day_1 := (3/4) * x - 17 in
    let remaining_after_day_2 := (2/3) * remaining_after_day_1 - 20 in
    let remaining_after_day_3 := (1/2) * remaining_after_day_2 - 23 in
      remaining_after_day_3 = 70
  ) → x = 394 :=
by
  intro h
  cases h with x h
  sorry

end total_pages_in_book_l545_545942


namespace find_a_n_and_b_n_find_T_n_l545_545885

-- Definitions
def S (n : ℕ) (hn : 0 < n) := 2 * n^2 + n

def a (n : ℕ) (hn : 0 < n) := S n hn - S (n - 1) (by linarith)
def a₁ : ℕ := 3

def b (n : ℕ) (hn : 0 < n) : ℝ := 2^n

def T (n : ℕ) : ℝ := ∑ i in finset.range n, (a (i + 1) (by linarith [i+1]) * b (i + 1) (by linarith [i+1]))

-- Problems
theorem find_a_n_and_b_n (n : ℕ) (hn : 0 < n) : 
  (a n hn = 4 * n - 1) ∧ (b n hn = 2^n) := 
sorry

theorem find_T_n (n : ℕ) (hn : 0 < n) : 
  T n = (4 * n - 5) * 2^(n + 1) + 10 :=
sorry

end find_a_n_and_b_n_find_T_n_l545_545885


namespace smallest_coprime_gt_one_l545_545711

theorem smallest_coprime_gt_one :
  ∃ n : ℕ, n > 1 ∧ n < 14 ∧ gcd n 2310 = 1 := 
begin
  use 13,
  split,
  { exact dec_trivial, -- 13 > 1
  },
  split,
  { exact dec_trivial, -- 13 < 14
  },
  { exact dec_trivial, -- gcd(13, 2310) = 1
  }
end

end smallest_coprime_gt_one_l545_545711


namespace count_three_digit_numbers_with_4_no_6_l545_545935

def is_digit (n : ℕ) := (n >= 0) ∧ (n < 10)

def three_digit_integer (n : ℕ) := (n >= 100) ∧ (n <= 999)

def contains_digit (d : ℕ) (n : ℕ) : Prop :=
  let hd := n / 100 in
  let td := (n / 10) % 10 in
  let od := n % 10 in
  hd = d ∨ td = d ∨ od = d

def does_not_contain_digit (d : ℕ) (n : ℕ) : Prop :=
  let hd := n / 100 in
  let td := (n / 10) % 10 in
  let od := n % 10 in
  hd ≠ d ∧ td ≠ d ∧ od ≠ d

theorem count_three_digit_numbers_with_4_no_6 : 
  ∃ n, n = 200 ∧
  ∀ x, (three_digit_integer x) → (contains_digit 4 x) → (does_not_contain_digit 6 x) → 
  sorry

end count_three_digit_numbers_with_4_no_6_l545_545935


namespace probability_product_positive_is_5_div_9_l545_545199

noncomputable def probability_positive_product : ℚ :=
  let interval := Set.Icc (-30 : ℝ) 15
  let length_interval := 45
  let length_neg := 30
  let length_pos := 15
  let prob_neg := (length_neg : ℚ) / length_interval
  let prob_pos := (length_pos : ℚ) / length_interval
  let prob_product_pos := prob_neg^2 + prob_pos^2
  prob_product_pos

theorem probability_product_positive_is_5_div_9 :
  probability_positive_product = 5 / 9 :=
by
  sorry

end probability_product_positive_is_5_div_9_l545_545199


namespace floor_inequality_solution_set_l545_545088

/-- Let ⌊x⌋ denote the greatest integer less than or equal to x.
    Prove that the solution set of the inequality ⌊x⌋² - 5⌊x⌋ - 36 ≤ 0 is {x | -4 ≤ x < 10}. -/
theorem floor_inequality_solution_set (x : ℝ) :
  (⌊x⌋^2 - 5 * ⌊x⌋ - 36 ≤ 0) ↔ -4 ≤ x ∧ x < 10 := by
    sorry

end floor_inequality_solution_set_l545_545088


namespace count_four_digit_numbers_divisible_by_5_ending_in_45_l545_545401

theorem count_four_digit_numbers_divisible_by_5_ending_in_45 : 
  ∃ (n : ℕ), n = 90 ∧ ∀ (x : ℕ), (1000 ≤ x ∧ x < 10000) ∧ (x % 100 = 45) → count x = n :=
sorry

end count_four_digit_numbers_divisible_by_5_ending_in_45_l545_545401


namespace perpendicular_BD_AC_l545_545975

open EuclideanGeometry

variables (A B C O L D : Point)
variables (h1 : ∠ B = 60) (h2 : is_circumcenter O (triangle A B C))
          (h3 : angle_bisector B L (angle ABC))
          (h4 : second_intersection D (circumcircle B O L) (circumcircle A B C))

theorem perpendicular_BD_AC : BD ⊥ AC :=
by
  sorry

end perpendicular_BD_AC_l545_545975


namespace product_positive_probability_l545_545210

theorem product_positive_probability :
  let interval := set.Icc (-30 : ℝ) 15 in
  let prob_neg := (30 : ℝ) / 45 in
  let prob_pos := (15 : ℝ) / 45 in
  let prob_product_neg := 2 * (prob_neg * prob_pos) in
  let prob_product_pos := (prob_neg ^ 2) + (prob_pos ^ 2) in
  (prob_product_pos = 5 / 9) :=
by
  sorry

end product_positive_probability_l545_545210


namespace length_of_string_l545_545746

/-!
A circular cylindrical post with a circumference of 4 feet has a string wrapped around it,
spiraling from the bottom of the post to the top of the post. The string evenly loops
around the post exactly four full times, starting at the bottom edge and finishing at
the top edge. Given the height of the post is 12 feet, prove that the length of the string
is 20 feet.
-/

theorem length_of_string (circumference height : ℝ) (loops : ℕ) (h_circ : circumference = 4)
  (h_height : height = 12) (h_loops : loops = 4) : 
  let hyp_length := real.sqrt (height / loops) ^ 2 + circumference ^ 2  in
  let total_length := hyp_length * loops in
  total_length = 20 := 
by
  sorry

end length_of_string_l545_545746


namespace initial_bees_l545_545180

theorem initial_bees (B : ℕ) (h : B + 8 = 24) : B = 16 := 
by {
  sorry
}

end initial_bees_l545_545180


namespace calculate_percentage_l545_545669

noncomputable def total_investment : ℝ := 1000
noncomputable def invested_at_rate : ℝ := 199.99999999999983
noncomputable def remaining_investment : ℝ := total_investment - invested_at_rate
noncomputable def fixed_rate : ℝ := 0.05
noncomputable def total_with_interest : ℝ := 1046
noncomputable def interest_earned : ℝ := total_with_interest - total_investment

theorem calculate_percentage :
  ∃ (P : ℝ), invested_at_rate * P = interest_earned - (remaining_investment * fixed_rate) ∧ P = 0.03 :=
by {
  -- definitions for Lean compatibility
  have h_total_investment : total_investment = 1000 := rfl,
  have h_invested_at_rate : invested_at_rate = 199.99999999999983 := rfl,
  have h_remaining_investment : remaining_investment = total_investment - invested_at_rate := rfl,
  have h_fixed_rate : fixed_rate = 0.05 := rfl,
  have h_total_with_interest : total_with_interest = 1046 := rfl,
  have h_interest_earned : interest_earned = total_with_interest - total_investment := rfl,
  
  -- skip the proof part
  have solution : invested_at_rate * 0.03 = 46 - (remaining_investment * fixed_rate) := sorry,

  -- construct P
  use 0.03,
  exact ⟨solution, rfl⟩,
}

end calculate_percentage_l545_545669


namespace positive_integers_count_l545_545021

theorem positive_integers_count (n : ℕ) :
  (27 < n ∧ n < 150) → (nat.card {n : ℕ | 27 < n ∧ n < 150} = 122) :=
by
  sorry

end positive_integers_count_l545_545021


namespace distinct_products_in_S_l545_545076

theorem distinct_products_in_S (M : ℕ) (hM : M > 0) :
  let S := {n : ℕ | M^2 ≤ n ∧ n < (M + 1)^2} in
  ∀ a b c d ∈ S, a ≠ b → a * c ≠ b * d := by
  sorry

end distinct_products_in_S_l545_545076


namespace total_divisors_l545_545759

theorem total_divisors (a b c : ℕ) (α β γ : ℕ) :
  let N := a^α * b^β * c^γ in
  N.factors.count a = α ∧ N.factors.count b = β ∧ N.factors.count c = γ →
  (N.divisors.card = (α+1) * (β+1) * (γ+1)) ∧
  ∀ (d ∈ N.divisors), N / d ∈ N.divisors ∧ d * (N / d) = N ∧
  (N.divisors.product = N^((α+1) * (β+1) * (γ+1) / 2)) :=
by intros; sorry

end total_divisors_l545_545759


namespace probability_product_positive_is_5_div_9_l545_545200

noncomputable def probability_positive_product : ℚ :=
  let interval := Set.Icc (-30 : ℝ) 15
  let length_interval := 45
  let length_neg := 30
  let length_pos := 15
  let prob_neg := (length_neg : ℚ) / length_interval
  let prob_pos := (length_pos : ℚ) / length_interval
  let prob_product_pos := prob_neg^2 + prob_pos^2
  prob_product_pos

theorem probability_product_positive_is_5_div_9 :
  probability_positive_product = 5 / 9 :=
by
  sorry

end probability_product_positive_is_5_div_9_l545_545200


namespace equilateral_triangle_union_area_l545_545334

theorem equilateral_triangle_union_area :
  let s := 3 in
  let area_one_triangle := (sqrt 3 / 4) * s * s in
  let total_area_without_overlap := 4 * area_one_triangle in
  let overlap_side := s / 3 in
  let overlap_area_one := (sqrt 3 / 4) * overlap_side * overlap_side in
  let total_overlap := 3 * overlap_area_one in
  let net_area := total_area_without_overlap - total_overlap in
  net_area = 33 * sqrt 3 / 4 :=
by
  let s := 3
  let area_one_triangle := (sqrt 3 / 4) * s * s
  let total_area_without_overlap := 4 * area_one_triangle
  let overlap_side := s / 3
  let overlap_area_one := (sqrt 3 / 4) * overlap_side * overlap_side
  let total_overlap := 3 * overlap_area_one
  let net_area := total_area_without_overlap - total_overlap
  sorry

end equilateral_triangle_union_area_l545_545334


namespace line_c_intersects_a_or_b_l545_545912

-- Definitions of skew lines, planes, and intersection condition
variable (a b c : Line)
variable (α β : Plane)
variable (Line_intersection : Line → Plane → Plane → Line)

-- Conditions
variable (a_in_α : a ∈ α)
variable (b_in_β : b ∈ β)
variable (α_intersect_β : Line_intersection α β c)
variable (a_skew_b : ¬ Parallel a b)

-- The proof goal
theorem line_c_intersects_a_or_b :
  (c ∈ a) ∨ (c ∈ b) := sorry

end line_c_intersects_a_or_b_l545_545912


namespace roll_die_eight_times_l545_545769

def is_odd_prime (n : ℕ) : Prop :=
  n = 3 ∨ n = 5

def probability_odd_prime_product (num_rolls : ℕ) : ℚ :=
  if (∀ i : Fin num_rolls, is_odd_prime (i.val)) then (1/3) ^ num_rolls else 0

theorem roll_die_eight_times 
  (std_die_rolls : Vector ℕ 8)
  (h1 : ∀ (i : Fin 8), is_odd_prime (std_die_rolls.nth i))
: probability_odd_prime_product 8 = 1 / 6561 := by
  sorry

end roll_die_eight_times_l545_545769


namespace probability_product_positive_of_independent_selection_l545_545207

theorem probability_product_positive_of_independent_selection :
  let I := set.Icc (-30 : ℝ) (15 : ℝ)
  let P := (λ (x y : ℝ), x ∈ I ∧ y ∈ I ∧ x * y > 0)
  (Prob { x : ℝ × ℝ | P x.1 x.2 } :
    ProbabilitySpace (I × I)) = 5 / 9 :=
by
  sorry

end probability_product_positive_of_independent_selection_l545_545207


namespace base_area_of_hemisphere_l545_545167

theorem base_area_of_hemisphere (r : ℝ) (π : ℝ): 
  4 * π * r^2 = 4 * π * r^2 ∧ 3 * π * r^2 = 9 → 
  π * r^2 = 3 := 
by
  intros h
  cases h with sphere_surface_area hemisphere_surface_area
  -- adding some obvious statements
  have hyp3 : 3 * π * r^2 = 9 := hemisphere_surface_area
  have r_sq := (3 : ℝ) / π

  sorry

end base_area_of_hemisphere_l545_545167


namespace perfect_square_trinomial_implies_possible_m_values_l545_545947

theorem perfect_square_trinomial_implies_possible_m_values (m : ℝ) :
  (∃ a : ℝ, ∀ x : ℝ, (x - a)^2 = x^2 - 2*m*x + 16) → (m = 4 ∨ m = -4) :=
by
  sorry

end perfect_square_trinomial_implies_possible_m_values_l545_545947


namespace volume_ratio_of_tetrahedron_to_cube_l545_545360

noncomputable def volume_of_tetrahedron (s : ℝ) : ℝ :=
  (s^3 * real.sqrt 2) / 12

noncomputable def volume_of_cube (s : ℝ) : ℝ :=
  (s^3 * real.sqrt 2) / 4

theorem volume_ratio_of_tetrahedron_to_cube (s : ℝ) (hs : s > 0) :
  volume_of_tetrahedron s / volume_of_cube s = 1 / 3 :=
by
  sorry

end volume_ratio_of_tetrahedron_to_cube_l545_545360


namespace work_time_ratio_l545_545268

theorem work_time_ratio (TA TB : ℝ) (hB : TB = 24) (hCombined : 1 / TA + 1 / TB = 1 / 4) : TA / TB = 1 / 5 := by
  have hTA : TA = 4.8 := 
    calc
      1 / TA = 1 / 4 - 1 / 24 : by rw [hCombined, hB]
      ... = 6 / 24 - 1 / 24    : by norm_num
      ... = 5 / 24             : by ring
      ... = 1 / (24 / 5)       : by norm_num
    TA = 24 / 5                : by rw [inv_div, inv_inv, norm_num]
  TA / TB = (24 / 5) / 24      : by rw [hB]
  ... = 1 / 5                  : by field_simp [hB]

end work_time_ratio_l545_545268


namespace find_value_of_a_l545_545647

theorem find_value_of_a (a : ℝ) : 
  (let y := λ x : ℝ, a * x in  y 0 + y 1 = 3) ↔ a = 3 := 
by
  sorry

end find_value_of_a_l545_545647


namespace part1_part2_part3_l545_545896

noncomputable def f (a x : ℝ) : ℝ := log a ((x + 2) / (x - 2))

def g (a x : ℝ) : ℝ := f a (2 / x)

theorem part1 (a : ℝ) (h : a > 0 ∧ a ≠ 1) : 2 = 2 := by
  sorry

theorem part2 (a x1 x2 : ℝ) (h : a > 0 ∧ a ≠ 1 ∧ x1 ≠ 0 ∧ x2 ≠ 0 ∧ x1 ≠ x2 ∧ x1 + x2 ≠ 0 ∧
                (2 / x1 > 2 ∨ 2 / x1 < -2) ∧ (2 / x2 > 2 ∨ 2 / x2 < -2)) :
  g a x1 + g a x2 = g a ((x1 + x2) / (1 + x1 * x2)) := by
  sorry

theorem part3 (a r : ℝ) (h : a > 0 ∧ a ≠ 1 ∧ (∀ x, a - 4 < x ∧ x < r → f a x > 1)) :
  a - r = 16 / 5 ∨ a - r = (11 - real.sqrt 41) / 2 := by
  sorry

end part1_part2_part3_l545_545896


namespace polar_equation_C_max_OM_ON_l545_545535

-- Conditions for the problem
def parametric_line (t α : ℝ) (hα : 0 ≤ α ∧ α < π) : ℝ × ℝ :=
( t * Real.cos α , 2 + t * Real.sin α )

def parametric_curve (β : ℝ) : ℝ × ℝ :=
( 2 * Real.cos β , 2 + 2 * Real.sin β )

-- Part (1): Prove the polar equation of curve C
theorem polar_equation_C:
  ∀ (β : ℝ),
  ∃ (ρ θ : ℝ), (ρ^2 = 4 * ρ * Real.sin θ) ∧ (ρ = 4 * Real.sin θ) :=
by
  sorry

-- Part (2): Prove the maximum value of |OM| + |ON|
theorem max_OM_ON:
  ∀ (α : ℝ) (t : ℝ) (hα : 0 ≤ α ∧ α < π),
  let M : ℝ × ℝ := (t * Real.cos α, 2 + t * Real.sin α),
      N : ℝ × ℝ := (t * Real.cos (α + π/2), 2 + t * Real.sin (α + π/2))
  in
  let |OM| := (M.1^2 + M.2^2)^(1/2),
      |ON| := (N.1^2 + N.2^2)^(1/2)
  in
  (∀ (θ : ℝ), θ ∈ (0, π/2) →
   |OM| + |ON| ≤ 4 * Real.sqrt 2) :=
by
  sorry

end polar_equation_C_max_OM_ON_l545_545535


namespace cloak_change_14_gold_coins_l545_545511

def exchange_rate (silver gold : ℕ) : Prop :=
  ∃ c : ℕ, (20 - 4) * c = silver ∧ (15 - 1) * c = silver

def cloak_purchase (paid_gold received_silver : ℕ) : Prop :=
  let exchange_rate := (5 * 14) / 3 in
  received_silver = 2 * exchange_rate

theorem cloak_change_14_gold_coins :
  exchange_rate 16 3 →
  exchange_rate 14 1 →
  cloak_purchase 14 10 := sorry

end cloak_change_14_gold_coins_l545_545511


namespace quadratic_function_solution_l545_545842

theorem quadratic_function_solution {a b : ℝ} (f : ℝ → ℝ) (h : ∀ x, f x = x^2 + a * x + b) :
  (∀ x, (f (f(x) + 2*x)) / (f x) = x^2 + 2023 * x + 2040) → 
  (f = λ x, x^2 + 2021 * x + 1) :=
begin
  sorry
end

end quadratic_function_solution_l545_545842


namespace fewest_presses_l545_545248

def reciprocal (x : ℝ) : ℝ := 1 / x

theorem fewest_presses (x : ℝ) (h : x = 16) : 
  reciprocal (reciprocal x) = x ∧ reciprocal (reciprocal 16) = 16 :=
by
  sorry

end fewest_presses_l545_545248


namespace average_books_minimum_l545_545045

noncomputable def average_books_per_student (total_students : ℕ) (students_with_0_books : ℕ) (students_with_1_book : ℕ) (students_with_2_books : ℕ) : ℝ :=
  let students_with_min_3_books := total_students - (students_with_0_books + students_with_1_book + students_with_2_books)
  let total_books := (students_with_0_books * 0) + (students_with_1_book * 1) + (students_with_2_books * 2) + (students_with_min_3_books * 3)
  total_books / total_students

theorem average_books_minimum :
  average_books_per_student 25 3 11 6 = 1.52 :=
by simp [average_books_per_student, show 25 - (3 + 11 + 6) = 5 from rfl, show ((3 * 0) + (11 * 1) + (6 * 2) + (5 * 3): ℝ) / 25 = 1.52 from rfl]; norm_num

end average_books_minimum_l545_545045


namespace maximize_profit_at_x_l545_545744

-- Define cost of green tea per kilogram
def cost_per_kg : ℝ := 50

-- Define sales volume based on selling price
def sales_volume (x : ℝ) : ℝ := -2 * x + 240

-- Define profit function based on selling price
def profit (x : ℝ) : ℝ := (x - cost_per_kg) * sales_volume x

-- Define the relationship between y and x
def relationship_y_x (x : ℝ) : ℝ := -2 * x ^ 2 + 340 * x - 12000

-- Main theorem statement
theorem maximize_profit_at_x : ∀ x : ℝ, (profit x = relationship_y_x x) → (∃ c : ℝ, c = 85 ∧ ∀ x : ℝ, relationship_y_x x ≤ relationship_y_x c) :=
by
  intro x h
  use 85
  split
  case goal 1 
  {
    sorry
  }
  case goal 2 
  {
    intro x' hx
    sorry
  }

end maximize_profit_at_x_l545_545744


namespace count_four_digit_numbers_divisible_by_5_ending_in_45_l545_545402

theorem count_four_digit_numbers_divisible_by_5_ending_in_45 : 
  ∃ (n : ℕ), n = 90 ∧ ∀ (x : ℕ), (1000 ≤ x ∧ x < 10000) ∧ (x % 100 = 45) → count x = n :=
sorry

end count_four_digit_numbers_divisible_by_5_ending_in_45_l545_545402


namespace minimum_PQ_value_l545_545987

variables (A B C D N P Q : Point)
variables (ac bd : ℝ)
variables (angle_dan : ℝ)

-- Conditions
def rhombus_ABCD (A B C D : Point) :=
  rhombus A B C D

def diagonals_AC_BD (A C : Point) (ac : ℝ) := (dist A C = ac)
def diagonals_BD (B D : Point) (bd : ℝ) := (dist B D = bd)

def angle_DAN_30 (D A N : Point) (angle_dan : ℝ) := (angle_dan = 30)

def point_on_AB (N : Point) := 
  ∃ (A B : Point), between A N B

def feet_perpendiculars (N P Q : Point) (AC BD : Line) := 
  (perpendicular N P AC) ∧ (perpendicular N Q BD)

noncomputable def min_PQ (P Q : Point) := dist P Q

-- Problem statement
theorem minimum_PQ_value
  (h1: rhombus_ABCD A B C D)
  (h2: diagonals_AC_BD A C 20)
  (h3: diagonals_BD B D 24)
  (h4: angle_DAN_30 D A N 30)
  (h5: point_on_AB N)
  (h6: feet_perpendiculars N P Q
          (line_through A C) (line_through B D)) :
  min_PQ P Q = 2 * sqrt 61 := 
sorry

end minimum_PQ_value_l545_545987


namespace chord_length_when_m_is_one_equation_of_line_shortest_chord_l545_545971

noncomputable def circle_eq : (ℝ × ℝ) → Prop := λ p, let (x, y) := p in x^2 + y^2 - 8*x + 11 = 0 

noncomputable def line_eq (m : ℝ) : (ℝ × ℝ) → Prop := 
  λ p, let (x, y) := p in (2*m + 1)*x + (m + 1)*y - 7*m - 4 = 0

theorem chord_length_when_m_is_one :
  let l := sqrt (5 - (abs(3 * 4 - 11) / sqrt (3^2 + 2^2))^2) * 2 in
  l = 6 * sqrt(13) / 13 := sorry

theorem equation_of_line_shortest_chord :
  ∃ m : ℝ, m = -2/3 ∧ 
  (∀ p : ℝ × ℝ, line_eq m p = (p.1 - p.2 - 2 = 0)) := sorry

end chord_length_when_m_is_one_equation_of_line_shortest_chord_l545_545971


namespace count_four_digit_numbers_divisible_by_five_ending_45_l545_545426

-- Define the conditions as necessary in Lean
def is_four_digit_number (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

def is_divisible_by_five (n : ℕ) : Prop :=
  n % 5 = 0

def ends_with_45 (n : ℕ) : Prop :=
  n % 100 = 45

-- Statement that there exists 90 such four-digit numbers
theorem count_four_digit_numbers_divisible_by_five_ending_45 : 
  { n : ℕ // is_four_digit_number n ∧ is_divisible_by_five n ∧ ends_with_45 n }.card = 90 :=
sorry

end count_four_digit_numbers_divisible_by_five_ending_45_l545_545426


namespace convert_B5F_base_10_l545_545822

theorem convert_B5F_base_10 : 
  ∀ (B F : ℕ) (h₁: B = 11) (h₂: F = 15),
  B*16^2 + 5*16^1 + F*16^0 = 2911 := 
by
  intros B F h₁ h₂
  rw [h₁, h₂]
  norm_num
  sorry

end convert_B5F_base_10_l545_545822


namespace arithmetic_sequence_terms_before_one_l545_545436

theorem arithmetic_sequence_terms_before_one :
  ∀ (a d : ℤ), a = 100 → d = -3 → (∃ n: ℕ, 1 = a + d * (n - 1) ∧ (n - 1) = 33) :=
by
  intros a d ha hd
  use 34
  rw [ha, hd]
  split
  {
    -- Prove that 1 = 100 + (-3) * (34 - 1)
    calc
      1 = 103 - 3 * 34   : by sorry
  }
  {
    -- Prove that (34 - 1) = 33
    calc
      34 - 1 = 33 : by sorry
  }
  sorry

end arithmetic_sequence_terms_before_one_l545_545436


namespace distance_between_points_on_parabola_l545_545370

theorem distance_between_points_on_parabola :
  ∀ (x1 x2 y1 y2 : ℝ), 
    (y1^2 = 4 * x1) → (y2^2 = 4 * x2) → (x2 = x1 + 2) → (|y2 - y1| = 4 * Real.sqrt x2 - 4 * Real.sqrt x1) →
    Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2) = 8 :=
by
  intros x1 x2 y1 y2 h1 h2 h3 h4
  sorry

end distance_between_points_on_parabola_l545_545370


namespace f_characterization_l545_545314

noncomputable def f : ℕ → ℕ := sorry
noncomputable def g : ℕ → ℕ := sorry

axiom f_surjective : Function.Surjective f
axiom f_divisibility : ∀ (m n : ℕ), m ∣ n ↔ f(m) ∣ f(n)
axiom g_bijection : ∀ (p : ℕ), Nat.Prime p → Nat.Prime (g p) ∧ Function.Bijective g

theorem f_characterization (n : ℕ) : ∃ (p e : ℕ) (l : List (ℕ × ℕ)),
  (nat.factorization n).map (λ ⟨p, e⟩, (g p, e)) = l.map (λ ⟨p, e⟩, (f p, e)) :=
sorry

end f_characterization_l545_545314


namespace angle_sum_l545_545440

-- Definitions for the given angles and conditions
variable (A F G B D : Type) 

-- Condition 1: ∠A = 30°
def angle_A : Real := 30

-- Condition 2: ∠AFG = ∠AGF
def angle_AFG : Real
def angle_AGF : Real := angle_AFG

-- Theorem statement
theorem angle_sum (h1 : angle_A = 30) (h2 : angle_AFG = angle_AGF) : 
  let angle_BFD := 180 - angle_A - 2 * angle_AFG
  angle_B + angle_D = 75 := 
sorry

end angle_sum_l545_545440


namespace right_triangle_area_l545_545690

theorem right_triangle_area (a b c : ℝ) (h1 : a = 30) (h2 : c = 34) (h3 : a^2 + b^2 = c^2) :
  (1 / 2) * a * b = 240 :=
by
  sorry

end right_triangle_area_l545_545690


namespace grade_on_second_test_l545_545792

variable (first_test_grade second_test_average : ℕ)
#check first_test_grade
#check second_test_average

theorem grade_on_second_test :
  first_test_grade = 78 →
  second_test_average = 81 →
  (first_test_grade + (second_test_average * 2 - first_test_grade)) / 2 = second_test_average →
  second_test_grade = 84 :=
by
  intros h1 h2 h3
  sorry

end grade_on_second_test_l545_545792


namespace product_of_random_numbers_greater_zero_l545_545204

noncomputable def random_product_positive_probability : ℝ := 
  let interval_length := 45
  let neg_interval_length := 30
  let pos_interval_length := 15
  let prob_neg := (neg_interval_length : ℝ) / interval_length
  let prob_pos := (pos_interval_length : ℝ) / interval_length
  prob_pos * prob_pos + prob_neg * prob_neg

-- Prove that the probability that the product of two randomly selected numbers
-- from the interval [-30, 15] is greater than zero is 5/9.
theorem product_of_random_numbers_greater_zero : 
  random_product_positive_probability = 5 / 9 := by
  sorry

end product_of_random_numbers_greater_zero_l545_545204


namespace silver_coins_change_l545_545506

-- Define the conditions
def condition1 : ℕ × ℕ := (20, 4) -- (20 silver coins, 4 gold coins change)
def condition2 : ℕ × ℕ := (15, 1) -- (15 silver coins, 1 gold coin change)
def cost_of_cloak_in_gold_coins : ℕ := 14

-- Define the theorem to be proven
theorem silver_coins_change (s1 g1 s2 g2 cloak_g : ℕ) (h1 : (s1, g1) = condition1) (h2 : (s2, g2) = condition2) :
  ∃ silver : ℕ, (silver = 10) :=
by {
  sorry
}

end silver_coins_change_l545_545506


namespace prob_both_A_and_B_l545_545144

variable (A B : ∀ (Ω : Type) [MeasurableSpace Ω], @ProbabilityMeasure Ω)
variable (Ω : Type) [MeasurableSpace Ω] [ProbabilityMeasure Ω]

-- conditions given in the problem
variable (pA : Probability A = 0.25)
variable (pB : Probability B = 0.35)
variable (pNotAB : Probability (λ w, ¬(A w) ∧ ¬(B w)) = 0.55)

-- goal to prove
theorem prob_both_A_and_B : 
  Probability (λ w, (A w) ∧ (B w)) = 0.15 :=
by 
  sorry

end prob_both_A_and_B_l545_545144


namespace area_of_right_triangle_l545_545705

-- Given definitions
def leg_a : ℝ := 30
def hypotenuse_c : ℝ := 34

-- The theorem statement
theorem area_of_right_triangle : 
  ∀ (b : ℝ), b = real.sqrt (hypotenuse_c^2 - leg_a^2) → 
  let area := 1 / 2 * leg_a * b in
  area = 240 := 
by
  intro b
  intro h
  let area := 1 / 2 * leg_a * b
  sorry

end area_of_right_triangle_l545_545705


namespace max_distance_inner_outer_vertex_l545_545767

noncomputable def side_length_from_perimeter (p : ℝ) : ℝ := p / 4

def inner_square_perimeter : ℝ := 24
def outer_square_perimeter : ℝ := 36

def inner_square_side_length : ℝ := side_length_from_perimeter inner_square_perimeter
def outer_square_side_length : ℝ := side_length_from_perimeter outer_square_perimeter

def max_vertex_distance (inner_side : ℝ) (outer_side : ℝ) : ℝ := 
  let d := (outer_side - inner_side) / 2
  let max_dist := real.sqrt ((inner_side + d) * (inner_side + d) + d * d)
  max_dist

theorem max_distance_inner_outer_vertex :
  max_vertex_distance inner_square_side_length outer_square_side_length = 7.5 * real.sqrt 2 :=
by
  sorry

end max_distance_inner_outer_vertex_l545_545767


namespace eval_definite_integral_of_sin_l545_545834

open Real

theorem eval_definite_integral_of_sin : ∫ x in 0..1, sin x = 1 - cos 1 :=
by
  sorry

end eval_definite_integral_of_sin_l545_545834


namespace sum_of_possible_values_of_x_l545_545645

theorem sum_of_possible_values_of_x :
  let sq_side := (x - 4)
  let rect_length := (x - 5)
  let rect_width := (x + 6)
  let sq_area := (sq_side)^2
  let rect_area := rect_length * rect_width
  (3 * (sq_area) = rect_area) → ∃ (x1 x2 : ℝ), (3 * (x1 - 4) ^ 2 = (x1 - 5) * (x1 + 6)) ∧ (3 * (x2 - 4) ^ 2 = (x2 - 5) * (x2 + 6)) ∧ (x1 + x2 = 12.5) := 
by
  sorry

end sum_of_possible_values_of_x_l545_545645


namespace count_three_digit_numbers_with_4_no_6_l545_545936

def is_digit (n : ℕ) := (n >= 0) ∧ (n < 10)

def three_digit_integer (n : ℕ) := (n >= 100) ∧ (n <= 999)

def contains_digit (d : ℕ) (n : ℕ) : Prop :=
  let hd := n / 100 in
  let td := (n / 10) % 10 in
  let od := n % 10 in
  hd = d ∨ td = d ∨ od = d

def does_not_contain_digit (d : ℕ) (n : ℕ) : Prop :=
  let hd := n / 100 in
  let td := (n / 10) % 10 in
  let od := n % 10 in
  hd ≠ d ∧ td ≠ d ∧ od ≠ d

theorem count_three_digit_numbers_with_4_no_6 : 
  ∃ n, n = 200 ∧
  ∀ x, (three_digit_integer x) → (contains_digit 4 x) → (does_not_contain_digit 6 x) → 
  sorry

end count_three_digit_numbers_with_4_no_6_l545_545936


namespace only_IV_is_true_l545_545298

-- Define the average function
def avg (x y : ℝ) := (3 * x + 2 * y) / 5

-- Definitions to check each statement
def associative := ∀ x y z : ℝ, avg (avg x y) z = avg x (avg y z)
def commutative := ∀ x y : ℝ, avg x y = avg y x
def distributes_over_mul := ∀ x y z : ℝ, avg x (y * z) = avg x y * avg x z
def mul_distributes_over_avg := ∀ x y z : ℝ, x * avg y z = avg (x * y) (x * z)
def has_identity := ∃ i : ℝ, ∀ x : ℝ, avg x i = x

-- Lean statement to prove only IV is true
theorem only_IV_is_true : ¬ associative ∧ ¬ commutative ∧ ¬ distributes_over_mul ∧ mul_distributes_over_avg ∧ ¬ has_identity :=
by
  sorry

end only_IV_is_true_l545_545298


namespace range_of_a_l545_545897

open Real

theorem range_of_a (a : ℝ) (h : a > 0) :
  (∀ x : ℝ, f'(x) = 3 * x^2 + 2 * a * x + 1 →
   (∀ r1 r2 : ℝ, (3 * r1^2 + 2 * a * r1 + 1 = 0) → (3 * r2^2 + 2 * a * r2 + 1 = 0) →
   (-1 < r1 ∧ r1 < 1 ∧ -1 < r2 ∧ r2 < 1))) →
  (a \in Ioo (sqrt 3) 2) :=
sorry

def f (x : ℝ) (a : ℝ)  : ℝ := x^3 + a * x^2 + x + 2

def f' (x : ℝ) (a : ℝ) : ℝ := 3 * x^2 + 2 * a * x + 1

def g (x : ℝ) (a : ℝ) : ℝ := 3 * x^2 + 2 * a * x + 1

end range_of_a_l545_545897


namespace compute_modulo_l545_545811

theorem compute_modulo :
  (2023 * 2024 * 2025 * 2026) % 7 = 0 := by
  sorry

end compute_modulo_l545_545811


namespace cards_given_l545_545595

-- Define the total cards Nell had originally
def original_cards : ℕ := 304

-- Define the cards Nell has left after giving some away
def cards_left : ℕ := 250

-- Prove that the difference between the original cards and the cards left is 54
theorem cards_given (original: ℕ) (left: ℕ) : original - left = 54 :=
by
  have h1 : original = original_cards := rfl
  have h2 : left = cards_left := rfl
  rw [h1, h2]
  sorry

end cards_given_l545_545595


namespace integral_tan_cos_equal_l545_545217

theorem integral_tan_cos_equal :
  ∫ x in -1..1, (tan x) ^ 11 + (cos x) ^ 21 = 2 * ∫ x in 0..1, (cos x) ^ 21 :=
by
  sorry

end integral_tan_cos_equal_l545_545217


namespace largest_root_of_equation_l545_545220

theorem largest_root_of_equation : ∃ (x : ℝ), (x - 37)^2 - 169 = 0 ∧ ∀ y, (y - 37)^2 - 169 = 0 → y ≤ x :=
by
  sorry

end largest_root_of_equation_l545_545220


namespace trig_expression_value_l545_545445

theorem trig_expression_value {θ : Real} (h : Real.tan θ = 2) :
  (2 * Real.sin θ - Real.cos θ) / (Real.sin θ + 2 * Real.cos θ) = 3 / 4 := 
by
  sorry

end trig_expression_value_l545_545445


namespace patriots_won_games_l545_545142

theorem patriots_won_games (C P M S T E : ℕ) 
  (hC : C > 25)
  (hPC : P > C)
  (hMP : M > P)
  (hSC : S > C)
  (hSP : S < P)
  (hTE : T > E) : 
  P = 35 :=
sorry

end patriots_won_games_l545_545142


namespace fB_does_not_satisfy_any_equations_l545_545377

-- Definitions based on the given conditions
def eq1 (f : ℝ → ℝ) := ∀ x y : ℝ, f(x + y) = f(x) + f(y)
def eq2 (f : ℝ → ℝ) := ∀ x y : ℝ, f(x * y) = f(x) + f(y)
def eq3 (f : ℝ → ℝ) := ∀ x y : ℝ, f(x + y) = f(x) * f(y)
def eq4 (f : ℝ → ℝ) := ∀ x y : ℝ, f(x * y) = f(x) * f(y)

-- Functions given in the problem
def fA (x : ℝ) := 3^x
def fB (x : ℝ) := x + x⁻¹
def fC (x : ℝ) := Real.log2 x
def fD (k : ℝ) (x : ℝ) := k * x

-- The theorem to be proved
theorem fB_does_not_satisfy_any_equations :
  ¬ (eq1 fB ∨ eq2 fB ∨ eq3 fB ∨ eq4 fB) ∧ (eq1 fA ∨ eq2 fA ∨ eq3 fA ∨ eq4 fA) ∧ (eq1 fC ∨ eq2 fC ∨ eq3 fC ∨ eq4 fC) ∧ (∃ k : ℝ, eq1 (fD k) ∨ eq2 (fD k) ∨ eq3 (fD k) ∨ eq4 (fD k)) :=
  by {
    -- Here, "sorry" is used to indicate the place where a proof should go
    sorry
  }

end fB_does_not_satisfy_any_equations_l545_545377


namespace cosine_largest_angle_l545_545884

def triangle (A B C : Type) := 
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ a ≠ b ∧ b ≠ c ∧ c ≠ a

variables {A B C : Type} [triangle A B C]

def height_from_a (A B C : Type) (h_a : ℝ) : Prop := 
  h_a = (1 / 3)

def height_from_b (A B C : Type) (h_b : ℝ) : Prop := 
  h_b = (1 / 5)

def height_from_c (A B C : Type) (h_c : ℝ) : Prop := 
  h_c = (1 / 6)

theorem cosine_largest_angle (h_a h_b h_c : ℝ) :
  height_from_a A B C h_a → height_from_b A B C h_b → height_from_c A B C h_c →
  ∃ (cosC : ℝ), cosC = -(1 / 15) :=
by
  intros ha hb hc
  exists -(1 / 15)
  sorry

end cosine_largest_angle_l545_545884


namespace harmonious_sets_l545_545306

-- Definitions of symbols, colors, and intensities.
inductive Symbol
| star | circle | square

inductive Color
| red | yellow | blue

inductive Intensity
| light | normal | dark

-- A card is a combination of these three properties.
structure Card where
  symbol : Symbol
  color : Color
  intensity : Intensity

-- Define harmonious sets of cards.
def harmonious (c1 c2 c3 : Card) : Prop :=
  (c1.symbol = c2.symbol ∧ c2.symbol = c3.symbol ∨ c1.symbol ≠ c2.symbol ∧ c2.symbol ≠ c3.symbol ∧ c1.symbol ≠ c3.symbol) ∧
  (c1.color = c2.color ∧ c2.color = c3.color ∨ c1.color ≠ c2.color ∧ c2.color ≠ c3.color ∧ c1.color ≠ c3.color) ∧
  (c1.intensity = c2.intensity ∧ c2.intensity = c3.intensity ∨ c1.intensity ≠ c2.intensity ∧ c2.intensity ≠ c3.intensity ∧ c1.intensity ≠ c3.intensity)

-- Theorem statement.
theorem harmonious_sets : ∃ (S : Finset (Finset Card)), S.card = 702 ∧ ∀ s ∈ S, s.card = 3 ∧ ∃ c1 c2 c3, {c1, c2, c3} = s ∧ harmonious c1 c2 c3 :=
by
  sorry

end harmonious_sets_l545_545306


namespace balloons_remaining_l545_545851
-- Importing the necessary libraries

-- Defining the conditions
def originalBalloons : Nat := 709
def givenBalloons : Nat := 221

-- Stating the theorem
theorem balloons_remaining : originalBalloons - givenBalloons = 488 := by
  sorry

end balloons_remaining_l545_545851


namespace hemisphere_base_area_l545_545171

theorem hemisphere_base_area (r : ℝ) (π : ℝ) (h₁ : π > 0) 
  (sphere_surface_area : 4 * π * r^2) 
  (hemisphere_surface_area : 3 * π * r^2 = 9) : 
  π * r^2 = 3 := 
by 
  sorry

end hemisphere_base_area_l545_545171


namespace tom_total_calories_l545_545192

-- Define the conditions
def c_weight : ℕ := 1
def c_calories_per_pound : ℕ := 51
def b_weight : ℕ := 2 * c_weight
def b_calories_per_pound : ℕ := c_calories_per_pound / 3

-- Define the total calories
def total_calories : ℕ := (c_weight * c_calories_per_pound) + (b_weight * b_calories_per_pound)

-- Prove the total calories Tom eats
theorem tom_total_calories : total_calories = 85 := by
  sorry

end tom_total_calories_l545_545192


namespace opposite_of_neg_five_l545_545642

theorem opposite_of_neg_five : ∃ (y : ℤ), -5 + y = 0 ∧ y = 5 :=
by
  use 5
  simp

end opposite_of_neg_five_l545_545642


namespace existence_of_a_l545_545079

noncomputable theory
open_locale big_operators

theorem existence_of_a (p : ℕ) [hp_prime : fact (nat.prime p)] (hp_ge5 : 5 ≤ p) :
  ∃ a : ℤ, 1 ≤ a ∧ a ≤ ↑(p - 2) ∧
    ¬(p^2 ∣ a^(p-1) - 1) ∧ ¬(p^2 ∣ (a + 1)^(p-1) - 1) :=
sorry

end existence_of_a_l545_545079


namespace prod_mod7_eq_zero_l545_545805

theorem prod_mod7_eq_zero :
  (2023 * 2024 * 2025 * 2026) % 7 = 0 := 
by {
  sorry
}

end prod_mod7_eq_zero_l545_545805


namespace point_coordinates_l545_545545

-- Definitions for conditions based on the problem statement
variable (x y : ℝ)
variable (P : (ℝ × ℝ))

-- Condition: P is to the right of the y-axis implies x > 0
def isRightOfYAxis (P : ℝ × ℝ) : Prop :=
  P.1 > 0

-- Condition: P is below the x-axis implies y < 0
def isBelowXAxis (P : ℝ × ℝ) : Prop :=
  P.2 < 0

-- Defining the point P
variable [hPx : x = P.1] [hPy : y = P.2]

-- Proof goal
theorem point_coordinates (P : ℝ × ℝ) (ht : isRightOfYAxis P ∧ isBelowXAxis P) : (P.1 > 0 ∧ P.2 < 0) :=
  by sorry

end point_coordinates_l545_545545


namespace sum_of_first_90_terms_l545_545464

def arithmetic_progression_sum (n : ℕ) (a d : ℚ) : ℚ :=
  (n : ℚ) / 2 * (2 * a + (n - 1) * d)

theorem sum_of_first_90_terms (a d : ℚ) :
  (arithmetic_progression_sum 15 a d = 150) →
  (arithmetic_progression_sum 75 a d = 75) →
  (arithmetic_progression_sum 90 a d = -112.5) :=
by
  sorry

end sum_of_first_90_terms_l545_545464


namespace projections_on_hypotenuse_l545_545049

variables {a b c p q : ℝ}
variables {ρa ρb : ℝ}

-- Given conditions
variable (h1 : 0 < a ∧ 0 < b ∧ 0 < c)
variable (h2 : a < b)
variable (h3 : p = a * a / c)
variable (h4 : q = b * b / c)
variable (h5 : ρa = (a * (b + c - a)) / (a + b + c))
variable (h6 : ρb = (b * (a + c - b)) / (a + b + c))

-- Proof goal
theorem projections_on_hypotenuse 
  (h_right_triangle: a^2 + b^2 = c^2) : p < ρa ∧ q > ρb :=
by
  sorry

end projections_on_hypotenuse_l545_545049


namespace right_triangle_area_l545_545694

/-- Given a right triangle with one leg of length 30 inches and a hypotenuse of 34 inches,
    the area of the triangle is 240 square inches. -/
theorem right_triangle_area (a b c : ℝ) (h1 : a = 30) (h2 : c = 34) (h3 : a^2 + b^2 = c^2) :
  (1/2) * a * b = 240 :=
by
  rw [h1, h2] at h3
  have hb : b = 16 := by
    rw [←h3]
    norm_num
  rw [h1, hb]
  norm_num
  sorry

end right_triangle_area_l545_545694


namespace right_triangles_with_leg_2012_l545_545527

theorem right_triangles_with_leg_2012 :
  ∀ (a b c : ℕ), a = 2012 ∧ a ^ 2 + b ^ 2 = c ^ 2 → 
  (b = 253005 ∧ c = 253013) ∨ 
  (b = 506016 ∧ c = 506020) ∨ 
  (b = 1012035 ∧ c = 1012037) ∨ 
  (b = 1509 ∧ c = 2515) :=
by
  intros
  sorry

end right_triangles_with_leg_2012_l545_545527


namespace largest_even_integer_sum_l545_545156

theorem largest_even_integer_sum : 
  let sum_first_25_even := 2 * (25 * 26) / 2
  ∃ n : ℕ, 
  n % 2 = 0 ∧ 
  sum_first_25_even = 5 * n - 20 ∧
  n = 134 :=
by
  let sum_first_25_even := 2 * (25 * 26) / 2
  have h_sum : sum_first_25_even = 650 := by norm_num
  use 134
  split
  · norm_num
  split
  · rw h_sum
    norm_num
  · rfl

end largest_even_integer_sum_l545_545156


namespace ABCD_is_parallelogram_l545_545063

-- Define the conditions in Lean
variables (A B C D O H : Type) [AffineSpace ℝ A] [AffineSpace ℝ B] [AffineSpace ℝ C] [AffineSpace ℝ D] [AffineSpace ℝ O] [AffineSpace ℝ H]
variables (quadrilateral : convex_quadrilateral A B C D)
variables (angle_B_eq_D : angle A B C = angle A D C)
variables (circumcenter_O : circumcenter A B C = O)
variables (orthocenter_H : orthocenter A D C = H)
variables (B_O_H_collinear : collinear ℝ (set_of_points [B, O, H]))

-- The statement that \(ABCD\) is a parallelogram
theorem ABCD_is_parallelogram : is_parallelogram A B C D :=
sorry

end ABCD_is_parallelogram_l545_545063


namespace median_to_hypotenuse_l545_545115

variable {α : Type*} [InnerProductSpace ℝ α] {A B C M : α}

-- Conditions
-- Let ABC be a right triangle with the hypotenuse AB.
-- Let M be the midpoint of AB.
def is_right_triangle (A B C : α) : Prop :=
  ∃ (a b c : ℝ), a^2 + b^2 = c^2 ∧ dist A B = c ∧ dist A C = a ∧ dist C B = b 

def is_midpoint (M A B : α) : Prop :=
  dist A M = dist B M ∧ (M -ᵥ A : α) + (M -ᵥ B : α) = 0

-- Question: Prove CM = AB / 2.
theorem median_to_hypotenuse (hABC : is_right_triangle A B C) (hM : is_midpoint M A B) :
  dist C M = (dist A B) / 2 := sorry

end median_to_hypotenuse_l545_545115


namespace triangle_area_PFK_l545_545005

-- Definitions according to the conditions
def parabola_eq (x y : ℝ) : Prop := y^2 = 4 * x
def F := (1 : ℝ, 0 : ℝ)
def K := (-1 : ℝ, 0 : ℝ)
def is_on_parabola (P : ℝ × ℝ) : Prop := parabola_eq P.1 P.2
def distance (A B : ℝ × ℝ) : ℝ := real.sqrt((B.1 - A.1)^2 + (B.2 - A.2)^2)

-- The main theorem to prove
theorem triangle_area_PFK (P : ℝ × ℝ) (hP : is_on_parabola P) (hPF : distance P F = 5) :
  let area := 1 / 2 * 2 * 4 in area = 4 :=
by
  sorry

end triangle_area_PFK_l545_545005


namespace product_of_random_numbers_greater_zero_l545_545201

noncomputable def random_product_positive_probability : ℝ := 
  let interval_length := 45
  let neg_interval_length := 30
  let pos_interval_length := 15
  let prob_neg := (neg_interval_length : ℝ) / interval_length
  let prob_pos := (pos_interval_length : ℝ) / interval_length
  prob_pos * prob_pos + prob_neg * prob_neg

-- Prove that the probability that the product of two randomly selected numbers
-- from the interval [-30, 15] is greater than zero is 5/9.
theorem product_of_random_numbers_greater_zero : 
  random_product_positive_probability = 5 / 9 := by
  sorry

end product_of_random_numbers_greater_zero_l545_545201


namespace find_triangles_geometric_sequence_l545_545149

theorem find_triangles_geometric_sequence (a b c : ℕ) (h1 : a ≤ b) (h2 : b ≤ c)
(h3 : a * a = b * b / c) (h4 : (a = 100 ∨ c = 100)) : (a, b, c) ∈ 
{(49, 70, 100), (64, 80, 100), (81, 90, 100), (100, 100, 100), 
(100, 110, 121), (100, 120, 144), (100, 130, 169), (100, 140, 196), 
(100, 150, 225), (100, 160, 256)} :=
sorry

end find_triangles_geometric_sequence_l545_545149


namespace count_three_digit_integers_with_4_and_without_6_l545_545922

def is_three_digit_integer (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

def contains_digit (n : ℕ) (d : ℕ) : Prop :=
  nat.digits 10 n |> list.any (λ x => x = d)

def does_not_contain_digit (n : ℕ) (d : ℕ) : Prop :=
  nat.digits 10 n |> list.all (λ x => x ≠ d)

theorem count_three_digit_integers_with_4_and_without_6 : 
  (nat.card {n : ℕ // is_three_digit_integer n ∧ contains_digit n 4 ∧ does_not_contain_digit n 6} = 200) :=
by
  sorry

end count_three_digit_integers_with_4_and_without_6_l545_545922


namespace first_expression_second_expression_l545_545797

-- Define the variables
variables {a x y : ℝ}

-- Statement for the first expression
theorem first_expression (a : ℝ) : (2 * a^2)^3 + (-3 * a^3)^2 = 17 * a^6 := sorry

-- Statement for the second expression
theorem second_expression (x y : ℝ) : (x + 3 * y) * (x - y) = x^2 + 2 * x * y - 3 * y^2 := sorry

end first_expression_second_expression_l545_545797


namespace solve_sqrt_equation_l545_545302

theorem solve_sqrt_equation (z : ℝ) : (sqrt (5 + 4 * z) = 8) → (z = 59 / 4) :=
by
  sorry

end solve_sqrt_equation_l545_545302


namespace root_polynomial_value_l545_545366

theorem root_polynomial_value (m : ℝ) (h : m^2 + 3 * m - 2022 = 0) : m^3 + 4 * m^2 - 2019 * m - 2023 = -1 :=
  sorry

end root_polynomial_value_l545_545366


namespace right_triangle_area_l545_545697

/-- Given a right triangle with one leg of length 30 inches and a hypotenuse of 34 inches,
    the area of the triangle is 240 square inches. -/
theorem right_triangle_area (a b c : ℝ) (h1 : a = 30) (h2 : c = 34) (h3 : a^2 + b^2 = c^2) :
  (1/2) * a * b = 240 :=
by
  rw [h1, h2] at h3
  have hb : b = 16 := by
    rw [←h3]
    norm_num
  rw [h1, hb]
  norm_num
  sorry

end right_triangle_area_l545_545697


namespace range_of_arcsin_cos_l545_545860

noncomputable def arcsin_range_cos (x : ℝ) (α : ℝ) : Prop :=
  x = Real.cos α ∧ α ∈ set.Icc (-Real.pi / 4) (3 * Real.pi / 4) → 
  set.Icc (-Real.pi / 4) (Real.pi / 2)

theorem range_of_arcsin_cos (x : ℝ) (α : ℝ) :
  x = Real.cos α ∧ α ∈ set.Icc (-Real.pi / 4) (3 * Real.pi / 4) →
  arcsin x ∈ set.Icc (-Real.pi / 4) (Real.pi / 2) := sorry

end range_of_arcsin_cos_l545_545860


namespace largest_of_five_consecutive_even_integers_l545_545162

theorem largest_of_five_consecutive_even_integers :
  (∃ n, 2 + 4 + ... + 50 = 5 * n - 20) →
  (∃ n, ∑ i in range 25, 2 * (i + 1) = 5 * n - 20) →
  (∃ n, n = 134) :=
begin
  sorry
end

end largest_of_five_consecutive_even_integers_l545_545162


namespace tom_total_calories_l545_545190

-- Define the conditions
def c_weight : ℕ := 1
def c_calories_per_pound : ℕ := 51
def b_weight : ℕ := 2 * c_weight
def b_calories_per_pound : ℕ := c_calories_per_pound / 3

-- Define the total calories
def total_calories : ℕ := (c_weight * c_calories_per_pound) + (b_weight * b_calories_per_pound)

-- Prove the total calories Tom eats
theorem tom_total_calories : total_calories = 85 := by
  sorry

end tom_total_calories_l545_545190


namespace a_n_formula_l545_545051

noncomputable def a : ℕ → ℝ
| 0     := 1 / 2        -- This corresponds to a₀ = (3^0 + 1) / 2 = 1
| (n+1) := 3 * a n - 1

theorem a_n_formula (n : ℕ) : a n = (3^n + 1) / 2 :=
by 
  sorry

end a_n_formula_l545_545051


namespace mean_of_all_students_is_76_l545_545591

-- Definitions
def M : ℝ := 84
def A : ℝ := 70
def ratio_m_a : ℝ := 3 / 4

-- Theorem statement
theorem mean_of_all_students_is_76 (m a : ℝ) (hm : m = ratio_m_a * a) : 
  ((63 * a) + 70 * a) / ((3 / 4 * a) + a) = 76 := 
sorry

end mean_of_all_students_is_76_l545_545591


namespace surface_integral_cylinder_solution_l545_545664

noncomputable def surface_integral (σ : Set (ℝ × ℝ × ℝ)) : ℝ :=
  ∫⁻ (x y z : ℝ) in σ, 4 * x^3 * (0, 1, 0 : ℝ×ℝ×ℝ).2.1 ∂ 1 +
  ∫⁻ (x y z : ℝ) in σ, 4 * y^3 * (1, 0, 0 : ℝ×ℝ×ℝ).1.1 ∂ 1 +
  ∫⁻ (x y z : ℝ) in σ, -6 * z^4 * (1, 0, 0 : ℝ×ℝ×ℝ).1.1 ∂ 1

theorem surface_integral_cylinder_solution
  (σ : Set (ℝ × ℝ × ℝ))
  (a h : ℝ)
  (hs : σ = { p : ℝ × ℝ × ℝ | p.1.1^2 + p.2.1^2 ≤ a^2 ∧ 0 ≤ p.1 ∧ p.1 ≤ h }) :
  surface_integral σ = 6 * π * a^2 * h * (a^2 - h^3 / 2) := 
sorry

end surface_integral_cylinder_solution_l545_545664


namespace arithmetic_sequence_sum_l545_545531

-- Define the arithmetic sequence and the condition on its sum
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a n = a 0 + n * d

-- Define the sum of the first 10 terms of the arithmetic sequence
noncomputable def S (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  ∑ i in finset.range (n + 1), a i

-- State the theorem to prove
theorem arithmetic_sequence_sum
  (a : ℕ → ℝ)
  (h_arith : is_arithmetic_sequence a)
  (h_sum : S a 9 = 120) :
  a 0 + a 9 = 24 :=
sorry

end arithmetic_sequence_sum_l545_545531


namespace cloak_change_in_silver_l545_545485

theorem cloak_change_in_silver :
  (∀ c : ℤ, (20 = c + 4) → (15 = c + 1)) →
  (5 * g = 3) →
  14 * gold / exchange_rate = 10 := 
sorry

end cloak_change_in_silver_l545_545485


namespace find_a_l545_545009

theorem find_a (a : ℝ) 
  (h1 : ∃ x ∈ ({a - 2, 2 * a ^ 2 + 5 * a, 12} : set ℝ), x = -3) : 
  a = -3 / 2 := 
by
  sorry

end find_a_l545_545009


namespace count_four_digit_numbers_divisible_by_5_end_45_l545_545417

theorem count_four_digit_numbers_divisible_by_5_end_45 : 
  {n : ℕ | 1000 ≤ n ∧ n < 10000 ∧ n % 100 = 45 ∧ n % 5 = 0}.to_finset.card = 90 :=
by
  sorry

end count_four_digit_numbers_divisible_by_5_end_45_l545_545417


namespace certain_number_k_l545_545948

theorem certain_number_k (x : ℕ) (k : ℕ) (h1 : x = 14) (h2 : 2^x - 2^(x-2) = k * 2^12) : k = 3 := by
  sorry

end certain_number_k_l545_545948


namespace smallest_square_side_length_l545_545105

theorem smallest_square_side_length (D : ℝ) (hD : D = 1) (figure : set (ℝ × ℝ)) (diam : ∀ (A B ∈ figure), dist A B ≤ D) :
  ∃ (side_length : ℝ), side_length = 1 ∧ (∀ (A B ∈ figure), dist A B ≤ side_length) :=
sorry

end smallest_square_side_length_l545_545105


namespace problem_statement_l545_545945

theorem problem_statement (x y : ℝ) (h : 2 * x - y = 8) : 6 - 2 * x + y = -2 := by
-- conditions from a)
assume h : 2 * x - y = 8,
-- the goal is to prove the equivalent answer
show 6 - 2 * x + y = -2,
sorry

end problem_statement_l545_545945


namespace problem_part_1_problem_part_2_l545_545381

noncomputable def f : ℝ → ℝ := λ x, (3 * real.sqrt 2 / 2) * real.sin (2 * x - real.pi / 6) + real.sqrt 2 / 2

theorem problem_part_1 
  (A ω : ℝ) (φ : ℝ) (B : ℝ)
  (hA : 0 < A) (hω : 0 < ω) (hφ : abs φ < real.pi / 2)
  (hmax : A + B = 2 * real.sqrt 2) 
  (hmin : B - A = -real.sqrt 2) 
  (hperiod : real.pi = 2 * real.pi / ω)
  (hpoint : f 0 = - real.sqrt 2 / 4) :
  ∃ k ∈ ℤ, f = λ x, (3 * real.sqrt 2 / 2) * real.sin (2 * x - real.pi / 6) + real.sqrt 2 / 2 ∧ 
  intervals of monotonic increase is (λ x, ∃ k ∈ ℤ, kπ - π / 6 ≤ x ≤ π / 3 + kπ) := sorry

theorem problem_part_2 
  (a α β : ℝ) 
  (hinterval : ∀ x, 0 ≤ x ∧ x ≤ 7 * real.pi / 12)
  (htwo_roots : α ≠ β)
  (ha_range : ∃ a, (a ∈ [real.sqrt 2 / 2, 2 * real.sqrt 2)) 
  (halpha_beta : α + β = 2 * real.pi / 3) :
  α + β = 2 * real.pi / 3 := sorry

end problem_part_1_problem_part_2_l545_545381


namespace largest_of_five_consecutive_even_integers_sum_l545_545158

theorem largest_of_five_consecutive_even_integers_sum (n : ℤ) :
  (∑ i in finset.range 25, (2 * (i + 1))) = 650 ∧
  (∑ i in finset.range 5, (n - 8 + 2 * i)) = 650 →
  n = 134 :=
by 
  sorry

end largest_of_five_consecutive_even_integers_sum_l545_545158


namespace sale_price_for_50_percent_profit_l545_545146

theorem sale_price_for_50_percent_profit
  (C L: ℝ)
  (h1: 892 - C = C - L)
  (h2: 1005 = 1.5 * C) :
  1.5 * C = 1005 :=
by
  sorry

end sale_price_for_50_percent_profit_l545_545146


namespace right_triangle_area_l545_545677

theorem right_triangle_area (leg1 leg2 hypotenuse : ℕ) (h_leg1 : leg1 = 30)
  (h_hypotenuse : hypotenuse = 34)
  (hypotenuse_sq : hypotenuse * hypotenuse = leg1 * leg1 + leg2 * leg2) :
  (1 / 2 : ℚ) * leg1 * leg2 = 240 := by
  sorry

end right_triangle_area_l545_545677


namespace number_of_terms_in_product_l545_545939

theorem number_of_terms_in_product 
  (a b c d e f g h i : ℕ) :
  (a + b + c + d) * (e + f + g + h + i) = 20 :=
sorry

end number_of_terms_in_product_l545_545939


namespace max_b_times_a_divisible_by_25_l545_545727

def digits := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

def divisible_by_25 (a b : ℕ) : Prop :=
  (10 * a + b) % 25 = 0

theorem max_b_times_a_divisible_by_25 :
  ∃ (a b : ℕ) (ha : a ∈ digits) (hb : b ∈ digits), divisible_by_25 a b ∧ b * a = 35 :=
by {
  sorry -- Proof to be completed.
}

end max_b_times_a_divisible_by_25_l545_545727


namespace integer_solution_unique_l545_545837

theorem integer_solution_unique
  (a b c d : ℤ)
  (h : a^2 + 5 * b^2 - 2 * c^2 - 2 * c * d - 3 * d^2 = 0) :
  a = 0 ∧ b = 0 ∧ c = 0 ∧ d = 0 :=
sorry

end integer_solution_unique_l545_545837


namespace find_length_OD1_l545_545069

noncomputable def length_OD1 (r OX_radius_radius_1 OX_radius_radius_3 : ℝ) 
    (sphere_radius : ℝ) (proof_radius_1 : r = 1) (proof_radius_3 : r = 3)
    (proof_sphere_radius : sphere_radius = 10)
    (hyp1 : OX_radius_radius_1 = 1) (hyp2 : OX_radius_radius_3 = 3) : ℝ :=
  let radius_for_1_face : ℝ := (sphere_radius ^ 2 - hyp1 ^ 2).sqrt
  let radius_for_other_face : ℝ := (sphere_radius ^ 2 - hyp2 ^ 2).sqrt
  let radius_for_3rd_face : ℝ := (sphere_radius ^ 2 - hyp2 ^ 2).sqrt
  let total_distance_squares : ℝ := radius_for_1_face ^ 2 + radius_for_other_face ^ 2 + radius_for_3rd_face ^ 2 
  total_distance_squares.sqrt

theorem find_length_OD1 : length_OD1 1 3 10 1 3 10 1 3 = 17 := by
  sorry

end find_length_OD1_l545_545069


namespace sum_of_first_five_terms_l545_545350

theorem sum_of_first_five_terms (a : ℕ → ℤ) (h1 : a 2 + a 4 = 6) : ∑ i in finset.range 5, a i = 15 :=
  sorry

end sum_of_first_five_terms_l545_545350


namespace alan_tickets_l545_545186

theorem alan_tickets (a m : ℕ) (h1 : a + m = 150) (h2 : m = 5 * a - 6) : a = 26 :=
by
  sorry

end alan_tickets_l545_545186


namespace problem_statement_l545_545388

noncomputable def verify_ratio (x y c d : ℝ) (h1 : 4 * x - 2 * y = c) (h2 : 6 * y - 12 * x = d) (h3 : d ≠ 0) : Prop :=
  c / d = -1/3

theorem problem_statement (x y c d : ℝ) (h1 : 4 * x - 2 * y = c) (h2 : 6 * y - 12 * x = d) (h3 : d ≠ 0) : verify_ratio x y c d h1 h2 h3 :=
  sorry

end problem_statement_l545_545388


namespace count_four_digit_numbers_divisible_by_5_ending_in_45_l545_545403

theorem count_four_digit_numbers_divisible_by_5_ending_in_45 : 
  ∃ (n : ℕ), n = 90 ∧ ∀ (x : ℕ), (1000 ≤ x ∧ x < 10000) ∧ (x % 100 = 45) → count x = n :=
sorry

end count_four_digit_numbers_divisible_by_5_ending_in_45_l545_545403


namespace sum_smallest_and_third_smallest_is_786_l545_545665

-- Define the set of digits used
def digits := {1, 6, 8}

-- Define a function to generate all permutations of a list
def permutations {α : Type*} (l : List α) : List (List α) := sorry

-- Define a function that forms a three-digit number from a list of three digits
def form_number (l : List Nat) : Nat :=
  match l with
  | [a, b, c] => 100 * a + 10 * b + c
  | _ => 0 -- Default case, should not happen

-- Prove that the sum of the smallest three-digit number and the third smallest three-digit number is 786
theorem sum_smallest_and_third_smallest_is_786 :
  let nums := (permutations [1, 6, 8]).map form_number
  let sorted_nums := nums.sort (λ a b => a < b)
  let smallest := sorted_nums.getD 0 0
  let third_smallest := sorted_nums.getD 2 0
  smallest + third_smallest = 786 :=
by
  sorry

end sum_smallest_and_third_smallest_is_786_l545_545665


namespace find_phi_l545_545343

def f (x : ℝ) : ℝ := sin x + (sqrt 3) * cos x

theorem find_phi :
  (∀ x : ℝ, f (x + φ) = f (-x + φ)) ↔ φ = π / 6 := by
  sorry

end find_phi_l545_545343


namespace x1_x2_eq_e2_l545_545034

variable (x1 x2 : ℝ)

-- Conditions
def condition1 : Prop := x1 * Real.exp x1 = Real.exp 2
def condition2 : Prop := x2 * Real.log x2 = Real.exp 2

-- The proof problem
theorem x1_x2_eq_e2 (hx1 : condition1 x1) (hx2 : condition2 x2) : x1 * x2 = Real.exp 2 := 
sorry

end x1_x2_eq_e2_l545_545034


namespace count_four_digit_numbers_divisible_by_5_and_ending_with_45_l545_545411

def is_four_digit_number (n : ℕ) : Prop := 1000 ≤ n ∧ n < 10000
def ends_with_45 (n : ℕ) : Prop := (n % 100) = 45 
def divisible_by_5 (n : ℕ) : Prop := (n % 5) = 0

theorem count_four_digit_numbers_divisible_by_5_and_ending_with_45 : 
  {n : ℕ | is_four_digit_number n ∧ ends_with_45 n ∧ divisible_by_5 n}.to_finset.card = 90 :=
sorry

end count_four_digit_numbers_divisible_by_5_and_ending_with_45_l545_545411


namespace tom_calories_l545_545194

theorem tom_calories :
  let carrot_pounds := 1
  let broccoli_pounds := 2 * carrot_pounds
  let carrot_calories_per_pound := 51
  let broccoli_calories_per_pound := carrot_calories_per_pound / 3
  let total_carrot_calories := carrot_pounds * carrot_calories_per_pound
  let total_broccoli_calories := broccoli_pounds * broccoli_calories_per_pound
  let total_calories := total_carrot_calories + total_broccoli_calories
  total_calories = 85 :=
by
  sorry

end tom_calories_l545_545194


namespace minimal_constant_c_equality_conditions_l545_545574

variable (n : ℕ) (x : Fin n → ℝ)

theorem minimal_constant_c (h : 2 ≤ n) :
  ∃ c : ℝ, (∀ (x : Fin n → ℝ), (∀ i, 0 ≤ x i) → 
    ∑ i j in Finset.off_diag (Finset.range n),
      x i * x j * (x i ^ 2 + x j ^ 2) ≤ c * (∑ i, x i) ^ 4) ∧
    (∀ c', (∀ (x : Fin n → ℝ), (∀ i, 0 ≤ x i) → 
      ∑ i j in Finset.off_diag (Finset.range n),
        x i * x j * (x i ^ 2 + x j ^ 2) ≤ c' * (∑ i, x i) ^ 4) → c ≤ c') ∧
    c = 1 / 8 := sorry

theorem equality_conditions (h : 2 ≤ n) :
  ∀ (x : Fin n → ℝ), (∀ i, 0 ≤ x i) → 
    (∑ i j in Finset.off_diag (Finset.range n),
      x i * x j * (x i ^ 2 + x j ^ 2) = (1 / 8) * (∑ i, x i) ^ 4 ↔
      (∃ i j, i ≠ j ∧ x i = x j ∧ ∀ k, k ≠ i ∧ k ≠ j → x k = 0)) := sorry

end minimal_constant_c_equality_conditions_l545_545574


namespace PQC_eq_PQD_l545_545786

open EuclideanGeometry

variables {A B C D M P Q : Point}

theorem PQC_eq_PQD 
    (h_convex : ConvexQuad ABCD)
    (h_midM : Midpoint M A B)
    (h_MC_eq_MD : MC = MD)
    (h_perpendicular_C : Perpendicular (LineThruPointSeg P BC) C)
    (h_perpendicular_D : Perpendicular (LineThruPointSeg P AD) D)
    (h_perpendicular_Q : Perpendicular (LineThruPointSeg P AB) Q)
    : Angle PQC = Angle PQD :=
by
  sorry

end PQC_eq_PQD_l545_545786


namespace tom_calories_l545_545195

theorem tom_calories :
  let carrot_pounds := 1
  let broccoli_pounds := 2 * carrot_pounds
  let carrot_calories_per_pound := 51
  let broccoli_calories_per_pound := carrot_calories_per_pound / 3
  let total_carrot_calories := carrot_pounds * carrot_calories_per_pound
  let total_broccoli_calories := broccoli_pounds * broccoli_calories_per_pound
  let total_calories := total_carrot_calories + total_broccoli_calories
  total_calories = 85 :=
by
  sorry

end tom_calories_l545_545195


namespace mass_of_basketball_l545_545104

-- Define the conditions
variables (M m l1 l2 x y : ℝ)
variables (h1 : m = 400)
variables (h2 : x = 9)
variables (h3 : y = 14)
variables (h4 : M * l1 = m * l2)
variables (h5 : M * (l1 + x) = 2 * m * (l2 - x))
variables (h6 : M * (l1 + y) = 3 * m * (l2 - y))

-- Statement to prove
theorem mass_of_basketball (M m l1 l2 x y : ℝ)
  (h1 : m = 400)
  (h2 : x = 9)
  (h3 : y = 14)
  (h4 : M * l1 = m * l2)
  (h5 : M * (l1 + x) = 2 * m * (l2 - x))
  (h6 : M * (l1 + y) = 3 * m * (l2 - y)) : M = 600 :=
begin
  sorry
end

end mass_of_basketball_l545_545104


namespace betty_books_l545_545283

variable (B : ℝ)
variable (h : B + (5/4) * B = 45)

theorem betty_books : B = 20 := by
  sorry

end betty_books_l545_545283


namespace arrangement_of_boys_and_girls_l545_545182

theorem arrangement_of_boys_and_girls :
  ∃ (arrangements : Finset (List (Sum (Fin 3) (Fin 3)))), arrangements.card = 144 ∧
    (∀ (l : List (Sum (Fin 3) (Fin 3))),
     l ∈ arrangements →
     (∃ i : ℕ, ([Sum.inl 0, Sum.inl 1] ⊂ l.drop i.take 2) ∧
      ∃ j : ℕ, ([Sum.inr 0, Sum.inr 1] ⊂ l.drop j.take 2))) :=
sorry

end arrangement_of_boys_and_girls_l545_545182


namespace sin_double_angle_abs_bound_l545_545114

theorem sin_double_angle_abs_bound (x : ℝ) (h : sin x > 0.9) : abs (sin (2 * x)) < 0.9 := 
sorry

end sin_double_angle_abs_bound_l545_545114


namespace min_value_func_l545_545367

theorem min_value_func {x : ℝ} (hx : x > 1) : 
  let y := x + 1 / (x - 1)
  in y ≥ 3 ∧ (y = 3 ↔ x = 2) :=
by
  sorry

end min_value_func_l545_545367


namespace count_four_digit_numbers_divisible_by_5_ending_in_45_l545_545420

theorem count_four_digit_numbers_divisible_by_5_ending_in_45 : 
  ∃ n : ℕ, (∀ x : ℕ, x ≥ 1000 ∧ x < 10000 ∧ x % 100 = 45 → (x % 5 = 0 ∧ x % 100 = 45)) 
  ∧ (n = 90) :=
by
  sorry

end count_four_digit_numbers_divisible_by_5_ending_in_45_l545_545420


namespace cloak_change_l545_545498

theorem cloak_change (silver_for_cloak1 silver_for_cloak2 : ℕ) (gold_change1 gold_change2 : ℕ) (silver_change : ℕ) :
  silver_for_cloak1 = 20 →
  gold_change1 = 4 →
  silver_for_cloak2 = 15 →
  gold_change2 = 1 →
  ∃ silver_cost_of_cloak : ℕ, 
    silver_cost_of_cloak = (20 - 4) * (5 / 3) →
    silver_change = 10 →
    14 * (5 / 3) - 8 = silver_change :=
by 
  assume h1 h2 h3 h4,
  use 16, 
  sorry

end cloak_change_l545_545498


namespace probability_product_positive_is_5_div_9_l545_545198

noncomputable def probability_positive_product : ℚ :=
  let interval := Set.Icc (-30 : ℝ) 15
  let length_interval := 45
  let length_neg := 30
  let length_pos := 15
  let prob_neg := (length_neg : ℚ) / length_interval
  let prob_pos := (length_pos : ℚ) / length_interval
  let prob_product_pos := prob_neg^2 + prob_pos^2
  prob_product_pos

theorem probability_product_positive_is_5_div_9 :
  probability_positive_product = 5 / 9 :=
by
  sorry

end probability_product_positive_is_5_div_9_l545_545198


namespace probability_at_A_after_8_steps_l545_545985

variables (A B C D : Type) {T : Type}
variable [hab: MetricSpace A,B,C,D]
variables (P : ℕ → ℚ)

axiom start_prob : P 0 = 1
axiom recursive_prob :
  ∀ n, P (n + 1) = 1/3 * (1 - P n)

/-- Prove that the probability of the bug being at vertex A after crawling
exactly 8 meters is 547/2187, expressed as p = n/2187, and find n, n = 547. -/
theorem probability_at_A_after_8_steps :
  P 8 = 547 / 2187 :=
sorry

end probability_at_A_after_8_steps_l545_545985


namespace find_triples_tan_l545_545315

open Real

theorem find_triples_tan (x y z : ℝ) :
  x * (1 - y^2) * (1 - z^2) + y * (1 - z^2) * (1 - x^2) + z * (1 - x^2) * (1 - y^2) = 4 * x * y * z → 
  ∃ (A B C : ℝ), x = tan A ∧ y = tan B ∧ z = tan C :=
by
  sorry

end find_triples_tan_l545_545315


namespace largest_number_divided_by_31_l545_545320

theorem largest_number_divided_by_31 : ∃ N : ℕ, (∃ k r : ℕ, k = 30 ∧ r < 31 ∧ N = 31 * k + r) ∧ N = 960 :=
by
  let k : ℕ := 30
  let r : ℕ := 30
  let N : ℕ := 31 * k + r
  have h1 : N = 960 := by simp [N, k, r]
  use N
  existsi k
  existsi r
  split
  { split
    { refl }
    { exact Nat.lt_of_succ_eq_succ rfl } }
  { exact h1 }

end largest_number_divided_by_31_l545_545320


namespace students_with_certificates_l545_545789

variable (C N : ℕ)

theorem students_with_certificates :
  (C + N = 120) ∧ (C = N + 36) → C = 78 :=
by
  sorry

end students_with_certificates_l545_545789


namespace area_of_gray_region_l545_545660

noncomputable def radius_of_inner_circle (r : ℝ) : Prop := (3 * r - r = 4) 
noncomputable def radius_of_outer_circle (r : ℝ) : ℝ := 3 * r

theorem area_of_gray_region (r : ℝ) (h1 : radius_of_inner_circle r) :
  let R := radius_of_outer_circle r
  in R = 6 → (π * R ^ 2) - (π * r ^ 2) =  32 * π :=
by
  intro h
  sorry

end area_of_gray_region_l545_545660


namespace derangements_five_friends_l545_545753

-- Define derangements
def derangements (n : ℕ) : ℕ := 
  if n = 0 then 1
  else if n = 1 then 0
  else (n - 1) * (derangements (n - 1) + derangements (n - 2))

-- Prove the number of derangements of 5 elements is 44
theorem derangements_five_friends : derangements 5 = 44 :=
by sorry

end derangements_five_friends_l545_545753


namespace algebraic_expression_increased_by_four_l545_545969

-- Definitions
def f (x y : ℝ) : ℝ := (x^2 * y) / (x - y)

-- The theorem we want to prove
theorem algebraic_expression_increased_by_four (x y : ℝ) (h : x ≠ y): 
  f (2 * x) (2 * y) = 4 * f x y := 
  by
  -- Placeholder for the actual proof
  sorry

end algebraic_expression_increased_by_four_l545_545969


namespace cloak_change_14_gold_coins_l545_545512

def exchange_rate (silver gold : ℕ) : Prop :=
  ∃ c : ℕ, (20 - 4) * c = silver ∧ (15 - 1) * c = silver

def cloak_purchase (paid_gold received_silver : ℕ) : Prop :=
  let exchange_rate := (5 * 14) / 3 in
  received_silver = 2 * exchange_rate

theorem cloak_change_14_gold_coins :
  exchange_rate 16 3 →
  exchange_rate 14 1 →
  cloak_purchase 14 10 := sorry

end cloak_change_14_gold_coins_l545_545512


namespace initial_rope_length_l545_545667

theorem initial_rope_length
  (n : ℕ) (p : ℕ) (final_piece_length : ℕ) (shortening : ℕ) (total_pieces : ℕ)
  (h1 : total_pieces = 12) 
  (h2 : p = 3)
  (h3 : shortening = 1)
  (h4 : final_piece_length = 15) :
  n = 192 :=
by
  let original_piece_length := final_piece_length + shortening
  let combined_length_of_three := original_piece_length * p
  let total_initial_length := combined_length_of_three * (total_pieces / p)
  have h5 : original_piece_length = 16 := by rw [h4, h3]; trivial
  have h6 : combined_length_of_three = 48 := by rw [h2, h5]; trivial
  have h7 : total_initial_length = 192 := by rw [h1, h6]; trivial
  exact h7.symm

end initial_rope_length_l545_545667


namespace fifteenth_prime_is_47_l545_545880

open Nat

def is_prime : ℕ → Prop
| 0       := false
| 1       := false
| (n + 2) := (∀ m, 2 ≤ m → m * m ≤ n + 2 → (n + 2) % m ≠ 0)

noncomputable def fifth_prime : ℕ :=
  by exact 11

noncomputable def fifteenth_prime : ℕ :=
  by exact 47

theorem fifteenth_prime_is_47 : ∀ n, is_prime n → List.indexOf n [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47] = 14 → 
  fifteenth_prime = n := sorry

end fifteenth_prime_is_47_l545_545880


namespace g_is_constant_l545_545031

variables {R : Type*} [AddCommGroup R] [Module ℝ R]

-- Given g is nonzero and satisfies the functional equation
axiom g_nonzero : ∃ x : ℝ, g x ≠ 0
axiom functional_eq : ∀ a b : ℝ, g(a + b) + g(a - b) = g a + g b

-- Prove g is a constant function
theorem g_is_constant : ∀ x y : ℝ, g x = g y := by
  sorry

end g_is_constant_l545_545031


namespace solve_for_z_l545_545124

theorem solve_for_z (z : ℂ) (i : ℂ) (h : i^2 = -1) : 3 + 2 * i * z = 5 - 3 * i * z → z = - (2 * i) / 5 :=
by
  intro h_equation
  -- Proof steps will be provided here.
  sorry

end solve_for_z_l545_545124


namespace count_four_digit_numbers_divisible_by_5_end_45_l545_545414

theorem count_four_digit_numbers_divisible_by_5_end_45 : 
  {n : ℕ | 1000 ≤ n ∧ n < 10000 ∧ n % 100 = 45 ∧ n % 5 = 0}.to_finset.card = 90 :=
by
  sorry

end count_four_digit_numbers_divisible_by_5_end_45_l545_545414


namespace probability_even_sum_l545_545850

theorem probability_even_sum :
  let cards := [1, 2, 3, 4];
  let angie := cards[0];
  let bridget := cards[1];
  let carlos := cards[2];
  let diego := cards[3];
  let favorable_outcomes := 
    [(angie, carlos), (bridget, carlos), (carlos, diego), (diego, carlos)] |>.filter (λ p, (p.1 + p.2) % 2 = 0);
    let total_outcomes := cards.permutations().length;
  (favorable_outcomes.length.toReal / total_outcomes.toReal) = (1 / 6 : ℝ) :=
by
  sorry

end probability_even_sum_l545_545850


namespace min_k_period_at_least_15_l545_545111

theorem min_k_period_at_least_15 (a b : ℚ) (h_pos_a : 0 < a) (h_pos_b : 0 < b)
    (h_period_a : ∃ m, a = m / (10^30 - 1))
    (h_period_b : ∃ n, b = n / (10^30 - 1))
    (h_period_ab : ∃ p, (a - b) = p / (10^30 - 1) ∧ 10^15 + 1 ∣ p) :
    ∃ k : ℕ, k = 6 ∧ (∃ q, (a + k * b) = q / (10^30 - 1) ∧ 10^15 + 1 ∣ q) :=
sorry

end min_k_period_at_least_15_l545_545111


namespace count_four_digit_numbers_divisible_by_5_ending_in_45_l545_545423

theorem count_four_digit_numbers_divisible_by_5_ending_in_45 : 
  ∃ n : ℕ, (∀ x : ℕ, x ≥ 1000 ∧ x < 10000 ∧ x % 100 = 45 → (x % 5 = 0 ∧ x % 100 = 45)) 
  ∧ (n = 90) :=
by
  sorry

end count_four_digit_numbers_divisible_by_5_ending_in_45_l545_545423


namespace length_of_each_section_25_l545_545552

theorem length_of_each_section_25 (x : ℝ) 
  (h1 : ∃ x, x > 0)
  (h2 : 1000 / x = 15 / (1 / 2 * 3 / 4))
  : x = 25 := 
  sorry

end length_of_each_section_25_l545_545552


namespace functional_equation_solution_l545_545310

theorem functional_equation_solution (f : ℤ → ℤ) :
  (∀ m n : ℤ, f (f (m + n)) = f m + f n) ↔
  (∃ a : ℤ, ∀ n : ℤ, f n = n + a ∨ f n = 0) :=
sorry

end functional_equation_solution_l545_545310


namespace count_four_digit_numbers_divisible_by_5_end_45_l545_545413

theorem count_four_digit_numbers_divisible_by_5_end_45 : 
  {n : ℕ | 1000 ≤ n ∧ n < 10000 ∧ n % 100 = 45 ∧ n % 5 = 0}.to_finset.card = 90 :=
by
  sorry

end count_four_digit_numbers_divisible_by_5_end_45_l545_545413


namespace gideon_age_in_future_years_l545_545337

-- Definitions based on the conditions
def marbles : ℕ := 100
def given_away_fraction : ℚ := 3 / 4
def remaining_marbles : ℕ := marbles - marbles * given_away_fraction
def multiplied_marbles : ℕ := remaining_marbles * 2
def current_age : ℕ := 45
def years_from_now : ℕ := multiplied_marbles - current_age

-- The theorem to be proven
theorem gideon_age_in_future_years : years_from_now = 5 :=
  by
    have rem_marbles_calc : remaining_marbles = 25 := by sorry
    have mult_marbles_calc : multiplied_marbles = 25 * 2 := by sorry
    have final_calc : multiplied_marbles - current_age = 50 - 45 := by sorry
    exact Nat.sub_eq_of_eq_add (by norm_num)

end gideon_age_in_future_years_l545_545337


namespace valid_combinations_l545_545774

-- Definitions based on conditions
def h : Nat := 4  -- number of herbs
def c : Nat := 6  -- number of crystals
def r : Nat := 3  -- number of negative reactions

-- Theorem statement based on the problem and solution
theorem valid_combinations : (h * c) - r = 21 := by
  sorry

end valid_combinations_l545_545774


namespace concurrent_lines_l545_545479

open EuclideanGeometry

-- Define the circles and their corresponding properties
variables {O O1 O2 A A1 A2 : Point}
variables {r r1 r2 : ℝ}

-- Define the radii conditions
axiom circle_radii_conditions :
  r > 0 ∧ r1 > 0 ∧ r2 > 0 ∧ r > r1 ∧ r > r2

-- Define the tangency conditions
axiom tangency_conditions : 
  -- C1 internally tangent to C at A1
  T (Circle.mk O r) (Circle.mk O1 r1) A1 ∧ 
  -- C2 internally tangent to C at A2
  T (Circle.mk O r) (Circle.mk O2 r2) A2 ∧
  -- C1 and C2 externally tangent to each other at A
  T_e (Circle.mk O1 r1) (Circle.mk O2 r2) A

-- The lines OO1, OO2, and the points A1, and A2
noncomputable def line_O_A := line_through O A
noncomputable def line_O1_A2 := line_through O1 A2
noncomputable def line_O2_A1 := line_through O2 A1

-- The theorem to be proved
theorem concurrent_lines :
  meet_at line_O_A line_O1_A2 line_O2_A1 :=
sorry

end concurrent_lines_l545_545479


namespace rogers_parents_paid_percentage_l545_545118

variables 
  (house_cost : ℝ)
  (down_payment_percentage : ℝ)
  (remaining_balance_owed : ℝ)
  (down_payment : ℝ := down_payment_percentage * house_cost)
  (remaining_balance_after_down : ℝ := house_cost - down_payment)
  (parents_payment : ℝ := remaining_balance_after_down - remaining_balance_owed)
  (percentage_paid_by_parents : ℝ := (parents_payment / remaining_balance_after_down) * 100)

theorem rogers_parents_paid_percentage
  (h1 : house_cost = 100000)
  (h2 : down_payment_percentage = 0.20)
  (h3 : remaining_balance_owed = 56000) :
  percentage_paid_by_parents = 30 :=
sorry

end rogers_parents_paid_percentage_l545_545118


namespace complex_modulus_inequality_l545_545113

theorem complex_modulus_inequality (z : ℂ) : (‖z‖ ^ 2 + 2 * ‖z - 1‖) ≥ 1 :=
by
  sorry

end complex_modulus_inequality_l545_545113


namespace area_of_symmetric_triangle_l545_545560

structure Triangle :=
  (X : Real × Real)
  (Y : Real × Real)
  (Z : Real × Real)
  (is_right_triangle : (Y.1 - X.1) * (Z.2 - X.2) / 2 = 1)

def symmetric_point (P Q R : Real × Real) : Real × Real :=
  let midpoint := ((Q.1 + R.1) / 2, (Q.2 + R.2) / 2)
  (2 * midpoint.1 - P.1, 2 * midpoint.2 - P.2)

def area (A B C : Real × Real) : Real :=
  abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2)) / 2

theorem area_of_symmetric_triangle (T : Triangle) : 
  let X' := symmetric_point T.X T.Y T.Z
  let Y' := symmetric_point T.Y T.X T.Z
  let Z' := symmetric_point T.Z T.X T.Y
  area X' Y' Z' = 3 := sorry

end area_of_symmetric_triangle_l545_545560


namespace no_equal_white_black_divisors_l545_545246

def isWhite (n : ℕ) : Prop :=
  n = 1 ∨ (∃ k : ℕ, 0 < k ∧ (Nat.prime_factors n).length = 2 * k)

def isBlack (n : ℕ) : Prop :=
  ¬isWhite n

def sumOfWhiteDivisors (n : ℕ) : ℕ :=
  (Finset.filter isWhite (Finset.divisors n)).sum id

def sumOfBlackDivisors (n : ℕ) : ℕ :=
  (Finset.filter isBlack (Finset.divisors n)).sum id

theorem no_equal_white_black_divisors (n : ℕ) (hn : 0 < n) :
    sumOfWhiteDivisors n ≠ sumOfBlackDivisors n :=
  sorry

end no_equal_white_black_divisors_l545_545246


namespace right_triangle_area_l545_545675

theorem right_triangle_area (leg1 leg2 hypotenuse : ℕ) (h_leg1 : leg1 = 30)
  (h_hypotenuse : hypotenuse = 34)
  (hypotenuse_sq : hypotenuse * hypotenuse = leg1 * leg1 + leg2 * leg2) :
  (1 / 2 : ℚ) * leg1 * leg2 = 240 := by
  sorry

end right_triangle_area_l545_545675


namespace find_norm_ratio_q_v_l545_545567

variables {V : Type*} [inner_product_space ℝ V]
variables (v w p q : V)

-- Definitions of projections in real inner product spaces
noncomputable def proj (a b : V) : V := (inner a b / inner b b) • b
noncomputable def norm_ratio (a b : V) : ℝ := ∥a∥ / ∥b∥

-- Assumptions
axiom p_proj_v_onto_w : p = proj v w
axiom q_proj_v_onto_p : q = proj v p
axiom norm_ratio_p_v : norm_ratio p v = 3 / 4

-- Theorem (stating the main problem)
theorem find_norm_ratio_q_v :
  norm_ratio q v = (3 / 4) ^ 2 := by sorry

end find_norm_ratio_q_v_l545_545567


namespace Christopher_joggers_eq_80_l545_545777

variable (T A C : ℕ)

axiom Tyson_joggers : T > 0                  -- Tyson bought a positive number of joggers.

axiom Alexander_condition : A = T + 22        -- Alexander bought 22 more joggers than Tyson.

axiom Christopher_condition : C = 20 * T      -- Christopher bought twenty times as many joggers as Tyson.

axiom Christopher_Alexander : C = A + 54     -- Christopher bought 54 more joggers than Alexander.

theorem Christopher_joggers_eq_80 : C = 80 := 
by
  sorry

end Christopher_joggers_eq_80_l545_545777


namespace diff_max_min_on_interval_l545_545623

def f (x : ℝ) := x^3 - 3 * x + 1

theorem diff_max_min_on_interval : 
  let a := -3 in
  let b := 0 in
  (∀ x, x ∈ Set.interval a b → x = a ∨ x = b ∨ (deriv f x = 0 → f '' Set.interval a b ⊆ Set.Icc (f a) (f b))) → (setImage (fun (x:ℝ) => f x) (Set.Icc (-3:ℝ) (0:ℝ))).sup id - 
  (setImage (fun (x:ℝ) => f x) (Set.Icc (-3:ℝ) (0:ℝ))).inf id = 20 := by
  sorry

end diff_max_min_on_interval_l545_545623


namespace power_function_evaluation_l545_545904

theorem power_function_evaluation :
  ∃ (a : ℝ), (∀ (x : ℝ), f x = x ^ a) ∧ f 3 = (Real.sqrt 3) / 3 → f 9 = 1 / 3 :=
by
  sorry

end power_function_evaluation_l545_545904


namespace length_A_l545_545085

open Real

noncomputable def point := ℝ × ℝ

def A : point := (0, 10)
def B : point := (0, 15)
def C : point := (3, 9)
def is_on_line_y_eq_x (P : point) : Prop := P.1 = P.2
def length (P Q : point) := sqrt ((Q.1 - P.1)^2 + (Q.2 - P.2)^2)

theorem length_A'B'_is_correct (A' B' : point)
  (hA' : is_on_line_y_eq_x A')
  (hB' : is_on_line_y_eq_x B')
  (hA'_line : ∃ m b, C.2 = m * C.1 + b ∧ A'.2 = m * A'.1 + b ∧ A.2 = m * A.1 + b)
  (hB'_line : ∃ m b, C.2 = m * C.1 + b ∧ B'.2 = m * B'.1 + b ∧ B.2 = m * B.1 + b)
  : length A' B' = 2.5 * sqrt 2 :=
sorry

end length_A_l545_545085


namespace quadratic_part_of_equation_l545_545153

theorem quadratic_part_of_equation (x: ℝ) :
  (x^2 - 8*x + 21 = |x - 5| + 4) → (x^2 - 8*x + 21) = x^2 - 8*x + 21 :=
by
  intros h
  sorry

end quadratic_part_of_equation_l545_545153


namespace find_n_cosine_l545_545839

theorem find_n_cosine :
  ∃ n : ℤ, 0 ≤ n ∧ n ≤ 180 ∧ real.cos (n * real.pi / 180) = real.cos (845 * real.pi / 180) ∧ n = 125 :=
sorry

end find_n_cosine_l545_545839


namespace squares_per_student_l545_545856

theorem squares_per_student :
  (∀ (bar_squares : ℕ) (gerald_bars : ℕ) (multiplier : ℕ) (students : ℕ), 
    bar_squares = 8 → gerald_bars = 7 → multiplier = 2 → students = 24 → 
    (gerald_bars + gerald_bars * multiplier) * bar_squares / students = 7) :=
by
  intros bar_squares gerald_bars multiplier students
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  norm_num
  sorry

end squares_per_student_l545_545856


namespace compute_modulo_l545_545812

theorem compute_modulo :
  (2023 * 2024 * 2025 * 2026) % 7 = 0 := by
  sorry

end compute_modulo_l545_545812


namespace rods_in_one_mile_l545_545362

/-- Definitions based on given conditions -/
def miles_to_furlongs := 8
def furlongs_to_rods := 40

/-- The theorem stating the number of rods in one mile -/
theorem rods_in_one_mile : (miles_to_furlongs * furlongs_to_rods) = 320 := 
  sorry

end rods_in_one_mile_l545_545362


namespace total_plums_correct_l545_545102

/-- Each picked number of plums. -/
def melanie_picked := 4
def dan_picked := 9
def sally_picked := 3
def ben_picked := 2 * (melanie_picked + dan_picked)
def sally_ate := 2

/-- The total number of plums picked in the end. -/
def total_plums_picked :=
  melanie_picked + dan_picked + sally_picked + ben_picked - sally_ate

theorem total_plums_correct : total_plums_picked = 40 := by
  sorry

end total_plums_correct_l545_545102


namespace arithmetic_sequence_a2015_l545_545375

theorem arithmetic_sequence_a2015 :
  ∀ {a : ℕ → ℤ}, (a 1 = 2 ∧ a 5 = 6 ∧ (∀ n, a (n + 1) = a n + a 2 - a 1)) → a 2015 = 2016 :=
by
  sorry

end arithmetic_sequence_a2015_l545_545375


namespace provisions_duration_l545_545738

theorem provisions_duration
  (boys_initial : ℕ) (days_initial : ℕ) (boys_added : ℕ)
  (total_provisions : ℕ := boys_initial * days_initial)
  (boys_total : ℕ := boys_initial + boys_added) :
  (total_provisions / boys_total) = 20 :=
by
  let provisions := 1500 * 25
  let boys := 1500 + 350
  have div_eq_20 : provisions / boys = 20 := by
    have : provisions = 37500 := rfl
    have : boys = 1850 := rfl
    have quotient : 37500 / 1850 = 20 := by norm_num
    exact quotient
  exact div_eq_20

end provisions_duration_l545_545738


namespace f_linear_1996_l545_545900

theorem f_linear_1996 (f : ℝ → ℝ)
  (h : ∀ x y : ℝ, f (x^3 + y^3) = (x + y) ((f x)^2 - (f x) * (f y) + (f y)^2)) :
  ∀ x : ℝ, f (1996 * x) = 1996 * (f x) :=
by
  sorry

end f_linear_1996_l545_545900


namespace both_A_and_B_are_Gnomes_l545_545960

inductive Inhabitant
| Elf
| Gnome

open Inhabitant

def lies_about_gold (i : Inhabitant) : Prop :=
  match i with
  | Elf => False
  | Gnome => True

def tells_truth_about_others (i : Inhabitant) : Prop :=
  match i with
  | Elf => False
  | Gnome => True

def A_statement : Prop := ∀ i : Inhabitant, lies_about_gold i → i = Gnome
def B_statement : Prop := ∀ i : Inhabitant, tells_truth_about_others i → i = Gnome

theorem both_A_and_B_are_Gnomes (A_statement_true : A_statement) (B_statement_true : B_statement) :
  ∀ i : Inhabitant, (lies_about_gold i ∧ tells_truth_about_others i) → i = Gnome :=
by
  sorry

end both_A_and_B_are_Gnomes_l545_545960


namespace length_AB_length_CD_l545_545534

noncomputable def angle_ABC : ℝ := 40
noncomputable def angle_ACD : ℝ := 50
noncomputable def BC : ℝ := 12
noncomputable def tan_40 : ℝ := Real.tan (Real.pi * 40 / 180)
noncomputable def tan_50 : ℝ := Real.tan (Real.pi * 50 / 180)

theorem length_AB (angle_ABC = 40) (BC = 12) :
  AB ≈ 10.07 :=
by
  let AB := BC * tan_40
  have h1 : AB ≈ 12 * tan_40 := sorry
  have h2 : AB ≈ 10.07 := sorry
  exact h2

theorem length_CD (length_AB (angle_ABC := 40) (BC := 12)) (angle_ACD = 50) :
  CD ≈ 12.00 :=
by
  let AB := 12 * tan_40
  let CD := AB * tan_50
  have h3 : AB ≈ 10.07 := sorry
  have h4 : CD ≈ 10.07 * tan_50 := sorry
  have h5 : CD ≈ 12.00 := sorry
  exact h5

end length_AB_length_CD_l545_545534


namespace slope_of_tangent_line_at_zero_l545_545327

noncomputable def f (x : ℝ) : ℝ := Real.exp x * Real.cos x

theorem slope_of_tangent_line_at_zero : (deriv f 0) = 1 := 
by
  sorry

end slope_of_tangent_line_at_zero_l545_545327


namespace simplify_expression_l545_545613

theorem simplify_expression : 4 * (15 / 5) * (24 / -60) = - (24 / 5) := 
by
  sorry

end simplify_expression_l545_545613


namespace cloak_change_l545_545496

theorem cloak_change (silver_for_cloak1 silver_for_cloak2 : ℕ) (gold_change1 gold_change2 : ℕ) (silver_change : ℕ) :
  silver_for_cloak1 = 20 →
  gold_change1 = 4 →
  silver_for_cloak2 = 15 →
  gold_change2 = 1 →
  ∃ silver_cost_of_cloak : ℕ, 
    silver_cost_of_cloak = (20 - 4) * (5 / 3) →
    silver_change = 10 →
    14 * (5 / 3) - 8 = silver_change :=
by 
  assume h1 h2 h3 h4,
  use 16, 
  sorry

end cloak_change_l545_545496


namespace largest_even_integer_sum_l545_545154

theorem largest_even_integer_sum : 
  let sum_first_25_even := 2 * (25 * 26) / 2
  ∃ n : ℕ, 
  n % 2 = 0 ∧ 
  sum_first_25_even = 5 * n - 20 ∧
  n = 134 :=
by
  let sum_first_25_even := 2 * (25 * 26) / 2
  have h_sum : sum_first_25_even = 650 := by norm_num
  use 134
  split
  · norm_num
  split
  · rw h_sum
    norm_num
  · rfl

end largest_even_integer_sum_l545_545154


namespace vans_for_field_trip_l545_545586

-- Definitions based on conditions
def students := 25
def adults := 5
def van_capacity := 5

-- Calculate total number of people
def total_people := students + adults

-- Calculate number of vans needed
def vans_needed := total_people / van_capacity

-- Theorem statement
theorem vans_for_field_trip : vans_needed = 6 := by
  -- Proof would go here
  sorry

end vans_for_field_trip_l545_545586


namespace part1_final_balance_part2_max_balance_step_l545_545743

def final_balance (initial : ℕ) (transactions : List ℤ) : ℕ :=
  initial + 10_000 * transactions.sum

theorem part1_final_balance : final_balance 70_000 [2, -3, 3.5, -2.5, 4, -1.2, 1, -0.8] = 110_000 :=
by
  sorry

def max_balance_step (initial : ℕ) (transactions : List ℤ) : ℕ :=
  let balances := transactions.scanl (λ acc x, acc + 10_000 * x) initial
  balances.zip (List.range (transactions.length + 1))
  |> List.argmax
  |> Option.map (λ ⟨bal, step⟩ => (step, bal))
  |> Option.getD 0 (0, initial)

theorem part2_max_balance_step : max_balance_step 70_000 [2, -3, 3.5, -2.5, 4, -1.2, 1, -0.8] = (5, 110_000) :=
by
  sorry

end part1_final_balance_part2_max_balance_step_l545_545743


namespace solution_l545_545309

noncomputable def problem_statement (f : ℝ → ℝ) : Prop :=
∀ x y : ℝ, (x + y) * (f x - f y) = (x - y) * f (x + y)

theorem solution (f : ℝ → ℝ) (h : problem_statement f) :
  ∃ a b : ℝ, ∀ x : ℝ, f x = a * x + b * x^2 := by
  sorry

end solution_l545_545309


namespace count_four_digit_numbers_divisible_by_five_ending_45_l545_545430

-- Define the conditions as necessary in Lean
def is_four_digit_number (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

def is_divisible_by_five (n : ℕ) : Prop :=
  n % 5 = 0

def ends_with_45 (n : ℕ) : Prop :=
  n % 100 = 45

-- Statement that there exists 90 such four-digit numbers
theorem count_four_digit_numbers_divisible_by_five_ending_45 : 
  { n : ℕ // is_four_digit_number n ∧ is_divisible_by_five n ∧ ends_with_45 n }.card = 90 :=
sorry

end count_four_digit_numbers_divisible_by_five_ending_45_l545_545430


namespace cyclic_quadrilateral_l545_545577

-- Define the data structures for points, lines and angles

structure Point where
  x : ℝ
  y : ℝ

structure Line where
  p1 : Point
  p2 : Point

def angle (A B C : Point) : ℝ := sorry -- This will represent the angle at B formed by the points A, B and C

def sin_angle (A B C : Point) : ℝ := Real.sin (angle A B C)

structure Quadrilateral where
  A B C D : Point

def convex (abcd : Quadrilateral) : Prop := sorry
def intersection (l1 l2 : Line) : Point := sorry

-- The main theorem statement
theorem cyclic_quadrilateral (A B C D : Point) (O : Point) 
  (h1 : Quadrilateral A B C D)
  (h2 : convex h1)
  (h3 : O = intersection (Line.mk A C) (Line.mk B D))
  (h4 : distance O A * sin_angle A O B + distance O C * sin_angle C O D
      = distance O B * sin_angle B O A + distance O D * sin_angle D O C) : cyclic h1 :=
  sorry

end cyclic_quadrilateral_l545_545577


namespace seeds_in_second_plot_l545_545332

theorem seeds_in_second_plot (S : ℕ) (h1 : 300 seeds were planted in the first plot) 
  (h2 : S seeds were planted in the second plot)
  (h3 : 75 seeds germinated in the first plot) 
  (h4 : 0.4 * S seeds germinated in the second plot) 
  (h5 : 75 + 0.4 * S = 0.31 * (300 + S)) : 
  S = 200 := 
sorry

end seeds_in_second_plot_l545_545332


namespace count_four_digit_numbers_divisible_by_5_ending_in_45_l545_545399

theorem count_four_digit_numbers_divisible_by_5_ending_in_45 : 
  ∃ (n : ℕ), n = 90 ∧ ∀ (x : ℕ), (1000 ≤ x ∧ x < 10000) ∧ (x % 100 = 45) → count x = n :=
sorry

end count_four_digit_numbers_divisible_by_5_ending_in_45_l545_545399


namespace solve_for_n_l545_545717

theorem solve_for_n (n : ℤ) (h : n + (n + 1) + (n + 2) + (n + 3) = 34) : n = 7 :=
by
  sorry

end solve_for_n_l545_545717


namespace reciprocal_of_abs_neg_two_l545_545148

theorem reciprocal_of_abs_neg_two : 1 / |(-2: ℤ)| = (1 / 2: ℚ) := by
  sorry

end reciprocal_of_abs_neg_two_l545_545148


namespace cloak_change_in_silver_l545_545484

theorem cloak_change_in_silver :
  (∀ c : ℤ, (20 = c + 4) → (15 = c + 1)) →
  (5 * g = 3) →
  14 * gold / exchange_rate = 10 := 
sorry

end cloak_change_in_silver_l545_545484


namespace domain_of_g_l545_545826

noncomputable def g (x : ℝ) : ℝ := 1 / ⌊ x^2 - 9 * x + 21 ⌋

theorem domain_of_g :
  { x : ℝ | ∃ y : ℝ, g x = y } = { x : ℝ | x ≤ 4 ∨ x ≥ 5 } :=
by
  sorry

end domain_of_g_l545_545826


namespace joy_pentagon_problem_l545_545073

theorem joy_pentagon_problem :
  let rods := {n | 1 ≤ n ∧ n ≤ 40}.erase 4 |>.erase 9 |>.erase 18 |>.erase 25,
      a := 4, b := 9, c := 18, d := 25,
      valid_rods := rods.filter (fun e => 0 < e ∧ e < 56)
  in valid_rods.card = 51 :=
by
  sorry

end joy_pentagon_problem_l545_545073


namespace triangle_incircle_area_inequality_l545_545074

theorem triangle_incircle_area_inequality
  (a b c R : ℝ)
  (h_positive : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < R)
  (P1 P2 P3 : ℝ)
  (h_areas_pos: 0 < P1 ∧ 0 < P2 ∧ 0 < P3):
  P1 = (1/2) * a * sqrt ((s-a) * s / R^2) ∧ 
  P2 = (1/2) * b * sqrt ((s-b) * s / R^2) ∧
  P3 = (1/2) * c * sqrt ((s-c) * s / R^2) →
  ((R^4)/(P1^2)) + ((R^4)/(P2^2)) + ((R^4)/(P3^2)) ≥ 16 :=
by
  intro h_P_areas
  sorry

end triangle_incircle_area_inequality_l545_545074


namespace find_cost_price_per_item_min_items_type_A_l545_545267

-- Definitions based on the conditions
def cost_A (x : ℝ) (y : ℝ) : Prop := 4 * x + 10 = 5 * y
def cost_B (x : ℝ) (y : ℝ) : Prop := 20 * x + 10 * y = 160

-- Proving the cost price per item of goods A and B
theorem find_cost_price_per_item : ∃ x y : ℝ, cost_A x y ∧ cost_B x y ∧ x = 5 ∧ y = 6 :=
by
  -- This is where the proof would go
  sorry

-- Additional conditions for part (2)
def profit_condition (a : ℕ) : Prop :=
  10 * (a - 30) + 8 * (200 - (a - 30)) - 5 * a - 6 * (200 - a) ≥ 640

-- Proving the minimum number of items of type A purchased
theorem min_items_type_A : ∃ a : ℕ, profit_condition a ∧ a ≥ 100 :=
by
  -- This is where the proof would go
  sorry

end find_cost_price_per_item_min_items_type_A_l545_545267


namespace least_value_of_n_l545_545533

variable {n : ℕ}

def languages (n : ℕ) : Prop :=
  n ≥ 3 ∧ (∀ (P : Finset ℕ), P.card = 3 → ∃ l, ∀ x ∈ P, speaks_language x l) ∧
  (∀ l, ¬ ∀ P : Finset ℕ, P.card ≤ n / 2 → speaks_language P l)

theorem least_value_of_n (n : ℕ) (h_lang : languages n) : n = 8 := 
  sorry

end least_value_of_n_l545_545533


namespace distinct_ways_to_place_digits_l545_545451

theorem distinct_ways_to_place_digits : 
    ∃ n : ℕ, 
    n = 120 ∧ 
    n = nat.factorial 5 := 
by
  sorry

end distinct_ways_to_place_digits_l545_545451


namespace days_with_equal_sundays_and_tuesdays_l545_545757

theorem days_with_equal_sundays_and_tuesdays :
  ∃ d : ℕ, d = 3 ∧ ∀ (start_day : ℕ), 
  (31 / 7 = 4 ∧ 31 % 7 = 3) → 
  (let sundays := 4 + (if start_day = 0 ∨ start_day = 5 then 1 else 0) in
   let tuesdays := 4 + (if start_day = 2 then 1 else 0) in
   sundays = tuesdays → (start_day = 3 ∨ start_day = 4 ∨ start_day = 5)) :=
begin
  sorry -- proof to be filled in later
end

end days_with_equal_sundays_and_tuesdays_l545_545757


namespace seq_15_l545_545865

noncomputable def seq (n : ℕ) : ℕ :=
  if n = 1 then 1 else if n = 2 then 2 else 2 * (n - 1) + 1 -- form inferred from solution

theorem seq_15 : seq 15 = 29 := by
  sorry

end seq_15_l545_545865


namespace trapezoid_triangle_equal_area_l545_545662

theorem trapezoid_triangle_equal_area
  {A B C D E F G O : Type*}
  [metric_space A] [metric_space B] [metric_space C]
  [metric_space D] [metric_space E] [metric_space F]
  [metric_space G] [metric_space O]
   (circle : set A)
   (O : A) -- Center of the circle
   (hO : O ∈ circle)
   (hdiam : ∀ P Q ∈ circle, (metric.distance O P) = (metric.distance O Q))
   (trapezoid : set A)
   (AB : ∀ (P Q : A), P ≠ Q → P ∈ trapezoid → Q ∉ circle)
   (isosceles_triangle : set A)
   (GF_parallel_AB : ∀ (P Q : A), P ∈ circle → Q ∈ circle → P ≠ Q → P ∈ isosceles_triangle → Q ∉ isosceles_triangle → (metric.distance P Q) = (metric.distance G F))
   (GE_parallel_BC : ∀ (P Q : A), P ∈ circle → Q ∈ circle → P ≠ Q → P ∈ isosceles_triangle → Q ∈ trapezoid → (metric.distance P Q) = (metric.distance G E)) :
   let area_trapezoid := area trapezoid in
   let area_triangle := area isosceles_triangle in
   area_trapezoid = area_triangle :=
sorry

end trapezoid_triangle_equal_area_l545_545662


namespace combined_salaries_of_A_B_C_D_l545_545644

theorem combined_salaries_of_A_B_C_D (salaryE : ℕ) (avg_salary : ℕ) (num_people : ℕ)
    (h1 : salaryE = 9000) (h2 : avg_salary = 8800) (h3 : num_people = 5) :
    (avg_salary * num_people) - salaryE = 35000 :=
by
  sorry

end combined_salaries_of_A_B_C_D_l545_545644


namespace find_a_l545_545618

theorem find_a
  (f : ℝ → ℝ) (g : ℝ → ℝ)
  (h_f : ∀ x, f x = (2 * x - 1) / 3 + 2)
  (h_g : ∀ x, g x = 5 - 2 * x)
  (h : ∀ a, f (g a) = 4 → a = 3 / 4) :
  ∃ a : ℝ, f (g a) = 4 ∧ a = 3 / 4 :=
by
  existsi (3 / 4)
  split
  · rw [h_g, h_f]
    sorry  -- Here the proof goes, but we skip it as per the instructions
  · refl

end find_a_l545_545618


namespace prod_mod7_eq_zero_l545_545809

theorem prod_mod7_eq_zero :
  (2023 * 2024 * 2025 * 2026) % 7 = 0 := 
by {
  sorry
}

end prod_mod7_eq_zero_l545_545809


namespace mason_attic_junk_items_l545_545585

theorem mason_attic_junk_items : 
  (useful_percentage valuable_percentage junk_percentage : ℕ) 
  (sold_useful sold_valuable current_useful : ℕ) 
  (initial_useful_percentage initial_valuable_percentage initial_junk_percentage : ℕ) 
  (current_useful_percentage current_valuable_percentage current_junk_percentage : ℕ)
  (total_items : ℕ) 
  (h1 : initial_useful_percentage = 20)
  (h2 : initial_valuable_percentage = 10)
  (h3 : initial_junk_percentage = 70)
  (h4 : sold_useful = 4)
  (h5 : sold_valuable = 3)
  (h6 : current_useful_percentage = 25)
  (h7 : current_valuable_percentage = 15)
  (h8 : current_junk_percentage = 60)
  (h9 : current_useful = 20)
  (h10 : 20 = 0.25 * total_items)
  (h11 : total_items = 80)
: total_items * 60 / 100 = 48 :=
by
  sorry

end mason_attic_junk_items_l545_545585


namespace number_of_connected_subsets_l545_545262

def isConnected (X : Finset ℕ) : Prop :=
  2 ≤ X.card ∧ ∃ m ∈ X, ∃ n ∈ X, m ≠ n ∧ m ∣ n

def countConnectedSubsets (S : Finset ℕ) : ℕ := 
  (S.sublists.filter isConnected).length

theorem number_of_connected_subsets : countConnectedSubsets (Finset.range 11 \ {0}) = 922 :=
  sorry

end number_of_connected_subsets_l545_545262


namespace perpendicular_distance_H_to_plane_EFG_l545_545815

structure Point3D :=
(x : ℝ)
(y : ℝ)
(z : ℝ)

def E : Point3D := ⟨5, 0, 0⟩
def F : Point3D := ⟨0, 3, 0⟩
def G : Point3D := ⟨0, 0, 4⟩
def H : Point3D := ⟨0, 0, 0⟩

def distancePointToPlane (H E F G : Point3D) : ℝ := sorry

theorem perpendicular_distance_H_to_plane_EFG :
  distancePointToPlane H E F G = 1.8 := sorry

end perpendicular_distance_H_to_plane_EFG_l545_545815


namespace largest_sequence_exists_l545_545318

def digits := {d : Nat // 0 < d ∧ d < 10} -- non-zero digits

-- Define N_k as the k-digit number from the digit sequence
def N : (List digits) → Nat
| [] => 0
| (a :: l) => a.val * 10 ^ l.length + N l

theorem largest_sequence_exists :
  ∃ (n : Nat) (a : Fin n.succ → digits), 
    (∀ (k : Nat), 1 ≤ k → k ≤ n → N (a '' {i | i < k}) ∣ N (a '' {i | i < (k + 1)})) ∧
    n = 4 :=
sorry

end largest_sequence_exists_l545_545318


namespace alan_tickets_l545_545185

theorem alan_tickets (a m : ℕ) (h1 : a + m = 150) (h2 : m = 5 * a - 6) : a = 26 :=
by
  sorry

end alan_tickets_l545_545185


namespace area_of_union_subtract_outside_l545_545128

theorem area_of_union_subtract_outside (r : ℝ) (hr : r > 0) :
  let A := metric.sphere (0, r) r,
      B := metric.sphere (r, 0) r,
      C := metric.sphere (0, -r) r,
      D := metric.sphere (-r, 0) r,
      O := metric.sphere (0, 0) (2*r),
      U := A.union (B.union (C.union D))
  in 2 * (measure_theory.measure (U) - (measure_theory.measure (O) - measure_theory.measure (U))) = 8 * (r^2) := 
sorry

end area_of_union_subtract_outside_l545_545128


namespace distinct_expressions_div_l545_545920

theorem distinct_expressions_div {n : ℕ} (h : n ≥ 2) : 
  let P (n : ℕ) : ℕ := 2^(n - 2)
  in (P n) = 2^(n-2) := by
  sorry

end distinct_expressions_div_l545_545920


namespace circle_distance_l545_545303

theorem circle_distance :
  ∃ c : ℝ × ℝ, (x^2 + y^2 = 4*x + 6*y + 3) → (dist c (8, 4) = real.sqrt 37) :=
sorry

end circle_distance_l545_545303


namespace distinct_ways_to_place_digits_l545_545456

theorem distinct_ways_to_place_digits :
  let n := 4 -- number of digits
  let k := 5 -- number of boxes
  (k * (n!)) = 120 := by
  sorry

end distinct_ways_to_place_digits_l545_545456


namespace fuel_calculation_l545_545293

def total_fuel_needed (empty_fuel_per_mile people_fuel_per_mile bag_fuel_per_mile num_passengers num_crew bags_per_person miles : ℕ) : ℕ :=
  let total_people := num_passengers + num_crew
  let total_bags := total_people * bags_per_person
  let total_fuel_per_mile := empty_fuel_per_mile + people_fuel_per_mile * total_people + bag_fuel_per_mile * total_bags
  total_fuel_per_mile * miles

theorem fuel_calculation :
  total_fuel_needed 20 3 2 30 5 2 400 = 106000 :=
by
  sorry

end fuel_calculation_l545_545293


namespace f_n_eq_l545_545609

def f_n (n : ℕ) (z a b : ℝ) : ℝ :=
  z ^ n + a * ∑ k in finset.range n, nat.choose n k * (a - k * b) ^ (k - 1) * (z + k * b) ^ (n - k)

theorem f_n_eq (n : ℕ) (z a b : ℝ) : f_n n z a b = (z + a) ^ n := 
by
  sorry

end f_n_eq_l545_545609


namespace distance_from_focus_to_asymptotes_l545_545006

def parabola_focus {x y : ℝ} (h : y^2 = 4 * x) : (ℝ × ℝ) := (1, 0)

def hyperbola_asymptotes {x y : ℝ} (h : x^2 - y^2 / 2 = 1) : (ℝ → ℝ) × (ℝ → ℝ) :=
  (λ x, x * Real.sqrt 2, λ x, -x * Real.sqrt 2)

noncomputable def distance_from_point_to_line {a b c : ℝ} {px py : ℝ} : ℝ :=
  abs (a * px + b * py + c) / Real.sqrt (a^2 + b^2)

theorem distance_from_focus_to_asymptotes :
  let p := parabola_focus (by sorry)
  let (l1, l2) := hyperbola_asymptotes (by sorry)
  distance_from_point_to_line 1 (-Real.sqrt 2) 0 1 0 = Real.sqrt 6 / 3 ∧
  distance_from_point_to_line 1 (Real.sqrt 2) 0 1 0 = Real.sqrt 6 / 3 :=
by sorry

end distance_from_focus_to_asymptotes_l545_545006


namespace find_m_n_difference_l545_545029

theorem find_m_n_difference (x y m n : ℤ)
  (hx : x = 2)
  (hy : y = -3)
  (hm : x + y = m)
  (hn : 2 * x - y = n) :
  m - n = -8 :=
by {
  sorry
}

end find_m_n_difference_l545_545029


namespace prop_two_prop_three_correct_answers_l545_545032

-- Define the notions of parallel and perpendicular for lines and planes
axiom parallel (x y : Type) : Prop
axiom perpendicular (x y : Type) : Prop

-- Propositions corresponding to the conditions and statements
axiom m_parallel_alpha (m α : Type) : parallel m α
axiom n_parallel_beta (n β : Type) : parallel n β
axiom m_parallel_n (m n : Type) : parallel m n
axiom alpha_perp_beta (α β : Type) : perpendicular α β
axiom m_perp_n (m n : Type) : perpendicular m n
axiom m_perp_alpha (m α : Type) : perpendicular m α
axiom m_perp_beta (m β : Type) : perpendicular m β

-- The propositions we need to prove
theorem prop_two (α β m : Type) (h1 : perpendicular α β) (h2 : parallel m α) : perpendicular m β :=
by sorry

theorem prop_three (α β m : Type) (h1 : perpendicular m β) (h2 : parallel β α) : perpendicular α β :=
by sorry

-- Equivalence of the problem statement to the output of the solution
theorem correct_answers (α β m n : Type)
  (h1 : perpendicular α β) (h2 : parallel m α)
  (h3 : perpendicular m β) (h4 : parallel β α) :
  prop_two α β m h1 h2 ∧ prop_three α β m h3 h4 :=
by split; sorry

end prop_two_prop_three_correct_answers_l545_545032


namespace average_of_tenths_and_thousandths_l545_545238

theorem average_of_tenths_and_thousandths :
  (0.4 + 0.005) / 2 = 0.2025 :=
by
  -- We skip the proof here
  sorry

end average_of_tenths_and_thousandths_l545_545238


namespace common_difference_arithmetic_seq_geometric_seq_l545_545090

theorem common_difference_arithmetic_seq_geometric_seq
  (a b c : ℝ)
  (h_pos : 0 < a ∧ 0 < b ∧ 0 < c)
  (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c)
  (h_geom_seq : b = a * c)
  (x : ℝ)
  (hx : x = log c a) :
  let d := (1 / (x + 1)) - x in
  d = (1 / x + 1) - 1 / (x + 1) ->
  d = (2 * x + 1) / (x * x + x) :=
sorry

end common_difference_arithmetic_seq_geometric_seq_l545_545090


namespace largest_of_five_consecutive_even_integers_sum_l545_545159

theorem largest_of_five_consecutive_even_integers_sum (n : ℤ) :
  (∑ i in finset.range 25, (2 * (i + 1))) = 650 ∧
  (∑ i in finset.range 5, (n - 8 + 2 * i)) = 650 →
  n = 134 :=
by 
  sorry

end largest_of_five_consecutive_even_integers_sum_l545_545159


namespace integer_solutions_count_l545_545019

theorem integer_solutions_count :
  (∃ (pairs : Finset (ℕ × ℕ)), pairs.card = 15 ∧
    ∀ (pair : ℕ × ℕ), pair ∈ pairs ↔ (∃ x y, pair = (x, y) ∧ (Nat.sqrt x + Nat.sqrt y = 14))) :=
by
  sorry

end integer_solutions_count_l545_545019


namespace fraction_red_marbles_after_doubling_l545_545474

theorem fraction_red_marbles_after_doubling (x : ℕ) (h : x > 0) :
  let blue_fraction : ℚ := 3 / 5
  let red_fraction := 1 - blue_fraction
  let initial_blue_marbles := blue_fraction * x
  let initial_red_marbles := red_fraction * x
  let new_red_marbles := 2 * initial_red_marbles
  let new_total_marbles := initial_blue_marbles + new_red_marbles
  let new_red_fraction := new_red_marbles / new_total_marbles
  new_red_fraction = 4 / 7 :=
sorry

end fraction_red_marbles_after_doubling_l545_545474


namespace vector_decomposition_unique_l545_545913

variable {m : ℝ}
def a : ℝ × ℝ := (1, 2)
def b : ℝ × ℝ := (m - 1, m + 3)

theorem vector_decomposition_unique (m : ℝ) : (m + 3 ≠ 2 * (m - 1)) ↔ (m ≠ 5) := 
sorry

end vector_decomposition_unique_l545_545913


namespace sum_of_valid_n_l545_545829

noncomputable def digit_product (n : ℕ) : ℕ :=
if n < 10 then n else (n / 10) * (n % 10)

theorem sum_of_valid_n : ∑ n in Finset.filter (λ n, digit_product n = n^2 - 15*n - 27) (Finset.range 100), n = 17 :=
by
  sorry

end sum_of_valid_n_l545_545829


namespace minimum_walking_distance_l545_545656

theorem minimum_walking_distance :
  ∃ d : ℕ, d = 410 ∧
    (∀ t1 t2 : ℕ, 1 ≤ t1 ∧ t1 < t2 ∧ t2 ≤ 10 → 
      (∑ n in finset.range (t2 - t1), 10 * (n + 1)) + (∑ d in finset.range (t2 - t1 - 1), 10 * (d + 1)) + d = 410) := 
sorry

end minimum_walking_distance_l545_545656


namespace find_incorrect_props_l545_545378

-- Define the propositions
def prop1 : Prop := ∀ (a b : Vector ℝ 3), has_common_endpoint a b -> collinear a b
def prop2 : Prop := ∀ (a b : Vector ℝ 3), comparable_magnitudes a b
def prop3 : Prop := ∀ (λ : ℝ) (a : Vector ℝ 3), λ • a = 0 -> λ = 0
def prop4 : Prop := ∀ (λ μ : ℝ) (a b : Vector ℝ 3), λ • a = μ • b -> collinear a b

-- Proof to verify the number of incorrect propositions
theorem find_incorrect_props : 
  (¬ prop1) ∧ prop2 ∧ (¬ prop3) ∧ (¬ prop4) -> 
  Σ n, (n = 3) :=
by
  sorry

end find_incorrect_props_l545_545378


namespace coin_toss_sequence_count_l545_545530

theorem coin_toss_sequence_count : 
    ∃ n : ℕ, 
      n = 4620 
    ∧ (∃ S : list char, 
        S.length = 20
        ∧ count_subseqs S "HH" = 3
        ∧ count_subseqs S "HT" = 4
        ∧ count_subseqs S "TH" = 6
        ∧ count_subseqs S "TT" = 6) :=
sorry

-- Definitions of count_subseqs and necessary predicates would also need to be provided

def count_subseqs (l : list char) (subseq : string) : ℕ :=
  sorry


end coin_toss_sequence_count_l545_545530


namespace polygon_projection_area_l545_545130

theorem polygon_projection_area (S : ℝ) (φ : ℝ) (polygon : Set (Point (ℝ ^ 3))) (planeP : Set (Point (ℝ ^ 3))) 
(angle_condition : ∀ (n : ℕ), 0 ≤ φ ∧ φ ≤ π / 2):
  let projection_area := S * cos φ in
  projection_area = S * cos φ :=
by
  sorry

end polygon_projection_area_l545_545130


namespace isosceles_triangle_AD_BC_ratio_l545_545542

theorem isosceles_triangle_AD_BC_ratio (A B C D : Type*)
  [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D]
  (angle_ABC_BCD_80 : ∀ (θ : ℝ), θ = 80 * (Real.pi / 180))
  (AB_eq_AC : dist A B = dist A C)
  (BC_eq_CD : dist B C = dist C D)
  (BC_length : ∀ (s : ℝ), dist B C = s) :
  ∃ (x : ℝ), x = 2 * Real.sqrt(2 * (1 - Real.cos(20 * (Real.pi / 180)))) :=
by
  sorry

end isosceles_triangle_AD_BC_ratio_l545_545542


namespace prove_value_of_m_l545_545878

-- Given conditions
def m : ℝ
axiom imaginary_unit (i : ℂ) : i ^ 2 = -1
axiom condition (h : (1 - 2 * (complex.i : ℂ)) / ((m : ℂ) - complex.i) > 0)

/-- Prove that m = 1/2 under the given conditions -/
theorem prove_value_of_m (i : ℂ) [imaginary_unit i] (h : (1 - 2 * i) / ((m : ℂ) - i) > 0) : m = 1/2 :=
  sorry

end prove_value_of_m_l545_545878


namespace sequence_identical_l545_545121

noncomputable def a (n : ℕ) : ℝ :=
  (1 / (2 * Real.sqrt 3)) * ((2 + Real.sqrt 3)^n - (2 - Real.sqrt 3)^n)

theorem sequence_identical (n : ℕ) :
  a (n + 1) = (a n + a (n + 2)) / 4 :=
by
  sorry

end sequence_identical_l545_545121


namespace marie_finishes_fourth_task_at_11_40_am_l545_545584

-- Define the given conditions
def start_time : ℕ := 7 * 60 -- start time in minutes from midnight (7:00 AM)
def second_task_end_time : ℕ := 9 * 60 + 20 -- end time of second task in minutes from midnight (9:20 AM)
def num_tasks : ℕ := 4 -- four tasks
def task_duration : ℕ := (second_task_end_time - start_time) / 2 -- duration of one task

-- Define the goal to prove: the end time of the fourth task
def fourth_task_finish_time : ℕ := second_task_end_time + 2 * task_duration

theorem marie_finishes_fourth_task_at_11_40_am : fourth_task_finish_time = 11 * 60 + 40 := by
  sorry

end marie_finishes_fourth_task_at_11_40_am_l545_545584


namespace magic_shop_change_l545_545489

theorem magic_shop_change :
  (∀ (cloak : Type), ∃ price_gold price_silver1 change_gold1 price_silver2 change_gold2, 
  price_silver1 = 20 ∧ change_gold1 = 4 ∧ 
  price_silver2 = 15 ∧ change_gold2 = 1 ∧ 
  price_gold = 14 ∧ 
  ∀ change_silver, 
    (20 - 4) * change_silver = 15 - 1 → -- Relation derived from the conditions
    (14 - (15 - 1) * change_silver / (20 - 4)) * change_silver = 10) := 
sorry

end magic_shop_change_l545_545489


namespace max_imaginary_part_l545_545818

-- Define the polynomial equation as a condition
def polynomial_eqn (z : ℂ) : Prop := z^12 - z^9 + z^6 - z^3 + 1 = 0

-- Define theta in terms of its range
def theta (θ : ℝ) : Prop := -90 ≤ θ ∧ θ ≤ 90

-- The mathematical statement to prove
theorem max_imaginary_part (z : ℂ) (θ : ℝ) (h1 : polynomial_eqn z) (h2 : θ = 84) : 
  ∃ (θ : ℝ), θ = 84 ∧ -90 ≤ θ ∧ θ ≤ 90 ∧ abs (z.imag) = sin θ :=
sorry

end max_imaginary_part_l545_545818


namespace ada_initial_seat_is_2_l545_545122

theorem ada_initial_seat_is_2 (bees_moved_left : Int) (ceci_moved_right : Int) (dee_eddie_swapped_twice : Bool) 
  (ada_returned_to_end_seat : Bool) : bees_moved_left = -1 → ceci_moved_right = 2 → dee_eddie_swapped_twice = True → ada_returned_to_end_seat = True → ada_initial_seat = 2 :=
by
  -- Specific details of the movements
  have h_bea := bees_moved_left
  have h_ceci := ceci_moved_right
  have h_dee_eddie := dee_eddie_swapped_twice
  have h_ada := ada_returned_to_end_seat
  
  -- Total displacement
  let total_displacement : Int := h_bea + h_ceci + (if h_dee_eddie then 0 else 0)
  have h_total_displacement := total_displacement
  
  -- Check if the total displacement and end seat condition imply Ada's initial seat was 2
  assumption sorry

end ada_initial_seat_is_2_l545_545122


namespace total_amount_paid_l545_545053

def original_price_per_card : Int := 12
def discount_per_card : Int := 2
def number_of_cards : Int := 10

theorem total_amount_paid :
  original_price_per_card - discount_per_card * number_of_cards = 100 :=
by
  sorry

end total_amount_paid_l545_545053


namespace ball_reaches_ground_l545_545135

-- Define the initial height equation of the ball
def height_eq (t : ℝ) : ℝ := -3.7 * t^2 + 4 * t + 8

-- State the theorem with the given conditions and correct answer
theorem ball_reaches_ground :
  ∃ t : ℝ, t > 0 ∧ height_eq t = 0 ∧ t = 78 / 37 :=
begin
  sorry
end

end ball_reaches_ground_l545_545135


namespace multiplication_of_decimals_l545_545308

def d1 := (4 * 10 ^ (-1) : ℝ)
def d2 := (6 * 10 ^ (-1) : ℝ)

theorem multiplication_of_decimals :
  d1 * d2 = 0.24 :=
sorry

end multiplication_of_decimals_l545_545308


namespace find_m_range_l545_545364

variable (f : ℝ → ℝ) (g : ℝ → ℝ)

-- Given conditions
axiom h1 : ∃ (k b : ℝ), (f = λ x, k * x + b)
axiom h2 : f 0 = 1
axiom h3 : f 1 = 3
axiom h4 : ∀ x, g x = 2^(f x)

-- Statement to be proven
theorem find_m_range (m : ℝ) : g (m^2 - 2) < g m ↔ -1 < m ∧ m < 2 :=
sorry

end find_m_range_l545_545364


namespace find_a_8_l545_545537

-- Define the arithmetic sequence and its sum formula.
def arithmetic_sequence (a : ℕ → ℕ) (d : ℕ) :=
  ∀ n, a (n + 1) = a n + d

-- Define the sum of the first 'n' terms in the arithmetic sequence.
def sum_of_first_n_terms (S : ℕ → ℕ) (a : ℕ → ℕ) (n : ℕ) :=
  S n = n * (a 1 + a n) / 2

-- Given conditions
def S_15_eq_90 (S : ℕ → ℕ) : Prop := S 15 = 90

-- Prove that a_8 is 6
theorem find_a_8 (S : ℕ → ℕ) (a : ℕ → ℕ) (d : ℕ)
  (h1 : arithmetic_sequence a d) (h2 : sum_of_first_n_terms S a 15)
  (h3 : S_15_eq_90 S) : a 8 = 6 :=
sorry

end find_a_8_l545_545537


namespace total_books_correct_l545_545556

-- Define the number of books each person has
def booksKeith : Nat := 20
def booksJason : Nat := 21
def booksMegan : Nat := 15

-- Define the total number of books they have together
def totalBooks : Nat := booksKeith + booksJason + booksMegan

-- Prove that the total number of books is 56
theorem total_books_correct : totalBooks = 56 := by
  sorry

end total_books_correct_l545_545556


namespace sequence_term_l545_545163

theorem sequence_term (S : ℕ → ℕ) (a : ℕ → ℕ) (n : ℕ) (hn : n > 0)
  (hSn : ∀ n, S n = n^2)
  (hrec : ∀ n, n > 1 → a n = S n - S (n-1)) :
  a n = 2 * n - 1 := by
  -- Base case
  cases n with
  | zero => contradiction  -- n > 0 implies n ≠ 0
  | succ n' =>
    cases n' with
    | zero => sorry  -- When n = 0 + 1 = 1, we need to show a 1 = 2 * 1 - 1 = 1 based on given conditions
    | succ k => sorry -- When n = k + 1, we use the provided recursive relation to prove the statement

end sequence_term_l545_545163


namespace quotient_of_34_div_7_l545_545225

theorem quotient_of_34_div_7 :
  ∃ A : ℤ, 34 = 7 * A + 6 ∧ A = 4 :=
begin
  let A := 4,
  use A,
  split,
  { simp, },
  { refl, }
end

end quotient_of_34_div_7_l545_545225


namespace min_value_of_quadratic_expression_l545_545346

theorem min_value_of_quadratic_expression (x y z : ℝ) (h : x + y + z = 1) :
  ∃ (u : ℝ), (2 * x^2 + 3 * y^2 + z^2 = u) ∧ u = 6 / 11 :=
sorry

end min_value_of_quadratic_expression_l545_545346


namespace quadratic_to_vertex_form_l545_545659

theorem quadratic_to_vertex_form (x : ℝ) :
  let y := 2 * x^2 - 12 * x - 12 in
  ∃ (a m n : ℝ), y = a * (x - m)^2 + n ∧ a = 2 ∧ m = 3 ∧ n = -30 :=
by
  sorry

end quadratic_to_vertex_form_l545_545659


namespace B_join_months_after_A_l545_545770

-- Definitions based on conditions
def capitalA (monthsA : ℕ) : ℕ := 3500 * monthsA
def capitalB (monthsB : ℕ) : ℕ := 9000 * monthsB

-- The condition that profit is in ratio 2:3 implies the ratio of their capitals should equal 2:3
def ratio_condition (x : ℕ) : Prop := 2 * (capitalB (12 - x)) = 3 * (capitalA 12)

-- Main theorem stating that B joined the business 5 months after A started
theorem B_join_months_after_A : ∃ x, ratio_condition x ∧ x = 5 :=
by
  use 5
  -- Proof would go here
  sorry

end B_join_months_after_A_l545_545770


namespace tangent_circle_inequality_l545_545992

variable {A B C : Type} [metric_space A] [metric_space B] [metric_space C]

variables {BC CA AB : ℝ} (p r t : ℝ)

def semiperimeter (BC CA AB : ℝ) : ℝ := (BC + CA + AB) / 2

theorem tangent_circle_inequality
  (h_triangle : ∃ (A B C : Type) [metric_space A] [metric_space B] [metric_space C], True) 
  (h_p : p = semiperimeter BC CA AB) 
  (h_r : r > 0)
  (h_semi_circles : True)
  (h_tangent_circle : True) :
  (p / 2) < t ∧ t ≤ (p / 2) + (1 - (Real.sqrt 3 / 2)) * r := by
  sorry

end tangent_circle_inequality_l545_545992


namespace coplanar_vectors_set_B_l545_545446

variables {V : Type*} [AddCommGroup V] [Module ℝ V] 
variables (a b c : V)

theorem coplanar_vectors_set_B
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  ∃ (k₁ k₂ : ℝ), 
    k₁ • (2 • a + b) + k₂ • (a + b + c) = 7 • a + 5 • b + 3 • c :=
by { sorry }

end coplanar_vectors_set_B_l545_545446


namespace fencing_cost_is_correct_l545_545635

def length : ℕ := 60
def cost_per_meter : ℕ := 27 -- using the closest integer value to 26.50
def breadth (l : ℕ) : ℕ := l - 20
def perimeter (l b : ℕ) : ℕ := 2 * l + 2 * b
def total_cost (P : ℕ) (c : ℕ) : ℕ := P * c

theorem fencing_cost_is_correct :
  total_cost (perimeter length (breadth length)) cost_per_meter = 5300 :=
  sorry

end fencing_cost_is_correct_l545_545635


namespace remainder_of_2x_plus_3uy_l545_545879

theorem remainder_of_2x_plus_3uy (x y u v : ℤ) (hxy : x = u * y + v) (hv : 0 ≤ v) (hv_ub : v < y) :
  (if 2 * v < y then (2 * v % y) else ((2 * v % y) % -y % y)) = 
  (if 2 * v < y then 2 * v else 2 * v - y) :=
by {
  sorry
}

end remainder_of_2x_plus_3uy_l545_545879


namespace alan_tickets_l545_545187

variables (A M : ℕ)

def condition1 := A + M = 150
def condition2 := M = 5 * A - 6

theorem alan_tickets : A = 26 :=
by
  have h1 : condition1 A M := sorry
  have h2 : condition2 A M := sorry
  sorry

end alan_tickets_l545_545187


namespace min_value_f_on_interval_l545_545955

noncomputable def f (x θ : Real) : Real :=
  sqrt 3 * Real.sin (2 * x + θ) + Real.cos (2 * x + θ)

theorem min_value_f_on_interval :
  ∃ θ : Real, 0 < θ ∧ θ < π ∧ (∀ x ∈ Set.Icc (-π / 4) (π / 6), (f x θ) ≥ - (sqrt 3)) :=
by
  sorry

end min_value_f_on_interval_l545_545955


namespace misha_odd_rolls_probability_l545_545589

def S : ℕ → ℕ
| 0 := 1
| 1 := 6
| 2 := 36
| n+3 := 6 * S (n+2) - S n

noncomputable def P : ℚ :=
(∑ n in _root_.finset.range ∞, (S (2 * n) : ℚ) / 6 ^ (2 * n + 3))

theorem misha_odd_rolls_probability 
  (H_nonzero : ∑' (n : ℕ), (S (2 * n) : ℚ) / 6 ^ (2 * n + 3) ≠ 0) :
  P = 216 / 431 :=
by {
    sorry
}

end misha_odd_rolls_probability_l545_545589


namespace cube_with_holes_l545_545289

-- Definitions and conditions
def edge_length_cube : ℝ := 4
def side_length_hole : ℝ := 2
def depth_hole : ℝ := 1
def number_of_holes : ℕ := 6

-- Prove that the total surface area including inside surfaces is 144 square meters
def total_surface_area_including_inside_surfaces : ℝ :=
  let original_surface_area := 6 * (edge_length_cube ^ 2)
  let area_removed_per_hole := side_length_hole ^ 2
  let area_exposed_inside_per_hole := 2 * (side_length_hole * depth_hole) + area_removed_per_hole
  original_surface_area - number_of_holes * area_removed_per_hole + number_of_holes * area_exposed_inside_per_hole

-- Prove that the total volume of material removed is 24 cubic meters
def total_volume_removed : ℝ :=
  number_of_holes * (side_length_hole ^ 2 * depth_hole)

theorem cube_with_holes :
  total_surface_area_including_inside_surfaces = 144 ∧ total_volume_removed = 24 :=
by
  sorry

end cube_with_holes_l545_545289


namespace perimeter_of_triangle_eq_28_l545_545143

-- Definitions of conditions
variables (p : ℝ)
def inradius : ℝ := 2.0
def area : ℝ := 28

-- Main theorem statement
theorem perimeter_of_triangle_eq_28 : p = 28 :=
  by
  -- The proof is omitted
  sorry

end perimeter_of_triangle_eq_28_l545_545143


namespace probability_product_positive_of_independent_selection_l545_545205

theorem probability_product_positive_of_independent_selection :
  let I := set.Icc (-30 : ℝ) (15 : ℝ)
  let P := (λ (x y : ℝ), x ∈ I ∧ y ∈ I ∧ x * y > 0)
  (Prob { x : ℝ × ℝ | P x.1 x.2 } :
    ProbabilitySpace (I × I)) = 5 / 9 :=
by
  sorry

end probability_product_positive_of_independent_selection_l545_545205


namespace cloak_change_l545_545497

theorem cloak_change (silver_for_cloak1 silver_for_cloak2 : ℕ) (gold_change1 gold_change2 : ℕ) (silver_change : ℕ) :
  silver_for_cloak1 = 20 →
  gold_change1 = 4 →
  silver_for_cloak2 = 15 →
  gold_change2 = 1 →
  ∃ silver_cost_of_cloak : ℕ, 
    silver_cost_of_cloak = (20 - 4) * (5 / 3) →
    silver_change = 10 →
    14 * (5 / 3) - 8 = silver_change :=
by 
  assume h1 h2 h3 h4,
  use 16, 
  sorry

end cloak_change_l545_545497


namespace product_of_five_integers_l545_545329

theorem product_of_five_integers (E F G H I : ℚ)
  (h1 : E + F + G + H + I = 110)
  (h2 : E / 2 = F / 3 ∧ F / 3 = G * 4 ∧ G * 4 = H * 2 ∧ H * 2 = I - 5) :
  E * F * G * H * I = 623400000 / 371293 := by
  sorry

end product_of_five_integers_l545_545329


namespace find_coordinates_l545_545761

noncomputable def point_satisfies_conditions (x y : ℝ) : Prop :=
  (y = 8) ∧ 
  (real.sqrt (x^2 + 64) = 14) ∧ 
  (real.sqrt ((x - 3)^2 + 1) = 12) ∧ 
  (x > 3)

theorem find_coordinates :
  (∃ (x y : ℝ), point_satisfies_conditions x y) →
  (∃ (x : ℝ), x = 3 + real.sqrt 143 ∧ y = 8) :=
by sorry

end find_coordinates_l545_545761


namespace geo_seq_sum_l545_545544

theorem geo_seq_sum 
  (a : ℕ → ℝ) (a₁ : ℝ) (q : ℝ) 
  (h_geom : ∀ n : ℕ, a n = a₁ * q^(n - 1))
  (h_a₁ : a 1 = 3)
  (h_a₄ : a 4 = 24)
  (h_q : q^3 = 8) :
  a 3 + a 4 + a 5 = 84 := 
by
  have h_q : q = 2 := sorry
  have h_a3 : a 3 = a₁ * q^2 := sorry
  have h_a3_calculated : a 3 = 12 := sorry
  have h_a5 : a 5 = a₁ * q^4 := sorry
  have h_a5_calculated : a 5 = 48 := sorry
  calc a 3 + a 4 + a 5 
    = 12 + 24 + 48 : by sorry
    = 84 : by sorry

end geo_seq_sum_l545_545544


namespace group_of_4_l545_545978

   variable {α : Type} -- α represents the type of individuals
   variable (knows : α → α → Prop) -- knows a b means a knows b

   -- Definition: in any group of 4 individuals, one knows all others
   def condition (V : Finset α) : Prop :=
     V.card = 4 → ∃ (a ∈ V), ∀ (b ∈ V), a ≠ b → knows a b

   -- Theorem: in any group of 4 people, there is always one person who knows all the other members
   theorem group_of_4 (V : Finset α) (h : condition knows V) : 
     ∃ (a ∈ V), ∀ (b ∈ V), a ≠ b → knows a b :=
     by
     sorry
   
end group_of_4_l545_545978


namespace induction_problem_l545_545112

open BigOperators

theorem induction_problem (n : ℕ) (h : n > 0) :
  ∑ i in Finset.range n, (i + 1) * (i + 2) * (i + 3) = (n * (n + 1) * (n + 2) * (n + 3)) / 4 := 
by
  sorry

end induction_problem_l545_545112


namespace triangle_properties_l545_545059
-- Importing all necessary libraries

-- Declaring necessary variables and using the given conditions
variables {a b c : ℝ}
variables {A B C : ℝ}
variables {m n : ℝ × ℝ}

-- Assuming the conditions
def conditions : Prop :=
  (a * sin B) = c ∧
  b = sqrt 3 ∧
  m = (2, c) ∧
  n = (b / 2 * cos C - sin A, cos B) ∧
  (m.1 * n.1 + m.2 * n.2 = 0)

-- Stating the theorem to prove
theorem triangle_properties (h : conditions) :
  B = π / 3 ∧
  (1 / 2) * a * c * sin B = 3 * sqrt 3 / 4 ∧
  a = sqrt 3 ∧
  c = sqrt 3 :=
by
  sorry

end triangle_properties_l545_545059


namespace solve_for_x_l545_545439

theorem solve_for_x (x y : ℚ) (h1 : 3 * x - y = 7) (h2 : x + 3 * y = 2) : x = 23 / 10 :=
by
  -- Proof is omitted
  sorry

end solve_for_x_l545_545439


namespace class_A_scores_more_uniform_l545_545737

-- Define the variances of the test scores for classes A and B
def variance_A := 13.2
def variance_B := 26.26

-- Theorem: Prove that the scores of the 10 students from class A are more uniform than those from class B
theorem class_A_scores_more_uniform :
  variance_A < variance_B :=
  by
    -- Assume the given variances and state the comparison
    have h : 13.2 < 26.26 := by sorry
    exact h

end class_A_scores_more_uniform_l545_545737


namespace nearest_higher_whole_number_l545_545795

open Real

theorem nearest_higher_whole_number :
  let term1 := sqrt 9
  let term2 := sqrt 16
  let term3 := 15 / 7
  let term4 := 33 / 8
  ceil (term1 + term2 + term3 + term4) = 14 :=
by
  let term1 := sqrt 9
  let term2 := sqrt 16
  let term3 := 15 / 7
  let term4 := 33 / 8
  have h_eq1 : term1 = 3 := by sorry
  have h_eq2 : term2 = 4 := by sorry
  have h_eq3 : term3 = 15 / 7 := by sorry
  have h_eq4 : term4 = 33 / 8 := by sorry
  let sum := term1 + term2 + term3 + term4
  have h_sum : sum = 13 + 15 / 56 := by sorry
  show ceil sum = 14
  calc
    ceil sum = ceil (13 + 15 / 56) : by sorry
    ... = 14 : by sorry

end nearest_higher_whole_number_l545_545795


namespace line_equation_solution_l545_545902

noncomputable def line_equation (P : ℝ × ℝ) (A : ℝ × ℝ) (B : ℝ × ℝ) : Prop :=
  ∃ (l : ℝ → ℝ), (l P.fst = P.snd) ∧ (∀ (x : ℝ), l x = 4 * x - 2) ∨ (∀ (x : ℝ), x = 1)

theorem line_equation_solution : line_equation (1, 2) (2, 3) (0, -5) :=
sorry

end line_equation_solution_l545_545902


namespace smallest_n_satisfies_l545_545990

def floor (x : Real) : Int := Int.floor x

theorem smallest_n_satisfies (n : ℕ) :
  (∑ k in Finset.range n.succ, floor (k / 15)) > 2011 → n = 253 :=
sorry

end smallest_n_satisfies_l545_545990


namespace prime_is_good_iff_not_2_l545_545782

open Nat

def is_good_prime (p : ℕ) [fact p.prime] : Prop :=
  (∃ k > 1, ∃ (n : Fin k → ℕ), (∀ i : Fin k, n i ≥ (p+1)/2) ∧ (∀ i : Fin k, (p^(n i) - 1) % n ((i+1) % k) = 0 ∧ Nat.coprime ((p^(n i) - 1) / n ((i+1) % k)) (n ((i+1) % k))))

theorem prime_is_good_iff_not_2 (p : ℕ) [fact p.prime] : is_good_prime p ↔ p ≠ 2 := 
by {
  sorry
}

end prime_is_good_iff_not_2_l545_545782


namespace cloak_change_14_gold_coins_l545_545510

def exchange_rate (silver gold : ℕ) : Prop :=
  ∃ c : ℕ, (20 - 4) * c = silver ∧ (15 - 1) * c = silver

def cloak_purchase (paid_gold received_silver : ℕ) : Prop :=
  let exchange_rate := (5 * 14) / 3 in
  received_silver = 2 * exchange_rate

theorem cloak_change_14_gold_coins :
  exchange_rate 16 3 →
  exchange_rate 14 1 →
  cloak_purchase 14 10 := sorry

end cloak_change_14_gold_coins_l545_545510


namespace proof_problem_l545_545313

def digit_sum (n : ℕ) : ℕ := n.digits.sum

def problem_conditions (A : Fin 19 → ℕ) : Prop := 
  (∑ i, A i = 2017) ∧ (∀ i, digit_sum (A i) = digit_sum (A 0))

theorem proof_problem 
  (A1 : Fin 19 → ℕ) 
  (A2 : Fin 19 → ℕ) 
  (hA1 : A1 = ![1000, 1000, 10, 7, 1])
  (hA2 : A2 = ![136, 28, 37, 46, 55, 64, 73, 82, 91, 100, 109, 118, 127, 136, 145, 154, 163, 172, 181, 190]) 
  : problem_conditions A1 ∧ problem_conditions A2 := 
sorry

end proof_problem_l545_545313


namespace correct_option_is_A_l545_545721

-- Define the options as terms
def optionA (x : ℝ) := (1/2) * x - 5 * x = 18
def optionB (x : ℝ) := (1/2) * x > 5 * x - 1
def optionC (y : ℝ) := 8 * y - 4
def optionD := 5 - 2 = 3

-- Define a function to check if an option is an equation
def is_equation (option : Prop) : Prop :=
  ∃ (x : ℝ), option = ((1/2) * x - 5 * x = 18)

-- Prove that optionA is the equation
theorem correct_option_is_A : is_equation (optionA x) :=
by
  sorry

end correct_option_is_A_l545_545721


namespace min_weighings_for_16_l545_545731

noncomputable def min_weighings (n : ℕ) : ℕ :=
match n with
| 16 := 3
| _  := sorry

theorem min_weighings_for_16 :
  min_weighings 16 = 3 := by
  sorry

end min_weighings_for_16_l545_545731


namespace complement_intersection_l545_545012

open Set

variable {R : Type} [LinearOrderedField R]

def P : Set R := {x | x^2 - 2*x ≥ 0}
def Q : Set R := {x | 1 < x ∧ x ≤ 3}

theorem complement_intersection : (compl P ∩ Q) = {x : R | 1 < x ∧ x < 2} := by
  sorry

end complement_intersection_l545_545012


namespace sum_of_faces_is_54_l545_545625

-- Conditions
def consecutive_even_numbers (n : ℕ) : Prop :=
  n % 2 = 0 ∧ ∃ k, n = 4 + 2 * k

def sum_of_opposite_faces_is_equal (faces : list ℕ) : Prop :=
  faces.length = 6 ∧
  ∃ (p1 p2 p3 p4 p5 p6 : ℕ), 
    (faces = [p1, p2, p3, p4, p5, p6] ∧
     consecutive_even_numbers p1 ∧ consecutive_even_numbers p2 ∧ consecutive_even_numbers p3 ∧ 
     consecutive_even_numbers p4 ∧ consecutive_even_numbers p5 ∧ consecutive_even_numbers p6 ∧
     p1 + p6 = p2 + p5 ∧ p1 + p6 = p3 + p4 ∧ p1 = 4)

-- Proof Problem
theorem sum_of_faces_is_54 (faces : list ℕ) (h1 : sum_of_opposite_faces_is_equal faces) : 
  faces.sum = 54 :=
by sorry

end sum_of_faces_is_54_l545_545625


namespace violet_balloons_remaining_l545_545979

def initial_count : ℕ := 7
def lost_count : ℕ := 3

theorem violet_balloons_remaining : initial_count - lost_count = 4 :=
by sorry

end violet_balloons_remaining_l545_545979


namespace c_profit_share_l545_545272

noncomputable theory

def investment_C (inv_total : ℝ) (a_b : ℝ) (b_c : ℝ) : ℝ :=
  (inv_total - a_b - b_c) / 3

def profit_share (total_profit : ℝ) (ratio : list ℝ) (index : ℕ) : ℝ :=
  (total_profit * ratio.getD index 0) / (ratio.foldr (+) 0)

theorem c_profit_share :
  let inv_total := 120000
      total_profit := 50000
      ratio := [4, 3, 2]
      A_B_diff := 14000
      B_C_diff := 8000
      C_investment := investment_C inv_total A_B_diff B_C_diff
      C_share := profit_share total_profit ratio 2 in
  C_share = 11111.11 :=
by
  sorry

end c_profit_share_l545_545272


namespace right_triangle_area_l545_545695

/-- Given a right triangle with one leg of length 30 inches and a hypotenuse of 34 inches,
    the area of the triangle is 240 square inches. -/
theorem right_triangle_area (a b c : ℝ) (h1 : a = 30) (h2 : c = 34) (h3 : a^2 + b^2 = c^2) :
  (1/2) * a * b = 240 :=
by
  rw [h1, h2] at h3
  have hb : b = 16 := by
    rw [←h3]
    norm_num
  rw [h1, hb]
  norm_num
  sorry

end right_triangle_area_l545_545695


namespace sum_of_reciprocals_B_l545_545988

-- Defining the set B
def B : Set ℕ := {n | ∃ (a b c d : ℕ), n = 2^a * 3^b * 5^c * 7^d}

-- Statement: Given the above definition, we need to prove the sum of the reciprocals is 35/8,
-- and hence, m+n = 43
theorem sum_of_reciprocals_B {m n : ℕ} (h_coprime : Nat.coprime m n) (h_sum_reciprocal : ∑' (n : ℕ) in B, (1 / n) = 35/8) : m + n = 43 :=
by sorry

end sum_of_reciprocals_B_l545_545988


namespace longer_diagonal_eq_l545_545519

variable (a b : ℝ)
variable (h_cd : CD = a) (h_bc : BC = b) (h_diag : AC = a) (h_ad : AD = 2 * b)

theorem longer_diagonal_eq (CD BC AC AD BD : ℝ) (h_cd : CD = a)
  (h_bc : BC = b) (h_diag : AC = CD) (h_ad : AD = 2 * b) :
  BD = Real.sqrt (a^2 + 3 * b^2) :=
sorry

end longer_diagonal_eq_l545_545519


namespace highest_power_of_seven_in_factorial_l545_545707

theorem highest_power_of_seven_in_factorial (n : ℕ) (h : n = 1000) :
  ∑ i in finset.range (n + 1), n / 7^i = 164 :=
by
  sorry

end highest_power_of_seven_in_factorial_l545_545707


namespace comb_n_plus_1_2_l545_545671

theorem comb_n_plus_1_2 (n : ℕ) (h : 0 < n) : 
  (n + 1).choose 2 = (n + 1) * n / 2 :=
by sorry

end comb_n_plus_1_2_l545_545671


namespace sum_f_eq_24136_l545_545342

def f (x : ℕ) : ℕ :=
  let g := x*x - 2017*x + 8052
  g + |g|

theorem sum_f_eq_24136 : 
  (∑ k in finset.range 2013, f (k + 1)) = 24136 :=
sorry

end sum_f_eq_24136_l545_545342


namespace location_of_z_in_fourth_quadrant_l545_545887

noncomputable def z : ℂ := 5 * complex.I / (2 * complex.I - 1)

theorem location_of_z_in_fourth_quadrant : (z.re > 0) ∧ (z.im < 0) :=
  sorry

end location_of_z_in_fourth_quadrant_l545_545887


namespace question_l545_545858

noncomputable def a_n (n : ℕ) : ℝ :=
  if n = 1 then 4
  else if n = 2 then 5
  else if n = 3 then 6
  else if n > 3 then 4 + (n - 1)

noncomputable def S : ℕ → ℝ
| 0       := 0
| (n + 1) := (n + 1) * (a_n (n + 1)) / 2

theorem question (n : ℕ) :
  (∀ a₁ a₂ a₃ a₄ : ℝ, a₁ = 4 → a₁ + a₂ + a₃ + a₄ = 18) →
  n = 3 ∨ n = 5 ↔ (S 5) / (S n) ∈ ℤ :=
by sorry

end question_l545_545858


namespace initial_puppies_count_l545_545760

-- Define the initial conditions
def initial_birds : Nat := 12
def initial_cats : Nat := 5
def initial_spiders : Nat := 15
def initial_total_animals : Nat := 25
def half_birds_sold : Nat := initial_birds / 2
def puppies_adopted : Nat := 3
def spiders_lost : Nat := 7

-- Define the remaining animals
def remaining_birds : Nat := initial_birds - half_birds_sold
def remaining_cats : Nat := initial_cats
def remaining_spiders : Nat := initial_spiders - spiders_lost

-- Define the total number of remaining animals excluding puppies
def remaining_non_puppy_animals : Nat := remaining_birds + remaining_cats + remaining_spiders

-- Define the remaining puppies
def remaining_puppies : Nat := initial_total_animals - remaining_non_puppy_animals
def initial_puppies : Nat := remaining_puppies + puppies_adopted

-- State the theorem
theorem initial_puppies_count :
  ∀ puppies : Nat, initial_puppies = 9 :=
by
  sorry

end initial_puppies_count_l545_545760


namespace ellipse_equation_existence_of_lambda_l545_545869

theorem ellipse_equation (a b : ℝ) (h1 : a > b) (h2 : 0 < b) (h3 : b^2 = 3/4 * a^2) (h4 : (1:ℝ)^2/a^2 + (3/2:ℝ)^2/b^2 = 1) :
  (a = 2) ∧ (b = sqrt 3) ∧ (∀ x y : ℝ, x^2/4 + y^2/3 = 1) := by
  sorry

theorem existence_of_lambda (a b : ℝ) (h1 : a > b) (h2 : 0 < b) (h3 : b^2 = 3/4 * a^2) (h4 : ∀ P Q : ℝ × ℝ, P ≠ (-2, 0) → P ≠ (2, 0) → Q ≠ (-2, 0) → Q ≠ (2, 0) → (P.1^2/4 + P.2^2/3 = 1) → (Q.1^2/4 + Q.2^2/3 = 1) → ∀ m n : ℝ, (1/2, 0) ∈ PQ → k_A1P = λ * k_A2Q) :
  ∃ λ : ℝ, λ = 3/5 := by
  sorry

end ellipse_equation_existence_of_lambda_l545_545869


namespace measure_of_C_and_max_perimeter_l545_545467

noncomputable def triangle_C_and_perimeter (a b c A B C : ℝ) (hABC : (2 * a + b) * Real.sin A + (2 * b + a) * Real.sin B = 2 * c * Real.sin C) (hc : c = Real.sqrt 3) : Prop :=
  (C = 2 * Real.pi / 3) ∧ (2 * Real.sin A + 2 * Real.sin B + c ≤ 2 + Real.sqrt 3)

-- Now the Lean theorem statement
theorem measure_of_C_and_max_perimeter (a b c A B C : ℝ) (hABC : (2 * a + b) * Real.sin A + (2 * b + a) * Real.sin B = 2 * c * Real.sin C) (hc : c = Real.sqrt 3) :
  triangle_C_and_perimeter a b c A B C hABC hc :=
by 
  sorry

end measure_of_C_and_max_perimeter_l545_545467


namespace lukas_games_played_l545_545582

-- Define the given conditions
def average_points_per_game : ℕ := 12
def total_points_scored : ℕ := 60

-- Define Lukas' number of games
def number_of_games (total_points : ℕ) (average_points : ℕ) : ℕ :=
  total_points / average_points

-- Theorem and statement to prove
theorem lukas_games_played :
  number_of_games total_points_scored average_points_per_game = 5 :=
by
  sorry

end lukas_games_played_l545_545582


namespace area_of_triangle_ABC_l545_545548

-- Define the problem conditions
variables {A B C D E : Type} [triangle ABC] [AB BC : AB = BC] [altitude BD : BD is altitude of ABC]
variable (BE : real) (h_BE : BE = 14)
variables {γ θ φ : real} 
  (h_tan_geom : (tan γ, tan θ, tan φ).isGeom)
  (h_cot_arith : (cot γ, cot θ, cot φ).isArith)

-- The proof statement that the area of triangle ABC is 49/3
theorem area_of_triangle_ABC : ∃ (AB BC BD : ℝ), is_triangle ABC ∧ 
  AB = BC ∧
  BD.is_altitude ABC ∧ 
  BE = 14 ∧ 
  (is_geom (tan γ, tan θ, tan φ)) ∧ 
  (is_arith (cot γ, cot θ, cot φ)) → 
  triangle_area ABC = 49 / 3 := 
  by sorry

end area_of_triangle_ABC_l545_545548


namespace min_sum_a1_a2_l545_545651

-- Define the condition predicate for the sequence
def satisfies_seq (a : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, n ≥ 1 → a (n + 2) = (a n + 2009) / (1 + a (n + 1))

-- State the main problem as a theorem in Lean 4
theorem min_sum_a1_a2 (a : ℕ → ℕ) (h_seq : satisfies_seq a) (h_pos : ∀ n, a n > 0) :
  a 1 * a 2 = 2009 → a 1 + a 2 = 90 :=
sorry

end min_sum_a1_a2_l545_545651


namespace volume_parallelepiped_l545_545442

variables (a b : ℝ^3)
variables (ha : ∥a∥ = 1) (hb : ∥b∥ = 1)
variables (angle_ab : real.angle a b = π / 4)

theorem volume_parallelepiped : 
  abs (a • ((a + b × a) × b)) = 1 / 2 :=
sorry

end volume_parallelepiped_l545_545442


namespace Alex_failing_implies_not_all_hw_on_time_l545_545046

-- Definitions based on the conditions provided
variable (Alex_submits_all_hw_on_time : Prop)
variable (Alex_passes_course : Prop)

-- Given condition: Submitting all homework assignments implies passing the course
axiom Mrs_Thompson_statement : Alex_submits_all_hw_on_time → Alex_passes_course

-- The problem: Prove that if Alex failed the course, then he did not submit all homework assignments on time
theorem Alex_failing_implies_not_all_hw_on_time (h : ¬Alex_passes_course) : ¬Alex_submits_all_hw_on_time :=
  by
  sorry

end Alex_failing_implies_not_all_hw_on_time_l545_545046


namespace abs_inequality_solution_l545_545150

theorem abs_inequality_solution (x : ℝ) (h : |x - 4| ≤ 6) : -2 ≤ x ∧ x ≤ 10 := 
sorry

end abs_inequality_solution_l545_545150


namespace proof_sum_q_p_x_l545_545014

def p (x : ℝ) : ℝ := |x| - 3
def q (x : ℝ) : ℝ := -|x|

-- define the list of x values
def x_values : List ℝ := [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]

-- define q_p_x to apply q to p of each x
def q_p_x : List ℝ := x_values.map (λ x => q (p x))

-- define the sum of q(p(x)) for given x values
def sum_q_p_x : ℝ := q_p_x.sum

theorem proof_sum_q_p_x : sum_q_p_x = -15 := by
  -- steps of solution
  sorry

end proof_sum_q_p_x_l545_545014


namespace cos_2θ_is_neg_half_lambda_range_l545_545355

-- Conditions for part 1
variable (θ : ℝ)
variable (z1 : ℂ := 2 * Complex.sin θ - Complex.I * Real.sqrt 3)
variable (z2 : ℂ := 1 + 2 * Complex.cos θ * Complex.I)

-- Conditions for part 2
variable (a : ℂ := Complex.mk (2 * Real.sin θ) (- Real.sqrt 3))
variable (b : ℂ := Complex.mk 1 (2 * Real.cos θ))
variable (λ : ℝ)

axiom θ_range : π / 3 ≤ θ ∧ θ ≤ π / 2

-- Part 1: Prove cos 2θ = -1/2
theorem cos_2θ_is_neg_half : z1 * z2 = (z1 * z2).re → Real.cos (2 * θ) = -1 / 2 :=
by
  assume h : z1 * z2 = (z1 * z2).re
  sorry

-- Part 2: Prove range of λ
theorem lambda_range : (a * lambda - b) * (a - lambda * b) = 0 → 
  λ ∈ Set.Iic (2 - Real.sqrt 3) ∪ Set.Ici (2 + Real.sqrt 3) :=
by
  assume h : (a * lambda - b) * (a - lambda * b) = 0
  sorry

end cos_2θ_is_neg_half_lambda_range_l545_545355


namespace ratio_of_boxes_loaded_l545_545788

variable (D N B : ℕ) 

-- Definitions as conditions
def night_crew_workers (D : ℕ) : ℕ := (4 * D) / 9
def day_crew_boxes (B : ℕ) : ℕ := (3 * B) / 4
def night_crew_boxes (B : ℕ) : ℕ := B / 4

theorem ratio_of_boxes_loaded :
  ∀ {D B : ℕ}, 
    night_crew_workers D ≠ 0 → 
    D ≠ 0 → 
    B ≠ 0 → 
    ((night_crew_boxes B) / (night_crew_workers D)) / ((day_crew_boxes B) / D) = 3 / 4 :=
by
  -- Proof
  sorry

end ratio_of_boxes_loaded_l545_545788


namespace sandwich_price_l545_545716

-- Definitions based on conditions
def price_of_soda : ℝ := 0.87
def total_cost : ℝ := 6.46
def num_soda : ℝ := 4
def num_sandwich : ℝ := 2

-- The key equation based on conditions
def total_cost_equation (S : ℝ) : Prop := 
  num_sandwich * S + num_soda * price_of_soda = total_cost

theorem sandwich_price :
  ∃ S : ℝ, total_cost_equation S ∧ S = 1.49 :=
by
  sorry

end sandwich_price_l545_545716


namespace hexagonal_pyramid_circumscribed_sphere_volume_l545_545067

structure RegularHexagonPyramid :=
(base_side : ℝ)
(height : ℝ)

def circumscribedSphereVolume (P : RegularHexagonPyramid) : ℝ :=
  let r := (P.height ^ 2 + (P.base_side * sqrt 3) ^ 2) / 2
  (4 / 3) * π * r ^ 3

theorem hexagonal_pyramid_circumscribed_sphere_volume :
  ∀ (P : RegularHexagonPyramid), P.base_side = sqrt 2 ∧ P.height = 2 →
  circumscribedSphereVolume P = 4 * sqrt 3 * π :=
by
  intros P h
  sorry

end hexagonal_pyramid_circumscribed_sphere_volume_l545_545067


namespace math_problem_l545_545447

theorem math_problem
  (p q r : ℕ) (hp : 0 < p) (hq : 0 < q) (hr : 0 < r)
  (h1 : p * q + r = 47)
  (h2 : q * r + p = 47)
  (h3 : r * p + q = 47) :
  p + q + r = 48 :=
sorry

end math_problem_l545_545447


namespace no_negative_roots_of_polynomial_l545_545921

def polynomial (x : ℝ) := x^4 - 5 * x^3 - 4 * x^2 - 7 * x + 4

theorem no_negative_roots_of_polynomial :
  ¬ ∃ (x : ℝ), x < 0 ∧ polynomial x = 0 :=
by
  sorry

end no_negative_roots_of_polynomial_l545_545921


namespace arrangement_not_possible_l545_545292

theorem arrangement_not_possible : ¬∃ (a : ℕ → ℕ) (b : ℕ → ℕ),
  (∀ k : ℕ, 1 ≤ k ∧ k ≤ 1986 → b k - a k = k + 1) ∧ 
  (∀ k : ℕ, 1 ≤ k ∧ k ≤ 1986 → 1 ≤ a k ∧ a k < b k ∧ b k ≤ 3972) ∧
  (∀ i : ℕ, 1 ≤ i ∧ i ≤ 3972 → ∃ k : ℕ, 1 ≤ k ∧ k ≤ 1986 ∧ (i = a k ∨ i = b k)) :=
begin
  sorry
end

end arrangement_not_possible_l545_545292


namespace probability_of_answering_phone_in_4_rings_l545_545754

/-- A proof statement that asserts the probability of answering the phone within the first four rings is equal to 9/10. -/
theorem probability_of_answering_phone_in_4_rings :
  (1/10) + (3/10) + (2/5) + (1/10) = 9/10 :=
by
  sorry

end probability_of_answering_phone_in_4_rings_l545_545754


namespace GODOT_value_l545_545997

theorem GODOT_value (G O D I T : ℕ) (h1 : G ≠ 0) (h2 : D ≠ 0) 
  (eq1 : 1000 * G + 100 * O + 10 * G + O + 1000 * D + 100 * I + 10 * D + I = 10000 * G + 1000 * O + 100 * D + 10 * O + T) : 
  10000 * G + 1000 * O + 100 * D + 10 * O + T = 10908 :=
by {
  sorry
}

end GODOT_value_l545_545997


namespace partition_teams_in_tournament_l545_545607

theorem partition_teams_in_tournament :
  ∃ (A B : finset ℕ), A.card + B.card = 14 ∧ (∀ a ∈ A, ∀ b ∈ B, won a b) := sorry

end partition_teams_in_tournament_l545_545607


namespace mary_spends_5_l545_545583

theorem mary_spends_5 (marco_has mary_has : ℕ) (marco_halves mary_spends : ℕ → ℕ) (mary_more marco_less : Prop) (mary_spends_amount : ℕ → Prop) :
  marco_has = 24 →
  mary_has = 15 →
  marco_halves marco_has = marco_has / 2 →
  marco_halves marco_has + mary_has > marco_has - marco_halves marco_has →
  marco_halves marco_has + mary_has - mary_spends mario_less = 10 →
  mary_spends_amount mary_spends =
  5 := by sorry

end mary_spends_5_l545_545583


namespace sum_of_valid_n_l545_545561

theorem sum_of_valid_n : 
  let valid_n := {n : ℕ | n < 2023 ∧ ∃ p : ℤ, 2 * n^2 + 3 * n = p^2 } 
  ∑ n in valid_n, n = 444 :=
sorry

end sum_of_valid_n_l545_545561


namespace alan_tickets_l545_545188

variables (A M : ℕ)

def condition1 := A + M = 150
def condition2 := M = 5 * A - 6

theorem alan_tickets : A = 26 :=
by
  have h1 : condition1 A M := sorry
  have h2 : condition2 A M := sorry
  sorry

end alan_tickets_l545_545188


namespace ryegrass_percentage_l545_545261

variable (R : ℝ) -- Percentage of ryegrass in seed mixture X

variable (W : ℝ) -- Weight of the final mixture
variable (X_cont_percent : ℝ) := 0.13333333333333332 -- Percent X in the final mixture
variable (Y_cont_percent : ℝ) := 1 - X_cont_percent -- Percent Y in the final mixture 

variable (ryegrass_Y : ℝ) := 0.25 -- Percentage of ryegrass in seed mixture Y
variable (total_ryegrass : ℝ) := 0.27 -- Percentage of ryegrass in the final mixture

theorem ryegrass_percentage :
  (R * X_cont_percent + ryegrass_Y * Y_cont_percent) = total_ryegrass → R = 0.4 :=
by
  sorry

end ryegrass_percentage_l545_545261


namespace sequence_count_21_l545_545026

noncomputable def f : ℕ → ℕ
| 3 := 1
| 4 := 1
| 5 := 2
| 6 := 3
| 7 := 4
| n := f (n - 4) + 2 * f (n - 5) + f (n - 6)

theorem sequence_count_21 : f 21 = ?
sorry

end sequence_count_21_l545_545026


namespace horner_value_v2_l545_545216

theorem horner_value_v2 (x : ℤ) (f : ℤ → ℤ) (h_f : f = λ x, x^5 + 4 * x^4 + x^2 + 20 * x + 16) (hx : x = -2) : 
  let v2 : ℤ := 
    let a := x + 4 in
    let b := a * x in
    let c := b + 0 in
    let d := c * x in
    let e := d + 1 in
    let f := e * x in
    let g := f + 20 in
    let v2 := g * x + 16 in
    v2 = -4
:= sorry

end horner_value_v2_l545_545216


namespace probability_ball_less_than_three_l545_545966

theorem probability_ball_less_than_three :
  let balls := [1, 2, 3, 4, 5] in
  let favorable_balls := [1, 2] in
  (favorable_balls.length : ℚ) / (balls.length : ℚ) = 2 / 5 :=
by
  sorry

end probability_ball_less_than_three_l545_545966


namespace add_fractions_11_12_7_15_l545_545286

/-- A theorem stating that the sum of 11/12 and 7/15 is 83/60. -/
theorem add_fractions_11_12_7_15 : (11 / 12) + (7 / 15) = (83 / 60) := 
by
  sorry

end add_fractions_11_12_7_15_l545_545286


namespace yellow_area_difference_l545_545976

-- Define the given constants
def A : ℕ := 20
def B : ℕ := 30
def a : ℕ := 4
def b : ℕ := 7

-- Define the areas of the large and small rectangles
def area_large : ℕ := A * B
def area_small : ℕ := a * b
def area_yellow : ℕ := area_large - area_small

-- Statement for the problem
theorem yellow_area_difference (A B a b : ℕ) (h₁ : A = 20) (h₂ : B = 30) (h₃ : a = 4) (h₄ : b = 7) :
  area_yellow = 572 :=
by
  rw [h₁, h₂, h₃, h₄]
  exact rfl

end yellow_area_difference_l545_545976


namespace revenue_decrease_1_percent_l545_545174

variable (T C : ℝ)  -- Assumption: T and C are real numbers representing the original tax and consumption

noncomputable def original_revenue : ℝ := T * C
noncomputable def new_tax_rate : ℝ := T * 0.90
noncomputable def new_consumption : ℝ := C * 1.10
noncomputable def new_revenue : ℝ := new_tax_rate T * new_consumption C

theorem revenue_decrease_1_percent :
  new_revenue T C = 0.99 * original_revenue T C := by
  sorry

end revenue_decrease_1_percent_l545_545174


namespace tangent_lines_count_l545_545599

-- Define the circles with their radii
def CircleA_radius := 3
def CircleB_radius := 5
def centers_distance := 7

-- Statement to prove
theorem tangent_lines_count : 
  ∃ (num_lines : ℕ), num_lines = 4 ∧ 
    (distance_between_centers : ℝ) (radius_A radius_B : ℝ), 
      radius_A = CircleA_radius ∧ 
      radius_B = CircleB_radius ∧ 
      centers_distance = 7 → 
      num_lines := 4 :=
by
  sorry

end tangent_lines_count_l545_545599


namespace polynomial_factorization_l545_545137

-- Definitions used in the conditions
def given_polynomial (a b c : ℝ) : ℝ :=
  a^3 * (b^2 - c^2) + b^3 * (c^2 - a^2) + c^3 * (a^2 - b^2)

def p (a b c : ℝ) : ℝ := -(a * b + a * c + b * c)

-- The Lean 4 statement to be proved
theorem polynomial_factorization (a b c : ℝ) :
  given_polynomial a b c = (a - b) * (b - c) * (c - a) * p a b c :=
by
  sorry

end polynomial_factorization_l545_545137


namespace probability_product_positive_of_independent_selection_l545_545206

theorem probability_product_positive_of_independent_selection :
  let I := set.Icc (-30 : ℝ) (15 : ℝ)
  let P := (λ (x y : ℝ), x ∈ I ∧ y ∈ I ∧ x * y > 0)
  (Prob { x : ℝ × ℝ | P x.1 x.2 } :
    ProbabilitySpace (I × I)) = 5 / 9 :=
by
  sorry

end probability_product_positive_of_independent_selection_l545_545206


namespace find_m_l545_545047

open_locale big_operators

-- Definitions
variables {a : ℕ → ℝ}
variable {m : ℕ}

-- Conditions
def geometric_condition (m : ℕ) : Prop :=
  a m * a (m + 2) = 2 * a (m + 1)

def product_condition (a : ℕ → ℝ) (m : ℕ) : Prop :=
  ∏ i in finset.range (2 * m + 1), a i = 128

-- Theorem to prove
theorem find_m (a : ℕ → ℝ) (m : ℕ) 
  (h1 : ∀ n ∈ finset.range (2 * m + 1), 0 < a n)
  (h2 : geometric_condition m)
  (h3 : product_condition a m) :
  m = 3 :=
sorry

end find_m_l545_545047


namespace number_not_divisible_by_4_or_6_l545_545273

theorem number_not_divisible_by_4_or_6 : 
  let count := (1000 - (Nat.floor (1000 / 4) + Nat.floor (1000 / 6) - Nat.floor (1000 / 12))) in
  667 = count :=
by
  sorry

end number_not_divisible_by_4_or_6_l545_545273


namespace solution_exists_l545_545387

-- Given conditions
def quadratic_inequality_solution_set (a b : ℝ) : Prop :=
  {x : ℝ | (1 < x ∧ x < 3)} = {x | x^2 - a * x - b < 0}

def solve_inequality (a b : ℝ) : set ℝ :=
  {x : ℝ | (2 * x + a) / (x + b) > 1}

-- Proof goal
theorem solution_exists (a b : ℝ) :
  quadratic_inequality_solution_set a b →
  solve_inequality 4 (-3) = {x | x > 3 ∨ x < -7} :=
sorry

end solution_exists_l545_545387


namespace rhombus_diagonal_AC_eq_5_l545_545066

theorem rhombus_diagonal_AC_eq_5 (A B C D : Type) [plane_geometry A B C D]
    (rhombus_ABCD : is_rhombus A B C D) (AB_eq_5 : distance A B = 5)
    (angle_BCD_eq_120 : angle B C D = 120) : distance A C = 5 :=
sorry

end rhombus_diagonal_AC_eq_5_l545_545066


namespace ellipse_equation_with_foci_l545_545056

theorem ellipse_equation_with_foci (M N P : ℝ × ℝ)
  (area_triangle : Real) (tan_M tan_N : ℝ)
  (h₁ : area_triangle = 1)
  (h₂ : tan_M = 1 / 2)
  (h₃ : tan_N = -2) :
  ∃ (a b : ℝ), (4 * x^2) / (15 : ℝ) + y^2 / (3 : ℝ) = 1 :=
by
  -- Definitions to meet given conditions would be here
  sorry

end ellipse_equation_with_foci_l545_545056


namespace count_four_digit_numbers_divisible_by_5_ending_in_45_l545_545425

theorem count_four_digit_numbers_divisible_by_5_ending_in_45 : 
  ∃ n : ℕ, (∀ x : ℕ, x ≥ 1000 ∧ x < 10000 ∧ x % 100 = 45 → (x % 5 = 0 ∧ x % 100 = 45)) 
  ∧ (n = 90) :=
by
  sorry

end count_four_digit_numbers_divisible_by_5_ending_in_45_l545_545425


namespace quadratic_roots_and_signs_l545_545125

theorem quadratic_roots_and_signs :
  (∃ x1 x2 : ℝ, (x1^2 - 13*x1 + 40 = 0) ∧ (x2^2 - 13*x2 + 40 = 0) ∧ x1 = 5 ∧ x2 = 8 ∧ 0 < x1 ∧ 0 < x2) :=
by
  sorry

end quadratic_roots_and_signs_l545_545125


namespace find_y_value_l545_545619

theorem find_y_value (t : ℝ) (x : ℝ) (y : ℝ)
  (h1 : x = 3 - 2 * t)
  (h2 : y = 3 * t + 6)
  (h3 : x = -6)
  : y = 19.5 :=
by {
  sorry
}

end find_y_value_l545_545619


namespace red_fraction_after_doubling_l545_545472

theorem red_fraction_after_doubling (x : ℕ) (h : (3/5 : ℚ) * x = (3/5 : ℚ) * x) : 
  let blue_marbles := (3/5 : ℚ) * x
      red_marbles := x - blue_marbles
      new_red_marbles := 2 * red_marbles 
      total_marbles := blue_marbles + new_red_marbles 
  in new_red_marbles / total_marbles = (4/7 : ℚ) :=
by 
  sorry

end red_fraction_after_doubling_l545_545472


namespace exercise_l545_545605

theorem exercise (n : ℕ) : 900 ∣ (6 ^ (2 * (n + 1)) - 2 ^ (n + 3) * 3 ^ (n + 2) + 36) :=
sorry

end exercise_l545_545605


namespace area_of_right_triangle_l545_545704

-- Given definitions
def leg_a : ℝ := 30
def hypotenuse_c : ℝ := 34

-- The theorem statement
theorem area_of_right_triangle : 
  ∀ (b : ℝ), b = real.sqrt (hypotenuse_c^2 - leg_a^2) → 
  let area := 1 / 2 * leg_a * b in
  area = 240 := 
by
  intro b
  intro h
  let area := 1 / 2 * leg_a * b
  sorry

end area_of_right_triangle_l545_545704


namespace count_valid_numbers_l545_545392

-- Define what it means to be a four-digit number that ends in 45
def is_valid_number (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999 ∧ n % 100 = 45 ∧ n % 5 = 0

-- Define the set of valid two-digit prefixes
def valid_prefixes : set ℕ := {ab | 10 ≤ ab ∧ ab ≤ 99}

-- Define the set of four-digit numbers that end in 45 and are divisible by 5
def valid_numbers : set ℕ := {n | ∃ ab : ℕ, ab ∈ valid_prefixes ∧ n = ab * 100 + 45}

-- State the theorem
theorem count_valid_numbers : (finset.card (finset.filter is_valid_number (finset.range 10000)) = 90) :=
sorry

end count_valid_numbers_l545_545392


namespace interval_of_increase_l545_545633

noncomputable def f (x : ℝ) := 2 * x^3 + 3 * x^2 - 12 * x + 1

theorem interval_of_increase :
  { x : ℝ | ∃ (a b : ℝ), a < b ∧ (∀ x ∈ Ioo a b, 0 < deriv f x) } =  (Ioo (-∞) (-2) ∪ Ioo 1 (∞)) :=
sorry

end interval_of_increase_l545_545633


namespace positional_relationship_l545_545344

-- Defining the conditions
variables (m n : Type) (α : Type)
variable [plane α] -- α is a plane
-- A function indicating that m is a line and is parallel to plane α
def line_parallel_to_plane (m : Type) (α : Type) [is_line m] [plane α] : Prop := sorry
-- A function indicating that n is a line contained in plane α
def line_in_plane (n : Type) (α : Type) [is_line n] [plane α] : Prop := sorry
-- The final relationship we want to prove
theorem positional_relationship
  (m n : Type) (α : Type) [is_line m] [is_line n] [plane α] :
  line_parallel_to_plane m α → line_in_plane n α → (parallel_lines m n ∨ different_planes m n) :=
by
  sorry

end positional_relationship_l545_545344


namespace correct_answer_l545_545228

variables (A B : polynomial ℝ) (a : ℝ)

theorem correct_answer (hB : B = 3 * a^2 - 5 * a - 7) (hMistake : A - 2 * B = -2 * a^2 + 3 * a + 6) :
  A + 2 * B = 10 * a^2 - 17 * a - 22 :=
by
  sorry

end correct_answer_l545_545228


namespace hexagon_area_l545_545576

theorem hexagon_area (A B C D E F : ℝ × ℝ) (b : ℝ)
  (hA : A = (0, 0)) (hB : B = (b, 4)) 
  (hAB : AB.dist = CD.dist)
  (hBC : BC.dist = DE.dist)
  (hCD : CD.dist = EF.dist)
  (hDE : DE.dist = FA.dist)
  (hEF : EF.dist = AB.dist)
  (hFAB : angle FAB = 120)
  (h1 : (fst A) = (fst F))
  (h2 : (snd F) = 8)
  (h3 : ∃ ys : list ℝ, ys = [0, 4, 8, 12, 16, 20] ∧ (∀ y, (y ∈ ys) → y ∈ {snd A, snd B, snd C, snd D, snd E, snd F})) :
  let m := 192,
      n := 3 in 
  m + n = 195 :=
sorry

end hexagon_area_l545_545576


namespace concyclicity_l545_545998

variables {A B C X P Q R U V W : Type*}

-- Defining the conditions
-- X is a point inside triangle ABC
def point_inside_triangle (A B C X : Type*) : Prop := sorry -- Details of the definition

-- Lines AX, BX, and CX intersect the circumcircle of triangle ABC again at points P, Q, and R, respectively
def lines_intersect_circumcircle (A B C X P Q R : Type*) : Prop := sorry -- Details of the definition

-- Point U lies on the segment XP
def point_on_segment (X P U : Type*) : Prop := sorry -- Details of the definition

-- Parallels to AB and AC passing through U intersect XQ and XR at V and W, respectively
def parallels_pass_through_point (A B C X U V W : Type*) : Prop := sorry -- Details of the definition

-- Concisely restating the conditions
axiom h1 : point_inside_triangle A B C X
axiom h2 : lines_intersect_circumcircle A B C X P Q R
axiom h3 : point_on_segment X P U
axiom h4 : parallels_pass_through_point A B C X U V W

-- Proving the final statement
theorem concyclicity : ∀ {A B C X P Q R U V W : Type*},
  point_inside_triangle A B C X →
  lines_intersect_circumcircle A B C X P Q R →
  point_on_segment X P U →
  parallels_pass_through_point A B C X U V W →
  (concyclic R W V Q) :=
by
  intros _ _ _ _ _ _ _ _ _ _ _ h1 h2 h3 h4
  sorry -- Proof omitted

end concyclicity_l545_545998


namespace not_out_performed_players_l545_545654

theorem not_out_performed_players (n : ℕ) (h : n ≥ 3) :
  (∃ (A B : ℕ) (C : ℕ), A ≠ B ∧ A ≠ C ∧ B ≠ C ∧ ∃ matches : set (ℕ × ℕ),
      (∀ x y, x ≠ y → (x = A → y = C ∨ x = C → y ≠ B)) ∧ (∃ players : fin n, true)) ↔ n ≥ 5 :=
begin
  sorry
end

end not_out_performed_players_l545_545654


namespace n_squared_plus_inverse_squared_plus_four_eq_102_l545_545033

theorem n_squared_plus_inverse_squared_plus_four_eq_102 (n : ℝ) (h : n + 1 / n = 10) :
    n^2 + 1 / n^2 + 4 = 102 :=
by sorry

end n_squared_plus_inverse_squared_plus_four_eq_102_l545_545033


namespace total_amount_paid_l545_545052

def original_price_per_card : Int := 12
def discount_per_card : Int := 2
def number_of_cards : Int := 10

theorem total_amount_paid :
  original_price_per_card - discount_per_card * number_of_cards = 100 :=
by
  sorry

end total_amount_paid_l545_545052


namespace value_of_fraction_pow_l545_545382

theorem value_of_fraction_pow (a b : ℤ) 
  (h1 : ∀ x, (x^2 + (a + 1)*x + a*b) ≤ 0 ↔ -1 ≤ x ∧ x ≤ 4) : 
  ((1 / 2 : ℚ) ^ (a + 2*b) = 4) :=
sorry

end value_of_fraction_pow_l545_545382


namespace solve_lambda_l545_545390

open Classical

variable {R : Type} [LinearOrderedField R]
variable {V : Type} [AddCommGroup V] [Module R V]
variable {a b : V} (λ : R)

-- Define the condition that vectors a and b are not parallel
def not_parallel (a b : V) : Prop :=
  ∀ (k : R), a ≠ k • b

-- Define the condition of two vectors being parallel
def parallel (v1 v2 : V) : Prop :=
  ∃ (k : R), v1 = k • v2

-- Given statements
theorem solve_lambda (h1 : not_parallel a b) (h2 : parallel (λ • a + b) (a + 2 • b)) : λ = 1/2 :=
by
  sorry

end solve_lambda_l545_545390


namespace area_of_QRUT_l545_545547

-- Define the given problem conditions
constant P Q R S T U : Type
constant PQT RSU : Type → Prop
constant is_rectangle : P → Q → R → S → Prop
constant side_length : P → Q → ℝ
constant equilateral_triangle : T → Type → Prop

-- Given conditions
axiom PQRS_is_rectangle : is_rectangle P Q R S
axiom PQ_length : side_length P Q = 2
axiom QR_length : side_length Q R = 4
axiom T_in_PQT : equilateral_triangle T (PQT)
axiom U_in_RSU : equilateral_triangle U (RSU)

-- Target proof
theorem area_of_QRUT : is_rectangle P Q R S → 
                      side_length P Q = 2 → 
                      side_length Q R = 4 → 
                      equilateral_triangle T (PQT) → 
                      equilateral_triangle U (RSU) → 
                      ∃ area : ℝ, area = 4 - sqrt 3 :=
by
  intros
  sorry -- The proof is not necessary as per the instructions.

end area_of_QRUT_l545_545547


namespace curve_cartesian_equation_line_cartesian_equation_min_FA_FB_l545_545068

theorem curve_cartesian_equation (ρ θ : ℝ) :
  (ρ * (1 + cos(θ)^2) = 8 * sin(θ)) → ∃ x y : ℝ, (x = ρ * cos(θ)) ∧ (y = ρ * sin(θ)) ∧ (x^2 = 4 * y) :=
by
  intros h
  use ρ * cos θ, ρ * sin θ
  split
  { refl }
  split
  { refl }
  sorry

theorem line_cartesian_equation_min_FA_FB (α t : ℝ) :
  (∀ α : ℝ, ∃ t : ℝ, (∀ x y : ℝ, (x = t * cos α) ∧ (y = 1 + t * sin α)) →
  let x := t * cos α in
  let y := 1 + t * sin α in
  (x^2 = 4 * y) → (∃ α : ℝ, |t * cos α * 4 * t * sin α| = 4 → y = 1)) :=
by
  intros h
  sorry

end curve_cartesian_equation_line_cartesian_equation_min_FA_FB_l545_545068


namespace ivan_total_pay_l545_545054

theorem ivan_total_pay (cost_per_card : ℕ) (number_of_cards : ℕ) (discount_per_card : ℕ) :
  cost_per_card = 12 → number_of_cards = 10 → discount_per_card = 2 →
  (number_of_cards * (cost_per_card - discount_per_card)) = 100 :=
by
  intro h1 h2 h3
  sorry

end ivan_total_pay_l545_545054


namespace complex_product_conjugate_l545_545365

def i : ℂ := complex.I

def z : ℂ := ( (1 - i) / (1 + i))^2016 + i

def z_conjugate : ℂ := conj z

theorem complex_product_conjugate : z * z_conjugate = 2 := by 
  sorry

end complex_product_conjugate_l545_545365


namespace largest_of_five_consecutive_even_integers_l545_545161

theorem largest_of_five_consecutive_even_integers :
  (∃ n, 2 + 4 + ... + 50 = 5 * n - 20) →
  (∃ n, ∑ i in range 25, 2 * (i + 1) = 5 * n - 20) →
  (∃ n, n = 134) :=
begin
  sorry
end

end largest_of_five_consecutive_even_integers_l545_545161


namespace right_triangles_with_leg_2012_l545_545521

theorem right_triangles_with_leg_2012 :
  ∃ (a b c : ℕ), (a = 2012 ∧ (a^2 + b^2 = c^2 ∨ b^2 + a^2 = c^2)) ∧
  (b = 253005 ∧ c = 253013 ∨ b = 506016 ∧ c = 506020 ∨ b = 1012035 ∧ c = 1012037 ∨ b = 1509 ∧ c = 2515) :=
begin
  sorry
end

end right_triangles_with_leg_2012_l545_545521


namespace count_four_digit_numbers_divisible_by_five_ending_45_l545_545429

-- Define the conditions as necessary in Lean
def is_four_digit_number (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

def is_divisible_by_five (n : ℕ) : Prop :=
  n % 5 = 0

def ends_with_45 (n : ℕ) : Prop :=
  n % 100 = 45

-- Statement that there exists 90 such four-digit numbers
theorem count_four_digit_numbers_divisible_by_five_ending_45 : 
  { n : ℕ // is_four_digit_number n ∧ is_divisible_by_five n ∧ ends_with_45 n }.card = 90 :=
sorry

end count_four_digit_numbers_divisible_by_five_ending_45_l545_545429


namespace log_base_5_domain_correct_l545_545275

def log_base_5_domain : Set ℝ := {x : ℝ | x > 0}

theorem log_base_5_domain_correct : (∀ x : ℝ, x > 0 ↔ x ∈ log_base_5_domain) :=
by sorry

end log_base_5_domain_correct_l545_545275


namespace pizza_slices_l545_545103

theorem pizza_slices (P T S : ℕ) (h1 : P = 2) (h2 : T = 16) : S = 8 :=
by
  -- to be filled in
  sorry

end pizza_slices_l545_545103


namespace eventually_first_l545_545632

def operation (seq : List ℕ) : List ℕ :=
  match seq with
  | [] => []
  | k::ks => (k::ks).take k.reverse ++ (k::ks).drop k

theorem eventually_first (seq : List (Fin 1994)) :
  (∀ (k : Nat), k ∈ seq -> 1 ≤ k ∧ k ≤ 1993) ∧
  (∀ (n : ℕ), n ≠ 1 → ∃ m ≥ n, 1 ≠ m ∧ operation (seq.take m) = (operation (seq.take m)).reverse) →
  ∃ n, operation^[n] seq.head = 1 :=
  sorry

end eventually_first_l545_545632


namespace count_desired_property_l545_545931

-- Define the property that a number is a three-digit positive integer
def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000

-- Define the property that a number contains at least one specific digit
def contains_digit (d : ℕ) (n : ℕ) : Prop :=
  list.any (nat.digits 10 n) (λ m, m = d)

-- Define the property that a number does not contain a specific digit
def does_not_contain_digit (d : ℕ) (n : ℕ) : Prop :=
  ¬ list.any (nat.digits 10 n) (λ m, m = d)

-- Define the overall property for the desired number
def desired_property (n : ℕ) : Prop :=
  is_three_digit n ∧ contains_digit 4 n ∧ does_not_contain_digit 6 n

-- State the theorem to prove the count of numbers with the desired property
theorem count_desired_property : finset.card (finset.filter desired_property (finset.range 1000)) = 200 := by
  sorry

end count_desired_property_l545_545931


namespace max_min_values_max_omega_increasing_l545_545892

noncomputable def f (x : ℝ) (ω : ℝ) : ℝ := 
  4 * sin (ω * x) * cos (ω * x + π / 3) + 2 * sqrt 3

theorem max_min_values 
  (ω : ℝ) (hω : ω = 1)
  (x : ℝ) (hx : x ∈ [ -π/4, π/6 ]) :
  f x 1 ≤ 2 + sqrt 3 ∧ f x 1 ≥ sqrt 3 - 1 ∧ 
  ((f x 1 = 2 + sqrt 3) ↔ x = π/12) ∧
  ((f x 1 = sqrt 3 - 1) ↔ x = -π/4) :=
sorry

theorem max_omega_increasing 
  (ω : ℝ) (hω_pos : ω > 0) :
  (∀ x ∈ [-π/4, π/6], 0 ≤ diff (f x ω)) → ω ≤ 1/2 :=
sorry

end max_min_values_max_omega_increasing_l545_545892


namespace range_of_a_l545_545380

def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

theorem range_of_a (a : ℝ) 
  (h : a ∈ (Set.Iic (-2) ∪ Set.Ioi (-1))) : f a + 1 ≥ f (a + 1) :=
  sorry

end range_of_a_l545_545380


namespace probability_A_does_not_lose_l545_545478

theorem probability_A_does_not_lose (pA_wins p_draw : ℝ) (hA_wins : pA_wins = 0.4) (h_draw : p_draw = 0.2) :
  pA_wins + p_draw = 0.6 :=
by
  sorry

end probability_A_does_not_lose_l545_545478


namespace motorcycle_license_combinations_l545_545771

theorem motorcycle_license_combinations : 
  let letters := 3 in
  let digits_per_position := 10 in
  let positions := 4 in
  letters * digits_per_position ^ positions = 30000 := by
    sorry

end motorcycle_license_combinations_l545_545771


namespace b_work_days_l545_545231

theorem b_work_days (W : ℝ) (a_rate : ℝ) (ab_rate : ℝ) (b_days : ℝ) :
    a_rate = W / 6 → ab_rate = W / 2.4 → b_days = 4 :=
by
  intros ha hab
  have hWpos : W > 0 := sorry  -- Assume W is positive as it's a unit of work
  have hb : b_rate = W / 4, from sorry  -- B's rate of work based on solution steps (derivation skipped)
  have b_days_calc : b_days = W / (W / 4), from sorry  -- Days for B to complete the work using the derived rate
  rw [div_div_eq_mul, mul_one_div_cancel] at b_days_calc
  rw [hb, ha, hab]
  exact b_days_calc
  assumption

end b_work_days_l545_545231


namespace problem_solution_l545_545339

theorem problem_solution (a b : ℤ) (h : (λ x : ℂ, x^2 + x - 1) ∣ (λ x : ℂ, a * x^3 - b * x^2 + 1)) : a = 2 := 
by {
    sorry
}

end problem_solution_l545_545339


namespace flower_pot_arrangements_l545_545852

/-- From 7 different pots of flowers, 5 are to be selected and placed 
in front of the podium such that two specific pots are not allowed to be placed 
in the very center. Prove that the number of different arrangements is 1800. -/
theorem flower_pot_arrangements : 
  let pots := {1, 2, 3, 4, 5, 6, 7} in
  let center_not_allowed := {1, 2} in
  finset.card {arr : finset (finset ℕ) // 
    arr ∈ (finset.powerset pots) ∧ 
    finset.card arr = 5 ∧ 
    ∃ center, center ∉ center_not_allowed} = 1800 :=
sorry

end flower_pot_arrangements_l545_545852


namespace second_train_speed_l545_545214

theorem second_train_speed :
  ∃ v : ℝ, (∀ t₁ t₂ : ℝ, (t₁ = 64 ∧ t₂ = v) → 
  ∀ t : ℝ, t = 2.5 → (t * t₁ + t * t₂ = 285) ) :=
begin
  use 50,
  intros t₁ t₂ h_t h_time,
  cases h_t with h_t1 h_t2,
  rw [h_t1, h_t2, h_time],
  linarith,
end

end second_train_speed_l545_545214


namespace arithmetic_sequence_100th_term_l545_545297

-- Define the first term and the common difference
def first_term : ℕ := 3
def common_difference : ℕ := 7

-- Define the formula for the nth term of an arithmetic sequence
def nth_term (a1 d n : ℕ) : ℕ := a1 + (n - 1) * d

-- Theorem: The 100th term of the arithmetic sequence is 696.
theorem arithmetic_sequence_100th_term :
  nth_term first_term common_difference 100 = 696 :=
  sorry

end arithmetic_sequence_100th_term_l545_545297


namespace find_angle_θ_l545_545914

variables {V : Type*} [inner_product_space ℝ V]
variables (a b c d : V)
variables (θ : ℝ)

-- conditions
axiom norm_a : ∥a∥ = 3
axiom norm_b : ∥b∥ = 3
axiom norm_c : ∥c∥ = 4
axiom norm_d : ∥d∥ = 2
axiom vector_relation : a × (a × c) + 2 • b = d
axiom angle_ab : real.angle a b = real.pi / 4

-- question: prove possible values of θ
theorem find_angle_θ : ∃ θ, θ = some_angle_values :=
sorry

end find_angle_θ_l545_545914


namespace servant_salary_excluding_turban_l545_545916

theorem servant_salary_excluding_turban:
  ∀ (annual_salary : ℕ) (turban_worth : ℕ) (months_worked : ℕ), 
    annual_salary = 90 →
    turban_worth = 10 →
    months_worked = 9 →
    let total_annual_salary := annual_salary + turban_worth in
    let entitled_salary := (3 / 4 : ℚ) * total_annual_salary in
    let received_money := entitled_salary - turban_worth in
    received_money = 65 :=
by
  sorry

end servant_salary_excluding_turban_l545_545916


namespace mod_product_2023_2024_2025_2026_l545_545802

theorem mod_product_2023_2024_2025_2026 :
  (2023 * 2024 * 2025 * 2026) % 7 = 0 :=
by
  have h2023 : 2023 % 7 = 6 := by norm_num
  have h2024 : 2024 % 7 = 0 := by norm_num
  have h2025 : 2025 % 7 = 1 := by norm_num
  have h2026 : 2026 % 7 = 2 := by norm_num
  calc
    (2023 * 2024 * 2025 * 2026) % 7
      = ((2023 % 7) * (2024 % 7) * (2025 % 7) * (2026 % 7)) % 7 : by rw [Nat.mul_mod, Nat.mul_mod, Nat.mul_mod, Nat.mul_mod]
  ... = (6 * 0 * 1 * 2) % 7 : by rw [h2023, h2024, h2025, h2026]
  ... = 0 % 7 : by norm_num
  ... = 0 : by norm_num

end mod_product_2023_2024_2025_2026_l545_545802


namespace least_days_to_repay_twice_l545_545554

-- Define the initial conditions
def borrowed_amount : ℝ := 15
def daily_interest_rate : ℝ := 0.10
def interest_per_day : ℝ := borrowed_amount * daily_interest_rate
def total_amount_to_repay : ℝ := 2 * borrowed_amount

-- Define the condition we want to prove
theorem least_days_to_repay_twice : ∃ (x : ℕ), (borrowed_amount + interest_per_day * x) ≥ total_amount_to_repay ∧ x = 10 :=
by
  sorry

end least_days_to_repay_twice_l545_545554


namespace tree_cost_l545_545918

theorem tree_cost (fence_length_yards : ℝ) (tree_width_feet : ℝ) (total_cost : ℝ) 
(h1 : fence_length_yards = 25) 
(h2 : tree_width_feet = 1.5) 
(h3 : total_cost = 400) : 
(total_cost / ((fence_length_yards * 3) / tree_width_feet) = 8) := 
by
  sorry

end tree_cost_l545_545918


namespace length_of_A_l545_545080

-- Define points A, B, and C as given in the problem
def A : ℝ × ℝ := (0, 10)
def B : ℝ × ℝ := (0, 15)
def C : ℝ × ℝ := (3, 9)

-- Define conditions explicitly
def on_line_y_eq_x (P : ℝ × ℝ) : Prop := P.1 = P.2
def intersects_at (A A' : ℝ × ℝ) (B B' : ℝ × ℝ) (C : ℝ × ℝ) : Prop :=
  ∃ k l : ℝ, A'.1 = A.1 + k * (C.1 - A.1) ∧ A'.2 = A.2 + k * (C.2 - A.2) ∧
             B'.1 = B.1 + l * (C.1 - B.1) ∧ B'.2 = B.2 + l * (C.2 - B.2)

-- Statement of the proof problem
theorem length_of_A'B' :
  ∃ A' B' : ℝ × ℝ, on_line_y_eq_x A' ∧ on_line_y_eq_x B' ∧ 
  intersects_at A A' B B' C ∧ (dist A' B' = 5 * real.sqrt 2) := by
  sorry

end length_of_A_l545_545080


namespace complement_union_l545_545242

universe u

variable {α : Type u} (U A B : Set α)

theorem complement_union {α : Type u} (U : Set α) (A B : Set α) (hU : U = {a, b, c, d}) (hA : A = {a, b}) (hB : B = {b, c, d}) : 
  (U \ A) ∪ (U \ B) = {a, c, d} :=
by
  sorry

end complement_union_l545_545242


namespace product_of_random_numbers_greater_zero_l545_545203

noncomputable def random_product_positive_probability : ℝ := 
  let interval_length := 45
  let neg_interval_length := 30
  let pos_interval_length := 15
  let prob_neg := (neg_interval_length : ℝ) / interval_length
  let prob_pos := (pos_interval_length : ℝ) / interval_length
  prob_pos * prob_pos + prob_neg * prob_neg

-- Prove that the probability that the product of two randomly selected numbers
-- from the interval [-30, 15] is greater than zero is 5/9.
theorem product_of_random_numbers_greater_zero : 
  random_product_positive_probability = 5 / 9 := by
  sorry

end product_of_random_numbers_greater_zero_l545_545203


namespace cloak_change_in_silver_l545_545487

theorem cloak_change_in_silver :
  (∀ c : ℤ, (20 = c + 4) → (15 = c + 1)) →
  (5 * g = 3) →
  14 * gold / exchange_rate = 10 := 
sorry

end cloak_change_in_silver_l545_545487


namespace final_solution_concentration_l545_545739

def concentration (mass : ℕ) (volume : ℕ) : ℕ := 
  (mass * 100) / volume

theorem final_solution_concentration :
  let volume1 := 4
  let conc1 := 4 -- percentage
  let volume2 := 2
  let conc2 := 10 -- percentage
  let mass1 := volume1 * conc1 / 100
  let mass2 := volume2 * conc2 / 100
  let total_mass := mass1 + mass2
  let total_volume := volume1 + volume2
  concentration total_mass total_volume = 6 :=
by
  sorry

end final_solution_concentration_l545_545739


namespace product_positive_probability_l545_545212

theorem product_positive_probability :
  let interval := set.Icc (-30 : ℝ) 15 in
  let prob_neg := (30 : ℝ) / 45 in
  let prob_pos := (15 : ℝ) / 45 in
  let prob_product_neg := 2 * (prob_neg * prob_pos) in
  let prob_product_pos := (prob_neg ^ 2) + (prob_pos ^ 2) in
  (prob_product_pos = 5 / 9) :=
by
  sorry

end product_positive_probability_l545_545212


namespace inverse_of_p_l545_545038

variables {p q r : Prop}

theorem inverse_of_p (m n : Prop) (hp : p = (m → n)) (hq : q = (¬m → ¬n)) (hr : r = (n → m)) : r = p ∧ r = (n → m) :=
by
  sorry

end inverse_of_p_l545_545038


namespace correct_choice_l545_545779

-- Definition of the functions
def f1 (x : ℝ) : ℝ := log 2 (2 ^ (-x))
def f2 (x : ℝ) : ℝ := (1 / 2) ^ (-x)
def f3 (x : ℝ) : ℝ := 1 / (x + 1)
def f4 (x : ℝ) : ℝ := x ^ 2

-- Definition that checks if a function is decreasing on ℝ
def is_decreasing_on_ℝ (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f x > f y

-- Theorem stating that f1 is the decreasing function on ℝ among the given functions
theorem correct_choice :
  is_decreasing_on_ℝ f1 ∧
  ¬ is_decreasing_on_ℝ f2 ∧
  ¬ is_decreasing_on_ℝ f3 ∧
  ¬ is_decreasing_on_ℝ f4 :=
sorry -- Proof is omitted

end correct_choice_l545_545779


namespace decrease_in_B_share_l545_545119

theorem decrease_in_B_share (a b c : ℝ) (x : ℝ) 
  (h1 : c = 495)
  (h2 : a + b + c = 1010)
  (h3 : (a - 25) / 3 = (b - x) / 2)
  (h4 : (a - 25) / 3 = (c - 15) / 5) :
  x = 10 :=
by
  sorry

end decrease_in_B_share_l545_545119


namespace pyramid_volume_l545_545845

noncomputable def volume_of_pyramid (a b c d: ℝ) (diagonal: ℝ) (angle: ℝ) : ℝ :=
  if (a = 10 ∧ d = 10 ∧ b = 5 ∧ c = 5 ∧ diagonal = 4 * Real.sqrt 5 ∧ angle = 45) then
    let base_area := 1 / 2 * (diagonal) * (Real.sqrt ((c * c) + (b * b)))
    let height := 10 / 3
    let volume := 1 / 3 * base_area * height
    volume
  else 0

theorem pyramid_volume :
  volume_of_pyramid 10 5 5 10 (4 * Real.sqrt 5) 45 = 500 / 9 :=
by
  sorry

end pyramid_volume_l545_545845


namespace find_coefficients_l545_545581

variables {V : Type*} [AddCommGroup V] [Module ℝ V]

theorem find_coefficients 
  (A B Q C P : V) 
  (hQ : Q = (5 / 7 : ℝ) • A + (2 / 7 : ℝ) • B)
  (hC : C = A + 2 • B)
  (hP : P = Q + C) : 
  ∃ s v : ℝ, P = s • A + v • B ∧ s = 12 / 7 ∧ v = 16 / 7 :=
by
  sorry

end find_coefficients_l545_545581


namespace count_valid_numbers_l545_545394

-- Define what it means to be a four-digit number that ends in 45
def is_valid_number (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999 ∧ n % 100 = 45 ∧ n % 5 = 0

-- Define the set of valid two-digit prefixes
def valid_prefixes : set ℕ := {ab | 10 ≤ ab ∧ ab ≤ 99}

-- Define the set of four-digit numbers that end in 45 and are divisible by 5
def valid_numbers : set ℕ := {n | ∃ ab : ℕ, ab ∈ valid_prefixes ∧ n = ab * 100 + 45}

-- State the theorem
theorem count_valid_numbers : (finset.card (finset.filter is_valid_number (finset.range 10000)) = 90) :=
sorry

end count_valid_numbers_l545_545394


namespace ratio_of_flowers_given_l545_545799

-- Definitions based on conditions
def Collin_flowers : ℕ := 25
def Ingrid_flowers_initial : ℕ := 33
def petals_per_flower : ℕ := 4
def Collin_petals_total : ℕ := 144

-- The ratio of the number of flowers Ingrid gave to Collin to the number of flowers Ingrid had initially
theorem ratio_of_flowers_given :
  let Ingrid_flowers_given := (Collin_petals_total - (Collin_flowers * petals_per_flower)) / petals_per_flower
  let ratio := Ingrid_flowers_given / Ingrid_flowers_initial
  ratio = 1 / 3 :=
by
  sorry

end ratio_of_flowers_given_l545_545799


namespace right_triangle_area_l545_545693

/-- Given a right triangle with one leg of length 30 inches and a hypotenuse of 34 inches,
    the area of the triangle is 240 square inches. -/
theorem right_triangle_area (a b c : ℝ) (h1 : a = 30) (h2 : c = 34) (h3 : a^2 + b^2 = c^2) :
  (1/2) * a * b = 240 :=
by
  rw [h1, h2] at h3
  have hb : b = 16 := by
    rw [←h3]
    norm_num
  rw [h1, hb]
  norm_num
  sorry

end right_triangle_area_l545_545693


namespace convert_724_base_9_to_base_5_l545_545301

namespace base_conversion

def base_n_to_decimal (n : ℕ) (digits : List ℕ) : ℕ :=
  List.foldr (λ (d acc : ℕ), d + n * acc) 0 digits

def decimal_to_base_n (n : ℕ) (num : ℕ) : List ℕ :=
  if num = 0 then [0]
  else List.reverse (List.unfoldr (λ x, if x = 0 then none else some (x % n, x / n)) num)

theorem convert_724_base_9_to_base_5 :
  let num_in_base_9 := [7, 2, 4]
  let num_in_decimal := base_n_to_decimal 9 num_in_base_9
  let num_in_base_5 := decimal_to_base_n 5 num_in_decimal
  num_in_base_5 = [4, 3, 2, 4] :=
by
  sorry

end base_conversion

end convert_724_base_9_to_base_5_l545_545301


namespace parabola_focus_l545_545317

/-- Given the equation of a parabola, the focus is calculated and verified. -/
theorem parabola_focus 
  (h k a : ℝ) (h_eq : h = 1) (k_eq : k = -3) (a_eq : a = 2) :
  let y := λ x : ℝ, a * x^2 + b * x + c
  ∃ focus : ℝ × ℝ, focus = (1, -23/8) :=
by
  -- Definitions and conditions from the problem
  sorry

end parabola_focus_l545_545317


namespace prod_mod7_eq_zero_l545_545807

theorem prod_mod7_eq_zero :
  (2023 * 2024 * 2025 * 2026) % 7 = 0 := 
by {
  sorry
}

end prod_mod7_eq_zero_l545_545807


namespace fencing_cost_proof_l545_545638

theorem fencing_cost_proof (L : ℝ) (B : ℝ) (c : ℝ) (total_cost : ℝ)
  (hL : L = 60) (hL_B : L = B + 20) (hc : c = 26.50) : 
  total_cost = 5300 :=
by
  sorry

end fencing_cost_proof_l545_545638


namespace range_of_a_l545_545004

theorem range_of_a (a : ℝ) (h1 : a ≠ 0) 
    (h2 : ∀ x : ℝ, (ax - 1) * (x + 1) < 0 ↔ 
                    x ∈ Set.Ioo (-∞) (1 / a) ∪ Set.Ioo (0) (∞)) :
    -1 ≤ a ∧ a < 0 := 
begin
  sorry
end

end range_of_a_l545_545004


namespace count_four_digit_numbers_divisible_by_5_and_ending_with_45_l545_545407

def is_four_digit_number (n : ℕ) : Prop := 1000 ≤ n ∧ n < 10000
def ends_with_45 (n : ℕ) : Prop := (n % 100) = 45 
def divisible_by_5 (n : ℕ) : Prop := (n % 5) = 0

theorem count_four_digit_numbers_divisible_by_5_and_ending_with_45 : 
  {n : ℕ | is_four_digit_number n ∧ ends_with_45 n ∧ divisible_by_5 n}.to_finset.card = 90 :=
sorry

end count_four_digit_numbers_divisible_by_5_and_ending_with_45_l545_545407


namespace find_m_value_l545_545578

theorem find_m_value (m : ℚ) :
  (m * 2 / 3 + m * 4 / 9 + m * 8 / 27 = 1) → m = 27 / 38 :=
by 
  intro h
  sorry

end find_m_value_l545_545578


namespace silver_coins_change_l545_545507

-- Define the conditions
def condition1 : ℕ × ℕ := (20, 4) -- (20 silver coins, 4 gold coins change)
def condition2 : ℕ × ℕ := (15, 1) -- (15 silver coins, 1 gold coin change)
def cost_of_cloak_in_gold_coins : ℕ := 14

-- Define the theorem to be proven
theorem silver_coins_change (s1 g1 s2 g2 cloak_g : ℕ) (h1 : (s1, g1) = condition1) (h2 : (s2, g2) = condition2) :
  ∃ silver : ℕ, (silver = 10) :=
by {
  sorry
}

end silver_coins_change_l545_545507


namespace sequence_sum_eq_l545_545718

def sequence (n : Nat) : Int :=
  if n % 2 = 0 then (n/2 : Int) + 4
  else -(n/2 : Int) - 3

def sequence_sum : Int :=
  ((0: Int) to 4000 by 1).map (fun i => sequence i).sum

theorem sequence_sum_eq : sequence_sum = 2000 := by
  sorry

end sequence_sum_eq_l545_545718


namespace opposite_of_neg_five_l545_545641

theorem opposite_of_neg_five : ∃ (y : ℤ), -5 + y = 0 ∧ y = 5 :=
by
  use 5
  simp

end opposite_of_neg_five_l545_545641


namespace grade_on_second_test_l545_545791

variable (first_test_grade second_test_average : ℕ)
#check first_test_grade
#check second_test_average

theorem grade_on_second_test :
  first_test_grade = 78 →
  second_test_average = 81 →
  (first_test_grade + (second_test_average * 2 - first_test_grade)) / 2 = second_test_average →
  second_test_grade = 84 :=
by
  intros h1 h2 h3
  sorry

end grade_on_second_test_l545_545791


namespace smallest_coprime_gt_one_l545_545710

theorem smallest_coprime_gt_one :
  ∃ n : ℕ, n > 1 ∧ n < 14 ∧ gcd n 2310 = 1 := 
begin
  use 13,
  split,
  { exact dec_trivial, -- 13 > 1
  },
  split,
  { exact dec_trivial, -- 13 < 14
  },
  { exact dec_trivial, -- gcd(13, 2310) = 1
  }
end

end smallest_coprime_gt_one_l545_545710


namespace locus_of_center_of_equilateral_triangle_l545_545846

-- Define the conditions.
structure Point :=
  (x : ℝ)
  (y : ℝ)

def line_through (P : Point) (slope : ℝ) : ℝ → ℝ :=
  λ x, slope * (x - P.x) + P.y

def passes_through (P : Point) (l : ℝ → ℝ) : Prop :=
  l P.x = P.y

-- Define the equilateral triangle condition.
def is_equilateral (A B C : Point) : Prop :=
  ∀ (P Q R : Point), (P = A ∧ Q = B ∧ R = C) ∨ (P = A ∧ Q = C ∧ R = B) ∨ (P = B ∧ Q = A ∧ R = C) ∨ (P = B ∧ Q = C ∧ R = A) ∨ (P = C ∧ Q = A ∧ R = B) ∨ (P = C ∧ Q = B ∧ R = A) →
  dist P Q = dist Q R ∧ dist Q R = dist R P

-- Define the orthocenter computation (dummy implementation here for structure).
def find_orthocenter (A B C : Point) : Point :=
  sorry

-- Prove that the locus of the center of the equilateral triangle is y = -1/4.
theorem locus_of_center_of_equilateral_triangle :
  ∀ a1 a2 a3 : ℝ,
    let P1 := Point.mk a1 (a1^2),
        P2 := Point.mk a2 (a2^2),
        P3 := Point.mk a3 (a3^2),
        l1 := line_through P1 (2 * a1),
        l2 := line_through P2 (2 * a2),
        l3 := line_through P3 (2 * a3),
        A := find_orthocenter P1 P2 P3
    in passes_through P1 l1 ∧ passes_through P2 l2 ∧ passes_through P3 l3 ∧
       is_equilateral P1 P2 P3 → A.y = -1/4 :=
begin
  intros a1 a2 a3 P1 P2 P3 l1 l2 l3 A h,
  sorry
end

end locus_of_center_of_equilateral_triangle_l545_545846


namespace value_of_f_neg_a_l545_545138

def f (x : ℝ) : ℝ := x^3 + Real.sin x + 1

theorem value_of_f_neg_a (a : ℝ) (h : f a = 2) : f (-a) = 0 :=
by
  sorry

end value_of_f_neg_a_l545_545138


namespace discount_savings_l545_545634

theorem discount_savings :
  let price := 30
  let discount1 := 5
  let discount2 := 0.25
  let cost_a := (price - discount1) * (1 - discount2)
  let cost_b := (price * (1 - discount2)) - discount1
  cost_a - cost_b = 1.25 :=
by
  let price := 30
  let discount1 := 5
  let discount2 := 0.25
  let cost_a := (price - discount1) * (1 - discount2)
  let cost_b := (price * (1 - discount2)) - discount1
  have h_cost_a : cost_a = 18.75 := sorry
  have h_cost_b : cost_b = 17.50 := sorry
  suffices : 18.75 - 17.50 = 1.25 by sorry
  exact sorry

end discount_savings_l545_545634


namespace ab_zero_proof_l545_545768

-- Given conditions
def square_side : ℝ := 3
def rect_short_side : ℝ := 3
def rect_long_side : ℝ := 6
def rect_area : ℝ := rect_short_side * rect_long_side
def split_side_proof (a b : ℝ) : Prop := a + b = rect_short_side

-- Lean theorem proving that ab = 0 given the conditions
theorem ab_zero_proof (a b : ℝ) 
  (h1 : square_side = 3)
  (h2 : rect_short_side = 3)
  (h3 : rect_long_side = 6)
  (h4 : rect_area = 18)
  (h5 : split_side_proof a b) : a * b = 0 := by
  sorry

end ab_zero_proof_l545_545768


namespace problem_statement_l545_545893

open Real

-- Definition of the function
def f (x : ℝ) := 4 * sin x * cos (x - π / 3) - sqrt 3

noncomputable def smallest_positive_period : ℝ :=
  π

-- Definition of the intervals where the function is monotonically decreasing
def decreasing_intervals (k : ℤ) : Set ℝ :=
  {x | k * π + 5 * π / 12 ≤ x ∧ x ≤ k * π + 11 * π / 12}

-- Problem Statement: Prove the smallest positive period and the decreasing intervals
theorem problem_statement :
  (∀ x, f (x + smallest_positive_period) = f x) ∧
  (∀ k : ℤ, ∀ x : ℝ, x ∈ decreasing_intervals k ↔ f' x < 0) :=
by
  sorry

end problem_statement_l545_545893


namespace max_four_sides_of_convex_lattice_polygon_l545_545606

-- Define lattice points as tuples of integers.
def is_lattice_point (p : ℤ × ℤ) : Prop := true

-- Define a convex polygon with vertices on lattice points.
structure lattice_polygon :=
  (n : ℕ) -- number of vertices
  (vertices : fin n → ℤ × ℤ) -- vertices are lattice points
  (convex : Prop) -- polygon is convex
  (no_inner_lattice_points : Prop) -- no lattice points inside the polygon
  (no_edge_lattice_points : Prop) -- no lattice points on the edges except vertices

-- Statement of the problem:
theorem max_four_sides_of_convex_lattice_polygon
  (P : lattice_polygon)
  (h₁ : P.convex)
  (h₂ : P.no_inner_lattice_points)
  (h₃ : P.no_edge_lattice_points) : 
  P.n ≤ 4 :=
sorry

end max_four_sides_of_convex_lattice_polygon_l545_545606


namespace sum_of_reciprocals_B_is_77_div_10_l545_545563

noncomputable def infinite_sum_B : ℚ :=
  let B := { n : ℕ | ∀ p ∣ n, p = 2 ∨ p = 3 ∨ p = 7 ∨ p = 11 } in
  ∑ n in B, (1 / n)

theorem sum_of_reciprocals_B_is_77_div_10 :
  infinite_sum_B = 77 / 10 :=
sorry

end sum_of_reciprocals_B_is_77_div_10_l545_545563


namespace min_cards_needed_l545_545600

/-- 
On a table, there are five types of number cards: 1, 3, 5, 7, and 9, with 30 cards of each type. 
Prove that the minimum number of cards required to ensure that the sum of the drawn card numbers 
can represent all integers from 1 to 200 is 26.
-/
theorem min_cards_needed : ∀ (cards_1 cards_3 cards_5 cards_7 cards_9 : ℕ), 
  cards_1 = 30 → cards_3 = 30 → cards_5 = 30 → cards_7 = 30 → cards_9 = 30 → 
  ∃ n, (n = 26) ∧ 
  (∀ k, 1 ≤ k ∧ k ≤ 200 → 
    ∃ a b c d e, 
      a ≤ cards_1 ∧ b ≤ cards_3 ∧ c ≤ cards_5 ∧ d ≤ cards_7 ∧ e ≤ cards_9 ∧ 
      k = a * 1 + b * 3 + c * 5 + d * 7 + e * 9) :=
by {
  sorry
}

end min_cards_needed_l545_545600


namespace smallest_common_factor_l545_545712

theorem smallest_common_factor : ∃ (n : ℕ), n > 0 ∧ (8 * n - 3).gcd (6 * n + 5) > 1 ∧ n = 33 := 
by
  use 33
  split
  · exact nat.zero_lt_succ 32
  split
  · sorry -- Proof that the gcd is greater than 1
  · rfl -- Proof that the integer is 33

end smallest_common_factor_l545_545712


namespace smallest_period_f_pi_intervals_of_monotonic_increase_range_of_m_l545_545000

noncomputable def f (x : ℝ) : ℝ :=
  cos (2 * x - π / 3) + sin x ^ 2 - cos x ^ 2 + sqrt 2

theorem smallest_period_f_pi :
  (∀ x : ℝ, f (x + π) = f x) ∧
  (∀ T > 0, (∀ x : ℝ, f (x + T) = f x) → T ≥ π) := 
sorry

theorem intervals_of_monotonic_increase (k : ℤ) :
  ∃ x : ℝ, 
  (k * π - π / 6 ≤ x ∧ x ≤ k * π + π / 3) ∧
  (∀ y : ℝ, 
    k * π - π / 6 ≤ y ∧ y ≤ k * π + π / 3 → f y < f (y + ε) → 
    ∃ ε > 0) := 
sorry

theorem range_of_m :
  (∃ t : ℝ, t ∈ Icc (π / 12) (π / 3) ∧
  (f t) ^ 2 - 2 * sqrt 2 * f t - m > 0) ↔
  m ∈ Iio (-1) := 
sorry

end smallest_period_f_pi_intervals_of_monotonic_increase_range_of_m_l545_545000


namespace compute_sum_of_cubes_l545_545094

-- Define the polynomial
def cubic_polynomial (x : ℝ) : ℝ := 3 * x^3 + 4 * x^2 - 200 * x + 5

-- Define the roots
variables (p q r : ℝ)
hypothesis roots_of_polynomial : cubic_polynomial p = 0 ∧ cubic_polynomial q = 0 ∧ cubic_polynomial r = 0

-- Define the proof problem
theorem compute_sum_of_cubes : (p+q+1)^3 + (q+r+1)^3 + (r+p+1)^3 = 24 :=
by {
  sorry -- Proof omitted
}

end compute_sum_of_cubes_l545_545094


namespace silver_coins_change_l545_545504

-- Define the conditions
def condition1 : ℕ × ℕ := (20, 4) -- (20 silver coins, 4 gold coins change)
def condition2 : ℕ × ℕ := (15, 1) -- (15 silver coins, 1 gold coin change)
def cost_of_cloak_in_gold_coins : ℕ := 14

-- Define the theorem to be proven
theorem silver_coins_change (s1 g1 s2 g2 cloak_g : ℕ) (h1 : (s1, g1) = condition1) (h2 : (s2, g2) = condition2) :
  ∃ silver : ℕ, (silver = 10) :=
by {
  sorry
}

end silver_coins_change_l545_545504


namespace ellipse_C2_equation_line_AB_equation_l545_545872

theorem ellipse_C2_equation :
  (∀ x y : ℝ, (x^2 / 4 + y^2 = 1) →
  (let C1_major := 4; C1_eccentricity := (sqrt 3) / 2 in
  (∀ a b c d : ℝ,
  C1_major = 4 → C1_eccentricity = (sqrt 3) / 2 →
  b = 2 → a = 4 →
  (b^2 = a^2 * (1 - C1_eccentricity^2) → 
  (y^2 / 16 + x^2 / 4 = 1))))) :=
sorry

theorem line_AB_equation :
  (∀ x y : ℝ, (x^2 / 4 + y^2 = 1) →
  (let C1_major := 4; C1_eccentricity := (sqrt 3) / 2 in
  (∀ a b c d : ℝ,
  C1_major = 4 → C1_eccentricity = (sqrt 3) / 2 →
  b = 2 → a = 4 →
  (b^2 = a^2 * (1 - C1_eccentricity^2) →
  (let x_A y_A x_B y_B : ℝ in
  x^2 / 4 + y^2 = 1 →
  ∀ k : ℝ, (x_B = 2 * x_A → y_B = 2 * y_A →
  (1 + 4 * k^2) * x_A^2 = 4 →
  (4 + k^2) * x_B^2 = 16 →
  (k = 1 ∨ k = -1) →
  (y = x ∨ y = -x))))) :=
sorry

end ellipse_C2_equation_line_AB_equation_l545_545872


namespace A_eq_B_l545_545571

def is_power_of_two (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 2^k

def is_type_A_sequence (seq : List ℕ) (n : ℕ) : Prop :=
  0 < seq.length ∧
  (seq.sorted (· ≥ ·)) ∧
  (∀ i, i < seq.length → is_power_of_two (seq[i] + 1)) ∧
  (seq.sum = n)

def is_type_B_sequence (seq : List ℕ) (n : ℕ) : Prop :=
  0 < seq.length ∧
  (seq.sorted (· ≥ ·)) ∧
  (∀ j, j < seq.length - 1 → seq[j] ≥ 2 * seq[j + 1]) ∧
  (seq.sum = n)

def A (n : ℕ) : ℕ :=
  (List.filter (λ seq, is_type_A_sequence seq n) (List.range (n + 1))).length

def B (n : ℕ) : ℕ :=
  (List.filter (λ seq, is_type_B_sequence seq n) (List.range (n + 1))).length

theorem A_eq_B (n : ℕ) : A n = B n :=
  sorry

end A_eq_B_l545_545571


namespace cloak_change_14_gold_coins_l545_545515

def exchange_rate (silver gold : ℕ) : Prop :=
  ∃ c : ℕ, (20 - 4) * c = silver ∧ (15 - 1) * c = silver

def cloak_purchase (paid_gold received_silver : ℕ) : Prop :=
  let exchange_rate := (5 * 14) / 3 in
  received_silver = 2 * exchange_rate

theorem cloak_change_14_gold_coins :
  exchange_rate 16 3 →
  exchange_rate 14 1 →
  cloak_purchase 14 10 := sorry

end cloak_change_14_gold_coins_l545_545515


namespace largest_angle_of_triangle_l545_545164

theorem largest_angle_of_triangle (A B C : ℝ) :
  A + B + C = 180 ∧ A + B = 126 ∧ abs (A - B) = 45 → max A (max B C) = 85.5 :=
by sorry

end largest_angle_of_triangle_l545_545164


namespace compute_modulo_l545_545810

theorem compute_modulo :
  (2023 * 2024 * 2025 * 2026) % 7 = 0 := by
  sorry

end compute_modulo_l545_545810


namespace cloak_change_l545_545502

theorem cloak_change (silver_for_cloak1 silver_for_cloak2 : ℕ) (gold_change1 gold_change2 : ℕ) (silver_change : ℕ) :
  silver_for_cloak1 = 20 →
  gold_change1 = 4 →
  silver_for_cloak2 = 15 →
  gold_change2 = 1 →
  ∃ silver_cost_of_cloak : ℕ, 
    silver_cost_of_cloak = (20 - 4) * (5 / 3) →
    silver_change = 10 →
    14 * (5 / 3) - 8 = silver_change :=
by 
  assume h1 h2 h3 h4,
  use 16, 
  sorry

end cloak_change_l545_545502


namespace find_ABC_sum_l545_545629

noncomputable def g (x : ℝ) (A B C : ℤ) : ℝ := x^2 / (A * x^2 + B * x + C)

theorem find_ABC_sum :
  (∀ x : ℝ, x > 3 → g x 3 (-3) (-6) > 0.3) →
  (∃ A B C : ℤ, A * (x + 1) * (x - 2) = A * x^2 + B * x + C ∧ (∀ x : ℝ, g x A B C = x^2 / (A * x^2 + B * x + C)) 
  ∧ (∀ x : ℝ, x > 3 → g x A B C > 0.3) 
  ∧ (∀ y : ℝ, horizontal_asymptote g x = 1 / A))
  → ∀ A B C : ℤ, A = 3 → B = -3 → C = -6 → A + B + C = -6 :=
begin
  intros h,
  sorry
end

end find_ABC_sum_l545_545629


namespace probability_co_captains_l545_545655

theorem probability_co_captains :
  let teams := [(4, 1), (6, 2), (7, 2), (9, 3)] in
  let total_teams := length teams in
  (1 / 4) * ((1*(1-1) / (4*(4-1))) +
             (2*(2-1) / (6*(6-1))) +
             (2*(2-1) / (7*(7-1))) +
             (3*(3-1) / (9*(9-1)))) = 83 / 1680 :=
begin
  sorry
end

end probability_co_captains_l545_545655


namespace cannot_be_division_l545_545565

def P := {x : ℤ | ∃ k : ℤ, x = 2 * k}

theorem cannot_be_division (a b : ℤ) (ha : a ∈ P) (hb : b ∈ P) :
  (∀ a b ∈ P, a / b ∈ P) → False :=
by
  sorry

end cannot_be_division_l545_545565


namespace prod_mod7_eq_zero_l545_545808

theorem prod_mod7_eq_zero :
  (2023 * 2024 * 2025 * 2026) % 7 = 0 := 
by {
  sorry
}

end prod_mod7_eq_zero_l545_545808


namespace length_tangent_segment_l545_545871

theorem length_tangent_segment {x y : ℝ} (C : ∀ x y, (x - 4)^2 + (y + 2)^2 = 5) :
  let line := λ x, x + 2,
  let center := (4, -2 : ℝ),
  let radius := real.sqrt 5,
  let d := 4 * real.sqrt 2,
  let tangent_length := real.sqrt (d^2 - radius^2)
  in tangent_length = 3 * real.sqrt 3 :=
by {
  sorry
}

end length_tangent_segment_l545_545871


namespace isosceles_triangle_perimeter_ratio_l545_545060

theorem isosceles_triangle_perimeter_ratio
  {a b S : ℝ}
  (h1 : ∀ (ABC : Triangle), ABC.is_isosceles ∧ ABC.SideAB = ABC.SideAC ∧ ABC.angle_bisectors AA_1 BB_1 CC_1)
  (h2 : TrianglesArea ABC A_1B_1C_1 = 9/2 * Area ABC) :
  Perimeter A_1B_1C_1 / Perimeter ABC = (2 + real.sqrt 19) / 15 :=
sorry

end isosceles_triangle_perimeter_ratio_l545_545060


namespace domain_sqrt_sin_cos_range_cos2_sin_l545_545735

-- Domain of the function y = sqrt(sin x) + sqrt(1/2 - cos x)
theorem domain_sqrt_sin_cos (x : ℝ) (k : ℤ) :
  (2 * k * real.pi + real.pi / 3 <= x ∧ x <= 2 * k * real.pi + real.pi) ↔
    (sin x >= 0 ∧ 1 / 2 - cos x >= 0) := sorry

-- Range of the function y = cos^2 x - sin x for x in [-π/4, π/4]
theorem range_cos2_sin (y : ℝ) :
  (∀ x : ℝ, -real.pi / 4 <= x ∧ x <= real.pi / 4 →
    y = cos x ^ 2 - sin x) ↔
    (-real.sqrt 2 / 2 <= y ∧ y <= real.sqrt 2 / 2) := sorry

end domain_sqrt_sin_cos_range_cos2_sin_l545_545735


namespace distinct_flavors_l545_545358

variable (x y : ℕ)

theorem distinct_flavors (h₁ : 0 ≤ x ∧ x ≤ 7) (h₂ : 0 ≤ y ∧ y ≤ 5) (h₃ : ¬(x = 0 ∧ y = 0)) :
  {f : ℚ // ∃ (x₁ x₂ : ℕ), x₁ ≤ 7 ∧ y ≤ 5 ∧ f = (x₁ : ℚ) / (y₁ : ℚ) ∧ ¬(x₁ = 0 ∧ y₁ = 0)}.card = 41 :=  
sorry

end distinct_flavors_l545_545358


namespace log_product_eq_two_l545_545290

open Real

theorem log_product_eq_two
  : log 5 / log 3 * log 6 / log 5 * log 9 / log 6 = 2 := by
  sorry

end log_product_eq_two_l545_545290


namespace right_triangle_area_l545_545685

theorem right_triangle_area (leg1 hypotenuse : ℝ) (h1 : leg1 = 30) (h2 : hypotenuse = 34) (h3 : (leg1 ^ 2 + (sqrt (hypotenuse ^ 2 - leg1 ^ 2)) ^ 2 = hypotenuse ^ 2)) :
  (1 / 2) * leg1 * (sqrt (hypotenuse ^ 2 - leg1 ^ 2)) = 240 :=
by
  sorry

end right_triangle_area_l545_545685


namespace distinct_ways_to_place_digits_l545_545455

theorem distinct_ways_to_place_digits :
  let digits := {1, 2, 3, 4}
  let boxes := 5
  let empty_box := 1
  -- There are 5! permutations of the list [0, 1, 2, 3, 4]
  let total_digits := insert 0 digits
  -- Resulting in 120 ways to place these digits in 5 boxes
  nat.factorial boxes = 120 :=
by 
  sorry

end distinct_ways_to_place_digits_l545_545455


namespace problem_1_problem_2_l545_545348

def f (x : ℝ) : ℝ := sin (2 * x + π / 6) - cos (2 * x + π / 3) + 2 * cos x ^ 2

theorem problem_1 : f (π / 12) = sqrt 3 + 1 := sorry

theorem problem_2 : ∀ k : ℤ, (∀ x : ℝ, x = k * π + π / 6 → f x = 3) → (∀ x : ℝ, f x ≤ 3) := sorry

end problem_1_problem_2_l545_545348


namespace final_hair_length_l545_545117

-- Define the initial conditions and the expected final result.
def initial_hair_length : ℕ := 14
def hair_growth (x : ℕ) : ℕ := x
def hair_cut : ℕ := 20

-- Prove that the final hair length is x - 6.
theorem final_hair_length (x : ℕ) : initial_hair_length + hair_growth x - hair_cut = x - 6 :=
by
  sorry

end final_hair_length_l545_545117


namespace banana_count_l545_545284

theorem banana_count (a : ℕ) (S : ℕ) (h1 : (∑ i in Finset.range 7, a * 2^i) = S) (h2 : a * 2^6 = 128) : a = 2 :=
by
  sorry

end banana_count_l545_545284


namespace right_triangle_area_l545_545682

theorem right_triangle_area (leg1 hypotenuse : ℝ) (h1 : leg1 = 30) (h2 : hypotenuse = 34) (h3 : (leg1 ^ 2 + (sqrt (hypotenuse ^ 2 - leg1 ^ 2)) ^ 2 = hypotenuse ^ 2)) :
  (1 / 2) * leg1 * (sqrt (hypotenuse ^ 2 - leg1 ^ 2)) = 240 :=
by
  sorry

end right_triangle_area_l545_545682


namespace distribution_students_l545_545943

theorem distribution_students (total_candies candies_per_student : ℕ) (h_total : total_candies = 81) (h_per_student : candies_per_student = 9) :
  total_candies / candies_per_student = 9 :=
by
  rw [h_total, h_per_student]
  exact Nat.div_self sorry

end distribution_students_l545_545943


namespace elena_pen_cost_l545_545833

theorem elena_pen_cost (cost_X : ℝ) (cost_Y : ℝ) (total_pens : ℕ) (brand_X_pens : ℕ) 
    (purchased_X_cost : cost_X = 4.0) (purchased_Y_cost : cost_Y = 2.8)
    (total_pens_condition : total_pens = 12) (brand_X_pens_condition : brand_X_pens = 8) :
    cost_X * brand_X_pens + cost_Y * (total_pens - brand_X_pens) = 43.20 :=
    sorry

end elena_pen_cost_l545_545833


namespace stacking_comics_order_count_l545_545590

theorem stacking_comics_order_count :
  let batman_comics := 8
  let superman_comics := 7
  let wonder_woman_comics := 6
  let flash_comics := 5
  let total_ways :=
    (Nat.factorial batman_comics) *
    (Nat.factorial superman_comics) *
    (Nat.factorial wonder_woman_comics) *
    (Nat.factorial flash_comics) *
    (Nat.factorial 4)
  total_ways = 421275894176000 :=
by
  let batman_comics := 8
  let superman_comics := 7
  let wonder_woman_comics := 6
  let flash_comics := 5
  let total_ways :=
    (Nat.factorial batman_comics) *
    (Nat.factorial superman_comics) *
    (Nat.factorial wonder_woman_comics) *
    (Nat.factorial flash_comics) *
    (Nat.factorial 4)
  show total_ways = 421275894176000 from sorry

end stacking_comics_order_count_l545_545590


namespace Tn_less_than_half_l545_545566

variable {a : ℕ → ℝ} {S : ℕ → ℝ} {b : ℕ → ℝ} {T : ℕ → ℝ}

-- Definitions and conditions
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
∀ n, a (n + 1) = a n + d

def geometric_sequence (a : ℕ → ℝ) : Prop :=
∀ n, a (n + 1) / a n = a n / a (n - 1)

def first_five_term_condition (a : ℕ → ℝ) (d : ℝ) : Prop :=
a 4 + 4 * d = 10

def bn_definition (b : ℕ → ℝ) (a : ℕ → ℝ) : Prop :=
∀ n, b n = 1 / ((a n - 1) * (a n + 1))

def tn_definition (T : ℕ → ℝ) (b : ℕ → ℝ) : Prop :=
∀ n, T n = ∑ i in range n, b i

-- Theorem statement
theorem Tn_less_than_half
  (d : ℝ)
  (hd : d ≠ 0)
  (ha_arith : arithmetic_sequence a d)
  (ha_geom : geometric_sequence a)
  (h5 : first_five_term_condition a d)
  (hb : bn_definition b a)
  (hT : tn_definition T b) :
  ∀ n, T n < 1 / 2 := sorry

end Tn_less_than_half_l545_545566


namespace count_three_digit_integers_with_4_and_without_6_l545_545924

def is_three_digit_integer (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

def contains_digit (n : ℕ) (d : ℕ) : Prop :=
  nat.digits 10 n |> list.any (λ x => x = d)

def does_not_contain_digit (n : ℕ) (d : ℕ) : Prop :=
  nat.digits 10 n |> list.all (λ x => x ≠ d)

theorem count_three_digit_integers_with_4_and_without_6 : 
  (nat.card {n : ℕ // is_three_digit_integer n ∧ contains_digit n 4 ∧ does_not_contain_digit n 6} = 200) :=
by
  sorry

end count_three_digit_integers_with_4_and_without_6_l545_545924


namespace green_shirt_pairs_l545_545048

theorem green_shirt_pairs (blue_shirts green_shirts total_pairs blue_blue_pairs : ℕ) 
(h1 : blue_shirts = 68) 
(h2 : green_shirts = 82) 
(h3 : total_pairs = 75) 
(h4 : blue_blue_pairs = 30) 
: (green_shirts - (blue_shirts - 2 * blue_blue_pairs)) / 2 = 37 := 
by 
  -- This is where the proof would be written, but we use sorry to skip it.
  sorry

end green_shirt_pairs_l545_545048


namespace right_triangle_area_l545_545678

theorem right_triangle_area (leg1 leg2 hypotenuse : ℕ) (h_leg1 : leg1 = 30)
  (h_hypotenuse : hypotenuse = 34)
  (hypotenuse_sq : hypotenuse * hypotenuse = leg1 * leg1 + leg2 * leg2) :
  (1 / 2 : ℚ) * leg1 * leg2 = 240 := by
  sorry

end right_triangle_area_l545_545678


namespace convex_if_ratio_le_half_l545_545722

noncomputable def curve_is_convex (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_gt : a > b) : Prop :=
  ∀ θ : ℝ, let k := b / a in k ≤ 1/2

theorem convex_if_ratio_le_half (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_gt : a > b) :
  curve_is_convex a b h_pos_a h_pos_b h_gt ↔ b / a ≤ 1/2 :=
sorry

end convex_if_ratio_le_half_l545_545722


namespace chocolate_per_student_l545_545853

theorem chocolate_per_student (b s t n : ℕ)
  (hb : b = 7)
  (hs : s = 8)
  (ht : t = 2)
  (hn : n = 24) :
  (b + b * t) * s / n = 7 :=
by
  rw [hb, hs, ht, hn]
  sorry

end chocolate_per_student_l545_545853


namespace right_triangle_area_l545_545687

theorem right_triangle_area (a b c : ℝ) (h1 : a = 30) (h2 : c = 34) (h3 : a^2 + b^2 = c^2) :
  (1 / 2) * a * b = 240 :=
by
  sorry

end right_triangle_area_l545_545687


namespace prism_volume_l545_545766

theorem prism_volume (r : ℝ) (V_sphere : ℝ) (V_prism : ℝ) :
  V_sphere = (4 / 3) * π * r^3 ∧ r = 3 / 2 ∧ V_prism = (3 * r)^2 * (√3 / 4) * (2 * r) →
  V_prism = 81 * (√3 / 4) :=
by
  -- proof would go here
  sorry

end prism_volume_l545_545766


namespace count_four_digit_numbers_divisible_by_5_end_45_l545_545415

theorem count_four_digit_numbers_divisible_by_5_end_45 : 
  {n : ℕ | 1000 ≤ n ∧ n < 10000 ∧ n % 100 = 45 ∧ n % 5 = 0}.to_finset.card = 90 :=
by
  sorry

end count_four_digit_numbers_divisible_by_5_end_45_l545_545415


namespace range_of_k_l545_545628

noncomputable def f (x k : ℝ) := x^3 - x^2 - x + k

theorem range_of_k (k : ℝ) : (-5 / 27 < k ∧ k < 1) ↔ (∃ x1 x2 x3 : ℝ, f x1 k = 0 ∧ f x2 k = 0 ∧ f x3 k = 0 ∧ x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3) :=
sorry

end range_of_k_l545_545628


namespace intersection_eq_singleton_l545_545086

def M : Set ℝ := {x | log 10 x > 0}
def N : Set ℝ := {2}

theorem intersection_eq_singleton : M ∩ N = {2} :=
by {
  sorry
}

end intersection_eq_singleton_l545_545086


namespace calculate_loss_percentage_l545_545028

noncomputable def cost_of_first_book : ℝ := 350
noncomputable def total_cost : ℝ := 600
noncomputable def gain_percentage : ℝ := 0.19
noncomputable def selling_price : ℝ := 297.5

theorem calculate_loss_percentage :
  let C1 := cost_of_first_book,
      C2 := total_cost - C1,
      S2 := C2 + gain_percentage * C2,
      S1 := selling_price,
      Loss := C1 - S1,
      Loss_percentage := (Loss / C1) * 100 in
  C1 = 350 ∧ total_cost = 600 ∧ gain_percentage = 0.19 ∧ S1 = S2 ∧ S1 = 297.5 ∧ Loss_percentage = 15 := 
by
  sorry

end calculate_loss_percentage_l545_545028


namespace f_periodic_zeros_in_interval_zeros_count_and_sum_l545_545099

-- Define the function and the conditions here
variable {f : ℝ → ℝ} 

axiom f_symmetry1 : ∀ x : ℝ, f (3 + x) = f (3 - x)
axiom f_symmetry2 : ∀ x : ℝ, f (8 + x) = f (8 - x)
axiom f_values : f 1 = 0 ∧ f 5 = 0 ∧ f 7 = 0

-- Problem (1): Prove the function is periodic with period 10
theorem f_periodic : ∃ T : ℝ, T = 10 ∧ ∀ x : ℝ, f (x + T) = f x := 
by sorry

-- Problem (2): Find all zeros of f(x) on [-10, 0]
theorem zeros_in_interval :
  (f (-1) = 0) ∧ (f (-3) = 0) ∧ (f (-5) = 0) ∧ (f (-9) = 0) :=
by sorry

-- Problem (3): Determine the number and sum of zeros in the interval [-2012, 2012]
theorem zeros_count_and_sum :
  (num_zeros : ℕ), (num_zeros = 1610) ∧ (sum_zeros : ℝ), (sum_zeros = 0) := 
by sorry

end f_periodic_zeros_in_interval_zeros_count_and_sum_l545_545099


namespace side_length_of_square_in_right_triangle_l545_545610

-- Define the variables and constants for the problem
variables (DE EF DF t : ℝ)
variable (h_triangle : DE = 5 ∧ EF = 12 ∧ DF = 13)

-- Define the altitude and its value in terms of given triangle sides
def altitude (DE EF DF : ℝ) : ℝ := (DE * EF) / DF

-- Define that t is the side length of the inscribed square
def side_length_square (DE EF DF : ℝ) (k : ℝ) : ℝ :=
  (DF * k) / (DF + k)

-- The proof statement
theorem side_length_of_square_in_right_triangle :
  ∀ (DE EF DF t : ℝ),
  DE = 5 ∧ EF = 12 ∧ DF = 13 →
  let k := altitude DE EF DF in
  t = side_length_square DE EF DF k →
  t = 780 / 169 :=
by
  intros
  sorry

end side_length_of_square_in_right_triangle_l545_545610


namespace solid_volume_correct_l545_545910

noncomputable def volume_of_solid (base_length : ℝ) (height : ℝ) : ℝ :=
  (sqrt 3 / 4) * base_length^2 * height

theorem solid_volume_correct :
  volume_of_solid 1 1 = sqrt 3 / 4 :=
by
  sorry

end solid_volume_correct_l545_545910


namespace hair_cut_length_l545_545733

-- Definitions corresponding to the conditions in the problem
def initial_length : ℕ := 18
def current_length : ℕ := 9

-- Statement to prove
theorem hair_cut_length : initial_length - current_length = 9 :=
by
  sorry

end hair_cut_length_l545_545733


namespace value_of_d_l545_545994

theorem value_of_d :
  ∀ (c d : ℝ), (∀ x, (f : ℝ → ℝ) = λ x, 5 * x + c) →
    (∀ x, (g : ℝ → ℝ) = λ x, c * x + 3) →
    (∀ x, (f (g x)) = 15 * x + d) → d = 18 :=
by
  sorry

end value_of_d_l545_545994


namespace Sam_balloons_correct_l545_545335

def Fred_balloons : Nat := 10
def Dan_balloons : Nat := 16
def Total_balloons : Nat := 72

def Sam_balloons : Nat := Total_balloons - Fred_balloons - Dan_balloons

theorem Sam_balloons_correct : Sam_balloons = 46 := by 
  have H : Sam_balloons = 72 - 10 - 16 := rfl
  simp at H
  exact H

end Sam_balloons_correct_l545_545335


namespace find_interval_for_inequality_l545_545312

open Set

theorem find_interval_for_inequality :
  {x : ℝ | (1 / (x^2 + 2) > 4 / x + 21 / 10)} = Ioo (-2 : ℝ) (0 : ℝ) := 
sorry

end find_interval_for_inequality_l545_545312


namespace general_eqn_of_curve_C_parametric_eqn_of_line_l_sum_of_reciprocals_PA_PB_l545_545968

-- Parametric equations of the curve C
def curve_C (θ : ℝ) : ℝ × ℝ := (2 + sqrt 5 * cos θ, sqrt 5 * sin θ)

-- General equation of curve C
def general_eqn_C (x y : ℝ) : Prop := (x - 2) ^ 2 + y ^ 2 = 5

-- Parametric equation of line l through P(1, -1) with slope 60 degrees
def line_l (t : ℝ) : ℝ × ℝ := (1 + 0.5 * t, -1 + (√3 / 2) * t)

-- The proof problem statements
theorem general_eqn_of_curve_C (θ : ℝ) :
  ∃ x y, curve_C θ = (x, y) ∧ general_eqn_C x y := 
sorry

theorem parametric_eqn_of_line_l (P : ℝ × ℝ) (m : ℝ) (t : ℝ) :
  P = (1, -1) ∧ m = √3 → ∃ x y, line_l t = (x, y) := 
sorry

theorem sum_of_reciprocals_PA_PB (P A B : ℝ × ℝ) :
  P = (1, -1) ∧ ∃ t1 t2, line_l t1 = A ∧ line_l t2 = B →
  abs (1 / dist P A) + abs (1 / dist P B) = sqrt (16 + 2 * sqrt 3) / 3 := 
sorry

end general_eqn_of_curve_C_parametric_eqn_of_line_l_sum_of_reciprocals_PA_PB_l545_545968


namespace quadrilateral_area_two_thirds_l545_545300

def triangle (A B C : ℝ × ℝ) : Prop :=
  ∃ (a b c : ℝ), 
  a > 0 ∧ b > 0 ∧ c > 0 ∧
  a + b > c ∧ a + c > b ∧ b + c > a

theorem quadrilateral_area_two_thirds (A B C : ℝ × ℝ) (h_triangle : triangle A B C)
  (a b c : ℝ) (h_a : a > 0) (h_b : b > 0) (h_c : c > 0)
  (h_ineq1 : a + b > c) (h_ineq2 : a + c > b) (h_ineq3 : b + c > a)
  (h_cond : c ≤ 3 * a) :
  ∃ (P Q : ℝ × ℝ),
  (∃ line : ℝ × ℝ → ℝ × ℝ → Prop, line A P ∧ line Q C) ∧
  let quad_area := (2 / 3) * (area_of_triangle A B C) in
  area_of_quadrilateral A P B Q = quad_area :=
begin
  sorry
end

end quadrilateral_area_two_thirds_l545_545300


namespace closest_fraction_to_sqrt_two_under_100_l545_545719
noncomputable def approx_sqrt_two : ℚ := 99 / 70

theorem closest_fraction_to_sqrt_two_under_100 :
  ∃ (p q : ℕ), p < 100 ∧ q < 100 ∧ (closest_fraction : ℚ) := 
  sqrt_two_distinct_digits (approx_sqrt_two) := 4142 :=
sorry

end closest_fraction_to_sqrt_two_under_100_l545_545719


namespace ratio_cars_to_trucks_l545_545260

theorem ratio_cars_to_trucks (T : ℕ) (C : ℕ) (total_vehicles : ℕ) :
  (T = 60) ∧ (total_vehicles = 2160) ∧ (C = 1920 / 4) → (C : 4 * T = 2 : 1) :=
by
  sorry

end ratio_cars_to_trucks_l545_545260


namespace base_area_of_hemisphere_l545_545165

theorem base_area_of_hemisphere (r : ℝ) (π : ℝ): 
  4 * π * r^2 = 4 * π * r^2 ∧ 3 * π * r^2 = 9 → 
  π * r^2 = 3 := 
by
  intros h
  cases h with sphere_surface_area hemisphere_surface_area
  -- adding some obvious statements
  have hyp3 : 3 * π * r^2 = 9 := hemisphere_surface_area
  have r_sq := (3 : ℝ) / π

  sorry

end base_area_of_hemisphere_l545_545165


namespace right_triangle_area_l545_545692

theorem right_triangle_area (a b c : ℝ) (h1 : a = 30) (h2 : c = 34) (h3 : a^2 + b^2 = c^2) :
  (1 / 2) * a * b = 240 :=
by
  sorry

end right_triangle_area_l545_545692


namespace distinct_ways_to_place_digits_l545_545452

theorem distinct_ways_to_place_digits : 
    ∃ n : ℕ, 
    n = 120 ∧ 
    n = nat.factorial 5 := 
by
  sorry

end distinct_ways_to_place_digits_l545_545452


namespace stars_sum_larger_emilio_sum_l545_545127

theorem stars_sum_larger_emilio_sum :
  let star_numbers := (list.range 30).map (λ n, n + 1)
  let emilio_numbers := star_numbers.map (λ n, nat.digits 10 n |> list.map (λ d, if d = 3 then 2 else d) |> nat.of_digits 10)
  star_numbers.sum - emilio_numbers.sum = 13 :=
by
  sorry

end stars_sum_larger_emilio_sum_l545_545127


namespace segment_lengths_l545_545962

noncomputable def geometric_problem := 
  let r := 7
  let CH := 10
  let a := 2 * Real.sqrt 6
  let AK := r - a
  let KB := r + a
  let CK := CH / 2
  AK * KB = CK * CK

theorem segment_lengths (AK KB : ℝ) (r CH a : ℝ) (h1 : r = 7) (h2 : CH = 10) (h3 : a = 2 * Real.sqrt 6) (h4 : AK = r - a) (h5 : KB = r + a) :
  AK * KB = (CH / 2) * (CH / 2) := by
  rw [h1, h2, h3, h4, h5]
  sorry

end segment_lengths_l545_545962


namespace imaginary_part_of_complex_l545_545631

theorem imaginary_part_of_complex : ∀ z : ℂ, z = i^2 * (1 + i) → z.im = -1 :=
by
  intro z
  intro h
  sorry

end imaginary_part_of_complex_l545_545631


namespace count_four_digit_numbers_divisible_by_5_ending_in_45_l545_545400

theorem count_four_digit_numbers_divisible_by_5_ending_in_45 : 
  ∃ (n : ℕ), n = 90 ∧ ∀ (x : ℕ), (1000 ≤ x ∧ x < 10000) ∧ (x % 100 = 45) → count x = n :=
sorry

end count_four_digit_numbers_divisible_by_5_ending_in_45_l545_545400


namespace no_intersection_abs_functions_l545_545435

open Real

theorem no_intersection_abs_functions : 
  ∀ f g : ℝ → ℝ, 
  (∀ x, f x = |2 * x + 5|) → 
  (∀ x, g x = -|3 * x - 2|) → 
  (∀ y, ∀ x1 x2, f x1 = y ∧ g x2 = y → y = 0 ∧ x1 = -5/2 ∧ x2 = 2/3 → (x1 ≠ x2)) → 
  (∃ x, f x = g x) → 
  false := 
  by
    intro f g hf hg h
    sorry

end no_intersection_abs_functions_l545_545435


namespace perpendicular_lines_b_value_l545_545627

theorem perpendicular_lines_b_value 
  (b : ℝ) 
  (line1 : ∀ x y : ℝ, x + 3 * y + 5 = 0 → True) 
  (line2 : ∀ x y : ℝ, b * x + 3 * y + 5 = 0 → True)
  (perpendicular_condition : (-1 / 3) * (-b / 3) = -1) : 
  b = -9 := 
sorry

end perpendicular_lines_b_value_l545_545627


namespace proper_subset_relationship_l545_545357

def A := {1, 2, 3}
def B := {2, 3}

theorem proper_subset_relationship : B ⊂ A := by
  sorry

end proper_subset_relationship_l545_545357


namespace number_of_strings_of_length_8_l545_545089

def string_conditions (s : list ℕ) : Prop :=
  ∀ i, 0 ≤ i ∧ i ≤ s.length - 4 → (s[i] + s[i+1] + s[i+2] + s[i+3] >= 2)

def S (n : ℕ) : finset (list ℕ) :=
  {s | s.length = n ∧ string_conditions s}

theorem number_of_strings_of_length_8 : (S 8).card = 21 :=
by sorry

end number_of_strings_of_length_8_l545_545089


namespace find_min_y_l545_545098

theorem find_min_y (x y : ℕ) (hx : x = y + 8) 
    (h : Nat.gcd ((x^3 + y^3) / (x + y)) (x * y) = 16) : 
    y = 4 :=
sorry

end find_min_y_l545_545098


namespace neg_p_l545_545147

variable (x : ℝ)

def p : Prop := ∃ x_0 : ℝ, x_0^2 + x_0 + 2 ≤ 0

theorem neg_p : ¬p ↔ ∀ x : ℝ, x^2 + x + 2 > 0 := by
  sorry

end neg_p_l545_545147


namespace magic_shop_change_l545_545494

theorem magic_shop_change :
  (∀ (cloak : Type), ∃ price_gold price_silver1 change_gold1 price_silver2 change_gold2, 
  price_silver1 = 20 ∧ change_gold1 = 4 ∧ 
  price_silver2 = 15 ∧ change_gold2 = 1 ∧ 
  price_gold = 14 ∧ 
  ∀ change_silver, 
    (20 - 4) * change_silver = 15 - 1 → -- Relation derived from the conditions
    (14 - (15 - 1) * change_silver / (20 - 4)) * change_silver = 10) := 
sorry

end magic_shop_change_l545_545494


namespace sum_first_2023_terms_l545_545039

open Nat Real

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^2 - 3*x + 2

-- Define the derivative of the function f'(x)
def f' (x : ℝ) : ℝ := 2*x - 3

-- Define the tangent line - zero sequence
def t_seq (n : ℕ) : ℝ → ℝ
| 0 := 0  -- Initial value can be arbitrary or given, set to 0 for simplicity
| (n+1) := λ t_n, t_n - (f t_n) / (f' t_n)

-- Define the a_n sequence
def a_seq (n : ℕ) : (ℕ → ℝ) → ℝ
| 1 := λ x_seq, 2
| (n+1) := λ x_seq, ln ((x_seq (n+1) - 2) / (x_seq (n+1) - 1))

-- Define the sum of the first n terms of the sequence a_n
def S (n : ℕ) (a_seq : ℕ → (ℕ → ℝ) → ℝ) (x_seq : ℕ → ℝ) : ℝ :=
  (Finset.range n).sum (λ i, a_seq (i + 1) x_seq)

-- Prove that S_2023 = 2^2024 - 2
theorem sum_first_2023_terms (x_seq : ℕ → ℝ) (a_seq : ℕ → (ℕ → ℝ) → ℝ) :
  S 2023 a_seq x_seq = 2^2024 - 2 := sorry

end sum_first_2023_terms_l545_545039


namespace find_BC_l545_545999

-- Definitions for the lengths of the sides AB, AC
def AB : ℝ := 16
def AC : ℝ := 5

-- Defining point P and length AP
def P : Type := sorry
def AP : ℝ := 4

-- Defining triangle ABC such that conditions are given 
-- so that the angle bisectors of ABC and BCA meet at P
noncomputable def triangle_incenter_meeting_condition := sorry

-- Main statement: Proving BC = 14
theorem find_BC : ℝ :=
  ∃ BC : ℝ, BC = 14 :=
begin
  -- Initial assumptions based on conditions
  assume (ABC_triangle : triangle_incenter_meeting_condition),
  -- BC is found to be 14
  use 14,
  sorry
end

end find_BC_l545_545999


namespace total_cost_of_bricking_all_paths_is_15820_l545_545252

noncomputable def parkLength : ℕ := 100
noncomputable def parkBreadth : ℕ := 80
noncomputable def widthPath1 : ℕ := 15
noncomputable def costPath1 : ℕ := 5
noncomputable def widthPath2 : ℕ := 8
noncomputable def costPath2 : ℕ := 7
noncomputable def widthPath3 : ℕ := 5
noncomputable def costPath3 : ℕ := 6

noncomputable def lengthDiagonal : ℕ := Math.sqrt (parkLength ^ 2 + parkBreadth ^ 2)

noncomputable def totalCost : ℕ :=
  (parkLength * widthPath1 * costPath1)
  + (parkBreadth * widthPath2 * costPath2)
  + (lengthDiagonal * widthPath3 * costPath3)

theorem total_cost_of_bricking_all_paths_is_15820 :
  totalCost = 15820 := by
  sorry

end total_cost_of_bricking_all_paths_is_15820_l545_545252


namespace apple_plum_ratio_l545_545280

variables (P : ℕ) (A : ℕ := 180) (F : ℕ := 96) (ratio : ℚ := 3)

theorem apple_plum_ratio :
  let total_fruits := (F * 5) / 2 in
  let plums := total_fruits - A in
  (P = plums) → (ratio = A / P) → ratio = 3 := 
by
  intros total_fruits plums HP HA
  rw [HP, HA]
  sorry

end apple_plum_ratio_l545_545280


namespace sum_of_intercepts_l545_545253

theorem sum_of_intercepts : 
  let line_eq : (ℝ × ℝ) → Prop := λ p, p.2 - 7 = -3 * (p.1 + 2)
  in (∃ x : ℝ, line_eq (x, 0) ∧ 
               ∃ y : ℝ, line_eq (0, y) ∧ 
               (x + y) = 4/3) :=
by 
  let line_eq := λ p : ℝ × ℝ, p.2 - 7 = -3 * (p.1 + 2)
  sorry

end sum_of_intercepts_l545_545253


namespace count_three_digit_numbers_with_4_no_6_l545_545934

def is_digit (n : ℕ) := (n >= 0) ∧ (n < 10)

def three_digit_integer (n : ℕ) := (n >= 100) ∧ (n <= 999)

def contains_digit (d : ℕ) (n : ℕ) : Prop :=
  let hd := n / 100 in
  let td := (n / 10) % 10 in
  let od := n % 10 in
  hd = d ∨ td = d ∨ od = d

def does_not_contain_digit (d : ℕ) (n : ℕ) : Prop :=
  let hd := n / 100 in
  let td := (n / 10) % 10 in
  let od := n % 10 in
  hd ≠ d ∧ td ≠ d ∧ od ≠ d

theorem count_three_digit_numbers_with_4_no_6 : 
  ∃ n, n = 200 ∧
  ∀ x, (three_digit_integer x) → (contains_digit 4 x) → (does_not_contain_digit 6 x) → 
  sorry

end count_three_digit_numbers_with_4_no_6_l545_545934


namespace product_b2_b7_l545_545092

def is_increasing_arithmetic_sequence (bs : ℕ → ℤ) :=
  ∀ n m : ℕ, n < m → bs n < bs m

def arithmetic_sequence (bs : ℕ → ℤ) (d : ℤ) :=
  ∀ n : ℕ, bs (n + 1) - bs n = d

theorem product_b2_b7 (bs : ℕ → ℤ) (d : ℤ) (h_incr : is_increasing_arithmetic_sequence bs)
    (h_arith : arithmetic_sequence bs d)
    (h_prod : bs 4 * bs 5 = 10) :
    bs 2 * bs 7 = -224 ∨ bs 2 * bs 7 = -44 :=
by
  sorry

end product_b2_b7_l545_545092


namespace right_triangle_area_l545_545691

theorem right_triangle_area (a b c : ℝ) (h1 : a = 30) (h2 : c = 34) (h3 : a^2 + b^2 = c^2) :
  (1 / 2) * a * b = 240 :=
by
  sorry

end right_triangle_area_l545_545691


namespace shaded_area_circle_l545_545972

-- Define the geometric setup
def circle_radius := 6
def diameters_perpendicular := true

-- Prove that the shaded area is 36 + 18 * π
theorem shaded_area_circle (r : ℝ) (h : r = circle_radius) 
    (perpendicular_diameters : diameters_perpendicular) :
    let θ := π/2 in
    let sector_area := θ / (2 * π) * π * r^2 in
    let triangle_area := 1/2 * r * r in
    2 * triangle_area + 2 * sector_area = 36 + 18 * π := 
by
  rw [h, circle_radius], -- Simplify radius
  rw [← mul_assoc, (by norm_num : (π / 2) / (2 * π) * π * 6 ^ 2 = 9 * π), (by norm_num : 1 / 2 * 6 * 6 = 18)], -- Simplify sector and triangle areas
  norm_num -- Combine everything

end shaded_area_circle_l545_545972


namespace trigonometric_identity_simplification_l545_545035

open Real

theorem trigonometric_identity_simplification (θ : ℝ) (hθ : 0 < θ ∧ θ < π / 4) :
  (sqrt (1 - 2 * sin (3 * π - θ) * sin (π / 2 + θ)) = cos θ - sin θ) :=
sorry

end trigonometric_identity_simplification_l545_545035


namespace cloak_change_in_silver_l545_545486

theorem cloak_change_in_silver :
  (∀ c : ℤ, (20 = c + 4) → (15 = c + 1)) →
  (5 * g = 3) →
  14 * gold / exchange_rate = 10 := 
sorry

end cloak_change_in_silver_l545_545486


namespace inequality_a_over_b_gt_a_plus_c_over_b_plus_d_gt_c_over_d_l545_545977

theorem inequality_a_over_b_gt_a_plus_c_over_b_plus_d_gt_c_over_d
  (a b c d : ℚ) 
  (h1 : a * d > b * c) 
  (h2 : (a : ℚ) / b > (c : ℚ) / d) : 
  (a / b > (a + c) / (b + d)) ∧ ((a + c) / (b + d) > c / d) :=
by 
  sorry

end inequality_a_over_b_gt_a_plus_c_over_b_plus_d_gt_c_over_d_l545_545977


namespace proposition_1_incorrect_proposition_2_incorrect_proposition_3_correct_l545_545889
noncomputable theory

def sin_eq (A B : ℝ) := real.sin (2 * A) = real.sin (2 * B)
def sin_cos (A B : ℝ) := real.sin A = real.cos B
def cos_product (A B C : ℝ) := real.cos A * real.cos B * real.cos C < 0

def is_isosceles (A B C : ℝ) := 
  ∃ a b c: ℝ, a = b ∨ b = c ∨ a = c ∧ A + B + C = π/2

def is_right_angled (A B C : ℝ) := 
  A + B = π/2 ∨ A + C = π/2 ∨ B + C = π/2

def is_obtuse (A B C : ℝ) := 
  A > π/2 ∨ B > π/2 ∨ C > π/2

theorem proposition_1_incorrect {A B C : ℝ} (h : sin_eq A B) : ¬is_isosceles A B C :=
sorry

theorem proposition_2_incorrect {A B C : ℝ} (h : sin_cos A B) : ¬is_right_angled A B C :=
sorry

theorem proposition_3_correct {A B C : ℝ} (h : cos_product A B C) : is_obtuse A B C :=
sorry


end proposition_1_incorrect_proposition_2_incorrect_proposition_3_correct_l545_545889


namespace students_in_fifth_group_l545_545175

theorem students_in_fifth_group (total_students g1 g2 g3 g4 : ℕ) (h_total : total_students = 40)
    (h_g1 : g1 = 6) (h_g2 : g2 = 9) (h_g3 : g3 = 8) (h_g4 : g4 = 7) : 
    ∃ g5 : ℕ, g5 = total_students - (g1 + g2 + g3 + g4) ∧ g5 = 10 := 
by
  have h_sum : g1 + g2 + g3 + g4 = 6 + 9 + 8 + 7 := by rw [h_g1, h_g2, h_g3, h_g4]
  have h_total_sum : total_students = 40 := h_total
  have h_calc : total_students - (g1 + g2 + g3 + g4) = 40 - (6 + 9 + 8 + 7) := by rw [←h_total_sum, h_sum]
  use 10
  exact ⟨rfl, h_calc⟩

end students_in_fifth_group_l545_545175


namespace ellipse_equation_line_AB_fixed_point_max_area_triangle_ABM_l545_545352

-- Definitions
def is_ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  a > b ∧ b > 0 ∧ (x / a)^2 + (y / b)^2 = 1

def is_on_positive_y_axis (y : ℝ) : Prop :=
  y = sqrt 3

def eccentricity_e (a b : ℝ) (e : ℝ) : Prop :=
  e = 1 / 2 ∧ e = sqrt (1 - (b / a)^2)

def product_of_slopes (MA_slope MB_slope : ℝ) : Prop :=
  MA_slope * MB_slope = 1 / 4

-- Problem 1
theorem ellipse_equation (a b : ℝ) (h1 : is_ellipse a b 1 b)
  (h2 : is_on_positive_y_axis (sqrt 3)) (h3 : eccentricity_e a b (1 / 2)) :
  (a^2 = 4 ∧ b = sqrt 3) :=
sorry

-- Problem 2
theorem line_AB_fixed_point (A B : ℝ × ℝ) (h1 : is_ellipse 2 (sqrt 3) A.1 A.2)
  (h2 : is_ellipse 2 (sqrt 3) B.1 B.2) (h3 : product_of_slopes (A.2 - sqrt 3) (B.2 - sqrt 3)) :
  ∃ N : ℝ × ℝ, N = (0, 2 * sqrt 3) :=
sorry

-- Problem 3
theorem max_area_triangle_ABM (A B : ℝ × ℝ) (h1 : is_ellipse 2 (sqrt 3) A.1 A.2)
  (h2 : is_ellipse 2 (sqrt 3) B.1 B.2) (h3 : product_of_slopes (A.2 - sqrt 3) (B.2 - sqrt 3)) :
  ∃ M : ℝ, M = sqrt 3 → ∃ S : ℝ, S = (sqrt 3) / 2 :=
sorry

end ellipse_equation_line_AB_fixed_point_max_area_triangle_ABM_l545_545352


namespace distinct_ways_to_place_digits_l545_545450

theorem distinct_ways_to_place_digits : 
    ∃ n : ℕ, 
    n = 120 ∧ 
    n = nat.factorial 5 := 
by
  sorry

end distinct_ways_to_place_digits_l545_545450


namespace solution1_solution2_l545_545616

-- Problem: Solving equations and finding their roots

-- Condition 1:
def equation1 (x : Real) : Prop := x^2 - 2 * x = -1

-- Condition 2:
def equation2 (x : Real) : Prop := (x + 3)^2 = 2 * x * (x + 3)

-- Correct answer 1
theorem solution1 : ∀ x : Real, equation1 x → x = 1 := 
by 
  sorry

-- Correct answer 2
theorem solution2 : ∀ x : Real, equation2 x → x = -3 ∨ x = 3 := 
by 
  sorry

end solution1_solution2_l545_545616


namespace equal_red_B_black_C_l545_545043

theorem equal_red_B_black_C (a : ℕ) (h_even : a % 2 = 0) :
  ∃ (x y k j l i : ℕ), x + y = a ∧ y + i + j = a ∧ i + k = y ∧ k + j = x ∧ i = k := 
  sorry

end equal_red_B_black_C_l545_545043


namespace average_test_score_for_class_l545_545041

def student_scores_1 : list ℝ := [90.5, 85.2, 88.7, 92.1, 80.3, 94.8, 89.6, 91.4, 84.9, 87.7]
def student_scores_2 : list ℝ := [85.9, 80.6, 84.1, 87.5, 75.7, 90.2, 85.0, 86.8, 80.3, 83.1, 77.2, 74.3, 80.3, 77.2, 70.4]
def student_scores_3 : list ℝ := [40.7, 62.5, 58.4, 70.2, 72.8, 68.1, 64.3, 66.9, 74.6, 76.2, 60.3, 78.9, 80.7, 82.5, 84.3, 86.1, 88.9, 61.4, 63.2, 65.0, 67.6, 69.4, 71.2, 73.8, 75.9]

theorem average_test_score_for_class : (list.sum (student_scores_1 ++ student_scores_2 ++ student_scores_3) / 50 = 78.058) :=
sorry

end average_test_score_for_class_l545_545041


namespace solve_eq_2_pow_x_plus_3_pow_y_eq_z_sq_l545_545323

theorem solve_eq_2_pow_x_plus_3_pow_y_eq_z_sq (x y z : ℕ) :
  ((x = 3 ∧ y = 0 ∧ z = 3) ∨ (x = 0 ∧ y = 1 ∧ z = 2) ∨ (x = 4 ∧ y = 2 ∧ z = 5)) →
  2^x + 3^y = z^2 :=
by
  sorry

end solve_eq_2_pow_x_plus_3_pow_y_eq_z_sq_l545_545323


namespace right_triangle_area_l545_545683

theorem right_triangle_area (leg1 hypotenuse : ℝ) (h1 : leg1 = 30) (h2 : hypotenuse = 34) (h3 : (leg1 ^ 2 + (sqrt (hypotenuse ^ 2 - leg1 ^ 2)) ^ 2 = hypotenuse ^ 2)) :
  (1 / 2) * leg1 * (sqrt (hypotenuse ^ 2 - leg1 ^ 2)) = 240 :=
by
  sorry

end right_triangle_area_l545_545683


namespace length_A_l545_545083

open Real

noncomputable def point := ℝ × ℝ

def A : point := (0, 10)
def B : point := (0, 15)
def C : point := (3, 9)
def is_on_line_y_eq_x (P : point) : Prop := P.1 = P.2
def length (P Q : point) := sqrt ((Q.1 - P.1)^2 + (Q.2 - P.2)^2)

theorem length_A'B'_is_correct (A' B' : point)
  (hA' : is_on_line_y_eq_x A')
  (hB' : is_on_line_y_eq_x B')
  (hA'_line : ∃ m b, C.2 = m * C.1 + b ∧ A'.2 = m * A'.1 + b ∧ A.2 = m * A.1 + b)
  (hB'_line : ∃ m b, C.2 = m * C.1 + b ∧ B'.2 = m * B'.1 + b ∧ B.2 = m * B.1 + b)
  : length A' B' = 2.5 * sqrt 2 :=
sorry

end length_A_l545_545083


namespace a_horses_is_18_l545_545235

-- Definitions of given conditions
def total_cost : ℕ := 435
def b_share : ℕ := 180
def horses_b : ℕ := 16
def months_b : ℕ := 9
def cost_b : ℕ := horses_b * months_b

def horses_c : ℕ := 18
def months_c : ℕ := 6
def cost_c : ℕ := horses_c * months_c

def total_cost_eq (x : ℕ) : Prop :=
  x * 8 + cost_b + cost_c = total_cost

-- Statement of the proof problem
theorem a_horses_is_18 (x : ℕ) : total_cost_eq x → x = 18 := 
sorry

end a_horses_is_18_l545_545235


namespace chocolate_per_student_l545_545854

theorem chocolate_per_student (b s t n : ℕ)
  (hb : b = 7)
  (hs : s = 8)
  (ht : t = 2)
  (hn : n = 24) :
  (b + b * t) * s / n = 7 :=
by
  rw [hb, hs, ht, hn]
  sorry

end chocolate_per_student_l545_545854


namespace odd_number_expression_parity_l545_545991

theorem odd_number_expression_parity (o n : ℕ) (ho : ∃ k : ℕ, o = 2 * k + 1) :
  (o^2 + n * o) % 2 = 1 ↔ n % 2 = 0 :=
by
  sorry

end odd_number_expression_parity_l545_545991


namespace isosceles_triangle_AM_lt_AB_l545_545116

variable {α : Type*} [OrderedCommGroup α]

structure IsoscelesTriangle (α : Type*) [OrderedCommGroup α] :=
(A B C : α)
(h_eq : AB = AC)

theorem isosceles_triangle_AM_lt_AB {A B C M : Point} (T : IsoscelesTriangle α) (hM : M ≠ B ∧ M ≠ C) :
  dist A M < dist A B :=
sorry

end isosceles_triangle_AM_lt_AB_l545_545116


namespace count_four_digit_numbers_divisible_by_5_end_45_l545_545416

theorem count_four_digit_numbers_divisible_by_5_end_45 : 
  {n : ℕ | 1000 ≤ n ∧ n < 10000 ∧ n % 100 = 45 ∧ n % 5 = 0}.to_finset.card = 90 :=
by
  sorry

end count_four_digit_numbers_divisible_by_5_end_45_l545_545416


namespace smallest_c_for_3_in_range_l545_545827

theorem smallest_c_for_3_in_range : 
  ∀ c : ℝ, (∃ x : ℝ, (x^2 - 6 * x + c) = 3) ↔ (c ≥ 12) :=
by {
  sorry
}

end smallest_c_for_3_in_range_l545_545827


namespace simplify_trig_expr_l545_545245

theorem simplify_trig_expr (x : ℝ) : 
  (1 + sin x) / cos x * (sin (2 * x)) / (2 * cos^2 (π / 4 - x / 2)) = 2 * sin x := 
sorry

end simplify_trig_expr_l545_545245


namespace expected_digits_die_l545_545750

noncomputable def expected_number_of_digits (numbers : List ℕ) : ℚ :=
  let one_digit_numbers := numbers.filter (λ n => n < 10)
  let two_digit_numbers := numbers.filter (λ n => n >= 10)
  let p_one_digit := (one_digit_numbers.length : ℚ) / (numbers.length : ℚ)
  let p_two_digit := (two_digit_numbers.length : ℚ) / (numbers.length : ℚ)
  p_one_digit * 1 + p_two_digit * 2

theorem expected_digits_die :
  expected_number_of_digits [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16] = 1.5833 := 
by
  sorry

end expected_digits_die_l545_545750


namespace length_of_PR_l545_545110

theorem length_of_PR {P Q R : Point} {radius : ℝ}
  (h1 : circle_with_radius P Q radius = 10)
  (h2 : P Q = 12)
  (R_mid : midpoint_minor_arc P Q R)
  : (distance P R = 2 * real.sqrt(10)) :=
sorry

end length_of_PR_l545_545110


namespace right_triangles_with_leg_2012_l545_545526

theorem right_triangles_with_leg_2012 :
  ∀ (a b c : ℕ), a = 2012 ∧ a ^ 2 + b ^ 2 = c ^ 2 → 
  (b = 253005 ∧ c = 253013) ∨ 
  (b = 506016 ∧ c = 506020) ∨ 
  (b = 1012035 ∧ c = 1012037) ∨ 
  (b = 1509 ∧ c = 2515) :=
by
  intros
  sorry

end right_triangles_with_leg_2012_l545_545526


namespace farm_horses_more_than_cows_l545_545596

variable (x : ℤ) -- number of cows initially, must be a positive integer

def initial_horses := 6 * x
def initial_cows := x
def horses_after_transaction := initial_horses - 30
def cows_after_transaction := initial_cows + 30

-- New ratio after transaction
def new_ratio := horses_after_transaction * 1 = 4 * cows_after_transaction

-- Prove that the farm owns 315 more horses than cows after transaction
theorem farm_horses_more_than_cows :
  new_ratio → horses_after_transaction - cows_after_transaction = 315 :=
by
  sorry

end farm_horses_more_than_cows_l545_545596


namespace mike_tires_changed_l545_545588

theorem mike_tires_changed :
  let motorcycles := 12 * 2,
      cars := 10 * 4,
      bicycles := 8 * 2,
      trucks := 5 * 18,
      atvs := 7 * 4,
      dual_axle_trailers := 4 * 8,
      triple_axle_boat_trailers := 3 * 12,
      unicycles := 2 * 1,
      dually_pickup_trucks := 6 * 6
  in motorcycles + cars + bicycles + trucks + atvs + dual_axle_trailers + triple_axle_boat_trailers + unicycles + dually_pickup_trucks = 304 :=
by
  let motorcycles := 12 * 2
  let cars := 10 * 4
  let bicycles := 8 * 2
  let trucks := 5 * 18
  let atvs := 7 * 4
  let dual_axle_trailers := 4 * 8
  let triple_axle_boat_trailers := 3 * 12
  let unicycles := 2 * 1
  let dually_pickup_trucks := 6 * 6
  have h : motorcycles + cars + bicycles + trucks + atvs + dual_axle_trailers + triple_axle_boat_trailers + unicycles + dually_pickup_trucks = 304 := 
    by linarith
  exact h

end mike_tires_changed_l545_545588


namespace sqrt_x_minus_2_range_l545_545974

theorem sqrt_x_minus_2_range (x : ℝ) : x - 2 ≥ 0 → x ≥ 2 :=
by sorry

end sqrt_x_minus_2_range_l545_545974


namespace ray_partitional_count_difference_l545_545572

def is_ray_partitional (n : ℕ) (X : Point) (R : Region) : Prop :=
  ∃ rays : list Ray, 
    length rays = n ∧
    divides_into_equal_triangles rays R

def count_ray_partitional_points (n : ℕ) (R : Region) : ℕ :=
  card {X : Point | X ∈ interior R ∧ is_ray_partitional n X R}

noncomputable def problem_statement (R : Region) : ℕ :=
  let count_150_ray := count_ray_partitional_points 150 R
  let count_90_ray := count_ray_partitional_points 90 R
  count_150_ray - count_90_ray

theorem ray_partitional_count_difference (R : Region) : 
  problem_statement R = 1444 :=
sorry

end ray_partitional_count_difference_l545_545572


namespace puppies_per_cage_calculation_l545_545255

noncomputable def initial_puppies : ℝ := 18.0
noncomputable def additional_puppies : ℝ := 3.0
noncomputable def total_puppies : ℝ := initial_puppies + additional_puppies
noncomputable def total_cages : ℝ := 4.2
noncomputable def puppies_per_cage : ℝ := total_puppies / total_cages

theorem puppies_per_cage_calculation :
  puppies_per_cage = 5.0 :=
by
  sorry

end puppies_per_cage_calculation_l545_545255


namespace only_real_solution_is_x4_minus_6_l545_545720

theorem only_real_solution_is_x4_minus_6 : 
  ∀ x : ℝ, 
    (sqrt (x - 2) + 1 = 0 → false) ∧ 
    (x / (x - 2) = 2 / (x - 2) → false) ∧ 
    ((2 * x^2 + x + 3 = 0) → false) ∧ 
    ((x^4 - 6 = 0) → (x = real.sqrt (real.sqrt 6) ∨ x = -real.sqrt (real.sqrt 6))) 
    :=
by
  sorry

end only_real_solution_is_x4_minus_6_l545_545720


namespace ryan_hours_english_is_6_l545_545835

def hours_chinese : Nat := 2

def hours_english (C : Nat) : Nat := C + 4

theorem ryan_hours_english_is_6 (C : Nat) (hC : C = hours_chinese) : hours_english C = 6 :=
by
  sorry

end ryan_hours_english_is_6_l545_545835


namespace fraction_red_marbles_after_doubling_l545_545473

theorem fraction_red_marbles_after_doubling (x : ℕ) (h : x > 0) :
  let blue_fraction : ℚ := 3 / 5
  let red_fraction := 1 - blue_fraction
  let initial_blue_marbles := blue_fraction * x
  let initial_red_marbles := red_fraction * x
  let new_red_marbles := 2 * initial_red_marbles
  let new_total_marbles := initial_blue_marbles + new_red_marbles
  let new_red_fraction := new_red_marbles / new_total_marbles
  new_red_fraction = 4 / 7 :=
sorry

end fraction_red_marbles_after_doubling_l545_545473


namespace propositions_truth_l545_545890

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a * 2^x - 2) / (2^x + 1)

theorem propositions_truth :
  (¬(∀ a x, (f a x).monotonic_on ℝ) ∧
   ∃ a : ℝ, ∀ x : ℝ, f a (-x) = -f a x ∧
   ¬(∃ a : ℝ, ∀ x : ℝ, f a (-x) = f a x) ∧
   ¬(∑ k in (-2016..2016), f 1 k = -1008)) :=
begin
  sorry
end

end propositions_truth_l545_545890


namespace squarable_numbers_l545_545580

def is_squarable (n : ℕ) : Prop :=
  ∃ (perm : List ℕ), 
    perm ~ List.range n ∧ 
    ∀ i, perm.get i + (i + 1) = (perm.get i + (i + 1)).sqrt ^ 2

theorem squarable_numbers : 
  (¬is_squarable 7) ∧ is_squarable 9 ∧ (¬is_squarable 11) ∧ is_squarable 15 := by
  sorry

end squarable_numbers_l545_545580


namespace perimeter_of_isosceles_triangle_l545_545061

theorem perimeter_of_isosceles_triangle (a b : ℕ) (h_isosceles : (a = 3 ∧ b = 4) ∨ (a = 4 ∧ b = 3)) :
  ∃ p : ℕ, p = 10 ∨ p = 11 :=
by
  sorry

end perimeter_of_isosceles_triangle_l545_545061


namespace clock_angle_at_10_40_l545_545222

theorem clock_angle_at_10_40 :
  let minute_angle := 6 * 40 in
  let hour_angle := (10 * 30) + (40 * 0.5) in
  abs (hour_angle - minute_angle) = 80 :=
by
  let minute_angle := 6 * 40
  let hour_angle := (10 * 30) + (40 * 0.5)
  have h : abs (hour_angle - minute_angle) = abs (320 - 240) := by sorry
  have t : abs (320 - 240) = 80 := by sorry
  exact Eq.trans h t

end clock_angle_at_10_40_l545_545222


namespace polygon_area_correct_l545_545598

def AreaOfPolygon : Real := 37.5

def polygonVertices : List (Real × Real) :=
  [(0, 0), (5, 0), (5, 5), (0, 5), (5, 10), (0, 10), (0, 0)]

theorem polygon_area_correct :
  (∃ (A : Real) (verts : List (Real × Real)),
    verts = polygonVertices ∧ A = AreaOfPolygon ∧ 
    A = 37.5) := by
  sorry

end polygon_area_correct_l545_545598


namespace average_age_of_both_teams_l545_545963

theorem average_age_of_both_teams (n_men : ℕ) (age_men : ℕ) (n_women : ℕ) (age_women : ℕ) :
  n_men = 8 → age_men = 35 → n_women = 6 → age_women = 30 → 
  (8 * 35 + 6 * 30) / (8 + 6) = 32.857 := 
by
  intros h1 h2 h3 h4
  -- Proof is omitted
  sorry

end average_age_of_both_teams_l545_545963


namespace sum_of_squares_of_consecutive_integers_is_perfect_square_l545_545608

theorem sum_of_squares_of_consecutive_integers_is_perfect_square (x : ℤ) :
  ∃ k : ℤ, k ^ 2 = x ^ 2 + (x + 1) ^ 2 + (x ^ 2 * (x + 1) ^ 2) :=
by
  use (x^2 + x + 1)
  sorry

end sum_of_squares_of_consecutive_integers_is_perfect_square_l545_545608


namespace find_angle_C_BO_l545_545057

-- Definitions from the problem
def Triangle (A B C O : Type) : Prop :=
  ∃ (α β γ: ℝ), 
  α =  ∠ B A O ∧ α = ∠  C A O ∧
  β = ∠ C B O ∧ β = ∠ A B O ∧
  γ = ∠ A C O ∧ γ = ∠ B C O ∧
  ∠ A O C = 110

theorem find_angle_C_BO (α β γ: ℝ)
  (h1 : α = 55)
  (h2 : 2 * β + 2 * γ = 70) :
   β = 20 := by
  sorry

end find_angle_C_BO_l545_545057


namespace sum_binom_values_l545_545714

theorem sum_binom_values : 
  (∑ n in (finset.filter (λ n, nat.choose 28 15 + nat.choose 28 n = nat.choose 29 16) (finset.range 29)), n) = 28 :=
by
  sorry

end sum_binom_values_l545_545714


namespace arithmetic_sequence_sum_l545_545539

variable {a : ℕ → ℝ}

-- Definition of an arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n+1) = a n + d

-- Definition of the fourth term condition
def a4_condition (a : ℕ → ℝ) : Prop :=
  a 4 = 2 - a 3

-- Definition of the sum of the first 6 terms
def sum_first_six_terms (a : ℕ → ℝ) : ℝ :=
  a 1 + a 2 + a 3 + a 4 + a 5 + a 6

-- Proof statement
theorem arithmetic_sequence_sum :
  is_arithmetic_sequence a →
  a4_condition a →
  sum_first_six_terms a = 6 :=
by
  sorry

end arithmetic_sequence_sum_l545_545539


namespace total_farm_area_l545_545776

theorem total_farm_area (num_sections : ℕ) (area_per_section : ℕ) (h_num_sections : num_sections = 5) (h_area_per_section : area_per_section = 60) : num_sections * area_per_section = 300 := 
by
  rw [h_num_sections, h_area_per_section]
  exact Nat.mul_eq_mul_left 5 60
  sorry

end total_farm_area_l545_545776


namespace calc_ff_of_one_fourth_l545_545379

noncomputable def f : ℝ → ℝ :=
λ x, if x ≤ 0 then 3^x else Real.log x / Real.log 2

theorem calc_ff_of_one_fourth : f (f (1/4)) = 1/9 :=
by
  sorry

end calc_ff_of_one_fourth_l545_545379


namespace mod_product_2023_2024_2025_2026_l545_545800

theorem mod_product_2023_2024_2025_2026 :
  (2023 * 2024 * 2025 * 2026) % 7 = 0 :=
by
  have h2023 : 2023 % 7 = 6 := by norm_num
  have h2024 : 2024 % 7 = 0 := by norm_num
  have h2025 : 2025 % 7 = 1 := by norm_num
  have h2026 : 2026 % 7 = 2 := by norm_num
  calc
    (2023 * 2024 * 2025 * 2026) % 7
      = ((2023 % 7) * (2024 % 7) * (2025 % 7) * (2026 % 7)) % 7 : by rw [Nat.mul_mod, Nat.mul_mod, Nat.mul_mod, Nat.mul_mod]
  ... = (6 * 0 * 1 * 2) % 7 : by rw [h2023, h2024, h2025, h2026]
  ... = 0 % 7 : by norm_num
  ... = 0 : by norm_num

end mod_product_2023_2024_2025_2026_l545_545800


namespace chord_length_product_constant_l545_545139

noncomputable def hyperbola := 
  {a b : ℝ // a > 0 ∧ b > 0 ∧ ∀ x y, 
   (a = 1 ∧ b = sqrt(2) ∧ x^2 - y^2 / 2 = 1)}

noncomputable def circle := 
  {x y : ℝ // x^2 + y^2 = 2 
   ∧ (x = sqrt(2) ∧ y = 0)}

theorem chord_length (x y : ℝ) (h : circle x y) :
  -- Given the circle condition and the hyperbola spec
  (∃ A : ℝ, A = sqrt(2)) ∧ hyperbola x y → 
  -- Proof specifying the length constraint as a tangent condition
  chord_length = 2 * sqrt(2) → 
  -- Conclusion for proving the equation 
  x^2 - y^2/2 = 1 := 
begin 
  sorry 
end

theorem product_constant (P M N x y : ℝ) (h : circle P) : 
  -- Given any point P on the circle, 
  -- tangent intersections on the hyperbola C leading to perpendicular O, M, N
  OM_PERP_ON →
  -- Proving the constant product condition holds
  (|PM| * |PN| = 2) := 
  sorry
end

end chord_length_product_constant_l545_545139


namespace min_value_fraction_sum_l545_545828

theorem min_value_fraction_sum : 
  ∀ (n : ℕ), n > 0 → (n / 3 + 27 / n) ≥ 6 :=
by
  sorry

end min_value_fraction_sum_l545_545828


namespace even_sum_probability_l545_545820

-- Definition of probabilities for the first wheel
def prob_first_even : ℚ := 2 / 6
def prob_first_odd  : ℚ := 4 / 6

-- Definition of probabilities for the second wheel
def prob_second_even : ℚ := 3 / 8
def prob_second_odd  : ℚ := 5 / 8

-- The expected probability of the sum being even
theorem even_sum_probability : prob_first_even * prob_second_even + prob_first_odd * prob_second_odd = 13 / 24 := by
  sorry

end even_sum_probability_l545_545820


namespace combined_net_profit_l545_545765

def cost_A := 80
def markup_rate_A := 0.20
def discount_rate_A := 0.15

def cost_B := 120
def markup_rate_B := 0.30
def discount_rate_B := 0.10

def cost_C := 200
def markup_rate_C := 0.40
def discount_rate_C := 0.25

def marked_price (cost : ℕ) (markup_rate : ℝ) : ℝ :=
  cost * (1 + markup_rate)

def selling_price (marked_price : ℝ) (discount_rate : ℝ) : ℝ :=
  marked_price * (1 - discount_rate)

def profit (cost : ℕ) (selling_price : ℝ) : ℝ :=
  selling_price - cost

def net_profit (cost_A cost_B cost_C : ℕ) (markup_rate_A markup_rate_B markup_rate_C : ℝ)
               (discount_rate_A discount_rate_B discount_rate_C : ℝ) : ℝ :=
  profit cost_A (selling_price (marked_price cost_A markup_rate_A) discount_rate_A) +
  profit cost_B (selling_price (marked_price cost_B markup_rate_B) discount_rate_B) +
  profit cost_C (selling_price (marked_price cost_C markup_rate_C) discount_rate_C)

theorem combined_net_profit : net_profit cost_A cost_B cost_C 
                                      markup_rate_A markup_rate_B markup_rate_C
                                      discount_rate_A discount_rate_B discount_rate_C = 32 := by
  sorry

end combined_net_profit_l545_545765


namespace point_not_on_line_l545_545356

theorem point_not_on_line (a c : ℝ) (h : a * c > 0) : ¬(0 = 2500 * a + c) := by
  sorry

end point_not_on_line_l545_545356


namespace marble_arrangement_l545_545550

theorem marble_arrangement :
  let blue := 6
  let yellow := 16
  let total := blue + yellow
  let arrangements := Nat.choose total blue
  arrangements % 1000 = 8 :=
by
  let blue := 6
  let yellow := 16
  let total := blue + yellow
  let arrangements := Nat.choose total blue
  have h1 : arrangements = 8008 := by sorry
  have h2 : 8008 % 1000 = 8 := by norm_num
  rw [h1, h2]
  rfl

end marble_arrangement_l545_545550


namespace annulus_area_correct_l545_545781

noncomputable def annulus_area (r s f : ℝ) (h1 : r > s) (h2 : r^2 = s^2 + f^2) : ℝ :=
  π * f^2

theorem annulus_area_correct (r s f : ℝ) (h1 : r > s) (h2 : r^2 = s^2 + f^2) :
  annulus_area r s f h1 h2 = π * f^2 :=
by
  rw [annulus_area, h2]
  sorry

end annulus_area_correct_l545_545781


namespace brittany_second_test_grade_l545_545794

theorem brittany_second_test_grade
  (first_test_grade second_test_grade : ℕ) 
  (average_after_second : ℕ)
  (h1 : first_test_grade = 78)
  (h2 : average_after_second = 81) 
  (h3 : (first_test_grade + second_test_grade) / 2 = average_after_second) :
  second_test_grade = 84 :=
by
  sorry

end brittany_second_test_grade_l545_545794


namespace count_four_digit_numbers_divisible_by_five_ending_45_l545_545431

-- Define the conditions as necessary in Lean
def is_four_digit_number (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

def is_divisible_by_five (n : ℕ) : Prop :=
  n % 5 = 0

def ends_with_45 (n : ℕ) : Prop :=
  n % 100 = 45

-- Statement that there exists 90 such four-digit numbers
theorem count_four_digit_numbers_divisible_by_five_ending_45 : 
  { n : ℕ // is_four_digit_number n ∧ is_divisible_by_five n ∧ ends_with_45 n }.card = 90 :=
sorry

end count_four_digit_numbers_divisible_by_five_ending_45_l545_545431


namespace find_a_and_sin_C_find_cos_2A_plus_pi_over_6_l545_545468

axiom triangle_ABC (a b c : ℝ) (A B C : ℝ) : Prop

variables (a b c : ℝ) (A B C : ℝ)
  (area : ℝ := 3 * Real.sqrt 15)
  (b_minus_c : b - c = 2)
  (cos_A : Real.cos A = -1/4)

# Even though these are variables, they are constant values for the proof context
noncomputable def cos_2A := 2 * (Real.cos A)^2 - 1
noncomputable def sin_2A := 2 * (Real.sin A) * (Real.cos A)
noncomputable def cos_sum := cos_2A A * Real.cos (Real.pi / 6) - sin_2A A * Real.sin (Real.pi / 6)

theorem find_a_and_sin_C (area_constr : b * c = 24) (cosA_constr : Real.cos A = -1/4):
  ∃ (a : ℝ) (sin_C : ℝ), a = 8 ∧ sin_C = Real.sqrt 15 / 8 := by
  sorry

theorem find_cos_2A_plus_pi_over_6 :
  cos_sum = (Real.sqrt 15 - 7 * Real.sqrt 3) / 16 := by
  sorry

end find_a_and_sin_C_find_cos_2A_plus_pi_over_6_l545_545468


namespace range_of_m_l545_545384

-- Definitions of f and g
def f (x : ℝ) : ℝ := x^2 - 4 * x + 3
def g (m x : ℝ) : ℝ := m * x + 3 - 2 * m

-- Statement of the theorem to prove
theorem range_of_m (m : ℝ) :
  (∀ x1 ∈ set.Icc 0 4, ∃ x2 ∈ set.Icc 0 4, f x1 = g m x2) ↔ (m ≤ -2 ∨ m ≥ 2) :=
by
  sorry -- Placeholder for the detailed proof steps

end range_of_m_l545_545384


namespace range_of_a_l545_545011

def A (a : ℝ) : set ℝ := {x | 2 * a ≤ x ∧ x ≤ a + 3}
def B : set ℝ := {x | x < -1 ∨ x > 5}

theorem range_of_a (a : ℝ) : (A a ∩ B = ∅) ↔ (1 / 2 ≤ a ∧ a ≤ 2 ∨ a > 3) :=
by
  sorry

end range_of_a_l545_545011


namespace least_common_denominator_sum_l545_545819

open Nat

theorem least_common_denominator_sum :
  ∃ lcd, lcd = lcm 2 (lcm 3 (lcm 4 (lcm 5 (lcm 6 (lcm 7 (lcm 8 9)))))) ∧ lcd = 2520 :=
by 
  sorry

end least_common_denominator_sum_l545_545819


namespace cloak_change_14_gold_coins_l545_545513

def exchange_rate (silver gold : ℕ) : Prop :=
  ∃ c : ℕ, (20 - 4) * c = silver ∧ (15 - 1) * c = silver

def cloak_purchase (paid_gold received_silver : ℕ) : Prop :=
  let exchange_rate := (5 * 14) / 3 in
  received_silver = 2 * exchange_rate

theorem cloak_change_14_gold_coins :
  exchange_rate 16 3 →
  exchange_rate 14 1 →
  cloak_purchase 14 10 := sorry

end cloak_change_14_gold_coins_l545_545513


namespace OH_squared_correct_l545_545564

noncomputable def OH_squared (O H : Point) (a b c R : ℝ) : ℝ :=
  9 * R^2 - (a^2 + b^2 + c^2)

theorem OH_squared_correct :
  ∀ (O H : Point) (a b c : ℝ) (R : ℝ),
    R = 7 →
    a^2 + b^2 + c^2 = 29 →
    OH_squared O H a b c R = 412 := by
  intros O H a b c R hR habc
  simp [OH_squared, hR, habc]
  sorry

end OH_squared_correct_l545_545564


namespace count_desired_property_l545_545927

-- Define the property that a number is a three-digit positive integer
def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000

-- Define the property that a number contains at least one specific digit
def contains_digit (d : ℕ) (n : ℕ) : Prop :=
  list.any (nat.digits 10 n) (λ m, m = d)

-- Define the property that a number does not contain a specific digit
def does_not_contain_digit (d : ℕ) (n : ℕ) : Prop :=
  ¬ list.any (nat.digits 10 n) (λ m, m = d)

-- Define the overall property for the desired number
def desired_property (n : ℕ) : Prop :=
  is_three_digit n ∧ contains_digit 4 n ∧ does_not_contain_digit 6 n

-- State the theorem to prove the count of numbers with the desired property
theorem count_desired_property : finset.card (finset.filter desired_property (finset.range 1000)) = 200 := by
  sorry

end count_desired_property_l545_545927


namespace mass_percentage_Cl_correct_l545_545840

-- Define the given condition
def mass_percentage_of_Cl := 66.04

-- Statement to prove
theorem mass_percentage_Cl_correct : mass_percentage_of_Cl = 66.04 :=
by
  -- This is where the proof would go, but we use sorry as placeholder.
  sorry

end mass_percentage_Cl_correct_l545_545840


namespace count_three_digit_numbers_with_4_no_6_l545_545933

def is_digit (n : ℕ) := (n >= 0) ∧ (n < 10)

def three_digit_integer (n : ℕ) := (n >= 100) ∧ (n <= 999)

def contains_digit (d : ℕ) (n : ℕ) : Prop :=
  let hd := n / 100 in
  let td := (n / 10) % 10 in
  let od := n % 10 in
  hd = d ∨ td = d ∨ od = d

def does_not_contain_digit (d : ℕ) (n : ℕ) : Prop :=
  let hd := n / 100 in
  let td := (n / 10) % 10 in
  let od := n % 10 in
  hd ≠ d ∧ td ≠ d ∧ od ≠ d

theorem count_three_digit_numbers_with_4_no_6 : 
  ∃ n, n = 200 ∧
  ∀ x, (three_digit_integer x) → (contains_digit 4 x) → (does_not_contain_digit 6 x) → 
  sorry

end count_three_digit_numbers_with_4_no_6_l545_545933


namespace cosine_order_l545_545529

theorem cosine_order (OM ON OB : ℝ) (h1 : 0 < OM) (h2 : OM < ON) (h3 : ON < OB) :
  (real.cos (58 * real.pi / 180)) < (real.cos (41 * real.pi / 180)) ∧ (real.cos (41 * real.pi / 180)) < (real.cos (25 * real.pi / 180)) := 
sorry

end cosine_order_l545_545529


namespace second_team_pies_l545_545071

theorem second_team_pies (total_pies first_team_pies third_team_pies second_team_pies : ℕ)
  (h_total: total_pies = 750)
  (h_first: first_team_pies = 235)
  (h_third: third_team_pies = 240) :
  second_team_pies = 750 - (235 + 240) :=
by {
  rw [h_total, h_first, h_third],
  norm_num,
  sorry
}

end second_team_pies_l545_545071


namespace real_solutions_count_l545_545937

theorem real_solutions_count : ∃ (S : set ℝ), (∀ x ∈ S, (2:ℝ)^(2*x+2) - (2:ℝ)^(x+3) - (2:ℝ)^x + 2 = 0) ∧ (S.card = 2) :=
sorry

end real_solutions_count_l545_545937


namespace small_pos_int_n_l545_545386

theorem small_pos_int_n (a : ℕ → ℕ) (n : ℕ) (a1_val : a 1 = 7)
  (recurrence: ∀ n, a (n + 1) = a n * (a n + 2)) :
  ∃ n : ℕ, a n > 2 ^ 4036 ∧ ∀ m : ℕ, (m < n) → a m ≤ 2 ^ 4036 :=
by
  sorry

end small_pos_int_n_l545_545386


namespace count_four_digit_numbers_divisible_by_five_ending_45_l545_545432

-- Define the conditions as necessary in Lean
def is_four_digit_number (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

def is_divisible_by_five (n : ℕ) : Prop :=
  n % 5 = 0

def ends_with_45 (n : ℕ) : Prop :=
  n % 100 = 45

-- Statement that there exists 90 such four-digit numbers
theorem count_four_digit_numbers_divisible_by_five_ending_45 : 
  { n : ℕ // is_four_digit_number n ∧ is_divisible_by_five n ∧ ends_with_45 n }.card = 90 :=
sorry

end count_four_digit_numbers_divisible_by_five_ending_45_l545_545432


namespace find_point_on_line_l545_545870

open EuclideanGeometry

variables {A B C P : Point} {d : ℝ}

-- Condition: Angle \(ABC\) and line \(P\) are given
def given_angle : Angle := ∠ A B C
def given_line : Line := P

-- Definition of distance
def distance (p1 p2 : Point) : ℝ := EuclideanGeometry.distance p1 p2

-- Find point \( x \) on line \(P\) satisfying the distance condition
def find_point_x (x : Point) : Prop :=
  x ∈ P ∧ distance x A = distance x B + d

-- The main theorem statement: We want to find such an \( x \)
theorem find_point_on_line (x : Point) :
  find_point_x x → ∃ x ∈ P, distance x A = distance x B + d :=
sorry

end find_point_on_line_l545_545870


namespace initial_calculated_average_was_23_l545_545132

theorem initial_calculated_average_was_23 (S : ℕ) (incorrect_sum : ℕ) (n : ℕ)
  (correct_sum : ℕ) (correct_average : ℕ) (wrong_read : ℕ) (correct_read : ℕ) :
  (n = 10) →
  (wrong_read = 26) →
  (correct_read = 36) →
  (correct_average = 24) →
  (correct_sum = n * correct_average) →
  (incorrect_sum = correct_sum - correct_read + wrong_read) →
  S = incorrect_sum →
  S / n = 23 :=
by
  intros
  sorry

end initial_calculated_average_was_23_l545_545132


namespace range_of_a_l545_545345

def p (a : ℝ) : Prop := ∀ k : ℝ, ∃ x y : ℝ, (y = k * x + 1) ∧ (x^2 + (y^2) / a = 1)
def q (a : ℝ) : Prop := ∃ x0 : ℝ, 4^x0 - 2^x0 - a ≤ 0

theorem range_of_a (a : ℝ) : ¬(p a ∧ q a) ∧ (p a ∨ q a) → -1/4 ≤ a ∧ a ≤ 1 :=
sorry

end range_of_a_l545_545345


namespace first_characteristic_number_approx_l545_545666

-- Define the kernel function
def kernel (x t : ℝ) : ℝ :=
  if x ≥ t then t else x

-- Define the interval bounds
def a := (0 : ℝ)
def b := (1 : ℝ)

-- The first characteristic number obtained through the trace method
def firstCharacteristicNumber : ℝ :=
  let A2 := 2 * ∫ x in 0..1, ∫ t in 0..x, t^2
  let A4 := 2 * ∫ x in 0..1, ∫ t in 0..x, (x*t - (x^2*t)/2 - (t^3)/6)^2
  in Real.sqrt (A2 / A4)

theorem first_characteristic_number_approx : abs (firstCharacteristicNumber - 2.48) < 0.01 :=
  sorry

end first_characteristic_number_approx_l545_545666


namespace odd_number_as_diff_of_squares_l545_545919

theorem odd_number_as_diff_of_squares :
    ∀ (x y : ℤ), 63 = x^2 - y^2 ↔ (x = 32 ∧ y = 31) ∨ (x = 12 ∧ y = 9) ∨ (x = 8 ∧ y = 1) := 
by
  sorry

end odd_number_as_diff_of_squares_l545_545919


namespace min_detectors_for_cross_l545_545749
open Nat

theorem min_detectors_for_cross : 
  ∀ (board_size : ℕ) (cross_size : ℕ),
  board_size = 5 → cross_size = 5 →
  ∀ (center_possible_positions : ℕ) (detector_states : ℕ → ℕ),
  center_possible_positions = (board_size - 2) * (board_size - 2) →
  (∀ n, detector_states n = 2^n) →
  ∃ n, 2^n ≥ center_possible_positions ∧ n = 4 :=
by
  intros board_size cross_size h_board h_cross center_possible_positions detector_states h_pos h_states
  use 4
  split
  next h => solve_by_elim
  next => sorry

end min_detectors_for_cross_l545_545749


namespace twin_birthday_difference_l545_545785

theorem twin_birthday_difference : 
  let A := 8 in 
  (A + 1) * (A + 1) - A * A = 17 :=
by
  sorry

end twin_birthday_difference_l545_545785


namespace new_volume_l545_545763

variable (l w h : ℝ)

-- Given conditions
def volume := l * w * h = 5000
def surface_area := l * w + l * h + w * h = 975
def sum_of_edges := l + w + h = 60

-- Statement to prove
theorem new_volume (h1 : volume l w h) (h2 : surface_area l w h) (h3 : sum_of_edges l w h) :
  (l + 2) * (w + 2) * (h + 2) = 7198 :=
by
  sorry

end new_volume_l545_545763


namespace bakery_flour_total_l545_545742

theorem bakery_flour_total :
  (0.2 + 0.1 + 0.15 + 0.05 + 0.1 = 0.6) :=
by {
  sorry
}

end bakery_flour_total_l545_545742


namespace train_speed_km_per_hr_l545_545773

theorem train_speed_km_per_hr 
  (length : ℝ) 
  (time : ℝ) 
  (h_length : length = 150) 
  (h_time : time = 9.99920006399488) : 
  length / time * 3.6 = 54.00287976961843 :=
by
  sorry

end train_speed_km_per_hr_l545_545773


namespace general_term_a_n_sum_of_b_n_l545_545538

noncomputable def a_n (n : ℕ) : ℚ := (n + 1) / 2

def b_n (n : ℕ) : ℚ := 1 / (2 * n * (a_n n))

def S_n (n : ℕ) : ℚ := ∑ i in Finset.range n, b_n (i + 1)

theorem general_term_a_n : ∀ (n : ℕ), (0 < n) → (a_7 = 4) → (a_19 = 2 * a_9) → a_n n = (n + 1) / 2 := by
  sorry

theorem sum_of_b_n (n : ℕ) : (0 < n) → S_n n = n / (n + 1) := by
  sorry

end general_term_a_n_sum_of_b_n_l545_545538


namespace trig_identity_l545_545229

theorem trig_identity (α : ℝ) :
  cos(α) * cos(α) - sin(2 * α) * sin(2 * α) = 
  cos(α) * cos(α) * cos(2 * α) - 2 * sin(α) * sin(α) * cos(α) * cos(α) :=
by
  sorry

end trig_identity_l545_545229


namespace find_a_from_function_property_l545_545649

theorem find_a_from_function_property {a : ℝ} (h : ∀ (x : ℝ), (0 ≤ x → x ≤ 1 → ax ≤ 3) ∧ (0 ≤ x → x ≤ 1 → ax ≥ 3)) :
  a = 3 :=
sorry

end find_a_from_function_property_l545_545649


namespace green_buttons_count_l545_545336

theorem green_buttons_count
  (yellow : ℕ) (black : ℕ) (G : ℕ)
  (initial_total : yellow + black + G)
  (given_to_mary : ℕ)
  (remaining : ℕ)
  (h1 : yellow = 4)
  (h2 : black = 2)
  (h3 : given_to_mary = 4)
  (h4 : remaining = 5)
  (h5 : initial_total - given_to_mary = remaining) :
  G = 3 :=
by {
  sorry
}

end green_buttons_count_l545_545336


namespace count_three_digit_integers_with_4_and_without_6_l545_545923

def is_three_digit_integer (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

def contains_digit (n : ℕ) (d : ℕ) : Prop :=
  nat.digits 10 n |> list.any (λ x => x = d)

def does_not_contain_digit (n : ℕ) (d : ℕ) : Prop :=
  nat.digits 10 n |> list.all (λ x => x ≠ d)

theorem count_three_digit_integers_with_4_and_without_6 : 
  (nat.card {n : ℕ // is_three_digit_integer n ∧ contains_digit n 4 ∧ does_not_contain_digit n 6} = 200) :=
by
  sorry

end count_three_digit_integers_with_4_and_without_6_l545_545923


namespace n_even_number_of_ways_l545_545460

-- Given conditions
def can_be_tiled (n : ℕ) : Prop := ∃ (pieces : list piece), pieces.length = n ∧ tiling pieces

-- Proving part 1: n must be even
theorem n_even (n : ℕ) (h : can_be_tiled n) : Even n :=
sorry

-- Proving part 2: More than 2 * 3^(k-1) ways to tile 5×2k rectangle for k ≥ 3
def f (n : ℕ) : ℕ -- Function representing the number of ways to tile a 5×2n rectangle
| 0 := 1
| 1 := 2
| 2 := 6
| (n + 3) := 2 * f (n + 2) + 2 * f (n + 1) + 4 * f n

theorem number_of_ways (k : ℕ) (hk : k ≥ 3) : f k > 2 * 3^(k-1) :=
sorry

end n_even_number_of_ways_l545_545460


namespace trig_identity_l545_545612

theorem trig_identity 
  (α : ℝ) 
  (h1 : sin α + sin (3 * α) - sin (5 * α) = sin α * (1 - 2 * cos (4 * α))) 
  (h2 : cos α - cos (3 * α) - cos (5 * α) = cos α * (1 - 2 * cos (4 * α))) : 
  (sin α + sin (3 * α) - sin (5 * α)) / (cos α - cos (3 * α) - cos (5 * α)) = tan α := 
by 
  sorry

end trig_identity_l545_545612


namespace magic_shop_change_l545_545492

theorem magic_shop_change :
  (∀ (cloak : Type), ∃ price_gold price_silver1 change_gold1 price_silver2 change_gold2, 
  price_silver1 = 20 ∧ change_gold1 = 4 ∧ 
  price_silver2 = 15 ∧ change_gold2 = 1 ∧ 
  price_gold = 14 ∧ 
  ∀ change_silver, 
    (20 - 4) * change_silver = 15 - 1 → -- Relation derived from the conditions
    (14 - (15 - 1) * change_silver / (20 - 4)) * change_silver = 10) := 
sorry

end magic_shop_change_l545_545492


namespace count_four_digit_numbers_divisible_by_5_and_ending_with_45_l545_545409

def is_four_digit_number (n : ℕ) : Prop := 1000 ≤ n ∧ n < 10000
def ends_with_45 (n : ℕ) : Prop := (n % 100) = 45 
def divisible_by_5 (n : ℕ) : Prop := (n % 5) = 0

theorem count_four_digit_numbers_divisible_by_5_and_ending_with_45 : 
  {n : ℕ | is_four_digit_number n ∧ ends_with_45 n ∧ divisible_by_5 n}.to_finset.card = 90 :=
sorry

end count_four_digit_numbers_divisible_by_5_and_ending_with_45_l545_545409


namespace arithmetic_sequence_general_term_sum_of_b_m_l545_545970

theorem arithmetic_sequence_general_term 
  (a : ℕ → ℕ)
  (h1 : a 3 + a 4 + a 5 = 84)
  (h2 : a 9 = 73) :
  ∀ n, a n = 9 * n - 8 := 
sorry

theorem sum_of_b_m 
  (a b : ℕ → ℕ)
  (h1 : a 3 + a 4 + a 5 = 84)
  (h2 : a 9 = 73)
  (h_gen : ∀ n, a n = 9 * n - 8)
  (b_def : ∀ m n, b m = 9^(2*m - 1) - 9^(m - 1))
  (S_m : ℕ → ℕ) :
  ∀ m, S_m m = (9^(2*m + 1) - 10 * 9^m + 1) / 80 :=
sorry

end arithmetic_sequence_general_term_sum_of_b_m_l545_545970


namespace series_sum_l545_545288

theorem series_sum :
  let s := (∑ k in (Finset.range 51), if even k then - (k : ℤ) else (k : ℤ))
  in s + 101 = 51 :=
by
  -- define the series component
  let s := (∑ k in (Finset.range 51), if even k then - (k : ℤ) else (k : ℤ))
  -- state the goal to prove
  show s + 101 = 51
  -- proof skipped
  -- sorry

end series_sum_l545_545288


namespace math_problem_l545_545274

-- Definitions of propositions
def Prop1 : Prop := 
  ∀ (l : Line) (P : Plane), (∀ (m : Line), m ∈ Plane ∧ l ⊥ m) → l ⊥ P

def Prop2 : Prop := 
  ∀ (l : Line) (P : Plane), (l ‖ P) → (∀ (m : Line), m ⊥ l → m ⊥ P)

def Prop3 : Prop := 
  ∀ (l1 l2 : Line) (P : Plane), (l1 ‖ P) ∧ (l2 ⊥ P) → l1 ⊥ l2

def Prop4 : Prop := 
  ∀ (l1 l2 : Line), (l1 ⊥ l2) → ∃! (P : Plane), (l1 ∈ P) ∧ (P ⊥ l2)

-- Statement of proof problem
theorem math_problem : 
  ¬Prop1 ∧ ¬Prop2 ∧ Prop3 ∧ Prop4 := by
  sorry

end math_problem_l545_545274


namespace train_speed_l545_545233

theorem train_speed 
  (train_length : ℝ) 
  (bridge_length : ℝ) 
  (crossing_time : ℝ) 
  (h_train_length : train_length = 400) 
  (h_bridge_length : bridge_length = 300) 
  (h_crossing_time : crossing_time = 45) : 
  (train_length + bridge_length) / crossing_time = 700 / 45 := 
  by
    rw [h_train_length, h_bridge_length, h_crossing_time]
    sorry

end train_speed_l545_545233


namespace tom_total_calories_l545_545191

-- Define the conditions
def c_weight : ℕ := 1
def c_calories_per_pound : ℕ := 51
def b_weight : ℕ := 2 * c_weight
def b_calories_per_pound : ℕ := c_calories_per_pound / 3

-- Define the total calories
def total_calories : ℕ := (c_weight * c_calories_per_pound) + (b_weight * b_calories_per_pound)

-- Prove the total calories Tom eats
theorem tom_total_calories : total_calories = 85 := by
  sorry

end tom_total_calories_l545_545191


namespace right_triangle_area_l545_545674

theorem right_triangle_area (leg1 leg2 hypotenuse : ℕ) (h_leg1 : leg1 = 30)
  (h_hypotenuse : hypotenuse = 34)
  (hypotenuse_sq : hypotenuse * hypotenuse = leg1 * leg1 + leg2 * leg2) :
  (1 / 2 : ℚ) * leg1 * leg2 = 240 := by
  sorry

end right_triangle_area_l545_545674


namespace number_of_candidates_l545_545131

-- Definitions for the given conditions
def total_marks : ℝ := 2000
def average_marks : ℝ := 40

-- Theorem to prove the number of candidates
theorem number_of_candidates : total_marks / average_marks = 50 := by
  sorry

end number_of_candidates_l545_545131


namespace project_budget_over_under_l545_545747

def projectA_budget : ℝ := 150000
def projectA_alloc_period : ℝ := 2 -- bi-monthly
def projectA_duration : ℝ := 18
def projectA_actual_by_9 : ℝ := 98450

def projectB_budget : ℝ := 120000
def projectB_alloc_period : ℝ := 3 -- quarterly
def projectB_duration : ℝ := 18
def projectB_actual_by_9 : ℝ := 72230

def projectC_budget : ℝ := 80000
def projectC_alloc_period : ℝ := 1 -- monthly
def projectC_duration : ℝ := 18
def projectC_actual_by_9 : ℝ := 43065

def expected_expenditure (budget alloc_period duration time_passed : ℝ) : ℝ :=
  (budget / (duration / alloc_period)) * (time_passed / alloc_period)

def total_difference (expected actual : ℝ) : ℝ := actual - expected

def total_over_budget {A B C : ℝ} : ℝ := A + B + C

theorem project_budget_over_under :
  let A := total_difference (expected_expenditure projectA_budget projectA_alloc_period projectA_duration 9) projectA_actual_by_9
  let B := total_difference (expected_expenditure projectB_budget projectB_alloc_period projectB_duration 9) projectB_actual_by_9
  let C := total_difference (expected_expenditure projectC_budget projectC_alloc_period projectC_duration 9) projectC_actual_by_9
  total_over_budget A B C = 38745 :=
by
  sorry

end project_budget_over_under_l545_545747


namespace miquels_theorem_l545_545866

-- Define the type of points in the plane and the concept of triangles
variables {α : Type} [EuclideanGeometry α]

-- Define points A, B, C forming triangle ABC
variables (A B C D E F : α)

-- Definitions of points D, E, F being on the sides of the triangle ABC
def on_sides (A B C D E F : α) :=
  (∃ (t : ℝ), 0 < t ∧ t < 1 ∧ D = lineThrough A B ∨ E = lineThrough B C ∨ F = lineThrough C A)

-- Define circumcircle and concurrent intersection concept
def circumcircle (X Y Z : α) := {P : α | ∃ r, dist X P = r ∧ dist Y P = r ∧ dist Z P = r}
def concurrent (C₁ C₂ C₃ : set α) := ∃ G, G ∈ C₁ ∧ G ∈ C₂ ∧ G ∈ C₃

theorem miquels_theorem
  (h_on_sides : on_sides A B C D E F) :
  concurrent (circumcircle A E D) (circumcircle B E F) (circumcircle C D F) :=
begin
  sorry
end

end miquels_theorem_l545_545866


namespace find_a_l545_545876

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
if h : x < 0 then a ^ x - 1 else 2 * x ^ 2

theorem find_a (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (h3 : ∀ m n : ℝ, f a m ≤ f a n ↔ m ≤ n)
  (h4 : f a a = 5 * a - 2) : a = 2 :=
sorry

end find_a_l545_545876


namespace unique_value_of_a_l545_545239

noncomputable def is_perfect_cube (x : ℕ) : Prop :=
∃ k : ℕ, k^3 = x

theorem unique_value_of_a (a : ℕ) (h₀ : a ∈ Set.Icc 1 Nat.infinity)
  (h₁ : ∀ n : ℕ, n ∈ Set.Icc 1 Nat.infinity → is_perfect_cube (4 * (a^n + 1))) : a = 1 := 
sorry

end unique_value_of_a_l545_545239


namespace quadratic_common_root_l545_545983

theorem quadratic_common_root :
  ∃ n : ℕ, n = 30 ∧
  (∃ (pairs : Finset (ℕ × ℕ)),
    (∀ p ∈ pairs, (1 ≤ p.1 ∧ p.1 < 1000) ∧ (1 ≤ p.2 ∧ p.2 < 1000)) ∧
    pairs.card = 30 ∧
    ∀ (a b : ℕ), 
    (a, b) ∈ pairs →
      ∃ (m : ℤ),
        m^2 + 2 * m + 1 + a * m + a + b = 0 ∧
        m^2 + a * m + b + 1 = 0 ∧ 
        (m + (2 * (m = -a / 2))) →
        m ∈ ℤ) :=
begin
  sorry
end

end quadratic_common_root_l545_545983


namespace problem_1_problem_2_l545_545240

open Real

theorem problem_1
  (a b m n : ℝ)
  (ha : a > 0)
  (hb : b > 0)
  (hm : m > 0)
  (hn : n > 0) :
  (m ^ 2 / a + n ^ 2 / b) ≥ ((m + n) ^ 2 / (a + b)) :=
sorry

theorem problem_2
  (x : ℝ)
  (hx1 : 0 < x)
  (hx2 : x < 1 / 2) :
  (2 / x + 9 / (1 - 2 * x)) ≥ 25 ∧ (2 / x + 9 / (1 - 2 * x)) = 25 ↔ x = 1 / 5 :=
sorry

end problem_1_problem_2_l545_545240


namespace compute_modulo_l545_545813

theorem compute_modulo :
  (2023 * 2024 * 2025 * 2026) % 7 = 0 := by
  sorry

end compute_modulo_l545_545813


namespace gcd_consecutive_b_l545_545008

theorem gcd_consecutive_b (n : ℕ) : 
  ∃ N, ∀ n ≥ N, gcd (2^n * n! + n) (2^(n+1) * (n+1)! + (n+1)) = 1 :=
begin
  sorry,
end

end gcd_consecutive_b_l545_545008


namespace positive_difference_150th_155th_term_l545_545223

def arithmetic_sequence_first_term : ℕ := 4
def arithmetic_sequence_common_difference : ℕ := 6

noncomputable def term (n : ℕ) : ℕ := 
  arithmetic_sequence_first_term + (n - 1) * arithmetic_sequence_common_difference

theorem positive_difference_150th_155th_term :
  abs (term 155 - term 150) = 30 := by
  sorry

end positive_difference_150th_155th_term_l545_545223


namespace equalized_distance_l545_545027

noncomputable def wall_width : ℝ := 320 -- wall width in centimeters
noncomputable def poster_count : ℕ := 6 -- number of posters
noncomputable def poster_width : ℝ := 30 -- width of each poster in centimeters
noncomputable def equal_distance : ℝ := 20 -- equal distance in centimeters to be proven

theorem equalized_distance :
  let total_posters_width := poster_count * poster_width
  let remaining_space := wall_width - total_posters_width
  let number_of_spaces := poster_count + 1
  remaining_space / number_of_spaces = equal_distance :=
by {
  sorry
}

end equalized_distance_l545_545027


namespace number_of_terms_in_product_l545_545938

theorem number_of_terms_in_product 
  (a b c d e f g h i : ℕ) :
  (a + b + c + d) * (e + f + g + h + i) = 20 :=
sorry

end number_of_terms_in_product_l545_545938


namespace number_of_real_roots_l545_545639

noncomputable def f (x : ℝ) : ℝ := x^3 - 6*x^2 + 9*x - 4

theorem number_of_real_roots : ∃! x : finset ℝ, (∀ y ∈ x, f y = 0) ∧ finset.card x = 2 := sorry

end number_of_real_roots_l545_545639


namespace conjugate_complex_quadrant_l545_545859

theorem conjugate_complex_quadrant (i : ℂ) (h_i : i = complex.I) : 
  let z := (1 : ℂ) / (1 - i)
  let conj_z := complex.conj z
  conj_z.re > 0 ∧ conj_z.im < 0 :=
by 
  sorry

end conjugate_complex_quadrant_l545_545859


namespace break_room_capacity_l545_545830

theorem break_room_capacity :
  let people_per_table := 8
  let number_of_tables := 4
  people_per_table * number_of_tables = 32 :=
by
  let people_per_table := 8
  let number_of_tables := 4
  have h : people_per_table * number_of_tables = 32 := by sorry
  exact h

end break_room_capacity_l545_545830


namespace count_four_digit_numbers_divisible_by_5_and_ending_with_45_l545_545406

def is_four_digit_number (n : ℕ) : Prop := 1000 ≤ n ∧ n < 10000
def ends_with_45 (n : ℕ) : Prop := (n % 100) = 45 
def divisible_by_5 (n : ℕ) : Prop := (n % 5) = 0

theorem count_four_digit_numbers_divisible_by_5_and_ending_with_45 : 
  {n : ℕ | is_four_digit_number n ∧ ends_with_45 n ∧ divisible_by_5 n}.to_finset.card = 90 :=
sorry

end count_four_digit_numbers_divisible_by_5_and_ending_with_45_l545_545406


namespace sum_and_product_of_roots_l545_545304

-- Coefficients of the quadratic equation
def a : ℝ := 2
def b : ℝ := -10
def c : ℝ := 12

-- Sum of the roots
def sum_of_roots (a b : ℝ) : ℝ := -b / a

-- Product of the roots
def product_of_roots (a c : ℝ) : ℝ := c / a

theorem sum_and_product_of_roots :
  sum_of_roots a b = 5 ∧ product_of_roots a c = 6 :=
by
  sorry

end sum_and_product_of_roots_l545_545304


namespace central_angle_is_one_l545_545882

-- Definitions of the given conditions
def arc_length : ℝ := 6
def sector_area : ℝ := 18

-- The central angle α to be proven
def central_angle (l : ℝ) (A : ℝ) : ℝ := 2 * A / l

theorem central_angle_is_one :
  central_angle 6 18 = 1 := 
by 
  sorry

end central_angle_is_one_l545_545882


namespace valid_paths_count_l545_545295

def grid_paths (rows cols : ℕ) (forbidden : list (ℕ × ℕ) × (ℕ × ℕ)) : ℕ := 
sorry

/- Define the grid dimensions and forbidden segments -/
def rows := 5
def cols := 10
def forbidden_segments := [((4, 4), (4, 3)), ((7, 2), (7, 1))]

/- State the problem -/
theorem valid_paths_count : grid_paths rows cols forbidden_segments = 1793 := 
sorry

end valid_paths_count_l545_545295


namespace right_triangle_area_l545_545676

theorem right_triangle_area (leg1 leg2 hypotenuse : ℕ) (h_leg1 : leg1 = 30)
  (h_hypotenuse : hypotenuse = 34)
  (hypotenuse_sq : hypotenuse * hypotenuse = leg1 * leg1 + leg2 * leg2) :
  (1 / 2 : ℚ) * leg1 * leg2 = 240 := by
  sorry

end right_triangle_area_l545_545676


namespace store_profit_is_32_percent_l545_545723

variables (C : ℝ) (mark_up1 mark_up2 discount : ℝ)
variables (FirstMarkupPrice SecondMarkupPrice SellingPrice Profit : ℝ)

-- prime conditions
def firstMarkupPrice : FirstMarkupPrice = C * (1 + mark_up1 / 100) := by
  have h := C * (1 + mark_up1 / 100)
  exact h

def secondMarkupPrice : SecondMarkupPrice = FirstMarkupPrice * (1 + mark_up2 / 100) := by
  have h := FirstMarkupPrice * (1 + mark_up2 / 100)
  exact h

def sellingPrice : SellingPrice = SecondMarkupPrice * (1 - discount / 100) := by
  have h := SecondMarkupPrice * (1 - discount / 100)
  exact h

def profit : Profit = SellingPrice - C := by
  have h := SellingPrice - C
  exact h

-- hypothesis conditions
axiom mark_up1_is_20 : mark_up1 = 20
axiom mark_up2_is_25 : mark_up2 = 25
axiom discount_is_12 : discount = 12

-- expected proof statement
theorem store_profit_is_32_percent : Profit = C * 0.32 := by
  sorry  -- the proof is not required according to instructions

end store_profit_is_32_percent_l545_545723


namespace max_possible_sum_l545_545540

theorem max_possible_sum (k n : ℕ) (h : k ≤ n) :
  let table := list.finRange (2 * k + 1) × list.finRange (2 * n + 1)
  let numbers := {1, 2, 3}
  ∃ f : table → ℕ, (∀ t, t ∈ table → f t ∈ numbers) ∧ 
   (∀ x y z w : table, (f x, f y, f z, f w) = (1, 2, 3, f w) ∨
   (f x, f y, f z, f w) = (1, 3, 2, f w) ∨
   (f x, f y, f z, f w) = (2, 1, 3, f w) ∨
   (f x, f y, f z, f w) = (2, 3, 1, f w) ∨
   (f x, f y, f z, f w) = (3, 1, 2, f w) ∨
   (f x, f y, f z, f w) = (3, 2, 1, f w)) →
  ∑ (i : table), f i = 9 * k * n + 6 * n + 5 * k + 3 := sorry

end max_possible_sum_l545_545540


namespace hemisphere_base_area_l545_545172

theorem hemisphere_base_area (r : ℝ) (π : ℝ) (h₁ : π > 0) 
  (sphere_surface_area : 4 * π * r^2) 
  (hemisphere_surface_area : 3 * π * r^2 = 9) : 
  π * r^2 = 3 := 
by 
  sorry

end hemisphere_base_area_l545_545172


namespace hyperbola_eccentricity_range_l545_545901

-- Lean 4 statement for the given problem.
theorem hyperbola_eccentricity_range {a b : ℝ} (ha : a > 0) (hb : b > 0)
  (h : ∀ (x y : ℝ), y = x * Real.sqrt 3 → y^2 / b^2 - x^2 / a^2 = 1 ∨ ∃ (z : ℝ), y = x * Real.sqrt 3 ∧ z^2 / b^2 - x^2 / a^2 = 1) :
  1 < Real.sqrt (a^2 + b^2) / a ∧ Real.sqrt (a^2 + b^2) / a < 2 :=
by
  sorry

end hyperbola_eccentricity_range_l545_545901


namespace find_m_l545_545333

-- Define the set U
def U : Finset ℕ := Finset.range 18

-- Define the sum function s(T) where T is a subset of U
def s (T : Finset ℕ) : ℕ := Finset.sum T id

-- Define the probability that s(T) is divisible by 3
def P₃ (U : Finset ℕ) : ℚ :=
  let subsets := Finset.powerset U
  let divisibleBy3 := subsets.filter (λ T, s T % 3 = 0)
  (Finset.card divisibleBy3 : ℚ) / (Finset.card subsets : ℚ)

-- The main theorem stating the probability and extracting m
theorem find_m : let p := P₃ U in p = 683 / 2048 := sorry

end find_m_l545_545333


namespace count_four_digit_numbers_divisible_by_5_ending_in_45_l545_545424

theorem count_four_digit_numbers_divisible_by_5_ending_in_45 : 
  ∃ n : ℕ, (∀ x : ℕ, x ≥ 1000 ∧ x < 10000 ∧ x % 100 = 45 → (x % 5 = 0 ∧ x % 100 = 45)) 
  ∧ (n = 90) :=
by
  sorry

end count_four_digit_numbers_divisible_by_5_ending_in_45_l545_545424


namespace solve_system_of_equations_l545_545553

def proof_problem (a b c : ℚ) : Prop :=
  ((a - b = 2) ∧ (c = -5) ∧ (2 * a - 6 * b = 2)) → 
  (a = 5 / 2 ∧ b = 1 / 2 ∧ c = -5)

theorem solve_system_of_equations (a b c : ℚ) :
  proof_problem a b c :=
  by
    sorry

end solve_system_of_equations_l545_545553


namespace remainder_1234_5678_9012_div_5_l545_545224

theorem remainder_1234_5678_9012_div_5 : (1234 * 5678 * 9012) % 5 = 4 := by
  sorry

end remainder_1234_5678_9012_div_5_l545_545224


namespace sum_of_negative_a_three_solutions_l545_545311

theorem sum_of_negative_a_three_solutions :
  ∀ (a : ℝ), 
    (∀ (x : ℝ), 
      (a < 0) → 
      (3 + (Real.tan x)^2 ≠ 0) →
      ((8 * Real.pi * a - Real.arcsin (Real.sin x) + 3 * Real.arccos (Real.cos x) - a * x) / 
       (3 + (Real.tan x)^2) = 0) →
      (∃! (x1 x2 x3: ℝ), 
        (8 * Real.pi * a - Real.arcsin (Real.sin x1) + 3 * Real.arccos (Real.cos x1) - a * x1 = 0) ∧
        (8 * Real.pi * a - Real.arcsin (Real.sin x2) + 3 * Real.arccos (Real.cos x2) - a * x2 = 0) ∧
        (8 * Real.pi * a - Real.arcsin (Real.sin x3) + 3 * Real.arccos (Real.cos x3) - a * x3 = 0) ∧
        (x1 ≠ x2 ∧ x1 ≠ x3 ∧ x2 ≠ x3)
      )
    ) →
  (∑ a in {-1, -2/3, -4/5} : set ℝ, a) = -2.47 := 
sorry

end sum_of_negative_a_three_solutions_l545_545311


namespace integer_regular_polygon_vertices_l545_545849

theorem integer_regular_polygon_vertices {n : ℕ} (h : n ≥ 3) : 
  (∃ (p : ℕ) (vertices : Fin p → ℤ × ℤ), 
    ∀ i, dist (vertices i) (vertices (i + 1) % p) = dist (vertices 0) (vertices 1)) 
    ↔ n = 4 :=
by
  sorry

end integer_regular_polygon_vertices_l545_545849


namespace slope_undefined_iff_vertical_l545_545874

theorem slope_undefined_iff_vertical (m : ℝ) :
  let M := (2 * m + 3, m)
  let N := (m - 2, 1)
  (2 * m + 3 - (m - 2) = 0 ∧ m - 1 ≠ 0) ↔ m = -5 :=
by
  sorry

end slope_undefined_iff_vertical_l545_545874


namespace right_triangle_area_l545_545679

theorem right_triangle_area (leg1 hypotenuse : ℝ) (h1 : leg1 = 30) (h2 : hypotenuse = 34) (h3 : (leg1 ^ 2 + (sqrt (hypotenuse ^ 2 - leg1 ^ 2)) ^ 2 = hypotenuse ^ 2)) :
  (1 / 2) * leg1 * (sqrt (hypotenuse ^ 2 - leg1 ^ 2)) = 240 :=
by
  sorry

end right_triangle_area_l545_545679


namespace count_four_digit_numbers_divisible_by_5_ending_in_45_l545_545421

theorem count_four_digit_numbers_divisible_by_5_ending_in_45 : 
  ∃ n : ℕ, (∀ x : ℕ, x ≥ 1000 ∧ x < 10000 ∧ x % 100 = 45 → (x % 5 = 0 ∧ x % 100 = 45)) 
  ∧ (n = 90) :=
by
  sorry

end count_four_digit_numbers_divisible_by_5_ending_in_45_l545_545421


namespace count_valid_numbers_l545_545391

-- Define what it means to be a four-digit number that ends in 45
def is_valid_number (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999 ∧ n % 100 = 45 ∧ n % 5 = 0

-- Define the set of valid two-digit prefixes
def valid_prefixes : set ℕ := {ab | 10 ≤ ab ∧ ab ≤ 99}

-- Define the set of four-digit numbers that end in 45 and are divisible by 5
def valid_numbers : set ℕ := {n | ∃ ab : ℕ, ab ∈ valid_prefixes ∧ n = ab * 100 + 45}

-- State the theorem
theorem count_valid_numbers : (finset.card (finset.filter is_valid_number (finset.range 10000)) = 90) :=
sorry

end count_valid_numbers_l545_545391


namespace count_three_digit_numbers_with_4_no_6_l545_545932

def is_digit (n : ℕ) := (n >= 0) ∧ (n < 10)

def three_digit_integer (n : ℕ) := (n >= 100) ∧ (n <= 999)

def contains_digit (d : ℕ) (n : ℕ) : Prop :=
  let hd := n / 100 in
  let td := (n / 10) % 10 in
  let od := n % 10 in
  hd = d ∨ td = d ∨ od = d

def does_not_contain_digit (d : ℕ) (n : ℕ) : Prop :=
  let hd := n / 100 in
  let td := (n / 10) % 10 in
  let od := n % 10 in
  hd ≠ d ∧ td ≠ d ∧ od ≠ d

theorem count_three_digit_numbers_with_4_no_6 : 
  ∃ n, n = 200 ∧
  ∀ x, (three_digit_integer x) → (contains_digit 4 x) → (does_not_contain_digit 6 x) → 
  sorry

end count_three_digit_numbers_with_4_no_6_l545_545932


namespace count_four_digit_numbers_divisible_by_5_end_45_l545_545418

theorem count_four_digit_numbers_divisible_by_5_end_45 : 
  {n : ℕ | 1000 ≤ n ∧ n < 10000 ∧ n % 100 = 45 ∧ n % 5 = 0}.to_finset.card = 90 :=
by
  sorry

end count_four_digit_numbers_divisible_by_5_end_45_l545_545418


namespace smallest_rel_prime_gt_1_2310_l545_545708

theorem smallest_rel_prime_gt_1_2310 :
  ∃ n > 1, (∀ m > 1, m < n → ¬ (nat.coprime m 2310)) ∧ nat.coprime n 2310 :=
  exists.intro 13 (and.intro (exists.intro (13:ℤ) 
  (and.intro (nat.succ_lt_succ (nat.succ_lt_succ (zero_lt_one)))
  (forall.intro (λ m (hm : m > 1) (hmn : m < 13), (not_intro 
    (λ (h : nat.gcd m 2310 = 1),
      have h2 : m ≥ 2 → m ≤ 3 → nat.gcd m 2310 ≠ 1, from sorry,
      have h5 : m ≤ 11, from sorry, 
      show false, from this (le_total m 11) h5))))))
  (nat.gcd_eq_one_of_coprime .
      (nat.prime_coprime (nat.prime_factorisation_complete 2) nat.factorisation_complete
        (or.elim (not_or_distrib.mp (not_or_distrib.mp (not_or_distrib.mp
      (not_or_distrib.mp (not_and_distrib.mpr (or.inr (not_and_distrib.mpr (or.inr (not_and_distrib.mpr (or.inr (not_and_distrib.mpr (or.inr (not_and_distrib.mpr (or.inl (nat.gcd_comm 13 2310)))))))))))))))) ⊥))))
⟩ sorry

end smallest_rel_prime_gt_1_2310_l545_545708


namespace find_unknown_rate_l545_545232

theorem find_unknown_rate
  (cost1 cost2 num1 num2 num3 total_avg : ℝ)
  (num1_eq : num1 = 3)
  (cost1_eq : cost1 = 100)
  (num2_eq : num2 = 6)
  (cost2_eq : cost2 = 150)
  (num3_eq : num3 = 2)
  (total_avg_eq : total_avg = 150) :
  let total_blanks := num1 + num2 + num3,
      total_cost := total_avg * total_blanks,
      known_cost := (num1 * cost1) + (num2 * cost2),
      unknown_cost := total_cost - known_cost,
      unknown_rate_per_blank := unknown_cost / num3
  in
      unknown_rate_per_blank = 225 :=
by
  sorry

end find_unknown_rate_l545_545232


namespace right_triangle_area_l545_545680

theorem right_triangle_area (leg1 hypotenuse : ℝ) (h1 : leg1 = 30) (h2 : hypotenuse = 34) (h3 : (leg1 ^ 2 + (sqrt (hypotenuse ^ 2 - leg1 ^ 2)) ^ 2 = hypotenuse ^ 2)) :
  (1 / 2) * leg1 * (sqrt (hypotenuse ^ 2 - leg1 ^ 2)) = 240 :=
by
  sorry

end right_triangle_area_l545_545680


namespace range_of_a_l545_545463

noncomputable def f (a x : ℝ) : ℝ := Real.log (x^2 - a * x - 3)

def monotonic_increasing (a : ℝ) : Prop :=
  ∀ x > 1, 2 * x - a > 0

def positive_argument (a : ℝ) : Prop :=
  ∀ x > 1, x^2 - a * x - 3 > 0

theorem range_of_a :
  {a : ℝ | monotonic_increasing a ∧ positive_argument a} = {a : ℝ | a ≤ -2} :=
sorry

end range_of_a_l545_545463


namespace count_four_digit_numbers_divisible_by_5_and_ending_with_45_l545_545410

def is_four_digit_number (n : ℕ) : Prop := 1000 ≤ n ∧ n < 10000
def ends_with_45 (n : ℕ) : Prop := (n % 100) = 45 
def divisible_by_5 (n : ℕ) : Prop := (n % 5) = 0

theorem count_four_digit_numbers_divisible_by_5_and_ending_with_45 : 
  {n : ℕ | is_four_digit_number n ∧ ends_with_45 n ∧ divisible_by_5 n}.to_finset.card = 90 :=
sorry

end count_four_digit_numbers_divisible_by_5_and_ending_with_45_l545_545410


namespace find_large_number_l545_545236

theorem find_large_number (L S : ℕ) 
  (h1 : L - S = 1335) 
  (h2 : L = 6 * S + 15) : 
  L = 1599 := 
by 
  -- proof omitted
  sorry

end find_large_number_l545_545236


namespace number_of_collinear_points_equals_49_l545_545640

noncomputable def count_collinear_points : ℕ :=
  49

-- Define the set of points
structure Cube :=
(vertices : Fin 8) 
(edge_midpoints : Fin 12) 
(face_centers : Fin 6) 
(cube_center : Unit)

def collinear_points_condition (c : Cube) : Prop :=
  -- Assume a predicate that describes whether three points are collinear
  sorry

-- Main theorem
theorem number_of_collinear_points_equals_49 {c : Cube} : 
  ∃ (sets_of_three_points : Finset (Fin 27)),
    (∀ s ∈ sets_of_three_points, collinear_points_condition s) ∧ sets_of_three_points.card = 49 :=
begin
  sorry
end

end number_of_collinear_points_equals_49_l545_545640


namespace area_of_right_triangle_l545_545706

-- Given definitions
def leg_a : ℝ := 30
def hypotenuse_c : ℝ := 34

-- The theorem statement
theorem area_of_right_triangle : 
  ∀ (b : ℝ), b = real.sqrt (hypotenuse_c^2 - leg_a^2) → 
  let area := 1 / 2 * leg_a * b in
  area = 240 := 
by
  intro b
  intro h
  let area := 1 / 2 * leg_a * b
  sorry

end area_of_right_triangle_l545_545706


namespace alcohol_percentage_in_mixed_solution_l545_545247

theorem alcohol_percentage_in_mixed_solution :
  let vol1 := 8
  let perc1 := 0.25
  let vol2 := 2
  let perc2 := 0.12
  let total_alcohol := (vol1 * perc1) + (vol2 * perc2)
  let total_volume := vol1 + vol2
  (total_alcohol / total_volume) * 100 = 22.4 := by
  sorry

end alcohol_percentage_in_mixed_solution_l545_545247


namespace base_area_of_hemisphere_l545_545166

theorem base_area_of_hemisphere (r : ℝ) (π : ℝ): 
  4 * π * r^2 = 4 * π * r^2 ∧ 3 * π * r^2 = 9 → 
  π * r^2 = 3 := 
by
  intros h
  cases h with sphere_surface_area hemisphere_surface_area
  -- adding some obvious statements
  have hyp3 : 3 * π * r^2 = 9 := hemisphere_surface_area
  have r_sq := (3 : ℝ) / π

  sorry

end base_area_of_hemisphere_l545_545166


namespace count_three_digit_integers_with_4_and_without_6_l545_545926

def is_three_digit_integer (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

def contains_digit (n : ℕ) (d : ℕ) : Prop :=
  nat.digits 10 n |> list.any (λ x => x = d)

def does_not_contain_digit (n : ℕ) (d : ℕ) : Prop :=
  nat.digits 10 n |> list.all (λ x => x ≠ d)

theorem count_three_digit_integers_with_4_and_without_6 : 
  (nat.card {n : ℕ // is_three_digit_integer n ∧ contains_digit n 4 ∧ does_not_contain_digit n 6} = 200) :=
by
  sorry

end count_three_digit_integers_with_4_and_without_6_l545_545926


namespace cartesian_equation_C1_cartesian_equation_C2_max_PM_plus_PN_l545_545385

noncomputable def curveC1 (θ : ℝ) : ℝ × ℝ := (2 * Real.cos θ, Real.sqrt 3 * Real.sin θ)

def curveC2 (α : ℝ) : ℝ × ℝ := (2 * Real.cos α, 2 * Real.sin α)

theorem cartesian_equation_C1 : ∀ (x y θ : ℝ), 
  (x, y) = curveC1 θ ↔ (x^2 / 4 + y^2 / 3 = 1) := 
by
  sorry

theorem cartesian_equation_C2 : ∀ (x y α : ℝ),
  (x, y) = curveC2 α ↔ (x^2 + y^2 = 4) :=
by
  sorry

theorem max_PM_plus_PN : ∀ (α : ℝ),
  let P := curveC2 α,
      M := (1, Real.sqrt 3),
      N := (1, -Real.sqrt 3) in
  Real.dist P M + Real.dist P N ≤ 2 * Real.sqrt 7 := 
by
  sorry

end cartesian_equation_C1_cartesian_equation_C2_max_PM_plus_PN_l545_545385


namespace part1_part2_l545_545906

noncomputable def A (a : ℝ) : set ℝ := {x | (x - 3) * (x - 3 * a - 5) < 0}
noncomputable def B : set ℝ := {x | -2 < x ∧ x < 7}

-- Part 1: If a = 4, then A ∩ B = {x | 3 < x ∧ x < 7}
theorem part1 : A 4 ∩ B = {x : ℝ | 3 < x ∧ x < 7} :=
by
  sorry

-- Part 2: If ∀ x ∈ A (a), x ∈ B => -7/3 ≤ a ≤ 2/3
theorem part2 (a : ℝ) : (∀ x ∈ A a, x ∈ B) ↔ (-7 / 3 ≤ a ∧ a ≤ 2 / 3) :=
by
  sorry

end part1_part2_l545_545906


namespace pies_difference_l545_545778

theorem pies_difference (time : ℕ) (alice_time : ℕ) (bob_time : ℕ) (charlie_time : ℕ)
    (h_time : time = 90) (h_alice : alice_time = 5) (h_bob : bob_time = 6) (h_charlie : charlie_time = 7) :
    (time / alice_time - time / bob_time) + (time / alice_time - time / charlie_time) = 9 := by
  sorry

end pies_difference_l545_545778


namespace mean_of_all_students_l545_545594

variable (M A m a : ℕ)
variable (M_val : M = 84)
variable (A_val : A = 70)
variable (ratio : m = 3 * a / 4)

theorem mean_of_all_students (M A m a : ℕ) (M_val : M = 84) (A_val : A = 70) (ratio : m = 3 * a / 4) :
    (63 * a + 70 * a) / (7 * a / 4) = 76 := by
  sorry

end mean_of_all_students_l545_545594


namespace parallel_lines_x_value_l545_545911

theorem parallel_lines_x_value 
  (l1_through : ∃ p1 p2 : ℝ × ℝ, (p1 = (-1, -2) ∧ p2 = (-1, 4) ∧ line_through p1 p2 = l1))
  (l2_through : ∃ q1 q2 : ℝ × ℝ, q1 = (2, 1) ∧ q2 = (x, 6) ∧ line_through q1 q2 = l2)
  (parallel_l1_l2 : parallel l1 l2) :
  x = 2 := 
sorry

end parallel_lines_x_value_l545_545911


namespace mass_percentage_67_86_l545_545322

noncomputable def mass_percentage (mass_fluorine total_mass: ℝ) :=
  (mass_fluorine / total_mass) * 100

def verify_mass_percentage : Prop :=
  ∃ compound : (total_mass mass_fluorine : ℝ), 
    compound.total_mass = 78.08 ∧ compound.mass_fluorine = 38.0 ∧
    mass_percentage compound.mass_fluorine compound.total_mass = 67.86

theorem mass_percentage_67_86 :
  ¬(∃ mass_fluorine total_mass : ℝ, 
    mass_percentage mass_fluorine total_mass = 67.86 ∧
    ∃ molar_mass CaF2:  (40.08 + 2 * 19.0), total_mass = molar_mass)
:= 
  begin
    sorry
  end

end mass_percentage_67_86_l545_545322


namespace count_four_digit_numbers_divisible_by_5_ending_in_45_l545_545398

theorem count_four_digit_numbers_divisible_by_5_ending_in_45 : 
  ∃ (n : ℕ), n = 90 ∧ ∀ (x : ℕ), (1000 ≤ x ∧ x < 10000) ∧ (x % 100 = 45) → count x = n :=
sorry

end count_four_digit_numbers_divisible_by_5_ending_in_45_l545_545398


namespace find_number_l545_545325

theorem find_number :
  (∃ x : ℝ, x * (3 + Real.sqrt 5) = 1) ∧ (x = (3 - Real.sqrt 5) / 4) :=
sorry

end find_number_l545_545325


namespace coupon_value_l545_545981

theorem coupon_value
  (bill : ℝ)
  (milk_cost : ℝ)
  (bread_cost : ℝ)
  (detergent_cost : ℝ)
  (banana_cost_per_pound : ℝ)
  (banana_weight : ℝ)
  (half_off : ℝ)
  (amount_left : ℝ)
  (total_without_coupon : ℝ)
  (total_spent : ℝ)
  (coupon_value : ℝ) :
  bill = 20 →
  milk_cost = 4 →
  bread_cost = 3.5 →
  detergent_cost = 10.25 →
  banana_cost_per_pound = 0.75 →
  banana_weight = 2 →
  half_off = 0.5 →
  amount_left = 4 →
  total_without_coupon = milk_cost * half_off + bread_cost + detergent_cost + banana_cost_per_pound * banana_weight →
  total_spent = bill - amount_left →
  coupon_value = total_without_coupon - total_spent →
  coupon_value = 1.25 :=
by
  sorry

end coupon_value_l545_545981


namespace right_triangles_with_leg_2012_l545_545522

theorem right_triangles_with_leg_2012 :
  ∃ (a b c : ℕ), (a = 2012 ∧ (a^2 + b^2 = c^2 ∨ b^2 + a^2 = c^2)) ∧
  (b = 253005 ∧ c = 253013 ∨ b = 506016 ∧ c = 506020 ∨ b = 1012035 ∧ c = 1012037 ∨ b = 1509 ∧ c = 2515) :=
begin
  sorry
end

end right_triangles_with_leg_2012_l545_545522


namespace distance_between_efrida_and_frazer_l545_545832

def distance_from_frazer_to_restaurant : ℝ := 11.082951062292475
def closer_distance : ℝ := 2
def distance_from_efrida_to_restaurant : ℝ := distance_from_frazer_to_restaurant - closer_distance

theorem distance_between_efrida_and_frazer :
  let a := distance_from_efrida_to_restaurant in
  let b := distance_from_frazer_to_restaurant in
  let c := real.sqrt (a^2 + b^2) in
  c ≈ 14.331 :=
by
  let a := distance_from_efrida_to_restaurant
  let b := distance_from_frazer_to_restaurant
  let c := real.sqrt (a^2 + b^2)
  have h : c ≈ 14.331 := sorry -- Proof placeholder
  exact h

end distance_between_efrida_and_frazer_l545_545832


namespace sum_common_divisors_60_18_l545_545844

/-- Sum of all positive divisors of 60 that are also divisors of 18 is 12. -/
theorem sum_common_divisors_60_18 : 
  ∑ d in (finset.filter (λ x, 60 ∣ x ∧ 18 ∣ x) (finset.range 61)), d = 12 :=
by 
  sorry

end sum_common_divisors_60_18_l545_545844


namespace jill_win_percentage_is_75_l545_545980

-- Definition of the conditions
def games_played_with_mark : Nat := 10
def mark_wins : Nat := 1
def games_played_with_jill : Nat := 2 * games_played_with_mark
def total_jenny_wins : Nat := 14
def jenny_wins_with_mark : Nat := games_played_with_mark - mark_wins
def jenny_wins_with_jill : Nat := total_jenny_wins - jenny_wins_with_mark
def jill_wins_with_jenny : Nat := games_played_with_jill - jenny_wins_with_jill
def jill_win_percentage : Rat := (jill_wins_with_jenny / games_played_with_jill.toRat) * 100

theorem jill_win_percentage_is_75 :
  jill_win_percentage = 75 := by
  sorry

end jill_win_percentage_is_75_l545_545980


namespace smallest_positive_integer_n_l545_545713

theorem smallest_positive_integer_n :
  ∃ n : ℕ, (n > 0 ∧ 17 * n % 7 = 2) ∧ ∀ m : ℕ, (m > 0 ∧ 17 * m % 7 = 2) → n ≤ m := 
sorry

end smallest_positive_integer_n_l545_545713


namespace count_valid_numbers_l545_545397

-- Define what it means to be a four-digit number that ends in 45
def is_valid_number (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999 ∧ n % 100 = 45 ∧ n % 5 = 0

-- Define the set of valid two-digit prefixes
def valid_prefixes : set ℕ := {ab | 10 ≤ ab ∧ ab ≤ 99}

-- Define the set of four-digit numbers that end in 45 and are divisible by 5
def valid_numbers : set ℕ := {n | ∃ ab : ℕ, ab ∈ valid_prefixes ∧ n = ab * 100 + 45}

-- State the theorem
theorem count_valid_numbers : (finset.card (finset.filter is_valid_number (finset.range 10000)) = 90) :=
sorry

end count_valid_numbers_l545_545397


namespace circle_contains_at_least_three_points_l545_545218

theorem circle_contains_at_least_three_points :
  ∀ (points : Fin 51 → ℝ × ℝ), ∃ (c : ℝ × ℝ), ∃ (r : ℝ),
    (r = 1 / 7) ∧
    (∃ i j k, i ≠ j ∧ i ≠ k ∧ j ≠ k ∧
      dist (points i) c ≤ r ∧
      dist (points j) c ≤ r ∧
      dist (points k) c ≤ r) :=
begin
  sorry
end

end circle_contains_at_least_three_points_l545_545218


namespace find_m_value_l545_545383

theorem find_m_value (x: ℝ) (m: ℝ) (hx: x > 2) (hm: m > 0) (h_min: ∀ y, (y = x + m / (x - 2)) → y ≥ 6) : m = 4 := 
sorry

end find_m_value_l545_545383


namespace max_total_distance_of_Bob_l545_545601

noncomputable def max_bob_total_distance (Alice_distances : Fin 99 → ℝ) (Bob : Fin 99 → ℝ) (distance_Alice_to_Bob : ℝ) : ℝ :=
  98 * distance_Alice_to_Bob + 1000

theorem max_total_distance_of_Bob (Alice Bob : Fin 100 → ℝ) (total_distance_Alice : ℝ) :
  (∀ i, i ≠ 99 → (Alice i) = 0 ∨ (Alice i ≤ total_distance_Alice)) →
  (∀ i, i ≠ 99 → (Bob i ≤ Alice i + distance_Alice_to_Bob) →
  total_distance_Alice = 1000 →
  distance_Alice_to_Bob = 1000 →
  max_bob_total_distance = 99000 :=
by
  sorry

end max_total_distance_of_Bob_l545_545601


namespace find_tangent_line_equation_l545_545136

def curve (x : ℝ) : ℝ := x^3 - 2 * x

def tangent_line_at_point : Prop :=
  ∃ m b, (m = 3 * 1^2 - 2) ∧ (curve 1 = -1) ∧ (∀ x y, y = m * (x - 1) + curve 1 ↔ x - y - 2 = 0)

theorem find_tangent_line_equation : tangent_line_at_point :=
by
  sorry

end find_tangent_line_equation_l545_545136


namespace correct_statement_about_cell_proliferation_l545_545226

variable (Cell : Type) (cytoplasm : Cell → CytoPlasm) (chromosomes : Cell → Set Chromosome)
variable (homologous : Chromosome → Chromosome → Prop)
variable (cellDivision : Cell → Cell → Prop)
variable (stableGeneticMaterial : Prop)
variable (randomUnequalDistribution : Cell → Prop)

-- Definitions corresponding to conditions
def at_late_stage_of_mitosis_diploid_animal_cells_each_pole_does_not_contain_homologous_chromosomes : Prop :=
  ∀ (c1 c2 : Cell), cellDivision c1 c2 → (∃ (ch1 ch2 : Chromosome), homologous ch1 ch2 → ¬ (ch1 ∈ chromosomes c2 ∧ ch2 ∈ chromosomes c2))

def genetic_material_is_randomly_distributed_during_cell_division : Prop :=
  ∀ (c : Cell), randomUnequalDistribution c

def at_late_stage_second_meiotic_division_chromosome_count_is_half_somatic : Prop :=
  ∀ (c1 c2 : Cell), cellDivision c1 c2 → c2.chromosomes.card = c1.chromosomes.card / 2

def alleles_separate_in_first_meiotic_division_independent_assortment_in_second : Prop :=
  ∀ (c1 c2 c3 : Cell), cellDivision c1 c2 ∧ cellDivision c2 c3 → 
    (∃ (ch1 ch2 : Chromosome), homologous ch1 ch2 ∧ ch1 ∈ chromosomes c2 ∧ ch2 ∈ chromosomes c2) ∧
    (∀ (ch3 ch4 : Chromosome), ¬ homologous ch3 ch4 ∧ ch3 ∈ chromosomes c3 ∧ ch4 ∈ chromosomes c3)

-- Statement to be proven
theorem correct_statement_about_cell_proliferation :
  ¬ at_late_stage_of_mitosis_diploid_animal_cells_each_pole_does_not_contain_homologous_chromosomes →
  stableGeneticMaterial →
  genetic_material_is_randomly_distributed_during_cell_division →
  ¬ at_late_stage_second_meiotic_division_chromosome_count_is_half_somatic →
  ¬ alleles_separate_in_first_meiotic_division_independent_assortment_in_second →
  genetic_material_is_randomly_distributed_during_cell_division :=
by
  intros h1 h2 h3 h4 h5
  exact h3

end correct_statement_about_cell_proliferation_l545_545226


namespace count_four_digit_numbers_divisible_by_5_ending_in_45_l545_545422

theorem count_four_digit_numbers_divisible_by_5_ending_in_45 : 
  ∃ n : ℕ, (∀ x : ℕ, x ≥ 1000 ∧ x < 10000 ∧ x % 100 = 45 → (x % 5 = 0 ∧ x % 100 = 45)) 
  ∧ (n = 90) :=
by
  sorry

end count_four_digit_numbers_divisible_by_5_ending_in_45_l545_545422


namespace average_width_is_correct_l545_545555

-- Definitions based on the conditions
def book_widths : List ℝ := [4, 0.5, 1.2, 3, 7.5, 2, 5, 9]
def num_books : ℕ := 8

-- The average width calculated
def average_width : ℝ := (List.sum book_widths) / (num_books)

-- Statement to be proved
theorem average_width_is_correct : average_width = 4.025 := by
  sorry

end average_width_is_correct_l545_545555


namespace find_beta_l545_545772

-- Definitions based on conditions
variables (m L g d k : ℝ)

-- Rotational inertia of the rod about its center
def I_cm : ℝ := m * d^2

-- Rotational inertia of the rod about the suspension point
def I_susp : ℝ := I_cm + m * (k * d)^2

-- Angular frequency in terms of beta, gravity, and length d
def omega (β : ℝ) : ℝ := β * Real.sqrt (g / d)

-- Simplified angular frequency derived from the rod's oscillations
def omega_simplified : ℝ := Real.sqrt (k * g / (1 + k^2) / d)

-- The proposition we need to prove: that omega (β) and omega_simplified are equivalent
theorem find_beta : ∃ β, omega β = omega_simplified :=
begin
  use Real.sqrt (k / (1 + k^2)),
  sorry
end

end find_beta_l545_545772


namespace largest_of_five_consecutive_even_integers_sum_l545_545157

theorem largest_of_five_consecutive_even_integers_sum (n : ℤ) :
  (∑ i in finset.range 25, (2 * (i + 1))) = 650 ∧
  (∑ i in finset.range 5, (n - 8 + 2 * i)) = 650 →
  n = 134 :=
by 
  sorry

end largest_of_five_consecutive_even_integers_sum_l545_545157


namespace smallest_rel_prime_gt_1_2310_l545_545709

theorem smallest_rel_prime_gt_1_2310 :
  ∃ n > 1, (∀ m > 1, m < n → ¬ (nat.coprime m 2310)) ∧ nat.coprime n 2310 :=
  exists.intro 13 (and.intro (exists.intro (13:ℤ) 
  (and.intro (nat.succ_lt_succ (nat.succ_lt_succ (zero_lt_one)))
  (forall.intro (λ m (hm : m > 1) (hmn : m < 13), (not_intro 
    (λ (h : nat.gcd m 2310 = 1),
      have h2 : m ≥ 2 → m ≤ 3 → nat.gcd m 2310 ≠ 1, from sorry,
      have h5 : m ≤ 11, from sorry, 
      show false, from this (le_total m 11) h5))))))
  (nat.gcd_eq_one_of_coprime .
      (nat.prime_coprime (nat.prime_factorisation_complete 2) nat.factorisation_complete
        (or.elim (not_or_distrib.mp (not_or_distrib.mp (not_or_distrib.mp
      (not_or_distrib.mp (not_and_distrib.mpr (or.inr (not_and_distrib.mpr (or.inr (not_and_distrib.mpr (or.inr (not_and_distrib.mpr (or.inr (not_and_distrib.mpr (or.inl (nat.gcd_comm 13 2310)))))))))))))))) ⊥))))
⟩ sorry

end smallest_rel_prime_gt_1_2310_l545_545709


namespace meat_cost_per_kg_l545_545658

noncomputable def cheese_kg : ℝ := 1.5
noncomputable def meat_kg : ℝ := 0.5
noncomputable def cheese_cost_per_kg : ℝ := 6
noncomputable def total_cost : ℝ := 13

def cost_of_cheese := cheese_kg * cheese_cost_per_kg
def cost_of_meat := total_cost - cost_of_cheese
def cost_per_kg_of_meat := cost_of_meat / meat_kg

theorem meat_cost_per_kg : cost_per_kg_of_meat = 8 := by
  sorry

end meat_cost_per_kg_l545_545658


namespace photos_on_last_page_l545_545018

noncomputable def total_photos : ℕ := 10 * 35 * 4
noncomputable def photos_per_page_after_reorganization : ℕ := 8
noncomputable def total_pages_needed : ℕ := (total_photos + photos_per_page_after_reorganization - 1) / photos_per_page_after_reorganization
noncomputable def pages_filled_in_first_6_albums : ℕ := 6 * 35
noncomputable def last_page_photos : ℕ := if total_pages_needed ≤ pages_filled_in_first_6_albums then 0 else total_photos % photos_per_page_after_reorganization

theorem photos_on_last_page : last_page_photos = 0 :=
by
  sorry

end photos_on_last_page_l545_545018


namespace significant_figures_left_l545_545847

theorem significant_figures_left (n : ℕ) (f : ℕ → ℕ) (accurate_digit : ℕ) :
  (∀ k, k < n → f k = 0) →
  (∀ k, f (n + k) ≠ 0) →
  (∀ k, k ≤ accurate_digit → f (n + k) ≠ 0) →
  all_significant_digits_are_on_left (f, n, accurate_digit) :=
by
  sorry

end significant_figures_left_l545_545847


namespace initial_sand_amount_l545_545270

theorem initial_sand_amount (loss : ℝ) (arrival : ℝ) (initial : ℝ) 
    (h_loss : loss = 2.4) (h_arrival : arrival = 1.7) (h_initial : initial = loss + arrival) : 
    initial = 4.1 :=
by
  rw [h_loss, h_arrival, h_initial]
  linarith

end initial_sand_amount_l545_545270


namespace number_of_odd_functions_l545_545780

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = - f x

def y1 : ℝ → ℝ := λ x, x^3
def y2 : ℝ → ℝ := λ x, 2 * x
def y3 : ℝ → ℝ := λ x, x^2 + 1
def y4 : ℝ → ℝ := λ x, 2 * sin x

theorem number_of_odd_functions : 
  (if is_odd_function y1 then 1 else 0) + 
  (if is_odd_function y2 then 1 else 0) + 
  (if is_odd_function y3 then 1 else 0) + 
  (if is_odd_function y4 then 1 else 0) = 2 := 
by 
  sorry

end number_of_odd_functions_l545_545780


namespace polynomial_expansion_a5_l545_545951

theorem polynomial_expansion_a5 :
  (x - 1) ^ 8 = (1 : ℤ) + a₁ * x + a₂ * x^2 + a₃ * x^3 + a₄ * x^4 + a₅ * x^5 + a₆ * x^6 + a₇ * x^7 + a₈ * x^8 →
  a₅ = -56 :=
by
  intro h
  -- The proof is omitted.
  sorry

end polynomial_expansion_a5_l545_545951


namespace cloak_change_l545_545500

theorem cloak_change (silver_for_cloak1 silver_for_cloak2 : ℕ) (gold_change1 gold_change2 : ℕ) (silver_change : ℕ) :
  silver_for_cloak1 = 20 →
  gold_change1 = 4 →
  silver_for_cloak2 = 15 →
  gold_change2 = 1 →
  ∃ silver_cost_of_cloak : ℕ, 
    silver_cost_of_cloak = (20 - 4) * (5 / 3) →
    silver_change = 10 →
    14 * (5 / 3) - 8 = silver_change :=
by 
  assume h1 h2 h3 h4,
  use 16, 
  sorry

end cloak_change_l545_545500


namespace length_of_CD_l545_545151

-- Define the condition indicating the volume of the region within 5 units of the line segment CD.
def volume_within_5_units_of_line_segment (CD_length : ℝ) : ℝ :=
  let cylinder_volume := 25 * pi * CD_length
  let hemispheres_volume := (500 / 3) * pi
  cylinder_volume + hemispheres_volume

-- State the theorem to prove that the length CD is 40/3 when the volume condition is given.
theorem length_of_CD (h : volume_within_5_units_of_line_segment CD_length = 500 * pi) : CD_length = 40 / 3 :=
  sorry

end length_of_CD_l545_545151


namespace smallest_n_l545_545369

theorem smallest_n (n : ℕ) (a : Fin n.succ → ℕ)
  (h1 : ∀ i j : Fin n.succ, i < j → a i < a j)
  (h2 : ∀ i : Fin n.succ, 0 < a i)
  (h3 : ∑ i, (1 : ℚ) / a i = 13 / 14) : 
  n = 3 :=
by sorry

end smallest_n_l545_545369


namespace bags_filled_l545_545107

def bags_filled_on_certain_day (x : ℕ) : Prop :=
  let bags := x + 3
  let total_cans := 8 * bags
  total_cans = 72

theorem bags_filled {x : ℕ} (h : bags_filled_on_certain_day x) : x = 6 :=
  sorry

end bags_filled_l545_545107


namespace original_price_of_pants_l545_545790

theorem original_price_of_pants (P : ℝ) 
  (sale_discount : ℝ := 0.50)
  (saturday_additional_discount : ℝ := 0.20)
  (savings : ℝ := 50.40)
  (saturday_effective_discount : ℝ := 0.40) :
  savings = 0.60 * P ↔ P = 84.00 :=
by
  sorry

end original_price_of_pants_l545_545790


namespace binary_to_decimal_l545_545621

theorem binary_to_decimal :
  let binary := [1, 1, 0, 0, 1, 0, 0, 1]
  in binary.enum_from 0 |>.map (λ (idx, bit), bit * 2^idx) |>.sum = 201 :=
by
  let binary := [1, 1, 0, 0, 1, 0, 0, 1]
  show binary.enum_from 0 |>.map (λ (idx, bit), bit * 2^idx) |>.sum = 201
  sorry

end binary_to_decimal_l545_545621


namespace max_area_enclosed_parabolas_l545_545817

def parabola_a (a : ℝ) := λ x : ℝ, -2 * x ^ 2 + 4 * a * x - 2 * a ^ 2 + a + 1
def parabola (x : ℝ) := x ^ 2 - 2 * x

theorem max_area_enclosed_parabolas : 
  ∀ (a : ℝ), (∃ x : ℝ, parabola x = parabola_a a x) → 
  ∃ A : ℝ, 
    (A = ∫ x in (7 - sqrt 154) / 6 .. (7 + sqrt 154) / 6,  3 * x^2 - 7 * x + 7 / 8) ∧ 
    (A ≤ 27 / (4 * sqrt 2)) :=
begin
  sorry
end

end max_area_enclosed_parabolas_l545_545817


namespace findAG_l545_545541

open Set
open Classical

noncomputable def circlesConfiguration (r : ℝ) (A B C D E F G : Point) : Prop :=
  let ω_A := Circle A r
  let ω_B := Circle B r
  let ω := Circle ((A+B)/2) (3*r/2)
  tangent ω_A ω_B ∧ congruent ω_A ω_B ∧
  tangentAt ω ω_A C ∧ passes_through ω B ∧ 
  collinear [C, A, B] ∧ 
  segment_intersect_line_segments [C, D, E, F, G] [ω_A, ω_B, ω] ∧
  dist D E = 6 ∧
  dist F G = 9

theorem findAG {r : ℝ} {A B C D E F G : Point} 
  (h : circlesConfiguration r A B C D E F G) : dist A G = 9 * sqrt 19 := 
  sorry

end findAG_l545_545541


namespace even_perfect_square_factors_count_l545_545433

theorem even_perfect_square_factors_count :
  let possible_a := {x // x ≤ 4 ∧ x ≥ 1 ∧ x % 2 = 0}
  let possible_b := {x // x ≤ 9 ∧ x % 2 = 0}
  finset.card (finset.product possible_a possible_b) = 10 := by
  sorry

end even_perfect_square_factors_count_l545_545433


namespace triangle_congruence_difference_l545_545058

theorem triangle_congruence_difference (x y : ℝ) 
  (h1 : ∃ t1 t2 t3 : ℝ, t1 = 3 ∧ t2 = 5 ∧ t3 = x ∧ triangle t1 t2 t3)
  (h2 : ∃ t1 t2 t3 : ℝ, t1 = y ∧ t2 = 3 ∧ t3 = 6 ∧ triangle t1 t2 t3)
  (congruent_triangles : congruent (3, 5, x) (y, 3, 6)) :
  x - y = 1 :=
by
  sorry

end triangle_congruence_difference_l545_545058


namespace volume_of_pyramid_BCHE_N_is_2sqrt5_l545_545296

/-
  We define our problem setup.
  AB = 4, BC = 2, CG = 3 define the dimensions.
  Point N being the midpoint of FG implies certain conditions.
  H and E are specific vertices.
-/

structure RectangularParallelepiped :=
  (AB BC CG : ℝ)

def midpoint (a b : ℝ) : ℝ := (a + b) / 2

def point_N (rp : RectangularParallelepiped) : ℝ := midpoint 0 rp.CG -- Simplified for representation

noncomputable def volume_pyramid (base_area height : ℝ) : ℝ := (1 / 3) * base_area * height

def base_BCHE_area (rp : RectangularParallelepiped) : ℝ :=
  let BC := rp.BC in
  let BE := real.sqrt (rp.AB ^ 2 + rp.BC ^ 2) in
  BC * BE

theorem volume_of_pyramid_BCHE_N_is_2sqrt5 (rp : RectangularParallelepiped)
  (h₁ : rp.AB = 4) (h₂ : rp.BC = 2) (h₃ : rp.CG = 3) :
  volume_pyramid (base_BCHE_area rp) (rp.CG / 2) = 2 * real.sqrt 5 :=
by
  sorry

end volume_of_pyramid_BCHE_N_is_2sqrt5_l545_545296


namespace sum_of_digits_l545_545532

theorem sum_of_digits :
  ∃ (E M V Y : ℕ), 
    (E ≠ M ∧ E ≠ V ∧ E ≠ Y ∧ M ≠ V ∧ M ≠ Y ∧ V ≠ Y) ∧
    (10 * Y + E) * (10 * M + E) = 111 * V ∧ 
    1 ≤ V ∧ V ≤ 9 ∧ 
    E + M + V + Y = 21 :=
by 
  sorry

end sum_of_digits_l545_545532


namespace quadratic_inequality_solution_l545_545040

theorem quadratic_inequality_solution (a b c : ℝ) (h : a < 0) 
  (h_sol : ∀ x, ax^2 + bx + c > 0 ↔ x > -2 ∧ x < 1) :
  ∀ x, ax^2 + (a + b) * x + c - a < 0 ↔ x < -3 ∨ x > 1 := 
sorry

end quadratic_inequality_solution_l545_545040


namespace women_science_majors_is_30_percent_l545_545480

noncomputable def percentage_women_science_majors (ns_percent : ℝ) (m_percent : ℝ) (m_sci_percent : ℝ) : ℝ :=
  let w_percent := 1 - m_percent
  let m_sci_total := m_percent * m_sci_percent
  let total_sci := 1 - ns_percent
  let w_sci_total := total_sci - m_sci_total
  (w_sci_total / w_percent) * 100

theorem women_science_majors_is_30_percent :
  percentage_women_science_majors 0.60 0.40 0.55 = 30 := by
  sorry

end women_science_majors_is_30_percent_l545_545480


namespace jason_placed_erasers_l545_545177

theorem jason_placed_erasers (initial_erasers new_total_erasers : ℕ) 
  (h1 : initial_erasers = 139) 
  (h2 : new_total_erasers = 270) : 
  new_total_erasers - initial_erasers = 131 :=
by
  rw [h1, h2]
  exact rfl

end jason_placed_erasers_l545_545177


namespace arithmetic_sequence_general_formula_l545_545373

-- Definitions of the arithmetic sequence {a_n} and geometric sequence {b_n}
def arithmetic_seq (a : ℕ → ℕ) : Prop :=
  ∃ d : ℕ, ∀ n : ℕ, a (n + 1) = a n + d

def geometric_seq (b : ℕ → ℕ) : Prop :=
  ∃ q : ℕ, ∀ n : ℕ, b (n + 1) = b n * q

-- Conditions
def a1 := 2
def sum_terms (a b : ℕ → ℕ) (n : ℕ) : ℕ :=
  (2 * n + 1) * 3 ^ n - 1

-- The statement to be proved
theorem arithmetic_sequence_general_formula (a b : ℕ → ℕ) 
  (h_arith : arithmetic_seq a) (h_geom : geometric_seq b)
  (h_a1 : a 1 = a1)
  (h_sum : ∀ n, ∑ i in finset.range n, a (i + 1) * b (i + 1) = sum_terms a b n) :
  ∀ n, a n = n + 1 :=
by
  sorry

end arithmetic_sequence_general_formula_l545_545373


namespace find_g_inverse_sum_l545_545568

noncomputable def g (x : ℝ) : ℝ :=
if x ≤ 2 then x^2 - 2 * x + 2 else 3 - x

theorem find_g_inverse_sum :
  (∃ x, g x = -2 ∧ x = 5) ∧
  (∃ x, g x = 0 ∧ x = 3) ∧
  (∃ x, g x = 2 ∧ x = 0) ∧
  (5 + 3 + 0 = 8) := by
  sorry

end find_g_inverse_sum_l545_545568


namespace selling_price_of_third_house_l545_545285

-- Definitions based on conditions
def commission_rate := 0.02
def sale_price_1 := 157000
def sale_price_2 := 125000
def total_commission := 15620

-- Definition of the commission for the first two houses
def commission_1 := sale_price_1 * commission_rate
def commission_2 := sale_price_2 * commission_rate
def total_commission_1_2 := commission_1 + commission_2

-- Commission for the third house
def commission_3 := total_commission - total_commission_1_2

-- Selling price of the third house
def selling_price_3 := commission_3 / commission_rate

theorem selling_price_of_third_house : selling_price_3 = 499000 := by
  sorry

end selling_price_of_third_house_l545_545285


namespace find_lambda_l545_545881

variables (a b : ℝ^3) (λ : ℝ)
-- Conditions
def angle_ab := real.angle a b = 2 * real.pi / 3
def norm_a := ∥a∥ = 1
def norm_b := ∥b∥ = 4
def orthogonality := (2 • a + λ • b) ⬝ a = 0

-- Question to answer
theorem find_lambda (ha : angle_ab a b) (hna : norm_a a) (hnb : norm_b b) (orth : orthogonality a b λ) : λ = 1 :=
sorry

end find_lambda_l545_545881


namespace right_triangle_area_l545_545696

/-- Given a right triangle with one leg of length 30 inches and a hypotenuse of 34 inches,
    the area of the triangle is 240 square inches. -/
theorem right_triangle_area (a b c : ℝ) (h1 : a = 30) (h2 : c = 34) (h3 : a^2 + b^2 = c^2) :
  (1/2) * a * b = 240 :=
by
  rw [h1, h2] at h3
  have hb : b = 16 := by
    rw [←h3]
    norm_num
  rw [h1, hb]
  norm_num
  sorry

end right_triangle_area_l545_545696


namespace find_right_triangles_with_leg_2012_l545_545523

theorem find_right_triangles_with_leg_2012 :
  ∃ x y z : ℕ, x^2 + 2012^2 = y^2 ∧ x + y + z = 2012 + 253005 + 253013 ∨
  x^2 + 2012^2 = z^2 ∧ x + y + z = 2012 + 506016 + 506020 ∨
  x^2 + 2012^2 = y^2 ∧ x + y + z = 2012 + 1012035 + 1012037 ∨
  x^2 + 2012^2 = y^2 ∧ x + y + z = 2012 + 1509 + 2515 ∧
  y ≠ 2012 :=
begin
  sorry  -- The actual proof is omitted as specified.
end

end find_right_triangles_with_leg_2012_l545_545523


namespace parabola_line_slope_l545_545903

noncomputable def parab_slope_product (p : ℝ) (hx : p > 0) : ℝ :=
  let x1 := x1 in
  let x2 := x2 in
  let y1 := y1 in
  let y2 := y2 in
  let AM_slope := (y1 - 2) / (x1 - 2) in
  let BM_slope := (y2 - 2) / (x2 - 2) in
  AM_slope * BM_slope

theorem parabola_line_slope (p : ℝ) (hx : p > 0) :
  (x1 + x2 = (8 + p) / 2) →
  (x1 * x2 = 4) →
  (y1 + y2 = p) →
  (y1 * y2 = -4 * p) →
  (5 * p * 5 * p = 2 * p * ((40 + 5 * p) / 2)) →
  parab_slope_product p hx = 4 := 
  sorry

end parabola_line_slope_l545_545903


namespace triangle_area_of_ellipse_l545_545462

theorem triangle_area_of_ellipse (x y : ℝ) (P F1 F2 : ℝ) : 
  let a := 6 in
  let b := 4 in
  let c := Real.sqrt (a^2 - b^2) in
  let dist_PF1 := 12 - |P - F2| in
  let dist_PF2 := 12 - dist_PF1 in
  |dist_PF1 + dist_PF2| = 2 * a ∧ dist_PF1^2 + dist_PF2^2 = (2 * c)^2 ∧ dist_PF1 * dist_PF2 = 32 →
  (1 / 2) * dist_PF1 * dist_PF2 = 16 :=
by 
  sorry

end triangle_area_of_ellipse_l545_545462


namespace rate_percent_simple_interest_l545_545282

-- Definitions based on conditions
def simple_interest (P : ℕ) (R : ℕ) (T : ℕ) : ℚ := (P * R * T) / 100

-- Problem statement
theorem rate_percent_simple_interest (P : ℕ) (R : ℕ) :
  simple_interest P R 10 = 6 / 5 * P → R = 12 :=
by
  intro h,
  sorry

end rate_percent_simple_interest_l545_545282


namespace prod_mod7_eq_zero_l545_545806

theorem prod_mod7_eq_zero :
  (2023 * 2024 * 2025 * 2026) % 7 = 0 := 
by {
  sorry
}

end prod_mod7_eq_zero_l545_545806


namespace buying_ways_l545_545178

theorem buying_ways (students : ℕ) (choices : ℕ) (at_least_one_pencil : ℕ) : 
  students = 4 ∧ choices = 2 ∧ at_least_one_pencil = 1 → 
  (choices^students - 1) = 15 :=
by
  sorry

end buying_ways_l545_545178


namespace find_ratio_l545_545091

noncomputable def points_and_midpoints (a b c : ℝ) (P Q R : ℝ × ℝ × ℝ) :=
  (P + Q) / 2 = (a, 0, 0) ∧
  (P + R) / 2 = (0, b, 0) ∧
  (Q + R) / 2 = (0, 0, c)

theorem find_ratio (a b c : ℝ) (P Q R : ℝ × ℝ × ℝ) 
  (h : points_and_midpoints a b c P Q R) :
  (P - Q).dist^2 + (P - R).dist^2 + (Q - R).dist^2 = 8 * (a^2 + b^2 + c^2) :=
sorry

end find_ratio_l545_545091


namespace valid_parametrizations_l545_545299

open Real

def line_eq (x y : ℝ) : Prop := y = -3 * x + 4

def param_A (t : ℝ) : ℝ × ℝ := (0 + t * 1, 4 + t * -3)
def param_B (t : ℝ) : ℝ × ℝ := (-2/3 + t * 3, 0 + t * -9)
def param_C (t : ℝ) : ℝ × ℝ := (-4/3 + t * 2, 8 + t * -6)
def param_D (t : ℝ) : ℝ × ℝ := (-2 + t * (1/2), 10 + t * -1)
def param_E (t : ℝ) : ℝ × ℝ := (1 + t * 4, 1 + t * -12)

theorem valid_parametrizations :
  (∀ t, line_eq (param_A t).1 (param_A t).2) ∧
  (∀ t, line_eq (param_B t).1 (param_B t).2) ∧
  (∀ t, line_eq (param_C t).1 (param_C t).2) ∧
  (∀ t, line_eq (param_D t).1 (param_D t).2) ∧
  (∀ t, line_eq (param_E t).1 (param_E t).2) :=
by
  sorry

end valid_parametrizations_l545_545299


namespace min_hexagon_area_l545_545663

theorem min_hexagon_area (T1 T2 : Set Point)
  (h_intersect : (T1 ∩ T2).finite)
  (h_regions : ∃ regions : FiniteSet (Set Point), 
                 regions.card = 7 ∧ 
                 ∀ r ∈ regions, is_triangle r ∨ is_hexagon r)
  (h_triangular_areas: ∃ trigs : FiniteSet (Set Point), 
                        trigs.card = 6 ∧ 
                        ∀ t ∈ trigs, area t = 1)
  (h_hexagon_area: ∃ h : Set Point, is_hexagon h ∧ h ∈ regions ∧ area h = A) :
  A ≥ 6 :=
sorry

end min_hexagon_area_l545_545663


namespace tetrahedron_labelings_zero_l545_545831

theorem tetrahedron_labelings_zero : 
  ∀ (label : Fin 4 → ℕ), 
    (∀ i, label i ∈ {1, 2, 3, 4}) → 
    (∃! label', IsRotation label label') → 
    (∀ f : Fin 4 → Fin 3, (label(f 0) + label(f 1) + label(f 2)) = (label(f 1) + label(f 2) + label(f 0))) → 
  false := 
by
  sorry

end tetrahedron_labelings_zero_l545_545831


namespace area_of_right_triangle_l545_545700

-- Given definitions
def leg_a : ℝ := 30
def hypotenuse_c : ℝ := 34

-- The theorem statement
theorem area_of_right_triangle : 
  ∀ (b : ℝ), b = real.sqrt (hypotenuse_c^2 - leg_a^2) → 
  let area := 1 / 2 * leg_a * b in
  area = 240 := 
by
  intro b
  intro h
  let area := 1 / 2 * leg_a * b
  sorry

end area_of_right_triangle_l545_545700


namespace count_four_digit_numbers_divisible_by_five_ending_45_l545_545427

-- Define the conditions as necessary in Lean
def is_four_digit_number (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

def is_divisible_by_five (n : ℕ) : Prop :=
  n % 5 = 0

def ends_with_45 (n : ℕ) : Prop :=
  n % 100 = 45

-- Statement that there exists 90 such four-digit numbers
theorem count_four_digit_numbers_divisible_by_five_ending_45 : 
  { n : ℕ // is_four_digit_number n ∧ is_divisible_by_five n ∧ ends_with_45 n }.card = 90 :=
sorry

end count_four_digit_numbers_divisible_by_five_ending_45_l545_545427


namespace f_of_log2_3_eq_24_l545_545894

-- define the function f according to the problem's conditions
noncomputable def f : ℝ → ℝ
| x := if x >= 4 then 2^x else f (x + 2)

-- main theorem to prove
theorem f_of_log2_3_eq_24 : f (1 + real.logb 2 3) = 24 :=
by
  sorry

end f_of_log2_3_eq_24_l545_545894


namespace highest_mean_possible_l545_545042

def max_arithmetic_mean (g : Matrix (Fin 3) (Fin 3) ℕ) : ℚ := 
  let mean (a b c d : ℕ) : ℚ := (a + b + c + d : ℚ) / 4
  let circles := [
    mean (g 0 0) (g 0 1) (g 1 0) (g 1 1),
    mean (g 0 1) (g 0 2) (g 1 1) (g 1 2),
    mean (g 1 0) (g 1 1) (g 2 0) (g 2 1),
    mean (g 1 1) (g 1 2) (g 2 1) (g 2 2)
  ]
  (circles.sum / 4)

theorem highest_mean_possible :
  ∃ g : Matrix (Fin 3) (Fin 3) ℕ, 
  (∀ i j, 1 ≤ g i j ∧ g i j ≤ 9) ∧ 
  max_arithmetic_mean g = 6.125 :=
by
  sorry

end highest_mean_possible_l545_545042


namespace proof_problem_l545_545013

-- Define the conditions and variables
variables {V : Type*} [inner_product_space ℝ V]
variables (a b : V)
variables (x1 x2 x3 x4 x5 y1 y2 y3 y4 y5 : V)

-- Conditions on vectors
variables (h_a_ne_zero : a ≠ 0)
variables (h_b_ne_zero : b ≠ 0)
variables (h_x : ∀ i, (i = x1 ∨ i = x2 ∨ i = x3 ∨ i = x4 ∨ i = x5) → (i = 2 • a + 3 • b ∨ i = 2 • b + 3 • a))
variables (h_y : ∀ i, (i = y1 ∨ i = y2 ∨ i = y3 ∨ i = y4 ∨ i = y5) → (i = 2 • a + 3 • b ∨ i = 2 • b + 3 • a))

-- Definition of S
def S := ⟪x1, y1⟫ + ⟪x2, y2⟫ + ⟪x3, y3⟫ + ⟪x4, y4⟫ + ⟪x5, y5⟫

-- Definitions of propositions
def proposition_2 : Prop := inner a b = 0 → S = ∥b∥^2
def proposition_4 : Prop := ∥b∥ > 4 * ∥a∥ → S > 0

-- Main statement
theorem proof_problem : proposition_2 a b x1 x2 x3 x4 x5 y1 y2 y3 y4 y5 ∧ proposition_4 a b x1 x2 x3 x4 x5 y1 y2 y3 y4 y5 :=
by sorry

end proof_problem_l545_545013


namespace increasing_interval_of_log_function_l545_545140

noncomputable def f (x : ℝ) : ℝ := x^2 - 3 * x + 2

theorem increasing_interval_of_log_function :
  ∀ x : ℝ, x < 1 → f x > 0 → (x ∈ (-∞, 1)) :=
begin
  intros x hx hfx,
  unfold f at hfx,
  sorry
end

end increasing_interval_of_log_function_l545_545140


namespace count_odd_numbers_between_101_and_499_l545_545020

theorem count_odd_numbers_between_101_and_499 : 
  (finset.filter (λ n => n % 2 = 1) (finset.Ico 101 500)).card = 200 := 
sorry

end count_odd_numbers_between_101_and_499_l545_545020


namespace proof_a_square_plus_a_plus_one_l545_545438

theorem proof_a_square_plus_a_plus_one (a : ℝ) (h : 2 * (5 - a) * (6 + a) = 100) : a^2 + a + 1 = -19 := 
by 
  sorry

end proof_a_square_plus_a_plus_one_l545_545438


namespace cereal_economy_ranking_l545_545264

noncomputable def rank_cereal (c_S q_S : ℝ) (h1 : ℝ) (h2 : ℝ) (h3 : ℝ) (h4 : ℝ) : Prop :=
  let c_M := 1.7 * c_S in
  let q_M := 1.25 * q_S in
  let c_L := 1.2 * c_M in
  let q_L := 1.5 * q_M in
  let cost_per_unit_S := c_S / q_S in
  let cost_per_unit_M := c_M / q_M in
  let cost_per_unit_L := c_L / q_L in
  cost_per_unit_L < cost_per_unit_S ∧ cost_per_unit_S < cost_per_unit_M

theorem cereal_economy_ranking (c_S q_S : ℝ) (h1 : c_S > 0) (h2 : q_S > 0) : rank_cereal c_S q_S 1.7 1.25 1.2 1.5 :=
by 
  unfold rank_cereal
  let c_M := 1.7 * c_S
  let q_M := 1.25 * q_S
  let c_L := 1.2 * c_M
  let q_L := 1.5 * q_M
  let cost_per_unit_S := c_S / q_S
  let cost_per_unit_M := c_M / q_M
  let cost_per_unit_L := c_L / q_L
  dsimp only
  sorry

end cereal_economy_ranking_l545_545264


namespace volume_is_zero_l545_545443

variables (a b : ℝ^3)

-- Given conditions
axiom unit_vector_a : ‖a‖ = 1
axiom unit_vector_b : ‖b‖ = 1
axiom angle_ab : real.angle a b = π / 4

-- Define the volume of the parallelepiped
def volume_parallelepiped := |a ⬝ ((a + (b ⬝ a)) ⬝ b)|

-- The goal is to prove that this volume is 0
theorem volume_is_zero : volume_parallelepiped a b = 0 :=
sorry

end volume_is_zero_l545_545443


namespace area_of_right_triangle_l545_545702

-- Given definitions
def leg_a : ℝ := 30
def hypotenuse_c : ℝ := 34

-- The theorem statement
theorem area_of_right_triangle : 
  ∀ (b : ℝ), b = real.sqrt (hypotenuse_c^2 - leg_a^2) → 
  let area := 1 / 2 * leg_a * b in
  area = 240 := 
by
  intro b
  intro h
  let area := 1 / 2 * leg_a * b
  sorry

end area_of_right_triangle_l545_545702


namespace investment_amount_l545_545941

noncomputable def present_value
  (future_value : ℝ)
  (interest_rate : ℝ)
  (number_of_periods : ℕ) : ℝ :=
future_value / (1 + interest_rate) ^ number_of_periods

theorem investment_amount (FV : ℝ) (r : ℝ) (n : ℕ) :
  FV = 500000 → r = 0.05 → n = 10 →
  present_value FV r n ≈ 306956.63 :=
by
  intros hFV hr hn
  rw [hFV, hr, hn]
  change 500000 / (1 + 0.05) ^ 10 ≈ 306956.63
  sorry

end investment_amount_l545_545941


namespace volume_is_zero_l545_545444

variables (a b : ℝ^3)

-- Given conditions
axiom unit_vector_a : ‖a‖ = 1
axiom unit_vector_b : ‖b‖ = 1
axiom angle_ab : real.angle a b = π / 4

-- Define the volume of the parallelepiped
def volume_parallelepiped := |a ⬝ ((a + (b ⬝ a)) ⬝ b)|

-- The goal is to prove that this volume is 0
theorem volume_is_zero : volume_parallelepiped a b = 0 :=
sorry

end volume_is_zero_l545_545444


namespace quadratic_function_expression_l545_545862

-- Given conditions
def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

def intersects_A (a b c : ℝ) : Prop := f a b c (-2) = 0

def intersects_B (a b c : ℝ) : Prop := f a b c 4 = 0

def maximum_value (a b c : ℝ) : Prop := f a b c 1 = 9

-- Prove the function expression
theorem quadratic_function_expression :
  ∃ a b c : ℝ, intersects_A a b c ∧ intersects_B a b c ∧ maximum_value a b c ∧ 
  ∀ x : ℝ, f a b c x = -x^2 + 2 * x + 8 := 
sorry

end quadratic_function_expression_l545_545862


namespace work_problem_l545_545230

/--
Given:
1. A and B together can finish the work in 16 days.
2. B alone can finish the work in 48 days.
To Prove:
A alone can finish the work in 24 days.
-/
theorem work_problem (a b : ℕ)
  (h1 : a + b = 16)
  (h2 : b = 48) :
  a = 24 := 
sorry

end work_problem_l545_545230


namespace ellipse_semi_focal_distance_range_l545_545877

theorem ellipse_semi_focal_distance_range (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : b < a) (h_ellipse : a^2 = b^2 + c^2) :
  1 < (b + c) / a ∧ (b + c) / a ≤ Real.sqrt 2 := 
sorry

end ellipse_semi_focal_distance_range_l545_545877


namespace correct_statement_count_l545_545276

theorem correct_statement_count :
  (if (∀ x : ℝ, x^2 - 3 * x + 2 = 0 → x = 1) then true ↔ 1 = 1 else false) ∧
  (if (∀ x : ℝ, x ≠ 1 → x^2 - 3 * x + 2 ≠ 0) then true ↔ 1 = 1 else false) ∧
  (¬(∀ x : ℝ, x^2 = 1 → x = 1) ↔ (∀ x : ℝ, x^2 = 1 → x ≠ 1)) ∧
  (∀ x : ℝ, x > 2 → x^2 - 3 * x + 2 > 0) ∧ (∃ x : ℝ, ¬(x > 2) ∧ x^2 - 3 * x + 2 > 0) ∧
  (∀ P Q : Prop, ¬(P ∧ Q) → (P ∨ Q))
:= by
  sorry

end correct_statement_count_l545_545276


namespace circle_center_distance_travelled_l545_545745

theorem circle_center_distance_travelled :
  ∀ (r : ℝ) (a b c : ℝ), r = 2 ∧ a = 9 ∧ b = 12 ∧ c = 15 → (a^2 + b^2 = c^2) → 
  ∃ (d : ℝ), d = 24 :=
by
  intros r a b c h1 h2
  sorry

end circle_center_distance_travelled_l545_545745


namespace sum_of_distinct_roots_eq_zero_l545_545573

theorem sum_of_distinct_roots_eq_zero
  (a b m n p : ℝ)
  (h1 : m ≠ n)
  (h2 : m ≠ p)
  (h3 : n ≠ p)
  (h_m : m^3 + a * m + b = 0)
  (h_n : n^3 + a * n + b = 0)
  (h_p : p^3 + a * p + b = 0) : 
  m + n + p = 0 :=
sorry

end sum_of_distinct_roots_eq_zero_l545_545573


namespace math_proof_problem_l545_545959

-- Define the problem conditions
variables {A B C a b c : ℝ}

-- Condition: sides opposite to angles A, B, and C are a, b, and c respectively
-- and the given equation involving cosine
def problem_condition : Prop :=
  c = 2 * Real.sqrt 3 ∧
  Real.cos (2 * C) - 3 * Real.cos (A + B) = 1

-- Define the measure of angle C (question 1)
def measure_C : ℝ :=
  (π / 3)

-- Define the maximum value of the area S of triangle ABC (question 2)
def max_area_S : ℝ :=
  3 * Real.sqrt 3

-- Theorem statement verifying the correctness of the problem solution
theorem math_proof_problem :
  problem_condition → 
  (C = measure_C) ∧ (∃ S, S = max_area_S) :=
sorry

end math_proof_problem_l545_545959


namespace magic_shop_change_l545_545493

theorem magic_shop_change :
  (∀ (cloak : Type), ∃ price_gold price_silver1 change_gold1 price_silver2 change_gold2, 
  price_silver1 = 20 ∧ change_gold1 = 4 ∧ 
  price_silver2 = 15 ∧ change_gold2 = 1 ∧ 
  price_gold = 14 ∧ 
  ∀ change_silver, 
    (20 - 4) * change_silver = 15 - 1 → -- Relation derived from the conditions
    (14 - (15 - 1) * change_silver / (20 - 4)) * change_silver = 10) := 
sorry

end magic_shop_change_l545_545493


namespace gnomes_remaining_l545_545620

theorem gnomes_remaining (westerville_gnomes : ℕ) (westerville_gnomes_count : westerville_gnomes = 20)
    (ravenswood_multiplier : ℕ) (ravenswood_multiplier_count : ravenswood_multiplier = 4)
    (percentage_taken : ℝ) (percentage_taken_count : percentage_taken = 0.40) :
    let ravenswood_initial_gnomes := westerville_gnomes * ravenswood_multiplier in
    let gnomes_taken := (ravenswood_initial_gnomes : ℝ) * percentage_taken in
    let gnomes_remaining := ravenswood_initial_gnomes - gnomes_taken in
    gnomes_remaining = 48 := by
  sorry

end gnomes_remaining_l545_545620


namespace area_triangle_DEF_l545_545256

/-- A point Q is chosen inside triangle DEF such that lines 
drawn through Q, parallel to the sides of triangle DEF, 
divide it into three smaller triangles with areas 16, 
25, and 36 respectively. Prove that the area of 
triangle DEF is 77. -/
theorem area_triangle_DEF (Q : Point) (DEF : Triangle)
  (area_small_1 area_small_2 area_small_3 : ℝ)
  (h1 : area_small_1 = 16)
  (h2 : area_small_2 = 25)
  (h3 : area_small_3 = 36)
  (h_parallel_1 : LineThrough Q parallel DEF.side1)
  (h_parallel_2 : LineThrough Q parallel DEF.side2)
  (h_parallel_3 : LineThrough Q parallel DEF.side3) :
  triangleArea DEF = 77 := 
sorry

end area_triangle_DEF_l545_545256


namespace change_in_expression_is_correct_l545_545466

def change_in_expression (x a : ℝ) : ℝ :=
  if increases : true then (x + a)^2 - 3 - (x^2 - 3)
  else (x - a)^2 - 3 - (x^2 - 3)

theorem change_in_expression_is_correct (x a : ℝ) :
  a > 0 → change_in_expression x a = 2 * a * x + a^2 ∨ change_in_expression x a = -(2 * a * x) + a^2 :=
by
  sorry

end change_in_expression_is_correct_l545_545466


namespace cloak_change_14_gold_coins_l545_545514

def exchange_rate (silver gold : ℕ) : Prop :=
  ∃ c : ℕ, (20 - 4) * c = silver ∧ (15 - 1) * c = silver

def cloak_purchase (paid_gold received_silver : ℕ) : Prop :=
  let exchange_rate := (5 * 14) / 3 in
  received_silver = 2 * exchange_rate

theorem cloak_change_14_gold_coins :
  exchange_rate 16 3 →
  exchange_rate 14 1 →
  cloak_purchase 14 10 := sorry

end cloak_change_14_gold_coins_l545_545514


namespace count_isosceles_points_l545_545065

noncomputable def evenly_spaced_points := Set (ℤ × ℤ)

def is_isosceles (A B C : (ℤ × ℤ)) : Prop :=
  let dist (X Y : (ℤ × ℤ)) := (X.1 - Y.1)^2 + (X.2 - Y.2)^2
  dist A B = dist A C ∨ dist A B = dist B C ∨ dist A C = dist B C

def valid_points (A B : (ℤ × ℤ)) : Finset (ℤ × ℤ) := 
  let all_points := Finset.univ.attach.filter (λ P, P.1 ≠ A ∧ P.1 ≠ B)
  all_points.filter (λ C, is_isosceles A B C)

theorem count_isosceles_points (A B : (ℤ × ℤ)) :
  evenly_spaced_points → 
  valid_points A B.card = 6 := 
sorry

end count_isosceles_points_l545_545065


namespace ratio_equality_l545_545106

open EuclideanGeometry

variables {A B C C1 C2 A1 A2 B1 B2 : Point}

-- Assuming the necessary conditions
axiom triangle_ABC : Triangle A B C
axiom on_side_C1C2 : OnSegment A B C1 ∧ OnSegment A B C2
axiom on_side_A1A2 : OnSegment B C A1 ∧ OnSegment B C A2
axiom on_side_B1B2 : OnSegment A C B1 ∧ OnSegment A C B2
axiom equal_lengths : dist A1 B2 = dist B1 C2 ∧ dist B1 C2 = dist C1 A2
axiom intersect_one_point : ∃ P : Point, Collinear A1 B2 P ∧ Collinear B1 C2 P ∧ Collinear C1 A2 P
axiom angles_60_degrees : ∠ A1 B2 B1 C2 = 60 ∧ ∠ B1 C2 C1 A2 = 60 ∧ ∠ C1 A2 A1 B2 = 60

theorem ratio_equality :
  dist A1 A2 / dist B C = dist B1 B2 / dist A C ∧ dist B1 B2 / dist A C = dist C1 C2 / dist A B :=
sorry  -- Proof omitted.

end ratio_equality_l545_545106


namespace product_positive_probability_l545_545211

theorem product_positive_probability :
  let interval := set.Icc (-30 : ℝ) 15 in
  let prob_neg := (30 : ℝ) / 45 in
  let prob_pos := (15 : ℝ) / 45 in
  let prob_product_neg := 2 * (prob_neg * prob_pos) in
  let prob_product_pos := (prob_neg ^ 2) + (prob_pos ^ 2) in
  (prob_product_pos = 5 / 9) :=
by
  sorry

end product_positive_probability_l545_545211


namespace probability_of_multiple_of_45_l545_545237

def single_digit_multiples_of_3 := {3, 6, 9}
def prime_numbers_less_than_20 := {2, 3, 5, 7, 11, 13, 17, 19}

def product_is_multiple_of_45 (a b : ℕ) : Prop :=
  a * b % 45 = 0

def possible_pairs : ℕ :=
  single_digit_multiples_of_3.card * prime_numbers_less_than_20.card

def successful_pairs : ℕ :=
  (single_digit_multiples_of_3.product prime_numbers_less_than_20).count (λ p, product_is_multiple_of_45 p.1 p.2)

def probability_of_success : ℚ :=
  successful_pairs / possible_pairs

theorem probability_of_multiple_of_45 :
  probability_of_success = 1 / 24 := sorry

end probability_of_multiple_of_45_l545_545237


namespace min_value_of_reciprocal_sum_l545_545995

open Real

theorem min_value_of_reciprocal_sum (x y : ℝ) (h1 : x > 0) (h2 : y > 0)
  (h3 : x + y = 12) (h4 : x * y = 20) : (1 / x + 1 / y) = 3 / 5 :=
sorry

end min_value_of_reciprocal_sum_l545_545995


namespace exist_xy_set_l545_545062

variable {n : ℕ}
variable (a : Fin n -> Fin n -> ℝ)
variable (rook_sum : Fin n -> Fin n -> ℝ -> Prop)

/-- 
Given that there exists an arrangement of n rooks on an n × n chessboard,
    such that no two rooks can attack each other and the sum of the covered
    numbers equals 1972, prove that there exist two sets of numbers x₁, x₂, 
    ..., xₙ and y₁, y₂, ..., yₙ such that for all k, m in the range of n, 
    a_{km} = x_k + y_m.
--/
theorem exist_xy_set (h : ∀ arrangement, rook_sum arrangement a = 1972) :
    ∃ x y, (∀ k m, a k m = x k + y m) := 
sorry

end exist_xy_set_l545_545062


namespace base_area_of_hemisphere_is_3_l545_545169

-- Given conditions
def surface_area_of_sphere (r : ℝ) : ℝ := 4 * Real.pi * r^2
def surface_area_of_hemisphere : ℝ := 9

-- Required to prove: The base area of the hemisphere is 3
theorem base_area_of_hemisphere_is_3 (r : ℝ) (h : surface_area_of_hemisphere = 9) :
  π * r^2 = 3 :=
sorry

end base_area_of_hemisphere_is_3_l545_545169


namespace classify_quadrilateral_equal_perpendicular_diagonals_l545_545952

theorem classify_quadrilateral_equal_perpendicular_diagonals
  (Q : Type) [quadrilateral Q]
  (h1 : diagonals Q equal_length)
  (h2 : diagonals Q perpendicular) :
  is_square Q :=
sorry

end classify_quadrilateral_equal_perpendicular_diagonals_l545_545952


namespace sequence_inequality_l545_545579
open Nat

variable (a : ℕ → ℝ)

noncomputable def conditions := 
  (a 1 ≥ 1) ∧ (∀ k : ℕ, a (k + 1) - a k ≥ 1)

theorem sequence_inequality (h : conditions a) : 
  ∀ n : ℕ, a (n + 1) ≥ n + 1 :=
sorry

end sequence_inequality_l545_545579


namespace count_two_digit_multiples_of_seven_l545_545022

-- Define the conditions based on the problem
def is_multiple_of_seven (n : ℕ) : Prop := n % 7 = 0
def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

-- The main statement to be proven
theorem count_two_digit_multiples_of_seven : 
  (finset.filter (λ n, is_multiple_of_seven n) (finset.Icc 10 99)).card = 13 := 
sorry

end count_two_digit_multiples_of_seven_l545_545022


namespace product_positive_probability_l545_545209

theorem product_positive_probability :
  let interval := set.Icc (-30 : ℝ) 15 in
  let prob_neg := (30 : ℝ) / 45 in
  let prob_pos := (15 : ℝ) / 45 in
  let prob_product_neg := 2 * (prob_neg * prob_pos) in
  let prob_product_pos := (prob_neg ^ 2) + (prob_pos ^ 2) in
  (prob_product_pos = 5 / 9) :=
by
  sorry

end product_positive_probability_l545_545209


namespace pradeep_marks_l545_545603

-- Conditions as definitions
def passing_percentage : ℝ := 0.35
def max_marks : ℕ := 600
def fail_difference : ℕ := 25

def passing_marks (total_marks : ℕ) (percentage : ℝ) : ℝ :=
  percentage * total_marks

def obtained_marks (passing_marks : ℝ) (difference : ℕ) : ℝ :=
  passing_marks - difference

-- Theorem statement
theorem pradeep_marks : obtained_marks (passing_marks max_marks passing_percentage) fail_difference = 185 := by
  sorry

end pradeep_marks_l545_545603


namespace conditional_probability_B_given_A_l545_545481

open Probability

noncomputable def problem := 
  let μ : ℝ := 100
  let σ : ℝ := 10
  let X : MeasureTheory.Measure ℝ := MeasureTheory.Measure.normal μ σ
  let A : Set ℝ := {x | 80 < x ∧ x ≤ 100}
  let B : Set ℝ := {x | 70 < x ∧ x ≤ 90}
  P(X|A).prob B = 27 / 95

theorem conditional_probability_B_given_A : problem :=
  begin
    sorry
  end

end conditional_probability_B_given_A_l545_545481


namespace square_side_length_l545_545954

theorem square_side_length (x : ℝ) (h : x ^ 2 = 4 * 3) : x = 2 * Real.sqrt 3 :=
by sorry

end square_side_length_l545_545954


namespace mean_of_all_students_l545_545593

variable (M A m a : ℕ)
variable (M_val : M = 84)
variable (A_val : A = 70)
variable (ratio : m = 3 * a / 4)

theorem mean_of_all_students (M A m a : ℕ) (M_val : M = 84) (A_val : A = 70) (ratio : m = 3 * a / 4) :
    (63 * a + 70 * a) / (7 * a / 4) = 76 := by
  sorry

end mean_of_all_students_l545_545593


namespace probability_bottom_vertex_l545_545957

def dodecahedron : Type := sorry

def is_top_vertex (v : dodecahedron) : Prop := sorry
def is_bottom_vertex (v : dodecahedron) : Prop := sorry
def adjacent_vertices (v : dodecahedron) : finset dodecahedron := sorry
def random_walk (v : dodecahedron) : dodecahedron := sorry

theorem probability_bottom_vertex
    (start_vertex : dodecahedron)
    (h_top : is_top_vertex start_vertex) :
    (finset.filter is_bottom_vertex (adjacent_vertices (random_walk (random_walk start_vertex)))).card.toR / 
    (adjacent_vertices (random_walk (random_walk start_vertex))).card.toR = 1 / 3 := sorry

end probability_bottom_vertex_l545_545957


namespace total_shaded_area_l545_545213

structure Rectangle where
  length : ℝ
  width : ℝ

structure RightTriangle where
  leg1 : ℝ
  leg2 : ℝ

def area_rectangle (r : Rectangle) : ℝ :=
  r.length * r.width

def area_triangle (t : RightTriangle) : ℝ :=
  (t.leg1 * t.leg2) / 2

def area_overlap : ℝ :=
  4 * 5

def overlap_correction_factor : ℝ :=
  0.5 * (3 * 3 / 2)

theorem total_shaded_area :
  let r1 := Rectangle.mk 4 12
  let r2 := Rectangle.mk 5 9
  let t := RightTriangle.mk 3 3
  area_rectangle r1 + area_rectangle r2 - area_overlap - overlap_correction_factor = 70.75 := by
  sorry

end total_shaded_area_l545_545213


namespace count_desired_property_l545_545929

-- Define the property that a number is a three-digit positive integer
def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000

-- Define the property that a number contains at least one specific digit
def contains_digit (d : ℕ) (n : ℕ) : Prop :=
  list.any (nat.digits 10 n) (λ m, m = d)

-- Define the property that a number does not contain a specific digit
def does_not_contain_digit (d : ℕ) (n : ℕ) : Prop :=
  ¬ list.any (nat.digits 10 n) (λ m, m = d)

-- Define the overall property for the desired number
def desired_property (n : ℕ) : Prop :=
  is_three_digit n ∧ contains_digit 4 n ∧ does_not_contain_digit 6 n

-- State the theorem to prove the count of numbers with the desired property
theorem count_desired_property : finset.card (finset.filter desired_property (finset.range 1000)) = 200 := by
  sorry

end count_desired_property_l545_545929


namespace sum_of_sequences_l545_545575

theorem sum_of_sequences :
  let (a_n b_n : ℕ → ℝ) := (
    λ n: ℕ, ((3:ℂ) + 4*complex.i)^n.re,
    λ n: ℕ, ((3:ℂ) + 4*complex.i)^n.im
  ) in
  (∑' n, (a_n n * b_n n) / 8^n) = 5/8 :=
by
  sorry

end sum_of_sequences_l545_545575


namespace range_of_a_l545_545888

-- Definitions of the curve C1 and the line l, and their corresponding points.
def curveC1 (x y : ℝ) : Prop := ((x - 2) ^ 2) / 2 + y ^ 2 = 1
def lineL (a x : ℝ) : Prop := x = -a - 1 / 8

-- Condition for four points equidistant from a point and a line
def equidistantPoints (a x y : ℝ) : Prop := y ^ 2 = (1 / 2) * (x + a)

-- Proving the range of values for a
theorem range_of_a (a : ℝ) :
    (∀ (x y : ℝ),
        curveC1 x y →
        (∃ (x' y' : ℝ),
            curveC1 x' y' ∧
            equidistantPoints a x' y')
    ) →
    (a > real.sqrt 2 / 2 ∧ a < 9 / 8) :=
begin
  sorry
end

end range_of_a_l545_545888


namespace cloak_change_l545_545501

theorem cloak_change (silver_for_cloak1 silver_for_cloak2 : ℕ) (gold_change1 gold_change2 : ℕ) (silver_change : ℕ) :
  silver_for_cloak1 = 20 →
  gold_change1 = 4 →
  silver_for_cloak2 = 15 →
  gold_change2 = 1 →
  ∃ silver_cost_of_cloak : ℕ, 
    silver_cost_of_cloak = (20 - 4) * (5 / 3) →
    silver_change = 10 →
    14 * (5 / 3) - 8 = silver_change :=
by 
  assume h1 h2 h3 h4,
  use 16, 
  sorry

end cloak_change_l545_545501


namespace wombat_clawing_l545_545294

variable (W : ℕ)
variable (R : ℕ := 1)

theorem wombat_clawing :
    (9 * W + 3 * R = 39) → (W = 4) :=
by 
  sorry

end wombat_clawing_l545_545294


namespace termites_count_l545_545775

theorem termites_count (total_workers monkeys : ℕ) (h1 : total_workers = 861) (h2 : monkeys = 239) : total_workers - monkeys = 622 :=
by
  -- The proof steps will go here
  sorry

end termites_count_l545_545775


namespace sum_of_coordinates_after_reflections_l545_545109

def point := (ℝ × ℝ)
def midpoint (A B : point) : point :=
  ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

def reflect_over_x (P : point) : point :=
  (P.1, -P.2)

def reflect_over_y (P : point) : point :=
  (-P.1, P.2)

def sum_of_coordinates (P : point) : ℝ :=
  P.1 + P.2

theorem sum_of_coordinates_after_reflections :
  let A : point := (3, 2)
  let B : point := (15, 18)
  let N : point := midpoint A B
  let N' : point := reflect_over_x N
  let N'' : point := reflect_over_y N'
  sum_of_coordinates N'' = -19 :=
by
  let A : point := (3, 2)
  let B : point := (15, 18)
  let N : point := midpoint A B
  let N' : point := reflect_over_x N
  let N'' : point := reflect_over_y N'
  show sum_of_coordinates N'' = -19
  sorry

end sum_of_coordinates_after_reflections_l545_545109


namespace min_value_of_sum_squares_l545_545097

noncomputable def min_value_sum_squares 
  (y1 y2 y3 : ℝ) (h1 : 2 * y1 + 3 * y2 + 4 * y3 = 120) 
  (h2 : 0 < y1) (h3 : 0 < y2) (h4 : 0 < y3) : ℝ :=
  y1^2 + y2^2 + y3^2

theorem min_value_of_sum_squares 
  (y1 y2 y3 : ℝ) (h1 : 2 * y1 + 3 * y2 + 4 * y3 = 120) 
  (h2 : 0 < y1) (h3 : 0 < y2) (h4 : 0 < y3) : 
  min_value_sum_squares y1 y2 y3 h1 h2 h3 h4 = 14400 / 29 := 
sorry

end min_value_of_sum_squares_l545_545097


namespace proof_problem_l545_545001

-- Given conditions and solution targets
def omega (ω : ℝ) : Prop :=
  (∃ π > 0, ω * (π / 2) = π) → ω = 2

def domain (f : ℝ → ℝ) : Prop :=
  ∀ x k : ℤ, f x ≠ ∞ → x ≠ (1/2 : ℝ) * k + (π / 8)

def tan_2alpha (α : ℝ) : Prop :=
  (∃ f : ℝ → ℝ, f (α / 2) = 3) → Real.tan (2 * α) = 4 / 3

-- Complete statement combining all
theorem proof_problem (ω : ℝ) (α : ℝ) (f : ℝ → ℝ) : 
  omega ω ∧ domain f ∧ tan_2alpha α :=
by { sorry }

end proof_problem_l545_545001


namespace tenth_term_l545_545764

noncomputable def sequence_term (n : ℕ) : ℝ :=
  (-1)^(n+1) * (Real.sqrt (1 + 2*(n - 1))) / (2^n)

theorem tenth_term :
  sequence_term 10 = Real.sqrt 19 / (2^10) :=
by
  sorry

end tenth_term_l545_545764


namespace min_value_of_f_l545_545340

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 2 * x^3 - 6 * x^2 + a

theorem min_value_of_f :
  (∀ x ∈ set.Icc (-2 : ℝ) 2, 2 * x^3 - 6 * x^2 + (3 : ℝ) ≤ 3) →
  (∃ x ∈ set.Icc (-2 : ℝ) 2, ∀ y ∈ set.Icc (-2 : ℝ) 2, f 3 y ≥ f 3 x) →
  ∃ x ∈ set.Icc (-2 : ℝ) 2, f 3 x = -37 :=
begin
  sorry
end

end min_value_of_f_l545_545340


namespace sin_cos_equiv_l545_545361

theorem sin_cos_equiv (θ : ℝ) (h : cos (2 * θ) = (Real.sqrt 2) / 3) : 
  (sin θ)^4 - (cos θ)^4 = - (Real.sqrt 2) / 3 := 
by
  sorry

end sin_cos_equiv_l545_545361


namespace monotonic_intervals_inequality_condition_l545_545895

noncomputable def f (x : ℝ) (m : ℝ) := Real.log x - m * x

theorem monotonic_intervals (m : ℝ) :
  (m ≤ 0 → ∀ x > 0, ∀ y > 0, x < y → f x m < f y m) ∧
  (m > 0 → (∀ x > 0, x < 1/m → ∀ y > x, y < 1/m → f x m < f y m) ∧ (∀ x ≥ 1/m, ∀ y > x, f x m > f y m)) :=
sorry

theorem inequality_condition (m : ℝ) (h : ∀ x ≥ 1, f x m ≤ (m - 1) / x - 2 * m + 1) :
  m ≥ 1/2 :=
sorry

end monotonic_intervals_inequality_condition_l545_545895


namespace trapezoid_area_l545_545783

theorem trapezoid_area (n : ℕ) (h : n = 1): 
  let area_triangle := 1 in 
  let total_area := 2 * n * area_triangle in 
  total_area = 2 := 
by 
  simp [area_triangle, total_area]
  rw ←h
  simp
  sorry

end trapezoid_area_l545_545783


namespace custom_op_neg2_neg3_l545_545758

  def custom_op (a b : ℤ) : ℤ := b^2 - a

  theorem custom_op_neg2_neg3 : custom_op (-2) (-3) = 11 :=
  by
    sorry
  
end custom_op_neg2_neg3_l545_545758


namespace number_of_divisors_of_500_l545_545072

theorem number_of_divisors_of_500 :
  {n : ℕ | 1 ≤ n ∧ n ≤ 500 ∧ 500 % n = 0}.card = 12 :=
sorry

end number_of_divisors_of_500_l545_545072


namespace polynomial_coeff_sum_is_15625_l545_545459

noncomputable def polynomial_sum_coeff : ℕ :=
  let p := (2 * (Polynomial.X : Polynomial ℤ) + 3)^6
  p.eval 1

theorem polynomial_coeff_sum_is_15625 : polynomial_sum_coeff = 15625 := by
  sorry

end polynomial_coeff_sum_is_15625_l545_545459


namespace cloak_change_in_silver_l545_545488

theorem cloak_change_in_silver :
  (∀ c : ℤ, (20 = c + 4) → (15 = c + 1)) →
  (5 * g = 3) →
  14 * gold / exchange_rate = 10 := 
sorry

end cloak_change_in_silver_l545_545488


namespace u_quadratic_equation_v_quartic_equation_l545_545604

def angle_R : ℝ := 90    -- R is 90 degrees

def u : ℝ := Real.cot (angle_R / 4)  -- u = cot(R/4)
def v : ℝ := 1 / Real.sin (angle_R / 4)  -- v = 1 / sin(R/4)

theorem u_quadratic_equation : u^2 - 2 * u - 1 = 0 := 
sorry

theorem v_quartic_equation : v^4 - 8 * v^2 + 8 = 0 := 
sorry

end u_quadratic_equation_v_quartic_equation_l545_545604


namespace smallest_k_for_inequality_l545_545843

theorem smallest_k_for_inequality (k : ℝ) :
  (∀ (n : ℕ), n ≥ 2 →
    ∀ (z : Fin n → ℂ), 
    (∀ i, (z i).re ≥ 0 ∧ (z i).im ≥ 0) →
    abs (Finset.univ.sum (λ i, z i)) ≥ (1 / k) * Finset.univ.sum (λ i, abs (z i))
  ) ↔ k = Real.sqrt 2 :=
begin
  sorry
end

end smallest_k_for_inequality_l545_545843


namespace complex_number_z_range_of_m_l545_545368

open Complex

theorem complex_number_z (z : ℂ) (H1 : ∀ w : ℂ, z + 2 * I = w → w.im = 0) (H2 : ∀ w : ℂ, z / (2 - I) = w → w.im = 0) : z = 4 - 2 * I :=
sorry

theorem range_of_m (m : ℝ) (z : ℂ) (H1 : ∀ w : ℂ, z + 2 * I = w → w.im = 0) (H2 : ∀ w : ℂ, z / (2 - I) = w → w.im = 0) (H3 : z = 4 - 2 * I) (z1 : ℂ) (H4 : z1 = conj z + (1 / (m - 1)) - (7 / (m + 2)) * I) : 
    (z1.re > 0 ∧ z1.im < 0) → m ∈ Set.Ioo (-2) (3 / 4) ∪ Set.Ioo (1) (3 / 2) :=
sorry

end complex_number_z_range_of_m_l545_545368


namespace cube_side_length_l545_545964

-- Given definitions and conditions
variables (a b c : ℝ) (a_pos : 0 < a) (b_pos : 0 < b) (c_pos : 0 < c)

-- Statement of the theorem
theorem cube_side_length (x : ℝ) : 
  ( ∃ (y z : ℝ), 
      y + x + z = c ∧ 
      x + z = c * a / b ∧
      y = c * x / b ∧
      z = c * x / a 
  ) → x = a * b * c / (a * b + b * c + c * a) :=
sorry

end cube_side_length_l545_545964


namespace largest_integral_x_l545_545319

theorem largest_integral_x (x y : ℤ) (h1 : (1 : ℚ)/4 < x/7) (h2 : x/7 < (2 : ℚ)/3) (h3 : x + y = 10) : x = 4 :=
by
  sorry

end largest_integral_x_l545_545319


namespace perpendicular_lines_parallel_lines_l545_545873

-- Define the given lines
def l1 (m : ℝ) (x y : ℝ) : ℝ := (m-2)*x + 3*y + 2*m
def l2 (m x y : ℝ) : ℝ := x + m*y + 6

-- The slope conditions for the lines to be perpendicular
def slopes_perpendicular (m : ℝ) : Prop :=
  (m - 2) * m = 3

-- The slope conditions for the lines to be parallel
def slopes_parallel (m : ℝ) : Prop :=
  m = -1

-- Perpendicular lines proof statement
theorem perpendicular_lines (m : ℝ) (x y : ℝ)
  (h1 : l1 m x y = 0)
  (h2 : l2 m x y = 0) :
  slopes_perpendicular m :=
sorry

-- Parallel lines proof statement
theorem parallel_lines (m : ℝ) (x y : ℝ)
  (h1 : l1 m x y = 0)
  (h2 : l2 m x y = 0) :
  slopes_parallel m :=
sorry

end perpendicular_lines_parallel_lines_l545_545873


namespace problem1_solution_problem2_solution_l545_545617

theorem problem1_solution (x : ℝ) (h1 : x ≠ 0) (h2 : x ≠ 2) : 
  (3 / x - 2 / (x - 2) = 0) ↔ (x = 6) :=
begin
  sorry
end

theorem problem2_solution (x : ℝ) (h1 : x ≠ 4) : 
  ¬(3 / (4 - x) + 2 = (1 - x) / (x - 4)) :=
begin
  sorry
end

end problem1_solution_problem2_solution_l545_545617


namespace equal_angles_in_isosceles_triangle_l545_545546

variable {A B C X N M T : Type} [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited X] [Inhabited N] [Inhabited M] [Inhabited T]

-- Conditions setup
variables (isosceles_ABC : ∃ (AC BC : ℝ), AC = BC)
          (point_X_on_AB : ∃ (X : A), true)
          (line_through_X_parallel_BC : ∃ (N : A), true)
          (line_through_X_parallel_AC : ∃ (M : B), true)
          (circle_k1_center_N_radius_NA : ∃ (k1 : {N // true} → ℝ), true)
          (circle_k2_center_M_radius_MB : ∃ (k2 : {M // true} → ℝ), true)
          (intersection_T_k1_k2 : T)

-- Define angles
variables (angle_NCM angle_NTM : ℝ)

-- Lean statement of the proof problem
theorem equal_angles_in_isosceles_triangle
  (isosceles_ABC : ∃ (AC BC : ℝ), AC = BC)
  (point_X_on_AB : ∃ (X : A), true)
  (line_through_X_parallel_BC : ∃ (N : A), true)
  (line_through_X_parallel_AC : ∃ (M : B), true)
  (circle_k1_center_N_radius_NA : ∃ (k1 : {N // true} → ℝ), true)
  (circle_k2_center_M_radius_MB : ∃ (k2 : {M // true} → ℝ), true)
  (intersection_T_k1_k2 : T)
  (angle_NCM angle_NTM : ℝ) :
  angle_NCM = angle_NTM :=
by
  sorry

end equal_angles_in_isosceles_triangle_l545_545546


namespace minimum_circle_area_l545_545891

noncomputable def f (x : ℝ) : ℝ :=
  1 + x - (x^2) / 2 + (x^3) / 3 - (x^4) / 4 + ∑ i in (finset.range 2013).filter (λ n, ¬(n = 0 ∨ n = 1 ∨ n = 2 ∨ n = 3)), (-1)^i * x^(i+1) / (i+1)

def F (x : ℝ) : ℝ := f (x + 4)

theorem minimum_circle_area (a b : ℤ) (hab : a < b) (hF_zero : ∀ x, F x = 0 → (a : ℝ) ≤ x ∧ x ≤ b) :
  ∃ r : ℝ, r = 1 ∧ π * r^2 = π :=
by
  sorry

end minimum_circle_area_l545_545891


namespace hyogeun_weight_l545_545437

noncomputable def weights_are_correct : Prop :=
  ∃ H S G : ℝ, 
    H + S + G = 106.6 ∧
    G = S - 7.7 ∧
    S = H - 4.8 ∧
    H = 41.3

theorem hyogeun_weight : weights_are_correct :=
by
  sorry

end hyogeun_weight_l545_545437


namespace sum_of_digits_divisible_by_18_l545_545120

theorem sum_of_digits_divisible_by_18 (N : ℤ) (h : (N / 10000 < 1 ∧ 99 ∣ N)) : 18 ∣ digit_sum N := 
sorry

-- Here, the function digit_sum should be defined to compute the sum of the digits of N.
def digit_sum (N : ℤ) : ℤ := 
let digits := N.digits 10 in  -- Convert N to its base 10 digits
digits.sum  -- Sum the list of digits

-- N.digits is assumed to be a function that returns the list of base-10 digits of N in Mathlib.

end sum_of_digits_divisible_by_18_l545_545120


namespace cloak_change_in_silver_l545_545482

theorem cloak_change_in_silver :
  (∀ c : ℤ, (20 = c + 4) → (15 = c + 1)) →
  (5 * g = 3) →
  14 * gold / exchange_rate = 10 := 
sorry

end cloak_change_in_silver_l545_545482


namespace _l545_545597

variables {n : ℕ} (O : Point) (A : Fin n → Point) (X : Point)

def divides_circle_equally (A : Fin n → Point) : Prop :=
  ∀ i : Fin n, distance O (A i) = distance O (A 0) ∧
               ∀ j : Fin n, angle O (A i) (A j) = (2 * i * Real.pi) / n

noncomputable theorem symmetric_points_form_regular_polygon
  (h1 : divides_circle_equally O A)
  (h2 : ∀ i : Fin n, symmetric_point_on_line O (A i) X ≠ X) :
  ∃ B : Fin n → Point, regular_polygon O B :=
sorry

end _l545_545597


namespace count_four_digit_numbers_divisible_by_five_ending_45_l545_545428

-- Define the conditions as necessary in Lean
def is_four_digit_number (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

def is_divisible_by_five (n : ℕ) : Prop :=
  n % 5 = 0

def ends_with_45 (n : ℕ) : Prop :=
  n % 100 = 45

-- Statement that there exists 90 such four-digit numbers
theorem count_four_digit_numbers_divisible_by_five_ending_45 : 
  { n : ℕ // is_four_digit_number n ∧ is_divisible_by_five n ∧ ends_with_45 n }.card = 90 :=
sorry

end count_four_digit_numbers_divisible_by_five_ending_45_l545_545428


namespace apps_more_than_files_l545_545823

theorem apps_more_than_files
  (initial_apps : ℕ)
  (initial_files : ℕ)
  (deleted_apps : ℕ)
  (deleted_files : ℕ)
  (remaining_apps : ℕ)
  (remaining_files : ℕ)
  (h1 : initial_apps - deleted_apps = remaining_apps)
  (h2 : initial_files - deleted_files = remaining_files)
  (h3 : initial_apps = 24)
  (h4 : initial_files = 9)
  (h5 : remaining_apps = 12)
  (h6 : remaining_files = 5) :
  remaining_apps - remaining_files = 7 :=
by {
  sorry
}

end apps_more_than_files_l545_545823


namespace relationship_mu_d_M_l545_545518

def data_occurrences : List ℕ :=
  List.replicate 30 1 ++ List.replicate 30 2 ++ List.replicate 30 3 ++ 
  List.replicate 30 4 ++ List.replicate 30 5 ++ List.replicate 30 6 ++ 
  List.replicate 30 7 ++ List.replicate 30 8 ++ List.replicate 30 9 ++ 
  List.replicate 30 10 ++ List.replicate 30 11 ++ List.replicate 30 12 ++ 
  List.replicate 30 13 ++ List.replicate 30 14 ++ List.replicate 30 15 ++ 
  List.replicate 30 16 ++ List.replicate 30 17 ++ List.replicate 30 18 ++ 
  List.replicate 30 19 ++ List.replicate 30 20 ++ List.replicate 30 21 ++ 
  List.replicate 30 22 ++ List.replicate 30 23 ++ List.replicate 30 24 ++ 
  List.replicate 30 25 ++ List.replicate 27 26 ++ List.replicate 27 27 ++ 
  List.replicate 27 28 ++ List.replicate 21 29 ++ List.replicate 21 30 ++ 
  List.replicate 21 31

noncomputable def mean (lst : List ℕ) : ℝ :=
  (lst.map (λ x, x.toNat)).sum / lst.length

noncomputable def median (lst : List ℕ) : ℕ :=
  let sorted := lst.qsort (≤)
  sorted.getD (sorted.length / 2) 0

noncomputable def modes_median (lst : List ℕ) : ℕ :=
  let modes := List.range 25
  let median_mode := modes.nth' (modes.length / 2)
  median_mode.getD 0

theorem relationship_mu_d_M :
  let μ := mean data_occurrences
  let M := median data_occurrences
  let d := modes_median data_occurrences
  M < d ∧ d < μ := by
  sorry

end relationship_mu_d_M_l545_545518


namespace number_of_elements_in_B_is_4_l545_545010

def A : Set ℕ := {x | x^2 + 2*x - 3 ≤ 0}
def B : Set (Set ℕ) := {C | C ⊆ A}

theorem number_of_elements_in_B_is_4 : B.to_finset.card = 4 := by
  sorry

end number_of_elements_in_B_is_4_l545_545010


namespace magnitude_w_one_l545_545982

def z : ℂ := ((-15 + 8*complex.I)^4 * (17 - 9*complex.I)^5) / (5 + 12*complex.I)

def w : ℂ := complex.conj(z) / z

theorem magnitude_w_one : complex.abs w = 1 := by
  sorry

end magnitude_w_one_l545_545982


namespace samantha_bus_time_l545_545611

def total_time_away (leave_time arrive_time : Nat) : Nat :=
  7 * 60 + 15 -- 7:15 am in minutes
  + 10 * 60  -- 10 hours converted to minutes 
  + 5 * 60 + 15 -- 5:15 pm in minutes

def total_school_activities (classes : Nat) (class_duration : Nat) (lunch_break : Nat) (extracurricular : Nat) : Nat :=
  classes * class_duration 
  + lunch_break 
  + extracurricular
 
def bus_time (total_time_away : Nat) (school_activities : Nat) : Nat :=
  total_time_away - school_activities

theorem samantha_bus_time :
  let leave_time := 7 * 60 + 15, -- 7:15 am
  let arrive_time := 17 * 60 + 15, -- 5:15 pm
  let classes := 8,
  let class_duration := 45,
  let lunch_break := 40,
  let extracurricular := 90,
  let total_time_away := total_time_away leave_time arrive_time,
  let total_school_activities := total_school_activities classes class_duration lunch_break extracurricular,
  bus_time total_time_away total_school_activities = 110 :=
by
  sorry

end samantha_bus_time_l545_545611


namespace passes_through_orthocenter_iff_l545_545075

-- Definitions of the mathematical objects in Lean
structure Triangle := (A B C : Point)
structure Line := (p1 p2 : Point)

def is_circumcenter (O : Point) (T : Triangle) : Prop := sorry
def is_parallel_to_angle_bisector (L : Line) (T : Triangle) : Prop := sorry
def is_orthocenter (H : Point) (T : Triangle) : Prop := sorry
def on_line (P : Point) (L : Line) : Prop := sorry
def angle (A B C : Point) : Real := sorry
def length (A B : Point) : Real := sorry

-- Given conditions in the problem
variables (A B C O H : Point)
variables (L : Line)
variable (T: Triangle := Triangle.mk A B C)

-- Main statement
theorem passes_through_orthocenter_iff :
  (is_circumcenter O T ∧ is_parallel_to_angle_bisector L T ∧ on_line O L) →
  (on_line H L ↔ (length A B = length A C ∨ angle A B C = 120)) :=
by
  intros
  sorry

end passes_through_orthocenter_iff_l545_545075


namespace distinctEquilateralTriangles_l545_545868

-- Define the eleven-sided regular polygon
structure RegularPolygon (n : ℕ) :=
(vertices : Finset ℝ) -- assume vertices are placed on plane with coordinates in ℝ

noncomputable def elevenSidedPolygon : RegularPolygon 11 := 
{ vertices := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11} -- assuming vertices are numbered from 1 to 11 }

-- Define a function to count the distinct equilateral triangles
noncomputable def countEquilateralTriangles (poly : RegularPolygon 11) : ℕ :=
sorry -- implementation is a complex mathematical calculation

-- Prove that the number of distinct equilateral triangles is 104
theorem distinctEquilateralTriangles : countEquilateralTriangles elevenSidedPolygon = 104 :=
sorry

end distinctEquilateralTriangles_l545_545868


namespace silver_coins_change_l545_545509

-- Define the conditions
def condition1 : ℕ × ℕ := (20, 4) -- (20 silver coins, 4 gold coins change)
def condition2 : ℕ × ℕ := (15, 1) -- (15 silver coins, 1 gold coin change)
def cost_of_cloak_in_gold_coins : ℕ := 14

-- Define the theorem to be proven
theorem silver_coins_change (s1 g1 s2 g2 cloak_g : ℕ) (h1 : (s1, g1) = condition1) (h2 : (s2, g2) = condition2) :
  ∃ silver : ℕ, (silver = 10) :=
by {
  sorry
}

end silver_coins_change_l545_545509


namespace ivan_total_pay_l545_545055

theorem ivan_total_pay (cost_per_card : ℕ) (number_of_cards : ℕ) (discount_per_card : ℕ) :
  cost_per_card = 12 → number_of_cards = 10 → discount_per_card = 2 →
  (number_of_cards * (cost_per_card - discount_per_card)) = 100 :=
by
  intro h1 h2 h3
  sorry

end ivan_total_pay_l545_545055


namespace sam_walked_distance_l545_545725

theorem sam_walked_distance
  (distance_apart : ℝ) (fred_speed : ℝ) (sam_speed : ℝ) (t : ℝ)
  (H1 : distance_apart = 35) (H2 : fred_speed = 2) (H3 : sam_speed = 5)
  (H4 : 2 * t + 5 * t = distance_apart) :
  5 * t = 25 :=
by
  -- Lean proof goes here
  sorry

end sam_walked_distance_l545_545725


namespace tom_gave_8_boxes_l545_545189

-- Define the given conditions and the question in terms of variables
variables (total_boxes : ℕ) (pieces_per_box : ℕ) (pieces_left : ℕ) (boxes_given : ℕ)

-- Specify the actual values for the given problem
def tom_initial_pieces := total_boxes * pieces_per_box
def pieces_given := tom_initial_pieces - pieces_left
def calculated_boxes_given := pieces_given / pieces_per_box

-- Prove the number of boxes Tom gave to his little brother
theorem tom_gave_8_boxes
  (h1 : total_boxes = 14)
  (h2 : pieces_per_box = 3)
  (h3 : pieces_left = 18)
  (h4 : calculated_boxes_given = boxes_given) :
  boxes_given = 8 :=
by
  sorry

end tom_gave_8_boxes_l545_545189


namespace AE_div_BC_l545_545543

noncomputable def point := ℝ × ℝ

variables (A B C D E : point)
variables (s : ℝ)

-- Definitions for distances
def distance (p q : point) : ℝ := real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Conditions for equilateral triangles
def equilateral (p q r : point) : Prop :=
  (distance p q = distance q r) ∧ (distance q r = distance r p)

-- Midpoint definition
def is_midpoint (m p q : point) : Prop :=
  (m.1 = (p.1 + q.1) / 2) ∧ (m.2 = (p.2 + q.2) / 2)

-- Given conditions as hypotheses
axiom ABC_equilateral : equilateral A B C
axiom BCD_equilateral : equilateral B C D
axiom CDE_equilateral : equilateral C D E
axiom E_is_midpoint : is_midpoint E C D

-- Define the points A, B, C, D, E
variables (Ax Ay Bx By Cx Cy Dx Dy Ex Ey : ℝ)
variables (s : ℝ)

-- Hypotheses with specific coordinates
hypothesis hA : A = (Ax, Ay)
hypothesis hB : B = (Bx, By)
hypothesis hC : C = (Cx, Cy)
hypothesis hD : D = (Dx, Dy)
hypothesis hE : E = (Ex, Ey)

-- Hypotheses regarding points and distances
hypothesis h1 : distance B C = s
hypothesis h2 : distance C D = s
hypothesis h3 : distance D E = s / 2
hypothesis h4 : distance C E = s / 2

-- Final theorem to prove
theorem AE_div_BC :
  distance A E / distance B C = real.sqrt 3 :=
by sorry

end AE_div_BC_l545_545543


namespace count_bases_for_perfect_square_l545_545434

/-- 
Theorem: The number of integers \( n \) such that \( 4 \leq n \leq 12 \) 
and \( 144_n \) (the number written as \( 144 \) in base \( n \)) 
is a perfect square is 8.
-/
theorem count_bases_for_perfect_square : 
  (finset.card (finset.filter (λ n, (n ≥ 4 ∧ n ≤ 12) ∧ (n^2 + 4 * n + 4 = (n + 2) * (n + 2))) 
    (finset.range (12 + 1)))) = 8 := 
begin
  -- the proof will be skipped as it is not required
  sorry
end

end count_bases_for_perfect_square_l545_545434


namespace system_of_equations_solution_l545_545126

theorem system_of_equations_solution :
  ∃ (x1 x2 x3 x4 x5 : ℝ),
  (x1 + 2 * x2 + 2 * x3 + 2 * x4 + 2 * x5 = 1) ∧
  (x1 + 3 * x2 + 4 * x3 + 4 * x4 + 4 * x5 = 2) ∧
  (x1 + 3 * x2 + 5 * x3 + 6 * x4 + 6 * x5 = 3) ∧
  (x1 + 3 * x2 + 5 * x3 + 7 * x4 + 8 * x5 = 4) ∧
  (x1 + 3 * x2 + 5 * x3 + 7 * x4 + 9 * x5 = 5) ∧
  (x1 = 1) ∧ (x2 = -1) ∧ (x3 = 1) ∧ (x4 = -1) ∧ (x5 = 1) := by
sorry

end system_of_equations_solution_l545_545126


namespace inequality_range_l545_545448

theorem inequality_range (y : ℝ) (b : ℝ) (hb : 0 < b) : (|y-5| + 2 * |y-2| > b) ↔ (b < 3) := 
sorry

end inequality_range_l545_545448


namespace measure_45_seconds_using_fuses_l545_545015

theorem measure_45_seconds_using_fuses :
  ∃ (f1 f2 : ℕ → bool), (∀ t, t > 60 → ¬f1 t ∧ ¬f2 t) ∧
  (∃ t1, t1 < 120 ∧ f1 t1 ∧ ¬f1 (t1 + 0) ∧ ∃ t2, t2 < 60 ∧ f2 t2 ∧ ∃ t3, t3 < 15 ∧ f2 (t2 + t3)) :=
begin
  sorry
end

end measure_45_seconds_using_fuses_l545_545015


namespace maximum_excellent_films_l545_545469

noncomputable def max_excellent_films (views scores : Fin 5 → ℕ) : ℕ :=
  let not_inferior (i j : Fin 5) : Prop := (views i > views j) ∨ (scores i > scores j)
  let is_excellent (i : Fin 5) : Prop := ∀ j : Fin 5, i ≠ j → not_inferior i j
  (Fin 5).sum (λ i, if is_excellent i then 1 else 0)

theorem maximum_excellent_films : ∃ max, max_excellent_films = max ∧ max = 5 :=
by
  -- proof logic here
  sorry

end maximum_excellent_films_l545_545469


namespace max_area_pxy_proof_l545_545133

noncomputable def max_area_pxy : ℝ :=
  let r := 1 in
  let area (x : ℝ) := (1/2) * sin x * (1 + cos x) in
  real.Sup (set.range (λ x : ℝ, area x))

theorem max_area_pxy_proof :
  max_area_pxy = (3 * real.sqrt 3 / 8) := sorry

end max_area_pxy_proof_l545_545133


namespace projection_of_b_on_a_l545_545371

open Real

variables (a b : EuclideanSpace ℝ (Fin 2))
variables (angle : ℝ)
variables (magnitude_a : ℝ)
variables (magnitude_b : ℝ)

-- Conditions
axiom angle_ab : angle = 120 * π / 180 -- converting degrees to radians
axiom magnitude_a_def : ‖a‖ = 2
axiom magnitude_b_def : ‖b‖ = 4
axiom angle_def :  inner a b = ‖a‖ * ‖b‖ * cos angle

theorem projection_of_b_on_a : (inner a b) / ‖a‖ = -2 :=
by
  rw angle_ab at angle_def,
  rw magnitude_a_def at angle_def,
  rw magnitude_b_def at angle_def,
  simp at angle_def,
  sorry

end projection_of_b_on_a_l545_545371


namespace angle_bisector_divides_longest_side_l545_545349

theorem angle_bisector_divides_longest_side :
  ∀ (a b c : ℕ) (p q : ℕ), a = 12 → b = 15 → c = 18 →
  p + q = c → p * b = q * a → p = 8 ∧ q = 10 :=
by
  intros a b c p q ha hb hc hpq hprop
  rw [ha, hb, hc] at *
  sorry

end angle_bisector_divides_longest_side_l545_545349


namespace inequality_problem_l545_545993

-- Define the conditions and the problem statement
theorem inequality_problem (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a + b + c = 1) :
  (a - b * c) / (a + b * c) + (b - c * a) / (b + c * a) + (c - a * b) / (c + a * b) ≤ 3 / 2 :=
sorry

end inequality_problem_l545_545993


namespace largest_consecutive_odd_integer_sum_l545_545646

theorem largest_consecutive_odd_integer_sum :
  let S := (List.range 30).map (λ k => 2*k + 1) in
  let sum_S := S.sum in
  ∃ m : ℤ, sum_S = 5 * m ∧ (m + 4) = 184 := by
  sorry

end largest_consecutive_odd_integer_sum_l545_545646


namespace pyramid_division_l545_545257

noncomputable def pyramid_height (m : ℝ) : ℝ → ℝ → ℝ → Prop :=
  λ (x y z : ℝ), 
    x = m / (real.cbrt 3) ∧
    z = m * (1 - real.cbrt (2 / 3)) ∧
    y = m / (real.cbrt 3) * (real.cbrt 2 - 1)

theorem pyramid_division (m : ℝ) (h_m : 0 < m) : ∃ x y z : ℝ, pyramid_height m x y z :=
by {
  use (m / (real.cbrt 3)),
  use (m / (real.cbrt 3) * (real.cbrt 2 - 1)),
  use (m * (1 - real.cbrt (2 / 3))),
  split,
  { refl },
  split,
  { refl },
  { refl },
  sorry
}

end pyramid_division_l545_545257


namespace point_of_tangency_of_circles_l545_545670

/--
Given two circles defined by the following equations:
1. \( x^2 - 2x + y^2 - 10y + 17 = 0 \)
2. \( x^2 - 8x + y^2 - 10y + 49 = 0 \)
Prove that the coordinates of the point of tangency of these circles are \( (2.5, 5) \).
-/
theorem point_of_tangency_of_circles :
  (∃ x y : ℝ, (x^2 - 2*x + y^2 - 10*y + 17 = 0) ∧ (x = 2.5) ∧ (y = 5)) ∧ 
  (∃ x' y' : ℝ, (x'^2 - 8*x' + y'^2 - 10*y' + 49 = 0) ∧ (x' = 2.5) ∧ (y' = 5)) :=
sorry

end point_of_tangency_of_circles_l545_545670


namespace systematic_sampling_selection_l545_545755

/-- A junior high school leader uses a systematic sampling method to select 50 students 
from a total of 800 students in the preparatory grade. The students are numbered from 1 to 800. 
If the number 7 is drawn, this theorem proves that the selected number from the group of 16 numbers 
between 33 and 48 should be 39. -/
theorem systematic_sampling_selection 
  (population_size : ℕ) (sample_size : ℕ) (num_drawn : ℕ) (sampling_interval : ℕ)
  (group_lower_bound group_upper_bound selected_number : ℕ) :
  population_size = 800 →
  sample_size = 50 →
  num_drawn = 7 →
  sampling_interval = population_size / sample_size →
  group_lower_bound = 33 →
  group_upper_bound = 48 →
  selected_number = num_drawn + 2 * sampling_interval →
  selected_number ∈ set.range (group_upper_bound - group_lower_bound + 1) →
  group_lower_bound ≤ selected_number ∧ selected_number ≤ group_upper_bound :=
by
  intros
  split
  all_goals
  apply sorry

end systematic_sampling_selection_l545_545755


namespace arithmetic_sequence_with_geometric_property_l545_545351

noncomputable def a_n (n : ℕ) : ℕ := n + 1

def b_n (n : ℕ) : ℚ := 1 / (a_n n * (a_n n - 1))

def S_n (n : ℕ) : ℚ := ∑ i in Finset.range n, b_n (i + 1)

theorem arithmetic_sequence_with_geometric_property :
  (∀ (d : ℤ), d ≠ 0 → ∀ (a₁ a₃ a₇ : ℤ), a₁ = 2
    ∧ a₃ = a₁ + 2 * d ∧ a₇ = a₁ + 6 * d
    ∧ (a₃)^2 = a₁ * (a₇)
    → ∀ n, a_n n = n + 1)
  ∧ (∀ n, S_n n = n / (n + 1)) :=
sorry

end arithmetic_sequence_with_geometric_property_l545_545351


namespace expand_and_sum_coefficients_l545_545307

def sum_of_coefficients (c : ℝ) : ℝ :=
  let expanded_form := -((5 - 2 * c) * (c + 3 * (5 - 2 * c)))
  (10 * c^2 - 55 * c + 75).coefficients.sum

theorem expand_and_sum_coefficients (c : ℝ) :
  sum_of_coefficients c = -30 := by
  sorry

end expand_and_sum_coefficients_l545_545307


namespace cubic_roots_identity_l545_545570

-- Define the problem
theorem cubic_roots_identity :
  let p q r : ℝ := polynomial.roots (3 * X^3 - 5 * X^2 + 50 * X - 7)
  (p + q - 2)^3 + (q + r - 2)^3 + (r + p - 2)^3 = 249 / 9 :=
by sorry

end cubic_roots_identity_l545_545570


namespace range_of_distances_l545_545661

-- Define the points P and Q
def P : ℝ × ℝ := (-1, 3)
def Q : ℝ × ℝ := (2, -1)

-- Compute the distance between points P and Q
def distance (P Q : ℝ × ℝ) : ℝ :=
  real.sqrt ((Q.1 - P.1) ^ 2 + (Q.2 - P.2) ^ 2)

-- State the theorem
theorem range_of_distances (d : ℝ) (h : d = distance P Q) : 
  d > 0 ∧ d ≤ 5 := sorry

end range_of_distances_l545_545661


namespace combined_profit_percentage_correct_l545_545265

-- Definitions based on the conditions
noncomputable def profit_percentage_A := 30
noncomputable def discount_percentage_A := 10
noncomputable def profit_percentage_B := 24
noncomputable def discount_percentage_B := 15
noncomputable def profit_percentage_C := 40
noncomputable def discount_percentage_C := 20

-- Function to calculate selling price without discount
noncomputable def selling_price_without_discount (cost_price profit_percentage : ℝ) : ℝ :=
  cost_price * (1 + profit_percentage / 100)

-- Assume cost price for simplicity
noncomputable def cost_price : ℝ := 100

-- Calculations based on the conditions
noncomputable def selling_price_A := selling_price_without_discount cost_price profit_percentage_A
noncomputable def selling_price_B := selling_price_without_discount cost_price profit_percentage_B
noncomputable def selling_price_C := selling_price_without_discount cost_price profit_percentage_C

-- Calculate total cost price and the total selling price without any discount
noncomputable def total_cost_price := 3 * cost_price
noncomputable def total_selling_price_without_discount := selling_price_A + selling_price_B + selling_price_C

-- Combined profit
noncomputable def combined_profit := total_selling_price_without_discount - total_cost_price

-- Combined profit percentage
noncomputable def combined_profit_percentage := (combined_profit / total_cost_price) * 100

theorem combined_profit_percentage_correct :
  combined_profit_percentage = 31.33 :=
by
  sorry

end combined_profit_percentage_correct_l545_545265


namespace num_revolutions_correct_l545_545271

-- Define the conditions
def wheel_diameter : ℝ := 8
def travel_distance_miles : ℝ := 2
def mile_to_feet : ℝ := 5280
def travel_distance_feet : ℝ := travel_distance_miles * mile_to_feet

-- Define the radius from the diameter
def wheel_radius : ℝ := wheel_diameter / 2

-- Define the circumference of the wheel
def circumference_wheel : ℝ := 2 * Real.pi * wheel_radius

-- Define the expected number of revolutions
def expected_revolutions : ℝ := 1320 / Real.pi

-- Prove the equivalence
theorem num_revolutions_correct
  : travel_distance_feet / circumference_wheel = expected_revolutions :=
  by
    sorry

end num_revolutions_correct_l545_545271


namespace right_triangle_area_l545_545698

/-- Given a right triangle with one leg of length 30 inches and a hypotenuse of 34 inches,
    the area of the triangle is 240 square inches. -/
theorem right_triangle_area (a b c : ℝ) (h1 : a = 30) (h2 : c = 34) (h3 : a^2 + b^2 = c^2) :
  (1/2) * a * b = 240 :=
by
  rw [h1, h2] at h3
  have hb : b = 16 := by
    rw [←h3]
    norm_num
  rw [h1, hb]
  norm_num
  sorry

end right_triangle_area_l545_545698


namespace OP_perpendicular_EF_in_cyclic_quadrilateral_l545_545946

variable {A B C D P O E F : Type} [metric_space A]

-- Given conditions for cyclic quadrilateral
def is_cyclic_quadrilateral (A B C D : A) :=
  ∃ (O : A), is_on_circle O A ∧ is_on_circle O B ∧ is_on_circle O C ∧ is_on_circle O D

-- Given diagonals intersecting at point P
variable (AC BD : A → A)
variable [intersecting_at P AC BD]

-- Intersection point F on line EF
variable (EF : A → A)
variable [intersecting_at F EF]

-- The theorem to prove that OP is perpendicular to EF
theorem OP_perpendicular_EF_in_cyclic_quadrilateral 
  (A B C D P O E F : A) [is_cyclic_quadrilateral A B C D] [intersecting_at P AC BD] [intersecting_at F EF] :
  perpendicular (OP) (EF) :=
sorry

end OP_perpendicular_EF_in_cyclic_quadrilateral_l545_545946


namespace right_triangle_area_l545_545684

theorem right_triangle_area (leg1 hypotenuse : ℝ) (h1 : leg1 = 30) (h2 : hypotenuse = 34) (h3 : (leg1 ^ 2 + (sqrt (hypotenuse ^ 2 - leg1 ^ 2)) ^ 2 = hypotenuse ^ 2)) :
  (1 / 2) * leg1 * (sqrt (hypotenuse ^ 2 - leg1 ^ 2)) = 240 :=
by
  sorry

end right_triangle_area_l545_545684


namespace point_Q_representation_l545_545101

-- Definitions
variables {C D Q : Type} [AddCommGroup C] [AddCommGroup D] [AddCommGroup Q] [Module ℝ C] [Module ℝ D] [Module ℝ Q]
variable (CQ : ℝ)
variable (QD : ℝ)
variable (r s : ℝ)

-- Given condition: ratio CQ:QD = 7:2
axiom CQ_QD_ratio : CQ / QD = 7 / 2

-- Proof goal: the affine combination representation of the point Q
theorem point_Q_representation : CQ / (CQ + QD) = 7 / 9 ∧ QD / (CQ + QD) = 2 / 9 :=
sorry

end point_Q_representation_l545_545101


namespace problem_solution_l545_545475
open ProbabilityTheory

-- Definitions
def num_white_balls : ℕ := 6
def num_black_balls : ℕ := 4
def total_balls : ℕ := num_white_balls + num_black_balls
def draws : ℕ := 3
def score_black_ball : ℕ := 5
def score_white_ball : ℕ := 0
def X := λ (num_black_draws : ℕ) => num_black_draws * score_black_ball

-- Probability that total score is greater than 5 and expected value of total score
theorem problem_solution :
  let p_white := num_white_balls / total_balls
  let p_black := num_black_balls / total_balls
  let ξ := binomial (draws : ℕ) p_black in
  (probability (λ n, X (n : ℕ) > 5) = 44 / 125) ∧ (expected_value (λ n, X (n : ℕ)) = 6)
:= by
  -- Here we would provide the proof
  sorry

end problem_solution_l545_545475


namespace radius_ratio_of_inscribed_and_circumscribed_spheres_l545_545259

theorem radius_ratio_of_inscribed_and_circumscribed_spheres
  (a : ℝ)
  (P A B C : Type*)
  (right_triangular_pyramid : P → A → B → C → Prop)
  (perpendicular_edges : ∀ (p : P) (a : A) (b : B) (c : C),
    right_triangular_pyramid p a b c → 
    ∃ (u v w : ℝ), u ≠ 0 ∧ v ≠ 0 ∧ w ≠ 0 ∧ 
    ∃ f : P → A → B → C → ℝ × ℝ × ℝ,
      f p a b c = (u, v, w) ∧  
      u² + v² + w² = (a:ℝ)² + (a:ℝ)² + (a:ℝ)²) :
  (Real.sqrt 3 - 1) / 3 :=
by
  sorry

end radius_ratio_of_inscribed_and_circumscribed_spheres_l545_545259


namespace count_four_digit_numbers_divisible_by_5_end_45_l545_545412

theorem count_four_digit_numbers_divisible_by_5_end_45 : 
  {n : ℕ | 1000 ≤ n ∧ n < 10000 ∧ n % 100 = 45 ∧ n % 5 = 0}.to_finset.card = 90 :=
by
  sorry

end count_four_digit_numbers_divisible_by_5_end_45_l545_545412


namespace red_fraction_after_doubling_l545_545471

theorem red_fraction_after_doubling (x : ℕ) (h : (3/5 : ℚ) * x = (3/5 : ℚ) * x) : 
  let blue_marbles := (3/5 : ℚ) * x
      red_marbles := x - blue_marbles
      new_red_marbles := 2 * red_marbles 
      total_marbles := blue_marbles + new_red_marbles 
  in new_red_marbles / total_marbles = (4/7 : ℚ) :=
by 
  sorry

end red_fraction_after_doubling_l545_545471


namespace average_incorrect_answers_is_correct_l545_545470

-- Definitions
def total_items : ℕ := 60
def liza_correct_answers : ℕ := (90 * total_items) / 100
def rose_correct_answers : ℕ := liza_correct_answers + 2
def max_correct_answers : ℕ := liza_correct_answers - 5

def liza_incorrect_answers : ℕ := total_items - liza_correct_answers
def rose_incorrect_answers : ℕ := total_items - rose_correct_answers
def max_incorrect_answers : ℕ := total_items - max_correct_answers

def average_incorrect_answers : ℚ :=
  (liza_incorrect_answers + rose_incorrect_answers + max_incorrect_answers) / 3

-- Theorem statement
theorem average_incorrect_answers_is_correct : average_incorrect_answers = 7 := by
  -- Proof goes here
  sorry

end average_incorrect_answers_is_correct_l545_545470


namespace original_area_is_sqrt2_l545_545037

theorem original_area_is_sqrt2 : 
  ∀ (S' S : ℝ), (∀ a b :ℝ, a = 1 ∧ b = 1 → S = (1/2) * a * b) → S' = 2 * sqrt 2 * S → S' = sqrt 2 :=
by
  intros S' S h_triangle h_area_relation
  have S_calc : S = 1/2 := by 
    specialize h_triangle 1 1 ⟨rfl, rfl⟩
    assumption
  rw [S_calc] at h_area_relation
  exact h_area_relation

#check original_area_is_sqrt2

end original_area_is_sqrt2_l545_545037


namespace smallest_nonprime_integer_in_range_l545_545569

/-
Define the conditions: 
1. 'm' is an integer greater than 1 and non-prime.
2. 'm' has no prime factors less than 15.
-/
def is_smallest_nonprime_with_primes_ge_15 (m : ℕ) : Prop :=
  m > 1 ∧ ¬prime m ∧ (∀ p : ℕ, prime p → p < 15 → ¬ (p ∣ m)) ∧ 
  ∀ k : ℕ, k > 1 ∧ ¬prime k ∧ (∀ p : ℕ, prime p → p < 15 → ¬ (p ∣ k)) → k ≥ m

/-
State the theorem to be proven:
For 'm' satisfying the conditions above, show that 'm' falls in the range 280 < m ≤ 290.
-/
theorem smallest_nonprime_integer_in_range : 
  ∃ m : ℕ, is_smallest_nonprime_with_primes_ge_15 m ∧ 280 < m ∧ m ≤ 290 :=
by
  sorry

end smallest_nonprime_integer_in_range_l545_545569


namespace find_150th_digit_of_fraction_l545_545219

theorem find_150th_digit_of_fraction :
  let cycle := "197530864" in
  let length := String.length cycle in
  (16 / 81 : ℚ) = 0.19 -- Placeholder for the repeating decimal
  ∧ 150 % length = 6
  ∧ cycle[6] = '0' <- 
sorry

end find_150th_digit_of_fraction_l545_545219


namespace area_rectangle_right_triangle_l545_545559

theorem area_rectangle_right_triangle {A B C D E F : Point} (right_triangle : is_right_triangle A B C)
  (parallels : are_parallel_lines E D A C ∧ are_parallel_lines E F B C)
  (intersect : E ∈ hypotenuse A B)
  (triangle_areas : area_triangle A D E = 512 ∧ area_triangle D F E = 32) :
  area_rectangle D E F G = 256 :=
by 
  sorry

end area_rectangle_right_triangle_l545_545559


namespace distinct_ways_to_place_digits_l545_545454

theorem distinct_ways_to_place_digits :
  let digits := {1, 2, 3, 4}
  let boxes := 5
  let empty_box := 1
  -- There are 5! permutations of the list [0, 1, 2, 3, 4]
  let total_digits := insert 0 digits
  -- Resulting in 120 ways to place these digits in 5 boxes
  nat.factorial boxes = 120 :=
by 
  sorry

end distinct_ways_to_place_digits_l545_545454


namespace inverse_sum_l545_545093

def f (x : ℝ) : ℝ := x * |x|

theorem inverse_sum (h1 : ∃ x : ℝ, f x = 9) (h2 : ∃ x : ℝ, f x = -81) :
  ∃ a b: ℝ, f a = 9 ∧ f b = -81 ∧ a + b = -6 :=
by
  sorry

end inverse_sum_l545_545093


namespace find_value_of_a_l545_545648

theorem find_value_of_a (a : ℝ) : 
  (let y := λ x : ℝ, a * x in  y 0 + y 1 = 3) ↔ a = 3 := 
by
  sorry

end find_value_of_a_l545_545648


namespace mod_product_2023_2024_2025_2026_l545_545801

theorem mod_product_2023_2024_2025_2026 :
  (2023 * 2024 * 2025 * 2026) % 7 = 0 :=
by
  have h2023 : 2023 % 7 = 6 := by norm_num
  have h2024 : 2024 % 7 = 0 := by norm_num
  have h2025 : 2025 % 7 = 1 := by norm_num
  have h2026 : 2026 % 7 = 2 := by norm_num
  calc
    (2023 * 2024 * 2025 * 2026) % 7
      = ((2023 % 7) * (2024 % 7) * (2025 % 7) * (2026 % 7)) % 7 : by rw [Nat.mul_mod, Nat.mul_mod, Nat.mul_mod, Nat.mul_mod]
  ... = (6 * 0 * 1 * 2) % 7 : by rw [h2023, h2024, h2025, h2026]
  ... = 0 % 7 : by norm_num
  ... = 0 : by norm_num

end mod_product_2023_2024_2025_2026_l545_545801


namespace white_pawn_on_white_square_l545_545249

theorem white_pawn_on_white_square (w b N_b N_w : ℕ) (h1 : w > b) (h2 : N_b < N_w) : ∃ k : ℕ, k > 0 :=
by 
  -- Let's assume a contradiction
  -- The proof steps would be written here
  sorry

end white_pawn_on_white_square_l545_545249


namespace min_distinct_values_l545_545254

-- defining conditions
def mode_occurrences (l : List ℕ) (m : ℕ) := l.count m = 11
def has_unique_mode (l : List ℕ) := ∃ m : ℕ, (∀ n : ℕ, l.count n ≠ 11 → m ≠ n) ∧ (l.filter (λ n, l.count n = 11)).length = 1
def list_length (l : List ℕ) := l.length = 2023

-- defining the theorem to be proved
theorem min_distinct_values (l : List ℕ) (h1 : list_length l) (h2 : has_unique_mode l) (h3 : ∃ m : ℕ, mode_occurrences l m) :
  (l.to_finset.card ≥ 203) :=
sorry

end min_distinct_values_l545_545254


namespace osborn_friday_time_l545_545602

-- Conditions
def time_monday : ℕ := 2
def time_tuesday : ℕ := 4
def time_wednesday : ℕ := 3
def time_thursday : ℕ := 4
def old_average_time_per_day : ℕ := 3
def school_days_per_week : ℕ := 5

-- Total time needed to match old average
def total_time_needed : ℕ := old_average_time_per_day * school_days_per_week

-- Total time spent from Monday to Thursday
def time_spent_mon_to_thu : ℕ := time_monday + time_tuesday + time_wednesday + time_thursday

-- Goal: Find time on Friday
def time_friday : ℕ := total_time_needed - time_spent_mon_to_thu

theorem osborn_friday_time : time_friday = 2 :=
by
  sorry

end osborn_friday_time_l545_545602


namespace village_total_population_l545_545740

theorem village_total_population (P : ℕ) (h : 0.9 * P = 45000) : P = 50000 :=
sorry

end village_total_population_l545_545740


namespace probability_of_two_specific_suits_l545_545950

noncomputable def probability_two_suits (suit1: Fin 4) (suit2: Fin 4) : ℝ :=
  let p_each := 1 / 4
  in p_each ^ 6

theorem probability_of_two_specific_suits 
  (suit1: Fin 4) (suit2: Fin 4) (h_suit1: suit1 ≠ suit2) :
  probability_two_suits suit1 suit2 = 1 / 4096 :=
by
  sorry

end probability_of_two_specific_suits_l545_545950


namespace find_right_triangles_with_leg_2012_l545_545525

theorem find_right_triangles_with_leg_2012 :
  ∃ x y z : ℕ, x^2 + 2012^2 = y^2 ∧ x + y + z = 2012 + 253005 + 253013 ∨
  x^2 + 2012^2 = z^2 ∧ x + y + z = 2012 + 506016 + 506020 ∨
  x^2 + 2012^2 = y^2 ∧ x + y + z = 2012 + 1012035 + 1012037 ∨
  x^2 + 2012^2 = y^2 ∧ x + y + z = 2012 + 1509 + 2515 ∧
  y ≠ 2012 :=
begin
  sorry  -- The actual proof is omitted as specified.
end

end find_right_triangles_with_leg_2012_l545_545525


namespace quadratic_inequality_prob_l545_545095

noncomputable def probability_quadratic_inequality : ℝ :=
  let total_interval := Icc (-4 : ℝ) 4
  let favorable_interval := Ioo (-2 : ℝ) 1
  (measure_theory.measure.restrict measure_theory.measure_space ℝ total_interval).measure favorable_interval / 
    (measure_theory.measure.restrict measure_theory.measure_space ℝ total_interval).measure total_interval

theorem quadratic_inequality_prob :
  probability_quadratic_inequality = 3 / 8 :=
sorry

end quadratic_inequality_prob_l545_545095


namespace intersection_A_B_l545_545905

def A : set ℝ := {y | ∃ (x : ℝ), y = Real.exp x}
def B : set ℝ := {x | x^2 - x - 6 ≤ 0}

theorem intersection_A_B : A ∩ B = {y | 0 < y ∧ y ≤ 3} :=
by 
  sorry

end intersection_A_B_l545_545905


namespace area_multiplier_l545_545953

noncomputable def original_area (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  in Real.sqrt (s * (s - a) * (s - b) * (s - c))

noncomputable def new_area (a b c : ℝ) : ℝ :=
  let s' := 3 * (a + b + c) / 2
  in Real.sqrt (s' * (s' - 3 * a) * (s' - 3 * b) * (s' - 3 * c))

theorem area_multiplier (a b c : ℝ) (h₀ : a > 0) (h₁ : b > 0) (h₂ : c > 0) :
  new_area a b c = 9 * original_area a b c :=
  sorry

end area_multiplier_l545_545953


namespace initial_tagged_fish_l545_545961

noncomputable def number_of_tagged_fish (total_pond_fish : ℕ) (caught_fish_second : ℕ) (tagged_fish_second : ℕ) : ℕ :=
  (tagged_fish_second * total_pond_fish) / caught_fish_second

theorem initial_tagged_fish (total_pond_fish caught_fish_second tagged_fish_second : ℕ) 
  (h_caught_fish_second : caught_fish_second = 80)
  (h_tagged_fish_second : tagged_fish_second = 2)
  (h_total_pond_fish : total_pond_fish = 3200) :
  number_of_tagged_fish total_pond_fish caught_fish_second tagged_fish_second = 80 :=
by
  rw [h_caught_fish_second, h_tagged_fish_second, h_total_pond_fish]
  unfold number_of_tagged_fish
  norm_num
  rfl

end initial_tagged_fish_l545_545961


namespace obtuse_angle_probability_l545_545108

noncomputable def point := (ℝ × ℝ)
def F : point := (0, 3)
def G : point := (5, 0)
def H : point := (2 * Real.pi + 2, 0)
def I : point := (2 * Real.pi + 2, 5)
def J : point := (0, 5)
def K : point := (3, 5)

def hexagon_area : ℝ := 5 * Real.pi + 7.5
def semicircle_area : ℝ := (17 * Real.pi) / 8

def probability_obtuse_angle_FQG : ℝ :=
  semicircle_area / hexagon_area

theorem obtuse_angle_probability : probability_obtuse_angle_FQG = 17 / (40 * Real.pi + 60) :=
by sorry

end obtuse_angle_probability_l545_545108


namespace right_triangle_area_l545_545689

theorem right_triangle_area (a b c : ℝ) (h1 : a = 30) (h2 : c = 34) (h3 : a^2 + b^2 = c^2) :
  (1 / 2) * a * b = 240 :=
by
  sorry

end right_triangle_area_l545_545689


namespace hemisphere_base_area_l545_545173

theorem hemisphere_base_area (r : ℝ) (π : ℝ) (h₁ : π > 0) 
  (sphere_surface_area : 4 * π * r^2) 
  (hemisphere_surface_area : 3 * π * r^2 = 9) : 
  π * r^2 = 3 := 
by 
  sorry

end hemisphere_base_area_l545_545173


namespace magic_shop_change_l545_545491

theorem magic_shop_change :
  (∀ (cloak : Type), ∃ price_gold price_silver1 change_gold1 price_silver2 change_gold2, 
  price_silver1 = 20 ∧ change_gold1 = 4 ∧ 
  price_silver2 = 15 ∧ change_gold2 = 1 ∧ 
  price_gold = 14 ∧ 
  ∀ change_silver, 
    (20 - 4) * change_silver = 15 - 1 → -- Relation derived from the conditions
    (14 - (15 - 1) * change_silver / (20 - 4)) * change_silver = 10) := 
sorry

end magic_shop_change_l545_545491


namespace tom_calories_l545_545193

theorem tom_calories :
  let carrot_pounds := 1
  let broccoli_pounds := 2 * carrot_pounds
  let carrot_calories_per_pound := 51
  let broccoli_calories_per_pound := carrot_calories_per_pound / 3
  let total_carrot_calories := carrot_pounds * carrot_calories_per_pound
  let total_broccoli_calories := broccoli_pounds * broccoli_calories_per_pound
  let total_calories := total_carrot_calories + total_broccoli_calories
  total_calories = 85 :=
by
  sorry

end tom_calories_l545_545193


namespace no_nat_divisor_l545_545078

def f (n : ℕ) : ℕ := ∑ k in Finset.range 2011, n ^ k

theorem no_nat_divisor (m : ℕ) (n : ℕ) (h₁ : 2 ≤ m) (h₂ : m ≤ 2010) : ¬ (m ∣ f n) :=
begin
  sorry
end

end no_nat_divisor_l545_545078


namespace largest_even_integer_sum_l545_545155

theorem largest_even_integer_sum : 
  let sum_first_25_even := 2 * (25 * 26) / 2
  ∃ n : ℕ, 
  n % 2 = 0 ∧ 
  sum_first_25_even = 5 * n - 20 ∧
  n = 134 :=
by
  let sum_first_25_even := 2 * (25 * 26) / 2
  have h_sum : sum_first_25_even = 650 := by norm_num
  use 134
  split
  · norm_num
  split
  · rw h_sum
    norm_num
  · rfl

end largest_even_integer_sum_l545_545155


namespace correctness_of_statements_l545_545907

def set_A := {x : ℤ | ∃ k : ℤ, x = 3 * k - 1}

theorem correctness_of_statements :
  (¬ (-1 ∈ set_A)) = false ∧
  (¬ (-11 ∈ set_A)) = true ∧
  ((3 : ℤ) * (k : ℤ) ^ 2 - 1 ∈ set_A) = true ∧
  ((-34 : ℤ) ∈ set_A) = true :=
by
  unfold set_A
  split
  all_goals sorry

end correctness_of_statements_l545_545907


namespace expansion_coefficients_binomial_expansion_value_l545_545241

-- Definitions and conditions for the first problem
def coefficients_equal (n : ℕ) : Prop :=
  (Nat.choose n 5 * 2^5) = (Nat.choose n 6 * 2^6)

def term_max_binomial (n : ℕ) : Prop :=
  ∃ k : ℕ, k = 4 ∧ (Nat.choose n k * (2^k)) = 1120

-- Combined proof problem for the first part
theorem expansion_coefficients (n : ℕ) (h : coefficients_equal n) : n = 8 ∧ term_max_binomial n := 
sorry

-- Definitions and conditions for the second problem
def expansion_50 (a : ℕ → ℤ) (x : ℤ) : Prop := 
  (2 - (x * 3)^(1/2)) ^ 50 = ∑ i in range 51, a i * x^i

theorem binomial_expansion_value :
  let a := λ (i : ℕ), (choose 50 i) * 2^(50 - i) * (- (3)^(i/2)) in
  ∃ b : ℕ → ℤ, expansion_50 b 1 ∧
   ((∑ i in range 0 25, b (2*i))^2) - ((∑ i in range 0 24, b (2*i + 1))^2) = 1 :=
sorry

end expansion_coefficients_binomial_expansion_value_l545_545241


namespace find_common_ratio_form_arithmetic_sequence_l545_545070

variable {α : Type*} [Field α] (a : α) (q : α)

-- Define the geometric sequence and sum of first n terms
def geo_seq (n : ℕ) := a * q^n
def S : ℕ → α
| 0       := 0
| (n + 1) := S n + geo_seq a q n

-- Assuming that S_3, S_9, S_6 form an arithmetic sequence
theorem find_common_ratio (h1 : 2 * S a q 9 = S a q 3 + S a q 6) : q = -1 / 2 := sorry

-- Prove that a_k, a_{k+6}, a_{k+3} form an arithmetic sequence for k ∈ ℕ*
theorem form_arithmetic_sequence (k : ℕ) (h2 : q = -1 / 2) : geo_seq a q k + geo_seq a q (k + 3) = 2 * geo_seq a q (k + 6) := sorry

end find_common_ratio_form_arithmetic_sequence_l545_545070


namespace right_triangle_area_l545_545672

theorem right_triangle_area (leg1 leg2 hypotenuse : ℕ) (h_leg1 : leg1 = 30)
  (h_hypotenuse : hypotenuse = 34)
  (hypotenuse_sq : hypotenuse * hypotenuse = leg1 * leg1 + leg2 * leg2) :
  (1 / 2 : ℚ) * leg1 * leg2 = 240 := by
  sorry

end right_triangle_area_l545_545672


namespace trajectory_of_moving_point_l545_545353

variables {A : Type*} [ComplexField A]

theorem trajectory_of_moving_point
  (n : ℕ)
  (z : A)
  (z_k : fin n → A)
  (l : ℝ)
  (h_l_constant : ∑ k, (abs (z - z_k k))^2 = l)
  (h_centroid : ∑ k, z_k k = 0) :
  (l > ∑ k, abs (z_k k)^2 → abs z^2 = (1 / n) * (l - ∑ k, abs (z_k k)^2)) ∧
  (l = ∑ k, abs (z_k k)^2 → z = 0) ∧
  (l < ∑ k, abs (z_k k)^2 → False) :=
sorry

end trajectory_of_moving_point_l545_545353


namespace area_under_curve_l545_545989

def f (x : ℝ) : ℝ := if 0 ≤ x ∧ x ≤ 4 then x^2 else if 4 < x ∧ x ≤ 10 then 3 * x - 8 else 0

def A1 : ℝ := ∫ x in (0 : ℝ)..4, x^2

def A2 : ℝ := ∫ x in (4 : ℝ)..10, 3 * x - 8

def K : ℝ := A1 + A2

theorem area_under_curve : K = 99.33 :=
  by
    unfold K A1 A2 f
    -- Calculation will be done in the proof
    sorry

end area_under_curve_l545_545989


namespace fencing_cost_is_correct_l545_545636

def length : ℕ := 60
def cost_per_meter : ℕ := 27 -- using the closest integer value to 26.50
def breadth (l : ℕ) : ℕ := l - 20
def perimeter (l b : ℕ) : ℕ := 2 * l + 2 * b
def total_cost (P : ℕ) (c : ℕ) : ℕ := P * c

theorem fencing_cost_is_correct :
  total_cost (perimeter length (breadth length)) cost_per_meter = 5300 :=
  sorry

end fencing_cost_is_correct_l545_545636


namespace faces_odd_parity_l545_545861

-- Define the main theorem
theorem faces_odd_parity (P : Polyhedron) (C : Finset Color)
  (h1 : ∀ v ∈ P.vertices, (P.faces_meeting_at v).card = 3)
  (h2 : C.card = 4)
  (h3 : ∀ f ∈ P.faces, ∃ c ∈ C, f.color = c)
  (h4 : ∀ e ∈ P.edges, (P.faces_sharing_edge e).pairwise (λ f₁ f₂, f₁.color ≠ f₂.color)) :
  let faces_odd_sided (c : Color) := {f ∈ P.faces | f.color = c ∧ odd (f.sides)}
  in  odd (faces_odd_sided (elements C 0)).card  = odd (faces_odd_sided (elements C 1)).card :=
begin
  sorry
end

end faces_odd_parity_l545_545861


namespace find_k_for_sum_of_cubes_l545_545316

theorem find_k_for_sum_of_cubes (k : ℝ) (r s : ℝ)
  (h1 : r + s = -2)
  (h2 : r * s = k / 3)
  (h3 : r^3 + s^3 = r + s) : k = 3 :=
by
  -- Sorry will be replaced by the actual proof
  sorry

end find_k_for_sum_of_cubes_l545_545316


namespace probability_product_positive_is_5_div_9_l545_545197

noncomputable def probability_positive_product : ℚ :=
  let interval := Set.Icc (-30 : ℝ) 15
  let length_interval := 45
  let length_neg := 30
  let length_pos := 15
  let prob_neg := (length_neg : ℚ) / length_interval
  let prob_pos := (length_pos : ℚ) / length_interval
  let prob_product_pos := prob_neg^2 + prob_pos^2
  prob_product_pos

theorem probability_product_positive_is_5_div_9 :
  probability_positive_product = 5 / 9 :=
by
  sorry

end probability_product_positive_is_5_div_9_l545_545197


namespace garden_volume_l545_545258

-- Define the dimensions of the garden
def length := 12
def width := 5
def height := 3

-- Define the volume function for a rectangular prism
def volume_prism (l w h : ℕ) : ℕ :=
  l * w * h

-- Prove that the volume of the garden is 180 cubic meters
theorem garden_volume : volume_prism length width height = 180 := by
  sorry

end garden_volume_l545_545258


namespace silver_coins_change_l545_545503

-- Define the conditions
def condition1 : ℕ × ℕ := (20, 4) -- (20 silver coins, 4 gold coins change)
def condition2 : ℕ × ℕ := (15, 1) -- (15 silver coins, 1 gold coin change)
def cost_of_cloak_in_gold_coins : ℕ := 14

-- Define the theorem to be proven
theorem silver_coins_change (s1 g1 s2 g2 cloak_g : ℕ) (h1 : (s1, g1) = condition1) (h2 : (s2, g2) = condition2) :
  ∃ silver : ℕ, (silver = 10) :=
by {
  sorry
}

end silver_coins_change_l545_545503


namespace minimize_perimeter_isosceles_l545_545328

noncomputable def inradius (A B C : ℝ) (r : ℝ) : Prop := sorry -- Define inradius

theorem minimize_perimeter_isosceles (A B C : ℝ) (r : ℝ) 
  (h1 : A + B + C = 180) -- Angles sum to 180 degrees
  (h2 : inradius A B C r) -- Given inradius
  (h3 : A = fixed_angle) -- Given fixed angle A
  : B = C :=
by sorry

end minimize_perimeter_isosceles_l545_545328


namespace count_two_digit_multiples_of_seven_l545_545023

-- Define the conditions based on the problem
def is_multiple_of_seven (n : ℕ) : Prop := n % 7 = 0
def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

-- The main statement to be proven
theorem count_two_digit_multiples_of_seven : 
  (finset.filter (λ n, is_multiple_of_seven n) (finset.Icc 10 99)).card = 13 := 
sorry

end count_two_digit_multiples_of_seven_l545_545023


namespace solution_l545_545291

noncomputable def problem_statement : Prop :=
  (log 10 (1 / 4) - log 10 25) / (2 * log 5 10 + log 5 (1 / 4)) + (log 3 4 * log 8 9) = 1 / 3

theorem solution : problem_statement := 
  sorry

end solution_l545_545291


namespace scientific_notation_of_29_47_thousand_l545_545734

theorem scientific_notation_of_29_47_thousand :
  (29.47 * 1000 = 2.947 * 10^4) :=
sorry

end scientific_notation_of_29_47_thousand_l545_545734


namespace solution_set_f_div_x_lt_zero_l545_545250

noncomputable def f : ℝ → ℝ := sorry

theorem solution_set_f_div_x_lt_zero :
  (∀ x, f (2 + (2 - x)) = f x) ∧
  (∀ x1 x2 : ℝ, x1 < 2 ∧ x2 < 2 ∧ x1 ≠ x2 → (f x1 - f x2) / (x1 - x2) < 0) ∧
  f 4 = 0 →
  { x : ℝ | f x / x < 0 } = { x | x < 0 } ∪ { x | 0 < x ∧ x < 4 } :=
sorry

end solution_set_f_div_x_lt_zero_l545_545250


namespace prime_1011_n_l545_545848

theorem prime_1011_n (n : ℕ) (h : n ≥ 2) : 
  n = 2 ∨ n = 3 ∨ (∀ m : ℕ, m ∣ (n^3 + n + 1) → m = 1 ∨ m = n^3 + n + 1) :=
by sorry

end prime_1011_n_l545_545848


namespace green_pen_count_l545_545044

theorem green_pen_count 
  (blue_pens green_pens : ℕ)
  (h_ratio : blue_pens = 5 * green_pens / 3)
  (h_blue_pens : blue_pens = 20)
  : green_pens = 12 :=
by
  sorry

end green_pen_count_l545_545044


namespace complement_of_A_in_U_is_02_l545_545243

-- Define Universal set U and set A
def U := {0, 1, 2, 3}
def A := {1, 3}

-- Predicate to check if an element belongs to the complement of A in U
def is_complement (x : ℕ) : Prop := x ∈ U ∧ x ∉ A

-- Theorem statement without proof
theorem complement_of_A_in_U_is_02 : {x | is_complement x} = {0, 2} :=
by sorry

end complement_of_A_in_U_is_02_l545_545243


namespace find_k_find_m_l545_545016

open Real

-- Definitions of vectors a and b
def a : ℝ × ℝ := (-1, 4)
def b : ℝ × ℝ := (2, 3)
def vec_scalar_mul (k : ℝ) (v : ℝ × ℝ) : ℝ × ℝ := (k * v.1, k * v.2)
def vec_add (v w : ℝ × ℝ) : ℝ × ℝ := (v.1 + w.1, v.2 + w.2)
def vec_sub (v w : ℝ × ℝ) : ℝ × ℝ := (v.1 - w.1, v.2 - w.2)
def collinear (u v : ℝ × ℝ) : Prop := ∃ λ : ℝ, vec_scalar_mul λ u = v

-- Part 1 condition: k such that k * a - 2 * b is collinear with a + 2 * b
def k_condition (k : ℝ) : Prop :=
  collinear (vec_sub (vec_scalar_mul k a) (vec_scalar_mul 2 b)) (vec_add a (vec_scalar_mul 2 b))

-- Part 1 proof statement
theorem find_k : ∃ k : ℝ, k_condition k := 
sorry

-- Definitions of vectors AB and BC based on given conditions with m
def AB : ℝ × ℝ := vec_sub (vec_scalar_mul 3 a) (vec_scalar_mul 2 b)
def BC (m : ℝ) : ℝ × ℝ := vec_add (vec_scalar_mul (-2) a) (vec_scalar_mul m b)

-- Part 2 condition: points A, B, C are collinear implies AB and BC are collinear
def m_condition (m : ℝ) : Prop :=
  collinear AB (BC m)

-- Part 2 proof statement
theorem find_m : ∃ m : ℝ, m_condition m := 
sorry

end find_k_find_m_l545_545016


namespace evaluation_at_x_4_l545_545615

noncomputable def simplified_expression (x : ℝ) :=
  (x - 1 - (3 / (x + 1))) / ((x^2 + 2 * x) / (x + 1))

theorem evaluation_at_x_4 : simplified_expression 4 = 1 / 2 :=
by
  sorry

end evaluation_at_x_4_l545_545615


namespace min_disks_to_guarantee_ten_with_same_label_l545_545557

theorem min_disks_to_guarantee_ten_with_same_label :
  ∀ (n : ℕ), n = 50 →
  ∀ (s : ℕ → ℕ), (∀ i, i ≤ n → s i = i) →
  (∑ i in finset.range (n + 1), s i = 1275) →
  (∃ k, k = 415 ∧  k = min_disks_ten_same_label s n) :=
begin
  sorry
  -- Proof is omitted as per the instructions
end

/-- Helper function to determine the minimum disks needed to guarantee ten disks of same label -/
noncomputable def min_disks_ten_same_label (s : ℕ → ℕ) (n : ℕ) : ℕ :=
-- Implementation is omitted, assuming it’s defined somewhere
sorry

end min_disks_to_guarantee_ten_with_same_label_l545_545557


namespace max_k_C_l545_545816

theorem max_k_C (n : ℕ) (h1 : Odd n) (h2 : 0 < n) :
  ∃ k : ℕ, (k = ((n + 1) / 2) ^ 2) := 
sorry

end max_k_C_l545_545816


namespace largest_allowed_set_size_correct_l545_545558

noncomputable def largest_allowed_set_size (N : ℕ) : ℕ :=
  N - Nat.floor (N / 4)

def is_allowed (S : Finset ℕ) : Prop :=
  ∀ (a b c : ℕ), a ∈ S → b ∈ S → c ∈ S → a ≠ b → b ≠ c → a ≠ c → (a ∣ b → b ∣ c → False)

theorem largest_allowed_set_size_correct (N : ℕ) (hN : 0 < N) : 
  ∃ S : Finset ℕ, is_allowed S ∧ S.card = largest_allowed_set_size N := sorry

end largest_allowed_set_size_correct_l545_545558


namespace cloak_change_l545_545499

theorem cloak_change (silver_for_cloak1 silver_for_cloak2 : ℕ) (gold_change1 gold_change2 : ℕ) (silver_change : ℕ) :
  silver_for_cloak1 = 20 →
  gold_change1 = 4 →
  silver_for_cloak2 = 15 →
  gold_change2 = 1 →
  ∃ silver_cost_of_cloak : ℕ, 
    silver_cost_of_cloak = (20 - 4) * (5 / 3) →
    silver_change = 10 →
    14 * (5 / 3) - 8 = silver_change :=
by 
  assume h1 h2 h3 h4,
  use 16, 
  sorry

end cloak_change_l545_545499


namespace hyperbola_property_l545_545756

noncomputable def hyperbola : set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1^2 / 4) - (p.2^2 / 3) = 1}

def left_focus : ℝ × ℝ :=
  (-√5, 0)

def right_focus : ℝ × ℝ :=
  (√5, 0)

def on_line_through_focus (F : ℝ × ℝ) (p : ℝ × ℝ) : Prop :=
  ∃ m b : ℝ, p.2 = m * p.1 + b ∧ F.2 = m * F.1 + b

def intersects_hyperbola (F : ℝ × ℝ) : set (ℝ × ℝ) :=
  {p | on_line_through_focus F p ∧ p ∈ hyperbola}

noncomputable def M : ℝ × ℝ := classical.some (nonempty_inter (intersects_hyperbola left_focus) hyperbola)
noncomputable def N : ℝ × ℝ := classical.some (nonempty_inter (intersects_hyperbola left_focus) ({p | p ∈ hyperbola ∧ p ≠ M}))

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  ( (p1.1 - p2.1)^2 + (p1.2 - p2.2)^2 ).sqrt

theorem hyperbola_property :
  abs (distance M right_focus + distance N right_focus - distance M N) = 8 :=
sorry

end hyperbola_property_l545_545756


namespace profit_without_discount_l545_545724

theorem profit_without_discount (CP SP_discount SP_without_discount : ℝ) (profit_discount profit_without_discount percent_discount : ℝ)
  (h1 : CP = 100) 
  (h2 : percent_discount = 0.05) 
  (h3 : profit_discount = 0.425) 
  (h4 : SP_discount = CP + profit_discount * CP) 
  (h5 : SP_discount = 142.5)
  (h6 : SP_without_discount = SP_discount / (1 - percent_discount)) : 
  profit_without_discount = ((SP_without_discount - CP) / CP) * 100 := 
by
  sorry

end profit_without_discount_l545_545724


namespace magic_shop_change_l545_545495

theorem magic_shop_change :
  (∀ (cloak : Type), ∃ price_gold price_silver1 change_gold1 price_silver2 change_gold2, 
  price_silver1 = 20 ∧ change_gold1 = 4 ∧ 
  price_silver2 = 15 ∧ change_gold2 = 1 ∧ 
  price_gold = 14 ∧ 
  ∀ change_silver, 
    (20 - 4) * change_silver = 15 - 1 → -- Relation derived from the conditions
    (14 - (15 - 1) * change_silver / (20 - 4)) * change_silver = 10) := 
sorry

end magic_shop_change_l545_545495


namespace total_zinc_in_mixture_l545_545657

-- Define the fractions of zinc in each alloy as given conditions
def fraction_zinc_alloy_A : ℚ := 5 / 8
def fraction_zinc_alloy_B : ℚ := 9 / 13
def fraction_zinc_alloy_C : ℚ := 3 / 5

-- Define the resulting proportions of the mixture
def proportion_alloy_A : ℚ := 0.40
def proportion_alloy_B : ℚ := 0.35
def proportion_alloy_C : ℚ := 0.25

-- Define the total weight of the mixture
variable W : ℝ

-- Define the expected total zinc as a function of W
def expected_total_zinc (W : ℝ) : ℝ := 0.6423 * W

-- State the proof problem
theorem total_zinc_in_mixture (W : ℝ) : 
  let zinc_from_A := proportion_alloy_A * W * fraction_zinc_alloy_A
  let zinc_from_B := proportion_alloy_B * W * fraction_zinc_alloy_B
  let zinc_from_C := proportion_alloy_C * W * fraction_zinc_alloy_C
  in zinc_from_A + zinc_from_B + zinc_from_C = expected_total_zinc W := sorry

end total_zinc_in_mixture_l545_545657


namespace problem_l545_545341

-- Given function definition
def f (a x : ℝ) : ℝ := a * log x - x^2

-- Proof problem in Lean 4 statement
theorem problem (a : ℝ) :
  (∀ p q : ℝ, 0 < p ∧ p < 1 ∧ 0 < q ∧ q < 1 ∧ p ≠ q → (f a p - f a q) / (p - q) ≥ 0) ↔ 3 ≤ a := 
sorry

end problem_l545_545341


namespace triangle_area_l545_545269

noncomputable def area_of_triangle_enclosed_by_lines : ℝ :=
let 
  line1 (x : ℝ) := (3/4) * x + 3,
  line2 (x : ℝ) := -2 * x + 8,
  line3 (y : ℝ) := y
in
let 
  vertice1 := ( -(4/3) : ℝ, 2 : ℝ ),
  vertice2 := ( 3 : ℝ, 2 : ℝ ),
  x_intersect := 20/11 : ℝ,
  y_intersect := 56/11 : ℝ,
  vertice3 := ( x_intersect, y_intersect ),
  base := 13/3 : ℝ,
  height := 34/11 : ℝ
in
(1/2 : ℝ) * base * height

theorem triangle_area : area_of_triangle_enclosed_by_lines ≈ 6.70 :=
sorry

end triangle_area_l545_545269


namespace imaginary_part_of_z_is_one_l545_545886

noncomputable def z : ℂ := 2 * complex.i / (1 - complex.i)

theorem imaginary_part_of_z_is_one : z.im = 1 :=
sorry

end imaginary_part_of_z_is_one_l545_545886
