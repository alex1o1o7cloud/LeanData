import Mathlib
import Mathlib.Algebra.BigOperators
import Mathlib.Algebra.Combinatorics
import Mathlib.Algebra.EuclideanDomain.Basic
import Mathlib.Algebra.Field
import Mathlib.Algebra.Group.Defs
import Mathlib.Analysis.Calculus.MeanValue
import Mathlib.Analysis.SpecialFunctions.ExpLog
import Mathlib.Combinatorics.Basic
import Mathlib.Combinatorics.Combinations
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Fintype.Basic
import Mathlib.Data.Int.Basic
import Mathlib.Data.List.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Binomial
import Mathlib.Data.Probability.Basic
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Rat.Defs
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.Integral
import Mathlib.NumberTheory.Factorization.Basic
import Mathlib.NumberTheory.Wilson
import Mathlib.Probability.Basic
import Mathlib.Tactic.Basic
import Mathlib.Tactic.Linarith

namespace pythagorean_theorem_isosceles_right_triangle_l381_381689

theorem pythagorean_theorem_isosceles_right_triangle :
  ∀ (a : ℝ), a = 1 → (a^2 + a^2 = (sqrt (a^2 + a^2))^2) :=
by
  intro a h_a
  sorry

end pythagorean_theorem_isosceles_right_triangle_l381_381689


namespace who_wears_which_dress_l381_381737

def girls := ["Katya", "Olya", "Liza", "Rita"]
def dresses := ["pink", "green", "yellow", "blue"]

variable (who_wears_dress : String → String)

/-- Conditions given in the problem --/
axiom Katya_not_pink_blue : who_wears_dress "Katya" ≠ "pink" ∧ who_wears_dress "Katya" ≠ "blue"
axiom between_green_liza_yellow : ∃ g, who_wears_dress "Katya" = "green" ∧ who_wears_dress "Rita" = "yellow"
axiom Rita_not_green_blue : who_wears_dress "Rita" ≠ "green" ∧ who_wears_dress "Rita" ≠ "blue"
axiom Olya_between_rita_pink : ∃ o, who_wears_dress "Olya" = "blue" ∧ who_wears_dress "Liza" = "pink"

theorem who_wears_which_dress :
  who_wears_dress "Katya" = "green" ∧
  who_wears_dress "Olya" = "blue" ∧
  who_wears_dress "Liza" = "pink" ∧
  who_wears_dress "Rita" = "yellow" :=
by
  sorry

end who_wears_which_dress_l381_381737


namespace circles_externally_tangent_l381_381065

noncomputable def circle1_center : ℝ × ℝ := (-1, 1)
noncomputable def circle1_radius : ℝ := 2
noncomputable def circle2_center : ℝ × ℝ := (2, -3)
noncomputable def circle2_radius : ℝ := 3

noncomputable def distance_centers : ℝ :=
  Real.sqrt ((circle1_center.1 - circle2_center.1)^2 + (circle1_center.2 - circle2_center.2)^2)

theorem circles_externally_tangent :
  distance_centers = circle1_radius + circle2_radius :=
by
  -- The proof will show that the distance between the centers is equal to the sum of the radii, 
  -- indicating they are externally tangent.
  sorry

end circles_externally_tangent_l381_381065


namespace tessellation_coloring_l381_381591

-- Define the tessellation problem
def tessellation (m n : ℕ) : Type :=
  {uv : (ℕ × ℕ) // uv.1 < m ∧ uv.2 < n}

-- Define adjacency for tiles in the tessellation
def adjacent_tiles (m n : ℕ) (u v : tessellation m n) : Prop :=
  ((u.val.1 = v.val.1 ∧ (u.val.2 + 1 = v.val.2 ∨ u.val.2 = v.val.2 + 1)) ∨
   (u.val.2 = v.val.2 ∧ (u.val.1 + 1 = v.val.1 ∨ u.val.1 = v.val.1 + 1)))

-- The least number of colors needed to shade the tessellation
theorem tessellation_coloring (m n : ℕ) :
  ∃ (k : ℕ), (∀ (c : tessellation m n → ℕ), (∀ (u v : tessellation m n), adjacent_tiles m n u v → c u ≠ c v) → k = 2) :=
begin
  sorry
end

end tessellation_coloring_l381_381591


namespace first_term_arithmetic_series_l381_381902

theorem first_term_arithmetic_series 
  (a d : ℚ) 
  (h1 : 30 * (2 * a + 59 * d) = 240)
  (h2 : 30 * (2 * a + 179 * d) = 3600) : 
  a = -353 / 15 :=
by
  have eq1 : 2 * a + 59 * d = 8 := by sorry
  have eq2 : 2 * a + 179 * d = 120 := by sorry
  sorry

end first_term_arithmetic_series_l381_381902


namespace complement_union_M_N_eq_ge_2_l381_381955

def U := set.univ ℝ
def M := {x : ℝ | x < 1}
def N := {x : ℝ | -1 < x ∧ x < 2}

theorem complement_union_M_N_eq_ge_2 :
  (U \ (M ∪ N)) = {x : ℝ | 2 ≤ x} :=
by sorry

end complement_union_M_N_eq_ge_2_l381_381955


namespace total_surface_area_proof_l381_381083

def radius : ℝ := 5
def height : ℝ := 10

def hemisphere_surface_area (r : ℝ) : ℝ := (1 / 2) * 4 * Real.pi * r^2
def cylindrical_surface_area (r h : ℝ) : ℝ := 2 * Real.pi * r * h
def total_surface_area (r h : ℝ) : ℝ := hemisphere_surface_area r + cylindrical_surface_area r h

theorem total_surface_area_proof : total_surface_area radius height = 150 * Real.pi :=
by
  unfold total_surface_area
  unfold hemisphere_surface_area
  unfold cylindrical_surface_area
  norm_num
  ring
  sorry

end total_surface_area_proof_l381_381083


namespace length_longest_side_triangle_l381_381891

theorem length_longest_side_triangle 
  (A B C : ℝ) 
  (tan_A : ℝ := 1 / 4) 
  (tan_B : ℝ := 3 / 5) 
  (a : ℝ := real.sqrt 2)
  (ha : tan (A) = (1 / 4))
  (hb : tan (B) = (3 / 5))
  (hc : a = real.sqrt 2)
  (angle_sum : A + B + C = π) :
  ∃ c, (∀ b, b ≠ a → c ≠ b) ∧ c = real.sqrt 17 := 
by
  sorry

end length_longest_side_triangle_l381_381891


namespace find_k_l381_381841

variable (k : ℝ)
def a : ℝ × ℝ × ℝ := (1, 1, 0)
def b : ℝ × ℝ × ℝ := (-1, 0, 2)

-- Define the dot product for 3D vectors
def dot_product (v w : ℝ × ℝ × ℝ) : ℝ := 
  v.1 * w.1 + v.2 * w.2 + v.3 * w.3

-- Define the condition which states that the vectors are perpendicular
def perpendicular_condition : Prop :=
  dot_product (k * a.1 + b.1, k * a.2 + b.2, k * a.3 + b.3) (2 * a.1 - b.1, 2 * a.2 - b.2, 2 * a.3 - b.3) = 0

theorem find_k (h : perpendicular_condition k) : k = 7 / 5 := by
  sorry

end find_k_l381_381841


namespace third_smallest_is_five_l381_381701

noncomputable def probability_third_smallest_is_five : ℚ :=
  let total_ways := (Nat.choose 15 8) in
  let favorable_ways := (Nat.choose 4 2) * (Nat.choose 10 5) in
  favorable_ways / total_ways

theorem third_smallest_is_five :
  probability_third_smallest_is_five = 4 / 17 := sorry

end third_smallest_is_five_l381_381701


namespace sin_neg_120_eq_l381_381225

def angle1 := -120
def angle2 := 240
def point := (-1 / 2, -Real.sqrt 3 / 2)

theorem sin_neg_120_eq :
  ∠ angle1 = angle2 ∧ ∃ coords, coords = point -> Real.sin angle1 = -Real.sqrt 3 / 2 :=
by
  sorry

end sin_neg_120_eq_l381_381225


namespace sum_first_100_terms_eq_50_l381_381080

noncomputable def a (n : ℕ) : ℚ :=
  if h : n > 0 then n * Real.cos (n * Real.pi) else 0

theorem sum_first_100_terms_eq_50 :
  (finset.range 100).sum (fun n => a (n + 1)) = 50 := 
sorry

end sum_first_100_terms_eq_50_l381_381080


namespace smallest_b_for_factorization_l381_381300

-- Let us state the problem conditions and the objective
theorem smallest_b_for_factorization :
  ∃ (b : ℕ), b = 92 ∧ ∃ (p q : ℤ), (p + q = b) ∧ (p * q = 2016) :=
begin
  sorry
end

end smallest_b_for_factorization_l381_381300


namespace part_a_part_b_l381_381793

def sequence (n : ℕ) := fin n → ℝ

def d_i (a : sequence n) (i : fin n) :=
  (finset.univ.filter (λ j : fin n, j.val ≤ i.val)).sup a -
  (finset.univ.filter (λ j : fin n, j.val ≥ i.val)).inf a

def d (a : sequence n) :=
  (finset.univ).sup (λ i : fin n, d_i a i)

theorem part_a (a : sequence n) (x : sequence n) (h : ∀ i j : fin n, i ≤ j → x i ≤ x j) :
  (finset.univ).sup (λ i : fin n, abs (x i - a i)) ≥ d a / 2 :=
sorry

theorem part_b (a : sequence n) :
  ∃ x : sequence n, (∀ i j : fin n, i ≤ j → x i ≤ x j) ∧ 
  (finset.univ).sup (λ i : fin n, abs (x i - a i)) = d a / 2 :=
sorry

end part_a_part_b_l381_381793


namespace max_y_coord_of_cos3theta_l381_381716

theorem max_y_coord_of_cos3theta :
  ∃ θ : ℝ, let r := cos (3 * θ) in let y := r * sin θ in y = 3 * sqrt 3 / 8 :=
sorry

end max_y_coord_of_cos3theta_l381_381716


namespace cyclic_hexagon_equilateral_side_lengths_l381_381613

theorem cyclic_hexagon_equilateral_side_lengths (A B C D E F K : Type)
  [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited D] [Inhabited E] [Inhabited F]
  [Inhabited K]
  (cyclic_hexagon : ∀ (hex : Finset (A ∪ B ∪ C ∪ D ∪ E ∪ F)), cyclic hex)
  (equilateral_sides : ∀ (P Q : A ∪ B ∪ C ∪ D), P ≠ Q → distance P Q = distance (Q ∪ R) (R ∪ S))
  (K_on_AE : K ∈ segment A E)
  (angle_BKC_eq_KFE : ∀ (BK C E : Type), angle BK C = angle K F E)
  (angle_CKD_eq_KFA : ∀ (C K D A : Type), angle C K D = angle K F A) :
  distance K C = distance K F :=
sorry

end cyclic_hexagon_equilateral_side_lengths_l381_381613


namespace positive_solution_eq_l381_381294

theorem positive_solution_eq (x : ℝ) (h₁ : x = 1) :
  sqrt (x + sqrt (x + sqrt (x + …))) = sqrt (x * sqrt (x * sqrt (x * …))) :=
by
  sorry

end positive_solution_eq_l381_381294


namespace dress_assignment_l381_381759

theorem dress_assignment :
  ∃ (Katya Olya Liza Rita : string),
    (Katya ≠ "Pink" ∧ Katya ≠ "Blue") ∧
    (Rita ≠ "Green" ∧ Rita ≠ "Blue") ∧
    ∃ (girl_in_green girl_in_yellow : string),
      (girl_in_green = Katya ∧ girl_in_yellow = Rita ∧ 
       (Liza = "Pink" ∧ Olya = "Blue") ∧
       (Katya = "Green" ∧ Olya = "Blue" ∧ Liza = "Pink" ∧ Rita = "Yellow")) ∧
    ((girl_in_green stands between Liza and girl_in_yellow) ∧
     (Olya stands between Rita and Liza)) :=
by
  sorry

end dress_assignment_l381_381759


namespace num_pos_3_digits_div_by_7_l381_381859

theorem num_pos_3_digits_div_by_7 : 
  let lower_bound := 100
  let upper_bound := 999 
  let divisor := 7
  let smallest_3_digit := 105 -- or can be computed explicitly by: (lower_bound + divisor - 1) / divisor * divisor
  let largest_3_digit := 994  -- or can be computed explicitly by: upper_bound / divisor * divisor
  List.length (List.filter (λ n, n % divisor = 0) (List.range' smallest_3_digit (largest_3_digit + 1))) = 128 :=
by
  sorry

end num_pos_3_digits_div_by_7_l381_381859


namespace probability_xy_earlier_than_xm_l381_381113

variable {Ω : Type*} [Fintype Ω] [ProbabilitySpace Ω]

-- Define the possibilities of arrival for Xiao Jun, Xiao Yan and Xiao Ming
def arrival_order : Finset (Finset.univ : Finset (Fin 3)) := by
  exact {finset₁ | ∃ (x : Ω), true}

-- Define the event that Xiao Yan arrives earlier than Xiao Ming
def event_xy_earlier_than_xm (ω : Ω) : Prop := by
  sorry

-- Prove the probability of Xiao Yan arrives earlier than Xiao Ming is 1/2
theorem probability_xy_earlier_than_xm :
  Probability (λ ω, event_xy_earlier_than_xm ω) = 1/2 :=
by
  sorry

end probability_xy_earlier_than_xm_l381_381113


namespace sum_g_values_l381_381721

noncomputable def g (x : ℝ) : ℝ := 2 * x^3 - 3 * x^2 + 4 * x - 3

theorem sum_g_values :
  (Finset.range 2022).sum (λ i, g ((i + 1 : ℝ) / 2023)) = -3033 :=
sorry

end sum_g_values_l381_381721


namespace allison_total_craft_supplies_l381_381179

theorem allison_total_craft_supplies (M_glue : ℕ) (M_paper : ℕ) (A_glue_more : ℕ) (M_paper_ratio : ℕ) :
  M_glue = 15 → M_paper = 30 → A_glue_more = 8 → M_paper_ratio = 6 →
  let A_glue := M_glue + A_glue_more in
  let A_paper := M_paper / M_paper_ratio in
  A_glue + A_paper = 28 :=
by
  intros h1 h2 h3 h4
  let A_glue := 15 + 8
  let A_paper := 30 / 6
  sorry

end allison_total_craft_supplies_l381_381179


namespace sin_neg_120_eq_sqrt_3_over_2_l381_381217

noncomputable def sin_neg_angle (θ : ℝ) : ℝ := -Real.sin θ

theorem sin_neg_120_eq_sqrt_3_over_2 :
  sin_neg_angle (120 * Real.pi / 180) = Real.sqrt 3 / 2 :=
by
  -- Use the identity sin(-θ) = -sin(θ)
  have h1 : sin_neg_angle (120 * Real.pi / 180) = -Real.sin (120 * Real.pi / 180) := rfl
  
  -- Simplify and use pre-defined instances for sin(120 degrees)
  have h2 : Real.sin (120 * Real.pi / 180) = Real.sin (2 * Real.pi / 3) := by 
    norm_num

  -- Calculate the sin value for 2π/3
  have h3 : Real.sin (2 * Real.pi / 3) = Real.sin (Real.pi - Real.pi / 3) := by 
    norm_num [Real.sin_pi_sub_div]
  
  -- Which further simplifies to
  have h4 : Real.sin (Real.pi - Real.pi / 3) = Real.sin (π/3) := by
    norm_num
  
  -- Since sin(π/3) = sqrt(3)/2
  have h5 : Real.sin (π/3) = Real.sqrt 3 / 2 := by
    norm_num
  
  -- Applying all above results
  rw [h1, h2, h3, h4, h5]
  norm_num

  -- Concluding the proof
  exact sorry

end sin_neg_120_eq_sqrt_3_over_2_l381_381217


namespace hours_per_day_l381_381132

theorem hours_per_day 
  (H : ℕ)
  (h1 : 6 * 8 * H = 48 * H)
  (h2 : 4 * 3 * 8 = 96)
  (h3 : (48 * H) / 75 = 96 / 30) : 
  H = 5 :=
by
  sorry

end hours_per_day_l381_381132


namespace max_a_l381_381348

noncomputable def f (a x : ℝ) : ℝ := 2 * Real.log x - a * x^2 + 3

theorem max_a (a m n : ℝ) (h₀ : 1 ≤ m ∧ m ≤ 5)
                      (h₁ : 1 ≤ n ∧ n ≤ 5)
                      (h₂ : n - m ≥ 2)
                      (h_eq : f a m = f a n) :
  a ≤ Real.log 3 / 4 :=
sorry

end max_a_l381_381348


namespace smallest_fraction_of_land_l381_381057

noncomputable def smallest_share (n : ℕ) : ℚ :=
  if n = 150 then 1 / (2 * 3^49) else 0

theorem smallest_fraction_of_land :
  smallest_share 150 = 1 / (2 * 3^49) :=
sorry

end smallest_fraction_of_land_l381_381057


namespace count_3_digit_numbers_divisible_by_7_l381_381854

theorem count_3_digit_numbers_divisible_by_7 : 
  {n : ℕ | 100 ≤ n ∧ n ≤ 999 ∧ n % 7 = 0}.to_finset.card = 128 := 
  sorry

end count_3_digit_numbers_divisible_by_7_l381_381854


namespace root_interval_k_l381_381349

noncomputable def f (x : ℝ) : ℝ := Real.log x + 2 * x - 6

theorem root_interval_k (k : ℤ) (h_cont : Continuous f) (h_mono : Monotone f)
  (h1 : f 2 < 0) (h2 : f 3 > 0) : k = 4 :=
by
  -- The proof part is omitted as per instruction.
  sorry

end root_interval_k_l381_381349


namespace dress_assignment_l381_381758

theorem dress_assignment :
  ∃ (Katya Olya Liza Rita : string),
    (Katya ≠ "Pink" ∧ Katya ≠ "Blue") ∧
    (Rita ≠ "Green" ∧ Rita ≠ "Blue") ∧
    ∃ (girl_in_green girl_in_yellow : string),
      (girl_in_green = Katya ∧ girl_in_yellow = Rita ∧ 
       (Liza = "Pink" ∧ Olya = "Blue") ∧
       (Katya = "Green" ∧ Olya = "Blue" ∧ Liza = "Pink" ∧ Rita = "Yellow")) ∧
    ((girl_in_green stands between Liza and girl_in_yellow) ∧
     (Olya stands between Rita and Liza)) :=
by
  sorry

end dress_assignment_l381_381758


namespace find_total_people_find_children_l381_381102

variables (x m : ℕ)

-- Given conditions translated into Lean

def group_b_more_people (x : ℕ) := x + 4
def sum_is_18_times_difference (x : ℕ) := (x + (x + 4)) = 18 * ((x + 4) - x)
def children_b_less_than_three_times (m : ℕ) := (3 * m) - 2
def adult_ticket_price := 100
def children_ticket_price := (100 * 60) / 100
def same_amount_spent (x m : ℕ) := 100 * (x - m) + (100 * 60 / 100) * m = 100 * ((group_b_more_people x) - (children_b_less_than_three_times m)) + (100 * 60 / 100) * (children_b_less_than_three_times m)

-- Proving the two propositions (question == answer given conditions)

theorem find_total_people (x : ℕ) (hx : sum_is_18_times_difference x) : x = 34 ∧ (group_b_more_people x) = 38 :=
by {
  sorry -- proof for x = 34 and group_b_people = 38 given that sum_is_18_times_difference x
}

theorem find_children (m : ℕ) (x : ℕ) (hx : sum_is_18_times_difference x) (hm : same_amount_spent x m) : m = 6 ∧ (children_b_less_than_three_times m) = 16 :=
by {
  sorry -- proof for m = 6 and children_b_people = 16 given sum_is_18_times_difference x and same_amount_spent x m
}

end find_total_people_find_children_l381_381102


namespace Cheryl_golf_tournament_cost_l381_381214

theorem Cheryl_golf_tournament_cost :
  let electricity_bill := 800 in
  let cell_phone_expenses := electricity_bill + 400 in
  let tournament_extra_cost := 0.20 * cell_phone_expenses in
  let total_tournament_cost := cell_phone_expenses + tournament_extra_cost in
  total_tournament_cost = 1440 :=
by
  sorry

end Cheryl_golf_tournament_cost_l381_381214


namespace Sn_2012_is_201_l381_381847

-- Define the function Sn for the number of ways to write down the array starting with n
def Sn : ℕ → ℕ
| 1       := 1
| 2       := 1
| 3       := 1
| 4       := 1
| 5       := 2
| 6       := 2
| n+1     :=
  if n ≥ 6 then
    (Sn (6) + Sn (5) + Sn (4) + Sn (3) + Sn (2) + Sn (1)) * 
    (n + 1) 
  else if n ≥ 5 then
    (Sn (5) + Sn (4) + Sn (3) + Sn (2) + Sn (1)) * 
    (n + 1)
  else if n ≥ 4 then
    (Sn (4) + Sn (3) + Sn (2) + Sn (1)) * 
    (n + 1)
  else if n ≥ 3 then
    (Sn (3) + Sn (2) + Sn (1)) * 
    (n + 1)
  else if n ≥ 2 then
    (Sn (2) + Sn (1)) * 
    (n + 1)
  else
    Sn (n)

-- The main theorem
theorem Sn_2012_is_201 : Sn 2012 = 201 :=
 by sorry

end Sn_2012_is_201_l381_381847


namespace Rover_has_46_spots_l381_381363

theorem Rover_has_46_spots (G C R : ℕ) 
  (h1 : G = 5 * C)
  (h2 : C = (1/2 : ℝ) * R - 5)
  (h3 : G + C = 108) : 
  R = 46 :=
by
  sorry

end Rover_has_46_spots_l381_381363


namespace production_company_profit_l381_381153

theorem production_company_profit
  (opening_weekend_box_office : ℝ)
  (run_multiplier : ℝ)
  (keep_percentage : ℝ)
  (production_cost : ℝ)
  (opening_weekend_box_office = 120)
  (run_multiplier = 3.5)
  (keep_percentage = 0.60)
  (production_cost = 60)
  (total_run_box_office := opening_weekend_box_office * run_multiplier)
  (amount_kept := total_run_box_office * keep_percentage)
  (profit := amount_kept - production_cost) :
  profit = 192 :=
by
  sorry

end production_company_profit_l381_381153


namespace find_n_l381_381276

theorem find_n (a b : ℕ) (ha_pos : 0 < a) (hb_pos : 0 < b) (h1 : ∃ n : ℕ, n - 76 = a^3) (h2 : ∃ n : ℕ, n + 76 = b^3) : ∃ n : ℕ, n = 140 :=
by 
  sorry

end find_n_l381_381276


namespace probability_three_digit_divisible_by_5_with_ones_digit_9_l381_381082

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def ones_digit (n : ℕ) : ℕ := n % 10

def is_divisible_by_5 (n : ℕ) : Prop := n % 5 = 0

theorem probability_three_digit_divisible_by_5_with_ones_digit_9 : 
  ∀ (M : ℕ), is_three_digit M → ones_digit M = 9 → ¬ is_divisible_by_5 M := by
  intros M h1 h2
  sorry

end probability_three_digit_divisible_by_5_with_ones_digit_9_l381_381082


namespace not_possible_127_points_l381_381393

theorem not_possible_127_points (n_correct n_unanswered n_incorrect : ℕ) :
  n_correct + n_unanswered + n_incorrect = 25 →
  127 ≠ 5 * n_correct + 2 * n_unanswered - n_incorrect :=
by
  intro h_total
  sorry

end not_possible_127_points_l381_381393


namespace log_sum_eq_l381_381251

theorem log_sum_eq : ((∑ k in (Finset.range 98).map (λ n, n+3), (Real.log (1 + 1 / (k:ℝ)) / Real.log 3) * (Real.log 3 / Real.log k) * (Real.log 3 / Real.log (k+1)))) = 1 - (1 / Real.log 101 / Real.log 3) :=
by
  sorry

end log_sum_eq_l381_381251


namespace minimal_value_expression_l381_381010

theorem minimal_value_expression (a b c : ℝ) (h₀ : 0 ≤ a) (h₁ : 0 ≤ b) (h₂ : 0 ≤ c) (h₃ : a + b + c = 1) :
  (a + (ab)^(1/3) + (abc)^(1/4)) ≥ (1/3 + 1/(3 * (3^(1/3))) + 1/(3 * (3^(1/4)))) :=
sorry

end minimal_value_expression_l381_381010


namespace question_proof_l381_381983

open Set

variable (U : Set ℝ := univ)
variable (M : Set ℝ := {x | x < 1})
variable (N : Set ℝ := {x | -1 < x ∧ x < 2})

theorem question_proof : {x | x ≥ 2} = compl (M ∪ N) :=
by
  sorry

end question_proof_l381_381983


namespace allison_craft_items_l381_381176

def glue_sticks (A B : Nat) : Prop := A = B + 8
def construction_paper (A B : Nat) : Prop := B = 6 * A

theorem allison_craft_items (Marie_glue_sticks Marie_paper_packs : Nat)
    (h1 : Marie_glue_sticks = 15)
    (h2 : Marie_paper_packs = 30) :
    ∃ (Allison_glue_sticks Allison_paper_packs total_items : Nat),
        glue_sticks Allison_glue_sticks Marie_glue_sticks ∧
        construction_paper Allison_paper_packs Marie_paper_packs ∧
        total_items = Allison_glue_sticks + Allison_paper_packs ∧
        total_items = 28 :=
by
    sorry

end allison_craft_items_l381_381176


namespace calculate_final_price_l381_381026

noncomputable def final_price (j_init p_init : ℝ) (j_inc p_inc : ℝ) (tax discount : ℝ) (j_quantity p_quantity : ℕ) : ℝ :=
  let j_new := j_init + j_inc
  let p_new := p_init * (1 + p_inc)
  let total_price := (j_new * j_quantity) + (p_new * p_quantity)
  let tax_amount := total_price * tax
  let price_with_tax := total_price + tax_amount
  let final_price := if j_quantity > 1 ∧ p_quantity >= 3 then price_with_tax * (1 - discount) else price_with_tax
  final_price

theorem calculate_final_price :
  final_price 30 100 10 (0.20) (0.07) (0.10) 2 5 = 654.84 :=
by
  sorry

end calculate_final_price_l381_381026


namespace proof_problem_l381_381506

variable (f : ℝ → ℝ)
variable (h_odd : ∀ x : ℝ, f (-x) = -f x)

-- Definition for statement 1
def statement1 := f 0 = 0

-- Definition for statement 2
def statement2 := (∃ x > 0, ∀ y > 0, f x ≥ f y) → (∃ x < 0, ∀ y < 0, f x ≤ f y)

-- Definition for statement 3
def statement3 := (∀ x ≥ 1, ∀ y ≥ 1, x < y → f x < f y) → (∀ x ≤ -1, ∀ y ≤ -1, x < y → f y < f x)

-- Definition for statement 4
def statement4 := (∀ x > 0, f x = x^2 - 2 * x) → (∀ x < 0, f x = -x^2 - 2 * x)

-- Combined proof problem
theorem proof_problem :
  (statement1 f) ∧ (statement2 f) ∧ (statement4 f) ∧ ¬ (statement3 f) :=
by sorry

end proof_problem_l381_381506


namespace triangle_area_is_24_l381_381569

structure Point where
  x : ℝ
  y : ℝ

def distance_x (A B : Point) : ℝ :=
  abs (B.x - A.x)

def distance_y (A C : Point) : ℝ :=
  abs (C.y - A.y)

def triangle_area (A B C : Point) : ℝ :=
  0.5 * distance_x A B * distance_y A C

noncomputable def A : Point := ⟨2, 2⟩
noncomputable def B : Point := ⟨8, 2⟩
noncomputable def C : Point := ⟨4, 10⟩

theorem triangle_area_is_24 : triangle_area A B C = 24 := 
  sorry

end triangle_area_is_24_l381_381569


namespace frog_arrangement_count_l381_381549

theorem frog_arrangement_count :
  let total_frogs := 8,
      green_frogs := 3,
      red_frogs := 4,
      blue_frog := 1,
      valid_arrangements := 8 * (Nat.choose 5 3) * (Nat.factorial 3) * (Nat.factorial 4)
  in (valid_arrangements = 11520) :=
by
  sorry

end frog_arrangement_count_l381_381549


namespace complement_union_eq_ge2_l381_381949

open Set

variables {U : Type} [PartialOrder U] [LinearOrder U]

def U : Set ℝ := univ
def M : Set ℝ := { x | x < 1 }
def N : Set ℝ := { x | -1 < x ∧ x < 2 }
def Complement_U (A : Set ℝ) : Set ℝ := U \ A

theorem complement_union_eq_ge2 : 
  Complement_U (M ∪ N) = { x : ℝ | x ≥ 2 } :=
by {
  sorry
}

end complement_union_eq_ge2_l381_381949


namespace a_8_value_l381_381813

variable {n : ℕ}
def S (n : ℕ) : ℕ := n^2
def a (n : ℕ) : ℕ := S n - S (n - 1)

theorem a_8_value : a 8 = 15 := by
  sorry

end a_8_value_l381_381813


namespace complement_union_eq_l381_381967

-- Definitions / Conditions
def U : Set ℝ := Set.univ
def M : Set ℝ := { x | x < 1 }
def N : Set ℝ := { x | -1 < x ∧ x < 2 }

-- Statement of the theorem
theorem complement_union_eq {x : ℝ} :
  {x | x ≥ 2} = (U \ (M ∪ N)) := sorry

end complement_union_eq_l381_381967


namespace probability_three_correct_letters_l381_381088

-- Define the conditions and the theorem statement
noncomputable def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

noncomputable def binomial_coefficient (n k : ℕ) : ℕ := factorial n / (factorial k * factorial (n - k))

def derangements (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | 1 => 0
  | k => (k - 1) * (derangements (k - 1) + derangements (k - 2))

theorem probability_three_correct_letters :
  let total_permutations := factorial 5,
      choose_three_correct := binomial_coefficient 5 3,
      derange_two := derangements 2,
      favorable_outcomes := choose_three_correct * derange_two
  in favorable_outcomes / total_permutations = 1 / 12 := 
by
  sorry

end probability_three_correct_letters_l381_381088


namespace candy_distribution_l381_381266

theorem candy_distribution :
  (∑ r in finset.range 7 \ finset.range 2, 
     ∑ w in (finset.range (8 - r)).filter (λ w, w ≥ 2), 
       nat.choose 8 r * nat.choose (8 - r) w) = 120 :=
by
  sorry

end candy_distribution_l381_381266


namespace dreamy_bookstore_sales_l381_381047

theorem dreamy_bookstore_sales :
  let total_sales_percent := 100
  let notebooks_percent := 45
  let bookmarks_percent := 25
  let neither_notebooks_nor_bookmarks_percent := total_sales_percent - (notebooks_percent + bookmarks_percent)
  neither_notebooks_nor_bookmarks_percent = 30 :=
by {
  sorry
}

end dreamy_bookstore_sales_l381_381047


namespace telescoping_log_sum_l381_381241

theorem telescoping_log_sum :
  ∑ k in Finset.range 98 \ Finset.range 2, (Real.log (1 + 1 / k) / Real.log 3) * (Real.log 3 / Real.log k) * (Real.log 3 / Real.log (k + 1)) = 1 - 1 / Real.log 101 :=
by
  sorry

end telescoping_log_sum_l381_381241


namespace egg_cartons_l381_381202

theorem egg_cartons (chickens eggs_per_chicken eggs_per_carton : ℕ) (h_chickens : chickens = 20) (h_eggs_per_chicken : eggs_per_chicken = 6) (h_eggs_per_carton : eggs_per_carton = 12) : 
  (chickens * eggs_per_chicken) / eggs_per_carton = 10 :=
by
  rw [h_chickens, h_eggs_per_chicken, h_eggs_per_carton] -- Replace the variables with the given values
  -- Calculate the number of eggs
  have h_eggs := 20 * 6
  -- Apply the number of eggs to find the number of cartons
  rw [show 20 * 6 = 120, from rfl, show 120 / 12 = 10, from rfl]
  sorry -- Placeholder for the detailed proof

end egg_cartons_l381_381202


namespace probability_distance_less_than_one_l381_381788

-- Definitions of the problem conditions
def Circle (x y : ℝ) : Prop := x^2 + y^2 = 4
def Line (x y : ℝ) : Prop := y = x

-- The goal is to prove the probability statement
theorem probability_distance_less_than_one : 
  let event_count := (λ θ : ℝ, θ ≤ 2 * Mathlib.Real.pi / 3) -- Section under central angle 60 degrees
  let total_circumference := 2 * Mathlib.Real.pi
  (event_count / total_circumference) = 1 / 3 :=
sorry -- Proof to be filled in

end probability_distance_less_than_one_l381_381788


namespace sum_of_squares_of_distances_correct_l381_381304

noncomputable def sum_of_squares_of_distances (r R : ℝ) : ℝ :=
  R^2 + r^2 - 2 * R * r

theorem sum_of_squares_of_distances_correct (r R : ℝ) : 
  (∃ (I O : ℝ) (D E F : ℝ), 
    let ID := (I - D) in
    let IE := (I - E) in
    let IF := (I - F) in
    I = r ∧ O = R ∧ (ID^2 + IE^2 + IF^2 = R^2 + r^2 - 2 * R * r)) :=
sorry

end sum_of_squares_of_distances_correct_l381_381304


namespace quadrilateral_inscribed_formed_l381_381156

theorem quadrilateral_inscribed_formed (A B C D P : Point) (hABCD : is_isosceles_trapezoid A B C D) (hP_inside : inside_trapezoid P A B C D) :
  ∃ Q R S T : Point, quadrilateral Q R S T ∧ inscribed_in_trapezoid Q R S T A B C D :=
sorry

end quadrilateral_inscribed_formed_l381_381156


namespace cos_gamma_is_correct_l381_381425

variable (x y z : ℝ)
variable (α β γ : ℝ)

-- Conditions
def cos_alpha := 2 / 5
def cos_beta := 3 / 5
def z_eq_2y := z = 2 * y

-- Hypothesis for cosines of angles
def cos_alpha_def := x / (Real.sqrt (x^2 + y^2 + z^2)) = cos_alpha
def cos_beta_def := y / (Real.sqrt (x^2 + y^2 + z^2)) = cos_beta

-- The theorem to prove
theorem cos_gamma_is_correct 
  (h1 : cos_alpha_def)
  (h2 : cos_beta_def)
  (h3 : z_eq_2y) :
  (cos γ = (2 * Real.sqrt 3) / 5) := 
sorry

end cos_gamma_is_correct_l381_381425


namespace a_50_sqrt_3_l381_381911

noncomputable def sequence (n : ℕ) : ℝ :=
  nat.rec_on n 0 (λ n a_n, (a_n + Real.sqrt 3) / (1 - Real.sqrt 3 * a_n))

theorem a_50_sqrt_3 : sequence 50 = Real.sqrt 3 := 
by
  sorry

end a_50_sqrt_3_l381_381911


namespace solve_log_equation_l381_381622

noncomputable def f (a b x : ℝ) : ℝ :=
  (Real.log x / (Real.log a) ^ 2) - (2 * (Real.log x / Real.log a) / (-Real.log b))

noncomputable def g (a x : ℝ) : ℝ :=
  3 * (Real.log x / Real.log a) * (Real.log x / Real.log a)

theorem solve_log_equation (a b x : ℝ) (ha_pos : a > 0) (hb_pos : b > 0) (hx_pos : x > 0) (ha_ne_one : a ≠ 1) :
  f a b x = g a x → x = 1 ∨ x = 1 / 2 :=
begin
  intros h,
  sorry
end

end solve_log_equation_l381_381622


namespace airplane_average_speed_l381_381185

theorem airplane_average_speed :
  ∃ v : ℝ, 
  (1140 = 12 * (0.9 * v) + 26 * (1.2 * v)) ∧ 
  v = 27.14 := 
by
  sorry

end airplane_average_speed_l381_381185


namespace percentage_against_proposal_l381_381027

theorem percentage_against_proposal (A F : ℕ) (h1 : F = A + 66) (h2 : A + F = 330) : 
  (A.toRat / 330 * 100) = 40 :=
by
  sorry

end percentage_against_proposal_l381_381027


namespace solve_system_of_equations_l381_381043

theorem solve_system_of_equations:
  (∀ (x y : ℝ), 2 * y - x - 2 * x * y = -1 ∧ 4 * x ^ 2 * y ^ 2 + x ^ 2 + 4 * y ^ 2 - 4 * x * y = 61 →
  (x, y) = (-6, -1/2) ∨ (x, y) = (1, 3) ∨ (x, y) = (1, -5/2) ∨ (x, y) = (5, -1/2)) :=
by
  sorry

end solve_system_of_equations_l381_381043


namespace find_alpha_beta_sum_l381_381821

theorem find_alpha_beta_sum
  (a : ℝ) (α β φ : ℝ)
  (h1 : 3 * Real.sin α + 4 * Real.cos α = a)
  (h2 : 3 * Real.sin β + 4 * Real.cos β = a)
  (h3 : α ≠ β)
  (h4 : 0 < α ∧ α < 2 * Real.pi)
  (h5 : 0 < β ∧ β < 2 * Real.pi)
  (hφ : φ = Real.arcsin (4/5)) :
  α + β = Real.pi - 2 * φ ∨ α + β = 3 * Real.pi - 2 * φ :=
by
  sorry

end find_alpha_beta_sum_l381_381821


namespace dress_assignment_l381_381764

-- Define the four girls
inductive Girl
| Katya
| Olya
| Liza
| Rita

-- Define the four dresses
inductive Dress
| Pink
| Green
| Yellow
| Blue

-- Define the function that assigns each girl a dress
def dressOf : Girl → Dress

-- Conditions as definitions
axiom KatyaNotPink : dressOf Girl.Katya ≠ Dress.Pink
axiom KatyaNotBlue : dressOf Girl.Katya ≠ Dress.Blue
axiom RitaNotGreen : dressOf Girl.Rita ≠ Dress.Green
axiom RitaNotBlue : dressOf Girl.Rita ≠ Dress.Blue

axiom GreenBetweenLizaAndYellow : 
  ∃ (arrangement : List Girl), 
    arrangement = [Girl.Liza, Girl.Katya, Girl.Rita] ∧ 
    (dressOf Girl.Katya = Dress.Green ∧ 
    dressOf Girl.Liza = Dress.Pink ∧ 
    dressOf Girl.Rita = Dress.Yellow)

axiom OlyaBetweenRitaAndPink : 
  ∃ (arrangement : List Girl),
    arrangement = [Girl.Rita, Girl.Olya, Girl.Liza] ∧ 
    (dressOf Girl.Olya = Dress.Blue ∧ 
     dressOf Girl.Rita = Dress.Yellow ∧ 
     dressOf Girl.Liza = Dress.Pink)

-- Problem: Determine the dress assignments
theorem dress_assignment : 
  dressOf Girl.Katya = Dress.Green ∧ 
  dressOf Girl.Olya = Dress.Blue ∧ 
  dressOf Girl.Liza = Dress.Pink ∧ 
  dressOf Girl.Rita = Dress.Yellow :=
sorry

end dress_assignment_l381_381764


namespace water_flow_total_l381_381054

theorem water_flow_total
  (R1 R2 R3 : ℕ)
  (h1 : R2 = 36)
  (h2 : R2 = (3 / 2) * R1)
  (h3 : R3 = (5 / 4) * R2)
  : R1 + R2 + R3 = 105 :=
sorry

end water_flow_total_l381_381054


namespace part1_part2_l381_381401

noncomputable def seq_a : ℕ → ℝ
| 1       := 1 / 2
| (n + 1) := seq_a n + seq_a 1

def arithmetic_seq (a : ℕ → ℝ) : Prop :=
∀ n : ℕ, a (n + 1) = a n + d

def seq_b (a : ℕ → ℝ) : ℕ → ℝ
| n := 1 / ((a n - 1 / 2) * (a (n + 1) - 1 / 2))

def sum_b (b : ℕ → ℝ) : ℕ → ℝ
| 0       := 0
| (n + 1) := sum_b n + b n

variable (f : ℝ → ℝ)
variable (a : ℕ → ℝ := λ n, (∑ i in (range (n + 1)).map (λ x, f (x / n)))) 

axiom f_property : ∀ x : ℝ, f x + f (1 - x) = 1

theorem part1 : (∀ p q : ℕ, seq_a (p + q) = seq_a p + seq_a q) → 
                arithmetic_seq seq_a := 
sorry

theorem part2 : (∑ i in (range (n + 1)).map (λ x, f (x / n))) = (n + 1) / 2 →
                (sum_b (seq_b a) n) = 4 * (n / (n + 1)) :=
sorry

end part1_part2_l381_381401


namespace exactly_three_correct_is_impossible_l381_381086

theorem exactly_three_correct_is_impossible (n : ℕ) (hn : n = 5) (f : Fin n → Fin n) :
  (∃ S : Finset (Fin n), S.card = 3 ∧ ∀ i ∈ S, f i = i) → False :=
by
  intros h
  sorry

end exactly_three_correct_is_impossible_l381_381086


namespace dress_assignment_l381_381769

-- Define the four girls
inductive Girl
| Katya
| Olya
| Liza
| Rita

-- Define the four dresses
inductive Dress
| Pink
| Green
| Yellow
| Blue

-- Define the function that assigns each girl a dress
def dressOf : Girl → Dress

-- Conditions as definitions
axiom KatyaNotPink : dressOf Girl.Katya ≠ Dress.Pink
axiom KatyaNotBlue : dressOf Girl.Katya ≠ Dress.Blue
axiom RitaNotGreen : dressOf Girl.Rita ≠ Dress.Green
axiom RitaNotBlue : dressOf Girl.Rita ≠ Dress.Blue

axiom GreenBetweenLizaAndYellow : 
  ∃ (arrangement : List Girl), 
    arrangement = [Girl.Liza, Girl.Katya, Girl.Rita] ∧ 
    (dressOf Girl.Katya = Dress.Green ∧ 
    dressOf Girl.Liza = Dress.Pink ∧ 
    dressOf Girl.Rita = Dress.Yellow)

axiom OlyaBetweenRitaAndPink : 
  ∃ (arrangement : List Girl),
    arrangement = [Girl.Rita, Girl.Olya, Girl.Liza] ∧ 
    (dressOf Girl.Olya = Dress.Blue ∧ 
     dressOf Girl.Rita = Dress.Yellow ∧ 
     dressOf Girl.Liza = Dress.Pink)

-- Problem: Determine the dress assignments
theorem dress_assignment : 
  dressOf Girl.Katya = Dress.Green ∧ 
  dressOf Girl.Olya = Dress.Blue ∧ 
  dressOf Girl.Liza = Dress.Pink ∧ 
  dressOf Girl.Rita = Dress.Yellow :=
sorry

end dress_assignment_l381_381769


namespace count_3_digit_numbers_divisible_by_7_l381_381853

theorem count_3_digit_numbers_divisible_by_7 : 
  {n : ℕ | 100 ≤ n ∧ n ≤ 999 ∧ n % 7 = 0}.to_finset.card = 128 := 
  sorry

end count_3_digit_numbers_divisible_by_7_l381_381853


namespace remaining_rice_l381_381092

theorem remaining_rice {q_0 : ℕ} {c : ℕ} {d : ℕ} 
    (h_q0 : q_0 = 52) 
    (h_c : c = 9) 
    (h_d : d = 3) : 
    q_0 - (c * d) = 25 := 
  by 
    -- Proof to be written here
    sorry

end remaining_rice_l381_381092


namespace complement_union_eq_ge_two_l381_381998

def U : Set ℝ := Set.univ
def M : Set ℝ := { x : ℝ | x < 1 }
def N : Set ℝ := { x : ℝ | -1 < x ∧ x < 2 }

theorem complement_union_eq_ge_two : { x : ℝ | x ≥ 2 } = U \ (M ∪ N) :=
by
  sorry

end complement_union_eq_ge_two_l381_381998


namespace angle_A_is_pi_over_3_side_lengths_b_and_c_l381_381381

-- Given definitions
variables {A B C : ℝ} {a b c : ℝ} [triangle : triangle ABC a b c]

-- Conditions:
hypothesis (h1 : 4 * sin^2((B + C) / 2) - cos(2 * A) = 7 / 2)
hypothesis (h2 : a = sqrt 3)
hypothesis (h3 : b + c = 3)

-- Problems to prove:
theorem angle_A_is_pi_over_3 (h1 : 4 * sin^2((B + C) / 2) - cos(2 * A) = 7 / 2) :
  A = π / 3 :=
sorry

theorem side_lengths_b_and_c (h1 : 4 * sin^2((B + C) / 2) - cos(2 * A) = 7 / 2) (h2 : a = sqrt 3) (h3 : b + c = 3) :
  (b = 2 ∧ c = 1) ∨ (b = 1 ∧ c = 2) :=
sorry

end angle_A_is_pi_over_3_side_lengths_b_and_c_l381_381381


namespace question_proof_l381_381986

open Set

variable (U : Set ℝ := univ)
variable (M : Set ℝ := {x | x < 1})
variable (N : Set ℝ := {x | -1 < x ∧ x < 2})

theorem question_proof : {x | x ≥ 2} = compl (M ∪ N) :=
by
  sorry

end question_proof_l381_381986


namespace curveC1_cartesian_equiv_min_distance_to_curveC1_l381_381832
noncomputable def curveC : ℝ × ℝ → ℝ
| (ρ, θ) => 2 * ρ * Real.sin θ + ρ * Real.cos θ

def curveC_param (ρ θ : ℝ) : Prop :=
curveC (ρ, θ) = 10

def curveC1 : ℝ × ℝ → Prop
| (x, y) => ∃ α, x = 3 * Real.cos α ∧ y = 2 * Real.sin α

def curveC1_cartesian (x y : ℝ) : Prop :=
(x^2 / 9) + (y^2 / 4) = 1

def point_distance_to_line (x y : ℝ) : ℝ :=
Real.abs (x + 2 * y - 10) / Real.sqrt 5

theorem curveC1_cartesian_equiv (x y : ℝ) : curveC1 (x, y) → curveC1_cartesian x y :=
by
  intro h
  cases h with α hα
  rw [hα.1, hα.2]
  simp [Real.cos_sq_add_sin_sq]

theorem min_distance_to_curveC1 : ∀ (x y : ℝ), curveC1 (x, y) → point_distance_to_line x y = Real.sqrt 5 :=
by
  intros x y h
  cases h with α hα
  rw [hα.1, hα.2]
  have hcos : Real.cos α = 3 / 5 := sorry
  have hsin : Real.sin α = 4 / 5 := sorry
  simp [point_distance_to_line, hcos, hsin]
  norm_num

end curveC1_cartesian_equiv_min_distance_to_curveC1_l381_381832


namespace exist_end_2015_l381_381789

def in_sequence (n : Nat) : Nat :=
  90 * n + 75

theorem exist_end_2015 :
  ∃ n : Nat, in_sequence n % 10000 = 2015 :=
by
  sorry

end exist_end_2015_l381_381789


namespace log_ceil_floor_sum_l381_381677

theorem log_ceil_floor_sum :
  ∑ k in Finset.range 2000, k * (⌈Real.log2 (k + 1)⌉ - ⌊Real.log2 (k + 1)⌋) = 1998953 :=
by
  sorry

end log_ceil_floor_sum_l381_381677


namespace ball_radius_l381_381627

theorem ball_radius (x r : ℝ) (h1 : x^2 + 256 = r^2) (h2 : r = x + 16) : r = 16 :=
by
  sorry

end ball_radius_l381_381627


namespace lowest_possible_both_languages_l381_381403

theorem lowest_possible_both_languages (H E Total B : ℕ) 
  (hH : H = 30) (hE : E = 20) (hTotal : Total = 40) (hInclusion : Total ≤ H + E - B) : 
  10 ≤ B := 
by
  -- Definitions and hypothesist were assumed and mathematical steps should fit this format
  rw [hH, hE, hTotal] at hInclusion
  linarith
  sorry

end lowest_possible_both_languages_l381_381403


namespace m₁_m₂_relationship_l381_381451

-- Defining the conditions
variables {Point Line : Type}
variables (intersect : Line → Line → Prop)
variables (coplanar : Line → Line → Prop)

-- Assumption that lines l₁ and l₂ are non-coplanar.
variables {l₁ l₂ : Line} (h_non_coplanar : ¬ coplanar l₁ l₂)

-- Assuming m₁ and m₂ both intersect with l₁ and l₂.
variables {m₁ m₂ : Line}
variables (h_intersect_m₁_l₁ : intersect m₁ l₁)
variables (h_intersect_m₁_l₂ : intersect m₁ l₂)
variables (h_intersect_m₂_l₁ : intersect m₂ l₁)
variables (h_intersect_m₂_l₂ : intersect m₂ l₂)

-- Statement to prove that m₁ and m₂ are either intersecting or non-coplanar.
theorem m₁_m₂_relationship :
  (¬ coplanar m₁ m₂) ∨ (∃ p : Point, (intersect m₁ m₂ ∧ intersect m₂ m₁)) :=
sorry

end m₁_m₂_relationship_l381_381451


namespace intersection_complement_l381_381479

open Set

def A : Set ℝ := {x | x < -1 ∨ x > 2}
def B : Set ℝ := {x | 0 ≤ x ∧ x ≤ 2}

theorem intersection_complement :
  A ∩ (univ \ B) = {x : ℝ | x < -1 ∨ x > 2} :=
by
  sorry

end intersection_complement_l381_381479


namespace compare_power_sizes_l381_381103

theorem compare_power_sizes :
  2 ^ 444 = 4 ^ 222 ∧ 4 ^ 222 < 3 ^ 333 :=
by
  have h1 : 2 ^ 444 = (2 ^ 4) ^ 111 :=
    by rw [←pow_mul, Nat.mul_comm, pow_mul]
  have h2 : 3 ^ 333 = (3 ^ 3) ^ 111 :=
    by rw [←pow_mul, Nat.mul_comm, pow_mul]
  have h3 : 4 ^ 222 = (4 ^ 2) ^ 111 :=
    by rw [←pow_mul, Nat.mul_comm, pow_mul]
  have h4 : 2 ^ 4 = 4 ^ 2 ∧ 4 ^ 2 < 3 ^ 3 :=
    ⟨by norm_num, by norm_num⟩
  have h5 : 2 ^ 4 = 4 ^ 2 := h4.1
  have h6 : 4 ^ 2 < 3 ^ 3 := h4.2
  sorry

end compare_power_sizes_l381_381103


namespace opinions_eventually_stable_l381_381620

def Opinion : Type := Bool -- + = true, - = false

def next_opinion (current: Opinion) (left: Opinion) (right: Opinion) : Opinion :=
  if left = right then current else !current

def update_opinions (opinions: List Opinion) : List Opinion :=
  match opinions with
  | [] => []
  | _ =>
    let n := opinions.length
    opinions.enum.map (λ (idx, opinion) =>
      let left := opinions.get! ((idx + n - 1) % n)
      let right := opinions.get! ((idx + 1) % n)
      next_opinion opinion left right
    )

def eventually_stable (opinions: List Opinion) : Prop :=
  ∃ n, (List.iterate update_opinions opinions n) = opinions

theorem opinions_eventually_stable :
  ∀ initial_opinions : List Opinion, initial_opinions.length = 101 →
  eventually_stable initial_opinions :=
by
  sorry

end opinions_eventually_stable_l381_381620


namespace alice_bob_meet_l381_381656

theorem alice_bob_meet (n : ℕ) (hn : n = 15) (start_a : ℕ) (start_b : ℕ) (move_a : ℕ) (move_b : ℕ) :
  start_a = hn ∧ start_b = hn ∧ move_a = 7 ∧ move_b = 10 → ∃ k : ℕ, k = 8 :=
begin
  intros h,
  sorry
end

end alice_bob_meet_l381_381656


namespace least_perimeter_triangle_l381_381892

theorem least_perimeter_triangle :
  ∃ (d e f : ℕ), (∃ D E F : ℝ, cos D = 3 / 5 ∧ cos E = 3 / 4 ∧ cos F = -1 / 3) ∧
  (d + e + f = 76) ∧
  (∃ D E F : ℝ, d = ⌊4 / 5 * 76⌋ ∧ e = ⌊sqrt 7 / 4 * 76⌋ ∧ f = ⌊2 * sqrt 2 / 3 * 76⌋) :=
by
  sorry

end least_perimeter_triangle_l381_381892


namespace good_point_pair_extension_line_l381_381435

structure Point (α : Type) [Add α] [Mul α] :=
(x y : α)

variables {α : Type} [Field α]
variables (A B C D M N : Point α)
variables (λ μ : α)

def is_good_point_pair (A B : Point α) (X : Point α) (λ : α) : Prop :=
  ∃ (AB : Point α) (AC : Point α), AB = ⟨B.x - A.x, B.y - A.y⟩ ∧ AC = ⟨X.x - A.x, X.y - A.y⟩ ∧ 
  AC.x = λ * AB.x ∧ AC.y = λ * AB.y

theorem good_point_pair_extension_line (A B : Point α) (M N : Point α) (λ μ : α)
  (h_good_M : is_good_point_pair A B M λ)
  (h_good_N : is_good_point_pair A B N μ)
  (h_cond : λ⁻¹ + μ⁻¹ = 2) : ¬(λ > 1 ∧ μ > 1) :=
sorry

end good_point_pair_extension_line_l381_381435


namespace geom_seq_common_ratio_l381_381429

theorem geom_seq_common_ratio (a : ℕ → ℝ) (q : ℝ)
  (h1 : ∀ n, a n > 0)
  (h2 : ∀ n, a (n + 1) = q * a n)
  (h3 : ∑ i in finset.range 4, a i = 10 * ∑ i in finset.range 2, a i) :
  q = 3 :=
sorry

end geom_seq_common_ratio_l381_381429


namespace probability_three_girls_chosen_l381_381139

theorem probability_three_girls_chosen :
  let total_members := 15;
  let boys := 7;
  let girls := 8;
  let total_ways := Nat.choose total_members 3;
  let girls_ways := Nat.choose girls 3;
  total_ways = Nat.choose 15 3 ∧ girls_ways = Nat.choose 8 3 →
  (girls_ways : ℚ) / (total_ways : ℚ) = 8 / 65 := 
by  
  sorry

end probability_three_girls_chosen_l381_381139


namespace amanda_quizzes_l381_381184

theorem amanda_quizzes (n : ℕ) (h1 : n > 0) (h2 : 92 * n + 97 = 93 * 5) : n = 4 :=
by
  sorry

end amanda_quizzes_l381_381184


namespace tiles_needed_l381_381161

def tile_area : ℕ := 3 * 4
def floor_area : ℕ := 36 * 60

theorem tiles_needed : floor_area / tile_area = 180 := by
  sorry

end tiles_needed_l381_381161


namespace probability_third_smallest_is_four_l381_381307

open Finset

-- We define a set S of distinct integers from 1 to 12
def S : Finset ℕ := (finset.range 12).map (λ n, n + 1)

-- We are selecting 5 elements from the set
def choose_five : Finset (finset ℕ) := S.powerset.filter (λ x, x.card = 5)

-- The specific condition we need to check
def third_smallest_is_four (x : finset ℕ) : Prop :=
  ∃ (a b c d e : ℕ), a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ d ≠ e ∧ {a, b, c, d, e} = x ∧ (list.sort nat.less_than [a, b, c, d, e]).nth 2 = some 4

-- We calculate the number of successful outcomes
def successful_outcomes : finset (finset ℕ) := choose_five.filter third_smallest_is_four

-- The probability is the ratio of successful outcomes to the total number of outcomes
def probability : ℚ := (successful_outcomes.card : ℚ) / (choose_five.card : ℚ)

theorem probability_third_smallest_is_four : probability = 7 / 33 :=
by
  sorry

end probability_third_smallest_is_four_l381_381307


namespace problem_statement_l381_381972

open Set

variable (U : Set ℝ) (M N : Set ℝ)

theorem problem_statement (hU : U = univ) (hM : M = {x | x < 1}) (hN : N = {x | -1 < x ∧ x < 2}) :
  {x | 2 ≤ x} = compl (M ∪ N) :=
sorry

end problem_statement_l381_381972


namespace part_a_part_b_part_c_part_d_part_e_part_f_l381_381910

variables (n M : ℕ)
variables (R : Fin n → ℕ) (r : Fin M → ℕ)

-- Ensure n ≥ 3
variable (h1 : n ≥ 3)

-- Theorem (a)
theorem part_a (hn_odd : n % 2 = 1 ∨ n % 2 = 0) : 
  ∀ i : Fin n, (∑ j in Finset.filter (λ k, r k % 2 = 0) (Finset.univ : Finset (Fin M)), if R i k then 1 else 0) % 2 = if n % 2 = 1 then 0 else 1 :=
sorry

-- Theorem (b)
theorem part_b : ∑ i : Fin M, (r i - 1) * r i = (n - 1) * n :=
sorry

-- Theorem (c)
theorem part_c : ∑ i : Fin n, R i = ∑ i : Fin M, r i :=
sorry

-- Theorem (d)
theorem part_d : ∑ i : Fin M, r i ≥ 2 * n + M - 3 :=
sorry

-- Theorem (e)
theorem part_e : ∑ i : Fin n, R i ≥ 3 * n - 3 :=
sorry

-- Theorem (f)
theorem part_f : (∑ i : Fin n, R i) / n ≥ (∑ i : Fin M, r i) / M :=
sorry

end part_a_part_b_part_c_part_d_part_e_part_f_l381_381910


namespace compute_logarithmic_sum_l381_381226

theorem compute_logarithmic_sum : 
  ∑ k in finset.Icc 3 100, (log 3 (1 + (1 : ℝ) / k) * log k 3 * log (k + 1) 3) = -1 := 
sorry

end compute_logarithmic_sum_l381_381226


namespace log_sum_eq_l381_381254

theorem log_sum_eq : ((∑ k in (Finset.range 98).map (λ n, n+3), (Real.log (1 + 1 / (k:ℝ)) / Real.log 3) * (Real.log 3 / Real.log k) * (Real.log 3 / Real.log (k+1)))) = 1 - (1 / Real.log 101 / Real.log 3) :=
by
  sorry

end log_sum_eq_l381_381254


namespace complement_union_M_N_eq_ge_2_l381_381954

def U := set.univ ℝ
def M := {x : ℝ | x < 1}
def N := {x : ℝ | -1 < x ∧ x < 2}

theorem complement_union_M_N_eq_ge_2 :
  (U \ (M ∪ N)) = {x : ℝ | 2 ≤ x} :=
by sorry

end complement_union_M_N_eq_ge_2_l381_381954


namespace consumption_reduction_l381_381909

theorem consumption_reduction (P C : ℝ) (hP : P > 0) (hC : C > 0) :
  let new_price := P * 1.25 in
  let new_consumption := C / 1.25 in
  ((C - new_consumption) / C) * 100 = 20 :=
by
  -- Insert proof here
  sorry

end consumption_reduction_l381_381909


namespace compute_logarithmic_sum_l381_381230

theorem compute_logarithmic_sum : 
  ∑ k in finset.Icc 3 100, (log 3 (1 + (1 : ℝ) / k) * log k 3 * log (k + 1) 3) = -1 := 
sorry

end compute_logarithmic_sum_l381_381230


namespace necessary_but_not_sufficient_for_x_gt_4_l381_381614

theorem necessary_but_not_sufficient_for_x_gt_4 (x : ℝ) : (x^2 > 16) → ¬ (x > 4) :=
by
  sorry

end necessary_but_not_sufficient_for_x_gt_4_l381_381614


namespace GeneralAngleMeasureThm_l381_381804

def MeasureOfAngle (inside : Bool) (arc1 arc2 : ℝ) : ℝ :=
  if inside then (arc1 + arc2) / 2 else (arc1 - arc2) / 2

theorem GeneralAngleMeasureThm (inside : Bool) (arc1 arc2 : ℝ) :
  ∃ (angle : ℝ), angle = MeasureOfAngle inside arc1 arc2 ∧
  (if inside then arc1 + arc2 = angle * 2 else arc1 - arc2 = angle * 2) :=
by
  use MeasureOfAngle inside arc1 arc2
  split
  sorry


end GeneralAngleMeasureThm_l381_381804


namespace transformed_cos_function_l381_381061

theorem transformed_cos_function :
  (∀ x, y = - sin (1/2 * x - π/6)) :=
by
  let initial_function := λ x, cos x
  let shifted_function := λ x, cos (x + π/3)
  let stretched_function := λ x, cos (1/2 * x + π/3)
  let transformed_function := λ x, - sin (1/2 * x - π/6)
  sorry

end transformed_cos_function_l381_381061


namespace sin_pi_over_4_plus_alpha_cos_pi_over_6_minus_2alpha_l381_381315

theorem sin_pi_over_4_plus_alpha (α : ℝ) (h1 : α ∈ (π / 2, π)) (h2 : sin α = 3 / 5) : 
  sin (π / 4 + α) = - sqrt 2 / 10 := 
sorry

theorem cos_pi_over_6_minus_2alpha (α : ℝ) (h1 : α ∈ (π / 2, π)) (h2 : sin α = 3 / 5) : 
  cos (π / 6 - 2 * α) = (7 * sqrt 3 - 24) / 50 := 
sorry

end sin_pi_over_4_plus_alpha_cos_pi_over_6_minus_2alpha_l381_381315


namespace problem_statement_l381_381977

open Set

variable (U : Set ℝ) (M N : Set ℝ)

theorem problem_statement (hU : U = univ) (hM : M = {x | x < 1}) (hN : N = {x | -1 < x ∧ x < 2}) :
  {x | 2 ≤ x} = compl (M ∪ N) :=
sorry

end problem_statement_l381_381977


namespace part1_monotonic_part2_h_lambda_range_l381_381343

-- We are dealing with real exponentials and logarithms
open Real

-- Part (1) conditions and statement
theorem part1_monotonic (x : ℝ) (h : x > 0) : 
  let f := λ x : ℝ, exp (-x) + log x
  in ∀ x y, 0 < x → 0 < y → x < y → f x < f y :=
sorry

-- Part (2) conditions and statement
theorem part2_h_lambda_range (λ : ℝ) (h₀ : 0 < λ) (h₁ : λ < exp 1) :
  let f := λ x : ℝ, exp (λ * x) - λ * log x
  in ∃ x₀ > 0, (∀ x > 0, f x₀ ≤ f x) ∧ (1 < f x₀ ∧ f x₀ < 2 * exp 1) :=
sorry

end part1_monotonic_part2_h_lambda_range_l381_381343


namespace lines_intersect_lines_perpendicular_lines_coincide_lines_parallel_l381_381839

/- Define lines l1 and l2 -/
def l1 (a : ℝ) (x y : ℝ) : Prop := a * x + 2 * y + 6 = 0
def l2 (a : ℝ) (x y : ℝ) : Prop := x + (a - 1) * y + a^2 - 1 = 0

/- Prove intersection condition -/
theorem lines_intersect (a : ℝ) : (∃ x y, l1 a x y ∧ l2 a x y) ↔ (a ≠ -1 ∧ a ≠ 2) := 
sorry

/- Prove perpendicular condition -/
theorem lines_perpendicular (a : ℝ) : (∃ x1 y1 x2 y2, l1 a x1 y1 ∧ l2 a x2 y2 ∧ x1 * x2 + y1 * y2 = 0) ↔ (a = 2 / 3) :=
sorry

/- Prove coincident condition -/
theorem lines_coincide (a : ℝ) : (∀ x y, l1 a x y ↔ l2 a x y) ↔ (a = 2) := 
sorry

/- Prove parallel condition -/
theorem lines_parallel (a : ℝ) : (∀ x1 y1 x2 y2, l1 a x1 y1 → l2 a x2 y2 → (x1 * y2 - y1 * x2) = 0) ↔ (a = -1) := 
sorry

end lines_intersect_lines_perpendicular_lines_coincide_lines_parallel_l381_381839


namespace intersected_squares_and_circles_l381_381672

def is_intersected_by_line (p q : ℕ) : Prop :=
  p = q

def total_intersections : ℕ := 504 * 2

theorem intersected_squares_and_circles :
  total_intersections = 1008 :=
by
  sorry

end intersected_squares_and_circles_l381_381672


namespace cloth_width_length_area_l381_381367

theorem cloth_width_length_area (length_cm area_m2 : ℕ) (h_length : length_cm = 150) (h_area : area_m2 = 3) :
  (area_m2 * 10000) / length_cm = 200 :=
by
  have h_area_cm : area_m2 * 10000 = 30000 := by sorry
  have h_width : (30000) / length_cm = 200 := by sorry
  rw [h_length] at h_width
  exact h_width

end cloth_width_length_area_l381_381367


namespace expression_ne_12_l381_381062

theorem expression_ne_12 (a b c : ℕ) (n : ℕ) 
  (h1 : b + c - a = n * a) 
  (h2 : a + c - b = n * b) 
  (h3 : a + b - c = n * c) : 
  (a + b) * (b + c) * (a + c) ≠ 12 * a * b * c :=
begin
  sorry
end

end expression_ne_12_l381_381062


namespace problem_statement_l381_381978

open Set

variable (U : Set ℝ) (M N : Set ℝ)

theorem problem_statement (hU : U = univ) (hM : M = {x | x < 1}) (hN : N = {x | -1 < x ∧ x < 2}) :
  {x | 2 ≤ x} = compl (M ∪ N) :=
sorry

end problem_statement_l381_381978


namespace find_number_l381_381131

theorem find_number (x : ℝ) (h : 0.05 * x = 12.75) : x = 255 :=
by
  sorry

end find_number_l381_381131


namespace harmonic_series_inequality_l381_381438

noncomputable def S (n : ℕ) : ℚ :=
1 + ∑ i in Finset.range n, (1 : ℚ) / (i + 1)

theorem harmonic_series_inequality (p q : ℕ) (hp : 0 < p) (hq : 0 < q) :
  max ((1 : ℚ) / p) ((1 : ℚ) / q) ≤ S p + S q - S (p * q) ∧ S p + S q - S (p * q) ≤ 1 := sorry

end harmonic_series_inequality_l381_381438


namespace problem_statement_l381_381973

open Set

variable (U : Set ℝ) (M N : Set ℝ)

theorem problem_statement (hU : U = univ) (hM : M = {x | x < 1}) (hN : N = {x | -1 < x ∧ x < 2}) :
  {x | 2 ≤ x} = compl (M ∪ N) :=
sorry

end problem_statement_l381_381973


namespace area_outside_circle_l381_381932

theorem area_outside_circle (ABC : Triangle) (A B P Q P' Q' : ABC.Point)
  (hBAC : angle A B C = 90) (h_circle_tangent_AB : tangent_to_circle A B P)
  (h_circle_tangent_AC : tangent_to_circle A C Q)
  (h_diametral_P_P' : diameter_opposite P P') (h_diametral_Q_Q' : diameter_opposite Q Q')
  (h_P'_on_BC : on_line_segment B C P') (h_Q'_on_BC : on_line_segment B C Q')
  (h_AB_eq_8 : AB.length = 8) :
  area (circle_ABC.outside_triangle ABC A B P) = pi - 2 :=
sorry

end area_outside_circle_l381_381932


namespace area_of_quadrilateral_MARE_l381_381412

section Geometry

variables {P Q R M A E O : Type}
variables [MetricSpace P]

-- Define the unit circle and quadrilateral MARE
def unit_circle (ω : set P) : Prop := sorry
def diameter (A M : P) : Prop := sorry
def angle_bisector (E R A M : P) : Prop := sorry
def same_area (T1 T2 : set P) : Prop := sorry

-- Define the quadrilateral MARE
def is_quadrilateral (A M R E : P) : Prop := sorry

-- Define the required conditions
variable {ω : set P}
variable {T_RAM T_REM : set P}

axiom condition1 : unit_circle ω
axiom condition2 : diameter A M
axiom condition3 : angle_bisector E R A M
axiom condition4 : same_area T_RAM T_REM
axiom condition5 : is_quadrilateral A M R E
axiom side_length : MetricDistance.dist A M = 2

-- State the problem
theorem area_of_quadrilateral_MARE
  (ω : set P) (A M R E : P)
  (T_RAM T_REM T_MARE : set P) :
  unit_circle ω →
  diameter A M →
  angle_bisector E R A M →
  same_area T_RAM T_REM →
  is_quadrilateral A M R E →
  MetricDistance.dist A M = 2 →
  area T_MARE = (8 * Real.sqrt 2) / 9 :=
begin
  sorry
end

end Geometry

end area_of_quadrilateral_MARE_l381_381412


namespace complement_union_eq_l381_381966

-- Definitions / Conditions
def U : Set ℝ := Set.univ
def M : Set ℝ := { x | x < 1 }
def N : Set ℝ := { x | -1 < x ∧ x < 2 }

-- Statement of the theorem
theorem complement_union_eq {x : ℝ} :
  {x | x ≥ 2} = (U \ (M ∪ N)) := sorry

end complement_union_eq_l381_381966


namespace dress_assignment_l381_381765

-- Define the four girls
inductive Girl
| Katya
| Olya
| Liza
| Rita

-- Define the four dresses
inductive Dress
| Pink
| Green
| Yellow
| Blue

-- Define the function that assigns each girl a dress
def dressOf : Girl → Dress

-- Conditions as definitions
axiom KatyaNotPink : dressOf Girl.Katya ≠ Dress.Pink
axiom KatyaNotBlue : dressOf Girl.Katya ≠ Dress.Blue
axiom RitaNotGreen : dressOf Girl.Rita ≠ Dress.Green
axiom RitaNotBlue : dressOf Girl.Rita ≠ Dress.Blue

axiom GreenBetweenLizaAndYellow : 
  ∃ (arrangement : List Girl), 
    arrangement = [Girl.Liza, Girl.Katya, Girl.Rita] ∧ 
    (dressOf Girl.Katya = Dress.Green ∧ 
    dressOf Girl.Liza = Dress.Pink ∧ 
    dressOf Girl.Rita = Dress.Yellow)

axiom OlyaBetweenRitaAndPink : 
  ∃ (arrangement : List Girl),
    arrangement = [Girl.Rita, Girl.Olya, Girl.Liza] ∧ 
    (dressOf Girl.Olya = Dress.Blue ∧ 
     dressOf Girl.Rita = Dress.Yellow ∧ 
     dressOf Girl.Liza = Dress.Pink)

-- Problem: Determine the dress assignments
theorem dress_assignment : 
  dressOf Girl.Katya = Dress.Green ∧ 
  dressOf Girl.Olya = Dress.Blue ∧ 
  dressOf Girl.Liza = Dress.Pink ∧ 
  dressOf Girl.Rita = Dress.Yellow :=
sorry

end dress_assignment_l381_381765


namespace kirsten_stole_beef_meatballs_l381_381365

-- Declare the noncomputable definition if necessary
noncomputable def beef_meatballs_stolen (original_beef : ℕ) (left_beef : ℕ) : ℕ := 
  original_beef - left_beef

-- Given the conditions as axioms
axiom original_beef_meatballs : ℕ := 15
axiom left_beef_meatballs : ℕ := 10

-- State the theorem to be proved
theorem kirsten_stole_beef_meatballs : beef_meatballs_stolen original_beef_meatballs left_beef_meatballs = 5 := 
by sorry

end kirsten_stole_beef_meatballs_l381_381365


namespace pages_at_end_of_march_l381_381710

theorem pages_at_end_of_march (daily_pages : ℕ) (initial_pages : ℕ) (days_in_march : ℕ) :
  daily_pages = 30 → initial_pages = 400 → days_in_march = 31 →
  initial_pages + daily_pages * days_in_march = 1330 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  exact rfl

end pages_at_end_of_march_l381_381710


namespace successful_arrangements_unique_l381_381694

-- Definitions for conditions
def board_size (n : ℕ) : ℕ := 2^n - 1

-- Function to determine if an arrangement is successful
def is_successful_arrangement (n : ℕ) (arrangement : fin (board_size n) → fin (board_size n) → ℤ) : Prop :=
  ∀ i j : fin (board_size n), 
  arrangement i j = arrangement (i - 1) j * arrangement (i + 1) j * arrangement i (j - 1) * arrangement i (j + 1)

-- Proof statement
theorem successful_arrangements_unique (n : ℕ) : 
  (∃! arrangement : fin (board_size n) → fin (board_size n) → ℤ, 
      (∀ i j, arrangement i j = 1 ∨ arrangement i j = -1) ∧ 
      is_successful_arrangement n arrangement) :=
sorry

end successful_arrangements_unique_l381_381694


namespace shortest_tangent_segment_length_l381_381933

theorem shortest_tangent_segment_length :
  let C1 := { p : ℝ × ℝ | (p.1 - 3)^2 + p.2^2 = 16 },
      C2 := { p : ℝ × ℝ | (p.1 + 12)^2 + p.2^2 = 225 } in
  ∃ R S : ℝ × ℝ,
    R ∈ C1 ∧ S ∈ C2 ∧
    ∀ T U : ℝ × ℝ, T ∈ C1 ∧ U ∈ C2 → dist(R, S) ≤ dist(T, U) →
    dist(R, S) = sqrt (16 - ((60 : ℝ) / 19)^2) + sqrt (225 - ((225 : ℝ) / 19)^2) :=
by
  sorry

end shortest_tangent_segment_length_l381_381933


namespace complement_union_eq_l381_381968

-- Definitions / Conditions
def U : Set ℝ := Set.univ
def M : Set ℝ := { x | x < 1 }
def N : Set ℝ := { x | -1 < x ∧ x < 2 }

-- Statement of the theorem
theorem complement_union_eq {x : ℝ} :
  {x | x ≥ 2} = (U \ (M ∪ N)) := sorry

end complement_union_eq_l381_381968


namespace max_students_per_class_l381_381538

theorem max_students_per_class (num_students : ℕ) (seats_per_bus : ℕ) (num_buses : ℕ) (k : ℕ) 
  (h_num_students : num_students = 920) 
  (h_seats_per_bus : seats_per_bus = 71) 
  (h_num_buses : num_buses = 16) 
  (h_class_size_bound : ∀ c, c ≤ k) : 
  k = 17 :=
sorry

end max_students_per_class_l381_381538


namespace shaded_fraction_correct_l381_381158

-- Definitions based on conditions
def Rectangle : Type := ℝ × ℝ  -- Representing width and height of the rectangle

def leftStrip (r : Rectangle) : Rectangle := (r.1 / 2, r.2) -- Left strip half width
def rightStrip (r : Rectangle) : Rectangle := (r.1 / 2, r.2) -- Right strip half width

def leftStripShadedFraction (r : Rectangle) : ℚ := 2 / 3  -- Two out of three parts shaded
def rightStripShadedFraction (r : Rectangle) : ℚ := 1 / 2  -- Two out of four parts shaded

-- Shaded fraction of the entire rectangle
def totalShadedFraction (r : Rectangle) : ℚ := 
  leftStripShadedFraction r * 1 / 2 + rightStripShadedFraction r * 1 / 2

-- Theorem to prove the shaded fraction is 7/12
theorem shaded_fraction_correct (r : Rectangle) : totalShadedFraction r = 7 / 12 :=
by {
  -- The proof would go here
  sorry
}

end shaded_fraction_correct_l381_381158


namespace range_of_m_if_doubling_function_l381_381379

noncomputable def f (x m : ℝ) : ℝ := real.log (real.exp x + m)

theorem range_of_m_if_doubling_function :
  (∃ a b : ℝ, a < b ∧
    ∀ x ∈ set.Icc a b, 2 * a ≤ f x ∧ f x ≤ 2 * b) →
  -1 / 4 < m ∧ m < 0 :=
by sorry

end range_of_m_if_doubling_function_l381_381379


namespace generalized_schur_inequality_l381_381611

theorem generalized_schur_inequality (t : ℝ) (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a^t * (a - b) * (a - c) + b^t * (b - c) * (b - a) + c^t * (c - a) * (c - b) ≥ 0 :=
sorry

end generalized_schur_inequality_l381_381611


namespace complement_of_M_l381_381003

def M : Set ℝ := {x | x^2 - 2 * x > 0}

def U : Set ℝ := Set.univ

theorem complement_of_M :
  (U \ M) = (Set.Icc 0 2) :=
by
  sorry

end complement_of_M_l381_381003


namespace constant_term_expansion_l381_381053

theorem constant_term_expansion :
  (∏ i in (range 6), (x - x⁻¹)) = -20 := sorry

end constant_term_expansion_l381_381053


namespace cube_new_surface_area_l381_381502

theorem cube_new_surface_area 
  (original_edge_length : ℝ) 
  (increase_percentage : ℝ)
  (new_surface_area : ℝ) 
  (h1 : original_edge_length = 7)
  (h2 : increase_percentage = 0.15)
  (h3 : new_surface_area = 388.815) :
  let new_edge_length := original_edge_length * (1 + increase_percentage) in
  let surface_area := 6 * (new_edge_length ^ 2) in
  surface_area = new_surface_area := by
{
  -- By defining new_edge_length and surface_area in terms of the given conditions,
  -- we will eventually prove surface_area = new_surface_area.
  sorry
}

end cube_new_surface_area_l381_381502


namespace greatest_y_least_y_greatest_integer_y_l381_381587

theorem greatest_y (y : ℤ) (H : (8 : ℝ) / 11 > y / 17) : y ≤ 12 :=
sorry

theorem least_y (y : ℤ) (H : (8 : ℝ) / 11 > y / 17) : y ≥ 12 :=
sorry

theorem greatest_integer_y : ∀ (y : ℤ), ((8 : ℝ) / 11 > y / 17) → y = 12 :=
by
  intro y H
  apply le_antisymm
  apply greatest_y y H
  apply least_y y H

end greatest_y_least_y_greatest_integer_y_l381_381587


namespace unique_function_l381_381283

noncomputable def f (x : ℝ) : ℝ := sorry -- We will prove that f(x) = A * x^(1 + sqrt(2))

theorem unique_function (A : ℝ) (f : ℝ → ℝ)
  (h_cont : ContinuousOn f (set.Ici 0)) 
  (h_pos : ∀ x > 0, 0 < f x) 
  (h_centroid : ∀ x0 > 0,
    (1 / x0) * ∫ t in 0..x0, t * f t = (1 / (x0 * ∫ t in 0..x0, f t)) * (∫ t in 0..x0, f t)^2) :
  ∃ (A : ℝ), ∀ (x : ℝ), f x = A * x^(1 + Real.sqrt 2) :=
begin
  -- Proof omitted
  sorry
end

end unique_function_l381_381283


namespace find_line_l_l381_381878

def circleO : ℝ × ℝ → Prop := λ p, (p.1^2 + p.2^2 = 4)
def circleC : ℝ × ℝ → Prop := λ p, (p.1^2 + p.2^2 + 4 * p.1 - 4 * p.2 + 4 = 0)
def symmetric_wrt_line (C1 C2 : ℝ × ℝ → Prop) (l : ℝ × ℝ → Prop) : Prop := 
  ∀ p, C1 p ↔ C2 (2 * line_midpoint p - p) where
    line_midpoint (q : ℝ × ℝ) : ℝ × ℝ := (-1, 1)

theorem find_line_l :
  symmetric_wrt_line circleO circleC (λ p, p.1 - p.2 + 2 = 0) :=
sorry

end find_line_l_l381_381878


namespace who_wears_which_dress_l381_381730

-- Define the possible girls
inductive Girl
| Katya | Olya | Liza | Rita
deriving DecidableEq

-- Define the possible dresses
inductive Dress
| Pink | Green | Yellow | Blue
deriving DecidableEq

-- Define the fact that each girl is wearing a dress
structure Wearing (girl : Girl) (dress : Dress) : Prop

-- Define the conditions
theorem who_wears_which_dress :
  (¬ Wearing Girl.Katya Dress.Pink ∧ ¬ Wearing Girl.Katya Dress.Blue) ∧
  (∀ g1 g2 g3, Wearing g1 Dress.Green → (Wearing g2 Dress.Pink ∧ Wearing g3 Dress.Yellow → (g2 = Girl.Liza ∧ (g3 = Girl.Rita)) ∨ (g3 = Girl.Liza ∧ g2 = Girl.Rita))) ∧
  (¬ Wearing Girl.Rita Dress.Green ∧ ¬ Wearing Girl.Rita Dress.Blue) ∧
  (∀ g1 g2, (Wearing g1 Dress.Pink ∧ Wearing g2 Dress.Yellow) → Girl.Olya = g2 ∧ Girl.Rita = g1) →
  (Wearing Girl.Katya Dress.Green ∧ Wearing Girl.Olya Dress.Blue ∧ Wearing Girl.Liza Dress.Pink ∧ Wearing Girl.Rita Dress.Yellow) :=
by
  sorry

end who_wears_which_dress_l381_381730


namespace ways_to_tile_200x3_divisible_by_3_l381_381157

-- Define the function f that counts ways to tile nx3 grid with 2x1 tiles
def f : ℕ → ℕ
| 0         := 1
| (2*(n+1)) := f (2*n) + 2 * (Finset.range (n+1)).sum (λ k => f (2*k))
| (2*n + 1) := 0  -- defined this way to avoid issues with odd inputs

-- Define the condition that f(200) is divisible by 3
theorem ways_to_tile_200x3_divisible_by_3 : f 200 % 3 = 0 :=
sorry

end ways_to_tile_200x3_divisible_by_3_l381_381157


namespace equilibrium_force_l381_381359

def f1 : ℝ × ℝ := (-2, -1)
def f2 : ℝ × ℝ := (-3, 2)
def f3 : ℝ × ℝ := (4, -3)
def expected_f4 : ℝ × ℝ := (1, 2)

theorem equilibrium_force :
  (1, 2) = -(f1 + f2 + f3) := 
by
  sorry

end equilibrium_force_l381_381359


namespace factorization_l381_381272

def f (x y z : ℝ) : ℝ :=
  (y^2 - z^2) * (1 + x * y) * (1 + x * z) + 
  (z^2 - x^2) * (1 + y * z) * (1 + x * y) + 
  (x^2 - y^2) * (1 + y * z) * (1 + x * z)

theorem factorization (x y z : ℝ) :
  f x y z = (y - z) * (z - x) * (x - y) * (x * y * z + x + y + z) :=
by 
  sorry

end factorization_l381_381272


namespace sum_abc_l381_381371

variable (a b c : ℝ)

-- Conditions given in the problem
def conditions : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ 
  a * b = 30 ∧ 
  a * c = 60 ∧ 
  b * c = 90

-- The proof goal
theorem sum_abc : conditions a b c → a + b + c = 11 * Real.sqrt 5 := 
by
  intros cond,
  sorry

end sum_abc_l381_381371


namespace time_to_assemble_l381_381455

def assembly_time (x: ℝ) : Prop :=
  let usual_baking_time := 1.5
      decorating_time := 1.0
      failed_baking_time := 2 * usual_baking_time
      total_time := 5.0
  in x + failed_baking_time + decorating_time = total_time

theorem time_to_assemble : assembly_time 1 :=
by
  unfold assembly_time
  simp
  norm_num
  sorry

end time_to_assemble_l381_381455


namespace probability_third_smallest_is_five_l381_381699

open Finset

noncomputable def prob_third_smallest_is_five : ℚ :=
  let total_ways := choose 15 8
  let favorable_ways := (choose 4 2) * (choose 10 5)
  in favorable_ways / total_ways

theorem probability_third_smallest_is_five :
  prob_third_smallest_is_five = 72 / 307 :=
by sorry

end probability_third_smallest_is_five_l381_381699


namespace smallest_period_f_intervals_monotonic_increase_max_min_f_interval_l381_381347

noncomputable def f (x : ℝ) : ℝ := (Real.sin x + Real.cos x) ^ 2 + 2 * (Real.cos x) ^ 2 - 2

theorem smallest_period_f : Real.Periodic f Real.pi :=
sorry

theorem intervals_monotonic_increase (k : ℤ) : 
  ∀ x, - (3 * Real.pi / 8) + k * Real.pi ≤ x ∧ x ≤ (Real.pi / 8) + k * Real.pi → Real.Derivative (f x) ≥ 0 :=
sorry

theorem max_min_f_interval : 
  ∃ (x_max x_min: ℝ), (Real.pi / 4) ≤ x_max ∧ x_max ≤ (3 * Real.pi / 4) ∧ f x_max = 1 ∧
  (Real.pi / 4) ≤ x_min ∧ x_min ≤ (3 * Real.pi / 4) ∧ f x_min = -Real.sqrt 2 :=
sorry

end smallest_period_f_intervals_monotonic_increase_max_min_f_interval_l381_381347


namespace arithmetic_sequence_sum_l381_381321

-- Define the given conditions
variables {a : ℕ → ℤ} (S : ℕ → ℤ)
def is_arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ (a1 d : ℤ), ∀ n, a n = a1 + n * d

def sum_first_n_terms (a : ℕ → ℤ) (S : ℕ → ℤ) : Prop :=
   ∀ n, S n = n / 2 * (2 * a 0 + (n - 1) * (a 1 - a 0))

lemma sum_condition_S3 (S : ℕ → ℤ) (h1 : S 3 = 0) : True := by trivial
lemma sum_condition_S5 (S : ℕ → ℤ) (h2 : S 5  = -5) : True := by trivial

theorem arithmetic_sequence_sum
{a : ℕ → ℤ} {S : ℕ → ℤ}
(h1 : is_arithmetic_sequence a)
(h2 : sum_first_n_terms a S)
(h3 : S 3 = 0) 
(h4 : S 5 = -5) :
(a 1 = 2 - 1) ∧
∀ n, (a n = -n + 2) ∧
(∀ n, ∑ i in finset.range n, a i / 2 ^ i = n / 2 ^ n) :=
sorry

end arithmetic_sequence_sum_l381_381321


namespace integral_f_eq_e_sub_one_l381_381533

open Real
open ComplexConjugate

-- Define the integrand function
def f (x : ℝ) : ℝ := x^2 + exp x - 1/3

-- Prove that the value of the definite integral of f from 0 to 1 is e - 1
theorem integral_f_eq_e_sub_one : 
  ∫ x in 0..1, f x = exp 1 - 1 :=
by
  sorry

end integral_f_eq_e_sub_one_l381_381533


namespace complex_conjugate_l381_381327

noncomputable def conj_z_solution : Complex :=
  conj (Complex.of_real (1 / 2) + Complex.I * (1 / 2))

theorem complex_conjugate (z: Complex) (i: Complex) (h: i = Complex.I)
  (h1: z * (1 + i) = 1) : conj z = conj_z_solution :=
by
  sorry

end complex_conjugate_l381_381327


namespace least_four_digit_solution_l381_381290

theorem least_four_digit_solution :
  ∃ x : ℕ, (x = 1011) ∧ 
           (5 * x ≡ 15 [MOD 20]) ∧ 
           (3 * x + 10 ≡ 19 [MOD 7]) ∧ 
           (-3 * x + 4 ≡ 2 * x [MOD 16]) :=
by
  sorry

end least_four_digit_solution_l381_381290


namespace unread_pages_when_a_is_11_l381_381134

variable (a : ℕ)

def total_pages : ℕ := 250
def pages_per_day : ℕ := 15

def unread_pages_after_a_days (a : ℕ) : ℕ := total_pages - pages_per_day * a

theorem unread_pages_when_a_is_11 : unread_pages_after_a_days 11 = 85 :=
by
  sorry

end unread_pages_when_a_is_11_l381_381134


namespace factor_expression_l381_381270

-- Define f(x, y, z) based on the given expression
def f (x y z : ℝ) : ℝ :=
  (y^2 - z^2) * (1 + x * y) * (1 + x * z) +
  (z^2 - x^2) * (1 + y * z) * (1 + x * y) +
  (x^2 - y^2) * (1 + y * z) * (1 + x * z)

-- Define the factored form of the expression
def factored_f (x y z : ℝ) : ℝ :=
  (y - z) * (z - x) * (x - y) * (x * y * z + x + y + z)

-- Prove that f(x, y, z) is equivalent to its factored form
theorem factor_expression (x y z : ℝ) : f(x, y, z) = factored_f(x, y, z) :=
  sorry

end factor_expression_l381_381270


namespace angle_AHB_correct_l381_381183

noncomputable def angle_AHB (BAC ABC : ℕ) : ℕ :=
  if h1: BAC = 40 ∧ ABC = 65 then
    105
  else
    sorry -- Other values have not been considered.

-- Now we state the theorem that asserts our problem.
theorem angle_AHB_correct (BAC ABC : ℕ) (BAC_eq : BAC = 40) (ABC_eq : ABC = 65) :
    angle_AHB BAC ABC = 105 :=
  by
    rw [angle_AHB]
    split_ifs
    apply rfl

end angle_AHB_correct_l381_381183


namespace different_total_scores_l381_381628

noncomputable def basket_scores (x y z : ℕ) : ℕ := x + 2 * y + 3 * z

def total_baskets := 7
def score_range := {n | 7 ≤ n ∧ n ≤ 21}

theorem different_total_scores : 
  ∃ (count : ℕ), count = 15 ∧ 
  ∀ n ∈ score_range, ∃ (x y z : ℕ), x + y + z = total_baskets ∧ basket_scores x y z = n :=
sorry

end different_total_scores_l381_381628


namespace sum_logarithms_l381_381234

theorem sum_logarithms :
  (∑ k in finset.Ico 3 101, real.logb 3 (1 + 1/(k:ℝ)) * real.logb k 3 * real.logb (k+1) 3) =
  1 - real.logb 2 3 / real.logb 2 101 :=
by
  sorry

end sum_logarithms_l381_381234


namespace rational_inequalities_l381_381801

theorem rational_inequalities (a b c d : ℚ)
  (h : a^3 - 2005 = b^3 + 2027 ∧ b^3 + 2027 = c^3 - 2822 ∧ c^3 - 2822 = d^3 + 2820) :
  c > a ∧ a > b ∧ b > d :=
by
  sorry

end rational_inequalities_l381_381801


namespace typhoon_probabilities_l381_381462

-- Defining the conditions
def probAtLeastOneHit : ℝ := 0.36

-- Defining the events and probabilities
def probOfHit (p : ℝ) := p
def probBothHit (p : ℝ) := p^2

def probAtLeastOne (p : ℝ) : ℝ := p^2 + 2 * p * (1 - p)

-- Defining the variable X as the number of cities hit by the typhoon
def P_X_0 (p : ℝ) : ℝ := (1 - p)^2
def P_X_1 (p : ℝ) : ℝ := 2 * p * (1 - p)
def E_X (p : ℝ) : ℝ := 2 * p

-- Main theorem
theorem typhoon_probabilities :
  ∀ (p : ℝ),
    probAtLeastOne p = probAtLeastOneHit → 
    p = 0.2 ∧ P_X_0 p = 0.64 ∧ P_X_1 p = 0.32 ∧ E_X p = 0.4 :=
by
  intros p h
  sorry

end typhoon_probabilities_l381_381462


namespace count_15_letter_arrangements_l381_381844

-- Define the combinatorial setup
def countArrangements : ℕ := 
  ∑ l in Finset.range 5, ∑ m in Finset.range 6, (Nat.choose 4 l) * (Nat.choose 6 m) * (Nat.choose 6 m)

-- Define the main theorem
theorem count_15_letter_arrangements :
  countArrangements = ∑ l in Finset.range 5, ∑ m in Finset.range 6, (Nat.choose 4 l) * (Nat.choose 6 m) * (Nat.choose 6 m) := 
by
  sorry

end count_15_letter_arrangements_l381_381844


namespace find_b_for_continuous_function_l381_381018

theorem find_b_for_continuous_function (b : ℝ) :
  (∀ x : ℝ, if x ≤ 5 then 4 * x ^ 2 + 3 else b * x + 2) 5 = 103 → b = 20.2 :=
by
  intro h
  sorry

end find_b_for_continuous_function_l381_381018


namespace hunter_wins_l381_381642

noncomputable def winning_strategy_for_hunter_exists : Prop :=
  ∃ (C₁ C₂ C₃ C₄ C₅ : ℕ × ℕ → ℕ),
    (∀ x y, C₁ (x, y) = x % 3 ∧ C₂ (x, y) = y % 3) ∧
    (∀ x y, C₃ (x, y) ∈ ({0, 1}: Set ℕ) ∧ C₄ (x, y) ∈ ({0, 1}: Set ℕ) ∧ C₅ (x, y) ∈ ({0, 1}: Set ℕ)) ∧
    ∃ (strategy : (ℕ × ℕ) → (ℕ → ℕ × ℕ)), -- Mapping of time to cell movements
      (∀ t, strategy t = (x, y) -> adjacent (strategy t) (strategy (t+1))) ∧ 
      (cannot_move_or_determine_start strategy)

-- Auxiliary assumptions and definitions must be added to model the 
-- conditions exactly, such as defining adjacent and cannot_move_or_determine_start.
def adjacent (A B : ℕ × ℕ) : Prop := 
  (A.1 = B.1 ∧ (A.2 = B.2 + 1 ∨ A.2 = B.2 - 1)) ∨
  (A.2 = B.2 ∧ (A.1 = B.1 + 1 ∨ A.1 = B.1 - 1))

-- Definition to determine if the rabbit cannot move or the hunter can determine the starting point.
def cannot_move_or_determine_start (strategy : ℕ → ℕ × ℕ) : Prop := sorry

theorem hunter_wins : winning_strategy_for_hunter_exists := sorry

end hunter_wins_l381_381642


namespace find_annual_interest_rate_l381_381417

noncomputable def compound_interest (P A : ℝ) (r : ℝ) (n t : ℕ) :=
  A = P * (1 + r / n) ^ (n * t)

theorem find_annual_interest_rate
  (P A : ℝ) (t n : ℕ) (r : ℝ)
  (hP : P = 6000)
  (hA : A = 6615)
  (ht : t = 2)
  (hn : n = 1)
  (hr : compound_interest P A r n t) :
  r = 0.05 :=
sorry

end find_annual_interest_rate_l381_381417


namespace find_max_f_l381_381794

noncomputable def a_seq : ℕ → ℕ
| 0       => 2
| (n + 1) => 4 * a_seq n - 4 * a_seq n

noncomputable def S : ℕ → ℕ
| 0       => a_seq 0
| (n + 1) => 2 * (2 * a_seq n + 1)

noncomputable def f (n : ℕ) : ℤ :=
(a_seq n / 2^(n-1)) * (-2 * n + 31) - 1

theorem find_max_f :
  ∃ n : ℕ, 0 < n ∧ f n = 239 :=
begin
  sorry
end

end find_max_f_l381_381794


namespace average_and_stddev_l381_381498

theorem average_and_stddev (a : ℝ)
  (h_avg : (1 + 2 + 3 + 4 + a) / 5 = 3) : 
  a = 5 ∧ 
  let l := [1, 2, 3, 4, 5] in
  let mean := (l.sum : ℝ) / l.length in
  let variance := (l.map (λ x, (x - mean) ^ 2)).sum / l.length in
  sqrt variance = sqrt 2 :=
by
  sorry

end average_and_stddev_l381_381498


namespace unique_solution_xyz_l381_381714

theorem unique_solution_xyz :
  ∀ (x y z : ℕ), x > 0 → y > 0 → z > 0 → prime y → ¬(3 ∣ z) → ¬(y ∣ z) →
  (x^3 - y^3 = z^2) ↔ (x, y, z) = (8, 7, 13) :=
by
  sorry

end unique_solution_xyz_l381_381714


namespace max_angle_A1MC1_l381_381396

theorem max_angle_A1MC1 (a : ℝ) (h_pos : 0 < a):
  let h := a / 2,
      M : ℝ := 1,
      angle_A1MC1 := real.pi / 2 in
  angle_A1MC1 = real.pi / 2 :=
by
  sorry

end max_angle_A1MC1_l381_381396


namespace ferry_total_tourists_l381_381143

-- Define the parameters for the arithmetic sequence.
def initial_tourists : ℕ := 100
def decrement_per_trip : ℕ := 2
def total_trips : ℕ := 7

-- Define a function to calculate the number of tourists on the i-th trip.
def tourists_on_trip (i : ℕ) : ℕ :=
  initial_tourists - decrement_per_trip * (i - 1)

-- Define a function to calculate the sum of tourists using the arithmetic series formula.
def total_tourists : ℕ :=
  let a := initial_tourists
  let d := decrement_per_trip
  let n := total_trips
  n * (2 * a + (n - 1) * -d) / 2

theorem ferry_total_tourists : total_tourists = 658 := by
  sorry

end ferry_total_tourists_l381_381143


namespace valid_grid_sizes_l381_381792

def valid_coloring (m n : ℕ) (coloring : ℕ → ℕ → Prop) :=
  (∀ i j, (i = 0 ∨ i = m - 1 ∨ j = 0 ∨ j = n - 1) → coloring i j = tt) ∧
  (∀ i j, i + 1 < m ∧ j + 1 < n → ¬ (coloring i j = coloring (i + 1) j ∧ coloring (i + 1) j = coloring i (j + 1) ∧ coloring i (j + 1) = coloring (i + 1) (j + 1))) ∧
  (∀ i j, i + 1 < m ∧ j + 1 < n → ¬ (coloring i j = coloring (i + 1) (j + 1) ∧ coloring (i + 1) j = coloring i (j + 1))) 

theorem valid_grid_sizes (m n : ℕ) (hm : m ≥ 3) (hn : n ≥ 3) :
  (∃ coloring : ℕ → ℕ → Prop, valid_coloring m n coloring) ↔ (m % 2 = 1 ∨ n % 2 = 1) :=
by sorry

end valid_grid_sizes_l381_381792


namespace min_distance_from_origin_to_line_l381_381466

theorem min_distance_from_origin_to_line : 
  let d := ∀ x y : ℝ, (x + y = 4) → ℝ 
  in min (fun d => let x := d.1; let y := d.2 in Real.sqrt (x^2 + y^2)) = 2 * Real.sqrt 2 :=
sorry

end min_distance_from_origin_to_line_l381_381466


namespace fraction_of_is_l381_381564

theorem fraction_of_is (a b c d e : ℚ) (h1 : a = 2) (h2 : b = 9) (h3 : c = 3) (h4 : d = 4) (h5 : e = 8/27) :
  (a / b) = e * (c / d) := 
sorry

end fraction_of_is_l381_381564


namespace min_rectangle_area_l381_381159

theorem min_rectangle_area : 
  ∃ (x y : ℕ), 2 * (x + y) = 80 ∧ x * y = 39 :=
by
  sorry

end min_rectangle_area_l381_381159


namespace solve_system_l381_381484

theorem solve_system :
  ∃ x y : ℚ, (4 * x - 7 * y = -20) ∧ (9 * x + 3 * y = -21) ∧ (x = -69 / 25) ∧ (y = 32 / 25) := by
  sorry

end solve_system_l381_381484


namespace cheryl_bill_cost_correct_l381_381212

def cheryl_electricity_bill_cost : Prop :=
  ∃ (E : ℝ), 
    (E + 400) + 0.20 * (E + 400) = 1440 ∧ 
    E = 800

theorem cheryl_bill_cost_correct : cheryl_electricity_bill_cost :=
by
  sorry

end cheryl_bill_cost_correct_l381_381212


namespace part1_part2_l381_381799

-- Define vectors a, b, and c
def vec_a := (1, 3)
def vec_b (x : ℤ) := (2 * x - 1, -x)
def vec_c := (7, -1)

-- Part 1: x such that 2a + b is perpendicular to a - 2b
theorem part1 (x : ℤ) : 
  let u := (2 * vec_a.1 + vec_b x.1, 2 * vec_a.2 + vec_b x.2) in
  let v := (vec_a.1 - 2 * vec_b x.1, vec_a.2 - 2 * vec_b x.2) in
  u.1 * v.1 + u.2 * v.2 = 0 → x = -1 := sorry

-- Part 2: magnitude of a - b given collinearity condition
theorem part2 (x : ℤ) (H : vec_a.1 + vec_b x.1 ≠ 0 ∨ vec_a.2 + vec_b x.2 ≠ 0) : 
  let u := (vec_a.1 + vec_b x.1, vec_a.2 + vec_b x.2) in
  let v := (vec_b x.1 - vec_c.1, vec_b x.2 - vec_c.2) in
  u.1 * v.2 = u.2 * v.1 → ∥(vec_a.1 - vec_b x.1, vec_a.2 - vec_b x.2)∥ = Real.sqrt 29 := sorry

end part1_part2_l381_381799


namespace question_proof_l381_381981

open Set

variable (U : Set ℝ := univ)
variable (M : Set ℝ := {x | x < 1})
variable (N : Set ℝ := {x | -1 < x ∧ x < 2})

theorem question_proof : {x | x ≥ 2} = compl (M ∪ N) :=
by
  sorry

end question_proof_l381_381981


namespace solution_system_l381_381866

theorem solution_system (x y : ℝ) (h1 : x * y = 8) (h2 : x^2 * y + x * y^2 + x + y = 80) :
  x^2 + y^2 = 5104 / 81 := by
  sorry

end solution_system_l381_381866


namespace number_of_managers_l381_381389

-- Definitions based on conditions
def manager_salary : ℕ := 5
def clerk_salary : ℕ := 2
def clerks_count : ℕ := 3
def total_salary : ℕ := 16

-- Statement of the problem to be proved
theorem number_of_managers :
  ∃ M : ℕ, (clerks_count * clerk_salary + M * manager_salary = total_salary) ∧ M = 2 :=
begin
  sorry
end

end number_of_managers_l381_381389


namespace sum_cos_frac_pi_eq_zero_l381_381189

theorem sum_cos_frac_pi_eq_zero :
  let fractions := [1/24, 5/24, 7/24, 11/24, 13/24, 17/24, 19/24, 23/24] in
  ∑ i in fractions, Real.cos (i * Real.pi) = 0 :=
by
  sorry

end sum_cos_frac_pi_eq_zero_l381_381189


namespace lady_bird_flour_needed_l381_381421

/-- Lady Bird uses 1.25 cups of flour to make 9 biscuits. 
    She is hosting 18 members, each getting 2 biscuits. 
    Prove that the total amount of flour needed is 5 cups. --/
theorem lady_bird_flour_needed :
  let cups_per_batch := (5 : ℝ) / 4,
      biscuits_per_batch := 9,
      guests := 18,
      biscuits_per_guest := 2,
      total_biscuits_needed := guests * biscuits_per_guest,
      batches_needed := total_biscuits_needed / biscuits_per_batch,
      total_flour_needed := batches_needed * cups_per_batch
  in total_flour_needed = 5 := by 
  sorry

end lady_bird_flour_needed_l381_381421


namespace find_x_parallel_l381_381361

def m : ℝ × ℝ := (-2, 4)
def n (x : ℝ) : ℝ × ℝ := (x, -1)

def parallel (u v : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ u.1 = k * v.1 ∧ u.2 = k * v.2

theorem find_x_parallel :
  parallel m (n x) → x = 1 / 2 := by 
sorry

end find_x_parallel_l381_381361


namespace ahmed_total_distance_l381_381655

theorem ahmed_total_distance (d : ℝ) (h : (3 / 4) * d = 12) : d = 16 := 
by 
  sorry

end ahmed_total_distance_l381_381655


namespace candy_store_problem_l381_381387

variable (S : ℝ)
variable (not_caught_percentage : ℝ) (sample_percentage : ℝ)
variable (caught_percentage : ℝ := 1 - not_caught_percentage)

theorem candy_store_problem
  (h1 : not_caught_percentage = 0.15)
  (h2 : sample_percentage = 25.88235294117647) :
  caught_percentage * sample_percentage = 22 := by
  sorry

end candy_store_problem_l381_381387


namespace polynomial_exists_int_coeff_l381_381127

theorem polynomial_exists_int_coeff (n : ℕ) (hn : n > 1) : 
  ∃ P : Polynomial ℤ × Polynomial ℤ × Polynomial ℤ → Polynomial ℤ, 
  ∀ x : Polynomial ℤ, P ⟨x^n, x^(n+1), x + x^(n+2)⟩ = x :=
by sorry

end polynomial_exists_int_coeff_l381_381127


namespace min_product_condition_l381_381312

noncomputable def point_on_line_minimizing_product (A B : Point) (ℓ : Line) (h_parallel : Parallel ℓ (Line_through A B)) : Point :=
sorry  -- This definition should return the intersection of the perpendicular bisector of AB and ℓ

theorem min_product_condition (A B : Point) (ℓ : Line) (h_parallel : Parallel ℓ (Line_through A B)) :
  let C := point_on_line_minimizing_product A B ℓ h_parallel in
  OnLine C ℓ → (∀ D : Point, OnLine D ℓ → (A.dist D * B.dist D) ≥ (A.dist C * B.dist C)) :=
by
  sorry

end min_product_condition_l381_381312


namespace root_in_interval_l381_381515

noncomputable def f (x : ℝ) : ℝ := x^2 + log x - 3

theorem root_in_interval :
  ∃ (c : ℝ), (c ∈ Ioo (3 / 2 : ℝ) (2 : ℝ)) ∧ f c = 0 :=
by
  sorry

end root_in_interval_l381_381515


namespace second_hand_distance_l381_381138

def radius : ℝ := 10
def time_minutes : ℝ := 15
def distance_in_cm (radius time_minutes : ℝ) : ℝ := 2 * time_minutes * radius * π

theorem second_hand_distance :
  distance_in_cm radius time_minutes = 300 * π := 
sorry

end second_hand_distance_l381_381138


namespace max_area_triangle_ABC_l381_381405

open scoped real

variables {A B C P Q : Type} 
variables [metric_space A] [metric_space B] [metric_space C] [metric_space P] [metric_space Q]
variables (A B C P Q : ℝ)

noncomputable def PA := 3
noncomputable def PB := 4
noncomputable def PC := 5
noncomputable def BC := 6
noncomputable def PQ := 2
noncomputable def QB := 1

theorem max_area_triangle_ABC : (∀ (PA PB PC BC PQ QB : ℝ), PA = 3 ∧ PB = 4 ∧ PC = 5 ∧ BC = 6 ∧ PQ = 2 ∧ QB = 1 → 
                                ∃ (A B C P Q : Type), [⟨A,B,C⟩] = 19 := 
begin
  assume h,
  sorry
end

end max_area_triangle_ABC_l381_381405


namespace binomial_distribution_n_value_l381_381005

open ProbabilityTheory

theorem binomial_distribution_n_value (n p : ℝ) (ξ : ℕ → bool) :
  (E ξ = 12) ∧ (V ξ = 4) ∧ (ξ follows binomial_distribution n p) → n = 18 :=
by
  sorry

end binomial_distribution_n_value_l381_381005


namespace question_proof_l381_381989

open Set

variable (U : Set ℝ := univ)
variable (M : Set ℝ := {x | x < 1})
variable (N : Set ℝ := {x | -1 < x ∧ x < 2})

theorem question_proof : {x | x ≥ 2} = compl (M ∪ N) :=
by
  sorry

end question_proof_l381_381989


namespace point_in_fourth_quadrant_l381_381340

open Real

theorem point_in_fourth_quadrant (α : ℝ) (hα : π < α ∧ α < 3 * π / 2 ) :
  let P := (tan α, cos α) in P.1 > 0 ∧ P.2 < 0 :=
  by
  sorry

end point_in_fourth_quadrant_l381_381340


namespace log2_function_domain_l381_381686

noncomputable def log2_domain : set ℝ :=
  { x : ℝ | (0 < x ∧ x < 1/2) ∨ (2 < x) }

theorem log2_function_domain :
  ∀ x : ℝ, (x > 0 ∧ (log 2 x) ^ 2 - 1 > 0) ↔ x ∈ log2_domain := by
  sorry

end log2_function_domain_l381_381686


namespace max_edges_in_8_points_graph_no_square_l381_381640

open Finset

-- Define what a graph is and the properties needed for the problem
structure Graph (V : Type*) :=
  (edges : Finset (V × V))
  (sym : ∀ {x y : V}, (x, y) ∈ edges ↔ (y, x) ∈ edges)
  (irrefl : ∀ {x : V}, ¬ (x, x) ∈ edges)

-- Define the conditions of the problem
def no_square {V : Type*} (G : Graph V) : Prop :=
  ∀ (a b c d : V), 
    (a, b) ∈ G.edges → (b, c) ∈ G.edges → (c, d) ∈ G.edges → (d, a) ∈ G.edges →
    (a, c) ∈ G.edges → (b, d) ∈ G.edges → False

-- Define 8 vertices
inductive Vertices
| A | B | C | D | E | F | G | H

-- Define the number of edges
noncomputable def max_edges_no_square : ℕ :=
  11

-- Define the final theorem
theorem max_edges_in_8_points_graph_no_square :
  ∃ (G : Graph Vertices), 
    no_square G ∧ (G.edges.card = max_edges_no_square) :=
sorry

end max_edges_in_8_points_graph_no_square_l381_381640


namespace probability_of_interval_l381_381499

-- Define the probability density function f(x) as given in the problem.
noncomputable def f (x : Real) : Real :=
  if x > 0 ∧ x < Real.pi / 3 then 3 / 2 * Real.sin (3 * x) else 0

-- Define the interval bounds a and b
def a : Real := Real.pi / 6
def b : Real := Real.pi / 4

-- State the theorem
theorem probability_of_interval : (intervalIntegral (a:=a) (b:=b) f) = Real.sqrt 2 / 4 := sorry

end probability_of_interval_l381_381499


namespace max_profit_achieved_at_optimal_price_l381_381136

noncomputable def profit_function (x : ℕ) : ℕ :=
  -100 * (x - 3) ^ 2 + 6400

theorem max_profit_achieved_at_optimal_price :
  ∃ (x : ℕ), 0 ≤ x ∧ x ≤ 11.5 ∧ profit_function x = 6400 := 
begin
  use 3,
  split,
  { linarith, },
  split,
  { linarith, },
  { simp [profit_function], },
  sorry
end

end max_profit_achieved_at_optimal_price_l381_381136


namespace perfect_square_divisors_count_l381_381366

/-- Define the product of factorials from 1! to 10! -/
def factorial_product : ℕ := ∏ i in (finset.range 11).map finset.succ, nat.factorial i

/-- Statement asserting the number of perfect square divisors of the product 1! * 2! * ... * 10! -/
theorem perfect_square_divisors_count :
  let P := factorial_product in
  (perfect_square_divisors P) = 1440 :=
by
  sorry

end perfect_square_divisors_count_l381_381366


namespace find_fake_coin_l381_381548

def coin_value (n : Nat) : Nat :=
  match n with
  | 1 => 1
  | 2 => 2
  | 3 => 3
  | 4 => 5
  | _ => 0

def coin_weight (n : Nat) : Nat :=
  match n with
  | 1 => 1
  | 2 => 2
  | 3 => 3
  | 4 => 5
  | _ => 0

def is_fake (weight : Nat) : Prop :=
  weight ≠ coin_weight 1 ∧ weight ≠ coin_weight 2 ∧ weight ≠ coin_weight 3 ∧ weight ≠ coin_weight 4

theorem find_fake_coin :
  ∃ (n : Nat) (w : Nat), (is_fake w) → ∃! (m : Nat), m ≠ w ∧ (m = coin_weight 1 ∨ m = coin_weight 2 ∨ m = coin_weight 3 ∨ m = coin_weight 4) := 
sorry

end find_fake_coin_l381_381548


namespace tangent_excircle_l381_381329

-- Define the geometric entities as per the Lean 4 structure
noncomputable def circle (center : Point) (radius : ℝ) := 
  { p : Point | dist p center = radius }

variables (A B C D K : Point)
variables (Γ : circle)
variables (Γ₁ : circle)

-- Assume the relevant geometric properties and conditions given in the problem
axiom diameter_AB : (A ≠ B) ∧ (AB = diameter Γ)
axiom point_on_circle : (C ∈ Γ) ∧ (C ≠ A) ∧ (C ≠ B)
axiom projection_D : (is_projection C AB D)
axiom point_K_segment : (K ∈ segment C D)
axiom semiperimeter_AC : (AC = semiperimeter ADK)

-- The theorem statement
theorem tangent_excircle :
  tangent_circle_excircle ADK A Γ₁ → tangent Γ Γ₁ :=
sorry

end tangent_excircle_l381_381329


namespace sin_neg_120_eq_l381_381223

def angle1 := -120
def angle2 := 240
def point := (-1 / 2, -Real.sqrt 3 / 2)

theorem sin_neg_120_eq :
  ∠ angle1 = angle2 ∧ ∃ coords, coords = point -> Real.sin angle1 = -Real.sqrt 3 / 2 :=
by
  sorry

end sin_neg_120_eq_l381_381223


namespace sum_abs_diff_range_l381_381785

theorem sum_abs_diff_range {a : Fin 2019 → ℝ}
  (h : ∀ i, 0 ≤ a i ∧ a i ≤ 2) :
  0 ≤ ∑ i in Finset.range 2019, ∑ j in Finset.range 2019, (if i < j then  abs (a i - a j) else 0) ∧
  ∑ i in Finset.range 2019, ∑ j in Finset.range 2019, (if i < j then abs (a i - a j) else 0) ≤ 2038180 :=
sorry

end sum_abs_diff_range_l381_381785


namespace true_propositions_correct_l381_381055

theorem true_propositions_correct :
  let p1 := ∀ a b : ℝ, a ≤ b → ¬(a < b)
  let p2 := ∀ x : ℝ, (1 : ℝ) * x^2 - x + 3 ≥ 0
  let p3 := ∀ r₁ r₂ : ℝ, r₁ = r₂ → π * r₁^2 = π * r₂^2
  let p4 := ∀ x : ℚ, x ≠ 0 → (∃ a b : ℕ, sqrt 2 * x = a/b) → ¬ irrational x
  p1 ∧ p2 ∧ p3 ∧ ¬p4 :=
by
  sorry

end true_propositions_correct_l381_381055


namespace count_houses_with_neither_feature_l381_381608

theorem count_houses_with_neither_feature :
  let (total : Nat) := 85 in
  let (garages : Nat) := 50 in
  let (pools : Nat) := 40 in
  let (both : Nat) := 35 in
  let (either : Nat) := garages + pools - both in
  let (neither : Nat) := total - either in
  neither = 30 :=
by
  sorry

end count_houses_with_neither_feature_l381_381608


namespace bookmarks_count_at_end_of_march_l381_381707

theorem bookmarks_count_at_end_of_march : 
  let daily_bookmarks := 30
  let current_bookmarks := 400
  let days_in_march := 31
  let total_bookmarks_in_march := daily_bookmarks * days_in_march
  let total_bookmarks_end_of_march := current_bookmarks + total_bookmarks_in_march
  in total_bookmarks_end_of_march = 1330 := by
  sorry

end bookmarks_count_at_end_of_march_l381_381707


namespace stability_measures_l381_381463

-- Define the conditions: scores from 5 recent math mock exams
variable (scores : List ℝ)

-- Define the question and correct answer as a theorem in Lean 4
theorem stability_measures (s : List ℝ) (h_len : s.length = 5) :
  (is_variance_and_range_best_measures s) :=
sorry

end stability_measures_l381_381463


namespace deepak_present_age_l381_381525

-- We start with the conditions translated into Lean definitions.

variables (R D : ℕ)

-- Condition 1: The ratio between Rahul's and Deepak's ages is 4:3.
def age_ratio := R * 3 = D * 4

-- Condition 2: After 6 years, Rahul's age will be 38 years.
def rahul_future_age := R + 6 = 38

-- The goal is to prove that D = 24 given the above conditions.
theorem deepak_present_age 
  (h1: age_ratio R D) 
  (h2: rahul_future_age R) : D = 24 :=
sorry

end deepak_present_age_l381_381525


namespace drive_feasibility_l381_381188

theorem drive_feasibility :
  ∀ (distance1 : ℝ) (distance2 : ℝ) (total_time : ℝ) (average_speed : ℝ),
    distance1 = 420 →
    distance2 = 273 →
    total_time = 11 →
    average_speed = 63 →
    (distance1 + distance2) / average_speed = total_time :=
by
  intros distance1 distance2 total_time average_speed
  intros h_distance1 h_distance2 h_total_time h_average_speed
  rw [h_distance1, h_distance2, h_total_time, h_average_speed]
  norm_num
  sorry

end drive_feasibility_l381_381188


namespace max_value_gcd_l381_381014

theorem max_value_gcd (n : ℕ) (a b c : ℕ) (h1 : a + b + c = 5 * n) (h2 : a > 0) (h3 : b > 0) (h4 : c > 0) : 
  let G := Nat.gcd a b + Nat.gcd b c + Nat.gcd c a in
  if 3 ∣ n then G = 5 * n else G = 4 * n := by
  sorry

end max_value_gcd_l381_381014


namespace maximum_possible_S_l381_381395

noncomputable def largest_S : ℕ :=
  let a : ℕ := 9
  let b : ℕ := 8
  let c : ℕ := 7
  let d : ℕ := 4
  let e : ℕ := 6
  let f : ℕ := 5
  (3 * (3 * a + 2 * b + 2 * c) = 3 * (2 * c + d + 2 * e) ∧ 3 * (2 * e + f + 2 * b)) / 3

theorem maximum_possible_S : largest_S = 40 := 
begin
  sorry
end

end maximum_possible_S_l381_381395


namespace estimate_avg_lifespan_correct_l381_381140

-- Define the conditions based on the given problem
def total_units : ℕ := 1000
def ratio : (ℕ × ℕ × ℕ) := (1, 2, 1)
def sampled_units : ℕ := 100
def avg_lifespan_first_branch : ℕ := 980
def avg_lifespan_second_branch : ℕ := 1020
def avg_lifespan_third_branch : ℕ := 1032

-- Define the number of units produced by each branch
def units_first_branch : ℕ := total_units * ratio.1 / (ratio.1 + ratio.2 + ratio.3)
def units_second_branch : ℕ := total_units * ratio.2 / (ratio.1 + ratio.2 + ratio.3)
def units_third_branch : ℕ := total_units * ratio.3 / (ratio.1 + ratio.2 + ratio.3)

-- Define the weighted mean of the average lifespans
def estimated_avg_lifespan : ℕ := 
  (units_first_branch * avg_lifespan_first_branch + units_second_branch * avg_lifespan_second_branch + units_third_branch * avg_lifespan_third_branch) / total_units

theorem estimate_avg_lifespan_correct : estimated_avg_lifespan = 1013 := by
  -- The proof would go here
  sorry

end estimate_avg_lifespan_correct_l381_381140


namespace exists_q_no_zero_in_decimal_l381_381472

theorem exists_q_no_zero_in_decimal : ∃ q : ℕ, ∀ (d : ℕ), q * 2 ^ 1967 ≠ 10 * d := 
sorry

end exists_q_no_zero_in_decimal_l381_381472


namespace volume_tetrahedron_constant_l381_381803

theorem volume_tetrahedron_constant (m n h : ℝ) (ϕ : ℝ) :
  ∃ V : ℝ, V = (1 / 6) * m * n * h * Real.sin ϕ :=
by
  sorry

end volume_tetrahedron_constant_l381_381803


namespace simplify_expression_l381_381038

theorem simplify_expression : 2023^2 - 2022 * 2024 = 1 := by
  sorry

end simplify_expression_l381_381038


namespace least_integer_gt_sqrt_750_l381_381590

theorem least_integer_gt_sqrt_750 : ∃ n : ℤ, n = 28 ∧ n > real.sqrt 750 ∧ ∀ m : ℤ, m > real.sqrt 750 → m ≥ 28 :=
by
  have h27_square : 27^2 = 729 := rfl
  have h28_square : 28^2 = 784 := rfl
  have sqrt_750_between : 729 < 750 ∧ 750 < 784 := by {
    split;
    norm_num;
  }
  use 28
  split
  rfl
  split
  calc 
    28 > sqrt 750 :=
    by {
      split;
      norm_num at *,
    }
  intros m h
  exact dec_trivial
  sorry

end least_integer_gt_sqrt_750_l381_381590


namespace problem_statement_l381_381032

-- Definitions based on problem conditions
def p (a b c : ℝ) : Prop := a > b → (a * c^2 > b * c^2)

def q : Prop := ∃ x_0 : ℝ, (x_0 > 0) ∧ (x_0 - 1 + Real.log x_0 = 0)

-- Main theorem
theorem problem_statement : (¬ (∀ a b c : ℝ, p a b c)) ∧ q :=
by sorry

end problem_statement_l381_381032


namespace distinct_real_roots_iff_l381_381342

-- Define f(x, a) := |x^2 - a| - x + 2
noncomputable def f (x a : ℝ) : ℝ := abs (x^2 - a) - x + 2

-- The proposition we need to prove
theorem distinct_real_roots_iff (a : ℝ) (h : 0 < a) : 
  (∃ (x1 x2 : ℝ), x1 ≠ x2 ∧ f x1 a = 0 ∧ f x2 a = 0) ↔ 4 < a :=
by
  sorry

end distinct_real_roots_iff_l381_381342


namespace hyperbola_focal_length_is_4_l381_381828

def hyperbola_focal_length (a b : ℝ) (x y : ℝ) (e : ℝ) (ha : a > 0) (hb : b > 0) (hx : e = 2)
  (hxy : (x = 2) ∧ (y = 3)) (h_eq : (x^2 / a^2) - (y^2 / b^2) = 1 ): ℝ := 
  2 * 2 * a

theorem hyperbola_focal_length_is_4 : hyperbola_focal_length 
  a b 2 3 2 (by norm_num) (by norm_num) (by norm_num) 
  ((by norm_num : 4 / a^2 - 9 / b^2 = 1)) = 4 :=
sorry

end hyperbola_focal_length_is_4_l381_381828


namespace equal_segments_l381_381673

noncomputable def circles_tangent_line (A B C D E F : Point) (omega Omega : Circle) (ell : Line) : Prop :=
(B = line_intersection ell omega ∧ C = line_intersection ell omega ∧ D = line_intersection ell omega ∧ E = line_intersection ell Omega ∧ F = line_intersection ell Omega ∧ 
BC = DE)

theorem equal_segments (A B C D E F : Point) (omega Omega : Circle) (ell : Line) 
    (h_inscribed : circles_tangent_line A B C D E F omega Omega ell) 
    (h_eq : dist B C = dist D E) : dist A B = dist E F := 
sorry

end equal_segments_l381_381673


namespace sum_logarithms_l381_381235

theorem sum_logarithms :
  (∑ k in finset.Ico 3 101, real.logb 3 (1 + 1/(k:ℝ)) * real.logb k 3 * real.logb (k+1) 3) =
  1 - real.logb 2 3 / real.logb 2 101 :=
by
  sorry

end sum_logarithms_l381_381235


namespace robin_total_cost_l381_381475

def num_letters_in_name (name : String) : Nat := name.length

def calculate_total_cost (names : List String) (cost_per_bracelet : Nat) : Nat :=
  let total_bracelets := names.foldl (fun acc name => acc + num_letters_in_name name) 0
  total_bracelets * cost_per_bracelet

theorem robin_total_cost : 
  calculate_total_cost ["Jessica", "Tori", "Lily", "Patrice"] 2 = 44 :=
by
  sorry

end robin_total_cost_l381_381475


namespace complement_union_eq_l381_381964

-- Definitions / Conditions
def U : Set ℝ := Set.univ
def M : Set ℝ := { x | x < 1 }
def N : Set ℝ := { x | -1 < x ∧ x < 2 }

-- Statement of the theorem
theorem complement_union_eq {x : ℝ} :
  {x | x ≥ 2} = (U \ (M ∪ N)) := sorry

end complement_union_eq_l381_381964


namespace initial_oranges_l381_381094

theorem initial_oranges (O : ℕ) (h1 : (1 / 4 : ℚ) * (1 / 2 : ℚ) * O = 39) (h2 : (1 / 8 : ℚ) * (1 / 2 : ℚ) * O = 4 + 78 - (1 / 4 : ℚ) * (1 / 2 : ℚ) * O) :
  O = 96 :=
by
  sorry

end initial_oranges_l381_381094


namespace common_ratio_three_l381_381427

-- Definition of a geometric sequence sums
noncomputable def geometric_sum (a q : ℝ) (n : ℕ) : ℝ :=
  if q = 1 then n * a else a * (1 - q^n) / (1 - q)

-- Conditions for the problem
def S (a q : ℝ) (n : ℕ) : ℝ := geometric_sum a q n

-- Theorem statement for proving the common ratio q is 3 given S₄ = 10S₂
theorem common_ratio_three (a : ℝ) (q : ℝ) (h_pos: a > 0) (h_eq: S a q 4 = 10 * S a q 2) : q = 3 :=
by
  sorry

end common_ratio_three_l381_381427


namespace problem_statement_l381_381008

def f (x : ℝ) : ℝ := x^2 - 4*x + 4

theorem problem_statement : f (f (f (f (f (f 2))))) = 4 :=
by
  sorry

end problem_statement_l381_381008


namespace find_m_n_sq_l381_381786

theorem find_m_n_sq (m n : ℕ) (h1 : 2 ≤ m) (h2 : 2 ≤ n)
  (h3 : (∏ k in finset.range (n - 2 + 1).map (λ k, k + 2),
    (k^3 - 1) / (k^3 + 1)) = (m^3 - 1) / (m^3 + 2)) :
  m^2 + n^2 = 20 :=
sorry

end find_m_n_sq_l381_381786


namespace dress_assignment_l381_381760

theorem dress_assignment :
  ∃ (Katya Olya Liza Rita : string),
    (Katya ≠ "Pink" ∧ Katya ≠ "Blue") ∧
    (Rita ≠ "Green" ∧ Rita ≠ "Blue") ∧
    ∃ (girl_in_green girl_in_yellow : string),
      (girl_in_green = Katya ∧ girl_in_yellow = Rita ∧ 
       (Liza = "Pink" ∧ Olya = "Blue") ∧
       (Katya = "Green" ∧ Olya = "Blue" ∧ Liza = "Pink" ∧ Rita = "Yellow")) ∧
    ((girl_in_green stands between Liza and girl_in_yellow) ∧
     (Olya stands between Rita and Liza)) :=
by
  sorry

end dress_assignment_l381_381760


namespace problem_statement_l381_381190

variables {A B C D E F: Type} [EuclideanGeometry A B C D E F]

-- Points E and F are on the sides AB and BC of the cyclic quadrilateral ABCD, respectively.
def E_on_AB (A B E: Point) := lies_on E A B
def F_on_BC (B C F: Point) := lies_on F B C
def cyclic_quadrilateral (A B C D: Point) := cyclic A B C D

-- Given angles:
def angle_BFE (B F E: Point) := angle B F E
def angle_BDE (B D E: Point) := angle B D E
def angle_condition (B F E D: Point) := angle_BFE B F E = 2 * angle_BDE B D E

-- Prove the following relationship between the sides:
theorem problem_statement (A B C D E F: Point) (hE: E_on_AB A B E) (hF: F_on_BC B C F)
  (hcyclic: cyclic_quadrilateral A B C D) (hangle: angle_condition B F E D):
  (distance E F / distance A E) = (distance C F / distance A E) + (distance C D / distance A D) :=
sorry

end problem_statement_l381_381190


namespace cos_45_deg_l381_381669

theorem cos_45_deg : Real.cos (Real.pi / 4) = Real.sqrt 2 / 2 :=
by
  sorry

end cos_45_deg_l381_381669


namespace area_of_gray_region_l381_381913

theorem area_of_gray_region {r R : ℝ} (h1 : R = 3 * r) (h2 : R - r = 3) : (π * R^2 - π * r^2) = 18 * π :=
by
  have hr : r = 1.5, from sorry
  have hR : R = 3 * 1.5, from sorry
  have hR' : R = 4.5, from sorry
  have outer_area : π * R^2 = 20.25 * π, from sorry
  have inner_area : π * r^2 = 2.25 * π, from sorry
  show π * R^2 - π * r^2 = 18 * π, from sorry

end area_of_gray_region_l381_381913


namespace log_sum_eq_l381_381253

theorem log_sum_eq : ((∑ k in (Finset.range 98).map (λ n, n+3), (Real.log (1 + 1 / (k:ℝ)) / Real.log 3) * (Real.log 3 / Real.log k) * (Real.log 3 / Real.log (k+1)))) = 1 - (1 / Real.log 101 / Real.log 3) :=
by
  sorry

end log_sum_eq_l381_381253


namespace particle_speed_l381_381645

noncomputable def position (t : ℝ) : ℝ × ℝ := (3 * t + 8, 5 * t - 15)

theorem particle_speed :
  (∃ t : ℝ, P = position t) →
  (∃ Δx Δy : ℝ, Δx = 3 ∧ Δy = 5) →
  (sqrt (3^2 + 5^2) = sqrt 34) :=
by
  intros h1 h
  sorry

end particle_speed_l381_381645


namespace greatest_y_least_y_greatest_integer_y_l381_381585

theorem greatest_y (y : ℤ) (H : (8 : ℝ) / 11 > y / 17) : y ≤ 12 :=
sorry

theorem least_y (y : ℤ) (H : (8 : ℝ) / 11 > y / 17) : y ≥ 12 :=
sorry

theorem greatest_integer_y : ∀ (y : ℤ), ((8 : ℝ) / 11 > y / 17) → y = 12 :=
by
  intro y H
  apply le_antisymm
  apply greatest_y y H
  apply least_y y H

end greatest_y_least_y_greatest_integer_y_l381_381585


namespace gcd_lcm_of_multiple_l381_381872

-- Define GCD and LCM for two integers x and y
def gcd (a b : ℕ) : ℕ := Nat.gcd a b
def lcm (a b : ℕ) : ℕ := Nat.lcm a b

theorem gcd_lcm_of_multiple (x y z : ℤ) (h : x = z * y) (h_y_nonzero : y ≠ 0) :
  gcd (Int.natAbs x) (Int.natAbs y) = Int.natAbs y ∧ lcm (Int.natAbs x) (Int.natAbs y) = Int.natAbs x :=
  sorry

end gcd_lcm_of_multiple_l381_381872


namespace roots_of_quadratic_are_real_and_distinct_l381_381258

theorem roots_of_quadratic_are_real_and_distinct
  (a b : ℝ)
  (h : 4 * a * b ≠ 27) :
  ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (a * x1^2 - 3 * x1 * Real.sqrt(3) + b = 0) ∧ (a * x2^2 - 3 * x2 * Real.sqrt(3) + b = 0) :=
sorry

end roots_of_quadratic_are_real_and_distinct_l381_381258


namespace bookmarks_count_at_end_of_march_l381_381708

theorem bookmarks_count_at_end_of_march : 
  let daily_bookmarks := 30
  let current_bookmarks := 400
  let days_in_march := 31
  let total_bookmarks_in_march := daily_bookmarks * days_in_march
  let total_bookmarks_end_of_march := current_bookmarks + total_bookmarks_in_march
  in total_bookmarks_end_of_march = 1330 := by
  sorry

end bookmarks_count_at_end_of_march_l381_381708


namespace maximize_expression_l381_381870

theorem maximize_expression (a b : ℝ) (h : a^2 = 1 - 4 * b^2) :
  ∀ x y : ℝ, abs(x * y) ≤ (1 / 4) →
  ∀ z : ℝ, z = (2 * a * b) / (|a| + 2 * |b|) → 
  z ≤ (sqrt 2 / 4) := by
  sorry

end maximize_expression_l381_381870


namespace compute_logarithmic_sum_l381_381227

theorem compute_logarithmic_sum : 
  ∑ k in finset.Icc 3 100, (log 3 (1 + (1 : ℝ) / k) * log k 3 * log (k + 1) 3) = -1 := 
sorry

end compute_logarithmic_sum_l381_381227


namespace total_rent_pasture_l381_381117

variable (A_oxen : ℕ) (A_months : ℕ) (B_oxen : ℕ) (B_months : ℕ) (C_oxen : ℕ) (C_months : ℕ) (C_rent : ℕ)

def oxenMonths (oxen months : ℕ) : ℕ := oxen * months

theorem total_rent_pasture :
  A_oxen = 10 → A_months = 7 →
  B_oxen = 12 → B_months = 5 →
  C_oxen = 15 → C_months = 3 →
  C_rent = 45 →
  let total_oxen_months := oxenMonths A_oxen A_months + oxenMonths B_oxen B_months + oxenMonths C_oxen C_months
  in let total_rent := (total_oxen_months * C_rent) / oxenMonths C_oxen C_months
  in total_rent = 175 :=
by
  intros hAoxen hAmonths hBoxen hBmonths hCoxen hCmonths hCrent
  rw [hAoxen, hAmonths, hBoxen, hBmonths, hCoxen, hCmonths, hCrent]
  let total_oxen_months := 10 * 7 + 12 * 5 + 15 * 3
  have h1: total_oxen_months = 175 := by norm_num
  let total_rent := (175 * 45) / 45
  have h2: total_rent = 175 := by norm_num
  exact h2

end total_rent_pasture_l381_381117


namespace triangle_is_obtuse_l381_381187

theorem triangle_is_obtuse (T : Triangle) (exterior_angle_smaller : ∀ α β γ, (exterior_angle T α < adjacent_interior_angle β γ)): T.has_obtuse_angle :=
sorry

end triangle_is_obtuse_l381_381187


namespace sum_of_first_110_terms_l381_381074

theorem sum_of_first_110_terms
  (a d : ℝ)
  (h1 : (10 : ℝ) * (2 * a + (10 - 1) * d) / 2 = 100)
  (h2 : (100 : ℝ) * (2 * a + (100 - 1) * d) / 2 = 10) :
  (110 : ℝ) * (2 * a + (110 - 1) * d) / 2 = -110 :=
  sorry

end sum_of_first_110_terms_l381_381074


namespace problem_statement_l381_381974

open Set

variable (U : Set ℝ) (M N : Set ℝ)

theorem problem_statement (hU : U = univ) (hM : M = {x | x < 1}) (hN : N = {x | -1 < x ∧ x < 2}) :
  {x | 2 ≤ x} = compl (M ∪ N) :=
sorry

end problem_statement_l381_381974


namespace max_common_initial_segment_l381_381064

theorem max_common_initial_segment (m n : ℕ) (h_coprime : Nat.gcd m n = 1) : 
  ∃ L, L = m + n - 2 := 
sorry

end max_common_initial_segment_l381_381064


namespace max_magnitude_l381_381838

-- Define the vectors a and b
def a (θ : ℝ) : ℝ × ℝ := (Real.cos θ, Real.sin θ)
def b : ℝ × ℝ := (Real.sqrt 3, -1)

-- Function to calculate the magnitude of a vector
noncomputable def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1 ^ 2 + v.2 ^ 2)

-- Maximum value of |2*a - b|
theorem max_magnitude (θ : ℝ) : ∃ θ : ℝ, magnitude (2 * a θ - b) = 4 := by
  sorry

end max_magnitude_l381_381838


namespace upstream_distance_is_48_l381_381149

variables (distance_downstream time_downstream time_upstream speed_stream : ℝ)
variables (speed_boat distance_upstream : ℝ)

-- Given conditions
axiom h1 : distance_downstream = 84
axiom h2 : time_downstream = 2
axiom h3 : time_upstream = 2
axiom h4 : speed_stream = 9

-- Define the effective speeds
def speed_downstream (speed_boat speed_stream : ℝ) := speed_boat + speed_stream
def speed_upstream (speed_boat speed_stream : ℝ) := speed_boat - speed_stream

-- Equations based on travel times and distances
axiom eq1 : distance_downstream = (speed_downstream speed_boat speed_stream) * time_downstream
axiom eq2 : distance_upstream = (speed_upstream speed_boat speed_stream) * time_upstream

-- Theorem to prove the distance rowed upstream is 48 km
theorem upstream_distance_is_48 :
  distance_upstream = 48 :=
by
  sorry

end upstream_distance_is_48_l381_381149


namespace triangle_ac_equal_3_l381_381517

-- Definitions of the given conditions
def side1 : ℝ := Real.sqrt 10
def side2 : ℝ := Real.sqrt 13

-- The theorem to prove
theorem triangle_ac_equal_3 {AC : ℝ} (h1 : AC = AC) (h2 : AC = AC) :
  AC = 3 := by
  sorry

end triangle_ac_equal_3_l381_381517


namespace num_pos_three_digit_div_by_seven_l381_381861

theorem num_pos_three_digit_div_by_seven : 
  ∃ n : ℕ, (∀ k : ℕ, k < n → (∃ m : ℕ, 100 ≤ 7 * m ∧ 7 * m ≤ 999)) ∧ n = 128 :=
by
  sorry

end num_pos_three_digit_div_by_seven_l381_381861


namespace unique_parallel_line_exists_l381_381874

-- Define the types for Plane and Line
structure Plane :=
  (name : String)

structure Line :=
  (name : String)

-- Definitions for the conditions
def is_parallel_to_plane (l : Line) (α : Plane) : Prop := sorry
def is_in_plane (l : Line) (α : Plane) : Prop := sorry

-- Definitions for the existence and uniqueness of a parallel line
def unique_parallel (l : Line) (α : Plane) : Prop := sorry

-- The statement to be proven
theorem unique_parallel_line_exists (l : Line) (α : Plane) :
  ¬ is_parallel_to_plane(l, α) ∧ ¬ is_in_plane(l, α) → unique_parallel(l, α) :=
sorry

end unique_parallel_line_exists_l381_381874


namespace area_of_rectangular_plot_l381_381516

theorem area_of_rectangular_plot (breadth : ℝ) (length : ℝ) 
    (h1 : breadth = 17) 
    (h2 : length = 3 * breadth) : 
    length * breadth = 867 := 
by
  sorry

end area_of_rectangular_plot_l381_381516


namespace sum_log_identity_l381_381249

noncomputable def log_base (b x : ℝ) : ℝ := Real.log x / Real.log b

theorem sum_log_identity :
  ∑ k in Finset.range (98) + 3, log_base 3 (1 + 1 / k) * log_base k 3 * log_base (k + 1) 3 =
  1 - 1 / log_base 3 101 :=
by
  sorry

end sum_log_identity_l381_381249


namespace distance_between_vertices_l381_381512

theorem distance_between_vertices :
  let equation : ℝ → ℝ → Prop := λ x y, √(x^2 + y^2) + |y - 2| = 4
  let vertex1 := (0, 3)
  let vertex2 := (0, -1)
  (abs ((vertex1.2) - (vertex2.2)) = 4) :=
by
  sorry

end distance_between_vertices_l381_381512


namespace lamp_switching_problem_l381_381929

theorem lamp_switching_problem (n : ℕ) (h_pos : n > 0) : (∃ f : ℕ → Fin n → Prop,
  (∀ i : Fin n, ¬ f 0 i) ∧
  (∀ k : ℕ, k > 0 → k ≤ n → ∃ s : Finset (Fin n), s.card = k ∧ ∀ i : Fin n, f (k - 1) i ↔ (i ∈ s) ∧ ¬ f k i) ∧
  (∀ i : Fin n, f n i) ↔ n ≠ 2) :=
begin
  sorry
end

end lamp_switching_problem_l381_381929


namespace perfect_number_representation_l381_381647

theorem perfect_number_representation : 
  8128 = 2^6 + 2^7 + 2^8 + 2^9 + 2^{10} + 2^{11} + 2^{12} :=
by
  sorry

end perfect_number_representation_l381_381647


namespace xiaoming_steps_l381_381547

-- Define what it means to count the steps Xiao Ming can take
def countSteps (n : ℕ) : ℕ :=
  (n / 2).nat_succ |> (λ maxTwoStepJumps, 
    Finset.range (nat_succ maxTwoStepJumps) 
      .sum (λ k, nat.choose (n - k) k))
      
-- Theorem statement
theorem xiaoming_steps : countSteps 10 = 89 :=
  sorry

end xiaoming_steps_l381_381547


namespace first_player_win_count_l381_381101

def winning_positions (n : ℕ) : ℕ :=
  let l : ℕ → Bool
    | 1 => true
    | 2 => false
    | 3 => true
    | 4 => true
    | k + 5 => !(l (k + 4) ∧ l (k + 2) ∧ l k)
  in (List.range n).countp (λ k => l (k + 1))

theorem first_player_win_count : winning_positions 100 = 71 := 
  sorry

end first_player_win_count_l381_381101


namespace m_in_A_l381_381019

variable (x : ℝ)
variable (A : Set ℝ := {x | x ≤ 2})
noncomputable def m : ℝ := Real.sqrt 2

theorem m_in_A : m ∈ A :=
sorry

end m_in_A_l381_381019


namespace question_proof_l381_381987

open Set

variable (U : Set ℝ := univ)
variable (M : Set ℝ := {x | x < 1})
variable (N : Set ℝ := {x | -1 < x ∧ x < 2})

theorem question_proof : {x | x ≥ 2} = compl (M ∪ N) :=
by
  sorry

end question_proof_l381_381987


namespace cookout_2006_kids_l381_381385

def kids_2004 : ℕ := 60
def kids_2005 : ℕ := kids_2004 / 2
def kids_2006 : ℕ := (2 * kids_2005) / 3

theorem cookout_2006_kids : kids_2006 = 20 := by
  sorry

end cookout_2006_kids_l381_381385


namespace cheryl_tournament_cost_is_1440_l381_381216

noncomputable def cheryl_electricity_bill : ℝ := 800
noncomputable def additional_for_cell_phone : ℝ := 400
noncomputable def cheryl_cell_phone_expenses : ℝ := cheryl_electricity_bill + additional_for_cell_phone
noncomputable def tournament_cost_percentage : ℝ := 0.2
noncomputable def additional_tournament_cost : ℝ := tournament_cost_percentage * cheryl_cell_phone_expenses
noncomputable def total_tournament_cost : ℝ := cheryl_cell_phone_expenses + additional_tournament_cost

theorem cheryl_tournament_cost_is_1440 : total_tournament_cost = 1440 := by
  sorry

end cheryl_tournament_cost_is_1440_l381_381216


namespace correct_assignment_l381_381771

structure GirlDressAssignment :=
  (Katya : String)
  (Olya : String)
  (Liza : String)
  (Rita : String)

def solution : GirlDressAssignment :=
  ⟨"Green", "Blue", "Pink", "Yellow"⟩

theorem correct_assignment
  (Katya_not_pink_or_blue : solution.Katya ≠ "Pink" ∧ solution.Katya ≠ "Blue")
  (Green_between_Liza_and_Yellow : 
    (solution.Katya = "Green" ∧ solution.Liza = "Pink" ∧ solution.Rita = "Yellow") ∧
    (solution.Katya = "Green" ∧ solution.Rita = "Yellow" ∧ solution.Liza = "Pink"))
  (Rita_not_green_or_blue : solution.Rita ≠ "Green" ∧ solution.Rita ≠ "Blue")
  (Olya_between_Rita_and_Pink : 
    (solution.Olya = "Blue" ∧ solution.Rita = "Yellow" ∧ solution.Liza = "Pink") ∧
    (solution.Olya = "Blue" ∧ solution.Liza = "Pink" ∧ solution.Rita = "Yellow"))
  : solution = ⟨"Green", "Blue", "Pink", "Yellow"⟩ := by
  sorry

end correct_assignment_l381_381771


namespace who_wears_which_dress_l381_381731

-- Define the possible girls
inductive Girl
| Katya | Olya | Liza | Rita
deriving DecidableEq

-- Define the possible dresses
inductive Dress
| Pink | Green | Yellow | Blue
deriving DecidableEq

-- Define the fact that each girl is wearing a dress
structure Wearing (girl : Girl) (dress : Dress) : Prop

-- Define the conditions
theorem who_wears_which_dress :
  (¬ Wearing Girl.Katya Dress.Pink ∧ ¬ Wearing Girl.Katya Dress.Blue) ∧
  (∀ g1 g2 g3, Wearing g1 Dress.Green → (Wearing g2 Dress.Pink ∧ Wearing g3 Dress.Yellow → (g2 = Girl.Liza ∧ (g3 = Girl.Rita)) ∨ (g3 = Girl.Liza ∧ g2 = Girl.Rita))) ∧
  (¬ Wearing Girl.Rita Dress.Green ∧ ¬ Wearing Girl.Rita Dress.Blue) ∧
  (∀ g1 g2, (Wearing g1 Dress.Pink ∧ Wearing g2 Dress.Yellow) → Girl.Olya = g2 ∧ Girl.Rita = g1) →
  (Wearing Girl.Katya Dress.Green ∧ Wearing Girl.Olya Dress.Blue ∧ Wearing Girl.Liza Dress.Pink ∧ Wearing Girl.Rita Dress.Yellow) :=
by
  sorry

end who_wears_which_dress_l381_381731


namespace ratio_value_l381_381865

theorem ratio_value (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h_diff : x ≠ y ∧ y ≠ z ∧ x ≠ z) 
(h1 : (y + 1) / (x - z + 1) = (x + y + 2) / (z + 2)) 
(h2 : (x + y + 2) / (z + 2) = (x + 1) / (y + 1)) :
  (x + 1) / (y + 1) = 2 :=
by
  sorry

end ratio_value_l381_381865


namespace g_pi_over_3_eq_1_l381_381638

noncomputable def f (ω ϕ : ℝ) : ℝ → ℝ :=
  λ x, 3 * Real.sin (ω * x + ϕ)

noncomputable def g (ω ϕ : ℝ) : ℝ → ℝ :=
  λ x, 3 * Real.cos (ω * x + ϕ) + 1

theorem g_pi_over_3_eq_1
  (ω ϕ : ℝ)
  (h : ∀ x : ℝ, f ω ϕ (π/3 + x) = f ω ϕ (π/3 - x)) :
  g ω ϕ (π/3) = 1 :=
sorry

end g_pi_over_3_eq_1_l381_381638


namespace positive_solution_eq_l381_381295

theorem positive_solution_eq (x : ℝ) (h₁ : x = 1) :
  sqrt (x + sqrt (x + sqrt (x + …))) = sqrt (x * sqrt (x * sqrt (x * …))) :=
by
  sorry

end positive_solution_eq_l381_381295


namespace greatest_integer_l381_381577

theorem greatest_integer (y : ℤ) : (8 / 11 : ℝ) > (y / 17 : ℝ) → y ≤ 12 :=
by sorry

end greatest_integer_l381_381577


namespace dress_assignment_l381_381755

variables {Girl : Type} [Finite Girl]
variables (Katya Olya Liza Rita Pink Green Yellow Blue : Girl)
variables (standing_between : Girl → Girl → Girl → Prop)

-- Conditions
variable (cond1 : Katya ≠ Pink ∧ Katya ≠ Blue)
variable (cond2 : standing_between Green Liza Yellow)
variable (cond3 : Rita ≠ Green ∧ Rita ≠ Blue)
variable (cond4 : standing_between Olya Rita Pink)

-- Theorem statement
theorem dress_assignment :
  Katya = Green ∧ Olya = Blue ∧ Liza = Pink ∧ Rita = Yellow := 
sorry

end dress_assignment_l381_381755


namespace sofia_running_time_l381_381481

theorem sofia_running_time :
  let distance_first_section := 100 -- meters
  let speed_first_section := 5 -- meters per second
  let distance_second_section := 300 -- meters
  let speed_second_section := 4 -- meters per second
  let num_laps := 6
  let time_first_section := distance_first_section / speed_first_section -- in seconds
  let time_second_section := distance_second_section / speed_second_section -- in seconds
  let time_per_lap := time_first_section + time_second_section -- in seconds
  let total_time_seconds := num_laps * time_per_lap -- in seconds
  let total_time_minutes := total_time_seconds / 60 -- integer division for minutes
  let remaining_seconds := total_time_seconds % 60 -- modulo for remaining seconds
  total_time_minutes = 9 ∧ remaining_seconds = 30 := 
  by
  sorry

end sofia_running_time_l381_381481


namespace fraction_of_area_is_correct_l381_381166
  
  -- Definitions based on problem conditions
  def large_square_area (a b : ℝ) : ℝ := (a + b) ^ 2

  def inscribed_square_area (b : ℝ) : ℝ := (2 * b) ^ 2

  def fraction_area (a b : ℝ) : ℝ :=
    inscribed_square_area b / large_square_area a b
  
  -- The theorem we need to prove
  theorem fraction_of_area_is_correct (a b : ℝ) (h1 : a = b * real.sqrt 3) :
    fraction_area a b = 4 - 2 * real.sqrt 3 := by
  sorry
  
end fraction_of_area_is_correct_l381_381166


namespace sin_alpha_through_point_l381_381875

variables {α : Type*} [LinearOrderedField α] [Real.Angle α]

noncomputable def sin_of_point (x y : α) (h : x^2 + y^2 ≠ 0) : α :=
  y / sqrt (x^2 + y^2)

theorem sin_alpha_through_point :
  sin_of_point (-1 : ℝ) 2 (by norm_num [pow_two]; linarith) = 2 * sqrt 5 / 5 :=
by sorry

end sin_alpha_through_point_l381_381875


namespace sum_floor_log3_l381_381275

theorem sum_floor_log3 (N : ℕ) (h : N = 512) : 
  (∑ n in Finset.range (N + 1), Int.floor (Real.log n / Real.log 3)) = 6015 :=
by
  sorry

end sum_floor_log3_l381_381275


namespace area_of_triangle_BOC_l381_381415

theorem area_of_triangle_BOC 
    (A B C O K : Type) 
    (h : Triangle A B C) 
    (hAC : dist A C = 14) 
    (hAB : dist A B = 6) 
    (circ : Circle A C) 
    (hO : midpoint O A C) 
    (hK : intersects circ B C K) 
    (h_angle : ∠ B A K = ∠ A C B) : 
    area (triangle B O C) = 21 :=
sorry

end area_of_triangle_BOC_l381_381415


namespace sum_first_110_terms_l381_381078

variable (a d : ℕ → ℤ) [is_arithmetic_sequence: ∀ n, a (n + 1) = a n + d n]

-- Given that the sum of the first 10 terms is 100
def sum_first_10_terms := (∑ i in Finset.range 10, a i) = 100

-- Given that the sum of the first 100 terms is 10
def sum_first_100_terms := (∑ i in Finset.range 100, a i) = 10

-- Prove that the sum of the first 110 terms is -110
theorem sum_first_110_terms (h1 : sum_first_10_terms a) (h2 : sum_first_100_terms a) : 
  (∑ i in Finset.range 110, a i) = -110 :=
sorry

end sum_first_110_terms_l381_381078


namespace factorial_division_l381_381680

-- Define the factorial function explicitly to ensure broad compatibility
def fact : ℕ → ℕ
| 0       := 1
| (n + 1) := (n + 1) * fact n

-- The main statement to prove
theorem factorial_division : (fact 50) / (fact 47) = 117600 :=
by
  -- The proof is skipped with 'sorry'
  sorry

end factorial_division_l381_381680


namespace midpoint_parallel_to_hypotenuse_l381_381133

variables {A B C D E H Q P M N : Type} [EuclideanSpace ℝ (Fin 3)]
variables {B H : point}
variables (lineAC : line ℝ) [hpH : is_perpendicular (lineBH) (lineAC)]
variables {Q} [hq : point_of_bisector_intersection B H AD (point_of_intersection)] 
variables {P} [hp : point_of_bisector_intersection B H CE (point_of_intersection)] 
variables {M N} [hm : is_midpoint P E M] [hn : is_midpoint Q D N]

theorem midpoint_parallel_to_hypotenuse (lineAC : line ℝ)
  (h_perp : is_perpendicular (lineBH) (lineAC))
  (h_bisectAD : bisects Q AD)
  (h_bisectCE : bisects P CE)
  (hm : is_midpoint P E M)
  (hn : is_midpoint Q D N) :
  is_parallel (line_through M N) (lineAC) := sorry

end midpoint_parallel_to_hypotenuse_l381_381133


namespace area_triangle_AEB_l381_381400

variable (A B C D F G E : Type)
variable (AB BC DF GC CD FG AE BE : ℝ)
variable [Field ℝ]

-- Conditions
variable (rectangle_ABCD : AB * CD = BC * AD)
variable (ab_length : AB = 10)
variable (bc_length : BC = 4)
variable (df_length : DF = 2)
variable (gc_length : GC = 3)
variable (af_intersect_bg_at_e : AE * BE = FE * GE)
variable (fg_length : FG = CD - DF - GC)

-- Goal
theorem area_triangle_AEB : Rectange ABCD → AB = 10 → BC = 4 → DF = 2 → GC = 3 
  → FG = CD - DF - GC → AF intersect BG = E 
  → area_of_triangle A B E = 40 := 
begin 
  sorry 
end

end area_triangle_AEB_l381_381400


namespace max_crate_weight_l381_381603

theorem max_crate_weight (carries_3_4_5 : ℕ → Prop)
  (crates : ℕ)
  (weights : ℕ → ℕ)
  (h_carries : (carries_3_4_5 3 ∨ carries_3_4_5 4 ∨ carries_3_4_5 5))
  (h_weight_min : ∀ n, n ≥ 1 → weights n ≥ 150) :
  (crates = 5 ∧ weights crates = 150) → crates * weights crates = 750 :=
by {
  intro h,
  cases h,
  rw h_left,
  rw h_right,
  exact rfl,
}

end max_crate_weight_l381_381603


namespace dress_assignments_l381_381748

structure GirlDress : Type :=
  (Katya Olya Liza Rita : String)

def dresses := ["Pink", "Green", "Yellow", "Blue"]

axiom not_pink_or_blue : GirlDress.Katya ≠ "Pink" ∧ GirlDress.Katya ≠ "Blue"
axiom green_between_liza_yellow : (GirlDress.Liza = "Pink" ∨ GirlDress.Rita = "Yellow") ∧
                                  GirlDress.Katya = "Green" ∧
                                  GirlDress.Rita = "Yellow" ∧ GirlDress.Liza = "Pink"
axiom not_green_or_blue : GirlDress.Rita ≠ "Green" ∧ GirlDress.Rita ≠ "Blue"
axiom olya_between_rita_pink : GirlDress.Olya ≠ "Pink" → GirlDress.Rita ≠ "Pink" → GirlDress.Liza = "Pink"

theorem dress_assignments (gd : GirlDress) :
  gd.Katya = "Green" ∧ gd.Olya = "Blue" ∧ gd.Liza = "Pink" ∧ gd.Rita = "Yellow" :=
by
  sorry

end dress_assignments_l381_381748


namespace triangle_right_angled_or_isosceles_l381_381016

theorem triangle_right_angled_or_isosceles 
  (α β γ : ℝ) 
  (h_angles : α + β + γ = Math.pi) 
  (h_eq_ratio : sin α / sin β = cos β / cos α) :
  (α = β) ∨ (α + β = Math.pi / 2) :=
sorry

end triangle_right_angled_or_isosceles_l381_381016


namespace today_is_wednesday_l381_381105

-- Definitions for days of the week for simplicity
inductive Day : Type
| Sunday | Monday | Tuesday | Wednesday | Thursday | Friday | Saturday
  deriving DecidableEq

-- Function to get the next day
def next_day (d : Day) : Day :=
  match d with
  | Day.Sunday    => Day.Monday
  | Day.Monday    => Day.Tuesday
  | Day.Tuesday   => Day.Wednesday
  | Day.Wednesday => Day.Thursday
  | Day.Thursday  => Day.Friday
  | Day.Friday    => Day.Saturday
  | Day.Saturday  => Day.Sunday

-- Function to get the day after tomorrow
def day_after_tomorrow (d : Day) : Day :=
  next_day (next_day d)

-- Function to get the day that is today when yesterday was tomorrow
def day_when_yesterday_was_tomorrow (d : Day) : Day :=
  match d with
  | Day.Sunday    => Day.Friday   -- 2 days back
  | Day.Monday    => Day.Saturday
  | Day.Tuesday   => Day.Sunday
  | Day.Wednesday => Day.Monday
  | Day.Thursday  => Day.Tuesday
  | Day.Friday    => Day.Wednesday
  | Day.Saturday  => Day.Thursday

-- Prove that today is Wednesday given the stated conditions
theorem today_is_wednesday (today : Day)
  (h1 : day_after_tomorrow day_when_yesterday_was_tomorrow (today) = Day.Sunday)
  (h2 : ∀ d, day_when_yesterday_was_tomorrow (d) = d → d = today):
  today = Day.Wednesday :=
by
  sorry

end today_is_wednesday_l381_381105


namespace sum_interior_angles_convex_polygon_number_of_triangles_convex_polygon_l381_381116

-- Define a convex n-gon and prove that the sum of its interior angles is (n-2) * 180 degrees
theorem sum_interior_angles_convex_polygon (n : ℕ) (h : 3 ≤ n) :
  ∃ (sum_of_angles : ℝ), sum_of_angles = (n-2) * 180 :=
sorry

-- Define a convex n-gon and prove that the number of triangles formed by dividing with non-intersecting diagonals is n-2
theorem number_of_triangles_convex_polygon (n : ℕ) (h : 3 ≤ n) :
  ∃ (num_of_triangles : ℕ), num_of_triangles = n-2 :=
sorry

end sum_interior_angles_convex_polygon_number_of_triangles_convex_polygon_l381_381116


namespace find_x_plus_y_l381_381331

theorem find_x_plus_y (x y : ℝ) (h1 : x + Real.cos y = 3005) (h2 : x + 3005 * Real.sin y = 3004) (h3 : 0 ≤ y ∧ y ≤ Real.pi / 2) : x + y = 3004 :=
by 
  sorry

end find_x_plus_y_l381_381331


namespace perpendicular_lines_a_between_values_l381_381518

theorem perpendicular_lines_a_between_values :
  ∃ a : ℝ, 
    (∃ l1 l2 : ℝ → ℝ → ℝ, (λ x y, a * x + 3 * y + 1 = 0) = l1 ∧ (λ x y, 2 * x + (a + 1) * y + 1 = 0) = l2) ∧ 
    (a * 2 + 3 * (a + 1) = 0) :=
sorry

end perpendicular_lines_a_between_values_l381_381518


namespace limit_calculation_l381_381123

noncomputable def limit_problem := 
  lim (λ x, (e^(Real.sin (Real.pi * x)) - 1) / (x - 1)) 1

theorem limit_calculation :
  (limit_problem) ^ (1^2 + 1) = Real.pi ^ 2 := 
sorry

end limit_calculation_l381_381123


namespace ratio_doctors_to_lawyers_l381_381496

-- Definitions based on conditions
def average_age_doctors := 35
def average_age_lawyers := 50
def combined_average_age := 40

-- Define variables
variables (d l : ℕ) -- d is number of doctors, l is number of lawyers

-- Hypothesis based on the problem statement
axiom h : (average_age_doctors * d + average_age_lawyers * l) = combined_average_age * (d + l)

-- The theorem we need to prove is the ratio of doctors to lawyers is 2:1
theorem ratio_doctors_to_lawyers : d = 2 * l :=
by sorry

end ratio_doctors_to_lawyers_l381_381496


namespace Hillary_activities_LCM_l381_381196

theorem Hillary_activities_LCM :
  let swim := 6
  let run := 4
  let cycle := 16
  Nat.lcm (Nat.lcm swim run) cycle = 48 :=
by
  sorry

end Hillary_activities_LCM_l381_381196


namespace therapy_charge_l381_381170

-- Define the charges
def first_hour_charge (S : ℝ) : ℝ := S + 50
def subsequent_hour_charge (S : ℝ) : ℝ := S

-- Define the total charge before service fee for 8 hours
def total_charge_8_hours_before_fee (F S : ℝ) : ℝ := F + 7 * S

-- Define the total charge including the service fee for 8 hours
def total_charge_8_hours (F S : ℝ) : ℝ := 1.10 * (F + 7 * S)

-- Define the total charge before service fee for 3 hours
def total_charge_3_hours_before_fee (F S : ℝ) : ℝ := F + 2 * S

-- Define the total charge including the service fee for 3 hours
def total_charge_3_hours (F S : ℝ) : ℝ := 1.10 * (F + 2 * S)

theorem therapy_charge (S F : ℝ) :
  (F = S + 50) → (1.10 * (F + 7 * S) = 900) → (1.10 * (F + 2 * S) = 371.87) :=
by {
  sorry
}

end therapy_charge_l381_381170


namespace sum_log_identity_l381_381246

noncomputable def log_base (b x : ℝ) : ℝ := Real.log x / Real.log b

theorem sum_log_identity :
  ∑ k in Finset.range (98) + 3, log_base 3 (1 + 1 / k) * log_base k 3 * log_base (k + 1) 3 =
  1 - 1 / log_base 3 101 :=
by
  sorry

end sum_log_identity_l381_381246


namespace find_angle_ACB_l381_381407

theorem find_angle_ACB 
  (D C A B E F : Type)
  (DC_parallel_AB : parallel DC AB)
  (EF_parallel_DC : parallel EF DC)
  (EF_between_DC_AB : between EF DC AB)
  (angle_DCA : angle DCA = 50)
  (angle_EFB : angle EFB = 80) :
  angle ACB = 50 :=
sorry

end find_angle_ACB_l381_381407


namespace solution_l381_381782

def f (x a b : ℝ) : ℝ := x^5 + a * x^3 + b * x - 8

theorem solution (a b : ℝ) (h : f (-2) a b = 10) : f 2 a b = -26 :=
by 
  -- Here we will skip the actual proof by using sorry
  sorry

end solution_l381_381782


namespace greatest_integer_l381_381582

theorem greatest_integer (y : ℤ) (h : (8 : ℚ) / 11 > y / 17) : y ≤ 12 :=
by
  have h₁ : (8 : ℚ) / 11 * 17 > y := by exact (div_mul_cancel _ (by norm_num : 17 ≠ 0))
  have h₂ : 136 / 11 > y := by rwa mul_comm _ 17 at h₁
  exact_mod_cast le_of_lt h₂

end greatest_integer_l381_381582


namespace sum_first_110_terms_l381_381077

variable (a d : ℕ → ℤ) [is_arithmetic_sequence: ∀ n, a (n + 1) = a n + d n]

-- Given that the sum of the first 10 terms is 100
def sum_first_10_terms := (∑ i in Finset.range 10, a i) = 100

-- Given that the sum of the first 100 terms is 10
def sum_first_100_terms := (∑ i in Finset.range 100, a i) = 10

-- Prove that the sum of the first 110 terms is -110
theorem sum_first_110_terms (h1 : sum_first_10_terms a) (h2 : sum_first_100_terms a) : 
  (∑ i in Finset.range 110, a i) = -110 :=
sorry

end sum_first_110_terms_l381_381077


namespace egg_cartons_l381_381200

theorem egg_cartons (chickens eggs_per_chicken eggs_per_carton : ℕ) (h_chickens : chickens = 20) (h_eggs_per_chicken : eggs_per_chicken = 6) (h_eggs_per_carton : eggs_per_carton = 12) : 
  (chickens * eggs_per_chicken) / eggs_per_carton = 10 :=
by
  rw [h_chickens, h_eggs_per_chicken, h_eggs_per_carton] -- Replace the variables with the given values
  -- Calculate the number of eggs
  have h_eggs := 20 * 6
  -- Apply the number of eggs to find the number of cartons
  rw [show 20 * 6 = 120, from rfl, show 120 / 12 = 10, from rfl]
  sorry -- Placeholder for the detailed proof

end egg_cartons_l381_381200


namespace team_A_played_18_games_l381_381045

def games_played_by_team_A
  (wA : ℚ) (wB : ℚ) (diff_wins : ℚ) (diff_losses : ℚ) : ℕ :=
  let a := (wB * (diff_wins + diff_losses) - diff_wins) / (wA * (wB - 1) + (1 - wB)) in 
  a.nat_abs

theorem team_A_played_18_games
  (wA : ℚ) (wB : ℚ) (diff_wins : ℚ) (diff_losses : ℚ)
  (h_wA : wA = 2 / 3)
  (h_wB : wB = 3 / 5)
  (h_diff_wins : diff_wins = 6)
  (h_diff_losses : diff_losses = 6) :
  games_played_by_team_A wA wB diff_wins diff_losses = 18 :=
by {
  rw [h_wA, h_wB, h_diff_wins, h_diff_losses],
  show (18 : ℚ).nat_abs = 18,
  exact nat.abs_of_nat 18,
  sorry
}

end team_A_played_18_games_l381_381045


namespace num_pos_three_digit_div_by_seven_l381_381863

theorem num_pos_three_digit_div_by_seven : 
  ∃ n : ℕ, (∀ k : ℕ, k < n → (∃ m : ℕ, 100 ≤ 7 * m ∧ 7 * m ≤ 999)) ∧ n = 128 :=
by
  sorry

end num_pos_three_digit_div_by_seven_l381_381863


namespace problem_l381_381535

noncomputable def f (x : ℝ) : ℝ := sin x
noncomputable def g (x : ℝ) : ℝ := 3 * sin (3 * x - π / 4)

theorem problem (x : ℝ) :
  (g x = 3 * sin (3 * x - π / 4)) ∧
  (∀ x, g (π / 4 - x) = g (π / 4 + x)) ∧
  (g (5 * π / 12) = 0) ∧
  (∀ x ∈ Icc (0 : ℝ) (π / 4), ∀ y ∈ Icc (0 : ℝ) (π / 4), x < y → g x < g y) :=
by sorry

end problem_l381_381535


namespace problem_solution_l381_381390

noncomputable def smallest_number_of_students := ∀ (n : ℕ), 
  (∀ (scores : List ℕ), 
    scores.length = n ∧ 
    (scores.count 95) = 7 ∧ 
    (∀ score ∈ scores, score ≥ 65) ∧ 
    (scores.sum / n.to_nat) = 80) 
  → (n ≥ 14)

theorem problem_solution : smallest_number_of_students :=
by 
  intros n scores h_scores,
  have h_len := h_scores.1,
  have h_95 := h_scores.2.1,
  have h_min := h_scores.2.2.1,
  have h_mean := h_scores.2.2.2,
  sorry -- proof

end problem_solution_l381_381390


namespace total_black_dots_l381_381085

theorem total_black_dots (num_butterflies : ℕ) (black_dots_per_butterfly : ℕ) (h1 : num_butterflies = 397) (h2 : black_dots_per_butterfly = 12) : num_butterflies * black_dots_per_butterfly = 4764 :=
by {
  rw [h1, h2],
  -- We need to prove 397 * 12 = 4764, which we assume as given here.
  sorry
}

end total_black_dots_l381_381085


namespace solve_inequality_l381_381485

theorem solve_inequality (a x : ℝ) :
  (a > 0 → (a - 1) / a < x ∧ x < 1) ∧ 
  (a = 0 → x < 1) ∧ 
  (a < 0 → x > (a - 1) / a ∨ x < 1) ↔ 
  (ax / (x - 1) < (a - 1) / (x - 1)) :=
sorry

end solve_inequality_l381_381485


namespace third_smallest_is_five_l381_381702

noncomputable def probability_third_smallest_is_five : ℚ :=
  let total_ways := (Nat.choose 15 8) in
  let favorable_ways := (Nat.choose 4 2) * (Nat.choose 10 5) in
  favorable_ways / total_ways

theorem third_smallest_is_five :
  probability_third_smallest_is_five = 4 / 17 := sorry

end third_smallest_is_five_l381_381702


namespace committeeFormation_l381_381637

-- Establish the given problem conditions in Lean

open Classical

-- Noncomputable because we are working with combinations and products
noncomputable def numberOfWaysToFormCommittee (numSchools : ℕ) (membersPerSchool : ℕ) (hostSchools : ℕ) (hostReps : ℕ) (nonHostReps : ℕ) : ℕ :=
  let totalSchools := numSchools
  let chooseHostSchools := Nat.choose totalSchools hostSchools
  let chooseHostRepsPerSchool := Nat.choose membersPerSchool hostReps
  let allHostRepsChosen := chooseHostRepsPerSchool ^ hostSchools
  let chooseNonHostRepsPerSchool := Nat.choose membersPerSchool nonHostReps
  let allNonHostRepsChosen := chooseNonHostRepsPerSchool ^ (totalSchools - hostSchools)
  chooseHostSchools * allHostRepsChosen * allNonHostRepsChosen

-- We now state our theorem
theorem committeeFormation : numberOfWaysToFormCommittee 4 6 2 3 1 = 86400 :=
by
  -- This is the lemma we need to prove
  sorry

end committeeFormation_l381_381637


namespace number_of_solution_pairs_l381_381293

theorem number_of_solution_pairs : 
  (∃ (n : ℕ), n = 21) ↔
  ∃ (f : ℕ × ℕ → Prop), 
    (∀ x y, f (x, y) ↔ (4 * x + 7 * y = 600)) ∧ 
    ((∃ k, f (k, 0)) ≠ true ∧ (∃ k, f (0, k)) ≠ true) :=
by {
  let f := λ (p : ℕ × ℕ), 4 * p.1 + 7 * p.2 = 600,
  have key : ∀ x y, f (x, y) ↔ 4 * x + 7 * y = 600 :=
    by { intro x, intro y, refl },
  exact ⟨λ ⟨n, hn⟩, ⟨f, ⟨key, ⟨⟨λ k, ⟨k, (lt_irrefl _)⟩, ⟨0, (lt_irrefl _).not_le⟩⟩⟩⟩⟩,
        λ ⟨f, ⟨key, ⟨h1, h2⟩⟩⟩, ⟨21, rfl⟩⟩,
  sorry
}

end number_of_solution_pairs_l381_381293


namespace find_a_find_P_l381_381619

-- Problem 1
theorem find_a (a : ℝ) (d : ℝ) (hA : (a, 6) = A) (hDist : d = 4) (hLine : 3 * a - 4 * 6 = 2) :
  a = 2 ∨ a = 46 / 3 :=
by
  sorry

-- Problem 2
theorem find_P (P : ℝ × ℝ) (hLine1 : x + 3 * y = 0) (hDistEq : distance P (0, 0) = distance P (x + 3 * y - 2 = 0)) :
  P = (3/5, -1/5) ∨ P = (-3/5, 1/5) :=
by
  sorry

end find_a_find_P_l381_381619


namespace question_proof_l381_381985

open Set

variable (U : Set ℝ := univ)
variable (M : Set ℝ := {x | x < 1})
variable (N : Set ℝ := {x | -1 < x ∧ x < 2})

theorem question_proof : {x | x ≥ 2} = compl (M ∪ N) :=
by
  sorry

end question_proof_l381_381985


namespace wff_formulas_l381_381684

inductive wff : Type
| prop : string -> wff
| neg : wff -> wff
| and : wff -> wff -> wff
| or : wff -> wff -> wff
| implies : wff -> wff -> wff
| iff : wff -> wff -> wff

def formula_1 := wff.implies (wff.implies (wff.neg (wff.prop "P")) (wff.prop "Q")) (wff.implies (wff.prop "Q") (wff.prop "P"))
def formula_4 := wff.iff (wff.prop "P") (wff.implies (wff.prop "R") (wff.prop "S"))
def formula_5 := wff.implies (wff.implies (wff.prop "P") (wff.implies (wff.prop "Q") (wff.prop "R"))) (wff.implies (wff.implies (wff.prop "P") (wff.prop "Q")) (wff.implies (wff.prop "P") (wff.prop "R")))

theorem wff_formulas : 
  (is_wff formula_1) ∧ 
  (is_wff formula_4) ∧ 
  (is_wff formula_5)
:= sorry

-- Auxiliary definitions to check if a formula is well-formed.
def is_wff : wff -> Prop
| (wff.prop _) := true
| (wff.neg A) := is_wff A
| (wff.and A B) := is_wff A ∧ is_wff B
| (wff.or A B) := is_wff A ∧ is_wff B
| (wff.implies A B) := is_wff A ∧ is_wff B
| (wff.iff A B) := is_wff A ∧ is_wff B

end wff_formulas_l381_381684


namespace complement_union_eq_ge_two_l381_381992

def U : Set ℝ := Set.univ
def M : Set ℝ := { x : ℝ | x < 1 }
def N : Set ℝ := { x : ℝ | -1 < x ∧ x < 2 }

theorem complement_union_eq_ge_two : { x : ℝ | x ≥ 2 } = U \ (M ∪ N) :=
by
  sorry

end complement_union_eq_ge_two_l381_381992


namespace divisor_sum_eq_floor_sum_l381_381001

def d (n : ℕ) : ℕ := 
  finset.card (finset.filter (λ m, n % m = 0) (finset.range (n + 1)))

def e (n : ℕ) : ℕ := 
  int.toNat (Int.floor (2000 / n))

theorem divisor_sum_eq_floor_sum : 
  (∑ n in finset.range 2000, d (n + 1)) = (∑ n in finset.range 2000, e (n + 1)) := 
sorry

end divisor_sum_eq_floor_sum_l381_381001


namespace avery_egg_cartons_filled_l381_381197

-- Definitions (conditions identified in step a)
def total_chickens : ℕ := 20
def eggs_per_chicken : ℕ := 6
def eggs_per_carton : ℕ := 12

-- Theorem statement (equivalent to the problem statement)
theorem avery_egg_cartons_filled : (total_chickens * eggs_per_chicken) / eggs_per_carton = 10 :=
by
  -- Proof omitted; sorry used to denote unfinished proof
  sorry

end avery_egg_cartons_filled_l381_381197


namespace part1_part2_l381_381125

noncomputable def seq_a : ℕ → ℕ
| 0     := 1
| (n+1) := seq_a n + 2 * seq_b n

noncomputable def seq_b : ℕ → ℕ
| 0     := 1
| (n+1) := seq_a n + seq_b n

theorem part1 (n : ℕ) : 
  (↑(seq_a (2 * n + 1)) : ℝ) / ↑(seq_b (2 * n + 1)) < real.sqrt 2 ∧ 
  (↑(seq_a (2 * n + 2)) : ℝ) / ↑(seq_b (2 * n + 2)) > real.sqrt 2 :=
sorry

theorem part2 (n : ℕ) :
  abs ((↑(seq_a (n + 1)) : ℝ) / ↑(seq_b (n + 1)) - real.sqrt 2) < 
  abs ((↑(seq_a n) : ℝ) / ↑(seq_b n) - real.sqrt 2) :=
sorry

end part1_part2_l381_381125


namespace magnitude_question_l381_381360

variables {V : Type} [inner_product_space ℝ V]

theorem magnitude_question (a b : V) (h1 : inner a b = 0) (h2 : ∥a∥ = 1) (h3 : ∥b∥ = 2) : ∥2 • a - b∥ = 2 * Real.sqrt 2 :=
by 
  sorry

end magnitude_question_l381_381360


namespace allison_craft_items_l381_381174

def glue_sticks (A B : Nat) : Prop := A = B + 8
def construction_paper (A B : Nat) : Prop := B = 6 * A

theorem allison_craft_items (Marie_glue_sticks Marie_paper_packs : Nat)
    (h1 : Marie_glue_sticks = 15)
    (h2 : Marie_paper_packs = 30) :
    ∃ (Allison_glue_sticks Allison_paper_packs total_items : Nat),
        glue_sticks Allison_glue_sticks Marie_glue_sticks ∧
        construction_paper Allison_paper_packs Marie_paper_packs ∧
        total_items = Allison_glue_sticks + Allison_paper_packs ∧
        total_items = 28 :=
by
    sorry

end allison_craft_items_l381_381174


namespace sufficient_but_not_necessary_condition_l381_381805

noncomputable def are_parallel (a : ℝ) : Prop :=
  (2 + a) * a * 3 * a = 3 * a * (a - 2)

theorem sufficient_but_not_necessary_condition :
  (are_parallel 4) ∧ (∃ a ≠ 4, are_parallel a) :=
by {
  sorry
}

end sufficient_but_not_necessary_condition_l381_381805


namespace problem_statement_l381_381976

open Set

variable (U : Set ℝ) (M N : Set ℝ)

theorem problem_statement (hU : U = univ) (hM : M = {x | x < 1}) (hN : N = {x | -1 < x ∧ x < 2}) :
  {x | 2 ≤ x} = compl (M ∪ N) :=
sorry

end problem_statement_l381_381976


namespace diameter_segments_division_l381_381894

-- Assuming the given conditions
def radius : ℝ := 6
def chord_length : ℝ := 10
def perpendicular_diameters (A B C D : Point) (O : Point) : Prop :=
    O = midpoint A B ∧ O = midpoint C D ∧ perpendicular A B C D

-- Define the final lengths to be proven
def segment1 : ℝ := 6 - real.sqrt 11
def segment2 : ℝ := 6 + real.sqrt 11

-- The theorem which encapsulates the given problem statement
theorem diameter_segments_division {A B C D H K : Point} (O : Point) 
    (radiusO : ∀ P, dist O P = radius) (perp_diams: perpendicular_diameters A B C D) 
    (chord_C_H (C H : Point) : dist C H = chord_length ∧ chord_intersects C H A B K) :
    ∃ AN NB : ℝ, AN = segment1 ∧ NB = segment2 :=
by
    sorry

end diameter_segments_division_l381_381894


namespace estimated_number_of_species_l381_381904

variables (N : ℕ) -- Define N as a natural number
variables (marked_initial : ℕ) (sample_total : ℕ) (sample_marked : ℕ)

-- Define the conditions from the problem
def condition_marked_initial : Prop := marked_initial = 1200
def condition_sample_total : Prop := sample_total = 1000
def condition_sample_marked : Prop := sample_marked = 100
def proportion_assumption : Prop := sample_marked * N = sample_total * marked_initial

-- Define the main theorem to be proved
theorem estimated_number_of_species (h1 : condition_marked_initial) (h2 : condition_sample_total) (h3 : condition_sample_marked) (h4 : proportion_assumption) : N = 12000 :=
by
  -- Proof steps would go here.
  sorry

end estimated_number_of_species_l381_381904


namespace avery_egg_cartons_filled_l381_381199

-- Definitions (conditions identified in step a)
def total_chickens : ℕ := 20
def eggs_per_chicken : ℕ := 6
def eggs_per_carton : ℕ := 12

-- Theorem statement (equivalent to the problem statement)
theorem avery_egg_cartons_filled : (total_chickens * eggs_per_chicken) / eggs_per_carton = 10 :=
by
  -- Proof omitted; sorry used to denote unfinished proof
  sorry

end avery_egg_cartons_filled_l381_381199


namespace base_7_representation_has_three_consecutive_digits_l381_381410

theorem base_7_representation_has_three_consecutive_digits :
  ∃ (n : ℕ), (n = 124) → 
  let base := 7 in
  let repr := nat.digits base n in
  repr.length = 3 ∧ repr.nth 0 = some 2 ∧ repr.nth 1 = some 3 ∧ repr.nth 2 = some 5 :=
by
  let n := 124
  let base := 7
  let repr := nat.digits base n
  have h_repr : repr = [2, 3, 5] := sorry
  show ∃ (n : ℕ), (n = 124) → repr.length = 3 ∧ repr.nth 0 = some 2 ∧ repr.nth 1 = some 3 ∧ repr.nth 2 = some 5
    from ⟨124, fun h => by rw [h, h_repr]; exact ⟨rfl, rfl, rfl, rfl⟩⟩

end base_7_representation_has_three_consecutive_digits_l381_381410


namespace correct_assignment_l381_381770

structure GirlDressAssignment :=
  (Katya : String)
  (Olya : String)
  (Liza : String)
  (Rita : String)

def solution : GirlDressAssignment :=
  ⟨"Green", "Blue", "Pink", "Yellow"⟩

theorem correct_assignment
  (Katya_not_pink_or_blue : solution.Katya ≠ "Pink" ∧ solution.Katya ≠ "Blue")
  (Green_between_Liza_and_Yellow : 
    (solution.Katya = "Green" ∧ solution.Liza = "Pink" ∧ solution.Rita = "Yellow") ∧
    (solution.Katya = "Green" ∧ solution.Rita = "Yellow" ∧ solution.Liza = "Pink"))
  (Rita_not_green_or_blue : solution.Rita ≠ "Green" ∧ solution.Rita ≠ "Blue")
  (Olya_between_Rita_and_Pink : 
    (solution.Olya = "Blue" ∧ solution.Rita = "Yellow" ∧ solution.Liza = "Pink") ∧
    (solution.Olya = "Blue" ∧ solution.Liza = "Pink" ∧ solution.Rita = "Yellow"))
  : solution = ⟨"Green", "Blue", "Pink", "Yellow"⟩ := by
  sorry

end correct_assignment_l381_381770


namespace dress_assignment_l381_381749

variables {Girl : Type} [Finite Girl]
variables (Katya Olya Liza Rita Pink Green Yellow Blue : Girl)
variables (standing_between : Girl → Girl → Girl → Prop)

-- Conditions
variable (cond1 : Katya ≠ Pink ∧ Katya ≠ Blue)
variable (cond2 : standing_between Green Liza Yellow)
variable (cond3 : Rita ≠ Green ∧ Rita ≠ Blue)
variable (cond4 : standing_between Olya Rita Pink)

-- Theorem statement
theorem dress_assignment :
  Katya = Green ∧ Olya = Blue ∧ Liza = Pink ∧ Rita = Yellow := 
sorry

end dress_assignment_l381_381749


namespace tommy_blocks_south_l381_381096

theorem tommy_blocks_south
    (North : ℕ)
    (East : ℕ)
    (West : ℕ)
    (FriendArea : ℕ)
    (Multiplier : ℕ)
    (s : ℕ)
    (EffectiveEast : ℕ) :
    North = 2 →
    East = 3 →
    West = 2 →
    FriendArea = 80 →
    Multiplier = 4 →
    EffectiveEast = East - West →
    4 * (North + s) = FriendArea →
    s = 18 :=
by
  intros hN hE hW hF hM hEffectiveEast hEquation
  rw [hN, hE, hW] at hEffectiveEast
  cases hEffectiveEast -- Ensures East - West = 1
  rw [hF, hM] at hEquation
  linarith

end tommy_blocks_south_l381_381096


namespace find_second_number_l381_381623

theorem find_second_number (a : ℕ) (c : ℕ) (x : ℕ) : 
  3 * a + 3 * x + 3 * c + 11 = 170 → a = 16 → c = 20 → x = 17 := 
by
  intros h1 h2 h3
  rw [h2, h3] at h1
  simp at h1
  sorry

end find_second_number_l381_381623


namespace complement_intersection_l381_381836

-- Conditions
def U : Set Int := {-1, 0, 1, 2, 3}
def A : Set Int := {-1, 0}
def B : Set Int := {0, 1, 2}

-- Theorem statement (proof not included)
theorem complement_intersection :
  let C_UA : Set Int := U \ A
  (C_UA ∩ B) = {1, 2} := 
by
  sorry

end complement_intersection_l381_381836


namespace battery_change_month_battery_change_in_november_l381_381364

theorem battery_change_month :
  (119 % 12) = 11 := by
  sorry

theorem battery_change_in_november (n : Nat) (h1 : n = 18) :
  let month := ((n - 1) * 7) % 12
  month = 11 := by
  sorry

end battery_change_month_battery_change_in_november_l381_381364


namespace cookout_2006_kids_l381_381386

def kids_2004 : ℕ := 60
def kids_2005 : ℕ := kids_2004 / 2
def kids_2006 : ℕ := (2 * kids_2005) / 3

theorem cookout_2006_kids : kids_2006 = 20 := by
  sorry

end cookout_2006_kids_l381_381386


namespace find_x_value_l381_381337

open Complex

theorem find_x_value (x : ℝ) (h : x^2 + x - 2 + (x^2 - 3*x + 2) * Complex.i = Conj (4 - 20 * Complex.i)) : x = -3 :=
by
  sorry

end find_x_value_l381_381337


namespace complement_union_eq_ge2_l381_381945

open Set

variables {U : Type} [PartialOrder U] [LinearOrder U]

def U : Set ℝ := univ
def M : Set ℝ := { x | x < 1 }
def N : Set ℝ := { x | -1 < x ∧ x < 2 }
def Complement_U (A : Set ℝ) : Set ℝ := U \ A

theorem complement_union_eq_ge2 : 
  Complement_U (M ∪ N) = { x : ℝ | x ≥ 2 } :=
by {
  sorry
}

end complement_union_eq_ge2_l381_381945


namespace max_students_per_class_l381_381545

-- Definitions used in Lean 4 statement:
def num_students := 920
def seats_per_bus := 71
def num_buses := 16

-- The main statement, showing this is the maximum value such that each class stays together within the given constraints.
theorem max_students_per_class : ∃ k, (∀ k' : ℕ, k' > k → 
  ¬∃ (classes : ℕ), classes * k' + (num_students - classes * k') ≤ seats_per_bus * num_buses ∧ k' <= seats_per_bus) ∧ k = 17 := 
by sorry

end max_students_per_class_l381_381545


namespace smallest_c_d_sum_l381_381325

theorem smallest_c_d_sum : ∃ (c d : ℕ), 2^12 * 7^6 = c^d ∧  (∀ (c' d' : ℕ), 2^12 * 7^6 = c'^d'  → (c + d) ≤ (c' + d')) ∧ c + d = 21954 := by
  sorry

end smallest_c_d_sum_l381_381325


namespace woman_l381_381151

open Real

def man_speed : ℝ := 5 -- miles per hour
def time_to_stop : ℝ := 5 / 60 -- hours
def wait_time : ℝ := 20 / 60 -- hours
def total_time : ℝ := (5 + 20) / 60 -- hours

theorem woman's_speed :
  ∃ woman_speed : ℝ, (woman_speed * time_to_stop = man_speed * total_time) ∧ woman_speed = 25 :=
by 
  -- placeholder for the actual proof
  sorry

end woman_l381_381151


namespace correct_proposition_is_D_l381_381825

-- Represents the propositions as Lean definitions
def proposition_A : Prop := ∀ (R2 : ℝ), R2 = 0.80 → (predictor contributes 80% to explained variable)
def proposition_B : Prop := ∀ (table : ℕ × ℕ → ℕ), 
                            (2 × 2 contingency table of two variables) → 
                            (larger difference in product of diagonal data) → 
                            (greater likelihood of no relationship)
def proposition_C : Prop := ∀ (R2 : ℝ), (correlation coefficient R2) → 
                            (smaller R2) → (larger residual sum of squares) → 
                            (better model fit)
def proposition_D : Prop := ∀ (r : ℝ), (|r| closer to 1) → 
                            (stronger linear correlation between variables)

-- The main statement to prove
theorem correct_proposition_is_D (hA : proposition_A) (hB : proposition_B) 
                                 (hC : proposition_C) (hD : proposition_D) : 
  hD ∧ ¬hA ∧ ¬hB ∧ ¬hC :=
by
  sorry

end correct_proposition_is_D_l381_381825


namespace question_proof_l381_381984

open Set

variable (U : Set ℝ := univ)
variable (M : Set ℝ := {x | x < 1})
variable (N : Set ℝ := {x | -1 < x ∧ x < 2})

theorem question_proof : {x | x ≥ 2} = compl (M ∪ N) :=
by
  sorry

end question_proof_l381_381984


namespace avg_other_days_visitors_l381_381147

theorem avg_other_days_visitors 
  (avg_sunday : ℕ) (total_days : ℕ) (sunday_count : ℕ) (avg_per_day : ℕ) 
  (other_days : ℕ) (total_avg_visitors : ℕ) (total_sundays_visitors : ℕ) 
  (total_other_days : ℕ) (total_days_visitors : ℕ) (V: ℕ) :
  avg_sunday = 540 →
  total_days = 30 →
  sunday_count = 5 →
  avg_per_day = 290 →
  other_days = total_days - sunday_count →
  total_avg_visitors = total_days * avg_per_day →
  total_sundays_visitors = sunday_count * avg_sunday →
  total_other_days = total_days - sunday_count →
  total_days_visitors = total_avg_visitors →
  total_sundays_visitors + other_days * V = total_days_visitors →
  V = 240 :=
by 
  intros h1 h2 h3 h4 h5 h6 h7 h8 h9 h10 h11
  sorry

end avg_other_days_visitors_l381_381147


namespace square_three_times_side_length_l381_381260

theorem square_three_times_side_length (a : ℝ) : 
  ∃ s, s = a * Real.sqrt 3 ∧ s ^ 2 = 3 * a ^ 2 := 
by 
  sorry

end square_three_times_side_length_l381_381260


namespace second_solution_volume_l381_381621

theorem second_solution_volume
  (V : ℝ)
  (h1 : 0.20 * 6 + 0.60 * V = 0.36 * (6 + V)) : 
  V = 4 :=
sorry

end second_solution_volume_l381_381621


namespace dress_assignment_l381_381763

-- Define the four girls
inductive Girl
| Katya
| Olya
| Liza
| Rita

-- Define the four dresses
inductive Dress
| Pink
| Green
| Yellow
| Blue

-- Define the function that assigns each girl a dress
def dressOf : Girl → Dress

-- Conditions as definitions
axiom KatyaNotPink : dressOf Girl.Katya ≠ Dress.Pink
axiom KatyaNotBlue : dressOf Girl.Katya ≠ Dress.Blue
axiom RitaNotGreen : dressOf Girl.Rita ≠ Dress.Green
axiom RitaNotBlue : dressOf Girl.Rita ≠ Dress.Blue

axiom GreenBetweenLizaAndYellow : 
  ∃ (arrangement : List Girl), 
    arrangement = [Girl.Liza, Girl.Katya, Girl.Rita] ∧ 
    (dressOf Girl.Katya = Dress.Green ∧ 
    dressOf Girl.Liza = Dress.Pink ∧ 
    dressOf Girl.Rita = Dress.Yellow)

axiom OlyaBetweenRitaAndPink : 
  ∃ (arrangement : List Girl),
    arrangement = [Girl.Rita, Girl.Olya, Girl.Liza] ∧ 
    (dressOf Girl.Olya = Dress.Blue ∧ 
     dressOf Girl.Rita = Dress.Yellow ∧ 
     dressOf Girl.Liza = Dress.Pink)

-- Problem: Determine the dress assignments
theorem dress_assignment : 
  dressOf Girl.Katya = Dress.Green ∧ 
  dressOf Girl.Olya = Dress.Blue ∧ 
  dressOf Girl.Liza = Dress.Pink ∧ 
  dressOf Girl.Rita = Dress.Yellow :=
sorry

end dress_assignment_l381_381763


namespace question_proof_l381_381988

open Set

variable (U : Set ℝ := univ)
variable (M : Set ℝ := {x | x < 1})
variable (N : Set ℝ := {x | -1 < x ∧ x < 2})

theorem question_proof : {x | x ≥ 2} = compl (M ∪ N) :=
by
  sorry

end question_proof_l381_381988


namespace find_initial_population_l381_381609

theorem find_initial_population
  (birth_rate : ℕ)
  (death_rate : ℕ)
  (net_growth_rate_percent : ℝ)
  (net_growth_rate_per_person : ℕ)
  (h1 : birth_rate = 32)
  (h2 : death_rate = 11)
  (h3 : net_growth_rate_percent = 2.1)
  (h4 : net_growth_rate_per_person = birth_rate - death_rate)
  (h5 : (net_growth_rate_per_person : ℝ) / 100 = net_growth_rate_percent / 100) :
  P = 1000 :=
by
  sorry

end find_initial_population_l381_381609


namespace exactly_three_correct_is_impossible_l381_381087

theorem exactly_three_correct_is_impossible (n : ℕ) (hn : n = 5) (f : Fin n → Fin n) :
  (∃ S : Finset (Fin n), S.card = 3 ∧ ∀ i ∈ S, f i = i) → False :=
by
  intros h
  sorry

end exactly_three_correct_is_impossible_l381_381087


namespace jack_paycheck_l381_381917

theorem jack_paycheck (P : ℝ) (h1 : 0.15 * 150 + 0.25 * (P - 150) + 30 + 70 / 100 * (P - (0.15 * 150 + 0.25 * (P - 150) + 30)) * 30 / 100 = 50) : P = 242.22 :=
sorry

end jack_paycheck_l381_381917


namespace find_2023rd_letter_in_sequence_l381_381565

-- Define the sequence as a list of characters
def sequence : List Char := ['F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'Y', 'X', 'W', 'V', 'U', 'T', 'S', 'R', 'Q', 'P', 'O', 'N', 'M', 'L', 'K', 'J', 'I', 'H', 'G', 'F']

-- Define the length of the sequence
def len_sequence : Nat := sequence.length

-- Define the position we want to find
def position : Nat := 2023

-- Calculate the zero-indexed position within one cycle of the sequence
def pos_in_cycle : Nat := (position - 1) % len_sequence

-- State the theorem
theorem find_2023rd_letter_in_sequence : sequence.get! pos_in_cycle = 'T' :=
by
  -- Proof goes here
  sorry

end find_2023rd_letter_in_sequence_l381_381565


namespace rate_of_speed_for_second_half_l381_381643

-- Definitions based on conditions
def total_time : ℝ := 20
def half_distance : ℝ := 240 / 2
def first_half_speed : ℝ := 10
def total_distance : ℝ := 240

-- The Lean statement to prove the rate of speed for the second half of the journey
theorem rate_of_speed_for_second_half : 
  let first_half_time := half_distance / first_half_speed in
  let second_half_time := total_time - first_half_time in
  let second_half_speed := half_distance / second_half_time in
  second_half_speed = 15 := by
  sorry

end rate_of_speed_for_second_half_l381_381643


namespace inequality_2n_squared_plus_3n_plus_1_l381_381467

theorem inequality_2n_squared_plus_3n_plus_1 (n : ℕ) (h: n > 0) : (2 * n^2 + 3 * n + 1)^n ≥ 6^n * (n! * n!) := 
by sorry

end inequality_2n_squared_plus_3n_plus_1_l381_381467


namespace average_children_in_families_with_children_l381_381099

theorem average_children_in_families_with_children : 
  ∀ (total_families : ℕ) (average_children_per_family : ℕ) (childless_families : ℕ), 
  total_families = 12 →
  average_children_per_family = 3 →
  childless_families = 3 →
  let total_children := total_families * average_children_per_family in
  let families_with_children := total_families - childless_families in
  let average_children_in_families_with_children := total_children / families_with_children in
  average_children_in_families_with_children = 4.0 :=
by
  intros total_families average_children_per_family childless_families h1 h2 h3
  let total_children := total_families * average_children_per_family
  let families_with_children := total_families - childless_families
  let average_children_in_families_with_children := total_children / families_with_children
  sorry

end average_children_in_families_with_children_l381_381099


namespace max_value_sqrt_sum_l381_381434

theorem max_value_sqrt_sum (x y z : ℝ) 
  (h1 : x + y + z = 2)
  (hx : x ≥ -2/3)
  (hy : y ≥ -1)
  (hz : z ≥ -2) :
  (∃ x y z ∈ ℝ, x + y + z = 2 ∧ x ≥ -2/3 ∧ y ≥ -1 ∧ z ≥ -2 ∧ (sqrt(3*x + 2) + sqrt(3*y + 4) + sqrt(3*z + 7)) = sqrt(57)) :=
sorry

end max_value_sqrt_sum_l381_381434


namespace pairwise_coprime_exists_n_eq_3_no_pairwise_coprime_exists_n_ge_4_l381_381031

theorem pairwise_coprime_exists_n_eq_3
  (a1 a2 a3 : ℕ)
  (ha1 : 0 < a1)
  (ha2 : 0 < a2)
  (ha3 : 0 < a3)
  (coprime_a : Nat.Coprime a1 a2 ∧ Nat.Coprime a1 a3 ∧ Nat.Coprime a2 a3)
  (exists_k : ∃ k1 k2 k3 : ℤ, k1 ∈ { -1, 1 } ∧ k2 ∈ { -1, 1 } ∧ k3 ∈ { -1, 1 } ∧ k1 * a1 + k2 * a2 + k3 * a3 = 0) :
  ∃ b1 b2 b3 : ℕ, 0 < b1 ∧ 0 < b2 ∧ 0 < b3 ∧ ∀ k : ℕ, Nat.Coprime (b1 + k * a1) (b2 + k * a2) ∧ Nat.Coprime (b1 + k * a1) (b3 + k * a3) ∧ Nat.Coprime (b2 + k * a2) (b3 + k * a3) :=
sorry

theorem no_pairwise_coprime_exists_n_ge_4
  (n : ℕ)
  (hn : n ≥ 4)
  (a : Fin n → ℕ)
  (ha : ∀ i, 0 < a i)
  (coprime_a : ∀ i j, i ≠ j → Nat.Coprime (a i) (a j))
  (exists_k : ∃ k : Fin n → ℤ, (∀ i : Fin n, k i ∈ { -1, 1 }) ∧ (Finset.univ.sum (λ i, k i * a i) = 0)) :
  ¬ ∃ b : Fin n → ℕ, (∀ i, 0 < b i) ∧ (∀ k : ℕ, ∀ i j, i ≠ j → Nat.Coprime (b i + k * a i) (b j + k * a j)) :=
sorry

end pairwise_coprime_exists_n_eq_3_no_pairwise_coprime_exists_n_ge_4_l381_381031


namespace num_pos_3_digits_div_by_7_l381_381858

theorem num_pos_3_digits_div_by_7 : 
  let lower_bound := 100
  let upper_bound := 999 
  let divisor := 7
  let smallest_3_digit := 105 -- or can be computed explicitly by: (lower_bound + divisor - 1) / divisor * divisor
  let largest_3_digit := 994  -- or can be computed explicitly by: upper_bound / divisor * divisor
  List.length (List.filter (λ n, n % divisor = 0) (List.range' smallest_3_digit (largest_3_digit + 1))) = 128 :=
by
  sorry

end num_pos_3_digits_div_by_7_l381_381858


namespace verify_correct_propositions_l381_381257

structure Vector := (x : ℝ) (y : ℝ) (z : ℝ)

def dot_product (v1 v2 : Vector) : ℝ := 
  v1.x * v2.x + v1.y * v2.y + v1.z * v2.z

def is_perpendicular (v1 v2 : Vector) : Prop :=
  dot_product(v1, v2) = 0

def equal_vectors (v1 v2 : Vector) : Prop :=
  v1.x = v2.x ∧ v1.y = v2.y ∧ v1.z = v2.z

def proposition_1 : Prop := 
  let a := Vector.mk 1 (-1) 2 in
  let b := Vector.mk 2 1 (-1/2) in
  is_perpendicular a b

def proposition_2 : Prop :=
  let a := Vector.mk 0 1 (-1) in
  let n := Vector.mk 1 (-1) (-1) in
  is_perpendicular a n ∧ ∃ (α : Vector → Prop), α a → α = n

def proposition_3 : Prop := 
  let n1 := Vector.mk 0 1 3 in
  let n2 := Vector.mk 1 0 2 in
  ∀ k : ℝ, equal_vectors n1 { x := k * n1.x, y := k * n1.y, z := k * n1.z }

def proposition_4 : Prop :=
  let ab := Vector.mk (-1) 1 1 in
  let bc := Vector.mk (-1) 1 0 in
  ∀ u t : ℝ, ab = Vector.mk 1 u t ∧ bc = Vector.mk 1 u t ∧ (u + t = 1)

def correct_propositions : list ℕ := [1, 4]

theorem verify_correct_propositions : correct_propositions = [1, 4] := 
  by 
    sorry

end verify_correct_propositions_l381_381257


namespace least_workers_needed_l381_381683

theorem least_workers_needed 
  (total_days : ℕ) 
  (days_worked : ℕ) 
  (initial_workers : ℕ) 
  (fraction_completed : ℚ) 
  (total_job : ℚ) 
  (remaining_days : ℕ) 
  (remaining_fraction : ℚ) 
  (worker_rate : ℚ) 
  (N : ℕ) : 
  total_days = 30 
  ∧ days_worked = 6 
  ∧ initial_workers = 10 
  ∧ fraction_completed = 1/4 
  ∧ total_job = 1 
  ∧ remaining_days = total_days - days_worked 
  ∧ remaining_fraction = total_job - fraction_completed 
  ∧ worker_rate = fraction_completed / (days_worked * initial_workers) 
  ∧ N = (remaining_fraction * 40) / 24 →
  ⌈N⌉ = 4 :=
by
  sorry

end least_workers_needed_l381_381683


namespace ace_then_king_probability_l381_381560

noncomputable def card_deck := {total_cards : ℕ, hearts : ℕ, clubs : ℕ, spades : ℕ, diamonds : ℕ, aces : ℕ, kings : ℕ}

noncomputable def modified_deck : card_deck :=
  { total_cards := 54,
    hearts := 14,
    clubs := 14,
    spades := 13,
    diamonds := 13,
    aces := 5,
    kings := 4 }

noncomputable def probability_first_ace (deck : card_deck) : ℚ :=
  deck.aces / deck.total_cards

noncomputable def probability_second_king (deck : card_deck) : ℚ :=
  deck.kings / (deck.total_cards - 1)

noncomputable def probability_ace_then_king (deck : card_deck) : ℚ :=
  probability_first_ace deck * probability_second_king deck

theorem ace_then_king_probability :
  probability_ace_then_king modified_deck = 10 / 1426 :=
by {
  sorry
}

end ace_then_king_probability_l381_381560


namespace sum_of_powers_of_2_and_mersenne_primes_is_sum_of_squares_l381_381333

theorem sum_of_powers_of_2_and_mersenne_primes_is_sum_of_squares 
  (n : ℕ)
  (a b c d : ℕ) 
  (h1 : n = 2^a + 2^b) 
  (h2 : a ≠ b) 
  (h3 : n = (2^c - 1) + (2^d - 1)) 
  (h4 : c ≠ d)
  (h5 : Nat.Prime (2^c - 1)) 
  (h6 : Nat.Prime (2^d - 1)) : 
  ∃ x y : ℕ, x ≠ y ∧ n = x^2 + y^2 := 
by
  sorry

end sum_of_powers_of_2_and_mersenne_primes_is_sum_of_squares_l381_381333


namespace solve_for_x_in_exp_eq_l381_381482

theorem solve_for_x_in_exp_eq (x : ℝ) :
  2 ^ (2 * x) - 6 * 2 ^ x + 8 = 0 ↔ x = 1 ∨ x = 2 :=
by
  sorry

end solve_for_x_in_exp_eq_l381_381482


namespace num_trucks_l381_381557

variables (T : ℕ) (num_cars : ℕ := 13) (total_wheels : ℕ := 100) (wheels_per_vehicle : ℕ := 4)

theorem num_trucks :
  (num_cars * wheels_per_vehicle + T * wheels_per_vehicle = total_wheels) -> T = 12 :=
by
  intro h
  -- skipping the proof implementation
  sorry

end num_trucks_l381_381557


namespace initially_calculated_average_l381_381497

theorem initially_calculated_average :
  ∀ (S_wrong S_correct : ℝ),
    (∃ l : list ℝ, l.length = 10 ∧ sum l = S_correct) →
    (S_correct / 10 = 16) →
    (S_wrong = S_correct - (36 - 26)) →
    (S_wrong / 10 = 15) :=
by
  sorry

end initially_calculated_average_l381_381497


namespace dress_assignments_l381_381743

structure GirlDress : Type :=
  (Katya Olya Liza Rita : String)

def dresses := ["Pink", "Green", "Yellow", "Blue"]

axiom not_pink_or_blue : GirlDress.Katya ≠ "Pink" ∧ GirlDress.Katya ≠ "Blue"
axiom green_between_liza_yellow : (GirlDress.Liza = "Pink" ∨ GirlDress.Rita = "Yellow") ∧
                                  GirlDress.Katya = "Green" ∧
                                  GirlDress.Rita = "Yellow" ∧ GirlDress.Liza = "Pink"
axiom not_green_or_blue : GirlDress.Rita ≠ "Green" ∧ GirlDress.Rita ≠ "Blue"
axiom olya_between_rita_pink : GirlDress.Olya ≠ "Pink" → GirlDress.Rita ≠ "Pink" → GirlDress.Liza = "Pink"

theorem dress_assignments (gd : GirlDress) :
  gd.Katya = "Green" ∧ gd.Olya = "Blue" ∧ gd.Liza = "Pink" ∧ gd.Rita = "Yellow" :=
by
  sorry

end dress_assignments_l381_381743


namespace num_pos_3_digits_div_by_7_l381_381856

theorem num_pos_3_digits_div_by_7 : 
  let lower_bound := 100
  let upper_bound := 999 
  let divisor := 7
  let smallest_3_digit := 105 -- or can be computed explicitly by: (lower_bound + divisor - 1) / divisor * divisor
  let largest_3_digit := 994  -- or can be computed explicitly by: upper_bound / divisor * divisor
  List.length (List.filter (λ n, n % divisor = 0) (List.range' smallest_3_digit (largest_3_digit + 1))) = 128 :=
by
  sorry

end num_pos_3_digits_div_by_7_l381_381856


namespace common_difference_of_arithmetic_sequence_l381_381529

variable {α : Type*} [LinearOrderedField α]

def arithmetic_sequence_sum (n : ℕ) (an : ℕ → α) : α :=
  (n : α) * an 1 + (n * (n - 1) / 2 * (an 2 - an 1))

theorem common_difference_of_arithmetic_sequence (S : ℕ → ℕ) (d : ℕ) (a1 a2 : ℕ)
  (h1 : ∀ n, S n = 4 * n ^ 2 - n)
  (h2 : a1 = S 1)
  (h3 : a2 = S 2 - S 1) :
  d = a2 - a1 → d = 8 := by
  sorry

end common_difference_of_arithmetic_sequence_l381_381529


namespace slope_of_line_l381_381207

def point1 : ℝ × ℝ := (2, -3)
def point2 : ℝ × ℝ := (-3, 4)

def slope (p1 p2 : ℝ × ℝ) : ℝ :=
  (p2.2 - p1.2) / (p2.1 - p1.1)

theorem slope_of_line :
  slope point1 point2 = -7/5 :=
by
  sorry

end slope_of_line_l381_381207


namespace sum_c_p_eq_5760_l381_381724

-- Define c(p) as the unique integer k such that |k - ∛p| < 1/2
def c (p : ℕ) : ℕ := if h : p > 0 then sorry else 0

theorem sum_c_p_eq_5760 : (∑ p in finset.range 1730, c p) = 5760 :=
by {
  sorry
}

end sum_c_p_eq_5760_l381_381724


namespace remainder_of_87_pow_88_plus_7_l381_381067

theorem remainder_of_87_pow_88_plus_7 :
  (87^88 + 7) % 88 = 8 :=
by sorry

end remainder_of_87_pow_88_plus_7_l381_381067


namespace area_triangle_AMN_l381_381559

-- Define points A, B, C
variables (A B C O M N : Type)

-- Define lengths AB, BC, and AC
variables (AB BC AC : ℝ)

-- Conditions
def conditions := AB = 15 ∧ BC = 30 ∧ AC = 22 ∧ 
                  (* insertion of additional conditions for incenter, parallel lines, and intersections *)

-- Define the incenter and the line through it parallel to BC
def incenter_condition := (* definition of the incenter properties and parallel condition *)

-- Define the area calculation
def area_ABC := (let s := (15 + 30 + 22) / 2 
                 in Real.sqrt (s * (s - 15) * (s - 30) * (s - 22)))

def area_AMN := area_ABC / 4

-- Lean statement to prove  
theorem area_triangle_AMN : 
  conditions → incenter_condition → 
  area_AMN = 21.98175 :=
by
  sorry

end area_triangle_AMN_l381_381559


namespace largest_percentage_increase_l381_381192
open Real

def student_counts : List (ℕ × ℕ) := [(2010, 120), (2011, 130), (2012, 150), (2013, 155), (2014, 160), (2015, 140), (2016, 150)]

def percentage_increase (prev next : ℕ) : ℝ :=
  ((next - prev) / prev.toReal) * 100

theorem largest_percentage_increase :
  ∃ prev_year next_year,
  (prev_year, next_year) ∈ [(2010, 2011), (2011, 2012), (2012, 2013), (2013, 2014), (2014, 2015), (2015, 2016)] ∧
  percentage_increase (student_counts.lookup prev_year).getD 0 (student_counts.lookup next_year).getD 0 
    > percentage_increase (student_counts.lookup 2010).getD 0 (student_counts.lookup 2011).getD 0 ∧
  percentage_increase (student_counts.lookup 2011).getD 0 (student_counts.lookup 2012).getD 0
    > percentage_increase (student_counts.lookup 2012).getD 0 (student_counts.lookup 2013).getD 0 ∧
  percentage_increase (student_counts.lookup 2012).getD 0 (student_counts.lookup 2013).getD 0
    > percentage_increase (student_counts.lookup 2013).getD 0 (student_counts.lookup 2014).getD 0 ∧
  percentage_increase (student_counts.lookup 2012).getD 0 (student_counts.lookup 2013).getD 0
    > percentage_increase (student_counts.lookup 2014).getD 0 (student_counts.lookup 2015).getD 0 ∧
  percentage_increase (student_counts.lookup 2012).getD 0 (student_counts.lookup 2013).getD 0
    > percentage_increase (student_counts.lookup 2015).getD 0 (student_counts.lookup 2016).getD 0 :=
by
  sorry

end largest_percentage_increase_l381_381192


namespace parker_total_stamps_l381_381551

-- Definitions based on conditions
def original_stamps := 430
def addie_stamps := 1890
def addie_fraction := 3 / 7
def stamps_added_by_addie := addie_fraction * addie_stamps

-- Theorem statement to prove the final number of stamps
theorem parker_total_stamps : original_stamps + stamps_added_by_addie = 1240 :=
by
  -- definitions instantiated above
  sorry  -- proof required

end parker_total_stamps_l381_381551


namespace problem_statement_l381_381021

-- Define the function f(x) as piecewise
def f (x : ℝ) (b c : ℝ) : ℝ := if x ≤ 0 then x^2 + b*x + c else 2

-- Given conditions
theorem problem_statement (b c : ℝ) (h1 : f (-4) b c = 2) (h2 : f (-2) b c = -2) : 
  ∃! x1 x2 x3 : ℝ, f x1 b c = x1 ∧ f x2 b c = x2 ∧ f x3 b c = x3 := sorry


end problem_statement_l381_381021


namespace find_side_AB_l381_381414

theorem find_side_AB 
  (A B C D E K: Type) -- The points of the triangle and related segments
  (h1 : height A B C D 6)       -- BD is 6
  (h2 : median A C B E 5)       -- CE is 5
  (h3 : distance_from_point_to_line K A C 1) -- Distance from K to AC is 1
  : side_length AB = (2 * Real.sqrt 145) / 3 :=
sorry

end find_side_AB_l381_381414


namespace original_price_sarees_l381_381069

/-- 
The sale price of sarees listed for some amount after successive discounts of 12% and 15% is Rs. 222.904.
What was the original price of the sarees?
-/
theorem original_price_sarees (sale_price : ℝ) (discount1 : ℝ) (discount2 : ℝ) (orig_price : ℝ) :
  sale_price = 222.904 →
  discount1 = 0.12 →
  discount2 = 0.15 →
  orig_price = 222.904 / ((1 - discount1) * (1 - discount2)) →
  orig_price = 297.86 :=
by
  intros h_sale_price h_discount1 h_discount2 h_orig_price
  rw [h_sale_price, h_discount1, h_discount2, h_orig_price]
  sorry

end original_price_sarees_l381_381069


namespace arrangement_count_l381_381625

-- Definitions based on conditions
def students := Fin 5

def stands_next_to (x y : students) (arrangement : List students) : Prop :=
  ∃ i, arrangement.nth i = some x ∧
       (arrangement.nth (i + 1) = some y ∨ arrangement.nth (i - 1) = some y)

def not_at_ends (x : students) (arrangement : List students) : Prop :=
  ∃ i, 0 < i ∧ i < 4 ∧ arrangement.nth i = some x

-- The math proof problem
theorem arrangement_count :
  ∃ arrangement : List students, 
  stands_next_to 1 2 arrangement ∧
  stands_next_to 2 1 arrangement ∧
  not_at_ends 1 arrangement ∧
  arrangement.length = 5 →
  (count_arrangements 5 1 36) :=
sorry

-- Dummy function for counting arrangements, to be replaced with the actual logic
def count_arrangements (n : ℕ) (x : ℕ) (y : ℕ) : Prop := true

end arrangement_count_l381_381625


namespace initial_employees_l381_381531

theorem initial_employees (E : ℕ)
  (salary_per_employee : ℕ)
  (laid_off_fraction : ℚ)
  (total_paid_remaining : ℕ)
  (remaining_employees : ℕ) :
  salary_per_employee = 2000 →
  laid_off_fraction = 1 / 3 →
  total_paid_remaining = 600000 →
  remaining_employees = total_paid_remaining / salary_per_employee →
  (2 / 3 : ℚ) * E = remaining_employees →
  E = 450 := by
  sorry

end initial_employees_l381_381531


namespace ML_parallel_BC_l381_381030

variables {A B C M N L : Type} [geometry A B C M N L]

axiom AM_eq_BN : AM = BN
axiom AMNC_cyclic : cyclic_quadrilateral A M N C
axiom BL_angle_bisector : angle_bisector B L

theorem ML_parallel_BC : parallel M L B C := 
by
  sorry

end ML_parallel_BC_l381_381030


namespace no_discount_profit_percentage_l381_381650

noncomputable def cost_price : ℝ := 100
noncomputable def discount_percentage : ℝ := 4 / 100  -- 4%
noncomputable def profit_percentage_with_discount : ℝ := 20 / 100  -- 20%

theorem no_discount_profit_percentage : 
  (1 + profit_percentage_with_discount) * cost_price / (1 - discount_percentage) / cost_price - 1 = 0.25 := by
  sorry

end no_discount_profit_percentage_l381_381650


namespace arrange_plants_l381_381664

theorem arrange_plants :
  let total_plants := 10 in
  let basil_plants := 5 in
  let tomato_plants := 5 in
  let group1_tomatoes := 2 in
  let group2_tomatoes := 3 in
  let entities := 7 in  -- 5 basil plants + 2 tomato groups
  let ways_to_arrange_entities := Nat.factorial entities in
  let choose_spots := Nat.choose entities basil_plants * Nat.choose 2 1 in
  let ways_to_arrange_group1 := Nat.factorial group1_tomatoes in
  let ways_to_arrange_group2 := Nat.factorial group2_tomatoes in
  ways_to_arrange_entities * choose_spots * ways_to_arrange_group1 * ways_to_arrange_group2 = 1_271_040 :=
begin
  sorry
end

end arrange_plants_l381_381664


namespace max_dn_eq_401_l381_381521

open BigOperators

def a (n : ℕ) : ℕ := 100 + n^2

def d (n : ℕ) : ℕ := Nat.gcd (a n) (a (n + 1))

theorem max_dn_eq_401 : ∃ n, d n = 401 ∧ ∀ m, d m ≤ 401 := by
  -- Proof will be filled here
  sorry

end max_dn_eq_401_l381_381521


namespace find_B_and_area_l381_381335

noncomputable def find_angle_B (a b c : ℝ) (B : ℝ) : Prop :=
  ∃ (A C : ℝ), (0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π ∧
  sqrt 3 * sin B + b * cos A = c ∧ B = π / 6)

noncomputable def triangle_area (a b c : ℝ) (A B C : ℝ) : ℝ :=
  1 / 2 * a * b * sin C

theorem find_B_and_area (a b c : ℝ) (A B C : ℝ)
  (h_a : a = sqrt 3 * c)
  (h_b : b = 2)
  (angles_pos : 0 < A ∧ 0 < B ∧ 0 < C)
  (angles_sum : A + B + C = π)
  (eqn : sqrt 3 * sin B + b * cos A = c) :
  (B = π / 6) ∧ (triangle_area a b c A B C = sqrt 3) :=
by
  sorry

end find_B_and_area_l381_381335


namespace count_3_digit_numbers_divisible_by_7_l381_381855

theorem count_3_digit_numbers_divisible_by_7 : 
  {n : ℕ | 100 ≤ n ∧ n ≤ 999 ∧ n % 7 = 0}.to_finset.card = 128 := 
  sorry

end count_3_digit_numbers_divisible_by_7_l381_381855


namespace problem_equivalent_l381_381940

def U : Set ℝ := Set.univ
def M : Set ℝ := {x | x < 1}
def N : Set ℝ := {x | -1 < x ∧ x < 2}

theorem problem_equivalent :
  {x : ℝ | x ≥ 2} = (U \ (M ∪ N)) := 
by sorry

end problem_equivalent_l381_381940


namespace log_sum_eq_l381_381250

theorem log_sum_eq : ((∑ k in (Finset.range 98).map (λ n, n+3), (Real.log (1 + 1 / (k:ℝ)) / Real.log 3) * (Real.log 3 / Real.log k) * (Real.log 3 / Real.log (k+1)))) = 1 - (1 / Real.log 101 / Real.log 3) :=
by
  sorry

end log_sum_eq_l381_381250


namespace stock_decrease_percent_l381_381651

theorem stock_decrease_percent (x : ℝ) (hx : x > 0) :
    ∃ (p : ℝ), p = (2 / 7) ∧ (1 - p) * (1.40 * x) = x :=
by
    have h : p = (2 / 7) ↔ (1 - p) * (1.40 * x) = x := 
        calc p = (2 / 7) 
            ↔ (1 - p) = (1 - (2 / 7)) : by sorry
            ↔ (1 - p) * (1.40 * x) = x : by sorry
    use (2 / 7)
    simp [h]
    sorry

end stock_decrease_percent_l381_381651


namespace irrational_number_among_given_l381_381660

-- Define the four given real numbers
def num1 := -7
def num2 := 3 / 7
def num3 := Real.sqrt 16
def num4 := Real.cbrt 9 

-- Define rationality and irrationality conditions
def is_rational (x : ℝ) : Prop := ∃ (a b : ℤ), b ≠ 0 ∧ x = a / b
def is_irrational (x : ℝ) : Prop := ¬is_rational x

-- State the main theorem
theorem irrational_number_among_given (n1 n2 n3 n4 : ℝ) :
  n1 = -7 → n2 = 3 / 7 → n3 = Real.sqrt 16 → n4 = Real.cbrt 9 →
  is_rational n1 ∧ is_rational n2 ∧ is_rational n3 ∧ is_irrational n4 := 
by
  sorry

end irrational_number_among_given_l381_381660


namespace total_price_correct_l381_381025

def original_price_jewelry := 30
def original_price_painting := 100
def price_increase_jewelry := 10
def price_increase_painting_percent := 0.20
def sales_tax_jewelry_percent := 0.06
def sales_tax_painting_percent := 0.08
def discount_percent := 0.10
def discount_conditions_met (n_jewelry n_paintings : ℕ) (total_price : ℝ) : Prop :=
  n_jewelry > 1 ∧ n_paintings >= 3 ∧ total_price >= 800
def n_jewelry := 2
def n_paintings := 5

theorem total_price_correct :
  let new_price_jewelry := original_price_jewelry + price_increase_jewelry
  let new_price_painting := original_price_painting + (price_increase_painting_percent * original_price_painting)
  let total_jewelry_price := new_price_jewelry * n_jewelry * (1 + sales_tax_jewelry_percent)
  let total_painting_price := new_price_painting * n_paintings * (1 + sales_tax_painting_percent)
  let total_price_before_discount := total_jewelry_price + total_painting_price
  let discount_applied_total := if discount_conditions_met n_jewelry n_paintings total_price_before_discount
                                then total_price_before_discount * (1 - discount_percent)
                                else total_price_before_discount
  discount_applied_total = 732.80 :=
by
  sorry

end total_price_correct_l381_381025


namespace perfect_square_divisor_probability_l381_381646

theorem perfect_square_divisor_probability :
  let n := 10!
  let total_divisors := (8 + 1) * (4 + 1) * (2 + 1) * (1 + 1)
  let perfect_square_divisors := (5) * (3) * (2) * (1)
  let prob := perfect_square_divisors / total_divisors
  let m := 1
  let n := 9
  m + n = 10 :=
by
  sorry

end perfect_square_divisor_probability_l381_381646


namespace sum_of_angles_l381_381877

noncomputable def alpha : ℝ := sorry -- Because we skip actual calculations.
noncomputable def beta : ℝ := sorry -- Because we skip actual calculations.

axiom obtuse_alpha : α > π / 2 ∧ α < π
axiom obtuse_beta : β > π / 2 ∧ β < π

axiom sin_alpha : Real.sin α = sqrt 5 / 5
axiom cos_beta : Real.cos β = -(3 * sqrt 10)/10

theorem sum_of_angles : α + β = 7 * π / 4 := 
by sorry

end sum_of_angles_l381_381877


namespace motorist_speed_second_half_l381_381644

theorem motorist_speed_second_half :
  ∀ (total_hours first_half_hours speed_first_half total_distance : ℕ),
  total_hours = 6 →
  first_half_hours = 3 →
  speed_first_half = 60 →
  total_distance = 324 →
  ((total_distance - speed_first_half * first_half_hours) / first_half_hours) = 48 :=
by
  intros total_hours first_half_hours speed_first_half total_distance
  intros h_total_hours h_first_half_hours h_speed_first_half h_total_distance
  rw [h_total_hours, h_first_half_hours, h_speed_first_half, h_total_distance]
  /-
    We skip the proof here.
    The expected result is:
    ((324 - 60 * 3) / 3) = 48
  -/
  sorry

end motorist_speed_second_half_l381_381644


namespace cookout_kids_2006_l381_381383

theorem cookout_kids_2006 :
  let kids_2004 := 60
  let kids_2005 := kids_2004 / 2
  let kids_2006 := (kids_2005 / 3) * 2
  in kids_2006 = 20 :=
by
  let kids_2004 := 60
  let kids_2005 := kids_2004 / 2
  let kids_2006 := (kids_2005 / 3) * 2
  have h : kids_2006 = 20 := sorry
  exact h

end cookout_kids_2006_l381_381383


namespace complement_union_M_N_eq_ge_2_l381_381958

def U := set.univ ℝ
def M := {x : ℝ | x < 1}
def N := {x : ℝ | -1 < x ∧ x < 2}

theorem complement_union_M_N_eq_ge_2 :
  (U \ (M ∪ N)) = {x : ℝ | 2 ≤ x} :=
by sorry

end complement_union_M_N_eq_ge_2_l381_381958


namespace digits_partition_impossible_l381_381210

theorem digits_partition_impossible : 
  ¬ ∃ (A B : Finset ℕ), 
    A.card = 4 ∧ B.card = 4 ∧ A ∪ B = {1, 2, 3, 4, 5, 7, 8, 9} ∧ A ∩ B = ∅ ∧ 
    A.sum id = B.sum id := 
by
  sorry

end digits_partition_impossible_l381_381210


namespace inequality_ge_half_l381_381332

open Real

theorem inequality_ge_half (n : ℕ) (n_pos : 0 < n) (a : Fin n → ℝ) 
  (a_pos : ∀ i : Fin n, 0 < a i)
  (sum_eq_one : (Finset.univ.sum (λ i => a i)) = 1) :
  (Finset.univ.sum (λ i : Fin n => a i ^ 2 / (a i + a ((i + 1) % n)))) ≥ 1 / 2 := 
sorry

end inequality_ge_half_l381_381332


namespace quadratic_z_and_u_l381_381354

variables (a b c α β γ : ℝ)
variable (d : ℝ)
variable (δ : ℝ)
variables (x₁ x₂ y₁ y₂ z₁ z₂ u₁ u₂ : ℝ)

-- Given conditions
variable (h_nonzero : a * α ≠ 0)
variable (h_discriminant1 : b^2 - 4 * a * c ≥ 0)
variable (h_discriminant2 : β^2 - 4 * α * γ ≥ 0)
variable (hx_roots_order : x₁ ≤ x₂)
variable (hy_roots_order : y₁ ≤ y₂)
variable (h_eq_discriminant1 : b^2 - 4 * a * c = d^2)
variable (h_eq_discriminant2 : β^2 - 4 * α * γ = δ^2)

-- Translate into mathematical constraints for the roots
variable (hx1 : x₁ = (-b - d) / (2 * a))
variable (hx2 : x₂ = (-b + d) / (2 * a))
variable (hy1 : y₁ = (-β - δ) / (2 * α))
variable (hy2 : y₂ = (-β + δ) / (2 * α))

-- Variables for polynomial equations roots
axiom h_z1 : z₁ = x₁ + y₁
axiom h_z2 : z₂ = x₂ + y₂
axiom h_u1 : u₁ = x₁ + y₂
axiom h_u2 : u₂ = x₂ + y₁

theorem quadratic_z_and_u :
  (2 * a * α) * z₂ * z₂ + 2 * (a * β + α * b) * z₁ + (2 * a * γ + 2 * α * c + b * β - d * δ) = 0 ∧
  (2 * a * α) * u₂ * u₂ + 2 * (a * β + α * b) * u₁ + (2 * a * γ + 2 * α * c + b * β + d * δ) = 0 := sorry

end quadratic_z_and_u_l381_381354


namespace triangle_similarity_l381_381265

variable {A B C A1 B1 C1 : Type} [LinearOrderedField A] [LinearOrderedField B]

def cyclicQuadrilateral (A B A1 B1 : Type) [LinearOrderedField A] [LinearOrderedField B] : Prop :=
  ∠ AB1B = 90 ∧ ∠ AA1B = 90 ∧ opposite_angles_sum_to_180 (AB1A1B)

theorem triangle_similarity (H : cyclicQuadrilateral A B A1 B1) : similar (triangle A1 B1 C) (triangle A B C) :=
sorry

end triangle_similarity_l381_381265


namespace zogian_words_count_l381_381028

theorem zogian_words_count :
  let num_letters := 6
  let max_word_length := 4
  let count_1_letter_words := num_letters
  let count_2_letter_words := num_letters ^ 2
  let count_3_letter_words := num_letters ^ 3
  let count_4_letter_words := num_letters ^ 4
  count_1_letter_words + count_2_letter_words + count_3_letter_words + count_4_letter_words = 1554 :=
by
  let num_letters := 6
  let max_word_length := 4
  let count_1_letter_words := num_letters
  let count_2_letter_words := num_letters ^ 2
  let count_3_letter_words := num_letters ^ 3
  let count_4_letter_words := num_letters ^ 4
  have h1 : count_1_letter_words = 6 := by rfl
  have h2 : count_2_letter_words = 36 := by norm_num
  have h3 : count_3_letter_words = 216 := by norm_num
  have h4 : count_4_letter_words = 1296 := by norm_num
  show 6 + 36 + 216 + 1296 = 1554 from by norm_num

end zogian_words_count_l381_381028


namespace num_pos_three_digit_div_by_seven_l381_381862

theorem num_pos_three_digit_div_by_seven : 
  ∃ n : ℕ, (∀ k : ℕ, k < n → (∃ m : ℕ, 100 ≤ 7 * m ∧ 7 * m ≤ 999)) ∧ n = 128 :=
by
  sorry

end num_pos_three_digit_div_by_seven_l381_381862


namespace distinct_roots_and_ratios_l381_381815

open Real

theorem distinct_roots_and_ratios (a b : ℝ) (h1 : a^2 - 3*a - 1 = 0) (h2 : b^2 - 3*b - 1 = 0) (h3 : a ≠ b) :
  b/a + a/b = -11 :=
sorry

end distinct_roots_and_ratios_l381_381815


namespace gcd_factorial_eight_nine_eq_8_factorial_l381_381289

theorem gcd_factorial_eight_nine_eq_8_factorial : Nat.gcd (Nat.factorial 8) (Nat.factorial 9) = Nat.factorial 8 := 
by 
  sorry

end gcd_factorial_eight_nine_eq_8_factorial_l381_381289


namespace sum_series_sum_of_squares_l381_381411

-- Sum of the series S_n = 1 * 2 + 2 * 3 + ... + n * (n + 1)
theorem sum_series (n : ℕ) : (Finset.range n).sum (λ k, (k + 1) * (k + 2)) = (n * (n + 1) * (n + 2)) / 3 :=
sorry

-- Sum of the squares 1^2 + 2^2 + ... + n^2
theorem sum_of_squares (n : ℕ) : (Finset.range n).sum (λ k, (k + 1) ^ 2) = (n * (n + 1) * (2 * n + 1)) / 6 :=
sorry

end sum_series_sum_of_squares_l381_381411


namespace complement_union_eq_l381_381970

-- Definitions / Conditions
def U : Set ℝ := Set.univ
def M : Set ℝ := { x | x < 1 }
def N : Set ℝ := { x | -1 < x ∧ x < 2 }

-- Statement of the theorem
theorem complement_union_eq {x : ℝ} :
  {x | x ≥ 2} = (U \ (M ∪ N)) := sorry

end complement_union_eq_l381_381970


namespace constructible_triangle_l381_381682

-- Setting up the conditions as Lean definitions
variables {A E F P O : Type} -- vertices and points as types

axiom vertex_A : A
axiom int_angle_bisector_A_E : angle_bisector A E BC -- Axiom for vertex A and intersection point E
axiom int_angle_bisector_A_F : angle_bisector A F circumcircle -- Axiom for vertex A and intersection point F with circumcircle
axiom point_on_incircle : on_incircle P incircle -- Axiom for point P on the incircle
axiom incenter_O : incenter O triangle -- Axiom for the incenter O of the triangle

-- Main theorem to prove the construction is feasible and such a triangle exists
theorem constructible_triangle :
  ∃ (Δ : Type) (A B C : Δ),
    vertex_A = A ∧ int_angle_bisector_A_E E ∧ int_angle_bisector_A_F F ∧ point_on_incircle P ∧ incenter_O O :=
sorry

end constructible_triangle_l381_381682


namespace parabola_problem_l381_381809

theorem parabola_problem (C : ℝ → ℝ → Prop) (F : ℝ × ℝ) (A B : ℝ × ℝ) (P : ℝ × ℝ)
    (l : ℝ × ℝ → Prop)
    (hC : ∀ x y, C x y ↔ y^2 = 4 * x)
    (hl : ∀ x y, l (x, y) ↔ y = x - P.1)
    (hF : F = (1, 0))
    (hA1 : A.2 > 0)
    (hB1 : B.2 < 0)
    (hA2 : l A)
    (hB2 : l B)
    (hC_A : C A.1 A.2)
    (hC_B : C B.1 B.2)
    (hP : P.2 = 0)
    (h_AF_plus_BF : abs (A.1 - F.1) + abs (B.1 - F.1) = 10)
    (h_AB_length : abs (A.1 - B.1) * sqrt 2 = 12 * sqrt 2) :
    P = (2, 0) ∧ ∃ λ : ℝ, abs (A.1 - B.1) = 12 * sqrt 2 → λ = 2 := 
sorry

end parabola_problem_l381_381809


namespace min_b_n_S_n_l381_381781

noncomputable def a (n : ℕ) : ℝ := ∫ x in 0..(n : ℝ), (2 * x + 1)

noncomputable def S (n : ℕ) : ℝ := ∑ k in Finset.range(n) \ k.succ, 1 / a k

noncomputable def b (n : ℕ) : ℝ := n - 8

theorem min_b_n_S_n (n : ℕ) (hn : 0 < n) : 
  (∀ x, b x * S x ≥ -4) ∧ ((∃ n : ℕ, b n * S n = -4) -> n = 2) := sorry

end min_b_n_S_n_l381_381781


namespace include_both_male_and_female_l381_381488

noncomputable def probability_includes_both_genders (total_students male_students female_students selected_students : ℕ) : ℚ :=
  let total_ways := Nat.choose total_students selected_students
  let all_female_ways := Nat.choose female_students selected_students
  (total_ways - all_female_ways) / total_ways

theorem include_both_male_and_female :
  probability_includes_both_genders 6 2 4 4 = 14 / 15 := 
by
  sorry

end include_both_male_and_female_l381_381488


namespace max_chord_length_l381_381824

theorem max_chord_length :
  let family_of_curves (θ : ℝ) (x y : ℝ) := 2 * (2 * Real.sin θ - Real.cos θ) * x^2 - (8 * Real.sin θ + Real.cos θ + 1) * y = 0
  ∃ θ : ℝ, ∀ x y : ℝ, y = 2 * x → family_of_curves θ x y → (x = 0) ∨ (x = (8 * Real.sin θ - Real.cos θ + 1) / (2 * Real.sin θ - Real.cos θ + 3)) ∧ abs (x - 0) * sqrt 5 = 8 * sqrt 5 :=
begin
  sorry
end

end max_chord_length_l381_381824


namespace factory_output_decrease_l381_381063

variable (original_output : ℝ := 100)
variable (first_increase : ℝ := 0.10)
variable (second_increase : ℝ := 0.30)
variable (new_output_step1 : ℝ := original_output * (1 + first_increase))
variable (new_output_step2 : ℝ := new_output_step1 * (1 + second_increase))
variable (decrease_amount : ℝ := original_output - new_output_step2)
variable (percentage_decrease : ℝ := (decrease_amount / new_output_step2) * 100)

theorem factory_output_decrease 
  (original_output : ℝ) (first_increase : ℝ) (second_increase : ℝ) :
  abs ((original_output * (1 + first_increase) * (1 + second_increase) - original_output) /
       (original_output * (1 + first_increase) * (1 + second_increase)) * 100) ≈ 30.07 :=
by
  sorry

end factory_output_decrease_l381_381063


namespace num_female_workers_l381_381629

def num_male_workers := 20
def num_child_workers := 5
def wage_male_worker := 25
def wage_female_worker := 20
def wage_child_worker := 8
def average_wage := 21

theorem num_female_workers : 
  (∃ F : ℕ, 
      let total_daily_wage := wage_male_worker * num_male_workers + 
                              wage_child_worker * num_child_workers + 
                              wage_female_worker * F,
          total_workers := num_male_workers + num_child_workers + F,
          avg_wage_per_day := total_daily_wage / total_workers
      in avg_wage_per_day = average_wage ∧ F = 15) :=
sorry

end num_female_workers_l381_381629


namespace quadrilateral_pts_coincide_l381_381448

theorem quadrilateral_pts_coincide
  (K L M N K' L' M' N' : ι) (A B P Q R S P' Q' R' S' : ι)
  (h₁ : inscribed_quad K L M N)
  (h₂ : inscribed_quad K' L' M' N')
  (chord_AB : line A B)
  (h₃ : intersects KL chord_AB P)
  (h₄ : intersects LM chord_AB Q)
  (h₅ : intersects MN chord_AB R)
  (h₆ : intersects NK chord_AB S)
  (h₇ : intersects K'L' chord_AB P')
  (h₈ : intersects L'M' chord_AB Q')
  (h₉ : intersects M'N' chord_AB R')
  (h₁₀ : intersects N'K' chord_AB S')
  (h₁₁ : P = P')
  (h₁₂ : Q = Q')
  (h₁₃ : R = R') : S = S' :=
sorry

end quadrilateral_pts_coincide_l381_381448


namespace telescoping_log_sum_l381_381243

theorem telescoping_log_sum :
  ∑ k in Finset.range 98 \ Finset.range 2, (Real.log (1 + 1 / k) / Real.log 3) * (Real.log 3 / Real.log k) * (Real.log 3 / Real.log (k + 1)) = 1 - 1 / Real.log 101 :=
by
  sorry

end telescoping_log_sum_l381_381243


namespace number_of_boys_l381_381374

theorem number_of_boys (T : ℕ) (h1 : 0.60 * T = 450) : 0.40 * T = 300 :=
by
  have h2 : T = 450 / 0.60 := by sorry
  have h3 : 0.40 * 750 = 300 := by sorry
  exact h3

end number_of_boys_l381_381374


namespace smallest_number_divisibility_l381_381108

theorem smallest_number_divisibility :
  ∃ n : ℕ, (n - 7) % 12 = 0 ∧
           (n - 7) % 16 = 0 ∧
           (n - 7) % 18 = 0 ∧
           (n - 7) % 21 = 0 ∧
           (n - 7) % 28 = 0 ∧
           n = 1015 :=
by {
  let n := 1015,
  use n,
  split, { norm_num }, split, { norm_num }, split, { norm_num }, split, { norm_num }, split, { norm_num }, 
  exact rfl,
  sorry,
}

end smallest_number_divisibility_l381_381108


namespace manager_salary_4200_l381_381121

theorem manager_salary_4200
    (avg_salary_employees : ℕ → ℕ → ℕ) 
    (total_salary_employees : ℕ → ℕ → ℕ)
    (new_avg_salary : ℕ → ℕ → ℕ)
    (total_salary_with_manager : ℕ → ℕ → ℕ) 
    (n_employees : ℕ)
    (employee_salary : ℕ) 
    (n_total : ℕ)
    (total_salary_before : ℕ)
    (avg_increase : ℕ)
    (new_employee_salary : ℕ) 
    (total_salary_after : ℕ) 
    (manager_salary : ℕ) :
    n_employees = 15 →
    employee_salary = 1800 →
    avg_increase = 150 →
    avg_salary_employees n_employees employee_salary = 1800 →
    total_salary_employees n_employees employee_salary = 27000 →
    new_avg_salary employee_salary avg_increase = 1950 →
    new_employee_salary = 1950 →
    total_salary_with_manager (n_employees + 1) new_employee_salary = 31200 →
    total_salary_before = 27000 →
    total_salary_after = 31200 →
    manager_salary = total_salary_after - total_salary_before →
    manager_salary = 4200 := 
by 
  intros 
  sorry

end manager_salary_4200_l381_381121


namespace smallest_number_property_l381_381110

theorem smallest_number_property : 
  ∃ n, ((n - 7) % 12 = 0) ∧ ((n - 7) % 16 = 0) ∧ ((n - 7) % 18 = 0) ∧ ((n - 7) % 21 = 0) ∧ ((n - 7) % 28 = 0) ∧ n = 1015 :=
by
  sorry  -- Proof is omitted

end smallest_number_property_l381_381110


namespace tan_neg_3pi_over_4_eq_one_l381_381277

theorem tan_neg_3pi_over_4_eq_one : Real.tan (-3 * Real.pi / 4) = 1 := 
by 
  sorry

end tan_neg_3pi_over_4_eq_one_l381_381277


namespace problem_part1_problem_part2_problem_part3_l381_381346

theorem problem_part1 (f : ℝ → ℝ)
  (h1 : ∀ x y, f (x + y) - f y = x * (x + 2 * y + 1))
  (h2 : f 1 = 0) : f 0 = -2 :=
sorry

theorem problem_part2 (f : ℝ → ℝ)
  (h1 : ∀ x y, f (x + y) - f y = x * (x + 2 * y + 1))
  (h2 : f 1 = 0) : ∀ x, f x = x ^ 2 + x - 2 :=
sorry

theorem problem_part3 (a : ℝ → ℝ → Prop)
  (f : ℝ → ℝ)
  (g : ℝ → ℝ)
  (h1 : ∀ x y, f (x + y) - f y = x * (x + 2 * y + 1))
  (h2 : f 1 = 0)
  (h3 : ∀ x, f x = x ^ 2 + x - 2)
  (P : ∀ x, (0 < x ∧ x < 1 / 2) → f x + 3 < 2 * x + a)
  (Q : ∀ x, (x ∈ set.Icc (-2) 2) → monotone (λ x, g x - a * x)) :
  ∃ A B, (A = {a | ∀ x, (0 < x ∧ x < 1 / 2 → f x + 3 < 2 * x + a)} ∧
          B = {a | ∀ x, (x ∈ set.Icc (-2) 2) → monotone (λ x, f x - a * x)} ∧
          (A ∩ (set.univ \ B) = {a | ∀ x, (1 ≤ a ∧ a < 5)})) :=
sorry

end problem_part1_problem_part2_problem_part3_l381_381346


namespace original_stone_count_145_l381_381046

theorem original_stone_count_145 : 
  ∃ (n : ℕ), (n ≡ 1 [MOD 18]) ∧ (n = 145) :=
by
  sorry

end original_stone_count_145_l381_381046


namespace c_share_l381_381616

theorem c_share (A B C : ℕ) (h1 : A = B / 2) (h2 : B = C / 2) (h3 : A + B + C = 392) : C = 224 :=
by
  sorry

end c_share_l381_381616


namespace dress_assignment_l381_381750

variables {Girl : Type} [Finite Girl]
variables (Katya Olya Liza Rita Pink Green Yellow Blue : Girl)
variables (standing_between : Girl → Girl → Girl → Prop)

-- Conditions
variable (cond1 : Katya ≠ Pink ∧ Katya ≠ Blue)
variable (cond2 : standing_between Green Liza Yellow)
variable (cond3 : Rita ≠ Green ∧ Rita ≠ Blue)
variable (cond4 : standing_between Olya Rita Pink)

-- Theorem statement
theorem dress_assignment :
  Katya = Green ∧ Olya = Blue ∧ Liza = Pink ∧ Rita = Yellow := 
sorry

end dress_assignment_l381_381750


namespace greatest_sum_of_vertex_products_is_405_l381_381173

noncomputable def greatest_sum_of_vertex_products : ℕ :=
  let face_labels := [0, 1, 2, 3, 8, 9]
  let vertices := 
    [[0, 1, 2], [0, 1, 3], [0, 2, 3], [0, 2, 8], 
     [1, 2, 8], [1, 3, 8], [1, 3, 9], [2, 3, 9]]
  let products := vertices.map (λ v, v.product (λ i, face_labels.nth i).get_or_else 1) 
  products.sum

theorem greatest_sum_of_vertex_products_is_405 :
  greatest_sum_of_vertex_products = 405 :=
begin
  sorry,
end

end greatest_sum_of_vertex_products_is_405_l381_381173


namespace f_even_l381_381471

def f (x : ℝ) : ℝ := x^2 - 5 * x * sin x

theorem f_even : ∀ x : ℝ, f (-x) = f x :=
by
  intros x
  unfold f
  simp [pow_two, sin_neg]
  sorry

end f_even_l381_381471


namespace symmetric_circle_eq_l381_381505

/-- Define the equation of the circle C -/
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 = 4

/-- Define the equation of the line l -/
def line_equation (x y : ℝ) : Prop := x + y - 1 = 0

/-- 
The symmetric circle to C with respect to line l 
has the equation (x - 1)^2 + (y - 1)^2 = 4.
-/
theorem symmetric_circle_eq (x y : ℝ) :
  (∃ x y : ℝ, circle_equation x y) → 
  (∃ x y : ℝ, line_equation x y) →
  (∃ x y : ℝ, (x - 1)^2 + (y - 1)^2 = 4) :=
by
  sorry

end symmetric_circle_eq_l381_381505


namespace greatest_integer_y_l381_381575

theorem greatest_integer_y (y : ℤ) : (8 : ℚ) / 11 > y / 17 ↔ y ≤ 12 := 
sorry

end greatest_integer_y_l381_381575


namespace even_odd_sum_difference_l381_381106

def sum_first_n_even (n : ℕ) : ℕ :=
  (n * (n - 1))

def sum_first_n_odd (n : ℕ) : ℕ :=
  n * n

theorem even_odd_sum_difference :
  sum_first_n_even 2500 - sum_first_n_odd 2500 = -2500 :=
sorry

end even_odd_sum_difference_l381_381106


namespace complement_union_eq_ge2_l381_381952

open Set

variables {U : Type} [PartialOrder U] [LinearOrder U]

def U : Set ℝ := univ
def M : Set ℝ := { x | x < 1 }
def N : Set ℝ := { x | -1 < x ∧ x < 2 }
def Complement_U (A : Set ℝ) : Set ℝ := U \ A

theorem complement_union_eq_ge2 : 
  Complement_U (M ∪ N) = { x : ℝ | x ≥ 2 } :=
by {
  sorry
}

end complement_union_eq_ge2_l381_381952


namespace arithmetic_sequence_sum_l381_381667

-- Conditions
def first_term : ℕ := 103
def common_difference : ℕ := 20
def last_term : ℕ := 483

-- Statement of the proof problem
theorem arithmetic_sequence_sum : 
  ∑ k in (Finset.range 20).map (λ n, n * common_difference + first_term), id = 5860 := 
by 
  sorry

end arithmetic_sequence_sum_l381_381667


namespace complement_union_M_N_eq_ge_2_l381_381961

def U := set.univ ℝ
def M := {x : ℝ | x < 1}
def N := {x : ℝ | -1 < x ∧ x < 2}

theorem complement_union_M_N_eq_ge_2 :
  (U \ (M ∪ N)) = {x : ℝ | 2 ≤ x} :=
by sorry

end complement_union_M_N_eq_ge_2_l381_381961


namespace parabola_increasing_condition_l381_381059

theorem parabola_increasing_condition (t : ℝ) :
  (∀ x : ℝ, (1 ≤ x → deriv (λ x, x^2 - 2 * t * x + 3) x > 0)) → t ≤ 1 :=
by
  sorry

end parabola_increasing_condition_l381_381059


namespace probability_both_truth_l381_381606

noncomputable def probability_A_truth : ℝ := 0.75
noncomputable def probability_B_truth : ℝ := 0.60

theorem probability_both_truth : 
  (probability_A_truth * probability_B_truth) = 0.45 :=
by sorry

end probability_both_truth_l381_381606


namespace G_20_l381_381508

def G (n : ℕ) : ℕ :=
  if n = 1 then 1
  else if n = 2 then 6
  else if n = 3 then 20
  else G (n - 1) + 5 * (3 * (n - 2) + 1)

theorem G_20 :
  G 20 = 696 :=
by
  have base1: G 1 = 1 := by rfl
  have base2: G 2 = 6 := by rfl
  have base3: G 3 = 20 := by rfl
  sorry

end G_20_l381_381508


namespace train_length_l381_381604

noncomputable def speed_km_hr : ℝ := 60
noncomputable def time_sec : ℝ := 3
noncomputable def speed_m_s := speed_km_hr * 1000 / 3600
noncomputable def length_of_train := speed_m_s * time_sec

theorem train_length :
  length_of_train = 50.01 := by
  sorry

end train_length_l381_381604


namespace sum_logarithms_l381_381232

theorem sum_logarithms :
  (∑ k in finset.Ico 3 101, real.logb 3 (1 + 1/(k:ℝ)) * real.logb k 3 * real.logb (k+1) 3) =
  1 - real.logb 2 3 / real.logb 2 101 :=
by
  sorry

end sum_logarithms_l381_381232


namespace students_present_l381_381537

theorem students_present (total_students : ℕ) (percentage_absent : ℝ) 
  (h1 : total_students = 50) 
  (h2 : percentage_absent = 0.1) : 
  (total_students * (1 - percentage_absent)).toInt = 45 :=
by
  sorry

end students_present_l381_381537


namespace total_price_correct_l381_381601

def price_refrigerator := 4275
def price_difference := 1490
def price_washing_machine := price_refrigerator - price_difference
def sales_tax_rate := 0.07
def total_price_before_tax := price_refrigerator + price_washing_machine
def sales_tax := total_price_before_tax * sales_tax_rate
def total_price_including_tax := total_price_before_tax + sales_tax

theorem total_price_correct :
    total_price_including_tax = 7554.20 := by
  sorry

end total_price_correct_l381_381601


namespace regular_ticket_price_l381_381558

variable (P : ℝ) -- Define the regular ticket price as a real number

-- Condition: Travis pays $1400 for his ticket after a 30% discount on a regular price P
axiom h : 0.70 * P = 1400

-- Theorem statement: Proving that the regular ticket price P equals $2000
theorem regular_ticket_price : P = 2000 :=
by 
  sorry

end regular_ticket_price_l381_381558


namespace drawn_number_sixth_group_l381_381899

theorem drawn_number_sixth_group (m : ℕ) (k : ℕ) (grp_size : ℕ) (units_digit : ℕ → ℕ) :
  m = 7 → k = 6 → grp_size = 10 →
  (∀ m k : ℕ, units_digit (m + k) = (m + k) % 10) →
  (∀ n : ℕ, ∀ grp_num : ℕ, 0 ≤ n ∧ n < 100 → 1 ≤ grp_num ∧ grp_num ≤ 10 →
   n ∈ range ((grp_num - 1) * grp_size, grp_num * grp_size)) →
  ∃ drawn_num : ℕ, drawn_num ∈ range (50, 60) ∧ units_digit (m + k) = drawn_num % 10 ∧ drawn_num = 53 :=
by
  intros h_m h_k h_grp_size h_units_digit h_range
  sorry

end drawn_number_sixth_group_l381_381899


namespace PedrinhoMeetsJoãozinhoOn_l381_381922

-- Define the days of the week
inductive Day : Type
| Sunday
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday
| Saturday
deriving DecidableEq, Repr

open Day

-- Hypotheses based on the problem conditions
def JoãozinhoLiesOn (d : Day) : Prop :=
  d = Tuesday ∨ d = Thursday ∨ d = Saturday

def JoãozinhoTellsTruthOn (d : Day) : Prop :=
  ¬ JoãozinhoLiesOn d

-- Joãozinho's responses
def JoãozinhoSaysTodayIs : Day := Saturday
def JoãozinhoSaysTomorrowIs : Day := Wednesday

-- The theorem we need to prove
theorem PedrinhoMeetsJoãozinhoOn :
  ∃ d : Day, JoãozinhoLiesOn d ∧ JoãozinhoSaysTodayIs = Saturday ∧ JoãozinhoSaysTomorrowIs = Wednesday ∧ d = Thursday :=
  sorry

end PedrinhoMeetsJoãozinhoOn_l381_381922


namespace correct_average_weight_l381_381050

theorem correct_average_weight (n : ℕ) (incorrect_avg_weight : ℝ) (initial_avg_weight : ℝ)
  (misread_weight correct_weight : ℝ) (boys_count : ℕ) :
  incorrect_avg_weight = 58.4 →
  n = 20 →
  misread_weight = 56 →
  correct_weight = 65 →
  boys_count = n →
  initial_avg_weight = (incorrect_avg_weight * n + (correct_weight - misread_weight)) / boys_count →
  initial_avg_weight = 58.85 :=
by
  intro h1 h2 h3 h4 h5 h_avg
  sorry

end correct_average_weight_l381_381050


namespace dress_assignment_l381_381753

variables {Girl : Type} [Finite Girl]
variables (Katya Olya Liza Rita Pink Green Yellow Blue : Girl)
variables (standing_between : Girl → Girl → Girl → Prop)

-- Conditions
variable (cond1 : Katya ≠ Pink ∧ Katya ≠ Blue)
variable (cond2 : standing_between Green Liza Yellow)
variable (cond3 : Rita ≠ Green ∧ Rita ≠ Blue)
variable (cond4 : standing_between Olya Rita Pink)

-- Theorem statement
theorem dress_assignment :
  Katya = Green ∧ Olya = Blue ∧ Liza = Pink ∧ Rita = Yellow := 
sorry

end dress_assignment_l381_381753


namespace number_of_distinct_values_l381_381999

section

variable (x : ℕ)

-- Define the minimum and maximum operations
def min_op (a b : ℕ) : ℕ := min a b
def max_op (a b : ℕ) : ℕ := max a b

-- Define the final expression
def expression (x : ℕ) : ℕ :=
  min_op 6 (max_op 4 (min_op x 5))

-- Theorem statement: the number of distinct values of the expression
theorem number_of_distinct_values : {y : ℕ | ∃ x : ℕ, expression x = y}.to_finset.card = 2 :=
sorry

end

end number_of_distinct_values_l381_381999


namespace max_value_l381_381442

open Real

theorem max_value (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 2 * x + 5 * y < 75) : 
  xy * (75 - 2 * x - 5 * y) ≤ 1562.5 := 
sorry

end max_value_l381_381442


namespace limit_integral_f_exp_l381_381928

def f (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x < 1 then x
  else if 1 ≤ x ∧ x < 2 then 2 - x
  else 0 -- Placeholder for the periodic function

axiom f_periodicity (x : ℝ) (n : ℕ) (hn : n ≥ 1) : f (x + 2 * n) = f x

noncomputable def integral_f_exp (a b : ℝ) : ℝ :=
  ∫ x in a..b, (f x) * exp (-x)

theorem limit_integral_f_exp : 
  tendsto (λ n : ℕ, integral_f_exp 0 (2 * n)) at_top (𝓝 1) := 
by 
  sorry

end limit_integral_f_exp_l381_381928


namespace problem1_problem2_l381_381780

variable (a b : ℝ)

theorem problem1 (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 4) :
  1/a + 1/(b+1) ≥ 4/5 := by
  sorry

theorem problem2 (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 4) :
  4/(a*b) + a/b ≥ (Real.sqrt 5 + 1) / 2 := by
  sorry

end problem1_problem2_l381_381780


namespace length_of_XY_l381_381931

-- Defining the points on the circle
variables (A B C D P Q X Y : Type*)
-- Lengths given in the problem
variables (AB_len CD_len AP_len CQ_len PQ_len : ℕ)
-- Points and lengths conditions
variables (h1 : AB_len = 11) (h2 : CD_len = 19)
variables (h3 : AP_len = 6) (h4 : CQ_len = 7)
variables (h5 : PQ_len = 27)

-- Assuming the Power of a Point theorem applied to P and Q
variables (PX_len PY_len QX_len QY_len : ℕ)
variables (h6 : PX_len = 1) (h7 : QY_len = 3)
variables (h8 : PX_len + PQ_len + QY_len = XY_len)

-- The final length of XY is to be found
def XY_len : ℕ := PX_len + PQ_len + QY_len

-- The goal is to show XY = 31
theorem length_of_XY : XY_len = 31 :=
  by
    sorry

end length_of_XY_l381_381931


namespace odd_sol_exists_l381_381469

theorem odd_sol_exists (n : ℕ) (hn : n > 0) : 
  ∃ (x_n y_n : ℕ), (x_n % 2 = 1) ∧ (y_n % 2 = 1) ∧ (x_n^2 + 7 * y_n^2 = 2^n) := 
sorry

end odd_sol_exists_l381_381469


namespace number_of_proper_subsets_of_P_l381_381834

open Set

-- Conditions
def M : Set ℕ := {0, 1, 2, 3, 4}
def N : Set ℕ := {1, 3, 5}
def P : Set ℕ := M ∩ N

-- Theorem to prove
theorem number_of_proper_subsets_of_P : card {s | s ⊂ P} = 3 :=
by
  sorry

end number_of_proper_subsets_of_P_l381_381834


namespace regression_lines_intersect_at_average_l381_381555

theorem regression_lines_intersect_at_average
  {x_vals1 x_vals2 : List ℝ} {y_vals1 y_vals2 : List ℝ}
  (n1 : x_vals1.length = 100) (n2 : x_vals2.length = 150)
  (mean_x1 : (List.sum x_vals1 / 100) = s) (mean_x2 : (List.sum x_vals2 / 150) = s)
  (mean_y1 : (List.sum y_vals1 / 100) = t) (mean_y2 : (List.sum y_vals2 / 150) = t)
  (regression_line1 : ℝ → ℝ)
  (regression_line2 : ℝ → ℝ)
  (on_line1 : ∀ x, regression_line1 x = (a1 * x + b1))
  (on_line2 : ∀ x, regression_line2 x = (a2 * x + b2))
  (sample_center1 : regression_line1 s = t)
  (sample_center2 : regression_line2 s = t) :
  regression_line1 s = regression_line2 s := sorry

end regression_lines_intersect_at_average_l381_381555


namespace projection_example_l381_381717

def proj (a b : ℝ × ℝ) : ℝ × ℝ :=
  let dot_product := a.fst * b.fst + a.snd * b.snd
  let magnitude_squared := b.fst * b.fst + b.snd * b.snd
  (dot_product / magnitude_squared * b.fst, dot_product / magnitude_squared * b.snd)

theorem projection_example : proj (3, 4) (1, 2) = (11 / 5, 22 / 5) :=
  sorry

end projection_example_l381_381717


namespace number_of_people_in_group_l381_381530

noncomputable def totalBill : ℝ := 211.0
noncomputable def tipPercent : ℝ := 0.15
noncomputable def totalBillWithTip : ℝ := totalBill * (1 + tipPercent)
noncomputable def perPersonShare : ℝ := 24.265

theorem number_of_people_in_group : totalBillWithTip / perPersonShare ≈ 10 := 
by 
  sorry


end number_of_people_in_group_l381_381530


namespace distance_between_vertices_l381_381511

theorem distance_between_vertices :
  let y1 := 3
  let y2 := -1
  abs (y1 - y2) = 4 :=
by
  let y1 := 3
  let y2 := -1
  show abs (y1 - y2) = 4
  from sorry

end distance_between_vertices_l381_381511


namespace domain_of_g_l381_381883

-- Definitions from the problem's conditions
def f : ℝ → ℝ := sorry  -- Assuming generic function f with real inputs and outputs

axiom domain_f_x_add_1 : ∀ x, f(x+1) ≠ none ↔ x ∈ Ioo (-2 : ℝ) 2

-- The statement to prove
theorem domain_of_g : ∀ x, (g(x) ≠ none ↔ x ∈ Ioo 0 3) :=
by
  -- Define g(x)
  let g (x : ℝ) : ℝ := f x / real.sqrt x
  sorry

end domain_of_g_l381_381883


namespace ratio_distance_eq_eccentricity_l381_381605

variable (a b c d e : ℝ)
variable (X : ℝ × ℝ)
variable [NonZeroEccentricity : e < 1]

definition distance_to_focus (X : ℝ × ℝ) (c : ℝ) : ℝ :=
  (X.1 - c)^2 + X.2^2

definition distance_to_directrix (X : ℝ × ℝ) (d : ℝ) : ℝ :=
  (X.1 - d)^2

theorem ratio_distance_eq_eccentricity :
  d = a / e →
  distance_to_focus X c / distance_to_directrix X d = e :=
sorry

end ratio_distance_eq_eccentricity_l381_381605


namespace least_groods_inequality_l381_381408

noncomputable def score_dropping (n : ℕ) : ℚ :=
  (n * (n + 1)) / 4

noncomputable def score_eating (n : ℕ) : ℚ :=
  n ^ 2

theorem least_groods_inequality :
  ∃ (n : ℕ), score_dropping n > score_eating n ∧
             ∀ (k : ℕ), k < n → ¬ (score_dropping k > score_eating k) :=
begin
  use 10,
  split,
  {
    -- Prove that score_dropping 10 > score_eating 10
    sorry
  },
  {
    -- Prove that for all k < 10, score_dropping k <= score_eating k
    intro k,
    intro h,
    -- We need to show that ¬ (score_dropping k > score_eating k)
    sorry
  }
end

end least_groods_inequality_l381_381408


namespace cos_C_area_proof_l381_381914

def triangleABC_conditions (A B C : ℝ) (a b c : ℝ) :=
  a = 3 ∧ c = 2 ∧ sin A = cos (π / 2 - B) ∧ (∀ a b c : ℝ, (c = 2) ∧ (a = b) → a = b)

theorem cos_C_area_proof (A B C : ℝ) (a b c : ℝ) (h : triangleABC_conditions A B C a b c) : 
  cos C = 7 / 9 ∧ (1 / 2 * a * b * sqrt(1 - (7 / 9)^2) = 2 * sqrt 2) :=
by
  sorry

end cos_C_area_proof_l381_381914


namespace dress_assignment_l381_381751

variables {Girl : Type} [Finite Girl]
variables (Katya Olya Liza Rita Pink Green Yellow Blue : Girl)
variables (standing_between : Girl → Girl → Girl → Prop)

-- Conditions
variable (cond1 : Katya ≠ Pink ∧ Katya ≠ Blue)
variable (cond2 : standing_between Green Liza Yellow)
variable (cond3 : Rita ≠ Green ∧ Rita ≠ Blue)
variable (cond4 : standing_between Olya Rita Pink)

-- Theorem statement
theorem dress_assignment :
  Katya = Green ∧ Olya = Blue ∧ Liza = Pink ∧ Rita = Yellow := 
sorry

end dress_assignment_l381_381751


namespace log_ceil_floor_sum_l381_381676

theorem log_ceil_floor_sum :
  ∑ k in Finset.range 2000, k * (⌈Real.log2 (k + 1)⌉ - ⌊Real.log2 (k + 1)⌋) = 1998953 :=
by
  sorry

end log_ceil_floor_sum_l381_381676


namespace only_n_equal_1_l381_381727

theorem only_n_equal_1 (n : ℕ) (h : n ≥ 1) : Nat.Prime (9^n - 2^n) ↔ n = 1 := by
  sorry

end only_n_equal_1_l381_381727


namespace complex_power_six_l381_381433

theorem complex_power_six (i : ℂ) (hi : i * i = -1) : (1 + i)^6 = -8 * i :=
by
  sorry

end complex_power_six_l381_381433


namespace ages_of_Mel_and_Lexi_l381_381436

theorem ages_of_Mel_and_Lexi (M L K : ℤ)
  (h1 : M = K - 3)
  (h2 : L = M + 2)
  (h3 : K = 60) :
  M = 57 ∧ L = 59 :=
  by
    -- Proof steps are omitted.
    sorry

end ages_of_Mel_and_Lexi_l381_381436


namespace frac_inequality_l381_381806

theorem frac_inequality (a b c d : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : d > c) (h4 : c > 0) : (a/c) > (b/d) := 
sorry

end frac_inequality_l381_381806


namespace number_of_possible_D_values_l381_381402

-- Define the concept of digits and their distinctness
def digit := {n : ℕ // n < 10}
def distinct {α : Type*} (s : set α) := ∀ ⦃x y⦄, x ∈ s → y ∈ s → x = y → false

-- Define the problem constants and conditions
variables (A B C D : digit)
variables (carry_over : Prop)

-- Conditions
def conditions : Prop :=
  distinct {A.val, B.val, C.val, D.val} ∧
  (∀ n m, (n + m).mod 10 = D.val) ∧ 
  (carry_over = (nc := (B.val + B.val) / 10 > 0))

-- Problem statement
theorem number_of_possible_D_values : ∃ count : ℕ, conditions A B C D carry_over ∧ count = 5 := by
  sorry

end number_of_possible_D_values_l381_381402


namespace avery_egg_cartons_l381_381204

theorem avery_egg_cartons 
  (num_chickens : ℕ) (eggs_per_chicken : ℕ) (carton_capacity : ℕ)
  (h1 : num_chickens = 20) (h2 : eggs_per_chicken = 6) (h3 : carton_capacity = 12) :
  (num_chickens * eggs_per_chicken) / carton_capacity = 10 :=
by sorry

end avery_egg_cartons_l381_381204


namespace sum_of_elements_when_n_is_5_sum_of_smallest_elements_in_subsets_l381_381323

open_locale big_operators

variables {n : ℕ} (hn : n ≥ 4)
def M := finset.range n

/-- Statement for question 1 -/
theorem sum_of_elements_when_n_is_5 :
  finset.sum (finset.powerset_len 3 (finset.range 5)) (λ s, s.sum id) = 90 :=
by sorry

/-- Statement for question 2 -/
theorem sum_of_smallest_elements_in_subsets :
  let Pn := finset.sum (finset.powerset_len 3 M) (λ s, s.min' (finset.card_pos.2 hn))
  in Pn = nat.choose (n + 1) 4 :=
by sorry

end sum_of_elements_when_n_is_5_sum_of_smallest_elements_in_subsets_l381_381323


namespace who_wears_which_dress_l381_381738

def girls := ["Katya", "Olya", "Liza", "Rita"]
def dresses := ["pink", "green", "yellow", "blue"]

variable (who_wears_dress : String → String)

/-- Conditions given in the problem --/
axiom Katya_not_pink_blue : who_wears_dress "Katya" ≠ "pink" ∧ who_wears_dress "Katya" ≠ "blue"
axiom between_green_liza_yellow : ∃ g, who_wears_dress "Katya" = "green" ∧ who_wears_dress "Rita" = "yellow"
axiom Rita_not_green_blue : who_wears_dress "Rita" ≠ "green" ∧ who_wears_dress "Rita" ≠ "blue"
axiom Olya_between_rita_pink : ∃ o, who_wears_dress "Olya" = "blue" ∧ who_wears_dress "Liza" = "pink"

theorem who_wears_which_dress :
  who_wears_dress "Katya" = "green" ∧
  who_wears_dress "Olya" = "blue" ∧
  who_wears_dress "Liza" = "pink" ∧
  who_wears_dress "Rita" = "yellow" :=
by
  sorry

end who_wears_which_dress_l381_381738


namespace game_configurations_count_l381_381896

-- Definitions of the game conditions
def grid_size : ℕ × ℕ := (5, 7)

-- The number of unique paths in the grid game
def count_configurations (m n : ℕ) : ℕ := Nat.choose (m + n) m

-- The main theorem stating the number of different possible situations in the game
theorem game_configurations_count : count_configurations 7 5 = 792 := by
  unfold count_configurations
  simp [Nat.choose]
  norm_num
  sorry

end game_configurations_count_l381_381896


namespace complement_A_in_U_l381_381444

open Set

variable {𝕜 : Type*} [LinearOrderedField 𝕜]

def A (x : 𝕜) : Prop := |x - (1 : 𝕜)| > 2
def U : Set 𝕜 := univ

theorem complement_A_in_U : (U \ {x : 𝕜 | A x}) = {x : 𝕜 | -1 ≤ x ∧ x ≤ 3} := by
  sorry

end complement_A_in_U_l381_381444


namespace isosceles_triangle_vertex_angle_l381_381398

theorem isosceles_triangle_vertex_angle (T : Triangle) (h_iso : T.is_isosceles) (h_angle : T.has_interior_angle 50) :
  T.vertex_angle = 50 ∨ T.vertex_angle = 80 := 
sorry

end isosceles_triangle_vertex_angle_l381_381398


namespace sum_of_reciprocals_lower_bound_l381_381818

theorem sum_of_reciprocals_lower_bound (n : ℕ) (h : 2 ≤ n) (a : Fin n → ℝ) 
  (hpos : ∀ i, 0 < a i) (hsum : (∑ i, a i) = 1) :
  (∑ i, 1 / a i) ≥ n^2 :=
sorry

end sum_of_reciprocals_lower_bound_l381_381818


namespace num_pos_3_digits_div_by_7_l381_381857

theorem num_pos_3_digits_div_by_7 : 
  let lower_bound := 100
  let upper_bound := 999 
  let divisor := 7
  let smallest_3_digit := 105 -- or can be computed explicitly by: (lower_bound + divisor - 1) / divisor * divisor
  let largest_3_digit := 994  -- or can be computed explicitly by: upper_bound / divisor * divisor
  List.length (List.filter (λ n, n % divisor = 0) (List.range' smallest_3_digit (largest_3_digit + 1))) = 128 :=
by
  sorry

end num_pos_3_digits_div_by_7_l381_381857


namespace functional_equation_solution_l381_381280

theorem functional_equation_solution (f : ℝ → ℝ) (x : ℝ) (hx : x ≠ 0 ∧ x ≠ 1) :
  (f x = (1/2) * (x + 1 - 1/x - 1/(1-x))) →
  (f x + f (1 / (1 - x)) = x) :=
sorry

end functional_equation_solution_l381_381280


namespace dress_assignment_l381_381754

variables {Girl : Type} [Finite Girl]
variables (Katya Olya Liza Rita Pink Green Yellow Blue : Girl)
variables (standing_between : Girl → Girl → Girl → Prop)

-- Conditions
variable (cond1 : Katya ≠ Pink ∧ Katya ≠ Blue)
variable (cond2 : standing_between Green Liza Yellow)
variable (cond3 : Rita ≠ Green ∧ Rita ≠ Blue)
variable (cond4 : standing_between Olya Rita Pink)

-- Theorem statement
theorem dress_assignment :
  Katya = Green ∧ Olya = Blue ∧ Liza = Pink ∧ Rita = Yellow := 
sorry

end dress_assignment_l381_381754


namespace people_between_katya_and_polina_l381_381124

-- Definitions based on given conditions
def is_next_to (a b : ℕ) : Prop := (b = a + 1) ∨ (b = a - 1)
def position_alena : ℕ := 1
def position_lena : ℕ := 5
def position_sveta (pos_sveta : ℕ) : Prop := pos_sveta + 1 = position_lena
def position_katya (pos_katya : ℕ) : Prop := pos_katya = 3
def position_polina (pos_polina : ℕ) : Prop := (is_next_to position_alena pos_polina)

-- The question: prove the number of people between Katya and Polina is 0
theorem people_between_katya_and_polina : 
  ∃ (pos_katya pos_polina : ℕ),
    position_katya pos_katya ∧ 
    position_polina pos_polina ∧ 
    pos_polina + 1 = pos_katya ∧
    pos_katya = 3 ∧ pos_polina = 2 := 
sorry

end people_between_katya_and_polina_l381_381124


namespace final_number_is_even_l381_381440

theorem final_number_is_even (a : Fin 1980 → ℕ) (h : ∃ p : Fin 1980 → ℕ, bij_on p univ univ ∧ ∀ i, a i = p i) : 
  ∃ x : ℕ, (∀ i, x = 
             let b i := abs (a (Fin.of_nat (2*i-1))) - a (Fin.of_nat (2*i))) in
             let c i := abs (b (Fin.of_nat (2*i-1))) -  b (Fin.of_nat (2*i)) in
             let d i := abs (c (Fin.of_nat (2*i-1))) -  c (Fin.of_nat (2*i)) in
             -- Continue pattern until last d sequence
             2 ) := sorry

end final_number_is_even_l381_381440


namespace arithmetic_sqrt_of_4_l381_381495

theorem arithmetic_sqrt_of_4 : ∃ x : ℚ, x^2 = 4 ∧ x > 0 → x = 2 :=
by {
  sorry
}

end arithmetic_sqrt_of_4_l381_381495


namespace Frank_final_amount_l381_381626

theorem Frank_final_amount (original_amount : ℝ) (rate : ℝ) (days : ℕ) (late_charges : ℕ) :
  original_amount = 500 ∧ rate = 0.02 ∧ days = 90 ∧ late_charges = 3 →
  let after_30_days := original_amount * (1 + rate)
  let after_60_days := after_30_days * (1 + rate)
  let after_90_days := after_60_days * (1 + rate)
  after_90_days = 530.604 :=
by {
  intros,
  sorry
}

end Frank_final_amount_l381_381626


namespace triangle_exists_l381_381681

-- Define the geometric conditions as structures and theorems.
structure TriangleConstructionConditions where
  alpha : ℝ   -- Angle at vertex A
  sb : ℝ      -- Median from vertex B
  mc : ℝ      -- Altitude from vertex C
  (alpha_bounds : 0 < alpha ∧ alpha < π)  -- Angle A constraints

theorem triangle_exists (cond : TriangleConstructionConditions) : 
  ∃ A B C : ℝ × ℝ,
  -- Assuming here to be Euclidean coordinates for points A, B, C
  let distance_A_B := (A.1 - B.1)^2 + (A.2 - B.2)^2,
      distance_B_C := (B.1 - C.1)^2 + (B.2 - C.2)^2,
      distance_C_A := (C.1 - A.1)^2 + (C.2 - A.2)^2,
      length_median_s_b := sqrt(((A.1 + C.1) / 2 - B.1)^2 + ((A.2 + C.2) / 2 - B.2)^2),
      length_altitude_m_c := C.2 - ((A.1 * (B.1 - C.1) + A.2 * (B.2 - C.2) - (B.1 * C.1 + B.2 * C.2)) / distance_A_B)
  in 
  distance_A_B * length_median_s_b = sb^2 ∧
  distance_B_C * length_altitude_m_c = mc^2 ∧
  length_median_s_b ≤ sqrt((A - C)^2) ∧   -- Ensure existence conditions
  length_altitude_m_c ≤ mc := sorry

end triangle_exists_l381_381681


namespace rectangle_integer_perimeters_l381_381649

theorem rectangle_integer_perimeters (n : ℕ) (cells : Finset ℕ) 
(h_divided : cells.card = 121) 
(h_known : ∃ S : Finset ℕ, S.card = 111 ∧ ∀ c ∈ S, (∃ a b : ℤ, a * b ∈ S)) : 
∃ U : Finset ℕ, U.card = 10 ∧ ∀ c ∈ U, (∃ a b : ℤ, a * b ∈ U) := 
sorry

end rectangle_integer_perimeters_l381_381649


namespace Vasya_wins_l381_381404

noncomputable def chessboard_win_strategy : Prop :=
  ∀ (Petya Vasya : ℕ → ℕ → Prop)
    (start := (1, 1))
    (board_size := (8, 8)),
    (∀ m n i j, Petya m n → Vasya i j → m ≠ i → n ≠ j) → -- Petya moves as a queen
    (∀ i j, Vasya i j → |i - j| ≤ 2) → -- Vasya moves 2 squares as a king
    (start.1 ≤ board_size.1 ∧ start.2 ≤ board_size.2) → -- starting condition
    (Vasya move_ensures_win start) -- asserting Vasya ensures win

theorem Vasya_wins :
  chessboard_win_strategy := 
sorry

end Vasya_wins_l381_381404


namespace butterfat_in_final_mixture_l381_381846

noncomputable def final_butterfat_percentage (gallons_of_35_percentage : ℕ) 
                                             (percentage_of_35_butterfat : ℝ) 
                                             (total_gallons : ℕ)
                                             (percentage_of_10_butterfat : ℝ) : ℝ :=
  let gallons_of_10 := total_gallons - gallons_of_35_percentage
  let butterfat_35 := gallons_of_35_percentage * percentage_of_35_butterfat
  let butterfat_10 := gallons_of_10 * percentage_of_10_butterfat
  let total_butterfat := butterfat_35 + butterfat_10
  (total_butterfat / total_gallons) * 100

theorem butterfat_in_final_mixture : 
  final_butterfat_percentage 8 0.35 12 0.10 = 26.67 :=
sorry

end butterfat_in_final_mixture_l381_381846


namespace find_ellipse_and_slope_l381_381353

/-- Given conditions -/
def parabola_C1 (x y : ℝ) : Prop :=
  x^2 = 4 * y

def ellipse_C2 (x y : ℝ) : Prop :=
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a > b ∧ y^2 / a^2 + x^2 / b^2 = 1

def common_chord_length (length : ℝ) : Prop :=
  length = 2 * Real.sqrt 6

/-- Focus F of parabola C1 and its relationship with ellipse C2 -/
def focus_F (F : ℝ × ℝ) : Prop :=
  F = (0, 1)

def focus_on_ellipse (C : ℝ × ℝ → Prop) (F : ℝ × ℝ) : Prop :=
  ∃ (a b : ℝ), a > b ∧ b > 0 ∧ a^2 - b^2 = 1 ∧ (C F.1 F.2)

/-- A line passes through the focus F and intersects the given curves -/
def intersect_line_points (F A B C D : ℝ × ℝ) : Prop :=
  (A.1, A.2) ≠ (B.1, B.2) ∧ (C.1, C.2) ≠ (D.1, D.2) ∧
  (A.1 - F.1, A.2 - F.2) = (F.1 - B.1, F.2 - B.2) ∧
  (C.1 - F.1, C.2 - F.2) = (F.1 - D.1, F.2 - D.2) ∧
  Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2) = Real.sqrt ((D.1 - B.1)^2 + (D.2 - B.2)^2)

/-- Proving the equation of ellipse C2 and the slope k -/
theorem find_ellipse_and_slope :
  ∃ (a b : ℝ), a = 3 ∧ b = 2 * Real.sqrt 2 ∧
  (∀ (x y : ℝ), ellipse_C2 x y ↔ y^2 / 9 + x^2 / 8 = 1) ∧
  (∀ (k : ℝ), k = Real.sqrt 6 / 4 ∨ k = -Real.sqrt 6 / 4) :=
by
  sorry

end find_ellipse_and_slope_l381_381353


namespace dress_assignments_l381_381744

structure GirlDress : Type :=
  (Katya Olya Liza Rita : String)

def dresses := ["Pink", "Green", "Yellow", "Blue"]

axiom not_pink_or_blue : GirlDress.Katya ≠ "Pink" ∧ GirlDress.Katya ≠ "Blue"
axiom green_between_liza_yellow : (GirlDress.Liza = "Pink" ∨ GirlDress.Rita = "Yellow") ∧
                                  GirlDress.Katya = "Green" ∧
                                  GirlDress.Rita = "Yellow" ∧ GirlDress.Liza = "Pink"
axiom not_green_or_blue : GirlDress.Rita ≠ "Green" ∧ GirlDress.Rita ≠ "Blue"
axiom olya_between_rita_pink : GirlDress.Olya ≠ "Pink" → GirlDress.Rita ≠ "Pink" → GirlDress.Liza = "Pink"

theorem dress_assignments (gd : GirlDress) :
  gd.Katya = "Green" ∧ gd.Olya = "Blue" ∧ gd.Liza = "Pink" ∧ gd.Rita = "Yellow" :=
by
  sorry

end dress_assignments_l381_381744


namespace rectangle_area_increase_l381_381648

theorem rectangle_area_increase (l w : ℝ) :
  let l' := 1.3 * l,
      w' := 1.15 * w,
      original_area := l * w,
      new_area := l' * w',
      percentage_increase := ((new_area - original_area) / original_area) * 100 in
  percentage_increase = 49.5 :=
by
  sorry

end rectangle_area_increase_l381_381648


namespace log_multiplication_identity_l381_381268

theorem log_multiplication_identity :
  log 3 / log 2 * (2 * (log 2 / log 3)) = 2 :=
by sorry

end log_multiplication_identity_l381_381268


namespace geometric_sequence_a4_a7_l381_381908

theorem geometric_sequence_a4_a7 (a : ℕ → ℝ) (h1 : ∃ a₁ a₁₀, a₁ * a₁₀ = -6 ∧ a 1 = a₁ ∧ a 10 = a₁₀) :
  a 4 * a 7 = -6 :=
sorry

end geometric_sequence_a4_a7_l381_381908


namespace quadratic_function_f2_eq_four_l381_381017

variable {a b : ℝ}
variable (h_distinct : a ≠ b)
variable (h_func : ∀ x, f x = x^2 + a*x + b)
variable (h_condition : f a = f b)

theorem quadratic_function_f2_eq_four : f 2 = 4 := sorry

end quadratic_function_f2_eq_four_l381_381017


namespace cos_difference_l381_381778

/-- Given conditions for cos and sin sums and prove the value of cos(α-β). -/
theorem cos_difference (α β : ℝ) (h1 : cos α + cos β = 1 / 2) (h2 : sin α + sin β = 1 / 3) 
  : cos (α - β) = -59 / 72 :=
by
  -- The final proof is omitted here.
  sorry

end cos_difference_l381_381778


namespace length_AB_l381_381320

def Quadrilateral (A B C D X Y : Type) [DecidableEq A] [DecidableEq B] [DecidableEq C] [DecidableEq D] :=
  AD_parallel_BC : AD ∥ BC
  angle_bisector_A : (angle_bisector A).intersects (segment CD) at X,
  segment_extends_BC_Y : (segment BC).extends_beyond C at Y,
  angle_AXC_90 : ∠AXC = 90,
  AD_length : AD = 19,
  CY_length : CY = 16

theorem length_AB (A B C D X Y : Type) [DecidableEq A] [DecidableEq B] [DecidableEq C] [DecidableEq D]
  (quad : Quadrilateral A B C D X Y) : 
  length_AB = 32 :=
  sorry

end length_AB_l381_381320


namespace who_wears_which_dress_l381_381739

def girls := ["Katya", "Olya", "Liza", "Rita"]
def dresses := ["pink", "green", "yellow", "blue"]

variable (who_wears_dress : String → String)

/-- Conditions given in the problem --/
axiom Katya_not_pink_blue : who_wears_dress "Katya" ≠ "pink" ∧ who_wears_dress "Katya" ≠ "blue"
axiom between_green_liza_yellow : ∃ g, who_wears_dress "Katya" = "green" ∧ who_wears_dress "Rita" = "yellow"
axiom Rita_not_green_blue : who_wears_dress "Rita" ≠ "green" ∧ who_wears_dress "Rita" ≠ "blue"
axiom Olya_between_rita_pink : ∃ o, who_wears_dress "Olya" = "blue" ∧ who_wears_dress "Liza" = "pink"

theorem who_wears_which_dress :
  who_wears_dress "Katya" = "green" ∧
  who_wears_dress "Olya" = "blue" ∧
  who_wears_dress "Liza" = "pink" ∧
  who_wears_dress "Rita" = "yellow" :=
by
  sorry

end who_wears_which_dress_l381_381739


namespace cookout_kids_2006_l381_381384

theorem cookout_kids_2006 :
  let kids_2004 := 60
  let kids_2005 := kids_2004 / 2
  let kids_2006 := (kids_2005 / 3) * 2
  in kids_2006 = 20 :=
by
  let kids_2004 := 60
  let kids_2005 := kids_2004 / 2
  let kids_2006 := (kids_2005 / 3) * 2
  have h : kids_2006 = 20 := sorry
  exact h

end cookout_kids_2006_l381_381384


namespace circle_property_l381_381422

-- The problem statement with given conditions
variables (A B C X Y A' O : Type) 
variables (hACgtAB : AC ≥ AB) -- WLOG assumption (not strictly necessary to formalize though)
variables (hABC_acute : acute_triangle A B C)
variables (hX_side_C : different_side X C (line AB))
variables (hY_side_B : different_side Y B (line AC))
variables (hBX_eq_AC : distance B X = distance A C)
variables (hCY_eq_AB : distance C Y = distance A B)
variables (hAX_eq_AY : distance A X = distance A Y)
variables (hA'_reflection : reflection A' A (perpendicular_bisector B C))
variables (hX_Y_different_side_AA' : different_side X Y (line AA'))

-- The proof problem to show that A, A', X, and Y lie on a circle
theorem circle_property :
  ∃ (O : circle_center), lie_on_circle A A' X Y O := sorry

end circle_property_l381_381422


namespace log_cos_sin_l381_381372

theorem log_cos_sin (b x a : ℝ) (hb : b > 1) (hsin : sin x > 0) (hcos : cos x > 0) (hlog : log b (sin x) = a) :
  log b (cos x * sin x) = 1/2 * log b (1 - b^(2 * a)) + a :=
sorry

end log_cos_sin_l381_381372


namespace sin_neg_120_eq_sqrt_3_over_2_l381_381218

noncomputable def sin_neg_angle (θ : ℝ) : ℝ := -Real.sin θ

theorem sin_neg_120_eq_sqrt_3_over_2 :
  sin_neg_angle (120 * Real.pi / 180) = Real.sqrt 3 / 2 :=
by
  -- Use the identity sin(-θ) = -sin(θ)
  have h1 : sin_neg_angle (120 * Real.pi / 180) = -Real.sin (120 * Real.pi / 180) := rfl
  
  -- Simplify and use pre-defined instances for sin(120 degrees)
  have h2 : Real.sin (120 * Real.pi / 180) = Real.sin (2 * Real.pi / 3) := by 
    norm_num

  -- Calculate the sin value for 2π/3
  have h3 : Real.sin (2 * Real.pi / 3) = Real.sin (Real.pi - Real.pi / 3) := by 
    norm_num [Real.sin_pi_sub_div]
  
  -- Which further simplifies to
  have h4 : Real.sin (Real.pi - Real.pi / 3) = Real.sin (π/3) := by
    norm_num
  
  -- Since sin(π/3) = sqrt(3)/2
  have h5 : Real.sin (π/3) = Real.sqrt 3 / 2 := by
    norm_num
  
  -- Applying all above results
  rw [h1, h2, h3, h4, h5]
  norm_num

  -- Concluding the proof
  exact sorry

end sin_neg_120_eq_sqrt_3_over_2_l381_381218


namespace multiple_of_interest_rate_l381_381501

theorem multiple_of_interest_rate (P r m : ℝ) (h1 : P * r^2 = 40) (h2 : P * (m * r)^2 = 360) : m = 3 :=
by
  sorry

end multiple_of_interest_rate_l381_381501


namespace four_color_theorem_l381_381446

theorem four_color_theorem (map : Type) (borders : map → ℕ) 
  (h : ∀ c : map, borders c % 3 = 0) : 
  ∃ (coloring : map → ℕ), ∀ c1 c2 : map, adjacent c1 c2 → coloring c1 ≠ coloring c2 ∧ coloring c1 < 4 ∧ coloring c2 < 4 := 
sorry

end four_color_theorem_l381_381446


namespace problem_statement_l381_381391

noncomputable def seq : ℕ → ℚ
| 1       := 1
| (n + 1) := (↑(n + 1) / n)^3

theorem problem_statement : seq 3 + seq 7 = 134 / 27 :=
by
  have h₃ : seq 3 = (3 / 2)^3 := rfl
  have h₃_exp : seq 3 = 27 / 8 := by norm_num [h₃]
  have h₇ : seq 7 = (7 / 6)^3 := rfl
  have h₇_exp : seq 7 = 343 / 216 := by norm_num [h₇]
  rw [h₃_exp, h₇_exp]
  have sum_exp : 27 / 8 + 343 / 216 = 1072 / 216 := by norm_num
  rw sum_exp
  have simp_exp : 1072 / 216 = 134 / 27 := by norm_num
  exact simp_exp

end problem_statement_l381_381391


namespace grocer_bought_100_pounds_of_coffee_l381_381641

theorem grocer_bought_100_pounds_of_coffee
  (initial_stock : ℕ)
  (initial_decaf_percent : ℝ)
  (second_batch_decaf_percent : ℝ)
  (final_decaf_percent : ℝ)
  (w_initial := 400)
  (p_initial := 0.20)
  (p_second := 0.50)
  (p_final := 0.26)
  (decaf_initial := p_initial * w_initial)
  (decaf_final (x : ℝ) := 0.5 * x)
  (total_weight (x : ℝ) := w_initial + x)
  (total_decaf (x : ℝ) := decaf_initial + decaf_final x) :
  ∃ x : ℝ, (total_decaf x / total_weight x = p_final) ∧ (x = 100) :=
by
  sorry

end grocer_bought_100_pounds_of_coffee_l381_381641


namespace prime_product_correct_l381_381306

theorem prime_product_correct 
    (p1 : Nat := 1021031) (pr1 : Prime p1)
    (p2 : Nat := 237019) (pr2 : Prime p2) :
    p1 * p2 = 241940557349 :=
by
  sorry

end prime_product_correct_l381_381306


namespace oblique_projection_correct_statement_l381_381413

theorem oblique_projection_correct_statement :
  (∀ (a b : ℝ), a = b → (a ≠ b ∨ a = b)) →
  (∀ (p q: ℝ), p = q → (∃ (r s: ℝ), r ≠ s ∧ r = s)) →
  (∀ (x y: ℝ), parallel x y → (∃ (x' y': ℝ), parallel x' y' ∧ x' = y')) :=
by
  sorry

end oblique_projection_correct_statement_l381_381413


namespace sqrt_fraction_sum_ineq_l381_381011

theorem sqrt_fraction_sum_ineq (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
    (a / Real.sqrt (a^2 + 8 * b * c) + b / Real.sqrt (b^2 + 8 * c * a) + c / Real.sqrt (c^2 + 8 * a * b) >= 1) :=
begin
  sorry
end

end sqrt_fraction_sum_ineq_l381_381011


namespace jason_total_spent_l381_381416

theorem jason_total_spent (h_shorts : ℝ) (h_jacket : ℝ) (h1 : h_shorts = 14.28) (h2 : h_jacket = 4.74) : h_shorts + h_jacket = 19.02 :=
by
  rw [h1, h2]
  norm_num

end jason_total_spent_l381_381416


namespace jacket_price_on_sunday_l381_381491

theorem jacket_price_on_sunday :
  let original_price := 200
  let first_discount := original_price * (1 / 2)
  let second_discount := first_discount * (3 / 4)
  let tax := 0.1 * second_discount
  second_discount + tax = 82.5 :=
by
  let original_price := 200
  let first_discount := original_price * (1 / 2)
  let second_discount := first_discount * (3 / 4)
  let tax := 0.1 * second_discount
  show second_discount + tax = 82.5
  sorry

end jacket_price_on_sunday_l381_381491


namespace q1_q2_q3_l381_381722

noncomputable def is_approaching_function (f g : ℝ → ℝ) : Prop :=
  monotone (λ x, f x - g x ∧ (∃ p, ∀ x, 0 < f x - g x ∧ f x - g x ≤ p))

noncomputable def part_a : Prop :=
  is_approaching_function (λ x, (2*x^2 + 9*x + 11) / (x + 2)) (λ x, 2*x + 5)

noncomputable def part_b : Prop :=
  ¬ is_approaching_function (λ x, (1/2)^x) (λ x, 1/2 * x)

noncomputable def part_c : Prop :=
  ∀ a, (is_approaching_function (λ x, x + (x^2 + 1)^(1/2)) (λ x, a*x)) → (a = 2)

theorem q1 : part_a := sorry

theorem q2 : part_b := sorry

theorem q3 : part_c := sorry

end q1_q2_q3_l381_381722


namespace alice_needs_to_contribute_l381_381172

def books_cost : ℝ := 15
def ben_cad : ℝ := 20
def exchange_rate : ℝ := 1.50

def ben_contribution : ℝ := ben_cad / exchange_rate
def alice_contribution : ℝ := books_cost - ben_contribution

theorem alice_needs_to_contribute :
  alice_contribution = 1.67 := by
  sorry

end alice_needs_to_contribute_l381_381172


namespace find_value_of_a_l381_381044

variable (a b : ℝ)

def varies_inversely (a : ℝ) (b_minus_one_sq : ℝ) : ℝ :=
  a * b_minus_one_sq

theorem find_value_of_a 
  (h₁ : ∀ b : ℝ, varies_inversely a ((b - 1) ^ 2) = 64)
  (h₂ : b = 5) : a = 4 :=
by sorry

end find_value_of_a_l381_381044


namespace compute_logarithmic_sum_l381_381228

theorem compute_logarithmic_sum : 
  ∑ k in finset.Icc 3 100, (log 3 (1 + (1 : ℝ) / k) * log k 3 * log (k + 1) 3) = -1 := 
sorry

end compute_logarithmic_sum_l381_381228


namespace part_I_distribution_and_expectation_part_II_conditional_probability_l381_381490

/-- Define the context for the problem, including players and their match winning probabilities.
    Define the score distribution for player A and its expectation. -/
open ProbabilityTheory

-- Define probabilities of A, B, C winning their respective matches.
def P_A_wins_B : ℝ := 2 / 3
def P_A_wins_C : ℝ := 2 / 3
def P_A_wins_D : ℝ := 2 / 3
def P_B_wins_C : ℝ := 3 / 5
def P_B_wins_D : ℝ := 3 / 5
def P_C_wins_D : ℝ := 1 / 2

-- Assume independence of individual match results.
axiom independence {A B : Prop} : Prob (A ∧ B) = Prob A * Prob B

-- Define the distribution table for A's score X.
def distribution_table : Π (X : ℕ), ℝ
| 0 := (1 / 3) ^ 3
| 1 := 3 * (2 / 3) * (1 / 3) ^ 2
| 2 := 3 * (2 / 3) ^ 2 * (1 / 3)
| 3 := (2 / 3) ^ 3
| _ := 0

-- Define the expectation of X for player A.
def expectation_X : ℝ := 0 * distribution_table 0 + 1 * distribution_table 1 + 2 * distribution_table 2 + 3 * distribution_table 3

-- Main theorem statement for Part (I): distribution and expectation of A's score.
theorem part_I_distribution_and_expectation :
  distribution_table 0 = 1 / 27 ∧
  distribution_table 1 = 2 / 9 ∧
  distribution_table 2 = 4 / 9 ∧
  distribution_table 3 = 8 / 27 ∧
  expectation_X = 2 := by sorry

-- Define the probability of A winning the championship.
-- Define the conditional probability that B wins given A wins.
def P_A_wins_championship : ℝ := (2 / 3) ^ 3 + (1 / 3) * (2 / 3) ^ 2 * (1 - (3 / 5) ^ 2) + 2 * (1 / 3) * (2 / 3) ^ 2 * (1 - (2 / 5) * (1 / 2))

def P_A_and_B_wins_championship : ℝ := (1 / 3) * (2 / 3) ^ 2 * (2 / 5) * (3 / 5) * 2 + 2 * (1 / 3) * (2 / 3) ^ 2 * (3 / 5) ^ 2

def P_B_given_A_wins_championship : ℝ := P_A_and_B_wins_championship / P_A_wins_championship

-- Main theorem statement for Part (Ⅱ): conditional probability B wins given A wins.
theorem part_II_conditional_probability :
  P_B_given_A_wins_championship = 15 / 53 := by sorry

end part_I_distribution_and_expectation_part_II_conditional_probability_l381_381490


namespace distance_between_vertices_l381_381513

theorem distance_between_vertices :
  let equation : ℝ → ℝ → Prop := λ x y, √(x^2 + y^2) + |y - 2| = 4
  let vertex1 := (0, 3)
  let vertex2 := (0, -1)
  (abs ((vertex1.2) - (vertex2.2)) = 4) :=
by
  sorry

end distance_between_vertices_l381_381513


namespace problem_equivalent_l381_381944

def U : Set ℝ := Set.univ
def M : Set ℝ := {x | x < 1}
def N : Set ℝ := {x | -1 < x ∧ x < 2}

theorem problem_equivalent :
  {x : ℝ | x ≥ 2} = (U \ (M ∪ N)) := 
by sorry

end problem_equivalent_l381_381944


namespace allison_total_supply_items_is_28_l381_381181

/-- Define the number of glue sticks Marie bought --/
def marie_glue_sticks : ℕ := 15
/-- Define the number of packs of construction paper Marie bought --/
def marie_construction_paper_packs : ℕ := 30
/-- Define the number of glue sticks Allison bought --/
def allison_glue_sticks : ℕ := marie_glue_sticks + 8
/-- Define the number of packs of construction paper Allison bought --/
def allison_construction_paper_packs : ℕ := marie_construction_paper_packs / 6
/-- Calculation of the total number of craft supply items Allison bought --/
def allison_total_supply_items : ℕ := allison_glue_sticks + allison_construction_paper_packs

/-- Prove that the total number of craft supply items Allison bought is equal to 28. --/
theorem allison_total_supply_items_is_28 : allison_total_supply_items = 28 :=
by sorry

end allison_total_supply_items_is_28_l381_381181


namespace complement_union_eq_ge_two_l381_381990

def U : Set ℝ := Set.univ
def M : Set ℝ := { x : ℝ | x < 1 }
def N : Set ℝ := { x : ℝ | -1 < x ∧ x < 2 }

theorem complement_union_eq_ge_two : { x : ℝ | x ≥ 2 } = U \ (M ∪ N) :=
by
  sorry

end complement_union_eq_ge_two_l381_381990


namespace quadratic_roots_l381_381068

theorem quadratic_roots (x : ℝ) : x^2 = 3 ↔ (x = real.sqrt 3 ∨ x = -real.sqrt 3) :=
sorry

end quadratic_roots_l381_381068


namespace complement_union_M_N_eq_ge_2_l381_381960

def U := set.univ ℝ
def M := {x : ℝ | x < 1}
def N := {x : ℝ | -1 < x ∧ x < 2}

theorem complement_union_M_N_eq_ge_2 :
  (U \ (M ∪ N)) = {x : ℝ | 2 ≤ x} :=
by sorry

end complement_union_M_N_eq_ge_2_l381_381960


namespace compute_logarithmic_sum_l381_381229

theorem compute_logarithmic_sum : 
  ∑ k in finset.Icc 3 100, (log 3 (1 + (1 : ℝ) / k) * log k 3 * log (k + 1) 3) = -1 := 
sorry

end compute_logarithmic_sum_l381_381229


namespace non_zero_inverses_in_Z_mod_p_wilson_theorem_l381_381441

-- Proof Problem 1: All non-zero elements in ℤ / pℤ have an inverse if p is prime.
theorem non_zero_inverses_in_Z_mod_p (p : ℕ) [fact (nat.prime p)] (k : ℤ) (hk : k % p ≠ 0) : 
  ∃ l : ℤ, (k * l) % p = 1 % p := 
sorry

-- Proof Problem 2: Wilson's theorem
theorem wilson_theorem (n : ℕ) : 
  n.is_prime ↔ ((n - 1)! ≡ -1 [MOD n]) := 
sorry

end non_zero_inverses_in_Z_mod_p_wilson_theorem_l381_381441


namespace triangle_area_is_24_l381_381568

-- Defining the vertices of the triangle
def A := (2, 2)
def B := (8, 2)
def C := (4, 10)

-- Calculate the area of the triangle
def area_of_triangle (A B C : ℕ × ℕ) : ℕ := 
  let base := |B.1 - A.1| 
  let height := |C.2 - A.2| 
  ((base * height) / 2)

-- Statement to prove
theorem triangle_area_is_24 : area_of_triangle A B C = 24 := 
by
  sorry

end triangle_area_is_24_l381_381568


namespace max_students_per_class_l381_381546

-- Definitions used in Lean 4 statement:
def num_students := 920
def seats_per_bus := 71
def num_buses := 16

-- The main statement, showing this is the maximum value such that each class stays together within the given constraints.
theorem max_students_per_class : ∃ k, (∀ k' : ℕ, k' > k → 
  ¬∃ (classes : ℕ), classes * k' + (num_students - classes * k') ≤ seats_per_bus * num_buses ∧ k' <= seats_per_bus) ∧ k = 17 := 
by sorry

end max_students_per_class_l381_381546


namespace dress_assignment_l381_381757

theorem dress_assignment :
  ∃ (Katya Olya Liza Rita : string),
    (Katya ≠ "Pink" ∧ Katya ≠ "Blue") ∧
    (Rita ≠ "Green" ∧ Rita ≠ "Blue") ∧
    ∃ (girl_in_green girl_in_yellow : string),
      (girl_in_green = Katya ∧ girl_in_yellow = Rita ∧ 
       (Liza = "Pink" ∧ Olya = "Blue") ∧
       (Katya = "Green" ∧ Olya = "Blue" ∧ Liza = "Pink" ∧ Rita = "Yellow")) ∧
    ((girl_in_green stands between Liza and girl_in_yellow) ∧
     (Olya stands between Rita and Liza)) :=
by
  sorry

end dress_assignment_l381_381757


namespace focus_of_parabola_l381_381715

-- Define the equation of the given parabola
def given_parabola (x y : ℝ) : Prop := y = - (1 / 8) * x^2

-- Define the condition for the focus of the parabola
def is_focus (focus : ℝ × ℝ) : Prop := focus = (0, -2)

-- State the theorem
theorem focus_of_parabola : ∃ (focus : ℝ × ℝ), given_parabola x y → is_focus focus :=
by
  -- Placeholder proof
  sorry

end focus_of_parabola_l381_381715


namespace who_wears_which_dress_l381_381736

def girls := ["Katya", "Olya", "Liza", "Rita"]
def dresses := ["pink", "green", "yellow", "blue"]

variable (who_wears_dress : String → String)

/-- Conditions given in the problem --/
axiom Katya_not_pink_blue : who_wears_dress "Katya" ≠ "pink" ∧ who_wears_dress "Katya" ≠ "blue"
axiom between_green_liza_yellow : ∃ g, who_wears_dress "Katya" = "green" ∧ who_wears_dress "Rita" = "yellow"
axiom Rita_not_green_blue : who_wears_dress "Rita" ≠ "green" ∧ who_wears_dress "Rita" ≠ "blue"
axiom Olya_between_rita_pink : ∃ o, who_wears_dress "Olya" = "blue" ∧ who_wears_dress "Liza" = "pink"

theorem who_wears_which_dress :
  who_wears_dress "Katya" = "green" ∧
  who_wears_dress "Olya" = "blue" ∧
  who_wears_dress "Liza" = "pink" ∧
  who_wears_dress "Rita" = "yellow" :=
by
  sorry

end who_wears_which_dress_l381_381736


namespace dress_assignments_l381_381746

structure GirlDress : Type :=
  (Katya Olya Liza Rita : String)

def dresses := ["Pink", "Green", "Yellow", "Blue"]

axiom not_pink_or_blue : GirlDress.Katya ≠ "Pink" ∧ GirlDress.Katya ≠ "Blue"
axiom green_between_liza_yellow : (GirlDress.Liza = "Pink" ∨ GirlDress.Rita = "Yellow") ∧
                                  GirlDress.Katya = "Green" ∧
                                  GirlDress.Rita = "Yellow" ∧ GirlDress.Liza = "Pink"
axiom not_green_or_blue : GirlDress.Rita ≠ "Green" ∧ GirlDress.Rita ≠ "Blue"
axiom olya_between_rita_pink : GirlDress.Olya ≠ "Pink" → GirlDress.Rita ≠ "Pink" → GirlDress.Liza = "Pink"

theorem dress_assignments (gd : GirlDress) :
  gd.Katya = "Green" ∧ gd.Olya = "Blue" ∧ gd.Liza = "Pink" ∧ gd.Rita = "Yellow" :=
by
  sorry

end dress_assignments_l381_381746


namespace dress_assignment_l381_381768

-- Define the four girls
inductive Girl
| Katya
| Olya
| Liza
| Rita

-- Define the four dresses
inductive Dress
| Pink
| Green
| Yellow
| Blue

-- Define the function that assigns each girl a dress
def dressOf : Girl → Dress

-- Conditions as definitions
axiom KatyaNotPink : dressOf Girl.Katya ≠ Dress.Pink
axiom KatyaNotBlue : dressOf Girl.Katya ≠ Dress.Blue
axiom RitaNotGreen : dressOf Girl.Rita ≠ Dress.Green
axiom RitaNotBlue : dressOf Girl.Rita ≠ Dress.Blue

axiom GreenBetweenLizaAndYellow : 
  ∃ (arrangement : List Girl), 
    arrangement = [Girl.Liza, Girl.Katya, Girl.Rita] ∧ 
    (dressOf Girl.Katya = Dress.Green ∧ 
    dressOf Girl.Liza = Dress.Pink ∧ 
    dressOf Girl.Rita = Dress.Yellow)

axiom OlyaBetweenRitaAndPink : 
  ∃ (arrangement : List Girl),
    arrangement = [Girl.Rita, Girl.Olya, Girl.Liza] ∧ 
    (dressOf Girl.Olya = Dress.Blue ∧ 
     dressOf Girl.Rita = Dress.Yellow ∧ 
     dressOf Girl.Liza = Dress.Pink)

-- Problem: Determine the dress assignments
theorem dress_assignment : 
  dressOf Girl.Katya = Dress.Green ∧ 
  dressOf Girl.Olya = Dress.Blue ∧ 
  dressOf Girl.Liza = Dress.Pink ∧ 
  dressOf Girl.Rita = Dress.Yellow :=
sorry

end dress_assignment_l381_381768


namespace problem_statement_l381_381979

open Set

variable (U : Set ℝ) (M N : Set ℝ)

theorem problem_statement (hU : U = univ) (hM : M = {x | x < 1}) (hN : N = {x | -1 < x ∧ x < 2}) :
  {x | 2 ≤ x} = compl (M ∪ N) :=
sorry

end problem_statement_l381_381979


namespace total_guests_l381_381093

theorem total_guests (G : ℕ) 
  (hwomen: ∃ n, n = G / 2)
  (hmen: 15 = 15)
  (hchildren: ∃ n, n = G - (G / 2 + 15))
  (men_leaving: ∃ n, n = 1/5 * 15)
  (children_leaving: 4 = 4)
  (people_stayed: 43 = G - ((1/5 * 15) + 4))
  : G = 50 := by
  sorry

end total_guests_l381_381093


namespace gwen_spending_l381_381308

theorem gwen_spending : 
    ∀ (initial_amount spent remaining : ℕ), 
    initial_amount = 7 → remaining = 5 → initial_amount - remaining = 2 :=
by
    sorry

end gwen_spending_l381_381308


namespace investment_duration_l381_381286

theorem investment_duration 
  (P : ℝ) (A : ℝ) (r : ℝ) (t : ℝ)
  (h1 : P = 939.60)
  (h2 : A = 1120)
  (h3 : r = 8) :
  t = 2.4 :=
by
  sorry

end investment_duration_l381_381286


namespace who_wears_which_dress_l381_381732

-- Define the possible girls
inductive Girl
| Katya | Olya | Liza | Rita
deriving DecidableEq

-- Define the possible dresses
inductive Dress
| Pink | Green | Yellow | Blue
deriving DecidableEq

-- Define the fact that each girl is wearing a dress
structure Wearing (girl : Girl) (dress : Dress) : Prop

-- Define the conditions
theorem who_wears_which_dress :
  (¬ Wearing Girl.Katya Dress.Pink ∧ ¬ Wearing Girl.Katya Dress.Blue) ∧
  (∀ g1 g2 g3, Wearing g1 Dress.Green → (Wearing g2 Dress.Pink ∧ Wearing g3 Dress.Yellow → (g2 = Girl.Liza ∧ (g3 = Girl.Rita)) ∨ (g3 = Girl.Liza ∧ g2 = Girl.Rita))) ∧
  (¬ Wearing Girl.Rita Dress.Green ∧ ¬ Wearing Girl.Rita Dress.Blue) ∧
  (∀ g1 g2, (Wearing g1 Dress.Pink ∧ Wearing g2 Dress.Yellow) → Girl.Olya = g2 ∧ Girl.Rita = g1) →
  (Wearing Girl.Katya Dress.Green ∧ Wearing Girl.Olya Dress.Blue ∧ Wearing Girl.Liza Dress.Pink ∧ Wearing Girl.Rita Dress.Yellow) :=
by
  sorry

end who_wears_which_dress_l381_381732


namespace sum_of_digits_of_min_N_l381_381666

-- Define the sequence of operations
def Bernardo (x : Int) : Int := 3 * x
def Silvia (x : Int) : Int := x + 100

-- Define the game ending condition
def gameEnds (x : Int) : Bool := x > 2000

-- Prove the sum of the digits of the smallest initial number N resulting in a win for Bernardo
theorem sum_of_digits_of_min_N : ∃ N : Int, 0 ≤ N ∧ N ≤ 1999 ∧ 
  let final_num := Bernardo (Silvia (Bernardo (Silvia (Bernardo N)))) 
  in 1900 ≤ final_num ∧ final_num ≤ 2000 ∧ (N.digits.sum = 8) := 
by
  sorry

end sum_of_digits_of_min_N_l381_381666


namespace avery_egg_cartons_l381_381205

theorem avery_egg_cartons 
  (num_chickens : ℕ) (eggs_per_chicken : ℕ) (carton_capacity : ℕ)
  (h1 : num_chickens = 20) (h2 : eggs_per_chicken = 6) (h3 : carton_capacity = 12) :
  (num_chickens * eggs_per_chicken) / carton_capacity = 10 :=
by sorry

end avery_egg_cartons_l381_381205


namespace exists_point_Q_l381_381155

variables {A B C P Q X Y : Point}
variables {α : RealAngle}

-- Given conditions
axiom angle_ABC_lt_90 (α : RealAngle) (h : α < 90) : 
  ∃ P : Point, is_inside_angle P A B C ∧ 
  (∃ Q : Point, is_inside_angle Q A B C ∧
  (∀ X Y : Point, (is_on_ray X B A) ∧ (is_on_ray Y B C) ∧ (angle X P Y = α) → 
  (angle X Q Y = 180 - 2 * α)))

-- The problem statement in Lean with the equivalent definitions
theorem exists_point_Q (α : RealAngle) (h1 : α < 90) (P : Point) (h2 : is_inside_angle P A B C):
  ∃ Q : Point, is_inside_angle Q A B C ∧ 
  (∀ X Y : Point, (is_on_ray X B A) ∧ (is_on_ray Y B C) ∧ (angle X P Y = α) → 
  (angle X Q Y = 180 - 2 * α)) := 
sorry

end exists_point_Q_l381_381155


namespace g_eq_f_4_minus_x_l381_381060

-- Definition of functions f and g
def f (x : ℝ) : ℝ := -- Function f, behavior is to be tailored as necessary
  if x >= -3 ∧ x <= 0 then -2 - x
  else if x >= 0 ∧ x <= 2 then real.sqrt(4 - (x - 2)^2) - 2
  else if x >= 2 ∧ x <= 3 then 2 * (x - 2)
  else 0 -- Out of the specified domain, defined as 0 for convenience

def g (x : ℝ) : ℝ := f(-x + 4)

-- The final theorem to be proven
theorem g_eq_f_4_minus_x : ∀ x : ℝ, g(x) = f(4 - x) := by
  sorry

end g_eq_f_4_minus_x_l381_381060


namespace complement_union_eq_l381_381971

-- Definitions / Conditions
def U : Set ℝ := Set.univ
def M : Set ℝ := { x | x < 1 }
def N : Set ℝ := { x | -1 < x ∧ x < 2 }

-- Statement of the theorem
theorem complement_union_eq {x : ℝ} :
  {x | x ≥ 2} = (U \ (M ∪ N)) := sorry

end complement_union_eq_l381_381971


namespace smallest_q_for_5_in_range_l381_381303

theorem smallest_q_for_5_in_range : ∃ q, (q = 9) ∧ (∃ x, (x^2 - 4 * x + q = 5)) := 
by 
  sorry

end smallest_q_for_5_in_range_l381_381303


namespace who_wears_which_dress_l381_381733

-- Define the possible girls
inductive Girl
| Katya | Olya | Liza | Rita
deriving DecidableEq

-- Define the possible dresses
inductive Dress
| Pink | Green | Yellow | Blue
deriving DecidableEq

-- Define the fact that each girl is wearing a dress
structure Wearing (girl : Girl) (dress : Dress) : Prop

-- Define the conditions
theorem who_wears_which_dress :
  (¬ Wearing Girl.Katya Dress.Pink ∧ ¬ Wearing Girl.Katya Dress.Blue) ∧
  (∀ g1 g2 g3, Wearing g1 Dress.Green → (Wearing g2 Dress.Pink ∧ Wearing g3 Dress.Yellow → (g2 = Girl.Liza ∧ (g3 = Girl.Rita)) ∨ (g3 = Girl.Liza ∧ g2 = Girl.Rita))) ∧
  (¬ Wearing Girl.Rita Dress.Green ∧ ¬ Wearing Girl.Rita Dress.Blue) ∧
  (∀ g1 g2, (Wearing g1 Dress.Pink ∧ Wearing g2 Dress.Yellow) → Girl.Olya = g2 ∧ Girl.Rita = g1) →
  (Wearing Girl.Katya Dress.Green ∧ Wearing Girl.Olya Dress.Blue ∧ Wearing Girl.Liza Dress.Pink ∧ Wearing Girl.Rita Dress.Yellow) :=
by
  sorry

end who_wears_which_dress_l381_381733


namespace find_dot_product_l381_381797

noncomputable def vector_space := ℝ

structure IsoscelesTrapezoid (α : Type*) [add_comm_group α] [module vector_space α] :=
(A B C D : α)
(AD_parallel_BC : ∀ (k : vector_space), A + k • D = B + k • C)
(angle_ABC_eq_60 : ∠ ABC = 60)
(BC_eq_2AD : ∥C - B∥ = 2 * ∥D - A∥)
(BC_eq_4 : ∥C - B∥ = 4)

variables {α : Type*} [inner_product_space vector_space α] {t : IsoscelesTrapezoid α}

noncomputable def vector_mod (v:α) : α := 1/3 • v

theorem find_dot_product (CE_eq_CD : t.vector_mod t.CD = t.CE ) :
  (t.vector_mod (t.C - t.A)) * (t.vector_mod (t.B - t.C)) = -10 :=
sorry

end find_dot_product_l381_381797


namespace angle_between_generatrices_is_60_degrees_l381_381812

-- Define a cone and its lateral surface unfolding into a semicircle.
structure Cone :=
  (generatrix_length : ℝ)
  (base_circumference : ℝ)
  (base_diameter : ℝ)

def lateral_surface_unfolds_to_semicircle (c : Cone) : Prop :=
  c.base_circumference = π * c.generatrix_length ∧
  c.base_diameter = c.generatrix_length

theorem angle_between_generatrices_is_60_degrees (c : Cone) 
  (h : lateral_surface_unfolds_to_semicircle c) : 
  ∃ θ : ℝ, θ = 60 ∧ θ = ∠(two_generatrices c) :=
sorry

end angle_between_generatrices_is_60_degrees_l381_381812


namespace greatest_y_least_y_greatest_integer_y_l381_381586

theorem greatest_y (y : ℤ) (H : (8 : ℝ) / 11 > y / 17) : y ≤ 12 :=
sorry

theorem least_y (y : ℤ) (H : (8 : ℝ) / 11 > y / 17) : y ≥ 12 :=
sorry

theorem greatest_integer_y : ∀ (y : ℤ), ((8 : ℝ) / 11 > y / 17) → y = 12 :=
by
  intro y H
  apply le_antisymm
  apply greatest_y y H
  apply least_y y H

end greatest_y_least_y_greatest_integer_y_l381_381586


namespace percentage_exceeds_l381_381888

-- Defining the constants and conditions
variables {y z x : ℝ}

-- Conditions
def condition1 (y x : ℝ) : Prop := x = 0.6 * y
def condition2 (x z : ℝ) : Prop := z = 1.25 * x

-- Proposition to prove
theorem percentage_exceeds (hyx : condition1 y x) (hxz : condition2 x z) : y = 4/3 * z :=
by 
  -- We skip the proof as requested
  sorry

end percentage_exceeds_l381_381888


namespace dress_assignment_l381_381752

variables {Girl : Type} [Finite Girl]
variables (Katya Olya Liza Rita Pink Green Yellow Blue : Girl)
variables (standing_between : Girl → Girl → Girl → Prop)

-- Conditions
variable (cond1 : Katya ≠ Pink ∧ Katya ≠ Blue)
variable (cond2 : standing_between Green Liza Yellow)
variable (cond3 : Rita ≠ Green ∧ Rita ≠ Blue)
variable (cond4 : standing_between Olya Rita Pink)

-- Theorem statement
theorem dress_assignment :
  Katya = Green ∧ Olya = Blue ∧ Liza = Pink ∧ Rita = Yellow := 
sorry

end dress_assignment_l381_381752


namespace problem_equivalent_l381_381939

def U : Set ℝ := Set.univ
def M : Set ℝ := {x | x < 1}
def N : Set ℝ := {x | -1 < x ∧ x < 2}

theorem problem_equivalent :
  {x : ℝ | x ≥ 2} = (U \ (M ∪ N)) := 
by sorry

end problem_equivalent_l381_381939


namespace quadrilateral_type_l381_381829

theorem quadrilateral_type (m n p q : ℝ) (h : m^2 + n^2 + p^2 + q^2 = 2 * m * n + 2 * p * q) : 
  (m = n ∧ p = q) ∨ (m ≠ n ∧ p ≠ q ∧ ∃ k : ℝ, k^2 * (m^2 + n^2) = p^2 + q^2) := 
sorry

end quadrilateral_type_l381_381829


namespace trapezoid_area_ratio_l381_381552

noncomputable def ef := 12
noncomputable def eo := 12
noncomputable def og := 12
noncomputable def gh := 12
noncomputable def fg := 18
noncomputable def eh := 18
noncomputable def oh := 18

def trapezoid_height := 9
def midpoint_div_area (area: ℝ) := area / 2
def segment_midpoint (length: ℝ) := length / 2

theorem trapezoid_area_ratio :
  let EFGH_area := (ef + gh) * trapezoid_height / 2,
      ZW_length := fg + eh,
      EFWZ_area := (ef + ZW_length) * (trapezoid_height / 2) / 2,
      ZWGH_area := (ZW_length + gh) * (trapezoid_height / 2) / 2
  in EFWZ_area = ZWGH_area ∧ EFWZ_area / ZWGH_area = 1 ∧ (1 + 1) = 2 := by
  sorry

end trapezoid_area_ratio_l381_381552


namespace school_total_students_l381_381456

theorem school_total_students (front_rank back_rank grades classes_per_grade : ℕ) 
  (h_front : front_rank = 12) (h_back : back_rank = 12) : 
  grades = 3 → classes_per_grade = 12 → 
  let students_in_class := front_rank - 1 + 1 + back_rank - 1 in
  let students_per_grade := classes_per_grade * students_in_class in
  let total_students := grades * students_per_grade in
  total_students = 828 := sorry

end school_total_students_l381_381456


namespace length_AB_l381_381129

theorem length_AB :
  let line_eq := ∀ x y : ℝ, y = x - 1
  let ellipse_eq := ∀ x y : ℝ, (x^2 / 4) + (y^2 / 3) = 1
  ∃ A B : ℝ × ℝ, line_eq A.1 A.2 ∧ line_eq B.1 B.2 ∧ ellipse_eq A.1 A.2 ∧ ellipse_eq B.1 B.2 → dist A B = 24 / 7 :=
sorry

end length_AB_l381_381129


namespace train_crossing_time_l381_381843

-- Define the parameters given in the problem
def train_length : ℝ := 150   -- Train length in meters
def bridge_length : ℝ := 250  -- Bridge length in meters
def train_speed_kmph : ℝ := 50  -- Speed in kilometers per hour

-- Convert train speed from kmph to m/s
def train_speed_mps : ℝ := train_speed_kmph * (1000 / 3600)

-- Define the total distance
def total_distance : ℝ := train_length + bridge_length

-- State the theorem to be proved
theorem train_crossing_time : (total_distance / train_speed_mps) ≈ 28.8 :=
by
  -- Proof is omitted
  sorry

end train_crossing_time_l381_381843


namespace intersection_of_sets_l381_381357

open Set

theorem intersection_of_sets :
  let M := {x : ℝ | x^2 - 2*x - 3 ≤ 0}
  let N := {x : ℝ | x > 0}
  M ∩ N = {x : ℝ | 0 < x ∧ x ≤ 3} :=
by 
  let M := {x : ℝ | x^2 - 2*x - 3 ≤ 0}
  let N := {x : ℝ | x > 0}
  have hM : M = {x : ℝ | -1 ≤ x ∧ x ≤ 3} := sorry 
  have hN : N = {x : ℝ | x > 0} := sorry 
  rw [hM, hN]
  ext x
  simp [and_assoc]
  sorry

end intersection_of_sets_l381_381357


namespace question_proof_l381_381982

open Set

variable (U : Set ℝ := univ)
variable (M : Set ℝ := {x | x < 1})
variable (N : Set ℝ := {x | -1 < x ∧ x < 2})

theorem question_proof : {x | x ≥ 2} = compl (M ∪ N) :=
by
  sorry

end question_proof_l381_381982


namespace problem_l381_381842

variables {ℝ : Type*} [linear_order ℝ]

-- Assume f and g are n times differentiable functions
variables {f g : ℝ → ℝ}

-- Assume x₀ is a point in the domain
variable (x₀ : ℝ)

-- Assume n is a natural number
variable (n : ℕ)

-- Conditions
-- 1. f^(k)(x₀) = g^(k)(x₀) for k = 0, 1, 2, ..., n-1
def cond1 (k : ℕ) (hk : k < n) : Prop := 
  deriv^[k] f x₀ = deriv^[k] g x₀ 

-- 2. f^(n)(x) > g^(n)(x) when x > x₀
def cond2 : Prop :=
  ∀ x, x > x₀ → deriv^[n] f x > deriv^[n] g x

-- Target Statement: f(x) > g(x) when x > x₀
theorem problem (h1 : ∀ k < n, cond1 x₀ n k) (h2 : cond2 x₀ f g n) :
  ∀ x, x > x₀ → f x > g x := 
  sorry

end problem_l381_381842


namespace who_wears_which_dress_l381_381741

def girls := ["Katya", "Olya", "Liza", "Rita"]
def dresses := ["pink", "green", "yellow", "blue"]

variable (who_wears_dress : String → String)

/-- Conditions given in the problem --/
axiom Katya_not_pink_blue : who_wears_dress "Katya" ≠ "pink" ∧ who_wears_dress "Katya" ≠ "blue"
axiom between_green_liza_yellow : ∃ g, who_wears_dress "Katya" = "green" ∧ who_wears_dress "Rita" = "yellow"
axiom Rita_not_green_blue : who_wears_dress "Rita" ≠ "green" ∧ who_wears_dress "Rita" ≠ "blue"
axiom Olya_between_rita_pink : ∃ o, who_wears_dress "Olya" = "blue" ∧ who_wears_dress "Liza" = "pink"

theorem who_wears_which_dress :
  who_wears_dress "Katya" = "green" ∧
  who_wears_dress "Olya" = "blue" ∧
  who_wears_dress "Liza" = "pink" ∧
  who_wears_dress "Rita" = "yellow" :=
by
  sorry

end who_wears_which_dress_l381_381741


namespace walking_speed_l381_381148

theorem walking_speed (d : ℝ) (t : ℝ) (h_d : d = 1250) (h_t : t = 15) : 
    let v_m_min := d / t,
        v_km_hr := (v_m_min * 60) / 1000
    in Float.ofReal v_km_hr ≈ 5.00 := 
by
  sorry -- Proof is not required

end walking_speed_l381_381148


namespace dress_assignment_l381_381767

-- Define the four girls
inductive Girl
| Katya
| Olya
| Liza
| Rita

-- Define the four dresses
inductive Dress
| Pink
| Green
| Yellow
| Blue

-- Define the function that assigns each girl a dress
def dressOf : Girl → Dress

-- Conditions as definitions
axiom KatyaNotPink : dressOf Girl.Katya ≠ Dress.Pink
axiom KatyaNotBlue : dressOf Girl.Katya ≠ Dress.Blue
axiom RitaNotGreen : dressOf Girl.Rita ≠ Dress.Green
axiom RitaNotBlue : dressOf Girl.Rita ≠ Dress.Blue

axiom GreenBetweenLizaAndYellow : 
  ∃ (arrangement : List Girl), 
    arrangement = [Girl.Liza, Girl.Katya, Girl.Rita] ∧ 
    (dressOf Girl.Katya = Dress.Green ∧ 
    dressOf Girl.Liza = Dress.Pink ∧ 
    dressOf Girl.Rita = Dress.Yellow)

axiom OlyaBetweenRitaAndPink : 
  ∃ (arrangement : List Girl),
    arrangement = [Girl.Rita, Girl.Olya, Girl.Liza] ∧ 
    (dressOf Girl.Olya = Dress.Blue ∧ 
     dressOf Girl.Rita = Dress.Yellow ∧ 
     dressOf Girl.Liza = Dress.Pink)

-- Problem: Determine the dress assignments
theorem dress_assignment : 
  dressOf Girl.Katya = Dress.Green ∧ 
  dressOf Girl.Olya = Dress.Blue ∧ 
  dressOf Girl.Liza = Dress.Pink ∧ 
  dressOf Girl.Rita = Dress.Yellow :=
sorry

end dress_assignment_l381_381767


namespace minimum_cubes_to_hide_buttons_l381_381141

def cube (c : Type) : Type :=
{ button : c, receptacles : fin 5 → c}

def valid_configuration (cubes : list (cube c)) : Prop :=
∀ (i : ℕ) (h : i < cubes.length), ∃ (j : ℕ) (hj : j < cubes.length), i ≠ j ∧
cubes.nth_le i h = cubes.nth_le j hj ∨ 
(∃ (k : fin 5), cubes.nth_le i h.receptacles k = cubes.nth_le j hj.button)

theorem minimum_cubes_to_hide_buttons : ∃ (n : ℕ), n = 5 ∧
∃ (cubes : list (cube c)), cubes.length = n ∧ valid_configuration cubes :=
sorry

end minimum_cubes_to_hide_buttons_l381_381141


namespace allison_total_supply_items_is_28_l381_381182

/-- Define the number of glue sticks Marie bought --/
def marie_glue_sticks : ℕ := 15
/-- Define the number of packs of construction paper Marie bought --/
def marie_construction_paper_packs : ℕ := 30
/-- Define the number of glue sticks Allison bought --/
def allison_glue_sticks : ℕ := marie_glue_sticks + 8
/-- Define the number of packs of construction paper Allison bought --/
def allison_construction_paper_packs : ℕ := marie_construction_paper_packs / 6
/-- Calculation of the total number of craft supply items Allison bought --/
def allison_total_supply_items : ℕ := allison_glue_sticks + allison_construction_paper_packs

/-- Prove that the total number of craft supply items Allison bought is equal to 28. --/
theorem allison_total_supply_items_is_28 : allison_total_supply_items = 28 :=
by sorry

end allison_total_supply_items_is_28_l381_381182


namespace number_of_sarees_l381_381524

-- Define variables representing the prices of one saree and one shirt
variables (X S T : ℕ)

-- Define the conditions 
def condition1 := X * S + 4 * T = 1600
def condition2 := S + 6 * T = 1600
def condition3 := 12 * T = 2400

-- The proof problem (statement only, without proof)
theorem number_of_sarees (X S T : ℕ) (h1 : condition1 X S T) (h2 : condition2 S T) (h3 : condition3 T) : X = 2 := by
  sorry

end number_of_sarees_l381_381524


namespace amy_spelling_problems_l381_381725

-- Defining the conditions as given in the problem
def math_problems : ℕ := 18
def problems_per_hour : ℕ := 4
def hours : ℕ := 6

-- The theorem to prove the number of spelling problems
theorem amy_spelling_problems : 
  math_problems = 18 → problems_per_hour = 4 → hours = 6 →
  (let total_problems := problems_per_hour * hours in
  total_problems - math_problems = 6) :=
begin
  intros h1 h2 h3,
  let total_problems := problems_per_hour * hours,
  have h4: total_problems = 24 := by rw [h2, h3]; norm_num,
  rw [h4, h1],
  norm_num,
  rfl,
end

end amy_spelling_problems_l381_381725


namespace allison_total_supply_items_is_28_l381_381180

/-- Define the number of glue sticks Marie bought --/
def marie_glue_sticks : ℕ := 15
/-- Define the number of packs of construction paper Marie bought --/
def marie_construction_paper_packs : ℕ := 30
/-- Define the number of glue sticks Allison bought --/
def allison_glue_sticks : ℕ := marie_glue_sticks + 8
/-- Define the number of packs of construction paper Allison bought --/
def allison_construction_paper_packs : ℕ := marie_construction_paper_packs / 6
/-- Calculation of the total number of craft supply items Allison bought --/
def allison_total_supply_items : ℕ := allison_glue_sticks + allison_construction_paper_packs

/-- Prove that the total number of craft supply items Allison bought is equal to 28. --/
theorem allison_total_supply_items_is_28 : allison_total_supply_items = 28 :=
by sorry

end allison_total_supply_items_is_28_l381_381180


namespace complex_fraction_identity_l381_381006

theorem complex_fraction_identity (c d : ℂ) (h_nonzero_c : c ≠ 0) (h_nonzero_d : d ≠ 0) (h_condition : c^2 + c * d + d^2 = 0) : 
  (c^12 + d^12) / (c + d)^12 = -2 :=
by sorry

end complex_fraction_identity_l381_381006


namespace distance_between_vertices_l381_381510

theorem distance_between_vertices :
  let y1 := 3
  let y2 := -1
  abs (y1 - y2) = 4 :=
by
  let y1 := 3
  let y2 := -1
  show abs (y1 - y2) = 4
  from sorry

end distance_between_vertices_l381_381510


namespace simplify_expression_l381_381039

variables (a b : ℝ) (h₁ : a ≠ 0) (h₂ : b ≠ 0) (h₃ : a ≠ b)

theorem simplify_expression :
  (a^4 - a^2 * b^2) / (a - b)^2 / (a * (a + b) / b^2) * (b^2 / a) = b^4 / (a - b) :=
by
  sorry

end simplify_expression_l381_381039


namespace irrational_sqrt2_gt1_l381_381029

theorem irrational_sqrt2_gt1 : ∃ x : ℝ, irrational x ∧ x > 1 :=
by
  let x := Real.sqrt 2
  have h_irr : irrational x, from sorry
  have h_gt_1 : x > 1, from sorry
  exact ⟨x, h_irr, h_gt_1⟩

end irrational_sqrt2_gt1_l381_381029


namespace find_difference_l381_381520

variables (x y : ℝ)

theorem find_difference (h1 : x * (y + 2) = 100) (h2 : y * (x + 2) = 60) : x - y = 20 :=
sorry

end find_difference_l381_381520


namespace find_circle_radius_l381_381807

def circle1_equation (x y : ℝ) (m : ℝ) := x^2 + y^2 = m
def circle2_equation (x y : ℝ) := x^2 + y^2 + 6*x - 8*y - 11 = 0
def internally_tangent (m : ℝ) : Prop :=
  ∃ x y : ℝ, 
    circle1_equation x y m ∧
    circle2_equation x y ∧
    (m = 1 ∨ m = 121)

theorem find_circle_radius (m : ℝ) : 
  internally_tangent m :=
begin
  sorry
end

end find_circle_radius_l381_381807


namespace allison_craft_items_l381_381175

def glue_sticks (A B : Nat) : Prop := A = B + 8
def construction_paper (A B : Nat) : Prop := B = 6 * A

theorem allison_craft_items (Marie_glue_sticks Marie_paper_packs : Nat)
    (h1 : Marie_glue_sticks = 15)
    (h2 : Marie_paper_packs = 30) :
    ∃ (Allison_glue_sticks Allison_paper_packs total_items : Nat),
        glue_sticks Allison_glue_sticks Marie_glue_sticks ∧
        construction_paper Allison_paper_packs Marie_paper_packs ∧
        total_items = Allison_glue_sticks + Allison_paper_packs ∧
        total_items = 28 :=
by
    sorry

end allison_craft_items_l381_381175


namespace correct_steps_ordered_l381_381912

-- Conditions
def step_CollectingData : ℕ := 1
def step_DesigningSurveyQuestionnaires : ℕ := 2
def step_EstimatingPopulation : ℕ := 3
def step_OrganizingData : ℕ := 4
def step_AnalyzingData : ℕ := 5

def steps : List ℕ :=
  [step_CollectingData,
   step_DesigningSurveyQuestionnaires,
   step_EstimatingPopulation,
   step_OrganizingData,
   step_AnalyzingData]

def correct_order : List ℕ :=
  [step_DesigningSurveyQuestionnaires,
   step_CollectingData,
   step_OrganizingData,
   step_AnalyzingData,
   step_EstimatingPopulation]

-- Theorem statement
theorem correct_steps_ordered : 
  ∃ (ordered_steps : List ℕ), ordered_steps = correct_order :=
by 
  use correct_order
  sorry

end correct_steps_ordered_l381_381912


namespace min_value_2_plus_y_l381_381328

theorem min_value_2_plus_y (y : ℝ) (x : ℝ) (h1 : y > 0) (h2 : x^2 + y - 3 = 0) : 2 + y = 2 := 
by
  have h3 : y = 3 - x^2 := by linarith
  have h4 : 2 + y = 2 + 3 - x^2 := by rw [h3]
  have h5 : 2 + y = 5 - x^2 := by linarith
  have h6 : x^2 < 3 := by linarith
  have h7 : 2 + y = 5 - 3 := by { have h8 : x^2 = 3, from sorry, rw [h8] at h5, assumption }
  exact h7

end min_value_2_plus_y_l381_381328


namespace sqrt_calculation_l381_381678

theorem sqrt_calculation :
  Real.sqrt ((2:ℝ)^4 * 3^2 * 5^2) = 60 := 
by sorry

end sqrt_calculation_l381_381678


namespace hyperbola_eccentricity_l381_381881

theorem hyperbola_eccentricity (a b : ℝ) (h1 : a > 0) (h2 : b > 0) 
    (h3 : (ab / √(a^2 + b^2) = b / 2)) : (eccentricity : ℝ) = 2 :=
by
  -- Definitions and steps to prove the eccentricity
  sorry

end hyperbola_eccentricity_l381_381881


namespace sum_log_diff_eq_1998953_l381_381675

def ceil_floor_diff (x : ℝ) : ℝ :=
  ⌈x⌉ - ⌊x⌋

theorem sum_log_diff_eq_1998953 : 
  (∑ k in Finset.range 2000, (k+1) * (ceil_floor_diff (Real.log (k+1) / Real.log 2))) = 1998953 := by
  sorry

end sum_log_diff_eq_1998953_l381_381675


namespace temp_below_zero_correct_l381_381369

def temp_above_zero_notation : ℤ → ℤ := λ temp, temp
def temp_below_zero_notation : ℤ → ℤ := λ temp, -temp

theorem temp_below_zero_correct (temp : ℤ) :
  temp_below_zero_notation 5 = -5 :=
by
  have h : temp_above_zero_notation 5 = 5 := rfl
  exact rfl

end temp_below_zero_correct_l381_381369


namespace tan_alpha_value_l381_381779

theorem tan_alpha_value (α : ℝ) (h : (sin α - 2 * cos α) / (3 * sin α + 5 * cos α) = -5) : 
  tan α = -23 / 16 := 
by 
  sorry -- Proof to be completed.

end tan_alpha_value_l381_381779


namespace triangle_ABC_AC_length_l381_381890

noncomputable def length_of_ac (x : ℝ) : ℝ :=
let y := (5 + Real.sqrt 17) / 2 in
2 * (y - 2)

theorem triangle_ABC_AC_length :
  ∀ (x : ℝ) (A B C D E : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] 
    (AB AC AD DC AE BC BE DE EC : ℝ) 
    (h1 : AD = DC = x) 
    (h2 : AB ⟂ AC) 
    (h3 : AE ⟂ BC) 
    (h4 : DE = EC = 2),
  2 * (x - 2) = 1 + Real.sqrt 17 :=
by
  sorry

end triangle_ABC_AC_length_l381_381890


namespace area_triangle_PAC_l381_381665

theorem area_triangle_PAC 
  (A B C D P : Type)
  (side_AB : ℝ)
  (PA PB : ℝ)
  (square : (A B C D : Type) → A ∈ B ∧ B ∈ C ∧ C ∈ D ∧ D ∈ A)
  (isosceles_triangle : (P A B : Type) → PA = 10 ∧ PB = 10 ∧ side_AB = 12) : 
  ∃ area : ℝ, area = 12 :=
  by 
  sorry

end area_triangle_PAC_l381_381665


namespace sum_log_identity_l381_381245

noncomputable def log_base (b x : ℝ) : ℝ := Real.log x / Real.log b

theorem sum_log_identity :
  ∑ k in Finset.range (98) + 3, log_base 3 (1 + 1 / k) * log_base k 3 * log_base (k + 1) 3 =
  1 - 1 / log_base 3 101 :=
by
  sorry

end sum_log_identity_l381_381245


namespace shaded_square_percentage_l381_381595

/-- 
Given a seven-by-seven square grid and 20 shaded squares,
prove that the percentage of the grid that is shaded is 40.82%.
-/
theorem shaded_square_percentage (total_squares : ℕ) (shaded_squares : ℕ) 
  (h_total : total_squares = 7 * 7) (h_shaded : shaded_squares = 20) : 
  (shaded_squares.to_real / total_squares.to_real * 100) = 40.82 :=
by
  -- Start the proof here
  revert h_total h_shaded total_squares shaded_squares
  sorry

end shaded_square_percentage_l381_381595


namespace find_cost_price_per_meter_l381_381602

/-- Given that a shopkeeper sells 200 meters of cloth for Rs. 12000 at a loss of Rs. 6 per meter,
we want to find the cost price per meter of cloth. Specifically, we need to prove that the
cost price per meter is Rs. 66. -/
theorem find_cost_price_per_meter
  (total_meters : ℕ := 200)
  (selling_price : ℕ := 12000)
  (loss_per_meter : ℕ := 6) :
  (selling_price + total_meters * loss_per_meter) / total_meters = 66 :=
sorry

end find_cost_price_per_meter_l381_381602


namespace trapezoid_area_is_correct_l381_381632

noncomputable def area_of_trapezoid (CM DM : ℝ) (AD BC KL : ℝ) : ℝ :=
  (AD + BC) / 2 * KL

theorem trapezoid_area_is_correct :
  ∀ (CM DM : ℝ) (AD BC KL : ℝ), CM = 4 → DM = 9 → AD = 18 → BC = 8 → KL = 12 →
  area_of_trapezoid CM DM AD BC KL = 156 :=
by
  intros CM DM AD BC KL hCM hDM hAD hBC hKL
  -- Verify the given values.
  rw [hCM, hDM, hAD, hBC, hKL]
  -- Substitute into the area calculation.
  exact rfl

end trapezoid_area_is_correct_l381_381632


namespace plane_cylinder_intersect_l381_381935

theorem plane_cylinder_intersect (O : Point) (a b c p q r : ℝ)
  (hO : O = (0, 0, 0))
  (h_point : (2a, 2b, 2c))
  (α β γ : ℝ)
  (h_plane : plane_pass_through (2a, 2b, 2c) (α, 0, 0) (0, β, 0) (0, 0, γ))
  (h_cylinder : cylinder_axis (2a, 2b, 2c) (p, q, r) (distance O (2a, 2b, 2c))) :
  (2a / p) + (2b / q) + (2c / r) = 1 :=
sorry

end plane_cylinder_intersect_l381_381935


namespace max_X_l381_381423

variable (S : Set ℕ) (S_def : S = {1, 2, 3, ..., 3000})

def is_bijective (f : S → S) : Prop := Function.Bijective f

theorem max_X 
  (f : S → S) (hf : is_bijective f) : 
  ∃ (g : S → S), is_bijective g ∧
  (∑ k in S, (max (f(f(k))) (f(g(k))) (g(f(k))) (g(g(k))) - min (f(f(k))) (f(g(k))) (g(f(k))) (g(g(k))))) ≥ 6000000 :=
sorry

end max_X_l381_381423


namespace sum_of_discounts_l381_381311

theorem sum_of_discounts
  (price_fox : ℝ)
  (price_pony : ℝ)
  (savings : ℝ)
  (discount_pony : ℝ) :
  (3 * price_fox * (F / 100) + 2 * price_pony * (discount_pony / 100) = savings) →
  (F + discount_pony = 22) :=
sorry


end sum_of_discounts_l381_381311


namespace general_term_of_sequence_l381_381310

theorem general_term_of_sequence
  (a : ℕ → ℝ) (b : ℕ → ℝ)
  (h_pos_a : ∀ n, 0 < a n)
  (h_pos_b : ∀ n, 0 < b n)
  (h_arith : ∀ n, 2 * b n = a n + a (n + 1))
  (h_geom : ∀ n, (a (n + 1))^2 = b n * b (n + 1))
  (h_a1 : a 1 = 1)
  (h_a2 : a 2 = 3)
  : ∀ n, a n = (n^2 + n) / 2 :=
by
  sorry

end general_term_of_sequence_l381_381310


namespace derivative_of_f_l381_381612

noncomputable def f (x : ℝ) : ℝ :=
  tan (sqrt (cos (1/3))) + (sin (31 * x))^2 / (31 * cos (62 * x))

theorem derivative_of_f :
  (deriv f) x = 2 * (sin (31 * x) * cos (31 * x) * cos (62 * x) + (sin (31 * x))^2 * sin (62 * x)) / (cos (62 * x))^2 :=
by
  sorry

end derivative_of_f_l381_381612


namespace sin_neg_120_eq_l381_381224

def angle1 := -120
def angle2 := 240
def point := (-1 / 2, -Real.sqrt 3 / 2)

theorem sin_neg_120_eq :
  ∠ angle1 = angle2 ∧ ∃ coords, coords = point -> Real.sin angle1 = -Real.sqrt 3 / 2 :=
by
  sorry

end sin_neg_120_eq_l381_381224


namespace sum_logarithms_l381_381233

theorem sum_logarithms :
  (∑ k in finset.Ico 3 101, real.logb 3 (1 + 1/(k:ℝ)) * real.logb k 3 * real.logb (k+1) 3) =
  1 - real.logb 2 3 / real.logb 2 101 :=
by
  sorry

end sum_logarithms_l381_381233


namespace log_sum_eq_l381_381255

theorem log_sum_eq : ((∑ k in (Finset.range 98).map (λ n, n+3), (Real.log (1 + 1 / (k:ℝ)) / Real.log 3) * (Real.log 3 / Real.log k) * (Real.log 3 / Real.log (k+1)))) = 1 - (1 / Real.log 101 / Real.log 3) :=
by
  sorry

end log_sum_eq_l381_381255


namespace sum_values_A_B_divisible_by_9_sum_of_all_possible_values_l381_381376

theorem sum_values_A_B_divisible_by_9 (A B : ℕ) (h1 : A ≤ 9) (h2 : B ≤ 9) (h3 : (A + B + 25) % 9 = 0) : A + B = 2 ∨ A + B = 11 :=
by sorry

theorem sum_of_all_possible_values :
  let possible_values := {x | ∃ (A B : ℕ), A ≤ 9 ∧ B ≤ 9 ∧ (A + B + 25) % 9 = 0 ∧ A + B = x} in
  ∑ x in possible_values, x = 13 :=
by sorry

end sum_values_A_B_divisible_by_9_sum_of_all_possible_values_l381_381376


namespace avery_egg_cartons_l381_381203

theorem avery_egg_cartons 
  (num_chickens : ℕ) (eggs_per_chicken : ℕ) (carton_capacity : ℕ)
  (h1 : num_chickens = 20) (h2 : eggs_per_chicken = 6) (h3 : carton_capacity = 12) :
  (num_chickens * eggs_per_chicken) / carton_capacity = 10 :=
by sorry

end avery_egg_cartons_l381_381203


namespace sum_of_first_110_terms_l381_381073

theorem sum_of_first_110_terms
  (a d : ℤ)
  (S : ℕ → ℤ)
  (h1 : S 10 = 100)
  (h2 : S 100 = 10)
  (h_sum : ∀ n, S n = n * (2 * a + (n - 1) * d) / 2) :
  S 110 = -110 :=
by {
  sorry,
}

end sum_of_first_110_terms_l381_381073


namespace miles_tankful_highway_l381_381631

variable (miles_tankful_city : ℕ)
variable (mpg_city : ℕ)
variable (mpg_highway : ℕ)

-- Relationship between miles per gallon in city and highway
axiom h_mpg_relation : mpg_highway = mpg_city + 18

-- Given the car travels 336 miles per tankful of gasoline in the city
axiom h_miles_tankful_city : miles_tankful_city = 336

-- Given the car travels 48 miles per gallon in the city
axiom h_mpg_city : mpg_city = 48

-- Prove the car travels 462 miles per tankful of gasoline on the highway
theorem miles_tankful_highway : ∃ (miles_tankful_highway : ℕ), miles_tankful_highway = (mpg_highway * (miles_tankful_city / mpg_city)) := 
by 
  exists (66 * (336 / 48)) -- Since 48 + 18 = 66 and 336 / 48 = 7, 66 * 7 = 462
  sorry

end miles_tankful_highway_l381_381631


namespace minimum_value_of_C_D_l381_381784

open Real

theorem minimum_value_of_C_D
  (x : ℝ) (C D : ℝ)
  (hx_pos : x > 0)
  (hC : x^2 + x⁻² = C)
  (hD : x + x⁻¹ = D)
  (hC_pos : C > 0)
  (hD_pos : D > 0) :
  ∃ m, m = 2 * sqrt 3 + 3 / 2 ∧ (∀ x, x > 0 → ∀ C D, x^2 + x⁻² = C → x + x⁻¹ = D → C > 0 → D > 0 → (C / (D - 2)) ≥ m) :=
by
  sorry

end minimum_value_of_C_D_l381_381784


namespace meishan_artwork_arrangement_l381_381692

def arrange_artworks : ℕ :=
  let calligraphy_units := 2
  let paintings := 2
  let architectural_work := 1
  let total_artworks := calligraphy_units + paintings + architectural_work
  
  -- combine calligraphy works into a single unit
  let combined_units := calligraphy_units - 1 + paintings + architectural_work

  -- arrange combined units and architectural work
  let arrange_combined := (3 : ℕ).factorial

  -- arrange calligraphy works within the combined unit
  let arrange_calligraphy := (2 : ℕ).factorial

  -- arrange paintings in allowed places
  let allowed_painting_positions := 3
  let arrange_paintings := (allowed_painting_positions.factorial) / ((allowed_painting_positions - paintings).factorial)
  
  -- total arrangement after correcting overcounting
  arrange_combined * arrange_calligraphy * arrange_paintings / 2

theorem meishan_artwork_arrangement (result : arrange_artworks = 36) : True :=
begin
  trivial,
end

end meishan_artwork_arrangement_l381_381692


namespace sum_of_first_110_terms_l381_381075

theorem sum_of_first_110_terms
  (a d : ℝ)
  (h1 : (10 : ℝ) * (2 * a + (10 - 1) * d) / 2 = 100)
  (h2 : (100 : ℝ) * (2 * a + (100 - 1) * d) / 2 = 10) :
  (110 : ℝ) * (2 * a + (110 - 1) * d) / 2 = -110 :=
  sorry

end sum_of_first_110_terms_l381_381075


namespace average_of_pqrs_l381_381370

theorem average_of_pqrs (p q r s : ℚ) (h : (5/4) * (p + q + r + s) = 20) : ((p + q + r + s) / 4) = 4 :=
sorry

end average_of_pqrs_l381_381370


namespace rectangles_excluding_squares_in_5x5_grid_l381_381845

-- Definition for a grid and counting rectangles excluding squares in a 5x5 grid
def count_rectangles_excluding_squares : Nat :=
  let total_rectangles := (Nat.choose 5 2) * (Nat.choose 5 2)
  let total_squares := (4 * (5 - 1)^2) + (3 * (5 - 2)^2) + (2 * (5 - 3)^2) + (1 * (5 - 4)^2) 
  total_rectangles - total_squares

-- Statement of the theorem
theorem rectangles_excluding_squares_in_5x5_grid :
  count_rectangles_excluding_squares = 70 :=
begin
  -- Proof is omitted
  sorry
end

end rectangles_excluding_squares_in_5x5_grid_l381_381845


namespace water_surface_no_obtuse_triangle_l381_381898

-- Define a cube-shaped container and its properties
structure Cube :=
  (side_length : ℝ)
  (nonneg : 0 ≤ side_length)

-- Define what a cross-section of a cube is
def cross_section (c : Cube) (θ φ : ℝ) : Set (Point ℝ) :=
  sorry -- placeholder for the actual cross-section calculation

-- Define what it means for the water surface to be a level plane
def level_plane (p : ℝ) : Set (Point ℝ) :=
  {pt | pt.z = p}

-- Define what it means for a water surface inside a tilted cube not forming an obtuse triangle
theorem water_surface_no_obtuse_triangle (c : Cube) (p : ℝ) :
  ∀ θ φ, ¬∃ (pts : Set (Point ℝ)), pts = cross_section c θ φ ∧ pts ⊆ level_plane p ∧ 
    is_obtuse_triangle pts :=
by
  sorry

end water_surface_no_obtuse_triangle_l381_381898


namespace tetrahedron_sequences_common_ratios_l381_381897

theorem tetrahedron_sequences_common_ratios (a : ℝ) (S₁ : ℝ) (V₁ : ℝ) :
  (∀ n : ℕ, (edges n) = a * (1 / 2)^n) ∧
  (∀ n : ℕ, (surface_areas n) = S₁ * (1 / 4)^n) ∧
  (∀ n : ℕ, (volumes n) = V₁ * (1 / 8)^n) := 
by
  -- Definitions based on the conditions
  let edges : ℕ → ℝ := λ n, a * (1 / 2)^n
  let surface_areas : ℕ → ℝ := λ n, S₁ * (1 / 4)^n
  let volumes : ℕ → ℝ := λ n, V₁ * (1 / 8)^n
  sorry

end tetrahedron_sequences_common_ratios_l381_381897


namespace isosceles_triangle_centroid_sum_l381_381256

theorem isosceles_triangle_centroid_sum 
  (a b : ℝ) 
  (h1 : b ≥ a) 
  (is_isosceles_triangle : ∀ (A B C : Type) (ab ac : ℝ), ab = ac → ab = b ∧ ac = b ∧ abc = a)
  (P : Type) 
  (is_centroid : ∀ {ABC : Type} (P : Point), P is_centroid_of ABC) 
  (A' B' C' : Type) 
  (meet_points : ∀ {ABC : Type} (P A B C A' B' C' : Point), 
    P is_centroid_of ABC → 
    A' = line_through P B meet line_through C opposite → 
    B' = line_through P C meet line_through A opposite → 
    C' = line_through P A meet line_through B opposite
  ) : 
  AA' + BB' + CC' = sqrt (4 * b ^ 2 - a ^ 2) :=
sorry

end isosceles_triangle_centroid_sum_l381_381256


namespace determine_x_2y_l381_381373

theorem determine_x_2y (x y : ℝ) (h1 : 2 * x + y = 7) (h2 : (x + y) / 3 = 5 / 3) : x + 2 * y = 8 :=
sorry

end determine_x_2y_l381_381373


namespace greatest_adjacent_to_one_count_good_cells_l381_381461

theorem greatest_adjacent_to_one (arr : array (Fin 3) (Fin 55) ℕ) 
(h1 : arr.read ⟨2, sorry⟩ ⟨0, sorry⟩ = 1)
(h2 : ∀ a ≤ 164, ∃ (i : Fin 3) (j : Fin 55), arr.read i j = a ∧ arr.read (i' : Fin 3) (j' : Fin 55) = a + 1 ∧ 
(adjacent_by_side i j i' j')) : 
∃ (i : Fin 3) (j : Fin 55), (adjacent_by_side 2 0 i j) ∧ arr.read i j = 2 := 
by
  sorry

theorem count_good_cells (arr : array (Fin 3) (Fin 55) ℕ) 
(h1 : arr.read ⟨2, sorry⟩ ⟨0, sorry⟩ = 1)
(h2 : ∀ a ≤ 164, ∃ (i : Fin 3) (j : Fin 55), arr.read i j = a ∧ arr.read (i' : Fin 3) (j' : Fin 55) = a + 1 ∧ 
(adjacent_by_side i j i' j')) :
∃ c, c = 82 ∧ (∀ (i : Fin 3) (j : Fin 55), (can_contain_165 i j ↔ (i, j) ∈ good_cells)) := 
by
  sorry

def adjacent_by_side (i j i' j' : Nat) : Prop :=
  (i = i' ∧ (j = j' + 1 ∨ j' = j + 1)) ∨ (j = j' ∧ (i = i' + 1 ∨ i' = i + 1))

def can_contain_165 (i j : Nat) (arr : array (Fin 3) (Fin 55) ℕ) : Prop :=
  ∀ a ≤ 164, ∃ (i' : Fin 3) (j' : Fin 55), arr.read i' j' = a ∧ adjacent_by_side i j i' j' ∧ arr.read i j = 165

def good_cells : set (Fin 3 × Fin 55) :=
  { (i, j) | can_contain_165 i j arr }

end greatest_adjacent_to_one_count_good_cells_l381_381461


namespace robin_total_spending_l381_381477

def jelly_bracelets_total_cost : ℕ :=
  let names := ["Jessica", "Tori", "Lily", "Patrice"]
  let total_letters := names.foldl (λ acc name => acc + name.length) 0
  total_letters * 2

theorem robin_total_spending : jelly_bracelets_total_cost = 44 := by
  sorry

end robin_total_spending_l381_381477


namespace blocks_needed_for_enclosure_l381_381704

noncomputable def volume_of_rectangular_prism (length: ℝ) (width: ℝ) (height: ℝ) : ℝ :=
  length * width * height

theorem blocks_needed_for_enclosure 
  (length width height thickness : ℝ)
  (H_length : length = 15)
  (H_width : width = 12)
  (H_height : height = 6)
  (H_thickness : thickness = 1.5) :
  volume_of_rectangular_prism length width height - 
  volume_of_rectangular_prism (length - 2 * thickness) (width - 2 * thickness) (height - thickness) = 594 :=
by
  sorry

end blocks_needed_for_enclosure_l381_381704


namespace max_in_set_l381_381868

theorem max_in_set (a : ℝ) (h : a = -2) :
  (∀ x ∈ ({-3 * a, 4 * a, 24 / a, a^2, (1 : ℝ)} : set ℝ), x ≤ 6) ∧ (6 ∈ ({-3 * a, 4 * a, 24 / a, a^2, (1 : ℝ)} : set ℝ)) :=
by {
  -- Instantiate the hypotheses
  -- skipped: proof steps
  sorry
}

end max_in_set_l381_381868


namespace sum_of_first_10_terms_l381_381528

-- Define the arithmetic sequence with the given conditions
def a (n : ℕ) : ℝ := a_1 + (n-1) * d

-- Given conditions
axiom a2 : a 2 = 3
axiom a9 : a 9 = 17

-- Define the sum of the first n terms of an arithmetic sequence
def S (n : ℕ) : ℝ := n * (a 1 + a n) / 2

-- Prove that the sum of the first 10 terms S_10 equals 100
theorem sum_of_first_10_terms : S 10 = 100 := by
  sorry

end sum_of_first_10_terms_l381_381528


namespace farm_corn_cobs_l381_381634

theorem farm_corn_cobs (rows_field1 rows_field2 cobs_per_row : Nat) (h1 : rows_field1 = 13) (h2 : rows_field2 = 16) (h3 : cobs_per_row = 4) : rows_field1 * cobs_per_row + rows_field2 * cobs_per_row = 116 := by
  sorry

end farm_corn_cobs_l381_381634


namespace ratio_equivalence_l381_381119

theorem ratio_equivalence (a b : ℝ) (hb : b ≠ 0) (h : a / b = 5 / 4) : (4 * a + 3 * b) / (4 * a - 3 * b) = 4 :=
sorry

end ratio_equivalence_l381_381119


namespace coin_sums_unique_count_l381_381639

def coin_set := [{1, 2}, {5, 2}, {10, 1}, {25, 1}, {50, 1}]

theorem coin_sums_unique_count (c : coin_set) : 
  ∃! s : finset ℕ, s.card = 12 ∧ ∀ x ∈ s, ∃ a b ∈ c, a ≠ b ∧ x = a + b := 
sorry

end coin_sums_unique_count_l381_381639


namespace part1_real_roots_part2_distinct_positive_integer_roots_l381_381823

noncomputable def equation := λ (m x : ℝ), m * x^2 - (m + 2) * x + 2 = 0

theorem part1_real_roots (m : ℝ) : ∃ x₁ x₂ : ℝ, equation m x₁ ∧ equation m x₂ := by
  sorry

theorem part2_distinct_positive_integer_roots (m : ℤ) : 
  (∃ x₁ x₂ : ℤ, x₁ ≠ x₂ ∧ x₁ > 0 ∧ x₂ > 0 ∧ equation m (x₁ : ℝ) ∧ equation m (x₂ : ℝ)) ↔ (m = 1) := by
  sorry

end part1_real_roots_part2_distinct_positive_integer_roots_l381_381823


namespace complement_union_M_N_eq_ge_2_l381_381956

def U := set.univ ℝ
def M := {x : ℝ | x < 1}
def N := {x : ℝ | -1 < x ∧ x < 2}

theorem complement_union_M_N_eq_ge_2 :
  (U \ (M ∪ N)) = {x : ℝ | 2 ≤ x} :=
by sorry

end complement_union_M_N_eq_ge_2_l381_381956


namespace product_zero_probability_l381_381100

def set : Finset ℤ := {-3, -2, -1, 0, 0, 2, 4, 5}

noncomputable def probability_product_zero : ℚ :=
  let total_ways := (Finset.card set).choose 2
  let favorable_ways := 6
  favorable_ways / total_ways
  
theorem product_zero_probability :
  probability_product_zero = 3 / 14 := by
  sorry

end product_zero_probability_l381_381100


namespace compute_expression_l381_381550

theorem compute_expression (x1 y1 x2 y2 x3 y3 : ℝ)
  (h1 : x1^3 - 3 * x1 * y1^2 = 2017)
  (h2 : y1^3 - 3 * x1^2 * y1 = 2016)
  (h3 : x2^3 - 3 * x2 * y2^2 = 2017)
  (h4 : y2^3 - 3 * x2^2 * y2 = 2016)
  (h5 : x3^3 - 3 * x3 * y3^2 = 2017)
  (h6 : y3^3 - 3 * x3^2 * y3 = 2016) :
  (2 - x1 / y1) * (2 - x2 / y2) * (2 - x3 / y3) = 26219 / 2016 := 
by
  sorry

end compute_expression_l381_381550


namespace external_angle_greater_than_internal_l381_381195

-- Define the angles of a triangle
variables {α β γ δ : ℝ}

-- The sum of the internal angles of a triangle
axiom sum_of_internal_angles (α β γ : ℝ) : α + β + γ = 180

-- The external angle is equal to the sum of the two non-adjacent angles
axiom external_angle_theorem (α β γ δ : ℝ) : δ = β + γ

-- The proof problem: Show that the external angle is greater than each internal non-adjacent angle.
theorem external_angle_greater_than_internal {α β γ δ : ℝ} 
  (h_sum : sum_of_internal_angles α β γ)
  (h_ext_angle : external_angle_theorem α β γ δ) : 
  (δ > β) ∧ (δ > γ) :=
sorry

end external_angle_greater_than_internal_l381_381195


namespace sin_neg_120_l381_381222

-- Define the angle in degrees
def deg_to_rad (d : ℝ) : ℝ := d * real.pi / 180

noncomputable def sin_deg (d : ℝ) : ℝ := real.sin (deg_to_rad d)

-- Main theorem
theorem sin_neg_120 :
  sin_deg (-120) = -real.sqrt 3 / 2 :=
by
  sorry

end sin_neg_120_l381_381222


namespace smallest_b_factors_l381_381301

theorem smallest_b_factors (b p q : ℤ) (H : p * q = 2016) : 
  (∀ k₁ k₂ : ℤ, k₁ * k₂ = 2016 → k₁ + k₂ ≥ p + q) → 
  b = 90 :=
by
  -- Here, we assume the premises stated for integers p, q such that their product is 2016.
  -- We need to fill in the proof steps which will involve checking all appropriate (p, q) pairs.
  sorry

end smallest_b_factors_l381_381301


namespace number_of_apples_and_erasers_l381_381478

def totalApplesAndErasers (a e : ℕ) : Prop :=
  a + e = 84

def applesPerFriend (a : ℕ) : ℕ :=
  a / 3

def erasersPerTeacher (e : ℕ) : ℕ :=
  e / 2

theorem number_of_apples_and_erasers (a e : ℕ) (h : totalApplesAndErasers a e) :
  applesPerFriend a = a / 3 ∧ erasersPerTeacher e = e / 2 :=
by
  sorry

end number_of_apples_and_erasers_l381_381478


namespace minimum_value_of_vector_expression_l381_381492

theorem minimum_value_of_vector_expression : 
  let a : ℝ × ℝ := (0, 3)
  let norm_a : ℝ := 3
  let norm_b : ℝ := 2
  let angle_ab : ℝ := real.pi / 3 -- 60 degrees in radians
  let dot_ab : ℝ := 3
  ∀ λ : ℝ, ∃ λ_min : ℝ, λ_min = 1/3 ∧ (λ_min * (norm_a)^2 - 2 * λ_min * dot_ab + norm_b^2 = 3) ∧
      ∀ λ' : ℝ, (λ' * (norm_a)^2 - 2 * λ' * dot_ab + norm_b^2) ≥ 3 :=
begin
  sorry
end

end minimum_value_of_vector_expression_l381_381492


namespace range_of_a_l381_381811

open Real

-- Define the function in question
def f (a x : ℝ) : ℝ := a * log x - (1/2) * x^2 + 6 * x

-- Define the derivative of the function
def f' (a x : ℝ) : ℝ := a / x - x + 6

-- State the theorem as required
theorem range_of_a (a : ℝ) (h : ∀ x > 0, f' a x ≤ 0) : a ∈ set.Iic (-9) := by
  sorry

end range_of_a_l381_381811


namespace inequality_proof_l381_381012

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  ( (2 * a + b + c) ^ 2 / (2 * a ^ 2 + (b + c) ^ 2) +
    (2 * b + a + c) ^ 2 / (2 * b ^ 2 + (c + a) ^ 2) +
    (2 * c + a + b) ^ 2 / (2 * c ^ 2 + (a + b) ^ 2)
  ) ≤ 8 := 
by
  sorry

end inequality_proof_l381_381012


namespace complement_union_eq_ge2_l381_381946

open Set

variables {U : Type} [PartialOrder U] [LinearOrder U]

def U : Set ℝ := univ
def M : Set ℝ := { x | x < 1 }
def N : Set ℝ := { x | -1 < x ∧ x < 2 }
def Complement_U (A : Set ℝ) : Set ℝ := U \ A

theorem complement_union_eq_ge2 : 
  Complement_U (M ∪ N) = { x : ℝ | x ≥ 2 } :=
by {
  sorry
}

end complement_union_eq_ge2_l381_381946


namespace range_of_a_is_one_to_infty_l381_381345

noncomputable def f (a x : ℝ) : ℝ := x^3 - 3*a*x + 1

theorem range_of_a_is_one_to_infty (a : ℝ) (h : 0 < a) :
  (∀ x ∈ Icc (0 : ℝ) 1, f a x ≤ f a (x + 1)) → (a ≥ 1) :=
by
  sorry

end range_of_a_is_one_to_infty_l381_381345


namespace telescoping_log_sum_l381_381239

theorem telescoping_log_sum :
  ∑ k in Finset.range 98 \ Finset.range 2, (Real.log (1 + 1 / k) / Real.log 3) * (Real.log 3 / Real.log k) * (Real.log 3 / Real.log (k + 1)) = 1 - 1 / Real.log 101 :=
by
  sorry

end telescoping_log_sum_l381_381239


namespace functional_equation_solution_l381_381281

theorem functional_equation_solution (f : ℝ → ℝ) (x : ℝ) (hx : x ≠ 0 ∧ x ≠ 1) :
  (f x = (1/2) * (x + 1 - 1/x - 1/(1-x))) →
  (f x + f (1 / (1 - x)) = x) :=
sorry

end functional_equation_solution_l381_381281


namespace num_pos_three_digit_div_by_seven_l381_381860

theorem num_pos_three_digit_div_by_seven : 
  ∃ n : ℕ, (∀ k : ℕ, k < n → (∃ m : ℕ, 100 ≤ 7 * m ∧ 7 * m ≤ 999)) ∧ n = 128 :=
by
  sorry

end num_pos_three_digit_div_by_seven_l381_381860


namespace fractions_order_l381_381597

theorem fractions_order :
  (21 / 17) < (18 / 13) ∧ (18 / 13) < (16 / 11) := by
  sorry

end fractions_order_l381_381597


namespace mans_speed_with_current_l381_381152

variable (v : ℝ) (current_speed : ℝ) (against_current_speed : ℝ)

theorem mans_speed_with_current
  (h_current_speed : current_speed = 3.2)
  (h_against_current_speed : against_current_speed = 8.6) :
  v - current_speed = against_current_speed → v + current_speed = 15 := by
  intro h
  rw [h_current_speed, h_against_current_speed] at h
  have h_v : v = 11.8 := by linarith
  rw [h_v, h_current_speed]
  norm_num

end mans_speed_with_current_l381_381152


namespace greatest_integer_y_l381_381573

theorem greatest_integer_y (y : ℤ) : (8 : ℚ) / 11 > y / 17 ↔ y ≤ 12 := 
sorry

end greatest_integer_y_l381_381573


namespace common_number_l381_381090

theorem common_number (a b c d e u v w : ℝ) (h1 : (a + b + c + d + e) / 5 = 7) 
                                            (h2 : (u + v + w) / 3 = 10) 
                                            (h3 : (a + b + c + d + e + u + v + w) / 8 = 8) 
                                            (h4 : a + b + c + d + e = 35) 
                                            (h5 : u + v + w = 30) 
                                            (h6 : a + b + c + d + e + u + v + w = 64) 
                                            (h7 : 35 + 30 = 65):
  d = u := 
by
  sorry

end common_number_l381_381090


namespace factor_of_quadratic_expression_l381_381873

def is_factor (a b : ℤ) : Prop := ∃ k, b = k * a

theorem factor_of_quadratic_expression (m : ℤ) :
  is_factor (m - 8) (m^2 - 5 * m - 24) :=
sorry

end factor_of_quadratic_expression_l381_381873


namespace exists_even_prime_not_prime_add_two_l381_381615

theorem exists_even_prime_not_prime_add_two : 
  ∃ n : ℕ, nat.prime n ∧ even n ∧ ¬ nat.prime (n + 2) :=
begin
  use 2,
  split,
  { exact nat.prime_two },
  split,
  { use 1, refl },
  { intro h,
    cases h with k hk,
    norm_num at hk }
end

end exists_even_prime_not_prime_add_two_l381_381615


namespace part_one_part_two_l381_381791

-- Part (1): Prove the solution set of the inequality f(x) < 0
def quadratic_function_inequality_solution_set (a b : ℝ) (x : ℝ): Prop :=
  a = -12 ∧ b = -2 ∧ 
    (λ x, a*x^2 + b*x + 2 < 0) x ↔ - (1:ℝ)/2 < x ∧ x < 1/3

-- Part (2): Prove the range of values for a         
def quadratic_function_no_solution_range (b : ℝ) : Set ℝ :=
  {a |  b = -1 ∧ 
     ∀ x, ¬ (λ x, a * x^2 - x + 2 < 0) x} 

def set_of_as (a : ℝ) : Prop := 
  a ≥ 1/8

theorem part_one: 
  ∀ x, quadratic_function_inequality_solution_set (-12) (-2) x := 
  sorry

theorem part_two: 
  quadratic_function_no_solution_range (-1) = set_of_as :=
  sorry

end part_one_part_two_l381_381791


namespace count_three_digit_numbers_divisible_by_7_l381_381848

def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

def is_divisible_by_7 (n : ℕ) : Prop :=
  n % 7 = 0

def count_three_digit_divisible_by_7 : ℕ :=
  (list.range' 105 890).countp (λ x, is_divisible_by_7 (x * 7))

theorem count_three_digit_numbers_divisible_by_7 :
  count_three_digit_divisible_by_7 = 128 :=
sorry

end count_three_digit_numbers_divisible_by_7_l381_381848


namespace parallogram_fourth_vertex_l381_381095

theorem parallogram_fourth_vertex (a b c d : ℝ) : 
  ∃ x y : ℝ, 
    (x, y) = (a + c, b + d) :=
by
  use a + c
  use b + d
  sorry

end parallogram_fourth_vertex_l381_381095


namespace complement_union_eq_ge_two_l381_381996

def U : Set ℝ := Set.univ
def M : Set ℝ := { x : ℝ | x < 1 }
def N : Set ℝ := { x : ℝ | -1 < x ∧ x < 2 }

theorem complement_union_eq_ge_two : { x : ℝ | x ≥ 2 } = U \ (M ∪ N) :=
by
  sorry

end complement_union_eq_ge_two_l381_381996


namespace problem_statement_l381_381975

open Set

variable (U : Set ℝ) (M N : Set ℝ)

theorem problem_statement (hU : U = univ) (hM : M = {x | x < 1}) (hN : N = {x | -1 < x ∧ x < 2}) :
  {x | 2 ≤ x} = compl (M ∪ N) :=
sorry

end problem_statement_l381_381975


namespace sum_of_first_110_terms_l381_381076

theorem sum_of_first_110_terms
  (a d : ℝ)
  (h1 : (10 : ℝ) * (2 * a + (10 - 1) * d) / 2 = 100)
  (h2 : (100 : ℝ) * (2 * a + (100 - 1) * d) / 2 = 10) :
  (110 : ℝ) * (2 * a + (110 - 1) * d) / 2 = -110 :=
  sorry

end sum_of_first_110_terms_l381_381076


namespace sin_ratios_l381_381336

noncomputable def question (a b c : ℝ) (OA OB OC : ℝ) (h₁ : 2 * a * OA + b * OB + (2 * real.sqrt 3 / 3) * c * OC = 0)
  (h₂ : OA + OB + OC = 0) : Prop :=
  (real.sin (OA * b / (2 * a)) / real.sin (OA * b / (2 * a))) = (1 : 2 : real.sqrt 3)

-- To define the Theorem
theorem sin_ratios (a b c : ℝ) (OA OB OC : ℝ) (h₁ : 2 * a * OA + b * OB + (2 * real.sqrt 3 / 3) * c * OC = 0)
  (h₂ : OA + OB + OC = 0) : question a b c OA OB OC h₁ h₂ :=
  sorry

end sin_ratios_l381_381336


namespace calf_grazing_area_l381_381162

noncomputable def initial_rope_length (A : ℝ) (r_final : ℝ) : ℝ :=
  let area_difference := A + π * r_final^2
  let r_initial_squared := area_difference / π
  real.sqrt r_initial_squared

theorem calf_grazing_area :
  initial_rope_length 1210 23 ≈ 12 := 
  sorry

end calf_grazing_area_l381_381162


namespace euler_identity_magnitude_l381_381706

theorem euler_identity_magnitude :
  abs (complex.exp (complex.I * (π / 3)) + complex.exp (complex.I * (5 * π / 6))) = sqrt 2 := sorry

end euler_identity_magnitude_l381_381706


namespace complement_union_eq_ge_two_l381_381995

def U : Set ℝ := Set.univ
def M : Set ℝ := { x : ℝ | x < 1 }
def N : Set ℝ := { x : ℝ | -1 < x ∧ x < 2 }

theorem complement_union_eq_ge_two : { x : ℝ | x ≥ 2 } = U \ (M ∪ N) :=
by
  sorry

end complement_union_eq_ge_two_l381_381995


namespace limit_derivative_at_1_l381_381826

def f (x : ℝ) : ℝ := sqrt x + 1

theorem limit_derivative_at_1 : 
  (tendsto (fun Δx => (f (1 + Δx) - f 1) / Δx) (𝓝 0) (𝓝 (1 / 2))) :=
by
  sorry  -- Proof goes here

end limit_derivative_at_1_l381_381826


namespace root_of_quadratic_one_of_them_is_four_l381_381663

theorem root_of_quadratic_one_of_them_is_four
  (k : ℝ) (h_k : k = 44) : (∃ x : ℝ, 2 * x^2 + 3 * x - k = 0) :=
by
  have h_eq : 2 * (4 : ℝ)^2 + 3 * 4 - k = 0,
  { calc
      2 * (16 : ℝ) + 3 * 4 - k = 2 * 16 + 12 - k : rfl
      ... = 44 - k : by norm_num
      ... = 44 - 44 : by rw h_k
      ... = 0 : by norm_num, },
  use 4,
  exact h_eq,
  sorry

end root_of_quadratic_one_of_them_is_four_l381_381663


namespace solution_system_l381_381867

theorem solution_system (x y : ℝ) (h1 : x * y = 8) (h2 : x^2 * y + x * y^2 + x + y = 80) :
  x^2 + y^2 = 5104 / 81 := by
  sorry

end solution_system_l381_381867


namespace factor_expression_l381_381269

-- Define f(x, y, z) based on the given expression
def f (x y z : ℝ) : ℝ :=
  (y^2 - z^2) * (1 + x * y) * (1 + x * z) +
  (z^2 - x^2) * (1 + y * z) * (1 + x * y) +
  (x^2 - y^2) * (1 + y * z) * (1 + x * z)

-- Define the factored form of the expression
def factored_f (x y z : ℝ) : ℝ :=
  (y - z) * (z - x) * (x - y) * (x * y * z + x + y + z)

-- Prove that f(x, y, z) is equivalent to its factored form
theorem factor_expression (x y z : ℝ) : f(x, y, z) = factored_f(x, y, z) :=
  sorry

end factor_expression_l381_381269


namespace parallelogram_area_twice_quadrilateral_area_l381_381493

theorem parallelogram_area_twice_quadrilateral_area (S : ℝ) (LMNP_area : ℝ) 
  (h : LMNP_area = 2 * S) : LMNP_area = 2 * S := 
by {
  sorry
}

end parallelogram_area_twice_quadrilateral_area_l381_381493


namespace dress_assignment_l381_381766

-- Define the four girls
inductive Girl
| Katya
| Olya
| Liza
| Rita

-- Define the four dresses
inductive Dress
| Pink
| Green
| Yellow
| Blue

-- Define the function that assigns each girl a dress
def dressOf : Girl → Dress

-- Conditions as definitions
axiom KatyaNotPink : dressOf Girl.Katya ≠ Dress.Pink
axiom KatyaNotBlue : dressOf Girl.Katya ≠ Dress.Blue
axiom RitaNotGreen : dressOf Girl.Rita ≠ Dress.Green
axiom RitaNotBlue : dressOf Girl.Rita ≠ Dress.Blue

axiom GreenBetweenLizaAndYellow : 
  ∃ (arrangement : List Girl), 
    arrangement = [Girl.Liza, Girl.Katya, Girl.Rita] ∧ 
    (dressOf Girl.Katya = Dress.Green ∧ 
    dressOf Girl.Liza = Dress.Pink ∧ 
    dressOf Girl.Rita = Dress.Yellow)

axiom OlyaBetweenRitaAndPink : 
  ∃ (arrangement : List Girl),
    arrangement = [Girl.Rita, Girl.Olya, Girl.Liza] ∧ 
    (dressOf Girl.Olya = Dress.Blue ∧ 
     dressOf Girl.Rita = Dress.Yellow ∧ 
     dressOf Girl.Liza = Dress.Pink)

-- Problem: Determine the dress assignments
theorem dress_assignment : 
  dressOf Girl.Katya = Dress.Green ∧ 
  dressOf Girl.Olya = Dress.Blue ∧ 
  dressOf Girl.Liza = Dress.Pink ∧ 
  dressOf Girl.Rita = Dress.Yellow :=
sorry

end dress_assignment_l381_381766


namespace mr_jones_loss_l381_381457

theorem mr_jones_loss :
  ∃ (C_1 C_2 : ℝ), 
    (1.2 = 1.2 * C_1 / 1.2) ∧ 
    (1.2 = 0.8 * C_2) ∧ 
    ((C_1 + C_2) - (2 * 1.2)) = -0.1 :=
by
  sorry

end mr_jones_loss_l381_381457


namespace actual_vs_expected_increase_rate_l381_381600

noncomputable def expected_increase_rate := by
  let x_sol := sorry -- Solve (1 + x)^4 = 2 for x
  exact x_sol

theorem actual_vs_expected_increase_rate :
  (0.5 - expected_increase_rate) = 0.086 := by
  sorry

end actual_vs_expected_increase_rate_l381_381600


namespace problem_equivalent_l381_381941

def U : Set ℝ := Set.univ
def M : Set ℝ := {x | x < 1}
def N : Set ℝ := {x | -1 < x ∧ x < 2}

theorem problem_equivalent :
  {x : ℝ | x ≥ 2} = (U \ (M ∪ N)) := 
by sorry

end problem_equivalent_l381_381941


namespace complement_union_eq_ge2_l381_381950

open Set

variables {U : Type} [PartialOrder U] [LinearOrder U]

def U : Set ℝ := univ
def M : Set ℝ := { x | x < 1 }
def N : Set ℝ := { x | -1 < x ∧ x < 2 }
def Complement_U (A : Set ℝ) : Set ℝ := U \ A

theorem complement_union_eq_ge2 : 
  Complement_U (M ∪ N) = { x : ℝ | x ≥ 2 } :=
by {
  sorry
}

end complement_union_eq_ge2_l381_381950


namespace find_x_l381_381712

open Real

theorem find_x (x : ℝ) (h : log 10 (5 * x) = 2) : x = 20 :=
by
  sorry

end find_x_l381_381712


namespace integral_converges_if_and_only_if_alpha_eq_beta_l381_381726

open Real

theorem integral_converges_if_and_only_if_alpha_eq_beta
  (α β : ℝ) (hα : 0 < α) (hβ : 0 < β) :
  (∫ x in (β : ℝ)..∞, sqrt (sqrt (x + α) - sqrt x) - sqrt (sqrt x - sqrt (x - β))) < ∞ ↔ α = β := sorry

end integral_converges_if_and_only_if_alpha_eq_beta_l381_381726


namespace compute_g_five_times_l381_381925

def g (x : ℤ) : ℤ :=
  if x ≥ 0 then - x^3 else x + 10

theorem compute_g_five_times (x : ℤ) (h : x = 2) : g (g (g (g (g x)))) = -8 := by
  sorry

end compute_g_five_times_l381_381925


namespace intersection_is_correct_l381_381833

def A : Set ℝ := {x | True}
def B : Set ℝ := {y | y ≥ 0}

theorem intersection_is_correct : A ∩ B = { x | x ≥ 0 } :=
by
  sorry

end intersection_is_correct_l381_381833


namespace jo_vs_lola_sum_difference_l381_381921

noncomputable def jo_sum : ℕ := (200 * (200 + 1)) / 2

def nearest_multiple_of_five (n : ℕ) : ℕ :=
  let r := n % 5
  if r = 0 then n else if r < 3 then n - r else n + (5 - r)

def lola_sum : ℕ := (List.ofFn (λ n => nearest_multiple_of_five (n + 1))).sum

theorem jo_vs_lola_sum_difference :
  abs (jo_sum - lola_sum) = 19000 :=
by
  let jo_sum := 20100
  let lola_sum := (List.range 200).sum (λ n => nearest_multiple_of_five (n + 1))
  have jo_sum_correct : jo_sum = 20100 := by sorry
  have lola_sum_correct : lola_sum = 1100 := by sorry
  rw [jo_sum_correct, lola_sum_correct]
  norm_num

end jo_vs_lola_sum_difference_l381_381921


namespace propositions_p_q_l381_381800

theorem propositions_p_q
  (p q : Prop)
  (h : ¬(p ∧ q) = False) : p ∧ q :=
by
  sorry

end propositions_p_q_l381_381800


namespace greatest_y_least_y_greatest_integer_y_l381_381588

theorem greatest_y (y : ℤ) (H : (8 : ℝ) / 11 > y / 17) : y ≤ 12 :=
sorry

theorem least_y (y : ℤ) (H : (8 : ℝ) / 11 > y / 17) : y ≥ 12 :=
sorry

theorem greatest_integer_y : ∀ (y : ℤ), ((8 : ℝ) / 11 > y / 17) → y = 12 :=
by
  intro y H
  apply le_antisymm
  apply greatest_y y H
  apply least_y y H

end greatest_y_least_y_greatest_integer_y_l381_381588


namespace max_area_region_T_l381_381392

noncomputable def maximum_area (r1 r2 r3 r4 : ℝ) (h1 : r1 = 2) (h2 : r2 = 4) (h3 : r3 = 6) (h4 : r4 = 8) : ℝ :=
  let area1 := π * r1^2
  let area2 := π * r2^2
  let area3 := π * r3^2
  let area4 := π * r4^2
  area1 + area3

theorem max_area_region_T : ∀ (r1 r2 r3 r4 : ℝ),
  r1 = 2 → r2 = 4 → r3 = 6 → r4 = 8 →
  maximum_area r1 r2 r3 r4 = 100 * π :=
by
  intros r1 r2 r3 r4 h1 h2 h3 h4
  have area1 : π * r1^2 = 4 * π := by rw [h1] ; norm_num
  have area2 : π * r2^2 = 16 * π := by rw [h2] ; norm_num
  have area3 : π * r3^2 = 36 * π := by rw [h3] ; norm_num
  have area4 : π * r4^2 = 64 * π := by rw [h4] ; norm_num
  have total_area := area4 + area3
  rw [area4, area3] at total_area
  norm_num at total_area
  exact total_area
  sorry

end max_area_region_T_l381_381392


namespace functional_equation_solution_l381_381278

theorem functional_equation_solution :
  ∀ (f : ℝ → ℝ),
  (∀ x : ℝ, x ≠ 0 ∧ x ≠ 1 → f(x) + f(1 / (1 - x)) = x) →
  (∀ x : ℝ, x ≠ 0 ∧ x ≠ 1 → f(x) = 1 / 2 * (x + 1 - 1 / x - 1 / (1 - x))) :=
by
  intros f h x hx,
  sorry

end functional_equation_solution_l381_381278


namespace sum_roots_equation_l381_381679

theorem sum_roots_equation :
  let f := λ x : ℝ, (3 * x + 5) * (x - 8) + (3 * x + 5) * (x - 7)
  (root1 root2 : ℝ), f root1 = 0 ∧ f root2 = 0 →
  root1 + root2 = 35 / 6 :=
by
  let f := λ x : ℝ, (3 * x + 5) * (x - 8) + (3 * x + 5) * (x - 7)
  have h1: f (-5 / 3) = 0 := sorry
  have h2: f (15 / 2) = 0 := sorry
  exact sorry

end sum_roots_equation_l381_381679


namespace original_area_ratio_l381_381886

theorem original_area_ratio (s : ℝ) : 
  let A_original := s^2 in
  let new_side := s^3 * π^(1/3) in
  let A_resultant := (new_side)^2 in
  A_original = (s^4 * π^(2/3)) * A_resultant :=
by sorry

end original_area_ratio_l381_381886


namespace sin_neg_120_eq_sqrt_3_over_2_l381_381219

noncomputable def sin_neg_angle (θ : ℝ) : ℝ := -Real.sin θ

theorem sin_neg_120_eq_sqrt_3_over_2 :
  sin_neg_angle (120 * Real.pi / 180) = Real.sqrt 3 / 2 :=
by
  -- Use the identity sin(-θ) = -sin(θ)
  have h1 : sin_neg_angle (120 * Real.pi / 180) = -Real.sin (120 * Real.pi / 180) := rfl
  
  -- Simplify and use pre-defined instances for sin(120 degrees)
  have h2 : Real.sin (120 * Real.pi / 180) = Real.sin (2 * Real.pi / 3) := by 
    norm_num

  -- Calculate the sin value for 2π/3
  have h3 : Real.sin (2 * Real.pi / 3) = Real.sin (Real.pi - Real.pi / 3) := by 
    norm_num [Real.sin_pi_sub_div]
  
  -- Which further simplifies to
  have h4 : Real.sin (Real.pi - Real.pi / 3) = Real.sin (π/3) := by
    norm_num
  
  -- Since sin(π/3) = sqrt(3)/2
  have h5 : Real.sin (π/3) = Real.sqrt 3 / 2 := by
    norm_num
  
  -- Applying all above results
  rw [h1, h2, h3, h4, h5]
  norm_num

  -- Concluding the proof
  exact sorry

end sin_neg_120_eq_sqrt_3_over_2_l381_381219


namespace dress_assignments_l381_381742

structure GirlDress : Type :=
  (Katya Olya Liza Rita : String)

def dresses := ["Pink", "Green", "Yellow", "Blue"]

axiom not_pink_or_blue : GirlDress.Katya ≠ "Pink" ∧ GirlDress.Katya ≠ "Blue"
axiom green_between_liza_yellow : (GirlDress.Liza = "Pink" ∨ GirlDress.Rita = "Yellow") ∧
                                  GirlDress.Katya = "Green" ∧
                                  GirlDress.Rita = "Yellow" ∧ GirlDress.Liza = "Pink"
axiom not_green_or_blue : GirlDress.Rita ≠ "Green" ∧ GirlDress.Rita ≠ "Blue"
axiom olya_between_rita_pink : GirlDress.Olya ≠ "Pink" → GirlDress.Rita ≠ "Pink" → GirlDress.Liza = "Pink"

theorem dress_assignments (gd : GirlDress) :
  gd.Katya = "Green" ∧ gd.Olya = "Blue" ∧ gd.Liza = "Pink" ∧ gd.Rita = "Yellow" :=
by
  sorry

end dress_assignments_l381_381742


namespace length_of_cube_side_l381_381500

noncomputable def paint_cost_per_kg : ℝ := 60
noncomputable def coverage_per_kg : ℝ := 20
noncomputable def total_painting_cost : ℝ := 1800

theorem length_of_cube_side :
  let kg_paint_used := total_painting_cost / paint_cost_per_kg in
  let total_area_covered := kg_paint_used * coverage_per_kg in
  let area_per_side := total_area_covered / 6 in 
  let side_length := real.sqrt area_per_side in
  side_length = 10 :=
by
  sorry

end length_of_cube_side_l381_381500


namespace dress_assignment_l381_381761

theorem dress_assignment :
  ∃ (Katya Olya Liza Rita : string),
    (Katya ≠ "Pink" ∧ Katya ≠ "Blue") ∧
    (Rita ≠ "Green" ∧ Rita ≠ "Blue") ∧
    ∃ (girl_in_green girl_in_yellow : string),
      (girl_in_green = Katya ∧ girl_in_yellow = Rita ∧ 
       (Liza = "Pink" ∧ Olya = "Blue") ∧
       (Katya = "Green" ∧ Olya = "Blue" ∧ Liza = "Pink" ∧ Rita = "Yellow")) ∧
    ((girl_in_green stands between Liza and girl_in_yellow) ∧
     (Olya stands between Rita and Liza)) :=
by
  sorry

end dress_assignment_l381_381761


namespace problem_l381_381362

noncomputable def f (x : ℝ) := cos x

theorem problem 
  (ω φ α β : ℝ) 
  (hω : ω > 0) 
  (hφ : π/3 < φ ∧ φ < π) 
  (hM : f (π / 6) = sqrt (3) / 2) 
  (hα : α ∈ set.Ioo 0 (π/2)) 
  (hβ : β ∈ set.Ioo 0 (π/2)) 
  (hfα : f α = 3 / 5) 
  (hfβ : f β = 12 / 13) :
  f = λ x, cos x ∧ f (2 * α - β) = 36 / 325 := 
by 
  sorry

end problem_l381_381362


namespace cubic_polynomial_root_type_l381_381305

noncomputable def cubic_roots : List Real :=
  [-2, 1, 3]

theorem cubic_polynomial_root_type :
  (x^3 - 2*x^2 - 5*x + 6 = 0) ∧
  (count_neg_real_roots cubic_roots = 1) ∧
  (count_pos_real_roots cubic_roots = 2) :=
by
  sorry

end cubic_polynomial_root_type_l381_381305


namespace sin_neg_120_l381_381221

-- Define the angle in degrees
def deg_to_rad (d : ℝ) : ℝ := d * real.pi / 180

noncomputable def sin_deg (d : ℝ) : ℝ := real.sin (deg_to_rad d)

-- Main theorem
theorem sin_neg_120 :
  sin_deg (-120) = -real.sqrt 3 / 2 :=
by
  sorry

end sin_neg_120_l381_381221


namespace problem_equivalent_l381_381937

def U : Set ℝ := Set.univ
def M : Set ℝ := {x | x < 1}
def N : Set ℝ := {x | -1 < x ∧ x < 2}

theorem problem_equivalent :
  {x : ℝ | x ≥ 2} = (U \ (M ∪ N)) := 
by sorry

end problem_equivalent_l381_381937


namespace fraction_time_at_0_is_one_third_l381_381636

noncomputable def fraction_time_at_0 (heads_prob tails_prob edge_prob : ℚ) : ℚ :=
let a₀ : ℚ := 1/3 in
a₀

theorem fraction_time_at_0_is_one_third :
  fraction_time_at_0 (2/5) (2/5) (1/5) = 1/3 :=
by sorry

end fraction_time_at_0_is_one_third_l381_381636


namespace who_wears_which_dress_l381_381734

-- Define the possible girls
inductive Girl
| Katya | Olya | Liza | Rita
deriving DecidableEq

-- Define the possible dresses
inductive Dress
| Pink | Green | Yellow | Blue
deriving DecidableEq

-- Define the fact that each girl is wearing a dress
structure Wearing (girl : Girl) (dress : Dress) : Prop

-- Define the conditions
theorem who_wears_which_dress :
  (¬ Wearing Girl.Katya Dress.Pink ∧ ¬ Wearing Girl.Katya Dress.Blue) ∧
  (∀ g1 g2 g3, Wearing g1 Dress.Green → (Wearing g2 Dress.Pink ∧ Wearing g3 Dress.Yellow → (g2 = Girl.Liza ∧ (g3 = Girl.Rita)) ∨ (g3 = Girl.Liza ∧ g2 = Girl.Rita))) ∧
  (¬ Wearing Girl.Rita Dress.Green ∧ ¬ Wearing Girl.Rita Dress.Blue) ∧
  (∀ g1 g2, (Wearing g1 Dress.Pink ∧ Wearing g2 Dress.Yellow) → Girl.Olya = g2 ∧ Girl.Rita = g1) →
  (Wearing Girl.Katya Dress.Green ∧ Wearing Girl.Olya Dress.Blue ∧ Wearing Girl.Liza Dress.Pink ∧ Wearing Girl.Rita Dress.Yellow) :=
by
  sorry

end who_wears_which_dress_l381_381734


namespace who_wears_which_dress_l381_381729

-- Define the possible girls
inductive Girl
| Katya | Olya | Liza | Rita
deriving DecidableEq

-- Define the possible dresses
inductive Dress
| Pink | Green | Yellow | Blue
deriving DecidableEq

-- Define the fact that each girl is wearing a dress
structure Wearing (girl : Girl) (dress : Dress) : Prop

-- Define the conditions
theorem who_wears_which_dress :
  (¬ Wearing Girl.Katya Dress.Pink ∧ ¬ Wearing Girl.Katya Dress.Blue) ∧
  (∀ g1 g2 g3, Wearing g1 Dress.Green → (Wearing g2 Dress.Pink ∧ Wearing g3 Dress.Yellow → (g2 = Girl.Liza ∧ (g3 = Girl.Rita)) ∨ (g3 = Girl.Liza ∧ g2 = Girl.Rita))) ∧
  (¬ Wearing Girl.Rita Dress.Green ∧ ¬ Wearing Girl.Rita Dress.Blue) ∧
  (∀ g1 g2, (Wearing g1 Dress.Pink ∧ Wearing g2 Dress.Yellow) → Girl.Olya = g2 ∧ Girl.Rita = g1) →
  (Wearing Girl.Katya Dress.Green ∧ Wearing Girl.Olya Dress.Blue ∧ Wearing Girl.Liza Dress.Pink ∧ Wearing Girl.Rita Dress.Yellow) :=
by
  sorry

end who_wears_which_dress_l381_381729


namespace greatest_integer_l381_381579

theorem greatest_integer (y : ℤ) : (8 / 11 : ℝ) > (y / 17 : ℝ) → y ≤ 12 :=
by sorry

end greatest_integer_l381_381579


namespace emani_more_than_howard_l381_381705

variable Emani_money : ℕ := 150
variable Howard_money : ℕ

theorem emani_more_than_howard :
  150 + Howard_money = 270 →
  150 - Howard_money = 30 :=
by
  assume h1 : 150 + Howard_money = 270
  rw [add_comm] at h1
  sorry

end emani_more_than_howard_l381_381705


namespace roots_eq_s_l381_381007

theorem roots_eq_s (n c d : ℝ) (h₁ : c * d = 6) (h₂ : c + d = n)
  (h₃ : c^2 + 1 / d = c^2 + d^2 + 1 / c): 
  (n + 217 / 6) = d^2 + 1/ c * (n + c + d)
  :=
by
  -- The proof will go here
  sorry

end roots_eq_s_l381_381007


namespace fraction_A_B_l381_381261

noncomputable def A : ℝ := ∑' n, if (odd n ∧ ¬ (4 ∣ n)) then 1 / (n ^ 2) else 0
noncomputable def B : ℝ := ∑' n, if (odd n ∧ (4 ∣ n)) then 1 / (n ^ 2) else 0

theorem fraction_A_B : (A / B) = 17 := 
by
  sorry

end fraction_A_B_l381_381261


namespace number_of_valid_five_digit_numbers_l381_381259

-- Define the conditions
def valid_five_digit_number (n : ℕ) : Prop :=
  let digits := [0, 1, 2, 3, 4] in
  let num_digits := (n.toString.length = 5) in
  let unique_digits := (n.digits.toFinset.card = 5) in
  let odd_even_adjacent := ∀ i, (i ∈ [0, 2] → (i+1) ∈ [1, 3]) ∧ (i ∈ [1, 3] → (i-1) ∈ [0, 2]) in
  let non_adj_zeros := ∀ j, ¬(n.digits.get? j = some 0 ∧ j = 0) in
  num_digits ∧ unique_digits ∧ odd_even_adjacent ∧ non_adj_zeros

-- Define the theorem
theorem number_of_valid_five_digit_numbers : 
  ∃ k : ℕ, (k = 36) ∧ ∀ n : ℕ, valid_five_digit_number n → 
    (n ∈ {m // valid_five_digit_number m}.card = 36) :=
by {
  sorry
}

end number_of_valid_five_digit_numbers_l381_381259


namespace tangent_line_eqn_at_y_intercept_l381_381820

noncomputable def f (x : ℝ) : ℝ := -exp (x + 1)

def P : ℝ × ℝ := (0, -exp 1)

theorem tangent_line_eqn_at_y_intercept :
  ∃ (m b : ℝ), (∀ x, f x = -exp (x + 1)) ∧ P = (0, -exp 1) ∧
  m = -exp (0 + 1) ∧ b = -exp 1 ∧
  (∀ x y, y = m * x + b ↔ y = -exp (0 + 1) * x - exp 1) := 
by
  sorry

end tangent_line_eqn_at_y_intercept_l381_381820


namespace problem_equivalent_l381_381936

def U : Set ℝ := Set.univ
def M : Set ℝ := {x | x < 1}
def N : Set ℝ := {x | -1 < x ∧ x < 2}

theorem problem_equivalent :
  {x : ℝ | x ≥ 2} = (U \ (M ∪ N)) := 
by sorry

end problem_equivalent_l381_381936


namespace rooks_control_chosen_squares_l381_381893

theorem rooks_control_chosen_squares (n : Nat) 
  (chessboard : Fin (2 * n) × Fin (2 * n)) 
  (chosen_squares : Finset (Fin (2 * n) × Fin (2 * n))) 
  (h : chosen_squares.card = 3 * n) :
  ∃ rooks : Finset (Fin (2 * n) × Fin (2 * n)), rooks.card = n ∧
  ∀ (square : Fin (2 * n) × Fin (2 * n)), square ∈ chosen_squares → 
  (square ∈ rooks ∨ ∃ (rook : Fin (2 * n) × Fin (2 * n)) (hr : rook ∈ rooks), 
  rook.1 = square.1 ∨ rook.2 = square.2) :=
sorry

end rooks_control_chosen_squares_l381_381893


namespace minimizing_expression_l381_381274

theorem minimizing_expression : 
  (∃ y ∈ ℝ, ∀ z ∈ ℝ, z = (2 - 4 * Real.sqrt 2) → 
    (∀ x, 0 ≤ x ∧ x ≤ 2 → |x^2 - x * y| ≤ z)) :=
begin
  sorry
end

end minimizing_expression_l381_381274


namespace parabola_equation_l381_381352

open Classical

noncomputable def circle_center : ℝ × ℝ := (2, 0)

theorem parabola_equation (vertex : ℝ × ℝ) (focus : ℝ × ℝ) :
  vertex = (0, 0) ∧ focus = circle_center → ∀ x y : ℝ, y^2 = 8 * x := by
  intro h
  sorry

end parabola_equation_l381_381352


namespace total_students_l381_381919

theorem total_students (h1 : ∀ (n : ℕ), n = 5 → Jaya_ranks_nth_from_top)
                       (h2 : ∀ (m : ℕ), m = 49 → Jaya_ranks_mth_from_bottom) :
  ∃ (total : ℕ), total = 53 :=
by
  sorry

end total_students_l381_381919


namespace rhombus_inscribed_circle_radius_l381_381107

theorem rhombus_inscribed_circle_radius (d1 d2 p : ℝ) (h_d1 : d1 = 18) (h_d2 : d2 = 30) (h_p : p = 68) :
  let a := (p / 4) in
  let area := (d1 * d2) / 2 in
  let r := area / (2 * a) in
  r = 135 / 17 :=
by
  sorry

end rhombus_inscribed_circle_radius_l381_381107


namespace modulus_of_complex_number_l381_381378

theorem modulus_of_complex_number (z : ℂ) (h : z = 2 / (1 - I)) : |z| = Real.sqrt 2 := by
  sorry

end modulus_of_complex_number_l381_381378


namespace smallest_b_for_factorization_l381_381299

-- Let us state the problem conditions and the objective
theorem smallest_b_for_factorization :
  ∃ (b : ℕ), b = 92 ∧ ∃ (p q : ℤ), (p + q = b) ∧ (p * q = 2016) :=
begin
  sorry
end

end smallest_b_for_factorization_l381_381299


namespace find_ordered_triple_l381_381334

theorem find_ordered_triple (a b c : ℝ)
  (h : ⟨a, b, c⟩ × ⟨5, 3, 8⟩ = ⟨-15, -26, 15⟩) :
  a = -11 / 16 ∧ b = 207 / 80 ∧ c = 3.88 :=
sorry

end find_ordered_triple_l381_381334


namespace bugs_eat_flowers_l381_381458

-- Define the problem conditions
def number_of_bugs : ℕ := 3
def flowers_per_bug : ℕ := 2

-- Define the expected outcome
def total_flowers_eaten : ℕ := 6

-- Prove that total flowers eaten is equal to the product of the number of bugs and flowers per bug
theorem bugs_eat_flowers : number_of_bugs * flowers_per_bug = total_flowers_eaten :=
by
  sorry

end bugs_eat_flowers_l381_381458


namespace reflective_beam_time_l381_381514

theorem reflective_beam_time :
  let BE := 2
  let EF := 2
  let FC := 2
  let speed := 1
  in (the beam must ricochet exactly once off the sides CD, AD, and AB in that order) →
     (BE = 2 ∧ EF = 2 ∧ FC = 2 ∧ speed = 1) →
     (time elapsed between firing from F and arriving at E) = 2 * Real.sqrt 61 := by
  sorry

end reflective_beam_time_l381_381514


namespace find_angle_y_l381_381409

-- Definitions for conditions
def lines_parallel (m n : Type) : Prop := ∀(x : Type), x ⟂ m ↔ x ⟂ n
def intersects_at (x m n : Type) (angle_m angle_n : ℕ) : Prop :=
  angle_m = 30 ∧ angle_n = 40
def angle_between (angle : ℕ) : Type → Type := 
  λ (line : Type), angle

-- The math problem translated to Lean 4 statement
theorem find_angle_y 
  (m n intersecting_line perp_line : Type)
  (H_parallel : lines_parallel m n)
  (H_intersect : intersects_at intersecting_line m n 30 40)
  (H_perp : perp_line ⟂ n)
  : angle_between 50 perp_line intersecting_line :=
sorry

end find_angle_y_l381_381409


namespace right_triangle_area_l381_381901

theorem right_triangle_area
  (a b c : ℝ)
  (h1 : a + b = 14)
  (h2 : c = 10)
  (h3 : a^2 + b^2 = c^2):
  area := (1 / 2) * a * b
  area = 24 := sorry

end right_triangle_area_l381_381901


namespace who_wears_which_dress_l381_381728

-- Define the possible girls
inductive Girl
| Katya | Olya | Liza | Rita
deriving DecidableEq

-- Define the possible dresses
inductive Dress
| Pink | Green | Yellow | Blue
deriving DecidableEq

-- Define the fact that each girl is wearing a dress
structure Wearing (girl : Girl) (dress : Dress) : Prop

-- Define the conditions
theorem who_wears_which_dress :
  (¬ Wearing Girl.Katya Dress.Pink ∧ ¬ Wearing Girl.Katya Dress.Blue) ∧
  (∀ g1 g2 g3, Wearing g1 Dress.Green → (Wearing g2 Dress.Pink ∧ Wearing g3 Dress.Yellow → (g2 = Girl.Liza ∧ (g3 = Girl.Rita)) ∨ (g3 = Girl.Liza ∧ g2 = Girl.Rita))) ∧
  (¬ Wearing Girl.Rita Dress.Green ∧ ¬ Wearing Girl.Rita Dress.Blue) ∧
  (∀ g1 g2, (Wearing g1 Dress.Pink ∧ Wearing g2 Dress.Yellow) → Girl.Olya = g2 ∧ Girl.Rita = g1) →
  (Wearing Girl.Katya Dress.Green ∧ Wearing Girl.Olya Dress.Blue ∧ Wearing Girl.Liza Dress.Pink ∧ Wearing Girl.Rita Dress.Yellow) :=
by
  sorry

end who_wears_which_dress_l381_381728


namespace avery_egg_cartons_filled_l381_381198

-- Definitions (conditions identified in step a)
def total_chickens : ℕ := 20
def eggs_per_chicken : ℕ := 6
def eggs_per_carton : ℕ := 12

-- Theorem statement (equivalent to the problem statement)
theorem avery_egg_cartons_filled : (total_chickens * eggs_per_chicken) / eggs_per_carton = 10 :=
by
  -- Proof omitted; sorry used to denote unfinished proof
  sorry

end avery_egg_cartons_filled_l381_381198


namespace max_students_per_class_l381_381542

theorem max_students_per_class
    (total_students : ℕ)
    (total_classes : ℕ)
    (bus_count : ℕ)
    (bus_seats : ℕ)
    (students_per_class : ℕ)
    (total_students = 920)
    (bus_count = 16)
    (bus_seats = 71)
    (∀ c < total_classes, students_per_class ≤ bus_seats) : 
    students_per_class ≤ 17 := 
by
    sorry

end max_students_per_class_l381_381542


namespace original_fraction_l381_381522

def fraction (a b c : ℕ) := 10 * a + b / 10 * c + a

theorem original_fraction (a b c : ℕ) (ha: a < 10) (hb : b < 10) (hc : c < 10) (h : b ≠ c):
  (fraction a b c = b / c) →
  (fraction 6 4 1 = 64 / 16) ∨ (fraction 9 8 4 = 98 / 49) ∨
  (fraction 9 5 1 = 95 / 19) ∨ (fraction 6 5 2 = 65 / 26) :=
sorry

end original_fraction_l381_381522


namespace math_problem_l381_381432

theorem math_problem
  (g : ℝ → ℝ)
  (h : ∀ x y : ℝ, g (g x + y) = g (x^2 + y) + 4 * g x * y) :
  let n := 2,
      s := 16 in
  n * s = 32 := 
by
  -- Sorry is placed here to indicate incomplete proof
  sorry

end math_problem_l381_381432


namespace f_decreasing_on_0_1_l381_381033

noncomputable def f (x : ℝ) : ℝ := x^2 + 2 * x⁻¹

theorem f_decreasing_on_0_1 : ∀ x₁ x₂ : ℝ, x₁ ∈ Set.Icc 0 1 → x₂ ∈ Set.Icc 0 1 → x₁ < x₂ → f x₁ > f x₂ := by
  sorry

end f_decreasing_on_0_1_l381_381033


namespace distribution_equality_l381_381004

noncomputable def characteristic_function_X : ℝ → ℂ := sorry
noncomputable def characteristic_function_Y : ℝ → ℂ := sorry
def analytic_function_f (x : ℂ) : ℂ := sorry
def analytic_function_g (x : ℂ) : ℂ := sorry
axiom f_phi_equals_g_psi (t : ℝ) : analytic_function_f (characteristic_function_X t) = analytic_function_g (characteristic_function_Y t)

theorem distribution_equality (X Y : Type) [measure_space X] [measure_space Y] 
  (varphi_X : ℝ → ℂ) (varphi_Y : ℝ → ℂ) (f : ℂ → ℂ) (g : ℂ → ℂ)
  (h : ∀ t: ℝ, f (varphi_X t) = g (varphi_Y t))
  (analytic_f : analytic_on ℂ f (ball 0 (1:ℝ)))
  (analytic_g : analytic_on ℂ g (ball 0 (1:ℝ)))
  (characteristic_X : characteristic_function_X = varphi_X)
  (characteristic_Y : characteristic_function_Y = varphi_Y) :
  dist_equiv X Y :=
sorry

end distribution_equality_l381_381004


namespace range_of_a_l381_381718

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, |x - 2 * a| + |x + 3| < 5) ↔ (-∞, -4] ∪ [1, ∞) :=
by
  sorry

end range_of_a_l381_381718


namespace inequality_of_f_log2015_l381_381020

noncomputable def f : ℝ → ℝ := sorry

theorem inequality_of_f_log2015 :
  (∀ x : ℝ, deriv f x > f x) →
  f (Real.log 2015) > 2015 * f 0 :=
by sorry

end inequality_of_f_log2015_l381_381020


namespace percentage_reduction_30_percent_l381_381066

variable (P S : ℝ)
variable (x : ℝ)
variable (new_receipts : ℝ)

-- Definition of the conditions
def sales_increase_by_50_percent (S : ℝ) : ℝ := 1.5 * S
def new_price (P : ℝ) (x : ℝ) : ℝ := P * (1 - x / 100)
def percentage_change_in_receipts (P S : ℝ) (x : ℝ) : Prop :=
  (new_price P x) * (sales_increase_by_50_percent S) = 1.05 * (P * S)

-- Lean theorem statement for proof
theorem percentage_reduction_30_percent (h : percentage_change_in_receipts P S x) : x = 30 :=
sorry

end percentage_reduction_30_percent_l381_381066


namespace complement_union_eq_ge_two_l381_381994

def U : Set ℝ := Set.univ
def M : Set ℝ := { x : ℝ | x < 1 }
def N : Set ℝ := { x : ℝ | -1 < x ∧ x < 2 }

theorem complement_union_eq_ge_two : { x : ℝ | x ≥ 2 } = U \ (M ∪ N) :=
by
  sorry

end complement_union_eq_ge_two_l381_381994


namespace imo_42nd_inequality_l381_381439

theorem imo_42nd_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a / Real.sqrt (a^2 + 8 * b * c)) + (b / Real.sqrt (b^2 + 8 * c * a)) + (c / Real.sqrt (c^2 + 8 * a * b)) ≥ 1 := by
  sorry

end imo_42nd_inequality_l381_381439


namespace sum_of_possible_t_values_l381_381593

open Real

def cos30 := cos (30 * pi / 180)
def sin30 := sin (30 * pi / 180)
def cos90 := cos (90 * pi / 180)
def sin90 := sin (90 * pi / 180)

noncomputable def valid_t_values : List ℝ := [30, 150, 210, 330]

theorem sum_of_possible_t_values :
  ∑ (x : ℝ) in valid_t_values.toFinset, x = 690 := by
  sorry

end sum_of_possible_t_values_l381_381593


namespace TCUS_area_TCUS_perimeter_l381_381137

noncomputable def radius : ℝ := 12
noncomputable def side_length : ℝ := radius

-- Point definitions
noncomputable def S : Point := ⟨0, 0⟩ -- Center of the circle
noncomputable def B : Point := ⟨radius, 0⟩
noncomputable def C : Point := ⟨radius / 2, (radius * sqrt 3) / 2⟩
noncomputable def T : Point := ⟨radius / 2, 0⟩ -- Midpoint of BC

-- Function to compute the distance between two points
noncomputable def distance (p1 p2 : Point) : ℝ :=
  sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

-- Variables to help define length and area
noncomputable def CT := distance T C
noncomputable def ST := distance S T
noncomputable def area_triangle_CST := (CT * ST) / 2
noncomputable def area_TCUS := 2 * area_triangle_CST
noncomputable def perimeter_TCUS := CT + ST + ST + CT

theorem TCUS_area : area_TCUS = 36 * sqrt 3 :=
by sorry

theorem TCUS_perimeter : perimeter_TCUS = 36 :=
by sorry

end TCUS_area_TCUS_perimeter_l381_381137


namespace max_borrowed_books_l381_381388

theorem max_borrowed_books :
  ∀ (students : ℕ) (avg_books : ℕ) 
    (no_book_students : ℕ) 
    (one_book_students : ℕ)
    (two_books_students : ℕ)
    (at_least_three_books_students : ℕ), 
  students = 20 →
  no_book_students = 2 →
  one_book_students = 8 →
  two_books_students = 3 →
  at_least_three_books_students = students - (no_book_students + one_book_students + two_books_students) →
  avg_books * students = 40 →
  at_least_three_books_total ≤ 26 →
  at_least_three_books_total = avg_books * students - (0 * no_book_students + 1 * one_book_students + 2 * two_books_students) →
  (at_least_three_books_total ≥ at_least_three_books_students * 3) →
  (∃ max_books, max_books = 8 ∧ max_books = at_least_three_books_total - (at_least_three_books_students - 1) * 3 ) :=
begin
  sorry
end

end max_borrowed_books_l381_381388


namespace even_P_conditions_l381_381319

open Nat

variable (P : ℕ → Prop)

theorem even_P_conditions (k : ℕ) (n : ℕ) :
  (∀ n : ℕ, n ∈ (range 1 1001) → P (2 * n)) →
  P 2002 →
  (∃ n : ℕ, k = 2 * n ∧ ¬P k) :=
by
  sorry

end even_P_conditions_l381_381319


namespace compute_logarithmic_sum_l381_381231

theorem compute_logarithmic_sum : 
  ∑ k in finset.Icc 3 100, (log 3 (1 + (1 : ℝ) / k) * log k 3 * log (k + 1) 3) = -1 := 
sorry

end compute_logarithmic_sum_l381_381231


namespace seq_diff_bound_l381_381926

theorem seq_diff_bound 
  (a : ℕ → ℝ) 
  (h : ∀ n : ℕ, |a (n + 1) - a n| ≤ 1) :
  ∀ n : ℕ, 
    let b := λ n, (∑ i in finset.range (n + 1), a (i + 1)) / (n + 1)
    in |b (n + 1) - b n| ≤ 1 / 2 :=
sorry

end seq_diff_bound_l381_381926


namespace part1_part2_l381_381819

open Real

/- Define the curve C in Cartesian coordinates -/
def curve_C (x y : ℝ) : Prop := x^2 + y^2 - 4 * x = 0

/- Parametric equations for the line l -/
def line_l (m t : ℝ) : ℝ × ℝ :=
  (m + (sqrt 2 / 2) * t, (sqrt 2 / 2) * t)

/- Equation for distance |AB| = sqrt(14) -/
def dist_AB (t1 t2 m : ℝ) : Prop :=
  abs (t1 - t2) = sqrt 14

/- Solutions m = 1 or m = 3 -/
theorem part1 (m : ℝ) (h : ∃ t1 t2, dist_AB t1 t2 m) :
  m = 1 ∨ m = 3 :=
sorry

/- Parametric equations for the curve C -/
def param_curve_C (θ : ℝ) : ℝ × ℝ :=
  (2 + 2 * cos θ, 2 * sin θ)

/- Range of x + y for point on curve C -/
theorem part2 :
  ∀ (x y : ℝ), curve_C x y →
  2 - 2 * sqrt 2 ≤ x + y ∧ x + y ≤ 2 + 2 * sqrt 2 :=
sorry


end part1_part2_l381_381819


namespace cheryl_tournament_cost_is_1440_l381_381215

noncomputable def cheryl_electricity_bill : ℝ := 800
noncomputable def additional_for_cell_phone : ℝ := 400
noncomputable def cheryl_cell_phone_expenses : ℝ := cheryl_electricity_bill + additional_for_cell_phone
noncomputable def tournament_cost_percentage : ℝ := 0.2
noncomputable def additional_tournament_cost : ℝ := tournament_cost_percentage * cheryl_cell_phone_expenses
noncomputable def total_tournament_cost : ℝ := cheryl_cell_phone_expenses + additional_tournament_cost

theorem cheryl_tournament_cost_is_1440 : total_tournament_cost = 1440 := by
  sorry

end cheryl_tournament_cost_is_1440_l381_381215


namespace beetles_meeting_l381_381787

theorem beetles_meeting {n : ℕ} (h₁ : n = 2023) :
  ∃ (beetles_count : ℕ), beetles_count = 4088485 ∧
  (∀ (initial_positions : fin n × fin n → Prop) (moves : ℕ → (fin n × fin n) → (fin n × fin n)),
   (∀ t pos, (moves (t+1) (moves t pos) = pos)) → 
   ∃ t pos₁ pos₂, t > 0 ∧ pos₁ ≠ pos₂ ∧ moves t pos₁ = moves t pos₂) :=
by
  use 4088485
  split
  · rfl
  · intros initial_positions moves h_moves
    sorry

end beetles_meeting_l381_381787


namespace range_of_a_l381_381790

theorem range_of_a (f : ℝ → ℝ) (h_mono : ∀ x y, x < y → f(x) < f(y))
  (h : ∀ a, f(2 - a^2) > f(a)) : ∀ a : ℝ, 2 - a^2 > a ↔ (a < -2 ∨ 1 < a) :=
by
  sorry

end range_of_a_l381_381790


namespace complement_union_l381_381449

def U : Set ℕ := {0, 1, 2, 3, 4}
def M : Set ℕ := {1, 2, 4}
def N : Set ℕ := {2, 3}

theorem complement_union (U : Set ℕ) (M : Set ℕ) (N : Set ℕ) (hU : U = {0, 1, 2, 3, 4}) (hM : M = {1, 2, 4}) (hN : N = {2, 3}) :
  (U \ M) ∪ N = {0, 2, 3} :=
by
  rw [hU, hM, hN] -- Substitute U, M, N definitions
  sorry -- Proof omitted

end complement_union_l381_381449


namespace probability_of_condition_l381_381154

def eccentricity (a b : ℕ) : ℝ := 
  Real.sqrt (1 - (b^2) / (a^2))

def satisfies_condition (a b : ℕ) : Prop :=
  eccentricity a b ≥ Real.sqrt 3 / 2

def count_valid_outcomes : ℕ :=
  List.length [ (2, 1), (3, 1), (4, 1), (4, 2), (5, 1), (5, 2), (6, 1), (6, 2), (6, 3) ]

def total_outcomes : ℕ := 36

def probability : ℝ := count_valid_outcomes / total_outcomes

theorem probability_of_condition : probability = 1 / 4 := sorry

end probability_of_condition_l381_381154


namespace radius_range_l381_381887

def circleEquation (x y : ℝ) (r : ℝ) := (x - 3)^2 + (y + 5)^2 = r^2

def distanceFromCenterToLine (A B C x₀ y₀ : ℝ) := (| A * x₀ + B * y₀ + C |) / real.sqrt (A^2 + B^2)

def distFromCenter := distanceFromCenterToLine 4 (-3) (-2) 3 (-5)
def distToLineCloser := distanceFromCenterToLine 4 (-3) (-7) 3 (-5)
def distToLineFarther := distanceFromCenterToLine 4 (-3) 3 3 (-5)

theorem radius_range (r : ℝ) : 
  (circleEquation 3 (-5) r) ∧ 
  (distFromCenter = 5) ∧ 
  (distToLineCloser = 4) ∧ 
  (distToLineFarther = 6) → 
  (4 < r ∧ r < 6) :=
sorry

end radius_range_l381_381887


namespace complement_union_eq_l381_381969

-- Definitions / Conditions
def U : Set ℝ := Set.univ
def M : Set ℝ := { x | x < 1 }
def N : Set ℝ := { x | -1 < x ∧ x < 2 }

-- Statement of the theorem
theorem complement_union_eq {x : ℝ} :
  {x | x ≥ 2} = (U \ (M ∪ N)) := sorry

end complement_union_eq_l381_381969


namespace fraction_of_is_l381_381563

theorem fraction_of_is (a b c d e : ℚ) (h1 : a = 2) (h2 : b = 9) (h3 : c = 3) (h4 : d = 4) (h5 : e = 8/27) :
  (a / b) = e * (c / d) := 
sorry

end fraction_of_is_l381_381563


namespace problem_1_solution_set_problem_2_range_of_a_l381_381443

section Problem1
  variables {x a : ℝ}
  def f (x : ℝ) : ℝ := abs (x - 1)
  def g (x : ℝ) : ℝ := 2 * abs (x - 2)
  def inequality := f x - g x ≤ x - 3

  theorem problem_1_solution_set : {x : ℝ | f x - g x ≤ x - 3} = { x | x ≤ 1 ∨ x ≥ 3 } := 
  sorry
end Problem1

section Problem2
  variables {x a m : ℝ}
  def f (x : ℝ) : ℝ := abs (x - 1)
  def g (x : ℝ) : ℝ := 2 * abs (x - a)
  def inequality2 := ∃ x : ℝ, ∀ (m > 1), f x + g x ≤ (m^2 + m + 4) / (m - 1)

  theorem problem_2_range_of_a : (∀ m > 1, ∃ x : ℝ, f x + g x ≤ (m^2 + m + 4) / (m-1)) →
                                -2 * Real.sqrt 6 - 2 ≤ a ∧ a ≤ 2 * Real.sqrt 6 + 4 :=
  sorry
end Problem2

end problem_1_solution_set_problem_2_range_of_a_l381_381443


namespace complete_the_square_l381_381262

theorem complete_the_square (y : ℤ) : y^2 + 14 * y + 60 = (y + 7)^2 + 11 :=
by
  sorry

end complete_the_square_l381_381262


namespace magnitude_possibilities_l381_381802

-- Given definitions and conditions
variables {a b c : ℝ}
variables {z1 z2 z3 : ℂ}

-- Magnitudes are 1
def magnitudes_equal_one : Prop :=
  |z1| = 1 ∧ |z2| = 1 ∧ |z3| = 1

-- Complex sum is real
def complex_sum_is_real : Prop :=
  (z1 / z2 + z2 / z3 + z3 / z1).im = 0

-- Conclusion
def possible_magnitudes (a b c : ℝ) (z1 z2 z3 : ℂ) := 
  (|a * z1 + b * z2 + c * z3|
   = sqrt ((a + b)^2 + c^2)
  ∨ |a * z1 + b * z2 + c * z3|
   = sqrt ((a + c)^2 + b^2)
  ∨ |a * z1 + b * z2 + c * z3|
   = sqrt ((b + c)^2 + a^2))

-- Proving the main theorem statement
theorem magnitude_possibilities 
  (h1 : magnitudes_equal_one)
  (h2 : complex_sum_is_real) :
  possible_magnitudes a b c z1 z2 z3 :=
sorry

end magnitude_possibilities_l381_381802


namespace angle_between_diagonals_of_right_prism_l381_381052

theorem angle_between_diagonals_of_right_prism (α : ℝ) :
  let β := Real.arccos (2 / Real.sqrt (8 + Real.sin (2 * α) ^ 2)) in
  ∀ (a b c a1 b1 c1 : ℝ),
    a = 1 → 
    b = a * Real.cos α → 
    c = a * Real.sin α → 
    let A1C := Real.sqrt (a1^2 + b^2) in 
    let B1C := Real.sqrt (b1^2 + c^2) in 
    let AA1BB1 := a1 = b1 ∧ a1 = c1 in
    β = Real.arccos (2 / (Real.sqrt (4 + Real.sin (2*α)^2)))  :=
by {
  intros β a b c a1 b1 c1 ha hb hc A1C B1C AA1BB1,
  sorry
}

end angle_between_diagonals_of_right_prism_l381_381052


namespace cube_volume_correct_l381_381654

-- Define the given conditions
def radius := 56 -- cm
def height := 20 -- cm
def wire_length : ℝ := 4 * Real.pi * radius + 2 * height

-- Define the edge length and cube volume
def edge_length : ℝ := wire_length / 12
def cube_volume : ℝ := edge_length^3

-- The theorem that verifies the volume
theorem cube_volume_correct : cube_volume = 201684.7 := by
  unfold cube_volume edge_length wire_length radius height
  norm_num1
  sorry

end cube_volume_correct_l381_381654


namespace quadratic_has_real_solutions_l381_381503

theorem quadratic_has_real_solutions (m : ℝ) : 
  (∃ x : ℝ, (m - 2) * x^2 - 2 * x + 1 = 0) → m ≤ 3 := 
by
  sorry

end quadratic_has_real_solutions_l381_381503


namespace total_goals_eq_eight_l381_381115

noncomputable theory

variables (A : ℝ) (goals_fifth : ℝ := 2) (average_increase : ℝ := 0.1)

def total_goals_in_5_matches (A : ℝ) : ℝ :=
  let total_goals_before_5th := 4 * A in
  let new_total_goals := total_goals_before_5th + goals_fifth in
  if (new_total_goals / 5 = A + average_increase) then 4 * A + 2 else 0

theorem total_goals_eq_eight (A : ℝ) (h : A = 1.5) :
  total_goals_in_5_matches A = 8 :=
by
  sorry

end total_goals_eq_eight_l381_381115


namespace Shyam_money_l381_381610

theorem Shyam_money (r g k s : ℕ) 
  (h1 : 7 * g = 17 * r) 
  (h2 : 7 * k = 17 * g)
  (h3 : 11 * s = 13 * k)
  (hr : r = 735) : 
  s = 2119 := 
by
  sorry

end Shyam_money_l381_381610


namespace initial_native_trees_l381_381144

theorem initial_native_trees (N : ℕ) 
    (monday_plant : 2 * N)
    (tuesday_plant : (2 / 3) * N)
    (total_plant : monday_plant + tuesday_plant = 80) : 
    N = 30 :=
by
  sorry

end initial_native_trees_l381_381144


namespace complement_union_eq_ge_two_l381_381997

def U : Set ℝ := Set.univ
def M : Set ℝ := { x : ℝ | x < 1 }
def N : Set ℝ := { x : ℝ | -1 < x ∧ x < 2 }

theorem complement_union_eq_ge_two : { x : ℝ | x ≥ 2 } = U \ (M ∪ N) :=
by
  sorry

end complement_union_eq_ge_two_l381_381997


namespace examination_is_30_hours_l381_381903

noncomputable def examination_time_in_hours : ℝ :=
  let total_questions := 200
  let type_a_problems := 10
  let total_time_on_type_a := 17.142857142857142
  let time_per_type_a := total_time_on_type_a / type_a_problems
  let time_per_type_b := time_per_type_a / 2
  let type_b_problems := total_questions - type_a_problems
  let total_time_on_type_b := time_per_type_b * type_b_problems
  let total_time_in_minutes := total_time_on_type_a * type_a_problems + total_time_on_type_b
  total_time_in_minutes / 60

theorem examination_is_30_hours :
  examination_time_in_hours = 30 := by
  sorry

end examination_is_30_hours_l381_381903


namespace sum_of_solutions_eq_l381_381719

theorem sum_of_solutions_eq : (∑ x in (finset.image id ({x ∈ (finset.Icc 4 53) | (x - 4)^2 = 49})), x) = 8 :=
by
  sorry

end sum_of_solutions_eq_l381_381719


namespace calc_log_expression_l381_381208

theorem calc_log_expression : 2 * Real.log 5 + Real.log 4 = 2 :=
by
  sorry

end calc_log_expression_l381_381208


namespace lunch_people_count_l381_381523

theorem lunch_people_count
  (C : ℝ)   -- total lunch cost including gratuity
  (G : ℝ)   -- gratuity rate
  (P : ℝ)   -- average price per person excluding gratuity
  (n : ℕ)   -- number of people
  (h1 : C = 207.0)  -- condition: total cost with gratuity
  (h2 : G = 0.15)   -- condition: gratuity rate of 15%
  (h3 : P = 12.0)   -- condition: average price per person
  (h4 : C = (1 + G) * n * P) -- condition: total cost with gratuity is (1 + gratuity rate) * number of people * average price per person
  : n = 15 :=       -- conclusion: number of people
sorry

end lunch_people_count_l381_381523


namespace pears_left_l381_381923

theorem pears_left (keith_initial : ℕ) (keith_given : ℕ) (mike_initial : ℕ) 
  (hk : keith_initial = 47) (hg : keith_given = 46) (hm : mike_initial = 12) :
  (keith_initial - keith_given) + mike_initial = 13 := by
  sorry

end pears_left_l381_381923


namespace allison_total_craft_supplies_l381_381178

theorem allison_total_craft_supplies (M_glue : ℕ) (M_paper : ℕ) (A_glue_more : ℕ) (M_paper_ratio : ℕ) :
  M_glue = 15 → M_paper = 30 → A_glue_more = 8 → M_paper_ratio = 6 →
  let A_glue := M_glue + A_glue_more in
  let A_paper := M_paper / M_paper_ratio in
  A_glue + A_paper = 28 :=
by
  intros h1 h2 h3 h4
  let A_glue := 15 + 8
  let A_paper := 30 / 6
  sorry

end allison_total_craft_supplies_l381_381178


namespace train_speed_fraction_l381_381653

theorem train_speed_fraction (T : ℝ) (hT : T = 3) : T / (T + 0.5) = 6 / 7 := by
  sorry

end train_speed_fraction_l381_381653


namespace probability_third_smallest_five_l381_381696

theorem probability_third_smallest_five :
  let S := finset.Icc 1 15 in
  let total_ways := S.card.choose 8 in
  let favorable_ways := (finset.Icc 6 15).card.choose 4 * (finset.Icc 1 4).card.choose 2 in
  (favorable_ways : ℚ) / total_ways = 4 / 21 :=
by {
  let S := finset.Icc 1 15,
  let total_ways := S.card.choose 8,
  let favorable_ways := (finset.Icc 6 15).card.choose 4 * (finset.Icc 1 4).card.choose 2,
  have h : favorable_ways = 1260 := rfl,
  have h2 : total_ways = 6435 := rfl,
  calc
    (favorable_ways : ℚ) / total_ways
        = (1260 : ℚ) / 6435 : by rw [h, h2]
    ... = 4 / 21 : by norm_num,
  sorry
}

end probability_third_smallest_five_l381_381696


namespace greatest_integer_l381_381583

theorem greatest_integer (y : ℤ) (h : (8 : ℚ) / 11 > y / 17) : y ≤ 12 :=
by
  have h₁ : (8 : ℚ) / 11 * 17 > y := by exact (div_mul_cancel _ (by norm_num : 17 ≠ 0))
  have h₂ : 136 / 11 > y := by rwa mul_comm _ 17 at h₁
  exact_mod_cast le_of_lt h₂

end greatest_integer_l381_381583


namespace triangle_area_is_24_l381_381566

-- Defining the vertices of the triangle
def A := (2, 2)
def B := (8, 2)
def C := (4, 10)

-- Calculate the area of the triangle
def area_of_triangle (A B C : ℕ × ℕ) : ℕ := 
  let base := |B.1 - A.1| 
  let height := |C.2 - A.2| 
  ((base * height) / 2)

-- Statement to prove
theorem triangle_area_is_24 : area_of_triangle A B C = 24 := 
by
  sorry

end triangle_area_is_24_l381_381566


namespace line_through_origin_line_parallel_to_given_line_perpendicular_to_given_l381_381287

-- Definitions of the lines
def l1 (x y : ℝ) : Prop := 3 * x + 4 * y + 5 = 0
def l2 (x y : ℝ) : Prop := 2 * x - 3 * y - 8 = 0

-- Intersection point
def M := (1 : ℝ, -2 : ℝ)

-- Equation of the line through origin
def line1 (x y : ℝ) : Prop := 2 * x + y = 0

-- Equation of the line parallel to the given line
def line_parallel (x y : ℝ) : Prop := 2 * x + y = 0

-- Equation of the line perpendicular to the given line
def line_perpendicular (x y : ℝ) : Prop := x - 2 * y - 5 = 0

-- Statements to prove
theorem line_through_origin (x y : ℝ) : l1 x y ∧ l2 x y ∧ (x, y) = M → line1 x y :=
by sorry

theorem line_parallel_to_given (x y : ℝ) : l1 x y ∧ l2 x y ∧ (x, y) = M → line_parallel x y :=
by sorry

theorem line_perpendicular_to_given (x y : ℝ) : l1 x y ∧ l2 x y ∧ (x, y) = M → line_perpendicular x y :=
by sorry

end line_through_origin_line_parallel_to_given_line_perpendicular_to_given_l381_381287


namespace altitude_correct_l381_381051

-- Define the given sides and area of the triangle
def AB : ℝ := 30
def BC : ℝ := 17
def AC : ℝ := 25
def area_ABC : ℝ := 120

-- The length of the altitude from the vertex C to the base AB
def height_C_to_AB : ℝ := 8

-- Problem statement to be proven
theorem altitude_correct : (1 / 2) * AB * height_C_to_AB = area_ABC :=
by
  sorry

end altitude_correct_l381_381051


namespace angle_CDA_proof_l381_381191

theorem angle_CDA_proof : 
  ∀ (A B C D E : Type) 
    (angle_ABE angle_DAB angle_CEB : ℝ),
  angle_ABE = 74 → 
  angle_DAB = 70 → 
  angle_CEB = 20 → 
  (180 - angle_ABE - angle_DAB) + angle_CEB - 
  (180 - ((180 - angle_ABE - angle_DAB) + angle_CEB)) = 92 :=
by
  intros A B C D E angle_ABE angle_DAB angle_CEB h_ABE h_DAB h_CEB
  have h1: 180 - angle_ABE - angle_DAB = 36 := by
    rw [h_ABE, h_DAB]
    norm_num
  have h2: (180 - angle_ABE - angle_DAB) + angle_CEB = 56 := by
    rw [h1, h_CEB]
    norm_num
  have h3: 180 - (180 - angle_ABE - angle_DAB + angle_CEB) = 124 := by
    rw h2
    norm_num
  show (180 - angle_ABE - angle_DAB) + angle_CEB - (180 - ((180 - angle_ABE - angle_DAB) + angle_CEB)) = 92
  rw [h1, h2, h3]
  norm_num

end angle_CDA_proof_l381_381191


namespace even_function_f_neg_one_l381_381431

noncomputable def f : ℝ → ℝ :=
  λ x, if x ≥ 0 then 2^x + Real.log (x + 3) / Real.log 2 else 2^(-x) + Real.log (3 - x) / Real.log 2

-- Because given conditions include continuity and property of even functions
theorem even_function_f_neg_one : f (-1) = 4 :=
by sorry

end even_function_f_neg_one_l381_381431


namespace number_of_quadruples_is_3924_l381_381291

def total_quadruples : ℕ := 10 ^ 4

def f (x y z u : ℕ) : ℚ :=
  (↑(x - y) / ↑(x + y)) + (↑(y - z) / ↑(y + z)) + (↑(z - u) / ↑(z + u)) + (↑(u - x) / ↑(u + x))

def A : finset (ℕ × ℕ × ℕ × ℕ) := 
  {t ∈ (finset.range 10).product (finset.range 10).product (finset.range 10).product (finset.range 10) |
  f t.1.1 t.1.2.1 t.1.2.2 t.2 > 0}

def B : finset (ℕ × ℕ × ℕ × ℕ) := 
  {t ∈ (finset.range 10).product (finset.range 10).product (finset.range 10).product (finset.range 10) |
  f t.1.1 t.1.2.1 t.1.2.2 t.2 < 0}

def C : finset (ℕ × ℕ × ℕ × ℕ) := 
  {t ∈ (finset.range 10).product (finset.range 10).product (finset.range 10).product (finset.range 10) |
  f t.1.1 t.1.2.1 t.1.2.2 t.2 = 0}

theorem number_of_quadruples_is_3924 :
  A.card = 3924 :=
sorry

end number_of_quadruples_is_3924_l381_381291


namespace find_a_10_l381_381070

variable {a_n : ℕ → ℕ} 
variable {S_n : ℕ → ℕ} 

-- Definitions from conditions
def condition_1 (n : ℕ) : Prop := S_n n = 2 * a_n n - 1
def condition_2 (n : ℕ) : Prop := S_n n = ∑ i in Finset.range (n+1), a_n i

-- Problem statement based on identified question and correct answer
theorem find_a_10 (h1 : ∀ n, condition_1 n) (h2 : ∀ n, condition_2 n) : 
    a_n 10 = 512 :=
sorry

end find_a_10_l381_381070


namespace train_speed_l381_381150

-- Definitions and conditions given in the problem
def v_g : ℝ := 62 -- speed of the goods train in kmph
def L : ℝ := 280 -- length of the goods train in meters
def t : ℝ := 9 -- time taken to pass the man in seconds
def k : ℝ := 5 / 18 -- kmph to m/s conversion factor

-- Proof statement that is mathematically equivalent to the problem statement
theorem train_speed (v : ℝ) : 
  L = (v + v_g) * k * t → v = 50 := 
by sorry

end train_speed_l381_381150


namespace participants_started_competition_l381_381900

-- Given conditions
variable (x : ℕ) (h1 : 0.4 * x * (1 / 4) = 30)

-- Definition of the problem in Lean 4 statement
theorem participants_started_competition (x : ℕ) (h1 : 0.4 * x * (1 / 4) = 30) : x = 300 :=
sorry

end participants_started_competition_l381_381900


namespace probability_one_pair_switch_three_unchanged_l381_381165

-- Definitions for the problem conditions
def total_boys : ℕ := 9
def choose (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))
def derangements (n : ℕ) : ℕ := Nat.factorial n * (∑i in Finset.range n, (-1 : ℤ)^i / Nat.factorial i)

-- Main theorem to prove the probability equals 1/32
theorem probability_one_pair_switch_three_unchanged :
  let favorable_outcomes := choose 9 3 * choose 6 2 * derangements 4
  let total_arrangements  := Nat.factorial 9
  let probability := favorable_outcomes / total_arrangements
  probability = 1 / 32 :=
by
  admit -- assuming the definitions are correct, this step would compute the values

end probability_one_pair_switch_three_unchanged_l381_381165


namespace smallest_number_divisibility_l381_381109

theorem smallest_number_divisibility :
  ∃ n : ℕ, (n - 7) % 12 = 0 ∧
           (n - 7) % 16 = 0 ∧
           (n - 7) % 18 = 0 ∧
           (n - 7) % 21 = 0 ∧
           (n - 7) % 28 = 0 ∧
           n = 1015 :=
by {
  let n := 1015,
  use n,
  split, { norm_num }, split, { norm_num }, split, { norm_num }, split, { norm_num }, split, { norm_num }, 
  exact rfl,
  sorry,
}

end smallest_number_divisibility_l381_381109


namespace general_term_b_sum_inequality_l381_381447

variable (a : ℕ → ℝ) (T : ℕ → ℝ) (S : ℕ → ℝ) (b : ℕ → ℝ)

-- Conditions
axiom condition1 : ∀ n : ℕ, T n = ∏ i in Finset.range n.succ, a (i + 1)
axiom condition2 : ∀ n : ℕ, 2 * a (n + 1) + T n.succ = 1
axiom condition3 : ∀ n : ℕ, b n = 1 + 1 / T n
axiom condition4 : ∀ n : ℕ, S n = ∑ i in Finset.range n.succ, a (i + 1)

-- Question 1: Find the general term formula for b_n
theorem general_term_b (n : ℕ) : b n = 2 ^ (n + 1) :=
sorry

-- Question 2: Prove the inequality for S_n
theorem sum_inequality (n : ℕ) : S n < (↑n / 2) + (0.5) * Real.log (T n + 1) - 0.25 :=
sorry

end general_term_b_sum_inequality_l381_381447


namespace ratio_yelling_to_hold_l381_381023

noncomputable def time_spent_yelling {router_time hold_multiplier total_time : ℕ} 
  (yelling_ratio : ℕ) : ℕ :=
total_time - (router_time + hold_multiplier * router_time)

theorem ratio_yelling_to_hold 
  (router_time : ℕ := 10)
  (hold_multiplier : ℕ := 6)
  (total_time : ℕ := 100)
  (yelling_ratio := 1 : 2) :
  let hold_time := hold_multiplier * router_time
  let yelling_time := time_spent_yelling router_time hold_multiplier total_time 
  (yelling_ratio : ℕ) in 
  yelling_time / hold_time = 1 / 2 := 
by
  let hold_time := hold_multiplier * router_time
  let yelling_time := time_spent_yelling router_time hold_multiplier total_time yelling_ratio
  have h1 : 10 + 60 + yelling_time = 100, from sorry
  have h2 : yelling_time = 30, from sorry
  have h3 : hold_time = 60, from sorry
  have h4 : yelling_time / hold_time = 1 / 2, from sorry
  exact h4

end ratio_yelling_to_hold_l381_381023


namespace correct_assignment_l381_381776

structure GirlDressAssignment :=
  (Katya : String)
  (Olya : String)
  (Liza : String)
  (Rita : String)

def solution : GirlDressAssignment :=
  ⟨"Green", "Blue", "Pink", "Yellow"⟩

theorem correct_assignment
  (Katya_not_pink_or_blue : solution.Katya ≠ "Pink" ∧ solution.Katya ≠ "Blue")
  (Green_between_Liza_and_Yellow : 
    (solution.Katya = "Green" ∧ solution.Liza = "Pink" ∧ solution.Rita = "Yellow") ∧
    (solution.Katya = "Green" ∧ solution.Rita = "Yellow" ∧ solution.Liza = "Pink"))
  (Rita_not_green_or_blue : solution.Rita ≠ "Green" ∧ solution.Rita ≠ "Blue")
  (Olya_between_Rita_and_Pink : 
    (solution.Olya = "Blue" ∧ solution.Rita = "Yellow" ∧ solution.Liza = "Pink") ∧
    (solution.Olya = "Blue" ∧ solution.Liza = "Pink" ∧ solution.Rita = "Yellow"))
  : solution = ⟨"Green", "Blue", "Pink", "Yellow"⟩ := by
  sorry

end correct_assignment_l381_381776


namespace greatest_integer_y_l381_381576

theorem greatest_integer_y (y : ℤ) : (8 : ℚ) / 11 > y / 17 ↔ y ≤ 12 := 
sorry

end greatest_integer_y_l381_381576


namespace find_triangle_angles_l381_381527

-- The definition of the sides of the triangle
def a : ℝ := 4
def b : ℝ := 2 * Real.sqrt 3
def c : ℝ := 2 + 2 * Real.sqrt 2

-- Definition of the angles α, β, and γ
def α : ℝ := sorry -- placeholder for calculation of α from Law of Cosines
def β : ℝ := 60 -- since β is explicitly given as 60 degrees
def γ : ℝ := 180 - α - β -- using the sum of the interior angles of a triangle

-- Theorem stating the angles of the triangle
theorem find_triangle_angles : 
  α = sorry ∧ β = 60 ∧ γ = 180 - α - β :=
by sorry

end find_triangle_angles_l381_381527


namespace sum_of_x_values_l381_381263

def mean_of_five (x : ℝ) : ℝ :=
  (3 + 7 + 12 + 24 + x) / 5

def median_of_five (x : ℝ) : ℝ :=
  if x < 3 then 7
  else if x < 7 then 7
  else if x < 12 then x
  else if x < 24 then 12
  else x

theorem sum_of_x_values : ∑ (x : ℝ) in {x | mean_of_five x = median_of_five x}, x = 3 :=
by
  -- Proof to be provided
  sorry

end sum_of_x_values_l381_381263


namespace sum_of_solutions_l381_381720

theorem sum_of_solutions :
  let f := λ x : ℝ, abs (x^2 - 8 * x + 15)
  let g := λ x : ℝ, 8 - x
  let solutions := { x | f x = g x }
  (solutions.filter (λ x, x ≤ 3 ∨ x ≥ 5)).sum = 7 :=
sorry

end sum_of_solutions_l381_381720


namespace find_value_of_a_l381_381358

theorem find_value_of_a (a : ℝ) :
  let U := {2, 4, 1 - a} in
  let A := {2, a^2 - a + 2} in
  let compl_U_A := {x | x ∈ U ∧ x ∉ A} in
  compl_U_A = {-1} → a = 2 :=
by
  let U := {2, 4, 1 - a}
  let A := {2, a^2 - a + 2}
  let compl_U_A := {x | x ∈ U ∧ x ∉ A}
  intro h
  have : compl_U_A = {-1} := h
  sorry

end find_value_of_a_l381_381358


namespace foreign_exchange_decline_l381_381135

theorem foreign_exchange_decline (x : ℝ) (h1 : 200 * (1 - x)^2 = 98) : 
  200 * (1 - x)^2 = 98 :=
by
  sorry

end foreign_exchange_decline_l381_381135


namespace knights_number_l381_381450

def is_simple (n : ℕ) : Prop :=
  n = 1 ∨ (n > 1 ∧ ¬ even n ∧ (∀ m : ℕ, m > 1 ∧ m < n → m ∣ n → n = m))

def is_knight (arr : list ℕ) (i : ℕ) : Prop :=
  let left_knights := list.countp (λ x => x = 1) (arr.take i)
  let right_knights := list.countp (λ x => x = 1) (arr.drop (i + 1))
  let difference := abs (left_knights - right_knights)
  is_simple difference

theorem knights_number :
  ∃ n : ℕ, list.countp (λ x => x = 1) (list.range 2019) = n ∧ (n = 0 ∨ n = 2 ∨ n = 4 ∨ n = 6 ∨ n = 8) :=
by
  sorry

end knights_number_l381_381450


namespace minimum_projects_for_30_points_l381_381652

theorem minimum_projects_for_30_points :
  (∑ n in finset.range 5, (n + 1) * 6) = 90 :=
by
  sorry

end minimum_projects_for_30_points_l381_381652


namespace sum_logarithms_l381_381236

theorem sum_logarithms :
  (∑ k in finset.Ico 3 101, real.logb 3 (1 + 1/(k:ℝ)) * real.logb k 3 * real.logb (k+1) 3) =
  1 - real.logb 2 3 / real.logb 2 101 :=
by
  sorry

end sum_logarithms_l381_381236


namespace angle_in_second_quadrant_l381_381876

theorem angle_in_second_quadrant (θ : ℝ) (h1 : sin θ * cos θ < 0) (h2 : cos θ - sin θ < 0) : 
  θ > π/2 ∧ θ < π := 
sorry

end angle_in_second_quadrant_l381_381876


namespace sin_sub_360_eq_sin_l381_381658

theorem sin_sub_360_eq_sin (α : ℝ) : sin (360 * (π / 180) - α) = sin α :=
by
  sorry

end sin_sub_360_eq_sin_l381_381658


namespace general_formula_for_b_n_range_of_m_l381_381318

section
variables {a b S : ℕ → ℝ}
variables {n m : ℕ}

-- Conditions about the geometric sequence a_n
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ n, a(n + 1) = q * a(n)

-- Define initial conditions
def initial_conditions (a S : ℕ → ℝ) : Prop :=
  a(1) = 1 ∧ 2*a(1) + a(2) = a(3) ∧ ∀ n, S n = ∑ i in range n, a(i + 1)

-- Condition for sequence {b_n}
def b_n_conditions (a S b : ℕ → ℝ) : Prop :=
  ∀ n, 2^(b(n)) = 4*a(n)*(S(n)+1)

-- Derived formula for sequence {T_n}
noncomputable def T_n (b : ℕ → ℝ) (n : ℕ) : ℝ :=
∑ i in range n, 1 / (b(i) * b(i + 1))


-- Main proof statements
theorem general_formula_for_b_n (a S b : ℕ → ℝ) (n : ℕ)
  (h_geom : is_geometric_sequence a)
  (h_init : initial_conditions a S)
  (h_b : b_n_conditions a S b) :
  b(n) = 2*n + 1 := by
  sorry

theorem range_of_m (b : ℕ → ℝ) (m : ℝ) (n : ℕ)
  (h_b : ∀ n, b(n) = 2 * n + 1) :
  m ∈ set.Icc 0 (1/15) → m ≤ T_n b n := by
  sorry
end

end general_formula_for_b_n_range_of_m_l381_381318


namespace no_natural_n_divisible_by_2019_l381_381209

theorem no_natural_n_divisible_by_2019 :
  ∀ n : ℕ, ¬ 2019 ∣ (n^2 + n + 2) :=
by sorry

end no_natural_n_divisible_by_2019_l381_381209


namespace infinitely_many_coprime_binomials_l381_381324

theorem infinitely_many_coprime_binomials (k l : ℕ) (hk : 0 < k) (hl : 0 < l) :
  ∃ᶠ n in at_top, n > k ∧ Nat.gcd (Nat.choose n k) l = 1 := by
  sorry

end infinitely_many_coprime_binomials_l381_381324


namespace count_three_digit_numbers_divisible_by_7_l381_381849

def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

def is_divisible_by_7 (n : ℕ) : Prop :=
  n % 7 = 0

def count_three_digit_divisible_by_7 : ℕ :=
  (list.range' 105 890).countp (λ x, is_divisible_by_7 (x * 7))

theorem count_three_digit_numbers_divisible_by_7 :
  count_three_digit_divisible_by_7 = 128 :=
sorry

end count_three_digit_numbers_divisible_by_7_l381_381849


namespace find_t_for_area_of_triangle_l381_381097

theorem find_t_for_area_of_triangle :
  ∃ (t : ℝ), 
  (∀ (A B C T U: ℝ × ℝ),
    A = (0, 10) → 
    B = (3, 0) → 
    C = (9, 0) → 
    T = (3/10 * (10 - t), t) →
    U = (9/10 * (10 - t), t) →
    2 * 15 = 3/10 * (10 - t) ^ 2) →
  t = 2.93 :=
by sorry

end find_t_for_area_of_triangle_l381_381097


namespace complement_union_M_N_eq_ge_2_l381_381957

def U := set.univ ℝ
def M := {x : ℝ | x < 1}
def N := {x : ℝ | -1 < x ∧ x < 2}

theorem complement_union_M_N_eq_ge_2 :
  (U \ (M ∪ N)) = {x : ℝ | 2 ≤ x} :=
by sorry

end complement_union_M_N_eq_ge_2_l381_381957


namespace complement_union_eq_ge2_l381_381948

open Set

variables {U : Type} [PartialOrder U] [LinearOrder U]

def U : Set ℝ := univ
def M : Set ℝ := { x | x < 1 }
def N : Set ℝ := { x | -1 < x ∧ x < 2 }
def Complement_U (A : Set ℝ) : Set ℝ := U \ A

theorem complement_union_eq_ge2 : 
  Complement_U (M ∪ N) = { x : ℝ | x ≥ 2 } :=
by {
  sorry
}

end complement_union_eq_ge2_l381_381948


namespace success_rate_increase_correct_l381_381024

def initial_attempts := 15
def initial_successes := 7
def new_attempts := 16
def new_success_rate := 3 / 4
def correct_increase := 15

noncomputable def percentage_increase : ℝ :=
  (19 / 31 - 7 / 15) * 100

theorem success_rate_increase_correct :
  (percentage_increase).round = correct_increase :=
by
  have total_initial_attempts := initial_attempts
  have total_initial_successes := initial_successes
  have total_new_attempts := new_attempts
  have total_new_successes := new_attempts * new_success_rate
  let total_attempts := total_initial_attempts + total_new_attempts
  let total_successes := total_initial_successes + total_new_successes
  
  let initial_rate := (total_initial_successes : ℝ) / total_initial_attempts
  let new_rate := (total_successes : ℝ) / total_attempts

  have computed_increase := (new_rate - initial_rate) * 100
  exact (computed_increase.round = correct_increase)

end success_rate_increase_correct_l381_381024


namespace tangent_tangent_value_l381_381341

variable {p : ℝ} (hp : p > 0)
variable {x y : ℝ} (hx : y^2 = p * x) (hy : y > 0)
variable {xM yM : ℝ} (hM : xM = 4 / p ∧ yM = 2)
variable {C1 : ℝ → ℝ} (hC1 : ∀ x, C1 x = e^(x + 1) - 1)

theorem tangent_tangent_value :
  (∀ x, y = sqrt (p * x) → xM = 4 / p ∧ yM = 2 → x = ln (p / 4) - 1) →
  (1/2 * p * log (4 * exp 2 / p) = 4) :=
by
  sorry

end tangent_tangent_value_l381_381341


namespace total_marbles_l381_381895

variable (r : ℝ)

-- Defining the conditions
def num_blue_marbles := r / 1.1
def num_green_marbles := 1.8 * r

-- Prove the total number of marbles is 3.709r
theorem total_marbles : r + num_blue_marbles r + num_green_marbles r = 3.709 * r := by
  sorry

end total_marbles_l381_381895


namespace power_function_at_3_l381_381884

theorem power_function_at_3 :
  ∃ (a : ℝ), (∀ x : ℝ, f x = x ^ a) → f 2 = 8 → f 3 = 27  :=
begin
  sorry
end

end power_function_at_3_l381_381884


namespace range_of_a4_is_l381_381356

noncomputable def range_of_a4 (a1 a2 a3 a4 : ℝ) : set ℝ := { 
  x | x = a4 ∧ 
  a1 + a2 + a3 = 0 ∧ 
  a1 > a2 > a3 ∧ 
  a1 * a4 ^ 2 + a2 * a4 - a2 = 0
}

theorem range_of_a4_is (a1 a2 a3 : ℝ) (h1 : a1 + a2 + a3 = 0) (h2 : a1 > a2) (h3 : a2 > a3) :
  ∃ a4, a4 ∈ range_of_a4 a1 a2 a3 a4 ∧ a4 > (-1 - real.sqrt 5) / 2 ∧ a4 < (-1 + real.sqrt 5) / 2 :=
sorry

end range_of_a4_is_l381_381356


namespace max_students_per_class_l381_381541

theorem max_students_per_class
    (total_students : ℕ)
    (total_classes : ℕ)
    (bus_count : ℕ)
    (bus_seats : ℕ)
    (students_per_class : ℕ)
    (total_students = 920)
    (bus_count = 16)
    (bus_seats = 71)
    (∀ c < total_classes, students_per_class ≤ bus_seats) : 
    students_per_class ≤ 17 := 
by
    sorry

end max_students_per_class_l381_381541


namespace euler_neg_pi_l381_381000

noncomputable def euler_formula (x : ℝ) : ℂ := complex.exp (complex.I * x)

theorem euler_neg_pi : euler_formula (-π) = -1 :=
by
  -- Proof steps are skipped
  sorry

end euler_neg_pi_l381_381000


namespace min_value_of_reciprocals_l381_381930

theorem min_value_of_reciprocals {x y a b : ℝ} 
  (h1 : 8 * x - y - 4 ≤ 0)
  (h2 : x + y + 1 ≥ 0)
  (h3 : y - 4 * x ≤ 0)
  (h4 : 2 = a * (1 / 2) + b * 1)
  (ha : a > 0)
  (hb : b > 0) :
  (1 / a) + (1 / b) = 9 / 2 :=
sorry

end min_value_of_reciprocals_l381_381930


namespace probability_third_smallest_five_l381_381695

theorem probability_third_smallest_five :
  let S := finset.Icc 1 15 in
  let total_ways := S.card.choose 8 in
  let favorable_ways := (finset.Icc 6 15).card.choose 4 * (finset.Icc 1 4).card.choose 2 in
  (favorable_ways : ℚ) / total_ways = 4 / 21 :=
by {
  let S := finset.Icc 1 15,
  let total_ways := S.card.choose 8,
  let favorable_ways := (finset.Icc 6 15).card.choose 4 * (finset.Icc 1 4).card.choose 2,
  have h : favorable_ways = 1260 := rfl,
  have h2 : total_ways = 6435 := rfl,
  calc
    (favorable_ways : ℚ) / total_ways
        = (1260 : ℚ) / 6435 : by rw [h, h2]
    ... = 4 / 21 : by norm_num,
  sorry
}

end probability_third_smallest_five_l381_381695


namespace correct_propositions_l381_381659
-- Import the entire math library to ensure all necessary definitions are available

-- Define the propositions to be used in Lean terms
variable {p q : Prop}
variable {a : ℝ}

-- Proposition 1: Logical statement for p and q
def prop1 : Prop :=
  ((p ∧ q) → (p ∨ q)) ∧ ¬((p ∨ q) → (p ∧ q))

-- Proposition 2: Statement involving negation and real numbers
def prop2 : Prop :=
  (¬(∃ x : ℝ, x^2 + 2*x + 2 ≤ 0) ↔ (∀ x : ℝ, x^2 + 2*x + 2 > 0))

-- Proposition 3: Quadratic inequality and its implications on a
def prop3 : Prop :=
  (¬(∃ x : ℝ, x^2 + (a - 1)*x + 1 < 0)) ↔ (-1 ≤ a ∧ a ≤ 3)

-- Proposition 4: Trigonometric statement and range of a quadratic inequality
def prop4 : Prop :=
  let p := (∃ x : ℝ, Real.tan x = 1) in
  let q := { x : ℝ | 1 < x ∧ x < 2 } = { x : ℝ | 1 < x ∧ x < 2 } in
  ¬(p ∨ ¬(interval_is_neg' 1 2))

-- The main theorem to be proved
theorem correct_propositions : prop2 ∧ prop3 ∧ prop4 :=
by
  sorry

end correct_propositions_l381_381659


namespace prove_problem_l381_381713

variable (x y : ℝ)
variable (solution1 : ℝ × ℝ := (1/3, 3))
variable (solution2 : ℝ × ℝ := (Real.root 4 (21 / 76), 2 * Real.root 4 (84 / 19)))

noncomputable def eq1 : Prop := 
  y - 2 * Real.sqrt(x * y) - Real.sqrt(y / x) + 2 = 0

noncomputable def eq2 : Prop := 
  3 * x^2 * y^2 + y^4 = 84

theorem prove_problem :
  (eq1 (solution1.1) (solution1.2) ∧ eq2 (solution1.1) (solution1.2)) ∨ 
  (eq1 (solution2.1) (solution2.2) ∧ eq2 (solution2.1) (solution2.2)) := sorry

end prove_problem_l381_381713


namespace curve_cusp_implies_arc_of_circle_arc_of_circle_implies_curve_cusp_l381_381034

structure CurveOfConstantWidth (h : ℝ) :=
  (width_constant : ∀ (A B : Point), distance A B = h)

structure Point := 
  (x : ℝ)
  (y : ℝ)

def is_cusp (P : Point) (curve : CurveOfConstantWidth h) : Prop := sorry

def is_arc_of_circle (radius : ℝ) (arc : Set Point) : Prop := sorry

theorem curve_cusp_implies_arc_of_circle {h : ℝ} (curve : CurveOfConstantWidth h) (P : Point) :
  is_cusp P curve → ∃ arc : Set Point, is_arc_of_circle h arc :=
sorry

theorem arc_of_circle_implies_curve_cusp {h : ℝ} (curve : CurveOfConstantWidth h) (arc : Set Point) :
  is_arc_of_circle h arc → ∃ P : Point, is_cusp P curve :=
sorry

end curve_cusp_implies_arc_of_circle_arc_of_circle_implies_curve_cusp_l381_381034


namespace complement_union_eq_ge_two_l381_381991

def U : Set ℝ := Set.univ
def M : Set ℝ := { x : ℝ | x < 1 }
def N : Set ℝ := { x : ℝ | -1 < x ∧ x < 2 }

theorem complement_union_eq_ge_two : { x : ℝ | x ≥ 2 } = U \ (M ∪ N) :=
by
  sorry

end complement_union_eq_ge_two_l381_381991


namespace linear_function_difference_l381_381487

noncomputable def linear_function (f : ℝ → ℝ) : Prop :=
  ∃ m b : ℝ, ∀ x : ℝ, f x = m * x + b

theorem linear_function_difference (f : ℝ → ℝ) 
  (h_linear : linear_function f)
  (h_cond1 : f 10 - f 5 = 20)
  (h_cond2 : f 0 = 3) :
  f 15 - f 5 = 40 :=
sorry

end linear_function_difference_l381_381487


namespace quadratic_function_range_l381_381355

noncomputable def quadratic_range : Set ℝ := {y | -2 ≤ y ∧ y < 2}

theorem quadratic_function_range :
  ∀ y : ℝ, 
    (∃ x : ℝ, -2 < x ∧ x < 1 ∧ y = x^2 + 2 * x - 1) ↔ (y ∈ quadratic_range) :=
by
  sorry

end quadratic_function_range_l381_381355


namespace constant_term_expansion_l381_381285

theorem constant_term_expansion : 
  let p := (x^2 + 2) * (1/x^2 - 1)^5 in
  constant_term p = 3 :=
by sorry

end constant_term_expansion_l381_381285


namespace tan_2alpha_cos_alpha_plus_pi_over_3_l381_381316

theorem tan_2alpha (α : ℝ) (h1 : cos α = -4/5) (h2 : α ∈ set.Ioo (π/2) π) : tan (2 * α) = -24/7 := 
sorry

theorem cos_alpha_plus_pi_over_3 (α : ℝ) (h1 : cos α = -4/5) (h2 : α ∈ set.Ioo (π/2) π) : cos (α + π/3) = (-4 - 3*sqrt 3)/10 := 
sorry

end tan_2alpha_cos_alpha_plus_pi_over_3_l381_381316


namespace intersection_product_l381_381831

-- Step 1: Define conditions
def parametric_line (t α : ℝ) : ℝ × ℝ :=
  (t * Real.cos α, -1 + t * Real.sin α)

def polar_curve (θ : ℝ) : ℝ :=
  6 * Real.sin θ

-- Step 2: Define the rectangular coordinate form of the curve
def rectangular_curve (x y : ℝ) : Prop :=
  x^2 + (y - 3)^2 = 9

-- Step 3: The point P
def point_P : ℝ × ℝ := (0, -1)

-- Step 4: Prove the intersection product
theorem intersection_product (α t1 t2 : ℝ) :
  (rectangular_curve (t1 * Real.cos α) (-1 + t1 * Real.sin α)) ∧
  (rectangular_curve (t2 * Real.cos α) (-1 + t2 * Real.sin α)) →
  (t1 * t2 = 7) :=
sorry

end intersection_product_l381_381831


namespace net_pay_rate_per_hour_l381_381142

-- Defining the given conditions
def travel_hours : ℕ := 3
def speed_mph : ℕ := 50
def fuel_efficiency : ℕ := 25 -- miles per gallon
def pay_rate_per_mile : ℚ := 0.60 -- dollars per mile
def gas_cost_per_gallon : ℚ := 2.50 -- dollars per gallon

-- Define the statement we want to prove
theorem net_pay_rate_per_hour : 
  (travel_hours * speed_mph * pay_rate_per_mile - 
  (travel_hours * speed_mph / fuel_efficiency) * gas_cost_per_gallon) / 
  travel_hours = 25 :=
by
  repeat {sorry}

end net_pay_rate_per_hour_l381_381142


namespace ratio_of_original_to_reversed_l381_381091

def original_number : ℕ := 21
def reversed_number : ℕ := 12

theorem ratio_of_original_to_reversed : 
  (original_number : ℚ) / (reversed_number : ℚ) = 7 / 4 := by
  sorry

end ratio_of_original_to_reversed_l381_381091


namespace intersection_points_l381_381507

def periodic_function (f : ℝ → ℝ) (p : ℝ) : Prop := 
  ∀ x : ℝ, f (x + p) = f x

def specific_function (f : ℝ → ℝ) : Prop := 
  ∀ x : ℝ, x ∈ (-1 : ℝ, 1] → f x = abs x

theorem intersection_points {f : ℝ → ℝ} 
  (hf_periodic : periodic_function f 2) 
  (hf_specific : specific_function f) :
  ∃ n : ℕ, n = 18 ∧ 
    (∃ L : list ℝ, (∀ x ∈ L, f x = real.log (abs x)) ∧ L.length = n) := 
sorry

end intersection_points_l381_381507


namespace probability_third_smallest_is_five_l381_381698

open Finset

noncomputable def prob_third_smallest_is_five : ℚ :=
  let total_ways := choose 15 8
  let favorable_ways := (choose 4 2) * (choose 10 5)
  in favorable_ways / total_ways

theorem probability_third_smallest_is_five :
  prob_third_smallest_is_five = 72 / 307 :=
by sorry

end probability_third_smallest_is_five_l381_381698


namespace farm_corn_cobs_l381_381635

theorem farm_corn_cobs (rows_field1 rows_field2 cobs_per_row : Nat) (h1 : rows_field1 = 13) (h2 : rows_field2 = 16) (h3 : cobs_per_row = 4) : rows_field1 * cobs_per_row + rows_field2 * cobs_per_row = 116 := by
  sorry

end farm_corn_cobs_l381_381635


namespace value_of_a_l381_381338

noncomputable def m := 1 / 5
noncomputable def n := 4 / 5

theorem value_of_a : ∀ (a : ℝ), 
  (m + n = 1) ∧ 
  (∀ m n > 0, 1 + 16 / n ≥ 25) ∧
  (∀ m n, y = x^a → y = 1 / 5 ∧ x = 1 / 25) →
  a = 1 / 2 :=
by
  intros
  sorry

end value_of_a_l381_381338


namespace correct_assignment_l381_381775

structure GirlDressAssignment :=
  (Katya : String)
  (Olya : String)
  (Liza : String)
  (Rita : String)

def solution : GirlDressAssignment :=
  ⟨"Green", "Blue", "Pink", "Yellow"⟩

theorem correct_assignment
  (Katya_not_pink_or_blue : solution.Katya ≠ "Pink" ∧ solution.Katya ≠ "Blue")
  (Green_between_Liza_and_Yellow : 
    (solution.Katya = "Green" ∧ solution.Liza = "Pink" ∧ solution.Rita = "Yellow") ∧
    (solution.Katya = "Green" ∧ solution.Rita = "Yellow" ∧ solution.Liza = "Pink"))
  (Rita_not_green_or_blue : solution.Rita ≠ "Green" ∧ solution.Rita ≠ "Blue")
  (Olya_between_Rita_and_Pink : 
    (solution.Olya = "Blue" ∧ solution.Rita = "Yellow" ∧ solution.Liza = "Pink") ∧
    (solution.Olya = "Blue" ∧ solution.Liza = "Pink" ∧ solution.Rita = "Yellow"))
  : solution = ⟨"Green", "Blue", "Pink", "Yellow"⟩ := by
  sorry

end correct_assignment_l381_381775


namespace find_distance_d_l381_381267

noncomputable def equilateral_triangle_side_length : ℝ := 300

def are_equidistant {A B C P Q O : ℝ → ℝ}
    (PA PB PC QA QB QC d : ℝ)
    (distance_eq_PA : PA = PB) (distance_eq_PB : PB = PC)
    (distance_eq_PC : PC = d) (distance_eq_QA : QA = QB)
    (distance_eq_QB : QB = QC) (distance_eq_QC : QC = d)
    (O_dist_A : O = d) (O_dist_B : O = d) (O_dist_C : O = d)
    (O_dist_P : O = d) (O_dist_Q : O = d) : Prop := 
  PA = PB ∧ PB = PC ∧ PC = d ∧ QA = QB ∧ QB = QC ∧ QC = d

theorem find_distance_d (side_length : ℝ) 
    (A B C P Q O : ℝ → ℝ) 
    (distance_eq_PA : A = d) (distance_eq_PB : B = d) (distance_eq_PC : C = d)
    (distance_eq_QA : P = d) (distance_eq_QB : Q = d) (distance_eq_QC : O = d)
    (right_angle_PAB_QAB : 90 = 90 ) 
    (PA PB PC QA QB QC : ℝ)
    (d : ℝ) :
  are_equidistant PA PB PC QA QB QC d = 100 * (real.sqrt 3) := sorry

end find_distance_d_l381_381267


namespace dylan_trip_time_l381_381693

def total_time_of_trip (d1 d2 d3 v1 v2 v3 b : ℕ) : ℝ :=
  let t1 := d1 / v1
  let t2 := d2 / v2
  let t3 := d3 / v3
  let time_riding := t1 + t2 + t3
  let time_breaks := b * 25 / 60
  time_riding + time_breaks

theorem dylan_trip_time :
  total_time_of_trip 400 150 700 50 40 60 3 = 24.67 :=
by
  unfold total_time_of_trip
  sorry

end dylan_trip_time_l381_381693


namespace initial_boys_count_l381_381118

theorem initial_boys_count (B : ℕ) (boys girls : ℕ)
  (h1 : boys = 3 * B)                             -- The ratio of boys to girls is 3:4
  (h2 : girls = 4 * B)                            -- The ratio of boys to girls is 3:4
  (h3 : boys - 10 = 4 * (girls - 20))             -- The final ratio after transfer is 4:5
  : boys = 90 :=                                  -- Prove initial boys count was 90
by 
  sorry

end initial_boys_count_l381_381118


namespace sum_of_first_110_terms_l381_381072

theorem sum_of_first_110_terms
  (a d : ℤ)
  (S : ℕ → ℤ)
  (h1 : S 10 = 100)
  (h2 : S 100 = 10)
  (h_sum : ∀ n, S n = n * (2 * a + (n - 1) * d) / 2) :
  S 110 = -110 :=
by {
  sorry,
}

end sum_of_first_110_terms_l381_381072


namespace probability_three_correct_letters_l381_381089

-- Define the conditions and the theorem statement
noncomputable def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

noncomputable def binomial_coefficient (n k : ℕ) : ℕ := factorial n / (factorial k * factorial (n - k))

def derangements (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | 1 => 0
  | k => (k - 1) * (derangements (k - 1) + derangements (k - 2))

theorem probability_three_correct_letters :
  let total_permutations := factorial 5,
      choose_three_correct := binomial_coefficient 5 3,
      derange_two := derangements 2,
      favorable_outcomes := choose_three_correct * derange_two
  in favorable_outcomes / total_permutations = 1 / 12 := 
by
  sorry

end probability_three_correct_letters_l381_381089


namespace problem_equivalent_l381_381942

def U : Set ℝ := Set.univ
def M : Set ℝ := {x | x < 1}
def N : Set ℝ := {x | -1 < x ∧ x < 2}

theorem problem_equivalent :
  {x : ℝ | x ≥ 2} = (U \ (M ∪ N)) := 
by sorry

end problem_equivalent_l381_381942


namespace gross_pay_calculation_l381_381889

theorem gross_pay_calculation
    (NetPay : ℕ) (Taxes : ℕ) (GrossPay : ℕ) 
    (h1 : NetPay = 315) 
    (h2 : Taxes = 135) 
    (h3 : GrossPay = NetPay + Taxes) : 
    GrossPay = 450 :=
by
    -- We need to prove this part
    sorry

end gross_pay_calculation_l381_381889


namespace possible_to_fill_grid_l381_381915

/-- Define the grid as a 2D array where each cell contains either 0 or 1. --/
def grid (f : ℕ → ℕ → ℕ) : Prop :=
  ∀ (i j : ℕ), i < 5 → j < 5 → f i j = 0 ∨ f i j = 1

/-- Ensure the sum of every 2x2 subgrid is divisible by 3. --/
def divisible_by_3_in_subgrid (f : ℕ → ℕ → ℕ) : Prop :=
  ∀ i j, i < 4 → j < 4 → (f i j + f (i+1) j + f i (j+1) + f (i+1) (j+1)) % 3 = 0

/-- Ensure both 0 and 1 are present in the grid. --/
def contains_0_and_1 (f : ℕ → ℕ → ℕ) : Prop :=
  (∃ i j, i < 5 ∧ j < 5 ∧ f i j = 0) ∧ (∃ i j, i < 5 ∧ j < 5 ∧ f i j = 1)

/-- The main theorem stating the possibility of such a grid. --/
theorem possible_to_fill_grid :
  ∃ f, grid f ∧ divisible_by_3_in_subgrid f ∧ contains_0_and_1 f :=
sorry

end possible_to_fill_grid_l381_381915


namespace sum_log_identity_l381_381247

noncomputable def log_base (b x : ℝ) : ℝ := Real.log x / Real.log b

theorem sum_log_identity :
  ∑ k in Finset.range (98) + 3, log_base 3 (1 + 1 / k) * log_base k 3 * log_base (k + 1) 3 =
  1 - 1 / log_base 3 101 :=
by
  sorry

end sum_log_identity_l381_381247


namespace sum_of_consecutive_even_numbers_l381_381594

theorem sum_of_consecutive_even_numbers (n : ℕ) (h : (n + 2)^2 - n^2 = 84) :
  n + (n + 2) = 42 :=
sorry

end sum_of_consecutive_even_numbers_l381_381594


namespace integral_x_squared_l381_381907

theorem integral_x_squared (a : ℝ) (h : a > 0) (h_expansion : (binomial_term_coeff (λ x, sqrt(x) + a / x) 6 0 = 60)) :
  ∫ x in (0 : ℝ)..a, x^2 = 8 / 3 :=
by
  sorry

def binomial_term_coeff (f : ℝ → ℝ) (n : ℕ) (term_idx : ℕ) : ℝ :=
  -- | Function to compute the coefficient of the given term in a binomial expansion.
  sorry

end integral_x_squared_l381_381907


namespace trigonometric_identity_l381_381668

theorem trigonometric_identity:
  cos 80 * cos 20 + sin 80 * sin 20 = 1 / 2 :=
by
  sorry

end trigonometric_identity_l381_381668


namespace dark_tiles_fraction_correct_l381_381164

-- Define the conditions
def grid_size : ℕ := 4
def dark_columns : List ℕ := [1, 3]
def dark_positions_row_3 : List ℕ := [1, 3]

-- Define the predicate for dark tiles in the given row and column
def is_dark_tile (row col : ℕ) : Bool :=
  if (col ∈ dark_columns) then true
  else if (row = 3 ∧ col ∈ dark_positions_row_3) then true
  else false

-- The total number of tiles in a 4x4 grid
def total_tiles : ℕ := grid_size * grid_size

-- Calculate the number of dark tiles
def num_dark_tiles : ℕ :=
  let col_dark_tiles := 4 -- Each dark column has 4 dark tiles
  let row_3_dark_tiles := dark_positions_row_3.length
  col_dark_tiles * dark_columns.length + row_3_dark_tiles

-- The fraction of the floor made up of darker tiles
def dark_tiles_fraction : ℚ := num_dark_tiles / total_tiles

theorem dark_tiles_fraction_correct :
  dark_tiles_fraction = (5 : ℚ) / 8 :=
by
  -- Define the number of dark tiles per the given conditions
  have h1 : num_dark_tiles = 10 := by
    simp [num_dark_tiles, dark_columns, dark_positions_row_3]
  -- Define the total number of tiles per the grid size
  have h2 : total_tiles = 16 := by
    simp [total_tiles, grid_size]
  -- Calculate the fraction
  simp [dark_tiles_fraction, h1, h2]
  norm_num
  sorry

end dark_tiles_fraction_correct_l381_381164


namespace zero_of_f_no_zeros_in_intervals_l381_381445

noncomputable def f (x : ℝ) : ℝ := (1 / 3) * x * Real.log x

theorem zero_of_f (x : ℝ) (hx : x > 0) : f x = 0 ↔ x = 1 := by
  sorry

theorem no_zeros_in_intervals (x : ℝ) (hx1 : x ∈ Set.Ioo (1 / Real.exp 1) 1 ∨ x ∈ Set.Ioo 1 (Real.exp 1)) : f x ≠ 0 := by
  intro hc
  have h : x = 1 := (zero_of_f x (hx1.resolve_left $ λ hx, by linarith)).mp hc
  linarith

end zero_of_f_no_zeros_in_intervals_l381_381445


namespace average_speed_Y_l381_381211

-- Definitions based on the conditions
def speed_X := 35 -- Car X's speed in miles per hour
def time_delay_Y := 1.2 -- Time delay for Car Y in hours
def distance_X_when_Y_started := 98 -- Remaining distance for Car X when Car Y started
def total_distance_X := distance_X_when_Y_started + (speed_X * time_delay_Y) -- Total distance for Car X

-- Prove the average speed of Car Y is 35 miles per hour
theorem average_speed_Y : ∃ (speed_Y : ℝ), speed_Y = 35 :=
  sorry

end average_speed_Y_l381_381211


namespace reciprocal_in_fourth_quadrant_l381_381906

-- Define the conditions in Lean 4
variables {a b : ℝ}

-- Main statement: Prove the location of the reciprocal of F
theorem reciprocal_in_fourth_quadrant (ha : a > 0) (hb : b > 0) (hab_gt1 : a^2 + b^2 > 1) :
  let F := complex.mk a b in
  let F_inv := 1 / F in
  F_inv.re > 0 ∧ F_inv.im < 0 ∧ complex.norm F_inv < 1 :=
by
  let F := complex.mk a b,
  let F_inv := 1 / F,
  sorry

end reciprocal_in_fourth_quadrant_l381_381906


namespace solve_eqn_l381_381041

theorem solve_eqn (x y : ℕ) (hx : 0 < x) (hy : 0 < y) :
  3 ^ x = 2 ^ x * y + 1 ↔ (x = 1 ∧ y = 1) ∨ (x = 2 ∧ y = 2) ∨ (x = 4 ∧ y = 5) := by
  sorry

end solve_eqn_l381_381041


namespace farthest_point_from_origin_l381_381112

/-- Define the distance function from the origin to a point in 2D. -/
def distFromOrigin (p : ℝ × ℝ) : ℝ :=
  real.sqrt (p.1^2 + p.2^2)

/-- Define the points to be considered. -/
def points : List (ℝ × ℝ) := [(-3, 4), (2, -3), (-5, 0), (0, -6), (4, 1)]

/-- Statement that (0, -6) is the farthest point from the origin among the given points. -/
theorem farthest_point_from_origin :
  (0, -6) ∈ points ∧ ∀ p ∈ points, distFromOrigin p ≤ distFromOrigin (0, -6) :=
by
  sorry

end farthest_point_from_origin_l381_381112


namespace sum_log_identity_l381_381248

noncomputable def log_base (b x : ℝ) : ℝ := Real.log x / Real.log b

theorem sum_log_identity :
  ∑ k in Finset.range (98) + 3, log_base 3 (1 + 1 / k) * log_base k 3 * log_base (k + 1) 3 =
  1 - 1 / log_base 3 101 :=
by
  sorry

end sum_log_identity_l381_381248


namespace sum_first_15_terms_l381_381796

-- Define an arithmetic sequence with a common difference d
variables (a : ℕ → ℝ) (d : ℝ)

-- Define the condition that d is not zero
axiom d_nonzero : d ≠ 0

-- Define the condition given in the problem 
axiom cond : (a 5)^2 + (a 7)^2 + 16 * d = (a 9)^2 + (a 11)^2

-- Define the sum of the first 15 terms of the arithmetic sequence
def S_15 := (15 / 2) * (2 * a 1 + 14 * d)

-- State the theorem to prove that the sum of the first 15 terms is 15
theorem sum_first_15_terms : S_15 a d = 15 :=
sorry

end sum_first_15_terms_l381_381796


namespace problem_statement_l381_381980

open Set

variable (U : Set ℝ) (M N : Set ℝ)

theorem problem_statement (hU : U = univ) (hM : M = {x | x < 1}) (hN : N = {x | -1 < x ∧ x < 2}) :
  {x | 2 ≤ x} = compl (M ∪ N) :=
sorry

end problem_statement_l381_381980


namespace monthly_production_increase_l381_381163

/-- A salt manufacturing company produced 3000 tonnes in January and increased its
    production by some tonnes every month over the previous month until the end
    of the year. Given that the average daily production was 116.71232876712328 tonnes,
    determine the monthly production increase. -/
theorem monthly_production_increase :
  let initial_production := 3000
  let daily_average_production := 116.71232876712328
  let days_per_year := 365
  let total_yearly_production := daily_average_production * days_per_year
  let months_per_year := 12
  ∃ (x : ℝ), total_yearly_production = (months_per_year / 2) * (2 * initial_production + (months_per_year - 1) * x) → x = 100 :=
sorry

end monthly_production_increase_l381_381163


namespace find_number_l381_381624

theorem find_number (x : ℝ) (h : 0.4 * x + 60 = x) : x = 100 :=
by
  sorry

end find_number_l381_381624


namespace two_point_form_eq_l381_381532

theorem two_point_form_eq (x y : ℝ) : 
  let A := (5, 6)
  let B := (-1, 2)
  (y - 6) / (2 - 6) = (x - 5) / (-1 - 5) := 
  sorry

end two_point_form_eq_l381_381532


namespace complement_union_eq_l381_381965

-- Definitions / Conditions
def U : Set ℝ := Set.univ
def M : Set ℝ := { x | x < 1 }
def N : Set ℝ := { x | -1 < x ∧ x < 2 }

-- Statement of the theorem
theorem complement_union_eq {x : ℝ} :
  {x | x ≥ 2} = (U \ (M ∪ N)) := sorry

end complement_union_eq_l381_381965


namespace count_three_digit_numbers_divisible_by_7_l381_381851

def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

def is_divisible_by_7 (n : ℕ) : Prop :=
  n % 7 = 0

def count_three_digit_divisible_by_7 : ℕ :=
  (list.range' 105 890).countp (λ x, is_divisible_by_7 (x * 7))

theorem count_three_digit_numbers_divisible_by_7 :
  count_three_digit_divisible_by_7 = 128 :=
sorry

end count_three_digit_numbers_divisible_by_7_l381_381851


namespace solution_set_l381_381827

def f(x : ℝ) : ℝ := 2016^x + Real.logBase 2016 (Real.sqrt (x^2 + 1) + x) - 2016^(-x) + 2

theorem solution_set (x : ℝ) : f(3*x + 1) + f(x) > 4 ↔ x > -1/4 := 
by
  sorry

end solution_set_l381_381827


namespace value_of_expression_l381_381869

variables (a b c d m : ℝ)

theorem value_of_expression (h1: a + b = 0) (h2: c * d = 1) (h3: |m| = 3) :
  (a + b) / m + m^2 - c * d = 8 :=
by
  sorry

end value_of_expression_l381_381869


namespace max_students_per_class_l381_381540

theorem max_students_per_class (num_students : ℕ) (seats_per_bus : ℕ) (num_buses : ℕ) (k : ℕ) 
  (h_num_students : num_students = 920) 
  (h_seats_per_bus : seats_per_bus = 71) 
  (h_num_buses : num_buses = 16) 
  (h_class_size_bound : ∀ c, c ≤ k) : 
  k = 17 :=
sorry

end max_students_per_class_l381_381540


namespace allison_total_craft_supplies_l381_381177

theorem allison_total_craft_supplies (M_glue : ℕ) (M_paper : ℕ) (A_glue_more : ℕ) (M_paper_ratio : ℕ) :
  M_glue = 15 → M_paper = 30 → A_glue_more = 8 → M_paper_ratio = 6 →
  let A_glue := M_glue + A_glue_more in
  let A_paper := M_paper / M_paper_ratio in
  A_glue + A_paper = 28 :=
by
  intros h1 h2 h3 h4
  let A_glue := 15 + 8
  let A_paper := 30 / 6
  sorry

end allison_total_craft_supplies_l381_381177


namespace at_least_one_sum_of_three_l381_381468

theorem at_least_one_sum_of_three (S : Finset ℕ) (hS₁ : S.card = 68) (hS₂ : ∀ x ∈ S, x < 100) :
  ∃ a b c d ∈ S, a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ a = b + c + d := 
sorry

end at_least_one_sum_of_three_l381_381468


namespace four_digit_sum_distinct_remainder_l381_381618

theorem four_digit_sum_distinct_remainder :
  (∑ x in (Finset.filter (λ n : ℕ, ∀ d1 d2 ∈ digits 10 n, d1 ≠ d2 ∧ 1000 ≤ n ∧ n < 10000) 
    (Finset.range 10000)), x) % 1000 = 720 := by
  sorry

end four_digit_sum_distinct_remainder_l381_381618


namespace complement_of_A_in_U_l381_381837

-- Define the universal set U
def U : Set ℝ := {x | -1 < x ∧ x ≤ 1}

-- Define the set A
def A : Set ℝ := {x | 1 / x ≥ 1}

-- Define the complement of A within U
def complement_U_A : Set ℝ := {x | x ∈ U ∧ x ∉ A}

-- Theorem statement
theorem complement_of_A_in_U : complement_U_A = {x | -1 < x ∧ x ≤ 0} :=
  sorry

end complement_of_A_in_U_l381_381837


namespace quadrant_of_conjugate_l381_381817

theorem quadrant_of_conjugate (z : ℂ) (H : z = (i ^ 2016) / (1 - i)) :
  let z_conj := conj z in
  z_conj.re > 0 ∧ z_conj.im < 0 :=
by
  sorry

end quadrant_of_conjugate_l381_381817


namespace possible_strings_after_moves_l381_381036

theorem possible_strings_after_moves : 
  let initial_string := "HHMMMMTT"
  let moves := [("HM", "MH"), ("MT", "TM"), ("TH", "HT")]
  let binom := Nat.choose 8 4
  binom = 70 := by
  sorry

end possible_strings_after_moves_l381_381036


namespace solve_inequality_a_half_range_of_a_always_nonnegative_l381_381351

noncomputable def f (a x : ℝ) : ℝ := a * x^2 - |x - 1| + 2 * a

theorem solve_inequality_a_half : 
  ∀ x : ℝ, 1/2 * x^2 - |x - 1| + 1 ≥ 0 ↔ (x ≥ 0 ∨ x ≤ -2) :=
begin
  sorry
end

theorem range_of_a_always_nonnegative : 
  ∀ a x : ℝ, a * x^2 - |x - 1| + 2 * a ≥ 0 → a ≥ (Real.sqrt 3 + 1) / 4 :=
begin
  sorry
end

end solve_inequality_a_half_range_of_a_always_nonnegative_l381_381351


namespace lines_intersect_l381_381688

theorem lines_intersect (m : ℝ) : ∃ (x y : ℝ), 3 * x + 2 * y + m = 0 ∧ (m^2 + 1) * x - 3 * y - 3 * m = 0 := 
by {
  sorry
}

end lines_intersect_l381_381688


namespace monotonicity_intervals_range_of_m_l381_381344

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := Real.exp x - (m * x^2 + x + 1)

-- Question (1)
theorem monotonicity_intervals (x : ℝ) :
  ∀ (x : ℝ), (f x 0 = Real.exp x - x - 1) ∧
    ((∀ x ∈ Iio (0 : ℝ), f' x < 0) ∧ (∀ x ∈ Ioi (0 : ℝ), f' x > 0)) := sorry

-- Question (2)
theorem range_of_m (m : ℝ) (x : ℝ) :
  (∀ x ∈ Ici 0, f x m ≥ 0) → m ∈ Set.Iic (1/2) := sorry

end monotonicity_intervals_range_of_m_l381_381344


namespace minimum_order_amount_to_get_discount_l381_381480

theorem minimum_order_amount_to_get_discount 
  (cost_quiche : ℝ) (cost_croissant : ℝ) (cost_biscuit : ℝ) (n_quiches : ℝ) (n_croissants : ℝ) (n_biscuits : ℝ)
  (discount_percent : ℝ) (total_with_discount : ℝ) (min_order_amount : ℝ) :
  cost_quiche = 15.0 → cost_croissant = 3.0 → cost_biscuit = 2.0 →
  n_quiches = 2 → n_croissants = 6 → n_biscuits = 6 →
  discount_percent = 0.10 → total_with_discount = 54.0 →
  (n_quiches * cost_quiche + n_croissants * cost_croissant + n_biscuits * cost_biscuit) * (1 - discount_percent) = total_with_discount →
  min_order_amount = 60.0 :=
by
  sorry

end minimum_order_amount_to_get_discount_l381_381480


namespace sum_log_diff_eq_1998953_l381_381674

def ceil_floor_diff (x : ℝ) : ℝ :=
  ⌈x⌉ - ⌊x⌋

theorem sum_log_diff_eq_1998953 : 
  (∑ k in Finset.range 2000, (k+1) * (ceil_floor_diff (Real.log (k+1) / Real.log 2))) = 1998953 := by
  sorry

end sum_log_diff_eq_1998953_l381_381674


namespace average_salary_of_all_workers_l381_381049

-- Definitions of conditions
def num_technicians : ℕ := 7
def num_total_workers : ℕ := 12
def num_other_workers : ℕ := num_total_workers - num_technicians

def avg_salary_technicians : ℝ := 12000
def avg_salary_others : ℝ := 6000

-- Total salary calculations
def total_salary_technicians : ℝ := num_technicians * avg_salary_technicians
def total_salary_others : ℝ := num_other_workers * avg_salary_others

def total_salary : ℝ := total_salary_technicians + total_salary_others

-- Proof statement: the average salary of all workers is 9500
theorem average_salary_of_all_workers : total_salary / num_total_workers = 9500 :=
by
  sorry

end average_salary_of_all_workers_l381_381049


namespace complement_union_eq_ge2_l381_381951

open Set

variables {U : Type} [PartialOrder U] [LinearOrder U]

def U : Set ℝ := univ
def M : Set ℝ := { x | x < 1 }
def N : Set ℝ := { x | -1 < x ∧ x < 2 }
def Complement_U (A : Set ℝ) : Set ℝ := U \ A

theorem complement_union_eq_ge2 : 
  Complement_U (M ∪ N) = { x : ℝ | x ≥ 2 } :=
by {
  sorry
}

end complement_union_eq_ge2_l381_381951


namespace cyclic_quadrilateral_l381_381382

variables {A B C D E F X Y Z : Point}

-- Definitions of points and conditions
def triangle_incircle_touches (A B C D E F : Point) : Prop :=
  incircle_of_triangle A B C touches B C at D ∧
  incircle_of_triangle A B C touches C A at E ∧
  incircle_of_triangle A B C touches A B at F

def interior_point (X : Point) (ABC : Triangle) : Prop :=
  X lies_in_triangle ABC

def incircle_touches_specific_points (X B C : Point) (D Y Z : Point) : Prop :=
  incircle_of_triangle X B C touches B C at D ∧
  incircle_of_triangle X B C touches C X at Y ∧
  incircle_of_triangle X B C touches X B at Z

-- The proof statement
theorem cyclic_quadrilateral {A B C D E F X Y Z : Point} :
  triangle_incircle_touches A B C D E F →
  interior_point X (triangle A B C) →
  incircle_touches_specific_points X B C D Y Z →
  cyclic_quadrilateral E F Z Y :=
sorry

end cyclic_quadrilateral_l381_381382


namespace domain_of_g_l381_381882

-- Definitions from the problem's conditions
def f : ℝ → ℝ := sorry  -- Assuming generic function f with real inputs and outputs

axiom domain_f_x_add_1 : ∀ x, f(x+1) ≠ none ↔ x ∈ Ioo (-2 : ℝ) 2

-- The statement to prove
theorem domain_of_g : ∀ x, (g(x) ≠ none ↔ x ∈ Ioo 0 3) :=
by
  -- Define g(x)
  let g (x : ℝ) : ℝ := f x / real.sqrt x
  sorry

end domain_of_g_l381_381882


namespace common_ratio_three_l381_381426

-- Definition of a geometric sequence sums
noncomputable def geometric_sum (a q : ℝ) (n : ℕ) : ℝ :=
  if q = 1 then n * a else a * (1 - q^n) / (1 - q)

-- Conditions for the problem
def S (a q : ℝ) (n : ℕ) : ℝ := geometric_sum a q n

-- Theorem statement for proving the common ratio q is 3 given S₄ = 10S₂
theorem common_ratio_three (a : ℝ) (q : ℝ) (h_pos: a > 0) (h_eq: S a q 4 = 10 * S a q 2) : q = 3 :=
by
  sorry

end common_ratio_three_l381_381426


namespace number_of_color_copies_l381_381309

def charge_shop_X (n : ℕ) : ℝ := 1.20 * n
def charge_shop_Y (n : ℕ) : ℝ := 1.70 * n
def difference := 20

theorem number_of_color_copies (n : ℕ) (h : charge_shop_Y n = charge_shop_X n + difference) : n = 40 :=
by {
  sorry
}

end number_of_color_copies_l381_381309


namespace who_wears_which_dress_l381_381735

def girls := ["Katya", "Olya", "Liza", "Rita"]
def dresses := ["pink", "green", "yellow", "blue"]

variable (who_wears_dress : String → String)

/-- Conditions given in the problem --/
axiom Katya_not_pink_blue : who_wears_dress "Katya" ≠ "pink" ∧ who_wears_dress "Katya" ≠ "blue"
axiom between_green_liza_yellow : ∃ g, who_wears_dress "Katya" = "green" ∧ who_wears_dress "Rita" = "yellow"
axiom Rita_not_green_blue : who_wears_dress "Rita" ≠ "green" ∧ who_wears_dress "Rita" ≠ "blue"
axiom Olya_between_rita_pink : ∃ o, who_wears_dress "Olya" = "blue" ∧ who_wears_dress "Liza" = "pink"

theorem who_wears_which_dress :
  who_wears_dress "Katya" = "green" ∧
  who_wears_dress "Olya" = "blue" ∧
  who_wears_dress "Liza" = "pink" ∧
  who_wears_dress "Rita" = "yellow" :=
by
  sorry

end who_wears_which_dress_l381_381735


namespace projection_of_a_in_direction_of_b_l381_381430

variables (e1 e2 : ℝ)
variables (a b : ℝ)

-- Define unit vectors and their properties.
def unit_vector (v : ℝ) : Prop := v = 1
def angle_between_vectors (v1 v2 : ℝ) (θ : ℝ) : Prop := θ = Real.pi / 3

-- Define vectors a and b
def a : ℝ := e1 + 3 * e2
def b : ℝ := 2 * e1

theorem projection_of_a_in_direction_of_b
  (he1 : unit_vector e1)
  (he2 : unit_vector e2)
  (hab : angle_between_vectors e1 e2 (Real.pi / 3)) :
  (a * b) / Real.sqrt (b * b) = 5 / 2 := by
  sorry

end projection_of_a_in_direction_of_b_l381_381430


namespace incircle_radius_of_right_triangle_l381_381098

theorem incircle_radius_of_right_triangle (A B C : Type*) [EuclideanGeometry A] [EuclideanGeometry B] [EuclideanGeometry C] 
  (h_right : is_right_triangle A B C)
  (angle_A : angle A B C = 60 * pi / 180)
  (AC : length A C = 12) :
  incircle_radius A B C = 12 * (4 * sqrt 3 - 3) / 13 :=
begin
  sorry
end

end incircle_radius_of_right_triangle_l381_381098


namespace game_cost_l381_381777

theorem game_cost (total_earned : ℕ) (spent_on_blades : ℕ) (games : ℕ) 
    (h1 : total_earned = 19) 
    (h2 : spent_on_blades = 11) 
    (h3 : games = 4) : 
    (total_earned - spent_on_blades) / games = 2 :=
by 
    rw [h1, h2, h3]
    norm_num
    sorry

end game_cost_l381_381777


namespace sum_log_identity_l381_381244

noncomputable def log_base (b x : ℝ) : ℝ := Real.log x / Real.log b

theorem sum_log_identity :
  ∑ k in Finset.range (98) + 3, log_base 3 (1 + 1 / k) * log_base k 3 * log_base (k + 1) 3 =
  1 - 1 / log_base 3 101 :=
by
  sorry

end sum_log_identity_l381_381244


namespace complement_union_eq_l381_381963

-- Definitions / Conditions
def U : Set ℝ := Set.univ
def M : Set ℝ := { x | x < 1 }
def N : Set ℝ := { x | -1 < x ∧ x < 2 }

-- Statement of the theorem
theorem complement_union_eq {x : ℝ} :
  {x | x ≥ 2} = (U \ (M ∪ N)) := sorry

end complement_union_eq_l381_381963


namespace students_neither_music_nor_art_l381_381114

theorem students_neither_music_nor_art : 
  ∀ (total_students music_students art_students both_students : ℕ),
    total_students = 500 → 
    music_students = 30 → 
    art_students = 20 → 
    both_students = 10 → 
    total_students - (music_students + art_students - both_students) = 460 :=
by
  intros total_students music_students art_students both_students ht hm ha hb
  rw [ht, hm, ha, hb]
  sorry

end students_neither_music_nor_art_l381_381114


namespace sequence_perfect_square_l381_381013

def sequence (a : ℕ → ℕ) : Prop :=
  a 0 = 1 ∧ a 1 = 1 ∧ (∀ n, a (n + 1) = 7 * a n - a (n - 1) - 2)

theorem sequence_perfect_square (a : ℕ → ℕ) (h : sequence a) : ∀ n, (∃ k : ℕ, a n = k^2) :=
by
  sorry

end sequence_perfect_square_l381_381013


namespace nested_radicals_solution_l381_381297

theorem nested_radicals_solution (x : ℝ) (h : 0 < x)
    (h1 : sqrt (x + sqrt (x + sqrt (x + ...))) = sqrt (x * sqrt (x * sqrt (x * ...))))
    : x = 2 :=
begin
    sorry
end

end nested_radicals_solution_l381_381297


namespace solve_for_x_l381_381040

-- Definitions based on conditions
def sixteen_power_condition (x : ℝ) : Prop := (16^x * 16^x * 16^x * 16^x = 256^5)

-- Theorem statement
theorem solve_for_x (x : ℝ) (h : sixteen_power_condition x) : x = 5 / 2 :=
by
  sorry

end solve_for_x_l381_381040


namespace math_problem_l381_381864

theorem math_problem 
  (x : ℝ)
  (h : x + sqrt (x^2 - 4) + 1 / (x - sqrt (x^2 - 4)) = 25) :
  x^2 + sqrt (x^4 - 4) + 1 / (x^2 - sqrt (x^4 - 4)) = 82.1762 := 
sorry

end math_problem_l381_381864


namespace shopkeeper_loss_percent_l381_381633

theorem shopkeeper_loss_percent (cost_price goods_lost_percent profit_percent : ℝ)
    (h_cost_price : cost_price = 100)
    (h_goods_lost_percent : goods_lost_percent = 0.4)
    (h_profit_percent : profit_percent = 0.1) :
    let initial_revenue := cost_price * (1 + profit_percent)
    let goods_lost_value := cost_price * goods_lost_percent
    let remaining_goods_value := cost_price - goods_lost_value
    let remaining_revenue := remaining_goods_value * (1 + profit_percent)
    let loss_in_revenue := initial_revenue - remaining_revenue
    let loss_percent := (loss_in_revenue / initial_revenue) * 100
    loss_percent = 40 := sorry

end shopkeeper_loss_percent_l381_381633


namespace fraction_of_fraction_l381_381562

theorem fraction_of_fraction (a b c d : ℚ) (h1 : a = 2) (h2 : b = 9) (h3 : c = 3) (h4 : d = 4) :
  (a / b) / (c / d) = (a * d) / (b * c) :=
by
  rw [div_mul_div, mul_comm c _] -- Using properties of divisions and multiplications.
  sorry

end fraction_of_fraction_l381_381562


namespace tiffany_homework_l381_381553

theorem tiffany_homework : 
  let x := 3 in
  6 * x + 4 * x = 30 :=
by
  sorry

end tiffany_homework_l381_381553


namespace travel_distance_and_cost_l381_381465

theorem travel_distance_and_cost (AC AB : ℝ) (h₀ : AC = 4000) (h₁ : AB = 4500) :
  let BC := (√(AB^2 - AC^2))
  let total_distance := AC + AB + BC
  let cost_AC_bus := 0.20 * AC
  let cost_AB_airplane := 150 + 0.12 * AB
  let cost_BC_bus := 0.20 * BC
  let min_total_cost := cost_AC_bus + cost_AB_airplane + cost_BC_bus
  total_distance = 10561.55 ∧ min_total_cost = 1902.31 :=
by 
  -- The proof goes here
  sorry

end travel_distance_and_cost_l381_381465


namespace telescoping_log_sum_l381_381238

theorem telescoping_log_sum :
  ∑ k in Finset.range 98 \ Finset.range 2, (Real.log (1 + 1 / k) / Real.log 3) * (Real.log 3 / Real.log k) * (Real.log 3 / Real.log (k + 1)) = 1 - 1 / Real.log 101 :=
by
  sorry

end telescoping_log_sum_l381_381238


namespace alice_decorates_140_sqft_l381_381657

theorem alice_decorates_140_sqft (total_area : ℕ) (a_ratio c_ratio : ℕ) (total_ratio : a_ratio + c_ratio = 5) : 
  total_area = 350 → a_ratio = 2 → c_ratio = 3 → 
  ∃ alice_area : ℕ, alice_area = 140 ∧ alice_area = total_area * a_ratio / total_ratio :=
by
  intros h_total_area h_a_ratio h_c_ratio
  rw [h_total_area, h_a_ratio, h_c_ratio]
  use 140
  split
  { refl }
  { norm_num }

end alice_decorates_140_sqft_l381_381657


namespace correct_assignment_l381_381772

structure GirlDressAssignment :=
  (Katya : String)
  (Olya : String)
  (Liza : String)
  (Rita : String)

def solution : GirlDressAssignment :=
  ⟨"Green", "Blue", "Pink", "Yellow"⟩

theorem correct_assignment
  (Katya_not_pink_or_blue : solution.Katya ≠ "Pink" ∧ solution.Katya ≠ "Blue")
  (Green_between_Liza_and_Yellow : 
    (solution.Katya = "Green" ∧ solution.Liza = "Pink" ∧ solution.Rita = "Yellow") ∧
    (solution.Katya = "Green" ∧ solution.Rita = "Yellow" ∧ solution.Liza = "Pink"))
  (Rita_not_green_or_blue : solution.Rita ≠ "Green" ∧ solution.Rita ≠ "Blue")
  (Olya_between_Rita_and_Pink : 
    (solution.Olya = "Blue" ∧ solution.Rita = "Yellow" ∧ solution.Liza = "Pink") ∧
    (solution.Olya = "Blue" ∧ solution.Liza = "Pink" ∧ solution.Rita = "Yellow"))
  : solution = ⟨"Green", "Blue", "Pink", "Yellow"⟩ := by
  sorry

end correct_assignment_l381_381772


namespace gasoline_tank_capacity_l381_381145

theorem gasoline_tank_capacity :
  ∀ (x : ℕ), (5 / 6 * (x : ℚ) - 18 = 1 / 3 * (x : ℚ)) → x = 36 :=
by
  sorry

end gasoline_tank_capacity_l381_381145


namespace price_of_pants_is_120_l381_381452

variable {P S : ℝ}

def price_of_pants (P : ℝ) : Prop :=
  let S := (3 / 4) * P in
  S + P + (P + 10) = 340

theorem price_of_pants_is_120 : price_of_pants 120 :=
by
  sorry

end price_of_pants_is_120_l381_381452


namespace sum_of_first_110_terms_l381_381071

theorem sum_of_first_110_terms
  (a d : ℤ)
  (S : ℕ → ℤ)
  (h1 : S 10 = 100)
  (h2 : S 100 = 10)
  (h_sum : ∀ n, S n = n * (2 * a + (n - 1) * d) / 2) :
  S 110 = -110 :=
by {
  sorry,
}

end sum_of_first_110_terms_l381_381071


namespace poly_solution_l381_381368

noncomputable def poly_Γ (p : ℝ → ℝ) : ℝ := sorry

theorem poly_solution (g : ℝ → ℝ) : 
  (g 0 = 1) ∧ (∀ n : ℕ, n ≥ 1 → poly_Γ (λ x, (3 * x^2 + 7 * x + 2)^n) = poly_Γ (λ x, g x^n))
  → (g = λ x, √61 * x + 1 ∨ g = λ x, -√61 * x + 1) :=
sorry

end poly_solution_l381_381368


namespace find_vertical_shift_l381_381206

theorem find_vertical_shift (A B C D : ℝ) (h1 : ∀ x, -3 ≤ A * Real.cos (B * x + C) + D ∧ A * Real.cos (B * x + C) + D ≤ 5) :
  D = 1 :=
by
  -- Here's where the proof would go
  sorry

end find_vertical_shift_l381_381206


namespace balls_in_base_l381_381454

theorem balls_in_base (n k : ℕ) (h1 : 165 = (n * (n + 1) * (n + 2)) / 6) (h2 : k = n * (n + 1) / 2) : k = 45 := 
by 
  sorry

end balls_in_base_l381_381454


namespace hyperbola_eccentricity_l381_381934

theorem hyperbola_eccentricity
  (a b : ℝ)
  (h : a > 0)
  (hb : b > 0)
  (x y : ℝ)
  (h_hyperbola : x^2 / a^2 - y^2 / b^2 = 1)
  (P Q F1 F2 : ℝ × ℝ)
  (h_PQ_right_branch : P ∈ { (x, y) : ℝ × ℝ | x^2 / a^2 - y^2 / b^2 = 1 ∧ x > 0})
  (h_QF2_relation : ∃ (m : ℝ), F2 = (0, 0) ∧ |Q - F2| = m ∧ |P - F2| = 2 * m)
  (h_F1Q_perpendicular : ∀ d, F1 = (d, 0) → (Q - F1) ⬝ (P - Q) = 0) :
  ∃ e : ℝ, e = sqrt 17 / 3 := sorry

end hyperbola_eccentricity_l381_381934


namespace smallest_number_property_l381_381111

theorem smallest_number_property : 
  ∃ n, ((n - 7) % 12 = 0) ∧ ((n - 7) % 16 = 0) ∧ ((n - 7) % 18 = 0) ∧ ((n - 7) % 21 = 0) ∧ ((n - 7) % 28 = 0) ∧ n = 1015 :=
by
  sorry  -- Proof is omitted

end smallest_number_property_l381_381111


namespace equal_cost_per_copy_l381_381599

theorem equal_cost_per_copy 
    (x : ℕ) 
    (h₁ : 2000 % x = 0) 
    (h₂ : 3000 % (x + 50) = 0) 
    (h₃ : 2000 / x = 3000 / (x + 50)) :
    (2000 : ℕ) / x = (3000 : ℕ) / (x + 50) :=
by
  sorry

end equal_cost_per_copy_l381_381599


namespace sum_and_product_of_three_numbers_l381_381596

variables (a b c : ℝ)

-- Conditions
axiom h1 : a + b = 35
axiom h2 : b + c = 47
axiom h3 : c + a = 52

-- Prove the sum and product
theorem sum_and_product_of_three_numbers : a + b + c = 67 ∧ a * b * c = 9600 :=
by {
  sorry
}

end sum_and_product_of_three_numbers_l381_381596


namespace floor_T_eq_100_l381_381924

noncomputable def T : ℝ :=
  ∑ i in Finset.range 100 | (i ≠ 0), real.sqrt (1 + (2 / (i : ℝ)^2 + 2 / ((i + 1) : ℝ)^2)^2)

theorem floor_T_eq_100 : (⌊T⌋ : ℤ) = 100 :=
  by
    sorry

end floor_T_eq_100_l381_381924


namespace line_on_plane_subset_l381_381711

variables (l α : Set)

-- Assume l is a line and α is a plane, and l is on the plane α
axiom line_on_plane : l ⊆ α

-- The given problem in Lean
theorem line_on_plane_subset : l ⊆ α :=
by exact line_on_plane

end line_on_plane_subset_l381_381711


namespace cost_of_cookbook_l381_381418

def cost_of_dictionary : ℕ := 11
def cost_of_dinosaur_book : ℕ := 19
def amount_saved : ℕ := 8
def amount_needed : ℕ := 29

theorem cost_of_cookbook :
  let total_cost := amount_saved + amount_needed
  let accounted_cost := cost_of_dictionary + cost_of_dinosaur_book
  total_cost - accounted_cost = 7 :=
by
  sorry

end cost_of_cookbook_l381_381418


namespace broken_line_AEC_correct_l381_381419

noncomputable def length_of_broken_line_AEC 
  (side_length : ℝ)
  (height_of_pyramid : ℝ)
  (radius_of_equiv_circle : ℝ) 
  (length_AE : ℝ)
  (length_AEC : ℝ) : Prop :=
  side_length = 230.0 ∧
  height_of_pyramid = 146.423 ∧
  radius_of_equiv_circle = height_of_pyramid ∧
  length_AE = ((230.0 * 186.184) / 218.837) ∧
  length_AEC = 2 * length_AE ∧
  round (length_AEC * 100) = 39136

theorem broken_line_AEC_correct :
  length_of_broken_line_AEC 230 146.423 (146.423) 195.681 391.362 :=
by
  sorry

end broken_line_AEC_correct_l381_381419


namespace drew_marbles_difference_l381_381690

def initial_marbles_difference (D M J X : ℕ) (hD: D = 200) (hM: M = 45) (hJ: J = 70)
  (h1 : (1/4 : ℚ) * D + M = X)
  (h2 : (1/8 : ℚ) * D + J = X) : Prop :=
  D - M = 155

theorem drew_marbles_difference :
  ∀ (D M J X : ℕ), D = 200 → M = 45 → J = 70 →
  ((1/4 : ℚ) * D + M = X) →
  ((1/8 : ℚ) * D + J = X) →
  initial_marbles_difference D M J X :=
by
  intros D M J X hD hM hJ h1 h2
  unfold initial_marbles_difference
  rw [hD, hM, hJ]
  apply Eq.refl 155

-- This concludes the statement, skipping the proof with 'sorry' if necessary.

end drew_marbles_difference_l381_381690


namespace fifth_odd_multiple_of_5_lt_100_is_45_l381_381572

theorem fifth_odd_multiple_of_5_lt_100_is_45 :
  ∃ x : ℕ, x = 10 * 5 - 5 ∧ x % 2 = 1 ∧ x % 5 = 0 ∧ x < 100 :=
by
  use 45
  split
  any_goals {"refl"}
  any_goals {sorry}

end fifth_odd_multiple_of_5_lt_100_is_45_l381_381572


namespace part1_real_roots_part2_distinct_positive_integer_roots_l381_381822

noncomputable def equation := λ (m x : ℝ), m * x^2 - (m + 2) * x + 2 = 0

theorem part1_real_roots (m : ℝ) : ∃ x₁ x₂ : ℝ, equation m x₁ ∧ equation m x₂ := by
  sorry

theorem part2_distinct_positive_integer_roots (m : ℤ) : 
  (∃ x₁ x₂ : ℤ, x₁ ≠ x₂ ∧ x₁ > 0 ∧ x₂ > 0 ∧ equation m (x₁ : ℝ) ∧ equation m (x₂ : ℝ)) ↔ (m = 1) := by
  sorry

end part1_real_roots_part2_distinct_positive_integer_roots_l381_381822


namespace solve_for_x_l381_381483

theorem solve_for_x (x : ℝ) : 
  27^x * 27^x * 27^x = 81^4 → x = 16 / 9 :=
by
  intro h
  sorry

end solve_for_x_l381_381483


namespace max_angle_MBA_l381_381830

noncomputable def parabola := { p : ℝ × ℝ | p.2^2 = 4 * p.1 }
def point_A := (1 : ℝ, 0 : ℝ)
def point_B := (-1 : ℝ, 0 : ℝ)

theorem max_angle_MBA (M : ℝ × ℝ) (hM : M ∈ parabola) :
  ∃θ, θ = Real.pi / 4 :=
by sorry

end max_angle_MBA_l381_381830


namespace max_area_isosceles_triangle_l381_381798

def isosceles_triangle (A B C : Point) : Prop :=
  (distance A B = distance A C) ∨ (distance B A = distance B C) ∨ (distance C A = distance C B)

def median (A B C D : Point) : Prop :=
  collinear A C D ∧ distance A D = distance D C

noncomputable def distance (A B : Point) : ℝ := sorry -- Assume distance is defined

-- Definition of a Point
structure Point := 
  (x : ℝ)
  (y : ℝ)

-- Define the condition for maximum area
noncomputable def max_area_condition (A B C D : Point) : Prop := 
  isosceles_triangle A B C ∧ 
  median A C D ∧
  distance B D = 3 ∧ 
  (∃ h : ℝ, ⟨0, 1⟩ = vector_of_point_to_line AC BD ⟨h, 0⟩)

theorem max_area_isosceles_triangle (A B C D : Point) 
  (h_ABC_isosceles : isosceles_triangle A B C)
  (h_median : median A C D)
  (h_BD_length : distance B D = 3) 
  (h_max_area_condition : max_area_condition A B C D) :
  ∃ area, area = ((BD_length) / 2) * AC_length := sorry

end max_area_isosceles_triangle_l381_381798


namespace functional_equation_solution_l381_381279

theorem functional_equation_solution :
  ∀ (f : ℝ → ℝ),
  (∀ x : ℝ, x ≠ 0 ∧ x ≠ 1 → f(x) + f(1 / (1 - x)) = x) →
  (∀ x : ℝ, x ≠ 0 ∧ x ≠ 1 → f(x) = 1 / 2 * (x + 1 - 1 / x - 1 / (1 - x))) :=
by
  intros f h x hx,
  sorry

end functional_equation_solution_l381_381279


namespace nelly_friends_l381_381459

noncomputable def total_earnings (nights: ℕ) (earnings_per_night: ℕ): ℕ := nights * earnings_per_night

noncomputable def pizzas_can_afford (total_earnings: ℕ) (pizza_cost: ℕ): ℕ := total_earnings / pizza_cost

noncomputable def total_people_fed (pizzas: ℕ) (people_per_pizza: ℕ): ℕ := pizzas * people_per_pizza

noncomputable def friends_nelly_buys_for (total_people: ℕ): ℕ := total_people - 1

theorem nelly_friends
  (pizza_cost: ℕ) 
  (people_per_pizza: ℕ)
  (earnings_per_night: ℕ)
  (nights: ℕ)
  (total_earnings: ℕ)
  (pizzas: ℕ)
  (total_people: ℕ):
  pizza_cost = 12 -> 
  people_per_pizza = 3 ->
  earnings_per_night = 4 -> 
  nights = 15 ->  
  total_earnings = total_earnings nights earnings_per_night -> 
  pizzas = pizzas_can_afford total_earnings pizza_cost ->
  total_people = total_people_fed pizzas people_per_pizza ->
  friends_nelly_buys_for total_people = 14 := by
    intros
    sorry

end nelly_friends_l381_381459


namespace parabola_shift_l381_381380

theorem parabola_shift (x : ℝ) : 
  let initial_parabola := x^2 - 2*x + 3
  let shifted_parabola := (x + 1)^2 - 2
  in shifted_parabola = x^2 - 1 :=
sorry

end parabola_shift_l381_381380


namespace smallest_b_factors_l381_381302

theorem smallest_b_factors (b p q : ℤ) (H : p * q = 2016) : 
  (∀ k₁ k₂ : ℤ, k₁ * k₂ = 2016 → k₁ + k₂ ≥ p + q) → 
  b = 90 :=
by
  -- Here, we assume the premises stated for integers p, q such that their product is 2016.
  -- We need to fill in the proof steps which will involve checking all appropriate (p, q) pairs.
  sorry

end smallest_b_factors_l381_381302


namespace f_2007_eq_0_l381_381330

-- Define even function and odd function properties
def is_even (f : ℝ → ℝ) := ∀ x, f (-x) = f x
def is_odd (f : ℝ → ℝ) := ∀ x, f (-x) = -f x

-- Define functions f and g
variables (f g : ℝ → ℝ)

-- Assume the given conditions
axiom even_f : is_even f
axiom odd_g : is_odd g
axiom g_def : ∀ x, g x = f (x - 1)

-- Prove that f(2007) = 0
theorem f_2007_eq_0 : f 2007 = 0 :=
sorry

end f_2007_eq_0_l381_381330


namespace sum_of_c_n_l381_381339

noncomputable def a (n : ℕ) : ℕ :=
  if h : n = 0 then 0 else n

noncomputable def b (n : ℕ) : ℕ :=
  if n = 1 then 1 else 2^(n) - 1

noncomputable def c (n : ℕ) : ℕ :=
  n * 2^(n)

noncomputable def T (n : ℕ) : ℕ :=
  ∑ i in finset.range n, (i + 1) * 2^(i + 1)

theorem sum_of_c_n (n : ℕ) : T n = 2^(n+1) * (n - 1) + 2 := 
  sorry

end sum_of_c_n_l381_381339


namespace solve_ff_eq_x_l381_381042

def f (x : ℝ) : ℝ := x^2 + 2 * x - 5

theorem solve_ff_eq_x :
  ∀ x : ℝ, f (f x) = x ↔ (x = ( -1 + Real.sqrt 21 ) / 2) ∨ (x = ( -1 - Real.sqrt 21 ) / 2) ∨
                          (x = ( -3 + Real.sqrt 17 ) / 2) ∨ (x = ( -3 - Real.sqrt 17 ) / 2) := 
by
  sorry

end solve_ff_eq_x_l381_381042


namespace problem_l381_381009

noncomputable def E : Set ℕ := {n ∈ finset.range 1 201 | n > 0}
noncomputable def G : Set ℕ := {a ∈ E | a ∣ 200 ∧ (∑ i in finset.range 1 101, a) = 10080 ∧ ∀ (i j : ℕ), 1 ≤ i → i < j → j ≤ 100 → a i + a j ≠ 201}

theorem problem
  (E : Set ℕ := {n ∈ finset.range 1 201 | n > 0})
  (G : Set ℕ := {a ∈ E | a ∣ 200 ∧ (∑ i in finset.range 1 101, a) = 10080 ∧ ∀ (i j : ℕ), 1 ≤ i → i < j → j ≤ 100 → a i + a j ≠ 201}) :
  (card (G.filter (λ x, x % 2 = 1)) % 4 = 0) ∧ (∃ c, ∀ G' : set ℕ, G' ⊆ E ∧ (∑ i in G', id) = 10080 ∧ ∀ (i j : ℕ), 1 ≤ i → i < j → j ≤ 100 → i + j ≠ 201 → ∑ i in G', i^2 = c) := by
  sorry

end problem_l381_381009


namespace simple_interest_rate_l381_381168

theorem simple_interest_rate (P R: ℝ) (T: ℝ) (H: T = 5) (H1: P * (1/6) = P * (R * T / 100)) : R = 10/3 :=
by {
  sorry
}

end simple_interest_rate_l381_381168


namespace constant_term_binomial_expansion_l381_381617

theorem constant_term_binomial_expansion : 
  ∀ (x : ℝ), (x - 1/x)^8 = ∑ (k : ℕ) in finset.range (9), (binomial 8 k) * (-1)^k * x^(8 - 2*k) → 
  has_constant_term (∑ (k : ℕ) in finset.range (9), (binomial 8 k) * (-1)^k * x^(8 - 2*k)) 70 :=
sorry

end constant_term_binomial_expansion_l381_381617


namespace train_cross_time_l381_381122

open Real

noncomputable def length_train1 := 190 -- in meters
noncomputable def length_train2 := 160 -- in meters
noncomputable def speed_train1 := 60 * (5/18) --speed_kmhr_to_msec 60 km/hr to m/s
noncomputable def speed_train2 := 40 * (5/18) -- speed_kmhr_to_msec 40 km/hr to m/s
noncomputable def relative_speed := speed_train1 + speed_train2 -- relative speed

theorem train_cross_time :
  (length_train1 + length_train2) / relative_speed = 350 / ((60 * (5/18)) + (40 * (5/18))) :=
by
  sorry -- The proof will be here initially just to validate the Lean statement

end train_cross_time_l381_381122


namespace ellipse_properties_l381_381322

noncomputable def ellipse_standard_eq (a b : ℝ) : set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1^2) / a^2 + (p.2^2) / b^2 = 1}

def is_equilateral_triangle (a b c : ℝ) : Prop :=
  a = b ∧ b = c

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem ellipse_properties (a b c : ℝ) (h1 : a > b) (h2 : b > 0)
    (h_focal : 2 * c = 4) (h_triangle : is_equilateral_triangle (2*b) (2*b) (2*c))
    (h_standard_eq : ellipse_standard_eq 2 sqrt(2) 2)
    (x m : ℝ) (h_on_line : x = -3)
    (h_F : (-2, 0)) (h_perpendicular : ⟦construct perpendicular from F to PF⟧)
    (M N : ℝ × ℝ) (h_M_on_ellipse : (M ∈ ellipse_standard_eq 2 sqrt(2)))
    (h_N_on_ellipse : (N ∈ ellipse_standard_eq 2 sqrt(2)))
    (d1 d2 : ℝ) (h_d1 : distance M (x, m)) (h_d2 : distance N (x, m)) :
  ellipse_standard_eq 2 sqrt(2) 2 :=
    sorry

end ellipse_properties_l381_381322


namespace total_cost_is_160_l381_381375

-- Define the costs of each dress
def CostOfPaulineDress := 30
def CostOfJeansDress := CostOfPaulineDress - 10
def CostOfIdasDress := CostOfJeansDress + 30
def CostOfPattysDress := CostOfIdasDress + 10

-- The total cost
def TotalCost := CostOfPaulineDress + CostOfJeansDress + CostOfIdasDress + CostOfPattysDress

-- Prove the total cost is $160
theorem total_cost_is_160 : TotalCost = 160 := by
  -- skipping the proof steps
  sorry

end total_cost_is_160_l381_381375


namespace sum_of_first_9_terms_l381_381509

noncomputable section

variable {a : ℕ → ℝ} {q : ℝ}

def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = a 1 * q ^ n

axiom a1_a4_a7_eq_2 (a : ℕ → ℝ) (q : ℝ) [geometric_sequence a q] : a 1 + a 4 + a 7 = 2
axiom a3_a6_a9_eq_18 (a : ℕ → ℝ) (q : ℝ) [geometric_sequence a q] : a 3 + a 6 + a 9 = 18
axiom positive_terms (a : ℕ → ℝ) [geometric_sequence a q] : ∀ n, 0 < a n

theorem sum_of_first_9_terms (a : ℕ → ℝ) (q : ℝ) [geometric_sequence a q] 
  (h1 : a1_a4_a7_eq_2 a q) (h2 : a3_a6_a9_eq_18 a q) (h3 : positive_terms a) : 
  (∑ i in Finset.range 9, a (i + 1)) = 26 := by
  sorry

end sum_of_first_9_terms_l381_381509


namespace lower_right_square_is_4_l381_381554

-- Define the initial grid as given in the problem.
def initialGrid : Matrix 4 4 ℕ :=
  ![[0, 2, 0, 3],
    [0, 0, 4, 0],
    [4, 0, 0, 0],
    [0, 1, 0, 0]]

-- Ensure that each digit from 1 to 4 appears exactly once in each row and each column.
def isLatinSquare (m : Matrix 4 4 ℕ) : Prop :=
  ∀ i j, ∀ n, n ∈ Finset.range 1 5 → (∃ k, m i k = n) ∧ (∃ k, m k j = n)

-- Main theorem statement
theorem lower_right_square_is_4 : ∃ m : Matrix 4 4 ℕ, 
  isLatinSquare m ∧ (initialGrid[0][1] = 2) ∧ (initialGrid[0][3] = 3) ∧ 
                        (initialGrid[1][2] = 4) ∧ (initialGrid[2][0] = 4) ∧ 
                        (initialGrid[3][1] = 1) ∧ (m[3][3] = 4) := 
sorry

end lower_right_square_is_4_l381_381554


namespace perimeter_of_square_l381_381494

theorem perimeter_of_square (A : ℝ) (hA : A = 400) : exists P : ℝ, P = 80 :=
by
  sorry

end perimeter_of_square_l381_381494


namespace telescoping_log_sum_l381_381242

theorem telescoping_log_sum :
  ∑ k in Finset.range 98 \ Finset.range 2, (Real.log (1 + 1 / k) / Real.log 3) * (Real.log 3 / Real.log k) * (Real.log 3 / Real.log (k + 1)) = 1 - 1 / Real.log 101 :=
by
  sorry

end telescoping_log_sum_l381_381242


namespace part_a_not_necessarily_isosceles_part_b_necessarily_isosceles_l381_381464

noncomputable def triangle (A B C : Type) := sorry

variables {A B C O M N : Type}
variables (AM CN BM BN : ℝ)

-- Conditions
variables (h1 : AO = CO)
variables (h2 : AM = CN ∨ BM = BN)

-- Proof problem for part (a)
theorem part_a_not_necessarily_isosceles (h : AM = CN) : 
  ¬(∀ (A B C : Type), is_isosceles_triangle A B C) := sorry

-- Proof problem for part (b)
theorem part_b_necessarily_isosceles (h : BM = BN) : 
  (∀ (A B C : Type), is_isosceles_triangle A B C) := sorry

end part_a_not_necessarily_isosceles_part_b_necessarily_isosceles_l381_381464


namespace Cheryl_golf_tournament_cost_l381_381213

theorem Cheryl_golf_tournament_cost :
  let electricity_bill := 800 in
  let cell_phone_expenses := electricity_bill + 400 in
  let tournament_extra_cost := 0.20 * cell_phone_expenses in
  let total_tournament_cost := cell_phone_expenses + tournament_extra_cost in
  total_tournament_cost = 1440 :=
by
  sorry

end Cheryl_golf_tournament_cost_l381_381213


namespace polynomial_divides_difference_l381_381002

theorem polynomial_divides_difference
  (P : polynomial ℤ) (a b : ℤ) : (a - b) ∣ (P.eval a - P.eval b) := 
sorry

end polynomial_divides_difference_l381_381002


namespace sum_logarithms_l381_381237

theorem sum_logarithms :
  (∑ k in finset.Ico 3 101, real.logb 3 (1 + 1/(k:ℝ)) * real.logb k 3 * real.logb (k+1) 3) =
  1 - real.logb 2 3 / real.logb 2 101 :=
by
  sorry

end sum_logarithms_l381_381237


namespace greatest_integer_l381_381578

theorem greatest_integer (y : ℤ) : (8 / 11 : ℝ) > (y / 17 : ℝ) → y ≤ 12 :=
by sorry

end greatest_integer_l381_381578


namespace who_wears_which_dress_l381_381740

def girls := ["Katya", "Olya", "Liza", "Rita"]
def dresses := ["pink", "green", "yellow", "blue"]

variable (who_wears_dress : String → String)

/-- Conditions given in the problem --/
axiom Katya_not_pink_blue : who_wears_dress "Katya" ≠ "pink" ∧ who_wears_dress "Katya" ≠ "blue"
axiom between_green_liza_yellow : ∃ g, who_wears_dress "Katya" = "green" ∧ who_wears_dress "Rita" = "yellow"
axiom Rita_not_green_blue : who_wears_dress "Rita" ≠ "green" ∧ who_wears_dress "Rita" ≠ "blue"
axiom Olya_between_rita_pink : ∃ o, who_wears_dress "Olya" = "blue" ∧ who_wears_dress "Liza" = "pink"

theorem who_wears_which_dress :
  who_wears_dress "Katya" = "green" ∧
  who_wears_dress "Olya" = "blue" ∧
  who_wears_dress "Liza" = "pink" ∧
  who_wears_dress "Rita" = "yellow" :=
by
  sorry

end who_wears_which_dress_l381_381740


namespace perimeter_square_III_l381_381273

theorem perimeter_square_III (perimeter_I perimeter_II : ℕ) (hI : perimeter_I = 12) (hII : perimeter_II = 24) : 
  let side_I := perimeter_I / 4 
  let side_II := perimeter_II / 4 
  let side_III := side_I + side_II 
  4 * side_III = 36 :=
by
  sorry

end perimeter_square_III_l381_381273


namespace greatest_integer_l381_381584

theorem greatest_integer (y : ℤ) (h : (8 : ℚ) / 11 > y / 17) : y ≤ 12 :=
by
  have h₁ : (8 : ℚ) / 11 * 17 > y := by exact (div_mul_cancel _ (by norm_num : 17 ≠ 0))
  have h₂ : 136 / 11 > y := by rwa mul_comm _ 17 at h₁
  exact_mod_cast le_of_lt h₂

end greatest_integer_l381_381584


namespace complement_union_M_N_eq_ge_2_l381_381959

def U := set.univ ℝ
def M := {x : ℝ | x < 1}
def N := {x : ℝ | -1 < x ∧ x < 2}

theorem complement_union_M_N_eq_ge_2 :
  (U \ (M ∪ N)) = {x : ℝ | 2 ≤ x} :=
by sorry

end complement_union_M_N_eq_ge_2_l381_381959


namespace expected_intersections_100gon_l381_381437

noncomputable def expected_intersections : ℝ :=
  let n := 100
  let total_pairs := (n * (n - 3) / 2)
  total_pairs * (1/3)

theorem expected_intersections_100gon :
  expected_intersections = 4850 / 3 :=
by
  sorry

end expected_intersections_100gon_l381_381437


namespace max_students_per_class_l381_381543

theorem max_students_per_class
    (total_students : ℕ)
    (total_classes : ℕ)
    (bus_count : ℕ)
    (bus_seats : ℕ)
    (students_per_class : ℕ)
    (total_students = 920)
    (bus_count = 16)
    (bus_seats = 71)
    (∀ c < total_classes, students_per_class ≤ bus_seats) : 
    students_per_class ≤ 17 := 
by
    sorry

end max_students_per_class_l381_381543


namespace problem_equivalent_l381_381938

def U : Set ℝ := Set.univ
def M : Set ℝ := {x | x < 1}
def N : Set ℝ := {x | -1 < x ∧ x < 2}

theorem problem_equivalent :
  {x : ℝ | x ≥ 2} = (U \ (M ∪ N)) := 
by sorry

end problem_equivalent_l381_381938


namespace function_zero_point_necessary_not_sufficient_l381_381783

theorem function_zero_point_necessary_not_sufficient (m : ℝ) :
  (|m| ≤ real.sqrt 3 / 3) →
  ∃ x : ℝ, -2 ≤ x ∧ x ≤ 2 ∧ (√(4 - x^2) / (x + 4) - m = 0) :=
begin
  intro h,
  sorry
end

end function_zero_point_necessary_not_sufficient_l381_381783


namespace telescoping_log_sum_l381_381240

theorem telescoping_log_sum :
  ∑ k in Finset.range 98 \ Finset.range 2, (Real.log (1 + 1 / k) / Real.log 3) * (Real.log 3 / Real.log k) * (Real.log 3 / Real.log (k + 1)) = 1 - 1 / Real.log 101 :=
by
  sorry

end telescoping_log_sum_l381_381240


namespace binary_operation_l381_381298

theorem binary_operation : 
  let a := 0b11011
  let b := 0b1101
  let c := 0b1010
  let result := 0b110011101  
  ((a * b) - c) = result := by
  sorry

end binary_operation_l381_381298


namespace sum_of_integer_solutions_l381_381486

theorem sum_of_integer_solutions : 
  (∑ x in {a : ℤ | 2 + a > 7 - 4 * a ∧ a < (4 + a) / 2 }, id) = 5 :=
by
  sorry

end sum_of_integer_solutions_l381_381486


namespace bills_sum_to_total_cost_l381_381453

/-- Mark's grocery items and their associated costs --/
def cost_cans_of_soup : ℕ := 6 * 2
def cost_loaves_of_bread : ℕ := 2 * 5
def cost_boxes_of_cereal : ℕ := 2 * 3
def cost_gallons_of_milk : ℕ := 2 * 4

/-- Total cost of groceries --/
def total_cost : ℕ := cost_cans_of_soup + cost_loaves_of_bread + cost_boxes_of_cereal + cost_gallons_of_milk

/-- Given the total cost, prove that it can be paid using 4 specific bills --/
theorem bills_sum_to_total_cost : total_cost = 36 ∧ (∃ (b1 b2 b3 b4 : ℕ), {b1, b2, b3, b4} = {20, 10, 5, 1} ∧ b1 + b2 + b3 + b4 = 36) :=
by
  sorry

end bills_sum_to_total_cost_l381_381453


namespace range_of_a_l381_381022
noncomputable theory

open Set

-- Given conditions
def A : Set ℝ := {x | -3 < x ∧ x < 2}
def B (a : ℝ) : Set ℝ := {x | a < x}

-- Problem statement: Prove that if ∀ x, x ∈ A → x ∈ B(a), then a ∈ (-∞, -3]
theorem range_of_a (a : ℝ) (h : A ⊆ B a) : a ∈ Iic (-3) :=
begin
  sorry
end

end range_of_a_l381_381022


namespace grams_of_fat_per_cup_l381_381037

theorem grams_of_fat_per_cup (cups_per_morning : ℕ) (cups_per_afternoon : ℕ) (cups_per_evening : ℕ) (days_per_week : ℕ) (total_fat_per_week : ℕ) :
  cups_per_morning = 3 →
  cups_per_afternoon = 2 →
  cups_per_evening = 5 →
  days_per_week = 7 →
  total_fat_per_week = 700 →
  total_fat_per_week / ((cups_per_morning + cups_per_afternoon + cups_per_evening) * days_per_week) = 10 :=
by
  intros h_morning h_afternoon h_evening h_days h_fat
  rw [h_morning, h_afternoon, h_evening, h_days, h_fat]
  norm_num
  sorry

end grams_of_fat_per_cup_l381_381037


namespace hypotenuse_longest_side_right_triangle_longest_side_opposite_obtuse_angle_l381_381470

theorem hypotenuse_longest_side_right_triangle (a b c : ℝ) (h : a^2 + b^2 = c^2) : a < c ∧ b < c :=
by {
  sorry
}

theorem longest_side_opposite_obtuse_angle (a b c γ : ℝ) (h1 : a^2 + b^2 - 2 * a * b * real.cos γ = c^2) (h2 : γ > π/2) : a < c ∧ b < c :=
by {
  sorry
}

end hypotenuse_longest_side_right_triangle_longest_side_opposite_obtuse_angle_l381_381470


namespace probability_third_smallest_is_five_l381_381700

open Finset

noncomputable def prob_third_smallest_is_five : ℚ :=
  let total_ways := choose 15 8
  let favorable_ways := (choose 4 2) * (choose 10 5)
  in favorable_ways / total_ways

theorem probability_third_smallest_is_five :
  prob_third_smallest_is_five = 72 / 307 :=
by sorry

end probability_third_smallest_is_five_l381_381700


namespace area_of_border_l381_381160

theorem area_of_border 
  (photo_height : ℕ) (photo_width : ℕ) (border_width : ℕ)
  (photo_height_eq : photo_height = 12) (photo_width_eq : photo_width = 15) (border_width_eq : border_width = 3) : 
  (border_area : ℕ) (border_area_eq : border_area = 198) := 
  sorry

end area_of_border_l381_381160


namespace coterminal_angle_equivalence_l381_381662

theorem coterminal_angle_equivalence (k : ℤ) : ∃ n : ℤ, -463 % 360 = (k * 360 + 257) % 360 :=
by
  sorry

end coterminal_angle_equivalence_l381_381662


namespace ratio_areas_l381_381880

-- Variables and definitions based on problem conditions
def side_length_sq (s : ℝ) : ℝ := s^2
def diagonal_sq (s : ℝ) : ℝ := s * Real.sqrt 2
def radius_circle (s : ℝ) : ℝ := (s * Real.sqrt 2) / 2
def area_circle (s : ℝ) : ℝ := π * (radius_circle s)^2

-- Theorem statement
theorem ratio_areas (s : ℝ) (h : s > 0) : (area_circle s) / (side_length_sq s) = π / 2 :=
by
  sorry

end ratio_areas_l381_381880


namespace pages_at_end_of_march_l381_381709

theorem pages_at_end_of_march (daily_pages : ℕ) (initial_pages : ℕ) (days_in_march : ℕ) :
  daily_pages = 30 → initial_pages = 400 → days_in_march = 31 →
  initial_pages + daily_pages * days_in_march = 1330 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  exact rfl

end pages_at_end_of_march_l381_381709


namespace triangle_area_is_24_l381_381567

-- Defining the vertices of the triangle
def A := (2, 2)
def B := (8, 2)
def C := (4, 10)

-- Calculate the area of the triangle
def area_of_triangle (A B C : ℕ × ℕ) : ℕ := 
  let base := |B.1 - A.1| 
  let height := |C.2 - A.2| 
  ((base * height) / 2)

-- Statement to prove
theorem triangle_area_is_24 : area_of_triangle A B C = 24 := 
by
  sorry

end triangle_area_is_24_l381_381567


namespace general_term_formula_minimal_lambda_l381_381814

noncomputable def a_n (n : ℕ) : ℕ := 2^n
noncomputable def S_n (n : ℕ) : ℕ := 2 * (a_n n) - 2
noncomputable def b_n (n : ℕ) : ℕ := n
noncomputable def c_n (n : ℕ) : ℚ := 1 / ((2 * (b_n n) - 1) * (2 * (b_n n) + 1))
noncomputable def T_n (n : ℕ) : ℚ := (Finset.range (n+1)).sum (λ i, c_n i)

theorem general_term_formula (n : ℕ) : a_n n = 2^(n) :=
sorry

theorem minimal_lambda (n : ℕ) : T_n n < 1/2 :=
sorry

end general_term_formula_minimal_lambda_l381_381814


namespace largest_prime_factor_of_5082_is_11_l381_381589

theorem largest_prime_factor_of_5082_is_11 : ∃ p : ℕ, p.prime ∧ p ∣ 5082 ∧ ∀ q : ℕ, q.prime ∧ q ∣ 5082 → q ≤ p := 
sorry

end largest_prime_factor_of_5082_is_11_l381_381589


namespace correct_statements_count_l381_381056

def condition1 : Prop := "The ray that bisects an angle in a triangle is the bisector of the triangle."
def condition2 : Prop := "The three medians of a triangle intersect at a point called the centroid of the triangle."
def condition3 : Prop := "The three altitudes of a triangle intersect at a point."
def condition4 : Prop := "A right triangle has only one altitude."

def is_correct_statement (statement : Prop) : Prop :=
  statement ∈ {condition2}

theorem correct_statements_count (h1 : ¬ condition1) (h2 : condition2) (h3 : ¬ condition3) (h4 : ¬ condition4) :
  (finset.filter is_correct_statement (finset.from_list [condition1, condition2, condition3, condition4])).card = 1 :=
sorry

end correct_statements_count_l381_381056


namespace smallest_n_modulo_l381_381592

theorem smallest_n_modulo :
  ∃ (n : ℕ), 0 < n ∧ 1031 * n % 30 = 1067 * n % 30 ∧ ∀ (m : ℕ), 0 < m ∧ 1031 * m % 30 = 1067 * m % 30 → n ≤ m :=
by
  sorry

end smallest_n_modulo_l381_381592


namespace ballpoint_pens_relationship_l381_381879

variables (x : ℕ) (y : ℝ)

def unit_price : ℝ := 18 / 12

theorem ballpoint_pens_relationship (h : y = unit_price * x) : y = (3 / 2) * x :=
by
  sorry

end ballpoint_pens_relationship_l381_381879


namespace dress_assignment_l381_381756

theorem dress_assignment :
  ∃ (Katya Olya Liza Rita : string),
    (Katya ≠ "Pink" ∧ Katya ≠ "Blue") ∧
    (Rita ≠ "Green" ∧ Rita ≠ "Blue") ∧
    ∃ (girl_in_green girl_in_yellow : string),
      (girl_in_green = Katya ∧ girl_in_yellow = Rita ∧ 
       (Liza = "Pink" ∧ Olya = "Blue") ∧
       (Katya = "Green" ∧ Olya = "Blue" ∧ Liza = "Pink" ∧ Rita = "Yellow")) ∧
    ((girl_in_green stands between Liza and girl_in_yellow) ∧
     (Olya stands between Rita and Liza)) :=
by
  sorry

end dress_assignment_l381_381756


namespace distance_between_AE_and_BF_l381_381394

-- Define the coordinates for all points in the rectangular parallelepiped
def A : ℝ × ℝ × ℝ := (0, 0, 0)
def B : ℝ × ℝ × ℝ := (60, 0, 0)
def D : ℝ × ℝ × ℝ := (0, 36, 0)
def A₁ : ℝ × ℝ × ℝ := (0, 0, 40)
def B₁ : ℝ × ℝ × ℝ := (60, 0, 40)
def C₁ : ℝ × ℝ × ℝ := (60, 36, 40)

-- Define the positions of points E and F
def E : ℝ × ℝ × ℝ := (30, 0, 40)
def F : ℝ × ℝ × ℝ := (60, 18, 40)

-- Define the parametric equations for lines AE and BF
def r_AE (t : ℝ) : ℝ × ℝ × ℝ := (30 * t, 0, 40 * t)
def r_BF (s : ℝ) : ℝ × ℝ × ℝ := (60, 18 * s, 40 * s)

-- Define the direction vectors for AE and BF
def d1 : ℝ × ℝ × ℝ := (30, 0, 40)
def d2 : ℝ × ℝ × ℝ := (0, 18, 40)

-- Define the vector B - A
def B_minus_A : ℝ × ℝ × ℝ := (60, 0, 0)

-- Cross product of direction vectors d1 and d2
def cross (x y : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (x.2 * y.3 - x.3 * y.2, x.3 * y.1 - x.1 * y.3, x.1 * y.2 - x.2 * y.1)

def d1_cross_d2 : ℝ × ℝ × ℝ := cross d1 d2

-- Dot product of B - A and d1 cross d2
def dot (x y : ℝ × ℝ × ℝ) : ℝ :=
  x.1 * y.1 + x.2 * y.2 + x.3 * y.3
  
def B_minus_A_dot_d1_cross_d2 : ℝ := dot B_minus_A d1_cross_d2

-- Magnitude of d1 cross d2
def magnitude (x : ℝ × ℝ × ℝ) : ℝ :=
  real.sqrt (x.1^2 + x.2^2 + x.3^2)
  
def magnitude_d1_cross_d2 : ℝ := magnitude d1_cross_d2

-- Distance between the lines AE and BF
def distance : ℝ :=
  (B_minus_A_dot_d1_cross_d2).abs / magnitude_d1_cross_d2
  
-- Theorem statement: the distance between lines AE and BF is 28.8
theorem distance_between_AE_and_BF : distance = 28.8 := by
  sorry

end distance_between_AE_and_BF_l381_381394


namespace car_speed_is_125_kmh_l381_381630

def distance : ℝ := 375  -- distance in km
def time : ℝ := 3       -- time in hours

def speed (d t : ℝ) : ℝ := d / t  -- speed formula in km/h

theorem car_speed_is_125_kmh : speed distance time = 125 := by
  sorry

end car_speed_is_125_kmh_l381_381630


namespace nested_radicals_solution_l381_381296

theorem nested_radicals_solution (x : ℝ) (h : 0 < x)
    (h1 : sqrt (x + sqrt (x + sqrt (x + ...))) = sqrt (x * sqrt (x * sqrt (x * ...))))
    : x = 2 :=
begin
    sorry
end

end nested_radicals_solution_l381_381296


namespace probability_third_smallest_five_l381_381697

theorem probability_third_smallest_five :
  let S := finset.Icc 1 15 in
  let total_ways := S.card.choose 8 in
  let favorable_ways := (finset.Icc 6 15).card.choose 4 * (finset.Icc 1 4).card.choose 2 in
  (favorable_ways : ℚ) / total_ways = 4 / 21 :=
by {
  let S := finset.Icc 1 15,
  let total_ways := S.card.choose 8,
  let favorable_ways := (finset.Icc 6 15).card.choose 4 * (finset.Icc 1 4).card.choose 2,
  have h : favorable_ways = 1260 := rfl,
  have h2 : total_ways = 6435 := rfl,
  calc
    (favorable_ways : ℚ) / total_ways
        = (1260 : ℚ) / 6435 : by rw [h, h2]
    ... = 4 / 21 : by norm_num,
  sorry
}

end probability_third_smallest_five_l381_381697


namespace no_line_exists_intersecting_hyperbola_midpoint_l381_381504

theorem no_line_exists_intersecting_hyperbola_midpoint :
  ¬ ∃ (l : ℝ → ℝ), 
    (∀ x y, (x, y) ∈ line (1, 1) l → x^2 - (y^2) / 2 = 1) ∧
    (∃ A B : ℝ × ℝ, A ≠ B ∧ (A.1, A.2) ∈ line (1, 1) l ∧ (B.1, B.2) ∈ line (1, 1) l ∧ (1, 1) = midpoint A B) := 
sorry

def line (P : ℝ × ℝ) (f : ℝ → ℝ) : set (ℝ × ℝ) :=
  { Q | ∃ x, Q = (x, f x) ∨ Q = (1, 1) }

def midpoint (A B : ℝ × ℝ) : ℝ × ℝ :=
  ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

end no_line_exists_intersecting_hyperbola_midpoint_l381_381504


namespace count_three_digit_numbers_divisible_by_7_l381_381850

def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

def is_divisible_by_7 (n : ℕ) : Prop :=
  n % 7 = 0

def count_three_digit_divisible_by_7 : ℕ :=
  (list.range' 105 890).countp (λ x, is_divisible_by_7 (x * 7))

theorem count_three_digit_numbers_divisible_by_7 :
  count_three_digit_divisible_by_7 = 128 :=
sorry

end count_three_digit_numbers_divisible_by_7_l381_381850


namespace symmetric_about_y_eq_x_l381_381489

-- Let f be a function whose graph is symmetric about the line y = x-2.
variable {α β : Type*} [LinearOrderedField α] [AddCommGroup β] [Module α β]
variable (f : α → β)
variable (hf_symm : ∀ x, f(x - 2) = f(2 - x))

-- Define the new function h(x) = f(x) + b.
def h (x : α) (b : β) := f x + b

-- The proof: For what choice of b is it true that h(x) = h⁻¹(x)?
theorem symmetric_about_y_eq_x (b : β) : (∀ x, h f x b = (h f) (h f x b)) ↔ b = 2 := 
by
  sorry

end symmetric_about_y_eq_x_l381_381489


namespace fraction_of_fraction_l381_381561

theorem fraction_of_fraction (a b c d : ℚ) (h1 : a = 2) (h2 : b = 9) (h3 : c = 3) (h4 : d = 4) :
  (a / b) / (c / d) = (a * d) / (b * c) :=
by
  rw [div_mul_div, mul_comm c _] -- Using properties of divisions and multiplications.
  sorry

end fraction_of_fraction_l381_381561


namespace rational_iff_expression_rational_l381_381526

def E (x : ℝ) : ℝ := x + real.sqrt (x^2 + 4) - (1 / (x + real.sqrt (x^2 + 4)))

theorem rational_iff_expression_rational (x : ℝ) :
  (∃ r : ℚ, x = r) ↔ (∃ r : ℚ, E x = r) :=
sorry

end rational_iff_expression_rational_l381_381526


namespace angle_between_a_c_is_correct_l381_381534

variables {V : Type*} [inner_product_space ℝ V]
variables (a b c : V)
variables (θ : ℝ)

-- Definitions from conditions: 
def norm_a := ∥a∥ = sqrt 2
def norm_b := ∥b∥ = sqrt 2
def norm_c := ∥c∥ = 3
def eqn := a ×ₗ ((a ×ₗ c) + b) = 0

-- Statement of the problem:
theorem angle_between_a_c_is_correct :
  norm_a a → norm_b b → norm_c c → eqn a b c → 
  θ = real.arccos (sqrt (7 / 18)) ∨ θ = real.arccos (-sqrt (7 / 18)) :=
sorry

end angle_between_a_c_is_correct_l381_381534


namespace correct_assignment_l381_381774

structure GirlDressAssignment :=
  (Katya : String)
  (Olya : String)
  (Liza : String)
  (Rita : String)

def solution : GirlDressAssignment :=
  ⟨"Green", "Blue", "Pink", "Yellow"⟩

theorem correct_assignment
  (Katya_not_pink_or_blue : solution.Katya ≠ "Pink" ∧ solution.Katya ≠ "Blue")
  (Green_between_Liza_and_Yellow : 
    (solution.Katya = "Green" ∧ solution.Liza = "Pink" ∧ solution.Rita = "Yellow") ∧
    (solution.Katya = "Green" ∧ solution.Rita = "Yellow" ∧ solution.Liza = "Pink"))
  (Rita_not_green_or_blue : solution.Rita ≠ "Green" ∧ solution.Rita ≠ "Blue")
  (Olya_between_Rita_and_Pink : 
    (solution.Olya = "Blue" ∧ solution.Rita = "Yellow" ∧ solution.Liza = "Pink") ∧
    (solution.Olya = "Blue" ∧ solution.Liza = "Pink" ∧ solution.Rita = "Yellow"))
  : solution = ⟨"Green", "Blue", "Pink", "Yellow"⟩ := by
  sorry

end correct_assignment_l381_381774


namespace taqeesha_grade_l381_381536

theorem taqeesha_grade (s : ℕ → ℕ) (h1 : (s 16) = 77) (h2 : (s 17) = 78) : s 17 - s 16 = 94 :=
by
  -- Add definitions and sorry to skip the proof
  sorry

end taqeesha_grade_l381_381536


namespace log2_x_neg_3_div_2_l381_381871

noncomputable def log₄_2 : ℝ := Real.log 2 / Real.log 4
noncomputable def log₄_8 : ℝ := Real.log 8 / Real.log 4
noncomputable def x : ℝ := log₄_2 ^ log₄_8

theorem log2_x_neg_3_div_2 : Real.log 2 x = -3 / 2 := 
  by sorry

end log2_x_neg_3_div_2_l381_381871


namespace area_difference_triangles_l381_381406

theorem area_difference_triangles
  (A B C F D : Type)
  (angle_FAB_right : true) 
  (angle_ABC_right : true) 
  (AB : Real) (hAB : AB = 5)
  (BC : Real) (hBC : BC = 3)
  (AF : Real) (hAF : AF = 7)
  (area_triangle : A -> B -> C -> Real)
  (angle_bet : A -> D -> F) 
  (angle_bet : B -> D -> C)
  (area_ADF : Real)
  (area_BDC : Real) : (area_ADF - area_BDC = 10) :=
sorry

end area_difference_triangles_l381_381406


namespace max_students_per_class_l381_381544

-- Definitions used in Lean 4 statement:
def num_students := 920
def seats_per_bus := 71
def num_buses := 16

-- The main statement, showing this is the maximum value such that each class stays together within the given constraints.
theorem max_students_per_class : ∃ k, (∀ k' : ℕ, k' > k → 
  ¬∃ (classes : ℕ), classes * k' + (num_students - classes * k') ≤ seats_per_bus * num_buses ∧ k' <= seats_per_bus) ∧ k = 17 := 
by sorry

end max_students_per_class_l381_381544


namespace half_radius_of_equal_area_circles_l381_381607

-- Definitions
def radius_from_circumference (C : ℝ) : ℝ := 
  C / (2 * Real.pi)

def half_radius (r : ℝ) : ℝ := 
  r / 2

-- Given Conditions
variable (x_circumference : ℝ) (areas_equal : Prop)

-- Mathematically equivalent proof problem
theorem half_radius_of_equal_area_circles 
  (hx : x_circumference = 18 * Real.pi)
  (areas_equal : ∀ (r₁ r₂ : ℝ), Real.pi * r₁^2 = Real.pi * r₂^2 → r₁ = r₂) :
  half_radius (radius_from_circumference x_circumference) = 4.5 := 
by
  sorry

end half_radius_of_equal_area_circles_l381_381607


namespace geom_seq_common_ratio_l381_381428

theorem geom_seq_common_ratio (a : ℕ → ℝ) (q : ℝ)
  (h1 : ∀ n, a n > 0)
  (h2 : ∀ n, a (n + 1) = q * a n)
  (h3 : ∑ i in finset.range 4, a i = 10 * ∑ i in finset.range 2, a i) :
  q = 3 :=
sorry

end geom_seq_common_ratio_l381_381428


namespace last_recess_break_duration_l381_381104

-- Definitions based on the conditions
def first_recess_break : ℕ := 15
def second_recess_break : ℕ := 15
def lunch_break : ℕ := 30
def total_outside_class_time : ℕ := 80

-- The theorem we need to prove
theorem last_recess_break_duration :
  total_outside_class_time = first_recess_break + second_recess_break + lunch_break + 20 :=
sorry

end last_recess_break_duration_l381_381104


namespace jenny_chocolate_squares_l381_381920

theorem jenny_chocolate_squares (mike_chocolates : ℕ) (jenny_chocolates : ℕ) 
  (h_mike : mike_chocolates = 20) 
  (h_jenny : jenny_chocolates = 3 * mike_chocolates + 5) :
  jenny_chocolates = 65 :=
by
  sorry

end jenny_chocolate_squares_l381_381920


namespace find_rate_of_interest_l381_381120

noncomputable def interest_rate (P R : ℝ) : Prop :=
  (400 = P * (1 + 4 * R / 100)) ∧ (500 = P * (1 + 6 * R / 100))

theorem find_rate_of_interest (R : ℝ) (P : ℝ) (h : interest_rate P R) :
  R = 25 :=
by
  sorry

end find_rate_of_interest_l381_381120


namespace Alyssa_Coins_Total_Value_l381_381598

theorem Alyssa_Coins_Total_Value (quarters : ℕ) (pennies : ℕ) (nickels : ℕ) (dimes : ℕ) : 
  (quarters = 12) → (pennies = 7) → (nickels = 20) → (dimes = 15) →
  let total_value := quarters * 0.25 + pennies * 0.01 + nickels * 0.05 + dimes * 0.10 in
  ∃ (bills : ℕ) (remaining_quarters remaining_pennies remaining_nickels : ℕ),
  bills = 1 ∧ 
  remaining_quarters = 2 ∧ 
  remaining_pennies = 2 ∧ 
  remaining_nickels = 1 ∧ 
  total_value = 5.57 :=
begin
  intros,
  sorry
end

end Alyssa_Coins_Total_Value_l381_381598


namespace no_divisor_form_3j_plus_2_l381_381424

theorem no_divisor_form_3j_plus_2 (k : ℕ) (n : ℕ) (hk_pos : 0 < k) (hn_divisors : finset.card (finset.filter (λ d, d ∣ n) (finset.range (n+1))) = k)
  (hn_cube : ∃ (m : ℕ), n = m^3) : ¬ ∃ (p : ℕ), prime p ∧ (∃ (j : ℕ), p = 3 * j + 2) ∧ p ∣ k :=
by {
  sorry
}

end no_divisor_form_3j_plus_2_l381_381424


namespace find_N_plus_10n_of_reals_l381_381473

theorem find_N_plus_10n_of_reals (x y z : ℝ) (h : 5 * (x + y + z) = x^2 + y^2 + z^2) :
  ∃ N n : ℝ, N = 75 ∧ n = 0 ∧ N + 10 * n = 75 :=
by {
  let A := x + y + z,
  let B := x^2 + y^2 + z^2,
  have h1 : B = 5 * A, from h,
  let C := xy + xz + yz,
  sorry
}

end find_N_plus_10n_of_reals_l381_381473


namespace quadrilateral_perimeter_l381_381795

theorem quadrilateral_perimeter (a : ℝ) (h : 0 < a) :
  let p := (((2*a) + (3*a) + (2*a * real.sqrt(5)) + a) / a)
  in p = 6 + 2 * real.sqrt(5) :=
by
  let p := (((2*a) + (3*a) + (2*a * real.sqrt(5)) + a) / a)
  have : p = 6 + 2 * real.sqrt(5) := sorry
  exact this

end quadrilateral_perimeter_l381_381795


namespace birches_count_l381_381194

/-- There are a total of 96 trees around a house arranged in a circle. 
    Trees can be either coniferous (spruce or pine) or deciduous (birch).
    The placement of the trees satisfies the following conditions:
    - From any coniferous tree, skipping one tree, one tree is coniferous, and the other is deciduous.
    - From any coniferous tree, skipping two trees, one tree is coniferous, and the other is deciduous.
    Prove that there are 32 birches among the 96 trees. -/
theorem birches_count (total_trees : ℕ) (coniferous deciduous : Type) 
  (T : Fin total_trees → coniferous ⊕ deciduous)
  (h_total : total_trees = 96)
  (h_property1 : ∀ i, (T i).isLeft → ((T (i + 2)).isLeft ≠ (T (i + 4)).isLeft))
  (h_property2 : ∀ i, (T i).isLeft → ((T (i + 3)).isLeft ≠ (T (i + 6)).isLeft)) :
  ∃ b : ℕ, b = 32 ∧ (∀ i, deciduous_cases (T i) → i = b) := 
sorry

end birches_count_l381_381194


namespace egg_cartons_l381_381201

theorem egg_cartons (chickens eggs_per_chicken eggs_per_carton : ℕ) (h_chickens : chickens = 20) (h_eggs_per_chicken : eggs_per_chicken = 6) (h_eggs_per_carton : eggs_per_carton = 12) : 
  (chickens * eggs_per_chicken) / eggs_per_carton = 10 :=
by
  rw [h_chickens, h_eggs_per_chicken, h_eggs_per_carton] -- Replace the variables with the given values
  -- Calculate the number of eggs
  have h_eggs := 20 * 6
  -- Apply the number of eggs to find the number of cartons
  rw [show 20 * 6 = 120, from rfl, show 120 / 12 = 10, from rfl]
  sorry -- Placeholder for the detailed proof

end egg_cartons_l381_381201


namespace valid_numbers_l381_381282

noncomputable def is_valid_n : ℕ → Prop :=
  λ n, ∀ (a b : ℕ), a ∣ n → b ∣ n → gcd a b = 1 → (a + b - 1) ∣ n

theorem valid_numbers :
  ∀ n : ℕ, is_valid_n n → (∃ p : ℕ, nat.prime p ∧ n = p^k ∨ n = 12) :=
by 
  sorry

end valid_numbers_l381_381282


namespace find_surface_area_of_sphere_l381_381084

noncomputable def prism_condition (A B C A1 B1 C1 O O' : Point) : Prop :=
  let CA := 2 * Real.sqrt 3
  let CB := 2 * Real.sqrt 3
  let AA1 := 4
  let angle_ACB := 120
  lateral_edges_perpendicular (A B C A1 B1 C1) ∧
  vertices_on_sphere_ABC (A B C A1 B1 C1 O) ∧
  ∠ACB = angle_ACB ∧
  CA = 2 * Real.sqrt 3 ∧
  CB = 2 * Real.sqrt 3 ∧
  AA1 = 4

theorem find_surface_area_of_sphere
  (A B C A1 B1 C1 O O' : Point) 
  (h : prism_condition A B C A1 B1 C1 O O') :
  surface_area_of_sphere O = 64 * Real.pi :=
sorry

end find_surface_area_of_sphere_l381_381084


namespace greatest_integer_l381_381580

theorem greatest_integer (y : ℤ) : (8 / 11 : ℝ) > (y / 17 : ℝ) → y ≤ 12 :=
by sorry

end greatest_integer_l381_381580


namespace interest_rate_is_10_percent_l381_381171

variables (monthly_payment : ℝ) (num_of_months : ℕ) (total_with_interest : ℝ)

-- Conditions
def total_without_interest : ℝ :=
  monthly_payment * num_of_months

def interest_amount : ℝ :=
  total_with_interest - total_without_interest

def interest_rate : ℝ :=
  (interest_amount / total_without_interest) * 100

theorem interest_rate_is_10_percent
  (h1 : monthly_payment = 100)
  (h2 : num_of_months = 12)
  (h3 : total_with_interest = 1320) :
  interest_rate monthly_payment num_of_months total_with_interest = 10 :=
by sorry

end interest_rate_is_10_percent_l381_381171


namespace triangle_area_is_24_l381_381571

structure Point where
  x : ℝ
  y : ℝ

def distance_x (A B : Point) : ℝ :=
  abs (B.x - A.x)

def distance_y (A C : Point) : ℝ :=
  abs (C.y - A.y)

def triangle_area (A B C : Point) : ℝ :=
  0.5 * distance_x A B * distance_y A C

noncomputable def A : Point := ⟨2, 2⟩
noncomputable def B : Point := ⟨8, 2⟩
noncomputable def C : Point := ⟨4, 10⟩

theorem triangle_area_is_24 : triangle_area A B C = 24 := 
  sorry

end triangle_area_is_24_l381_381571


namespace regression_lines_intersect_at_mean_l381_381556

noncomputable def mean (s : List ℝ) : ℝ :=
s.sum / s.length

def regression_line (s : List (ℝ × ℝ)) : (ℝ → ℝ) :=
λ x, sorry -- definition of the regression line is omitted for simplicity

theorem regression_lines_intersect_at_mean
  (dataA dataB : List (ℝ × ℝ))
  (⦃meanX meanY : ℝ⦄)
  (hmeanA : mean (dataA.map Prod.fst) = meanX)
  (hmeanB : mean (dataA.map Prod.snd) = meanY)
  (hmeanC : mean (dataB.map Prod.fst) = meanX)
  (hmeanD : mean (dataB.map Prod.snd) = meanY) :
  let m := regression_line dataA in
  let n := regression_line dataB in
  m meanX = meanY ∧ n meanX = meanY :=
by
  sorry

end regression_lines_intersect_at_mean_l381_381556


namespace sin_phase_shift_right_l381_381264

theorem sin_phase_shift_right (x : ℝ) : 
  ∀ x, sin (2 * x - π / 4) = sin (2 * (x - π / 8)) := 
sorry

end sin_phase_shift_right_l381_381264


namespace proof_problem_l381_381808

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g (x : ℝ) : ℝ := (x + 1) * f x

axiom domain_f : ∀ x : ℝ, true
axiom even_f : ∀ x : ℝ, f (2 * x - 1) = f (-(2 * x - 1))
axiom mono_g_neg_inf_minus_1 : ∀ x y : ℝ, x ≤ y → x ≤ -1 → y ≤ -1 → g x ≤ g y

-- Proof Problem Statement
theorem proof_problem :
  (∀ x y : ℝ, x ≤ y → -1 ≤ x → -1 ≤ y → g x ≤ g y) ∧
  (∀ a b : ℝ, g a + g b > 0 → a + b + 2 > 0) :=
by
  sorry

end proof_problem_l381_381808


namespace certain_number_l381_381130

-- Define the conditions as variables
variables {x : ℝ}

-- Define the proof problem
theorem certain_number (h : 0.15 * x = 0.025 * 450) : x = 75 :=
sorry

end certain_number_l381_381130


namespace count_3_digit_numbers_divisible_by_7_l381_381852

theorem count_3_digit_numbers_divisible_by_7 : 
  {n : ℕ | 100 ≤ n ∧ n ≤ 999 ∧ n % 7 = 0}.to_finset.card = 128 := 
  sorry

end count_3_digit_numbers_divisible_by_7_l381_381852


namespace tan_neg_210_eq_neg_sqrt_3_div_3_l381_381128

theorem tan_neg_210_eq_neg_sqrt_3_div_3 : Real.tan (-210 * Real.pi / 180) = - (Real.sqrt 3 / 3) :=
by
  sorry

end tan_neg_210_eq_neg_sqrt_3_div_3_l381_381128


namespace largest_sum_two_largest_angles_in_ABCD_l381_381399

-- Definitions as per the conditions
def is_arithmetic_progression (angles : List ℝ) : Prop :=
  (∀ i j k, i < j ∧ j < k ∧ k < angles.length → angles[j] - angles[i] = angles[k] - angles[j])

def is_angle_in_ABC_equal (α β : ℝ) (γ δ : ℝ) : Prop :=
  α = γ ∧ β = δ

-- The angles are in an arithmetic progression
def pentagon_arith_progression (a d : ℝ) : List ℝ :=
  [a, a + d, a + 2 * d, a + 3 * d, a + 4 * d]

-- The sum of the pentagon angles
def sum_of_pentagon_angles (angles : List ℝ) : ℝ :=
  List.sum angles

-- Prove the largest sum of the two largest angles
theorem largest_sum_two_largest_angles_in_ABCD (a d : ℝ) (α β γ δ : ℝ)
  (h1 : is_arithmetic_progression (pentagon_arith_progression a d))
  (h2 : sum_of_pentagon_angles (pentagon_arith_progression a d) = 540)
  (h3 : is_angle_in_ABC_equal α β γ δ) :
  3 * (a + 3 * d) + 3 * (a + 4 * d) = 360 :=
sorry

end largest_sum_two_largest_angles_in_ABCD_l381_381399


namespace robin_total_spending_l381_381476

def jelly_bracelets_total_cost : ℕ :=
  let names := ["Jessica", "Tori", "Lily", "Patrice"]
  let total_letters := names.foldl (λ acc name => acc + name.length) 0
  total_letters * 2

theorem robin_total_spending : jelly_bracelets_total_cost = 44 := by
  sorry

end robin_total_spending_l381_381476


namespace sin_neg_120_l381_381220

-- Define the angle in degrees
def deg_to_rad (d : ℝ) : ℝ := d * real.pi / 180

noncomputable def sin_deg (d : ℝ) : ℝ := real.sin (deg_to_rad d)

-- Main theorem
theorem sin_neg_120 :
  sin_deg (-120) = -real.sqrt 3 / 2 :=
by
  sorry

end sin_neg_120_l381_381220


namespace part1_part2_l381_381317

def first_order_ratio_increasing (f : ℝ → ℝ) : Prop :=
  ∀ (x y : ℝ), 0 < x → x < y → (f x) / x < (f y) / y

def second_order_ratio_increasing (f : ℝ → ℝ) : Prop :=
  ∀ (x y : ℝ), 0 < x → x < y → (f x) / x^2 < (f y) / y^2

noncomputable def f (h : ℝ) (x : ℝ) : ℝ :=
  x^3 - 2 * h * x^2 - h * x

theorem part1 (h : ℝ) (h1 : first_order_ratio_increasing (f h)) (h2 : ¬ second_order_ratio_increasing (f h)) :
  h < 0 :=
sorry

theorem part2 (f : ℝ → ℝ) (h : second_order_ratio_increasing f) (h2 : ∃ k > 0, ∀ x > 0, f x < k) :
  ∃ k, k = 0 ∧ ∀ x > 0, f x < k :=
sorry

end part1_part2_l381_381317


namespace num_permutations_div3_l381_381292

theorem num_permutations_div3 :
  (finset.univ : finset (equiv.perm (fin 5))).filter (λ π : equiv.perm (fin 5),
    (π 0 * π 1 * π 2 + π 1 * π 2 * π 3 + π 2 * π 3 * π 4 + π 3 * π 4 * π 0 + π 4 * π 0 * π 1) % 3 = 0
  ).card = 80 :=
sorry

end num_permutations_div3_l381_381292


namespace robin_total_cost_l381_381474

def num_letters_in_name (name : String) : Nat := name.length

def calculate_total_cost (names : List String) (cost_per_bracelet : Nat) : Nat :=
  let total_bracelets := names.foldl (fun acc name => acc + num_letters_in_name name) 0
  total_bracelets * cost_per_bracelet

theorem robin_total_cost : 
  calculate_total_cost ["Jessica", "Tori", "Lily", "Patrice"] 2 = 44 :=
by
  sorry

end robin_total_cost_l381_381474


namespace problem1_problem2_problem3_l381_381685

noncomputable def floor_ceiling_function (x : ℝ) : ℝ := 
  int.floor x + int.ceil x

noncomputable def floor (x : ℝ) : ℤ := int.floor x
noncomputable def ceiling (x : ℝ) : ℤ := int.ceil x

-- Definition of the primary function f
noncomputable def f (x : ℝ) : ℤ := ceiling (x * floor x)

theorem problem1 :
  f (-3 / 2) = 3 ∧ f (3 / 2) = 2 :=
by anti : sorry

theorem problem2 :
  ¬ (∀ x in [-2, 2], f (-x) = f x ∨ f (-x) = -f x) :=
by anti : sorry

theorem problem3 :
  ∀ (x : ℝ), -1 ≤ x ∧ x ≤ 1 → 
  floor_ceiling_function x = 
    if x = -1 then -2 else
    if -1 < x ∧ x < 0 then -1 else
    if x = 0 then 0 else
    if 0 < x ∧ x < 1 then 1 else
    if x = 1 then 2 else 0 :=
by anti : sorry

end problem1_problem2_problem3_l381_381685


namespace max_students_per_class_l381_381539

theorem max_students_per_class (num_students : ℕ) (seats_per_bus : ℕ) (num_buses : ℕ) (k : ℕ) 
  (h_num_students : num_students = 920) 
  (h_seats_per_bus : seats_per_bus = 71) 
  (h_num_buses : num_buses = 16) 
  (h_class_size_bound : ∀ c, c ≤ k) : 
  k = 17 :=
sorry

end max_students_per_class_l381_381539


namespace triangle_area_is_24_l381_381570

structure Point where
  x : ℝ
  y : ℝ

def distance_x (A B : Point) : ℝ :=
  abs (B.x - A.x)

def distance_y (A C : Point) : ℝ :=
  abs (C.y - A.y)

def triangle_area (A B C : Point) : ℝ :=
  0.5 * distance_x A B * distance_y A C

noncomputable def A : Point := ⟨2, 2⟩
noncomputable def B : Point := ⟨8, 2⟩
noncomputable def C : Point := ⟨4, 10⟩

theorem triangle_area_is_24 : triangle_area A B C = 24 := 
  sorry

end triangle_area_is_24_l381_381570


namespace roots_of_equation_l381_381126

theorem roots_of_equation (a b c : ℝ) (h : a ≠ b ∧ b ≠ c ∧ a ≠ c) :
  ∀ x : ℝ, a^2 * (x - b) / (a - b) * (x - c) / (a - c) + b^2 * (x - a) / (b - a) * (x - c) / (b - c) + c^2 * (x - a) / (c - a) * (x - b) / (c - b) = x^2 :=
by
  intros
  sorry

end roots_of_equation_l381_381126


namespace cube_minus_self_divisible_by_10_l381_381723

theorem cube_minus_self_divisible_by_10 (k : ℤ) : 10 ∣ ((5 * k) ^ 3 - 5 * k) :=
by sorry

end cube_minus_self_divisible_by_10_l381_381723


namespace complement_union_M_N_eq_ge_2_l381_381962

def U := set.univ ℝ
def M := {x : ℝ | x < 1}
def N := {x : ℝ | -1 < x ∧ x < 2}

theorem complement_union_M_N_eq_ge_2 :
  (U \ (M ∪ N)) = {x : ℝ | 2 ≤ x} :=
by sorry

end complement_union_M_N_eq_ge_2_l381_381962


namespace marked_price_of_appliance_l381_381691

theorem marked_price_of_appliance (x : ℝ) 
  (h1 : ∀ y, y ≤ 200 → (discount y = 0))
  (h2 : ∀ y, 200 < y ∧ y ≤ 500 → (discount y = (y - 200) * 0.1))
  (h3 : ∀ y, y > 500 → (discount y = (300 - 300 * 0.9) + (y - 500) * 0.2))
  (h4 : discount x = 330) : 
  x = 2000 :=
by
  sorry

end marked_price_of_appliance_l381_381691


namespace central_angle_of_sector_l381_381885

theorem central_angle_of_sector 
  (r : ℝ) (s : ℝ) (c : ℝ)
  (h1 : r = 5)
  (h2 : s = 15)
  (h3 : c = 2 * π * r) :
  ∃ n : ℝ, (n * s * π / 180 = c) ∧ n = 120 :=
by
  use 120
  sorry

end central_angle_of_sector_l381_381885


namespace correct_assignment_l381_381773

structure GirlDressAssignment :=
  (Katya : String)
  (Olya : String)
  (Liza : String)
  (Rita : String)

def solution : GirlDressAssignment :=
  ⟨"Green", "Blue", "Pink", "Yellow"⟩

theorem correct_assignment
  (Katya_not_pink_or_blue : solution.Katya ≠ "Pink" ∧ solution.Katya ≠ "Blue")
  (Green_between_Liza_and_Yellow : 
    (solution.Katya = "Green" ∧ solution.Liza = "Pink" ∧ solution.Rita = "Yellow") ∧
    (solution.Katya = "Green" ∧ solution.Rita = "Yellow" ∧ solution.Liza = "Pink"))
  (Rita_not_green_or_blue : solution.Rita ≠ "Green" ∧ solution.Rita ≠ "Blue")
  (Olya_between_Rita_and_Pink : 
    (solution.Olya = "Blue" ∧ solution.Rita = "Yellow" ∧ solution.Liza = "Pink") ∧
    (solution.Olya = "Blue" ∧ solution.Liza = "Pink" ∧ solution.Rita = "Yellow"))
  : solution = ⟨"Green", "Blue", "Pink", "Yellow"⟩ := by
  sorry

end correct_assignment_l381_381773


namespace min_max_magnitudes_proof_l381_381840

noncomputable def min_max_sum_magnitudes (a b : ℝ × ℝ) (ha : ‖a‖ = 1) (hb : ‖b‖ = 2) :
  ℝ × ℝ :=
let s := ‖(a.1 + b.1, a.2 + b.2)‖ + ‖(a.1 - b.1, a.2 - b.2)‖ in
(4, 2 * Real.sqrt 5)

theorem min_max_magnitudes_proof (a b : ℝ × ℝ) (ha : ‖a‖ = 1) (hb : ‖b‖ = 2) :
  min_max_sum_magnitudes a b ha hb = (4, 2 * Real.sqrt 5) :=
sorry

end min_max_magnitudes_proof_l381_381840


namespace expression_simplification_l381_381670

theorem expression_simplification :
  (2 ^ 2 / 3 + (-(3 ^ 2) + 5) + (-(3) ^ 2) * ((2 / 3) ^ 2)) = 4 / 3 :=
sorry

end expression_simplification_l381_381670


namespace value_of_x_squared_plus_inverse_squared_l381_381816

theorem value_of_x_squared_plus_inverse_squared (x : ℝ) (hx : x ≠ 0) (h : x^4 + (1 / x^4) = 2) : x^2 + (1 / x^2) = 2 :=
sorry

end value_of_x_squared_plus_inverse_squared_l381_381816


namespace complement_union_eq_ge2_l381_381947

open Set

variables {U : Type} [PartialOrder U] [LinearOrder U]

def U : Set ℝ := univ
def M : Set ℝ := { x | x < 1 }
def N : Set ℝ := { x | -1 < x ∧ x < 2 }
def Complement_U (A : Set ℝ) : Set ℝ := U \ A

theorem complement_union_eq_ge2 : 
  Complement_U (M ∪ N) = { x : ℝ | x ≥ 2 } :=
by {
  sorry
}

end complement_union_eq_ge2_l381_381947


namespace dogs_with_flea_collars_l381_381193

-- Conditions
def T : ℕ := 80
def Tg : ℕ := 45
def B : ℕ := 6
def N : ℕ := 1

-- Goal: prove the number of dogs with flea collars is 40 given the above conditions
theorem dogs_with_flea_collars : ∃ F : ℕ, F = 40 ∧ T = Tg + F - B + N := 
by
  use 40
  sorry

end dogs_with_flea_collars_l381_381193


namespace initial_amount_l381_381420

theorem initial_amount (H P L : ℝ) (C : ℝ) (n : ℕ) (T M : ℝ) 
  (hH : H = 10) 
  (hP : P = 2) 
  (hC : C = 1.25) 
  (hn : n = 4) 
  (hL : L = 3) 
  (hT : T = H + P + n * C) 
  (hM : M = T + L) : 
  M = 20 := 
sorry

end initial_amount_l381_381420


namespace factorization_l381_381271

def f (x y z : ℝ) : ℝ :=
  (y^2 - z^2) * (1 + x * y) * (1 + x * z) + 
  (z^2 - x^2) * (1 + y * z) * (1 + x * y) + 
  (x^2 - y^2) * (1 + y * z) * (1 + x * z)

theorem factorization (x y z : ℝ) :
  f x y z = (y - z) * (z - x) * (x - y) * (x * y * z + x + y + z) :=
by 
  sorry

end factorization_l381_381271


namespace num_solutions_eq_3_l381_381015

theorem num_solutions_eq_3 : 
  ∃ (x1 x2 x3 : ℝ), (∀ x : ℝ, 2^x - 2 * (⌊x⌋:ℝ) - 1 = 0 → x = x1 ∨ x = x2 ∨ x = x3) 
  ∧ ¬ ∃ x4, (2^x4 - 2 * (⌊x4⌋:ℝ) - 1 = 0 ∧ x4 ≠ x1 ∧ x4 ≠ x2 ∧ x4 ≠ x3) :=
sorry

end num_solutions_eq_3_l381_381015


namespace find_n_find_rational_terms_find_largest_terms_l381_381377

-- The first proof problem statement: find n
theorem find_n :
  (∀ (x : ℝ), 
   let a := C n 0 ::
   let b := (1 / 2^2) * C n 2 ::
   let c := (2 * (1 / 2) * C n 1) ::
   (a + b + c = 0) -> n = 8) := sorry

-- The second proof problem statement: find the rational terms
theorem find_rational_terms (n : ℕ) :
  (∀ (x : ℝ), 
   let expr := (sqrt x + 1 / 2 / sqrt (sqrt x)) ^ n ::
   let terms := [x^4, 35/8 * x, 1/256/x^2] in
   (expansion expr).filter (λ t, is_rational(t)) = terms) := sorry

-- The third proof problem statement: find the terms with the largest coefficient
theorem find_largest_terms (n : ℕ) :
  (∀ (x : ℝ), 
   let terms := [7 * x^(5/2), 7 * x^(7/4)] in
   (expansion_with_largest_coeff (sqrt x + 1 / 2 / sqrt (sqrt x))^n) = terms) := sorry

end find_n_find_rational_terms_find_largest_terms_l381_381377


namespace problem_equivalent_l381_381943

def U : Set ℝ := Set.univ
def M : Set ℝ := {x | x < 1}
def N : Set ℝ := {x | -1 < x ∧ x < 2}

theorem problem_equivalent :
  {x : ℝ | x ≥ 2} = (U \ (M ∪ N)) := 
by sorry

end problem_equivalent_l381_381943


namespace third_smallest_is_five_l381_381703

noncomputable def probability_third_smallest_is_five : ℚ :=
  let total_ways := (Nat.choose 15 8) in
  let favorable_ways := (Nat.choose 4 2) * (Nat.choose 10 5) in
  favorable_ways / total_ways

theorem third_smallest_is_five :
  probability_third_smallest_is_five = 4 / 17 := sorry

end third_smallest_is_five_l381_381703


namespace intersection_M_N_eq_1_l381_381835

-- Define M and N according to the conditions
def M : Set ℕ := {1, 2}
def N : Set ℝ := {x | x * (x - 2) < 0}

-- The proof goal
theorem intersection_M_N_eq_1 : M ∩ N = {1} := by
  sorry

end intersection_M_N_eq_1_l381_381835


namespace largest_divisor_of_product_of_seven_visible_numbers_l381_381186

theorem largest_divisor_of_product_of_seven_visible_numbers
  (die_faces : Finset ℕ) (h_die_faces : die_faces = {1, 2, 3, 4, 5, 6, 7, 8}) (hidden_face : ℕ)
  (h_hidden : hidden_face ∈ die_faces) :
  let Q := (die_faces.erase hidden_face).prod id
  in 48 ∣ Q :=
by
  sorry

end largest_divisor_of_product_of_seven_visible_numbers_l381_381186


namespace P_zero_value_l381_381927

noncomputable def P (x b c : ℚ) : ℚ := x ^ 2 + b * x + c

theorem P_zero_value (b c : ℚ)
  (h1 : P (P 1 b c) b c = 0)
  (h2 : P (P (-2) b c) b c = 0)
  (h3 : P 1 b c ≠ P (-2) b c) :
  P 0 b c = -5 / 2 :=
sorry

end P_zero_value_l381_381927


namespace angle_at_vertex_C_is_45_degrees_l381_381048

variable (A B C O : Type)
variable [EuclideanGeometry A]
variable [EuclideanGeometry B]
variable [EuclideanGeometry C]
variable [EuclideanGeometry O]

variable (ABC : Triangle A B C)
variable (altitudes_intersect_at_O : Orthocenter ABC = O)
variable (OC_eq_AB : dist O C = dist A B)

theorem angle_at_vertex_C_is_45_degrees :
  ∠ A C B = 45 := by
  sorry

end angle_at_vertex_C_is_45_degrees_l381_381048


namespace hockey_league_total_games_l381_381146

theorem hockey_league_total_games 
  (divisions : ℕ)
  (teams_per_division : ℕ)
  (intra_division_games : ℕ)
  (inter_division_games : ℕ) :
  divisions = 2 →
  teams_per_division = 6 →
  intra_division_games = 4 →
  inter_division_games = 2 →
  (divisions * ((teams_per_division * (teams_per_division - 1)) / 2) * intra_division_games) + 
  ((divisions / 2) * (divisions / 2) * teams_per_division * teams_per_division * inter_division_games) = 192 :=
by
  intros h_div h_teams h_intra h_inter
  sorry

end hockey_league_total_games_l381_381146


namespace AmyBenDifference_l381_381661

theorem AmyBenDifference :
  let A := 12 - (3 * 4)
  let B := (12 - 3) * 4
  A - B = -36 :=
by
  let A := 12 - (3 * 4)
  let B := (12 - 3) * 4
  show A - B = -36 from sorry

end AmyBenDifference_l381_381661


namespace greatest_integer_y_l381_381574

theorem greatest_integer_y (y : ℤ) : (8 : ℚ) / 11 > y / 17 ↔ y ≤ 12 := 
sorry

end greatest_integer_y_l381_381574


namespace sum_first_110_terms_l381_381079

variable (a d : ℕ → ℤ) [is_arithmetic_sequence: ∀ n, a (n + 1) = a n + d n]

-- Given that the sum of the first 10 terms is 100
def sum_first_10_terms := (∑ i in Finset.range 10, a i) = 100

-- Given that the sum of the first 100 terms is 10
def sum_first_100_terms := (∑ i in Finset.range 100, a i) = 10

-- Prove that the sum of the first 110 terms is -110
theorem sum_first_110_terms (h1 : sum_first_10_terms a) (h2 : sum_first_100_terms a) : 
  (∑ i in Finset.range 110, a i) = -110 :=
sorry

end sum_first_110_terms_l381_381079


namespace find_g_two_fifths_l381_381058

noncomputable def g : ℝ → ℝ :=
sorry -- The function g(x) is not explicitly defined.

theorem find_g_two_fifths :
  (∀ x, 0 ≤ x ∧ x ≤ 1 → g x = 0 → g 0 = 0) ∧
  (∀ x y, 0 ≤ x → x < y → y ≤ 1 → g x ≤ g y) ∧
  (∀ x, 0 ≤ x ∧ x ≤ 1 → g (1 - x) = 1 - g x) ∧
  (∀ x, 0 ≤ x ∧ x ≤ 1 → g (x / 5) = g x / 3)
  → g (2 / 5) = 1 / 3 :=
sorry

end find_g_two_fifths_l381_381058


namespace dress_assignments_l381_381745

structure GirlDress : Type :=
  (Katya Olya Liza Rita : String)

def dresses := ["Pink", "Green", "Yellow", "Blue"]

axiom not_pink_or_blue : GirlDress.Katya ≠ "Pink" ∧ GirlDress.Katya ≠ "Blue"
axiom green_between_liza_yellow : (GirlDress.Liza = "Pink" ∨ GirlDress.Rita = "Yellow") ∧
                                  GirlDress.Katya = "Green" ∧
                                  GirlDress.Rita = "Yellow" ∧ GirlDress.Liza = "Pink"
axiom not_green_or_blue : GirlDress.Rita ≠ "Green" ∧ GirlDress.Rita ≠ "Blue"
axiom olya_between_rita_pink : GirlDress.Olya ≠ "Pink" → GirlDress.Rita ≠ "Pink" → GirlDress.Liza = "Pink"

theorem dress_assignments (gd : GirlDress) :
  gd.Katya = "Green" ∧ gd.Olya = "Blue" ∧ gd.Liza = "Pink" ∧ gd.Rita = "Yellow" :=
by
  sorry

end dress_assignments_l381_381745


namespace derivative_at_1_is_neg_half_l381_381350

noncomputable def f (f'_1 : ℝ) (x : ℝ) : ℝ := sqrt x + 2 * f'_1 * x

theorem derivative_at_1_is_neg_half :
  ∃ f'_1 : ℝ, (∂ (fun x => sqrt x + 2 * f'_1 * x) x| 1) = -1 / 2 :=
begin
  use -1 / 2,
  sorry
end

end derivative_at_1_is_neg_half_l381_381350


namespace term_no_x_in_binomial_expansion_l381_381081

theorem term_no_x_in_binomial_expansion :
    ∃ (r : ℕ), 6 - 2 * r = 0 ∧
                  (∀ y ≠ 0, 
                    (∃ T, 
                      T = (-1:ℤ)^r * (Nat.choose 6 r) * y^(r-6) ∧ 
                      T = -20 * y^(-3))) :=
by
  sorry

end term_no_x_in_binomial_expansion_l381_381081


namespace dress_assignment_l381_381762

theorem dress_assignment :
  ∃ (Katya Olya Liza Rita : string),
    (Katya ≠ "Pink" ∧ Katya ≠ "Blue") ∧
    (Rita ≠ "Green" ∧ Rita ≠ "Blue") ∧
    ∃ (girl_in_green girl_in_yellow : string),
      (girl_in_green = Katya ∧ girl_in_yellow = Rita ∧ 
       (Liza = "Pink" ∧ Olya = "Blue") ∧
       (Katya = "Green" ∧ Olya = "Blue" ∧ Liza = "Pink" ∧ Rita = "Yellow")) ∧
    ((girl_in_green stands between Liza and girl_in_yellow) ∧
     (Olya stands between Rita and Liza)) :=
by
  sorry

end dress_assignment_l381_381762


namespace permutation_identity_l381_381314

/-
Given permutations A_m^n representing the number of ways to arrange n items out of m items, expressed as m! / (m - n)!. Prove that for n = 9, the equation 3 * (8! / (8 - (n - 1))!) = 4 * (9! / (9 - (n - 2))!) holds.
-/
open_locale big_operators

noncomputable def A (m n : ℕ) : ℕ := m! / (m - n)!

theorem permutation_identity (n : ℕ) (h : n = 9) : 
  3 * (A 8 (n - 1)) = 4 * (A 9 (n - 2)) :=
by
  rw h
  sorry

end permutation_identity_l381_381314


namespace f_periodic_function_l381_381810

noncomputable def f : ℝ → ℝ := sorry

theorem f_periodic_function (h1 : ∀ x : ℝ, f (-x) = f x)
    (h2 : ∀ x : ℝ, f (x + 4) = f x + f 2)
    (h3 : f 1 = 2) : 
    f 2013 = 2 := sorry

end f_periodic_function_l381_381810


namespace log_sum_eq_l381_381252

theorem log_sum_eq : ((∑ k in (Finset.range 98).map (λ n, n+3), (Real.log (1 + 1 / (k:ℝ)) / Real.log 3) * (Real.log 3 / Real.log k) * (Real.log 3 / Real.log (k+1)))) = 1 - (1 / Real.log 101 / Real.log 3) :=
by
  sorry

end log_sum_eq_l381_381252


namespace coplanar_points_l381_381284

theorem coplanar_points (b : ℝ) : 
  let v1 := ![1, 0, b],
      v2 := ![0, 1, b],
      v3 := ![b, 1, 0] in
  matrix.det ![
    v1.to_list,
    v2.to_list,
    v3.to_list
  ] = 0 ↔ b = 0 ∨ b = -1 :=
by sorry

end coplanar_points_l381_381284


namespace cos_A_given_conditions_l381_381313

theorem cos_A_given_conditions (A : ℝ) (h1 : 0 < A) (h2 : A < π / 2) (h3 : cos (2 * A) = 3 / 5) : 
  cos A = 2 * sqrt 5 / 5 :=
by
  sorry

end cos_A_given_conditions_l381_381313


namespace complement_union_eq_ge2_l381_381953

open Set

variables {U : Type} [PartialOrder U] [LinearOrder U]

def U : Set ℝ := univ
def M : Set ℝ := { x | x < 1 }
def N : Set ℝ := { x | -1 < x ∧ x < 2 }
def Complement_U (A : Set ℝ) : Set ℝ := U \ A

theorem complement_union_eq_ge2 : 
  Complement_U (M ∪ N) = { x : ℝ | x ≥ 2 } :=
by {
  sorry
}

end complement_union_eq_ge2_l381_381953


namespace isos_triangle_BKD_with_BK_and_BD_as_legs_l381_381397

-- Given an isosceles triangle ABC with AB = AC
variables (A B C K D : Point)
variables (AB AC BC CD BK BD : Length)
variables (h_iso : AB = AC)
variables (h_bisector1 : IsAngleBisector A B C K)
variables (h_pointD : OnLine D AC)
variables (h_equalBC : BC = CD)

theorem isos_triangle_BKD_with_BK_and_BD_as_legs :
  isosceles_triangle BK BD := by
  sorry

end isos_triangle_BKD_with_BK_and_BD_as_legs_l381_381397


namespace price_of_table_l381_381167

-- Given the conditions:
def chair_table_eq1 (C T : ℝ) : Prop := 2 * C + T = 0.6 * (C + 2 * T)
def chair_table_eq2 (C T : ℝ) : Prop := C + T = 72

-- Prove that the price of one table is $63
theorem price_of_table (C T : ℝ) (h1 : chair_table_eq1 C T) (h2 : chair_table_eq2 C T) : T = 63 := by
  sorry

end price_of_table_l381_381167


namespace friday_vs_tuesday_l381_381460

def tuesday_amount : ℝ := 8.5
def wednesday_amount : ℝ := 5.5 * tuesday_amount
def thursday_amount : ℝ := wednesday_amount + 0.10 * wednesday_amount
def friday_amount : ℝ := 0.75 * thursday_amount

theorem friday_vs_tuesday :
  friday_amount - tuesday_amount = 30.06875 :=
sorry

end friday_vs_tuesday_l381_381460


namespace no_pairs_of_positive_integers_l381_381687

theorem no_pairs_of_positive_integers (x y : ℕ) (hx : 0 < x) (hy : 0 < y) : ¬ (x^2 + y^2 = x^4) :=
begin
  sorry
end

end no_pairs_of_positive_integers_l381_381687


namespace jasmine_milk_gallons_l381_381918

theorem jasmine_milk_gallons (G : ℝ) 
  (coffee_cost_per_pound : ℝ) (milk_cost_per_gallon : ℝ) (total_cost : ℝ)
  (coffee_pounds : ℝ) :
  coffee_cost_per_pound = 2.50 →
  milk_cost_per_gallon = 3.50 →
  total_cost = 17 →
  coffee_pounds = 4 →
  total_cost - coffee_pounds * coffee_cost_per_pound = G * milk_cost_per_gallon →
  G = 2 :=
by
  intros
  sorry

end jasmine_milk_gallons_l381_381918


namespace train_length_110_l381_381169

variable {L : ℕ}  -- Length of the train in meters
variable {S : ℕ}  -- Speed of the train in meters per second

-- Conditions
def condition1 := S = (L + 160) / 15
def condition2 := S = (L + 250) / 20

theorem train_length_110 (h1 : condition1) (h2 : condition2) : L = 110 := by sorry

end train_length_110_l381_381169


namespace find_focus_of_parabola_l381_381288

open Real

theorem find_focus_of_parabola :
  ∃ (x y : ℝ), y = 2 * x^2 - 4 * x - 1 ∧ focus_x = 1 ∧ focus_y = -23 / 8 :=
by
  let vertex_x : ℝ := 1
  let vertex_y : ℝ := -3
  let vertex_distance_to_focus : ℝ := 1 / (4 * 2)
  let focus_x := vertex_x
  let focus_y := vertex_y + vertex_distance_to_focus
  existsi (focus_x, focus_y)
  show y = 2 * x^2 - 4 * x - 1
  sorry

end find_focus_of_parabola_l381_381288


namespace divisible_by_4003_l381_381035

-- Define the factorial up to n (as a helper)
noncomputable def prod_up_to (n : ℕ) : ℕ := List.prod (List.range (n + 1))

-- Define the main problem statement
theorem divisible_by_4003 :
  (prod_up_to 2001 + prod_up_to 2002 2001 - 4002) % 4003 = 0 :=
sorry

end divisible_by_4003_l381_381035


namespace sqrt_pow_inv_l381_381671

theorem sqrt_pow_inv (h1 : Real.sqrt 9 = 3) (h2 : (-2022)^0 = 1) (h3 : 2⁻¹ = 1/2) : 
  Real.sqrt 9 - (-2022)^0 + 2⁻¹ = 5 / 2 := 
by 
  sorry

end sqrt_pow_inv_l381_381671


namespace translate_line_upwards_l381_381905

theorem translate_line_upwards (x y : ℝ) (h : y = x) : y + 2 = x + 2 :=
by {
  rw h,
}

end translate_line_upwards_l381_381905


namespace problem_statement_l381_381519

def count_special_numbers : ℕ := 18

theorem problem_statement :
  let count_identical_digits (n : ℕ) : ℕ :=
    if h1 : (1000 <= n ∧ n < 10000) ∧ (n / 1000 = 1) ∧ 
            ((n / 100 % 10 = n / 10 % 10 ∧ n / 10 % 10 = n % 10) ∨ 
             (n / 100 % 10 = n / 1000 ∧ n / 10 % 10 = n / 100 % 10) ∨ 
             (n % 10 = n / 100 % 10 ∧ n % 10 = n / 10 % 10)) 
    then 1 
    else 0
  in ∑ n in (list.range 9000), count_identical_digits (n + 1000) = count_special_numbers :=
sorry

end problem_statement_l381_381519


namespace dress_assignments_l381_381747

structure GirlDress : Type :=
  (Katya Olya Liza Rita : String)

def dresses := ["Pink", "Green", "Yellow", "Blue"]

axiom not_pink_or_blue : GirlDress.Katya ≠ "Pink" ∧ GirlDress.Katya ≠ "Blue"
axiom green_between_liza_yellow : (GirlDress.Liza = "Pink" ∨ GirlDress.Rita = "Yellow") ∧
                                  GirlDress.Katya = "Green" ∧
                                  GirlDress.Rita = "Yellow" ∧ GirlDress.Liza = "Pink"
axiom not_green_or_blue : GirlDress.Rita ≠ "Green" ∧ GirlDress.Rita ≠ "Blue"
axiom olya_between_rita_pink : GirlDress.Olya ≠ "Pink" → GirlDress.Rita ≠ "Pink" → GirlDress.Liza = "Pink"

theorem dress_assignments (gd : GirlDress) :
  gd.Katya = "Green" ∧ gd.Olya = "Blue" ∧ gd.Liza = "Pink" ∧ gd.Rita = "Yellow" :=
by
  sorry

end dress_assignments_l381_381747


namespace complement_union_eq_ge_two_l381_381993

def U : Set ℝ := Set.univ
def M : Set ℝ := { x : ℝ | x < 1 }
def N : Set ℝ := { x : ℝ | -1 < x ∧ x < 2 }

theorem complement_union_eq_ge_two : { x : ℝ | x ≥ 2 } = U \ (M ∪ N) :=
by
  sorry

end complement_union_eq_ge_two_l381_381993


namespace time_after_interval_l381_381916

/-- 
Given an initial time on a 12-hour digital clock of 3:15:15 PM,
and a time interval to be added of 196 hours, 58 minutes, and 16 seconds,
we aim to find the time in the format A:B:C and verify that the sum A + B + C equals 52.
-/
theorem time_after_interval (init_hour init_min init_sec added_hours added_min added_sec : ℕ) 
    (h_ini : init_hour = 3)
    (m_ini : init_min = 15)
    (s_ini : init_sec = 15)
    (h_add : added_hours = 196)
    (m_add : added_min = 58)
    (s_add : added_sec = 16) :
  let final_hour := 8 in
  let final_min := 13 in
  let final_sec := 31 in
  let A := final_hour in
  let B := final_min in
  let C := final_sec in
  A + B + C = 52 :=
by
  sorry

end time_after_interval_l381_381916


namespace dot_product_AB_AC_l381_381326

noncomputable def OA := vector ℝ 3
noncomputable def OB := vector ℝ 3
noncomputable def OC := vector ℝ 3

def unit_vector (v : vector ℝ 3) : Prop := (v.dot v = 1)

axiom OA_unit : unit_vector OA
axiom OB_unit : unit_vector OB
axiom OC_unit : unit_vector OC

axiom OA_OB_OC_relation : (1/2) • OA + OB + OC = 0

def AB := OB - OA
def AC := OC - OA

theorem dot_product_AB_AC :
  AB ⋅ AC = 5 / 8 :=
by sorry

end dot_product_AB_AC_l381_381326


namespace greatest_integer_l381_381581

theorem greatest_integer (y : ℤ) (h : (8 : ℚ) / 11 > y / 17) : y ≤ 12 :=
by
  have h₁ : (8 : ℚ) / 11 * 17 > y := by exact (div_mul_cancel _ (by norm_num : 17 ≠ 0))
  have h₂ : 136 / 11 > y := by rwa mul_comm _ 17 at h₁
  exact_mod_cast le_of_lt h₂

end greatest_integer_l381_381581
