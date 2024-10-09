import Mathlib

namespace find_payment_y_l781_78146

variable (X Y : Real)

axiom h1 : X + Y = 570
axiom h2 : X = 1.2 * Y

theorem find_payment_y : Y = 570 / 2.2 := by
  sorry

end find_payment_y_l781_78146


namespace car_not_sold_probability_l781_78150

theorem car_not_sold_probability (a b : ℕ) (h : a = 5) (k : b = 6) : (b : ℚ) / (a + b : ℚ) = 6 / 11 :=
  by
    rw [h, k]
    norm_num

end car_not_sold_probability_l781_78150


namespace suitable_M_unique_l781_78124

noncomputable def is_suitable_M (M : ℝ) : Prop :=
  ∀ (a b c : ℝ), (0 < a) → (0 < b) → (0 < c) →
  (1 + M ≤ a + M / (a * b)) ∨ 
  (1 + M ≤ b + M / (b * c)) ∨ 
  (1 + M ≤ c + M / (c * a))

theorem suitable_M_unique : is_suitable_M (1/2) ∧ 
  (∀ (M : ℝ), is_suitable_M M → M = 1/2) :=
by
  sorry

end suitable_M_unique_l781_78124


namespace jenny_total_distance_seven_hops_l781_78138

noncomputable def sum_geometric_series (a r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r ^ n) / (1 - r)

theorem jenny_total_distance_seven_hops :
  let a := (1 / 4 : ℚ)
  let r := (3 / 4 : ℚ)
  let n := 7
  sum_geometric_series a r n = (14197 / 16384 : ℚ) :=
by
  sorry

end jenny_total_distance_seven_hops_l781_78138


namespace calc_expression_l781_78130

theorem calc_expression : 2012 * 2016 - 2014^2 = -4 := by
  sorry

end calc_expression_l781_78130


namespace similar_triangles_PQ_length_l781_78187

theorem similar_triangles_PQ_length (XY YZ QR : ℝ) (hXY : XY = 8) (hYZ : YZ = 16) (hQR : QR = 24)
  (hSimilar : ∃ (k : ℝ), XY = k * 8 ∧ YZ = k * 16 ∧ QR = k * 24) : (∃ (PQ : ℝ), PQ = 12) :=
by 
  -- Here we need to prove the theorem using similarity and given equalities
  sorry

end similar_triangles_PQ_length_l781_78187


namespace total_cookies_l781_78143

-- Define the number of bags and cookies per bag
def num_bags : Nat := 37
def cookies_per_bag : Nat := 19

-- The theorem stating the total number of cookies
theorem total_cookies : num_bags * cookies_per_bag = 703 := by
  sorry

end total_cookies_l781_78143


namespace common_point_geometric_lines_l781_78165

-- Define that a, b, c form a geometric progression given common ratio r
def geometric_prog (a b c r : ℝ) : Prop := b = a * r ∧ c = a * r^2

-- Prove that all lines with the equation ax + by = c pass through the point (-1, 1)
theorem common_point_geometric_lines (a b c r x y : ℝ) (h : geometric_prog a b c r) :
  a * x + b * y = c → (x, y) = (-1, 1) :=
by
  sorry

end common_point_geometric_lines_l781_78165


namespace tommy_gum_given_l781_78117

variable (original_gum : ℕ) (luis_gum : ℕ) (final_total_gum : ℕ)

-- Defining the conditions
def conditions := original_gum = 25 ∧ luis_gum = 20 ∧ final_total_gum = 61

-- The theorem stating that Tommy gave Maria 16 pieces of gum
theorem tommy_gum_given (t_gum : ℕ) (h : conditions original_gum luis_gum final_total_gum) :
  t_gum = final_total_gum - (original_gum + luis_gum) → t_gum = 16 :=
by
  intros h
  sorry

end tommy_gum_given_l781_78117


namespace polynomial_degree_is_14_l781_78112

noncomputable def polynomial_degree (a b c d e f g h : ℝ) : ℕ :=
  if a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧ e ≠ 0 ∧ f ≠ 0 ∧ g ≠ 0 ∧ h ≠ 0 then 14 else 0

theorem polynomial_degree_is_14 (a b c d e f g h : ℝ) (h_neq0 : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧ e ≠ 0 ∧ f ≠ 0 ∧ g ≠ 0 ∧ h ≠ 0) :
  polynomial_degree a b c d e f g h = 14 :=
by sorry

end polynomial_degree_is_14_l781_78112


namespace parabola_line_intersection_sum_l781_78162

theorem parabola_line_intersection_sum (r s : ℝ) (h_r : r = 20 - 10 * Real.sqrt 38) (h_s : s = 20 + 10 * Real.sqrt 38) :
  r + s = 40 := by
  sorry

end parabola_line_intersection_sum_l781_78162


namespace cone_volume_l781_78126

theorem cone_volume (l : ℝ) (h : ℝ) (r : ℝ) (V : ℝ) 
  (hl : l = 15)    -- slant height
  (hh : h = 9)     -- vertical height
  (hr : r^2 = 144) -- radius squared from Pythagorean theorem
  : V = 432 * Real.pi :=
by
  -- Proof is omitted. Hence, we write sorry to denote skipped proof.
  sorry

end cone_volume_l781_78126


namespace valid_seating_arrangements_l781_78182

def num_people : Nat := 10
def total_arrangements : Nat := Nat.factorial num_people
def restricted_group_arrangements : Nat := Nat.factorial 7 * Nat.factorial 4
def valid_arrangements : Nat := total_arrangements - restricted_group_arrangements

theorem valid_seating_arrangements : valid_arrangements = 3507840 := by
  sorry

end valid_seating_arrangements_l781_78182


namespace equidistant_points_quadrants_l781_78120

open Real

theorem equidistant_points_quadrants : 
  ∀ x y : ℝ, 
    (4 * x + 6 * y = 24) → (|x| = |y|) → 
    ((0 < x ∧ 0 < y) ∨ (x < 0 ∧ 0 < y)) :=
by
  sorry

end equidistant_points_quadrants_l781_78120


namespace min_value_of_expression_l781_78161

theorem min_value_of_expression (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h_sum : x + y + z = 1) : 
  36 ≤ (1/x + 4/y + 9/z) :=
sorry

end min_value_of_expression_l781_78161


namespace rectangle_width_to_length_ratio_l781_78109

theorem rectangle_width_to_length_ratio {w : ℕ} 
  (h1 : ∀ (l : ℕ), l = 10)
  (h2 : ∀ (p : ℕ), p = 32)
  (h3 : ∀ (P : ℕ), P = 2 * 10 + 2 * w) :
  (w : ℚ) / 10 = 3 / 5 :=
by
  sorry

end rectangle_width_to_length_ratio_l781_78109


namespace combined_moment_l781_78103

-- Definitions based on given conditions
variables (P Q Z : ℝ) -- Positions of the points and center of mass
variables (p q : ℝ) -- Masses of the points
variables (Mom_s : ℝ → ℝ) -- Moment function relative to axis s

-- Given:
-- 1. Positions P and Q with masses p and q respectively
-- 2. Combined point Z with total mass p + q
-- 3. Moments relative to the axis s: Mom_s P and Mom_s Q
-- To Prove: Moment of the combined point Z relative to axis s
-- is the sum of the moments of P and Q relative to the same axis

theorem combined_moment (hZ : Z = (P * p + Q * q) / (p + q)) :
  Mom_s Z = Mom_s P + Mom_s Q :=
sorry

end combined_moment_l781_78103


namespace heights_inscribed_circle_inequality_l781_78186

theorem heights_inscribed_circle_inequality
  {h₁ h₂ r : ℝ} (h₁_pos : 0 < h₁) (h₂_pos : 0 < h₂) (r_pos : 0 < r)
  (triangle_heights : ∃ (a b c : ℝ), 0 < a ∧ 0 < b ∧ 0 < c ∧ a * h₁ = b * h₂ ∧ 
                                       a + b > c ∧ h₁ = 2 * r * (a + b + c) / (a * b)):
  (1 / (2 * r) < 1 / h₁ + 1 / h₂ ∧ 1 / h₁ + 1 / h₂ < 1 / r) :=
sorry

end heights_inscribed_circle_inequality_l781_78186


namespace octagon_diagonal_ratio_l781_78197

theorem octagon_diagonal_ratio (P : ℝ → ℝ → Prop) (d1 d2 : ℝ) (h1 : P d1 d2) : d1 / d2 = Real.sqrt 2 / 2 :=
sorry

end octagon_diagonal_ratio_l781_78197


namespace min_fence_posts_needed_l781_78156

-- Definitions for the problem conditions
def area_length : ℕ := 72
def regular_side : ℕ := 30
def sloped_side : ℕ := 33
def interval : ℕ := 15

-- The property we want to prove
theorem min_fence_posts_needed : 3 * ((sloped_side + interval - 1) / interval) + 3 * ((regular_side + interval - 1) / interval) = 6 := 
by
  sorry

end min_fence_posts_needed_l781_78156


namespace children_per_block_l781_78184

theorem children_per_block {children total_blocks : ℕ} 
  (h_total_blocks : total_blocks = 9) 
  (h_total_children : children = 54) : 
  (children / total_blocks = 6) :=
by
  -- Definitions from conditions
  have h1 : total_blocks = 9 := h_total_blocks
  have h2 : children = 54 := h_total_children

  -- Goal to prove
  -- children / total_blocks = 6
  sorry

end children_per_block_l781_78184


namespace triangle_area_l781_78135

theorem triangle_area (X Y Z : ℝ) (r R : ℝ)
  (h1 : r = 7)
  (h2 : R = 25)
  (h3 : 2 * Real.cos Y = Real.cos X + Real.cos Z) :
  ∃ (p q r : ℕ), (p * Real.sqrt q / r = 133) ∧ (p + q + r = 135) :=
  sorry

end triangle_area_l781_78135


namespace old_man_coins_l781_78139

theorem old_man_coins (x y : ℕ) (h : x ≠ y) (h_condition : x^2 - y^2 = 81 * (x - y)) : x + y = 81 := 
sorry

end old_man_coins_l781_78139


namespace erwan_spending_l781_78114

def discount (price : ℕ) (percent : ℕ) : ℕ :=
  price - (price * percent / 100)

theorem erwan_spending (shoe_original_price : ℕ := 200) 
  (shoe_discount : ℕ := 30)
  (shirt_price : ℕ := 80)
  (num_shirts : ℕ := 2)
  (pants_price : ℕ := 150)
  (second_store_discount : ℕ := 20)
  (jacket_price : ℕ := 250)
  (tie_price : ℕ := 40)
  (hat_price : ℕ := 60)
  (watch_price : ℕ := 120)
  (wallet_price : ℕ := 49)
  (belt_price : ℕ := 35)
  (belt_discount : ℕ := 25)
  (scarf_price : ℕ := 45)
  (scarf_discount : ℕ := 10)
  (rewards_points_discount : ℕ := 5)
  (sales_tax : ℕ := 8)
  (gift_card : ℕ := 50)
  (shipping_fee : ℕ := 5)
  (num_shipping_stores : ℕ := 2) :
  ∃ total : ℕ,
    total = 85429 :=
by
  have first_store := discount shoe_original_price shoe_discount
  have second_store_total := pants_price + (shirt_price * num_shirts)
  have second_store := discount second_store_total second_store_discount
  have tie_half_price := tie_price / 2
  have hat_half_price := hat_price / 2
  have third_store := jacket_price + (tie_half_price + hat_half_price)
  have fourth_store := watch_price
  have fifth_store := discount belt_price belt_discount + discount scarf_price scarf_discount
  have subtotal := first_store + second_store + third_store + fourth_store + fifth_store
  have after_rewards_points := subtotal - (subtotal * rewards_points_discount / 100)
  have after_gift_card := after_rewards_points - gift_card
  have after_shipping_fees := after_gift_card + (shipping_fee * num_shipping_stores)
  have total := after_shipping_fees + (after_shipping_fees * sales_tax / 100)
  use total / 100 -- to match the monetary value in cents
  sorry

end erwan_spending_l781_78114


namespace plan1_maximizes_B_winning_probability_l781_78128

open BigOperators

-- Definitions for the conditions
def prob_A_wins : ℚ := 3/4
def prob_B_wins : ℚ := 1/4

-- Plan 1 probabilities
def prob_B_win_2_0 : ℚ := prob_B_wins^2
def prob_B_win_2_1 : ℚ := (Nat.choose 2 1) * prob_B_wins * prob_A_wins * prob_B_wins
def prob_B_win_plan1 : ℚ := prob_B_win_2_0 + prob_B_win_2_1

-- Plan 2 probabilities
def prob_B_win_3_0 : ℚ := prob_B_wins^3
def prob_B_win_3_1 : ℚ := (Nat.choose 3 1) * prob_B_wins^2 * prob_A_wins * prob_B_wins
def prob_B_win_3_2 : ℚ := (Nat.choose 4 2) * prob_B_wins^2 * prob_A_wins^2 * prob_B_wins
def prob_B_win_plan2 : ℚ := prob_B_win_3_0 + prob_B_win_3_1 + prob_B_win_3_2

-- Theorem statement
theorem plan1_maximizes_B_winning_probability :
  prob_B_win_plan1 > prob_B_win_plan2 :=
by
  sorry

end plan1_maximizes_B_winning_probability_l781_78128


namespace container_volume_ratio_l781_78142

variables (A B C : ℝ)

theorem container_volume_ratio (h1 : (2 / 3) * A = (1 / 2) * B) (h2 : (1 / 2) * B = (3 / 5) * C) :
  A / C = 6 / 5 :=
sorry

end container_volume_ratio_l781_78142


namespace largest_result_l781_78157

theorem largest_result (a b c : ℕ) (h1 : a = 0 / 100) (h2 : b = 0 * 100) (h3 : c = 100 - 0) : 
  c > a ∧ c > b :=
by
  sorry

end largest_result_l781_78157


namespace reflected_circle_center_l781_78116

theorem reflected_circle_center
  (original_center : ℝ × ℝ) 
  (reflection_line : ℝ × ℝ → ℝ × ℝ)
  (hc : original_center = (8, -3))
  (hl : ∀ (p : ℝ × ℝ), reflection_line p = (-p.2, -p.1))
  : reflection_line original_center = (3, -8) :=
sorry

end reflected_circle_center_l781_78116


namespace find_number_l781_78136

theorem find_number (x : ℕ) (h : x + 15 = 96) : x = 81 := 
sorry

end find_number_l781_78136


namespace probability_even_sum_l781_78188

theorem probability_even_sum (x y : ℕ) (h : x + y ≤ 10) : 
  (∃ (p : ℚ), p = 6 / 11 ∧ (x + y) % 2 = 0) :=
sorry

end probability_even_sum_l781_78188


namespace tangent_line_equation_inequality_range_l781_78104

noncomputable def f (x : ℝ) : ℝ := x * Real.log x

theorem tangent_line_equation :
  let x := Real.exp 1
  ∀ e : ℝ, e = Real.exp 1 → 
  ∀ y : ℝ, y = f (Real.exp 1) → 
  ∀ a b : ℝ, (y = a * Real.exp 1 + b) ∧ (a = 2) ∧ (b = -e) := sorry

theorem inequality_range (x : ℝ) (hx : x > 0) :
  (f x - 1/2 ≤ (3/2) * x^2 + a * x) → ∀ a : ℝ, a ≥ -2 := sorry

end tangent_line_equation_inequality_range_l781_78104


namespace largest_possible_n_l781_78102

theorem largest_possible_n (k : ℕ) (hk : k > 0) : ∃ n, n = 3 * k - 1 := 
  sorry

end largest_possible_n_l781_78102


namespace maxEccentricity_l781_78191

noncomputable def majorAxisLength := 4
noncomputable def majorSemiAxis := 2
noncomputable def leftVertexParabolaEq (y : ℝ) := y^2 = -3
noncomputable def distanceCondition (c : ℝ) := 2^2 / c - 2 ≥ 1

theorem maxEccentricity : ∃ c : ℝ, distanceCondition c ∧ (c ≤ 4 / 3) ∧ (c / majorSemiAxis = 2 / 3) :=
by
  sorry

end maxEccentricity_l781_78191


namespace problem_l781_78148

theorem problem (f : ℕ → ℝ) 
  (h_def : ∀ x, f x = Real.cos (x * Real.pi / 3)) 
  (h_period : ∀ x, f (x + 6) = f x) : 
  (Finset.sum (Finset.range 2018) f) = 0 := 
by
  sorry

end problem_l781_78148


namespace JuanitaDessertCost_l781_78147

-- Define costs as constants
def brownieCost : ℝ := 2.50
def regularScoopCost : ℝ := 1.00
def premiumScoopCost : ℝ := 1.25
def deluxeScoopCost : ℝ := 1.50
def syrupCost : ℝ := 0.50
def nutsCost : ℝ := 1.50
def whippedCreamCost : ℝ := 0.75
def cherryCost : ℝ := 0.25

-- Define the total cost calculation
def totalCost : ℝ := brownieCost + regularScoopCost + premiumScoopCost +
                     deluxeScoopCost + syrupCost + syrupCost + nutsCost + whippedCreamCost + cherryCost

-- The proof problem: Prove that total cost equals $9.75
theorem JuanitaDessertCost : totalCost = 9.75 :=
by
  -- Proof is omitted
  sorry

end JuanitaDessertCost_l781_78147


namespace unripe_oranges_after_days_l781_78111

-- Definitions and Conditions
def sacks_per_day := 65
def days := 6

-- Statement to prove
theorem unripe_oranges_after_days : sacks_per_day * days = 390 := by
  sorry

end unripe_oranges_after_days_l781_78111


namespace target_hit_probability_l781_78134

theorem target_hit_probability (prob_A_hits : ℝ) (prob_B_hits : ℝ) (hA : prob_A_hits = 0.5) (hB : prob_B_hits = 0.6) :
  (1 - (1 - prob_A_hits) * (1 - prob_B_hits)) = 0.8 := 
by 
  sorry

end target_hit_probability_l781_78134


namespace total_gray_area_trees_l781_78180

/-- 
Three aerial photos were taken by the drone, each capturing the same number of trees.
First rectangle has 100 trees in total and 82 trees in the white area.
Second rectangle has 90 trees in total and 82 trees in the white area.
Prove that the number of trees in gray areas in both rectangles is 26.
-/
theorem total_gray_area_trees : (100 - 82) + (90 - 82) = 26 := 
by sorry

end total_gray_area_trees_l781_78180


namespace am_gm_inequality_l781_78194

theorem am_gm_inequality (a b c : ℝ) (h : a * b * c = 1 / 8) : 
  a^2 + b^2 + c^2 + a^2 * b^2 + b^2 * c^2 + c^2 * a^2 ≥ 15 / 16 :=
sorry

end am_gm_inequality_l781_78194


namespace blue_bead_probability_no_adjacent_l781_78149

theorem blue_bead_probability_no_adjacent :
  let total_beads := 9
  let blue_beads := 5
  let green_beads := 3
  let red_bead := 1
  let total_permutations := Nat.factorial total_beads / (Nat.factorial blue_beads * Nat.factorial green_beads * Nat.factorial red_bead)
  let valid_arrangements := (Nat.factorial 4) / (Nat.factorial 3 * Nat.factorial 1)
  let no_adjacent_valid := 4
  let probability_no_adj := (no_adjacent_valid : ℚ) / total_permutations
  probability_no_adj = (1 : ℚ) / 126 := 
by
  sorry

end blue_bead_probability_no_adjacent_l781_78149


namespace cone_base_circumference_l781_78132

theorem cone_base_circumference 
  (r : ℝ) 
  (θ : ℝ) 
  (h₁ : r = 5) 
  (h₂ : θ = 225) : 
  (θ / 360 * 2 * Real.pi * r) = (25 * Real.pi / 4) :=
by
  -- Proof skipped
  sorry

end cone_base_circumference_l781_78132


namespace delivery_parcels_problem_l781_78178

theorem delivery_parcels_problem (x : ℝ) (h1 : 2 + 2 * (1 + x) + 2 * (1 + x) ^ 2 = 7.28) : 
  2 + 2 * (1 + x) + 2 * (1 + x) ^ 2 = 7.28 :=
by
  exact h1

end delivery_parcels_problem_l781_78178


namespace ratio_of_price_l781_78192

-- Definitions from conditions
def original_price : ℝ := 3.00
def tom_pay_price : ℝ := 9.00

-- Theorem stating the ratio
theorem ratio_of_price : tom_pay_price / original_price = 3 := by
  sorry

end ratio_of_price_l781_78192


namespace infinitely_many_composite_values_l781_78129

theorem infinitely_many_composite_values (k m : ℕ) 
  (h_k : k ≥ 2) : 
  ∃ n : ℕ, n = 4 * k^4 ∧ ∀ m : ℕ, ∃ x y : ℕ, x > 1 ∧ y > 1 ∧ m^4 + n = x * y :=
by
  sorry

end infinitely_many_composite_values_l781_78129


namespace simplify_expression_l781_78145

theorem simplify_expression : 
  2 * (1 / (2 * 3) + 1 / (3 * 4) + 1 / (4 * 5) + 1 / (5 * 6) + 1 / (6 * 7) + 1 / (7 * 8)) = 3 / 4 :=
by
  sorry

end simplify_expression_l781_78145


namespace Nicky_time_before_catchup_l781_78105

-- Define the given speeds and head start time as constants
def v_C : ℕ := 5 -- Cristina's speed in meters per second
def v_N : ℕ := 3 -- Nicky's speed in meters per second
def t_H : ℕ := 12 -- Head start in seconds

-- Define the running time until catch up
def time_Nicky_run : ℕ := t_H + (36 / (v_C - v_N))

-- Prove that the time Nicky has run before Cristina catches up to him is 30 seconds
theorem Nicky_time_before_catchup : time_Nicky_run = 30 :=
by
  -- Add the steps for the proof
  sorry

end Nicky_time_before_catchup_l781_78105


namespace calculate_expression_l781_78198

theorem calculate_expression
  (x y : ℚ)
  (D E : ℚ × ℚ)
  (hx : x = (D.1 + E.1) / 2)
  (hy : y = (D.2 + E.2) / 2)
  (hD : D = (15, -3))
  (hE : E = (-4, 12)) :
  3 * x - 5 * y = -6 :=
by
  subst hD
  subst hE
  subst hx
  subst hy
  sorry

end calculate_expression_l781_78198


namespace sum_of_interior_angles_n_plus_3_l781_78172

-- Define the condition that the sum of the interior angles of a convex polygon with n sides is 1260 degrees
def sum_of_interior_angles (n : ℕ) : ℕ := 180 * (n - 2)

-- Prove that given the above condition for n, the sum of the interior angles of a convex polygon with n + 3 sides is 1800 degrees
theorem sum_of_interior_angles_n_plus_3 (n : ℕ) (h : sum_of_interior_angles n = 1260) : 
  sum_of_interior_angles (n + 3) = 1800 :=
by
  sorry

end sum_of_interior_angles_n_plus_3_l781_78172


namespace proportion_solution_l781_78122

theorem proportion_solution (x : ℝ) (h : 0.75 / x = 5 / 8) : x = 1.2 := 
by 
suffices h₀ : x = 6 / 5 by sorry
suffices h₁ : 6 / 5 = 1.2 by sorry
-- Proof steps go here
sorry

end proportion_solution_l781_78122


namespace find_least_q_l781_78169

theorem find_least_q : 
  ∃ q : ℕ, 
    (q ≡ 0 [MOD 7]) ∧ 
    (q ≥ 1000) ∧ 
    (q ≡ 1 [MOD 3]) ∧ 
    (q ≡ 1 [MOD 4]) ∧ 
    (q ≡ 1 [MOD 5]) ∧ 
    (q = 1141) :=
by
  sorry

end find_least_q_l781_78169


namespace max_planes_15_points_l781_78125

theorem max_planes_15_points (P : Finset (Fin 15)) (hP : ∀ (p1 p2 p3 : Fin 15), p1 ≠ p2 → p2 ≠ p3 → p1 ≠ p3) :
  P.card = 15 → (∃ planes : Finset (Finset (Fin 15)), planes.card = 455) := by
  sorry

end max_planes_15_points_l781_78125


namespace smallest_positive_period_of_f_range_of_f_in_interval_l781_78115

noncomputable def f (a : ℝ) (x : ℝ) := a * Real.sin x - Real.sqrt 3 * Real.cos x

theorem smallest_positive_period_of_f (a : ℝ) (h : f a (π / 3) = 0) :
  ∃ T : ℝ, T = 2 * π ∧ (∀ x, f a (x + T) = f a x) :=
sorry

theorem range_of_f_in_interval (a : ℝ) (h : f a (π / 3) = 0) :
  ∀ x ∈ Set.Icc (π / 2) (3 * π / 2), -1 ≤ f a x ∧ f a x ≤ 2 :=
sorry

end smallest_positive_period_of_f_range_of_f_in_interval_l781_78115


namespace percentage_selected_B_l781_78176

-- Definitions for the given conditions
def candidates := 7900
def selected_A := (6 / 100) * candidates
def selected_B := selected_A + 79

-- The question to be answered
def P_B := (selected_B / candidates) * 100

-- Proof statement
theorem percentage_selected_B : P_B = 7 := 
by
  -- Canonical statement placeholder 
  sorry

end percentage_selected_B_l781_78176


namespace empty_seats_correct_l781_78113

def children_count : ℕ := 52
def adult_count : ℕ := 29
def total_seats : ℕ := 95

theorem empty_seats_correct :
  total_seats - (children_count + adult_count) = 14 :=
by
  sorry

end empty_seats_correct_l781_78113


namespace initial_coins_l781_78168

-- Define the condition for the initial number of coins
variable (x : Nat) -- x represents the initial number of coins

-- The main statement theorem that needs proof
theorem initial_coins (h : x + 8 = 29) : x = 21 := 
by { sorry } -- placeholder for the proof

end initial_coins_l781_78168


namespace groceries_delivered_l781_78133

variables (S C P g T G : ℝ)
theorem groceries_delivered (hS : S = 14500) (hC : C = 14600) (hP : P = 1.5) (hg : g = 0.05) (hT : T = 40) :
  G = 800 :=
by {
  sorry
}

end groceries_delivered_l781_78133


namespace percentage_change_difference_l781_78179

-- Define the initial and final percentages of students
def initial_liked_percentage : ℝ := 0.4
def initial_disliked_percentage : ℝ := 0.6
def final_liked_percentage : ℝ := 0.8
def final_disliked_percentage : ℝ := 0.2

-- Define the problem statement
theorem percentage_change_difference :
  (final_liked_percentage - initial_liked_percentage) + 
  (initial_disliked_percentage - final_disliked_percentage) = 0.6 :=
sorry

end percentage_change_difference_l781_78179


namespace shelves_full_percentage_l781_78195

-- Define the conditions as constants
def ridges_per_record : Nat := 60
def cases : Nat := 4
def shelves_per_case : Nat := 3
def records_per_shelf : Nat := 20
def total_ridges : Nat := 8640

-- Define the total number of records
def total_records := total_ridges / ridges_per_record

-- Define the total capacity of the shelves
def total_capacity := cases * shelves_per_case * records_per_shelf

-- Define the percentage of shelves that are full
def percentage_full := (total_records * 100) / total_capacity

-- State the theorem that the percentage of the shelves that are full is 60%
theorem shelves_full_percentage : percentage_full = 60 := 
by
  sorry

end shelves_full_percentage_l781_78195


namespace carl_highway_miles_l781_78123

theorem carl_highway_miles
  (city_mpg : ℕ)
  (highway_mpg : ℕ)
  (city_miles : ℕ)
  (gas_cost_per_gallon : ℕ)
  (total_cost : ℕ)
  (h1 : city_mpg = 30)
  (h2 : highway_mpg = 40)
  (h3 : city_miles = 60)
  (h4 : gas_cost_per_gallon = 3)
  (h5 : total_cost = 42)
  : (total_cost - (city_miles / city_mpg) * gas_cost_per_gallon) / gas_cost_per_gallon * highway_mpg = 480 := 
by
  sorry

end carl_highway_miles_l781_78123


namespace problem_statement_l781_78154

noncomputable def x : ℝ := sorry -- Let x be a real number satisfying the condition

theorem problem_statement (x_real_cond : x + 1/x = 3) : 
  (x^12 - 7*x^8 + 2*x^4) = 44387*x - 15088 :=
sorry

end problem_statement_l781_78154


namespace probability_floor_sqrt_100x_eq_180_given_floor_sqrt_x_eq_18_l781_78189

open Real

noncomputable def probability_event : ℝ :=
  ((327.61 - 324) / (361 - 324))

theorem probability_floor_sqrt_100x_eq_180_given_floor_sqrt_x_eq_18 :
  probability_event = 361 / 3700 :=
by
  -- Conditions and calculations supplied in the problem
  sorry

end probability_floor_sqrt_100x_eq_180_given_floor_sqrt_x_eq_18_l781_78189


namespace sixth_term_geometric_mean_l781_78177

variable (a d : ℝ)

-- Define the arithmetic progression terms
def a_n (n : ℕ) := a + (n - 1) * d

-- Provided condition: second term is the geometric mean of the 1st and 4th terms
def condition (a d : ℝ) := a_n a d 2 = Real.sqrt (a_n a d 1 * a_n a d 4)

-- The goal to be proved: sixth term is the geometric mean of the 4th and 9th terms
theorem sixth_term_geometric_mean (a d : ℝ) (h : condition a d) : 
  a_n a d 6 = Real.sqrt (a_n a d 4 * a_n a d 9) :=
sorry

end sixth_term_geometric_mean_l781_78177


namespace find_a_l781_78199

theorem find_a :
  ∀ (a : ℝ), 
  (∀ x : ℝ, 2 * x^2 - 2016 * x + 2016^2 - 2016 * a - 1 = a^2) → 
  (∃ x1 x2 : ℝ, 2 * x1^2 - 2016 * x1 + 2016^2 - 2016 * a - 1 - a^2 = 0 ∧
                 2 * x2^2 - 2016 * x2 + 2016^2 - 2016 * a - 1 - a^2 = 0 ∧
                 x1 < a ∧ a < x2) → 
  2015 < a ∧ a < 2017 :=
by sorry

end find_a_l781_78199


namespace parabola_x_intercepts_count_l781_78153

theorem parabola_x_intercepts_count : 
  let equation := fun y : ℝ => -3 * y^2 + 2 * y + 3
  ∃! x : ℝ, ∃ y : ℝ, y = 0 ∧ x = equation y :=
by
  sorry

end parabola_x_intercepts_count_l781_78153


namespace range_of_m_l781_78190

theorem range_of_m (a b c : ℝ) (m : ℝ) (h1 : a > b) (h2 : b > c) (h3 : 1 / (a - b) + m / (b - c) ≥ 9 / (a - c)) :
  m ≥ 4 :=
sorry

end range_of_m_l781_78190


namespace deepak_present_age_l781_78163

-- Define the variables R and D
variables (R D : ℕ)

-- The conditions:
-- 1. After 4 years, Rahul's age will be 32 years.
-- 2. The ratio between Rahul and Deepak's ages is 4:3.
def rahul_age_after_4 : Prop := R + 4 = 32
def age_ratio : Prop := R / D = 4 / 3

-- The statement we want to prove:
theorem deepak_present_age (h1 : rahul_age_after_4 R) (h2 : age_ratio R D) : D = 21 :=
by sorry

end deepak_present_age_l781_78163


namespace coffee_tea_soda_l781_78110

theorem coffee_tea_soda (Pcoffee Ptea Psoda Pboth_no_soda : ℝ)
  (H1 : 0.9 = Pcoffee)
  (H2 : 0.8 = Ptea)
  (H3 : 0.7 = Psoda) :
  0.0 = Pboth_no_soda :=
  sorry

end coffee_tea_soda_l781_78110


namespace average_percentage_l781_78100

theorem average_percentage (s1 s2 : ℕ) (a1 a2 : ℕ) (n : ℕ)
  (h1 : s1 = 15) (h2 : a1 = 70) (h3 : s2 = 10) (h4 : a2 = 90) (h5 : n = 25)
  : ((s1 * a1 + s2 * a2) / n : ℕ) = 78 :=
by
  -- We include sorry to skip the proof part.
  sorry

end average_percentage_l781_78100


namespace coefficient_of_q_l781_78167

theorem coefficient_of_q (q' : ℤ → ℤ) (h : ∀ q, q' q = 3 * q - 3) (h₁ : q' (q' 4) = 72) : 
  ∀ q, q' q = 3 * q - 3 :=
  sorry

end coefficient_of_q_l781_78167


namespace product_of_odd_implies_sum_is_odd_l781_78119

theorem product_of_odd_implies_sum_is_odd (a b c : ℤ) (h : a * b * c % 2 = 1) : (a + b + c) % 2 = 1 :=
sorry

end product_of_odd_implies_sum_is_odd_l781_78119


namespace problem_l781_78131

theorem problem (p q : Prop) (m : ℝ):
  (p = (m > 1)) →
  (q = (-2 ≤ m ∧ m ≤ 2)) →
  (¬q = (m < -2 ∨ m > 2)) →
  (¬(p ∧ q)) →
  (p ∨ q) →
  (¬q) →
  m > 2 :=
by
  sorry

end problem_l781_78131


namespace unobserved_planet_exists_l781_78185

theorem unobserved_planet_exists
  (n : ℕ) (h_n_eq : n = 15)
  (planets : Fin n → Type)
  (dist : ∀ (i j : Fin n), ℝ)
  (h_distinct : ∀ (i j : Fin n), i ≠ j → dist i j ≠ dist j i)
  (nearest : ∀ i : Fin n, Fin n)
  (h_nearest : ∀ i : Fin n, nearest i ≠ i)
  : ∃ i : Fin n, ∀ j : Fin n, nearest j ≠ i := by
  sorry

end unobserved_planet_exists_l781_78185


namespace arithmetic_sequence_6000th_term_l781_78107

theorem arithmetic_sequence_6000th_term :
  ∀ (p r : ℕ), 
  (2 * p) = 2 * p → 
  (2 * p + 2 * r = 14) → 
  (14 + 2 * r = 4 * p - r) → 
  (2 * p + (6000 - 1) * 4 = 24006) :=
by 
  intros p r h h1 h2
  sorry

end arithmetic_sequence_6000th_term_l781_78107


namespace proof_problem_l781_78121

noncomputable def f (x : ℝ) : ℝ := Real.exp x

noncomputable def g (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := a * x + b

theorem proof_problem
  (a : ℝ) (b : ℝ) (x : ℝ)
  (h₀ : 0 ≤ a)
  (h₁ : a ≤ 1 / 2)
  (h₂ : b = 1)
  (h₃ : 0 ≤ x) :
  (1 / f x) + (x / g x a b) ≥ 1 := by
    sorry

end proof_problem_l781_78121


namespace marys_score_l781_78160

theorem marys_score (C ω S : ℕ) (H1 : S = 30 + 4 * C - ω) (H2 : S > 80)
  (H3 : (∀ C1 ω1 C2 ω2, (C1 ≠ C2 → 30 + 4 * C1 - ω1 ≠ 30 + 4 * C2 - ω2))) : 
  S = 119 :=
sorry

end marys_score_l781_78160


namespace bakery_total_items_l781_78196

theorem bakery_total_items (total_money : ℝ) (cupcake_cost : ℝ) (pastry_cost : ℝ) (max_cupcakes : ℕ) (remaining_money : ℝ) (total_items : ℕ) :
  total_money = 50 ∧ cupcake_cost = 3 ∧ pastry_cost = 2.5 ∧ max_cupcakes = 16 ∧ remaining_money = 2 ∧ total_items = max_cupcakes + 0 → total_items = 16 :=
by
  sorry

end bakery_total_items_l781_78196


namespace triangle_sides_inequality_triangle_sides_equality_condition_l781_78108

theorem triangle_sides_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (triangle_cond : a + b > c ∧ b + c > a ∧ c + a > b) :
  a^2 * b * (a - b) + b^2 * c * (b - c) + c^2 * a * (c - a) ≥ 0 := 
sorry

theorem triangle_sides_equality_condition (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (triangle_cond : a + b > c ∧ b + c > a ∧ c + a > b) :
  a^2 * b * (a - b) + b^2 * c * (b - c) + c^2 * a * (c - a) = 0 ↔ a = b ∧ b = c := 
sorry

end triangle_sides_inequality_triangle_sides_equality_condition_l781_78108


namespace find_x_l781_78140

theorem find_x (x : ℝ) (h : x^2 + 75 = (x - 20)^2) : x = 8.125 :=
by
  sorry

end find_x_l781_78140


namespace Alissa_presents_equal_9_l781_78127

def Ethan_presents : ℝ := 31.0
def difference : ℝ := 22.0
def Alissa_presents := Ethan_presents - difference

theorem Alissa_presents_equal_9 : Alissa_presents = 9.0 := 
by sorry

end Alissa_presents_equal_9_l781_78127


namespace sum_of_24_consecutive_integers_is_square_l781_78183

theorem sum_of_24_consecutive_integers_is_square : ∃ n : ℕ, ∃ k : ℕ, (n > 0) ∧ (24 * (2 * n + 23)) = k * k ∧ k * k = 324 :=
by
  sorry

end sum_of_24_consecutive_integers_is_square_l781_78183


namespace polynomial_evaluation_l781_78171

theorem polynomial_evaluation (y : ℝ) (hy : y^2 - 3 * y - 9 = 0) : y^3 - 3 * y^2 - 9 * y + 7 = 7 := 
  sorry

end polynomial_evaluation_l781_78171


namespace part1_part2_l781_78164

-- Define the quadratic equation and its discriminant
def quadratic_discriminant (a b c : ℝ) : ℝ :=
  b^2 - 4 * a * c

-- Define the conditions
def quadratic_equation (m : ℝ) : ℝ :=
  quadratic_discriminant 1 (-2) (-3 * m^2)

-- Part 1: Prove the quadratic equation always has two distinct real roots
theorem part1 (m : ℝ) : 
  quadratic_equation m > 0 :=
by
  sorry

-- Part 2: Find the value of m given the roots satisfy the equation α + 2β = 5
theorem part2 (α β m : ℝ) (h1 : α + β = 2) (h2 : α + 2 * β = 5) : 
  m = 1 ∨ m = -1 :=
by
  sorry


end part1_part2_l781_78164


namespace inequality_proof_l781_78118

theorem inequality_proof
  (n : ℕ) (hn : n ≥ 3) (x y z : ℝ) (hxyz_pos : x > 0 ∧ y > 0 ∧ z > 0)
  (hxyz_sum : x + y + z = 1) :
  (1 / x^(n-1) - x) * (1 / y^(n-1) - y) * (1 / z^(n-1) - z) ≥ ((3^n - 1) / 3)^3 :=
by sorry

end inequality_proof_l781_78118


namespace evan_books_l781_78193

theorem evan_books (B M : ℕ) (h1 : B = 200 - 40) (h2 : M * B + 60 = 860) : M = 5 :=
by {
  sorry  -- proof is omitted as per instructions
}

end evan_books_l781_78193


namespace cost_of_ingredients_l781_78158

theorem cost_of_ingredients :
  let popcorn_earnings := 50
  let cotton_candy_earnings := 3 * popcorn_earnings
  let total_earnings_per_day := popcorn_earnings + cotton_candy_earnings
  let total_earnings := total_earnings_per_day * 5
  let rent := 30
  let earnings_after_rent := total_earnings - rent
  earnings_after_rent - 895 = 75 :=
by
  let popcorn_earnings := 50
  let cotton_candy_earnings := 3 * popcorn_earnings
  let total_earnings_per_day := popcorn_earnings + cotton_candy_earnings
  let total_earnings := total_earnings_per_day * 5
  let rent := 30
  let earnings_after_rent := total_earnings - rent
  show earnings_after_rent - 895 = 75
  sorry

end cost_of_ingredients_l781_78158


namespace additional_pots_produced_l781_78137

theorem additional_pots_produced (first_hour_time_per_pot last_hour_time_per_pot : ℕ) :
  first_hour_time_per_pot = 6 →
  last_hour_time_per_pot = 5 →
  60 / last_hour_time_per_pot - 60 / first_hour_time_per_pot = 2 :=
by
  intros
  sorry

end additional_pots_produced_l781_78137


namespace min_value_expr_l781_78152

open Real

theorem min_value_expr (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  ∃ c, c = 4 * sqrt 3 - 6 ∧ ∀ (z w : ℝ), z = x ∧ w = y → (3 * z) / (3 * z + 2 * w) + w / (2 * z + w) ≥ c :=
by
  sorry

end min_value_expr_l781_78152


namespace bridge_length_is_219_l781_78166

noncomputable def length_of_bridge (train_length : ℕ) (train_speed_kmh : ℤ) (time_seconds : ℕ) : ℝ :=
  let train_speed_ms : ℝ := train_speed_kmh * (1000 / 3600)
  let total_distance : ℝ := train_speed_ms * time_seconds
  total_distance - train_length

theorem bridge_length_is_219 :
  length_of_bridge 156 45 30 = 219 :=
by
  sorry

end bridge_length_is_219_l781_78166


namespace find_q_l781_78106

theorem find_q (p q : ℕ) (hp_prime : Nat.Prime p) (hq_prime : Nat.Prime q) (hp_congr : 5 * p ≡ 3 [MOD 4]) (hq_def : q = 13 * p + 2) : q = 41 := 
sorry

end find_q_l781_78106


namespace max_value_f1_l781_78175

-- Definitions for the conditions
def f (x a b : ℝ) : ℝ := x^2 + a * b * x + a + 2 * b

-- Lean theorem statements
theorem max_value_f1 (a b : ℝ) (h : a + 2 * b = 4) :
  f 0 a b = 4 → f 1 a b ≤ 7 :=
sorry

end max_value_f1_l781_78175


namespace xyz_zero_if_equation_zero_l781_78173

theorem xyz_zero_if_equation_zero (x y z : ℚ) 
  (h : x^3 + 3 * y^3 + 9 * z^3 - 9 * x * y * z = 0) : 
  x = 0 ∧ y = 0 ∧ z = 0 := 
by 
  sorry

end xyz_zero_if_equation_zero_l781_78173


namespace perpendicular_lines_condition_l781_78174

theorem perpendicular_lines_condition (A1 B1 C1 A2 B2 C2 : ℝ) :
  (A1 * A2 + B1 * B2 = 0) ↔ (A1 * A2) / (B1 * B2) = -1 := sorry

end perpendicular_lines_condition_l781_78174


namespace no_solution_for_x_y_z_seven_n_plus_eight_is_perfect_square_l781_78181

theorem no_solution_for_x_y_z (a : ℕ) : 
  ¬ ∃ (x y z : ℚ), x^2 + y^2 + z^2 = 8 * a + 7 :=
by
  sorry

theorem seven_n_plus_eight_is_perfect_square (n : ℕ) :
  ∃ x : ℕ, 7^n + 8 = x^2 ↔ n = 0 :=
by
  sorry

end no_solution_for_x_y_z_seven_n_plus_eight_is_perfect_square_l781_78181


namespace percentage_of_students_who_speak_lies_l781_78151

theorem percentage_of_students_who_speak_lies
  (T : ℝ)    -- percentage of students who speak the truth
  (I : ℝ)    -- percentage of students who speak both truth and lies
  (U : ℝ)    -- probability of a randomly selected student speaking the truth or lies
  (H_T : T = 0.3)
  (H_I : I = 0.1)
  (H_U : U = 0.4) :
  ∃ (L : ℝ), L = 0.2 :=
by
  sorry

end percentage_of_students_who_speak_lies_l781_78151


namespace cost_of_whistle_l781_78159

theorem cost_of_whistle (cost_yoyo : ℕ) (total_spent : ℕ) (cost_yoyo_equals : cost_yoyo = 24) (total_spent_equals : total_spent = 38) : (total_spent - cost_yoyo) = 14 :=
by
  sorry

end cost_of_whistle_l781_78159


namespace desired_average_score_is_correct_l781_78155

-- Conditions
def average_score_9_tests : ℕ := 82
def score_10th_test : ℕ := 92

-- Desired average score
def desired_average_score : ℕ := 83

-- Total score for 10 tests
def total_score_10_tests (avg9 : ℕ) (score10 : ℕ) : ℕ :=
  9 * avg9 + score10

-- Main theorem statement to prove
theorem desired_average_score_is_correct :
  total_score_10_tests average_score_9_tests score_10th_test / 10 = desired_average_score :=
by
  sorry

end desired_average_score_is_correct_l781_78155


namespace jason_pears_count_l781_78144

theorem jason_pears_count 
  (initial_pears : ℕ)
  (given_to_keith : ℕ)
  (received_from_mike : ℕ)
  (final_pears : ℕ)
  (h_initial : initial_pears = 46)
  (h_given : given_to_keith = 47)
  (h_received : received_from_mike = 12)
  (h_final : final_pears = 12) :
  initial_pears - given_to_keith + received_from_mike = final_pears :=
sorry

end jason_pears_count_l781_78144


namespace conference_duration_excluding_breaks_l781_78141

-- Definitions based on the conditions
def total_hours : Nat := 14
def additional_minutes : Nat := 20
def break_minutes : Nat := 15

-- Total time including breaks
def total_time_minutes : Nat := total_hours * 60 + additional_minutes
-- Number of breaks
def number_of_breaks : Nat := total_hours
-- Total break time
def total_break_minutes : Nat := number_of_breaks * break_minutes

-- Proof statement
theorem conference_duration_excluding_breaks :
  total_time_minutes - total_break_minutes = 650 := by
  sorry

end conference_duration_excluding_breaks_l781_78141


namespace minimum_value_of_f_l781_78170

noncomputable def f (x m : ℝ) := (1 / 3) * x^3 - x + m

theorem minimum_value_of_f (m : ℝ) (h_max : f (-1) m = 1) : 
  f 1 m = -1 / 3 :=
by
  sorry

end minimum_value_of_f_l781_78170


namespace save_water_negate_l781_78101

/-- If saving 30cm^3 of water is denoted as +30cm^3, then wasting 10cm^3 of water is denoted as -10cm^3. -/
theorem save_water_negate :
  (∀ (save_waste : ℤ → ℤ), save_waste 30 = 30 → save_waste (-10) = -10) :=
by
  sorry

end save_water_negate_l781_78101
