import Mathlib

namespace profit_without_discount_l1551_155174

theorem profit_without_discount (CP SP MP : ℝ) (discountRate profitRate : ℝ)
  (h1 : CP = 100)
  (h2 : discountRate = 0.05)
  (h3 : profitRate = 0.235)
  (h4 : SP = CP * (1 + profitRate))
  (h5 : MP = SP / (1 - discountRate)) :
  (((MP - CP) / CP) * 100) = 30 := 
sorry

end profit_without_discount_l1551_155174


namespace factor_diff_of_squares_l1551_155170

theorem factor_diff_of_squares (y : ℝ) : 25 - 16 * y^2 = (5 - 4 * y) * (5 + 4 * y) := 
sorry

end factor_diff_of_squares_l1551_155170


namespace people_going_to_zoo_l1551_155196

theorem people_going_to_zoo (buses people_per_bus total_people : ℕ) 
  (h1 : buses = 3) 
  (h2 : people_per_bus = 73) 
  (h3 : total_people = buses * people_per_bus) : 
  total_people = 219 := by
  rw [h1, h2] at h3
  exact h3

end people_going_to_zoo_l1551_155196


namespace twice_a_plus_one_non_negative_l1551_155160

theorem twice_a_plus_one_non_negative (a : ℝ) : 2 * a + 1 ≥ 0 :=
sorry

end twice_a_plus_one_non_negative_l1551_155160


namespace sum_converges_to_one_l1551_155122

noncomputable def series_sum (n: ℕ) : ℝ :=
  if n ≥ 2 then (6 * n^3 - 2 * n^2 - 2 * n + 1) / (n^6 - 2 * n^5 + 2 * n^4 - n^3 + n^2 - 2 * n)
  else 0

theorem sum_converges_to_one : 
  (∑' n, series_sum n) = 1 := by
  sorry

end sum_converges_to_one_l1551_155122


namespace B_share_in_profit_l1551_155141

theorem B_share_in_profit (A B C : ℝ) (total_profit : ℝ) 
    (h1 : A = 3 * B)
    (h2 : B = (2/3) * C)
    (h3 : total_profit = 6600) :
    (B / (A + B + C)) * total_profit = 1200 := 
by
  sorry

end B_share_in_profit_l1551_155141


namespace pagoda_lanterns_l1551_155102

-- Definitions
def top_layer_lanterns (a₁ : ℕ) : ℕ := a₁
def bottom_layer_lanterns (a₁ : ℕ) : ℕ := a₁ * 2^6
def sum_of_lanterns (a₁ : ℕ) : ℕ := (a₁ * (1 - 2^7)) / (1 - 2)
def total_lanterns : ℕ := 381
def layers : ℕ := 7
def common_ratio : ℕ := 2

-- Problem Statement
theorem pagoda_lanterns (a₁ : ℕ) (h : sum_of_lanterns a₁ = total_lanterns) : 
  top_layer_lanterns a₁ + bottom_layer_lanterns a₁ = 195 := sorry

end pagoda_lanterns_l1551_155102


namespace smallest_positive_integer_k_l1551_155133

theorem smallest_positive_integer_k:
  ∀ T : ℕ, ∀ n : ℕ, (T = n * (n + 1) / 2) → ∃ m : ℕ, 81 * T + 10 = m * (m + 1) / 2 :=
by
  intro T n h
  sorry

end smallest_positive_integer_k_l1551_155133


namespace math_problem_l1551_155178

theorem math_problem :
  (-1)^2024 + (-10) / (1/2) * 2 + (2 - (-3)^3) = -10 := by
  sorry

end math_problem_l1551_155178


namespace samuel_apples_left_l1551_155131

def bonnieApples : ℕ := 8
def extraApples : ℕ := 20
def samuelTotalApples : ℕ := bonnieApples + extraApples
def samuelAte : ℕ := samuelTotalApples / 2
def samuelRemainingAfterEating : ℕ := samuelTotalApples - samuelAte
def samuelUsedForPie : ℕ := samuelRemainingAfterEating / 7
def samuelFinalRemaining : ℕ := samuelRemainingAfterEating - samuelUsedForPie

theorem samuel_apples_left :
  samuelFinalRemaining = 12 := by
  sorry

end samuel_apples_left_l1551_155131


namespace find_ABC_plus_DE_l1551_155193

theorem find_ABC_plus_DE (ABCDE : Nat) (h1 : ABCDE = 13579 * 6) : (ABCDE / 1000 + ABCDE % 1000 % 100) = 888 :=
by
  sorry

end find_ABC_plus_DE_l1551_155193


namespace false_propositions_l1551_155161

open Classical

theorem false_propositions :
  ¬ (∀ x : ℝ, x^2 + 3 < 0) ∧ ¬ (∀ x : ℕ, x^2 > 1) ∧ (∃ x : ℤ, x^5 < 1) ∧ ¬ (∃ x : ℚ, x^2 = 3) :=
by
  sorry

end false_propositions_l1551_155161


namespace neg_p_is_correct_l1551_155104

def is_positive_integer (x : ℕ) : Prop := x > 0

def proposition_p (x : ℕ) : Prop := (1 / 2 : ℝ) ^ x ≤ 1 / 2

def negation_of_p : Prop := ∃ x : ℕ, is_positive_integer x ∧ ¬ proposition_p x

theorem neg_p_is_correct : negation_of_p :=
sorry

end neg_p_is_correct_l1551_155104


namespace dennis_total_cost_l1551_155117

-- Define the cost of items and quantities
def cost_pants : ℝ := 110.0
def cost_socks : ℝ := 60.0
def quantity_pants : ℝ := 4
def quantity_socks : ℝ := 2
def discount_rate : ℝ := 0.30

-- Define the total costs before and after discount
def total_cost_pants_before_discount : ℝ := cost_pants * quantity_pants
def total_cost_socks_before_discount : ℝ := cost_socks * quantity_socks
def total_cost_before_discount : ℝ := total_cost_pants_before_discount + total_cost_socks_before_discount
def total_discount : ℝ := total_cost_before_discount * discount_rate
def total_cost_after_discount : ℝ := total_cost_before_discount - total_discount

-- Theorem asserting the total amount after discount
theorem dennis_total_cost : total_cost_after_discount = 392 := by 
  sorry

end dennis_total_cost_l1551_155117


namespace no_right_triangle_l1551_155186

theorem no_right_triangle (a b c : ℝ) (h₁ : a = Real.sqrt 3) (h₂ : b = 2) (h₃ : c = Real.sqrt 5) : 
  a^2 + b^2 ≠ c^2 :=
by
  sorry

end no_right_triangle_l1551_155186


namespace ratio_current_to_past_l1551_155184

-- Conditions
def current_posters : ℕ := 22
def posters_after_summer (p : ℕ) : ℕ := p + 6
def posters_two_years_ago : ℕ := 14

-- Proof problem statement
theorem ratio_current_to_past (h₁ : current_posters = 22) (h₂ : posters_two_years_ago = 14) : 
  (current_posters / Nat.gcd current_posters posters_two_years_ago) = 11 ∧ 
  (posters_two_years_ago / Nat.gcd current_posters posters_two_years_ago) = 7 :=
by
  sorry

end ratio_current_to_past_l1551_155184


namespace quadratic_root_expression_value_l1551_155165

theorem quadratic_root_expression_value (a : ℝ) 
  (h : a^2 - 2 * a - 3 = 0) : 2 * a^2 - 4 * a + 1 = 7 :=
by
  sorry

end quadratic_root_expression_value_l1551_155165


namespace value_of_a_l1551_155185

theorem value_of_a (x a : ℤ) (h : x = 3 ∧ x^2 = a) : a = 9 :=
sorry

end value_of_a_l1551_155185


namespace average_study_diff_l1551_155107

theorem average_study_diff (diff : List ℤ) (h_diff : diff = [15, -5, 25, -10, 5, 20, -15]) :
  (List.sum diff) / (List.length diff) = 5 := by
  sorry

end average_study_diff_l1551_155107


namespace boys_difference_twice_girls_l1551_155163

theorem boys_difference_twice_girls :
  ∀ (total_students girls boys : ℕ),
  total_students = 68 →
  girls = 28 →
  boys = total_students - girls →
  2 * girls - boys = 16 :=
by
  intros total_students girls boys h1 h2 h3
  sorry

end boys_difference_twice_girls_l1551_155163


namespace maximize_revenue_l1551_155105

theorem maximize_revenue (p : ℝ) (hp : p ≤ 30) :
  (p = 12 ∨ p = 13) → (∀ p : ℤ, p ≤ 30 → 200 * p - 8 * p * p ≤ 1248) :=
by
  intros h1 h2
  sorry

end maximize_revenue_l1551_155105


namespace roots_cubic_reciprocal_sum_l1551_155155

theorem roots_cubic_reciprocal_sum (a b c : ℝ) 
(h₁ : a + b + c = 12) (h₂ : a * b + b * c + c * a = 27) (h₃ : a * b * c = 18) :
  1 / a^3 + 1 / b^3 + 1 / c^3 = 13 / 24 :=
by
  sorry

end roots_cubic_reciprocal_sum_l1551_155155


namespace pastries_eaten_l1551_155143

theorem pastries_eaten (total_p: ℕ)
  (hare_fraction: ℚ)
  (dormouse_fraction: ℚ)
  (hare_eaten: ℕ)
  (remaining_after_hare: ℕ)
  (dormouse_eaten: ℕ)
  (final_remaining: ℕ) 
  (hatter_with_left: ℕ) :
  (final_remaining = hatter_with_left) -> hare_fraction = 5 / 16 -> dormouse_fraction = 7 / 11 -> hatter_with_left = 8 -> total_p = 32 -> 
  (total_p = hare_eaten + remaining_after_hare) -> (remaining_after_hare - dormouse_eaten = hatter_with_left) -> (hare_eaten = 10) ∧ (dormouse_eaten = 14) := 
by {
  sorry
}

end pastries_eaten_l1551_155143


namespace area_of_region_W_l1551_155159

structure Rhombus (P Q R T : Type) :=
  (side_length : ℝ)
  (angle_Q : ℝ)

def Region_W
  (P Q R T : Type)
  (r : Rhombus P Q R T)
  (h_side : r.side_length = 5)
  (h_angle : r.angle_Q = 90) : ℝ :=
6.25

theorem area_of_region_W
  {P Q R T : Type}
  (r : Rhombus P Q R T)
  (h_side : r.side_length = 5)
  (h_angle : r.angle_Q = 90) :
  Region_W P Q R T r h_side h_angle = 6.25 :=
sorry

end area_of_region_W_l1551_155159


namespace find_number_l1551_155172

theorem find_number (x : ℝ) (h : 1345 - x / 20.04 = 1295) : x = 1002 :=
sorry

end find_number_l1551_155172


namespace smallest_a_for_quadratic_poly_l1551_155100

theorem smallest_a_for_quadratic_poly (a : ℕ) (a_pos : 0 < a) :
  (∃ b c : ℤ, ∀ x : ℝ, 0 < x ∧ x < 1 → a*x^2 + b*x + c = 0 → (2 : ℝ)^2 - (4 : ℝ)*(a * c) < 0 ∧ b^2 - 4*a*c ≥ 1) → a ≥ 5 := 
sorry

end smallest_a_for_quadratic_poly_l1551_155100


namespace patrick_savings_ratio_l1551_155182

theorem patrick_savings_ratio (S : ℕ) (bike_cost : ℕ) (lent_amt : ℕ) (remaining_amt : ℕ)
  (h1 : bike_cost = 150)
  (h2 : lent_amt = 50)
  (h3 : remaining_amt = 25)
  (h4 : S = remaining_amt + lent_amt) :
  (S / bike_cost : ℚ) = 1 / 2 := 
sorry

end patrick_savings_ratio_l1551_155182


namespace min_value_f_l1551_155108

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := a * x ^ 3 + b * Real.arcsin x + 3

theorem min_value_f (a b : ℝ) (hab : a ≠ 0 ∧ b ≠ 0) (hmax : ∃ x, f a b x = 10) : ∃ y, f a b y = -4 := by
  sorry

end min_value_f_l1551_155108


namespace phone_number_C_value_l1551_155166

/-- 
In a phone number formatted as ABC-DEF-GHIJ, each letter symbolizes a distinct digit.
Digits in each section ABC, DEF, and GHIJ are in ascending order i.e., A < B < C, D < E < F, and G < H < I < J.
Moreover, D, E, F are consecutive odd digits, and G, H, I, J are consecutive even digits.
Also, A + B + C = 15. Prove that the value of C is 9. 
-/
theorem phone_number_C_value :
  ∃ (A B C D E F G H I J : ℕ), 
  A < B ∧ B < C ∧ D < E ∧ E < F ∧ G < H ∧ H < I ∧ I < J ∧
  (D % 2 = 1) ∧ (E % 2 = 1) ∧ (F % 2 = 1) ∧
  (G % 2 = 0) ∧ (H % 2 = 0) ∧ (I % 2 = 0) ∧ (J % 2 = 0) ∧
  (E = D + 2) ∧ (F = D + 4) ∧ (H = G + 2) ∧ (I = G + 4) ∧ (J = G + 6) ∧
  A + B + C = 15 ∧
  C = 9 := by 
  sorry

end phone_number_C_value_l1551_155166


namespace solve_for_ab_l1551_155146

theorem solve_for_ab (a b : ℤ) 
  (h1 : a + 3 * b = 27) 
  (h2 : 5 * a + 4 * b = 47) : 
  a + b = 11 :=
sorry

end solve_for_ab_l1551_155146


namespace inverse_function_value_l1551_155177

def f (x : ℝ) : ℝ := 2 * x ^ 3 - 3

theorem inverse_function_value :
  f 3 = 51 :=
by
  sorry

end inverse_function_value_l1551_155177


namespace complement_intersection_l1551_155149

def setM : Set ℝ := { x | 2 / x < 1 }
def setN : Set ℝ := { y | ∃ x, y = Real.sqrt (x - 1) }

theorem complement_intersection 
  (R : Set ℝ) : ((R \ setM) ∩ setN = { y | 0 ≤ y ∧ y ≤ 2 }) :=
  sorry

end complement_intersection_l1551_155149


namespace op_two_four_l1551_155115

def op (a b : ℝ) : ℝ := 5 * a + 2 * b

theorem op_two_four : op 2 4 = 18 := by
  sorry

end op_two_four_l1551_155115


namespace zucchini_pounds_l1551_155151

theorem zucchini_pounds :
  let eggplants_pounds := 5
  let eggplants_cost_per_pound := 2.00
  let tomatoes_pounds := 4
  let tomatoes_cost_per_pound := 3.50
  let onions_pounds := 3
  let onions_cost_per_pound := 1.00
  let basil_pounds := 1
  let basil_cost_per_half_pound := 2.50
  let quarts := 4
  let cost_per_quart := 10.00
  let total_cost := quarts * cost_per_quart
  let cost_of_eggplants := eggplants_pounds * eggplants_cost_per_pound
  let cost_of_tomatoes := tomatoes_pounds * tomatoes_cost_per_pound
  let cost_of_onions := onions_pounds * onions_cost_per_pound
  let cost_of_basil := basil_pounds * (basil_cost_per_half_pound * 2)
  let other_ingredients_cost := cost_of_eggplants + cost_of_tomatoes + cost_of_onions + cost_of_basil
  let cost_of_zucchini := total_cost - other_ingredients_cost
  let zucchini_cost_per_pound := 2.00
  let pounds_of_zucchini := cost_of_zucchini / zucchini_cost_per_pound
  pounds_of_zucchini = 4 :=
by
  sorry

end zucchini_pounds_l1551_155151


namespace strange_number_l1551_155123

theorem strange_number (x : ℤ) (h : (x - 7) * 7 = (x - 11) * 11) : x = 18 :=
sorry

end strange_number_l1551_155123


namespace data_division_into_groups_l1551_155194

-- Conditions
def data_set_size : Nat := 90
def max_value : Nat := 141
def min_value : Nat := 40
def class_width : Nat := 10

-- Proof statement
theorem data_division_into_groups : (max_value - min_value) / class_width + 1 = 11 :=
by
  sorry

end data_division_into_groups_l1551_155194


namespace find_eighth_number_l1551_155118

-- Define the given problem with the conditions
noncomputable def sum_of_sixteen_numbers := 16 * 55
noncomputable def sum_of_first_eight_numbers := 8 * 60
noncomputable def sum_of_last_eight_numbers := 8 * 45
noncomputable def sum_of_last_nine_numbers := 9 * 50
noncomputable def sum_of_first_ten_numbers := 10 * 62

-- Define what we want to prove
theorem find_eighth_number :
  (exists (x : ℕ), x = 90) →
  sum_of_first_eight_numbers = 480 →
  sum_of_last_eight_numbers = 360 →
  sum_of_last_nine_numbers = 450 →
  sum_of_first_ten_numbers = 620 →
  sum_of_sixteen_numbers = 880 →
  x = 90 :=
by sorry

end find_eighth_number_l1551_155118


namespace samantha_hike_distance_l1551_155188

theorem samantha_hike_distance :
  let A : ℝ × ℝ := (0, 0)  -- Samantha's starting point
  let B := (0, 3)           -- Point after walking northward 3 miles
  let C := (5 / (2 : ℝ) * Real.sqrt 2, 3) -- Point after walking 5 miles at 45 degrees eastward
  (dist A C = Real.sqrt 86 / 2) :=
by
  let A : ℝ × ℝ := (0, 0)
  let B : ℝ × ℝ := (0, 3)
  let C : ℝ × ℝ := (5 / (2 : ℝ) * Real.sqrt 2, 3)
  show dist A C = Real.sqrt 86 / 2
  sorry

end samantha_hike_distance_l1551_155188


namespace lcm_inequality_l1551_155125

theorem lcm_inequality
  (a b c d e : ℤ)
  (h1 : 1 ≤ a)
  (h2 : a < b)
  (h3 : b < c)
  (h4 : c < d)
  (h5 : d < e) :
  (1 : ℚ) / Int.lcm a b + (1 : ℚ) / Int.lcm b c + 
  (1 : ℚ) / Int.lcm c d + (1 : ℚ) / Int.lcm d e ≤ (15 : ℚ) / 16 := by
  sorry

end lcm_inequality_l1551_155125


namespace number_of_distinct_linear_recurrences_l1551_155147

open BigOperators

/-
  Let p be a prime positive integer.
  Define a mod-p recurrence of degree n to be a sequence {a_k}_{k >= 0} of numbers modulo p 
  satisfying a relation of the form:

  ai+n = c_n-1 ai+n-1 + ... + c_1 ai+1 + c_0 ai
  for all i >= 0, where c_0, c_1, ..., c_n-1 are integers and c_0 not equivalent to 0 mod p.
  Compute the number of distinct linear recurrences of degree at most n in terms of p and n.
-/
theorem number_of_distinct_linear_recurrences (p n : ℕ) (hp : Nat.Prime p) : 
  ∃ d : ℕ, 
    (∀ {a : ℕ → ℕ} {c : ℕ → ℕ} (h : ∀ i, a (i + n) = ∑ j in Finset.range n, c j * a (i + j))
     (hc0 : c 0 ≠ 0), 
      d = (1 - n * (p - 1) / (p + 1) + p^2 * (p^(2 * n) - 1) / (p + 1)^2 : ℚ)) :=
  sorry

end number_of_distinct_linear_recurrences_l1551_155147


namespace ratio_A_B_correct_l1551_155158

-- Define the shares of A, B, and C
def A_share := 372
def B_share := 93
def C_share := 62

-- Total amount distributed
def total_share := A_share + B_share + C_share

-- The ratio of A's share to B's share
def ratio_A_to_B := A_share / B_share

theorem ratio_A_B_correct : 
  total_share = 527 ∧ 
  ¬(B_share = (1 / 4) * C_share) ∧ 
  ratio_A_to_B = 4 := 
by
  sorry

end ratio_A_B_correct_l1551_155158


namespace circle_center_coordinates_l1551_155119

theorem circle_center_coordinates (b c p q : ℝ) 
    (h_circle_eq : ∀ x y : ℝ, x^2 + y^2 - 2 * p * x - 2 * q * y + 2 * q - 1 = 0) 
    (h_quad_roots : ∀ x : ℝ, x^2 + b * x + c = 0) 
    (h_condition : b^2 - 4 * c ≥ 0) : 
    (p = -b / 2) ∧ (q = (1 + c) / 2) := 
sorry

end circle_center_coordinates_l1551_155119


namespace deposit_paid_l1551_155124

variable (P : ℝ) (Deposit Remaining : ℝ)

-- Define the conditions
def deposit_condition : Prop := Deposit = 0.10 * P
def remaining_condition : Prop := Remaining = 0.90 * P
def remaining_amount_given : Prop := Remaining = 1170

-- The goal to prove: the deposit paid is $130
theorem deposit_paid (h₁ : deposit_condition P Deposit) (h₂ : remaining_condition P Remaining) (h₃ : remaining_amount_given Remaining) : 
  Deposit = 130 :=
  sorry

end deposit_paid_l1551_155124


namespace area_of_hexagon_correct_l1551_155136

variable (α β γ : ℝ) (S : ℝ) (r R : ℝ)
variable (AB BC AC : ℝ)
variable (A' B' C' : ℝ)

noncomputable def area_of_hexagon (AB BC AC : ℝ) (R : ℝ) (S : ℝ) (r : ℝ) : ℝ :=
  2 * (S / (r * r))

theorem area_of_hexagon_correct
  (hAB : AB = 13) (hBC : BC = 14) (hAC : AC = 15)
  (hR : R = 65 / 8) (hS : S = 1344 / 65) :
  area_of_hexagon AB BC AC R S r = 2 * (S / (r * r)) :=
sorry

end area_of_hexagon_correct_l1551_155136


namespace midpoint_uniqueness_l1551_155199

-- Define a finite set of points in the plane
axiom S : Finset (ℝ × ℝ)

-- Define what it means for P to be the midpoint of a segment
def is_midpoint (P A A' : ℝ × ℝ) : Prop :=
  P.1 = (A.1 + A'.1) / 2 ∧ P.2 = (A.2 + A'.2) / 2

-- Statement of the problem
theorem midpoint_uniqueness (P Q : ℝ × ℝ) :
  (∀ A ∈ S, ∃ A' ∈ S, is_midpoint P A A') →
  (∀ A ∈ S, ∃ A' ∈ S, is_midpoint Q A A') →
  P = Q :=
sorry

end midpoint_uniqueness_l1551_155199


namespace find_y_l1551_155148

theorem find_y (x y : ℤ) (h1 : x^2 - 2*x + 5 = y + 3) (h2 : x = -3) : y = 17 := by
  sorry

end find_y_l1551_155148


namespace value_does_not_appear_l1551_155106

theorem value_does_not_appear : 
  let f : ℕ → ℕ := fun x => 5*x^5 + 4*x^4 + 3*x^3 + 2*x^2 + x + 1
  let x := 2
  let values := [14, 31, 64, 129, 259]
  127 ∉ values :=
by
  sorry

end value_does_not_appear_l1551_155106


namespace stewart_farm_horseFood_l1551_155168

variable (sheep horses horseFoodPerHorse : ℕ)
variable (ratio_sh_to_hs : ℕ × ℕ)
variable (totalHorseFood : ℕ)

noncomputable def horse_food_per_day (sheep : ℕ) (ratio_sh_to_hs : ℕ × ℕ) (totalHorseFood : ℕ) : ℕ :=
  let horses := (sheep * ratio_sh_to_hs.2) / ratio_sh_to_hs.1
  totalHorseFood / horses

theorem stewart_farm_horseFood (h_ratio : ratio_sh_to_hs = (4, 7))
                                (h_sheep : sheep = 32)
                                (h_total : totalHorseFood = 12880) :
    horse_food_per_day sheep ratio_sh_to_hs totalHorseFood = 230 := by
  sorry

end stewart_farm_horseFood_l1551_155168


namespace work_rate_D_time_A_B_D_time_D_l1551_155162

def workRate (person : String) : ℚ :=
  if person = "A" then 1/12 else
  if person = "B" then 1/6 else
  if person = "A_D" then 1/4 else
  0

theorem work_rate_D : workRate "A_D" - workRate "A" = 1/6 := by
  sorry

theorem time_A_B_D : (1 / (workRate "A" + workRate "B" + (workRate "A_D" - workRate "A"))) = 2.4 := by
  sorry
  
theorem time_D : (1 / (workRate "A_D" - workRate "A")) = 6 := by
  sorry

end work_rate_D_time_A_B_D_time_D_l1551_155162


namespace price_reduction_l1551_155121

theorem price_reduction (x : ℝ) : 
  188 * (1 - x) ^ 2 = 108 :=
sorry

end price_reduction_l1551_155121


namespace slices_per_friend_l1551_155144

theorem slices_per_friend (total_slices friends : ℕ) (h1 : total_slices = 16) (h2 : friends = 4) : (total_slices / friends) = 4 :=
by
  sorry

end slices_per_friend_l1551_155144


namespace at_least_one_not_less_than_two_l1551_155179

theorem at_least_one_not_less_than_two (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a + 1 / b) ≥ 2 ∨ (b + 1 / c) ≥ 2 ∨ (c + 1 / a) ≥ 2 :=
sorry

end at_least_one_not_less_than_two_l1551_155179


namespace gcd_g102_g103_eq_one_l1551_155101

def g (x : ℤ) : ℤ := x^2 - 2*x + 2023

theorem gcd_g102_g103_eq_one : Nat.gcd (g 102).natAbs (g 103).natAbs = 1 := by
  sorry

end gcd_g102_g103_eq_one_l1551_155101


namespace smallest_AAB_l1551_155153

theorem smallest_AAB (A B : ℕ) (hA : 1 ≤ A ∧ A ≤ 9) (hB : 1 ≤ B ∧ B ≤ 9) 
  (h : 10 * A + B = (110 * A + B) / 7) : 110 * A + B = 996 :=
by
  sorry

end smallest_AAB_l1551_155153


namespace choose_bar_length_l1551_155113

theorem choose_bar_length (x : ℝ) (h1 : 1 < x) (h2 : x < 4) : x = 3 :=
by
  sorry

end choose_bar_length_l1551_155113


namespace apples_to_mangos_equivalent_l1551_155192

-- Definitions and conditions
def apples_worth_mangos (a b : ℝ) : Prop := (5 / 4) * 16 * a = 10 * b

-- Theorem statement
theorem apples_to_mangos_equivalent : 
  ∀ (a b : ℝ), apples_worth_mangos a b → (3 / 4) * 12 * a = 4.5 * b :=
by
  intro a b
  intro h
  sorry

end apples_to_mangos_equivalent_l1551_155192


namespace total_amount_sold_l1551_155137

theorem total_amount_sold (metres_sold : ℕ) (loss_per_metre cost_price_per_metre : ℕ) 
  (h1 : metres_sold = 600) (h2 : loss_per_metre = 5) (h3 : cost_price_per_metre = 35) :
  (cost_price_per_metre - loss_per_metre) * metres_sold = 18000 :=
by
  sorry

end total_amount_sold_l1551_155137


namespace find_constant_c_and_t_l1551_155183

noncomputable def exists_constant_c_and_t (c : ℝ) (t : ℝ) : Prop :=
∀ (x1 x2 m : ℝ), (x1^2 - m * x1 - c = 0) ∧ (x2^2 - m * x2 - c = 0) →
  (t = 1 / ((1 + m^2) * x1^2) + 1 / ((1 + m^2) * x2^2))

theorem find_constant_c_and_t : ∃ (c t : ℝ), exists_constant_c_and_t c t ∧ c = 2 ∧ t = 3 / 2 :=
sorry

end find_constant_c_and_t_l1551_155183


namespace fido_yard_area_reach_l1551_155181

theorem fido_yard_area_reach (s r : ℝ) (h1 : r = s / (2 * Real.sqrt 2)) (h2 : ∃ (a b : ℕ), (Real.pi * Real.sqrt a) / b = Real.pi * (r ^ 2) / (2 * s^2 * Real.sqrt 2) ) :
  ∃ (a b : ℕ), a * b = 64 :=
by
  sorry

end fido_yard_area_reach_l1551_155181


namespace point_D_number_l1551_155116

theorem point_D_number (x : ℝ) :
    (5 + 8 - 10 + x = -5 - 8 + 10 - x) ↔ x = -3 :=
by
  sorry

end point_D_number_l1551_155116


namespace team_selection_l1551_155112

theorem team_selection :
  let teachers := 5
  let students := 10
  (teachers * students = 50) :=
by
  sorry

end team_selection_l1551_155112


namespace temperature_at_tian_du_peak_height_of_mountain_peak_l1551_155128

-- Problem 1: Temperature at the top of Tian Du Peak
theorem temperature_at_tian_du_peak
  (height : ℝ) (drop_rate : ℝ) (initial_temp : ℝ)
  (H : height = 1800) (D : drop_rate = 0.6) (I : initial_temp = 18) :
  (initial_temp - (height / 100 * drop_rate)) = 7.2 :=
by
  sorry

-- Problem 2: Height of the mountain peak
theorem height_of_mountain_peak
  (drop_rate : ℝ) (foot_temp top_temp : ℝ)
  (D : drop_rate = 0.6) (F : foot_temp = 10) (T : top_temp = -8) :
  (foot_temp - top_temp) / drop_rate * 100 = 3000 :=
by
  sorry

end temperature_at_tian_du_peak_height_of_mountain_peak_l1551_155128


namespace math_problem_l1551_155180

noncomputable def proof_problem (n : ℝ) (A B : ℝ) : Prop :=
  A = n^2 ∧ B = n^2 + 1 ∧ (1 * n^4 + 2 * n^2 + 3 + 2 * (n^2 + 1) + 1 = 5 * (2 * n^2 + 1)) → 
  A + B = 7 + 4 * Real.sqrt 2

theorem math_problem (n : ℝ) (A B : ℝ) :
  proof_problem n A B :=
sorry

end math_problem_l1551_155180


namespace is_periodic_l1551_155138

noncomputable def f : ℝ → ℝ := sorry

axiom domain (x : ℝ) : true
axiom not_eq_neg1_and_not_eq_0 (x : ℝ) : f x ≠ -1 ∧ f x ≠ 0
axiom functional_eq (x y : ℝ) : f (x - y) = - (f x / (1 + f y))

theorem is_periodic : ∃ p, p > 0 ∧ ∀ x, f (x + p) = f x :=
sorry

end is_periodic_l1551_155138


namespace suresh_work_hours_l1551_155157

variable (x : ℕ) -- Number of hours Suresh worked

theorem suresh_work_hours :
  (1/15 : ℝ) * x + (4 * (1/10 : ℝ)) = 1 -> x = 9 :=
by
  sorry

end suresh_work_hours_l1551_155157


namespace units_digit_of_a_l1551_155111

theorem units_digit_of_a :
  (2003^2004 - 2004^2003) % 10 = 7 :=
by
  sorry

end units_digit_of_a_l1551_155111


namespace percentage_increase_l1551_155114

variable {x y : ℝ}
variable {P : ℝ} -- percentage

theorem percentage_increase (h1 : y = x * (1 + P / 100)) (h2 : x = y * 0.5882352941176471) : P = 70 := 
by
  sorry

end percentage_increase_l1551_155114


namespace problem_solved_prob_l1551_155152

theorem problem_solved_prob (pA pB : ℝ) (HA : pA = 1 / 3) (HB : pB = 4 / 5) :
  ((1 - (1 - pA) * (1 - pB)) = 13 / 15) :=
by
  sorry

end problem_solved_prob_l1551_155152


namespace handshaking_pairs_l1551_155132

-- Definition of the problem: Given 8 people, pair them up uniquely and count the ways modulo 1000
theorem handshaking_pairs (N : ℕ) (H : N=105) : (N % 1000) = 105 :=
by {
  -- The proof is omitted.
  sorry
}

end handshaking_pairs_l1551_155132


namespace probability_not_orange_not_white_l1551_155120

theorem probability_not_orange_not_white (num_orange num_black num_white : ℕ)
    (h_orange : num_orange = 8) (h_black : num_black = 7) (h_white : num_white = 6) :
    (num_black : ℚ) / (num_orange + num_black + num_white : ℚ) = 1 / 3 :=
  by
    -- Solution will be here.
    sorry

end probability_not_orange_not_white_l1551_155120


namespace bruce_bhishma_meet_again_l1551_155103

theorem bruce_bhishma_meet_again (L S_B S_H : ℕ) (hL : L = 600) (hSB : S_B = 30) (hSH : S_H = 20) : 
  ∃ t : ℕ, t = 60 ∧ (t * S_B - t * S_H) % L = 0 :=
by
  sorry

end bruce_bhishma_meet_again_l1551_155103


namespace range_of_composite_function_l1551_155109

noncomputable def range_of_function : Set ℝ :=
  {y | ∃ x : ℝ, y = (1/2) ^ (|x + 1|)}

theorem range_of_composite_function : range_of_function = Set.Ioc 0 1 :=
by
  sorry

end range_of_composite_function_l1551_155109


namespace tangent_line_at_pi_l1551_155139

noncomputable def tangent_equation (x : ℝ) : ℝ := x * Real.sin x

theorem tangent_line_at_pi :
  let f := tangent_equation
  let f' := fun x => Real.sin x + x * Real.cos x
  let x : ℝ := Real.pi
  let y : ℝ := f x
  let slope : ℝ := f' x
  y + slope * x - Real.pi^2 = 0 :=
by
  -- This is where the proof would go
  sorry

end tangent_line_at_pi_l1551_155139


namespace diamond_evaluation_l1551_155150

-- Define the diamond operation as a function using the given table
def diamond (a b : ℕ) : ℕ :=
  match (a, b) with
  | (1, 1) => 4 | (1, 2) => 1 | (1, 3) => 3 | (1, 4) => 2
  | (2, 1) => 1 | (2, 2) => 3 | (2, 3) => 2 | (2, 4) => 4
  | (3, 1) => 3 | (3, 2) => 2 | (3, 3) => 4 | (3, 4) => 1
  | (4, 1) => 2 | (4, 2) => 4 | (4, 3) => 1 | (4, 4) => 3
  | (_, _) => 0  -- default case (should not occur)

-- State the proof problem
theorem diamond_evaluation : diamond (diamond 3 1) (diamond 4 2) = 1 := by
  sorry

end diamond_evaluation_l1551_155150


namespace ternary_to_decimal_l1551_155127

def to_decimal (ternary : Nat) : Nat :=
  match ternary with
  | 121 => 1 * 3^2 + 2 * 3^1 + 1 * 3^0
  | _ => 0

theorem ternary_to_decimal : to_decimal 121 = 16 := by
  sorry

end ternary_to_decimal_l1551_155127


namespace double_neg_five_eq_five_l1551_155142

theorem double_neg_five_eq_five : -(-5) = 5 := 
sorry

end double_neg_five_eq_five_l1551_155142


namespace students_play_both_football_and_cricket_l1551_155173

theorem students_play_both_football_and_cricket :
  ∀ (total F C N both : ℕ),
  total = 460 →
  F = 325 →
  C = 175 →
  N = 50 →
  total - N = F + C - both →
  both = 90 :=
by
  intros
  sorry

end students_play_both_football_and_cricket_l1551_155173


namespace find_range_m_l1551_155195

def p (m : ℝ) : Prop := m > 2 ∨ m < -2
def q (m : ℝ) : Prop := 1 < m ∧ m < 3

theorem find_range_m (h₁ : ¬ p m) (h₂ : q m) : (1 : ℝ) < m ∧ m ≤ 2 :=
by sorry

end find_range_m_l1551_155195


namespace induction_step_l1551_155171

theorem induction_step (x y : ℕ) (k : ℕ) (odd_k : k % 2 = 1) 
  (hk : (x + y) ∣ (x^k + y^k)) : (x + y) ∣ (x^(k+2) + y^(k+2)) :=
sorry

end induction_step_l1551_155171


namespace natural_number_195_is_solution_l1551_155175

-- Define the conditions
def is_odd_digit (n : ℕ) : Prop :=
  n > 0 ∧ n % 2 = 1

def all_digits_odd (n : ℕ) : Prop :=
  ∀ d : ℕ, n / 10 ^ d % 10 < 10 → is_odd_digit (n / 10 ^ d % 10)

-- Define the proof problem
theorem natural_number_195_is_solution :
  195 < 200 ∧ all_digits_odd 195 ∧ (∃ (a b : ℕ), 10 ≤ a ∧ a < 100 ∧ 10 ≤ b ∧ b < 100 ∧ a * b = 195) :=
by
  sorry

end natural_number_195_is_solution_l1551_155175


namespace power_ordering_l1551_155126

theorem power_ordering (a b c : ℝ) : 
  (a = 2^30) → (b = 6^10) → (c = 3^20) → (a < b) ∧ (b < c) :=
by
  intros ha hb hc
  rw [ha, hb, hc]
  have h1 : 6^10 = (3 * 2)^10 := by sorry
  have h2 : 3^20 = (3^10)^2 := by sorry
  have h3 : 2^30 = (2^10)^3 := by sorry
  sorry

end power_ordering_l1551_155126


namespace total_seven_flights_time_l1551_155198

def time_for_nth_flight (n : ℕ) : ℕ :=
  25 + (n - 1) * 8

def total_time_for_flights (n : ℕ) : ℕ :=
  (Finset.range n).sum (λ k => time_for_nth_flight (k + 1))

theorem total_seven_flights_time :
  total_time_for_flights 7 = 343 :=
  by
    sorry

end total_seven_flights_time_l1551_155198


namespace min_cars_needed_l1551_155156

theorem min_cars_needed (h1 : ∀ d ∈ Finset.range 7, ∃ s : Finset ℕ, s.card = 2 ∧ (∃ n : ℕ, 7 * (n - 10) ≥ 2 * n)) : 
  ∃ n, n ≥ 14 :=
by
  sorry

end min_cars_needed_l1551_155156


namespace max_flags_l1551_155167

theorem max_flags (n : ℕ) (h1 : ∀ k, n = 9 * k) (h2 : n ≤ 200)
  (h3 : ∃ m, n = 9 * m + k ∧ k ≤ 2 ∧ k + 1 ≠ 0 ∧ k - 2 ≠ 0) : n = 198 :=
by {
  sorry
}

end max_flags_l1551_155167


namespace average_monthly_increase_l1551_155187

theorem average_monthly_increase (x : ℝ) (turnover_january turnover_march : ℝ)
  (h_jan : turnover_january = 2)
  (h_mar : turnover_march = 2.88)
  (h_growth : turnover_march = turnover_january * (1 + x) * (1 + x)) :
  x = 0.2 :=
by
  sorry

end average_monthly_increase_l1551_155187


namespace travel_time_l1551_155140

-- Definitions: 
def speed := 20 -- speed in km/hr
def distance := 160 -- distance in km

-- Proof statement: 
theorem travel_time (s : ℕ) (d : ℕ) (h1 : s = speed) (h2 : d = distance) : 
  d / s = 8 :=
by {
  sorry
}

end travel_time_l1551_155140


namespace functional_equation_solution_l1551_155110

theorem functional_equation_solution :
  ∀ (f : ℝ → ℝ), (∀ x y : ℝ, (f x + y) * (f (x - y) + 1) = f (f (x * f (x + 1)) - y * f (y - 1))) → (∀ x : ℝ, f x = x) :=
by
  intros f h x
  -- Proof would go here
  sorry

end functional_equation_solution_l1551_155110


namespace average_rainfall_per_hour_eq_l1551_155154

-- Define the conditions
def february_days_non_leap_year : ℕ := 28
def hours_per_day : ℕ := 24
def total_rainfall_in_inches : ℕ := 280
def total_hours_in_february : ℕ := february_days_non_leap_year * hours_per_day

-- Define the goal
theorem average_rainfall_per_hour_eq :
  total_rainfall_in_inches / total_hours_in_february = 5 / 12 :=
sorry

end average_rainfall_per_hour_eq_l1551_155154


namespace red_given_red_l1551_155176

def p_i (i : ℕ) : ℚ := sorry
axiom lights_probs_eq : p_i 1 + p_i 2 = 2 / 3
axiom lights_probs_eq2 : p_i 1 + p_i 3 = 2 / 3
axiom green_given_green : p_i 1 / (p_i 1 + p_i 2) = 3 / 4
axiom total_prob : p_i 1 + p_i 2 + p_i 3 + p_i 4 = 1

theorem red_given_red : (p_i 4 / (p_i 3 + p_i 4)) = 1 / 2 := 
sorry

end red_given_red_l1551_155176


namespace max_gcd_of_polynomials_l1551_155164

def max_gcd (a b : ℤ) : ℤ :=
  let g := Nat.gcd a.natAbs b.natAbs
  Int.ofNat g

theorem max_gcd_of_polynomials :
  ∃ n : ℕ, (n > 0) → max_gcd (14 * ↑n + 5) (9 * ↑n + 2) = 4 :=
by
  sorry

end max_gcd_of_polynomials_l1551_155164


namespace find_larger_number_l1551_155189

theorem find_larger_number (x y : ℕ) (h1 : x + y = 55) (h2 : x - y = 15) : x = 35 := by 
  -- proof will go here
  sorry

end find_larger_number_l1551_155189


namespace average_primes_30_50_l1551_155190

/-- The theorem statement for proving the average of all prime numbers between 30 and 50 is 39.8 -/
theorem average_primes_30_50 : (31 + 37 + 41 + 43 + 47) / 5 = 39.8 :=
  by
  sorry

end average_primes_30_50_l1551_155190


namespace rhombus_side_length_l1551_155129

variables (r α : ℝ) (hα : 0 < α ∧ α < π / 2) (hr : 0 < r)

theorem rhombus_side_length (r α : ℝ) (hα : 0 < α ∧ α < π / 2) (hr : 0 < r) :
  ∃ s : ℝ, s = 2 * r / Real.sin α :=
sorry

end rhombus_side_length_l1551_155129


namespace largest_possible_number_of_markers_l1551_155145

theorem largest_possible_number_of_markers (n_m n_c : ℕ) 
  (h_m : n_m = 72) (h_c : n_c = 48) : Nat.gcd n_m n_c = 24 :=
by
  sorry

end largest_possible_number_of_markers_l1551_155145


namespace triangle_perimeter_l1551_155134

/-- In a triangle ABC, where sides a, b, c are opposite to angles A, B, C respectively.
Given the area of the triangle = 15 * sqrt 3 / 4, 
angle A = 60 degrees and 5 * sin B = 3 * sin C,
prove that the perimeter of triangle ABC is 8 + sqrt 19. -/
theorem triangle_perimeter
  (a b c : ℝ)
  (A B C : ℝ)
  (hA : A = 60)
  (h_area : (1 / 2) * b * c * (Real.sin (A / (180 / Real.pi))) = 15 * Real.sqrt 3 / 4)
  (h_sin : 5 * Real.sin B = 3 * Real.sin C) :
  a + b + c = 8 + Real.sqrt 19 :=
sorry

end triangle_perimeter_l1551_155134


namespace correct_remove_parentheses_l1551_155191

theorem correct_remove_parentheses (a b c d : ℝ) :
  (a - (5 * b - (2 * c - 1)) = a - 5 * b + 2 * c - 1) :=
by sorry

end correct_remove_parentheses_l1551_155191


namespace cycle_cost_price_l1551_155135

theorem cycle_cost_price (SP : ℝ) (loss_percentage : ℝ) (C : ℝ) 
  (h1 : SP = 1360) 
  (h2 : loss_percentage = 0.15) :
  SP = (1 - loss_percentage) * C → C = 1600 :=
by
  sorry

end cycle_cost_price_l1551_155135


namespace solution_of_ab_l1551_155130

theorem solution_of_ab (a b : ℝ) 
  (h1 : ∀ x : ℝ, (ax^2 + b > 0 ↔ x < -1/2 ∨ x > 1/3)) : 
  a * b = 24 := 
sorry

end solution_of_ab_l1551_155130


namespace less_than_half_l1551_155169

theorem less_than_half (a b c : ℝ) (h₁ : a = 43.2) (h₂ : b = 0.5) (h₃ : c = 42.7) : a - b = c := by
  sorry

end less_than_half_l1551_155169


namespace cave_depth_l1551_155197

theorem cave_depth (current_depth remaining_distance : ℕ) (h₁ : current_depth = 849) (h₂ : remaining_distance = 369) :
  current_depth + remaining_distance = 1218 :=
by
  sorry

end cave_depth_l1551_155197
