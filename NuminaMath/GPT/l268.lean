import Mathlib

namespace find_foreign_language_score_l268_268729

variable (c m f : ℝ)

theorem find_foreign_language_score
  (h1 : (c + m + f) / 3 = 94)
  (h2 : (c + m) / 2 = 92) :
  f = 98 := by
  sorry

end find_foreign_language_score_l268_268729


namespace carl_cost_l268_268294

theorem carl_cost (property_damage medical_bills : ℝ) (insurance_coverage : ℝ) (carl_coverage : ℝ) (H1 : property_damage = 40000) (H2 : medical_bills = 70000) (H3 : insurance_coverage = 0.80) (H4 : carl_coverage = 0.20) :
  carl_coverage * (property_damage + medical_bills) = 22000 :=
by
  sorry

end carl_cost_l268_268294


namespace minimum_perimeter_l268_268265

noncomputable def minimum_perimeter_triangle (l m n : ℕ) : ℕ :=
  l + m + n

theorem minimum_perimeter :
  ∀ (l m n : ℕ),
    (l > m) → (m > n) → 
    ((∃ k : ℕ, 10^4 ∣ 3^l - 3^m + k * 10^4) ∧ (∃ k : ℕ, 10^4 ∣ 3^m - 3^n + k * 10^4) ∧ (∃ k : ℕ, 10^4 ∣ 3^l - 3^n + k * 10^4)) →
    minimum_perimeter_triangle l m n = 3003 :=
by
  intros l m n hlm hmn hmod
  sorry

end minimum_perimeter_l268_268265


namespace cone_surface_area_ratio_l268_268041

theorem cone_surface_area_ratio (l : ℝ) (h_l_pos : 0 < l) :
  let θ := (120 * Real.pi) / 180 -- converting 120 degrees to radians
  let side_area := (1/2) * l^2 * θ
  let r := l / 3
  let base_area := Real.pi * r^2
  let surface_area := side_area + base_area
  side_area ≠ 0 → 
  surface_area / side_area = 4 / 3 := 
by
  -- Provide the proof here
  sorry

end cone_surface_area_ratio_l268_268041


namespace darkest_cell_product_l268_268631

theorem darkest_cell_product (a b c d : ℕ)
  (h1 : a > 1) (h2 : b > 1) (h3 : c = a * b)
  (h4 : d = c * (9 * 5) * (9 * 11)) :
  d = 245025 :=
by
  sorry

end darkest_cell_product_l268_268631


namespace problem_statement_equality_condition_l268_268091

theorem problem_statement (x y z : ℝ) (hx : 0 <= x) (hy : 0 <= y) (hz : 0 <= z) :
  (1 + y * z) / (1 + x^2) + (1 + z * x) / (1 + y^2) + (1 + x * y) / (1 + z^2) >= 2 :=
sorry

theorem equality_condition (x y z : ℝ) (hx : 0 <= x) (hy : 0 <= y) (hz : 0 <= z) :
  (1 + y * z) / (1 + x^2) + (1 + z * x) / (1 + y^2) + (1 + x * y) / (1 + z^2) = 2 ↔ x = 0 ∧ y = 0 ∧ z = 0 :=
sorry

end problem_statement_equality_condition_l268_268091


namespace king_zenobius_more_descendants_l268_268191

-- Conditions
def descendants_paphnutius (p2_descendants p1_descendants: ℕ) := 
  2 + 60 * p2_descendants + 20 * p1_descendants = 142

def descendants_zenobius (z3_descendants z1_descendants : ℕ) := 
  4 + 35 * z3_descendants + 35 * z1_descendants = 144

-- Main statement
theorem king_zenobius_more_descendants:
  ∀ (p2_descendants p1_descendants z3_descendants z1_descendants : ℕ),
    descendants_paphnutius p2_descendants p1_descendants →
    descendants_zenobius z3_descendants z1_descendants →
    144 > 142 :=
by
  intros
  sorry

end king_zenobius_more_descendants_l268_268191


namespace sum_of_legs_eq_40_l268_268561

theorem sum_of_legs_eq_40
  (x : ℝ)
  (h1 : x > 0)
  (h2 : x^2 + (x + 2)^2 = 29^2) :
  x + (x + 2) = 40 :=
by
  sorry

end sum_of_legs_eq_40_l268_268561


namespace phase_shift_of_sine_l268_268304

theorem phase_shift_of_sine :
  let a := 3
  let b := 4
  let c := - (Real.pi / 4)
  let phase_shift := -(c / b)
  phase_shift = Real.pi / 16 :=
by
  sorry

end phase_shift_of_sine_l268_268304


namespace smallest_n_l268_268705

theorem smallest_n (n : ℕ) (h : (17 * n - 1) % 11 = 0) : n = 2 := 
by 
    sorry

end smallest_n_l268_268705


namespace max_circles_in_annulus_l268_268124

theorem max_circles_in_annulus (r_inner r_outer : ℝ) (h1 : r_inner = 1) (h2 : r_outer = 9) :
  ∃ n : ℕ, n = 3 ∧ ∀ r : ℝ, r = (r_outer - r_inner) / 2 → r * 3 ≤ 360 :=
sorry

end max_circles_in_annulus_l268_268124


namespace junior_girls_count_l268_268131

theorem junior_girls_count 
  (total_players : ℕ) 
  (boys_percentage : ℝ) 
  (junior_girls : ℕ)
  (h_team : total_players = 50)
  (h_boys_pct : boys_percentage = 0.6)
  (h_junior_girls : junior_girls = ((total_players : ℝ) * (1 - boys_percentage) * 0.5)) : 
  junior_girls = 10 := 
by 
  sorry

end junior_girls_count_l268_268131


namespace number_subtracted_l268_268767

theorem number_subtracted (x y : ℕ) (h₁ : x = 48) (h₂ : 5 * x - y = 102) : y = 138 :=
by
  rw [h₁] at h₂
  sorry

end number_subtracted_l268_268767


namespace fraction_zero_implies_x_eq_two_l268_268518

theorem fraction_zero_implies_x_eq_two (x : ℝ) (h : (x^2 - 4) / (x + 2) = 0) : x = 2 :=
sorry

end fraction_zero_implies_x_eq_two_l268_268518


namespace problem_1_problem_2_l268_268925

variable (a b c : ℝ)

theorem problem_1 
  (hp : 0 < a) 
  (hq : 0 < b) 
  (hr : 0 < c) 
  (hs : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  abc ≤ 1/9 := 
sorry

theorem problem_2 
  (hp : 0 < a) 
  (hq : 0 < b) 
  (hr : 0 < c) 
  (hs : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (abc)) := 
sorry

end problem_1_problem_2_l268_268925


namespace minimum_value_expression_l268_268051

open Real

theorem minimum_value_expression (α β : ℝ) :
  ∃ x y : ℝ, x = 3 * cos α + 4 * sin β ∧ y = 3 * sin α + 4 * cos β ∧
    ((x - 7) ^ 2 + (y - 12) ^ 2) = 242 - 14 * sqrt 193 :=
sorry

end minimum_value_expression_l268_268051


namespace johns_weekly_earnings_increase_l268_268527

noncomputable def percentageIncrease (original new : ℝ) : ℝ :=
  ((new - original) / original) * 100

theorem johns_weekly_earnings_increase :
  percentageIncrease 30 40 = 33.33 :=
by
  sorry

end johns_weekly_earnings_increase_l268_268527


namespace probability_of_union_l268_268787

def total_cards : ℕ := 52
def king_of_hearts : ℕ := 1
def spades : ℕ := 13

theorem probability_of_union :
  let P_A := king_of_hearts / total_cards
  let P_B := spades / total_cards
  (P_A + P_B) = (7 / 26) :=
by
  sorry

end probability_of_union_l268_268787


namespace complex_sum_l268_268197

open Complex

theorem complex_sum (w : ℂ) (h : w^2 - w + 1 = 0) :
  w^103 + w^104 + w^105 + w^106 + w^107 = -1 :=
sorry

end complex_sum_l268_268197


namespace g_g1_eq_43_l268_268165

def g (x : ℝ) : ℝ := 2 * x^2 + 3 * x - 1

theorem g_g1_eq_43 : g (g 1) = 43 :=
by
  sorry

end g_g1_eq_43_l268_268165


namespace range_of_a_l268_268024

noncomputable def f (a x : ℝ) : ℝ := 2 * a^x - exp(1) * x^2

theorem range_of_a (a : ℝ) (x₁ x₂ : ℝ) 
  (h1 : x₁ < x₂)
  (h2 : ∀ x : ℝ, 
          let f' := 2 * a^x * log (a) - 2 * exp(1) * x in 
          f' = 0 → (x = x₁ ∧ ∀ y < x₁, f y < f x) 
             ∨ (x = x₂ ∧ ∀ y > x₂, f y > f x)) 
  (h3 : a > 0) 
  (h4 : a ≠ 1) 
  : (1 / exp(1) < a) ∧ (a < 1) :=
sorry

end range_of_a_l268_268024


namespace coin_collection_problem_l268_268088

theorem coin_collection_problem (n : ℕ) 
  (quarters : ℕ := n / 2)
  (half_dollars : ℕ := 2 * (n / 2))
  (value_nickels : ℝ := 0.05 * n)
  (value_quarters : ℝ := 0.25 * (n / 2))
  (value_half_dollars : ℝ := 0.5 * (2 * (n / 2)))
  (total_value : ℝ := value_nickels + value_quarters + value_half_dollars) :
  total_value = 67.5 ∨ total_value = 135 :=
sorry

end coin_collection_problem_l268_268088


namespace problem_1_problem_2_l268_268834

variable (a b c : ℝ)

theorem problem_1 
  (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  a * b * c ≤ 1 / 9 :=
sorry

theorem problem_2 
  (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * (a * b * c)^(1/2)) :=
sorry

end problem_1_problem_2_l268_268834


namespace sqrt_x_plus_5_l268_268416

theorem sqrt_x_plus_5 (x : ℝ) (h : x = -1) : Real.sqrt (x + 5) = 2 :=
by
  sorry

end sqrt_x_plus_5_l268_268416


namespace find_x_collinear_l268_268151

def vec := ℝ × ℝ

def collinear (u v: vec): Prop :=
  ∃ k: ℝ, u = (k * v.1, k * v.2)

theorem find_x_collinear:
  ∀ (x: ℝ), (let a : vec := (1, 2)
              let b : vec := (x, 1)
              collinear a (a.1 - b.1, a.2 - b.2)) → x = 1 / 2 :=
by
  intros x h
  sorry

end find_x_collinear_l268_268151


namespace pen_defect_probability_l268_268519

theorem pen_defect_probability :
  ∀ (n m : ℕ) (k : ℚ), n = 12 → m = 4 → k = 2 → 
  (8 / 12) * (7 / 11) = 141 / 330 := 
by
  intros n m k h1 h2 h3
  sorry

end pen_defect_probability_l268_268519


namespace probability_two_red_two_green_l268_268431

theorem probability_two_red_two_green (total_red total_blue total_green : ℕ)
  (total_marbles total_selected : ℕ) (probability : ℚ)
  (h_total_marbles: total_marbles = total_red + total_blue + total_green)
  (h_total_selected: total_selected = 4)
  (h_red_selected: 2 ≤ total_red)
  (h_green_selected: 2 ≤ total_green)
  (h_total_selected_le: total_selected ≤ total_marbles)
  (h_probability: probability = (Nat.choose total_red 2 * Nat.choose total_green 2) / (Nat.choose total_marbles total_selected))
  (h_total_red: total_red = 12)
  (h_total_blue: total_blue = 8)
  (h_total_green: total_green = 5):
  probability = 2 / 39 :=
by
  sorry

end probability_two_red_two_green_l268_268431


namespace problem_1_problem_2_l268_268874

theorem problem_1 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) 
  (h_sum : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  abc ≤ 1 / 9 := 
sorry

theorem problem_2 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) 
  (h_sum : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * real.sqrt (abc)) := 
sorry

end problem_1_problem_2_l268_268874


namespace no_integer_n_such_that_squares_l268_268715

theorem no_integer_n_such_that_squares :
  ¬ ∃ n : ℤ, (∃ k1 : ℤ, 10 * n - 1 = k1 ^ 2) ∧
             (∃ k2 : ℤ, 13 * n - 1 = k2 ^ 2) ∧
             (∃ k3 : ℤ, 85 * n - 1 = k3 ^ 2) := 
by sorry

end no_integer_n_such_that_squares_l268_268715


namespace number_of_leap_years_l268_268438

noncomputable def is_leap_year (year : ℕ) : Prop :=
  (year % 1300 = 300 ∨ year % 1300 = 700) ∧ 2000 ≤ year ∧ year ≤ 5000

noncomputable def leap_years : List ℕ :=
  [2900, 4200, 3300, 4600]

theorem number_of_leap_years : leap_years.length = 4 ∧ ∀ y ∈ leap_years, is_leap_year y := by
  sorry

end number_of_leap_years_l268_268438


namespace distribute_weights_l268_268541

theorem distribute_weights (max_weight : ℕ) (w_gbeans w_milk w_carrots w_apples w_bread w_rice w_oranges w_pasta : ℕ)
  (h_max_weight : max_weight = 20)
  (h_w_gbeans : w_gbeans = 4)
  (h_w_milk : w_milk = 6)
  (h_w_carrots : w_carrots = 2 * w_gbeans)
  (h_w_apples : w_apples = 3)
  (h_w_bread : w_bread = 1)
  (h_w_rice : w_rice = 5)
  (h_w_oranges : w_oranges = 2)
  (h_w_pasta : w_pasta = 3)
  : (w_gbeans + w_milk + w_carrots + w_apples + w_bread - 2 = max_weight) ∧ 
    (w_rice + w_oranges + w_pasta + 2 ≤ max_weight) :=
by
  sorry

end distribute_weights_l268_268541


namespace point_above_line_l268_268343

/-- Given the point (-2, t) lies above the line x - 2y + 4 = 0,
    we want to prove t ∈ (1, +∞) -/
theorem point_above_line (t : ℝ) : (-2 - 2 * t + 4 > 0) → t > 1 :=
sorry

end point_above_line_l268_268343


namespace find_angle_degree_l268_268468

-- Define the angle
variable {x : ℝ}

-- Define the conditions
def complement (x : ℝ) : ℝ := 90 - x
def supplement (x : ℝ) : ℝ := 180 - x

-- Define the given condition
def condition (x : ℝ) : Prop := complement x = (1/3) * (supplement x)

-- The theorem statement
theorem find_angle_degree (x : ℝ) (h : condition x) : x = 45 :=
by
  sorry

end find_angle_degree_l268_268468


namespace simplify_problem_1_simplify_problem_2_l268_268550

-- Problem 1: Statement of Simplification Proof
theorem simplify_problem_1 :
  (- (99 + (71 / 72)) * 36 = - (3599 + 1 / 2)) :=
by sorry

-- Problem 2: Statement of Simplification Proof
theorem simplify_problem_2 :
  (-3 * (1 / 4) - 2.5 * (-2.45) + (7 / 2) * (1 / 4) = 6 + 1 / 4) :=
by sorry

end simplify_problem_1_simplify_problem_2_l268_268550


namespace driver_actual_speed_l268_268423

theorem driver_actual_speed (v t : ℝ) 
  (h1 : t > 0) 
  (h2 : v > 0) 
  (cond : v * t = (v + 18) * (2 / 3 * t)) : 
  v = 36 :=
by 
  sorry

end driver_actual_speed_l268_268423


namespace lowest_position_of_vasya_l268_268237

-- Definitions of conditions
def num_cyclists : ℕ := 500
def num_stages : ℕ := 15
def vasya_position_each_stage : ℕ := 7

-- Theorem statement
theorem lowest_position_of_vasya (H1 : ∀ (s: ℕ), s ∈ finset.range(num_stages) → 
(num_cyclists + 1) - vasya_position_each_stage > (num_cyclists - 90))

(assumption_vasya :
  ∀ s ∈ finset.range(num_stages), vasya_position_each_stage < num_cyclists):
  ∃ (lowest_position: ℕ), lowest_position = 91 :=
sorry

end lowest_position_of_vasya_l268_268237


namespace largest_four_digit_integer_congruent_to_17_mod_26_l268_268255

theorem largest_four_digit_integer_congruent_to_17_mod_26 :
  ∃ x : ℤ, 1000 ≤ x ∧ x < 10000 ∧ x % 26 = 17 ∧ x = 9978 :=
by
  sorry

end largest_four_digit_integer_congruent_to_17_mod_26_l268_268255


namespace analytical_expression_f_min_value_f_range_of_k_l268_268802

noncomputable def max_real (a b : ℝ) : ℝ :=
  if a ≥ b then a else b

noncomputable def f (x : ℝ) : ℝ :=
  max_real (|x + 1|) (|x - 2|)

noncomputable def g (x k : ℝ) : ℝ :=
  x^2 - k * f x

-- Problem 1: Proving the analytical expression of f(x)
theorem analytical_expression_f (x : ℝ) :
  f x = if x < 0.5 then 2 - x else x + 1 :=
sorry

-- Problem 2: Proving the minimum value of f(x)
theorem min_value_f : ∃ x : ℝ, (∀ y : ℝ, f y ≥ f x) ∧ f x = 3 / 2 :=
sorry

-- Problem 3: Proving the range of k
theorem range_of_k (k : ℝ) :
  (∀ x : ℝ, x ≤ -1 → (g x k) ≤ (g (x - 1) k)) → k ≤ 2 :=
sorry

end analytical_expression_f_min_value_f_range_of_k_l268_268802


namespace part1_part2_l268_268955

def pos_real := {x : ℝ // 0 < x}

variables (a b c : pos_real)

theorem part1 (h : a.val ^ (3/2) + b.val ^ (3/2) + c.val ^ (3/2) = 1) : a.val * b.val * c.val ≤ 1 / 9 :=
sorry

theorem part2 (h : a.val ^ (3/2) + b.val ^ (3/2) + c.val ^ (3/2) = 1) : 
  a.val / (b.val + c.val) + b.val / (a.val + c.val) + c.val / (a.val + b.val) ≤ 1 / (2 * Real.sqrt (a.val * b.val * c.val)) :=
sorry

end part1_part2_l268_268955


namespace max_min_xy_l268_268403

theorem max_min_xy (x y a : ℝ) (h1 : x + y = a) (h2 : x^2 + y^2 = -a^2 + 2) : (x * y) ∈ { real.min (-1), real.max (1) } :=
by
  have h : a^2 ≤ 4 := 
  sorry
  have xy_eq : x * y = (a^2 - 2) / 2 := 
  sorry
  split
  { calc min (xy_eq) = -1 :=
    sorry
    calc max (xy_eq) = 1 :=
    sorry
  }

end max_min_xy_l268_268403


namespace boat_speed_in_still_water_l268_268427

-- Definitions for conditions
variables (V_b V_s : ℝ)

-- The conditions provided for the problem
def along_stream := V_b + V_s = 13
def against_stream := V_b - V_s = 5

-- The theorem we want to prove
theorem boat_speed_in_still_water (h1 : along_stream V_b V_s) (h2 : against_stream V_b V_s) : V_b = 9 :=
sorry

end boat_speed_in_still_water_l268_268427


namespace cosine_identity_l268_268483

theorem cosine_identity
  (α β : ℝ)
  (h1 : Real.sin (α - β) = 1 / 3)
  (h2 : Real.cos α * Real.sin β = 1 / 6) :
  Real.cos (2 * α + 2 * β) = 1 / 9 :=
  sorry

end cosine_identity_l268_268483


namespace trajectory_of_G_l268_268497

def circle (x y : ℝ) := (x + real.sqrt 7)^2 + y^2 = 64
def N : ℝ × ℝ := (real.sqrt 7, 0)
def on_circle (x y : ℝ) := circle x y
def midpoint (A B Q : ℝ × ℝ) := 2 * Q = A + B
def perp_vector (A B Q G : ℝ × ℝ) := (G - Q).1 * (B - A).1 + (G - Q).2 * (B - A).2 = 0
def major_axis : ℝ := 4
def semi_focal_distance : ℝ := real.sqrt 7

theorem trajectory_of_G :
  ∀ (P Q G : ℝ × ℝ), on_circle P.1 P.2 →
  midpoint N P Q →
  perp_vector P N Q G →
  (G.1^2 / 16) + (G.2^2 / 9) = 1 :=
by
  assume P Q G h_circle h_midpoint h_perpendicular
  -- Proof omitted
  sorry

end trajectory_of_G_l268_268497


namespace probability_multiple_of_3_or_4_l268_268569

theorem probability_multiple_of_3_or_4 :
  let numbers := {n | 1 ≤ n ∧ n ≤ 30},
      multiples_of_3 := {n | n ∈ numbers ∧ n % 3 = 0},
      multiples_of_4 := {n | n ∈ numbers ∧ n % 4 = 0},
      multiples_of_12 := {n | n ∈ numbers ∧ n % 12 = 0},
      favorable_outcomes := multiples_of_3 ∪ multiples_of_4,
      double_counted_outcomes := multiples_of_12,
      total_favorable_outcomes := set.card favorable_outcomes - set.card double_counted_outcomes,
      total_outcomes := set.card numbers in
  total_favorable_outcomes / total_outcomes = 1 / 2 := by
  sorry

end probability_multiple_of_3_or_4_l268_268569


namespace binary_subtraction_result_l268_268013

theorem binary_subtraction_result :
  let x := 0b1101101 -- binary notation for 109
  let y := 0b11101   -- binary notation for 29
  let z := 0b101010  -- binary notation for 42
  let product := x * y
  let result := product - z
  result = 0b10000010001 := -- binary notation for 3119
by
  sorry

end binary_subtraction_result_l268_268013


namespace part1_part2_l268_268983

noncomputable theory

variables {a b c : ℝ}

-- Condition that a, b, c are positive numbers and the given equation holds
def pos_numbers : Prop := a > 0 ∧ b > 0 ∧ c > 0
def sum_condition : Prop := (a ^ (3/2)) + (b ^ (3/2)) + (c ^ (3/2)) = 1

-- Proof goal for the first part: abc <= 1/9
theorem part1 (h₁ : pos_numbers) (h₂ : sum_condition) : 
  a * b * c ≤ 1 / 9 :=
sorry

-- Proof goal for the second part: a/(b+c) + b/(a+c) + c/(a+b) <= 1/(2 * sqrt(abc))
theorem part2 (h₁ : pos_numbers) (h₂ : sum_condition) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * sqrt (a * b * c)) :=
sorry

end part1_part2_l268_268983


namespace jellybean_problem_l268_268728

theorem jellybean_problem 
    (T L A : ℕ) 
    (h1 : T = L + 24) 
    (h2 : A = L / 2) 
    (h3 : T = 34) : 
    A = 5 := 
by 
  sorry

end jellybean_problem_l268_268728


namespace todd_ate_cupcakes_l268_268204

def total_cupcakes_baked := 68
def packages := 6
def cupcakes_per_package := 6
def total_packaged_cupcakes := packages * cupcakes_per_package
def remaining_cupcakes := total_cupcakes_baked - total_packaged_cupcakes

theorem todd_ate_cupcakes : total_cupcakes_baked - remaining_cupcakes = 36 := by
  sorry

end todd_ate_cupcakes_l268_268204


namespace problem_1_problem_2_l268_268830

variable (a b c : ℝ)

theorem problem_1 
  (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  a * b * c ≤ 1 / 9 :=
sorry

theorem problem_2 
  (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * (a * b * c)^(1/2)) :=
sorry

end problem_1_problem_2_l268_268830


namespace evan_books_in_ten_years_l268_268790

def E4 : ℕ := 400
def E_now : ℕ := E4 - 80
def E2 : ℕ := E_now / 2
def E10 : ℕ := 6 * E2 + 120

theorem evan_books_in_ten_years : E10 = 1080 := by
sorry

end evan_books_in_ten_years_l268_268790


namespace will_can_buy_correct_amount_of_toys_l268_268737

-- Define the initial conditions as constants
def initial_amount : Int := 57
def amount_spent : Int := 27
def cost_per_toy : Int := 6

-- Lemma stating the problem to prove.
theorem will_can_buy_correct_amount_of_toys : (initial_amount - amount_spent) / cost_per_toy = 5 :=
by
  sorry

end will_can_buy_correct_amount_of_toys_l268_268737


namespace probability_not_eat_pizza_l268_268404

theorem probability_not_eat_pizza (P_eat_pizza : ℚ) (h : P_eat_pizza = 5 / 8) : 
  ∃ P_not_eat_pizza : ℚ, P_not_eat_pizza = 3 / 8 :=
by
  use 1 - P_eat_pizza
  sorry

end probability_not_eat_pizza_l268_268404


namespace triangle_angle_C_and_equilateral_l268_268495

variables (a b c A B C : ℝ)
variables (h_triangle : 0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π)
variables (h_perpendicular : (a + c) * (a - c) + (b - a) * b = 0)
variables (h_sine : 2 * (Real.sin (A / 2)) ^ 2 + 2 * (Real.sin (B / 2)) ^ 2 = 1)

theorem triangle_angle_C_and_equilateral (h_triangle : 0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π) 
                                         (h_perpendicular : (a + c) * (a - c) + (b - a) * b = 0)
                                         (h_sine : 2 * (Real.sin (A / 2)) ^ 2 + 2 * (Real.sin (B / 2)) ^ 2 = 1) :
  C = π / 3 ∧ A = π / 3 ∧ B = π / 3 :=
sorry

end triangle_angle_C_and_equilateral_l268_268495


namespace circumcenter_on_line_AM_l268_268546

open EuclideanGeometry

-- Define points and their positions
variables {P : Type*} [MetricSpace P] [InnerProductSpace ℝ P] (A M B C : P)
variables (hAinside : ∃ X Y : P, ∃ α β : ℝ, 0 < α ∧ 0 < β ∧ 
                        Angle A X M = α ∧ Angle A Y M = β)

-- Define the ray reflection conditions
variables (h_refl : ∃ R S : P, is_ray A R ∧ is_ray R S ∧ is_ray S A ∧
                    reflects R B ∧ reflects S C ∧ Angle A R B = Angle B C S)

-- Define the reflection and angle equality
variables (h_reflect : ∀ X Y Z : P, Angle_of_reflection X Y = Angle_of_reflection Y Z)

-- Define the circumcenter of triangle
axiom center_of_circumcircle_tri (x y z : P) : ∃ c : P, c = circumcenter x y z

-- Mathematical statement to prove equivalence
theorem circumcenter_on_line_AM : ∃ O : P, O = circumcenter B C M ∧ lies_on_line O A M :=
by {
  sorry
}

end circumcenter_on_line_AM_l268_268546


namespace lines_parallel_if_perpendicular_to_plane_l268_268510

axiom line : Type
axiom plane : Type

-- Definitions of perpendicular and parallel
axiom perp : line → plane → Prop
axiom parallel : line → line → Prop

variables (a b : line) (α : plane)

theorem lines_parallel_if_perpendicular_to_plane (h1 : perp a α) (h2 : perp b α) : parallel a b :=
sorry

end lines_parallel_if_perpendicular_to_plane_l268_268510


namespace root_product_l268_268730

theorem root_product : (Real.sqrt (Real.sqrt 81) * Real.cbrt 27 * Real.sqrt 9 = 27) :=
by
  sorry

end root_product_l268_268730


namespace max_m_plus_n_l268_268577

noncomputable def quadratic_function (x : ℝ) : ℝ := -x^2 + 3

theorem max_m_plus_n (m n : ℝ) (h : n = quadratic_function m) : m + n ≤ 13/4 :=
sorry

end max_m_plus_n_l268_268577


namespace abc_le_one_ninth_sum_ratios_le_one_over_two_sqrt_abc_l268_268811

open Real

variables (a b c : ℝ)

-- Condition: a, b, c are positive numbers
variables (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
-- Condition: a^(3/2) + b^(3/2) + c^(3/2) = 1
variable  (h_sum : a^(3/2) + b^(3/2) + c^(3/2) = 1)

-- Question 1: Prove abc ≤ 1/9
theorem abc_le_one_ninth : a * b * c ≤ 1 / 9 :=
  sorry

-- Question 2: Prove (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ 1 / (2 * sqrt (a * b * c))
theorem sum_ratios_le_one_over_two_sqrt_abc : 
  (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ 1 / (2 * sqrt (a * b * c)) :=
  sorry

end abc_le_one_ninth_sum_ratios_le_one_over_two_sqrt_abc_l268_268811


namespace dave_has_20_more_than_derek_l268_268785

-- Define the amounts of money Derek and Dave start with
def initial_amount_derek : ℕ := 40
def initial_amount_dave : ℕ := 50

-- Define the amounts Derek spends
def spend_derek_lunch_self1 : ℕ := 14
def spend_derek_lunch_dad : ℕ := 11
def spend_derek_lunch_self2 : ℕ := 5
def spend_derek_dessert_sister : ℕ := 8

-- Define the amounts Dave spends
def spend_dave_lunch_mom : ℕ := 7
def spend_dave_lunch_cousin : ℕ := 12
def spend_dave_snacks_friends : ℕ := 9

-- Define calculations for total spending
def total_spend_derek : ℕ :=
  spend_derek_lunch_self1 + spend_derek_lunch_dad + spend_derek_lunch_self2 + spend_derek_dessert_sister

def total_spend_dave : ℕ :=
  spend_dave_lunch_mom + spend_dave_lunch_cousin + spend_dave_snacks_friends

-- Define remaining amount of money
def remaining_derek : ℕ :=
  initial_amount_derek - total_spend_derek

def remaining_dave : ℕ :=
  initial_amount_dave - total_spend_dave

-- Define the property to be proved
theorem dave_has_20_more_than_derek : remaining_dave - remaining_derek = 20 := by
  sorry

end dave_has_20_more_than_derek_l268_268785


namespace johny_travelled_South_distance_l268_268049

theorem johny_travelled_South_distance :
  ∃ S : ℝ, S + (S + 20) + 2 * (S + 20) = 220 ∧ S = 40 :=
by
  sorry

end johny_travelled_South_distance_l268_268049


namespace max_m_n_value_l268_268583

theorem max_m_n_value : ∀ (m n : ℝ), (n = -m^2 + 3) → m + n ≤ 13 / 4 :=
by
  intros m n h
  -- The proof will go here, which is omitted for now.
  sorry

end max_m_n_value_l268_268583


namespace radical_product_l268_268732

def fourth_root (x : ℝ) : ℝ := x ^ (1/4)
def third_root (x : ℝ) : ℝ := x ^ (1/3)
def square_root (x : ℝ) : ℝ := x ^ (1/2)

theorem radical_product :
  fourth_root 81 * third_root 27 * square_root 9 = 27 := 
by
  sorry

end radical_product_l268_268732


namespace lowest_position_l268_268234

theorem lowest_position (num_cyclists : ℕ) (num_stages : ℕ) (vasya_position : ℕ) :
  num_cyclists = 500 →
  num_stages = 15 →
  vasya_position = 7 →
  ∃ n, n = 91 :=
by
  intros
  sorry

end lowest_position_l268_268234


namespace at_least_one_equation_has_real_roots_l268_268158

noncomputable def has_two_distinct_real_roots (a b c : ℝ) : Prop :=
  (4 * b^2 - 4 * a * c > 0) ∨ (4 * c^2 - 4 * a * b > 0) ∨ (4 * a^2 - 4 * b * c > 0)

theorem at_least_one_equation_has_real_roots (a b c : ℝ) (h : a ≠ b ∧ b ≠ c ∧ c ≠ a) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) :
  has_two_distinct_real_roots a b c :=
by
  sorry

end at_least_one_equation_has_real_roots_l268_268158


namespace max_volume_of_open_top_box_l268_268757

noncomputable def box_max_volume (x : ℝ) : ℝ :=
  (10 - 2 * x) * (16 - 2 * x) * x

theorem max_volume_of_open_top_box : ∃ x : ℝ, 0 < x ∧ x < 5 ∧ box_max_volume x = 144 :=
by
  sorry

end max_volume_of_open_top_box_l268_268757


namespace part1_part2_l268_268999

-- a), b), c are positive real numbers and ${a}^{\frac{3}{2}}+{b}^{\frac{3}{2}}+{c}^{\frac{3}{2}}=1$
variables (a b c : ℝ)
axiom a_pos : 0 < a
axiom b_pos : 0 < b
axiom c_pos : 0 < c
axiom sum_eq_one : a^(3/2) + b^(3/2) + c^(3/2) = 1

-- Prove (1) and (2)
theorem part1 : abc ≤ 1 / 9 := sorry

theorem part2 : a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (abc)) := sorry

end part1_part2_l268_268999


namespace factorize_expression_l268_268010

theorem factorize_expression (x : ℝ) : (x - 1) * (x + 3) + 4 = (x + 1) ^ 2 :=
by sorry

end factorize_expression_l268_268010


namespace jerry_feathers_left_l268_268185

def hawk_feathers : ℕ := 37
def eagle_feathers : ℝ := 17.5 * hawk_feathers
def total_feathers : ℝ := hawk_feathers + eagle_feathers
def feathers_to_sister : ℝ := 0.45 * total_feathers
def remaining_feathers_after_sister : ℝ := total_feathers - feathers_to_sister
def feathers_sold : ℝ := 0.85 * remaining_feathers_after_sister
def final_remaining_feathers : ℝ := remaining_feathers_after_sister - feathers_sold

theorem jerry_feathers_left : ⌊final_remaining_feathers⌋₊ = 56 := by
  sorry

end jerry_feathers_left_l268_268185


namespace part1_part2_l268_268893

variables {a b c : ℝ}

theorem part1 (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h : a^(3/2) + b^(3/2) + c^(3/2) = 1) : abc ≤ 1/9 := 
sorry

theorem part2 (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (abc)) :=
sorry

end part1_part2_l268_268893


namespace percentage_temporary_employees_is_correct_l268_268682

noncomputable def percentage_temporary_employees
    (technicians_percentage : ℝ) (skilled_laborers_percentage : ℝ) (unskilled_laborers_percentage : ℝ)
    (permanent_technicians_percentage : ℝ) (permanent_skilled_laborers_percentage : ℝ)
    (permanent_unskilled_laborers_percentage : ℝ) : ℝ :=
  let total_workers : ℝ := 100
  let total_temporary_technicians := technicians_percentage * (1 - permanent_technicians_percentage / 100)
  let total_temporary_skilled_laborers := skilled_laborers_percentage * (1 - permanent_skilled_laborers_percentage / 100)
  let total_temporary_unskilled_laborers := unskilled_laborers_percentage * (1 - permanent_unskilled_laborers_percentage / 100)
  let total_temporary_workers := total_temporary_technicians + total_temporary_skilled_laborers + total_temporary_unskilled_laborers
  (total_temporary_workers / total_workers) * 100

theorem percentage_temporary_employees_is_correct :
  percentage_temporary_employees 40 35 25 60 45 35 = 51.5 :=
by
  sorry

end percentage_temporary_employees_is_correct_l268_268682


namespace increase_in_surface_area_l268_268141

-- Define the edge length of the original cube and other conditions
variable (a : ℝ)

-- Define the increase in surface area problem
theorem increase_in_surface_area (h : 1 ≤ 27) : 
  let original_surface_area := 6 * a^2
  let smaller_cube_edge := a / 3
  let smaller_surface_area := 6 * (smaller_cube_edge)^2
  let total_smaller_surface_area := 27 * smaller_surface_area
  total_smaller_surface_area - original_surface_area = 12 * a^2 :=
by
  -- Provided the proof to satisfy Lean 4 syntax requirements to check for correctness
  sorry

end increase_in_surface_area_l268_268141


namespace average_T_is_10_l268_268216

def count_adjacent_bg_pairs (row : List (string)) : ℕ :=
  (List.zipWith (λ a b => if (a = "B" ∧ b = "G") ∨ (a = "G" ∧ b = "B") then 1 else 0) row (row.tail)).sum

theorem average_T_is_10 (row : List (string)) :
  (List.length row = 20) →
  (row.count "B" = 8) →
  (row.count "G" = 12) →
  (count_adjacent_bg_pairs row).to_real / 19 = 10 :=
by
  sorry

end average_T_is_10_l268_268216


namespace problem_part1_problem_part2_l268_268948

variable {a b c : ℝ}

theorem problem_part1 (h : a > 0 ∧ b > 0 ∧ c > 0) (sum_eq : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  abc ≤ 1 / 9 :=
sorry

theorem problem_part2 (h : a > 0 ∧ b > 0 ∧ c > 0) (sum_eq : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (abc)) :=
sorry

end problem_part1_problem_part2_l268_268948


namespace digit_at_1286th_position_l268_268347

def naturally_written_sequence : ℕ → ℕ := sorry

theorem digit_at_1286th_position : naturally_written_sequence 1286 = 3 :=
sorry

end digit_at_1286th_position_l268_268347


namespace picnic_students_count_l268_268434

theorem picnic_students_count (x : ℕ) (h1 : (x / 2) + (x / 3) + (x / 4) = 65) : x = 60 :=
by
  -- Proof goes here
  sorry

end picnic_students_count_l268_268434


namespace zoo_animals_total_l268_268444

-- Conditions as definitions
def initial_animals : ℕ := 68
def gorillas_sent_away : ℕ := 6
def hippopotamus_adopted : ℕ := 1
def rhinos_taken_in : ℕ := 3
def lion_cubs_born : ℕ := 8
def meerkats_per_cub : ℕ := 2

-- Theorem to prove the resulting number of animals
theorem zoo_animals_total :
  (initial_animals - gorillas_sent_away + hippopotamus_adopted + rhinos_taken_in + lion_cubs_born + meerkats_per_cub * lion_cubs_born) = 90 :=
by 
  sorry

end zoo_animals_total_l268_268444


namespace Zenobius_more_descendants_l268_268189

/-- Total number of descendants in King Pafnutius' lineage --/
def descendants_Pafnutius : Nat :=
  2 + 60 * 2 + 20 * 1

/-- Total number of descendants in King Zenobius' lineage --/
def descendants_Zenobius : Nat :=
  4 + 35 * 3 + 35 * 1

theorem Zenobius_more_descendants : descendants_Zenobius > descendants_Pafnutius := by
  sorry

end Zenobius_more_descendants_l268_268189


namespace find_g_3_l268_268152

-- Definitions and conditions
variable (g : ℝ → ℝ)
variable (h : ∀ x : ℝ, g (x - 1) = 2 * x + 6)

-- Theorem: Proof problem corresponding to the problem
theorem find_g_3 : g 3 = 14 :=
by
  -- Insert proof here
  sorry

end find_g_3_l268_268152


namespace total_profit_correct_l268_268745

variables (x y : ℝ) -- B's investment and period
variables (B_profit : ℝ) -- profit received by B
variable (A_investment : ℝ) -- A's investment

-- Given conditions
def A_investment_cond := A_investment = 3 * x
def period_cond := 2 * y
def B_profit_given := B_profit = 4500
def total_profit := 7 * B_profit

theorem total_profit_correct :
  (A_investment = 3 * x)
  ∧ (B_profit = 4500)
  ∧ ((6 * x * 2 * y) / (x * y) = 6)
  → total_profit = 31500 :=
by sorry

end total_profit_correct_l268_268745


namespace probability_empty_chair_on_sides_7_chairs_l268_268698

theorem probability_empty_chair_on_sides_7_chairs :
  (1 : ℚ) / (35 : ℚ) = 0.2 := by
  sorry

end probability_empty_chair_on_sides_7_chairs_l268_268698


namespace original_weight_calculation_l268_268599

-- Conditions
variable (postProcessingWeight : ℝ) (originalWeight : ℝ)
variable (lostPercentage : ℝ)

-- Problem Statement
theorem original_weight_calculation
  (h1 : postProcessingWeight = 240)
  (h2 : lostPercentage = 0.40) :
  originalWeight = 400 :=
sorry

end original_weight_calculation_l268_268599


namespace sin_2theta_value_l268_268667

theorem sin_2theta_value (θ : ℝ) (h : ∑' n, (Real.sin θ)^(2 * n) = 3) : Real.sin (2 * θ) = (2 * Real.sqrt 2) / 3 :=
by
  sorry

end sin_2theta_value_l268_268667


namespace sum_of_all_possible_values_of_x_l268_268099

noncomputable def sum_of_roots_of_equation : ℚ :=
  let eq : Polynomial ℚ := 4 * Polynomial.X ^ 2 + 3 * Polynomial.X - 5
  let roots := eq.roots
  roots.sum

theorem sum_of_all_possible_values_of_x :
  sum_of_roots_of_equation = -3/4 := 
  sorry

end sum_of_all_possible_values_of_x_l268_268099


namespace find_x_l268_268509

def a : ℝ × ℝ := (-2, 0)
def b : ℝ × ℝ := (2, 1)
def c (x : ℝ) : ℝ × ℝ := (x, -1)
def scalar_multiply (k : ℝ) (v : ℝ × ℝ) : ℝ × ℝ := (k * v.1, k * v.2)
def vector_add (v1 v2 : ℝ × ℝ) : ℝ × ℝ := (v1.1 + v2.1, v1.2 + v2.2)
def collinear (v1 v2 : ℝ × ℝ) : Prop := v1.1 * v2.2 = v1.2 * v2.1

theorem find_x :
  ∃ x : ℝ, collinear (vector_add (scalar_multiply 3 a) b) (c x) ∧ x = 4 :=
by
  sorry

end find_x_l268_268509


namespace selection_methods_at_least_one_AB_l268_268800

theorem selection_methods_at_least_one_AB : 
  ∀ (C : ℕ → ℕ → ℕ),
    (C 10 4) - (C 8 4) = 140 :=
by
  sorry

end selection_methods_at_least_one_AB_l268_268800


namespace max_m_plus_n_l268_268578

noncomputable def quadratic_function (x : ℝ) : ℝ := -x^2 + 3

theorem max_m_plus_n (m n : ℝ) (h : n = quadratic_function m) : m + n ≤ 13/4 :=
sorry

end max_m_plus_n_l268_268578


namespace am_gm_inequality_l268_268153

theorem am_gm_inequality (a b c : ℝ) (h : a * b * c = 1 / 8) : 
  a^2 + b^2 + c^2 + a^2 * b^2 + b^2 * c^2 + c^2 * a^2 ≥ 15 / 16 :=
sorry

end am_gm_inequality_l268_268153


namespace range_f_l268_268733

noncomputable def f (x : ℝ) : ℝ := 1 / (x^2 + 1)

theorem range_f : Set.Ioo 0 1 ∪ {1} = {y : ℝ | ∃ x : ℝ, f x = y} :=
by 
  sorry

end range_f_l268_268733


namespace problem1_problem2_l268_268895

open Real

variable {a b c : ℝ}

-- Condition: a, b, c are positive and a^{3/2} + b^{3/2} + c^{3/2} = 1
def conditions (a b c : ℝ) := a > 0 ∧ b > 0 ∧ c > 0 ∧ a^(3/2) + b^(3/2) + c^(3/2) = 1

-- Problem 1: Prove that abc <= 1/9
theorem problem1 (ha : conditions a b c) : a * b * c ≤ 1 / 9 :=
sorry

-- Problem 2: Prove that a/(b+c) + b/(a+c) + c/(a+b) <= 1 / (2 * sqrt(abc))
theorem problem2 (ha : conditions a b c) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * sqrt (a * b * c)) :=
sorry

end problem1_problem2_l268_268895


namespace find_time_period_l268_268633

theorem find_time_period (P r CI : ℝ) (n : ℕ) (A : ℝ) (t : ℝ) 
  (hP : P = 10000)
  (hr : r = 0.15)
  (hCI : CI = 3886.25)
  (hn : n = 1)
  (hA : A = P + CI)
  (h_formula : A = P * (1 + r / n) ^ (n * t)) : 
  t = 2 := 
  sorry

end find_time_period_l268_268633


namespace arrangement_count_l268_268588

def no_adjacent_students_arrangements (teachers students : ℕ) : ℕ :=
  if teachers = 3 ∧ students = 3 then 144 else 0

theorem arrangement_count :
  no_adjacent_students_arrangements 3 3 = 144 :=
by
  sorry

end arrangement_count_l268_268588


namespace smallest_n_divisible_l268_268262

theorem smallest_n_divisible (n : ℕ) : 
  (450 ∣ n^3) ∧ (2560 ∣ n^4) ↔ n = 60 :=
by {
  sorry
}

end smallest_n_divisible_l268_268262


namespace Petya_cannot_achieve_goal_l268_268128

theorem Petya_cannot_achieve_goal (n : ℕ) (h : n ≥ 2) :
  ¬ (∃ (G : ℕ → Prop), (∀ i : ℕ, (G i ↔ (G ((i + 2) % (2 * n))))) ∨ (G (i + 1) ≠ G (i + 2))) :=
sorry

end Petya_cannot_achieve_goal_l268_268128


namespace find_k_l268_268647

noncomputable def arithmetic_sequence_sum (a₁ d : ℕ) (n : ℕ) : ℕ :=
  n * a₁ + (n * (n-1)) / 2 * d

theorem find_k (a₁ d : ℕ) (S : ℕ → ℕ) (k : ℕ) 
  (h₁ : a₁ = 1) (h₂ : d = 2) (h₃ : ∀ n, S (n+2) = 28 + S n) :
  k = 6 := by
  sorry

end find_k_l268_268647


namespace admission_charge_l268_268125

variable (A : ℝ) -- Admission charge in dollars
variable (tour_charge : ℝ)
variable (group1_size : ℕ)
variable (group2_size : ℕ)
variable (total_earnings : ℝ)

-- Given conditions
axiom h1 : tour_charge = 6
axiom h2 : group1_size = 10
axiom h3 : group2_size = 5
axiom h4 : total_earnings = 240
axiom h5 : (group1_size * A + group1_size * tour_charge) + (group2_size * A) = total_earnings

theorem admission_charge : A = 12 :=
by
  sorry

end admission_charge_l268_268125


namespace problem1_problem2_l268_268844

variable {a b c : ℝ}

-- Conditions: a, b, c are positive numbers and a^(3/2) + b^(3/2) + c^(3/2) = 1
axiom pos_a : 0 < a
axiom pos_b : 0 < b
axiom pos_c : 0 < c
axiom sum_pow : a^(3/2) + b^(3/2) + c^(3/2) = 1

-- Problem (1): Prove abc ≤ 1/9 given the conditions
theorem problem1 : abc ≤ 1 / 9 := by
  sorry

-- Problem (2): Prove (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ (1 / (2 * (abc) ^ (1/2))) given the conditions
theorem problem2 : (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ 1 / (2 * (abc) ^ (1/2)) := by
  sorry

end problem1_problem2_l268_268844


namespace amber_worked_hours_l268_268286

-- Define the variables and conditions
variables (A : ℝ) (Armand_hours : ℝ) (Ella_hours : ℝ)
variables (h1 : Armand_hours = A / 3) (h2 : Ella_hours = 2 * A)
variables (h3 : A + Armand_hours + Ella_hours = 40)

-- Prove the statement
theorem amber_worked_hours : A = 12 :=
by
  sorry

end amber_worked_hours_l268_268286


namespace intersection_of_A_and_B_l268_268677

def A : Set ℝ := { x | x > 2 ∨ x < -1 }
def B : Set ℝ := { x | (x + 1) * (4 - x) < 4 }

theorem intersection_of_A_and_B :
  A ∩ B = { x | x > 3 ∨ x < -1 } := sorry

end intersection_of_A_and_B_l268_268677


namespace least_area_exists_l268_268113

-- Definition of the problem conditions
def is_rectangle (l w : ℕ) : Prop :=
  2 * (l + w) = 120

def area (l w : ℕ) := l * w

-- Statement of the proof problem
theorem least_area_exists :
  ∃ (l w : ℕ), is_rectangle l w ∧ (∀ (l' w' : ℕ), is_rectangle l' w' → area l w ≤ area l' w') ∧ area l w = 59 :=
sorry

end least_area_exists_l268_268113


namespace probability_correct_match_l268_268435

/-- 
A contest organizer uses a game where six historical figures are paired with quotes incorrectly list next to their portraits. 
Participants should guess which quote belongs to which historical figure. What is the probability that a participant guessing 
at random will match all six correctly?
-/
theorem probability_correct_match : 
  (1 / Nat.factorial 6 : ℚ) = 1 / 720 :=
by
  simp [Nat.factorial]
  sorry

end probability_correct_match_l268_268435


namespace fraction_dropped_l268_268116

theorem fraction_dropped (f : ℝ) 
  (h1 : 0 ≤ f ∧ f ≤ 1) 
  (initial_passengers : ℝ) 
  (final_passenger_count : ℝ)
  (first_pickup : ℝ)
  (second_pickup : ℝ) 
  (first_drop_factor : ℝ)
  (second_drop_factor : ℕ):
  initial_passengers = 270 →
  final_passenger_count = 242 →
  first_pickup = 280 →
  second_pickup = 12 →
  first_drop_factor = f →
  second_drop_factor = 2 →
  ((initial_passengers - initial_passengers * first_drop_factor) + first_pickup) / second_drop_factor + second_pickup = final_passenger_count →
  f = 1 / 3 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end fraction_dropped_l268_268116


namespace value_of_k_l268_268513

theorem value_of_k (k : ℝ) (h1 : k ≠ 0) (h2 : ∀ x₁ x₂ : ℝ, x₁ < x₂ → (k * x₁ - 100) < (k * x₂ - 100)) : k = 1 :=
by
  have h3 : k > 0 :=
    sorry -- We know that if y increases as x increases, then k > 0
  have h4 : k = 1 :=
    sorry -- For this specific problem, we can take k = 1 which satisfies the conditions
  exact h4

end value_of_k_l268_268513


namespace minimize_distance_sum_l268_268809

open Real

noncomputable def distance_squared (x y : ℝ × ℝ) : ℝ :=
  (x.1 - y.1)^2 + (x.2 - y.2)^2

theorem minimize_distance_sum : 
  ∀ P : ℝ × ℝ, (P.1 = P.2) → 
    let A : ℝ × ℝ := (1, -1)
    let B : ℝ × ℝ := (2, 2)
    (distance_squared P A + distance_squared P B) ≥ 
    (distance_squared (1, 1) A + distance_squared (1, 1) B) := by
  intro P hP
  let A : ℝ × ℝ := (1, -1)
  let B : ℝ × ℝ := (2, 2)
  sorry

end minimize_distance_sum_l268_268809


namespace eccentricity_of_ellipse_l268_268496

theorem eccentricity_of_ellipse 
  (a b c m n : ℝ) 
  (h1 : a > b) 
  (h2 : b > 0) 
  (h3 : m > 0) 
  (h4 : n > 0) 
  (ellipse_eq : ∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1 -> (m^2 + n^2 > x^2 + y^2))
  (hyperbola_eq : ∀ x y : ℝ, x^2 / m^2 - y^2 / n^2 = 1 -> (m^2 + n^2 > x^2 - y^2))
  (same_foci: ∀ (x y : ℝ), x^2 / a^2 + y^2 / b^2 <= 1 → x^2 / m^2 - y^2 / n^2 = 1)
  (geometric_mean : c^2 = a * m)
  (arithmetic_mean : 2 * n^2 = 2 * m^2 + c^2) : 
  (c / a = 1 / 2) :=
sorry

end eccentricity_of_ellipse_l268_268496


namespace problem_1_problem_2_l268_268828

variable (a b c : ℝ)

theorem problem_1 
  (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  a * b * c ≤ 1 / 9 :=
sorry

theorem problem_2 
  (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * (a * b * c)^(1/2)) :=
sorry

end problem_1_problem_2_l268_268828


namespace max_m_n_value_l268_268585

theorem max_m_n_value : ∀ (m n : ℝ), (n = -m^2 + 3) → m + n ≤ 13 / 4 :=
by
  intros m n h
  -- The proof will go here, which is omitted for now.
  sorry

end max_m_n_value_l268_268585


namespace add_fractions_l268_268302

theorem add_fractions: (2 / 5) + (3 / 8) = 31 / 40 := 
by 
  sorry

end add_fractions_l268_268302


namespace petya_digits_sum_l268_268371

def distinct_digits (a b c d : ℕ) : Prop :=
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d

theorem petya_digits_sum (a b c d : ℕ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) (h4 : d ≠ 0)
    (h_distinct: distinct_digits a b c d)
    (h_sum: 6 * (a + b + c + d) * 1111 = 73326) :
    a + b + c + d = 11 ∧ {a, b, c, d} = {1, 2, 3, 5} :=
by
  sorry

end petya_digits_sum_l268_268371


namespace Liam_savings_after_trip_and_bills_l268_268538

theorem Liam_savings_after_trip_and_bills :
  let trip_cost := 7000
  let bills_cost := 3500
  let monthly_savings := 500
  let years := 2
  let total_savings := monthly_savings * 12 * years
  total_savings - bills_cost - trip_cost = 1500 := by
  let trip_cost := 7000
  let bills_cost := 3500
  let monthly_savings := 500
  let years := 2
  let total_savings := monthly_savings * 12 * years
  sorry

end Liam_savings_after_trip_and_bills_l268_268538


namespace sector_area_l268_268501

theorem sector_area (θ : ℝ) (r : ℝ) (hθ : θ = 2 * Real.pi / 5) (hr : r = 20) :
  1 / 2 * r^2 * θ = 80 * Real.pi := by
  sorry

end sector_area_l268_268501


namespace wine_age_problem_l268_268218

theorem wine_age_problem
  (C F T B Bo : ℕ)
  (h1 : F = 3 * C)
  (h2 : C = 4 * T)
  (h3 : B = (1 / 2 : ℝ) * T)
  (h4 : Bo = 2 * F)
  (h5 : C = 40) :
  F = 120 ∧ T = 10 ∧ B = 5 ∧ Bo = 240 := 
  by
    sorry

end wine_age_problem_l268_268218


namespace abc_le_one_ninth_sum_ratios_le_one_over_two_sqrt_abc_l268_268818

open Real

variables (a b c : ℝ)

-- Condition: a, b, c are positive numbers
variables (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
-- Condition: a^(3/2) + b^(3/2) + c^(3/2) = 1
variable  (h_sum : a^(3/2) + b^(3/2) + c^(3/2) = 1)

-- Question 1: Prove abc ≤ 1/9
theorem abc_le_one_ninth : a * b * c ≤ 1 / 9 :=
  sorry

-- Question 2: Prove (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ 1 / (2 * sqrt (a * b * c))
theorem sum_ratios_le_one_over_two_sqrt_abc : 
  (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ 1 / (2 * sqrt (a * b * c)) :=
  sorry

end abc_le_one_ninth_sum_ratios_le_one_over_two_sqrt_abc_l268_268818


namespace th150th_letter_is_B_l268_268594

def pattern := "ABCD".data

def nth_letter_in_pattern (n : ℕ) : Char :=
  let len := pattern.length
  pattern.get n % len

theorem th150th_letter_is_B :
  nth_letter_in_pattern 150 = 'B' :=
by {
  -- This proof is placed here as a placeholder
  sorry
}

end th150th_letter_is_B_l268_268594


namespace students_taking_history_but_not_statistics_l268_268604

theorem students_taking_history_but_not_statistics :
  ∀ (total_students history_students statistics_students history_or_statistics_both : ℕ),
    total_students = 90 →
    history_students = 36 →
    statistics_students = 32 →
    history_or_statistics_both = 57 →
    history_students - (history_students + statistics_students - history_or_statistics_both) = 25 :=
by intros; sorry

end students_taking_history_but_not_statistics_l268_268604


namespace sum_of_possible_b_values_l268_268241

noncomputable def g (x b : ℝ) : ℝ := x^2 - b * x + 3 * b

theorem sum_of_possible_b_values :
  (∀ (x₀ x₁ : ℝ), g x₀ x₁ = 0 → g x₀ x₁ = (x₀ - x₁) * (x₀ - 3)) → ∃ b : ℝ, b = 12 ∨ b = 16 :=
sorry

end sum_of_possible_b_values_l268_268241


namespace determine_a_l268_268143

theorem determine_a : ∀ (a b c : ℤ), 
  (∀ x : ℤ, (x - a) * (x - 5) + 1 = (x + b) * (x + c)) → (a = 3 ∨ a = 7) :=
by
  sorry

end determine_a_l268_268143


namespace sum_of_first_9_terms_45_l268_268648

-- Define the arithmetic sequence and sum of terms in the sequence
def S (n : ℕ) : ℕ := sorry  -- Placeholder for the sum of the first n terms of the sequence
def a (n : ℕ) : ℕ := sorry  -- Placeholder for the n-th term of the sequence

-- Given conditions
axiom condition1 : a 3 + a 5 + a 7 = 15

-- Proof goal
theorem sum_of_first_9_terms_45 : S 9 = 45 :=
by
  sorry

end sum_of_first_9_terms_45_l268_268648


namespace problem_statement_l268_268342

open Finset

-- Definitions
def subjects : Finset String := {"politics", "history", "geography", "physics", "chemistry", "biology", "technology"}
def remaining_subjects_after_physics : Finset String := erase subjects "physics"

-- Problem Statement 
theorem problem_statement :
  (card (choose remaining_subjects_after_physics 2) = 15) ∧ 
  (card (choose subjects 3) * card (choose subjects 3) ≠ 0) → 
  (card (choose remaining_subjects_after_physics 2) * card (choose remaining_subjects_after_physics 2) / (card (choose subjects 3) * card (choose subjects 3)) = (9 : ℝ) / 49) :=
by {
  sorry
}

end problem_statement_l268_268342


namespace exponential_comparison_l268_268673

theorem exponential_comparison (x y a b : ℝ) (hx : x > y) (hy : y > 1) (ha : 0 < a) (hb : a < b) (hb' : b < 1) : 
  a^x < b^y :=
sorry

end exponential_comparison_l268_268673


namespace system_solution_l268_268719

theorem system_solution :
  ∃ x y : ℝ, (16 * x^2 + 8 * x * y + 4 * y^2 + 20 * x + 2 * y = -7) ∧ 
            (8 * x^2 - 16 * x * y + 2 * y^2 + 20 * x - 14 * y = -11) ∧
            x = -3 / 4 ∧ y = 1 / 2 :=
by
  sorry

end system_solution_l268_268719


namespace find_omega_increasing_intervals_l268_268651

noncomputable def f (ω x : ℝ) : ℝ :=
  (Real.sin (ω * x) + Real.cos (ω * x))^2 + 2 * (Real.cos (ω * x))^2

noncomputable def g (x : ℝ) : ℝ :=
  let ω := 3/2
  f ω (x - (Real.pi / 2))

theorem find_omega (ω : ℝ) (h₀ : ω > 0) (h₁ : ∀ x : ℝ, f ω (x + 2*Real.pi / (2*ω)) = f ω x) :
  ω = 3/2 :=
  sorry

theorem increasing_intervals (k : ℤ) :
  ∃ a b, 
  a = (2/3 * k * Real.pi + Real.pi / 4) ∧ 
  b = (2/3 * k * Real.pi + 7 * Real.pi / 12) ∧
  ∀ x, a ≤ x ∧ x ≤ b → g x < g (x + 1) :=
  sorry

end find_omega_increasing_intervals_l268_268651


namespace range_of_a_l268_268021

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 2 * a^x - exp 1 * x^2

theorem range_of_a (a : ℝ) (x1 x2 : ℝ) 
  (h_min : is_local_min (f a) x1)
  (h_max : is_local_max (f a) x2)
  (h_cond1 : a > 0)
  (h_cond2 : a ≠ 1)
  (h_cond3 : x1 < x2) : 
  (1 / real.exp 1 < a) ∧ (a < 1) :=
sorry

end range_of_a_l268_268021


namespace problem1_problem2_l268_268900

open Real

variable {a b c : ℝ}

-- Condition: a, b, c are positive and a^{3/2} + b^{3/2} + c^{3/2} = 1
def conditions (a b c : ℝ) := a > 0 ∧ b > 0 ∧ c > 0 ∧ a^(3/2) + b^(3/2) + c^(3/2) = 1

-- Problem 1: Prove that abc <= 1/9
theorem problem1 (ha : conditions a b c) : a * b * c ≤ 1 / 9 :=
sorry

-- Problem 2: Prove that a/(b+c) + b/(a+c) + c/(a+b) <= 1 / (2 * sqrt(abc))
theorem problem2 (ha : conditions a b c) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * sqrt (a * b * c)) :=
sorry

end problem1_problem2_l268_268900


namespace decrease_percent_revenue_l268_268742

theorem decrease_percent_revenue (T C : ℝ) (hT : T > 0) (hC : C > 0) :
  let original_revenue := T * C
  let new_tax := 0.80 * T
  let new_consumption := 1.05 * C
  let new_revenue := new_tax * new_consumption
  let decrease_in_revenue := original_revenue - new_revenue
  let decrease_percent := (decrease_in_revenue / original_revenue) * 100
  decrease_percent = 16 := 
by
  sorry

end decrease_percent_revenue_l268_268742


namespace find_number_l268_268747

-- Define the certain number x
variable (x : ℤ)

-- Define the conditions as given in part a)
def conditions : Prop :=
  x + 10 - 2 = 44

-- State the theorem that we need to prove
theorem find_number (h : conditions x) : x = 36 :=
by sorry

end find_number_l268_268747


namespace height_of_the_carton_l268_268107

noncomputable def carton_height : ℕ :=
  let carton_length := 25
  let carton_width := 42
  let soap_box_length := 7
  let soap_box_width := 6
  let soap_box_height := 10
  let max_soap_boxes := 150
  let boxes_per_row := carton_length / soap_box_length
  let boxes_per_column := carton_width / soap_box_width
  let boxes_per_layer := boxes_per_row * boxes_per_column
  let layers := max_soap_boxes / boxes_per_layer
  layers * soap_box_height

theorem height_of_the_carton :
  carton_height = 70 :=
by
  -- The computation and necessary assumptions for proving the height are encapsulated above.
  sorry

end height_of_the_carton_l268_268107


namespace men_entered_count_l268_268346

variable (M W x : ℕ)

noncomputable def initial_ratio : Prop := M = 4 * W / 5
noncomputable def men_entered : Prop := M + x = 14
noncomputable def women_double : Prop := 2 * (W - 3) = 14

theorem men_entered_count (M W x : ℕ) (h1 : initial_ratio M W) (h2 : men_entered M x) (h3 : women_double W) : x = 6 := by
  sorry

end men_entered_count_l268_268346


namespace child_l268_268452

-- Definitions of the given conditions
def total_money : ℕ := 35
def adult_ticket_cost : ℕ := 8
def number_of_children : ℕ := 9

-- Statement of the math proof problem
theorem child's_ticket_cost : ∃ C : ℕ, total_money - adult_ticket_cost = C * number_of_children ∧ C = 3 :=
by
  sorry

end child_l268_268452


namespace necessary_but_not_sufficient_condition_l268_268607

theorem necessary_but_not_sufficient_condition (a b : ℤ) :
  (a ≠ 1 ∨ b ≠ 2) → (a + b ≠ 3) ∧ ¬((a + b ≠ 3) → (a ≠ 1 ∨ b ≠ 2)) :=
sorry

end necessary_but_not_sufficient_condition_l268_268607


namespace vasya_rank_91_l268_268224

theorem vasya_rank_91 {n_cyclists : ℕ} {n_stages : ℕ} 
    (n_cyclists_eq : n_cyclists = 500) 
    (n_stages_eq : n_stages = 15) 
    (no_ties : ∀ (i j : ℕ), i < j → ∀ (s : fin n_stages), ¬(same_time i j s)) 
    (vasya_7th : ∀ (s : fin n_stages), ∀ (i : ℕ), i < 6 → better_than i 6 s) :
    possible_rank vasya ≤ 91 :=
sorry

end vasya_rank_91_l268_268224


namespace examination_students_total_l268_268605

/-
  Problem Statement:
  Given:
  - 35% of the students passed the examination.
  - 546 students failed the examination.

  Prove:
  - The total number of students who appeared for the examination is 840.
-/

theorem examination_students_total (T : ℝ) (h1 : 0.35 * T + 0.65 * T = T) (h2 : 0.65 * T = 546) : T = 840 :=
by
  -- skipped proof part
  sorry

end examination_students_total_l268_268605


namespace part1_part2_l268_268892

variables {a b c : ℝ}

theorem part1 (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h : a^(3/2) + b^(3/2) + c^(3/2) = 1) : abc ≤ 1/9 := 
sorry

theorem part2 (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (abc)) :=
sorry

end part1_part2_l268_268892


namespace imaginary_part_of_z_l268_268650

theorem imaginary_part_of_z {z : ℂ} (h : (1 + z) / I = 1 - z) : z.im = 1 := 
sorry

end imaginary_part_of_z_l268_268650


namespace smallest_positive_b_l268_268060

def periodic_10 (f : ℝ → ℝ) : Prop :=
  ∀ x, f (x - 10) = f x

theorem smallest_positive_b
  (f : ℝ → ℝ)
  (h : periodic_10 f) :
  ∀ x, f ((x - 20) / 2) = f (x / 2) :=
by
  sorry

end smallest_positive_b_l268_268060


namespace probability_z_l268_268246

variable (p q x y z : ℝ)

-- Conditions
def condition1 : Prop := z = p * y + q * x
def condition2 : Prop := x = p + q * x^2
def condition3 : Prop := y = q + p * y^2
def condition4 : Prop := x ≠ y

-- Theorem Statement
theorem probability_z : condition1 p q x y z ∧ condition2 p q x ∧ condition3 p q y ∧ condition4 x y → z = 2 * q := by
  sorry

end probability_z_l268_268246


namespace general_equation_of_curve_l268_268655

theorem general_equation_of_curve
  (t : ℝ) (ht : t > 0)
  (x : ℝ) (hx : x = (Real.sqrt t) - (1 / (Real.sqrt t)))
  (y : ℝ) (hy : y = 3 * (t + 1 / t) + 2) :
  x^2 = (y - 8) / 3 := by
  sorry

end general_equation_of_curve_l268_268655


namespace cos_arcsin_of_fraction_l268_268000

theorem cos_arcsin_of_fraction : ∀ x, x = 8 / 17 → x ∈ set.Icc (-1:ℝ) 1 → Real.cos (Real.arcsin x) = 15 / 17 :=
by
  intros x hx h_range
  rw hx
  have h : (x:ℝ)^2 + Real.cos (Real.arcsin x)^2 = 1 := Real.sin_sq_add_cos_sq (Real.arcsin x)
  sorry

end cos_arcsin_of_fraction_l268_268000


namespace sum_a1_a11_l268_268035

theorem sum_a1_a11 
  (a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 a_10 a_11 : ℤ) 
  (h1 : a_0 = -512) 
  (h2 : -2 = a_0 + a_1 + a_2 + a_3 + a_4 + a_5 + a_6 + a_7 + a_8 + a_9 + a_10 + a_11) 
  : a_1 + a_2 + a_3 + a_4 + a_5 + a_6 + a_7 + a_8 + a_9 + a_10 + a_11 = 510 :=
sorry

end sum_a1_a11_l268_268035


namespace binomial_prime_div_l268_268548

theorem binomial_prime_div {p : ℕ} {m : ℕ} (hp : Nat.Prime p) (hm : 0 < m) : (Nat.choose (p ^ m) p - p ^ (m - 1)) % p ^ m = 0 := 
  sorry

end binomial_prime_div_l268_268548


namespace problem1_problem2_l268_268360

def M := { x : ℝ | 0 < x ∧ x < 1 }

theorem problem1 :
  { x : ℝ | |2 * x - 1| < 1 } = M :=
by
  simp [M]
  sorry

theorem problem2 (a b : ℝ) (ha : a ∈ M) (hb : b ∈ M) :
  (a * b + 1) > (a + b) :=
by
  simp [M] at ha hb
  sorry

end problem1_problem2_l268_268360


namespace first_loan_amount_l268_268163

theorem first_loan_amount :
  ∃ (L₁ L₂ : ℝ) (r : ℝ),
  (L₂ = 4700) ∧
  (L₁ = L₂ + 1500) ∧
  (0.09 * L₂ + r * L₁ = 617) ∧
  (L₁ = 6200) :=
by 
  sorry

end first_loan_amount_l268_268163


namespace find_xy_l268_268502

theorem find_xy (x y : ℝ) (h : (x^2 + 6 * x + 12) * (5 * y^2 + 2 * y + 1) = 12 / 5) : 
    x * y = 3 / 5 :=
sorry

end find_xy_l268_268502


namespace tan_sum_series_inv_tan_l268_268795

theorem tan_sum_series_inv_tan (n : ℕ) (h : n = 2009) : 
  tan (∑ k in Finset.range n, arctan (1 / (2 * (k+1)^2))) = 2009 / 2010 := by
  sorry

end tan_sum_series_inv_tan_l268_268795


namespace length_of_square_cut_off_l268_268273

theorem length_of_square_cut_off 
  (x : ℝ) 
  (h_eq : (48 - 2 * x) * (36 - 2 * x) * x = 5120) : 
  x = 8 := 
sorry

end length_of_square_cut_off_l268_268273


namespace cube_minus_self_divisible_by_6_l268_268210

theorem cube_minus_self_divisible_by_6 (n : ℕ) : 6 ∣ (n^3 - n) :=
sorry

end cube_minus_self_divisible_by_6_l268_268210


namespace proof_part1_proof_part2_l268_268853

theorem proof_part1 (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) 
                    (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
                    (a * b * c ≤ 1 / 9) := 
by
  sorry

theorem proof_part2 (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) 
                    (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
                    (a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * (a * b * c)^(1/2))) := 
by
  sorry

end proof_part1_proof_part2_l268_268853


namespace problem_part1_problem_part2_l268_268946

variable {a b c : ℝ}

theorem problem_part1 (h : a > 0 ∧ b > 0 ∧ c > 0) (sum_eq : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  abc ≤ 1 / 9 :=
sorry

theorem problem_part2 (h : a > 0 ∧ b > 0 ∧ c > 0) (sum_eq : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (abc)) :=
sorry

end problem_part1_problem_part2_l268_268946


namespace correct_multiplication_result_l268_268087

theorem correct_multiplication_result (x : ℕ) (h : 9 * x = 153) : 6 * x = 102 :=
by {
  -- We would normally provide a detailed proof here, but as per instruction, we add sorry.
  sorry
}

end correct_multiplication_result_l268_268087


namespace min_value_f_a_neg3_max_value_g_ge_7_l268_268805

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := (Real.exp x) * (x^2 + a * x + 1)

noncomputable def g (x : ℝ) (b : ℝ) : ℝ := 2 * x^3 + 3 * (b + 1) * x^2 + 6 * b * x + 6

theorem min_value_f_a_neg3 (h : -3 ≤ -1) : 
  (∀ x : ℝ, f x (-3) ≥ -Real.exp 2) := 
sorry

theorem max_value_g_ge_7 (a : ℝ) (h : a ≤ -1) (b : ℝ) (h_b : b = a + 1) :
  ∃ m : ℝ, (∀ x : ℝ, g x b ≤ m) ∧ (m ≥ 7) := 
sorry

end min_value_f_a_neg3_max_value_g_ge_7_l268_268805


namespace part1_part2_l268_268865

variable (a b c : ℝ)

theorem part1 (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) : abc ≤ 1/9 :=
by
  sorry

theorem part2 (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
    (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ 1 / (2 * sqrt(abc)) :=
by
  sorry

end part1_part2_l268_268865


namespace cosine_of_arcsine_l268_268001

theorem cosine_of_arcsine (h : -1 ≤ (8 : ℝ) / 17 ∧ (8 : ℝ) / 17 ≤ 1) : 
  Real.cos (Real.arcsin (8 / 17)) = 15 / 17 :=
sorry

end cosine_of_arcsine_l268_268001


namespace ratio_e_f_l268_268029

theorem ratio_e_f (a b c d e f : ℚ)
  (h1 : a / b = 1 / 3)
  (h2 : b / c = 2)
  (h3 : c / d = 1 / 2)
  (h4 : d / e = 3)
  (h5 : a * b * c / (d * e * f) = 0.25) :
  e / f = 9 / 4 :=
sorry

end ratio_e_f_l268_268029


namespace floor_x_floor_x_eq_42_l268_268147

theorem floor_x_floor_x_eq_42 (x : ℝ) : (⌊x * ⌊x⌋⌋ = 42) ↔ (7 ≤ x ∧ x < 43 / 6) :=
by sorry

end floor_x_floor_x_eq_42_l268_268147


namespace number_of_tulips_l268_268348

theorem number_of_tulips (T : ℕ) (roses : ℕ) (used_flowers : ℕ) (extra_flowers : ℕ) 
  (total_flowers : ℕ) (h1 : roses = 37) (h2 : used_flowers = 70) 
  (h3 : extra_flowers = 3) (h4: total_flowers = 73) 
  (h5 : T + roses = total_flowers) : T = 36 := 
by
  sorry

end number_of_tulips_l268_268348


namespace problem_1_problem_2_l268_268881

theorem problem_1 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) 
  (h_sum : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  abc ≤ 1 / 9 := 
sorry

theorem problem_2 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) 
  (h_sum : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * real.sqrt (abc)) := 
sorry

end problem_1_problem_2_l268_268881


namespace find_T_l268_268587

variable (a b c T : ℕ)

theorem find_T (h1 : a + b + c = 84) (h2 : a - 5 = T) (h3 : b + 9 = T) (h4 : 5 * c = T) : T = 40 :=
sorry

end find_T_l268_268587


namespace solution_set_inequality_l268_268586

theorem solution_set_inequality (x : ℝ) : (x-3) * (x-1) > 0 → (x < 1 ∨ x > 3) :=
by sorry

end solution_set_inequality_l268_268586


namespace mason_water_intake_l268_268725

theorem mason_water_intake
  (Theo_Daily : ℕ := 8)
  (Roxy_Daily : ℕ := 9)
  (Total_Weekly : ℕ := 168)
  (Days_Per_Week : ℕ := 7) :
  (∃ M : ℕ, M * Days_Per_Week = Total_Weekly - (Theo_Daily + Roxy_Daily) * Days_Per_Week ∧ M = 7) :=
  by
  sorry

end mason_water_intake_l268_268725


namespace probability_blue_tile_l268_268610

def is_congruent_to_3_mod_7 (n : ℕ) : Prop := n % 7 = 3

def num_blue_tiles (n : ℕ) : ℕ := (n / 7) + 1

theorem probability_blue_tile : 
  num_blue_tiles 70 / 70 = 1 / 7 :=
by
  sorry

end probability_blue_tile_l268_268610


namespace problem_1_problem_2_l268_268825

variable (a b c : ℝ)

theorem problem_1 
  (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  a * b * c ≤ 1 / 9 :=
sorry

theorem problem_2 
  (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * (a * b * c)^(1/2)) :=
sorry

end problem_1_problem_2_l268_268825


namespace yogurt_combinations_l268_268449

theorem yogurt_combinations (f : ℕ) (t : ℕ) (h_f : f = 4) (h_t : t = 6) :
  (f * (t.choose 2) = 60) :=
by
  rw [h_f, h_t]
  sorry

end yogurt_combinations_l268_268449


namespace largest_packet_size_gcd_l268_268694

theorem largest_packet_size_gcd:
    ∀ (n1 n2 : ℕ), n1 = 36 → n2 = 60 → Nat.gcd n1 n2 = 12 :=
by
  intros n1 n2 h1 h2
  -- Sorry is added because the proof is not required as per the instructions
  sorry

end largest_packet_size_gcd_l268_268694


namespace radius_increase_l268_268085

/-- Proving that the radius increases by 7/π inches when the circumference increases from 50 inches to 64 inches -/
theorem radius_increase (C₁ C₂ : ℝ) (h₁ : C₁ = 50) (h₂ : C₂ = 64) :
  (C₂ / (2 * Real.pi) - C₁ / (2 * Real.pi)) = 7 / Real.pi :=
by
  sorry

end radius_increase_l268_268085


namespace infinite_k_values_l268_268309

theorem infinite_k_values (k : ℕ) : (∃ k, ∀ (a b c : ℕ),
  (a = 64 ∧ b ≥ 0 ∧ c = 0 ∧ k = 2^a * 3^b * 5^c) ↔
  Nat.lcm (Nat.lcm (2^8) (2^24 * 3^12)) k = 2^64) →
  ∃ (b : ℕ), true :=
by
  sorry

end infinite_k_values_l268_268309


namespace total_slices_l268_268249

theorem total_slices {slices_per_pizza pizzas : ℕ} (h1 : slices_per_pizza = 2) (h2 : pizzas = 14) : 
  slices_per_pizza * pizzas = 28 :=
by
  -- This is where the proof would go, but we are omitting it as instructed.
  sorry

end total_slices_l268_268249


namespace maximum_value_at_vertex_l268_268328

-- Defining the parabola as a function
def parabola (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- Defining the vertex condition
def vertex_condition (a b c : ℝ) := ∀ x : ℝ, parabola a b c x = a * x^2 + b * x + c

-- Defining the condition that the parabola opens downward
def opens_downward (a : ℝ) := a < 0

-- Defining the vertex coordinates condition
def vertex_coordinates (a b c : ℝ) := 
  ∃ (x₀ y₀ : ℝ), x₀ = 2 ∧ y₀ = -3 ∧ parabola a b c x₀ = y₀

-- The main theorem statement
theorem maximum_value_at_vertex (a b c : ℝ) (h1 : opens_downward a) (h2 : vertex_coordinates a b c) : ∃ y₀, y₀ = -3 ∧ ∀ x : ℝ, parabola a b c x ≤ y₀ :=
by
  sorry

end maximum_value_at_vertex_l268_268328


namespace cricket_team_right_handed_players_l268_268094

theorem cricket_team_right_handed_players
  (total_players : ℕ)
  (throwers : ℕ)
  (non_throwers : ℕ := total_players - throwers)
  (left_handed_non_throwers : ℕ := non_throwers / 3)
  (right_handed_throwers : ℕ := throwers)
  (right_handed_non_throwers : ℕ := non_throwers - left_handed_non_throwers)
  (total_right_handed : ℕ := right_handed_throwers + right_handed_non_throwers)
  (h1 : total_players = 70)
  (h2 : throwers = 37)
  (h3 : left_handed_non_throwers = non_throwers / 3) :
  total_right_handed = 59 :=
by
  rw [h1, h2] at *
  -- The remaining parts of the proof here are omitted for brevity.
  sorry

end cricket_team_right_handed_players_l268_268094


namespace intersection_A_B_l268_268658

open Set

noncomputable def A : Set ℤ := {-1, 0, 1, 2, 3, 4, 5}

noncomputable def B : Set ℤ := {b | ∃ n : ℤ, b = n^2 - 1}

theorem intersection_A_B :
  A ∩ B = {-1, 0, 3} :=
by {
  sorry
}

end intersection_A_B_l268_268658


namespace part1_part2_l268_268867

variable (a b c : ℝ)

theorem part1 (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) : abc ≤ 1/9 :=
by
  sorry

theorem part2 (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
    (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ 1 / (2 * sqrt(abc)) :=
by
  sorry

end part1_part2_l268_268867


namespace share_of_a_l268_268266

def shares_sum (a b c : ℝ) := a + b + c = 366
def share_a (a b c : ℝ) := a = 1/2 * (b + c)
def share_b (a b c : ℝ) := b = 2/3 * (a + c)

theorem share_of_a (a b c : ℝ) 
  (h1 : shares_sum a b c) 
  (h2 : share_a a b c) 
  (h3 : share_b a b c) : 
  a = 122 := 
by 
  -- Proof goes here
  sorry

end share_of_a_l268_268266


namespace weekly_deficit_is_2800_l268_268050

def daily_intake (day : String) : ℕ :=
  if day = "Monday" then 2500 else 
  if day = "Tuesday" then 2600 else 
  if day = "Wednesday" then 2400 else 
  if day = "Thursday" then 2700 else 
  if day = "Friday" then 2300 else 
  if day = "Saturday" then 3500 else 
  if day = "Sunday" then 2400 else 0

def daily_expenditure (day : String) : ℕ :=
  if day = "Monday" then 3000 else 
  if day = "Tuesday" then 3200 else 
  if day = "Wednesday" then 2900 else 
  if day = "Thursday" then 3100 else 
  if day = "Friday" then 2800 else 
  if day = "Saturday" then 3000 else 
  if day = "Sunday" then 2700 else 0

def daily_deficit (day : String) : ℤ :=
  daily_expenditure day - daily_intake day

def weekly_caloric_deficit : ℤ :=
  daily_deficit "Monday" +
  daily_deficit "Tuesday" +
  daily_deficit "Wednesday" +
  daily_deficit "Thursday" +
  daily_deficit "Friday" +
  daily_deficit "Saturday" +
  daily_deficit "Sunday"

theorem weekly_deficit_is_2800 : weekly_caloric_deficit = 2800 := by
  sorry

end weekly_deficit_is_2800_l268_268050


namespace find_input_values_f_l268_268182

theorem find_input_values_f (f : ℤ → ℤ) 
  (h_def : ∀ x, f (2 * x + 3) = (x - 3) * (x + 4))
  (h_val : ∃ y, f y = 170) : 
  ∃ (a b : ℤ), (a = -25 ∧ b = 29) ∧ (f a = 170 ∧ f b = 170) :=
by
  sorry

end find_input_values_f_l268_268182


namespace part1_part2_l268_268982

noncomputable theory

variables {a b c : ℝ}

-- Condition that a, b, c are positive numbers and the given equation holds
def pos_numbers : Prop := a > 0 ∧ b > 0 ∧ c > 0
def sum_condition : Prop := (a ^ (3/2)) + (b ^ (3/2)) + (c ^ (3/2)) = 1

-- Proof goal for the first part: abc <= 1/9
theorem part1 (h₁ : pos_numbers) (h₂ : sum_condition) : 
  a * b * c ≤ 1 / 9 :=
sorry

-- Proof goal for the second part: a/(b+c) + b/(a+c) + c/(a+b) <= 1/(2 * sqrt(abc))
theorem part2 (h₁ : pos_numbers) (h₂ : sum_condition) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * sqrt (a * b * c)) :=
sorry

end part1_part2_l268_268982


namespace rabbit_speed_l268_268174

theorem rabbit_speed (s : ℕ) (h : (s * 2 + 4) * 2 = 188) : s = 45 :=
sorry

end rabbit_speed_l268_268174


namespace tammy_earnings_after_3_weeks_l268_268554

noncomputable def oranges_picked_per_day (num_trees : ℕ) (oranges_per_tree : ℕ) : ℕ :=
  num_trees * oranges_per_tree

noncomputable def packs_sold_per_day (oranges_per_day : ℕ) (oranges_per_pack : ℕ) : ℕ :=
  oranges_per_day / oranges_per_pack

noncomputable def total_packs_sold_in_weeks (packs_per_day : ℕ) (days_in_week : ℕ) (num_weeks : ℕ) : ℕ :=
  packs_per_day * days_in_week * num_weeks

noncomputable def money_earned (total_packs : ℕ) (price_per_pack : ℕ) : ℕ :=
  total_packs * price_per_pack

theorem tammy_earnings_after_3_weeks :
  let num_trees := 10
  let oranges_per_tree := 12
  let oranges_per_pack := 6
  let price_per_pack := 2
  let days_in_week := 7
  let num_weeks := 3
  oranges_picked_per_day num_trees oranges_per_tree /
  oranges_per_pack *
  days_in_week *
  num_weeks *
  price_per_pack = 840 :=
by {
  sorry
}

end tammy_earnings_after_3_weeks_l268_268554


namespace proof_part1_proof_part2_l268_268849

theorem proof_part1 (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) 
                    (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
                    (a * b * c ≤ 1 / 9) := 
by
  sorry

theorem proof_part2 (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) 
                    (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
                    (a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * (a * b * c)^(1/2))) := 
by
  sorry

end proof_part1_proof_part2_l268_268849


namespace number_of_intersections_l268_268069

noncomputable def y1 (x: ℝ) : ℝ := (x - 1) ^ 4
noncomputable def y2 (x: ℝ) : ℝ := 2 ^ (abs x) - 2

theorem number_of_intersections : (∃ x₁ x₂ x₃ x₄ : ℝ, y1 x₁ = y2 x₁ ∧ y1 x₂ = y2 x₂ ∧ y1 x₃ = y2 x₃ ∧ y1 x₄ = y2 x₄ ∧ x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₃ ≠ x₄) :=
sorry

end number_of_intersections_l268_268069


namespace total_salmon_now_l268_268456

def initial_salmon : ℕ := 500

def increase_factor : ℕ := 10

theorem total_salmon_now : initial_salmon * increase_factor = 5000 := by
  sorry

end total_salmon_now_l268_268456


namespace slices_per_pie_l268_268756

variable (S : ℕ) -- Let S be the number of slices per pie

theorem slices_per_pie (h1 : 5 * S * 9 = 180) : S = 4 := by
  sorry

end slices_per_pie_l268_268756


namespace local_value_of_7_in_diff_l268_268596

-- Definitions based on conditions
def local_value (n : ℕ) (d : ℕ) : ℕ :=
  if h : d < 10 ∧ (n / Nat.pow 10 (Nat.log 10 n - Nat.log 10 d)) % 10 = d then
    d * Nat.pow 10 (Nat.log 10 n - Nat.log 10 d)
  else
    0

def diff (a b : ℕ) : ℕ := a - b

-- Question translated to Lean 4 statement
theorem local_value_of_7_in_diff :
  local_value (diff 100889 (local_value 28943712 3)) 7 = 70000 :=
by sorry

end local_value_of_7_in_diff_l268_268596


namespace proof_part1_proof_part2_l268_268915

variable {a b c : ℝ}

def pos_and_sum_of_powers (a b c : ℝ) : Prop :=
  0 < a ∧ 0 < b ∧ 0 < c ∧ a^(3/2) + b^(3/2) + c^(3/2) = 1

theorem proof_part1 (h : pos_and_sum_of_powers a b c) : abc ≤ 1 / 9 :=
sorry

theorem proof_part2 (h : pos_and_sum_of_powers a b c) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (abc)) :=
sorry

end proof_part1_proof_part2_l268_268915


namespace mean_identity_l268_268219

theorem mean_identity (x y z : ℝ) 
  (h_arith_mean : (x + y + z) / 3 = 10)
  (h_geom_mean : Real.cbrt (x * y * z) = 7) 
  (h_harm_mean : 3 / (1 / x + 1 / y + 1 / z) = 4) :
  x^2 + y^2 + z^2 = 385.5 :=
by
  sorry

end mean_identity_l268_268219


namespace zoo_problem_l268_268122

variables
  (parrots : ℕ)
  (snakes : ℕ)
  (monkeys : ℕ)
  (elephants : ℕ)
  (zebras : ℕ)
  (f : ℚ)

-- Conditions from the problem
theorem zoo_problem
  (h1 : parrots = 8)
  (h2 : snakes = 3 * parrots)
  (h3 : monkeys = 2 * snakes)
  (h4 : elephants = f * (parrots + snakes))
  (h5 : zebras = elephants - 3)
  (h6 : monkeys - zebras = 35) :
  f = 1 / 2 :=
sorry

end zoo_problem_l268_268122


namespace r_has_money_l268_268741

-- Define the variables and the conditions in Lean
variable (p q r : ℝ)
variable (h1 : p + q + r = 4000)
variable (h2 : r = (2/3) * (p + q))

-- Define the proof statement
theorem r_has_money : r = 1600 := 
  by
    sorry

end r_has_money_l268_268741


namespace min_n_such_that_no_more_possible_l268_268799

-- Define a seven-cell corner as a specific structure within the grid
inductive Corner
| cell7 : Corner

-- Function to count the number of cells clipped out by n corners
def clipped_cells (n : ℕ) : ℕ := 7 * n

-- Statement to be proven
theorem min_n_such_that_no_more_possible (n : ℕ) (h_n : n ≥ 3) (h_max : n < 4) :
  ¬ ∃ k : ℕ, k > n ∧ clipped_cells k ≤ 64 :=
by {
  sorry -- Proof goes here
}

end min_n_such_that_no_more_possible_l268_268799


namespace range_of_a_l268_268222

noncomputable def f (a x : ℝ) : ℝ :=
  Real.exp (x-2) + (1/3) * x^3 - (3/2) * x^2 + 2 * x - Real.log (x-1) + a

theorem range_of_a (a : ℝ) :
  (∀ y : ℝ, ∃ x : ℝ, (1 < x → f a x = y) ↔ ∃ z : ℝ, 1 < z → f a (f a z) = y) →
  a ≤ 1/3 :=
sorry

end range_of_a_l268_268222


namespace range_of_a_l268_268020

open Real

noncomputable def f (a e x : ℝ) : ℝ := 2 * a^x - e * x^2

theorem range_of_a (a e x₁ x₂ : ℝ) (h_a_pos : 0 < a) (h_a_ne_one : a ≠ 1) (h_x₁ : f a e x₁ = f a e x + f a e 1) (h_min : deriv (f a e) x₁ = 0) (h_max : deriv (f a e) x₂ = 0) (h_x₁_lt_x₂ : x₁ < x₂) :
  1 / exp 1 < a ∧ a < 1 :=
sorry

end range_of_a_l268_268020


namespace mortgage_payoff_months_l268_268188

-- Declare the initial payment (P), the common ratio (r), and the total amount (S)
def initial_payment : ℕ := 100
def common_ratio : ℕ := 3
def total_amount : ℕ := 12100

-- Define a function that calculates the sum of a geometric series
noncomputable def geom_series_sum (P : ℕ) (r : ℕ) (n : ℕ) : ℕ :=
  P * (1 - r ^ n) / (1 - r)

-- The statement we need to prove
theorem mortgage_payoff_months : ∃ n : ℕ, geom_series_sum initial_payment common_ratio n = total_amount :=
by
  sorry -- Proof to be provided

end mortgage_payoff_months_l268_268188


namespace ratio_of_girls_to_boys_l268_268681

theorem ratio_of_girls_to_boys (total_students girls boys : ℕ) (h_ratio : girls = 4 * (girls + boys) / 7) (h_total : total_students = 70) : 
  girls = 40 ∧ boys = 30 :=
by
  sorry

end ratio_of_girls_to_boys_l268_268681


namespace largest_number_among_options_l268_268460

theorem largest_number_among_options :
  let A := 0.983
  let B := 0.9829
  let C := 0.9831
  let D := 0.972
  let E := 0.9819
  C > A ∧ C > B ∧ C > D ∧ C > E :=
by
  sorry

end largest_number_among_options_l268_268460


namespace extreme_values_max_min_on_interval_coordinates_midpoint_parallel_tangents_l268_268504

-- Given function
def f (x : ℝ) : ℝ := x^3 - 12 * x + 12

-- Definition of derivative
def f' (x : ℝ) : ℝ := (3 : ℝ) * x^2 - (12 : ℝ)

-- Part 1: Extreme values
theorem extreme_values : 
  (f (-2) = 28) ∧ (f 2 = -4) :=
by
  sorry

-- Part 2: Maximum and minimum values on the interval [-3, 4]
theorem max_min_on_interval :
  (∀ x, -3 ≤ x ∧ x ≤ 4 → f x ≤ 28) ∧ (∀ x, -3 ≤ x ∧ x ≤ 4 → f x ≥ -4) :=
by
  sorry

-- Part 3: Coordinates of midpoint A and B with parallel tangents
theorem coordinates_midpoint_parallel_tangents :
  (f' x1 = f' x2 ∧ x1 + x2 = 0) → ((x1 + x2) / 2 = 0 ∧ (f x1 + f x2) / 2 = 12) :=
by
  sorry

end extreme_values_max_min_on_interval_coordinates_midpoint_parallel_tangents_l268_268504


namespace problem_part1_problem_part2_l268_268951

variable {a b c : ℝ}

theorem problem_part1 (h : a > 0 ∧ b > 0 ∧ c > 0) (sum_eq : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  abc ≤ 1 / 9 :=
sorry

theorem problem_part2 (h : a > 0 ∧ b > 0 ∧ c > 0) (sum_eq : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (abc)) :=
sorry

end problem_part1_problem_part2_l268_268951


namespace longer_diagonal_of_rhombus_l268_268440

theorem longer_diagonal_of_rhombus {a b d1 : ℕ} (h1 : a = b) (h2 : a = 65) (h3 : d1 = 60) : 
  ∃ d2, (d2^2) = (2 * (a^2) - (d1^2)) ∧ d2 = 110 :=
by
  sorry

end longer_diagonal_of_rhombus_l268_268440


namespace problem1_problem2_l268_268897

open Real

variable {a b c : ℝ}

-- Condition: a, b, c are positive and a^{3/2} + b^{3/2} + c^{3/2} = 1
def conditions (a b c : ℝ) := a > 0 ∧ b > 0 ∧ c > 0 ∧ a^(3/2) + b^(3/2) + c^(3/2) = 1

-- Problem 1: Prove that abc <= 1/9
theorem problem1 (ha : conditions a b c) : a * b * c ≤ 1 / 9 :=
sorry

-- Problem 2: Prove that a/(b+c) + b/(a+c) + c/(a+b) <= 1 / (2 * sqrt(abc))
theorem problem2 (ha : conditions a b c) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * sqrt (a * b * c)) :=
sorry

end problem1_problem2_l268_268897


namespace find_cosine_of_dihedral_angle_l268_268413

def dihedral_cosine (R r : ℝ) (α β : ℝ) : Prop :=
  R = 2 * r ∧ β = Real.pi / 4 → Real.cos α = 8 / 9

theorem find_cosine_of_dihedral_angle : ∃ α, ∀ R r : ℝ, dihedral_cosine R r α (Real.pi / 4) :=
sorry

end find_cosine_of_dihedral_angle_l268_268413


namespace julian_younger_than_frederick_by_20_l268_268704

noncomputable def Kyle: ℕ := 25
noncomputable def Tyson: ℕ := 20
noncomputable def Julian : ℕ := Kyle - 5
noncomputable def Frederick : ℕ := 2 * Tyson

theorem julian_younger_than_frederick_by_20 : Frederick - Julian = 20 :=
by
  sorry

end julian_younger_than_frederick_by_20_l268_268704


namespace profit_per_meter_l268_268768

theorem profit_per_meter
  (total_meters : ℕ)
  (selling_price : ℕ)
  (cost_price_per_meter : ℕ)
  (total_cost_price : ℕ := cost_price_per_meter * total_meters)
  (total_profit : ℕ := selling_price - total_cost_price)
  (profit_per_meter : ℕ := total_profit / total_meters) :
  total_meters = 75 ∧ selling_price = 4950 ∧ cost_price_per_meter = 51 → profit_per_meter = 15 :=
by
  intros h
  sorry

end profit_per_meter_l268_268768


namespace range_of_a_l268_268019

noncomputable section

open Real

def f (a x : ℝ) : ℝ := 2 * a ^ x - exp 1 * x ^ 2

def f' (a x : ℝ) : ℝ := 2 * a ^ x * log a - 2 * exp 1 * x

def f'' (a x : ℝ) : ℝ := 2 * a ^ x * (log a) ^ 2 - 2 * exp 1

theorem range_of_a (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (x1 x2 : ℝ)
  (hx1 : f' a x1 = 0) (hx2 : f' a x2 = 0) (hx1_min : f'' a x1 > 0)
  (hx2_max : f'' a x2 < 0) (h : x1 < x2) : a ∈ Ioo (1 / exp 1) 1 :=
sorry

end range_of_a_l268_268019


namespace M_inter_N_l268_268097

def M : Set ℝ := {y | ∃ x : ℝ, y = 2^(-x)}
def N : Set ℝ := {y | ∃ x : ℝ, y = Real.sin x}

theorem M_inter_N : M ∩ N = {y | 0 < y ∧ y ≤ 1} :=
by
  sorry

end M_inter_N_l268_268097


namespace problem_1_problem_2_l268_268922

variable (a b c : ℝ)

theorem problem_1 
  (hp : 0 < a) 
  (hq : 0 < b) 
  (hr : 0 < c) 
  (hs : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  abc ≤ 1/9 := 
sorry

theorem problem_2 
  (hp : 0 < a) 
  (hq : 0 < b) 
  (hr : 0 < c) 
  (hs : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (abc)) := 
sorry

end problem_1_problem_2_l268_268922


namespace min_value_a_plus_2b_minus_3c_l268_268530

theorem min_value_a_plus_2b_minus_3c
  (a b c : ℝ)
  (h : ∀ (x y : ℝ), x + 2 * y - 3 ≤ a * x + b * y + c ∧ a * x + b * y + c ≤ x + 2 * y + 3) :
  ∃ m : ℝ, m = a + 2 * b - 3 * c ∧ m = -4 :=
by
  sorry

end min_value_a_plus_2b_minus_3c_l268_268530


namespace problem_lean_l268_268335

theorem problem_lean (a : ℝ) (h : a - 1/a = 5) : a^2 + 1/a^2 = 27 := by
  sorry

end problem_lean_l268_268335


namespace circle_equation_exists_l268_268030

theorem circle_equation_exists :
  ∃ (x_c y_c r : ℝ), 
  x_c > 0 ∧ y_c > 0 ∧ 0 < r ∧ r < 5 ∧ (∀ x y : ℝ, (x - x_c)^2 + (y - y_c)^2 = r^2) :=
sorry

end circle_equation_exists_l268_268030


namespace cyclist_speed_l268_268272

theorem cyclist_speed:
  ∀ (c : ℝ), 
  ∀ (hiker_speed : ℝ), 
  (hiker_speed = 4) → 
  (4 * (5 / 60) + 4 * (25 / 60) = c * (5 / 60)) → 
  c = 24 := 
by
  intros c hiker_speed hiker_speed_def distance_eq
  sorry

end cyclist_speed_l268_268272


namespace percent_of_a_l268_268215

theorem percent_of_a (a b : ℝ) (h : a = 1.2 * b) : 4 * b = (10 / 3) * a :=
sorry

end percent_of_a_l268_268215


namespace problem1_problem2_l268_268899

open Real

variable {a b c : ℝ}

-- Condition: a, b, c are positive and a^{3/2} + b^{3/2} + c^{3/2} = 1
def conditions (a b c : ℝ) := a > 0 ∧ b > 0 ∧ c > 0 ∧ a^(3/2) + b^(3/2) + c^(3/2) = 1

-- Problem 1: Prove that abc <= 1/9
theorem problem1 (ha : conditions a b c) : a * b * c ≤ 1 / 9 :=
sorry

-- Problem 2: Prove that a/(b+c) + b/(a+c) + c/(a+b) <= 1 / (2 * sqrt(abc))
theorem problem2 (ha : conditions a b c) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * sqrt (a * b * c)) :=
sorry

end problem1_problem2_l268_268899


namespace least_positive_multiple_of_24_gt_450_l268_268258

theorem least_positive_multiple_of_24_gt_450 : 
  ∃ n : ℕ, n > 450 ∧ (∃ k : ℕ, n = 24 * k) → n = 456 :=
by 
  sorry

end least_positive_multiple_of_24_gt_450_l268_268258


namespace pattern_expression_equality_l268_268368

theorem pattern_expression_equality (n : ℕ) : ((n - 1) * (n + 1)) + 1 = n^2 :=
  sorry

end pattern_expression_equality_l268_268368


namespace fill_tank_with_leak_l268_268615

theorem fill_tank_with_leak (R L T: ℝ)
(h1: R = 1 / 7) (h2: L = 1 / 56) (h3: R - L = 1 / T) : T = 8 := by
  sorry

end fill_tank_with_leak_l268_268615


namespace largest_four_digit_congruent_17_mod_26_l268_268250

theorem largest_four_digit_congruent_17_mod_26 : 
  ∃ k : ℤ, (26 * k + 17 < 10000) ∧ (1000 ≤ 26 * k + 17) ∧ (26 * k + 17) ≡ 17 [MOD 26] ∧ (26 * k + 17 = 9972) :=
by
  sorry

end largest_four_digit_congruent_17_mod_26_l268_268250


namespace avg_price_of_returned_tshirts_l268_268383

-- Define the conditions as Lean definitions
def avg_price_50_tshirts := 750
def num_tshirts := 50
def num_returned_tshirts := 7
def avg_price_remaining_43_tshirts := 720

-- The correct price of the 7 returned T-shirts
def correct_avg_price_returned := 6540 / 7

-- The proof statement
theorem avg_price_of_returned_tshirts :
  (num_tshirts * avg_price_50_tshirts - (num_tshirts - num_returned_tshirts) * avg_price_remaining_43_tshirts) / num_returned_tshirts = correct_avg_price_returned :=
by
  sorry

end avg_price_of_returned_tshirts_l268_268383


namespace smallest_prime_dividing_sum_l268_268415

theorem smallest_prime_dividing_sum :
  ∃ p : ℕ, Prime p ∧ p ∣ (7^14 + 11^15) ∧ ∀ q : ℕ, Prime q ∧ q ∣ (7^14 + 11^15) → p ≤ q := by
  sorry

end smallest_prime_dividing_sum_l268_268415


namespace statement_c_correct_l268_268471

theorem statement_c_correct (a b c : ℝ) (h : a * c^2 > b * c^2) : a > b :=
by sorry

end statement_c_correct_l268_268471


namespace cos_double_angle_l268_268474

variable {α β : Real}

-- Definitions from the conditions
def sin_diff_condition : Prop := sin (α - β) = 1 / 3
def cos_sin_condition : Prop := cos α * sin β = 1 / 6

-- The main theorem 
theorem cos_double_angle (h₁ : sin_diff_condition) (h₂ : cos_sin_condition) : cos (2 * α + 2 * β) = 1 / 9 :=
by sorry

end cos_double_angle_l268_268474


namespace h_h_3_eq_2915_l268_268052

def h (x : ℕ) : ℕ := 3 * x^2 + x + 1

theorem h_h_3_eq_2915 : h (h 3) = 2915 := by
  sorry

end h_h_3_eq_2915_l268_268052


namespace inequality_solution_l268_268793

theorem inequality_solution (x : ℝ) :
  (7 / 36 + (abs (2 * x - (1 / 6)))^2 < 5 / 12) ↔
  (x ∈ Set.Ioo ((1 / 12 - (Real.sqrt 2 / 6))) ((1 / 12 + (Real.sqrt 2 / 6)))) :=
by
  sorry

end inequality_solution_l268_268793


namespace different_lists_count_l268_268774

def numberOfLists : Nat := 5

theorem different_lists_count :
  let conditions := ∃ (d : Fin 6 → ℕ), d 0 + d 1 + d 2 + d 3 + d 4 + d 5 = 5 ∧
                                      ∀ i, d i ≤ 5 ∧
                                      ∀ i j, i < j → d i ≥ d j
  conditions →
  numberOfLists = 5 :=
sorry

end different_lists_count_l268_268774


namespace converse_equiv_l268_268559

/-
Original proposition: If ¬p then ¬q
Converse of the original proposition: If ¬q then ¬p
Equivalent proposition to the converse proposition: If p then q

We need to prove that (If ¬p then ¬q) implies (If q then p)
-/
theorem converse_equiv (p q : Prop) : (¬p → ¬q) → (p → q) :=
sorry

end converse_equiv_l268_268559


namespace percentage_problem_l268_268740

theorem percentage_problem (x : ℝ) (h : 0.20 * x = 100) : 1.20 * x = 600 :=
sorry

end percentage_problem_l268_268740


namespace cannot_determine_orange_groups_l268_268589

-- Definitions of the conditions
def oranges := 87
def bananas := 290
def bananaGroups := 2
def bananasPerGroup := 145

-- Lean statement asserting that the number of groups of oranges 
-- cannot be determined from the given conditions
theorem cannot_determine_orange_groups:
  ∀ (number_of_oranges_per_group : ℕ), 
  (bananasPerGroup * bananaGroups = bananas) ∧ (oranges = 87) → 
  ¬(∃ (number_of_orange_groups : ℕ), oranges = number_of_oranges_per_group * number_of_orange_groups) :=
by
  sorry -- Since we are not required to provide the proof here

end cannot_determine_orange_groups_l268_268589


namespace vasya_maximum_rank_l268_268236

theorem vasya_maximum_rank {n : ℕ} (cyclists stages : ℕ) (VasyaPlace : ℕ) 
  (rankings : Π (s : fin stages), fin cyclists):
  cyclists = 500 → stages = 15 → VasyaPlace = 7 →
  (∀ s, rankings s VasyaPlace = 7) →
  (∀ s t i, rankings s i ≠ rankings t i) →
  (∀ s, ∃ l, list.nodup l ∧ (∀ i j, i < j → rankings s i < rankings s j) ∧ list.length l = 500) →
  (∃ (maxRank : ℕ), maxRank = 91) :=
by
  intros hcyclists hstages hplace hvasya_place hdistinct_rankings hstage_rankings
  use 91
  sorry

end vasya_maximum_rank_l268_268236


namespace problem_1_problem_2_l268_268929

variable (a b c : ℝ)

theorem problem_1 
  (hp : 0 < a) 
  (hq : 0 < b) 
  (hr : 0 < c) 
  (hs : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  abc ≤ 1/9 := 
sorry

theorem problem_2 
  (hp : 0 < a) 
  (hq : 0 < b) 
  (hr : 0 < c) 
  (hs : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (abc)) := 
sorry

end problem_1_problem_2_l268_268929


namespace fisher_catch_l268_268244

theorem fisher_catch (x y : ℕ) (h1 : x + y = 80)
  (h2 : ∃ a : ℕ, x = 9 * a)
  (h3 : ∃ b : ℕ, y = 11 * b) :
  x = 36 ∧ y = 44 :=
by
  sorry

end fisher_catch_l268_268244


namespace range_of_a_l268_268810

def p (a : ℝ) : Prop := ∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → x^2 - a ≥ 0
def q (a : ℝ) : Prop := ∃ x : ℝ, x^2 + 2 * a * x + 2 - a = 0

theorem range_of_a (a : ℝ) (hp : p a) (hq : q a) : a ≤ -2 ∨ a = 1 := 
by sorry

end range_of_a_l268_268810


namespace problem_1_problem_2_l268_268880

theorem problem_1 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) 
  (h_sum : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  abc ≤ 1 / 9 := 
sorry

theorem problem_2 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) 
  (h_sum : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * real.sqrt (abc)) := 
sorry

end problem_1_problem_2_l268_268880


namespace stickers_difference_l268_268136

theorem stickers_difference (X : ℕ) :
  let Cindy_initial := X
  let Dan_initial := X
  let Cindy_after := Cindy_initial - 15
  let Dan_after := Dan_initial + 18
  Dan_after - Cindy_after = 33 := by
  sorry

end stickers_difference_l268_268136


namespace largest_possible_last_digit_l268_268395

theorem largest_possible_last_digit (D : Fin 3003 → Nat) :
  D 0 = 2 →
  (∀ i : Fin 3002, (10 * D i + D (i + 1)) % 17 = 0 ∨ (10 * D i + D (i + 1)) % 23 = 0) →
  D 3002 = 9 :=
sorry

end largest_possible_last_digit_l268_268395


namespace solve_inequality_l268_268629

theorem solve_inequality (x : ℝ) : 
  3*x^2 + 2*x - 3 > 10 - 2*x ↔ x < ( -2 - Real.sqrt 43 ) / 3 ∨ x > ( -2 + Real.sqrt 43 ) / 3 := 
by
  sorry

end solve_inequality_l268_268629


namespace ten_differences_le_100_exists_l268_268015

theorem ten_differences_le_100_exists (s : Finset ℤ) (h_card : s.card = 101) (h_range : ∀ x ∈ s, 0 ≤ x ∧ x ≤ 1000) :
∃ S : Finset ℕ, S.card = 10 ∧ (∀ y ∈ S, y ≤ 100) :=
by {
  sorry
}

end ten_differences_le_100_exists_l268_268015


namespace range_of_b_l268_268801

theorem range_of_b (a b : ℝ) (h1 : 0 ≤ a + b) (h2 : a + b < 1) (h3 : 2 ≤ a - b) (h4 : a - b < 3) :
  -3 / 2 < b ∧ b < -1 / 2 :=
by
  sorry

end range_of_b_l268_268801


namespace problem_1_problem_2_l268_268833

variable (a b c : ℝ)

theorem problem_1 
  (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  a * b * c ≤ 1 / 9 :=
sorry

theorem problem_2 
  (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * (a * b * c)^(1/2)) :=
sorry

end problem_1_problem_2_l268_268833


namespace ratio_of_segments_l268_268712

theorem ratio_of_segments (a b x : ℝ) (h₁ : a = 9 * x) (h₂ : b = 99 * x) : b / a = 11 := by
  sorry

end ratio_of_segments_l268_268712


namespace part1_part2_l268_268988

noncomputable theory

variables {a b c : ℝ}

-- Condition that a, b, c are positive numbers and the given equation holds
def pos_numbers : Prop := a > 0 ∧ b > 0 ∧ c > 0
def sum_condition : Prop := (a ^ (3/2)) + (b ^ (3/2)) + (c ^ (3/2)) = 1

-- Proof goal for the first part: abc <= 1/9
theorem part1 (h₁ : pos_numbers) (h₂ : sum_condition) : 
  a * b * c ≤ 1 / 9 :=
sorry

-- Proof goal for the second part: a/(b+c) + b/(a+c) + c/(a+b) <= 1/(2 * sqrt(abc))
theorem part2 (h₁ : pos_numbers) (h₂ : sum_condition) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * sqrt (a * b * c)) :=
sorry

end part1_part2_l268_268988


namespace find_m_l268_268318

def A (m : ℤ) : Set ℤ := {2, 5, m ^ 2 - m}
def B (m : ℤ) : Set ℤ := {2, m + 3}

theorem find_m (m : ℤ) : A m ∩ B m = B m → m = 3 := by
  sorry

end find_m_l268_268318


namespace shaded_region_area_l268_268686

noncomputable def area_of_shaded_region (a b c d : ℝ) (area_rect : ℝ) : ℝ :=
  let dg : ℝ := (a * d) / (c + d)
  let area_triangle : ℝ := 0.5 * dg * b
  area_rect - area_triangle

theorem shaded_region_area :
  area_of_shaded_region 12 5 12 4 (4 * 5) = 85 / 8 :=
by
  simp [area_of_shaded_region]
  sorry

end shaded_region_area_l268_268686


namespace domain_of_reciprocal_shifted_function_l268_268556

def domain_of_function (x : ℝ) : Prop :=
  x ≠ 1

theorem domain_of_reciprocal_shifted_function : 
  ∀ x : ℝ, (∃ y : ℝ, y = 1 / (x - 1)) ↔ domain_of_function x :=
by 
  sorry

end domain_of_reciprocal_shifted_function_l268_268556


namespace three_sum_eq_nine_seven_five_l268_268038

theorem three_sum_eq_nine_seven_five {a b c : ℝ} 
    (h1 : b + c = 15 - 2 * a)
    (h2 : a + c = -10 - 4 * b)
    (h3 : a + b = 8 - 2 * c) : 
    3 * a + 3 * b + 3 * c = 9.75 := 
by
    sorry

end three_sum_eq_nine_seven_five_l268_268038


namespace find_five_digit_number_l268_268079

theorem find_five_digit_number (x : ℕ) (hx : 10000 ≤ x ∧ x < 100000)
  (h : 10 * x + 1 = 3 * (100000 + x) ∨ 3 * (10 * x + 1) = 100000 + x) :
  x = 42857 :=
sorry

end find_five_digit_number_l268_268079


namespace contrapositive_l268_268393

theorem contrapositive (a b : ℝ) : (a ≠ 0 ∨ b ≠ 0) → a^2 + b^2 ≠ 0 :=
by
  intro h
  sorry

end contrapositive_l268_268393


namespace part1_part2_l268_268964

def pos_real := {x : ℝ // 0 < x}

variables (a b c : pos_real)

theorem part1 (h : a.val ^ (3/2) + b.val ^ (3/2) + c.val ^ (3/2) = 1) : a.val * b.val * c.val ≤ 1 / 9 :=
sorry

theorem part2 (h : a.val ^ (3/2) + b.val ^ (3/2) + c.val ^ (3/2) = 1) : 
  a.val / (b.val + c.val) + b.val / (a.val + c.val) + c.val / (a.val + b.val) ≤ 1 / (2 * Real.sqrt (a.val * b.val * c.val)) :=
sorry

end part1_part2_l268_268964


namespace pieces_of_meat_per_slice_eq_22_l268_268203

def number_of_pepperoni : Nat := 30
def number_of_ham : Nat := 2 * number_of_pepperoni
def number_of_sausage : Nat := number_of_pepperoni + 12
def total_meat : Nat := number_of_pepperoni + number_of_ham + number_of_sausage
def number_of_slices : Nat := 6

theorem pieces_of_meat_per_slice_eq_22 : total_meat / number_of_slices = 22 :=
by
  sorry

end pieces_of_meat_per_slice_eq_22_l268_268203


namespace part1_part2_l268_268894

variables {a b c : ℝ}

theorem part1 (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h : a^(3/2) + b^(3/2) + c^(3/2) = 1) : abc ≤ 1/9 := 
sorry

theorem part2 (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (abc)) :=
sorry

end part1_part2_l268_268894


namespace abc_le_one_ninth_sum_ratios_le_one_over_two_sqrt_abc_l268_268813

open Real

variables (a b c : ℝ)

-- Condition: a, b, c are positive numbers
variables (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
-- Condition: a^(3/2) + b^(3/2) + c^(3/2) = 1
variable  (h_sum : a^(3/2) + b^(3/2) + c^(3/2) = 1)

-- Question 1: Prove abc ≤ 1/9
theorem abc_le_one_ninth : a * b * c ≤ 1 / 9 :=
  sorry

-- Question 2: Prove (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ 1 / (2 * sqrt (a * b * c))
theorem sum_ratios_le_one_over_two_sqrt_abc : 
  (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ 1 / (2 * sqrt (a * b * c)) :=
  sorry

end abc_le_one_ninth_sum_ratios_le_one_over_two_sqrt_abc_l268_268813


namespace david_still_has_l268_268425

variable (P L S R : ℝ)

def initial_amount : ℝ := 1800
def post_spending_condition (S : ℝ) : ℝ := S - 800
def remaining_money (P S : ℝ) : ℝ := P - S

theorem david_still_has :
  ∀ (S : ℝ),
    initial_amount = P →
    post_spending_condition S = L →
    remaining_money P S = R →
    R = L →
    R = 500 :=
by
  intros S hP hL hR hCl
  sorry

end david_still_has_l268_268425


namespace division_of_product_l268_268261

theorem division_of_product :
  (1.6 * 0.5) / 1 = 0.8 :=
sorry

end division_of_product_l268_268261


namespace employee_monthly_wage_l268_268205

theorem employee_monthly_wage 
(revenue : ℝ)
(tax_rate : ℝ)
(marketing_rate : ℝ)
(operational_cost_rate : ℝ)
(wage_rate : ℝ)
(num_employees : ℕ)
(h_revenue : revenue = 400000)
(h_tax_rate : tax_rate = 0.10)
(h_marketing_rate : marketing_rate = 0.05)
(h_operational_cost_rate : operational_cost_rate = 0.20)
(h_wage_rate : wage_rate = 0.15)
(h_num_employees : num_employees = 10) :
(revenue * (1 - tax_rate) * (1 - marketing_rate) * (1 - operational_cost_rate) * wage_rate / num_employees = 4104) :=
by
  sorry

end employee_monthly_wage_l268_268205


namespace last_four_digits_of_5_pow_2011_l268_268367

theorem last_four_digits_of_5_pow_2011 :
  (5^2011) % 10000 = 8125 := 
by
  -- Using modular arithmetic and periodicity properties of powers of 5.
  sorry

end last_four_digits_of_5_pow_2011_l268_268367


namespace no_possible_seating_arrangement_l268_268417

theorem no_possible_seating_arrangement : 
  ¬(∃ (students : Fin 11 → Fin 4),
    ∀ (i : Fin 11),
    ∃ (s1 s2 s3 s4 s5 : Fin 11),
      s1 = i ∧ 
      (s2 = (i + 1) % 11) ∧ 
      (s3 = (i + 2) % 11) ∧ 
      (s4 = (i + 3) % 11) ∧ 
      (s5 = (i + 4) % 11) ∧
      ∃ (g1 g2 g3 g4 : Fin 4),
        (students s1 = g1) ∧ 
        (students s2 = g2) ∧ 
        (students s3 = g3) ∧ 
        (students s4 = g4) ∧ 
        (students s5).val ≠ (students s1).val ∧ 
        (students s5).val ≠ (students s2).val ∧ 
        (students s5).val ≠ (students s3).val ∧ 
        (students s5).val ≠ (students s4).val) :=
sorry

end no_possible_seating_arrangement_l268_268417


namespace sufficient_not_necessary_condition_l268_268312

theorem sufficient_not_necessary_condition (a b : ℝ) (h : (a - b) * a^2 > 0) : a > b ∧ a ≠ 0 :=
by {
  sorry
}

end sufficient_not_necessary_condition_l268_268312


namespace parallel_trans_l268_268330

variables {Line : Type} (a b c : Line)

-- Define parallel relation
def parallel (x y : Line) : Prop := sorry -- Replace 'sorry' with the actual definition

-- The main theorem
theorem parallel_trans (h1 : parallel a c) (h2 : parallel b c) : parallel a b :=
sorry

end parallel_trans_l268_268330


namespace range_of_a_l268_268654

theorem range_of_a (a : ℝ) : (∃ x : ℝ, x^2 - a * x - a ≤ -3) ↔ (a ≤ -6 ∨ a ≥ 2) :=
by
  sorry

end range_of_a_l268_268654


namespace petya_four_digits_l268_268372

theorem petya_four_digits :
  ∃ (a b c d : ℕ), 
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ 
    a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧ 
    6 * (a + b + c + d) * 1111 = 73326 ∧ 
    (∃ S, S = a + b + c + d ∧ S = 11) :=
begin
  use 1, 2, 3, 5,
  split, { intro h, exact dec_trivial },
  split, { intro h, exact dec_trivial },
  split, { intro h, exact dec_trivial },
  split, { intro h, exact dec_trivial },
  split, { intro h, exact dec_trivial },
  split, { intro h, exact dec_trivial },
  split, { intro h, exact dec_trivial },
  split, { intro h, exact dec_trivial },
  split, { intro h, exact dec_trivial },
  split, { intro h, exact dec_trivial },
  split,
  { exact dec_trivial },
  use 11,
  split,
  { exact dec_trivial },
  { exact dec_trivial }
end

end petya_four_digits_l268_268372


namespace sufficient_condition_for_inequality_l268_268657

theorem sufficient_condition_for_inequality (a : ℝ) (h : 0 < a ∧ a < 4) :
  ∀ x : ℝ, a * x^2 + a * x + 1 > 0 :=
by
  sorry

end sufficient_condition_for_inequality_l268_268657


namespace number_of_sweaters_l268_268043

theorem number_of_sweaters 
(total_price_shirts : ℝ)
(total_shirts : ℕ)
(total_price_sweaters : ℝ)
(price_difference : ℝ) :
total_price_shirts = 400 ∧ total_shirts = 25 ∧ total_price_sweaters = 1500 ∧ price_difference = 4 →
(total_price_sweaters / ((total_price_shirts / total_shirts) + price_difference) = 75) :=
by
  intros
  sorry

end number_of_sweaters_l268_268043


namespace add_base8_l268_268772

theorem add_base8 : 
  let a := 2 * 8^2 + 4 * 8^1 + 6 * 8^0
  let b := 5 * 8^2 + 7 * 8^1 + 3 * 8^0
  let c := 6 * 8^1 + 2 * 8^0
  let sum := a + b + c
  sum = 1 * 8^3 + 1 * 8^2 + 2 * 8^1 + 3 * 8^0 :=
by
  -- Proof skipped
  sorry

end add_base8_l268_268772


namespace part1_part2_l268_268979

noncomputable theory

variables {a b c : ℝ}

-- Condition that a, b, c are positive numbers and the given equation holds
def pos_numbers : Prop := a > 0 ∧ b > 0 ∧ c > 0
def sum_condition : Prop := (a ^ (3/2)) + (b ^ (3/2)) + (c ^ (3/2)) = 1

-- Proof goal for the first part: abc <= 1/9
theorem part1 (h₁ : pos_numbers) (h₂ : sum_condition) : 
  a * b * c ≤ 1 / 9 :=
sorry

-- Proof goal for the second part: a/(b+c) + b/(a+c) + c/(a+b) <= 1/(2 * sqrt(abc))
theorem part2 (h₁ : pos_numbers) (h₂ : sum_condition) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * sqrt (a * b * c)) :=
sorry

end part1_part2_l268_268979


namespace factor_difference_of_squares_l268_268420

theorem factor_difference_of_squares (a b : ℝ) : 
    (∃ A B : ℝ, a = A ∧ b = B) → 
    (a^2 - b^2 = (a + b) * (a - b)) :=
by
  intros h
  sorry

end factor_difference_of_squares_l268_268420


namespace general_formula_a_sum_bn_l268_268315

noncomputable section

open Nat

-- Define the sequence Sn
def S (n : ℕ) : ℕ := 2^n + n - 1

-- Define the sequence an
def a (n : ℕ) : ℕ := 1 + 2^(n-1)

-- Define the sequence bn
def b (n : ℕ) : ℕ := 2 * n * (a n - 1)

-- Define the sum Tn
def T (n : ℕ) : ℕ := n * 2^n

-- Proposition 1: General formula for an
theorem general_formula_a (n : ℕ) : a n = 1 + 2^(n-1) :=
by
  sorry

-- Proposition 2: Sum of first n terms of bn
theorem sum_bn (n : ℕ) : T n = 2 + (n - 1) * 2^(n+1) :=
by
  sorry

end general_formula_a_sum_bn_l268_268315


namespace complex_magnitude_comparison_l268_268644

open Complex

theorem complex_magnitude_comparison :
  let z1 := (5 : ℂ) + (3 : ℂ) * I
  let z2 := (5 : ℂ) + (4 : ℂ) * I
  abs z1 < abs z2 :=
by 
  let z1 := (5 : ℂ) + (3 : ℂ) * I
  let z2 := (5 : ℂ) + (4 : ℂ) * I
  sorry

end complex_magnitude_comparison_l268_268644


namespace part1_part2_l268_268993

-- a), b), c are positive real numbers and ${a}^{\frac{3}{2}}+{b}^{\frac{3}{2}}+{c}^{\frac{3}{2}}=1$
variables (a b c : ℝ)
axiom a_pos : 0 < a
axiom b_pos : 0 < b
axiom c_pos : 0 < c
axiom sum_eq_one : a^(3/2) + b^(3/2) + c^(3/2) = 1

-- Prove (1) and (2)
theorem part1 : abc ≤ 1 / 9 := sorry

theorem part2 : a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (abc)) := sorry

end part1_part2_l268_268993


namespace vasya_lowest_position_l268_268226

theorem vasya_lowest_position
  (num_cyclists : ℕ)
  (num_stages : ℕ)
  (num_ahead : ℕ)
  (position_vasya : ℕ)
  (total_time : List ℕ)
  (unique_total_times : total_time.nodup)
  (stage_positions : List (List ℕ))
  (unique_stage_positions : ∀ stage ∈ stage_positions, stage.nodup)
  (vasya_consistent : ∀ stage ∈ stage_positions, stage.nth position_vasya = some num_ahead) :
  num_ahead * num_stages + 1 = 91 :=
by
  sorry

end vasya_lowest_position_l268_268226


namespace remainder_when_divided_by_7_l268_268168

theorem remainder_when_divided_by_7 (n : ℕ) (h : (2 * n) % 7 = 4) : n % 7 = 2 :=
  by sorry

end remainder_when_divided_by_7_l268_268168


namespace part1_part2_l268_268862

variable (a b c : ℝ)

theorem part1 (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) : abc ≤ 1/9 :=
by
  sorry

theorem part2 (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
    (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ 1 / (2 * sqrt(abc)) :=
by
  sorry

end part1_part2_l268_268862


namespace intersection_of_sets_l268_268329

-- Define sets A and B
def A : Set ℕ := {0, 1, 2}
def B : Set ℕ := {1, 2, 3}

-- State the theorem
theorem intersection_of_sets : A ∩ B = {1, 2} := by
  sorry

end intersection_of_sets_l268_268329


namespace solution_set_of_inequality_l268_268469

theorem solution_set_of_inequality :
  {x : ℝ | (3 * x - 1) / (2 - x) ≥ 0} = {x : ℝ | 1/3 ≤ x ∧ x < 2} :=
by
  sorry

end solution_set_of_inequality_l268_268469


namespace negate_existential_l268_268564

theorem negate_existential :
  ¬ (∃ x0 : ℝ, x0^2 - 2 * x0 + 4 > 0) ↔ ∀ x : ℝ, x^2 - 2 * x + 4 ≤ 0 :=
by
  sorry

end negate_existential_l268_268564


namespace talent_show_l268_268076

theorem talent_show (B G : ℕ) (h1 : G = B + 22) (h2 : G + B = 34) : G = 28 :=
by
  sorry

end talent_show_l268_268076


namespace cos_double_angle_sum_l268_268488

variables {α β : ℝ}

theorem cos_double_angle_sum (h1: sin (α - β) = 1 / 3) (h2: cos α * sin β = 1 / 6) :
  cos (2 * α + 2 * β) = 1 / 9 :=
sorry

end cos_double_angle_sum_l268_268488


namespace find_150th_letter_l268_268593

def repeating_sequence : String := "ABCD"

def position := 150

theorem find_150th_letter :
  repeating_sequence[(position % 4) - 1] = 'B' := 
sorry

end find_150th_letter_l268_268593


namespace vasya_maximum_rank_l268_268235

theorem vasya_maximum_rank {n : ℕ} (cyclists stages : ℕ) (VasyaPlace : ℕ) 
  (rankings : Π (s : fin stages), fin cyclists):
  cyclists = 500 → stages = 15 → VasyaPlace = 7 →
  (∀ s, rankings s VasyaPlace = 7) →
  (∀ s t i, rankings s i ≠ rankings t i) →
  (∀ s, ∃ l, list.nodup l ∧ (∀ i j, i < j → rankings s i < rankings s j) ∧ list.length l = 500) →
  (∃ (maxRank : ℕ), maxRank = 91) :=
by
  intros hcyclists hstages hplace hvasya_place hdistinct_rankings hstage_rankings
  use 91
  sorry

end vasya_maximum_rank_l268_268235


namespace problem_1_problem_2_l268_268935

variable {a b c : ℝ}

theorem problem_1 (hp : 0 < a ∧ 0 < b ∧ 0 < c)
    (h1 : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
    abc ≤ 1/9 := 
by 
  apply sorry

theorem problem_2 (hp : 0 < a ∧ 0 < b ∧ 0 < c)
    (h1 : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
    a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * real.sqrt (abc)) := 
by 
  apply sorry

end problem_1_problem_2_l268_268935


namespace negation_of_p_l268_268171

-- Define the proposition p
def p : Prop := ∀ x : ℝ, 2 * x^2 + 1 > 0

-- State the negation of p
theorem negation_of_p : ¬p ↔ ∃ x : ℝ, 2 * x^2 + 1 ≤ 0 := sorry

end negation_of_p_l268_268171


namespace f_is_periodic_l268_268198

noncomputable def f (x : ℝ) : ℝ := x - ⌈x⌉

theorem f_is_periodic : ∀ x : ℝ, f (x + 1) = f x :=
by 
  intro x
  sorry

end f_is_periodic_l268_268198


namespace fraction_of_Bhupathi_is_point4_l268_268771

def abhinav_and_bhupathi_amounts (A B : ℝ) : Prop :=
  A + B = 1210 ∧ B = 484

theorem fraction_of_Bhupathi_is_point4 (A B : ℝ) (x : ℝ) (h : abhinav_and_bhupathi_amounts A B) :
  (4 / 15) * A = x * B → x = 0.4 :=
by
  sorry

end fraction_of_Bhupathi_is_point4_l268_268771


namespace probability_multiple_of_3_or_4_l268_268572

-- Given the numbers 1 through 30 are written on 30 cards one number per card,
-- and Sara picks one of the 30 cards at random,
-- the probability that the number on her card is a multiple of 3 or 4 is 1/2.

-- Define the set of numbers from 1 to 30
def numbers := finset.range 30 \ {0}

-- Define what it means to be a multiple of 3 or 4 within the given range
def is_multiple_of_3_or_4 (n : ℕ) : Prop :=
  n % 3 = 0 ∨ n % 4 = 0

-- Define the set of multiples of 3 or 4 within the given range
def multiples_of_3_or_4 := numbers.filter is_multiple_of_3_or_4

-- The probability calculation
theorem probability_multiple_of_3_or_4 : 
  (multiples_of_3_or_4.card : ℚ) / numbers.card = 1 / 2 :=
begin
  -- The set multiples_of_3_or_4 contains 15 elements
  have h_multiples_card : multiples_of_3_or_4.card = 15, sorry,
  -- The set numbers contains 30 elements
  have h_numbers_card : numbers.card = 30, sorry,
  -- Therefore, the probability is 15/30 = 1/2
  rw [h_multiples_card, h_numbers_card],
  norm_num,
end

end probability_multiple_of_3_or_4_l268_268572


namespace cos_double_angle_l268_268476

variables {α β : ℝ}

-- Conditions
def condition1 : Prop := sin (α - β) = 1 / 3
def condition2 : Prop := cos α * sin β = 1 / 6

-- Statement to prove
theorem cos_double_angle (h1 : condition1) (h2 : condition2) : cos (2 * α + 2 * β) = 1 / 9 :=
by
  -- proof goes here
  sorry

end cos_double_angle_l268_268476


namespace petya_goal_unachievable_l268_268127

theorem petya_goal_unachievable (n : Nat) (hn : n ≥ 2) : 
  ¬(∃ (arrangement : Fin 2n → Bool), ∀ i, (arrangement i = !arrangement ((i + 1) % (2 * n))) → false) :=
by
  sorry

end petya_goal_unachievable_l268_268127


namespace trig_identity_proof_l268_268525

theorem trig_identity_proof 
  (α p q : ℝ)
  (hp : p ≠ 0) (hq : q ≠ 0)
  (tangent : Real.tan α = p / q) :
  Real.sin (2 * α) = 2 * p * q / (p^2 + q^2) ∧
  Real.cos (2 * α) = (q^2 - p^2) / (q^2 + p^2) ∧
  Real.tan (2 * α) = (2 * p * q) / (q^2 - p^2) :=
by
  sorry

end trig_identity_proof_l268_268525


namespace probability_multiple_of_3_or_4_l268_268571

-- Given the numbers 1 through 30 are written on 30 cards one number per card,
-- and Sara picks one of the 30 cards at random,
-- the probability that the number on her card is a multiple of 3 or 4 is 1/2.

-- Define the set of numbers from 1 to 30
def numbers := finset.range 30 \ {0}

-- Define what it means to be a multiple of 3 or 4 within the given range
def is_multiple_of_3_or_4 (n : ℕ) : Prop :=
  n % 3 = 0 ∨ n % 4 = 0

-- Define the set of multiples of 3 or 4 within the given range
def multiples_of_3_or_4 := numbers.filter is_multiple_of_3_or_4

-- The probability calculation
theorem probability_multiple_of_3_or_4 : 
  (multiples_of_3_or_4.card : ℚ) / numbers.card = 1 / 2 :=
begin
  -- The set multiples_of_3_or_4 contains 15 elements
  have h_multiples_card : multiples_of_3_or_4.card = 15, sorry,
  -- The set numbers contains 30 elements
  have h_numbers_card : numbers.card = 30, sorry,
  -- Therefore, the probability is 15/30 = 1/2
  rw [h_multiples_card, h_numbers_card],
  norm_num,
end

end probability_multiple_of_3_or_4_l268_268571


namespace bacon_strips_needed_l268_268455

theorem bacon_strips_needed (plates : ℕ) (eggs_per_plate : ℕ) (bacon_per_plate : ℕ) (customers : ℕ) :
  eggs_per_plate = 2 →
  bacon_per_plate = 2 * eggs_per_plate →
  customers = 14 →
  plates = customers →
  plates * bacon_per_plate = 56 := by
  sorry

end bacon_strips_needed_l268_268455


namespace abc_le_one_ninth_sum_ratios_le_one_over_two_sqrt_abc_l268_268822

open Real

variables (a b c : ℝ)

-- Condition: a, b, c are positive numbers
variables (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
-- Condition: a^(3/2) + b^(3/2) + c^(3/2) = 1
variable  (h_sum : a^(3/2) + b^(3/2) + c^(3/2) = 1)

-- Question 1: Prove abc ≤ 1/9
theorem abc_le_one_ninth : a * b * c ≤ 1 / 9 :=
  sorry

-- Question 2: Prove (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ 1 / (2 * sqrt (a * b * c))
theorem sum_ratios_le_one_over_two_sqrt_abc : 
  (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ 1 / (2 * sqrt (a * b * c)) :=
  sorry

end abc_le_one_ninth_sum_ratios_le_one_over_two_sqrt_abc_l268_268822


namespace arithmetic_sum_nine_l268_268807

noncomputable def arithmetic_sequence_sum (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  n / 2 * (a 1 + a n)

theorem arithmetic_sum_nine (a : ℕ → ℝ)
  (h1 : ∀ n, a (n + 1) - a n = a 2 - a 1)
  (h2 : a 4 = 9)
  (h3 : a 6 = 11) : arithmetic_sequence_sum a 9 = 90 :=
by
  sorry

end arithmetic_sum_nine_l268_268807


namespace least_pounds_of_sugar_l268_268288

theorem least_pounds_of_sugar :
  ∃ s : ℝ, (∀ f : ℝ, (f ≥ 6 + s / 2 ∧ f ≤ 2 * s) → s = 4) :=
by {
    use 4,
    sorry
}

end least_pounds_of_sugar_l268_268288


namespace proof_part1_proof_part2_l268_268857

theorem proof_part1 (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) 
                    (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
                    (a * b * c ≤ 1 / 9) := 
by
  sorry

theorem proof_part2 (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) 
                    (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
                    (a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * (a * b * c)^(1/2))) := 
by
  sorry

end proof_part1_proof_part2_l268_268857


namespace standard_eq_hyperbola_trajectory_of_q_l268_268804

/-- Given a hyperbola with foci on the x-axis, a real axis length of 4√2, and eccentricity e = √5/2, 
    the standard equation of the hyperbola is x²/8 - y²/2 = 1. -/
theorem standard_eq_hyperbola (a b c : ℝ) (e : ℝ) 
    (h1 : 2*a = 4*sqrt 2)
    (h2 : e = sqrt 5 / 2)
    (h3 : c = e * a)
    (h4 : a^2 + b^2 = c^2) :
    (a = 2*sqrt 2) ∧ (b = sqrt 2) ∧ 
    (∀ x y : ℝ, (x^2) / (a^2) - (y^2) / (b^2) = 1 ↔ x^2/8 - y^2/2 = 1) :=
sorry -- This statement will be proven.

/-- Given a hyperbola with its real axis A₁A₂ and moving point P on the hyperbola,
    point Q satisfies A₁Q ⊥ A₁P and A₂Q ⊥ A₂P, the trajectory of point Q is
    x²/8 - y²/4 = 1 excluding the two vertices.
-/
theorem trajectory_of_q (x y x₀ y₀ : ℝ)
    (a b : ℝ)
    (h1 : 2*a = 4*sqrt 2)
    (h2 : a = 2*sqrt 2)
    (h3 : b = sqrt 2)
    (h4 : x₀ ∈ set_of (fun p => p ≠ ± (2*sqrt 2))) 
    (h5 : x₀^2 / a^2 - y₀^2 / b^2 = 1) 
    (h6 : y / (x+2*sqrt 2) * y₀ / (x₀+2*sqrt 2) = -1) 
    (h7 : y / (x-2*sqrt 2) * y₀ / (x₀-2*sqrt 2) = -1) :
    ∀ x y : ℝ, (x^2 / 8 - y^2 / 4 = 1 ↔ x ∉ set_of (λ p, p = ± (2*sqrt 2))) :=
sorry -- This statement will be proven.

end standard_eq_hyperbola_trajectory_of_q_l268_268804


namespace inscribed_circle_radius_l268_268522

theorem inscribed_circle_radius
  (A p s : ℝ) (h1 : A = p) (h2 : s = p / 2) (r : ℝ) (h3 : A = r * s) :
  r = 2 :=
sorry

end inscribed_circle_radius_l268_268522


namespace problem1_problem2_l268_268837

variable {a b c : ℝ}

-- Conditions: a, b, c are positive numbers and a^(3/2) + b^(3/2) + c^(3/2) = 1
axiom pos_a : 0 < a
axiom pos_b : 0 < b
axiom pos_c : 0 < c
axiom sum_pow : a^(3/2) + b^(3/2) + c^(3/2) = 1

-- Problem (1): Prove abc ≤ 1/9 given the conditions
theorem problem1 : abc ≤ 1 / 9 := by
  sorry

-- Problem (2): Prove (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ (1 / (2 * (abc) ^ (1/2))) given the conditions
theorem problem2 : (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ 1 / (2 * (abc) ^ (1/2)) := by
  sorry

end problem1_problem2_l268_268837


namespace painting_price_after_5_years_l268_268070

variable (P : ℝ)
-- Conditions on price changes over the years
def year1_price (P : ℝ) := P * 1.30
def year2_price (P : ℝ) := year1_price P * 0.80
def year3_price (P : ℝ) := year2_price P * 1.25
def year4_price (P : ℝ) := year3_price P * 0.90
def year5_price (P : ℝ) := year4_price P * 1.15

theorem painting_price_after_5_years (P : ℝ) :
  year5_price P = 1.3455 * P := by
  sorry

end painting_price_after_5_years_l268_268070


namespace max_mn_on_parabola_l268_268582

theorem max_mn_on_parabola :
  ∀ m n : ℝ, (n = -m^2 + 3) → (m + n ≤ 13 / 4) :=
by
  sorry

end max_mn_on_parabola_l268_268582


namespace part1_part2_l268_268992

-- a), b), c are positive real numbers and ${a}^{\frac{3}{2}}+{b}^{\frac{3}{2}}+{c}^{\frac{3}{2}}=1$
variables (a b c : ℝ)
axiom a_pos : 0 < a
axiom b_pos : 0 < b
axiom c_pos : 0 < c
axiom sum_eq_one : a^(3/2) + b^(3/2) + c^(3/2) = 1

-- Prove (1) and (2)
theorem part1 : abc ≤ 1 / 9 := sorry

theorem part2 : a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (abc)) := sorry

end part1_part2_l268_268992


namespace problem_1_problem_2_l268_268831

variable (a b c : ℝ)

theorem problem_1 
  (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  a * b * c ≤ 1 / 9 :=
sorry

theorem problem_2 
  (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * (a * b * c)^(1/2)) :=
sorry

end problem_1_problem_2_l268_268831


namespace problem_1_problem_2_l268_268933

variable {a b c : ℝ}

theorem problem_1 (hp : 0 < a ∧ 0 < b ∧ 0 < c)
    (h1 : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
    abc ≤ 1/9 := 
by 
  apply sorry

theorem problem_2 (hp : 0 < a ∧ 0 < b ∧ 0 < c)
    (h1 : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
    a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * real.sqrt (abc)) := 
by 
  apply sorry

end problem_1_problem_2_l268_268933


namespace altitude_line_equation_equal_distance_lines_l268_268154

-- Define the points A, B, and C
def A : ℝ × ℝ := (4, 0)
def B : ℝ × ℝ := (8, 10)
def C : ℝ × ℝ := (0, 6)

-- The equation of the line for the altitude from A to BC
theorem altitude_line_equation :
  ∃ (a b c : ℝ), 2 * a - 3 * b + 14 = 0 :=
sorry

-- The equations of the line passing through B such that the distances from A and C are equal
theorem equal_distance_lines :
  ∃ (a b c : ℝ), (7 * a - 6 * b + 4 = 0) ∧ (3 * a + 2 * b - 44 = 0) :=
sorry

end altitude_line_equation_equal_distance_lines_l268_268154


namespace min_distance_parabola_l268_268018

open Real

theorem min_distance_parabola {P : ℝ × ℝ} (hP : P.2^2 = 4 * P.1) : ∃ m : ℝ, m = 2 * sqrt 3 ∧ ∀ Q : ℝ × ℝ, Q = (4, 0) → dist P Q ≥ m :=
by sorry

end min_distance_parabola_l268_268018


namespace purely_imaginary_l268_268339

theorem purely_imaginary {m : ℝ} (h1 : m^2 - 3 * m = 0) (h2 : m^2 - 5 * m + 6 ≠ 0) : m = 0 :=
sorry

end purely_imaginary_l268_268339


namespace part1_part2_l268_268887

variables {a b c : ℝ}

theorem part1 (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h : a^(3/2) + b^(3/2) + c^(3/2) = 1) : abc ≤ 1/9 := 
sorry

theorem part2 (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (abc)) :=
sorry

end part1_part2_l268_268887


namespace probability_multiple_of_3_or_4_l268_268566

theorem probability_multiple_of_3_or_4 :
  let numbers := Finset.range 30
  let multiples_of_3 := {n ∈ numbers | n % 3 = 0}
  let multiples_of_4 := {n ∈ numbers | n % 4 = 0}
  let multiples_of_12 := {n ∈ numbers | n % 12 = 0}
  let favorable_count := multiples_of_3.card + multiples_of_4.card - multiples_of_12.card
  let probability := (favorable_count : ℚ) / numbers.card
  probability = (1 / 2 : ℚ) :=
by
  sorry

end probability_multiple_of_3_or_4_l268_268566


namespace ratio_of_money_spent_l268_268280

theorem ratio_of_money_spent (h : ∀(a b c : ℕ), a + b + c = 75) : 
  (25 / 75 = 1 / 3) ∧ 
  (40 / 75 = 4 / 3) ∧ 
  (10 / 75 = 2 / 15) :=
by
  sorry

end ratio_of_money_spent_l268_268280


namespace abc_le_one_ninth_sum_ratios_le_one_over_two_sqrt_abc_l268_268812

open Real

variables (a b c : ℝ)

-- Condition: a, b, c are positive numbers
variables (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
-- Condition: a^(3/2) + b^(3/2) + c^(3/2) = 1
variable  (h_sum : a^(3/2) + b^(3/2) + c^(3/2) = 1)

-- Question 1: Prove abc ≤ 1/9
theorem abc_le_one_ninth : a * b * c ≤ 1 / 9 :=
  sorry

-- Question 2: Prove (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ 1 / (2 * sqrt (a * b * c))
theorem sum_ratios_le_one_over_two_sqrt_abc : 
  (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ 1 / (2 * sqrt (a * b * c)) :=
  sorry

end abc_le_one_ninth_sum_ratios_le_one_over_two_sqrt_abc_l268_268812


namespace intersection_M_N_l268_268507

-- Define the set M based on the given condition
def M : Set ℝ := { x | x^2 > 1 }

-- Define the set N based on the given elements
def N : Set ℝ := { x | x = -2 ∨ x = -1 ∨ x = 0 ∨ x = 1 ∨ x = 2 }

-- Prove that the intersection of M and N is {-2, 2}
theorem intersection_M_N : M ∩ N = { -2, 2 } := by
  sorry

end intersection_M_N_l268_268507


namespace weekly_allowance_is_8_l268_268284

variable (A : ℝ)

def condition_1 (A : ℝ) : Prop := ∃ A : ℝ, A / 2 + 8 = 12

theorem weekly_allowance_is_8 (A : ℝ) (h : condition_1 A) : A = 8 :=
sorry

end weekly_allowance_is_8_l268_268284


namespace find_circle_equation_l268_268646

noncomputable def center (m : ℝ) := (3 * m, m)

def radius (m : ℝ) : ℝ := 3 * m

def circle_eq (m : ℝ) (x y : ℝ) : Prop :=
  (x - 3 * m)^2 + (y - m)^2 = (radius m)^2

def point_A : ℝ × ℝ := (6, 1)

theorem find_circle_equation (m : ℝ) :
  (radius m = 3 * m ∧ center m = (3 * m, m) ∧ 
   point_A = (6, 1) ∧
   circle_eq m 6 1) →
  (circle_eq 1 x y ∨ circle_eq 37 x y) :=
by
  sorry

end find_circle_equation_l268_268646


namespace max_min_product_xy_theorem_l268_268399

noncomputable def max_min_product_xy (x y a : ℝ) (h1 : x + y = a) (h2 : x^2 + y^2 = -a^2 + 2) : Prop :=
  -1 ≤ x * y ∧ x * y ≤ 1/2

theorem max_min_product_xy_theorem (x y a : ℝ) (h1 : x + y = a) (h2 : x^2 + y^2 = -a^2 + 2) :
  max_min_product_xy x y a h1 h2 :=
sorry

end max_min_product_xy_theorem_l268_268399


namespace lowest_position_l268_268233

theorem lowest_position (num_cyclists : ℕ) (num_stages : ℕ) (vasya_position : ℕ) :
  num_cyclists = 500 →
  num_stages = 15 →
  vasya_position = 7 →
  ∃ n, n = 91 :=
by
  intros
  sorry

end lowest_position_l268_268233


namespace range_of_a_l268_268027

noncomputable def f (a e x : ℝ) := 2 * a^x - e * x^2
noncomputable def f' (a e x : ℝ) := 2 * a^x * Real.log a - 2 * e * x
noncomputable def f'' (a e x : ℝ) := 2 * a^x * (Real.log a)^2 - 2 * e

theorem range_of_a (a e x1 x2 : ℝ) (ha1 : a > 0) (ha2 : a ≠ 1) (hx : x1 < x2)
  (hmin : ∀ x, f' a e x = 0 → x = x1 ∨ x = x2)
  (hmax : f'' a e x1 > 0 ∧ f'' a e x2 < 0) :
  a ∈ Ioo (1 / Real.exp 1) 1 :=
by
  sorry

end range_of_a_l268_268027


namespace manny_problem_l268_268540

noncomputable def num_slices_left (num_pies : Nat) (slices_per_pie : Nat) (num_classmates : Nat) (num_teachers : Nat) (num_slices_per_person : Nat) : Nat :=
  let total_slices := num_pies * slices_per_pie
  let total_people := 1 + num_classmates + num_teachers
  let slices_taken := total_people * num_slices_per_person
  total_slices - slices_taken

theorem manny_problem : num_slices_left 3 10 24 1 1 = 4 := by
  sorry

end manny_problem_l268_268540


namespace right_triangle_of_pythagorean_l268_268676

theorem right_triangle_of_pythagorean
  (A B C : Type)
  [MetricSpace A] [MetricSpace B] [MetricSpace C]
  (AB BC CA : ℝ)
  (h : AB^2 = BC^2 + CA^2) : ∃ (c : ℕ), c = 90 :=
by
  sorry

end right_triangle_of_pythagorean_l268_268676


namespace quadratic_inequality_solution_l268_268386

theorem quadratic_inequality_solution (x : ℝ) :
    -15 * x^2 + 10 * x + 5 > 0 ↔ (-1 / 3 : ℝ) < x ∧ x < 1 :=
by
  sorry

end quadratic_inequality_solution_l268_268386


namespace p_neither_sufficient_nor_necessary_l268_268053

theorem p_neither_sufficient_nor_necessary (x y : ℝ) :
  (x > 1 ∧ y > 1) ↔ ¬((x > 1 ∧ y > 1) → (x + y > 3)) ∧ ¬((x + y > 3) → (x > 1 ∧ y > 1)) :=
by
  sorry

end p_neither_sufficient_nor_necessary_l268_268053


namespace remainder_549547_div_7_l268_268735

theorem remainder_549547_div_7 : 549547 % 7 = 5 :=
by
  sorry

end remainder_549547_div_7_l268_268735


namespace eval_expression_l268_268268

theorem eval_expression : 10 * 1.8 - 2 * 1.5 / 0.3 = 8 := 
by
  sorry

end eval_expression_l268_268268


namespace probability_seating_7_probability_seating_n_l268_268703

-- Definitions and theorem for case (a): n = 7
def num_ways_to_seat_7 (n : ℕ) (k : ℕ) : ℕ := n * (n - 1) * (n - 2) / k!

def valid_arrangements_7 : ℕ := 1

theorem probability_seating_7 : 
  let total_ways := num_ways_to_seat_7 7 3 in
  let valid_ways := valid_arrangements_7 in
  total_ways = 35 ∧ (valid_ways : ℚ) / total_ways = 0.2 := 
by
  sorry

-- Definitions and theorem for case (b): general n
def choose (n k : ℕ) : ℕ := n * (n - 1) / 2

def valid_arrangements_n (n : ℕ) : ℚ := (n - 4) * (n - 5) / 2

theorem probability_seating_n (n : ℕ) (hn : n ≥ 6) : 
  let total_ways := choose (n - 1) 2 in
  let valid_ways := valid_arrangements_n n in
  total_ways = (n - 1) * (n - 2) / 2 ∧ 
  (valid_ways : ℚ) / total_ways = (n - 4) * (n - 5) / ((n - 1) * (n - 2)) :=
by
  sorry

end probability_seating_7_probability_seating_n_l268_268703


namespace least_value_a2000_l268_268441

theorem least_value_a2000 (a : ℕ → ℕ)
  (h1 : ∀ m n, (m ∣ n) → (m < n) → (a m ∣ a n))
  (h2 : ∀ m n, (m ∣ n) → (m < n) → (a m < a n)) :
  a 2000 >= 128 :=
sorry

end least_value_a2000_l268_268441


namespace a_plus_b_minus_c_in_S_l268_268017

-- Define the sets P, Q, and S
def P := {x : ℤ | ∃ k : ℤ, x = 3 * k}
def Q := {x : ℤ | ∃ k : ℤ, x = 3 * k + 1}
def S := {x : ℤ | ∃ k : ℤ, x = 3 * k - 1}

-- Define the elements a, b, and c as members of sets P, Q, and S respectively
variables (a b c : ℤ)
variable (ha : a ∈ P) -- a ∈ P
variable (hb : b ∈ Q) -- b ∈ Q
variable (hc : c ∈ S) -- c ∈ S

-- Theorem statement proving the question
theorem a_plus_b_minus_c_in_S : a + b - c ∈ S := sorry

end a_plus_b_minus_c_in_S_l268_268017


namespace part1_part2_l268_268975

variable (a b c : ℝ)

-- Condition: a, b, c are positive numbers and a^(3/2) + b^(3/2) + c^(3/2) = 1
axiom pos_a : a > 0
axiom pos_b : b > 0
axiom pos_c : c > 0
axiom sum_condition : a^(3/2) + b^(3/2) + c^(3/2) = 1

-- Proof Problem (1): abc ≤ 1/9
theorem part1 : abc ≤ 1/9 := 
by {
  have h := sorry,
  exact h
}

-- Proof Problem (2): a/(b+c) + b/(a+c) + c/(a+b) ≤ 1/(2sqrt(abc))
theorem part2 : a/(b+c) + b/(a+c) + c/(a+b) ≤ 1/(2*sqrt(abc)) := 
by {
  have h := sorry,
  exact h
}

end part1_part2_l268_268975


namespace circle_count_gt_l268_268354

axiom points_not_collinear {n : ℕ} (P : fin 2n+3 → ℝ × ℝ) :
  ∀ i j k : fin 2n+3, i ≠ j → j ≠ k → i ≠ k → ¬ collinear (P i) (P j) (P k)

axiom points_not_concyclic {n : ℕ} (P : fin 2n+3 → ℝ × ℝ) :
  ∀ i j k l : fin 2n+3, i ≠ j → j ≠ k → k ≠ l → l ≠ i → ¬ concyclic (P i) (P j) (P k) (P l)

def num_circles {n : ℕ} (P : fin 2n+3 → ℝ × ℝ) : ℕ := 
  { K | ∃ i j k : fin 2n+3, i ≠ j ∧ j ≠ k ∧ i ≠ k ∧
    (set.count_inside_outside (P i) (P j) (P k) P).fst = n ∧ 
    (set.count_inside_outside (P i) (P j) (P k) P).snd = n }.count

theorem circle_count_gt {n : ℕ} (P : fin 2n+3 → ℝ × ℝ) :
  (num_circles P) > (1/real.pi * nat.choose (2n + 3) 2) :=
by {
  sorry
}

end circle_count_gt_l268_268354


namespace find_a_and_b_l268_268536

noncomputable def f (x : ℝ) : ℝ := abs (Real.log (x + 1))

theorem find_a_and_b
  (a b : ℝ)
  (h1 : a < b)
  (h2 : f a = f ((- (b + 1)) / (b + 2)))
  (h3 : f (10 * a + 6 * b + 21) = 4 * Real.log 2) :
  a = - 2 / 5 ∧ b = - 1 / 3 :=
sorry

end find_a_and_b_l268_268536


namespace talent_show_l268_268075

theorem talent_show (B G : ℕ) (h1 : G = B + 22) (h2 : G + B = 34) : G = 28 :=
by
  sorry

end talent_show_l268_268075


namespace problem1_problem2_l268_268836

variable {a b c : ℝ}

-- Conditions: a, b, c are positive numbers and a^(3/2) + b^(3/2) + c^(3/2) = 1
axiom pos_a : 0 < a
axiom pos_b : 0 < b
axiom pos_c : 0 < c
axiom sum_pow : a^(3/2) + b^(3/2) + c^(3/2) = 1

-- Problem (1): Prove abc ≤ 1/9 given the conditions
theorem problem1 : abc ≤ 1 / 9 := by
  sorry

-- Problem (2): Prove (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ (1 / (2 * (abc) ^ (1/2))) given the conditions
theorem problem2 : (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ 1 / (2 * (abc) ^ (1/2)) := by
  sorry

end problem1_problem2_l268_268836


namespace rabbit_speed_correct_l268_268175

-- Define the conditions given in the problem
def rabbit_speed (x : ℝ) : Prop :=
2 * (2 * x + 4) = 188

-- State the main theorem using the defined conditions
theorem rabbit_speed_correct : ∃ x : ℝ, rabbit_speed x ∧ x = 45 :=
by
  sorry

end rabbit_speed_correct_l268_268175


namespace area_of_triangle_ADC_l268_268687

-- Define the constants for the problem
variable (BD DC : ℝ)
variable (abd_area adc_area : ℝ)

-- Given conditions
axiom ratio_condition : BD / DC = 5 / 2
axiom area_abd : abd_area = 35

-- Define the theorem to be proved
theorem area_of_triangle_ADC :
  ∃ adc_area, adc_area = 14 ∧ abd_area / adc_area = BD / DC := 
sorry

end area_of_triangle_ADC_l268_268687


namespace find_digits_sum_l268_268378

theorem find_digits_sum
  (a b c d : ℕ)
  (distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (non_zero : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0)
  (sum_of_numbers : 6 * (a + b + c + d) * 1111 = 73326) :
  {a, b, c, d} = {1, 2, 3, 5} :=
by
  sorry

end find_digits_sum_l268_268378


namespace Jessie_weight_loss_l268_268186

theorem Jessie_weight_loss :
  let initial_weight := 74
  let current_weight := 67
  (initial_weight - current_weight) = 7 :=
by
  sorry

end Jessie_weight_loss_l268_268186


namespace max_m_n_value_l268_268584

theorem max_m_n_value : ∀ (m n : ℝ), (n = -m^2 + 3) → m + n ≤ 13 / 4 :=
by
  intros m n h
  -- The proof will go here, which is omitted for now.
  sorry

end max_m_n_value_l268_268584


namespace tamika_greater_probability_l268_268388

-- Definitions for the conditions
def tamika_results : Set ℕ := {11 * 12, 11 * 13, 12 * 13}
def carlos_result : ℕ := 2 + 3 + 4

-- Theorem stating the problem
theorem tamika_greater_probability : 
  (∀ r ∈ tamika_results, r > carlos_result) → (1 : ℚ) = 1 := 
by
  intros h
  sorry

end tamika_greater_probability_l268_268388


namespace triangle_area_correct_l268_268547

noncomputable def area_of_triangle (A B C : (ℝ × ℝ)) : ℝ :=
  let (x1, y1) := A
  let (x2, y2) := B
  let (x3, y3) := C
  (1 / 2) * abs (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))

theorem triangle_area_correct : 
  area_of_triangle (0, 0) (2, 0) (2, 3) = 3 :=
by
  sorry

end triangle_area_correct_l268_268547


namespace mean_identity_example_l268_268220

theorem mean_identity_example {x y z : ℝ} 
  (h1 : x + y + z = 30)
  (h2 : x * y * z = 343)
  (h3 : x * y + y * z + z * x = 257.25) :
  x^2 + y^2 + z^2 = 385.5 :=
by
  sorry

end mean_identity_example_l268_268220


namespace total_money_from_selling_watermelons_l268_268120

-- Given conditions
def weight_of_one_watermelon : ℝ := 23
def price_per_pound : ℝ := 2
def number_of_watermelons : ℝ := 18

-- Statement to be proved
theorem total_money_from_selling_watermelons : 
  (weight_of_one_watermelon * price_per_pound) * number_of_watermelons = 828 := 
by 
  sorry

end total_money_from_selling_watermelons_l268_268120


namespace find_n_divisible_by_6_l268_268150

theorem find_n_divisible_by_6 (n : Nat) : (71230 + n) % 6 = 0 ↔ n = 2 ∨ n = 8 := by
  sorry

end find_n_divisible_by_6_l268_268150


namespace largest_number_value_l268_268411

theorem largest_number_value (x : ℕ) (h : 7 * x - 3 * x = 40) : 7 * x = 70 :=
by
  sorry

end largest_number_value_l268_268411


namespace problem_1_problem_2_l268_268940

variable {a b c : ℝ}

theorem problem_1 (hp : 0 < a ∧ 0 < b ∧ 0 < c)
    (h1 : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
    abc ≤ 1/9 := 
by 
  apply sorry

theorem problem_2 (hp : 0 < a ∧ 0 < b ∧ 0 < c)
    (h1 : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
    a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * real.sqrt (abc)) := 
by 
  apply sorry

end problem_1_problem_2_l268_268940


namespace friend_decks_l268_268247

-- Definitions for conditions
def price_per_deck : ℕ := 8
def victor_decks : ℕ := 6
def total_spent : ℕ := 64

-- Conclusion based on the conditions
theorem friend_decks : (64 - (6 * 8)) / 8 = 2 := by
  sorry

end friend_decks_l268_268247


namespace part1_part2_l268_268962

def pos_real := {x : ℝ // 0 < x}

variables (a b c : pos_real)

theorem part1 (h : a.val ^ (3/2) + b.val ^ (3/2) + c.val ^ (3/2) = 1) : a.val * b.val * c.val ≤ 1 / 9 :=
sorry

theorem part2 (h : a.val ^ (3/2) + b.val ^ (3/2) + c.val ^ (3/2) = 1) : 
  a.val / (b.val + c.val) + b.val / (a.val + c.val) + c.val / (a.val + b.val) ≤ 1 / (2 * Real.sqrt (a.val * b.val * c.val)) :=
sorry

end part1_part2_l268_268962


namespace Petya_digits_sum_l268_268374

theorem Petya_digits_sum (a b c d : ℕ) (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (h_nonzero : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0) :
  (24 * (a + b + c + d) * 1111 = 73326) →
  {a, b, c, d} = {1, 2, 3, 5} :=
by
  sorry

end Petya_digits_sum_l268_268374


namespace letter_150_in_pattern_l268_268595

-- Define the repeating pattern
def pattern : List Char := ['A', 'B', 'C', 'D']

-- Define the function to get the n-th letter in the infinite repetition of the pattern
def nth_letter_in_pattern (n : Nat) : Char :=
  pattern.get! ((n - 1) % pattern.length)

-- Theorem statement
theorem letter_150_in_pattern : nth_letter_in_pattern 150 = 'B' :=
  sorry

end letter_150_in_pattern_l268_268595


namespace total_apples_packed_correct_l268_268362

-- Define the daily production of apples under normal conditions
def apples_per_box := 40
def boxes_per_day := 50
def days_per_week := 7
def apples_per_day := apples_per_box * boxes_per_day

-- Define the change in daily production for the next week
def fewer_apples := 500
def apples_per_day_next_week := apples_per_day - fewer_apples

-- Define the weekly production in normal and next conditions
def apples_first_week := apples_per_day * days_per_week
def apples_second_week := apples_per_day_next_week * days_per_week

-- Define the total apples packed in two weeks
def total_apples_packed := apples_first_week + apples_second_week

-- Prove the total apples packed is 24500
theorem total_apples_packed_correct : total_apples_packed = 24500 := by
  sorry

end total_apples_packed_correct_l268_268362


namespace problem1_problem2_l268_268903

open Real

variable {a b c : ℝ}

-- Condition: a, b, c are positive and a^{3/2} + b^{3/2} + c^{3/2} = 1
def conditions (a b c : ℝ) := a > 0 ∧ b > 0 ∧ c > 0 ∧ a^(3/2) + b^(3/2) + c^(3/2) = 1

-- Problem 1: Prove that abc <= 1/9
theorem problem1 (ha : conditions a b c) : a * b * c ≤ 1 / 9 :=
sorry

-- Problem 2: Prove that a/(b+c) + b/(a+c) + c/(a+b) <= 1 / (2 * sqrt(abc))
theorem problem2 (ha : conditions a b c) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * sqrt (a * b * c)) :=
sorry

end problem1_problem2_l268_268903


namespace petya_digits_l268_268381

def are_distinct (a b c d : Nat) : Prop :=
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d 

def non_zero_digits (a b c d : Nat) : Prop :=
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0

theorem petya_digits :
  ∃ (a b c d : Nat), are_distinct a b c d ∧ non_zero_digits a b c d ∧ (a + b + c + d = 11) ∧ (a = 1 ∨ a = 2 ∨ a = 3 ∨ a = 5) ∧
  (b = 1 ∨ b = 2 ∨ b = 3 ∨ b = 5) ∧ (c = 1 ∨ c = 2 ∨ c = 3 ∨ c = 5) ∧ (d = 1 ∨ d = 2 ∨ d = 3 ∨ d = 5) :=
by
  sorry

end petya_digits_l268_268381


namespace union_of_A_and_B_l268_268528

def A : Set Int := {-1, 1, 2}
def B : Set Int := {-2, -1, 0}

theorem union_of_A_and_B : A ∪ B = {-2, -1, 0, 1, 2} :=
by
  sorry

end union_of_A_and_B_l268_268528


namespace red_shells_correct_l268_268282

-- Define the conditions
def total_shells : Nat := 291
def green_shells : Nat := 49
def non_red_green_shells : Nat := 166

-- Define the number of red shells as per the given conditions
def red_shells : Nat :=
  total_shells - green_shells - non_red_green_shells

-- State the theorem
theorem red_shells_correct : red_shells = 76 :=
by
  sorry

end red_shells_correct_l268_268282


namespace petya_digits_l268_268380

def are_distinct (a b c d : Nat) : Prop :=
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d 

def non_zero_digits (a b c d : Nat) : Prop :=
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0

theorem petya_digits :
  ∃ (a b c d : Nat), are_distinct a b c d ∧ non_zero_digits a b c d ∧ (a + b + c + d = 11) ∧ (a = 1 ∨ a = 2 ∨ a = 3 ∨ a = 5) ∧
  (b = 1 ∨ b = 2 ∨ b = 3 ∨ b = 5) ∧ (c = 1 ∨ c = 2 ∨ c = 3 ∨ c = 5) ∧ (d = 1 ∨ d = 2 ∨ d = 3 ∨ d = 5) :=
by
  sorry

end petya_digits_l268_268380


namespace hyperbola_min_dist_l268_268653

noncomputable def hyperbola_min_value : ℝ :=
  let a := 3 in
  let b := Real.sqrt 6 in
  let c := Real.sqrt (a^2 + b^2) in -- Focal distance
  2 * b^2 / a + 12

theorem hyperbola_min_dist (A B : ℝ × ℝ) (F1 F2 : ℝ × ℝ)
  (h_hyperbola : ∀ x y, x^2 / 9 - y^2 / 6 = 1)
  (h_F1 : F1 = (-c, 0))
  (h_F2 : F2 = (c, 0))
  (h_line : ∃ k, ∀ p, p ∈ [[A, B]] → p.1 = k * (p.2 - ((-c) + c)/2)) :
  (|A - F2| + |B - F2|) = 16 :=
by {
  sorry
}

end hyperbola_min_dist_l268_268653


namespace fraction_length_EF_of_GH_l268_268058

theorem fraction_length_EF_of_GH (GH GE EH GF FH EF : ℝ)
  (h1 : GE = 3 * EH)
  (h2 : GF = 4 * FH)
  (h3 : GE + EH = GH)
  (h4 : GF + FH = GH) :
  EF / GH = 1 / 20 := by 
  sorry

end fraction_length_EF_of_GH_l268_268058


namespace vertical_throw_time_l268_268149

theorem vertical_throw_time (h v g t : ℝ)
  (h_def: h = v * t - (1/2) * g * t^2)
  (initial_v: v = 25)
  (gravity: g = 10)
  (target_h: h = 20) :
  t = 1 ∨ t = 4 := 
by
  sorry

end vertical_throw_time_l268_268149


namespace largest_four_digit_congruent_17_mod_26_l268_268251

theorem largest_four_digit_congruent_17_mod_26 : 
  ∃ k : ℤ, (26 * k + 17 < 10000) ∧ (1000 ≤ 26 * k + 17) ∧ (26 * k + 17) ≡ 17 [MOD 26] ∧ (26 * k + 17 = 9972) :=
by
  sorry

end largest_four_digit_congruent_17_mod_26_l268_268251


namespace metallic_sphere_radius_l268_268612

theorem metallic_sphere_radius 
  (r_wire : ℝ)
  (h_wire : ℝ)
  (r_sphere : ℝ) 
  (V_sphere : ℝ)
  (V_wire : ℝ)
  (h_wire_eq : h_wire = 16)
  (r_wire_eq : r_wire = 12)
  (V_wire_eq : V_wire = π * r_wire^2 * h_wire)
  (V_sphere_eq : V_sphere = (4/3) * π * r_sphere^3)
  (volume_eq : V_sphere = V_wire) :
  r_sphere = 12 :=
by
  sorry

end metallic_sphere_radius_l268_268612


namespace correct_permutations_ВЕКТОР_correct_permutations_ЛИНИЯ_correct_permutations_ПАРАБОЛА_correct_permutations_БИССЕКТРИСА_correct_permutations_МАТЕМАТИКА_l268_268616

noncomputable def permutations_ВЕКТОР : ℕ := 6!
def answer_ВЕКТОР := 720

noncomputable def permutations_ЛИНИЯ : ℕ := 5! / 2!
def answer_ЛИНИЯ := 60

noncomputable def permutations_ПАРАБОЛА : ℕ := 8! / 3!
def answer_ПАРАБОЛА := 6720

noncomputable def permutations_БИССЕКТРИСА : ℕ := 11! / (3! * 2!)
def answer_БИССЕКТРИСА := 3326400

noncomputable def permutations_МАТЕМАТИКА : ℕ := 10! / (3! * 2! * 2!)
def answer_МАТЕМАТИКА := 151200

theorem correct_permutations_ВЕКТОР : permutations_ВЕКТОР = answer_ВЕКТОР := by
  sorry

theorem correct_permutations_ЛИНИЯ : permutations_ЛИНИЯ = answer_ЛИНИЯ := by
  sorry

theorem correct_permutations_ПАРАБОЛА : permutations_ПАРАБОЛА = answer_ПАРАБОЛА := by
  sorry

theorem correct_permutations_БИССЕКТРИСА : permutations_БИССЕКТРИСА = answer_БИССЕКТРИСА := by
  sorry

theorem correct_permutations_МАТЕМАТИКА : permutations_МАТЕМАТИКА = answer_МАТЕМАТИКА := by
  sorry

end correct_permutations_ВЕКТОР_correct_permutations_ЛИНИЯ_correct_permutations_ПАРАБОЛА_correct_permutations_БИССЕКТРИСА_correct_permutations_МАТЕМАТИКА_l268_268616


namespace problem_1_problem_2_l268_268873

theorem problem_1 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) 
  (h_sum : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  abc ≤ 1 / 9 := 
sorry

theorem problem_2 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) 
  (h_sum : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * real.sqrt (abc)) := 
sorry

end problem_1_problem_2_l268_268873


namespace problem1_problem2_l268_268137

variables (a b : ℝ)

-- Problem 1: Prove that 3a^2 - 6a^2 - a^2 = -4a^2
theorem problem1 : (3 * a^2 - 6 * a^2 - a^2 = -4 * a^2) :=
by sorry

-- Problem 2: Prove that (5a - 3b) - 3(a^2 - 2b) = -3a^2 + 5a + 3b
theorem problem2 : ((5 * a - 3 * b) - 3 * (a^2 - 2 * b) = -3 * a^2 + 5 * a + 3 * b) :=
by sorry

end problem1_problem2_l268_268137


namespace fraction_product_l268_268135

theorem fraction_product :
  (7 / 4) * (8 / 14) * (16 / 24) * (32 / 48) * (28 / 7) * (15 / 9) *
  (50 / 25) * (21 / 35) = 32 / 3 :=
by
  sorry

end fraction_product_l268_268135


namespace batsman_average_l268_268428

theorem batsman_average (avg_20 : ℕ) (avg_10 : ℕ) (total_matches_20 : ℕ) (total_matches_10 : ℕ) :
  avg_20 = 40 → avg_10 = 20 → total_matches_20 = 20 → total_matches_10 = 10 →
  (800 + 200) / 30 = 33.33 :=
by
  sorry

end batsman_average_l268_268428


namespace max_possible_value_l268_268628

theorem max_possible_value (a b : ℝ) (h : ∀ n : ℕ, 1 ≤ n → n ≤ 2008 → a + b = a^n + b^n) :
  ∃ a b, ∀ n : ℕ, 1 ≤ n → n ≤ 2008 → a + b = a^n + b^n → ∃ s : ℝ, (s = 0 ∨ s = 1 ∨ s = 2) →
  max (1 / a^(2009) + 1 / b^(2009)) = 2 :=
sorry

end max_possible_value_l268_268628


namespace cos_double_angle_proof_l268_268480

variable {α β : ℝ}

theorem cos_double_angle_proof (h1 : sin (α - β) = 1 / 3) (h2 : cos α * sin β = 1 / 6) :
  cos (2 * α + 2 * β) = 1 / 9 :=
sorry

end cos_double_angle_proof_l268_268480


namespace all_zero_l268_268208

def circle_condition (x : Fin 2007 → ℤ) : Prop :=
  ∀ i : Fin 2007, x i + x (i+1) + x (i+2) + x (i+3) + x (i+4) = 2 * (x (i+1) + x (i+2)) + 2 * (x (i+3) + x (i+4))

theorem all_zero (x : Fin 2007 → ℤ) (h : circle_condition x) : ∀ i, x i = 0 :=
sorry

end all_zero_l268_268208


namespace problem_part1_problem_part2_l268_268949

variable {a b c : ℝ}

theorem problem_part1 (h : a > 0 ∧ b > 0 ∧ c > 0) (sum_eq : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  abc ≤ 1 / 9 :=
sorry

theorem problem_part2 (h : a > 0 ∧ b > 0 ∧ c > 0) (sum_eq : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (abc)) :=
sorry

end problem_part1_problem_part2_l268_268949


namespace problem_1_problem_2_l268_268098

universe u

/-- Assume the universal set U is the set of real numbers -/
def U : Set ℝ := Set.univ

/-- Define set A -/
def A : Set ℝ := {x : ℝ | x ≥ 1}

/-- Define set B -/
def B : Set ℝ := {x : ℝ | -1 ≤ x ∧ x ≤ 2}

/-- Prove the intersection of A and B -/
theorem problem_1 : (A ∩ B) = {x : ℝ | 1 ≤ x ∧ x ≤ 2} :=
sorry

/-- Prove the complement of the union of A and B -/
theorem problem_2 : (U \ (A ∪ B)) = {x : ℝ | x < -1} :=
sorry

end problem_1_problem_2_l268_268098


namespace cylinder_height_in_hemisphere_l268_268765

theorem cylinder_height_in_hemisphere
  (OA : ℝ)
  (OB : ℝ)
  (r_cylinder : ℝ)
  (r_hemisphere : ℝ)
  (inscribed : r_cylinder = 3 ∧ r_hemisphere = 7)
  (h_OA : OA = r_hemisphere)
  (h_OB : OB = r_cylinder) :
  OA^2 - OB^2 = 40 := by
sory

end cylinder_height_in_hemisphere_l268_268765


namespace part1_part2_l268_268990

noncomputable theory

variables {a b c : ℝ}

-- Condition that a, b, c are positive numbers and the given equation holds
def pos_numbers : Prop := a > 0 ∧ b > 0 ∧ c > 0
def sum_condition : Prop := (a ^ (3/2)) + (b ^ (3/2)) + (c ^ (3/2)) = 1

-- Proof goal for the first part: abc <= 1/9
theorem part1 (h₁ : pos_numbers) (h₂ : sum_condition) : 
  a * b * c ≤ 1 / 9 :=
sorry

-- Proof goal for the second part: a/(b+c) + b/(a+c) + c/(a+b) <= 1/(2 * sqrt(abc))
theorem part2 (h₁ : pos_numbers) (h₂ : sum_condition) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * sqrt (a * b * c)) :=
sorry

end part1_part2_l268_268990


namespace proof_part1_proof_part2_l268_268908

variable {a b c : ℝ}

def pos_and_sum_of_powers (a b c : ℝ) : Prop :=
  0 < a ∧ 0 < b ∧ 0 < c ∧ a^(3/2) + b^(3/2) + c^(3/2) = 1

theorem proof_part1 (h : pos_and_sum_of_powers a b c) : abc ≤ 1 / 9 :=
sorry

theorem proof_part2 (h : pos_and_sum_of_powers a b c) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (abc)) :=
sorry

end proof_part1_proof_part2_l268_268908


namespace derivative_y_l268_268148

open Real

noncomputable def y (x : ℝ) : ℝ :=
  log (2 * x - 3 + sqrt (4 * x ^ 2 - 12 * x + 10)) -
  sqrt (4 * x ^ 2 - 12 * x + 10) * arctan (2 * x - 3)

theorem derivative_y (x : ℝ) : 
  (deriv y x) = - arctan (2 * x - 3) / sqrt (4 * x ^ 2 - 12 * x + 10) :=
by
  sorry

end derivative_y_l268_268148


namespace matchstick_triangles_l268_268084

theorem matchstick_triangles (perimeter : ℕ) (h_perimeter : perimeter = 30) : 
  ∃ n : ℕ, n = 17 ∧ 
  (∀ a b c : ℕ, a + b + c = perimeter → a > 0 → b > 0 → c > 0 → 
                a + b > c ∧ a + c > b ∧ b + c > a → 
                a ≤ b ∧ b ≤ c → n = 17) := 
sorry

end matchstick_triangles_l268_268084


namespace students_calculation_l268_268242

def number_of_stars : ℝ := 3.0
def students_per_star : ℝ := 41.33333333
def total_students : ℝ := 124

theorem students_calculation : number_of_stars * students_per_star = total_students := 
by
  sorry

end students_calculation_l268_268242


namespace supplement_complement_l268_268086

theorem supplement_complement (angle1 angle2 : ℝ) 
  (h_complementary : angle1 + angle2 = 90) : 
   180 - angle1 = 90 + angle2 := by
  sorry

end supplement_complement_l268_268086


namespace not_make_all_numbers_equal_l268_268778

theorem not_make_all_numbers_equal (n : ℕ) (h : n ≥ 3)
  (a : Fin n → ℕ) (h1 : ∃ (i : Fin n), a i = 1 ∧ (∀ (j : Fin n), j ≠ i → a j = 0)) :
  ¬ ∃ x, ∀ i : Fin n, a i = x :=
by
  sorry

end not_make_all_numbers_equal_l268_268778


namespace rabbit_speed_l268_268173

theorem rabbit_speed (s : ℕ) (h : (s * 2 + 4) * 2 = 188) : s = 45 :=
sorry

end rabbit_speed_l268_268173


namespace average_value_of_T_l268_268217

noncomputable def expected_value_T (B G : ℕ) : ℚ :=
  let total_pairs := 19
  let prob_bg := (B / (B + G)) * (G / (B + G))
  2 * total_pairs * prob_bg

theorem average_value_of_T 
  (B G : ℕ) (hB : B = 8) (hG : G = 12) : 
  expected_value_T B G = 9 :=
by
  rw [expected_value_T, hB, hG]
  norm_num
  sorry

end average_value_of_T_l268_268217


namespace bus_speed_excluding_stoppages_l268_268300

theorem bus_speed_excluding_stoppages 
  (V : ℝ) -- Denote the average speed excluding stoppages as V
  (h1 : 30 / 1 = 30) -- condition 1: average speed including stoppages is 30 km/hr
  (h2 : 1 / 2 = 0.5) -- condition 2: The bus is moving for 0.5 hours per hour due to 30 min stoppage
  (h3 : V = 2 * 30) -- from the condition that the bus must cover the distance in half the time
  : V = 60 :=
by {
  sorry -- proof is not required
}

end bus_speed_excluding_stoppages_l268_268300


namespace line_product_l268_268004

theorem line_product (b m : ℝ) (h1: b = -1) (h2: m = 2) : m * b = -2 :=
by
  rw [h1, h2]
  norm_num


end line_product_l268_268004


namespace largest_x_value_l268_268303

theorem largest_x_value (x : ℝ) :
  (x ≠ 9) ∧ (x ≠ -4) ∧ ((x ^ 2 - x - 72) / (x - 9) = 5 / (x + 4)) → x = -3 :=
sorry

end largest_x_value_l268_268303


namespace compare_abc_l268_268320

noncomputable def a : ℝ := Real.log 10 / Real.log 5
noncomputable def b : ℝ := Real.log 12 / Real.log 6
noncomputable def c : ℝ := Real.log 14 / Real.log 7

theorem compare_abc : a > b ∧ b > c := by
  sorry

end compare_abc_l268_268320


namespace joey_more_fish_than_peter_l268_268620

-- Define the conditions
variables (A P J : ℕ)

-- Condition that Ali's fish weight is twice that of Peter's
def ali_double_peter (A P : ℕ) : Prop := A = 2 * P

-- Condition that Ali caught 12 kg of fish
def ali_caught_12 (A : ℕ) : Prop := A = 12

-- Condition that the total weight of the fish is 25 kg
def total_weight (A P J : ℕ) : Prop := A + P + J = 25

-- Prove that Joey caught 1 kg more fish than Peter
theorem joey_more_fish_than_peter (A P J : ℕ) :
  ali_double_peter A P → ali_caught_12 A → total_weight A P J → J = 1 :=
by 
  intro h1 h2 h3
  sorry

end joey_more_fish_than_peter_l268_268620


namespace problem_part1_problem_part2_l268_268947

variable {a b c : ℝ}

theorem problem_part1 (h : a > 0 ∧ b > 0 ∧ c > 0) (sum_eq : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  abc ≤ 1 / 9 :=
sorry

theorem problem_part2 (h : a > 0 ∧ b > 0 ∧ c > 0) (sum_eq : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (abc)) :=
sorry

end problem_part1_problem_part2_l268_268947


namespace son_l268_268110

variable (S M : ℕ)

theorem son's_age
  (h1 : M = S + 24)
  (h2 : M + 2 = 2 * (S + 2))
  : S = 22 :=
sorry

end son_l268_268110


namespace initial_nickels_proof_l268_268414

def initial_nickels (N : ℕ) (D : ℕ) (total_value : ℝ) : Prop :=
  D = 3 * N ∧
  total_value = (N + 2 * N) * 0.05 + 3 * N * 0.10 ∧
  total_value = 9

theorem initial_nickels_proof : ∃ N, ∃ D, (initial_nickels N D 9) → (N = 20) :=
by
  sorry

end initial_nickels_proof_l268_268414


namespace part1_part2_l268_268888

variables {a b c : ℝ}

theorem part1 (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h : a^(3/2) + b^(3/2) + c^(3/2) = 1) : abc ≤ 1/9 := 
sorry

theorem part2 (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (abc)) :=
sorry

end part1_part2_l268_268888


namespace find_coefficients_l268_268305

theorem find_coefficients (a b p q : ℝ) :
    (∀ x : ℝ, (2 * x - 1) ^ 20 - (a * x + b) ^ 20 = (x^2 + p * x + q) ^ 10) →
    a = -2 * b ∧ (b = 1 ∨ b = -1) ∧ p = -1 ∧ q = 1 / 4 :=
by 
    sorry

end find_coefficients_l268_268305


namespace fraction_value_l268_268630

theorem fraction_value :
  2 + (3 / (4 + (5 / 6))) = 76 / 29 :=
by
  sorry

end fraction_value_l268_268630


namespace minimum_mn_l268_268032

noncomputable def f (x : ℝ) (n m : ℝ) : ℝ := Real.log x - n * x + Real.log m + 1

noncomputable def f' (x : ℝ) (n : ℝ) : ℝ := 1/x - n

theorem minimum_mn (m n x_0 : ℝ) (h_m : m > 1) (h_tangent : 2*x_0 - (f x_0 n m) + 1 = 0) :
  mn = e * ((1/x_0 - 1) ^ 2 - 1) :=
sorry

end minimum_mn_l268_268032


namespace possible_measures_for_angle_A_l268_268068

-- Definition of angles A and B, and their relationship
def is_supplementary_angles (A B : ℕ) : Prop := A + B = 180

def is_multiple_of (A B : ℕ) : Prop := ∃ k : ℕ, k ≥ 1 ∧ A = k * B

-- Prove there are 17 possible measures for angle A.
theorem possible_measures_for_angle_A : 
  (∀ (A B : ℕ), (A > 0) ∧ (B > 0) ∧ is_multiple_of A B ∧ is_supplementary_angles A B → 
  A = B * 17) := 
sorry

end possible_measures_for_angle_A_l268_268068


namespace product_of_roots_l268_268161

noncomputable def f (a b c d : ℝ) (x : ℝ) : ℝ :=
  a * x^3 + b * x^2 + c * x + d

noncomputable def f_prime (a b c : ℝ) (x : ℝ) : ℝ :=
  3 * a * x^2 + 2 * b * x + c

theorem product_of_roots (a b c d x₁ x₂ : ℝ) 
  (h1 : f a b c d 0 = 0)
  (h2 : f a b c d x₁ = 0)
  (h3 : f a b c d x₂ = 0)
  (h_ext1 : f_prime a b c 1 = 0)
  (h_ext2 : f_prime a b c 2 = 0) :
  x₁ * x₂ = 6 :=
sorry

end product_of_roots_l268_268161


namespace cosine_of_arcsine_l268_268002

theorem cosine_of_arcsine (h : -1 ≤ (8 : ℝ) / 17 ∧ (8 : ℝ) / 17 ≤ 1) : 
  Real.cos (Real.arcsin (8 / 17)) = 15 / 17 :=
sorry

end cosine_of_arcsine_l268_268002


namespace range_of_a_l268_268023

open Real

noncomputable def f (a e x : ℝ) : ℝ := 2 * a^x - e * x^2

theorem range_of_a (x1 x2 a e : ℝ) (h : 0 < a ∧ a ≠ 1 ∧ x1 < x2) 
    (h1 : ∀ x, (deriv (f a e)) x = 0 → x = x1 ∨ x = x2) 
    (h2 : ∀ x, x1 < x → x < x2 → (iter_deriv (f a e) 2) x > 0) 
    (h3 : ∀ x, x < x1 ∨ x > x2 → (iter_deriv (f a e) 2) x < 0) :
    1 / exp 1 < a ∧ a < 1 :=
sorry

end range_of_a_l268_268023


namespace max_mn_on_parabola_l268_268581

theorem max_mn_on_parabola :
  ∀ m n : ℝ, (n = -m^2 + 3) → (m + n ≤ 13 / 4) :=
by
  sorry

end max_mn_on_parabola_l268_268581


namespace revenue_increase_l268_268545

open Real

theorem revenue_increase
  (P Q : ℝ)
  (hP : 0 < P)
  (hQ : 0 < Q) :
  let R := P * Q
  let P_new := P * 1.60
  let Q_new := Q * 0.65
  let R_new := P_new * Q_new
  (R_new - R) / R * 100 = 4 := by
sorry

end revenue_increase_l268_268545


namespace part1_part2_l268_268997

-- a), b), c are positive real numbers and ${a}^{\frac{3}{2}}+{b}^{\frac{3}{2}}+{c}^{\frac{3}{2}}=1$
variables (a b c : ℝ)
axiom a_pos : 0 < a
axiom b_pos : 0 < b
axiom c_pos : 0 < c
axiom sum_eq_one : a^(3/2) + b^(3/2) + c^(3/2) = 1

-- Prove (1) and (2)
theorem part1 : abc ≤ 1 / 9 := sorry

theorem part2 : a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (abc)) := sorry

end part1_part2_l268_268997


namespace division_remainder_l268_268743

theorem division_remainder (x y : ℕ) (hx : 0 < x) (hy : 0 < y) 
  (hrem : x % y = 3) (hdiv : (x : ℚ) / y = 96.15) : y = 20 :=
sorry

end division_remainder_l268_268743


namespace area_ratio_proof_l268_268275

variables (BE CE DE AE : ℝ)
variables (S_alpha S_beta S_gamma S_delta : ℝ)
variables (x : ℝ)

-- Definitions for the given conditions
def BE_val := 80
def CE_val := 60
def DE_val := 40
def AE_val := 30

-- Expressing the ratios
def S_alpha_ratio := 2
def S_beta_ratio := 2

-- Assuming areas in terms of x
def S_alpha_val := 2 * x
def S_beta_val := 2 * x
def S_delta_val := x
def S_gamma_val := 2 * x

-- Problem statement
theorem area_ratio_proof
  (BE := BE_val)
  (CE := CE_val)
  (DE := DE_val)
  (AE := AE_val)
  (S_alpha := S_alpha_val)
  (S_beta := S_beta_val)
  (S_gamma := S_gamma_val)
  (S_delta := S_delta_val) :
  (S_gamma + S_delta) / (S_alpha + S_beta) = 5 / 4 :=
by
  sorry

end area_ratio_proof_l268_268275


namespace cos_17_pi_over_6_l268_268289

noncomputable def rad_to_deg (r : ℝ) : ℝ := r * 180 / Real.pi

theorem cos_17_pi_over_6 : Real.cos (17 * Real.pi / 6) = - (Real.sqrt 3) / 2 := by
  sorry

end cos_17_pi_over_6_l268_268289


namespace problem_1_problem_2_l268_268923

variable (a b c : ℝ)

theorem problem_1 
  (hp : 0 < a) 
  (hq : 0 < b) 
  (hr : 0 < c) 
  (hs : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  abc ≤ 1/9 := 
sorry

theorem problem_2 
  (hp : 0 < a) 
  (hq : 0 < b) 
  (hr : 0 < c) 
  (hs : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (abc)) := 
sorry

end problem_1_problem_2_l268_268923


namespace hard_hats_remaining_l268_268181

theorem hard_hats_remaining :
  ∀ (initial_pink : ℕ) (initial_green : ℕ) (initial_yellow : ℕ)
    (carl_pink_taken : ℕ) (john_pink_taken : ℕ) (john_green_taken : ℕ),
    initial_pink = 26 → initial_green = 15 → initial_yellow = 24 →
    carl_pink_taken = 4 → john_pink_taken = 6 →
    john_green_taken = 2 * john_pink_taken →
    initial_pink - carl_pink_taken - john_pink_taken + 
    initial_green - john_green_taken +
    initial_yellow = 43 :=
by
  intros
  rw [a_0] at *
  rw [a_1] at *
  rw [a_2] at *
  rw [a_3] at *
  rw [a_4] at *
  rw [a_5] at *
  linarith

example : hard_hats_remaining 26 15 24 4 6 (2 * 6) := rfl 

end hard_hats_remaining_l268_268181


namespace probability_multiple_of_3_or_4_l268_268570

theorem probability_multiple_of_3_or_4 :
  let numbers := {n | 1 ≤ n ∧ n ≤ 30},
      multiples_of_3 := {n | n ∈ numbers ∧ n % 3 = 0},
      multiples_of_4 := {n | n ∈ numbers ∧ n % 4 = 0},
      multiples_of_12 := {n | n ∈ numbers ∧ n % 12 = 0},
      favorable_outcomes := multiples_of_3 ∪ multiples_of_4,
      double_counted_outcomes := multiples_of_12,
      total_favorable_outcomes := set.card favorable_outcomes - set.card double_counted_outcomes,
      total_outcomes := set.card numbers in
  total_favorable_outcomes / total_outcomes = 1 / 2 := by
  sorry

end probability_multiple_of_3_or_4_l268_268570


namespace part1_part2_l268_268994

-- a), b), c are positive real numbers and ${a}^{\frac{3}{2}}+{b}^{\frac{3}{2}}+{c}^{\frac{3}{2}}=1$
variables (a b c : ℝ)
axiom a_pos : 0 < a
axiom b_pos : 0 < b
axiom c_pos : 0 < c
axiom sum_eq_one : a^(3/2) + b^(3/2) + c^(3/2) = 1

-- Prove (1) and (2)
theorem part1 : abc ≤ 1 / 9 := sorry

theorem part2 : a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (abc)) := sorry

end part1_part2_l268_268994


namespace general_integral_solution_l268_268794

theorem general_integral_solution (C : ℝ) (x y : ℝ) :
    cos(y)^2 * (cos(x) / sin(x)) * differentiable_at ℝ x +
    sin(x)^2 * (sin(y) / cos(y)) * differentiable_at ℝ y = 0 ->
    tan(y)^2 - cot(x)^2 = C :=
by
    sorry

end general_integral_solution_l268_268794


namespace sequence_general_term_correctness_l268_268722

def sequenceGeneralTerm (n : ℕ) : ℤ :=
  if n % 2 = 1 then
    0
  else
    (-1) ^ (n / 2 + 1)

theorem sequence_general_term_correctness (n : ℕ) :
  (∀ m, sequenceGeneralTerm m = 0 ↔ m % 2 = 1) ∧
  (∀ k, sequenceGeneralTerm k = (-1) ^ (k / 2 + 1) ↔ k % 2 = 0) :=
by
  sorry

end sequence_general_term_correctness_l268_268722


namespace carl_cost_l268_268293

theorem carl_cost (property_damage medical_bills : ℝ) (insurance_coverage : ℝ) (carl_coverage : ℝ) (H1 : property_damage = 40000) (H2 : medical_bills = 70000) (H3 : insurance_coverage = 0.80) (H4 : carl_coverage = 0.20) :
  carl_coverage * (property_damage + medical_bills) = 22000 :=
by
  sorry

end carl_cost_l268_268293


namespace vasya_lowest_position_l268_268227

theorem vasya_lowest_position
  (n : ℕ) (m : ℕ) (num_cyclists : ℕ) (vasya_place : ℕ) :
  num_cyclists = 500 →
  n = 15 →
  vasya_place = 7 →
  ∀ (stages : fin n → fin num_cyclists) (no_identical_times : ∀ i j : fin n, i ≠ j → 
  ∀ k l : fin num_cyclists, stages i k ≠ stages j l),
  ∃ (lowest_position : ℕ), lowest_position = 91 := 
by sorry

end vasya_lowest_position_l268_268227


namespace part1_part2_l268_268977

variable (a b c : ℝ)

-- Condition: a, b, c are positive numbers and a^(3/2) + b^(3/2) + c^(3/2) = 1
axiom pos_a : a > 0
axiom pos_b : b > 0
axiom pos_c : c > 0
axiom sum_condition : a^(3/2) + b^(3/2) + c^(3/2) = 1

-- Proof Problem (1): abc ≤ 1/9
theorem part1 : abc ≤ 1/9 := 
by {
  have h := sorry,
  exact h
}

-- Proof Problem (2): a/(b+c) + b/(a+c) + c/(a+b) ≤ 1/(2sqrt(abc))
theorem part2 : a/(b+c) + b/(a+c) + c/(a+b) ≤ 1/(2*sqrt(abc)) := 
by {
  have h := sorry,
  exact h
}

end part1_part2_l268_268977


namespace sin_double_angle_series_l268_268671

theorem sin_double_angle_series (θ : ℝ) (h : ∑' (n : ℕ), (sin θ)^(2 * n) = 3) :
  sin (2 * θ) = (2 * sqrt 2) / 3 :=
sorry

end sin_double_angle_series_l268_268671


namespace Mark_marbles_correct_l268_268003

def Connie_marbles : ℕ := 323
def Juan_marbles : ℕ := Connie_marbles + 175
def Mark_marbles : ℕ := 3 * Juan_marbles

theorem Mark_marbles_correct : Mark_marbles = 1494 := 
by
  sorry

end Mark_marbles_correct_l268_268003


namespace least_x_value_l268_268680

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem least_x_value (x : ℕ) (p : ℕ) (hp : is_prime p) (h : x / (12 * p) = 2) : x = 48 := by
  sorry

end least_x_value_l268_268680


namespace vasya_lowest_position_l268_268232

noncomputable theory

def number_of_cyclists := 500
def number_of_stages := 15
def position_of_vasya_each_stage := 7

theorem vasya_lowest_position (total_cyclists : ℕ) (stages : ℕ) (position_each_stage : ℕ)
  (h_total_cyclists : total_cyclists = number_of_cyclists)
  (h_stages : stages = number_of_stages)
  (h_position_each_stage : position_each_stage = position_of_vasya_each_stage)
  (no_identical_times : ∀ (i j : ℕ), i ≠ j → ∀ (stage : ℕ), stage ≤ stages → ∀ (t : ℕ), t ≤ total_cyclists → 
    (time : Π (s : ℕ) (c : ℕ), c < total_cyclists → ℕ), 
    time stage i < time stage j ∨ time stage j < time stage i):
  ∃ lowest_pos, lowest_pos = 91 := sorry

end vasya_lowest_position_l268_268232


namespace problem_1_problem_2_l268_268941

variable {a b c : ℝ}

theorem problem_1 (hp : 0 < a ∧ 0 < b ∧ 0 < c)
    (h1 : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
    abc ≤ 1/9 := 
by 
  apply sorry

theorem problem_2 (hp : 0 < a ∧ 0 < b ∧ 0 < c)
    (h1 : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
    a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * real.sqrt (abc)) := 
by 
  apply sorry

end problem_1_problem_2_l268_268941


namespace independence_test_result_l268_268344

noncomputable def independence_test (P : ℝ → ℝ) (H0 : Prop) (X Y : Type) : Prop :=
  (∀ p, P p = 0.001) → H0 → (1 - P 10.83) = 0.999

theorem independence_test_result (P : ℝ → ℝ) (H0 : Prop) (X Y : Type) :
  (∀ p, P p = 0.001) → 
  H0 → 
  (1 - P 10.83) = 0.999 :=
by
  intro hP hH0
  have : P 10.83 = 0.001 := hP 10.83
  rw [this]
  norm_num

end independence_test_result_l268_268344


namespace abc_le_one_ninth_sum_ratios_le_one_over_two_sqrt_abc_l268_268821

open Real

variables (a b c : ℝ)

-- Condition: a, b, c are positive numbers
variables (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
-- Condition: a^(3/2) + b^(3/2) + c^(3/2) = 1
variable  (h_sum : a^(3/2) + b^(3/2) + c^(3/2) = 1)

-- Question 1: Prove abc ≤ 1/9
theorem abc_le_one_ninth : a * b * c ≤ 1 / 9 :=
  sorry

-- Question 2: Prove (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ 1 / (2 * sqrt (a * b * c))
theorem sum_ratios_le_one_over_two_sqrt_abc : 
  (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ 1 / (2 * sqrt (a * b * c)) :=
  sorry

end abc_le_one_ninth_sum_ratios_le_one_over_two_sqrt_abc_l268_268821


namespace proof_part1_proof_part2_l268_268907

variable {a b c : ℝ}

def pos_and_sum_of_powers (a b c : ℝ) : Prop :=
  0 < a ∧ 0 < b ∧ 0 < c ∧ a^(3/2) + b^(3/2) + c^(3/2) = 1

theorem proof_part1 (h : pos_and_sum_of_powers a b c) : abc ≤ 1 / 9 :=
sorry

theorem proof_part2 (h : pos_and_sum_of_powers a b c) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (abc)) :=
sorry

end proof_part1_proof_part2_l268_268907


namespace remainder_when_divided_by_15_l268_268106

def N (k : ℤ) : ℤ := 35 * k + 25

theorem remainder_when_divided_by_15 (k : ℤ) : (N k) % 15 = 10 := 
by 
  -- proof would go here
  sorry

end remainder_when_divided_by_15_l268_268106


namespace count_non_decreasing_digits_of_12022_l268_268276

/-- Proof that the number of digits left in the number 12022 that form a non-decreasing sequence is 3. -/
theorem count_non_decreasing_digits_of_12022 : 
  let num := [1, 2, 0, 2, 2]
  let remaining := [1, 2, 2] -- non-decreasing sequence from 12022
  List.length remaining = 3 :=
by
  let num := [1, 2, 0, 2, 2]
  let remaining := [1, 2, 2]
  have h : List.length remaining = 3 := rfl
  exact h

end count_non_decreasing_digits_of_12022_l268_268276


namespace no_matching_option_for_fraction_l268_268196

theorem no_matching_option_for_fraction (m n : ℕ) (h : m = 16 ^ 500) : 
  (m / 8 ≠ 8 ^ 499) ∧ 
  (m / 8 ≠ 4 ^ 999) ∧ 
  (m / 8 ≠ 2 ^ 1998) ∧ 
  (m / 8 ≠ 4 ^ 498) ∧ 
  (m / 8 ≠ 2 ^ 1994) := 
by {
  sorry
}

end no_matching_option_for_fraction_l268_268196


namespace hyperbola_asymptote_l268_268160

theorem hyperbola_asymptote (a : ℝ) (h : a > 0) (hyp_eq : ∀ x y : ℝ, (x^2) / (a^2) - (y^2) / 81 = 1 → y = 3 * x) : a = 3 := 
by
  sorry

end hyperbola_asymptote_l268_268160


namespace determinant_zero_l268_268007

def matrix_determinant (x y z : ℝ) : ℝ :=
  Matrix.det ![
    ![1, x, y + z],
    ![1, x + y, z],
    ![1, x + z, y]
  ]

theorem determinant_zero (x y z : ℝ) : matrix_determinant x y z = 0 := 
by
  sorry

end determinant_zero_l268_268007


namespace solve_part_a_solve_part_b_solve_part_c_l268_268213

-- Part (a)
theorem solve_part_a (x : ℝ) : 
  (2 * x^2 + 3 * x - 1)^2 - 5 * (2 * x^2 + 3 * x + 3) + 24 = 0 ↔ 
  x = 1 ∨ x = -2 ∨ x = 0.5 ∨ x = -2.5 := sorry

-- Part (b)
theorem solve_part_b (x : ℝ) : 
  (x - 1) * (x + 3) * (x + 4) * (x + 8) = -96 ↔ 
  x = 0 ∨ x = -7 ∨ x = (-7 + Real.sqrt 33) / 2 ∨ x = (-7 - Real.sqrt 33) / 2 := sorry

-- Part (c)
theorem solve_part_c (x : ℝ) (hx : x ≠ 0) : 
  (x - 1) * (x - 2) * (x - 4) * (x - 8) = 4 * x^2 ↔ 
  x = 4 + 2 * Real.sqrt 2 ∨ x = 4 - 2 * Real.sqrt 2 := sorry

end solve_part_a_solve_part_b_solve_part_c_l268_268213


namespace Petya_digits_sum_l268_268375

theorem Petya_digits_sum (a b c d : ℕ) (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (h_nonzero : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0) :
  (24 * (a + b + c + d) * 1111 = 73326) →
  {a, b, c, d} = {1, 2, 3, 5} :=
by
  sorry

end Petya_digits_sum_l268_268375


namespace original_average_l268_268555

theorem original_average (A : ℝ)
  (h : 2 * A = 160) : A = 80 :=
by sorry

end original_average_l268_268555


namespace tournament_game_count_l268_268047

/-- In a tournament with 25 players where each player plays 4 games against each other,
prove that the total number of games played is 1200. -/
theorem tournament_game_count : 
  let n := 25
  let games_per_pair := 4
  let total_games := (n * (n - 1) / 2) * games_per_pair
  total_games = 1200 :=
by
  -- Definitions based on the conditions
  let n := 25
  let games_per_pair := 4

  -- Calculating the total number of games
  let total_games := (n * (n - 1) / 2) * games_per_pair

  -- This is the main goal to prove
  have h : total_games = 1200 := sorry
  exact h

end tournament_game_count_l268_268047


namespace determine_N_l268_268798

theorem determine_N (N : ℕ) : (Nat.choose N 5 = 3003) ↔ (N = 15) :=
by
  sorry

end determine_N_l268_268798


namespace compute_expression_l268_268782

theorem compute_expression : 2 + 5 * 3 - 4 + 6 * 2 / 3 = 17 :=
by
  sorry

end compute_expression_l268_268782


namespace cosine_identity_l268_268482

theorem cosine_identity
  (α β : ℝ)
  (h1 : Real.sin (α - β) = 1 / 3)
  (h2 : Real.cos α * Real.sin β = 1 / 6) :
  Real.cos (2 * α + 2 * β) = 1 / 9 :=
  sorry

end cosine_identity_l268_268482


namespace sum_of_b_for_one_solution_l268_268470

theorem sum_of_b_for_one_solution (b : ℝ) (has_single_solution : ∃ x, 3 * x^2 + (b + 12) * x + 11 = 0) :
  ∃ b₁ b₂ : ℝ, (3 * x^2 + (b + 12) * x + 11) = 0 ∧ b₁ + b₂ = -24 := by
  sorry

end sum_of_b_for_one_solution_l268_268470


namespace cosine_identity_l268_268484

theorem cosine_identity
  (α β : ℝ)
  (h1 : Real.sin (α - β) = 1 / 3)
  (h2 : Real.cos α * Real.sin β = 1 / 6) :
  Real.cos (2 * α + 2 * β) = 1 / 9 :=
  sorry

end cosine_identity_l268_268484


namespace max_value_a_l268_268660

noncomputable def setA (a : ℝ) : Set ℝ := { x | (x - 1) * (x - a) ≥ 0 }
noncomputable def setB (a : ℝ) : Set ℝ := { x | x ≥ a - 1 }

theorem max_value_a (a : ℝ) :
  (setA a ∪ setB a = Set.univ) → a ≤ 2 := by
  sorry

end max_value_a_l268_268660


namespace ellipse_range_of_k_l268_268323

theorem ellipse_range_of_k (k : ℝ) :
  (∃ (eq : ((x y : ℝ) → (x ^ 2 / (3 + k) + y ^ 2 / (2 - k) = 1))),
  ((3 + k > 0) ∧ (2 - k > 0) ∧ (3 + k ≠ 2 - k))) ↔
  (k ∈ Set.Ioo (-3 : ℝ) ((-1) / 2) ∪ Set.Ioo ((-1) / 2) 2) :=
by sorry

end ellipse_range_of_k_l268_268323


namespace factor_difference_of_squares_l268_268421

theorem factor_difference_of_squares (a b : ℝ) : 
    (∃ A B : ℝ, a = A ∧ b = B) → 
    (a^2 - b^2 = (a + b) * (a - b)) :=
by
  intros h
  sorry

end factor_difference_of_squares_l268_268421


namespace part_a_part_b_l268_268358

-- Define what it means for a number to be "surtido"
def is_surtido (A : ℕ) : Prop :=
  ∀ n, (1 ≤ n → n ≤ (A.digits 10).sum → ∃ B : ℕ, n = (B.digits 10).sum) 

-- Part (a): Prove that if 1, 2, 3, 4, 5, 6, 7, and 8 can be expressed as sums of digits in A, then A is "surtido".
theorem part_a (A : ℕ)
  (h1 : ∃ B1 : ℕ, 1 = (B1.digits 10).sum)
  (h2 : ∃ B2 : ℕ, 2 = (B2.digits 10).sum)
  (h3 : ∃ B3 : ℕ, 3 = (B3.digits 10).sum)
  (h4 : ∃ B4 : ℕ, 4 = (B4.digits 10).sum)
  (h5 : ∃ B5 : ℕ, 5 = (B5.digits 10).sum)
  (h6 : ∃ B6 : ℕ, 6 = (B6.digits 10).sum)
  (h7 : ∃ B7 : ℕ, 7 = (B7.digits 10).sum)
  (h8 : ∃ B8 : ℕ, 8 = (B8.digits 10).sum) : is_surtido A :=
sorry

-- Part (b): Determine if having the sums 1, 2, 3, 4, 5, 6, and 7 as sums of digits in A implies that A is "surtido".
theorem part_b (A : ℕ)
  (h1 : ∃ B1 : ℕ, 1 = (B1.digits 10).sum)
  (h2 : ∃ B2 : ℕ, 2 = (B2.digits 10).sum)
  (h3 : ∃ B3 : ℕ, 3 = (B3.digits 10).sum)
  (h4 : ∃ B4 : ℕ, 4 = (B4.digits 10).sum)
  (h5 : ∃ B5 : ℕ, 5 = (B5.digits 10).sum)
  (h6 : ∃ B6 : ℕ, 6 = (B6.digits 10).sum)
  (h7 : ∃ B7 : ℕ, 7 = (B7.digits 10).sum) : ¬is_surtido A :=
sorry

end part_a_part_b_l268_268358


namespace solve_polynomial_equation_l268_268717

theorem solve_polynomial_equation :
  ∃ z, (z^5 + 40 * z^3 + 80 * z - 32 = 0) →
  ∃ x, (x = z + 4) ∧ ((x - 2)^5 + (x - 6)^5 = 32) :=
by
  sorry

end solve_polynomial_equation_l268_268717


namespace petya_digits_l268_268376

theorem petya_digits (a b c d : ℕ) (distinct_digits : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (non_zero : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0)
  (sum_formed_numbers : 6 * (a + b + c + d) * 1111 = 73326) :
  {a, b, c, d} = {1, 2, 3, 5} :=
by 
  sorry

end petya_digits_l268_268376


namespace find_position_2002_l268_268713

def T (n : ℕ) : ℕ := n * (n + 1) / 2
def a (n : ℕ) : ℕ := T n + 1

theorem find_position_2002 : ∃ row col : ℕ, 1 ≤ row ∧ 1 ≤ col ∧ (a (row - 1) + (col - 1) = 2002 ∧ row = 15 ∧ col = 49) := 
sorry

end find_position_2002_l268_268713


namespace problem1_problem2_l268_268901

open Real

variable {a b c : ℝ}

-- Condition: a, b, c are positive and a^{3/2} + b^{3/2} + c^{3/2} = 1
def conditions (a b c : ℝ) := a > 0 ∧ b > 0 ∧ c > 0 ∧ a^(3/2) + b^(3/2) + c^(3/2) = 1

-- Problem 1: Prove that abc <= 1/9
theorem problem1 (ha : conditions a b c) : a * b * c ≤ 1 / 9 :=
sorry

-- Problem 2: Prove that a/(b+c) + b/(a+c) + c/(a+b) <= 1 / (2 * sqrt(abc))
theorem problem2 (ha : conditions a b c) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * sqrt (a * b * c)) :=
sorry

end problem1_problem2_l268_268901


namespace part1_part2_l268_268869

variable (a b c : ℝ)

theorem part1 (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) : abc ≤ 1/9 :=
by
  sorry

theorem part2 (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
    (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ 1 / (2 * sqrt(abc)) :=
by
  sorry

end part1_part2_l268_268869


namespace expected_value_replacement_seeds_is_200_l268_268433

noncomputable def germination_prob : ℝ := 0.9
noncomputable def num_seeds_sown : ℕ := 1000
noncomputable def replacement_factor : ℕ := 2

def prob_distribution : distribution := binomial num_seeds_sown (1 - germination_prob)

def expected_num_replacement_seeds : ℝ :=
2 * (num_seeds_sown * (1 - germination_prob))

theorem expected_value_replacement_seeds_is_200 :
  expected_num_replacement_seeds = 200 :=
by
  sorry

end expected_value_replacement_seeds_is_200_l268_268433


namespace part1_part2_l268_268984

noncomputable theory

variables {a b c : ℝ}

-- Condition that a, b, c are positive numbers and the given equation holds
def pos_numbers : Prop := a > 0 ∧ b > 0 ∧ c > 0
def sum_condition : Prop := (a ^ (3/2)) + (b ^ (3/2)) + (c ^ (3/2)) = 1

-- Proof goal for the first part: abc <= 1/9
theorem part1 (h₁ : pos_numbers) (h₂ : sum_condition) : 
  a * b * c ≤ 1 / 9 :=
sorry

-- Proof goal for the second part: a/(b+c) + b/(a+c) + c/(a+b) <= 1/(2 * sqrt(abc))
theorem part2 (h₁ : pos_numbers) (h₂ : sum_condition) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * sqrt (a * b * c)) :=
sorry

end part1_part2_l268_268984


namespace Tammy_earnings_3_weeks_l268_268551

theorem Tammy_earnings_3_weeks
  (trees : ℕ)
  (oranges_per_tree_per_day : ℕ)
  (oranges_per_pack : ℕ)
  (price_per_pack : ℕ)
  (weeks : ℕ) :
  trees = 10 →
  oranges_per_tree_per_day = 12 →
  oranges_per_pack = 6 →
  price_per_pack = 2 →
  weeks = 3 →
  (trees * oranges_per_tree_per_day * weeks * 7) / oranges_per_pack * price_per_pack = 840 :=
by
  intro ht ht12 h6 h2 h3
  -- proof to be filled in here
  sorry

end Tammy_earnings_3_weeks_l268_268551


namespace Liam_savings_after_trip_and_bills_l268_268537

theorem Liam_savings_after_trip_and_bills :
  let trip_cost := 7000
  let bills_cost := 3500
  let monthly_savings := 500
  let years := 2
  let total_savings := monthly_savings * 12 * years
  total_savings - bills_cost - trip_cost = 1500 := by
  let trip_cost := 7000
  let bills_cost := 3500
  let monthly_savings := 500
  let years := 2
  let total_savings := monthly_savings * 12 * years
  sorry

end Liam_savings_after_trip_and_bills_l268_268537


namespace range_of_m_l268_268352

theorem range_of_m (m : ℝ) (h : 0 < m)
  (subset_cond : ∀ x y : ℝ, x - 4 ≤ 0 → y ≥ 0 → mx - y ≥ 0 → (x - 2)^2 + (y - 2)^2 ≤ 8) :
  m ≤ 1 :=
sorry

end range_of_m_l268_268352


namespace part1_part2_l268_268968

variable (a b c : ℝ)

-- Condition: a, b, c are positive numbers and a^(3/2) + b^(3/2) + c^(3/2) = 1
axiom pos_a : a > 0
axiom pos_b : b > 0
axiom pos_c : c > 0
axiom sum_condition : a^(3/2) + b^(3/2) + c^(3/2) = 1

-- Proof Problem (1): abc ≤ 1/9
theorem part1 : abc ≤ 1/9 := 
by {
  have h := sorry,
  exact h
}

-- Proof Problem (2): a/(b+c) + b/(a+c) + c/(a+b) ≤ 1/(2sqrt(abc))
theorem part2 : a/(b+c) + b/(a+c) + c/(a+b) ≤ 1/(2*sqrt(abc)) := 
by {
  have h := sorry,
  exact h
}

end part1_part2_l268_268968


namespace problem1_problem2_l268_268840

variable {a b c : ℝ}

-- Conditions: a, b, c are positive numbers and a^(3/2) + b^(3/2) + c^(3/2) = 1
axiom pos_a : 0 < a
axiom pos_b : 0 < b
axiom pos_c : 0 < c
axiom sum_pow : a^(3/2) + b^(3/2) + c^(3/2) = 1

-- Problem (1): Prove abc ≤ 1/9 given the conditions
theorem problem1 : abc ≤ 1 / 9 := by
  sorry

-- Problem (2): Prove (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ (1 / (2 * (abc) ^ (1/2))) given the conditions
theorem problem2 : (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ 1 / (2 * (abc) ^ (1/2)) := by
  sorry

end problem1_problem2_l268_268840


namespace problem1_problem2_l268_268843

variable {a b c : ℝ}

-- Conditions: a, b, c are positive numbers and a^(3/2) + b^(3/2) + c^(3/2) = 1
axiom pos_a : 0 < a
axiom pos_b : 0 < b
axiom pos_c : 0 < c
axiom sum_pow : a^(3/2) + b^(3/2) + c^(3/2) = 1

-- Problem (1): Prove abc ≤ 1/9 given the conditions
theorem problem1 : abc ≤ 1 / 9 := by
  sorry

-- Problem (2): Prove (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ (1 / (2 * (abc) ^ (1/2))) given the conditions
theorem problem2 : (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ 1 / (2 * (abc) ^ (1/2)) := by
  sorry

end problem1_problem2_l268_268843


namespace part1_part2_l268_268891

variables {a b c : ℝ}

theorem part1 (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h : a^(3/2) + b^(3/2) + c^(3/2) = 1) : abc ≤ 1/9 := 
sorry

theorem part2 (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (abc)) :=
sorry

end part1_part2_l268_268891


namespace part1_part2_l268_268998

-- a), b), c are positive real numbers and ${a}^{\frac{3}{2}}+{b}^{\frac{3}{2}}+{c}^{\frac{3}{2}}=1$
variables (a b c : ℝ)
axiom a_pos : 0 < a
axiom b_pos : 0 < b
axiom c_pos : 0 < c
axiom sum_eq_one : a^(3/2) + b^(3/2) + c^(3/2) = 1

-- Prove (1) and (2)
theorem part1 : abc ≤ 1 / 9 := sorry

theorem part2 : a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (abc)) := sorry

end part1_part2_l268_268998


namespace hyperbola_ratio_l268_268327

theorem hyperbola_ratio (a b c : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h_hyperbola : ∀ x y, (x^2 / a^2) - (y^2 / b^2) = 1)
  (h_foci_distance : c^2 = a^2 + b^2)
  (h_midpoint_on_hyperbola : ∀ x y, 
    (x, y) = (-(c / 2), c / 2) → ∃ (k l : ℝ), (k^2 / a^2) - (l^2 / b^2) = 1) :
  c / a = (Real.sqrt 10 + Real.sqrt 2) / 2 := 
sorry

end hyperbola_ratio_l268_268327


namespace rectangle_area_l268_268396

theorem rectangle_area (b l: ℕ) (h1: l = 3 * b) (h2: 2 * (l + b) = 120) : l * b = 675 := 
by 
  sorry

end rectangle_area_l268_268396


namespace a_alone_completes_in_eight_days_l268_268422

variable (a b : Type)
variables (days_ab : ℝ) (days_a : ℝ) (days_ab_2 : ℝ)

noncomputable def days := ℝ

axiom work_together_four_days : days_ab = 4
axiom work_together_266666_days : days_ab_2 = 8 / 3

theorem a_alone_completes_in_eight_days (a b : Type) (days_ab : ℝ) (days_a : ℝ) (days_ab_2 : ℝ)
  (work_together_four_days : days_ab = 4)
  (work_together_266666_days : days_ab_2 = 8 / 3) :
  days_a = 8 :=
by
  sorry

end a_alone_completes_in_eight_days_l268_268422


namespace rate_per_square_meter_l268_268562

-- Define the conditions
def length (L : ℝ) := L = 8
def width (W : ℝ) := W = 4.75
def total_cost (C : ℝ) := C = 34200
def area (A : ℝ) (L W : ℝ) := A = L * W
def rate (R C A : ℝ) := R = C / A

-- The theorem to prove
theorem rate_per_square_meter (L W C A R : ℝ) 
  (hL : length L) (hW : width W) (hC : total_cost C) (hA : area A L W) : 
  rate R C A :=
by
  -- By the conditions, length is 8, width is 4.75, and total cost is 34200.
  simp [length, width, total_cost, area, rate] at hL hW hC hA ⊢
  -- It remains to calculate the rate and use conditions
  have hA : A = L * W := hA
  rw [hL, hW] at hA
  have hA' : A = 8 * 4.75 := by simp [hA]
  rw [hA']
  simp [rate]
  sorry -- The detailed proof is omitted.

end rate_per_square_meter_l268_268562


namespace initial_pineapple_sweets_l268_268755

-- Define constants for initial number of flavored sweets and actions taken
def initial_cherry_sweets : ℕ := 30
def initial_strawberry_sweets : ℕ := 40
def total_remaining_sweets : ℕ := 55

-- Define Aaron's actions
def aaron_eats_half_sweets (n : ℕ) : ℕ := n / 2
def aaron_gives_to_friend : ℕ := 5

-- Calculate remaining sweets after Aaron's actions
def remaining_cherry_sweets : ℕ := initial_cherry_sweets - (aaron_eats_half_sweets initial_cherry_sweets) - aaron_gives_to_friend
def remaining_strawberry_sweets : ℕ := initial_strawberry_sweets - (aaron_eats_half_sweets initial_strawberry_sweets)

-- Define the problem to prove
theorem initial_pineapple_sweets :
  (total_remaining_sweets - (remaining_cherry_sweets + remaining_strawberry_sweets)) * 2 = 50 :=
by sorry -- Placeholder for the actual proof

end initial_pineapple_sweets_l268_268755


namespace space_shuttle_new_orbital_speed_l268_268114

noncomputable def new_orbital_speed (v_1 : ℝ) (delta_v : ℝ) : ℝ :=
  let v_new := v_1 + delta_v
  v_new * 3600

theorem space_shuttle_new_orbital_speed : 
  new_orbital_speed 2 (500 / 1000) = 9000 :=
by 
  sorry

end space_shuttle_new_orbital_speed_l268_268114


namespace probability_parallel_vectors_l268_268661

def vector_a : ℝ × ℝ := (2, 1)
def vector_b (x y : ℝ) : ℝ × ℝ := (x, y)
def x_values : Set ℝ := {-1, 0, 1, 2}
def y_values : Set ℝ := {-1, 0, 1}

def is_parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem probability_parallel_vectors :
  (finset.filter
    (λ (p : ℝ × ℝ), is_parallel vector_a p)
    (finset.cartesianProduct
      (finset.of_set x_values)
      (finset.of_set y_values))
  ).card.to_real /
  ((finset.cartesianProduct 
      (finset.of_set x_values)
      (finset.of_set y_values)
  ).card.to_real) = 1 / 6 :=
sorry

end probability_parallel_vectors_l268_268661


namespace problem_1_problem_2_l268_268937

variable {a b c : ℝ}

theorem problem_1 (hp : 0 < a ∧ 0 < b ∧ 0 < c)
    (h1 : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
    abc ≤ 1/9 := 
by 
  apply sorry

theorem problem_2 (hp : 0 < a ∧ 0 < b ∧ 0 < c)
    (h1 : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
    a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * real.sqrt (abc)) := 
by 
  apply sorry

end problem_1_problem_2_l268_268937


namespace john_total_distance_l268_268693

theorem john_total_distance :
  let speed1 := 35
  let time1 := 2
  let distance1 := speed1 * time1

  let speed2 := 55
  let time2 := 3
  let distance2 := speed2 * time2

  let total_distance := distance1 + distance2

  total_distance = 235 := by
    sorry

end john_total_distance_l268_268693


namespace intersection_M_N_l268_268710

def M : Set ℤ := {0, 1, 2}
def N : Set ℤ := {x | x^2 - 3*x + 2 ≤ 0}

theorem intersection_M_N:
  M ∩ N = {1, 2} :=
sorry

end intersection_M_N_l268_268710


namespace proof_part1_proof_part2_l268_268913

variable {a b c : ℝ}

def pos_and_sum_of_powers (a b c : ℝ) : Prop :=
  0 < a ∧ 0 < b ∧ 0 < c ∧ a^(3/2) + b^(3/2) + c^(3/2) = 1

theorem proof_part1 (h : pos_and_sum_of_powers a b c) : abc ≤ 1 / 9 :=
sorry

theorem proof_part2 (h : pos_and_sum_of_powers a b c) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (abc)) :=
sorry

end proof_part1_proof_part2_l268_268913


namespace principal_sum_l268_268446

theorem principal_sum (R P : ℝ) (h : (P * (R + 3) * 3) / 100 = (P * R * 3) / 100 + 81) : P = 900 :=
by
  sorry

end principal_sum_l268_268446


namespace point_B_third_quadrant_l268_268045

theorem point_B_third_quadrant (m n : ℝ) (hm : m < 0) (hn : n < 0) :
  (-m * n < 0) ∧ (m < 0) :=
by
  sorry

end point_B_third_quadrant_l268_268045


namespace carX_travel_distance_after_carY_started_l268_268626

-- Define the conditions
def carX_speed : ℝ := 35
def carY_speed : ℝ := 40
def delay_time : ℝ := 1.2

-- Define the problem to prove the question is equal to the correct answer given the conditions
theorem carX_travel_distance_after_carY_started : 
  ∃ t : ℝ, carY_speed * t = carX_speed * t + carX_speed * delay_time ∧ 
           carX_speed * t = 294 :=
by
  sorry

end carX_travel_distance_after_carY_started_l268_268626


namespace sin_alpha_at_point_l268_268517

open Real

theorem sin_alpha_at_point (α : ℝ) (P : ℝ × ℝ) (hP : P = (1, -2)) :
  sin α = -2 * sqrt 5 / 5 :=
sorry

end sin_alpha_at_point_l268_268517


namespace distance_eq_l268_268164

open Real

variables (a b c d p q: ℝ)

-- Conditions from step a)
def onLine1 : Prop := b = (p-1)*a + q
def onLine2 : Prop := d = (p-1)*c + q

-- Theorem about the distance between points (a, b) and (c, d)
theorem distance_eq : 
  onLine1 a b p q → 
  onLine2 c d p q → 
  dist (a, b) (c, d) = abs (a - c) * sqrt (1 + (p - 1)^2) := 
by
  intros h1 h2
  sorry

end distance_eq_l268_268164


namespace least_positive_integer_k_l268_268635

noncomputable def least_k (a : ℝ) (n : ℕ) : ℝ :=
  (1 : ℝ) / ((n + 1 : ℝ) ^ 3)

theorem least_positive_integer_k :
  ∃ k : ℕ , (∀ a : ℝ, ∀ n : ℕ,
  (0 ≤ a ∧ a ≤ 1) → (a^k * (1 - a)^n < least_k a n)) ∧
  (∀ k' : ℕ, k' < 4 → ¬(∀ a : ℝ, ∀ n : ℕ, (0 ≤ a ∧ a ≤ 1) → (a^k' * (1 - a)^n < least_k a n))) :=
sorry

end least_positive_integer_k_l268_268635


namespace intersection_A_B_l268_268499

-- Definitions based on the conditions
def A : Set ℝ := {x | x ≥ 0}
def B : Set ℤ := {x | -2 < x ∧ x < 2}

-- Proof statement
theorem intersection_A_B :
  (A ∩ (B : Set ℝ)) = ({0, 1} : Set ℝ) :=
by
  sorry

end intersection_A_B_l268_268499


namespace ball_hits_ground_l268_268065

theorem ball_hits_ground : 
  ∃ t : ℚ, -4.9 * t^2 + 4 * t + 10 = 0 ∧ t = 10 / 7 :=
by sorry

end ball_hits_ground_l268_268065


namespace problem1_problem2_l268_268896

open Real

variable {a b c : ℝ}

-- Condition: a, b, c are positive and a^{3/2} + b^{3/2} + c^{3/2} = 1
def conditions (a b c : ℝ) := a > 0 ∧ b > 0 ∧ c > 0 ∧ a^(3/2) + b^(3/2) + c^(3/2) = 1

-- Problem 1: Prove that abc <= 1/9
theorem problem1 (ha : conditions a b c) : a * b * c ≤ 1 / 9 :=
sorry

-- Problem 2: Prove that a/(b+c) + b/(a+c) + c/(a+b) <= 1 / (2 * sqrt(abc))
theorem problem2 (ha : conditions a b c) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * sqrt (a * b * c)) :=
sorry

end problem1_problem2_l268_268896


namespace smallest_common_multiple_five_digit_l268_268598

def is_multiple (a b : ℕ) : Prop := ∃ k, a = k * b

def smallest_five_digit_multiple_of_3_and_5 (x : ℕ) : Prop :=
  is_multiple x 3 ∧ is_multiple x 5 ∧ 10000 ≤ x ∧ x ≤ 99999 ∧ (∀ y, (10000 ≤ y ∧ y ≤ 99999 ∧ is_multiple y 3 ∧ is_multiple y 5) → x ≤ y)

theorem smallest_common_multiple_five_digit : smallest_five_digit_multiple_of_3_and_5 10005 :=
sorry

end smallest_common_multiple_five_digit_l268_268598


namespace part1_part2_l268_268966

def pos_real := {x : ℝ // 0 < x}

variables (a b c : pos_real)

theorem part1 (h : a.val ^ (3/2) + b.val ^ (3/2) + c.val ^ (3/2) = 1) : a.val * b.val * c.val ≤ 1 / 9 :=
sorry

theorem part2 (h : a.val ^ (3/2) + b.val ^ (3/2) + c.val ^ (3/2) = 1) : 
  a.val / (b.val + c.val) + b.val / (a.val + c.val) + c.val / (a.val + b.val) ≤ 1 / (2 * Real.sqrt (a.val * b.val * c.val)) :=
sorry

end part1_part2_l268_268966


namespace lowest_position_of_vasya_l268_268238

-- Definitions of conditions
def num_cyclists : ℕ := 500
def num_stages : ℕ := 15
def vasya_position_each_stage : ℕ := 7

-- Theorem statement
theorem lowest_position_of_vasya (H1 : ∀ (s: ℕ), s ∈ finset.range(num_stages) → 
(num_cyclists + 1) - vasya_position_each_stage > (num_cyclists - 90))

(assumption_vasya :
  ∀ s ∈ finset.range(num_stages), vasya_position_each_stage < num_cyclists):
  ∃ (lowest_position: ℕ), lowest_position = 91 :=
sorry

end lowest_position_of_vasya_l268_268238


namespace proof_part1_proof_part2_l268_268914

variable {a b c : ℝ}

def pos_and_sum_of_powers (a b c : ℝ) : Prop :=
  0 < a ∧ 0 < b ∧ 0 < c ∧ a^(3/2) + b^(3/2) + c^(3/2) = 1

theorem proof_part1 (h : pos_and_sum_of_powers a b c) : abc ≤ 1 / 9 :=
sorry

theorem proof_part2 (h : pos_and_sum_of_powers a b c) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (abc)) :=
sorry

end proof_part1_proof_part2_l268_268914


namespace problem_statement_l268_268621

theorem problem_statement :
  (-2010)^2011 = - (2010 ^ 2011) :=
by
  -- proof to be filled in
  sorry

end problem_statement_l268_268621


namespace problem1_problem2_l268_268845

variable {a b c : ℝ}

-- Conditions: a, b, c are positive numbers and a^(3/2) + b^(3/2) + c^(3/2) = 1
axiom pos_a : 0 < a
axiom pos_b : 0 < b
axiom pos_c : 0 < c
axiom sum_pow : a^(3/2) + b^(3/2) + c^(3/2) = 1

-- Problem (1): Prove abc ≤ 1/9 given the conditions
theorem problem1 : abc ≤ 1 / 9 := by
  sorry

-- Problem (2): Prove (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ (1 / (2 * (abc) ^ (1/2))) given the conditions
theorem problem2 : (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ 1 / (2 * (abc) ^ (1/2)) := by
  sorry

end problem1_problem2_l268_268845


namespace parabola_focus_on_line_l268_268685

theorem parabola_focus_on_line (p : ℝ) (h₁ : 0 < p) (h₂ : (2 * (p / 2) + 0 - 2 = 0)) : p = 2 :=
sorry

end parabola_focus_on_line_l268_268685


namespace percentage_passed_l268_268093

-- Definitions corresponding to the conditions
def F_H : ℝ := 25
def F_E : ℝ := 35
def F_B : ℝ := 40

-- Main theorem stating the question's proof.
theorem percentage_passed :
  (100 - (F_H + F_E - F_B)) = 80 :=
by
  -- we can transcribe the remaining process here if needed.
  sorry

end percentage_passed_l268_268093


namespace proof_part1_proof_part2_l268_268917

variable {a b c : ℝ}

def pos_and_sum_of_powers (a b c : ℝ) : Prop :=
  0 < a ∧ 0 < b ∧ 0 < c ∧ a^(3/2) + b^(3/2) + c^(3/2) = 1

theorem proof_part1 (h : pos_and_sum_of_powers a b c) : abc ≤ 1 / 9 :=
sorry

theorem proof_part2 (h : pos_and_sum_of_powers a b c) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (abc)) :=
sorry

end proof_part1_proof_part2_l268_268917


namespace largest_four_digit_integer_congruent_to_17_mod_26_l268_268254

theorem largest_four_digit_integer_congruent_to_17_mod_26 :
  ∃ x : ℤ, 1000 ≤ x ∧ x < 10000 ∧ x % 26 = 17 ∧ x = 9978 :=
by
  sorry

end largest_four_digit_integer_congruent_to_17_mod_26_l268_268254


namespace average_and_variance_of_new_data_set_l268_268063

theorem average_and_variance_of_new_data_set
  (avg : ℝ) (var : ℝ) (constant : ℝ)
  (h_avg : avg = 2.8)
  (h_var : var = 3.6)
  (h_const : constant = 60) :
  (avg + constant = 62.8) ∧ (var = 3.6) :=
sorry

end average_and_variance_of_new_data_set_l268_268063


namespace find_ordered_pairs_of_b_c_l268_268636

theorem find_ordered_pairs_of_b_c : 
  ∃! (pairs : ℕ × ℕ), 
    (pairs.1 > 0 ∧ pairs.2 > 0) ∧ 
    (pairs.1 * pairs.1 = 4 * pairs.2) ∧ 
    (pairs.2 * pairs.2 = 4 * pairs.1) :=
sorry

end find_ordered_pairs_of_b_c_l268_268636


namespace bus_initial_passengers_l268_268746

theorem bus_initial_passengers (M W : ℕ) 
  (h1 : W = M / 2) 
  (h2 : M - 16 = W + 8) : 
  M + W = 72 :=
sorry

end bus_initial_passengers_l268_268746


namespace x_plus_p_eq_2p_plus_2_l268_268512

-- Define the conditions and the statement to be proved
theorem x_plus_p_eq_2p_plus_2 (x p : ℝ) (h1 : x > 2) (h2 : |x - 2| = p) : x + p = 2 * p + 2 :=
by
  -- Proof goes here
  sorry

end x_plus_p_eq_2p_plus_2_l268_268512


namespace solve_trig_eq_l268_268356

open Real

theorem solve_trig_eq (x a : ℝ) (hx1 : 0 < x) (hx2 : x < 2 * π) (ha : a > 0) :
    (sin (3 * x) + a * sin (2 * x) + 2 * sin x = 0) →
    (0 < a ∧ a < 2 → x = 0 ∨ x = π) ∧ 
    (a > 5 / 2 → ∃ α, (x = α ∨ x = 2 * π - α)) :=
by sorry

end solve_trig_eq_l268_268356


namespace dimes_count_l268_268738

def num_dimes (total_in_cents : ℤ) (value_quarter value_dime value_nickel : ℤ) (num_each : ℤ) : Prop :=
  total_in_cents = num_each * (value_quarter + value_dime + value_nickel)

theorem dimes_count (num_each : ℤ) :
  num_dimes 440 25 10 5 num_each → num_each = 11 :=
by sorry

end dimes_count_l268_268738


namespace inequality_proof_l268_268708

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h_sum : a + b + c = 1) :
  (1 + a) * (1 + b) * (1 + c) ≥ 8 * (1 - a) * (1 - b) * (1 - c) :=
by
  sorry

end inequality_proof_l268_268708


namespace cos_double_angle_l268_268492

theorem cos_double_angle 
  (α β : ℝ) 
  (h1 : Real.sin (α - β) = 1/3) 
  (h2 : Real.cos α * Real.sin β = 1/6) :
  Real.cos (2 * α + 2 * β) = 1/9 :=
sorry

end cos_double_angle_l268_268492


namespace part1_part2_l268_268883

variables {a b c : ℝ}

theorem part1 (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h : a^(3/2) + b^(3/2) + c^(3/2) = 1) : abc ≤ 1/9 := 
sorry

theorem part2 (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (abc)) :=
sorry

end part1_part2_l268_268883


namespace proof_part1_proof_part2_l268_268911

variable {a b c : ℝ}

def pos_and_sum_of_powers (a b c : ℝ) : Prop :=
  0 < a ∧ 0 < b ∧ 0 < c ∧ a^(3/2) + b^(3/2) + c^(3/2) = 1

theorem proof_part1 (h : pos_and_sum_of_powers a b c) : abc ≤ 1 / 9 :=
sorry

theorem proof_part2 (h : pos_and_sum_of_powers a b c) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (abc)) :=
sorry

end proof_part1_proof_part2_l268_268911


namespace tammy_earnings_after_3_weeks_l268_268553

noncomputable def oranges_picked_per_day (num_trees : ℕ) (oranges_per_tree : ℕ) : ℕ :=
  num_trees * oranges_per_tree

noncomputable def packs_sold_per_day (oranges_per_day : ℕ) (oranges_per_pack : ℕ) : ℕ :=
  oranges_per_day / oranges_per_pack

noncomputable def total_packs_sold_in_weeks (packs_per_day : ℕ) (days_in_week : ℕ) (num_weeks : ℕ) : ℕ :=
  packs_per_day * days_in_week * num_weeks

noncomputable def money_earned (total_packs : ℕ) (price_per_pack : ℕ) : ℕ :=
  total_packs * price_per_pack

theorem tammy_earnings_after_3_weeks :
  let num_trees := 10
  let oranges_per_tree := 12
  let oranges_per_pack := 6
  let price_per_pack := 2
  let days_in_week := 7
  let num_weeks := 3
  oranges_picked_per_day num_trees oranges_per_tree /
  oranges_per_pack *
  days_in_week *
  num_weeks *
  price_per_pack = 840 :=
by {
  sorry
}

end tammy_earnings_after_3_weeks_l268_268553


namespace unicorn_rope_problem_l268_268770

/-
  A unicorn is tethered by a 24-foot golden rope to the base of a sorcerer's cylindrical tower
  whose radius is 10 feet. The rope is attached to the tower at ground level and to the unicorn
  at a height of 6 feet. The unicorn has pulled the rope taut, and the end of the rope is 6 feet
  from the nearest point on the tower.
  The length of the rope that is touching the tower is given as:
  ((96 - sqrt(36)) / 6) feet,
  where 96, 36, and 6 are positive integers, and 6 is prime.
  We need to prove that the sum of these integers is 138.
-/
theorem unicorn_rope_problem : 
  let d := 96
  let e := 36
  let f := 6
  d + e + f = 138 := by
  sorry

end unicorn_rope_problem_l268_268770


namespace number_of_valid_arithmetic_sequences_l268_268458

theorem number_of_valid_arithmetic_sequences : 
  ∃ S : Finset (Finset ℕ), 
  S.card = 16 ∧ 
  ∀ s ∈ S, s.card = 3 ∧ 
  (∃ a b c : ℕ, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ s = {a, b, c} ∧ 1 ≤ a ∧ a ≤ 9 ∧ 1 ≤ b ∧ b ≤ 9 ∧ 1 ≤ c ∧ c ≤ 9 ∧ 
  (b - a = c - b) ∧ (a % 2 = 0 ∨ b % 2 = 0 ∨ c % 2 = 0)) := 
sorry

end number_of_valid_arithmetic_sequences_l268_268458


namespace math_problem_l268_268340

noncomputable def m : ℕ := 294
noncomputable def n : ℕ := 81
noncomputable def d : ℕ := 3

axiom circle_radius (r : ℝ) : r = 42
axiom chords_length (l : ℝ) : l = 78
axiom intersection_distance (d : ℝ) : d = 18

theorem math_problem :
  let m := 294
  let n := 81
  let d := 3
  m + n + d = 378 :=
by {
  -- Proof omitted
  sorry
}

end math_problem_l268_268340


namespace part1_part2_l268_268965

def pos_real := {x : ℝ // 0 < x}

variables (a b c : pos_real)

theorem part1 (h : a.val ^ (3/2) + b.val ^ (3/2) + c.val ^ (3/2) = 1) : a.val * b.val * c.val ≤ 1 / 9 :=
sorry

theorem part2 (h : a.val ^ (3/2) + b.val ^ (3/2) + c.val ^ (3/2) = 1) : 
  a.val / (b.val + c.val) + b.val / (a.val + c.val) + c.val / (a.val + b.val) ≤ 1 / (2 * Real.sqrt (a.val * b.val * c.val)) :=
sorry

end part1_part2_l268_268965


namespace angle_A_measure_in_triangle_l268_268184

theorem angle_A_measure_in_triangle (A B C : ℝ) 
  (h1 : B = 15)
  (h2 : C = 3 * B) 
  (angle_sum : A + B + C = 180) :
  A = 120 :=
by
  -- We'll fill in the proof steps later
  sorry

end angle_A_measure_in_triangle_l268_268184


namespace ceiling_example_l268_268145

/-- Lean 4 statement of the proof problem:
    Prove that ⌈4 (8 - 1/3)⌉ = 31.
-/
theorem ceiling_example : Int.ceil (4 * (8 - (1 / 3 : ℝ))) = 31 := 
by
  sorry

end ceiling_example_l268_268145


namespace find_wall_width_l268_268332

noncomputable def wall_width (painting_width : ℝ) (painting_height : ℝ) (wall_height : ℝ) (painting_coverage : ℝ) : ℝ :=
  (painting_width * painting_height) / (painting_coverage * wall_height)

-- Given constants
def painting_width : ℝ := 2
def painting_height : ℝ := 4
def wall_height : ℝ := 5
def painting_coverage : ℝ := 0.16
def expected_width : ℝ := 10

theorem find_wall_width : wall_width painting_width painting_height wall_height painting_coverage = expected_width := 
by
  sorry

end find_wall_width_l268_268332


namespace probability_blue_or_green_is_two_thirds_l268_268080

-- Definitions for the given conditions
def blue_faces := 3
def red_faces := 2
def green_faces := 1
def total_faces := blue_faces + red_faces + green_faces
def successful_outcomes := blue_faces + green_faces

-- Probability definition
def probability_blue_or_green := (successful_outcomes : ℚ) / total_faces

-- The theorem we want to prove
theorem probability_blue_or_green_is_two_thirds :
  probability_blue_or_green = (2 / 3 : ℚ) :=
by
  -- here would be the proof steps, but we replace them with sorry as per the instructions
  sorry

end probability_blue_or_green_is_two_thirds_l268_268080


namespace problem_1_problem_2_l268_268827

variable (a b c : ℝ)

theorem problem_1 
  (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  a * b * c ≤ 1 / 9 :=
sorry

theorem problem_2 
  (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * (a * b * c)^(1/2)) :=
sorry

end problem_1_problem_2_l268_268827


namespace initial_incorrect_average_l268_268392

theorem initial_incorrect_average (S_correct S_wrong : ℝ) :
  (S_correct = S_wrong - 26 + 36) →
  (S_correct / 10 = 19) →
  (S_wrong / 10 = 18) :=
by
  sorry

end initial_incorrect_average_l268_268392


namespace cos_double_angle_l268_268493

theorem cos_double_angle 
  (α β : ℝ) 
  (h1 : Real.sin (α - β) = 1/3) 
  (h2 : Real.cos α * Real.sin β = 1/6) :
  Real.cos (2 * α + 2 * β) = 1/9 :=
sorry

end cos_double_angle_l268_268493


namespace min_value_expression_l268_268201

theorem min_value_expression :
  ∀ (x y z w : ℝ), x > 0 → y > 0 → z > 0 → w > 0 → x = y → x + y + z + w = 1 →
  (x + y + z) / (x * y * z * w) ≥ 1024 :=
by
  intros x y z w hx hy hz hw hxy hsum
  sorry

end min_value_expression_l268_268201


namespace point_G_six_l268_268146

theorem point_G_six : 
  ∃ (A B C D E F G : ℕ), 
    1 ≤ A ∧ A ≤ 10 ∧
    1 ≤ B ∧ B ≤ 10 ∧
    1 ≤ C ∧ C ≤ 10 ∧
    1 ≤ D ∧ D ≤ 10 ∧
    1 ≤ E ∧ E ≤ 10 ∧
    1 ≤ F ∧ F ≤ 10 ∧
    1 ≤ G ∧ G ≤ 10 ∧
    (A + B = A + C + D) ∧ 
    (A + B = B + E + F) ∧
    (A + B = C + F + G) ∧
    (A + B = D + E + G) ∧ 
    (A + B = 12) →
    G = 6 := 
by
  sorry

end point_G_six_l268_268146


namespace find_p_r_l268_268194

-- Definitions of the polynomials
def f (x : ℝ) (p q : ℝ) : ℝ := x^2 + p * x + q
def g (x : ℝ) (r s : ℝ) : ℝ := x^2 + r * x + s

-- Lean statement of the proof problem:
theorem find_p_r (p q r s : ℝ) (h1 : p ≠ r) (h2 : g (-p / 2) r s = 0) 
  (h3 : f (-r / 2) p q = 0) (h4 : ∀ x : ℝ, f x p q = g x r s) 
  (h5 : f 50 p q = -50) : p + r = -200 := 
sorry

end find_p_r_l268_268194


namespace value_of_k_l268_268140

def f (x : ℝ) : ℝ := 4 * x^2 - 3 * x + 5
def g (x : ℝ) (k : ℝ) : ℝ := 2 * x^2 - k * x + 7

theorem value_of_k (k : ℝ) : f 5 - g 5 k = 40 → k = 1.4 := by
  sorry

end value_of_k_l268_268140


namespace problem_1_problem_2_l268_268879

theorem problem_1 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) 
  (h_sum : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  abc ≤ 1 / 9 := 
sorry

theorem problem_2 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) 
  (h_sum : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * real.sqrt (abc)) := 
sorry

end problem_1_problem_2_l268_268879


namespace power_of_fraction_l268_268133

theorem power_of_fraction :
  ( (2 / 5: ℝ) ^ 7 = 128 / 78125) :=
by
  sorry

end power_of_fraction_l268_268133


namespace coat_price_reduction_l268_268430

theorem coat_price_reduction (original_price reduction_amount : ℝ) 
  (h1 : original_price = 500) (h2 : reduction_amount = 400) :
  (reduction_amount / original_price) * 100 = 80 :=
by {
  sorry -- This is where the proof would go
}

end coat_price_reduction_l268_268430


namespace problem_statement_l268_268061

noncomputable def solveProblem : ℝ :=
  let a := 2
  let b := -3
  let c := 1
  a + b + c

-- The theorem statement to ensure a + b + c equals 0
theorem problem_statement : solveProblem = 0 := by
  sorry

end problem_statement_l268_268061


namespace problem_1_problem_2_l268_268875

theorem problem_1 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) 
  (h_sum : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  abc ≤ 1 / 9 := 
sorry

theorem problem_2 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) 
  (h_sum : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * real.sqrt (abc)) := 
sorry

end problem_1_problem_2_l268_268875


namespace savings_account_amount_l268_268618

theorem savings_account_amount (stimulus : ℕ) (wife_ratio first_son_ratio wife_share first_son_share second_son_share : ℕ) : 
    stimulus = 2000 →
    wife_ratio = 2 / 5 →
    first_son_ratio = 2 / 5 →
    wife_share = wife_ratio * stimulus →
    first_son_share = first_son_ratio * (stimulus - wife_share) →
    second_son_share = 40 / 100 * (stimulus - wife_share - first_son_share) →
    (stimulus - wife_share - first_son_share - second_son_share) = 432 :=
by
  sorry

end savings_account_amount_l268_268618


namespace range_of_a_l268_268025

noncomputable def f (a : ℝ) (e : ℝ) (x : ℝ) := 2*a^x - e*x^2

theorem range_of_a (a e x1 x2 : ℝ) (ha : a > 0) (ha' : a ≠ 1) (hx1 : ∃ x1, is_local_min f a e x1)
  (hx2 : ∃ x2, is_local_max f a e x2) (hx1x2 : x1 < x2) :
  a ∈ set.Ioo (1 / real.exp 1) 1 := sorry

end range_of_a_l268_268025


namespace trig_problem_l268_268642

theorem trig_problem (α : ℝ) (h : Real.tan α = 2) : 
  (Real.sin α + 2 * Real.cos α) / (Real.sin α - Real.cos α) = 4 :=
by
  sorry

end trig_problem_l268_268642


namespace abs_eq_case_solution_l268_268663

theorem abs_eq_case_solution :
  ∃! x : ℝ, |x - 2| = |x - 3| + |x - 4| + |x - 5| := sorry

end abs_eq_case_solution_l268_268663


namespace problem_1_problem_2_l268_268932

variable {a b c : ℝ}

theorem problem_1 (hp : 0 < a ∧ 0 < b ∧ 0 < c)
    (h1 : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
    abc ≤ 1/9 := 
by 
  apply sorry

theorem problem_2 (hp : 0 < a ∧ 0 < b ∧ 0 < c)
    (h1 : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
    a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * real.sqrt (abc)) := 
by 
  apply sorry

end problem_1_problem_2_l268_268932


namespace finite_odd_divisors_condition_l268_268012

theorem finite_odd_divisors_condition (k : ℕ) (hk : 0 < k) :
  (∃ N : ℕ, ∀ n : ℕ, n > N → ¬ (n % 2 = 1 ∧ n ∣ k^n + 1)) ↔ (∃ c : ℕ, k + 1 = 2^c) :=
by sorry

end finite_odd_divisors_condition_l268_268012


namespace percentage_of_red_shirts_l268_268683

theorem percentage_of_red_shirts
  (Total : ℕ) 
  (P_blue P_green : ℝ) 
  (N_other : ℕ)
  (H_Total : Total = 600)
  (H_P_blue : P_blue = 0.45) 
  (H_P_green : P_green = 0.15) 
  (H_N_other : N_other = 102) :
  ( (Total - (P_blue * Total + P_green * Total + N_other)) / Total ) * 100 = 23 := by
  sorry

end percentage_of_red_shirts_l268_268683


namespace vasya_lowest_position_l268_268225

theorem vasya_lowest_position
  (num_cyclists : ℕ)
  (num_stages : ℕ)
  (num_ahead : ℕ)
  (position_vasya : ℕ)
  (total_time : List ℕ)
  (unique_total_times : total_time.nodup)
  (stage_positions : List (List ℕ))
  (unique_stage_positions : ∀ stage ∈ stage_positions, stage.nodup)
  (vasya_consistent : ∀ stage ∈ stage_positions, stage.nth position_vasya = some num_ahead) :
  num_ahead * num_stages + 1 = 91 :=
by
  sorry

end vasya_lowest_position_l268_268225


namespace problem_1_problem_2_l268_268938

variable {a b c : ℝ}

theorem problem_1 (hp : 0 < a ∧ 0 < b ∧ 0 < c)
    (h1 : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
    abc ≤ 1/9 := 
by 
  apply sorry

theorem problem_2 (hp : 0 < a ∧ 0 < b ∧ 0 < c)
    (h1 : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
    a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * real.sqrt (abc)) := 
by 
  apply sorry

end problem_1_problem_2_l268_268938


namespace angle_F_measure_l268_268689

theorem angle_F_measure (α β γ : ℝ) (hD : α = 84) (hAngleSum : α + β + γ = 180) (hBeta : β = 4 * γ + 18) :
  γ = 15.6 := by
  sorry

end angle_F_measure_l268_268689


namespace problem_1_problem_2_l268_268936

variable {a b c : ℝ}

theorem problem_1 (hp : 0 < a ∧ 0 < b ∧ 0 < c)
    (h1 : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
    abc ≤ 1/9 := 
by 
  apply sorry

theorem problem_2 (hp : 0 < a ∧ 0 < b ∧ 0 < c)
    (h1 : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
    a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * real.sqrt (abc)) := 
by 
  apply sorry

end problem_1_problem_2_l268_268936


namespace machine_produces_one_item_in_40_seconds_l268_268751

theorem machine_produces_one_item_in_40_seconds :
  (60 * 1) / 90 * 60 = 40 :=
by
  sorry

end machine_produces_one_item_in_40_seconds_l268_268751


namespace inequality_holds_iff_m_eq_n_l268_268467

theorem inequality_holds_iff_m_eq_n (m n : ℕ) (hm : 0 < m) (hn : 0 < n) :
  (∀ (α β : ℝ), 
    ⌊(m + n) * α⌋ + ⌊(m + n) * β⌋ ≥ 
    ⌊m * α⌋ + ⌊m * β⌋ + ⌊n * (α + β)⌋) ↔ m = n :=
by
  sorry

end inequality_holds_iff_m_eq_n_l268_268467


namespace abc_le_one_ninth_sum_ratios_le_one_over_two_sqrt_abc_l268_268817

open Real

variables (a b c : ℝ)

-- Condition: a, b, c are positive numbers
variables (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
-- Condition: a^(3/2) + b^(3/2) + c^(3/2) = 1
variable  (h_sum : a^(3/2) + b^(3/2) + c^(3/2) = 1)

-- Question 1: Prove abc ≤ 1/9
theorem abc_le_one_ninth : a * b * c ≤ 1 / 9 :=
  sorry

-- Question 2: Prove (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ 1 / (2 * sqrt (a * b * c))
theorem sum_ratios_le_one_over_two_sqrt_abc : 
  (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ 1 / (2 * sqrt (a * b * c)) :=
  sorry

end abc_le_one_ninth_sum_ratios_le_one_over_two_sqrt_abc_l268_268817


namespace two_pow_2014_mod_seven_l268_268337

theorem two_pow_2014_mod_seven : 
  ∃ r : ℕ, 2 ^ 2014 ≡ r [MOD 7] → r = 2 :=
sorry

end two_pow_2014_mod_seven_l268_268337


namespace domain_of_log_function_l268_268394

theorem domain_of_log_function :
  {x : ℝ | x^2 - x > 0} = {x : ℝ | x < 0} ∪ {x : ℝ | x > 1} :=
by
  sorry

end domain_of_log_function_l268_268394


namespace total_apples_packed_l268_268363

def apples_packed_daily (apples_per_box : ℕ) (boxes_per_day : ℕ) : ℕ :=
  apples_per_box * boxes_per_day

def apples_packed_first_week (apples_per_box : ℕ) (boxes_per_day : ℕ) (days_first_week : ℕ) : ℕ :=
  apples_packed_daily apples_per_box boxes_per_day * days_first_week

def apples_packed_second_week (apples_per_box : ℕ) (boxes_per_day : ℕ) (days_second_week : ℕ) (decrease_per_day : ℕ) : ℕ :=
  (apples_packed_daily apples_per_box boxes_per_day - decrease_per_day) * days_second_week

theorem total_apples_packed (apples_per_box : ℕ) (boxes_per_day : ℕ) (days_first_week : ℕ) (days_second_week : ℕ) (decrease_per_day : ℕ) :
  apples_per_box = 40 →
  boxes_per_day = 50 →
  days_first_week = 7 →
  days_second_week = 7 →
  decrease_per_day = 500 →
  apples_packed_first_week apples_per_box boxes_per_day days_first_week + apples_packed_second_week apples_per_box boxes_per_day days_second_week decrease_per_day = 24500 :=
  by
  intros h₁ h₂ h₃ h₄ h₅
  rw [h₁, h₂, h₃, h₄, h₅]
  dsimp [apples_packed_first_week, apples_packed_second_week, apples_packed_daily]
  sorry

end total_apples_packed_l268_268363


namespace part1_part2_l268_268864

variable (a b c : ℝ)

theorem part1 (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) : abc ≤ 1/9 :=
by
  sorry

theorem part2 (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
    (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ 1 / (2 * sqrt(abc)) :=
by
  sorry

end part1_part2_l268_268864


namespace cost_price_per_meter_l268_268600

theorem cost_price_per_meter (selling_price : ℝ) (total_meters : ℕ) (profit_per_meter : ℝ)
  (h1 : selling_price = 8925)
  (h2 : total_meters = 85)
  (h3 : profit_per_meter = 5) :
  (selling_price - total_meters * profit_per_meter) / total_meters = 100 := by
  sorry

end cost_price_per_meter_l268_268600


namespace problem_1_problem_2_l268_268872

theorem problem_1 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) 
  (h_sum : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  abc ≤ 1 / 9 := 
sorry

theorem problem_2 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) 
  (h_sum : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * real.sqrt (abc)) := 
sorry

end problem_1_problem_2_l268_268872


namespace problem_1_problem_2_l268_268877

theorem problem_1 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) 
  (h_sum : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  abc ≤ 1 / 9 := 
sorry

theorem problem_2 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) 
  (h_sum : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * real.sqrt (abc)) := 
sorry

end problem_1_problem_2_l268_268877


namespace even_n_divisible_into_equal_triangles_l268_268639

theorem even_n_divisible_into_equal_triangles (n : ℕ) (hn : 3 < n) :
  (∃ (triangles : ℕ), triangles = n) ↔ (∃ (k : ℕ), n = 2 * k) := 
sorry

end even_n_divisible_into_equal_triangles_l268_268639


namespace triangle_area_integral_bound_l268_268117

def S := 200
def AC := 20
def dist_A_to_tangent := 25
def dist_C_to_tangent := 16
def largest_integer_not_exceeding (S : ℕ) (n : ℕ) : ℕ := n

theorem triangle_area_integral_bound (AC : ℕ) (dist_A_to_tangent : ℕ) (dist_C_to_tangent : ℕ) (S : ℕ) : 
  AC = 20 ∧ dist_A_to_tangent = 25 ∧ dist_C_to_tangent = 16 → largest_integer_not_exceeding S 20 = 10 :=
by
  sorry

end triangle_area_integral_bound_l268_268117


namespace find_x2_plus_y2_l268_268641

theorem find_x2_plus_y2 (x y : ℝ) (h : (x ^ 2 + y ^ 2 + 1) * (x ^ 2 + y ^ 2 - 3) = 5) : x ^ 2 + y ^ 2 = 4 := 
by 
  sorry

end find_x2_plus_y2_l268_268641


namespace least_m_value_l268_268298

def recursive_sequence (x : ℕ → ℚ) : Prop :=
  x 0 = 3 ∧ ∀ n, x (n + 1) = (x n ^ 2 + 9 * x n + 20) / (x n + 8)

theorem least_m_value (x : ℕ → ℚ) (h : recursive_sequence x) : ∃ m, m > 0 ∧ x m ≤ 3 + 1 / 2^10 ∧ ∀ k, k > 0 → k < m → x k > 3 + 1 / 2^10 :=
sorry

end least_m_value_l268_268298


namespace part1_part2_l268_268884

variables {a b c : ℝ}

theorem part1 (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h : a^(3/2) + b^(3/2) + c^(3/2) = 1) : abc ≤ 1/9 := 
sorry

theorem part2 (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (abc)) :=
sorry

end part1_part2_l268_268884


namespace max_cables_cut_l268_268407

def initial_cameras : ℕ := 200
def initial_cables : ℕ := 345
def resulting_clusters : ℕ := 8

theorem max_cables_cut :
  ∃ (cables_cut : ℕ), resulting_clusters = 8 ∧ initial_cables - cables_cut = (initial_cables - cables_cut) - (resulting_clusters - 1) ∧ cables_cut = 153 :=
by
  sorry

end max_cables_cut_l268_268407


namespace problem1_problem2_l268_268905

open Real

variable {a b c : ℝ}

-- Condition: a, b, c are positive and a^{3/2} + b^{3/2} + c^{3/2} = 1
def conditions (a b c : ℝ) := a > 0 ∧ b > 0 ∧ c > 0 ∧ a^(3/2) + b^(3/2) + c^(3/2) = 1

-- Problem 1: Prove that abc <= 1/9
theorem problem1 (ha : conditions a b c) : a * b * c ≤ 1 / 9 :=
sorry

-- Problem 2: Prove that a/(b+c) + b/(a+c) + c/(a+b) <= 1 / (2 * sqrt(abc))
theorem problem2 (ha : conditions a b c) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * sqrt (a * b * c)) :=
sorry

end problem1_problem2_l268_268905


namespace problem1_problem2_l268_268842

variable {a b c : ℝ}

-- Conditions: a, b, c are positive numbers and a^(3/2) + b^(3/2) + c^(3/2) = 1
axiom pos_a : 0 < a
axiom pos_b : 0 < b
axiom pos_c : 0 < c
axiom sum_pow : a^(3/2) + b^(3/2) + c^(3/2) = 1

-- Problem (1): Prove abc ≤ 1/9 given the conditions
theorem problem1 : abc ≤ 1 / 9 := by
  sorry

-- Problem (2): Prove (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ (1 / (2 * (abc) ^ (1/2))) given the conditions
theorem problem2 : (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ 1 / (2 * (abc) ^ (1/2)) := by
  sorry

end problem1_problem2_l268_268842


namespace intersection_M_N_l268_268055

def M : Set ℕ := {1, 2, 4, 8}
def N : Set ℕ := {x | x ∣ 4 ∧ 0 < x}

theorem intersection_M_N :
  M ∩ N = {1, 2, 4} :=
sorry

end intersection_M_N_l268_268055


namespace proof_part1_proof_part2_l268_268848

theorem proof_part1 (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) 
                    (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
                    (a * b * c ≤ 1 / 9) := 
by
  sorry

theorem proof_part2 (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) 
                    (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
                    (a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * (a * b * c)^(1/2))) := 
by
  sorry

end proof_part1_proof_part2_l268_268848


namespace circle_radius_eq_five_l268_268144

theorem circle_radius_eq_five : 
  ∀ (x y : ℝ), (x^2 + y^2 - 6 * x + 8 * y = 0) → (∃ r : ℝ, ((x - 3)^2 + (y + 4)^2 = r^2) ∧ r = 5) :=
by
  sorry

end circle_radius_eq_five_l268_268144


namespace sin_double_angle_l268_268668

theorem sin_double_angle (θ : ℝ)
  (h : ∑' n : ℕ, (Real.sin θ)^(2 * n) = 3) :
  Real.sin (2 * θ) = (2 * Real.sqrt 2) / 3 :=
sorry

end sin_double_angle_l268_268668


namespace wooden_toy_price_l268_268071

noncomputable def price_of_hat : ℕ := 10
noncomputable def total_money : ℕ := 100
noncomputable def hats_bought : ℕ := 3
noncomputable def change_received : ℕ := 30
noncomputable def total_spent := total_money - change_received
noncomputable def cost_of_hats := hats_bought * price_of_hat

theorem wooden_toy_price :
  ∃ (W : ℕ), total_spent = 2 * W + cost_of_hats ∧ W = 20 := 
by 
  sorry

end wooden_toy_price_l268_268071


namespace trigonometric_identity_l268_268500

theorem trigonometric_identity 
  (x : ℝ)
  (h : Real.cos (π / 6 - x) = - Real.sqrt 3 / 3) :
  Real.cos (5 * π / 6 + x) + Real.sin (2 * π / 3 - x) = 0 :=
by
  sorry

end trigonometric_identity_l268_268500


namespace triangle_side_length_l268_268412

-- Defining basic properties and known lengths of the similar triangles
def GH : ℝ := 8
def HI : ℝ := 16
def YZ : ℝ := 24
def XY : ℝ := 12

-- Defining the similarity condition for triangles GHI and XYZ
def triangles_similar : Prop := 
  -- The similarity of the triangles implies proportionality of the sides
  (XY / GH = YZ / HI)

-- The theorem statement to prove
theorem triangle_side_length (h_sim : triangles_similar) : XY = 12 :=
by
  -- assuming the similarity condition and known lengths
  sorry -- This will be the detailed proof

end triangle_side_length_l268_268412


namespace part1_part2_l268_268971

variable (a b c : ℝ)

-- Condition: a, b, c are positive numbers and a^(3/2) + b^(3/2) + c^(3/2) = 1
axiom pos_a : a > 0
axiom pos_b : b > 0
axiom pos_c : c > 0
axiom sum_condition : a^(3/2) + b^(3/2) + c^(3/2) = 1

-- Proof Problem (1): abc ≤ 1/9
theorem part1 : abc ≤ 1/9 := 
by {
  have h := sorry,
  exact h
}

-- Proof Problem (2): a/(b+c) + b/(a+c) + c/(a+b) ≤ 1/(2sqrt(abc))
theorem part2 : a/(b+c) + b/(a+c) + c/(a+b) ≤ 1/(2*sqrt(abc)) := 
by {
  have h := sorry,
  exact h
}

end part1_part2_l268_268971


namespace number_of_pencils_is_11_l268_268129

noncomputable def numberOfPencils (A B : ℕ) :  ℕ :=
  2 * A + 1 * B

theorem number_of_pencils_is_11 (A B : ℕ) (h1 : A + 2 * B = 16) (h2 : A + B = 9) : numberOfPencils A B = 11 :=
  sorry

end number_of_pencils_is_11_l268_268129


namespace find_digits_sum_l268_268379

theorem find_digits_sum
  (a b c d : ℕ)
  (distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (non_zero : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0)
  (sum_of_numbers : 6 * (a + b + c + d) * 1111 = 73326) :
  {a, b, c, d} = {1, 2, 3, 5} :=
by
  sorry

end find_digits_sum_l268_268379


namespace relationship_between_x_and_y_l268_268333

theorem relationship_between_x_and_y (x y : ℝ) (h1 : 2 * x - y > 3 * x) (h2 : x + 2 * y < 2 * y) :
  x < 0 ∧ y > 0 :=
sorry

end relationship_between_x_and_y_l268_268333


namespace probability_is_half_l268_268575

-- Define the set of numbers from 1 to 30
def numbers : Finset ℕ := (Finset.range 30).map ⟨Nat.succ, Nat.succ_injective⟩

-- Define the set of multiples of 3 from 1 to 30
def multiples_of_3 : Finset ℕ := numbers.filter (λ n, n % 3 = 0)

-- Define the set of multiples of 4 from 1 to 30
def multiples_of_4 : Finset ℕ := numbers.filter (λ n, n % 4 = 0)

-- Define the set of multiples of 12 from 1 to 30 (multiples of both 3 and 4)
def multiples_of_12 : Finset ℕ := numbers.filter (λ n, n % 12 = 0)

-- Calculate the probability using the principle of inclusion-exclusion
def favorable_outcomes : ℕ := multiples_of_3.card + multiples_of_4.card - multiples_of_12.card

-- Total number of outcomes
def total_outcomes : ℕ := numbers.card

-- Calculate the probability
def probability : ℚ := favorable_outcomes / total_outcomes

-- Prove that the probability is 1/2
theorem probability_is_half : probability = 1 / 2 := by
  sorry

end probability_is_half_l268_268575


namespace find_arrays_l268_268632

-- Defines a condition where positive integers satisfy the given properties
def satisfies_conditions (a b c d : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ 
  a ∣ b * c * d - 1 ∧ 
  b ∣ a * c * d - 1 ∧ 
  c ∣ a * b * d - 1 ∧ 
  d ∣ a * b * c - 1

-- The theorem that any four positive integers satisfying the conditions are either (2, 3, 7, 11) or (2, 3, 11, 13)
theorem find_arrays :
  ∀ a b c d : ℕ, satisfies_conditions a b c d → 
    (a = 2 ∧ b = 3 ∧ c = 7 ∧ d = 11) ∨ 
    (a = 2 ∧ b = 3 ∧ c = 11 ∧ d = 13) :=
by
  intro a b c d h
  sorry

end find_arrays_l268_268632


namespace tensor_value_l268_268006

variables (h : ℝ)

def tensor (x y : ℝ) : ℝ := x^2 - y^2

theorem tensor_value : tensor h (tensor h h) = h^2 :=
by 
-- Complete proof body not required, 'sorry' is used for omitted proof
sorry

end tensor_value_l268_268006


namespace monthly_income_P_l268_268606

theorem monthly_income_P (P Q R : ℝ)
  (h1 : (P + Q) / 2 = 5050)
  (h2 : (Q + R) / 2 = 6250)
  (h3 : (P + R) / 2 = 5200) :
  P = 4000 := 
sorry

end monthly_income_P_l268_268606


namespace vasya_lowest_position_l268_268228

theorem vasya_lowest_position
  (n : ℕ) (m : ℕ) (num_cyclists : ℕ) (vasya_place : ℕ) :
  num_cyclists = 500 →
  n = 15 →
  vasya_place = 7 →
  ∀ (stages : fin n → fin num_cyclists) (no_identical_times : ∀ i j : fin n, i ≠ j → 
  ∀ k l : fin num_cyclists, stages i k ≠ stages j l),
  ∃ (lowest_position : ℕ), lowest_position = 91 := 
by sorry

end vasya_lowest_position_l268_268228


namespace problem1_problem2_l268_268906

open Real

variable {a b c : ℝ}

-- Condition: a, b, c are positive and a^{3/2} + b^{3/2} + c^{3/2} = 1
def conditions (a b c : ℝ) := a > 0 ∧ b > 0 ∧ c > 0 ∧ a^(3/2) + b^(3/2) + c^(3/2) = 1

-- Problem 1: Prove that abc <= 1/9
theorem problem1 (ha : conditions a b c) : a * b * c ≤ 1 / 9 :=
sorry

-- Problem 2: Prove that a/(b+c) + b/(a+c) + c/(a+b) <= 1 / (2 * sqrt(abc))
theorem problem2 (ha : conditions a b c) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * sqrt (a * b * c)) :=
sorry

end problem1_problem2_l268_268906


namespace trip_total_charge_is_correct_l268_268691

-- Define the initial fee
def initial_fee : ℝ := 2.35

-- Define the charge per increment
def charge_per_increment : ℝ := 0.35

-- Define the increment size in miles
def increment_size : ℝ := 2 / 5

-- Define the total distance of the trip
def trip_distance : ℝ := 3.6

-- Define the total charge function
def total_charge (initial : ℝ) (increment_charge : ℝ) (increment : ℝ) (distance : ℝ) : ℝ :=
  initial + (distance / increment) * increment_charge

-- Prove the total charge for a trip of 3.6 miles is $5.50
theorem trip_total_charge_is_correct :
  total_charge initial_fee charge_per_increment increment_size trip_distance = 5.50 :=
by
  sorry

end trip_total_charge_is_correct_l268_268691


namespace bus_stop_time_l268_268301

theorem bus_stop_time
  (speed_without_stoppage : ℝ := 54)
  (speed_with_stoppage : ℝ := 45)
  (distance_diff : ℝ := speed_without_stoppage - speed_with_stoppage)
  (distance : ℝ := distance_diff)
  (speed_km_per_min : ℝ := speed_without_stoppage / 60) :
  distance / speed_km_per_min = 10 :=
by
  -- The proof steps would go here.
  sorry

end bus_stop_time_l268_268301


namespace least_positive_multiple_24_gt_450_l268_268256

theorem least_positive_multiple_24_gt_450 : ∃ n : ℕ, n > 450 ∧ n % 24 = 0 ∧ n = 456 :=
by
  use 456
  sorry

end least_positive_multiple_24_gt_450_l268_268256


namespace salary_reduction_l268_268239

noncomputable def percentageIncrease : ℝ := 16.27906976744186 / 100

theorem salary_reduction (S R : ℝ) (P : ℝ) (h1 : R = S * (1 - P / 100)) (h2 : S = R * (1 + percentageIncrease)) : P = 14 :=
by
  sorry

end salary_reduction_l268_268239


namespace range_of_m_increasing_function_l268_268321

theorem range_of_m_increasing_function :
  (2 : ℝ) ≤ m ∧ m ≤ 4 ↔ ∀ x : ℝ, (1 / 3 : ℝ) * x ^ 3 - (4 * m - 1) * x ^ 2 + (15 * m ^ 2 - 2 * m - 7) * x + 2 ≤ 
                                 ((1 / 3 : ℝ) * (x + 1) ^ 3 - (4 * m - 1) * (x + 1) ^ 2 + (15 * m ^ 2 - 2 * m - 7) * (x + 1) + 2) :=
by
  sorry

end range_of_m_increasing_function_l268_268321


namespace base_5_to_decimal_l268_268784

theorem base_5_to_decimal : 
  let b5 := [1, 2, 3, 4] -- base-5 number 1234 in list form
  let decimal := 194
  (b5[0] * 5^3 + b5[1] * 5^2 + b5[2] * 5^1 + b5[3] * 5^0) = decimal :=
by
  -- Proof details go here
  sorry

end base_5_to_decimal_l268_268784


namespace area_of_triangle_ABC_l268_268524

theorem area_of_triangle_ABC
  (A B C : ℝ)
  (a b c : ℝ)
  (sin_C_eq : Real.sin C = Real.sqrt 3 / 3)
  (sin_CBA_eq : Real.sin C + Real.sin (B - A) = Real.sin (2 * A))
  (a_minus_b_eq : a - b = 3 - Real.sqrt 6)
  (c_eq : c = Real.sqrt 3) :
  1 / 2 * a * b * Real.sin C = 3 * Real.sqrt 2 / 2 := sorry

end area_of_triangle_ABC_l268_268524


namespace goose_eggs_hatching_l268_268206

theorem goose_eggs_hatching (x : ℝ) :
  (∃ n_hatched : ℝ, 3 * (2 * n_hatched / 20) = 110 ∧ x = n_hatched / 550) →
  x = 2 / 3 :=
by
  intro h
  sorry

end goose_eggs_hatching_l268_268206


namespace petya_digits_l268_268377

theorem petya_digits (a b c d : ℕ) (distinct_digits : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (non_zero : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0)
  (sum_formed_numbers : 6 * (a + b + c + d) * 1111 = 73326) :
  {a, b, c, d} = {1, 2, 3, 5} :=
by 
  sorry

end petya_digits_l268_268377


namespace cos_double_angle_proof_l268_268479

variable {α β : ℝ}

theorem cos_double_angle_proof (h1 : sin (α - β) = 1 / 3) (h2 : cos α * sin β = 1 / 6) :
  cos (2 * α + 2 * β) = 1 / 9 :=
sorry

end cos_double_angle_proof_l268_268479


namespace rabbit_speed_correct_l268_268176

-- Define the conditions given in the problem
def rabbit_speed (x : ℝ) : Prop :=
2 * (2 * x + 4) = 188

-- State the main theorem using the defined conditions
theorem rabbit_speed_correct : ∃ x : ℝ, rabbit_speed x ∧ x = 45 :=
by
  sorry

end rabbit_speed_correct_l268_268176


namespace average_of_all_digits_l268_268726

theorem average_of_all_digits {a b : ℕ} (n : ℕ) (x y : ℕ) (h1 : a = 6) (h2 : b = 4) (h3 : n = 10) (h4 : x = 58) (h5 : y = 113) :
  ((a * x + b * y) / n = 80) :=
  sorry

end average_of_all_digits_l268_268726


namespace part1_part2_l268_268860

variable (a b c : ℝ)

theorem part1 (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) : abc ≤ 1/9 :=
by
  sorry

theorem part2 (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
    (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ 1 / (2 * sqrt(abc)) :=
by
  sorry

end part1_part2_l268_268860


namespace part1_part2_l268_268981

noncomputable theory

variables {a b c : ℝ}

-- Condition that a, b, c are positive numbers and the given equation holds
def pos_numbers : Prop := a > 0 ∧ b > 0 ∧ c > 0
def sum_condition : Prop := (a ^ (3/2)) + (b ^ (3/2)) + (c ^ (3/2)) = 1

-- Proof goal for the first part: abc <= 1/9
theorem part1 (h₁ : pos_numbers) (h₂ : sum_condition) : 
  a * b * c ≤ 1 / 9 :=
sorry

-- Proof goal for the second part: a/(b+c) + b/(a+c) + c/(a+b) <= 1/(2 * sqrt(abc))
theorem part2 (h₁ : pos_numbers) (h₂ : sum_condition) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * sqrt (a * b * c)) :=
sorry

end part1_part2_l268_268981


namespace chuck_team_leads_by_2_l268_268011

open Nat

noncomputable def chuck_team_score_first_quarter := 9 * 2 + 5 * 1
noncomputable def yellow_team_score_first_quarter := 7 * 2 + 4 * 3

noncomputable def chuck_team_score_second_quarter := 6 * 2 + 3 * 3
noncomputable def yellow_team_score_second_quarter := 5 * 2 + 2 * 3 + 3 * 1

noncomputable def chuck_team_score_third_quarter := 4 * 2 + 2 * 3 + 6 * 1
noncomputable def yellow_team_score_third_quarter := 6 * 2 + 2 * 3

noncomputable def chuck_team_score_fourth_quarter := 8 * 2 + 1 * 3
noncomputable def yellow_team_score_fourth_quarter := 4 * 2 + 3 * 3 + 2 * 1

noncomputable def chuck_team_technical_fouls := 3
noncomputable def yellow_team_technical_fouls := 2

noncomputable def total_chuck_team_score :=
  chuck_team_score_first_quarter + chuck_team_score_second_quarter + 
  chuck_team_score_third_quarter + chuck_team_score_fourth_quarter + 
  chuck_team_technical_fouls

noncomputable def total_yellow_team_score :=
  yellow_team_score_first_quarter + yellow_team_score_second_quarter + 
  yellow_team_score_third_quarter + yellow_team_score_fourth_quarter + 
  yellow_team_technical_fouls

noncomputable def chuck_team_lead :=
  total_chuck_team_score - total_yellow_team_score

theorem chuck_team_leads_by_2 :
  chuck_team_lead = 2 :=
by
  sorry

end chuck_team_leads_by_2_l268_268011


namespace technician_completion_percentage_l268_268278

noncomputable def percentage_completed (D : ℝ) : ℝ :=
  let total_distance := 2.20 * D
  let completed_distance := 1.12 * D
  (completed_distance / total_distance) * 100

theorem technician_completion_percentage (D : ℝ) (hD : D > 0) :
  percentage_completed D = 50.91 :=
by
  sorry

end technician_completion_percentage_l268_268278


namespace find_constant_a_l268_268505

noncomputable def f (a t : ℝ) : ℝ := (t - 2)^2 - 4 - a

theorem find_constant_a :
  (∃ (a : ℝ),
    (∀ (t : ℝ), -1 ≤ t ∧ t ≤ 1 → |f a t| ≤ 4) ∧ 
    (∃ (t : ℝ), -1 ≤ t ∧ t ≤ 1 ∧ |f a t| = 4)) →
  a = 1 :=
sorry

end find_constant_a_l268_268505


namespace remainder_when_divided_by_15_l268_268104

def N (k : ℤ) : ℤ := 35 * k + 25

theorem remainder_when_divided_by_15 (k : ℤ) : (N k) % 15 = 10 := 
by 
  -- proof would go here
  sorry

end remainder_when_divided_by_15_l268_268104


namespace inequality_solution_l268_268516

theorem inequality_solution (m : ℝ) : 
  (∀ x : ℝ, 2 * x + 7 > 3 * x + 2 ∧ 2 * x - 2 < 2 * m → x < 5) → m ≥ 4 :=
by
  sorry

end inequality_solution_l268_268516


namespace total_money_from_selling_watermelons_l268_268121

-- Given conditions
def weight_of_one_watermelon : ℝ := 23
def price_per_pound : ℝ := 2
def number_of_watermelons : ℝ := 18

-- Statement to be proved
theorem total_money_from_selling_watermelons : 
  (weight_of_one_watermelon * price_per_pound) * number_of_watermelons = 828 := 
by 
  sorry

end total_money_from_selling_watermelons_l268_268121


namespace distribute_balls_into_boxes_l268_268664

/--
Given 6 distinguishable balls and 3 distinguishable boxes, 
there are 3^6 = 729 ways to distribute the balls into the boxes.
-/
theorem distribute_balls_into_boxes : (3 : ℕ)^6 = 729 := 
by
  sorry

end distribute_balls_into_boxes_l268_268664


namespace find_range_of_m_l268_268331

-- Define the propositions p and q
def p (m : ℝ) : Prop := ∃ x y : ℝ, x ≠ y ∧ x^2 + m * x + 1 = 0 ∧ y^2 + m * y + 1 = 0
def q (m : ℝ) : Prop := 1 < m ∧ m < 3

theorem find_range_of_m (m : ℝ) (h1 : ¬(p m ∧ q m)) (h2 : ¬¬p m) : m ≥ 3 ∨ m < -2 :=
by 
  sorry

end find_range_of_m_l268_268331


namespace sum_of_numbers_ge_1_1_l268_268637

theorem sum_of_numbers_ge_1_1 :
  let numbers := [1.4, 0.9, 1.2, 0.5, 1.3]
  let threshold := 1.1
  let filtered_numbers := numbers.filter (fun x => x >= threshold)
  let sum_filtered := filtered_numbers.sum
  sum_filtered = 3.9 :=
by {
  sorry
}

end sum_of_numbers_ge_1_1_l268_268637


namespace proof_part1_proof_part2_l268_268858

theorem proof_part1 (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) 
                    (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
                    (a * b * c ≤ 1 / 9) := 
by
  sorry

theorem proof_part2 (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) 
                    (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
                    (a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * (a * b * c)^(1/2))) := 
by
  sorry

end proof_part1_proof_part2_l268_268858


namespace given_trig_identity_l268_268033

variable {x : ℂ} {α : ℝ} {n : ℕ}

theorem given_trig_identity (h : x + 1/x = 2 * Real.cos α) : x^n + 1/x^n = 2 * Real.cos (n * α) :=
sorry

end given_trig_identity_l268_268033


namespace value_of_a_l268_268036

theorem value_of_a (x : ℝ) (h : (1 - x^32) ≠ 0):
  (8 * a / (1 - x^32) = 
   2 / (1 - x) + 2 / (1 + x) + 
   4 / (1 + x^2) + 8 / (1 + x^4) + 
   16 / (1 + x^8) + 32 / (1 + x^16)) → 
  a = 8 := sorry

end value_of_a_l268_268036


namespace bryden_amount_correct_l268_268750

-- Each state quarter has a face value of $0.25.
def face_value (q : ℕ) : ℝ := 0.25 * q

-- The collector offers to buy the state quarters for 1500% of their face value.
def collector_multiplier : ℝ := 15

-- Bryden has 10 state quarters.
def bryden_quarters : ℕ := 10

-- Calculate the amount Bryden will get for his 10 state quarters.
def amount_received : ℝ := collector_multiplier * face_value bryden_quarters

-- Prove that the amount received by Bryden equals $37.5.
theorem bryden_amount_correct : amount_received = 37.5 :=
by
  sorry

end bryden_amount_correct_l268_268750


namespace problem1_problem2_l268_268902

open Real

variable {a b c : ℝ}

-- Condition: a, b, c are positive and a^{3/2} + b^{3/2} + c^{3/2} = 1
def conditions (a b c : ℝ) := a > 0 ∧ b > 0 ∧ c > 0 ∧ a^(3/2) + b^(3/2) + c^(3/2) = 1

-- Problem 1: Prove that abc <= 1/9
theorem problem1 (ha : conditions a b c) : a * b * c ≤ 1 / 9 :=
sorry

-- Problem 2: Prove that a/(b+c) + b/(a+c) + c/(a+b) <= 1 / (2 * sqrt(abc))
theorem problem2 (ha : conditions a b c) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * sqrt (a * b * c)) :=
sorry

end problem1_problem2_l268_268902


namespace min_ab_diff_value_l268_268534

noncomputable def min_ab_diff (x y z : ℝ) : ℝ :=
  let A := Real.sqrt (x + 3) + Real.sqrt (y + 6) + Real.sqrt (z + 12)
  let B := Real.sqrt (x + 2) + Real.sqrt (y + 2) + Real.sqrt (z + 2)
  A^2 - B^2

theorem min_ab_diff_value : ∀ (x y z : ℝ),
  0 ≤ x → 0 ≤ y → 0 ≤ z → min_ab_diff x y z = 36 :=
by
  intros x y z hx hy hz
  sorry

end min_ab_diff_value_l268_268534


namespace part1_part2_l268_268960

def pos_real := {x : ℝ // 0 < x}

variables (a b c : pos_real)

theorem part1 (h : a.val ^ (3/2) + b.val ^ (3/2) + c.val ^ (3/2) = 1) : a.val * b.val * c.val ≤ 1 / 9 :=
sorry

theorem part2 (h : a.val ^ (3/2) + b.val ^ (3/2) + c.val ^ (3/2) = 1) : 
  a.val / (b.val + c.val) + b.val / (a.val + c.val) + c.val / (a.val + b.val) ≤ 1 / (2 * Real.sqrt (a.val * b.val * c.val)) :=
sorry

end part1_part2_l268_268960


namespace talent_show_girls_count_l268_268074

theorem talent_show_girls_count (B G : ℕ) (h1 : B + G = 34) (h2 : G = B + 22) : G = 28 :=
by
  sorry

end talent_show_girls_count_l268_268074


namespace benjamin_weekly_walks_l268_268132

def walking_miles_in_week
  (work_days_per_week : ℕ)
  (work_distance_per_day : ℕ)
  (dog_walks_per_day : ℕ)
  (dog_walk_distance : ℕ)
  (best_friend_visits_per_week : ℕ)
  (best_friend_distance : ℕ)
  (store_visits_per_week : ℕ)
  (store_distance : ℕ)
  (hike_distance_per_week : ℕ) : ℕ :=
  (work_days_per_week * work_distance_per_day) +
  (dog_walks_per_day * dog_walk_distance * 7) +
  (best_friend_visits_per_week * (best_friend_distance * 2)) +
  (store_visits_per_week * (store_distance * 2)) +
  hike_distance_per_week

theorem benjamin_weekly_walks :
  walking_miles_in_week 5 (8 * 2) 2 3 1 5 2 4 10 = 158 := 
  by
    sorry

end benjamin_weekly_walks_l268_268132


namespace isosceles_triangle_perimeter_l268_268523

theorem isosceles_triangle_perimeter 
  (a b : ℕ) 
  (h_iso : a = b ∨ a = 3 ∨ b = 3) 
  (h_sides : a = 6 ∨ b = 6) 
  : a + b + 3 = 15 := by
  sorry

end isosceles_triangle_perimeter_l268_268523


namespace sum_end_digit_7_l268_268781

theorem sum_end_digit_7 (n : ℕ) : ¬ (n * (n + 1) ≡ 14 [MOD 20]) :=
by
  intro h
  -- Place where you'd continue the proof, but for now we use sorry
  sorry

end sum_end_digit_7_l268_268781


namespace height_of_cylinder_is_2sqrt10_l268_268766

noncomputable def cylinder_height (r_cylinder r_hemisphere : ℝ) (cylinder_inscribed : Prop) : ℝ :=
if cylinder_inscribed ∧ r_cylinder = 3 ∧ r_hemisphere = 7 then 2 * Real.sqrt 10 else 0

theorem height_of_cylinder_is_2sqrt10 : cylinder_height 3 7 (cylinder_inscribed := true) = 2 * Real.sqrt 10 :=
by
  sorry

end height_of_cylinder_is_2sqrt10_l268_268766


namespace part1_part2_l268_268991

-- a), b), c are positive real numbers and ${a}^{\frac{3}{2}}+{b}^{\frac{3}{2}}+{c}^{\frac{3}{2}}=1$
variables (a b c : ℝ)
axiom a_pos : 0 < a
axiom b_pos : 0 < b
axiom c_pos : 0 < c
axiom sum_eq_one : a^(3/2) + b^(3/2) + c^(3/2) = 1

-- Prove (1) and (2)
theorem part1 : abc ≤ 1 / 9 := sorry

theorem part2 : a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (abc)) := sorry

end part1_part2_l268_268991


namespace express_w_l268_268498

theorem express_w (w a b c : ℝ) (x y z : ℝ)
  (h_distinct : a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ a ≠ w ∧ b ≠ w ∧ c ≠ w)
  (h1 : x + y + z = 1)
  (h2 : x * a^2 + y * b^2 + z * c^2 = w^2)
  (h3 : x * a^3 + y * b^3 + z * c^3 = w^3)
  (h4 : x * a^4 + y * b^4 + z * c^4 = w^4) :
  w = - (a * b * c) / (a * b + b * c + c * a) :=
sorry

end express_w_l268_268498


namespace eithan_savings_l268_268617

theorem eithan_savings :
  let amount := 2000 : ℝ 
  let wife_share := (2/5) * amount 
  let remaining_after_wife := amount - wife_share  
  let first_son_share := (2/5) * remaining_after_wife 
  let remaining_after_first_son := remaining_after_wife - first_son_share 
  let second_son_share := (40/100) * remaining_after_first_son 
  let savings := remaining_after_first_son - second_son_share 
  savings = 432 :=
by
  sorry

end eithan_savings_l268_268617


namespace prove_root_property_l268_268353

-- Define the quadratic equation and its roots
theorem prove_root_property :
  let r := -4 + Real.sqrt 226
  let s := -4 - Real.sqrt 226
  (r + 4) * (s + 4) = -226 :=
by
  -- the proof steps go here (omitted)
  sorry

end prove_root_property_l268_268353


namespace probability_multiple_of_3_or_4_l268_268567

theorem probability_multiple_of_3_or_4 :
  let numbers := Finset.range 30
  let multiples_of_3 := {n ∈ numbers | n % 3 = 0}
  let multiples_of_4 := {n ∈ numbers | n % 4 = 0}
  let multiples_of_12 := {n ∈ numbers | n % 12 = 0}
  let favorable_count := multiples_of_3.card + multiples_of_4.card - multiples_of_12.card
  let probability := (favorable_count : ℚ) / numbers.card
  probability = (1 / 2 : ℚ) :=
by
  sorry

end probability_multiple_of_3_or_4_l268_268567


namespace sum_of_values_l268_268531

def f (x : ℝ) : ℝ := x^2 + 2 * x + 2

theorem sum_of_values (z₁ z₂ : ℝ) (h₁ : f (3 * z₁) = 10) (h₂ : f (3 * z₂) = 10) :
  z₁ + z₂ = - (2 / 9) :=
by
  sorry

end sum_of_values_l268_268531


namespace paint_cost_of_cube_l268_268429

theorem paint_cost_of_cube (cost_per_kg : ℕ) (coverage_per_kg : ℕ) (side_length : ℕ) (total_cost : ℕ) 
  (h1 : cost_per_kg = 20)
  (h2 : coverage_per_kg = 15)
  (h3 : side_length = 5)
  (h4 : total_cost = 200) : 
  (6 * side_length^2 / coverage_per_kg) * cost_per_kg = total_cost :=
by
  sorry

end paint_cost_of_cube_l268_268429


namespace probability_seating_7_probability_seating_n_l268_268702

-- Definitions and theorem for case (a): n = 7
def num_ways_to_seat_7 (n : ℕ) (k : ℕ) : ℕ := n * (n - 1) * (n - 2) / k!

def valid_arrangements_7 : ℕ := 1

theorem probability_seating_7 : 
  let total_ways := num_ways_to_seat_7 7 3 in
  let valid_ways := valid_arrangements_7 in
  total_ways = 35 ∧ (valid_ways : ℚ) / total_ways = 0.2 := 
by
  sorry

-- Definitions and theorem for case (b): general n
def choose (n k : ℕ) : ℕ := n * (n - 1) / 2

def valid_arrangements_n (n : ℕ) : ℚ := (n - 4) * (n - 5) / 2

theorem probability_seating_n (n : ℕ) (hn : n ≥ 6) : 
  let total_ways := choose (n - 1) 2 in
  let valid_ways := valid_arrangements_n n in
  total_ways = (n - 1) * (n - 2) / 2 ∧ 
  (valid_ways : ℚ) / total_ways = (n - 4) * (n - 5) / ((n - 1) * (n - 2)) :=
by
  sorry

end probability_seating_7_probability_seating_n_l268_268702


namespace value_of_f_prime_at_2_l268_268322

theorem value_of_f_prime_at_2 :
  ∃ (f' : ℝ → ℝ), 
  (∀ (x : ℝ), f' x = 2 * x + 3 * f' 2 + 1 / x) →
  f' 2 = - (9 / 4) := 
by 
  sorry

end value_of_f_prime_at_2_l268_268322


namespace part1_part2_l268_268890

variables {a b c : ℝ}

theorem part1 (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h : a^(3/2) + b^(3/2) + c^(3/2) = 1) : abc ≤ 1/9 := 
sorry

theorem part2 (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (abc)) :=
sorry

end part1_part2_l268_268890


namespace exists_num_with_digit_sum_div_by_11_l268_268714

-- Helper function to sum the digits of a natural number
def digit_sum (n : ℕ) : ℕ :=
  (n.digits 10).sum

-- Main theorem statement
theorem exists_num_with_digit_sum_div_by_11 (N : ℕ) :
  ∃ k : ℕ, k < 39 ∧ (digit_sum (N + k)) % 11 = 0 :=
sorry

end exists_num_with_digit_sum_div_by_11_l268_268714


namespace part1_part2_l268_268986

noncomputable theory

variables {a b c : ℝ}

-- Condition that a, b, c are positive numbers and the given equation holds
def pos_numbers : Prop := a > 0 ∧ b > 0 ∧ c > 0
def sum_condition : Prop := (a ^ (3/2)) + (b ^ (3/2)) + (c ^ (3/2)) = 1

-- Proof goal for the first part: abc <= 1/9
theorem part1 (h₁ : pos_numbers) (h₂ : sum_condition) : 
  a * b * c ≤ 1 / 9 :=
sorry

-- Proof goal for the second part: a/(b+c) + b/(a+c) + c/(a+b) <= 1/(2 * sqrt(abc))
theorem part2 (h₁ : pos_numbers) (h₂ : sum_condition) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * sqrt (a * b * c)) :=
sorry

end part1_part2_l268_268986


namespace part1_part2_l268_268985

noncomputable theory

variables {a b c : ℝ}

-- Condition that a, b, c are positive numbers and the given equation holds
def pos_numbers : Prop := a > 0 ∧ b > 0 ∧ c > 0
def sum_condition : Prop := (a ^ (3/2)) + (b ^ (3/2)) + (c ^ (3/2)) = 1

-- Proof goal for the first part: abc <= 1/9
theorem part1 (h₁ : pos_numbers) (h₂ : sum_condition) : 
  a * b * c ≤ 1 / 9 :=
sorry

-- Proof goal for the second part: a/(b+c) + b/(a+c) + c/(a+b) <= 1/(2 * sqrt(abc))
theorem part2 (h₁ : pos_numbers) (h₂ : sum_condition) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * sqrt (a * b * c)) :=
sorry

end part1_part2_l268_268985


namespace problem_solution_l268_268359

-- Define the set M based on the condition |2x - 1| < 1
def M : Set ℝ := {x | 0 < x ∧ x < 1}

-- Main theorem composed of two parts
theorem problem_solution :
  (∀ x, |2 * x - 1| < 1 ↔ 0 < x ∧ x < 1) ∧
  (∀ a b ∈ M, (a * b + 1) > (a + b)) :=
by
  split
  -- Part 1: Prove the equivalence of |2x - 1| < 1 and the set definition of M
  · intro x
    split
    -- Prove |2x - 1| < 1 → 0 < x ∧ x < 1
    sorry
    -- Prove 0 < x ∧ x < 1 → |2x - 1| < 1
    sorry
  -- Part 2: Prove ab + 1 > a + b for all a, b in M
  · intros a b ha hb
    have ha' : 0 < a ∧ a < 1 := ha
    have hb' : 0 < b ∧ b < 1 := hb
    -- Prove the inequality
    sorry

end problem_solution_l268_268359


namespace divisor_is_50_l268_268754

theorem divisor_is_50 (D : ℕ) (h1 : ∃ n, n = 44 * 432 ∧ n % 44 = 0)
                      (h2 : ∃ n, n = 44 * 432 ∧ n % D = 8) : D = 50 :=
by
  sorry

end divisor_is_50_l268_268754


namespace cos_double_angle_l268_268473

variable {α β : Real}

-- Definitions from the conditions
def sin_diff_condition : Prop := sin (α - β) = 1 / 3
def cos_sin_condition : Prop := cos α * sin β = 1 / 6

-- The main theorem 
theorem cos_double_angle (h₁ : sin_diff_condition) (h₂ : cos_sin_condition) : cos (2 * α + 2 * β) = 1 / 9 :=
by sorry

end cos_double_angle_l268_268473


namespace range_of_a_l268_268028

noncomputable def f (a e x : ℝ) : ℝ := 2 * a^x - e * x^2

theorem range_of_a {a e x1 x2 : ℝ} 
  (h1 : 0 < a)
  (h2 : a ≠ 1)
  (h3 : x1 < x2)
  (hx1_min : ∀ x, f a e x ≥ f a e x1)
  (hx2_max : ∀ x, f a e x ≤ f a e x2) :
  ∀ b, (∃ a, (1 / real.exp 1 < a) ∧ (a < 1) ∧ a = b) :=
sorry

end range_of_a_l268_268028


namespace part1_part2_l268_268859

variable (a b c : ℝ)

theorem part1 (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) : abc ≤ 1/9 :=
by
  sorry

theorem part2 (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
    (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ 1 / (2 * sqrt(abc)) :=
by
  sorry

end part1_part2_l268_268859


namespace height_of_inscribed_cylinder_l268_268759

-- Define the necessary properties and conditions
variable (radius_cylinder : ℝ) (radius_hemisphere : ℝ)
variable (parallel_bases : Prop)

-- Assume given conditions
axiom h1 : radius_cylinder = 3
axiom h2 : radius_hemisphere = 7
axiom h3 : parallel_bases = true

-- The statement that needs to be proved
theorem height_of_inscribed_cylinder : parallel_bases → sqrt (radius_hemisphere ^ 2 - radius_cylinder ^ 2) = sqrt 40 :=
  by
    intros _
    rw [h1, h2]
    simp
    sorry  -- Proof omitted

end height_of_inscribed_cylinder_l268_268759


namespace largest_composite_not_written_l268_268345

theorem largest_composite_not_written (n : ℕ) (hn : n = 2022) : ¬ ∃ d > 1, 2033 = n + d := 
by
  sorry

end largest_composite_not_written_l268_268345


namespace problem_1_problem_2_l268_268829

variable (a b c : ℝ)

theorem problem_1 
  (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  a * b * c ≤ 1 / 9 :=
sorry

theorem problem_2 
  (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * (a * b * c)^(1/2)) :=
sorry

end problem_1_problem_2_l268_268829


namespace train_cross_tunnel_time_l268_268769

noncomputable def train_length : ℝ := 800 -- in meters
noncomputable def train_speed : ℝ := 78 * 1000 / 3600 -- converted to meters per second
noncomputable def tunnel_length : ℝ := 500 -- in meters
noncomputable def total_distance : ℝ := train_length + tunnel_length -- total distance to travel

theorem train_cross_tunnel_time : total_distance / train_speed / 60 = 1 := by
  sorry

end train_cross_tunnel_time_l268_268769


namespace days_worked_together_l268_268089

theorem days_worked_together (W : ℝ) (h1 : ∀ (a b : ℝ), (a + b) * 40 = W) 
                             (h2 : ∀ a, a * 16 = W) 
                             (x : ℝ) 
                             (h3 : (x * (W / 40) + 12 * (W / 16)) = W) : 
                             x = 10 := 
by
  sorry

end days_worked_together_l268_268089


namespace journey_total_distance_l268_268048

-- Define the conditions
def miles_already_driven : ℕ := 642
def miles_to_drive : ℕ := 558

-- The total distance of the journey
def total_distance : ℕ := miles_already_driven + miles_to_drive

-- Prove that the total distance of the journey equals 1200 miles
theorem journey_total_distance : total_distance = 1200 := 
by
  -- here the proof would go
  sorry

end journey_total_distance_l268_268048


namespace part1_part2_l268_268970

variable (a b c : ℝ)

-- Condition: a, b, c are positive numbers and a^(3/2) + b^(3/2) + c^(3/2) = 1
axiom pos_a : a > 0
axiom pos_b : b > 0
axiom pos_c : c > 0
axiom sum_condition : a^(3/2) + b^(3/2) + c^(3/2) = 1

-- Proof Problem (1): abc ≤ 1/9
theorem part1 : abc ≤ 1/9 := 
by {
  have h := sorry,
  exact h
}

-- Proof Problem (2): a/(b+c) + b/(a+c) + c/(a+b) ≤ 1/(2sqrt(abc))
theorem part2 : a/(b+c) + b/(a+c) + c/(a+b) ≤ 1/(2*sqrt(abc)) := 
by {
  have h := sorry,
  exact h
}

end part1_part2_l268_268970


namespace problem1_problem2_l268_268839

variable {a b c : ℝ}

-- Conditions: a, b, c are positive numbers and a^(3/2) + b^(3/2) + c^(3/2) = 1
axiom pos_a : 0 < a
axiom pos_b : 0 < b
axiom pos_c : 0 < c
axiom sum_pow : a^(3/2) + b^(3/2) + c^(3/2) = 1

-- Problem (1): Prove abc ≤ 1/9 given the conditions
theorem problem1 : abc ≤ 1 / 9 := by
  sorry

-- Problem (2): Prove (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ (1 / (2 * (abc) ^ (1/2))) given the conditions
theorem problem2 : (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ 1 / (2 * (abc) ^ (1/2)) := by
  sorry

end problem1_problem2_l268_268839


namespace calculate_f_at_2_l268_268193

def f (x : ℝ) : ℝ := 15 * x ^ 5 - 24 * x ^ 4 + 33 * x ^ 3 - 42 * x ^ 2 + 51 * x

theorem calculate_f_at_2 : f 2 = 294 := by
  sorry

end calculate_f_at_2_l268_268193


namespace A_alone_days_l268_268267

noncomputable def days_for_A (r_A r_B r_C : ℝ) : ℝ :=
  1 / r_A

theorem A_alone_days
  (r_A r_B r_C : ℝ) 
  (h1 : r_A + r_B = 1 / 3)
  (h2 : r_B + r_C = 1 / 6)
  (h3 : r_A + r_C = 1 / 4) :
  days_for_A r_A r_B r_C = 4.8 := by
  sorry

end A_alone_days_l268_268267


namespace leading_digit_not_necessarily_one_l268_268092

-- Define a condition to check if the leading digit of a number is the same
def same_leading_digit (x: ℕ) (n: ℕ) : Prop :=
  (Nat.digits 10 x).head? = (Nat.digits 10 (x^n)).head?

-- Theorem stating the digit does not need to be 1 under given conditions
theorem leading_digit_not_necessarily_one :
  (∃ x: ℕ, x > 1 ∧ same_leading_digit x 2 ∧ same_leading_digit x 3) ∧ 
  (∃ x: ℕ, x > 1 ∧ ∀ n: ℕ, 1 ≤ n ∧ n ≤ 2015 → same_leading_digit x n) :=
sorry

end leading_digit_not_necessarily_one_l268_268092


namespace family_savings_amount_l268_268619

theorem family_savings_amount : 
  let total := 2000 
  let given_to_wife := 2 / 5 * total 
  let remaining_after_wife := total - given_to_wife 
  let given_to_first_son := 2 / 5 * remaining_after_wife 
  let remaining_after_first_son := remaining_after_wife - given_to_first_son 
  let given_to_second_son := 40 / 100 * remaining_after_first_son 
  let remaining_amount := remaining_after_first_son - given_to_second_son 
  in remaining_amount = 432 := 
by
  sorry

end family_savings_amount_l268_268619


namespace proof_part1_proof_part2_l268_268847

theorem proof_part1 (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) 
                    (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
                    (a * b * c ≤ 1 / 9) := 
by
  sorry

theorem proof_part2 (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) 
                    (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
                    (a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * (a * b * c)^(1/2))) := 
by
  sorry

end proof_part1_proof_part2_l268_268847


namespace probability_is_half_l268_268574

-- Define the set of numbers from 1 to 30
def numbers : Finset ℕ := (Finset.range 30).map ⟨Nat.succ, Nat.succ_injective⟩

-- Define the set of multiples of 3 from 1 to 30
def multiples_of_3 : Finset ℕ := numbers.filter (λ n, n % 3 = 0)

-- Define the set of multiples of 4 from 1 to 30
def multiples_of_4 : Finset ℕ := numbers.filter (λ n, n % 4 = 0)

-- Define the set of multiples of 12 from 1 to 30 (multiples of both 3 and 4)
def multiples_of_12 : Finset ℕ := numbers.filter (λ n, n % 12 = 0)

-- Calculate the probability using the principle of inclusion-exclusion
def favorable_outcomes : ℕ := multiples_of_3.card + multiples_of_4.card - multiples_of_12.card

-- Total number of outcomes
def total_outcomes : ℕ := numbers.card

-- Calculate the probability
def probability : ℚ := favorable_outcomes / total_outcomes

-- Prove that the probability is 1/2
theorem probability_is_half : probability = 1 / 2 := by
  sorry

end probability_is_half_l268_268574


namespace banana_count_l268_268408

-- Variables representing the number of bananas, oranges, and apples
variables (B O A : ℕ)

-- Conditions translated from the problem statement
def conditions : Prop :=
  (O = 2 * B) ∧
  (A = 2 * O) ∧
  (B + O + A = 35)

-- Theorem to prove the number of bananas is 5 given the conditions
theorem banana_count (B O A : ℕ) (h : conditions B O A) : B = 5 :=
sorry

end banana_count_l268_268408


namespace evaluate_polynomial_given_condition_l268_268008

theorem evaluate_polynomial_given_condition :
  ∀ x : ℝ, x > 0 → x^2 - 2 * x - 8 = 0 → (x^3 - 2 * x^2 - 8 * x + 4 = 4) := 
by
  intro x hx hcond
  sorry

end evaluate_polynomial_given_condition_l268_268008


namespace biggest_number_l268_268426

theorem biggest_number (A B C D : ℕ) (h1 : A / B = 2 / 3) (h2 : B / C = 3 / 4) (h3 : C / D = 4 / 5) (h4 : A + B + C + D = 1344) : D = 480 := 
sorry

end biggest_number_l268_268426


namespace part1_part2_l268_268957

def pos_real := {x : ℝ // 0 < x}

variables (a b c : pos_real)

theorem part1 (h : a.val ^ (3/2) + b.val ^ (3/2) + c.val ^ (3/2) = 1) : a.val * b.val * c.val ≤ 1 / 9 :=
sorry

theorem part2 (h : a.val ^ (3/2) + b.val ^ (3/2) + c.val ^ (3/2) = 1) : 
  a.val / (b.val + c.val) + b.val / (a.val + c.val) + c.val / (a.val + b.val) ≤ 1 / (2 * Real.sqrt (a.val * b.val * c.val)) :=
sorry

end part1_part2_l268_268957


namespace range_of_a_for_decreasing_function_l268_268031

noncomputable def f (a x : ℝ) : ℝ :=
  if x ≤ 1 then (a - 2) * x + 4 else 3 * a / x

theorem range_of_a_for_decreasing_function :
  (∀ x1 x2 : ℝ, x1 ≤ x2 → f a x1 ≥ f a x2) ↔ (0 < a ∧ a ≤ 1) :=
sorry

end range_of_a_for_decreasing_function_l268_268031


namespace inequality_and_equality_condition_l268_268199

theorem inequality_and_equality_condition (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_ab : 1 ≤ a * b) :
  (1 / (1 + a) + 1 / (1 + b) ≤ 1) ∧ (1 / (1 + a) + 1 / (1 + b) = 1 ↔ a * b = 1) :=
by
  sorry

end inequality_and_equality_condition_l268_268199


namespace equal_naturals_of_infinite_divisibility_l268_268096

theorem equal_naturals_of_infinite_divisibility
  (a b : ℕ)
  (h : ∀ᶠ n in Filter.atTop, (a^(n + 1) + b^(n + 1)) % (a^n + b^n) = 0) :
  a = b :=
sorry

end equal_naturals_of_infinite_divisibility_l268_268096


namespace problem_1_problem_2_l268_268832

variable (a b c : ℝ)

theorem problem_1 
  (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  a * b * c ≤ 1 / 9 :=
sorry

theorem problem_2 
  (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * (a * b * c)^(1/2)) :=
sorry

end problem_1_problem_2_l268_268832


namespace student_attends_all_three_l268_268044

open Finset

variables (F G C : Finset ℕ) (n : ℕ)

theorem student_attends_all_three (hF : F.card = 22) (hG : G.card = 21) 
                               (hC : C.card = 18) (hn : n = 30) : 
  ∃ s, s ∈ F ∧ s ∈ G ∧ s ∈ C :=
sorry

end student_attends_all_three_l268_268044


namespace solve_inequality_l268_268718

-- Define the conditions
def condition_inequality (x : ℝ) : Prop := abs x + abs (2 * x - 3) ≥ 6

-- Define the solution set form
def solution_set (x : ℝ) : Prop := x ≤ -1 ∨ x ≥ 3

-- State the theorem
theorem solve_inequality (x : ℝ) : condition_inequality x → solution_set x := 
by 
  sorry

end solve_inequality_l268_268718


namespace petya_digits_sum_l268_268370

def distinct_digits (a b c d : ℕ) : Prop :=
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d

theorem petya_digits_sum (a b c d : ℕ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) (h4 : d ≠ 0)
    (h_distinct: distinct_digits a b c d)
    (h_sum: 6 * (a + b + c + d) * 1111 = 73326) :
    a + b + c + d = 11 ∧ {a, b, c, d} = {1, 2, 3, 5} :=
by
  sorry

end petya_digits_sum_l268_268370


namespace Aunt_Lucy_gift_correct_l268_268690

def Jade_initial : ℕ := 38
def Julia_initial : ℕ := Jade_initial / 2
def Jack_initial : ℕ := 12
def John_initial : ℕ := 15
def Jane_initial : ℕ := 20

def Aunt_Mary_gift : ℕ := 65
def Aunt_Susan_gift : ℕ := 70

def total_initial : ℕ :=
  Jade_initial + Julia_initial + Jack_initial + John_initial + Jane_initial

def total_after_gifts : ℕ := 225
def total_gifts : ℕ := total_after_gifts - total_initial
def Aunt_Lucy_gift : ℕ := total_gifts - (Aunt_Mary_gift + Aunt_Susan_gift)

theorem Aunt_Lucy_gift_correct :
  Aunt_Lucy_gift = total_after_gifts - total_initial - (Aunt_Mary_gift + Aunt_Susan_gift) := by
  sorry

end Aunt_Lucy_gift_correct_l268_268690


namespace largest_4_digit_congruent_to_17_mod_26_l268_268252

theorem largest_4_digit_congruent_to_17_mod_26 :
  ∃ x, x < 10000 ∧ x ≥ 1000 ∧ x % 26 = 17 ∧ (∀ y, y < 10000 ∧ y ≥ 1000 ∧ y % 26 = 17 → y ≤ x) ∧ x = 9972 := 
by
  sorry

end largest_4_digit_congruent_to_17_mod_26_l268_268252


namespace expansion_coefficient_l268_268532

theorem expansion_coefficient :
  ∀ (x : ℝ), (∃ (a₀ a₁ a₂ b : ℝ), x^6 + x^4 = a₀ + a₁ * (x + 2) + a₂ * (x + 2)^2 + b * (x + 2)^3) →
  (a₀ = 0 ∧ a₁ = 0 ∧ a₂ = 0 ∧ b = -168) :=
by
  sorry

end expansion_coefficient_l268_268532


namespace problem_1_problem_2_l268_268826

variable (a b c : ℝ)

theorem problem_1 
  (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  a * b * c ≤ 1 / 9 :=
sorry

theorem problem_2 
  (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * (a * b * c)^(1/2)) :=
sorry

end problem_1_problem_2_l268_268826


namespace tan_alpha_result_l268_268037

theorem tan_alpha_result (α : ℝ) (h : Real.tan (α - Real.pi / 4) = 1 / 6) : Real.tan α = 7 / 5 :=
by
  sorry

end tan_alpha_result_l268_268037


namespace marie_daily_rent_l268_268364

noncomputable def daily_revenue (bread_loaves : ℕ) (bread_price : ℝ) (cakes : ℕ) (cake_price : ℝ) : ℝ :=
  bread_loaves * bread_price + cakes * cake_price

noncomputable def total_profit (daily_revenue : ℝ) (days : ℕ) (cash_register_cost : ℝ) : ℝ :=
  cash_register_cost

noncomputable def daily_profit (total_profit : ℝ) (days : ℕ) : ℝ :=
  total_profit / days

noncomputable def daily_profit_after_electricity (daily_profit : ℝ) (electricity_cost : ℝ) : ℝ :=
  daily_profit - electricity_cost

noncomputable def daily_rent (daily_revenue : ℝ) (daily_profit_after_electricity : ℝ) : ℝ :=
  daily_revenue - daily_profit_after_electricity

theorem marie_daily_rent
  (bread_loaves : ℕ) (bread_price : ℝ) (cakes : ℕ) (cake_price : ℝ)
  (days : ℕ) (cash_register_cost : ℝ) (electricity_cost : ℝ) :
  bread_loaves = 40 → bread_price = 2 → cakes = 6 → cake_price = 12 →
  days = 8 → cash_register_cost = 1040 → electricity_cost = 2 →
  daily_rent (daily_revenue bread_loaves bread_price cakes cake_price)
             (daily_profit_after_electricity (daily_profit (total_profit (daily_revenue bread_loaves bread_price cakes cake_price) days cash_register_cost) days) electricity_cost) = 24 :=
by
  intros h0 h1 h2 h3 h4 h5 h6
  sorry

end marie_daily_rent_l268_268364


namespace sin_sum_leq_3_sqrt3_over_2_l268_268177

theorem sin_sum_leq_3_sqrt3_over_2 
  (A B C : ℝ) 
  (h₁ : A + B + C = Real.pi) 
  (h₂ : 0 < A ∧ A < Real.pi)
  (h₃ : 0 < B ∧ B < Real.pi)
  (h₄ : 0 < C ∧ C < Real.pi) :
  Real.sin A + Real.sin B + Real.sin C ≤ 3 * Real.sqrt 3 / 2 :=
sorry

end sin_sum_leq_3_sqrt3_over_2_l268_268177


namespace trig_identity_evaluation_l268_268290

theorem trig_identity_evaluation :
  4 * Real.cos (50 * Real.pi / 180) - Real.tan (40 * Real.pi / 180) = Real.sqrt 3 :=
by
  sorry

end trig_identity_evaluation_l268_268290


namespace max_min_values_l268_268016

open Real

noncomputable def circle_condition (x y : ℝ) :=
  (x - 3) ^ 2 + (y - 3) ^ 2 = 6

theorem max_min_values (x y : ℝ) (hx : circle_condition x y) :
  ∃ k k' d d', 
    k = 3 + 2 * sqrt 2 ∧
    k' = 3 - 2 * sqrt 2 ∧
    k = y / x ∧
    k' = y / x ∧
    d = sqrt ((x - 2) ^ 2 + y ^ 2) ∧
    d' = sqrt ((x - 2) ^ 2 + y ^ 2) ∧
    d = sqrt (10) + sqrt (6) ∧
    d' = sqrt (10) - sqrt (6) :=
sorry

end max_min_values_l268_268016


namespace find_quad_function_l268_268656

-- Define the quadratic function with the given conditions
def quad_function (a b c : ℝ) (f : ℝ → ℝ) :=
  ∀ x, f x = a * x^2 + b * x + c

-- Define the values y(-2) = -3, y(-1) = -4, y(0) = -3, y(2) = 5
def given_points (f : ℝ → ℝ) :=
  f (-2) = -3 ∧ f (-1) = -4 ∧ f 0 = -3 ∧ f 2 = 5

-- Prove that y = x^2 + 2x - 3 satisfies the given points
theorem find_quad_function : ∃ f : ℝ → ℝ, (quad_function 1 2 (-3) f) ∧ (given_points f) :=
by
  sorry

end find_quad_function_l268_268656


namespace coin_toss_probability_l268_268102

open ProbabilityTheory
open MeasureTheory

noncomputable def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (nat.choose n k) * (p^k) * ((1 - p)^(n - k))

theorem coin_toss_probability :
  binomial_probability 5 2 0.5 = 0.3125 := 
by sorry

end coin_toss_probability_l268_268102


namespace x2_plus_y2_lt_1_l268_268209

theorem x2_plus_y2_lt_1 (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x^3 + y^3 = x - y) : x^2 + y^2 < 1 :=
sorry

end x2_plus_y2_lt_1_l268_268209


namespace pascal_row_20_fifth_sixth_sum_l268_268465

-- Conditions from the problem
def pascal_element (n k : ℕ) : ℕ := Nat.choose n k

-- Question translated to a Lean theorem
theorem pascal_row_20_fifth_sixth_sum :
  pascal_element 20 4 + pascal_element 20 5 = 20349 :=
by
  sorry

end pascal_row_20_fifth_sixth_sum_l268_268465


namespace euler_criterion_l268_268535

theorem euler_criterion (p : ℕ) (a : ℕ) (hp : Nat.Prime p) (hp_gt_two : p > 2) (ha : 1 ≤ a ∧ a ≤ p - 1) : 
  (∃ b : ℕ, b^2 % p = a % p) ↔ a^((p - 1) / 2) % p = 1 :=
sorry

end euler_criterion_l268_268535


namespace part1_part2_l268_268870

variable (a b c : ℝ)

theorem part1 (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) : abc ≤ 1/9 :=
by
  sorry

theorem part2 (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
    (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ 1 / (2 * sqrt(abc)) :=
by
  sorry

end part1_part2_l268_268870


namespace part1_part2_l268_268974

variable (a b c : ℝ)

-- Condition: a, b, c are positive numbers and a^(3/2) + b^(3/2) + c^(3/2) = 1
axiom pos_a : a > 0
axiom pos_b : b > 0
axiom pos_c : c > 0
axiom sum_condition : a^(3/2) + b^(3/2) + c^(3/2) = 1

-- Proof Problem (1): abc ≤ 1/9
theorem part1 : abc ≤ 1/9 := 
by {
  have h := sorry,
  exact h
}

-- Proof Problem (2): a/(b+c) + b/(a+c) + c/(a+b) ≤ 1/(2sqrt(abc))
theorem part2 : a/(b+c) + b/(a+c) + c/(a+b) ≤ 1/(2*sqrt(abc)) := 
by {
  have h := sorry,
  exact h
}

end part1_part2_l268_268974


namespace carl_personal_owe_l268_268291

def property_damage : ℝ := 40000
def medical_bills : ℝ := 70000
def insurance_coverage : ℝ := 0.8
def carl_responsibility : ℝ := 0.2
def total_cost : ℝ := property_damage + medical_bills
def carl_owes : ℝ := total_cost * carl_responsibility

theorem carl_personal_owe : carl_owes = 22000 := by
  sorry

end carl_personal_owe_l268_268291


namespace abc_le_one_ninth_sum_ratios_le_one_over_two_sqrt_abc_l268_268814

open Real

variables (a b c : ℝ)

-- Condition: a, b, c are positive numbers
variables (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
-- Condition: a^(3/2) + b^(3/2) + c^(3/2) = 1
variable  (h_sum : a^(3/2) + b^(3/2) + c^(3/2) = 1)

-- Question 1: Prove abc ≤ 1/9
theorem abc_le_one_ninth : a * b * c ≤ 1 / 9 :=
  sorry

-- Question 2: Prove (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ 1 / (2 * sqrt (a * b * c))
theorem sum_ratios_le_one_over_two_sqrt_abc : 
  (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ 1 / (2 * sqrt (a * b * c)) :=
  sorry

end abc_le_one_ninth_sum_ratios_le_one_over_two_sqrt_abc_l268_268814


namespace boys_went_down_the_slide_total_l268_268269

/-- Conditions -/
def a : Nat := 87
def b : Nat := 46
def c : Nat := 29

/-- The main proof problem -/
theorem boys_went_down_the_slide_total :
  a + b + c = 162 :=
by
  sorry

end boys_went_down_the_slide_total_l268_268269


namespace function_identity_l268_268622

theorem function_identity {f : ℕ → ℕ} (h₀ : f 1 > 0) 
  (h₁ : ∀ m n : ℕ, f (m^2 + n^2) = f m^2 + f n^2) : 
  ∀ n : ℕ, f n = n :=
by
  sorry

end function_identity_l268_268622


namespace negation_equiv_l268_268508

open Nat

theorem negation_equiv (P : Prop) :
  (¬ (∃ n : ℕ, (n! * n!) > (2^n))) ↔ (∀ n : ℕ, (n! * n!) ≤ (2^n)) :=
by
  sorry

end negation_equiv_l268_268508


namespace remaining_hard_hats_l268_268180

theorem remaining_hard_hats 
  (pink_initial : ℕ)
  (green_initial : ℕ)
  (yellow_initial : ℕ)
  (carl_takes_pink : ℕ)
  (john_takes_pink : ℕ)
  (john_takes_green : ℕ) :
  john_takes_green = 2 * john_takes_pink →
  pink_initial = 26 →
  green_initial = 15 →
  yellow_initial = 24 →
  carl_takes_pink = 4 →
  john_takes_pink = 6 →
  ∃ pink_remaining green_remaining yellow_remaining total_remaining, 
    pink_remaining = pink_initial - carl_takes_pink - john_takes_pink ∧
    green_remaining = green_initial - john_takes_green ∧
    yellow_remaining = yellow_initial ∧
    total_remaining = pink_remaining + green_remaining + yellow_remaining ∧
    total_remaining = 43 :=
by
  sorry

end remaining_hard_hats_l268_268180


namespace junior_girls_count_l268_268130

def total_players: Nat := 50
def boys_percentage: Real := 0.60
def girls_percentage: Real := 1.0 - boys_percentage
def half: Real := 0.5
def number_of_girls: Nat := (total_players: Real) * girls_percentage |> Nat.floor
def junior_girls: Nat := (number_of_girls: Real) * half |> Nat.floor

theorem junior_girls_count : junior_girls = 10 := by
  sorry

end junior_girls_count_l268_268130


namespace part1_part2_l268_268861

variable (a b c : ℝ)

theorem part1 (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) : abc ≤ 1/9 :=
by
  sorry

theorem part2 (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
    (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ 1 / (2 * sqrt(abc)) :=
by
  sorry

end part1_part2_l268_268861


namespace stratified_sampling_medium_stores_l268_268178

noncomputable def total_stores := 300
noncomputable def large_stores := 30
noncomputable def medium_stores := 75
noncomputable def small_stores := 195
noncomputable def sample_size := 20

theorem stratified_sampling_medium_stores : 
  (medium_stores : ℕ) * (sample_size : ℕ) / (total_stores : ℕ) = 5 :=
by
  sorry

end stratified_sampling_medium_stores_l268_268178


namespace range_of_x_f_greater_than_4_l268_268056

noncomputable def f (x : ℝ) : ℝ :=
  if x < 1 then 2^(-x) else x^2

theorem range_of_x_f_greater_than_4 :
  { x : ℝ | f x > 4 } = { x : ℝ | x < -2 ∨ x > 2 } :=
by
  sorry

end range_of_x_f_greater_than_4_l268_268056


namespace part1_part2_l268_268885

variables {a b c : ℝ}

theorem part1 (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h : a^(3/2) + b^(3/2) + c^(3/2) = 1) : abc ≤ 1/9 := 
sorry

theorem part2 (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (abc)) :=
sorry

end part1_part2_l268_268885


namespace proof_part1_proof_part2_l268_268855

theorem proof_part1 (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) 
                    (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
                    (a * b * c ≤ 1 / 9) := 
by
  sorry

theorem proof_part2 (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) 
                    (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
                    (a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * (a * b * c)^(1/2))) := 
by
  sorry

end proof_part1_proof_part2_l268_268855


namespace problem_1_problem_2_l268_268927

variable (a b c : ℝ)

theorem problem_1 
  (hp : 0 < a) 
  (hq : 0 < b) 
  (hr : 0 < c) 
  (hs : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  abc ≤ 1/9 := 
sorry

theorem problem_2 
  (hp : 0 < a) 
  (hq : 0 < b) 
  (hr : 0 < c) 
  (hs : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (abc)) := 
sorry

end problem_1_problem_2_l268_268927


namespace area_of_rectangular_garden_l268_268066

-- Definitions based on conditions
def width : ℕ := 15
def length : ℕ := 3 * width
def area : ℕ := length * width

-- The theorem we want to prove
theorem area_of_rectangular_garden : area = 675 :=
by sorry

end area_of_rectangular_garden_l268_268066


namespace Zenobius_more_descendants_l268_268190

/-- Total number of descendants in King Pafnutius' lineage --/
def descendants_Pafnutius : Nat :=
  2 + 60 * 2 + 20 * 1

/-- Total number of descendants in King Zenobius' lineage --/
def descendants_Zenobius : Nat :=
  4 + 35 * 3 + 35 * 1

theorem Zenobius_more_descendants : descendants_Zenobius > descendants_Pafnutius := by
  sorry

end Zenobius_more_descendants_l268_268190


namespace product_of_roots_l268_268731

theorem product_of_roots : 
  (Real.root 81 4) * (Real.root 27 3) * (Real.sqrt 9) = 27 :=
by
  sorry

end product_of_roots_l268_268731


namespace combined_weight_is_correct_l268_268638

-- Frank and Gwen's candy weights
def frank_candy : ℕ := 10
def gwen_candy : ℕ := 7

-- The combined weight of candy
def combined_weight : ℕ := frank_candy + gwen_candy

-- Theorem that states the combined weight is 17 pounds
theorem combined_weight_is_correct : combined_weight = 17 :=
by
  -- proves that 10 + 7 = 17
  sorry

end combined_weight_is_correct_l268_268638


namespace vasya_lowest_position_l268_268231

noncomputable theory

def number_of_cyclists := 500
def number_of_stages := 15
def position_of_vasya_each_stage := 7

theorem vasya_lowest_position (total_cyclists : ℕ) (stages : ℕ) (position_each_stage : ℕ)
  (h_total_cyclists : total_cyclists = number_of_cyclists)
  (h_stages : stages = number_of_stages)
  (h_position_each_stage : position_each_stage = position_of_vasya_each_stage)
  (no_identical_times : ∀ (i j : ℕ), i ≠ j → ∀ (stage : ℕ), stage ≤ stages → ∀ (t : ℕ), t ≤ total_cyclists → 
    (time : Π (s : ℕ) (c : ℕ), c < total_cyclists → ℕ), 
    time stage i < time stage j ∨ time stage j < time stage i):
  ∃ lowest_pos, lowest_pos = 91 := sorry

end vasya_lowest_position_l268_268231


namespace num_ways_choose_officers_8_l268_268684

def numWaysToChooseOfficers (n : ℕ) : ℕ := n * (n - 1) * (n - 2)

theorem num_ways_choose_officers_8 : numWaysToChooseOfficers 8 = 336 := by
  sorry

end num_ways_choose_officers_8_l268_268684


namespace find_a_l268_268162

theorem find_a (a : ℝ) (h : -1 ^ 2 + 2 * -1 + a = 0) : a = 1 :=
sorry

end find_a_l268_268162


namespace probability_is_half_l268_268576

-- Define the set of numbers from 1 to 30
def numbers : Finset ℕ := (Finset.range 30).map ⟨Nat.succ, Nat.succ_injective⟩

-- Define the set of multiples of 3 from 1 to 30
def multiples_of_3 : Finset ℕ := numbers.filter (λ n, n % 3 = 0)

-- Define the set of multiples of 4 from 1 to 30
def multiples_of_4 : Finset ℕ := numbers.filter (λ n, n % 4 = 0)

-- Define the set of multiples of 12 from 1 to 30 (multiples of both 3 and 4)
def multiples_of_12 : Finset ℕ := numbers.filter (λ n, n % 12 = 0)

-- Calculate the probability using the principle of inclusion-exclusion
def favorable_outcomes : ℕ := multiples_of_3.card + multiples_of_4.card - multiples_of_12.card

-- Total number of outcomes
def total_outcomes : ℕ := numbers.card

-- Calculate the probability
def probability : ℚ := favorable_outcomes / total_outcomes

-- Prove that the probability is 1/2
theorem probability_is_half : probability = 1 / 2 := by
  sorry

end probability_is_half_l268_268576


namespace carl_personal_owe_l268_268292

def property_damage : ℝ := 40000
def medical_bills : ℝ := 70000
def insurance_coverage : ℝ := 0.8
def carl_responsibility : ℝ := 0.2
def total_cost : ℝ := property_damage + medical_bills
def carl_owes : ℝ := total_cost * carl_responsibility

theorem carl_personal_owe : carl_owes = 22000 := by
  sorry

end carl_personal_owe_l268_268292


namespace smallest_x_abs_eq_15_l268_268307

theorem smallest_x_abs_eq_15 :
  ∃ x : ℝ, |5 * x - 3| = 15 ∧ ∀ y : ℝ, |5 * y - 3| = 15 → x ≤ y :=
sorry

end smallest_x_abs_eq_15_l268_268307


namespace variance_of_heights_l268_268311
-- Importing all necessary libraries

-- Define a list of heights
def heights : List ℕ := [160, 162, 159, 160, 159]

-- Define the function to calculate the mean of a list of natural numbers
def mean (list : List ℕ) : ℚ :=
  list.sum / list.length

-- Define the function to calculate the variance of a list of natural numbers
def variance (list : List ℕ) : ℚ :=
  let μ := mean list
  (list.map (λ x => (x - μ) ^ 2)).sum / list.length

-- The theorem statement that proves the variance is 6/5
theorem variance_of_heights : variance heights = 6 / 5 :=
  sorry

end variance_of_heights_l268_268311


namespace czakler_inequality_czakler_equality_pairs_l268_268533

theorem czakler_inequality (x y : ℝ) (h : (x + 1) * (y + 2) = 8) : (xy - 10)^2 ≥ 64 :=
sorry

theorem czakler_equality_pairs (x y : ℝ) (h : (x + 1) * (y + 2) = 8) :
(xy - 10)^2 = 64 ↔ (x, y) = (1,2) ∨ (x, y) = (-3, -6) :=
sorry

end czakler_inequality_czakler_equality_pairs_l268_268533


namespace part1_part2_l268_268978

variable (a b c : ℝ)

-- Condition: a, b, c are positive numbers and a^(3/2) + b^(3/2) + c^(3/2) = 1
axiom pos_a : a > 0
axiom pos_b : b > 0
axiom pos_c : c > 0
axiom sum_condition : a^(3/2) + b^(3/2) + c^(3/2) = 1

-- Proof Problem (1): abc ≤ 1/9
theorem part1 : abc ≤ 1/9 := 
by {
  have h := sorry,
  exact h
}

-- Proof Problem (2): a/(b+c) + b/(a+c) + c/(a+b) ≤ 1/(2sqrt(abc))
theorem part2 : a/(b+c) + b/(a+c) + c/(a+b) ≤ 1/(2*sqrt(abc)) := 
by {
  have h := sorry,
  exact h
}

end part1_part2_l268_268978


namespace start_of_range_l268_268077

variable (x : ℕ)

theorem start_of_range (h : ∃ (n : ℕ), n ≤ 79 ∧ n % 11 = 0 ∧ x = 79 - 3 * 11) 
(h4 : ∀ (k : ℕ), 0 ≤ k ∧ k < 4 → ∃ (y : ℕ), y = 79 - (k * 11) ∧ y % 11 = 0) :
  x = 44 := by
  sorry

end start_of_range_l268_268077


namespace finalize_proof_l268_268608

noncomputable def factorial_proof_problem : ℝ :=
  (Nat.factorial 9 ^ 2) / Real.sqrt (Nat.factorial 6) + (3 / 7 * 4 ^ 3)

theorem finalize_proof : factorial_proof_problem = 4906624027 :=
  by
    sorry

end finalize_proof_l268_268608


namespace part_a_part_b_l268_268424

noncomputable def same_start_digit (n x : ℕ) : Prop :=
  ∃ d : ℕ, ∀ k : ℕ, (k ≤ n) → (x * 10^(k-1) ≤ d * 10^(k-1) + 10^(k-1) - 1) ∧ ((d * 10^(k-1)) < x * 10^(k-1))

theorem part_a (x : ℕ) : 
  (same_start_digit 3 x) → ¬(∃ d : ℕ, d = 1) → false :=
  sorry

theorem part_b (x : ℕ) : 
  (same_start_digit 2015 x) → ¬(∃ d : ℕ, d = 1) → false :=
  sorry

end part_a_part_b_l268_268424


namespace part1_part2_l268_268868

variable (a b c : ℝ)

theorem part1 (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) : abc ≤ 1/9 :=
by
  sorry

theorem part2 (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
    (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ 1 / (2 * sqrt(abc)) :=
by
  sorry

end part1_part2_l268_268868


namespace alyssa_allowance_l268_268285

-- Definition using the given problem
def weekly_allowance (A : ℝ) : Prop :=
  A / 2 + 8 = 12

-- Theorem to prove that weekly allowance is 8 dollars
theorem alyssa_allowance : ∃ A : ℝ, weekly_allowance A ∧ A = 8 :=
by
  use 8
  unfold weekly_allowance
  exact eq.refl _

end alyssa_allowance_l268_268285


namespace range_of_a_l268_268022

open Real

noncomputable def f (a x : ℝ) : ℝ := 2 * a^x - exp 1 * x^2

theorem range_of_a {x1 x2 : ℝ} {a : ℝ} (h1 : a > 0) (h2 : a ≠ 1)
    (h3 : x1 < x2)
    (h4 : ∀ x, deriv (f a) x = 0 ↔ x = x1 ∨ x = x2)
    (h5 : ∀ x, deriv (f a) x1 < 0 ∧ deriv (f a) x2 > 0)
    (h6 : deriv (f a) x1 = 0 ∧ deriv (f a) x2 = 0) :
    a ∈ Ioo (1 / exp 1) 1 :=
sorry

end range_of_a_l268_268022


namespace prob_of_king_or_queen_top_l268_268521

/-- A standard deck comprises 52 cards, with 13 ranks and 4 suits, each rank having one card per suit. -/
def standard_deck : Set (String × String) :=
Set.prod { "Ace", "2", "3", "4", "5", "6", "7", "8", "9", "10", "Jack", "Queen", "King" }
          { "Hearts", "Diamonds", "Clubs", "Spades" }

/-- There are four cards of rank King and four of rank Queen in the standard deck. -/
def count_kings_and_queens : Nat := 
4 + 4

/-- The total number of cards in a standard deck is 52. -/
def total_cards : Nat := 52

/-- The probability that the top card is either a King or a Queen is 2/13. -/
theorem prob_of_king_or_queen_top :
  (count_kings_and_queens / total_cards : ℚ) = (2 / 13 : ℚ) :=
sorry

end prob_of_king_or_queen_top_l268_268521


namespace probability_three_different_suits_is_169_over_425_l268_268451

def card_deck := Finset (Fin 52)
def suits : Finset (Fin 4) := {0, 1, 2, 3}
def cards_of_suit (s : Fin 4) : Finset (Fin 52) := sorry -- Assume we have a function to give cards of each suit

noncomputable def probability_three_different_suits : ℚ :=
  let total_cards := (52 : ℕ)
  let first_prob := 1
  let second_prob := (39 : 51 : ℚ)
  let third_prob := (26 : 50 : ℚ)
  first_prob * second_prob * third_prob

theorem probability_three_different_suits_is_169_over_425 :
  probability_three_different_suits = (169 : 425 : ℚ) := 
by {
  sorry
}

end probability_three_different_suits_is_169_over_425_l268_268451


namespace compare_P_Q_l268_268627

-- Define the structure of the number a with 2010 digits of 1
def a := 10^2010 - 1

-- Define P and Q based on a
def P := 24 * a^2
def Q := 24 * a^2 + 4 * a

-- Define the theorem to compare P and Q
theorem compare_P_Q : Q > P := by
  sorry

end compare_P_Q_l268_268627


namespace hens_count_l268_268752

theorem hens_count (H C : ℕ) (h_heads : H + C = 60) (h_feet : 2 * H + 4 * C = 200) : H = 20 :=
by
  sorry

end hens_count_l268_268752


namespace cos_of_double_angles_l268_268487

theorem cos_of_double_angles (α β : ℝ) 
  (h1 : Real.sin (α - β) = 1 / 3) 
  (h2 : Real.cos α * Real.sin β = 1 / 6) : 
  Real.cos (2 * α + 2 * β) = 1 / 9 :=
by
  sorry

end cos_of_double_angles_l268_268487


namespace ball_hits_ground_at_l268_268558

variable (t : ℚ) 

def height_eqn (t : ℚ) : ℚ :=
  -16 * t^2 + 30 * t + 50

theorem ball_hits_ground_at :
  (height_eqn t = 0) -> t = 47 / 16 :=
by
  sorry

end ball_hits_ground_at_l268_268558


namespace part1_part2_l268_268987

noncomputable theory

variables {a b c : ℝ}

-- Condition that a, b, c are positive numbers and the given equation holds
def pos_numbers : Prop := a > 0 ∧ b > 0 ∧ c > 0
def sum_condition : Prop := (a ^ (3/2)) + (b ^ (3/2)) + (c ^ (3/2)) = 1

-- Proof goal for the first part: abc <= 1/9
theorem part1 (h₁ : pos_numbers) (h₂ : sum_condition) : 
  a * b * c ≤ 1 / 9 :=
sorry

-- Proof goal for the second part: a/(b+c) + b/(a+c) + c/(a+b) <= 1/(2 * sqrt(abc))
theorem part2 (h₁ : pos_numbers) (h₂ : sum_condition) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * sqrt (a * b * c)) :=
sorry

end part1_part2_l268_268987


namespace statement_C_correct_l268_268736

theorem statement_C_correct (a b : ℝ) (h1 : a < b) (h2 : a * b ≠ 0) : (1 / a) > (1 / b) :=
sorry

end statement_C_correct_l268_268736


namespace consecutive_integer_sum_l268_268157

noncomputable def sqrt17 : ℝ := Real.sqrt 17

theorem consecutive_integer_sum : ∃ (a b : ℤ), (b = a + 1) ∧ (a < sqrt17 ∧ sqrt17 < b) ∧ (a + b = 9) :=
by
  sorry

end consecutive_integer_sum_l268_268157


namespace probability_multiple_of_3_or_4_l268_268573

-- Given the numbers 1 through 30 are written on 30 cards one number per card,
-- and Sara picks one of the 30 cards at random,
-- the probability that the number on her card is a multiple of 3 or 4 is 1/2.

-- Define the set of numbers from 1 to 30
def numbers := finset.range 30 \ {0}

-- Define what it means to be a multiple of 3 or 4 within the given range
def is_multiple_of_3_or_4 (n : ℕ) : Prop :=
  n % 3 = 0 ∨ n % 4 = 0

-- Define the set of multiples of 3 or 4 within the given range
def multiples_of_3_or_4 := numbers.filter is_multiple_of_3_or_4

-- The probability calculation
theorem probability_multiple_of_3_or_4 : 
  (multiples_of_3_or_4.card : ℚ) / numbers.card = 1 / 2 :=
begin
  -- The set multiples_of_3_or_4 contains 15 elements
  have h_multiples_card : multiples_of_3_or_4.card = 15, sorry,
  -- The set numbers contains 30 elements
  have h_numbers_card : numbers.card = 30, sorry,
  -- Therefore, the probability is 15/30 = 1/2
  rw [h_multiples_card, h_numbers_card],
  norm_num,
end

end probability_multiple_of_3_or_4_l268_268573


namespace otimes_computation_l268_268797

-- Definition of ⊗ given m
def otimes (a b m : ℕ) : ℚ := (m * a + b) / (2 * a * b)

-- The main theorem we need to prove
theorem otimes_computation (m : ℕ) (h : otimes 1 4 m = otimes 2 3 m) :
  otimes 3 4 6 = 11 / 12 :=
sorry

end otimes_computation_l268_268797


namespace find_denomination_of_bills_l268_268590

variables 
  (bills_13 : ℕ)  -- Denomination of the bills Tim has 13 of
  (bills_5 : ℕ := 5)  -- Denomination of the bills Tim has 11 of, which are $5 bills
  (bills_1 : ℕ := 1)  -- Denomination of the bills Tim has 17 of, which are $1 bills
  (total_amt : ℕ := 128)  -- Total amount Tim needs to pay
  (num_bills_13 : ℕ := 13)  -- Number of bills of unknown denomination
  (num_bills_5 : ℕ := 11)  -- Number of $5 bills
  (num_bills_1 : ℕ := 17)  -- Number of $1 bills
  (min_bills : ℕ := 16)  -- Minimum number of bills to be used

theorem find_denomination_of_bills : 
  num_bills_13 * bills_13 + num_bills_5 * bills_5 + num_bills_1 * bills_1 = total_amt →
  num_bills_13 + num_bills_5 + num_bills_1 ≥ min_bills → 
  bills_13 = 4 :=
by
  intros h1 h2
  sorry

end find_denomination_of_bills_l268_268590


namespace remaining_amount_to_pay_l268_268271

-- Define the constants and conditions
def total_cost : ℝ := 1300
def first_deposit : ℝ := 0.10 * total_cost
def second_deposit : ℝ := 2 * first_deposit
def promotional_discount : ℝ := 0.05 * total_cost
def interest_rate : ℝ := 0.02

-- Define the function to calculate the final payment
def final_payment (total_cost first_deposit second_deposit promotional_discount interest_rate : ℝ) : ℝ :=
  let total_paid := first_deposit + second_deposit
  let remaining_balance := total_cost - total_paid
  let remaining_after_discount := remaining_balance - promotional_discount
  remaining_after_discount * (1 + interest_rate)

-- Define the theorem to be proven
theorem remaining_amount_to_pay :
  final_payment total_cost first_deposit second_deposit promotional_discount interest_rate = 861.90 :=
by
  -- The proof goes here
  sorry

end remaining_amount_to_pay_l268_268271


namespace base_12_addition_l268_268450

theorem base_12_addition (A B: ℕ) (hA: A = 10) (hB: B = 11) : 
  8 * 12^2 + A * 12 + 2 + (3 * 12^2 + B * 12 + 7) = 1 * 12^3 + 0 * 12^2 + 9 * 12 + 9 := 
by
  sorry

end base_12_addition_l268_268450


namespace part1_part2_l268_268959

def pos_real := {x : ℝ // 0 < x}

variables (a b c : pos_real)

theorem part1 (h : a.val ^ (3/2) + b.val ^ (3/2) + c.val ^ (3/2) = 1) : a.val * b.val * c.val ≤ 1 / 9 :=
sorry

theorem part2 (h : a.val ^ (3/2) + b.val ^ (3/2) + c.val ^ (3/2) = 1) : 
  a.val / (b.val + c.val) + b.val / (a.val + c.val) + c.val / (a.val + b.val) ≤ 1 / (2 * Real.sqrt (a.val * b.val * c.val)) :=
sorry

end part1_part2_l268_268959


namespace Connie_needs_more_money_l268_268783

-- Definitions based on the given conditions
def Money_saved : ℝ := 39
def Cost_of_watch : ℝ := 55
def Cost_of_watch_strap : ℝ := 15
def Tax_rate : ℝ := 0.08

-- Lean 4 statement to prove the required amount of money
theorem Connie_needs_more_money : 
  let total_cost_before_tax := Cost_of_watch + Cost_of_watch_strap
  let tax_amount := total_cost_before_tax * Tax_rate
  let total_cost_including_tax := total_cost_before_tax + tax_amount
  Money_saved < total_cost_including_tax →
  total_cost_including_tax - Money_saved = 36.60 :=
by
  sorry

end Connie_needs_more_money_l268_268783


namespace speed_of_current_l268_268100

theorem speed_of_current (upstream_time : ℝ) (downstream_time : ℝ) :
    upstream_time = 25 / 60 ∧ downstream_time = 12 / 60 →
    ( (60 / downstream_time - 60 / upstream_time) / 2 ) = 1.3 :=
by
  -- Introduce the conditions
  intro h
  -- Simplify using given facts
  have h1 := h.1
  have h2 := h.2
  -- Calcuation of the speed of current
  sorry

end speed_of_current_l268_268100


namespace find_150th_letter_in_pattern_l268_268592

theorem find_150th_letter_in_pattern : 
  (let sequence := "ABCD";
   sequence.length = 4 → 
   sequence[(150 % 4)] = 'B') :=
by
  sorry

end find_150th_letter_in_pattern_l268_268592


namespace problem_statement_l268_268720

def scientific_notation (n: ℝ) (mantissa: ℝ) (exponent: ℤ) : Prop :=
  n = mantissa * 10 ^ exponent

theorem problem_statement : scientific_notation 320000 3.2 5 :=
by {
  sorry
}

end problem_statement_l268_268720


namespace part_a_part_b_part_c_l268_268200

variable (p : ℕ) (k : ℕ)

theorem part_a (hp : Prime p) (h : p = 4 * k + 1) :
  ∃ x : ℤ, (x^2 + 1) % p = 0 :=
by
  sorry

theorem part_b (hp : Prime p) (h : p = 4 * k + 1)
  (x : ℤ) (r1 r2 s1 s2 : ℕ)
  (hr1 : 0 ≤ r1) (hr2 : 0 ≤ r2) (hr1_lt : r1 < Nat.sqrt p) (hr2_lt : r2 < Nat.sqrt p)
  (hs1 : 0 ≤ s1) (hs2 : 0 ≤ s2) (hs1_lt : s1 < Nat.sqrt p) (hs2_lt : s2 < Nat.sqrt p)
  (hneq : (r1, s1) ≠ (r2, s2)) :
  ∃ (r1 r2 s1 s2 : ℕ), (r1 * x + s1) % p = (r2 * x + s2) % p :=
by
  sorry

theorem part_c (hp : Prime p) (h : p = 4 * k + 1)
  (x : ℤ) (r1 r2 s1 s2 : ℕ)
  (hr1 : 0 ≤ r1) (hr2 : 0 ≤ r2) (hr1_lt : r1 < Nat.sqrt p) (hr2_lt : r2 < Nat.sqrt p)
  (hs1 : 0 ≤ s1) (hs2 : 0 ≤ s2) (hs1_lt : s1 < Nat.sqrt p) (hs2_lt : s2 < Nat.sqrt p)
  (hneq : (r1, s1) ≠ (r2, s2)):
  p = (Int.ofNat (r1 - r2))^2 + (Int.ofNat (s1 - s2))^2 :=
by
  sorry

end part_a_part_b_part_c_l268_268200


namespace find_radius_l268_268155

def setA : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 4}
def setB (r : ℝ) : Set (ℝ × ℝ) := {p | (p.1 - 3)^2 + (p.2 - 4)^2 = r^2}

theorem find_radius (r : ℝ) (h : r > 0) :
  (∃! p : ℝ × ℝ, p ∈ setA ∧ p ∈ setB r) ↔ (r = 3 ∨ r = 7) :=
by
  sorry

end find_radius_l268_268155


namespace different_lists_count_l268_268775

def numberOfLists : Nat := 5

theorem different_lists_count :
  let conditions := ∃ (d : Fin 6 → ℕ), d 0 + d 1 + d 2 + d 3 + d 4 + d 5 = 5 ∧
                                      ∀ i, d i ≤ 5 ∧
                                      ∀ i j, i < j → d i ≥ d j
  conditions →
  numberOfLists = 5 :=
sorry

end different_lists_count_l268_268775


namespace Angelina_drive_time_equation_l268_268777

theorem Angelina_drive_time_equation (t : ℝ) 
    (h_speed1 : ∀ t: ℝ, 70 * t = 70 * t)
    (h_stop : 0.5 = 0.5) 
    (h_speed2 : ∀ t: ℝ, 90 * t = 90 * t) 
    (h_total_distance : 300 = 300) 
    (h_total_time : 4 = 4) 
    : 70 * t + 90 * (3.5 - t) = 300 :=
by
  sorry

end Angelina_drive_time_equation_l268_268777


namespace quadratic_two_real_roots_quadratic_no_real_roots_l268_268324

theorem quadratic_two_real_roots (k : ℝ) :
  (∃ x : ℝ, 2 * x^2 - (4 * k - 1) * x + (2 * k^2 - 1) = 0) → 
  k ≤ 9 / 8 :=
by
  sorry

theorem quadratic_no_real_roots (k : ℝ) :
  ¬ (∃ x : ℝ, 2 * x^2 - (4 * k - 1) * x + (2 * k^2 - 1) = 0) → 
  k > 9 / 8 :=
by
  sorry

end quadratic_two_real_roots_quadratic_no_real_roots_l268_268324


namespace total_cost_of_car_rental_l268_268126

theorem total_cost_of_car_rental :
  ∀ (rental_cost_per_day mileage_cost_per_mile : ℝ) (days rented : ℕ) (miles_driven : ℕ),
  rental_cost_per_day = 30 →
  mileage_cost_per_mile = 0.25 →
  rented = 5 →
  miles_driven = 500 →
  rental_cost_per_day * rented + mileage_cost_per_mile * miles_driven = 275 := by
  sorry

end total_cost_of_car_rental_l268_268126


namespace problem_1_problem_2_l268_268926

variable (a b c : ℝ)

theorem problem_1 
  (hp : 0 < a) 
  (hq : 0 < b) 
  (hr : 0 < c) 
  (hs : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  abc ≤ 1/9 := 
sorry

theorem problem_2 
  (hp : 0 < a) 
  (hq : 0 < b) 
  (hr : 0 < c) 
  (hs : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (abc)) := 
sorry

end problem_1_problem_2_l268_268926


namespace mrs_hilt_apple_pies_l268_268543

-- Given definitions
def total_pies := 30 * 5
def pecan_pies := 16

-- The number of apple pies
def apple_pies := total_pies - pecan_pies

-- The proof statement
theorem mrs_hilt_apple_pies : apple_pies = 134 :=
by
  sorry -- Proof step to be filled

end mrs_hilt_apple_pies_l268_268543


namespace part1_part2_l268_268958

def pos_real := {x : ℝ // 0 < x}

variables (a b c : pos_real)

theorem part1 (h : a.val ^ (3/2) + b.val ^ (3/2) + c.val ^ (3/2) = 1) : a.val * b.val * c.val ≤ 1 / 9 :=
sorry

theorem part2 (h : a.val ^ (3/2) + b.val ^ (3/2) + c.val ^ (3/2) = 1) : 
  a.val / (b.val + c.val) + b.val / (a.val + c.val) + c.val / (a.val + b.val) ≤ 1 / (2 * Real.sqrt (a.val * b.val * c.val)) :=
sorry

end part1_part2_l268_268958


namespace range_of_a_l268_268072

theorem range_of_a (a : ℝ) : (∀ x : ℝ, x^2 - (a^2 + a) * x + a^3 > 0 ↔ (x < a^2 ∨ x > a)) → (0 ≤ a ∧ a ≤ 1) :=
by
  intros h
  sorry

end range_of_a_l268_268072


namespace slope_of_intersection_line_l268_268139

theorem slope_of_intersection_line 
    (x y : ℝ)
    (h1 : x^2 + y^2 - 6*x + 4*y - 20 = 0)
    (h2 : x^2 + y^2 - 2*x - 6*y + 10 = 0) :
    ∃ m : ℝ, m = 0.4 := 
sorry

end slope_of_intersection_line_l268_268139


namespace problem_1_problem_2_l268_268939

variable {a b c : ℝ}

theorem problem_1 (hp : 0 < a ∧ 0 < b ∧ 0 < c)
    (h1 : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
    abc ≤ 1/9 := 
by 
  apply sorry

theorem problem_2 (hp : 0 < a ∧ 0 < b ∧ 0 < c)
    (h1 : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
    a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * real.sqrt (abc)) := 
by 
  apply sorry

end problem_1_problem_2_l268_268939


namespace final_value_of_A_l268_268123

-- Define the initial value of A
def initial_value (A : ℤ) : Prop := A = 15

-- Define the reassignment condition
def reassignment_cond (A : ℤ) : Prop := A = -A + 5

-- The theorem stating that given the initial value and reassignment condition, the final value of A is -10
theorem final_value_of_A (A : ℤ) (h1 : initial_value A) (h2 : reassignment_cond A) : A = -10 := by
  sorry

end final_value_of_A_l268_268123


namespace repunit_polynomial_characterization_l268_268744

noncomputable def is_repunit (n : ℕ) : Prop :=
  ∃ k : ℕ, n = (10^k - 1) / 9

def polynomial_condition (f : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, is_repunit n → is_repunit (f n)

theorem repunit_polynomial_characterization :
  ∀ (f : ℕ → ℕ), polynomial_condition f ↔
  ∃ m r : ℕ, m ≥ 0 ∧ r ≥ 1 - m ∧ ∀ n : ℕ, f n = (10^r * (9 * n + 1)^m - 1) / 9 :=
by
  sorry

end repunit_polynomial_characterization_l268_268744


namespace range_of_f_l268_268734

def f (x : ℝ) : ℝ := 1 / (x^2 + 1)

theorem range_of_f : set.Ioc 0 1 = set_of (λ y, ∃ x : ℝ, f x = y) :=
by
  sorry

end range_of_f_l268_268734


namespace problem_part1_problem_part2_l268_268953

variable {a b c : ℝ}

theorem problem_part1 (h : a > 0 ∧ b > 0 ∧ c > 0) (sum_eq : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  abc ≤ 1 / 9 :=
sorry

theorem problem_part2 (h : a > 0 ∧ b > 0 ∧ c > 0) (sum_eq : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (abc)) :=
sorry

end problem_part1_problem_part2_l268_268953


namespace sin_2theta_value_l268_268666

theorem sin_2theta_value (θ : ℝ) (h : ∑' n, (Real.sin θ)^(2 * n) = 3) : Real.sin (2 * θ) = (2 * Real.sqrt 2) / 3 :=
by
  sorry

end sin_2theta_value_l268_268666


namespace find_k_l268_268062

open Complex

noncomputable def possible_values_of_k (a b c d e : ℂ) (k : ℂ) : Prop :=
  (a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧ e ≠ 0) ∧
  (a * k^4 + b * k^3 + c * k^2 + d * k + e = 0) ∧
  (b * k^4 + c * k^3 + d * k^2 + e * k + a = 0)

theorem find_k (a b c d e : ℂ) (k : ℂ) :
  possible_values_of_k a b c d e k → k^5 = 1 :=
by
  intro h
  sorry

#check find_k

end find_k_l268_268062


namespace part1_part2_l268_268963

def pos_real := {x : ℝ // 0 < x}

variables (a b c : pos_real)

theorem part1 (h : a.val ^ (3/2) + b.val ^ (3/2) + c.val ^ (3/2) = 1) : a.val * b.val * c.val ≤ 1 / 9 :=
sorry

theorem part2 (h : a.val ^ (3/2) + b.val ^ (3/2) + c.val ^ (3/2) = 1) : 
  a.val / (b.val + c.val) + b.val / (a.val + c.val) + c.val / (a.val + b.val) ≤ 1 / (2 * Real.sqrt (a.val * b.val * c.val)) :=
sorry

end part1_part2_l268_268963


namespace no_common_real_root_l268_268624

theorem no_common_real_root (a b : ℚ) : 
  ¬ ∃ (r : ℝ), (r^5 - r - 1 = 0) ∧ (r^2 + a * r + b = 0) :=
by
  sorry

end no_common_real_root_l268_268624


namespace train_speed_in_km_hr_l268_268447

noncomputable def train_length : ℝ := 110
noncomputable def bridge_length : ℝ := 132
noncomputable def crossing_time : ℝ := 9.679225661947045
noncomputable def distance_covered : ℝ := train_length + bridge_length
noncomputable def speed_m_s : ℝ := distance_covered / crossing_time
noncomputable def speed_km_hr : ℝ := speed_m_s * 3.6

theorem train_speed_in_km_hr : speed_km_hr = 90.0216 := by
  sorry

end train_speed_in_km_hr_l268_268447


namespace vertex_parabola_shape_l268_268529

theorem vertex_parabola_shape
  (a d : ℕ) (ha : 0 < a) (hd : 0 < d) :
  ∃ (P : ℝ → ℝ → Prop), 
  (∀ t : ℝ, ∃ (x y : ℝ), P x y ∧ (x = (-t / (2 * a))) ∧ (y = -a * (x^2) + d)) ∧
  (∀ x y : ℝ, P x y ↔ (y = -a * (x^2) + d)) :=
by
  sorry

end vertex_parabola_shape_l268_268529


namespace largest_real_number_mu_l268_268634

noncomputable def largest_mu : ℝ := 13 / 2

theorem largest_real_number_mu (
  a b c d : ℝ
) (h1 : a ≥ 0) (h2 : b ≥ 0) (h3 : c ≥ 0) (h4 : d ≥ 0) :
  (a^2 + b^2 + c^2 + d^2) ≥ (largest_mu * a * b + b * c + 2 * c * d) :=
sorry

end largest_real_number_mu_l268_268634


namespace prob1_prob2_l268_268780

theorem prob1 : -2 + 5 - |(-8 : ℤ)| + (-5) = -10 := 
by
  sorry

theorem prob2 : (-2 : ℤ)^2 * 5 - (-2)^3 / 4 = 22 := 
by
  sorry

end prob1_prob2_l268_268780


namespace factor_x_squared_minus_144_l268_268792

theorem factor_x_squared_minus_144 (x : ℝ) : x^2 - 144 = (x - 12) * (x + 12) :=
by
  sorry

end factor_x_squared_minus_144_l268_268792


namespace optionA_is_multiple_of_5_optionB_is_multiple_of_5_optionC_is_multiple_of_5_optionD_is_multiple_of_5_optionE_is_not_multiple_of_5_l268_268419

-- Definitions of the options
def optionA : ℕ := 2019^2 - 2014^2
def optionB : ℕ := 2019^2 * 10^2
def optionC : ℕ := 2020^2 / 101^2
def optionD : ℕ := 2010^2 - 2005^2
def optionE : ℕ := 2015^2 / 5^2

-- Statements to be proven
theorem optionA_is_multiple_of_5 : optionA % 5 = 0 := by sorry
theorem optionB_is_multiple_of_5 : optionB % 5 = 0 := by sorry
theorem optionC_is_multiple_of_5 : optionC % 5 = 0 := by sorry
theorem optionD_is_multiple_of_5 : optionD % 5 = 0 := by sorry
theorem optionE_is_not_multiple_of_5 : optionE % 5 ≠ 0 := by sorry

end optionA_is_multiple_of_5_optionB_is_multiple_of_5_optionC_is_multiple_of_5_optionD_is_multiple_of_5_optionE_is_not_multiple_of_5_l268_268419


namespace problem_1_problem_2_l268_268924

variable (a b c : ℝ)

theorem problem_1 
  (hp : 0 < a) 
  (hq : 0 < b) 
  (hr : 0 < c) 
  (hs : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  abc ≤ 1/9 := 
sorry

theorem problem_2 
  (hp : 0 < a) 
  (hq : 0 < b) 
  (hr : 0 < c) 
  (hs : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (abc)) := 
sorry

end problem_1_problem_2_l268_268924


namespace problem1_problem2_l268_268841

variable {a b c : ℝ}

-- Conditions: a, b, c are positive numbers and a^(3/2) + b^(3/2) + c^(3/2) = 1
axiom pos_a : 0 < a
axiom pos_b : 0 < b
axiom pos_c : 0 < c
axiom sum_pow : a^(3/2) + b^(3/2) + c^(3/2) = 1

-- Problem (1): Prove abc ≤ 1/9 given the conditions
theorem problem1 : abc ≤ 1 / 9 := by
  sorry

-- Problem (2): Prove (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ (1 / (2 * (abc) ^ (1/2))) given the conditions
theorem problem2 : (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ 1 / (2 * (abc) ^ (1/2)) := by
  sorry

end problem1_problem2_l268_268841


namespace probability_multiple_of_3_or_4_l268_268568

theorem probability_multiple_of_3_or_4 :
  let numbers := {n | 1 ≤ n ∧ n ≤ 30},
      multiples_of_3 := {n | n ∈ numbers ∧ n % 3 = 0},
      multiples_of_4 := {n | n ∈ numbers ∧ n % 4 = 0},
      multiples_of_12 := {n | n ∈ numbers ∧ n % 12 = 0},
      favorable_outcomes := multiples_of_3 ∪ multiples_of_4,
      double_counted_outcomes := multiples_of_12,
      total_favorable_outcomes := set.card favorable_outcomes - set.card double_counted_outcomes,
      total_outcomes := set.card numbers in
  total_favorable_outcomes / total_outcomes = 1 / 2 := by
  sorry

end probability_multiple_of_3_or_4_l268_268568


namespace proof_part1_proof_part2_l268_268909

variable {a b c : ℝ}

def pos_and_sum_of_powers (a b c : ℝ) : Prop :=
  0 < a ∧ 0 < b ∧ 0 < c ∧ a^(3/2) + b^(3/2) + c^(3/2) = 1

theorem proof_part1 (h : pos_and_sum_of_powers a b c) : abc ≤ 1 / 9 :=
sorry

theorem proof_part2 (h : pos_and_sum_of_powers a b c) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (abc)) :=
sorry

end proof_part1_proof_part2_l268_268909


namespace BC_length_47_l268_268317

theorem BC_length_47 (A B C D : ℝ) (h₁ : A ≠ B) (h₂ : B ≠ C) (h₃ : B ≠ D)
  (h₄ : dist A C = 20) (h₅ : dist A D = 45) (h₆ : dist B D = 13)
  (h₇ : C = 0) (h₈ : D = 0) (h₉ : B = A + 43) :
  dist B C = 47 :=
sorry

end BC_length_47_l268_268317


namespace a_gives_b_head_start_l268_268602

theorem a_gives_b_head_start (Va Vb L H : ℝ) 
    (h1 : Va = (20 / 19) * Vb)
    (h2 : L / Va = (L - H) / Vb) : 
    H = (1 / 20) * L := sorry

end a_gives_b_head_start_l268_268602


namespace probability_case_7_probability_case_n_l268_268701

-- Define the scenario for n = 7
def probability_empty_chair_7 : Prop :=
  ∃ p : ℚ, p = 1 / 35

-- Theorem for the case when there are exactly 7 chairs
theorem probability_case_7 : probability_empty_chair_7 :=
begin
  -- Proof skipped
  sorry
end

-- Define the general scenario for n chairs (n >= 6)
def probability_empty_chair_n (n : ℕ) (h : n ≥ 6) : Prop :=
  ∃ p : ℚ, p = (n - 4) * (n - 5) / ((n - 1) * (n - 2))

-- Theorem for the case when there are n chairs (n ≥ 6)
theorem probability_case_n (n : ℕ) (h : n ≥ 6) : probability_empty_chair_n n h :=
begin
  -- Proof skipped
  sorry
end

end probability_case_7_probability_case_n_l268_268701


namespace fewest_students_possible_l268_268437

theorem fewest_students_possible (N : ℕ) :
  (N % 5 = 2) ∧ (N % 6 = 3) ∧ (N % 8 = 4) ↔ N = 59 :=
by
  sorry

end fewest_students_possible_l268_268437


namespace marbles_problem_l268_268297

theorem marbles_problem
  (cindy_original : ℕ)
  (lisa_original : ℕ)
  (h1 : cindy_original = 20)
  (h2 : cindy_original = lisa_original + 5)
  (marbles_given : ℕ)
  (h3 : marbles_given = 12) :
  (lisa_original + marbles_given) - (cindy_original - marbles_given) = 19 :=
by
  sorry

end marbles_problem_l268_268297


namespace income_percent_greater_l268_268167

variable (A B : ℝ)

-- Condition: A's income is 25% less than B's income
def income_condition (A B : ℝ) : Prop :=
  A = 0.75 * B

-- Statement: B's income is 33.33% greater than A's income
theorem income_percent_greater (A B : ℝ) (h : income_condition A B) :
  B = A * (4 / 3) := by
sorry

end income_percent_greater_l268_268167


namespace least_positive_multiple_24_gt_450_l268_268257

theorem least_positive_multiple_24_gt_450 : ∃ n : ℕ, n > 450 ∧ n % 24 = 0 ∧ n = 456 :=
by
  use 456
  sorry

end least_positive_multiple_24_gt_450_l268_268257


namespace find_constant_k_l268_268739

theorem find_constant_k 
  (k : ℝ)
  (h : ∀ x : ℝ, -x^2 - (k + 10)*x - 8 = -(x - 2)*(x - 4)) : 
  k = -16 :=
sorry

end find_constant_k_l268_268739


namespace joan_writing_time_l268_268187

theorem joan_writing_time
  (total_time : ℕ)
  (time_piano : ℕ)
  (time_reading : ℕ)
  (time_exerciser : ℕ)
  (h1 : total_time = 120)
  (h2 : time_piano = 30)
  (h3 : time_reading = 38)
  (h4 : time_exerciser = 27) : 
  total_time - (time_piano + time_reading + time_exerciser) = 25 :=
by
  sorry

end joan_writing_time_l268_268187


namespace strawberries_eaten_l268_268281

-- Definitions based on the conditions
def strawberries_picked : ℕ := 35
def strawberries_remaining : ℕ := 33

-- Statement of the proof problem
theorem strawberries_eaten :
  strawberries_picked - strawberries_remaining = 2 :=
by
  sorry

end strawberries_eaten_l268_268281


namespace scientific_notation_240000_l268_268064

theorem scientific_notation_240000 :
  240000 = 2.4 * 10^5 :=
by
  sorry

end scientific_notation_240000_l268_268064


namespace flour_baking_soda_ratio_l268_268688

theorem flour_baking_soda_ratio 
  (sugar flour baking_soda : ℕ)
  (h1 : sugar = 2000)
  (h2 : 5 * flour = 6 * sugar)
  (h3 : 8 * (baking_soda + 60) = flour) :
  flour / baking_soda = 10 := by
  sorry

end flour_baking_soda_ratio_l268_268688


namespace part1_part2_l268_268967

variable (a b c : ℝ)

-- Condition: a, b, c are positive numbers and a^(3/2) + b^(3/2) + c^(3/2) = 1
axiom pos_a : a > 0
axiom pos_b : b > 0
axiom pos_c : c > 0
axiom sum_condition : a^(3/2) + b^(3/2) + c^(3/2) = 1

-- Proof Problem (1): abc ≤ 1/9
theorem part1 : abc ≤ 1/9 := 
by {
  have h := sorry,
  exact h
}

-- Proof Problem (2): a/(b+c) + b/(a+c) + c/(a+b) ≤ 1/(2sqrt(abc))
theorem part2 : a/(b+c) + b/(a+c) + c/(a+b) ≤ 1/(2*sqrt(abc)) := 
by {
  have h := sorry,
  exact h
}

end part1_part2_l268_268967


namespace quadratic_factorization_value_of_a_l268_268515

theorem quadratic_factorization_value_of_a (a : ℝ) :
  (∀ x : ℝ, 2 * x^2 - 8 * x + a = 0 ↔ 2 * (x - 2)^2 = 4) → a = 4 :=
by
  intro h
  sorry

end quadratic_factorization_value_of_a_l268_268515


namespace union_of_sets_l268_268054

def M := {x : ℝ | -1 < x ∧ x < 1}
def N := {x : ℝ | x^2 - 3 * x ≤ 0}

theorem union_of_sets : M ∪ N = {x : ℝ | -1 < x ∧ x ≤ 3} :=
by sorry

end union_of_sets_l268_268054


namespace number_of_pickers_is_221_l268_268103
-- Import necessary Lean and math libraries

/--
Given the conditions:
1. The number of pickers fills 100 drums of raspberries per day.
2. The number of pickers fills 221 drums of grapes per day.
3. In 77 days, the pickers would fill 17017 drums of grapes.
Prove that the number of pickers is 221.
-/
theorem number_of_pickers_is_221
  (P : ℕ)
  (d1 : P * 100 = 100 * P)
  (d2 : P * 221 = 221 * P)
  (d17 : P * 221 * 77 = 17017) : 
  P = 221 := 
sorry

end number_of_pickers_is_221_l268_268103


namespace four_cubic_feet_to_cubic_inches_l268_268134

theorem four_cubic_feet_to_cubic_inches (h : 1 = 12) : 4 * (12^3) = 6912 :=
by
  sorry

end four_cubic_feet_to_cubic_inches_l268_268134


namespace cos_of_double_angles_l268_268485

theorem cos_of_double_angles (α β : ℝ) 
  (h1 : Real.sin (α - β) = 1 / 3) 
  (h2 : Real.cos α * Real.sin β = 1 / 6) : 
  Real.cos (2 * α + 2 * β) = 1 / 9 :=
by
  sorry

end cos_of_double_angles_l268_268485


namespace max_value_of_seq_l268_268665

theorem max_value_of_seq (S : ℕ → ℤ) (a : ℕ → ℤ) (hS : ∀ n, S n = -n^2 + 6 * n + 7)
  (h_a_def : ∀ n, a n = S n - S (n - 1)) : ∃ max_val, max_val = 12 ∧ ∀ n, a n ≤ max_val :=
by
  sorry

end max_value_of_seq_l268_268665


namespace problem_1_problem_2_l268_268871

theorem problem_1 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) 
  (h_sum : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  abc ≤ 1 / 9 := 
sorry

theorem problem_2 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) 
  (h_sum : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * real.sqrt (abc)) := 
sorry

end problem_1_problem_2_l268_268871


namespace chessboard_accessible_squares_l268_268748

def is_accessible (board_size : ℕ) (central_exclusion_count : ℕ) (total_squares central_inaccessible : ℕ) : Prop :=
  total_squares = board_size * board_size ∧
  central_inaccessible = central_exclusion_count + 1 + 14 + 14 ∧
  board_size = 15 ∧
  total_squares - central_inaccessible = 196

theorem chessboard_accessible_squares :
  is_accessible 15 29 225 29 :=
by {
  sorry
}

end chessboard_accessible_squares_l268_268748


namespace proof_part1_proof_part2_l268_268852

theorem proof_part1 (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) 
                    (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
                    (a * b * c ≤ 1 / 9) := 
by
  sorry

theorem proof_part2 (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) 
                    (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
                    (a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * (a * b * c)^(1/2))) := 
by
  sorry

end proof_part1_proof_part2_l268_268852


namespace fraction_doubled_unchanged_l268_268172

theorem fraction_doubled_unchanged (x y : ℝ) (h : x ≠ y) : 
  (2 * x) / (2 * x - 2 * y) = x / (x - y) :=
by
  sorry

end fraction_doubled_unchanged_l268_268172


namespace proof_part1_proof_part2_l268_268912

variable {a b c : ℝ}

def pos_and_sum_of_powers (a b c : ℝ) : Prop :=
  0 < a ∧ 0 < b ∧ 0 < c ∧ a^(3/2) + b^(3/2) + c^(3/2) = 1

theorem proof_part1 (h : pos_and_sum_of_powers a b c) : abc ≤ 1 / 9 :=
sorry

theorem proof_part2 (h : pos_and_sum_of_powers a b c) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (abc)) :=
sorry

end proof_part1_proof_part2_l268_268912


namespace problem_1_problem_2_l268_268823

variable (a b c : ℝ)

theorem problem_1 
  (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  a * b * c ≤ 1 / 9 :=
sorry

theorem problem_2 
  (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * (a * b * c)^(1/2)) :=
sorry

end problem_1_problem_2_l268_268823


namespace positive_integer_triplet_sum_l268_268409

noncomputable def exists_unique_pos_int_triplet : ∃ a b c : ℕ, 
  0 < a ∧ 0 < b ∧ 0 < c ∧ a ≤ b ∧ b ≤ c ∧ (25 / 84 = 1 / a + 1 / (a * b) + 1 / (a * b * c)) :=
sorry

theorem positive_integer_triplet_sum : (a b c : ℕ) (h1: 0 < a) (h2: 0 < b) (h3: 0 < c) 
  (h4: a ≤ b) (h5: b ≤ c) (h6: 25 / 84 = 1 / a + 1 / (a * b) + 1 / (a * b * c)) :
  a + b + c = 17 :=
sorry

end positive_integer_triplet_sum_l268_268409


namespace toms_remaining_speed_l268_268078

-- Defining the constants and conditions
def total_distance : ℝ := 100
def first_leg_distance : ℝ := 50
def first_leg_speed : ℝ := 20
def avg_speed : ℝ := 28.571428571428573

-- Proving Tom's speed during the remaining part of the trip
theorem toms_remaining_speed :
  ∃ (remaining_leg_speed : ℝ),
    (remaining_leg_speed = 50) ∧
    (total_distance = first_leg_distance + 50) ∧
    ((first_leg_distance / first_leg_speed + 50 / remaining_leg_speed) = total_distance / avg_speed) :=
by
  sorry

end toms_remaining_speed_l268_268078


namespace cost_of_450_chocolates_l268_268101

theorem cost_of_450_chocolates :
  ∀ (cost_per_box : ℝ) (candies_per_box total_candies : ℕ),
  cost_per_box = 7.50 →
  candies_per_box = 30 →
  total_candies = 450 →
  (total_candies / candies_per_box : ℝ) * cost_per_box = 112.50 :=
by
  intros cost_per_box candies_per_box total_candies h1 h2 h3
  sorry

end cost_of_450_chocolates_l268_268101


namespace miles_per_book_l268_268544

theorem miles_per_book (total_miles : ℝ) (books_read : ℝ) (miles_per_book : ℝ) : 
  total_miles = 6760 ∧ books_read = 15 → miles_per_book = 450.67 := 
by
  sorry

end miles_per_book_l268_268544


namespace part1_part2_l268_268863

variable (a b c : ℝ)

theorem part1 (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) : abc ≤ 1/9 :=
by
  sorry

theorem part2 (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
    (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ 1 / (2 * sqrt(abc)) :=
by
  sorry

end part1_part2_l268_268863


namespace temperature_of_Huangshan_at_night_l268_268240

theorem temperature_of_Huangshan_at_night 
  (T_morning : ℤ) (Rise_noon : ℤ) (Drop_night : ℤ)
  (h1 : T_morning = -12) (h2 : Rise_noon = 8) (h3 : Drop_night = 10) :
  T_morning + Rise_noon - Drop_night = -14 :=
by
  sorry

end temperature_of_Huangshan_at_night_l268_268240


namespace selection_ways_l268_268749

-- Step a): Define the conditions
def number_of_boys := 26
def number_of_girls := 24

-- Step c): State the problem
theorem selection_ways :
  number_of_boys + number_of_girls = 50 := by
  sorry

end selection_ways_l268_268749


namespace bahs_from_yahs_l268_268040

theorem bahs_from_yahs (b r y : ℝ) 
  (h1 : 18 * b = 30 * r) 
  (h2 : 10 * r = 25 * y) : 
  1250 * y = 300 * b := 
by
  sorry

end bahs_from_yahs_l268_268040


namespace part1_part2_l268_268969

variable (a b c : ℝ)

-- Condition: a, b, c are positive numbers and a^(3/2) + b^(3/2) + c^(3/2) = 1
axiom pos_a : a > 0
axiom pos_b : b > 0
axiom pos_c : c > 0
axiom sum_condition : a^(3/2) + b^(3/2) + c^(3/2) = 1

-- Proof Problem (1): abc ≤ 1/9
theorem part1 : abc ≤ 1/9 := 
by {
  have h := sorry,
  exact h
}

-- Proof Problem (2): a/(b+c) + b/(a+c) + c/(a+b) ≤ 1/(2sqrt(abc))
theorem part2 : a/(b+c) + b/(a+c) + c/(a+b) ≤ 1/(2*sqrt(abc)) := 
by {
  have h := sorry,
  exact h
}

end part1_part2_l268_268969


namespace abc_le_one_ninth_sum_ratios_le_one_over_two_sqrt_abc_l268_268820

open Real

variables (a b c : ℝ)

-- Condition: a, b, c are positive numbers
variables (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
-- Condition: a^(3/2) + b^(3/2) + c^(3/2) = 1
variable  (h_sum : a^(3/2) + b^(3/2) + c^(3/2) = 1)

-- Question 1: Prove abc ≤ 1/9
theorem abc_le_one_ninth : a * b * c ≤ 1 / 9 :=
  sorry

-- Question 2: Prove (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ 1 / (2 * sqrt (a * b * c))
theorem sum_ratios_le_one_over_two_sqrt_abc : 
  (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ 1 / (2 * sqrt (a * b * c)) :=
  sorry

end abc_le_one_ninth_sum_ratios_le_one_over_two_sqrt_abc_l268_268820


namespace sum_of_xyz_l268_268214

theorem sum_of_xyz (x y z : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h : (x + y + z)^3 - x^3 - y^3 - z^3 = 504) : x + y + z = 9 :=
by {
  sorry
}

end sum_of_xyz_l268_268214


namespace GregPPO_reward_correct_l268_268034

-- Define the maximum ProcGen reward
def maxProcGenReward : ℕ := 240

-- Define the maximum CoinRun reward in the more challenging version
def maxCoinRunReward : ℕ := maxProcGenReward / 2

-- Define the percentage reward obtained by Greg's PPO algorithm
def percentageRewardObtained : ℝ := 0.9

-- Calculate the reward obtained by Greg's PPO algorithm
def rewardGregPPO : ℝ := percentageRewardObtained * maxCoinRunReward

-- The theorem to prove the correct answer
theorem GregPPO_reward_correct : rewardGregPPO = 108 := by
  sorry

end GregPPO_reward_correct_l268_268034


namespace probability_case_7_probability_case_n_l268_268700

-- Define the scenario for n = 7
def probability_empty_chair_7 : Prop :=
  ∃ p : ℚ, p = 1 / 35

-- Theorem for the case when there are exactly 7 chairs
theorem probability_case_7 : probability_empty_chair_7 :=
begin
  -- Proof skipped
  sorry
end

-- Define the general scenario for n chairs (n >= 6)
def probability_empty_chair_n (n : ℕ) (h : n ≥ 6) : Prop :=
  ∃ p : ℚ, p = (n - 4) * (n - 5) / ((n - 1) * (n - 2))

-- Theorem for the case when there are n chairs (n ≥ 6)
theorem probability_case_n (n : ℕ) (h : n ≥ 6) : probability_empty_chair_n n h :=
begin
  -- Proof skipped
  sorry
end

end probability_case_7_probability_case_n_l268_268700


namespace problem_f_2008_mod_100_l268_268057

theorem problem_f_2008_mod_100 : 
    let f : ℕ → ℕ := λ n, if n = 1 then 1 
                        else if n = 2 then 1 
                        else f (n - 1) + f (n - 2)
    in (f 2008) % 100 = 71 :=
by
  let f := λ n, if n = 1 then 1 else if n = 2 then 1 else ((Nat.fib (n - 1)) + (Nat.fib (n - 2)))
  have h : f(1) = 1 := rfl
  have h2 : f(2) = 1 := rfl
  -- Further proof steps involve using the properties and precomputed result
  -- We assume the provided solution as fact for the Lean statement
  sorry

end problem_f_2008_mod_100_l268_268057


namespace find_y_l268_268410

theorem find_y (y z : ℕ) (h1 : 50 = y * 10) (h2 : 300 = 50 * z) : y = 5 :=
by
  sorry

end find_y_l268_268410


namespace problem_1_problem_2_l268_268919

variable (a b c : ℝ)

theorem problem_1 
  (hp : 0 < a) 
  (hq : 0 < b) 
  (hr : 0 < c) 
  (hs : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  abc ≤ 1/9 := 
sorry

theorem problem_2 
  (hp : 0 < a) 
  (hq : 0 < b) 
  (hr : 0 < c) 
  (hs : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (abc)) := 
sorry

end problem_1_problem_2_l268_268919


namespace correct_answer_l268_268709

def P : Set ℝ := {1, 2, 3}
def Q : Set ℝ := {x | 2 ≤ x ∧ x ≤ 3}

theorem correct_answer : P ∩ Q ⊆ P := by
  sorry

end correct_answer_l268_268709


namespace problem_1_problem_2_l268_268934

variable {a b c : ℝ}

theorem problem_1 (hp : 0 < a ∧ 0 < b ∧ 0 < c)
    (h1 : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
    abc ≤ 1/9 := 
by 
  apply sorry

theorem problem_2 (hp : 0 < a ∧ 0 < b ∧ 0 < c)
    (h1 : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
    a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * real.sqrt (abc)) := 
by 
  apply sorry

end problem_1_problem_2_l268_268934


namespace find_f_20_l268_268350

noncomputable def f : ℝ → ℝ := sorry

theorem find_f_20 :
  (∀ x : ℝ, f x = f (-x)) →
  (∀ x : ℝ, f x = f (2 - x)) →
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → f x = x - 1 / 2) →
  f 20 = - 1 / 2 :=
sorry

end find_f_20_l268_268350


namespace shape_area_l268_268391

theorem shape_area (f : ℝ → ℝ) (h : ∀ x, f x = Real.exp x - 1) : 
  (∫ x in 0..1, f x) = Real.exp 1 - 2 :=
by
  rw [←h]
  rw [Real.integral_exp_sub_1]
  rw [Real.exp_one]
  Sorry

end shape_area_l268_268391


namespace net_change_salary_l268_268448

/-- Given an initial salary S and a series of percentage changes:
    20% increase, 10% decrease, 15% increase, and 5% decrease,
    prove that the net change in salary is 17.99%. -/
theorem net_change_salary (S : ℝ) :
  (1.20 * 0.90 * 1.15 * 0.95 - 1) * S = 0.1799 * S :=
sorry

end net_change_salary_l268_268448


namespace problem_1_problem_2_l268_268920

variable (a b c : ℝ)

theorem problem_1 
  (hp : 0 < a) 
  (hq : 0 < b) 
  (hr : 0 < c) 
  (hs : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  abc ≤ 1/9 := 
sorry

theorem problem_2 
  (hp : 0 < a) 
  (hq : 0 < b) 
  (hr : 0 < c) 
  (hs : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (abc)) := 
sorry

end problem_1_problem_2_l268_268920


namespace part1_part2_l268_268889

variables {a b c : ℝ}

theorem part1 (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h : a^(3/2) + b^(3/2) + c^(3/2) = 1) : abc ≤ 1/9 := 
sorry

theorem part2 (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (abc)) :=
sorry

end part1_part2_l268_268889


namespace angle_inclination_of_l_range_l268_268159

noncomputable def angle_of_inclination_range (M A B : ℝ × ℝ) : Set ℝ := sorry

theorem angle_inclination_of_l_range :
  ∀ (M A B : ℝ × ℝ),
    M = (1, 0) →
    A = (2, 1) →
    B = (0, Real.sqrt 3) →
    angle_of_inclination_range M A B = Set.Icc (Real.pi / 4) (2 * Real.pi / 3) :=
by
  intros
  sorry

end angle_inclination_of_l_range_l268_268159


namespace cos_double_angle_l268_268491

theorem cos_double_angle 
  (α β : ℝ) 
  (h1 : Real.sin (α - β) = 1/3) 
  (h2 : Real.cos α * Real.sin β = 1/6) :
  Real.cos (2 * α + 2 * β) = 1/9 :=
sorry

end cos_double_angle_l268_268491


namespace cylinder_height_inscribed_in_hemisphere_l268_268764

theorem cylinder_height_inscribed_in_hemisphere (r_cylinder r_hemisphere : ℝ) (h_r_cylinder : r_cylinder = 3) (h_r_hemisphere : r_hemisphere = 7) :
  ∃ h : ℝ, h = sqrt (r_hemisphere^2 - r_cylinder^2) ∧ h = sqrt 40 :=
by
  use sqrt (r_hemisphere^2 - r_cylinder^2)
  have h_eq : sqrt (7^2 - 3^2) = sqrt 40, by sorry
  split
  { rw [h_r_hemisphere, h_r_cylinder] }
  { exact h_eq }

end cylinder_height_inscribed_in_hemisphere_l268_268764


namespace final_spent_l268_268773

-- Define all the costs.
def albertoExpenses : ℤ := 2457 + 374 + 520 + 129 + 799
def albertoDiscountExhaust : ℤ := (799 * 5) / 100
def albertoTotalBeforeLoyaltyDiscount : ℤ := albertoExpenses - albertoDiscountExhaust
def albertoLoyaltyDiscount : ℤ := (albertoTotalBeforeLoyaltyDiscount * 7) / 100
def albertoFinal : ℤ := albertoTotalBeforeLoyaltyDiscount - albertoLoyaltyDiscount

def samaraExpenses : ℤ := 25 + 467 + 79 + 175 + 599 + 225
def samaraSalesTax : ℤ := (samaraExpenses * 6) / 100
def samaraFinal : ℤ := samaraExpenses + samaraSalesTax

def difference : ℤ := albertoFinal - samaraFinal

theorem final_spent (h : difference = 2278) : true :=
  sorry

end final_spent_l268_268773


namespace max_min_xy_l268_268402

theorem max_min_xy (x y a : ℝ) (h1 : x + y = a) (h2 : x^2 + y^2 = -a^2 + 2) : (x * y) ∈ { real.min (-1), real.max (1) } :=
by
  have h : a^2 ≤ 4 := 
  sorry
  have xy_eq : x * y = (a^2 - 2) / 2 := 
  sorry
  split
  { calc min (xy_eq) = -1 :=
    sorry
    calc max (xy_eq) = 1 :=
    sorry
  }

end max_min_xy_l268_268402


namespace lisa_more_marbles_than_cindy_l268_268296

-- Definitions
def initial_cindy_marbles : ℕ := 20
def difference_cindy_lisa : ℕ := 5
def cindy_gives_lisa : ℕ := 12

-- Assuming Cindy's initial marbles are 20, which are 5 more than Lisa's marbles, and Cindy gives Lisa 12 marbles,
-- prove that Lisa now has 19 more marbles than Cindy.
theorem lisa_more_marbles_than_cindy :
  let lisa_initial_marbles := initial_cindy_marbles - difference_cindy_lisa,
      lisa_current_marbles := lisa_initial_marbles + cindy_gives_lisa,
      cindy_current_marbles := initial_cindy_marbles - cindy_gives_lisa
  in lisa_current_marbles - cindy_current_marbles = 19 :=
by {
  sorry
}

end lisa_more_marbles_than_cindy_l268_268296


namespace total_selling_price_correct_l268_268613

-- Definitions of initial purchase prices in different currencies
def init_price_eur : ℕ := 600
def init_price_gbp : ℕ := 450
def init_price_usd : ℕ := 750

-- Definitions of initial exchange rates
def init_exchange_rate_eur_to_usd : ℝ := 1.1
def init_exchange_rate_gbp_to_usd : ℝ := 1.3

-- Definitions of profit percentages for each article
def profit_percent_eur : ℝ := 0.08
def profit_percent_gbp : ℝ := 0.1
def profit_percent_usd : ℝ := 0.15

-- Definitions of new exchange rates at the time of selling
def new_exchange_rate_eur_to_usd : ℝ := 1.15
def new_exchange_rate_gbp_to_usd : ℝ := 1.25

-- Calculation of purchase prices in USD
def purchase_price_in_usd₁ : ℝ := init_price_eur * init_exchange_rate_eur_to_usd
def purchase_price_in_usd₂ : ℝ := init_price_gbp * init_exchange_rate_gbp_to_usd
def purchase_price_in_usd₃ : ℝ := init_price_usd

-- Calculation of selling prices including profit in USD
def selling_price_in_usd₁ : ℝ := (init_price_eur + (init_price_eur * profit_percent_eur)) * new_exchange_rate_eur_to_usd
def selling_price_in_usd₂ : ℝ := (init_price_gbp + (init_price_gbp * profit_percent_gbp)) * new_exchange_rate_gbp_to_usd
def selling_price_in_usd₃ : ℝ := init_price_usd * (1 + profit_percent_usd)

-- Total selling price in USD
def total_selling_price_in_usd : ℝ :=
  selling_price_in_usd₁ + selling_price_in_usd₂ + selling_price_in_usd₃

-- Proof goal: total selling price should equal 2225.85 USD
theorem total_selling_price_correct :
  total_selling_price_in_usd = 2225.85 :=
by
  sorry

end total_selling_price_correct_l268_268613


namespace proof_part1_proof_part2_l268_268854

theorem proof_part1 (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) 
                    (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
                    (a * b * c ≤ 1 / 9) := 
by
  sorry

theorem proof_part2 (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) 
                    (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
                    (a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * (a * b * c)^(1/2))) := 
by
  sorry

end proof_part1_proof_part2_l268_268854


namespace infinite_solutions_iff_l268_268357

theorem infinite_solutions_iff (a b c d : ℤ) :
  (∃ᶠ x in at_top, ∃ᶠ y in at_top, x^2 + a * x + b = y^2 + c * y + d) ↔ (a^2 - 4 * b = c^2 - 4 * d) :=
by sorry

end infinite_solutions_iff_l268_268357


namespace problem_1_problem_2_l268_268878

theorem problem_1 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) 
  (h_sum : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  abc ≤ 1 / 9 := 
sorry

theorem problem_2 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) 
  (h_sum : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * real.sqrt (abc)) := 
sorry

end problem_1_problem_2_l268_268878


namespace evaluate_expression_l268_268597

theorem evaluate_expression :
  let a := 24
  let b := 7
  3 * (a + b) ^ 2 - (a ^ 2 + b ^ 2) = 2258 :=
by
  let a := 24
  let b := 7
  sorry

end evaluate_expression_l268_268597


namespace talent_show_girls_count_l268_268073

theorem talent_show_girls_count (B G : ℕ) (h1 : B + G = 34) (h2 : G = B + 22) : G = 28 :=
by
  sorry

end talent_show_girls_count_l268_268073


namespace find_number_l268_268274

-- Define the condition: a number exceeds by 40 from its 3/8 part.
def exceeds_by_40_from_its_fraction (x : ℝ) := x = (3/8) * x + 40

-- The theorem: prove that the number is 64 given the condition.
theorem find_number (x : ℝ) (h : exceeds_by_40_from_its_fraction x) : x = 64 := 
by
  sorry

end find_number_l268_268274


namespace problem_1_problem_2_l268_268882

theorem problem_1 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) 
  (h_sum : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  abc ≤ 1 / 9 := 
sorry

theorem problem_2 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) 
  (h_sum : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * real.sqrt (abc)) := 
sorry

end problem_1_problem_2_l268_268882


namespace find_fourth_number_in_proportion_l268_268674

-- Define the given conditions
def x : ℝ := 0.39999999999999997
def proportion (y : ℝ) := 0.60 / x = 6 / y

-- State the theorem to be proven
theorem find_fourth_number_in_proportion :
  proportion y → y = 4 :=
by
  intro h
  sorry

end find_fourth_number_in_proportion_l268_268674


namespace problem_1_problem_2_l268_268928

variable (a b c : ℝ)

theorem problem_1 
  (hp : 0 < a) 
  (hq : 0 < b) 
  (hr : 0 < c) 
  (hs : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  abc ≤ 1/9 := 
sorry

theorem problem_2 
  (hp : 0 < a) 
  (hq : 0 < b) 
  (hr : 0 < c) 
  (hs : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (abc)) := 
sorry

end problem_1_problem_2_l268_268928


namespace problem_I_problem_II_l268_268095

def f (x : ℝ) : ℝ := abs (x + 2) + abs (x - 3)

theorem problem_I (x : ℝ) : (f x > 7 - x) ↔ (x < -6 ∨ x > 2) := 
by 
  sorry

theorem problem_II (m : ℝ) : (∃ x : ℝ, f x ≤ abs (3 * m - 2)) ↔ (m ≤ -1 ∨ m ≥ 7 / 3) := 
by 
  sorry

end problem_I_problem_II_l268_268095


namespace expand_product_l268_268009

def poly1 (x : ℝ) := 4 * x + 2
def poly2 (x : ℝ) := 3 * x - 1
def poly3 (x : ℝ) := x + 6

theorem expand_product (x : ℝ) :
  (poly1 x) * (poly2 x) * (poly3 x) = 12 * x^3 + 74 * x^2 + 10 * x - 12 :=
by
  sorry

end expand_product_l268_268009


namespace part1_part2_l268_268996

-- a), b), c are positive real numbers and ${a}^{\frac{3}{2}}+{b}^{\frac{3}{2}}+{c}^{\frac{3}{2}}=1$
variables (a b c : ℝ)
axiom a_pos : 0 < a
axiom b_pos : 0 < b
axiom c_pos : 0 < c
axiom sum_eq_one : a^(3/2) + b^(3/2) + c^(3/2) = 1

-- Prove (1) and (2)
theorem part1 : abc ≤ 1 / 9 := sorry

theorem part2 : a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (abc)) := sorry

end part1_part2_l268_268996


namespace parabola_vertex_x_coord_l268_268560

theorem parabola_vertex_x_coord (a b c : ℝ)
  (h1 : 5 = a * 2^2 + b * 2 + c)
  (h2 : 5 = a * 8^2 + b * 8 + c)
  (h3 : 11 = a * 9^2 + b * 9 + c) :
  5 = (2 + 8) / 2 := 
sorry

end parabola_vertex_x_coord_l268_268560


namespace stratified_sampling_l268_268108

-- Definitions of the classes and their student counts
def class1_students : Nat := 54
def class2_students : Nat := 42

-- Definition of total students to be sampled
def total_sampled_students : Nat := 16

-- Definition of the number of students to be selected from each class
def students_selected_from_class1 : Nat := 9
def students_selected_from_class2 : Nat := 7

-- The proof problem
theorem stratified_sampling :
  students_selected_from_class1 + students_selected_from_class2 = total_sampled_students ∧ 
  students_selected_from_class1 * (class2_students + class1_students) = class1_students * total_sampled_students :=
by
  sorry

end stratified_sampling_l268_268108


namespace num_two_digit_palindromes_l268_268511

theorem num_two_digit_palindromes : 
  let is_palindrome (n : ℕ) : Prop := (n / 10) = (n % 10)
  ∃ n : ℕ, 10 ≤ n ∧ n < 90 ∧ is_palindrome n →
  ∃ count : ℕ, count = 9 := 
sorry

end num_two_digit_palindromes_l268_268511


namespace percentage_difference_is_50_percent_l268_268542

-- Definitions of hourly wages
def Mike_hourly_wage : ℕ := 14
def Phil_hourly_wage : ℕ := 7

-- Calculating the percentage difference
theorem percentage_difference_is_50_percent :
  (Mike_hourly_wage - Phil_hourly_wage) * 100 / Mike_hourly_wage = 50 :=
by
  sorry

end percentage_difference_is_50_percent_l268_268542


namespace problem_part1_problem_part2_l268_268944

variable {a b c : ℝ}

theorem problem_part1 (h : a > 0 ∧ b > 0 ∧ c > 0) (sum_eq : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  abc ≤ 1 / 9 :=
sorry

theorem problem_part2 (h : a > 0 ∧ b > 0 ∧ c > 0) (sum_eq : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (abc)) :=
sorry

end problem_part1_problem_part2_l268_268944


namespace fraction_spoiled_l268_268387

-- Create the variables for initial conditions
variables (initial_stock sold_stock new_stock total_stock : ℕ)

-- Define the conditions
def steve_conditions : Prop :=
  initial_stock = 200 ∧
  sold_stock = 50 ∧
  new_stock = 200 ∧
  total_stock = 300

-- Define the problem statement
theorem fraction_spoiled (h : steve_conditions initial_stock sold_stock new_stock total_stock) :
  let remaining_before_spoil := initial_stock - sold_stock in
  let spoiled := total_stock - new_stock in
  (spoiled : ℚ) / remaining_before_spoil = 2 / 3 :=
by
  sorry

end fraction_spoiled_l268_268387


namespace sum_of_three_consecutive_natural_numbers_not_prime_l268_268625

theorem sum_of_three_consecutive_natural_numbers_not_prime (n : ℕ) : 
  ¬ Prime (n + (n+1) + (n+2)) := by
  sorry

end sum_of_three_consecutive_natural_numbers_not_prime_l268_268625


namespace pencil_cost_l268_268112

theorem pencil_cost (P : ℝ) (h1 : 24 * P + 18 = 30) : P = 0.5 :=
by
  sorry

end pencil_cost_l268_268112


namespace desired_lines_l268_268316

-- Defining the given entities
variables {Point : Type}
variables [MetricSpace Point]

variable (A O B : Point)
variable (l : AffineSubspace ℝ Point) -- Line l

-- Condition: Given an angle ∠AOB
def angleAOB (A O B: Point) : ℕ := sorry  -- A placeholder for the actual angle calculation

-- Desired properties of line l₁ such that the angle between l and l₁ equals ∠AOB
def isCandidateLine (l  l₁: AffineSubspace ℝ Point) (angleAOB: ℕ) : Prop := 
  sorry  -- A placeholder for the characterization of angle between lines

-- Variables for candidate lines
variable (XO YO : AffineSubspace ℝ Point)

-- Statement to be Proven
theorem desired_lines (A O B: Point) (l XO YO: AffineSubspace ℝ Point):
  isCandidateLine l XO (angleAOB A O B) ∧ 
  isCandidateLine l YO (angleAOB A O B) :=
sorry

end desired_lines_l268_268316


namespace probability_seven_chairs_probability_n_chairs_l268_268696
-- Importing necessary library to ensure our Lean code can be built successfully

-- Definition for case where n = 7
theorem probability_seven_chairs : 
  let total_seating := 7 * 6 * 5 / 6 
  let favorable_seating := 1 
  let probability := favorable_seating / total_seating 
  probability = 1 / 35 := 
by 
  sorry

-- Definition for general case where n ≥ 6
theorem probability_n_chairs (n : ℕ) (h : n ≥ 6) : 
  let total_seating := (n - 1) * (n - 2) / 2 
  let favorable_seating := (n - 4) * (n - 5) / 2 
  let probability := favorable_seating / total_seating 
  probability = (n - 4) * (n - 5) / ((n - 1) * (n - 2)) := 
by 
  sorry

end probability_seven_chairs_probability_n_chairs_l268_268696


namespace max_flow_increase_proof_l268_268454

noncomputable def max_flow_increase : ℕ :=
  sorry

theorem max_flow_increase_proof
  (initial_pipes_AB: ℕ) (initial_pipes_BC: ℕ) (flow_increase_per_pipes_swap: ℕ)
  (swap_increase: initial_pipes_AB = 10)
  (swap_increase_2: initial_pipes_BC = 10)
  (flow_increment: flow_increase_per_pipes_swap = 30) : 
  max_flow_increase = 150 :=
  sorry

end max_flow_increase_proof_l268_268454


namespace height_of_cylinder_l268_268761

theorem height_of_cylinder (r_h r_c h : ℝ) (h1 : r_h = 7) (h2 : r_c = 3) :
  h = 2 * Real.sqrt 10 ↔ 
  h = Real.sqrt (r_h^2 - r_c^2) :=
by {
  rw [h1, h2],
  sorry
}

end height_of_cylinder_l268_268761


namespace celina_paid_multiple_of_diego_l268_268563

theorem celina_paid_multiple_of_diego
  (D : ℕ) (x : ℕ)
  (h_total : (x + 1) * D + 1000 = 50000)
  (h_positive : D > 0) :
  x = 48 :=
sorry

end celina_paid_multiple_of_diego_l268_268563


namespace proof_part1_proof_part2_l268_268916

variable {a b c : ℝ}

def pos_and_sum_of_powers (a b c : ℝ) : Prop :=
  0 < a ∧ 0 < b ∧ 0 < c ∧ a^(3/2) + b^(3/2) + c^(3/2) = 1

theorem proof_part1 (h : pos_and_sum_of_powers a b c) : abc ≤ 1 / 9 :=
sorry

theorem proof_part2 (h : pos_and_sum_of_powers a b c) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (abc)) :=
sorry

end proof_part1_proof_part2_l268_268916


namespace bus_capacity_fraction_l268_268270

theorem bus_capacity_fraction
  (capacity : ℕ)
  (x : ℚ)
  (return_fraction : ℚ)
  (total_people : ℕ)
  (capacity_eq : capacity = 200)
  (return_fraction_eq : return_fraction = 4/5)
  (total_people_eq : total_people = 310)
  (people_first_trip_eq : 200 * x + 200 * 4/5 = 310) :
  x = 3/4 :=
by
  sorry

end bus_capacity_fraction_l268_268270


namespace inequality_not_holds_l268_268166

variable (x y : ℝ)

theorem inequality_not_holds (h1 : x > 1) (h2 : 1 > y) : x - 1 ≤ 1 - y :=
sorry

end inequality_not_holds_l268_268166


namespace part1_part2_l268_268956

def pos_real := {x : ℝ // 0 < x}

variables (a b c : pos_real)

theorem part1 (h : a.val ^ (3/2) + b.val ^ (3/2) + c.val ^ (3/2) = 1) : a.val * b.val * c.val ≤ 1 / 9 :=
sorry

theorem part2 (h : a.val ^ (3/2) + b.val ^ (3/2) + c.val ^ (3/2) = 1) : 
  a.val / (b.val + c.val) + b.val / (a.val + c.val) + c.val / (a.val + b.val) ≤ 1 / (2 * Real.sqrt (a.val * b.val * c.val)) :=
sorry

end part1_part2_l268_268956


namespace stock_values_l268_268369

theorem stock_values (AA_invest : ℕ) (BB_invest : ℕ) (CC_invest : ℕ)
  (AA_first_year_increase : ℝ) (BB_first_year_decrease : ℝ) (CC_first_year_change : ℝ)
  (AA_second_year_decrease : ℝ) (BB_second_year_increase : ℝ) (CC_second_year_increase : ℝ)
  (A_final : ℝ) (B_final : ℝ) (C_final : ℝ) :
  AA_invest = 150 → BB_invest = 100 → CC_invest = 50 →
  AA_first_year_increase = 1.10 → BB_first_year_decrease = 0.70 → CC_first_year_change = 1 →
  AA_second_year_decrease = 0.95 → BB_second_year_increase = 1.10 → CC_second_year_increase = 1.08 →
  A_final = (AA_invest * AA_first_year_increase) * AA_second_year_decrease →
  B_final = (BB_invest * BB_first_year_decrease) * BB_second_year_increase →
  C_final = (CC_invest * CC_first_year_change) * CC_second_year_increase →
  C_final < B_final ∧ B_final < A_final :=
by
  intros
  sorry

end stock_values_l268_268369


namespace sin_double_angle_series_l268_268670

theorem sin_double_angle_series (θ : ℝ) (h : ∑' (n : ℕ), (sin θ)^(2 * n) = 3) :
  sin (2 * θ) = (2 * sqrt 2) / 3 :=
sorry

end sin_double_angle_series_l268_268670


namespace coefficients_sum_eq_zero_l268_268306

theorem coefficients_sum_eq_zero 
  (a b c : ℝ)
  (f g h : ℝ → ℝ)
  (h1 : ∀ x, f x = a * x^2 + b * x + c)
  (h2 : ∀ x, g x = b * x^2 + c * x + a)
  (h3 : ∀ x, h x = c * x^2 + a * x + b)
  (h4 : ∃ x : ℝ, f x = 0 ∧ g x = 0 ∧ h x = 0) :
  a + b + c = 0 := 
sorry

end coefficients_sum_eq_zero_l268_268306


namespace part1_part2_l268_268886

variables {a b c : ℝ}

theorem part1 (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h : a^(3/2) + b^(3/2) + c^(3/2) = 1) : abc ≤ 1/9 := 
sorry

theorem part2 (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (abc)) :=
sorry

end part1_part2_l268_268886


namespace abc_le_one_ninth_sum_ratios_le_one_over_two_sqrt_abc_l268_268819

open Real

variables (a b c : ℝ)

-- Condition: a, b, c are positive numbers
variables (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
-- Condition: a^(3/2) + b^(3/2) + c^(3/2) = 1
variable  (h_sum : a^(3/2) + b^(3/2) + c^(3/2) = 1)

-- Question 1: Prove abc ≤ 1/9
theorem abc_le_one_ninth : a * b * c ≤ 1 / 9 :=
  sorry

-- Question 2: Prove (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ 1 / (2 * sqrt (a * b * c))
theorem sum_ratios_le_one_over_two_sqrt_abc : 
  (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ 1 / (2 * sqrt (a * b * c)) :=
  sorry

end abc_le_one_ninth_sum_ratios_le_one_over_two_sqrt_abc_l268_268819


namespace purely_imaginary_complex_l268_268338

theorem purely_imaginary_complex (m : ℝ) :
  (m^2 - 3 * m = 0) → (m^2 - 5 * m + 6 ≠ 0) → m = 0 :=
begin
  intros h_real h_imag,
  -- The proof will go here
  sorry
end

end purely_imaginary_complex_l268_268338


namespace problem_1_problem_2_l268_268921

variable (a b c : ℝ)

theorem problem_1 
  (hp : 0 < a) 
  (hq : 0 < b) 
  (hr : 0 < c) 
  (hs : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  abc ≤ 1/9 := 
sorry

theorem problem_2 
  (hp : 0 < a) 
  (hq : 0 < b) 
  (hr : 0 < c) 
  (hs : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (abc)) := 
sorry

end problem_1_problem_2_l268_268921


namespace inscribed_cylinder_height_l268_268760

theorem inscribed_cylinder_height (R: ℝ) (r: ℝ) (h: ℝ) : 
  (R = 7) → (r = 3) → (h = 2 * real.sqrt 10) :=
by 
  intro R_eq r_eq
  have : R = 7 := R_eq
  have : r = 3 := r_eq
  -- Height calculation and proof will go here, actually skipped due to 'sorry'
  sorry

end inscribed_cylinder_height_l268_268760


namespace problem_1_problem_2_l268_268931

variable {a b c : ℝ}

theorem problem_1 (hp : 0 < a ∧ 0 < b ∧ 0 < c)
    (h1 : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
    abc ≤ 1/9 := 
by 
  apply sorry

theorem problem_2 (hp : 0 < a ∧ 0 < b ∧ 0 < c)
    (h1 : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
    a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * real.sqrt (abc)) := 
by 
  apply sorry

end problem_1_problem_2_l268_268931


namespace total_weight_correct_l268_268308

-- Conditions for the weights of different types of candies
def frank_chocolate_weight : ℝ := 3
def gwen_chocolate_weight : ℝ := 2
def frank_gummy_bears_weight : ℝ := 2
def gwen_gummy_bears_weight : ℝ := 2.5
def frank_caramels_weight : ℝ := 1
def gwen_caramels_weight : ℝ := 1
def frank_hard_candy_weight : ℝ := 4
def gwen_hard_candy_weight : ℝ := 1.5

-- Combined weights of each type of candy
def chocolate_weight : ℝ := frank_chocolate_weight + gwen_chocolate_weight
def gummy_bears_weight : ℝ := frank_gummy_bears_weight + gwen_gummy_bears_weight
def caramels_weight : ℝ := frank_caramels_weight + gwen_caramels_weight
def hard_candy_weight : ℝ := frank_hard_candy_weight + gwen_hard_candy_weight

-- Total weight of the Halloween candy haul
def total_halloween_weight : ℝ := 
  chocolate_weight +
  gummy_bears_weight +
  caramels_weight +
  hard_candy_weight

-- Theorem to prove the total weight is 17 pounds
theorem total_weight_correct : total_halloween_weight = 17 := by
  sorry

end total_weight_correct_l268_268308


namespace distributive_property_l268_268459

theorem distributive_property (a b : ℝ) : 3 * a * (2 * a - b) = 6 * a^2 - 3 * a * b :=
by
  sorry

end distributive_property_l268_268459


namespace perimeter_regular_polygon_l268_268779

-- Definitions of the conditions
def side_length : ℕ := 8
def exterior_angle : ℕ := 72
def sum_of_exterior_angles : ℕ := 360

-- Number of sides calculation
def num_sides : ℕ := sum_of_exterior_angles / exterior_angle

-- Perimeter calculation
def perimeter (n : ℕ) (l : ℕ) : ℕ := n * l

-- Theorem statement
theorem perimeter_regular_polygon : perimeter num_sides side_length = 40 :=
by
  sorry

end perimeter_regular_polygon_l268_268779


namespace member_sum_of_two_others_l268_268287

def numMembers : Nat := 1978
def numCountries : Nat := 6

theorem member_sum_of_two_others :
  ∃ m : ℕ, m ∈ Finset.range numMembers.succ ∧
  ∃ a b : ℕ, a ∈ Finset.range numMembers.succ ∧ b ∈ Finset.range numMembers.succ ∧ 
  ∃ country : Fin (numCountries + 1), (a = m + b ∧ country = country) :=
by
  sorry

end member_sum_of_two_others_l268_268287


namespace total_gum_l268_268591

-- Define the conditions
def original_gum : ℕ := 38
def additional_gum : ℕ := 16

-- Define the statement to be proved
theorem total_gum : original_gum + additional_gum = 54 :=
by
  -- Proof omitted
  sorry

end total_gum_l268_268591


namespace rhombus_area_l268_268603

theorem rhombus_area (side d1 d2 : ℝ) (h_side : side = 25) (h_d1 : d1 = 30) (h_diag : d2 = 40) :
  (d1 * d2) / 2 = 600 :=
by
  rw [h_d1, h_diag]
  norm_num

end rhombus_area_l268_268603


namespace least_positive_multiple_of_24_gt_450_l268_268259

theorem least_positive_multiple_of_24_gt_450 : 
  ∃ n : ℕ, n > 450 ∧ (∃ k : ℕ, n = 24 * k) → n = 456 :=
by 
  sorry

end least_positive_multiple_of_24_gt_450_l268_268259


namespace methane_reaction_l268_268083

noncomputable def methane_reacts_with_chlorine
  (moles_CH₄ : ℕ)
  (moles_Cl₂ : ℕ)
  (moles_CCl₄ : ℕ)
  (moles_HCl_produced : ℕ) : Prop :=
  moles_CH₄ = 3 ∧ 
  moles_Cl₂ = 12 ∧ 
  moles_CCl₄ = 3 ∧ 
  moles_HCl_produced = 12

theorem methane_reaction : 
  methane_reacts_with_chlorine 3 12 3 12 :=
by sorry

end methane_reaction_l268_268083


namespace probability_empty_chair_on_sides_7_chairs_l268_268699

theorem probability_empty_chair_on_sides_7_chairs :
  (1 : ℚ) / (35 : ℚ) = 0.2 := by
  sorry

end probability_empty_chair_on_sides_7_chairs_l268_268699


namespace inequality_proof_l268_268494

theorem inequality_proof (a b : ℝ) (h₁ : a ≥ b) (h₂ : b > 0) : 
  2 * a ^ 3 - b ^ 3 ≥ 2 * a * b ^ 2 - a ^ 2 * b := 
by
  sorry

end inequality_proof_l268_268494


namespace train_complete_time_l268_268245

noncomputable def train_time_proof : Prop :=
  ∃ (t_x : ℕ) (v_x : ℝ) (v_y : ℝ),
    v_y = 140 / 3 ∧
    t_x = 140 / v_x ∧
    (∃ t : ℝ, 
      t * v_x = 60.00000000000001 ∧
      t * v_y = 140 - 60.00000000000001) ∧
    t_x = 4

theorem train_complete_time : train_time_proof := by
  sorry

end train_complete_time_l268_268245


namespace voucher_distribution_preferred_plan_l268_268611

section shopping_voucher

variables (X : Type) [fintype X] (draw : finset X)

-- Condition: original draw setup with probabilities
def original_draw (x : X) : ℚ :=
  if x = 200 then 1/45
  else if x = 80 then 16/45
  else if x = 10 then 28/45
  else 0

-- Voucher distribution proof statement
theorem voucher_distribution :
  original_draw X 200 = 1 / 45 ∧ original_draw X 80 = 16 / 45 ∧ original_draw X 10 = 28 / 45 :=
sorry

-- Improvement Plan A setup
def planA_draw (x : X) : ℚ :=
  if x = 200 then 1/22
  else if x = 80 then 9/22
  else if x = 10 then 6/11
  else 0

-- Improvement Plan B setup
def planB_draw (x : X) : ℚ :=
  if x = 210 then 1/45
  else if x = 90 then 16/45
  else if x = 20 then 28/45
  else 0

-- Expected value calculations
def expected_value (P : X → ℚ) (values : X → ℚ) : ℚ :=
  ∑ x, P x * values x

def planA_value := expected_value (planA_draw X) id
def planB_value := expected_value (planB_draw X) id

-- Preferred plan proof statement
theorem preferred_plan : planB_value > planA_value :=
sorry

end shopping_voucher

end voucher_distribution_preferred_plan_l268_268611


namespace smallest_positive_period_of_f_l268_268464

noncomputable def f (x : ℝ) : ℝ := 1 - 3 * Real.sin (x + Real.pi / 4) ^ 2

theorem smallest_positive_period_of_f : ∀ x : ℝ, f (x + Real.pi) = f x :=
by
  intros
  -- Proof is omitted
  sorry

end smallest_positive_period_of_f_l268_268464


namespace count_even_integers_between_300_and_800_with_specified_digits_l268_268662

open Finset

def even_integers_count : ℕ := 52

theorem count_even_integers_between_300_and_800_with_specified_digits :
  let digits := {3, 4, 5, 6, 7, 8}
  ∃! n ∈ (Icc 300 800).filter (λ x, (∃ u t h : ℕ, x = 100 * h + 10 * t + u ∧ u % 2 = 0 ∧ h ∈ digits ∧ t ∈ digits ∧ u ∈ digits ∧ h ≠ t ∧ t ≠ u ∧ h ≠ u)), even_integers_count = 52 :=
sorry

end count_even_integers_between_300_and_800_with_specified_digits_l268_268662


namespace number_of_groups_l268_268724

noncomputable def original_students : ℕ := 22 + 2

def students_per_group : ℕ := 8

theorem number_of_groups : original_students / students_per_group = 3 :=
by
  sorry

end number_of_groups_l268_268724


namespace determine_g_two_l268_268195

variables (a b c d p q r s : ℝ) -- Define variables a, b, c, d, p, q, r, s as real numbers
variables (h₁ : a < b) (h₂ : b < c) (h₃ : c < d) -- The conditions a < b < c < d

noncomputable def f (x : ℝ) : ℝ := x^4 + a*x^3 + b*x^2 + c*x + d
noncomputable def g (x : ℝ) : ℝ := (x - 1/p) * (x - 1/q) * (x - 1/r) * (x - 1/s)

noncomputable def g_two := g 2
noncomputable def f_two := f 2

theorem determine_g_two :
  g_two a b c d = (16 + 8*a + 4*b + 2*c + d) / (p*q*r*s) :=
sorry

end determine_g_two_l268_268195


namespace water_inflow_rate_in_tank_A_l268_268243

-- Definitions from the conditions
def capacity := 20
def inflow_rate_B := 4
def extra_time_A := 5

-- Target variable
noncomputable def inflow_rate_A : ℕ :=
  let time_B := capacity / inflow_rate_B
  let time_A := time_B + extra_time_A
  capacity / time_A

-- Hypotheses
def tank_capacity : capacity = 20 := rfl
def tank_B_inflow : inflow_rate_B = 4 := rfl
def tank_A_extra_time : extra_time_A = 5 := rfl

-- Theorem statement
theorem water_inflow_rate_in_tank_A : inflow_rate_A = 2 := by
  -- Proof would go here
  sorry

end water_inflow_rate_in_tank_A_l268_268243


namespace problem_1_problem_2_l268_268930

variable (a b c : ℝ)

theorem problem_1 
  (hp : 0 < a) 
  (hq : 0 < b) 
  (hr : 0 < c) 
  (hs : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  abc ≤ 1/9 := 
sorry

theorem problem_2 
  (hp : 0 < a) 
  (hq : 0 < b) 
  (hr : 0 < c) 
  (hs : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (abc)) := 
sorry

end problem_1_problem_2_l268_268930


namespace problem1_problem2_l268_268846

variable {a b c : ℝ}

-- Conditions: a, b, c are positive numbers and a^(3/2) + b^(3/2) + c^(3/2) = 1
axiom pos_a : 0 < a
axiom pos_b : 0 < b
axiom pos_c : 0 < c
axiom sum_pow : a^(3/2) + b^(3/2) + c^(3/2) = 1

-- Problem (1): Prove abc ≤ 1/9 given the conditions
theorem problem1 : abc ≤ 1 / 9 := by
  sorry

-- Problem (2): Prove (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ (1 / (2 * (abc) ^ (1/2))) given the conditions
theorem problem2 : (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ 1 / (2 * (abc) ^ (1/2)) := by
  sorry

end problem1_problem2_l268_268846


namespace part1_part2_l268_268973

variable (a b c : ℝ)

-- Condition: a, b, c are positive numbers and a^(3/2) + b^(3/2) + c^(3/2) = 1
axiom pos_a : a > 0
axiom pos_b : b > 0
axiom pos_c : c > 0
axiom sum_condition : a^(3/2) + b^(3/2) + c^(3/2) = 1

-- Proof Problem (1): abc ≤ 1/9
theorem part1 : abc ≤ 1/9 := 
by {
  have h := sorry,
  exact h
}

-- Proof Problem (2): a/(b+c) + b/(a+c) + c/(a+b) ≤ 1/(2sqrt(abc))
theorem part2 : a/(b+c) + b/(a+c) + c/(a+b) ≤ 1/(2*sqrt(abc)) := 
by {
  have h := sorry,
  exact h
}

end part1_part2_l268_268973


namespace winning_probability_correct_l268_268067

-- Define the conditions
def numPowerBalls : ℕ := 30
def numLuckyBalls : ℕ := 49
def numChosenBalls : ℕ := 6

-- Define the probability of picking the correct PowerBall
def powerBallProb : ℚ := 1 / numPowerBalls

-- Define the combination function for choosing LuckyBalls
noncomputable def combination (n k : ℕ) : ℕ := n.choose k

-- Define the probability of picking the correct LuckyBalls
noncomputable def luckyBallProb : ℚ := 1 / (combination numLuckyBalls numChosenBalls)

-- Define the total winning probability
noncomputable def totalWinningProb : ℚ := powerBallProb * luckyBallProb

-- State the theorem to prove
theorem winning_probability_correct : totalWinningProb = 1 / 419512480 :=
by
  sorry

end winning_probability_correct_l268_268067


namespace cos_double_angle_proof_l268_268481

variable {α β : ℝ}

theorem cos_double_angle_proof (h1 : sin (α - β) = 1 / 3) (h2 : cos α * sin β = 1 / 6) :
  cos (2 * α + 2 * β) = 1 / 9 :=
sorry

end cos_double_angle_proof_l268_268481


namespace total_revenue_correct_l268_268389

def KwikETaxCenter : Type := ℕ

noncomputable def federal_return_price : ℕ := 50
noncomputable def state_return_price : ℕ := 30
noncomputable def quarterly_business_taxes_price : ℕ := 80
noncomputable def international_return_price : ℕ := 100
noncomputable def value_added_service_price : ℕ := 75

noncomputable def federal_returns_sold : ℕ := 60
noncomputable def state_returns_sold : ℕ := 20
noncomputable def quarterly_returns_sold : ℕ := 10
noncomputable def international_returns_sold : ℕ := 13
noncomputable def value_added_services_sold : ℕ := 25

noncomputable def international_discount : ℕ := 20

noncomputable def calculate_total_revenue 
   (federal_price : ℕ) (state_price : ℕ) 
   (quarterly_price : ℕ) (international_price : ℕ) 
   (value_added_price : ℕ)
   (federal_sold : ℕ) (state_sold : ℕ) 
   (quarterly_sold : ℕ) (international_sold : ℕ) 
   (value_added_sold : ℕ)
   (discount : ℕ) : ℕ := 
    (federal_price * federal_sold) 
  + (state_price * state_sold) 
  + (quarterly_price * quarterly_sold) 
  + ((international_price - discount) * international_sold) 
  + (value_added_price * value_added_sold)

theorem total_revenue_correct :
  calculate_total_revenue federal_return_price state_return_price 
                          quarterly_business_taxes_price international_return_price 
                          value_added_service_price
                          federal_returns_sold state_returns_sold 
                          quarterly_returns_sold international_returns_sold 
                          value_added_services_sold 
                          international_discount = 7315 := 
  by sorry

end total_revenue_correct_l268_268389


namespace commercial_duration_l268_268366

/-- Michael was watching a TV show, which was aired for 1.5 hours. 
    During this time, there were 3 commercials. 
    The TV show itself, not counting commercials, was 1 hour long. 
    Prove that each commercial lasted 10 minutes. -/
theorem commercial_duration (total_time : ℝ) (num_commercials : ℕ) (show_time : ℝ)
  (h1 : total_time = 1.5) (h2 : num_commercials = 3) (h3 : show_time = 1) :
  (total_time - show_time) / num_commercials * 60 = 10 := 
sorry

end commercial_duration_l268_268366


namespace find_k_for_infinite_solutions_l268_268472

noncomputable def has_infinitely_many_solutions (k : ℝ) : Prop :=
  ∀ x : ℝ, 5 * (3 * x - k) = 3 * (5 * x + 15)

theorem find_k_for_infinite_solutions :
  has_infinitely_many_solutions (-9) :=
by
  sorry

end find_k_for_infinite_solutions_l268_268472


namespace find_polynomial_value_l268_268156

theorem find_polynomial_value (x y : ℝ) 
  (h1 : 3 * x + y = 12) 
  (h2 : x + 3 * y = 16) : 
  10 * x^2 + 14 * x * y + 10 * y^2 = 422.5 := 
by 
  sorry

end find_polynomial_value_l268_268156


namespace projectile_max_height_l268_268439

def h (t : ℝ) : ℝ := -9 * t^2 + 36 * t + 24

theorem projectile_max_height : ∃ t : ℝ, h t = 60 := 
sorry

end projectile_max_height_l268_268439


namespace yellow_shirts_count_l268_268640

theorem yellow_shirts_count (total_shirts blue_shirts green_shirts red_shirts yellow_shirts : ℕ) 
  (h1 : total_shirts = 36) 
  (h2 : blue_shirts = 8) 
  (h3 : green_shirts = 11) 
  (h4 : red_shirts = 6) 
  (h5 : yellow_shirts = total_shirts - (blue_shirts + green_shirts + red_shirts)) :
  yellow_shirts = 11 :=
by
  sorry

end yellow_shirts_count_l268_268640


namespace find_n_for_k_eq_1_l268_268334

theorem find_n_for_k_eq_1 (n : ℤ) (h : (⌊(n^2 : ℤ) / 5⌋ - ⌊n / 2⌋^2 = 1)) : n = 5 := 
by 
  sorry

end find_n_for_k_eq_1_l268_268334


namespace probability_multiple_of_3_or_4_l268_268565

theorem probability_multiple_of_3_or_4 :
  let numbers := Finset.range 30
  let multiples_of_3 := {n ∈ numbers | n % 3 = 0}
  let multiples_of_4 := {n ∈ numbers | n % 4 = 0}
  let multiples_of_12 := {n ∈ numbers | n % 12 = 0}
  let favorable_count := multiples_of_3.card + multiples_of_4.card - multiples_of_12.card
  let probability := (favorable_count : ℚ) / numbers.card
  probability = (1 / 2 : ℚ) :=
by
  sorry

end probability_multiple_of_3_or_4_l268_268565


namespace find_c_and_general_formula_l268_268046

noncomputable def seq (a : ℕ → ℕ) (c : ℕ) := ∀ n : ℕ, a (n + 1) = a n + c * 2^n

theorem find_c_and_general_formula : 
  ∀ (c : ℕ) (a : ℕ → ℕ),
    (a 1 = 2) →
    (seq a c) →
    ((a 3) = (a 1) * ((a 2) / (a 1))^2) →
    ((a 2) = (a 1) * (a 2) / (a 1)) →
    c = 1 ∧ (∀ n, a n = 2^n) := 
by
  sorry

end find_c_and_general_formula_l268_268046


namespace samples_from_workshop_l268_268436

theorem samples_from_workshop (T S P : ℕ) (hT : T = 2048) (hS : S = 128) (hP : P = 256) : 
  (s : ℕ) → (s : ℕ) = (256 * 128 / 2048) → s = 16 :=
by
  intros s hs
  rw [Nat.div_eq (256 * 128) 2048] at hs
  sorry

end samples_from_workshop_l268_268436


namespace student_percentage_l268_268445

theorem student_percentage (s1 s3 overall : ℕ) (percentage_second_subject : ℕ) :
  s1 = 60 →
  s3 = 85 →
  overall = 75 →
  (s1 + percentage_second_subject + s3) / 3 = overall →
  percentage_second_subject = 80 := by
  intros h1 h2 h3 h4
  sorry

end student_percentage_l268_268445


namespace number_of_arrangements_l268_268786

theorem number_of_arrangements (teachers students : Finset ℕ) (h_teachers : teachers.card = 2) (h_students : students.card = 6) :
  let A_teacher := (teachers.choose 1).card,
      A_students := (students.choose 3).card
  in A_teacher * A_students = 40 :=
by
  sorry

end number_of_arrangements_l268_268786


namespace problem1_problem2_l268_268835

variable {a b c : ℝ}

-- Conditions: a, b, c are positive numbers and a^(3/2) + b^(3/2) + c^(3/2) = 1
axiom pos_a : 0 < a
axiom pos_b : 0 < b
axiom pos_c : 0 < c
axiom sum_pow : a^(3/2) + b^(3/2) + c^(3/2) = 1

-- Problem (1): Prove abc ≤ 1/9 given the conditions
theorem problem1 : abc ≤ 1 / 9 := by
  sorry

-- Problem (2): Prove (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ (1 / (2 * (abc) ^ (1/2))) given the conditions
theorem problem2 : (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ 1 / (2 * (abc) ^ (1/2)) := by
  sorry

end problem1_problem2_l268_268835


namespace farmer_kent_income_l268_268118

-- Define the constants and conditions
def watermelon_weight : ℕ := 23
def price_per_pound : ℕ := 2
def number_of_watermelons : ℕ := 18

-- Construct the proof statement
theorem farmer_kent_income : 
  price_per_pound * watermelon_weight * number_of_watermelons = 828 := 
by
  -- Skipping the proof here, just stating the theorem.
  sorry

end farmer_kent_income_l268_268118


namespace johns_weekly_earnings_increase_l268_268526

noncomputable def percentageIncrease (original new : ℝ) : ℝ :=
  ((new - original) / original) * 100

theorem johns_weekly_earnings_increase :
  percentageIncrease 30 40 = 33.33 :=
by
  sorry

end johns_weekly_earnings_increase_l268_268526


namespace deviation_expectation_greater_l268_268248

def frequency_of_heads (m n : Nat) : ℚ := m / n

def deviation_of_frequency (m n : Nat) : ℚ := frequency_of_heads m n - 0.5

def absolute_deviation_of_frequency (m n : Nat) : ℚ := abs (deviation_of_frequency m n)

def expected_absolute_deviation (n : Nat) : ℚ :=
  -- Specify how to calculate the expected absolute deviation formally
  sorry

theorem deviation_expectation_greater (m1 m10 m100 : Nat) :
  expected_absolute_deviation 10 > expected_absolute_deviation 100 :=
sorry

end deviation_expectation_greater_l268_268248


namespace odd_function_expression_l268_268643

noncomputable def f (x : ℝ) : ℝ :=
if x ≥ 0 then x^2 - 2*x else -((-x)^2 - 2*(-x))

theorem odd_function_expression (f : ℝ → ℝ)
  (h_odd : ∀ x : ℝ, f (-x) = -f x)
  (h_pos : ∀ x : ℝ, 0 ≤ x → f x = x^2 - 2*x) :
  ∀ x : ℝ, f x = x * (|x| - 2) :=
by
  sorry

end odd_function_expression_l268_268643


namespace larger_tent_fabric_amount_l268_268277

-- Define the fabric used for the small tent
def small_tent_fabric : ℝ := 4

-- Define the fabric computation for the larger tent
def larger_tent_fabric (small_tent_fabric : ℝ) : ℝ :=
  2 * small_tent_fabric

-- Theorem stating the amount of fabric needed for the larger tent
theorem larger_tent_fabric_amount : larger_tent_fabric small_tent_fabric = 8 :=
by
  -- Skip the actual proof
  sorry

end larger_tent_fabric_amount_l268_268277


namespace difference_in_ages_is_54_l268_268405

theorem difference_in_ages_is_54 (c d : ℕ) (h1 : 10 ≤ c ∧ c < 100 ∧ 10 ≤ d ∧ d < 100) 
    (h2 : 10 * c + d - (10 * d + c) = 9 * (c - d)) 
    (h3 : 10 * c + d + 10 = 3 * (10 * d + c + 10)) : 
    10 * c + d - (10 * d + c) = 54 :=
by
sorry

end difference_in_ages_is_54_l268_268405


namespace coin_diameter_l268_268678

theorem coin_diameter (r : ℝ) (h : r = 7) : 2 * r = 14 := by
  rw [h]
  norm_num

end coin_diameter_l268_268678


namespace linear_regression_change_l268_268169

theorem linear_regression_change : ∀ (x : ℝ), ∀ (y : ℝ), 
  y = 2 - 3.5 * x → (y - (2 - 3.5 * (x + 1))) = 3.5 :=
by
  intros x y h
  sorry

end linear_regression_change_l268_268169


namespace max_rectangle_perimeter_l268_268310

theorem max_rectangle_perimeter (n : ℕ) (a b : ℕ) (ha : a * b = 180) (hb: ∀ (a b : ℕ),  6 ∣ (a * b) → a * b = 180): 
  2 * (a + b) ≤ 184 :=
sorry

end max_rectangle_perimeter_l268_268310


namespace proof_part1_proof_part2_l268_268918

variable {a b c : ℝ}

def pos_and_sum_of_powers (a b c : ℝ) : Prop :=
  0 < a ∧ 0 < b ∧ 0 < c ∧ a^(3/2) + b^(3/2) + c^(3/2) = 1

theorem proof_part1 (h : pos_and_sum_of_powers a b c) : abc ≤ 1 / 9 :=
sorry

theorem proof_part2 (h : pos_and_sum_of_powers a b c) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (abc)) :=
sorry

end proof_part1_proof_part2_l268_268918


namespace gcd_of_XY_is_6_l268_268514

theorem gcd_of_XY_is_6 (X Y : ℕ) (h1 : Nat.lcm X Y = 180)
  (h2 : X * 6 = Y * 5) : Nat.gcd X Y = 6 :=
sorry

end gcd_of_XY_is_6_l268_268514


namespace vasya_lowest_position_l268_268230

theorem vasya_lowest_position
  (n_cyclists : ℕ) (n_stages : ℕ) 
  (stage_positions : ℕ → ℕ → ℕ) -- a function that takes a stage and a cyclist and returns the position (e.g., stage_positions(stage, cyclist) = position)
  (total_time : ℕ → ℕ)  -- a function that takes a cyclist and returns their total time
  (distinct_times : ∀ (c1 c2 : ℕ), c1 ≠ c2 → (total_time c1 ≠ total_time c2) ∧ 
                   ∀ (s : ℕ), stage_positions s c1 ≠ stage_positions s c2)
  (vasya_position : ℕ) (hv : ∀ (s : ℕ), s < n_stages → stage_positions s vasya_position = 7) :
  vasya_position = 91 :=
sorry

end vasya_lowest_position_l268_268230


namespace cylinder_height_in_hemisphere_l268_268763

theorem cylinder_height_in_hemisphere 
  (R : ℝ) (r : ℝ) (h : ℝ) 
  (hemisphere_radius : R = 7) 
  (cylinder_radius : r = 3) 
  (height_eq : h = 2 * Real.sqrt 10) : 
  R^2 = h^2 + r^2 :=
by
  have : h = 2 * Real.sqrt 10 := height_eq
  have : R = 7 := hemisphere_radius
  have : r = 3 := cylinder_radius
  sorry

end cylinder_height_in_hemisphere_l268_268763


namespace sum_of_distinct_prime_divisors_1728_l268_268082

theorem sum_of_distinct_prime_divisors_1728 : 
  (2 + 3 = 5) :=
sorry

end sum_of_distinct_prime_divisors_1728_l268_268082


namespace king_zenobius_more_descendants_l268_268192

-- Conditions
def descendants_paphnutius (p2_descendants p1_descendants: ℕ) := 
  2 + 60 * p2_descendants + 20 * p1_descendants = 142

def descendants_zenobius (z3_descendants z1_descendants : ℕ) := 
  4 + 35 * z3_descendants + 35 * z1_descendants = 144

-- Main statement
theorem king_zenobius_more_descendants:
  ∀ (p2_descendants p1_descendants z3_descendants z1_descendants : ℕ),
    descendants_paphnutius p2_descendants p1_descendants →
    descendants_zenobius z3_descendants z1_descendants →
    144 > 142 :=
by
  intros
  sorry

end king_zenobius_more_descendants_l268_268192


namespace max_area_rectangle_l268_268776

theorem max_area_rectangle (l w : ℝ) 
  (h1 : 2 * l + 2 * w = 60) 
  (h2 : l - w = 10) : 
  l * w = 200 := 
by
  sorry

end max_area_rectangle_l268_268776


namespace part1_part2_l268_268995

-- a), b), c are positive real numbers and ${a}^{\frac{3}{2}}+{b}^{\frac{3}{2}}+{c}^{\frac{3}{2}}=1$
variables (a b c : ℝ)
axiom a_pos : 0 < a
axiom b_pos : 0 < b
axiom c_pos : 0 < c
axiom sum_eq_one : a^(3/2) + b^(3/2) + c^(3/2) = 1

-- Prove (1) and (2)
theorem part1 : abc ≤ 1 / 9 := sorry

theorem part2 : a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (abc)) := sorry

end part1_part2_l268_268995


namespace correct_judgment_l268_268645

def P := Real.pi < 2
def Q := Real.pi > 3

theorem correct_judgment : (P ∨ Q) ∧ ¬P := by
  sorry

end correct_judgment_l268_268645


namespace relationship_among_abc_l268_268319

noncomputable def a : ℝ := Real.sqrt 6 + Real.sqrt 7
noncomputable def b : ℝ := Real.sqrt 5 + Real.sqrt 8
def c : ℝ := 5

theorem relationship_among_abc : c < b ∧ b < a :=
by
  sorry

end relationship_among_abc_l268_268319


namespace cos_double_angle_l268_268477

variables {α β : ℝ}

-- Conditions
def condition1 : Prop := sin (α - β) = 1 / 3
def condition2 : Prop := cos α * sin β = 1 / 6

-- Statement to prove
theorem cos_double_angle (h1 : condition1) (h2 : condition2) : cos (2 * α + 2 * β) = 1 / 9 :=
by
  -- proof goes here
  sorry

end cos_double_angle_l268_268477


namespace problem_1_problem_2_l268_268876

theorem problem_1 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) 
  (h_sum : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  abc ≤ 1 / 9 := 
sorry

theorem problem_2 (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) 
  (h_sum : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * real.sqrt (abc)) := 
sorry

end problem_1_problem_2_l268_268876


namespace height_of_cylinder_correct_l268_268762

noncomputable def height_of_cylinder_inscribed_in_hemisphere : ℝ :=
  let radius_hemisphere := 7 in
  let radius_cylinder := 3 in
  real.sqrt (radius_hemisphere^2 - radius_cylinder^2)

theorem height_of_cylinder_correct :
  height_of_cylinder_inscribed_in_hemisphere = real.sqrt 40 :=
by
  -- Proof skipped
  sorry

end height_of_cylinder_correct_l268_268762


namespace abc_le_one_ninth_sum_ratios_le_one_over_two_sqrt_abc_l268_268816

open Real

variables (a b c : ℝ)

-- Condition: a, b, c are positive numbers
variables (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
-- Condition: a^(3/2) + b^(3/2) + c^(3/2) = 1
variable  (h_sum : a^(3/2) + b^(3/2) + c^(3/2) = 1)

-- Question 1: Prove abc ≤ 1/9
theorem abc_le_one_ninth : a * b * c ≤ 1 / 9 :=
  sorry

-- Question 2: Prove (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ 1 / (2 * sqrt (a * b * c))
theorem sum_ratios_le_one_over_two_sqrt_abc : 
  (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ 1 / (2 * sqrt (a * b * c)) :=
  sorry

end abc_le_one_ninth_sum_ratios_le_one_over_two_sqrt_abc_l268_268816


namespace average_words_per_hour_l268_268753

-- Define the given conditions
variables (W : ℕ) (H : ℕ)

-- State constants for the known values
def words := 60000
def writing_hours := 100

-- Define theorem to prove the average words per hour during the writing phase
theorem average_words_per_hour (h : W = words) (h2 : H = writing_hours) : (W / H) = 600 := by
  sorry

end average_words_per_hour_l268_268753


namespace problem1_problem2_l268_268904

open Real

variable {a b c : ℝ}

-- Condition: a, b, c are positive and a^{3/2} + b^{3/2} + c^{3/2} = 1
def conditions (a b c : ℝ) := a > 0 ∧ b > 0 ∧ c > 0 ∧ a^(3/2) + b^(3/2) + c^(3/2) = 1

-- Problem 1: Prove that abc <= 1/9
theorem problem1 (ha : conditions a b c) : a * b * c ≤ 1 / 9 :=
sorry

-- Problem 2: Prove that a/(b+c) + b/(a+c) + c/(a+b) <= 1 / (2 * sqrt(abc))
theorem problem2 (ha : conditions a b c) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * sqrt (a * b * c)) :=
sorry

end problem1_problem2_l268_268904


namespace part1_part2_l268_268961

def pos_real := {x : ℝ // 0 < x}

variables (a b c : pos_real)

theorem part1 (h : a.val ^ (3/2) + b.val ^ (3/2) + c.val ^ (3/2) = 1) : a.val * b.val * c.val ≤ 1 / 9 :=
sorry

theorem part2 (h : a.val ^ (3/2) + b.val ^ (3/2) + c.val ^ (3/2) = 1) : 
  a.val / (b.val + c.val) + b.val / (a.val + c.val) + c.val / (a.val + b.val) ≤ 1 / (2 * Real.sqrt (a.val * b.val * c.val)) :=
sorry

end part1_part2_l268_268961


namespace range_of_a_l268_268026

noncomputable def function_f (x a : ℝ) : ℝ := 2 * a^x - exp(1) * x^2

theorem range_of_a (a x1 x2 : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (h3 : x1 < x2) 
  (h_min : ∀ x, function_f x a = function_f x1 a → x = x1) 
  (h_max : ∀ x, function_f x a = function_f x2 a → x = x2) 
  (h_critical : ∀ x, 0 = (2 * a^x * log a - 2 * exp(1) * x) → x = x1 ∨ x = x2) :
  a ∈ Ioo (1 / exp(1)) 1 :=
sorry

end range_of_a_l268_268026


namespace problem_part1_problem_part2_l268_268945

variable {a b c : ℝ}

theorem problem_part1 (h : a > 0 ∧ b > 0 ∧ c > 0) (sum_eq : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  abc ≤ 1 / 9 :=
sorry

theorem problem_part2 (h : a > 0 ∧ b > 0 ∧ c > 0) (sum_eq : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (abc)) :=
sorry

end problem_part1_problem_part2_l268_268945


namespace find_real_a_l268_268351

open Complex

noncomputable def pure_imaginary (z : ℂ) : Prop :=
  z.re = 0 ∧ z.im ≠ 0

theorem find_real_a (a : ℝ) (i : ℂ) (h_i : i = Complex.I) :
  pure_imaginary ((2 + i) * (a - (2 * i))) ↔ a = -1 :=
by
  sorry

end find_real_a_l268_268351


namespace regular_price_one_bag_l268_268115

theorem regular_price_one_bag (p : ℕ) (h : 3 * p + 5 = 305) : p = 100 :=
by
  sorry

end regular_price_one_bag_l268_268115


namespace positive_difference_prime_factors_159137_l268_268260

-- Lean 4 Statement Following the Instructions
theorem positive_difference_prime_factors_159137 :
  (159137 = 11 * 17 * 23 * 37) → (37 - 23 = 14) :=
by
  intro h
  sorry -- Proof will be written here

end positive_difference_prime_factors_159137_l268_268260


namespace average_age_of_5_people_l268_268721

theorem average_age_of_5_people (avg_age_18 : ℕ) (avg_age_9 : ℕ) (age_15th : ℕ) (total_persons: ℕ) (persons_9: ℕ) (remaining_persons: ℕ) : 
  avg_age_18 = 15 ∧ 
  avg_age_9 = 16 ∧ 
  age_15th = 56 ∧ 
  total_persons = 18 ∧ 
  persons_9 = 9 ∧ 
  remaining_persons = 5 → 
  (avg_age_18 * total_persons - avg_age_9 * persons_9 - age_15th) / remaining_persons = 14 := 
sorry

end average_age_of_5_people_l268_268721


namespace abc_le_one_ninth_sum_ratios_le_one_over_two_sqrt_abc_l268_268815

open Real

variables (a b c : ℝ)

-- Condition: a, b, c are positive numbers
variables (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
-- Condition: a^(3/2) + b^(3/2) + c^(3/2) = 1
variable  (h_sum : a^(3/2) + b^(3/2) + c^(3/2) = 1)

-- Question 1: Prove abc ≤ 1/9
theorem abc_le_one_ninth : a * b * c ≤ 1 / 9 :=
  sorry

-- Question 2: Prove (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ 1 / (2 * sqrt (a * b * c))
theorem sum_ratios_le_one_over_two_sqrt_abc : 
  (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ 1 / (2 * sqrt (a * b * c)) :=
  sorry

end abc_le_one_ninth_sum_ratios_le_one_over_two_sqrt_abc_l268_268815


namespace ratio_of_red_to_blue_beads_l268_268457

theorem ratio_of_red_to_blue_beads (red_beads blue_beads : ℕ) (h1 : red_beads = 30) (h2 : blue_beads = 20) :
    (red_beads / Nat.gcd red_beads blue_beads) = 3 ∧ (blue_beads / Nat.gcd red_beads blue_beads) = 2 := 
by 
    -- Proof will go here
    sorry

end ratio_of_red_to_blue_beads_l268_268457


namespace problem_1_problem_2_l268_268325

def f (x : ℝ) : ℝ := x^2 - 4 * x + 6

theorem problem_1 (m : ℝ) (h_mono : ∀ x y, m ≤ x → x ≤ y → y ≤ m + 1 → f y ≤ f x) : m ≤ 1 :=
  sorry

theorem problem_2 (a b : ℝ) (h_min : a < b) 
  (h_min_val : ∀ x, a ≤ x ∧ x ≤ b → f a ≤ f x)
  (h_max_val : ∀ x, a ≤ x ∧ x ≤ b → f x ≤ f b) 
  (h_fa_eq_a : f a = a) (h_fb_eq_b : f b = b) : a = 2 ∧ b = 3 :=
  sorry

end problem_1_problem_2_l268_268325


namespace option_C_incorrect_l268_268707

structure Line := (point1 point2 : ℝ × ℝ × ℝ)
structure Plane := (point : ℝ × ℝ × ℝ) (normal : ℝ × ℝ × ℝ)

variables (m n : Line) (α β : Plane)

def line_in_plane (l : Line) (p : Plane) : Prop := sorry
def line_perpendicular_to_plane (l : Line) (p : Plane) : Prop := sorry
def plane_perpendicular_to_plane (p1 p2 : Plane) : Prop := sorry
def line_parallel_to_plane (l : Line) (p : Plane) : Prop := sorry
def lines_parallel (l1 l2 : Line) : Prop := sorry
def lines_perpendicular (l1 l2 : Line) : Prop := sorry
def planes_parallel (p1 p2 : Plane) : Prop := sorry

theorem option_C_incorrect 
  (h1 : line_in_plane m α)
  (h2 : line_parallel_to_plane n α)
  (h3 : lines_parallel m n) :
  false :=
sorry

end option_C_incorrect_l268_268707


namespace maximal_cardinality_set_l268_268355

theorem maximal_cardinality_set (n : ℕ) (h_n : n ≥ 2) :
  ∃ M : Finset (ℕ × ℕ), ∀ (j k : ℕ), (1 ≤ j ∧ j < k ∧ k ≤ n) → 
  ((j, k) ∈ M → ∀ m, (k, m) ∉ M) ∧ 
  M.card = ⌊(n * n / 4 : ℝ)⌋ :=
by
  sorry

end maximal_cardinality_set_l268_268355


namespace problem_1_problem_2_l268_268942

variable {a b c : ℝ}

theorem problem_1 (hp : 0 < a ∧ 0 < b ∧ 0 < c)
    (h1 : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
    abc ≤ 1/9 := 
by 
  apply sorry

theorem problem_2 (hp : 0 < a ∧ 0 < b ∧ 0 < c)
    (h1 : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
    a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * real.sqrt (abc)) := 
by 
  apply sorry

end problem_1_problem_2_l268_268942


namespace football_sampling_l268_268614

theorem football_sampling :
  ∀ (total_members football_members basketball_members volleyball_members total_sample : ℕ),
  total_members = 120 →
  football_members = 40 →
  basketball_members = 60 →
  volleyball_members = 20 →
  total_sample = 24 →
  (total_sample * football_members / (football_members + basketball_members + volleyball_members) = 8) :=
by 
  intros total_members football_members basketball_members volleyball_members total_sample h_total_members h_football_members h_basketball_members h_volleyball_members h_total_sample
  sorry

end football_sampling_l268_268614


namespace largest_integral_k_for_real_distinct_roots_l268_268264

theorem largest_integral_k_for_real_distinct_roots :
  ∃ k : ℤ, (k < 9) ∧ (∀ k' : ℤ, k' < 9 → k' ≤ k) :=
sorry

end largest_integral_k_for_real_distinct_roots_l268_268264


namespace cos_of_double_angles_l268_268486

theorem cos_of_double_angles (α β : ℝ) 
  (h1 : Real.sin (α - β) = 1 / 3) 
  (h2 : Real.cos α * Real.sin β = 1 / 6) : 
  Real.cos (2 * α + 2 * β) = 1 / 9 :=
by
  sorry

end cos_of_double_angles_l268_268486


namespace max_cables_cut_l268_268406

theorem max_cables_cut 
  (initial_computers : ℕ)
  (initial_cables : ℕ)
  (final_clusters : ℕ)
  (H1 : initial_computers = 200)
  (H2 : initial_cables = 345)
  (H3 : final_clusters = 8) 
  : ∃ (cut_cables : ℕ), cut_cables = 153 :=
by
  use 153
  sorry

end max_cables_cut_l268_268406


namespace petya_four_digits_l268_268373

theorem petya_four_digits :
  ∃ (a b c d : ℕ), 
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ 
    a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧ 
    6 * (a + b + c + d) * 1111 = 73326 ∧ 
    (∃ S, S = a + b + c + d ∧ S = 11) :=
begin
  use 1, 2, 3, 5,
  split, { intro h, exact dec_trivial },
  split, { intro h, exact dec_trivial },
  split, { intro h, exact dec_trivial },
  split, { intro h, exact dec_trivial },
  split, { intro h, exact dec_trivial },
  split, { intro h, exact dec_trivial },
  split, { intro h, exact dec_trivial },
  split, { intro h, exact dec_trivial },
  split, { intro h, exact dec_trivial },
  split, { intro h, exact dec_trivial },
  split,
  { exact dec_trivial },
  use 11,
  split,
  { exact dec_trivial },
  { exact dec_trivial }
end

end petya_four_digits_l268_268373


namespace problem_part1_problem_part2_l268_268943

variable {a b c : ℝ}

theorem problem_part1 (h : a > 0 ∧ b > 0 ∧ c > 0) (sum_eq : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  abc ≤ 1 / 9 :=
sorry

theorem problem_part2 (h : a > 0 ∧ b > 0 ∧ c > 0) (sum_eq : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (abc)) :=
sorry

end problem_part1_problem_part2_l268_268943


namespace smallest_positive_integer_x_l268_268014

theorem smallest_positive_integer_x :
  ∃ (x : ℕ), 0 < x ∧ (45 * x + 13) % 17 = 5 % 17 ∧ ∀ y : ℕ, 0 < y ∧ (45 * y + 13) % 17 = 5 % 17 → y ≥ x := 
sorry

end smallest_positive_integer_x_l268_268014


namespace problem_eval_at_x_eq_3_l268_268466

theorem problem_eval_at_x_eq_3 : ∀ x : ℕ, x = 3 → (x^x)^(x^x) = 27^27 :=
by
  intros x hx
  rw [hx]
  sorry

end problem_eval_at_x_eq_3_l268_268466


namespace Yoongi_has_fewest_apples_l268_268695

def Jungkook_apples : Nat := 6 * 3
def Yoongi_apples : Nat := 4
def Yuna_apples : Nat := 5

theorem Yoongi_has_fewest_apples :
  Yoongi_apples < Jungkook_apples ∧ Yoongi_apples < Yuna_apples :=
by
  sorry

end Yoongi_has_fewest_apples_l268_268695


namespace problem_1_problem_2_l268_268824

variable (a b c : ℝ)

theorem problem_1 
  (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  a * b * c ≤ 1 / 9 :=
sorry

theorem problem_2 
  (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * (a * b * c)^(1/2)) :=
sorry

end problem_1_problem_2_l268_268824


namespace max_min_product_xy_theorem_l268_268398

noncomputable def max_min_product_xy (x y a : ℝ) (h1 : x + y = a) (h2 : x^2 + y^2 = -a^2 + 2) : Prop :=
  -1 ≤ x * y ∧ x * y ≤ 1/2

theorem max_min_product_xy_theorem (x y a : ℝ) (h1 : x + y = a) (h2 : x^2 + y^2 = -a^2 + 2) :
  max_min_product_xy x y a h1 h2 :=
sorry

end max_min_product_xy_theorem_l268_268398


namespace part1_part2_l268_268866

variable (a b c : ℝ)

theorem part1 (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) : abc ≤ 1/9 :=
by
  sorry

theorem part2 (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
    (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ 1 / (2 * sqrt(abc)) :=
by
  sorry

end part1_part2_l268_268866


namespace interest_rate_annual_l268_268090

theorem interest_rate_annual :
  ∃ R : ℝ, 
    (5000 * 2 * R / 100) + (3000 * 4 * R / 100) = 2640 ∧ 
    R = 12 :=
sorry

end interest_rate_annual_l268_268090


namespace math_test_total_questions_l268_268727

theorem math_test_total_questions (Q : ℕ) (h : Q - 38 = 7) : Q = 45 :=
by
  sorry

end math_test_total_questions_l268_268727


namespace solve_for_x_in_equation_l268_268716

theorem solve_for_x_in_equation (x : ℝ)
  (h : (2 / 7) * (1 / 4) * x = 12) : x = 168 :=
sorry

end solve_for_x_in_equation_l268_268716


namespace solve_floor_sum_eq_125_l268_268212

def floorSum (x : ℕ) : ℕ :=
  (x - 1) * x * (4 * x + 1) / 6

theorem solve_floor_sum_eq_125 (x : ℕ) (h_pos : 0 < x) : floorSum x = 125 → x = 6 := by
  sorry

end solve_floor_sum_eq_125_l268_268212


namespace marbles_lost_l268_268207

theorem marbles_lost (initial_marbles lost_marbles gifted_marbles remaining_marbles : ℕ) 
  (h_initial : initial_marbles = 85)
  (h_gifted : gifted_marbles = 25)
  (h_remaining : remaining_marbles = 43)
  (h_before_gifting : remaining_marbles + gifted_marbles = initial_marbles - lost_marbles) :
  lost_marbles = 17 :=
by
  sorry

end marbles_lost_l268_268207


namespace gcd_136_1275_l268_268723

theorem gcd_136_1275 : Nat.gcd 136 1275 = 17 := by
sorry

end gcd_136_1275_l268_268723


namespace part1_part2_l268_268972

variable (a b c : ℝ)

-- Condition: a, b, c are positive numbers and a^(3/2) + b^(3/2) + c^(3/2) = 1
axiom pos_a : a > 0
axiom pos_b : b > 0
axiom pos_c : c > 0
axiom sum_condition : a^(3/2) + b^(3/2) + c^(3/2) = 1

-- Proof Problem (1): abc ≤ 1/9
theorem part1 : abc ≤ 1/9 := 
by {
  have h := sorry,
  exact h
}

-- Proof Problem (2): a/(b+c) + b/(a+c) + c/(a+b) ≤ 1/(2sqrt(abc))
theorem part2 : a/(b+c) + b/(a+c) + c/(a+b) ≤ 1/(2*sqrt(abc)) := 
by {
  have h := sorry,
  exact h
}

end part1_part2_l268_268972


namespace parallel_lines_a_eq_2_l268_268679

theorem parallel_lines_a_eq_2 {a : ℝ} :
  (∀ x y : ℝ, a * x + (a + 2) * y + 2 = 0 ∧ x + a * y - 2 = 0 → False) ↔ a = 2 :=
by
  sorry

end parallel_lines_a_eq_2_l268_268679


namespace work_done_resistive_force_l268_268432

noncomputable def mass : ℝ := 0.01  -- 10 grams converted to kilograms
noncomputable def v1 : ℝ := 400.0  -- initial speed in m/s
noncomputable def v2 : ℝ := 100.0  -- final speed in m/s

noncomputable def kinetic_energy (m v : ℝ) : ℝ := 0.5 * m * v^2

theorem work_done_resistive_force :
  let KE1 := kinetic_energy mass v1
  let KE2 := kinetic_energy mass v2
  KE1 - KE2 = 750 :=
by
  sorry

end work_done_resistive_force_l268_268432


namespace pentagon_largest_angle_l268_268341

variable (F G H I J : ℝ)

-- Define the conditions given in the problem
axiom angle_sum : F + G + H + I + J = 540
axiom angle_F : F = 80
axiom angle_G : G = 100
axiom angle_HI : H = I
axiom angle_J : J = 2 * H + 20

-- Statement that the largest angle in the pentagon is 190°
theorem pentagon_largest_angle : max F (max G (max H (max I J))) = 190 :=
sorry

end pentagon_largest_angle_l268_268341


namespace fraction_changes_l268_268675

theorem fraction_changes (x y : ℝ) (h : 0 < x ∧ 0 < y) :
  (x + y) / (x * y) = 2 * ((2 * x + 2 * y) / (2 * x * 2 * y)) :=
by
  sorry

end fraction_changes_l268_268675


namespace lillian_candies_l268_268539

theorem lillian_candies (initial_candies : ℕ) (additional_candies : ℕ) (total_candies : ℕ) :
  initial_candies = 88 → additional_candies = 5 → total_candies = initial_candies + additional_candies → total_candies = 93 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end lillian_candies_l268_268539


namespace cos_double_angle_l268_268478

variables {α β : ℝ}

-- Conditions
def condition1 : Prop := sin (α - β) = 1 / 3
def condition2 : Prop := cos α * sin β = 1 / 6

-- Statement to prove
theorem cos_double_angle (h1 : condition1) (h2 : condition2) : cos (2 * α + 2 * β) = 1 / 9 :=
by
  -- proof goes here
  sorry

end cos_double_angle_l268_268478


namespace num_fixed_points_Q_le_n_l268_268706

-- Define the polynomial P with integer coefficients of degree n > 1
variables {n k : ℕ} (P : Polynomial ℤ)

-- Conditions
-- Degree of polynomial P
hypothesis (hP : P.degree > 1)

-- Positive integer k
hypothesis (hk : k > 0)

-- Define the iterated polynomial Q, with P applied k times
noncomputable def Q : Polynomial ℤ := 
(iterate k (fun T => Polynomial.comp T P)) Polynomial.X

-- Statement of the theorem
theorem num_fixed_points_Q_le_n : 
  ∀ (t : ℤ), Q.eval t = t → ∃ m, 0 ≤ m ∧ m ≤ n := sorry

end num_fixed_points_Q_le_n_l268_268706


namespace vasya_rank_91_l268_268223

theorem vasya_rank_91 {n_cyclists : ℕ} {n_stages : ℕ} 
    (n_cyclists_eq : n_cyclists = 500) 
    (n_stages_eq : n_stages = 15) 
    (no_ties : ∀ (i j : ℕ), i < j → ∀ (s : fin n_stages), ¬(same_time i j s)) 
    (vasya_7th : ∀ (s : fin n_stages), ∀ (i : ℕ), i < 6 → better_than i 6 s) :
    possible_rank vasya ≤ 91 :=
sorry

end vasya_rank_91_l268_268223


namespace smallest_angle_of_trapezoid_l268_268806

theorem smallest_angle_of_trapezoid 
  (a d : ℝ) 
  (h1 : a + 3 * d = 140)
  (h2 : ∀ i j k l : ℝ, i + j = k + l → i + j = 180 ∧ k + l = 180) :
  a = 40 :=
by
  sorry

end smallest_angle_of_trapezoid_l268_268806


namespace farmer_kent_income_l268_268119

-- Define the constants and conditions
def watermelon_weight : ℕ := 23
def price_per_pound : ℕ := 2
def number_of_watermelons : ℕ := 18

-- Construct the proof statement
theorem farmer_kent_income : 
  price_per_pound * watermelon_weight * number_of_watermelons = 828 := 
by
  -- Skipping the proof here, just stating the theorem.
  sorry

end farmer_kent_income_l268_268119


namespace solve_eqn_l268_268382

theorem solve_eqn (x y : ℝ) (h : x^2 + y^2 = 12 * x - 8 * y - 56) : x + y = 2 := by
  sorry

end solve_eqn_l268_268382


namespace max_min_product_xy_l268_268401

-- Definition of conditions
variables (a x y : ℝ)
def condition_1 : Prop := x + y = a
def condition_2 : Prop := x^2 + y^2 = -a^2 + 2

-- The main theorem statement
theorem max_min_product_xy (a : ℝ) (ha_range : -2 ≤ a ∧ a ≤ 2): 
  (∀ x y : ℝ, condition_1 a x y ∧ condition_2 a x y → (x * y) ≤ (1 / 3)) ∧
  (∀ x y : ℝ, condition_1 a x y ∧ condition_2 a x y → (x * y) ≥ (-1)) :=
sorry

end max_min_product_xy_l268_268401


namespace probability_two_of_three_survive_l268_268649

-- Let's define the necessary components
def binomial_coefficient (n k : ℕ) : ℕ := Nat.choose n k

-- Define the probability of exactly 2 out of 3 seedlings surviving
theorem probability_two_of_three_survive (p : ℝ) (h0 : 0 ≤ p) (h1 : p ≤ 1) :
  binomial_coefficient 3 2 * p^2 * (1 - p) = 3 * p^2 * (1 - p) :=
by
  sorry

end probability_two_of_three_survive_l268_268649


namespace problem1_problem2_l268_268838

variable {a b c : ℝ}

-- Conditions: a, b, c are positive numbers and a^(3/2) + b^(3/2) + c^(3/2) = 1
axiom pos_a : 0 < a
axiom pos_b : 0 < b
axiom pos_c : 0 < c
axiom sum_pow : a^(3/2) + b^(3/2) + c^(3/2) = 1

-- Problem (1): Prove abc ≤ 1/9 given the conditions
theorem problem1 : abc ≤ 1 / 9 := by
  sorry

-- Problem (2): Prove (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ (1 / (2 * (abc) ^ (1/2))) given the conditions
theorem problem2 : (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≤ 1 / (2 * (abc) ^ (1/2)) := by
  sorry

end problem1_problem2_l268_268838


namespace cos_double_angle_sum_l268_268490

variables {α β : ℝ}

theorem cos_double_angle_sum (h1: sin (α - β) = 1 / 3) (h2: cos α * sin β = 1 / 6) :
  cos (2 * α + 2 * β) = 1 / 9 :=
sorry

end cos_double_angle_sum_l268_268490


namespace vasya_lowest_position_l268_268229

theorem vasya_lowest_position
  (n_cyclists : ℕ) (n_stages : ℕ) 
  (stage_positions : ℕ → ℕ → ℕ) -- a function that takes a stage and a cyclist and returns the position (e.g., stage_positions(stage, cyclist) = position)
  (total_time : ℕ → ℕ)  -- a function that takes a cyclist and returns their total time
  (distinct_times : ∀ (c1 c2 : ℕ), c1 ≠ c2 → (total_time c1 ≠ total_time c2) ∧ 
                   ∀ (s : ℕ), stage_positions s c1 ≠ stage_positions s c2)
  (vasya_position : ℕ) (hv : ∀ (s : ℕ), s < n_stages → stage_positions s vasya_position = 7) :
  vasya_position = 91 :=
sorry

end vasya_lowest_position_l268_268229


namespace machine_initial_value_l268_268397

-- Conditions
def initial_value (P : ℝ) : Prop := P * (0.75 ^ 2) = 4000

noncomputable def initial_market_value : ℝ := 4000 / (0.75 ^ 2)

-- Proof problem statement
theorem machine_initial_value (P : ℝ) (h : initial_value P) : P = 4000 / (0.75 ^ 2) :=
by
  sorry

end machine_initial_value_l268_268397


namespace total_borders_length_is_15_l268_268443

def garden : ℕ × ℕ := (6, 7)
def num_beds : ℕ := 5
def total_length_of_borders (length width : ℕ) : ℕ := 15

theorem total_borders_length_is_15 :
  ∃ a b : ℕ, 
  garden = (a, b) ∧ 
  num_beds = 5 ∧ 
  total_length_of_borders a b = 15 :=
by
  use (6, 7)
  rw [garden]
  rw [num_beds]
  exact ⟨rfl, rfl, sorry⟩

end total_borders_length_is_15_l268_268443


namespace problem1_problem2_l268_268313

theorem problem1 (a b : ℤ) (h₁ : |a| = 5) (h₂ : |b| = 2) (h₃ : a > b) : a + b = 7 ∨ a + b = 3 := 
by sorry

theorem problem2 (a b : ℤ) (h₁ : |a| = 5) (h₂ : |b| = 2) (h₃ : |a + b| = |a| - |b|) : (a = -5 ∧ b = 2) ∨ (a = 5 ∧ b = -2) := 
by sorry

end problem1_problem2_l268_268313


namespace abs_x_minus_2y_is_square_l268_268796

theorem abs_x_minus_2y_is_square (x y : ℕ) (h : ∃ k : ℤ, x^2 - 4 * y + 1 = (x - 2 * y) * (1 - 2 * y) * k) : ∃ m : ℕ, x - 2 * y = m ^ 2 := by
  sorry

end abs_x_minus_2y_is_square_l268_268796


namespace garden_borders_length_l268_268442

theorem garden_borders_length 
  (a b c d e : ℕ)
  (h1 : 6 * 7 = a^2 + b^2 + c^2 + d^2 + e^2)
  (h2 : a * a + b * b + c * c + d * d + e * e = 42) -- This is analogous to the condition
    
: 15 = (4*a + 4*b + 4*c + 4*d + 4*e - 2*(6 + 7)) / 2 :=
by sorry

end garden_borders_length_l268_268442


namespace ac_lt_bc_of_a_gt_b_and_c_lt_0_l268_268418

theorem ac_lt_bc_of_a_gt_b_and_c_lt_0 {a b c : ℝ} (h1 : a > b) (h2 : c < 0) : a * c < b * c :=
  sorry

end ac_lt_bc_of_a_gt_b_and_c_lt_0_l268_268418


namespace age_of_youngest_child_l268_268788

theorem age_of_youngest_child (mother_fee : ℝ) (child_fee_per_year : ℝ) 
  (total_fee : ℝ) (t : ℝ) (y : ℝ) (child_fee : ℝ)
  (h_mother_fee : mother_fee = 2.50)
  (h_child_fee_per_year : child_fee_per_year = 0.25)
  (h_total_fee : total_fee = 4.00)
  (h_child_fee : child_fee = total_fee - mother_fee)
  (h_y : y = 6 - 2 * t)
  (h_fee_eq : child_fee = y * child_fee_per_year) : y = 2 := 
by
  sorry

end age_of_youngest_child_l268_268788


namespace proof_part1_proof_part2_l268_268850

theorem proof_part1 (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) 
                    (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
                    (a * b * c ≤ 1 / 9) := 
by
  sorry

theorem proof_part2 (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) 
                    (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
                    (a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * (a * b * c)^(1/2))) := 
by
  sorry

end proof_part1_proof_part2_l268_268850


namespace proof_part1_proof_part2_l268_268856

theorem proof_part1 (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) 
                    (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
                    (a * b * c ≤ 1 / 9) := 
by
  sorry

theorem proof_part2 (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) 
                    (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
                    (a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * (a * b * c)^(1/2))) := 
by
  sorry

end proof_part1_proof_part2_l268_268856


namespace circle_center_coordinates_l268_268557

theorem circle_center_coordinates :
  let p1 := (2, -3)
  let p2 := (8, 9)
  let midpoint (x₁ y₁ x₂ y₂ : ℝ) : ℝ × ℝ := ((x₁ + x₂) / 2, (y₁ + y₂) / 2)
  midpoint (2 : ℝ) (-3) 8 9 = (5, 3) :=
by
  sorry

end circle_center_coordinates_l268_268557


namespace Lyka_saves_for_8_weeks_l268_268361

theorem Lyka_saves_for_8_weeks : 
  ∀ (C I W : ℕ), C = 160 → I = 40 → W = 15 → (C - I) / W = 8 := 
by 
  intros C I W hC hI hW
  sorry

end Lyka_saves_for_8_weeks_l268_268361


namespace opposite_numbers_pow_sum_zero_l268_268672

theorem opposite_numbers_pow_sum_zero (a b : ℝ) (h : a + b = 0) : a^5 + b^5 = 0 :=
by sorry

end opposite_numbers_pow_sum_zero_l268_268672


namespace problem_part1_problem_part2_l268_268950

variable {a b c : ℝ}

theorem problem_part1 (h : a > 0 ∧ b > 0 ∧ c > 0) (sum_eq : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  abc ≤ 1 / 9 :=
sorry

theorem problem_part2 (h : a > 0 ∧ b > 0 ∧ c > 0) (sum_eq : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (abc)) :=
sorry

end problem_part1_problem_part2_l268_268950


namespace find_a5_l268_268183

variable {a : ℕ → ℝ}

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n m : ℕ, ∃ q : ℝ, a (n + m) = a n * q ^ m

theorem find_a5
  (h : geometric_sequence a)
  (h3 : a 3 = 2)
  (h7 : a 7 = 8) :
  a 5 = 4 :=
sorry

end find_a5_l268_268183


namespace steps_taken_l268_268453

noncomputable def andrewSpeed : ℝ := 1 -- Let Andrew's speed be represented by 1 feet per minute
noncomputable def benSpeed : ℝ := 3 * andrewSpeed -- Ben's speed is 3 times Andrew's speed
noncomputable def totalDistance : ℝ := 21120 -- Distance between the houses in feet
noncomputable def andrewStep : ℝ := 3 -- Each step of Andrew covers 3 feet

theorem steps_taken : (totalDistance / (andrewSpeed + benSpeed)) * andrewSpeed / andrewStep = 1760 := by
  sorry -- proof to be filled in later

end steps_taken_l268_268453


namespace product_at_n_equals_three_l268_268789

theorem product_at_n_equals_three : (3 - 2) * (3 - 1) * 3 * (3 + 1) * (3 + 2) = 120 := by
  sorry

end product_at_n_equals_three_l268_268789


namespace tank_third_dimension_l268_268758

theorem tank_third_dimension (x : ℕ) (h1 : 4 * 5 = 20) (h2 : 2 * (4 * x) + 2 * (5 * x) = 18 * x) (h3 : (40 + 18 * x) * 20 = 1520) :
  x = 2 :=
by
  sorry

end tank_third_dimension_l268_268758


namespace part1_part2_l268_268989

noncomputable theory

variables {a b c : ℝ}

-- Condition that a, b, c are positive numbers and the given equation holds
def pos_numbers : Prop := a > 0 ∧ b > 0 ∧ c > 0
def sum_condition : Prop := (a ^ (3/2)) + (b ^ (3/2)) + (c ^ (3/2)) = 1

-- Proof goal for the first part: abc <= 1/9
theorem part1 (h₁ : pos_numbers) (h₂ : sum_condition) : 
  a * b * c ≤ 1 / 9 :=
sorry

-- Proof goal for the second part: a/(b+c) + b/(a+c) + c/(a+b) <= 1/(2 * sqrt(abc))
theorem part2 (h₁ : pos_numbers) (h₂ : sum_condition) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * sqrt (a * b * c)) :=
sorry

end part1_part2_l268_268989


namespace no_prime_factor_congruent_to_7_mod_8_l268_268314

open Nat

theorem no_prime_factor_congruent_to_7_mod_8 (n : ℕ) (hn : 0 < n) : 
  ¬ (∃ p : ℕ, p.Prime ∧ p ∣ 2^n + 1 ∧ p % 8 = 7) :=
sorry

end no_prime_factor_congruent_to_7_mod_8_l268_268314


namespace max_mn_on_parabola_l268_268580

theorem max_mn_on_parabola :
  ∀ m n : ℝ, (n = -m^2 + 3) → (m + n ≤ 13 / 4) :=
by
  sorry

end max_mn_on_parabola_l268_268580


namespace flight_duration_l268_268549

theorem flight_duration (h m : ℕ) (H1 : 11 * 60 + 7 < 14 * 60 + 45) (H2 : 0 < m) (H3 : m < 60) :
  h + m = 41 := 
sorry

end flight_duration_l268_268549


namespace solve_system_l268_268385

-- Define the system of equations
def system_of_equations (a b c x y z : ℝ) :=
  x ≠ y ∧
  a ≠ 0 ∧
  c ≠ 0 ∧
  (x + z) * a = x - y ∧
  (x + z) * b = x^2 - y^2 ∧
  (x + z)^2 * (b^2 / (a^2 * c)) = (x^3 + x^2 * y - x * y^2 - y^3)

-- Proof goal: establish the values of x, y, and z
theorem solve_system (a b c x y z : ℝ) (h : system_of_equations a b c x y z):
  x = (a^3 * c + b) / (2 * a) ∧
  y = (b - a^3 * c) / (2 * a) ∧
  z = (2 * a^2 * c - a^3 * c - b) / (2 * a) :=
by
  sorry

end solve_system_l268_268385


namespace intersection_points_range_l268_268461

theorem intersection_points_range (a : ℝ) :
  (∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧ a = x₁^3 - 3 * x₁ ∧
  a = x₂^3 - 3 * x₂ ∧ a = x₃^3 - 3 * x₃) ↔ (-2 < a ∧ a < 2) :=
sorry

end intersection_points_range_l268_268461


namespace minimum_elements_union_l268_268211

open Set

def A : Finset ℕ := sorry
def B : Finset ℕ := sorry

variable (size_A : A.card = 25)
variable (size_B : B.card = 18)
variable (at_least_10_not_in_A : (B \ A).card ≥ 10)

theorem minimum_elements_union : (A ∪ B).card = 35 :=
by
  sorry

end minimum_elements_union_l268_268211


namespace problem1_problem2_l268_268898

open Real

variable {a b c : ℝ}

-- Condition: a, b, c are positive and a^{3/2} + b^{3/2} + c^{3/2} = 1
def conditions (a b c : ℝ) := a > 0 ∧ b > 0 ∧ c > 0 ∧ a^(3/2) + b^(3/2) + c^(3/2) = 1

-- Problem 1: Prove that abc <= 1/9
theorem problem1 (ha : conditions a b c) : a * b * c ≤ 1 / 9 :=
sorry

-- Problem 2: Prove that a/(b+c) + b/(a+c) + c/(a+b) <= 1 / (2 * sqrt(abc))
theorem problem2 (ha : conditions a b c) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * sqrt (a * b * c)) :=
sorry

end problem1_problem2_l268_268898


namespace remainder_when_divided_by_15_l268_268105

def N (k : ℤ) : ℤ := 35 * k + 25

theorem remainder_when_divided_by_15 (k : ℤ) : (N k) % 15 = 10 := 
by 
  -- proof would go here
  sorry

end remainder_when_divided_by_15_l268_268105


namespace original_decimal_number_l268_268711

theorem original_decimal_number (x : ℝ) (h : x / 100 = x - 1.485) : x = 1.5 := 
by
  sorry

end original_decimal_number_l268_268711


namespace find_n_l268_268081

theorem find_n {
    n : ℤ
   } (h1 : 0 ≤ n) (h2 : n < 103) (h3 : 99 * n ≡ 72 [ZMOD 103]) :
    n = 52 :=
sorry

end find_n_l268_268081


namespace find_a_prove_f_pos_l268_268503

noncomputable def f (x a : ℝ) : ℝ := (x - a) * Real.log x + (1 / 2) * x

theorem find_a (a x0 : ℝ) (hx0 : x0 > 0) (h_tangent : (x0 - a) * Real.log x0 + (1 / 2) * x0 = (1 / 2) * x0 ∧ Real.log x0 - a / x0 + 3 / 2 = 1 / 2) :
  a = 1 :=
sorry

theorem prove_f_pos (a : ℝ) (h_range : 1 / (2 * Real.exp 1) < a ∧ a < 2 * Real.sqrt (Real.exp 1)) (x : ℝ) (hx : x > 0) :
  f x a > 0 :=
sorry

end find_a_prove_f_pos_l268_268503


namespace urban_general_hospital_problem_l268_268623

theorem urban_general_hospital_problem
  (a b c d : ℕ)
  (h1 : b = 3 * c)
  (h2 : a = 2 * b)
  (h3 : d = c / 2)
  (h4 : 2 * a + 3 * b + 4 * c + 5 * d = 1500) :
  5 * d = 1500 / 11 := by
  sorry

end urban_general_hospital_problem_l268_268623


namespace hotdog_cost_l268_268349

theorem hotdog_cost
  (h s : ℕ) -- Make sure to assume that the cost in cents is a natural number 
  (h1 : 3 * h + 2 * s = 360)
  (h2 : 2 * h + 3 * s = 390) :
  h = 60 :=

sorry

end hotdog_cost_l268_268349


namespace proof_part1_proof_part2_l268_268910

variable {a b c : ℝ}

def pos_and_sum_of_powers (a b c : ℝ) : Prop :=
  0 < a ∧ 0 < b ∧ 0 < c ∧ a^(3/2) + b^(3/2) + c^(3/2) = 1

theorem proof_part1 (h : pos_and_sum_of_powers a b c) : abc ≤ 1 / 9 :=
sorry

theorem proof_part2 (h : pos_and_sum_of_powers a b c) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (abc)) :=
sorry

end proof_part1_proof_part2_l268_268910


namespace compute_pqr_l268_268039

theorem compute_pqr
  (p q r : ℤ)
  (hp : p ≠ 0)
  (hq : q ≠ 0)
  (hr : r ≠ 0)
  (h_sum : p + q + r = 30)
  (h_eq : 1 / p + 1 / q + 1 / r + 240 / (p * q * r) = 1) :
  p * q * r = 1080 := by
  sorry

end compute_pqr_l268_268039


namespace area_square_B_l268_268390

theorem area_square_B (a b : ℝ) (h1 : a^2 = 25) (h2 : abs (a - b) = 4) : b^2 = 81 :=
by
  sorry

end area_square_B_l268_268390


namespace triangle_largest_angle_l268_268179

theorem triangle_largest_angle (x : ℝ) (hx : x + 2 * x + 3 * x = 180) :
  3 * x = 90 :=
by
  sorry

end triangle_largest_angle_l268_268179


namespace longest_side_of_enclosure_l268_268283

theorem longest_side_of_enclosure (l w : ℝ) (hlw : 2*l + 2*w = 240) (harea : l*w = 2880) : max l w = 72 := 
by {
  sorry
}

end longest_side_of_enclosure_l268_268283


namespace find_coordinates_of_C_l268_268808

structure Point where
  x : Int
  y : Int

def isSymmetricalAboutXAxis (A B : Point) : Prop :=
  A.x = B.x ∧ A.y = -B.y

def isSymmetricalAboutOrigin (B C : Point) : Prop :=
  C.x = -B.x ∧ C.y = -B.y

theorem find_coordinates_of_C :
  ∃ C : Point, let A := Point.mk 2 (-3)
               let B := Point.mk 2 3
               isSymmetricalAboutXAxis A B →
               isSymmetricalAboutOrigin B C →
               C = Point.mk (-2) (-3) :=
by
  sorry

end find_coordinates_of_C_l268_268808


namespace max_min_product_xy_l268_268400

-- Definition of conditions
variables (a x y : ℝ)
def condition_1 : Prop := x + y = a
def condition_2 : Prop := x^2 + y^2 = -a^2 + 2

-- The main theorem statement
theorem max_min_product_xy (a : ℝ) (ha_range : -2 ≤ a ∧ a ≤ 2): 
  (∀ x y : ℝ, condition_1 a x y ∧ condition_2 a x y → (x * y) ≤ (1 / 3)) ∧
  (∀ x y : ℝ, condition_1 a x y ∧ condition_2 a x y → (x * y) ≥ (-1)) :=
sorry

end max_min_product_xy_l268_268400


namespace problem_statement_l268_268336

def f (x : ℝ) : ℝ := x^2
def g (x : ℝ) : ℝ := 3 * x^2 + 4

theorem problem_statement : f (g (-3)) = 961 := by
  sorry

end problem_statement_l268_268336


namespace problem_1_problem_2_l268_268652

-- Proof Problem 1
theorem problem_1 (x : ℝ) : (x^2 + 2 > |x - 4| - |x - 1|) ↔ (x > 1 ∨ x ≤ -1) :=
sorry

-- Proof Problem 2
theorem problem_2 (a : ℝ) (x₁ x₂ : ℝ) : 
  (∀ x₁ x₂, x₁^2 + 2 ≥ |x₂ - a| - |x₂ - 1|) → (-1 ≤ a ∧ a ≤ 3) :=
sorry

end problem_1_problem_2_l268_268652


namespace households_selected_l268_268042

theorem households_selected (H : ℕ) (M L S n h : ℕ)
  (h1 : H = 480)
  (h2 : M = 200)
  (h3 : L = 160)
  (h4 : H = M + L + S)
  (h5 : h = 6)
  (h6 : (h : ℚ) / n = (S : ℚ) / H) : n = 24 :=
by
  sorry

end households_selected_l268_268042


namespace problem_part1_problem_part2_l268_268952

variable {a b c : ℝ}

theorem problem_part1 (h : a > 0 ∧ b > 0 ∧ c > 0) (sum_eq : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  abc ≤ 1 / 9 :=
sorry

theorem problem_part2 (h : a > 0 ∧ b > 0 ∧ c > 0) (sum_eq : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (abc)) :=
sorry

end problem_part1_problem_part2_l268_268952


namespace determine_a_l268_268659

def A (a : ℝ) : Set ℝ := {1, a}
def B (a : ℝ) : Set ℝ := {a^2}

theorem determine_a (a : ℝ) (A_union_B_eq_A : A a ∪ B a = A a) : a = -1 ∨ a = 0 := by
  sorry

end determine_a_l268_268659


namespace Sheila_weekly_earnings_l268_268384

-- Definitions based on the conditions
def hours_per_day_MWF : ℕ := 8
def hours_per_day_TT : ℕ := 6
def hourly_wage : ℕ := 7
def days_MWF : ℕ := 3
def days_TT : ℕ := 2

-- Theorem that Sheila earns $252 per week
theorem Sheila_weekly_earnings : (hours_per_day_MWF * hourly_wage * days_MWF) + (hours_per_day_TT * hourly_wage * days_TT) = 252 :=
by 
  sorry

end Sheila_weekly_earnings_l268_268384


namespace total_charge_for_trip_l268_268692

noncomputable def calc_total_charge (initial_fee : ℝ) (additional_charge : ℝ) (miles : ℝ) (increment : ℝ) :=
  initial_fee + (additional_charge * (miles / increment))

theorem total_charge_for_trip :
  calc_total_charge 2.35 0.35 3.6 (2 / 5) = 8.65 :=
by
  sorry

end total_charge_for_trip_l268_268692


namespace problem_part1_problem_part2_l268_268954

variable {a b c : ℝ}

theorem problem_part1 (h : a > 0 ∧ b > 0 ∧ c > 0) (sum_eq : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  abc ≤ 1 / 9 :=
sorry

theorem problem_part2 (h : a > 0 ∧ b > 0 ∧ c > 0) (sum_eq : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (abc)) :=
sorry

end problem_part1_problem_part2_l268_268954


namespace part1_part2_l268_268980

noncomputable theory

variables {a b c : ℝ}

-- Condition that a, b, c are positive numbers and the given equation holds
def pos_numbers : Prop := a > 0 ∧ b > 0 ∧ c > 0
def sum_condition : Prop := (a ^ (3/2)) + (b ^ (3/2)) + (c ^ (3/2)) = 1

-- Proof goal for the first part: abc <= 1/9
theorem part1 (h₁ : pos_numbers) (h₂ : sum_condition) : 
  a * b * c ≤ 1 / 9 :=
sorry

-- Proof goal for the second part: a/(b+c) + b/(a+c) + c/(a+b) <= 1/(2 * sqrt(abc))
theorem part2 (h₁ : pos_numbers) (h₂ : sum_condition) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * sqrt (a * b * c)) :=
sorry

end part1_part2_l268_268980


namespace Ethan_uses_8_ounces_each_l268_268299

def Ethan (b: ℕ): Prop :=
  let number_of_candles := 10 - 3
  let total_coconut_oil := number_of_candles * 1
  let total_beeswax := 63 - total_coconut_oil
  let beeswax_per_candle := total_beeswax / number_of_candles
  beeswax_per_candle = b

theorem Ethan_uses_8_ounces_each (b: ℕ) (hb: Ethan b): b = 8 :=
  sorry

end Ethan_uses_8_ounces_each_l268_268299


namespace intercepts_l268_268506

def line_equation (x y : ℝ) : Prop :=
  5 * x + 3 * y - 15 = 0

theorem intercepts (a b : ℝ) : line_equation a 0 ∧ line_equation 0 b → (a = 3 ∧ b = 5) :=
  sorry

end intercepts_l268_268506


namespace difference_of_two_numbers_l268_268221

theorem difference_of_two_numbers :
  ∃ S : ℕ, S * 16 + 15 = 1600 ∧ 1600 - S = 1501 :=
by
  sorry

end difference_of_two_numbers_l268_268221


namespace equation_c_is_linear_l268_268263

-- Define the condition for being a linear equation with one variable
def is_linear_equation_with_one_variable (eq : ℝ → Prop) : Prop :=
  ∃ a b : ℝ, a ≠ 0 ∧ ∀ x : ℝ, eq x ↔ (a * x + b = 0)

-- The given equation to check is (x - 1) / 2 = 1, which simplifies to x = 3
def equation_c (x : ℝ) : Prop := (x - 1) / 2 = 1

-- Prove that the given equation is a linear equation with one variable
theorem equation_c_is_linear :
  is_linear_equation_with_one_variable equation_c :=
sorry

end equation_c_is_linear_l268_268263


namespace divisibility_by_7_l268_268142

theorem divisibility_by_7 (A X : Nat) (h1 : A < 10) (h2 : X < 10) : (100001 * A + 100010 * X) % 7 = 0 := 
by
  sorry

end divisibility_by_7_l268_268142


namespace part1_part2_l268_268976

variable (a b c : ℝ)

-- Condition: a, b, c are positive numbers and a^(3/2) + b^(3/2) + c^(3/2) = 1
axiom pos_a : a > 0
axiom pos_b : b > 0
axiom pos_c : c > 0
axiom sum_condition : a^(3/2) + b^(3/2) + c^(3/2) = 1

-- Proof Problem (1): abc ≤ 1/9
theorem part1 : abc ≤ 1/9 := 
by {
  have h := sorry,
  exact h
}

-- Proof Problem (2): a/(b+c) + b/(a+c) + c/(a+b) ≤ 1/(2sqrt(abc))
theorem part2 : a/(b+c) + b/(a+c) + c/(a+b) ≤ 1/(2*sqrt(abc)) := 
by {
  have h := sorry,
  exact h
}

end part1_part2_l268_268976


namespace sara_total_cents_l268_268059

def number_of_quarters : ℕ := 11
def value_per_quarter : ℕ := 25

theorem sara_total_cents : number_of_quarters * value_per_quarter = 275 := by
  sorry

end sara_total_cents_l268_268059


namespace sum_series_eq_l268_268005

open BigOperators

theorem sum_series_eq : 
  ∑ n in Finset.range 256, (1 : ℝ) / ((2 * (n + 1 : ℕ) - 3) * (2 * (n + 1 : ℕ) + 1)) = -257 / 513 := 
by 
  sorry

end sum_series_eq_l268_268005


namespace cos_double_angle_l268_268475

variable {α β : Real}

-- Definitions from the conditions
def sin_diff_condition : Prop := sin (α - β) = 1 / 3
def cos_sin_condition : Prop := cos α * sin β = 1 / 6

-- The main theorem 
theorem cos_double_angle (h₁ : sin_diff_condition) (h₂ : cos_sin_condition) : cos (2 * α + 2 * β) = 1 / 9 :=
by sorry

end cos_double_angle_l268_268475


namespace husband_monthly_savings_l268_268111

theorem husband_monthly_savings :
  let wife_weekly_savings := 100
  let weeks_in_month := 4
  let months := 4
  let total_weeks := weeks_in_month * months
  let wife_savings := wife_weekly_savings * total_weeks
  let stock_price := 50
  let number_of_shares := 25
  let invested_half := stock_price * number_of_shares
  let total_savings := invested_half * 2
  let husband_savings := total_savings - wife_savings
  let monthly_husband_savings := husband_savings / months
  monthly_husband_savings = 225 := 
by 
  sorry

end husband_monthly_savings_l268_268111


namespace Carlos_gave_Rachel_21_blocks_l268_268295

def initial_blocks : Nat := 58
def remaining_blocks : Nat := 37
def given_blocks : Nat := initial_blocks - remaining_blocks

theorem Carlos_gave_Rachel_21_blocks : given_blocks = 21 :=
by
  sorry

end Carlos_gave_Rachel_21_blocks_l268_268295


namespace polynomial_expansion_l268_268791

theorem polynomial_expansion : (x + 3) * (x - 6) * (x + 2) = x^3 - x^2 - 24 * x - 36 := 
by
  sorry

end polynomial_expansion_l268_268791


namespace proof_part1_proof_part2_l268_268851

theorem proof_part1 (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) 
                    (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) : 
                    (a * b * c ≤ 1 / 9) := 
by
  sorry

theorem proof_part2 (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) 
                    (h4 : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
                    (a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * (a * b * c)^(1/2))) := 
by
  sorry

end proof_part1_proof_part2_l268_268851


namespace find_a_l268_268463

def mul_op (a b : ℝ) : ℝ := 2 * a - b^2

theorem find_a (a : ℝ) (h : mul_op a 3 = 7) : a = 8 :=
sorry

end find_a_l268_268463


namespace locus_of_orthocenter_l268_268803

theorem locus_of_orthocenter (A_x A_y : ℝ) (h_A : A_x = 0 ∧ A_y = 2)
    (c_r : ℝ) (h_c : c_r = 2) 
    (M_x M_y Q_x Q_y : ℝ)
    (h_circle : Q_x^2 + Q_y^2 = c_r^2)
    (h_tangent : M_x ≠ 0 ∧ (M_y - 2) / M_x = -Q_x / Q_y)
    (h_M_on_tangent : M_x^2 + (M_y - 2)^2 = 4 ∧ M_x ≠ 0)
    (H_x H_y : ℝ)
    (h_orthocenter : (H_x - A_x)^2 + (H_y - A_y + 2)^2 = 4) :
    (H_x^2 + (H_y - 2)^2 = 4) ∧ (H_x ≠ 0) := 
sorry

end locus_of_orthocenter_l268_268803


namespace max_m_plus_n_l268_268579

noncomputable def quadratic_function (x : ℝ) : ℝ := -x^2 + 3

theorem max_m_plus_n (m n : ℝ) (h : n = quadratic_function m) : m + n ≤ 13/4 :=
sorry

end max_m_plus_n_l268_268579


namespace area_of_triangle_is_11_25_l268_268279

noncomputable def area_of_triangle : ℝ :=
  let A := (1 / 2, 2)
  let B := (8, 2)
  let C := (2, 5)
  let base := (B.1 - A.1 : ℝ)
  let height := (C.2 - A.2 : ℝ)
  0.5 * base * height

theorem area_of_triangle_is_11_25 :
  area_of_triangle = 11.25 := sorry

end area_of_triangle_is_11_25_l268_268279


namespace lawn_unmowed_fraction_l268_268365

noncomputable def rate_mary : ℚ := 1 / 6
noncomputable def rate_tom : ℚ := 1 / 3

theorem lawn_unmowed_fraction :
  (1 : ℚ) - ((1 * rate_tom) + (2 * (rate_mary + rate_tom))) = 1 / 6 :=
by
  -- This part will be the actual proof which we are skipping
  sorry

end lawn_unmowed_fraction_l268_268365


namespace range_of_m_l268_268202

noncomputable def M (m : ℝ) : Set ℝ := {x | x + m ≥ 0}
def N : Set ℝ := {x | x^2 - 2 * x - 8 < 0}
def U : Set ℝ := Set.univ
def CU_M (m : ℝ) : Set ℝ := {x | x < -m}
def empty_intersection (m : ℝ) : Prop := (CU_M m ∩ N = ∅)

theorem range_of_m (m : ℝ) : empty_intersection m → m ≥ 2 := by
  sorry

end range_of_m_l268_268202


namespace tangent_line_at_1_0_monotonic_intervals_l268_268326

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^2 - a * x + 2 * Real.log x

noncomputable def f_derivative (x : ℝ) (a : ℝ) : ℝ := (2 * x^2 - a * x + 2) / x

theorem tangent_line_at_1_0 (a : ℝ) (h : a = 1) :
  ∀ x y : ℝ, 
  (f x a, f 1 a) = (0, x - 1) → 
  y = 3 * x - 3 := 
sorry

theorem monotonic_intervals (a : ℝ) :
  (∀ x : ℝ, 0 < x → f_derivative x a ≥ 0) ↔ (a ≤ 4) ∧ 
  (∀ x : ℝ, 0 < x → 
    (0 < x ∧ x < (a - Real.sqrt (a^2 - 16)) / 4) ∨ 
    ((a + Real.sqrt (a^2 - 16)) / 4 < x) 
  ) :=
sorry

end tangent_line_at_1_0_monotonic_intervals_l268_268326


namespace probability_seven_chairs_probability_n_chairs_l268_268697
-- Importing necessary library to ensure our Lean code can be built successfully

-- Definition for case where n = 7
theorem probability_seven_chairs : 
  let total_seating := 7 * 6 * 5 / 6 
  let favorable_seating := 1 
  let probability := favorable_seating / total_seating 
  probability = 1 / 35 := 
by 
  sorry

-- Definition for general case where n ≥ 6
theorem probability_n_chairs (n : ℕ) (h : n ≥ 6) : 
  let total_seating := (n - 1) * (n - 2) / 2 
  let favorable_seating := (n - 4) * (n - 5) / 2 
  let probability := favorable_seating / total_seating 
  probability = (n - 4) * (n - 5) / ((n - 1) * (n - 2)) := 
by 
  sorry

end probability_seven_chairs_probability_n_chairs_l268_268697


namespace Tammy_earnings_3_weeks_l268_268552

theorem Tammy_earnings_3_weeks
  (trees : ℕ)
  (oranges_per_tree_per_day : ℕ)
  (oranges_per_pack : ℕ)
  (price_per_pack : ℕ)
  (weeks : ℕ) :
  trees = 10 →
  oranges_per_tree_per_day = 12 →
  oranges_per_pack = 6 →
  price_per_pack = 2 →
  weeks = 3 →
  (trees * oranges_per_tree_per_day * weeks * 7) / oranges_per_pack * price_per_pack = 840 :=
by
  intro ht ht12 h6 h2 h3
  -- proof to be filled in here
  sorry

end Tammy_earnings_3_weeks_l268_268552


namespace lines_intersect_at_l268_268109

noncomputable def L₁ (t : ℝ) : ℝ × ℝ := (2 - t, -3 + 4 * t)
noncomputable def L₂ (u : ℝ) : ℝ × ℝ := (-1 + 5 * u, 6 - 7 * u)
noncomputable def point_of_intersection : ℝ × ℝ := (2 / 13, 69 / 13)

theorem lines_intersect_at :
  ∃ t u : ℝ, L₁ t = point_of_intersection ∧ L₂ u = point_of_intersection := 
sorry

end lines_intersect_at_l268_268109


namespace train_length_is_150_l268_268601

noncomputable def train_length_crossing_post (t_post : ℕ := 10) : ℕ := 10
noncomputable def train_length_crossing_platform (length_platform : ℕ := 150) (t_platform : ℕ := 20) : ℕ := 20
def train_constant_speed (L v : ℚ) (t_post t_platform : ℚ) (length_platform : ℚ) : Prop :=
  v = L / t_post ∧ v = (L + length_platform) / t_platform

theorem train_length_is_150 (L : ℚ) (t_post t_platform : ℚ) (length_platform : ℚ) (H : train_constant_speed L v t_post t_platform length_platform) : 
  L = 150 :=
by
  sorry

end train_length_is_150_l268_268601


namespace smallest_positive_x_for_maximum_l268_268138

noncomputable def f (x : ℝ) : ℝ := Real.sin (x / 4) + Real.cos (x / 9)

theorem smallest_positive_x_for_maximum (x : ℝ) :
  (∀ k m : ℤ, x = 360 * (1 + k) ∧ x = 3600 * m ∧ 0 < x → x = 3600) :=
by
  sorry

end smallest_positive_x_for_maximum_l268_268138


namespace sin_double_angle_l268_268669

theorem sin_double_angle (θ : ℝ)
  (h : ∑' n : ℕ, (Real.sin θ)^(2 * n) = 3) :
  Real.sin (2 * θ) = (2 * Real.sqrt 2) / 3 :=
sorry

end sin_double_angle_l268_268669


namespace cos_double_angle_sum_l268_268489

variables {α β : ℝ}

theorem cos_double_angle_sum (h1: sin (α - β) = 1 / 3) (h2: cos α * sin β = 1 / 6) :
  cos (2 * α + 2 * β) = 1 / 9 :=
sorry

end cos_double_angle_sum_l268_268489


namespace magic_square_x_value_l268_268520

theorem magic_square_x_value 
  (a b c d e f g h : ℤ) 
  (h1 : x + b + c = d + e + c)
  (h2 : x + f + e = a + b + d)
  (h3 : x + e + c = a + g + 19)
  (h4 : b + f + e = a + g + 96) 
  (h5 : 19 = b)
  (h6 : 96 = c)
  (h7 : 1 = f)
  (h8 : a + d + x = b + c + f) : 
    x = 200 :=
by
  sorry

end magic_square_x_value_l268_268520


namespace factor_of_quadratic_l268_268609

theorem factor_of_quadratic (m : ℝ) : (∀ x, (x + 6) * (x + a) = x ^ 2 - mx - 42) → m = 1 :=
by sorry

end factor_of_quadratic_l268_268609


namespace largest_4_digit_congruent_to_17_mod_26_l268_268253

theorem largest_4_digit_congruent_to_17_mod_26 :
  ∃ x, x < 10000 ∧ x ≥ 1000 ∧ x % 26 = 17 ∧ (∀ y, y < 10000 ∧ y ≥ 1000 ∧ y % 26 = 17 → y ≤ x) ∧ x = 9972 := 
by
  sorry

end largest_4_digit_congruent_to_17_mod_26_l268_268253


namespace quadratic_has_real_roots_iff_l268_268170

theorem quadratic_has_real_roots_iff (k : ℝ) : (∃ x : ℝ, x^2 + 2*x - k = 0) ↔ k ≥ -1 :=
by
  sorry

end quadratic_has_real_roots_iff_l268_268170


namespace locus_of_centers_l268_268462

-- The Lean 4 statement
theorem locus_of_centers (a b : ℝ) 
  (C1 : (x y : ℝ) → x^2 + y^2 = 1)
  (C2 : (x y : ℝ) → (x - 3)^2 + y^2 = 25) :
  4 * a^2 + 4 * b^2 - 52 * a - 169 = 0 :=
sorry

end locus_of_centers_l268_268462
