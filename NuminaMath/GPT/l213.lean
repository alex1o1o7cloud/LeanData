import Mathlib

namespace real_roots_iff_l213_213624

theorem real_roots_iff (k : ℝ) :
  (∃ x : ℝ, x^2 + 2 * k * x + 3 * k^2 + 2 * k = 0) ↔ (-1 ≤ k ∧ k ≤ 0) :=
by sorry

end real_roots_iff_l213_213624


namespace determinant_of_triangle_angles_l213_213078

theorem determinant_of_triangle_angles (α β γ : ℝ) (h : α + β + γ = Real.pi) :
  Matrix.det ![
    ![Real.tan α, Real.sin α * Real.cos α, 1],
    ![Real.tan β, Real.sin β * Real.cos β, 1],
    ![Real.tan γ, Real.sin γ * Real.cos γ, 1]
  ] = 0 :=
by
  -- Proof statement goes here
  sorry

end determinant_of_triangle_angles_l213_213078


namespace smallest_part_when_divided_l213_213833

theorem smallest_part_when_divided (total : ℝ) (a b c : ℝ) (h_total : total = 150)
                                   (h_a : a = 3) (h_b : b = 5) (h_c : c = 7/2) :
                                   min (min (3 * (total / (a + b + c))) (5 * (total / (a + b + c)))) ((7/2) * (total / (a + b + c))) = 3 * (total / (a + b + c)) :=
by
  -- Mathematical steps have been omitted
  sorry

end smallest_part_when_divided_l213_213833


namespace determinant_of_A_l213_213189

def A : Matrix (Fin 3) (Fin 3) ℤ := ![
  ![3, 0, -2],
  ![8, 5, -4],
  ![3, 3, 6]
]

theorem determinant_of_A : A.det = 108 := by
  sorry

end determinant_of_A_l213_213189


namespace max_value_4287_5_l213_213386

noncomputable def maximum_value_of_expression (x y : ℝ) := x * y * (105 - 2 * x - 5 * y)

theorem max_value_4287_5 (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 2 * x + 5 * y < 105) :
  maximum_value_of_expression x y ≤ 4287.5 :=
sorry

end max_value_4287_5_l213_213386


namespace min_C_over_D_l213_213559

theorem min_C_over_D (x C D : ℝ) (h1 : x^2 + 1/x^2 = C) (h2 : x + 1/x = D) (hC_pos : 0 < C) (hD_pos : 0 < D) : 
  (∃ m : ℝ, m = 2 * Real.sqrt 2 ∧ ∀ y : ℝ, y = C / D → y ≥ m) :=
  sorry

end min_C_over_D_l213_213559


namespace gcf_48_160_120_l213_213101

theorem gcf_48_160_120 : Nat.gcd (Nat.gcd 48 160) 120 = 8 := by
  sorry

end gcf_48_160_120_l213_213101


namespace product_of_two_numbers_l213_213259

theorem product_of_two_numbers (x y : ℝ) (h1 : x + y = 8) (h2 : x^2 + y^2 = 120) :
  x * y = -20 :=
sorry

end product_of_two_numbers_l213_213259


namespace hyperbola_eccentricity_l213_213550

-- Define the conditions of the hyperbola and points
variables {a b c m d : ℝ} (ha : a > 0) (hb : b > 0) 
noncomputable def F1 : ℝ := sorry -- Placeholder for focus F1
noncomputable def F2 : ℝ := sorry -- Placeholder for focus F2
noncomputable def P : ℝ := sorry  -- Placeholder for point P

-- Define the sides of the triangle in terms of an arithmetic progression
def PF2 (m d : ℝ) : ℝ := m - d
def PF1 (m : ℝ) : ℝ := m
def F1F2 (m d : ℝ) : ℝ := m + d

-- Prove that the eccentricity is 5 given the conditions
theorem hyperbola_eccentricity 
  (m d : ℝ) (hc : c = (5 / 2) * d )  
  (h1 : PF1 m = 2 * a)
  (h2 : F1F2 m d = 2 * c)
  (h3 : (PF2 m d)^2 + (PF1 m)^2 = (F1F2 m d)^2 ) :
  (c / a) = 5 := 
sorry

end hyperbola_eccentricity_l213_213550


namespace magic_square_y_value_l213_213777

/-- In a magic square, where the sum of three entries in any row, column, or diagonal is the same value.
    Given the entries as shown below, prove that \(y = -38\).
    The entries are: 
    - \( y \) at position (1,1)
    - 23 at position (1,2)
    - 101 at position (1,3)
    - 4 at position (2,1)
    The remaining positions are denoted as \( a, b, c, d, e \).
-/
theorem magic_square_y_value :
    ∃ y a b c d e: ℤ,
        y + 4 + c = y + 23 + 101 ∧ -- Condition from first column and first row
        23 + a + d = 101 + b + 4 ∧ -- Condition from middle column and diagonal
        c + d + e = 101 + b + e ∧ -- Condition from bottom row and rightmost column
        y + 23 + 101 = 4 + a + b → -- Condition from top row
        y = -38 := 
by
    sorry

end magic_square_y_value_l213_213777


namespace pages_in_first_chapter_l213_213080

theorem pages_in_first_chapter (x : ℕ) (h1 : x + 43 = 80) : x = 37 :=
by
  sorry

end pages_in_first_chapter_l213_213080


namespace find_m_correct_l213_213647

noncomputable def find_m (Q : Point) (B : List Point) (m : ℝ) : Prop :=
  let circle_area := 4 * Real.pi
  let radius := 2
  let area_sector_B1B2 := Real.pi / 3
  let area_region_B1B2 := 1 / 8
  let area_triangle_B1B2 := area_sector_B1B2 - area_region_B1B2 * circle_area
  let area_sector_B4B5 := Real.pi / 3
  let area_region_B4B5 := 1 / 10
  let area_triangle_B4B5 := area_sector_B4B5 - area_region_B4B5 * circle_area
  let area_sector_B9B10 := Real.pi / 3
  let area_region_B9B10 := 4 / 15 - Real.sqrt 3 / m
  let area_triangle_B9B10 := area_sector_B9B10 - area_region_B9B10 * circle_area
  m = 3

theorem find_m_correct (Q : Point) (B : List Point) : find_m Q B 3 :=
by
  unfold find_m
  sorry

end find_m_correct_l213_213647


namespace square_root_of_4_is_pm2_l213_213598

theorem square_root_of_4_is_pm2 : ∃ (x : ℤ), x * x = 4 ∧ (x = 2 ∨ x = -2) := by
  sorry

end square_root_of_4_is_pm2_l213_213598


namespace number_of_pairs_lcm_600_l213_213699

theorem number_of_pairs_lcm_600 :
  ∃ n, n = 53 ∧ (∀ m n : ℕ, (m ≤ n ∧ m > 0 ∧ n > 0 ∧ Nat.lcm m n = 600) ↔ n = 53) := sorry

end number_of_pairs_lcm_600_l213_213699


namespace remainder_of_171_divided_by_21_l213_213985

theorem remainder_of_171_divided_by_21 : 
  ∃ r, 171 = (21 * 8) + r ∧ r = 3 := 
by
  sorry

end remainder_of_171_divided_by_21_l213_213985


namespace multiple_of_second_number_l213_213895

def main : IO Unit := do
  IO.println s!"Proof problem statement in Lean 4."

theorem multiple_of_second_number (x m : ℕ) 
  (h1 : 19 = m * x + 3) 
  (h2 : 19 + x = 27) : 
  m = 2 := 
sorry

end multiple_of_second_number_l213_213895


namespace no_monotonically_decreasing_l213_213719

variable (f : ℝ → ℝ)

theorem no_monotonically_decreasing (x1 x2 : ℝ) (h1 : ∃ x1 x2, x1 < x2 ∧ f x1 ≤ f x2) : ∀ x1 x2, x1 < x2 → f x1 > f x2 → False :=
by
  intros x1 x2 h2 h3
  obtain ⟨a, b, h4, h5⟩ := h1
  have contra := h5
  sorry

end no_monotonically_decreasing_l213_213719


namespace contradiction_method_assumption_l213_213452

theorem contradiction_method_assumption (a b c : ℝ) :
  (¬(a > 0 ∨ b > 0 ∨ c > 0) → false) :=
sorry

end contradiction_method_assumption_l213_213452


namespace remainder_when_divided_by_2000_l213_213109

def S : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}

noncomputable def count_disjoint_subsets (S : Set ℕ) : ℕ :=
  let totalWays := 3^12
  let emptyACases := 2*2^12
  let bothEmptyCase := 1
  (totalWays - emptyACases + bothEmptyCase) / 2

theorem remainder_when_divided_by_2000 : count_disjoint_subsets S % 2000 = 1625 := by
  sorry

end remainder_when_divided_by_2000_l213_213109


namespace average_price_of_towels_l213_213126

-- Definitions based on conditions
def cost_towel1 : ℕ := 3 * 100
def cost_towel2 : ℕ := 5 * 150
def cost_towel3 : ℕ := 2 * 600
def total_cost : ℕ := cost_towel1 + cost_towel2 + cost_towel3
def total_towels : ℕ := 3 + 5 + 2
def average_price : ℕ := total_cost / total_towels

-- Statement to be proved
theorem average_price_of_towels :
  average_price = 225 :=
by
  sorry

end average_price_of_towels_l213_213126


namespace find_number_l213_213050

-- Define the problem constants
def total : ℝ := 1.794
def part1 : ℝ := 0.123
def part2 : ℝ := 0.321
def target : ℝ := 1.350

-- The equivalent proof problem
theorem find_number (x : ℝ) (h : part1 + part2 + x = total) : x = target := by
  -- Proof is intentionally omitted
  sorry

end find_number_l213_213050


namespace intersection_M_N_l213_213627

-- Define the sets based on the given conditions
def M : Set ℝ := {x | x + 2 < 0}
def N : Set ℝ := {x | x + 1 < 0}

-- State the theorem to prove the intersection
theorem intersection_M_N :
  M ∩ N = {x | x < -2} := by
sorry

end intersection_M_N_l213_213627


namespace value_of_expression_l213_213665

theorem value_of_expression :
  (0.00001 * (0.01)^2 * 1000) / 0.001 = 10^(-3) :=
by
  -- Proof goes here
  sorry

end value_of_expression_l213_213665


namespace last_two_digits_square_l213_213687

theorem last_two_digits_square (n : ℕ) (hnz : (n % 10 ≠ 0) ∧ ((n ^ 2) % 100 = n % 10 * 11)): ((n ^ 2) % 100 = 44) :=
sorry

end last_two_digits_square_l213_213687


namespace problem_statement_l213_213750

theorem problem_statement {n d : ℕ} (hn : 0 < n) (hd : 0 < d) (h1 : d ∣ n) (h2 : d^2 * n + 1 ∣ n^2 + d^2) :
  n = d^2 :=
sorry

end problem_statement_l213_213750


namespace unique_triple_property_l213_213124

theorem unique_triple_property (a b c : ℕ) (h1 : a ∣ b * c + 1) (h2 : b ∣ a * c + 1) (h3 : c ∣ a * b + 1) (ha : a > 1) (hb : b > 1) (hc : c > 1) :
  (a = 2 ∧ b = 3 ∧ c = 7) :=
by
  sorry

end unique_triple_property_l213_213124


namespace parallel_lines_regular_ngon_l213_213872

def closed_n_hop_path (n : ℕ) (a : Fin (n + 1) → Fin n) : Prop :=
∀ i j : Fin n, a (i + 1) + a i = a (j + 1) + a j → i = j

theorem parallel_lines_regular_ngon (n : ℕ) (a : Fin (n + 1) → Fin n):
  (Even n → ∃ i j : Fin n, i ≠ j ∧ a (i + 1) + a i = a (j + 1) + a j) ∧
  (Odd n → ¬(∃ i j : Fin n, i ≠ j ∧ a (i + 1) + a i = a (j + 1) + a j ∧ ∀ k l : Fin n, k ≠ l → a (k + 1) + k ≠ a (l + 1) + l)) :=
by
  sorry

end parallel_lines_regular_ngon_l213_213872


namespace simplify_fraction_l213_213250

theorem simplify_fraction : 
  (5 / (2 * Real.sqrt 27 + 3 * Real.sqrt 12 + Real.sqrt 108)) = (5 * Real.sqrt 3 / 54) :=
by
  -- Proof will be provided here
  sorry

end simplify_fraction_l213_213250


namespace geometric_sequence_q_l213_213070

theorem geometric_sequence_q (a : ℕ → ℝ) (q : ℝ) (h_pos : ∀ n, a n > 0)
  (h_geom : ∀ n, a (n + 1) = a n * q)
  (h2 : a 2 * a 6 = 16)
  (h3 : a 4 + a 8 = 8) :
  q = 1 :=
by
  sorry

end geometric_sequence_q_l213_213070


namespace fraction_comparison_l213_213253

theorem fraction_comparison
  (a b c d : ℝ)
  (h1 : a / b < c / d)
  (h2 : b > 0)
  (h3 : d > 0)
  (h4 : b > d) :
  (a + c) / (b + d) < (1 / 2) * (a / b + c / d) :=
by
  sorry

end fraction_comparison_l213_213253


namespace petya_wins_max_margin_l213_213886

theorem petya_wins_max_margin {P1 P2 V1 V2 : ℕ} 
  (h1 : P1 = V1 + 9)
  (h2 : V2 = P2 + 9)
  (h3 : P1 + P2 + V1 + V2 = 27)
  (h4 : P1 + P2 > V1 + V2) :
  ∃ m : ℕ, m = 9 ∧ P1 + P2 - (V1 + V2) = m :=
by
  sorry

end petya_wins_max_margin_l213_213886


namespace determinant_value_l213_213759

variable (a1 b1 b2 c1 c2 c3 d1 d2 d3 d4 : ℝ)

def matrix_det : ℝ :=
  Matrix.det ![
    ![a1, b1, c1, d1],
    ![a1, b2, c2, d2],
    ![a1, b2, c3, d3],
    ![a1, b2, c3, d4]
  ]

theorem determinant_value : 
  matrix_det a1 b1 b2 c1 c2 c3 d1 d2 d3 d4 = 
  a1 * (b2 - b1) * (c3 - c2) * (d4 - d3) :=
by
  sorry

end determinant_value_l213_213759


namespace root_implies_m_values_l213_213577

theorem root_implies_m_values (m : ℝ) :
  (∃ x : ℝ, x = 1 ∧ (m + 2) * x^2 - 2 * x + m^2 - 2 * m - 6 = 0) →
  (m = 3 ∨ m = -2) :=
by
  sorry

end root_implies_m_values_l213_213577


namespace dice_probability_theorem_l213_213384

def at_least_three_same_value_probability (num_dice : ℕ) (num_sides : ℕ) : ℚ :=
  if num_dice = 5 ∧ num_sides = 10 then
    -- Calculating the probability
    (81 / 10000) + (9 / 20000) + (1 / 10000)
  else
    0

theorem dice_probability_theorem :
  at_least_three_same_value_probability 5 10 = 173 / 20000 :=
by
  sorry

end dice_probability_theorem_l213_213384


namespace value_at_minus_two_l213_213558

def f (x : ℝ) (a b c : ℝ) := a * x^5 + b * x^3 + c * x + 1

theorem value_at_minus_two (a b c : ℝ) (h : f 2 a b c = -1) : f (-2) a b c = 3 := by
  sorry

end value_at_minus_two_l213_213558


namespace servings_per_pie_l213_213715

theorem servings_per_pie (serving_apples : ℝ) (guests : ℕ) (pies : ℕ) (apples_per_guest : ℝ)
  (H_servings: serving_apples = 1.5) 
  (H_guests: guests = 12)
  (H_pies: pies = 3)
  (H_apples_per_guest: apples_per_guest = 3) :
  (guests * apples_per_guest) / (serving_apples * pies) = 8 :=
by
  rw [H_servings, H_guests, H_pies, H_apples_per_guest]
  sorry

end servings_per_pie_l213_213715


namespace f_decreasing_on_positive_reals_f_greater_than_2_div_x_plus_2_l213_213714

noncomputable def f (x : ℝ) : ℝ := if x > 0 then (Real.log (1 + x)) / x else 0

theorem f_decreasing_on_positive_reals :
  ∀ x y : ℝ, 0 < x → 0 < y → x < y → f y < f x :=
sorry

theorem f_greater_than_2_div_x_plus_2 :
  ∀ x : ℝ, 0 < x → f x > 2 / (x + 2) :=
sorry

end f_decreasing_on_positive_reals_f_greater_than_2_div_x_plus_2_l213_213714


namespace age_of_B_l213_213068

theorem age_of_B (A B : ℕ) (h1 : A + 10 = 2 * (B - 10)) (h2 : A = B + 11) : B = 41 :=
by
  -- Proof not required as per instructions
  sorry

end age_of_B_l213_213068


namespace alternating_colors_probability_l213_213287

theorem alternating_colors_probability :
  let total_balls : ℕ := 10
  let white_balls : ℕ := 5
  let black_balls : ℕ := 5
  let successful_outcomes : ℕ := 2
  let total_outcomes : ℕ := Nat.choose total_balls white_balls
  (successful_outcomes : ℚ) / (total_outcomes : ℚ) = (1 / 126) := 
by
  let total_balls := 10
  let white_balls := 5
  let black_balls := 5
  let successful_outcomes := 2
  let total_outcomes := Nat.choose total_balls white_balls
  have h_total_outcomes : total_outcomes = 252 := sorry
  have h_probability : (successful_outcomes : ℚ) / (total_outcomes : ℚ) = (1 / 126) := sorry
  exact h_probability

end alternating_colors_probability_l213_213287


namespace dice_roll_probability_bounds_l213_213104

noncomputable def dice_roll_probability : Prop :=
  let n := 80
  let p := (1 : ℝ) / 6
  let q := 1 - p
  let epsilon := 2.58 / 24
  let lower_bound := (p - epsilon) * n
  let upper_bound := (p + epsilon) * n
  5 ≤ lower_bound ∧ upper_bound ≤ 22

theorem dice_roll_probability_bounds :
  dice_roll_probability :=
sorry

end dice_roll_probability_bounds_l213_213104


namespace theater_loss_l213_213404

-- Define the conditions
def theater_capacity : Nat := 50
def ticket_price : Nat := 8
def tickets_sold : Nat := 24

-- Define the maximum revenue and actual revenue
def max_revenue : Nat := theater_capacity * ticket_price
def actual_revenue : Nat := tickets_sold * ticket_price

-- Define the money lost by not selling out
def money_lost : Nat := max_revenue - actual_revenue

-- Theorem statement to prove
theorem theater_loss : money_lost = 208 := by
  sorry

end theater_loss_l213_213404


namespace coordinate_P_condition_1_coordinate_P_condition_2_coordinate_P_condition_3_l213_213293

-- Definition of the conditions
def condition_1 (Px: ℝ) (Py: ℝ) : Prop := Px = 0

def condition_2 (Px: ℝ) (Py: ℝ) : Prop := Py = Px + 3

def condition_3 (Px: ℝ) (Py: ℝ) : Prop := 
  abs Py = 2 ∧ Px > 0 ∧ Py < 0

-- Proof problem for condition 1
theorem coordinate_P_condition_1 : ∃ (Px Py: ℝ), condition_1 Px Py ∧ Px = 0 ∧ Py = -7 := 
  sorry

-- Proof problem for condition 2
theorem coordinate_P_condition_2 : ∃ (Px Py: ℝ), condition_2 Px Py ∧ Px = 10 ∧ Py = 13 :=
  sorry

-- Proof problem for condition 3
theorem coordinate_P_condition_3 : ∃ (Px Py: ℝ), condition_3 Px Py ∧ Px = 5/2 ∧ Py = -2 :=
  sorry

end coordinate_P_condition_1_coordinate_P_condition_2_coordinate_P_condition_3_l213_213293


namespace solve_cos_sin_eq_one_l213_213753

open Real

theorem solve_cos_sin_eq_one (n : ℕ) (hn : n > 0) :
  {x : ℝ | cos x ^ n - sin x ^ n = 1} = {x : ℝ | ∃ k : ℤ, x = k * π} :=
by
  sorry

end solve_cos_sin_eq_one_l213_213753


namespace total_amount_shared_l213_213494

theorem total_amount_shared (A B C : ℕ) (h1 : 3 * B = 5 * A) (h2 : B = 25) (h3 : 5 * C = 8 * B) : A + B + C = 80 := by
  sorry

end total_amount_shared_l213_213494


namespace find_numbers_between_1000_and_4000_l213_213696

theorem find_numbers_between_1000_and_4000 :
  ∃ (x : ℤ), 1000 ≤ x ∧ x ≤ 4000 ∧
             (x % 11 = 2) ∧
             (x % 13 = 12) ∧
             (x % 19 = 18) ∧
             (x = 1234 ∨ x = 3951) :=
sorry

end find_numbers_between_1000_and_4000_l213_213696


namespace john_february_phone_bill_l213_213338

-- Define given conditions
def base_cost : ℕ := 30
def included_hours : ℕ := 50
def overage_cost_per_minute : ℕ := 15 -- costs per minute in cents
def hours_talked_in_February : ℕ := 52

-- Define conversion from dollars to cents
def cents_per_dollar : ℕ := 100

-- Define total cost calculation
def total_cost (base_cost : ℕ) (included_hours : ℕ) (overage_cost_per_minute : ℕ) (hours_talked : ℕ) : ℕ :=
  let extra_minutes := (hours_talked - included_hours) * 60
  let extra_cost := extra_minutes * overage_cost_per_minute
  base_cost * cents_per_dollar + extra_cost

-- State the theorem
theorem john_february_phone_bill : total_cost base_cost included_hours overage_cost_per_minute hours_talked_in_February = 4800 := by
  sorry

end john_february_phone_bill_l213_213338


namespace compute_difference_of_squares_l213_213018

theorem compute_difference_of_squares : (303^2 - 297^2) = 3600 := by
  sorry

end compute_difference_of_squares_l213_213018


namespace solve_inequality_system_l213_213705

theorem solve_inequality_system (x : ℝ) 
  (h1 : 2 * (x - 1) < x + 3)
  (h2 : (x + 1) / 3 - x < 3) : 
  -4 < x ∧ x < 5 := 
  sorry

end solve_inequality_system_l213_213705


namespace combined_votes_l213_213756

theorem combined_votes {A B : ℕ} (h1 : A = 14) (h2 : 2 * B = A) : A + B = 21 := 
by 
sorry

end combined_votes_l213_213756


namespace triangle_area_l213_213481

-- Define the conditions and problem
def BC : ℝ := 10
def height_from_A : ℝ := 12
def AC : ℝ := 13

-- State the main theorem
theorem triangle_area (BC height_from_A AC : ℝ) (hBC : BC = 10) (hheight : height_from_A = 12) (hAC : AC = 13) : 
  (1/2 * BC * height_from_A) = 60 :=
by 
  -- Insert the proof
  sorry

end triangle_area_l213_213481


namespace garden_area_l213_213087

-- Given that the garden is a square with certain properties
variables (s A P : ℕ)

-- Conditions:
-- The perimeter of the square garden is 28 feet
def perimeter_condition : Prop := P = 28

-- The area of the garden is equal to the perimeter plus 21
def area_condition : Prop := A = P + 21

-- The perimeter of a square garden with side length s
def perimeter_def : Prop := P = 4 * s

-- The area of a square garden with side length s
def area_def : Prop := A = s * s

-- Prove that the area A is 49 square feet
theorem garden_area : perimeter_condition P → area_condition P A → perimeter_def s P → area_def s A → A = 49 :=
by 
  sorry

end garden_area_l213_213087


namespace complement_set_P_l213_213898

open Set

theorem complement_set_P (P : Set ℝ) (hP : P = {x : ℝ | x ≥ 1}) : Pᶜ = {x : ℝ | x < 1} :=
sorry

end complement_set_P_l213_213898


namespace sum_a_b_range_l213_213387

noncomputable def f (x : ℝ) : ℝ := 3 / (1 + 3 * x^4)

theorem sum_a_b_range : let a := 0
                       let b := 3
                       a + b = 3 := by
  sorry

end sum_a_b_range_l213_213387


namespace relationship_among_abc_l213_213040

noncomputable def a := Real.sqrt 5 + 2
noncomputable def b := 2 - Real.sqrt 5
noncomputable def c := Real.sqrt 5 - 2

theorem relationship_among_abc : a > c ∧ c > b :=
by
  sorry

end relationship_among_abc_l213_213040


namespace hibiscus_flower_ratio_l213_213567

theorem hibiscus_flower_ratio (x : ℕ) 
  (h1 : 2 + x + 4 * x = 22) : x / 2 = 2 := 
sorry

end hibiscus_flower_ratio_l213_213567


namespace minimum_value_expr_l213_213261

noncomputable def expr (x y z : ℝ) : ℝ := 
  3 * x^2 + 2 * x * y + 3 * y^2 + 2 * y * z + 3 * z^2 - 3 * x + 3 * y - 3 * z + 9

theorem minimum_value_expr : 
  ∃ (x y z : ℝ), ∀ (a b c : ℝ), expr a b c ≥ expr x y z ∧ expr x y z = 3/2 :=
sorry

end minimum_value_expr_l213_213261


namespace seashells_given_to_Joan_l213_213265

def S_original : ℕ := 35
def S_now : ℕ := 17

theorem seashells_given_to_Joan :
  (S_original - S_now) = 18 := by
  sorry

end seashells_given_to_Joan_l213_213265


namespace base7_calculation_result_l213_213601

-- Define the base 7 addition and multiplication
def base7_add (a b : ℕ) := (a + b)
def base7_mul (a b : ℕ) := (a * b)

-- Represent the given numbers in base 10 for calculations:
def num1 : ℕ := 2 * 7 + 5 -- 25 in base 7
def num2 : ℕ := 3 * 7^2 + 3 * 7 + 4 -- 334 in base 7
def mul_factor : ℕ := 2 -- 2 in base 7

-- Addition result
def sum : ℕ := base7_add num1 num2

-- Multiplication result
def result : ℕ := base7_mul sum mul_factor

-- Proving the result is equal to the final answer in base 7
theorem base7_calculation_result : result = 6 * 7^2 + 6 * 7 + 4 := 
by sorry

end base7_calculation_result_l213_213601


namespace distance_between_intersections_is_sqrt3_l213_213557

noncomputable def intersection_distance : ℝ :=
  let C1_polar := (θ : ℝ) → θ = (2 * Real.pi / 3)
  let C2_standard := (x y : ℝ) → (x + Real.sqrt 3)^2 + (y + 2)^2 = 1
  let C3 := (θ : ℝ) → θ = (Real.pi / 3) 
  let C3_cartesian := (x y : ℝ) → y = Real.sqrt 3 * x
  let center := (-Real.sqrt 3, -2)
  let dist_to_C3 := abs (-3 + 2) / 2
  2 * Real.sqrt (1 - (dist_to_C3)^2)

theorem distance_between_intersections_is_sqrt3:
  intersection_distance = Real.sqrt 3 := by
  sorry

end distance_between_intersections_is_sqrt3_l213_213557


namespace shares_of_c_l213_213813

theorem shares_of_c (a b c : ℝ) (h1 : 3 * a = 4 * b) (h2 : 4 * b = 7 * c) (h3 : a + b + c = 427): 
  c = 84 :=
by {
  sorry
}

end shares_of_c_l213_213813


namespace min_value_of_w_l213_213281

noncomputable def w (x y : ℝ) : ℝ := 2 * x^2 + 3 * y^2 + 8 * x - 6 * y + 30

theorem min_value_of_w : ∃ x y : ℝ, ∀ (a b : ℝ), w x y ≤ w a b ∧ w x y = 19 :=
by
  sorry

end min_value_of_w_l213_213281


namespace zeros_of_shifted_function_l213_213073

def f (x : ℝ) : ℝ := x^2 - 1

theorem zeros_of_shifted_function :
  {x : ℝ | f (x - 1) = 0} = {0, 2} :=
sorry

end zeros_of_shifted_function_l213_213073


namespace solution_set_of_inequality_l213_213402

variable (m x : ℝ)

-- Defining the condition
def inequality (m x : ℝ) := x^2 - (2 * m - 1) * x + m^2 - m > 0

-- Problem statement
theorem solution_set_of_inequality (h : inequality m x) : x < m-1 ∨ x > m :=
  sorry

end solution_set_of_inequality_l213_213402


namespace methane_required_l213_213829

def mole_of_methane (moles_of_oxygen : ℕ) : ℕ := 
  if moles_of_oxygen = 2 then 1 else 0

theorem methane_required (moles_of_oxygen : ℕ) : 
  moles_of_oxygen = 2 → mole_of_methane moles_of_oxygen = 1 := 
by 
  intros h
  simp [mole_of_methane, h]

end methane_required_l213_213829


namespace evaluate_complex_fraction_l213_213441

def complex_fraction : Prop :=
  let expr : ℚ := 2 + (3 / (4 + (5 / 6)))
  expr = 76 / 29

theorem evaluate_complex_fraction : complex_fraction :=
by
  let expr : ℚ := 2 + (3 / (4 + (5 / 6)))
  show expr = 76 / 29
  sorry

end evaluate_complex_fraction_l213_213441


namespace min_distinct_integers_for_ap_and_gp_l213_213353

theorem min_distinct_integers_for_ap_and_gp (n : ℕ) :
  (∀ (b q a d : ℤ), b ≠ 0 ∧ q ≠ 0 ∧ q ≠ 1 ∧ q ≠ -1 →
    (∃ (i : ℕ), i < 5 → b * (q ^ i) = a + i * d) ∧ 
    (∃ (j : ℕ), j < 5 → b * (q ^ j) ≠ a + j * d) ↔ n ≥ 6) :=
by {
  sorry
}

end min_distinct_integers_for_ap_and_gp_l213_213353


namespace market_value_calculation_l213_213153

variables (annual_dividend_per_share face_value yield market_value : ℝ)

axiom annual_dividend_definition : annual_dividend_per_share = 0.09 * face_value
axiom face_value_definition : face_value = 100
axiom yield_definition : yield = 0.25

theorem market_value_calculation (annual_dividend_per_share face_value yield market_value : ℝ) 
  (h1: annual_dividend_per_share = 0.09 * face_value)
  (h2: face_value = 100)
  (h3: yield = 0.25):
  market_value = annual_dividend_per_share / yield :=
sorry

end market_value_calculation_l213_213153


namespace line_through_points_l213_213900

theorem line_through_points (x1 y1 x2 y2 : ℝ) (h1 : (x1, y1) = (2, 8)) (h2 : (x2, y2) = (5, 2)) :
  ∃ m b : ℝ, (∀ x, y = m * x + b → (x, y) = (2,8) ∨ (x, y) = (5, 2)) ∧ (m + b = 10) :=
by
  sorry

end line_through_points_l213_213900


namespace min_value_x_plus_y_l213_213755

theorem min_value_x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : 4 / x + 9 / y = 1) : x + y = 25 :=
sorry

end min_value_x_plus_y_l213_213755


namespace mapping_image_l213_213007

theorem mapping_image (f : ℕ → ℕ) (h : ∀ x, f x = x + 1) : f 3 = 4 :=
by {
  sorry
}

end mapping_image_l213_213007


namespace circle_area_l213_213446

theorem circle_area (r : ℝ) (h : 3 * (1 / (2 * π * r)) = r) : 
  π * r^2 = 3 / 2 :=
by
  sorry

end circle_area_l213_213446


namespace mike_disk_space_l213_213179

theorem mike_disk_space (F L T : ℕ) (hF : F = 26) (hL : L = 2) : T = 28 :=
by
  have h : T = F + L := by sorry
  rw [hF, hL] at h
  assumption

end mike_disk_space_l213_213179


namespace total_money_divided_l213_213917

theorem total_money_divided (A B C : ℝ) (hA : A = 280) (h1 : A = (2 / 3) * (B + C)) (h2 : B = (2 / 3) * (A + C)) :
  A + B + C = 700 := by
  sorry

end total_money_divided_l213_213917


namespace Tigers_Sharks_min_games_l213_213642

open Nat

theorem Tigers_Sharks_min_games (N : ℕ) : 
  (let total_games := 3 + N
   let sharks_wins := 1 + N
   sharks_wins * 20 ≥ total_games * 19) ↔ N ≥ 37 := 
by
  sorry

end Tigers_Sharks_min_games_l213_213642


namespace quadratic_term_free_solution_l213_213191

theorem quadratic_term_free_solution (m : ℝ) : 
  (∀ x : ℝ, ∃ (p : ℝ → ℝ), (x + m) * (x^2 + 2*x - 1) = p x + (2 + m) * x^2) → m = -2 :=
by
  intro H
  sorry

end quadratic_term_free_solution_l213_213191


namespace find_angle_A_l213_213023

noncomputable def angle_A (a b : ℝ) (B : ℝ) : ℝ :=
  Real.arcsin ((a * Real.sin B) / b)

theorem find_angle_A :
  ∀ (a b : ℝ) (angle_B : ℝ), 0 < a → 0 < b → 0 < angle_B → angle_B < 180 →
  a = Real.sqrt 2 →
  b = Real.sqrt 3 →
  angle_B = 60 →
  angle_A a b angle_B = 45 :=
by
  intros a b angle_B h1 h2 h3 h4 ha hb hB
  have ha' : a = Real.sqrt 2 := ha
  have hb' : b = Real.sqrt 3 := hb
  have hB' : angle_B = 60 := hB
  -- Proof omitted for demonstration
  sorry

end find_angle_A_l213_213023


namespace right_triangle_sides_l213_213192

theorem right_triangle_sides (a b c : ℕ) (h1 : a < b) 
  (h2 : 2 * c / 2 = c) 
  (h3 : exists x y, (x + y = 8 ∧ a < b) ∨ (x + y = 9 ∧ a < b)) 
  (h4 : a^2 + b^2 = c^2) : 
  a = 3 ∧ b = 4 ∧ c = 5 := 
by
  sorry

end right_triangle_sides_l213_213192


namespace koala_fiber_consumption_l213_213652

theorem koala_fiber_consumption (x : ℝ) (h : 0.40 * x = 8) : x = 20 :=
sorry

end koala_fiber_consumption_l213_213652


namespace evaluate_box_2_neg1_0_l213_213208

def box (a b c : ℤ) : ℚ := a^b - b^c + c^a

theorem evaluate_box_2_neg1_0 : box 2 (-1) 0 = -1/2 := 
by
  sorry

end evaluate_box_2_neg1_0_l213_213208


namespace find_f6_l213_213381

-- Define the function f and the necessary properties
variable (f : ℕ+ → ℕ+)
variable (h1 : ∀ n : ℕ+, f n + f (n + 1) + f (f n) = 3 * n + 1)
variable (h2 : f 1 ≠ 1)

-- State the theorem to prove that f(6) = 5
theorem find_f6 : f 6 = 5 :=
sorry

end find_f6_l213_213381


namespace distributive_addition_over_multiplication_not_hold_l213_213392

def complex_add (z1 z2 : ℝ × ℝ) : ℝ × ℝ :=
(z1.1 + z2.1, z1.2 + z2.2)

def complex_mul (z1 z2 : ℝ × ℝ) : ℝ × ℝ :=
(z1.1 * z2.1 - z1.2 * z2.2, z1.1 * z2.2 + z1.2 * z2.1)

theorem distributive_addition_over_multiplication_not_hold (x y x1 y1 x2 y2 : ℝ) :
  complex_add (x, y) (complex_mul (x1, y1) (x2, y2)) ≠
    complex_mul (complex_add (x, y) (x1, y1)) (complex_add (x, y) (x2, y2)) :=
sorry

end distributive_addition_over_multiplication_not_hold_l213_213392


namespace solve_equation1_solve_equation2_l213_213618

theorem solve_equation1 (x : ℝ) (h1 : 2 * x - 9 = 4 * x) : x = -9 / 2 :=
by
  sorry

theorem solve_equation2 (x : ℝ) (h2 : 5 / 2 * x - 7 / 3 * x = 4 / 3 * 5 - 5) : x = 10 :=
by
  sorry

end solve_equation1_solve_equation2_l213_213618


namespace area_new_rectangle_l213_213635

theorem area_new_rectangle (a b : ℝ) :
  (b + 2 * a) * (b - a) = b^2 + a * b - 2 * a^2 := by
sorry

end area_new_rectangle_l213_213635


namespace packs_of_yellow_bouncy_balls_l213_213228

-- Define the conditions and the question in Lean
variables (GaveAwayGreen : ℝ) (BoughtGreen : ℝ) (BouncyBallsPerPack : ℝ) (TotalKeptBouncyBalls : ℝ) (Y : ℝ)

-- Assume the given conditions
axiom cond1 : GaveAwayGreen = 4.0
axiom cond2 : BoughtGreen = 4.0
axiom cond3 : BouncyBallsPerPack = 10.0
axiom cond4 : TotalKeptBouncyBalls = 80.0

-- Define the theorem statement
theorem packs_of_yellow_bouncy_balls (h1 : GaveAwayGreen = 4.0) (h2 : BoughtGreen = 4.0) (h3 : BouncyBallsPerPack = 10.0) (h4 : TotalKeptBouncyBalls = 80.0) : Y = 8 :=
sorry

end packs_of_yellow_bouncy_balls_l213_213228


namespace AM_GM_inequality_l213_213977

theorem AM_GM_inequality (a b c d : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : d > 0) : 
  (a / b + b / c + c / d + d / a) ≥ 4 := 
sorry

end AM_GM_inequality_l213_213977


namespace total_daily_salary_l213_213884

def manager_salary : ℕ := 5
def clerk_salary : ℕ := 2
def num_managers : ℕ := 2
def num_clerks : ℕ := 3

theorem total_daily_salary : num_managers * manager_salary + num_clerks * clerk_salary = 16 := by
    sorry

end total_daily_salary_l213_213884


namespace acorns_given_is_correct_l213_213891

-- Define initial conditions
def initial_acorns : ℕ := 16
def remaining_acorns : ℕ := 9

-- Define the number of acorns given to her sister
def acorns_given : ℕ := initial_acorns - remaining_acorns

-- Theorem statement
theorem acorns_given_is_correct : acorns_given = 7 := by
  sorry

end acorns_given_is_correct_l213_213891


namespace range_of_a_l213_213771

theorem range_of_a (a : ℝ) :
  (∀ x ∈ Set.Icc 1 2, x^2 - a ≥ 0) ∧ (∃ x : ℝ, x^2 + 2 * a * x + a + 2 = 0) → a ≤ -1 :=
by
  sorry

end range_of_a_l213_213771


namespace prism_faces_eq_nine_l213_213729

-- Define the condition: a prism with 21 edges
def prism_edges (n : ℕ) := n = 21

-- Define the number of sides on each polygonal base
def num_sides (L : ℕ) := 3 * L = 21

-- Define the total number of faces
def total_faces (F : ℕ) (L : ℕ) := F = L + 2

-- The theorem we want to prove
theorem prism_faces_eq_nine (n L F : ℕ) 
  (h1 : prism_edges n)
  (h2 : num_sides L)
  (h3 : total_faces F L) :
  F = 9 := 
sorry

end prism_faces_eq_nine_l213_213729


namespace corrected_mean_l213_213066

theorem corrected_mean (mean_incorrect : ℝ) (number_of_observations : ℕ) (wrong_observation correct_observation : ℝ) : 
  mean_incorrect = 36 → 
  number_of_observations = 50 → 
  wrong_observation = 23 → 
  correct_observation = 43 → 
  (mean_incorrect * number_of_observations + (correct_observation - wrong_observation)) / number_of_observations = 36.4 :=
by
  intros h_mean_incorrect h_number_of_observations h_wrong_observation h_correct_observation
  have S_incorrect : ℝ := mean_incorrect * number_of_observations
  have difference : ℝ := correct_observation - wrong_observation
  have S_correct : ℝ := S_incorrect + difference
  have mean_correct : ℝ := S_correct / number_of_observations
  sorry

end corrected_mean_l213_213066


namespace solution_l213_213678

def question (x : ℝ) : Prop := (x - 5) / ((x - 3) ^ 2) < 0

theorem solution :
  {x : ℝ | question x} = {x : ℝ | x < 3} ∪ {x : ℝ | 3 < x ∧ x < 5} :=
by {
  sorry
}

end solution_l213_213678


namespace distance_from_rachel_to_nicholas_l213_213487

def distance (speed time : ℝ) := speed * time

theorem distance_from_rachel_to_nicholas :
  distance 2 5 = 10 :=
by
  -- Proof goes here
  sorry

end distance_from_rachel_to_nicholas_l213_213487


namespace part1_part2_l213_213614

-- Definition of the function f
def f (x m : ℝ) : ℝ := abs (x - m) + abs (x + 3)

-- Part 1: For m = 1, the solution set of f(x) >= 6
theorem part1 (x : ℝ) : f x 1 ≥ 6 ↔ x ≤ -4 ∨ x ≥ 2 := 
by 
  sorry

-- Part 2: If the inequality f(x) ≤ 2m - 5 has a solution with respect to x, then m ≥ 8
theorem part2 (m : ℝ) (h : ∃ x, f x m ≤ 2 * m - 5) : m ≥ 8 :=
by
  sorry

end part1_part2_l213_213614


namespace mean_equals_l213_213015

theorem mean_equals (z : ℝ) :
    (7 + 10 + 15 + 21) / 4 = (18 + z) / 2 → z = 8.5 := 
by
    intro h
    sorry

end mean_equals_l213_213015


namespace mass_percentage_O_in_CaO_l213_213800

theorem mass_percentage_O_in_CaO :
  (16.00 / (40.08 + 16.00)) * 100 = 28.53 :=
by
  sorry

end mass_percentage_O_in_CaO_l213_213800


namespace quadratic_least_value_l213_213672

variable (a b c : ℝ)

theorem quadratic_least_value (h_a_pos : a > 0)
  (h_c_eq : ∀ x : ℝ, a * x^2 + b * x + c ≥ 9) :
  c = 9 + b^2 / (4 * a) :=
by
  sorry

end quadratic_least_value_l213_213672


namespace abs_sum_bound_l213_213552

theorem abs_sum_bound (k : ℝ) : (∀ x : ℝ, |x + 2| + |x + 1| > k) → k < 1 :=
by {
  sorry
}

end abs_sum_bound_l213_213552


namespace John_spent_15_dollars_on_soap_l213_213033

theorem John_spent_15_dollars_on_soap (number_of_bars : ℕ) (weight_per_bar : ℝ) (cost_per_pound : ℝ)
  (h1 : number_of_bars = 20) (h2 : weight_per_bar = 1.5) (h3 : cost_per_pound = 0.5) :
  (number_of_bars * weight_per_bar * cost_per_pound) = 15 :=
by
  sorry

end John_spent_15_dollars_on_soap_l213_213033


namespace contrapositive_of_odd_even_l213_213668

-- Definitions as conditions
def is_odd (n : ℕ) : Prop := ∃ k, n = 2 * k + 1
def is_even (n : ℕ) : Prop := ∃ k, n = 2 * k

-- Main statement
theorem contrapositive_of_odd_even :
  (∀ a b : ℕ, is_odd a ∧ is_odd b → is_even (a + b)) →
  (∀ a b : ℕ, ¬ is_even (a + b) → ¬ (is_odd a ∧ is_odd b)) := 
by
  intros h a b h1
  sorry

end contrapositive_of_odd_even_l213_213668


namespace exists_schoolchild_who_participated_in_all_competitions_l213_213321

theorem exists_schoolchild_who_participated_in_all_competitions
    (competitions : Fin 50 → Finset ℕ)
    (h_card : ∀ i, (competitions i).card = 30)
    (h_unique : ∀ i j, i ≠ j → competitions i ≠ competitions j)
    (h_intersect : ∀ S : Finset (Fin 50), S.card = 30 → 
      ∃ x, ∀ i ∈ S, x ∈ competitions i) :
    ∃ x, ∀ i, x ∈ competitions i :=
by
  sorry

end exists_schoolchild_who_participated_in_all_competitions_l213_213321


namespace south_movement_notation_l213_213401

/-- If moving north 8m is denoted as +8m, then moving south 5m is denoted as -5m. -/
theorem south_movement_notation (north south : ℤ) (h1 : north = 8) (h2 : south = -north) : south = -5 :=
by
  sorry

end south_movement_notation_l213_213401


namespace find_k_l213_213990

theorem find_k 
  (c : ℝ) (a₁ : ℝ) (S : ℕ → ℝ) (k : ℝ)
  (h1 : ∀ n, S (n+1) = c * S n) 
  (h2 : S 1 = 3 + k)
  (h3 : ∀ n, S n = 3^n + k) :
  k = -1 :=
sorry

end find_k_l213_213990


namespace find_polynomial_value_l213_213145

theorem find_polynomial_value (x y : ℝ) 
  (h1 : 3 * x + y = 12) 
  (h2 : x + 3 * y = 16) : 
  10 * x^2 + 14 * x * y + 10 * y^2 = 422.5 := 
by 
  sorry

end find_polynomial_value_l213_213145


namespace value_added_after_doubling_l213_213201

theorem value_added_after_doubling (x v : ℝ) (h1 : x = 4) (h2 : 2 * x + v = x / 2 + 20) : v = 14 :=
by
  sorry

end value_added_after_doubling_l213_213201


namespace percentage_of_whole_is_10_l213_213681

def part : ℝ := 0.01
def whole : ℝ := 0.1

theorem percentage_of_whole_is_10 : (part / whole) * 100 = 10 := by
  sorry

end percentage_of_whole_is_10_l213_213681


namespace comparison_of_logs_l213_213555

noncomputable def a : ℝ := Real.logb 4 6
noncomputable def b : ℝ := Real.logb 4 0.2
noncomputable def c : ℝ := Real.logb 2 3

theorem comparison_of_logs : c > a ∧ a > b := by
  sorry

end comparison_of_logs_l213_213555


namespace temperature_difference_l213_213357

theorem temperature_difference :
  let T_midnight := -4
  let T_10am := 5
  T_10am - T_midnight = 9 :=
by
  let T_midnight := -4
  let T_10am := 5
  show T_10am - T_midnight = 9
  sorry

end temperature_difference_l213_213357


namespace number_of_remaining_red_points_l213_213102

/-- 
Given a grid where the distance between any two adjacent points in a row or column is 1,
and any green point can turn points within a distance of no more than 1 into green every second.
Initial state of the grid is given. Determine the number of red points after 4 seconds.
-/
def remaining_red_points_after_4_seconds (initial_state : List (List Bool)) : Nat := 
41 -- assume this is the computed number after applying the infection rule for 4 seconds

theorem number_of_remaining_red_points (initial_state : List (List Bool)) :
  remaining_red_points_after_4_seconds initial_state = 41 := 
sorry

end number_of_remaining_red_points_l213_213102


namespace decimal_equivalent_of_fraction_l213_213971

theorem decimal_equivalent_of_fraction :
  (16 : ℚ) / 50 = 32 / 100 :=
by sorry

end decimal_equivalent_of_fraction_l213_213971


namespace time_to_pass_faster_train_l213_213074

noncomputable def speed_slower_train_kmph : ℝ := 36
noncomputable def speed_faster_train_kmph : ℝ := 45
noncomputable def length_faster_train_m : ℝ := 225.018
noncomputable def kmph_to_mps_factor : ℝ := 1000 / 3600

noncomputable def relative_speed_mps : ℝ := (speed_slower_train_kmph + speed_faster_train_kmph) * kmph_to_mps_factor

theorem time_to_pass_faster_train : 
  (length_faster_train_m / relative_speed_mps) = 10.001 := 
sorry

end time_to_pass_faster_train_l213_213074


namespace other_asymptote_l213_213716

theorem other_asymptote (a b : ℝ) :
  (∀ x y : ℝ, y = 2 * x → y - b = a * (x - (-4))) ∧
  (∀ c d : ℝ, c = -4) →
  ∃ m b' : ℝ, m = -1/2 ∧ b' = -10 ∧ ∀ x y : ℝ, y = m * x + b' :=
by
  sorry

end other_asymptote_l213_213716


namespace no_perfect_square_with_one_digit_appending_l213_213480

def append_digit (n : Nat) (d : Fin 10) : Nat :=
  n * 10 + d.val

theorem no_perfect_square_with_one_digit_appending :
  ∀ n : Nat, (∃ k : Nat, k * k = n) → 
  (¬ (∃ d1 : Fin 10, ∃ k : Nat, k * k = append_digit n d1.val) ∧
   ¬ (∃ d2 : Fin 10, ∃ d3 : Fin 10, ∃ k : Nat, k * k = d2.val * 10 ^ (Nat.digits 10 n).length + n * 10 + d3.val)) :=
by sorry

end no_perfect_square_with_one_digit_appending_l213_213480


namespace distance_between_Q_and_R_l213_213975

noncomputable def distance_QR : Real :=
  let YZ := 9
  let XZ := 12
  let XY := 15
  
  -- assume QY = QX and tangent to YZ at Y, and RX = RY and tangent to XZ at X
  let QY := 12.5
  let QX := 12.5
  let RY := 12.5
  let RX := 12.5

  -- calculate and return the distance QR based on these assumptions
  (QX^2 + RY^2 - 2 * QX * RX * Real.cos 90)^(1/2)

theorem distance_between_Q_and_R (YZ XZ XY : ℝ) (QY QX RY RX : ℝ) (h1 : YZ = 9) (h2 : XZ = 12) (h3 : XY = 15)
  (h4 : QY = 12.5) (h5 : QX = 12.5) (h6 : RY = 12.5) (h7 : RX = 12.5) :
  distance_QR = 15 :=
by
  sorry

end distance_between_Q_and_R_l213_213975


namespace surface_area_of_given_cube_l213_213967

-- Define the cube with its volume
def volume_of_cube : ℝ := 4913

-- Define the side length of the cube
def side_of_cube : ℝ := volume_of_cube^(1/3)

-- Define the surface area of the cube
def surface_area_of_cube (side : ℝ) : ℝ := 6 * (side^2)

-- Statement of the theorem
theorem surface_area_of_given_cube : 
  surface_area_of_cube side_of_cube = 1734 := 
by
  -- Proof goes here
  sorry

end surface_area_of_given_cube_l213_213967


namespace sequence_general_term_l213_213144

-- Given a sequence {a_n} whose sum of the first n terms S_n = 2a_n - 1,
-- prove that the general formula for the n-th term a_n is 2^(n-1).

theorem sequence_general_term (S : ℕ → ℕ) (a : ℕ → ℕ)
    (h₁ : ∀ n : ℕ, S n = 2 * a n - 1)
    (h₂ : S 1 = 1) : ∀ n : ℕ, a (n + 1) = 2 ^ n :=
by
  sorry

end sequence_general_term_l213_213144


namespace no_solutions_to_equation_l213_213871

theorem no_solutions_to_equation : ¬∃ x : ℝ, (x ≠ 0) ∧ (x ≠ 5) ∧ ((2 * x ^ 2 - 10 * x) / (x ^ 2 - 5 * x) = x - 3) :=
by
  sorry

end no_solutions_to_equation_l213_213871


namespace ned_pieces_left_l213_213962

def boxes_bought : ℝ := 14.0
def boxes_given : ℝ := 7.0
def pieces_per_box : ℝ := 6.0
def boxes_left (bought : ℝ) (given : ℝ) : ℝ := bought - given
def total_pieces (boxes : ℝ) (pieces_per_box : ℝ) : ℝ := boxes * pieces_per_box

theorem ned_pieces_left : total_pieces (boxes_left boxes_bought boxes_given) pieces_per_box = 42.0 := by
  sorry

end ned_pieces_left_l213_213962


namespace greatest_good_number_smallest_bad_number_l213_213333

def is_good (M : ℕ) : Prop :=
  ∃ (a b c d : ℕ), M ≤ a ∧ a < b ∧ b ≤ c ∧ c < d ∧ d ≤ M + 49 ∧ (a * d = b * c)

def is_good_iff_exists_xy (M : ℕ) : Prop :=
  ∃ (x y : ℕ), x ≤ y ∧ M ≤ x * y ∧ (x + 1) * (y + 1) ≤ M + 49

theorem greatest_good_number : ∃ (M : ℕ), is_good M ∧ ∀ (N : ℕ), is_good N → N ≤ M :=
  by
    use 576
    sorry

theorem smallest_bad_number : ∃ (M : ℕ), ¬is_good M ∧ ∀ (N : ℕ), ¬is_good N → M ≤ N :=
  by
    use 443
    sorry

end greatest_good_number_smallest_bad_number_l213_213333


namespace number_of_students_more_than_pets_l213_213827

theorem number_of_students_more_than_pets 
  (students_per_classroom pets_per_classroom num_classrooms : ℕ)
  (h1 : students_per_classroom = 20)
  (h2 : pets_per_classroom = 3)
  (h3 : num_classrooms = 5) :
  (students_per_classroom * num_classrooms) - (pets_per_classroom * num_classrooms) = 85 := 
by
  sorry

end number_of_students_more_than_pets_l213_213827


namespace max_distance_l213_213375

theorem max_distance (front_lifespan : ℕ) (rear_lifespan : ℕ)
  (h_front : front_lifespan = 21000)
  (h_rear : rear_lifespan = 28000) :
  ∃ (max_dist : ℕ), max_dist = 24000 :=
by
  sorry

end max_distance_l213_213375


namespace wall_length_l213_213219

theorem wall_length (side_mirror : ℝ) (width_wall : ℝ) (length_wall : ℝ) 
  (h_mirror: side_mirror = 18) 
  (h_width: width_wall = 32)
  (h_area: (side_mirror ^ 2) * 2 = width_wall * length_wall):
  length_wall = 20.25 := 
by 
  -- The following 'sorry' is a placeholder for the proof
  sorry

end wall_length_l213_213219


namespace calc_power_expression_l213_213443

theorem calc_power_expression (a b c : ℕ) (h₁ : b = 2) (h₂ : c = 3) :
  3^15 * (3^b)^5 / (3^c)^6 = 2187 := 
sorry

end calc_power_expression_l213_213443


namespace Dan_tshirts_total_l213_213663

theorem Dan_tshirts_total :
  (let rate1 := 1 / 12
   let rate2 := 1 / 6
   let hour := 60
   let tshirts_first_hour := hour * rate1
   let tshirts_second_hour := hour * rate2
   let total_tshirts := tshirts_first_hour + tshirts_second_hour
   total_tshirts) = 15 := by
  sorry

end Dan_tshirts_total_l213_213663


namespace number_of_cuboids_painted_l213_213569

/--
Suppose each cuboid has 6 outer faces and Amelia painted a total of 36 faces.
Prove that the number of cuboids Amelia painted is 6.
-/
theorem number_of_cuboids_painted (total_faces : ℕ) (faces_per_cuboid : ℕ) 
  (h1 : total_faces = 36) (h2 : faces_per_cuboid = 6) :
  total_faces / faces_per_cuboid = 6 := 
by {
  sorry
}

end number_of_cuboids_painted_l213_213569


namespace maximum_automobiles_on_ferry_l213_213009

-- Define the conditions
def ferry_capacity_tons : ℕ := 50
def automobile_min_weight : ℕ := 1600
def automobile_max_weight : ℕ := 3200

-- Define the conversion factor from tons to pounds
def ton_to_pound : ℕ := 2000

-- Define the converted ferry capacity in pounds
def ferry_capacity_pounds := ferry_capacity_tons * ton_to_pound

-- Proof statement
theorem maximum_automobiles_on_ferry : 
  ferry_capacity_pounds / automobile_min_weight = 62 :=
by
  -- Given: ferry capacity is 50 tons and 1 ton = 2000 pounds
  -- Therefore, ferry capacity in pounds is 50 * 2000 = 100000 pounds
  -- The weight of the lightest automobile is 1600 pounds
  -- Maximum number of automobiles = 100000 / 1600 = 62.5
  -- Rounding down to the nearest whole number gives 62
  sorry  -- Proof steps would be filled here

end maximum_automobiles_on_ferry_l213_213009


namespace people_going_to_movie_l213_213216

variable (people_per_car : ℕ) (number_of_cars : ℕ)

theorem people_going_to_movie (h1 : people_per_car = 6) (h2 : number_of_cars = 18) : 
    (people_per_car * number_of_cars) = 108 := 
by
  sorry

end people_going_to_movie_l213_213216


namespace ch4_contains_most_atoms_l213_213285

def molecule_atoms (molecule : String) : Nat :=
  match molecule with
  | "O₂"   => 2
  | "NH₃"  => 4
  | "CO"   => 2
  | "CH₄"  => 5
  | _      => 0

theorem ch4_contains_most_atoms :
  ∀ (a b c d : Nat), 
  a = molecule_atoms "O₂" →
  b = molecule_atoms "NH₃" →
  c = molecule_atoms "CO" →
  d = molecule_atoms "CH₄" →
  d > a ∧ d > b ∧ d > c :=
by
  intros
  sorry

end ch4_contains_most_atoms_l213_213285


namespace express_y_in_terms_of_x_l213_213543

theorem express_y_in_terms_of_x (x y : ℝ) (h : 5 * x + y = 1) : y = 1 - 5 * x :=
by
  sorry

end express_y_in_terms_of_x_l213_213543


namespace intersection_is_correct_l213_213528

def M : Set ℤ := {x | x^2 + 3 * x + 2 > 0}
def N : Set ℤ := {-2, -1, 0, 1, 2}

theorem intersection_is_correct : M ∩ N = {0, 1, 2} := by
  sorry

end intersection_is_correct_l213_213528


namespace fruit_prices_l213_213741

theorem fruit_prices (x y : ℝ) 
  (h₁ : 3 * x + 2 * y = 40) 
  (h₂ : 2 * x + 3 * y = 35) : 
  x = 10 ∧ y = 5 :=
by
  sorry

end fruit_prices_l213_213741


namespace radius_of_C3_correct_l213_213296

noncomputable def radius_of_C3
  (C1 C2 C3 : Type)
  (r1 r2 : ℝ)
  (A B T : Type)
  (TA : ℝ) : ℝ :=
if h1 : r1 = 2 ∧ r2 = 3
    ∧ (TA = 4) -- Conditions 1 and 2
   then 8
   else 0

-- Proof statement
theorem radius_of_C3_correct
  (C1 C2 C3 : Type)
  (r1 r2 : ℝ)
  (A B T : Type)
  (TA : ℝ)
  (h1 : r1 = 2)
  (h2 : r2 = 3)
  (h3 : TA = 4) :
  radius_of_C3 C1 C2 C3 r1 r2 A B T TA = 8 :=
by 
  sorry

end radius_of_C3_correct_l213_213296


namespace sum_of_first_and_last_l213_213712

noncomputable section

variables {A B C D E F G H I : ℕ}

theorem sum_of_first_and_last :
  (D = 8) →
  (A + B + C + D = 50) →
  (B + C + D + E = 50) →
  (C + D + E + F = 50) →
  (D + E + F + G = 50) →
  (E + F + G + H = 50) →
  (F + G + H + I = 50) →
  (A + I = 92) :=
by
  intros hD h1 h2 h3 h4 h5 h6
  sorry

end sum_of_first_and_last_l213_213712


namespace problem_statement_l213_213984

theorem problem_statement (m n : ℝ) (h1 : 1 + 27 = m) (h2 : 3 + 9 = n) : |m - n| = 16 := by
  sorry

end problem_statement_l213_213984


namespace max_quarters_is_13_l213_213190

noncomputable def number_of_quarters (total_value : ℝ) (quarters nickels dimes : ℝ) : Prop :=
  total_value = 4.55 ∧
  quarters = nickels ∧
  dimes = quarters / 2 ∧
  (0.25 * quarters + 0.05 * nickels + 0.05 * quarters / 2 = 4.55)

theorem max_quarters_is_13 : ∃ q : ℝ, number_of_quarters 4.55 q q (q / 2) ∧ q = 13 :=
by
  sorry

end max_quarters_is_13_l213_213190


namespace macaroon_count_l213_213076

def baked_red_macaroons : ℕ := 50
def baked_green_macaroons : ℕ := 40
def ate_green_macaroons : ℕ := 15
def ate_red_macaroons := 2 * ate_green_macaroons

def remaining_macaroons : ℕ := (baked_red_macaroons - ate_red_macaroons) + (baked_green_macaroons - ate_green_macaroons)

theorem macaroon_count : remaining_macaroons = 45 := by
  sorry

end macaroon_count_l213_213076


namespace rows_per_floor_l213_213906

theorem rows_per_floor
  (right_pos : ℕ) (left_pos : ℕ)
  (floors : ℕ) (total_cars : ℕ)
  (h_right : right_pos = 5) (h_left : left_pos = 4)
  (h_floors : floors = 10) (h_total : total_cars = 1600) :
  ∃ rows_per_floor : ℕ, rows_per_floor = 20 :=
by {
  sorry
}

end rows_per_floor_l213_213906


namespace initial_investment_l213_213299

theorem initial_investment :
  ∃ x : ℝ, P = 705.03 ∧ r = 0.12 ∧ n = 5 ∧ P = x * (1 + r)^n ∧ x = 400 :=
by
  let P := 705.03
  let r := 0.12
  let n := 5
  use 400
  simp [P, r, n]
  sorry

end initial_investment_l213_213299


namespace correlation_height_weight_l213_213913

def is_functional_relationship (pair: String) : Prop :=
  pair = "The area of a square and its side length" ∨
  pair = "The distance traveled by a vehicle moving at a constant speed and time"

def has_no_correlation (pair: String) : Prop :=
  pair = "A person's height and eyesight"

def is_correlation (pair: String) : Prop :=
  ¬ is_functional_relationship pair ∧ ¬ has_no_correlation pair

theorem correlation_height_weight :
  is_correlation "A person's height and weight" :=
by sorry

end correlation_height_weight_l213_213913


namespace find_pairs_l213_213169

theorem find_pairs (m n : ℕ) (h1 : 0 < m) (h2 : 0 < n)
  (h3 : (m^2 - n) ∣ (m + n^2)) (h4 : (n^2 - m) ∣ (n + m^2)) :
  (m = 2 ∧ n = 2) ∨ (m = 3 ∧ n = 3) ∨ (m = 1 ∧ n = 2) ∨ (m = 2 ∧ n = 1) ∨ (m = 3 ∧ n = 2) ∨ (m = 2 ∧ n = 3) := by
  sorry

end find_pairs_l213_213169


namespace calvin_gym_duration_l213_213516

theorem calvin_gym_duration (initial_weight loss_per_month final_weight : ℕ) (h1 : initial_weight = 250)
    (h2 : loss_per_month = 8) (h3 : final_weight = 154) : 
    (initial_weight - final_weight) / loss_per_month = 12 :=
by 
  sorry

end calvin_gym_duration_l213_213516


namespace t_sum_max_min_l213_213691

noncomputable def t_max (a b : ℝ) (h : (a - b / 2) ^ 2 = 1 - 7 / 4 * b ^ 2) : ℝ := sorry
noncomputable def t_min (a b : ℝ) (h : (a - b / 2) ^ 2 = 1 - 7 / 4 * b ^ 2) : ℝ := sorry

theorem t_sum_max_min (a b : ℝ) (h : (a - b / 2) ^ 2 = 1 - 7 / 4 * b ^ 2) :
  t_max a b h + t_min a b h = 16 / 7 := sorry

end t_sum_max_min_l213_213691


namespace inverse_property_l213_213645

-- Given conditions
variable (f : ℝ → ℝ) (f_inv : ℝ → ℝ)
variable (hf_injective : Function.Injective f)
variable (hf_surjective : Function.Surjective f)
variable (h_inverse : ∀ y : ℝ, f (f_inv y) = y)
variable (hf_property : ∀ x : ℝ, f (-x) + f (x) = 3)

-- The proof goal
theorem inverse_property (x : ℝ) : (f_inv (x - 1) + f_inv (4 - x)) = 0 :=
by
  sorry

end inverse_property_l213_213645


namespace james_vegetable_consumption_l213_213578

theorem james_vegetable_consumption : 
  (1/4 + 1/4) * 2 * 7 + 3 = 10 := 
by
  sorry

end james_vegetable_consumption_l213_213578


namespace rectangle_perimeter_is_36_l213_213773

theorem rectangle_perimeter_is_36 (a b : ℕ) (h : a ≠ b) (h1 : a * b = 2 * (2 * a + 2 * b) - 8) : 2 * (a + b) = 36 :=
  sorry

end rectangle_perimeter_is_36_l213_213773


namespace value_of_x_plus_y_pow_2023_l213_213634

theorem value_of_x_plus_y_pow_2023 (x y : ℝ) (h : abs (x - 2) + abs (y + 3) = 0) : 
  (x + y) ^ 2023 = -1 := 
sorry

end value_of_x_plus_y_pow_2023_l213_213634


namespace problem_l213_213314

def f (x a : ℝ) : ℝ := x^2 + a*x - 3*a - 9

theorem problem (a : ℝ) (h : ∀ x : ℝ, f x a ≥ 0) : f 1 a = 4 :=
sorry

end problem_l213_213314


namespace martha_painting_rate_l213_213745

noncomputable def martha_square_feet_per_hour
  (width1 : ℕ) (width2 : ℕ) (height : ℕ) (coats : ℕ) (total_hours : ℕ) 
  (pair1_walls : ℕ) (pair2_walls : ℕ) : ℕ :=
  let pair1_total_area := width1 * height * pair1_walls
  let pair2_total_area := width2 * height * pair2_walls
  let total_area := pair1_total_area + pair2_total_area
  let total_paint_area := total_area * coats
  total_paint_area / total_hours

theorem martha_painting_rate :
  martha_square_feet_per_hour 12 16 10 3 42 2 2 = 40 :=
by
  -- Proof goes here
  sorry

end martha_painting_rate_l213_213745


namespace function_range_cosine_identity_l213_213383

theorem function_range_cosine_identity
  (f : ℝ → ℝ)
  (ω : ℝ)
  (h₀ : 0 < ω)
  (h₁ : ∀ x, f x = (1/2) * Real.cos (ω * x) - (Real.sqrt 3 / 2) * Real.sin (ω * x))
  (h₂ : ∀ x, f (x + π / ω) = f x) :
  Set.Icc (f (-π / 3)) (f (π / 6)) = Set.Icc (-1 / 2) 1 :=
by
  sorry

end function_range_cosine_identity_l213_213383


namespace johns_equation_l213_213735

theorem johns_equation (a b c d e : ℤ) (ha : a = 2) (hb : b = 3) 
  (hc : c = 4) (hd : d = 5) : 
  a - (b - (c * (d - e))) = a - b - c * d + e ↔ e = 8 := 
by
  sorry

end johns_equation_l213_213735


namespace simplify_expression_l213_213661

theorem simplify_expression : 1 - (1 / (1 + Real.sqrt 5)) + (1 / (1 - Real.sqrt 5)) = 1 := 
by sorry

end simplify_expression_l213_213661


namespace find_k_l213_213722

theorem find_k (k : ℝ) (h : (2 * (7:ℝ)^2) + 3 * 7 - k = 0) : k = 119 := by
  sorry

end find_k_l213_213722


namespace maximum_number_of_buses_l213_213202

-- Definitions of the conditions
def bus_stops : ℕ := 9 -- There are 9 bus stops in the city.
def max_common_stops : ℕ := 1 -- Any two buses have at most one common stop.
def stops_per_bus : ℕ := 3 -- Each bus stops at exactly three stops.

-- The main theorem statement
theorem maximum_number_of_buses (n : ℕ) :
  (bus_stops = 9) →
  (max_common_stops = 1) →
  (stops_per_bus = 3) →
  n ≤ 12 :=
sorry

end maximum_number_of_buses_l213_213202


namespace boys_count_l213_213365

/-
Conditions:
1. The total number of members in the chess team is 26.
2. 18 members were present at the last session.
3. One-third of the girls attended the session.
4. All of the boys attended the session.
-/
def TotalMembers : Nat := 26
def LastSessionAttendance : Nat := 18
def GirlsAttendance (G : Nat) : Nat := G / 3
def BoysAttendance (B : Nat) : Nat := B

/-
Main theorem statement:
Prove that the number of boys in the chess team is 14.
-/
theorem boys_count (B G : Nat) (h1 : B + G = TotalMembers) (h2 : GirlsAttendance G + BoysAttendance B = LastSessionAttendance) : B = 14 :=
by
  sorry

end boys_count_l213_213365


namespace c_difference_correct_l213_213725

noncomputable def find_c_difference (a b c : ℝ) (h1 : a + b + c = 3) (h2 : a^2 + b^2 + c^2 = 20) : ℝ :=
  2 * Real.sqrt 34

theorem c_difference_correct (a b c : ℝ) (h1 : a + b + c = 3) (h2 : a^2 + b^2 + c^2 = 20) :
  find_c_difference a b c h1 h2 = 2 * Real.sqrt 34 := 
sorry

end c_difference_correct_l213_213725


namespace usual_time_is_25_l213_213162

-- Definitions 
variables {S T : ℝ} (h1 : S * T = 5 / 4 * S * (T - 5))

-- Theorem statement
theorem usual_time_is_25 (h : S * T = 5 / 4 * S * (T - 5)) : T = 25 :=
by 
-- Using the assumption h, we'll derive that T = 25
sorry

end usual_time_is_25_l213_213162


namespace solve_for_x_l213_213466

theorem solve_for_x (x : ℝ) (h1 : x > 0) (h2 : 3 * x^2 + 8 * x - 16 = 0) : x = 4 / 3 :=
by
  sorry

end solve_for_x_l213_213466


namespace square_side_length_difference_l213_213972

theorem square_side_length_difference : 
  let side_A := Real.sqrt 25
  let side_B := Real.sqrt 81
  side_B - side_A = 4 :=
by
  sorry

end square_side_length_difference_l213_213972


namespace time_to_print_800_flyers_l213_213546

theorem time_to_print_800_flyers (x : ℝ) (h1 : 0 < x) :
  (1 / 6) + (1 / x) = 1 / 1.5 ↔ ∀ y : ℝ, 800 / 6 + 800 / x = 800 / 1.5 :=
by sorry

end time_to_print_800_flyers_l213_213546


namespace circle_diameter_l213_213509

theorem circle_diameter (A : ℝ) (π : ℝ) (r : ℝ) (d : ℝ) (h1 : A = 64 * π) (h2 : A = π * r^2) (h3 : d = 2 * r) :
  d = 16 :=
by
  sorry

end circle_diameter_l213_213509


namespace orange_preference_percentage_l213_213817

theorem orange_preference_percentage 
  (red blue green yellow purple orange : ℕ)
  (total : ℕ)
  (h_red : red = 75)
  (h_blue : blue = 80)
  (h_green : green = 50)
  (h_yellow : yellow = 45)
  (h_purple : purple = 60)
  (h_orange : orange = 55)
  (h_total : total = red + blue + green + yellow + purple + orange) :
  (orange * 100) / total = 15 :=
by
sorry

end orange_preference_percentage_l213_213817


namespace heidi_and_liam_paint_in_15_minutes_l213_213500

-- Definitions
def Heidi_rate : ℚ := 1 / 60
def Liam_rate : ℚ := 1 / 90
def combined_rate : ℚ := Heidi_rate + Liam_rate
def painting_time : ℚ := 15

-- Theorem to Prove
theorem heidi_and_liam_paint_in_15_minutes : painting_time * combined_rate = 5 / 12 := by
  sorry

end heidi_and_liam_paint_in_15_minutes_l213_213500


namespace time_taken_by_C_l213_213028

theorem time_taken_by_C (days_A B C : ℕ) (work_done_A work_done_B work_done_C : ℚ) 
  (h1 : days_A = 40) (h2 : work_done_A = 10 * (1/40)) 
  (h3 : days_B = 40) (h4 : work_done_B = 10 * (1/40)) 
  (h5 : work_done_C = 1/2)
  (h6 : 10 * work_done_C = 1/2) :
  (10 * 2) = 20 := 
sorry

end time_taken_by_C_l213_213028


namespace min_value_of_reciprocal_sum_l213_213135

-- Define the problem
theorem min_value_of_reciprocal_sum (a b : ℝ) (h1 : a > 0) (h2 : b > 0) 
  (h3 : ∃ x y : ℝ, (x^2 + y^2 + 2 * x - 4 * y + 1 = 0) ∧ (2 * a * x - b * y + 2 = 0)):
  ∃ (m : ℝ), m = 4 ∧ (1 / a + 1 / b) ≥ m :=
by
  sorry

end min_value_of_reciprocal_sum_l213_213135


namespace cyclist_speed_north_l213_213770

theorem cyclist_speed_north (v : ℝ) :
  (∀ d t : ℝ, d = 50 ∧ t = 1 ∧ 40 * t + v * t = d) → v = 10 :=
by
  sorry

end cyclist_speed_north_l213_213770


namespace next_unique_digits_date_l213_213008

-- Define the conditions
def is_after (d1 d2 : String) : Prop := sorry -- Placeholder, needs a date comparison function
def has_8_unique_digits (date : String) : Prop := sorry -- Placeholder, needs a function to check unique digits

-- Specify the problem and assertion
theorem next_unique_digits_date :
  ∀ date : String, is_after date "11.08.1999" → has_8_unique_digits date → date = "17.06.2345" :=
by
  sorry

end next_unique_digits_date_l213_213008


namespace inequality_of_sum_of_squares_l213_213098

theorem inequality_of_sum_of_squares (a b c : ℝ) (h : a * b + b * c + a * c = 1) : (a + b + c) ^ 2 ≥ 3 :=
sorry

end inequality_of_sum_of_squares_l213_213098


namespace function_range_ge_4_l213_213616

variable {x : ℝ}

theorem function_range_ge_4 (h : x > 0) : 2 * x + 2 * x⁻¹ ≥ 4 :=
sorry

end function_range_ge_4_l213_213616


namespace solve_equation_l213_213302

theorem solve_equation :
  ∀ x : ℝ, 
    (1 / (x + 8) + 1 / (x + 5) = 1 / (x + 11) + 1 / (x + 4)) ↔ 
      (x = (-3 + Real.sqrt 37) / 2 ∨ x = (-3 - Real.sqrt 37) / 2) := 
by
  intro x
  sorry

end solve_equation_l213_213302


namespace geometric_mean_45_80_l213_213025

theorem geometric_mean_45_80 : ∃ x : ℝ, x^2 = 45 * 80 ∧ (x = 60 ∨ x = -60) := 
by 
  sorry

end geometric_mean_45_80_l213_213025


namespace angle_C_in_triangle_ABC_l213_213520

noncomputable def find_angle_C (A B C : ℝ) (h1 : 3 * Real.sin A + 4 * Real.cos B = 6) (h2 : 4 * Real.sin B + 3 * Real.cos A = 1) (h3 : A + B + C = Real.pi) : Prop :=
  C = Real.pi / 6

theorem angle_C_in_triangle_ABC (A B C : ℝ) (h1 : 3 * Real.sin A + 4 * Real.cos B = 6) (h2 : 4 * Real.sin B + 3 * Real.cos A = 1) (h3 : A + B + C = Real.pi) : find_angle_C A B C h1 h2 h3 :=
by
  -- proof omitted
  sorry

end angle_C_in_triangle_ABC_l213_213520


namespace count_yellow_balls_l213_213864

theorem count_yellow_balls (total white green yellow red purple : ℕ) (prob : ℚ)
  (h_total : total = 100)
  (h_white : white = 50)
  (h_green : green = 30)
  (h_red : red = 9)
  (h_purple : purple = 3)
  (h_prob : prob = 0.88) :
  yellow = 8 :=
by
  -- The proof will be here
  sorry

end count_yellow_balls_l213_213864


namespace maximum_ab_l213_213134

open Real

theorem maximum_ab (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 6 * a + 5 * b = 75) :
  ab ≤ 46.875 :=
by
  -- proof goes here
  sorry

end maximum_ab_l213_213134


namespace most_stable_yield_l213_213125

theorem most_stable_yield (S_A S_B S_C S_D : ℝ)
  (h₁ : S_A = 3.6)
  (h₂ : S_B = 2.89)
  (h₃ : S_C = 13.4)
  (h₄ : S_D = 20.14) : 
  S_B < S_A ∧ S_B < S_C ∧ S_B < S_D :=
by {
  sorry -- Proof skipped as per instructions
}

end most_stable_yield_l213_213125


namespace ice_cream_ordering_ways_l213_213021

-- Define the possible choices for each category.
def cone_choices : Nat := 2
def scoop_choices : Nat := 1 + 10 + 20  -- Total choices for 1, 2, and 3 scoops.
def topping_choices : Nat := 1 + 4 + 6  -- Total choices for no topping, 1 topping, and 2 toppings.

-- Theorem to state the number of ways ice cream can be ordered.
theorem ice_cream_ordering_ways : cone_choices * scoop_choices * topping_choices = 748 := by
  let calc_cone := cone_choices  -- Number of cone choices.
  let calc_scoop := scoop_choices  -- Number of scoop combinations.
  let calc_topping := topping_choices  -- Number of topping combinations.
  have h1 : calc_cone * calc_scoop * calc_topping = 748 := sorry  -- Calculation hint.
  exact h1

end ice_cream_ordering_ways_l213_213021


namespace no_integer_solutions_for_system_l213_213523

theorem no_integer_solutions_for_system :
  ∀ (y z : ℤ),
    (2 * y^2 - 2 * y * z - z^2 = 15) ∧ 
    (6 * y * z + 2 * z^2 = 60) ∧ 
    (y^2 + 8 * z^2 = 90) 
    → False :=
by 
  intro y z
  simp
  sorry

end no_integer_solutions_for_system_l213_213523


namespace triangle_inequality_l213_213875

theorem triangle_inequality (a b c S : ℝ)
  (h : a ≠ b ∧ b ≠ c ∧ c ≠ a)   -- a, b, c are sides of a non-isosceles triangle
  (S_def : S = Real.sqrt ((a + b + c) / 2 * ((a + b + c) / 2 - a) * ((a + b + c) / 2 - b) * ((a + b + c) / 2 - c))) :
  (a^3) / ((a - b) * (a - c)) + (b^3) / ((b - c) * (b - a)) + (c^3) / ((c - a) * (c - b)) > 2 * 3^(3/4) * S :=
by
  sorry

end triangle_inequality_l213_213875


namespace height_of_E_l213_213184

variable {h_E h_F h_G h_H : ℝ}

theorem height_of_E (h1 : h_E + h_F + h_G + h_H = 2 * (h_E + h_F))
                    (h2 : (h_E + h_F) / 2 = (h_E + h_G) / 2 - 4)
                    (h3 : h_H = h_E - 10)
                    (h4 : h_F + h_G = 288) :
  h_E = 139 :=
by
  sorry

end height_of_E_l213_213184


namespace area_of_triangle_ABC_l213_213814

theorem area_of_triangle_ABC (A B C : ℝ) (a b c : ℝ) 
  (h1 : b = 2) (h2 : c = 3) (h3 : C = 2 * B): 
  ∃ S : ℝ, S = 1/2 * b * c * (Real.sin A) ∧ S = 15 * (Real.sqrt 7) / 16 :=
by
  sorry

end area_of_triangle_ABC_l213_213814


namespace trees_died_in_typhoon_l213_213694

-- Define the total number of trees, survived trees, and died trees
def total_trees : ℕ := 14

def survived_trees (S : ℕ) : ℕ := S

def died_trees (S : ℕ) : ℕ := S + 4

-- The Lean statement that formalizes the proof problem
theorem trees_died_in_typhoon : ∃ S : ℕ, survived_trees S + died_trees S = total_trees ∧ died_trees S = 9 :=
by
  -- Provide a placeholder for the proof
  sorry

end trees_died_in_typhoon_l213_213694


namespace correct_option_for_ruler_length_l213_213855

theorem correct_option_for_ruler_length (A B C D : String) (correct_answer : String) : 
  A = "two times as longer as" ∧ 
  B = "twice the length of" ∧ 
  C = "three times longer of" ∧ 
  D = "twice long than" ∧ 
  correct_answer = B := 
by
  sorry

end correct_option_for_ruler_length_l213_213855


namespace find_a_cubed_minus_b_cubed_l213_213666

theorem find_a_cubed_minus_b_cubed (a b : ℝ) (h1 : a - b = 6) (h2 : a^2 + b^2 = 66) : a^3 - b^3 = 486 := 
by 
  sorry

end find_a_cubed_minus_b_cubed_l213_213666


namespace maximum_area_of_rectangular_farm_l213_213147

theorem maximum_area_of_rectangular_farm :
  ∃ l w : ℕ, 2 * (l + w) = 160 ∧ l * w = 1600 :=
by
  sorry

end maximum_area_of_rectangular_farm_l213_213147


namespace minimum_dot_product_l213_213728

noncomputable def ellipse (x y : ℝ) : Prop := (x^2 / 36) + (y^2 / 9) = 1

def K : (ℝ × ℝ) := (2, 0)

def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

def vector_sub (u v : ℝ × ℝ) : ℝ × ℝ := (u.1 - v.1, u.2 - v.2)

theorem minimum_dot_product (M N : ℝ × ℝ) (hM : ellipse M.1 M.2) (hN : ellipse N.1 N.2) (h : dot_product (vector_sub M K) (vector_sub N K) = 0) :
  ∃ α β : ℝ, 0 ≤ α ∧ α < 2 * Real.pi ∧ 0 ≤ β ∧ β < 2 * Real.pi ∧ M = (6 * Real.cos α, 3 * Real.sin α) ∧ N = (6 * Real.cos β, 3 * Real.sin β) ∧
  (∃ C : ℝ, C = 23 / 3 ∧ ∀ M N, ellipse M.1 M.2 → ellipse N.1 N.2 → dot_product (vector_sub M K) (vector_sub N K) = 0 → dot_product (vector_sub M K) (vector_sub (vector_sub M N) K) >= C) :=
sorry

end minimum_dot_product_l213_213728


namespace find_certain_number_l213_213467

def certain_number (x : ℤ) : Prop := x - 9 = 5

theorem find_certain_number (x : ℤ) (h : certain_number x) : x = 14 :=
by
  sorry

end find_certain_number_l213_213467


namespace regular_polygon_exterior_angle_l213_213937

theorem regular_polygon_exterior_angle (n : ℕ) (h : 1 ≤ n) :
  (360 : ℝ) / (n : ℝ) = 60 → n = 6 :=
by
  intro h1
  sorry

end regular_polygon_exterior_angle_l213_213937


namespace dividend_value_l213_213141

def dividend (divisor quotient remainder : ℝ) := (divisor * quotient) + remainder

theorem dividend_value :
  dividend 35.8 21.65 11.3 = 786.47 :=
by
  sorry

end dividend_value_l213_213141


namespace distinct_license_plates_count_l213_213574

def num_digit_choices : Nat := 10
def num_letter_choices : Nat := 26
def num_digits : Nat := 5
def num_letters : Nat := 3

theorem distinct_license_plates_count :
  (num_digit_choices ^ num_digits) * (num_letter_choices ^ num_letters) = 1757600000 := 
sorry

end distinct_license_plates_count_l213_213574


namespace ham_block_cut_mass_distribution_l213_213498

theorem ham_block_cut_mass_distribution
  (length width height : ℝ) (mass : ℝ)
  (parallelogram_side1 parallelogram_side2 : ℝ)
  (condition1 : length = 12) 
  (condition2 : width = 12) 
  (condition3 : height = 35)
  (condition4 : mass = 5)
  (condition5 : parallelogram_side1 = 15) 
  (condition6 : parallelogram_side2 = 20) :
  ∃ (mass_piece1 mass_piece2 : ℝ),
    mass_piece1 = 1.7857 ∧ mass_piece2 = 3.2143 :=
by
  sorry

end ham_block_cut_mass_distribution_l213_213498


namespace product_of_consecutive_integers_l213_213506

theorem product_of_consecutive_integers (n : ℤ) :
  n * (n + 1) * (n + 2) = (n + 1)^3 - (n + 1) :=
by
  sorry

end product_of_consecutive_integers_l213_213506


namespace total_letters_sent_l213_213378

def letters_January : ℕ := 6
def letters_February : ℕ := 9
def letters_March : ℕ := 3 * letters_January

theorem total_letters_sent : letters_January + letters_February + letters_March = 33 :=
by
  -- This is where the proof would go
  sorry

end total_letters_sent_l213_213378


namespace chrom_replication_not_in_prophase_I_l213_213692

-- Definitions for the conditions
def chrom_replication (stage : String) : Prop := 
  stage = "Interphase"

def chrom_shortening_thickening (stage : String) : Prop := 
  stage = "Prophase I"

def pairing_homologous_chromosomes (stage : String) : Prop := 
  stage = "Prophase I"

def crossing_over (stage : String) : Prop :=
  stage = "Prophase I"

-- Stating the theorem
theorem chrom_replication_not_in_prophase_I :
  chrom_replication "Interphase" ∧ 
  chrom_shortening_thickening "Prophase I" ∧ 
  pairing_homologous_chromosomes "Prophase I" ∧ 
  crossing_over "Prophase I" → 
  ¬ chrom_replication "Prophase I" := 
by
  sorry

end chrom_replication_not_in_prophase_I_l213_213692


namespace probability_A_mc_and_B_tf_probability_at_least_one_mc_l213_213662

section ProbabilityQuiz

variable (total_questions : ℕ) (mc_questions : ℕ) (tf_questions : ℕ)

def prob_A_mc_and_B_tf (total_questions mc_questions tf_questions : ℕ) : ℚ :=
  (mc_questions * tf_questions : ℚ) / (total_questions * (total_questions - 1))

def prob_at_least_one_mc (total_questions mc_questions tf_questions : ℕ) : ℚ :=
  1 - ((tf_questions * (tf_questions - 1) : ℚ) / (total_questions * (total_questions - 1)))

theorem probability_A_mc_and_B_tf :
  prob_A_mc_and_B_tf 10 6 4 = 4 / 15 := by
  sorry

theorem probability_at_least_one_mc :
  prob_at_least_one_mc 10 6 4 = 13 / 15 := by
  sorry

end ProbabilityQuiz

end probability_A_mc_and_B_tf_probability_at_least_one_mc_l213_213662


namespace velocity_at_2_l213_213928

variable (t : ℝ) (s : ℝ)

noncomputable def displacement (t : ℝ) : ℝ := t^2 + 3 / t

noncomputable def velocity (t : ℝ) : ℝ := (deriv displacement) t

theorem velocity_at_2 : velocity t = 2 * 2 - (3 / 4) := by
  sorry

end velocity_at_2_l213_213928


namespace number_of_teams_in_league_l213_213479

theorem number_of_teams_in_league (n : ℕ) :
  (6 * n * (n - 1)) / 2 = 396 ↔ n = 12 :=
by
  sorry

end number_of_teams_in_league_l213_213479


namespace janet_wait_time_l213_213683

theorem janet_wait_time 
  (janet_speed : ℝ)
  (sister_speed : ℝ)
  (lake_width : ℝ)
  (h1 : janet_speed = 30)
  (h2 : sister_speed = 12)
  (h3 : lake_width = 60) 
  :
  60 / 12 - 60 / 30 = 3 :=
by
  sorry

end janet_wait_time_l213_213683


namespace unique_polynomial_l213_213060

-- Define the conditions
def valid_polynomial (P : ℝ → ℝ) : Prop :=
  ∃ (p : Polynomial ℝ), Polynomial.degree p > 0 ∧ ∀ (z : ℝ), z ≠ 0 → P z = Polynomial.eval z p

-- The main theorem
theorem unique_polynomial (P : ℝ → ℝ) (hP : valid_polynomial P) :
  (∀ (z : ℝ), z ≠ 0 → P z ≠ 0 → P (1/z) ≠ 0 → 
  1 / P z + 1 / P (1 / z) = z + 1 / z) → ∀ x, P x = x :=
by
  sorry

end unique_polynomial_l213_213060


namespace count_two_digit_numbers_l213_213176

theorem count_two_digit_numbers : (99 - 10 + 1) = 90 := by
  sorry

end count_two_digit_numbers_l213_213176


namespace find_first_number_l213_213077

def is_lcm (a b l : ℕ) : Prop := l = Nat.lcm a b

theorem find_first_number :
  ∃ (a b : ℕ), (5 * b) = a ∧ (4 * b) = b ∧ is_lcm a b 80 ∧ a = 20 :=
by
  sorry

end find_first_number_l213_213077


namespace average_age_of_three_l213_213626

theorem average_age_of_three (Tonya_age John_age Mary_age : ℕ)
  (h1 : John_age = 2 * Mary_age)
  (h2 : Tonya_age = 2 * John_age)
  (h3 : Tonya_age = 60) :
  (Tonya_age + John_age + Mary_age) / 3 = 35 := by
  sorry

end average_age_of_three_l213_213626


namespace simplify_and_evaluate_expression_l213_213349

theorem simplify_and_evaluate_expression (a : ℚ) (h : a = -3/2) :
  (a + 2 - 5/(a - 2)) / ((2 * a^2 - 6 * a) / (a - 2)) = -1/2 :=
by
  rw [h]
  sorry

end simplify_and_evaluate_expression_l213_213349


namespace spending_spring_months_l213_213903

theorem spending_spring_months (spend_end_March spend_end_June : ℝ)
  (h1 : spend_end_March = 1) (h2 : spend_end_June = 4) :
  (spend_end_June - spend_end_March) = 3 :=
by
  rw [h1, h2]
  norm_num

end spending_spring_months_l213_213903


namespace goose_eggs_count_l213_213808

theorem goose_eggs_count (E : ℕ) 
  (hatch_ratio : ℝ := 1/4)
  (survival_first_month_ratio : ℝ := 4/5)
  (survival_first_year_ratio : ℝ := 3/5)
  (survived_first_year : ℕ := 120) :
  ((survival_first_year_ratio * (survival_first_month_ratio * hatch_ratio * E)) = survived_first_year) → E = 1000 :=
by
  intro h
  sorry

end goose_eggs_count_l213_213808


namespace find_f_value_l213_213431

noncomputable def f (x : ℝ) : ℝ := x + Real.sin x + 1

theorem find_f_value : f 2019 + f (-2019) = 2 :=
by
  sorry

end find_f_value_l213_213431


namespace trigonometric_identity_l213_213724

open Real

theorem trigonometric_identity (α : ℝ) (h1 : cos α = -4 / 5) (h2 : π < α ∧ α < (3 * π / 2)) :
    (1 + tan (α / 2)) / (1 - tan (α / 2)) = -1 / 2 := by
  sorry

end trigonometric_identity_l213_213724


namespace ten_thousand_points_length_l213_213061

theorem ten_thousand_points_length (a b : ℝ) (d : ℝ) 
  (h1 : d = a / 99) 
  (h2 : b = 9999 * d) : b = 101 * a := by
  sorry

end ten_thousand_points_length_l213_213061


namespace problem_1_l213_213188

theorem problem_1 (x : ℝ) (h : x^2 = 2) : (3 * x)^2 - 4 * (x^3)^2 = -14 :=
by {
  sorry
}

end problem_1_l213_213188


namespace larger_triangle_perimeter_l213_213329

-- Given conditions
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

def is_isosceles (t : Triangle) : Prop :=
  t.a = t.b ∨ t.a = t.c ∨ t.b = t.c

def similar (t1 t2 : Triangle) (k : ℝ) : Prop :=
  t1.a / t2.a = k ∧ t1.b / t2.b = k ∧ t1.c / t2.c = k

def perimeter (t : Triangle) : ℝ :=
  t.a + t.b + t.c

-- Define specific triangles based on the problem
def smaller_triangle : Triangle := {a := 12, b := 12, c := 15}
def larger_triangle_ratio : ℝ := 2
def larger_triangle : Triangle := {a := 12 * larger_triangle_ratio, b := 12 * larger_triangle_ratio, c := 15 * larger_triangle_ratio}

-- Main theorem statement
theorem larger_triangle_perimeter : perimeter larger_triangle = 78 :=
by 
  sorry

end larger_triangle_perimeter_l213_213329


namespace probability_not_all_dice_show_different_l213_213581

noncomputable def probability_not_all_same : ℚ :=
  let total_outcomes := 8^5
  let same_number_outcomes := 8
  (total_outcomes - same_number_outcomes) / total_outcomes

theorem probability_not_all_dice_show_different : 
  probability_not_all_same = 4095 / 4096 := by
  sorry

end probability_not_all_dice_show_different_l213_213581


namespace meal_preppers_activity_setters_count_l213_213686

-- Definitions for the problem conditions
def num_friends : ℕ := 6
def num_meal_preppers : ℕ := 3

-- Statement of the theorem
theorem meal_preppers_activity_setters_count :
  (num_friends.choose num_meal_preppers) = 20 :=
by
  -- Proof would go here
  sorry

end meal_preppers_activity_setters_count_l213_213686


namespace find_A_when_B_is_largest_l213_213086

theorem find_A_when_B_is_largest :
  ∃ A : ℕ, ∃ B : ℕ, A = 17 * 25 + B ∧ B < 17 ∧ B = 16 ∧ A = 441 :=
by
  sorry

end find_A_when_B_is_largest_l213_213086


namespace fly_distance_to_ceiling_l213_213368

theorem fly_distance_to_ceiling :
  ∀ (x y z : ℝ), 
  (x = 3) → 
  (y = 4) → 
  (z * z + 25 = 49) →
  z = 2 * Real.sqrt 6 :=
by
  sorry

end fly_distance_to_ceiling_l213_213368


namespace perfect_square_k_l213_213878

theorem perfect_square_k (a b k : ℝ) (h : ∃ c : ℝ, a^2 + 2*(k-3)*a*b + 9*b^2 = (a + c*b)^2) : 
  k = 6 ∨ k = 0 := 
sorry

end perfect_square_k_l213_213878


namespace isosceles_triangle_side_l213_213089

theorem isosceles_triangle_side (a : ℝ) : 
  (10 - a = 7 ∨ 10 - a = 6) ↔ (a = 3 ∨ a = 4) := 
by sorry

end isosceles_triangle_side_l213_213089


namespace trig_expression_value_l213_213970

theorem trig_expression_value (x : ℝ) (h : Real.tan x = 1/2) :
  (2 * Real.sin x + 3 * Real.cos x) / (Real.cos x - Real.sin x) = 8 :=
by
  sorry

end trig_expression_value_l213_213970


namespace fresh_grapes_water_content_l213_213456

theorem fresh_grapes_water_content:
  ∀ (P : ℝ), 
  (∀ (x y : ℝ), P = x) → 
  (∃ (fresh_grapes dry_grapes : ℝ), fresh_grapes = 25 ∧ dry_grapes = 3.125 ∧ 
  (100 - P) / 100 * fresh_grapes = 0.8 * dry_grapes ) → 
  P = 90 :=
by 
  sorry

end fresh_grapes_water_content_l213_213456


namespace initial_pups_per_mouse_l213_213609

-- Definitions from the problem's conditions
def initial_mice : ℕ := 8
def stress_factor : ℕ := 2
def second_round_pups : ℕ := 6
def total_mice : ℕ := 280

-- Define a variable for the initial number of pups each mouse had
variable (P : ℕ)

-- Lean statement to prove the number of initial pups per mouse
theorem initial_pups_per_mouse (P : ℕ) (initial_mice stress_factor second_round_pups total_mice : ℕ) :
  total_mice = initial_mice + initial_mice * P + (initial_mice + initial_mice * P) * second_round_pups - stress_factor * (initial_mice + initial_mice * P) → 
  P = 6 := 
by
  sorry

end initial_pups_per_mouse_l213_213609


namespace total_vowels_written_l213_213267

-- Define the vowels and the condition
def vowels : List Char := ['A', 'E', 'I', 'O', 'U']
def num_vowels : Nat := vowels.length
def times_written : Nat := 2

-- Assert the total number of vowels written
theorem total_vowels_written : (num_vowels * times_written) = 10 := by
  sorry

end total_vowels_written_l213_213267


namespace probability_math_majors_consecutive_l213_213362

noncomputable def total_ways := Nat.choose 11 4 -- Number of ways to choose 5 persons out of 12 (fixing one)
noncomputable def favorable_ways := 12         -- Number of ways to arrange 5 math majors consecutively around a round table

theorem probability_math_majors_consecutive :
  (favorable_ways : ℚ) / total_ways = 2 / 55 :=
by
  sorry

end probability_math_majors_consecutive_l213_213362


namespace group_made_l213_213881

-- Definitions based on the problem's conditions
def teachers_made : Nat := 28
def total_products : Nat := 93

-- Theorem to prove that the group made 65 recycled materials
theorem group_made : total_products - teachers_made = 65 := by
  sorry

end group_made_l213_213881


namespace Mo_tea_cups_l213_213920

theorem Mo_tea_cups (n t : ℕ) 
  (h1 : 2 * n + 5 * t = 36)
  (h2 : 5 * t = 2 * n + 14) : 
  t = 5 :=
by
  sorry

end Mo_tea_cups_l213_213920


namespace order_abc_l213_213325

noncomputable def a : ℝ := Real.log 0.8 / Real.log 0.7
noncomputable def b : ℝ := Real.log 0.9 / Real.log 1.1
noncomputable def c : ℝ := Real.exp (0.9 * Real.log 1.1)

theorem order_abc : b < a ∧ a < c := by
  sorry

end order_abc_l213_213325


namespace number_of_male_students_l213_213227

noncomputable def avg_all : ℝ := 90
noncomputable def avg_male : ℝ := 84
noncomputable def avg_female : ℝ := 92
noncomputable def count_female : ℕ := 24

theorem number_of_male_students (M : ℕ) (T : ℕ) :
  avg_all * (M + count_female) = avg_male * M + avg_female * count_female →
  T = M + count_female →
  M = 8 :=
by
  intro h_avg h_count
  sorry

end number_of_male_students_l213_213227


namespace second_expression_l213_213790

variable (a b : ℕ)

theorem second_expression (h : 89 = ((2 * a + 16) + b) / 2) (ha : a = 34) : b = 94 :=
by
  sorry

end second_expression_l213_213790


namespace cucumber_new_weight_l213_213341

-- Definitions for the problem conditions
def initial_weight : ℝ := 100
def initial_water_percentage : ℝ := 0.99
def final_water_percentage : ℝ := 0.96
noncomputable def new_weight : ℝ := initial_weight * (1 - initial_water_percentage) / (1 - final_water_percentage)

-- The theorem stating the problem to be solved
theorem cucumber_new_weight : new_weight = 25 :=
by
  -- Skipping the proof for now
  sorry

end cucumber_new_weight_l213_213341


namespace smallest_ratio_is_three_l213_213320

theorem smallest_ratio_is_three (m n : ℕ) (a : ℕ) (h1 : 2^m + 1 = a * (2^n + 1)) (h2 : a > 1) : a = 3 :=
sorry

end smallest_ratio_is_three_l213_213320


namespace gear_rotations_l213_213418

-- Definitions from the conditions
def gearA_teeth : ℕ := 12
def gearB_teeth : ℕ := 54

-- The main problem: prove that gear A needs 9 rotations and gear B needs 2 rotations
theorem gear_rotations :
  ∃ x y : ℕ, 12 * x = 54 * y ∧ x = 9 ∧ y = 2 := by
  sorry

end gear_rotations_l213_213418


namespace find_point_P_l213_213607

def f (x : ℝ) : ℝ := x^4 - 2 * x

def tangent_line_perpendicular (x y : ℝ) : Prop :=
  (f x) = y ∧ (4 * x^3 - 2 = 2)

theorem find_point_P :
  ∃ (x y : ℝ), tangent_line_perpendicular x y ∧ x = 1 ∧ y = -1 :=
sorry

end find_point_P_l213_213607


namespace original_inhabitants_7200_l213_213585

noncomputable def original_inhabitants (X : ℝ) : Prop :=
  let initial_decrease := 0.9 * X
  let final_decrease := 0.75 * initial_decrease
  final_decrease = 4860

theorem original_inhabitants_7200 : ∃ X : ℝ, original_inhabitants X ∧ X = 7200 := by
  use 7200
  unfold original_inhabitants
  simp
  sorry

end original_inhabitants_7200_l213_213585


namespace volume_of_tetrahedron_equiv_l213_213054

noncomputable def volume_tetrahedron (D1 D2 D3 : ℝ) 
  (h1 : D1 = 24) (h2 : D3 = 20) (h3 : D2 = 16) : ℝ :=
  30 * Real.sqrt 6

theorem volume_of_tetrahedron_equiv (D1 D2 D3 : ℝ) 
  (h1 : D1 = 24) (h2 : D3 = 20) (h3 : D2 = 16) :
  volume_tetrahedron D1 D2 D3 h1 h2 h3 = 30 * Real.sqrt 6 :=
  sorry

end volume_of_tetrahedron_equiv_l213_213054


namespace area_of_right_triangle_l213_213003

theorem area_of_right_triangle (h : ℝ) 
  (a b : ℝ) 
  (h_a_triple : b = 3 * a)
  (h_hypotenuse : h ^ 2 = a ^ 2 + b ^ 2) : 
  (1 / 2) * a * b = (3 * h ^ 2) / 20 :=
by
  sorry

end area_of_right_triangle_l213_213003


namespace expression_simplified_l213_213877

theorem expression_simplified (d : ℤ) (h : d ≠ 0) :
  let a := 24
  let b := 61
  let c := 96
  a + b + c = 181 ∧ 
  (15 * d ^ 2 + 7 * d + 15 + (3 * d + 9) ^ 2 = a * d ^ 2 + b * d + c) := by
{
  sorry
}

end expression_simplified_l213_213877


namespace sum_of_digits_is_15_l213_213276

theorem sum_of_digits_is_15
  (A B C D E : ℕ) 
  (h_distinct: A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧ B ≠ C ∧ B ≠ D ∧ B ≠ E ∧ C ≠ D ∧ C ≠ E ∧ D ≠ E)
  (h_digits: A < 10 ∧ B < 10 ∧ C < 10 ∧ D < 10 ∧ E < 10)
  (h_divisible_by_9: (A * 10000 + B * 1000 + C * 100 + D * 10 + E) % 9 = 0) 
  : A + B + C + D + E = 15 := 
sorry

end sum_of_digits_is_15_l213_213276


namespace algebraic_expression_value_l213_213731

theorem algebraic_expression_value (x : ℝ) (h : x^2 + 3*x - 1 = 0) :
  (x - 3)^2 - (2*x + 1)*(2*x - 1) - 3*x = 7 :=
sorry

end algebraic_expression_value_l213_213731


namespace minimum_sum_of_nine_consecutive_integers_l213_213576

-- We will define the consecutive sequence and the conditions as described.
structure ConsecutiveIntegers (a1 a2 a3 a4 a5 a6 a7 a8 a9 : ℕ) : Prop :=
(seq : a1 + 1 = a2 ∧ a2 + 1 = a3 ∧ a3 + 1 = a4 ∧ a4 + 1 = a5 ∧ a5 + 1 = a6 ∧ a6 + 1 = a7 ∧ a7 + 1 = a8 ∧ a8 + 1 = a9)
(sq_cond : ∃ k : ℕ, (a1 + a3 + a5 + a7 + a9) = k * k)
(cube_cond : ∃ l : ℕ, (a2 + a4 + a6 + a8) = l * l * l)

theorem minimum_sum_of_nine_consecutive_integers :
  ∃ a1 a2 a3 a4 a5 a6 a7 a8 a9 : ℕ,
  ConsecutiveIntegers a1 a2 a3 a4 a5 a6 a7 a8 a9 ∧ (a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 + a9 = 18000) :=
  sorry

end minimum_sum_of_nine_consecutive_integers_l213_213576


namespace right_triangle_area_l213_213734

theorem right_triangle_area
  (hypotenuse : ℝ) (leg1 : ℝ) (leg2 : ℝ)
  (hypotenuse_eq : hypotenuse = 13)
  (leg1_eq : leg1 = 5)
  (pythagorean_eq : hypotenuse^2 = leg1^2 + leg2^2) :
  (1 / 2) * leg1 * leg2 = 30 :=
by
  sorry

end right_triangle_area_l213_213734


namespace intersection_of_M_and_N_l213_213980

def M : Set ℝ := { x | |x + 1| ≤ 1}

def N : Set ℝ := {-1, 0, 1}

theorem intersection_of_M_and_N : M ∩ N = {-1, 0} :=
by
  sorry

end intersection_of_M_and_N_l213_213980


namespace shaded_area_of_larger_circle_l213_213956

theorem shaded_area_of_larger_circle (R r : ℝ) (A_larger A_smaller : ℝ)
  (hR : R = 9)
  (hr : r = 4.5)
  (hA_larger : A_larger = Real.pi * R^2)
  (hA_smaller : A_smaller = 3 * Real.pi * r^2) :
  A_larger - A_smaller = 20.25 * Real.pi := by
  sorry

end shaded_area_of_larger_circle_l213_213956


namespace factor_of_increase_l213_213364

-- Define the conditions
def interest_rate : ℝ := 0.25
def time_period : ℕ := 4

-- Define the principal amount as a variable
variable (P : ℝ)

-- Define the simple interest formula
def simple_interest (P : ℝ) (R : ℝ) (T : ℕ) : ℝ := P * R * (T : ℝ)

-- Define the total amount function
def total_amount (P : ℝ) (SI : ℝ) : ℝ := P + SI

-- The theorem that we need to prove: The factor by which the sum of money increases is 2
theorem factor_of_increase :
  total_amount P (simple_interest P interest_rate time_period) = 2 * P := by
  sorry

end factor_of_increase_l213_213364


namespace factor_expression_l213_213300

theorem factor_expression (a b : ℕ) (h_factor : (x - a) * (x - b) = x^2 - 18 * x + 72) (h_nonneg : 0 ≤ a ∧ 0 ≤ b) (h_order : a > b) : 4 * b - a = 27 := by
  sorry

end factor_expression_l213_213300


namespace coffee_vacation_days_l213_213919

theorem coffee_vacation_days 
  (pods_per_day : ℕ := 3)
  (pods_per_box : ℕ := 30)
  (box_cost : ℝ := 8.00)
  (total_spent : ℝ := 32) :
  (total_spent / box_cost) * pods_per_box / pods_per_day = 40 := 
by 
  sorry

end coffee_vacation_days_l213_213919


namespace greatest_C_inequality_l213_213105

theorem greatest_C_inequality (α x y z : ℝ) (hα_pos : 0 < α) (hx_pos : 0 < x) (hy_pos : 0 < y) (hz_pos : 0 < z) (h_xyz_sum : x * y + y * z + z * x = α) : 
  16 ≤ (1 + α / x^2) * (1 + α / y^2) * (1 + α / z^2) / (x / z + z / x + 2) :=
sorry

end greatest_C_inequality_l213_213105


namespace sum_of_x_and_y_l213_213774

-- Define integers x and y
variables (x y : ℤ)

-- Define conditions
def condition1 : Prop := x - y = 200
def condition2 : Prop := y = 250

-- Define the main statement
theorem sum_of_x_and_y (h1 : condition1 x y) (h2 : condition2 y) : x + y = 700 := 
by
  sorry

end sum_of_x_and_y_l213_213774


namespace wolf_nobel_laureates_l213_213232

/-- 31 scientists that attended a certain workshop were Wolf Prize laureates,
and some of them were also Nobel Prize laureates. Of the scientists who attended
that workshop and had not received the Wolf Prize, the number of scientists who had
received the Nobel Prize was 3 more than the number of scientists who had not received
the Nobel Prize. In total, 50 scientists attended that workshop, and 25 of them were
Nobel Prize laureates. Prove that the number of Wolf Prize laureates who were also
Nobel Prize laureates is 3. -/
theorem wolf_nobel_laureates (W N total W' N' W_N : ℕ)  
  (hW : W = 31) (hN : N = 25) (htotal : total = 50) 
  (hW' : W' = total - W) (hN' : N' = total - N) 
  (hcondition : N' - W' = 3) :
  W_N = N - W' :=
by
  sorry

end wolf_nobel_laureates_l213_213232


namespace min_value_frac_l213_213423

theorem min_value_frac (x y a b c d : ℝ) (hx : 0 < x) (hy : 0 < y)
  (harith : x + y = a + b) (hgeo : x * y = c * d) : (a + b) ^ 2 / (c * d) ≥ 4 := 
by sorry

end min_value_frac_l213_213423


namespace solve_r_l213_213606

variable (r : ℝ)

theorem solve_r : (r + 3) / (r - 2) = (r - 1) / (r + 1) → r = -1/7 := by
  sorry

end solve_r_l213_213606


namespace semicircle_perimeter_l213_213411

theorem semicircle_perimeter (r : ℝ) (π : ℝ) (h : 0 < π) (r_eq : r = 14):
  (14 * π + 28) = 14 * π + 28 :=
by
  sorry

end semicircle_perimeter_l213_213411


namespace average_age_of_group_l213_213744

theorem average_age_of_group :
  let n_graders := 40
  let n_parents := 50
  let n_teachers := 10
  let avg_age_graders := 12
  let avg_age_parents := 35
  let avg_age_teachers := 45
  let total_individuals := n_graders + n_parents + n_teachers
  let total_age := n_graders * avg_age_graders + n_parents * avg_age_parents + n_teachers * avg_age_teachers
  (total_age : ℚ) / total_individuals = 26.8 :=
by
  sorry

end average_age_of_group_l213_213744


namespace soaked_part_solution_l213_213708

theorem soaked_part_solution 
  (a b : ℝ) (c : ℝ) 
  (h : c * (2/3) * a * b = 2 * a^2 * b^3 + (1/3) * a^3 * b^2) :
  c = 3 * a * b^2 + (1/2) * a^2 * b :=
by
  sorry

end soaked_part_solution_l213_213708


namespace total_weight_of_beef_l213_213929

-- Define the conditions
def packages_weight := 4
def first_butcher_packages := 10
def second_butcher_packages := 7
def third_butcher_packages := 8

-- Define the total weight calculation
def total_weight := (first_butcher_packages * packages_weight) +
                    (second_butcher_packages * packages_weight) +
                    (third_butcher_packages * packages_weight)

-- The statement to prove
theorem total_weight_of_beef : total_weight = 100 := by
  -- proof goes here
  sorry

end total_weight_of_beef_l213_213929


namespace worker_surveys_per_week_l213_213584

theorem worker_surveys_per_week :
  let regular_rate := 30
  let cellphone_rate := regular_rate + 0.20 * regular_rate
  let surveys_with_cellphone := 50
  let earnings := 3300
  cellphone_rate = regular_rate + 0.20 * regular_rate →
  earnings = surveys_with_cellphone * cellphone_rate →
  regular_rate = 30 →
  surveys_with_cellphone = 50 →
  earnings = 3300 →
  surveys_with_cellphone = 50 := sorry

end worker_surveys_per_week_l213_213584


namespace sides_of_triangle_inequality_l213_213590

theorem sides_of_triangle_inequality (a b c : ℝ) (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) :
  a^2 * b * (a - b) + b^2 * c * (b - c) + c^2 * a * (c - a) ≥ 0 ∧ 
  (a^2 * b * (a - b) + b^2 * c * (b - c) + c^2 * a * (c - a) = 0 ↔ a = b ∧ b = c) :=
sorry

end sides_of_triangle_inequality_l213_213590


namespace decimal_representation_prime_has_zeros_l213_213092

theorem decimal_representation_prime_has_zeros (p : ℕ) [Fact (Nat.Prime p)] : 
  ∃ n : ℕ, n > 0 ∧ ∃ k : ℕ, 10^2002 ∣ p^n * 10^k :=
sorry

end decimal_representation_prime_has_zeros_l213_213092


namespace kelly_held_longest_l213_213460

variable (K : ℕ)

-- Conditions
def Brittany_held (K : ℕ) : ℕ := K - 20
def Buffy_held : ℕ := 120

-- Theorem to prove
theorem kelly_held_longest (h : K > Buffy_held) : K > 120 :=
by sorry

end kelly_held_longest_l213_213460


namespace solve_for_x_l213_213594

-- Define the variables and conditions
variable (x : ℚ)

-- Define the given condition
def condition : Prop := (x + 4)/(x - 3) = (x - 2)/(x + 2)

-- State the theorem that x = -2/11 is a solution to the condition
theorem solve_for_x (h : condition x) : x = -2 / 11 := by
  sorry

end solve_for_x_l213_213594


namespace non_empty_solution_set_range_l213_213204

theorem non_empty_solution_set_range {a : ℝ} 
  (h : ∃ x : ℝ, |x + 2| + |x - 3| ≤ a) : 
  a ≥ 5 :=
sorry

end non_empty_solution_set_range_l213_213204


namespace hens_count_l213_213486

theorem hens_count (H R : ℕ) (h₁ : H = 9 * R - 5) (h₂ : H + R = 75) : H = 67 :=
by {
  sorry
}

end hens_count_l213_213486


namespace oranges_count_l213_213394

noncomputable def initial_oranges (O : ℕ) : Prop :=
  let apples := 14
  let blueberries := 6
  let remaining_fruits := 26
  13 + (O - 1) + 5 = remaining_fruits

theorem oranges_count (O : ℕ) (h : initial_oranges O) : O = 9 :=
by
  have eq : 13 + (O - 1) + 5 = 26 := h
  -- Simplify the equation to find O
  sorry

end oranges_count_l213_213394


namespace minor_premise_l213_213116

-- Definitions
def Rectangle : Type := sorry
def Square : Type := sorry
def Parallelogram : Type := sorry

axiom rectangle_is_parallelogram : Rectangle → Parallelogram
axiom square_is_rectangle : Square → Rectangle
axiom square_is_parallelogram : Square → Parallelogram

-- Problem statement
theorem minor_premise : ∀ (S : Square), ∃ (R : Rectangle), square_is_rectangle S = R :=
by
  sorry

end minor_premise_l213_213116


namespace positive_difference_perimeters_l213_213235

theorem positive_difference_perimeters (length width : ℝ) 
    (cut_rectangles : ℕ) 
    (H : length = 6 ∧ width = 9 ∧ cut_rectangles = 4) : 
    ∃ (p1 p2 : ℝ), (p1 = 24 ∧ p2 = 15) ∧ (abs (p1 - p2) = 9) :=
by
  sorry

end positive_difference_perimeters_l213_213235


namespace book_cost_l213_213850

theorem book_cost (x : ℝ) 
  (h1 : Vasya_has = x - 150)
  (h2 : Tolya_has = x - 200)
  (h3 : (x - 150) + (x - 200) / 2 = x + 100) : x = 700 :=
sorry

end book_cost_l213_213850


namespace tank_empty_time_l213_213636

theorem tank_empty_time 
  (time_to_empty_leak : ℝ) 
  (inlet_rate_per_minute : ℝ) 
  (tank_volume : ℝ) 
  (net_time_to_empty : ℝ) : 
  time_to_empty_leak = 7 → 
  inlet_rate_per_minute = 6 → 
  tank_volume = 6048.000000000001 → 
  net_time_to_empty = 12 :=
by
  intros h1 h2 h3
  sorry

end tank_empty_time_l213_213636


namespace bonus_percentage_correct_l213_213711

/-
Tom serves 10 customers per hour and works for 8 hours, earning 16 bonus points.
We need to find the percentage of bonus points per customer served.
-/

def customers_per_hour : ℕ := 10
def hours_worked : ℕ := 8
def total_bonus_points : ℕ := 16

def total_customers_served : ℕ := customers_per_hour * hours_worked
def bonus_percentage : ℕ := (total_bonus_points * 100) / total_customers_served

theorem bonus_percentage_correct : bonus_percentage = 20 := by
  sorry

end bonus_percentage_correct_l213_213711


namespace boat_travel_distance_downstream_l213_213221

-- Define the given conditions
def speed_boat_still : ℝ := 22
def speed_stream : ℝ := 5
def time_downstream : ℝ := 5

-- Define the effective speed and the computed distance
def effective_speed_downstream : ℝ := speed_boat_still + speed_stream
def distance_traveled_downstream : ℝ := effective_speed_downstream * time_downstream

-- State the proof problem that distance_traveled_downstream is 135 km
theorem boat_travel_distance_downstream :
  distance_traveled_downstream = 135 :=
by
  -- The proof will go here
  sorry

end boat_travel_distance_downstream_l213_213221


namespace ratio_of_triangle_areas_l213_213289

theorem ratio_of_triangle_areas (kx ky k : ℝ)
(n m : ℕ) (h1 : n > 0) (h2 : m > 0) :
  let A := (1 / 2) * (ky / m) * (kx / 2)
  let B := (1 / 2) * (kx / n) * (ky / 2)
  (A / B) = (n / m) :=
by
  sorry

end ratio_of_triangle_areas_l213_213289


namespace average_is_20_l213_213959

-- Define the numbers and the variable n
def a := 3
def b := 16
def c := 33
def n := 27
def d := n + 1

-- Define the sum of the numbers
def sum := a + b + c + d

-- Define the average as sum divided by 4
def average := sum / 4

-- Prove that the average is 20
theorem average_is_20 : average = 20 := by
  sorry

end average_is_20_l213_213959


namespace solve_for_x_l213_213533

theorem solve_for_x (x : ℝ) (h : 3 * x - 5 * x + 6 * x = 150) : x = 37.5 :=
by
  sorry

end solve_for_x_l213_213533


namespace length_CD_l213_213525

-- Given data
variables {A B C D : Type} [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited D]
variables (AB BC : ℝ)

noncomputable def triangle_ABC : Prop :=
  AB = 5 ∧ BC = 7 ∧ ∃ (angle_ABC : ℝ), angle_ABC = 90

-- The target condition to prove
theorem length_CD {CD : ℝ} (h : triangle_ABC AB BC) : CD = 7 :=
by {
  -- proof would be here
  sorry
}

end length_CD_l213_213525


namespace isosceles_triangle_perimeter_l213_213951

-- Define the conditions
def equilateral_triangle_side : ℕ := 15
def isosceles_triangle_side : ℕ := 15
def isosceles_triangle_base : ℕ := 10

-- Define the theorem to prove the perimeter of the isosceles triangle
theorem isosceles_triangle_perimeter : 
  (2 * isosceles_triangle_side + isosceles_triangle_base = 40) :=
by
  -- Placeholder for the actual proof
  sorry

end isosceles_triangle_perimeter_l213_213951


namespace original_price_l213_213343

theorem original_price (price_paid original_price : ℝ) 
  (h₁ : price_paid = 5) 
  (h₂ : price_paid = original_price / 10) : 
  original_price = 50 := by
  sorry

end original_price_l213_213343


namespace min_value_of_a_l213_213587

theorem min_value_of_a : 
  ∃ (a : ℤ), ∃ x y : ℤ, x ≠ y ∧ |x| ≤ 10 ∧ (x - y^2 = a) ∧ (y - x^2 = a) ∧ a = -111 :=
by
  sorry

end min_value_of_a_l213_213587


namespace green_apples_count_l213_213046

-- Definitions for the conditions
def total_apples : ℕ := 9
def red_apples : ℕ := 7

-- Theorem stating the number of green apples
theorem green_apples_count : total_apples - red_apples = 2 := by
  sorry

end green_apples_count_l213_213046


namespace fold_point_area_sum_l213_213610

noncomputable def fold_point_area (AB AC : ℝ) (angle_B : ℝ) : ℝ :=
  let BC := Real.sqrt (AB ^ 2 + AC ^ 2)
  -- Assuming the fold point area calculation as per the problem's solution
  let q := 270
  let r := 324
  let s := 3
  q * Real.pi - r * Real.sqrt s

theorem fold_point_area_sum (AB AC : ℝ) (angle_B : ℝ) (hAB : AB = 36) (hAC : AC = 72) (hangle_B : angle_B = π / 2) :
  let S := fold_point_area AB AC angle_B
  ∃ q r s : ℕ, S = q * Real.pi - r * Real.sqrt s ∧ q + r + s = 597 :=
by
  sorry

end fold_point_area_sum_l213_213610


namespace part1_part2_l213_213852

-- Define the function f
def f (x a : ℝ) : ℝ := abs (x - 1) + abs (x - a)

-- Problem (1)
theorem part1 (a : ℝ) (h : a = 1) : 
  ∀ x : ℝ, f x a ≥ 2 ↔ x ≤ 0 ∨ x ≥ 2 := 
  sorry

-- Problem (2)
theorem part2 (a : ℝ) (h : a > 1) : 
  (∀ x : ℝ, f x a + abs (x - 1) ≥ 2) ↔ a ≥ 3 := 
  sorry

end part1_part2_l213_213852


namespace solution_set_of_inequality_l213_213620

noncomputable def f : ℝ → ℝ := sorry
noncomputable def f' : ℝ → ℝ := sorry

theorem solution_set_of_inequality (H1 : f 1 = 1)
  (H2 : ∀ x : ℝ, x * f' x < 1 / 2) :
  {x : ℝ | f (Real.log x ^ 2) < (Real.log x ^ 2) / 2 + 1 / 2} = 
  {x : ℝ | 0 < x ∧ x < 1 / 10} ∪ {x : ℝ | x > 10} :=
sorry

end solution_set_of_inequality_l213_213620


namespace train_crosses_second_platform_in_20_sec_l213_213824

theorem train_crosses_second_platform_in_20_sec
  (length_train : ℝ)
  (length_first_platform : ℝ)
  (time_first_platform : ℝ)
  (length_second_platform : ℝ)
  (time_second_platform : ℝ):

  length_train = 100 ∧
  length_first_platform = 350 ∧
  time_first_platform = 15 ∧
  length_second_platform = 500 →
  time_second_platform = 20 := by
  sorry

end train_crosses_second_platform_in_20_sec_l213_213824


namespace fraction_decimal_equivalent_l213_213602

theorem fraction_decimal_equivalent : (7 / 16 : ℝ) = 0.4375 :=
by
  sorry

end fraction_decimal_equivalent_l213_213602


namespace find_multiple_l213_213236

theorem find_multiple (n : ℕ) (h₁ : n = 5) (m : ℕ) (h₂ : 7 * n - 15 > m * n) : m = 3 :=
by
  sorry

end find_multiple_l213_213236


namespace radical_axis_eq_l213_213600

-- Definitions of the given circles
def circle1_eq (x y : ℝ) : Prop := x^2 + y^2 - 4 * x + 6 * y = 0
def circle2_eq (x y : ℝ) : Prop := x^2 + y^2 - 6 * x = 0

-- The theorem proving that the equation of the radical axis is 3x - y - 9 = 0
theorem radical_axis_eq (x y : ℝ) :
  (circle1_eq x y) ∧ (circle2_eq x y) → 3 * x - y - 9 = 0 :=
sorry

end radical_axis_eq_l213_213600


namespace production_days_l213_213566

noncomputable def daily_production (n : ℕ) : Prop :=
50 * n + 90 = 58 * (n + 1)

theorem production_days (n : ℕ) (h : daily_production n) : n = 4 :=
by sorry

end production_days_l213_213566


namespace carol_first_toss_six_probability_l213_213367

theorem carol_first_toss_six_probability :
  let p := 1 / 6
  let prob_no_six := (5 / 6: ℚ)
  let prob_carol_first_six := prob_no_six^3 * p
  let prob_cycle := prob_no_six^4
  (prob_carol_first_six / (1 - prob_cycle)) = 125 / 671 :=
by
  let p := (1 / 6:ℚ)
  let prob_no_six := (5 / 6: ℚ)
  let prob_carol_first_six := prob_no_six^3 * p
  let prob_cycle := prob_no_six^4
  have sum_geo_series : prob_carol_first_six / (1 - prob_cycle) = 125 / 671 := sorry
  exact sum_geo_series

end carol_first_toss_six_probability_l213_213367


namespace sin_func_even_min_period_2pi_l213_213399

noncomputable def f (x : ℝ) : ℝ := Real.sin (13 * Real.pi / 2 - x)

theorem sin_func_even_min_period_2pi :
  (∀ x : ℝ, f (-x) = f x) ∧ (∀ T > 0, (∀ x : ℝ, f (x + T) = f x) → T ≥ 2 * Real.pi) ∧ (∀ x : ℝ, f (x + 2 * Real.pi) = f x) :=
by
  sorry

end sin_func_even_min_period_2pi_l213_213399


namespace roots_squared_sum_l213_213544

theorem roots_squared_sum :
  (∀ x, x^2 + 2 * x - 8 = 0 → (x = x1 ∨ x = x2)) →
  x1 + x2 = -2 ∧ x1 * x2 = -8 →
  x1^2 + x2^2 = 20 :=
by
  intros roots_eq_sum_prod_eq
  sorry

end roots_squared_sum_l213_213544


namespace solve_for_a_l213_213802

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if h : x >= 0 then 4^x else 2^(a - x)

theorem solve_for_a (a : ℝ) (h : a ≠ 1) (h_f_eq : f a (1 - a) = f a (a - 1)) : a = 1 / 2 :=
by
  sorry

end solve_for_a_l213_213802


namespace find_length_BD_l213_213051

theorem find_length_BD (c : ℝ) (h : c ≥ Real.sqrt 7) :
  ∃BD, BD = Real.sqrt (c^2 - 7) :=
sorry

end find_length_BD_l213_213051


namespace leap_day_2040_is_tuesday_l213_213869

-- Define the given condition that 29th February 2012 is Wednesday
def feb_29_2012_is_wednesday : Prop := sorry

-- Define the calculation of the day of the week for February 29, 2040
def day_of_feb_29_2040 (initial_day : Nat) : Nat := (10228 % 7 + initial_day) % 7

-- Define the proof statement
theorem leap_day_2040_is_tuesday : feb_29_2012_is_wednesday →
  (day_of_feb_29_2040 3 = 2) := -- Here, 3 represents Wednesday and 2 represents Tuesday
sorry

end leap_day_2040_is_tuesday_l213_213869


namespace binomial_expansion_terms_l213_213880

theorem binomial_expansion_terms (x n : ℝ) (hn : n = 8) : 
  ∃ t, t = 3 :=
  sorry

end binomial_expansion_terms_l213_213880


namespace eduardo_ate_fraction_of_remaining_l213_213186

theorem eduardo_ate_fraction_of_remaining (init_cookies : ℕ) (nicole_fraction : ℚ) (remaining_percent : ℚ) :
  init_cookies = 600 →
  nicole_fraction = 2 / 5 →
  remaining_percent = 24 / 100 →
  (360 - (600 * 24 / 100)) / 360 = 3 / 5 := by
  sorry

end eduardo_ate_fraction_of_remaining_l213_213186


namespace value_of_4_and_2_l213_213474

noncomputable def custom_and (a b : ℕ) : ℕ :=
  ((a + b) * (a - b)) ^ 2

theorem value_of_4_and_2 : custom_and 4 2 = 144 :=
  sorry

end value_of_4_and_2_l213_213474


namespace sonny_cookie_problem_l213_213410

theorem sonny_cookie_problem 
  (total_boxes : ℕ) (boxes_sister : ℕ) (boxes_cousin : ℕ) (boxes_left : ℕ) (boxes_brother : ℕ) : 
  total_boxes = 45 → boxes_sister = 9 → boxes_cousin = 7 → boxes_left = 17 → 
  boxes_brother = total_boxes - boxes_left - boxes_sister - boxes_cousin → 
  boxes_brother = 12 :=
by
  intros h_total h_sister h_cousin h_left h_brother
  rw [h_total, h_sister, h_cousin, h_left] at h_brother
  exact h_brother

end sonny_cookie_problem_l213_213410


namespace find_f_neg12_add_f_14_l213_213757

noncomputable def f (x : ℝ) : ℝ := 1 + Real.log (Real.sqrt (x^2 - 2*x + 2) - x + 1)

theorem find_f_neg12_add_f_14 : f (-12) + f 14 = 2 :=
by
  -- The hard part, the actual proof, is left as sorry.
  sorry

end find_f_neg12_add_f_14_l213_213757


namespace xy_exists_5n_l213_213942

theorem xy_exists_5n (n : ℕ) (hpos : 0 < n) :
  ∃ x y : ℤ, x^2 + y^2 = 5^n ∧ Int.gcd x 5 = 1 ∧ Int.gcd y 5 = 1 :=
sorry

end xy_exists_5n_l213_213942


namespace residue_system_mod_3n_l213_213605

theorem residue_system_mod_3n (n : ℕ) (h_odd : n % 2 = 1) :
  ∃ (a b : ℕ → ℕ) (k : ℕ), 
  (∀ i, a i = 3 * i - 2) ∧ 
  (∀ i, b i = 3 * i - 3) ∧
  (∀ i (k : ℕ), 0 < k ∧ k < n → 
    (a i + a (i + 1)) % (3 * n) ≠ (a i + b i) % (3 * n) ∧ 
    (a i + b i) % (3 * n) ≠ (b i + b (i + k)) % (3 * n) ∧ 
    (a i + a (i + 1)) % (3 * n) ≠ (b i + b (i + k)) % (3 * n)) :=
sorry

end residue_system_mod_3n_l213_213605


namespace largest_number_l213_213653

-- Define the set elements with b = -3
def neg_5b (b : ℤ) : ℤ := -5 * b
def pos_3b (b : ℤ) : ℤ := 3 * b
def frac_30_b (b : ℤ) : ℤ := 30 / b
def b_sq (b : ℤ) : ℤ := b * b

-- Prove that when b = -3, the largest element in the set {-5b, 3b, 30/b, b^2, 2} is 15
theorem largest_number (b : ℤ) (h : b = -3) : max (max (max (max (neg_5b b) (pos_3b b)) (frac_30_b b)) (b_sq b)) 2 = 15 :=
by {
  sorry
}

end largest_number_l213_213653


namespace inequality_relationship_l213_213982

noncomputable def a := 1 / 2023
noncomputable def b := Real.exp (-2022 / 2023)
noncomputable def c := (Real.cos (1 / 2023)) / 2023

theorem inequality_relationship : b > a ∧ a > c :=
by
  -- Initializing and defining the variables
  let a := a
  let b := b
  let c := c
  -- Providing the required proof
  sorry

end inequality_relationship_l213_213982


namespace Clea_Rides_Escalator_Alone_l213_213027

-- Defining the conditions
variables (x y k : ℝ)
def Clea_Walking_Speed := x
def Total_Distance := y = 75 * x
def Time_with_Moving_Escalator := 30 * (x + k) = y
def Escalator_Speed := k = 1.5 * x

-- Stating the proof problem
theorem Clea_Rides_Escalator_Alone : 
  Total_Distance x y → 
  Time_with_Moving_Escalator x y k → 
  Escalator_Speed x k → 
  y / k = 50 :=
by
  intros
  sorry

end Clea_Rides_Escalator_Alone_l213_213027


namespace nalani_net_amount_l213_213280

-- Definitions based on the conditions
def luna_birth := 10 -- Luna gave birth to 10 puppies
def stella_birth := 14 -- Stella gave birth to 14 puppies
def luna_sold := 8 -- Nalani sold 8 puppies from Luna's litter
def stella_sold := 10 -- Nalani sold 10 puppies from Stella's litter
def luna_price := 200 -- Price per puppy for Luna's litter is $200
def stella_price := 250 -- Price per puppy for Stella's litter is $250
def luna_cost := 80 -- Cost of raising each puppy from Luna's litter is $80
def stella_cost := 90 -- Cost of raising each puppy from Stella's litter is $90

-- Theorem stating the net amount received by Nalani
theorem nalani_net_amount : 
        luna_sold * luna_price + stella_sold * stella_price - 
        (luna_birth * luna_cost + stella_birth * stella_cost) = 2040 :=
by 
  sorry

end nalani_net_amount_l213_213280


namespace count_semiprimes_expressed_as_x_cubed_minus_1_l213_213340

open Nat

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

def is_semiprime (n : ℕ) : Prop :=
  ∃ p q : ℕ, is_prime p ∧ is_prime q ∧ p * q = n

theorem count_semiprimes_expressed_as_x_cubed_minus_1 :
  (∃ S : Finset ℕ, 
    S.card = 4 ∧ 
    ∀ n ∈ S, n < 2018 ∧ 
    ∃ x : ℕ, x > 0 ∧ x^3 - 1 = n ∧ is_semiprime n) :=
sorry

end count_semiprimes_expressed_as_x_cubed_minus_1_l213_213340


namespace part1_solution_part2_solution_l213_213496

noncomputable def f (x a : ℝ) : ℝ := abs (x - a) * x + abs (x - 2) * (x - a)

theorem part1_solution (a : ℝ) (h : a = 1) :
  {x : ℝ | f x a < 0} = {x : ℝ | x < 1} :=
by
  sorry

theorem part2_solution (x : ℝ) (hx : x < 1) :
  {a : ℝ | f x a < 0} = {a : ℝ | 1 ≤ a} :=
by
  sorry

end part1_solution_part2_solution_l213_213496


namespace sharpener_difference_l213_213700

/-- A hand-crank pencil sharpener can sharpen one pencil every 45 seconds.
An electric pencil sharpener can sharpen one pencil every 20 seconds.
The total available time is 360 seconds (i.e., 6 minutes).
Prove that the difference in the number of pencils sharpened 
by the electric sharpener and the hand-crank sharpener in 360 seconds is 10 pencils. -/
theorem sharpener_difference (time : ℕ) (hand_crank_rate : ℕ) (electric_rate : ℕ) 
(h_time : time = 360) (h_hand_crank : hand_crank_rate = 45) (h_electric : electric_rate = 20) :
  (time / electric_rate) - (time / hand_crank_rate) = 10 := by
  sorry

end sharpener_difference_l213_213700


namespace triangle_inequality_sum_l213_213832

theorem triangle_inequality_sum (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : a + b > c) (h5 : b + c > a) (h6 : c + a > b) :
  (c / (a + b)) + (a / (b + c)) + (b / (c + a)) > 1 :=
by
  sorry

end triangle_inequality_sum_l213_213832


namespace XYZStockPriceIs75_l213_213198

/-- XYZ stock price model 
Starts at $50, increases by 200% in first year, 
then decreases by 50% in second year.
-/
def XYZStockPriceEndOfSecondYear : ℝ :=
  let initialPrice := 50
  let firstYearIncreaseRate := 2.0
  let secondYearDecreaseRate := 0.5
  let priceAfterFirstYear := initialPrice * (1 + firstYearIncreaseRate)
  let priceAfterSecondYear := priceAfterFirstYear * (1 - secondYearDecreaseRate)
  priceAfterSecondYear

theorem XYZStockPriceIs75 : XYZStockPriceEndOfSecondYear = 75 := by
  sorry

end XYZStockPriceIs75_l213_213198


namespace parabola_directrix_l213_213816

theorem parabola_directrix
  (p : ℝ) (hp : p > 0)
  (O : ℝ × ℝ := (0,0))
  (Focus_F : ℝ × ℝ := (p / 2, 0))
  (Point_P : ℝ × ℝ)
  (Point_Q : ℝ × ℝ)
  (H1 : Point_P.1 = p / 2 ∧ Point_P.2^2 = 2 * p * Point_P.1)
  (H2 : Point_P.1 = Point_P.1) -- This comes out of the perpendicularity of PF to x-axis
  (H3 : Point_Q.2 = 0)
  (H4 : ∃ k_OP slope_OP, slope_OP = 2 ∧ ∃ k_PQ slope_PQ, slope_PQ = -1 / 2 ∧ k_OP * k_PQ = -1)
  (H5 : abs (Point_Q.1 - Focus_F.1) = 6) :
  x = -3 / 2 := 
sorry

end parabola_directrix_l213_213816


namespace smallest_prime_sum_of_three_different_primes_is_19_l213_213964

theorem smallest_prime_sum_of_three_different_primes_is_19 :
  ∃ (p : ℕ), Prime p ∧ p = 19 ∧ (∀ a b c : ℕ, a ≠ b → b ≠ c → a ≠ c → Prime a → Prime b → Prime c → a + b + c = p → p ≥ 19) :=
by
  sorry

end smallest_prime_sum_of_three_different_primes_is_19_l213_213964


namespace randy_initial_blocks_l213_213542

theorem randy_initial_blocks (x : ℕ) (used_blocks : ℕ) (left_blocks : ℕ) 
  (h1 : used_blocks = 36) (h2 : left_blocks = 23) (h3 : x = used_blocks + left_blocks) :
  x = 59 := by 
  sorry

end randy_initial_blocks_l213_213542


namespace find_a_l213_213762

noncomputable def f (x a : ℝ) : ℝ := x - Real.log (x + 2) + Real.exp (x - a) + 4 * Real.exp (a - x)

theorem find_a (a : ℝ) :
  (∃ x₀ : ℝ, f x₀ a = 3) → a = -Real.log 2 - 1 :=
by
  sorry

end find_a_l213_213762


namespace intersection_M_N_l213_213999

def M : Set ℝ := {x | x < 1}
def N : Set ℝ := {x | 0 < x ∧ x < 2}

theorem intersection_M_N : M ∩ N = {x | 0 < x ∧ x < 1} :=
by
  sorry

end intersection_M_N_l213_213999


namespace annual_interest_payment_l213_213056

def principal : ℝ := 10000
def quarterly_rate : ℝ := 0.05

theorem annual_interest_payment :
  (principal * quarterly_rate * 4) = 2000 :=
by sorry

end annual_interest_payment_l213_213056


namespace double_windows_downstairs_eq_twelve_l213_213279

theorem double_windows_downstairs_eq_twelve
  (D : ℕ)
  (H1 : ∀ d, d = D → 4 * d + 32 = 80) :
  D = 12 :=
by
  sorry

end double_windows_downstairs_eq_twelve_l213_213279


namespace segment_order_l213_213447

def angle_sum_triangle (A B C : ℝ) : Prop := A + B + C = 180

def order_segments (angles_ABC angles_XYZ angles_ZWX : ℝ → ℝ → ℝ) : Prop :=
  let A := angles_ABC 55 60
  let B := angles_XYZ 95 70
  ∀ (XY YZ ZX WX WZ: ℝ), 
    YZ < ZX ∧ ZX < XY ∧ ZX < WZ ∧ WZ < WX

theorem segment_order:
  ∀ (A B C X Y Z W : Type)
  (XYZ_ang ZWX_ang : ℝ), 
  angle_sum_triangle 55 60 65 →
  angle_sum_triangle 95 70 15 →
  order_segments (angles_ABC) (angles_XYZ) (angles_ZWX)
:= sorry

end segment_order_l213_213447


namespace eleventh_term_of_sequence_l213_213842

def inversely_proportional_sequence (a : ℕ → ℝ) (c : ℝ) : Prop :=
  ∀ n : ℕ, a n * a (n + 1) = c

theorem eleventh_term_of_sequence :
  ∃ a : ℕ → ℝ,
    (a 1 = 3) ∧
    (a 2 = 6) ∧
    inversely_proportional_sequence a 18 ∧
    a 11 = 3 :=
by
  sorry

end eleventh_term_of_sequence_l213_213842


namespace flower_growth_l213_213806

theorem flower_growth (total_seeds : ℕ) (seeds_per_bed : ℕ) (max_grow_per_bed : ℕ) (h1 : total_seeds = 55) (h2 : seeds_per_bed = 15) (h3 : max_grow_per_bed = 60) : total_seeds ≤ 55 :=
by
  -- use the given conditions
  have h4 : total_seeds = 55 := h1
  sorry -- Proof goes here, omitted as instructed

end flower_growth_l213_213806


namespace fraction_of_b_eq_two_thirds_l213_213354

theorem fraction_of_b_eq_two_thirds (A B : ℝ) (x : ℝ) (h1 : A + B = 1210) (h2 : B = 484)
  (h3 : (2/3) * A = x * B) : x = 2/3 :=
by
  sorry

end fraction_of_b_eq_two_thirds_l213_213354


namespace shortest_distance_from_circle_to_line_l213_213963

theorem shortest_distance_from_circle_to_line :
  let circle := { p : ℝ × ℝ | (p.1 - 5)^2 + (p.2 - 3)^2 = 9 }
  let line := { p : ℝ × ℝ | 3 * p.1 + 4 * p.2 - 2 = 0 }
  ∀ (M : ℝ × ℝ), M ∈ circle → ∃ d : ℝ, d = 2 ∧ ∀ q ∈ line, dist M q = d := 
sorry

end shortest_distance_from_circle_to_line_l213_213963


namespace journey_total_distance_l213_213596

theorem journey_total_distance (D : ℝ) (h_train : D * (3 / 5) = t) (h_bus : D * (7 / 20) = b) (h_walk : D * (1 - ((3 / 5) + (7 / 20))) = 6.5) : D = 130 :=
by
  sorry

end journey_total_distance_l213_213596


namespace not_all_polynomials_sum_of_cubes_l213_213440

theorem not_all_polynomials_sum_of_cubes :
  ¬ ∀ P : Polynomial ℤ, ∃ Q : Polynomial ℤ, P = Q^3 + Q^3 + Q^3 :=
by
  sorry

end not_all_polynomials_sum_of_cubes_l213_213440


namespace twins_age_l213_213769

theorem twins_age (x : ℕ) (h : (x + 1) * (x + 1) = x * x + 5) : x = 2 :=
by
  sorry

end twins_age_l213_213769


namespace nth_equation_l213_213035

theorem nth_equation (n : ℕ) : (n + 1)^2 - 1 = n * (n + 2) := 
by 
  sorry

end nth_equation_l213_213035


namespace necessary_but_not_sufficient_condition_l213_213107

variable (a : ℝ) (x : ℝ)

def inequality_holds_for_all_real_numbers (a : ℝ) : Prop :=
    ∀ x : ℝ, (a * x^2 - a * x + 1 > 0)

theorem necessary_but_not_sufficient_condition :
  (0 < a ∧ a < 4) ↔
  (inequality_holds_for_all_real_numbers a) :=
by
  sorry

end necessary_but_not_sufficient_condition_l213_213107


namespace fraction_of_girls_l213_213001

variable {T G B : ℕ}
variable (ratio : ℚ)

theorem fraction_of_girls (X : ℚ) (h1 : ∀ (G : ℕ) (T : ℕ), X * G = (1/4) * T)
  (h2 : ratio = 5 / 3) (h3 : ∀ (G : ℕ) (B : ℕ), B / G = ratio) :
  X = 2 / 3 :=
by 
  sorry

end fraction_of_girls_l213_213001


namespace total_pens_count_l213_213925

def total_pens (red black blue : ℕ) : ℕ :=
  red + black + blue

theorem total_pens_count :
  let red := 8
  let black := red + 10
  let blue := red + 7
  total_pens red black blue = 41 :=
by
  sorry

end total_pens_count_l213_213925


namespace number_of_fowls_l213_213330

theorem number_of_fowls (chickens : ℕ) (ducks : ℕ) (h1 : chickens = 28) (h2 : ducks = 18) : chickens + ducks = 46 :=
by
  sorry

end number_of_fowls_l213_213330


namespace range_of_k_for_domain_real_l213_213462

theorem range_of_k_for_domain_real (k : ℝ) :
  (∀ x : ℝ, k * x^2 - 6 * k * x + (k + 8) ≥ 0) ↔ (0 ≤ k ∧ k ≤ 1) :=
sorry

end range_of_k_for_domain_real_l213_213462


namespace find_temp_friday_l213_213689

-- Definitions for conditions
variables (M T W Th F : ℝ)

-- Condition 1: Average temperature for Monday to Thursday is 48 degrees
def avg_temp_mon_thu : Prop := (M + T + W + Th) / 4 = 48

-- Condition 2: Average temperature for Tuesday to Friday is 46 degrees
def avg_temp_tue_fri : Prop := (T + W + Th + F) / 4 = 46

-- Condition 3: Temperature on Monday is 39 degrees
def temp_monday : Prop := M = 39

-- Theorem: Temperature on Friday is 31 degrees
theorem find_temp_friday (h1 : avg_temp_mon_thu M T W Th)
                         (h2 : avg_temp_tue_fri T W Th F)
                         (h3 : temp_monday M) :
  F = 31 :=
sorry

end find_temp_friday_l213_213689


namespace remainder_division_l213_213112

theorem remainder_division (N : ℤ) (hN : N % 899 = 63) : N % 29 = 5 := 
by 
  sorry

end remainder_division_l213_213112


namespace find_number_l213_213242

noncomputable def calc1 : Float := 0.47 * 1442
noncomputable def calc2 : Float := 0.36 * 1412
noncomputable def diff : Float := calc1 - calc2

theorem find_number :
  ∃ (n : Float), (diff + n = 6) :=
sorry

end find_number_l213_213242


namespace katherine_bottle_caps_l213_213554

-- Define the initial number of bottle caps Katherine has
def initial_bottle_caps : ℕ := 34

-- Define the number of bottle caps eaten by the hippopotamus
def eaten_bottle_caps : ℕ := 8

-- Define the remaining number of bottle caps Katherine should have
def remaining_bottle_caps : ℕ := initial_bottle_caps - eaten_bottle_caps

-- Theorem stating that Katherine will have 26 bottle caps after the hippopotamus eats 8 of them
theorem katherine_bottle_caps : remaining_bottle_caps = 26 := by
  sorry

end katherine_bottle_caps_l213_213554


namespace point_between_lines_l213_213059

theorem point_between_lines (b : ℝ) (h1 : 6 * 5 - 8 * b + 1 < 0) (h2 : 3 * 5 - 4 * b + 5 > 0) : b = 4 :=
  sorry

end point_between_lines_l213_213059


namespace problem1_problem2_problem3_problem4_l213_213513

-- (1) Prove (1 + sqrt 3) * (2 - sqrt 3) = -1 + sqrt 3
theorem problem1 : (1 + Real.sqrt 3) * (2 - Real.sqrt 3) = -1 + Real.sqrt 3 :=
by sorry

-- (2) Prove (sqrt 36 * sqrt 12) / sqrt 3 = 12
theorem problem2 : (Real.sqrt 36 * Real.sqrt 12) / Real.sqrt 3 = 12 :=
by sorry

-- (3) Prove sqrt 18 - sqrt 8 + sqrt (1 / 8) = (5 * sqrt 2) / 4
theorem problem3 : Real.sqrt 18 - Real.sqrt 8 + Real.sqrt (1 / 8) = (5 * Real.sqrt 2) / 4 :=
by sorry

-- (4) Prove (3 * sqrt 18 + (1 / 5) * sqrt 50 - 4 * sqrt (1 / 2)) / sqrt 32 = 2
theorem problem4 : (3 * Real.sqrt 18 + (1 / 5) * Real.sqrt 50 - 4 * Real.sqrt (1 / 2)) / Real.sqrt 32 = 2 :=
by sorry

end problem1_problem2_problem3_problem4_l213_213513


namespace carpet_length_is_9_l213_213071

noncomputable def carpet_length (width : ℝ) (living_room_area : ℝ) (coverage : ℝ) : ℝ :=
  living_room_area * coverage / width

theorem carpet_length_is_9 (width : ℝ) (living_room_area : ℝ) (coverage : ℝ) (length := carpet_length width living_room_area coverage) :
    width = 4 → living_room_area = 48 → coverage = 0.75 → length = 9 := by
  intros
  sorry

end carpet_length_is_9_l213_213071


namespace probability_B_in_A_is_17_over_24_l213_213675

open Set

def set_A : Set (ℝ × ℝ) := {p : ℝ × ℝ | abs p.1 + abs p.2 <= 2}
def set_B : Set (ℝ × ℝ) := {p : ℝ × ℝ | p ∈ set_A ∧ p.2 <= p.1 ^ 2}

noncomputable def area (s : Set (ℝ × ℝ)) : ℝ := sorry -- Assume we have means to compute the area of a set

theorem probability_B_in_A_is_17_over_24 :
  (area set_B / area set_A) = 17 / 24 :=
sorry

end probability_B_in_A_is_17_over_24_l213_213675


namespace factorize_expr_l213_213682

theorem factorize_expr (a x y : ℝ) : a * x^2 - 4 * a * y^2 = a * (x + 2 * y) * (x - 2 * y) :=
by
  sorry

end factorize_expr_l213_213682


namespace solve_for_x_l213_213650

theorem solve_for_x 
    (x : ℝ) 
    (h : (4 * x - 2) / (5 * x - 5) = 3 / 4) 
    : x = -7 :=
sorry

end solve_for_x_l213_213650


namespace smallest_factor_l213_213632

theorem smallest_factor (x : ℕ) (h1 : 936 = 2^3 * 3^1 * 13^1)
  (h2 : ∃ (x : ℕ), (936 * x) % 2^5 = 0 ∧ (936 * x) % 3^3 = 0 ∧ (936 * x) % 13^2 = 0) : x = 468 := 
sorry

end smallest_factor_l213_213632


namespace sequence_general_term_l213_213455

/-- Given the sequence {a_n} defined by a_n = 2^n * a_{n-1} for n > 1 and a_1 = 1,
    prove that the general term a_n = 2^((n^2 + n - 2) / 2) -/
theorem sequence_general_term (a : ℕ → ℕ) (h1 : a 1 = 1) 
  (h2 : ∀ n > 1, a n = 2^n * a (n-1)) :
  ∀ n, a n = 2^((n^2 + n - 2) / 2) :=
sorry

end sequence_general_term_l213_213455


namespace solve_for_x_l213_213788

variable {a b c x : ℝ}
variable (h_pos : a > 0 ∧ b > 0 ∧ c > 0)
variable (h_eq : (x - a - b) / c + (x - b - c) / a + (x - c - a) / b = 3)

theorem solve_for_x (h_pos : a > 0 ∧ b > 0 ∧ c > 0) (h_eq : (x - a - b) / c + (x - b - c) / a + (x - c - a) / b = 3) : 
  x = a + b + c :=
sorry

end solve_for_x_l213_213788


namespace alice_book_payment_l213_213085

/--
Alice is in the UK and wants to purchase a book priced at £25.
If one U.S. dollar is equivalent to £0.75, 
then Alice needs to pay 33.33 USD for the book.
-/
theorem alice_book_payment :
  ∀ (price_gbp : ℝ) (conversion_rate : ℝ), 
  price_gbp = 25 → conversion_rate = 0.75 → 
  (price_gbp / conversion_rate) = 33.33 :=
by
  intros price_gbp conversion_rate hprice hrate
  rw [hprice, hrate]
  sorry

end alice_book_payment_l213_213085


namespace fraction_equals_half_l213_213767

def numerator : ℤ := 1 - 2 + 4 - 8 + 16 - 32 + 64
def denominator : ℤ := 2 - 4 + 8 - 16 + 32 - 64 + 128

theorem fraction_equals_half : (numerator : ℚ) / (denominator : ℚ) = 1 / 2 :=
by
  sorry

end fraction_equals_half_l213_213767


namespace inclination_of_line_l213_213822

theorem inclination_of_line (α : ℝ) (h1 : ∃ l : ℝ, ∀ x y : ℝ, x + y + 1 = 0 → y = -x - 1) : α = 135 :=
by
  sorry

end inclination_of_line_l213_213822


namespace tangent_parallel_l213_213740

theorem tangent_parallel (a b : ℝ) 
  (h1 : b = (1 / 3) * a^3 - (1 / 2) * a^2 + 1) 
  (h2 : (a^2 - a) = 2) : 
  a = 2 ∨ a = -1 :=
by {
  -- proof skipped
  sorry
}

end tangent_parallel_l213_213740


namespace ratio_goats_sold_to_total_l213_213403

-- Define the conditions
variables (G S : ℕ) (total_revenue goat_sold : ℕ)
-- The ratio of goats to sheep is 5:7
axiom ratio_goats_to_sheep : G = (5/7) * S
-- The total number of sheep and goats is 360
axiom total_animals : G + S = 360
-- Mr. Mathews makes $7200 from selling some goats and 2/3 of the sheep
axiom selling_conditions : 40 * goat_sold + 30 * (2/3) * S = 7200

-- Prove the ratio of the number of goats sold to the total number of goats
theorem ratio_goats_sold_to_total : goat_sold / G = 1 / 2 := by
  sorry

end ratio_goats_sold_to_total_l213_213403


namespace total_seats_taken_l213_213069

def students_per_bus : ℝ := 14.0
def number_of_buses : ℝ := 2.0

theorem total_seats_taken :
  students_per_bus * number_of_buses = 28.0 :=
by
  sorry

end total_seats_taken_l213_213069


namespace f_x_minus_one_l213_213010

noncomputable def f (x : ℝ) : ℝ := x^2 - 2 * x + 5

theorem f_x_minus_one (x : ℝ) : f (x - 1) = x^2 - 4 * x + 8 :=
by
  sorry

end f_x_minus_one_l213_213010


namespace emma_deposit_withdraw_ratio_l213_213022

theorem emma_deposit_withdraw_ratio (initial_balance withdrawn new_balance : ℤ) 
  (h1 : initial_balance = 230) 
  (h2 : withdrawn = 60) 
  (h3 : new_balance = 290) 
  (deposited : ℤ) 
  (h_deposit : new_balance = initial_balance - withdrawn + deposited) :
  (deposited / withdrawn = 2) := 
sorry

end emma_deposit_withdraw_ratio_l213_213022


namespace wrongly_entered_mark_l213_213671

theorem wrongly_entered_mark (x : ℝ) : 
  (∀ (correct_mark avg_increase pupils : ℝ), 
  correct_mark = 45 ∧ avg_increase = 0.5 ∧ pupils = 80 ∧
  (avg_increase * pupils = (x - correct_mark)) →
  x = 85) :=
by 
  intro correct_mark avg_increase pupils
  rintro ⟨hc, ha, hp, h⟩
  sorry

end wrongly_entered_mark_l213_213671


namespace find_numbers_satisfying_conditions_l213_213318

theorem find_numbers_satisfying_conditions (x y z : ℝ)
(h1 : x + y + z = 11 / 18)
(h2 : 1 / x + 1 / y + 1 / z = 18)
(h3 : 2 / y = 1 / x + 1 / z) :
x = 1 / 9 ∧ y = 1 / 6 ∧ z = 1 / 3 :=
sorry

end find_numbers_satisfying_conditions_l213_213318


namespace pre_image_of_f_l213_213094

theorem pre_image_of_f (x y : ℝ) (f : ℝ × ℝ → ℝ × ℝ) 
  (h : f = λ p => (2 * p.1 + p.2, p.1 - 2 * p.2)) :
  f (1, 0) = (2, 1) := by
  sorry

end pre_image_of_f_l213_213094


namespace find_v_3_l213_213775

def u (x : ℤ) : ℤ := 4 * x - 9

def v (z : ℤ) : ℤ := z^2 + 4 * z - 1

theorem find_v_3 : v 3 = 20 := by
  sorry

end find_v_3_l213_213775


namespace system_solution_unique_l213_213640

theorem system_solution_unique (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) 
  (eq1 : x ^ 3 + 2 * y ^ 2 + 1 / (4 * z) = 1)
  (eq2 : y ^ 3 + 2 * z ^ 2 + 1 / (4 * x) = 1)
  (eq3 : z ^ 3 + 2 * x ^ 2 + 1 / (4 * y) = 1) :
  (x, y, z) = ( ( (-1 + Real.sqrt 3) / 2), ((-1 + Real.sqrt 3) / 2), ((-1 + Real.sqrt 3) / 2) ) := 
by
  sorry

end system_solution_unique_l213_213640


namespace speed_of_A_is_7_l213_213037

theorem speed_of_A_is_7
  (x : ℝ)
  (h1 : ∀ t : ℝ, t = 1)
  (h2 : ∀ y : ℝ, y = 3)
  (h3 : ∀ n : ℕ, n = 10)
  (h4 : x + 3 = 10) :
  x = 7 := by
  sorry

end speed_of_A_is_7_l213_213037


namespace increasing_interval_of_f_l213_213604

noncomputable def f (x : ℝ) : ℝ := 3 * Real.sin (2 * Real.pi / 3 - 2 * x)

theorem increasing_interval_of_f :
  ∃ a b : ℝ, f x = 3 * Real.sin (2 * Real.pi / 3 - 2 * x) ∧ (a = 7 * Real.pi / 12) ∧ (b = 13 * Real.pi / 12) ∧ ∀ x1 x2, a ≤ x1 ∧ x1 < x2 ∧ x2 ≤ b → f x1 < f x2 := 
sorry

end increasing_interval_of_f_l213_213604


namespace opposite_of_pi_is_neg_pi_l213_213488

-- Definition that the opposite of a number x is -1 * x
def opposite (x : ℝ) : ℝ := -1 * x

-- Theorem stating that the opposite of π is -π
theorem opposite_of_pi_is_neg_pi : opposite π = -π := 
  sorry

end opposite_of_pi_is_neg_pi_l213_213488


namespace find_a_l213_213946

theorem find_a (a b c : ℤ) (h : (∀ x : ℝ, (x - a) * (x - 5) + 4 = (x + b) * (x + c))) :
  a = 0 ∨ a = 1 :=
sorry

end find_a_l213_213946


namespace part_a_answer_part_b_answer_l213_213923

noncomputable def part_a_problem : Prop :=
  ∃! (x k : ℕ), x > 0 ∧ k > 0 ∧ 3^k - 1 = x^3

noncomputable def part_b_problem (n : ℕ) : Prop :=
  n > 1 ∧ n ≠ 3 → ∀ (x k : ℕ), ¬ (x > 0 ∧ k > 0 ∧ 3^k - 1 = x^n)

theorem part_a_answer : part_a_problem :=
  sorry

theorem part_b_answer (n : ℕ) : part_b_problem n :=
  sorry

end part_a_answer_part_b_answer_l213_213923


namespace eighty_first_number_in_set_l213_213032

theorem eighty_first_number_in_set : ∃ n : ℕ, n = 81 ∧ ∀ k : ℕ, (k = 8 * (n - 1) + 5) → k = 645 := by
  sorry

end eighty_first_number_in_set_l213_213032


namespace unique_solution_iff_d_ne_4_l213_213420

theorem unique_solution_iff_d_ne_4 (c d : ℝ) : 
  (∃! (x : ℝ), 4 * x - 7 + c = d * x + 2) ↔ d ≠ 4 := 
by 
  sorry

end unique_solution_iff_d_ne_4_l213_213420


namespace conclusion_l213_213303

-- Assuming U is the universal set and Predicates represent Mems, Ens, and Veens
variable (U : Type)
variable (Mem : U → Prop)
variable (En : U → Prop)
variable (Veen : U → Prop)

-- Hypotheses
variable (h1 : ∀ x, Mem x → En x)          -- Hypothesis I: All Mems are Ens
variable (h2 : ∀ x, En x → ¬Veen x)        -- Hypothesis II: No Ens are Veens

-- To be proven
theorem conclusion (x : U) : (Mem x → ¬Veen x) ∧ (Mem x → ¬Veen x) := sorry

end conclusion_l213_213303


namespace kaeli_problems_per_day_l213_213836

-- Definitions based on conditions
def problems_solved_per_day_marie_pascale : ℕ := 4
def total_problems_marie_pascale : ℕ := 72
def total_problems_kaeli : ℕ := 126

-- Number of days both took should be the same
def number_of_days : ℕ := total_problems_marie_pascale / problems_solved_per_day_marie_pascale

-- Kaeli solves 54 more problems than Marie-Pascale
def extra_problems_kaeli : ℕ := 54

-- Definition that Kaeli's total problems solved is that of Marie-Pascale plus 54
axiom kaeli_total_problems (h : total_problems_marie_pascale + extra_problems_kaeli = total_problems_kaeli) : True

-- Now to find x, the problems solved per day by Kaeli
def x : ℕ := total_problems_kaeli / number_of_days

-- Prove that x = 7
theorem kaeli_problems_per_day (h : total_problems_marie_pascale + extra_problems_kaeli = total_problems_kaeli) : x = 7 := by
  sorry

end kaeli_problems_per_day_l213_213836


namespace B_max_at_125_l213_213093

noncomputable def B (k : ℕ) : ℝ := (Nat.choose 500 k) * (0.3 : ℝ) ^ k

theorem B_max_at_125 :
  ∃ k, 0 ≤ k ∧ k ≤ 500 ∧ (∀ n, 0 ≤ n ∧ n ≤ 500 → B k ≥ B n) ∧ k = 125 :=
by
  sorry

end B_max_at_125_l213_213093


namespace abs_inequality_example_l213_213016

theorem abs_inequality_example (x : ℝ) : abs (5 - x) < 6 ↔ -1 < x ∧ x < 11 :=
by 
  sorry

end abs_inequality_example_l213_213016


namespace inletRate_is_3_l213_213290

def volumeTank (v_cubic_feet : ℕ) : ℕ :=
  1728 * v_cubic_feet

def outletRate1 : ℕ := 9 -- rate of first outlet in cubic inches/min
def outletRate2 : ℕ := 6 -- rate of second outlet in cubic inches/min
def tankVolume : ℕ := volumeTank 30 -- tank volume in cubic inches
def minutesToEmpty : ℕ := 4320 -- time to empty the tank in minutes

def effectiveRate (inletRate : ℕ) : ℕ :=
  outletRate1 + outletRate2 - inletRate

theorem inletRate_is_3 : (15 - 3) * minutesToEmpty = tankVolume :=
  by simp [outletRate1, outletRate2, tankVolume, minutesToEmpty]; sorry

end inletRate_is_3_l213_213290


namespace polynomial_evaluation_l213_213149

theorem polynomial_evaluation (x : ℝ) :
  x * (x * (x * (3 - x) - 5) + 15) - 2 = -x^4 + 3*x^3 - 5*x^2 + 15*x - 2 :=
by
  sorry

end polynomial_evaluation_l213_213149


namespace inequality_solution_l213_213288

theorem inequality_solution (a b : ℝ)
  (h₁ : ∀ x, - (1 : ℝ) / 2 < x ∧ x < (1 : ℝ) / 3 → ax^2 + bx + (2 : ℝ) > 0)
  (h₂ : - (1 : ℝ) / 2 = -(b / a))
  (h₃ : (- (1 : ℝ) / 6) = 2 / a) :
  a - b = -10 :=
sorry

end inequality_solution_l213_213288


namespace geometric_sequence_fifth_term_is_32_l213_213861

-- Defining the geometric sequence conditions
variables (a r : ℝ)

def third_term := a * r^2 = 18
def fourth_term := a * r^3 = 24
def fifth_term := a * r^4

theorem geometric_sequence_fifth_term_is_32 (h1 : third_term a r) (h2 : fourth_term a r) : 
  fifth_term a r = 32 := 
by
  sorry

end geometric_sequence_fifth_term_is_32_l213_213861


namespace color_opposite_gold_is_yellow_l213_213416

-- Define the colors as a datatype for clarity
inductive Color
| B | Y | O | K | S | G

-- Define the type for each face's color
structure CubeFaces :=
(top front right back left bottom : Color)

-- Given conditions
def first_view (c : CubeFaces) : Prop :=
  c.top = Color.B ∧ c.front = Color.Y ∧ c.right = Color.O

def second_view (c : CubeFaces) : Prop :=
  c.top = Color.B ∧ c.front = Color.K ∧ c.right = Color.O

def third_view (c : CubeFaces) : Prop :=
  c.top = Color.B ∧ c.front = Color.S ∧ c.right = Color.O

-- Problem statement
theorem color_opposite_gold_is_yellow (c : CubeFaces) :
  first_view c → second_view c → third_view c → (c.back = Color.G) → (c.front = Color.Y) :=
by
  sorry

end color_opposite_gold_is_yellow_l213_213416


namespace find_A_from_AB9_l213_213193

theorem find_A_from_AB9 (A B : ℕ) (h1 : 0 ≤ A ∧ A ≤ 9) (h2 : 0 ≤ B ∧ B ≤ 9) (h3 : 100 * A + 10 * B + 9 = 459) : A = 4 :=
sorry

end find_A_from_AB9_l213_213193


namespace remainder_11_pow_1000_mod_500_l213_213551

theorem remainder_11_pow_1000_mod_500 : (11 ^ 1000) % 500 = 1 :=
by
  have h1 : 11 % 5 = 1 := by norm_num
  have h2 : (11 ^ 10) % 100 = 1 := by
    -- Some steps omitted to satisfy conditions; normally would be generalized
    sorry
  have h3 : 500 = 5 * 100 := by norm_num
  -- Further omitted steps aligning with the Chinese Remainder Theorem application.
  sorry

end remainder_11_pow_1000_mod_500_l213_213551


namespace thomas_spends_40000_in_a_decade_l213_213391

/-- 
Thomas spends 4k dollars every year on his car insurance.
One decade is 10 years.
-/
def spending_per_year : ℕ := 4000

def years_in_a_decade : ℕ := 10

/-- 
We need to prove that the total amount Thomas spends in a decade on car insurance equals $40,000.
-/
theorem thomas_spends_40000_in_a_decade : spending_per_year * years_in_a_decade = 40000 := by
  sorry

end thomas_spends_40000_in_a_decade_l213_213391


namespace friend_reading_time_l213_213939

theorem friend_reading_time (S : ℝ) (H1 : S > 0) (H2 : 3 = 2 * (3 / 2)) : 
  (1.5 / (5 * S)) = 0.3 :=
by 
  sorry

end friend_reading_time_l213_213939


namespace find_number_l213_213910

theorem find_number : ∃ x : ℝ, 3 * x - 1 = 2 * x ∧ x = 1 := sorry

end find_number_l213_213910


namespace painters_workdays_l213_213720

theorem painters_workdays (five_painters_days : ℝ) (four_painters_days : ℝ) : 
  (5 * five_painters_days = 9) → (4 * four_painters_days = 9) → (four_painters_days = 2.25) :=
by
  intros h1 h2
  sorry

end painters_workdays_l213_213720


namespace zoo_ticket_problem_l213_213024

theorem zoo_ticket_problem :
  ∀ (total_amount adult_ticket_cost children_ticket_cost : ℕ)
    (num_adult_tickets : ℕ),
  total_amount = 119 →
  adult_ticket_cost = 21 →
  children_ticket_cost = 14 →
  num_adult_tickets = 4 →
  6 = (num_adult_tickets + (total_amount - num_adult_tickets * adult_ticket_cost) / children_ticket_cost) :=
by 
  intros total_amount adult_ticket_cost children_ticket_cost num_adult_tickets 
         total_amt_eq adult_ticket_cost_eq children_ticket_cost_eq num_adult_tickets_eq
  sorry

end zoo_ticket_problem_l213_213024


namespace number_of_teams_l213_213657

theorem number_of_teams (n : ℕ) 
  (h1 : ∀ (i j : ℕ), i ≠ j → (games_played : ℕ) = 4) 
  (h2 : ∀ (i j : ℕ), i ≠ j → (count : ℕ) = 760) : 
  n = 20 := 
by 
  sorry

end number_of_teams_l213_213657


namespace fraction_to_terminating_decimal_l213_213278

theorem fraction_to_terminating_decimal : (49 : ℚ) / 160 = 0.30625 := 
sorry

end fraction_to_terminating_decimal_l213_213278


namespace paths_via_checkpoint_l213_213477

/-- Define the grid configuration -/
structure Point :=
  (x : ℕ) (y : ℕ)

/-- Calculate the binomial coefficient -/
def binomial (n k : ℕ) : ℕ :=
  n.choose k

/-- Define points A, B, C -/
def A : Point := ⟨0, 0⟩
def B : Point := ⟨5, 4⟩
def C : Point := ⟨3, 2⟩

/-- Calculate number of paths from A to C -/
def paths_A_to_C : ℕ :=
  binomial (3 + 2) 2

/-- Calculate number of paths from C to B -/
def paths_C_to_B : ℕ :=
  binomial (2 + 2) 2

/-- Calculate total number of paths from A to B via C -/
def total_paths_A_to_B_via_C : ℕ :=
  (paths_A_to_C * paths_C_to_B)

theorem paths_via_checkpoint :
  total_paths_A_to_B_via_C = 60 :=
by
  -- The proof is skipped as per the instruction
  sorry

end paths_via_checkpoint_l213_213477


namespace smallest_n_for_terminating_decimal_l213_213957

theorem smallest_n_for_terminating_decimal :
  ∃ n : ℕ, n > 0 ∧ (∀ m : ℕ, m > 0 ∧ (∀ p: ℕ, (p ∣ (m + 150) → (p = 2 ∨ p = 5)) → m ≥ n)) ∧
  (∃ two_five_factors : ℕ, (two_five_factors = 5 ^ 3 * 2 ^ 3) ∧ (n + 150 = two_five_factors)) := 
by 
   exact ⟨50, by sorry⟩

end smallest_n_for_terminating_decimal_l213_213957


namespace quincy_sold_more_than_jake_l213_213352

variables (T : ℕ) (Jake Quincy : ℕ)

def thors_sales (T : ℕ) := T
def jakes_sales (T : ℕ) := T + 10
def quincys_sales (T : ℕ) := 10 * T

theorem quincy_sold_more_than_jake (h1 : jakes_sales T = Jake) 
  (h2 : quincys_sales T = Quincy) (h3 : Quincy = 200) : 
  Quincy - Jake = 170 :=
by
  sorry

end quincy_sold_more_than_jake_l213_213352


namespace fencing_required_l213_213262

theorem fencing_required
  (L : ℝ) (W : ℝ) (A : ℝ) (hL : L = 20) (hA : A = 80) (hW : A = L * W) :
  (L + 2 * W) = 28 :=
by {
  sorry
}

end fencing_required_l213_213262


namespace quadratic_ineq_solution_range_l213_213961

theorem quadratic_ineq_solution_range (a : ℝ) : 
  (∃ x : ℝ, 1 < x ∧ x < 4 ∧ 2*x^2 - 8*x - 4 - a > 0) ↔ a < -4 :=
by
  sorry

end quadratic_ineq_solution_range_l213_213961


namespace quadratic_roots_l213_213617

theorem quadratic_roots : ∀ x : ℝ, x * (x - 2) = 2 - x ↔ (x = 2 ∨ x = -1) := by
  intros
  sorry

end quadratic_roots_l213_213617


namespace sum_of_digits_l213_213439

theorem sum_of_digits :
  ∃ (E M V Y : ℕ), 
    (E ≠ M ∧ E ≠ V ∧ E ≠ Y ∧ M ≠ V ∧ M ≠ Y ∧ V ≠ Y) ∧
    (10 * Y + E) * (10 * M + E) = 111 * V ∧ 
    1 ≤ V ∧ V ≤ 9 ∧ 
    E + M + V + Y = 21 :=
by 
  sorry

end sum_of_digits_l213_213439


namespace trig_identity_problem_l213_213388

theorem trig_identity_problem {α : ℝ} (h : Real.tan α = 3) : 
  2 * Real.sin α ^ 2 - Real.sin α * Real.cos α + Real.cos α ^ 2 = 8 / 5 := 
by
  sorry

end trig_identity_problem_l213_213388


namespace lukas_avg_points_per_game_l213_213766

theorem lukas_avg_points_per_game (total_points games_played : ℕ) (h_total_points : total_points = 60) (h_games_played : games_played = 5) :
  (total_points / games_played = 12) :=
by
  sorry

end lukas_avg_points_per_game_l213_213766


namespace crayons_in_judahs_box_l213_213097

theorem crayons_in_judahs_box (karen_crayons beatrice_crayons gilbert_crayons judah_crayons : ℕ)
  (h1 : karen_crayons = 128)
  (h2 : beatrice_crayons = karen_crayons / 2)
  (h3 : gilbert_crayons = beatrice_crayons / 2)
  (h4 : judah_crayons = gilbert_crayons / 4) :
  judah_crayons = 8 :=
by {
  sorry
}

end crayons_in_judahs_box_l213_213097


namespace volume_expansion_rate_l213_213113

theorem volume_expansion_rate (R m : ℝ) (h1 : R = 1) (h2 : (4 * π * (m^3 - 1) / 3) / (m - 1) = 28 * π / 3) : m = 2 :=
sorry

end volume_expansion_rate_l213_213113


namespace tangential_quadrilateral_perpendicular_diagonals_l213_213820

-- Define what it means for a quadrilateral to be tangential
def is_tangential_quadrilateral (a b c d : ℝ) : Prop :=
  a + c = b + d

-- Define what it means for a quadrilateral to be a kite
def is_kite (a b c d : ℝ) : Prop :=
  a = b ∧ c = d

-- Define what it means for the diagonals of a quadrilateral to be perpendicular
def diagonals_perpendicular (a b c d : ℝ) : Prop :=
  sorry -- Actual geometric definition needs to be elaborated

-- Main statement to prove
theorem tangential_quadrilateral_perpendicular_diagonals (a b c d : ℝ) :
  is_tangential_quadrilateral a b c d → 
  (diagonals_perpendicular a b c d ↔ is_kite a b c d) := 
sorry

end tangential_quadrilateral_perpendicular_diagonals_l213_213820


namespace number_of_equilateral_triangles_l213_213311

noncomputable def parabola_equilateral_triangles (y x : ℝ) : Prop :=
  y^2 = 4 * x

theorem number_of_equilateral_triangles : ∃ n : ℕ, n = 2 ∧
  ∀ (a b c d e : ℝ), 
    (parabola_equilateral_triangles (a - 1) b) ∧ 
    (parabola_equilateral_triangles (c - 1) d) ∧ 
    ((a = e ∧ b = 0) ∨ (c = e ∧ d = 0)) → n = 2 :=
by 
  sorry

end number_of_equilateral_triangles_l213_213311


namespace part1_solution_set_part2_a_range_l213_213152

noncomputable def g (x : ℝ) (a : ℝ) : ℝ := abs x + 2 * abs (x + 2 - a)

-- Part 1: When a = 3, solving the inequality
theorem part1_solution_set (x : ℝ) : g x 3 ≤ 4 ↔ -2/3 ≤ x ∧ x ≤ 2 :=
by
  sorry

-- Part 2: Finding the range of a such that f(x) = g(x-2) >= 1 for all x in ℝ
noncomputable def f (x : ℝ) (a : ℝ) : ℝ := g (x - 2) a

theorem part2_a_range : (∀ x : ℝ, f x a ≥ 1) ↔ a ≤ 1 ∨ a ≥ 3 :=
by
  sorry

end part1_solution_set_part2_a_range_l213_213152


namespace c_range_l213_213857

open Real

theorem c_range (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h1 : 1 / a + 1 / b = 1)
  (h2 : 1 / (a + b) + 1 / c = 1) : 1 < c ∧ c ≤ 4 / 3 := 
sorry

end c_range_l213_213857


namespace geometric_sequence_tenth_term_l213_213257

theorem geometric_sequence_tenth_term :
  let a := 4
  let r := (4 / 3 : ℚ)
  a * r ^ 9 = (1048576 / 19683 : ℚ) :=
by
  sorry

end geometric_sequence_tenth_term_l213_213257


namespace solve_system_equations_l213_213622

theorem solve_system_equations :
  ∃ x y : ℚ, (5 * x * (y + 6) = 0 ∧ 2 * x + 3 * y = 1) ∧
  (x = 0 ∧ y = 1 / 3 ∨ x = 19 / 2 ∧ y = -6) :=
by
  sorry

end solve_system_equations_l213_213622


namespace perpendicular_lines_l213_213489

theorem perpendicular_lines {a : ℝ} :
  a*(a-1) + (1-a)*(2*a+3) = 0 → (a = 1 ∨ a = -3) := 
by
  intro h
  sorry

end perpendicular_lines_l213_213489


namespace expressions_equal_l213_213091

theorem expressions_equal {x y z : ℤ} : (x + 2 * y * z = (x + y) * (x + 2 * z)) ↔ (x + y + 2 * z = 1) :=
by
  sorry

end expressions_equal_l213_213091


namespace find_ratio_l213_213630

theorem find_ratio (x y c d : ℝ) (h1 : 8 * x - 6 * y = c) (h2 : 12 * y - 18 * x = d) (hx : x ≠ 0) (hy : y ≠ 0) (hd : d ≠ 0) : c / d = -1 := by
  sorry

end find_ratio_l213_213630


namespace compute_fraction_power_mul_l213_213177

theorem compute_fraction_power_mul : ((1 / 3: ℚ) ^ 4) * (1 / 5) = (1 / 405) := by
  -- proof goes here
  sorry

end compute_fraction_power_mul_l213_213177


namespace flight_duration_l213_213304

theorem flight_duration (h m : ℕ) (Hh : h = 2) (Hm : m = 32) : h + m = 34 := by
  sorry

end flight_duration_l213_213304


namespace sum_even_integers_12_to_46_l213_213890

theorem sum_even_integers_12_to_46 : 
  let a1 := 12
  let d := 2
  let an := 46
  let n := (an - a1) / d + 1
  let Sn := n * (a1 + an) / 2
  Sn = 522 := 
by
  let a1 := 12 
  let d := 2 
  let an := 46
  let n := (an - a1) / d + 1 
  let Sn := n * (a1 + an) / 2
  sorry

end sum_even_integers_12_to_46_l213_213890


namespace company_starts_to_make_profit_in_third_year_first_option_more_cost_effective_l213_213547

-- Define the conditions about the fishing company's boat purchase and expenses
def initial_purchase_cost : ℕ := 980000
def first_year_expenses : ℕ := 120000
def expense_increment : ℕ := 40000
def annual_income : ℕ := 500000

-- Prove that the company starts to make a profit in the third year
theorem company_starts_to_make_profit_in_third_year : 
  ∃ (year : ℕ), year = 3 ∧ 
  annual_income * year > initial_purchase_cost + first_year_expenses + (expense_increment * (year - 1) * year / 2) :=
sorry

-- Prove that the first option is more cost-effective
theorem first_option_more_cost_effective : 
  (annual_income * 3 - (initial_purchase_cost + first_year_expenses + expense_increment * (3 - 1) * 3 / 2) + 260000) > 
  (annual_income * 5 - (initial_purchase_cost + first_year_expenses + expense_increment * (5 - 1) * 5 / 2) + 80000) :=
sorry

end company_starts_to_make_profit_in_third_year_first_option_more_cost_effective_l213_213547


namespace remainder_when_divided_by_13_is_11_l213_213698

theorem remainder_when_divided_by_13_is_11 
  (n : ℕ) (h1 : n = 349) (h2 : n % 17 = 9) : 
  349 % 13 = 11 := 
by 
  sorry

end remainder_when_divided_by_13_is_11_l213_213698


namespace sequence_strictly_increasing_from_14_l213_213400

def a (n : ℕ) : ℤ := n^4 - 20 * n^2 - 10 * n + 1

theorem sequence_strictly_increasing_from_14 :
  ∀ n : ℕ, n ≥ 14 → a (n + 1) > a n :=
by
  sorry

end sequence_strictly_increasing_from_14_l213_213400


namespace larger_number_is_20_l213_213062

theorem larger_number_is_20 (a b : ℕ) (h1 : a + b = 9 * (a - b)) (h2 : a + b = 36) (h3 : a > b) : a = 20 :=
by
  sorry

end larger_number_is_20_l213_213062


namespace charles_whistles_l213_213215

theorem charles_whistles (S C : ℕ) (h1 : S = 45) (h2 : S = C + 32) : C = 13 := 
by
  sorry

end charles_whistles_l213_213215


namespace union_of_sets_l213_213063

def A := { x : ℝ | -1 ≤ x ∧ x ≤ 5 }
def B := { x : ℝ | 3 < x ∧ x < 9 }

theorem union_of_sets : (A ∪ B) = { x : ℝ | -1 ≤ x ∧ x < 9 } :=
by
  sorry

end union_of_sets_l213_213063


namespace problem_l213_213482

noncomputable def p : Prop :=
  ∀ x : ℝ, (0 < x) → Real.exp x > 1 + x

def q (f : ℝ → ℝ) : Prop :=
  (∀ x : ℝ, f (-x) + 2 = -(f x + 2)) → ∀ x : ℝ, f (-x) = f x - 4

theorem problem (f : ℝ → ℝ) : p ∨ q f :=
  sorry

end problem_l213_213482


namespace linear_eq_m_value_l213_213786

theorem linear_eq_m_value (x m : ℝ) (h : 2 * x + m = 5) (hx : x = 1) : m = 3 :=
by
  -- Here we would carry out the proof steps
  sorry

end linear_eq_m_value_l213_213786


namespace correct_statement_l213_213497

theorem correct_statement (a b : ℝ) (h_a : a ≥ 0) (h_b : b ≥ 0) : (a ≥ 0 ∧ b ≥ 0) :=
by
  exact ⟨h_a, h_b⟩

end correct_statement_l213_213497


namespace minimum_rooms_to_accommodate_fans_l213_213461

/-
Each hotel room can accommodate no more than 3 people. The hotel manager knows 
that a group of 100 football fans, who support three different teams, will soon 
arrive. A room can only house either men or women; and fans of different teams 
cannot be housed together. Prove that at least 37 rooms are needed to accommodate 
all the fans.
-/

noncomputable def minimum_rooms_needed (total_fans : ℕ) (fans_per_room : ℕ) : ℕ :=
  if h : fans_per_room > 0 then (total_fans + fans_per_room - 1) / fans_per_room else 0

theorem minimum_rooms_to_accommodate_fans :
  ∀ (total_fans : ℕ) (fans_per_room : ℕ)
    (num_teams : ℕ) (num_genders : ℕ),
  total_fans = 100 →
  fans_per_room = 3 →
  num_teams = 3 →
  num_genders = 2 →
  (minimum_rooms_needed total_fans fans_per_room) ≥ 37 :=
by
  intros total_fans fans_per_room num_teams num_genders h_total h_per_room h_teams h_genders
  -- Proof goes here
  sorry

end minimum_rooms_to_accommodate_fans_l213_213461


namespace blake_change_l213_213194

-- Definitions based on conditions
def n_l : ℕ := 4
def n_c : ℕ := 6
def p_l : ℕ := 2
def p_c : ℕ := 4 * p_l
def amount_given : ℕ := 6 * 10

-- Total cost calculations derived from the conditions
def total_cost_lollipops : ℕ := n_l * p_l
def total_cost_chocolates : ℕ := n_c * p_c
def total_cost : ℕ := total_cost_lollipops + total_cost_chocolates

-- Calculating the change
def change : ℕ := amount_given - total_cost

-- Theorem stating the final answer
theorem blake_change : change = 4 := sorry

end blake_change_l213_213194


namespace remainder_sum_mod_14_l213_213522

theorem remainder_sum_mod_14 
  (a b c : ℕ) 
  (ha : a % 14 = 5) 
  (hb : b % 14 = 5) 
  (hc : c % 14 = 5) :
  (a + b + c) % 14 = 1 := 
by
  sorry

end remainder_sum_mod_14_l213_213522


namespace rebecca_end_of_day_money_eq_l213_213754

-- Define the costs for different services
def haircut_cost   := 30
def perm_cost      := 40
def dye_job_cost   := 60
def extension_cost := 80

-- Define the supply costs for the services
def haircut_supply_cost   := 5
def dye_job_supply_cost   := 10
def extension_supply_cost := 25

-- Today's appointments
def num_haircuts   := 5
def num_perms      := 3
def num_dye_jobs   := 2
def num_extensions := 1

-- Additional incomes and expenses
def tips           := 75
def daily_expenses := 45

-- Calculate the total earnings and costs
def total_service_revenue : ℕ := 
  num_haircuts * haircut_cost +
  num_perms * perm_cost +
  num_dye_jobs * dye_job_cost +
  num_extensions * extension_cost

def total_revenue : ℕ := total_service_revenue + tips

def total_supply_cost : ℕ := 
  num_haircuts * haircut_supply_cost +
  num_dye_jobs * dye_job_supply_cost +
  num_extensions * extension_supply_cost

def end_of_day_money : ℕ := total_revenue - total_supply_cost - daily_expenses

-- Lean statement to prove Rebecca will have $430 at the end of the day
theorem rebecca_end_of_day_money_eq : end_of_day_money = 430 := by
  sorry

end rebecca_end_of_day_money_eq_l213_213754


namespace NumberOfRootsForEquation_l213_213615

noncomputable def numRootsAbsEq : ℕ :=
  let f := (fun x : ℝ => abs (abs (abs (abs (x - 1) - 9) - 9) - 3))
  let roots : List ℝ := [27, -25, 11, -9, 9, -7]
  roots.length

theorem NumberOfRootsForEquation : numRootsAbsEq = 6 := by
  sorry

end NumberOfRootsForEquation_l213_213615


namespace device_prices_within_budget_l213_213331

-- Given conditions
def x : ℝ := 12 -- Price of each type A device in thousands of dollars
def y : ℝ := 10 -- Price of each type B device in thousands of dollars
def budget : ℝ := 110 -- The budget in thousands of dollars

-- Conditions as given equations and inequalities
def condition1 : Prop := 3 * x - 2 * y = 16
def condition2 : Prop := 3 * y - 2 * x = 6
def budget_condition (a : ℕ) : Prop := 12 * a + 10 * (10 - a) ≤ budget

-- Theorem to prove
theorem device_prices_within_budget :
  condition1 ∧ condition2 ∧
  (∀ a : ℕ, a ≤ 5 → budget_condition a) :=
by sorry

end device_prices_within_budget_l213_213331


namespace slope_of_tangent_at_point_l213_213867

theorem slope_of_tangent_at_point (x : ℝ) (y : ℝ) (h_curve : y = x^3)
    (h_slope : 3*x^2 = 3) : (x = 1 ∧ y = 1) ∨ (x = -1 ∧ y = -1) := 
sorry

end slope_of_tangent_at_point_l213_213867


namespace difference_q_r_l213_213469

-- Conditions
variables (p q r : ℕ) (x : ℕ)
variables (h_ratio : 3 * x = p) (h_ratio2 : 7 * x = q) (h_ratio3 : 12 * x = r)
variables (h_diff_pq : q - p = 3200)

-- Proof problem to solve
theorem difference_q_r : q - p = 3200 → 12 * x - 7 * x = 4000 :=
by 
  intro h_diff_pq
  rw [h_ratio, h_ratio2, h_ratio3] at *
  sorry

end difference_q_r_l213_213469


namespace same_color_combination_sum_l213_213518

theorem same_color_combination_sum (m n : ℕ) (coprime_mn : Nat.gcd m n = 1)
  (prob_together : ∀ (total_candies : ℕ), total_candies = 20 →
    let terry_red := Nat.choose 8 2;
    let total_cases := Nat.choose total_candies 2;
    let prob_terry_red := terry_red / total_cases;
    
    let mary_red_given_terry := Nat.choose 6 2;
    let reduced_total_cases := Nat.choose 18 2;
    let prob_mary_red_given_terry := mary_red_given_terry / reduced_total_cases;
    
    let both_red := prob_terry_red * prob_mary_red_given_terry;
    
    let terry_blue := Nat.choose 12 2;
    let prob_terry_blue := terry_blue / total_cases;
    
    let mary_blue_given_terry := Nat.choose 10 2;
    let prob_mary_blue_given_terry := mary_blue_given_terry / reduced_total_cases;
    
    let both_blue := prob_terry_blue * prob_mary_blue_given_terry;
    
    let mixed_red_blue := Nat.choose 8 1 * Nat.choose 12 1;
    let prob_mixed_red_blue := mixed_red_blue / total_cases;
    let both_mixed := prob_mixed_red_blue;
    
    let prob_same_combination := both_red + both_blue + both_mixed;
    
    prob_same_combination = m / n
  ) :
  m + n = 5714 :=
by
  sorry

end same_color_combination_sum_l213_213518


namespace transform_equation_l213_213851

theorem transform_equation (x : ℝ) (h₁ : x ≠ 3 / 2) (h₂ : 5 - 3 * x = 1) :
  x = 4 / 3 :=
sorry

end transform_equation_l213_213851


namespace find_x_l213_213669

theorem find_x (x : ℝ) (A B : Set ℝ) (hA : A = {1, 4, x}) (hB : B = {1, x^2}) (h_inter : A ∩ B = B) : x = -2 ∨ x = 2 ∨ x = 0 :=
sorry

end find_x_l213_213669


namespace count_ordered_pairs_l213_213654

theorem count_ordered_pairs : 
  ∃ n : ℕ, n = 136 ∧ 
  ∀ a b : ℝ, 
    (∃ x y : ℤ, a * x + b * y = 2 ∧ (x : ℝ)^2 + (y : ℝ)^2 = 65) → n = 136 := 
sorry

end count_ordered_pairs_l213_213654


namespace smallest_digits_to_append_l213_213553

theorem smallest_digits_to_append (n : ℕ) (h : n = 2014) : 
  ∃ k : ℕ, (10^4 * n + k) % 2520 = 0 ∧ ∀ m, (10^m * n + k) % 2520 ≠ 0 → m > 4 := by
sorry

end smallest_digits_to_append_l213_213553


namespace line_equation_intersections_l213_213704

theorem line_equation_intersections (m b k : ℝ) (h1 : b ≠ 0) 
  (h2 : m * 2 + b = 7) (h3 : abs (k^2 + 8*k + 7 - (m*k + b)) = 4) :
  m = 6 ∧ b = -5 :=
by {
  sorry
}

end line_equation_intersections_l213_213704


namespace percentage_more_l213_213205

variable (J : ℝ) -- Juan's income
noncomputable def Tim_income := 0.60 * J -- T = 0.60J
noncomputable def Mart_income := 0.84 * J -- M = 0.84J

theorem percentage_more {J : ℝ} (T := Tim_income J) (M := Mart_income J) :
  ((M - T) / T) * 100 = 40 := by
  sorry

end percentage_more_l213_213205


namespace annulus_area_l213_213969

theorem annulus_area (B C RW : ℝ) (h1 : B > C)
  (h2 : B^2 - (C + 5)^2 = RW^2) : 
  π * RW^2 = π * (B^2 - (C + 5)^2) :=
by
  sorry

end annulus_area_l213_213969


namespace cube_volume_given_surface_area_l213_213405

/-- Surface area of a cube given the side length. -/
def surface_area (side_length : ℝ) := 6 * side_length^2

/-- Volume of a cube given the side length. -/
def volume (side_length : ℝ) := side_length^3

theorem cube_volume_given_surface_area :
  ∃ side_length : ℝ, surface_area side_length = 24 ∧ volume side_length = 8 :=
by
  sorry

end cube_volume_given_surface_area_l213_213405


namespace parabola_has_one_x_intercept_l213_213207

-- Define the equation of the parabola.
def parabola (y : ℝ) : ℝ := -3 * y ^ 2 + 2 * y + 4

-- Prove that the number of x-intercepts of the graph of the parabola is 1.
theorem parabola_has_one_x_intercept : (∃! y : ℝ, parabola y = 4) :=
by
  sorry

end parabola_has_one_x_intercept_l213_213207


namespace length_of_AB_l213_213150

theorem length_of_AB (A B P Q : ℝ) 
  (hp : 0 < P) (hp' : P < 1) 
  (hq : 0 < Q) (hq' : Q < 1) 
  (H1 : P = 3 / 7) (H2 : Q = 5 / 12)
  (H3 : P * (1 - Q) + Q * (1 - P) = 4) : 
  (B - A) = 336 / 11 :=
by
  sorry

end length_of_AB_l213_213150


namespace bie_l213_213765

noncomputable def surface_area_of_sphere (PA AB AC : ℝ) (hPA_AB : PA = AB) (hPA : PA = 2) (hAC : AC = 4) (r : ℝ) : ℝ :=
  let PC := Real.sqrt (PA ^ 2 + AC ^ 2)
  let radius := PC / 2
  4 * Real.pi * radius ^ 2

theorem bie'zhi_tetrahedron_surface_area
  (PA AB AC : ℝ)
  (hPA_AB : PA = AB)
  (hPA : PA = 2)
  (hAC : AC = 4)
  (PC : ℝ := Real.sqrt (PA ^ 2 + AC ^ 2))
  (r : ℝ := PC / 2)
  (surface_area : ℝ := 4 * Real.pi * r ^ 2)
  :
  surface_area = 20 * Real.pi := 
sorry

end bie_l213_213765


namespace find_number_l213_213763

theorem find_number (x : ℤ) : 45 - (28 - (x - (15 - 16))) = 55 ↔ x = 37 :=
by
  sorry

end find_number_l213_213763


namespace A_n_squared_l213_213335

-- Define C(n-2)
def C_n_2 (n : ℕ) : ℕ := n * (n - 1) / 2

-- Define A_n_2
def A_n_2 (n : ℕ) : ℕ := Nat.factorial n / Nat.factorial (n - 2)

theorem A_n_squared (n : ℕ) (hC : C_n_2 n = 15) : A_n_2 n = 30 := by
  sorry

end A_n_squared_l213_213335


namespace problem_solution_count_l213_213072

theorem problem_solution_count (n : ℕ) (h1 : (80 * n) ^ 40 > n ^ 80) (h2 : n ^ 80 > 3 ^ 160) : 
  ∃ s : Finset ℕ, s.card = 70 ∧ ∀ x ∈ s, 10 ≤ x ∧ x ≤ 79 :=
by
  sorry

end problem_solution_count_l213_213072


namespace ratio_of_perimeter_to_length_XY_l213_213310

noncomputable def XY : ℝ := 17
noncomputable def XZ : ℝ := 8
noncomputable def YZ : ℝ := 15
noncomputable def ZD : ℝ := 240 / 17

-- Defining the perimeter P
noncomputable def P : ℝ := 17 + 2 * (240 / 17)

-- Finally, the statement with the ratio in the desired form
theorem ratio_of_perimeter_to_length_XY : 
  (P / XY) = (654 / 289) :=
by
  sorry

end ratio_of_perimeter_to_length_XY_l213_213310


namespace sums_equal_l213_213573

theorem sums_equal (A B C : Type) (a b c : ℕ) :
  (a + b + c) = (a + (b + c)) ∧
  (a + b + c) = (b + (c + a)) ∧
  (a + b + c) = (c + (a + b)) :=
by 
  sorry

end sums_equal_l213_213573


namespace sheepdog_speed_l213_213887

theorem sheepdog_speed 
  (T : ℝ) (t : ℝ) (sheep_speed : ℝ) (initial_distance : ℝ)
  (total_distance_speed : ℝ) :
  T = 20  →
  t = 20 →
  sheep_speed = 12 →
  initial_distance = 160 →
  total_distance_speed = 20 →
  total_distance_speed * T = initial_distance + sheep_speed * t := 
by sorry

end sheepdog_speed_l213_213887


namespace solve_prime_equation_l213_213593

def is_prime (n : ℕ) : Prop := Nat.Prime n

theorem solve_prime_equation (p q : ℕ) (hp : is_prime p) (hq : is_prime q) (h : p ^ 2 - 6 * p * q + q ^ 2 + 3 * q - 1 = 0) :
  (p = 17 ∧ q = 3) :=
sorry

end solve_prime_equation_l213_213593


namespace minimum_value_of_expression_l213_213249

theorem minimum_value_of_expression
  (a b c : ℝ)
  (h : 2 * a + 2 * b + c = 8) :
  ∃ x, (x = (a - 1)^2 + (b + 2)^2 + (c - 3)^2) ∧ x ≥ (49 / 9) :=
sorry

end minimum_value_of_expression_l213_213249


namespace q_value_at_2_l213_213787

def q (x d e : ℤ) : ℤ := x^2 + d*x + e

theorem q_value_at_2 (d e : ℤ) 
  (h1 : ∃ p : ℤ → ℤ, ∀ x, x^4 + 8*x^2 + 49 = (q x d e) * (p x))
  (h2 : ∃ r : ℤ → ℤ, ∀ x, 2*x^4 + 5*x^2 + 36*x + 7 = (q x d e) * (r x)) :
  q 2 d e = 5 := 
sorry

end q_value_at_2_l213_213787


namespace total_profit_is_50_l213_213974

-- Define the initial conditions
def initial_milk : ℕ := 80
def initial_water : ℕ := 20
def milk_cost_per_liter : ℕ := 22
def first_mixture_milk : ℕ := 40
def first_mixture_water : ℕ := 5
def first_mixture_price : ℕ := 19
def second_mixture_milk : ℕ := 25
def second_mixture_water : ℕ := 10
def second_mixture_price : ℕ := 18
def third_mixture_milk : ℕ := initial_milk - (first_mixture_milk + second_mixture_milk)
def third_mixture_water : ℕ := 5
def third_mixture_price : ℕ := 21

-- Define variables for revenue calculations
def first_mixture_revenue : ℕ := (first_mixture_milk + first_mixture_water) * first_mixture_price
def second_mixture_revenue : ℕ := (second_mixture_milk + second_mixture_water) * second_mixture_price
def third_mixture_revenue : ℕ := (third_mixture_milk + third_mixture_water) * third_mixture_price
def total_revenue : ℕ := first_mixture_revenue + second_mixture_revenue + third_mixture_revenue

-- Define the total milk cost
def total_milk_used : ℕ := first_mixture_milk + second_mixture_milk + third_mixture_milk
def total_cost : ℕ := total_milk_used * milk_cost_per_liter

-- Define the profit as the difference between total revenue and total cost
def profit : ℕ := total_revenue - total_cost

-- Prove that the total profit is Rs. 50
theorem total_profit_is_50 : profit = 50 := by
  sorry

end total_profit_is_50_l213_213974


namespace pyramid_volume_l213_213328

theorem pyramid_volume (b : ℝ) (h₀ : b > 0) :
  let base_area := (b * b * (Real.sqrt 3)) / 4
  let height := b / 2
  let volume := (1 / 3) * base_area * height
  volume = (b^3 * (Real.sqrt 3)) / 24 :=
sorry

end pyramid_volume_l213_213328


namespace plumber_salary_percentage_l213_213053

def salary_construction_worker : ℕ := 100
def salary_electrician : ℕ := 2 * salary_construction_worker
def total_salary_without_plumber : ℕ := 2 * salary_construction_worker + salary_electrician
def total_labor_cost : ℕ := 650
def salary_plumber : ℕ := total_labor_cost - total_salary_without_plumber
def percentage_salary_plumber_as_construction_worker (x y : ℕ) : ℕ := (x * 100) / y

theorem plumber_salary_percentage :
  percentage_salary_plumber_as_construction_worker salary_plumber salary_construction_worker = 250 :=
by 
  sorry

end plumber_salary_percentage_l213_213053


namespace geometric_sequence_a3_l213_213743

theorem geometric_sequence_a3 (q : ℝ) (a : ℕ → ℝ) (h1 : a 1 = 1) (h2 : a 5 = 4) (h3 : ∀ n, a n = a 1 * q ^ (n - 1)) : a 3 = 2 :=
by
  sorry

end geometric_sequence_a3_l213_213743


namespace integer_values_between_fractions_l213_213240

theorem integer_values_between_fractions :
  let a := 4 / (Real.sqrt 3 + Real.sqrt 2)
  let b := 4 / (Real.sqrt 5 - Real.sqrt 3)
  ((⌊b⌋ - ⌈a⌉) + 1) = 6 :=
by sorry

end integer_values_between_fractions_l213_213240


namespace downstream_speed_is_28_l213_213332

-- Define the speed of the man in still water
def speed_in_still_water : ℝ := 24

-- Define the speed of the man rowing upstream
def speed_upstream : ℝ := 20

-- Define the speed of the stream
def speed_stream : ℝ := speed_in_still_water - speed_upstream

-- Define the speed of the man rowing downstream
def speed_downstream : ℝ := speed_in_still_water + speed_stream

-- The main theorem stating that the speed of the man rowing downstream is 28 kmph
theorem downstream_speed_is_28 : speed_downstream = 28 := by
  sorry

end downstream_speed_is_28_l213_213332


namespace range_of_x_l213_213848

noncomputable def f (x : ℝ) : ℝ := Real.log (1 + |x|) - 1 / (1 + x^2)

theorem range_of_x :
  ∀ x : ℝ, (f x > f (2*x - 1)) ↔ (1/3 < x ∧ x < 1) :=
by
  sorry

end range_of_x_l213_213848


namespace tan_square_proof_l213_213868

theorem tan_square_proof (θ : ℝ) (h : Real.tan θ = 2) : 
  1 / (Real.sin θ ^ 2 - Real.cos θ ^ 2) = 5 / 3 := by
  sorry

end tan_square_proof_l213_213868


namespace pies_from_apples_l213_213603

theorem pies_from_apples 
  (initial_apples : ℕ) (handed_out_apples : ℕ) (apples_per_pie : ℕ) 
  (remaining_apples := initial_apples - handed_out_apples) 
  (pies := remaining_apples / apples_per_pie) 
  (h1 : initial_apples = 75) 
  (h2 : handed_out_apples = 19) 
  (h3 : apples_per_pie = 8) : 
  pies = 7 :=
by
  rw [h1, h2, h3]
  sorry

end pies_from_apples_l213_213603


namespace range_of_a_l213_213570

theorem range_of_a (a : ℝ) : (∀ x : ℝ, (x - a) * (x + 1 - a) >= 0 → x ≠ 1) ↔ (1 < a ∧ a < 2) := 
sorry

end range_of_a_l213_213570


namespace one_fourth_in_one_eighth_l213_213427

theorem one_fourth_in_one_eighth : (1 / 8) / (1 / 4) = 1 / 2 := by
  sorry

end one_fourth_in_one_eighth_l213_213427


namespace total_pencils_l213_213100

variable (C Y M D : ℕ)

-- Conditions
def cheryl_has_thrice_as_cyrus (h1 : C = 3 * Y) : Prop := true
def madeline_has_half_of_cheryl (h2 : M = 63 ∧ C = 2 * M) : Prop := true
def daniel_has_25_percent_of_total (h3 : D = (C + Y + M) / 4) : Prop := true

-- Total number of pencils for all four
theorem total_pencils (h1 : C = 3 * Y) (h2 : M = 63 ∧ C = 2 * M) (h3 : D = (C + Y + M) / 4) :
  C + Y + M + D = 289 :=
by { sorry }

end total_pencils_l213_213100


namespace gcd_18_30_45_l213_213041

theorem gcd_18_30_45 : Nat.gcd (Nat.gcd 18 30) 45 = 3 :=
by
  sorry

end gcd_18_30_45_l213_213041


namespace smallest_angle_satisfying_trig_eqn_l213_213637

theorem smallest_angle_satisfying_trig_eqn :
  ∃ x : ℝ, 0 < x ∧ 8 * (Real.sin x)^2 * (Real.cos x)^4 - 8 * (Real.sin x)^4 * (Real.cos x)^2 = 1 ∧ x = 10 :=
by
  sorry

end smallest_angle_satisfying_trig_eqn_l213_213637


namespace number_of_children_l213_213371

-- Define conditions as per step A
def king_age := 35
def queen_age := 35
def num_sons := 3
def min_num_daughters := 1
def total_children_age_initial := 35
def max_num_children := 20

-- Equivalent Lean statement
theorem number_of_children 
  (king_age_eq : king_age = 35)
  (queen_age_eq : queen_age = 35)
  (num_sons_eq : num_sons = 3)
  (min_num_daughters_ge : min_num_daughters ≥ 1)
  (total_children_age_initial_eq : total_children_age_initial = 35)
  (max_num_children_le : max_num_children ≤ 20)
  (n : ℕ)
  (d : ℕ)
  (total_ages_eq : 70 + 2 * n = 35 + (d + 3) * n) :
  d + 3 = 7 ∨ d + 3 = 9 := sorry

end number_of_children_l213_213371


namespace solve_for_x_l213_213503

theorem solve_for_x (x : ℝ) (y : ℝ) (h : y = 1) (h_eq : y = 1 / (3 * x^2 + 2 * x + 1)) : x = 0 ∨ x = -2 / 3 :=
by
  sorry

end solve_for_x_l213_213503


namespace monotonicity_of_f_l213_213428

noncomputable def f (a x : ℝ) : ℝ := x^3 + a * x^2 + 1

theorem monotonicity_of_f (a x : ℝ) :
  (a > 0 → ((∀ x, (x < -2 * a / 3 → f a x' > f a x) ∧ (x > 0 → f a x' > f a x)) ∧ (∀ x, (-2 * a / 3 < x ∧ x < 0 → f a x' < f a x)))) ∧
  (a = 0 → ∀ x, f a x' > f a x) ∧
  (a < 0 → ((∀ x, (x < 0 → f a x' > f a x) ∧ (x > -2 * a / 3 → f a x' > f a x)) ∧ (∀ x, (0 < x ∧ x < -2 * a / 3 → f a x' < f a x)))) :=
sorry

end monotonicity_of_f_l213_213428


namespace total_nominal_income_l213_213106

theorem total_nominal_income
  (c1 : 8700 * ((1 + 0.06 / 12) ^ 6 - 1) = 264.28)
  (c2 : 8700 * ((1 + 0.06 / 12) ^ 5 - 1) = 219.69)
  (c3 : 8700 * ((1 + 0.06 / 12) ^ 4 - 1) = 175.31)
  (c4 : 8700 * ((1 + 0.06 / 12) ^ 3 - 1) = 131.15)
  (c5 : 8700 * ((1 + 0.06 / 12) ^ 2 - 1) = 87.22)
  (c6 : 8700 * (1 + 0.06 / 12 - 1) = 43.5) :
  264.28 + 219.69 + 175.31 + 131.15 + 87.22 + 43.5 = 921.15 := by
  sorry

end total_nominal_income_l213_213106


namespace engagement_ring_savings_l213_213409

theorem engagement_ring_savings 
  (yearly_salary : ℝ) 
  (monthly_savings : ℝ) 
  (monthly_salary := yearly_salary / 12) 
  (ring_cost := 2 * monthly_salary) 
  (saving_months := ring_cost / monthly_savings) 
  (h_salary : yearly_salary = 60000) 
  (h_savings : monthly_savings = 1000) :
  saving_months = 10 := 
sorry

end engagement_ring_savings_l213_213409


namespace trigonometric_expression_value_l213_213171

theorem trigonometric_expression_value (α : ℝ) (h : Real.tan α = 2) :
  (Real.sin α ^ 2 - Real.cos α ^ 2) / (2 * Real.sin α * Real.cos α + Real.cos α ^ 2) = 3 / 5 := 
sorry

end trigonometric_expression_value_l213_213171


namespace problem_equiv_proof_l213_213459

variable (a b : ℝ)
variable (h₁ : a > 0) (h₂ : b > 0) (h₃ : a + b = 1)

theorem problem_equiv_proof :
  (2 ^ a + 2 ^ b ≥ 2 * Real.sqrt 2) ∧
  (Real.log a / Real.log 2 + Real.log b / Real.log 2 ≤ -2) ∧
  (a ^ 2 + b ^ 2 ≥ 1 / 2) :=
by
  sorry

end problem_equiv_proof_l213_213459


namespace flag_yellow_area_percentage_l213_213646

theorem flag_yellow_area_percentage (s w : ℝ) (h_flag_area : s > 0)
  (h_width_positive : w > 0) (h_cross_area : 4 * s * w - 3 * w^2 = 0.49 * s^2) :
  (w^2 / s^2) * 100 = 12.25 :=
by
  sorry

end flag_yellow_area_percentage_l213_213646


namespace smallest_positive_difference_l213_213835

theorem smallest_positive_difference (a b : ℤ) (h : 17 * a + 6 * b = 13) : (∃ n : ℤ, n > 0 ∧ n = a - b) → n = 17 :=
by sorry

end smallest_positive_difference_l213_213835


namespace find_a_l213_213212

theorem find_a (a : ℝ) (ha : a ≠ 0)
  (h_area : (1/2) * (a/2) * a^2 = 2) :
  a = 2 ∨ a = -2 :=
sorry

end find_a_l213_213212


namespace shares_difference_l213_213376

noncomputable def Faruk_share (V : ℕ) : ℕ := (3 * (V / 5))
noncomputable def Ranjith_share (V : ℕ) : ℕ := (7 * (V / 5))

theorem shares_difference {V : ℕ} (hV : V = 1500) : 
  Ranjith_share V - Faruk_share V = 1200 :=
by
  rw [Faruk_share, Ranjith_share]
  subst hV
  -- It's just a declaration of the problem and sorry is used to skip the proof.
  sorry

end shares_difference_l213_213376


namespace sum_of_interior_angles_increases_l213_213442

theorem sum_of_interior_angles_increases (n : ℕ) (h : n ≥ 3) : (n-2) * 180 > (n-3) * 180 :=
by
  sorry

end sum_of_interior_angles_increases_l213_213442


namespace domain_of_g_l213_213862

def f : ℝ → ℝ := sorry  -- Placeholder for the function f

noncomputable def g (x : ℝ) : ℝ := f (x - 1) / Real.sqrt (2 * x + 1)

theorem domain_of_g :
  ∀ x : ℝ, g x ≠ 0 → (-1/2 < x ∧ x ≤ 3) :=
by
  intro x hx
  sorry

end domain_of_g_l213_213862


namespace range_of_k_l213_213036

theorem range_of_k (k : ℝ) :
  ∀ x : ℝ, ∃ a b c : ℝ, (a = k-1) → (b = -2) → (c = 1) → (a ≠ 0) → ((b^2 - 4 * a * c) ≥ 0) → k ≤ 2 ∧ k ≠ 1 :=
by
  sorry

end range_of_k_l213_213036


namespace find_m_l213_213560

-- Given the condition
def condition (m : ℕ) := (1 / 5 : ℝ)^m * (1 / 4 : ℝ)^2 = 1 / (10 : ℝ)^4

-- Theorem to prove that m is 4 given the condition
theorem find_m (m : ℕ) (h : condition m) : m = 4 :=
sorry

end find_m_l213_213560


namespace triangle_area_is_54_l213_213856

-- Define the sides of the triangle
def side1 : ℕ := 9
def side2 : ℕ := 12
def side3 : ℕ := 15

-- Verify that it is a right triangle using the Pythagorean theorem
def isRightTriangle (a b c : ℕ) : Prop := a * a + b * b = c * c

-- Define the area calculation for a right triangle
def areaRightTriangle (a b : ℕ) : ℕ := Nat.div (a * b) 2

-- State the theorem (Problem) to prove
theorem triangle_area_is_54 :
  isRightTriangle side1 side2 side3 ∧ areaRightTriangle side1 side2 = 54 :=
by
  sorry

end triangle_area_is_54_l213_213856


namespace monthly_average_decrease_rate_l213_213164

-- Conditions
def january_production : Float := 1.6 * 10^6
def march_production : Float := 0.9 * 10^6
def rate_decrease : Float := 0.25

-- Proof Statement: we need to prove that the monthly average decrease rate x = 0.25 satisfies the given condition
theorem monthly_average_decrease_rate :
  january_production * (1 - rate_decrease) * (1 - rate_decrease) = march_production := by
  sorry

end monthly_average_decrease_rate_l213_213164


namespace probability_MAME_on_top_l213_213490

theorem probability_MAME_on_top : 
  let num_sections := 8
  let favorable_outcome := 1
  (favorable_outcome : ℝ) / (num_sections : ℝ) = 1 / 8 :=
by 
  sorry

end probability_MAME_on_top_l213_213490


namespace combined_PPF_two_females_combined_PPF_two_males_combined_PPF_male_female_l213_213819

-- Definition: Combined PPF for two females
theorem combined_PPF_two_females (K : ℝ) (h : K ≤ 40) :
  (∀ K₁ K₂, K = K₁ + K₂ →  40 - 2 * K₁ + 40 - 2 * K₂ = 80 - 2 * K) := sorry

-- Definition: Combined PPF for two males
theorem combined_PPF_two_males (K : ℝ) (h : K ≤ 16) :
  (∀ K₁ K₂, K₁ = 0.5 * K → K₂ = 0.5 * K → 64 - K₁^2 + 64 - K₂^2 = 128 - 0.5 * K^2) := sorry

-- Definition: Combined PPF for one male and one female (piecewise)
theorem combined_PPF_male_female (K : ℝ) :
  (K ≤ 1 → (∀ K₁ K₂, K₁ = K → K₂ = 0 → 64 - K₁^2 + 40 - 2 * K₂ = 104 - K^2)) ∧
  (1 < K ∧ K ≤ 21 → (∀ K₁ K₂, K₁ = 1 → K₂ = K - 1 → 64 - K₁^2 + 40 - 2 * K₂ = 105 - 2 * K)) ∧
  (21 < K ∧ K ≤ 28 → (∀ K₁ K₂, K₁ = K - 20 → K₂ = 20 → 64 - K₁^2 + 40 - 2 * K₂ = 40 * K - K^2 - 336)) := sorry

end combined_PPF_two_females_combined_PPF_two_males_combined_PPF_male_female_l213_213819


namespace truck_distance_on_7_gallons_l213_213860

theorem truck_distance_on_7_gallons :
  ∀ (d : ℝ) (g₁ g₂ : ℝ), d = 240 → g₁ = 5 → g₂ = 7 → (d / g₁) * g₂ = 336 :=
by
  intros d g₁ g₂ h₁ h₂ h₃
  rw [h₁, h₂, h₃]
  sorry

end truck_distance_on_7_gallons_l213_213860


namespace inequality_ab2_bc2_ca2_leq_27_div_8_l213_213825

theorem inequality_ab2_bc2_ca2_leq_27_div_8 (a b c : ℝ) (h : a ≥ b) (h1 : b ≥ c) (h2 : c ≥ 0) (h3 : a + b + c = 3) :
  ab^2 + bc^2 + ca^2 ≤ 27 / 8 :=
sorry

end inequality_ab2_bc2_ca2_leq_27_div_8_l213_213825


namespace solve_frac_eq_l213_213185

-- Define the fractional function
def frac_eq (x : ℝ) : Prop := (x + 2) / (x - 1) = 0

-- State the theorem
theorem solve_frac_eq : frac_eq (-2) :=
by
  unfold frac_eq
  -- Use sorry to skip the proof
  sorry

end solve_frac_eq_l213_213185


namespace cos_three_pi_over_two_l213_213322

theorem cos_three_pi_over_two : Real.cos (3 * Real.pi / 2) = 0 :=
by
  -- Provided as correct by the solution steps role
  sorry

end cos_three_pi_over_two_l213_213322


namespace ahn_largest_number_l213_213629

def largest_number_ahn_can_get : ℕ :=
  let n := 10
  2 * (200 - n)

theorem ahn_largest_number :
  (10 ≤ 99) →
  (10 ≤ 99) →
  largest_number_ahn_can_get = 380 := 
by
-- Conditions: n is a two-digit integer with range 10 ≤ n ≤ 99
-- Proof is skipped
  sorry

end ahn_largest_number_l213_213629


namespace prob_at_least_6_heads_eq_l213_213703

-- define the number of coin flips
def n := 8

-- define the number of possible outcomes (2^n)
def total_outcomes := 2 ^ n

-- define the binomial coefficients for cases: 6 heads, 7 heads, 8 heads
def binom_8_6 := Nat.choose 8 6
def binom_8_7 := Nat.choose 8 7
def binom_8_8 := Nat.choose 8 8

-- calculate the favorable outcomes for at least 6 heads
def favorable_outcomes := binom_8_6 + binom_8_7 + binom_8_8

-- define the probability of getting at least 6 heads
def probability := (favorable_outcomes : ℚ) / total_outcomes

theorem prob_at_least_6_heads_eq : probability = 37 / 256 := by
  sorry

end prob_at_least_6_heads_eq_l213_213703


namespace least_possible_value_expression_l213_213373

theorem least_possible_value_expression :
  ∃ (x : ℝ), (x + 2) * (x + 3) * (x + 4) * (x + 5) + 2040 = 2039 :=
sorry

end least_possible_value_expression_l213_213373


namespace modular_home_total_cost_l213_213038

theorem modular_home_total_cost :
  let kitchen_sqft := 400
  let bathroom_sqft := 200
  let bedroom_sqft := 300
  let living_area_cost_per_sqft := 110
  let kitchen_cost := 28000
  let bathroom_cost := 12000
  let bedroom_cost := 18000
  let total_sqft := 3000
  let required_kitchens := 1
  let required_bathrooms := 2
  let required_bedrooms := 3
  let total_cost := required_kitchens * kitchen_cost +
                    required_bathrooms * bathroom_cost +
                    required_bedrooms * bedroom_cost +
                    (total_sqft - (required_kitchens * kitchen_sqft + required_bathrooms * bathroom_sqft + required_bedrooms * bedroom_sqft)) * living_area_cost_per_sqft
  total_cost = 249000 := 
by
  let kitchen_sqft := 400
  let bathroom_sqft := 200
  let bedroom_sqft := 300
  let living_area_cost_per_sqft := 110
  let kitchen_cost := 28000
  let bathroom_cost := 12000
  let bedroom_cost := 18000
  let total_sqft := 3000
  let required_kitchens := 1
  let required_bathrooms := 2
  let required_bedrooms := 3
  let total_cost := required_kitchens * kitchen_cost +
                    required_bathrooms * bathroom_cost +
                    required_bedrooms * bedroom_cost +
                    (total_sqft - (required_kitchens * kitchen_sqft + required_bathrooms * bathroom_sqft + required_bedrooms * bedroom_sqft)) * living_area_cost_per_sqft
  have h : total_cost = 249000 := sorry
  exact h

end modular_home_total_cost_l213_213038


namespace find_x_l213_213220

open Real

def a : ℝ × ℝ := (3, 4)
def b : ℝ × ℝ := (2, -1)

noncomputable def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

theorem find_x (x : ℝ) : 
  let ab := (a.1 + x * b.1, a.2 + x * b.2)
  let minus_b := (-b.1, -b.2)
  dot_product ab minus_b = 0 
  → x = -2 / 5 :=
by
  intros
  sorry

end find_x_l213_213220


namespace permutations_eq_factorial_l213_213511

theorem permutations_eq_factorial (n : ℕ) : 
  (∃ Pn : ℕ, Pn = n!) := 
sorry

end permutations_eq_factorial_l213_213511


namespace packs_of_yellow_balls_l213_213948

theorem packs_of_yellow_balls (Y : ℕ) : 
  3 * 19 + Y * 19 + 8 * 19 = 399 → Y = 10 :=
by sorry

end packs_of_yellow_balls_l213_213948


namespace X_investment_l213_213505

theorem X_investment (P : ℝ) 
  (Y_investment : ℝ := 42000)
  (Z_investment : ℝ := 48000)
  (Z_joins_at : ℝ := 4)
  (total_profit : ℝ := 14300)
  (Z_share : ℝ := 4160) :
  (P * 12 / (P * 12 + Y_investment * 12 + Z_investment * (12 - Z_joins_at))) * total_profit = Z_share → P = 35700 :=
by
  sorry

end X_investment_l213_213505


namespace speed_of_freight_train_l213_213521

-- Definitions based on the conditions
def distance := 390  -- The towns are 390 km apart
def express_speed := 80  -- The express train travels at 80 km per hr
def travel_time := 3  -- They pass one another 3 hr later

-- The freight train travels 30 km per hr slower than the express train
def freight_speed := express_speed - 30

-- The statement that we aim to prove:
theorem speed_of_freight_train : freight_speed = 50 := 
by 
  sorry

end speed_of_freight_train_l213_213521


namespace option_C_correct_l213_213995

theorem option_C_correct (a : ℤ) : (a = 3 → a = a + 1 → a = 4) :=
by {
  sorry
}

end option_C_correct_l213_213995


namespace range_of_m_for_subset_l213_213239

open Set

variable (m : ℝ)

def A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 5}
def B : Set ℝ := {x | (2 * m - 1) ≤ x ∧ x ≤ (m + 1)}

theorem range_of_m_for_subset (m : ℝ) : B m ⊆ A ↔ m ∈ Icc (-(1 / 2) : ℝ) (2 : ℝ) ∨ m > (2 : ℝ) :=
by
  sorry

end range_of_m_for_subset_l213_213239


namespace polynomial_term_count_l213_213973

open Nat

theorem polynomial_term_count (N : ℕ) (h : (N.choose 5) = 2002) : N = 17 :=
by
  sorry

end polynomial_term_count_l213_213973


namespace canteen_distances_l213_213883

theorem canteen_distances 
  (B G C : ℝ)
  (hB : B = 600)
  (hBG : G = 800)
  (hBC_eq_2x : ∃ x, C = 2 * x ∧ B = G + x + x) :
  G = 800 / 3 :=
by
  sorry

end canteen_distances_l213_213883


namespace compare_combined_sums_l213_213356

def numeral1 := 7524258
def numeral2 := 523625072

def place_value_2_numeral1 := 200000 + 20
def place_value_5_numeral1 := 50000 + 500
def combined_sum_numeral1 := place_value_2_numeral1 + place_value_5_numeral1

def place_value_2_numeral2 := 200000000 + 20
def place_value_5_numeral2 := 500000 + 50
def combined_sum_numeral2 := place_value_2_numeral2 + place_value_5_numeral2

def difference := combined_sum_numeral2 - combined_sum_numeral1

theorem compare_combined_sums :
  difference = 200249550 := by
  sorry

end compare_combined_sums_l213_213356


namespace find_a_l213_213858

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := 3^x + a / (3^x + 1)

theorem find_a (a : ℝ) : 
  (∀ x : ℝ, 3^x + a / (3^x + 1) ≥ 5) ∧ (∃ x : ℝ, 3^x + a / (3^x + 1) = 5) 
  → a = 9 := 
by 
  intro h
  sorry

end find_a_l213_213858


namespace solve_for_x_l213_213222

theorem solve_for_x :
  ∃ (x : ℝ), x ≠ 0 ∧ (5 * x)^10 = (10 * x)^5 ∧ x = 2 / 5 :=
by
  sorry

end solve_for_x_l213_213222


namespace inf_coprime_naturals_l213_213782

theorem inf_coprime_naturals (a b : ℤ) (h : a ≠ b) : 
  ∃ᶠ n in Filter.atTop, Nat.gcd (Int.natAbs (a + n)) (Int.natAbs (b + n)) = 1 := 
sorry

end inf_coprime_naturals_l213_213782


namespace greatest_product_obtainable_l213_213317

theorem greatest_product_obtainable :
  ∃ x : ℤ, ∃ y : ℤ, x + y = 300 ∧ x * y = 22500 :=
by
  sorry

end greatest_product_obtainable_l213_213317


namespace sum_of_fractions_and_decimal_l213_213308

theorem sum_of_fractions_and_decimal :
  (3 / 10) + (5 / 100) + (7 / 1000) + 0.001 = 0.358 :=
by
  sorry

end sum_of_fractions_and_decimal_l213_213308


namespace stripe_area_l213_213478

-- Definitions based on conditions
def diameter : ℝ := 40
def stripe_width : ℝ := 4
def revolutions : ℝ := 3

-- The statement we want to prove
theorem stripe_area (π : ℝ) : 
  (revolutions * π * diameter * stripe_width) = 480 * π :=
by
  sorry

end stripe_area_l213_213478


namespace katy_brownies_l213_213840

-- Define the conditions
def ate_monday : ℕ := 5
def ate_tuesday : ℕ := 2 * ate_monday

-- Define the question
def total_brownies : ℕ := ate_monday + ate_tuesday

-- State the proof problem
theorem katy_brownies : total_brownies = 15 := by
  sorry

end katy_brownies_l213_213840


namespace max_isosceles_tris_2017_gon_l213_213495

theorem max_isosceles_tris_2017_gon :
  ∀ (n : ℕ), n = 2017 →
  ∃ (t : ℕ), (∃ (d : ℕ), d = 2014 ∧ 2015 = (n - 2)) →
  t = 2010 :=
by
  sorry

end max_isosceles_tris_2017_gon_l213_213495


namespace initial_action_figures_l213_213582

theorem initial_action_figures (x : ℕ) (h : x + 4 - 1 = 6) : x = 3 :=
by {
  sorry
}

end initial_action_figures_l213_213582


namespace find_a_and_b_function_value_at_0_function_positive_x_less_than_7_over_6_l213_213042

def linear_function (a b x : ℝ) : ℝ := a * x + b

theorem find_a_and_b : ∃ (a b : ℝ), 
  linear_function a b 1 = 1 ∧ 
  linear_function a b 2 = -5 ∧ 
  a = -6 ∧ 
  b = 7 :=
sorry

theorem function_value_at_0 : 
  ∀ a b, 
  a = -6 → b = 7 → 
  linear_function a b 0 = 7 :=
sorry

theorem function_positive_x_less_than_7_over_6 :
  ∀ a b x, 
  a = -6 → b = 7 → 
  x < 7 / 6 → 
  linear_function a b x > 0 :=
sorry

end find_a_and_b_function_value_at_0_function_positive_x_less_than_7_over_6_l213_213042


namespace percent_correct_l213_213849

theorem percent_correct (x : ℕ) : 
  (5 * 100.0 / 7) = 71.43 :=
by
  sorry

end percent_correct_l213_213849


namespace apple_capacity_l213_213809

/-- Question: What is the largest possible number of apples that can be held by the 6 boxes and 4 extra trays?
 Conditions:
 - Paul has 6 boxes.
 - Each box contains 12 trays.
 - Paul has 4 extra trays.
 - Each tray can hold 8 apples.
 Answer: 608 apples
-/
theorem apple_capacity :
  let boxes := 6
  let trays_per_box := 12
  let extra_trays := 4
  let apples_per_tray := 8
  let total_trays := (boxes * trays_per_box) + extra_trays
  let total_apples_capacity := total_trays * apples_per_tray
  total_apples_capacity = 608 := 
by
  sorry

end apple_capacity_l213_213809


namespace remainder_division_x_squared_minus_one_l213_213110

variable (f g h : ℝ → ℝ)

noncomputable def remainder_when_divided_by_x_squared_minus_one (x : ℝ) : ℝ :=
-7 * x - 9

theorem remainder_division_x_squared_minus_one (h1 : ∀ x, f x = g x * (x - 1) + 8) (h2 : ∀ x, f x = h x * (x + 1) + 1) :
  ∀ x, f x % (x^2 - 1) = -7 * x - 9 :=
sorry

end remainder_division_x_squared_minus_one_l213_213110


namespace solve_hours_l213_213026

variable (x y : ℝ)

-- Conditions
def Condition1 : x > 0 := sorry
def Condition2 : y > 0 := sorry
def Condition3 : (2:ℝ) / 3 * y / x + (3 * x * y - 2 * y^2) / (3 * x) = x * y / (x + y) + 2 := sorry
def Condition4 : 2 * y / (x + y) = (3 * x - 2 * y) / (3 * x) := sorry

-- Question: How many hours would it take for A and B to complete the task alone?
theorem solve_hours : x = 6 ∧ y = 3 := 
by
  -- Use assumed conditions and variables to define the context
  have h1 := Condition1
  have h2 := Condition2
  have h3 := Condition3
  have h4 := Condition4
  -- Combine analytical relationship and solve for x and y 
  sorry

end solve_hours_l213_213026


namespace non_congruent_triangles_perimeter_18_l213_213248

theorem non_congruent_triangles_perimeter_18 :
  ∃ (triangles : Finset (Finset ℕ)), triangles.card = 11 ∧
  (∀ t ∈ triangles, t.card = 3 ∧ (∃ a b c : ℕ, t = {a, b, c} ∧ a + b + c = 18 ∧ a + b > c ∧ a + c > b ∧ b + c > a)) :=
sorry

end non_congruent_triangles_perimeter_18_l213_213248


namespace kimiko_watched_4_videos_l213_213752

/-- Kimiko's videos. --/
def first_video_length := 120
def second_video_length := 270
def last_two_video_length := 60
def total_time_watched := 510

theorem kimiko_watched_4_videos :
  first_video_length + second_video_length + last_two_video_length + last_two_video_length = total_time_watched → 
  4 = 4 :=
by
  intro h
  sorry

end kimiko_watched_4_videos_l213_213752


namespace jason_car_count_l213_213342

theorem jason_car_count :
  ∀ (red green purple total : ℕ),
  (green = 4 * red) →
  (red = purple + 6) →
  (purple = 47) →
  (total = purple + red + green) →
  total = 312 :=
by
  intros red green purple total h1 h2 h3 h4
  sorry

end jason_car_count_l213_213342


namespace perpendicular_and_intersection_l213_213791

variables (x y : ℚ)

def line1 := 4 * y - 3 * x = 15
def line4 := 3 * y + 4 * x = 15

theorem perpendicular_and_intersection :
  (4 * y - 3 * x = 15) ∧ (3 * y + 4 * x = 15) →
  let m1 := (3 : ℚ) / 4
  let m4 := -(4 : ℚ) / 3
  m1 * m4 = -1 ∧
  ∃ x y : ℚ, 4*y - 3*x = 15 ∧ 3*y + 4*x = 15 ∧ x = 15/32 ∧ y = 35/8 :=
by
  sorry

end perpendicular_and_intersection_l213_213791


namespace problem_statement_l213_213526

variable (a b c p q r α β γ : ℝ)

-- Given conditions
def plane_condition : Prop := (a / α) + (b / β) + (c / γ) = 1
def sphere_conditions : Prop := p^3 = α ∧ q^3 = β ∧ r^3 = γ

-- The statement to prove
theorem problem_statement (h_plane : plane_condition a b c α β γ) (h_sphere : sphere_conditions p q r α β γ) :
  (a / p^3) + (b / q^3) + (c / r^3) = 1 := sorry

end problem_statement_l213_213526


namespace age_of_teacher_l213_213524

/-- Given that the average age of 23 students is 22 years, and the average age increases
by 1 year when the teacher's age is included, prove that the teacher's age is 46 years. -/
theorem age_of_teacher (n : ℕ) (s_avg : ℕ) (new_avg : ℕ) (teacher_age : ℕ) :
  n = 23 →
  s_avg = 22 →
  new_avg = s_avg + 1 →
  teacher_age = new_avg * (n + 1) - s_avg * n →
  teacher_age = 46 :=
by
  intros h_n h_s_avg h_new_avg h_teacher_age
  sorry

end age_of_teacher_l213_213524


namespace thousands_digit_is_0_or_5_l213_213529

theorem thousands_digit_is_0_or_5 (n t : ℕ) (h₁ : n > 1000000) (h₂ : n % 40 = t) (h₃ : n % 625 = t) : 
  ((n / 1000) % 10 = 0) ∨ ((n / 1000) % 10 = 5) :=
sorry

end thousands_digit_is_0_or_5_l213_213529


namespace line_intercepts_of_3x_minus_y_plus_6_eq_0_l213_213625

theorem line_intercepts_of_3x_minus_y_plus_6_eq_0 :
  (∃ y, 3 * 0 - y + 6 = 0 ∧ y = 6) ∧ (∃ x, 3 * x - 0 + 6 = 0 ∧ x = -2) :=
by
  sorry

end line_intercepts_of_3x_minus_y_plus_6_eq_0_l213_213625


namespace trig_identity_evaluation_l213_213414

theorem trig_identity_evaluation :
  let θ1 := 70 * Real.pi / 180 -- angle 70 degrees in radians
  let θ2 := 10 * Real.pi / 180 -- angle 10 degrees in radians
  let θ3 := 20 * Real.pi / 180 -- angle 20 degrees in radians
  (Real.tan θ1 * Real.cos θ2 * (Real.sqrt 3 * Real.tan θ3 - 1) = -1) := 
by 
  sorry

end trig_identity_evaluation_l213_213414


namespace minimum_positive_period_of_f_is_pi_l213_213841

noncomputable def f (x : ℝ) : ℝ :=
  (Real.cos (x + Real.pi / 4))^2 - (Real.sin (x + Real.pi / 4))^2

theorem minimum_positive_period_of_f_is_pi :
  ∃ T > 0, (∀ x, f (x + T) = f x) ∧ (∀ T', T' > 0 ∧ (∀ x, f (x + T') = f x) → T' ≥ T) ∧ T = Real.pi :=
sorry

end minimum_positive_period_of_f_is_pi_l213_213841


namespace mean_goals_correct_l213_213834

-- Definitions based on problem conditions
def players_with_3_goals := 4
def players_with_4_goals := 3
def players_with_5_goals := 1
def players_with_6_goals := 2

-- The total number of goals scored
def total_goals := (3 * players_with_3_goals) + (4 * players_with_4_goals) + (5 * players_with_5_goals) + (6 * players_with_6_goals)

-- The total number of players
def total_players := players_with_3_goals + players_with_4_goals + players_with_5_goals + players_with_6_goals

-- The mean number of goals
def mean_goals := total_goals.toFloat / total_players.toFloat

theorem mean_goals_correct : mean_goals = 4.1 := by
  sorry

end mean_goals_correct_l213_213834


namespace derek_alice_pair_l213_213612

-- Variables and expressions involved
variable (x b c : ℝ)

-- Definitions of the conditions
def derek_eq := |x + 3| = 5 
def alice_eq := ∀ a, (a - 2) * (a + 8) = a^2 + b * a + c

-- The theorem to prove
theorem derek_alice_pair : derek_eq x → alice_eq b c → (b, c) = (6, -16) :=
by
  intros h1 h2
  sorry

end derek_alice_pair_l213_213612


namespace factor_poly1_factor_poly2_factor_poly3_l213_213064

-- Define the three polynomial functions.
def poly1 (x : ℝ) : ℝ := 2 * x^4 - 2
def poly2 (x : ℝ) : ℝ := x^4 - 18 * x^2 + 81
def poly3 (y : ℝ) : ℝ := (y^2 - 1)^2 + 11 * (1 - y^2) + 24

-- Formulate the goals: proving that each polynomial equals its respective factored form.
theorem factor_poly1 (x : ℝ) : poly1 x = 2 * (x^2 + 1) * (x + 1) * (x - 1) :=
sorry

theorem factor_poly2 (x : ℝ) : poly2 x = (x + 3)^2 * (x - 3)^2 :=
sorry

theorem factor_poly3 (y : ℝ) : poly3 y = (y + 2) * (y - 2) * (y + 3) * (y - 3) :=
sorry

end factor_poly1_factor_poly2_factor_poly3_l213_213064


namespace total_matches_played_l213_213326

-- Definitions
def victories_points := 3
def draws_points := 1
def defeats_points := 0
def points_after_5_games := 8
def games_played := 5
def target_points := 40
def remaining_wins_required := 9

-- Statement to prove
theorem total_matches_played :
  ∃ M : ℕ, points_after_5_games + victories_points * remaining_wins_required < target_points -> M = games_played + remaining_wins_required + 1 :=
sorry

end total_matches_played_l213_213326


namespace SufficientCondition_l213_213571

theorem SufficientCondition :
  ∀ x y z : ℤ, x = z ∧ y = x - 1 → x * (x - y) + y * (y - z) + z * (z - x) = 2 :=
by
  intros x y z h
  cases h with
  | intro h1 h2 =>
  sorry

end SufficientCondition_l213_213571


namespace max_tickets_l213_213545

-- Define the conditions
def ticket_cost (n : ℕ) : ℝ :=
  if n ≤ 6 then 15 * n
  else 13.5 * n

-- Define the main theorem
theorem max_tickets (budget : ℝ) : (∀ n : ℕ, ticket_cost n ≤ budget) → budget = 120 → n ≤ 8 :=
  by
  sorry

end max_tickets_l213_213545


namespace find_f_prime_one_l213_213908

noncomputable def f (x : ℝ) : ℝ := 1 / (1 + x)
def f_condition (x : ℝ) : Prop := f (1 / x) = x / (1 + x)

theorem find_f_prime_one : f_condition 1 → deriv f 1 = -1 / 4 := by
  intro h
  sorry

end find_f_prime_one_l213_213908


namespace arithmetic_sequence_general_formula_l213_213029

theorem arithmetic_sequence_general_formula (a : ℤ) :
  ∀ n : ℕ, n ≥ 1 → (∃ a_1 a_2 a_3 : ℤ, a_1 = a - 1 ∧ a_2 = a + 1 ∧ a_3 = a + 3) →
  (a + 2 * n - 3 = a - 1 + (n - 1) * 2) :=
by
  intros n hn h_exists
  rcases h_exists with ⟨a_1, a_2, a_3, h1, h2, h3⟩
  sorry

end arithmetic_sequence_general_formula_l213_213029


namespace even_n_has_parallel_pair_odd_n_cannot_have_exactly_one_parallel_pair_l213_213115

-- Definitions for the conditions in Lean 4
def regular_n_gon (n : ℕ) := true -- Dummy definition; actual geometric properties not needed for statement

def connected_path_visits_each_vertex_once (n : ℕ) := true -- Dummy definition; actual path properties not needed for statement

def parallel_pair (i j p q : ℕ) (n : ℕ) : Prop := (i + j) % n = (p + q) % n

-- Statements for part (a) and (b)

theorem even_n_has_parallel_pair (n : ℕ) (h_even : n % 2 = 0) 
  (h_path : connected_path_visits_each_vertex_once n) : 
  ∃ (i j p q : ℕ), i ≠ p ∧ j ≠ q ∧ parallel_pair i j p q n := 
sorry

theorem odd_n_cannot_have_exactly_one_parallel_pair (n : ℕ) (h_odd : n % 2 = 1) 
  (h_path : connected_path_visits_each_vertex_once n) : 
  ¬∃ (i j p q : ℕ), i ≠ p ∧ j ≠ q ∧ parallel_pair i j p q n ∧ 
  (∀ (i' j' p' q' : ℕ), (i' ≠ p' ∨ j' ≠ q') → ¬parallel_pair i' j' p' q' n) := 
sorry

end even_n_has_parallel_pair_odd_n_cannot_have_exactly_one_parallel_pair_l213_213115


namespace part1_part2_l213_213504

def divides (a b : ℕ) := ∃ k : ℕ, b = k * a

def fibonacci : ℕ → ℕ
| 0 => 0
| 1 => 1
| (n+2) => fibonacci (n+1) + fibonacci n

theorem part1 (m n : ℕ) (h : divides m n) : divides (fibonacci m) (fibonacci n) :=
sorry

theorem part2 (m n : ℕ) : Nat.gcd (fibonacci m) (fibonacci n) = fibonacci (Nat.gcd m n) :=
sorry

end part1_part2_l213_213504


namespace problem_eq_l213_213122

theorem problem_eq : 
  ∀ x y : ℝ, x ≠ 0 → y ≠ 0 → y = x / (x + 1) → (x - y + 4 * x * y) / (x * y) = 5 :=
by
  intros x y hx hnz hyxy
  sorry

end problem_eq_l213_213122


namespace man_age_difference_l213_213830

theorem man_age_difference (S M : ℕ) (h1 : S = 22) (h2 : M + 2 = 2 * (S + 2)) : M - S = 24 :=
by
  sorry

end man_age_difference_l213_213830


namespace find_angle_l213_213337

-- Define the conditions
variables (x : ℝ)

-- Conditions given in the problem
def angle_complement_condition (x : ℝ) := (10 : ℝ) + 3 * x
def complementary_condition (x : ℝ) := x + angle_complement_condition x = 90

-- Prove that the angle x equals to 20 degrees
theorem find_angle : (complementary_condition x) → x = 20 := 
by
  -- Placeholder for the proof
  sorry

end find_angle_l213_213337


namespace four_consecutive_even_impossible_l213_213344

def is_four_consecutive_even_sum (S : ℕ) : Prop :=
  ∃ n : ℤ, S = 4 * n + 12

theorem four_consecutive_even_impossible :
  ¬ is_four_consecutive_even_sum 34 :=
by
  sorry

end four_consecutive_even_impossible_l213_213344


namespace smallest_four_digit_divisible_by_53_l213_213651

theorem smallest_four_digit_divisible_by_53 : ∃ n : ℕ, 1000 ≤ n ∧ n ≤ 9999 ∧ n % 53 = 0 ∧ ∀ m : ℕ, (1000 ≤ m ∧ m ≤ 9999 ∧ m % 53 = 0) → n ≤ m :=
by
  sorry

end smallest_four_digit_divisible_by_53_l213_213651


namespace find_f_neg2003_l213_213412

noncomputable def f (x : ℝ) : ℝ := sorry

theorem find_f_neg2003 (f_defined : ∀ x : ℝ, ∃ y : ℝ, f y = x → f y ≠ 0)
  (cond1 : ∀ ⦃x y w : ℝ⦄, x > y → (f x + x ≥ w → w ≥ f y + y → ∃ z, y ≤ z ∧ z ≤ x ∧ f z = w - z))
  (cond2 : ∃ u : ℝ, f u = 0 ∧ ∀ v : ℝ, f v = 0 → u ≤ v)
  (cond3 : f 0 = 1)
  (cond4 : f (-2003) ≤ 2004)
  (cond5 : ∀ x y : ℝ, f x * f y = f (x * f y + y * f x + x * y)) :
  f (-2003) = 2004 :=
sorry

end find_f_neg2003_l213_213412


namespace arithmetic_sequence_an_12_l213_213853

theorem arithmetic_sequence_an_12 {a : ℕ → ℝ} (h_arith : ∀ n, a (n + 1) - a n = a 1 - a 0)
  (h_a3 : a 3 = 9)
  (h_a6 : a 6 = 15) :
  a 12 = 27 := 
sorry

end arithmetic_sequence_an_12_l213_213853


namespace difference_of_squares_l213_213911

theorem difference_of_squares (x y : ℝ) (h1 : x + y = 10) (h2 : x - y = 19) : x^2 - y^2 = 190 :=
by
  sorry

end difference_of_squares_l213_213911


namespace base7_addition_problem_l213_213936

theorem base7_addition_problem
  (X Y : ℕ) :
  (5 * 7^1 + X * 7^0 + Y * 7^0 + 0 * 7^2 + 6 * 7^1 + 2 * 7^0) = (6 * 7^1 + 4 * 7^0 + X * 7^0 + 0 * 7^2) →
  X + 6 = 1 * 7 + 4 →
  Y + 2 = X →
  X + Y = 8 :=
by
  intro h1 h2 h3
  sorry

end base7_addition_problem_l213_213936


namespace buttons_pattern_total_buttons_sum_l213_213801

-- Define the sequence of the number of buttons in each box
def buttons_in_box (n : ℕ) : ℕ := 3^(n-1)

-- Define the sum of buttons up to the n-th box
def total_buttons (n : ℕ) : ℕ := (3^n - 1) / 2

-- Theorem statements to prove
theorem buttons_pattern (n : ℕ) : buttons_in_box n = 3^(n-1) := by
  sorry

theorem total_buttons_sum (n : ℕ) : total_buttons n = (3^n - 1) / 2 := by
  sorry

end buttons_pattern_total_buttons_sum_l213_213801


namespace fraction_product_l213_213960

theorem fraction_product : (2 / 9) * (5 / 11) = 10 / 99 := 
by
  sorry

end fraction_product_l213_213960


namespace jayda_spending_l213_213306

theorem jayda_spending
  (J A : ℝ)
  (h1 : A = J + (2/5) * J)
  (h2 : J + A = 960) :
  J = 400 :=
by
  sorry

end jayda_spending_l213_213306


namespace complex_square_l213_213163

theorem complex_square (a b : ℝ) (i : ℂ) (h1 : a + b * i - 2 * i = 2 - b * i) : 
  (a + b * i) ^ 2 = 3 + 4 * i := 
by {
  -- Proof steps skipped (using sorry to indicate proof is required)
  sorry
}

end complex_square_l213_213163


namespace power_sum_is_99_l213_213979

theorem power_sum_is_99 : 3^4 + (-3)^3 + (-3)^2 + (-3)^1 + 3^1 + 3^2 + 3^3 = 99 :=
by sorry

end power_sum_is_99_l213_213979


namespace inequality_solution_l213_213044

theorem inequality_solution (a : ℝ) (h : 1 < a) : ∀ x : ℝ, a ^ (2 * x + 1) > (1 / a) ^ (2 * x) ↔ x > -1 / 4 :=
by
  sorry

end inequality_solution_l213_213044


namespace distance_PQ_parallel_x_max_distance_PQ_l213_213784

open Real

def parabola (x : ℝ) : ℝ := x^2

/--
1. When PQ is parallel to the x-axis, find the distance from point O to PQ.
-/
theorem distance_PQ_parallel_x (m : ℝ) (h₁ : m ≠ 0) (h₂ : parabola m = 1) : 
  ∃ d : ℝ, d = 1 := by
  sorry

/--
2. Find the maximum value of the distance from point O to PQ.
-/
theorem max_distance_PQ (a b : ℝ) (h₁ : a * b = -1) (h₂ : ∀ x, ∃ y, y = a * x + b) :
  ∃ d : ℝ, d = 1 := by
  sorry

end distance_PQ_parallel_x_max_distance_PQ_l213_213784


namespace raghu_investment_l213_213435

theorem raghu_investment (R T V : ℝ) (h1 : T = 0.9 * R) (h2 : V = 1.1 * T) (h3 : R + T + V = 5780) : R = 2000 :=
by
  sorry

end raghu_investment_l213_213435


namespace ravi_nickels_l213_213048

variables (n q d : ℕ)

-- Defining the conditions
def quarters (n : ℕ) : ℕ := n + 2
def dimes (q : ℕ) : ℕ := q + 4

-- Using these definitions to form the Lean theorem
theorem ravi_nickels : 
  ∃ n, q = quarters n ∧ d = dimes q ∧ 
  (0.05 * n + 0.25 * q + 0.10 * d : ℝ) = 3.50 ∧ n = 6 :=
sorry

end ravi_nickels_l213_213048


namespace necessary_not_sufficient_l213_213803

-- Define the function y = x^2 - 2ax + 1
def quadratic_function (a x : ℝ) : ℝ := x^2 - 2 * a * x + 1

-- Define strict monotonicity on the interval [1, +∞)
def strictly_increasing_on (f : ℝ → ℝ) (I : Set ℝ) : Prop :=
  ∀ ⦃x y⦄, x ∈ I → y ∈ I → x < y → f x < f y

-- Define the condition for the function to be strictly increasing on [1, +∞)
def condition_strict_increasing (a : ℝ) : Prop :=
  strictly_increasing_on (quadratic_function a) (Set.Ici 1)

-- The condition to prove
theorem necessary_not_sufficient (a : ℝ) :
  condition_strict_increasing a → (a ≤ 0) := sorry

end necessary_not_sufficient_l213_213803


namespace missing_pieces_l213_213047

-- Definitions based on the conditions.
def total_pieces : ℕ := 500
def border_pieces : ℕ := 75
def trevor_pieces : ℕ := 105
def joe_pieces : ℕ := 3 * trevor_pieces

-- Prove the number of missing pieces is 5.
theorem missing_pieces : total_pieces - (border_pieces + trevor_pieces + joe_pieces) = 5 := by
  sorry

end missing_pieces_l213_213047


namespace complex_division_simplification_l213_213674

theorem complex_division_simplification (i : ℂ) (h_i : i * i = -1) : (1 - 3 * i) / (2 - i) = 1 - i := by
  sorry

end complex_division_simplification_l213_213674


namespace circular_garden_area_l213_213931

theorem circular_garden_area (r : ℝ) (A C : ℝ) (h_radius : r = 6) (h_relationship : C = (1 / 3) * A) 
  (h_circumference : C = 2 * Real.pi * r) (h_area : A = Real.pi * r ^ 2) : 
  A = 36 * Real.pi :=
by
  sorry

end circular_garden_area_l213_213931


namespace inequality_proof_l213_213430

theorem inequality_proof (a b : ℝ) (h₁ : a ≥ b) (h₂ : b > 0) : 
  2 * a ^ 3 - b ^ 3 ≥ 2 * a * b ^ 2 - a ^ 2 * b := 
by
  sorry

end inequality_proof_l213_213430


namespace payment_methods_20_yuan_l213_213638

theorem payment_methods_20_yuan :
  let ten_yuan_note := 10
  let five_yuan_note := 5
  let one_yuan_note := 1
  ∃ (methods : Nat), 
    methods = 9 ∧ 
    ∃ (num_10 num_5 num_1 : Nat),
      (num_10 * ten_yuan_note + num_5 * five_yuan_note + num_1 * one_yuan_note = 20) →
      methods = 9 :=
sorry

end payment_methods_20_yuan_l213_213638


namespace physics_experiment_l213_213049

theorem physics_experiment (x : ℕ) (h : 1 + x + (x + 1) * x = 36) :
  1 + x + (x + 1) * x = 36 :=
  by                        
  exact h

end physics_experiment_l213_213049


namespace archers_in_golden_l213_213510

variables (soldiers archers swordsmen wearing_golden wearing_black : ℕ)
variables (truth_swordsmen_black lie_swordsmen_golden lie_archers_black truth_archers_golden : ℕ)

-- Given conditions
variables (cond1 : archers + swordsmen = 55)
variables (cond2 : wearing_golden + wearing_black = 55)
variables (cond3 : truth_swordsmen_black + lie_swordsmen_golden + lie_archers_black + truth_archers_golden = 55)
variables (cond4 : wearing_golden = 44)
variables (cond5 : archers = 33)
variables (cond6 : truth_swordsmen_black + lie_archers_black = 22)

-- Define the mathematic equivalent proof problem
theorem archers_in_golden : archers = 22 :=
by
  sorry

end archers_in_golden_l213_213510


namespace sum_of_converted_2016_is_correct_l213_213406

theorem sum_of_converted_2016_is_correct :
  (20.16 + 20.16 + 20.16 + 201.6 + 201.6 + 201.6 = 463.68 ∨
   2.016 + 2.016 + 2.016 + 20.16 + 20.16 + 20.16 = 46.368) :=
by
  sorry

end sum_of_converted_2016_is_correct_l213_213406


namespace probability_of_neither_is_correct_l213_213131

-- Definitions of the given conditions
def total_buyers : ℕ := 100
def cake_buyers : ℕ := 50
def muffin_buyers : ℕ := 40
def both_cake_and_muffin_buyers : ℕ := 19

-- Define the probability calculation function
def probability_neither (total : ℕ) (cake : ℕ) (muffin : ℕ) (both : ℕ) : ℚ :=
  let buyers_neither := total - (cake + muffin - both)
  (buyers_neither : ℚ) / (total : ℚ)

-- State the main theorem to ensure it is equivalent to our mathematical problem
theorem probability_of_neither_is_correct :
  probability_neither total_buyers cake_buyers muffin_buyers both_cake_and_muffin_buyers = 0.29 := 
sorry

end probability_of_neither_is_correct_l213_213131


namespace find_a_in_terms_of_y_l213_213619

theorem find_a_in_terms_of_y (a b y : ℝ) (h1 : a ≠ b) (h2 : a^3 - b^3 = 27 * y^3) (h3 : a - b = 3 * y) :
  a = 3 * y :=
sorry

end find_a_in_terms_of_y_l213_213619


namespace factor_2310_two_digit_numbers_l213_213659

theorem factor_2310_two_digit_numbers :
  ∃ (a b : ℕ), 10 ≤ a ∧ a < 100 ∧ 10 ≤ b ∧ b < 100 ∧ a * b = 2310 ∧ ∀ (c d : ℕ), 10 ≤ c ∧ c < 100 ∧ 10 ≤ d ∧ d < 100 ∧ c * d = 2310 → (c = a ∧ d = b) ∨ (c = b ∧ d = a) :=
by {
  sorry
}

end factor_2310_two_digit_numbers_l213_213659


namespace coplanar_condition_l213_213591

-- Definitions representing points A, B, C, D and the origin O in a vector space over the reals
variables {V : Type*} [AddCommGroup V] [Module ℝ V] (O A B C D : V)

-- The main statement of the problem
theorem coplanar_condition (h : (2 : ℝ) • (A - O) - (3 : ℝ) • (B - O) + (7 : ℝ) • (C - O) + k • (D - O) = 0) :
  k = -6 :=
sorry

end coplanar_condition_l213_213591


namespace triangle_rectangle_ratio_l213_213747

-- Definitions of the perimeter conditions and the relationship between length and width of the rectangle.
def equilateral_triangle_side_length (t : ℕ) : Prop :=
  3 * t = 24

def rectangle_dimensions (l w : ℕ) : Prop :=
  2 * l + 2 * w = 24 ∧ l = 2 * w

-- The main theorem stating the desired ratio.
theorem triangle_rectangle_ratio (t l w : ℕ) 
  (ht : equilateral_triangle_side_length t) (hlw : rectangle_dimensions l w) : t / w = 2 :=
by
  sorry

end triangle_rectangle_ratio_l213_213747


namespace sum_of_coefficients_l213_213799

-- Define the polynomial P(x)
def P (x : ℝ) : ℝ := 5 * (2 * x^8 - 3 * x^5 + 9 * x^3 - 6) + 4 * (7 * x^6 - 2 * x^3 + 8)

-- Statement to prove that the sum of the coefficients of P(x) is 62
theorem sum_of_coefficients : P 1 = 62 := sorry

end sum_of_coefficients_l213_213799


namespace d_is_greatest_l213_213251

variable (p : ℝ)

def a := p - 1
def b := p + 2
def c := p - 3
def d := p + 4

theorem d_is_greatest : d > b ∧ d > a ∧ d > c := 
by sorry

end d_is_greatest_l213_213251


namespace ratio_of_black_to_white_tiles_l213_213789

theorem ratio_of_black_to_white_tiles
  (original_width : ℕ)
  (original_height : ℕ)
  (original_black_tiles : ℕ)
  (original_white_tiles : ℕ)
  (border_width : ℕ)
  (border_height : ℕ)
  (extended_width : ℕ)
  (extended_height : ℕ)
  (new_white_tiles : ℕ)
  (total_white_tiles : ℕ)
  (total_black_tiles : ℕ)
  (ratio_black_to_white : ℚ)
  (h1 : original_width = 5)
  (h2 : original_height = 6)
  (h3 : original_black_tiles = 12)
  (h4 : original_white_tiles = 18)
  (h5 : border_width = 1)
  (h6 : border_height = 1)
  (h7 : extended_width = original_width + 2 * border_width)
  (h8 : extended_height = original_height + 2 * border_height)
  (h9 : new_white_tiles = (extended_width * extended_height) - (original_width * original_height))
  (h10 : total_white_tiles = original_white_tiles + new_white_tiles)
  (h11 : total_black_tiles = original_black_tiles)
  (h12 : ratio_black_to_white = total_black_tiles / total_white_tiles) :
  ratio_black_to_white = 3 / 11 := 
sorry

end ratio_of_black_to_white_tiles_l213_213789


namespace probability_points_one_unit_apart_l213_213532

theorem probability_points_one_unit_apart :
  let points := 10
  let rect_length := 3
  let rect_width := 2
  let total_pairs := (points * (points - 1)) / 2
  let favorable_pairs := 10  -- derived from solution steps
  (favorable_pairs / total_pairs : ℚ) = (2 / 9 : ℚ) :=
by
  sorry

end probability_points_one_unit_apart_l213_213532


namespace average_weight_of_B_C_D_E_l213_213623

theorem average_weight_of_B_C_D_E 
    (W_A W_B W_C W_D W_E : ℝ)
    (h1 : (W_A + W_B + W_C)/3 = 60)
    (h2 : W_A = 87)
    (h3 : (W_A + W_B + W_C + W_D)/4 = 65)
    (h4 : W_E = W_D + 3) :
    (W_B + W_C + W_D + W_E)/4 = 64 :=
by {
    sorry
}

end average_weight_of_B_C_D_E_l213_213623


namespace neg_cos_ge_a_l213_213912

theorem neg_cos_ge_a (a : ℝ) : (¬ ∃ x : ℝ, Real.cos x ≥ a) ↔ a = 2 := 
sorry

end neg_cos_ge_a_l213_213912


namespace sequence_general_term_l213_213197

theorem sequence_general_term (a : ℕ+ → ℤ) (h₁ : a 1 = 2) (h₂ : ∀ n : ℕ+, a (n + 1) = a n - 1) :
  ∀ n : ℕ+, a n = 3 - n := 
sorry

end sequence_general_term_l213_213197


namespace slope_of_intersection_points_l213_213866

theorem slope_of_intersection_points : 
  (∀ t : ℝ, ∃ x y : ℝ, (2 * x + 3 * y = 10 * t + 4) ∧ (x + 4 * y = 3 * t + 3)) → 
  (∀ t1 t2 : ℝ, t1 ≠ t2 → ((2 * ((10 * t1 + 4)  / 2) + 3 * ((-5/3 * t1 - 2/3)) = (10 * t1 + 4)) ∧ (2 * ((10 * t2 + 4) / 2) + 3 * ((-5/3 * t2 - 2/3)) = (10 * t2 + 4))) → 
  (31 * (((-5/3 * t1 - 2/3) - (-5/3 * t2 - 2/3)) / ((10 * t1 + 4) / 2 - (10 * t2 + 4) / 2)) = -4)) :=
sorry

end slope_of_intersection_points_l213_213866


namespace vitamin_C_in_apple_juice_l213_213359

theorem vitamin_C_in_apple_juice (A O : ℝ) 
  (h₁ : A + O = 185) 
  (h₂ : 2 * A + 3 * O = 452) :
  A = 103 :=
sorry

end vitamin_C_in_apple_juice_l213_213359


namespace moles_of_silver_nitrate_needed_l213_213541

structure Reaction :=
  (reagent1 : String)
  (reagent2 : String)
  (product1 : String)
  (product2 : String)
  (ratio_reagent1_to_product2 : ℕ) -- Moles of reagent1 to product2 in the balanced reaction

def silver_nitrate_hydrochloric_acid_reaction : Reaction :=
  { reagent1 := "AgNO3",
    reagent2 := "HCl",
    product1 := "AgCl",
    product2 := "HNO3",
    ratio_reagent1_to_product2 := 1 }

theorem moles_of_silver_nitrate_needed
  (reaction : Reaction)
  (hCl_initial_moles : ℕ)
  (hno3_target_moles : ℕ) :
  hno3_target_moles = 2 →
  (reaction.ratio_reagent1_to_product2 = 1 ∧ hCl_initial_moles = 2) →
  (hno3_target_moles = reaction.ratio_reagent1_to_product2 * 2 ∧ hno3_target_moles = 2) :=
by
  sorry

end moles_of_silver_nitrate_needed_l213_213541


namespace no_solution_to_system_l213_213730

theorem no_solution_to_system :
  ¬ ∃ (x y : ℝ), (3 * x - 4 * y = 8) ∧ (6 * x - 8 * y = 18) :=
by
  sorry

end no_solution_to_system_l213_213730


namespace new_lamp_height_is_correct_l213_213476

-- Define the height of the old lamp
def old_lamp_height : ℝ := 1

-- Define the additional height of the new lamp
def additional_height : ℝ := 1.33

-- Proof statement
theorem new_lamp_height_is_correct :
  old_lamp_height + additional_height = 2.33 :=
sorry

end new_lamp_height_is_correct_l213_213476


namespace student_count_l213_213210

noncomputable def numberOfStudents (decreaseInAverageWeight totalWeightDecrease : ℕ) : ℕ :=
  totalWeightDecrease / decreaseInAverageWeight

theorem student_count 
  (decreaseInAverageWeight : ℕ)
  (totalWeightDecrease : ℕ)
  (condition_avg_weight_decrease : decreaseInAverageWeight = 4)
  (condition_weight_difference : totalWeightDecrease = 92 - 72) :
  numberOfStudents decreaseInAverageWeight totalWeightDecrease = 5 := by 
  -- We are not providing the proof details as per the instruction
  sorry

end student_count_l213_213210


namespace find_unknown_number_l213_213214

def op (a b : ℝ) := a * (b ^ (1 / 2))

theorem find_unknown_number (x : ℝ) (h : op 4 x = 12) : x = 9 :=
by
  sorry

end find_unknown_number_l213_213214


namespace cricket_average_l213_213166

theorem cricket_average (x : ℝ) (h1 : 15 * x + 121 = 16 * (x + 6)) : x = 25 := by
  -- proof goes here, but we skip it with sorry
  sorry

end cricket_average_l213_213166


namespace gcd_115_161_l213_213954

theorem gcd_115_161 : Nat.gcd 115 161 = 23 := by
  sorry

end gcd_115_161_l213_213954


namespace find_dividend_l213_213273

theorem find_dividend (x D : ℕ) (q r : ℕ) (h_q : q = 4) (h_r : r = 3)
  (h_div : D = x * q + r) (h_sum : D + x + q + r = 100) : D = 75 :=
by
  sorry

end find_dividend_l213_213273


namespace alex_avg_speed_l213_213315

theorem alex_avg_speed (v : ℝ) : 
  (4.5 * v + 2.5 * 12 + 1.5 * 24 + 8 = 164) → v = 20 := 
by 
  intro h
  sorry

end alex_avg_speed_l213_213315


namespace exists_three_numbers_sum_to_zero_l213_213837

theorem exists_three_numbers_sum_to_zero (s : Finset ℤ) (h_card : s.card = 101) (h_abs : ∀ x ∈ s, |x| ≤ 99) :
  ∃ (a b c : ℤ), a ∈ s ∧ b ∈ s ∧ c ∈ s ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a + b + c = 0 :=
by {
  sorry
}

end exists_three_numbers_sum_to_zero_l213_213837


namespace ingrid_income_l213_213012

theorem ingrid_income (I : ℝ) (h1 : 0.30 * 56000 = 16800) 
  (h2 : ∀ (I : ℝ), 0.40 * I = 0.4 * I) 
  (h3 : 0.35625 * (56000 + I) = 16800 + 0.4 * I) : 
  I = 49142.86 := 
by 
  sorry

end ingrid_income_l213_213012


namespace dentist_ratio_l213_213055

-- Conditions
def cost_cleaning : ℕ := 70
def cost_filling : ℕ := 120
def cost_extraction : ℕ := 290

-- Theorem statement
theorem dentist_ratio : (cost_cleaning + 2 * cost_filling + cost_extraction) / cost_filling = 5 := 
by
  -- To be proven
  sorry

end dentist_ratio_l213_213055


namespace cost_of_green_shirts_l213_213656

noncomputable def total_cost_kindergarten : ℝ := 101 * 5.8
noncomputable def total_cost_first_grade : ℝ := 113 * 5
noncomputable def total_cost_second_grade : ℝ := 107 * 5.6
noncomputable def total_cost_all_but_third : ℝ := total_cost_kindergarten + total_cost_first_grade + total_cost_second_grade
noncomputable def total_third_grade : ℝ := 2317 - total_cost_all_but_third
noncomputable def cost_per_third_grade_shirt : ℝ := total_third_grade / 108

theorem cost_of_green_shirts : cost_per_third_grade_shirt = 5.25 := sorry

end cost_of_green_shirts_l213_213656


namespace sum_of_possible_values_l213_213323

theorem sum_of_possible_values (N : ℝ) (h : N * (N - 4) = -21) :
    ∃ N₁ N₂ : ℝ, (N₁ + N₂ = 4 ∧ N₁ * (N₁ - 4) = -21 ∧ N₂ * (N₂ - 4) = -21) :=
sorry

end sum_of_possible_values_l213_213323


namespace elena_probability_at_least_one_correct_l213_213592

-- Conditions
def total_questions := 30
def choices_per_question := 4
def guessed_questions := 6
def incorrect_probability_single := 3 / 4

-- Expression for the probability of missing all guessed questions
def probability_all_incorrect := (incorrect_probability_single) ^ guessed_questions

-- Calculation from the solution
def probability_at_least_one_correct := 1 - probability_all_incorrect

-- Problem statement to prove
theorem elena_probability_at_least_one_correct : probability_at_least_one_correct = 3367 / 4096 :=
by sorry

end elena_probability_at_least_one_correct_l213_213592


namespace turtles_remaining_on_log_l213_213389
-- Importing necessary modules

-- Defining the problem
def initial_turtles : ℕ := 9
def turtles_climbed : ℕ := (initial_turtles * 3) - 2
def total_turtles : ℕ := initial_turtles + turtles_climbed
def remaining_turtles : ℕ := total_turtles / 2

-- Stating the proof problem
theorem turtles_remaining_on_log : remaining_turtles = 17 := 
  sorry

end turtles_remaining_on_log_l213_213389


namespace original_amount_of_cooking_oil_l213_213283

theorem original_amount_of_cooking_oil (X : ℝ) (H : (2 / 5 * X + 300) + (1 / 2 * (X - (2 / 5 * X + 300)) - 200) + 800 = X) : X = 2500 :=
by simp at H; linarith

end original_amount_of_cooking_oil_l213_213283


namespace simplified_sum_l213_213718

theorem simplified_sum :
  (-1 : ℤ) ^ 2002 + (-1 : ℤ) ^ 2003 + 2 ^ 2004 - 2 ^ 2003 = 2 ^ 2003 := 
by 
  sorry -- Proof skipped

end simplified_sum_l213_213718


namespace divides_sequence_l213_213151

theorem divides_sequence (a : ℕ → ℕ) (n k: ℕ) (h0 : a 0 = 0) (h1 : a 1 = 1) 
  (hrec : ∀ m, a (m + 2) = 2 * a (m + 1) + a m) :
  (2^k ∣ a n) ↔ (2^k ∣ n) :=
sorry

end divides_sequence_l213_213151


namespace total_material_ordered_l213_213776

theorem total_material_ordered :
  12.468 + 4.6278 + 7.9101 + 8.3103 + 5.6327 = 38.9499 :=
by
  sorry

end total_material_ordered_l213_213776


namespace principal_trebled_after_5_years_l213_213396

theorem principal_trebled_after_5_years (P R: ℝ) (n: ℝ) :
  (P * R * 10 / 100 = 700) →
  ((P * R * n + 3 * P * R * (10 - n)) / 100 = 1400) →
  n = 5 :=
by
  intros h1 h2
  sorry

end principal_trebled_after_5_years_l213_213396


namespace max_value_of_trig_expr_l213_213123

variable (x : ℝ)

theorem max_value_of_trig_expr : 
  (∀ x : ℝ, 3 * Real.cos x + 4 * Real.sin x ≤ 5) ∧
  (∃ x : ℝ, 3 * Real.cos x + 4 * Real.sin x = 5) :=
sorry

end max_value_of_trig_expr_l213_213123


namespace nancy_crystal_beads_l213_213843

-- Definitions of given conditions
def price_crystal : ℕ := 9
def price_metal : ℕ := 10
def sets_metal : ℕ := 2
def total_spent : ℕ := 29

-- Statement of the proof problem
theorem nancy_crystal_beads : ∃ x : ℕ, price_crystal * x + price_metal * sets_metal = total_spent ∧ x = 1 := by
  sorry

end nancy_crystal_beads_l213_213843


namespace problem_evaluation_l213_213792

theorem problem_evaluation : (726 * 726) - (725 * 727) = 1 := 
by 
  sorry

end problem_evaluation_l213_213792


namespace height_of_table_l213_213564

/-- 
Given:
1. Combined initial measurement (l + h - w + t) = 40
2. Combined changed measurement (w + h - l + t) = 34
3. Width of each wood block (w) = 6 inches
4. Visible edge-on thickness of the table (t) = 4 inches
Prove:
The height of the table (h) is 33 inches.
-/
theorem height_of_table (l h t w : ℕ) (h_combined_initial : l + h - w + t = 40)
    (h_combined_changed : w + h - l + t = 34) (h_width : w = 6) (h_thickness : t = 4) : 
    h = 33 :=
by
  sorry

end height_of_table_l213_213564


namespace Joey_SAT_Weeks_l213_213155

theorem Joey_SAT_Weeks
    (hours_per_night : ℕ) (nights_per_week : ℕ)
    (hours_per_weekend_day : ℕ) (days_per_weekend : ℕ)
    (total_hours : ℕ) (weekly_hours : ℕ) (weeks : ℕ)
    (h1 : hours_per_night = 2) (h2 : nights_per_week = 5)
    (h3 : hours_per_weekend_day = 3) (h4 : days_per_weekend = 2)
    (h5 : total_hours = 96) (h6 : weekly_hours = 16)
    (h7 : weekly_hours = (hours_per_night * nights_per_week) + (hours_per_weekend_day * days_per_weekend)) :
  weeks = total_hours / weekly_hours :=
sorry

end Joey_SAT_Weeks_l213_213155


namespace Vasya_distance_fraction_l213_213902

variable (a b c d s : ℝ)

theorem Vasya_distance_fraction :
  (a = b / 2) →
  (c = a + d) →
  (d = s / 10) →
  (a + b + c + d = s) →
  (b / s = 0.4) :=
by
  intros h1 h2 h3 h4
  sorry

end Vasya_distance_fraction_l213_213902


namespace parabola_vertex_l213_213271

theorem parabola_vertex (x y : ℝ) : 
  (∀ x y, y^2 - 8*y + 4*x = 12 → (x, y) = (7, 4)) :=
by
  intros x y h
  sorry

end parabola_vertex_l213_213271


namespace emily_saves_more_using_promotion_a_l213_213319

-- Definitions based on conditions
def price_per_pair : ℕ := 50
def promotion_a_cost : ℕ := price_per_pair + price_per_pair / 2
def promotion_b_cost : ℕ := price_per_pair + (price_per_pair - 20)

-- Statement to prove the savings
theorem emily_saves_more_using_promotion_a :
  promotion_b_cost - promotion_a_cost = 5 := by
  sorry

end emily_saves_more_using_promotion_a_l213_213319


namespace value_of_4_inch_cube_l213_213697

noncomputable def value_per_cubic_inch (n : ℕ) : ℝ :=
  match n with
  | 1 => 300
  | _ => 1.1 ^ (n - 1) * 300

def cube_volume (n : ℕ) : ℝ :=
  n^3

noncomputable def total_value (n : ℕ) : ℝ :=
  cube_volume n * value_per_cubic_inch n

theorem value_of_4_inch_cube : total_value 4 = 25555 := by
  admit

end value_of_4_inch_cube_l213_213697


namespace geometric_sequence_a4_l213_213976

theorem geometric_sequence_a4 :
    ∀ (a : ℕ → ℝ) (n : ℕ), 
    a 1 = 2 → 
    (∀ n : ℕ, a (n + 1) = 3 * a n) → 
    a 4 = 54 :=
by
  sorry

end geometric_sequence_a4_l213_213976


namespace police_station_distance_l213_213561

theorem police_station_distance (thief_speed police_speed: ℝ) (delay chase_time: ℝ) 
  (h_thief_speed: thief_speed = 20) 
  (h_police_speed: police_speed = 40) 
  (h_delay: delay = 1)
  (h_chase_time: chase_time = 4) : 
  ∃ D: ℝ, D = 60 :=
by
  sorry

end police_station_distance_l213_213561


namespace jane_purchased_pudding_l213_213568

theorem jane_purchased_pudding (p : ℕ) 
  (ice_cream_cost_per_cone : ℕ := 5)
  (num_ice_cream_cones : ℕ := 15)
  (pudding_cost_per_cup : ℕ := 2)
  (cost_difference : ℕ := 65)
  (total_ice_cream_cost : ℕ := num_ice_cream_cones * ice_cream_cost_per_cone) 
  (total_pudding_cost : ℕ := p * pudding_cost_per_cup) :
  total_ice_cream_cost = total_pudding_cost + cost_difference → p = 5 :=
by
  sorry

end jane_purchased_pudding_l213_213568


namespace solve_basketball_points_l213_213993

noncomputable def y_points_other_members (x : ℕ) : ℕ :=
  let d_points := (1 / 3) * x
  let e_points := (3 / 8) * x
  let f_points := 18
  let total := x
  total - d_points - e_points - f_points

theorem solve_basketball_points (x : ℕ) (h1: x > 0) (h2: ∃ y ≤ 24, y = y_points_other_members x) :
  ∃ y, y = 21 :=
by
  sorry

end solve_basketball_points_l213_213993


namespace early_finish_hours_l213_213114

theorem early_finish_hours 
  (h : Nat) 
  (total_customers : Nat) 
  (num_workers : Nat := 3)
  (service_rate : Nat := 7) 
  (full_hours : Nat := 8)
  (total_customers_served : total_customers = 154) 
  (two_workers_hours : Nat := 2 * full_hours * service_rate) 
  (early_worker_customers : Nat := h * service_rate)
  (total_service : total_customers = two_workers_hours + early_worker_customers) : 
  h = 6 :=
by
  sorry

end early_finish_hours_l213_213114


namespace solve_for_x_l213_213226

theorem solve_for_x (x y : ℝ) (h : x / (x - 1) = (y^2 + 3 * y - 5) / (y^2 + 3 * y - 7)) :
  x = (y^2 + 3 * y - 5) / 2 :=
by 
  sorry

end solve_for_x_l213_213226


namespace expression_equals_5776_l213_213132

-- Define constants used in the problem
def a : ℕ := 476
def b : ℕ := 424
def c : ℕ := 4

-- Define the expression using the constants
def expression : ℕ := (a + b) ^ 2 - c * a * b

-- The target proof statement
theorem expression_equals_5776 : expression = 5776 := by
  sorry

end expression_equals_5776_l213_213132


namespace units_digit_27_64_l213_213255

/-- 
  Given that the units digit of 27 is 7, 
  and the units digit of 64 is 4, 
  prove that the units digit of 27 * 64 is 8.
-/
theorem units_digit_27_64 : 
  ∀ (n m : ℕ), 
  (n % 10 = 7) → 
  (m % 10 = 4) → 
  ((n * m) % 10 = 8) :=
by
  intros n m h1 h2
  -- Utilize modular arithmetic properties
  sorry

end units_digit_27_64_l213_213255


namespace perimeter_of_region_l213_213677

theorem perimeter_of_region : 
  let side := 1
  let diameter := side
  let radius := diameter / 2
  let full_circumference := 2 * Real.pi * radius
  let arc_length := (3 / 4) * full_circumference
  let total_arcs := 4
  let perimeter := total_arcs * arc_length
  perimeter = 3 * Real.pi :=
by 
  sorry

end perimeter_of_region_l213_213677


namespace fraction_irreducible_l213_213953

theorem fraction_irreducible (n : ℤ) : Int.gcd (21 * n + 4) (14 * n + 3) = 1 :=
by 
  sorry

end fraction_irreducible_l213_213953


namespace cannot_be_simultaneous_squares_l213_213768

theorem cannot_be_simultaneous_squares (x y : ℕ) :
  ¬ (∃ a b : ℤ, x^2 + y = a^2 ∧ y^2 + x = b^2) :=
by
  sorry

end cannot_be_simultaneous_squares_l213_213768


namespace neg_neg_eq_pos_l213_213472

theorem neg_neg_eq_pos : -(-2023) = 2023 :=
by
  sorry

end neg_neg_eq_pos_l213_213472


namespace neither_sufficient_nor_necessary_l213_213693

-- Definitions based on given conditions
def propA (a b : ℕ) : Prop := a + b ≠ 4
def propB (a b : ℕ) : Prop := a ≠ 1 ∧ b ≠ 3

-- Theorem statement (proof not required)
theorem neither_sufficient_nor_necessary (a b : ℕ) :
  ¬ (propA a b → propB a b) ∧ ¬ (propB a b → propA a b) := 
sorry

end neither_sufficient_nor_necessary_l213_213693


namespace percentage_error_calc_l213_213649

theorem percentage_error_calc (x : ℝ) (h : x ≠ 0) : 
  let correct_result := x * (5 / 3)
  let incorrect_result := x * (3 / 5)
  let percentage_error := (correct_result - incorrect_result) / correct_result * 100
  percentage_error = 64 := by
  sorry

end percentage_error_calc_l213_213649


namespace sum_of_digits_of_Joey_age_next_time_is_twice_Liam_age_l213_213119

-- Definitions of the conditions
def L := 2
def C := 2 * L^2  -- Chloe's age today based on Liam's age
def J := C + 3    -- Joey's age today

-- The future time when Joey's age is twice Liam's age
def future_time : ℕ := (sorry : ℕ) -- Placeholder for computation of 'n'
lemma compute_n : 2 * (L + future_time) = J + future_time := sorry

-- Joey's age at future time when it is twice Liam's age
def age_at_future_time : ℕ := J + future_time

-- Sum of the two digits of Joey's age at that future time
def digit_sum (n : ℕ) : ℕ := (n / 10) + (n % 10)

-- Final statement: sum of the digits of Joey's age at the specified future time
theorem sum_of_digits_of_Joey_age_next_time_is_twice_Liam_age :
  digit_sum age_at_future_time = 9 :=
by
  exact sorry

end sum_of_digits_of_Joey_age_next_time_is_twice_Liam_age_l213_213119


namespace calculate_49_squared_l213_213748

theorem calculate_49_squared : 
  ∀ (a b : ℕ), a = 50 → b = 2 → (a - b)^2 = a^2 - 2 * a * b + b^2 → (49^2 = 50^2 - 196) :=
by
  intro a b h1 h2 h3
  sorry

end calculate_49_squared_l213_213748


namespace root_bounds_l213_213313

noncomputable def sqrt (r : ℝ) (n : ℕ) := r^(1 / n)

theorem root_bounds (a b c d : ℝ) (n p x y : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) (hn : 0 < n) (hp : 0 < p) (hx : 0 < x) (hy : 0 < y) :
  sqrt d y < sqrt (a * b * c * d) (n + p + x + y) ∧
  sqrt (a * b * c * d) (n + p + x + y) < sqrt a n := 
sorry

end root_bounds_l213_213313


namespace remainder_of_N_mod_D_l213_213407

/-- The given number N and the divisor 252 defined in terms of its prime factors. -/
def N : ℕ := 9876543210123456789
def D : ℕ := 252

/-- The remainders of N modulo 4, 9, and 7 as given in the solution -/
def N_mod_4 : ℕ := 1
def N_mod_9 : ℕ := 0
def N_mod_7 : ℕ := 6

theorem remainder_of_N_mod_D :
  N % D = 27 :=
by
  sorry

end remainder_of_N_mod_D_l213_213407


namespace number_of_persons_in_room_l213_213538

theorem number_of_persons_in_room (n : ℕ) (h : n * (n - 1) / 2 = 78) : n = 13 :=
by
  /- We have:
     n * (n - 1) / 2 = 78,
     We need to prove n = 13 -/
  sorry

end number_of_persons_in_room_l213_213538


namespace ticket_difference_l213_213436

/-- 
  Define the initial number of tickets Billy had,
  the number of tickets after buying a yoyo,
  and state the proof that the difference is 16.
--/

theorem ticket_difference (initial_tickets : ℕ) (remaining_tickets : ℕ) 
  (h₁ : initial_tickets = 48) (h₂ : remaining_tickets = 32) : 
  initial_tickets - remaining_tickets = 16 :=
by
  /- This is where the prover would go, 
     no need to implement it as we know the expected result -/
  sorry

end ticket_difference_l213_213436


namespace soda_quantity_difference_l213_213475

noncomputable def bottles_of_diet_soda := 19
noncomputable def bottles_of_regular_soda := 60
noncomputable def bottles_of_cherry_soda := 35
noncomputable def bottles_of_orange_soda := 45

theorem soda_quantity_difference : 
  (max bottles_of_regular_soda (max bottles_of_diet_soda 
    (max bottles_of_cherry_soda bottles_of_orange_soda)) 
  - min bottles_of_regular_soda (min bottles_of_diet_soda 
    (min bottles_of_cherry_soda bottles_of_orange_soda))) = 41 := 
by
  sorry

end soda_quantity_difference_l213_213475


namespace number_of_members_l213_213231

theorem number_of_members (n : ℕ) (h1 : ∀ m : ℕ, m = n → m * m = 1936) : n = 44 :=
by
  -- Proof omitted
  sorry

end number_of_members_l213_213231


namespace range_of_a_l213_213501

theorem range_of_a (a : ℝ) : (∀ x : ℝ, x < a + 2 → x ≤ 2) ↔ a ≤ 0 := by
  sorry

end range_of_a_l213_213501


namespace second_shift_fraction_of_total_l213_213243

theorem second_shift_fraction_of_total (W E : ℕ) (h1 : ∀ (W : ℕ), E = (3 * W / 4))
  : let W₁ := W
    let E₁ := E
    let widgets_first_shift := W₁ * E₁
    let widgets_per_second_shift_employee := (2 * W₁) / 3
    let second_shift_employees := (4 * E₁) / 3
    let widgets_second_shift := (2 * W₁ / 3) * (4 * E₁ / 3)
    let total_widgets := widgets_first_shift + widgets_second_shift
    let fraction_second_shift := widgets_second_shift / total_widgets
    fraction_second_shift = 8 / 17 :=
sorry

end second_shift_fraction_of_total_l213_213243


namespace g_neither_even_nor_odd_l213_213733

noncomputable def g (x : ℝ) : ℝ := 3 ^ (x^2 - 3) - |x| + Real.sin x

theorem g_neither_even_nor_odd : ∀ x : ℝ, g x ≠ g (-x) ∧ g x ≠ -g (-x) := 
by
  intro x
  sorry

end g_neither_even_nor_odd_l213_213733


namespace smallest_positive_perfect_square_divisible_by_2_3_5_l213_213688

/-- Problem statement: 
    The smallest positive perfect square that is divisible by 2, 3, and 5 is 900.
-/
theorem smallest_positive_perfect_square_divisible_by_2_3_5 : 
  ∃ n : ℕ, 0 < n ∧ n * n % 2 = 0 ∧ n * n % 3 = 0 ∧ n * n % 5 = 0 ∧ n * n = 900 := 
by
  sorry

end smallest_positive_perfect_square_divisible_by_2_3_5_l213_213688


namespace initial_dragon_fruits_remaining_kiwis_l213_213879

variable (h d k : ℕ)    -- h: initial number of cantaloupes, d: initial number of dragon fruits, k: initial number of kiwis
variable (d_rem : ℕ)    -- d_rem: remaining number of dragon fruits after all cantaloupes are used up
variable (k_rem : ℕ)    -- k_rem: remaining number of kiwis after all cantaloupes are used up

axiom condition1 : d = 3 * h + 10
axiom condition2 : k = 2 * d
axiom condition3 : d_rem = 130
axiom condition4 : (d - d_rem) = 2 * h
axiom condition5 : k_rem = k - 10 * h

theorem initial_dragon_fruits (h : ℕ) (d : ℕ) (k : ℕ) (d_rem : ℕ) : 
  3 * h + 10 = d → 
  2 * d = k → 
  d_rem = 130 →
  2 * h + d_rem = d → 
  h = 120 → 
  d = 370 :=
by 
  intros
  sorry

theorem remaining_kiwis (h : ℕ) (d : ℕ) (k : ℕ) (k_rem : ℕ) : 
  3 * h + 10 = d → 
  2 * d = k → 
  h = 120 →
  k_rem = k - 10 * h → 
  k_rem = 140 :=
by 
  intros
  sorry

end initial_dragon_fruits_remaining_kiwis_l213_213879


namespace exponent_simplification_l213_213095

theorem exponent_simplification : (7^3 * (2^5)^3) / (7^2 * 2^(3*3)) = 448 := by
  sorry

end exponent_simplification_l213_213095


namespace evaluate_expression_l213_213118

theorem evaluate_expression (x : ℤ) (h : x = 5) : 
  3 * (3 * (3 * (3 * (3 * x + 2) + 2) + 2) + 2) + 2 = 1457 := 
by
  rw [h]
  sorry

end evaluate_expression_l213_213118


namespace profit_percentage_example_l213_213339

noncomputable def profit_percentage (cp_total : ℝ) (cp_count : ℕ) (sp_total : ℝ) (sp_count : ℕ) : ℝ :=
  let cp_per_article := cp_total / cp_count
  let sp_per_article := sp_total / sp_count
  let profit_per_article := sp_per_article - cp_per_article
  (profit_per_article / cp_per_article) * 100

theorem profit_percentage_example : profit_percentage 25 15 33 12 = 65 :=
by
  sorry

end profit_percentage_example_l213_213339


namespace percy_swimming_hours_l213_213246

theorem percy_swimming_hours :
  let weekday_hours_per_day := 2
  let weekdays := 5
  let weekend_hours := 3
  let weeks := 4
  let total_weekday_hours_per_week := weekday_hours_per_day * weekdays
  let total_weekend_hours_per_week := weekend_hours
  let total_hours_per_week := total_weekday_hours_per_week + total_weekend_hours_per_week
  let total_hours_over_weeks := total_hours_per_week * weeks
  total_hours_over_weeks = 64 :=
by
  sorry

end percy_swimming_hours_l213_213246


namespace function_passes_through_point_l213_213366

theorem function_passes_through_point (a : ℝ) (h_a_pos : a > 0) (h_a_ne_one : a ≠ 1) :
  ∃ P : ℝ × ℝ, P = (1, -1) ∧ ∀ x : ℝ, (y = a^(x-1) - 2) → y = -1 := by
  sorry

end function_passes_through_point_l213_213366


namespace minimize_a2_b2_l213_213195

theorem minimize_a2_b2 (a b t : ℝ) (h : 2 * a + b = 2 * t) : ∃ a b, (2 * a + b = 2 * t) ∧ (a^2 + b^2 = 4 * t^2 / 5) :=
by
  sorry

end minimize_a2_b2_l213_213195


namespace factor_correct_l213_213772

noncomputable def p (b : ℝ) : ℝ := 221 * b^2 + 17 * b
def factored_form (b : ℝ) : ℝ := 17 * b * (13 * b + 1)

theorem factor_correct (b : ℝ) : p b = factored_form b := by
  sorry

end factor_correct_l213_213772


namespace total_distance_walked_l213_213374

noncomputable def desk_to_fountain_distance : ℕ := 30
noncomputable def number_of_trips : ℕ := 4

theorem total_distance_walked :
  2 * desk_to_fountain_distance * number_of_trips = 240 :=
by
  sorry

end total_distance_walked_l213_213374


namespace symmetric_point_proof_l213_213534

def Point3D := (ℝ × ℝ × ℝ)

def symmetric_point_yOz (p : Point3D) : Point3D :=
  let (x, y, z) := p
  (-x, y, z)

theorem symmetric_point_proof :
  symmetric_point_yOz (1, -2, 3) = (-1, -2, 3) :=
by
  sorry

end symmetric_point_proof_l213_213534


namespace canonical_equations_of_line_l213_213268

/-- Given two planes: 
  Plane 1: 4 * x + y + z + 2 = 0
  Plane 2: 2 * x - y - 3 * z - 8 = 0
  Prove that the canonical equations of the line formed by their intersection are:
  (x - 1) / -2 = (y + 6) / 14 = z / -6 -/
theorem canonical_equations_of_line :
  (∃ x y z : ℝ, 4 * x + y + z + 2 = 0 ∧ 2 * x - y - 3 * z - 8 = 0) →
  (∀ x y z : ℝ, ((x - 1) / -2 = (y + 6) / 14) ∧ ((y + 6) / 14 = z / -6)) :=
by
  sorry

end canonical_equations_of_line_l213_213268


namespace average_first_six_numbers_l213_213258

theorem average_first_six_numbers (A : ℝ) (h1 : (11 : ℝ) * 9.9 = (6 * A + 6 * 11.4 - 22.5)) : A = 10.5 :=
by sorry

end average_first_six_numbers_l213_213258


namespace new_table_capacity_is_six_l213_213723

-- Definitions based on the conditions
def total_tables : ℕ := 40
def extra_new_tables : ℕ := 12
def total_customers : ℕ := 212
def original_table_capacity : ℕ := 4

-- Main statement to prove
theorem new_table_capacity_is_six (O N C : ℕ) 
  (h1 : O + N = total_tables)
  (h2 : N = O + extra_new_tables)
  (h3 : O * original_table_capacity + N * C = total_customers) :
  C = 6 :=
sorry

end new_table_capacity_is_six_l213_213723


namespace fraction_of_crop_to_CD_is_correct_l213_213173

-- Define the trapezoid with given conditions
structure Trapezoid :=
  (AB CD AD BC : ℝ)
  (angleA angleD : ℝ)
  (h: ℝ) -- height
  (Area Trapezoid total_area close_area_to_CD: ℝ) 

-- Assumptions
axiom AB_eq_CD (T : Trapezoid) : T.AB = 150 
axiom CD_eq_CD (T : Trapezoid) : T.CD = 200
axiom AD_eq_CD (T : Trapezoid) : T.AD = 130
axiom BC_eq_CD (T : Trapezoid) : T.BC = 130
axiom angleA_eq_75 (T : Trapezoid) : T.angleA = 75
axiom angleD_eq_75 (T : Trapezoid) : T.angleD = 75

-- The fraction calculation
noncomputable def fraction_to_CD (T : Trapezoid) : ℝ :=
  T.close_area_to_CD / T.total_area

-- Theorem stating the fraction of the crop that is brought to the longer base CD is 15/28
theorem fraction_of_crop_to_CD_is_correct (T : Trapezoid) 
  (h_pos : 0 < T.h)
  (total_area_def : T.total_area = (T.AB + T.CD) * T.h / 2)
  (close_area_def : T.close_area_to_CD = ((T.h / 4) * (T.AB + T.CD))) : 
  fraction_to_CD T = 15 / 28 :=
  sorry

end fraction_of_crop_to_CD_is_correct_l213_213173


namespace sufficient_not_necessary_l213_213372

theorem sufficient_not_necessary (x : ℝ) :
  (x > 1 → x^2 - 2*x + 1 > 0) ∧ (¬(x^2 - 2*x + 1 > 0 → x > 1)) := by
  sorry

end sufficient_not_necessary_l213_213372


namespace infinite_geometric_series_sum_l213_213847

theorem infinite_geometric_series_sum :
  let a := (5 : ℚ) / 3
  let r := -(3 : ℚ) / 4
  ∑' n : ℕ, a * r ^ n = 20 / 21 := by
  sorry

end infinite_geometric_series_sum_l213_213847


namespace ara_height_l213_213305

theorem ara_height (shea_height_now : ℝ) (shea_growth_percent : ℝ) (ara_growth_fraction : ℝ)
    (height_now : shea_height_now = 75) (growth_percent : shea_growth_percent = 0.25) 
    (growth_fraction : ara_growth_fraction = (2/3)) : 
    ∃ ara_height_now : ℝ, ara_height_now = 70 := by
  sorry

end ara_height_l213_213305


namespace shorter_leg_of_right_triangle_with_hypotenuse_65_l213_213075

theorem shorter_leg_of_right_triangle_with_hypotenuse_65 (a b : ℕ) (h : a^2 + b^2 = 65^2) : a = 16 ∨ b = 16 :=
by sorry

end shorter_leg_of_right_triangle_with_hypotenuse_65_l213_213075


namespace correct_proposition_l213_213641

-- Definitions of the propositions p and q
def p : Prop := ∀ x : ℝ, (x > 1 → x > 2)
def q : Prop := ∀ x y : ℝ, (x + y ≠ 2 → x ≠ 1 ∨ y ≠ 1)

-- The proof problem statement
theorem correct_proposition : ¬p ∧ q :=
by
  -- Assuming p is false (i.e., ¬p is true) and q is true
  sorry

end correct_proposition_l213_213641


namespace avg_of_multiples_of_4_is_even_l213_213580

theorem avg_of_multiples_of_4_is_even (m n : ℤ) (hm : m % 4 = 0) (hn : n % 4 = 0) :
  (m + n) / 2 % 2 = 0 := sorry

end avg_of_multiples_of_4_is_even_l213_213580


namespace multiply_expression_l213_213811

variable (y : ℝ)

theorem multiply_expression : 
  (16 * y^3) * (12 * y^5) * (1 / (4 * y)^3) = 3 * y^5 := by
  sorry

end multiply_expression_l213_213811


namespace poly_sum_of_squares_iff_nonneg_l213_213987

open Polynomial

variable {R : Type*} [Ring R] [OrderedRing R]

theorem poly_sum_of_squares_iff_nonneg (A : Polynomial ℝ) :
  (∃ P Q : Polynomial ℝ, A = P^2 + Q^2) ↔ ∀ x : ℝ, 0 ≤ A.eval x := sorry

end poly_sum_of_squares_iff_nonneg_l213_213987


namespace minimum_value_of_m_plus_n_l213_213433

-- Define the conditions and goals as a Lean 4 statement with a proof goal.
theorem minimum_value_of_m_plus_n (m n : ℝ) (h : m * n > 0) (hA : m + n = 3 * m * n) : m + n = 4 / 3 :=
sorry

end minimum_value_of_m_plus_n_l213_213433


namespace num_intersections_circle_line_eq_two_l213_213794

theorem num_intersections_circle_line_eq_two :
  ∃ (points : Finset (ℝ × ℝ)), {p : ℝ × ℝ | p.1 ^ 2 + p.2 ^ 2 = 25 ∧ p.1 = 3} = points ∧ points.card = 2 :=
by
  sorry

end num_intersections_circle_line_eq_two_l213_213794


namespace calculateBooksRemaining_l213_213761

noncomputable def totalBooksRemaining
    (initialBooks : ℕ)
    (n : ℕ)
    (a₁ : ℕ)
    (d : ℕ)
    (borrowedBooks : ℕ)
    (returnedBooks : ℕ) : ℕ :=
  let sumDonations := n * (2 * a₁ + (n - 1) * d) / 2
  let totalAfterDonations := initialBooks + sumDonations
  totalAfterDonations - borrowedBooks + returnedBooks

theorem calculateBooksRemaining :
  totalBooksRemaining 1000 15 2 2 350 270 = 1160 :=
by
  sorry

end calculateBooksRemaining_l213_213761


namespace total_rounds_played_l213_213419

/-- William and Harry played some rounds of tic-tac-toe.
    William won 5 more rounds than Harry.
    William won 10 rounds.
    Prove that the total number of rounds they played is 15. -/
theorem total_rounds_played (williams_wins : ℕ) (harrys_wins : ℕ)
  (h1 : williams_wins = 10)
  (h2 : williams_wins = harrys_wins + 5) :
  williams_wins + harrys_wins = 15 := 
by
  sorry

end total_rounds_played_l213_213419


namespace Miles_trombones_count_l213_213345

theorem Miles_trombones_count :
  let fingers := 10
  let trumpets := fingers - 3
  let hands := 2
  let guitars := hands + 2
  let french_horns := guitars - 1
  let heads := 1
  let trombones := heads + 2
  trumpets + guitars + french_horns + trombones = 17 → trombones = 3 :=
by
  intros h
  sorry

end Miles_trombones_count_l213_213345


namespace cannot_form_1x1x2_blocks_l213_213030

theorem cannot_form_1x1x2_blocks :
  let edge_length := 7
  let total_cubes := edge_length * edge_length * edge_length
  let central_cube := (3, 3, 3)
  let remaining_cubes := total_cubes - 1
  let checkerboard_color (x y z : Nat) : Bool := (x + y + z) % 2 = 0
  let num_white (k : Nat) := if k % 2 = 0 then 25 else 24
  let num_black (k : Nat) := if k % 2 = 0 then 24 else 25
  let total_white := 170
  let total_black := 171
  total_black > total_white →
  ¬(remaining_cubes % 2 = 0 ∧ total_white % 2 = 0 ∧ total_black % 2 = 0) → 
  ∀ (block: Nat × Nat × Nat → Bool) (x y z : Nat), block (x, y, z) = ((x*y*z) % 2 = 0) := sorry

end cannot_form_1x1x2_blocks_l213_213030


namespace find_x_l213_213621

theorem find_x (a b x : ℝ) (h1 : 2^a = x) (h2 : 3^b = x) (h3 : 1/a + 1/b = 1) : x = 6 :=
sorry

end find_x_l213_213621


namespace no_such_function_exists_l213_213408

theorem no_such_function_exists 
  (f : ℝ → ℝ) 
  (h_f_pos : ∀ x, 0 < x → 0 < f x) 
  (h_eq : ∀ x y, 0 < x → 0 < y → f (x + y) = f x + f y + (1 / 2012)) : 
  false :=
sorry

end no_such_function_exists_l213_213408


namespace tank_capacity_l213_213351

theorem tank_capacity (x : ℝ) (h₁ : (3/4) * x = (1/3) * x + 18) : x = 43.2 := sorry

end tank_capacity_l213_213351


namespace cos_4theta_value_l213_213312

theorem cos_4theta_value (theta : ℝ) 
  (h : ∑' n : ℕ, (Real.cos theta)^(2 * n) = 8) : 
  Real.cos (4 * theta) = 1 / 8 := 
sorry

end cos_4theta_value_l213_213312


namespace find_n_on_angle_bisector_l213_213994

theorem find_n_on_angle_bisector (M : ℝ × ℝ) (hM : M = (3 * n - 2, 2 * n + 7) ∧ M.1 + M.2 = 0) : 
    n = -1 :=
by
  sorry

end find_n_on_angle_bisector_l213_213994


namespace sum_of_m_n_l213_213140

-- Define the setup for the problem
def side_length_of_larger_square := 3
def side_length_of_smaller_square := 1
def side_length_of_given_rectangle_l1 := 1
def side_length_of_given_rectangle_l2 := 3
def total_area_of_larger_square := side_length_of_larger_square * side_length_of_larger_square
def area_of_smaller_square := side_length_of_smaller_square * side_length_of_smaller_square
def area_of_given_rectangle := side_length_of_given_rectangle_l1 * side_length_of_given_rectangle_l2

-- Define the variable for the area of rectangle R
def area_of_R := total_area_of_larger_square - (area_of_smaller_square + area_of_given_rectangle)

-- Given the problem statement, we need to find m and n such that the area of R is m/n.
def m := 5
def n := 1

-- We need to prove that m + n = 6 given these conditions
theorem sum_of_m_n : m + n = 6 := by
  sorry

end sum_of_m_n_l213_213140


namespace fly_distance_from_ceiling_l213_213425

/-- 
Assume a room where two walls and the ceiling meet at right angles at point P.
Let point P be the origin (0, 0, 0). 
Let the fly's position be (2, 7, z), where z is the distance from the ceiling.
Given the fly is 2 meters from one wall, 7 meters from the other wall, 
and 10 meters from point P, prove that the fly is at a distance sqrt(47) from the ceiling.
-/
theorem fly_distance_from_ceiling : 
  ∀ (z : ℝ), 
  (2^2 + 7^2 + z^2 = 10^2) → 
  z = Real.sqrt 47 :=
by 
  intro z h
  sorry

end fly_distance_from_ceiling_l213_213425


namespace determine_h_l213_213935

variable {R : Type*} [CommRing R]

def h_poly (x : R) : R := -8*x^4 + 2*x^3 + 4*x^2 - 6*x + 2

theorem determine_h (x : R) :
  (8*x^4 - 4*x^2 + 2 + h_poly x = 2*x^3 - 6*x + 4) ->
  h_poly x = -8*x^4 + 2*x^3 + 4*x^2 - 6*x + 2 :=
by
  intro h
  sorry

end determine_h_l213_213935


namespace percentage_increase_B_more_than_C_l213_213229

noncomputable def percentage_increase :=
  let C_m := 14000
  let A_annual := 470400
  let A_m := A_annual / 12
  let B_m := (2 / 5) * A_m
  ((B_m - C_m) / C_m) * 100

theorem percentage_increase_B_more_than_C : percentage_increase = 12 :=
  sorry

end percentage_increase_B_more_than_C_l213_213229


namespace ratio_m_q_l213_213797

theorem ratio_m_q (m n p q : ℚ) (h1 : m / n = 25) (h2 : p / n = 5) (h3 : p / q = 1 / 15) : 
  m / q = 1 / 3 :=
by 
  sorry

end ratio_m_q_l213_213797


namespace constant_term_of_expansion_l213_213088

noncomputable def constant_term := 
  (20: ℕ) * (216: ℕ) * (1/27: ℚ) = (160: ℕ)

theorem constant_term_of_expansion : constant_term :=
  by sorry

end constant_term_of_expansion_l213_213088


namespace value_of_coefficients_l213_213831

theorem value_of_coefficients (a₀ a₁ a₂ a₃ : ℤ) (x : ℤ) :
  (5 * x + 4) ^ 3 = a₀ + a₁ * x + a₂ * x ^ 2 + a₃ * x ^ 3 →
  x = -1 →
  (a₀ + a₂) - (a₁ + a₃) = -1 :=
by
  sorry

end value_of_coefficients_l213_213831


namespace smallest_possible_z_l213_213238

theorem smallest_possible_z :
  ∃ (z : ℕ), (z = 6) ∧ 
  ∃ (u w x y : ℕ), u < w ∧ w < x ∧ x < y ∧ y < z ∧ 
  u.succ = w ∧ w.succ = x ∧ x.succ = y ∧ y.succ = z ∧ 
  u^3 + w^3 + x^3 + y^3 = z^3 :=
by
  use 6
  sorry

end smallest_possible_z_l213_213238


namespace length_of_each_piece_l213_213133

-- Definitions based on conditions
def total_length : ℝ := 42.5
def number_of_pieces : ℝ := 50

-- The statement that we need to prove
theorem length_of_each_piece (h1 : total_length = 42.5) (h2 : number_of_pieces = 50) : 
  total_length / number_of_pieces = 0.85 := 
by
  sorry

end length_of_each_piece_l213_213133


namespace grassy_pathway_area_correct_l213_213270

-- Define the dimensions of the plot and the pathway width
def length_plot : ℝ := 15
def width_plot : ℝ := 10
def width_pathway : ℝ := 2

-- Define the required areas
def total_area : ℝ := (length_plot + 2 * width_pathway) * (width_plot + 2 * width_pathway)
def plot_area : ℝ := length_plot * width_plot
def grassy_pathway_area : ℝ := total_area - plot_area

-- Prove that the area of the grassy pathway is 116 m²
theorem grassy_pathway_area_correct : grassy_pathway_area = 116 := by
  sorry

end grassy_pathway_area_correct_l213_213270


namespace bugs_diagonally_at_least_9_unoccupied_l213_213011

theorem bugs_diagonally_at_least_9_unoccupied (bugs : ℕ × ℕ → Prop) :
  let board_size := 9
  let cells := (board_size * board_size)
  let black_cells := 45
  let white_cells := 36
  ∃ unoccupied_cells ≥ 9, true := 
sorry

end bugs_diagonally_at_least_9_unoccupied_l213_213011


namespace solve_part_a_solve_part_b_l213_213690

-- Part (a)
theorem solve_part_a (x : ℝ) (h1 : 36 * x^2 - 1 = (6 * x + 1) * (6 * x - 1)) :
  (3 / (1 - 6 * x) = 2 / (6 * x + 1) - (8 + 9 * x) / (36 * x^2 - 1)) ↔ x = 1 / 3 :=
sorry

-- Part (b)
theorem solve_part_b (z : ℝ) (h2 : 1 - z^2 = (1 + z) * (1 - z)) :
  (3 / (1 - z^2) = 2 / (1 + z)^2 - 5 / (1 - z)^2) ↔ z = -3 / 7 :=
sorry

end solve_part_a_solve_part_b_l213_213690


namespace arithmetic_mean_l213_213531

variables (x y z : ℝ)

def condition1 : Prop := 1 / (x * y) = y / (z - x + 1)
def condition2 : Prop := 1 / (x * y) = 2 / (z + 1)

theorem arithmetic_mean (h1 : condition1 x y z) (h2 : condition2 x y z) : x = (z + y) / 2 :=
by
  sorry

end arithmetic_mean_l213_213531


namespace calc_length_RS_l213_213932

-- Define the trapezoid properties
def trapezoid (PQRS : Type) (PR QS : ℝ) (h A : ℝ) : Prop :=
  PR = 12 ∧ QS = 20 ∧ h = 10 ∧ A = 180

-- Define the length of the side RS
noncomputable def length_RS (PQRS : Type) (PR QS h A : ℝ) : ℝ :=
  18 - 0.5 * Real.sqrt 44 - 5 * Real.sqrt 3

-- Define the theorem statement
theorem calc_length_RS {PQRS : Type} (PR QS h A : ℝ) :
  trapezoid PQRS PR QS h A → length_RS PQRS PR QS h A = 18 - 0.5 * Real.sqrt 44 - 5 * Real.sqrt 3 :=
by
  intros
  exact Eq.refl (18 - 0.5 * Real.sqrt 44 - 5 * Real.sqrt 3)

end calc_length_RS_l213_213932


namespace min_students_l213_213039

theorem min_students (b g : ℕ) 
  (h1 : 3 * b = 2 * g) 
  (h2 : ∃ k : ℕ, b + g = 10 * k) : b + g = 38 :=
sorry

end min_students_l213_213039


namespace simple_interest_rate_l213_213260

theorem simple_interest_rate (P : ℝ) (r : ℝ) (T : ℝ) (SI : ℝ)
  (h1 : SI = P / 5)
  (h2 : T = 10)
  (h3 : SI = (P * r * T) / 100) :
  r = 2 :=
by
  sorry

end simple_interest_rate_l213_213260


namespace quadratic_inequality_solution_set_l213_213237

variable (a b c : ℝ)

theorem quadratic_inequality_solution_set (h1 : ∀ x : ℝ, ax^2 + bx + c > 0 → (-1 / 3 < x ∧ x < 2)) :
  ∀ x : ℝ, cx^2 + bx + a < 0 → (-3 < x ∧ x < 1 / 2) :=
by
  sorry

end quadratic_inequality_solution_set_l213_213237


namespace sum_of_interior_diagonals_l213_213536

theorem sum_of_interior_diagonals (a b c : ℝ)
  (h₁ : 2 * (a * b + b * c + c * a) = 166)
  (h₂ : a + b + c = 16) :
  4 * Real.sqrt (a ^ 2 + b ^ 2 + c ^ 2) = 12 * Real.sqrt 10 :=
by
  sorry

end sum_of_interior_diagonals_l213_213536


namespace symmetric_points_l213_213726

theorem symmetric_points (m n : ℤ) (h1 : m - 1 = -3) (h2 : 1 = n - 1) : m + n = 0 := by
  sorry

end symmetric_points_l213_213726


namespace championship_positions_l213_213996

def positions_valid : Prop :=
  ∃ (pos_A pos_B pos_D pos_E pos_V pos_G : ℕ),
  (pos_A = pos_B + 3) ∧
  (pos_D < pos_E ∧ pos_E < pos_B) ∧
  (pos_V < pos_G) ∧
  (pos_D = 1) ∧
  (pos_E = 2) ∧
  (pos_B = 3) ∧
  (pos_V = 4) ∧
  (pos_G = 5) ∧
  (pos_A = 6)

theorem championship_positions : positions_valid :=
by
  sorry

end championship_positions_l213_213996


namespace regular_polygon_perimeter_l213_213327

theorem regular_polygon_perimeter (side_length : ℝ) (exterior_angle : ℝ) (n : ℕ) 
  (h1 : side_length = 7) (h2 : exterior_angle = 90) (h3 : 180 * (n - 2) / n = 90) : 
  n * side_length = 28 :=
by 
  sorry

end regular_polygon_perimeter_l213_213327


namespace complex_number_quadrant_l213_213429

def inSecondQuadrant (z : ℂ) : Prop :=
  z.re < 0 ∧ z.im > 0

theorem complex_number_quadrant : inSecondQuadrant (i / (1 - i)) :=
by
  sorry

end complex_number_quadrant_l213_213429


namespace Haman_initial_trays_l213_213470

theorem Haman_initial_trays 
  (eggs_in_tray : ℕ)
  (total_eggs_sold : ℕ)
  (trays_dropped : ℕ)
  (additional_trays : ℕ)
  (trays_finally_sold : ℕ)
  (std_trays_sold : total_eggs_sold / eggs_in_tray = trays_finally_sold) 
  (eggs_in_tray_def : eggs_in_tray = 30) 
  (total_eggs_sold_def : total_eggs_sold = 540)
  (trays_dropped_def : trays_dropped = 2)
  (additional_trays_def : additional_trays = 7) :
  trays_finally_sold - additional_trays + trays_dropped = 13 := 
by 
  sorry

end Haman_initial_trays_l213_213470


namespace evaluate_f_at_2_l213_213751

def f (x : ℝ) : ℝ := x^3 - 2*x^2 + 3*x - 1

theorem evaluate_f_at_2 : f 2 = 5 := by
  sorry

end evaluate_f_at_2_l213_213751


namespace min_birthdays_on_wednesday_l213_213161

theorem min_birthdays_on_wednesday 
  (W X : ℕ) 
  (h1 : W + 6 * X = 50) 
  (h2 : W > X) : 
  W = 8 := 
sorry

end min_birthdays_on_wednesday_l213_213161


namespace albert_number_solution_l213_213988

theorem albert_number_solution (A B C : ℝ) 
  (h1 : A = 2 * B + 1) 
  (h2 : B = 2 * C + 1) 
  (h3 : C = 2 * A + 2) : 
  A = -11 / 7 := 
by 
  sorry

end albert_number_solution_l213_213988


namespace intersection_of_A_and_B_l213_213707

variable (x y : ℝ)

def A := {y : ℝ | ∃ x > 1, y = Real.log x / Real.log 2}
def B := {y : ℝ | ∃ x > 1, y = (1 / 2) ^ x}

theorem intersection_of_A_and_B :
  (A ∩ B) = {y : ℝ | 0 < y ∧ y < 1 / 2} :=
by sorry

end intersection_of_A_and_B_l213_213707


namespace shelby_initial_money_l213_213742

-- Definitions based on conditions
def cost_of_first_book : ℕ := 8
def cost_of_second_book : ℕ := 4
def cost_of_each_poster : ℕ := 4
def number_of_posters : ℕ := 2

-- Number to prove (initial money)
def initial_money : ℕ := 20

-- Theorem statement
theorem shelby_initial_money :
    (cost_of_first_book + cost_of_second_book + (number_of_posters * cost_of_each_poster)) = initial_money := by
    sorry

end shelby_initial_money_l213_213742


namespace complement_A_A_inter_complement_B_l213_213507

open Set

def U : Set ℝ := univ
def A : Set ℝ := {x | 1 < x ∧ x < 4}
def B : Set ℝ := {x | x^2 - 2*x - 3 ≤ 0}

theorem complement_A : compl A = {x | x ≤ 1 ∨ 4 ≤ x} :=
by sorry

theorem A_inter_complement_B : A ∩ compl B = {x | 3 < x ∧ x < 4} :=
by sorry

end complement_A_A_inter_complement_B_l213_213507


namespace find_speed_range_l213_213888

noncomputable def runningErrorB (v : ℝ) : ℝ := abs ((300 / v) - 7)
noncomputable def runningErrorC (v : ℝ) : ℝ := abs ((480 / v) - 11)

theorem find_speed_range (v : ℝ) :
  (runningErrorB v + runningErrorC v ≤ 2) →
  33.33 ≤ v ∧ v ≤ 48.75 := sorry

end find_speed_range_l213_213888


namespace joe_purchased_360_gallons_l213_213644

def joe_initial_paint (P : ℝ) : Prop :=
  let first_week_paint := (1/4) * P
  let remaining_paint := (3/4) * P
  let second_week_paint := (1/2) * remaining_paint
  let total_used_paint := first_week_paint + second_week_paint
  total_used_paint = 225

theorem joe_purchased_360_gallons : ∃ P : ℝ, joe_initial_paint P ∧ P = 360 :=
by
  sorry

end joe_purchased_360_gallons_l213_213644


namespace simplify_polynomial_sum_l213_213284

/- Define the given polynomials -/
def polynomial1 (x : ℝ) : ℝ := (5 * x^10 + 8 * x^9 + 3 * x^8)
def polynomial2 (x : ℝ) : ℝ := (2 * x^12 + 3 * x^10 + x^9 + 4 * x^8 + 6 * x^4 + 7 * x^2 + 9)
def resultant_polynomial (x : ℝ) : ℝ := (2 * x^12 + 8 * x^10 + 9 * x^9 + 7 * x^8 + 6 * x^4 + 7 * x^2 + 9)

theorem simplify_polynomial_sum (x : ℝ) :
  polynomial1 x + polynomial2 x = resultant_polynomial x :=
by
  sorry

end simplify_polynomial_sum_l213_213284


namespace jake_more_balloons_than_allan_l213_213380

-- Define the initial and additional balloons for Allan
def initial_allan_balloons : Nat := 2
def additional_allan_balloons : Nat := 3

-- Total balloons Allan has in the park
def total_allan_balloons : Nat := initial_allan_balloons + additional_allan_balloons

-- Number of balloons Jake has
def jake_balloons : Nat := 6

-- The proof statement
theorem jake_more_balloons_than_allan : jake_balloons - total_allan_balloons = 1 := by
  sorry

end jake_more_balloons_than_allan_l213_213380


namespace new_energy_vehicle_sales_growth_l213_213217

theorem new_energy_vehicle_sales_growth (x : ℝ) :
  let sales_jan := 64
  let sales_feb := 64 * (1 + x)
  let sales_mar := 64 * (1 + x)^2
  (sales_jan + sales_feb + sales_mar = 244) :=
sorry

end new_energy_vehicle_sales_growth_l213_213217


namespace relationship_between_m_and_n_l213_213224

theorem relationship_between_m_and_n (f : ℝ → ℝ) (a : ℝ) 
  (h_even : ∀ x : ℝ, f x = f (-x)) 
  (h_mono_inc : ∀ x y : ℝ, 0 ≤ x → x ≤ y → f x ≤ f y) 
  (m_def : f (-1) = f 1) 
  (n_def : f (a^2 + 2*a + 3) > f 1) :
  f (-1) < f (a^2 + 2*a + 3) := 
by 
  sorry

end relationship_between_m_and_n_l213_213224


namespace maximize_xyplusxzplusyzplusy2_l213_213710

theorem maximize_xyplusxzplusyzplusy2 (x y z : ℝ) (h1 : x + 2 * y + z = 7) (h2 : y ≥ 0) :
  xy + xz + yz + y^2 ≤ 10.5 :=
sorry

end maximize_xyplusxzplusyzplusy2_l213_213710


namespace correct_option_l213_213295

-- Definitions based on conditions
def sentence_structure : String := "He’s never interested in what ______ is doing."

def option_A : String := "no one else"
def option_B : String := "anyone else"
def option_C : String := "someone else"
def option_D : String := "nobody else"

-- The proof statement
theorem correct_option : option_B = "anyone else" := by
  sorry

end correct_option_l213_213295


namespace no_solution_eq1_l213_213924

   theorem no_solution_eq1 : ¬ ∃ x, (3 - x) / (x - 4) - 1 / (4 - x) = 1 :=
   by
     sorry
   
end no_solution_eq1_l213_213924


namespace sum_divides_product_iff_l213_213512

theorem sum_divides_product_iff (n : ℕ) : 
  (n*(n+1)/2) ∣ n! ↔ ∃ (a b : ℕ), 1 < a ∧ 1 < b ∧ a * b = n + 1 ∧ a ≤ n ∧ b ≤ n :=
sorry

end sum_divides_product_iff_l213_213512


namespace fraction_of_students_older_than_4_years_l213_213244

-- Definitions based on conditions
def total_students := 50
def students_younger_than_3 := 20
def students_not_between_3_and_4 := 25
def students_older_than_4 := students_not_between_3_and_4 - students_younger_than_3
def fraction_older_than_4 := students_older_than_4 / total_students

-- Theorem to prove the desired fraction
theorem fraction_of_students_older_than_4_years : fraction_older_than_4 = 1/10 :=
by
  sorry

end fraction_of_students_older_than_4_years_l213_213244


namespace number_of_teams_l213_213863

theorem number_of_teams (x : ℕ) (h : x * (x - 1) = 90) : x = 10 :=
sorry

end number_of_teams_l213_213863


namespace isosceles_triangle_inequality_degenerate_triangle_a_zero_degenerate_triangle_double_b_l213_213182

section isosceles_triangle

variables (a b k : ℝ)

/-- Prove the inequality for an isosceles triangle -/
theorem isosceles_triangle_inequality (h_perimeter : k = a + 2 * b) (ha_pos : a > 0) :
  k / 2 < a + b ∧ a + b < 3 * k / 4 :=
sorry

/-- Prove the inequality for degenerate triangle with a = 0 -/
theorem degenerate_triangle_a_zero (b k : ℝ) (h_perimeter : k = 2 * b) :
  k / 2 ≤ b ∧ b < 3 * k / 4 :=
sorry

/-- Prove the inequality for degenerate triangle with a = 2b -/
theorem degenerate_triangle_double_b (b k : ℝ) (h_perimeter : k = 4 * b) :
  k / 2 < b ∧ b ≤ 3 * k / 4 :=
sorry

end isosceles_triangle

end isosceles_triangle_inequality_degenerate_triangle_a_zero_degenerate_triangle_double_b_l213_213182


namespace sequence_a_n_perfect_square_l213_213034

theorem sequence_a_n_perfect_square :
  (∃ a : ℕ → ℤ, ∃ b : ℕ → ℤ,
    a 0 = 1 ∧ b 0 = 0 ∧
    (∀ n : ℕ, a (n + 1) = 7 * a n + 6 * b n - 3) ∧
    (∀ n : ℕ, b (n + 1) = 8 * a n + 7 * b n - 4) ∧
    (∀ n : ℕ, ∃ k : ℤ, a n = k^2)) :=
sorry

end sequence_a_n_perfect_square_l213_213034


namespace potions_needed_l213_213515

-- Definitions
def galleons_to_knuts (galleons : Int) : Int := galleons * 17 * 23
def sickles_to_knuts (sickles : Int) : Int := sickles * 23

-- Conditions from the problem
def cost_of_owl_in_knuts : Int := galleons_to_knuts 2 + sickles_to_knuts 1 + 5
def knuts_per_potion : Int := 9

-- Prove the number of potions needed is 90
theorem potions_needed : cost_of_owl_in_knuts / knuts_per_potion = 90 := by
  sorry

end potions_needed_l213_213515


namespace masha_problem_l213_213798

noncomputable def sum_arithmetic_series (a l n : ℕ) : ℕ :=
  (n * (a + l)) / 2

theorem masha_problem : 
  let a_even := 372
  let l_even := 506
  let n_even := 67
  let a_odd := 373
  let l_odd := 505
  let n_odd := 68
  let S_even := sum_arithmetic_series a_even l_even n_even
  let S_odd := sum_arithmetic_series a_odd l_odd n_odd
  S_odd - S_even = 439 := 
by sorry

end masha_problem_l213_213798


namespace a_n_formula_l213_213896

variable {a : ℕ+ → ℝ}  -- Defining a_n as a sequence from positive natural numbers to real numbers
variable {S : ℕ+ → ℝ}  -- Defining S_n as a sequence from positive natural numbers to real numbers

-- Given conditions
axiom S_def (n : ℕ+) : S n = a n / 2 + 1 / a n - 1
axiom a_pos (n : ℕ+) : a n > 0

-- Conjecture to be proved
theorem a_n_formula (n : ℕ+) : a n = Real.sqrt (2 * n + 1) - Real.sqrt (2 * n - 1) := 
sorry -- proof to be done

end a_n_formula_l213_213896


namespace triangle_square_ratio_l213_213949

theorem triangle_square_ratio (t s : ℝ) 
  (h1 : 3 * t = 15) 
  (h2 : 4 * s = 12) : 
  t / s = 5 / 3 :=
by 
  -- skipping the proof
  sorry

end triangle_square_ratio_l213_213949


namespace marble_prism_weight_l213_213904

theorem marble_prism_weight :
  let height := 8
  let base_side := 2
  let density := 2700
  let volume := base_side * base_side * height
  volume * density = 86400 :=
by
  let height := 8
  let base_side := 2
  let density := 2700
  let volume := base_side * base_side * height
  sorry

end marble_prism_weight_l213_213904


namespace proof_part1_proof_part2_l213_213897

-- Proof problem for the first part (1)
theorem proof_part1 (m : ℝ) : m^3 * m^6 + (-m^3)^3 = 0 := 
by
  sorry

-- Proof problem for the second part (2)
theorem proof_part2 (a : ℝ) : a * (a - 2) - 2 * a * (1 - 3 * a) = 7 * a^2 - 4 * a := 
by
  sorry

end proof_part1_proof_part2_l213_213897


namespace domain_of_f_l213_213889

theorem domain_of_f (x : ℝ) : (1 - x > 0) ∧ (2 * x + 1 > 0) ↔ - (1 / 2 : ℝ) < x ∧ x < 1 :=
by
  sorry

end domain_of_f_l213_213889


namespace jelly_beans_in_jar_y_l213_213779

-- Definitions of the conditions
def total_beans : ℕ := 1200
def number_beans_in_jar_y (y : ℕ) := y
def number_beans_in_jar_x (y : ℕ) := 3 * y - 400

-- The main theorem to be proven
theorem jelly_beans_in_jar_y (y : ℕ) :
  number_beans_in_jar_x y + number_beans_in_jar_y y = total_beans → 
  y = 400 := 
by
  sorry

end jelly_beans_in_jar_y_l213_213779


namespace determine_m_range_l213_213209

variable {R : Type} [OrderedCommGroup R]

-- Define the odd function f: ℝ → ℝ
def odd_function (f : ℝ → ℝ) := ∀ x, f (-x) = -f x

-- Define the increasing function f: ℝ → ℝ
def increasing_function (f : ℝ → ℝ) := ∀ x y, x < y → f x < f y

-- Define the main theorem
theorem determine_m_range (f : ℝ → ℝ) (odd_f : odd_function f) (inc_f : increasing_function f) :
    (∀ θ : ℝ, f (Real.cos (2 * θ) - 5) + f (2 * m + 4 * Real.sin θ) > 0) → m > 5 :=
by
  sorry

end determine_m_range_l213_213209


namespace find_surface_area_of_sphere_l213_213519

variables (a b c : ℝ)

-- The conditions given in the problem
def condition1 := a * b = 6
def condition2 := b * c = 2
def condition3 := a * c = 3
def vertices_on_sphere := true  -- Assuming vertices on tensor sphere condition for mathematical completion

theorem find_surface_area_of_sphere
  (h1 : condition1 a b)
  (h2 : condition2 b c)
  (h3 : condition3 a c)
  (h4 : vertices_on_sphere) :
  4 * Real.pi * ((Real.sqrt (a^2 + b^2 + c^2)) / 2)^2 = 14 * Real.pi :=
  sorry

end find_surface_area_of_sphere_l213_213519


namespace hot_dogs_remainder_l213_213180

theorem hot_dogs_remainder : 25197641 % 6 = 1 :=
by
  sorry

end hot_dogs_remainder_l213_213180


namespace raccoon_carrots_hid_l213_213955

theorem raccoon_carrots_hid 
  (r : ℕ)
  (b : ℕ)
  (h1 : 5 * r = 8 * b)
  (h2 : b = r - 3) 
  : 5 * r = 40 :=
by
  sorry

end raccoon_carrots_hid_l213_213955


namespace sum_three_smallest_m_l213_213548

theorem sum_three_smallest_m :
  (∃ a m, 
    (a - 2 + a + a + 2) / 3 = 7 
    ∧ m % 4 = 3 
    ∧ m ≠ 5 ∧ m ≠ 7 ∧ m ≠ 9 
    ∧ (5 + 7 + 9 + m) % 4 = 0 
    ∧ m > 0) 
  → 3 + 11 + 15 = 29 :=
sorry

end sum_three_smallest_m_l213_213548


namespace larger_number_l213_213348

theorem larger_number (HCF LCM a b : ℕ) (h_hcf : HCF = 28) (h_factors: 12 * 15 * HCF = LCM) (h_prod : a * b = HCF * LCM) :
  max a b = 180 :=
sorry

end larger_number_l213_213348


namespace find_number_l213_213989

variable (a : ℕ) (n : ℕ)

theorem find_number (h₁ : a = 105) (h₂ : a ^ 3 = 21 * n * 45 * 49) : n = 25 :=
by
  sorry

end find_number_l213_213989


namespace area_of_quadrilateral_l213_213395

theorem area_of_quadrilateral (θ : ℝ) (sin_θ : Real.sin θ = 4/5) (b1 b2 : ℝ) (h: ℝ) (base1 : b1 = 14) (base2 : b2 = 20) (height : h = 8) : 
  (1 / 2) * (b1 + b2) * h = 136 := by
  sorry

end area_of_quadrilateral_l213_213395


namespace sum_of_first_5n_l213_213468

theorem sum_of_first_5n (n : ℕ) : 
  (n * (n + 1) / 2) + 210 = ((4 * n) * (4 * n + 1) / 2) → 
  (5 * n) * (5 * n + 1) / 2 = 465 :=
by sorry

end sum_of_first_5n_l213_213468


namespace unique_k_value_l213_213058

noncomputable def findK (k : ℝ) : Prop :=
  ∃ (x : ℝ), (x^2 - k) * (x + k + 1) = x^3 + k * (x^2 - x - 4) ∧ k ≠ 0 ∧ k = -3

theorem unique_k_value : ∀ (k : ℝ), findK k :=
by
  intro k
  sorry

end unique_k_value_l213_213058


namespace expression_equals_k_times_10_pow_1007_l213_213922

theorem expression_equals_k_times_10_pow_1007 :
  (3^1006 + 7^1007)^2 - (3^1006 - 7^1007)^2 = 588 * 10^1007 := by
  sorry

end expression_equals_k_times_10_pow_1007_l213_213922


namespace sixth_grade_percentage_combined_l213_213485

def maplewood_percentages := [10, 20, 15, 15, 10, 15, 15]
def brookside_percentages := [16, 14, 13, 12, 12, 18, 15]

def maplewood_students := 150
def brookside_students := 180

def sixth_grade_maplewood := maplewood_students * (maplewood_percentages.get! 6) / 100
def sixth_grade_brookside := brookside_students * (brookside_percentages.get! 6) / 100

def total_students := maplewood_students + brookside_students
def total_sixth_graders := sixth_grade_maplewood + sixth_grade_brookside

def sixth_grade_percentage := total_sixth_graders / total_students * 100

theorem sixth_grade_percentage_combined : sixth_grade_percentage = 15 := by 
  sorry

end sixth_grade_percentage_combined_l213_213485


namespace lcm_18_24_30_l213_213870

theorem lcm_18_24_30 :
  let a := 18
  let b := 24
  let c := 30
  let lcm := 360
  (∀ x > 0, x ∣ a ∧ x ∣ b ∧ x ∣ c → x ∣ lcm) ∧ (∀ y > 0, y ∣ lcm → y ∣ a ∧ y ∣ b ∧ y ∣ c) :=
by {
  let a := 18
  let b := 24
  let c := 30
  let lcm := 360
  sorry
}

end lcm_18_24_30_l213_213870


namespace determine_a_l213_213983

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
if x < 1 then x^3 + 1 else x^2 - a * x

theorem determine_a (a : ℝ) : 
  f (f 0 a) a = -2 → a = 3 :=
by
  sorry

end determine_a_l213_213983


namespace valid_n_values_l213_213717

theorem valid_n_values (n x y : ℤ) (h1 : n * (x - 3) = y + 3) (h2 : x + n = 3 * (y - n)) :
  n = 1 ∨ n = 2 ∨ n = 3 ∨ n = 7 :=
sorry

end valid_n_values_l213_213717


namespace chloe_and_friends_points_l213_213413

-- Define the conditions as Lean definitions and then state the theorem to be proven.

def total_pounds_recycled : ℕ := 28 + 2

def pounds_per_point : ℕ := 6

def points_earned (total_pounds : ℕ) (pounds_per_point : ℕ) : ℕ :=
  total_pounds / pounds_per_point

theorem chloe_and_friends_points :
  points_earned total_pounds_recycled pounds_per_point = 5 :=
by
  sorry

end chloe_and_friends_points_l213_213413


namespace factor_of_quadratic_expression_l213_213347

def is_factor (a b : ℤ) : Prop := ∃ k, b = k * a

theorem factor_of_quadratic_expression (m : ℤ) :
  is_factor (m - 8) (m^2 - 5 * m - 24) :=
sorry

end factor_of_quadratic_expression_l213_213347


namespace mark_charged_more_hours_l213_213275

theorem mark_charged_more_hours (P K M : ℕ) 
  (h_total : P + K + M = 144)
  (h_pat_kate : P = 2 * K)
  (h_pat_mark : P = M / 3) : M - K = 80 := 
by
  sorry

end mark_charged_more_hours_l213_213275


namespace notebook_distribution_l213_213020

theorem notebook_distribution (x : ℕ) : 
  (∃ k₁ : ℕ, x = 3 * k₁ + 1) ∧ (∃ k₂ : ℕ, x = 4 * k₂ - 2) → (x - 1) / 3 = (x + 2) / 4 :=
by
  sorry

end notebook_distribution_l213_213020


namespace simplify_expression_l213_213484

theorem simplify_expression : 4 * Real.sqrt 5 + Real.sqrt 45 - Real.sqrt 8 + 4 * Real.sqrt 2 = 7 * Real.sqrt 5 + 2 * Real.sqrt 2 :=
by sorry

end simplify_expression_l213_213484


namespace value_of_mathematics_l213_213882

def letter_value (n : ℕ) : ℤ :=
  -- The function to assign values based on position modulo 8
  match n % 8 with
  | 1 => 1
  | 2 => 2
  | 3 => 3
  | 4 => 0
  | 5 => -1
  | 6 => -2
  | 7 => -3
  | 0 => 0
  | _ => 0 -- This case is practically unreachable

def letter_position (c : Char) : ℕ :=
  -- The function to find the position of a character in the alphabet
  c.toNat - 'a'.toNat + 1

def value_of_word (word : String) : ℤ :=
  -- The function to calculate the sum of values of letters in the word
  word.foldr (fun c acc => acc + letter_value (letter_position c)) 0

theorem value_of_mathematics : value_of_word "mathematics" = 6 := 
  by
    sorry -- Proof to be completed

end value_of_mathematics_l213_213882


namespace find_x4_y4_l213_213065

theorem find_x4_y4 (x y : ℝ) (h1 : x + y = 5) (h2 : x^2 + y^2 = 15) : x^4 + y^4 = 175 := by
  sorry

end find_x4_y4_l213_213065


namespace find_original_intensity_l213_213096

variable (I : ℝ)  -- Define intensity of the original red paint (in percentage).

-- Conditions:
variable (fractionReplaced : ℝ) (newIntensity : ℝ) (replacingIntensity : ℝ)
  (fractionReplaced_eq : fractionReplaced = 0.8)
  (newIntensity_eq : newIntensity = 30)
  (replacingIntensity_eq : replacingIntensity = 25)

-- Theorem statement:
theorem find_original_intensity :
  (1 - fractionReplaced) * I + fractionReplaced * replacingIntensity = newIntensity → I = 50 :=
sorry

end find_original_intensity_l213_213096


namespace cars_on_river_road_l213_213901

theorem cars_on_river_road (B C : ℕ) (h1 : B = C - 40) (h2 : B * 3 = C) : C = 60 := 
sorry

end cars_on_river_road_l213_213901


namespace discount_price_equation_correct_l213_213556

def original_price := 200
def final_price := 148
variable (a : ℝ) -- assuming a is a real number representing the percentage discount

theorem discount_price_equation_correct :
  original_price * (1 - a / 100) ^ 2 = final_price :=
sorry

end discount_price_equation_correct_l213_213556


namespace price_of_candied_grape_l213_213316

theorem price_of_candied_grape (x : ℝ) (h : 15 * 2 + 12 * x = 48) : x = 1.5 :=
by
  sorry

end price_of_candied_grape_l213_213316


namespace solve_inequality_l213_213424

theorem solve_inequality (a b : ℝ) (h : ∀ x, (x > 1 ∧ x < 2) ↔ (x - a) * (x - b) < 0) : a + b = 3 :=
sorry

end solve_inequality_l213_213424


namespace least_stamps_l213_213807

theorem least_stamps (s t : ℕ) (h : 5 * s + 7 * t = 48) : s + t = 8 :=
by sorry

end least_stamps_l213_213807


namespace number_of_candidates_l213_213128

-- Definitions for the given conditions
def total_marks : ℝ := 2000
def average_marks : ℝ := 40

-- Theorem to prove the number of candidates
theorem number_of_candidates : total_marks / average_marks = 50 := by
  sorry

end number_of_candidates_l213_213128


namespace determine_c_square_of_binomial_l213_213458

theorem determine_c_square_of_binomial (c : ℝ) : (∀ x : ℝ, 16 * x^2 + 40 * x + c = (4 * x + 5)^2) → c = 25 :=
by
  intro h
  have key := h 0
  -- By substitution, we skip the expansion steps and immediately conclude the value of c
  sorry

end determine_c_square_of_binomial_l213_213458


namespace sum_of_first_3030_terms_l213_213286

-- Define geometric sequence sum for n terms
noncomputable def geom_sum (a r : ℝ) (n : ℕ) : ℝ :=
  a * (1 - r ^ n) / (1 - r)

-- Given conditions
axiom geom_sum_1010 (a r : ℝ) (hr : r ≠ 1) : geom_sum a r 1010 = 100
axiom geom_sum_2020 (a r : ℝ) (hr : r ≠ 1) : geom_sum a r 2020 = 190

-- Prove that the sum of the first 3030 terms is 271
theorem sum_of_first_3030_terms (a r : ℝ) (hr : r ≠ 1) :
  geom_sum a r 3030 = 271 :=
by
  sorry

end sum_of_first_3030_terms_l213_213286


namespace area_of_square_l213_213294

theorem area_of_square (d : ℝ) (hd : d = 14 * Real.sqrt 2) : 
  ∃ (A : ℝ), A = 196 := by
  sorry

end area_of_square_l213_213294


namespace half_angle_in_first_quadrant_l213_213449

theorem half_angle_in_first_quadrant (α : ℝ) (h : 0 < α ∧ α < π / 2) : 0 < α / 2 ∧ α / 2 < π / 4 :=
by
  sorry

end half_angle_in_first_quadrant_l213_213449


namespace frustum_volume_correct_l213_213346

noncomputable def base_length := 20 -- cm
noncomputable def base_width := 10 -- cm
noncomputable def original_altitude := 12 -- cm
noncomputable def cut_height := 6 -- cm
noncomputable def base_area := base_length * base_width -- cm^2
noncomputable def original_volume := (1 / 3 : ℚ) * base_area * original_altitude -- cm^3
noncomputable def top_area := base_area / 4 -- cm^2
noncomputable def smaller_pyramid_volume := (1 / 3 : ℚ) * top_area * cut_height -- cm^3
noncomputable def frustum_volume := original_volume - smaller_pyramid_volume -- cm^3

theorem frustum_volume_correct :
  frustum_volume = 700 :=
by
  sorry

end frustum_volume_correct_l213_213346


namespace tiffany_ate_pies_l213_213958

theorem tiffany_ate_pies (baking_days : ℕ) (pies_per_day : ℕ) (wc_per_pie : ℕ) 
                         (remaining_wc : ℕ) (total_pies : ℕ) (total_wc : ℕ) :
  baking_days = 11 → pies_per_day = 3 → wc_per_pie = 2 → remaining_wc = 58 →
  total_pies = pies_per_day * baking_days → total_wc = total_pies * wc_per_pie →
  (total_wc - remaining_wc) / wc_per_pie = 4 :=
by 
  intros h1 h2 h3 h4 h5 h6
  sorry

end tiffany_ate_pies_l213_213958


namespace annual_interest_rate_l213_213137

theorem annual_interest_rate 
  (P A : ℝ) 
  (hP : P = 136) 
  (hA : A = 150) 
  : (A - P) / P = 0.10 :=
by sorry

end annual_interest_rate_l213_213137


namespace complex_square_simplification_l213_213793

theorem complex_square_simplification (i : ℂ) (h : i^2 = -1) : (4 - 3 * i)^2 = 7 - 24 * i :=
by {
  sorry
}

end complex_square_simplification_l213_213793


namespace remaining_area_is_correct_l213_213781

-- Define the given conditions:
def original_length : ℕ := 25
def original_width : ℕ := 35
def square_side : ℕ := 7

-- Define a function to calculate the area of the original cardboard:
def area_original : ℕ := original_length * original_width

-- Define a function to calculate the area of one square corner:
def area_corner : ℕ := square_side * square_side

-- Define a function to calculate the total area removed:
def total_area_removed : ℕ := 4 * area_corner

-- Define a function to calculate the remaining area:
def area_remaining : ℕ := area_original - total_area_removed

-- The theorem we want to prove:
theorem remaining_area_is_correct : area_remaining = 679 := by
  -- Here, we would provide the proof if required, but we use sorry for now.
  sorry

end remaining_area_is_correct_l213_213781


namespace integer_roots_l213_213583

-- Define the polynomial
def poly (x : ℤ) : ℤ := x^3 - 4 * x^2 - 11 * x + 24

-- State the theorem
theorem integer_roots : {x : ℤ | poly x = 0} = {-1, 2, 3} := 
  sorry

end integer_roots_l213_213583


namespace fixed_monthly_fee_december_l213_213200

theorem fixed_monthly_fee_december (x y : ℝ) 
    (h1 : x + y = 15.00) 
    (h2 : x + 2 + 3 * y = 25.40) : 
    x = 10.80 :=
by
  sorry

end fixed_monthly_fee_december_l213_213200


namespace quadratic_roots_relation_l213_213002

variable (a b c X1 X2 : ℝ)

theorem quadratic_roots_relation (h : a ≠ 0) : 
  (X1 + X2 = -b / a) ∧ (X1 * X2 = c / a) :=
sorry

end quadratic_roots_relation_l213_213002


namespace geometric_sequence_S6_div_S3_l213_213738

theorem geometric_sequence_S6_div_S3 (a : ℕ → ℝ) 
  (S : ℕ → ℝ)
  (h1 : a 1 + a 3 = 5 / 4)
  (h2 : a 2 + a 4 = 5 / 2)
  (hS : ∀ n, S n = a 1 * (1 - (2:ℝ) ^ n) / (1 - 2)) :
  S 6 / S 3 = 9 :=
by
  sorry

end geometric_sequence_S6_div_S3_l213_213738


namespace simplify_evaluate_l213_213914

def f (x y : ℝ) : ℝ := 4 * x^2 * y - (6 * x * y - 3 * (4 * x - 2) - x^2 * y) + 1

theorem simplify_evaluate : f (-2) (1/2) = -13 := by
  sorry

end simplify_evaluate_l213_213914


namespace strip_covers_cube_l213_213081

   -- Define the given conditions
   def strip_length := 12
   def strip_width := 1
   def cube_edge := 1
   def layers := 2

   -- Define the main statement to be proved
   theorem strip_covers_cube : 
     (strip_length >= 6 * cube_edge / layers) ∧ 
     (strip_width >= cube_edge) ∧ 
     (layers == 2) → 
     true :=
   by
     intro h
     sorry
   
end strip_covers_cube_l213_213081


namespace mark_total_spending_l213_213444

variable (p_tomato_cost : ℕ) (p_apple_cost : ℕ) 
variable (pounds_tomato : ℕ) (pounds_apple : ℕ)

def total_cost (p_tomato_cost : ℕ) (pounds_tomato : ℕ) (p_apple_cost : ℕ) (pounds_apple : ℕ) : ℕ :=
  (p_tomato_cost * pounds_tomato) + (p_apple_cost * pounds_apple)

theorem mark_total_spending :
  total_cost 5 2 6 5 = 40 :=
by
  sorry

end mark_total_spending_l213_213444


namespace geometric_common_ratio_l213_213597

noncomputable def geo_sum (a₁ q : ℝ) (n : ℕ) : ℝ :=
  a₁ * (1 - q^n) / (1 - q)

theorem geometric_common_ratio (a₁ : ℝ) (q : ℝ) (n : ℕ) 
  (h : 2 * geo_sum a₁ q n = geo_sum a₁ q (n + 1) + geo_sum a₁ q (n + 2)) : q = -2 :=
by
  sorry

end geometric_common_ratio_l213_213597


namespace find_x_l213_213014

theorem find_x (x : ℕ) : 3 * 2^x + 5 * 2^x = 2048 → x = 8 := by
  sorry

end find_x_l213_213014


namespace scrabble_score_l213_213245

-- Definitions derived from conditions
def value_first_and_third : ℕ := 1
def value_middle : ℕ := 8
def multiplier : ℕ := 3

-- Prove the total points earned by Jeremy
theorem scrabble_score : (value_first_and_third * 2 + value_middle) * multiplier = 30 :=
by
  sorry

end scrabble_score_l213_213245


namespace calculate_expression_l213_213084

theorem calculate_expression :
  (Real.sqrt 3) ^ 0 + 2 ^ (-1 : ℤ) + Real.sqrt 2 * Real.cos (Real.pi / 4) - |(-1:ℝ) / 2| = 2 := 
by
  sorry

end calculate_expression_l213_213084


namespace correct_student_mark_l213_213589

theorem correct_student_mark :
  ∀ (total_marks total_correct_marks incorrect_mark correct_average students : ℝ)
  (h1 : total_marks = students * 100)
  (h2 : incorrect_mark = 60)
  (h3 : correct_average = 95)
  (h4 : total_correct_marks = students * correct_average),
  total_marks - incorrect_mark + (total_correct_marks - (total_marks - incorrect_mark)) = 10 :=
by
  intros total_marks total_correct_marks incorrect_mark correct_average students h1 h2 h3 h4
  sorry

end correct_student_mark_l213_213589


namespace like_terms_sum_l213_213701

theorem like_terms_sum (m n : ℕ) (h1 : 6 * x ^ 5 * y ^ (2 * n) = 6 * x ^ m * y ^ 4) : m + n = 7 := by
  sorry

end like_terms_sum_l213_213701


namespace algebra_expression_value_l213_213508

theorem algebra_expression_value (a : ℝ) (h : a^2 - 4 * a - 6 = 0) : a^2 - 4 * a + 3 = 9 :=
by
  sorry

end algebra_expression_value_l213_213508


namespace ways_to_distribute_items_l213_213454

/-- The number of ways to distribute 5 different items into 4 identical bags, with some bags possibly empty, is 36. -/
theorem ways_to_distribute_items : ∃ (n : ℕ), n = 36 := by
  sorry

end ways_to_distribute_items_l213_213454


namespace multiply_exp_result_l213_213233

theorem multiply_exp_result : 121 * (5 ^ 4) = 75625 :=
by
  sorry

end multiply_exp_result_l213_213233


namespace exists_term_not_of_form_l213_213764

theorem exists_term_not_of_form (a d : ℕ) (h_seq : ∀ i j : ℕ, (i < 40 ∧ j < 40 ∧ i ≠ j) → a + i * d ≠ a + j * d)
  (pos_a : a > 0) (pos_d : d > 0) 
  : ∃ h : ℕ, h < 40 ∧ ¬ ∃ k l : ℕ, a + h * d = 2^k + 3^l :=
by {
  sorry
}

end exists_term_not_of_form_l213_213764


namespace abs_b_leq_one_l213_213947

theorem abs_b_leq_one (a b : ℝ) (h : ∀ x : ℝ, a * Real.cos x + b * Real.cos (3 * x) ≤ 1) : |b| ≤ 1 := 
sorry

end abs_b_leq_one_l213_213947


namespace avg_weight_increase_l213_213631

theorem avg_weight_increase
  (A : ℝ) -- Initial average weight
  (n : ℕ) -- Initial number of people
  (w_old : ℝ) -- Weight of the person being replaced
  (w_new : ℝ) -- Weight of the new person
  (h_n : n = 8) -- Initial number of people is 8
  (h_w_old : w_old = 85) -- Weight of the replaced person is 85
  (h_w_new : w_new = 105) -- Weight of the new person is 105
  : ((8 * A + (w_new - w_old)) / 8) - A = 2.5 := 
sorry

end avg_weight_increase_l213_213631


namespace min_value_expression_l213_213143

noncomputable def expression (x y : ℝ) := 2 * x^2 + 3 * x * y + 4 * y^2 - 8 * x - 6 * y

theorem min_value_expression : ∀ x y : ℝ, expression x y ≥ -14 :=
by
  sorry

end min_value_expression_l213_213143


namespace solution_interval_l213_213899

theorem solution_interval (x : ℝ) : (x^2 / (x - 5)^2 > 0) ↔ (x ∈ Set.Iio 0 ∪ Set.Ioi 0 ∩ Set.Iio 5 ∪ Set.Ioi 5) :=
by
  sorry

end solution_interval_l213_213899


namespace solve_fractional_eq_l213_213579

noncomputable def fractional_eq (x : ℝ) : Prop := 
  (3 / (x^2 - 3 * x) + (x - 1) / (x - 3) = 1)

noncomputable def not_zero_denom (x : ℝ) : Prop := 
  (x^2 - 3 * x ≠ 0) ∧ (x - 3 ≠ 0)

theorem solve_fractional_eq : fractional_eq (-3/2) ∧ not_zero_denom (-3/2) :=
by
  sorry

end solve_fractional_eq_l213_213579


namespace solve_for_n_l213_213915

def number_of_balls : ℕ := sorry

axiom A : number_of_balls = 2

theorem solve_for_n (n : ℕ) (h : (1 + 1 + n = number_of_balls) ∧ ((n : ℝ) / (1 + 1 + n) = 1 / 2)) : n = 2 :=
sorry

end solve_for_n_l213_213915


namespace max_smaller_boxes_fit_l213_213004

theorem max_smaller_boxes_fit (length_large width_large height_large : ℝ)
  (length_small width_small height_small : ℝ)
  (h1 : length_large = 6)
  (h2 : width_large = 5)
  (h3 : height_large = 4)
  (hs1 : length_small = 0.60)
  (hs2 : width_small = 0.50)
  (hs3 : height_small = 0.40) :
  length_large * width_large * height_large / (length_small * width_small * height_small) = 1000 := 
  by
  sorry

end max_smaller_boxes_fit_l213_213004


namespace percentage_of_uninsured_part_time_l213_213670

noncomputable def number_of_employees := 330
noncomputable def uninsured_employees := 104
noncomputable def part_time_employees := 54
noncomputable def probability_neither := 0.5606060606060606

theorem percentage_of_uninsured_part_time:
  (13 / 104) * 100 = 12.5 := 
by 
  -- Here you can assume proof steps would occur/assertions to align with the solution found
  sorry

end percentage_of_uninsured_part_time_l213_213670


namespace smallest_n_satisfying_conditions_l213_213839

theorem smallest_n_satisfying_conditions : ∃ (n : ℕ), (n > 0) ∧ (∀ (a b : ℤ), ∃ (α β : ℝ), 
    α ≠ β ∧ (0 < α) ∧ (α < 1) ∧ (0 < β) ∧ (β < 1) ∧ (n * α^2 + a * α + b = 0) ∧ (n * β^2 + a * β + b = 0)
 ) ∧ (∀ (m : ℕ), m > 0 ∧ m < n → ¬ (∀ (a b : ℤ), ∃ (α β : ℝ), 
    α ≠ β ∧ (0 < α) ∧ (α < 1) ∧ (0 < β) ∧ (β < 1) ∧ (m * α^2 + a * α + b = 0) ∧ (m * β^2 + a * β + b = 0))) := 
sorry

end smallest_n_satisfying_conditions_l213_213839


namespace no_six_odd_numbers_sum_to_one_l213_213968

theorem no_six_odd_numbers_sum_to_one (a b c d e f : ℕ)
  (ha : a % 2 = 1) (hb : b % 2 = 1) (hc : c % 2 = 1) (hd : d % 2 = 1) (he : e % 2 = 1) (hf : f % 2 = 1)
  (h_diff : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ d ≠ e ∧ d ≠ f ∧ e ≠ f)
  (h_pos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ 0 < e ∧ 0 < f) :
  (1 / a : ℝ) + 1 / b + 1 / c + 1 / d + 1 / e + 1 / f ≠ 1 :=
by
  sorry

end no_six_odd_numbers_sum_to_one_l213_213968


namespace find_y_value_l213_213045

theorem find_y_value
  (y z : ℝ)
  (h1 : y + z + 175 = 360)
  (h2 : z = y + 10) :
  y = 88 :=
by
  sorry

end find_y_value_l213_213045


namespace at_least_one_angle_ge_60_l213_213737

theorem at_least_one_angle_ge_60 (A B C : ℝ) (hA : A < 60) (hB : B < 60) (hC : C < 60) (h_sum : A + B + C = 180) : false :=
sorry

end at_least_one_angle_ge_60_l213_213737


namespace carrots_total_l213_213057

variables (initiallyPicked : Nat) (thrownOut : Nat) (pickedNextDay : Nat)

def totalCarrots (initiallyPicked : Nat) (thrownOut : Nat) (pickedNextDay : Nat) :=
  initiallyPicked - thrownOut + pickedNextDay

theorem carrots_total (h1 : initiallyPicked = 19)
                     (h2 : thrownOut = 4)
                     (h3 : pickedNextDay = 46) :
  totalCarrots initiallyPicked thrownOut pickedNextDay = 61 :=
by
  sorry

end carrots_total_l213_213057


namespace area_proportions_and_point_on_line_l213_213828

theorem area_proportions_and_point_on_line (T : ℝ × ℝ) :
  (∃ r s : ℝ, T = (r, s) ∧ s = -(5 / 3) * r + 10 ∧ 1 / 2 * 6 * s = 7.5) 
  ↔ T.1 + T.2 = 7 :=
by { sorry }

end area_proportions_and_point_on_line_l213_213828


namespace totalSolutions_l213_213575

noncomputable def systemOfEquations (a b c d a1 b1 c1 d1 x y : ℝ) : Prop :=
  a * x^2 + b * x * y + c * y^2 = d ∧ a1 * x^2 + b1 * x * y + c1 * y^2 = d1

theorem totalSolutions 
  (a b c d a1 b1 c1 d1 : ℝ) 
  (h₀ : a ≠ 0 ∨ b ≠ 0 ∨ c ≠ 0)
  (h₁ : a1 ≠ 0 ∨ b1 ≠ 0 ∨ c1 ≠ 0) :
  ∃ x y : ℝ, systemOfEquations a b c d a1 b1 c1 d1 x y :=
sorry

end totalSolutions_l213_213575


namespace combined_distance_l213_213660

theorem combined_distance (t1 t2 : ℕ) (s1 s2 : ℝ)
  (h1 : t1 = 30) (h2 : s1 = 9.5) (h3 : t2 = 45) (h4 : s2 = 8.3)
  : (s1 * t1 + s2 * t2) = 658.5 := 
by
  sorry

end combined_distance_l213_213660


namespace other_asymptote_l213_213079

/-- Problem Statement:
One of the asymptotes of a hyperbola is y = 2x. The foci have the same 
x-coordinate, which is 4. Prove that the equation of the other asymptote
of the hyperbola is y = -2x + 16.
-/
theorem other_asymptote (focus_x : ℝ) (asymptote1: ℝ → ℝ) (asymptote2 : ℝ → ℝ) :
  focus_x = 4 →
  (∀ x, asymptote1 x = 2 * x) →
  (asymptote2 4 = 8) → 
  (∀ x, asymptote2 x = -2 * x + 16) :=
sorry

end other_asymptote_l213_213079


namespace percentage_increase_l213_213174

-- Define the initial and final prices as constants
def P_inicial : ℝ := 5.00
def P_final : ℝ := 5.55

-- Define the percentage increase proof
theorem percentage_increase : ((P_final - P_inicial) / P_inicial) * 100 = 11 := 
by
  sorry

end percentage_increase_l213_213174


namespace dilation_complex_l213_213658

theorem dilation_complex :
  let c := (1 : ℂ) - (2 : ℂ) * I
  let k := 3
  let z := -1 + I
  (k * (z - c) + c = -5 + 7 * I) :=
by
  sorry

end dilation_complex_l213_213658


namespace range_of_a_l213_213894

theorem range_of_a (a : ℝ) : (∀ x ∈ Set.Icc (-2 : ℝ) 3, 2 * x > x ^ 2 + a) → a < -8 :=
by sorry

end range_of_a_l213_213894


namespace total_balloons_l213_213417

-- Define the number of balloons each person has
def joan_balloons : ℕ := 40
def melanie_balloons : ℕ := 41

-- State the theorem about the total number of balloons
theorem total_balloons : joan_balloons + melanie_balloons = 81 :=
by
  sorry

end total_balloons_l213_213417


namespace calculate_expression_l213_213540

theorem calculate_expression :
  (π - 1)^0 + 4 * Real.sin (Real.pi / 4) - Real.sqrt 8 + abs (-3) = 4 := by
  sorry

end calculate_expression_l213_213540


namespace cosine_identity_example_l213_213815

theorem cosine_identity_example {α : ℝ} (h : Real.sin (π / 3 - α) = 1 / 3) : Real.cos (π / 3 + 2 * α) = -7 / 9 :=
by sorry

end cosine_identity_example_l213_213815


namespace probability_not_touch_outer_edge_l213_213108

def checkerboard : ℕ := 10

def total_squares : ℕ := checkerboard * checkerboard

def perimeter_squares : ℕ := 4 * checkerboard - 4

def inner_squares : ℕ := total_squares - perimeter_squares

def probability : ℚ := inner_squares / total_squares

theorem probability_not_touch_outer_edge : probability = 16 / 25 :=
by
  sorry

end probability_not_touch_outer_edge_l213_213108


namespace part1_part2_part3_l213_213599

noncomputable def f (x : ℝ) : ℝ := 3 * x - Real.exp x + 1

theorem part1 :
  ∃ x0 > 0, f x0 = 0 :=
sorry

theorem part2 (x0 : ℝ) (h1 : f x0 = 0) :
  ∀ x, f x ≤ (3 - Real.exp x0) * (x - x0) :=
sorry

theorem part3 (m x1 x2 : ℝ) (h1 : m > 0) (h2 : x1 < x2) (h3 : f x1 = m) (h4 : f x2 = m):
  x2 - x1 < 2 - 3 * m / 4 :=
sorry

end part1_part2_part3_l213_213599


namespace radius_wire_is_4_cm_l213_213301

noncomputable def radius_of_wire_cross_section (r_sphere : ℝ) (length_wire : ℝ) : ℝ :=
  let volume_sphere := (4 / 3) * Real.pi * r_sphere^3
  let volume_wire := volume_sphere / length_wire
  Real.sqrt (volume_wire / Real.pi)

theorem radius_wire_is_4_cm :
  radius_of_wire_cross_section 12 144 = 4 :=
by
  unfold radius_of_wire_cross_section
  sorry

end radius_wire_is_4_cm_l213_213301


namespace find_a_l213_213876

-- Define what it means for P(X = k) to be given by a particular function
def P (X : ℕ) (a : ℕ) := X / (2 * a)

-- Define the condition on the probabilities
def sum_of_probabilities_is_one (a : ℕ) :=
  (1 / (2 * a) + 2 / (2 * a) + 3 / (2 * a) + 4 / (2 * a)) = 1

-- The theorem to prove
theorem find_a (a : ℕ) (h : sum_of_probabilities_is_one a) : a = 5 :=
by sorry

end find_a_l213_213876


namespace proof_problem_l213_213254

open Real

noncomputable def problem_condition1 (A B : ℝ) : Prop :=
  (sin A - sin B) * (sin A + sin B) = sin (π/3 - B) * sin (π/3 + B)

noncomputable def problem_condition2 (b c : ℝ) (a : ℝ) (dot_product : ℝ) : Prop :=
  b * c * cos (π / 3) = dot_product ∧ a = 2 * sqrt 7

noncomputable def problem_condition3 (a b c : ℝ) : Prop := 
  a^2 = (b + c)^2 - 3 * b * c

noncomputable def problem_condition4 (b c : ℝ) : Prop := 
  b < c

theorem proof_problem (A B : ℝ) (a b c dot_product : ℝ)
  (h1 : problem_condition1 A B)
  (h2 : problem_condition2 b c a dot_product)
  (h3 : problem_condition3 a b c)
  (h4 : problem_condition4 b c) :
  (A = π / 3) ∧ (b = 4 ∧ c = 6) :=
by {
  sorry
}

end proof_problem_l213_213254


namespace johns_overall_loss_l213_213854

noncomputable def johns_loss_percentage : ℝ :=
  let cost_A := 1000 * 2
  let cost_B := 1500 * 3
  let cost_C := 2000 * 4
  let discount_A := 0.1
  let discount_B := 0.15
  let discount_C := 0.2
  let cost_A_after_discount := cost_A * (1 - discount_A)
  let cost_B_after_discount := cost_B * (1 - discount_B)
  let cost_C_after_discount := cost_C * (1 - discount_C)
  let total_cost_after_discount := cost_A_after_discount + cost_B_after_discount + cost_C_after_discount
  let import_tax_rate := 0.08
  let import_tax := total_cost_after_discount * import_tax_rate
  let total_cost_incl_tax := total_cost_after_discount + import_tax
  let cost_increase_rate_C := 0.04
  let new_cost_C := 2000 * (4 + 4 * cost_increase_rate_C)
  let adjusted_total_cost := cost_A_after_discount + cost_B_after_discount + new_cost_C
  let total_selling_price := (800 * 3) + (70 * 3 + 1400 * 3.5 + 900 * 5) + (130 * 2.5 + 130 * 3 + 130 * 5)
  let gain_or_loss := total_selling_price - adjusted_total_cost
  let loss_percentage := (gain_or_loss / adjusted_total_cost) * 100
  loss_percentage

theorem johns_overall_loss : abs (johns_loss_percentage + 4.09) < 0.01 := sorry

end johns_overall_loss_l213_213854


namespace correct_factorization_l213_213350

-- Definitions of the options given in the problem
def optionA (a : ℝ) := a^3 - a = a * (a^2 - 1)
def optionB (a b : ℝ) := a^2 - 4 * b^2 = (a + 4 * b) * (a - 4 * b)
def optionC (a : ℝ) := a^2 - 2 * a - 8 = a * (a - 2) - 8
def optionD (a : ℝ) := a^2 - a + 1/4 = (a - 1/2)^2

-- Stating the proof problem
theorem correct_factorization : ∀ (a : ℝ), optionD a :=
by
  sorry

end correct_factorization_l213_213350


namespace total_lines_correct_l213_213067

-- Define the shapes and their corresponding lines
def triangles := 12
def squares := 8
def pentagons := 4
def hexagons := 6
def octagons := 2

def triangle_sides := 3
def square_sides := 4
def pentagon_sides := 5
def hexagon_sides := 6
def octagon_sides := 8

def lines_in_triangles := triangles * triangle_sides
def lines_in_squares := squares * square_sides
def lines_in_pentagons := pentagons * pentagon_sides
def lines_in_hexagons := hexagons * hexagon_sides
def lines_in_octagons := octagons * octagon_sides

def shared_lines_ts := 5
def shared_lines_ph := 3
def shared_lines_ho := 1

def total_lines_triangles := lines_in_triangles - shared_lines_ts
def total_lines_squares := lines_in_squares - shared_lines_ts
def total_lines_pentagons := lines_in_pentagons - shared_lines_ph
def total_lines_hexagons := lines_in_hexagons - shared_lines_ph - shared_lines_ho
def total_lines_octagons := lines_in_octagons - shared_lines_ho

-- The statement to prove
theorem total_lines_correct :
  total_lines_triangles = 31 ∧
  total_lines_squares = 27 ∧
  total_lines_pentagons = 17 ∧
  total_lines_hexagons = 32 ∧
  total_lines_octagons = 15 :=
by sorry

end total_lines_correct_l213_213067


namespace area_closed_figure_sqrt_x_x_cube_l213_213385

noncomputable def integral_diff_sqrt_x_cube (a b : ℝ) :=
∫ x in a..b, (Real.sqrt x - x^3)

theorem area_closed_figure_sqrt_x_x_cube :
  integral_diff_sqrt_x_cube 0 1 = 5 / 12 :=
by
  sorry

end area_closed_figure_sqrt_x_x_cube_l213_213385


namespace trajectory_line_or_hyperbola_l213_213382

theorem trajectory_line_or_hyperbola
  (a b : ℝ)
  (ab_pos : a * b > 0)
  (f : ℝ → ℝ)
  (hf : ∀ x, f x = a * x^2 + b) :
  (∃ s t : ℝ, f (s-t) * f (s+t) = (f s)^2) →
  (∃ s t : ℝ, ((t = 0) ∨ (a * t^2 - 2 * a * s^2 + 2 * b = 0))) → true := sorry

end trajectory_line_or_hyperbola_l213_213382


namespace total_silver_dollars_l213_213633

-- Definitions based on conditions
def chiu_silver_dollars : ℕ := 56
def phung_silver_dollars : ℕ := chiu_silver_dollars + 16
def ha_silver_dollars : ℕ := phung_silver_dollars + 5

-- Theorem statement
theorem total_silver_dollars : chiu_silver_dollars + phung_silver_dollars + ha_silver_dollars = 205 :=
by
  -- We use "sorry" to fill in the proof part as instructed
  sorry

end total_silver_dollars_l213_213633


namespace find_values_of_M_l213_213664

theorem find_values_of_M :
  ∃ M : ℕ, 
    (M = 81 ∨ M = 92) ∧ 
    (∃ a b : ℕ, 1 ≤ a ∧ a ≤ 9 ∧ 1 ≤ b ∧ b ≤ 9 ∧ M = 10 * a + b ∧
     (∃ k : ℕ, k ^ 3 = 9 * (a - b) ∧ k > 0)) :=
sorry

end find_values_of_M_l213_213664


namespace average_book_width_l213_213758

-- Define the widths of the books as given in the problem conditions
def widths : List ℝ := [3, 7.5, 1.25, 0.75, 4, 12]

-- Define the number of books from the problem conditions
def num_books : ℝ := 6

-- We prove that the average width of the books is equal to 4.75
theorem average_book_width : (widths.sum / num_books) = 4.75 :=
by
  sorry

end average_book_width_l213_213758


namespace greatest_possible_bent_strips_l213_213941

theorem greatest_possible_bent_strips (strip_count : ℕ) (cube_length cube_faces flat_strip_cover : ℕ) 
  (unit_squares_per_face total_squares flat_strips unit_squares_covered_by_flats : ℕ):
  strip_count = 18 →
  cube_length = 3 →
  cube_faces = 6 →
  flat_strip_cover = 3 →
  unit_squares_per_face = cube_length * cube_length →
  total_squares = cube_faces * unit_squares_per_face →
  flat_strips = 4 →
  unit_squares_covered_by_flats = flat_strips * flat_strip_cover →
  ∃ bent_strips,
  flat_strips * flat_strip_cover + bent_strips * flat_strip_cover = total_squares 
  ∧ bent_strips = 14 := by
  intros
  -- skipped proof
  sorry

end greatest_possible_bent_strips_l213_213941


namespace derivative_at_pi_div_2_l213_213992

noncomputable def f (x : ℝ) : ℝ := 3 * Real.sin x - 4 * Real.cos x

theorem derivative_at_pi_div_2 : (deriv f (Real.pi / 2)) = 4 := 
by
  sorry

end derivative_at_pi_div_2_l213_213992


namespace number_of_social_science_papers_selected_is_18_l213_213859

def total_social_science_papers : ℕ := 54
def total_humanities_papers : ℕ := 60
def total_other_papers : ℕ := 39
def total_selected_papers : ℕ := 51

def number_of_social_science_papers_selected : ℕ :=
  (total_social_science_papers * total_selected_papers) / (total_social_science_papers + total_humanities_papers + total_other_papers)

theorem number_of_social_science_papers_selected_is_18 :
  number_of_social_science_papers_selected = 18 :=
by 
  -- Proof to be provided
  sorry

end number_of_social_science_papers_selected_is_18_l213_213859


namespace courses_students_problem_l213_213826

theorem courses_students_problem :
  let courses := Fin 6 -- represent 6 courses
  let students := Fin 20 -- represent 20 students
  (∀ (C C' : courses), ∀ (S : Finset students), S.card = 5 → 
    ¬ ((∀ s ∈ S, ∃ s_courses : Finset courses, C ∈ s_courses ∧ C' ∈ s_courses) ∨ 
       (∀ s ∈ S, ∃ s_courses : Finset courses, C ∉ s_courses ∧ C' ∉ s_courses))) :=
by sorry

end courses_students_problem_l213_213826


namespace dwarfs_truthful_count_l213_213272

theorem dwarfs_truthful_count (x y : ℕ)
  (h1 : x + y = 10)
  (h2 : x + 2 * y = 16) :
  x = 4 :=
by
  sorry

end dwarfs_truthful_count_l213_213272


namespace misha_scored_48_in_second_attempt_l213_213927

theorem misha_scored_48_in_second_attempt (P1 P2 P3 : ℕ)
  (h1 : P2 = 2 * P1)
  (h2 : P3 = (3 / 2) * P2)
  (h3 : 24 ≤ P1)
  (h4 : (3 / 2) * 2 * P1 = 72) : P2 = 48 :=
by sorry

end misha_scored_48_in_second_attempt_l213_213927


namespace smallest_positive_integer_x_l213_213844

def smallest_x (x : ℕ) : Prop :=
  x > 0 ∧ (450 * x) % 625 = 0

theorem smallest_positive_integer_x :
  ∃ x : ℕ, smallest_x x ∧ ∀ y : ℕ, smallest_x y → x ≤ y ∧ x = 25 :=
by {
  sorry
}

end smallest_positive_integer_x_l213_213844


namespace average_price_initial_l213_213981

noncomputable def total_cost_initial (P : ℕ) := 5 * P
noncomputable def total_cost_remaining := 3 * 12
noncomputable def total_cost_returned := 2 * 32

theorem average_price_initial (P : ℕ) : total_cost_initial P = total_cost_remaining + total_cost_returned → P = 20 := 
by
  sorry

end average_price_initial_l213_213981


namespace sum_of_remainders_eq_24_l213_213586

theorem sum_of_remainders_eq_24 (a b c : ℕ) 
  (h1 : a % 30 = 13) (h2 : b % 30 = 19) (h3 : c % 30 = 22) :
  (a + b + c) % 30 = 24 :=
by
  sorry

end sum_of_remainders_eq_24_l213_213586


namespace bianca_points_earned_l213_213749

-- Define the constants and initial conditions
def points_per_bag : ℕ := 5
def total_bags : ℕ := 17
def not_recycled_bags : ℕ := 8

-- Define a function to calculate the number of recycled bags
def recycled_bags (total: ℕ) (not_recycled: ℕ) : ℕ :=
  total - not_recycled

-- Define a function to calculate the total points earned
def total_points_earned (bags: ℕ) (points_per_bag: ℕ) : ℕ :=
  bags * points_per_bag

-- State the theorem
theorem bianca_points_earned : total_points_earned (recycled_bags total_bags not_recycled_bags) points_per_bag = 45 :=
by
  sorry

end bianca_points_earned_l213_213749


namespace updated_mean_of_decrement_l213_213199

theorem updated_mean_of_decrement 
  (mean_initial : ℝ)
  (num_observations : ℕ)
  (decrement_per_observation : ℝ)
  (h1 : mean_initial = 200)
  (h2 : num_observations = 50)
  (h3 : decrement_per_observation = 6) : 
  (mean_initial * num_observations - decrement_per_observation * num_observations) / num_observations = 194 :=
by
  sorry

end updated_mean_of_decrement_l213_213199


namespace area_of_trapezoid_l213_213812

noncomputable def triangle_XYZ_is_isosceles : Prop := 
  ∃ (X Y Z : Type) (XY XZ : ℝ), XY = XZ

noncomputable def identical_smaller_triangles (area : ℝ) (num : ℕ) : Prop := 
  num = 9 ∧ area = 3

noncomputable def total_area_large_triangle (total_area : ℝ) : Prop := 
  total_area = 135

noncomputable def trapezoid_contains_smaller_triangles (contained : ℕ) : Prop :=
  contained = 4

theorem area_of_trapezoid (XYZ_area smaller_triangle_area : ℝ) 
    (num_smaller_triangles contained_smaller_triangles : ℕ) : 
    triangle_XYZ_is_isosceles → 
    identical_smaller_triangles smaller_triangle_area num_smaller_triangles →
    total_area_large_triangle XYZ_area →
    trapezoid_contains_smaller_triangles contained_smaller_triangles →
    (XYZ_area - contained_smaller_triangles * smaller_triangle_area) = 123 :=
by
  intros iso smaller_triangles total_area contained
  sorry

end area_of_trapezoid_l213_213812


namespace christina_rearrangements_l213_213810

-- define the main conditions
def rearrangements (n : Nat) : Nat := Nat.factorial n

def half (n : Nat) : Nat := n / 2

def time_for_first_half (r : Nat) : Nat := r / 12

def time_for_second_half (r : Nat) : Nat := r / 18

def total_time_in_minutes (t1 t2 : Nat) : Nat := t1 + t2

def total_time_in_hours (t : Nat) : Nat := t / 60

-- statement proving that the total time will be 420 hours
theorem christina_rearrangements : 
  rearrangements 9 = 362880 →
  half (rearrangements 9) = 181440 →
  time_for_first_half 181440 = 15120 →
  time_for_second_half 181440 = 10080 →
  total_time_in_minutes 15120 10080 = 25200 →
  total_time_in_hours 25200 = 420 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end christina_rearrangements_l213_213810


namespace problem_equivalent_to_l213_213252

theorem problem_equivalent_to (x : ℝ)
  (A : x^2 = 5*x - 6 ↔ x = 2 ∨ x = 3)
  (B : x^2 - 5*x + 6 = 0 ↔ x = 2 ∨ x = 3)
  (C : x = x + 1 ↔ false)
  (D : x^2 - 5*x + 7 = 1 ↔ x = 2 ∨ x = 3)
  (E : x^2 - 1 = 5*x - 7 ↔ x = 2 ∨ x = 3) :
  ¬ (x = x + 1) :=
by sorry

end problem_equivalent_to_l213_213252


namespace midpoint_trajectory_of_moving_point_l213_213136

/-- Given a fixed point A (4, -3) and a moving point B on the circle (x+1)^2 + y^2 = 4, prove that 
    the equation of the trajectory of the midpoint M of the line segment AB is 
    (x - 3/2)^2 + (y + 3/2)^2 = 1. -/
theorem midpoint_trajectory_of_moving_point {x y : ℝ} :
  (∃ (B : ℝ × ℝ), (B.1 + 1)^2 + B.2^2 = 4 ∧ 
    (x, y) = ((B.1 + 4) / 2, (B.2 - 3) / 2)) →
  (x - 3/2)^2 + (y + 3/2)^2 = 1 :=
by sorry

end midpoint_trajectory_of_moving_point_l213_213136


namespace remainder_when_three_times_number_minus_seven_divided_by_seven_l213_213264

theorem remainder_when_three_times_number_minus_seven_divided_by_seven (n : ℤ) (h : n % 7 = 2) : (3 * n - 7) % 7 = 6 := by
  sorry

end remainder_when_three_times_number_minus_seven_divided_by_seven_l213_213264


namespace avg_wx_l213_213397

theorem avg_wx (w x y : ℝ) (h1 : 3 / w + 3 / x = 3 / y) (h2 : w * x = y) : (w + x) / 2 = 1 / 2 :=
by
  -- omitted proof
  sorry

end avg_wx_l213_213397


namespace speed_ratio_l213_213158

theorem speed_ratio (a b v1 v2 S : ℝ) (h1 : S = a * (v1 + v2)) (h2 : S = b * (v1 - v2)) (h3 : a ≠ b) : 
  v1 / v2 = (a + b) / (b - a) :=
by
  -- proof skipped
  sorry

end speed_ratio_l213_213158


namespace weight_of_a_l213_213845

theorem weight_of_a (a b c d e : ℝ)
  (h1 : (a + b + c) / 3 = 84)
  (h2 : (a + b + c + d) / 4 = 80)
  (h3 : e = d + 8)
  (h4 : (b + c + d + e) / 4 = 79) :
  a = 80 :=
by
  sorry

end weight_of_a_l213_213845


namespace seashells_initial_count_l213_213241

theorem seashells_initial_count (S : ℝ) (h : S + 4.0 = 10) : S = 6.0 :=
by
  sorry

end seashells_initial_count_l213_213241


namespace solve_fraction_problem_l213_213206

noncomputable def x_value (a b c d : ℤ) : ℝ :=
  (a + b * Real.sqrt c) / d

theorem solve_fraction_problem (a b c d : ℤ) (h1 : x_value a b c d = (5 + 5 * Real.sqrt 5) / 4)
  (h2 : (4 * x_value a b c d) / 5 - 2 = 5 / x_value a b c d) :
  (a * c * d) / b = 20 := by
  sorry

end solve_fraction_problem_l213_213206


namespace parabola_equation_l213_213537

theorem parabola_equation (M : ℝ × ℝ) (hM : M = (5, 3))
    (h_dist : ∀ a : ℝ, |5 + 1/(4*a)| = 6) :
    (y = (1/12)*x^2) ∨ (y = -(1/36)*x^2) :=
sorry

end parabola_equation_l213_213537


namespace cds_probability_l213_213530

def probability (total favorable : ℕ) : ℚ := favorable / total

theorem cds_probability :
  probability 120 24 = 1 / 5 :=
by
  sorry

end cds_probability_l213_213530


namespace bitcoin_donation_l213_213111

theorem bitcoin_donation (x : ℝ) (h : 3 * (80 - x) / 2 - 10 = 80) : x = 20 :=
sorry

end bitcoin_donation_l213_213111


namespace evaluate_polynomial_at_neg_one_l213_213944

-- Define the polynomial f(x)
def f (x : ℝ) : ℝ := 1 + 2 * x + x^2 - 3 * x^3 + 2 * x^4

-- Define the value x at which we want to evaluate f
def x_val : ℝ := -1

-- State the theorem with the result using Horner's method
theorem evaluate_polynomial_at_neg_one : f x_val = 6 :=
by
  -- Approach to solution is in solution steps, skipped here
  sorry

end evaluate_polynomial_at_neg_one_l213_213944


namespace inequality_for_distinct_integers_l213_213183

-- Define the necessary variables and conditions
variable {a b c : ℤ}

-- Ensure a, b, and c are pairwise distinct integers
def pairwise_distinct (a b c : ℤ) : Prop := a ≠ b ∧ b ≠ c ∧ c ≠ a

-- The main theorem statement
theorem inequality_for_distinct_integers 
  (h : pairwise_distinct a b c) : 
  (a^3 + b^3 + c^3) / 3 ≥ a * b * c + Real.sqrt (3 * (a * b + b * c + c * a + 1)) :=
by
  sorry

end inequality_for_distinct_integers_l213_213183


namespace survey_population_l213_213823

-- Definitions based on conditions
def number_of_packages := 10
def dozens_per_package := 10
def sets_per_dozen := 12

-- Derived from conditions
def total_sets := number_of_packages * dozens_per_package * sets_per_dozen

-- Populations for the proof
def population_quality : ℕ := total_sets
def population_satisfaction : ℕ := total_sets

-- Proof statement
theorem survey_population:
  (population_quality = 1200) ∧ (population_satisfaction = 1200) := by
  sorry

end survey_population_l213_213823


namespace integer_solutions_for_xyz_l213_213909

theorem integer_solutions_for_xyz (x y z : ℤ) : 
  (x - y - 1)^3 + (y - z - 2)^3 + (z - x + 3)^3 = 18 ↔
  (x = y ∧ y = z) ∨
  (x = y - 1 ∧ y = z) ∨
  (x = y ∧ y = z + 5) ∨
  (x = y + 4 ∧ y = z + 5) ∨
  (x = y + 4 ∧ z = y) ∨
  (x = y - 1 ∧ z = y + 4) :=
by {
  sorry
}

end integer_solutions_for_xyz_l213_213909


namespace distance_karen_covers_l213_213309

theorem distance_karen_covers
  (books_per_shelf : ℕ)
  (shelves : ℕ)
  (distance_to_library : ℕ)
  (h1 : books_per_shelf = 400)
  (h2 : shelves = 4)
  (h3 : distance_to_library = books_per_shelf * shelves) :
  2 * distance_to_library = 3200 := 
by
  sorry

end distance_karen_covers_l213_213309


namespace trees_probability_l213_213450

theorem trees_probability (num_maple num_oak num_birch total_slots total_trees : ℕ) 
                         (maple_count oak_count birch_count : Prop)
                         (prob_correct : Prop) :
  num_maple = 4 →
  num_oak = 5 →
  num_birch = 6 →
  total_trees = 15 →
  total_slots = 10 →
  maple_count → oak_count → birch_count →
  prob_correct →
  (m + n = 57) :=
by
  intros
  sorry

end trees_probability_l213_213450


namespace baseball_football_difference_is_five_l213_213203

-- Define the conditions
def total_cards : ℕ := 125
def baseball_cards : ℕ := 95
def some_more : ℕ := baseball_cards - 3 * (total_cards - baseball_cards)

-- Define the number of football cards
def football_cards : ℕ := total_cards - baseball_cards

-- Define the difference between the number of baseball cards and three times the number of football cards
def difference : ℕ := baseball_cards - 3 * football_cards

-- Statement of the proof
theorem baseball_football_difference_is_five : difference = 5 := 
by
  sorry

end baseball_football_difference_is_five_l213_213203


namespace james_pays_37_50_l213_213539

/-- 
James gets 20 singing lessons.
First lesson is free.
After the first 10 paid lessons, he only needs to pay for every other lesson.
Each lesson costs $5.
His uncle pays for half.
Prove that James pays $37.50.
--/

theorem james_pays_37_50 :
  let first_lessons := 1
  let total_lessons := 20
  let paid_lessons := 10
  let remaining_lessons := total_lessons - first_lessons - paid_lessons
  let paid_remaining_lessons := (remaining_lessons + 1) / 2
  let total_paid_lessons := paid_lessons + paid_remaining_lessons
  let cost_per_lesson := 5
  let total_payment := total_paid_lessons * cost_per_lesson
  let payment_by_james := total_payment / 2
  payment_by_james = 37.5 := 
by
  sorry

end james_pays_37_50_l213_213539


namespace square_divisibility_l213_213099

theorem square_divisibility (n : ℤ) : n^2 % 4 = 0 ∨ n^2 % 4 = 1 := sorry

end square_divisibility_l213_213099


namespace opposite_of_neg_two_thirds_l213_213432

theorem opposite_of_neg_two_thirds : -(- (2 / 3)) = (2 / 3) :=
by
  sorry

end opposite_of_neg_two_thirds_l213_213432


namespace allen_change_l213_213930

-- Define the cost per box and the number of boxes
def cost_per_box : ℕ := 7
def num_boxes : ℕ := 5

-- Define the total cost including the tip
def total_cost := num_boxes * cost_per_box
def tip := total_cost / 7
def total_paid := total_cost + tip

-- Define the amount given to the delivery person
def amount_given : ℕ := 100

-- Define the change received
def change := amount_given - total_paid

-- The statement to prove
theorem allen_change : change = 60 :=
by
  -- sorry is used here to skip the proof, as per the instruction
  sorry

end allen_change_l213_213930


namespace incorrect_conclusion_l213_213966

theorem incorrect_conclusion (a b c : ℝ) (h1 : 0 < a) (h2 : a < 1) (h3 : b > c) (h4 : c > 0) : ¬ (a / b > a / c) :=
sorry

end incorrect_conclusion_l213_213966


namespace percent_decaf_coffee_l213_213363

variable (initial_stock new_stock decaf_initial_percent decaf_new_percent : ℝ)
variable (initial_stock_pos new_stock_pos : initial_stock > 0 ∧ new_stock > 0)

theorem percent_decaf_coffee :
    initial_stock = 400 → 
    decaf_initial_percent = 20 → 
    new_stock = 100 → 
    decaf_new_percent = 60 → 
    (100 * ((decaf_initial_percent / 100 * initial_stock + decaf_new_percent / 100 * new_stock) / (initial_stock + new_stock))) = 28 := 
by
  sorry

end percent_decaf_coffee_l213_213363


namespace sum_of_coefficients_at_1_l213_213117

def P (x : ℝ) := 2 * (4 * x^8 - 3 * x^5 + 9)
def Q (x : ℝ) := 9 * (x^6 + 2 * x^3 - 8)
def R (x : ℝ) := P x + Q x

theorem sum_of_coefficients_at_1 : R 1 = -25 := by
  sorry

end sum_of_coefficients_at_1_l213_213117


namespace gcd_of_differences_is_10_l213_213361

theorem gcd_of_differences_is_10 (a b c : ℕ) (h1 : b > a) (h2 : c > b) (h3 : c > a)
  (h4 : b - a = 20) (h5 : c - b = 50) (h6 : c - a = 70) : Int.gcd (b - a) (Int.gcd (c - b) (c - a)) = 10 := 
sorry

end gcd_of_differences_is_10_l213_213361


namespace root_fraction_power_l213_213943

theorem root_fraction_power (a : ℝ) (ha : a = 5) : 
  (a^(1/3)) / (a^(1/5)) = a^(2/15) := by
  sorry

end root_fraction_power_l213_213943


namespace boxes_A_B_cost_condition_boxes_B_profit_condition_l213_213628

/-
Part 1: Prove the number of brand A boxes is 60 and number of brand B boxes is 40 given the cost condition.
-/
theorem boxes_A_B_cost_condition (x : ℕ) (y : ℕ) :
  80 * x + 130 * y = 10000 ∧ x + y = 100 → x = 60 ∧ y = 40 :=
by sorry

/-
Part 2: Prove the number of brand B boxes should be at least 54 given the profit condition.
-/
theorem boxes_B_profit_condition (y : ℕ) :
  40 * (100 - y) + 70 * y ≥ 5600 → y ≥ 54 :=
by sorry

end boxes_A_B_cost_condition_boxes_B_profit_condition_l213_213628


namespace min_value_ax_over_rR_l213_213421

theorem min_value_ax_over_rR (a b c r R : ℝ) (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
  (h_le_b : a ≤ b) (h_le_c : a ≤ c) (h_inradius : ∀ (a b c : ℝ), r = 2 * area / (a + b + c))
  (h_circumradius : ∀ (a b c : ℝ), R = (a * b * c) / (4 * area))
  (x : ℝ) (h_x : x = (b + c - a) / 2) (area : ℝ) :
  (a * x / (r * R)) ≥ 3 :=
sorry

end min_value_ax_over_rR_l213_213421


namespace roots_of_polynomial_l213_213952

theorem roots_of_polynomial :
  (3 * (2 + Real.sqrt 3)^4 - 19 * (2 + Real.sqrt 3)^3 + 34 * (2 + Real.sqrt 3)^2 - 19 * (2 + Real.sqrt 3) + 3 = 0) ∧ 
  (3 * (2 - Real.sqrt 3)^4 - 19 * (2 - Real.sqrt 3)^3 + 34 * (2 - Real.sqrt 3)^2 - 19 * (2 - Real.sqrt 3) + 3 = 0) ∧
  (3 * ((7 + Real.sqrt 13) / 6)^4 - 19 * ((7 + Real.sqrt 13) / 6)^3 + 34 * ((7 + Real.sqrt 13) / 6)^2 - 19 * ((7 + Real.sqrt 13) / 6) + 3 = 0) ∧
  (3 * ((7 - Real.sqrt 13) / 6)^4 - 19 * ((7 - Real.sqrt 13) / 6)^3 + 34 * ((7 - Real.sqrt 13) / 6)^2 - 19 * ((7 - Real.sqrt 13) / 6) + 3 = 0) :=
by sorry

end roots_of_polynomial_l213_213952


namespace find_cosine_of_angle_subtraction_l213_213491

variable (α : ℝ)
variable (h : Real.sin ((Real.pi / 6) - α) = 1 / 3)

theorem find_cosine_of_angle_subtraction :
  Real.cos ((2 * Real.pi / 3) - α) = -1 / 3 :=
by
  exact sorry

end find_cosine_of_angle_subtraction_l213_213491


namespace john_experience_when_mike_started_l213_213746

-- Definitions from the conditions
variable (J O M : ℕ)
variable (h1 : J = 20) -- James currently has 20 years of experience
variable (h2 : O - 8 = 2 * (J - 8)) -- 8 years ago, John had twice as much experience as James
variable (h3 : J + O + M = 68) -- Combined experience is 68 years

-- Theorem to prove
theorem john_experience_when_mike_started : O - M = 16 := 
by
  -- Proof steps go here
  sorry

end john_experience_when_mike_started_l213_213746


namespace proof_of_a_b_and_T_l213_213893

-- Define sequences and the given conditions

def a (n : ℕ) : ℕ := 2^(n-1)

def b (n : ℕ) : ℕ := 2 * n

def S (n : ℕ) : ℕ := 2^n - 1

def c (n : ℕ) : ℚ := 1 / ((b n)^2 - 1)

def T (n : ℕ) : ℚ := (n : ℚ) / (2 * n + 1)

axiom b_condition : ∀ n : ℕ, n > 0 → (b n + 2 * n = 2 * (b (n-1)) + 4)

axiom S_condition : ∀ n : ℕ, S n = 2^n - 1

theorem proof_of_a_b_and_T (n : ℕ) (h : n > 0) : 
  (∀ k, a k = 2^(k-1)) ∧ 
  (∀ k, b k = 2 * k) ∧ 
  (∀ k, T k = (k : ℚ) / (2 * k + 1)) := by
  sorry

end proof_of_a_b_and_T_l213_213893


namespace letters_with_both_l213_213721

/-
In a certain alphabet, some letters contain a dot and a straight line. 
36 letters contain a straight line but do not contain a dot. 
The alphabet has 60 letters, all of which contain either a dot or a straight line or both. 
There are 4 letters that contain a dot but do not contain a straight line. 
-/
def L_no_D : ℕ := 36
def D_no_L : ℕ := 4
def total_letters : ℕ := 60

theorem letters_with_both (DL : ℕ) : 
  total_letters = D_no_L + L_no_D + DL → 
  DL = 20 :=
by
  intros h
  sorry

end letters_with_both_l213_213721


namespace find_compound_interest_principal_l213_213453

noncomputable def SI (P R T: ℝ) := (P * R * T) / 100
noncomputable def CI (P R T: ℝ) := P * (1 + R / 100)^T - P

theorem find_compound_interest_principal :
  let SI_amount := 3500.000000000004
  let SI_years := 2
  let SI_rate := 6
  let CI_years := 2
  let CI_rate := 10
  let SI_value := SI SI_amount SI_rate SI_years
  let P := 4000
  (SI_value = (CI P CI_rate CI_years) / 2) →
  P = 4000 :=
by
  intros
  sorry

end find_compound_interest_principal_l213_213453


namespace nested_g_of_2_l213_213709

def g (x : ℤ) : ℤ := x^2 - 4*x + 3

theorem nested_g_of_2 : g (g (g (g (g (g 2))))) = 1394486148248 := by
  sorry

end nested_g_of_2_l213_213709


namespace constant_fraction_condition_l213_213613

theorem constant_fraction_condition 
    (a1 b1 c1 a2 b2 c2 : ℝ) : 
    (∀ x : ℝ, (a1 * x^2 + b1 * x + c1) / (a2 * x^2 + b2 * x + c2) = k) ↔ 
    (a1 / a2 = b1 / b2 ∧ b1 / b2 = c1 / c2) :=
by
  sorry

end constant_fraction_condition_l213_213613


namespace weight_of_b_l213_213965

theorem weight_of_b (a b c : ℝ) (h1 : a + b + c = 135) (h2 : a + b = 80) (h3 : b + c = 82) : b = 27 :=
by
  sorry

end weight_of_b_l213_213965


namespace num_points_P_on_ellipse_l213_213437

noncomputable def ellipse : Set (ℝ × ℝ) := {p | (p.1)^2 / 16 + (p.2)^2 / 9 = 1}
noncomputable def line : Set (ℝ × ℝ) := {p | p.1 / 4 + p.2 / 3 = 1}
noncomputable def area_triangle (p1 p2 p3 : ℝ × ℝ) : ℝ := 
  (1 / 2) * abs (p1.1 * (p2.2 - p3.2) + p2.1 * (p3.2 - p1.2) + p3.1 * (p1.2 - p2.2))

theorem num_points_P_on_ellipse (A B : ℝ × ℝ) 
  (hA_on_line : A ∈ line) (hA_on_ellipse : A ∈ ellipse) 
  (hB_on_line : B ∈ line) (hB_on_ellipse : B ∈ ellipse)
  : ∃ P1 P2 : ℝ × ℝ, P1 ∈ ellipse ∧ P2 ∈ ellipse ∧ 
    area_triangle A B P1 = 3 ∧ area_triangle A B P2 = 3 ∧ 
    P1 ≠ P2 ∧ 
    (∀ P : ℝ × ℝ, P ∈ ellipse ∧ area_triangle A B P = 3 → P = P1 ∨ P = P2) := 
sorry

end num_points_P_on_ellipse_l213_213437


namespace quadratic_rewrite_l213_213159

theorem quadratic_rewrite :
  ∃ a d : ℤ, (∀ x : ℝ, x^2 + 500 * x + 2500 = (x + a)^2 + d) ∧ (d / a) = -240 := by
  sorry

end quadratic_rewrite_l213_213159


namespace sandy_gain_percent_l213_213298

def gain_percent (purchase_price repair_cost selling_price : ℕ) : ℕ :=
  let total_cost := purchase_price + repair_cost
  let gain := selling_price - total_cost
  (gain * 100) / total_cost

theorem sandy_gain_percent :
  gain_percent 900 300 1260 = 5 :=
by
  sorry

end sandy_gain_percent_l213_213298


namespace artifacts_per_wing_l213_213154

theorem artifacts_per_wing (P A w_wings p_wings a_wings : ℕ) (hp1 : w_wings = 8)
  (hp2 : A = 4 * P) (hp3 : p_wings = 3) (hp4 : (∃ L S : ℕ, L = 1 ∧ S = 12 ∧ P = 2 * S + L))
  (hp5 : a_wings = w_wings - p_wings) :
  A / a_wings = 20 :=
by
  sorry

end artifacts_per_wing_l213_213154


namespace new_students_admitted_l213_213187

theorem new_students_admitted (orig_students : ℕ := 35) (increase_cost : ℕ := 42) (orig_expense : ℕ := 400) (dim_avg_expense : ℤ := 1) :
  ∃ (x : ℕ), x = 7 :=
by
  sorry

end new_students_admitted_l213_213187


namespace exists_line_intersecting_all_segments_l213_213334

theorem exists_line_intersecting_all_segments 
  (segments : List (ℝ × ℝ)) 
  (h1 : ∀ (P Q R : (ℝ × ℝ)), P ∈ segments → Q ∈ segments → R ∈ segments → ∃ (L : ℝ × ℝ → Prop), L P ∧ L Q ∧ L R) :
  ∃ (L : ℝ × ℝ → Prop), ∀ (S : (ℝ × ℝ)), S ∈ segments → L S :=
by
  sorry

end exists_line_intersecting_all_segments_l213_213334


namespace incorrect_conclusion_l213_213360

theorem incorrect_conclusion
  (a b : ℝ) 
  (h₁ : 1/a < 1/b) 
  (h₂ : 1/b < 0) 
  (h₃ : a < 0) 
  (h₄ : b < 0) 
  (h₅ : a > b) : ¬ (|a| + |b| > |a + b|) := 
sorry

end incorrect_conclusion_l213_213360


namespace sum_of_integers_ways_l213_213499

theorem sum_of_integers_ways (n : ℕ) (h : n > 0) : 
  ∃ ways : ℕ, ways = 2^(n-1) := sorry

end sum_of_integers_ways_l213_213499


namespace pyramid_layers_total_l213_213031

-- Since we are dealing with natural number calculations, noncomputable is generally not needed.

-- Definition of the pyramid layers and the number of balls in each layer
def number_of_balls (n : ℕ) : ℕ := n ^ 2

-- Given conditions for the layers
def third_layer_balls : ℕ := number_of_balls 3
def fifth_layer_balls : ℕ := number_of_balls 5

-- Statement of the problem proving that their sum is 34
theorem pyramid_layers_total : third_layer_balls + fifth_layer_balls = 34 := by
  sorry -- proof to be provided

end pyramid_layers_total_l213_213031


namespace polygon_sides_l213_213282

theorem polygon_sides (n : ℕ) 
  (h1 : sum_interior_angles = 180 * (n - 2))
  (h2 : sum_exterior_angles = 360)
  (h3 : sum_interior_angles = 3 * sum_exterior_angles) : 
  n = 8 :=
by
  sorry

end polygon_sides_l213_213282


namespace ratio_of_elements_l213_213434

theorem ratio_of_elements (total_weight : ℕ) (element_B_weight : ℕ) 
  (h_total : total_weight = 324) (h_B : element_B_weight = 270) :
  (total_weight - element_B_weight) / element_B_weight = 1 / 5 :=
by
  sorry

end ratio_of_elements_l213_213434


namespace sum_fractions_bounds_l213_213527

theorem sum_fractions_bounds {a b c : ℝ} (h : a * b * c = 1) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  1 < (a / (a + 1)) + (b / (b + 1)) + (c / (c + 1)) ∧ 
  (a / (a + 1)) + (b / (b + 1)) + (c / (c + 1)) < 2 :=
  sorry

end sum_fractions_bounds_l213_213527


namespace inequality_solution_set_empty_range_l213_213978

theorem inequality_solution_set_empty_range (m : ℝ) :
  (∀ x : ℝ, mx^2 - mx - 1 < 0) ↔ -4 < m ∧ m ≤ 0 :=
by
  sorry

end inequality_solution_set_empty_range_l213_213978


namespace ratio_problem_l213_213950

theorem ratio_problem
  (a b c d e : ℚ)
  (h1 : a / b = 5)
  (h2 : b / c = 1 / 4)
  (h3 : c / d = 7)
  (h4 : d / e = 2) :
  e / a = 2 / 35 := 
sorry

end ratio_problem_l213_213950


namespace minimum_area_integer_triangle_l213_213565

theorem minimum_area_integer_triangle :
  ∃ (p q : ℤ), p ≠ 0 ∧ q ≠ 0 ∧ (∃ (p q : ℤ), 2 ∣ (16 * p - 30 * q)) 
  → (∃ (area : ℝ), area = (1/2 : ℝ) * |16 * p - 30 * q| ∧ area = 1) :=
by
  sorry

end minimum_area_integer_triangle_l213_213565


namespace incorrect_conclusions_l213_213572

theorem incorrect_conclusions :
  let p := (∀ x y : ℝ, x * y ≠ 6 → x ≠ 2 ∨ y ≠ 3)
  let q := (2, 1) ∈ { p : ℝ × ℝ | p.2 = 2 * p.1 - 3 }
  (p ∨ ¬q) = false ∧ (¬p ∨ q) = false ∧ (p ∧ ¬q) = false :=
by
  sorry

end incorrect_conclusions_l213_213572


namespace system_of_equations_l213_213451

theorem system_of_equations (x y k : ℝ) 
  (h1 : x + 2 * y = k + 2) 
  (h2 : 2 * x - 3 * y = 3 * k - 1) : 
  x + 9 * y = 7 :=
  sorry

end system_of_equations_l213_213451


namespace intersection_M_N_l213_213297

def M := { y : ℝ | ∃ x : ℝ, y = 2^x }
def N := { y : ℝ | ∃ x : ℝ, y = 2 * Real.sin x }

theorem intersection_M_N : M ∩ N = { y : ℝ | 0 < y ∧ y ≤ 2 } :=
by
  sorry

end intersection_M_N_l213_213297


namespace perimeter_less_than_1_km_perimeter_less_than_1_km_zero_thickness_perimeter_to_area_ratio_l213_213196

section Problem

-- Definitions based on the problem conditions

-- Condition: Side length of each square is 1 cm
def side_length : ℝ := 1

-- Condition: Thickness of the nail for parts a) and b)
def nail_thickness_a := 0.1
def nail_thickness_b := 0

-- Given a perimeter P and area S, the perimeter cannot exceed certain thresholds based on problem analysis

theorem perimeter_less_than_1_km (P : ℝ) (S : ℝ) (r : ℝ) (h1 : 2 * S ≥ r * P) (h2 : r = 0.1) : P < 1000 * 100 :=
  sorry

theorem perimeter_less_than_1_km_zero_thickness (P : ℝ) (S : ℝ) (r : ℝ) (h1 : 2 * S ≥ r * P) (h2 : r = 0) : P < 1000 * 100 :=
  sorry

theorem perimeter_to_area_ratio (P : ℝ) (S : ℝ) (h : P / S ≤ 700) : P / S < 100000 :=
  sorry

end Problem

end perimeter_less_than_1_km_perimeter_less_than_1_km_zero_thickness_perimeter_to_area_ratio_l213_213196


namespace neg_pi_lt_neg_314_l213_213129

theorem neg_pi_lt_neg_314 (h : Real.pi > 3.14) : -Real.pi < -3.14 :=
sorry

end neg_pi_lt_neg_314_l213_213129


namespace range_of_m_l213_213307

theorem range_of_m (m : ℝ) : (∀ x : ℝ, x > 0 → (y = 1 - 3 * m / x) → y > 0) ↔ (m > 1 / 3) :=
sorry

end range_of_m_l213_213307


namespace max_ahn_achieve_max_ahn_achieve_attained_l213_213680

def is_two_digit_integer (n : ℕ) : Prop :=
  10 ≤ n ∧ n ≤ 99

theorem max_ahn_achieve :
  ∀ (n : ℕ), is_two_digit_integer n → 3 * (300 - n) ≤ 870 := 
by sorry

theorem max_ahn_achieve_attained :
  3 * (300 - 10) = 870 := 
by norm_num

end max_ahn_achieve_max_ahn_achieve_attained_l213_213680


namespace successive_discounts_eq_single_discount_l213_213157

theorem successive_discounts_eq_single_discount :
  ∀ (x : ℝ), (1 - 0.15) * (1 - 0.25) * x = (1 - 0.3625) * x :=
by
  intro x
  sorry

end successive_discounts_eq_single_discount_l213_213157


namespace product_n_equals_7200_l213_213005

theorem product_n_equals_7200 :
  (3 - 1) * 3 * (3 + 1) * (3 + 2) * (3 + 3) * (3 ^ 2 + 1) = 7200 := by
  sorry

end product_n_equals_7200_l213_213005


namespace deductive_reasoning_example_is_A_l213_213019

def isDeductive (statement : String) : Prop := sorry

-- Define conditions
def optionA : String := "Since y = 2^x is an exponential function, the function y = 2^x passes through the fixed point (0,1)"
def optionB : String := "Guessing the general formula for the sequence 1/(1×2), 1/(2×3), 1/(3×4), ... as a_n = 1/(n(n+1)) (n ∈ ℕ⁺)"
def optionC : String := "Drawing an analogy from 'In a plane, two lines perpendicular to the same line are parallel' to infer 'In space, two planes perpendicular to the same plane are parallel'"
def optionD : String := "From the circle's equation in the Cartesian coordinate plane (x-a)² + (y-b)² = r², predict that the equation of a sphere in three-dimensional Cartesian coordinates is (x-a)² + (y-b)² + (z-c)² = r²"

theorem deductive_reasoning_example_is_A : isDeductive optionA :=
by
  sorry

end deductive_reasoning_example_is_A_l213_213019


namespace rectangle_area_is_243_square_meters_l213_213138

-- Given conditions
variables (w l : ℝ)
-- The width is one-third of its length
def width_one_third_length : Prop := w = l / 3

-- The perimeter of the rectangle is 72 meters
def perimeter_72 : Prop := 2 * (w + l) = 72

-- The area of the rectangle
def area : ℝ := w * l

-- Proof statement
theorem rectangle_area_is_243_square_meters 
  (h1 : width_one_third_length w l) 
  (h2 : perimeter_72 w l) : area w l = 243 := 
by 
sorry

end rectangle_area_is_243_square_meters_l213_213138


namespace fraction_inhabitable_l213_213517

-- Define the constants based on the given conditions
def fraction_water : ℚ := 3 / 5
def fraction_inhabitable_land : ℚ := 3 / 4

-- Define the theorem to prove that the fraction of Earth's surface that is inhabitable is 3/10
theorem fraction_inhabitable (w h : ℚ) (hw : w = fraction_water) (hh : h = fraction_inhabitable_land) : 
  (1 - w) * h = 3 / 10 :=
by
  sorry

end fraction_inhabitable_l213_213517


namespace not_true_n_gt_24_l213_213445

theorem not_true_n_gt_24 (n : ℕ) (h : 1/3 + 1/4 + 1/6 + 1/n = 1) : n ≤ 24 := 
by
  -- Placeholder for the proof
  sorry

end not_true_n_gt_24_l213_213445


namespace greatest_third_side_l213_213595

-- Given data and the Triangle Inequality theorem
theorem greatest_third_side (c : ℕ) (h1 : 8 < c) (h2 : c < 22) : c = 21 :=
by
  sorry

end greatest_third_side_l213_213595


namespace find_original_number_l213_213676

variable (x : ℝ)

def tripled := 3 * x
def doubled := 2 * tripled
def subtracted := doubled - 9
def trebled := 3 * subtracted

theorem find_original_number (h : trebled = 90) : x = 6.5 := by
  sorry

end find_original_number_l213_213676


namespace original_triangle_area_l213_213167

-- Define the conditions
def dimensions_quadrupled (original_area new_area : ℝ) : Prop :=
  4^2 * original_area = new_area

-- Define the statement to be proved
theorem original_triangle_area {new_area : ℝ} (h : new_area = 64) :
  ∃ (original_area : ℝ), dimensions_quadrupled original_area new_area ∧ original_area = 4 :=
by
  sorry

end original_triangle_area_l213_213167


namespace number_of_triangles_l213_213234

theorem number_of_triangles (n : ℕ) (h : n = 10) : ∃ k, k = 120 ∧ n.choose 3 = k :=
by
  sorry

end number_of_triangles_l213_213234


namespace gcd_lcm_product_l213_213139

theorem gcd_lcm_product (a b : ℕ) (h : Nat.gcd a b * Nat.lcm a b = 360) :
  ∃ (s : Finset ℕ), s.card = 12 ∧ ∀ d ∈ s, d = Nat.gcd a b :=
sorry

end gcd_lcm_product_l213_213139


namespace total_nails_needed_l213_213090

-- Define the conditions
def nails_already_have : ℕ := 247
def nails_found : ℕ := 144
def nails_to_buy : ℕ := 109

-- The statement to prove
theorem total_nails_needed : nails_already_have + nails_found + nails_to_buy = 500 := by
  -- The proof goes here
  sorry

end total_nails_needed_l213_213090


namespace total_percentage_increase_l213_213256

noncomputable def initialSalary : ℝ := 60
noncomputable def firstRaisePercent : ℝ := 10
noncomputable def secondRaisePercent : ℝ := 15
noncomputable def promotionRaisePercent : ℝ := 20

theorem total_percentage_increase :
  let finalSalary := initialSalary * (1 + firstRaisePercent / 100) * (1 + secondRaisePercent / 100) * (1 + promotionRaisePercent / 100)
  let increase := finalSalary - initialSalary
  let percentageIncrease := (increase / initialSalary) * 100
  percentageIncrease = 51.8 := by
  sorry

end total_percentage_increase_l213_213256


namespace find_carl_age_l213_213336

variables (Alice Bob Carl : ℝ)

-- Conditions
def average_age : Prop := (Alice + Bob + Carl) / 3 = 15
def carl_twice_alice : Prop := Carl - 5 = 2 * Alice
def bob_fraction_alice : Prop := Bob + 4 = (3 / 4) * (Alice + 4)

-- Conjecture
theorem find_carl_age : average_age Alice Bob Carl ∧ carl_twice_alice Alice Carl ∧ bob_fraction_alice Alice Bob → Carl = 34.818 :=
by
  sorry

end find_carl_age_l213_213336


namespace amy_final_money_l213_213732

theorem amy_final_money :
  let initial_money := 2
  let chore_payment := 5 * 13
  let birthday_gift := 3
  let toy_cost := 12
  let remaining_money := initial_money + chore_payment + birthday_gift - toy_cost
  let grandparents_reward := 2 * remaining_money
  remaining_money + grandparents_reward = 174 := 
by
  sorry

end amy_final_money_l213_213732


namespace find_b_plus_m_l213_213463

section MatrixPower

open Matrix

-- Define our matrices
def A (b m : ℕ) : Matrix (Fin 3) (Fin 3) ℕ :=
  ![![1, 3, b], 
    ![0, 1, 5], 
    ![0, 0, 1]]

def B : Matrix (Fin 3) (Fin 3) ℕ :=
  ![![1, 27, 3008], 
    ![0, 1, 45], 
    ![0, 0, 1]]

-- The problem statement
noncomputable def power_eq_matrix (b m : ℕ) : Prop :=
  (A b m) ^ m = B

-- The final goal
theorem find_b_plus_m (b m : ℕ) (h : power_eq_matrix b m) : b + m = 283 := sorry

end MatrixPower

end find_b_plus_m_l213_213463


namespace cos_C_in_triangle_l213_213156

theorem cos_C_in_triangle
  (A B C : ℝ)
  (sin_A : Real.sin A = 4 / 5)
  (cos_B : Real.cos B = 3 / 5) :
  Real.cos C = 7 / 25 :=
sorry

end cos_C_in_triangle_l213_213156


namespace plane_passing_through_A_perpendicular_to_BC_l213_213611

-- Define the points A, B, and C
structure Point3D :=
(x : ℝ)
(y : ℝ)
(z : ℝ)

def A : Point3D := { x := -3, y := 7, z := 2 }
def B : Point3D := { x := 3, y := 5, z := 1 }
def C : Point3D := { x := 4, y := 5, z := 3 }

-- Define the vector BC as the difference between points C and B
def vectorBC (B C : Point3D) : Point3D :=
{ x := C.x - B.x,
  y := C.y - B.y,
  z := C.z - B.z }

-- Define the equation of the plane passing through point A and 
-- perpendicular to vector BC
def plane_eq (A : Point3D) (n : Point3D) (x y z : ℝ) : Prop :=
n.x * (x - A.x) + n.y * (y - A.y) + n.z * (z - A.z) = 0

-- Define the proof problem
theorem plane_passing_through_A_perpendicular_to_BC :
  ∀ (x y z : ℝ), plane_eq A (vectorBC B C) x y z ↔ x + 2 * z - 1 = 0 :=
by
  -- the proof part
  sorry

end plane_passing_through_A_perpendicular_to_BC_l213_213611


namespace incorrect_option_C_l213_213230

-- Definitions of increasing and decreasing functions
def increasing (f : ℝ → ℝ) := ∀ x₁ x₂, x₁ ≤ x₂ → f x₁ ≤ f x₂
def decreasing (f : ℝ → ℝ) := ∀ x₁ x₂, x₁ ≤ x₂ → f x₁ ≥ f x₂

-- The incorrectness of option C
theorem incorrect_option_C (f g : ℝ → ℝ) 
  (h₁ : increasing f) 
  (h₂ : decreasing g) : ¬ increasing (fun x => f x + g x) := 
sorry

end incorrect_option_C_l213_213230


namespace angle_between_plane_and_base_l213_213821

variable (α k : ℝ)
variable (hα : ∀ S A B C : ℝ, S ≠ A ∧ S ≠ B ∧ S ≠ C ∧ A ≠ B ∧ A ≠ C ∧ B ≠ C)
variable (h_ratio : ∀ A D S : ℝ, AD / DS = k)

theorem angle_between_plane_and_base (α k : ℝ) 
  (hα : ∀ S A B C : ℝ, S ≠ A ∧ S ≠ B ∧ S ≠ C ∧ A ≠ B ∧ A ≠ C ∧ B ≠ C)
  (h_ratio : ∀ A D S : ℝ, AD / DS = k) 
  : ∃ γ : ℝ, γ = Real.arctan (k / (k + 3) * Real.tan α) :=
by
  sorry

end angle_between_plane_and_base_l213_213821


namespace avg_annual_growth_rate_equation_l213_213211

variable (x : ℝ)
def foreign_trade_income_2007 : ℝ := 250 -- million yuan
def foreign_trade_income_2009 : ℝ := 360 -- million yuan

theorem avg_annual_growth_rate_equation :
  2.5 * (1 + x) ^ 2 = 3.6 := sorry

end avg_annual_growth_rate_equation_l213_213211


namespace sum_of_squares_of_roots_l213_213213

-- Define the roots of the polynomial and Vieta's conditions
variables {p q r : ℝ}

-- Given conditions from Vieta's formulas
def vieta_conditions (p q r : ℝ) : Prop :=
  p + q + r = 7 / 3 ∧
  p * q + p * r + q * r = 2 / 3 ∧
  p * q * r = 4 / 3

-- Statement that sum of squares of roots equals to 37/9 given Vieta's conditions
theorem sum_of_squares_of_roots 
  (h : vieta_conditions p q r) : 
  p^2 + q^2 + r^2 = 37 / 9 := 
sorry

end sum_of_squares_of_roots_l213_213213


namespace greatest_divisor_arithmetic_sequence_sum_l213_213684

theorem greatest_divisor_arithmetic_sequence_sum (x c : ℕ) (hx : 0 < x) (hc : 0 < c) : 
  ∃ k, (15 * (x + 7 * c)) = 15 * k :=
sorry

end greatest_divisor_arithmetic_sequence_sum_l213_213684


namespace intersection_A_B_l213_213017

def set_A (x : ℝ) : Prop := 2 * x^2 + 5 * x - 3 ≤ 0

def set_B (x : ℝ) : Prop := -2 < x

theorem intersection_A_B :
  {x : ℝ | set_A x} ∩ {x : ℝ | set_B x} = {x : ℝ | -2 < x ∧ x ≤ 1/2} := 
by {
  sorry
}

end intersection_A_B_l213_213017


namespace calculate_expression_l213_213083

theorem calculate_expression : 
  (1 - Real.sqrt 2)^0 + |(2 - Real.sqrt 5)| + (-1)^2022 - (1/3) * Real.sqrt 45 = 0 :=
by
  sorry

end calculate_expression_l213_213083


namespace yujin_wire_length_is_correct_l213_213918

def junhoe_wire_length : ℝ := 134.5
def multiplicative_factor : ℝ := 1.06
def yujin_wire_length (junhoe_length : ℝ) (factor : ℝ) : ℝ := junhoe_length * factor

theorem yujin_wire_length_is_correct : 
  yujin_wire_length junhoe_wire_length multiplicative_factor = 142.57 := 
by 
  sorry

end yujin_wire_length_is_correct_l213_213918


namespace find_X_l213_213873

def r (X Y : ℕ) : ℕ := X^2 + Y^2

theorem find_X (X : ℕ) (h : r X 7 = 338) : X = 17 := by
  sorry

end find_X_l213_213873


namespace number_leaves_remainder_3_l213_213514

theorem number_leaves_remainder_3 (n : ℕ) (h1 : 1680 % 9 = 0) (h2 : 1680 = n * 9) : 1680 % 1677 = 3 := by
  sorry

end number_leaves_remainder_3_l213_213514


namespace first_train_speed_l213_213727

noncomputable def speed_first_train (length1 length2 : ℝ) (speed2 time : ℝ) : ℝ :=
  let distance := (length1 + length2) / 1000
  let time_hours := time / 3600
  (distance / time_hours) - speed2

theorem first_train_speed :
  speed_first_train 100 280 30 18.998480121590273 = 42 :=
by
  sorry

end first_train_speed_l213_213727


namespace num_divisible_by_10_l213_213938

theorem num_divisible_by_10 (a b d : ℕ) (h1 : 100 ≤ a) (h2 : a ≤ 500) (h3 : 100 ≤ b) (h4 : b ≤ 500) (h5 : Nat.gcd d 10 = 10) :
  (b - a) / d + 1 = 41 := by
  sorry

end num_divisible_by_10_l213_213938


namespace rectangle_same_color_exists_l213_213377

def color := ℕ -- We use ℕ as a stand-in for three colors {0, 1, 2}

def same_color_rectangle_exists (coloring : (Fin 4) → (Fin 82) → color) : Prop :=
  ∃ (i j : Fin 4) (k l : Fin 82), i ≠ j ∧ k ≠ l ∧
    coloring i k = coloring i l ∧
    coloring j k = coloring j l ∧
    coloring i k = coloring j k

theorem rectangle_same_color_exists :
  ∀ (coloring : (Fin 4) → (Fin 82) → color),
  same_color_rectangle_exists coloring :=
by
  sorry

end rectangle_same_color_exists_l213_213377


namespace camp_cedar_counselors_l213_213464

theorem camp_cedar_counselors (boys : ℕ) (girls : ℕ) (total_children : ℕ) (counselors : ℕ)
  (h1 : boys = 40)
  (h2 : girls = 3 * boys)
  (h3 : total_children = boys + girls)
  (h4 : counselors = total_children / 8) : 
  counselors = 20 :=
by sorry

end camp_cedar_counselors_l213_213464


namespace product_equation_l213_213679

/-- Given two numbers x and y such that x + y = 20 and x - y = 4,
    the product of three times the larger number and the smaller number is 288. -/
theorem product_equation (x y : ℕ) (h1 : x + y = 20) (h2 : x - y = 4) (h3 : x > y) : 3 * x * y = 288 := 
sorry

end product_equation_l213_213679


namespace number_of_ordered_triplets_l213_213639

theorem number_of_ordered_triplets :
  ∃ count : ℕ, (∀ (a b c : ℕ), lcm a b = 1000 ∧ lcm b c = 2000 ∧ lcm c a = 2000 →
  count = 70) :=
sorry

end number_of_ordered_triplets_l213_213639


namespace investment_Y_l213_213562

theorem investment_Y
  (X_investment : ℝ)
  (Y_investment : ℝ)
  (Z_investment : ℝ)
  (X_months : ℝ)
  (Y_months : ℝ)
  (Z_months : ℝ)
  (total_profit : ℝ)
  (Z_profit_share : ℝ)
  (h1 : X_investment = 36000)
  (h2 : Z_investment = 48000)
  (h3 : X_months = 12)
  (h4 : Y_months = 12)
  (h5 : Z_months = 8)
  (h6 : total_profit = 13970)
  (h7 : Z_profit_share = 4064) :
  Y_investment = 75000 := by
  -- Proof omitted
  sorry

end investment_Y_l213_213562


namespace rewrite_equation_l213_213457

theorem rewrite_equation (x y : ℝ) (h : 2 * x - y = 4) : y = 2 * x - 4 :=
by
  sorry

end rewrite_equation_l213_213457


namespace smallest_integer_inequality_l213_213702

theorem smallest_integer_inequality :
  (∃ n : ℤ, ∀ x y z : ℝ, (x + y + z)^2 ≤ (n:ℝ) * (x^2 + y^2 + z^2)) ∧
  ∀ m : ℤ, (∀ x y z : ℝ, (x + y + z)^2 ≤ (m:ℝ) * (x^2 + y^2 + z^2)) → 3 ≤ m :=
  sorry

end smallest_integer_inequality_l213_213702


namespace calculate_3_diamond_4_l213_213291

-- Define the operations
def op (a b : ℝ) : ℝ := a^2 + 2 * a * b
def diamond (a b : ℝ) : ℝ := 4 * a + 6 * b - op a b

-- State the theorem
theorem calculate_3_diamond_4 : diamond 3 4 = 3 := by
  sorry

end calculate_3_diamond_4_l213_213291


namespace time_to_sweep_one_room_l213_213796

theorem time_to_sweep_one_room (x : ℕ) :
  (10 * x) = (2 * 9 + 6 * 2) → x = 3 := by
  sorry

end time_to_sweep_one_room_l213_213796


namespace complex_fraction_equivalence_l213_213082

/-- The complex number 2 / (1 - i) is equal to 1 + i. -/
theorem complex_fraction_equivalence : (2 : ℂ) / (1 - (I : ℂ)) = 1 + (I : ℂ) := by
  sorry

end complex_fraction_equivalence_l213_213082


namespace not_directly_nor_inversely_proportional_l213_213120

theorem not_directly_nor_inversely_proportional :
  ∀ (x y : ℝ),
    ((2 * x + y = 5) ∨ (2 * x + 3 * y = 12)) ∧
    ((¬ (∃ k : ℝ, x = k * y)) ∧ (¬ (∃ k : ℝ, x * y = k))) := sorry

end not_directly_nor_inversely_proportional_l213_213120


namespace postage_problem_l213_213148

theorem postage_problem (n : ℕ) (h_positive : n > 0) (h_postage : ∀ k, k ∈ List.range 121 → ∃ a b c : ℕ, 6 * a + n * b + (n + 2) * c = k) :
  6 * n * (n + 2) - (6 + n + (n + 2)) = 120 → n = 8 := 
by
  sorry

end postage_problem_l213_213148


namespace complement_A_in_U_l213_213218

universe u

-- Define the universal set U and set A.
def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7}
def A : Set ℕ := {1, 3, 5, 7}

-- Define the complement of A in U.
def complement (A U: Set ℕ) : Set ℕ :=
  {x ∈ U | x ∉ A}

-- Statement to prove.
theorem complement_A_in_U :
  complement A U = {2, 4, 6} :=
sorry

end complement_A_in_U_l213_213218


namespace derivative_at_0_l213_213493

def f (x : ℝ) : ℝ := x + x^2

theorem derivative_at_0 : deriv f 0 = 1 := by
  -- Proof goes here
  sorry

end derivative_at_0_l213_213493


namespace problem_l213_213921

def A : Set ℝ := {x | x^2 + x - 6 > 0}
def B : Set ℝ := {x | 0 < x ∧ x < 6}
def C := (Aᶜ) ∩ B

theorem problem : C = {x | 0 < x ∧ x ≤ 2} :=
by
  sorry

end problem_l213_213921


namespace lamps_purchased_min_type_B_lamps_l213_213390

variables (x y m : ℕ)

def total_lamps := x + y = 50
def total_cost := 40 * x + 65 * y = 2500
def profit_type_A := 60 - 40 = 20
def profit_type_B := 100 - 65 = 35
def profit_requirement := 20 * (50 - m) + 35 * m ≥ 1400

theorem lamps_purchased (h₁ : total_lamps x y) (h₂ : total_cost x y) : 
  x = 30 ∧ y = 20 :=
  sorry

theorem min_type_B_lamps (h₃ : profit_type_A) (h₄ : profit_type_B) (h₅ : profit_requirement m) : 
  m ≥ 27 :=
  sorry

end lamps_purchased_min_type_B_lamps_l213_213390


namespace students_count_l213_213916

theorem students_count :
  ∀ (sets marbles_per_set marbles_per_student total_students : ℕ),
    sets = 3 →
    marbles_per_set = 32 →
    marbles_per_student = 4 →
    total_students = (sets * marbles_per_set) / marbles_per_student →
    total_students = 24 :=
by
  intros sets marbles_per_set marbles_per_student total_students
  intros h_sets h_marbles_per_set h_marbles_per_student h_total_students
  rw [h_sets, h_marbles_per_set, h_marbles_per_student] at h_total_students
  exact h_total_students

end students_count_l213_213916


namespace find_central_angle_l213_213052

noncomputable def sector := 
  {R : ℝ // R > 0}

noncomputable def central_angle (R : ℝ) : ℝ := 
  (6 - 2 * R) / R

theorem find_central_angle :
  ∃ α : ℝ, (α = 1 ∨ α = 4) ∧ 
  (∃ R : ℝ, 
    (2 * R + α * R = 6) ∧ 
    (1 / 2 * R^2 * α = 2)) := 
by {
  sorry
}

end find_central_angle_l213_213052


namespace average_speed_is_55_l213_213685

theorem average_speed_is_55 
  (initial_reading : ℕ) (final_reading : ℕ) (time_hours : ℕ)
  (H1 : initial_reading = 15951) 
  (H2 : final_reading = 16061)
  (H3 : time_hours = 2) : 
  (final_reading - initial_reading) / time_hours = 55 :=
by
  sorry

end average_speed_is_55_l213_213685


namespace age_of_new_person_l213_213358

theorem age_of_new_person 
    (n : ℕ) 
    (T : ℕ := n * 14) 
    (n_eq : n = 9) 
    (new_average : (T + A) / (n + 1) = 16) 
    (A : ℕ) : A = 34 :=
by
  sorry

end age_of_new_person_l213_213358


namespace democrats_ratio_l213_213780

theorem democrats_ratio (F M: ℕ) 
  (h_total_participants : F + M = 810)
  (h_female_democrats : 135 * 2 = F)
  (h_male_democrats : (1 / 4) * M = 135) : 
  (270 / 810 = 1 / 3) :=
by 
  sorry

end democrats_ratio_l213_213780


namespace helicopter_rental_cost_l213_213778

theorem helicopter_rental_cost
  (hours_per_day : ℕ)
  (total_days : ℕ)
  (total_cost : ℕ)
  (H1 : hours_per_day = 2)
  (H2 : total_days = 3)
  (H3 : total_cost = 450) :
  total_cost / (hours_per_day * total_days) = 75 :=
by
  sorry

end helicopter_rental_cost_l213_213778


namespace total_pages_read_l213_213783

-- Define the average pages read by Lucas for the first four days.
def day1_4_avg : ℕ := 42

-- Define the average pages read by Lucas for the next two days.
def day5_6_avg : ℕ := 50

-- Define the pages read on the last day.
def day7 : ℕ := 30

-- Define the total number of days for which measurement is provided.
def total_days : ℕ := 7

-- Prove that the total number of pages Lucas read is 298.
theorem total_pages_read : 
  4 * day1_4_avg + 2 * day5_6_avg + day7 = 298 := 
by 
  sorry

end total_pages_read_l213_213783


namespace exists_five_positive_integers_sum_20_product_420_l213_213736
-- Import the entirety of Mathlib to ensure all necessary definitions are available

-- Lean statement for the proof problem
theorem exists_five_positive_integers_sum_20_product_420 :
  ∃ (a b c d e : ℕ), a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧ a + b + c + d + e = 20 ∧ a * b * c * d * e = 420 :=
sorry

end exists_five_positive_integers_sum_20_product_420_l213_213736


namespace algebra_expression_l213_213127

theorem algebra_expression (a b : ℝ) (h : a - b = 3) : 1 + a - b = 4 :=
sorry

end algebra_expression_l213_213127


namespace mod_exp_result_l213_213379

theorem mod_exp_result :
  (2 ^ 46655) % 9 = 1 :=
by
  sorry

end mod_exp_result_l213_213379


namespace factorize_difference_of_squares_l213_213905

variable (x : ℝ)

theorem factorize_difference_of_squares :
  x^2 - 1 = (x + 1) * (x - 1) :=
sorry

end factorize_difference_of_squares_l213_213905


namespace martinez_family_combined_height_l213_213818

def chiquita_height := 5
def mr_martinez_height := chiquita_height + 2
def mrs_martinez_height := chiquita_height - 1
def son_height := chiquita_height + 3
def combined_height := chiquita_height + mr_martinez_height + mrs_martinez_height + son_height

theorem martinez_family_combined_height : combined_height = 24 :=
by
  sorry

end martinez_family_combined_height_l213_213818


namespace trig_identity_on_line_l213_213473

theorem trig_identity_on_line (α : ℝ) (h : Real.tan α = 2) :
  Real.sin α ^ 2 - Real.cos α ^ 2 + Real.sin α * Real.cos α = 1 :=
sorry

end trig_identity_on_line_l213_213473


namespace problem_statement_l213_213907

noncomputable def find_sum (x y : ℝ) : ℝ := x + y

theorem problem_statement (x y : ℝ)
  (hx : |x| + x + y = 12)
  (hy : x + |y| - y = 14) :
  find_sum x y = 22 / 5 :=
sorry

end problem_statement_l213_213907


namespace distance_between_towns_l213_213997

theorem distance_between_towns 
  (x : ℝ) 
  (h1 : x / 100 - x / 110 = 0.15) : 
  x = 165 := 
by 
  sorry

end distance_between_towns_l213_213997


namespace arithmetic_sequence_term_l213_213170

theorem arithmetic_sequence_term {a : ℕ → ℤ} 
  (h1 : a 4 = -4) 
  (h2 : a 8 = 4) : 
  a 12 = 12 := 
by 
  sorry

end arithmetic_sequence_term_l213_213170


namespace range_of_m_l213_213121

theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, (x-1)^2 < m^2 → |1 - (x-1)/3| < 2) → (abs m ≤ 3) :=
by
  sorry

end range_of_m_l213_213121


namespace opposite_of_six_is_negative_six_l213_213160

theorem opposite_of_six_is_negative_six : -6 = -6 :=
by
  sorry

end opposite_of_six_is_negative_six_l213_213160


namespace sufficient_conditions_for_positive_product_l213_213130

theorem sufficient_conditions_for_positive_product (a b : ℝ) :
  (a > 0 ∧ b > 0) ∨ (a < 0 ∧ b < 0) ∨ (a > 1 ∧ b > 1) → a * b > 0 :=
by sorry

end sufficient_conditions_for_positive_product_l213_213130


namespace sum_of_three_consecutive_even_numbers_l213_213178

theorem sum_of_three_consecutive_even_numbers (m : ℤ) (h : ∃ k, m = 2 * k) : 
  m + (m + 2) + (m + 4) = 3 * m + 6 :=
by
  sorry

end sum_of_three_consecutive_even_numbers_l213_213178


namespace range_of_a_l213_213940

noncomputable def f (x : ℝ) : ℝ := 6 / x - x

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, 1 < x → x^2 + a * x - 6 > 0) ↔ 5 ≤ a :=
by
  sorry

end range_of_a_l213_213940


namespace find_k_l213_213277

variables (a b : ℝ × ℝ)
variables (k : ℝ)

def vector_a : ℝ × ℝ := (1, 2)
def vector_b : ℝ × ℝ := (2, -1)

def k_a_plus_b (k : ℝ) : ℝ × ℝ := (k * vector_a.1 + vector_b.1, k * vector_a.2 + vector_b.2)
def a_minus_2b : ℝ × ℝ := (vector_a.1 - 2 * vector_b.1, vector_a.2 - 2 * vector_b.2)

def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

theorem find_k (k : ℝ) : dot_product (k_a_plus_b k) a_minus_2b = 0 ↔ k = 2 :=
by
  sorry

end find_k_l213_213277


namespace registration_methods_l213_213838

-- Define the number of students and groups
def num_students : ℕ := 4
def num_groups : ℕ := 3

-- Theorem stating the total number of different registration methods
theorem registration_methods : (num_groups ^ num_students) = 81 := 
by sorry

end registration_methods_l213_213838


namespace total_tiles_needed_l213_213795

-- Define the dimensions of the dining room
def dining_room_length : ℕ := 15
def dining_room_width : ℕ := 20

-- Define the width of the border
def border_width : ℕ := 2

-- Areas for one-foot by one-foot border tiles
def one_foot_tile_border_tiles : ℕ :=
  2 * (dining_room_width + (dining_room_width - 2 * border_width)) + 
  2 * ((dining_room_length - 2) + (dining_room_length - 2 * border_width))

-- Dimensions of the inner area
def inner_length : ℕ := dining_room_length - 2 * border_width
def inner_width : ℕ := dining_room_width - 2 * border_width

-- Area for two-foot by two-foot tiles
def inner_area : ℕ := inner_length * inner_width
def two_foot_tile_inner_tiles : ℕ := inner_area / 4

-- Total number of tiles
def total_tiles : ℕ := one_foot_tile_border_tiles + two_foot_tile_inner_tiles

-- Prove that the total number of tiles needed is 168
theorem total_tiles_needed : total_tiles = 168 := sorry

end total_tiles_needed_l213_213795


namespace sqrt_meaningful_range_l213_213225

-- Define the condition
def sqrt_condition (x : ℝ) : Prop := 1 - 3 * x ≥ 0

-- State the theorem
theorem sqrt_meaningful_range (x : ℝ) (h : sqrt_condition x) : x ≤ 1 / 3 :=
sorry

end sqrt_meaningful_range_l213_213225


namespace garden_length_l213_213471

theorem garden_length :
  ∀ (w : ℝ) (l : ℝ),
  (l = 2 * w) →
  (2 * l + 2 * w = 150) →
  l = 50 :=
by
  intros w l h1 h2
  sorry

end garden_length_l213_213471


namespace correct_operation_l213_213013

theorem correct_operation (a b x y m : Real) :
  (¬((a^2 * b)^2 = a^2 * b^2)) ∧
  (¬(a^6 / a^2 = a^3)) ∧
  (¬((x + y)^2 = x^2 + y^2)) ∧
  ((-m)^7 / (-m)^2 = -m^5) :=
by
  sorry

end correct_operation_l213_213013


namespace min_value_f_l213_213006

noncomputable def f (x : ℝ) := 2 * (Real.sin x)^3 + (Real.cos x)^2

theorem min_value_f : ∃ x, 0 ≤ x ∧ x ≤ Real.pi / 2 ∧ f x = 26 / 27 :=
by
  sorry

end min_value_f_l213_213006


namespace sharing_watermelons_l213_213415

theorem sharing_watermelons (h : 8 = people_per_watermelon) : people_for_4_watermelons = 32 :=
by
  let people_per_watermelon := 8
  let watermelons := 4
  let people_for_4_watermelons := people_per_watermelon * watermelons
  sorry

end sharing_watermelons_l213_213415


namespace mango_rate_l213_213355

theorem mango_rate (x : ℕ) : 
  (sells_rate : ℕ) = 3 → 
  (profit_percent : ℕ) = 50 → 
  (buying_price : ℚ) = 2 := by
  sorry

end mango_rate_l213_213355


namespace circumference_of_back_wheel_l213_213223

theorem circumference_of_back_wheel
  (C_f : ℝ) (C_b : ℝ) (D : ℝ) (N_b : ℝ)
  (h1 : C_f = 30)
  (h2 : D = 1650)
  (h3 : (N_b + 5) * C_f = D)
  (h4 : N_b * C_b = D) :
  C_b = 33 :=
sorry

end circumference_of_back_wheel_l213_213223


namespace total_investment_amount_l213_213706

-- Define the conditions
def total_interest_in_one_year : ℝ := 1023
def invested_at_6_percent : ℝ := 8200
def interest_rate_6_percent : ℝ := 0.06
def interest_rate_7_5_percent : ℝ := 0.075

-- Define the equation based on the conditions
def interest_from_6_percent_investment : ℝ := invested_at_6_percent * interest_rate_6_percent

def total_investment_is_correct (T : ℝ) : Prop :=
  let interest_from_7_5_percent_investment := (T - invested_at_6_percent) * interest_rate_7_5_percent
  interest_from_6_percent_investment + interest_from_7_5_percent_investment = total_interest_in_one_year

-- Statement to prove
theorem total_investment_amount : total_investment_is_correct 15280 :=
by
  unfold total_investment_is_correct
  unfold interest_from_6_percent_investment
  simp
  sorry

end total_investment_amount_l213_213706


namespace number_of_white_balls_l213_213465

theorem number_of_white_balls (x : ℕ) (h : (x : ℚ) / (x + 12) = 2 / 3) : x = 24 :=
sorry

end number_of_white_balls_l213_213465


namespace an_squared_diff_consec_cubes_l213_213263

theorem an_squared_diff_consec_cubes (a b : ℕ → ℤ) (n : ℕ) :
  a 1 = 1 → b 1 = 0 →
  (∀ n ≥ 1, a (n + 1) = 7 * (a n) + 12 * (b n) + 6) →
  (∀ n ≥ 1, b (n + 1) = 4 * (a n) + 7 * (b n) + 3) →
  a n ^ 2 = (b n + 1) ^ 3 - (b n) ^ 3 :=
by
  sorry

end an_squared_diff_consec_cubes_l213_213263


namespace basic_printer_total_price_l213_213804

theorem basic_printer_total_price (C P : ℝ) (hC : C = 1500) (hP : P = (1/3) * (C + 500 + P)) : C + P = 2500 := 
by
  sorry

end basic_printer_total_price_l213_213804


namespace expected_value_equals_51_l213_213608

noncomputable def expected_value_8_sided_die : ℝ :=
  (1 / 8) * (2 * 1^2 + 2 * 2^2 + 2 * 3^2 + 2 * 4^2 + 2 * 5^2 + 2 * 6^2 + 2 * 7^2 + 2 * 8^2)

theorem expected_value_equals_51 :
  expected_value_8_sided_die = 51 := 
  by 
    sorry

end expected_value_equals_51_l213_213608


namespace prize_winners_l213_213172

theorem prize_winners (n : ℕ) (p1 p2 : ℝ) (h1 : n = 100) (h2 : p1 = 0.4) (h3 : p2 = 0.2) :
  ∃ winners : ℕ, winners = (p2 * (p1 * n)) ∧ winners = 8 :=
by
  sorry

end prize_winners_l213_213172


namespace coin_die_sum_probability_l213_213713

theorem coin_die_sum_probability : 
  let coin_sides := [5, 15]
  let die_sides := [1, 2, 3, 4, 5, 6]
  let ben_age := 18
  (1 / 2 : ℚ) * (1 / 6 : ℚ) = 1 / 12 :=
by
  sorry

end coin_die_sum_probability_l213_213713


namespace hyperbola_real_axis_length_l213_213874

theorem hyperbola_real_axis_length :
  (∃ a : ℝ, (∀ x y : ℝ, (x^2 / 9 - y^2 = 1) → (2 * a = 6))) :=
sorry

end hyperbola_real_axis_length_l213_213874


namespace sum_of_vertices_l213_213103

theorem sum_of_vertices (n : ℕ) (h1 : 6 * n + 12 * n = 216) : 8 * n = 96 :=
by
  -- Proof is omitted intentionally
  sorry

end sum_of_vertices_l213_213103


namespace mechanical_moles_l213_213998

-- Define the conditions
def condition_one (x y : ℝ) : Prop :=
  x + y = 1 / 5

def condition_two (x y : ℝ) : Prop :=
  (1 / (3 * x)) + (2 / (3 * y)) = 10

-- Define the main theorem using the defined conditions
theorem mechanical_moles (x y : ℝ) (h1 : condition_one x y) (h2 : condition_two x y) :
  x = 1 / 30 ∧ y = 1 / 6 :=
  sorry

end mechanical_moles_l213_213998


namespace elsa_final_marbles_l213_213986

def initial_marbles : ℕ := 40
def marbles_lost_at_breakfast : ℕ := 3
def marbles_given_to_susie : ℕ := 5
def marbles_bought_by_mom : ℕ := 12
def twice_marbles_given_back : ℕ := 2 * marbles_given_to_susie

theorem elsa_final_marbles :
    initial_marbles
    - marbles_lost_at_breakfast
    - marbles_given_to_susie
    + marbles_bought_by_mom
    + twice_marbles_given_back = 54 := 
by
    sorry

end elsa_final_marbles_l213_213986


namespace find_area_of_plot_l213_213805

def area_of_plot (B : ℝ) (L : ℝ) (A : ℝ) : Prop :=
  L = 0.75 * B ∧ B = 21.908902300206645 ∧ A = L * B

theorem find_area_of_plot (B L A : ℝ) (h : area_of_plot B L A) : A = 360 := by
  sorry

end find_area_of_plot_l213_213805


namespace count_nine_in_1_to_1000_l213_213000

def count_nine_units : ℕ := 100
def count_nine_tens : ℕ := 100
def count_nine_hundreds : ℕ := 100

theorem count_nine_in_1_to_1000 :
  count_nine_units + count_nine_tens + count_nine_hundreds = 300 :=
by
  sorry

end count_nine_in_1_to_1000_l213_213000


namespace smallest_add_to_multiple_of_4_l213_213370

theorem smallest_add_to_multiple_of_4 : ∃ n : ℕ, n > 0 ∧ (587 + n) % 4 = 0 ∧ ∀ m : ℕ, m > 0 ∧ (587 + m) % 4 = 0 → n ≤ m :=
  sorry

end smallest_add_to_multiple_of_4_l213_213370


namespace sqrt3_f_pi6_lt_f_pi3_l213_213667

open Real

noncomputable def f (x : ℝ) : ℝ := sorry

axiom f_derivative_tan_lt (x : ℝ) (h : 0 < x ∧ x < π / 2) : f x < (deriv f x) * tan x

theorem sqrt3_f_pi6_lt_f_pi3 :
  sqrt 3 * f (π / 6) < f (π / 3) :=
by
  sorry

end sqrt3_f_pi6_lt_f_pi3_l213_213667


namespace geometric_series_sum_l213_213448

-- Define the geometric series
def geometricSeries (a r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r ^ n) / (1 - r)

-- Define the conditions
def a : ℚ := 1 / 4
def r : ℚ := 1 / 4
def n : ℕ := 5

-- Define the sum of the first n terms using the provided formula
def S_n := geometricSeries a r n

-- State the theorem: the sum S_5 equals the given answer
theorem geometric_series_sum :
  S_n = 1023 / 3072 :=
by
  sorry

end geometric_series_sum_l213_213448


namespace cost_price_computer_table_l213_213426

theorem cost_price_computer_table (S : ℝ) (C : ℝ) (h1 : S = C * 1.15) (h2 : S = 5750) : C = 5000 :=
by
  sorry

end cost_price_computer_table_l213_213426


namespace average_weight_of_abc_l213_213438

theorem average_weight_of_abc (A B C : ℝ) (h1 : (A + B) / 2 = 40) (h2 : (B + C) / 2 = 43) (h3 : B = 40) : 
  (A + B + C) / 3 = 42 := 
sorry

end average_weight_of_abc_l213_213438


namespace vector_subtraction_correct_l213_213175

def a : ℝ × ℝ := (2, 1)
def b : ℝ × ℝ := (-3, 4)

theorem vector_subtraction_correct : (a - b) = (5, -3) :=
by 
  have h1 : a = (2, 1) := by rfl
  have h2 : b = (-3, 4) := by rfl
  sorry

end vector_subtraction_correct_l213_213175


namespace tickets_difference_l213_213739

theorem tickets_difference :
  let tickets_won := 48.5
  let yoyo_cost := 11.7
  let keychain_cost := 6.3
  let plush_toy_cost := 16.2
  let total_cost := yoyo_cost + keychain_cost + plush_toy_cost
  let tickets_left := tickets_won - total_cost
  tickets_won - tickets_left = total_cost :=
by
  sorry

end tickets_difference_l213_213739


namespace minimum_police_officers_needed_l213_213324

def grid := (5, 8)
def total_intersections : ℕ := 54
def max_distance_to_police := 2

theorem minimum_police_officers_needed (min_police_needed : ℕ) :
  (min_police_needed = 6) := sorry

end minimum_police_officers_needed_l213_213324


namespace parallel_lines_m_values_l213_213673

theorem parallel_lines_m_values (m : ℝ) :
  (∀ (x y : ℝ), (m - 2) * x - y + 5 = 0) ∧ 
  (∀ (x y : ℝ), (m - 2) * x + (3 - m) * y + 2 = 0) → 
  (m = 2 ∨ m = 4) :=
sorry

end parallel_lines_m_values_l213_213673


namespace johns_pace_l213_213247

variable {J : ℝ} -- John's pace during his final push

theorem johns_pace
  (steve_speed : ℝ := 3.8)
  (initial_gap : ℝ := 15)
  (finish_gap : ℝ := 2)
  (time : ℝ := 42.5)
  (steve_covered : ℝ := steve_speed * time)
  (john_covered : ℝ := steve_covered + initial_gap + finish_gap)
  (johns_pace_equation : J * time = john_covered) :
  J = 4.188 :=
by
  sorry

end johns_pace_l213_213247


namespace at_least_one_fuse_blows_l213_213393

theorem at_least_one_fuse_blows (pA pB : ℝ) (hA : pA = 0.85) (hB : pB = 0.74) (independent : ∀ (A B : Prop), A ∧ B → ¬(A ∨ B)) :
  1 - (1 - pA) * (1 - pB) = 0.961 :=
by
  sorry

end at_least_one_fuse_blows_l213_213393


namespace chord_equation_l213_213695

-- Definitions and conditions
def parabola (x y : ℝ) := y^2 = 8 * x
def point_Q := (4, 1)

-- Statement to prove
theorem chord_equation :
  ∃ (m : ℝ) (c : ℝ), m = 4 ∧ c = -15 ∧
    ∀ (x y : ℝ), (parabola x y ∧ x + y = 8 ∧ y + y = 2) →
      4 * x - y = 15 :=
by
  sorry -- Proof elided

end chord_equation_l213_213695


namespace machine_A_sprockets_per_hour_l213_213846

theorem machine_A_sprockets_per_hour :
  ∀ (A T : ℝ),
    (T > 0 ∧
    (∀ P Q, P = 1.1 * A ∧ Q = 330 / P ∧ Q = 330 / A + 10) →
      A = 3) := 
by
  intro A T
  intro h
  sorry

end machine_A_sprockets_per_hour_l213_213846


namespace angle_AOC_is_45_or_15_l213_213292

theorem angle_AOC_is_45_or_15 (A O B C : Type) (α β γ : ℝ) 
  (h1 : α = 30) (h2 : β = 15) : γ = 45 ∨ γ = 15 :=
sorry

end angle_AOC_is_45_or_15_l213_213292


namespace max_value_neg_a_inv_l213_213535

theorem max_value_neg_a_inv (a : ℝ) (h : a < 0) : a + (1 / a) ≤ -2 := 
by
  sorry

end max_value_neg_a_inv_l213_213535


namespace prove_product_of_b_l213_213483

noncomputable def g (x b : ℝ) := b / (5 * x - 7)

noncomputable def g_inv (y b : ℝ) := (b + 7 * y) / (5 * y)

theorem prove_product_of_b (b1 b2 : ℝ) (h1 : g 3 b1 = g_inv (b1 + 2) b1) (h2 : g 3 b2 = g_inv (b2 + 2) b2) :
  b1 * b2 = -22.39 := by
  sorry

end prove_product_of_b_l213_213483


namespace determine_ABC_l213_213274

-- Define values in the new base system
def base_representation (A B C : ℕ) : ℕ :=
  A * (A+1)^7 + A * (A+1)^6 + A * (A+1)^5 + C * (A+1)^4 + B * (A+1)^3 + B * (A+1)^2 + B * (A+1) + C

-- The conditions given by the problem
def condition (A B C : ℕ) : Prop :=
  ((A+1)^8 - 2*(A+1)^4 + 1) = base_representation A B C

-- The theorem to be proved
theorem determine_ABC : ∃ (A B C : ℕ), A = 2 ∧ B = 0 ∧ C = 1 ∧ condition A B C :=
by
  existsi 2
  existsi 0
  existsi 1
  unfold condition base_representation
  sorry

end determine_ABC_l213_213274


namespace order_of_a_b_c_l213_213422

noncomputable def a : ℝ := Real.log 2 / Real.log 3 -- a = log_3 2
noncomputable def b : ℝ := Real.log 2 -- b = ln 2
noncomputable def c : ℝ := Real.sqrt 5 -- c = 5^(1/2)

theorem order_of_a_b_c : a < b ∧ b < c := by
  sorry

end order_of_a_b_c_l213_213422


namespace barbara_initial_candies_l213_213549

noncomputable def initialCandies (used left: ℝ) := used + left

theorem barbara_initial_candies (used left: ℝ) (h_used: used = 9.0) (h_left: left = 9) : initialCandies used left = 18 := 
by
  rw [h_used, h_left]
  norm_num
  sorry

end barbara_initial_candies_l213_213549


namespace num_2_coins_l213_213043

open Real

theorem num_2_coins (x y z : ℝ) (h1 : x + y + z = 900)
                     (h2 : x + 2 * y + 5 * z = 1950)
                     (h3 : z = 0.5 * x) : y = 450 :=
by sorry

end num_2_coins_l213_213043


namespace max_value_f_at_e_l213_213181

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x

theorem max_value_f_at_e (h : 0 < x) : 
  ∃ e : ℝ, (∀ x : ℝ, 0 < x → f x ≤ f e) ∧ e = Real.exp 1 :=
by
  sorry

end max_value_f_at_e_l213_213181


namespace sum_of_four_primes_div_by_60_l213_213588

open Nat

def is_prime (n : ℕ) : Prop := Nat.Prime n

theorem sum_of_four_primes_div_by_60
  (p q r s : ℕ)
  (hp : is_prime p)
  (hq : is_prime q)
  (hr : is_prime r)
  (hs : is_prime s)
  (horder : 5 < p ∧ p < q ∧ q < r ∧ r < s ∧ s < p + 10) :
  (p + q + r + s) % 60 = 0 :=
by
  sorry


end sum_of_four_primes_div_by_60_l213_213588


namespace value_of_f_at_sqrt2_l213_213168

noncomputable def f (x : ℝ) : ℝ := x^5 - 5 * x^4 + 10 * x^3 - 10 * x^2 + 5 * x - 1

theorem value_of_f_at_sqrt2 :
  f (1 + Real.sqrt 2) = 4 * Real.sqrt 2 := by
  sorry

end value_of_f_at_sqrt2_l213_213168


namespace number_of_silverware_per_setting_l213_213865

-- Conditions
def silverware_weight_per_piece := 4   -- in ounces
def plates_per_setting := 2
def plate_weight := 12  -- in ounces
def tables := 15
def settings_per_table := 8
def backup_settings := 20
def total_weight := 5040  -- in ounces

-- Let's define variables in our conditions
def settings := tables * settings_per_table + backup_settings
def plates_weight_per_setting := plates_per_setting * plate_weight
def total_silverware_weight (S : Nat) := S * silverware_weight_per_piece * settings
def total_plate_weight := plates_weight_per_setting * settings

-- Define the required proof statement
theorem number_of_silverware_per_setting : 
  ∃ S : Nat, (total_silverware_weight S + total_plate_weight = total_weight) ∧ S = 3 :=
by {
  sorry -- proof will be provided here
}

end number_of_silverware_per_setting_l213_213865


namespace equilateral_triangle_of_roots_of_unity_l213_213926

open Complex

/-- Given three distinct non-zero complex numbers z1, z2, z3 such that z1 * z2 = z3 ^ 2 and z2 * z3 = z1 ^ 2.
Prove that if z2 = z1 * alpha, then alpha is a cube root of unity and the points corresponding to z1, z2, z3
form an equilateral triangle in the complex plane -/
theorem equilateral_triangle_of_roots_of_unity {z1 z2 z3 : ℂ} (h1 : z1 ≠ 0) (h2 : z2 ≠ 0) (h3 : z3 ≠ 0)
  (h_distinct : z1 ≠ z2 ∧ z2 ≠ z3 ∧ z1 ≠ z3)
  (h1_2 : z1 * z2 = z3 ^ 2) (h2_3 : z2 * z3 = z1 ^ 2) (alpha : ℂ) (hz2 : z2 = z1 * alpha) :
  alpha^3 = 1 ∧ ∃ (w1 w2 w3 : ℂ), (w1 = z1) ∧ (w2 = z2) ∧ (w3 = z3) ∧ ((w1, w2, w3) = (z1, z1 * α, z3) 
  ∨ (w1, w2, w3) = (z3, z1, z1 * α) ∨ (w1, w2, w3) = (z1 * α, z3, z1)) 
  ∧ dist w1 w2 = dist w2 w3 ∧ dist w2 w3 = dist w3 w1 := sorry

end equilateral_triangle_of_roots_of_unity_l213_213926


namespace find_circle_center_l213_213892

-- Define the conditions as hypotheses
def line1 (x y : ℝ) : Prop := 5 * x - 2 * y = 40
def line2 (x y : ℝ) : Prop := 5 * x - 2 * y = 10
def line_center_constraint (x y : ℝ) : Prop := 3 * x - 4 * y = 0

-- Define the function for the equidistant line
def line_eq (x y : ℝ) : Prop := 5 * x - 2 * y = 25

-- Prove that the center of the circle satisfying the given conditions is (50/7, 75/14)
theorem find_circle_center (x y : ℝ) 
(h1 : line_eq x y)
(h2 : line_center_constraint x y) : 
(x = 50 / 7 ∧ y = 75 / 14) :=
sorry

end find_circle_center_l213_213892


namespace volume_of_pyramid_l213_213369

theorem volume_of_pyramid (A B C : ℝ × ℝ)
  (hA : A = (0, 0)) (hB : B = (28, 0)) (hC : C = (12, 20))
  (D : ℝ × ℝ) (hD : D = ((B.1 + C.1) / 2, (B.2 + C.2) / 2))
  (E : ℝ × ℝ) (hE : E = ((C.1 + A.1) / 2, (C.2 + A.2) / 2))
  (F : ℝ × ℝ) (hF : F = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)) :
  (∃ h : ℝ, h = 10 ∧ ∃ V : ℝ, V = (1 / 3) * 70 * h ∧ V = 700 / 3) :=
by sorry

end volume_of_pyramid_l213_213369


namespace Ahmed_total_distance_traveled_l213_213492

/--
Ahmed stops one-quarter of the way to the store.
He continues for 12 km to reach the store.
Prove that the total distance Ahmed travels is 16 km.
-/
theorem Ahmed_total_distance_traveled
  (D : ℝ) (h1 : D > 0)  -- D is the total distance to the store, assumed to be positive
  (h_stop : D / 4 + 12 = D) : D = 16 := 
sorry

end Ahmed_total_distance_traveled_l213_213492


namespace solve_equation_error_step_l213_213643

theorem solve_equation_error_step 
  (equation : ∀ x : ℝ, (x - 1) / 2 + 1 = (2 * x + 1) / 3) :
  ∃ (step : ℕ), step = 1 ∧
  let s1 := ((x - 1) / 2 + 1) * 6;
  ∀ (x : ℝ), s1 ≠ (((2 * x + 1) / 3) * 6) :=
by
  sorry

end solve_equation_error_step_l213_213643


namespace ratio_problem_l213_213648

theorem ratio_problem (X : ℕ) :
  (18 : ℕ) * 360 = 9 * X → X = 720 :=
by
  intro h
  sorry

end ratio_problem_l213_213648


namespace arithmetic_sequence_a2_a8_l213_213142

theorem arithmetic_sequence_a2_a8 (a : ℕ → ℤ) (d : ℤ)
  (h1 : ∀ n, a n = a 1 + (n - 1) * d)
  (h2 : a 3 + a 4 + a 5 + a 6 + a 7 = 450) :
  a 2 + a 8 = 180 :=
by
  sorry

end arithmetic_sequence_a2_a8_l213_213142


namespace option_c_correct_l213_213885

theorem option_c_correct (a b : ℝ) : (a * b^2)^2 = a^2 * b^4 := by
  sorry

end option_c_correct_l213_213885


namespace find_positive_product_l213_213269

variable (a b c d e f : ℝ)

-- Define the condition that exactly one of the products is positive
def exactly_one_positive (p1 p2 p3 p4 p5 : ℝ) : Prop :=
  (p1 > 0 ∧ p2 < 0 ∧ p3 < 0 ∧ p4 < 0 ∧ p5 < 0) ∨
  (p1 < 0 ∧ p2 > 0 ∧ p3 < 0 ∧ p4 < 0 ∧ p5 < 0) ∨
  (p1 < 0 ∧ p2 < 0 ∧ p3 > 0 ∧ p4 < 0 ∧ p5 < 0) ∨
  (p1 < 0 ∧ p2 < 0 ∧ p3 < 0 ∧ p4 > 0 ∧ p5 < 0) ∨
  (p1 < 0 ∧ p2 < 0 ∧ p3 < 0 ∧ p4 < 0 ∧ p5 > 0)

theorem find_positive_product (h : a ≠ 0) (h' : b ≠ 0) (h'' : c ≠ 0) (h''' : d ≠ 0) (h'''' : e ≠ 0) (h''''' : f ≠ 0) 
  (exactly_one : exactly_one_positive (a * c * d) (a * c * e) (b * d * e) (b * d * f) (b * e * f)) :
  b * d * e > 0 :=
sorry

end find_positive_product_l213_213269


namespace borrowed_years_l213_213563

noncomputable def principal : ℝ := 5396.103896103896
noncomputable def interest_rate : ℝ := 0.06
noncomputable def total_returned : ℝ := 8310

theorem borrowed_years :
  ∃ t : ℝ, (total_returned - principal) = principal * interest_rate * t ∧ t = 9 :=
by
  sorry

end borrowed_years_l213_213563


namespace arithmetic_square_root_of_9_l213_213655

theorem arithmetic_square_root_of_9 : ∃ y : ℕ, y^2 = 9 ∧ y = 3 :=
by
  sorry

end arithmetic_square_root_of_9_l213_213655


namespace find_common_ratio_l213_213146

variable {α : Type*} [LinearOrderedField α] [NormedLinearOrderedField α]

def geometric_sequence (a : ℕ → α) (q : α) : Prop := ∀ n, a (n+1) = q * a n

def sum_first_n_terms (a : ℕ → α) (S : ℕ → α) : Prop := ∀ n, S n = (Finset.range n).sum a

theorem find_common_ratio
  (a : ℕ → α)
  (S : ℕ → α)
  (q : α)
  (pos_terms : ∀ n, 0 < a n)
  (geometric_seq : geometric_sequence a q)
  (sum_eq : sum_first_n_terms a S)
  (eqn : S 1 + 2 * S 5 = 3 * S 3) :
  q = (2:α)^(3 / 2) / 2^(3 / 2) :=
by
  sorry

end find_common_ratio_l213_213146


namespace percent_within_one_std_dev_l213_213934

theorem percent_within_one_std_dev (m d : ℝ) (dist : ℝ → ℝ)
  (symm : ∀ x, dist (m + x) = dist (m - x))
  (less_than_upper_bound : ∀ x, (x < (m + d)) → dist x < 0.92) :
  ∃ p : ℝ, p = 0.84 :=
by
  sorry

end percent_within_one_std_dev_l213_213934


namespace cost_of_white_car_l213_213502

variable (W : ℝ)
variable (red_cars white_cars : ℕ)
variable (rent_red rent_white : ℝ)
variable (rented_hours : ℝ)
variable (total_earnings : ℝ)

theorem cost_of_white_car 
  (h1 : red_cars = 3)
  (h2 : white_cars = 2) 
  (h3 : rent_red = 3)
  (h4 : rented_hours = 3)
  (h5 : total_earnings = 2340) :
  2 * W * (rented_hours * 60) + 3 * rent_red * (rented_hours * 60) = total_earnings → 
  W = 2 :=
by 
  sorry

end cost_of_white_car_l213_213502


namespace no_prime_solutions_l213_213266

theorem no_prime_solutions (p q : ℕ) (hp : p > 5) (hq : q > 5) (pp : Nat.Prime p) (pq : Nat.Prime q)
  (h : p * q ∣ (5^p - 2^p) * (5^q - 2^q)) : False :=
sorry

end no_prime_solutions_l213_213266


namespace least_area_exists_l213_213165

-- Definition of the problem conditions
def is_rectangle (l w : ℕ) : Prop :=
  2 * (l + w) = 120

def area (l w : ℕ) := l * w

-- Statement of the proof problem
theorem least_area_exists :
  ∃ (l w : ℕ), is_rectangle l w ∧ (∀ (l' w' : ℕ), is_rectangle l' w' → area l w ≤ area l' w') ∧ area l w = 59 :=
sorry

end least_area_exists_l213_213165


namespace distance_to_right_focus_l213_213945

open Real

-- Define the elements of the problem
variable (a c : ℝ)
variable (P : ℝ × ℝ) -- Point P on the hyperbola
variable (F1 F2 : ℝ × ℝ) -- Left and right foci
variable (D : ℝ) -- The left directrix

-- Define conditions as Lean statements
def hyperbola_eq : Prop := (a ≠ 0) ∧ (c ≠ 0) ∧ (P.1^2 / a^2 - P.2^2 / 16 = 1)
def point_on_right_branch : Prop := P.1 > 0
def distance_diff : Prop := abs (dist P F1 - dist P F2) = 6
def distance_to_left_directrix : Prop := abs (P.1 - D) = 34 / 5

-- Define theorem to prove the distance from P to the right focus
theorem distance_to_right_focus
  (hp : hyperbola_eq a c P)
  (hbranch : point_on_right_branch P)
  (hdiff : distance_diff P F1 F2)
  (hdirectrix : distance_to_left_directrix P D) :
  dist P F2 = 16 / 3 :=
sorry

end distance_to_right_focus_l213_213945


namespace exists_prime_not_dividing_difference_l213_213760

theorem exists_prime_not_dividing_difference {m : ℕ} (hm : m ≠ 1) : 
  ∃ p : ℕ, Nat.Prime p ∧ ∀ n : ℕ, ¬ p ∣ (n^n - m) := 
sorry

end exists_prime_not_dividing_difference_l213_213760


namespace difference_between_new_and_original_l213_213785

variables (x y : ℤ) -- Declaring variables x and y as integers

-- The original number is represented as 10*x + y, and the new number after swapping is 10*y + x.
-- We need to prove that the difference between the new number and the original number is -9*x + 9*y.
theorem difference_between_new_and_original (x y : ℤ) :
  (10 * y + x) - (10 * x + y) = -9 * x + 9 * y :=
by
  sorry -- Proof placeholder

end difference_between_new_and_original_l213_213785


namespace latest_time_to_reach_80_degrees_l213_213398

theorem latest_time_to_reach_80_degrees :
  ∀ (t : ℝ), (-t^2 + 14 * t + 40 = 80) → t ≤ 10 :=
by
  sorry

end latest_time_to_reach_80_degrees_l213_213398


namespace find_A_and_B_l213_213991

theorem find_A_and_B : 
  ∃ A B : ℝ, 
    (A = 6.5 ∧ B = 0.5) ∧
    (∀ x : ℝ, (8 * x - 17) / ((3 * x + 5) * (x - 3)) = A / (3 * x + 5) + B / (x - 3)) :=
by
  sorry

end find_A_and_B_l213_213991


namespace number_of_keepers_l213_213933

theorem number_of_keepers (hens goats camels : ℕ) (keepers feet heads : ℕ)
  (h_hens : hens = 50)
  (h_goats : goats = 45)
  (h_camels : camels = 8)
  (h_equation : (2 * hens + 4 * goats + 4 * camels + 2 * keepers) = (hens + goats + camels + keepers + 224))
  : keepers = 15 :=
by
sorry

end number_of_keepers_l213_213933
