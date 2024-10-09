import Mathlib

namespace max_integer_a_for_real_roots_l1922_192219

theorem max_integer_a_for_real_roots (a : ℤ) :
  (((a - 1) * x^2 - 2 * x + 3 = 0) ∧ a ≠ 1) → a ≤ 0 ∧ (∀ b : ℤ, ((b - 1) * x^2 - 2 * x + 3 = 0) ∧ a ≠ 1 → b ≤ 0) :=
sorry

end max_integer_a_for_real_roots_l1922_192219


namespace average_rainfall_correct_l1922_192297

-- Define the monthly rainfall
def january_rainfall := 150
def february_rainfall := 200
def july_rainfall := 366
def other_months_rainfall := 100

-- Calculate total yearly rainfall
def total_yearly_rainfall := 
  january_rainfall + 
  february_rainfall + 
  july_rainfall + 
  (9 * other_months_rainfall)

-- Calculate total hours in a year
def days_per_month := 30
def total_days_in_year := 12 * days_per_month
def hours_per_day := 24
def total_hours_in_year := total_days_in_year * hours_per_day

-- Calculate average rainfall per hour
def average_rainfall_per_hour := 
  total_yearly_rainfall / total_hours_in_year

theorem average_rainfall_correct :
  average_rainfall_per_hour = (101 / 540) := sorry

end average_rainfall_correct_l1922_192297


namespace machine_b_finishes_in_12_hours_l1922_192268

noncomputable def machine_b_time : ℝ :=
  let rA := 1 / 4  -- rate of Machine A
  let rC := 1 / 6  -- rate of Machine C
  let rTotalTogether := 1 / 2  -- rate of all machines working together
  let rB := (rTotalTogether - rA - rC)  -- isolate the rate of Machine B
  1 / rB  -- time for Machine B to finish the job

theorem machine_b_finishes_in_12_hours : machine_b_time = 12 :=
by
  sorry

end machine_b_finishes_in_12_hours_l1922_192268


namespace triplet_sum_not_equal_two_l1922_192203

theorem triplet_sum_not_equal_two :
  ¬((1.2 + -2.2 + 2) = 2) ∧ ¬((- 4 / 3 + - 2 / 3 + 3) = 2) :=
by
  sorry

end triplet_sum_not_equal_two_l1922_192203


namespace remainder_ab_mod_n_l1922_192288

theorem remainder_ab_mod_n (n : ℕ) (a c : ℤ) (h1 : a * c ≡ 1 [ZMOD n]) (h2 : b = a * c) :
    (a * b % n) = (a % n) :=
  by
  sorry

end remainder_ab_mod_n_l1922_192288


namespace distance_Xiaolan_to_Xiaohong_reverse_l1922_192272

def Xiaohong_to_Xiaolan := 30
def Xiaolu_to_Xiaohong := 26
def Xiaolan_to_Xiaolu := 28

def total_perimeter : ℕ := Xiaohong_to_Xiaolan + Xiaolan_to_Xiaolu + Xiaolu_to_Xiaohong

theorem distance_Xiaolan_to_Xiaohong_reverse : total_perimeter - Xiaohong_to_Xiaolan = 54 :=
by
  rw [total_perimeter]
  norm_num
  sorry

end distance_Xiaolan_to_Xiaohong_reverse_l1922_192272


namespace no_value_of_a_l1922_192235

theorem no_value_of_a (a : ℝ) (x y : ℝ) : ¬∃ x1 x2 : ℝ, (x1 ≠ x2) ∧ (x1^2 + y^2 + 2 * x1 = abs (x1 - a) - 1) ∧ (x2^2 + y^2 + 2 * x2 = abs (x2 - a) - 1) := 
by
  sorry

end no_value_of_a_l1922_192235


namespace path_area_and_cost_l1922_192265

theorem path_area_and_cost:
  let length_grass_field := 75
  let width_grass_field := 55
  let path_width := 3.5
  let cost_per_sq_meter := 2
  let length_with_path := length_grass_field + 2 * path_width
  let width_with_path := width_grass_field + 2 * path_width
  let area_with_path := length_with_path * width_with_path
  let area_grass_field := length_grass_field * width_grass_field
  let area_path := area_with_path - area_grass_field
  let cost_of_construction := area_path * cost_per_sq_meter
  area_path = 959 ∧ cost_of_construction = 1918 :=
by
  sorry

end path_area_and_cost_l1922_192265


namespace chocolate_game_winner_l1922_192284

-- Definitions of conditions for the problem
def chocolate_bar (m n : ℕ) := m * n

-- Theorem statement with conditions and conclusion
theorem chocolate_game_winner (m n : ℕ) (h1 : chocolate_bar m n = 48) : 
  ( ∃ first_player_wins : true, true) :=
by sorry

end chocolate_game_winner_l1922_192284


namespace measure_of_C_angle_maximum_area_triangle_l1922_192253

-- Proof Problem 1: Measure of angle C
theorem measure_of_C_angle (A B C : ℝ) (a b c : ℝ)
  (h1 : 0 < C ∧ C < Real.pi)
  (m n : ℝ × ℝ)
  (h2 : m = (Real.sin A, Real.sin B))
  (h3 : n = (Real.cos B, Real.cos A))
  (h4 : m.1 * n.1 + m.2 * n.2 = -Real.sin (2 * C)) :
  C = 2 * Real.pi / 3 :=
sorry

-- Proof Problem 2: Maximum area of triangle ABC
theorem maximum_area_triangle (A B C : ℝ) (a b c S : ℝ)
  (h1 : c = 2 * Real.sqrt 3)
  (h2 : Real.cos C = -1 / 2)
  (h3 : S = 1 / 2 * a * b * Real.sin (2 * Real.pi / 3)): 
  S ≤ Real.sqrt 3 :=
sorry

end measure_of_C_angle_maximum_area_triangle_l1922_192253


namespace find_y_l1922_192244

theorem find_y (x y : ℝ) (h1 : x = 4 * y) (h2 : (1 / 2) * x = 1) : y = 1 / 2 :=
by
  sorry

end find_y_l1922_192244


namespace garden_breadth_l1922_192237

theorem garden_breadth (perimeter length breadth : ℕ) 
    (h₁ : perimeter = 680)
    (h₂ : length = 258)
    (h₃ : perimeter = 2 * (length + breadth)) : 
    breadth = 82 := 
sorry

end garden_breadth_l1922_192237


namespace solution_set_of_inequality_l1922_192247

theorem solution_set_of_inequality :
  { x : ℝ | x^2 + x - 2 < 0 } = { x : ℝ | -2 < x ∧ x < 1 } :=
sorry

end solution_set_of_inequality_l1922_192247


namespace jason_initial_cards_l1922_192271

-- Definitions based on conditions
def cards_given_away : ℕ := 4
def cards_left : ℕ := 5

-- Theorem to prove
theorem jason_initial_cards : cards_given_away + cards_left = 9 :=
by sorry

end jason_initial_cards_l1922_192271


namespace sqrt_sum_inequality_l1922_192292

variable (a b c d : ℝ)

theorem sqrt_sum_inequality
  (h1 : 0 < a)
  (h2 : a < b)
  (h3 : b < c)
  (h4 : c < d)
  (h5 : a + d = b + c) :
  Real.sqrt a + Real.sqrt d < Real.sqrt b + Real.sqrt c :=
by
  sorry

end sqrt_sum_inequality_l1922_192292


namespace exists_small_area_triangle_l1922_192260

structure LatticePoint where
  x : Int
  y : Int

def isValidPoint (p : LatticePoint) : Prop := 
  |p.x| ≤ 2 ∧ |p.y| ≤ 2

def noThreeCollinear (points : List LatticePoint) : Prop := 
  ∀ (p1 p2 p3 : LatticePoint), p1 ∈ points → p2 ∈ points → p3 ∈ points → 
  ((p2.x - p1.x) * (p3.y - p1.y) ≠ (p3.x - p1.x) * (p2.y - p1.y))

def triangleArea (p1 p2 p3 : LatticePoint) : ℝ :=
  0.5 * |(p1.x * (p2.y - p3.y) + p2.x * (p3.y - p1.y) + p3.x * (p1.y - p2.y) : ℝ)|

theorem exists_small_area_triangle
  (points : List LatticePoint)
  (h1 : ∀ p ∈ points, isValidPoint p)
  (h2 : noThreeCollinear points) :
  ∃ (p1 p2 p3 : LatticePoint), p1 ∈ points ∧ p2 ∈ points ∧ p3 ∈ points ∧ triangleArea p1 p2 p3 ≤ 2 :=
sorry

end exists_small_area_triangle_l1922_192260


namespace piecewise_linear_function_y_at_x_10_l1922_192283

theorem piecewise_linear_function_y_at_x_10
  (k1 k2 : ℝ)
  (y : ℝ → ℝ)
  (hx1 : ∀ x < 0, y x = k1 * x)
  (hx2 : ∀ x ≥ 0, y x = k2 * x)
  (h_y_pos : y 2 = 4)
  (h_y_neg : y (-5) = -20) :
  y 10 = 20 :=
by
  sorry

end piecewise_linear_function_y_at_x_10_l1922_192283


namespace area_of_sector_l1922_192232

theorem area_of_sector (r : ℝ) (n : ℝ) (h_r : r = 3) (h_n : n = 120) : 
  (n / 360) * π * r^2 = 3 * π :=
by
  rw [h_r, h_n] -- Plugin in the given values first
  norm_num     -- Normalize numerical expressions
  sorry        -- Placeholder for further simplification if needed. 

end area_of_sector_l1922_192232


namespace quadratic_no_real_roots_l1922_192223

theorem quadratic_no_real_roots (a b c d : ℝ)  :
  a^2 - 4 * b < 0 → c^2 - 4 * d < 0 → ( (a + c) / 2 )^2 - 4 * ( (b + d) / 2 ) < 0 :=
by
  sorry

end quadratic_no_real_roots_l1922_192223


namespace hands_in_class_not_including_peters_l1922_192250

def total_students : ℕ := 11
def hands_per_student : ℕ := 2
def peter_hands : ℕ := 2

theorem hands_in_class_not_including_peters :  (total_students * hands_per_student) - peter_hands = 20 :=
by
  sorry

end hands_in_class_not_including_peters_l1922_192250


namespace range_of_a_l1922_192207

-- Define the set M
def M : Set ℝ := { x | -1 ≤ x ∧ x < 2 }

-- Define the set N
def N (a : ℝ) : Set ℝ := { x | x ≤ a }

-- The theorem to be proved
theorem range_of_a (a : ℝ) (h : (M ∩ N a).Nonempty) : a ≥ -1 := sorry

end range_of_a_l1922_192207


namespace taylor_scores_l1922_192259

theorem taylor_scores 
    (yellow_scores : ℕ := 78)
    (white_ratio : ℕ := 7)
    (black_ratio : ℕ := 6)
    (total_ratio : ℕ := white_ratio + black_ratio)
    (difference_ratio : ℕ := white_ratio - black_ratio) :
    (2/3 : ℚ) * (yellow_scores * difference_ratio / total_ratio) = 4 := by
  sorry

end taylor_scores_l1922_192259


namespace remaining_bollards_to_be_installed_l1922_192279

-- Definitions based on the problem's conditions
def total_bollards_per_side := 4000
def sides := 2
def total_bollards := total_bollards_per_side * sides
def installed_fraction := 3 / 4
def installed_bollards := installed_fraction * total_bollards
def remaining_bollards := total_bollards - installed_bollards

-- The statement to be proved
theorem remaining_bollards_to_be_installed : remaining_bollards = 2000 :=
  by sorry

end remaining_bollards_to_be_installed_l1922_192279


namespace distance_center_to_plane_l1922_192210

noncomputable def sphere_center_to_plane_distance 
  (volume : ℝ) (AB AC : ℝ) (angleACB : ℝ) : ℝ :=
  let R := (3 * volume / 4 / Real.pi)^(1 / 3);
  let circumradius := AB / (2 * Real.sin (angleACB / 2));
  Real.sqrt (R^2 - circumradius^2)

theorem distance_center_to_plane 
  (volume : ℝ) (AB : ℝ) (angleACB : ℝ)
  (h_volume : volume = 500 * Real.pi / 3)
  (h_AB : AB = 4 * Real.sqrt 3)
  (h_angleACB : angleACB = Real.pi / 3) :
  sphere_center_to_plane_distance volume AB angleACB = 3 :=
by
  sorry

end distance_center_to_plane_l1922_192210


namespace simplify_frac_l1922_192213

theorem simplify_frac :
  (1 / (1 / (Real.sqrt 3 + 2) + 2 / (Real.sqrt 5 - 2))) = 
  (Real.sqrt 3 - 2 * Real.sqrt 5 - 2) :=
by
  sorry

end simplify_frac_l1922_192213


namespace exists_infinite_solutions_l1922_192281

noncomputable def infinite_solutions_exist (m : ℕ) : Prop := 
  ∃ (a b c : ℕ), 0 < a ∧ 0 < b ∧ 0 < c ∧  (1 / a + 1 / b + 1 / c + 1 / (a * b * c) = m / (a + b + c))

theorem exists_infinite_solutions : infinite_solutions_exist 12 :=
  sorry

end exists_infinite_solutions_l1922_192281


namespace determine_m_l1922_192296

theorem determine_m (a b : ℝ) (m : ℝ) :
  (2 * (a ^ 2 - 2 * a * b - b ^ 2) - (a ^ 2 + m * a * b + 2 * b ^ 2)) = a ^ 2 - (4 + m) * a * b - 4 * b ^ 2 →
  ¬(∃ (c : ℝ), (a ^ 2 - (4 + m) * a * b - 4 * b ^ 2) = a ^ 2 + c * (a * b) + k) →
  m = -4 :=
sorry

end determine_m_l1922_192296


namespace smallest_k_l1922_192220

theorem smallest_k :
  ∃ k : ℤ, k > 1 ∧ k % 13 = 1 ∧ k % 8 = 1 ∧ k % 4 = 1 ∧ k = 105 :=
by
  sorry

end smallest_k_l1922_192220


namespace abs_x_minus_1_le_1_is_equivalent_to_x_le_2_l1922_192291

theorem abs_x_minus_1_le_1_is_equivalent_to_x_le_2 (x : ℝ) :
  (|x - 1| ≤ 1) ↔ (x ≤ 2) := sorry

end abs_x_minus_1_le_1_is_equivalent_to_x_le_2_l1922_192291


namespace range_of_k_for_empty_solution_set_l1922_192299

theorem range_of_k_for_empty_solution_set :
  ∀ (k : ℝ), (∀ (x : ℝ), k * x^2 - 2 * |x - 1| + 3 * k < 0 → False) ↔ k ≥ 1 :=
by sorry

end range_of_k_for_empty_solution_set_l1922_192299


namespace base_five_product_l1922_192211

open Nat

/-- Definition of the base 5 representation of 131 and 21 --/
def n131 := 1 * 5^2 + 3 * 5^1 + 1 * 5^0
def n21 := 2 * 5^1 + 1 * 5^0

/-- Definition of the expected result in base 5 --/
def expected_result := 3 * 5^3 + 2 * 5^2 + 5 * 5^1 + 1 * 5^0

/-- Claim to prove that the product of 131_5 and 21_5 equals 3251_5 --/
theorem base_five_product : n131 * n21 = expected_result := by sorry

end base_five_product_l1922_192211


namespace problem_l1922_192266

theorem problem (A B : ℝ) (h₀ : 0 < A) (h₁ : 0 < B) (h₂ : B > A) (n c : ℝ) 
  (h₃ : B = A * (1 + n / 100)) (h₄ : A = B * (1 - c / 100)) :
  A * Real.sqrt (100 + n) = B * Real.sqrt (100 - c) :=
by
  sorry

end problem_l1922_192266


namespace smallest_percent_both_coffee_tea_l1922_192294

noncomputable def smallest_percent_coffee_tea (P_C P_T P_not_C_or_T : ℝ) : ℝ :=
  let P_C_or_T := 1 - P_not_C_or_T
  let P_C_and_T := P_C + P_T - P_C_or_T
  P_C_and_T

theorem smallest_percent_both_coffee_tea :
  smallest_percent_coffee_tea 0.9 0.85 0.15 = 0.9 :=
by
  sorry

end smallest_percent_both_coffee_tea_l1922_192294


namespace johns_piano_total_cost_l1922_192285

theorem johns_piano_total_cost : 
  let piano_cost := 500
  let original_lessons_cost := 20 * 40
  let discount := (25 / 100) * original_lessons_cost
  let discounted_lessons_cost := original_lessons_cost - discount
  let sheet_music_cost := 75
  let maintenance_fees := 100
  let total_cost := piano_cost + discounted_lessons_cost + sheet_music_cost + maintenance_fees
  total_cost = 1275 := 
by
  let piano_cost := 500
  let original_lessons_cost := 800
  let discount := 200
  let discounted_lessons_cost := 600
  let sheet_music_cost := 75
  let maintenance_fees := 100
  let total_cost := piano_cost + discounted_lessons_cost + sheet_music_cost + maintenance_fees
  -- Proof skipped
  sorry

end johns_piano_total_cost_l1922_192285


namespace cos_alpha_second_quadrant_l1922_192270

variable (α : Real)
variable (h₁ : α ∈ Set.Ioo (π / 2) π)
variable (h₂ : Real.sin α = 5 / 13)

theorem cos_alpha_second_quadrant : Real.cos α = -12 / 13 := by
  sorry

end cos_alpha_second_quadrant_l1922_192270


namespace soldier_rearrangement_20x20_soldier_rearrangement_21x21_l1922_192209

theorem soldier_rearrangement_20x20 (d : ℝ) : d ≤ 10 * Real.sqrt 2 :=
by
  -- Problem (a) setup and conditions
  sorry

theorem soldier_rearrangement_21x21 (d : ℝ) : d ≤ 10 * Real.sqrt 2 :=
by
  -- Problem (b) setup and conditions
  sorry

end soldier_rearrangement_20x20_soldier_rearrangement_21x21_l1922_192209


namespace sin_double_angle_l1922_192289

theorem sin_double_angle (A : ℝ) (h1 : π / 2 < A) (h2 : A < π) (h3 : Real.sin A = 4 / 5) : Real.sin (2 * A) = -24 / 25 := 
by 
  sorry

end sin_double_angle_l1922_192289


namespace unique_solution_for_divisibility_l1922_192224

theorem unique_solution_for_divisibility (a b : ℕ) (ha : a > 0) (hb : b > 0) :
  (a^2 + b^2) ∣ (a^3 + 1) ∧ (a^2 + b^2) ∣ (b^3 + 1) → (a = 1 ∧ b = 1) :=
by
  intro h
  sorry

end unique_solution_for_divisibility_l1922_192224


namespace burrito_count_l1922_192240

def burrito_orders (wraps beef_fillings chicken_fillings : ℕ) :=
  if wraps = 5 ∧ beef_fillings >= 4 ∧ chicken_fillings >= 3 then 25 else 0

theorem burrito_count : burrito_orders 5 4 3 = 25 := by
  sorry

end burrito_count_l1922_192240


namespace find_m_l1922_192280

theorem find_m (m : ℝ) (h : |m - 4| = |2 * m + 7|) : m = -11 ∨ m = -1 :=
sorry

end find_m_l1922_192280


namespace remainder_is_3_l1922_192217

-- Define the polynomial p(x)
def p (x : ℝ) := x^3 - 3 * x + 5

-- Define the divisor d(x)
def d (x : ℝ) := x - 1

-- The theorem: remainder when p(x) is divided by d(x)
theorem remainder_is_3 : p 1 = 3 := by 
  sorry

end remainder_is_3_l1922_192217


namespace price_increase_problem_l1922_192228

variable (P P' x : ℝ)

theorem price_increase_problem
  (h1 : P' = P * (1 + x / 100))
  (h2 : P = P' * (1 - 23.076923076923077 / 100)) :
  x = 30 :=
by
  sorry

end price_increase_problem_l1922_192228


namespace intersection_with_y_axis_l1922_192286

theorem intersection_with_y_axis :
  (∃ y : ℝ, y = -(0 + 2)^2 + 6 ∧ (0, y) = (0, 2)) :=
by
  sorry

end intersection_with_y_axis_l1922_192286


namespace segment_length_tangent_circles_l1922_192277

theorem segment_length_tangent_circles
  (r1 r2 : ℝ)
  (h1 : r1 > 0)
  (h2 : r2 > 0)
  (h3 : 7 - 4 * Real.sqrt 3 ≤ r1 / r2)
  (h4 : r1 / r2 ≤ 7 + 4 * Real.sqrt 3)
  :
  ∃ d : ℝ, d^2 = (1 / 12) * (14 * r1 * r2 - r1^2 - r2^2) :=
sorry

end segment_length_tangent_circles_l1922_192277


namespace rope_length_l1922_192214

theorem rope_length (x S : ℝ) (H1 : x + 7 * S = 140)
(H2 : x - S = 20) : x = 35 := by
sorry

end rope_length_l1922_192214


namespace width_at_bottom_of_stream_l1922_192273

theorem width_at_bottom_of_stream 
    (top_width : ℝ) (area : ℝ) (height : ℝ) (bottom_width : ℝ) :
    top_width = 10 → area = 640 → height = 80 → 
    2 * area = height * (top_width + bottom_width) → 
    bottom_width = 6 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  -- Finding bottom width
  have h5 : 2 * 640 = 80 * (10 + bottom_width) := h4
  norm_num at h5
  linarith [h5]

#check width_at_bottom_of_stream

end width_at_bottom_of_stream_l1922_192273


namespace abs_neg_three_l1922_192208

theorem abs_neg_three : abs (-3) = 3 :=
by
  sorry

end abs_neg_three_l1922_192208


namespace actual_distance_traveled_l1922_192278

theorem actual_distance_traveled (D : ℝ) (T : ℝ) (h1 : D = 15 * T) (h2 : D + 35 = 25 * T) : D = 52.5 := 
by
  sorry

end actual_distance_traveled_l1922_192278


namespace quadratic_coefficients_l1922_192212

theorem quadratic_coefficients :
  ∀ x : ℝ, x * (x + 2) = 5 * (x - 2) → ∃ a b c : ℝ, a = 1 ∧ b = -3 ∧ c = 10 ∧ a * x^2 + b * x + c = 0 := by
  intros x h
  use 1, -3, 10
  sorry

end quadratic_coefficients_l1922_192212


namespace total_travel_time_l1922_192252

/-
Define the conditions:
1. Distance_1 is 150 miles,
2. Speed_1 is 50 mph,
3. Stop_time is 0.5 hours,
4. Distance_2 is 200 miles,
5. Speed_2 is 75 mph.

and prove that the total time equals 6.17 hours.
-/

theorem total_travel_time :
  let distance1 := 150
  let speed1 := 50
  let stop_time := 0.5
  let distance2 := 200
  let speed2 := 75
  let time1 := distance1 / speed1
  let time2 := distance2 / speed2
  time1 + stop_time + time2 = 6.17 :=
by {
  -- sorry to skip the proof part
  sorry
}

end total_travel_time_l1922_192252


namespace lily_has_26_dollars_left_for_coffee_l1922_192225

-- Define the initial amount of money Lily has
def initialMoney : ℕ := 60

-- Define the costs of items
def celeryCost : ℕ := 5
def cerealCost : ℕ := 12 / 2
def breadCost : ℕ := 8
def milkCost : ℕ := 10 * 9 / 10
def potatoCostEach : ℕ := 1
def numberOfPotatoes : ℕ := 6
def totalPotatoCost : ℕ := potatoCostEach * numberOfPotatoes

-- Define the total amount spent on the items
def totalSpent : ℕ := celeryCost + cerealCost + breadCost + milkCost + totalPotatoCost

-- Define the amount left for coffee
def amountLeftForCoffee : ℕ := initialMoney - totalSpent

-- The theorem to prove
theorem lily_has_26_dollars_left_for_coffee :
  amountLeftForCoffee = 26 := by
  sorry

end lily_has_26_dollars_left_for_coffee_l1922_192225


namespace pentagon_quadrilateral_sum_of_angles_l1922_192231

   theorem pentagon_quadrilateral_sum_of_angles
     (exterior_angle_pentagon : ℕ := 72)
     (interior_angle_pentagon : ℕ := 108)
     (sum_interior_angles_quadrilateral : ℕ := 360)
     (reflex_angle : ℕ := 252) :
     (sum_interior_angles_quadrilateral - reflex_angle = interior_angle_pentagon) :=
   by
     sorry
   
end pentagon_quadrilateral_sum_of_angles_l1922_192231


namespace solve_system_l1922_192229

theorem solve_system (x₁ x₂ x₃ : ℝ) (h₁ : 2 * x₁^2 / (1 + x₁^2) = x₂) (h₂ : 2 * x₂^2 / (1 + x₂^2) = x₃) (h₃ : 2 * x₃^2 / (1 + x₃^2) = x₁) :
  (x₁ = 0 ∧ x₂ = 0 ∧ x₃ = 0) ∨ (x₁ = 1 ∧ x₂ = 1 ∧ x₃ = 1) :=
sorry

end solve_system_l1922_192229


namespace expected_value_T_l1922_192298

def boys_girls_expected_value (M N : ℕ) : ℚ :=
  2 * ((M / (M + N : ℚ)) * (N / (M + N - 1 : ℚ)))

theorem expected_value_T (M N : ℕ) (hM : M = 10) (hN : N = 10) :
  boys_girls_expected_value M N = 20 / 19 :=
by 
  rw [hM, hN]
  sorry

end expected_value_T_l1922_192298


namespace negation_of_p_l1922_192258

variable (x : ℝ)

-- Define the original proposition p
def p := ∀ x, x^2 < 1 → x < 1

-- Define the negation of p
def neg_p := ∃ x₀, x₀^2 ≥ 1 ∧ x₀ < 1

-- State the theorem that negates p
theorem negation_of_p : ¬ p ↔ neg_p :=
by
  sorry

end negation_of_p_l1922_192258


namespace largest_divisor_of_expression_l1922_192205

theorem largest_divisor_of_expression (x : ℤ) (h_odd : x % 2 = 1) : 
  1200 ∣ ((10 * x - 4) * (10 * x) * (5 * x + 15)) := 
  sorry

end largest_divisor_of_expression_l1922_192205


namespace most_prolific_mathematician_is_euler_l1922_192290

noncomputable def prolific_mathematician (collected_works_volume_count: ℕ) (publishing_organization: String) : String :=
  if collected_works_volume_count > 75 ∧ publishing_organization = "Swiss Society of Natural Sciences" then
    "Leonhard Euler"
  else
    "Unknown"

theorem most_prolific_mathematician_is_euler :
  prolific_mathematician 76 "Swiss Society of Natural Sciences" = "Leonhard Euler" :=
by
  sorry

end most_prolific_mathematician_is_euler_l1922_192290


namespace find_n_mod_11_l1922_192236

theorem find_n_mod_11 : ∃ n : ℕ, 0 ≤ n ∧ n ≤ 10 ∧ n ≡ 50000 [MOD 11] ∧ n = 5 :=
sorry

end find_n_mod_11_l1922_192236


namespace minimum_pie_pieces_l1922_192241

theorem minimum_pie_pieces (p q : ℕ) (h_coprime : Nat.gcd p q = 1) : 
  ∃ n, (∀ k, k = p ∨ k = q → (n ≠ 0 → n % k = 0)) ∧ n = p + q - 1 :=
by {
  sorry
}

end minimum_pie_pieces_l1922_192241


namespace ratio_red_to_green_apple_l1922_192226

def total_apples : ℕ := 496
def green_apples : ℕ := 124
def red_apples : ℕ := total_apples - green_apples

theorem ratio_red_to_green_apple :
  red_apples / green_apples = 93 / 31 :=
by
  sorry

end ratio_red_to_green_apple_l1922_192226


namespace xy_relationship_l1922_192218

theorem xy_relationship : 
  (∀ x y, (x = 1 ∧ y = 1) ∨ (x = 2 ∧ y = 4) ∨ (x = 3 ∧ y = 9) ∨ (x = 4 ∧ y = 16) ∨ (x = 5 ∧ y = 25) 
  → y = x * x) :=
by {
  sorry
}

end xy_relationship_l1922_192218


namespace rank_of_A_l1922_192263

def A : Matrix (Fin 3) (Fin 5) ℝ :=
  ![![1, 2, 3, 5, 8],
    ![0, 1, 4, 6, 9],
    ![0, 0, 1, 7, 10]]

theorem rank_of_A : A.rank = 3 :=
by sorry

end rank_of_A_l1922_192263


namespace value_of_expression_l1922_192276

variable (a b : ℝ)

def system_of_equations : Prop :=
  (2 * a - b = 12) ∧ (a + 2 * b = 8)

theorem value_of_expression (h : system_of_equations a b) : 3 * a + b = 20 :=
  sorry

end value_of_expression_l1922_192276


namespace probability_of_three_correct_deliveries_l1922_192262

-- Define a combination function
def combination (n k : ℕ) : ℕ :=
  n.choose k

-- Define the factorial function
def factorial : ℕ → ℕ
  | 0 => 1
  | (n + 1) => (n + 1) * factorial n

-- Define the problem with conditions and derive the required probability
theorem probability_of_three_correct_deliveries :
  (combination 5 3) / (factorial 5) = 1 / 12 := by
  sorry

end probability_of_three_correct_deliveries_l1922_192262


namespace Barbier_theorem_for_delta_curves_l1922_192274

def delta_curve (h : ℝ) : Type := sorry 
def can_rotate_freely_in_3gon (K : delta_curve h) : Prop := sorry
def length_of_curve (K : delta_curve h) : ℝ := sorry

theorem Barbier_theorem_for_delta_curves
  (K : delta_curve h)
  (h : ℝ)
  (H : can_rotate_freely_in_3gon K)
  : length_of_curve K = (2 * Real.pi * h) / 3 := 
sorry

end Barbier_theorem_for_delta_curves_l1922_192274


namespace crayons_left_l1922_192275

theorem crayons_left (initial_crayons : ℕ) (kiley_fraction : ℚ) (joe_fraction : ℚ) 
  (initial_crayons_eq : initial_crayons = 48) 
  (kiley_fraction_eq : kiley_fraction = 1/4) 
  (joe_fraction_eq : joe_fraction = 1/2): 
  (initial_crayons - initial_crayons * kiley_fraction - (initial_crayons - initial_crayons * kiley_fraction) * joe_fraction) = 18 := 
by 
  sorry

end crayons_left_l1922_192275


namespace ellipse_focus_xaxis_l1922_192242

theorem ellipse_focus_xaxis (k : ℝ) (h : 1 - k > 2 + k ∧ 2 + k > 0) : -2 < k ∧ k < -1/2 :=
by sorry

end ellipse_focus_xaxis_l1922_192242


namespace edwards_final_money_l1922_192249

def small_lawn_rate : ℕ := 8
def medium_lawn_rate : ℕ := 12
def large_lawn_rate : ℕ := 15

def first_garden_rate : ℕ := 10
def second_garden_rate : ℕ := 12
def additional_garden_rate : ℕ := 15

def num_small_lawns : ℕ := 3
def num_medium_lawns : ℕ := 1
def num_large_lawns : ℕ := 1
def num_gardens_cleaned : ℕ := 5

def fuel_expense : ℕ := 10
def equipment_rental_expense : ℕ := 15
def initial_savings : ℕ := 7

theorem edwards_final_money : 
  (num_small_lawns * small_lawn_rate + 
   num_medium_lawns * medium_lawn_rate + 
   num_large_lawns * large_lawn_rate + 
   (first_garden_rate + second_garden_rate + (num_gardens_cleaned - 2) * additional_garden_rate) + 
   initial_savings - 
   (fuel_expense + equipment_rental_expense)) = 100 := 
  by 
  -- The proof goes here
  sorry

end edwards_final_money_l1922_192249


namespace polygon_diagonalization_l1922_192254

theorem polygon_diagonalization (n : ℕ) (h : n ≥ 3) : 
  ∃ (triangles : ℕ), triangles = n - 2 ∧ 
  (∀ (polygons : ℕ), 3 ≤ polygons → polygons < n → ∃ k, k = polygons - 2) := 
by {
  -- base case
  sorry
}

end polygon_diagonalization_l1922_192254


namespace inequality_holds_l1922_192257

variable (b : ℝ)

theorem inequality_holds (b : ℝ) : (3 * b - 1) * (4 * b + 1) > (2 * b + 1) * (5 * b - 3) :=
by
  sorry

end inequality_holds_l1922_192257


namespace valid_m_values_l1922_192216

theorem valid_m_values :
  ∃ (m : ℕ), (m ∣ 720) ∧ (m ≠ 1) ∧ (m ≠ 720) ∧ ((720 / m) > 1) ∧ ((30 - 2) = 28) := 
sorry

end valid_m_values_l1922_192216


namespace value_of_a_set_of_x_l1922_192234

open Real

noncomputable def f (x a : ℝ) : ℝ := sin (x + π / 6) + sin (x - π / 6) + cos x + a

theorem value_of_a : ∀ a, (∀ x, f x a ≤ 1) → a = -1 :=
sorry

theorem set_of_x (a : ℝ) (k : ℤ) : a = -1 →
  {x : ℝ | f x a = 0} = {x | ∃ k : ℤ, x = 2 * k * π ∨ x = 2 * k * π + 2 * π / 3} :=
sorry

end value_of_a_set_of_x_l1922_192234


namespace gcd_of_a_and_b_is_one_l1922_192230

theorem gcd_of_a_and_b_is_one {a b : ℕ} (h1 : a > b) (h2 : Nat.gcd (a + b) (a - b) = 1) : Nat.gcd a b = 1 :=
by
  sorry

end gcd_of_a_and_b_is_one_l1922_192230


namespace find_sale_month_4_l1922_192246

-- Define the given sales data
def sale_month_1: ℕ := 5124
def sale_month_2: ℕ := 5366
def sale_month_3: ℕ := 5808
def sale_month_5: ℕ := 6124
def sale_month_6: ℕ := 4579
def average_sale_per_month: ℕ := 5400

-- Define the goal: Sale in the fourth month
def sale_month_4: ℕ := 5399

-- Prove that the total sales conforms to the given average sale
theorem find_sale_month_4 :
  sale_month_1 + sale_month_2 + sale_month_3 + sale_month_4 + sale_month_5 + sale_month_6 = 6 * average_sale_per_month :=
by
  sorry

end find_sale_month_4_l1922_192246


namespace printer_diff_l1922_192293

theorem printer_diff (A B : ℚ) (hA : A * 60 = 35) (hAB : (A + B) * 24 = 35) : B - A = 7 / 24 := by
  sorry

end printer_diff_l1922_192293


namespace omega_range_l1922_192248

theorem omega_range (ω : ℝ) (a b : ℝ) (hω_pos : ω > 0) (h_range : π ≤ a ∧ a < b ∧ b ≤ 2 * π)
  (h_sin : Real.sin (ω * a) + Real.sin (ω * b) = 2) :
  ω ∈ Set.Icc (9 / 4 : ℝ) (5 / 2) ∪ Set.Ici (13 / 4) :=
by
  sorry

end omega_range_l1922_192248


namespace exists_subset_no_double_l1922_192267

theorem exists_subset_no_double (s : Finset ℕ) (h₁ : s = Finset.range 3000) :
  ∃ t : Finset ℕ, t.card = 2000 ∧ (∀ x ∈ t, ∀ y ∈ t, x ≠ 2 * y ∧ y ≠ 2 * x) :=
by
  sorry

end exists_subset_no_double_l1922_192267


namespace minimum_people_in_troupe_l1922_192245

-- Let n be the number of people in the troupe.
variable (n : ℕ)

-- Conditions: n must be divisible by 8, 10, and 12.
def is_divisible_by (m k : ℕ) := m % k = 0
def divides_all (n : ℕ) := is_divisible_by n 8 ∧ is_divisible_by n 10 ∧ is_divisible_by n 12

-- The minimum number of people in the troupe that can form groups of 8, 10, or 12 with none left over.
theorem minimum_people_in_troupe (n : ℕ) : divides_all n → n = 120 :=
by
  sorry

end minimum_people_in_troupe_l1922_192245


namespace trains_cross_time_l1922_192255

theorem trains_cross_time 
  (len_train1 len_train2 : ℕ) 
  (speed_train1_kmph speed_train2_kmph : ℕ) 
  (len_train1_eq : len_train1 = 200) 
  (len_train2_eq : len_train2 = 300) 
  (speed_train1_eq : speed_train1_kmph = 70) 
  (speed_train2_eq : speed_train2_kmph = 50) 
  : (500 / (120 * 1000 / 3600)) = 15 := 
by sorry

end trains_cross_time_l1922_192255


namespace monks_mantou_l1922_192238

theorem monks_mantou (x y : ℕ) (h1 : x + y = 100) (h2 : 3 * x + y / 3 = 100) :
  (3 * x + (100 - x) / 3 = 100) ∧ (x + y = 100 ∧ 3 * x + y / 3 = 100) :=
by
  sorry

end monks_mantou_l1922_192238


namespace max_value_of_exp_sum_l1922_192221

theorem max_value_of_exp_sum (a b : ℝ) (h_a_pos : 0 < a) (h_b_pos : 0 < b) (h_ab_pos : 0 < a * b) :
    ∃ θ : ℝ, a * Real.exp θ + b * Real.exp (-θ) = 2 * Real.sqrt (a * b) :=
by
  sorry

end max_value_of_exp_sum_l1922_192221


namespace avg_percentage_decrease_l1922_192222

theorem avg_percentage_decrease (x : ℝ) 
  (h : 16 * (1 - x)^2 = 9) : x = 0.25 :=
sorry

end avg_percentage_decrease_l1922_192222


namespace ticket_price_l1922_192264

variable (x : ℝ)

def tickets_condition1 := 3 * x
def tickets_condition2 := 5 * x
def total_spent := 3 * x + 5 * x

theorem ticket_price : total_spent x = 32 → x = 4 :=
by
  -- Proof steps will be provided here.
  sorry

end ticket_price_l1922_192264


namespace opposite_of_neg_2023_l1922_192256

theorem opposite_of_neg_2023 : -(-2023) = 2023 := 
by 
  sorry

end opposite_of_neg_2023_l1922_192256


namespace larger_integer_is_50_l1922_192201

-- Definition of the problem conditions.
def is_two_digit (x : ℕ) : Prop := 10 ≤ x ∧ x ≤ 99

def problem_conditions (m n : ℕ) : Prop := 
  is_two_digit m ∧ is_two_digit n ∧
  (m + n) / 2 = m + n / 100

-- Statement of the proof problem.
theorem larger_integer_is_50 (m n : ℕ) (h : problem_conditions m n) : max m n = 50 :=
  sorry

end larger_integer_is_50_l1922_192201


namespace probability_snow_at_least_once_l1922_192227

-- Define the probabilities given in the conditions
def p_day_1_3 : ℚ := 1 / 3
def p_day_4_7 : ℚ := 1 / 4
def p_day_8_10 : ℚ := 1 / 2

-- Define the complementary no-snow probabilities
def p_no_snow_day_1_3 : ℚ := 2 / 3
def p_no_snow_day_4_7 : ℚ := 3 / 4
def p_no_snow_day_8_10 : ℚ := 1 / 2

-- Compute the total probability of no snow for all ten days
def p_no_snow_all_days : ℚ :=
  (p_no_snow_day_1_3 ^ 3) * (p_no_snow_day_4_7 ^ 4) * (p_no_snow_day_8_10 ^ 3)

-- Define the proof problem: Calculate probability of at least one snow day
theorem probability_snow_at_least_once : (1 - p_no_snow_all_days) = 2277 / 2304 := by
  sorry

end probability_snow_at_least_once_l1922_192227


namespace proof_problem_l1922_192251

variable (α β : ℝ) (a b : ℝ × ℝ) (m : ℝ)
variable (hα : 0 < α ∧ α < Real.pi / 4)
variable (hβ : β = Real.pi)
variable (ha_def : a = (Real.tan (α + β / 4) - 1, 0))
variable (hb : b = (Real.cos α, 2))
variable (ha_dot : a.1 * b.1 + a.2 * b.2 = m)

-- Proof statement
theorem proof_problem :
  (0 < α ∧ α < Real.pi / 4) ∧
  β = Real.pi ∧
  a = (Real.tan (α + β / 4) - 1, 0) ∧
  b = (Real.cos α, 2) ∧
  (a.1 * b.1 + a.2 * b.2 = m) →
  (2 * Real.cos α * Real.cos α + Real.sin (2 * (α + β))) / (Real.cos α - Real.sin β) = 2 * (m + 2) := by
  sorry

end proof_problem_l1922_192251


namespace solve_inequality_l1922_192282

theorem solve_inequality (x : ℝ) : 3 * (x + 1) > 9 → x > 2 :=
by sorry

end solve_inequality_l1922_192282


namespace minimum_participants_l1922_192239

theorem minimum_participants 
  (x y z : ℕ)
  (h_andrei : 3 * x + 1 = 61)
  (h_dima : 4 * y + 1 = 61)
  (h_lenya : 5 * z + 1 = 61) : 
  x = 20 ∧ y = 15 ∧ z = 12 :=
by
  sorry

end minimum_participants_l1922_192239


namespace distance_rowed_downstream_l1922_192215

-- Define the conditions
def speed_in_still_water (b s: ℝ) := b - s = 60 / 4
def speed_of_stream (s: ℝ) := s = 3
def time_downstream (t: ℝ) := t = 4

-- Define the function that computes the downstream speed
def downstream_speed (b s t: ℝ) := (b + s) * t

-- The theorem we want to prove
theorem distance_rowed_downstream (b s t : ℝ) 
    (h1 : speed_in_still_water b s)
    (h2 : speed_of_stream s)
    (h3 : time_downstream t) : 
    downstream_speed b s t = 84 := by
    sorry

end distance_rowed_downstream_l1922_192215


namespace tony_solving_puzzles_time_l1922_192206

theorem tony_solving_puzzles_time : ∀ (warm_up_time long_puzzle_ratio num_long_puzzles : ℕ),
  warm_up_time = 10 →
  long_puzzle_ratio = 3 →
  num_long_puzzles = 2 →
  (warm_up_time + long_puzzle_ratio * warm_up_time * num_long_puzzles) = 70 :=
by
  intros
  sorry

end tony_solving_puzzles_time_l1922_192206


namespace compute_expression_l1922_192200

theorem compute_expression :
    (3 + 5)^2 + (3^2 + 5^2 + 3 * 5) = 113 := 
by sorry

end compute_expression_l1922_192200


namespace anne_speed_ratio_l1922_192202

variable (B A A' : ℝ) (hours_to_clean_together : ℝ) (hours_to_clean_with_new_anne : ℝ)

-- Conditions
def cleaning_condition_1 := (A + B) * 4 = 1 -- Combined rate for 4 hours
def cleaning_condition_2 := A = 1 / 12      -- Anne's rate alone
def cleaning_condition_3 := (A' + B) * 3 = 1 -- Combined rate for 3 hours with new Anne's rate

-- Theorem to Prove
theorem anne_speed_ratio (h1 : cleaning_condition_1 B A)
                         (h2 : cleaning_condition_2 A)
                         (h3 : cleaning_condition_3 B A') :
                         (A' / A) = 2 :=
by sorry

end anne_speed_ratio_l1922_192202


namespace expression_evaluation_l1922_192233

theorem expression_evaluation : (50 + 12) ^ 2 - (12 ^ 2 + 50 ^ 2) = 1200 := 
by
  sorry

end expression_evaluation_l1922_192233


namespace g_at_5_l1922_192261

-- Define the function g(x) that satisfies the given condition
def g (x : ℝ) : ℝ := sorry

-- Axiom stating that the function g satisfies the given equation for all x ∈ ℝ
axiom g_condition : ∀ x : ℝ, 3 * g x + 4 * g (1 - x) = 6 * x^2

-- The theorem to prove
theorem g_at_5 : g 5 = -66 / 7 :=
by
  -- Proof will be added here.
  sorry

end g_at_5_l1922_192261


namespace two_pow_2023_mod_17_l1922_192295

theorem two_pow_2023_mod_17 : (2 ^ 2023) % 17 = 4 := 
by
  sorry

end two_pow_2023_mod_17_l1922_192295


namespace polynomial_proof_l1922_192287

theorem polynomial_proof (x : ℝ) : 
  (2 * x^2 + 5 * x + 4) = (2 * x^2 + 5 * x - 2) + (10 * x + 6) :=
by sorry

end polynomial_proof_l1922_192287


namespace solve_for_a_l1922_192204

theorem solve_for_a (a : ℤ) :
  (|2 * a + 1| = 3) ↔ (a = 1 ∨ a = -2) :=
by
  sorry

end solve_for_a_l1922_192204


namespace problem_part1_problem_part2_l1922_192243

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (x + 1) / Real.log a
noncomputable def g (a : ℝ) (t : ℝ) (x : ℝ) : ℝ := 2 * Real.log (2 * x + t) / Real.log a

theorem problem_part1 (a t : ℝ) (ha_pos : a > 0) (ha_ne_one : a ≠ 1) :
  f a 1 - g a t 1 = 0 → t = -2 + Real.sqrt 2 :=
sorry

theorem problem_part2 (a t : ℝ) (ha_bound : 0 < a ∧ a < 1) :
  (∀ x, 0 ≤ x ∧ x ≤ 15 → f a x ≥ g a t x) → t ≥ 1 :=
sorry

end problem_part1_problem_part2_l1922_192243


namespace repeating_decimal_to_fraction_l1922_192269

theorem repeating_decimal_to_fraction : (∃ (x : ℚ), x = 0.4 + 4 / 9) :=
sorry

end repeating_decimal_to_fraction_l1922_192269
