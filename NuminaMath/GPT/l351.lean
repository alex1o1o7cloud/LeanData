import Mathlib

namespace NUMINAMATH_GPT_moles_of_water_formed_l351_35141

-- Definitions
def moles_of_H2SO4 : Nat := 3
def moles_of_NaOH : Nat := 3
def moles_of_NaHSO4 : Nat := 3
def moles_of_H2O := moles_of_NaHSO4

-- Theorem
theorem moles_of_water_formed :
  moles_of_H2SO4 = 3 →
  moles_of_NaOH = 3 →
  moles_of_NaHSO4 = 3 →
  moles_of_H2O = 3 :=
by
  intros h1 h2 h3
  rw [moles_of_H2O]
  exact h3

end NUMINAMATH_GPT_moles_of_water_formed_l351_35141


namespace NUMINAMATH_GPT_inequality_solution_l351_35126

theorem inequality_solution (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (hxy : x + y = 4) : 
  (1 / x + 4 / y) ≥ 9 / 4 := 
sorry

end NUMINAMATH_GPT_inequality_solution_l351_35126


namespace NUMINAMATH_GPT_smallest_gcd_for_system_l351_35124

theorem smallest_gcd_for_system :
  ∃ n : ℕ, n > 0 ∧ 
    (∀ a b c : ℤ,
     gcd (gcd a b) c = n →
     ∃ x y z : ℤ, 
       (x + 2*y + 3*z = a) ∧ 
       (2*x + y - 2*z = b) ∧ 
       (3*x + y + 5*z = c)) ∧ 
  n = 28 :=
sorry

end NUMINAMATH_GPT_smallest_gcd_for_system_l351_35124


namespace NUMINAMATH_GPT_pumpkins_total_weight_l351_35169

-- Define the weights of the pumpkins as given in the conditions
def first_pumpkin_weight : ℝ := 4
def second_pumpkin_weight : ℝ := 8.7

-- Prove that the total weight of the two pumpkins is 12.7 pounds
theorem pumpkins_total_weight : first_pumpkin_weight + second_pumpkin_weight = 12.7 := by
  sorry

end NUMINAMATH_GPT_pumpkins_total_weight_l351_35169


namespace NUMINAMATH_GPT_solve_system_of_equations_l351_35174

theorem solve_system_of_equations :
  ∀ (x y z : ℚ), 
    (x * y = x + 2 * y ∧
     y * z = y + 3 * z ∧
     z * x = z + 4 * x) ↔
    (x = 0 ∧ y = 0 ∧ z = 0) ∨
    (x = 25 / 9 ∧ y = 25 / 7 ∧ z = 25 / 4) := by
  sorry

end NUMINAMATH_GPT_solve_system_of_equations_l351_35174


namespace NUMINAMATH_GPT_find_initial_pomelos_l351_35178

theorem find_initial_pomelos (g w w' g' : ℕ) 
  (h1 : w = 3 * g)
  (h2 : w' = w - 90)
  (h3 : g' = g - 60)
  (h4 : w' = 4 * g' - 26) 
  : g = 176 :=
by
  sorry

end NUMINAMATH_GPT_find_initial_pomelos_l351_35178


namespace NUMINAMATH_GPT_solution_set_range_l351_35117

theorem solution_set_range (x : ℝ) : 
  (2 * |x - 10| + 3 * |x - 20| ≤ 35) ↔ (9 ≤ x ∧ x ≤ 23) :=
sorry

end NUMINAMATH_GPT_solution_set_range_l351_35117


namespace NUMINAMATH_GPT_product_of_large_integers_l351_35167

theorem product_of_large_integers :
  ∃ A B : ℤ, A > 10^2009 ∧ B > 10^2009 ∧ A * B = 3^(4^5) + 4^(5^6) :=
by
  sorry

end NUMINAMATH_GPT_product_of_large_integers_l351_35167


namespace NUMINAMATH_GPT_nina_age_l351_35116

theorem nina_age : ∀ (M L A N : ℕ), 
  (M = L - 5) → 
  (L = A + 6) → 
  (N = A + 2) → 
  (M = 16) → 
  N = 17 :=
by
  intros M L A N h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_nina_age_l351_35116


namespace NUMINAMATH_GPT_food_drive_total_cans_l351_35175

def total_cans_brought (M J R : ℕ) : ℕ := M + J + R

theorem food_drive_total_cans (M J R : ℕ) 
  (h1 : M = 4 * J) 
  (h2 : J = 2 * R + 5) 
  (h3 : M = 100) : 
  total_cans_brought M J R = 135 :=
by sorry

end NUMINAMATH_GPT_food_drive_total_cans_l351_35175


namespace NUMINAMATH_GPT_fraction_product_l351_35127

theorem fraction_product : (1 / 2) * (1 / 3) * (1 / 6) * 120 = 10 / 3 :=
by
  sorry

end NUMINAMATH_GPT_fraction_product_l351_35127


namespace NUMINAMATH_GPT_brownie_pieces_count_l351_35173

def area_of_pan (length width : ℕ) : ℕ := length * width

def area_of_piece (side : ℕ) : ℕ := side * side

def number_of_pieces (pan_area piece_area : ℕ) : ℕ := pan_area / piece_area

theorem brownie_pieces_count :
  let pan_length := 24
  let pan_width := 15
  let piece_side := 3
  let pan_area := area_of_pan pan_length pan_width
  let piece_area := area_of_piece piece_side
  number_of_pieces pan_area piece_area = 40 :=
by
  sorry

end NUMINAMATH_GPT_brownie_pieces_count_l351_35173


namespace NUMINAMATH_GPT_sum_first_five_terms_l351_35114

theorem sum_first_five_terms (a1 a2 a3 : ℝ) (S5 : ℝ) 
  (h1 : a1 * a3 = 8 * a2)
  (h2 : (a1 + a2) = 24) :
  S5 = 31 :=
sorry

end NUMINAMATH_GPT_sum_first_five_terms_l351_35114


namespace NUMINAMATH_GPT_solution_system_linear_eqns_l351_35123

theorem solution_system_linear_eqns
    (a1 b1 c1 a2 b2 c2 : ℝ)
    (h1: a1 * 6 + b1 * 3 = c1)
    (h2: a2 * 6 + b2 * 3 = c2) :
    (4 * a1 * 22 + 3 * b1 * 33 = 11 * c1) ∧
    (4 * a2 * 22 + 3 * b2 * 33 = 11 * c2) :=
by
    sorry

end NUMINAMATH_GPT_solution_system_linear_eqns_l351_35123


namespace NUMINAMATH_GPT_triangle_inequality_l351_35183

variable {x y z : ℝ}
variable {A B C : ℝ}

theorem triangle_inequality (hA: A > 0) (hB : B > 0) (hC : C > 0) (h_sum : A + B + C = π):
  x^2 + y^2 + z^2 ≥ 2 * y * z * Real.sin A + 2 * z * x * Real.sin B - 2 * x * y * Real.cos C := by
  sorry

end NUMINAMATH_GPT_triangle_inequality_l351_35183


namespace NUMINAMATH_GPT_total_number_of_flowers_l351_35190

theorem total_number_of_flowers : 
  let red_roses := 1491
  let yellow_carnations := 3025
  let white_roses := 1768
  let purple_tulips := 2150
  let pink_daisies := 3500
  let blue_irises := 2973
  let orange_marigolds := 4234
  red_roses + yellow_carnations + white_roses + purple_tulips + pink_daisies + blue_irises + orange_marigolds = 19141 :=
by 
  sorry

end NUMINAMATH_GPT_total_number_of_flowers_l351_35190


namespace NUMINAMATH_GPT_votes_diff_eq_70_l351_35153

noncomputable def T : ℝ := 350
def votes_against (T : ℝ) : ℝ := 0.40 * T
def votes_favor (T : ℝ) (X : ℝ) : ℝ := votes_against T + X

theorem votes_diff_eq_70 :
  ∃ X : ℝ, 350 = votes_against T + votes_favor T X → X = 70 :=
by
  sorry

end NUMINAMATH_GPT_votes_diff_eq_70_l351_35153


namespace NUMINAMATH_GPT_combinatorial_proof_l351_35185

noncomputable def combinatorial_identity (n m k : ℕ) (h1 : 1 ≤ k) (h2 : k < m) (h3 : m < n) : ℕ :=
  let summation_term (i : ℕ) := Nat.choose k i * Nat.choose n (m - i)
  List.sum (List.map summation_term (List.range (k + 1)))

theorem combinatorial_proof (n m k : ℕ) (h1 : 1 ≤ k) (h2 : k < m) (h3 : m < n) :
  combinatorial_identity n m k h1 h2 h3 = Nat.choose (n + k) m :=
sorry

end NUMINAMATH_GPT_combinatorial_proof_l351_35185


namespace NUMINAMATH_GPT_nina_basketball_cards_l351_35197

theorem nina_basketball_cards (cost_toy cost_shirt cost_card total_spent : ℕ) (n_toys n_shirts n_cards n_packs_result : ℕ)
  (h1 : cost_toy = 10)
  (h2 : cost_shirt = 6)
  (h3 : cost_card = 5)
  (h4 : n_toys = 3)
  (h5 : n_shirts = 5)
  (h6 : total_spent = 70)
  (h7 : n_packs_result =  2)
  : (3 * cost_toy + 5 * cost_shirt + n_cards * cost_card = total_spent) → n_cards = n_packs_result :=
by
  sorry

end NUMINAMATH_GPT_nina_basketball_cards_l351_35197


namespace NUMINAMATH_GPT_john_initial_investment_in_alpha_bank_is_correct_l351_35100

-- Definition of the problem conditions
def initial_investment : ℝ := 2000
def alpha_rate : ℝ := 0.04
def beta_rate : ℝ := 0.06
def final_amount : ℝ := 2398.32
def years : ℕ := 3

-- Alpha Bank growth factor after 3 years
def alpha_growth_factor : ℝ := (1 + alpha_rate) ^ years

-- Beta Bank growth factor after 3 years
def beta_growth_factor : ℝ := (1 + beta_rate) ^ years

-- The main theorem
theorem john_initial_investment_in_alpha_bank_is_correct (x : ℝ) 
  (hx : x * alpha_growth_factor + (initial_investment - x) * beta_growth_factor = final_amount) : 
  x = 246.22 :=
sorry

end NUMINAMATH_GPT_john_initial_investment_in_alpha_bank_is_correct_l351_35100


namespace NUMINAMATH_GPT_equal_sets_implies_value_of_m_l351_35107

theorem equal_sets_implies_value_of_m (m : ℝ) (A B : Set ℝ) (hA : A = {3, m}) (hB : B = {3 * m, 3}) (hAB : A = B) : m = 0 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_equal_sets_implies_value_of_m_l351_35107


namespace NUMINAMATH_GPT_find_a_and_b_solve_inequality_l351_35150

def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := x^2 + a * x + b

theorem find_a_and_b (a b : ℝ) (h : ∀ x : ℝ, f x a b > 0 ↔ x < 0 ∨ x > 2) : a = -2 ∧ b = 0 :=
by sorry

theorem solve_inequality (a b : ℝ) (m : ℝ) (h1 : a = -2) (h2 : b = 0) :
  (∀ x : ℝ, f x a b < m^2 - 1 ↔ 
    (m = 0 → ∀ x : ℝ, false) ∧
    (m > 0 → (1 - m < x ∧ x < 1 + m)) ∧
    (m < 0 → (1 + m < x ∧ x < 1 - m))) :=
by sorry

end NUMINAMATH_GPT_find_a_and_b_solve_inequality_l351_35150


namespace NUMINAMATH_GPT_car_R_speed_l351_35118

theorem car_R_speed (v : ℝ) (h1 : ∀ t_R t_P : ℝ, t_R * v = 800 ∧ t_P * (v + 10) = 800) (h2 : ∀ t_R t_P : ℝ, t_P + 2 = t_R) :
  v = 50 := by
  sorry

end NUMINAMATH_GPT_car_R_speed_l351_35118


namespace NUMINAMATH_GPT_ratio_garbage_zane_dewei_l351_35132

-- Define the weights of garbage picked up by Daliah, Dewei, and Zane.
def daliah_garbage : ℝ := 17.5
def dewei_garbage : ℝ := daliah_garbage - 2
def zane_garbage : ℝ := 62

-- The theorem that we need to prove
theorem ratio_garbage_zane_dewei : zane_garbage / dewei_garbage = 4 :=
by
  sorry

end NUMINAMATH_GPT_ratio_garbage_zane_dewei_l351_35132


namespace NUMINAMATH_GPT_BB_digit_value_in_5BB3_l351_35111

theorem BB_digit_value_in_5BB3 (B : ℕ) (h : 2 * B + 8 % 9 = 0) : B = 5 :=
sorry

end NUMINAMATH_GPT_BB_digit_value_in_5BB3_l351_35111


namespace NUMINAMATH_GPT_fill_pool_time_l351_35138

-- Define the conditions
def pool_volume : ℕ := 15000
def hoses1_rate : ℕ := 2
def hoses1_count : ℕ := 2
def hoses2_rate : ℕ := 3
def hoses2_count : ℕ := 2

-- Calculate the total delivery rate
def total_delivery_rate : ℕ :=
  (hoses1_rate * hoses1_count) + (hoses2_rate * hoses2_count)

-- Calculate the time to fill the pool in minutes
def time_to_fill_in_minutes : ℕ :=
  pool_volume / total_delivery_rate

-- Calculate the time to fill the pool in hours
def time_to_fill_in_hours : ℕ :=
  time_to_fill_in_minutes / 60

-- The theorem to prove
theorem fill_pool_time : time_to_fill_in_hours = 25 := by
  sorry

end NUMINAMATH_GPT_fill_pool_time_l351_35138


namespace NUMINAMATH_GPT_simplified_expression_value_l351_35180

theorem simplified_expression_value (a b : ℝ) (ha : a = -1) (hb : b = 1 / 4) :
  (a + 2 * b) ^ 2 + (a + 2 * b) * (a - 2 * b) = 1 := 
by
  sorry

end NUMINAMATH_GPT_simplified_expression_value_l351_35180


namespace NUMINAMATH_GPT_total_miles_traveled_l351_35113

noncomputable def distance_to_first_museum : ℕ := 5
noncomputable def distance_to_second_museum : ℕ := 15
noncomputable def distance_to_cultural_center : ℕ := 10
noncomputable def extra_detour : ℕ := 3

theorem total_miles_traveled : 
  (2 * (distance_to_first_museum + extra_detour) + 2 * distance_to_second_museum + 2 * distance_to_cultural_center) = 66 :=
  by
  sorry

end NUMINAMATH_GPT_total_miles_traveled_l351_35113


namespace NUMINAMATH_GPT_rank_best_buy_LMS_l351_35139

theorem rank_best_buy_LMS (c_S q_S : ℝ) :
  let c_M := 1.75 * c_S
  let q_M := 1.1 * q_S
  let c_L := 1.25 * c_M
  let q_L := 1.5 * q_M
  (c_S / q_S) > (c_M / q_M) ∧ (c_M / q_M) > (c_L / q_L) :=
by
  sorry

end NUMINAMATH_GPT_rank_best_buy_LMS_l351_35139


namespace NUMINAMATH_GPT_lloyd_total_hours_worked_l351_35154

noncomputable def total_hours_worked (daily_hours : ℝ) (regular_rate : ℝ) (overtime_multiplier: ℝ) (total_earnings : ℝ) : ℝ :=
  let regular_hours := 7.5
  let regular_pay := regular_hours * regular_rate
  if total_earnings <= regular_pay then daily_hours else
  let overtime_pay := total_earnings - regular_pay
  let overtime_hours := overtime_pay / (regular_rate * overtime_multiplier)
  regular_hours + overtime_hours

theorem lloyd_total_hours_worked :
  total_hours_worked 7.5 5.50 1.5 66 = 10.5 :=
by
  sorry

end NUMINAMATH_GPT_lloyd_total_hours_worked_l351_35154


namespace NUMINAMATH_GPT_sin_cos_eq_l351_35151

theorem sin_cos_eq (α : ℝ) (h : Real.tan α = 2) : Real.sin α * Real.cos α = 2 / 5 := sorry

end NUMINAMATH_GPT_sin_cos_eq_l351_35151


namespace NUMINAMATH_GPT_sequence_term_position_l351_35115

theorem sequence_term_position :
  ∃ n : ℕ, ∀ k : ℕ, (k = 7 + 6 * (n - 1)) → k = 2005 → n = 334 :=
by
  sorry

end NUMINAMATH_GPT_sequence_term_position_l351_35115


namespace NUMINAMATH_GPT_ann_has_30_more_cards_than_anton_l351_35163

theorem ann_has_30_more_cards_than_anton (heike_cards : ℕ) (anton_cards : ℕ) (ann_cards : ℕ) 
  (h1 : anton_cards = 3 * heike_cards)
  (h2 : ann_cards = 6 * heike_cards)
  (h3 : ann_cards = 60) : ann_cards - anton_cards = 30 :=
by
  sorry

end NUMINAMATH_GPT_ann_has_30_more_cards_than_anton_l351_35163


namespace NUMINAMATH_GPT_symmetric_circle_l351_35143

-- Define given circle equation
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 4 * x - 8 * y + 12 = 0

-- Define the line of symmetry
def line_equation (x y : ℝ) : Prop :=
  x + 2 * y - 5 = 0

-- Define the symmetric circle equation we need to prove
def symm_circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 = 8

-- Lean 4 theorem statement
theorem symmetric_circle (x y : ℝ) :
  (∃ a b : ℝ, circle_equation 2 4 ∧ line_equation a b ∧ (a, b) = (0, 0)) →
  symm_circle_equation x y :=
by sorry

end NUMINAMATH_GPT_symmetric_circle_l351_35143


namespace NUMINAMATH_GPT_inequality_abc_l351_35157

variable (a b c : ℝ)
variable (ha : a > 0)
variable (hb : b > 0)
variable (hc : c > 0)
variable (cond : a + b + c = (1 / a) + (1 / b) + (1 / c))

theorem inequality_abc : a + b + c ≥ 3 / (a * b * c) :=
sorry

end NUMINAMATH_GPT_inequality_abc_l351_35157


namespace NUMINAMATH_GPT_point_in_fourth_quadrant_l351_35161

def inFourthQuadrant (x y : Int) : Prop :=
  x > 0 ∧ y < 0

theorem point_in_fourth_quadrant :
  inFourthQuadrant 2 (-3) :=
by
  sorry

end NUMINAMATH_GPT_point_in_fourth_quadrant_l351_35161


namespace NUMINAMATH_GPT_rowing_distance_l351_35189

theorem rowing_distance (D : ℝ) 
  (boat_speed : ℝ) 
  (stream_speed : ℝ) 
  (total_time : ℝ) 
  (downstream_speed : ℝ := boat_speed + stream_speed) 
  (upstream_speed : ℝ := boat_speed - stream_speed)
  (downstream_time : ℝ := D / downstream_speed)
  (upstream_time : ℝ := D / upstream_speed)
  (round_trip_time : ℝ := downstream_time + upstream_time) 
  (h1 : boat_speed = 16) 
  (h2 : stream_speed = 2) 
  (h3 : total_time = 914.2857142857143)
  (h4 : round_trip_time = total_time) :
  D = 720 :=
by sorry

end NUMINAMATH_GPT_rowing_distance_l351_35189


namespace NUMINAMATH_GPT_minimum_value_7a_4b_l351_35145

noncomputable def original_cond (a b : ℝ) (ha : 0 < a) (hb : 0 < b) : Prop :=
  (2 / (3 * a + b)) + (1 / (a + 2 * b)) = 4

theorem minimum_value_7a_4b (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  original_cond a b ha hb → 7 * a + 4 * b = 9 / 4 :=
by
  sorry

end NUMINAMATH_GPT_minimum_value_7a_4b_l351_35145


namespace NUMINAMATH_GPT_water_force_on_dam_l351_35177

-- Given conditions
def density : Real := 1000  -- kg/m^3
def gravity : Real := 10    -- m/s^2
def a : Real := 5.7         -- m
def b : Real := 9.0         -- m
def h : Real := 4.0         -- m

-- Prove that the force is 544000 N under the given conditions
theorem water_force_on_dam : ∃ (F : Real), F = 544000 :=
by
  sorry  -- proof goes here

end NUMINAMATH_GPT_water_force_on_dam_l351_35177


namespace NUMINAMATH_GPT_sequence_term_geometric_l351_35182

theorem sequence_term_geometric :
  ∀ (a : ℕ → ℕ), 
    a 1 = 1 →
    (∀ n, n ≥ 2 → (a n) / (a (n - 1)) = 2^(n-1)) →
    a 101 = 2^5050 :=
by
  sorry

end NUMINAMATH_GPT_sequence_term_geometric_l351_35182


namespace NUMINAMATH_GPT_mrs_smith_strawberries_l351_35120

theorem mrs_smith_strawberries (girls : ℕ) (strawberries_per_girl : ℕ) 
                                (h1 : girls = 8) (h2 : strawberries_per_girl = 6) :
    girls * strawberries_per_girl = 48 := by
  sorry

end NUMINAMATH_GPT_mrs_smith_strawberries_l351_35120


namespace NUMINAMATH_GPT_lizette_third_quiz_score_l351_35166

theorem lizette_third_quiz_score :
  ∀ (x : ℕ),
  (2 * 95 + x) / 3 = 94 → x = 92 :=
by
  intro x h
  have h1 : 2 * 95 = 190 := by norm_num
  have h2 : 3 * 94 = 282 := by norm_num
  sorry

end NUMINAMATH_GPT_lizette_third_quiz_score_l351_35166


namespace NUMINAMATH_GPT_greatest_possible_sum_of_10_integers_l351_35164

theorem greatest_possible_sum_of_10_integers (a b c d e f g h i j : ℕ) 
  (h_prod : a * b * c * d * e * f * g * h * i * j = 1024) : 
  a + b + c + d + e + f + g + h + i + j ≤ 1033 :=
sorry

end NUMINAMATH_GPT_greatest_possible_sum_of_10_integers_l351_35164


namespace NUMINAMATH_GPT_oscar_leap_longer_l351_35194

noncomputable def elmer_strides (poles : ℕ) (strides_per_gap : ℕ) (distance_miles : ℝ) : ℝ :=
  let total_distance := distance_miles * 5280  -- convert miles to feet
  let total_strides := (poles - 1) * strides_per_gap
  total_distance / total_strides

noncomputable def oscar_leaps (poles : ℕ) (leaps_per_gap : ℕ) (distance_miles : ℝ) : ℝ :=
  let total_distance := distance_miles * 5280  -- convert miles to feet
  let total_leaps := (poles - 1) * leaps_per_gap
  total_distance / total_leaps

theorem oscar_leap_longer (poles : ℕ) (strides_per_gap leaps_per_gap : ℕ) (distance_miles : ℝ) :
  poles = 51 -> strides_per_gap = 50 -> leaps_per_gap = 15 -> distance_miles = 1.25 ->
  let elmer_stride := elmer_strides poles strides_per_gap distance_miles
  let oscar_leap := oscar_leaps poles leaps_per_gap distance_miles
  (oscar_leap - elmer_stride) * 12 = 74 :=
by
  intros h_poles h_strides h_leaps h_distance
  have elmer_stride := elmer_strides poles strides_per_gap distance_miles
  have oscar_leap := oscar_leaps poles leaps_per_gap distance_miles
  sorry

end NUMINAMATH_GPT_oscar_leap_longer_l351_35194


namespace NUMINAMATH_GPT_sallys_change_l351_35195

-- Define the total cost calculation:
def totalCost (numFrames : Nat) (costPerFrame : Nat) : Nat :=
  numFrames * costPerFrame

-- Define the change calculation:
def change (totalAmount : Nat) (amountPaid : Nat) : Nat :=
  amountPaid - totalAmount

-- Define the specific conditions in the problem:
def numFrames := 3
def costPerFrame := 3
def amountPaid := 20

-- Prove that the change Sally gets is $11:
theorem sallys_change : change (totalCost numFrames costPerFrame) amountPaid = 11 := by
  sorry

end NUMINAMATH_GPT_sallys_change_l351_35195


namespace NUMINAMATH_GPT_combined_salary_ABC_and_E_l351_35125

def salary_D : ℕ := 7000
def avg_salary : ℕ := 9000
def num_individuals : ℕ := 5

theorem combined_salary_ABC_and_E :
  (avg_salary * num_individuals - salary_D) = 38000 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_combined_salary_ABC_and_E_l351_35125


namespace NUMINAMATH_GPT_find_a_l351_35155

noncomputable def f (a x : ℝ) : ℝ := a * x * (x - 2)^2

theorem find_a (a : ℝ) (h1 : a ≠ 0)
  (h2 : ∃ x : ℝ, f a x = 32) :
  a = 27 :=
sorry

end NUMINAMATH_GPT_find_a_l351_35155


namespace NUMINAMATH_GPT_time_comparison_l351_35158

noncomputable def pedestrian_speed : Real := 6.5
noncomputable def cyclist_speed : Real := 20.0
noncomputable def distance_between_points_B_A : Real := 4 * Real.pi - 6.5
noncomputable def alley_distance : Real := 4 * Real.pi - 6.5
noncomputable def combined_speed_3 : Real := pedestrian_speed + cyclist_speed
noncomputable def combined_speed_2 : Real := 21.5
noncomputable def time_scenario_3 : Real := (4 * Real.pi - 6.5) / combined_speed_3
noncomputable def time_scenario_2 : Real := (10.5 - 2 * Real.pi) / combined_speed_2

theorem time_comparison : time_scenario_2 < time_scenario_3 :=
by
  sorry

end NUMINAMATH_GPT_time_comparison_l351_35158


namespace NUMINAMATH_GPT_product_of_random_numbers_greater_zero_l351_35198

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

end NUMINAMATH_GPT_product_of_random_numbers_greater_zero_l351_35198


namespace NUMINAMATH_GPT_tens_digit_of_13_pow_3007_l351_35147

theorem tens_digit_of_13_pow_3007 : 
  (13 ^ 3007 / 10) % 10 = 1 :=
sorry

end NUMINAMATH_GPT_tens_digit_of_13_pow_3007_l351_35147


namespace NUMINAMATH_GPT_solve_lambda_l351_35165

variable (a b : ℝ × ℝ)
variable (lambda : ℝ)

def perpendicular (a b : ℝ × ℝ) : Prop :=
  a.1 * b.1 + a.2 * b.2 = 0

axiom a_def : a = (-3, 2)
axiom b_def : b = (-1, 0)
axiom perp_def : perpendicular (a.1 + lambda * b.1, a.2 + lambda * b.2) b

theorem solve_lambda : lambda = -3 :=
by
  sorry

end NUMINAMATH_GPT_solve_lambda_l351_35165


namespace NUMINAMATH_GPT_average_price_per_person_excluding_gratuity_l351_35108

def total_cost_with_gratuity : ℝ := 207.00
def gratuity_rate : ℝ := 0.15
def number_of_people : ℕ := 15

theorem average_price_per_person_excluding_gratuity :
  (total_cost_with_gratuity / (1 + gratuity_rate) / number_of_people) = 12.00 :=
by
  sorry

end NUMINAMATH_GPT_average_price_per_person_excluding_gratuity_l351_35108


namespace NUMINAMATH_GPT_ramu_profit_percent_l351_35184

noncomputable def profit_percent (purchase_price repair_cost selling_price : ℝ) : ℝ :=
  let total_cost := purchase_price + repair_cost
  let profit := selling_price - total_cost
  (profit / total_cost) * 100

theorem ramu_profit_percent :
  profit_percent 42000 13000 64500 = 17.27 :=
by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_ramu_profit_percent_l351_35184


namespace NUMINAMATH_GPT_problem_solution_l351_35142

theorem problem_solution (k : ℕ) (hk : k ≥ 2) : 
  (∀ m n : ℕ, 1 ≤ m ∧ m ≤ k → 1 ≤ n ∧ n ≤ k → m ≠ n → ¬ k ∣ (n^(n-1) - m^(m-1))) ↔ (k = 2 ∨ k = 3) :=
by
  sorry

end NUMINAMATH_GPT_problem_solution_l351_35142


namespace NUMINAMATH_GPT_odometer_problem_l351_35140

theorem odometer_problem :
  ∃ (a b c : ℕ), 1 ≤ a ∧ a + b + c ≤ 10 ∧ (11 * c - 10 * a - b) % 6 = 0 ∧ a^2 + b^2 + c^2 = 54 :=
by
  sorry

end NUMINAMATH_GPT_odometer_problem_l351_35140


namespace NUMINAMATH_GPT_x_minus_y_div_x_eq_4_7_l351_35146

-- Definitions based on the problem's conditions
axiom y_div_x_eq_3_7 (x y : ℝ) : y / x = 3 / 7

-- The main problem to prove
theorem x_minus_y_div_x_eq_4_7 (x y : ℝ) (h : y / x = 3 / 7) : (x - y) / x = 4 / 7 := by
  sorry

end NUMINAMATH_GPT_x_minus_y_div_x_eq_4_7_l351_35146


namespace NUMINAMATH_GPT_value_of_x_squared_plus_9y_squared_l351_35135

theorem value_of_x_squared_plus_9y_squared {x y : ℝ}
    (h1 : x + 3 * y = 6)
    (h2 : x * y = -9) :
    x^2 + 9 * y^2 = 90 :=
by
  sorry

end NUMINAMATH_GPT_value_of_x_squared_plus_9y_squared_l351_35135


namespace NUMINAMATH_GPT_correct_average_calculation_l351_35171

-- Conditions as definitions
def incorrect_average := 5
def num_values := 10
def incorrect_num := 26
def correct_num := 36

-- Statement to prove
theorem correct_average_calculation : 
  (incorrect_average * num_values + (correct_num - incorrect_num)) / num_values = 6 :=
by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_correct_average_calculation_l351_35171


namespace NUMINAMATH_GPT_find_sum_l351_35136

noncomputable def sumPutAtSimpleInterest (R: ℚ) (P: ℚ) := 
  let I := P * R * 5 / 100
  I + 90 = P * (R + 6) * 5 / 100 → P = 300

theorem find_sum (R: ℚ) (P: ℚ) : sumPutAtSimpleInterest R P := by
  sorry

end NUMINAMATH_GPT_find_sum_l351_35136


namespace NUMINAMATH_GPT_uncle_age_when_seokjin_is_12_l351_35130

-- Definitions for the conditions
def mother_age_when_seokjin_born : ℕ := 32
def uncle_is_younger_by : ℕ := 3
def seokjin_age : ℕ := 12

-- Definition for the main hypothesis
theorem uncle_age_when_seokjin_is_12 :
  let mother_age_when_seokjin_is_12 := mother_age_when_seokjin_born + seokjin_age
  let uncle_age_when_seokjin_is_12 := mother_age_when_seokjin_is_12 - uncle_is_younger_by
  uncle_age_when_seokjin_is_12 = 41 :=
by
  sorry

end NUMINAMATH_GPT_uncle_age_when_seokjin_is_12_l351_35130


namespace NUMINAMATH_GPT_tangent_line_slope_angle_l351_35128

theorem tangent_line_slope_angle (θ : ℝ) : 
  (∃ k : ℝ, (∀ x y, k * x - y = 0) ∧ ∀ x y, x^2 + y^2 - 4 * x + 3 = 0) →
  θ = π / 6 ∨ θ = 5 * π / 6 := by
  sorry

end NUMINAMATH_GPT_tangent_line_slope_angle_l351_35128


namespace NUMINAMATH_GPT_fg_of_2_l351_35192

def f (x : ℤ) : ℤ := 3 * x + 2
def g (x : ℤ) : ℤ := (x + 1)^2

theorem fg_of_2 : f (g 2) = 29 := by
  sorry

end NUMINAMATH_GPT_fg_of_2_l351_35192


namespace NUMINAMATH_GPT_add_base8_l351_35160

theorem add_base8 : 
  let a := 2 * 8^2 + 4 * 8^1 + 6 * 8^0
  let b := 5 * 8^2 + 7 * 8^1 + 3 * 8^0
  let c := 6 * 8^1 + 2 * 8^0
  let sum := a + b + c
  sum = 1 * 8^3 + 1 * 8^2 + 2 * 8^1 + 3 * 8^0 :=
by
  -- Proof skipped
  sorry

end NUMINAMATH_GPT_add_base8_l351_35160


namespace NUMINAMATH_GPT_range_of_a_l351_35102

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 + 3 * x^2 - x + 2

-- Prove that if f(x) is decreasing on ℝ, then a must be less than or equal to -3
theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, (3 * a * x^2 + 6 * x - 1) < 0 ) → a ≤ -3 :=
sorry

end NUMINAMATH_GPT_range_of_a_l351_35102


namespace NUMINAMATH_GPT_compositeQuotientCorrect_l351_35196

namespace CompositeNumbersProof

def firstFiveCompositesProduct : ℕ :=
  21 * 22 * 24 * 25 * 26

def subsequentFiveCompositesProduct : ℕ :=
  27 * 28 * 30 * 32 * 33

def compositeQuotient : ℚ :=
  firstFiveCompositesProduct / subsequentFiveCompositesProduct

theorem compositeQuotientCorrect : compositeQuotient = 1 / 1964 := by sorry

end CompositeNumbersProof

end NUMINAMATH_GPT_compositeQuotientCorrect_l351_35196


namespace NUMINAMATH_GPT_roadster_paving_company_cement_usage_l351_35170

theorem roadster_paving_company_cement_usage :
  let L := 10
  let T := 5.1
  L + T = 15.1 :=
by
  -- proof is omitted
  sorry

end NUMINAMATH_GPT_roadster_paving_company_cement_usage_l351_35170


namespace NUMINAMATH_GPT_perfect_square_trinomial_m_l351_35148

theorem perfect_square_trinomial_m (m : ℤ) :
  (∃ a b : ℤ, (b^2 = 25) ∧ (a + b)^2 = x^2 - (m - 3) * x + 25) → (m = 13 ∨ m = -7) :=
by
  sorry

end NUMINAMATH_GPT_perfect_square_trinomial_m_l351_35148


namespace NUMINAMATH_GPT_jina_has_1_koala_bear_l351_35199

theorem jina_has_1_koala_bear:
  let teddies := 5
  let bunnies := 3 * teddies
  let additional_teddies := 2 * bunnies
  let total_teddies := teddies + additional_teddies
  let total_bunnies_and_teddies := total_teddies + bunnies
  let total_mascots := 51
  let koala_bears := total_mascots - total_bunnies_and_teddies
  koala_bears = 1 :=
by
  sorry

end NUMINAMATH_GPT_jina_has_1_koala_bear_l351_35199


namespace NUMINAMATH_GPT_all_push_ups_total_l351_35181

-- Definitions derived from the problem's conditions
def ZacharyPushUps := 47
def DavidPushUps := ZacharyPushUps + 15
def EmilyPushUps := DavidPushUps * 2
def TotalPushUps := ZacharyPushUps + DavidPushUps + EmilyPushUps

-- The statement to be proved
theorem all_push_ups_total : TotalPushUps = 233 := by
  sorry

end NUMINAMATH_GPT_all_push_ups_total_l351_35181


namespace NUMINAMATH_GPT_profit_percentage_is_25_l351_35191

theorem profit_percentage_is_25 
  (selling_price : ℝ) (cost_price : ℝ) 
  (sp_val : selling_price = 600) 
  (cp_val : cost_price = 480) : 
  (selling_price - cost_price) / cost_price * 100 = 25 := by
  sorry

end NUMINAMATH_GPT_profit_percentage_is_25_l351_35191


namespace NUMINAMATH_GPT_average_speed_is_42_l351_35159

theorem average_speed_is_42 (v t : ℝ) (h : t > 0)
  (h_eq : v * t = (v + 21) * (2/3) * t) : v = 42 :=
by
  sorry

end NUMINAMATH_GPT_average_speed_is_42_l351_35159


namespace NUMINAMATH_GPT_ship_B_has_highest_rt_no_cars_l351_35104

def ship_percentage_with_no_cars (total_rt: ℕ) (percent_with_cars: ℕ) : ℕ :=
  total_rt - (percent_with_cars * total_rt) / 100

theorem ship_B_has_highest_rt_no_cars :
  let A_rt := 30
  let A_with_cars := 25
  let B_rt := 50
  let B_with_cars := 15
  let C_rt := 20
  let C_with_cars := 35
  let A_no_cars := ship_percentage_with_no_cars A_rt A_with_cars
  let B_no_cars := ship_percentage_with_no_cars B_rt B_with_cars
  let C_no_cars := ship_percentage_with_no_cars C_rt C_with_cars
  A_no_cars < B_no_cars ∧ C_no_cars < B_no_cars := by
  sorry

end NUMINAMATH_GPT_ship_B_has_highest_rt_no_cars_l351_35104


namespace NUMINAMATH_GPT_range_of_a_l351_35121

open Set

-- Define proposition p
def p (x : ℝ) : Prop := x^2 + 2 * x - 3 > 0

-- Define proposition q
def q (x a : ℝ) : Prop := (x - a) / (x - a - 1) > 0

-- Define negation of p
def not_p (x : ℝ) : Prop := -3 ≤ x ∧ x ≤ 1

-- Define negation of q
def not_q (x a : ℝ) : Prop := a ≤ x ∧ x ≤ a + 1

-- Main theorem to prove the range of a
theorem range_of_a (a : ℝ) : (∀ x : ℝ, a ≤ x ∧ x ≤ a + 1 → -3 ≤ x ∧ x ≤ 1) → a ∈ Icc (-3 : ℝ) (0 : ℝ) :=
by
  intro h
  -- skipped detailed proof
  sorry

end NUMINAMATH_GPT_range_of_a_l351_35121


namespace NUMINAMATH_GPT_distance_with_father_l351_35106

variable (total_distance driven_with_mother driven_with_father: ℝ)

theorem distance_with_father :
  total_distance = 0.67 ∧ driven_with_mother = 0.17 → driven_with_father = 0.50 := 
by
  sorry

end NUMINAMATH_GPT_distance_with_father_l351_35106


namespace NUMINAMATH_GPT_sum_of_decimals_as_fraction_l351_35103

theorem sum_of_decimals_as_fraction :
  (0.2 : ℝ) + (0.03 : ℝ) + (0.004 : ℝ) + (0.0006 : ℝ) + (0.00007 : ℝ) + (0.000008 : ℝ) + (0.0000009 : ℝ) = 
  (2340087 / 10000000 : ℝ) :=
sorry

end NUMINAMATH_GPT_sum_of_decimals_as_fraction_l351_35103


namespace NUMINAMATH_GPT_find_p_l351_35137

theorem find_p (A B C p q r s : ℝ) (h₀ : A ≠ 0)
  (h₁ : r + s = -B / A)
  (h₂ : r * s = C / A)
  (h₃ : r^3 + s^3 = -p) :
  p = (B^3 - 3 * A * B * C + 2 * A^2 * C^2) / A^3 :=
sorry

end NUMINAMATH_GPT_find_p_l351_35137


namespace NUMINAMATH_GPT_first_grade_muffins_total_l351_35156

theorem first_grade_muffins_total :
  let muffins_brier : ℕ := 218
  let muffins_macadams : ℕ := 320
  let muffins_flannery : ℕ := 417
  let muffins_smith : ℕ := 292
  let muffins_jackson : ℕ := 389
  muffins_brier + muffins_macadams + muffins_flannery + muffins_smith + muffins_jackson = 1636 :=
by
  apply sorry

end NUMINAMATH_GPT_first_grade_muffins_total_l351_35156


namespace NUMINAMATH_GPT_max_x_plus_y_l351_35131

theorem max_x_plus_y (x y : ℝ) (h : x^2 + y^2 + x * y = 1) : x + y ≤ 2 * Real.sqrt (3) / 3 :=
sorry

end NUMINAMATH_GPT_max_x_plus_y_l351_35131


namespace NUMINAMATH_GPT_function_C_is_odd_and_decreasing_l351_35101

-- Conditions
def f (x : ℝ) : ℝ := -x^3 - x

-- Odd function condition
def is_odd (f : ℝ → ℝ) : Prop :=
∀ x : ℝ, f (-x) = -f x

-- Strictly decreasing condition
def is_strictly_decreasing (f : ℝ → ℝ) : Prop :=
∀ x1 x2 : ℝ, x1 < x2 → f x1 > f x2

-- The theorem we want to prove
theorem function_C_is_odd_and_decreasing : 
  is_odd f ∧ is_strictly_decreasing f :=
by
  sorry

end NUMINAMATH_GPT_function_C_is_odd_and_decreasing_l351_35101


namespace NUMINAMATH_GPT_third_day_sales_correct_l351_35112

variable (a : ℕ)

def firstDaySales := a
def secondDaySales := a + 4
def thirdDaySales := 2 * (a + 4) - 7
def expectedSales := 2 * a + 1

theorem third_day_sales_correct : thirdDaySales a = expectedSales a :=
by
  -- Main proof goes here
  sorry

end NUMINAMATH_GPT_third_day_sales_correct_l351_35112


namespace NUMINAMATH_GPT_martians_cannot_hold_hands_l351_35149

-- Define the number of hands each Martian possesses
def hands_per_martian := 3

-- Define the number of Martians
def number_of_martians := 7

-- Define the total number of hands
def total_hands := hands_per_martian * number_of_martians

-- Prove that it is not possible for the seven Martians to hold hands with each other
theorem martians_cannot_hold_hands :
  ¬ ∃ (pairs : ℕ), 2 * pairs = total_hands :=
by
  sorry

end NUMINAMATH_GPT_martians_cannot_hold_hands_l351_35149


namespace NUMINAMATH_GPT_k_value_l351_35172

open Real

noncomputable def k_from_roots (α β : ℝ) : ℝ := - (α + β)

theorem k_value (k : ℝ) (α β : ℝ) (h1 : α + β = -k) (h2 : α * β = 8) (h3 : (α+3) + (β+3) = k) (h4 : (α+3) * (β+3) = 12) : k = 3 :=
by
  -- Here we skip the proof as instructed.
  sorry

end NUMINAMATH_GPT_k_value_l351_35172


namespace NUMINAMATH_GPT_total_spent_l351_35134

variable (T_L J_L C_L S_L T_C J_C C_C S_C D_C A_C : ℝ)

/-- Conditions from the problem setup --/
def conditions :=
  T_L = 40 ∧
  J_L = 0.5 * T_L ∧
  C_L = 2 * T_L ∧
  S_L = 3 * J_L ∧
  T_C = 0.25 * T_L ∧
  J_C = 3 * J_L ∧
  C_C = 0.5 * C_L ∧
  S_C = S_L ∧
  D_C = 2 * S_C ∧
  A_C = 0.5 * J_C

/-- Total spent by Lisa --/
def total_Lisa := T_L + J_L + C_L + S_L

/-- Total spent by Carly --/
def total_Carly := T_C + J_C + C_C + S_C + D_C + A_C

/-- Combined total spent by Lisa and Carly --/
theorem total_spent :
  conditions T_L J_L C_L S_L T_C J_C C_C S_C D_C A_C →
  total_Lisa T_L J_L C_L S_L + total_Carly T_C J_C C_C S_C D_C A_C = 520 :=
by
  sorry

end NUMINAMATH_GPT_total_spent_l351_35134


namespace NUMINAMATH_GPT_stimulus_check_total_l351_35133

def find_stimulus_check (T : ℝ) : Prop :=
  let amount_after_wife := T * (3/5)
  let amount_after_first_son := amount_after_wife * (3/5)
  let amount_after_second_son := amount_after_first_son * (3/5)
  amount_after_second_son = 432

theorem stimulus_check_total (T : ℝ) : find_stimulus_check T → T = 2000 := by
  sorry

end NUMINAMATH_GPT_stimulus_check_total_l351_35133


namespace NUMINAMATH_GPT_oranges_to_apples_equiv_apples_for_36_oranges_l351_35122

-- Conditions
def weight_equiv (oranges apples : ℕ) : Prop :=
  9 * oranges = 6 * apples

-- Question (Theorem to Prove)
theorem oranges_to_apples_equiv_apples_for_36_oranges:
  ∃ (apples : ℕ), apples = 24 ∧ weight_equiv 36 apples :=
by
  use 24
  sorry

end NUMINAMATH_GPT_oranges_to_apples_equiv_apples_for_36_oranges_l351_35122


namespace NUMINAMATH_GPT_unique_solution_ffx_eq_27_l351_35193

-- Definition of the function f(x)
def f (x : ℝ) : ℝ := x^3 - 3 * x^2 - 9 * x + 27

-- Prove that there is exactly one solution for f(f(x)) = 27 in the domain -3 ≤ x ≤ 5
theorem unique_solution_ffx_eq_27 :
  (∃! x : ℝ, -3 ≤ x ∧ x ≤ 5 ∧ f (f x) = 27) :=
by
  sorry

end NUMINAMATH_GPT_unique_solution_ffx_eq_27_l351_35193


namespace NUMINAMATH_GPT_cone_lateral_surface_area_l351_35179

theorem cone_lateral_surface_area (l d : ℝ) (h_l : l = 5) (h_d : d = 8) : 
  (π * (d / 2) * l) = 20 * π :=
by
  sorry

end NUMINAMATH_GPT_cone_lateral_surface_area_l351_35179


namespace NUMINAMATH_GPT_probability_same_color_is_117_200_l351_35162

/-- There are eight green balls, five red balls, and seven blue balls in a bag. 
    A ball is taken from the bag, its color recorded, then placed back in the bag.
    A second ball is taken and its color recorded. -/
def probability_two_balls_same_color : ℚ :=
  let pGreen := (8 : ℚ) / 20
  let pRed := (5 : ℚ) / 20
  let pBlue := (7 : ℚ) / 20
  pGreen^2 + pRed^2 + pBlue^2

theorem probability_same_color_is_117_200 : probability_two_balls_same_color = 117 / 200 := by
  sorry

end NUMINAMATH_GPT_probability_same_color_is_117_200_l351_35162


namespace NUMINAMATH_GPT_weekly_allowance_l351_35144

variable (A : ℝ)   -- declaring A as a real number

theorem weekly_allowance (h1 : (3/5 * A) + 1/3 * (2/5 * A) + 1 = A) : 
  A = 3.75 :=
sorry

end NUMINAMATH_GPT_weekly_allowance_l351_35144


namespace NUMINAMATH_GPT_intersection_P_Q_l351_35105

def P : Set ℝ := {y | ∃ x : ℝ, y = -x^2 + 2}
def Q : Set ℝ := {y | ∃ x : ℝ, y = -x + 2}

theorem intersection_P_Q : P ∩ Q = {y | y ≤ 2} :=
sorry

end NUMINAMATH_GPT_intersection_P_Q_l351_35105


namespace NUMINAMATH_GPT_total_marbles_l351_35109

theorem total_marbles (mary_marbles : ℕ) (joan_marbles : ℕ) (h1 : mary_marbles = 9) (h2 : joan_marbles = 3) : mary_marbles + joan_marbles = 12 := by
  sorry

end NUMINAMATH_GPT_total_marbles_l351_35109


namespace NUMINAMATH_GPT_mr_smith_total_cost_l351_35168

noncomputable def total_cost : ℝ :=
  let adult_price := 30
  let child_price := 15
  let teen_price := 25
  let senior_discount := 0.10
  let college_discount := 0.05
  let senior_price := adult_price * (1 - senior_discount)
  let college_price := adult_price * (1 - college_discount)
  let soda_price := 2
  let iced_tea_price := 3
  let coffee_price := 4
  let juice_price := 1.50
  let wine_price := 6
  let buffet_cost := 2 * adult_price + 2 * senior_price + 3 * child_price + teen_price + 2 * college_price
  let drinks_cost := 3 * soda_price + 2 * iced_tea_price + coffee_price + juice_price + 2 * wine_price
  buffet_cost + drinks_cost

theorem mr_smith_total_cost : total_cost = 270.50 :=
by
  sorry

end NUMINAMATH_GPT_mr_smith_total_cost_l351_35168


namespace NUMINAMATH_GPT_inverse_solution_correct_l351_35188

noncomputable def f (a b c x : ℝ) : ℝ :=
  1 / (a * x^2 + b * x + c)

theorem inverse_solution_correct (a b c x : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  f a b c x = 1 ↔ x = (-b + Real.sqrt (b^2 - 4 * a * (c - 1))) / (2 * a) ∨
               x = (-b - Real.sqrt (b^2 - 4 * a * (c - 1))) / (2 * a) :=
by
  sorry

end NUMINAMATH_GPT_inverse_solution_correct_l351_35188


namespace NUMINAMATH_GPT_min_value_sin_function_l351_35187

theorem min_value_sin_function (α β : ℝ) (h : -5 * (Real.sin α) ^ 2 + (Real.sin β) ^ 2 = 3 * Real.sin α) :
  ∃ x : ℝ, x = Real.sin α ∧ (Real.sin α) ^ 2 + (Real.sin β) ^ 2 = 0 :=
sorry

end NUMINAMATH_GPT_min_value_sin_function_l351_35187


namespace NUMINAMATH_GPT_axis_of_symmetry_of_function_l351_35110

theorem axis_of_symmetry_of_function 
  (f : ℝ → ℝ)
  (h : ∀ x, f x = 3 * Real.cos x - Real.sqrt 3 * Real.sin x)
  : ∃ k : ℤ, x = k * Real.pi - Real.pi / 6 ∧ x = Real.pi - Real.pi / 6 :=
sorry

end NUMINAMATH_GPT_axis_of_symmetry_of_function_l351_35110


namespace NUMINAMATH_GPT_symmetric_point_origin_l351_35186

-- Define the point P
structure Point3D where
  x : Int
  y : Int
  z : Int

def P : Point3D := { x := 1, y := 3, z := -5 }

-- Define the symmetric function w.r.t. the origin
def symmetric_with_origin (p : Point3D) : Point3D :=
  { x := -p.x, y := -p.y, z := -p.z }

-- Define the expected result
def Q : Point3D := { x := -1, y := -3, z := 5 }

-- The theorem to prove
theorem symmetric_point_origin : symmetric_with_origin P = Q := by
  sorry

end NUMINAMATH_GPT_symmetric_point_origin_l351_35186


namespace NUMINAMATH_GPT_solve_fraction_equation_l351_35152

-- Defining the function f
def f (x : ℝ) : ℝ := x + 4

-- Statement of the problem
theorem solve_fraction_equation (x : ℝ) :
  (3 * f (x - 2)) / f 0 + 4 = f (2 * x + 1) ↔ x = 2 / 5 := by
  sorry

end NUMINAMATH_GPT_solve_fraction_equation_l351_35152


namespace NUMINAMATH_GPT_profit_share_ratio_l351_35176

theorem profit_share_ratio (P Q : ℝ) (hP : P = 40000) (hQ : Q = 60000) : P / Q = 2 / 3 :=
by
  rw [hP, hQ]
  norm_num

end NUMINAMATH_GPT_profit_share_ratio_l351_35176


namespace NUMINAMATH_GPT_triangle_angle_contradiction_l351_35129

theorem triangle_angle_contradiction :
  ∀ (α β γ : ℝ), α + β + γ = 180 ∧ α > 60 ∧ β > 60 ∧ γ > 60 → False :=
by
  sorry

end NUMINAMATH_GPT_triangle_angle_contradiction_l351_35129


namespace NUMINAMATH_GPT_total_footprints_l351_35119

def pogo_footprints_per_meter : ℕ := 4
def grimzi_footprints_per_6_meters : ℕ := 3
def distance_traveled_meters : ℕ := 6000

theorem total_footprints : (pogo_footprints_per_meter * distance_traveled_meters) + (grimzi_footprints_per_6_meters * (distance_traveled_meters / 6)) = 27000 :=
by
  sorry

end NUMINAMATH_GPT_total_footprints_l351_35119
