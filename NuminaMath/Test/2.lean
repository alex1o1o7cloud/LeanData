namespace solve_for_x

import Mathlib

theorem solve_for_x : (∃ x : ℝ, 5 * x + 4 = -6) → x = -2 := 
by
  sorry

end solve_for_x

namespace age_ratio_in_1_year

import Mathlib

variable (j m x : ℕ)

-- Conditions
def condition1 (j m : ℕ) : Prop :=
  j - 3 = 2 * (m - 3)

def condition2 (j m : ℕ) : Prop :=
  j - 5 = 3 * (m - 5)

-- Question
def age_ratio (j m x : ℕ) : Prop :=
  (j + x) * 2 = 3 * (m + x)

theorem age_ratio_in_1_year (j m x : ℕ) :
  condition1 j m → condition2 j m → age_ratio j m 1 :=
by
  sorry

end age_ratio_in_1_year

namespace parallel_lines_slope

import Mathlib

theorem parallel_lines_slope (m : ℚ) (h : (x - y = 1) → (m + 3) * x + m * y - 8 = 0) :
  m = -3 / 2 :=
sorry

end parallel_lines_slope

namespace equation_of_plane_passing_through_points

import Mathlib

/-
Let M1, M2, and M3 be points in three-dimensional space.
M1 = (1, 2, 0)
M2 = (1, -1, 2)
M3 = (0, 1, -1)
We need to prove that the plane passing through these points has the equation 5x - 2y - 3z - 1 = 0.
-/

structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

def M1 : Point3D := ⟨1, 2, 0⟩
def M2 : Point3D := ⟨1, -1, 2⟩
def M3 : Point3D := ⟨0, 1, -1⟩

theorem equation_of_plane_passing_through_points :
  ∃ (a b c d : ℝ), (∀ (P : Point3D), 
  P = M1 ∨ P = M2 ∨ P = M3 → a * P.x + b * P.y + c * P.z + d = 0)
  ∧ a = 5 ∧ b = -2 ∧ c = -3 ∧ d = -1 :=
by
  sorry

end equation_of_plane_passing_through_points

namespace each_friend_paid

import Mathlib

def cottage_cost_per_hour : ℕ := 5
def rental_duration_hours : ℕ := 8
def total_cost := cottage_cost_per_hour * rental_duration_hours
def cost_per_person := total_cost / 2

theorem each_friend_paid : cost_per_person = 20 :=
by 
  sorry

end each_friend_paid

namespace tina_pink_pens

import Mathlib

def number_pink_pens (P G B : ℕ) : Prop :=
  G = P - 9 ∧
  B = P - 6 ∧
  P + G + B = 21

theorem tina_pink_pens :
  ∃ (P G B : ℕ), number_pink_pens P G B ∧ P = 12 :=
by
  sorry

end tina_pink_pens

namespace polynomial_coefficients_sum_and_difference

import Mathlib

theorem polynomial_coefficients_sum_and_difference :
  ∀ (a_0 a_1 a_2 a_3 a_4 : ℤ),
  (∀ (x : ℤ), (2 * x - 3)^4 = a_0 + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4) →
  (a_1 + a_2 + a_3 + a_4 = -80) ∧ ((a_0 + a_2 + a_4)^2 - (a_1 + a_3)^2 = 625) :=
by
  intros a_0 a_1 a_2 a_3 a_4 h
  sorry

end polynomial_coefficients_sum_and_difference

namespace width_of_river

import Mathlib

def river_depth : ℝ := 7
def flow_rate_kmph : ℝ := 4
def volume_per_minute : ℝ := 35000

noncomputable def flow_rate_mpm : ℝ := (flow_rate_kmph * 1000) / 60

theorem width_of_river : 
  ∃ w : ℝ, 
    volume_per_minute = flow_rate_mpm * river_depth * w ∧
    w = 75 :=
by
  use 75
  field_simp [flow_rate_mpm, river_depth, volume_per_minute]
  norm_num
  sorry

end width_of_river

namespace impossible_digit_placement

import Mathlib

-- Define the main variables and assumptions
variable (A B C : ℕ)
variable (h_sum : A + B = 45)
variable (h_segmentSum : 3 * A + B = 6 * C)

-- Define the impossible placement problem
theorem impossible_digit_placement :
  ¬(∃ A B C, A + B = 45 ∧ 3 * A + B = 6 * C ∧ 2 * A = 6 * C - 45) :=
by
  sorry

end impossible_digit_placement

namespace smallest_k_for_inequality

import Mathlib

theorem smallest_k_for_inequality : 
  ∃ k : ℕ,  k > 0 ∧ ( (k-10) ^ 5026 ≥ 2013 ^ 2013 ) ∧ 
  (∀ m : ℕ, m > 0 ∧ ((m-10) ^ 5026) ≥ 2013 ^ 2013 → m ≥ 55) :=
sorry

end smallest_k_for_inequality

namespace ways_to_divide_day

import Mathlib

theorem ways_to_divide_day : 
  ∃ nm_count: ℕ, nm_count = 72 ∧ ∀ n m: ℕ, 0 < n ∧ 0 < m ∧ n * m = 72000 → 
  ∃ nm_pairs: ℕ, nm_pairs = 72 * 2 :=
sorry

end ways_to_divide_day

namespace laura_saves_more_with_promotion_A

import Mathlib

def promotion_A_cost (pair_price : ℕ) : ℕ :=
  let second_pair_price := pair_price / 2
  pair_price + second_pair_price

def promotion_B_cost (pair_price : ℕ) : ℕ :=
  let discount := pair_price * 20 / 100
  pair_price + (pair_price - discount)

def savings (pair_price : ℕ) : ℕ :=
  promotion_B_cost pair_price - promotion_A_cost pair_price

theorem laura_saves_more_with_promotion_A :
  savings 50 = 15 :=
  by
  -- The detailed proof will be added here
  sorry

end laura_saves_more_with_promotion_A

namespace opposite_neg_two

import Mathlib

def opposite (x : Int) : Int := -x

theorem opposite_neg_two : opposite (-2) = 2 := by
  sorry

end opposite_neg_two

namespace union_sets

import Mathlib

noncomputable def setA : Set ℝ := { x | x^2 - 3*x - 4 ≤ 0 }
noncomputable def setB : Set ℝ := { x | 1 < x ∧ x < 5 }

theorem union_sets :
  (setA ∪ setB) = { x | -1 ≤ x ∧ x < 5 } :=
by
  sorry

end union_sets

namespace inequality_2_inequality_4

import Mathlib

variables (a b : ℝ)
variables (h₁ : 0 < a) (h₂ : 0 < b)

theorem inequality_2 (h₁ : 0 < a) (h₂ : 0 < b) : a > |a - b| - b :=
by
  sorry

theorem inequality_4 (h₁ : 0 < a) (h₂ : 0 < b) : ab + 2 / ab > 2 :=
by
  sorry

end inequality_2_inequality_4

namespace find_p

import Mathlib

noncomputable def area_of_ABC (p : ℚ) : ℚ :=
  128 - 6 * p

theorem find_p (p : ℚ) : area_of_ABC p = 45 → p = 83 / 6 := by
  intro h
  sorry

end find_p

namespace abs_expression_value

import Mathlib

theorem abs_expression_value (x : ℤ) (h : x = -2023) :
  abs (2 * abs (abs x - x) - abs x) - x = 8092 :=
by {
  -- Proof will be provided here
  sorry
}

end abs_expression_value

namespace ellipse_equation_and_fixed_point_proof

import Mathlib

theorem ellipse_equation_and_fixed_point_proof :
  (∀ (m n : ℝ), (m > 0) → (n > 0) → (m ≠ n) →
                (m * 0^2 + n * (-2)^2 = 1) ∧ (m * (3/2)^2 + n * (-1)^2 = 1) → 
                (m = 1/3 ∧ n = 1/4)) ∧
                (∀ (M N : ℝ × ℝ), ∃ H : ℝ × ℝ,
                (∃ (P : ℝ × ℝ), P = (1, -2)) ∧
                (∃ (A : ℝ × ℝ), A = (0, -2)) ∧
                (∃ (B : ℝ × ℝ), B = (3/2, -1)) ∧
                (∃ (T : ℝ × ℝ), ∀ x, M.1 = x) ∧
                (∃ K : ℝ × ℝ, K = (0, -2)) →
                M.1 * N.2 - M.2 * N.1 = 0) :=
by
  sorry

end ellipse_equation_and_fixed_point_proof

namespace find_value

import Mathlib

theorem find_value (x : ℝ) (h : 3 * x + 2 = 11) : 5 * x - 3 = 12 :=
by
  sorry

end find_value

namespace factorize_expr

import Mathlib

theorem factorize_expr (x : ℝ) : (x - 1) * (x + 3) + 4 = (x + 1) ^ 2 :=
by
  sorry

end factorize_expr

namespace study_time

import Mathlib

theorem study_time (n_mcq n_fitb : ℕ) (t_mcq t_fitb : ℕ) (total_minutes_per_hour : ℕ) 
  (h1 : n_mcq = 30) (h2 : n_fitb = 30) (h3 : t_mcq = 15) (h4 : t_fitb = 25) (h5 : total_minutes_per_hour = 60) : 
  n_mcq * t_mcq + n_fitb * t_fitb = 20 * total_minutes_per_hour := 
by 
  -- This is a placeholder for the proof
  sorry

end study_time

namespace probability_of_exactly_one_red_ball

import Mathlib

-- Definitions based on the conditions:
def total_balls : ℕ := 5
def red_balls : ℕ := 2
def white_balls : ℕ := 3
def draw_count : ℕ := 2

-- Required to calculate combinatory values
def choose (n k : ℕ) : ℕ :=
  if k > n then 0
  else Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Definitions of probabilities (though we won't use them explicitly for the statement):
def total_events : ℕ := choose total_balls draw_count
def no_red_ball_events : ℕ := choose white_balls draw_count
def one_red_ball_events : ℕ := choose red_balls 1 * choose white_balls 1

-- Probability Functions (for context):
def probability (events : ℕ) (total_events : ℕ) : ℚ := events / total_events

-- Lean 4 statement:
theorem probability_of_exactly_one_red_ball :
  probability one_red_ball_events total_events = 3/5 := by
  sorry

end probability_of_exactly_one_red_ball

namespace football_cost_correct

import Mathlib

variable (total_spent_on_toys : ℝ := 12.30)
variable (spent_on_marbles : ℝ := 6.59)

theorem football_cost_correct :
  (total_spent_on_toys - spent_on_marbles = 5.71) :=
by
  sorry

end football_cost_correct

namespace quadratic_function_has_specific_k

import Mathlib

theorem quadratic_function_has_specific_k (k : ℤ) :
  (∀ x : ℝ, ∃ y : ℝ, y = (k-1)*x^(k^2-k+2) + k*x - 1) ↔ k = 0 :=
by
  sorry

end quadratic_function_has_specific_k

namespace find_number_of_rabbits

import Mathlib

def total_heads (R P : ℕ) : ℕ := R + P
def total_legs (R P : ℕ) : ℕ := 4 * R + 2 * P

theorem find_number_of_rabbits (R P : ℕ)
  (h1 : total_heads R P = 60)
  (h2 : total_legs R P = 192) :
  R = 36 := by
  sorry

end find_number_of_rabbits

namespace pyramid_layers_total

import Mathlib -- ensuring necessary libraries are imported

-- Since we are dealing with natural number calculations, noncomputable is generally not needed.

-- Definition of the pyramid layers and the number of balls in each layer
def number_of_balls (n : ℕ) : ℕ := n ^ 2

-- Given conditions for the layers
def third_layer_balls : ℕ := number_of_balls 3
def fifth_layer_balls : ℕ := number_of_balls 5

-- Statement of the problem proving that their sum is 34
theorem pyramid_layers_total : third_layer_balls + fifth_layer_balls = 34 := by
  sorry -- proof to be provided

end pyramid_layers_total

namespace find_g

import Mathlib

theorem find_g (x : ℝ) (g : ℝ → ℝ) :
  2 * x^5 - 4 * x^3 + 3 * x^2 + g x = 7 * x^4 - 5 * x^3 + x^2 - 9 * x + 2 →
  g x = -2 * x^5 + 7 * x^4 - x^3 - 2 * x^2 - 9 * x + 2 :=
by
  intro h
  sorry

end find_g

namespace cos_theta_value_projection_value

import Mathlib

noncomputable def vec_a : (ℝ × ℝ) := (3, 1)
noncomputable def vec_b : (ℝ × ℝ) := (-2, 4)

theorem cos_theta_value :
  let a := vec_a
  let b := vec_b
  let dot_product := a.1 * b.1 + a.2 * b.2
  let magnitude_a := Real.sqrt (a.1^2 + a.2^2)
  let magnitude_b := Real.sqrt (b.1^2 + b.2^2)
  dot_product / (magnitude_a * magnitude_b) = - Real.sqrt 2 / 10 :=
by 
  sorry

theorem projection_value :
  let a := vec_a
  let b := vec_b
  let dot_product := a.1 * b.1 + a.2 * b.2
  let magnitude_a := Real.sqrt (a.1^2 + a.2^2)
  let magnitude_b := Real.sqrt (b.1^2 + b.2^2)
  let cos_theta := dot_product / (magnitude_a * magnitude_b)
  cos_theta = - Real.sqrt 2 / 10 →
  magnitude_a * cos_theta = - Real.sqrt 5 / 5 :=
by 
  sorry

end cos_theta_value_projection_value

namespace smallest_base

import Mathlib

theorem smallest_base (b : ℕ) (h1 : b^2 ≤ 125) (h2 : 125 < b^3) : b = 6 := by
  sorry

end smallest_base

namespace common_ratio_geometric_sequence

import Mathlib

theorem common_ratio_geometric_sequence (a b c d : ℤ) (h1 : a = 10) (h2 : b = -20) (h3 : c = 40) (h4 : d = -80) :
    b / a = -2 ∧ c = b * -2 ∧ d = c * -2 := by
  sorry

end common_ratio_geometric_sequence

namespace cos_double_angle

import Mathlib

-- Definition of the terminal condition
def terminal_side_of_angle (α : ℝ) (x y : ℝ) : Prop :=
  (x = 1) ∧ (y = Real.sqrt 3) ∧ (x^2 + y^2 = 1)

-- Prove the required statement
theorem cos_double_angle (α : ℝ) :
  (terminal_side_of_angle α 1 (Real.sqrt 3)) →
  Real.cos (2 * α + Real.pi / 2) = - Real.sqrt 3 / 2 :=
by
  sorry

end cos_double_angle

namespace correct_statements_count

import Mathlib

theorem correct_statements_count :
  (∀ x > 0, x > Real.sin x) ∧
  (¬ (∀ x > 0, x - Real.log x > 0) ↔ (∃ x > 0, x - Real.log x ≤ 0)) ∧
  ¬ (∀ p q : Prop, (p ∨ q) → (p ∧ q)) →
  2 = 2 :=
by sorry

end correct_statements_count

namespace relationship_of_ys

import Mathlib

variables {k y1 y2 y3 : ℝ}

theorem relationship_of_ys (h : k < 0) 
  (h1 : y1 = k / -4) 
  (h2 : y2 = k / 2) 
  (h3 : y3 = k / 3) : 
  y1 > y3 ∧ y3 > y2 :=
by 
  sorry

end relationship_of_ys

namespace puzzles_sold_correct

import Mathlib

def science_kits_sold : ℕ := 45
def puzzles_sold : ℕ := science_kits_sold - 9

theorem puzzles_sold_correct : puzzles_sold = 36 := by
  -- Proof will be provided here
  sorry

end puzzles_sold_correct

namespace average_speed

import Mathlib

-- Defining conditions
def speed_first_hour : ℕ := 100  -- The car travels 100 km in the first hour
def speed_second_hour : ℕ := 60  -- The car travels 60 km in the second hour
def total_distance : ℕ := speed_first_hour + speed_second_hour  -- Total distance traveled

def total_time : ℕ := 2  -- Total time taken in hours

-- Stating the theorem
theorem average_speed : total_distance / total_time = 80 := 
by
  sorry

end average_speed

namespace sum_of_ages_53

import Mathlib

variable (B D : ℕ)

def Ben_3_years_younger_than_Dan := B + 3 = D
def Ben_is_25 := B = 25
def sum_of_their_ages (B D : ℕ) := B + D

theorem sum_of_ages_53 : ∀ (B D : ℕ), Ben_3_years_younger_than_Dan B D → Ben_is_25 B → sum_of_their_ages B D = 53 :=
by
  sorry

end sum_of_ages_53

namespace shop_owner_profitable

import Mathlib

noncomputable def shop_owner_profit (CP_SP_difference_percentage: ℚ) (CP: ℚ) (buy_cheat_percentage: ℚ) (sell_cheat_percentage: ℚ) (buy_discount_percentage: ℚ) (sell_markup_percentage: ℚ) : ℚ := 
  CP_SP_difference_percentage * 100

theorem shop_owner_profitable :
  shop_owner_profit ((114 * (110 / 80 / 100) - 90) / 90) 1 0.14 0.20 0.10 0.10 = 74.17 := 
by
  sorry

end shop_owner_profitable

namespace people_counted

import Mathlib

-- Define the conditions
def first_day_count (second_day_count : ℕ) : ℕ := 2 * second_day_count
def second_day_count : ℕ := 500

-- Define the total count
def total_count (first_day : ℕ) (second_day : ℕ) : ℕ := first_day + second_day

-- Statement of the proof problem: Prove that the total count is 1500 given the conditions
theorem people_counted : total_count (first_day_count second_day_count) second_day_count = 1500 := by
  sorry

end people_counted

namespace part_A_part_B_part_D

import Mathlib

variables (c d : ℤ)

def multiple_of_5 (x : ℤ) : Prop := ∃ k : ℤ, x = 5 * k
def multiple_of_10 (x : ℤ) : Prop := ∃ k : ℤ, x = 10 * k

-- Given conditions
axiom h1 : multiple_of_5 c
axiom h2 : multiple_of_10 d

-- Problems to prove
theorem part_A : multiple_of_5 d := by sorry
theorem part_B : multiple_of_5 (c - d) := by sorry
theorem part_D : multiple_of_5 (c + d) := by sorry

end part_A_part_B_part_D

namespace opposite_sqrt3

import Mathlib

def opposite (x : ℝ) : ℝ := -x

theorem opposite_sqrt3 :
  opposite (Real.sqrt 3) = -Real.sqrt 3 :=
by
  sorry

end opposite_sqrt3

namespace amount_of_juice_p_in_a

import Mathlib

  def total_p : ℚ := 24
  def total_v : ℚ := 25
  def ratio_a : ℚ := 4 / 1
  def ratio_y : ℚ := 1 / 5

  theorem amount_of_juice_p_in_a :
    ∃ P_a : ℚ, ∃ V_a : ℚ, ∃ P_y : ℚ, ∃ V_y : ℚ,
      P_a / V_a = ratio_a ∧ P_y / V_y = ratio_y ∧
      P_a + P_y = total_p ∧ V_a + V_y = total_v ∧ P_a = 20 :=
  by
    sorry
  
end amount_of_juice_p_in_a

namespace second_quadrant_point

import Mathlib

theorem second_quadrant_point (x : ℝ) (h1 : x < 2) (h2 : x > 1/2) : 
  (x-2 < 0) ∧ (2*x-1 > 0) ↔ (1/2 < x ∧ x < 2) :=
by
  sorry

end second_quadrant_point

namespace exterior_angle_of_regular_octagon

import Mathlib

def sum_of_interior_angles (n : ℕ) : ℕ := 180 * (n - 2)
def interior_angle (s : ℕ) (n : ℕ) : ℕ := sum_of_interior_angles n / s
def exterior_angle (ia : ℕ) : ℕ := 180 - ia

theorem exterior_angle_of_regular_octagon : 
    exterior_angle (interior_angle 8 8) = 45 := 
by 
  sorry

end exterior_angle_of_regular_octagon

namespace michelle_total_payment

import Mathlib
noncomputable def michelle_base_cost := 25
noncomputable def included_talk_time := 40 -- in hours
noncomputable def text_cost := 10 -- in cents per message
noncomputable def extra_talk_cost := 15 -- in cents per minute
noncomputable def february_texts_sent := 200
noncomputable def february_talk_time := 41 -- in hours

theorem michelle_total_payment : 
  25 + ((200 * 10) / 100) + (((41 - 40) * 60 * 15) / 100) = 54 := by
  sorry

end michelle_total_payment

namespace valid_lineups_count

import Mathlib

-- Define the conditions
def total_players : ℕ := 15
def max : ℕ := 1
def rex : ℕ := 1
def tex : ℕ := 1

-- Proving the number of valid lineups
theorem valid_lineups_count :
  ∃ n, n = 5 ∧ total_players = 15 ∧ max + rex + tex ≤ 1 → n = 2277 :=
sorry

end valid_lineups_count

namespace flower_team_participation

import Mathlib

-- Definitions based on the conditions in the problem
def num_rows : ℕ := 60
def first_row_people : ℕ := 40
def people_increment : ℕ := 1

-- Statement to be proved in Lean
theorem flower_team_participation (x : ℕ) (hx : 1 ≤ x ∧ x ≤ num_rows) : 
  ∃ y : ℕ, y = first_row_people - people_increment + x :=
by
  -- Placeholder for the proof
  sorry

end flower_team_participation

namespace decimal_to_fraction_correct

import Mathlib

-- Define a structure representing our initial decimal to fraction conversion
structure DecimalFractionConversion :=
  (decimal: ℚ)
  (vulgar_fraction: ℚ)
  (simplified_fraction: ℚ)

-- Define the conditions provided in the problem
def conversion_conditions : DecimalFractionConversion :=
  { decimal := 35 / 100,
    vulgar_fraction := 35 / 100,
    simplified_fraction := 7 / 20 }

-- State the theorem we aim to prove
theorem decimal_to_fraction_correct :
  conversion_conditions.simplified_fraction = 7 / 20 := by
  sorry

end decimal_to_fraction_correct

namespace total_amount_spent_on_cookies

import Mathlib

def days_in_april : ℕ := 30
def cookies_per_day : ℕ := 3
def cost_per_cookie : ℕ := 18

theorem total_amount_spent_on_cookies : days_in_april * cookies_per_day * cost_per_cookie = 1620 := by
  sorry

end total_amount_spent_on_cookies

namespace cookies_last_days

import Mathlib

variable (c1 c2 t : ℕ)

/-- Jackson's oldest son gets 4 cookies after school each day, and his youngest son gets 2 cookies. 
There are 54 cookies in the box, so the number of days the box will last is 9. -/
theorem cookies_last_days (h1 : c1 = 4) (h2 : c2 = 2) (h3 : t = 54) : 
  t / (c1 + c2) = 9 := by
  sorry

end cookies_last_days

namespace gift_boxes_in_3_days

import Mathlib
-- Conditions:
def inchesPerBox := 18
def dailyWrapper := 90
-- "how many gift boxes will he be able to wrap every 3 days?"
theorem gift_boxes_in_3_days : 3 * (dailyWrapper / inchesPerBox) = 15 :=
by
  sorry

end gift_boxes_in_3_days

namespace given_conditions_implies_correct_answer

import Mathlib

noncomputable def is_binomial_coefficient_equal (n : ℕ) : Prop := 
  Nat.choose n 2 = Nat.choose n 6

noncomputable def sum_of_odd_terms (n : ℕ) : ℕ :=
  2 ^ (n - 1)

theorem given_conditions_implies_correct_answer (n : ℕ) (h : is_binomial_coefficient_equal n) : 
  n = 8 ∧ sum_of_odd_terms n = 128 := by 
  sorry

end given_conditions_implies_correct_answer

namespace fraction_B_compared_to_A_and_C

import Mathlib

theorem fraction_B_compared_to_A_and_C
    (A B C : ℕ) 
    (h1 : A = (B + C) / 3) 
    (h2 : A = B + 35) 
    (h3 : A + B + C = 1260) : 
    (∃ x : ℚ, B = x * (A + C) ∧ x = 2 / 7) :=
by
  sorry

end fraction_B_compared_to_A_and_C

namespace largest_k_consecutive_sum

import Mathlib

theorem largest_k_consecutive_sum (k : ℕ) (h1 : (∃ n : ℕ, 3^12 = k * n + (k*(k-1))/2)) : k ≤ 729 :=
by
  -- Proof omitted for brevity
  sorry

end largest_k_consecutive_sum

namespace y_relationship_range_of_x

import Mathlib

-- Definitions based on conditions
variable (x : ℝ) (y : ℝ)

-- Condition: Perimeter of the isosceles triangle is 6 cm
def perimeter_is_6 (x : ℝ) (y : ℝ) : Prop :=
  2 * x + y = 6

-- Condition: Function relationship of y in terms of x
def y_function (x : ℝ) : ℝ :=
  6 - 2 * x

-- Prove the functional relationship y = 6 - 2x
theorem y_relationship (x : ℝ) : y = y_function x ↔ perimeter_is_6 x y := by
  sorry

-- Prove the range of values for x
theorem range_of_x (x : ℝ) : 3 / 2 < x ∧ x < 3 ↔ (0 < y_function x ∧ perimeter_is_6 x (y_function x)) := by
  sorry

end y_relationship_range_of_x

namespace find_number_of_students

import Mathlib

variables (n : ℕ)
variables (avg_A avg_B avg_C excl_avg_A excl_avg_B excl_avg_C : ℕ)
variables (new_avg_A new_avg_B new_avg_C : ℕ)
variables (excluded_students : ℕ)

theorem find_number_of_students :
  avg_A = 80 ∧ avg_B = 85 ∧ avg_C = 75 ∧
  excl_avg_A = 20 ∧ excl_avg_B = 25 ∧ excl_avg_C = 15 ∧
  excluded_students = 5 ∧
  new_avg_A = 90 ∧ new_avg_B = 95 ∧ new_avg_C = 85 →
  n = 35 :=
by
  sorry

end find_number_of_students

namespace abc_equivalence

import Mathlib

theorem abc_equivalence (n : ℕ) (k : ℤ) (a b c : ℤ)
  (hn : 0 < n) (hk : k % 2 = 1)
  (h : a^n + k * b = b^n + k * c ∧ b^n + k * c = c^n + k * a) :
  a = b ∧ b = c := 
sorry

end abc_equivalence

namespace coloring_possible

import Mathlib

-- Define what it means for a graph to be planar and bipartite
def planar_graph (G : Type) : Prop := sorry
def bipartite_graph (G : Type) : Prop := sorry

-- The planar graph G results after subdivision without introducing new intersections
def subdivided_graph (G : Type) : Type := sorry

-- Main theorem to prove
theorem coloring_possible (G : Type) (h1 : planar_graph G) : 
  bipartite_graph (subdivided_graph G) :=
sorry

end coloring_possible

namespace sum_of_repeating_decimals

import Mathlib

-- Definitions of repeating decimals x and y
def x : ℚ := 25 / 99
def y : ℚ := 87 / 99

-- The assertion that the sum of these repeating decimals is equal to 112/99 as a fraction
theorem sum_of_repeating_decimals: x + y = 112 / 99 := by
  sorry

end sum_of_repeating_decimals

namespace rate_of_mangoes_per_kg

import Mathlib

variable (grapes_qty : ℕ := 8)
variable (grapes_rate_per_kg : ℕ := 70)
variable (mangoes_qty : ℕ := 9)
variable (total_amount_paid : ℕ := 1055)

theorem rate_of_mangoes_per_kg :
  (total_amount_paid - grapes_qty * grapes_rate_per_kg) / mangoes_qty = 55 :=
by
  sorry

end rate_of_mangoes_per_kg

namespace fraction_value

import Mathlib

theorem fraction_value : (20 * 21) / (2 + 0 + 2 + 1) = 84 := by
  sorry

end fraction_value

namespace part1_part2

import Mathlib

def f (x a : ℝ) : ℝ := abs (x - a) + 2 * x

theorem part1 (x : ℝ) : f x (-1) ≤ 0 ↔ x ≤ -1/3 :=
by
  sorry

theorem part2 (a : ℝ) : (∀ x : ℝ, x ≥ -1 → f x a ≥ 0) ↔ (a ≤ -3 ∨ a ≥ 1) :=
by
  sorry

end part1_part2

namespace number_of_cooks

import Mathlib

variable (C W : ℕ)

-- Conditions
def initial_ratio := 3 * W = 8 * C
def new_ratio := 4 * C = W + 12

theorem number_of_cooks (h1 : initial_ratio W C) (h2 : new_ratio W C) : C = 9 := by
  sorry

end number_of_cooks

namespace compute_expression

import Mathlib

theorem compute_expression : ((-5) * 3) - (7 * (-2)) + ((-4) * (-6)) = 23 := by
  sorry

end compute_expression

namespace factor_expression

import Mathlib

theorem factor_expression (x : ℝ) :
  x^3 - 6 * x^2 + 9 * x = x * (x - 3)^2 := 
  sorry

end factor_expression

namespace find_b

import Mathlib

theorem find_b (b : ℝ) (y : ℝ) : (4 * 3 + 2 * y = b) ∧ (3 * 3 + 6 * y = 3 * b) → b = 27 :=
by
sorry

end find_b

namespace tetrahedron_circumscribed_sphere_radius

import Mathlib

open Real

theorem tetrahedron_circumscribed_sphere_radius :
  ∀ (A B C D : ℝ × ℝ × ℝ), 
    dist A B = 5 →
    dist C D = 5 →
    dist A C = sqrt 34 →
    dist B D = sqrt 34 →
    dist A D = sqrt 41 →
    dist B C = sqrt 41 →
    ∃ (R : ℝ), R = 5 * sqrt 2 / 2 :=
by
  intros A B C D hAB hCD hAC hBD hAD hBC
  sorry

end tetrahedron_circumscribed_sphere_radius

namespace expression_equals_one

import Mathlib

variable {R : Type*} [Field R]
variables (x y z : R)

theorem expression_equals_one (h₁ : x ≠ y) (h₂ : x ≠ z) (h₃ : y ≠ z) :
    (x^2 / ((x - y) * (x - z)) + y^2 / ((y - x) * (y - z)) + z^2 / ((z - x) * (z - y))) = 1 :=
by sorry

end expression_equals_one

namespace sally_pokemon_cards

import Mathlib

variable (X : ℤ)

theorem sally_pokemon_cards : X + 41 + 20 = 34 → X = -27 :=
by
  sorry

end sally_pokemon_cards

namespace ab_equiv

import Mathlib

theorem ab_equiv (a b : ℝ) (hb : b ≠ 0) (h : (a - b) / b = 3 / 7) : a / b = 10 / 7 :=
by
  sorry

end ab_equiv

namespace circle_radius_triple_area

import Mathlib

/-- Given the area of a circle is tripled when its radius r is increased by n, prove that 
    r = n * (sqrt(3) - 1) / 2 -/
theorem circle_radius_triple_area (r n : ℝ) (h : π * (r + n) ^ 2 = 3 * π * r ^ 2) :
  r = n * (Real.sqrt 3 - 1) / 2 :=
sorry

end circle_radius_triple_area

namespace ladder_base_distance

import Mathlib

theorem ladder_base_distance
  (c : ℕ) (b : ℕ) (hypotenuse : c = 13) (wall_height : b = 12) :
  ∃ x : ℕ, x^2 + b^2 = c^2 ∧ x = 5 := by
  sorry

end ladder_base_distance

namespace katherine_time_20

import Mathlib

noncomputable def time_katherine_takes (k : ℝ) :=
  let time_naomi_takes_per_website := (5/4) * k
  let total_websites := 30
  let total_time_naomi := 750
  time_naomi_takes_per_website = 25 ∧ k = 20

theorem katherine_time_20 :
  ∃ k : ℝ, time_katherine_takes k :=
by
  use 20
  sorry

end katherine_time_20

namespace original_average

import Mathlib

theorem original_average (n : ℕ) (A : ℝ) (new_avg : ℝ) 
  (h1 : n = 25) 
  (h2 : new_avg = 140) 
  (h3 : 2 * A = new_avg) : A = 70 :=
sorry

end original_average

namespace scientific_notation_equivalence

import Mathlib

/-- The scientific notation for 20.26 thousand hectares in square meters is equal to 2.026 × 10^9. -/
theorem scientific_notation_equivalence :
  (20.26 * 10^3 * 10^4) = 2.026 * 10^9 := 
sorry

end scientific_notation_equivalence

namespace log_inequality

import Mathlib

noncomputable def log (x : ℝ) : ℝ := Real.log x / Real.log 10

theorem log_inequality (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a ≠ b) (h5 : b ≠ c) (h6 : c ≠ a) :
  log ((a + b) / 2) + log ((b + c) / 2) + log ((c + a) / 2) > log a + log b + log c :=
by
  sorry

end log_inequality

namespace carl_sold_each_watermelon_for_3

import Mathlib

def profit : ℕ := 105
def final_watermelons : ℕ := 18
def starting_watermelons : ℕ := 53
def sold_watermelons : ℕ := starting_watermelons - final_watermelons
def price_per_watermelon : ℕ := profit / sold_watermelons

theorem carl_sold_each_watermelon_for_3 :
  price_per_watermelon = 3 :=
by
  sorry

end carl_sold_each_watermelon_for_3

namespace volume_remaining_proof

import Mathlib

noncomputable def volume_remaining_part (v_original v_total_small : ℕ) : ℕ := v_original - v_total_small

def original_edge_length := 9
def small_edge_length := 3
def num_edges := 12

def volume_original := original_edge_length ^ 3
def volume_small := small_edge_length ^ 3
def volume_total_small := num_edges * volume_small

theorem volume_remaining_proof : volume_remaining_part volume_original volume_total_small = 405 := by
  sorry

end volume_remaining_proof

namespace complex_sum_identity

import Mathlib

theorem complex_sum_identity (z : ℂ) (h : z^2 + z + 1 = 0) : 
  z^100 + z^101 + z^102 + z^103 + z^104 = -1 := 
by 
  sorry

end complex_sum_identity

namespace combined_total_value_of_items

import Mathlib

theorem combined_total_value_of_items :
  let V1 := 87.50 / 0.07
  let V2 := 144 / 0.12
  let V3 := 50 / 0.05
  let total1 := 1000 + V1
  let total2 := 1000 + V2
  let total3 := 1000 + V3
  total1 + total2 + total3 = 6450 := 
by
  sorry

end combined_total_value_of_items

namespace rectangle_area

import Mathlib

-- Define the rectangular properties
variables {w l d x : ℝ}
def width (w : ℝ) : ℝ := w
def length (w : ℝ) : ℝ := 3 * w
def diagonal (w : ℝ) : ℝ := x

theorem rectangle_area (w x : ℝ) (hw : w ^ 2 + (3 * w) ^ 2 = x ^ 2) : w * 3 * w = 3 / 10 * x ^ 2 :=
by 
  sorry

end rectangle_area

namespace moving_circle_fixed_point

import Mathlib

def parabola (p : ℝ × ℝ) : Prop := p.2^2 = 4 * p.1

def tangent_line (c : ℝ × ℝ) (r : ℝ) : Prop :=
  abs (c.1 + 1) = r

theorem moving_circle_fixed_point :
  ∀ (c : ℝ × ℝ) (r : ℝ),
    parabola c →
    tangent_line c r →
    (1, 0) ∈ {p : ℝ × ℝ | dist c p = r} :=
by
  intro c r hc ht
  sorry

end moving_circle_fixed_point

namespace sum_combinatorial_identity

import Mathlib

theorem sum_combinatorial_identity (k m n : ℕ) (hk : 0 < k) (hm : 0 < m) (hn : 0 < n) (hkn : k ≤ n) :
  (∑ r in Finset.range (m + 1), k * (Nat.choose m r) * (Nat.choose n k) / ((r + k) * (Nat.choose (m + n) (r + k)))) = 1 :=
sorry

end sum_combinatorial_identity

namespace cost_per_pack

import Mathlib

variable (total_amount : ℕ) (number_of_packs : ℕ)

theorem cost_per_pack (h1 : total_amount = 132) (h2 : number_of_packs = 11) : 
  total_amount / number_of_packs = 12 := by
  sorry

end cost_per_pack

namespace janets_total_pockets

import Mathlib

-- Define the total number of dresses
def totalDresses : ℕ := 36

-- Define the dresses with pockets
def dressesWithPockets : ℕ := totalDresses / 2

-- Define the dresses without pockets
def dressesWithoutPockets : ℕ := totalDresses - dressesWithPockets

-- Define the dresses with one hidden pocket
def dressesWithOneHiddenPocket : ℕ := (40 * dressesWithoutPockets) / 100

-- Define the dresses with 2 pockets
def dressesWithTwoPockets : ℕ := dressesWithPockets / 3

-- Define the dresses with 3 pockets
def dressesWithThreePockets : ℕ := dressesWithPockets / 4

-- Define the dresses with 4 pockets
def dressesWithFourPockets : ℕ := dressesWithPockets - dressesWithTwoPockets - dressesWithThreePockets

-- Calculate the total number of pockets
def totalPockets : ℕ := 
  2 * dressesWithTwoPockets + 
  3 * dressesWithThreePockets + 
  4 * dressesWithFourPockets + 
  dressesWithOneHiddenPocket

-- The theorem to prove the total number of pockets
theorem janets_total_pockets : totalPockets = 63 :=
  by
    -- Proof is omitted, use 'sorry'
    sorry

end janets_total_pockets

namespace proof_strictly_increasing_sequence

import Mathlib

noncomputable def exists_strictly_increasing_sequence : Prop :=
  ∃ a : ℕ → ℕ, 
    (∀ n m : ℕ, n < m → a n < a m) ∧
    (∀ m : ℕ, ∃ i j : ℕ, m = a i + a j) ∧
    (∀ n : ℕ, 0 < n → a n > n^2 / 16)

theorem proof_strictly_increasing_sequence : exists_strictly_increasing_sequence :=
  sorry

end proof_strictly_increasing_sequence

namespace sequence_u5_value

import Mathlib

theorem sequence_u5_value (u : ℕ → ℝ) 
  (h_rec : ∀ n, u (n + 2) = 2 * u (n + 1) + u n)
  (h_u3 : u 3 = 9) 
  (h_u6 : u 6 = 128) : 
  u 5 = 53 := 
sorry

end sequence_u5_value

namespace sum_of_squares_of_roots_eq

import Mathlib

-- Definitions derived directly from conditions
def a := 5
def b := 2
def c := -15

-- Sum of roots
def sum_of_roots : ℚ := (-b : ℚ) / a

-- Product of roots
def product_of_roots : ℚ := (c : ℚ) / a

-- Sum of the squares of the roots
def sum_of_squares_of_roots : ℚ := sum_of_roots^2 - 2 * product_of_roots

-- The statement that needs to be proved
theorem sum_of_squares_of_roots_eq : sum_of_squares_of_roots = 154 / 25 :=
by
  sorry

end sum_of_squares_of_roots_eq

namespace units_of_Product_C_sold

import Mathlib

-- Definitions of commission rates
def commission_rate_A : ℝ := 0.05
def commission_rate_B : ℝ := 0.07
def commission_rate_C : ℝ := 0.10

-- Definitions of revenues per unit
def revenue_A : ℝ := 1500
def revenue_B : ℝ := 2000
def revenue_C : ℝ := 3500

-- Definition of units sold
def units_A : ℕ := 5
def units_B : ℕ := 3

-- Commission calculations for Product A and B
def commission_A : ℝ := commission_rate_A * revenue_A * units_A
def commission_B : ℝ := commission_rate_B * revenue_B * units_B

-- Previous average commission and new average commission
def previous_avg_commission : ℝ := 100
def new_avg_commission : ℝ := 250

-- The main proof statement
theorem units_of_Product_C_sold (x : ℝ) (h1 : new_avg_commission = previous_avg_commission + 150)
  (h2 : total_units = units_A + units_B + x)
  (h3 : total_new_commission = commission_A + commission_B + (commission_rate_C * revenue_C * x))
  : x = 12 :=
by
  sorry

end units_of_Product_C_sold

namespace rahul_share_is_100

import Mathlib

-- Definitions of the conditions
def rahul_rate := 1/3
def rajesh_rate := 1/2
def total_payment := 250

-- Definition of their work rate when they work together
def combined_rate := rahul_rate + rajesh_rate

-- Definition of the total value of the work done in one day when both work together
noncomputable def combined_work_value := total_payment / combined_rate

-- Definition of Rahul's share for the work done in one day
noncomputable def rahul_share := rahul_rate * combined_work_value

-- The theorem we need to prove
theorem rahul_share_is_100 : rahul_share = 100 := by
  sorry

end rahul_share_is_100

namespace factorial_mod_10

import Mathlib

-- Define the factorial function
def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

-- Define the problem statement
theorem factorial_mod_10 : factorial 10 % 13 = 7 :=
by sorry

end factorial_mod_10

namespace man_climbs_out_of_well_in_65_days

import Mathlib

theorem man_climbs_out_of_well_in_65_days (depth climb slip net_days last_climb : ℕ) 
  (h_depth : depth = 70)
  (h_climb : climb = 6)
  (h_slip : slip = 5)
  (h_net_days : net_days = 64)
  (h_last_climb : last_climb = 1) :
  ∃ days : ℕ, days = net_days + last_climb ∧ days = 65 := by
  sorry

end man_climbs_out_of_well_in_65_days

namespace hundredth_ring_square_count

import Mathlib

-- Conditions
def center_rectangle : ℤ × ℤ := (1, 2)
def first_ring_square_count : ℕ := 10
def square_count_nth_ring (n : ℕ) : ℕ := 8 * n + 2

-- Problem Statement
theorem hundredth_ring_square_count : square_count_nth_ring 100 = 802 := 
  sorry

end hundredth_ring_square_count

namespace projection_area_rectangular_board

import Mathlib

noncomputable def projection_area (AB BC NE MN : ℝ) (ABCD_perp_ground : Prop) (E_mid_AB : Prop) (light_at_M : Prop) : ℝ :=
  let width := AB
  let height := BC
  let shadow_width := 5
  (1 / 2) * (width + shadow_width) * height

theorem projection_area_rectangular_board (AB BC NE MN : ℝ) (ABCD_perp_ground : Prop) (E_mid_AB : Prop) (light_at_M : Prop) :
  AB = 3 → BC = 2 → NE = 3 → MN = 5 → projection_area AB BC NE MN ABCD_perp_ground E_mid_AB light_at_M = 8 :=
by
  intros
  sorry

end projection_area_rectangular_board

namespace find_prime_pair

import Mathlib

noncomputable def is_prime (n : ℕ) : Prop := 
  n > 1 ∧ ∀ m : ℕ, m > 1 ∧ m < n → n % m ≠ 0

def has_integer_root (p q : ℕ) : Prop :=
  ∃ x : ℤ, x^4 + p * x^3 - q = 0

theorem find_prime_pair :
  ∃ (p q : ℕ), is_prime p ∧ is_prime q ∧ has_integer_root p q ∧ p = 2 ∧ q = 3 := by
  sorry

end find_prime_pair

namespace remainder_of_3_pow_17_mod_7

import Mathlib

theorem remainder_of_3_pow_17_mod_7 :
  (3^17 % 7) = 5 :=
by 
  sorry

end remainder_of_3_pow_17_mod_7

namespace evaluate_expression

import Mathlib

theorem evaluate_expression :
  let x := (1 : ℚ) / 2
  let y := (3 : ℚ) / 4
  let z := -6
  let w := 2
  (x^2 * y^4 * z * w = - (243 / 256)) := 
by {
  let x := (1 : ℚ) / 2
  let y := (3 : ℚ) / 4
  let z := -6
  let w := 2
  sorry
}

end evaluate_expression

namespace min_value_x_plus_y

import Mathlib

theorem min_value_x_plus_y (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : 4 / x + 1 / y = 1 / 2) : x + y ≥ 18 := sorry

end min_value_x_plus_y

namespace find_second_number

import Mathlib

-- The Lean statement for the given math problem:

theorem find_second_number
  (x y z : ℝ)  -- Represent the three numbers
  (h1 : x = 2 * y)  -- The first number is twice the second
  (h2 : z = (1/3) * x)  -- The third number is one-third of the first
  (h3 : x + y + z = 110)  -- The sum of the three numbers is 110
  : y = 30 :=  -- The second number is 30
sorry

end find_second_number

namespace range_of_a

import Mathlib

noncomputable def f (x : ℝ) : ℝ :=
if x ≥ 0 then x^2 + 2 * x else -(x^2 + 2 * x)

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, f (-x) = -f x) ∧ (∀ x : ℝ, x ≥ 0 → f x = x^2 + 2 * x) →
  f (2 - a^2) > f a ↔ -2 < a ∧ a < 1 :=
by
  sorry

end range_of_a

namespace votes_difference

import Mathlib

theorem votes_difference (T : ℕ) (V_a : ℕ) (V_f : ℕ) 
  (h1 : T = 330) (h2 : V_a = 40 * T / 100) (h3 : V_f = T - V_a) : V_f - V_a = 66 :=
by
  sorry

end votes_difference

namespace range_of_a

import Mathlib

noncomputable def problem_statement : Prop :=
  ∃ x : ℝ, (1 ≤ x) ∧ (∀ a : ℝ, (1 + 1 / x) ^ (x + a) ≥ Real.exp 1 → a ≥ 1 / Real.log 2 - 1)

theorem range_of_a : problem_statement :=
sorry

end range_of_a

namespace marbles_problem_a_marbles_problem_b

import Mathlib

-- Define the problem as Lean statements.

-- Part (a): m = 2004, n = 2006
theorem marbles_problem_a (m n : ℕ) (h_m : m = 2004) (h_n : n = 2006) :
  ∃ (marbles : ℕ → ℕ → ℕ), 
  (∀ i j, 1 ≤ i ∧ i ≤ m ∧ 1 ≤ j ∧ j ≤ n → marbles i j = 1) := 
sorry

-- Part (b): m = 2005, n = 2006
theorem marbles_problem_b (m n : ℕ) (h_m : m = 2005) (h_n : n = 2006) :
  ∃ (marbles : ℕ → ℕ → ℕ), 
  (∀ i j, 1 ≤ i ∧ i ≤ m ∧ 1 ≤ j ∧ j ≤ n → marbles i j = 1) → false := 
sorry

end marbles_problem_a_marbles_problem_b

namespace fraction_nonneg_if_x_ge_m8

import Mathlib

noncomputable def denominator (x : ℝ) : ℝ := x^2 + 4*x + 13
noncomputable def numerator (x : ℝ) : ℝ := x + 8

theorem fraction_nonneg_if_x_ge_m8 (x : ℝ) (hx : x ≥ -8) : numerator x / denominator x ≥ 0 :=
by sorry

end fraction_nonneg_if_x_ge_m8

namespace ellipse_equation

import Mathlib

theorem ellipse_equation (a b c : ℝ) :
  (2 * a = 10) ∧ (c / a = 4 / 5) →
  ((x:ℝ)^2 / 25 + (y:ℝ)^2 / 9 = 1) ∨ ((x:ℝ)^2 / 9 + (y:ℝ)^2 / 25 = 1) :=
by
  sorry

end ellipse_equation

namespace smallest_next_divisor

import Mathlib

def is_even (n : ℕ) : Prop := n % 2 = 0
def is_divisor (a b : ℕ) : Prop := b % a = 0

theorem smallest_next_divisor 
  (m : ℕ) 
  (h1 : 1000 ≤ m ∧ m < 10000) 
  (h2 : is_even m) 
  (h3 : is_divisor 171 m)
  : ∃ k, k > 171 ∧ k = 190 ∧ is_divisor k m := 
by
  sorry

end smallest_next_divisor

namespace initial_bottle_caps

import Mathlib

variable (x : Nat)

theorem initial_bottle_caps (h : x + 3 = 29) : x = 26 := by
  sorry

end initial_bottle_caps

namespace select_4_people_arrangement_3_day_new_year

import Mathlib

def select_4_people_arrangement (n k : ℕ) : ℕ :=
  Nat.choose n 2 * Nat.factorial (n - 2) / Nat.factorial 2

theorem select_4_people_arrangement_3_day_new_year :
  select_4_people_arrangement 7 4 = 420 :=
by
  -- proof to be filled in
  sorry

end select_4_people_arrangement_3_day_new_year

namespace ara_height

import Mathlib

/-
Conditions:
1. Shea's height increased by 25%.
2. Shea is now 65 inches tall.
3. Ara grew by three-quarters as many inches as Shea did.

Prove Ara's height is 61.75 inches.
-/

def shea_original_height (x : ℝ) : Prop := 1.25 * x = 65

def ara_growth (growth : ℝ) (shea_growth : ℝ) : Prop := growth = (3 / 4) * shea_growth

def shea_growth (original_height : ℝ) : ℝ := 0.25 * original_height

theorem ara_height (shea_orig_height : ℝ) (shea_now_height : ℝ) (ara_growth_inches : ℝ) :
  shea_original_height shea_orig_height → 
  shea_now_height = 65 →
  ara_growth ara_growth_inches (shea_now_height - shea_orig_height) →
  shea_orig_height + ara_growth_inches = 61.75 :=
by
  sorry

end ara_height

namespace arithmetic_sequence_values

import Mathlib

noncomputable def common_difference (a₁ a₂ : ℕ) : ℕ := (a₂ - a₁) / 2

theorem arithmetic_sequence_values (x y z d: ℕ) 
    (h₁: d = common_difference 7 11) 
    (h₂: x = 7 + d) 
    (h₃: y = 11 + d) 
    (h₄: z = y + d): 
    x = 9 ∧ y = 13 ∧ z = 15 :=
by {
  sorry
}

end arithmetic_sequence_values

namespace symmetric_circle

import Mathlib

theorem symmetric_circle
    (x y : ℝ)
    (circle_eq : x^2 + y^2 + 4 * x - 1 = 0) :
    (x - 2)^2 + y^2 = 5 :=
sorry

end symmetric_circle

namespace correct_propositions_count

import Mathlib

theorem correct_propositions_count (a b : ℝ) :
  (∀ a b, a > b → a + 1 > b + 1) ∧
  (∀ a b, a > b → a - 1 > b - 1) ∧
  (∀ a b, a > b → -2 * a < -2 * b) ∧
  (¬ ∀ a b, a > b → 2 * a < 2 * b) → 
  3 = 3 :=
by
  intro h
  sorry

end correct_propositions_count

namespace sine_triangle

import Mathlib

theorem sine_triangle (a b c : ℝ) (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) (h_perimeter : a + b + c ≤ 2 * Real.pi)
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (ha_pi : a < Real.pi) (hb_pi : b < Real.pi) (hc_pi : c < Real.pi):
  ∃ (x y z : ℝ), x = Real.sin a ∧ y = Real.sin b ∧ z = Real.sin c ∧ x + y > z ∧ y + z > x ∧ x + z > y :=
by
  sorry

end sine_triangle

namespace greatest_median

import Mathlib

theorem greatest_median (k m r s t : ℕ) (h1 : k < m) (h2 : m < r) (h3 : r < s) (h4 : s < t) (h5 : (k + m + r + s + t) = 80) (h6 : t = 42) : r = 17 :=
by
  sorry

end greatest_median

namespace find_roots

import Mathlib

noncomputable def P (x : ℝ) : ℝ := x^4 - 3 * x^3 + 3 * x^2 - x - 6

theorem find_roots : {x : ℝ | P x = 0} = {-1, 1, 2} :=
by
  sorry

end find_roots

namespace line_parallel_slope

import Mathlib

theorem line_parallel_slope (m : ℝ) :
  (2 * 8 = m * m) →
  m = -4 :=
by
  intro h
  sorry

end line_parallel_slope

namespace fixed_point_always

import Mathlib

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := 2^x + Real.logb a (x + 1) + 3

theorem fixed_point_always (a : ℝ) (h : a > 0 ∧ a ≠ 1) : f 0 a = 4 :=
by
  sorry

end fixed_point_always

namespace total_area_of_hexagon_is_693

import Mathlib

-- Conditions
def hexagon_side1_length := 3
def hexagon_side2_length := 2
def angle_between_length3_sides := 120
def all_internal_triangles_are_equilateral := true
def number_of_triangles := 6

-- Define the problem statement
theorem total_area_of_hexagon_is_693 
  (a1 : hexagon_side1_length = 3)
  (a2 : hexagon_side2_length = 2)
  (a3 : angle_between_length3_sides = 120)
  (a4 : all_internal_triangles_are_equilateral = true)
  (a5 : number_of_triangles = 6) :
  total_area_of_hexagon = 693 :=
by
  sorry

end total_area_of_hexagon_is_693

namespace inheritance_amount

import Mathlib

def is_inheritance_amount (x : ℝ) : Prop :=
  let federal_tax := 0.25 * x
  let remaining_after_fed := x - federal_tax
  let state_tax := 0.12 * remaining_after_fed
  let total_tax_paid := federal_tax + state_tax
  total_tax_paid = 15600

theorem inheritance_amount : 
  ∃ x, is_inheritance_amount x ∧ x = 45882 := 
by
  sorry

end inheritance_amount

namespace point_not_in_plane

import Mathlib

def is_in_plane (p0 : ℝ × ℝ × ℝ) (n : ℝ × ℝ × ℝ) (p : ℝ × ℝ × ℝ) : Prop :=
  let (x0, y0, z0) := p0
  let (nx, ny, nz) := n
  let (x, y, z) := p
  (nx * (x - x0) + ny * (y - y0) + nz * (z - z0)) = 0

theorem point_not_in_plane :
  ¬ is_in_plane (1, 2, 3) (1, 1, 1) (-2, 5, 4) :=
by
  sorry

end point_not_in_plane

namespace neither_5_nor_6_nice_1200

import Mathlib

def is_k_nice (N k : ℕ) : Prop := N % k = 1

def count_k_nice_up_to (k n : ℕ) : ℕ :=
(n + (k - 1)) / k

def count_neither_5_nor_6_nice_up_to (n : ℕ) : ℕ :=
  let count_5_nice := count_k_nice_up_to 5 n
  let count_6_nice := count_k_nice_up_to 6 n
  let count_5_and_6_nice := count_k_nice_up_to 30 n
  n - (count_5_nice + count_6_nice - count_5_and_6_nice)

theorem neither_5_nor_6_nice_1200 : count_neither_5_nor_6_nice_up_to 1200 = 800 := 
by
  sorry

end neither_5_nor_6_nice_1200

namespace solution_set_of_inequality

import Mathlib

theorem solution_set_of_inequality :
  { x : ℝ | |x + 1| + |x - 4| ≥ 7 } = { x : ℝ | x ≤ -2 ∨ x ≥ 5 } := sorry

end solution_set_of_inequality

namespace find_xyz_sum

import Mathlib

variables {x y z : ℝ}

def system_of_equations (x y z : ℝ) : Prop :=
  x > 0 ∧ y > 0 ∧ z > 0 ∧
  (x^2 + x * y + y^2 = 12) ∧
  (y^2 + y * z + z^2 = 9) ∧
  (z^2 + z * x + x^2 = 21)

theorem find_xyz_sum (x y z : ℝ) (h : system_of_equations x y z) : 
  x * y + y * z + z * x = 12 :=
sorry

end find_xyz_sum

namespace total_wet_surface_area_is_correct

import Mathlib

def cisternLength : ℝ := 8
def cisternWidth : ℝ := 4
def waterDepth : ℝ := 1.25

def bottomSurfaceArea : ℝ := cisternLength * cisternWidth
def longerSideSurfaceArea (depth : ℝ) : ℝ := depth * cisternLength * 2
def shorterSideSurfaceArea (depth : ℝ) : ℝ := depth * cisternWidth * 2

def totalWetSurfaceArea : ℝ :=
  bottomSurfaceArea + longerSideSurfaceArea waterDepth + shorterSideSurfaceArea waterDepth

theorem total_wet_surface_area_is_correct :
  totalWetSurfaceArea = 62 := by
  sorry

end total_wet_surface_area_is_correct

namespace remainder_when_7x_div_9

import Mathlib

theorem remainder_when_7x_div_9 (x : ℕ) (h : x % 9 = 5) : (7 * x) % 9 = 8 :=
sorry

end remainder_when_7x_div_9

namespace typing_speed_in_6_minutes

import Mathlib

theorem typing_speed_in_6_minutes (total_chars : ℕ) (chars_first_minute : ℕ) (chars_last_minute : ℕ) (chars_other_minutes : ℕ) :
  total_chars = 2098 →
  chars_first_minute = 112 →
  chars_last_minute = 97 →
  chars_other_minutes = 1889 →
  (1889 / 6 : ℝ) < 315 → 
  ¬(∀ n, 1 ≤ n ∧ n ≤ 14 - 6 + 1 → chars_other_minutes / 6 ≥ 946) :=
by
  -- Given that analyzing the content, 
  -- proof is skipped here, replace this line with the actual proof.
  sorry

end typing_speed_in_6_minutes

namespace smallest_lcm_value

import Mathlib

def is_five_digit (x : ℕ) : Prop :=
  10000 ≤ x ∧ x < 100000

theorem smallest_lcm_value :
  ∃ (m n : ℕ), is_five_digit m ∧ is_five_digit n ∧ Nat.gcd m n = 5 ∧ Nat.lcm m n = 20030010 :=
by
  sorry

end smallest_lcm_value

namespace Adam_bought_9_cat_food_packages

import Mathlib

def num_cat_food_packages (c : ℕ) : Prop :=
  let cat_cans := 10 * c
  let dog_cans := 7 * 5
  cat_cans = dog_cans + 55

theorem Adam_bought_9_cat_food_packages : num_cat_food_packages 9 :=
by
  unfold num_cat_food_packages
  sorry

end Adam_bought_9_cat_food_packages

namespace sequence_periodic

import Mathlib

def last_digit (n : ℕ) : ℕ := n % 10

noncomputable def a_n (n : ℕ) : ℕ := last_digit (n^(n^n))

theorem sequence_periodic :
  ∃ period : ℕ, period = 20 ∧ ∀ n m : ℕ, n ≡ m [MOD period] → a_n n = a_n m :=
sorry

end sequence_periodic

namespace sufficient_but_not_necessary

import Mathlib

theorem sufficient_but_not_necessary (x : ℝ) : (x > 1/2 → 2 * x^2 + x - 1 > 0) ∧ ¬(2 * x^2 + x - 1 > 0 → x > 1 / 2) := 
by
  sorry

end sufficient_but_not_necessary

namespace system_of_equations_solution

import Mathlib

theorem system_of_equations_solution :
  ∀ (x : Fin 100 → ℝ), 
  (x 0 + x 1 + x 2 = 0) ∧ 
  (x 1 + x 2 + x 3 = 0) ∧ 
  -- Continue for all other equations up to
  (x 98 + x 99 + x 0 = 0) ∧ 
  (x 99 + x 0 + x 1 = 0)
  → ∀ (i : Fin 100), x i = 0 :=
by
  intros x h
  -- We can insert the detailed solving steps here
  sorry

end system_of_equations_solution

namespace product_of_roots

import Mathlib

theorem product_of_roots (a b c : ℤ) (h_eqn : a = 12 ∧ b = 60 ∧ c = -720) :
  (c : ℚ) / a = -60 :=
by sorry

end product_of_roots

namespace f_neg2_minus_f_neg3

import Mathlib

-- Given conditions
variable (f : ℝ → ℝ)
variable (odd_f : ∀ x, f (-x) = - f x)
variable (h : f 3 - f 2 = 1)

-- Goal to prove
theorem f_neg2_minus_f_neg3 : f (-2) - f (-3) = 1 := by
  sorry

end f_neg2_minus_f_neg3

namespace coffee_consumption

import Mathlib

-- Defining the necessary variables and conditions
variable (Ivory_cons Brayan_cons : ℕ)
variable (hr : ℕ := 1)
variable (hrs : ℕ := 5)

-- Condition: Brayan drinks twice as much coffee as Ivory
def condition1 := Brayan_cons = 2 * Ivory_cons

-- Condition: Brayan drinks 4 cups of coffee in an hour
def condition2 := Brayan_cons = 4

-- The proof problem
theorem coffee_consumption : ∀ (Ivory_cons Brayan_cons : ℕ), (Brayan_cons = 2 * Ivory_cons) → 
  (Brayan_cons = 4) → 
  ((Brayan_cons * hrs) + (Ivory_cons * hrs) = 30) :=
by
  intro hBrayan hIvory hr
  sorry

end coffee_consumption

namespace side_length_of_smaller_square

import Mathlib

theorem side_length_of_smaller_square (s : ℝ) (A1 A2 : ℝ) (h1 : 5 * 5 = A1 + A2) (h2 : 2 * A2 = A1 + 25)  : s = 5 * Real.sqrt 3 / 3 :=
by
  sorry

end side_length_of_smaller_square

namespace cylinder_surface_area

import Mathlib

theorem cylinder_surface_area
  (l : ℝ) (r : ℝ) (unfolded_square_side : ℝ) (base_circumference : ℝ)
  (hl : unfolded_square_side = 2 * π)
  (hl_gen : l = 2 * π)
  (hc : base_circumference = 2 * π)
  (hr : r = 1) :
  2 * π * r * (r + l) = 2 * π + 4 * π^2 :=
by
  sorry

end cylinder_surface_area

namespace verify_extrema

import Mathlib

noncomputable def f (x : ℝ) : ℝ := (1 / 4) * x^4 - 2 * x^3 + (11 / 2) * x^2 - 6 * x + (9 / 4)

theorem verify_extrema :
  f 1 = 0 ∧ f 2 = 1 ∧ f 3 = 0 := by
  sorry

end verify_extrema

namespace last_digit_of_2_pow_2004

import Mathlib

theorem last_digit_of_2_pow_2004 : (2 ^ 2004) % 10 = 6 := 
by {
  sorry
}

end last_digit_of_2_pow_2004

namespace calc_expr_solve_fractional_eq

import Mathlib

-- Problem 1: Calculate the expression
theorem calc_expr : (-2)^2 - (64:ℝ)^(1/3) + (-3)^0 - (1/3)^0 = 0 := 
by 
  -- provide intermediate steps (not necessary as per the question requirements)
  sorry

-- Problem 2: Solve the fractional equation
theorem solve_fractional_eq (x : ℝ) (h : x ≠ -1) : 
  (x / (x + 1) = 5 / (2 * x + 2) - 1) ↔ x = 3 / 4 := 
by 
  -- provide intermediate steps (not necessary as per the question requirements)
  sorry

end calc_expr_solve_fractional_eq

namespace calculator_transform_implication

import Mathlib

noncomputable def transform (x n S : ℕ) : Prop :=
  (S > x^n + 1)

theorem calculator_transform_implication (x n S : ℕ) (hx : 0 < x) (hn : 0 < n) (hS : 0 < S) 
  (h_transform: transform x n S) : S > x^n + x - 1 := by
  sorry

end calculator_transform_implication

namespace rice_less_than_beans_by_30

import Mathlib

noncomputable def GB : ℝ := 60
noncomputable def S : ℝ := 50

theorem rice_less_than_beans_by_30 (R : ℝ) (x : ℝ) (h1 : R = 60 - x) (h2 : (2/3) * R + (4/5) * S + GB = 120) : 60 - R = 30 :=
by 
  -- Proof steps would go here, but they are not required for this task.
  sorry

end rice_less_than_beans_by_30

namespace lemonade_lemons_per_glass

import Mathlib

def number_of_glasses : ℕ := 9
def total_lemons : ℕ := 18
def lemons_per_glass : ℕ := 2

theorem lemonade_lemons_per_glass :
  total_lemons / number_of_glasses = lemons_per_glass :=
by
  sorry

end lemonade_lemons_per_glass

namespace a_equals_5

import Mathlib

def f (x : ℝ) (a : ℝ) : ℝ := x^3 + a*x^2 + 3*x - 9
def f' (x : ℝ) (a : ℝ) : ℝ := 3*x^2 + 2*a*x + 3

theorem a_equals_5 (a : ℝ) : 
  (∃ x : ℝ, x = -3 ∧ f' x a = 0) → a = 5 := 
by
  sorry

end a_equals_5

namespace square_inequality

import Mathlib

theorem square_inequality (a b : ℝ) (h1 : a + b < 0) (h2 : b > 0) : a^2 > b^2 :=
by
  sorry

end square_inequality

namespace smallest_m_plus_n

import Mathlib

theorem smallest_m_plus_n (m n : ℕ) (h1 : m > n) (h2 : n ≥ 1) 
(h3 : 1000 ∣ 1978^m - 1978^n) : m + n = 106 :=
sorry

end smallest_m_plus_n

namespace find_c_deg3

import Mathlib

-- Define the polynomials f and g.
def f (x : ℚ) : ℚ := 2 - 10 * x + 4 * x^2 - 5 * x^3 + 7 * x^4
def g (x : ℚ) : ℚ := 5 - 3 * x - 8 * x^3 + 11 * x^4

-- The statement that needs proof.
theorem find_c_deg3 (c : ℚ) : (∀ x : ℚ, f x + c * g x ≠ 0 → f x + c * g x = 2 - 10 * x + 4 * x^2 - 5 * x^3 - c * 8 * x^3) ↔ c = -7 / 11 :=
sorry

end find_c_deg3

namespace candle_cost_correct

import Mathlib

-- Variables and conditions
def candles_per_cake : Nat := 8
def num_cakes : Nat := 3
def candles_needed : Nat := candles_per_cake * num_cakes

def candles_per_box : Nat := 12
def boxes_needed : Nat := candles_needed / candles_per_box

def cost_per_box : ℝ := 2.5
def total_cost : ℝ := boxes_needed * cost_per_box

-- Proof statement
theorem candle_cost_correct :
  total_cost = 5 := by
  sorry

end candle_cost_correct

namespace smallest_digit_never_in_units_place_of_odd_numbers

import Mathlib

-- Define the set of units digits of odd numbers
def units_digits_odd_numbers : Set ℕ := {1, 3, 5, 7, 9}

-- Prove that the smallest digit not found in the units place of an odd number is 0
theorem smallest_digit_never_in_units_place_of_odd_numbers : ∀ d, d ∉ units_digits_odd_numbers → d = 0 :=
by
  sorry

end smallest_digit_never_in_units_place_of_odd_numbers

namespace fraction_comparison

import Mathlib

theorem fraction_comparison : 
  (15 / 11 : ℝ) > (17 / 13 : ℝ) ∧ (17 / 13 : ℝ) > (19 / 15 : ℝ) :=
by
  sorry

end fraction_comparison

namespace sum_single_digits

import Mathlib

theorem sum_single_digits (P Q R : ℕ) (hP : P ≠ Q) (hQ : Q ≠ R) (hR : R ≠ P)
  (h1 : R + R = 10)
  (h_sum : ∃ (P Q R : ℕ), P * 100 + 70 + R + 390 + R = R * 100 + Q * 10) :
  P + Q + R = 13 := 
sorry

end sum_single_digits

namespace abs_eq_condition

import Mathlib

theorem abs_eq_condition (a b : ℝ) : |a - b| = |a - 1| + |b - 1| ↔ (a - 1) * (b - 1) ≤ 0 :=
sorry

end abs_eq_condition

namespace farm_problem

import Mathlib

variable (H R : ℕ)

-- Conditions
def initial_relation : Prop := R = H + 6
def hens_updated : Prop := H + 8 = 20
def current_roosters (H R : ℕ) : ℕ := R + 4

-- Theorem statement
theorem farm_problem (H R : ℕ)
  (h1 : initial_relation H R)
  (h2 : hens_updated H) :
  current_roosters H R = 22 :=
by
  sorry

end farm_problem

namespace value_of_m_sub_n

import Mathlib

theorem value_of_m_sub_n (m n : ℤ) (h1 : |m| = 5) (h2 : n^2 = 36) (h3 : m * n < 0) : m - n = 11 ∨ m - n = -11 := 
by 
  sorry

end value_of_m_sub_n

namespace lcm_12_15_18

import Mathlib

theorem lcm_12_15_18 : Nat.lcm (Nat.lcm 12 15) 18 = 180 := by
  sorry

end lcm_12_15_18

namespace possible_sums_of_digits

import Mathlib

def is_four_digit_number (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

def all_digits_nonzero (A : ℕ) : Prop :=
  let a := (A / 1000) % 10
  let b := (A / 100) % 10
  let c := (A / 10) % 10
  let d := (A % 10)
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0

def reverse_number (A : ℕ) : ℕ :=
  let a := (A / 1000) % 10
  let b := (A / 100) % 10
  let c := (A / 10) % 10
  let d := (A % 10)
  1000 * d + 100 * c + 10 * b + a

def sum_of_digits (A : ℕ) : ℕ :=
  let a := (A / 1000) % 10
  let b := (A / 100) % 10
  let c := (A / 10) % 10
  let d := (A % 10)
  a + b + c + d

theorem possible_sums_of_digits (A B : ℕ) 
  (h_four_digit : is_four_digit_number A) 
  (h_nonzero_digits : all_digits_nonzero A) 
  (h_reverse : B = reverse_number A) 
  (h_divisible : (A + B) % 109 = 0) : 
  sum_of_digits A = 14 ∨ sum_of_digits A = 23 ∨ sum_of_digits A = 28 := 
sorry

end possible_sums_of_digits

namespace cannot_bisect_segment_with_ruler

import Mathlib

noncomputable def projective_transformation (A B M : Point) : Point :=
  -- This definition will use an unspecified projective transformation that leaves A and B invariant
  sorry

theorem cannot_bisect_segment_with_ruler (A B : Point) (method : Point -> Point -> Point) :
  (forall (phi : Point -> Point), phi A = A -> phi B = B -> phi (method A B) ≠ method A B) ->
  ¬ (exists (M : Point), method A B = M) := by
  sorry

end cannot_bisect_segment_with_ruler

namespace John_pays_amount

import Mathlib

/-- Prove the amount John pays given the conditions -/
theorem John_pays_amount
  (total_candies : ℕ)
  (candies_paid_by_dave : ℕ)
  (cost_per_candy : ℚ)
  (candies_paid_by_john := total_candies - candies_paid_by_dave)
  (total_cost_paid_by_john := candies_paid_by_john * cost_per_candy) :
  total_candies = 20 →
  candies_paid_by_dave = 6 →
  cost_per_candy = 1.5 →
  total_cost_paid_by_john = 21 := 
by
  intros h1 h2 h3
  -- Proof is skipped
  sorry

end John_pays_amount

namespace incircle_hexagon_area_ratio

import Mathlib

noncomputable def area_hexagon (s : ℝ) : ℝ :=
  (3 * Real.sqrt 3 / 2) * s^2

noncomputable def radius_incircle (s : ℝ) : ℝ :=
  (s * Real.sqrt 3) / 2

noncomputable def area_incircle (r : ℝ) : ℝ :=
  Real.pi * r^2

noncomputable def area_ratio (s : ℝ) : ℝ :=
  let A_hexagon := area_hexagon s
  let r := radius_incircle s
  let A_incircle := area_incircle r
  A_incircle / A_hexagon

theorem incircle_hexagon_area_ratio (s : ℝ) (h : s = 1) :
  area_ratio s = (Real.pi * Real.sqrt 3) / 6 :=
by
  sorry

end incircle_hexagon_area_ratio

namespace ab_divisibility

import Mathlib

theorem ab_divisibility (a b : ℕ) (h_a : a ≥ 2) (h_b : b ≥ 2) : 
  (ab - 1) % ((a - 1) * (b - 1)) = 0 ↔ (a = 2 ∧ b = 2) ∨ (a = 3 ∧ b = 3) :=
sorry

end ab_divisibility

namespace coordinates_of_P

import Mathlib

theorem coordinates_of_P (A B : ℝ × ℝ × ℝ) (m : ℝ) :
  A = (1, 0, 2) ∧ B = (1, -3, 1) ∧ (0, 0, m) = (0, 0, -3) :=
by 
  sorry

end coordinates_of_P

namespace black_rectangle_ways

import Mathlib

theorem black_rectangle_ways : ∑ a in Finset.range 5, ∑ b in Finset.range 5, (5 - a) * (5 - b) = 225 := sorry

end black_rectangle_ways

namespace right_triangle_sqrt

import Mathlib

noncomputable def sqrt_2 := Real.sqrt 2
noncomputable def sqrt_3 := Real.sqrt 3
noncomputable def sqrt_5 := Real.sqrt 5

theorem right_triangle_sqrt: 
  (sqrt_2 ^ 2 + sqrt_3 ^ 2 = sqrt_5 ^ 2) :=
by
  sorry

end right_triangle_sqrt

namespace amount_borrowed

import Mathlib

variable (P : ℝ)
variable (interest_paid : ℝ) -- Interest paid on borrowing
variable (interest_earned : ℝ) -- Interest earned on lending
variable (gain_per_year : ℝ)

variable (h1 : interest_paid = P * 4 * 2 / 100)
variable (h2 : interest_earned = P * 6 * 2 / 100)
variable (h3 : gain_per_year = 160)
variable (h4 : gain_per_year = (interest_earned - interest_paid) / 2)

theorem amount_borrowed : P = 8000 := by
  sorry

end amount_borrowed

namespace rhombus_diagonal_length

import Mathlib

theorem rhombus_diagonal_length (d2 : ℝ) (area : ℝ) (d1 : ℝ) (h1 : d2 = 80) (h2 : area = 2480) (h3 : area = (d1 * d2) / 2) : d1 = 62 :=
by sorry

end rhombus_diagonal_length

namespace final_elephants_count

import Mathlib

def E_0 : Int := 30000
def R_exodus : Int := 2880
def H_exodus : Int := 4
def R_entry : Int := 1500
def H_entry : Int := 7
def E_final : Int := E_0 - (R_exodus * H_exodus) + (R_entry * H_entry)

theorem final_elephants_count : E_final = 28980 := by
  sorry

end final_elephants_count

namespace total_monthly_bill_working_from_home

import Mathlib

def original_monthly_bill : ℝ := 60
def percentage_increase : ℝ := 0.30

theorem total_monthly_bill_working_from_home :
  original_monthly_bill + (original_monthly_bill * percentage_increase) = 78 := by
  sorry

end total_monthly_bill_working_from_home

namespace sequence_contains_composite

import Mathlib

noncomputable def is_composite (n : ℕ) : Prop :=
  ∃ d : ℕ, d > 1 ∧ d < n ∧ n % d = 0

theorem sequence_contains_composite (a : ℕ → ℕ) (h : ∀ n, a (n+1) = 2 * a n + 1 ∨ a (n+1) = 2 * a n - 1) :
  ∃ n, is_composite (a n) :=
sorry

end sequence_contains_composite

namespace problem_part1_problem_part2

import Mathlib

variable (a : ℝ)

noncomputable def f (x : ℝ) : ℝ := a - 2 / (2^x + 1)

theorem problem_part1 (h : ∀ x : ℝ, f (-x) = -f x) : a = 1 :=
sorry

theorem problem_part2 : ∀ x1 x2 : ℝ, x1 < x2 → f x1 < f x2 :=
sorry

end problem_part1_problem_part2

namespace math_proof_problem

import Mathlib

noncomputable def M : ℝ :=
  let x := (Real.sqrt (Real.sqrt 7 + 3) + Real.sqrt (Real.sqrt 7 - 3)) / (Real.sqrt (Real.sqrt 7 + 2))
  let y := Real.sqrt (5 - 2 * Real.sqrt 6)
  x - y

theorem math_proof_problem :
  M = (Real.sqrt 57 - 6 * Real.sqrt 6 + 4) / 3 :=
by
  sorry

end math_proof_problem

namespace find_initial_passengers

import Mathlib

def initial_passengers_found (P : ℕ) : Prop :=
  let after_first_station := (2 / 3 : ℚ) * P + 280
  let after_second_station := (1 / 2 : ℚ) * after_first_station + 12
  after_second_station = 242

theorem find_initial_passengers :
  ∃ P : ℕ, initial_passengers_found P ∧ P = 270 :=
by
  sorry

end find_initial_passengers

namespace percentage_error

import Mathlib

theorem percentage_error (x : ℚ) : 
  let incorrect_result := (3/5 : ℚ) * x
  let correct_result := (5/3 : ℚ) * x
  let ratio := incorrect_result / correct_result
  let percentage_error := (1 - ratio) * 100
  percentage_error = 64 :=
by
  let incorrect_result := (3/5 : ℚ) * x
  let correct_result := (5/3 : ℚ) * x
  let ratio := incorrect_result / correct_result
  let percentage_error := (1 - ratio) * 100
  sorry

end percentage_error

namespace cost_of_milkshake_is_correct

import Mathlib

-- Definitions related to the problem conditions
def initial_amount : ℕ := 15
def spent_on_cupcakes : ℕ := initial_amount * (1 / 3)
def remaining_after_cupcakes : ℕ := initial_amount - spent_on_cupcakes
def spent_on_sandwich : ℕ := remaining_after_cupcakes * (20 / 100)
def remaining_after_sandwich : ℕ := remaining_after_cupcakes - spent_on_sandwich
def remaining_after_milkshake : ℕ := 4
def cost_of_milkshake : ℕ := remaining_after_sandwich - remaining_after_milkshake

-- The theorem stating the equivalent proof problem
theorem cost_of_milkshake_is_correct :
  cost_of_milkshake = 4 :=
sorry

end cost_of_milkshake_is_correct

namespace max_value_of_expression

import Mathlib

open Real

theorem max_value_of_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h : x + y + z = 1) : 
  x^4 * y^2 * z ≤ 1024 / 7^7 :=
sorry

end max_value_of_expression

namespace total_material_ordered

import Mathlib

theorem total_material_ordered :
  12.468 + 4.6278 + 7.9101 + 8.3103 + 5.6327 = 38.9499 :=
by
  sorry

end total_material_ordered

namespace bridge_length_is_235

import Mathlib

noncomputable def length_of_bridge (train_length : ℕ) (train_speed_kmh : ℕ) (time_sec : ℕ) : ℕ :=
  let train_speed_ms := (train_speed_kmh * 1000) / 3600
  let total_distance := train_speed_ms * time_sec
  let bridge_length := total_distance - train_length
  bridge_length

theorem bridge_length_is_235 :
  length_of_bridge 140 45 30 = 235 :=
by 
  sorry

end bridge_length_is_235

namespace evaluate_expression

import Mathlib

theorem evaluate_expression : 
  101^3 + 3 * (101^2) * 2 + 3 * 101 * (2^2) + 2^3 = 1092727 := 
by 
  sorry

end evaluate_expression

namespace sufficient_but_not_necessary_condition

import Mathlib

theorem sufficient_but_not_necessary_condition (a : ℝ) :
  (∃ a, (1 + a) ^ 6 = 64) →
  (a = 1 → (1 + a) ^ 6 = 64) ∧ ¬(∀ a, ((1 + a) ^ 6 = 64 → a = 1)) :=
by
  sorry

end sufficient_but_not_necessary_condition

namespace smallest_positive_period_symmetry_axis_range_of_f

import Mathlib
noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x + (Real.pi / 6))

theorem smallest_positive_period :
  ∃ T > 0, ∀ x, f (x + T) = f x ∧ T = Real.pi := sorry

theorem symmetry_axis (k : ℤ) :
  ∃ k : ℤ, ∃ x : ℝ, f x = f (x + k * (Real.pi / 2)) ∧ x = (Real.pi / 6) + k * (Real.pi / 2) := sorry

theorem range_of_f : 
  ∀ x, -Real.pi / 12 ≤ x ∧ x ≤ Real.pi / 2 → -1/2 ≤ f x ∧ f x ≤ 1 := sorry

end smallest_positive_period_symmetry_axis_range_of_f

namespace derivative_f_at_1

import Mathlib

-- Define the function f(x) = x * ln(x)
noncomputable def f (x : ℝ) : ℝ := x * Real.log x

-- State the theorem to prove f'(1) = 1
theorem derivative_f_at_1 : (deriv f 1) = 1 :=
sorry

end derivative_f_at_1

namespace multiple_of_regular_rate_is_1_5

import Mathlib

-- Definitions
def hourly_rate := 5.50
def regular_hours := 7.5
def total_hours := 10.5
def total_earnings := 66.0
def excess_hours := total_hours - regular_hours
def regular_earnings := regular_hours * hourly_rate
def excess_earnings := total_earnings - regular_earnings
def rate_per_excess_hour := excess_earnings / excess_hours
def multiple_of_regular_rate := rate_per_excess_hour / hourly_rate

-- Statement of the problem
theorem multiple_of_regular_rate_is_1_5 : multiple_of_regular_rate = 1.5 :=
by
  -- Note: The proof is not required, hence sorry is used.
  sorry

end multiple_of_regular_rate_is_1_5

namespace remainder_when_divided_by_10

import Mathlib

theorem remainder_when_divided_by_10 : 
  (2468 * 7391 * 90523) % 10 = 4 :=
by
  sorry

end remainder_when_divided_by_10

namespace smallest_n_for_inequality

import Mathlib

theorem smallest_n_for_inequality (n : ℕ) : 5 + 3 * n > 300 ↔ n = 99 := by
  sorry

end smallest_n_for_inequality

namespace chelsea_sugar_bags

import Mathlib

variable (n : ℕ)

-- Defining the conditions as hypotheses
def initial_sugar : ℕ := 24
def remaining_sugar : ℕ := 21
def sugar_lost : ℕ := initial_sugar - remaining_sugar
def torn_bag_sugar : ℕ := 2 * sugar_lost

-- Define the statement to prove
theorem chelsea_sugar_bags :
  n = initial_sugar / torn_bag_sugar → n = 4 :=
by
  sorry

end chelsea_sugar_bags

namespace min_value_SN64_by_aN_is_17_over_2

import Mathlib

noncomputable def a_n (n : ℕ) : ℕ := 2 * n
noncomputable def S_n (n : ℕ) : ℕ := n^2 + n

theorem min_value_SN64_by_aN_is_17_over_2 :
  ∃ (n : ℕ), 2 ≤ n ∧ (a_2 = 4 ∧ S_10 = 110) →
  ((S_n n + 64) / a_n n) = 17 / 2 :=
by
  sorry

end min_value_SN64_by_aN_is_17_over_2

namespace minimum_n_minus_m

import Mathlib

noncomputable def f (x : Real) : Real :=
    (Real.sin x) * (Real.sin (x + Real.pi / 3)) - 1 / 4

theorem minimum_n_minus_m (m n : Real) (h : m < n) 
  (h_domain : ∀ x, m ≤ x ∧ x ≤ n → -1 / 2 ≤ f x ∧ f x ≤ 1 / 4) :
  n - m = 2 * Real.pi / 3 :=
by
  sorry

end minimum_n_minus_m

namespace calculate_product

import Mathlib

noncomputable def complex_number_r (r : ℂ) : Prop :=
r^6 = 1 ∧ r ≠ 1

theorem calculate_product (r : ℂ) (h₁ : r^6 = 1) (h₂ : r ≠ 1) :
  (r - 1) * (r^2 - 1) * (r^3 - 1) * (r^4 - 1) * (r^5 - 1) = 2 := 
sorry

end calculate_product

namespace max_k_value

import Mathlib

def maximum_k (k : ℕ) : ℕ := 2

theorem max_k_value
  (k : ℕ)
  (h1 : 2 * k + 1 ≤ 20)  -- Condition implicitly implied by having subsets of a 20-element set
  (h2 : ∀ (s t : Finset (Fin 20)), s.card = 7 → t.card = 7 → s ≠ t → (s ∩ t).card = k) : k ≤ maximum_k k := 
by {
  sorry
}

end max_k_value

namespace find_c

import Mathlib

noncomputable def f (x c : ℝ) := x * (x - c) ^ 2
noncomputable def f' (x c : ℝ) := 3 * x ^ 2 - 4 * c * x + c ^ 2
noncomputable def f'' (x c : ℝ) := 6 * x - 4 * c

theorem find_c (c : ℝ) : f' 2 c = 0 ∧ f'' 2 c < 0 → c = 6 :=
by {
  sorry
}

end find_c

namespace trapezoid_area

import Mathlib

-- Define the properties of the isosceles trapezoid
structure IsoscelesTrapezoid where
  leg : ℝ
  diagonal : ℝ
  longer_base : ℝ
  is_isosceles : True
  legs_equal : True

-- Provide the specific conditions of the problem
def trapezoid : IsoscelesTrapezoid := {
  leg := 40,
  diagonal := 50,
  longer_base := 60,
  is_isosceles := True.intro,
  legs_equal := True.intro
}

-- State the main theorem to translate the proof problem into Lean
theorem trapezoid_area (T : IsoscelesTrapezoid) : T = trapezoid →
  (∃ A : ℝ, A = (15000 - 2000 * Real.sqrt 11) / 9) :=
by
  intros h
  sorry

end trapezoid_area

namespace hcl_reaction

import Mathlib

theorem hcl_reaction
  (stoichiometry : ∀ (HCl NaHCO3 H2O CO2 NaCl : ℕ), HCl = NaHCO3 ∧ H2O = NaHCO3 ∧ CO2 = NaHCO3 ∧ NaCl = NaHCO3)
  (naHCO3_moles : ℕ)
  (reaction_moles : naHCO3_moles = 3) :
  ∃ (HCl_moles : ℕ), HCl_moles = naHCO3_moles :=
by
  sorry

end hcl_reaction

namespace total_cost_of_purchase

import Mathlib

theorem total_cost_of_purchase :
  let sandwich_cost := 3
  let soda_cost := 2
  let num_sandwiches := 5
  let num_sodas := 8
  let total_cost := (num_sandwiches * sandwich_cost) + (num_sodas * soda_cost)
  total_cost = 31 :=
by
  sorry

end total_cost_of_purchase

namespace DianasInitialSpeed

import Mathlib

open Nat

theorem DianasInitialSpeed
  (total_distance : ℕ)
  (initial_time : ℕ)
  (tired_speed : ℕ)
  (total_time : ℕ)
  (distance_when_tired : ℕ)
  (initial_distance : ℕ)
  (initial_speed : ℕ)
  (initial_hours : ℕ) :
  total_distance = 10 →
  initial_time = 2 →
  tired_speed = 1 →
  total_time = 6 →
  distance_when_tired = tired_speed * (total_time - initial_time) →
  initial_distance = total_distance - distance_when_tired →
  initial_distance = initial_speed * initial_time →
  initial_speed = 3 := by
  sorry

end DianasInitialSpeed

namespace alice_wins_chomp

import Mathlib

def symmetrical_strategy (n : ℕ) : Prop :=
  ∃ strategy : (ℕ × ℕ) → (ℕ × ℕ), 
  (∀ turn : ℕ × ℕ, 
    strategy turn = 
      if turn = (1,1) then (1,1)
      else if turn.fst = 2 ∧ turn.snd = 2 then (2,2)
      else if turn.fst = 1 then (turn.snd, 1)
      else (1, turn.fst)) 

theorem alice_wins_chomp (n : ℕ) (h : 1 ≤ n) : 
  symmetrical_strategy n := 
sorry

end alice_wins_chomp

namespace negation_of_p

import Mathlib

variable {x : ℝ}

def p := ∀ x : ℝ, x^3 - x^2 + 1 < 0

theorem negation_of_p : ¬p ↔ ∃ x : ℝ, x^3 - x^2 + 1 ≥ 0 := by
  sorry

end negation_of_p

namespace determine_a

import Mathlib

def A := {x : ℝ | x < 6}
def B (a : ℝ) := {x : ℝ | x - a < 0}

theorem determine_a (a : ℝ) (h : A ⊆ B a) : 6 ≤ a := 
sorry

end determine_a

namespace proof_problem

import Mathlib

-- Given conditions: 
variables (a b c d : ℝ)
axiom condition : (2 * a + b) / (b + 2 * c) = (c + 3 * d) / (4 * d + a)

-- Proof problem statement:
theorem proof_problem : (a = c ∨ 3 * a + 4 * b + 5 * c + 6 * d = 0 ∨ (a = c ∧ 3 * a + 4 * b + 5 * c + 6 * d = 0)) :=
by
  sorry

end proof_problem

namespace ratio_sum_2_or_4

import Mathlib

theorem ratio_sum_2_or_4 (a b c d : ℝ) 
  (h1 : a / b + b / c + c / d + d / a = 6)
  (h2 : a / c + b / d + c / a + d / b = 8) : 
  (a / b + c / d = 2) ∨ (a / b + c / d = 4) :=
sorry

end ratio_sum_2_or_4

namespace transistors_in_2002_transistors_in_2010

import Mathlib

-- Definitions based on the conditions
def mooresLawDoubling (initial_transistors : ℕ) (years : ℕ) : ℕ :=
  initial_transistors * 2^(years / 2)

-- Conditions
def initial_transistors := 2000000
def year_1992 := 1992
def year_2002 := 2002
def year_2010 := 2010

-- Questions translated into proof targets
theorem transistors_in_2002 : mooresLawDoubling initial_transistors (year_2002 - year_1992) = 64000000 := by
  sorry

theorem transistors_in_2010 : mooresLawDoubling (mooresLawDoubling initial_transistors (year_2002 - year_1992)) (year_2010 - year_2002) = 1024000000 := by
  sorry

end transistors_in_2002_transistors_in_2010

namespace Maria_waist_size_correct

import Mathlib

noncomputable def waist_size_mm (waist_size_in : ℕ) (mm_per_ft : ℝ) (in_per_ft : ℕ) : ℝ :=
  (waist_size_in : ℝ) / (in_per_ft : ℝ) * mm_per_ft

theorem Maria_waist_size_correct :
  let waist_size_in := 27
  let mm_per_ft := 305
  let in_per_ft := 12
  waist_size_mm waist_size_in mm_per_ft in_per_ft = 686.3 :=
by
  sorry

end Maria_waist_size_correct

namespace largest_value_of_b

import Mathlib

theorem largest_value_of_b (b : ℚ) (h : (2 * b + 5) * (b - 1) = 6 * b) : b = 5 / 2 :=
by
  sorry

end largest_value_of_b

namespace B_profit

import Mathlib

-- Definitions based on conditions
def investment_ratio (B_invest A_invest : ℕ) : Prop := A_invest = 3 * B_invest
def period_ratio (B_period A_period : ℕ) : Prop := A_period = 2 * B_period
def total_profit (total : ℕ) : Prop := total = 28000
def B_share (total : ℕ) := total / 7

-- Theorem statement based on the proof problem
theorem B_profit (B_invest A_invest B_period A_period total : ℕ)
  (h1 : investment_ratio B_invest A_invest)
  (h2 : period_ratio B_period A_period)
  (h3 : total_profit total) :
  B_share total = 4000 :=
by
  sorry

end B_profit

namespace collinear_points

import Mathlib

-- Define collinear points function
def collinear (x1 y1 z1 x2 y2 z2 x3 y3 z3: ℝ) : Prop :=
  ∀ (a b c : ℝ), a * (y2 - y1) * (z3 - z1) + b * (z2 - z1) * (x3 - x1) + c * (x2 - x1) * (y3 - y1) = 0

-- Problem statement
theorem collinear_points (a b : ℝ)
  (h : collinear 2 a b a 3 b a b 4) :
  a + b = -2 :=
sorry

end collinear_points

namespace find_coefficients

import Mathlib

variables {V : Type*} [AddCommGroup V] [Module ℝ V]

-- Definitions based on conditions
def A' (A B : V) : V := (3 : ℝ) • (B - A) + A
def B' (B C : V) : V := (3 : ℝ) • (C - B) + C

-- The problem statement
theorem find_coefficients (A A' B B' : V) (p q r : ℝ) 
  (hB : B = (1/4 : ℝ) • A + (3/4 : ℝ) • A') 
  (hC : C = (1/4 : ℝ) • B + (3/4 : ℝ) • B') : 
  ∃ (p q r : ℝ), A = p • A' + q • B + r • B' ∧ p = 4/13 ∧ q = 12/13 ∧ r = 48/13 :=
sorry

end find_coefficients

namespace packs_sold_in_other_villages

import Mathlib

theorem packs_sold_in_other_villages
  (packs_v1 : ℕ) (packs_v2 : ℕ) (h1 : packs_v1 = 23) (h2 : packs_v2 = 28) :
  packs_v1 + packs_v2 = 51 := 
by {
  sorry
}

end packs_sold_in_other_villages

namespace river_depth_mid_June

import Mathlib

theorem river_depth_mid_June (D : ℝ) : 
    (∀ (mid_May mid_June mid_July : ℝ),
    mid_May = 5 →
    mid_June = mid_May + D →
    mid_July = 3 * mid_June →
    mid_July = 45) →
    D = 10 :=
by
    sorry

end river_depth_mid_June

namespace mia_study_time

import Mathlib

theorem mia_study_time 
  (T : ℕ)
  (watching_tv_exercise_social_media : T = 1440 ∧ 
    ∃ study_time : ℚ, 
    (study_time = (1 / 4) * 
      (((27 / 40) * T - (9 / 80) * T) / 
        (T * 1 / 40 - (1 / 5) * T - (1 / 8) * T))
    )) :
  T = 1440 → study_time = 202.5 := 
by
  sorry

end mia_study_time

namespace product_remainder

import Mathlib

-- Define the product of the consecutive numbers
def product := 86 * 87 * 88 * 89 * 90 * 91 * 92

-- Lean statement to state the problem
theorem product_remainder :
  product % 7 = 0 :=
by
  sorry

end product_remainder

namespace expression_value

import Mathlib

noncomputable def expr := (1.90 * (1 / (1 - (3: ℝ)^(1/4)))) + (1 / (1 + (3: ℝ)^(1/4))) + (2 / (1 + (3: ℝ)^(1/2)))

theorem expression_value : expr = -2 := 
by
  sorry

end expression_value

namespace problem_statement

import Mathlib  -- Import Mathlib to handle mathematical operations

theorem problem_statement : (-1:ℤ) ^ 4 - (2 - (-3:ℤ) ^ 2) = 6 := by
  sorry  -- Proof will be provided separately

end problem_statement

namespace marble_solid_color_percentage

import Mathlib

theorem marble_solid_color_percentage (a b : ℕ) (h1 : a = 5) (h2 : b = 85) : a + b = 90 := 
by
  sorry

end marble_solid_color_percentage

namespace rosie_pie_count

import Mathlib

def total_apples : ℕ := 40
def initial_apples_required : ℕ := 3
def apples_per_pie : ℕ := 5

theorem rosie_pie_count : (total_apples - initial_apples_required) / apples_per_pie = 7 :=
by
  sorry

end rosie_pie_count

namespace probability_at_least_one_consonant

import Mathlib

def letters := ["k", "h", "a", "n", "t", "k", "a", "r"]
def consonants := ["k", "h", "n", "t", "r"]
def vowels := ["a", "a"]

def num_letters := 7
def num_consonants := 5
def num_vowels := 2

def probability_no_consonants : ℚ := (num_vowels / num_letters) * ((num_vowels - 1) / (num_letters - 1))

def complement_rule (p: ℚ) : ℚ := 1 - p

theorem probability_at_least_one_consonant :
  complement_rule probability_no_consonants = 20/21 :=
by
  sorry

end probability_at_least_one_consonant

namespace sum_of_first_n_natural_numbers

import Mathlib

theorem sum_of_first_n_natural_numbers (n : ℕ) (h : n * (n + 1) / 2 = 190) : n = 19 :=
sorry

end sum_of_first_n_natural_numbers

namespace sum_abs_a1_to_a10

import Mathlib

def S (n : ℕ) : ℤ := n^2 - 4 * n + 2
def a (n : ℕ) : ℤ := if n = 1 then S 1 else S n - S (n - 1)

theorem sum_abs_a1_to_a10 : (|a 1| + |a 2| + |a 3| + |a 4| + |a 5| + |a 6| + |a 7| + |a 8| + |a 9| + |a 10| = 66) := 
by
  sorry

end sum_abs_a1_to_a10

namespace kim_total_ounces

import Mathlib

def quarts_to_ounces (q : ℚ) : ℚ := q * 32

def bottle_quarts : ℚ := 1.5
def can_ounces : ℚ := 12
def bottle_ounces : ℚ := quarts_to_ounces bottle_quarts

def total_ounces : ℚ := bottle_ounces + can_ounces

theorem kim_total_ounces : total_ounces = 60 :=
by
  -- Proof will go here
  sorry

end kim_total_ounces

namespace actual_cost_of_article

import Mathlib

theorem actual_cost_of_article (x : ℝ) (h : 0.76 * x = 760) : x = 1000 :=
by 
  sorry

end actual_cost_of_article

namespace correct_option_is_c

import Mathlib

variable {x y : ℕ}

theorem correct_option_is_c (hx : (x^2)^3 = x^6) :
  (∀ x : ℕ, x * x^2 ≠ x^2) →
  (∀ x y : ℕ, (x + y)^2 ≠ x^2 + y^2) →
  (∃ x : ℕ, x^2 + x^2 ≠ x^4) →
  (x^2)^3 = x^6 :=
by
  intros h1 h2 h3
  exact hx

end correct_option_is_c

namespace num_positive_k_for_solution_to_kx_minus_18_eq_3k

import Mathlib

theorem num_positive_k_for_solution_to_kx_minus_18_eq_3k : 
  ∃ (k_vals : Finset ℕ), 
  (∀ k ∈ k_vals, ∃ x : ℤ, k * x - 18 = 3 * k) ∧ 
  k_vals.card = 6 :=
by
  sorry

end num_positive_k_for_solution_to_kx_minus_18_eq_3k

namespace determinant_of_given_matrix

import Mathlib

noncomputable def given_matrix : Matrix (Fin 4) (Fin 4) ℤ :=
![![1, -3, 3, 2], ![0, 5, -1, 0], ![4, -2, 1, 0], ![0, 0, 0, 6]]

theorem determinant_of_given_matrix :
  Matrix.det given_matrix = -270 := by
  sorry

end determinant_of_given_matrix

namespace balloon_ratio

import Mathlib

theorem balloon_ratio 
  (initial_blue : ℕ) (initial_purple : ℕ) (balloons_left : ℕ)
  (h1 : initial_blue = 303)
  (h2 : initial_purple = 453)
  (h3 : balloons_left = 378) :
  (balloons_left / (initial_blue + initial_purple) : ℚ) = (1 / 2 : ℚ) :=
by
  sorry

end balloon_ratio

namespace number_of_factors_and_perfect_square_factors

import Mathlib

open Nat

-- Define the number 1320 and its prime factorization.
def n : ℕ := 1320
def prime_factors : List (ℕ × ℕ) := [(2, 2), (3, 1), (5, 1), (11, 1)]

-- Define a function to count factors.
def count_factors (factors : List (ℕ × ℕ)) : ℕ :=
  factors.foldl (λ acc ⟨_, exp⟩ => acc * (exp + 1)) 1

-- Define a function to count perfect square factors.
def count_perfect_square_factors (factors : List (ℕ × ℕ)) : ℕ :=
  factors.foldl (λ acc ⟨prime, exp⟩ => acc * (if exp % 2 == 0 then exp / 2 + 1 else 1)) 1

theorem number_of_factors_and_perfect_square_factors :
  count_factors prime_factors = 24 ∧ count_perfect_square_factors prime_factors = 2 :=
by
  sorry

end number_of_factors_and_perfect_square_factors

namespace selection_methods_eq_total_students

import Mathlib

def num_boys := 36
def num_girls := 28
def total_students : ℕ := num_boys + num_girls

theorem selection_methods_eq_total_students :
    total_students = 64 :=
by
  -- Placeholder for the proof
  sorry

end selection_methods_eq_total_students

namespace cement_tesss_street

import Mathlib

-- Definitions of the given conditions
def cement_lexis_street : ℝ := 10
def total_cement_used : ℝ := 15.1

-- Proof statement to show the amount of cement used to pave Tess's street
theorem cement_tesss_street : total_cement_used - cement_lexis_street = 5.1 :=
by 
  -- Add proof steps to show the theorem is valid.
  sorry

end cement_tesss_street

namespace percent_relation

import Mathlib

theorem percent_relation (x y z w : ℝ) (h1 : x = 1.25 * y) (h2 : y = 0.40 * z) (h3 : z = 1.10 * w) :
  (x / w) * 100 = 55 := by sorry

end percent_relation

namespace vector_dot_product

import Mathlib

-- Definitions of the vectors
def a : ℝ × ℝ := (2, 1)
def b : ℝ × ℝ := (-1, 2)

-- Definition of the dot product for 2D vectors
def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

-- Main statement to prove
theorem vector_dot_product :
  dot_product (a.1 + b.1, a.2 + b.2) (a.1 - b.1, a.2 - b.2) = 0 :=
by
  sorry

end vector_dot_product

namespace expected_pairs_of_adjacent_face_cards_is_44_over_17

import Mathlib
noncomputable def expected_adjacent_face_card_pairs : ℚ :=
  12 * (11 / 51)

theorem expected_pairs_of_adjacent_face_cards_is_44_over_17 :
  expected_adjacent_face_card_pairs = 44 / 17 :=
by
  sorry

end expected_pairs_of_adjacent_face_cards_is_44_over_17

namespace algebra_simplification

import Mathlib

theorem algebra_simplification (a b : ℤ) (h : ∀ x : ℤ, x^2 - 6 * x + b = (x - a)^2 - 1) : b - a = 5 := by
  sorry

end algebra_simplification

namespace solution_A

import Mathlib

def P : Set ℕ := {1, 2, 3, 4}

theorem solution_A (A : Set ℕ) (h1 : A ⊆ P) 
  (h2 : ∀ x ∈ A, 2 * x ∉ A) 
  (h3 : ∀ x ∈ (P \ A), 2 * x ∉ (P \ A)): 
    A = {2} ∨ A = {1, 4} ∨ A = {2, 3} ∨ A = {1, 3, 4} :=
sorry

end solution_A

namespace inscribed_cube_volume

import Mathlib

noncomputable def side_length_of_inscribed_cube (d : ℝ) : ℝ :=
d / Real.sqrt 3

noncomputable def volume_of_inscribed_cube (s : ℝ) : ℝ :=
s^3

theorem inscribed_cube_volume :
  (volume_of_inscribed_cube (side_length_of_inscribed_cube 12)) = 192 * Real.sqrt 3 :=
by
  sorry

end inscribed_cube_volume

namespace range_of_real_number_m

import Mathlib

open Set

variable {m : ℝ}

theorem range_of_real_number_m (U : Set ℝ) (A : Set ℝ) (B : Set ℝ) (h1 : U = univ) (h2 : A = { x | x < 1 }) (h3 : B = { x | x ≥ m }) (h4 : compl A ⊆ B) : m ≤ 1 := by
  sorry

end range_of_real_number_m

namespace smallest_number_is_27

import Mathlib

theorem smallest_number_is_27 (a b c : ℕ) (h_mean : (a + b + c) / 3 = 30) (h_median : b = 28) (h_largest : c = b + 7) : a = 27 :=
by {
  sorry
}

end smallest_number_is_27

namespace at_least_two_consecutive_heads_probability

import Mathlib

noncomputable def probability_at_least_two_consecutive_heads : ℚ := 
  let total_outcomes := 16
  let unfavorable_outcomes := 8
  1 - (unfavorable_outcomes / total_outcomes)

theorem at_least_two_consecutive_heads_probability :
  probability_at_least_two_consecutive_heads = 1 / 2 := 
by
  sorry

end at_least_two_consecutive_heads_probability

namespace ratio_of_volume_to_surface_area

import Mathlib

-- Definitions of the given conditions
def unit_cube_volume : ℕ := 1
def total_cubes : ℕ := 8
def volume := total_cubes * unit_cube_volume
def exposed_faces (center_cube_faces : ℕ) (side_cube_faces : ℕ) (top_cube_faces : ℕ) : ℕ :=
  center_cube_faces + 6 * side_cube_faces + top_cube_faces
def surface_area := exposed_faces 1 5 5
def ratio := volume / surface_area

-- The main theorem statement
theorem ratio_of_volume_to_surface_area : ratio = 2 / 9 := by
  sorry

end ratio_of_volume_to_surface_area

namespace part_I_part_II

import Mathlib

noncomputable
def x₀ : ℝ := 2

noncomputable
def f (x m : ℝ) : ℝ := |x - m| + |x + 1/m| - x₀

theorem part_I (x : ℝ) : |x + 3| - 2 * x - 1 < 0 ↔ x > 2 :=
by sorry

theorem part_II (m : ℝ) (h : m > 0) :
  (∃ x : ℝ, f x m = 0) → m = 1 :=
by sorry

end part_I_part_II

namespace compare_trig_values

import Mathlib

noncomputable def a : ℝ := Real.sin (2 * Real.pi / 7)
noncomputable def b : ℝ := Real.tan (5 * Real.pi / 7)
noncomputable def c : ℝ := Real.cos (5 * Real.pi / 7)

theorem compare_trig_values :
  (0 < 2 * Real.pi / 7 ∧ 2 * Real.pi / 7 < Real.pi / 2) →
  (Real.pi / 2 < 5 * Real.pi / 7 ∧ 5 * Real.pi / 7 < 3 * Real.pi / 4) →
  b < c ∧ c < a :=
by
  intro h1 h2
  sorry

end compare_trig_values

namespace polynomial_q_correct

import Mathlib

noncomputable def polynomial_q (x : ℝ) : ℝ :=
  -x^6 + 12*x^5 + 9*x^4 + 14*x^3 - 5*x^2 + 17*x + 1

noncomputable def polynomial_rhs (x : ℝ) : ℝ :=
  x^6 + 12*x^5 + 13*x^4 + 14*x^3 + 17*x + 3

noncomputable def polynomial_2 (x : ℝ) : ℝ :=
  2*x^6 + 4*x^4 + 5*x^2 + 2

theorem polynomial_q_correct (x : ℝ) : 
  polynomial_q x = polynomial_rhs x - polynomial_2 x := 
by
  sorry

end polynomial_q_correct

namespace sqrt_57_in_range

import Mathlib

theorem sqrt_57_in_range (h1 : 49 < 57) (h2 : 57 < 64) (h3 : 7^2 = 49) (h4 : 8^2 = 64) : 7 < Real.sqrt 57 ∧ Real.sqrt 57 < 8 := by
  sorry

end sqrt_57_in_range

namespace yards_gained

import Mathlib

variable {G : ℤ}

theorem yards_gained (h : -5 + G = 3) : G = 8 :=
  by
  sorry

end yards_gained

namespace possible_items

import Mathlib

-- Mathematical definitions derived from the conditions.
def item_cost_kopecks (a : ℕ) : ℕ := 100 * a + 99
def total_cost_kopecks : ℕ := 20083

-- The theorem stating the possible number of items Kolya could have bought.
theorem possible_items (a n : ℕ) (hn : n * item_cost_kopecks a = total_cost_kopecks) :
  n = 17 ∨ n = 117 :=
sorry

end possible_items

namespace portion_of_work_done

import Mathlib

variable (P W : ℕ)

-- Given conditions
def work_rate_P (P W : ℕ) : ℕ := W / 16
def work_rate_2P (P W : ℕ) : ℕ := 2 * (work_rate_P P W)

-- Lean theorem
theorem portion_of_work_done (h : work_rate_2P P W * 4 = W / 2) : 
    work_rate_2P P W * 4 = W / 2 := 
by 
  sorry

end portion_of_work_done

namespace participants_in_robbery

import Mathlib

variables (A B V G : Prop)

theorem participants_in_robbery
  (h1 : ¬G → (B ∧ ¬A))
  (h2 : V → (¬A ∧ ¬B))
  (h3 : G → B)
  (h4 : B → (A ∨ V)) :
  A ∧ B ∧ G :=
by
  sorry

end participants_in_robbery

namespace ordered_pairs_count

import Mathlib

theorem ordered_pairs_count :
  (∃ (A B : ℕ), 0 < A ∧ 0 < B ∧ A % 2 = 0 ∧ B % 2 = 0 ∧ (A / 8) = (8 / B))
  → (∃ (n : ℕ), n = 5) :=
by {
  sorry
}

end ordered_pairs_count

namespace father_three_times_marika_in_year

import Mathlib

-- Define the given conditions as constants.
def marika_age_2004 : ℕ := 8
def father_age_2004 : ℕ := 32

-- Define the proof goal.
theorem father_three_times_marika_in_year :
  ∃ (x : ℕ), father_age_2004 + x = 3 * (marika_age_2004 + x) → 2004 + x = 2008 := 
by {
  sorry
}

end father_three_times_marika_in_year

namespace general_term_defines_sequence

import Mathlib

/-- Sequence definition -/
def seq (a : ℕ → ℚ) : Prop :=
  a 1 = 2 ∧ ∀ n, a (n + 1) = (2 * a n + 6) / (a n + 1)

/-- General term formula -/
def general_term (n : ℕ) : ℚ :=
  (3 * 4 ^ n + 2 * (-1) ^ n) / (4 ^ n - (-1) ^ n)

/-- Theorem stating that the general term formula defines the sequence -/
theorem general_term_defines_sequence : ∀ (a : ℕ → ℚ), seq a → ∀ n, a n = general_term n :=
by
  intros a h_seq n
  sorry

end general_term_defines_sequence

namespace percentage_more_than

import Mathlib

variable (P Q : ℝ)

-- P gets 20% more than Q
def getsMoreThan (P Q : ℝ) : Prop :=
  P = 1.20 * Q

-- Q gets 20% less than P
def getsLessThan (Q P : ℝ) : Prop :=
  Q = 0.80 * P

theorem percentage_more_than :
  getsLessThan Q P → getsMoreThan P Q := 
sorry

end percentage_more_than

namespace num_divisors_720

import Mathlib

-- Define the number 720 and its prime factorization
def n : ℕ := 720
def pf : List (ℕ × ℕ) := [(2, 4), (3, 2), (5, 1)]

-- Define the function to calculate the number of divisors from prime factorization
def num_divisors (pf : List (ℕ × ℕ)) : ℕ :=
  pf.foldr (λ p acc => acc * (p.snd + 1)) 1

-- Statement to prove
theorem num_divisors_720 : num_divisors pf = 30 :=
  by
  -- Placeholder for the actual proof
  sorry

end num_divisors_720

namespace highest_number_paper

import Mathlib

theorem highest_number_paper (n : ℕ) (h : (1 : ℝ) / n = 0.010526315789473684) : n = 95 :=
sorry

end highest_number_paper

namespace number_of_floors

import Mathlib

def hours_per_room : ℕ := 6
def hourly_rate : ℕ := 15
def total_earnings : ℕ := 3600
def rooms_per_floor : ℕ := 10

theorem number_of_floors : 
  (total_earnings / hourly_rate / hours_per_room) / rooms_per_floor = 4 := by
  sorry

end number_of_floors

namespace floor_e_sub_6_eq_neg_4

import Mathlib

theorem floor_e_sub_6_eq_neg_4 :
  (⌊(e:Real) - 6⌋ = -4) :=
by
  let h₁ : 2 < (e:Real) := sorry -- assuming e is the base of natural logarithms
  let h₂ : (e:Real) < 3 := sorry
  sorry

end floor_e_sub_6_eq_neg_4

namespace pen_tip_movement

import Mathlib

-- Definitions for the conditions
def condition_a := "Point movement becomes a line"
def condition_b := "Line movement becomes a surface"
def condition_c := "Surface movement becomes a solid"
def condition_d := "Intersection of surfaces results in a line"

-- The main statement we need to prove
theorem pen_tip_movement (phenomenon : String) : 
  phenomenon = "the pen tip quickly sliding on the paper to write the number 6" →
  condition_a = "Point movement becomes a line" :=
by
  intros
  sorry

end pen_tip_movement

namespace max_area_right_triangle_in_semicircle

import Mathlib

theorem max_area_right_triangle_in_semicircle :
  ∀ (r : ℝ), r = 1/2 → 
  ∃ (x y : ℝ), x^2 + y^2 = r^2 ∧ y > 0 ∧ 
  (∀ (x' y' : ℝ), x'^2 + y'^2 = r^2 ∧ y' > 0 → (1/2) * x * y ≥ (1/2) * x' * y') ∧ 
  (1/2) * x * y = 3 * Real.sqrt 3 / 32 := 
sorry

end max_area_right_triangle_in_semicircle

namespace syllogism_correct

import Mathlib

theorem syllogism_correct 
  (natnum : ℕ → Prop) 
  (intnum : ℤ → Prop) 
  (is_natnum  : natnum 4) 
  (natnum_to_intnum : ∀ n, natnum n → intnum n) : intnum 4 :=
by
  sorry

end syllogism_correct

namespace fractions_sum_correct

import Mathlib

noncomputable def fractions_sum : ℝ := (3 / 20) + (5 / 200) + (7 / 2000) + 5

theorem fractions_sum_correct : fractions_sum = 5.1785 :=
by
  sorry

end fractions_sum_correct

namespace union_of_A_B_complement_intersection

import Mathlib

def U : Set ℝ := Set.univ

def A : Set ℝ := { x | -x^2 + 2*x + 15 ≤ 0 }

def B : Set ℝ := { x | |x - 5| < 1 }

theorem union_of_A_B :
  A ∪ B = { x | x ≤ -3 ∨ x > 4 } :=
by
  sorry

theorem complement_intersection :
  (U \ A) ∩ B = { x | 4 < x ∧ x < 5 } :=
by
  sorry

end union_of_A_B_complement_intersection

namespace nonnegative_integer_with_divisors_is_multiple_of_6

import Mathlib

-- Definitions as per conditions in (a)
def has_two_distinct_divisors_with_distance (n : ℕ) : Prop := ∃ d1 d2 : ℕ,
  d1 ≠ d2 ∧ d1 ∣ n ∧ d2 ∣ n ∧
  (d1:ℚ) - n / 3 = n / 3 - (d2:ℚ)

-- Main statement to prove as derived in (c)
theorem nonnegative_integer_with_divisors_is_multiple_of_6 (n : ℕ) :
  n > 0 ∧ has_two_distinct_divisors_with_distance n → ∃ k : ℕ, n = 6 * k :=
by
  sorry

end nonnegative_integer_with_divisors_is_multiple_of_6

namespace blocks_before_jess_turn

import Mathlib

def blocks_at_start : Nat := 54
def players : Nat := 5
def rounds : Nat := 5
def father_removes_block_in_6th_round : Nat := 1

theorem blocks_before_jess_turn :
    (blocks_at_start - (players * rounds + father_removes_block_in_6th_round)) = 28 :=
by 
    sorry

end blocks_before_jess_turn

namespace intersection_complement

import Mathlib

-- Definitions of the sets
def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 2}
def B : Set ℕ := {2, 3}

-- The complement of B in U
def complement_U (U B : Set ℕ) : Set ℕ := U \ B

-- Statement to prove
theorem intersection_complement : A ∩ (complement_U U B) = {1} := 
by 
  sorry

end intersection_complement

namespace bathroom_area_is_50_square_feet

import Mathlib

/-- A bathroom has 10 6-inch tiles along its width and 20 6-inch tiles along its length. --/
def bathroom_width_inches := 10 * 6
def bathroom_length_inches := 20 * 6

/-- Convert width and length from inches to feet. --/
def bathroom_width_feet := bathroom_width_inches / 12
def bathroom_length_feet := bathroom_length_inches / 12

/-- Calculate the square footage of the bathroom. --/
def bathroom_square_footage := bathroom_width_feet * bathroom_length_feet

/-- The square footage of the bathroom is 50 square feet. --/
theorem bathroom_area_is_50_square_feet : bathroom_square_footage = 50 := by
  sorry

end bathroom_area_is_50_square_feet

namespace find_x

import Mathlib

theorem find_x 
  (x : ℝ)
  (h : 120 + 80 + x + x = 360) : 
  x = 80 :=
sorry

end find_x

namespace not_taking_ship_probability

import Mathlib.Data.Real.Basic

-- Real non-negative numbers as probabilities
variables (P_train P_ship P_car P_airplane : ℝ)

-- Conditions
axiom h_train : 0 ≤ P_train ∧ P_train ≤ 1 ∧ P_train = 0.3
axiom h_ship : 0 ≤ P_ship ∧ P_ship ≤ 1 ∧ P_ship = 0.1
axiom h_car : 0 ≤ P_car ∧ P_car ≤ 1 ∧ P_car = 0.4
axiom h_airplane : 0 ≤ P_airplane ∧ P_airplane ≤ 1 ∧ P_airplane = 0.2

-- Prove that the probability of not taking a ship is 0.9
theorem not_taking_ship_probability : 1 - P_ship = 0.9 :=
by
  sorry

end not_taking_ship_probability

namespace polar_to_rectangular

import Mathlib

theorem polar_to_rectangular :
  let x := 16
  let y := 12
  let r := Real.sqrt (x^2 + y^2)
  let θ := Real.arctan (y / x)
  let new_r := 2 * r
  let new_θ := θ / 2
  let cos_half_θ := Real.sqrt ((1 + (x / r)) / 2)
  let sin_half_θ := Real.sqrt ((1 - (x / r)) / 2)
  let new_x := new_r * cos_half_θ
  let new_y := new_r * sin_half_θ
  new_x = 40 * Real.sqrt 0.9 ∧ new_y = 40 * Real.sqrt 0.1 := by
  sorry

end polar_to_rectangular

namespace remainder_when_divided_by_100

import Mathlib

theorem remainder_when_divided_by_100 (n : ℤ) (h : ∃ a : ℤ, n = 100 * a - 1) : 
  (n^3 + n^2 + 2 * n + 3) % 100 = 1 :=
by 
  sorry

end remainder_when_divided_by_100

namespace thabo_number_of_hardcover_nonfiction_books

import Mathlib

variables (P_f H_f P_nf H_nf A : ℕ)

theorem thabo_number_of_hardcover_nonfiction_books
  (h1 : P_nf = H_nf + 15)
  (h2 : H_f = P_f + 10)
  (h3 : P_f = 3 * A)
  (h4 : A + H_f = 70)
  (h5 : P_f + H_f + P_nf + H_nf + A = 250) :
  H_nf = 30 :=
by {
  sorry
}

end thabo_number_of_hardcover_nonfiction_books

namespace solve_for_y

import Mathlib

theorem solve_for_y (y : ℝ) (h : 6 * y^(1/3) - 3 * (y / y^(2/3)) = 10 + 2 * y^(1/3)) : y = 1000 := 
by
  sorry

end solve_for_y

namespace mans_rate_in_still_water

import Mathlib

-- Definitions from the conditions
def speed_with_stream : ℝ := 10
def speed_against_stream : ℝ := 6

-- The statement to prove the man's rate in still water is as expected.
theorem mans_rate_in_still_water : (speed_with_stream + speed_against_stream) / 2 = 8 := by
  sorry

end mans_rate_in_still_water

namespace total_potatoes

import Mathlib

theorem total_potatoes (monday_to_friday_potatoes : ℕ) (double_potatoes : ℕ) 
(lunch_potatoes_mon_fri : ℕ) (lunch_potatoes_weekend : ℕ)
(dinner_potatoes_mon_fri : ℕ) (dinner_potatoes_weekend : ℕ)
(h1 : monday_to_friday_potatoes = 5)
(h2 : double_potatoes = 10)
(h3 : lunch_potatoes_mon_fri = 25)
(h4 : lunch_potatoes_weekend = 20)
(h5 : dinner_potatoes_mon_fri = 40)
(h6 : dinner_potatoes_weekend = 26)
  : monday_to_friday_potatoes * 5 + double_potatoes * 2 + dinner_potatoes_mon_fri * 5 + (double_potatoes + 3) * 2 = 111 := 
sorry

end total_potatoes

namespace total_wheels_in_parking_lot

import Mathlib

def num_cars : ℕ := 14
def num_bikes : ℕ := 10
def wheels_per_car : ℕ := 4
def wheels_per_bike : ℕ := 2

theorem total_wheels_in_parking_lot :
  (num_cars * wheels_per_car) + (num_bikes * wheels_per_bike) = 76 :=
by
  sorry

end total_wheels_in_parking_lot

namespace range_of_a

import Mathlib

def A (x : ℝ) : Prop := (x - 1) * (x - 2) ≥ 0
def B (a x : ℝ) : Prop := x ≥ a

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, A x ∨ B a x) ↔ a ≤ 1 :=
sorry

end range_of_a

namespace common_chord_line

import Mathlib

-- Define the first circle equation
def circle1 (x y : ℝ) : Prop := x^2 + y^2 + 4*x - 6*y + 4 = 0

-- Define the second circle equation
def circle2 (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 4*y - 3 = 0

-- Definition of the line equation for the common chord
def line (x y : ℝ) : Prop := 2*x - 2*y + 7 = 0

theorem common_chord_line (x y : ℝ) (h1 : circle1 x y) (h2 : circle2 x y) : line x y :=
by
  sorry

end common_chord_line

namespace tom_found_dimes

import Mathlib

theorem tom_found_dimes :
  let quarters := 10
  let nickels := 4
  let pennies := 200
  let total_value := 5
  let value_quarters := 0.25 * quarters
  let value_nickels := 0.05 * nickels
  let value_pennies := 0.01 * pennies
  let total_other := value_quarters + value_nickels + value_pennies
  let value_dimes := total_value - total_other
  value_dimes / 0.10 = 3 := sorry

end tom_found_dimes

namespace sum_of_areas_is_correct

import Mathlib

/-- Define the lengths of the rectangles -/
def lengths : List ℕ := [4, 16, 36, 64, 100]

/-- Define the common base width of the rectangles -/
def base_width : ℕ := 3

/-- Define the area of a rectangle given its length and a common base width -/
def area (length : ℕ) : ℕ := base_width * length

/-- Compute the total area of the given rectangles -/
def total_area : ℕ := (lengths.map area).sum

/-- Theorem stating that the total area of the five rectangles is 660 -/
theorem sum_of_areas_is_correct : total_area = 660 := by
  sorry

end sum_of_areas_is_correct

namespace circumscribed_circle_radius

import Mathlib

noncomputable def radius_of_circumscribed_circle (a b : ℝ) : ℝ :=
  (Real.sqrt (a^2 + b^2)) / 2

theorem circumscribed_circle_radius (a r l b R : ℝ)
  (h1 : r = 1)
  (h2 : a = 2 * Real.sqrt 3)
  (h3 : b = 3)
  (h4 : l = a)
  (h5 : R = radius_of_circumscribed_circle l b) :
  R = Real.sqrt 21 / 2 :=
by
  sorry

end circumscribed_circle_radius

namespace 

import Mathlib

noncomputable def number_of_m_gons_with_two_acute_angles (m n : ℕ) (h1 : 4 < m) (h2 : m < n) : ℕ :=
  (2 * n + 1) * (Nat.choose (n + 1) (m - 1) + Nat.choose n (m - 1))

theorem number_of_m_gons_proof {m n : ℕ} (h1 : 4 < m) (h2 : m < n) :
  number_of_m_gons_with_two_acute_angles m n h1 h2 =
  (2 * n + 1) * ((Nat.choose (n + 1) (m - 1)) + (Nat.choose n (m - 1))) :=
sorry

end 

namespace sum_is_zero

import Mathlib

theorem sum_is_zero (a b c : ℝ) (h₀ : a ≠ 0) (h₁ : b ≠ 0) (h₂ : c ≠ 0) (h₃ : a + b + c = 0) :
  (a / |a|) + (b / |b|) + (c / |c|) + ((a * b * c) / |a * b * c|) = 0 :=
by
  sorry

end sum_is_zero

namespace sin_log_infinite_zeros_in_01

import Mathlib

noncomputable def f (x : ℝ) : ℝ := Real.sin (Real.log x)

theorem sin_log_infinite_zeros_in_01 : ∃ (S : Set ℝ), S = {x | 0 < x ∧ x < 1 ∧ f x = 0} ∧ Set.Infinite S := 
sorry

end sin_log_infinite_zeros_in_01

namespace imag_part_z_is_3

import Mathlib

namespace ComplexMultiplication

-- Define the imaginary unit i
def i := Complex.I

-- Define the complex number z
def z := (1 + 2 * i) * (2 - i)

-- Define the imaginary part of a complex number
def imag_part (z : ℂ) : ℂ := Complex.im z

-- Statement to prove: The imaginary part of z = 3
theorem imag_part_z_is_3 : imag_part z = 3 := by
  sorry

end ComplexMultiplication

end imag_part_z_is_3

namespace cube_mod_35_divisors

import Mathlib

theorem cube_mod_35_divisors (a : ℤ) : (35 ∣ a^3 - 1) ↔
  (∃ k : ℤ, a = 35 * k + 1) ∨ 
  (∃ k : ℤ, a = 35 * k + 11) ∨ 
  (∃ k : ℤ, a = 35 * k + 16) :=
by sorry

end cube_mod_35_divisors

namespace largest_fraction

import Mathlib

theorem largest_fraction (a b c d e : ℝ) (h1 : 1 < a) (h2 : a < b) (h3 : b < c) (h4 : c < d) (h5 : d < e) :
  (b + d + e) / (a + c) > max ((a + b + e) / (c + d))
                        (max ((a + d) / (b + e))
                            (max ((b + c) / (a + e)) ((c + e) / (a + b + d)))) := 
sorry

end largest_fraction

namespace ordered_triple_unique

import Mathlib

theorem ordered_triple_unique (a b c : ℝ) (h2 : a > 2) (h3 : b > 2) (h4 : c > 2)
    (h : (a + 3)^2 / (b + c - 3) + (b + 5)^2 / (c + a - 5) + (c + 7)^2 / (a + b - 7) = 49) :
    a = 7 ∧ b = 5 ∧ c = 3 :=
sorry

end ordered_triple_unique

namespace pyramid_x_value

import Mathlib

theorem pyramid_x_value (x y : ℝ) 
  (h1 : 150 = 10 * x)
  (h2 : 225 = x * 15)
  (h3 : 1800 = 150 * y * 225) :
  x = 15 :=
sorry

end pyramid_x_value

namespace radius_of_cookie

import Mathlib

theorem radius_of_cookie (x y : ℝ) : 
  (x^2 + y^2 + x - 5 * y = 10) → 
  ∃ r, (r = Real.sqrt (33 / 2)) :=
by
  sorry

end radius_of_cookie

namespace arithmetic_mean_is_correct

import Mathlib

variable (x a : ℝ)
variable (hx : x ≠ 0)

theorem arithmetic_mean_is_correct : 
  (1/2 * ((x + 2 * a) / x - 1 + (x - 3 * a) / x + 1)) = (1 - a / (2 * x)) := 
  sorry

end arithmetic_mean_is_correct

namespace cricket_average_increase

import Mathlib

theorem cricket_average_increase :
  ∀ (x : ℝ), (11 * (33 + x) = 407) → (x = 4) :=
  by 
  intros x hx
  sorry

end cricket_average_increase

namespace binom_18_6

import Mathlib

def binomial (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem binom_18_6 : binomial 18 6 = 18564 := 
by
  sorry

end binom_18_6

namespace parallelogram_area

import Mathlib

def base : ℕ := 34
def height : ℕ := 18
def area_of_parallelogram (b h : ℕ) : ℕ := b * h

theorem parallelogram_area : area_of_parallelogram base height = 612 := by
  sorry

end parallelogram_area

namespace man_work_days

import Mathlib

theorem man_work_days (M : ℕ) (h1 : (1 : ℝ)/M + (1 : ℝ)/10 = 1/5) : M = 10 :=
sorry

end man_work_days

namespace n_squared_divides_2n_plus_1

import Mathlib

theorem n_squared_divides_2n_plus_1 (n : ℕ) (hn : n > 0) :
  (n ^ 2) ∣ (2 ^ n + 1) ↔ (n = 1 ∨ n = 3) :=
by sorry

end n_squared_divides_2n_plus_1

namespace estimate_y_value

import Mathlib

theorem estimate_y_value : 
  ∀ (x : ℝ), x = 25 → 0.50 * x - 0.81 = 11.69 :=
by 
  intro x h
  rw [h]
  norm_num


end estimate_y_value

namespace three_digit_numbers_mod_1000

import Mathlib

theorem three_digit_numbers_mod_1000 (n : ℕ) (h_lower : 100 ≤ n) (h_upper : n ≤ 999) : 
  (n^2 ≡ n [MOD 1000]) ↔ (n = 376 ∨ n = 625) :=
by sorry

end three_digit_numbers_mod_1000

namespace ben_eggs_left

import Mathlib

def initial_eggs : ℕ := 50
def day1_morning : ℕ := 5
def day1_afternoon : ℕ := 4
def day2_morning : ℕ := 8
def day2_evening : ℕ := 3
def day3_afternoon : ℕ := 6
def day3_night : ℕ := 2

theorem ben_eggs_left : initial_eggs - (day1_morning + day1_afternoon + day2_morning + day2_evening + day3_afternoon + day3_night) = 22 := 
by
  sorry

end ben_eggs_left

namespace find_angle_A

import Mathlib

-- Conditions
def is_triangle (A B C : ℝ) : Prop := A + B + C = 180
def B_is_two_C (B C : ℝ) : Prop := B = 2 * C
def B_is_80 (B : ℝ) : Prop := B = 80

-- Theorem statement
theorem find_angle_A (A B C : ℝ) (h₁ : is_triangle A B C) (h₂ : B_is_two_C B C) (h₃ : B_is_80 B) : A = 60 := by
  sorry

end find_angle_A

namespace parrots_count

import Mathlib

theorem parrots_count (p r : ℕ) : 2 * p + 4 * r = 26 → p + r = 10 → p = 7 := by
  intros h1 h2
  sorry

end parrots_count

namespace john_needs_packs

import Mathlib

-- Definitions based on conditions
def utensils_per_pack : Nat := 30
def utensils_types : Nat := 3
def spoons_per_pack : Nat := utensils_per_pack / utensils_types
def spoons_needed : Nat := 50

-- Statement to prove
theorem john_needs_packs : (50 / spoons_per_pack) = 5 :=
by
  -- To complete the proof
  sorry

end john_needs_packs

namespace thabo_books

import Mathlib

theorem thabo_books (H P F : ℕ) (h1 : P = H + 20) (h2 : F = 2 * P) (h3 : H + P + F = 280) : H = 55 :=
by
  sorry

end thabo_books

namespace necessary_and_sufficient_condition

import Mathlib

-- Define the first circle
def circle1 (m : ℝ) : Set (ℝ × ℝ) :=
  { p | (p.1 + m)^2 + p.2^2 = 1 }

-- Define the second circle
def circle2 : Set (ℝ × ℝ) :=
  { p | (p.1 - 2)^2 + p.2^2 = 4 }

-- Define the condition -1 ≤ m ≤ 1
def condition (m : ℝ) : Prop :=
  -1 ≤ m ∧ m ≤ 1

-- Define the property for circles having common points
def circlesHaveCommonPoints (m : ℝ) : Prop :=
  ∃ p : ℝ × ℝ, p ∈ circle1 m ∧ p ∈ circle2

-- The final statement
theorem necessary_and_sufficient_condition (m : ℝ) :
  condition m → circlesHaveCommonPoints m ↔ (-5 ≤ m ∧ m ≤ 1) :=
by
  sorry

end necessary_and_sufficient_condition

namespace muscovy_more_than_cayuga

import Mathlib

theorem muscovy_more_than_cayuga
  (M C K : ℕ)
  (h1 : M + C + K = 90)
  (h2 : M = 39)
  (h3 : M = 2 * C + 3 + C) :
  M - C = 27 := by
  sorry

end muscovy_more_than_cayuga

namespace jill_spent_more

import Mathlib

def cost_per_ball_red : ℝ := 1.50
def cost_per_ball_yellow : ℝ := 1.25
def cost_per_ball_blue : ℝ := 1.00

def packs_red : ℕ := 5
def packs_yellow : ℕ := 4
def packs_blue : ℕ := 3

def balls_per_pack_red : ℕ := 18
def balls_per_pack_yellow : ℕ := 15
def balls_per_pack_blue : ℕ := 12

def balls_red : ℕ := packs_red * balls_per_pack_red
def balls_yellow : ℕ := packs_yellow * balls_per_pack_yellow
def balls_blue : ℕ := packs_blue * balls_per_pack_blue

def cost_red : ℝ := balls_red * cost_per_ball_red
def cost_yellow : ℝ := balls_yellow * cost_per_ball_yellow
def cost_blue : ℝ := balls_blue * cost_per_ball_blue

def combined_cost_yellow_blue : ℝ := cost_yellow + cost_blue

theorem jill_spent_more : cost_red = combined_cost_yellow_blue + 24 := by
  sorry

end jill_spent_more

namespace minimum_seats_occupied

import Mathlib

theorem minimum_seats_occupied (total_seats : ℕ) (h : total_seats = 180) : 
  ∃ occupied_seats : ℕ, occupied_seats = 45 ∧ 
  ∀ additional_person,
    (∀ i : ℕ, i < total_seats → 
     (occupied_seats ≤ i → i < occupied_seats + 1 ∨ i > occupied_seats + 1)) →
    additional_person = occupied_seats + 1  :=
by
  sorry

end minimum_seats_occupied

namespace find_cost_prices

import Mathlib

-- These represent the given selling prices of the items.
def SP_computer_table : ℝ := 3600
def SP_office_chair : ℝ := 5000
def SP_bookshelf : ℝ := 1700

-- These represent the percentage markups and discounts as multipliers.
def markup_computer_table : ℝ := 1.20
def markup_office_chair : ℝ := 1.25
def discount_bookshelf : ℝ := 0.85

-- The problem requires us to find the cost prices. We will define these as variables.
variable (C O B : ℝ)

theorem find_cost_prices :
  (SP_computer_table = C * markup_computer_table) ∧
  (SP_office_chair = O * markup_office_chair) ∧
  (SP_bookshelf = B * discount_bookshelf) →
  (C = 3000) ∧ (O = 4000) ∧ (B = 2000) :=
by
  sorry

end find_cost_prices

namespace value_of_a_plus_b

import Mathlib

theorem value_of_a_plus_b (a b : ℝ) : 
  (∀ x : ℝ, (x > -4 ∧ x < 1) ↔ (ax^2 + bx - 2 > 0)) → 
  a = 1/2 → 
  b = 3/2 → 
  a + b = 2 := 
by 
  intro h cond_a cond_b 
  rw [cond_a, cond_b]
  norm_num

end value_of_a_plus_b

namespace max_value_of_sample

import Mathlib

theorem max_value_of_sample 
  (x : Fin 5 → ℤ)
  (h_different : ∀ i j, i ≠ j → x i ≠ x j)
  (h_mean : (x 0 + x 1 + x 2 + x 3 + x 4) / 5 = 7)
  (h_variance : ((x 0 - 7)^2 + (x 1 - 7)^2 + (x 2 - 7)^2 + (x 3 - 7)^2 + (x 4 - 7)^2) / 5 = 4)
  : ∃ i, x i = 10 := 
sorry

end max_value_of_sample

namespace bella_truck_stamps_more

import Mathlib

def num_of_truck_stamps (T R : ℕ) : Prop :=
  11 + T + R = 38 ∧ R = T - 13

theorem bella_truck_stamps_more (T R : ℕ) (h : num_of_truck_stamps T R) : T - 11 = 9 := sorry

end bella_truck_stamps_more

namespace easter_eggs_problem

import Mathlib

noncomputable def mia_rate : ℕ := 24
noncomputable def billy_rate : ℕ := 10
noncomputable def total_hours : ℕ := 5
noncomputable def total_eggs : ℕ := 170

theorem easter_eggs_problem :
  (mia_rate + billy_rate) * total_hours = total_eggs :=
by
  sorry

end easter_eggs_problem

namespace sum_at_simple_interest

import Mathlib

theorem sum_at_simple_interest
  (P R : ℝ)  -- P is the principal amount, R is the rate of interest
  (H1 : (9 * P * (R + 5) / 100 - 9 * P * R / 100 = 1350)) :
  P = 3000 :=
by
  sorry

end sum_at_simple_interest

namespace exists_a

import Mathlib

noncomputable def a : ℕ → ℕ := sorry

theorem exists_a : a (a (a (a 1))) = 458329 :=
by
  -- proof skipped
  sorry

end exists_a

namespace water_consumption_and_bill_34_7

import Mathlib

noncomputable def calculate_bill (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x ≤ 1 then 20.8 * x
  else if 1 < x ∧ x ≤ (5 / 3) then 27.8 * x - 7
  else 32 * x - 14

theorem water_consumption_and_bill_34_7 (x : ℝ) :
  calculate_bill 1.5 = 34.7 ∧ 5 * 1.5 = 7.5 ∧ 3 * 1.5 = 4.5 ∧ 
  5 * 2.6 + (5 * 1.5 - 5) * 4 = 23 ∧ 
  4.5 * 2.6 = 11.7 :=
  sorry

end water_consumption_and_bill_34_7

namespace subtraction_and_multiplication_problem

import Mathlib

theorem subtraction_and_multiplication_problem :
  (5 / 6 - 1 / 3) * 3 / 4 = 3 / 8 :=
by sorry

end subtraction_and_multiplication_problem

namespace smallest_positive_multiple

import Mathlib

theorem smallest_positive_multiple (a : ℕ) (h : a > 0) : ∃ a > 0, (31 * a) % 103 = 7 := 
sorry

end smallest_positive_multiple

namespace simplify_fraction

import Mathlib

theorem simplify_fraction (b : ℕ) (hb : b = 5) : (15 * b^4) / (90 * b^3 * b) = 1 / 6 := by
  sorry

end simplify_fraction

namespace ashok_total_subjects

import Mathlib

variable (n : ℕ) (T : ℕ)

theorem ashok_total_subjects (h_ave_all : 75 * n = T + 80)
                       (h_ave_first : T = 74 * (n - 1)) :
  n = 6 := sorry

end ashok_total_subjects

namespace daily_avg_for_entire_month_is_correct

import Mathlib

-- conditions
def avg_first_25_days := 63
def days_first_25 := 25
def avg_last_5_days := 33
def days_last_5 := 5
def total_days := days_first_25 + days_last_5

-- question: What is the daily average for the entire month?
theorem daily_avg_for_entire_month_is_correct : 
  (avg_first_25_days * days_first_25 + avg_last_5_days * days_last_5) / total_days = 58 := by
  sorry

end daily_avg_for_entire_month_is_correct

namespace margaret_time

import Mathlib

def factorial (n : Nat) : Nat :=
  match n with
  | 0 => 1
  | n+1 => (n+1) * factorial n

def total_permutations (n : Nat) : Nat :=
  factorial n

def total_time_in_minutes (total_permutations : Nat) (rate : Nat) : Nat :=
  total_permutations / rate

def time_in_hours_and_minutes (total_minutes : Nat) : Nat × Nat :=
  let hours := total_minutes / 60
  let minutes := total_minutes % 60
  (hours, minutes)

theorem margaret_time :
  let n := 8
  let r := 15
  let permutations := total_permutations n
  let total_minutes := total_time_in_minutes permutations r
  time_in_hours_and_minutes total_minutes = (44, 48) := by
  sorry

end margaret_time

namespace symmetric_points_origin

import Mathlib

theorem symmetric_points_origin (a b : ℝ) (h : (1, 2) = (-a, -b)) : a = -1 ∧ b = -2 :=
sorry

end symmetric_points_origin

namespace cute_angle_of_isosceles_cute_triangle

import Mathlib

theorem cute_angle_of_isosceles_cute_triangle (A B C : ℝ) 
    (h1 : B = 2 * C)
    (h2 : A + B + C = 180)
    (h3 : A = B ∨ A = C) :
    A = 45 ∨ A = 72 :=
sorry

end cute_angle_of_isosceles_cute_triangle

namespace volume_of_convex_polyhedron

import Mathlib

variables {S1 S2 S : ℝ} {h : ℝ}

theorem volume_of_convex_polyhedron (S1 S2 S h : ℝ) :
  (h > 0) → (S1 ≥ 0) → (S2 ≥ 0) → (S ≥ 0) →
  ∃ V, V = (h / 6) * (S1 + S2 + 4 * S) :=
by {
  sorry
}

end volume_of_convex_polyhedron

namespace polygon_sides_eq_five

import Mathlib

def number_of_diagonals (n : ℕ) : ℕ := (n * (n - 3)) / 2

theorem polygon_sides_eq_five (n : ℕ) (h : n - number_of_diagonals n = 0) : n = 5 :=
by
  sorry

end polygon_sides_eq_five

namespace simplify_fraction_expression

import Mathlib

variable (d : ℤ)

theorem simplify_fraction_expression : (6 + 4 * d) / 9 + 3 = (33 + 4 * d) / 9 := by
  sorry

end simplify_fraction_expression

namespace sampling_method_D_is_the_correct_answer

import Mathlib

def sampling_method_A_is_simple_random_sampling : Prop :=
  false

def sampling_method_B_is_simple_random_sampling : Prop :=
  false

def sampling_method_C_is_simple_random_sampling : Prop :=
  false

def sampling_method_D_is_simple_random_sampling : Prop :=
  true

theorem sampling_method_D_is_the_correct_answer :
  sampling_method_A_is_simple_random_sampling = false ∧
  sampling_method_B_is_simple_random_sampling = false ∧
  sampling_method_C_is_simple_random_sampling = false ∧
  sampling_method_D_is_simple_random_sampling = true :=
by
  sorry

end sampling_method_D_is_the_correct_answer

namespace exercise

import Mathlib

noncomputable def f : ℝ → ℝ := sorry

axiom h1 : ∀ x, 0 ≤ x → x ≤ 1 → 0 ≤ f x ∧ f x ≤ 1
axiom h2 : ∀ x y : ℝ, 0 ≤ x → x ≤ 1 → 0 ≤ y → y ≤ 1 → f x + f y = f (f x + y)

theorem exercise : ∀ x, 0 ≤ x → x ≤ 1 → f (f x) = f x := 
by 
  sorry

end exercise

namespace third_part_of_division

import Mathlib

noncomputable def divide_amount (total_amount : ℝ) : (ℝ × ℝ × ℝ) :=
  let part1 := (1/2)/(1/2 + 2/3 + 3/4) * total_amount
  let part2 := (2/3)/(1/2 + 2/3 + 3/4) * total_amount
  let part3 := (3/4)/(1/2 + 2/3 + 3/4) * total_amount
  (part1, part2, part3)

theorem third_part_of_division :
  divide_amount 782 = (261.0, 214.66666666666666, 306.0) :=
by
  sorry

end third_part_of_division

namespace number_of_integers_with_three_divisors

import Mathlib

def has_exactly_three_positive_divisors (n : ℕ) : Prop :=
  ∃ p : ℕ, Nat.Prime p ∧ n = p * p

theorem number_of_integers_with_three_divisors (n : ℕ) :
  n = 2012 → Nat.card { x : ℕ | x ≤ n ∧ has_exactly_three_positive_divisors x } = 14 :=
by
  sorry

end number_of_integers_with_three_divisors

namespace maximize_tetrahedron_volume

import Mathlib

noncomputable def volume_of_tetrahedron (a : ℝ) : ℝ :=
  a / 6

theorem maximize_tetrahedron_volume (a : ℝ) (h_a : 0 < a) 
  (P Q X Y : ℝ × ℝ × ℝ) (h_PQ : dist P Q = 1) (h_XY : dist X Y = 1) :
  volume_of_tetrahedron a = a / 6 :=
by
  sorry

end maximize_tetrahedron_volume

namespace farmer_goats_sheep_unique_solution

import Mathlib

theorem farmer_goats_sheep_unique_solution:
  ∃ g h : ℕ, 0 < g ∧ 0 < h ∧ 28 * g + 30 * h = 1200 ∧ h > g :=
by
  sorry

end farmer_goats_sheep_unique_solution

namespace find_a

import Mathlib

theorem find_a (a : ℝ) : 
  (∀ (i : ℂ), i^2 = -1 → (a * i / (2 - i) + 1 = 2 * i)) → a = 5 :=
by
  intro h
  sorry

end find_a

namespace min_value_expr

import Mathlib

theorem min_value_expr (a b : ℝ) (h : a * b > 0) : (a^4 + 4 * b^4 + 1) / (a * b) ≥ 4 := 
sorry

end min_value_expr

namespace negation_of_prop

import Mathlib

theorem negation_of_prop :
  ¬(∀ x : ℝ, x^3 - x^2 + 1 > 0) ↔ ∃ x : ℝ, x^3 - x^2 + 1 ≤ 0 :=
by
  sorry

end negation_of_prop

namespace doubled_team_completes_half_in_three_days

import Mathlib

theorem doubled_team_completes_half_in_three_days
  (R : ℝ) -- Combined work rate of the original team
  (h : R * 12 = W) -- Original team completes the work W in 12 days
  (W : ℝ) : -- Total work to be done
  (2 * R) * 3 = W/2 := -- Doubled team completes half the work in 3 days
by 
  sorry

end doubled_team_completes_half_in_three_days

namespace magnesium_is_limiting

import Mathlib

-- Define the conditions
def moles_Mg : ℕ := 4
def moles_CO2 : ℕ := 2
def moles_O2 : ℕ := 2 -- represent excess O2, irrelevant to limiting reagent
def mag_ox_reaction (mg : ℕ) (o2 : ℕ) (mgo : ℕ) : Prop := 2 * mg + o2 = 2 * mgo
def mag_carbon_reaction (mg : ℕ) (co2 : ℕ) (mgco3 : ℕ) : Prop := mg + co2 = mgco3

-- Assume Magnesium is the limiting reagent for both reactions
theorem magnesium_is_limiting (mgo : ℕ) (mgco3 : ℕ) :
  mag_ox_reaction moles_Mg moles_O2 mgo ∧ mag_carbon_reaction moles_Mg moles_CO2 mgco3 →
  mgo = 4 ∧ mgco3 = 4 :=
by
  sorry

end magnesium_is_limiting

namespace total_meals_per_week

import Mathlib

-- Definitions for the conditions
def first_restaurant_meals := 20
def second_restaurant_meals := 40
def third_restaurant_meals := 50
def days_in_week := 7

-- The theorem for the total meals per week
theorem total_meals_per_week : 
  (first_restaurant_meals + second_restaurant_meals + third_restaurant_meals) * days_in_week = 770 := 
by
  sorry

end total_meals_per_week

namespace roots_polynomial_sum

import Mathlib

theorem roots_polynomial_sum :
  ∀ (p q r : ℂ), (p^3 - 3*p^2 - p + 3 = 0) ∧ (q^3 - 3*q^2 - q + 3 = 0) ∧ (r^3 - 3*r^2 - r + 3 = 0) →
  (1 / (p - 2) + 1 / (q - 2) + 1 / (r - 2) = 1) :=
by
  intros p q r h
  sorry

end roots_polynomial_sum

namespace overall_loss_is_450

import Mathlib

noncomputable def total_worth_stock : ℝ := 22499.999999999996

noncomputable def selling_price_20_percent_stock (W : ℝ) : ℝ :=
    0.20 * W * 1.10

noncomputable def selling_price_80_percent_stock (W : ℝ) : ℝ :=
    0.80 * W * 0.95

noncomputable def total_selling_price (W : ℝ) : ℝ :=
    selling_price_20_percent_stock W + selling_price_80_percent_stock W

noncomputable def overall_loss (W : ℝ) : ℝ :=
    W - total_selling_price W

theorem overall_loss_is_450 :
  overall_loss total_worth_stock = 450 := by
  sorry

end overall_loss_is_450

namespace number_of_real_b

import Mathlib

noncomputable def count_integer_roots_of_quadratic_eq_b : ℕ :=
  let pairs := [(1, 64), (2, 32), (4, 16), (8, 8), (-1, -64), (-2, -32), (-4, -16), (-8, -8)]
  pairs.length

theorem number_of_real_b : count_integer_roots_of_quadratic_eq_b = 8 :=
by {
  -- sorry is used to skip the proof
  sorry
}

end number_of_real_b

namespace value_of_v_3

import Mathlib

-- Defining the polynomial
def f (x : ℝ) : ℝ := x^6 - 5*x^5 + 6*x^4 + x^2 + 0.3*x + 2

-- Given evaluation point
def eval_point : ℝ := -2

-- Horner's method intermediate value v_3
def v_3_using_horner_method (x : ℝ) : ℝ :=
  let V0 := 1
  let V1 := x * V0 - 5
  let V2 := x * V1 + 6
  let V3 := x * V2 -- x^3 term is zero
  V3

-- Statement to prove
theorem value_of_v_3 :
  v_3_using_horner_method eval_point = -40 :=
by 
  -- Proof to be completed later
  sorry

end value_of_v_3

namespace hyperbola_proof

import Mathlib

noncomputable def hyperbola_equation (x y : ℝ) : Prop :=
  y^2 / 16 - x^2 / 4 = 1

def hyperbola_conditions (origin : ℝ × ℝ) (eccentricity : ℝ) (radius : ℝ) (focus : ℝ × ℝ) : Prop :=
  origin = (0, 0) ∧
  focus.1 = 0 ∧
  eccentricity = Real.sqrt 5 / 2 ∧
  radius = 2

theorem hyperbola_proof :
  ∃ (C : ℝ → ℝ → Prop),
    (∀ (x y : ℝ), hyperbola_conditions (0, 0) (Real.sqrt 5 / 2) 2 (0, c) → 
    C x y ↔ hyperbola_equation x y) :=
by
  sorry

end hyperbola_proof

namespace find_pairs

import Mathlib

theorem find_pairs (a b : ℕ) (q r : ℕ) (h1 : a > 0) (h2 : b > 0)
  (h3 : a^2 + b^2 = q * (a + b) + r) (h4 : 0 ≤ r) (h5 : r < a + b)
  (h6 : q^2 + r = 1977) :
  (a, b) = (50, 37) ∨ (a, b) = (50, 7) ∨ (a, b) = (37, 50) ∨ (a, b) = (7, 50) :=
  sorry

end find_pairs

namespace avg_speed_is_40

import Mathlib

noncomputable def average_speed (x : ℝ) : ℝ :=
  let time1 := x / 40
  let time2 := 2 * x / 20
  let total_time := time1 + time2
  let total_distance := 5 * x
  total_distance / total_time

theorem avg_speed_is_40 (x : ℝ) (hx : x > 0) :
  average_speed x = 40 := by
  sorry

end avg_speed_is_40

namespace A_share_of_annual_gain

import Mathlib

-- Definitions based on the conditions
def investment_A (x : ℝ) : ℝ := 12 * x
def investment_B (x : ℝ) : ℝ := 12 * x
def investment_C (x : ℝ) : ℝ := 12 * x
def total_investment (x : ℝ) : ℝ := investment_A x + investment_B x + investment_C x
def annual_gain : ℝ := 15000

-- Theorem based on the question and correct answer
theorem A_share_of_annual_gain (x : ℝ) : (investment_A x / total_investment x) * annual_gain = 5000 :=
by
  sorry

end A_share_of_annual_gain

namespace series_sum_equals_4290

import Mathlib

theorem series_sum_equals_4290 :
  ∑ n in Finset.range 10, n.succ * (n + 2) * (n + 3) = 4290 := 
by
  sorry

end series_sum_equals_4290

namespace point_below_line

import Mathlib

theorem point_below_line (a : ℝ) (h : 2 * a - 3 > 3) : a > 3 :=
sorry

end point_below_line

namespace find_first_5digits_of_M

import Mathlib

def last6digits (n : ℕ) : ℕ := n % 1000000

def first5digits (n : ℕ) : ℕ := n / 10

theorem find_first_5digits_of_M (M : ℕ) (h1 : last6digits M = last6digits (M^2)) (h2 : M > 999999) : first5digits M = 60937 := 
by sorry

end find_first_5digits_of_M

namespace intersection_A_B

import Mathlib

def A := { x : ℝ | x / (x - 1) ≥ 0 }
def B := { y : ℝ | ∃ x : ℝ, y = 3 * x^2 + 1 }

theorem intersection_A_B : A ∩ B = { y : ℝ | y > 1 } :=
by sorry

end intersection_A_B

namespace range_of_m

import Mathlib

theorem range_of_m (m : ℝ) :
  (∃ x : ℝ, x > 0 ∧ (2 * x + m) / (x - 2) + (x - 1) / (2 - x) = 3) ↔ (m > -7 ∧ m ≠ -3) :=
by
  sorry

end range_of_m

namespace function_value

import Mathlib

noncomputable def log_base (a x : ℝ) : ℝ := Real.log x / Real.log a

theorem function_value (a b : ℝ) (h₀ : a > 0) (h₁ : a ≠ 1) (h₂ : log_base a (2 + b) = 1) (h₃ : log_base a (8 + b) = 2) : a + b = 4 :=
by
  sorry

end function_value

namespace sum_largest_and_smallest

import Mathlib

-- Define the three-digit number properties
def hundreds_digit := 4
def tens_digit := 8
def A : ℕ := sorry  -- Placeholder for the digit A

-- Define the number based on the digits
def number (A : ℕ) : ℕ := 100 * hundreds_digit + 10 * tens_digit + A

-- Hypotheses
axiom A_range : 0 ≤ A ∧ A ≤ 9

-- Largest and smallest possible numbers
def largest_number := number 9
def smallest_number := number 0

-- Prove the sum
theorem sum_largest_and_smallest : largest_number + smallest_number = 969 :=
by
  sorry

end sum_largest_and_smallest

namespace 

import Mathlib

-- Define arithmetic progression with first term a₁ and common difference d
def arithmetic_progression (a₁ d n : ℤ) : ℤ := a₁ + (n - 1) * d

-- Define the sum of the first 7 terms of the arithmetic progression
def sum_first_7_terms (a₁ d : ℤ) : ℤ := 7 * a₁ + 21 * d

-- Define the conditions given
def condition1 (a₁ d : ℤ) : Prop := 
  (arithmetic_progression a₁ d 7) * (arithmetic_progression a₁ d 12) > (sum_first_7_terms a₁ d) + 20

def condition2 (a₁ d : ℤ) : Prop := 
  (arithmetic_progression a₁ d 9) * (arithmetic_progression a₁ d 10) < (sum_first_7_terms a₁ d) + 44

-- The main problem to prove
def problem (a₁ : ℤ) (d : ℤ) : Prop := 
  condition1 a₁ d ∧ condition2 a₁ d

-- The theorem statement to prove
theorem possible_values_of_a₁ (a₁ d : ℤ) : problem a₁ d → a₁ = -9 ∨ a₁ = -8 ∨ a₁ = -7 ∨ a₁ = -6 ∨ a₁ = -4 ∨ a₁ = -3 ∨ a₁ = -2 ∨ a₁ = -1 := 
by sorry

end 

namespace paint_house

import Mathlib

theorem paint_house (n s h : ℕ) (h_pos : 0 < h)
    (rate_eq : ∀ (x : ℕ), 0 < x → ∃ t : ℕ, x * t = n * h) :
    (n + s) * (nh / (n + s)) = n * h := 
sorry

end paint_house

namespace sphere_radius_equal

import Mathlib

theorem sphere_radius_equal (r : ℝ) 
  (hvol : (4 / 3) * Real.pi * r^3 = 4 * Real.pi * r^2) : r = 3 :=
sorry

end sphere_radius_equal

namespace fraction_to_decimal

import Mathlib

theorem fraction_to_decimal : (58 : ℚ) / 125 = 0.464 := by
  sorry

end fraction_to_decimal

namespace number_of_diagonals_octagon_heptagon_diff

import Mathlib

def number_of_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

theorem number_of_diagonals_octagon_heptagon_diff :
  let A := number_of_diagonals 8
  let B := number_of_diagonals 7
  A - B = 6 :=
by
  sorry

end number_of_diagonals_octagon_heptagon_diff

namespace neg_p

import Mathlib

noncomputable def f (a x : ℝ) : ℝ := a^x - x - a

theorem neg_p :
  ∃ (a : ℝ), a > 0 ∧ a ≠ 1 ∧ ∀ (x : ℝ), f a x ≠ 0 :=
sorry

end neg_p

namespace rational_includes_integers_and_fractions

import Mathlib

def is_integer (x : ℤ) : Prop := true
def is_fraction (x : ℚ) : Prop := true
def is_rational (x : ℚ) : Prop := true

theorem rational_includes_integers_and_fractions : 
  (∀ x : ℤ, is_integer x → is_rational (x : ℚ)) ∧ 
  (∀ x : ℚ, is_fraction x → is_rational x) :=
by {
  sorry -- Proof to be filled in
}

end rational_includes_integers_and_fractions

namespace categorize_numbers

import Mathlib

def numbers : List ℚ := [-16/10, -5/6, 89/10, -7, 1/12, 0, 25]

def is_positive (x : ℚ) : Prop := x > 0
def is_negative_fraction (x : ℚ) : Prop := x < 0 ∧ x.den ≠ 1
def is_negative_integer (x : ℚ) : Prop := x < 0 ∧ x.den = 1

theorem categorize_numbers :
  { x | x ∈ numbers ∧ is_positive x } = { 89 / 10, 1 / 12, 25 } ∧
  { x | x ∈ numbers ∧ is_negative_fraction x } = { -5 / 6 } ∧
  { x | x ∈ numbers ∧ is_negative_integer x } = { -7 } := by
  sorry

end categorize_numbers

namespace sum_leq_six_of_quadratic_roots

import Mathlib

theorem sum_leq_six_of_quadratic_roots (a b : ℤ) (h1 : a ≠ -1) (h2 : b ≠ -1) 
  (h3 : ∃ r1 r2 : ℤ, r1 ≠ r2 ∧ x^2 + ab * x + (a + b) = 0 ∧ 
         x = r1 ∧ x = r2) : a + b ≤ 6 :=
by
  sorry

end sum_leq_six_of_quadratic_roots

namespace point_translation

import Mathlib

theorem point_translation :
  ∃ (x_old y_old x_new y_new : ℤ),
  (x_old = 1 ∧ y_old = -2) ∧
  (x_new = x_old + 2) ∧
  (y_new = y_old + 3) ∧
  (x_new = 3) ∧
  (y_new = 1) :=
sorry

end point_translation

namespace larry_gave_52_apples

import Mathlib

-- Define the initial and final count of Joyce's apples
def initial_apples : ℝ := 75.0
def final_apples : ℝ := 127.0

-- Define the number of apples Larry gave Joyce
def apples_given : ℝ := final_apples - initial_apples

-- The theorem stating that Larry gave Joyce 52 apples
theorem larry_gave_52_apples : apples_given = 52 := by
  sorry

end larry_gave_52_apples

namespace lizzy_final_amount

import Mathlib

-- Define constants
def m : ℕ := 80   -- cents from mother
def f : ℕ := 40   -- cents from father
def s : ℕ := 50   -- cents spent on candy
def u : ℕ := 70   -- cents from uncle
def t : ℕ := 90   -- cents for the toy
def c : ℕ := 110  -- cents change she received

-- Define the final amount calculation
def final_amount : ℕ := m + f - s + u - t + c

-- Prove the final amount is 160
theorem lizzy_final_amount : final_amount = 160 := by
  sorry

end lizzy_final_amount

namespace isosceles_triangle_sides

import Mathlib

theorem isosceles_triangle_sides (a b : ℝ) (h1 : 2 * a + a = 14 ∨ 2 * a + a = 18)
  (h2 : a + b = 18 ∨ a + b = 14) : 
  (a = 14/3 ∧ b = 40/3 ∨ a = 6 ∧ b = 8) :=
by
  sorry

end isosceles_triangle_sides

namespace area_of_sector

import Mathlib

theorem area_of_sector (s θ : ℝ) (r : ℝ) (h_s : s = 4) (h_θ : θ = 2) (h_r : r = s / θ) :
  (1 / 2) * r^2 * θ = 4 :=
by
  sorry

end area_of_sector

namespace find_ABC_base10

import Mathlib

theorem find_ABC_base10
  (A B C : ℕ)
  (h1 : 0 < A ∧ A < 6)
  (h2 : 0 < B ∧ B < 6)
  (h3 : 0 < C ∧ C < 6)
  (h4 : A ≠ B ∧ B ≠ C ∧ A ≠ C)
  (h5 : B + C = 6)
  (h6 : A + 1 = C)
  (h7 : A + B = C) :
  100 * A + 10 * B + C = 415 :=
by
  sorry

end find_ABC_base10

namespace negation_of_proposition

import Mathlib

-- Given condition
def original_statement (a : ℝ) : Prop :=
  ∃ x : ℝ, a*x^2 - 2*a*x + 1 ≤ 0

-- Correct answer (negation statement)
def negated_statement (a : ℝ) : Prop :=
  ∀ x : ℝ, a*x^2 - 2*a*x + 1 > 0

-- Statement to prove
theorem negation_of_proposition (a : ℝ) :
  ¬ (original_statement a) ↔ (negated_statement a) :=
by 
  sorry

end negation_of_proposition

namespace cost_of_green_pill

import Mathlib

-- Let the cost of a green pill be g and the cost of a pink pill be p
variables (g p : ℕ)
-- Beth takes two green pills and one pink pill each day
-- A green pill costs twice as much as a pink pill
-- The total cost for the pills over three weeks (21 days) is $945

theorem cost_of_green_pill : 
  (2 * g + p) * 21 = 945 ∧ g = 2 * p → g = 18 :=
by
  sorry

end cost_of_green_pill

namespace rectangle_width_decrease

import Mathlib

theorem rectangle_width_decrease (a b : ℝ) (p x : ℝ) 
  (hp : p ≥ 0) (hx : x ≥ 0)
  (area_eq : a * b = (a * (1 + p / 100)) * (b * (1 - x / 100))) :
  x = (100 * p) / (100 + p) := 
by
  sorry

end rectangle_width_decrease

namespace algebra_inequality

import Mathlib

theorem algebra_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h_sum : a^3 + b^3 + c^3 = 3) : 
  1 / (a^2 + a + 1) + 1 / (b^2 + b + 1) + 1 / (c^2 + c + 1) ≥ 1 := 
by 
  sorry

end algebra_inequality

namespace farmer_plants_rows_per_bed

import Mathlib

theorem farmer_plants_rows_per_bed 
    (bean_seedlings : ℕ) (beans_per_row : ℕ)
    (pumpkin_seeds : ℕ) (pumpkins_per_row : ℕ)
    (radishes : ℕ) (radishes_per_row : ℕ)
    (plant_beds : ℕ)
    (h1 : bean_seedlings = 64)
    (h2 : beans_per_row = 8)
    (h3 : pumpkin_seeds = 84)
    (h4 : pumpkins_per_row = 7)
    (h5 : radishes = 48)
    (h6 : radishes_per_row = 6)
    (h7 : plant_beds = 14) : 
    (bean_seedlings / beans_per_row + pumpkin_seeds / pumpkins_per_row + radishes / radishes_per_row) / plant_beds = 2 :=
by
  sorry

end farmer_plants_rows_per_bed

namespace pet_shop_ways

import Mathlib

theorem pet_shop_ways (puppies : ℕ) (kittens : ℕ) (turtles : ℕ)
  (h_puppies : puppies = 10) (h_kittens : kittens = 8) (h_turtles : turtles = 5) : 
  (puppies * kittens * turtles = 400) :=
by
  sorry

end pet_shop_ways

namespace coeff_z_in_third_eq

import Mathlib

-- Definitions for the conditions
def eq1 (x y z : ℝ) : Prop := 6 * x - 5 * y + 3 * z = 22
def eq2 (x y z : ℝ) : Prop := 4 * x + 8 * y - 11 * z = 7
def eq3 (x y z : ℝ) : Prop := 5 * x - 6 * y + z = 6
def sum_condition (x y z : ℝ) : Prop := x + y + z = 10

-- Theorem statement
theorem coeff_z_in_third_eq : ∀ (x y z : ℝ), eq1 x y z → eq2 x y z → eq3 x y z → sum_condition x y z → (1 = 1) :=
by
  intros
  sorry

end coeff_z_in_third_eq

namespace isosceles_triangle_perimeter

import Mathlib

theorem isosceles_triangle_perimeter (a b : ℕ) (h1 : a = 9) (h2 : b = 4) (h3 : b < a + a) : a + a + b = 22 := by
  sorry

end isosceles_triangle_perimeter

namespace equivalent_expression

import Mathlib

theorem equivalent_expression :
  (5+3) * (5^2 + 3^2) * (5^4 + 3^4) * (5^8 + 3^8) * (5^16 + 3^16) * 
  (5^32 + 3^32) * (5^64 + 3^64) = 5^128 - 3^128 := 
  sorry

end equivalent_expression

namespace bridge_construction

import Mathlib

-- Definitions used in the Lean statement based on conditions.
def rate (workers : ℕ) (days : ℕ) : ℚ := 1 / (workers * days)

-- The problem statement: prove that if 60 workers working together can build the bridge in 3 days, 
-- then 120 workers will take 1.5 days to build the bridge.
theorem bridge_construction (t : ℚ) : 
  (rate 60 3) * 120 * t = 1 → t = 1.5 := by
  sorry

end bridge_construction

namespace blonde_hair_count

import Mathlib

theorem blonde_hair_count (total_people : ℕ) (percentage_blonde : ℕ) (h_total : total_people = 600) (h_percentage : percentage_blonde = 30) : 
  (percentage_blonde * total_people / 100) = 180 :=
by
  -- Conditions from the problem
  have h1 : total_people = 600 := h_total
  have h2 : percentage_blonde = 30 := h_percentage
  -- Start the proof
  sorry

end blonde_hair_count

namespace moles_H2O_formed

import Mathlib

-- Define the conditions
def moles_HCl : ℕ := 6
def moles_CaCO3 : ℕ := 3
def moles_CaCl2 : ℕ := 3
def moles_CO2 : ℕ := 3

-- Proposition that we need to prove
theorem moles_H2O_formed : moles_CaCl2 = 3 ∧ moles_CO2 = 3 ∧ moles_CaCO3 = 3 ∧ moles_HCl = 6 → moles_CaCO3 = 3 := by
  sorry

end moles_H2O_formed

namespace sum_of_consecutive_odds_eq_169

import Mathlib

theorem sum_of_consecutive_odds_eq_169 : 
  (∃ n : ℕ, (∑ i in Finset.range (n+1), if i % 2 = 1 then i else 0) = 169) ↔ n = 13 :=
by
  sorry

end sum_of_consecutive_odds_eq_169

namespace field_area

import Mathlib

def length : ℝ := 80 -- Length of the uncovered side
def total_fencing : ℝ := 97 -- Total fencing required

theorem field_area : ∃ (W L : ℝ), L = length ∧ 2 * W + L = total_fencing ∧ L * W = 680 := by
  sorry

end field_area

namespace edward_mowed_lawns

import Mathlib

theorem edward_mowed_lawns (L : ℕ) (h1 : 8 * L + 7 = 47) : L = 5 :=
by
  sorry

end edward_mowed_lawns

namespace fred_red_marbles

import Mathlib

theorem fred_red_marbles (total_marbles : ℕ) (dark_blue_marbles : ℕ) (green_marbles : ℕ) (red_marbles : ℕ) 
  (h1 : total_marbles = 63) 
  (h2 : dark_blue_marbles ≥ total_marbles / 3)
  (h3 : green_marbles = 4)
  (h4 : red_marbles = total_marbles - dark_blue_marbles - green_marbles) : 
  red_marbles = 38 := 
sorry

end fred_red_marbles

namespace f_of_x

import Mathlib

variable (f : ℝ → ℝ)

theorem f_of_x (x : ℝ) (h : f (x - 1 / x) = x^2 + 1 / x^2) : f x = x^2 + 2 :=
sorry

end f_of_x

namespace average_marks_all_students_proof

import Mathlib

-- Definitions based on the given conditions
def class1_student_count : ℕ := 35
def class2_student_count : ℕ := 45
def class1_average_marks : ℕ := 40
def class2_average_marks : ℕ := 60

-- Total marks calculations
def class1_total_marks : ℕ := class1_student_count * class1_average_marks
def class2_total_marks : ℕ := class2_student_count * class2_average_marks
def total_marks : ℕ := class1_total_marks + class2_total_marks

-- Total student count
def total_student_count : ℕ := class1_student_count + class2_student_count

-- Average marks of all students
noncomputable def average_marks_all_students : ℚ := total_marks / total_student_count

-- Lean statement to prove
theorem average_marks_all_students_proof
  (h1 : class1_student_count = 35)
  (h2 : class2_student_count = 45)
  (h3 : class1_average_marks = 40)
  (h4 : class2_average_marks = 60) :
  average_marks_all_students = 51.25 := by
  sorry

end average_marks_all_students_proof

namespace solve_for_k

import Mathlib

def f (n : ℤ) : ℤ :=
  if n % 2 = 1 then n + 5 else n / 2

theorem solve_for_k (k : ℤ) (h1 : k % 2 = 1) (h2 : f (f (f k)) = 57) : k = 223 :=
by
  -- Proof will be provided here
  sorry

end solve_for_k

namespace transformation_composition

import Mathlib

-- Define the transformations f and g
def f (m n : ℝ) : ℝ × ℝ := (m, -n)
def g (m n : ℝ) : ℝ × ℝ := (-m, -n)

-- The proof statement that we need to prove
theorem transformation_composition : g (f (-3) 2).1 (f (-3) 2).2 = (3, 2) :=
by sorry

end transformation_composition

namespace number_of_students_playing_soccer

import Mathlib

variables (T B girls_total soccer_total G no_girls_soccer perc_boys_soccer : ℕ)

-- Conditions:
def total_students := T = 420
def boys_students := B = 312
def girls_students := G = 420 - 312
def girls_not_playing_soccer := no_girls_soccer = 63
def perc_boys_play_soccer := perc_boys_soccer = 82
def girls_playing_soccer := G - no_girls_soccer = 45

-- Proof Problem:
theorem number_of_students_playing_soccer (h1 : total_students T) (h2 : boys_students B) (h3 : girls_students G) (h4 : girls_not_playing_soccer no_girls_soccer) (h5 : girls_playing_soccer G no_girls_soccer) (h6 : perc_boys_play_soccer perc_boys_soccer) : soccer_total = 250 :=
by {
  -- The proof would be inserted here.
  sorry
}

end number_of_students_playing_soccer

namespace fixed_point_always_on_line

import Mathlib

theorem fixed_point_always_on_line (a : ℝ) (h : a ≠ 0) :
  (a + 2) * 1 + (1 - a) * 1 - 3 = 0 :=
by
  sorry

end fixed_point_always_on_line

namespace dividend_is_correct

import Mathlib

def divisor : ℕ := 17
def quotient : ℕ := 9
def remainder : ℕ := 6

def calculate_dividend (divisor : ℕ) (quotient : ℕ) (remainder : ℕ) : ℕ :=
  (divisor * quotient) + remainder

theorem dividend_is_correct : calculate_dividend divisor quotient remainder = 159 :=
  by sorry

end dividend_is_correct

namespace right_triangle_condition

import Mathlib

theorem right_triangle_condition (a b c : ℝ) (h : c^2 - a^2 = b^2) : 
  ∃ (A B C : ℝ), A + B + C = 180 ∧ A = 90 ∧ B + C = 90 :=
by sorry

end right_triangle_condition

namespace sum_interior_ninth_row

import Mathlib

-- Define Pascal's Triangle and the specific conditions
def pascal_sum (n : ℕ) : ℕ := 2^(n - 1)

def sum_interior_numbers (n : ℕ) : ℕ := pascal_sum n - 2

theorem sum_interior_ninth_row :
  sum_interior_numbers 9 = 254 := 
by {
  sorry
}

end sum_interior_ninth_row

namespace units_digit_odd_product

import Mathlib

theorem units_digit_odd_product (l : List ℕ) (h_odds : ∀ n ∈ l, n % 2 = 1) :
  (∀ x ∈ l, x % 10 = 5) ↔ (5 ∈ l) := by
  sorry

end units_digit_odd_product

namespace water_tank_capacity

import Mathlib

theorem water_tank_capacity (C : ℝ) (h : 0.70 * C - 0.40 * C = 36) : C = 120 :=
sorry

end water_tank_capacity

namespace range_of_m

import Mathlib

variables (m : ℝ)

def p : Prop := ∀ x : ℝ, 0 < x → (1/2 : ℝ)^x + m - 1 < 0
def q : Prop := ∃ x : ℝ, 0 < x ∧ m * x^2 + 4 * x - 1 = 0

theorem range_of_m (h : p m ∧ q m) : -4 ≤ m ∧ m ≤ 0 :=
sorry

end range_of_m

namespace manufacturing_section_degrees

import Mathlib

variable (percentage_manufacturing : ℝ) (total_degrees : ℝ)

theorem manufacturing_section_degrees
  (h1 : percentage_manufacturing = 0.40)
  (h2 : total_degrees = 360) :
  percentage_manufacturing * total_degrees = 144 := 
by 
  sorry

end manufacturing_section_degrees

namespace color_opposite_orange_is_indigo

import Mathlib

-- Define the colors
inductive Color
| O | B | Y | S | V | I

-- Define a structure representing a view of the cube
structure CubeView where
  top : Color
  front : Color
  right : Color

-- Given views
def view1 := CubeView.mk Color.B Color.Y Color.S
def view2 := CubeView.mk Color.B Color.V Color.S
def view3 := CubeView.mk Color.B Color.I Color.Y

-- The statement to be proved: the color opposite to orange (O) is indigo (I), given the views
theorem color_opposite_orange_is_indigo (v1 v2 v3 : CubeView) :
  v1 = view1 →
  v2 = view2 →
  v3 = view3 →
  ∃ opposite_color : Color, opposite_color = Color.I :=
  by
    sorry

end color_opposite_orange_is_indigo

namespace square_units_digit

import Mathlib

theorem square_units_digit (n : ℕ) (h : (n^2 / 10) % 10 = 7) : n^2 % 10 = 6 := 
sorry

end square_units_digit

namespace complement_of_M_in_U

import Mathlib

def U := Set.univ (α := ℝ)
def M := {x : ℝ | x < -2 ∨ x > 8}
def compl_M := {x : ℝ | -2 ≤ x ∧ x ≤ 8}

theorem complement_of_M_in_U : compl_M = U \ M :=
by
  sorry

end complement_of_M_in_U

namespace find_number

import Mathlib

-- Define the condition that one-third of a certain number is 300% of 134
def one_third_eq_300percent_number (n : ℕ) : Prop :=
  n / 3 = 3 * 134

-- State the theorem that the number is 1206 given the above condition
theorem find_number (n : ℕ) (h : one_third_eq_300percent_number n) : n = 1206 :=
  by sorry

end find_number

namespace total_height_of_buildings

import Mathlib

-- Definitions based on the conditions
def tallest_building : ℤ := 100
def second_tallest_building : ℤ := tallest_building / 2
def third_tallest_building : ℤ := second_tallest_building / 2
def fourth_tallest_building : ℤ := third_tallest_building / 5

-- Use the definitions to state the theorem
theorem total_height_of_buildings : 
  tallest_building + second_tallest_building + third_tallest_building + fourth_tallest_building = 180 := by
  sorry

end total_height_of_buildings

namespace rectangular_solid_sum_of_edges

import Mathlib

noncomputable def sum_of_edges (x y z : ℝ) := 4 * (x + y + z)

theorem rectangular_solid_sum_of_edges :
  ∃ (x y z : ℝ), (x * y * z = 512) ∧ (2 * (x * y + y * z + z * x) = 384) ∧
  (∃ (r a : ℝ), x = a / r ∧ y = a ∧ z = a * r) ∧ sum_of_edges x y z = 96 :=
by
  sorry

end rectangular_solid_sum_of_edges

namespace farmer_trees

import Mathlib

theorem farmer_trees (x n m : ℕ) 
  (h1 : x + 20 = n^2) 
  (h2 : x - 39 = m^2) : 
  x = 880 := 
by sorry

end farmer_trees

namespace units_digit_17_times_29

import Mathlib

theorem units_digit_17_times_29 :
  (17 * 29) % 10 = 3 :=
by
  sorry

end units_digit_17_times_29

namespace minimum_value_f

import Mathlib

noncomputable def f (a b c : ℝ) : ℝ :=
  a / (Real.sqrt (a^2 + 8*b*c)) + b / (Real.sqrt (b^2 + 8*a*c)) + c / (Real.sqrt (c^2 + 8*a*b))

theorem minimum_value_f (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  1 ≤ f a b c := by
  sorry

end minimum_value_f

namespace find_tan_of_cos_in_4th_quadrant

import Mathlib

-- Given conditions
variable (α : ℝ) (h1 : Real.cos α = 3/5) (h2 : α > 3*Real.pi/2 ∧ α < 2*Real.pi)

-- Lean statement to prove the question
theorem find_tan_of_cos_in_4th_quadrant : Real.tan α = - (4 / 3) := 
by
  sorry

end find_tan_of_cos_in_4th_quadrant

namespace rank_siblings

import Mathlib

variable (Person : Type) (Dan Elena Finn : Person)

variable (height : Person → ℝ)

-- Conditions
axiom different_heights : height Dan ≠ height Elena ∧ height Elena ≠ height Finn ∧ height Finn ≠ height Dan
axiom one_true_statement : (¬ (height Elena = max (max (height Dan) (height Elena)) (height Finn)) ∨ height Finn = max (max (height Dan) (height Elena)) (height Finn) ∨ height Dan ≠ min (min (height Dan) (height Elena)) (height Finn)) 
  ∧ (¬ (height Elena = max (max (height Dan) (height Elena)) (height Finn)) ∨ height Finn = max (max (height Dan) (height Elena)) (height Finn) ∨ height Dan ≠ min (min (height Dan) (height Elena)) (height Finn))
  ∧ (¬ (height Elena = max (max (height Dan) (height Elena)) (height Finn)) ∨ height Finn = max (max (height Dan) (height Elena)) (height Finn) ∨ height Dan ≠ min (min (height Dan) (height Elena)) (height Finn))

theorem rank_siblings : height Finn > height Elena ∧ height Elena > height Dan := by
  sorry

end rank_siblings

namespace domain_correct

import Mathlib

def domain_of_function (x : ℝ) : Prop :=
  (∃ y : ℝ, y = 2 / Real.sqrt (x + 1)) ∧ Real.sqrt (x + 1) ≠ 0

theorem domain_correct (x : ℝ) : domain_of_function x ↔ (x > -1) := by
  sorry

end domain_correct

namespace joels_age_when_dad_twice

import Mathlib

theorem joels_age_when_dad_twice
  (joel_age_now : ℕ)
  (dad_age_now : ℕ)
  (years : ℕ)
  (H1 : joel_age_now = 5)
  (H2 : dad_age_now = 32)
  (H3 : years = 22)
  (H4 : dad_age_now + years = 2 * (joel_age_now + years))
  : joel_age_now + years = 27 := 
by sorry

end joels_age_when_dad_twice

namespace ratio_ashley_mary

import Mathlib

-- Definitions based on conditions
def sum_ages (A M : ℕ) := A + M = 22
def ashley_age (A : ℕ) := A = 8

-- Theorem stating the ratio of Ashley's age to Mary's age
theorem ratio_ashley_mary (A M : ℕ) 
  (h1 : sum_ages A M)
  (h2 : ashley_age A) : 
  (A : ℚ) / (M : ℚ) = 4 / 7 :=
by
  -- Skipping the proof as specified
  sorry

end ratio_ashley_mary

namespace ratio_second_part

import Mathlib

theorem ratio_second_part (first_part second_part total : ℕ) 
  (h_ratio_percent : 50 = 100 * first_part / total) 
  (h_first_part : first_part = 10) : 
  second_part = 10 := by
  have h_total : total = 2 * first_part := by sorry
  sorry

end ratio_second_part

namespace max_vertex_value_in_cube

import Mathlib

def transform_black (v : ℕ) (e1 e2 e3 : ℕ) : ℕ :=
  e1 + e2 + e3

def transform_white (v : ℕ) (d1 d2 d3 : ℕ) : ℕ :=
  d1 + d2 + d3

def max_value_after_transformation (initial_values : Fin 8 → ℕ) : ℕ :=
  -- Combination of transformations and iterations are derived here
  42648

theorem max_vertex_value_in_cube :
  ∀ (initial_values : Fin 8 → ℕ),
  (∀ i, 1 ≤ initial_values i ∧ initial_values i ≤ 8) →
  (∃ (final_value : ℕ), final_value = max_value_after_transformation initial_values) → final_value = 42648 :=
by {
  sorry
}

end max_vertex_value_in_cube

namespace red_blue_card_sum

import Mathlib

theorem red_blue_card_sum (N : ℕ) (r b : ℕ → ℕ) (h_r : ∀ i, 1 ≤ r i ∧ r i ≤ N) (h_b : ∀ i, 1 ≤ b i ∧ b i ≤ N):
  ∃ (A B : Finset ℕ), A ≠ ∅ ∧ B ≠ ∅ ∧ (∑ i in A, r i) = ∑ j in B, b j :=
by
  sorry

end red_blue_card_sum

namespace number_of_sequences

import Mathlib

theorem number_of_sequences (n k : ℕ) (h₁ : 1 ≤ k) (h₂ : k ≤ n) :
  ∃ C : ℕ, C = Nat.choose (Nat.floor ((n + 2 - k) / 2) + k - 1) k :=
sorry

end number_of_sequences

namespace part_1_part_2_part_3

import Mathlib

variable {f : ℝ → ℝ}

axiom C1 : ∀ x y : ℝ, f (x + y) = f x + f y
axiom C2 : ∀ x : ℝ, x > 0 → f x < 0
axiom C3 : f 3 = -4

theorem part_1 : f 0 = 0 :=
by
  sorry

theorem part_2 : ∀ x : ℝ, f (-x) = -f x :=
by
  sorry

theorem part_3 : ∀ x : ℝ, -9 ≤ x ∧ x ≤ 9 → f x ≤ 12 ∧ f x ≥ -12 :=
by
  sorry

end part_1_part_2_part_3

namespace coordinates_with_respect_to_origin

import Mathlib

theorem coordinates_with_respect_to_origin :
  ∀ (point : ℝ × ℝ), point = (3, -2) → point = (3, -2) := by
  intro point h
  exact h

end coordinates_with_respect_to_origin

namespace MrsHiltReadTotalChapters

import Mathlib

-- Define the number of books and chapters per book
def numberOfBooks : ℕ := 4
def chaptersPerBook : ℕ := 17

-- Define the total number of chapters Mrs. Hilt read
def totalChapters (books : ℕ) (chapters : ℕ) : ℕ := books * chapters

-- The main statement to be proved
theorem MrsHiltReadTotalChapters : totalChapters numberOfBooks chaptersPerBook = 68 := by
  sorry

end MrsHiltReadTotalChapters

namespace solve_system_eqs

import Mathlib

theorem solve_system_eqs (x y : ℝ) (h1 : (1 + x) * (1 + x^2) * (1 + x^4) = 1 + y^7)
                            (h2 : (1 + y) * (1 + y^2) * (1 + y^4) = 1 + x^7) :
  (x = 0 ∧ y = 0) ∨ (x = -1 ∧ y = -1) :=
sorry

end solve_system_eqs

namespace hayley_friends

import Mathlib

theorem hayley_friends (total_stickers : ℕ) (stickers_per_friend : ℕ) (h1 : total_stickers = 72) (h2 : stickers_per_friend = 8) : (total_stickers / stickers_per_friend) = 9 :=
by
  sorry

end hayley_friends

namespace abc_le_one_eighth

import Mathlib

theorem abc_le_one_eighth (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h : a / (1 + a) + b / (1 + b) + c / (1 + c) = 1) : a * b * c ≤ 1 / 8 :=
by
  sorry

end abc_le_one_eighth

namespace sum_f_div_2009_equals_1005

import Mathlib

def f (x : ℚ) : ℚ := x^5 / (5*x^4 - 10*x^3 + 10*x^2 - 5*x + 1)

theorem sum_f_div_2009_equals_1005 :
  (∑ i in Finset.range (2009+1).succ, f (i / 2009)) = 1005 :=
sorry

end sum_f_div_2009_equals_1005

namespace tank_full_capacity

import Mathlib

theorem tank_full_capacity (w c : ℕ) (h1 : w = c / 6) (h2 : w + 4 = c / 3) : c = 12 :=
sorry

end tank_full_capacity

namespace calculate_expression

import Mathlib

theorem calculate_expression : 
  (2^10 + (3^6 / 3^2)) = 1105 := 
by 
  -- Steps involve intermediate calculations
  -- for producing (2^10 = 1024), (3^6 = 729), (3^2 = 9)
  -- and then finding (729 / 9 = 81), (1024 + 81 = 1105)
  sorry

end calculate_expression

namespace condition_an_necessary_but_not_sufficient

import Mathlib

-- Definitions for the sequence and properties
def is_geometric_sequence (a : ℕ → ℝ) (r : ℝ) : Prop :=
  ∀ n, n ≥ 1 → a (n + 1) = r * (a n)

def condition_an (a : ℕ → ℝ) : Prop :=
  ∀ n, n ≥ 2 → a n = 2 * a (n - 1)

-- The theorem statement
theorem condition_an_necessary_but_not_sufficient (a : ℕ → ℝ) :
  (∀ n, n ≥ 1 → a (n + 1) = 2 * (a n)) → (condition_an a) ∧ ¬(is_geometric_sequence a 2) :=
by
  sorry

end condition_an_necessary_but_not_sufficient

namespace Petya_wins_optimally

import Mathlib

-- Defining the game state and rules
inductive GameState
| PetyaWin
| VasyaWin

-- Rules of the game
def game_rule (n : ℕ) : Prop :=
  n > 0 ∧ (n % 3 = 0 ∨ n % 3 = 1 ∨ n % 3 = 2)

-- Determine the winner given the initial number of minuses
def determine_winner (n : ℕ) : GameState :=
  if n % 3 = 0 then GameState.PetyaWin else GameState.VasyaWin

-- Theorem: Petya will win the game if both play optimally
theorem Petya_wins_optimally (n : ℕ) (h1 : n = 2021) (h2 : game_rule n) : determine_winner n = GameState.PetyaWin :=
by {
  sorry
}

end Petya_wins_optimally

namespace cucumbers_after_purchase

import Mathlib

theorem cucumbers_after_purchase (C U : ℕ) (h1 : C + U = 10) (h2 : C = 4) : U + 2 = 8 := by
  sorry

end cucumbers_after_purchase

namespace baker_remaining_cakes

import Mathlib

def initial_cakes : ℝ := 167.3
def sold_cakes : ℝ := 108.2
def remaining_cakes : ℝ := initial_cakes - sold_cakes

theorem baker_remaining_cakes : remaining_cakes = 59.1 := by
  sorry

end baker_remaining_cakes

namespace positive_iff_sum_and_product_positive

import Mathlib

theorem positive_iff_sum_and_product_positive (a b : ℝ) :
  (a > 0 ∧ b > 0) ↔ (a + b > 0 ∧ a * b > 0) :=
by
  sorry

end positive_iff_sum_and_product_positive

namespace travel_days

import Mathlib

variable (a b d : ℕ)

theorem travel_days (h1 : a + d = 11) (h2 : b + d = 21) (h3 : a + b = 12) : a + b + d = 22 :=
by sorry

end travel_days

namespace arc_length

import Mathlib

theorem arc_length (C : ℝ) (theta : ℝ) (hC : C = 100) (htheta : theta = 30) :
  (theta / 360) * C = 25 / 3 :=
by sorry

end arc_length

namespace sqrt_144_times_3_squared

import Mathlib

theorem sqrt_144_times_3_squared :
  ( (Real.sqrt 144) * 3 ) ^ 2 = 1296 := by
  sorry

end sqrt_144_times_3_squared

namespace temperature_difference_correct

import Mathlib

def avg_high : ℝ := 9
def avg_low : ℝ := -5
def temp_difference : ℝ := avg_high - avg_low

theorem temperature_difference_correct : temp_difference = 14 := by
  sorry

end temperature_difference_correct

namespace range_of_x

import Mathlib

noncomputable def f (x : ℝ) : ℝ := Real.exp x - Real.exp (-x) + 1

theorem range_of_x (x : ℝ) (h : f (2 * x - 1) + f (4 - x^2) > 2) : x ∈ Set.Ioo (-1 : ℝ) 3 :=
by
  sorry

end range_of_x

namespace trader_profit

import Mathlib

theorem trader_profit (P : ℝ) (hP : 0 < P) : 
  let purchase_price := 0.80 * P
  let selling_price := 1.36 * P
  let profit := selling_price - P
  (profit / P) * 100 = 36 :=
by
  -- The proof will go here
  sorry

end trader_profit

namespace input_statement_is_INPUT

import Mathlib

namespace ProgrammingStatements

-- Definitions of each type of statement
def PRINT_is_output : Prop := True
def INPUT_is_input : Prop := True
def THEN_is_conditional : Prop := True
def END_is_termination : Prop := True

-- The proof problem
theorem input_statement_is_INPUT :
  INPUT_is_input := by
  sorry

end ProgrammingStatements

end input_statement_is_INPUT

namespace problem_I_problem_II

import Mathlib

noncomputable def f (x m : ℝ) : ℝ := |x + m^2| + |x - 2*m - 3|

theorem problem_I (x m : ℝ) : f x m ≥ 2 :=
by 
  sorry

theorem problem_II (m : ℝ) : f 2 m ≤ 16 ↔ -3 ≤ m ∧ m ≤ Real.sqrt 14 - 1 :=
by 
  sorry

end problem_I_problem_II

namespace find_sum_invested

import Mathlib

theorem find_sum_invested (P : ℝ)
  (h1 : P * 18 / 100 * 2 - P * 12 / 100 * 2 = 504) :
  P = 4200 := 
sorry

end find_sum_invested

namespace find_pos_int

import Mathlib

theorem find_pos_int (n p : ℕ) (h_prime : Nat.Prime p) (h_pos_n : 0 < n) (h_pos_p : 0 < p) : 
  n^8 - p^5 = n^2 + p^2 → (n = 2 ∧ p = 3) :=
by
  sorry

end find_pos_int

namespace set_intersection_eq

import Mathlib.Data.Real.Basic

def A : Set ℝ := {x | |x - 1| ≤ 2}
def B : Set ℝ := {x | x^2 - 4 * x > 0}

theorem set_intersection_eq :
  A ∩ (Set.univ \ B) = {x | 0 ≤ x ∧ x ≤ 3} := by
  sorry

end set_intersection_eq

namespace solve_for_x

import Mathlib

theorem solve_for_x (x : ℝ) (h : 144 / 0.144 = 14.4 / x) : x = 0.0144 := 
by
  sorry

end solve_for_x

namespace monotone_intervals_range_of_t_for_three_roots

import Mathlib

def f (t x : ℝ) : ℝ := x^3 - 2 * x^2 + x + t

def f_prime (x : ℝ) : ℝ := 3 * x^2 - 4 * x + 1

-- 1. Monotonic intervals
theorem monotone_intervals (t : ℝ) :
  (∀ x, f_prime x > 0 → x < 1/3 ∨ x > 1) ∧
  (∀ x, f_prime x < 0 → 1/3 < x ∧ x < 1) :=
sorry

-- 2. Range of t for three real roots
theorem range_of_t_for_three_roots (t : ℝ) :
  (∃ a b : ℝ, f t a = 0 ∧ f t b = 0 ∧ a ≠ b ∧
   a = 1/3 ∧ b = 1 ∧
   -4/27 + t > 0 ∧ t < 0) :=
sorry

end monotone_intervals_range_of_t_for_three_roots

namespace yulia_max_candies

import Mathlib

def maxCandies (totalCandies : ℕ) (horizontalCandies : ℕ) (verticalCandies : ℕ) (diagonalCandies : ℕ) : ℕ :=
  totalCandies - min (2 * horizontalCandies + 3 * diagonalCandies) (3 * diagonalCandies + 2 * verticalCandies)

-- Constants
def totalCandies : ℕ := 30
def horizontalMoveCandies : ℕ := 2
def verticalMoveCandies : ℕ := 2
def diagonalMoveCandies : ℕ := 3
def path1_horizontalMoves : ℕ := 5
def path1_diagonalMoves : ℕ := 2
def path2_verticalMoves : ℕ := 1
def path2_diagonalMoves : ℕ := 5

theorem yulia_max_candies :
  maxCandies totalCandies (path1_horizontalMoves + path2_verticalMoves) 0 (path1_diagonalMoves + path2_diagonalMoves) = 14 :=
by
  sorry

end yulia_max_candies

namespace g_at_100

import Mathlib

-- Defining that g is a function from positive real numbers to real numbers
def g : ℝ → ℝ := sorry

-- The given conditions
axiom functional_equation (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : 
  x * g y - y * g x = g (x / y)

axiom g_one : g 1 = 1

-- The theorem to prove
theorem g_at_100 : g 100 = 50 :=
by
  sorry

end g_at_100

namespace total_population_milburg

import Mathlib

def num_children : ℕ := 2987
def num_adults : ℕ := 2269

theorem total_population_milburg : num_children + num_adults = 5256 := by
  sorry

end total_population_milburg

namespace angelina_speed_from_grocery_to_gym

import Mathlib

theorem angelina_speed_from_grocery_to_gym
    (v : ℝ)
    (hv : v > 0)
    (home_to_grocery_distance : ℝ := 150)
    (grocery_to_gym_distance : ℝ := 200)
    (time_difference : ℝ := 10)
    (time_home_to_grocery : ℝ := home_to_grocery_distance / v)
    (time_grocery_to_gym : ℝ := grocery_to_gym_distance / (2 * v))
    (h_time_diff : time_home_to_grocery - time_grocery_to_gym = time_difference) :
    2 * v = 10 := by
  sorry

end angelina_speed_from_grocery_to_gym

namespace probability_of_green

import Mathlib

theorem probability_of_green : 
  ∀ (P_red P_orange P_yellow P_green : ℝ), 
    P_red = 0.25 → P_orange = 0.35 → P_yellow = 0.1 → 
    P_red + P_orange + P_yellow + P_green = 1 →
    P_green = 0.3 :=
by
  intros P_red P_orange P_yellow P_green h_red h_orange h_yellow h_total
  sorry

end probability_of_green

namespace 

import Mathlib.Data.Real.Basic

noncomputable def max_expression (a b : ℝ) : ℝ :=
(a + b) ^ 2 / (a ^ 2 + 2 * a * b + b ^ 2)

theorem max_value_is_one {a b : ℝ} (ha : 0 < a) (hb : 0 < b) :
  max_expression a b ≤ 1 :=
sorry

end 

namespace polynomial_divisibility

import Mathlib

theorem polynomial_divisibility (n : ℕ) : 120 ∣ (n^5 - 5*n^3 + 4*n) :=
sorry

end polynomial_divisibility

namespace sine_shift

import Mathlib

variable (m : ℝ)

theorem sine_shift (h : Real.sin 5.1 = m) : Real.sin 365.1 = m :=
by
  sorry

end sine_shift

namespace value_of_g_at_neg3

import Mathlib

def g (x : ℚ) : ℚ := (6 * x + 2) / (x - 2)

theorem value_of_g_at_neg3 : g (-3) = 16 / 5 := by
  sorry

end value_of_g_at_neg3

namespace circle_division_parts

import Mathlib

-- Define the number of parts a circle is divided into by the chords.
noncomputable def numberOfParts (n : ℕ) : ℚ :=
  (n^4 - 6*n^3 + 23*n^2 - 18*n + 24) / 24

-- Prove that the number of parts is given by the defined function.
theorem circle_division_parts (n : ℕ) : numberOfParts n = (n^4 - 6*n^3 + 23*n^2 - 18*n + 24) / 24 := by
  sorry

end circle_division_parts

namespace rate_of_grapes_calculation

import Mathlib

theorem rate_of_grapes_calculation (total_cost cost_mangoes cost_grapes : ℕ) (rate_grapes : ℕ):
  total_cost = 1125 →
  cost_mangoes = 9 * 55 →
  cost_grapes = 9 * rate_grapes →
  total_cost = cost_grapes + cost_mangoes →
  rate_grapes = 70 :=
by
  intros h1 h2 h3 h4
  sorry

end rate_of_grapes_calculation

namespace ratio_of_juice_to_bread

import Mathlib

variable (total_money : ℕ) (money_left : ℕ) (cost_bread : ℕ) (cost_butter : ℕ) (cost_juice : ℕ)

def compute_ratio (total_money money_left cost_bread cost_butter cost_juice : ℕ) : ℕ :=
  cost_juice / cost_bread

theorem ratio_of_juice_to_bread :
  total_money = 15 →
  money_left = 6 →
  cost_bread = 2 →
  cost_butter = 3 →
  total_money - money_left - (cost_bread + cost_butter) = cost_juice →
  compute_ratio total_money money_left cost_bread cost_butter cost_juice = 2 :=
by
  intros
  sorry

end ratio_of_juice_to_bread

namespace horse_food_needed

import Mathlib

theorem horse_food_needed
  (ratio_sheep_horses : ℕ := 6)
  (ratio_horses_sheep : ℕ := 7)
  (horse_food_per_day : ℕ := 230)
  (sheep_on_farm : ℕ := 48)
  (units : ℕ := sheep_on_farm / ratio_sheep_horses)
  (horses_on_farm : ℕ := units * ratio_horses_sheep) :
  horses_on_farm * horse_food_per_day = 12880 := by
  sorry

end horse_food_needed

namespace wire_cut

import Mathlib

theorem wire_cut (x : ℝ) (h1 : x + (100 - x) = 100) (h2 : x = (7/13) * (100 - x)) : x = 35 :=
sorry

end wire_cut

namespace smallest_y_square

import Mathlib

theorem smallest_y_square (y n : ℕ) (h1 : y = 10) (h2 : ∀ m : ℕ, (∃ z : ℕ, m * y = z^2) ↔ (m = n)) : n = 10 :=
sorry

end smallest_y_square

namespace distance_between_towns_proof

import Mathlib

noncomputable def distance_between_towns : ℕ :=
  let distance := 300
  let time_after_departure := 2
  let remaining_distance := 40
  let speed_difference := 10
  let total_distance_covered := distance - remaining_distance
  let speed_slower_train := 60
  let speed_faster_train := speed_slower_train + speed_difference
  let relative_speed := speed_slower_train + speed_faster_train
  distance

theorem distance_between_towns_proof 
  (distance : ℕ) 
  (time_after_departure : ℕ) 
  (remaining_distance : ℕ) 
  (speed_difference : ℕ) 
  (h1 : distance = 300) 
  (h2 : time_after_departure = 2) 
  (h3 : remaining_distance = 40) 
  (h4 : speed_difference = 10) 
  (speed_slower_train speed_faster_train relative_speed : ℕ)
  (h_speed_faster : speed_faster_train = speed_slower_train + speed_difference)
  (h_relative_speed : relative_speed = speed_slower_train + speed_faster_train) :
  distance = 300 :=
by {
  sorry
}

end distance_between_towns_proof

namespace arithmetic_seq_sum

import Mathlib

/-- Given an arithmetic sequence {a_n} such that a_5 + a_6 + a_7 = 15,
prove that the sum of the first 11 terms of the sequence S_11 is 55. -/
theorem arithmetic_seq_sum (a : ℕ → ℕ) (S : ℕ → ℕ)
  (h₁ : a 5 + a 6 + a 7 = 15)
  (h₂ : ∀ n, S n = n * (a 1 + a n) / 2) :
  S 11 = 55 :=
sorry

end arithmetic_seq_sum

namespace vermont_clicked_ads

import Mathlib

theorem vermont_clicked_ads :
  let ads1 := 12
  let ads2 := 2 * ads1
  let ads3 := ads2 + 24
  let ads4 := 3 * ads2 / 4
  let total_ads := ads1 + ads2 + ads3 + ads4
  let ads_clicked := 2 * total_ads / 3
  ads_clicked = 68 := by
  let ads1 := 12
  let ads2 := 2 * ads1
  let ads3 := ads2 + 24
  let ads4 := 3 * ads2 / 4
  let total_ads := ads1 + ads2 + ads3 + ads4
  let ads_clicked := 2 * total_ads / 3
  have h1 : ads_clicked = 68 := by sorry
  exact h1

end vermont_clicked_ads

namespace cubic_yard_to_cubic_meter

import Mathlib

theorem cubic_yard_to_cubic_meter : 
  let yard_to_foot := 3
  let foot_to_meter := 0.3048
  let side_length_in_meters := yard_to_foot * foot_to_meter
  (side_length_in_meters)^3 = 0.764554 :=
by
  sorry

end cubic_yard_to_cubic_meter

namespace discount_is_10_percent

import Mathlib

variable (C : ℝ)  -- Cost of the item
variable (S S' : ℝ)  -- Selling prices with and without discount

-- Conditions
def condition1 : Prop := S = 1.20 * C
def condition2 : Prop := S' = 1.30 * C

-- The proposition to prove
theorem discount_is_10_percent (h1 : condition1 C S) (h2 : condition2 C S') : S' - S = 0.10 * C := by
  sorry

end discount_is_10_percent

namespace number_of_parallelograms

import Mathlib

def binom (n k : ℕ) : ℕ := Nat.choose n k

theorem number_of_parallelograms (n : ℕ) : 
  (∑ i in Finset.range (n + 2 + 1), binom i 4) = 3 * binom (n + 2) 4 :=
by
  sorry

end number_of_parallelograms

namespace range_of_u

import Mathlib

variable (a b u : ℝ)

theorem range_of_u (ha : a > 0) (hb : b > 0) (hab : a + b = 1) : 
  (∀ x : ℝ, x > 0 → a^2 + b^2 ≥ x ↔ x ≤ 16) :=
sorry

end range_of_u

namespace discounted_price

import Mathlib

variable (marked_price : ℝ) (discount_rate : ℝ)
variable (marked_price_def : marked_price = 150)
variable (discount_rate_def : discount_rate = 20)

theorem discounted_price (hmp : marked_price = 150) (hdr : discount_rate = 20) : 
  marked_price - (discount_rate / 100) * marked_price = 120 := by
  rw [hmp, hdr]
  sorry

end discounted_price

namespace arnold_total_protein

import Mathlib

-- Definitions and conditions
def collagen_protein_per_scoop : ℕ := 18 / 2
def protein_powder_protein_per_scoop : ℕ := 21
def steak_protein : ℕ := 56

def collagen_scoops : ℕ := 1
def protein_powder_scoops : ℕ := 1
def steaks : ℕ := 1

-- Statement of the theorem/problem
theorem arnold_total_protein : 
  (collagen_protein_per_scoop * collagen_scoops) + 
  (protein_powder_protein_per_scoop * protein_powder_scoops) + 
  (steak_protein * steaks) = 86 :=
by
  sorry

end arnold_total_protein

namespace total_bricks_calculation

import Mathlib

def bricks_in_row : Nat := 30
def rows_in_wall : Nat := 50
def number_of_walls : Nat := 2
def total_bricks_for_both_walls : Nat := 3000

theorem total_bricks_calculation (h1 : bricks_in_row = 30) 
                                      (h2 : rows_in_wall = 50) 
                                      (h3 : number_of_walls = 2) : 
                                      bricks_in_row * rows_in_wall * number_of_walls = total_bricks_for_both_walls :=
by
  sorry

end total_bricks_calculation

namespace benny_leftover_money

import Mathlib

-- Define the conditions
def initial_money : ℕ := 67
def spent_money : ℕ := 34

-- Define the leftover money calculation
def leftover_money : ℕ := initial_money - spent_money

-- Prove that Benny had 33 dollars left over
theorem benny_leftover_money : leftover_money = 33 :=
by 
  -- Proof
  sorry

end benny_leftover_money

namespace solve_system_of_equations

import Mathlib

theorem solve_system_of_equations :
  ∃ (x y : ℤ), 2 * x + y = 7 ∧ 4 * x + 5 * y = 11 ∧ x = 4 ∧ y = -1 :=
by
  sorry

end solve_system_of_equations

namespace evaluate_expression

import Mathlib

theorem evaluate_expression : (2^3002 * 3^3004) / (6^3003) = (3 / 2) := by
  sorry

end evaluate_expression

namespace range_of_x_in_function

import Mathlib

theorem range_of_x_in_function : ∀ (x : ℝ), (2 - x ≥ 0) ∧ (x + 2 ≠ 0) ↔ (x ≤ 2 ∧ x ≠ -2) :=
by
  intro x
  sorry

end range_of_x_in_function

namespace mean_of_points_scored

import Mathlib

def mean (lst : List ℕ) : ℚ :=
  (lst.sum : ℚ) / lst.length

theorem mean_of_points_scored (lst : List ℕ)
  (h1 : lst = [81, 73, 83, 86, 73]) : 
  mean lst = 79.2 :=
by
  rw [h1, mean]
  sorry

end mean_of_points_scored

namespace solve_for_y

import Mathlib

theorem solve_for_y : ∀ y : ℝ, (y - 5)^3 = (1 / 27)⁻¹ → y = 8 :=
by
  intro y
  intro h
  sorry

end solve_for_y

namespace white_rabbit_hop_distance_per_minute

import Mathlib

-- Definitions for given conditions
def brown_hop_per_minute : ℕ := 12
def total_distance_in_5_minutes : ℕ := 135
def brown_distance_in_5_minutes : ℕ := 5 * brown_hop_per_minute

-- The statement we need to prove
theorem white_rabbit_hop_distance_per_minute (W : ℕ) (h1 : brown_hop_per_minute = 12) (h2 : total_distance_in_5_minutes = 135) :
  W = 15 :=
by
  sorry

end white_rabbit_hop_distance_per_minute

namespace expected_score_shooting_competition

import Mathlib

theorem expected_score_shooting_competition (hit_rate : ℝ)
  (miss_both_score : ℝ) (hit_one_score : ℝ) (hit_both_score : ℝ)
  (prob_0 : ℝ) (prob_10 : ℝ) (prob_15 : ℝ) :
  hit_rate = 4 / 5 →
  miss_both_score = 0 →
  hit_one_score = 10 →
  hit_both_score = 15 →
  prob_0 = (1 - 4 / 5) * (1 - 4 / 5) →
  prob_10 = 2 * (4 / 5) * (1 - 4 / 5) →
  prob_15 = (4 / 5) * (4 / 5) →
  (0 * prob_0 + 10 * prob_10 + 15 * prob_15) = 12.8 :=
by
  intros h_hit_rate h_miss_both_score h_hit_one_score h_hit_both_score
         h_prob_0 h_prob_10 h_prob_15
  sorry

end expected_score_shooting_competition

namespace 

import Mathlib

theorem x_coordinate_incenter_eq {x y : ℝ} :
  (y = 0 → x + y = 3 → x = 0) → 
  (y = x → y = -x + 3 → x = 3 / 2) :=
by
  sorry

end 

namespace a_beats_b_by_one_round_in_4_round_race_a_beats_b_by_T_a_minus_T_b

import Mathlib

noncomputable def T_a : ℝ := 7.5
noncomputable def T_b : ℝ := 10
noncomputable def rounds_a (n : ℕ) : ℝ := n * T_a
noncomputable def rounds_b (n : ℕ) : ℝ := n * T_b

theorem a_beats_b_by_one_round_in_4_round_race :
  rounds_a 4 = rounds_b 3 := by
  sorry

theorem a_beats_b_by_T_a_minus_T_b :
  T_b - T_a = 2.5 := by
  sorry

end a_beats_b_by_one_round_in_4_round_race_a_beats_b_by_T_a_minus_T_b

namespace angle_triple_complement

import Mathlib

theorem angle_triple_complement (x : ℝ) (h1 : x = 3 * (90 - x)) : x = 67.5 := 
by {
  sorry
}

end angle_triple_complement

namespace contrapositive_equiv

import Mathlib

variable {α : Type}  -- Type of elements
variable (P : Set α) (a b : α)

theorem contrapositive_equiv (h : a ∈ P → b ∉ P) : b ∈ P → a ∉ P :=
by
  sorry

end contrapositive_equiv

namespace figure_perimeter

import Mathlib

-- Define the side length of the square and the triangles.
def square_side_length : ℕ := 3
def triangle_side_length : ℕ := 2

-- Calculate the perimeter of the figure
def perimeter (a b : ℕ) : ℕ := 2 * a + 2 * b

-- Statement to prove
theorem figure_perimeter : perimeter square_side_length triangle_side_length = 10 := 
by 
  -- "sorry" denotes that the proof is omitted.
  sorry

end figure_perimeter

namespace sum_of_squares_of_coefficients

import Mathlib

theorem sum_of_squares_of_coefficients :
  let poly := 5 * (X^6 + 4 * X^4 + 2 * X^2 + 1)
  let coeffs := [5, 20, 10, 5]
  (coeffs.map (λ c => c * c)).sum = 550 := 
by
  sorry

end sum_of_squares_of_coefficients

namespace floor_difference

import Mathlib

theorem floor_difference (x : ℝ) (h : x = 15.3) : 
  (⌊ x^2 ⌋ - ⌊ x ⌋ * ⌊ x ⌋ + 5) = 14 := 
by
  -- Skipping proof
  sorry

end floor_difference

namespace factorize_expression

import Mathlib

theorem factorize_expression (x y : ℝ) : 4 * x^2 - 2 * x * y = 2 * x * (2 * x - y) := 
by
  sorry

end factorize_expression

namespace num_handshakes_ten_women

import Mathlib

def num_handshakes (n : ℕ) : ℕ :=
(n * (n - 1)) / 2

theorem num_handshakes_ten_women :
  num_handshakes 10 = 45 :=
by
  sorry

end num_handshakes_ten_women

namespace games_within_division

import Mathlib

theorem games_within_division (N M : ℕ) (h1 : N > 3 * M) (h2 : M > 5) (h3 : 2 * N + 6 * M = 76) : 2 * N = 40 :=
by {
  sorry
}

end games_within_division

namespace james_muffins_correct

import Mathlib

-- Arthur baked 115 muffins
def arthur_muffins : ℕ := 115

-- James baked 12 times as many muffins as Arthur
def james_multiplier : ℕ := 12

-- The number of muffins James baked
def james_muffins : ℕ := arthur_muffins * james_multiplier

-- The expected result
def expected_james_muffins : ℕ := 1380

-- The statement we want to prove
theorem james_muffins_correct : james_muffins = expected_james_muffins := by
  sorry

end james_muffins_correct

namespace color_blocks_probability_at_least_one_box_match

import Mathlib

/-- Given Ang, Ben, and Jasmin each having 6 blocks of different colors (red, blue, yellow, white, green, and orange) 
    and they independently place one of their blocks into each of 6 empty boxes, 
    the proof shows that the probability that at least one box receives 3 blocks all of the same color is 1/6. 
    Since 1/6 is equal to the fraction m/n where m=1 and n=6 are relatively prime, thus m+n=7. -/
theorem color_blocks_probability_at_least_one_box_match (p : ℕ × ℕ) (h : p = (1, 6)) : p.1 + p.2 = 7 :=
by {
  sorry
}

end color_blocks_probability_at_least_one_box_match

namespace common_roots_cubic_polynomials

import Mathlib

theorem common_roots_cubic_polynomials (a b : ℝ) :
  (∃ r s : ℝ, r ≠ s ∧ (r^3 + a * r^2 + 17 * r + 10 = 0) ∧ (s^3 + a * s^2 + 17 * s + 10 = 0) ∧ 
               (r^3 + b * r^2 + 20 * r + 12 = 0) ∧ (s^3 + b * s^2 + 20 * s + 12 = 0)) →
  (a, b) = (-6, -7) :=
by sorry

end common_roots_cubic_polynomials

namespace derivative_at_one

import Mathlib

-- Definition of the function
def f (x : ℝ) : ℝ := x^2

-- Condition
def x₀ : ℝ := 1

-- Problem statement
theorem derivative_at_one : (deriv f x₀) = 2 :=
sorry

end derivative_at_one

namespace angle_difference

import Mathlib

-- Define the conditions
variables (A B : ℝ) 

def is_parallelogram := A + B = 180
def smaller_angle := A = 70
def larger_angle := B = 180 - 70

-- State the theorem to be proved
theorem angle_difference (A B : ℝ) (h1 : is_parallelogram A B) (h2 : smaller_angle A) : B - A = 40 := by
  sorry

end angle_difference

namespace a7_b7_equals_29

import Mathlib

noncomputable def a : ℂ := sorry
noncomputable def b : ℂ := sorry

def cond1 := a + b = 1
def cond2 := a^2 + b^2 = 3
def cond3 := a^3 + b^3 = 4
def cond4 := a^4 + b^4 = 7
def cond5 := a^5 + b^5 = 11

theorem a7_b7_equals_29 : cond1 ∧ cond2 ∧ cond3 ∧ cond4 ∧ cond5 → a^7 + b^7 = 29 :=
by
  sorry

end a7_b7_equals_29

namespace maximize_k

import Mathlib

open Real

theorem maximize_k (x y : ℝ) (h₁ : 0 < x) (h₂ : 0 < y) (h₃ : log x + log y = 0)
  (h₄ : ∀ x y : ℝ, 0 < x → 0 < y → k * (x + 2 * y) ≤ x^2 + 4 * y^2) : k ≤ sqrt 2 :=
sorry

end maximize_k

namespace fraction_meaningful

import Mathlib

theorem fraction_meaningful (a : ℝ) : (∃ b, b = 2 / (a + 1)) → a ≠ -1 :=
by
  sorry

end fraction_meaningful

namespace candy_store_truffle_price

import Mathlib

def total_revenue : ℝ := 212
def fudge_revenue : ℝ := 20 * 2.5
def pretzels_revenue : ℝ := 3 * 12 * 2.0
def truffles_quantity : ℕ := 5 * 12

theorem candy_store_truffle_price (total_revenue fudge_revenue pretzels_revenue truffles_quantity : ℝ) : 
  (total_revenue - (fudge_revenue + pretzels_revenue)) / truffles_quantity = 1.50 := 
by 
  sorry

end candy_store_truffle_price

namespace MountainRidgeAcademy

import Mathlib 

theorem MountainRidgeAcademy (j s : ℕ) 
  (h1 : 3/4 * j = 1/2 * s) : s = 3/2 * j := 
by 
  sorry

end MountainRidgeAcademy

namespace least_number_subtracted_divisible_17

import Mathlib

theorem least_number_subtracted_divisible_17 :
  ∃ n : ℕ, 165826 - n % 17 = 0 ∧ n = 12 :=
by
  use 12
  sorry  -- Proof will go here.

end least_number_subtracted_divisible_17

namespace minimum_value_of_expression

import Mathlib

noncomputable def min_squared_distance (a b c d : ℝ) : ℝ :=
  (a - c)^2 + (b - d)^2

theorem minimum_value_of_expression
  (a b c d : ℝ)
  (h1 : 4 * a^2 + b^2 - 8 * b + 12 = 0)
  (h2 : c^2 - 8 * c + 4 * d^2 + 12 = 0) :
  min_squared_distance a b c d = 42 - 16 * Real.sqrt 5 :=
sorry

end minimum_value_of_expression

namespace intersection_with_y_axis

import Mathlib

-- Define the given function
def f (x : ℝ) := x^2 + x - 2

-- Prove that the intersection point with the y-axis is (0, -2)
theorem intersection_with_y_axis : f 0 = -2 :=
by {
  sorry
}

end intersection_with_y_axis

namespace number_of_lines

import Mathlib

-- Define point P
structure Point where
  x : ℝ
  y : ℝ

-- Define a line by its equation ax + by + c = 0
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ
  nonzero: a ≠ 0 ∨ b ≠ 0

-- Definition of a line passing through a point P
def passes_through (l : Line) (P : Point) : Prop :=
  l.a * P.x + l.b * P.y + l.c = 0

-- Definition of a line having equal intercepts on x-axis and y-axis
def equal_intercepts (l : Line) : Prop :=
  l.a ≠ 0 ∧ l.b ≠ 0 ∧ l.a = l.b

-- Definition of a specific point P
def P : Point := { x := 1, y := 2 }

-- The theorem statement
theorem number_of_lines : ∃ (lines : Finset Line), (∀ l ∈ lines, passes_through l P ∧ equal_intercepts l) ∧ lines.card = 2 := by
  sorry

end number_of_lines

namespace pastries_total

import Mathlib

-- We start by defining the conditions
def Calvin_pastries (Frank_pastries Grace_pastries : ℕ) : ℕ := Frank_pastries + 8
def Phoebe_pastries (Frank_pastries Grace_pastries : ℕ) : ℕ := Frank_pastries + 8
def Grace_pastries : ℕ := 30
def have_same_pastries (Calvin_pastries Phoebe_pastries Grace_pastries : ℕ) : Prop :=
  Calvin_pastries + 5 = Grace_pastries ∧ Phoebe_pastries + 5 = Grace_pastries

-- Total number of pastries held by Calvin, Phoebe, Frank, and Grace
def total_pastries (Frank_pastries Calvin_pastries Phoebe_pastries Grace_pastries : ℕ) : ℕ :=
  Frank_pastries + Calvin_pastries + Phoebe_pastries + Grace_pastries

-- The statement to prove
theorem pastries_total (Frank_pastries : ℕ) : 
  have_same_pastries (Calvin_pastries Frank_pastries Grace_pastries) (Phoebe_pastries Frank_pastries Grace_pastries) Grace_pastries → 
  Frank_pastries + Calvin_pastries Frank_pastries Grace_pastries + Phoebe_pastries Frank_pastries Grace_pastries + Grace_pastries = 97 :=
by
  sorry

end pastries_total

namespace problem_statement

import Mathlib

noncomputable def necessary_but_not_sufficient_condition (x y : ℝ) (hx : x > 0) : Prop :=
  (x > |y| → x > y) ∧ ¬ (x > y → x > |y|)

theorem problem_statement
  (x y : ℝ)
  (hx : x > 0)
  : necessary_but_not_sufficient_condition x y hx :=
sorry

end problem_statement

namespace sum_a5_a6_a7

import Mathlib

def geometric_sequence (a : ℕ → ℤ) : Prop :=
  ∃ q : ℤ, ∀ n : ℕ, a (n + 1) = q * a n

variables (a : ℕ → ℤ)
variables (h_geo : geometric_sequence a)
variables (h1 : a 2 + a 3 = 1)
variables (h2 : a 3 + a 4 = -2)

theorem sum_a5_a6_a7 : a 5 + a 6 + a 7 = 24 :=
by
  sorry

end sum_a5_a6_a7

namespace not_p_suff_not_q

import Mathlib

theorem not_p_suff_not_q (x : ℝ) :
  ¬(|x| ≥ 1) → ¬(x^2 + x - 6 ≥ 0) :=
sorry

end not_p_suff_not_q

namespace odd_coefficients_in_binomial_expansion

import Mathlib

theorem odd_coefficients_in_binomial_expansion :
  let a : Fin 9 → ℕ := fun k => Nat.choose 8 k
  (Finset.filter (fun k => a k % 2 = 1) (Finset.Icc 0 8)).card = 2 := by
  sorry

end odd_coefficients_in_binomial_expansion

namespace eighth_grade_students

import Mathlib

def avg_books (total_books : ℕ) (num_students : ℕ) : ℚ :=
  total_books / num_students

theorem eighth_grade_students (x : ℕ) (y : ℕ)
  (h1 : x + y = 1800)
  (h2 : y = x - 150)
  (h3 : avg_books x 1800 = 1.5 * avg_books (x - 150) 1800) :
  y = 450 :=
by {
  sorry
}

end eighth_grade_students

namespace sum_after_50_rounds

import Mathlib

def initial_states : List ℤ := [1, 0, -1]

def operation (n : ℤ) : ℤ :=
  match n with
  | 1   => n * n * n
  | 0   => n * n
  | -1  => -n
  | _ => n  -- although not necessary for current problem, this covers other possible states

def process_calculator (state : ℤ) (times: ℕ) : ℤ :=
  if state = 1 then state
  else if state = 0 then state
  else if state = -1 then state * (-1) ^ times
  else state

theorem sum_after_50_rounds :
  let final_states := initial_states.map (fun s => process_calculator s 50)
  final_states.sum = 2 := by
  simp only [initial_states, process_calculator]
  simp
  sorry

end sum_after_50_rounds

namespace min_value_sum_reciprocal

import Mathlib

open Real

theorem min_value_sum_reciprocal (x y z : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) 
    (h_pos_z : 0 < z) (h_sum : x + y + z = 3) : 
    1 / (x + 3 * y) + 1 / (y + 3 * z) + 1 / (z + 3 * x) ≥ 3 / 4 :=
by
  sorry

end min_value_sum_reciprocal

namespace number_of_dots_in_120_circles

import Mathlib

theorem number_of_dots_in_120_circles :
  ∃ n : ℕ, (n = 14) ∧ (∀ m : ℕ, m * (m + 1) / 2 + m ≤ 120 → m ≤ n) :=
by
  sorry

end number_of_dots_in_120_circles

namespace integer_pairs_satisfy_equation

import Mathlib

theorem integer_pairs_satisfy_equation :
  ∀ (x y : ℤ), (x^2 * y + y^2 = x^3) → (x = 0 ∧ y = 0) ∨ (x = -4 ∧ y = -8) :=
by
  sorry

end integer_pairs_satisfy_equation

namespace find_e

import Mathlib

noncomputable def f (x : ℝ) (c : ℝ) := 5 * x + 2 * c

noncomputable def g (x : ℝ) (c : ℝ) := c * x^2 + 3

noncomputable def fg (x : ℝ) (c : ℝ) := f (g x c) c

theorem find_e (c : ℝ) (e : ℝ) (h1 : f (g x c) c = 15 * x^2 + e) (h2 : 5 * c = 15) : e = 21 :=
by
  sorry

end find_e

namespace total_scarves

import Mathlib

def total_yarns_red : ℕ := 2
def total_yarns_blue : ℕ := 6
def total_yarns_yellow : ℕ := 4
def scarves_per_yarn : ℕ := 3

theorem total_scarves : 
  (total_yarns_red * scarves_per_yarn) + 
  (total_yarns_blue * scarves_per_yarn) + 
  (total_yarns_yellow * scarves_per_yarn) = 36 := 
by
  sorry

end total_scarves

namespace trader_loss_percentage

import Mathlib

def profit_loss_percentage (SP1 SP2 CP1 CP2 : ℚ) : ℚ :=
  ((SP1 + SP2) - (CP1 + CP2)) / (CP1 + CP2) * 100

theorem trader_loss_percentage :
  let SP1 := 325475
  let SP2 := 325475
  let CP1 := SP1 / (1 + 0.10)
  let CP2 := SP2 / (1 - 0.10)
  profit_loss_percentage SP1 SP2 CP1 CP2 = -1 := by
  sorry

end trader_loss_percentage

namespace final_hair_length_is_14

import Mathlib

def initial_hair_length : ℕ := 24

def half_hair_cut (l : ℕ) : ℕ := l / 2

def hair_growth (l : ℕ) : ℕ := l + 4

def final_hair_cut (l : ℕ) : ℕ := l - 2

theorem final_hair_length_is_14 :
  final_hair_cut (hair_growth (half_hair_cut initial_hair_length)) = 14 := by
  sorry

end final_hair_length_is_14

namespace parabola_vertex_y_axis_opens_upwards

import Mathlib

theorem parabola_vertex_y_axis_opens_upwards :
  ∃ (a b c : ℝ), (a > 0) ∧ (b = 0) ∧ y = a * x^2 + b * x + c := 
sorry

end parabola_vertex_y_axis_opens_upwards

namespace polynomial_value

import Mathlib

variable (a b : ℝ)

theorem polynomial_value :
  2 * a + 3 * b = 5 → 6 * a + 9 * b - 12 = 3 :=
by
  intro h
  sorry

end polynomial_value

namespace arctan_addition_formula

import Mathlib

noncomputable def arctan_add : ℝ :=
  Real.arctan (1 / 3) + Real.arctan (3 / 8)

theorem arctan_addition_formula :
  arctan_add = Real.arctan (17 / 21) :=
by
  sorry

end arctan_addition_formula

namespace polynomial_coeff_fraction_eq_neg_122_div_121

import Mathlib

theorem polynomial_coeff_fraction_eq_neg_122_div_121
  (a0 a1 a2 a3 a4 a5 : ℤ)
  (h1 : (2 - 1) ^ 5 = a0 + a1 * 1 + a2 * 1^2 + a3 * 1^3 + a4 * 1^4 + a5 * 1^5)
  (h2 : (2 - (-1)) ^ 5 = a0 + a1 * (-1) + a2 * (-1)^2 + a3 * (-1)^3 + a4 * (-1)^4 + a5 * (-1)^5)
  (h_sum1 : a0 + a1 + a2 + a3 + a4 + a5 = 1)
  (h_sum2 : a0 - a1 + a2 - a3 + a4 - a5 = 243) :
  (a0 + a2 + a4) / (a1 + a3 + a5) = - 122 / 121 :=
sorry

end polynomial_coeff_fraction_eq_neg_122_div_121

namespace find_reflection_line

import Mathlib

-- Definition of the original and reflected vertices
structure Point :=
  (x : ℝ)
  (y : ℝ)

def D : Point := {x := 1, y := 2}
def E : Point := {x := 6, y := 7}
def F : Point := {x := -5, y := 5}
def D' : Point := {x := 1, y := -4}
def E' : Point := {x := 6, y := -9}
def F' : Point := {x := -5, y := -7}

theorem find_reflection_line (M : ℝ) :
  (D.y + D'.y) / 2 = M ∧ (E.y + E'.y) / 2 = M ∧ (F.y + F'.y) / 2 = M → M = -1 :=
by
  intros
  sorry

end find_reflection_line

namespace sum_floor_expression

import Mathlib

theorem sum_floor_expression (p : ℕ) (h_prime : Nat.Prime p) (h_form : ∃ k : ℕ, p = 4 * k + 1) :
  ∑ i in Finset.range p \ {0}, (Int.floor ((2 * i ^ 2 : ℤ) / p) - 2 * Int.floor ((i ^ 2 : ℤ) / p)) = (p - 1) / 2 := 
sorry

end sum_floor_expression

namespace general_formula_a_n_T_n_greater_than_S_n

import Mathlib

variable {n : ℕ}
variable {a S T : ℕ → ℕ}

-- Initial Conditions
def a_n (n : ℕ) : ℕ := 2 * n + 3
def S_n (n : ℕ) : ℕ := (n * (2 * n + 8)) / 2
def b_n (n : ℕ) : ℕ := if n % 2 = 1 then a_n n - 6 else 2 * a_n n
def T_n (n : ℕ) : ℕ := (n / 2 * (6 * n + 14) / 2) + ((n + 1) / 2 * (6 * n + 14) / 2) - 10

-- Given
axiom S_4_eq_32 : S_n 4 = 32
axiom T_3_eq_16 : T_n 3 = 16

-- First proof: general formula of {a_n}
theorem general_formula_a_n : ∀ n : ℕ, a_n n = 2 * n + 3 := by
  sorry

-- Second proof: For n > 5: T_n > S_n
theorem T_n_greater_than_S_n (n : ℕ) (h : n > 5) : T_n n > S_n n := by
  sorry

end general_formula_a_n_T_n_greater_than_S_n

namespace find_multiplier

import Mathlib

variable {a b : ℝ} 

theorem find_multiplier (h1 : 3 * a = x * b) (h2 : a ≠ 0 ∧ b ≠ 0) (h3 : a / 4 = b / 3) : x = 4 :=
by
  sorry

end find_multiplier

namespace quilt_width_is_eight

import Mathlib

def length := 7
def cost_per_square_foot := 40
def total_cost := 2240
def area := total_cost / cost_per_square_foot

theorem quilt_width_is_eight :
  area / length = 8 := by
  sorry

end quilt_width_is_eight

namespace max_heaps_of_stones

import Mathlib

noncomputable def stones : ℕ := 660
def max_heaps : ℕ := 30
def differs_less_than_twice (a b : ℕ) : Prop := a < 2 * b

theorem max_heaps_of_stones (h : ℕ) :
  (∀ i j : ℕ, i ≠ j → differs_less_than_twice (i+j) stones) → max_heaps = 30 :=
sorry

end max_heaps_of_stones

namespace find_a_b_c

import Mathlib

theorem find_a_b_c :
  ∃ a b c : ℕ, a = 1 ∧ b = 17 ∧ c = 2 ∧ (Nat.gcd a c = 1) ∧ a + b + c = 20 :=
by {
  -- the proof would go here
  sorry
}

end find_a_b_c

namespace ratio_change_factor_is_5

import Mathlib

-- Definitions based on problem conditions
def original_bleach : ℕ := 4
def original_detergent : ℕ := 40
def original_water : ℕ := 100

-- Simplified original ratio
def original_bleach_ratio : ℕ := original_bleach / 4
def original_detergent_ratio : ℕ := original_detergent / 4
def original_water_ratio : ℕ := original_water / 4

-- Altered conditions
def altered_detergent : ℕ := 60
def altered_water : ℕ := 300

-- Simplified altered ratio of detergent to water
def altered_detergent_ratio : ℕ := altered_detergent / 60
def altered_water_ratio : ℕ := altered_water / 60

-- Proof that the ratio change factor is 5
theorem ratio_change_factor_is_5 : 
  (original_water_ratio / altered_water_ratio) = 5
  := by
    have original_detergent_ratio : ℕ := 10
    have original_water_ratio : ℕ := 25
    have altered_detergent_ratio : ℕ := 1
    have altered_water_ratio : ℕ := 5
    sorry

end ratio_change_factor_is_5

namespace symmetric_colors_different_at_8281_div_2

import Mathlib

def is_red (n : ℕ) : Prop :=
  ∃ (x y : ℕ), x > 0 ∧ y > 0 ∧ n = 81 * x + 100 * y

def is_blue (n : ℕ) : Prop :=
  ¬ is_red n

theorem symmetric_colors_different_at_8281_div_2 :
  ∃ n : ℕ, (is_red n ∧ is_blue (8281 - n)) ∨ (is_blue n ∧ is_red (8281 - n)) ∧ 2 * n = 8281 :=
by
  sorry

end symmetric_colors_different_at_8281_div_2

namespace difference_between_advertised_and_actual_mileage

import Mathlib

def advertised_mileage : ℕ := 35

def city_mileage_regular : ℕ := 30
def highway_mileage_premium : ℕ := 40
def traffic_mileage_diesel : ℕ := 32

def gallons_regular : ℕ := 4
def gallons_premium : ℕ := 4
def gallons_diesel : ℕ := 4

def total_miles_driven : ℕ :=
  (gallons_regular * city_mileage_regular) + 
  (gallons_premium * highway_mileage_premium) + 
  (gallons_diesel * traffic_mileage_diesel)

def total_gallons_used : ℕ :=
  gallons_regular + gallons_premium + gallons_diesel

def weighted_average_mpg : ℤ :=
  total_miles_driven / total_gallons_used

theorem difference_between_advertised_and_actual_mileage :
  advertised_mileage - weighted_average_mpg = 1 :=
by
  -- proof to be filled in
  sorry

end difference_between_advertised_and_actual_mileage

namespace meet_at_centroid

import Mathlib

-- Definitions of positions
def Harry : ℝ × ℝ := (10, -3)
def Sandy : ℝ × ℝ := (2, 7)
def Ron : ℝ × ℝ := (6, 1)

-- Mathematical proof problem statement
theorem meet_at_centroid : 
    (Harry.1 + Sandy.1 + Ron.1) / 3 = 6 ∧ (Harry.2 + Sandy.2 + Ron.2) / 3 = 5 / 3 := 
by
  sorry

end meet_at_centroid

namespace distance_to_base_is_42

import Mathlib

theorem distance_to_base_is_42 (x : ℕ) (hx : 4 * x + 3 * (x + 3) = x + (x + 1) + (x + 2) + (x + 3) + (x + 4) + (x + 5) + (x + 6)) :
  4 * x = 36 ∨ 4 * x + 6 = 42 := 
by
  sorry

end distance_to_base_is_42

namespace problem

import Mathlib

def A : Set ℤ := {-1, 0, 1}
def B : Set ℤ := {x | ∃ t ∈ A, x = t^2}

theorem problem (A_def : A = {-1, 0, 1}) : B = {0, 1} :=
by sorry

end problem

namespace purely_imaginary_iff_real_iff_second_quadrant_iff

import Mathlib

def Z (m : ℝ) : ℂ := ⟨m^2 - 2 * m - 3, m^2 + 3 * m + 2⟩

theorem purely_imaginary_iff (m : ℝ) : (Z m).re = 0 ∧ (Z m).im ≠ 0 ↔ m = 3 :=
by sorry

theorem real_iff (m : ℝ) : (Z m).im = 0 ↔ m = -1 ∨ m = -2 :=
by sorry

theorem second_quadrant_iff (m : ℝ) : (Z m).re < 0 ∧ (Z m).im > 0 ↔ -1 < m ∧ m < 3 :=
by sorry

end purely_imaginary_iff_real_iff_second_quadrant_iff

namespace sequence_general_term

import Mathlib

theorem sequence_general_term 
  (a : ℕ → ℝ)
  (h₀ : a 1 = 1)
  (h₁ : a 2 = 1 / 3)
  (h₂ : ∀ n : ℕ, 2 ≤ n → a n * a (n - 1) + a n * a (n + 1) = 2 * a (n - 1) * a (n + 1)) :
  ∀ n : ℕ, 1 ≤ n → a n = 1 / (2 * n - 1) := 
by
  sorry

end sequence_general_term

namespace 

import Mathlib

def A (a : ℝ) : Set ℝ := {1, 3, a^2}
def B (a : ℝ) : Set ℝ := {1, a + 2}

theorem num_real_a_satisfy_union {a : ℝ} : (A a ∪ B a) = A a → ∃! a, (A a ∪ B a) = A a := 
by sorry

end 

namespace goods_train_speed

import Mathlib

noncomputable def passenger_train_speed := 64 -- in km/h
noncomputable def passing_time := 18 -- in seconds
noncomputable def goods_train_length := 420 -- in meters
noncomputable def relative_speed_kmh := 84 -- in km/h (derived from solution)

theorem goods_train_speed :
  (∃ V_g, relative_speed_kmh = V_g + passenger_train_speed) →
  (goods_train_length / (passing_time / 3600): ℝ) = relative_speed_kmh →
  V_g = 20 :=
by
  intro h1 h2
  sorry

end goods_train_speed

namespace fuel_remaining

import Mathlib

-- Definitions given in the conditions of the original problem
def initial_fuel : ℕ := 48
def fuel_consumption_rate : ℕ := 8

-- Lean 4 statement of the mathematical proof problem
theorem fuel_remaining (x : ℕ) : 
  ∃ y : ℕ, y = initial_fuel - fuel_consumption_rate * x :=
sorry

end fuel_remaining

namespace sin_add_alpha

import Mathlib

theorem sin_add_alpha (α : ℝ) (h : Real.cos (α - π / 3) = -1 / 2) : 
    Real.sin (π / 6 + α) = -1 / 2 :=
sorry

end sin_add_alpha

namespace geometric_sequence_sum

import Mathlib

variable (a : ℕ → ℝ)
variable (q : ℝ)

axiom h1 : a 1 + a 2 = 20
axiom h2 : a 3 + a 4 = 40
axiom h3 : q^2 = 2

theorem geometric_sequence_sum : a 5 + a 6 = 80 :=
by
  sorry

end geometric_sequence_sum

namespace tv_cost

import Mathlib

-- Definitions from the problem conditions
def fraction_on_furniture : ℚ := 3 / 4
def total_savings : ℚ := 1800
def fraction_on_tv : ℚ := 1 - fraction_on_furniture  -- Fraction of savings on TV

-- The proof problem statement
theorem tv_cost : total_savings * fraction_on_tv = 450 := by
  sorry

end tv_cost

namespace steve_paid_18

import Mathlib

-- Define the conditions
def mike_price : ℝ := 5
def steve_multiplier : ℝ := 2
def shipping_rate : ℝ := 0.8

-- Define Steve's cost calculation
def steve_total_cost : ℝ :=
  let steve_dvd_price := steve_multiplier * mike_price
  let shipping_cost := shipping_rate * steve_dvd_price
  steve_dvd_price + shipping_cost

-- Prove that Steve's total payment is 18.
theorem steve_paid_18 : steve_total_cost = 18 := by
  -- Provide a placeholder for the proof
  sorry

end steve_paid_18

namespace find_r4

import Mathlib

variable (r : ℝ)

theorem find_r4 (h : (r + 1/r)^2 = 5) :
  r^4 + 1/r^4 = 7 :=
sorry

end find_r4

namespace nico_reads_wednesday

import Mathlib

def pages_monday := 20
def pages_tuesday := 12
def total_pages := 51
def pages_wednesday := total_pages - (pages_monday + pages_tuesday) 

theorem nico_reads_wednesday :
  pages_wednesday = 19 :=
by
  sorry

end nico_reads_wednesday

namespace second_number_value

import Mathlib

theorem second_number_value (A B C : ℝ) 
    (h1 : A + B + C = 98) 
    (h2 : A = (2/3) * B) 
    (h3 : C = (8/5) * B) : 
    B = 30 :=
by 
  sorry

end second_number_value

namespace parameterization_theorem

import Mathlib

theorem parameterization_theorem (a b c d : ℝ) (h1 : b = 1) (h2 : d = -3) (h3 : a + b = 4) (h4 : c + d = 5) :
  a^2 + b^2 + c^2 + d^2 = 83 :=
by
  sorry

end parameterization_theorem

namespace sum_floor_div_2_pow_25

import Mathlib

/--
Prove that the remainder of the sum 
∑_i=0^2015 ⌊2^i / 25⌋ 
when divided by 100 is 14,
where ⌊x⌋ denotes the floor of x.
-/
theorem sum_floor_div_2_pow_25 :
  (∑ i in Finset.range 2016, (2^i / 25 : ℤ)) % 100 = 14 :=
by
  sorry

end sum_floor_div_2_pow_25

namespace cos_sq_sub_sin_sq

import Mathlib

noncomputable def cos_sq_sub_sin_sq_eq := 
  ∀ (α : ℝ), α ∈ Set.Ioo 0 Real.pi → (Real.sin α + Real.cos α = Real.sqrt 3 / 3) →
  (Real.cos α) ^ 2 - (Real.sin α) ^ 2 = -Real.sqrt 5 / 3

theorem cos_sq_sub_sin_sq :
  cos_sq_sub_sin_sq_eq := 
by
  intros α hα h_eq
  sorry

end cos_sq_sub_sin_sq

namespace intersection_of_A_and_B

import Mathlib

def A : Set ℕ := {0, 1, 2}
def B : Set ℕ := {1, 2, 3, 4}

theorem intersection_of_A_and_B :
  A ∩ B = {1, 2} :=
by sorry

end intersection_of_A_and_B

namespace combinations_with_common_subjects

import Mathlib

-- Conditions and known facts
def subjects : Finset String := {"politics", "history", "geography", "physics", "chemistry", "biology", "technology"}
def personA_must_choose : Finset String := {"physics", "politics"}
def personB_cannot_choose : String := "technology"
def total_combinations : Nat := Nat.choose 7 3
def valid_combinations : Nat := Nat.choose 5 1 * Nat.choose 6 3
def non_common_subject_combinations : Nat := 4 + 4

-- We need to prove this statement
theorem combinations_with_common_subjects : valid_combinations - non_common_subject_combinations = 92 := by
  sorry

end combinations_with_common_subjects

namespace max_a_avoiding_lattice_points

import Mathlib

def is_lattice_point (x y : ℤ) : Prop :=
  true  -- Placeholder for (x, y) being in lattice points.

def passes_through_lattice_point (m : ℚ) (x : ℤ) : Prop :=
  is_lattice_point x (⌊m * x + 2⌋)

theorem max_a_avoiding_lattice_points :
  ∀ {a : ℚ}, (∀ x : ℤ, (0 < x ∧ x ≤ 100) → ¬passes_through_lattice_point ((1 : ℚ) / 2) x ∧ ¬passes_through_lattice_point (a - 1) x) →
  a = 50 / 99 :=
by
  sorry

end max_a_avoiding_lattice_points

namespace Cartesian_eq_C2_correct_distance_AB_correct

import Mathlib

-- Part I: Proving the Cartesian equation of curve (C2)
noncomputable def equation_of_C2 (x y : ℝ) (α : ℝ) : Prop :=
  x = 4 * Real.cos α ∧ y = 4 + 4 * Real.sin α

def Cartesian_eq_C2 (x y : ℝ) : Prop :=
  x^2 + (y - 4)^2 = 16

theorem Cartesian_eq_C2_correct (x y α : ℝ) (h : equation_of_C2 x y α) : Cartesian_eq_C2 x y :=
by sorry

-- Part II: Proving the distance |AB| given polar equations
noncomputable def polar_eq_C1 (theta : ℝ) : ℝ :=
  4 * Real.sin theta

noncomputable def polar_eq_C2 (theta : ℝ) : ℝ :=
  8 * Real.sin theta

def distance_AB (rho1 rho2 : ℝ) : ℝ :=
  abs (rho1 - rho2)

theorem distance_AB_correct : distance_AB (polar_eq_C1 (π / 3)) (polar_eq_C2 (π / 3)) = 2 * Real.sqrt 3 :=
by sorry

end Cartesian_eq_C2_correct_distance_AB_correct

namespace routes_from_A_to_B_in_4_by_3_grid

import Mathlib

-- Problem: Given a 4 by 3 rectangular grid, and movement allowing only right (R) or down (D),
-- prove that the number of different routes from point A to point B is 35.
def routes_4_by_3 : ℕ :=
  let n_moves := 3 + 4  -- Total moves required are 3 Rs and 4 Ds
  let r_moves := 3      -- Number of Right moves (R)
  Nat.choose (n_moves) (r_moves) -- Number of ways to choose 3 Rs from 7 moves

theorem routes_from_A_to_B_in_4_by_3_grid : routes_4_by_3 = 35 := by {
  sorry -- Proof omitted
}

end routes_from_A_to_B_in_4_by_3_grid

namespace common_point_exists

import Mathlib

theorem common_point_exists (a b c : ℝ) :
  ∃ x y : ℝ, y = a * x ^ 2 - b * x + c ∧ y = b * x ^ 2 - c * x + a ∧ y = c * x ^ 2 - a * x + b :=
  sorry

end common_point_exists

namespace john_ate_half_package

import Mathlib

def fraction_of_package_john_ate (servings : ℕ) (calories_per_serving : ℕ) (calories_consumed : ℕ) : ℚ :=
  calories_consumed / (servings * calories_per_serving : ℚ)

theorem john_ate_half_package (servings : ℕ) (calories_per_serving : ℕ) (calories_consumed : ℕ) 
    (h_servings : servings = 3) (h_calories_per_serving : calories_per_serving = 120) (h_calories_consumed : calories_consumed = 180) :
    fraction_of_package_john_ate servings calories_per_serving calories_consumed = 1 / 2 :=
by
  -- Replace the actual proof with sorry to ensure the statement compiles.
  sorry

end john_ate_half_package

namespace simplify_exponent_multiplication

import Mathlib

theorem simplify_exponent_multiplication :
  (10 ^ 0.25) * (10 ^ 0.15) * (10 ^ 0.35) * (10 ^ 0.05) * (10 ^ 0.85) * (10 ^ 0.35) = 10 ^ 2 := by
  sorry

end simplify_exponent_multiplication

namespace minimum_number_is_correct

import Mathlib

-- Define the operations and conditions on the digits
def transform (n : ℕ) : ℕ :=
if 2 ≤ n then n - 2 + 1 else n

noncomputable def minimum_transformed_number (l : List ℕ) : List ℕ :=
l.map transform

def initial_number : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9]

def expected_number : List ℕ := [1, 0, 1, 0, 1, 0, 1, 0, 1]

theorem minimum_number_is_correct :
  minimum_transformed_number initial_number = expected_number := 
by
  -- sorry is a placeholder for the proof
  sorry

end minimum_number_is_correct

namespace find_middle_part_value

import Mathlib.Data.Real.Basic

-- Define the ratios
def ratio1 := 1 / 2
def ratio2 := 1 / 4
def ratio3 := 1 / 8

-- Total sum
def total_sum := 120

-- Parts proportional to ratios
def part1 (x : ℝ) := x
def part2 (x : ℝ) := ratio1 * x
def part3 (x : ℝ) := ratio2 * x

-- Equation representing the sum of the parts equals to the total sum
def equation (x : ℝ) : Prop :=
  part1 x + part2 x / 2 + part2 x = x * (1 + ratio1 + ratio2)

-- Defining the middle part
def middle_part (x : ℝ) := ratio1 * x

theorem find_middle_part_value :
  ∃ x : ℝ, equation x ∧ middle_part x = 34.2857 := sorry

end find_middle_part_value

namespace slope_of_perpendicular_line

import Mathlib

theorem slope_of_perpendicular_line 
  (x1 y1 x2 y2 : ℤ)
  (h : x1 = 3 ∧ y1 = -4 ∧ x2 = -6 ∧ y2 = 2) : 
∃ m : ℚ, m = 3/2 :=
by
  sorry

end slope_of_perpendicular_line

namespace sequence_values_induction_proof

import Mathlib

def seq (a : ℕ → ℤ) := a 1 = 3 ∧ ∀ n : ℕ, a (n + 1) = a n ^ 2 - 2 * n * a n + 2

theorem sequence_values (a : ℕ → ℤ) (h : seq a) :
  a 2 = 5 ∧ a 3 = 7 ∧ a 4 = 9 :=
sorry

theorem induction_proof (a : ℕ → ℤ) (h : seq a) :
  ∀ n : ℕ, a n = 2 * n + 1 :=
sorry

end sequence_values_induction_proof

namespace polyhedron_volume

import Mathlib

-- Define the properties of the polygons
def isosceles_right_triangle (a : ℝ) := a ≠ 0 ∧ ∀ (x y : ℝ), x = y

def square (side : ℝ) := side = 2

def equilateral_triangle (side : ℝ) := side = 2 * Real.sqrt 2

-- Define the conditions
def condition_AE : Prop := isosceles_right_triangle 2
def condition_B : Prop := square 2
def condition_C : Prop := square 2
def condition_D : Prop := square 2
def condition_G : Prop := equilateral_triangle (2 * Real.sqrt 2)

-- Define the polyhedron volume calculation problem
theorem polyhedron_volume (hA : condition_AE) (hE : condition_AE) (hF : condition_AE) (hB : condition_B) (hC : condition_C) (hD : condition_D) (hG : condition_G) : 
  ∃ V : ℝ, V = 16 := 
sorry

end polyhedron_volume

namespace price_per_liter_after_discount

import Mathlib

-- Define the initial conditions
def num_bottles : ℕ := 6
def liters_per_bottle : ℝ := 2
def original_total_cost : ℝ := 15
def discounted_total_cost : ℝ := 12

-- Calculate the total number of liters
def total_liters : ℝ := num_bottles * liters_per_bottle

-- Define the expected price per liter after discount
def expected_price_per_liter : ℝ := 1

-- Lean query to verify the expected price per liter
theorem price_per_liter_after_discount : (discounted_total_cost / total_liters) = expected_price_per_liter := by
  sorry

end price_per_liter_after_discount

namespace units_digit_product_is_2

import Mathlib

def units_digit_product : ℕ := 
  (10 * 11 * 12 * 13 * 14 * 15 * 16) / 800 % 10

theorem units_digit_product_is_2 : units_digit_product = 2 := 
by
  sorry

end units_digit_product_is_2

namespace 

import Mathlib

theorem cone_base_diameter {r l : ℝ} 
  (h₁ : π * r * l + π * r^2 = 3 * π) 
  (h₂ : 2 * π * r = π * l) : 
  2 * r = 2 :=
by
  sorry

end 

namespace triangle_area_is_correct

import Mathlib

structure Point where
  x : ℝ
  y : ℝ

noncomputable def area_of_triangle (A B C : Point) : ℝ :=
  0.5 * (abs (A.x * (B.y - C.y) + B.x * (C.y - A.y) + C.x * (A.y - B.y)))

def A : Point := ⟨0, 0⟩
def B : Point := ⟨2, 2⟩
def C : Point := ⟨2, 0⟩

theorem triangle_area_is_correct : area_of_triangle A B C = 2 := by
  sorry

end triangle_area_is_correct

namespace Claire_takes_6_photos

import Mathlib

-- Define the number of photos Claire has taken
variable (C : ℕ)

-- Define the conditions as stated in the problem
def Lisa_photos := 3 * C
def Robert_photos := C + 12
def same_number_photos := Lisa_photos C = Robert_photos C

-- The goal is to prove that C = 6
theorem Claire_takes_6_photos (h : same_number_photos C) : C = 6 := by
  sorry

end Claire_takes_6_photos

namespace emails_received_in_afternoon

import Mathlib

theorem emails_received_in_afternoon (A : ℕ) 
  (h1 : 4 + (A - 3) = 9) : 
  A = 8 :=
by
  sorry

end emails_received_in_afternoon

namespace minimize_J

import Mathlib

noncomputable def H (p q : ℝ) : ℝ :=
  -3 * p * q + 4 * p * (1 - q) + 4 * (1 - p) * q - 5 * (1 - p) * (1 - q)

noncomputable def J (p : ℝ) : ℝ :=
  if p < 0 then 0 else if p > 1 then 1 else if (9 * p - 5 > 4 - 7 * p) then 9 * p - 5 else 4 - 7 * p

theorem minimize_J :
  ∃ (p : ℝ), 0 ≤ p ∧ p ≤ 1 ∧ J p = J (9 / 16) := by
  sorry

end minimize_J

namespace find_k_inv_h_of_10

import Mathlib

-- Assuming h and k are functions with appropriate properties
variables (h k : ℝ → ℝ)
variables (h_inv : ℝ → ℝ) (k_inv : ℝ → ℝ)

-- Given condition: h_inv (k(x)) = 4 * x - 5
axiom h_inv_k_eq : ∀ x, h_inv (k x) = 4 * x - 5

-- Statement to prove
theorem find_k_inv_h_of_10 :
  k_inv (h 10) = 15 / 4 := 
sorry

end find_k_inv_h_of_10

namespace milk_for_flour

import Mathlib

theorem milk_for_flour (milk flour use_flour : ℕ) (h1 : milk = 75) (h2 : flour = 300) (h3 : use_flour = 900) : (use_flour/flour * milk) = 225 :=
by sorry

end milk_for_flour

namespace abs_inequality_solution

import Mathlib

theorem abs_inequality_solution (x : ℝ) : |x - 1| + |x - 3| < 8 ↔ -2 < x ∧ x < 6 :=
by sorry

end abs_inequality_solution

namespace extreme_values_number_of_zeros

import Mathlib

noncomputable def f (x : ℝ) : ℝ := 2 * x^3 - 3 * x^2 - 12 * x + 5
noncomputable def g (x m : ℝ) : ℝ := f x - m

theorem extreme_values :
  (∀ x : ℝ, f x ≤ 12) ∧ (f (-1) = 12) ∧ (∀ x : ℝ, -15 ≤ f x) ∧ (f 2 = -15) := 
sorry

theorem number_of_zeros (m : ℝ) :
  (m > 12 ∨ m < -15 → ∃! x : ℝ, g x m = 0) ∧
  (m = 12 ∨ m = -15 → ∃ x y : ℝ, x ≠ y ∧ g x m = 0 ∧ g y m = 0) ∧
  (-15 < m ∧ m < 12 → ∃ x y z : ℝ, x ≠ y ∧ x ≠ z ∧ y ≠ z ∧ g x m = 0 ∧ g y m = 0 ∧ g z m = 0) :=
sorry

end extreme_values_number_of_zeros

namespace opposite_of_a_is_2

import Mathlib

theorem opposite_of_a_is_2 (a : ℤ) (h : -a = 2) : a = -2 := 
by
  -- proof to be provided
  sorry

end opposite_of_a_is_2

namespace system1_solution_system2_solution

import Mathlib

-- For System (1)
theorem system1_solution (x y : ℝ) (h1 : y = 2 * x) (h2 : 3 * y + 2 * x = 8) : x = 1 ∧ y = 2 :=
by
  sorry

-- For System (2)
theorem system2_solution (s t : ℝ) (h1 : 2 * s - 3 * t = 2) (h2 : (s + 2 * t) / 3 = 3 / 2) : s = 5 / 2 ∧ t = 1 :=
by
  sorry

end system1_solution_system2_solution

namespace problem1_problem2

import Mathlib

-- Statement for Question (1)
theorem problem1 (x : ℝ) (h : |x - 1| + x ≥ x + 2) : x ≤ -1 ∨ x ≥ 3 :=
  sorry

-- Statement for Question (2)
theorem problem2 (a : ℝ) (h : ∀ x : ℝ, |x - a| + x ≤ 3 * x → x ≥ 2) : a = 6 :=
  sorry

end problem1_problem2

namespace rectangle_ratio

import Mathlib

theorem rectangle_ratio (t a b : ℝ) (h₀ : b = 2 * a) (h₁ : (t + 2 * a) ^ 2 = 3 * t ^ 2) : b / a = 2 :=
by
  sorry

end rectangle_ratio

namespace page_copy_cost

import Mathlib

theorem page_copy_cost (cost_per_4_pages : ℕ) (page_count : ℕ) (dollar_to_cents : ℕ) : cost_per_4_pages = 8 → page_count = 4 → dollar_to_cents = 100 → (1500 * (page_count / cost_per_4_pages) = 750) :=
by
  intros
  sorry

end page_copy_cost

namespace michael_water_left

import Mathlib

theorem michael_water_left :
  let initial_water := 5
  let given_water := (18 / 7 : ℚ) -- using rational number to represent the fractions
  let remaining_water := initial_water - given_water
  remaining_water = 17 / 7 :=
by
  sorry

end michael_water_left

namespace sum_of_first_five_terms_is_31

import Mathlib

variable (a : ℕ → ℝ) (q : ℝ)

-- The geometric sequence a_n
def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * q

-- Condition 1: a_2 * a_3 = 2 * a_1
def condition1 (a : ℕ → ℝ) (q : ℝ) : Prop :=
  a 2 * a 3 = 2 * a 1

-- Condition 2: The arithmetic mean of a_4 and 2 * a_7 is 5/4
def condition2 (a : ℕ → ℝ) (q : ℝ) : Prop :=
  (a 4 + 2 * a 7) / 2 = 5 / 4

-- Sum of the first 5 terms of the geometric sequence
def S_5 (a : ℕ → ℝ) : ℝ :=
  a 0 + a 1 + a 2 + a 3 + a 4

-- The theorem to prove
theorem sum_of_first_five_terms_is_31 (a : ℕ → ℝ) (q : ℝ) 
  (hg : is_geometric_sequence a q) 
  (hc1 : condition1 a q) 
  (hc2 : condition2 a q) : 
  S_5 a = 31 := by
  sorry

end sum_of_first_five_terms_is_31

namespace problem

import Mathlib

theorem problem (x y : ℕ) (hxpos : 0 < x ∧ x < 20) (hypos : 0 < y ∧ y < 20) (h : x + y + x * y = 119) : 
  x + y = 24 ∨ x + y = 21 ∨ x + y = 20 :=
by sorry

end problem

namespace trains_cross_time

import Mathlib

def speed_in_m_per_s (speed_in_km_per_hr : Float) : Float :=
  (speed_in_km_per_hr * 1000) / 3600

def relative_speed (speed1 : Float) (speed2 : Float) : Float :=
  speed1 + speed2

def total_distance (length1 : Float) (length2 : Float) : Float :=
  length1 + length2

def time_to_cross (total_dist : Float) (relative_spd : Float) : Float :=
  total_dist / relative_spd

theorem trains_cross_time 
  (length_train1 : Float := 270)
  (speed_train1 : Float := 120)
  (length_train2 : Float := 230.04)
  (speed_train2 : Float := 80) :
  time_to_cross (total_distance length_train1 length_train2) 
                (relative_speed (speed_in_m_per_s speed_train1) 
                                (speed_in_m_per_s speed_train2)) = 9 := 
by
  sorry

end trains_cross_time

namespace equal_perimeter_triangle_side_length

import Mathlib

theorem equal_perimeter_triangle_side_length (s: ℝ) : 
    ∀ (pentagon_perimeter triangle_perimeter: ℝ), 
    (pentagon_perimeter = 5 * 5) → 
    (triangle_perimeter = 3 * s) → 
    (pentagon_perimeter = triangle_perimeter) → 
    s = 25 / 3 :=
by
  intro pentagon_perimeter triangle_perimeter h1 h2 h3
  sorry

end equal_perimeter_triangle_side_length

namespace quadratic_range

import Mathlib

-- Definition of the quadratic function
def quadratic_function (x : ℝ) : ℝ := x^2 - 4 * x + 7

-- Defining the range of the quadratic function for the interval -1 < x < 4
theorem quadratic_range (y : ℝ) : 3 ≤ y ∧ y < 12 ↔ ∃ x : ℝ, -1 < x ∧ x < 4 ∧ y = quadratic_function x :=
by
  sorry

end quadratic_range

namespace initial_quantity_of_milk

import Mathlib

theorem initial_quantity_of_milk (A B C : ℝ) 
    (h1 : B = 0.375 * A)
    (h2 : C = 0.625 * A)
    (h3 : B + 148 = C - 148) : A = 1184 :=
by
  sorry

end initial_quantity_of_milk

namespace line_circle_no_intersection

import Mathlib

theorem line_circle_no_intersection :
  ∀ (x y : ℝ), (5 * x + 8 * y = 10) → ¬ (x^2 + y^2 = 1) :=
by
  intro x y hline hcirc
  -- Proof omitted
  sorry

end line_circle_no_intersection

namespace arrangement_count_correct

import Mathlib

def num_arrangements_exactly_two_females_next_to_each_other (males : ℕ) (females : ℕ) : ℕ :=
  if males = 4 ∧ females = 3 then 3600 else 0

theorem arrangement_count_correct :
  num_arrangements_exactly_two_females_next_to_each_other 4 3 = 3600 :=
by
  sorry

end arrangement_count_correct

namespace ellipse_equation

import Mathlib

-- Definitions based on the problem conditions
def hyperbola_foci (x y : ℝ) : Prop := 2 * x^2 - 2 * y^2 = 1
def passes_through_point (p : ℝ × ℝ) (x y : ℝ) : Prop := p = (1, -3 / 2)

-- The statement to be proved
theorem ellipse_equation (c : ℝ) (a b : ℝ) :
    hyperbola_foci (-1) 0 ∧ hyperbola_foci 1 0 ∧
    passes_through_point (1, -3 / 2) 1 (-3 / 2) ∧
    (a = 2) ∧ (b = Real.sqrt 3) ∧ (c = 1)
    → ∀ x y : ℝ, x^2 / 4 + y^2 / 3 = 1 :=
by
  sorry

end ellipse_equation

namespace zorbs_of_60_deg

import Mathlib

-- Define the measurement on Zorblat
def zorbs_in_full_circle := 600
-- Define the Earth angle in degrees
def earth_degrees_full_circle := 360
def angle_in_degrees := 60
-- Calculate the equivalent angle in zorbs
def zorbs_in_angle := zorbs_in_full_circle * angle_in_degrees / earth_degrees_full_circle

theorem zorbs_of_60_deg (h1 : zorbs_in_full_circle = 600)
                        (h2 : earth_degrees_full_circle = 360)
                        (h3 : angle_in_degrees = 60) :
  zorbs_in_angle = 100 :=
by sorry

end zorbs_of_60_deg

namespace not_in_range

import Mathlib

noncomputable def g (x c: ℝ) : ℝ := x^2 + c * x + 5

theorem not_in_range (c : ℝ) (hc : -2 * Real.sqrt 2 < c ∧ c < 2 * Real.sqrt 2) :
  ∀ x : ℝ, g x c ≠ 3 :=
by
  intros
  sorry

end not_in_range

namespace smallest_m

import Mathlib

open Real

noncomputable def y_seq (m : ℕ) : Type := Fin m → ℝ

theorem smallest_m (m : ℕ) (y : y_seq m) (h1 : ∀ i : Fin m, |y i| ≤ 1/2)
  (h2 : ∑ i, |y i| = 10 + |∑ i, y i|) : m = 20 :=
sorry

end smallest_m

namespace arithmetic_sequence_8th_term

import Mathlib

theorem arithmetic_sequence_8th_term (a d : ℤ) :
  (a + d = 25) ∧ (a + 5 * d = 49) → (a + 7 * d = 61) :=
by
  sorry

end arithmetic_sequence_8th_term

namespace base9_addition_correct

import Mathlib

-- Definition of base 9 addition problem.
def add_base9 (a b c : ℕ) : ℕ :=
  let sum := a + b + c -- Sum in base 10
  let d0 := sum % 9 -- Least significant digit in base 9
  let carry1 := sum / 9
  (carry1 + carry1 / 9 * 9 + carry1 % 9) + d0 -- Sum in base 9 considering carry

-- The specific values converted to base 9 integers
def n1 := 3 * 9^2 + 4 * 9 + 6
def n2 := 8 * 9^2 + 0 * 9 + 2
def n3 := 1 * 9^2 + 5 * 9 + 7

-- The expected result converted to base 9 integer
def expected_sum := 1 * 9^3 + 4 * 9^2 + 1 * 9 + 6

theorem base9_addition_correct : add_base9 n1 n2 n3 = expected_sum := by
  -- Proof will be provided here
  sorry

end base9_addition_correct

namespace original_loaf_had_27_slices

import Mathlib

def original_slices : ℕ :=
  let slices_andy_ate := 3 * 2
  let slices_for_toast := 2 * 10
  let slices_left := 1
  slices_andy_ate + slices_for_toast + slices_left

theorem original_loaf_had_27_slices (n : ℕ) (slices_andy_ate : ℕ) (slices_for_toast : ℕ) (slices_left : ℕ) :
  slices_andy_ate = 6 → slices_for_toast = 20 → slices_left = 1 → n = slices_andy_ate + slices_for_toast + slices_left → n = 27 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  exact h4

-- Verifying the statement
example : original_slices = 27 := by
  have h1 : 3 * 2 = 6 := rfl
  have h2 : 2 * 10 = 20 := rfl
  have h3 : 1 = 1 := rfl
  exact original_loaf_had_27_slices original_slices 6 20 1 h1 h2 h3 rfl

end original_loaf_had_27_slices

namespace squared_distance_focus_product_tangents

import Mathlib

variable {a b : ℝ}
variable {x0 y0 : ℝ}
variable {P Q R F : ℝ × ℝ}

-- Conditions
def is_ellipse (x y : ℝ) (a b : ℝ) : Prop :=
  (x^2 / a^2) + (y^2 / b^2) = 1

def outside_ellipse (x0 y0 : ℝ) (a b : ℝ) : Prop :=
  (x0^2 / a^2) + (y0^2 / b^2) > 1

-- Question (statement we need to prove)
theorem squared_distance_focus_product_tangents
  (h_ellipse : is_ellipse Q.1 Q.2 a b)
  (h_ellipse' : is_ellipse R.1 R.2 a b)
  (h_outside : outside_ellipse x0 y0 a b)
  (h_a_greater_b : a > b) :
  ‖P - F‖^2 > ‖Q - F‖ * ‖R - F‖ := sorry

end squared_distance_focus_product_tangents

namespace initial_percentage_increase

import Mathlib

variable (S : ℝ) (P : ℝ)

theorem initial_percentage_increase :
  (S + (P / 100) * S) - 0.10 * (S + (P / 100) * S) = S + 0.15 * S →
  P = 16.67 :=
by
  sorry

end initial_percentage_increase

namespace prove_inequality1_prove_inequality2_prove_inequality3_prove_inequality5

import Mathlib

-- Definition of the inequalities to be proven using the rearrangement inequality
def inequality1 (a b : ℝ) : Prop := a^2 + b^2 ≥ 2 * a * b
def inequality2 (a b c : ℝ) : Prop := a^2 + b^2 + c^2 ≥ a * b + b * c + c * a
def inequality3 (a b : ℝ) : Prop := a^2 + b^2 + 1 ≥ a * b + b + a
def inequality5 (x y : ℝ) : Prop := x^3 + y^3 ≥ x^2 * y + x * y^2

-- Proofs required for each inequality
theorem prove_inequality1 (a b : ℝ) : inequality1 a b := 
by sorry  -- This can be proved using the rearrangement inequality

theorem prove_inequality2 (a b c : ℝ) : inequality2 a b c := 
by sorry  -- This can be proved using the rearrangement inequality

theorem prove_inequality3 (a b : ℝ) : inequality3 a b := 
by sorry  -- This can be proved using the rearrangement inequality

theorem prove_inequality5 (x y : ℝ) (hx : x ≥ y) (hy : 0 < y) : inequality5 x y := 
by sorry  -- This can be proved using the rearrangement inequality

end prove_inequality1_prove_inequality2_prove_inequality3_prove_inequality5

namespace coefficient_of_x2_in_expansion_of_x_minus_2_to_the_5

import Mathlib

theorem coefficient_of_x2_in_expansion_of_x_minus_2_to_the_5 :
  ∀ (a_0 a_1 a_2 a_3 a_4 a_5 : ℝ),
  (x - 2) ^ 5 = a_0 + a_1*x + a_2*x^2 + a_3*x^3 + a_4*x^4 + a_5*x^5 → a_2 = -80 := by
  sorry

end coefficient_of_x2_in_expansion_of_x_minus_2_to_the_5

namespace find_power

import Mathlib

theorem find_power (x y : ℕ) (h1 : 2^x - 2^y = 3 * 2^11) (h2 : x = 13) : y = 11 :=
sorry

end find_power

namespace student_in_eighth_group

import Mathlib

-- Defining the problem: total students and their assignment into groups
def total_students : ℕ := 50
def students_assigned_numbers (n : ℕ) : Prop := n > 0 ∧ n ≤ total_students

-- Grouping students: Each group has 5 students
def grouped_students (group_num student_num : ℕ) : Prop := 
  student_num > (group_num - 1) * 5 ∧ student_num ≤ group_num * 5

-- Condition: Student 12 is selected from the third group
def condition : Prop := grouped_students 3 12

-- Goal: the number of the student selected from the eighth group is 37
theorem student_in_eighth_group : condition → grouped_students 8 37 :=
by
  sorry

end student_in_eighth_group

namespace shaded_area_correct

import Mathlib

noncomputable def total_shaded_area (floor_length : ℝ) (floor_width : ℝ) (tile_size : ℝ) (circle_radius : ℝ) : ℝ :=
  let tile_area := tile_size ^ 2
  let circle_area := Real.pi * circle_radius ^ 2
  let shaded_area_per_tile := tile_area - circle_area
  let floor_area := floor_length * floor_width
  let number_of_tiles := floor_area / tile_area
  number_of_tiles * shaded_area_per_tile 

theorem shaded_area_correct : total_shaded_area 12 15 2 1 = 180 - 45 * Real.pi := sorry

end shaded_area_correct

namespace min_value_of_function

import Mathlib

theorem min_value_of_function (p : ℝ) : 
  ∃ x : ℝ, (x^2 - 2 * p * x + 2 * p^2 + 2 * p - 1) = -2 := sorry

end min_value_of_function

namespace calc_factorial_sum

import Mathlib

theorem calc_factorial_sum : 6 * Nat.factorial 6 + 5 * Nat.factorial 5 + Nat.factorial 5 = 5040 := by
  sorry

end calc_factorial_sum

namespace slope_of_line

import Mathlib

theorem slope_of_line (x1 y1 x2 y2 : ℝ)
  (h1 : 4 * y1 + 6 * x1 = 0)
  (h2 : 4 * y2 + 6 * x2 = 0)
  (h1x2 : x1 ≠ x2) :
  (y2 - y1) / (x2 - x1) = -3 / 2 :=
by sorry

end slope_of_line

namespace find_n_for_arithmetic_sequence

import Mathlib

variable {a : ℕ → ℤ}

-- Conditions
def is_arithmetic_sequence (a : ℕ → ℤ) (a₁ : ℤ) (d : ℤ) : Prop :=
  ∀ n : ℕ, a n = a₁ + n * d

theorem find_n_for_arithmetic_sequence (h_arith : is_arithmetic_sequence a (-1) 2)
  (h_nth_term : ∃ n : ℕ, a n = 15) : ∃ n : ℕ, n = 9 :=
by
  sorry

end find_n_for_arithmetic_sequence

namespace cheezit_bag_weight

import Mathlib

-- Definitions based on the conditions of the problem
def cheezit_bags : ℕ := 3
def calories_per_ounce : ℕ := 150
def run_minutes : ℕ := 40
def calories_per_minute : ℕ := 12
def excess_calories : ℕ := 420

-- Main theorem stating the question with the solution
theorem cheezit_bag_weight (x : ℕ) : 
  (calories_per_ounce * cheezit_bags * x) - (run_minutes * calories_per_minute) = excess_calories → 
  x = 2 :=
by
  sorry

end cheezit_bag_weight

namespace final_silver_tokens

import Mathlib

structure TokenCounts :=
  (red : ℕ)
  (blue : ℕ)

def initial_tokens : TokenCounts := { red := 100, blue := 50 }

def exchange_booth1 (tokens : TokenCounts) : TokenCounts :=
  { red := tokens.red - 3, blue := tokens.blue + 2 }

def exchange_booth2 (tokens : TokenCounts) : TokenCounts :=
  { red := tokens.red + 1, blue := tokens.blue - 3 }

noncomputable def max_exchanges (initial : TokenCounts) : ℕ × ℕ :=
  let x := 48
  let y := 47
  (x, y)

noncomputable def silver_tokens (x y : ℕ) : ℕ := x + y

theorem final_silver_tokens (x y : ℕ) (tokens : TokenCounts) 
  (hx : tokens.red = initial_tokens.red - 3 * x + y)
  (hy : tokens.blue = initial_tokens.blue + 2 * x - 3 * y) 
  (hx_le : tokens.red >= 3 → false)
  (hy_le : tokens.blue >= 3 → false) : 
  silver_tokens x y = 95 :=
by {
  sorry
}

end final_silver_tokens

namespace beautiful_39th_moment

import Mathlib

def is_beautiful (h : ℕ) (mm : ℕ) : Prop :=
  (h + mm) % 12 = 0

def start_time := (7, 49)

noncomputable def find_39th_beautiful_moment : ℕ × ℕ :=
  (15, 45)

theorem beautiful_39th_moment :
  find_39th_beautiful_moment = (15, 45) :=
by
  sorry

end beautiful_39th_moment

namespace seq_is_geometric_from_second

import Mathlib

namespace sequence_problem

-- Definitions and conditions
def S : ℕ → ℕ
| 0 => 0
| 1 => 1
| 2 => 2
| (n + 1) => 3 * S n - 2 * S (n - 1)

-- Recursive definition for sum of sequence terms
axiom S_rec_relation (n : ℕ) (h : n ≥ 2) : 
  S (n + 1) - 3 * S n + 2 * S (n - 1) = 0

-- Prove the sequence is geometric from the second term
theorem seq_is_geometric_from_second :
  ∃ (a : ℕ → ℕ), (∀ n ≥ 2, a (n + 1) = 2 * a n) ∧ 
  (a 1 = 1) ∧ 
  (a 2 = 1) :=
by
  sorry

end sequence_problem

end seq_is_geometric_from_second

namespace bill_toys

import Mathlib

variable (B H : ℕ)

theorem bill_toys (h1 : H = B / 2 + 9) (h2 : B + H = 99) : B = 60 := by
  sorry

end bill_toys

namespace arrange_letters

import Mathlib

-- Definitions based on conditions
def total_letters := 6
def identical_bs := 2 -- Number of B's that are identical
def distinct_as := 3  -- Number of A's that are distinct
def distinct_ns := 1  -- Number of N's that are distinct

-- Now formulate the proof statement
theorem arrange_letters :
    (Nat.factorial total_letters) / (Nat.factorial identical_bs) = 360 :=
by
  sorry

end arrange_letters

namespace line_intersects_ellipse_all_possible_slopes

import Mathlib

theorem line_intersects_ellipse_all_possible_slopes (m : ℝ) :
  m^2 ≥ 1 / 5 ↔ ∃ x y : ℝ, (y = m * x - 3) ∧ (4 * x^2 + 25 * y^2 = 100) := sorry

end line_intersects_ellipse_all_possible_slopes

namespace percentage_of_all_students_with_cars

import Mathlib

def seniors := 300
def percent_seniors_with_cars := 0.40
def lower_grades := 1500
def percent_lower_grades_with_cars := 0.10

theorem percentage_of_all_students_with_cars :
  (120 + 150) / 1800 * 100 = 15 := by
  sorry

end percentage_of_all_students_with_cars

namespace angle_measure

import Mathlib

variable (x : ℝ)

noncomputable def is_supplement (x : ℝ) : Prop := 180 - x = 3 * (90 - x) - 60

theorem angle_measure : is_supplement x → x = 15 :=
by
  sorry

end angle_measure

namespace f_2_equals_12

import Mathlib

noncomputable def f (x : ℝ) : ℝ :=
if x < 0 then 2 * x^3 + x^2 else - (2 * (-x)^3 + (-x)^2)

theorem f_2_equals_12 : f 2 = 12 := by
  sorry

end f_2_equals_12

namespace total_students

import Mathlib

variable (T : ℕ)

-- Conditions
def is_girls_percentage (T : ℕ) := 60 / 100 * T
def is_boys_percentage (T : ℕ) := 40 / 100 * T
def boys_not_in_clubs (number_of_boys : ℕ) := 2 / 3 * number_of_boys

theorem total_students (h1 : is_girls_percentage T + is_boys_percentage T = T)
  (h2 : boys_not_in_clubs (is_boys_percentage T) = 40) : T = 150 :=
by
  sorry

end total_students

namespace range_of_n

import Mathlib

def hyperbola_equation (m n : ℝ) : Prop :=
  (m^2 + n) * (3 * m^2 - n) > 0

def foci_distance (m n : ℝ) : Prop :=
  (m^2 + n) + (3 * m^2 - n) = 4

theorem range_of_n (m n : ℝ) :
  hyperbola_equation m n ∧ foci_distance m n →
  -1 < n ∧ n < 3 :=
by
  intro h
  have hyperbola_condition := h.1
  have distance_condition := h.2
  sorry

end range_of_n

namespace sin_600_eq_neg_sqrt_3_div_2

import Mathlib

theorem sin_600_eq_neg_sqrt_3_div_2 : Real.sin (600 * (Real.pi / 180)) = - (Real.sqrt 3 / 2) :=
by
  sorry

end sin_600_eq_neg_sqrt_3_div_2

namespace volunteer_org_percentage_change

import Mathlib

theorem volunteer_org_percentage_change 
  (initial_membership : ℝ)
  (fall_increase_rate : ℝ)
  (spring_decrease_rate : ℝ) :
  (initial_membership = 100) →
  (fall_increase_rate = 0.05) →
  (spring_decrease_rate = 0.19) →
  (14.95 : ℝ) =
  ((initial_membership * (1 + fall_increase_rate)) * (1 - spring_decrease_rate)
  - initial_membership) / initial_membership * 100 := by
  sorry

end volunteer_org_percentage_change

namespace count_integers_P_leq_0

import Mathlib

def P(x : ℤ) : ℤ := 
  (x - 1^3) * (x - 2^3) * (x - 3^3) * (x - 4^3) * (x - 5^3) *
  (x - 6^3) * (x - 7^3) * (x - 8^3) * (x - 9^3) * (x - 10^3) *
  (x - 11^3) * (x - 12^3) * (x - 13^3) * (x - 14^3) * (x - 15^3) *
  (x - 16^3) * (x - 17^3) * (x - 18^3) * (x - 19^3) * (x - 20^3) *
  (x - 21^3) * (x - 22^3) * (x - 23^3) * (x - 24^3) * (x - 25^3) *
  (x - 26^3) * (x - 27^3) * (x - 28^3) * (x - 29^3) * (x - 30^3) *
  (x - 31^3) * (x - 32^3) * (x - 33^3) * (x - 34^3) * (x - 35^3) *
  (x - 36^3) * (x - 37^3) * (x - 38^3) * (x - 39^3) * (x - 40^3) *
  (x - 41^3) * (x - 42^3) * (x - 43^3) * (x - 44^3) * (x - 45^3) *
  (x - 46^3) * (x - 47^3) * (x - 48^3) * (x - 49^3) * (x - 50^3)

theorem count_integers_P_leq_0 : 
  ∃ n : ℕ, n = 15650 ∧ ∀ k : ℤ, (P k ≤ 0) → (n = 15650) :=
by sorry

end count_integers_P_leq_0

namespace expected_coin_worth_is_two

import Mathlib

-- Define the conditions
def p_heads : ℚ := 4 / 5
def p_tails : ℚ := 1 / 5
def gain_heads : ℚ := 5
def loss_tails : ℚ := -10

-- Expected worth calculation
def expected_worth : ℚ := (p_heads * gain_heads) + (p_tails * loss_tails)

-- Lean 4 statement to prove
theorem expected_coin_worth_is_two : expected_worth = 2 := by
  sorry

end expected_coin_worth_is_two

namespace arithmetic_sequence_sum_ratio

import Mathlib

theorem arithmetic_sequence_sum_ratio
  (S : ℕ → ℚ)
  (a : ℕ → ℚ)
  (h1 : ∀ n, S n = n * (a 1 + a n) / 2)
  (h2 : a 5 / a 3 = 7 / 3) :
  S 5 / S 3 = 5 := 
by
  sorry

end arithmetic_sequence_sum_ratio

namespace pencils_given_out

import Mathlib

theorem pencils_given_out
  (num_children : ℕ)
  (pencils_per_student : ℕ)
  (dozen : ℕ)
  (children : num_children = 46)
  (dozen_def : dozen = 12)
  (pencils_def : pencils_per_student = 4 * dozen) :
  num_children * pencils_per_student = 2208 :=
by {
  sorry
}

end pencils_given_out

namespace land_plot_side_length

import Mathlib

theorem land_plot_side_length (A : ℝ) (h : A = Real.sqrt 1024) : Real.sqrt A = 32 := 
by sorry

end land_plot_side_length

namespace four_digit_number_properties

import Mathlib

theorem four_digit_number_properties :
  ∃ (a b c d : ℕ), 
    a + b + c + d = 8 ∧ 
    a = 3 * b ∧ 
    d = 4 * c ∧ 
    1000 * a + 100 * b + 10 * c + d = 6200 :=
by
  sorry

end four_digit_number_properties

namespace product_of_dice_divisible_by_9

import Mathlib

-- Define the probability of rolling a number divisible by 3
def prob_roll_div_by_3 : ℚ := 1/6

-- Define the probability of rolling a number not divisible by 3
def prob_roll_not_div_by_3 : ℚ := 2/3

-- Define the probability that the product of numbers rolled on 6 dice is divisible by 9
def prob_product_div_by_9 : ℚ := 449/729

-- Main statement of the problem
theorem product_of_dice_divisible_by_9 :
  (1 - ((prob_roll_not_div_by_3^6) + 
        (6 * prob_roll_div_by_3 * (prob_roll_not_div_by_3^5)) + 
        (15 * (prob_roll_div_by_3^2) * (prob_roll_not_div_by_3^4)))) = prob_product_div_by_9 :=
by {
  sorry
}

end product_of_dice_divisible_by_9

namespace conditional_probability

import Mathlib

-- Definitions of the events and probabilities given in the conditions
def event_A (red : ℕ) : Prop := red % 3 = 0
def event_B (red blue : ℕ) : Prop := red + blue > 8

-- The actual values of probabilities calculated in the solution
def P_A : ℚ := 1/3
def P_B : ℚ := 1/3
def P_AB : ℚ := 5/36

-- Definition of conditional probability
def P_B_given_A : ℚ := P_AB / P_A

-- The claim we want to prove
theorem conditional_probability :
  P_B_given_A = 5 / 12 :=
sorry

end conditional_probability

namespace geometric_sequence_sum

import Mathlib

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q, ∀ n, a (n + 1) = a n * q

variables {a : ℕ → ℝ}

theorem geometric_sequence_sum (h1 : is_geometric_sequence a) (h2 : a 1 * a 2 = 8 * a 0)
  (h3 : (a 3 + 2 * a 4) / 2 = 20) :
  (a 0 * (2^5 - 1)) = 31 :=
by
  sorry

end geometric_sequence_sum

namespace wam_gm_gt_hm

import Mathlib

noncomputable def wam (w v a b : ℝ) : ℝ := w * a + v * b
noncomputable def gm (a b : ℝ) : ℝ := Real.sqrt (a * b)
noncomputable def hm (a b : ℝ) : ℝ := (2 * a * b) / (a + b)

theorem wam_gm_gt_hm
  (a b w v : ℝ)
  (h1 : 0 < a ∧ 0 < b)
  (h2 : 0 < w ∧ 0 < v)
  (h3 : w + v = 1)
  (h4 : a ≠ b) :
  wam w v a b > gm a b ∧ gm a b > hm a b :=
by
  -- Proof omitted
  sorry

end wam_gm_gt_hm

namespace saved_money_is_30

import Mathlib

def week_payout : ℕ := 5 * 3
def total_payout (weeks: ℕ) : ℕ := weeks * week_payout
def shoes_cost : ℕ := 120
def remaining_weeks : ℕ := 6
def remaining_earnings : ℕ := total_payout remaining_weeks
def saved_money : ℕ := shoes_cost - remaining_earnings

theorem saved_money_is_30 : saved_money = 30 := by
  -- Proof steps go here
  sorry

end saved_money_is_30

namespace number_divisibility

import Mathlib

def A_n (n : ℕ) : ℕ := (10^(3^n) - 1) / 9

theorem number_divisibility (n : ℕ) :
  (3^n ∣ A_n n) ∧ ¬ (3^(n + 1) ∣ A_n n) := by
  sorry

end number_divisibility

namespace solve_equation

import Mathlib

theorem solve_equation (x : ℝ) (h1 : x ≠ 2) (h2 : x ≠ -2) :
  (3 - x^2) / (x + 2) + (2 * x^2 - 8) / (x^2 - 4) = 3 ↔ 
  x = (-1 + Real.sqrt 5) / 2 ∨ x = (-1 - Real.sqrt 5) / 2 := 
by
  sorry

end solve_equation

namespace km_to_leaps

import Mathlib

theorem km_to_leaps (a b c d e f : ℕ) :
  (2 * a) * strides = (3 * b) * leaps →
  (4 * c) * dashes = (5 * d) * strides →
  (6 * e) * dashes = (7 * f) * kilometers →
  1 * kilometers = (90 * b * d * e) / (56 * a * c * f) * leaps :=
by
  -- Using the given conditions to derive the answer
  intro h1 h2 h3
  sorry

end km_to_leaps

namespace M_subset_N

import Mathlib

variable (f g : ℝ → ℝ) (a : ℝ)

def M : Set ℝ := {x | abs (f x) + abs (g x) < a}
def N : Set ℝ := {x | abs (f x + g x) < a}

theorem M_subset_N (h : a > 0) : M f g a ⊆ N f g a := by
  sorry

end M_subset_N

namespace negation_of_proposition

import Mathlib

variables (a b : ℕ)

def is_even (n : ℕ) : Prop := ∃ k, n = 2 * k

def both_even (a b : ℕ) : Prop := is_even a ∧ is_even b

def sum_even (a b : ℕ) : Prop := is_even (a + b)

theorem negation_of_proposition : ¬ (both_even a b → sum_even a b) ↔ ¬both_even a b ∨ ¬sum_even a b :=
by sorry

end negation_of_proposition

namespace bounded_roots

import Mathlib

open Polynomial

noncomputable def P : ℤ[X] := sorry -- Replace with actual polynomial if necessary

theorem bounded_roots (P : ℤ[X]) (n : ℕ) (hPdeg : P.degree = n) (hdec : 1 ≤ n) :
  ∀ k : ℤ, (P.eval k) ^ 2 = 1 → ∃ (r s : ℕ), r + s ≤ n + 2 := 
by 
  sorry

end bounded_roots

namespace number_of_crayons_given_to_friends

import Mathlib

def totalCrayonsLostOrGivenAway := 229
def crayonsLost := 16
def crayonsGivenToFriends := totalCrayonsLostOrGivenAway - crayonsLost

theorem number_of_crayons_given_to_friends :
  crayonsGivenToFriends = 213 :=
by
  sorry

end number_of_crayons_given_to_friends

namespace calculation_correct

import Mathlib

def expression : ℝ := 200 * 375 * 0.0375 * 5

theorem calculation_correct : expression = 14062.5 := 
by
  sorry

end calculation_correct

namespace abs_a_gt_neg_b

import Mathlib

variable {a b : ℝ}

theorem abs_a_gt_neg_b (h : a < b ∧ b < 0) : |a| > -b :=
by
  sorry

end abs_a_gt_neg_b

namespace calculate_subtraction

import Mathlib

theorem calculate_subtraction :
  ∀ (x : ℕ), (49 = 50 - 1) → (49^2 = 50^2 - 99)
  := by
  intros x h
  sorry

end calculate_subtraction

namespace find_number

import Mathlib

variable (x : ℝ)

theorem find_number (h : 2 * x - 6 = (1/4) * x + 8) : x = 8 :=
sorry

end find_number

namespace elephant_weight_equivalence

import Mathlib

-- Define the conditions as variables
def elephants := 1000000000
def buildings := 25000

-- Define the question and expected answer
def expected_answer := 40000

-- State the theorem
theorem elephant_weight_equivalence:
  (elephants / buildings = expected_answer) :=
by
  sorry

end elephant_weight_equivalence

namespace sum_floor_parity

import Mathlib

theorem sum_floor_parity (m n : ℕ) (hm : 0 < m) (hn : 0 < n) : 
  (∑ i in Finset.range (m * n), (-1) ^ (i / m + i / n) = 0) ↔ ((m % 2 = 0 ∧ n % 2 = 1) ∨ (m % 2 = 1 ∧ n % 2 = 0)) :=
by sorry

end sum_floor_parity

namespace selling_price_is_1260

import Mathlib

-- Definitions based on conditions
def purchase_price : ℕ := 900
def repair_cost : ℕ := 300
def gain_percent : ℕ := 5 -- percentage as a natural number

-- Known variables
def total_cost : ℕ := purchase_price + repair_cost
def gain_amount : ℕ := (gain_percent * total_cost) / 100
def selling_price : ℕ := total_cost + gain_amount

-- The theorem we want to prove
theorem selling_price_is_1260 : selling_price = 1260 := by
  sorry

end selling_price_is_1260

namespace correct_exponent_calculation

import Mathlib

theorem correct_exponent_calculation (x : ℝ) : (-x^3)^4 = x^12 := 
by sorry

end correct_exponent_calculation

namespace remainder_7547_div_11

import Mathlib

theorem remainder_7547_div_11 : 7547 % 11 = 10 :=
by
  sorry

end remainder_7547_div_11

namespace sandy_more_tokens_than_siblings

import Mathlib

-- Define the initial conditions
def initial_tokens : ℕ := 3000000
def initial_transaction_fee_percent : ℚ := 0.10
def value_increase_percent : ℚ := 0.20
def additional_tokens : ℕ := 500000
def additional_transaction_fee_percent : ℚ := 0.07
def sandy_keep_percent : ℚ := 0.40
def siblings : ℕ := 7
def sibling_transaction_fee_percent : ℚ := 0.05

-- Define the main theorem to prove
theorem sandy_more_tokens_than_siblings :
  let received_initial_tokens := initial_tokens * (1 - initial_transaction_fee_percent)
  let increased_tokens := received_initial_tokens * (1 + value_increase_percent)
  let received_additional_tokens := additional_tokens * (1 - additional_transaction_fee_percent)
  let total_tokens := increased_tokens + received_additional_tokens
  let sandy_tokens := total_tokens * sandy_keep_percent
  let remaining_tokens := total_tokens * (1 - sandy_keep_percent)
  let each_sibling_tokens := remaining_tokens / siblings * (1 - sibling_transaction_fee_percent)
  sandy_tokens - each_sibling_tokens = 1180307.1428 := sorry

end sandy_more_tokens_than_siblings

namespace lcm_36_225

import Mathlib

theorem lcm_36_225 : Nat.lcm 36 225 = 900 := by
  -- Defining the factorizations as given
  let fact_36 : 36 = 2^2 * 3^2 := by rfl
  let fact_225 : 225 = 3^2 * 5^2 := by rfl

  -- Indicating what LCM we need to prove
  show Nat.lcm 36 225 = 900

  -- Proof (skipped)
  sorry

end lcm_36_225

namespace spinsters_count

import Mathlib

theorem spinsters_count (S C : ℕ) (h1 : S / C = 2 / 9) (h2 : C = S + 42) : S = 12 := by
  sorry

end spinsters_count

namespace minimum_value_expression

import Mathlib

open Real

theorem minimum_value_expression (α β : ℝ) :
  ∃ x y : ℝ, x = 3 * cos α + 4 * sin β ∧ y = 3 * sin α + 4 * cos β ∧
    ((x - 7) ^ 2 + (y - 12) ^ 2) = 242 - 14 * sqrt 193 :=
sorry

end minimum_value_expression

namespace 

import Mathlib

theorem roy_missed_days {hours_per_day days_per_week actual_hours_week missed_days : ℕ}
    (h1 : hours_per_day = 2)
    (h2 : days_per_week = 5)
    (h3 : actual_hours_week = 6)
    (expected_hours_week : ℕ := hours_per_day * days_per_week)
    (missed_hours : ℕ := expected_hours_week - actual_hours_week)
    (missed_days := missed_hours / hours_per_day) :
  missed_days = 2 := by
  sorry

end 

namespace rectangle_area_ratio

import Mathlib

-- Define points in complex plane or as tuples (for 2D geometry)
structure Point where
  x : ℝ
  y : ℝ

-- Rectangle vertices
def A : Point := {x := 0, y := 0}
def B : Point := {x := 1, y := 0}
def C : Point := {x := 1, y := 2}
def D : Point := {x := 0, y := 2}

-- Centroid of triangle BCD
def E : Point := {x := 1.0, y := 1.333}

-- Point F such that DF = 1/4 * DA
def F : Point := {x := 1.5, y := 0}

-- Calculate areas of triangles and quadrilateral
noncomputable def area_triangle (P Q R : Point) : ℝ :=
  0.5 * abs ((Q.x - P.x) * (R.y - P.y) - (Q.y - P.y) * (R.x - P.x))

noncomputable def area_rectangle : ℝ :=
  2.0  -- Area of rectangle ABCD (1 * 2)

noncomputable def problem_statement : Prop :=
  let area_DFE := area_triangle D F E
  let area_ABEF := area_rectangle - area_triangle A B F - area_triangle D A F
  area_DFE / area_ABEF = 1 / 10.5

theorem rectangle_area_ratio :
  problem_statement :=
by
  sorry

end rectangle_area_ratio

namespace minimum_a_plus_2b_no_a_b_such_that

import Mathlib

noncomputable def minimum_value (a b : ℝ) :=
  a + 2 * b

theorem minimum_a_plus_2b (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 2 * a * b = a + 2 * b + 3) : 
  minimum_value a b ≥ 6 :=
sorry

theorem no_a_b_such_that (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 2 * a * b = a + 2 * b + 3) : 
  a^2 + 4 * b^2 ≠ 17 :=
sorry

end minimum_a_plus_2b_no_a_b_such_that

namespace problem_statement

import Mathlib

theorem problem_statement (a b c : ℝ) (h : a * c^2 > b * c^2) (hc : c ≠ 0) : 
  a > b :=
by 
  sorry

end problem_statement

namespace angle_B_is_40_degrees

import Mathlib

theorem angle_B_is_40_degrees (angle_A angle_B angle_C : ℝ)
  (h1 : angle_A = 3 * angle_B)
  (h2 : angle_B = 2 * angle_C)
  (triangle_sum : angle_A + angle_B + angle_C = 180) :
  angle_B = 40 :=
by
  sorry

end angle_B_is_40_degrees

namespace find_x_value

import Mathlib

theorem find_x_value (PQ_is_straight_line : True) 
  (angles_on_line : List ℕ) (h : angles_on_line = [x, x, x, x, x])
  (sum_of_angles : angles_on_line.sum = 180) :
  x = 36 :=
by
  sorry

end find_x_value

namespace Tony_fills_pool_in_90_minutes

import Mathlib

def minutes (r : ℚ) : ℚ := 1 / r

theorem Tony_fills_pool_in_90_minutes (J S T : ℚ) 
  (hJ : J = 1 / 30)       -- Jim's rate in pools per minute
  (hS : S = 1 / 45)       -- Sue's rate in pools per minute
  (h_combined : J + S + T = 1 / 15) -- Combined rate of all three

  : minutes T = 90 :=     -- Tony can fill the pool alone in 90 minutes
by sorry

end Tony_fills_pool_in_90_minutes

namespace opposite_of_x_is_positive

import Mathlib

-- Assume a rational number x
def x : ℚ := -1 / 2023

-- Theorem stating the opposite of x is 1 / 2023
theorem opposite_of_x_is_positive : -x = 1 / 2023 :=
by
  -- Required part of Lean syntax; not containing any solution steps
  sorry

end opposite_of_x_is_positive

namespace transmitted_word_is_PAROHOD

import Mathlib

-- Define the binary representation of each letter in the Russian alphabet.
def binary_repr : String → String
| "А" => "00000"
| "Б" => "00001"
| "В" => "00011"
| "Г" => "00111"
| "Д" => "00101"
| "Е" => "00110"
| "Ж" => "01100"
| "З" => "01011"
| "И" => "01001"
| "Й" => "11000"
| "К" => "01010"
| "Л" => "01011"
| "М" => "01101"
| "Н" => "01111"
| "О" => "01100"
| "П" => "01110"
| "Р" => "01010"
| "С" => "01100"
| "Т" => "01001"
| "У" => "01111"
| "Ф" => "11101"
| "Х" => "11011"
| "Ц" => "11100"
| "Ч" => "10111"
| "Ш" => "11110"
| "Щ" => "11110"
| "Ь" => "00010"
| "Ы" => "00011"
| "Ъ" => "00101"
| "Э" => "11100"
| "Ю" => "01111"
| "Я" => "11111"
| _  => "00000" -- default case

-- Define the received scrambled word.
def received_word : List String := ["Э", "А", "В", "Щ", "О", "Щ", "И"]

-- The target transmitted word is "ПАРОХОД" which corresponds to ["П", "А", "Р", "О", "Х", "О", "Д"]
def transmitted_word : List String := ["П", "А", "Р", "О", "Х", "О", "Д"]

-- Lean 4 proof statement to show that the received scrambled word reconstructs to the transmitted word.
theorem transmitted_word_is_PAROHOD (b_repr : String → String)
(received : List String) :
  received = received_word →
  transmitted_word.map b_repr = received.map b_repr → transmitted_word = ["П", "А", "Р", "О", "Х", "О", "Д"] :=
by 
  intros h_received h_repr_eq
  exact sorry

end transmitted_word_is_PAROHOD

namespace cities_with_highest_increase

import Mathlib

-- Define population changes for each city
def cityF_initial := 30000
def cityF_final := 45000
def cityG_initial := 55000
def cityG_final := 77000
def cityH_initial := 40000
def cityH_final := 60000
def cityI_initial := 70000
def cityI_final := 98000
def cityJ_initial := 25000
def cityJ_final := 37500

-- Function to calculate percentage increase
def percentage_increase (initial final : ℕ) : ℚ :=
  ((final - initial) : ℚ) / (initial : ℚ) * 100

-- Theorem stating cities F, H, and J had the highest percentage increase
theorem cities_with_highest_increase :
  percentage_increase cityF_initial cityF_final = 50 ∧
  percentage_increase cityH_initial cityH_final = 50 ∧
  percentage_increase cityJ_initial cityJ_final = 50 ∧
  percentage_increase cityG_initial cityG_final < 50 ∧
  percentage_increase cityI_initial cityI_final < 50 :=
by
-- Proof omitted
sorry

end cities_with_highest_increase

namespace calculate_abs_mul

import Mathlib

theorem calculate_abs_mul : |(-3 : ℤ)| * 2 = 6 := 
by 
  -- |(-3)| equals 3 and 3 * 2 equals 6.
  -- The "sorry" is used to complete the statement without proof.
  sorry

end calculate_abs_mul

namespace total_number_of_cats

import Mathlib

def Cat := Type -- Define a type of Cat.

variable (A B C: Cat) -- Declaring three cats A, B, and C.

variable (kittens_A: Fin 4 → {gender : Bool // (2 : Fin 4).val = 2 ∧ (2 : Fin 4).val = 2}) -- 4 kittens: 2 males, 2 females.
variable (kittens_B: Fin 3 → {gender : Bool // (1 : Fin 3).val = 1 ∧ (2 : Fin 3).val = 2}) -- 3 kittens: 1 male, 2 females.
variable (kittens_C: Fin 5 → {gender : Bool // (3 : Fin 5).val = 3 ∧ (2 : Fin 5).val = 2}) -- 5 kittens: 3 males, 2 females.

variable (extra_kittens: Fin 2 → {gender : Bool // (1 : Fin 2).val = 1 ∧ (1 : Fin 2).val = 1}) -- 2 kittens of the additional female kitten of Cat A.

theorem total_number_of_cats : 
  3 + 4 + 2 + 3 + 5 = 17 :=
by
  sorry

end total_number_of_cats

namespace nap_duration_is_two_hours

import Mathlib

-- Conditions as definitions in Lean
def naps_per_week : ℕ := 3
def days : ℕ := 70
def total_nap_hours : ℕ := 60

-- Calculate the duration of each nap
theorem nap_duration_is_two_hours :
  ∃ (nap_duration : ℕ), nap_duration = 2 ∧
  (days / 7) * naps_per_week * nap_duration = total_nap_hours :=
by
  sorry

end nap_duration_is_two_hours

namespace normal_line_eq

import Mathlib

variable {x : ℝ}

noncomputable def f (x : ℝ) : ℝ := 2 * x^2 - 3 * x + 1

theorem normal_line_eq (x_0 : ℝ) (h : x_0 = 1) :
  ∃ y_0 : ℝ, y_0 = f x_0 ∧ 
  ∀ x y : ℝ, y = -(x - 1) + y_0 ↔ f 1 = 0 ∧ y = -x + 1 :=
by
  sorry

end normal_line_eq

namespace fraction_simplification

import Mathlib

variable (a b x : ℝ)
variable (h1 : x = a / b)
variable (h2 : a ≠ b)
variable (h3 : b ≠ 0)
variable (h4 : a = b * x ^ 2)

theorem fraction_simplification : (a + b) / (a - b) = (x ^ 2 + 1) / (x ^ 2 - 1) := by
  sorry

end fraction_simplification

namespace sequence_term_1000

import Mathlib

theorem sequence_term_1000 (a : ℕ → ℤ) 
  (h1 : a 1 = 2010) 
  (h2 : a 2 = 2011) 
  (h3 : ∀ n, 1 ≤ n → a n + a (n + 1) + a (n + 2) = 2 * n) : 
  a 1000 = 2676 :=
sorry

end sequence_term_1000

namespace wendy_full_face_time

import Mathlib

-- Define the constants based on the conditions
def num_products := 5
def wait_time := 5
def makeup_time := 30

-- Calculate the total time to put on "full face"
def total_time (products : ℕ) (wait_time : ℕ) (makeup_time : ℕ) : ℕ :=
  (products - 1) * wait_time + makeup_time

-- The theorem stating that Wendy's full face routine takes 50 minutes
theorem wendy_full_face_time : total_time num_products wait_time makeup_time = 50 :=
by {
  -- the proof would be provided here, for now we use sorry
  sorry
}

end wendy_full_face_time

namespace max_value_of_f

import Mathlib

theorem max_value_of_f :
  ∀ (x : ℝ), -5 ≤ x ∧ x ≤ 13 → ∃ (y : ℝ), y = x - 5 ∧ y ≤ 8 ∧ y >= -10 ∧ 
  (∀ (z : ℝ), z = (x - 5) → z ≤ 8) := 
by
  sorry

end max_value_of_f

namespace shared_property

import Mathlib

-- Definitions of the shapes
structure Parallelogram where
  sides_equal    : Bool -- Parallelograms have opposite sides equal but not necessarily all four.

structure Rectangle where
  sides_equal    : Bool -- Rectangles have opposite sides equal.
  diagonals_equal: Bool

structure Rhombus where
  sides_equal: Bool -- Rhombuses have all sides equal.
  diagonals_perpendicular: Bool -- Diagonals of a rhombus are perpendicular.

structure Square where
  sides_equal: Bool -- Squares have all sides equal.
  diagonals_perpendicular: Bool -- Diagonals of a square are perpendicular.
  diagonals_equal: Bool -- Diagonals of a square are equal in length.

-- Definitions of properties
def all_sides_equal (p1 p2 p3 p4 : Parallelogram) := p1.sides_equal ∧ p2.sides_equal ∧ p3.sides_equal ∧ p4.sides_equal
def diagonals_equal (r1 r2 r3 : Rectangle) (s1 s2 : Square) := r1.diagonals_equal ∧ r2.diagonals_equal ∧ s1.diagonals_equal ∧ s2.diagonals_equal
def diagonals_perpendicular (r1 : Rhombus) (s1 s2 : Square) := r1.diagonals_perpendicular ∧ s1.diagonals_perpendicular ∧ s2.diagonals_perpendicular
def diagonals_bisect_each_other (p1 p2 p3 p4 : Parallelogram) (r1 : Rectangle) (r2 : Rhombus) (s1 s2 : Square) := True -- All these shapes have diagonals that bisect each other.

-- The statement we need to prove
theorem shared_property (p1 p2 p3 p4 : Parallelogram) (r1 r2 : Rectangle) (r3 : Rhombus) (s1 s2 : Square) : 
  (diagonals_bisect_each_other p1 p2 p3 p4 r1 r3 s1 s2) :=
by
  sorry

end shared_property

namespace binomial_coefficient_plus_ten

import Mathlib.Data.Nat.Choose.Basic
import Mathlib.Data.Nat.Factorial.Basic

theorem binomial_coefficient_plus_ten :
  Nat.choose 9 5 + 10 = 136 := 
by
  sorry

end binomial_coefficient_plus_ten

namespace sum_m_n

import Mathlib

-- Define the conditions and the result

def probabilityOfNo3x3RedSquare : ℚ :=
  65408 / 65536

def gcd_65408_65536 := Nat.gcd 65408 65536

def simplifiedProbability : ℚ :=
  probabilityOfNo3x3RedSquare / gcd_65408_65536

def m : ℕ :=
  511

def n : ℕ :=
  512

theorem sum_m_n : m + n = 1023 := by
  sorry

end sum_m_n

namespace value_of_a3_a6_a9

import Mathlib

variable (a : ℕ → ℝ) (d : ℝ)

-- Condition: The common difference is 2
axiom common_difference : d = 2

-- Condition: a_1 + a_4 + a_7 = -50
axiom sum_a1_a4_a7 : a 1 + a 4 + a 7 = -50

-- The goal: a_3 + a_6 + a_9 = -38
theorem value_of_a3_a6_a9 : a 3 + a 6 + a 9 = -38 := 
by 
  sorry

end value_of_a3_a6_a9

namespace find_multiplier

import Mathlib

theorem find_multiplier (x : ℕ) (h₁ : 3 * x = (26 - x) + 26) (h₂ : x = 13) : 3 = 3 := 
by 
  sorry

end find_multiplier

namespace estimate_time_pm

import Mathlib

-- Definitions from the conditions
def school_start_time : ℕ := 12
def classes : List String := ["Maths", "History", "Geography", "Science", "Music"]
def class_time : ℕ := 45  -- in minutes
def break_time : ℕ := 15  -- in minutes
def classes_up_to_science : List String := ["Maths", "History", "Geography", "Science"]
def total_classes_time : ℕ := classes_up_to_science.length * (class_time + break_time)

-- Lean statement to prove that given the conditions, the time is 4 pm
theorem estimate_time_pm :
  school_start_time + (total_classes_time / 60) = 16 :=
by
  sorry

end estimate_time_pm

namespace f_increasing_interval

import Mathlib

noncomputable def f (x : ℝ) : ℝ := Real.log (x^2 - 3 * x - 4)

def domain_f (x : ℝ) : Prop := (x < -1) ∨ (x > 4)

def increasing_g (a b : ℝ) : Prop := ∀ x y, a < x → x < y → y < b → (x^2 - 3 * x - 4 < y^2 - 3 * y - 4)

theorem f_increasing_interval :
  ∀ x, domain_f x → increasing_g 4 (a) → increasing_g 4 (b) → 
    (4 < x ∧ x < b) → (f x < f (b - 0.1)) := sorry

end f_increasing_interval

namespace find_second_number

import Mathlib

-- Defining the ratios and sum condition
def ratio (a b c : ℕ) := 5*a = 3*b ∧ 3*b = 4*c

theorem find_second_number (a b c : ℕ) (h_ratio : ratio a b c) (h_sum : a + b + c = 108) : b = 27 :=
by
  sorry

end find_second_number

namespace total_area_three_plots

import Mathlib

variable (x y z A : ℝ)

theorem total_area_three_plots :
  (x = (2 / 5) * A) →
  (z = x - 16) →
  (y = (9 / 8) * z) →
  (A = x + y + z) →
  A = 96 :=
by
  intros h1 h2 h3 h4
  sorry

end total_area_three_plots

namespace f_2010_plus_f_2011

import Mathlib

-- Definition of f being an odd function
def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f (x)

-- Conditions in Lean 4
variables (f : ℝ → ℝ)

axiom f_odd : odd_function f
axiom f_symmetry : ∀ x, f (1 + x) = f (1 - x)
axiom f_1 : f 1 = 2

-- The theorem to be proved
theorem f_2010_plus_f_2011 : f (2010) + f (2011) = -2 :=
by
  sorry

end f_2010_plus_f_2011

namespace find_number_of_spiders

import Mathlib

theorem find_number_of_spiders (S : ℕ) (h1 : (1 / 2) * S = 5) : S = 10 := sorry

end find_number_of_spiders

namespace trapezoid_smallest_angle

import Mathlib

theorem trapezoid_smallest_angle (a d : ℝ) 
  (h1 : a + 3 * d = 140)
  (h2 : 2 * a + 3 * d = 180) : 
  a = 20 :=
by
  sorry

end trapezoid_smallest_angle

namespace product_eq_1519000000_div_6561

import Mathlib

-- Given conditions
def P (X : ℚ) : ℚ := X - 5
def Q (X : ℚ) : ℚ := X + 5
def R (X : ℚ) : ℚ := X / 2
def S (X : ℚ) : ℚ := 2 * X

theorem product_eq_1519000000_div_6561 
  (X : ℚ) 
  (h : (P X) + (Q X) + (R X) + (S X) = 100) :
  (P X) * (Q X) * (R X) * (S X) = 1519000000 / 6561 := 
by sorry

end product_eq_1519000000_div_6561

namespace sin_alpha_value

import Mathlib

theorem sin_alpha_value (α : ℝ) 
  (h1 : Real.sin (α - Real.pi / 4) = (7 * Real.sqrt 2) / 10)
  (h2 : Real.cos (2 * α) = 7 / 25) : 
  Real.sin α = 3 / 5 :=
sorry

end sin_alpha_value

namespace tangent_line_intersecting_lines

import Mathlib

variable (x y : ℝ)

-- Definition of the circle
def circle_C : Prop := (x - 3)^2 + (y - 4)^2 = 4

-- Definition of the point
def point_A : Prop := x = 1 ∧ y = 0

-- (I) Prove that if l is tangent to circle C and passes through A, l is 3x - 4y - 3 = 0
theorem tangent_line (l : ℝ → ℝ) (h : ∀ x, l x = k * (x - 1)) :
  (∀ {x y}, circle_C x y → 3 * x - 4 * y - 3 = 0) :=
by
  sorry

-- (II) Prove that the maximum area of triangle CPQ intersecting circle C is 2, and l's equations are y = 7x - 7 or y = x - 1
theorem intersecting_lines (k : ℝ) :
  (∃ x y, circle_C x y ∧ point_A x y) →
  (∃ k : ℝ, k = 7 ∨ k = 1) :=
by
  sorry

end tangent_line_intersecting_lines

namespace ratio_greater_than_one_ratio_greater_than_one_neg

import Mathlib

theorem ratio_greater_than_one (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a / b > 1) : a > b :=
by
  sorry

theorem ratio_greater_than_one_neg (a b : ℝ) (h1 : a < 0) (h2 : b < 0) (h3 : a / b > 1) : a < b :=
by
  sorry

end ratio_greater_than_one_ratio_greater_than_one_neg

namespace minimum_value_y_range_of_a

import Mathlib

noncomputable def f (x a : ℝ) : ℝ := x^2 - 2*a*x - 1 + a

theorem minimum_value_y (x : ℝ) 
  (hx_pos : x > 0) : (f x 2 / x) = -2 :=
by sorry

theorem range_of_a : 
  ∀ a : ℝ, ∀ x ∈ (Set.Icc 0 2), (f x a) ≤ a ↔ a ≥ 3 / 4 :=
by sorry

end minimum_value_y_range_of_a

namespace pete_total_blocks_traveled

import Mathlib

theorem pete_total_blocks_traveled : 
    ∀ (walk_to_garage : ℕ) (bus_to_post_office : ℕ), 
    walk_to_garage = 5 → bus_to_post_office = 20 → 
    ((walk_to_garage + bus_to_post_office) * 2) = 50 :=
by
  intros walk_to_garage bus_to_post_office h_walk h_bus
  sorry

end pete_total_blocks_traveled

namespace example_equation

import Mathlib

-- Define what it means to be an equation in terms of containing an unknown and being an equality
def is_equation (expr : Prop) (contains_unknown : Prop) : Prop :=
  (contains_unknown ∧ expr)

-- Prove that 4x + 2 = 10 is an equation
theorem example_equation : is_equation (4 * x + 2 = 10) (∃ x : ℝ, true) :=
  by sorry

end example_equation

namespace fraction_pizza_covered_by_pepperoni

import Mathlib

theorem fraction_pizza_covered_by_pepperoni :
  (∀ (r_pizz : ℝ) (n_pepp : ℕ) (d_pepp : ℝ),
      r_pizz = 8 ∧ n_pepp = 32 ∧ d_pepp = 2 →
      (n_pepp * π * (d_pepp / 2)^2) / (π * r_pizz^2) = 1 / 2) :=
sorry

end fraction_pizza_covered_by_pepperoni

namespace max_ab

import Mathlib

theorem max_ab (a b : ℝ) (h : a + b = 1) : ab ≤ 1 / 4 :=
by
  sorry

end max_ab

namespace still_need_more_volunteers

import Mathlib

def total_volunteers_needed : ℕ := 80
def students_volunteering_per_class : ℕ := 4
def number_of_classes : ℕ := 5
def teacher_volunteers : ℕ := 10
def total_student_volunteers : ℕ := students_volunteering_per_class * number_of_classes
def total_volunteers_so_far : ℕ := total_student_volunteers + teacher_volunteers

theorem still_need_more_volunteers : total_volunteers_needed - total_volunteers_so_far = 50 := by
  sorry

end still_need_more_volunteers

namespace inequality_proof

import Mathlib

variables {a1 a2 a3 b1 b2 b3 : ℝ}

theorem inequality_proof (h1 : 0 < a1) (h2 : 0 < a2) (h3 : 0 < a3) 
                         (h4 : 0 < b1) (h5 : 0 < b2) (h6 : 0 < b3):
  (a1 * b2 + a2 * b1 + a2 * b3 + a3 * b2 + a3 * b1 + a1 * b3)^2 
  ≥ 4 * (a1 * a2 + a2 * a3 + a3 * a1) * (b1 * b2 + b2 * b3 + b3 * b1) := 
sorry

end inequality_proof

namespace sample_size_of_survey

import Mathlib

def eighth_grade_students : ℕ := 350
def selected_students : ℕ := 50

theorem sample_size_of_survey : selected_students = 50 :=
by sorry

end sample_size_of_survey

namespace coprime_pairs_solution

import Mathlib

theorem coprime_pairs_solution (x y : ℕ) (hx : x ∣ y^2 + 210) (hy : y ∣ x^2 + 210) (hxy : Nat.gcd x y = 1) :
  (x = 1 ∧ y = 1) ∨ (x = 1 ∧ y = 211) :=
by sorry

end coprime_pairs_solution

namespace problem

import Mathlib

theorem problem (C D : ℝ) (h : ∀ x : ℝ, x ≠ 4 → 
  (C / (x - 4)) + D * (x + 2) = (-2 * x^3 + 8 * x^2 + 35 * x + 48) / (x - 4)) : 
  C + D = 174 :=
sorry

end problem

namespace hollis_student_loan

import Mathlib

theorem hollis_student_loan
  (interest_loan1 : ℝ)
  (interest_loan2 : ℝ)
  (total_loan1 : ℝ)
  (total_loan2 : ℝ)
  (additional_amount : ℝ)
  (total_interest_paid : ℝ) :
  interest_loan1 = 0.07 →
  total_loan1 = total_loan2 + additional_amount →
  additional_amount = 1500 →
  total_interest_paid = 617 →
  total_loan2 = 4700 →
  total_loan1 * interest_loan1 + total_loan2 * interest_loan2 = total_interest_paid →
  total_loan2 = 4700 :=
by
  sorry

end hollis_student_loan

namespace time_difference_correct

import Mathlib

-- Definitions based on conditions
def malcolm_speed : ℝ := 5 -- Malcolm's speed in minutes per mile
def joshua_speed : ℝ := 7 -- Joshua's speed in minutes per mile
def race_length : ℝ := 12 -- Length of the race in miles

-- Calculate times based on speeds and race length
def malcolm_time : ℝ := malcolm_speed * race_length
def joshua_time : ℝ := joshua_speed * race_length

-- The statement that the difference in finish times is 24 minutes
theorem time_difference_correct : joshua_time - malcolm_time = 24 :=
by
  -- Proof goes here
  sorry

end time_difference_correct

namespace min_det_is_neg_six

import Mathlib

-- Define the set of possible values for a, b, c, d
def values : List ℤ := [-1, 1, 2]

-- Define the determinant function for a 2x2 matrix
def det (a b c d : ℤ) : ℤ := a * d - b * c

-- State the theorem that the minimum value of the determinant is -6
theorem min_det_is_neg_six :
  ∃ (a b c d : ℤ), a ∈ values ∧ b ∈ values ∧ c ∈ values ∧ d ∈ values ∧ 
  (∀ (a' b' c' d' : ℤ), a' ∈ values → b' ∈ values → c' ∈ values → d' ∈ values → det a b c d ≤ det a' b' c' d') ∧ det a b c d = -6 :=
by
  sorry

end min_det_is_neg_six

namespace probability_of_male_selected

import Mathlib

-- Define the total number of students
def num_students : ℕ := 100

-- Define the number of male students
def num_male_students : ℕ := 25

-- Define the number of students selected
def num_students_selected : ℕ := 20

theorem probability_of_male_selected :
  (num_students_selected : ℚ) / num_students = 1 / 5 :=
by
  sorry

end probability_of_male_selected

namespace chocolates_cost

import Mathlib

-- Define the conditions given in the problem.
def boxes_needed (candies_total : ℕ) (candies_per_box : ℕ) : ℕ := 
    candies_total / candies_per_box

def total_cost_without_discount (num_boxes : ℕ) (cost_per_box : ℕ) : ℕ := 
    num_boxes * cost_per_box

def discount (total_cost : ℕ) : ℕ := 
    total_cost * 10 / 100

def final_cost (total_cost : ℕ) (discount : ℕ) : ℕ :=
    total_cost - discount

-- Theorem stating the total cost of buying 660 chocolate after discount is $138.60
theorem chocolates_cost (candies_total : ℕ) (candies_per_box : ℕ) (cost_per_box : ℕ) : 
     candies_total = 660 ∧ candies_per_box = 30 ∧ cost_per_box = 7 → 
     final_cost (total_cost_without_discount (boxes_needed candies_total candies_per_box) cost_per_box) 
          (discount (total_cost_without_discount (boxes_needed candies_total candies_per_box) cost_per_box)) = 13860 := 
by 
    intros h
    let ⟨h1, h2, h3⟩ := h 
    sorry 

end chocolates_cost

namespace paint_liters_needed

import Mathlib

theorem paint_liters_needed :
  let cost_brushes : ℕ := 20
  let cost_canvas : ℕ := 3 * cost_brushes
  let cost_paint_per_liter : ℕ := 8
  let total_costs : ℕ := 120
  ∃ (liters_of_paint : ℕ), cost_brushes + cost_canvas + cost_paint_per_liter * liters_of_paint = total_costs ∧ liters_of_paint = 5 :=
by
  sorry

end paint_liters_needed

namespace parabola_intersections

import Mathlib

theorem parabola_intersections :
  (∀ x y, (y = 4 * x^2 + 4 * x - 7) ↔ (y = x^2 + 5)) →
  (∃ (points : List (ℝ × ℝ)),
    (points = [(-2, 9), (2, 9)]) ∧
    (∀ p ∈ points, ∃ x, p = (x, x^2 + 5) ∧ y = 4 * x^2 + 4 * x - 7)) :=
by sorry

end parabola_intersections

namespace center_of_gravity_shift_center_of_gravity_shift_result

import Mathlib

variable (l s : ℝ) (s_val : s = 60)
#check (s_val : s = 60)

theorem center_of_gravity_shift : abs ((l / 2) - ((l - s) / 2)) = s / 2 := 
by sorry

theorem center_of_gravity_shift_result : (s / 2 = 30) :=
by sorry

end center_of_gravity_shift_center_of_gravity_shift_result

namespace problem_statement

import Mathlib

noncomputable def x : ℝ := (3 + Real.sqrt 8) ^ 30
noncomputable def n : ℤ := Int.floor x
noncomputable def f : ℝ := x - n

theorem problem_statement : x * (1 - f) = 1 := 
by 
  sorry

end problem_statement

namespace total_profit_correct

import Mathlib

noncomputable def total_profit (a b c : ℕ) (c_share : ℕ) : ℕ :=
  let ratio := a + b + c
  let part_value := c_share / c
  ratio * part_value

theorem total_profit_correct (h_a : ℕ := 5000) (h_b : ℕ := 8000) (h_c : ℕ := 9000) (h_c_share : ℕ := 36000) :
  total_profit h_a h_b h_c h_c_share = 88000 :=
by
  sorry

end total_profit_correct

namespace proof_numbers_exist

import Mathlib

noncomputable def exists_numbers : Prop :=
  ∃ a b c : ℕ, a > 10^10 ∧ b > 10^10 ∧ c > 10^10 ∧
  (a * b % (a + 2012) = 0) ∧
  (a * c % (a + 2012) = 0) ∧
  (b * c % (b + 2012) = 0) ∧
  (a * b * c % (b + 2012) = 0) ∧
  (a * b * c % (c + 2012) = 0)

theorem proof_numbers_exist : exists_numbers :=
  sorry

end proof_numbers_exist

namespace ellipse_with_foci_on_x_axis

import Mathlib

theorem ellipse_with_foci_on_x_axis (a : ℝ) :
  (∀ x y : ℝ, (x^2) / (a - 5) + (y^2) / 2 = 1 →  
   (∃ cx cy : ℝ, ∀ x', cx - x' = a - 5 ∧ cy = 2)) → 
  a > 7 :=
by sorry

end ellipse_with_foci_on_x_axis

namespace water_in_pool_after_35_days

import Mathlib

theorem water_in_pool_after_35_days :
  ∀ (initial_amount : ℕ) (evap_rate : ℕ) (cycle_days : ℕ) (add_amount : ℕ) (total_days : ℕ),
  initial_amount = 300 → evap_rate = 1 → cycle_days = 5 → add_amount = 5 → total_days = 35 →
  initial_amount - evap_rate * total_days + (total_days / cycle_days) * add_amount = 300 :=
by
  intros initial_amount evap_rate cycle_days add_amount total_days h₁ h₂ h₃ h₄ h₅
  sorry

end water_in_pool_after_35_days

namespace complex_number_solution

import Mathlib

theorem complex_number_solution (a b : ℝ) (i : ℂ) (h₀ : Complex.I = i)
  (h₁ : (a - 2* (i^3)) / (b + i) = i) : a + b = 1 :=
by 
  sorry

end complex_number_solution

namespace poly_expansion

import Mathlib

def poly1 (z : ℝ) := 5 * z^3 + 4 * z^2 - 3 * z + 7
def poly2 (z : ℝ) := 2 * z^4 - z^3 + z - 2
def poly_product (z : ℝ) := 10 * z^7 + 6 * z^6 - 10 * z^5 + 22 * z^4 - 13 * z^3 - 11 * z^2 + 13 * z - 14

theorem poly_expansion (z : ℝ) : poly1 z * poly2 z = poly_product z := by
  sorry

end poly_expansion

namespace simplify_and_evaluate_expression

import Mathlib

variable (a b : ℤ)

theorem simplify_and_evaluate_expression (h1 : a = 1) (h2 : b = -1) :
  (3 * a^2 * b - 2 * (a * b - (3/2) * a^2 * b) + a * b - 2 * a^2 * b) = -3 := by
  sorry

end simplify_and_evaluate_expression

namespace al_told_the_truth

import Mathlib

-- Definitions of G, S, and B based on each pirate's claim
def tom_G := 10
def tom_S := 8
def tom_B := 11

def al_G := 9
def al_S := 11
def al_B := 10

def pit_G := 10
def pit_S := 10
def pit_B := 9

def jim_G := 8
def jim_S := 10
def jim_B := 11

-- Condition that the total number of coins is 30
def total_coins (G : ℕ) (S : ℕ) (B : ℕ) : Prop := G + S + B = 30

-- The assertion that only Al told the truth
theorem al_told_the_truth :
  (total_coins tom_G tom_S tom_B → false) →
  (total_coins al_G al_S al_B) →
  (total_coins pit_G pit_S pit_B → false) →
  (total_coins jim_G jim_S jim_B → false) →
  true :=
by
  intros
  sorry

end al_told_the_truth

namespace find_a_monotonic_intervals_exp_gt_xsquare_plus_one

import Mathlib

-- Define the function f(x) and its derivative f'(x)
noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.exp x - a * x - 1
noncomputable def f_prime (x : ℝ) (a : ℝ) : ℝ := Real.exp x - a

-- Prove that a = 2 given the slope condition at x = 0
theorem find_a (a : ℝ) (h : f_prime 0 a = -1) : a = 2 :=
by sorry

-- Characteristics of the function f(x)
theorem monotonic_intervals (a : ℝ) (h : a = 2) :
  ∀ x : ℝ, (x ≤ Real.log 2 → f_prime x a ≤ 0) ∧ (x >= Real.log 2 → f_prime x a >= 0) :=
by sorry

-- Prove that e^x > x^2 + 1 when x > 0
theorem exp_gt_xsquare_plus_one (x : ℝ) (hx : x > 0) : Real.exp x > x^2 + 1 :=
by sorry

end find_a_monotonic_intervals_exp_gt_xsquare_plus_one

namespace initial_friends_count

import Mathlib

variable (F : ℕ)
variable (players_quit : ℕ)
variable (lives_per_player : ℕ)
variable (total_remaining_lives : ℕ)

theorem initial_friends_count
  (h1 : players_quit = 7)
  (h2 : lives_per_player = 8)
  (h3 : total_remaining_lives = 72) :
  F = 16 :=
by
  have h4 : 8 * (F - 7) = 72 := by sorry   -- Derived from given conditions
  have : 8 * F - 56 = 72 := by sorry        -- Simplify equation
  have : 8 * F = 128 := by sorry           -- Add 56 to both sides
  have : F = 16 := by sorry                -- Divide both sides by 8
  exact this                               -- Final result

end initial_friends_count

namespace no_naturals_satisfy_divisibility_condition

import Mathlib

theorem no_naturals_satisfy_divisibility_condition :
  ∀ (a b c : ℕ), ¬ (2013 * (a * b + b * c + c * a) ∣ a^2 + b^2 + c^2) :=
by
  sorry

end no_naturals_satisfy_divisibility_condition

namespace 

import Mathlib

noncomputable def f (x : ℝ) : ℝ := 2^x + x - 5

theorem root_in_interval_sum_eq_three {a b : ℤ} (h1 : b - a = 1) (h2 : ∃ x : ℝ, a < x ∧ x < b ∧ f x = 0) :
  a + b = 3 :=
by
  sorry

end 

namespace sum_of_80th_equation

import Mathlib

theorem sum_of_80th_equation : (2 * 80 + 1) + (5 * 80 - 1) = 560 := by
  sorry

end sum_of_80th_equation

namespace integer_type

import Mathlib

theorem integer_type (f : ℕ) (h : f = 14) (x : ℕ) (hx : 3150 * f = x * x) : f > 0 :=
by
  sorry

end integer_type

namespace rectangle_side_ratio

import Mathlib

noncomputable def sin_30_deg := 1 / 2

theorem rectangle_side_ratio 
  (a b c : ℝ) 
  (h1 : a + b = 2 * c) 
  (h2 : a * b = (c ^ 2) / 2) :
  (a / b = 3 + 2 * Real.sqrt 2) ∨ (a / b = 3 - 2 * Real.sqrt 2) :=
by
  sorry

end rectangle_side_ratio

namespace ratio_quadrilateral

import Mathlib

theorem ratio_quadrilateral
  (ABCD_area : ℝ)
  (h_ABCD : ABCD_area = 40)
  (K L M N : Type)
  (AK KB : ℝ)
  (h_ratio : AK / KB = BL / LC ∧ BL / LC = CM / MD ∧ CM / MD = DN / NA)
  (KLMN_area : ℝ)
  (h_KLMN : KLMN_area = 25) :
  (AK / (AK + KB) = 1 / 4 ∨ AK / (AK + KB) = 3 / 4) :=
sorry

end ratio_quadrilateral

namespace point_in_second_quadrant

import Mathlib

theorem point_in_second_quadrant (P : ℝ × ℝ)
  (h1 : P.1 < 0) -- Point P is in the second quadrant, so its x-coordinate is negative
  (h2 : 0 < P.2) -- Point P is in the second quadrant, so its y-coordinate is positive
  (h3 : |P.2| = 3) -- The distance from P to the x-axis is 3
  (h4 : |P.1| = 4) -- The distance from P to the y-axis is 4
  : P = (-4, 3) := 
  sorry

end point_in_second_quadrant

namespace largest_square_perimeter

import Mathlib

-- Define the conditions
def rectangle_length : ℕ := 80
def rectangle_width : ℕ := 60

-- Define the theorem to prove
theorem largest_square_perimeter : 4 * rectangle_width = 240 := by
  -- The proof steps are omitted
  sorry

end largest_square_perimeter

namespace jessica_current_age

import Mathlib

-- Definitions and conditions from the problem
def J (M : ℕ) : ℕ := M / 2
def M : ℕ := 60

-- Lean statement for the proof problem
theorem jessica_current_age : J M + 10 = 40 :=
by
  sorry

end jessica_current_age

namespace product_computation

import Mathlib

def product_fraction_compute : ℕ :=
  ∏ n in Finset.range (22 - 2 + 1), (n + 5) * (if (1 ≤ n + 2) then 1 else 0)

theorem product_computation : product_fraction_compute = 14950 := by
  sorry

end product_computation

namespace problem_1_problem_2

import Mathlib

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

end problem_1_problem_2

namespace number_of_ways_to_make_125_quacks_using_1_5_25_125_coins

import Mathlib

def num_ways_to_make_125_quacks_using_coins : ℕ :=
  have h : ∃ (a b c d : ℕ), a + 5 * b + 25 * c + 125 * d = 125 := sorry
  82

theorem number_of_ways_to_make_125_quacks_using_1_5_25_125_coins : num_ways_to_make_125_quacks_using_coins = 82 := 
  sorry

end number_of_ways_to_make_125_quacks_using_1_5_25_125_coins

namespace pencil_groups

import Mathlib

theorem pencil_groups (total_pencils number_per_group number_of_groups : ℕ) 
  (h_total: total_pencils = 25) 
  (h_group: number_per_group = 5) 
  (h_eq: total_pencils = number_per_group * number_of_groups) : 
  number_of_groups = 5 :=
by
  sorry

end pencil_groups

namespace track_meet_total_people

import Mathlib

theorem track_meet_total_people (B G : ℕ) (H1 : B = 30)
  (H2 : ∃ G, (3 * G) / 5 + (2 * G) / 5 = G)
  (H3 : ∀ G, 2 * G / 5 = 10) :
  B + G = 55 :=
by
  sorry

end track_meet_total_people

namespace least_x

import Mathlib

theorem least_x (x p : ℕ) (h1 : 0 < x) (h2: Nat.Prime p) (h3: ∃ q : ℕ, Nat.Prime q ∧ q % 2 = 1 ∧ x = 11 * p * q) : x ≥ 66 := 
sorry

end least_x

namespace inequality_pos_real

import Mathlib

theorem inequality_pos_real (
  a b c : ℝ
) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  abc ≥ (a + b + c) / (1 / a^2 + 1 / b^2 + 1 / c^2) ∧ 
  (a + b + c) / (1 / a^2 + 1 / b^2 + 1 / c^2) ≥ (a + b - c) * (b + c - a) * (c + a - b) := 
sorry

end inequality_pos_real

namespace roja_speed

import Mathlib

theorem roja_speed (R : ℕ) (h1 : 3 + R = 7) : R = 7 - 3 :=
by sorry

end roja_speed

namespace problem_solution

import Mathlib

theorem problem_solution : (6 * 7 * 8 * 9 * 10) / (6 + 7 + 8 + 9 + 10) = 756 := by
  sorry

end problem_solution

namespace segment_length

import Mathlib
noncomputable def cube_root27 : ℝ := 3

theorem segment_length : ∀ (x : ℝ), (|x - cube_root27| = 4) → ∃ (a b : ℝ), (a = cube_root27 + 4) ∧ (b = cube_root27 - 4) ∧ |a - b| = 8 :=
by
  sorry

end segment_length

namespace cost_of_one_unit_each

import Mathlib

variables (x y z : ℝ)

theorem cost_of_one_unit_each
  (h1 : 2 * x + 3 * y + z = 130)
  (h2 : 3 * x + 5 * y + z = 205) :
  x + y + z = 55 :=
by
  sorry

end cost_of_one_unit_each

namespace outlet_two_rate

import Mathlib

/-- Definitions and conditions for the problem -/
def tank_volume_feet : ℝ := 20
def inlet_rate_cubic_inches_per_min : ℝ := 5
def outlet_one_rate_cubic_inches_per_min : ℝ := 9
def empty_time_minutes : ℝ := 2880
def cubic_feet_to_cubic_inches : ℝ := 1728
def tank_volume_cubic_inches := tank_volume_feet * cubic_feet_to_cubic_inches

/-- Statement to prove the rate of the other outlet pipe -/
theorem outlet_two_rate (x : ℝ) :
  tank_volume_cubic_inches / empty_time_minutes = outlet_one_rate_cubic_inches_per_min + x - inlet_rate_cubic_inches_per_min → 
  x = 8 :=
by
  sorry

end outlet_two_rate

namespace half_of_number

import Mathlib

theorem half_of_number (x : ℝ) (h : (4 / 15 * 5 / 7 * x - 4 / 9 * 2 / 5 * x = 8)) : (1 / 2 * x = 315) :=
sorry

end half_of_number

namespace average_score_for_girls

import Mathlib

variable (A a B b : ℕ)
variable (h1 : 71 * A + 76 * a = 74 * (A + a))
variable (h2 : 81 * B + 90 * b = 84 * (B + b))
variable (h3 : 71 * A + 81 * B = 79 * (A + B))

theorem average_score_for_girls
  (h1 : 71 * A + 76 * a = 74 * (A + a))
  (h2 : 81 * B + 90 * b = 84 * (B + b))
  (h3 : 71 * A + 81 * B = 79 * (A + B))
  : (76 * a + 90 * b) / (a + b) = 84 := by
  sorry

end average_score_for_girls

namespace nth_term_correct

import Mathlib

noncomputable def term_in_sequence (n : ℕ) : ℚ :=
  2^n / (2^n + 3)

theorem nth_term_correct (n : ℕ) : term_in_sequence n = 2^n / (2^n + 3) :=
by
  sorry

end nth_term_correct

namespace perfect_square_condition

import Mathlib

theorem perfect_square_condition (x m : ℝ) (h : ∃ k : ℝ, x^2 + x + 2*m = k^2) : m = 1/8 := 
sorry

end perfect_square_condition

namespace vibrations_proof

import Mathlib

-- Define the conditions
def vibrations_lowest : ℕ := 1600
def increase_percentage : ℕ := 60
def use_time_minutes : ℕ := 5

-- Convert percentage to a multiplier
def percentage_to_multiplier (p : ℕ) : ℤ := (p : ℤ) / 100

-- Calculate the vibrations per second at the highest setting
def vibrations_highest := vibrations_lowest + (vibrations_lowest * percentage_to_multiplier increase_percentage).toNat

-- Convert time from minutes to seconds
def use_time_seconds := use_time_minutes * 60

-- Calculate the total vibrations Matt experiences
noncomputable def total_vibrations : ℕ := vibrations_highest * use_time_seconds

-- State the theorem
theorem vibrations_proof : total_vibrations = 768000 := 
by
  sorry

end vibrations_proof

namespace inscribed_squares_ratio

import Mathlib

theorem inscribed_squares_ratio (a b : ℝ) (h_triangle : 5^2 + 12^2 = 13^2)
    (h_square1 : a = 25 / 37) (h_square2 : b = 10) :
    a / b = 25 / 370 :=
by 
  sorry

end inscribed_squares_ratio

namespace largest_binomial_coefficient_term_largest_coefficient_term_remainder_mod_7

import Mathlib

-- Definitions and conditions
def binomial_expansion (x : ℕ) (n : ℕ) : ℕ := (x + (3 * x^2))^n

-- Problem statements
theorem largest_binomial_coefficient_term :
  ∀ x n,
  (x + 3 * x^2)^n = (x + 3 * x^2)^7 →
  (2^n = 128) →
  ∃ t : ℕ, t = 2835 * x^11 := 
by sorry

theorem largest_coefficient_term :
  ∀ x n,
  (x + 3 * x^2)^n = (x + 3 * x^2)^7 →
  exists t, t = 5103 * x^13 :=
by sorry

theorem remainder_mod_7 :
  ∀ x n,
  x = 3 →
  n = 2016 →
  (x + (3 * x^2))^n % 7 = 1 :=
by sorry

end largest_binomial_coefficient_term_largest_coefficient_term_remainder_mod_7

namespace find_m

import Mathlib

noncomputable def f (x a : ℝ) : ℝ := x - a

theorem find_m (a m : ℝ) :
  (∀ x, 0 ≤ x ∧ x ≤ 4 → f x a ≤ 2) →
  (∃ x, -2 ≤ x ∧ x ≤ 4 ∧ -1 - f (x + 1) a ≤ m) :=
sorry

end find_m

namespace sum_of_squares_of_reciprocals

import Mathlib

-- Definitions based on the problem's conditions
variables (a b : ℝ) (hab : a + b = 3 * a * b + 1) (h_an : a ≠ 0) (h_bn : b ≠ 0)

-- Statement of the problem to be proved
theorem sum_of_squares_of_reciprocals :
  (1 / a^2) + (1 / b^2) = (4 * a * b + 10) / (a^2 * b^2) :=
sorry

end sum_of_squares_of_reciprocals

namespace fractional_eq_range_m

import Mathlib

theorem fractional_eq_range_m (m : ℝ) :
  (∃ x : ℝ, x ≥ 0 ∧ (m / (x - 2) + 1 = x / (2 - x))) → m ≤ 2 ∧ m ≠ -2 :=
by
  sorry

end fractional_eq_range_m

namespace village_population_rate_decrease

import Mathlib

/--
Village X has a population of 78,000, which is decreasing at a certain rate \( R \) per year.
Village Y has a population of 42,000, which is increasing at the rate of 800 per year.
In 18 years, the population of the two villages will be equal.
We aim to prove that the rate of decrease in population per year for Village X is 1200.
-/
theorem village_population_rate_decrease (R : ℝ) 
  (hx : 78000 - 18 * R = 42000 + 18 * 800) : 
  R = 1200 :=
by
  sorry

end village_population_rate_decrease

namespace smallest_positive_multiple_of_3_4_5_is_60

import Mathlib

theorem smallest_positive_multiple_of_3_4_5_is_60 :
  ∃ n : ℕ, n > 0 ∧ (n % 3 = 0) ∧ (n % 4 = 0) ∧ (n % 5 = 0) ∧ n = 60 :=
by
  use 60
  sorry

end smallest_positive_multiple_of_3_4_5_is_60

namespace full_day_students

import Mathlib

def total_students : ℕ := 80
def percentage_half_day_students : ℕ := 25

theorem full_day_students : 
  (total_students - (total_students * percentage_half_day_students / 100)) = 60 := by
  sorry

end full_day_students

namespace symmetric_line_equation

import Mathlib

theorem symmetric_line_equation (x y : ℝ) :
  (∃ x y : ℝ, 3 * x + 4 * y = 2) →
  (4 * x + 3 * y = 2) :=
by
  intros h
  sorry

end symmetric_line_equation

namespace maximum_value_of_m

import Mathlib

theorem maximum_value_of_m (x y : ℝ) (hx : x > 1 / 2) (hy : y > 1) : 
    (4 * x^2 / (y - 1) + y^2 / (2 * x - 1)) ≥ 8 := 
sorry

end maximum_value_of_m

namespace trapezoid_bisector_segment_length

import Mathlib

-- Definitions of the conditions
variables {a b c d t : ℝ}

noncomputable def semiperimeter (a b c d : ℝ) : ℝ := (a + b + c + d) / 2

-- The theorem statement
theorem trapezoid_bisector_segment_length
  (p : ℝ)
  (h_p : p = semiperimeter a b c d) :
  t^2 = (4 * b * d) / (b + d)^2 * (p - a) * (p - c) :=
sorry

end trapezoid_bisector_segment_length

namespace prod_f_roots

import Mathlib

def f (x : ℂ) : ℂ := x^2 + 1

theorem prod_f_roots :
  ∀ (x₁ x₂ x₃ x₄ x₅ : ℂ),
  (x₁^5 - x₁^2 + 5 = 0) →
  (x₂^5 - x₂^2 + 5 = 0) →
  (x₃^5 - x₃^2 + 5 = 0) →
  (x₄^5 - x₂^2 + 5 = 0) →
  (x₅^5 - x₅^2 + 5 = 0) →
  (∏ k in ({x₁, x₂, x₃, x₄, x₅} : Finset ℂ), f k) = 37 :=
by
  sorry

end prod_f_roots

namespace gcd_a_b

import Mathlib

def a : ℕ := 6666666
def b : ℕ := 999999999

theorem gcd_a_b : Nat.gcd a b = 3 := by
  sorry

end gcd_a_b

namespace susan_hours_per_day

import Mathlib

theorem susan_hours_per_day (h : ℕ) 
  (works_five_days_a_week : Prop)
  (paid_vacation_days : ℕ)
  (unpaid_vacation_days : ℕ)
  (missed_pay : ℕ)
  (hourly_rate : ℕ)
  (total_vacation_days : ℕ)
  (total_workdays_in_2_weeks : ℕ)
  (paid_vacation_days_eq : paid_vacation_days = 6)
  (unpaid_vacation_days_eq : unpaid_vacation_days = 4)
  (missed_pay_eq : missed_pay = 480)
  (hourly_rate_eq : hourly_rate = 15)
  (total_vacation_days_eq : total_vacation_days = 14)
  (total_workdays_in_2_weeks_eq : total_workdays_in_2_weeks = 10)
  (total_unpaid_hours_in_4_days : unpaid_vacation_days * hourly_rate = missed_pay) :
  h = 8 :=
by 
  -- We need to show that Susan works 8 hours a day
  sorry

end susan_hours_per_day

namespace hot_sauce_container_size

import Mathlib

theorem hot_sauce_container_size :
  let serving_size := 0.5
  let servings_per_day := 3
  let days := 20
  let total_consumed := servings_per_day * serving_size * days
  let one_quart := 32
  one_quart - total_consumed = 2 :=
by
  sorry

end hot_sauce_container_size

namespace remainder_when_2x_divided_by_7

import Mathlib

theorem remainder_when_2x_divided_by_7 (x y r : ℤ) (h1 : x = 10 * y + 3)
    (h2 : 2 * x = 7 * (3 * y) + r) (h3 : 11 * y - x = 2) : r = 1 := by
  sorry

end remainder_when_2x_divided_by_7

namespace min_distance_sq

import Mathlib

theorem min_distance_sq (x y : ℝ) (h : x - y - 1 = 0) : (x - 2) ^ 2 + (y - 2) ^ 2 = 1 / 2 :=
sorry

end min_distance_sq

namespace probability_unit_square_not_touch_central_2x2_square

import Mathlib

-- Given a 6x6 checkerboard with a marked 2x2 square at the center,
-- prove that the probability of choosing a unit square that does not touch
-- the marked 2x2 square is 2/3.

theorem probability_unit_square_not_touch_central_2x2_square : 
    let total_squares := 36
    let touching_squares := 12
    let squares_not_touching := total_squares - touching_squares
    (squares_not_touching : ℚ) / (total_squares : ℚ) = 2 / 3 := by
  sorry

end probability_unit_square_not_touch_central_2x2_square

namespace negate_neg_two

import Mathlib

theorem negate_neg_two : -(-2) = 2 := by
  -- The proof goes here
  sorry

end negate_neg_two

namespace find_w

import Mathlib

variables {x y : ℚ}

def w : ℚ × ℚ := (-48433 / 975, 2058 / 325)

def vec1 : ℚ × ℚ := (3, 2)
def vec2 : ℚ × ℚ := (3, 4)

def proj (u v : ℚ × ℚ) : ℚ × ℚ :=
  let dot_uv := u.1 * v.1 + u.2 * v.2
  let dot_vv := v.1 * v.1 + v.2 * v.2
  (dot_uv / dot_vv * v.1, dot_uv / dot_vv * v.2)

def p1 : ℚ × ℚ := (47 / 13, 31 / 13)
def p2 : ℚ × ℚ := (85 / 25, 113 / 25)

theorem find_w (hw : w = (x, y)) :
  proj ⟨x, y⟩ vec1 = p1 ∧
  proj ⟨x, y⟩ vec2 = p2 :=
sorry

end find_w

namespace relationship_y1_y2_y3

import Mathlib

theorem relationship_y1_y2_y3 (c y1 y2 y3 : ℝ) :
  (y1 = (-(1^2) + 2 * 1 + c))
  ∧ (y2 = (-(2^2) + 2 * 2 + c))
  ∧ (y3 = (-(5^2) + 2 * 5 + c))
  → (y2 > y1 ∧ y1 > y3) :=
by
  intro h
  sorry

end relationship_y1_y2_y3

namespace units_digit_of_7_pow_6_pow_5

import Mathlib

theorem units_digit_of_7_pow_6_pow_5 : (7^(6^5)) % 10 = 1 := by
  -- Proof goes here
  sorry

end units_digit_of_7_pow_6_pow_5

namespace vertex_coordinates_quadratic_through_point_a_less_than_neg_2_fifth

import Mathlib

noncomputable def quadratic_function (a : ℝ) (x : ℝ) : ℝ := (x + 1) * (a * x + 2 * a + 2)

theorem vertex_coordinates (a : ℝ) (H : a = 1) : 
    (∃ v_x v_y : ℝ, quadratic_function a v_x = v_y ∧ v_x = -5 / 2 ∧ v_y = -9 / 4) := 
by {
    sorry
}

theorem quadratic_through_point : 
    (∃ a : ℝ, (quadratic_function a 0 = -2) ∧ (∀ x, quadratic_function a x = -2 * (x + 1)^2)) := 
by {
    sorry
}

theorem a_less_than_neg_2_fifth 
  (x1 x2 y1 y2 a : ℝ) (H1 : x1 + x2 = 2) (H2 : x1 < x2) (H3 : y1 > y2) 
  (Hfunc : ∀ x, quadratic_function (a * x + 2 * a + 2) (x + 1) = quadratic_function x y) :
    a < -2 / 5 := 
by {
    sorry
}

end vertex_coordinates_quadratic_through_point_a_less_than_neg_2_fifth

namespace Eric_test_score

import Mathlib

theorem Eric_test_score (n : ℕ) (old_avg new_avg : ℚ) (Eric_score : ℚ) :
  n = 22 →
  old_avg = 84 →
  new_avg = 85 →
  Eric_score = (n * new_avg) - ((n - 1) * old_avg) →
  Eric_score = 106 :=
by
  intros h1 h2 h3 h4
  sorry

end Eric_test_score

namespace milk_left

import Mathlib

theorem milk_left (initial_milk : ℝ) (given_away : ℝ) (h_initial : initial_milk = 5) (h_given : given_away = 18 / 4) :
  ∃ remaining_milk : ℝ, remaining_milk = initial_milk - given_away ∧ remaining_milk = 1 / 2 :=
by
  use 1 / 2
  sorry

end milk_left

namespace digit_B_divisible_by_9

import Mathlib

theorem digit_B_divisible_by_9 (B : ℕ) (h1 : B ≤ 9) (h2 : (4 + B + B + 1 + 3) % 9 = 0) : B = 5 := 
by {
  /- Proof omitted -/
  sorry
}

end digit_B_divisible_by_9

namespace negation_correct

import Mathlib

-- Define the original statement as a predicate
def original_statement (x : ℝ) : Prop := x > 1 → x^2 ≤ x

-- Define the negation of the original statement as a predicate
def negated_statement : Prop := ∃ x : ℝ, x > 1 ∧ x^2 > x

-- Define the theorem that the negation of the original statement implies the negated statement
theorem negation_correct :
  ¬ (∀ x : ℝ, original_statement x) ↔ negated_statement := by
  sorry

end negation_correct

namespace fraction_of_usual_speed

import Mathlib

-- Definitions based on conditions
variable (S R : ℝ)
variable (h1 : S * 60 = R * 72)

-- Goal statement
theorem fraction_of_usual_speed (h1 : S * 60 = R * 72) : R / S = 5 / 6 :=
by
  sorry

end fraction_of_usual_speed

namespace find_a

import Mathlib

def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, x ≠ 0 → f (-x) = -f x

noncomputable def f (a x : ℝ) : ℝ := (1 / (2^x - 1)) + a

theorem find_a (a : ℝ) : 
  is_odd_function (f a) → a = 1 / 2 :=
by
  sorry

end find_a

namespace side_length_S2

import Mathlib

def square_side_length 
  (w h : ℕ)
  (R1 R2 : ℕ → ℕ → Prop) 
  (S1 S2 S3 : ℕ → Prop) 
  (r s : ℕ) 
  (combined_rectangle : ℕ × ℕ → Prop)
  (cond1 : combined_rectangle (3330, 2030))
  (cond2 : R1 r s) 
  (cond3 : R2 r s) 
  (cond4 : S1 (r + s)) 
  (cond5 : S2 s) 
  (cond6 : S3 (r + s)) 
  (cond7 : 2 * r + s = 2030) 
  (cond8 : 2 * r + 3 * s = 3330) : Prop :=
  s = 650

theorem side_length_S2 (w h : ℕ)
  (R1 R2 : ℕ → ℕ → Prop) 
  (S1 S2 S3 : ℕ → Prop) 
  (r s : ℕ) 
  (combined_rectangle : ℕ × ℕ → Prop)
  (cond1 : combined_rectangle (3330, 2030))
  (cond2 : R1 r s) 
  (cond3 : R2 r s) 
  (cond4 : S1 (r + s)) 
  (cond5 : S2 s) 
  (cond6 : S3 (r + s)) 
  (cond7 : 2 * r + s = 2030) 
  (cond8 : 2 * r + 3 * s = 3330) : square_side_length w h R1 R2 S1 S2 S3 r s combined_rectangle cond1 cond2 cond3 cond4 cond5 cond6 cond7 cond8 :=
sorry

end side_length_S2

namespace abs_expression_equals_one

import Mathlib

theorem abs_expression_equals_one : 
  abs (abs (-(abs (2 - 3)) + 2) - 2) = 1 := 
  sorry

end abs_expression_equals_one

namespace inclination_angle_x_eq_one

import Mathlib

noncomputable def inclination_angle_of_vertical_line (x : ℝ) : ℝ :=
if x = 1 then 90 else 0

theorem inclination_angle_x_eq_one :
  inclination_angle_of_vertical_line 1 = 90 :=
by
  sorry

end inclination_angle_x_eq_one

namespace michael_average_speed

import Mathlib

-- Definitions of conditions
def motorcycle_speed := 20 -- mph
def motorcycle_time := 40 / 60 -- hours
def jogging_speed := 5 -- mph
def jogging_time := 60 / 60 -- hours

-- Define the total distance
def motorcycle_distance := motorcycle_speed * motorcycle_time
def jogging_distance := jogging_speed * jogging_time
def total_distance := motorcycle_distance + jogging_distance

-- Define the total time
def total_time := motorcycle_time + jogging_time

-- The proof statement to be proven
theorem michael_average_speed :
  total_distance / total_time = 11 := 
sorry

end michael_average_speed

namespace problem_solution

import Mathlib

theorem problem_solution :
  (- (5 : ℚ) / 12) ^ 2023 * (12 / 5) ^ 2023 = -1 := 
by
  sorry

end problem_solution

namespace log_equivalence

import Mathlib

noncomputable def log_base (b x : ℝ) : ℝ := Real.log x / Real.log b

theorem log_equivalence (x : ℝ) (h : log_base 16 (x - 3) = 1 / 2) : log_base 256 (x + 1) = 3 / 8 :=
  sorry

end log_equivalence

namespace marley_total_fruits

import Mathlib

theorem marley_total_fruits (louis_oranges : ℕ) (louis_apples : ℕ) 
                            (samantha_oranges : ℕ) (samantha_apples : ℕ)
                            (marley_oranges : ℕ) (marley_apples : ℕ) : 
  (louis_oranges = 5) → (louis_apples = 3) → 
  (samantha_oranges = 8) → (samantha_apples = 7) → 
  (marley_oranges = 2 * louis_oranges) → (marley_apples = 3 * samantha_apples) → 
  (marley_oranges + marley_apples = 31) :=
by
  intros
  sorry

end marley_total_fruits

namespace time_relationship

import Mathlib

variable (T x : ℝ)
variable (h : T = x + (2/6) * x)

theorem time_relationship : T = (4/3) * x := by 
sorry

end time_relationship

namespace complex_product

import Mathlib

theorem complex_product (z1 z2 : ℂ) (h1 : Complex.abs z1 = 1) (h2 : Complex.abs z2 = 1) 
(h3 : z1 + z2 = -7/5 + (1/5) * Complex.I) : 
  z1 * z2 = 24/25 - (7/25) * Complex.I :=
by
  sorry

end complex_product

namespace sandy_earnings_correct

import Mathlib

def hourly_rate : ℕ := 15
def hours_worked_friday : ℕ := 10
def hours_worked_saturday : ℕ := 6
def hours_worked_sunday : ℕ := 14

def earnings_friday : ℕ := hours_worked_friday * hourly_rate
def earnings_saturday : ℕ := hours_worked_saturday * hourly_rate
def earnings_sunday : ℕ := hours_worked_sunday * hourly_rate

def total_earnings : ℕ := earnings_friday + earnings_saturday + earnings_sunday

theorem sandy_earnings_correct : total_earnings = 450 := by
  sorry

end sandy_earnings_correct

namespace fraction_of_second_year_given_not_third_year

import Mathlib

theorem fraction_of_second_year_given_not_third_year (total_students : ℕ) 
  (third_year_students : ℕ) (second_year_students : ℕ) :
  third_year_students = total_students * 30 / 100 →
  second_year_students = total_students * 10 / 100 →
  ↑second_year_students / (total_students - third_year_students) = (1 : ℚ) / 7 :=
by
  -- Proof omitted
  sorry

end fraction_of_second_year_given_not_third_year

namespace bus_passengers

import Mathlib

def passengers_after_first_stop := 7

def passengers_after_second_stop := passengers_after_first_stop - 3 + 5

def passengers_after_third_stop := passengers_after_second_stop - 2 + 4

theorem bus_passengers (passengers_after_first_stop passengers_after_second_stop passengers_after_third_stop : ℕ) : passengers_after_third_stop = 11 :=
by
  sorry

end bus_passengers

namespace cave_depth

import Mathlib

theorem cave_depth (current_depth remaining_distance : ℕ) (h₁ : current_depth = 849) (h₂ : remaining_distance = 369) :
  current_depth + remaining_distance = 1218 :=
by
  sorry

end cave_depth

namespace min_sum_one_over_xy

import Mathlib

theorem min_sum_one_over_xy (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x + y = 6) : 
  ∃ c, (∀ x y, (x > 0) → (y > 0) → (x + y = 6) → (c ≤ (1/x + 1/y))) ∧ (c = 2 / 3) :=
by 
  sorry

end min_sum_one_over_xy

namespace variance_of_scores

import Mathlib

-- Define the list of scores
def scores : List ℕ := [110, 114, 121, 119, 126]

-- Define the formula for variance calculation
def variance (l : List ℕ) : ℚ :=
  let n := l.length
  let mean := (l.sum : ℚ) / n
  (l.map (λ x => ((x : ℚ) - mean) ^ 2)).sum / n

-- The main theorem to be proved
theorem variance_of_scores :
  variance scores = 30.8 := 
  by
    sorry

end variance_of_scores

namespace part_I_part_II

import Mathlib

-- Define the conditions given in the problem
def set_A : Set ℝ := { x | -1 < x ∧ x < 3 }
def set_B (a b : ℝ) : Set ℝ := { x | x^2 - a * x + b < 0 }

-- Part I: Prove that if A = B, then a = 2 and b = -3
theorem part_I (a b : ℝ) (h : set_A = set_B a b) : a = 2 ∧ b = -3 :=
sorry

-- Part II: Prove that if b = 3 and A ∩ B ⊇ B, then the range of a is [-2√3, 4]
theorem part_II (a : ℝ) (b : ℝ := 3) (h : set_A ∩ set_B a b ⊇ set_B a b) : -2 * Real.sqrt 3 ≤ a ∧ a ≤ 4 :=
sorry

end part_I_part_II

namespace max_subsequences_2001

import Mathlib

theorem max_subsequences_2001 (seq : List ℕ) (h_len : seq.length = 2001) : 
  ∃ n : ℕ, n = 667^3 :=
sorry

end max_subsequences_2001

namespace percent_of_a_is_4b

import Mathlib

variable (a b : ℝ)
variable (h : a = 1.2 * b)

theorem percent_of_a_is_4b :
  (4 * b) = (10 / 3 * 100 * a) / 100 :=
by sorry

end percent_of_a_is_4b

namespace donut_selection_count

import Mathlib

def num_donut_selections : ℕ :=
  Nat.choose 9 3

theorem donut_selection_count : num_donut_selections = 84 := 
by
  sorry

end donut_selection_count

namespace crayons_per_box

import Mathlib

-- Define the conditions
def crayons : ℕ := 80
def boxes : ℕ := 10

-- State the proof problem
theorem crayons_per_box : (crayons / boxes) = 8 := by
  sorry

end crayons_per_box

namespace remainder_when_added_then_divided

import Mathlib

def num1 : ℕ := 2058167
def num2 : ℕ := 934
def divisor : ℕ := 8

theorem remainder_when_added_then_divided :
  (num1 + num2) % divisor = 5 := 
sorry

end remainder_when_added_then_divided

namespace sum_distinct_prime_factors_of_expr

import Mathlib

theorem sum_distinct_prime_factors_of_expr : 
  ∑ p in {2, 3, 7}, p = 12 :=
by
  -- The proof will be written here.
  sorry

end sum_distinct_prime_factors_of_expr

namespace relay_race_total_time

import Mathlib

theorem relay_race_total_time :
  let t1 := 55
  let t2 := t1 + 0.25 * t1
  let t3 := t2 - 0.20 * t2
  let t4 := t1 + 0.30 * t1
  let t5 := 80
  let t6 := t5 - 0.20 * t5
  let t7 := t5 + 0.15 * t5
  let t8 := t7 - 0.05 * t7
  t1 + t2 + t3 + t4 + t5 + t6 + t7 + t8 = 573.65 :=
by
  sorry

end relay_race_total_time

namespace range_of_a

import Mathlib

theorem range_of_a (a : ℝ) : 
  (∀ P Q : ℝ × ℝ, P ≠ Q ∧ P.snd = a * P.fst ^ 2 - 1 ∧ Q.snd = a * Q.fst ^ 2 - 1 ∧ 
  P.fst + P.snd = -(Q.fst + Q.snd)) →
  a > 3 / 4 :=
by
  sorry

end range_of_a

namespace total_marbles_proof

import Mathlib 

def red_marble_condition (b r : ℕ) : Prop :=
  r = b + (3 * b / 10)

def yellow_marble_condition (r y : ℕ) : Prop :=
  y = r + (5 * r / 10)

def total_marbles (b r y : ℕ) : ℕ :=
  r + b + y

theorem total_marbles_proof (b r y : ℕ)
  (h1 : red_marble_condition b r)
  (h2 : yellow_marble_condition r y) :
  total_marbles b r y = 425 * r / 130 :=
by {
  sorry
}

end total_marbles_proof

namespace circles_intersect

import Mathlib

-- Definition of the first circle
def C1 (x y : ℝ) : Prop := x^2 + y^2 + 2*x + 8*y - 8 = 0

-- Definition of the second circle
def C2 (x y : ℝ) : Prop := x^2 + y^2 - 4*x + 4*y - 8 = 0

-- Proving that the circles defined by C1 and C2 intersect
theorem circles_intersect : ∃ (x y : ℝ), C1 x y ∧ C2 x y :=
by sorry

end circles_intersect

namespace find_a_minus_b

import Mathlib

theorem find_a_minus_b (a b : ℤ) 
  (h1 : 3015 * a + 3019 * b = 3023) 
  (h2 : 3017 * a + 3021 * b = 3025) : 
  a - b = -3 := 
sorry

end find_a_minus_b

namespace measure_angle_WYZ

import Mathlib

def angle_XYZ : ℝ := 45
def angle_XYW : ℝ := 15

theorem measure_angle_WYZ : angle_XYZ - angle_XYW = 30 := by
  sorry

end measure_angle_WYZ

namespace probability_heads_exactly_8_in_10

import Mathlib

def fair_coin_probability (n k : ℕ) : ℚ := (Nat.choose n k : ℚ) / (2 ^ n)

theorem probability_heads_exactly_8_in_10 :
  fair_coin_probability 10 8 = 45 / 1024 :=
by 
  sorry

end probability_heads_exactly_8_in_10

namespace correct_option_B

import Mathlib

-- Given conditions
variable (f : ℝ → ℝ)
variable (h_even : ∀ x : ℝ, f x = f (-x))
variable (h_mono_inc : ∀ (a b : ℝ), 0 ≤ a ∧ a ≤ b → f a ≤ f b)

-- Theorem statement
theorem correct_option_B : f (-2) > f (-1) ∧ f (-1) > f (0) :=
by
  sorry

end correct_option_B

namespace find_perpendicular_slope_value

import Mathlib

theorem find_perpendicular_slope_value (a : ℝ) (h : a * (a + 2) = -1) : a = -1 := 
  sorry

end find_perpendicular_slope_value

namespace valid_three_digit_numbers_count

import Mathlib

def is_prime_or_even (d : ℕ) : Prop :=
  d = 2 ∨ d = 3 ∨ d = 5 ∨ d = 7

noncomputable def count_valid_numbers : ℕ :=
  (4 * 4) -- number of valid combinations for hundreds and tens digits

theorem valid_three_digit_numbers_count : count_valid_numbers = 16 :=
by 
  -- outline the structure of the proof here, but we use sorry to indicate the proof is not complete
  sorry

end valid_three_digit_numbers_count

namespace arithmetic_mean_of_two_digit_multiples_of_5

import Mathlib

theorem arithmetic_mean_of_two_digit_multiples_of_5:
  let smallest := 10
  let largest := 95
  let num_terms := 18
  let sum := 945
  let mean := (sum : ℝ) / (num_terms : ℝ)
  mean = 52.5 :=
by
  sorry

end arithmetic_mean_of_two_digit_multiples_of_5

namespace min_value_inequality_equality_condition

import Mathlib

theorem min_value_inequality (a b : ℝ) (ha : 1 < a) (hb : 1 < b) :
  (b^2 / (a - 1) + a^2 / (b - 1)) ≥ 8 :=
sorry

theorem equality_condition (a b : ℝ) (ha : 1 < a) (hb : 1 < b) :
  (b^2 / (a - 1) + a^2 / (b - 1) = 8) ↔ ((a = 2) ∧ (b = 2)) :=
sorry

end min_value_inequality_equality_condition

namespace problem1_line_equation_problem2_circle_equation

import Mathlib

-- Problem 1: Equation of a specific line
def line_intersection (x y : ℝ) : Prop := 
  2 * x + y - 8 = 0 ∧ x - 2 * y + 1 = 0

def line_perpendicular (x y : ℝ) : Prop :=
  6 * x - 8 * y + 3 = 0

noncomputable def find_line (x y : ℝ) : Prop :=
  ∃ (l : ℝ), (8 * x + 6 * y + l = 0) ∧ 
  line_intersection x y ∧ line_perpendicular x y

theorem problem1_line_equation : ∃ (x y : ℝ), find_line x y :=
sorry

-- Problem 2: Equation of a specific circle
def point_A (x y : ℝ) : Prop := 
  x = 5 ∧ y = 2

def point_B (x y : ℝ) : Prop := 
  x = 3 ∧ y = -2

def center_on_line (x y : ℝ) : Prop :=
  2 * x - y = 3

noncomputable def find_circle (x y r : ℝ) : Prop :=
  ((x - 2)^2 + (y - 1)^2 = r) ∧
  ∃ x1 y1 x2 y2, point_A x1 y1 ∧ point_B x2 y2 ∧ center_on_line x y ∧ ((x1 - x)^2 + (y1 - y)^2 = r)

theorem problem2_circle_equation : ∃ (x y r : ℝ), find_circle x y 10 :=
sorry

end problem1_line_equation_problem2_circle_equation

namespace hexagonal_prism_sum_maximum

import Mathlib

noncomputable def hexagonal_prism_max_sum (h_u h_v h_w h_x h_y h_z : ℕ) (u v w x y z : ℝ) : ℝ :=
  u + v + w + x + y + z

def max_sum_possible (h_u h_v h_w h_x h_y h_z : ℕ) : ℝ :=
  if h_u = 4 ∧ h_v = 7 ∧ h_w = 10 ∨
     h_u = 4 ∧ h_x = 7 ∧ h_y = 10 ∨
     h_u = 4 ∧ h_y = 7 ∧ h_z = 10 ∨
     h_v = 4 ∧ h_x = 7 ∧ h_w = 10 ∨
     h_v = 4 ∧ h_y = 7 ∧ h_z = 10 ∨
     h_w = 4 ∧ h_x = 7 ∧ h_z = 10
  then 78
  else 0

theorem hexagonal_prism_sum_maximum (h_u h_v h_w h_x h_y h_z : ℕ) :
  max_sum_possible h_u h_v h_w h_x h_y h_z = 78 → ∃ (u v w x y z : ℝ), hexagonal_prism_max_sum h_u h_v h_w h_x h_y h_z u v w x y z = 78 := 
by 
  sorry

end hexagonal_prism_sum_maximum

namespace calc_xy_square

import Mathlib

theorem calc_xy_square
  (x y z : ℝ)
  (h1 : 2 * x * (y + z) = 1 + y * z)
  (h2 : 1 / x - 2 / y = 3 / 2)
  (h3 : x + y + 1 / 2 = 0) :
  (x + y + z) ^ 2 = 1 :=
by
  sorry

end calc_xy_square

namespace y_coordinate_sum_of_circle_on_y_axis

import Mathlib

-- Define the properties of the circle
def center := (-3, 1)
def radius := 8

-- Define the equation of the circle
def circle_eq (x y : ℝ) : Prop :=
  (x + 3) ^ 2 + (y - 1) ^ 2 = 64

-- Define the Lean theorem statement
theorem y_coordinate_sum_of_circle_on_y_axis 
  (h₁ : center = (-3, 1)) 
  (h₂ : radius = 8) 
  (h₃ : ∀ y : ℝ, circle_eq 0 y → (∃ y1 y2 : ℝ, y = y1 ∨ y = y2) ) : 
  ∃ y1 y2 : ℝ, (y1 + y2 = 2) ∧ (circle_eq 0 y1) ∧ (circle_eq 0 y2) := 
by 
  sorry

end y_coordinate_sum_of_circle_on_y_axis

namespace find_number

import Mathlib

theorem find_number (n : ℕ) (some_number : ℕ) 
  (h : (1/5 : ℝ)^n * (1/4 : ℝ)^(18 : ℕ) = 1 / (2 * (some_number : ℝ)^n))
  (hn : n = 35) : some_number = 10 := 
by 
  sorry

end find_number

namespace new_tax_rate

import Mathlib

-- Condition definitions
def previous_tax_rate : ℝ := 0.20
def initial_income : ℝ := 1000000
def new_income : ℝ := 1500000
def additional_taxes_paid : ℝ := 250000

-- Theorem statement
theorem new_tax_rate : 
  ∃ T : ℝ, 
    (new_income * T = initial_income * previous_tax_rate + additional_taxes_paid) ∧ 
    T = 0.30 :=
by sorry

end new_tax_rate

namespace sum_of_areas_of_disks

import Mathlib

theorem sum_of_areas_of_disks (r : ℝ) (a b c : ℕ) (h : a + b + c = 123) :
  ∃ (r : ℝ), (15 * Real.pi * r^2 = Real.pi * ((105 / 4) - 15 * Real.sqrt 3) ∧ r = 1 - (Real.sqrt 3) / 2) := 
by
  sorry

end sum_of_areas_of_disks

namespace find_exponent

import Mathlib

theorem find_exponent 
  (h1 : (1 : ℝ) / 9 = 3 ^ (-2 : ℝ))
  (h2 : (3 ^ (20 : ℝ) : ℝ) / 9 = 3 ^ x) : 
  x = 18 :=
by sorry

end find_exponent

namespace largest_increase_between_2006_and_2007

import Mathlib

-- Define the number of students taking the AMC in each year
def students_2002 := 50
def students_2003 := 55
def students_2004 := 63
def students_2005 := 70
def students_2006 := 75
def students_2007_AMC10 := 90
def students_2007_AMC12 := 15

-- Define the total number of students participating in any AMC contest each year
def total_students_2002 := students_2002
def total_students_2003 := students_2003
def total_students_2004 := students_2004
def total_students_2005 := students_2005
def total_students_2006 := students_2006
def total_students_2007 := students_2007_AMC10 + students_2007_AMC12

-- Function to calculate percentage increase
def percentage_increase (old new : ℕ) : ℚ :=
  ((new - old : ℕ) : ℚ) / old * 100

-- Calculate percentage increases between the years
def inc_2002_2003 := percentage_increase total_students_2002 total_students_2003
def inc_2003_2004 := percentage_increase total_students_2003 total_students_2004
def inc_2004_2005 := percentage_increase total_students_2004 total_students_2005
def inc_2005_2006 := percentage_increase total_students_2005 total_students_2006
def inc_2006_2007 := percentage_increase total_students_2006 total_students_2007

-- Prove that the largest percentage increase is between 2006 and 2007
theorem largest_increase_between_2006_and_2007 :
  inc_2006_2007 > inc_2005_2006 ∧
  inc_2006_2007 > inc_2004_2005 ∧
  inc_2006_2007 > inc_2003_2004 ∧
  inc_2006_2007 > inc_2002_2003 := 
by {
  sorry
}

end largest_increase_between_2006_and_2007

namespace john_weight_end

import Mathlib

def initial_weight : ℝ := 220
def loss_percentage : ℝ := 0.1
def weight_loss : ℝ := loss_percentage * initial_weight
def weight_gain_back : ℝ := 2
def net_weight_loss : ℝ := weight_loss - weight_gain_back
def final_weight : ℝ := initial_weight - net_weight_loss

theorem john_weight_end :
  final_weight = 200 := 
by 
  sorry

end john_weight_end

namespace percentage_of_second_solution_is_16point67

import Mathlib

open Real

def percentage_second_solution (x : ℝ) : Prop :=
  let v₁ := 50
  let c₁ := 0.10
  let c₂ := x / 100
  let v₂ := 200 - v₁
  let c_final := 0.15
  let v_final := 200
  (c₁ * v₁) + (c₂ * v₂) = (c_final * v_final)

theorem percentage_of_second_solution_is_16point67 :
  ∃ x, percentage_second_solution x ∧ x = (50/3) :=
sorry

end percentage_of_second_solution_is_16point67

namespace vanya_exam_scores

import Mathlib

/-- Vanya's exam scores inequality problem -/
theorem vanya_exam_scores
  (M R P : ℕ) -- scores in Mathematics, Russian language, and Physics respectively
  (hR : R = M - 10)
  (hP : P = M - 7)
  (h_bound : ∀ (k : ℕ), M + k ≤ 100 ∧ P + k ≤ 100 ∧ R + k ≤ 100) :
  ¬ (M = 100 ∧ P = 100) ∧ ¬ (M = 100 ∧ R = 100) ∧ ¬ (P = 100 ∧ R = 100) :=
by {
  sorry
}

end vanya_exam_scores

namespace max_value_m

import Mathlib

variable {a b m : ℝ}

theorem max_value_m (ha : a > 0) (hb : b > 0) 
  (h : ∀ a b, (3 / a) + (1 / b) ≥ m / (a + 3 * b)) : m ≤ 12 :=
by 
  sorry

end max_value_m

namespace total_cans_collected

import Mathlib

variable (bags_saturday : ℕ) (bags_sunday : ℕ) (cans_per_bag : ℕ)

def total_bags : ℕ := bags_saturday + bags_sunday

theorem total_cans_collected 
  (h_sat : bags_saturday = 5)
  (h_sun : bags_sunday = 3)
  (h_cans : cans_per_bag = 5) : 
  total_bags bags_saturday bags_sunday * cans_per_bag = 40 :=
by
  sorry

end total_cans_collected

namespace line_y_intercept

import Mathlib

theorem line_y_intercept (x1 y1 x2 y2 : ℝ) (h1 : (x1, y1) = (2, 9)) (h2 : (x2, y2) = (5, 21)) :
    ∃ b : ℝ, (∀ x : ℝ, y = 4 * x + b) ∧ (b = 1) :=
by
  use 1
  sorry

end line_y_intercept

namespace number_of_terms_in_arithmetic_sequence

import Mathlib

-- Define the necessary conditions
def a := 2
def d := 5
def l := 1007  -- last term

-- Prove the number of terms in the sequence
theorem number_of_terms_in_arithmetic_sequence : 
  ∃ n : ℕ, l = a + (n - 1) * d ∧ n = 202 :=
by
  sorry

end number_of_terms_in_arithmetic_sequence

namespace units_digit_3m_squared_plus_2m

import Mathlib

def m : ℕ := 2017^2 + 2^2017

theorem units_digit_3m_squared_plus_2m : (3 * (m^2 + 2^m)) % 10 = 9 := by
  sorry

end units_digit_3m_squared_plus_2m

namespace how_many_years_older

import Mathlib

-- Definitions of the conditions
variables (a b c : ℕ)
def b_is_16 : Prop := b = 16
def b_is_twice_c : Prop := b = 2 * c
def sum_is_42 : Prop := a + b + c = 42

-- Statement of the proof problem
theorem how_many_years_older (h1 : b_is_16 b) (h2 : b_is_twice_c b c) (h3 : sum_is_42 a b c) : a - b = 2 :=
by
  sorry

end how_many_years_older

namespace deaths_during_operation

import Mathlib

noncomputable def initial_count : ℕ := 1000
noncomputable def first_day_remaining (n : ℕ) := 5 * n / 6
noncomputable def second_day_remaining (n : ℕ) := (35 * n / 48) - 1
noncomputable def third_day_remaining (n : ℕ) := (105 * n / 192) - 3 / 4

theorem deaths_during_operation : ∃ n : ℕ, initial_count - n = 472 ∧ n = 528 :=
  by sorry

end deaths_during_operation

namespace range_of_m

import Mathlib

theorem range_of_m (x : ℝ) (m : ℝ) (hx : 0 < x ∧ x < π) 
  (h : Real.cot (x / 3) = m * Real.cot x): m > 3 ∨ m < 0 :=
sorry

end range_of_m

namespace sum_base9_to_base9_eq

import Mathlib

-- Definition of base 9 numbers
def base9_to_base10 (n : ℕ) : ℕ :=
  let digit1 := n % 10
  let digit2 := (n / 10) % 10
  let digit3 := (n / 100) % 10
  digit1 + 9 * digit2 + 81 * digit3

-- Definition of base 10 to base 9 conversion
def base10_to_base9 (n : ℕ) : ℕ :=
  let digit1 := n % 9
  let digit2 := (n / 9) % 9
  let digit3 := (n / 81) % 9
  digit1 + 10 * digit2 + 100 * digit3

-- The theorem to prove
theorem sum_base9_to_base9_eq :
  let x := base9_to_base10 236
  let y := base9_to_base10 327
  let z := base9_to_base10 284
  base10_to_base9 (x + y + z) = 858 :=
by {
  sorry
}

end sum_base9_to_base9_eq

namespace period_sine_transformed

import Mathlib

theorem period_sine_transformed (x : ℝ) : 
  let y := 3 * Real.sin ((x / 3) + (Real.pi / 4))
  ∃ p : ℝ, (∀ x : ℝ, y = 3 * Real.sin ((x + p) / 3 + (Real.pi / 4)) ↔ y = 3 * Real.sin ((x / 3) + (Real.pi / 4))) ∧ p = 6 * Real.pi :=
sorry

end period_sine_transformed

namespace arccos_neg_one_eq_pi

import Mathlib

theorem arccos_neg_one_eq_pi : Real.arccos (-1) = Real.pi :=
by
  sorry

end arccos_neg_one_eq_pi

namespace two_numbers_equal

import Mathlib

variables {a b c : ℝ}
variable (h1 : a + b^2 + c^2 = a^2 + b + c^2)
variable (h2 : a^2 + b + c^2 = a^2 + b^2 + c)

theorem two_numbers_equal (h1 : a + b^2 + c^2 = a^2 + b + c^2) (h2 : a^2 + b + c^2 = a^2 + b^2 + c) :
  a = b ∨ a = c ∨ b = c :=
by
  sorry

end two_numbers_equal

namespace CarmenBrushLengthInCentimeters

import Mathlib

-- Given conditions
def CarlaBrushLengthInInches : ℝ := 12
def CarmenBrushPercentIncrease : ℝ := 0.5
def InchToCentimeterConversionFactor : ℝ := 2.5

-- Question: What is Carmen's brush length in centimeters?
-- Proof Goal: Prove that Carmen's brush length in centimeters is 45 cm.
theorem CarmenBrushLengthInCentimeters :
  let CarmenBrushLengthInInches := CarlaBrushLengthInInches * (1 + CarmenBrushPercentIncrease)
  CarmenBrushLengthInInches * InchToCentimeterConversionFactor = 45 := by
  -- sorry is used as a placeholder for the completed proof
  sorry

end CarmenBrushLengthInCentimeters

namespace integer_a_for_factoring

import Mathlib

theorem integer_a_for_factoring (a : ℤ) :
  (∃ c d : ℤ, (x - a) * (x - 10) + 1 = (x + c) * (x + d)) → (a = 8 ∨ a = 12) :=
by
  sorry

end integer_a_for_factoring

namespace relationship_among_abc

import Mathlib

noncomputable
def a := 0.2 ^ 1.5

noncomputable
def b := 2 ^ 0.1

noncomputable
def c := 0.2 ^ 1.3

theorem relationship_among_abc : a < c ∧ c < b := by
  sorry

end relationship_among_abc

namespace max_value_m

import Mathlib

noncomputable def quadratic_function (a b c : ℝ) (x : ℝ) := a * x^2 + b * x + c

theorem max_value_m (a b c : ℝ) (h1 : a ≠ 0)
  (h2 : ∀ x : ℝ, quadratic_function a b c (x-4) = quadratic_function a b c (2-x))
  (h3 : ∀ x : ℝ, 0 < x ∧ x < 2 → quadratic_function a b c x ≤ ( (x+1)/2 )^2)
  (h4 : ∀ x : ℝ, quadratic_function a b c x ≥ 0)
  (h_min : ∃ x : ℝ, quadratic_function a b c x = 0) :
  ∃ (m : ℝ), m > 1 ∧ (∃ t : ℝ, ∀ x : ℝ, 1 ≤ x ∧ x ≤ m → quadratic_function a b c (x+t) ≤ x) ∧ m = 9 := 
sorry

end max_value_m

namespace sin_cos_quad_ineq

import Mathlib

open Real

theorem sin_cos_quad_ineq (x : ℝ) : 
  2 * (sin x) ^ 4 + 3 * (sin x) ^ 2 * (cos x) ^ 2 + 5 * (cos x) ^ 4 ≤ 5 :=
by
  sorry

end sin_cos_quad_ineq

namespace smallest_perimeter_even_integer_triangl

import Mathlib

theorem smallest_perimeter_even_integer_triangl (n : ℕ) (h : n > 2) :
  let a := 2 * n - 2
  let b := 2 * n
  let c := 2 * n + 2
  2 * n - 2 + 2 * n > 2 * n + 2 ∧
  2 * n - 2 + 2 * n + 2 > 2 * n ∧
  2 * n + 2 * n + 2 > 2 * n - 2 ∧ 
  2 * 3 - 2 + 2 * 3 + 2 * 3 + 2 = 18 :=
by
  { sorry }

end smallest_perimeter_even_integer_triangl

namespace triangle_is_right_angled

import Mathlib

structure Point where
  x : ℝ
  y : ℝ

def vector (P Q : Point) : Point :=
  { x := Q.x - P.x, y := Q.y - P.y }

def dot_product (u v : Point) : ℝ :=
  u.x * v.x + u.y * v.y

def is_right_angle_triangle (A B C : Point) : Prop :=
  let AB := vector A B
  let BC := vector B C
  dot_product AB BC = 0

theorem triangle_is_right_angled :
  let A := { x := 2, y := 5 }
  let B := { x := 5, y := 2 }
  let C := { x := 10, y := 7 }
  is_right_angle_triangle A B C :=
by
  sorry

end triangle_is_right_angled

namespace black_balls_in_box

import Mathlib

theorem black_balls_in_box (B : ℕ) (probability : ℚ) 
  (h1 : probability = 0.38095238095238093) 
  (h2 : B / (14 + B) = probability) : 
  B = 9 := by
  sorry

end black_balls_in_box

namespace tiling_implies_divisibility

import Mathlib

def is_divisible_by (a b : Nat) : Prop := ∃ k : Nat, a = k * b

noncomputable def can_be_tiled (m n a b : Nat) : Prop :=
  a * b > 0 ∧ -- positivity condition for rectangle dimensions
  (∃ f_horiz : Fin (a * b) → Fin m, 
   ∃ g_vert : Fin (a * b) → Fin n, 
   True) -- A placeholder to denote tiling condition.

theorem tiling_implies_divisibility (m n a b : Nat)
  (hmn_pos : 0 < m ∧ 0 < n ∧ 0 < a ∧ 0 < b)
  (h_tiling : can_be_tiled m n a b) :
  is_divisible_by a m ∨ is_divisible_by b n :=
by
  sorry

end tiling_implies_divisibility

namespace triangle_inequality_from_condition

import Mathlib

theorem triangle_inequality_from_condition (a b c : ℝ)
  (h : (a^2 + b^2 + c^2)^2 > 2 * (a^4 + b^4 + c^4)) :
  a + b > c ∧ a + c > b ∧ b + c > a :=
by 
  sorry

end triangle_inequality_from_condition

namespace find_number_with_10_questions

import Mathlib

theorem find_number_with_10_questions (n : ℕ) (h : n ≤ 1000) : n = 300 :=
by
  sorry

end find_number_with_10_questions

namespace rectangle_perimeter

import Mathlib

-- Conditions
def is_right_triangle (a b c : ℕ) : Prop :=
  a^2 + b^2 = c^2

def area_of_triangle (base height : ℕ) : ℕ :=
  (base * height) / 2

def area_of_rectangle (length width : ℕ) : ℕ :=
  length * width

def perimeter_of_rectangle (length width : ℕ) : ℕ :=
  2 * (length + width)

-- Given conditions from the problem
def a : ℕ := 9
def b : ℕ := 12
def c : ℕ := 15
def width_of_rectangle : ℕ := 6

-- Main theorem
theorem rectangle_perimeter :
  is_right_triangle a b c →
  area_of_triangle a b = area_of_rectangle (area_of_triangle a b / width_of_rectangle) width_of_rectangle →
  perimeter_of_rectangle (area_of_triangle a b / width_of_rectangle) width_of_rectangle = 30 :=
by
  sorry

end rectangle_perimeter

namespace beef_weight_after_processing

import Mathlib

def original_weight : ℝ := 861.54
def weight_loss_percentage : ℝ := 0.35
def retained_percentage : ℝ := 1 - weight_loss_percentage
def weight_after_processing (w : ℝ) := retained_percentage * w

theorem beef_weight_after_processing :
  weight_after_processing original_weight = 560.001 :=
by
  sorry

end beef_weight_after_processing

namespace valid_base6_number_2015

import Mathlib

def is_valid_base6_digit (d : Nat) : Prop :=
  d = 0 ∨ d = 1 ∨ d = 2 ∨ d = 3 ∨ d = 4 ∨ d = 5

def is_base6_number (n : Nat) : Prop :=
  ∀ (digit : Nat), digit ∈ (n.digits 10) → is_valid_base6_digit digit

theorem valid_base6_number_2015 : is_base6_number 2015 := by
  sorry

end valid_base6_number_2015

namespace quadratic_one_real_root_positive_m

import Mathlib

theorem quadratic_one_real_root_positive_m (m : ℝ) (h : m > 0) :
  (∀ x : ℝ, x^2 + 6 * m * x + 2 * m = 0 → ((6 * m)^2 - 4 * 1 * (2 * m) = 0)) → m = 2 / 9 :=
by
  sorry

end quadratic_one_real_root_positive_m

namespace largest_sum_is_5_over_6

import Mathlib

def sum_1 := (1/3) + (1/7)
def sum_2 := (1/3) + (1/8)
def sum_3 := (1/3) + (1/2)
def sum_4 := (1/3) + (1/9)
def sum_5 := (1/3) + (1/4)

theorem largest_sum_is_5_over_6 : (sum_3 = 5/6) ∧ ((sum_3 > sum_1) ∧ (sum_3 > sum_2) ∧ (sum_3 > sum_4) ∧ (sum_3 > sum_5)) :=
by
  sorry

end largest_sum_is_5_over_6

namespace axis_of_symmetry_parabola

import Mathlib

theorem axis_of_symmetry_parabola : 
  ∀ (x : ℝ), 2 * (x - 3)^2 - 5 = 2 * (x - 3)^2 - 5 → (∃ h : ℝ, h = 3 ∧ ∀ x : ℝ, h = 3) :=
by
  sorry

end axis_of_symmetry_parabola

namespace problem

import Mathlib

theorem problem (k : ℕ) (hk : 0 < k) (n : ℕ) : 
  (∃ p : ℕ, n = 2 * 3 ^ (k - 1) * p ∧ 0 < p) ↔ 3^k ∣ (2^n - 1) := 
by 
  sorry

end problem

namespace problem_1_solution_problem_2_solution_problem_3_solution_problem_4_solution

import Mathlib

noncomputable def problem_1 : Int :=
  (-3) + 5 - (-3)

theorem problem_1_solution : problem_1 = 5 := by
  sorry

noncomputable def problem_2 : ℚ :=
  (-1/3 - 3/4 + 5/6) * (-24)

theorem problem_2_solution : problem_2 = 6 := by
  sorry

noncomputable def problem_3 : ℚ :=
  1 - (1/9) * (-1/2 - 2^2)

theorem problem_3_solution : problem_3 = 3/2 := by
  sorry

noncomputable def problem_4 : ℚ :=
  ((-1)^2023) * (18 - (-2) * 3) / (15 - 3^3)

theorem problem_4_solution : problem_4 = 2 := by
  sorry

end problem_1_solution_problem_2_solution_problem_3_solution_problem_4_solution

namespace symm_diff_A_B

import Mathlib

-- Define sets A and B
def A : Set ℤ := {1, 2}
def B : Set ℤ := {x : ℤ | abs x < 2}

-- Define set difference
def set_diff (S T : Set ℤ) : Set ℤ := {x | x ∈ S ∧ x ∉ T}

-- Define symmetric difference
def symm_diff (S T : Set ℤ) : Set ℤ := (set_diff S T) ∪ (set_diff T S)

-- Define the expression we need to prove
theorem symm_diff_A_B : symm_diff A B = {-1, 0, 2} := by
  sorry

end symm_diff_A_B

namespace least_n_divisibility_condition

import Mathlib

theorem least_n_divisibility_condition :
  ∃ n : ℕ, 0 < n ∧ ∀ k : ℕ, 1 ≤ k ∧ k ≤ n → (k ∣ (n^2 - n + 1) ↔ (n = 5 ∧ k = 3)) := 
sorry

end least_n_divisibility_condition

namespace train_length_is_correct

import Mathlib
import Mathlib.Data.Real.Basic

noncomputable def train_speed_kmh : ℝ := 40
noncomputable def train_speed_ms : ℝ := train_speed_kmh * (5 / 18)
noncomputable def cross_time : ℝ := 25.2
noncomputable def train_length : ℝ := train_speed_ms * cross_time

theorem train_length_is_correct : train_length = 280.392 := by
  sorry

end train_length_is_correct

namespace collinear_vectors_x_eq_neg_two

import Mathlib

theorem collinear_vectors_x_eq_neg_two (x : ℝ) (a b : ℝ×ℝ) :
  a = (1, 2) → b = (x, -4) → a.1 * b.2 = a.2 * b.1 → x = -2 :=
by
  intro ha hb hc
  sorry

end collinear_vectors_x_eq_neg_two

namespace reduction_amount_is_250

import Mathlib

-- Definitions from the conditions
def original_price : ℝ := 500
def reduction_rate : ℝ := 0.5

-- The statement to be proved
theorem reduction_amount_is_250 : (reduction_rate * original_price) = 250 := by
  sorry

end reduction_amount_is_250

namespace smallest_number_diminished_by_16_divisible

import Mathlib

theorem smallest_number_diminished_by_16_divisible (n : ℕ) :
  (∃ n, ∀ k ∈ [4, 6, 8, 10], (n - 16) % k = 0 ∧ n = 136) :=
by
  sorry

end smallest_number_diminished_by_16_divisible

namespace find_value_of_a

import Mathlib

variables (a : ℚ)

-- Definitions based on the conditions
def Brian_has_mar_bles : ℚ := 3 * a
def Caden_original_mar_bles : ℚ := 4 * Brian_has_mar_bles a
def Daryl_original_mar_bles : ℚ := 2 * Caden_original_mar_bles a
def Caden_after_give_10 : ℚ := Caden_original_mar_bles a - 10
def Daryl_after_receive_10 : ℚ := Daryl_original_mar_bles a + 10

-- Together Caden and Daryl now have 190 marbles
def together_mar_bles : ℚ := Caden_after_give_10 a + Daryl_after_receive_10 a

theorem find_value_of_a : together_mar_bles a = 190 → a = 95 / 18 :=
by
  sorry

end find_value_of_a

namespace tetrahedron_volume

import Mathlib

variable {R : ℝ}
variable {S1 S2 S3 S4 : ℝ}
variable {V : ℝ}

theorem tetrahedron_volume (R : ℝ) (S1 S2 S3 S4 V : ℝ) :
  V = (1 / 3) * R * (S1 + S2 + S3 + S4) :=
sorry

end tetrahedron_volume

namespace find_k_max_product

import Mathlib

theorem find_k_max_product : 
  (∃ k : ℝ, (3 : ℝ) * (x ^ 2) - 4 * x + k = 0 ∧ 16 - 12 * k ≥ 0 ∧ (∀ x1 x2 : ℝ, x1 * x2 = k / 3 → x1 + x2 = 4 / 3 → x1 * x2 ≤ (2 / 3) ^ 2)) →
  k = 4 / 3 :=
by 
  sorry

end find_k_max_product

namespace median_possible_values

import Mathlib

theorem median_possible_values (S : Finset ℤ)
  (h : S.card = 10)
  (h_contains : {5, 7, 12, 15, 18, 21} ⊆ S) :
  ∃! n : ℕ, n = 5 :=
by
   sorry

end median_possible_values

namespace diamond_19_98

import Mathlib

variable {R : Type} [LinearOrderedField R]

noncomputable def diamond (x y : R) : R := sorry

axiom diamond_axiom1 : ∀ (x y : R) (hx : 0 < x) (hy : 0 < y), diamond (x * y) y = x * (diamond y y)

axiom diamond_axiom2 : ∀ (x : R) (hx : 0 < x), diamond (diamond x 1) x = diamond x 1

axiom diamond_axiom3 : diamond 1 1 = 1

theorem diamond_19_98 : diamond (19 : R) (98 : R) = 19 := 
sorry

end diamond_19_98

namespace negation_of_p

import Mathlib

-- Define the proposition p
def p : Prop := ∃ n : ℕ, 2^n > 100

-- Goal is to show the negation of p
theorem negation_of_p : (¬ p) = (∀ n : ℕ, 2^n ≤ 100) :=
by
  sorry

end negation_of_p

namespace loss_equals_cost_price_of_some_balls

import Mathlib

-- Conditions
def cost_price_per_ball := 60
def selling_price_for_17_balls := 720
def number_of_balls := 17

-- Calculations
def total_cost_price := number_of_balls * cost_price_per_ball
def loss := total_cost_price - selling_price_for_17_balls

-- Proof statement
theorem loss_equals_cost_price_of_some_balls : (loss / cost_price_per_ball) = 5 :=
by
  -- Proof would go here
  sorry

end loss_equals_cost_price_of_some_balls

namespace num_fixed_last_two_digits

import Mathlib

theorem num_fixed_last_two_digits : 
  ∃ c : ℕ, c = 36 ∧ ∀ (a : ℕ), 2 ≤ a ∧ a ≤ 101 → 
    (∃ N : ℕ, ∀ n : ℕ, n ≥ N → (a^(2^n) % 100 = a^(2^N) % 100)) ↔ (a = c ∨ c ≠ 36) :=
sorry

end num_fixed_last_two_digits

namespace danny_distance_to_work

import Mathlib -- Importing Mathlib to avoid import errors

-- Define the conditions and the problem in terms of Lean definitions
def distance_to_first_friend : ℕ := 8
def distance_to_second_friend : ℕ := distance_to_first_friend / 2
def total_distance_driven_so_far : ℕ := distance_to_first_friend + distance_to_second_friend
def distance_to_work : ℕ := 3 * total_distance_driven_so_far

-- Lean statement to be proven
theorem danny_distance_to_work :
  distance_to_work = 36 :=
by
  -- This is the proof placeholder
  sorry

end danny_distance_to_work

namespace distance_between_polar_points

import Mathlib

noncomputable def polar_to_rect (r θ : ℝ) : ℝ × ℝ :=
  (r * Real.cos θ, r * Real.sin θ)

noncomputable def distance (A B : ℝ × ℝ) : ℝ :=
  Real.sqrt ((B.1 - A.1) ^ 2 + (B.2 - A.2) ^ 2)

theorem distance_between_polar_points :
  let A := polar_to_rect 1 (Real.pi / 6)
  let B := polar_to_rect 2 (-Real.pi / 2)
  distance A B = Real.sqrt 7 :=
by
  sorry

end distance_between_polar_points

namespace find_swimming_speed

import Mathlib

variable (S : ℝ)

def is_average_speed (x y avg : ℝ) : Prop :=
  avg = 2 * x * y / (x + y)

theorem find_swimming_speed
  (running_speed : ℝ := 7)
  (average_speed : ℝ := 4)
  (h : is_average_speed S running_speed average_speed) :
  S = 2.8 :=
by sorry

end find_swimming_speed

namespace fourth_person_height_is_82

import Mathlib

theorem fourth_person_height_is_82 (H : ℕ)
    (h1: (H + (H + 2) + (H + 4) + (H + 10)) / 4 = 76)
    (h_diff1: H + 2 - H = 2)
    (h_diff2: H + 4 - (H + 2) = 2)
    (h_diff3: H + 10 - (H + 4) = 6) :
  (H + 10) = 82 := 
sorry

end fourth_person_height_is_82

namespace intersection_of_lines

import Mathlib

theorem intersection_of_lines : 
  let x := (5 : ℚ) / 9
  let y := (5 : ℚ) / 3
  (y = 3 * x ∧ y - 5 = -6 * x) ↔ (x, y) = ((5 : ℚ) / 9, (5 : ℚ) / 3) := 
by 
  sorry

end intersection_of_lines

namespace ceil_sum_sqrt_eval

import Mathlib

theorem ceil_sum_sqrt_eval : 
  (⌈Real.sqrt 2⌉ + ⌈Real.sqrt 22⌉ + ⌈Real.sqrt 222⌉) = 22 := 
by
  sorry

end ceil_sum_sqrt_eval

namespace total_amount_shared

import Mathlib

-- conditions as definitions
def Parker_share : ℕ := 50
def ratio_part_Parker : ℕ := 2
def ratio_total_parts : ℕ := 2 + 3 + 4
def value_of_one_part : ℕ := Parker_share / ratio_part_Parker

-- question translated to Lean statement with expected correct answer
theorem total_amount_shared : ratio_total_parts * value_of_one_part = 225 := by
  sorry

end total_amount_shared

namespace digits_subtraction_eq_zero

import Mathlib

theorem digits_subtraction_eq_zero (d A B : ℕ) (h1 : d > 8)
  (h2 : A < d) (h3 : B < d)
  (h4 : A * d + B + A * d + A = 2 * d + 3 * d + 4) :
  A - B = 0 :=
by sorry

end digits_subtraction_eq_zero

namespace total_sum_vowels

import Mathlib

theorem total_sum_vowels :
  let A := 3
  let E := 5
  let I := 4
  let O := 2
  let U := 6
  A + E + I + O + U = 20 := by
  let A := 3
  let E := 5
  let I := 4
  let O := 2
  let U := 6
  sorry

end total_sum_vowels

namespace robotics_club_neither

import Mathlib

theorem robotics_club_neither (n c e b neither : ℕ) (h1 : n = 80) (h2 : c = 50) (h3 : e = 40) (h4 : b = 25) :
  neither = n - (c - b + e - b + b) :=
by 
  rw [h1, h2, h3, h4]
  sorry

end robotics_club_neither

namespace age_of_other_man_replaced

import Mathlib

-- Define the conditions
variables (A : ℝ) (x : ℝ)
variable (average_age_women : ℝ := 50)
variable (num_men : ℕ := 10)
variable (increase_age : ℝ := 6)
variable (one_man_age : ℝ := 22)

-- State the theorem to be proved
theorem age_of_other_man_replaced :
  2 * average_age_women - (one_man_age + x) = 10 * (A + increase_age) - 10 * A →
  x = 18 :=
by
  sorry

end age_of_other_man_replaced

namespace expenses_opposite_to_income

import Mathlib

theorem expenses_opposite_to_income (income_5 : ℤ) (h_income : income_5 = 5) : -income_5 = -5 :=
by
  -- proof is omitted
  sorry

end expenses_opposite_to_income

namespace James_balloons_correct

import Mathlib

def Amy_balloons : ℕ := 101
def diff_balloons : ℕ := 131
def James_balloons (a : ℕ) (d : ℕ) : ℕ := a + d

theorem James_balloons_correct : James_balloons Amy_balloons diff_balloons = 232 :=
by
  sorry

end James_balloons_correct

namespace polynomial_horner_value

import Mathlib

def f (x : ℤ) : ℤ :=
  7 * x^7 + 6 * x^6 + 5 * x^5 + 4 * x^4 + 3 * x^3 + 2 * x^2 + x

def horner (x : ℤ) : ℤ :=
  ((((((7 * x + 6) * x + 5) * x + 4) * x + 3) * x + 2) * x + 1)

theorem polynomial_horner_value :
  horner 3 = 262 := by
  sorry

end polynomial_horner_value

namespace simple_annual_interest_rate

import Mathlib

noncomputable def monthly_interest_payment : ℝ := 216
noncomputable def principal_amount : ℝ := 28800
noncomputable def number_of_months_in_a_year : ℕ := 12

theorem simple_annual_interest_rate :
  ((monthly_interest_payment * number_of_months_in_a_year) / principal_amount) * 100 = 9 := by
sorry

end simple_annual_interest_rate

namespace time_for_Dawson

import Mathlib

variable (D : ℝ)
variable (Henry_time : ℝ := 7)
variable (avg_time : ℝ := 22.5)

theorem time_for_Dawson (h : avg_time = (D + Henry_time) / 2) : D = 38 := 
by 
  sorry

end time_for_Dawson

namespace find_x

import Mathlib

theorem find_x (x : ℤ) (h : (4 + x) / (6 + x) = (2 + x) / (3 + x)) : x = 0 :=
by
  sorry

end find_x

namespace total_pages

import Mathlib

theorem total_pages (x : ℕ) (h : 9 + 180 + 3 * (x - 99) = 1392) : x = 500 :=
by
  sorry

end total_pages

namespace bamboo_break_height

import Mathlib

-- Conditions provided in the problem
def original_height : ℝ := 20  -- 20 chi
def distance_tip_to_root : ℝ := 6  -- 6 chi

-- Function to check if the height of the break satisfies the equation
def equationHolds (x : ℝ) : Prop :=
  (original_height - x) ^ 2 - x ^ 2 = distance_tip_to_root ^ 2

-- Main statement to prove the height of the break is 9.1 chi
theorem bamboo_break_height : equationHolds 9.1 :=
by
  sorry

end bamboo_break_height

namespace Trent_tears

import Mathlib

def onions_per_pot := 4
def pots_of_soup := 6
def tears_per_3_onions := 2

theorem Trent_tears:
  (onions_per_pot * pots_of_soup) / 3 * tears_per_3_onions = 16 :=
by
  sorry

end Trent_tears

namespace absolute_difference_probability

import Mathlib

-- Define the conditions
def num_red_marbles : ℕ := 1500
def num_black_marbles : ℕ := 2000
def total_marbles : ℕ := num_red_marbles + num_black_marbles

def P_s : ℚ :=
  let ways_to_choose_2_red := (num_red_marbles * (num_red_marbles - 1)) / 2
  let ways_to_choose_2_black := (num_black_marbles * (num_black_marbles - 1)) / 2
  let total_favorable_outcomes := ways_to_choose_2_red + ways_to_choose_2_black
  total_favorable_outcomes / (total_marbles * (total_marbles - 1) / 2)

def P_d : ℚ :=
  (num_red_marbles * num_black_marbles) / (total_marbles * (total_marbles - 1) / 2)

-- Prove the statement
theorem absolute_difference_probability : |P_s - P_d| = 1 / 50 := by
  sorry

end absolute_difference_probability

namespace relationship_m_n

import Mathlib

variable (a b : ℝ)
variable (m n : ℝ)

theorem relationship_m_n (h1 : a > b) (h2 : b > 0) (hm : m = Real.sqrt a - Real.sqrt b) (hn : n = Real.sqrt (a - b)) : m < n := sorry

end relationship_m_n

namespace food_cost_max

import Mathlib

theorem food_cost_max (x : ℝ) (total_cost : ℝ) (tax_rate : ℝ) (tip_rate : ℝ) (max_total : ℝ) (food_cost_max : ℝ) :
  total_cost = x * (1 + tax_rate + tip_rate) →
  tax_rate = 0.07 →
  tip_rate = 0.15 →
  max_total = 50 →
  total_cost ≤ max_total →
  food_cost_max = 50 / 1.22 →
  x ≤ food_cost_max :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end food_cost_max

namespace Emily_used_10_dimes

import Mathlib

theorem Emily_used_10_dimes
  (p n d : ℕ)
  (h1 : p + n + d = 50)
  (h2 : p + 5 * n + 10 * d = 200) :
  d = 10 := by
  sorry

end Emily_used_10_dimes

namespace sufficient_condition_for_ellipse

import Mathlib

theorem sufficient_condition_for_ellipse (m : ℝ) (h : m^2 > 5) : m^2 > 4 := by
  sorry

end sufficient_condition_for_ellipse

namespace find_x_in_coconut_grove

import Mathlib

theorem find_x_in_coconut_grove
  (x : ℕ)
  (h1 : (x + 2) * 30 + x * 120 + (x - 2) * 180 = 300 * x)
  (h2 : 3 * x ≠ 0) :
  x = 10 :=
by
  sorry

end find_x_in_coconut_grove

namespace kathleen_money_left

import Mathlib

def june_savings : ℕ := 21
def july_savings : ℕ := 46
def august_savings : ℕ := 45

def school_supplies_expenses : ℕ := 12
def new_clothes_expenses : ℕ := 54

def total_savings : ℕ := june_savings + july_savings + august_savings
def total_expenses : ℕ := school_supplies_expenses + new_clothes_expenses

def total_money_left : ℕ := total_savings - total_expenses

theorem kathleen_money_left : total_money_left = 46 :=
by
  sorry

end kathleen_money_left

namespace average_sale_six_months

import Mathlib

-- Define the sales for the first five months
def sale_month1 : ℕ := 6335
def sale_month2 : ℕ := 6927
def sale_month3 : ℕ := 6855
def sale_month4 : ℕ := 7230
def sale_month5 : ℕ := 6562

-- Define the required sale for the sixth month
def sale_month6 : ℕ := 5091

-- Proof that the desired average sale for the six months is 6500
theorem average_sale_six_months : 
  (sale_month1 + sale_month2 + sale_month3 + sale_month4 + sale_month5 + sale_month6) / 6 = 6500 :=
by
  sorry

end average_sale_six_months

namespace digit_sum_10_pow_93_minus_937

import Mathlib

-- Define a function to compute the sum of digits of a number
def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem digit_sum_10_pow_93_minus_937 :
  sum_of_digits (10^93 - 937) = 819 :=
by
  sorry

end digit_sum_10_pow_93_minus_937

namespace maximum_value_x2_add_3xy_add_y2

import Mathlib

-- Define the conditions
variables {x y : ℝ}

-- State the theorem
theorem maximum_value_x2_add_3xy_add_y2 
  (hx : 0 < x) 
  (hy : 0 < y) 
  (h : 3 * x^2 - 2 * x * y + 5 * y^2 = 12) :
  ∃ e f g h : ℕ,
    x^2 + 3 * x * y + y^2 = (1144 + 204 * Real.sqrt 15) / 91 ∧ e + f + g + h = 1454 :=
sorry

end maximum_value_x2_add_3xy_add_y2

namespace simplify_expression

import Mathlib

variable (m : ℕ) (h1 : m ≠ 2) (h2 : m ≠ 3)

theorem simplify_expression : 
  (m - 3) / (2 * m - 4) / (m + 2 - 5 / (m - 2)) = 1 / (2 * m + 6) :=
by sorry

end simplify_expression

namespace vector_operation_result

import Mathlib

-- Definitions of vectors a and b
def a : ℝ × ℝ := (-1, 1)
def b : ℝ × ℝ := (2, -3)

-- The operation 2a - b
def operation (a b : ℝ × ℝ) : ℝ × ℝ :=
(2 * a.1 - b.1, 2 * a.2 - b.2)

-- The theorem stating the result of the operation
theorem vector_operation_result : operation a b = (-4, 5) :=
by
  sorry

end vector_operation_result

namespace sum_possible_values_q

import Mathlib

/-- If natural numbers k, l, p, and q satisfy the given conditions,
the sum of all possible values of q is 4 --/
theorem sum_possible_values_q (k l p q : ℕ) 
    (h1 : ∀ a b : ℝ, a ≠ b → a * b = l → a + b = k → (∃ (c d : ℝ), c + d = (k * (l + 1)) / l ∧ c * d = (l + 2 + 1 / l))) 
    (h2 : a + 1 / b ≠ b + 1 / a)
    : q = 4 :=
sorry

end sum_possible_values_q

namespace positive_difference_of_squares

import Mathlib

theorem positive_difference_of_squares (a b : ℕ) (h1 : a + b = 40) (h2 : a - b = 8) : a^2 - b^2 = 320 :=
by
  sorry

end positive_difference_of_squares

namespace max_sum_pyramid

import Mathlib

theorem max_sum_pyramid (F_pentagonal : ℕ) (F_rectangular : ℕ) (E_pentagonal : ℕ) (E_rectangular : ℕ) (V_pentagonal : ℕ) (V_rectangular : ℕ)
  (original_faces : ℕ) (original_edges : ℕ) (original_vertices : ℕ)
  (H1 : original_faces = 7)
  (H2 : original_edges = 15)
  (H3 : original_vertices = 10)
  (H4 : F_pentagonal = 11)
  (H5 : E_pentagonal = 20)
  (H6 : V_pentagonal = 11)
  (H7 : F_rectangular = 10)
  (H8 : E_rectangular = 19)
  (H9 : V_rectangular = 11) :
  max (F_pentagonal + E_pentagonal + V_pentagonal) (F_rectangular + E_rectangular + V_rectangular) = 42 :=
by
  sorry

end max_sum_pyramid

namespace students_drawn_in_sample

import Mathlib

def total_people : ℕ := 1600
def number_of_teachers : ℕ := 100
def sample_size : ℕ := 80
def number_of_students : ℕ := total_people - number_of_teachers
def expected_students_sample : ℕ := 75

theorem students_drawn_in_sample : (sample_size * number_of_students) / total_people = expected_students_sample :=
by
  -- The proof steps would go here
  sorry

end students_drawn_in_sample

namespace square_area_from_triangle_perimeter

import Mathlib

noncomputable def perimeter_triangle (a b c : ℝ) : ℝ := a + b + c

noncomputable def side_length_square (perimeter : ℝ) : ℝ := perimeter / 4

noncomputable def area_square (side_length : ℝ) : ℝ := side_length * side_length

theorem square_area_from_triangle_perimeter 
  (a b c : ℝ) 
  (h₁ : a = 5.5) 
  (h₂ : b = 7.5) 
  (h₃ : c = 11) 
  (h₄ : perimeter_triangle a b c = 24) 
  : area_square (side_length_square (perimeter_triangle a b c)) = 36 := 
by 
  simp [perimeter_triangle, side_length_square, area_square, h₁, h₂, h₃, h₄]
  sorry

end square_area_from_triangle_perimeter

namespace cheetah_catches_deer_in_10_minutes

import Mathlib
import Mathlib.Algebra.Group.Defs

noncomputable def deer_speed : ℝ := 50 -- miles per hour
noncomputable def cheetah_speed : ℝ := 60 -- miles per hour
noncomputable def time_difference : ℝ := 2 / 60 -- 2 minutes converted to hours
noncomputable def distance_deer : ℝ := deer_speed * time_difference
noncomputable def speed_difference : ℝ := cheetah_speed - deer_speed
noncomputable def catch_up_time : ℝ := distance_deer / speed_difference

theorem cheetah_catches_deer_in_10_minutes :
  catch_up_time * 60 = 10 :=
by
  sorry

end cheetah_catches_deer_in_10_minutes

namespace roots_of_quadratic

import Mathlib

theorem roots_of_quadratic (α β : ℝ) (h1 : α^2 - 4*α - 5 = 0) (h2 : β^2 - 4*β - 5 = 0) :
  3*α^4 + 10*β^3 = 2593 := 
by
  sorry

end roots_of_quadratic

namespace probability_two_heads_one_tail_in_three_tosses

import Mathlib

theorem probability_two_heads_one_tail_in_three_tosses
(P : ℕ → Prop) (pr : ℤ) : 
  (∀ n, P n → pr = 1 / 2) -> 
  P 3 → pr = 3 / 8 :=
by
  sorry

end probability_two_heads_one_tail_in_three_tosses

namespace required_moles_of_H2O

import Mathlib

-- Definition of the balanced chemical reaction
def balanced_reaction_na_to_naoh_and_H2 : Prop :=
  ∀ (NaH H2O NaOH H2 : ℕ), NaH + H2O = NaOH + H2

-- The given moles of NaH
def moles_NaH : ℕ := 2

-- Assertion that we need to prove: amount of H2O required is 2 moles
theorem required_moles_of_H2O (balanced : balanced_reaction_na_to_naoh_and_H2) : 
  (2 * 1) = 2 :=
by
  sorry

end required_moles_of_H2O

namespace joe_total_cars

import Mathlib

def initial_cars := 50
def multiplier := 3

theorem joe_total_cars : initial_cars + (multiplier * initial_cars) = 200 := by
  sorry

end joe_total_cars

namespace man_speed_in_still_water

import Mathlib

theorem man_speed_in_still_water
  (vm vs : ℝ)
  (h1 : vm + vs = 6)  -- effective speed downstream
  (h2 : vm - vs = 4)  -- effective speed upstream
  : vm = 5 := 
by
  sorry

end man_speed_in_still_water

namespace manuscript_age_in_decimal

import Mathlib

-- Given conditions
def octal_number : ℕ := 12345

-- Translate the problem statement into Lean:
theorem manuscript_age_in_decimal : (1 * 8^4 + 2 * 8^3 + 3 * 8^2 + 4 * 8^1 + 5 * 8^0) = 5349 :=
by
  sorry

end manuscript_age_in_decimal

namespace average_weight_a_b

import Mathlib

variables (A B C : ℝ)

theorem average_weight_a_b (h1 : (A + B + C) / 3 = 43)
                          (h2 : (B + C) / 2 = 43)
                          (h3 : B = 37) :
                          (A + B) / 2 = 40 :=
by
  sorry

end average_weight_a_b

namespace third_test_point

import Mathlib

noncomputable def test_points : ℝ × ℝ × ℝ :=
  let x1 := 2 + 0.618 * (4 - 2)
  let x2 := 2 + 4 - x1
  let x3 := 4 - 0.618 * (4 - x1)
  (x1, x2, x3)

theorem third_test_point :
  let x1 := 2 + 0.618 * (4 - 2)
  let x2 := 2 + 4 - x1
  let x3 := 4 - 0.618 * (4 - x1)
  x1 > x2 → x3 = 3.528 :=
by
  intros
  sorry

end third_test_point

namespace two_digit_numbers_condition

import Mathlib

theorem two_digit_numbers_condition : ∃ a b : ℕ, 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧
    10 * a + b ≥ 10 ∧ 10 * a + b ≤ 99 ∧
    (10 * a + b) / (a + b) = (a + b) / 3 ∧ 
    (10 * a + b = 27 ∨ 10 * a + b = 48) := 
by
    sorry

end two_digit_numbers_condition

namespace a3_value

import Mathlib

theorem a3_value (a : ℝ) (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℝ) (x : ℝ) :
  ( (1 + x) * (a - x) ^ 6 = a₀ + a₁ * x + a₂ * x^2 + a₃ * x^3 + a₄ * x^4 + a₅ * x^5 + a₆ * x^6 + a₇ * x^7 ) →
  ( a₀ + a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ = 0 ) →
  a = 1 →
  a₃ = -5 :=
by
  sorry

end a3_value

namespace Liza_reads_more_pages_than_Suzie

import Mathlib

def Liza_reading_speed : ℕ := 20
def Suzie_reading_speed : ℕ := 15
def hours : ℕ := 3

theorem Liza_reads_more_pages_than_Suzie :
  Liza_reading_speed * hours - Suzie_reading_speed * hours = 15 := by
  sorry

end Liza_reads_more_pages_than_Suzie

namespace sum_of_favorite_numbers

import Mathlib

def Glory_favorite_number : ℕ := 450
def Misty_favorite_number : ℕ := Glory_favorite_number / 3

theorem sum_of_favorite_numbers : Misty_favorite_number + Glory_favorite_number = 600 :=
by
  sorry

end sum_of_favorite_numbers

namespace find_cookies_per_tray

import Mathlib

def trays_baked_per_day := 2
def days_of_baking := 6
def cookies_eaten_by_frank := 1
def cookies_eaten_by_ted := 4
def cookies_left := 134

theorem find_cookies_per_tray (x : ℕ) (h : 12 * x - 10 = 134) : x = 12 :=
by
  sorry

end find_cookies_per_tray

namespace count_three_digit_numbers

import Mathlib

def count_decreasing_digit_numbers : ℕ :=
  ∑ h in Finset.range 10 \ {0, 1}, ∑ t in Finset.range h, t

theorem count_three_digit_numbers :
  count_decreasing_digit_numbers = 120 :=
sorry

end count_three_digit_numbers

namespace find_current

import Mathlib

open Complex

noncomputable def V : ℂ := 2 + I
noncomputable def Z : ℂ := 2 - 4 * I

theorem find_current :
  V / Z = (1 / 2) * I := 
sorry

end find_current

namespace find_x

import Mathlib

-- Define the functions δ (delta) and φ (phi)
def delta (x : ℚ) : ℚ := 4 * x + 9
def phi (x : ℚ) : ℚ := 9 * x + 8

-- State the theorem with conditions and question, and assert the answer
theorem find_x :
  (delta ∘ phi) x = 11 → x = -5/6 := by
  intros
  sorry

end find_x

namespace gross_profit_without_discount

import Mathlib

variable (C P : ℝ) -- Defining the cost and the full price as real numbers

-- Condition 1: Merchant sells an item at 10% discount (0.9P)
-- Condition 2: Makes a gross profit of 20% of the cost (0.2C)
-- SP = C + GP implies 0.9 P = 1.2 C

theorem gross_profit_without_discount :
  (0.9 * P = 1.2 * C) → ((C / 3) / C * 100 = 33.33) :=
by
  intro h
  sorry

end gross_profit_without_discount

namespace watermelon_cost_100

import Mathlib

variable (a b : ℕ) -- costs of one watermelon and one melon respectively
variable (x : ℕ) -- number of watermelons in the container

theorem watermelon_cost_100 :
  (∀ x, (1 : ℚ) = x / 160 + (150 - x) / 120 ∧ 120 * a = 30 * b ∧ 120 * a + 30 * b = 24000 ∧ x = 120) →
  a = 100 :=
by
  intro h
  sorry

end watermelon_cost_100

namespace carina_coffee

import Mathlib

def total_coffee (t f : ℕ) : ℕ := 10 * t + 5 * f

theorem carina_coffee (t : ℕ) (h1 : t = 3) (f : ℕ) (h2 : f = t + 2) : total_coffee t f = 55 := by
  sorry

end carina_coffee

namespace smallest_b_factors

import Mathlib

theorem smallest_b_factors (p q b : ℤ) (hpq : p * q = 1764) (hb : b = p + q) (hposp : p > 0) (hposq : q > 0) :
  b = 84 :=
by
  sorry

end smallest_b_factors

namespace reliability_is_correct

import Mathlib

-- Define the probabilities of each switch functioning properly.
def P_A : ℝ := 0.9
def P_B : ℝ := 0.8
def P_C : ℝ := 0.7

-- Define the system reliability.
def reliability : ℝ := P_A * P_B * P_C

-- The theorem stating the reliability of the system.
theorem reliability_is_correct : reliability = 0.504 := by
  sorry

end reliability_is_correct

namespace triangle_equilateral_if_arithmetic_sequences

import Mathlib

theorem triangle_equilateral_if_arithmetic_sequences
  (A B C : ℝ) (a b c : ℝ)
  (h_angles : A + B + C = 180)
  (h_angle_seq : ∃ (N : ℝ), A = B - N ∧ C = B + N)
  (h_sides : ∃ (n : ℝ), a = b - n ∧ c = b + n) :
  A = B ∧ B = C ∧ a = b ∧ b = c :=
sorry

end triangle_equilateral_if_arithmetic_sequences

namespace james_has_43_oreos

import Mathlib

def james_oreos (jordan : ℕ) : ℕ := 7 + 4 * jordan

theorem james_has_43_oreos (jordan : ℕ) (total : ℕ) (h1 : total = jordan + james_oreos jordan) (h2 : total = 52) : james_oreos jordan = 43 :=
by
  sorry

end james_has_43_oreos

namespace convert_denominators_to_integers

import Mathlib

def original_equation (x : ℝ) : Prop :=
  (x + 1) / 0.4 - (0.2 * x - 1) / 0.7 = 1

def transformed_equation (x : ℝ) : Prop :=
  (10 * x + 10) / 4 - (2 * x - 10) / 7 = 1

theorem convert_denominators_to_integers (x : ℝ) 
  (h : original_equation x) : transformed_equation x :=
sorry

end convert_denominators_to_integers

namespace acres_used_for_corn

import Mathlib

theorem acres_used_for_corn (total_acres : ℕ) (ratio_beans ratio_wheat ratio_corn : ℕ)
    (h_total : total_acres = 1034)
    (h_ratio : ratio_beans = 5 ∧ ratio_wheat = 2 ∧ ratio_corn = 4) : 
    ratio_corn * (total_acres / (ratio_beans + ratio_wheat + ratio_corn)) = 376 := 
by
  -- Proof goes here
  sorry

end acres_used_for_corn

namespace sum_of_fractions

import Mathlib

theorem sum_of_fractions : (3 / 20 : ℝ) + (5 / 50 : ℝ) + (7 / 2000 : ℝ) = 0.2535 :=
by sorry

end sum_of_fractions

namespace find_a7

import Mathlib

-- Definitions for geometric progression and conditions
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
∀ n, ∃ q : ℝ, a n = a 1 * q^(n-1)

def condition1 (a : ℕ → ℝ) : Prop :=
a 2 * a 4 * a 5 = a 3 * a 6

def condition2 (a : ℕ → ℝ) : Prop :=
a 9 * a 10 = -8

-- Theorem stating the equivalence given the conditions
theorem find_a7 (a : ℕ → ℝ) (h : is_geometric_sequence a) (h1 : condition1 a) (h2 : condition2 a) : 
    a 7 = -2 :=
sorry

end find_a7

namespace inequality_1_inequality_2

import Mathlib

variable (a b : ℝ)

-- Conditions
axiom pos_a : a > 0
axiom pos_b : b > 0
axiom sum_of_cubes_eq_two : a^3 + b^3 = 2

-- Question 1
theorem inequality_1 : (a + b) * (a^5 + b^5) ≥ 4 :=
by
  sorry

-- Question 2
theorem inequality_2 : a + b ≤ 2 :=
by
  sorry

end inequality_1_inequality_2

namespace 

import Mathlib

noncomputable def length_CB (CD DA CF : ℕ) (DF_parallel_AB : Prop) := 9 * (CD + DA) / CD

theorem length_of_CB {CD DA CF : ℕ} (DF_parallel_AB : Prop):
  CD = 3 → DA = 12 → CF = 9 → CB = 9 * 5 := by
  sorry

end 

namespace y_divides_x_squared

import Mathlib

-- Define the conditions and proof problem in Lean 4
theorem y_divides_x_squared (x y : ℕ) (hx : 0 < x) (hy : 0 < y) 
(h : ∃ (n : ℕ), n = (x^2 / y) + (y^2 / x)) : y ∣ x^2 :=
by {
  -- Proof steps are skipped
  sorry
}

end y_divides_x_squared

namespace find_fraction

import Mathlib

variable (n : ℚ) (x : ℚ)

theorem find_fraction (h1 : n = 0.5833333333333333) (h2 : n = 1/3 + x) : x = 0.25 := by
  sorry

end find_fraction

namespace sufficient_but_not_necessary_condition

import Mathlib

theorem sufficient_but_not_necessary_condition (a : ℝ) : (a = 2 → |a| = 2) ∧ (|a| = 2 → a = 2 ∨ a = -2) :=
by
  sorry

end sufficient_but_not_necessary_condition

namespace negation_exists_eq_forall

import Mathlib

theorem negation_exists_eq_forall (h : ¬ ∃ x : ℝ, x^2 + 2*x + 5 = 0) : ∀ x : ℝ, x^2 + 2*x + 5 ≠ 0 := 
by
  sorry

end negation_exists_eq_forall

namespace fencing_required

import Mathlib

theorem fencing_required (L W : ℝ) (hL : L = 20) (hA : 20 * W = 60) : 2 * W + L = 26 :=
by
  sorry

end fencing_required

namespace find_v_3

import Mathlib

def u (x : ℤ) : ℤ := 4 * x - 9

def v (z : ℤ) : ℤ := z^2 + 4 * z - 1

theorem find_v_3 : v 3 = 20 := by
  sorry

end find_v_3

namespace problem1_problem2_problem3_problem4

import Mathlib

-- Problem 1: 27 - 16 + (-7) - 18 = -14
theorem problem1 : 27 - 16 + (-7) - 18 = -14 := 
by 
  sorry

-- Problem 2: (-6) * (-3/4) / (-3/2) = -3
theorem problem2 : (-6) * (-3/4) / (-3/2) = -3 := 
by
  sorry

-- Problem 3: (1/2 - 3 + 5/6 - 7/12) / (-1/36) = 81
theorem problem3 : (1/2 - 3 + 5/6 - 7/12) / (-1/36) = 81 := 
by
  sorry

-- Problem 4: -2^4 + 3 * (-1)^4 - (-2)^3 = -5
theorem problem4 : -2^4 + 3 * (-1)^4 - (-2)^3 = -5 := 
by
  sorry

end problem1_problem2_problem3_problem4

namespace fraction_subtraction

import Mathlib

def lcm (a b : ℕ) : ℕ := Nat.lcm a b

theorem fraction_subtraction : 
  (18 / 42 - 2 / 9) = (13 / 63) := 
by 
  sorry

end fraction_subtraction

namespace number_of_ordered_pairs

import Mathlib

def count_valid_pairs : ℕ :=
  ∑ y in Finset.range 149, (150 - y) / (y * (y + 2))

theorem number_of_ordered_pairs :
  count_valid_pairs = ∑ y in Finset.range 149, (150 - y) / (y * (y + 2)) := by
  sorry

end number_of_ordered_pairs

namespace range_of_a

import Mathlib

theorem range_of_a (a : ℝ) : (¬ ∃ x : ℝ, x + 5 > 3 ∧ x > a ∧ x ≤ -2) ↔ a ≤ -2 :=
by
  sorry

end range_of_a

namespace small_boxes_in_large_box

import Mathlib

def number_of_chocolate_bars_in_small_box := 25
def total_number_of_chocolate_bars := 375

theorem small_boxes_in_large_box : total_number_of_chocolate_bars / number_of_chocolate_bars_in_small_box = 15 := by
  sorry

end small_boxes_in_large_box

namespace factor_expression

import Mathlib

theorem factor_expression (a b c : ℝ) :
  let num := (a^3 - b^3)^3 + (b^3 - c^3)^3 + (c^3 - a^3)^3
  let denom := (a - b)^3 + (b - c)^3 + (c - a)^3
  (denom ≠ 0) →
  num / denom = (a^2 + a*b + b^2) * (b^2 + b*c + c^2) * (c^2 + c*a + a^2) :=
by
  sorry

end factor_expression

namespace primes_unique

import Mathlib

-- Let's define that p, q, r are prime numbers, and define the main conditions.
def is_prime (n : ℕ) : Prop := Nat.Prime n

theorem primes_unique (p q r : ℕ) 
  (hp : is_prime p) 
  (hq : is_prime q)
  (hr : is_prime r)
  (div1 : (p^4 - 1) % (q * r) = 0)
  (div2 : (q^4 - 1) % (p * r) = 0)
  (div3 : (r^4 - 1) % (p * q) = 0) :
  (p = 2 ∧ q = 3 ∧ r = 5) ∨ (p = 2 ∧ q = 5 ∧ r = 3) ∨ (p = 3 ∧ q = 2 ∧ r = 5) ∨ 
  (p = 3 ∧ q = 5 ∧ r = 2) ∨ (p = 5 ∧ q = 2 ∧ r = 3) ∨ (p = 5 ∧ q = 3 ∧ r = 2) :=
by sorry

end primes_unique

namespace paintings_in_four_weeks

import Mathlib

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

end paintings_in_four_weeks

namespace time_for_A_to_complete_work

import Mathlib

-- Defining the work rates and the condition
def workRateA (a : ℕ) : ℚ := 1 / a
def workRateB : ℚ := 1 / 12
def workRateC : ℚ := 1 / 24
def combinedWorkRate (a : ℕ) : ℚ := workRateA a + workRateB + workRateC
def togetherWorkRate : ℚ := 1 / 4

-- Stating the theorem
theorem time_for_A_to_complete_work : 
  ∃ (a : ℕ), combinedWorkRate a = togetherWorkRate ∧ a = 8 :=
by
  sorry

end time_for_A_to_complete_work

namespace verify_parabola_D

import Mathlib

def vertex_form (a h k : ℝ) (x : ℝ) : ℝ := a * (x - h)^2 + k

def parabola_vertex (y : ℝ → ℝ) (h k : ℝ) : Prop :=
  ∀ x, y x = vertex_form (-1) h k x

-- Given conditions
def h : ℝ := 2
def k : ℝ := 3

-- Possible expressions
def parabola_A (x : ℝ) : ℝ := -((x + 2)^2) - 3
def parabola_B (x : ℝ) : ℝ := -((x - 2)^2) - 3
def parabola_C (x : ℝ) : ℝ := -((x + 2)^2) + 3
def parabola_D (x : ℝ) : ℝ := -((x - 2)^2) + 3

theorem verify_parabola_D : parabola_vertex parabola_D 2 3 :=
by
  -- Placeholder for the proof
  sorry

end verify_parabola_D

namespace magic_square_y_value

import Mathlib

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

end magic_square_y_value

namespace actual_distance_between_towns

import Mathlib

def map_scale : ℕ := 600000
def distance_on_map : ℕ := 2

theorem actual_distance_between_towns :
  (distance_on_map * map_scale) / 100 / 1000 = 12 :=
by
  sorry

end actual_distance_between_towns

namespace tank_capacity

import Mathlib

theorem tank_capacity (T : ℕ) (h1 : T > 0) 
    (h2 : (2 * T) / 5 + 15 + 20 = T - 25) : 
    T = 100 := 
  by 
    sorry

end tank_capacity

namespace union_complement_eq

import Mathlib

def U : Set ℕ := {0, 1, 2, 3, 4}
def A : Set ℕ := {0, 1, 2, 3}
def B : Set ℕ := {2, 3, 4}
def complement (U A : Set ℕ) : Set ℕ := {x ∈ U | x ∉ A}

theorem union_complement_eq :
  (complement U A ∪ B) = {2, 3, 4} :=
by
  sorry

end union_complement_eq

namespace smallest_value_y

import Mathlib

theorem smallest_value_y (y : ℝ) : (|y - 8| = 15) → y = -7 :=
by
  sorry

end smallest_value_y

namespace proof_of_problem_statement

import Mathlib

noncomputable def problem_statement : Prop :=
  ∀ (k : ℝ) (m : ℝ),
    (0 < m ∧ m < 3/2) → 
    (-3/(4 * m) = k) → 
    (k < -1/2)

theorem proof_of_problem_statement : problem_statement :=
  sorry

end proof_of_problem_statement

namespace exist_prime_not_dividing

import Mathlib

theorem exist_prime_not_dividing (p : ℕ) (hp : Prime p) : 
  ∃ q : ℕ, Prime q ∧ ∀ n : ℕ, 0 < n → ¬ (q ∣ n^p - p) := 
sorry

end exist_prime_not_dividing

namespace number_of_houses_built

import Mathlib

def original_houses : ℕ := 20817
def current_houses : ℕ := 118558
def houses_built : ℕ := current_houses - original_houses

theorem number_of_houses_built :
  houses_built = 97741 := by
  sorry

end number_of_houses_built

namespace problem

import Mathlib

theorem problem (a b : ℝ) (n : ℕ) (ha : a > 0) (hb : b > 0) 
  (h1 : 1 / a + 1 / b = 1) : 
  (a + b)^n - a^n - b^n ≥ 2^(2*n) - 2^(n + 1) := 
by
  sorry

end problem

namespace royWeight

import Mathlib

-- Define the problem conditions
def johnWeight : ℕ := 81
def johnHeavierBy : ℕ := 77

-- Define the main proof problem
theorem royWeight : (johnWeight - johnHeavierBy) = 4 := by
  sorry

end royWeight

namespace athlete_D_is_selected

import Mathlib

-- Define the average scores and variances of athletes
def avg_A : ℝ := 9.5
def var_A : ℝ := 6.6
def avg_B : ℝ := 9.6
def var_B : ℝ := 6.7
def avg_C : ℝ := 9.5
def var_C : ℝ := 6.7
def avg_D : ℝ := 9.6
def var_D : ℝ := 6.6

-- Define what it means for an athlete to be good and stable
def good_performance (avg : ℝ) : Prop := avg ≥ 9.6
def stable_play (variance : ℝ) : Prop := variance ≤ 6.6

-- Combine conditions for selecting the athlete
def D_is_suitable : Prop := good_performance avg_D ∧ stable_play var_D

-- State the theorem to be proved
theorem athlete_D_is_selected : D_is_suitable := 
by 
  sorry

end athlete_D_is_selected

namespace ryan_weekly_commuting_time

import Mathlib

-- Define Ryan's commuting conditions
def bike_time (biking_days: Nat) : Nat := biking_days * 30
def bus_time (bus_days: Nat) : Nat := bus_days * 40
def friend_time (friend_days: Nat) : Nat := friend_days * 10

-- Calculate total commuting time per week
def total_commuting_time (biking_days bus_days friend_days: Nat) : Nat := 
  bike_time biking_days + bus_time bus_days + friend_time friend_days

-- Given conditions
def biking_days : Nat := 1
def bus_days : Nat := 3
def friend_days : Nat := 1

-- Formal statement to prove
theorem ryan_weekly_commuting_time : 
  total_commuting_time biking_days bus_days friend_days = 160 := by 
  sorry

end ryan_weekly_commuting_time

namespace problem_statement

import Mathlib

open Real

noncomputable def f (ω varphi : ℝ) (x : ℝ) := 2 * sin (ω * x + varphi)

theorem problem_statement (ω varphi : ℝ) (x1 x2 : ℝ) (hω_pos : ω > 0) (hvarphi_abs : abs varphi < π / 2)
    (hf0 : f ω varphi 0 = -1) (hmonotonic : ∀ x y, π / 18 < x ∧ x < y ∧ y < π / 3 → f ω varphi x < f ω varphi y)
    (hshift : ∀ x, f ω varphi (x + π) = f ω varphi x)
    (hx1x2_interval : -17 * π / 12 < x1 ∧ x1 < -2 * π / 3 ∧ -17 * π / 12 < x2 ∧ x2 < -2 * π / 3 ∧ x1 ≠ x2)
    (heq_fx : f ω varphi x1 = f ω varphi x2) :
    f ω varphi (x1 + x2) = -1 :=
sorry

end problem_statement

namespace probability_product_positive_is_5_div_9

import Mathlib

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

end probability_product_positive_is_5_div_9

namespace min_value_alpha_beta_gamma

import Mathlib

def is_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

def is_cube (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k * k = n

def is_fifth_power (n : ℕ) : Prop :=
  ∃ k : ℕ, k ^ 5 = n

def A (α β γ : ℕ) : ℕ := 2 ^ α * 3 ^ β * 5 ^ γ

def condition_1 (α β γ : ℕ) : Prop :=
  is_square (A α β γ / 2)

def condition_2 (α β γ : ℕ) : Prop :=
  is_cube (A α β γ / 3)

def condition_3 (α β γ : ℕ) : Prop :=
  is_fifth_power (A α β γ / 5)

theorem min_value_alpha_beta_gamma (α β γ : ℕ) :
  condition_1 α β γ → condition_2 α β γ → condition_3 α β γ →
  α + β + γ = 31 :=
sorry

end min_value_alpha_beta_gamma

namespace f_m_eq_five

import Mathlib

def f (x : ℝ) (a : ℝ) : ℝ :=
  x^3 + a * x + 3

axiom f_neg_m : ∀ (m a : ℝ), f (-m) a = 1

theorem f_m_eq_five (m a : ℝ) (h : f (-m) a = 1) : f m a = 5 :=
  by sorry

end f_m_eq_five

namespace 

import Mathlib

theorem line_through_point_parallel_to_line {x y : ℝ} 
  (point : x = 1 ∧ y = 0) 
  (parallel_line : ∃ c : ℝ, ∀ x y : ℝ, x - 2 * y + c = 0) :
  x - 2 * y - 1 = 0 := 
by
  sorry

end 

namespace total_cups_needed

import Mathlib

-- Define the known conditions
def ratio_butter : ℕ := 2
def ratio_flour : ℕ := 3
def ratio_sugar : ℕ := 5
def total_sugar_in_cups : ℕ := 10

-- Define the parts-to-cups conversion
def cup_per_part := total_sugar_in_cups / ratio_sugar

-- Define the amounts of each ingredient in cups
def butter_in_cups := ratio_butter * cup_per_part
def flour_in_cups := ratio_flour * cup_per_part
def sugar_in_cups := ratio_sugar * cup_per_part

-- Define the total number of cups
def total_cups := butter_in_cups + flour_in_cups + sugar_in_cups

-- Theorem to prove
theorem total_cups_needed : total_cups = 20 := by
  sorry

end total_cups_needed

namespace johns_number

import Mathlib

theorem johns_number (n : ℕ) 
  (h1 : 125 ∣ n) 
  (h2 : 30 ∣ n) 
  (h3 : 800 ≤ n ∧ n ≤ 2000) : 
  n = 1500 :=
sorry

end johns_number

namespace tulips_sum

import Mathlib

def tulips_total (arwen_tulips : ℕ) (elrond_tulips : ℕ) : ℕ := arwen_tulips + elrond_tulips

theorem tulips_sum : tulips_total 20 (2 * 20) = 60 := by
  sorry

end tulips_sum

namespace find_n

import Mathlib

theorem find_n (n : ℕ) (h : 2 ^ 3 * 5 * n = Nat.factorial 10) : n = 45360 :=
sorry

end find_n

namespace sum_of_exponents

import Mathlib

def power_sum_2021 (a : ℕ → ℤ) (n : ℕ → ℕ) (r : ℕ) : Prop :=
  (∀ k, 1 ≤ k ∧ k ≤ r → (a k = 1 ∨ a k = -1)) ∧
  (a 1 * 3 ^ n 1 + a 2 * 3 ^ n 2 + a 3 * 3 ^ n 3 + a 4 * 3 ^ n 4 + a 5 * 3 ^ n 5 + a 6 * 3 ^ n 6 = 2021) ∧
  (n 1 = 7 ∧ n 2 = 5 ∧ n 3 = 4 ∧ n 4 = 2 ∧ n 5 = 1 ∧ n 6 = 0) ∧
  (a 1 = 1 ∧ a 2 = -1 ∧ a 3 = 1 ∧ a 4 = -1 ∧ a 5 = 1 ∧ a 6 = -1)

theorem sum_of_exponents : ∃ (a : ℕ → ℤ) (n : ℕ → ℕ) (r : ℕ), power_sum_2021 a n r ∧ (n 1 + n 2 + n 3 + n 4 + n 5 + n 6 = 19) :=
by {
  sorry
}

end sum_of_exponents

namespace rational_b_if_rational_a

import Mathlib

theorem rational_b_if_rational_a (x : ℚ) (h_rational : ∃ a : ℚ, a = x / (x^2 - x + 1)) :
  ∃ b : ℚ, b = x^2 / (x^4 - x^2 + 1) :=
by
  sorry

end rational_b_if_rational_a

namespace tram_speed

import Mathlib

variables (V : ℝ)

theorem tram_speed (h : (V + 5) / (V - 5) = 600 / 225) : V = 11 :=
sorry

end tram_speed

namespace prove_a5

import Mathlib

-- Definition of the conditions
def expansion (x : ℤ) (a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 : ℤ) :=
  (x - 1) ^ 8 = a_0 + a_1 * (1 + x) + a_2 * (1 + x)^2 + a_3 * (1 + x)^3 + a_4 * (1 + x)^4 + 
               a_5 * (1 + x)^5 + a_6 * (1 + x)^6 + a_7 * (1 + x)^7 + a_8 * (1 + x)^8

-- Given condition
axiom condition (a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 : ℤ) : ∀ x : ℤ, expansion x a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8

-- The target problem: proving a_5 = -448
theorem prove_a5 (a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 : ℤ) : a_5 = -448 :=
by
  sorry

end prove_a5

namespace probability_sum_of_five

import Mathlib

def total_outcomes : ℕ := 36
def favorable_outcomes : ℕ := 4

theorem probability_sum_of_five :
  favorable_outcomes / total_outcomes = 1 / 9 := 
by
  sorry

end probability_sum_of_five

namespace replace_asterisk

import Mathlib

theorem replace_asterisk :
  ∃ x : ℤ, (x / 21) * (63 / 189) = 1 ∧ x = 63 := sorry

end replace_asterisk

namespace problem_Ashwin_Sah

import Mathlib

def sqrt_int (n : ℤ) : Prop := ∃ m : ℤ, m * m = n

theorem problem_Ashwin_Sah (a b : ℕ) (k : ℤ) (x y : ℕ) :
  (∀ a b : ℕ, ∃ k : ℤ, (a^2 + b^2 + 2 = k * a * b )) →
  (∀ (a b : ℕ), a ≤ b ∨ b < a) →
  (∀ (a b : ℕ), sqrt_int (((k * a) * (k * a) - 4 * (a^2 + 2)))) →
  ∀ (x y : ℕ), (x + y) % 2017 = 24 := by
  sorry

end problem_Ashwin_Sah

namespace expand_product

import Mathlib

theorem expand_product (y : ℝ) : 3 * (y - 6) * (y + 9) = 3 * y^2 + 9 * y - 162 :=
by sorry

end expand_product

namespace rex_lesson_schedule

import Mathlib

-- Define the total lessons and weeks
def total_lessons : ℕ := 40
def weeks_completed : ℕ := 6
def weeks_remaining : ℕ := 4

-- Define the proof statement
theorem rex_lesson_schedule : (weeks_completed + weeks_remaining) * 4 = total_lessons := by
  -- Proof placeholder, to be filled in 
  sorry

end rex_lesson_schedule

namespace ted_alex_age_ratio

import Mathlib

theorem ted_alex_age_ratio (t a : ℕ) 
  (h1 : t - 3 = 4 * (a - 3))
  (h2 : t - 5 = 5 * (a - 5)) : 
  ∃ x : ℕ, (t + x) / (a + x) = 3 ∧ x = 1 :=
by
  sorry

end ted_alex_age_ratio

namespace initial_lychees_count

import Mathlib

theorem initial_lychees_count (L : ℕ) (h1 : L / 2 = 2 * 100 * 5 / 5 * 5) : L = 500 :=
by sorry

end initial_lychees_count

namespace correct_value_of_A_sub_B

import Mathlib

variable {x y : ℝ}

-- Given two polynomials A and B where B = 3x - 2y, and a mistaken equation A + B = x - y,
-- we want to prove the correct value of A - B.
theorem correct_value_of_A_sub_B (A B : ℝ) (h1 : B = 3 * x - 2 * y) (h2 : A + B = x - y) :
  A - B = -5 * x + 3 * y :=
by
  sorry

end correct_value_of_A_sub_B

namespace correct_bushes_needed

import Mathlib

def yield_per_bush := 10
def containers_per_zucchini := 3
def zucchinis_needed := 36
def bushes_needed (yield_per_bush containers_per_zucchini zucchinis_needed : ℕ) : ℕ :=
  Nat.ceil ((zucchinis_needed * containers_per_zucchini : ℕ) / yield_per_bush)

theorem correct_bushes_needed : bushes_needed yield_per_bush containers_per_zucchini zucchinis_needed = 11 := 
by
  sorry

end correct_bushes_needed

namespace cuckoo_chime_78

import Mathlib

-- Define the arithmetic sum for the cuckoo clock problem
def cuckoo_chime_sum (n a l : Nat) : Nat :=
  n * (a + l) / 2

-- Main theorem
theorem cuckoo_chime_78 : 
  cuckoo_chime_sum 12 1 12 = 78 := 
by
  -- Proof part can be written here
  sorry

end cuckoo_chime_78

namespace find_a_for_arithmetic_progression_roots

import Mathlib

theorem find_a_for_arithmetic_progression_roots (x a : ℝ) : 
  (∀ (x : ℝ), x^4 - a*x^2 + 1 = 0) → 
  (∃ (t1 t2 : ℝ), t1 > 0 ∧ t2 > 0 ∧ (t2 = 9*t1) ∧ (t1 + t2 = a) ∧ (t1 * t2 = 1)) → 
  (a = 10/3) := 
  by 
    intros h1 h2
    sorry

end find_a_for_arithmetic_progression_roots

namespace apples_left_total

import Mathlib

-- Define the initial conditions
def FrankApples : ℕ := 36
def SusanApples : ℕ := 3 * FrankApples
def SusanLeft : ℕ := SusanApples / 2
def FrankLeft : ℕ := (2 / 3) * FrankApples

-- Define the total apples left
def total_apples_left (SusanLeft FrankLeft : ℕ) : ℕ := SusanLeft + FrankLeft

-- Given conditions transformed to Lean
theorem apples_left_total : 
  total_apples_left (SusanApples / 2) ((2 / 3) * FrankApples) = 78 := by
  sorry

end apples_left_total

namespace insert_zeros_between_digits_is_cube

import Mathlib

def is_perfect_cube (n : ℕ) : Prop := ∃ k : ℕ, k^3 = n

theorem insert_zeros_between_digits_is_cube (k b : ℕ) (h_b : b ≥ 4) 
  : is_perfect_cube (1 * b^(3*(1+k)) + 3 * b^(2*(1+k)) + 3 * b^(1+k) + 1) :=
sorry

end insert_zeros_between_digits_is_cube

namespace solve_for_y

import Mathlib

theorem solve_for_y (x y : ℝ) (h : x + 2 * y = 6) : y = (-x + 6) / 2 :=
  sorry

end solve_for_y

namespace find_a_for_extraneous_roots_find_a_for_no_solution

import Mathlib

-- Define the original fractional equation
def eq_fraction (x a: ℝ) : Prop := (x - a) / (x - 2) - 5 / x = 1

-- Proposition for extraneous roots
theorem find_a_for_extraneous_roots (a: ℝ) (extraneous_roots : ∃ x : ℝ, (x - a) / (x - 2) - 5 / x = 1 ∧ (x = 0 ∨ x = 2)): a = 2 := by 
sorry

-- Proposition for no solution
theorem find_a_for_no_solution (a: ℝ) (no_solution : ∀ x : ℝ, (x - a) / (x - 2) - 5 / x ≠ 1): a = -3 ∨ a = 2 := by 
sorry

end find_a_for_extraneous_roots_find_a_for_no_solution

namespace y_intercepts_count

import Mathlib

theorem y_intercepts_count : 
  ∀ (a b c : ℝ), a = 3 ∧ b = (-4) ∧ c = 5 → (b^2 - 4*a*c < 0) → ∀ y : ℝ, x = 3*y^2 - 4*y + 5 → x ≠ 0 :=
by
  sorry

end y_intercepts_count

namespace simplify_and_evaluate

import Mathlib

noncomputable 
def expr (a b : ℚ) := 2*(a^2*b - 2*a*b) - 3*(a^2*b - 3*a*b) + a^2*b

theorem simplify_and_evaluate :
  let a := (-2 : ℚ) 
  let b := (1/3 : ℚ)
  expr a b = -10/3 :=
by
  sorry

end simplify_and_evaluate

namespace strawberries_harvest

import Mathlib

theorem strawberries_harvest (length : ℕ) (width : ℕ) 
  (plants_per_sqft : ℕ) (strawberries_per_plant : ℕ) 
  (area := length * width) (total_plants := plants_per_sqft * area) 
  (total_strawberries := strawberries_per_plant * total_plants) :
  length = 10 → width = 9 →
  plants_per_sqft = 5 → strawberries_per_plant = 8 →
  total_strawberries = 3600 := by
  sorry

end strawberries_harvest

namespace exp_mono_increasing

import Mathlib

theorem exp_mono_increasing (x y : ℝ) (h : x ≤ y) : (2:ℝ)^x ≤ (2:ℝ)^y :=
sorry

end exp_mono_increasing

namespace polynomial_sum_at_points

import Mathlib

def P (x : ℝ) : ℝ := x^5 - 1.7 * x^3 + 2.5

theorem polynomial_sum_at_points :
  P 19.1 + P (-19.1) = 5 := by
  sorry

end polynomial_sum_at_points

namespace value_of_m

import Mathlib

theorem value_of_m (m : ℤ) (h : ∃ x : ℤ, x = 2 ∧ x^2 + m * x + 2 = 0) : m = -3 :=
by
  sorry

end value_of_m

namespace ab_equals_six

import Mathlib

variable (a b : ℝ)

theorem ab_equals_six (h : a / 2 = 3 / b) : a * b = 6 := 
by sorry

end ab_equals_six

namespace base_b_sum_correct

import Mathlib

def sum_double_digit_numbers (b : ℕ) : ℕ :=
  (b * (b - 1) * (b ^ 2 - b + 1)) / 2

def base_b_sum (b : ℕ) : ℕ :=
  b ^ 2 + 12 * b + 5

theorem base_b_sum_correct : ∃ b : ℕ, sum_double_digit_numbers b = base_b_sum b ∧ b = 15 :=
by
  sorry

end base_b_sum_correct

namespace find_t

import Mathlib

theorem find_t (t : ℝ) (h : (1 / (t+3) + 3 * t / (t+3) - 4 / (t+3)) = 5) : t = -9 :=
by
  sorry

end find_t

namespace sum_of_four_digit_integers_up_to_4999

import Mathlib

theorem sum_of_four_digit_integers_up_to_4999 : 
  let a := 1000
  let l := 4999
  let n := l - a + 1
  let S := (n / 2) * (a + l)
  S = 11998000 := 
by
  sorry

end sum_of_four_digit_integers_up_to_4999

namespace prime_has_property_p

import Mathlib

theorem prime_has_property_p (n : ℕ) (hn : Prime n) (a : ℕ) (h : n ∣ a^n - 1) : n^2 ∣ a^n - 1 := by
  sorry

end prime_has_property_p

namespace certain_number_divisible

import Mathlib

theorem certain_number_divisible (x : ℤ) (n : ℤ) (h1 : 0 < n ∧ n < 11) (h2 : x - n = 11 * k) (h3 : n = 1) : x = 12 :=
by sorry

end certain_number_divisible

namespace green_eyed_snack_min

import Mathlib

variable {total_count green_eyes_count snack_bringers_count : ℕ}

def least_green_eyed_snack_bringers (total_count green_eyes_count snack_bringers_count : ℕ) : ℕ :=
  green_eyes_count - (total_count - snack_bringers_count)

theorem green_eyed_snack_min 
  (h_total : total_count = 35)
  (h_green_eyes : green_eyes_count = 18)
  (h_snack_bringers : snack_bringers_count = 24)
  : least_green_eyed_snack_bringers total_count green_eyes_count snack_bringers_count = 7 :=
by
  rw [h_total, h_green_eyes, h_snack_bringers]
  unfold least_green_eyed_snack_bringers
  norm_num

end green_eyed_snack_min

namespace problem_statement

import Mathlib

theorem problem_statement 
  (p q r s : ℝ) 
  (h : (p - q) * (r - s) / ((q - r) * (s - p)) = 3 / 7) : 
  (p - r) * (q - s) / ((p - q) * (r - s)) = -3 / 2 := 
    sorry

end problem_statement

namespace age_difference_64

import Mathlib

variables (Patrick Michael Monica : ℕ)
axiom age_ratio_1 : ∃ (x : ℕ), Patrick = 3 * x ∧ Michael = 5 * x
axiom age_ratio_2 : ∃ (y : ℕ), Michael = 3 * y ∧ Monica = 5 * y
axiom age_sum : Patrick + Michael + Monica = 196

theorem age_difference_64 : Monica - Patrick = 64 :=
by {
  sorry
}

end age_difference_64

namespace alice_age_2005

import Mathlib

-- Definitions
variables (x : ℕ) (age_Alice_2000 age_Grandmother_2000 : ℕ)
variables (born_Alice born_Grandmother : ℕ)

-- Conditions
def alice_grandmother_relation_at_2000 := age_Alice_2000 = x ∧ age_Grandmother_2000 = 3 * x
def birth_year_sum := born_Alice + born_Grandmother = 3870
def birth_year_Alice := born_Alice = 2000 - x
def birth_year_Grandmother := born_Grandmother = 2000 - 3 * x

-- Proving the main statement: Alice's age at the end of 2005
theorem alice_age_2005 : 
  alice_grandmother_relation_at_2000 x age_Alice_2000 age_Grandmother_2000 ∧ 
  birth_year_sum born_Alice born_Grandmother ∧ 
  birth_year_Alice x born_Alice ∧ 
  birth_year_Grandmother x born_Grandmother 
  → 2005 - 2000 + age_Alice_2000 = 37 := 
by 
  intros
  sorry

end alice_age_2005

namespace pencil_price_is_99c

import Mathlib

noncomputable def one_pencil_cost (total_spent : ℝ) (notebook_price : ℝ) (notebook_count : ℕ) 
                                  (ruler_pack_price : ℝ) (eraser_price : ℝ) (eraser_count : ℕ) 
                                  (pencil_count : ℕ) (discount : ℝ) (tax : ℝ) : ℝ :=
  let notebooks_cost := notebook_count * notebook_price
  let discount_amount := discount * notebooks_cost
  let discounted_notebooks_cost := notebooks_cost - discount_amount
  let other_items_cost := ruler_pack_price + (eraser_count * eraser_price)
  let subtotal := discounted_notebooks_cost + other_items_cost
  let pencils_total_after_tax := total_spent - subtotal
  let pencils_total_before_tax := pencils_total_after_tax / (1 + tax)
  let pencil_price := pencils_total_before_tax / pencil_count
  pencil_price

theorem pencil_price_is_99c : one_pencil_cost 7.40 0.85 2 0.60 0.20 5 4 0.15 0.10 = 0.99 := 
sorry

end pencil_price_is_99c

namespace sqrt_one_half_eq_sqrt_two_over_two

import Mathlib

theorem sqrt_one_half_eq_sqrt_two_over_two : Real.sqrt (1 / 2) = Real.sqrt 2 / 2 :=
by sorry

end sqrt_one_half_eq_sqrt_two_over_two

namespace geometric_sequence_sum

import Mathlib

theorem geometric_sequence_sum (a_n : ℕ → ℕ) (h1 : a_n 2 = 2) (h2 : a_n 5 = 16) :
  (∑ k in Finset.range n, a_n k * a_n (k + 1)) = (2 / 3) * (4^n - 1) := 
sorry

end geometric_sequence_sum

namespace y_intercept_of_line

import Mathlib

theorem y_intercept_of_line (x y : ℝ) : x + 2 * y + 6 = 0 → x = 0 → y = -3 :=
by
  sorry

end y_intercept_of_line

namespace find_smallest_subtract

import Mathlib

-- Definitions for multiples
def is_mul_2 (n : ℕ) : Prop := 2 ∣ n
def is_mul_3 (n : ℕ) : Prop := 3 ∣ n
def is_mul_5 (n : ℕ) : Prop := 5 ∣ n

-- Statement of the problem
theorem find_smallest_subtract (x : ℕ) :
  (is_mul_2 (134 - x)) ∧ (is_mul_3 (134 - x)) ∧ (is_mul_5 (134 - x)) → x = 14 :=
by
  sorry

end find_smallest_subtract

namespace mike_corvette_average_speed

import Mathlib

theorem mike_corvette_average_speed
  (D : ℚ) (v : ℚ) (total_distance : ℚ)
  (first_half_distance : ℚ) (second_half_time_ratio : ℚ)
  (total_time : ℚ) (average_rate : ℚ) :
  total_distance = 640 ∧
  first_half_distance = total_distance / 2 ∧
  second_half_time_ratio = 3 ∧
  average_rate = 40 →
  v = 80 :=
by
  intros h
  have total_distance_eq : total_distance = 640 := h.1
  have first_half_distance_eq : first_half_distance = total_distance / 2 := h.2.1
  have second_half_time_ratio_eq : second_half_time_ratio = 3 := h.2.2.1
  have average_rate_eq : average_rate = 40 := h.2.2.2
  sorry

end mike_corvette_average_speed

namespace correct_meteor_passing_time

import Mathlib

theorem correct_meteor_passing_time :
  let T1 := 7
  let T2 := 13
  let harmonic_mean := (2 * T1 * T2) / (T1 + T2)
  harmonic_mean = 9.1 := 
by
  sorry

end correct_meteor_passing_time

namespace trains_crossing_time_correct

import Mathlib

def convert_kmph_to_mps (speed_kmph : ℕ) : ℚ := (speed_kmph * 5) / 18

def time_to_cross_each_other 
  (length_train1 length_train2 speed_kmph_train1 speed_kmph_train2 : ℕ) : ℚ :=
  let speed_train1 := convert_kmph_to_mps speed_kmph_train1
  let speed_train2 := convert_kmph_to_mps speed_kmph_train2
  let relative_speed := speed_train2 - speed_train1
  let total_distance := length_train1 + length_train2
  (total_distance : ℚ) / relative_speed

theorem trains_crossing_time_correct :
  time_to_cross_each_other 200 150 40 46 = 210 := by
  sorry

end trains_crossing_time_correct

namespace horse_revolutions

import Mathlib

theorem horse_revolutions (r1 r2  : ℝ) (rev1 rev2 : ℕ)
  (h1 : r1 = 30) (h2 : rev1 = 20) (h3 : r2 = 10) : rev2 = 60 :=
by
  sorry

end horse_revolutions

namespace rainfall_difference

import Mathlib

-- Define the conditions
def first_day_rainfall : ℕ := 26
def second_day_rainfall : ℕ := 34
def third_day_rainfall : ℕ := second_day_rainfall - 12
def total_rainfall_this_year : ℕ := first_day_rainfall + second_day_rainfall + third_day_rainfall
def average_rainfall : ℕ := 140

-- Define the statement to prove
theorem rainfall_difference : average_rainfall - total_rainfall_this_year = 58 := by
  -- Add your proof here
  sorry

end rainfall_difference

namespace village_Y_initial_population

import Mathlib

def population_X := 76000
def decrease_rate_X := 1200
def increase_rate_Y := 800
def years := 17

def population_X_after_17_years := population_X - decrease_rate_X * years
def population_Y_after_17_years (P : Nat) := P + increase_rate_Y * years

theorem village_Y_initial_population (P : Nat) (h : population_Y_after_17_years P = population_X_after_17_years) : P = 42000 :=
by
  sorry

end village_Y_initial_population

namespace lamp_count_and_profit

import Mathlib

-- Define the parameters given in the problem
def total_lamps : ℕ := 50
def total_cost : ℕ := 2500
def cost_A : ℕ := 40
def cost_B : ℕ := 65
def marked_A : ℕ := 60
def marked_B : ℕ := 100
def discount_A : ℕ := 10 -- percent
def discount_B : ℕ := 30 -- percent

-- Derived definitions from the solution
def lamps_A : ℕ := 30
def lamps_B : ℕ := 20
def selling_price_A : ℕ := marked_A * (100 - discount_A) / 100
def selling_price_B : ℕ := marked_B * (100 - discount_B) / 100
def profit_A : ℕ := selling_price_A - cost_A
def profit_B : ℕ := selling_price_B - cost_B
def total_profit : ℕ := (profit_A * lamps_A) + (profit_B * lamps_B)

-- Lean statement
theorem lamp_count_and_profit :
  lamps_A + lamps_B = total_lamps ∧
  (cost_A * lamps_A + cost_B * lamps_B) = total_cost ∧
  total_profit = 520 := by
  -- proofs will go here
  sorry

end lamp_count_and_profit

namespace range_of_f

import Mathlib

theorem range_of_f (x : ℝ) (h1 : 1 ≤ x) (h2 : x ≤ 2) : -3 ≤ (3^x - 6/x) ∧ (3^x - 6/x) ≤ 6 :=
by
  sorry

end range_of_f

namespace near_square_qoutient

import Mathlib

def is_near_square (n : ℕ) : Prop := ∃ k : ℕ, n = k * (k + 1)

theorem near_square_qoutient (n : ℕ) (hn : is_near_square n) : 
  ∃ a b : ℕ, is_near_square a ∧ is_near_square b ∧ n = a / b := 
sorry

end near_square_qoutient

namespace common_ratio_geometric_series

import Mathlib 

theorem common_ratio_geometric_series :
  let a := (7 : ℚ) / 8
  let b := - (5 : ℚ) / 12
  (b / a) = - (10 : ℚ) / 21 :=
by
  -- definitions
  let a := (7 : ℚ) / 8
  let b := - (5 : ℚ) / 12
  -- assertion
  have ratio := b / a
  sorry

end common_ratio_geometric_series

namespace different_prime_factors_of_factorial_eq_10

import Mathlib

-- First, define n as 30
def n : ℕ := 30

-- Define a list of primes less than 30
def primesLessThanN : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

-- This is the theorem stating the number of distinct prime factors of 30!
theorem different_prime_factors_of_factorial_eq_10 : 
  (primesLessThanN.filter (Nat.Prime)).length = 10 := by 
  sorry

end different_prime_factors_of_factorial_eq_10