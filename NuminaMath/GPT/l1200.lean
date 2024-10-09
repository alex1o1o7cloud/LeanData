import Mathlib

namespace solve_tan_equation_l1200_120009

theorem solve_tan_equation (x : ℝ) (k : ℤ) :
  8.456 * (Real.tan x)^2 * (Real.tan (3 * x))^2 * Real.tan (4 * x) = 
  (Real.tan x)^2 - (Real.tan (3 * x))^2 + Real.tan (4 * x) ->
  x = π * k ∨ x = π / 4 * (2 * k + 1) := sorry

end solve_tan_equation_l1200_120009


namespace fox_can_eat_80_fox_cannot_eat_65_l1200_120073
-- import the required library

-- Define the conditions for the problem.
def total_candies := 100
def piles := 3
def fox_eat_equalize (fox: ℕ) (pile1: ℕ) (pile2: ℕ): ℕ :=
  if pile1 = pile2 then fox + pile1 else fox + pile2 - pile1

-- Statement for part (a)
theorem fox_can_eat_80: ∃ c₁ c₂ c₃: ℕ, (c₁ + c₂ + c₃ = total_candies) ∧ 
  (∃ x: ℕ, (fox_eat_equalize (c₁ + c₂ + c₃ - x) c₁ c₂ = 80) ∨ 
              (fox_eat_equalize x c₁ c₂  = 80)) :=
sorry

-- Statement for part (b)
theorem fox_cannot_eat_65: ¬ (∃ c₁ c₂ c₃: ℕ, (c₁ + c₂ + c₃ = total_candies) ∧ 
  (∃ x: ℕ, (fox_eat_equalize (c₁ + c₂ + c₃ - x) c₁ c₂ = 65) ∨ 
              (fox_eat_equalize x c₁ c₂  = 65))) :=
sorry

end fox_can_eat_80_fox_cannot_eat_65_l1200_120073


namespace simon_removes_exactly_180_silver_coins_l1200_120097

theorem simon_removes_exactly_180_silver_coins :
  ∀ (initial_total_coins initial_gold_percentage final_gold_percentage : ℝ) 
  (initial_silver_coins final_total_coins final_silver_coins silver_coins_removed : ℕ),
  initial_total_coins = 200 → 
  initial_gold_percentage = 0.02 →
  final_gold_percentage = 0.2 →
  initial_silver_coins = (initial_total_coins * (1 - initial_gold_percentage)) → 
  final_total_coins = (4 / final_gold_percentage) →
  final_silver_coins = (final_total_coins - 4) →
  silver_coins_removed = (initial_silver_coins - final_silver_coins) →
  silver_coins_removed = 180 :=
by
  intros initial_total_coins initial_gold_percentage final_gold_percentage 
         initial_silver_coins final_total_coins final_silver_coins silver_coins_removed
  sorry

end simon_removes_exactly_180_silver_coins_l1200_120097


namespace arithmetic_sequence_common_difference_l1200_120099

theorem arithmetic_sequence_common_difference
  (a : ℕ → ℝ) (S : ℕ → ℝ) (d : ℝ)
  (h1 : a 1 = 30)
  (h2 : ∀ n, S n = n * (a 1 + (n - 1) / 2 * d))
  (h3 : S 12 = S 19) :
  d = -2 :=
by
  sorry

end arithmetic_sequence_common_difference_l1200_120099


namespace smallest_positive_debt_resolvable_l1200_120028

/-- The value of a sheep in dollars -/
def sheep_value : ℕ := 250

/-- The value of a lamb in dollars -/
def lamb_value : ℕ := 150

/-- Given a debt D that can be expressed in the form of 250s + 150l for integers s and l,
prove that the smallest positive amount of D is 50 dollars -/
theorem smallest_positive_debt_resolvable : 
  ∃ (s l : ℤ), sheep_value * s + lamb_value * l = 50 :=
sorry

end smallest_positive_debt_resolvable_l1200_120028


namespace problem1_problem2_l1200_120008

-- Problem 1: Proving the range of m values for the given inequality
theorem problem1 (m : ℝ) : (∀ x : ℝ, |x + 1| + |x - m| ≥ 3) ↔ (m ≤ -4 ∨ m ≥ 2) :=
sorry

-- Problem 2: Proving the range of m values given a non-empty solution set for the inequality
theorem problem2 (m : ℝ) : (∃ x : ℝ, |m + 1| - 2 * m ≥ x^2 - x) ↔ (m ≤ 5/4) :=
sorry

end problem1_problem2_l1200_120008


namespace isosceles_triangle_perimeter_l1200_120093

theorem isosceles_triangle_perimeter {a b : ℕ} (h₁ : a = 4) (h₂ : b = 9) (h₃ : ∀ x y z : ℕ, 
  (x = a ∧ y = a ∧ z = b) ∨ (x = b ∧ y = b ∧ z = a) → 
  (x + y > z ∧ x + z > y ∧ y + z > x)) : 
  (a = 4 ∧ b = 9) → a + a + b = 22 :=
by sorry

end isosceles_triangle_perimeter_l1200_120093


namespace remainder_7_pow_700_div_100_l1200_120042

theorem remainder_7_pow_700_div_100 : (7 ^ 700) % 100 = 1 := 
  by sorry

end remainder_7_pow_700_div_100_l1200_120042


namespace total_board_length_l1200_120094

-- Defining the lengths of the pieces of the board
def shorter_piece_length : ℕ := 23
def longer_piece_length : ℕ := 2 * shorter_piece_length

-- Stating the theorem that the total length of the board is 69 inches
theorem total_board_length : shorter_piece_length + longer_piece_length = 69 :=
by
  -- The proof is omitted for now
  sorry

end total_board_length_l1200_120094


namespace train_speed_in_km_hr_l1200_120050

-- Definitions based on conditions
def train_length : ℝ := 150  -- meters
def crossing_time : ℝ := 6  -- seconds

-- Definition for conversion factor
def meters_per_second_to_km_per_hour (speed_mps : ℝ) : ℝ := speed_mps * 3.6

-- Main theorem
theorem train_speed_in_km_hr : meters_per_second_to_km_per_hour (train_length / crossing_time) = 90 :=
by
  sorry

end train_speed_in_km_hr_l1200_120050


namespace winning_percentage_l1200_120053

noncomputable def total_votes (votes_winner votes_margin : ℕ) : ℕ :=
  votes_winner + (votes_winner - votes_margin)

noncomputable def percentage_votes (votes_winner total_votes : ℕ) : ℝ :=
  (votes_winner : ℝ) / (total_votes : ℝ) * 100

theorem winning_percentage
  (votes_winner : ℕ)
  (votes_margin : ℕ)
  (h_winner : votes_winner = 775)
  (h_margin : votes_margin = 300) :
  percentage_votes votes_winner (total_votes votes_winner votes_margin) = 62 :=
sorry

end winning_percentage_l1200_120053


namespace min_balls_draw_l1200_120049

def box1_red := 40
def box1_green := 30
def box1_yellow := 25
def box1_blue := 15

def box2_red := 35
def box2_green := 25
def box2_yellow := 20

def min_balls_to_draw_to_get_20_balls_of_single_color (totalRed totalGreen totalYellow totalBlue : ℕ) : ℕ :=
  let maxNoColor :=
    (min totalRed 19) + (min totalGreen 19) + (min totalYellow 19) + (min totalBlue 15)
  maxNoColor + 1

theorem  min_balls_draw {r1 r2 g1 g2 y1 y2 b1 : ℕ} :
  r1 = box1_red -> g1 = box1_green -> y1 = box1_yellow -> b1 = box1_blue ->
  r2 = box2_red -> g2 = box2_green -> y2 = box2_yellow ->
  min_balls_to_draw_to_get_20_balls_of_single_color (r1 + r2) (g1 + g2) (y1 + y2) b1 = 73 :=
by
  intros
  unfold min_balls_to_draw_to_get_20_balls_of_single_color
  sorry

end min_balls_draw_l1200_120049


namespace angle_no_complement_greater_than_90_l1200_120015

-- Definition of angle
def angle (A : ℝ) : Prop := 
  A = 100 + (15 / 60)

-- Definition of complement
def has_complement (A : ℝ) : Prop :=
  A < 90

-- Theorem: Angles greater than 90 degrees do not have complements
theorem angle_no_complement_greater_than_90 {A : ℝ} (h: angle A) : ¬ has_complement A :=
by sorry

end angle_no_complement_greater_than_90_l1200_120015


namespace product_of_3_point_6_and_0_point_25_l1200_120010

theorem product_of_3_point_6_and_0_point_25 : 3.6 * 0.25 = 0.9 := 
by 
  sorry

end product_of_3_point_6_and_0_point_25_l1200_120010


namespace circle_tangent_x_axis_l1200_120062

theorem circle_tangent_x_axis (x y : ℝ) (h_center : (x, y) = (-3, 4)) (h_tangent : y = 4) :
  ∃ r : ℝ, r = 4 ∧ (∀ x y, (x + 3)^2 + (y - 4)^2 = 16) :=
sorry

end circle_tangent_x_axis_l1200_120062


namespace unique_solution_for_system_l1200_120084

theorem unique_solution_for_system (a : ℝ) :
  (∀ x y z : ℝ, x^2 + y^2 + z^2 + 4 * y = 0 ∧ x + a * y + a * z - a = 0 →
    (a = 2 ∨ a = -2)) :=
by
  intros x y z h
  sorry

end unique_solution_for_system_l1200_120084


namespace min_value_of_quadratic_l1200_120021

theorem min_value_of_quadratic : ∀ x : ℝ, z = x^2 + 16*x + 20 → ∃ m : ℝ, m ≤ z :=
by
  sorry

end min_value_of_quadratic_l1200_120021


namespace minimize_sum_of_squares_of_roots_l1200_120019

theorem minimize_sum_of_squares_of_roots (m : ℝ) (h : 100 - 20 * m ≥ 0) :
  (∀ a b : ℝ, (∀ x : ℝ, 5 * x^2 - 10 * x + m = 0 → x = a ∨ x = b) → (4 - 2 * m / 5) ≥ (4 - 2 * 5 / 5)) :=
by
  sorry

end minimize_sum_of_squares_of_roots_l1200_120019


namespace melissa_points_per_game_l1200_120014

theorem melissa_points_per_game (total_points : ℕ) (games_played : ℕ) (h1 : total_points = 1200) (h2 : games_played = 10) : (total_points / games_played) = 120 := 
by
  -- Here we would insert the proof steps, but we use sorry to represent the omission
  sorry

end melissa_points_per_game_l1200_120014


namespace minimum_value_l1200_120048

/-- 
Given \(a > 0\), \(b > 0\), and \(a + 2b = 1\),
prove that the minimum value of \(\frac{2}{a} + \frac{1}{b}\) is 8.
-/
theorem minimum_value (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + 2 * b = 1) : 
  (∀ a b : ℝ, (a > 0) → (b > 0) → (a + 2 * b = 1) → (∃ c : ℝ, c = 8 ∧ ∀ x y : ℝ, (x = a) → (y = b) → (c ≤ (2 / x) + (1 / y)))) :=
sorry

end minimum_value_l1200_120048


namespace first_and_second_bags_l1200_120055

def bags_apples (A B C : ℕ) : Prop :=
  (A + B + C = 24) ∧ (B + C = 18) ∧ (A + C = 19)

theorem first_and_second_bags (A B C : ℕ) (h : bags_apples A B C) :
  A + B = 11 :=
sorry

end first_and_second_bags_l1200_120055


namespace original_faculty_is_287_l1200_120001

noncomputable def original_faculty (F : ℝ) : Prop :=
  (F * 0.85 * 0.80 = 195)

theorem original_faculty_is_287 : ∃ F : ℝ, original_faculty F ∧ F = 287 := 
by 
  use 287
  sorry

end original_faculty_is_287_l1200_120001


namespace total_surface_area_of_prism_l1200_120047

-- Define the conditions of the problem
def sphere_radius (R : ℝ) := R > 0
def prism_circumscribed_around_sphere (R : ℝ) := True  -- Placeholder as the concept assertion, actual geometry handling not needed here
def prism_height (R : ℝ) := 2 * R

-- Define the main theorem to be proved
theorem total_surface_area_of_prism (R : ℝ) (hR : sphere_radius R) (hCircumscribed : prism_circumscribed_around_sphere R) (hHeight : prism_height R = 2 * R) : 
  ∃ (S : ℝ), S = 12 * R^2 * Real.sqrt 3 :=
sorry

end total_surface_area_of_prism_l1200_120047


namespace initial_slices_ham_l1200_120038

def total_sandwiches : ℕ := 50
def slices_per_sandwich : ℕ := 3
def additional_slices_needed : ℕ := 119

-- Calculate the total number of slices needed to make 50 sandwiches.
def total_slices_needed : ℕ := total_sandwiches * slices_per_sandwich

-- Prove the initial number of slices of ham Anna has.
theorem initial_slices_ham : total_slices_needed - additional_slices_needed = 31 := by
  sorry

end initial_slices_ham_l1200_120038


namespace difference_in_total_cost_l1200_120096

theorem difference_in_total_cost
  (item_price : ℝ := 15)
  (tax_rate1 : ℝ := 0.08)
  (tax_rate2 : ℝ := 0.072)
  (discount : ℝ := 0.005)
  (correct_difference : ℝ := 0.195) :
  let discounted_tax_rate := tax_rate2 - discount
  let total_price_with_tax_rate1 := item_price * (1 + tax_rate1)
  let total_price_with_discounted_tax_rate := item_price * (1 + discounted_tax_rate)
  total_price_with_tax_rate1 - total_price_with_discounted_tax_rate = correct_difference := by
  sorry

end difference_in_total_cost_l1200_120096


namespace sum_of_integers_from_1_to_10_l1200_120081

theorem sum_of_integers_from_1_to_10 :
  (Finset.range 11).sum id = 55 :=
sorry

end sum_of_integers_from_1_to_10_l1200_120081


namespace find_m_l1200_120002

theorem find_m (m : ℝ) (h : 2 / m = (m + 1) / 3) : m = -3 := by
  sorry

end find_m_l1200_120002


namespace certain_number_divisibility_l1200_120068

theorem certain_number_divisibility (n : ℕ) (p : ℕ) (h : p = 1) (h2 : 4864 * 9 * n % 12 = 0) : n = 43776 :=
by {
  sorry
}

end certain_number_divisibility_l1200_120068


namespace cube_minus_self_divisible_by_6_l1200_120011

theorem cube_minus_self_divisible_by_6 (n : ℕ) : 6 ∣ (n^3 - n) :=
sorry

end cube_minus_self_divisible_by_6_l1200_120011


namespace sin_of_angle_l1200_120067

theorem sin_of_angle (α : ℝ) (x y : ℝ) (h1 : x = -3) (h2 : y = -4) (r : ℝ) (hr : r = Real.sqrt (x^2 + y^2)) : 
  Real.sin α = -4 / r := 
by
  -- Definitions
  let y := -4
  let x := -3
  let r := Real.sqrt (x^2 + y^2)
  -- Proof
  sorry

end sin_of_angle_l1200_120067


namespace total_students_is_45_l1200_120023

def num_students_in_class 
  (excellent_chinese : ℕ) 
  (excellent_math : ℕ) 
  (excellent_both : ℕ) 
  (no_excellent : ℕ) : ℕ :=
  excellent_chinese + excellent_math - excellent_both + no_excellent

theorem total_students_is_45 
  (h1 : excellent_chinese = 15)
  (h2 : excellent_math = 18)
  (h3 : excellent_both = 8)
  (h4 : no_excellent = 20) : 
  num_students_in_class excellent_chinese excellent_math excellent_both no_excellent = 45 := 
  by 
    sorry

end total_students_is_45_l1200_120023


namespace find_x_given_k_l1200_120095

-- Define the equation under consideration
def equation (x : ℝ) : Prop := (x - 3) / (x - 4) = (x - 5) / (x - 8)

theorem find_x_given_k {k : ℝ} (h : k = 7) : ∀ x : ℝ, x ≠ 4 ∧ x ≠ 8 → equation x → x = 2 :=
by
  intro x hx h_eq
  sorry

end find_x_given_k_l1200_120095


namespace darnel_jogging_l1200_120072

variable (j s : ℝ)

theorem darnel_jogging :
  s = 0.875 ∧ s = j + 0.125 → j = 0.750 :=
by
  intros h
  have h1 : s = 0.875 := h.1
  have h2 : s = j + 0.125 := h.2
  sorry

end darnel_jogging_l1200_120072


namespace brooke_kent_ratio_l1200_120016

theorem brooke_kent_ratio :
  ∀ (alison brooke brittany kent : ℕ),
  (kent = 1000) →
  (alison = 4000) →
  (alison = brittany / 2) →
  (brittany = 4 * brooke) →
  brooke / kent = 2 :=
by
  intros alison brooke brittany kent kent_val alison_val alison_brittany brittany_brooke
  sorry

end brooke_kent_ratio_l1200_120016


namespace number_wall_problem_l1200_120037

theorem number_wall_problem (m : ℤ) : 
  ((m + 5) + 16 + 18 = 56) → (m = 17) :=
by
  sorry

end number_wall_problem_l1200_120037


namespace exists_intersecting_line_l1200_120045

/-- Represents a segment as a pair of endpoints in a 2D plane. -/
structure Segment where
  x : ℝ
  y1 : ℝ
  y2 : ℝ

open Segment

/-- Given several parallel segments with the property that for any three of these segments, 
there exists a line that intersects all three of them, prove that 
there is a line that intersects all the segments. -/
theorem exists_intersecting_line (segments : List Segment)
  (h : ∀ s1 s2 s3 : Segment, s1 ∈ segments → s2 ∈ segments → s3 ∈ segments → 
       ∃ a b : ℝ, (s1.y1 <= a * s1.x + b) ∧ (a * s1.x + b <= s1.y2) ∧ 
                   (s2.y1 <= a * s2.x + b) ∧ (a * s2.x + b <= s2.y2) ∧ 
                   (s3.y1 <= a * s3.x + b) ∧ (a * s3.x + b <= s3.y2)) :
  ∃ a b : ℝ, ∀ s : Segment, s ∈ segments → (s.y1 <= a * s.x + b) ∧ (a * s.x + b <= s.y2) := 
sorry

end exists_intersecting_line_l1200_120045


namespace sum_of_cubes_divisible_by_middle_integer_l1200_120030

theorem sum_of_cubes_divisible_by_middle_integer (a : ℤ) : 
  (a - 1)^3 + a^3 + (a + 1)^3 ∣ 3 * a :=
sorry

end sum_of_cubes_divisible_by_middle_integer_l1200_120030


namespace simplify_sqrt_l1200_120017

noncomputable def simplify_expression : ℝ :=
  Real.sqrt (8 + 6 * Real.sqrt 2) + Real.sqrt (8 - 6 * Real.sqrt 2)

theorem simplify_sqrt (h : simplify_expression = 2 * Real.sqrt 6) : 
    Real.sqrt (8 + 6 * Real.sqrt 2) + Real.sqrt (8 - 6 * Real.sqrt 2) = 2 * Real.sqrt 6 :=
  by sorry

end simplify_sqrt_l1200_120017


namespace fraction_of_top_10_lists_l1200_120077

theorem fraction_of_top_10_lists (total_members : ℝ) (min_top_10_lists : ℝ) (fraction : ℝ) 
  (h1 : total_members = 765) (h2 : min_top_10_lists = 191.25) : 
    min_top_10_lists / total_members = fraction := by
  have h3 : fraction = 0.25 := by sorry
  rw [h1, h2, h3]
  sorry

end fraction_of_top_10_lists_l1200_120077


namespace distance_to_lake_l1200_120056

theorem distance_to_lake (d : ℝ) :
  ¬ (d ≥ 10) → ¬ (d ≤ 9) → d ≠ 7 → d ∈ Set.Ioo 9 10 :=
by
  intros h1 h2 h3
  sorry

end distance_to_lake_l1200_120056


namespace perimeter_of_square_C_l1200_120022

theorem perimeter_of_square_C (a b : ℝ) 
  (hA : 4 * a = 16) 
  (hB : 4 * b = 32) : 
  4 * (a + b) = 48 := by
  sorry

end perimeter_of_square_C_l1200_120022


namespace train_passing_time_l1200_120059

/-- The problem defines a train of length 110 meters traveling at 40 km/hr, 
    passing a man who is running at 5 km/hr in the opposite direction.
    We want to prove that the time it takes for the train to pass the man is 8.8 seconds. -/
theorem train_passing_time :
  ∀ (train_length : ℕ) (train_speed man_speed : ℕ), 
  train_length = 110 → train_speed = 40 → man_speed = 5 →
  (∃ time : ℚ, time = 8.8) :=
by
  intros train_length train_speed man_speed h_train_length h_train_speed h_man_speed
  sorry

end train_passing_time_l1200_120059


namespace sum_of_interior_angles_l1200_120051

theorem sum_of_interior_angles (h_triangle : ∀ (a b c : ℝ), a + b + c = 180)
    (h_quadrilateral : ∀ (a b c d : ℝ), a + b + c + d = 360) :
  (∀ (n : ℕ), n ≥ 3 → ∀ (angles : Fin n → ℝ), (Finset.univ.sum angles) = (n-2) * 180) :=
by
  intro n h_n angles
  sorry

end sum_of_interior_angles_l1200_120051


namespace sum_of_ages_of_sarahs_friends_l1200_120089

noncomputable def sum_of_ages (a b c : ℕ) : ℕ := a + b + c

theorem sum_of_ages_of_sarahs_friends (a b c : ℕ) (h_distinct : ∀ (x y z : ℕ), x ≠ y ∧ y ≠ z ∧ x ≠ z)
  (h_single_digits : ∀ (x y z : ℕ), x < 10 ∧ y < 10 ∧ z < 10)
  (h_product_36 : ∃ (x y : ℕ), x * y = 36 ∧ x ≠ y)
  (h_factor_36 : ∀ (x y z : ℕ), x ∣ 36 ∧ y ∣ 36 ∧ z ∣ 36) :
  ∃ (a b c : ℕ), sum_of_ages a b c = 16 := 
sorry

end sum_of_ages_of_sarahs_friends_l1200_120089


namespace fish_count_l1200_120078

theorem fish_count (T : ℕ) :
  (T > 10 ∧ T ≤ 18) ∧ ((T > 18 ∧ T > 15 ∧ ¬(T > 10)) ∨ (¬(T > 18) ∧ T > 15 ∧ T > 10) ∨ (T > 18 ∧ ¬(T > 15) ∧ T > 10)) →
  T = 16 ∨ T = 17 ∨ T = 18 :=
sorry

end fish_count_l1200_120078


namespace age_difference_l1200_120071

theorem age_difference (A B C : ℕ) (h : A + B = B + C + 13) : A = C + 13 :=
by
  sorry

end age_difference_l1200_120071


namespace number_of_zeros_of_h_l1200_120080

noncomputable def f (x : ℝ) : ℝ := 2 * x
noncomputable def g (x : ℝ) : ℝ := 3 - x^2
noncomputable def h (x : ℝ) : ℝ := f x - g x

theorem number_of_zeros_of_h : ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ h x1 = 0 ∧ h x2 = 0 ∧ ∀ x, h x = 0 → (x = x1 ∨ x = x2) :=
by
  sorry

end number_of_zeros_of_h_l1200_120080


namespace river_width_l1200_120043

def bridge_length : ℕ := 295
def additional_length : ℕ := 192
def total_width : ℕ := 487

theorem river_width (h1 : bridge_length = 295) (h2 : additional_length = 192) : bridge_length + additional_length = total_width := by
  sorry

end river_width_l1200_120043


namespace mason_grandmother_age_l1200_120032

theorem mason_grandmother_age (mason_age: ℕ) (sydney_age: ℕ) (father_age: ℕ) (grandmother_age: ℕ)
  (h1: mason_age = 20)
  (h2: mason_age * 3 = sydney_age)
  (h3: sydney_age + 6 = father_age)
  (h4: father_age * 2 = grandmother_age) : 
  grandmother_age = 132 :=
by
  sorry

end mason_grandmother_age_l1200_120032


namespace apples_harvested_l1200_120074

variable (A P : ℕ)
variable (h₁ : P = 3 * A) (h₂ : P - A = 120)

theorem apples_harvested : A = 60 := 
by
  -- proof will go here
  sorry

end apples_harvested_l1200_120074


namespace jack_total_cost_l1200_120070

def plan_base_cost : ℕ := 25

def cost_per_text : ℕ := 8

def free_hours : ℕ := 25

def cost_per_extra_minute : ℕ := 10

def texts_sent : ℕ := 150

def hours_talked : ℕ := 26

def total_cost (base_cost : ℕ) (texts_sent : ℕ) (cost_per_text : ℕ) (hours_talked : ℕ) 
               (free_hours : ℕ) (cost_per_extra_minute : ℕ) : ℕ :=
  base_cost + (texts_sent * cost_per_text) / 100 + 
  ((hours_talked - free_hours) * 60 * cost_per_extra_minute) / 100

theorem jack_total_cost : 
  total_cost plan_base_cost texts_sent cost_per_text hours_talked free_hours cost_per_extra_minute = 43 :=
by
  sorry

end jack_total_cost_l1200_120070


namespace tim_younger_than_jenny_l1200_120060

def tim_age : ℕ := 5
def rommel_age : ℕ := 3 * tim_age
def jenny_age : ℕ := rommel_age + 2
def combined_ages_rommel_jenny : ℕ := rommel_age + jenny_age
def uncle_age : ℕ := 2 * combined_ages_rommel_jenny
noncomputable def aunt_age : ℝ := (uncle_age + jenny_age : ℕ) / 2

theorem tim_younger_than_jenny : jenny_age - tim_age = 12 :=
by {
  -- Placeholder proof
  sorry
}

end tim_younger_than_jenny_l1200_120060


namespace solve_quadratic_eq1_solve_quadratic_eq2_l1200_120058

-- Define the first equation
theorem solve_quadratic_eq1 (x : ℝ) : x^2 - 6 * x - 6 = 0 ↔ x = 3 + Real.sqrt 15 ∨ x = 3 - Real.sqrt 15 := by
  sorry

-- Define the second equation
theorem solve_quadratic_eq2 (x : ℝ) : 2 * x^2 - 3 * x + 1 = 0 ↔ x = 1 ∨ x = 1 / 2 := by
  sorry

end solve_quadratic_eq1_solve_quadratic_eq2_l1200_120058


namespace problem_l1200_120012

theorem problem 
  (x : ℝ) 
  (h1 : x ∈ Set.Icc (-3 : ℝ) 3) 
  (h2 : x ≠ -5/3) : 
  (4 * x ^ 2 + 2) / (5 + 3 * x) ≥ 1 ↔ x ∈ (Set.Icc (-3) (-3/4) ∪ Set.Icc 1 3) :=
sorry

end problem_l1200_120012


namespace paving_cost_is_16500_l1200_120035

-- Define the given conditions
def length : ℝ := 5.5
def width : ℝ := 3.75
def rate_per_sq_meter : ℝ := 800

-- Define the area calculation
def area (L W : ℝ) : ℝ := L * W

-- Define the cost calculation
def cost (A rate : ℝ) : ℝ := A * rate

-- The theorem to prove that the cost of paving the floor is 16500
theorem paving_cost_is_16500 : cost (area length width) rate_per_sq_meter = 16500 :=
by
  -- Proof is omitted here
  sorry

end paving_cost_is_16500_l1200_120035


namespace total_profit_l1200_120075

variable (InvestmentA InvestmentB InvestmentTimeA InvestmentTimeB ShareA : ℝ)
variable (hA : InvestmentA = 150)
variable (hB : InvestmentB = 200)
variable (hTimeA : InvestmentTimeA = 12)
variable (hTimeB : InvestmentTimeB = 6)
variable (hShareA : ShareA = 60)

theorem total_profit (TotalProfit : ℝ) :
  (ShareA / 3) * 5 = TotalProfit := 
by
  sorry

end total_profit_l1200_120075


namespace carolyn_total_monthly_practice_l1200_120040

-- Define the constants and relationships given in the problem
def daily_piano_practice : ℕ := 20
def times_violin_practice : ℕ := 3
def days_week : ℕ := 6
def weeks_month : ℕ := 4
def daily_violin_practice : ℕ := daily_piano_practice * times_violin_practice
def total_daily_practice : ℕ := daily_piano_practice + daily_violin_practice
def weekly_practice_time : ℕ := total_daily_practice * days_week
def monthly_practice_time : ℕ := weekly_practice_time * weeks_month

-- The proof statement with the final result
theorem carolyn_total_monthly_practice : monthly_practice_time = 1920 := by
  sorry

end carolyn_total_monthly_practice_l1200_120040


namespace part1_part2_l1200_120086

noncomputable def choose (n : ℕ) (k : ℕ) : ℕ :=
  n.choose k

theorem part1 :
  let internal_medicine_doctors := 12
  let surgeons := 8
  let total_doctors := internal_medicine_doctors + surgeons
  let team_size := 5
  let doctors_left := total_doctors - 1 - 1 -- as one internal medicine must participate and one surgeon cannot
  choose doctors_left (team_size - 1) = 3060 := by
  sorry

theorem part2 :
  let internal_medicine_doctors := 12
  let surgeons := 8
  let total_doctors := internal_medicine_doctors + surgeons
  let team_size := 5
  let only_internal_medicine := choose internal_medicine_doctors team_size
  let only_surgeons := choose surgeons team_size
  let total_ways := choose total_doctors team_size
  total_ways - only_internal_medicine - only_surgeons = 14656 := by
  sorry

end part1_part2_l1200_120086


namespace distance_between_trees_l1200_120087

theorem distance_between_trees (n : ℕ) (len : ℝ) (d : ℝ) 
  (h1 : n = 26) 
  (h2 : len = 400) 
  (h3 : len / (n - 1) = d) : 
  d = 16 :=
by
  sorry

end distance_between_trees_l1200_120087


namespace range_a_l1200_120004

noncomputable def A (a : ℝ) : Set ℝ := {x | 2 * a + 1 ≤ x ∧ x ≤ 3 * a - 5}

noncomputable def B : Set ℝ := {x | x < -1 ∨ x > 16}

theorem range_a (a : ℝ) : (A a ∩ B = A a) → (a < 6 ∨ a > 7.5) :=
by
  intro h
  sorry

end range_a_l1200_120004


namespace store_revenue_after_sale_l1200_120034

/--
A store has 2000 items, each normally selling for $50. 
They offer an 80% discount and manage to sell 90% of the items. 
The store owes $15,000 to creditors. Prove that the store has $3,000 left after the sale.
-/
theorem store_revenue_after_sale :
  let items := 2000
  let retail_price := 50
  let discount := 0.8
  let sale_percentage := 0.9
  let debt := 15000
  let items_sold := items * sale_percentage
  let discount_amount := retail_price * discount
  let sale_price_per_item := retail_price - discount_amount
  let total_revenue := items_sold * sale_price_per_item
  let money_left := total_revenue - debt
  money_left = 3000 :=
by
  sorry

end store_revenue_after_sale_l1200_120034


namespace larger_number_l1200_120033

theorem larger_number (x y : ℕ) (h1 : x + y = 40) (h2 : x - y = 4) : x = 22 := by
  sorry

end larger_number_l1200_120033


namespace gretchen_charge_per_drawing_l1200_120024

-- Given conditions
def sold_on_Saturday : ℕ := 24
def sold_on_Sunday : ℕ := 16
def total_amount : ℝ := 800
def total_drawings := sold_on_Saturday + sold_on_Sunday

-- Assertion to prove
theorem gretchen_charge_per_drawing (x : ℝ) (h : total_drawings * x = total_amount) : x = 20 :=
by
  sorry

end gretchen_charge_per_drawing_l1200_120024


namespace inequality_transitivity_l1200_120027

theorem inequality_transitivity (a b c : ℝ) (h : a > b) : 
  a + c > b + c :=
sorry

end inequality_transitivity_l1200_120027


namespace hexagon_angle_R_l1200_120052

theorem hexagon_angle_R (F I G U R E : ℝ) 
  (h1 : F = I ∧ I = R ∧ R = E)
  (h2 : G + U = 180) 
  (sum_angles_hexagon : F + I + G + U + R + E = 720) : 
  R = 135 :=
by sorry

end hexagon_angle_R_l1200_120052


namespace not_polynomial_option_B_l1200_120003

-- Definitions
def is_polynomial (expr : String) : Prop :=
  -- Assuming we have a function that determines if a given string expression is a polynomial.
  sorry

def option_A : String := "m+n"
def option_B : String := "x=1"
def option_C : String := "xy"
def option_D : String := "0"

-- Problem Statement
theorem not_polynomial_option_B : ¬ is_polynomial option_B := 
sorry

end not_polynomial_option_B_l1200_120003


namespace total_students_l1200_120029

theorem total_students (p q r s : ℕ) 
  (h1 : 1 < p)
  (h2 : p < q)
  (h3 : q < r)
  (h4 : r < s)
  (h5 : p * q * r * s = 1365) :
  p + q + r + s = 28 :=
sorry

end total_students_l1200_120029


namespace tangent_line_ln_x_xsq_l1200_120061

theorem tangent_line_ln_x_xsq (x y : ℝ) (h_curve : y = Real.log x + x^2) (h_point : (x, y) = (1, 1)) :
  3 * x - y - 2 = 0 :=
sorry

end tangent_line_ln_x_xsq_l1200_120061


namespace concentration_third_flask_l1200_120085

-- Definitions based on the conditions in the problem
def first_flask_acid := 10
def second_flask_acid := 20
def third_flask_acid := 30
def concentration_first_flask := 0.05
def concentration_second_flask := 70 / 300

-- Problem statement in Lean
theorem concentration_third_flask (W1 W2 : ℝ) (h1 : 10 / (10 + W1) = 0.05)
 (h2 : 20 / (20 + W2) = 70 / 300):
  (30 / (30 + (W1 + W2))) * 100 = 10.5 := 
sorry

end concentration_third_flask_l1200_120085


namespace total_items_correct_l1200_120036

-- Defining the number of each type of items ordered by Betty
def slippers := 6
def lipstick := 4
def hair_color := 8

-- The total number of items ordered by Betty
def total_items := slippers + lipstick + hair_color

-- The statement asserting that the total number of items is 18
theorem total_items_correct : total_items = 18 := 
by 
  -- sorry allows us to skip the proof
  sorry

end total_items_correct_l1200_120036


namespace sqrt_of_16_l1200_120026

theorem sqrt_of_16 (x : ℝ) (hx : x^2 = 16) : x = 4 ∨ x = -4 := 
by
  sorry

end sqrt_of_16_l1200_120026


namespace sqrt_mul_l1200_120091

theorem sqrt_mul (h₁ : 0 ≤ 3) (h₂ : 0 ≤ 12) : Real.sqrt 3 * Real.sqrt 12 = 6 := by
  sorry

end sqrt_mul_l1200_120091


namespace min_b_for_quadratic_factorization_l1200_120092

theorem min_b_for_quadratic_factorization : ∃ b : ℕ, b = 84 ∧ ∃ p q : ℤ, p + q = b ∧ p * q = 1760 :=
by
  sorry

end min_b_for_quadratic_factorization_l1200_120092


namespace evaluate_expression_l1200_120054

theorem evaluate_expression : -(16 / 4 * 7 - 50 + 5 * 7) = -13 :=
by
  sorry

end evaluate_expression_l1200_120054


namespace no_outliers_in_dataset_l1200_120018

theorem no_outliers_in_dataset :
  let D := [7, 20, 34, 34, 40, 42, 42, 44, 52, 58]
  let Q1 := 34
  let Q3 := 44
  let IQR := Q3 - Q1
  let lower_threshold := Q1 - 1.5 * IQR
  let upper_threshold := Q3 + 1.5 * IQR
  (∀ x ∈ D, x ≥ lower_threshold) ∧ (∀ x ∈ D, x ≤ upper_threshold) →
  ∀ x ∈ D, ¬(x < lower_threshold ∨ x > upper_threshold) :=
by 
  sorry

end no_outliers_in_dataset_l1200_120018


namespace cost_per_night_l1200_120006

variable (x : ℕ)

theorem cost_per_night (h : 3 * x - 100 = 650) : x = 250 :=
sorry

end cost_per_night_l1200_120006


namespace find_h_l1200_120025

-- Define the polynomial f(x)
def f (x : ℤ) := x^4 - 2 * x^3 + x - 1

-- Define the condition that f(x) + h(x) = 3x^2 + 5x - 4
def condition (f h : ℤ → ℤ) := ∀ x, f x + h x = 3 * x^2 + 5 * x - 4

-- Define the solution for h(x) to be proved
def h_solution (x : ℤ) := -x^4 + 2 * x^3 + 3 * x^2 + 4 * x - 3

-- State the theorem to be proved
theorem find_h (h : ℤ → ℤ) (H : condition f h) : h = h_solution :=
by
  sorry

end find_h_l1200_120025


namespace probability_two_students_next_to_each_other_l1200_120039

theorem probability_two_students_next_to_each_other : (2 * Nat.factorial 9) / Nat.factorial 10 = 1 / 5 :=
by
  sorry

end probability_two_students_next_to_each_other_l1200_120039


namespace problem_statement_l1200_120065

def h (x : ℝ) : ℝ := 3 * x + 2
def k (x : ℝ) : ℝ := 2 * x - 3

theorem problem_statement : (h (k (h 3))) / (k (h (k 3))) = 59 / 19 := by
  sorry

end problem_statement_l1200_120065


namespace total_marbles_l1200_120076

-- Definitions based on given conditions
def ratio_white := 2
def ratio_purple := 3
def ratio_red := 5
def ratio_blue := 4
def ratio_green := 6
def blue_marbles := 24

-- Definition of sum of ratio parts
def sum_of_ratio_parts := ratio_white + ratio_purple + ratio_red + ratio_blue + ratio_green

-- Definition of ratio of blue marbles to total
def ratio_blue_to_total := ratio_blue / sum_of_ratio_parts

-- Proof goal: total number of marbles
theorem total_marbles : blue_marbles / ratio_blue_to_total = 120 := by
  sorry

end total_marbles_l1200_120076


namespace find_xy_l1200_120041

theorem find_xy (x y : ℝ) :
  (x - 8) ^ 2 + (y - 9) ^ 2 + (x - y) ^ 2 = 1 / 3 ↔ 
  (x = 25 / 3 ∧ y = 26 / 3) :=
by
  sorry

end find_xy_l1200_120041


namespace parallel_planes_perpendicular_planes_l1200_120064

variables {A1 B1 C1 D1 A2 B2 C2 D2 : ℝ}

-- Parallelism Condition
theorem parallel_planes (h₁ : A1 ≠ 0) (h₂ : B1 ≠ 0) (h₃ : C1 ≠ 0) (h₄ : A2 ≠ 0) (h₅ : B2 ≠ 0) (h₆ : C2 ≠ 0) :
  (A1 / A2 = B1 / B2 ∧ B1 / B2 = C1 / C2) ↔ (∃ k : ℝ, (A1 = k * A2) ∧ (B1 = k * B2) ∧ (C1 = k * C2)) :=
sorry

-- Perpendicularity Condition
theorem perpendicular_planes :
  A1 * A2 + B1 * B2 + C1 * C2 = 0 :=
sorry

end parallel_planes_perpendicular_planes_l1200_120064


namespace initial_passengers_l1200_120090

theorem initial_passengers (P : ℕ) (H1 : P - 263 + 419 = 725) : P = 569 :=
by
  sorry

end initial_passengers_l1200_120090


namespace three_digit_number_l1200_120007

theorem three_digit_number (x y z : ℕ) 
  (h1: z^2 = x * y)
  (h2: y = (x + z) / 6)
  (h3: x - z = 4) :
  100 * x + 10 * y + z = 824 := 
by sorry

end three_digit_number_l1200_120007


namespace product_of_primes_is_even_l1200_120079

-- Define the conditions for P and Q to cover P, Q, P-Q, and P+Q being prime and positive
def is_prime (n : ℕ) : Prop := ¬ (n = 0 ∨ n = 1) ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem product_of_primes_is_even {P Q : ℕ} (hP : is_prime P) (hQ : is_prime Q) 
  (hPQ_diff : is_prime (P - Q)) (hPQ_sum : is_prime (P + Q)) 
  (hPosP : P > 0) (hPosQ : Q > 0) 
  (hPosPQ_diff : P - Q > 0) (hPosPQ_sum : P + Q > 0) : 
  ∃ k : ℕ, P * Q * (P - Q) * (P + Q) = 2 * k := 
sorry

end product_of_primes_is_even_l1200_120079


namespace correct_option_B_l1200_120057

theorem correct_option_B (x : ℝ) : (1 - x)^2 = 1 - 2 * x + x^2 :=
sorry

end correct_option_B_l1200_120057


namespace first_investment_percentage_l1200_120083

variable (P : ℝ)
variable (x : ℝ := 1400)  -- investment amount in the first investment
variable (y : ℝ := 600)   -- investment amount at 8 percent
variable (income_difference : ℝ := 92)
variable (total_investment : ℝ := 2000)
variable (rate_8_percent : ℝ := 0.08)
variable (exceed_by : ℝ := 92)

theorem first_investment_percentage :
  P * x - rate_8_percent * y = exceed_by →
  total_investment = x + y →
  P = 0.10 :=
by
  -- Solution steps can be filled here if needed
  sorry

end first_investment_percentage_l1200_120083


namespace triangle_area_l1200_120005

theorem triangle_area {a c : ℝ} (h_a : a = 3 * Real.sqrt 3) (h_c : c = 2) (angle_B : ℝ) (h_B : angle_B = Real.pi / 3) : 
  (1 / 2) * a * c * Real.sin angle_B = 9 / 2 :=
by
  rw [h_a, h_c, h_B]
  sorry

end triangle_area_l1200_120005


namespace days_of_earning_l1200_120066

theorem days_of_earning (T D d : ℕ) (hT : T = 165) (hD : D = 33) (h : d = T / D) :
  d = 5 :=
by sorry

end days_of_earning_l1200_120066


namespace race_dead_heat_l1200_120088

theorem race_dead_heat 
  (L Vb : ℝ) 
  (speed_a : ℝ := (16/15) * Vb)
  (speed_c : ℝ := (20/15) * Vb) 
  (time_a : ℝ := L / speed_a)
  (time_b : ℝ := L / Vb)
  (time_c : ℝ := L / speed_c) :
  (1 / (16 / 15) = 3 / 4) → 
  (1 - 3 / 4) = 1 / 4 :=
by 
  sorry

end race_dead_heat_l1200_120088


namespace combined_sale_price_correct_l1200_120082

-- Define constants for purchase costs of items A, B, and C.
def purchase_cost_A : ℝ := 650
def purchase_cost_B : ℝ := 350
def purchase_cost_C : ℝ := 400

-- Define profit percentages for items A, B, and C.
def profit_percentage_A : ℝ := 0.40
def profit_percentage_B : ℝ := 0.25
def profit_percentage_C : ℝ := 0.30

-- Define the desired sale prices for items A, B, and C based on profit margins.
def sale_price_A : ℝ := purchase_cost_A * (1 + profit_percentage_A)
def sale_price_B : ℝ := purchase_cost_B * (1 + profit_percentage_B)
def sale_price_C : ℝ := purchase_cost_C * (1 + profit_percentage_C)

-- Calculate the combined sale price for all three items.
def combined_sale_price : ℝ := sale_price_A + sale_price_B + sale_price_C

-- The theorem stating that the combined sale price for all three items is $1867.50.
theorem combined_sale_price_correct :
  combined_sale_price = 1867.50 := 
sorry

end combined_sale_price_correct_l1200_120082


namespace log5_of_15625_l1200_120063

-- Define the logarithm function in base 5
def log_base_5 (n : ℕ) : ℕ := sorry

-- State the theorem with the given condition and conclude the desired result
theorem log5_of_15625 : log_base_5 15625 = 6 :=
by sorry

end log5_of_15625_l1200_120063


namespace find_a_l1200_120031

theorem find_a (a : ℝ) (h : (1 / Real.log 2 / Real.log a) + (1 / Real.log 3 / Real.log a) + (1 / Real.log 5 / Real.log a) = 2) : a = Real.sqrt 30 := 
by 
  sorry

end find_a_l1200_120031


namespace class_8_3_final_score_is_correct_l1200_120044

def class_8_3_singing_quality : ℝ := 92
def class_8_3_spirit : ℝ := 80
def class_8_3_coordination : ℝ := 70

def final_score (singing_quality spirit coordination : ℝ) : ℝ :=
  0.4 * singing_quality + 0.3 * spirit + 0.3 * coordination

theorem class_8_3_final_score_is_correct :
  final_score class_8_3_singing_quality class_8_3_spirit class_8_3_coordination = 81.8 :=
by
  sorry

end class_8_3_final_score_is_correct_l1200_120044


namespace pipe_A_fill_time_l1200_120098

theorem pipe_A_fill_time (x : ℝ) (h₁ : x > 0) (h₂ : 1 / x + 1 / 15 = 1 / 6) : x = 10 :=
by
  sorry

end pipe_A_fill_time_l1200_120098


namespace emily_total_spent_l1200_120046

-- Define the given conditions.
def cost_per_flower : ℕ := 3
def num_roses : ℕ := 2
def num_daisies : ℕ := 2

-- Calculate the total number of flowers and the total cost.
def total_flowers : ℕ := num_roses + num_daisies
def total_cost : ℕ := total_flowers * cost_per_flower

-- Statement: Prove that Emily spent 12 dollars.
theorem emily_total_spent : total_cost = 12 := by
  sorry

end emily_total_spent_l1200_120046


namespace find_m_l1200_120013

def vec_a (m : ℝ) : ℝ × ℝ := (1, 2 * m)
def vec_b (m : ℝ) : ℝ × ℝ := (m + 1, 1)

def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

theorem find_m (m : ℝ) : dot_product (vec_a m) (vec_b m) = 0 ↔ m = -1/3 := by 
  sorry

end find_m_l1200_120013


namespace polygon_parallel_edges_l1200_120020

theorem polygon_parallel_edges (n : ℕ) (h : n > 2) :
  (∃ i j, i ≠ j ∧ (i + 1) % n = (j + 1) % n) ↔ (∃ k, n = 2 * k) :=
  sorry

end polygon_parallel_edges_l1200_120020


namespace value_of_a_l1200_120069

theorem value_of_a (a : ℝ) : (a^2 - 4) / (a - 2) = 0 → a ≠ 2 → a = -2 :=
by 
  intro h1 h2
  sorry

end value_of_a_l1200_120069


namespace emily_small_gardens_l1200_120000

theorem emily_small_gardens (total_seeds : ℕ) (big_garden_seeds : ℕ) (seeds_per_small_garden : ℕ) (num_small_gardens : ℕ) :
  total_seeds = 41 →
  big_garden_seeds = 29 →
  seeds_per_small_garden = 4 →
  num_small_gardens = (total_seeds - big_garden_seeds) / seeds_per_small_garden →
  num_small_gardens = 3 :=
by
  intros h_total h_big h_seeds_per_small h_num_small
  rw [h_total, h_big, h_seeds_per_small] at h_num_small
  exact h_num_small

end emily_small_gardens_l1200_120000
