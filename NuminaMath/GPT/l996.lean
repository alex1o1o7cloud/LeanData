import Mathlib

namespace NUMINAMATH_GPT_amount_spent_on_tracksuit_l996_99699

-- Definitions based on the conditions
def original_price (x : ℝ) := x
def discount_rate : ℝ := 0.20
def savings : ℝ := 30
def actual_spent (x : ℝ) := 0.8 * x

-- Theorem statement derived from the proof translation
theorem amount_spent_on_tracksuit (x : ℝ) (h : (original_price x) * discount_rate = savings) :
  actual_spent x = 120 :=
by
  sorry

end NUMINAMATH_GPT_amount_spent_on_tracksuit_l996_99699


namespace NUMINAMATH_GPT_door_X_is_inner_sanctuary_l996_99669

  variable (X Y Z W : Prop)
  variable (A B C D E F G H : Prop)
  variable (is_knight : Prop → Prop)

  -- Each statement according to the conditions in the problem.
  variable (stmt_A : X)
  variable (stmt_B : Y ∨ Z)
  variable (stmt_C : is_knight A ∧ is_knight B)
  variable (stmt_D : X ∧ Y)
  variable (stmt_E : X ∧ Y)
  variable (stmt_F : is_knight D ∨ is_knight E)
  variable (stmt_G : is_knight C → is_knight F)
  variable (stmt_H : is_knight G ∧ is_knight H → is_knight A)

  theorem door_X_is_inner_sanctuary :
    is_knight A → is_knight B → is_knight C → is_knight D → is_knight E → is_knight F → is_knight G → is_knight H → X :=
  sorry
  
end NUMINAMATH_GPT_door_X_is_inner_sanctuary_l996_99669


namespace NUMINAMATH_GPT_hexagon_inscribed_in_square_area_l996_99639

noncomputable def hexagon_area (side_length : ℝ) : ℝ :=
  (3 * Real.sqrt 3 / 2) * side_length^2

theorem hexagon_inscribed_in_square_area (AB BC : ℝ) (BDEF_square : BDEF_is_square) (hAB : AB = 2) (hBC : BC = 2) :
  hexagon_area (2 * Real.sqrt 2) = 12 * Real.sqrt 3 :=
by
  sorry

-- Definitions to assume the necessary conditions in the theorem (placeholders)
-- Assuming a structure of BDEF_is_square to represent the property that BDEF is a square
structure BDEF_is_square :=
(square : Prop)

end NUMINAMATH_GPT_hexagon_inscribed_in_square_area_l996_99639


namespace NUMINAMATH_GPT_choose_starting_team_l996_99625

-- Definitions derived from the conditions
def team_size : ℕ := 18
def selected_goalie (n : ℕ) : ℕ := n
def selected_players (m : ℕ) (k : ℕ) : ℕ := Nat.choose m k

-- The number of ways to choose the starting team
theorem choose_starting_team :
  let n := team_size
  let k := 7
  selected_goalie n * selected_players (n - 1) k = 222768 :=
by
  simp only [team_size, selected_goalie, selected_players]
  sorry

end NUMINAMATH_GPT_choose_starting_team_l996_99625


namespace NUMINAMATH_GPT_scientific_notation_15510000_l996_99603

theorem scientific_notation_15510000 : 
  ∃ (a : ℝ) (n : ℤ), (1 ≤ |a| ∧ |a| < 10) ∧ 15510000 = a * 10^n ∧ a = 1.551 ∧ n = 7 :=
by
  sorry

end NUMINAMATH_GPT_scientific_notation_15510000_l996_99603


namespace NUMINAMATH_GPT_range_distance_PQ_l996_99693

noncomputable def point_P (α : ℝ) : ℝ × ℝ × ℝ := (3 * Real.cos α, 3 * Real.sin α, 1)
noncomputable def point_Q (β : ℝ) : ℝ × ℝ × ℝ := (2 * Real.cos β, 2 * Real.sin β, 1)

noncomputable def distance_PQ (α β : ℝ) : ℝ :=
  Real.sqrt ((3 * Real.cos α - 2 * Real.cos β)^2 +
             (3 * Real.sin α - 2 * Real.sin β)^2 +
             (1 - 1)^2)

theorem range_distance_PQ : 
  ∀ α β : ℝ, 1 ≤ distance_PQ α β ∧ distance_PQ α β ≤ 5 := 
by
  intros
  sorry

end NUMINAMATH_GPT_range_distance_PQ_l996_99693


namespace NUMINAMATH_GPT_asha_money_remaining_l996_99623

-- Given conditions as definitions in Lean
def borrowed_from_brother : ℕ := 20
def borrowed_from_father : ℕ := 40
def borrowed_from_mother : ℕ := 30
def gift_from_granny : ℕ := 70
def initial_savings : ℕ := 100

-- Total amount of money Asha has
def total_money : ℕ := borrowed_from_brother + borrowed_from_father + borrowed_from_mother + gift_from_granny + initial_savings

-- Money spent by Asha
def money_spent : ℕ := (3 * total_money) / 4

-- Money remaining with Asha
def money_remaining : ℕ := total_money - money_spent

-- Theorem stating the result
theorem asha_money_remaining : money_remaining = 65 := by
  sorry

end NUMINAMATH_GPT_asha_money_remaining_l996_99623


namespace NUMINAMATH_GPT_jace_total_distance_l996_99601

noncomputable def total_distance (s1 s2 s3 s4 s5 : ℝ) (t1 t2 t3 t4 t5 : ℝ) : ℝ :=
  s1 * t1 + s2 * t2 + s3 * t3 + s4 * t4 + s5 * t5

theorem jace_total_distance :
  total_distance 50 65 60 75 55 3 4.5 2.75 1.8333 2.6667 = 891.67 := by
  sorry

end NUMINAMATH_GPT_jace_total_distance_l996_99601


namespace NUMINAMATH_GPT_yearly_return_of_1500_investment_l996_99685

theorem yearly_return_of_1500_investment 
  (combined_return_percent : ℝ)
  (total_investment : ℕ)
  (return_500 : ℕ)
  (investment_500 : ℕ)
  (investment_1500 : ℕ) :
  combined_return_percent = 0.085 →
  total_investment = (investment_500 + investment_1500) →
  return_500 = (investment_500 * 7 / 100) →
  investment_500 = 500 →
  investment_1500 = 1500 →
  total_investment = 2000 →
  (return_500 + investment_1500 * combined_return_percent * 100) = (combined_return_percent * total_investment * 100) →
  ((investment_1500 * (9 : ℝ)) / 100) + return_500 = 0.085 * total_investment →
  (investment_1500 * 7 / 100) = investment_1500 →
  (investment_1500 / investment_1500) = (13500 / 1500) →
  (9 : ℝ) = 9 :=
sorry

end NUMINAMATH_GPT_yearly_return_of_1500_investment_l996_99685


namespace NUMINAMATH_GPT_trigonometric_expression_value_l996_99636

variable (θ : ℝ)

-- Conditions
axiom tan_theta_eq_two : Real.tan θ = 2

-- Theorem to prove
theorem trigonometric_expression_value : 
  Real.sin θ * Real.sin θ + 
  Real.sin θ * Real.cos θ - 
  2 * Real.cos θ * Real.cos θ = 4 / 5 := 
by
  sorry

end NUMINAMATH_GPT_trigonometric_expression_value_l996_99636


namespace NUMINAMATH_GPT_coordinates_of_point_P_l996_99616

theorem coordinates_of_point_P (x y : ℝ) (h1 : x > 0) (h2 : y < 0) (h3 : abs y = 2) (h4 : abs x = 4) : (x, y) = (4, -2) :=
by
  sorry

end NUMINAMATH_GPT_coordinates_of_point_P_l996_99616


namespace NUMINAMATH_GPT_annual_rent_per_square_foot_is_156_l996_99609

-- Given conditions
def monthly_rent : ℝ := 1300
def length : ℝ := 10
def width : ℝ := 10
def area : ℝ := length * width
def annual_rent : ℝ := monthly_rent * 12

-- Proof statement: Annual rent per square foot
theorem annual_rent_per_square_foot_is_156 : 
  annual_rent / area = 156 := by
  sorry

end NUMINAMATH_GPT_annual_rent_per_square_foot_is_156_l996_99609


namespace NUMINAMATH_GPT_ratio_of_numbers_l996_99694

theorem ratio_of_numbers (x y : ℕ) (h1 : x + y = 33) (h2 : x = 22) : y / x = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_numbers_l996_99694


namespace NUMINAMATH_GPT_find_value_of_P_l996_99643

def f (x : ℝ) : ℝ := (x^2 + x - 2)^2002 + 3

theorem find_value_of_P :
  f ( (Real.sqrt 5) / 2 - 1 / 2 ) = 4 := by
  sorry

end NUMINAMATH_GPT_find_value_of_P_l996_99643


namespace NUMINAMATH_GPT_find_n_l996_99676

noncomputable def C (n : ℕ) : ℝ :=
  352 * (1 - 1 / 2 ^ n) / (1 - 1 / 2)

noncomputable def D (n : ℕ) : ℝ :=
  992 * (1 - 1 / (-2) ^ n) / (1 + 1 / 2)

theorem find_n (n : ℕ) (h : 1 ≤ n) : C n = D n ↔ n = 1 := by
  sorry

end NUMINAMATH_GPT_find_n_l996_99676


namespace NUMINAMATH_GPT_perpendicular_distance_l996_99689

structure Vertex :=
  (x : ℝ) (y : ℝ) (z : ℝ)

def S : Vertex := ⟨6, 0, 0⟩
def P : Vertex := ⟨0, 0, 0⟩
def Q : Vertex := ⟨0, 5, 0⟩
def R : Vertex := ⟨0, 0, 4⟩

noncomputable def distance_from_point_to_plane (S P Q R : Vertex) : ℝ := sorry

theorem perpendicular_distance (S P Q R : Vertex) (hS : S = ⟨6, 0, 0⟩) (hP : P = ⟨0, 0, 0⟩) (hQ : Q = ⟨0, 5, 0⟩) (hR : R = ⟨0, 0, 4⟩) :
  distance_from_point_to_plane S P Q R = 6 :=
  sorry

end NUMINAMATH_GPT_perpendicular_distance_l996_99689


namespace NUMINAMATH_GPT_abs_sum_less_abs_diff_l996_99615

theorem abs_sum_less_abs_diff {a b : ℝ} (hab : a * b < 0) : |a + b| < |a - b| :=
sorry

end NUMINAMATH_GPT_abs_sum_less_abs_diff_l996_99615


namespace NUMINAMATH_GPT_max_integer_inequality_l996_99611

theorem max_integer_inequality (a b c: ℝ) (h₀: 0 < a) (h₁: 0 < b) (h₂: 0 < c) :
  (a^2 / (b / 29 + c / 31) + b^2 / (c / 29 + a / 31) + c^2 / (a / 29 + b / 31)) ≥ 14 * (a + b + c) :=
sorry

end NUMINAMATH_GPT_max_integer_inequality_l996_99611


namespace NUMINAMATH_GPT_Andrew_has_5_more_goats_than_twice_Adam_l996_99681

-- Definitions based on conditions
def goats_Adam := 7
def goats_Ahmed := 13
def goats_Andrew := goats_Ahmed + 6
def twice_goats_Adam := 2 * goats_Adam

-- Theorem statement
theorem Andrew_has_5_more_goats_than_twice_Adam :
  goats_Andrew - twice_goats_Adam = 5 :=
by
  sorry

end NUMINAMATH_GPT_Andrew_has_5_more_goats_than_twice_Adam_l996_99681


namespace NUMINAMATH_GPT_tank_filled_to_depth_l996_99652

noncomputable def tank_volume (R H r d : ℝ) : ℝ := R^2 * H * Real.pi - (r^2 * H * Real.pi)

theorem tank_filled_to_depth (R H r d : ℝ) (h_cond : R = 5 ∧ H = 12 ∧ r = 2 ∧ d = 3) :
  tank_volume R H r d = 110 * Real.pi - 96 :=
sorry

end NUMINAMATH_GPT_tank_filled_to_depth_l996_99652


namespace NUMINAMATH_GPT_expand_binom_l996_99606

theorem expand_binom (x : ℝ) : (x + 3) * (4 * x - 8) = 4 * x^2 + 4 * x - 24 :=
by
  sorry

end NUMINAMATH_GPT_expand_binom_l996_99606


namespace NUMINAMATH_GPT_at_least_3_students_same_score_l996_99679

-- Conditions
def initial_points : ℕ := 6
def correct_points : ℕ := 4
def incorrect_points : ℤ := -1
def num_questions : ℕ := 6
def num_students : ℕ := 51

-- Question
theorem at_least_3_students_same_score :
  ∃ score : ℤ, ∃ students_with_same_score : ℕ, students_with_same_score ≥ 3 :=
by
  sorry

end NUMINAMATH_GPT_at_least_3_students_same_score_l996_99679


namespace NUMINAMATH_GPT_shopkeeper_profit_percent_l996_99698

theorem shopkeeper_profit_percent
  (initial_value : ℝ)
  (percent_lost_theft : ℝ)
  (percent_total_loss : ℝ)
  (remaining_value : ℝ)
  (total_loss_value : ℝ)
  (selling_price : ℝ)
  (profit : ℝ)
  (profit_percent : ℝ)
  (h_initial_value : initial_value = 100)
  (h_percent_lost_theft : percent_lost_theft = 20)
  (h_percent_total_loss : percent_total_loss = 12)
  (h_remaining_value : remaining_value = initial_value - (percent_lost_theft / 100) * initial_value)
  (h_total_loss_value : total_loss_value = (percent_total_loss / 100) * initial_value)
  (h_selling_price : selling_price = initial_value - total_loss_value)
  (h_profit : profit = selling_price - remaining_value)
  (h_profit_percent : profit_percent = (profit / remaining_value) * 100) :
  profit_percent = 10 := by
  sorry

end NUMINAMATH_GPT_shopkeeper_profit_percent_l996_99698


namespace NUMINAMATH_GPT_find_original_price_l996_99691

-- Defining constants and variables
def original_price (P : ℝ) : Prop :=
  let cost_after_repairs := P + 13000
  let selling_price := 66900
  let profit := selling_price - cost_after_repairs
  let profit_percent := profit / P * 100
  profit_percent = 21.636363636363637

theorem find_original_price : ∃ P : ℝ, original_price P :=
  by
  sorry

end NUMINAMATH_GPT_find_original_price_l996_99691


namespace NUMINAMATH_GPT_college_girls_count_l996_99688

theorem college_girls_count (B G : ℕ) (h1 : B / G = 8 / 5) (h2 : B + G = 546) : G = 210 :=
by
  sorry

end NUMINAMATH_GPT_college_girls_count_l996_99688


namespace NUMINAMATH_GPT_initial_bananas_l996_99671

theorem initial_bananas (x B : ℕ) (h1 : 840 * x = B) (h2 : 420 * (x + 2) = B) : x = 2 :=
by
  sorry

end NUMINAMATH_GPT_initial_bananas_l996_99671


namespace NUMINAMATH_GPT_mike_found_four_more_seashells_l996_99627

/--
Given:
1. Mike initially found 6.0 seashells.
2. The total number of seashells Mike had after finding more is 10.

Prove:
Mike found 4.0 more seashells.
-/
theorem mike_found_four_more_seashells (initial_seashells : ℝ) (total_seashells : ℝ)
  (h1 : initial_seashells = 6.0)
  (h2 : total_seashells = 10.0) :
  total_seashells - initial_seashells = 4.0 :=
by
  sorry

end NUMINAMATH_GPT_mike_found_four_more_seashells_l996_99627


namespace NUMINAMATH_GPT_triangle_is_isosceles_l996_99697

theorem triangle_is_isosceles
    (A B C : ℝ)
    (h_angle_sum : A + B + C = 180)
    (h_sinB : Real.sin B = 2 * Real.cos C * Real.sin A)
    : (A = C) := 
by
    sorry

end NUMINAMATH_GPT_triangle_is_isosceles_l996_99697


namespace NUMINAMATH_GPT_fraction_books_sold_l996_99624

theorem fraction_books_sold (B : ℕ) (F : ℚ)
  (hc1 : F * B * 4 = 288)
  (hc2 : F * B + 36 = B) :
  F = 2 / 3 :=
by
  sorry

end NUMINAMATH_GPT_fraction_books_sold_l996_99624


namespace NUMINAMATH_GPT_number_of_possible_values_l996_99610

theorem number_of_possible_values (b : ℕ) (hb4 : 4 ∣ b) (hb24 : b ∣ 24) (hpos : 0 < b) : ∃ n, n = 4 :=
by
  sorry

end NUMINAMATH_GPT_number_of_possible_values_l996_99610


namespace NUMINAMATH_GPT_prove_expression_l996_99614

def otimes (a b : ℚ) : ℚ := a^2 / b

theorem prove_expression : ((otimes (otimes 1 2) 3) - (otimes 1 (otimes 2 3))) = -2/3 :=
by 
  sorry

end NUMINAMATH_GPT_prove_expression_l996_99614


namespace NUMINAMATH_GPT_number_of_impossible_d_l996_99647

-- Define the problem parameters and conditions
def perimeter_diff (t s : ℕ) : ℕ := 3 * t - 4 * s
def side_diff (t s d : ℕ) : ℕ := t - s - d
def square_perimeter_positive (s : ℕ) : Prop := s > 0

-- Define the proof problem
theorem number_of_impossible_d (t s d : ℕ) (h1 : perimeter_diff t s = 1575) (h2 : side_diff t s d = 0) (h3 : square_perimeter_positive s) : 
    ∃ n, n = 525 ∧ ∀ d, d ≤ 525 → ¬ (3 * d > 1575) :=
    sorry

end NUMINAMATH_GPT_number_of_impossible_d_l996_99647


namespace NUMINAMATH_GPT_average_of_other_two_numbers_l996_99621

theorem average_of_other_two_numbers
  (avg_5_numbers : ℕ → ℚ)
  (sum_3_numbers : ℕ → ℚ)
  (h1 : ∀ n, avg_5_numbers n = 20)
  (h2 : ∀ n, sum_3_numbers n = 48)
  (h3 : ∀ n, ∃ x y z p q : ℚ, avg_5_numbers n = (x + y + z + p + q) / 5)
  (h4 : ∀ n, sum_3_numbers n = x + y + z) :
  ∃ u v : ℚ, ((u + v) / 2 = 26) :=
by sorry

end NUMINAMATH_GPT_average_of_other_two_numbers_l996_99621


namespace NUMINAMATH_GPT_find_least_positive_n_l996_99654

theorem find_least_positive_n (n : ℕ) : 
  let m := 143
  m = 11 * 13 → 
  (3^5 ≡ 1 [MOD m^2]) →
  (3^39 ≡ 1 [MOD (13^2)]) →
  n = 195 :=
sorry

end NUMINAMATH_GPT_find_least_positive_n_l996_99654


namespace NUMINAMATH_GPT_dodecagon_diagonals_l996_99661

def D (n : ℕ) : ℕ := n * (n - 3) / 2

theorem dodecagon_diagonals : D 12 = 54 :=
by
  sorry

end NUMINAMATH_GPT_dodecagon_diagonals_l996_99661


namespace NUMINAMATH_GPT_train_total_travel_time_l996_99617

noncomputable def totalTravelTime (d1 d2 s1 s2 : ℝ) : ℝ :=
  (d1 / s1) + (d2 / s2)

theorem train_total_travel_time : 
  totalTravelTime 150 200 50 80 = 5.5 :=
by
  sorry

end NUMINAMATH_GPT_train_total_travel_time_l996_99617


namespace NUMINAMATH_GPT_box_ratio_l996_99686

theorem box_ratio (h : ℤ) (l : ℤ) (w : ℤ) (v : ℤ)
  (H_height : h = 12)
  (H_length : l = 3 * h)
  (H_volume : l * w * h = 3888)
  (H_length_multiple : ∃ m, l = m * w) :
  l / w = 4 := by
  sorry

end NUMINAMATH_GPT_box_ratio_l996_99686


namespace NUMINAMATH_GPT_part1_part2_l996_99648

/-- Given that in triangle ABC, sides opposite to angles A, B, and C are a, b, and c respectively
     if 2a sin B = sqrt(3) b and A is an acute angle, then A = 60 degrees. -/
theorem part1 {a b : ℝ} {A B : ℝ} (h1 : 2 * a * Real.sin B = Real.sqrt 3 * b)
  (h2 : 0 < A ∧ A < Real.pi / 2) : A = Real.pi / 3 :=
sorry

/-- Given that in triangle ABC, sides opposite to angles A, B, and C are a, b, and c respectively
     if b = 5, c = sqrt(5), and cos C = 9 / 10, then a = 4 or a = 5. -/
theorem part2 {a b c : ℝ} {C : ℝ} (h1 : b = 5) (h2 : c = Real.sqrt 5) 
  (h3 : Real.cos C = 9 / 10) : a = 4 ∨ a = 5 :=
sorry

end NUMINAMATH_GPT_part1_part2_l996_99648


namespace NUMINAMATH_GPT_solve_for_y_l996_99659

theorem solve_for_y (y : ℝ) (h : 1 / 4 - 1 / 5 = 4 / y) : y = 80 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_y_l996_99659


namespace NUMINAMATH_GPT_lindy_total_distance_traveled_l996_99600

theorem lindy_total_distance_traveled 
    (initial_distance : ℕ)
    (jack_speed : ℕ)
    (christina_speed : ℕ)
    (lindy_speed : ℕ) 
    (meet_time : ℕ)
    (distance : ℕ) :
    initial_distance = 150 →
    jack_speed = 7 →
    christina_speed = 8 →
    lindy_speed = 10 →
    meet_time = initial_distance / (jack_speed + christina_speed) →
    distance = lindy_speed * meet_time →
    distance = 100 := 
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end NUMINAMATH_GPT_lindy_total_distance_traveled_l996_99600


namespace NUMINAMATH_GPT_cubic_sum_of_roots_l996_99640

theorem cubic_sum_of_roots (r s a b : ℝ) (h1 : r + s = a) (h2 : r * s = b) : 
  r^3 + s^3 = a^3 - 3 * a * b :=
by
  sorry

end NUMINAMATH_GPT_cubic_sum_of_roots_l996_99640


namespace NUMINAMATH_GPT_find_line_equation_l996_99635

theorem find_line_equation 
  (ellipse_eq : ∀ x y : ℝ, x^2 / 4 + y^2 / 3 = 1)
  (P : ℝ × ℝ) (P_coord : P = (1, 3/2))
  (line_l : ∀ x : ℝ, ℝ)
  (line_eq : ∀ x : ℝ, y = k * x + b) 
  (intersects : ∀ A B : ℝ × ℝ, A ≠ P ∧ B ≠ P)
  (perpendicular : ∀ A B : ℝ × ℝ, (A.1 - 1) * (B.1 - 1) + (A.2 - 3 / 2) * (B.2 - 3 / 2) = 0)
  (bisected_by_y_axis : ∀ A B : ℝ × ℝ, A.1 + B.1 = 0) :
  ∃ k : ℝ, k = 3 / 2 ∨ k = -3 / 2 :=
sorry

end NUMINAMATH_GPT_find_line_equation_l996_99635


namespace NUMINAMATH_GPT_progress_regress_ratio_l996_99620

theorem progress_regress_ratio :
  let progress_rate := 1.2
  let regress_rate := 0.8
  let log2 := 0.3010
  let log3 := 0.4771
  let target_ratio := 10000
  (progress_rate / regress_rate) ^ 23 = target_ratio :=
by
  sorry

end NUMINAMATH_GPT_progress_regress_ratio_l996_99620


namespace NUMINAMATH_GPT_find_original_number_l996_99683

theorem find_original_number (x : ℤ) (h : (x + 5) % 23 = 0) : x = 18 :=
sorry

end NUMINAMATH_GPT_find_original_number_l996_99683


namespace NUMINAMATH_GPT_solve_math_problem_l996_99678

noncomputable def math_problem : Prop :=
  ∃ (ω α β : ℂ), (ω^5 = 1) ∧ (ω ≠ 1) ∧ (α = ω + ω^2) ∧ (β = ω^3 + ω^4) ∧
  (∀ x : ℂ, x^2 + x + 3 = 0 → x = α ∨ x = β) ∧ (α + β = -1) ∧ (α * β = 3)

theorem solve_math_problem : math_problem := sorry

end NUMINAMATH_GPT_solve_math_problem_l996_99678


namespace NUMINAMATH_GPT_radius_of_circle_l996_99668

variable (r M N : ℝ)

theorem radius_of_circle (h1 : M = Real.pi * r^2) 
  (h2 : N = 2 * Real.pi * r) 
  (h3 : M / N = 15) : 
  r = 30 :=
sorry

end NUMINAMATH_GPT_radius_of_circle_l996_99668


namespace NUMINAMATH_GPT_problem_extraneous_root_l996_99602

theorem problem_extraneous_root (m : ℤ) :
  (∃ x, x = -4 ∧ (x + 4 = 0) ∧ ((x-1)/(x+4) = m/(x+4)) ∧ (m = -5)) :=
sorry

end NUMINAMATH_GPT_problem_extraneous_root_l996_99602


namespace NUMINAMATH_GPT_quadratic_vertex_position_l996_99607

theorem quadratic_vertex_position (a p q m : ℝ) (ha : 0 < a) (hpq : p < q) (hA : p = a * (-1 - m)^2) (hB : q = a * (3 - m)^2) : m ≠ 2 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_vertex_position_l996_99607


namespace NUMINAMATH_GPT_problem1_problem2_l996_99618

noncomputable def f : ℝ → ℝ := -- we assume f is noncomputable since we know its explicit form in the desired interval
sorry

axiom periodic_f (x : ℝ) : f (x + 5) = f x
axiom odd_f {x : ℝ} (h : -1 ≤ x ∧ x ≤ 1) : f (-x) = -f x
axiom quadratic_f {x : ℝ} (h : 1 ≤ x ∧ x ≤ 4) : f x = 2 * (x - 2) ^ 2 - 5
axiom minimum_f : f 2 = -5

theorem problem1 : f 1 + f 4 = 0 :=
by
  sorry

theorem problem2 {x : ℝ} (h : 1 ≤ x ∧ x ≤ 4) : f x = 2 * x ^ 2 - 8 * x + 3 :=
by
  sorry

end NUMINAMATH_GPT_problem1_problem2_l996_99618


namespace NUMINAMATH_GPT_num_possible_radii_l996_99666

theorem num_possible_radii :
  ∃ S : Finset ℕ, S.card = 11 ∧ ∀ r ∈ S, (∃ k : ℕ, 150 = k * r) ∧ r ≠ 150 :=
by
  sorry

end NUMINAMATH_GPT_num_possible_radii_l996_99666


namespace NUMINAMATH_GPT_find_a_l996_99655

theorem find_a (a : ℝ) :
  (∃ x y : ℝ, x - 2 * y + 1 = 0 ∧ x + 3 * y - 1 = 0 ∧ ¬(∀ x y : ℝ, ax + 2 * y - 3 = 0)) →
  (∃ p q : ℝ, ax + 2 * q - 3 = 0 ∧ (a = -1 ∨ a = 2 / 3)) :=
by {
  sorry
}

end NUMINAMATH_GPT_find_a_l996_99655


namespace NUMINAMATH_GPT_productivity_increase_l996_99631

theorem productivity_increase :
  (∃ d : ℝ, 
   (∀ n : ℕ, 0 < n → n ≤ 30 → 
      (5 + (n - 1) * d ≥ 0) ∧ 
      (30 * 5 + (30 * 29 / 2) * d = 390) ∧ 
      1 / 100 < d ∧ d < 1) ∧
      d = 0.52) :=
sorry

end NUMINAMATH_GPT_productivity_increase_l996_99631


namespace NUMINAMATH_GPT_simplified_value_l996_99664

theorem simplified_value :
  (245^2 - 205^2) / 40 = 450 := by
  sorry

end NUMINAMATH_GPT_simplified_value_l996_99664


namespace NUMINAMATH_GPT_exists_convex_2011_gon_on_parabola_not_exists_convex_2012_gon_on_parabola_l996_99633

-- Define the parabola as a function
def parabola (x : ℝ) : ℝ := x^2

-- N-gon properties
def is_convex_ngon (N : ℕ) (vertices : List (ℝ × ℝ)) : Prop :=
  -- Placeholder for checking properties; actual implementation would validate convexity and equilateral nature.
  sorry 

-- Statement for 2011-gon
theorem exists_convex_2011_gon_on_parabola :
  ∃ (vertices : List (ℝ × ℝ)), is_convex_ngon 2011 vertices ∧ ∀ v ∈ vertices, v.2 = parabola v.1 :=
sorry

-- Statement for 2012-gon
theorem not_exists_convex_2012_gon_on_parabola :
  ¬ ∃ (vertices : List (ℝ × ℝ)), is_convex_ngon 2012 vertices ∧ ∀ v ∈ vertices, v.2 = parabola v.1 :=
sorry

end NUMINAMATH_GPT_exists_convex_2011_gon_on_parabola_not_exists_convex_2012_gon_on_parabola_l996_99633


namespace NUMINAMATH_GPT_smallest_largest_multiples_l996_99667

theorem smallest_largest_multiples : 
  ∃ l g, l >= 10 ∧ l < 100 ∧ g >= 100 ∧ g < 1000 ∧
  (2 ∣ l) ∧ (3 ∣ l) ∧ (5 ∣ l) ∧ 
  (2 ∣ g) ∧ (3 ∣ g) ∧ (5 ∣ g) ∧
  (∀ n, n >= 10 ∧ n < 100 ∧ (2 ∣ n ∧ 3 ∣ n ∧ 5 ∣ n) → l ≤ n) ∧
  (∀ n, n >= 100 ∧ n < 1000 ∧ (2 ∣ n ∧ 3 ∣ n ∧ 5 ∣ n) → g >= n) ∧
  l = 30 ∧ g = 990 := 
by 
  sorry

end NUMINAMATH_GPT_smallest_largest_multiples_l996_99667


namespace NUMINAMATH_GPT_marks_in_physics_l996_99687

-- Definitions of the variables
variables (P C M : ℕ)

-- Conditions
def condition1 : Prop := P + C + M = 210
def condition2 : Prop := P + M = 180
def condition3 : Prop := P + C = 140

-- The statement to prove
theorem marks_in_physics (h1 : condition1 P C M) (h2 : condition2 P M) (h3 : condition3 P C) : P = 110 :=
sorry

end NUMINAMATH_GPT_marks_in_physics_l996_99687


namespace NUMINAMATH_GPT_amount_amys_money_l996_99673

def initial_dollars : ℝ := 2
def chores_payment : ℝ := 5 * 13
def birthday_gift : ℝ := 3
def total_after_gift : ℝ := initial_dollars + chores_payment + birthday_gift

def investment_percentage : ℝ := 0.20
def invested_amount : ℝ := investment_percentage * total_after_gift

def interest_rate : ℝ := 0.10
def interest_amount : ℝ := interest_rate * invested_amount
def total_investment : ℝ := invested_amount + interest_amount

def cost_of_toy : ℝ := 12
def remaining_after_toy : ℝ := total_after_gift - cost_of_toy

def grandparents_gift : ℝ := 2 * remaining_after_toy
def total_including_investment : ℝ := grandparents_gift + total_investment

def donation_percentage : ℝ := 0.25
def donated_amount : ℝ := donation_percentage * total_including_investment

def final_amount : ℝ := total_including_investment - donated_amount

theorem amount_amys_money :
  final_amount = 98.55 := by
  sorry

end NUMINAMATH_GPT_amount_amys_money_l996_99673


namespace NUMINAMATH_GPT_sum_of_a_b_l996_99626

variable {a b : ℝ}

theorem sum_of_a_b (h1 : a^2 = 4) (h2 : b^2 = 9) (h3 : a * b < 0) : a + b = 1 ∨ a + b = -1 := 
by 
  sorry

end NUMINAMATH_GPT_sum_of_a_b_l996_99626


namespace NUMINAMATH_GPT_max_volume_day1_l996_99651

-- Define volumes of the containers
def volumes : List ℕ := [9, 13, 17, 19, 20, 38]

-- Define conditions: sold containers volumes
def condition_on_first_day (s: List ℕ) := s.length = 3
def condition_on_second_day (s: List ℕ) := s.length = 2

-- Define condition: total and relative volumes sold
def volume_sold_first_day (s: List ℕ) : ℕ := s.foldr (λ x acc => x + acc) 0
def volume_sold_second_day (s: List ℕ) : ℕ := s.foldr (λ x acc => x + acc) 0

def volume_sold_total (s1 s2: List ℕ) := volume_sold_first_day s1 + volume_sold_second_day s2 = 116
def volume_ratio (s1 s2: List ℕ) := volume_sold_first_day s1 = 2 * volume_sold_second_day s2 

-- The goal is to prove the maximum possible volume_sold_first_day
theorem max_volume_day1 (s1 s2: List ℕ) 
  (h1: condition_on_first_day s1)
  (h2: condition_on_second_day s2)
  (h3: volume_sold_total s1 s2)
  (h4: volume_ratio s1 s2) : 
  ∃(max_volume: ℕ), max_volume = 66 :=
sorry

end NUMINAMATH_GPT_max_volume_day1_l996_99651


namespace NUMINAMATH_GPT_product_of_solutions_eq_neg_ten_l996_99684

theorem product_of_solutions_eq_neg_ten :
  (∃ x₁ x₂, -20 = -2 * x₁^2 - 6 * x₁ ∧ -20 = -2 * x₂^2 - 6 * x₂ ∧ x₁ ≠ x₂ ∧ x₁ * x₂ = -10) :=
by
  sorry

end NUMINAMATH_GPT_product_of_solutions_eq_neg_ten_l996_99684


namespace NUMINAMATH_GPT_sum_of_series_is_918_l996_99608

-- Define the first term a, common difference d, last term a_n,
-- and the number of terms n calculated from the conditions.
def first_term : Int := -300
def common_difference : Int := 3
def last_term : Int := 309
def num_terms : Int := 204 -- calculated as per the solution

-- Compute the sum of the arithmetic series
def sum_arithmetic_series (a d : Int) (n : Int) : Int :=
  n * (2 * a + (n - 1) * d) / 2

-- Prove that the sum of the series is 918
theorem sum_of_series_is_918 :
  sum_arithmetic_series first_term common_difference num_terms = 918 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_series_is_918_l996_99608


namespace NUMINAMATH_GPT_gcd_power_diff_l996_99605

theorem gcd_power_diff (m n : ℕ) (h1 : m = 2^2021 - 1) (h2 : n = 2^2000 - 1) :
  Nat.gcd m n = 2097151 :=
by sorry

end NUMINAMATH_GPT_gcd_power_diff_l996_99605


namespace NUMINAMATH_GPT_area_of_rectangle_l996_99674

namespace RectangleArea

variable (l b : ℕ)
variable (h1 : l = 3 * b)
variable (h2 : 2 * (l + b) = 88)

theorem area_of_rectangle : l * b = 363 :=
by
  -- We will prove this in Lean 
  sorry

end RectangleArea

end NUMINAMATH_GPT_area_of_rectangle_l996_99674


namespace NUMINAMATH_GPT_geom_seq_product_l996_99672

theorem geom_seq_product (a : ℕ → ℝ) (r : ℝ) 
  (h_geom : ∀ n, a (n + 1) = r * a n)
  (h_prod : a 5 * a 14 = 5) :
  a 8 * a 9 * a 10 * a 11 = 10 := 
sorry

end NUMINAMATH_GPT_geom_seq_product_l996_99672


namespace NUMINAMATH_GPT_inclination_angle_l996_99629

theorem inclination_angle (θ : ℝ) : 
  (∃ (x y : ℝ), x + y - 3 = 0) → θ = 3 * Real.pi / 4 := 
sorry

end NUMINAMATH_GPT_inclination_angle_l996_99629


namespace NUMINAMATH_GPT_part_a_part_b_part_c_l996_99622

def f (n d : ℕ) : ℕ := sorry

theorem part_a (n : ℕ) (h_even_n : n % 2 = 0) : f n 0 ≤ n :=
sorry

theorem part_b (n d : ℕ) (h_even_n_minus_d : (n - d) % 2 = 0) : f n d ≤ (n + d) / (d + 1) :=
sorry

theorem part_c (n : ℕ) (h_even_n : n % 2 = 0) : f n 0 = n :=
sorry

end NUMINAMATH_GPT_part_a_part_b_part_c_l996_99622


namespace NUMINAMATH_GPT_evaluate_g_at_5_l996_99658

def g (x : ℝ) : ℝ := 5 * x + 2

theorem evaluate_g_at_5 : g 5 = 27 := by
  sorry

end NUMINAMATH_GPT_evaluate_g_at_5_l996_99658


namespace NUMINAMATH_GPT_marek_sequence_sum_l996_99657

theorem marek_sequence_sum (x : ℝ) :
  let a := x
  let b := (a + 4) / 4 - 4
  let c := (b + 4) / 4 - 4
  let d := (c + 4) / 4 - 4
  (a + 4) / 4 * 4 + (b + 4) / 4 * 4 + (c + 4) / 4 * 4 + (d + 4) / 4 * 4 = 80 →
  x = 38 :=
by
  sorry

end NUMINAMATH_GPT_marek_sequence_sum_l996_99657


namespace NUMINAMATH_GPT_least_x_value_l996_99630

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem least_x_value (x : ℕ) (p : ℕ) (hp : is_prime p) (h : x / (12 * p) = 2) : x = 48 := by
  sorry

end NUMINAMATH_GPT_least_x_value_l996_99630


namespace NUMINAMATH_GPT_simplify_expression_l996_99680

theorem simplify_expression :
  (1 / (Real.log 3 / Real.log 12 + 1) + 1 / (Real.log 2 / Real.log 8 + 1) + 1 / (Real.log 9 / Real.log 18 + 1)) = 7 / 4 := 
sorry

end NUMINAMATH_GPT_simplify_expression_l996_99680


namespace NUMINAMATH_GPT_count_perfect_cubes_l996_99613

theorem count_perfect_cubes (a b : ℕ) (h1 : a = 200) (h2 : b = 1200) :
  ∃ n, n = 5 ∧ ∀ x, (x^3 > a) ∧ (x^3 < b) → (x = 6 ∨ x = 7 ∨ x = 8 ∨ x = 9 ∨ x = 10) := 
sorry

end NUMINAMATH_GPT_count_perfect_cubes_l996_99613


namespace NUMINAMATH_GPT_profit_percentage_is_50_l996_99692

/--
Assumption:
- Initial machine cost: Rs 10,000
- Repair cost: Rs 5,000
- Transportation charges: Rs 1,000
- Selling price: Rs 24,000

To prove:
- The profit percentage is 50%
-/

def initial_cost : ℕ := 10000
def repair_cost : ℕ := 5000
def transportation_charges : ℕ := 1000
def selling_price : ℕ := 24000
def total_cost : ℕ := initial_cost + repair_cost + transportation_charges
def profit : ℕ := selling_price - total_cost

theorem profit_percentage_is_50 :
  (profit * 100) / total_cost = 50 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_profit_percentage_is_50_l996_99692


namespace NUMINAMATH_GPT_eq_or_sum_zero_l996_99649

variables (a b c d : ℝ)

theorem eq_or_sum_zero (h : (3 * a + 2 * b) / (2 * b + 4 * c) = (4 * c + 3 * d) / (3 * d + 3 * a)) :
  3 * a = 4 * c ∨ 3 * a + 3 * d + 2 * b + 4 * c = 0 :=
by sorry

end NUMINAMATH_GPT_eq_or_sum_zero_l996_99649


namespace NUMINAMATH_GPT_odd_function_condition_l996_99695

noncomputable def f (x a b : ℝ) : ℝ := x * |x + a| + b

theorem odd_function_condition (a b : ℝ) :
  (∀ x : ℝ, f (-x) a b = -f x a b) ↔ a^2 + b^2 = 0 :=
by
  sorry

end NUMINAMATH_GPT_odd_function_condition_l996_99695


namespace NUMINAMATH_GPT_clubsuit_subtraction_l996_99656

def clubsuit (x y : ℕ) := 4 * x + 6 * y

theorem clubsuit_subtraction :
  (clubsuit 5 3) - (clubsuit 1 4) = 10 :=
by
  sorry

end NUMINAMATH_GPT_clubsuit_subtraction_l996_99656


namespace NUMINAMATH_GPT_speed_limit_l996_99653

theorem speed_limit (x : ℝ) (h₀ : 0 < x) :
  (11 / (x + 1.5) + 8 / x ≥ 12 / (x + 2) + 2) → x ≤ 4 := 
sorry

end NUMINAMATH_GPT_speed_limit_l996_99653


namespace NUMINAMATH_GPT_mosquito_shadow_speed_l996_99660

theorem mosquito_shadow_speed
  (v : ℝ) (t : ℝ) (h : ℝ) (cos_theta : ℝ) (v_shadow : ℝ)
  (hv : v = 0.5) (ht : t = 20) (hh : h = 6) (hcos_theta : cos_theta = 0.6) :
  v_shadow = 0 ∨ v_shadow = 0.8 :=
  sorry

end NUMINAMATH_GPT_mosquito_shadow_speed_l996_99660


namespace NUMINAMATH_GPT_circle_radius_5_l996_99670

theorem circle_radius_5 (c : ℝ) : 
  (∀ x y : ℝ, x^2 + 10 * x + y^2 + 2 * y + c = 0) → 
  (∀ x y : ℝ, (x + 5)^2 + (y + 1)^2 = 25) → 
  c = 51 :=
sorry

end NUMINAMATH_GPT_circle_radius_5_l996_99670


namespace NUMINAMATH_GPT_maximum_BD_cyclic_quad_l996_99682

theorem maximum_BD_cyclic_quad (AB BC CD : ℤ) (BD : ℝ)
  (h_side_bounds : AB < 15 ∧ BC < 15 ∧ CD < 15)
  (h_distinct_sides : AB ≠ BC ∧ BC ≠ CD ∧ CD ≠ AB)
  (h_AB_value : AB = 13)
  (h_BC_value : BC = 5)
  (h_CD_value : CD = 8)
  (h_sides_product : BC * CD = AB * (10 : ℤ)) :
  BD = Real.sqrt 179 := 
by 
  sorry

end NUMINAMATH_GPT_maximum_BD_cyclic_quad_l996_99682


namespace NUMINAMATH_GPT_find_r_l996_99675

-- Declaring the roots of the first polynomial
variables (a b m : ℝ)
-- Declaring the roots of the second polynomial
variables (p r : ℝ)

-- Assumptions based on the given conditions
def roots_of_first_eq : Prop :=
  a + b = m ∧ a * b = 3

def roots_of_second_eq : Prop :=
  ∃ (p : ℝ), (a^2 + 1/b) * (b^2 + 1/a) = r

-- The desired theorem
theorem find_r 
  (h1 : roots_of_first_eq a b m)
  (h2 : (a^2 + 1/b) * (b^2 + 1/a) = r) :
  r = 46/3 := by sorry

end NUMINAMATH_GPT_find_r_l996_99675


namespace NUMINAMATH_GPT_A_inter_B_A_subset_C_l996_99663

namespace MathProof

def A := {x : ℝ | x^2 - 6*x + 8 ≤ 0 }
def B := {x : ℝ | (x - 1)/(x - 3) ≥ 0 }
def C (a : ℝ) := {x : ℝ | x^2 - (2*a + 4)*x + a^2 + 4*a ≤ 0 }

theorem A_inter_B : (A ∩ B) = {x : ℝ | 3 < x ∧ x ≤ 4} := sorry

theorem A_subset_C (a : ℝ) : (A ⊆ C a) ↔ (0 ≤ a ∧ a ≤ 2) := sorry

end MathProof

end NUMINAMATH_GPT_A_inter_B_A_subset_C_l996_99663


namespace NUMINAMATH_GPT_train_length_is_correct_l996_99638

noncomputable def length_of_train 
  (time_to_cross : ℝ) 
  (bridge_length : ℝ) 
  (train_speed_kmph : ℝ) : ℝ :=
  let train_speed_mps := train_speed_kmph * (1000 / 3600)
  let distance_covered := train_speed_mps * time_to_cross
  distance_covered - bridge_length

theorem train_length_is_correct :
  length_of_train 23.998080153587715 140 36 = 99.98080153587715 :=
by sorry

end NUMINAMATH_GPT_train_length_is_correct_l996_99638


namespace NUMINAMATH_GPT_derivative_at_one_third_l996_99644

noncomputable def f (x : ℝ) : ℝ := Real.log (2 - 3 * x)

theorem derivative_at_one_third : (deriv f (1 / 3) = -3) := by
  sorry

end NUMINAMATH_GPT_derivative_at_one_third_l996_99644


namespace NUMINAMATH_GPT_soda_cost_per_ounce_l996_99650

/-- 
  Peter brought $2 with him, left with $0.50, and bought 6 ounces of soda.
  Prove that the cost per ounce of soda is $0.25.
-/
theorem soda_cost_per_ounce (initial_money final_money : ℝ) (amount_spent ounces_soda cost_per_ounce : ℝ)
  (h1 : initial_money = 2)
  (h2 : final_money = 0.5)
  (h3 : amount_spent = initial_money - final_money)
  (h4 : amount_spent = 1.5)
  (h5 : ounces_soda = 6)
  (h6 : cost_per_ounce = amount_spent / ounces_soda) :
  cost_per_ounce = 0.25 :=
by sorry

end NUMINAMATH_GPT_soda_cost_per_ounce_l996_99650


namespace NUMINAMATH_GPT_instantaneous_velocity_at_3_l996_99637

-- Define the displacement function s(t)
def displacement (t : ℝ) : ℝ := 2 * t^3

-- Define the time at which we want to calculate the instantaneous velocity
def time : ℝ := 3

-- Define the expected instantaneous velocity at t=3
def expected_velocity : ℝ := 54

-- Define the derivative of the displacement function as the velocity function
noncomputable def velocity (t : ℝ) : ℝ := deriv displacement t

-- Theorem: Prove that the instantaneous velocity at t=3 is 54
theorem instantaneous_velocity_at_3 : velocity time = expected_velocity := 
by {
  -- Here the detailed proof should go, but we skip it with sorry
  sorry
}

end NUMINAMATH_GPT_instantaneous_velocity_at_3_l996_99637


namespace NUMINAMATH_GPT_total_crayons_l996_99645

theorem total_crayons (crayons_per_child : ℕ) (num_children : ℕ) 
      (h1 : crayons_per_child = 18) (h2 : num_children = 36) : 
        crayons_per_child * num_children = 648 := by
  sorry

end NUMINAMATH_GPT_total_crayons_l996_99645


namespace NUMINAMATH_GPT_initial_population_l996_99690

theorem initial_population (P : ℝ) (h1 : P * 1.05 * 0.95 = 9975) : P = 10000 :=
by
  sorry

end NUMINAMATH_GPT_initial_population_l996_99690


namespace NUMINAMATH_GPT_share_pizza_l996_99677

variable (Yoojung_slices Minyoung_slices total_slices : ℕ)
variable (Y : ℕ)

theorem share_pizza :
  Yoojung_slices = Y ∧
  Minyoung_slices = Y + 2 ∧
  total_slices = 10 ∧
  Yoojung_slices + Minyoung_slices = total_slices →
  Y = 4 :=
by
  sorry

end NUMINAMATH_GPT_share_pizza_l996_99677


namespace NUMINAMATH_GPT_slope_of_line_l996_99634

theorem slope_of_line
  (k : ℝ) 
  (hk : 0 < k) 
  (h1 : ¬ (2 / Real.sqrt (k^2 + 1) = 3 * 2 * Real.sqrt (1 - 8 * k^2) / Real.sqrt (k^2 + 1))) 
  : k = 1 / 3 :=
sorry

end NUMINAMATH_GPT_slope_of_line_l996_99634


namespace NUMINAMATH_GPT_focus_of_parabola_x2_eq_neg_4y_l996_99642

theorem focus_of_parabola_x2_eq_neg_4y :
  (∀ x y : ℝ, x^2 = -4 * y → focus = (0, -1)) := 
sorry

end NUMINAMATH_GPT_focus_of_parabola_x2_eq_neg_4y_l996_99642


namespace NUMINAMATH_GPT_total_area_of_L_shaped_figure_l996_99628

-- Define the specific lengths for each segment
def bottom_rect_length : ℕ := 10
def bottom_rect_width : ℕ := 6
def central_rect_length : ℕ := 4
def central_rect_width : ℕ := 4
def top_rect_length : ℕ := 5
def top_rect_width : ℕ := 1

-- Calculate the area of each rectangle
def bottom_rect_area : ℕ := bottom_rect_length * bottom_rect_width
def central_rect_area : ℕ := central_rect_length * central_rect_width
def top_rect_area : ℕ := top_rect_length * top_rect_width

-- Given the length and width of the rectangles, calculate the total area of the L-shaped figure
theorem total_area_of_L_shaped_figure : 
  bottom_rect_area + central_rect_area + top_rect_area = 81 := by
  sorry

end NUMINAMATH_GPT_total_area_of_L_shaped_figure_l996_99628


namespace NUMINAMATH_GPT_find_number_l996_99641

theorem find_number (n : ℝ) (h : n - (1004 / 20.08) = 4970) : n = 5020 := 
by {
  sorry
}

end NUMINAMATH_GPT_find_number_l996_99641


namespace NUMINAMATH_GPT_no_base_b_square_of_integer_l996_99619

theorem no_base_b_square_of_integer (b : ℕ) : ¬(∃ n : ℕ, n^2 = b^2 + 3 * b + 1) → b < 4 ∨ b > 8 := by
  sorry

end NUMINAMATH_GPT_no_base_b_square_of_integer_l996_99619


namespace NUMINAMATH_GPT_ticket_savings_l996_99665

def single_ticket_cost : ℝ := 1.50
def package_cost : ℝ := 5.75
def num_tickets_needed : ℝ := 40

theorem ticket_savings :
  (num_tickets_needed * single_ticket_cost) - 
  ((num_tickets_needed / 5) * package_cost) = 14.00 :=
by
  sorry

end NUMINAMATH_GPT_ticket_savings_l996_99665


namespace NUMINAMATH_GPT_exists_within_distance_l996_99604

theorem exists_within_distance (a : ℝ) (n : ℕ) (h₁ : a > 0) (h₂ : n > 0) :
  ∃ k : ℕ, k < n ∧ ∃ m : ℤ, |k * a - m| < 1 / n :=
by
  sorry

end NUMINAMATH_GPT_exists_within_distance_l996_99604


namespace NUMINAMATH_GPT_local_extrema_l996_99696

noncomputable def f (x : ℝ) := 3 * x^3 - 9 * x^2 + 3

theorem local_extrema :
  (∃ x, x = 0 ∧ ∀ δ > 0, ∃ ε > 0, ∀ y, abs (y - x) < ε → f y ≤ f x) ∧
  (∃ x, x = 2 ∧ ∀ δ > 0, ∃ ε > 0, ∀ y, abs (y - x) < ε → f y ≥ f x) :=
sorry

end NUMINAMATH_GPT_local_extrema_l996_99696


namespace NUMINAMATH_GPT_factor_expression_l996_99612

theorem factor_expression (b : ℝ) : 275 * b^2 + 55 * b = 55 * b * (5 * b + 1) := by
  sorry

end NUMINAMATH_GPT_factor_expression_l996_99612


namespace NUMINAMATH_GPT_fuel_first_third_l996_99632

-- Defining constants based on conditions
def total_fuel := 60
def fuel_second_third := total_fuel / 3
def fuel_final_third := fuel_second_third / 2

-- Defining what we need to prove
theorem fuel_first_third :
  total_fuel - (fuel_second_third + fuel_final_third) = 30 :=
by
  sorry

end NUMINAMATH_GPT_fuel_first_third_l996_99632


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_condition_l996_99646

noncomputable def necessary_but_not_sufficient (x : ℝ) : Prop :=
  (3 - x >= 0 → |x - 1| ≤ 2) ∧ ¬(3 - x >= 0 ↔ |x - 1| ≤ 2)

theorem necessary_but_not_sufficient_condition (x : ℝ) :
  necessary_but_not_sufficient x :=
sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_condition_l996_99646


namespace NUMINAMATH_GPT_area_between_circles_of_octagon_l996_99662

-- Define some necessary geometric terms and functions
noncomputable def cot (θ : ℝ) : ℝ := 1 / Real.tan θ
noncomputable def csc (θ : ℝ) : ℝ := 1 / Real.sin θ

/-- The main theorem stating the area between the inscribed and circumscribed circles of a regular octagon is π. -/
theorem area_between_circles_of_octagon :
  let side_length := 2
  let θ := Real.pi / 8 -- 22.5 degrees in radians
  let apothem := cot θ
  let circum_radius := csc θ
  let area_between_circles := π * (circum_radius^2 - apothem^2)
  area_between_circles = π :=
by
  sorry

end NUMINAMATH_GPT_area_between_circles_of_octagon_l996_99662
