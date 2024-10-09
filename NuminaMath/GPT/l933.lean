import Mathlib

namespace minimum_ceiling_height_l933_93386

def is_multiple_of_0_1 (h : ℝ) : Prop := ∃ (k : ℤ), h = k / 10

def football_field_illuminated (h : ℝ) : Prop :=
  ∀ (x y : ℝ), 0 ≤ x ∧ x ≤ 100 ∧ 0 ≤ y ∧ y ≤ 80 →
  (x^2 + y^2 ≤ h^2) ∨ ((x - 100)^2 + y^2 ≤ h^2) ∨
  (x^2 + (y - 80)^2 ≤ h^2) ∨ ((x - 100)^2 + (y - 80)^2 ≤ h^2)

theorem minimum_ceiling_height :
  ∃ (h : ℝ), football_field_illuminated h ∧ is_multiple_of_0_1 h ∧ h = 32.1 :=
sorry

end minimum_ceiling_height_l933_93386


namespace math_quiz_l933_93397

theorem math_quiz (x : ℕ) : 
  (∃ x ≥ 14, (∃ y : ℕ, 16 = x + y + 1) → (6 * x - 2 * y ≥ 75)) → 
  x ≥ 14 :=
by
  sorry

end math_quiz_l933_93397


namespace area_to_paint_correct_l933_93334

-- Define the measurements used in the problem
def wall_height : ℕ := 10
def wall_length : ℕ := 15
def window1_height : ℕ := 3
def window1_length : ℕ := 5
def window2_height : ℕ := 2
def window2_length : ℕ := 2

-- Definition of areas
def wall_area : ℕ := wall_height * wall_length
def window1_area : ℕ := window1_height * window1_length
def window2_area : ℕ := window2_height * window2_length

-- Definition of total area to paint
def total_area_to_paint : ℕ := wall_area - (window1_area + window2_area)

-- Theorem statement to prove the total area to paint is 131 square feet
theorem area_to_paint_correct : total_area_to_paint = 131 := by
  sorry

end area_to_paint_correct_l933_93334


namespace confidence_level_unrelated_l933_93378

noncomputable def chi_squared_value : ℝ := 8.654

theorem confidence_level_unrelated :
  chi_squared_value > 6.635 →
  (100 - 99) = 1 :=
by
  sorry

end confidence_level_unrelated_l933_93378


namespace xy_difference_l933_93369

theorem xy_difference (x y : ℝ) (h1 : x + y = 10) (h2 : x^2 - y^2 = 40) : x - y = 4 :=
by
  sorry

end xy_difference_l933_93369


namespace lcm_24_90_l933_93307

theorem lcm_24_90 : lcm 24 90 = 360 :=
by 
-- lcm is the least common multiple of 24 and 90.
-- lcm 24 90 is defined as 360.
sorry

end lcm_24_90_l933_93307


namespace increased_hypotenuse_length_l933_93393

theorem increased_hypotenuse_length :
  let AB := 24
  let BC := 10
  let AB' := AB + 6
  let BC' := BC + 6
  let AC := Real.sqrt (AB^2 + BC^2)
  let AC' := Real.sqrt (AB'^2 + BC'^2)
  AC' - AC = 8 := by
  sorry

end increased_hypotenuse_length_l933_93393


namespace arithmetic_sequence_common_difference_l933_93374

theorem arithmetic_sequence_common_difference 
  (d : ℝ) (h : d ≠ 0) (a : ℕ → ℝ)
  (h1 : a 1 = 9 * d)
  (h_arith : ∀ n, a (n + 1) = a n + d)
  (k : ℕ) :
  (a k)^2 = (a 1) * (a (2 * k)) → k = 4 :=
by
  sorry

end arithmetic_sequence_common_difference_l933_93374


namespace probability_of_fourth_roll_l933_93317

-- Define the conditions 
structure Die :=
(fair : Bool) 
(biased_six : Bool)
(biased_one : Bool)

-- Define the probability function
def roll_prob (d : Die) (f : Bool) : ℚ :=
  if d.fair then 1/6
  else if d.biased_six then if f then 1/2 else 1/10
  else if d.biased_one then if f then 1/10 else 1/5
  else 0

def probability_of_fourth_six (p q : ℕ) (r1 r2 r3 : Bool) (d : Die) : ℚ :=
  (if r1 && r2 && r3 then roll_prob d true else 0) 

noncomputable def final_probability (d1 d2 d3 : Die) (prob_fair distorted_rolls : Bool) : ℚ :=
  let fair_prob := if distorted_rolls then roll_prob d1 true else roll_prob d1 false
  let biased_six_prob := if distorted_rolls then roll_prob d2 true else roll_prob d2 false
  let total_prob := fair_prob + biased_six_prob
  let fair := fair_prob / total_prob
  let biased_six := biased_six_prob / total_prob
  fair * roll_prob d1 true + biased_six * roll_prob d2 true

theorem probability_of_fourth_roll
  (d1 : Die) (d2 : Die) (d3 : Die)
  (h1 : d1.fair = true)
  (h2 : d2.biased_six = true)
  (h3 : d3.biased_one = true)
  (h4 : ∀ d, d1 = d ∨ d2 = d ∨ d3 = d)
  (r1 r2 r3 : Bool)
  : ∃ p q : ℕ, p + q = 11 ∧ final_probability d1 d2 d3 true = 5/6 := 
sorry

end probability_of_fourth_roll_l933_93317


namespace Liz_needs_more_money_l933_93380

theorem Liz_needs_more_money (P : ℝ) (h1 : P = 30000 + 2500) (h2 : 0.80 * P = 26000) : 30000 - (0.80 * P) = 4000 :=
by
  sorry

end Liz_needs_more_money_l933_93380


namespace expression_evaluation_l933_93323

theorem expression_evaluation (a b c d : ℤ) : 
  a / b - c * d^2 = a / (b - c * d^2) :=
sorry

end expression_evaluation_l933_93323


namespace union_complement_eq_l933_93321

def U : Set ℤ := {-2, -1, 0, 1, 2, 3}
def A : Set ℤ := {-1, 0, 1, 2}
def B : Set ℤ := {-1, 0, 3}

theorem union_complement_eq :
  A ∪ (U \ B) = {-2, -1, 0, 1, 2} := by
  sorry

end union_complement_eq_l933_93321


namespace cones_sold_l933_93304

-- Define the conditions
variable (milkshakes : Nat)
variable (cones : Nat)

-- Assume the given conditions
axiom h1 : milkshakes = 82
axiom h2 : milkshakes = cones + 15

-- State the theorem to prove
theorem cones_sold : cones = 67 :=
by
  -- Proof goes here
  sorry

end cones_sold_l933_93304


namespace fraction_sum_l933_93364

theorem fraction_sum (y : ℝ) (a b : ℤ) (h : y = 3.834834834) (h_frac : y = (a : ℝ) / b) (h_coprime : Int.gcd a b = 1) : a + b = 4830 :=
sorry

end fraction_sum_l933_93364


namespace ends_with_two_zeros_l933_93341

theorem ends_with_two_zeros (x y : ℕ) (h : (x^2 + x * y + y^2) % 10 = 0) : (x^2 + x * y + y^2) % 100 = 0 :=
sorry

end ends_with_two_zeros_l933_93341


namespace intersection_A_B_l933_93395

def is_defined (x : ℝ) : Prop := x^2 - 1 ≥ 0

def range_of_y (y : ℝ) : Prop := y ≥ 0

def A_set : Set ℝ := { x | is_defined x }
def B_set : Set ℝ := { y | range_of_y y }

theorem intersection_A_B : A_set ∩ B_set = { x | 1 ≤ x } := 
sorry

end intersection_A_B_l933_93395


namespace sum_of_distinct_digits_l933_93342

theorem sum_of_distinct_digits
  (w x y z : ℕ)
  (h1 : y + w = 10)
  (h2 : x + y = 9)
  (h3 : w + z = 10)
  (h4 : w ≠ x ∧ w ≠ y ∧ w ≠ z ∧ x ≠ y ∧ x ≠ z ∧ y ≠ z)
  (hw : w < 10) (hx : x < 10) (hy : y < 10) (hz : z < 10) :
  w + x + y + z = 20 := sorry

end sum_of_distinct_digits_l933_93342


namespace ball_count_l933_93305

theorem ball_count (r b y : ℕ) 
  (h1 : b + y = 9) 
  (h2 : r + y = 5) 
  (h3 : r + b = 6) : 
  r + b + y = 10 := 
  sorry

end ball_count_l933_93305


namespace equivalent_single_discount_l933_93303

theorem equivalent_single_discount (P : ℝ) (hP : 0 < P) : 
    let first_discount : ℝ := 0.15
    let second_discount : ℝ := 0.25
    let single_discount : ℝ := 0.3625
    (1 - first_discount) * (1 - second_discount) * P = (1 - single_discount) * P := by
    sorry

end equivalent_single_discount_l933_93303


namespace find_incorrect_statement_l933_93353

theorem find_incorrect_statement :
  ¬ (∀ a b c : ℝ, c ≠ 0 → (a < b → a * c^2 < b * c^2)) :=
by
  sorry

end find_incorrect_statement_l933_93353


namespace finite_S_k_iff_k_power_of_2_l933_93388

def S_k_finite (k : ℕ) : Prop :=
  ∃ (n a b : ℕ), (n ≠ 0 ∧ n % 2 = 1) ∧ (a + b = k) ∧ (Nat.gcd a b = 1) ∧ (n ∣ (a^n + b^n))

theorem finite_S_k_iff_k_power_of_2 (k : ℕ) (h : k > 1) : 
  (∀ n a b, n ≠ 0 → n % 2 = 1 → a + b = k → Nat.gcd a b = 1 → n ∣ (a^n + b^n) → false) ↔ 
  ∃ m : ℕ, k = 2^m :=
by
  sorry

end finite_S_k_iff_k_power_of_2_l933_93388


namespace problem_31_36_l933_93331

noncomputable def is_prime (n : ℕ) : Prop := sorry

theorem problem_31_36 (p k : ℕ) (hp : is_prime (4 * k + 1)) :
  (∃ x y m : ℕ, x^2 + y^2 = m * p) ∧ (∀ m > 1, ∃ x y m1 : ℕ, x^2 + y^2 = m * p ∧ 0 < m1 ∧ m1 < m) :=
by sorry

end problem_31_36_l933_93331


namespace lcm_of_18_and_20_l933_93394

theorem lcm_of_18_and_20 : Nat.lcm 18 20 = 180 := by
  sorry

end lcm_of_18_and_20_l933_93394


namespace value_of_w_div_x_l933_93348

theorem value_of_w_div_x (w x y : ℝ) 
  (h1 : w / x = a) 
  (h2 : w / y = 1 / 5) 
  (h3 : (x + y) / y = 2.2) : 
  w / x = 6 / 25 := by
  sorry

end value_of_w_div_x_l933_93348


namespace calculate_new_average_weight_l933_93328

noncomputable def new_average_weight (original_team_weight : ℕ) (num_original_players : ℕ) 
 (new_player1_weight : ℕ) (new_player2_weight : ℕ) (num_new_players : ℕ) : ℕ :=
 (original_team_weight + new_player1_weight + new_player2_weight) / (num_original_players + num_new_players)

theorem calculate_new_average_weight : 
  new_average_weight 847 7 110 60 2 = 113 := 
by 
sorry

end calculate_new_average_weight_l933_93328


namespace attendance_difference_l933_93392

theorem attendance_difference :
  let a := 65899
  let b := 66018
  b - a = 119 :=
sorry

end attendance_difference_l933_93392


namespace range_of_m_l933_93362

variable (x m : ℝ)

def alpha (x : ℝ) : Prop := x ≤ -5
def beta (x m : ℝ) : Prop := 2 * m - 3 ≤ x ∧ x ≤ 2 * m + 1

theorem range_of_m (x : ℝ) : (∀ x, beta x m → alpha x) → m ≤ -3 := by
  sorry

end range_of_m_l933_93362


namespace trapezoid_perimeter_l933_93350

theorem trapezoid_perimeter (height : ℝ) (radius : ℝ) (LM KN : ℝ) (LM_eq : LM = 16.5) (KN_eq : KN = 37.5)
  (LK MN : ℝ) (LK_eq : LK = 37.5) (MN_eq : MN = 37.5) (H : height = 36) (R : radius = 11) : 
  (LM + KN + LK + MN) = 129 :=
by
  -- The proof is omitted; only the statement is provided as specified.
  sorry

end trapezoid_perimeter_l933_93350


namespace find_area_of_triangle_ABQ_l933_93330

noncomputable def area_triangle_ABQ {A B C P Q R : Type*}
  (AP PB : ℝ) (area_ABC area_ABQ : ℝ) (h_areas_equal : area_ABQ = 15 / 2)
  (h_triangle_area : area_ABC = 15) (h_AP : AP = 3) (h_PB : PB = 2)
  (h_AB : AB = AP + PB) : Prop := area_ABQ = 15

theorem find_area_of_triangle_ABQ
  (A B C P Q R : Type*) (AP PB : ℝ)
  (h_triangle_area : area_ABC = 15)
  (h_AP : AP = 3) (h_PB : PB = 2)
  (h_AB : AB = AP + PB) (h_areas_equal : area_ABQ = 15 / 2) :
  area_ABQ = 15 := sorry

end find_area_of_triangle_ABQ_l933_93330


namespace remainder_3_pow_20_mod_7_l933_93377

theorem remainder_3_pow_20_mod_7 : (3^20) % 7 = 2 := 
by sorry

end remainder_3_pow_20_mod_7_l933_93377


namespace bill_original_selling_price_l933_93316

variable (P : ℝ) (S : ℝ) (S_new : ℝ)

theorem bill_original_selling_price :
  (S = P + 0.10 * P) ∧ (S_new = 0.90 * P + 0.27 * P) ∧ (S_new = S + 28) →
  S = 440 :=
by
  intro h
  sorry

end bill_original_selling_price_l933_93316


namespace compute_expression_l933_93355

theorem compute_expression :
  (5 + 7)^2 + (5^2 + 7^2) * 2 = 292 := by
  sorry

end compute_expression_l933_93355


namespace determine_event_C_l933_93315

variable (A B C : Prop)
variable (Tallest Shortest : Prop)
variable (Running LongJump ShotPut : Prop)

variables (part_A_Running part_A_LongJump part_A_ShotPut
           part_B_Running part_B_LongJump part_B_ShotPut
           part_C_Running part_C_LongJump part_C_ShotPut : Prop)

variable (not_tallest_A : ¬Tallest → A)
variable (not_tallest_ShotPut : Tallest → ¬ShotPut)
variable (shortest_LongJump : Shortest → LongJump)
variable (not_shortest_B : ¬Shortest → B)
variable (not_running_B : ¬Running → B)

theorem determine_event_C :
  (¬Tallest → A) →
  (Tallest → ¬ShotPut) →
  (Shortest → LongJump) →
  (¬Shortest → B) →
  (¬Running → B) →
  part_C_Running :=
by
  intros h1 h2 h3 h4 h5
  sorry

end determine_event_C_l933_93315


namespace max_value_fraction_l933_93327

theorem max_value_fraction (x y : ℝ) (hx : x > 0) (hy : y > 0) : 
  (x / (2 * x + y) + y / (x + 2 * y)) <= (2 / 3) := 
sorry

end max_value_fraction_l933_93327


namespace num_black_squares_in_37th_row_l933_93387

-- Define the total number of squares in the n-th row
def total_squares_in_row (n : ℕ) : ℕ := 2 * n - 1

-- Define the number of black squares in the n-th row
def black_squares_in_row (n : ℕ) : ℕ := (total_squares_in_row n - 1) / 2

theorem num_black_squares_in_37th_row : black_squares_in_row 37 = 36 :=
by
  sorry

end num_black_squares_in_37th_row_l933_93387


namespace find_a3_a4_a5_l933_93339

variable (a : ℕ → ℝ)

-- Recurrence relation for the sequence (condition for n ≥ 2)
axiom rec_relation (n : ℕ) (h : n ≥ 2) : 2 * a n = a (n - 1) + a (n + 1)

-- Additional conditions
axiom cond1 : a 1 + a 3 + a 5 = 9
axiom cond2 : a 3 + a 5 + a 7 = 15

-- Statement to prove
theorem find_a3_a4_a5 : a 3 + a 4 + a 5 = 12 :=
  sorry

end find_a3_a4_a5_l933_93339


namespace find_cos_sum_l933_93337

-- Defining the conditions based on the problem
variable (P A B C D : Type) (α β : ℝ)

-- Assumptions stating the given conditions
def regular_quadrilateral_pyramid (P A B C D : Type) : Prop :=
  -- Placeholder for the exact definition of a regular quadrilateral pyramid
  sorry

def dihedral_angle_lateral_base (P A B C D : Type) (α : ℝ) : Prop :=
  -- Placeholder for the exact property stating the dihedral angle between lateral face and base is α
  sorry

def dihedral_angle_adjacent_lateral (P A B C D : Type) (β : ℝ) : Prop :=
  -- Placeholder for the exact property stating the dihedral angle between two adjacent lateral faces is β
  sorry

-- The final theorem that we want to prove
theorem find_cos_sum (P A B C D : Type) (α β : ℝ)
  (H1 : regular_quadrilateral_pyramid P A B C D)
  (H2 : dihedral_angle_lateral_base P A B C D α)
  (H3 : dihedral_angle_adjacent_lateral P A B C D β) :
  2 * Real.cos β + Real.cos (2 * α) = -1 :=
sorry

end find_cos_sum_l933_93337


namespace billy_restaurant_bill_l933_93335

def adults : ℕ := 2
def children : ℕ := 5
def meal_cost : ℕ := 3

def total_people : ℕ := adults + children
def total_bill : ℕ := total_people * meal_cost

theorem billy_restaurant_bill : total_bill = 21 := 
by
  -- This is the placeholder for the proof.
  sorry

end billy_restaurant_bill_l933_93335


namespace find_f_property_l933_93361

noncomputable def f : ℝ → ℝ := sorry

theorem find_f_property :
  (f 0 = 3) ∧ (∀ x y : ℝ, f (xy) = f ((x^2 + y^2) / 2) + (x - y)^2) →
  (∀ x : ℝ, 0 ≤ x → f x = 3 - 2 * x) :=
by
  intros hypothesis
  -- Proof would be placed here
  sorry

end find_f_property_l933_93361


namespace circles_intersect_probability_l933_93368

noncomputable def probability_circles_intersect : ℝ :=
  sorry

theorem circles_intersect_probability :
  probability_circles_intersect = (5 * Real.sqrt 2 - 7) / 4 :=
  sorry

end circles_intersect_probability_l933_93368


namespace series_sum_eq_l933_93318

theorem series_sum_eq :
  (1^25 + 2^24 + 3^23 + 4^22 + 5^21 + 6^20 + 7^19 + 8^18 + 9^17 + 10^16 + 
  11^15 + 12^14 + 13^13 + 14^12 + 15^11 + 16^10 + 17^9 + 18^8 + 19^7 + 20^6 + 
  21^5 + 22^4 + 23^3 + 24^2 + 25^1) = 66071772829247409 := 
by
  sorry

end series_sum_eq_l933_93318


namespace trapezoid_area_l933_93313

noncomputable def area_of_trapezoid : ℝ :=
  let y1 := 12
  let y2 := 5
  let x1 := 12 / 2
  let x2 := 5 / 2
  ((x1 + x2) / 2) * (y1 - y2)

theorem trapezoid_area : area_of_trapezoid = 29.75 := by
  sorry

end trapezoid_area_l933_93313


namespace arithmetic_sequence_term_number_l933_93332

theorem arithmetic_sequence_term_number
  (a : ℕ → ℤ)
  (ha1 : a 1 = 1)
  (ha2 : a 2 = 3)
  (n : ℕ)
  (hn : a n = 217) :
  n = 109 :=
sorry

end arithmetic_sequence_term_number_l933_93332


namespace find_remainder_l933_93359

-- Definitions
variable (x y : ℕ)
variable (h1 : x > 0)
variable (h2 : y > 0)
variable (h3 : (x : ℝ) / y = 96.15)
variable (h4 : approximately_equal (y : ℝ) 60)

-- Target statement
theorem find_remainder : x % y = 9 :=
sorry

end find_remainder_l933_93359


namespace no_such_function_exists_l933_93399

theorem no_such_function_exists :
  ¬ ∃ (f : ℕ → ℕ), ∀ n : ℕ, f (f n) = n + 1987 := 
sorry

end no_such_function_exists_l933_93399


namespace hexagon_unique_intersection_points_are_45_l933_93379

-- Definitions related to hexagon for the proof problem
def hexagon_vertices : ℕ := 6
def sides_of_hexagon : ℕ := 6
def diagonals_of_hexagon : ℕ := 9
def total_line_segments : ℕ := 15
def total_intersections : ℕ := 105
def vertex_intersections_per_vertex : ℕ := 10
def total_vertex_intersections : ℕ := 60

-- Final Proof Statement that needs to be proved
theorem hexagon_unique_intersection_points_are_45 :
  total_intersections - total_vertex_intersections = 45 :=
by
  sorry

end hexagon_unique_intersection_points_are_45_l933_93379


namespace expression_value_l933_93398

theorem expression_value (a b : ℝ) (h₁ : a - 2 * b = 0) (h₂ : b ≠ 0) : 
  ( (b / (a - b) + 1) * (a^2 - b^2) / a^2 ) = 3 / 2 := 
by 
  sorry

end expression_value_l933_93398


namespace max_valid_committees_l933_93357

-- Define the conditions
def community_size : ℕ := 20
def english_speakers : ℕ := 10
def german_speakers : ℕ := 10
def french_speakers : ℕ := 10
def total_subsets : ℕ := Nat.choose community_size 3
def invalid_subsets_per_language : ℕ := Nat.choose 10 3

-- Lean statement to verify the number of valid committees
theorem max_valid_committees :
  total_subsets - 3 * invalid_subsets_per_language = 1020 :=
by
  simp [community_size, total_subsets, invalid_subsets_per_language]
  sorry

end max_valid_committees_l933_93357


namespace sufficient_but_not_necessary_l933_93311

theorem sufficient_but_not_necessary (m : ℕ) :
  m = 9 → m > 8 ∧ ∃ k : ℕ, k > 8 ∧ k ≠ 9 :=
by
  sorry

end sufficient_but_not_necessary_l933_93311


namespace rs_value_l933_93389

theorem rs_value (r s : ℝ) (h1 : 0 < r) (h2 : 0 < s) (h3 : r^2 + s^2 = 2) (h4 : r^4 + s^4 = 15 / 8) :
  r * s = (Real.sqrt 17) / 4 := 
sorry

end rs_value_l933_93389


namespace number_of_geese_is_correct_l933_93326

noncomputable def number_of_ducks := 37
noncomputable def total_number_of_birds := 95
noncomputable def number_of_geese := total_number_of_birds - number_of_ducks

theorem number_of_geese_is_correct : number_of_geese = 58 := by
  sorry

end number_of_geese_is_correct_l933_93326


namespace find_cost_price_l933_93382

theorem find_cost_price (SP : ℝ) (profit_percent : ℝ) (CP : ℝ) (h1 : SP = 715) (h2 : profit_percent = 0.10) (h3 : SP = CP * (1 + profit_percent)) : 
  CP = 650 :=
by
  sorry

end find_cost_price_l933_93382


namespace beavers_working_on_home_l933_93367

noncomputable def initial_beavers : ℝ := 2.0
noncomputable def additional_beavers : ℝ := 1.0

theorem beavers_working_on_home : initial_beavers + additional_beavers = 3.0 :=
by
  sorry

end beavers_working_on_home_l933_93367


namespace merchant_marked_price_l933_93312

theorem merchant_marked_price (L P x S : ℝ)
  (h1 : L = 100)
  (h2 : P = 70)
  (h3 : S = 0.8 * x)
  (h4 : 0.8 * x - 70 = 0.3 * (0.8 * x)) :
  x = 125 :=
by
  sorry

end merchant_marked_price_l933_93312


namespace percentage_of_men_l933_93365

theorem percentage_of_men (M W : ℝ) (h1 : M + W = 1) (h2 : 0.60 * M + 0.2364 * W = 0.40) : M = 0.45 :=
by
  sorry

end percentage_of_men_l933_93365


namespace arithmetic_sequence_k_is_10_l933_93349

noncomputable def a_n (n : ℕ) (d : ℝ) : ℝ := (n - 1) * d

theorem arithmetic_sequence_k_is_10 (d : ℝ) (h : d ≠ 0) : 
  (∃ k : ℕ, a_n k d = (a_n 1 d) + (a_n 2 d) + (a_n 3 d) + (a_n 4 d) + (a_n 5 d) + (a_n 6 d) + (a_n 7 d) ∧ k = 10) := 
by
  sorry

end arithmetic_sequence_k_is_10_l933_93349


namespace increase_by_40_percent_l933_93383

theorem increase_by_40_percent (initial_number : ℕ) (increase_rate : ℕ) :
  initial_number = 150 → increase_rate = 40 →
  initial_number + (increase_rate / 100 * initial_number) = 210 := by
  sorry

end increase_by_40_percent_l933_93383


namespace verify_option_a_l933_93356

-- Define Option A's condition
def option_a_condition (a : ℝ) : Prop :=
  2 * a^2 - 4 * a + 2 = 2 * (a - 1)^2

-- State the theorem that Option A's factorization is correct
theorem verify_option_a (a : ℝ) : option_a_condition a := by sorry

end verify_option_a_l933_93356


namespace number_of_correct_propositions_is_zero_l933_93352

-- Defining the propositions as functions
def proposition1 (f : ℝ → ℝ) (increasing_pos : ∀ x > 0, f x ≤ f (x + 1))
  (increasing_neg : ∀ x < 0, f x ≤ f (x + 1)) : Prop :=
  ∀ x1 x2, x1 ≤ x2 → f x1 ≤ f x2

def proposition2 (a b : ℝ) (no_intersection : ∀ x, a * x^2 + b * x + 2 ≠ 0) : Prop :=
  b^2 < 8 * a ∧ (a > 0 ∨ (a = 0 ∧ b = 0))

def proposition3 : Prop :=
  ∀ x, (x ≥ 1 → (x^2 - 2 * x - 3) ≥ (x^2 - 2 * (x + 1) - 3))

-- The main theorem to prove
theorem number_of_correct_propositions_is_zero :
  ∀ (f : ℝ → ℝ)
    (increasing_pos : ∀ x > 0, f x ≤ f (x + 1))
    (increasing_neg : ∀ x < 0, f x ≤ f (x + 1))
    (a b : ℝ)
    (no_intersection : ∀ x, a * x^2 + b * x + 2 ≠ 0),
    (¬ proposition1 f increasing_pos increasing_neg ∧
     ¬ proposition2 a b no_intersection ∧
     ¬ proposition3) :=
by
  sorry

end number_of_correct_propositions_is_zero_l933_93352


namespace candy_bar_calories_l933_93309

theorem candy_bar_calories:
  ∀ (calories_per_candy_bar : ℕ) (num_candy_bars : ℕ), 
  calories_per_candy_bar = 3 → 
  num_candy_bars = 5 → 
  calories_per_candy_bar * num_candy_bars = 15 :=
by
  sorry

end candy_bar_calories_l933_93309


namespace find_triplets_l933_93329

noncomputable def triplets_solution (x y z : ℝ) : Prop := 
  (x^2 + y^2 = -x + 3*y + z) ∧ 
  (y^2 + z^2 = x + 3*y - z) ∧ 
  (x^2 + z^2 = 2*x + 2*y - z) ∧ 
  (x ≠ y) ∧ (y ≠ z) ∧ (x ≠ z)

theorem find_triplets : 
  { (x, y, z) : ℝ × ℝ × ℝ | triplets_solution x y z } = 
  { (0, 1, -2), (-3/2, 5/2, -1/2) } :=
sorry

end find_triplets_l933_93329


namespace right_triangle_tangent_length_l933_93322

theorem right_triangle_tangent_length (DE DF : ℝ) (h1 : DE = 7) (h2 : DF = Real.sqrt 85)
  (h3 : ∀ (EF : ℝ), DE^2 + EF^2 = DF^2 → EF = 6): FQ = 6 :=
by
  sorry

end right_triangle_tangent_length_l933_93322


namespace treasure_coins_l933_93340

theorem treasure_coins (m : ℕ) 
  (h : (m / 3 + 1) + (m / 4 + 5) + (m / 5 + 20) = m) : 
  m = 120 := 
sorry

end treasure_coins_l933_93340


namespace solution_set_inequality_l933_93371

theorem solution_set_inequality (x : ℝ) (h : x ≠ 0) : 
  (x - 1) / x > 1 → x < 0 := 
by 
  sorry

end solution_set_inequality_l933_93371


namespace angle_in_parallelogram_l933_93396

theorem angle_in_parallelogram (EFGH : Parallelogram) (angle_EFG angle_FGH : ℝ)
  (h1 : angle_EFG = angle_FGH + 90) : angle_EHG = 45 :=
by sorry

end angle_in_parallelogram_l933_93396


namespace find_m_l933_93302

theorem find_m (a b m : ℝ) (h1 : 2^a = m) (h2 : 5^b = m) (h3 : 1/a + 1/b = 2) : m = Real.sqrt 10 :=
sorry

end find_m_l933_93302


namespace tan_7pi_over_6_l933_93366

noncomputable def tan_periodic (θ : ℝ) : Prop :=
  ∀ k : ℤ, Real.tan (θ + k * Real.pi) = Real.tan θ

theorem tan_7pi_over_6 : Real.tan (7 * Real.pi / 6) = Real.sqrt 3 / 3 :=
by
  sorry

end tan_7pi_over_6_l933_93366


namespace max_S_n_l933_93301

/-- Arithmetic sequence proof problem -/
theorem max_S_n (a : ℕ → ℝ) (S : ℕ → ℝ) 
  (h1 : a 1 + a 3 + a 5 = 15)
  (h2 : a 2 + a 4 + a 6 = 0)
  (d : ℝ) (h3 : ∀ n, a (n + 1) = a n + d) :
  (∃ n, S n = 30) :=
sorry

end max_S_n_l933_93301


namespace composite_expr_l933_93308

open Nat

def is_composite (n : ℕ) : Prop :=
  ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ a * b = n

theorem composite_expr (n : ℕ) : n ≥ 2 ↔ is_composite (3^(2*n + 1) - 2^(2*n + 1) - 6^n) :=
sorry

end composite_expr_l933_93308


namespace mary_total_nickels_l933_93325

theorem mary_total_nickels (n1 n2 : ℕ) (h1 : n1 = 7) (h2 : n2 = 5) : n1 + n2 = 12 := by
  sorry

end mary_total_nickels_l933_93325


namespace problem_equivalent_l933_93358

noncomputable def h (y : ℝ) : ℝ := y^5 - y^3 + 2
noncomputable def k (y : ℝ) : ℝ := y^2 - 3

theorem problem_equivalent (y₁ y₂ y₃ y₄ y₅ : ℝ) (h_roots : ∀ y, h y = 0 ↔ y = y₁ ∨ y = y₂ ∨ y = y₃ ∨ y = y₄ ∨ y = y₅) :
  (k y₁) * (k y₂) * (k y₃) * (k y₄) * (k y₅) = 104 :=
sorry

end problem_equivalent_l933_93358


namespace combined_age_of_staff_l933_93300

/--
In a school, the average age of a class of 50 students is 25 years. 
The average age increased by 2 years when the ages of 5 additional 
staff members, including the teacher, are also taken into account. 
Prove that the combined age of these 5 staff members is 235 years.
-/
theorem combined_age_of_staff 
    (n_students : ℕ) (avg_age_students : ℕ) (n_staff : ℕ) (avg_age_total : ℕ)
    (h1 : n_students = 50) 
    (h2 : avg_age_students = 25) 
    (h3 : n_staff = 5) 
    (h4 : avg_age_total = 27) :
  n_students * avg_age_students + (n_students + n_staff) * avg_age_total - 
  n_students * avg_age_students = 235 :=
by
  sorry

end combined_age_of_staff_l933_93300


namespace find_positive_integer_x_l933_93376

theorem find_positive_integer_x :
  ∃ x : ℕ, x > 0 ∧ (5 * x + 1) / (x - 1) > 2 * x + 2 ∧
  ∀ y : ℕ, y > 0 ∧ (5 * y + 1) / (y - 1) > 2 * x + 2 → y = 2 :=
sorry

end find_positive_integer_x_l933_93376


namespace square_of_equal_side_of_inscribed_triangle_l933_93306

theorem square_of_equal_side_of_inscribed_triangle :
  ∀ (x y : ℝ),
  (x^2 + 9 * y^2 = 9) →
  ((x = 0) → (y = 1)) →
  ((x ≠ 0) → y = (x + 1)) →
  square_of_side = (324 / 25) :=
by
  intros x y hEllipse hVertex hSlope
  sorry

end square_of_equal_side_of_inscribed_triangle_l933_93306


namespace hexadecagon_area_l933_93363

theorem hexadecagon_area (r : ℝ) : 
  (∃ A : ℝ, A = 4 * r^2 * Real.sqrt (2 - Real.sqrt 2)) :=
sorry

end hexadecagon_area_l933_93363


namespace calories_in_dressing_l933_93347

noncomputable def lettuce_calories : ℝ := 50
noncomputable def carrot_calories : ℝ := 2 * lettuce_calories
noncomputable def crust_calories : ℝ := 600
noncomputable def pepperoni_calories : ℝ := crust_calories / 3
noncomputable def cheese_calories : ℝ := 400

noncomputable def salad_calories : ℝ := lettuce_calories + carrot_calories
noncomputable def pizza_calories : ℝ := crust_calories + pepperoni_calories + cheese_calories

noncomputable def salad_eaten : ℝ := salad_calories / 4
noncomputable def pizza_eaten : ℝ := pizza_calories / 5

noncomputable def total_eaten : ℝ := salad_eaten + pizza_eaten

theorem calories_in_dressing : ((330 : ℝ) - total_eaten) = 52.5 := by
  sorry

end calories_in_dressing_l933_93347


namespace count_sums_of_three_cubes_l933_93320

theorem count_sums_of_three_cubes :
  let possible_sums := {n | ∃ a b c : ℕ, 1 ≤ a ∧ a ≤ 9 ∧ 1 ≤ b ∧ b ≤ 9 ∧ 1 ≤ c ∧ c ≤ 9 ∧ n = a^3 + b^3 + c^3}
  ∃ unique_sums : Finset ℕ, (∀ x ∈ possible_sums, x < 1000) ∧ unique_sums.card = 153 :=
by sorry

end count_sums_of_three_cubes_l933_93320


namespace range_of_s_l933_93344

noncomputable def s (x : ℝ) := 1 / (2 + x)^3

theorem range_of_s :
  Set.range s = {y : ℝ | y < 0} ∪ {y : ℝ | y > 0} :=
by
  sorry

end range_of_s_l933_93344


namespace range_of_a_l933_93370

/--
Let f be a function defined on the interval [-1, 1] that is increasing and odd.
If f(-a+1) + f(4a-5) > 0, then the range of the real number a is (4/3, 3/2].
-/
theorem range_of_a
  (f : ℝ → ℝ)
  (h_dom : ∀ x, -1 ≤ x ∧ x ≤ 1 → f x = f x)  -- domain condition
  (h_incr : ∀ x y, x < y → f x < f y)          -- increasing condition
  (h_odd : ∀ x, f (-x) = - f x)                -- odd function condition
  (a : ℝ)
  (h_ineq : f (-a + 1) + f (4 * a - 5) > 0) :
  4 / 3 < a ∧ a ≤ 3 / 2 :=
sorry

end range_of_a_l933_93370


namespace arithmetic_sequence_x_value_l933_93336

theorem arithmetic_sequence_x_value
  (x : ℝ)
  (h₁ : 2 * x - (1 / 3) = (x + 4) - 2 * x) :
  x = 13 / 3 := by
  sorry

end arithmetic_sequence_x_value_l933_93336


namespace cylinder_height_proof_l933_93351

noncomputable def cone_base_radius : ℝ := 15
noncomputable def cone_height : ℝ := 25
noncomputable def cylinder_base_radius : ℝ := 10
noncomputable def cylinder_water_height : ℝ := 18.75

theorem cylinder_height_proof :
  (1 / 3 * π * cone_base_radius^2 * cone_height) = π * cylinder_base_radius^2 * cylinder_water_height :=
by sorry

end cylinder_height_proof_l933_93351


namespace meat_cost_per_pound_l933_93343

def total_cost_box : ℝ := 5
def cost_per_bell_pepper : ℝ := 1.5
def num_bell_peppers : ℝ := 4
def num_pounds_meat : ℝ := 2
def total_spent : ℝ := 17

theorem meat_cost_per_pound : total_spent - (total_cost_box + num_bell_peppers * cost_per_bell_pepper) = 6 -> 
                             6 / num_pounds_meat = 3 := by
  sorry

end meat_cost_per_pound_l933_93343


namespace sector_area_is_nine_l933_93375

-- Given the conditions: the perimeter of the sector is 12 cm and the central angle is 2 radians
def sector_perimeter_radius (r : ℝ) :=
  4 * r = 12

def sector_angle : ℝ := 2

-- Prove that the area of the sector is 9 cm²
theorem sector_area_is_nine (r : ℝ) (s : ℝ) (h : sector_perimeter_radius r) (h_angle : sector_angle = 2) :
  s = 9 :=
by
  sorry

end sector_area_is_nine_l933_93375


namespace algebraic_expression_evaluation_l933_93360

theorem algebraic_expression_evaluation (a : ℝ) (h : a^2 + 2 * a - 1 = 0) :
  ( ( (a^2 - 1) / (a^2 - 2 * a + 1) - 1 / (1 - a)) / (1 / (a^2 - a)) ) = 1 :=
by sorry

end algebraic_expression_evaluation_l933_93360


namespace geometric_progression_common_ratio_l933_93324

theorem geometric_progression_common_ratio (r : ℝ) :
  (r > 0) ∧ (r^3 + r^2 + r - 1 = 0) ↔
  r = ( -1 + ((19 + 3 * Real.sqrt 33)^(1/3)) + ((19 - 3 * Real.sqrt 33)^(1/3)) ) / 3 :=
by
  sorry

end geometric_progression_common_ratio_l933_93324


namespace volume_of_rectangular_solid_l933_93345

theorem volume_of_rectangular_solid
  (a b c : ℝ)
  (h1 : a * b = 3)
  (h2 : a * c = 5)
  (h3 : b * c = 15) :
  a * b * c = 15 :=
sorry

end volume_of_rectangular_solid_l933_93345


namespace smallest_k_for_square_l933_93333

theorem smallest_k_for_square : ∃ k : ℕ, (2016 * 2017 * 2018 * 2019 + k) = n^2 ∧ k = 1 :=
by
  use 1
  sorry

end smallest_k_for_square_l933_93333


namespace number_of_boys_at_reunion_l933_93354

theorem number_of_boys_at_reunion (n : ℕ) (h : (n * (n - 1)) / 2 = 21) : n = 7 :=
by
  sorry

end number_of_boys_at_reunion_l933_93354


namespace find_SSE_l933_93384

theorem find_SSE (SST SSR : ℝ) (h1 : SST = 13) (h2 : SSR = 10) : SST - SSR = 3 :=
by
  sorry

end find_SSE_l933_93384


namespace number_of_players_in_tournament_l933_93310

theorem number_of_players_in_tournament (G : ℕ) (h1 : G = 42) (h2 : ∀ n : ℕ, G = n * (n - 1)) : ∃ n : ℕ, G = 42 ∧ n = 7 :=
by
  -- Let's suppose n is the number of players, then we need to prove
  -- ∃ n : ℕ, 42 = n * (n - 1) ∧ n = 7
  sorry

end number_of_players_in_tournament_l933_93310


namespace pharmacy_incurs_loss_l933_93346

variable (a b : ℝ)
variable (h : a < b)

theorem pharmacy_incurs_loss 
  (H : (41 * a + 59 * b) > 100 * (a + b) / 2) : true :=
by
  sorry

end pharmacy_incurs_loss_l933_93346


namespace prove_identical_numbers_l933_93372

variable {x y : ℝ}

theorem prove_identical_numbers (hx : x ≠ 0) (hy : y ≠ 0)
    (h1 : x + (1 / y^2) = y + (1 / x^2))
    (h2 : y^2 + (1 / x) = x^2 + (1 / y)) : x = y :=
by 
  sorry

end prove_identical_numbers_l933_93372


namespace pairs_satisfy_ineq_l933_93381

theorem pairs_satisfy_ineq (x y : ℝ) :
  |Real.sin x - Real.sin y| + Real.sin x * Real.sin y ≤ 0 ↔
  ∃ n m : ℤ, x = n * Real.pi ∧ y = m * Real.pi := 
sorry

end pairs_satisfy_ineq_l933_93381


namespace calculate_plot_size_in_acres_l933_93391

theorem calculate_plot_size_in_acres :
  let bottom_edge_cm : ℝ := 15
  let top_edge_cm : ℝ := 10
  let height_cm : ℝ := 10
  let cm_to_miles : ℝ := 3
  let miles_to_acres : ℝ := 640
  let trapezoid_area_cm2 := (bottom_edge_cm + top_edge_cm) * height_cm / 2
  let trapezoid_area_miles2 := trapezoid_area_cm2 * (cm_to_miles ^ 2)
  (trapezoid_area_miles2 * miles_to_acres) = 720000 :=
by
  sorry

end calculate_plot_size_in_acres_l933_93391


namespace visited_both_countries_l933_93338

theorem visited_both_countries {Total Iceland Norway Neither Both : ℕ} 
  (h1 : Total = 50) 
  (h2 : Iceland = 25)
  (h3 : Norway = 23)
  (h4 : Neither = 23) 
  (h5 : Total - Neither = 27) 
  (h6 : Iceland + Norway - Both = 27) : 
  Both = 21 := 
by
  sorry

end visited_both_countries_l933_93338


namespace max_regions_two_convex_polygons_l933_93319

theorem max_regions_two_convex_polygons (M N : ℕ) (hM : M > N) :
    ∃ R, R = 2 * N + 2 := 
sorry

end max_regions_two_convex_polygons_l933_93319


namespace fraction_of_white_roses_l933_93390

open Nat

def rows : ℕ := 10
def roses_per_row : ℕ := 20
def total_roses : ℕ := rows * roses_per_row
def red_roses : ℕ := total_roses / 2
def pink_roses : ℕ := 40
def white_roses : ℕ := total_roses - red_roses - pink_roses
def remaining_roses : ℕ := white_roses + pink_roses
def fraction_white_roses : ℚ := white_roses / remaining_roses

theorem fraction_of_white_roses :
  fraction_white_roses = 3 / 5 :=
by
  sorry

end fraction_of_white_roses_l933_93390


namespace circle_radius_eq_one_l933_93385

theorem circle_radius_eq_one (x y : ℝ) : (16 * x^2 - 32 * x + 16 * y^2 + 64 * y + 64 = 0) → (1 = 1) :=
by
  intros h
  sorry

end circle_radius_eq_one_l933_93385


namespace value_of_x_l933_93314

theorem value_of_x (x : ℤ) (h : x + 3 = 4 ∨ x + 3 = -4) : x = 1 ∨ x = -7 := sorry

end value_of_x_l933_93314


namespace problem1_problem2_problem3_l933_93373

-- Problem 1
theorem problem1 (x : ℝ) (h : 0 < x ∧ x < 1/2) : 
  (1/2 * x * (1 - 2 * x) ≤ 1/16) := sorry

-- Problem 2
theorem problem2 (x : ℝ) (h : 0 < x) : 
  (2 - x - 4 / x ≤ -2) := sorry

-- Problem 3
theorem problem3 (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + y = 4) : 
  (1 / x + 3 / y ≥ 1 + Real.sqrt 3 / 2) := sorry

end problem1_problem2_problem3_l933_93373
