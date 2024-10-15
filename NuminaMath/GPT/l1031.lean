import Mathlib

namespace NUMINAMATH_GPT_distance_to_x_axis_l1031_103140

def point_P : ℝ × ℝ := (2, -3)

theorem distance_to_x_axis : abs (point_P.snd) = 3 := by
  sorry

end NUMINAMATH_GPT_distance_to_x_axis_l1031_103140


namespace NUMINAMATH_GPT_kids_tubing_and_rafting_l1031_103192

theorem kids_tubing_and_rafting 
  (total_kids : ℕ) 
  (one_fourth_tubing : ℕ)
  (half_rafting : ℕ)
  (h1 : total_kids = 40)
  (h2 : one_fourth_tubing = total_kids / 4)
  (h3 : half_rafting = one_fourth_tubing / 2) :
  half_rafting = 5 :=
by
  sorry

end NUMINAMATH_GPT_kids_tubing_and_rafting_l1031_103192


namespace NUMINAMATH_GPT_poly_at_2_eq_0_l1031_103185

def poly (x : ℝ) : ℝ := x^6 - 12 * x^5 + 60 * x^4 - 160 * x^3 + 240 * x^2 - 192 * x + 64

theorem poly_at_2_eq_0 : poly 2 = 0 := by
  sorry

end NUMINAMATH_GPT_poly_at_2_eq_0_l1031_103185


namespace NUMINAMATH_GPT_candle_blow_out_l1031_103163

-- Definitions related to the problem.
def funnel := true -- Simplified representation of the funnel
def candle_lit := true -- Simplified representation of the lit candle
def airflow_concentration (align: Bool) : Prop :=
if align then true -- Airflow intersects the flame correctly
else false -- Airflow does not intersect the flame correctly

theorem candle_blow_out (align : Bool) : funnel ∧ candle_lit ∧ airflow_concentration align → align := sorry

end NUMINAMATH_GPT_candle_blow_out_l1031_103163


namespace NUMINAMATH_GPT_f_value_at_3_l1031_103175

noncomputable def f : ℝ → ℝ := sorry

def odd_function (f : ℝ → ℝ) :=
  ∀ x : ℝ, f (-x) = -f x

def periodic_shift (f : ℝ → ℝ) :=
  ∀ x : ℝ, f (x + 2) = f x + 2

theorem f_value_at_3 (h_odd : odd_function f) (h_value : f (-1) = 1/2) (h_periodic : periodic_shift f) : 
  f 3 = 3 / 2 := 
sorry

end NUMINAMATH_GPT_f_value_at_3_l1031_103175


namespace NUMINAMATH_GPT_shekar_biology_marks_l1031_103174

theorem shekar_biology_marks (M S SS E A n B : ℕ) 
  (hM : M = 76)
  (hS : S = 65)
  (hSS : SS = 82)
  (hE : E = 67)
  (hA : A = 73)
  (hn : n = 5)
  (hA_eq : A = (M + S + SS + E + B) / n) : 
  B = 75 := 
by
  rw [hM, hS, hSS, hE, hn, hA] at hA_eq
  sorry

end NUMINAMATH_GPT_shekar_biology_marks_l1031_103174


namespace NUMINAMATH_GPT_cary_ivy_removal_days_correct_l1031_103125

noncomputable def cary_ivy_removal_days (initial_ivy : ℕ) (ivy_removed_per_day : ℕ) (ivy_growth_per_night : ℕ) : ℕ :=
  initial_ivy / (ivy_removed_per_day - ivy_growth_per_night)

theorem cary_ivy_removal_days_correct :
  cary_ivy_removal_days 40 6 2 = 10 :=
by
  -- The body of the proof is omitted; it will be filled with the actual proof.
  sorry

end NUMINAMATH_GPT_cary_ivy_removal_days_correct_l1031_103125


namespace NUMINAMATH_GPT_students_in_second_class_l1031_103198

theorem students_in_second_class 
    (avg1 : ℝ)
    (n1 : ℕ)
    (avg2 : ℝ)
    (total_avg : ℝ)
    (x : ℕ)
    (h1 : avg1 = 40)
    (h2 : n1 = 26)
    (h3 : avg2 = 60)
    (h4 : total_avg = 53.1578947368421)
    (h5 : (n1 * avg1 + x * avg2) / (n1 + x) = total_avg) :
  x = 50 :=
by
  sorry

end NUMINAMATH_GPT_students_in_second_class_l1031_103198


namespace NUMINAMATH_GPT_remainder_of_p_l1031_103170

theorem remainder_of_p (p : ℤ) (h1 : p = 35 * 17 + 10) : p % 35 = 10 := 
  sorry

end NUMINAMATH_GPT_remainder_of_p_l1031_103170


namespace NUMINAMATH_GPT_book_selling_price_l1031_103168

def cost_price : ℕ := 225
def profit_percentage : ℚ := 0.20
def selling_price := cost_price + (profit_percentage * cost_price)

theorem book_selling_price :
  selling_price = 270 :=
by
  sorry

end NUMINAMATH_GPT_book_selling_price_l1031_103168


namespace NUMINAMATH_GPT_sequence_period_16_l1031_103178

theorem sequence_period_16 (a : ℝ) (h : a > 0) 
  (u : ℕ → ℝ) (h1 : u 1 = a) (h2 : ∀ n, u (n + 1) = -1 / (u n + 1)) : 
  u 16 = a :=
sorry

end NUMINAMATH_GPT_sequence_period_16_l1031_103178


namespace NUMINAMATH_GPT_games_bought_at_garage_sale_l1031_103134

-- Definitions based on conditions
def games_from_friend : ℕ := 2
def defective_games : ℕ := 2
def good_games : ℕ := 2

-- Prove the number of games bought at the garage sale equals 2
theorem games_bought_at_garage_sale (G : ℕ) 
  (h : games_from_friend + G - defective_games = good_games) : G = 2 :=
by 
  -- use the given information and work out the proof here
  sorry

end NUMINAMATH_GPT_games_bought_at_garage_sale_l1031_103134


namespace NUMINAMATH_GPT_geometric_seq_arith_mean_l1031_103153

theorem geometric_seq_arith_mean 
  (b : ℕ → ℝ) 
  (r : ℝ) 
  (b_geom : ∀ n, b (n + 1) = r * b n)
  (h_arith_mean : b 9 = (3 + 5) / 2) :
  b 1 * b 17 = 16 :=
by
  sorry

end NUMINAMATH_GPT_geometric_seq_arith_mean_l1031_103153


namespace NUMINAMATH_GPT_original_number_is_144_l1031_103105

theorem original_number_is_144 (A B C : ℕ) (A_digit : A < 10) (B_digit : B < 10) (C_digit : C < 10)
  (h1 : 100 * A + 10 * B + B = 144)
  (h2 : A * B * B = 10 * A + C)
  (h3 : (10 * A + C) % 10 = C) : 100 * A + 10 * B + B = 144 := 
sorry

end NUMINAMATH_GPT_original_number_is_144_l1031_103105


namespace NUMINAMATH_GPT_product_N_l1031_103100

theorem product_N (A D D1 A1 : ℤ) (N : ℤ) 
  (h1 : D = A - N)
  (h2 : D1 = D + 7)
  (h3 : A1 = A - 2)
  (h4 : |D1 - A1| = 8) : 
  N = 1 → N = 17 → N * 17 = 17 :=
by
  sorry

end NUMINAMATH_GPT_product_N_l1031_103100


namespace NUMINAMATH_GPT_fg_at_2_l1031_103126

def f (x : ℝ) : ℝ := x^3
def g (x : ℝ) : ℝ := 2*x + 5

theorem fg_at_2 : f (g 2) = 729 := by
  sorry

end NUMINAMATH_GPT_fg_at_2_l1031_103126


namespace NUMINAMATH_GPT_function_is_monotonically_increasing_l1031_103132

theorem function_is_monotonically_increasing (a : ℝ) :
  (∀ x : ℝ, (x^2 + 2*x + a) ≥ 0) ↔ (1 ≤ a) := 
sorry

end NUMINAMATH_GPT_function_is_monotonically_increasing_l1031_103132


namespace NUMINAMATH_GPT_largest_of_three_l1031_103107

theorem largest_of_three (a b c : ℝ) (h₁ : a = 43.23) (h₂ : b = 2/5) (h₃ : c = 21.23) :
  max (max a b) c = a :=
by
  sorry

end NUMINAMATH_GPT_largest_of_three_l1031_103107


namespace NUMINAMATH_GPT_volume_of_triangular_pyramid_l1031_103147

variable (a b : ℝ)

noncomputable def volume_of_pyramid (a b : ℝ) : ℝ :=
  (a * b / 12) * Real.sqrt (3 * b ^ 2 - a ^ 2)

theorem volume_of_triangular_pyramid (a b : ℝ) :
  volume_of_pyramid a b = (a * b / 12) * Real.sqrt (3 * b ^ 2 - a ^ 2) :=
by
  sorry

end NUMINAMATH_GPT_volume_of_triangular_pyramid_l1031_103147


namespace NUMINAMATH_GPT_daryl_max_crate_weight_l1031_103120

variable (crates : ℕ) (weight_nails : ℕ) (bags_nails : ℕ)
variable (weight_hammers : ℕ) (bags_hammers : ℕ) (weight_planks : ℕ)
variable (bags_planks : ℕ) (weight_left_out : ℕ)

def max_weight_per_crate (total_weight: ℕ) (total_crates: ℕ) : ℕ :=
  total_weight / total_crates

-- State the problem in Lean
theorem daryl_max_crate_weight
  (h1 : crates = 15) 
  (h2 : bags_nails = 4) 
  (h3 : weight_nails = 5)
  (h4 : bags_hammers = 12) 
  (h5 : weight_hammers = 5) 
  (h6 : bags_planks = 10) 
  (h7 : weight_planks = 30) 
  (h8 : weight_left_out = 80):
  max_weight_per_crate ((bags_nails * weight_nails + bags_hammers * weight_hammers + bags_planks * weight_planks) - weight_left_out) crates = 20 :=
  by sorry

end NUMINAMATH_GPT_daryl_max_crate_weight_l1031_103120


namespace NUMINAMATH_GPT_find_a_values_l1031_103193

def setA : Set ℝ := {-1, 1/2, 1}
def setB (a : ℝ) : Set ℝ := {x | a * x^2 = 1 ∧ a ≥ 0}

def full_food (A B : Set ℝ) : Prop := A ⊆ B ∨ B ⊆ A
def partial_food (A B : Set ℝ) : Prop := (∃ x, x ∈ A ∧ x ∈ B) ∧ ¬(A ⊆ B ∨ B ⊆ A)

theorem find_a_values :
  ∀ a : ℝ, full_food setA (setB a) ∨ partial_food setA (setB a) ↔ a = 0 ∨ a = 1 ∨ a = 4 := 
by
  sorry

end NUMINAMATH_GPT_find_a_values_l1031_103193


namespace NUMINAMATH_GPT_consecutive_odd_numbers_first_l1031_103167

theorem consecutive_odd_numbers_first :
  ∃ x : ℤ, 11 * x = 3 * (x + 4) + 4 * (x + 2) + 16 ∧ x = 9 :=
by 
  sorry

end NUMINAMATH_GPT_consecutive_odd_numbers_first_l1031_103167


namespace NUMINAMATH_GPT_min_value_of_3x_plus_4y_is_5_l1031_103169

theorem min_value_of_3x_plus_4y_is_5 :
  ∀ (x y : ℝ), 0 < x → 0 < y → (3 / x + 1 / y = 5) → (∃ (b : ℝ), b = 3 * x + 4 * y ∧ ∀ (x y : ℝ), 0 < x → 0 < y → (3 / x + 1 / y = 5) → 3 * x + 4 * y ≥ b) :=
by
  intro x y x_pos y_pos h_eq
  let b := 5
  use b
  simp [b]
  sorry

end NUMINAMATH_GPT_min_value_of_3x_plus_4y_is_5_l1031_103169


namespace NUMINAMATH_GPT_amy_total_score_correct_l1031_103173

def amyTotalScore (points_per_treasure : ℕ) (treasures_first_level : ℕ) (treasures_second_level : ℕ) : ℕ :=
  (points_per_treasure * treasures_first_level) + (points_per_treasure * treasures_second_level)

theorem amy_total_score_correct:
  amyTotalScore 4 6 2 = 32 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_amy_total_score_correct_l1031_103173


namespace NUMINAMATH_GPT_tan_sin_div_l1031_103148

theorem tan_sin_div (tan_30_sq : Real := (Real.sin 30 / Real.cos 30) ^ 2)
                    (sin_30_sq : Real := (Real.sin 30) ^ 2):
                    tan_30_sq = (1/3) → sin_30_sq = (1/4) → 
                    (tan_30_sq - sin_30_sq) / (tan_30_sq * sin_30_sq) = 1 :=
by
  sorry

end NUMINAMATH_GPT_tan_sin_div_l1031_103148


namespace NUMINAMATH_GPT_number_of_diagonals_in_decagon_l1031_103183

-- Definition of the problem condition: a polygon with n = 10 sides
def n : ℕ := 10

-- Theorem stating the number of diagonals in a regular decagon
theorem number_of_diagonals_in_decagon : (n * (n - 3)) / 2 = 35 :=
by
  -- Proof steps will go here
  sorry

end NUMINAMATH_GPT_number_of_diagonals_in_decagon_l1031_103183


namespace NUMINAMATH_GPT_find_k_range_l1031_103136

open Nat

def a_n (n : ℕ) : ℕ := 2^ (5 - n)

def b_n (n : ℕ) (k : ℤ) : ℤ := n + k

def c_n (n : ℕ) (k : ℤ) : ℤ :=
if (a_n n : ℤ) ≤ (b_n n k) then b_n n k else a_n n

theorem find_k_range : 
  (∀ n ∈ { m : ℕ | m > 0 }, c_n 5 = a_n 5 ∧ c_n 5 ≤ c_n n) → 
  (∃ k : ℤ, -5 ≤ k ∧ k ≤ -3) :=
by
  sorry

end NUMINAMATH_GPT_find_k_range_l1031_103136


namespace NUMINAMATH_GPT_blocks_for_sculpture_l1031_103149

noncomputable def volume_block := 8 * 3 * 1
noncomputable def radius_cylinder := 3
noncomputable def height_cylinder := 8
noncomputable def volume_cylinder := Real.pi * radius_cylinder^2 * height_cylinder
noncomputable def blocks_needed := Nat.ceil (volume_cylinder / volume_block)

theorem blocks_for_sculpture : blocks_needed = 10 := by
  sorry

end NUMINAMATH_GPT_blocks_for_sculpture_l1031_103149


namespace NUMINAMATH_GPT_classrooms_students_guinea_pigs_difference_l1031_103160

theorem classrooms_students_guinea_pigs_difference :
  let students_per_classroom := 22
  let guinea_pigs_per_classroom := 3
  let number_of_classrooms := 5
  let total_students := students_per_classroom * number_of_classrooms
  let total_guinea_pigs := guinea_pigs_per_classroom * number_of_classrooms
  total_students - total_guinea_pigs = 95 :=
  by
    sorry

end NUMINAMATH_GPT_classrooms_students_guinea_pigs_difference_l1031_103160


namespace NUMINAMATH_GPT_axis_of_symmetry_l1031_103152

-- Define the condition for the parabola equation
def parabola_equation (x y : ℝ) : Prop :=
  x = -4 * y^2

-- Define the statement that needs to be proven
theorem axis_of_symmetry (x : ℝ) (y : ℝ) (h : parabola_equation x y) : x = 1 / 16 :=
  sorry

end NUMINAMATH_GPT_axis_of_symmetry_l1031_103152


namespace NUMINAMATH_GPT_select_terms_from_sequence_l1031_103102

theorem select_terms_from_sequence (k : ℕ) (hk : k ≥ 3) :
  ∃ (terms : Fin k → ℚ), (∀ i j : Fin k, i < j → (terms j - terms i) = (j.val - i.val) / k!) ∧
  (∀ i : Fin k, terms i ∈ {x : ℚ | ∃ n : ℕ, x = 1 / (n : ℚ)}) :=
by
  sorry

end NUMINAMATH_GPT_select_terms_from_sequence_l1031_103102


namespace NUMINAMATH_GPT_cans_per_bag_l1031_103171

theorem cans_per_bag (bags_on_Saturday bags_on_Sunday total_cans : ℕ) (h_saturday : bags_on_Saturday = 3) (h_sunday : bags_on_Sunday = 4) (h_total : total_cans = 63) :
  (total_cans / (bags_on_Saturday + bags_on_Sunday) = 9) :=
by {
  sorry
}

end NUMINAMATH_GPT_cans_per_bag_l1031_103171


namespace NUMINAMATH_GPT_fish_filets_total_l1031_103127

def fish_caught_by_ben : ℕ := 4
def fish_caught_by_judy : ℕ := 1
def fish_caught_by_billy : ℕ := 3
def fish_caught_by_jim : ℕ := 2
def fish_caught_by_susie : ℕ := 5
def fish_thrown_back : ℕ := 3
def filets_per_fish : ℕ := 2

theorem fish_filets_total : 
  (fish_caught_by_ben + fish_caught_by_judy + fish_caught_by_billy + fish_caught_by_jim + fish_caught_by_susie - fish_thrown_back) * filets_per_fish = 24 := 
by
  sorry

end NUMINAMATH_GPT_fish_filets_total_l1031_103127


namespace NUMINAMATH_GPT_complement_intersection_l1031_103110

def U : Set ℝ := fun x => True
def A : Set ℝ := fun x => x < 0
def B : Set ℝ := fun x => x = -2 ∨ x = -1 ∨ x = 0 ∨ x = 1 ∨ x = 2

theorem complement_intersection (hU : ∀ x : ℝ, U x) :
  ((compl A) ∩ B) = {0, 1, 2} :=
by {
  sorry
}

end NUMINAMATH_GPT_complement_intersection_l1031_103110


namespace NUMINAMATH_GPT_supplement_of_complement_is_125_l1031_103130

-- Definition of the initial angle
def initial_angle : ℝ := 35

-- Definition of the complement of the angle
def complement_angle (θ : ℝ) : ℝ := 90 - θ

-- Definition of the supplement of an angle
def supplement_angle (θ : ℝ) : ℝ := 180 - θ

-- Main theorem statement
theorem supplement_of_complement_is_125 : 
  supplement_angle (complement_angle initial_angle) = 125 := 
by
  sorry

end NUMINAMATH_GPT_supplement_of_complement_is_125_l1031_103130


namespace NUMINAMATH_GPT_odd_function_value_l1031_103106

noncomputable def f (x : ℝ) : ℝ := sorry -- Placeholder for the function definition

-- Prove that f(-1/2) = -1/2 given the conditions
theorem odd_function_value :
  (∀ x : ℝ, f (-x) = -f x) ∧ (∀ x : ℝ, 0 ≤ x ∧ x < 1 → f x = x) →
  f (-1/2) = -1/2 :=
by
  sorry

end NUMINAMATH_GPT_odd_function_value_l1031_103106


namespace NUMINAMATH_GPT_buying_pets_l1031_103117

theorem buying_pets {puppies kittens hamsters birds : ℕ} :
(∃ pets : ℕ, pets = 12 * 8 * 10 * 5 * 4 * 3 * 2) ∧ 
puppies = 12 ∧ kittens = 8 ∧ hamsters = 10 ∧ birds = 5 → 
12 * 8 * 10 * 5 * 4 * 3 * 2 = 115200 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_buying_pets_l1031_103117


namespace NUMINAMATH_GPT_measure_of_one_interior_angle_of_regular_octagon_l1031_103137

-- Define the conditions for the problem
def num_sides : ℕ := 8
def sum_of_interior_angles (n : ℕ) : ℕ := 180 * (n - 2)

-- Statement of the problem
theorem measure_of_one_interior_angle_of_regular_octagon : 
  (sum_of_interior_angles num_sides) / num_sides = 135 :=
by
  -- Conditions are defined above
  -- Proof steps are omitted
  sorry

end NUMINAMATH_GPT_measure_of_one_interior_angle_of_regular_octagon_l1031_103137


namespace NUMINAMATH_GPT_parabola_p_q_r_sum_l1031_103131

noncomputable def parabola_vertex (p q r : ℝ) (x_vertex y_vertex : ℝ) :=
  ∀ (x : ℝ), p * (x - x_vertex) ^ 2 + y_vertex = p * x ^ 2 + q * x + r

theorem parabola_p_q_r_sum
  (p q r : ℝ)
  (vertex_x vertex_y : ℝ)
  (hx_vertex : vertex_x = 3)
  (hy_vertex : vertex_y = 10)
  (h_vertex : parabola_vertex p q r vertex_x vertex_y)
  (h_contains : p * (0 - 3) ^ 2 + 10 = 7) :
  p + q + r = 23 / 3 :=
sorry

end NUMINAMATH_GPT_parabola_p_q_r_sum_l1031_103131


namespace NUMINAMATH_GPT_max_b_no_lattice_points_line_l1031_103155

theorem max_b_no_lattice_points_line (b : ℝ) (h : ∀ (m : ℝ), 0 < m ∧ m < b → ∀ (x : ℤ), 0 < (x : ℝ) ∧ (x : ℝ) ≤ 150 → ¬∃ (y : ℤ), y = m * x + 5) :
  b ≤ 1 / 151 :=
by sorry

end NUMINAMATH_GPT_max_b_no_lattice_points_line_l1031_103155


namespace NUMINAMATH_GPT_pizza_cost_difference_l1031_103112

theorem pizza_cost_difference :
  let p := 12 -- Cost of plain pizza
  let m := 3 -- Cost of mushrooms
  let o := 4 -- Cost of olives
  let s := 12 -- Total number of slices
  (m + o + p) / s * 10 - (m + o + p) / s * 2 = 12.67 :=
by
  sorry

end NUMINAMATH_GPT_pizza_cost_difference_l1031_103112


namespace NUMINAMATH_GPT_sequence_an_l1031_103121

variable (a : ℕ → ℕ)
variable (S : ℕ → ℕ)

-- Conditions
axiom S_formula (n : ℕ) (h₁ : n > 0) : S n = 2 * a n - 2

-- Proof goal
theorem sequence_an (n : ℕ) (h₁ : n > 0) : a n = 2 ^ n := by
  sorry

end NUMINAMATH_GPT_sequence_an_l1031_103121


namespace NUMINAMATH_GPT_find_multiple_of_ron_l1031_103128

variable (R_d R_g R_n m : ℕ)

def rodney_can_lift_146 : Prop := R_d = 146
def combined_weight_239 : Prop := R_d + R_g + R_n = 239
def rodney_twice_as_roger : Prop := R_d = 2 * R_g
def roger_seven_less_than_multiple_of_ron : Prop := R_g = m * R_n - 7

theorem find_multiple_of_ron (h1 : rodney_can_lift_146 R_d) 
                             (h2 : combined_weight_239 R_d R_g R_n) 
                             (h3 : rodney_twice_as_roger R_d R_g) 
                             (h4 : roger_seven_less_than_multiple_of_ron R_g R_n m) 
                             : m = 4 :=
by 
    sorry

end NUMINAMATH_GPT_find_multiple_of_ron_l1031_103128


namespace NUMINAMATH_GPT_sum_of_acute_angles_l1031_103177

theorem sum_of_acute_angles (α β : Real) (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2)
    (h1 : Real.sin α = 2 * Real.sqrt 5 / 5)
    (h2 : Real.sin β = 3 * Real.sqrt 10 / 10) :
    α + β = 3 * Real.pi / 4 :=
sorry

end NUMINAMATH_GPT_sum_of_acute_angles_l1031_103177


namespace NUMINAMATH_GPT_remainder_when_dividing_sum_l1031_103151

theorem remainder_when_dividing_sum (k m : ℤ) (c d : ℤ) (h1 : c = 60 * k + 47) (h2 : d = 42 * m + 17) :
  (c + d) % 21 = 1 :=
by
  sorry

end NUMINAMATH_GPT_remainder_when_dividing_sum_l1031_103151


namespace NUMINAMATH_GPT_roots_of_equation_l1031_103144

theorem roots_of_equation (a b c : ℝ) (h : a ≠ b ∧ b ≠ c ∧ a ≠ c) :
  ∀ x : ℝ, a^2 * (x - b) / (a - b) * (x - c) / (a - c) + b^2 * (x - a) / (b - a) * (x - c) / (b - c) + c^2 * (x - a) / (c - a) * (x - b) / (c - b) = x^2 :=
by
  intros
  sorry

end NUMINAMATH_GPT_roots_of_equation_l1031_103144


namespace NUMINAMATH_GPT_paul_prays_more_than_bruce_l1031_103114

-- Conditions as definitions in Lean 4
def prayers_per_day_paul := 20
def prayers_per_sunday_paul := 2 * prayers_per_day_paul
def prayers_per_day_bruce := prayers_per_day_paul / 2
def prayers_per_sunday_bruce := 2 * prayers_per_sunday_paul

def weekly_prayers_paul := 6 * prayers_per_day_paul + prayers_per_sunday_paul
def weekly_prayers_bruce := 6 * prayers_per_day_bruce + prayers_per_sunday_bruce

-- Statement of the proof problem
theorem paul_prays_more_than_bruce :
  (weekly_prayers_paul - weekly_prayers_bruce) = 20 := by
  sorry

end NUMINAMATH_GPT_paul_prays_more_than_bruce_l1031_103114


namespace NUMINAMATH_GPT_sum_integers_neg40_to_60_l1031_103196

theorem sum_integers_neg40_to_60 : (Finset.range (60 + 41)).sum (fun i => i - 40) = 1010 := by
  sorry

end NUMINAMATH_GPT_sum_integers_neg40_to_60_l1031_103196


namespace NUMINAMATH_GPT_total_cost_of_shirts_is_24_l1031_103158

-- Definitions based on conditions
def cost_first_shirt : ℕ := 15
def cost_difference : ℕ := 6
def cost_second_shirt : ℕ := cost_first_shirt - cost_difference

-- The proof problem statement: Calculate total cost given the conditions
theorem total_cost_of_shirts_is_24 : cost_first_shirt + cost_second_shshirt = 24 :=
by
  sorry

end NUMINAMATH_GPT_total_cost_of_shirts_is_24_l1031_103158


namespace NUMINAMATH_GPT_twelve_percent_greater_l1031_103184

theorem twelve_percent_greater :
  ∃ x : ℝ, x = 80 + (12 / 100) * 80 := sorry

end NUMINAMATH_GPT_twelve_percent_greater_l1031_103184


namespace NUMINAMATH_GPT_competition_score_difference_l1031_103138

theorem competition_score_difference :
  let perc_60 := 0.20
  let perc_75 := 0.25
  let perc_85 := 0.15
  let perc_90 := 0.30
  let perc_95 := 0.10
  let mean := (perc_60 * 60) + (perc_75 * 75) + (perc_85 * 85) + (perc_90 * 90) + (perc_95 * 95)
  let median := 85
  (median - mean = 5) := by
sorry

end NUMINAMATH_GPT_competition_score_difference_l1031_103138


namespace NUMINAMATH_GPT_max_value_point_l1031_103122

noncomputable def f (x : ℝ) : ℝ := x + Real.cos (2 * x)

theorem max_value_point : ∃ x ∈ Set.Ioo 0 Real.pi, (∀ y ∈ Set.Ioo 0 Real.pi, f x ≥ f y) ∧ x = Real.pi / 12 :=
by sorry

end NUMINAMATH_GPT_max_value_point_l1031_103122


namespace NUMINAMATH_GPT_total_miles_driven_l1031_103101

-- Conditions
def miles_darius : ℕ := 679
def miles_julia : ℕ := 998

-- Proof statement
theorem total_miles_driven : miles_darius + miles_julia = 1677 := 
by
  -- placeholder for the proof steps
  sorry

end NUMINAMATH_GPT_total_miles_driven_l1031_103101


namespace NUMINAMATH_GPT_sara_wrapping_paper_l1031_103124

theorem sara_wrapping_paper (s : ℚ) (l : ℚ) (total : ℚ) : 
  total = 3 / 8 → 
  l = 2 * s →
  4 * s + 2 * l = total → 
  s = 3 / 64 :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_sara_wrapping_paper_l1031_103124


namespace NUMINAMATH_GPT_interval_comparison_l1031_103159

theorem interval_comparison (x : ℝ) :
  ((x - 1) * (x + 3) < 0) → ¬((x + 1) * (x - 3) < 0) ∧ ¬((x + 1) * (x - 3) < 0 → (x - 1) * (x + 3) < 0) :=
by
  sorry

end NUMINAMATH_GPT_interval_comparison_l1031_103159


namespace NUMINAMATH_GPT_cos_neg_two_pi_over_three_eq_l1031_103164

noncomputable def cos_neg_two_pi_over_three : ℝ := -2 * Real.pi / 3

theorem cos_neg_two_pi_over_three_eq :
  Real.cos cos_neg_two_pi_over_three = -1 / 2 :=
sorry

end NUMINAMATH_GPT_cos_neg_two_pi_over_three_eq_l1031_103164


namespace NUMINAMATH_GPT_num_Q_polynomials_l1031_103157

noncomputable def P (x : ℝ) : ℝ := (x - 1) * (x - 4) * (x - 5)

#check Exists

theorem num_Q_polynomials :
  ∃ (Q : Polynomial ℝ), 
  (∃ (R : Polynomial ℝ), R.degree = 3 ∧ P (Q.eval x) = P x * R.eval x) ∧
  Q.degree = 2 ∧ (Q.coeff 1 = 6) ∧ (∃ (n : ℕ), n = 22) :=
sorry

end NUMINAMATH_GPT_num_Q_polynomials_l1031_103157


namespace NUMINAMATH_GPT_smallest_prime_after_five_consecutive_nonprimes_l1031_103154

theorem smallest_prime_after_five_consecutive_nonprimes :
  ∃ p : ℕ, Nat.Prime p ∧ 
          (∀ n : ℕ, n < p → ¬ (n ≥ 24 ∧ n < 29 ∧ ¬ Nat.Prime n)) ∧
          p = 29 :=
by
  sorry

end NUMINAMATH_GPT_smallest_prime_after_five_consecutive_nonprimes_l1031_103154


namespace NUMINAMATH_GPT_roots_equation_l1031_103142

theorem roots_equation (a b c d : ℝ) (h₁ : a ≠ 0) (h₂ : a * 4^3 + b * 4^2 + c * 4 + d = 0) (h₃ : a * (-3)^3 + b * (-3)^2 + c * (-3) + d = 0) :
  (b + c) / a = -13 :=
by
  sorry

end NUMINAMATH_GPT_roots_equation_l1031_103142


namespace NUMINAMATH_GPT_cab_driver_income_day3_l1031_103104

theorem cab_driver_income_day3 :
  let income1 := 200
  let income2 := 150
  let income4 := 400
  let income5 := 500
  let avg_income := 400
  let total_income := avg_income * 5 
  total_income - (income1 + income2 + income4 + income5) = 750 := by
  sorry

end NUMINAMATH_GPT_cab_driver_income_day3_l1031_103104


namespace NUMINAMATH_GPT_mac_runs_faster_than_apple_l1031_103191

theorem mac_runs_faster_than_apple :
  let Apple_speed := 3 -- miles per hour
  let Mac_speed := 4 -- miles per hour
  let Distance := 24 -- miles
  let Apple_time := Distance / Apple_speed -- hours
  let Mac_time := Distance / Mac_speed -- hours
  let Time_difference := (Apple_time - Mac_time) * 60 -- converting hours to minutes
  Time_difference = 120 := by
  sorry

end NUMINAMATH_GPT_mac_runs_faster_than_apple_l1031_103191


namespace NUMINAMATH_GPT_max_marks_are_700_l1031_103161

/-- 
A student has to obtain 33% of the total marks to pass.
The student got 175 marks and failed by 56 marks.
Prove that the maximum marks are 700.
-/
theorem max_marks_are_700 (M : ℝ) (h1 : 0.33 * M = 175 + 56) : M = 700 :=
sorry

end NUMINAMATH_GPT_max_marks_are_700_l1031_103161


namespace NUMINAMATH_GPT_school_count_l1031_103166

theorem school_count (n : ℕ) (h1 : 2 * n - 1 = 69) (h2 : n < 76) (h3 : n > 29) : (2 * n - 1) / 3 = 23 :=
by
  sorry

end NUMINAMATH_GPT_school_count_l1031_103166


namespace NUMINAMATH_GPT_smallest_n_for_g4_l1031_103118

def g (n : ℕ) : ℕ :=
  (List.range (n + 1)).foldl (λ acc a => acc + (List.range (n + 1)).countP (λ b => a * a + b * b = n)) 0

theorem smallest_n_for_g4 : ∃ n : ℕ, g n = 4 ∧ 
  (∀ m : ℕ, m < n → g m ≠ 4) :=
by
  use 65
  -- Proof would go here
  sorry

end NUMINAMATH_GPT_smallest_n_for_g4_l1031_103118


namespace NUMINAMATH_GPT_find_t_squared_l1031_103145
noncomputable section

-- Definitions of the given conditions
def hyperbola_opens_vertically (x y : ℝ) : Prop :=
  (y^2 / 4 - 5 * x^2 / 16 = 1)

-- Statement of the problem
theorem find_t_squared (t : ℝ) 
  (h1 : hyperbola_opens_vertically 4 (-3))
  (h2 : hyperbola_opens_vertically 0 (-2))
  (h3 : hyperbola_opens_vertically 2 t) : 
  t^2 = 8 := 
sorry -- Proof is omitted, it's just the statement

end NUMINAMATH_GPT_find_t_squared_l1031_103145


namespace NUMINAMATH_GPT_area_of_isosceles_triangle_PQR_l1031_103133

noncomputable def area_of_triangle (P Q R : ℝ) (PQ PR QR PS QS SR : ℝ) : Prop :=
PQ = 17 ∧ PR = 17 ∧ QR = 16 ∧ PS = 15 ∧ QS = 8 ∧ SR = 8 →
(1 / 2) * QR * PS = 120

theorem area_of_isosceles_triangle_PQR :
  ∀ (P Q R : ℝ), 
  ∀ (PQ PR QR PS QS SR : ℝ), 
  PQ = 17 → PR = 17 → QR = 16 → PS = 15 → QS = 8 → SR = 8 →
  area_of_triangle P Q R PQ PR QR PS QS SR := 
by
  intros P Q R PQ PR QR PS QS SR hPQ hPR hQR hPS hQS hSR
  unfold area_of_triangle
  simp [hPQ, hPR, hQR, hPS, hQS, hSR]
  sorry

end NUMINAMATH_GPT_area_of_isosceles_triangle_PQR_l1031_103133


namespace NUMINAMATH_GPT_percentage_of_birth_in_june_l1031_103135

theorem percentage_of_birth_in_june (total_scientists: ℕ) (born_in_june: ℕ) (h_total: total_scientists = 150) (h_june: born_in_june = 15) : (born_in_june * 100 / total_scientists) = 10 := 
by 
  sorry

end NUMINAMATH_GPT_percentage_of_birth_in_june_l1031_103135


namespace NUMINAMATH_GPT_painter_red_cells_count_l1031_103180

open Nat

/-- Prove the number of red cells painted by the painter in the given 2000 x 70 grid. -/
theorem painter_red_cells_count :
  let rows := 2000
  let columns := 70
  let lcm_rc := Nat.lcm rows columns -- Calculate the LCM of row and column counts
  lcm_rc = 14000 := by
sorry

end NUMINAMATH_GPT_painter_red_cells_count_l1031_103180


namespace NUMINAMATH_GPT_quadratic_function_characterization_l1031_103111

variable (f : ℝ → ℝ)

def quadratic_function_satisfies_conditions (f : ℝ → ℝ) : Prop :=
  (f 0 = 2) ∧ (∀ x, f (x + 1) - f x = 2 * x - 1)

theorem quadratic_function_characterization
  (hf : quadratic_function_satisfies_conditions f) : 
  (∀ x, f x = x^2 - 2 * x + 2) ∧ 
  (f (-1) = 5) ∧ 
  (f 1 = 1) ∧ 
  (f 2 = 2) := by
sorry

end NUMINAMATH_GPT_quadratic_function_characterization_l1031_103111


namespace NUMINAMATH_GPT_arithmetic_sequence_a9_l1031_103181

theorem arithmetic_sequence_a9 (S : ℕ → ℤ) (a : ℕ → ℤ) :
  S 8 = 4 * a 3 → a 7 = -2 → a 9 = -6 := by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_a9_l1031_103181


namespace NUMINAMATH_GPT_trajectory_midpoint_l1031_103119

theorem trajectory_midpoint {x y : ℝ} (hx : 2 * y + 1 = 2 * (2 * x)^2 + 1) :
  y = 4 * x^2 := 
by sorry

end NUMINAMATH_GPT_trajectory_midpoint_l1031_103119


namespace NUMINAMATH_GPT_find_g3_l1031_103186

noncomputable def g : ℝ → ℝ := sorry

theorem find_g3 (h : ∀ x : ℝ, g (3^x) + x * g (3^(-x)) = x) : g 3 = 1 :=
sorry

end NUMINAMATH_GPT_find_g3_l1031_103186


namespace NUMINAMATH_GPT_part1_part2_l1031_103156

-- Definitions based on the conditions
def a_i (i : ℕ) : ℕ := sorry -- Define ai's values based on the given conditions
def f (n : ℕ) : ℕ := sorry  -- Define f(n) as the number of n-digit wave numbers satisfying the given conditions

-- Prove the first part: f(10) = 3704
theorem part1 : f 10 = 3704 := sorry

-- Prove the second part: f(2008) % 13 = 10
theorem part2 : (f 2008) % 13 = 10 := sorry

end NUMINAMATH_GPT_part1_part2_l1031_103156


namespace NUMINAMATH_GPT_find_n_value_l1031_103190

theorem find_n_value : 
  ∃ (n : ℕ), ∀ (a b c : ℕ), 
    a + b + c = 200 ∧ 
    (∃ bc ca ab : ℕ, bc = b * c ∧ ca = c * a ∧ ab = a * b ∧ n = bc ∧ n = ca ∧ n = ab) → 
    n = 199 := sorry

end NUMINAMATH_GPT_find_n_value_l1031_103190


namespace NUMINAMATH_GPT_liquid_left_after_evaporation_l1031_103188

-- Definitions
def solution_y (total_mass : ℝ) : ℝ × ℝ :=
  (0.30 * total_mass, 0.70 * total_mass) -- liquid_x, water

def evaporate_water (initial_water : ℝ) (evaporated_mass : ℝ) : ℝ :=
  initial_water - evaporated_mass

-- Condition that new solution is 45% liquid x
theorem liquid_left_after_evaporation 
  (initial_mass : ℝ) 
  (evaporated_mass : ℝ)
  (added_mass : ℝ)
  (new_percentage_liquid_x : ℝ) :
  initial_mass = 8 → 
  evaporated_mass = 4 → 
  added_mass = 4 →
  new_percentage_liquid_x = 0.45 →
  solution_y initial_mass = (2.4, 5.6) →
  evaporate_water 5.6 evaporated_mass = 1.6 →
  solution_y added_mass = (1.2, 2.8) →
  2.4 + 1.2 = 3.6 →
  1.6 + 2.8 = 4.4 →
  0.45 * (3.6 + 4.4) = 3.6 →
  4 = 2.4 + 1.6 := sorry

end NUMINAMATH_GPT_liquid_left_after_evaporation_l1031_103188


namespace NUMINAMATH_GPT_rate_percent_l1031_103195

noncomputable def calculate_rate (P: ℝ) : ℝ :=
  let I : ℝ := 320
  let t : ℝ := 2
  I * 100 / (P * t)

theorem rate_percent (P: ℝ) (hP: P > 0) : calculate_rate P = 4 := 
by
  sorry

end NUMINAMATH_GPT_rate_percent_l1031_103195


namespace NUMINAMATH_GPT_final_student_count_l1031_103115

def initial_students := 150
def students_joined := 30
def students_left := 15

theorem final_student_count : initial_students + students_joined - students_left = 165 := by
  sorry

end NUMINAMATH_GPT_final_student_count_l1031_103115


namespace NUMINAMATH_GPT_marbles_lost_l1031_103194

def initial_marbles := 8
def current_marbles := 6

theorem marbles_lost : initial_marbles - current_marbles = 2 :=
by
  sorry

end NUMINAMATH_GPT_marbles_lost_l1031_103194


namespace NUMINAMATH_GPT_sum_ratio_arithmetic_sequence_l1031_103176

noncomputable def sum_of_arithmetic_sequence (n : ℕ) : ℝ := sorry

theorem sum_ratio_arithmetic_sequence (S : ℕ → ℝ) (hS : ∀ n, S n = sum_of_arithmetic_sequence n)
  (h_cond : S 3 / S 6 = 1 / 3) :
  S 6 / S 12 = 3 / 10 :=
sorry

end NUMINAMATH_GPT_sum_ratio_arithmetic_sequence_l1031_103176


namespace NUMINAMATH_GPT_ivan_travel_time_l1031_103139

theorem ivan_travel_time (d V_I V_P : ℕ) (h1 : d = 3 * V_I * 40)
  (h2 : ∀ t, t = d / V_P + 10) : 
  (d / V_I = 75) :=
by
  sorry

end NUMINAMATH_GPT_ivan_travel_time_l1031_103139


namespace NUMINAMATH_GPT_smallest_possible_value_l1031_103146

theorem smallest_possible_value 
  (a : ℂ)
  (h : 8 * a^2 + 6 * a + 2 = 0) :
  ∃ z : ℂ, z = 3 * a + 1 ∧ z.re = -1 / 8 :=
by
  sorry

end NUMINAMATH_GPT_smallest_possible_value_l1031_103146


namespace NUMINAMATH_GPT_cylindrical_to_rectangular_l1031_103165

theorem cylindrical_to_rectangular (r θ z : ℝ) (h1 : r = 6) (h2 : θ = π / 3) (h3 : z = 2) :
  (r * Real.cos θ, r * Real.sin θ, z) = (3, 3 * Real.sqrt 3, 2) := 
by 
  rw [h1, h2, h3]
  sorry

end NUMINAMATH_GPT_cylindrical_to_rectangular_l1031_103165


namespace NUMINAMATH_GPT_problem_inequality_l1031_103150

theorem problem_inequality (x y z : ℝ) (hx : 1 ≤ x) (hy : 1 ≤ y) (hz : 1 ≤ z) :
  (x^2 - 2*x + 2) * (y^2 - 2*y + 2) * (z^2 - 2*z + 2) ≤ (x*y*z)^2 - 2*(x*y*z) + 2 := sorry

end NUMINAMATH_GPT_problem_inequality_l1031_103150


namespace NUMINAMATH_GPT_percentage_paid_to_X_l1031_103141

theorem percentage_paid_to_X (X Y : ℝ) (h1 : X + Y = 880) (h2 : Y = 400) : 
  ((X / Y) * 100) = 120 :=
by
  sorry

end NUMINAMATH_GPT_percentage_paid_to_X_l1031_103141


namespace NUMINAMATH_GPT_circle_radius_zero_l1031_103182

-- Define the given circle equation
def circle_eq (x y : ℝ) : Prop := x^2 - 4 * x + y^2 - 6 * y + 13 = 0

-- The proof problem statement
theorem circle_radius_zero : ∀ (x y : ℝ), circle_eq x y → 0 = 0 :=
by
  sorry

end NUMINAMATH_GPT_circle_radius_zero_l1031_103182


namespace NUMINAMATH_GPT_factorize_expression_l1031_103172

theorem factorize_expression (a : ℝ) : a^2 + 5 * a = a * (a + 5) :=
sorry

end NUMINAMATH_GPT_factorize_expression_l1031_103172


namespace NUMINAMATH_GPT_eve_distance_ran_more_l1031_103108

variable (ran walked : ℝ)

def eve_distance_difference (ran walked : ℝ) : ℝ :=
  ran - walked

theorem eve_distance_ran_more :
  eve_distance_difference 0.7 0.6 = 0.1 :=
by
  sorry

end NUMINAMATH_GPT_eve_distance_ran_more_l1031_103108


namespace NUMINAMATH_GPT_extreme_values_l1031_103162

noncomputable def f (x : ℝ) := (1/3) * x^3 - 4 * x + 6

theorem extreme_values :
  (∃ x : ℝ, f x = 34/3 ∧ (x = -2 ∨ x = 4)) ∧
  (∃ x : ℝ, f x = 2/3 ∧ x = 2) ∧
  (∀ x ∈ Set.Icc (-3 : ℝ) 4, f x ≤ 34/3 ∧ 2/3 ≤ f x) :=
by
  sorry

end NUMINAMATH_GPT_extreme_values_l1031_103162


namespace NUMINAMATH_GPT_recurring_decimals_sum_correct_l1031_103103

noncomputable def recurring_decimals_sum : ℚ :=
  let x := (2:ℚ) / 3
  let y := (4:ℚ) / 9
  x + y

theorem recurring_decimals_sum_correct :
  recurring_decimals_sum = 10 / 9 := 
  sorry

end NUMINAMATH_GPT_recurring_decimals_sum_correct_l1031_103103


namespace NUMINAMATH_GPT_protective_additive_increase_l1031_103179

def percentIncrease (old_val new_val : ℕ) : ℚ :=
  (new_val - old_val) / old_val * 100

theorem protective_additive_increase :
  percentIncrease 45 60 = 33.33 := 
sorry

end NUMINAMATH_GPT_protective_additive_increase_l1031_103179


namespace NUMINAMATH_GPT_find_smallest_subtract_l1031_103143

-- Definitions for multiples
def is_mul_2 (n : ℕ) : Prop := 2 ∣ n
def is_mul_3 (n : ℕ) : Prop := 3 ∣ n
def is_mul_5 (n : ℕ) : Prop := 5 ∣ n

-- Statement of the problem
theorem find_smallest_subtract (x : ℕ) :
  (is_mul_2 (134 - x)) ∧ (is_mul_3 (134 - x)) ∧ (is_mul_5 (134 - x)) → x = 14 :=
by
  sorry

end NUMINAMATH_GPT_find_smallest_subtract_l1031_103143


namespace NUMINAMATH_GPT_survey_blue_percentage_l1031_103197

-- Conditions
def red (r : ℕ) := r = 70
def blue (b : ℕ) := b = 80
def green (g : ℕ) := g = 50
def yellow (y : ℕ) := y = 70
def orange (o : ℕ) := o = 30

-- Total responses sum
def total_responses (r b g y o : ℕ) := r + b + g + y + o = 300

-- Percentage of blue respondents
def blue_percentage (b total : ℕ) := (b : ℚ) / total * 100 = 26 + 2/3

-- Theorem statement
theorem survey_blue_percentage (r b g y o : ℕ) (H_red : red r) (H_blue : blue b) (H_green : green g) (H_yellow : yellow y) (H_orange : orange o) (H_total : total_responses r b g y o) : blue_percentage b 300 :=
by {
  sorry
}

end NUMINAMATH_GPT_survey_blue_percentage_l1031_103197


namespace NUMINAMATH_GPT_triangle_expression_value_l1031_103123

theorem triangle_expression_value :
  ∀ (A B C : ℝ) (a b c : ℝ),
  A = 60 ∧ b = 1 ∧ (1 / 2) * b * c * (Real.sin A) = Real.sqrt 3 →
  (a + 2 * b - 3 * c) / (Real.sin A + 2 * Real.sin B - 3 * Real.sin C) = 2 * (Real.sqrt 39) / 3 :=
by
  intro A B C a b c
  rintro ⟨hA, hb, h_area⟩
  sorry

end NUMINAMATH_GPT_triangle_expression_value_l1031_103123


namespace NUMINAMATH_GPT_problem_condition_neither_sufficient_nor_necessary_l1031_103129

theorem problem_condition_neither_sufficient_nor_necessary 
  (m n : ℕ) (hm : m > 0) (hn : n > 0) (a b : ℝ) :
  (a > b → a^(m + n) + b^(m + n) > a^n * b^m + a^m * b^n) ∧
  (a^(m + n) + b^(m + n) > a^n * b^m + a^m * b^n → a > b) = false :=
by sorry

end NUMINAMATH_GPT_problem_condition_neither_sufficient_nor_necessary_l1031_103129


namespace NUMINAMATH_GPT_sum_of_real_solutions_l1031_103113

theorem sum_of_real_solutions (x : ℝ) (h : (x^2 + 2*x + 3)^( (x^2 + 2*x + 3)^( (x^2 + 2*x + 3) )) = 2012) : 
  ∃ (x1 x2 : ℝ), (x1 + x2 = -2) ∧ (x1^2 + 2*x1 + 3 = x2^2 + 2*x2 + 3 ∧ x2^2 + 2*x2 + 3 = x^2 + 2*x + 3) := 
by
  sorry

end NUMINAMATH_GPT_sum_of_real_solutions_l1031_103113


namespace NUMINAMATH_GPT_frank_has_3_cookies_l1031_103189

-- The definitions and conditions based on the problem statement
def num_cookies_millie : ℕ := 4
def num_cookies_mike : ℕ := 3 * num_cookies_millie
def num_cookies_frank : ℕ := (num_cookies_mike / 2) - 3

-- The theorem stating the question and the correct answer
theorem frank_has_3_cookies : num_cookies_frank = 3 :=
by 
  -- This is where the proof steps would go, but for now we use sorry
  sorry

end NUMINAMATH_GPT_frank_has_3_cookies_l1031_103189


namespace NUMINAMATH_GPT_f_2019_is_zero_l1031_103199

noncomputable def f : ℝ → ℝ := sorry

axiom f_is_non_negative
  (x : ℝ) : 0 ≤ f x

axiom f_satisfies_condition
  (a b c : ℝ) : f (a^3) + f (b^3) + f (c^3) = 3 * f a * f b * f c

axiom f_one_not_one : f 1 ≠ 1

theorem f_2019_is_zero : f 2019 = 0 := 
  sorry

end NUMINAMATH_GPT_f_2019_is_zero_l1031_103199


namespace NUMINAMATH_GPT_crowdfunding_highest_level_backing_l1031_103109

-- Definitions according to the conditions
def lowest_level_backing : ℕ := 50
def second_level_backing : ℕ := 10 * lowest_level_backing
def highest_level_backing : ℕ := 100 * lowest_level_backing
def total_raised : ℕ := (2 * highest_level_backing) + (3 * second_level_backing) + (10 * lowest_level_backing)

-- Statement of the problem
theorem crowdfunding_highest_level_backing (h: total_raised = 12000) :
  highest_level_backing = 5000 :=
sorry

end NUMINAMATH_GPT_crowdfunding_highest_level_backing_l1031_103109


namespace NUMINAMATH_GPT_tiles_difference_l1031_103187

-- Definitions based on given conditions
def initial_blue_tiles : Nat := 20
def initial_green_tiles : Nat := 10
def first_border_green_tiles : Nat := 24
def second_border_green_tiles : Nat := 36

-- Problem statement
theorem tiles_difference :
  initial_green_tiles + first_border_green_tiles + second_border_green_tiles - initial_blue_tiles = 50 :=
by
  sorry

end NUMINAMATH_GPT_tiles_difference_l1031_103187


namespace NUMINAMATH_GPT_order_of_abc_l1031_103116

noncomputable def a : ℝ := Real.log 3 / Real.log 4
noncomputable def b : ℝ := Real.log 4 / Real.log 3
noncomputable def c : ℝ := Real.log (4/3) / Real.log (3/4)

theorem order_of_abc : b > a ∧ a > c := by
  sorry

end NUMINAMATH_GPT_order_of_abc_l1031_103116
