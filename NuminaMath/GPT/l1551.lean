import Mathlib

namespace NUMINAMATH_GPT_work_completed_in_8_days_l1551_155103

theorem work_completed_in_8_days 
  (A_complete : ℕ → Prop)
  (B_complete : ℕ → Prop)
  (C_complete : ℕ → Prop)
  (A_can_complete_in_10_days : A_complete 10)
  (B_can_complete_in_20_days : B_complete 20)
  (C_can_complete_in_30_days : C_complete 30)
  (A_leaves_5_days_before_completion : ∀ x : ℕ, x ≥ 5 → A_complete (x - 5))
  (C_leaves_3_days_before_completion : ∀ x : ℕ, x ≥ 3 → C_complete (x - 3)) :
  ∃ x : ℕ, x = 8 := sorry

end NUMINAMATH_GPT_work_completed_in_8_days_l1551_155103


namespace NUMINAMATH_GPT_avg_abc_43_l1551_155198

variables (A B C : ℝ)

def avg_ab (A B : ℝ) : Prop := (A + B) / 2 = 40
def avg_bc (B C : ℝ) : Prop := (B + C) / 2 = 43
def weight_b (B : ℝ) : Prop := B = 37

theorem avg_abc_43 (A B C : ℝ) (h1 : avg_ab A B) (h2 : avg_bc B C) (h3 : weight_b B) :
  (A + B + C) / 3 = 43 :=
by
  sorry

end NUMINAMATH_GPT_avg_abc_43_l1551_155198


namespace NUMINAMATH_GPT_downstream_speed_l1551_155118

def Vm : ℝ := 31  -- speed in still water
def Vu : ℝ := 25  -- speed upstream
def Vs := Vm - Vu  -- speed of stream

theorem downstream_speed : Vm + Vs = 37 := 
by
  sorry

end NUMINAMATH_GPT_downstream_speed_l1551_155118


namespace NUMINAMATH_GPT_area_of_triangle_DEF_l1551_155155

theorem area_of_triangle_DEF :
  ∃ (DEF : Type) (area_u1 area_u2 area_u3 area_triangle : ℝ),
  area_u1 = 25 ∧
  area_u2 = 16 ∧
  area_u3 = 64 ∧
  area_triangle = area_u1 + area_u2 + area_u3 ∧
  area_triangle = 289 :=
by
  sorry

end NUMINAMATH_GPT_area_of_triangle_DEF_l1551_155155


namespace NUMINAMATH_GPT_unique_solution_m_l1551_155131

theorem unique_solution_m (m : ℝ) : (∃ x : ℝ, 3 * x^2 - 6 * x + m = 0 ∧ (∀ y₁ y₂ : ℝ, 3 * y₁^2 - 6 * y₂ + m = 0 → y₁ = y₂)) → m = 3 :=
by
  sorry

end NUMINAMATH_GPT_unique_solution_m_l1551_155131


namespace NUMINAMATH_GPT_smallest_k_for_sixty_four_gt_four_nineteen_l1551_155177

-- Definitions of the conditions
def sixty_four (k : ℕ) : ℕ := 64^k
def four_nineteen : ℕ := 4^19

-- The theorem to prove
theorem smallest_k_for_sixty_four_gt_four_nineteen (k : ℕ) : sixty_four k > four_nineteen ↔ k ≥ 7 := 
by
  sorry

end NUMINAMATH_GPT_smallest_k_for_sixty_four_gt_four_nineteen_l1551_155177


namespace NUMINAMATH_GPT_smallest_digit_divisible_by_11_l1551_155144

theorem smallest_digit_divisible_by_11 :
  ∃ (d : ℕ), d < 10 ∧ ∀ n : ℕ, (n + 45000 + 1000 + 457 + d) % 11 = 0 → d = 5 :=
by {
  sorry
}

end NUMINAMATH_GPT_smallest_digit_divisible_by_11_l1551_155144


namespace NUMINAMATH_GPT_percent_defective_units_l1551_155128

variable (D : ℝ) -- Let D represent the percent of units produced that are defective

theorem percent_defective_units
  (h1 : 0.05 * D = 0.4) : 
  D = 8 :=
by sorry

end NUMINAMATH_GPT_percent_defective_units_l1551_155128


namespace NUMINAMATH_GPT_compound_weight_l1551_155122

noncomputable def weightB : ℝ := 275
noncomputable def ratioAtoB : ℝ := 2 / 10

theorem compound_weight (weightA weightB total_weight : ℝ) 
  (h1 : ratioAtoB = 2 / 10) 
  (h2 : weightB = 275) 
  (h3 : weightA = weightB * (2 / 10)) 
  (h4 : total_weight = weightA + weightB) : 
  total_weight = 330 := 
by sorry

end NUMINAMATH_GPT_compound_weight_l1551_155122


namespace NUMINAMATH_GPT_trader_loss_percent_l1551_155159

noncomputable def CP1 : ℝ := 325475 / 1.13
noncomputable def CP2 : ℝ := 325475 / 0.87
noncomputable def TCP : ℝ := CP1 + CP2
noncomputable def TSP : ℝ := 325475 * 2
noncomputable def profit_or_loss : ℝ := TSP - TCP
noncomputable def profit_or_loss_percent : ℝ := (profit_or_loss / TCP) * 100

theorem trader_loss_percent : profit_or_loss_percent = -1.684 := by 
  sorry

end NUMINAMATH_GPT_trader_loss_percent_l1551_155159


namespace NUMINAMATH_GPT_highest_score_l1551_155162

variable (avg runs_excluding: ℕ)
variable (innings remaining_innings total_runs total_runs_excluding H L: ℕ)

axiom batting_average (h_avg: avg = 60) (h_innings: innings = 46) : total_runs = avg * innings
axiom diff_highest_lowest_score (h_diff: H - L = 190) : true
axiom avg_excluding_high_low (h_avg_excluding: runs_excluding = 58) (h_remaining_innings: remaining_innings = 44) : total_runs_excluding = runs_excluding * remaining_innings
axiom sum_high_low : total_runs - total_runs_excluding = 208

theorem highest_score (h_avg: avg = 60) (h_innings: innings = 46) (h_diff: H - L = 190) (h_avg_excluding: runs_excluding = 58) (h_remaining_innings: remaining_innings = 44)
    (calc_total_runs: total_runs = avg * innings) 
    (calc_total_runs_excluding: total_runs_excluding = runs_excluding * remaining_innings)
    (calc_sum_high_low: total_runs - total_runs_excluding = 208) : H = 199 :=
by
  sorry

end NUMINAMATH_GPT_highest_score_l1551_155162


namespace NUMINAMATH_GPT_triangle_perimeter_correct_l1551_155139

noncomputable def triangle_perimeter (a b x : ℕ) : ℕ := a + b + x

theorem triangle_perimeter_correct :
  ∀ (x : ℕ), (2 + 4 + x = 10) → 2 < x → x < 6 → (∀ k : ℕ, k = x → k % 2 = 0) → triangle_perimeter 2 4 x = 10 :=
by
  intros x h1 h2 h3
  rw [triangle_perimeter, h1]
  sorry

end NUMINAMATH_GPT_triangle_perimeter_correct_l1551_155139


namespace NUMINAMATH_GPT_ancient_chinese_poem_l1551_155182

theorem ancient_chinese_poem (x : ℕ) :
  (7 * x + 7 = 9 * (x - 1)) :=
sorry

end NUMINAMATH_GPT_ancient_chinese_poem_l1551_155182


namespace NUMINAMATH_GPT_perpendicular_case_parallel_case_l1551_155175

noncomputable def vector_a : ℝ × ℝ := (1, 2)
noncomputable def vector_b : ℝ × ℝ := (-3, 2)
noncomputable def k_perpendicular : ℝ := 19
noncomputable def k_parallel : ℝ := -1/3

-- Define the operations used:
def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

def scalar_mult (k : ℝ) (v : ℝ × ℝ) : ℝ × ℝ :=
  (k * v.1, k * v.2)

def vector_add (v1 v2 : ℝ × ℝ) : ℝ × ℝ :=
  (v1.1 + v2.1, v1.2 + v2.2)

def vector_sub (v1 v2 : ℝ × ℝ) : ℝ × ℝ :=
  (v1.1 - v2.1, v1.2 - v2.2)

-- Perpendicular case: 
theorem perpendicular_case : dot_product (vector_add (scalar_mult k_perpendicular vector_a) vector_b) (vector_sub vector_a (scalar_mult 3 vector_b)) = 0 := sorry

-- Parallel case:
theorem parallel_case : ∃ c : ℝ, vector_add (scalar_mult k_parallel vector_a) vector_b = scalar_mult c (vector_sub vector_a (scalar_mult 3 vector_b)) ∧ c < 0 := sorry

end NUMINAMATH_GPT_perpendicular_case_parallel_case_l1551_155175


namespace NUMINAMATH_GPT_gcd_factorial_7_8_l1551_155126

theorem gcd_factorial_7_8 : Nat.gcd (Nat.factorial 7) (Nat.factorial 8) = 5040 := 
by
  sorry

end NUMINAMATH_GPT_gcd_factorial_7_8_l1551_155126


namespace NUMINAMATH_GPT_find_d_value_l1551_155183

theorem find_d_value 
  (x y d : ℝ)
  (h1 : 7^(3 * x - 1) * 3^(4 * y - 3) = 49^x * d^y)
  (h2 : x + y = 4) :
  d = 27 :=
by 
  sorry

end NUMINAMATH_GPT_find_d_value_l1551_155183


namespace NUMINAMATH_GPT_cucumber_weight_evaporation_l1551_155115

theorem cucumber_weight_evaporation :
  let w_99 := 50
  let p_99 := 0.99
  let evap_99 := 0.01
  let w_98 := 30
  let p_98 := 0.98
  let evap_98 := 0.02
  let w_97 := 20
  let p_97 := 0.97
  let evap_97 := 0.03

  let initial_water_99 := p_99 * w_99
  let dry_matter_99 := w_99 - initial_water_99
  let evaporated_water_99 := evap_99 * initial_water_99
  let new_weight_99 := (initial_water_99 - evaporated_water_99) + dry_matter_99

  let initial_water_98 := p_98 * w_98
  let dry_matter_98 := w_98 - initial_water_98
  let evaporated_water_98 := evap_98 * initial_water_98
  let new_weight_98 := (initial_water_98 - evaporated_water_98) + dry_matter_98

  let initial_water_97 := p_97 * w_97
  let dry_matter_97 := w_97 - initial_water_97
  let evaporated_water_97 := evap_97 * initial_water_97
  let new_weight_97 := (initial_water_97 - evaporated_water_97) + dry_matter_97

  let total_new_weight := new_weight_99 + new_weight_98 + new_weight_97
  total_new_weight = 98.335 :=
 by
  sorry

end NUMINAMATH_GPT_cucumber_weight_evaporation_l1551_155115


namespace NUMINAMATH_GPT_number_sequence_53rd_l1551_155130

theorem number_sequence_53rd (n : ℕ) (h₁ : n = 53) : n = 53 :=
by {
  sorry
}

end NUMINAMATH_GPT_number_sequence_53rd_l1551_155130


namespace NUMINAMATH_GPT_profit_percentage_of_revenues_l1551_155184

theorem profit_percentage_of_revenues (R P : ℝ)
  (H1 : R > 0)
  (H2 : P > 0)
  (H3 : P * 0.98 = R * 0.098) :
  (P / R) * 100 = 10 := by
  sorry

end NUMINAMATH_GPT_profit_percentage_of_revenues_l1551_155184


namespace NUMINAMATH_GPT_variance_ξ_l1551_155190

variable (P : ℕ → ℝ) (ξ : ℕ)

-- conditions
axiom P_0 : P 0 = 1 / 5
axiom P_1 : P 1 + P 2 = 4 / 5
axiom E_ξ : (0 * P 0 + 1 * P 1 + 2 * P 2) = 1

-- proof statement
theorem variance_ξ : (0 - 1)^2 * P 0 + (1 - 1)^2 * P 1 + (2 - 1)^2 * P 2 = 2 / 5 :=
by sorry

end NUMINAMATH_GPT_variance_ξ_l1551_155190


namespace NUMINAMATH_GPT_julia_average_speed_l1551_155181

-- Define the conditions as constants
def total_distance : ℝ := 28
def total_time : ℝ := 4

-- Define the theorem stating Julia's average speed
theorem julia_average_speed : total_distance / total_time = 7 := by
  sorry

end NUMINAMATH_GPT_julia_average_speed_l1551_155181


namespace NUMINAMATH_GPT_find_other_number_l1551_155165

theorem find_other_number 
  (A B : ℕ) 
  (h1 : A = 385) 
  (h2 : Nat.lcm A B = 2310) 
  (h3 : Nat.gcd A B = 30) : 
  B = 180 := 
by
  sorry

end NUMINAMATH_GPT_find_other_number_l1551_155165


namespace NUMINAMATH_GPT_central_angle_probability_l1551_155151

theorem central_angle_probability (A : ℝ) (x : ℝ)
  (h1 : A > 0)
  (h2 : (x / 360) * A / A = 1 / 8) : 
  x = 45 := 
by
  sorry

end NUMINAMATH_GPT_central_angle_probability_l1551_155151


namespace NUMINAMATH_GPT_find_replaced_man_weight_l1551_155133

variable (n : ℕ) (new_weight old_avg_weight : ℝ) (weight_inc : ℝ) (W : ℝ)

theorem find_replaced_man_weight 
  (h1 : n = 8) 
  (h2 : new_weight = 68) 
  (h3 : weight_inc = 1) 
  (h4 : 8 * (old_avg_weight + 1) = 8 * old_avg_weight + (new_weight - W)) 
  : W = 60 :=
by
  sorry

end NUMINAMATH_GPT_find_replaced_man_weight_l1551_155133


namespace NUMINAMATH_GPT_find_k_value_l1551_155106

theorem find_k_value (x k : ℝ) (h : x = -3) (h_eq : k * (x - 2) - 4 = k - 2 * x) : k = -5/3 := by
  sorry

end NUMINAMATH_GPT_find_k_value_l1551_155106


namespace NUMINAMATH_GPT_eval_expression_l1551_155129

theorem eval_expression : 3 ^ 2 - (4 * 2) = 1 :=
by
  sorry

end NUMINAMATH_GPT_eval_expression_l1551_155129


namespace NUMINAMATH_GPT_son_is_four_times_younger_l1551_155109

-- Given Conditions
def son_age : ℕ := 9
def dad_age : ℕ := 36
def age_difference : ℕ := dad_age - son_age -- Ensure the difference in ages

-- The proof problem
theorem son_is_four_times_younger : dad_age / son_age = 4 :=
by
  -- Ensure the conditions are correct and consistent.
  have h1 : dad_age = 36 := rfl
  have h2 : son_age = 9 := rfl
  have h3 : dad_age - son_age = 27 := rfl
  sorry

end NUMINAMATH_GPT_son_is_four_times_younger_l1551_155109


namespace NUMINAMATH_GPT_cuberoot_eq_l1551_155145

open Real

theorem cuberoot_eq (x : ℝ) (h: (5:ℝ) * x + 4 = (5:ℝ) ^ 3 / (2:ℝ) ^ 3) : x = 93 / 40 := by
  sorry

end NUMINAMATH_GPT_cuberoot_eq_l1551_155145


namespace NUMINAMATH_GPT_max_whole_number_n_l1551_155160

theorem max_whole_number_n (n : ℕ) : (1/2 + n/9 < 1) → n ≤ 4 :=
by
  sorry

end NUMINAMATH_GPT_max_whole_number_n_l1551_155160


namespace NUMINAMATH_GPT_horner_method_v3_correct_l1551_155193

-- Define the polynomial function according to Horner's method
def horner (x : ℝ) : ℝ :=
  (((((3 * x - 2) * x + 2) * x - 4) * x) * x - 7)

-- Given the value of x
def x_val : ℝ := 2

-- Define v_3 based on the polynomial evaluated at x = 2 using Horner's method
def v3 : ℝ := horner x_val

-- Theorem stating what we need to prove
theorem horner_method_v3_correct : v3 = 16 :=
  by
    sorry

end NUMINAMATH_GPT_horner_method_v3_correct_l1551_155193


namespace NUMINAMATH_GPT_arnel_number_of_boxes_l1551_155191

def arnel_kept_pencils : ℕ := 10
def number_of_friends : ℕ := 5
def pencils_per_friend : ℕ := 8
def pencils_per_box : ℕ := 5

theorem arnel_number_of_boxes : ∃ (num_boxes : ℕ), 
  (number_of_friends * pencils_per_friend) + arnel_kept_pencils = num_boxes * pencils_per_box ∧ 
  num_boxes = 10 := sorry

end NUMINAMATH_GPT_arnel_number_of_boxes_l1551_155191


namespace NUMINAMATH_GPT_money_leftover_is_90_l1551_155138

-- Define constants and given conditions.
def jars_quarters : ℕ := 4
def quarters_per_jar : ℕ := 160
def jars_dimes : ℕ := 4
def dimes_per_jar : ℕ := 300
def jars_nickels : ℕ := 2
def nickels_per_jar : ℕ := 500

def value_per_quarter : ℝ := 0.25
def value_per_dime : ℝ := 0.10
def value_per_nickel : ℝ := 0.05

def bike_cost : ℝ := 240
def total_quarters := jars_quarters * quarters_per_jar
def total_dimes := jars_dimes * dimes_per_jar
def total_nickels := jars_nickels * nickels_per_jar

-- Calculate the total money Jenn has in quarters, dimes, and nickels.
def total_value_quarters : ℝ := total_quarters * value_per_quarter
def total_value_dimes : ℝ := total_dimes * value_per_dime
def total_value_nickels : ℝ := total_nickels * value_per_nickel

def total_money : ℝ := total_value_quarters + total_value_dimes + total_value_nickels

-- Calculate the money left after buying the bike.
def money_left : ℝ := total_money - bike_cost

-- Prove that the amount of money left is precisely $90.
theorem money_leftover_is_90 : money_left = 90 :=
by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_money_leftover_is_90_l1551_155138


namespace NUMINAMATH_GPT_angle_between_lines_l1551_155152

theorem angle_between_lines :
  let L1 := {p : ℝ × ℝ | p.1 = -3}  -- Line x+3=0
  let L2 := {p: ℝ × ℝ | p.1 + p.2 - 3 = 0}  -- Line x+y-3=0
  ∃ θ : ℝ, 0 < θ ∧ θ < 180 ∧ θ = 45 :=
sorry

end NUMINAMATH_GPT_angle_between_lines_l1551_155152


namespace NUMINAMATH_GPT_intersection_A_B_l1551_155174

-- Define the sets A and B
def A : Set ℤ := {0, 1, 2}
def B : Set ℤ := {x : ℤ | x^2 < 3}

-- Prove that A ∩ B = {0, 1}
theorem intersection_A_B :
  A ∩ B = {0, 1} :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_intersection_A_B_l1551_155174


namespace NUMINAMATH_GPT_bus_fare_max_profit_passenger_count_change_l1551_155134

noncomputable def demand (p : ℝ) : ℝ := 3000 - 20 * p
noncomputable def train_fare : ℝ := 10
noncomputable def train_capacity : ℝ := 1000
noncomputable def bus_cost (y : ℝ) : ℝ := y + 5

theorem bus_fare_max_profit : 
  ∃ (p_bus : ℝ), 
  p_bus = 50.5 ∧ 
  p_bus * (demand p_bus - train_capacity) - bus_cost (demand p_bus - train_capacity) = 
  p_bus * (demand p_bus - train_capacity) - (demand p_bus - train_capacity + 5) := 
sorry

theorem passenger_count_change :
  (demand train_fare - train_capacity) + train_capacity - demand 75.5 = 500 :=
sorry

end NUMINAMATH_GPT_bus_fare_max_profit_passenger_count_change_l1551_155134


namespace NUMINAMATH_GPT_rich_total_distance_l1551_155100

-- Define the given conditions 
def distance_house_to_sidewalk := 20
def distance_down_road := 200
def total_distance_so_far := distance_house_to_sidewalk + distance_down_road
def distance_left_turn := 2 * total_distance_so_far
def distance_to_intersection := total_distance_so_far + distance_left_turn
def distance_half := distance_to_intersection / 2
def total_distance_one_way := distance_to_intersection + distance_half

-- Define the theorem to be proven 
theorem rich_total_distance : total_distance_one_way * 2 = 1980 :=
by 
  -- This line is to complete the 'prove' demand of the theorem
  sorry

end NUMINAMATH_GPT_rich_total_distance_l1551_155100


namespace NUMINAMATH_GPT_perimeter_is_140_l1551_155197

-- Definitions for conditions
def width (w : ℝ) := w
def length (w : ℝ) := width w + 10
def perimeter (w : ℝ) := 2 * (length w + width w)

-- Cost condition
def cost_condition (w : ℝ) : Prop := (perimeter w) * 6.5 = 910

-- Proving that if cost_condition holds, the perimeter is 140
theorem perimeter_is_140 (w : ℝ) (h : cost_condition w) : perimeter w = 140 :=
by sorry

end NUMINAMATH_GPT_perimeter_is_140_l1551_155197


namespace NUMINAMATH_GPT_find_unknown_number_l1551_155187

-- Defining the conditions of the problem
def equation (x : ℝ) : Prop := (45 + x / 89) * 89 = 4028

-- Stating the theorem to be proved
theorem find_unknown_number : equation 23 :=
by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_find_unknown_number_l1551_155187


namespace NUMINAMATH_GPT_fraction_irreducible_l1551_155124

theorem fraction_irreducible (n : ℤ) : Int.gcd (39 * n + 4) (26 * n + 3) = 1 :=
sorry

end NUMINAMATH_GPT_fraction_irreducible_l1551_155124


namespace NUMINAMATH_GPT_diff_hours_l1551_155153

def hours_English : ℕ := 7
def hours_Spanish : ℕ := 4

theorem diff_hours : hours_English - hours_Spanish = 3 :=
by
  sorry

end NUMINAMATH_GPT_diff_hours_l1551_155153


namespace NUMINAMATH_GPT_museum_admission_ratio_l1551_155158

theorem museum_admission_ratio (a c : ℕ) (h1 : 30 * a + 15 * c = 2700) (h2 : 2 ≤ a) (h3 : 2 ≤ c) :
  a / (180 - 2 * a) = 2 :=
by
  sorry

end NUMINAMATH_GPT_museum_admission_ratio_l1551_155158


namespace NUMINAMATH_GPT_rectangle_area_change_l1551_155196

theorem rectangle_area_change (x : ℝ) :
  let L := 1 -- arbitrary non-zero value for length
  let W := 1 -- arbitrary non-zero value for width
  (1 + x / 100) * (1 - x / 100) = 1.01 -> x = 10 := 
by
  sorry

end NUMINAMATH_GPT_rectangle_area_change_l1551_155196


namespace NUMINAMATH_GPT_second_neighbor_brought_less_l1551_155167

theorem second_neighbor_brought_less (n1 n2 : ℕ) (htotal : ℕ) (h1 : n1 = 75) (h_total : n1 + n2 = 125) :
  n1 - n2 = 25 :=
by
  sorry

end NUMINAMATH_GPT_second_neighbor_brought_less_l1551_155167


namespace NUMINAMATH_GPT_chess_tournament_l1551_155180

theorem chess_tournament (n : ℕ) (h1 : 10 * 9 * n / 2 = 90) : n = 2 :=
by
  sorry

end NUMINAMATH_GPT_chess_tournament_l1551_155180


namespace NUMINAMATH_GPT_ernie_can_make_circles_l1551_155173

-- Make a statement of the problem in Lean 4
theorem ernie_can_make_circles (total_boxes : ℕ) (ali_boxes_per_circle : ℕ) (ernie_boxes_per_circle : ℕ) (ali_circles : ℕ) 
  (h1 : total_boxes = 80) (h2 : ali_boxes_per_circle = 8) (h3 : ernie_boxes_per_circle = 10) (h4 : ali_circles = 5) :
  (total_boxes - ali_boxes_per_circle * ali_circles) / ernie_boxes_per_circle = 4 := 
by 
  -- Proof of the theorem
  sorry

end NUMINAMATH_GPT_ernie_can_make_circles_l1551_155173


namespace NUMINAMATH_GPT_bus_trip_children_difference_l1551_155136

theorem bus_trip_children_difference :
  let initial := 41
  let final :=
    initial
    - 12 + 5   -- First bus stop
    - 7 + 10   -- Second bus stop
    - 14 + 3   -- Third bus stop
    - 9 + 6    -- Fourth bus stop
  initial - final = 18 :=
by sorry

end NUMINAMATH_GPT_bus_trip_children_difference_l1551_155136


namespace NUMINAMATH_GPT_range_of_a_l1551_155141

/-- Proposition p: ∀ x ∈ [1,2], x² - a ≥ 0 -/
def prop_p (a : ℝ) : Prop := 
  ∀ x : ℝ, (1 ≤ x ∧ x ≤ 2) → x^2 - a ≥ 0

/-- Proposition q: ∃ x₀ ∈ ℝ, x + 2ax₀ + 2 - a = 0 -/
def prop_q (a : ℝ) : Prop := 
  ∃ x₀ : ℝ, ∃ x : ℝ, x + 2 * a * x₀ + 2 - a = 0

theorem range_of_a (a : ℝ) (h : prop_p a ∧ prop_q a) : a ≤ -2 ∨ a = 1 :=
sorry

end NUMINAMATH_GPT_range_of_a_l1551_155141


namespace NUMINAMATH_GPT_possible_values_l1551_155176

noncomputable def matrixN (x y z : ℂ) : Matrix (Fin 3) (Fin 3) ℂ :=
  ![![x, y, z], ![z, x, y], ![y, z, x]]

theorem possible_values (x y z : ℂ) (h1 : (matrixN x y z)^3 = 1)
  (h2 : x * y * z = 1) : x^3 + y^3 + z^3 = 4 ∨ x^3 + y^3 + z^3 = -2 :=
  sorry

end NUMINAMATH_GPT_possible_values_l1551_155176


namespace NUMINAMATH_GPT_symmetric_line_eq_l1551_155140

theorem symmetric_line_eq (x y : ℝ) (h : 3 * x + 4 * y + 5 = 0) : 3 * x - 4 * y + 5 = 0 :=
sorry

end NUMINAMATH_GPT_symmetric_line_eq_l1551_155140


namespace NUMINAMATH_GPT_cuboid_volume_l1551_155137

theorem cuboid_volume (a b c : ℝ) (ha : a = 4) (hb : b = 5) (hc : c = 6) : a * b * c = 120 :=
by
  sorry

end NUMINAMATH_GPT_cuboid_volume_l1551_155137


namespace NUMINAMATH_GPT_most_suitable_sampling_method_l1551_155107

/-- A unit has 28 elderly people, 54 middle-aged people, and 81 young people. 
    A sample of 36 people needs to be drawn in a way that accounts for age.
    The most suitable method for drawing a sample is to exclude one elderly person first,
    then use stratified sampling. -/
theorem most_suitable_sampling_method 
  (elderly : ℕ) (middle_aged : ℕ) (young : ℕ) (sample_size : ℕ) (suitable_method : String)
  (condition1 : elderly = 28) 
  (condition2 : middle_aged = 54) 
  (condition3 : young = 81) 
  (condition4 : sample_size = 36) 
  (condition5 : suitable_method = "Exclude one elderly person first, then stratify sampling") : 
  suitable_method = "Exclude one elderly person first, then stratify sampling" := 
by sorry

end NUMINAMATH_GPT_most_suitable_sampling_method_l1551_155107


namespace NUMINAMATH_GPT_positive_real_triangle_inequality_l1551_155149

theorem positive_real_triangle_inequality
    (a b c : ℝ)
    (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
    (h : 5 * a * b * c > a^3 + b^3 + c^3) :
    a + b > c ∧ a + c > b ∧ b + c > a :=
by
  sorry

end NUMINAMATH_GPT_positive_real_triangle_inequality_l1551_155149


namespace NUMINAMATH_GPT_ferris_break_length_l1551_155163

-- Definitions of the given conditions
def audrey_work_rate := 1 / 4  -- Audrey completes 1/4 of the job per hour
def ferris_work_rate := 1 / 3  -- Ferris completes 1/3 of the job per hour
def total_work_time := 2       -- They worked together for 2 hours
def num_breaks := 6            -- Ferris took 6 breaks during the work period

-- The theorem to prove the length of each break Ferris took
theorem ferris_break_length (break_length : ℝ) :
  (audrey_work_rate * total_work_time) + 
  (ferris_work_rate * (total_work_time - (break_length / 60) * num_breaks)) = 1 →
  break_length = 2.5 :=
by
  sorry

end NUMINAMATH_GPT_ferris_break_length_l1551_155163


namespace NUMINAMATH_GPT_max_jogs_l1551_155156

theorem max_jogs (jags jigs jogs jugs : ℕ) : 2 * jags + 3 * jigs + 8 * jogs + 5 * jugs = 72 → jags ≥ 1 → jigs ≥ 1 → jugs ≥ 1 → jogs ≤ 7 :=
by
  sorry

end NUMINAMATH_GPT_max_jogs_l1551_155156


namespace NUMINAMATH_GPT_consecutive_cubes_perfect_square_l1551_155127

theorem consecutive_cubes_perfect_square :
  ∃ n k : ℕ, (n + 1)^3 - n^3 = k^2 ∧ 
             (∀ m l : ℕ, (m + 1)^3 - m^3 = l^2 → n ≤ m) :=
sorry

end NUMINAMATH_GPT_consecutive_cubes_perfect_square_l1551_155127


namespace NUMINAMATH_GPT_M_gt_N_l1551_155192

variable (x : ℝ)

def M := x^2 + 4 * x - 2

def N := 6 * x - 5

theorem M_gt_N : M x > N x := sorry

end NUMINAMATH_GPT_M_gt_N_l1551_155192


namespace NUMINAMATH_GPT_intersection_of_S_and_T_l1551_155171

-- Define the sets S and T in Lean
def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

-- Prove that S ∩ T = T
theorem intersection_of_S_and_T : S ∩ T = T := 
by {
  sorry -- placeholder for proof
}

end NUMINAMATH_GPT_intersection_of_S_and_T_l1551_155171


namespace NUMINAMATH_GPT_total_weight_of_settings_l1551_155105

-- Define the problem conditions
def weight_silverware_per_piece : ℕ := 4
def pieces_per_setting : ℕ := 3
def weight_plate_per_piece : ℕ := 12
def plates_per_setting : ℕ := 2
def tables : ℕ := 15
def settings_per_table : ℕ := 8
def backup_settings : ℕ := 20

-- Define the calculations
def total_settings_needed : ℕ :=
  (tables * settings_per_table) + backup_settings

def weight_silverware_per_setting : ℕ :=
  pieces_per_setting * weight_silverware_per_piece

def weight_plates_per_setting : ℕ :=
  plates_per_setting * weight_plate_per_piece

def total_weight_per_setting : ℕ :=
  weight_silverware_per_setting + weight_plates_per_setting

def total_weight_all_settings : ℕ :=
  total_settings_needed * total_weight_per_setting

-- Prove the solution
theorem total_weight_of_settings :
  total_weight_all_settings = 5040 :=
sorry

end NUMINAMATH_GPT_total_weight_of_settings_l1551_155105


namespace NUMINAMATH_GPT_tangent_line_eqn_c_range_l1551_155169

noncomputable def f (x : ℝ) := 3 * x * Real.log x + 2

theorem tangent_line_eqn :
  let k := 3 
  let x₀ := 1 
  let y₀ := f x₀ 
  y = k * (x - x₀) + y₀ ↔ 3*x - y - 1 = 0 :=
by sorry

theorem c_range (x : ℝ) (hx : 1 < x) (c : ℝ) :
  f x ≤ x^2 - c * x → c ≤ 1 - 3 * Real.log 2 :=
by sorry

end NUMINAMATH_GPT_tangent_line_eqn_c_range_l1551_155169


namespace NUMINAMATH_GPT_jared_sent_in_november_l1551_155112

noncomputable def text_messages (n : ℕ) : ℕ :=
  match n with
  | 0 => 1  -- November
  | 1 => 2  -- December
  | 2 => 4  -- January
  | 3 => 8  -- February
  | 4 => 16 -- March
  | _ => 0

theorem jared_sent_in_november : text_messages 0 = 1 :=
sorry

end NUMINAMATH_GPT_jared_sent_in_november_l1551_155112


namespace NUMINAMATH_GPT_rug_floor_coverage_l1551_155104

/-- A rectangular rug with side lengths of 2 feet and 7 feet is placed on an irregularly-shaped floor composed of a square with an area of 36 square feet and a right triangle adjacent to one of the square's sides, with leg lengths of 6 feet and 4 feet. If the surface of the rug does not extend beyond the area of the floor, then the fraction of the area of the floor that is not covered by the rug is 17/24. -/
theorem rug_floor_coverage : (48 - 14) / 48 = 17 / 24 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_rug_floor_coverage_l1551_155104


namespace NUMINAMATH_GPT_marbles_initial_count_l1551_155157

theorem marbles_initial_count :
  let total_customers := 20
  let marbles_per_customer := 15
  let marbles_remaining := 100
  ∃ initial_marbles, initial_marbles = total_customers * marbles_per_customer + marbles_remaining :=
by
  let total_customers := 20
  let marbles_per_customer := 15
  let marbles_remaining := 100
  existsi (total_customers * marbles_per_customer + marbles_remaining)
  rfl

end NUMINAMATH_GPT_marbles_initial_count_l1551_155157


namespace NUMINAMATH_GPT_prop_2_l1551_155146

variables (m n : Plane → Prop) (α β γ : Plane)

def perpendicular (m : Line) (α : Plane) : Prop :=
  -- define perpendicular relationship between line and plane
  sorry

def parallel (m : Line) (n : Line) : Prop :=
  -- define parallel relationship between two lines
  sorry

-- The proof of proposition (2) converted into Lean 4 statement
theorem prop_2 (hm₁ : perpendicular m α) (hn₁ : perpendicular n α) : parallel m n :=
  sorry

end NUMINAMATH_GPT_prop_2_l1551_155146


namespace NUMINAMATH_GPT_find_m_if_f_monotonic_l1551_155113

noncomputable def f (m n : ℝ) (x : ℝ) : ℝ :=
  4 * x^3 + m * x^2 + (m - 3) * x + n

def is_monotonically_increasing_on_ℝ (f : ℝ → ℝ) : Prop :=
  ∀ x1 x2 : ℝ, x1 ≤ x2 → f x1 ≤ f x2

theorem find_m_if_f_monotonic (m n : ℝ)
  (h : is_monotonically_increasing_on_ℝ (f m n)) :
  m = 6 :=
sorry

end NUMINAMATH_GPT_find_m_if_f_monotonic_l1551_155113


namespace NUMINAMATH_GPT_prove_RoseHasMoney_l1551_155116
noncomputable def RoseHasMoney : Prop :=
  let cost_of_paintbrush := 2.40
  let cost_of_paints := 9.20
  let cost_of_easel := 6.50
  let total_cost := cost_of_paintbrush + cost_of_paints + cost_of_easel
  let additional_money_needed := 11
  let money_rose_has := total_cost - additional_money_needed
  money_rose_has = 7.10

theorem prove_RoseHasMoney : RoseHasMoney :=
  sorry

end NUMINAMATH_GPT_prove_RoseHasMoney_l1551_155116


namespace NUMINAMATH_GPT_certain_number_correct_l1551_155168

theorem certain_number_correct (x : ℝ) (h1 : 213 * 16 = 3408) (h2 : 213 * x = 340.8) : x = 1.6 := by
  sorry

end NUMINAMATH_GPT_certain_number_correct_l1551_155168


namespace NUMINAMATH_GPT_cos_alpha_sqrt_l1551_155166

theorem cos_alpha_sqrt {α : ℝ} (h1 : Real.sin (π - α) = 1 / 3) (h2 : π / 2 ≤ α ∧ α ≤ π) : 
  Real.cos α = - (2 * Real.sqrt 2) / 3 := 
by
  sorry

end NUMINAMATH_GPT_cos_alpha_sqrt_l1551_155166


namespace NUMINAMATH_GPT_land_area_in_acres_l1551_155199

-- Define the conditions given in the problem.
def length_cm : ℕ := 30
def width_cm : ℕ := 20
def scale_cm_to_mile : ℕ := 1  -- 1 cm corresponds to 1 mile.
def sq_mile_to_acres : ℕ := 640  -- 1 square mile corresponds to 640 acres.

-- Define the statement to be proved.
theorem land_area_in_acres :
  (length_cm * width_cm * sq_mile_to_acres) = 384000 := 
  by sorry

end NUMINAMATH_GPT_land_area_in_acres_l1551_155199


namespace NUMINAMATH_GPT_count_integers_l1551_155102

def Q (x : ℝ) : ℝ := (x - 1) * (x - 4) * (x - 9) * (x - 16) * (x - 25) * (x - 36) * (x - 49) * (x - 64) * (x - 81)

theorem count_integers (Q_le_0 : ∀ n : ℤ, Q n ≤ 0 → ∃ k : ℕ, k = 53) : ∃ k : ℕ, k = 53 := by
  sorry

end NUMINAMATH_GPT_count_integers_l1551_155102


namespace NUMINAMATH_GPT_small_bottles_sold_percentage_l1551_155110

theorem small_bottles_sold_percentage
  (small_bottles : ℕ) (big_bottles : ℕ) (percent_sold_big_bottles : ℝ)
  (remaining_bottles : ℕ) (percent_sold_small_bottles : ℝ) :
  small_bottles = 6000 ∧
  big_bottles = 14000 ∧
  percent_sold_big_bottles = 0.23 ∧
  remaining_bottles = 15580 ∧ 
  percent_sold_small_bottles / 100 * 6000 + 0.23 * 14000 + remaining_bottles = small_bottles + big_bottles →
  percent_sold_small_bottles = 37 := 
by
  intros
  exact sorry

end NUMINAMATH_GPT_small_bottles_sold_percentage_l1551_155110


namespace NUMINAMATH_GPT_dan_remaining_marbles_l1551_155154

-- Define the initial number of marbles Dan has
def initial_marbles : ℕ := 64

-- Define the number of marbles Dan gave to Mary
def marbles_given : ℕ := 14

-- Define the number of remaining marbles
def remaining_marbles : ℕ := initial_marbles - marbles_given

-- State the theorem
theorem dan_remaining_marbles : remaining_marbles = 50 := by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_dan_remaining_marbles_l1551_155154


namespace NUMINAMATH_GPT_volume_proportionality_l1551_155121

variable (W V : ℕ)
variable (k : ℚ)

-- Given conditions
theorem volume_proportionality (h1 : V = k * W) (h2 : W = 112) (h3 : k = 3 / 7) :
  V = 48 := by
  sorry

end NUMINAMATH_GPT_volume_proportionality_l1551_155121


namespace NUMINAMATH_GPT_gcd_840_1764_l1551_155170

theorem gcd_840_1764 : gcd 840 1764 = 84 := 
by
  -- proof steps will go here
  sorry

end NUMINAMATH_GPT_gcd_840_1764_l1551_155170


namespace NUMINAMATH_GPT_evaluate_f_i_l1551_155178

noncomputable def f (x : ℂ) : ℂ :=
  (x^5 + 2 * x^3 + x) / (x + 1)

theorem evaluate_f_i : f (Complex.I) = 0 := 
  sorry

end NUMINAMATH_GPT_evaluate_f_i_l1551_155178


namespace NUMINAMATH_GPT_boat_capacity_per_trip_l1551_155194

theorem boat_capacity_per_trip (trips_per_day : ℕ) (total_people : ℕ) (days : ℕ) :
  trips_per_day = 4 → total_people = 96 → days = 2 → (total_people / (trips_per_day * days)) = 12 :=
by
  intros
  sorry

end NUMINAMATH_GPT_boat_capacity_per_trip_l1551_155194


namespace NUMINAMATH_GPT_solve_for_y_l1551_155186

theorem solve_for_y (x y : ℝ) (h1 : x - y = 8) (h2 : x + y = 10) : y = 1 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_y_l1551_155186


namespace NUMINAMATH_GPT_factorization_x12_minus_729_l1551_155161

theorem factorization_x12_minus_729 (x : ℝ) : 
  x^12 - 729 = (x^2 + 3) * (x^4 - 3 * x^2 + 9) * (x^3 - 3) * (x^3 + 3) :=
by sorry

end NUMINAMATH_GPT_factorization_x12_minus_729_l1551_155161


namespace NUMINAMATH_GPT_two_cos_30_eq_sqrt_3_l1551_155148

open Real

-- Given condition: cos 30 degrees is sqrt(3)/2
def cos_30_eq : cos (π / 6) = sqrt 3 / 2 := 
sorry

-- Goal: to prove that 2 * cos 30 degrees = sqrt(3)
theorem two_cos_30_eq_sqrt_3 : 2 * cos (π / 6) = sqrt 3 :=
by
  rw [cos_30_eq]
  sorry

end NUMINAMATH_GPT_two_cos_30_eq_sqrt_3_l1551_155148


namespace NUMINAMATH_GPT_solve_equation_l1551_155142

def equation (x : ℝ) := (x / (x - 2)) + (2 / (x^2 - 4)) = 1

theorem solve_equation (x : ℝ) (h1 : x ≠ 2) (h2 : x ≠ -2) : 
  equation x ↔ x = -3 :=
by
  sorry

end NUMINAMATH_GPT_solve_equation_l1551_155142


namespace NUMINAMATH_GPT_min_value_of_a_l1551_155172

theorem min_value_of_a
  (f : ℝ → ℝ)
  (h_even : ∀ x : ℝ, f x = f (-x))
  (h_mono : ∀ x y : ℝ, 0 ≤ x → x ≤ y → f x ≤ f y)
  (a : ℝ)
  (h_cond : f (Real.logb 2 a) + f (Real.logb (1/2) a) ≤ 2 * f 1) :
  a = 1/2 := sorry

end NUMINAMATH_GPT_min_value_of_a_l1551_155172


namespace NUMINAMATH_GPT_speed_in_still_water_l1551_155114

theorem speed_in_still_water (U D : ℝ) (hU : U = 15) (hD : D = 25) : (U + D) / 2 = 20 :=
by
  rw [hU, hD]
  norm_num

end NUMINAMATH_GPT_speed_in_still_water_l1551_155114


namespace NUMINAMATH_GPT_verify_total_amount_spent_by_mary_l1551_155179

def shirt_price : Float := 13.04
def shirt_sales_tax_rate : Float := 0.07

def jacket_original_price_gbp : Float := 15.34
def jacket_discount_rate : Float := 0.20
def jacket_sales_tax_rate : Float := 0.085
def conversion_rate_usd_per_gbp : Float := 1.28

def scarf_price : Float := 7.90
def hat_price : Float := 9.13
def hat_scarf_sales_tax_rate : Float := 0.065

def total_amount_spent_by_mary : Float :=
  let shirt_total := shirt_price * (1 + shirt_sales_tax_rate)
  let jacket_discounted := jacket_original_price_gbp * (1 - jacket_discount_rate)
  let jacket_total_gbp := jacket_discounted * (1 + jacket_sales_tax_rate)
  let jacket_total_usd := jacket_total_gbp * conversion_rate_usd_per_gbp
  let hat_scarf_combined_price := scarf_price + hat_price
  let hat_scarf_total := hat_scarf_combined_price * (1 + hat_scarf_sales_tax_rate)
  shirt_total + jacket_total_usd + hat_scarf_total

theorem verify_total_amount_spent_by_mary : total_amount_spent_by_mary = 49.13 :=
by sorry

end NUMINAMATH_GPT_verify_total_amount_spent_by_mary_l1551_155179


namespace NUMINAMATH_GPT_find_views_multiplier_l1551_155111

theorem find_views_multiplier (M: ℝ) (h: 4000 * M + 50000 = 94000) : M = 11 :=
by
  sorry

end NUMINAMATH_GPT_find_views_multiplier_l1551_155111


namespace NUMINAMATH_GPT_larger_integer_is_24_l1551_155185

theorem larger_integer_is_24 {x : ℤ} (h1 : ∃ x, 4 * x = x + 6) :
  ∃ y, y = 4 * x ∧ y = 24 := by
  sorry

end NUMINAMATH_GPT_larger_integer_is_24_l1551_155185


namespace NUMINAMATH_GPT_trigonometric_expression_value_l1551_155135

-- Define the line equation and the conditions about the slope angle
def line_eq (x y : ℝ) : Prop := 6 * x - 2 * y - 5 = 0

-- The slope angle alpha
variable (α : ℝ)

-- Given conditions
axiom slope_tan : Real.tan α = 3

-- The expression we need to prove equals -2
theorem trigonometric_expression_value :
  (Real.sin (Real.pi - α) + Real.cos (-α)) / (Real.sin (-α) - Real.cos (Real.pi + α)) = -2 :=
by
  sorry

end NUMINAMATH_GPT_trigonometric_expression_value_l1551_155135


namespace NUMINAMATH_GPT_triangle_inequality_proof_l1551_155119

variable (a b c : ℝ)

-- Condition that a, b, c are side lengths of a triangle
axiom triangle_inequality1 : a + b > c
axiom triangle_inequality2 : b + c > a
axiom triangle_inequality3 : c + a > b

-- Theorem stating the required inequality and the condition for equality
theorem triangle_inequality_proof :
  a^2 * b * (a - b) + b^2 * c * (b - c) + c^2 * a * (c - a) ≥ 0 ∧
  (a^2 * b * (a - b) + b^2 * c * (b - c) + c^2 * a * (c - a) = 0 ↔ a = b ∧ b = c ∧ c = a) :=
sorry

end NUMINAMATH_GPT_triangle_inequality_proof_l1551_155119


namespace NUMINAMATH_GPT_wind_velocity_l1551_155125

def pressure (P A V : ℝ) (k : ℝ) : Prop :=
  P = k * A * V^2

theorem wind_velocity (k : ℝ) (h_initial : pressure 4 4 8 k) (h_final : pressure 64 16 v k) : v = 16 := by
  sorry

end NUMINAMATH_GPT_wind_velocity_l1551_155125


namespace NUMINAMATH_GPT_manufacturing_section_degrees_l1551_155108

variable (percentage_manufacturing : ℝ) (total_degrees : ℝ)

theorem manufacturing_section_degrees
  (h1 : percentage_manufacturing = 0.40)
  (h2 : total_degrees = 360) :
  percentage_manufacturing * total_degrees = 144 := 
by 
  sorry

end NUMINAMATH_GPT_manufacturing_section_degrees_l1551_155108


namespace NUMINAMATH_GPT_find_toonies_l1551_155150

-- Define the number of coins and their values
variables (L T : ℕ) -- L represents the number of loonies, T represents the number of toonies

-- Define the conditions
def total_coins := L + T = 10
def total_value := 1 * L + 2 * T = 14

-- Define the theorem to be proven
theorem find_toonies (L T : ℕ) (h1 : total_coins L T) (h2 : total_value L T) : T = 4 :=
by
  sorry

end NUMINAMATH_GPT_find_toonies_l1551_155150


namespace NUMINAMATH_GPT_bricks_in_wall_l1551_155147

-- Definitions for individual working times and breaks
def Bea_build_time := 8  -- hours
def Bea_break_time := 10 / 60  -- hours per hour
def Ben_build_time := 12  -- hours
def Ben_break_time := 15 / 60  -- hours per hour

-- Total effective rates
def Bea_effective_rate (h : ℕ) := h / (Bea_build_time * (1 - Bea_break_time))
def Ben_effective_rate (h : ℕ) := h / (Ben_build_time * (1 - Ben_break_time))

-- Decreased rate due to talking
def total_effective_rate (h : ℕ) := Bea_effective_rate h + Ben_effective_rate h - 12

-- Define the Lean proof statement
theorem bricks_in_wall (h : ℕ) :
  (6 * total_effective_rate h = h) → h = 127 :=
by sorry

end NUMINAMATH_GPT_bricks_in_wall_l1551_155147


namespace NUMINAMATH_GPT_measure_of_C_and_max_perimeter_l1551_155143

noncomputable def triangle_C_and_perimeter (a b c A B C : ℝ) (hABC : (2 * a + b) * Real.sin A + (2 * b + a) * Real.sin B = 2 * c * Real.sin C) (hc : c = Real.sqrt 3) : Prop :=
  (C = 2 * Real.pi / 3) ∧ (2 * Real.sin A + 2 * Real.sin B + c ≤ 2 + Real.sqrt 3)

-- Now the Lean theorem statement
theorem measure_of_C_and_max_perimeter (a b c A B C : ℝ) (hABC : (2 * a + b) * Real.sin A + (2 * b + a) * Real.sin B = 2 * c * Real.sin C) (hc : c = Real.sqrt 3) :
  triangle_C_and_perimeter a b c A B C hABC hc :=
by 
  sorry

end NUMINAMATH_GPT_measure_of_C_and_max_perimeter_l1551_155143


namespace NUMINAMATH_GPT_flowerbed_seeds_l1551_155101

theorem flowerbed_seeds (n_fbeds n_seeds_per_fbed total_seeds : ℕ)
    (h1 : n_fbeds = 8)
    (h2 : n_seeds_per_fbed = 4) :
    total_seeds = n_fbeds * n_seeds_per_fbed := by
  sorry

end NUMINAMATH_GPT_flowerbed_seeds_l1551_155101


namespace NUMINAMATH_GPT_value_of_a_l1551_155164

-- Define the sets A and B and the intersection condition
def A (a : ℝ) : Set ℝ := {a ^ 2, a + 1, -3}
def B (a : ℝ) : Set ℝ := {a - 3, 2 * a - 1, a ^ 2 + 1}

theorem value_of_a (a : ℝ) (h : A a ∩ B a = {-3}) : a = -1 :=
by {
  -- Insert proof here when ready, using h to show a = -1
  sorry
}

end NUMINAMATH_GPT_value_of_a_l1551_155164


namespace NUMINAMATH_GPT_min_value_expression_l1551_155123

theorem min_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) : 
    (x + 1 / y) ^ 2 + (y + 1 / (2 * x)) ^ 2 ≥ 3 + 2 * Real.sqrt 2 := 
sorry

end NUMINAMATH_GPT_min_value_expression_l1551_155123


namespace NUMINAMATH_GPT_annual_rent_per_square_foot_l1551_155117

theorem annual_rent_per_square_foot 
  (monthly_rent : ℕ) 
  (length : ℕ) 
  (width : ℕ) 
  (area : ℕ)
  (annual_rent : ℕ) : 
  monthly_rent = 3600 → 
  length = 18 → 
  width = 20 → 
  area = length * width → 
  annual_rent = monthly_rent * 12 → 
  annual_rent / area = 120 :=
by
  sorry

end NUMINAMATH_GPT_annual_rent_per_square_foot_l1551_155117


namespace NUMINAMATH_GPT_intersection_compl_A_compl_B_l1551_155120

open Set

variable (x y : ℝ)

def U : Set ℝ := univ
def A : Set ℝ := {x | -1 < x ∧ x < 4}
def B : Set ℝ := {y | ∃ x, y = x + 1 ∧ -1 < x ∧ x < 4}

theorem intersection_compl_A_compl_B (U A B : Set ℝ) (hU : U = univ) (hA : A = {x | -1 < x ∧ x < 4}) (hB : B = {y | ∃ x, y = x + 1 ∧ -1 < x ∧ x < 4}):
  (Aᶜ ∩ Bᶜ) = (Iic (-1) ∪ Ici 5) :=
by
  sorry

end NUMINAMATH_GPT_intersection_compl_A_compl_B_l1551_155120


namespace NUMINAMATH_GPT_distance_AC_l1551_155132

theorem distance_AC (t_Eddy t_Freddy : ℕ) (d_AB : ℝ) (speed_ratio : ℝ) : 
  t_Eddy = 3 ∧ t_Freddy = 4 ∧ d_AB = 510 ∧ speed_ratio = 2.2666666666666666 → 
  ∃ d_AC : ℝ, d_AC = 300 :=
by 
  intros h
  obtain ⟨hE, hF, hD, hR⟩ := h
  -- Declare velocities
  let v_Eddy : ℝ := d_AB / t_Eddy
  let v_Freddy : ℝ := v_Eddy / speed_ratio
  let d_AC : ℝ := v_Freddy * t_Freddy
  -- Prove the distance
  use d_AC
  sorry

end NUMINAMATH_GPT_distance_AC_l1551_155132


namespace NUMINAMATH_GPT_parallel_line_perpendicular_line_l1551_155188

theorem parallel_line (x y : ℝ) (h : y = 2 * x + 3) : ∃ a : ℝ, 3 * x - 2 * y + a = 0 :=
by
  use 1
  sorry

theorem perpendicular_line  (x y : ℝ) (h : y = -x / 2) : ∃ c : ℝ, 3 * x - 2 * y + c = 0 :=
by
  use -5
  sorry

end NUMINAMATH_GPT_parallel_line_perpendicular_line_l1551_155188


namespace NUMINAMATH_GPT_solve_for_x_l1551_155189

theorem solve_for_x :
  { x : Real | ⌊ 2 * x * ⌊ x ⌋ ⌋ = 58 } = {x : Real | 5.8 ≤ x ∧ x < 5.9} :=
sorry

end NUMINAMATH_GPT_solve_for_x_l1551_155189


namespace NUMINAMATH_GPT_winner_last_year_ounces_l1551_155195

/-- Definition of the problem conditions -/
def ouncesPerHamburger : ℕ := 4
def hamburgersTonyaAte : ℕ := 22

/-- Theorem stating the desired result -/
theorem winner_last_year_ounces :
  hamburgersTonyaAte * ouncesPerHamburger = 88 :=
by
  sorry

end NUMINAMATH_GPT_winner_last_year_ounces_l1551_155195
