import Mathlib

namespace NUMINAMATH_GPT_students_between_hoseok_and_minyoung_l1814_181461

def num_students : Nat := 13
def hoseok_position_from_right : Nat := 9
def minyoung_position_from_left : Nat := 8

theorem students_between_hoseok_and_minyoung
    (n : Nat)
    (h : n = num_students)
    (p_h : n - hoseok_position_from_right + 1 = 5)
    (p_m : minyoung_position_from_left = 8):
    ∃ k : Nat, k = 2 :=
by
  sorry

end NUMINAMATH_GPT_students_between_hoseok_and_minyoung_l1814_181461


namespace NUMINAMATH_GPT_total_distance_of_journey_l1814_181443

-- Definitions corresponding to conditions in the problem
def electric_distance : ℝ := 30 -- The first 30 miles were in electric mode
def gasoline_consumption_rate : ℝ := 0.03 -- Gallons per mile for gasoline mode
def average_mileage : ℝ := 50 -- Miles per gallon for the entire trip

-- Final goal: proving the total distance is 90 miles
theorem total_distance_of_journey (d : ℝ) :
  (d / (gasoline_consumption_rate * (d - electric_distance)) = average_mileage) → d = 90 :=
by
  sorry

end NUMINAMATH_GPT_total_distance_of_journey_l1814_181443


namespace NUMINAMATH_GPT_gcd_660_924_l1814_181449

theorem gcd_660_924 : Nat.gcd 660 924 = 132 := by
  sorry

end NUMINAMATH_GPT_gcd_660_924_l1814_181449


namespace NUMINAMATH_GPT_smallest_n_l1814_181404

theorem smallest_n (n : ℕ) : 
  (2^n + 5^n - n) % 1000 = 0 ↔ n = 797 :=
sorry

end NUMINAMATH_GPT_smallest_n_l1814_181404


namespace NUMINAMATH_GPT_range_of_m_min_value_a2_2b2_3c2_l1814_181471

theorem range_of_m (x m : ℝ) (h : ∀ x : ℝ, abs (x + 3) + abs (x + m) ≥ 2 * m) : m ≤ 1 :=
sorry

theorem min_value_a2_2b2_3c2 (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a + b + c = 1) :
  ∃ (a b c : ℝ), a = 6/11 ∧ b = 3/11 ∧ c = 2/11 ∧ a^2 + 2 * b^2 + 3 * c^2 = 6/11 :=
sorry

end NUMINAMATH_GPT_range_of_m_min_value_a2_2b2_3c2_l1814_181471


namespace NUMINAMATH_GPT_periodic_minus_decimal_is_correct_l1814_181468

-- Definitions based on conditions

def periodic_63_as_fraction : ℚ := 63 / 99
def decimal_63_as_fraction : ℚ := 63 / 100
def difference : ℚ := periodic_63_as_fraction - decimal_63_as_fraction

-- Lean 4 statement to prove the mathematically equivalent proof problem
theorem periodic_minus_decimal_is_correct :
  difference = 7 / 1100 :=
by
  sorry

end NUMINAMATH_GPT_periodic_minus_decimal_is_correct_l1814_181468


namespace NUMINAMATH_GPT_like_terms_ratio_l1814_181438

theorem like_terms_ratio (m n : ℕ) (h₁ : m - 2 = 2) (h₂ : 3 = 2 * n - 1) : m / n = 2 := 
by
  sorry

end NUMINAMATH_GPT_like_terms_ratio_l1814_181438


namespace NUMINAMATH_GPT_both_false_of_not_or_l1814_181417

-- Define propositions p and q
variables (p q : Prop)

-- The condition given: ¬(p ∨ q)
theorem both_false_of_not_or (h : ¬(p ∨ q)) : ¬ p ∧ ¬ q :=
by {
  sorry
}

end NUMINAMATH_GPT_both_false_of_not_or_l1814_181417


namespace NUMINAMATH_GPT_find_positive_square_root_l1814_181430

theorem find_positive_square_root (x : ℝ) (h_pos : x > 0) (h_eq : x^2 = 625) : x = 25 :=
sorry

end NUMINAMATH_GPT_find_positive_square_root_l1814_181430


namespace NUMINAMATH_GPT_domain_f_l1814_181455

noncomputable def f (x : ℝ) : ℝ := (Real.sqrt x) / ((x^2) - 4)

theorem domain_f : {x : ℝ | 0 ≤ x ∧ x ≠ 2} = {x | 0 ≤ x ∧ x < 2} ∪ {x | x > 2} :=
by sorry

end NUMINAMATH_GPT_domain_f_l1814_181455


namespace NUMINAMATH_GPT_add_base8_l1814_181402

-- Define x and y in base 8 and their sum in base 8
def x := 24 -- base 8
def y := 157 -- base 8
def result := 203 -- base 8

theorem add_base8 : (x + y) = result := 
by sorry

end NUMINAMATH_GPT_add_base8_l1814_181402


namespace NUMINAMATH_GPT_who_had_second_value_card_in_first_game_l1814_181458

variable (A B C : ℕ)
variable (x y z : ℕ)
variable (points_A points_B points_C : ℕ)

-- Provided conditions
variable (h1 : x < y ∧ y < z)
variable (h2 : points_A = 20)
variable (h3 : points_B = 10)
variable (h4 : points_C = 9)
variable (number_of_games : ℕ)
variable (h5 : number_of_games = 3)
variable (h6 : A + B + C = 39)  -- This corresponds to points_A + points_B + points_C = 39.
variable (h7 : ∃ x y z, x + y + z = 13 ∧ x < y ∧ y < z)
variable (h8 : B = z)

-- Question/Proof to establish
theorem who_had_second_value_card_in_first_game :
  ∃ p : ℕ, p = C :=
sorry

end NUMINAMATH_GPT_who_had_second_value_card_in_first_game_l1814_181458


namespace NUMINAMATH_GPT_new_tv_cost_l1814_181426

/-
Mark bought his first TV which was 24 inches wide and 16 inches tall. It cost $672.
His new TV is 48 inches wide and 32 inches tall.
The first TV was $1 more expensive per square inch compared to his newest TV.
Prove that the cost of his new TV is $1152.
-/

theorem new_tv_cost :
  let width_first_tv := 24
  let height_first_tv := 16
  let cost_first_tv := 672
  let width_new_tv := 48
  let height_new_tv := 32
  let discount_per_square_inch := 1
  let area_first_tv := width_first_tv * height_first_tv
  let cost_per_square_inch_first_tv := cost_first_tv / area_first_tv
  let cost_per_square_inch_new_tv := cost_per_square_inch_first_tv - discount_per_square_inch
  let area_new_tv := width_new_tv * height_new_tv
  let cost_new_tv := cost_per_square_inch_new_tv * area_new_tv
  cost_new_tv = 1152 := by
  sorry

end NUMINAMATH_GPT_new_tv_cost_l1814_181426


namespace NUMINAMATH_GPT_einstein_fundraising_l1814_181485

def boxes_of_pizza : Nat := 15
def packs_of_potato_fries : Nat := 40
def cans_of_soda : Nat := 25
def price_per_box : ℝ := 12
def price_per_pack : ℝ := 0.3
def price_per_can : ℝ := 2
def goal_amount : ℝ := 500

theorem einstein_fundraising : goal_amount - (boxes_of_pizza * price_per_box + packs_of_potato_fries * price_per_pack + cans_of_soda * price_per_can) = 258 := by
  sorry

end NUMINAMATH_GPT_einstein_fundraising_l1814_181485


namespace NUMINAMATH_GPT_wine_age_proof_l1814_181410

-- Definitions based on conditions
def Age_Carlo_Rosi : ℕ := 40
def Age_Twin_Valley : ℕ := Age_Carlo_Rosi / 4
def Age_Franzia : ℕ := 3 * Age_Carlo_Rosi

-- We'll use a definition to represent the total age of the three brands of wine.
def Total_Age : ℕ := Age_Franzia + Age_Carlo_Rosi + Age_Twin_Valley

-- Statement to be proven
theorem wine_age_proof : Total_Age = 170 :=
by {
  sorry -- Proof goes here
}

end NUMINAMATH_GPT_wine_age_proof_l1814_181410


namespace NUMINAMATH_GPT_maximum_area_of_triangle_l1814_181496

theorem maximum_area_of_triangle :
  ∃ (b c : ℝ), (a = 2) ∧ (A = 60 * Real.pi / 180) ∧
  (∀ S : ℝ, S = (1/2) * b * c * Real.sin A → S ≤ Real.sqrt 3) :=
by sorry

end NUMINAMATH_GPT_maximum_area_of_triangle_l1814_181496


namespace NUMINAMATH_GPT_triangle_angle_bisector_theorem_l1814_181489

variable {α : Type*} [LinearOrderedField α]

theorem triangle_angle_bisector_theorem (A B C D : α)
  (h1 : A^2 = (C + D) * (B - (B * D / C)))
  (h2 : B / C = (B * D / C) / D) :
  A^2 = C * B - D * (B * D / C) := 
  by
  sorry

end NUMINAMATH_GPT_triangle_angle_bisector_theorem_l1814_181489


namespace NUMINAMATH_GPT_number_of_ways_to_divide_friends_l1814_181412

theorem number_of_ways_to_divide_friends :
  let friends := 8
  let teams := 4
  (teams ^ friends) = 65536 := by
  sorry

end NUMINAMATH_GPT_number_of_ways_to_divide_friends_l1814_181412


namespace NUMINAMATH_GPT_distance_of_intersection_points_l1814_181499

def C1 (x y : ℝ) : Prop := x - y + 4 = 0
def C2 (x y : ℝ) : Prop := (x + 2)^2 + (y - 1)^2 = 1

theorem distance_of_intersection_points {A B : ℝ × ℝ} (hA1 : C1 A.fst A.snd) (hA2 : C2 A.fst A.snd)
  (hB1 : C1 B.fst B.snd) (hB2 : C2 B.fst B.snd) : dist A B = Real.sqrt 2 := by
  sorry

end NUMINAMATH_GPT_distance_of_intersection_points_l1814_181499


namespace NUMINAMATH_GPT_tan_x_value_l1814_181446

noncomputable def f (x : ℝ) : ℝ := Real.sin x - Real.cos x

theorem tan_x_value:
  (∀ x : ℝ, deriv f x = 2 * f x) → (∀ x : ℝ, f x = Real.sin x - Real.cos x) → (∀ x : ℝ, Real.tan x = 3) := 
by
  intros h_deriv h_f
  sorry

end NUMINAMATH_GPT_tan_x_value_l1814_181446


namespace NUMINAMATH_GPT_inequality_proof_l1814_181434

theorem inequality_proof
  (a b c : ℝ)
  (h1 : a > 0)
  (h2 : b > 0)
  (h3 : c > 0)
  (h4 : a + b + c = 1) :
  a * (1 + b - c) ^ (1 / 3) + b * (1 + c - a) ^ (1 / 3) + c * (1 + a - b) ^ (1 / 3) ≤ 1 := 
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l1814_181434


namespace NUMINAMATH_GPT_range_of_a_l1814_181408

noncomputable def f (x : ℝ) : ℝ := 
  if h : x ≤ 1 then x^2 - x + 3 else 0

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, f x ≥ |x / 2 + a|) ↔ -47 / 16 ≤ a ∧ a ≤ 2 := sorry

end NUMINAMATH_GPT_range_of_a_l1814_181408


namespace NUMINAMATH_GPT_solve_for_n_l1814_181457

theorem solve_for_n (n : ℝ) : 0.03 * n + 0.05 * (30 + n) + 2 = 8.5 → n = 62.5 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_solve_for_n_l1814_181457


namespace NUMINAMATH_GPT_oranges_to_apples_ratio_l1814_181453

theorem oranges_to_apples_ratio :
  ∀ (total_fruits : ℕ) (weight_oranges : ℕ) (weight_apples : ℕ),
  total_fruits = 12 →
  weight_oranges = 10 →
  weight_apples = total_fruits - weight_oranges →
  weight_oranges / weight_apples = 5 :=
by
  intros total_fruits weight_oranges weight_apples h1 h2 h3
  sorry

end NUMINAMATH_GPT_oranges_to_apples_ratio_l1814_181453


namespace NUMINAMATH_GPT_percentage_decrease_hours_worked_l1814_181451

theorem percentage_decrease_hours_worked (B H : ℝ) (h₁ : H > 0) (h₂ : B > 0)
  (h_assistant1 : (1.8 * B) = B * 1.8) (h_assistant2 : (2 * (B / H)) = (1.8 * B) / (0.9 * H)) : 
  ((H - (0.9 * H)) / H) * 100 = 10 := 
by
  sorry

end NUMINAMATH_GPT_percentage_decrease_hours_worked_l1814_181451


namespace NUMINAMATH_GPT_graph_is_hyperbola_l1814_181401

theorem graph_is_hyperbola : ∀ (x y : ℝ), x^2 - 18 * y^2 - 6 * x + 4 * y + 9 = 0 → ∃ a b c d : ℝ, a * (x - b)^2 - c * (y - d)^2 = 1 :=
by
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_graph_is_hyperbola_l1814_181401


namespace NUMINAMATH_GPT_bookA_net_change_bookB_net_change_bookC_net_change_l1814_181448

-- Define the price adjustments for Book A
def bookA_initial_price := 100.0
def bookA_after_first_adjustment := bookA_initial_price * (1 - 0.5)
def bookA_after_second_adjustment := bookA_after_first_adjustment * (1 + 0.6)
def bookA_final_price := bookA_after_second_adjustment * (1 + 0.1)
def bookA_net_percentage_change := (bookA_final_price - bookA_initial_price) / bookA_initial_price * 100

-- Define the price adjustments for Book B
def bookB_initial_price := 100.0
def bookB_after_first_adjustment := bookB_initial_price * (1 + 0.2)
def bookB_after_second_adjustment := bookB_after_first_adjustment * (1 - 0.3)
def bookB_final_price := bookB_after_second_adjustment * (1 + 0.25)
def bookB_net_percentage_change := (bookB_final_price - bookB_initial_price) / bookB_initial_price * 100

-- Define the price adjustments for Book C
def bookC_initial_price := 100.0
def bookC_after_first_adjustment := bookC_initial_price * (1 + 0.4)
def bookC_after_second_adjustment := bookC_after_first_adjustment * (1 - 0.1)
def bookC_final_price := bookC_after_second_adjustment * (1 - 0.05)
def bookC_net_percentage_change := (bookC_final_price - bookC_initial_price) / bookC_initial_price * 100

-- Statements to prove the net percentage changes
theorem bookA_net_change : bookA_net_percentage_change = -12 := by
  sorry

theorem bookB_net_change : bookB_net_percentage_change = 5 := by
  sorry

theorem bookC_net_change : bookC_net_percentage_change = 19.7 := by
  sorry

end NUMINAMATH_GPT_bookA_net_change_bookB_net_change_bookC_net_change_l1814_181448


namespace NUMINAMATH_GPT_sequence_geometric_l1814_181483

theorem sequence_geometric (a : ℕ → ℝ) (h : ∀ n, a n ≠ 0)
  (h_arith : 2 * a 2 = a 1 + a 3)
  (h_geom : a 3 ^ 2 = a 2 * a 4)
  (h_recip_arith : 2 / a 4 = 1 / a 3 + 1 / a 5) :
  a 3 ^ 2 = a 1 * a 5 :=
sorry

end NUMINAMATH_GPT_sequence_geometric_l1814_181483


namespace NUMINAMATH_GPT_prove_a_lt_one_l1814_181472

/-- Given the function f defined as -2 * ln x + 1 / 2 * (x^2 + 1) - a * x,
    where a > 0, if f(x) ≥ 0 holds in the interval (1, ∞)
    and f(x) = 0 has a unique solution, then a < 1. -/
theorem prove_a_lt_one (f : ℝ → ℝ) (a : ℝ) 
    (h1 : ∀ x, f x = -2 * Real.log x + 1 / 2 * (x^2 + 1) - a * x)
    (h2 : a > 0)
    (h3 : ∀ x, x > 1 → f x ≥ 0)
    (h4 : ∃! x, f x = 0) : 
    a < 1 :=
by
  sorry

end NUMINAMATH_GPT_prove_a_lt_one_l1814_181472


namespace NUMINAMATH_GPT_trig_identity_proof_l1814_181479

noncomputable def sin_30 : Real := 1 / 2
noncomputable def cos_120 : Real := -1 / 2
noncomputable def cos_45 : Real := Real.sqrt 2 / 2
noncomputable def tan_30 : Real := Real.sqrt 3 / 3

theorem trig_identity_proof : 
  sin_30 + cos_120 + 2 * cos_45 - Real.sqrt 3 * tan_30 = Real.sqrt 2 - 1 := 
by
  sorry

end NUMINAMATH_GPT_trig_identity_proof_l1814_181479


namespace NUMINAMATH_GPT_max_min_x2_minus_xy_plus_y2_l1814_181470

theorem max_min_x2_minus_xy_plus_y2 (x y: ℝ) (h : |5 * x + y| + |5 * x - y| = 20) : 
  3 ≤ x^2 - x * y + y^2 ∧ x^2 - x * y + y^2 ≤ 124 := 
sorry

end NUMINAMATH_GPT_max_min_x2_minus_xy_plus_y2_l1814_181470


namespace NUMINAMATH_GPT_find_A_l1814_181462

def spadesuit (A B : ℝ) : ℝ := 4 * A + 3 * B - 2

theorem find_A (A : ℝ) : spadesuit A 7 = 40 ↔ A = 21 / 4 :=
by
  sorry

end NUMINAMATH_GPT_find_A_l1814_181462


namespace NUMINAMATH_GPT_arithmetic_seq_fraction_l1814_181469

theorem arithmetic_seq_fraction (a : ℕ → ℤ) (d : ℤ) 
  (h1 : ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d) 
  (h2 : a 1 + a 10 = a 9) 
  (d_ne_zero : d ≠ 0) : 
  (a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9) / a 10 = 27 / 8 := 
sorry

end NUMINAMATH_GPT_arithmetic_seq_fraction_l1814_181469


namespace NUMINAMATH_GPT_alice_bob_meeting_point_l1814_181429

def meet_same_point (turns : ℕ) : Prop :=
  ∃ n : ℕ, turns = 2 * n ∧ 18 ∣ (7 * n - (7 * n + n))

theorem alice_bob_meeting_point :
  meet_same_point 36 :=
by
  sorry

end NUMINAMATH_GPT_alice_bob_meeting_point_l1814_181429


namespace NUMINAMATH_GPT_find_plane_speed_l1814_181492

-- Defining the values in the problem
def distance_with_wind : ℝ := 420
def distance_against_wind : ℝ := 350
def wind_speed : ℝ := 23

-- The speed of the plane in still air
def plane_speed_in_still_air : ℝ := 253

-- Proof goal: Given the conditions, the speed of the plane in still air is 253 mph
theorem find_plane_speed :
  ∃ p : ℝ, (distance_with_wind / (p + wind_speed) = distance_against_wind / (p - wind_speed)) ∧ p = plane_speed_in_still_air :=
by
  use plane_speed_in_still_air
  have h : plane_speed_in_still_air = 253 := rfl
  sorry

end NUMINAMATH_GPT_find_plane_speed_l1814_181492


namespace NUMINAMATH_GPT_smallest_number_condition_l1814_181403

theorem smallest_number_condition :
  ∃ n : ℕ, (n + 1) % 12 = 0 ∧
           (n + 1) % 18 = 0 ∧
           (n + 1) % 24 = 0 ∧
           (n + 1) % 32 = 0 ∧
           (n + 1) % 40 = 0 ∧
           n = 2879 :=
sorry

end NUMINAMATH_GPT_smallest_number_condition_l1814_181403


namespace NUMINAMATH_GPT_value_of_x_plus_4_l1814_181421

theorem value_of_x_plus_4 (x : ℝ) (h : 2 * x + 6 = 16) : x + 4 = 9 :=
by
  sorry

end NUMINAMATH_GPT_value_of_x_plus_4_l1814_181421


namespace NUMINAMATH_GPT_find_p_l1814_181428

open Real

variable (A : ℝ × ℝ)
variable (p : ℝ) (hp : p > 0)

-- Conditions
def on_parabola (A : ℝ × ℝ) (p : ℝ) : Prop := A.snd^2 = 2 * p * A.fst
def dist_focus (A : ℝ × ℝ) (p : ℝ) : Prop := sqrt ((A.fst - p / 2)^2 + A.snd^2) = 12
def dist_y_axis (A : ℝ × ℝ) : Prop := abs (A.fst) = 9

-- Theorem to prove
theorem find_p (h1 : on_parabola A p) (h2 : dist_focus A p) (h3 : dist_y_axis A) : p = 6 :=
sorry

end NUMINAMATH_GPT_find_p_l1814_181428


namespace NUMINAMATH_GPT_price_of_refrigerator_l1814_181475

variable (R W : ℝ)

theorem price_of_refrigerator 
  (h1 : W = R - 1490) 
  (h2 : R + W = 7060) 
  : R = 4275 :=
sorry

end NUMINAMATH_GPT_price_of_refrigerator_l1814_181475


namespace NUMINAMATH_GPT_triangle_inequality_l1814_181416

theorem triangle_inequality (a b c : ℝ) (α : ℝ) 
  (h_triangle_sides : a + b > c ∧ b + c > a ∧ c + a > b) 
  (h_cosine_rule : a^2 = b^2 + c^2 - 2 * b * c * Real.cos α) :
  (2 * b * c * Real.cos α) / (b + c) < (b + c - a) ∧ (b + c - a) < (2 * b * c) / a := 
sorry

end NUMINAMATH_GPT_triangle_inequality_l1814_181416


namespace NUMINAMATH_GPT_distance_A_C_15_l1814_181423

noncomputable def distance_from_A_to_C : ℝ := 
  let AB := 6
  let AC := AB + (3 * AB) / 2
  AC

theorem distance_A_C_15 (A B C D : ℝ) (h1 : A < B) (h2 : B < C) (h3 : C < D)
  (h4 : D - A = 24) (h5 : D - B = 3 * (B - A)) 
  (h6 : C = (B + D) / 2) :
  distance_from_A_to_C = 15 :=
by sorry

end NUMINAMATH_GPT_distance_A_C_15_l1814_181423


namespace NUMINAMATH_GPT_solution_set_of_inequality_l1814_181407

theorem solution_set_of_inequality :
  {x : ℝ | -1 < x ∧ x < 2} = {x : ℝ | (x - 2) / (x + 1) < 0} :=
sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l1814_181407


namespace NUMINAMATH_GPT_two_trains_crossing_time_l1814_181473

theorem two_trains_crossing_time
  (length_train: ℝ) (time_telegraph_post_first: ℝ) (time_telegraph_post_second: ℝ)
  (length_train_eq: length_train = 120) 
  (time_telegraph_post_first_eq: time_telegraph_post_first = 10) 
  (time_telegraph_post_second_eq: time_telegraph_post_second = 15) :
  (2 * length_train) / (length_train / time_telegraph_post_first + length_train / time_telegraph_post_second) = 12 :=
by
  sorry

end NUMINAMATH_GPT_two_trains_crossing_time_l1814_181473


namespace NUMINAMATH_GPT_last_two_digits_of_sum_of_factorials_l1814_181487

-- Problem statement: Sum of factorials from 1 to 15
def sum_factorials (n : ℕ) : ℕ :=
  (Finset.range n).sum (fun k => Nat.factorial k)

-- Define the main problem
theorem last_two_digits_of_sum_of_factorials : 
  (sum_factorials 15) % 100 = 13 :=
by 
  sorry

end NUMINAMATH_GPT_last_two_digits_of_sum_of_factorials_l1814_181487


namespace NUMINAMATH_GPT_part_I_part_II_l1814_181422

noncomputable def a (n : Nat) : Nat := sorry

def is_odd (n : Nat) : Prop := n % 2 = 1

theorem part_I
  (h : a 1 = 19) :
  a 2014 = 98 := by
  sorry

theorem part_II
  (h1: ∀ n : Nat, is_odd (a n))
  (h2: ∀ n m : Nat, a n = a m) -- constant sequence
  (h3: ∀ n : Nat, a n > 1) :
  ∃ k : Nat, a k = 5 := by
  sorry


end NUMINAMATH_GPT_part_I_part_II_l1814_181422


namespace NUMINAMATH_GPT_acute_triangle_integers_count_l1814_181486

theorem acute_triangle_integers_count :
  ∃ (x_vals : List ℕ), (∀ x ∈ x_vals, 7 < x ∧ x < 33 ∧ (if x > 20 then x^2 < 569 else x > Int.sqrt 231)) ∧ x_vals.length = 8 :=
by
  sorry

end NUMINAMATH_GPT_acute_triangle_integers_count_l1814_181486


namespace NUMINAMATH_GPT_total_spent_l1814_181491

theorem total_spent (B D : ℝ) (h1 : D = 0.7 * B) (h2 : B = D + 15) : B + D = 85 :=
sorry

end NUMINAMATH_GPT_total_spent_l1814_181491


namespace NUMINAMATH_GPT_original_ratio_l1814_181436

theorem original_ratio (F J : ℚ) (hJ : J = 180) (h_ratio : (F + 45) / J = 3 / 2) : F / J = 5 / 4 :=
by
  sorry

end NUMINAMATH_GPT_original_ratio_l1814_181436


namespace NUMINAMATH_GPT_parts_of_cut_square_l1814_181414

theorem parts_of_cut_square (folds_to_one_by_one : ℕ) : folds_to_one_by_one = 9 :=
  sorry

end NUMINAMATH_GPT_parts_of_cut_square_l1814_181414


namespace NUMINAMATH_GPT_calc_S_5_minus_S_4_l1814_181493

def sum_sequence (a : ℕ → ℕ) (S : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, S n = 2 * a n - 2

theorem calc_S_5_minus_S_4 {a : ℕ → ℕ} {S : ℕ → ℕ}
  (h : sum_sequence a S) : S 5 - S 4 = 32 :=
by
  sorry

end NUMINAMATH_GPT_calc_S_5_minus_S_4_l1814_181493


namespace NUMINAMATH_GPT_total_chairs_l1814_181459

-- Define the conditions as constants
def living_room_chairs : ℕ := 3
def kitchen_chairs : ℕ := 6
def dining_room_chairs : ℕ := 8
def outdoor_patio_chairs : ℕ := 12

-- State the goal to prove
theorem total_chairs : 
  living_room_chairs + kitchen_chairs + dining_room_chairs + outdoor_patio_chairs = 29 := 
by
  -- The proof is not required as per instructions
  sorry

end NUMINAMATH_GPT_total_chairs_l1814_181459


namespace NUMINAMATH_GPT_Cody_initial_money_l1814_181456

-- Define the conditions
def initial_money (x : ℕ) : Prop :=
  x + 9 - 19 = 35

-- Define the theorem we need to prove
theorem Cody_initial_money : initial_money 45 :=
by
  -- Add a placeholder for the proof
  sorry

end NUMINAMATH_GPT_Cody_initial_money_l1814_181456


namespace NUMINAMATH_GPT_minimal_value_of_function_l1814_181409

theorem minimal_value_of_function (x : ℝ) (hx : x > 1 / 2) :
  (x = 1 → (x^2 + 1) / x = 2) ∧
  (∀ y, (∀ z, z > 1 / 2 → y ≤ (z^2 + 1) / z) → y = 2) :=
by {
  sorry
}

end NUMINAMATH_GPT_minimal_value_of_function_l1814_181409


namespace NUMINAMATH_GPT_pen_cost_is_2_25_l1814_181480

variables (p i : ℝ)

def total_cost (p i : ℝ) : Prop := p + i = 2.50
def pen_more_expensive (p i : ℝ) : Prop := p = 2 + i

theorem pen_cost_is_2_25 (p i : ℝ) 
  (h1 : total_cost p i) 
  (h2 : pen_more_expensive p i) : 
  p = 2.25 := 
by
  sorry

end NUMINAMATH_GPT_pen_cost_is_2_25_l1814_181480


namespace NUMINAMATH_GPT_inequality_subtraction_l1814_181498

variable (a b : ℝ)

theorem inequality_subtraction (h : a > b) : a - 5 > b - 5 :=
sorry

end NUMINAMATH_GPT_inequality_subtraction_l1814_181498


namespace NUMINAMATH_GPT_determine_n_for_square_l1814_181427

theorem determine_n_for_square (n : ℕ) : (∃ a : ℕ, 5^n + 4 = a^2) ↔ n = 1 :=
by
-- The proof will be included here, but for now, we just provide the structure
sorry

end NUMINAMATH_GPT_determine_n_for_square_l1814_181427


namespace NUMINAMATH_GPT_bricks_for_wall_l1814_181466

theorem bricks_for_wall
  (wall_length : ℕ) (wall_height : ℕ) (wall_width : ℕ)
  (brick_length : ℕ) (brick_height : ℕ) (brick_width : ℕ)
  (L_eq : wall_length = 600) (H_eq : wall_height = 400) (W_eq : wall_width = 2050)
  (l_eq : brick_length = 30) (h_eq : brick_height = 12) (w_eq : brick_width = 10)
  : (wall_length * wall_height * wall_width) / (brick_length * brick_height * brick_width) = 136667 :=
by
  sorry

end NUMINAMATH_GPT_bricks_for_wall_l1814_181466


namespace NUMINAMATH_GPT_return_trip_time_l1814_181413

theorem return_trip_time 
  (d p w : ℝ) 
  (h1 : d = 90 * (p - w))
  (h2 : ∀ t, t = d / p → d / (p + w) = t - 15) : 
  d / (p + w) = 64 :=
by
  sorry

end NUMINAMATH_GPT_return_trip_time_l1814_181413


namespace NUMINAMATH_GPT_john_new_earnings_after_raise_l1814_181464

-- Definition of original earnings and raise percentage
def original_earnings : ℝ := 50
def raise_percentage : ℝ := 0.50

-- Calculate raise amount and new earnings after raise
def raise_amount : ℝ := raise_percentage * original_earnings
def new_earnings : ℝ := original_earnings + raise_amount

-- Math proof problem: Prove new earnings after raise equals $75
theorem john_new_earnings_after_raise : new_earnings = 75 := by
  sorry

end NUMINAMATH_GPT_john_new_earnings_after_raise_l1814_181464


namespace NUMINAMATH_GPT_ruby_candies_l1814_181490

theorem ruby_candies (number_of_friends : ℕ) (candies_per_friend : ℕ) (total_candies : ℕ)
  (h1 : number_of_friends = 9)
  (h2 : candies_per_friend = 4)
  (h3 : total_candies = number_of_friends * candies_per_friend) :
  total_candies = 36 :=
by {
  sorry
}

end NUMINAMATH_GPT_ruby_candies_l1814_181490


namespace NUMINAMATH_GPT_chain_of_tangent_circles_iff_l1814_181478

-- Define the circles, their centers, and the conditions
structure Circle := 
  (center : ℝ × ℝ) 
  (radius : ℝ)

structure TangentData :=
  (circle1 : Circle)
  (circle2 : Circle)
  (angle : ℝ)

-- Non-overlapping condition
def non_overlapping (c1 c2 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  let dist := (x2 - x1)^2 + (y2 - y1)^2
  dist > (c1.radius + c2.radius)^2

-- Existence of tangent circles condition
def exists_chain_of_tangent_circles (c1 c2 : Circle) (n : ℕ) : Prop :=
  ∃ (tangent_circle : Circle), tangent_circle.radius = c1.radius ∨ tangent_circle.radius = c2.radius

-- Angle condition
def angle_condition (ang : ℝ) (n : ℕ) : Prop :=
  ∃ (k : ℤ), ang = k * (360 / n)

-- Final theorem to prove
theorem chain_of_tangent_circles_iff (c1 c2 : Circle) (t : TangentData) (n : ℕ) 
  (h1 : non_overlapping c1 c2) 
  (h2 : t.circle1 = c1 ∧ t.circle2 = c2) 
  : exists_chain_of_tangent_circles c1 c2 n ↔ angle_condition t.angle n := 
  sorry

end NUMINAMATH_GPT_chain_of_tangent_circles_iff_l1814_181478


namespace NUMINAMATH_GPT_solve_for_k_l1814_181497

theorem solve_for_k (x : ℝ) (k : ℝ) (h₁ : 2 * x - 1 = 3) (h₂ : 3 * x + k = 0) : k = -6 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_k_l1814_181497


namespace NUMINAMATH_GPT_identify_conic_section_hyperbola_l1814_181476

-- Defining the variables and constants in the Lean environment
variable (x y : ℝ)

-- The given equation in function form
def conic_section_eq : Prop := (x - 3) ^ 2 = 4 * (y + 2) ^ 2 + 25

-- The expected type of conic section (Hyperbola)
def is_hyperbola : Prop := 
  ∃ (a b c d e f : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ a * x^2 - b * y^2 + c * x + d * y + e = f

-- The theorem statement to prove
theorem identify_conic_section_hyperbola (h : conic_section_eq x y) : is_hyperbola x y := by
  sorry

end NUMINAMATH_GPT_identify_conic_section_hyperbola_l1814_181476


namespace NUMINAMATH_GPT_num_houses_with_digit_7_in_range_l1814_181467

-- Define the condition for a number to contain a digit 7
def contains_digit_7 (n : Nat) : Prop :=
  (n / 10 = 7) || (n % 10 = 7)

-- The main theorem
theorem num_houses_with_digit_7_in_range (h : Nat) (H1 : 1 ≤ h ∧ h ≤ 70) : 
  ∃! n, 1 ≤ n ∧ n ≤ 70 ∧ contains_digit_7 n :=
sorry

end NUMINAMATH_GPT_num_houses_with_digit_7_in_range_l1814_181467


namespace NUMINAMATH_GPT_solve_fraction_eq_l1814_181415

theorem solve_fraction_eq (x : ℝ) :
  (1 / ((x - 1) * (x - 2)) + 1 / ((x - 2) * (x - 3)) + 1 / ((x - 3) * (x - 4)) = 1 / 6) ↔ 
  (x = 7 ∨ x = -2) := 
by
  sorry

end NUMINAMATH_GPT_solve_fraction_eq_l1814_181415


namespace NUMINAMATH_GPT_find_g1_l1814_181463

variables {f g : ℝ → ℝ}

def odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x
def even_function (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = g x

theorem find_g1 (hf : odd_function f)
                (hg : even_function g)
                (h1 : f (-1) + g 1 = 2)
                (h2 : f 1 + g (-1) = 4) :
                g 1 = 3 :=
sorry

end NUMINAMATH_GPT_find_g1_l1814_181463


namespace NUMINAMATH_GPT_town_population_original_l1814_181465

noncomputable def original_population (n : ℕ) : Prop :=
  let increased_population := n + 1500
  let decreased_population := (85 / 100 : ℚ) * increased_population
  decreased_population = n + 1455

theorem town_population_original : ∃ n : ℕ, original_population n ∧ n = 1200 :=
by
  sorry

end NUMINAMATH_GPT_town_population_original_l1814_181465


namespace NUMINAMATH_GPT_difference_in_surface_area_l1814_181437

-- Defining the initial conditions
def original_length : ℝ := 6
def original_width : ℝ := 5
def original_height : ℝ := 4
def cube_side : ℝ := 2

-- Define the surface area calculation for a rectangular solid
def surface_area_rectangular_prism (l w h : ℝ) : ℝ :=
  2 * (l * w + l * h + w * h)

-- Define the surface area of the cube
def surface_area_cube (a : ℝ) : ℝ :=
  6 * a * a

-- Define the removed face areas when cube is extracted
def exposed_faces_area (a : ℝ) : ℝ :=
  2 * (a * a)

-- Define the problem statement in Lean
theorem difference_in_surface_area :
  surface_area_rectangular_prism original_length original_width original_height
  - (surface_area_rectangular_prism original_length original_width original_height - surface_area_cube cube_side + exposed_faces_area cube_side) = 12 :=
by
  sorry

end NUMINAMATH_GPT_difference_in_surface_area_l1814_181437


namespace NUMINAMATH_GPT_contractor_absent_days_l1814_181424

theorem contractor_absent_days
    (total_days : ℤ) (work_rate : ℤ) (fine_rate : ℤ) (total_amount : ℤ)
    (x y : ℤ)
    (h1 : total_days = 30)
    (h2 : work_rate = 25)
    (h3 : fine_rate = 75) -- fine_rate here is multiplied by 10 to avoid decimals
    (h4 : total_amount = 4250) -- total_amount multiplied by 10 for the same reason
    (h5 : x + y = total_days)
    (h6 : work_rate * x - fine_rate * y = total_amount) :
  y = 10 := 
by
  -- Here, we would provide the proof steps.
  sorry

end NUMINAMATH_GPT_contractor_absent_days_l1814_181424


namespace NUMINAMATH_GPT_measure_angle_C_and_area_l1814_181431

noncomputable def triangleProblem (a b c A B C : ℝ) :=
  (a + b = 5) ∧ (c = Real.sqrt 7) ∧ (4 * Real.sin ((A + B) / 2)^2 - Real.cos (2 * C) = 7 / 2)

theorem measure_angle_C_and_area (a b c A B C : ℝ) (h: triangleProblem a b c A B C) :
  C = Real.pi / 3 ∧ (1 / 2) * a * b * Real.sin C = (3 * Real.sqrt 3) / 2 :=
by
  obtain ⟨ha, hb, hc⟩ := h
  sorry

end NUMINAMATH_GPT_measure_angle_C_and_area_l1814_181431


namespace NUMINAMATH_GPT_range_of_m_l1814_181445

theorem range_of_m (m : ℝ) : (∀ x : ℝ, |x-3| + |x+4| ≥ |2*m-1|) ↔ (-3 ≤ m ∧ m ≤ 4) := by
  sorry

end NUMINAMATH_GPT_range_of_m_l1814_181445


namespace NUMINAMATH_GPT_locus_is_circle_l1814_181494

open Complex

noncomputable def circle_center (a b : ℝ) : ℂ := Complex.ofReal (-a / (a^2 + b^2)) + Complex.I * (b / (a^2 + b^2))
noncomputable def circle_radius (a b : ℝ) : ℝ := 1 / Real.sqrt (a^2 + b^2)

theorem locus_is_circle (z0 z1 z : ℂ) (h1 : abs (z1 - z0) = abs z1) (h2 : z0 ≠ 0) (h3 : z1 * z = -1) :
  ∃ (a b : ℝ), z0 = Complex.ofReal a + Complex.I * b ∧
    (∃ c : ℂ, z = c ∧ 
      (c.re + a / (a^2 + b^2))^2 + (c.im - b / (a^2 + b^2))^2 = 1 / (a^2 + b^2)) := by
  sorry

end NUMINAMATH_GPT_locus_is_circle_l1814_181494


namespace NUMINAMATH_GPT_tangent_line_b_value_l1814_181411

theorem tangent_line_b_value (a k b : ℝ) 
  (h_curve : ∀ x, x^3 + a * x + 1 = 3 ↔ x = 2)
  (h_derivative : k = 3 * 2^2 - 3)
  (h_tangent : 3 = k * 2 + b) : b = -15 :=
sorry

end NUMINAMATH_GPT_tangent_line_b_value_l1814_181411


namespace NUMINAMATH_GPT_binomial_coeff_x5y3_in_expansion_eq_56_l1814_181481

theorem binomial_coeff_x5y3_in_expansion_eq_56:
  let n := 8
  let k := 3
  let binom_coeff := Nat.choose n k
  binom_coeff = 56 := 
by sorry

end NUMINAMATH_GPT_binomial_coeff_x5y3_in_expansion_eq_56_l1814_181481


namespace NUMINAMATH_GPT_michelle_has_total_crayons_l1814_181477

noncomputable def michelle_crayons : ℕ :=
  let type1_crayons_per_box := 5
  let type2_crayons_per_box := 12
  let type1_boxes := 4
  let type2_boxes := 3
  let missing_crayons := 2
  (type1_boxes * type1_crayons_per_box - missing_crayons) + (type2_boxes * type2_crayons_per_box)

theorem michelle_has_total_crayons : michelle_crayons = 54 :=
by
  -- The proof step would go here, but it is omitted according to instructions.
  sorry

end NUMINAMATH_GPT_michelle_has_total_crayons_l1814_181477


namespace NUMINAMATH_GPT_sum_partition_ominous_years_l1814_181450

def is_ominous (n : ℕ) : Prop :=
  n = 1 ∨ Nat.Prime n

theorem sum_partition_ominous_years :
  ∀ n : ℕ, (¬ ∃ (A B : Finset ℕ), A ∪ B = Finset.range (n + 1) ∧ A ∩ B = ∅ ∧ 
    (A.sum id = B.sum id ∧ A.card = B.card)) ↔ is_ominous n := 
sorry

end NUMINAMATH_GPT_sum_partition_ominous_years_l1814_181450


namespace NUMINAMATH_GPT_smallest_sum_of_squares_l1814_181454

theorem smallest_sum_of_squares (x y : ℤ) (h : x^2 - y^2 = 231) :
  x^2 + y^2 ≥ 281 :=
sorry

end NUMINAMATH_GPT_smallest_sum_of_squares_l1814_181454


namespace NUMINAMATH_GPT_equivalent_expression_l1814_181442

theorem equivalent_expression (a b c : ℝ) (h : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0) (h1 : a + b + c = 0) :
  (a^4 * b^4 + a^4 * c^4 + b^4 * c^4) / ((a^2 - b*c)^2 * (b^2 - a*c)^2 * (c^2 - a*b)^2) = 
  1 / (a^2 - b*c)^2 :=
by
  sorry

end NUMINAMATH_GPT_equivalent_expression_l1814_181442


namespace NUMINAMATH_GPT_total_collection_value_l1814_181474

theorem total_collection_value (total_stickers : ℕ) (partial_stickers : ℕ) (partial_value : ℕ)
  (same_value : ∀ (stickers : ℕ), stickers = total_stickers → stickers * partial_value / partial_stickers = stickers * (partial_value / partial_stickers)):
  partial_value = 24 ∧ partial_stickers = 6 ∧ total_stickers = 18 → total_stickers * (partial_value / partial_stickers) = 72 :=
by {
  sorry
}

end NUMINAMATH_GPT_total_collection_value_l1814_181474


namespace NUMINAMATH_GPT_factorial_expression_l1814_181405

theorem factorial_expression :
  7 * (Nat.factorial 7) + 6 * (Nat.factorial 6) + 2 * (Nat.factorial 6) = 41040 := by
  sorry

end NUMINAMATH_GPT_factorial_expression_l1814_181405


namespace NUMINAMATH_GPT_find_x_l1814_181444

theorem find_x
    (x : ℝ)
    (l : ℝ := 4 * x)
    (w : ℝ := x + 8)
    (area_eq_twice_perimeter : l * w = 2 * (2 * l + 2 * w)) :
    x = 2 :=
by
  sorry

end NUMINAMATH_GPT_find_x_l1814_181444


namespace NUMINAMATH_GPT_tom_age_ratio_l1814_181420

theorem tom_age_ratio (T : ℕ) (h1 : T = 3 * (3 : ℕ)) (h2 : T - 5 = 3 * ((T / 3) - 10)) : T / 5 = 9 := 
by
  sorry

end NUMINAMATH_GPT_tom_age_ratio_l1814_181420


namespace NUMINAMATH_GPT_find_a_l1814_181433

noncomputable def base25_num : ℕ := 3 * 25^7 + 1 * 25^6 + 4 * 25^5 + 2 * 25^4 + 6 * 25^3 + 5 * 25^2 + 2 * 25^1 + 3 * 25^0

theorem find_a (a : ℤ) (h0 : 0 ≤ a) (h1 : a ≤ 14) : ((base25_num - a) % 12 = 0) → a = 2 := 
sorry

end NUMINAMATH_GPT_find_a_l1814_181433


namespace NUMINAMATH_GPT_decrease_in_silver_coins_l1814_181425

theorem decrease_in_silver_coins
  (a : ℕ) (h₁ : 2 * a = 3 * (50 - a))
  (h₂ : a + (50 - a) = 50) :
  (5 * (50 - a) - 3 * a = 10) :=
by
sorry

end NUMINAMATH_GPT_decrease_in_silver_coins_l1814_181425


namespace NUMINAMATH_GPT_certain_number_l1814_181439

theorem certain_number (n q1 q2: ℕ) (h1 : 49 = n * q1 + 4) (h2 : 66 = n * q2 + 6): n = 15 :=
sorry

end NUMINAMATH_GPT_certain_number_l1814_181439


namespace NUMINAMATH_GPT_expression_B_between_2_and_3_l1814_181419

variable (a b : ℝ)
variable (h : 3 * a = 5 * b)

theorem expression_B_between_2_and_3 : 2 < (|a + b| / b) ∧ (|a + b| / b) < 3 :=
by sorry

end NUMINAMATH_GPT_expression_B_between_2_and_3_l1814_181419


namespace NUMINAMATH_GPT_train_passing_time_l1814_181406

-- Definitions based on the conditions
def length_T1 : ℕ := 800
def speed_T1_kmph : ℕ := 108
def length_T2 : ℕ := 600
def speed_T2_kmph : ℕ := 72

-- Converting kmph to mps
def convert_kmph_to_mps (speed_kmph : ℕ) : ℕ := speed_kmph * 1000 / 3600
def speed_T1_mps : ℕ := convert_kmph_to_mps speed_T1_kmph
def speed_T2_mps : ℕ := convert_kmph_to_mps speed_T2_kmph

-- Calculating relative speed and total length
def relative_speed_T1_T2 : ℕ := speed_T1_mps - speed_T2_mps
def total_length_T1_T2 : ℕ := length_T1 + length_T2

-- Proving the time to pass
theorem train_passing_time : total_length_T1_T2 / relative_speed_T1_T2 = 140 := by
  sorry

end NUMINAMATH_GPT_train_passing_time_l1814_181406


namespace NUMINAMATH_GPT_small_boxes_count_correct_l1814_181482

-- Definitions of constants
def feet_per_large_box_seal : ℕ := 4
def feet_per_medium_box_seal : ℕ := 2
def feet_per_small_box_seal : ℕ := 1
def feet_per_box_label : ℕ := 1

def large_boxes_packed : ℕ := 2
def medium_boxes_packed : ℕ := 8
def total_tape_used : ℕ := 44

-- Definition for the total tape used for large and medium boxes
def tape_used_large_boxes : ℕ := (large_boxes_packed * feet_per_large_box_seal) + (large_boxes_packed * feet_per_box_label)
def tape_used_medium_boxes : ℕ := (medium_boxes_packed * feet_per_medium_box_seal) + (medium_boxes_packed * feet_per_box_label)
def tape_used_large_and_medium_boxes : ℕ := tape_used_large_boxes + tape_used_medium_boxes
def tape_used_small_boxes : ℕ := total_tape_used - tape_used_large_and_medium_boxes

-- The number of small boxes packed
def small_boxes_packed : ℕ := tape_used_small_boxes / (feet_per_small_box_seal + feet_per_box_label)

-- Proof problem statement
theorem small_boxes_count_correct (n : ℕ) (h : small_boxes_packed = n) : n = 5 :=
by
  sorry

end NUMINAMATH_GPT_small_boxes_count_correct_l1814_181482


namespace NUMINAMATH_GPT_jack_pays_back_expected_amount_l1814_181400

-- Definitions from the conditions
def principal : ℝ := 1200
def interest_rate : ℝ := 0.10

-- Definition for proof
def interest : ℝ := principal * interest_rate
def total_amount : ℝ := principal + interest

-- Lean statement for the proof problem
theorem jack_pays_back_expected_amount : total_amount = 1320 := by
  sorry

end NUMINAMATH_GPT_jack_pays_back_expected_amount_l1814_181400


namespace NUMINAMATH_GPT_evaluate_expression_l1814_181460

theorem evaluate_expression (a : ℝ) : (a^7 + a^7 + a^7 - a^7) = a^8 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l1814_181460


namespace NUMINAMATH_GPT_lcm_18_20_l1814_181432

theorem lcm_18_20 : Nat.lcm 18 20 = 180 := by
  sorry

end NUMINAMATH_GPT_lcm_18_20_l1814_181432


namespace NUMINAMATH_GPT_total_letters_in_all_names_l1814_181484

theorem total_letters_in_all_names :
  let jonathan_first := 8
  let jonathan_surname := 10
  let younger_sister_first := 5
  let younger_sister_surname := 10
  let older_brother_first := 6
  let older_brother_surname := 10
  let youngest_sibling_first := 4
  let youngest_sibling_hyphenated_surname := 15
  jonathan_first + jonathan_surname + younger_sister_first + younger_sister_surname +
  older_brother_first + older_brother_surname + youngest_sibling_first + youngest_sibling_hyphenated_surname = 68 := by
  sorry

end NUMINAMATH_GPT_total_letters_in_all_names_l1814_181484


namespace NUMINAMATH_GPT_sum_of_first_70_odd_integers_l1814_181452

theorem sum_of_first_70_odd_integers : 
  let sum_even := 70 * (70 + 1)
  let sum_odd := 70 ^ 2
  let diff := sum_even - sum_odd
  diff = 70 → sum_odd = 4900 :=
by
  intros
  sorry

end NUMINAMATH_GPT_sum_of_first_70_odd_integers_l1814_181452


namespace NUMINAMATH_GPT_brian_stones_l1814_181418

variable (W B : ℕ)
variable (total_stones : ℕ := 100)
variable (G : ℕ := 40)
variable (Gr : ℕ := 60)

theorem brian_stones :
  (W > B) →
  ((W + B = total_stones) ∧ (G + Gr = total_stones) ∧ (W = 60)) :=
by
  sorry

end NUMINAMATH_GPT_brian_stones_l1814_181418


namespace NUMINAMATH_GPT_smallest_possible_a_l1814_181495

noncomputable def f (a b c : ℕ) (x : ℝ) : ℝ := a * x^2 + b * x + ↑c

theorem smallest_possible_a
  (a b c : ℕ)
  (r s : ℝ)
  (h_arith_seq : b - a = c - b)
  (h_order_pos : 0 < a ∧ a < b ∧ b < c)
  (h_distinct : r ≠ s)
  (h_rs_2017 : r * s = 2017)
  (h_fr_eq_s : f a b c r = s)
  (h_fs_eq_r : f a b c s = r) :
  a = 1 := sorry

end NUMINAMATH_GPT_smallest_possible_a_l1814_181495


namespace NUMINAMATH_GPT_largest_sum_is_7_over_12_l1814_181440

-- Define the five sums
def sum1 : ℚ := 1/3 + 1/4
def sum2 : ℚ := 1/3 + 1/5
def sum3 : ℚ := 1/3 + 1/6
def sum4 : ℚ := 1/3 + 1/9
def sum5 : ℚ := 1/3 + 1/8

-- Define the problem statement
theorem largest_sum_is_7_over_12 : 
  max (max (max sum1 sum2) (max sum3 sum4)) sum5 = 7/12 := 
by
  sorry

end NUMINAMATH_GPT_largest_sum_is_7_over_12_l1814_181440


namespace NUMINAMATH_GPT_solve_n_minus_m_l1814_181435

theorem solve_n_minus_m :
  ∃ m n, 
    (m ≡ 4 [MOD 7]) ∧ 100 ≤ m ∧ m < 1000 ∧ 
    (n ≡ 4 [MOD 7]) ∧ 1000 ≤ n ∧ n < 10000 ∧ 
    n - m = 903 :=
by
  sorry

end NUMINAMATH_GPT_solve_n_minus_m_l1814_181435


namespace NUMINAMATH_GPT_sqrt_diff_approx_l1814_181447

theorem sqrt_diff_approx : abs ((Real.sqrt 122) - (Real.sqrt 120) - 0.15) < 0.01 := 
sorry

end NUMINAMATH_GPT_sqrt_diff_approx_l1814_181447


namespace NUMINAMATH_GPT_sequence_recurrence_l1814_181441

theorem sequence_recurrence (a : ℕ → ℝ) (h₀ : a 1 = 1) (h : ∀ n : ℕ, n ≥ 1 → a (n + 1) = (n / (n + 1)) * a n) :
  ∀ n : ℕ, n ≥ 1 → a n = 1 / n :=
by
  intro n hn
  exact sorry

end NUMINAMATH_GPT_sequence_recurrence_l1814_181441


namespace NUMINAMATH_GPT_inequality_proof_l1814_181488

open Real

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  a^4 * b^b * c^c ≥ a⁻¹ * b⁻¹ * c⁻¹ :=
sorry

end NUMINAMATH_GPT_inequality_proof_l1814_181488
