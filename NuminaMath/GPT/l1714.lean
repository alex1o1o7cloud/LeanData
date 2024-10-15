import Mathlib

namespace NUMINAMATH_GPT_sum_of_products_of_roots_l1714_171447

theorem sum_of_products_of_roots :
  ∀ (p q r : ℝ), (4 * p^3 - 6 * p^2 + 17 * p - 10 = 0) ∧ 
                 (4 * q^3 - 6 * q^2 + 17 * q - 10 = 0) ∧ 
                 (4 * r^3 - 6 * r^2 + 17 * r - 10 = 0) →
                 (p * q + q * r + r * p = 17 / 4) :=
by
  sorry

end NUMINAMATH_GPT_sum_of_products_of_roots_l1714_171447


namespace NUMINAMATH_GPT_fraction_of_visitors_l1714_171410

variable (V E U : ℕ)
variable (H1 : E = U)
variable (H2 : 600 - E - 150 = 450)

theorem fraction_of_visitors (H3 : 600 = E + 150 + 450) : (450 : ℚ) / 600 = (3 : ℚ) / 4 :=
by
  apply sorry

end NUMINAMATH_GPT_fraction_of_visitors_l1714_171410


namespace NUMINAMATH_GPT_total_texts_sent_l1714_171430

def texts_sent_monday_allison : ℕ := 5
def texts_sent_monday_brittney : ℕ := 5
def texts_sent_tuesday_allison : ℕ := 15
def texts_sent_tuesday_brittney : ℕ := 15

theorem total_texts_sent : (texts_sent_monday_allison + texts_sent_monday_brittney) + 
                           (texts_sent_tuesday_allison + texts_sent_tuesday_brittney) = 40 :=
by
  sorry

end NUMINAMATH_GPT_total_texts_sent_l1714_171430


namespace NUMINAMATH_GPT_transform_polynomial_l1714_171474

variable (x z : ℝ)

theorem transform_polynomial (h1 : z = x - 1 / x) (h2 : x^4 - 3 * x^3 - 2 * x^2 + 3 * x + 1 = 0) :
  x^2 * (z^2 - 3 * z) = 0 :=
sorry

end NUMINAMATH_GPT_transform_polynomial_l1714_171474


namespace NUMINAMATH_GPT_horse_revolutions_l1714_171492

noncomputable def carousel_revolutions (r1 r2 d1 : ℝ) : ℝ :=
  (d1 * r1) / r2

theorem horse_revolutions :
  carousel_revolutions 30 10 40 = 120 :=
by
  sorry

end NUMINAMATH_GPT_horse_revolutions_l1714_171492


namespace NUMINAMATH_GPT_complex_arithmetic_1_complex_arithmetic_2_l1714_171408

-- Proof Problem 1
theorem complex_arithmetic_1 : 
  (1 : ℂ) * (-2 - 4 * I) - (7 - 5 * I) + (1 + 7 * I) = -8 + 8 * I := 
sorry

-- Proof Problem 2
theorem complex_arithmetic_2 : 
  (1 + I) * (2 + I) + (5 + I) / (1 - I) + (1 - I) ^ 2 = 3 + 4 * I := 
sorry

end NUMINAMATH_GPT_complex_arithmetic_1_complex_arithmetic_2_l1714_171408


namespace NUMINAMATH_GPT_sum_of_consecutive_at_least_20_sum_of_consecutive_greater_than_20_l1714_171427

noncomputable def sum_of_consecutive_triplets (a : Fin 12 → ℕ) (i : Fin 12) : ℕ :=
a i + a ((i + 1) % 12) + a ((i + 2) % 12)

theorem sum_of_consecutive_at_least_20 :
  ∀ (a : Fin 12 → ℕ), (∀ i : Fin 12, (1 ≤ a i ∧ a i ≤ 12) ∧ ∀ (j k : Fin 12), j ≠ k → a j ≠ a k) →
  ∃ i : Fin 12, sum_of_consecutive_triplets a i ≥ 20 :=
by
  sorry

theorem sum_of_consecutive_greater_than_20 :
  ∀ (a : Fin 12 → ℕ), (∀ i : Fin 12, (1 ≤ a i ∧ a i ≤ 12) ∧ ∀ (j k : Fin 12), j ≠ k → a j ≠ a k) →
  ∃ i : Fin 12, sum_of_consecutive_triplets a i > 20 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_consecutive_at_least_20_sum_of_consecutive_greater_than_20_l1714_171427


namespace NUMINAMATH_GPT_neg_ex_iff_forall_geq_0_l1714_171452

theorem neg_ex_iff_forall_geq_0 :
  ¬(∃ x_0 : ℝ, x_0^2 - x_0 + 1 < 0) ↔ ∀ x : ℝ, x^2 - x + 1 ≥ 0 :=
by
  sorry

end NUMINAMATH_GPT_neg_ex_iff_forall_geq_0_l1714_171452


namespace NUMINAMATH_GPT_number_of_y_axis_returns_l1714_171470

-- Definitions based on conditions
noncomputable def unit_length : ℝ := 0.5
noncomputable def diagonal_length : ℝ := Real.sqrt 2 * unit_length
noncomputable def pen_length_cm : ℝ := 8000 * 100 -- converting meters to cm
noncomputable def circle_length (n : ℕ) : ℝ := ((3 + Real.sqrt 2) * n ^ 2 + 2 * n) * unit_length

-- The main theorem
theorem number_of_y_axis_returns : ∃ n : ℕ, circle_length n ≤ pen_length_cm ∧ circle_length (n+1) > pen_length_cm :=
sorry

end NUMINAMATH_GPT_number_of_y_axis_returns_l1714_171470


namespace NUMINAMATH_GPT_setB_is_empty_l1714_171441

noncomputable def setB := {x : ℝ | x^2 + 1 = 0}

theorem setB_is_empty : setB = ∅ :=
by
  sorry

end NUMINAMATH_GPT_setB_is_empty_l1714_171441


namespace NUMINAMATH_GPT_find_b_l1714_171476

theorem find_b (g : ℝ → ℝ) (g_inv : ℝ → ℝ) (b : ℝ) (h_g_def : ∀ x, g x = 1 / (3 * x + b)) (h_g_inv_def : ∀ x, g_inv x = (1 - 3 * x) / (3 * x)) :
  b = 3 :=
by
  sorry

end NUMINAMATH_GPT_find_b_l1714_171476


namespace NUMINAMATH_GPT_minimum_S_n_at_1008_a1008_neg_a1009_pos_common_difference_pos_l1714_171423

variable {a : ℕ → ℝ}
variable {S : ℕ → ℝ}

-- Conditions based on the problem statements
axiom a1_neg : a 1 < 0
axiom S2015_neg : S 2015 < 0
axiom S2016_pos : S 2016 > 0

-- Defining n value where S_n reaches its minimum
def n_min := 1008

theorem minimum_S_n_at_1008 : S n_min = S 1008 := sorry

-- Additional theorems to satisfy the provided conditions
theorem a1008_neg : a 1008 < 0 := sorry
theorem a1009_pos : a 1009 > 0 := sorry
theorem common_difference_pos : ∀ n : ℕ, a (n + 1) - a n > 0 := sorry

end NUMINAMATH_GPT_minimum_S_n_at_1008_a1008_neg_a1009_pos_common_difference_pos_l1714_171423


namespace NUMINAMATH_GPT_triangle_area_40_l1714_171439

noncomputable def area_of_triangle (base height : ℕ) : ℕ :=
  base * height / 2

theorem triangle_area_40
  (a : ℕ) (P B Q : (ℕ × ℕ)) (PB_side : (P.1 = 0 ∧ P.2 = 0) ∧ (B.1 = 10 ∧ B.2 = 0))
  (Q_vert_aboveP : Q.1 = 0 ∧ Q.2 = 8)
  (PQ_perp_PB : P.1 = Q.1)
  (PQ_length : (Q.snd - P.snd) = 8) :
  area_of_triangle 10 8 = 40 := by
  sorry

end NUMINAMATH_GPT_triangle_area_40_l1714_171439


namespace NUMINAMATH_GPT_range_f_g_f_eq_g_implies_A_l1714_171435

open Set

noncomputable def f (x : ℝ) : ℝ := x^2 + 1
noncomputable def g (x : ℝ) : ℝ := 4 * x + 1

theorem range_f_g :
  (range f ∩ Icc 1 17 = Icc 1 17) ∧ (range g ∩ Icc 1 17 = Icc 1 17) :=
sorry

theorem f_eq_g_implies_A :
  ∀ A ⊆ Icc 0 4, (∀ x ∈ A, f x = g x) → A = {0} ∨ A = {4} ∨ A = {0, 4} :=
sorry

end NUMINAMATH_GPT_range_f_g_f_eq_g_implies_A_l1714_171435


namespace NUMINAMATH_GPT_most_likely_outcome_l1714_171443

-- Define the probabilities for each outcome
def P_all_boys := (1/2)^6
def P_all_girls := (1/2)^6
def P_3_girls_3_boys := (Nat.choose 6 3) * (1/2)^6
def P_4_one_2_other := 2 * (Nat.choose 6 2) * (1/2)^6

-- Terms with values of each probability
lemma outcome_A : P_all_boys = 1 / 64 := by sorry
lemma outcome_B : P_all_girls = 1 / 64 := by sorry
lemma outcome_C : P_3_girls_3_boys = 20 / 64 := by sorry
lemma outcome_D : P_4_one_2_other = 30 / 64 := by sorry

-- Prove the main statement
theorem most_likely_outcome :
  P_4_one_2_other > P_all_boys ∧ P_4_one_2_other > P_all_girls ∧ P_4_one_2_other > P_3_girls_3_boys :=
by
  rw [outcome_A, outcome_B, outcome_C, outcome_D]
  sorry

end NUMINAMATH_GPT_most_likely_outcome_l1714_171443


namespace NUMINAMATH_GPT_least_number_divisible_l1714_171465

theorem least_number_divisible (x : ℕ) (h1 : x = 857) 
  (h2 : (x + 7) % 24 = 0) 
  (h3 : (x + 7) % 36 = 0) 
  (h4 : (x + 7) % 54 = 0) :
  (x + 7) % 32 = 0 := 
sorry

end NUMINAMATH_GPT_least_number_divisible_l1714_171465


namespace NUMINAMATH_GPT_slope_CD_l1714_171403

def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 6*x + 4*y - 8 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 16*x + 8*y + 40 = 0

theorem slope_CD :
  ∀ C D : ℝ × ℝ, circle1 C.1 C.2 → circle2 D.1 D.2 → 
  (C ≠ D → (D.2 - C.2) / (D.1 - C.1) = 5 / 2) := 
by
  -- proof to be completed
  sorry

end NUMINAMATH_GPT_slope_CD_l1714_171403


namespace NUMINAMATH_GPT_sum_of_discount_rates_l1714_171458

theorem sum_of_discount_rates : 
  let fox_price := 15
  let pony_price := 20
  let fox_pairs := 3
  let pony_pairs := 2
  let total_savings := 9
  let pony_discount := 18.000000000000014
  let fox_discount := 4
  let total_discount_rate := fox_discount + pony_discount
  total_discount_rate = 22.000000000000014 := by
sorry

end NUMINAMATH_GPT_sum_of_discount_rates_l1714_171458


namespace NUMINAMATH_GPT_find_A_salary_l1714_171461

theorem find_A_salary (A B : ℝ) (h1 : A + B = 2000) (h2 : 0.05 * A = 0.15 * B) : A = 1500 :=
sorry

end NUMINAMATH_GPT_find_A_salary_l1714_171461


namespace NUMINAMATH_GPT_break_even_machines_l1714_171409

def cost_parts : ℤ := 3600
def cost_patent : ℤ := 4500
def selling_price : ℤ := 180

def total_costs : ℤ := cost_parts + cost_patent

def machines_to_break_even : ℤ := total_costs / selling_price

theorem break_even_machines :
  machines_to_break_even = 45 := by
  sorry

end NUMINAMATH_GPT_break_even_machines_l1714_171409


namespace NUMINAMATH_GPT_czechoslovak_inequality_l1714_171459

-- Define the triangle and the points
structure Triangle (α : Type) [LinearOrderedRing α] :=
(A B C : α × α)

variables {α : Type} [LinearOrderedRing α]

-- Define the condition that O is on the segment AB but is not a vertex
def on_segment (O A B : α × α) : Prop :=
  ∃ x : α, 0 < x ∧ x < 1 ∧ O = (A.1 + x * (B.1 - A.1), A.2 + x * (B.2 - A.2))

-- Define the dot product for vectors
def dot (u v: α × α) : α := u.1 * v.1 + u.2 * v.2

-- Main statement
theorem czechoslovak_inequality (T : Triangle α) (O : α × α) (hO : on_segment O T.A T.B) :
  dot O T.C * dot T.A T.B < dot T.A O * dot T.B T.C + dot T.B O * dot T.A T.C :=
sorry

end NUMINAMATH_GPT_czechoslovak_inequality_l1714_171459


namespace NUMINAMATH_GPT_calc_product_eq_243_l1714_171437

theorem calc_product_eq_243 : 
  (1 / 3) * 9 * (1 / 27) * 81 * (1 / 243) * 729 * (1 / 2187) * 6561 * (1 / 19683) * 59049 = 243 :=
by
  sorry

end NUMINAMATH_GPT_calc_product_eq_243_l1714_171437


namespace NUMINAMATH_GPT_man_walking_time_l1714_171475

theorem man_walking_time
  (T : ℕ) -- Let T be the time (in minutes) the man usually arrives at the station.
  (usual_arrival_home : ℕ) -- The time (in minutes) they usually arrive home, which is T + 30.
  (early_arrival : ℕ) (walking_start_time : ℕ) (early_home_arrival : ℕ)
  (usual_arrival_home_eq : usual_arrival_home = T + 30)
  (early_arrival_eq : early_arrival = T - 60)
  (walking_start_time_eq : walking_start_time = early_arrival)
  (early_home_arrival_eq : early_home_arrival = T)
  (time_saved : ℕ) (half_time_walk : ℕ)
  (time_saved_eq : time_saved = 30)
  (half_time_walk_eq : half_time_walk = time_saved / 2) :
  walking_start_time = half_time_walk := by
  sorry

end NUMINAMATH_GPT_man_walking_time_l1714_171475


namespace NUMINAMATH_GPT_range_of_a_l1714_171463

theorem range_of_a (a : ℝ) : (∀ x : ℝ, x^2 + a * x + 4 < 0 → false) → (-4 ≤ a ∧ a ≤ 4) :=
by 
  sorry

end NUMINAMATH_GPT_range_of_a_l1714_171463


namespace NUMINAMATH_GPT_solve_absolute_value_equation_l1714_171417

theorem solve_absolute_value_equation :
  {x : ℝ | 3 * x^2 + 3 * x + 6 = abs (-20 + 5 * x)} = {1.21, -3.87} :=
by
  sorry

end NUMINAMATH_GPT_solve_absolute_value_equation_l1714_171417


namespace NUMINAMATH_GPT_no_spiky_two_digit_numbers_l1714_171451

def is_spiky (n : ℕ) : Prop :=
  ∃ a b : ℕ, a ≠ 0 ∧
             10 ≤ n ∧ n < 100 ∧
             n = 10 * a + b ∧
             n = a + b^3 - 2 * a

theorem no_spiky_two_digit_numbers : ∀ n, 10 ≤ n ∧ n < 100 → ¬ is_spiky n :=
by
  intro n h
  sorry

end NUMINAMATH_GPT_no_spiky_two_digit_numbers_l1714_171451


namespace NUMINAMATH_GPT_find_rate_of_current_l1714_171484

-- Given speed of the boat in still water (km/hr)
def boat_speed : ℤ := 20

-- Given time of travel downstream (hours)
def time_downstream : ℚ := 24 / 60

-- Given distance travelled downstream (km)
def distance_downstream : ℤ := 10

-- To find: rate of the current (km/hr)
theorem find_rate_of_current (c : ℚ) 
  (h1 : distance_downstream = (boat_speed + c) * time_downstream) : 
  c = 5 := 
by sorry

end NUMINAMATH_GPT_find_rate_of_current_l1714_171484


namespace NUMINAMATH_GPT_average_age_of_4_students_l1714_171446

theorem average_age_of_4_students :
  let total_age_15 := 15 * 15
  let age_15th := 25
  let total_age_9 := 16 * 9
  (total_age_15 - total_age_9 - age_15th) / 4 = 14 :=
by
  sorry

end NUMINAMATH_GPT_average_age_of_4_students_l1714_171446


namespace NUMINAMATH_GPT_work_completion_days_l1714_171480

theorem work_completion_days (x : ℕ) (h_ratio : 5 * 18 = 3 * 30) : 30 = 30 :=
by {
    sorry
}

end NUMINAMATH_GPT_work_completion_days_l1714_171480


namespace NUMINAMATH_GPT_sequence_bounded_l1714_171442

open Classical

noncomputable def bounded_sequence (a : ℕ → ℝ) (M : ℝ) :=
  ∀ n : ℕ, n > 0 → a n < M

theorem sequence_bounded {a : ℕ → ℝ} (h0 : 0 ≤ a 1 ∧ a 1 ≤ 2)
  (h : ∀ n : ℕ, n > 0 → a (n + 1) = a n + (a n)^2 / n^3) :
  ∃ M : ℝ, 0 < M ∧ bounded_sequence a M :=
by
  sorry

end NUMINAMATH_GPT_sequence_bounded_l1714_171442


namespace NUMINAMATH_GPT_fifteenth_triangular_number_is_120_l1714_171426

def triangular_number (n : ℕ) : ℕ := (n * (n + 1)) / 2

theorem fifteenth_triangular_number_is_120 : triangular_number 15 = 120 := by
  sorry

end NUMINAMATH_GPT_fifteenth_triangular_number_is_120_l1714_171426


namespace NUMINAMATH_GPT_johny_distance_l1714_171469

noncomputable def distance_south : ℕ := 40
variable (E : ℕ)
noncomputable def distance_east : ℕ := E
noncomputable def distance_north (E : ℕ) : ℕ := 2 * E
noncomputable def total_distance (E : ℕ) : ℕ := distance_south + distance_east E + distance_north E

theorem johny_distance :
  ∀ E : ℕ, total_distance E = 220 → E - distance_south = 20 :=
by
  intro E
  intro h
  rw [total_distance, distance_north, distance_east, distance_south] at h
  sorry

end NUMINAMATH_GPT_johny_distance_l1714_171469


namespace NUMINAMATH_GPT_union_of_A_and_B_l1714_171429

def setA : Set ℝ := { x : ℝ | abs (x - 3) < 2 }
def setB : Set ℝ := { x : ℝ | (x + 1) / (x - 2) ≤ 0 }

theorem union_of_A_and_B : setA ∪ setB = { x : ℝ | -1 ≤ x ∧ x < 5 } :=
by
  sorry

end NUMINAMATH_GPT_union_of_A_and_B_l1714_171429


namespace NUMINAMATH_GPT_area_of_shaded_quadrilateral_l1714_171433

-- The problem setup
variables 
  (triangle : Type) [Nonempty triangle]
  (area : triangle → ℝ)
  (EFA FAB FBD CEDF : triangle)
  (h_EFA : area EFA = 5)
  (h_FAB : area FAB = 9)
  (h_FBD : area FBD = 9)
  (h_partition : ∀ t, t = EFA ∨ t = FAB ∨ t = FBD ∨ t = CEDF)

-- The goal to prove
theorem area_of_shaded_quadrilateral (EFA FAB FBD CEDF : triangle) 
  (h_EFA : area EFA = 5) (h_FAB : area FAB = 9) (h_FBD : area FBD = 9)
  (h_partition : ∀ t, t = EFA ∨ t = FAB ∨ t = FBD ∨ t = CEDF) : 
  area CEDF = 45 :=
by
  sorry

end NUMINAMATH_GPT_area_of_shaded_quadrilateral_l1714_171433


namespace NUMINAMATH_GPT_correct_exponent_operation_l1714_171486

theorem correct_exponent_operation (a b : ℝ) : 
  (a^3 * a^2 ≠ a^6) ∧ 
  (6 * a^6 / (2 * a^2) ≠ 3 * a^3) ∧ 
  ((-a^2)^3 = -a^6) ∧ 
  ((-2 * a * b^2)^2 ≠ 2 * a^2 * b^4) :=
by
  sorry

end NUMINAMATH_GPT_correct_exponent_operation_l1714_171486


namespace NUMINAMATH_GPT_usual_time_to_catch_bus_l1714_171483

theorem usual_time_to_catch_bus (S T : ℝ) (h1 : S / ((5/4) * S) = (T + 5) / T) : T = 25 :=
by sorry

end NUMINAMATH_GPT_usual_time_to_catch_bus_l1714_171483


namespace NUMINAMATH_GPT_min_links_for_weights_l1714_171454

def min_links_to_break (n : ℕ) : ℕ :=
  if n = 60 then 3 else sorry

theorem min_links_for_weights (n : ℕ) (h1 : n = 60) :
  min_links_to_break n = 3 :=
by
  rw [h1]
  trivial

end NUMINAMATH_GPT_min_links_for_weights_l1714_171454


namespace NUMINAMATH_GPT_evaluate_expression_l1714_171453

theorem evaluate_expression : 3^(1^(2^3)) + ((3^1)^2)^2 = 84 := 
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l1714_171453


namespace NUMINAMATH_GPT_remainder_when_divided_by_r_minus_1_l1714_171401

def f (r : Int) : Int := r^14 - r + 5

theorem remainder_when_divided_by_r_minus_1 : f 1 = 5 := by
  sorry

end NUMINAMATH_GPT_remainder_when_divided_by_r_minus_1_l1714_171401


namespace NUMINAMATH_GPT_land_profit_each_son_l1714_171473

theorem land_profit_each_son :
  let hectares : ℝ := 3
  let m2_per_hectare : ℝ := 10000
  let total_sons : ℕ := 8
  let area_per_son := (hectares * m2_per_hectare) / total_sons
  let m2_per_portion : ℝ := 750
  let profit_per_portion : ℝ := 500
  let periods_per_year : ℕ := 12 / 3

  (area_per_son / m2_per_portion * profit_per_portion * periods_per_year = 10000) :=
by
  sorry

end NUMINAMATH_GPT_land_profit_each_son_l1714_171473


namespace NUMINAMATH_GPT_non_honda_red_percentage_l1714_171491

-- Define the conditions
def total_cars : ℕ := 900
def honda_percentage_red : ℝ := 0.90
def total_percentage_red : ℝ := 0.60
def honda_cars : ℕ := 500

-- The statement to prove
theorem non_honda_red_percentage : 
  (0.60 * 900 - 0.90 * 500) / (900 - 500) * 100 = 22.5 := 
  by sorry

end NUMINAMATH_GPT_non_honda_red_percentage_l1714_171491


namespace NUMINAMATH_GPT_boys_total_count_l1714_171440

theorem boys_total_count 
  (avg_age_all: ℤ) (avg_age_first6: ℤ) (avg_age_last6: ℤ)
  (total_first6: ℤ) (total_last6: ℤ) (total_age_all: ℤ) :
  avg_age_all = 50 →
  avg_age_first6 = 49 →
  avg_age_last6 = 52 →
  total_first6 = 6 * avg_age_first6 →
  total_last6 = 6 * avg_age_last6 →
  total_age_all = total_first6 + total_last6 →
  total_age_all = avg_age_all * 13 :=
by
  intros h_avg_all h_avg_first6 h_avg_last6 h_total_first6 h_total_last6 h_total_age_all
  rw [h_avg_all, h_avg_first6, h_avg_last6] at *
  -- Proof steps skipped
  sorry

end NUMINAMATH_GPT_boys_total_count_l1714_171440


namespace NUMINAMATH_GPT_smaller_cuboid_width_l1714_171498

theorem smaller_cuboid_width
  (length_orig width_orig height_orig : ℕ)
  (length_small height_small : ℕ)
  (num_small_cuboids : ℕ)
  (volume_orig : ℕ := length_orig * width_orig * height_orig)
  (volume_small : ℕ := length_small * width_small * height_small)
  (H1 : length_orig = 18)
  (H2 : width_orig = 15)
  (H3 : height_orig = 2)
  (H4 : length_small = 5)
  (H5 : height_small = 3)
  (H6 : num_small_cuboids = 6)
  (H_volume_match : num_small_cuboids * volume_small = volume_orig)
  : width_small = 6 := by
  sorry

end NUMINAMATH_GPT_smaller_cuboid_width_l1714_171498


namespace NUMINAMATH_GPT_sum_of_coeffs_eq_negative_21_l1714_171416

noncomputable def expand_and_sum_coeff (d : ℤ) : ℤ :=
  let expression := -(4 - d) * (d + 2 * (4 - d))
  let expanded_form := -d^2 + 12*d - 32
  let sum_of_coeffs := -1 + 12 - 32
  sum_of_coeffs

theorem sum_of_coeffs_eq_negative_21 (d : ℤ) : expand_and_sum_coeff d = -21 := by
  sorry

end NUMINAMATH_GPT_sum_of_coeffs_eq_negative_21_l1714_171416


namespace NUMINAMATH_GPT_tetrahedron_inscribed_sphere_radius_l1714_171449

theorem tetrahedron_inscribed_sphere_radius (a : ℝ) (r : ℝ) (a_pos : 0 < a) :
  (r = a * (Real.sqrt 6 + 1) / 8) ∨ 
  (r = a * (Real.sqrt 6 - 1) / 8) :=
sorry

end NUMINAMATH_GPT_tetrahedron_inscribed_sphere_radius_l1714_171449


namespace NUMINAMATH_GPT_real_root_range_of_a_l1714_171468

theorem real_root_range_of_a (a : ℝ) : 
  (∃ x : ℝ, x^2 + x + |a - 1/4| + |a| = 0) ↔ (0 ≤ a ∧ a ≤ 1/4) :=
by
  sorry

end NUMINAMATH_GPT_real_root_range_of_a_l1714_171468


namespace NUMINAMATH_GPT_FI_squared_correct_l1714_171478

noncomputable def FI_squared : ℝ :=
  let A : ℝ × ℝ := (0, 0)
  let B : ℝ × ℝ := (4, 0)
  let C : ℝ × ℝ := (4, 4)
  let D : ℝ × ℝ := (0, 4)
  let E : ℝ × ℝ := (3, 0)
  let H : ℝ × ℝ := (0, 1)
  let F : ℝ × ℝ := (4, 1)
  let G : ℝ × ℝ := (1, 4)
  let I : ℝ × ℝ := (3, 0)
  let J : ℝ × ℝ := (0, 1)
  let FI_squared := (4 - 3)^2 + (1 - 0)^2
  FI_squared

theorem FI_squared_correct : FI_squared = 2 :=
by
  sorry

end NUMINAMATH_GPT_FI_squared_correct_l1714_171478


namespace NUMINAMATH_GPT_find_expression_value_l1714_171405

theorem find_expression_value (x : ℝ) : 
  let a := 2015 * x + 2014
  let b := 2015 * x + 2015
  let c := 2015 * x + 2016
  a^2 + b^2 + c^2 - a * b - b * c - c * a = 3 :=
by
  let a := 2015 * x + 2014
  let b := 2015 * x + 2015
  let c := 2015 * x + 2016
  have h : a^2 + b^2 + c^2 - a * b - b * c - c * a = 3 := sorry
  exact h

end NUMINAMATH_GPT_find_expression_value_l1714_171405


namespace NUMINAMATH_GPT_value_of_x_l1714_171407

theorem value_of_x (m n : ℝ) (z x : ℝ) (hz : z ≠ 0) (hx : x = m * (n / z) ^ 3) (hconst : 5 * (16 ^ 3) = m * (n ^ 3)) (hz_const : z = 64) : x = 5 / 64 :=
by
  -- proof omitted
  sorry

end NUMINAMATH_GPT_value_of_x_l1714_171407


namespace NUMINAMATH_GPT_michael_boxes_l1714_171487

theorem michael_boxes (total_blocks boxes_per_box : ℕ) (h1: total_blocks = 16) (h2: boxes_per_box = 2) :
  total_blocks / boxes_per_box = 8 :=
by
  sorry

end NUMINAMATH_GPT_michael_boxes_l1714_171487


namespace NUMINAMATH_GPT_solve_for_r_l1714_171457

theorem solve_for_r (r : ℝ) (h: (r + 9) / (r - 3) = (r - 2) / (r + 5)) : r = -39 / 19 :=
sorry

end NUMINAMATH_GPT_solve_for_r_l1714_171457


namespace NUMINAMATH_GPT_series_solution_eq_l1714_171413

theorem series_solution_eq (x : ℝ) 
  (h : (∃ a : ℕ → ℝ, (∀ n, a n = 1 + 6 * n) ∧ (∑' n, a n * x^n = 100))) :
  x = 23/25 ∨ x = 1/50 :=
sorry

end NUMINAMATH_GPT_series_solution_eq_l1714_171413


namespace NUMINAMATH_GPT_probability_of_break_in_first_50_meters_l1714_171481

theorem probability_of_break_in_first_50_meters (total_length favorable_length : ℝ) 
  (h_total_length : total_length = 320) 
  (h_favorable_length : favorable_length = 50) : 
  (favorable_length / total_length) = 0.15625 := 
sorry

end NUMINAMATH_GPT_probability_of_break_in_first_50_meters_l1714_171481


namespace NUMINAMATH_GPT_range_of_a_l1714_171425

theorem range_of_a (a : ℝ) (h : a > 0) (h1 : ∀ x : ℝ, |a * x - 1| + |a * x - a| ≥ 1) : a ≥ 2 := 
sorry

end NUMINAMATH_GPT_range_of_a_l1714_171425


namespace NUMINAMATH_GPT_coconuts_total_l1714_171455

theorem coconuts_total (B_trips : Nat) (Ba_coconuts_per_trip : Nat) (Br_coconuts_per_trip : Nat) (combined_trips : Nat) (B_totals : B_trips = 12) (Ba_coconuts : Ba_coconuts_per_trip = 4) (Br_coconuts : Br_coconuts_per_trip = 8) : combined_trips * (Ba_coconuts_per_trip + Br_coconuts_per_trip) = 144 := 
by
  simp [B_totals, Ba_coconuts, Br_coconuts]
  sorry

end NUMINAMATH_GPT_coconuts_total_l1714_171455


namespace NUMINAMATH_GPT_max_min_values_in_region_l1714_171434

-- Define the function
def z (x y : ℝ) : ℝ := 4 * x^2 + y^2 - 16 * x - 4 * y + 20

-- Define the region D
def D (x y : ℝ) : Prop := (0 ≤ x) ∧ (x - 2 * y ≤ 0) ∧ (x + y - 6 ≤ 0)

-- Define the proof problem
theorem max_min_values_in_region :
  (∀ (x y : ℝ), D x y → z x y ≥ 0) ∧
  (∀ (x y : ℝ), D x y → z x y ≤ 32) :=
by 
  sorry -- Proof omitted

end NUMINAMATH_GPT_max_min_values_in_region_l1714_171434


namespace NUMINAMATH_GPT_meeting_point_l1714_171418

theorem meeting_point (n : ℕ) (petya_start vasya_start petya_end vasya_end meeting_lamp : ℕ) : 
  n = 100 → petya_start = 1 → vasya_start = 100 → petya_end = 22 → vasya_end = 88 → meeting_lamp = 64 :=
by
  intros h_n h_p_start h_v_start h_p_end h_v_end
  sorry

end NUMINAMATH_GPT_meeting_point_l1714_171418


namespace NUMINAMATH_GPT_total_cost_price_is_584_l1714_171477

-- Define the costs of individual items
def cost_watch : ℕ := 144
def cost_bracelet : ℕ := 250
def cost_necklace : ℕ := 190

-- The proof statement: the total cost price is 584
theorem total_cost_price_is_584 : cost_watch + cost_bracelet + cost_necklace = 584 :=
by
  -- We skip the proof steps here, assuming the above definitions are correct.
  sorry

end NUMINAMATH_GPT_total_cost_price_is_584_l1714_171477


namespace NUMINAMATH_GPT_ryan_spends_7_hours_on_english_l1714_171428

variable (C : ℕ)
variable (E : ℕ)

def hours_spent_on_english (C : ℕ) : ℕ := C + 2

theorem ryan_spends_7_hours_on_english :
  C = 5 → E = hours_spent_on_english C → E = 7 :=
by
  intro hC hE
  rw [hC] at hE
  exact hE

end NUMINAMATH_GPT_ryan_spends_7_hours_on_english_l1714_171428


namespace NUMINAMATH_GPT_find_m_l1714_171489

theorem find_m
  (h1 : ∃ (m : ℝ), ∃ (focus_parabola : ℝ × ℝ), focus_parabola = (0, 1/2)
       ∧ ∃ (focus_ellipse : ℝ × ℝ), focus_ellipse = (0, Real.sqrt (m - 2))
       ∧ focus_parabola = focus_ellipse) :
  ∃ (m : ℝ), m = 9/4 :=
by
  sorry

end NUMINAMATH_GPT_find_m_l1714_171489


namespace NUMINAMATH_GPT_decimal_equivalent_of_one_tenth_squared_l1714_171450

theorem decimal_equivalent_of_one_tenth_squared : 
  (1 / 10 : ℝ)^2 = 0.01 := by
  sorry

end NUMINAMATH_GPT_decimal_equivalent_of_one_tenth_squared_l1714_171450


namespace NUMINAMATH_GPT_absolute_difference_of_integers_l1714_171419

theorem absolute_difference_of_integers (x y : ℤ) (h1 : (x + y) / 2 = 15) (h2 : Int.sqrt (x * y) + 6 = 15) : |x - y| = 24 :=
  sorry

end NUMINAMATH_GPT_absolute_difference_of_integers_l1714_171419


namespace NUMINAMATH_GPT_no_solution_for_k_l1714_171464

theorem no_solution_for_k 
  (a1 a2 a3 a4 : ℝ) 
  (h_pos1 : 0 < a1) (h_pos2 : a1 < a2) 
  (h_pos3 : a2 < a3) (h_pos4 : a3 < a4) 
  (x1 x2 x3 x4 k : ℝ) 
  (h1 : x1 + x2 + x3 + x4 = 1) 
  (h2 : a1 * x1 + a2 * x2 + a3 * x3 + a4 * x4 = k) 
  (h3 : a1^2 * x1 + a2^2 * x2 + a3^2 * x3 + a4^2 * x4 = k^2) 
  (hx1 : 0 ≤ x1) (hx2 : 0 ≤ x2) (hx3 : 0 ≤ x3) (hx4 : 0 ≤ x4) :
  false := 
sorry

end NUMINAMATH_GPT_no_solution_for_k_l1714_171464


namespace NUMINAMATH_GPT_velocity_of_current_l1714_171493

theorem velocity_of_current
  (v c : ℝ) 
  (h1 : 32 = (v + c) * 6) 
  (h2 : 14 = (v - c) * 6) :
  c = 1.5 :=
by
  sorry

end NUMINAMATH_GPT_velocity_of_current_l1714_171493


namespace NUMINAMATH_GPT_right_triangle_shorter_leg_l1714_171415

theorem right_triangle_shorter_leg :
  ∃ (a b : ℤ), a < b ∧ a^2 + b^2 = 65^2 ∧ a = 16 :=
by
  sorry

end NUMINAMATH_GPT_right_triangle_shorter_leg_l1714_171415


namespace NUMINAMATH_GPT_exponential_difference_l1714_171488

theorem exponential_difference (f : ℕ → ℕ) (x : ℕ) (h : f x = 3^x) : f (x + 2) - f x = 8 * f x :=
by sorry

end NUMINAMATH_GPT_exponential_difference_l1714_171488


namespace NUMINAMATH_GPT_total_unique_plants_l1714_171421

noncomputable def bed_A : ℕ := 600
noncomputable def bed_B : ℕ := 550
noncomputable def bed_C : ℕ := 400
noncomputable def bed_D : ℕ := 300

noncomputable def intersection_A_B : ℕ := 75
noncomputable def intersection_A_C : ℕ := 125
noncomputable def intersection_B_D : ℕ := 50
noncomputable def intersection_A_B_C : ℕ := 25

theorem total_unique_plants : 
  bed_A + bed_B + bed_C + bed_D - intersection_A_B - intersection_A_C - intersection_B_D + intersection_A_B_C = 1625 := 
by
  sorry

end NUMINAMATH_GPT_total_unique_plants_l1714_171421


namespace NUMINAMATH_GPT_sum_gcd_lcm_l1714_171485

theorem sum_gcd_lcm (a b : ℕ) (ha : a = 45) (hb : b = 4095) :
    Nat.gcd a b + Nat.lcm a b = 4140 :=
by
  sorry

end NUMINAMATH_GPT_sum_gcd_lcm_l1714_171485


namespace NUMINAMATH_GPT_sandy_gain_percent_is_10_l1714_171462

def total_cost (purchase_price repair_costs : ℕ) := purchase_price + repair_costs

def gain (selling_price total_cost : ℕ) := selling_price - total_cost

def gain_percent (gain total_cost : ℕ) := (gain / total_cost : ℚ) * 100

theorem sandy_gain_percent_is_10 
  (purchase_price : ℕ := 900)
  (repair_costs : ℕ := 300)
  (selling_price : ℕ := 1320) :
  gain_percent (gain selling_price (total_cost purchase_price repair_costs)) 
               (total_cost purchase_price repair_costs) = 10 := 
by
  simp [total_cost, gain, gain_percent]
  sorry

end NUMINAMATH_GPT_sandy_gain_percent_is_10_l1714_171462


namespace NUMINAMATH_GPT_growth_rate_equation_l1714_171414

variable (a x : ℝ)

-- Condition: The number of visitors in March is three times that of January
def visitors_in_march := 3 * a

-- Condition: The average growth rate of visitors in February and March is x
def growth_rate := x

-- Statement to prove
theorem growth_rate_equation 
  (h : (1 + x)^2 = 3) : true :=
by sorry

end NUMINAMATH_GPT_growth_rate_equation_l1714_171414


namespace NUMINAMATH_GPT_find_unknown_number_l1714_171494

theorem find_unknown_number (x : ℤ) :
  (20 + 40 + 60) / 3 = 5 + (20 + 60 + x) / 3 → x = 25 :=
by
  sorry

end NUMINAMATH_GPT_find_unknown_number_l1714_171494


namespace NUMINAMATH_GPT_find_contaminated_constant_l1714_171412

theorem find_contaminated_constant (contaminated_constant : ℝ) (x : ℝ) 
  (h1 : 2 * (x - 3) - contaminated_constant = x + 1) 
  (h2 : x = 9) : contaminated_constant = 2 :=
  sorry

end NUMINAMATH_GPT_find_contaminated_constant_l1714_171412


namespace NUMINAMATH_GPT_largest_int_starting_with_8_l1714_171460

theorem largest_int_starting_with_8 (n : ℕ) : 
  (n / 100 = 8) ∧ (n >= 800) ∧ (n < 900) ∧ ∀ (d : ℕ), (d ∣ n ∧ d ≠ 0 ∧ d ≠ 7) → d ∣ 864 → (n ≤ 864) :=
sorry

end NUMINAMATH_GPT_largest_int_starting_with_8_l1714_171460


namespace NUMINAMATH_GPT_repeating_decimal_to_fraction_l1714_171444

theorem repeating_decimal_to_fraction : (0.3666666 : ℚ) = 11 / 30 :=
by sorry

end NUMINAMATH_GPT_repeating_decimal_to_fraction_l1714_171444


namespace NUMINAMATH_GPT_remainder_when_multiplied_by_three_and_divided_by_eighth_prime_l1714_171436

def first_seven_primes : List ℕ := [2, 3, 5, 7, 11, 13, 17]

def sum_first_seven_primes : ℕ := first_seven_primes.sum

def eighth_prime : ℕ := 19

theorem remainder_when_multiplied_by_three_and_divided_by_eighth_prime :
  ((sum_first_seven_primes * 3) % eighth_prime = 3) :=
by
  sorry

end NUMINAMATH_GPT_remainder_when_multiplied_by_three_and_divided_by_eighth_prime_l1714_171436


namespace NUMINAMATH_GPT_find_k_l1714_171404

noncomputable def f (x : ℝ) : ℝ := 6 * x^2 + 4 * x - (1 / x) + 2

noncomputable def g (x : ℝ) (k : ℝ) : ℝ := x^2 + 3 * x - k

theorem find_k (k : ℝ) : 
  f 3 - g 3 k = 5 → 
  k = - 134 / 3 :=
by
  sorry

end NUMINAMATH_GPT_find_k_l1714_171404


namespace NUMINAMATH_GPT_train_speed_l1714_171499

-- Definition for the given conditions
def distance : ℕ := 240 -- distance in meters
def time_seconds : ℕ := 6 -- time in seconds
def conversion_factor : ℕ := 3600 -- seconds to hour conversion factor
def meters_in_km : ℕ := 1000 -- meters to kilometers conversion factor

-- The proof goal
theorem train_speed (d : ℕ) (t : ℕ) (cf : ℕ) (mk : ℕ) (h1 : d = distance) (h2 : t = time_seconds) (h3 : cf = conversion_factor) (h4 : mk = meters_in_km) :
  (d * cf / t) / mk = 144 :=
by sorry

end NUMINAMATH_GPT_train_speed_l1714_171499


namespace NUMINAMATH_GPT_ashok_borrowed_l1714_171482

theorem ashok_borrowed (P : ℝ) (h : 11400 = P * (6 / 100 * 2 + 9 / 100 * 3 + 14 / 100 * 4)) : P = 12000 :=
by
  sorry

end NUMINAMATH_GPT_ashok_borrowed_l1714_171482


namespace NUMINAMATH_GPT_smallest_value_of_y1_y2_y3_sum_l1714_171400

noncomputable def y_problem := 
  ∃ (y1 y2 y3 : ℝ), 
  (y1 + 3 * y2 + 5 * y3 = 120) 
  ∧ (y1^2 + y2^2 + y3^2 = 720 / 7)

theorem smallest_value_of_y1_y2_y3_sum :
  (∃ (y1 y2 y3 : ℝ), 0 < y1 ∧ 0 < y2 ∧ 0 < y3 ∧ (y1 + 3 * y2 + 5 * y3 = 120) 
  ∧ (y1^2 + y2^2 + y3^2 = 720 / 7)) :=
by 
  sorry

end NUMINAMATH_GPT_smallest_value_of_y1_y2_y3_sum_l1714_171400


namespace NUMINAMATH_GPT_right_triangle_sides_l1714_171424

theorem right_triangle_sides (a b c : ℝ) (h : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a + b + c = 60 → h = 12 → a^2 + b^2 = c^2 → a * b = 12 * c → 
  (a = 15 ∧ b = 20 ∧ c = 25) ∨ (a = 20 ∧ b = 15 ∧ c = 25) :=
by sorry

end NUMINAMATH_GPT_right_triangle_sides_l1714_171424


namespace NUMINAMATH_GPT_graph_passes_through_fixed_point_l1714_171495

noncomputable def fixed_point (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) : Prop :=
  (∀ x y : ℝ, y = a * x + 2 → (x, y) = (-1, 2))

theorem graph_passes_through_fixed_point (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) : fixed_point a h1 h2 :=
sorry

end NUMINAMATH_GPT_graph_passes_through_fixed_point_l1714_171495


namespace NUMINAMATH_GPT_abs_neg_eight_plus_three_pow_zero_eq_nine_l1714_171438

theorem abs_neg_eight_plus_three_pow_zero_eq_nine :
  |-8| + 3^0 = 9 :=
by
  sorry

end NUMINAMATH_GPT_abs_neg_eight_plus_three_pow_zero_eq_nine_l1714_171438


namespace NUMINAMATH_GPT_geometric_increasing_condition_l1714_171490

structure GeometricSequence (a₁ q : ℝ) (a : ℕ → ℝ) :=
  (rec_rel : ∀ n : ℕ, a (n + 1) = a n * q)

def is_increasing (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a n < a (n + 1)

theorem geometric_increasing_condition (a₁ q : ℝ) (a : ℕ → ℝ) (h : GeometricSequence a₁ q a) :
  ¬ (q > 1 ↔ is_increasing a) := sorry

end NUMINAMATH_GPT_geometric_increasing_condition_l1714_171490


namespace NUMINAMATH_GPT_work_done_by_A_alone_l1714_171479

theorem work_done_by_A_alone (Wb : ℝ) (Wa : ℝ) (D : ℝ) :
  Wa = 3 * Wb →
  (Wb + Wa) * 18 = D →
  D = 72 → 
  (D / Wa) = 24 := 
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_work_done_by_A_alone_l1714_171479


namespace NUMINAMATH_GPT_marathon_y_distance_l1714_171422

theorem marathon_y_distance (miles_per_marathon : ℕ) (yards_per_marathon : ℕ) (yards_per_mile : ℕ) (num_marathons : ℕ) (total_yards : ℕ) (y : ℕ) 
  (H1 : miles_per_marathon = 26) 
  (H2 : yards_per_marathon = 312) 
  (H3 : yards_per_mile = 1760) 
  (H4 : num_marathons = 8) 
  (H5 : total_yards = num_marathons * yards_per_marathon) 
  (H6 : total_yards % yards_per_mile = y) 
  (H7 : 0 ≤ y) 
  (H8 : y < yards_per_mile) : 
  y = 736 :=
by 
  sorry

end NUMINAMATH_GPT_marathon_y_distance_l1714_171422


namespace NUMINAMATH_GPT_total_floor_area_is_correct_l1714_171467

-- Define the combined area of the three rugs
def combined_area_of_rugs : ℕ := 212

-- Define the area covered by exactly two layers of rug
def area_covered_by_two_layers : ℕ := 24

-- Define the area covered by exactly three layers of rug
def area_covered_by_three_layers : ℕ := 24

-- Define the total floor area covered by the rugs
def total_floor_area_covered : ℕ :=
  combined_area_of_rugs - area_covered_by_two_layers - 2 * area_covered_by_three_layers

-- The theorem stating the total floor area covered
theorem total_floor_area_is_correct : total_floor_area_covered = 140 := by
  sorry

end NUMINAMATH_GPT_total_floor_area_is_correct_l1714_171467


namespace NUMINAMATH_GPT_carnations_third_bouquet_l1714_171431

theorem carnations_third_bouquet (bouquet1 bouquet2 bouquet3 : ℕ) 
  (h1 : bouquet1 = 9) (h2 : bouquet2 = 14) 
  (h3 : (bouquet1 + bouquet2 + bouquet3) / 3 = 12) : bouquet3 = 13 :=
by
  sorry

end NUMINAMATH_GPT_carnations_third_bouquet_l1714_171431


namespace NUMINAMATH_GPT_not_divisible_by_44_l1714_171456

theorem not_divisible_by_44 (k : ℤ) (n : ℤ) (h1 : n = k * (k + 1) * (k + 2)) (h2 : 11 ∣ n) : ¬ (44 ∣ n) :=
sorry

end NUMINAMATH_GPT_not_divisible_by_44_l1714_171456


namespace NUMINAMATH_GPT_xy_product_given_conditions_l1714_171432

variable (x y : ℝ)

theorem xy_product_given_conditions (hx : x - y = 5) (hx3 : x^3 - y^3 = 35) : x * y = -6 :=
by
  sorry

end NUMINAMATH_GPT_xy_product_given_conditions_l1714_171432


namespace NUMINAMATH_GPT_lcm_36_225_l1714_171402

theorem lcm_36_225 : Nat.lcm 36 225 = 900 := by
  -- Defining the factorizations as given
  let fact_36 : 36 = 2^2 * 3^2 := by rfl
  let fact_225 : 225 = 3^2 * 5^2 := by rfl

  -- Indicating what LCM we need to prove
  show Nat.lcm 36 225 = 900

  -- Proof (skipped)
  sorry

end NUMINAMATH_GPT_lcm_36_225_l1714_171402


namespace NUMINAMATH_GPT_find_m_l1714_171497

theorem find_m 
  (m : ℕ) 
  (hm_pos : 0 < m) 
  (h1 : Nat.lcm 30 m = 90) 
  (h2 : Nat.lcm m 45 = 180) : 
  m = 36 := 
sorry

end NUMINAMATH_GPT_find_m_l1714_171497


namespace NUMINAMATH_GPT_arithmetic_sequence_term_20_l1714_171466

theorem arithmetic_sequence_term_20
  (a : ℕ := 2)
  (d : ℕ := 4)
  (n : ℕ := 20) :
  a + (n - 1) * d = 78 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_term_20_l1714_171466


namespace NUMINAMATH_GPT_positive_root_condition_negative_root_condition_zero_root_condition_l1714_171448

-- Positive root condition
theorem positive_root_condition {a b : ℝ} (h : a * b < 0) : ∃ x : ℝ, a * x + b = 0 ∧ x > 0 :=
by
  sorry

-- Negative root condition
theorem negative_root_condition {a b : ℝ} (h : a * b > 0) : ∃ x : ℝ, a * x + b = 0 ∧ x < 0 :=
by
  sorry

-- Root equal to zero condition
theorem zero_root_condition {a b : ℝ} (h₁ : b = 0) (h₂ : a ≠ 0) : ∃ x : ℝ, a * x + b = 0 ∧ x = 0 :=
by
  sorry

end NUMINAMATH_GPT_positive_root_condition_negative_root_condition_zero_root_condition_l1714_171448


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_l1714_171472

open Nat

theorem arithmetic_sequence_sum (m n : Nat) (d : ℤ) (a_1 : ℤ)
    (hnm : n ≠ m)
    (hSn : (n * (2 * a_1 + (n - 1) * d) / 2) = n / m)
    (hSm : (m * (2 * a_1 + (m - 1) * d) / 2) = m / n) :
  ((m + n) * (2 * a_1 + (m + n - 1) * d) / 2) > 4 := by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_l1714_171472


namespace NUMINAMATH_GPT_ship_length_is_correct_l1714_171406

-- Define the variables
variables (L E S C : ℝ)

-- Define the given conditions
def condition1 (L E S C : ℝ) : Prop := 320 * E = L + 320 * (S - C)
def condition2 (L E S C : ℝ) : Prop := 80 * E = L - 80 * (S + C)

-- Mathematical statement to be proven
theorem ship_length_is_correct
  (L E S C : ℝ)
  (h1 : condition1 L E S C)
  (h2 : condition2 L E S C) :
  L = 26 * E + (2 / 3) * E :=
sorry

end NUMINAMATH_GPT_ship_length_is_correct_l1714_171406


namespace NUMINAMATH_GPT_problem_xy_minimized_problem_x_y_minimized_l1714_171471

open Real

theorem problem_xy_minimized (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 8 * y - x * y = 0) :
  x = 16 ∧ y = 2 ∧ x * y = 32 := 
sorry

theorem problem_x_y_minimized (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 8 * y - x * y = 0) :
  x = 8 + 2 * sqrt 2 ∧ y = 1 + sqrt 2 ∧ x + y = 9 + 4 * sqrt 2 := 
sorry

end NUMINAMATH_GPT_problem_xy_minimized_problem_x_y_minimized_l1714_171471


namespace NUMINAMATH_GPT_total_cats_handled_last_year_l1714_171411

theorem total_cats_handled_last_year (num_adult_cats : ℕ) (two_thirds_female : ℕ) (seventy_five_percent_litters : ℕ) 
                                     (kittens_per_litter : ℕ) (adopted_returned : ℕ) :
  num_adult_cats = 120 →
  two_thirds_female = (2 * num_adult_cats) / 3 →
  seventy_five_percent_litters = (3 * two_thirds_female) / 4 →
  kittens_per_litter = 3 →
  adopted_returned = 15 →
  num_adult_cats + seventy_five_percent_litters * kittens_per_litter + adopted_returned = 315 := 
by
  intros h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_total_cats_handled_last_year_l1714_171411


namespace NUMINAMATH_GPT_can_adjust_to_357_l1714_171420

structure Ratio (L O V : ℕ) :=
(lemon : ℕ)
(oil : ℕ)
(vinegar : ℕ)

def MixA : Ratio 1 2 3 := ⟨1, 2, 3⟩
def MixB : Ratio 3 4 5 := ⟨3, 4, 5⟩
def TargetC : Ratio 3 5 7 := ⟨3, 5, 7⟩

theorem can_adjust_to_357 (x y : ℕ) (hA : x * MixA.lemon + y * MixB.lemon = 3 * (x + y))
    (hO : x * MixA.oil + y * MixB.oil = 5 * (x + y))
    (hV : x * MixA.vinegar + y * MixB.vinegar = 7 * (x + y)) :
    (∃ a b : ℕ, x = 3 * a ∧ y = 2 * b) :=
sorry

end NUMINAMATH_GPT_can_adjust_to_357_l1714_171420


namespace NUMINAMATH_GPT_isosceles_triangle_perimeter_l1714_171496

theorem isosceles_triangle_perimeter (a b : ℕ)
  (h_eqn : ∀ x : ℕ, (x - 4) * (x - 2) = 0 → x = 4 ∨ x = 2)
  (h_isosceles : ∃ a b : ℕ, (a = 4 ∧ b = 2) ∨ (a = 2 ∧ b = 4) ∨ (a = 4 ∧ b = 4)) :
  a + a + b = 10 :=
by
  sorry

end NUMINAMATH_GPT_isosceles_triangle_perimeter_l1714_171496


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_condition_l1714_171445

-- Definitions of conditions
def condition_p (x : ℝ) := (x - 1) * (x + 2) ≤ 0
def condition_q (x : ℝ) := abs (x + 1) ≤ 1

-- The theorem statement
theorem necessary_but_not_sufficient_condition (x : ℝ) :
  (∀ x, condition_q x → condition_p x) ∧ ¬(∀ x, condition_p x → condition_q x) := 
by
  sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_condition_l1714_171445
