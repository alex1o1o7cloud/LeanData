import Mathlib

namespace NUMINAMATH_GPT_necessary_but_not_sufficient_l105_10523

theorem necessary_but_not_sufficient (x : ℝ) (h : x < 4) : x < 0 ∨ true :=
by
  sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_l105_10523


namespace NUMINAMATH_GPT_second_chapter_pages_l105_10525

/-- A book has 2 chapters across 81 pages. The first chapter is 13 pages long. -/
theorem second_chapter_pages (total_pages : ℕ) (first_chapter_pages : ℕ) (second_chapter_pages : ℕ) : 
  total_pages = 81 → 
  first_chapter_pages = 13 → 
  second_chapter_pages = total_pages - first_chapter_pages → 
  second_chapter_pages = 68 := 
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end NUMINAMATH_GPT_second_chapter_pages_l105_10525


namespace NUMINAMATH_GPT_initial_shed_bales_zero_l105_10588

def bales_in_barn_initial : ℕ := 47
def bales_added_by_benny : ℕ := 35
def bales_in_barn_total : ℕ := 82

theorem initial_shed_bales_zero (b_shed : ℕ) :
  bales_in_barn_initial + bales_added_by_benny = bales_in_barn_total → b_shed = 0 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_initial_shed_bales_zero_l105_10588


namespace NUMINAMATH_GPT_Joshua_share_correct_l105_10553

noncomputable def Joshua_share (J : ℝ) : ℝ :=
  3 * J

noncomputable def Jasmine_share (J : ℝ) : ℝ :=
  J / 2

theorem Joshua_share_correct (J : ℝ) (h : J + 3 * J + J / 2 = 120) :
  Joshua_share J = 80.01 := by
  sorry

end NUMINAMATH_GPT_Joshua_share_correct_l105_10553


namespace NUMINAMATH_GPT_compare_negatives_l105_10570

theorem compare_negatives : -2 < -3 / 2 :=
by sorry

end NUMINAMATH_GPT_compare_negatives_l105_10570


namespace NUMINAMATH_GPT_double_sum_evaluation_l105_10535

theorem double_sum_evaluation :
  ∑' m:ℕ, ∑' n:ℕ, (if m > 0 ∧ n > 0 then 1 / (m * n * (m + n + 2)) else 0) = -Real.pi^2 / 6 :=
sorry

end NUMINAMATH_GPT_double_sum_evaluation_l105_10535


namespace NUMINAMATH_GPT_hall_volume_l105_10520

theorem hall_volume (length breadth : ℝ) (h : ℝ)
  (h_length : length = 15) (h_breadth : breadth = 12)
  (h_area : 2 * (length * breadth) = 2 * (breadth * h) + 2 * (length * h)) :
  length * breadth * h = 8004 := 
by
  -- Proof not required
  sorry

end NUMINAMATH_GPT_hall_volume_l105_10520


namespace NUMINAMATH_GPT_melanie_missed_games_l105_10566

-- Define the total number of games and the number of games attended by Melanie
def total_games : ℕ := 7
def games_attended : ℕ := 3

-- Define the number of games missed as total games minus games attended
def games_missed : ℕ := total_games - games_attended

-- Theorem stating the number of games missed by Melanie
theorem melanie_missed_games : games_missed = 4 := by
  -- The proof is omitted
  sorry

end NUMINAMATH_GPT_melanie_missed_games_l105_10566


namespace NUMINAMATH_GPT_find_height_of_tank_A_l105_10573

noncomputable def height_of_tank_A (C_A C_B h_B ratio V_ratio : ℝ) : ℝ :=
  let r_A := C_A / (2 * Real.pi)
  let r_B := C_B / (2 * Real.pi)
  let V_A := Real.pi * (r_A ^ 2) * ratio
  let V_B := Real.pi * (r_B ^ 2) * h_B
  (V_ratio * V_B) / (Real.pi * (r_A ^ 2))

theorem find_height_of_tank_A :
  height_of_tank_A 8 10 8 10 0.8000000000000001 = 10 :=
by
  sorry

end NUMINAMATH_GPT_find_height_of_tank_A_l105_10573


namespace NUMINAMATH_GPT_initial_men_count_l105_10550

theorem initial_men_count (M : ℕ) (P : ℕ) :
  P = M * 20 →
  P = (M + 650) * 109 / 9 →
  M = 1000 :=
by
  sorry

end NUMINAMATH_GPT_initial_men_count_l105_10550


namespace NUMINAMATH_GPT_find_m_l105_10510

theorem find_m (m : ℝ) :
  (∀ x : ℝ, 0 < x → (m^2 - m - 1) * x < 0) → m = -1 :=
by sorry

end NUMINAMATH_GPT_find_m_l105_10510


namespace NUMINAMATH_GPT_range_of_a_l105_10597

noncomputable def f (x a : ℝ) : ℝ := x^2 - 2 * a * x + 1

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, x ≥ -1 → f x a ≥ a) ↔ -2 ≤ a ∧ a ≤ (-1 + Real.sqrt 5) / 2 := 
sorry

end NUMINAMATH_GPT_range_of_a_l105_10597


namespace NUMINAMATH_GPT_factorization_correct_l105_10554

theorem factorization_correct (x : ℝ) : 
  x^4 - 5*x^2 - 36 = (x^2 + 4)*(x + 3)*(x - 3) :=
sorry

end NUMINAMATH_GPT_factorization_correct_l105_10554


namespace NUMINAMATH_GPT_max_value_frac_sixth_roots_eq_two_l105_10505

noncomputable def max_value_frac_sixth_roots (α β : ℝ) (t : ℝ) (q : ℝ) : ℝ :=
  if α + β = t ∧ α^2 + β^2 = t ∧ α^3 + β^3 = t ∧ α^4 + β^4 = t ∧ α^5 + β^5 = t then
    max (1 / α^6 + 1 / β^6) 2
  else
    0

theorem max_value_frac_sixth_roots_eq_two (α β : ℝ) (t : ℝ) (q : ℝ) :
  (α + β = t ∧ α^2 + β^2 = t ∧ α^3 + β^3 = t ∧ α^4 + β^4 = t ∧ α^5 + β^5 = t) →
  ∃ m, max_value_frac_sixth_roots α β t q = m ∧ m = 2 :=
sorry

end NUMINAMATH_GPT_max_value_frac_sixth_roots_eq_two_l105_10505


namespace NUMINAMATH_GPT_correct_intersection_l105_10504

def M : Set ℕ := {1, 2, 3}
def N : Set ℕ := {2, 3, 4}

theorem correct_intersection : M ∩ N = {2, 3} := by sorry

end NUMINAMATH_GPT_correct_intersection_l105_10504


namespace NUMINAMATH_GPT_sequence_exists_l105_10533

theorem sequence_exists
  {a_0 b_0 c_0 a b c : ℤ}
  (gcd1 : Int.gcd (Int.gcd a_0 b_0) c_0 = 1)
  (gcd2 : Int.gcd (Int.gcd a b) c = 1) :
  ∃ (n : ℕ) (a_seq b_seq c_seq : Fin (n + 1) → ℤ),
    a_seq 0 = a_0 ∧ b_seq 0 = b_0 ∧ c_seq 0 = c_0 ∧ 
    a_seq n = a ∧ b_seq n = b ∧ c_seq n = c ∧
    ∀ (i : Fin n), (a_seq i) * (a_seq i.succ) + (b_seq i) * (b_seq i.succ) + (c_seq i) * (c_seq i.succ) = 1 :=
sorry

end NUMINAMATH_GPT_sequence_exists_l105_10533


namespace NUMINAMATH_GPT_fraction_subtraction_l105_10559

theorem fraction_subtraction :
  ((2 + 4 + 6 : ℚ) / (1 + 3 + 5) - (1 + 3 + 5) / (2 + 4 + 6) = 7 / 12) :=
by
  sorry

end NUMINAMATH_GPT_fraction_subtraction_l105_10559


namespace NUMINAMATH_GPT_problem_1_a_problem_1_b_problem_2_l105_10527

def set_A : Set ℝ := {x | 3 ≤ x ∧ x < 6}
def set_B : Set ℝ := {x | 2 < x ∧ x < 9}
def complement_B : Set ℝ := {x | x ≤ 2 ∨ x ≥ 9}
def set_union (s₁ s₂ : Set ℝ) := {x | x ∈ s₁ ∨ x ∈ s₂}
def set_inter (s₁ s₂ : Set ℝ) := {x | x ∈ s₁ ∧ x ∈ s₂}

theorem problem_1_a :
  set_inter set_A set_B = {x : ℝ | 3 ≤ x ∧ x < 6} :=
sorry

theorem problem_1_b :
  set_union complement_B set_A = {x : ℝ | x ≤ 2 ∨ (3 ≤ x ∧ x < 6) ∨ x ≥ 9} :=
sorry

def set_C (a : ℝ) : Set ℝ := {x | a < x ∧ x < a + 1}

theorem problem_2 (a : ℝ) :
  (set_C a ⊆ set_B) → (2 ≤ a ∧ a ≤ 8) :=
sorry

end NUMINAMATH_GPT_problem_1_a_problem_1_b_problem_2_l105_10527


namespace NUMINAMATH_GPT_range_of_x_l105_10545

noncomputable def T (x : ℝ) : ℝ := |(2 * x - 1)|

theorem range_of_x (x : ℝ) (h : ∀ a : ℝ, T x ≥ |1 + a| - |2 - a|) : 
  x ≤ -1 ∨ 2 ≤ x :=
by
  sorry

end NUMINAMATH_GPT_range_of_x_l105_10545


namespace NUMINAMATH_GPT_find_all_quartets_l105_10552

def is_valid_quartet (a b c d : ℕ) : Prop :=
  a + b = c * d ∧
  a * b = c + d ∧
  a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ a ≠ c ∧ a ≠ d ∧ b ≠ d

theorem find_all_quartets :
  ∀ (a b c d : ℕ),
  is_valid_quartet a b c d ↔
  (a, b, c, d) = (1, 5, 3, 2) ∨ 
  (a, b, c, d) = (1, 5, 2, 3) ∨ 
  (a, b, c, d) = (5, 1, 3, 2) ∨
  (a, b, c, d) = (5, 1, 2, 3) ∨ 
  (a, b, c, d) = (2, 3, 1, 5) ∨ 
  (a, b, c, d) = (3, 2, 1, 5) ∨ 
  (a, b, c, d) = (2, 3, 5, 1) ∨ 
  (a, b, c, d) = (3, 2, 5, 1) := by
  sorry

end NUMINAMATH_GPT_find_all_quartets_l105_10552


namespace NUMINAMATH_GPT_operation_value_l105_10543

variable (a b : ℤ)

theorem operation_value (h : (21 - 1) * (9 - 1) = 160) : a = 21 :=
by
  sorry

end NUMINAMATH_GPT_operation_value_l105_10543


namespace NUMINAMATH_GPT_complement_intersection_l105_10572

open Set

def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7}
def A : Set ℕ := {2, 4, 5, 7}
def B : Set ℕ := {3, 4, 5}

theorem complement_intersection :
  (U \ A) ∩ (U \ B) = {1, 6} :=
by
  sorry

end NUMINAMATH_GPT_complement_intersection_l105_10572


namespace NUMINAMATH_GPT_average_age_in_club_l105_10594

theorem average_age_in_club :
  let women_avg_age := 32
  let men_avg_age := 38
  let children_avg_age := 10
  let women_count := 12
  let men_count := 18
  let children_count := 10
  let total_ages := (women_avg_age * women_count) + (men_avg_age * men_count) + (children_avg_age * children_count)
  let total_people := women_count + men_count + children_count
  let overall_avg_age := (total_ages : ℝ) / (total_people : ℝ)
  overall_avg_age = 29.2 := by
  sorry

end NUMINAMATH_GPT_average_age_in_club_l105_10594


namespace NUMINAMATH_GPT_average_mileage_is_correct_l105_10509

noncomputable def total_distance : ℝ := 150 + 200
noncomputable def sedan_efficiency : ℝ := 25
noncomputable def truck_efficiency : ℝ := 15
noncomputable def sedan_miles : ℝ := 150
noncomputable def truck_miles : ℝ := 200

noncomputable def total_gas_used : ℝ := (sedan_miles / sedan_efficiency) + (truck_miles / truck_efficiency)
noncomputable def average_gas_mileage : ℝ := total_distance / total_gas_used

theorem average_mileage_is_correct :
  average_gas_mileage = 18.1 := 
by
  sorry

end NUMINAMATH_GPT_average_mileage_is_correct_l105_10509


namespace NUMINAMATH_GPT_perpendicular_exists_l105_10584

-- Definitions for geometric entities involved

structure Point where
  x : ℝ
  y : ℝ

structure Line where
  p1 : Point
  p2 : Point

structure Circle where
  center : Point
  radius : ℝ

-- Definitions for conditions in the problem

-- Condition 1: Point C is not on the circle
def point_not_on_circle (C : Point) (circle : Circle) : Prop :=
  (C.x - circle.center.x)^2 + (C.y - circle.center.y)^2 ≠ circle.radius^2

-- Condition 2: Point C is on the circle
def point_on_circle (C : Point) (circle : Circle) : Prop :=
  (C.x - circle.center.x)^2 + (C.y - circle.center.y)^2 = circle.radius^2

-- Definitions for lines and perpendicularity
def is_perpendicular (line1 : Line) (line2 : Line) : Prop :=
  (line1.p1.x - line1.p2.x) * (line2.p1.x - line2.p2.x) +
  (line1.p1.y - line1.p2.y) * (line2.p1.y - line2.p2.y) = 0

noncomputable def perpendicular_from_point_to_line (C : Point) (line : Line) (circle : Circle) 
  (h₁ : point_not_on_circle C circle ∨ point_on_circle C circle) : Line := 
  sorry

-- The Lean statement for part (a) and (b) combined into one proof.
theorem perpendicular_exists (C : Point) (lineAB : Line) (circle : Circle) 
  (h₁ : point_not_on_circle C circle ∨ point_on_circle C circle) : 
  ∃ (line_perpendicular : Line), is_perpendicular line_perpendicular lineAB ∧ 
  (line_perpendicular.p1 = C ∨ line_perpendicular.p2 = C) :=
  sorry

end NUMINAMATH_GPT_perpendicular_exists_l105_10584


namespace NUMINAMATH_GPT_negation_of_every_square_positive_l105_10502

theorem negation_of_every_square_positive :
  ¬(∀ n : ℕ, n^2 > 0) ↔ ∃ n : ℕ, n^2 ≤ 0 := sorry

end NUMINAMATH_GPT_negation_of_every_square_positive_l105_10502


namespace NUMINAMATH_GPT_find_ab_l105_10507

theorem find_ab
  (a b c : ℝ)
  (h1 : a - b = 3)
  (h2 : a^2 + b^2 = 27)
  (h3 : a + b + c = 10)
  (h4 : a^3 - b^3 = 36)
  : a * b = -15 :=
by
  sorry

end NUMINAMATH_GPT_find_ab_l105_10507


namespace NUMINAMATH_GPT_triangle_area_correct_l105_10599

/-- Define the points of the triangle -/
def x1 : ℝ := -4
def y1 : ℝ := 2
def x2 : ℝ := 2
def y2 : ℝ := 8
def x3 : ℝ := -2
def y3 : ℝ := -2

/-- Define the area calculation function -/
def triangle_area (x1 y1 x2 y2 x3 y3 : ℝ) : ℝ :=
  0.5 * (abs (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)))

/-- Define the area of the given triangle -/
def given_triangle_area : ℝ :=
  triangle_area x1 y1 x2 y2 x3 y3

/-- The goal is to prove that the area of the given triangle is 22 square units -/
theorem triangle_area_correct : given_triangle_area = 22 := by
  sorry

end NUMINAMATH_GPT_triangle_area_correct_l105_10599


namespace NUMINAMATH_GPT_servings_of_peanut_butter_l105_10516

theorem servings_of_peanut_butter :
  let peanutButterInJar := 37 + 4 / 5
  let oneServing := 1 + 1 / 2
  let servings := 25 + 1 / 5
  (peanutButterInJar / oneServing) = servings :=
by
  let peanutButterInJar := 37 + 4 / 5
  let oneServing := 1 + 1 / 2
  let servings := 25 + 1 / 5
  sorry

end NUMINAMATH_GPT_servings_of_peanut_butter_l105_10516


namespace NUMINAMATH_GPT_hyperbola_focal_length_l105_10590

theorem hyperbola_focal_length (x y : ℝ) : 
  (∃ h : x^2 / 9 - y^2 / 4 = 1, 
   ∀ a b : ℝ, a^2 = 9 → b^2 = 4 → 2 * Real.sqrt (a^2 + b^2) = 2 * Real.sqrt 13) :=
by sorry

end NUMINAMATH_GPT_hyperbola_focal_length_l105_10590


namespace NUMINAMATH_GPT_inequality_2n_squared_plus_3n_plus_1_l105_10521

theorem inequality_2n_squared_plus_3n_plus_1 (n : ℕ) (h: n > 0) : (2 * n^2 + 3 * n + 1)^n ≥ 6^n * (n! * n!) := 
by sorry

end NUMINAMATH_GPT_inequality_2n_squared_plus_3n_plus_1_l105_10521


namespace NUMINAMATH_GPT_field_length_l105_10540

-- Definitions of the conditions
def pond_area : ℝ := 25  -- area of the square pond
def width_to_length_ratio (w l : ℝ) : Prop := l = 2 * w  -- length is double the width
def pond_to_field_ratio (pond_area field_area : ℝ) : Prop := pond_area = (1/8) * field_area  -- pond area is 1/8 of field area

-- Statement to prove
theorem field_length (w l : ℝ) (h1 : width_to_length_ratio w l) (h2 : pond_to_field_ratio pond_area (l * w)) : l = 20 :=
by sorry

end NUMINAMATH_GPT_field_length_l105_10540


namespace NUMINAMATH_GPT_num_distinct_pos_factors_81_l105_10576

-- Define what it means to be a factor
def is_factor (n d : ℕ) : Prop := d ∣ n

-- 81 is 3^4 by definition
def eighty_one : ℕ := 3 ^ 4

-- Define the list of positive factors of 81
def pos_factors_81 : List ℕ := [1, 3, 9, 27, 81]

-- Formal statement that 81 has 5 distinct positive factors
theorem num_distinct_pos_factors_81 : (pos_factors_81.length = 5) :=
by sorry

end NUMINAMATH_GPT_num_distinct_pos_factors_81_l105_10576


namespace NUMINAMATH_GPT_tyrone_gives_non_integer_marbles_to_eric_l105_10582

theorem tyrone_gives_non_integer_marbles_to_eric
  (T_init : ℕ) (E_init : ℕ) (x : ℚ)
  (hT : T_init = 120) (hE : E_init = 18)
  (h_eq : T_init - x = 3 * (E_init + x)) :
  ¬ (∃ n : ℕ, x = n) :=
by
  sorry

end NUMINAMATH_GPT_tyrone_gives_non_integer_marbles_to_eric_l105_10582


namespace NUMINAMATH_GPT_value_of_y_l105_10571

theorem value_of_y (x y : ℤ) (h1 : x - y = 6) (h2 : x + y = 12) : y = 3 := 
by
  sorry

end NUMINAMATH_GPT_value_of_y_l105_10571


namespace NUMINAMATH_GPT_find_a_l105_10528

noncomputable def f (x : ℝ) (a : ℝ) := (2 / x) - 2 + 2 * a * Real.log x

theorem find_a (a : ℝ) (h : ∃ x ∈ Set.Icc (1/2 : ℝ) 2, f x a = 0) : a = 1 := by
  sorry

end NUMINAMATH_GPT_find_a_l105_10528


namespace NUMINAMATH_GPT_micah_total_envelopes_l105_10593

-- Define the conditions as hypotheses
def weight_threshold := 5
def stamps_for_heavy := 5
def stamps_for_light := 2
def total_stamps := 52
def light_envelopes := 6

-- Noncomputable because we are using abstract reasoning rather than computational functions
noncomputable def total_envelopes : ℕ :=
  light_envelopes + (total_stamps - light_envelopes * stamps_for_light) / stamps_for_heavy

-- The theorem to prove
theorem micah_total_envelopes : total_envelopes = 14 := by
  sorry

end NUMINAMATH_GPT_micah_total_envelopes_l105_10593


namespace NUMINAMATH_GPT_lines_parallel_l105_10522

theorem lines_parallel 
  (a : ℝ) (b : ℝ) (c : ℝ)
  (α : ℝ) (β : ℝ) (γ : ℝ)
  (h1 : Real.log (Real.sin α) + Real.log (Real.sin γ) = 2 * Real.log (Real.sin β)) :
  (∀ x y : ℝ, ∀ a b c : ℝ, 
    (x * (Real.sin α)^2 + y * Real.sin α = a) → 
    (x * (Real.sin β)^2 + y * Real.sin γ = c) →
    (-Real.sin α = -((Real.sin β)^2 / Real.sin γ))) :=
sorry

end NUMINAMATH_GPT_lines_parallel_l105_10522


namespace NUMINAMATH_GPT_julieta_total_spent_l105_10561

def original_price_backpack : ℕ := 50
def original_price_ring_binder : ℕ := 20
def quantity_ring_binders : ℕ := 3
def price_increase_backpack : ℕ := 5
def price_decrease_ring_binder : ℕ := 2

def total_spent (original_price_backpack original_price_ring_binder quantity_ring_binders price_increase_backpack price_decrease_ring_binder : ℕ) : ℕ :=
  let new_price_backpack := original_price_backpack + price_increase_backpack
  let new_price_ring_binder := original_price_ring_binder - price_decrease_ring_binder
  new_price_backpack + (new_price_ring_binder * quantity_ring_binders)

theorem julieta_total_spent :
  total_spent original_price_backpack original_price_ring_binder quantity_ring_binders price_increase_backpack price_decrease_ring_binder = 109 :=
by 
  -- Proof steps are omitted intentionally
  sorry

end NUMINAMATH_GPT_julieta_total_spent_l105_10561


namespace NUMINAMATH_GPT_evaporation_rate_is_200_ml_per_hour_l105_10557

-- Definitions based on the given conditions
def faucet_drip_rate : ℕ := 40 -- ml per minute
def running_time : ℕ := 9 -- hours
def dumped_water : ℕ := 12000 -- ml (converted from liters)
def water_left : ℕ := 7800 -- ml

-- Alias for total water dripped in running_time
noncomputable def total_dripped_water : ℕ := faucet_drip_rate * 60 * running_time

-- Total water that should have been in the bathtub without evaporation
noncomputable def total_without_evaporation : ℕ := total_dripped_water - dumped_water

-- Water evaporated
noncomputable def evaporated_water : ℕ := total_without_evaporation - water_left

-- Evaporation rate in ml/hour
noncomputable def evaporation_rate : ℕ := evaporated_water / running_time

-- The goal theorem statement
theorem evaporation_rate_is_200_ml_per_hour : evaporation_rate = 200 := by
  -- proof here
  sorry

end NUMINAMATH_GPT_evaporation_rate_is_200_ml_per_hour_l105_10557


namespace NUMINAMATH_GPT_equivalent_knicks_l105_10512

theorem equivalent_knicks (knicks knacks knocks : ℕ) (h1 : 5 * knicks = 3 * knacks) (h2 : 4 * knacks = 6 * knocks) :
  36 * knocks = 40 * knicks :=
by
  sorry

end NUMINAMATH_GPT_equivalent_knicks_l105_10512


namespace NUMINAMATH_GPT_time_for_A_to_complete_race_l105_10538

theorem time_for_A_to_complete_race
  (V_A V_B : ℝ) (T_A : ℝ)
  (h1 : V_B = 975 / T_A) (h2 : V_B = 2.5) :
  T_A = 390 :=
by
  sorry

end NUMINAMATH_GPT_time_for_A_to_complete_race_l105_10538


namespace NUMINAMATH_GPT_not_p_and_p_or_q_implies_q_l105_10567

theorem not_p_and_p_or_q_implies_q (p q : Prop) (h1 : ¬ p) (h2 : p ∨ q) : q :=
by
  have h3 : p := sorry
  have h4 : false := sorry
  exact sorry

end NUMINAMATH_GPT_not_p_and_p_or_q_implies_q_l105_10567


namespace NUMINAMATH_GPT_committee_count_is_252_l105_10501

/-- Definition of binomial coefficient -/
def binom (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

/-- Problem statement: Number of ways to choose a 5-person committee from a club of 10 people is 252 -/
theorem committee_count_is_252 : binom 10 5 = 252 :=
by
  sorry

end NUMINAMATH_GPT_committee_count_is_252_l105_10501


namespace NUMINAMATH_GPT_cos_triple_angle_l105_10526

theorem cos_triple_angle (θ : ℝ) (h : Real.cos θ = 1 / 3) : Real.cos (3 * θ) = -23 / 27 := by
  sorry

end NUMINAMATH_GPT_cos_triple_angle_l105_10526


namespace NUMINAMATH_GPT_sufficient_not_necessary_condition_l105_10568

theorem sufficient_not_necessary_condition {x : ℝ} (h : 1 < x ∧ x < 2) : x < 2 ∧ ¬(∀ x, x < 2 → (1 < x ∧ x < 2)) :=
by
  sorry

end NUMINAMATH_GPT_sufficient_not_necessary_condition_l105_10568


namespace NUMINAMATH_GPT_find_two_digit_number_l105_10577

def is_positive (n : ℕ) := n > 0
def is_even (n : ℕ) := n % 2 = 0
def is_multiple_of_9 (n : ℕ) := n % 9 = 0
def product_of_digits_is_square (n : ℕ) : Prop :=
  let tens := n / 10
  let units := n % 10
  ∃ k : ℕ, (tens * units) = k * k

theorem find_two_digit_number (N : ℕ) 
  (h_pos : is_positive N) 
  (h_ev : is_even N) 
  (h_mult_9 : is_multiple_of_9 N)
  (h_prod_square : product_of_digits_is_square N) 
: N = 90 := by 
  sorry

end NUMINAMATH_GPT_find_two_digit_number_l105_10577


namespace NUMINAMATH_GPT_train_length_l105_10574

theorem train_length (speed : ℝ) (time : ℝ) (bridge_length : ℝ) (total_distance : ℝ) (train_length : ℝ) 
  (h1 : speed = 48) (h2 : time = 45) (h3 : bridge_length = 300)
  (h4 : total_distance = speed * time) (h5 : train_length = total_distance - bridge_length) : 
  train_length = 1860 :=
sorry

end NUMINAMATH_GPT_train_length_l105_10574


namespace NUMINAMATH_GPT_range_of_z_l105_10586

theorem range_of_z (x y : ℝ) (h : x^2 / 16 + y^2 / 9 = 1) : -5 ≤ x + y ∧ x + y ≤ 5 :=
sorry

end NUMINAMATH_GPT_range_of_z_l105_10586


namespace NUMINAMATH_GPT_find_second_term_l105_10595

theorem find_second_term (A B : ℕ) (h1 : A / B = 3 / 4) (h2 : (A + 10) / (B + 10) = 4 / 5) : B = 40 :=
sorry

end NUMINAMATH_GPT_find_second_term_l105_10595


namespace NUMINAMATH_GPT_enlarged_sticker_height_l105_10531

theorem enlarged_sticker_height (original_width original_height new_width : ℕ) 
  (h1 : original_width = 3) 
  (h2 : original_height = 2) 
  (h3 : new_width = 12) : (new_width / original_width) * original_height = 8 := 
by 
  -- Prove the height of the enlarged sticker is 8 inches
  sorry

end NUMINAMATH_GPT_enlarged_sticker_height_l105_10531


namespace NUMINAMATH_GPT_quadratic_inequality_solution_l105_10579

theorem quadratic_inequality_solution (x : ℝ) : x^2 - 5 * x + 6 < 0 ↔ 2 < x ∧ x < 3 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_inequality_solution_l105_10579


namespace NUMINAMATH_GPT_ship_length_in_emilys_steps_l105_10575

variable (L E S : ℝ)

-- Conditions from the problem:
variable (cond1 : 240 * E = L + 240 * S)
variable (cond2 : 60 * E = L - 60 * S)

-- Theorem to prove:
theorem ship_length_in_emilys_steps (cond1 : 240 * E = L + 240 * S) (cond2 : 60 * E = L - 60 * S) : 
  L = 96 * E := 
sorry

end NUMINAMATH_GPT_ship_length_in_emilys_steps_l105_10575


namespace NUMINAMATH_GPT_min_value_l105_10514

theorem min_value (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 3 * x + y + 1 / x + 2 / y = 13 / 2) :
  x - 1 / y ≥ - 1 / 2 :=
sorry

end NUMINAMATH_GPT_min_value_l105_10514


namespace NUMINAMATH_GPT_arithmetic_sequence_150th_term_l105_10578

theorem arithmetic_sequence_150th_term :
  let a1 := 3
  let d := 5
  let n := 150
  a1 + (n - 1) * d = 748 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_150th_term_l105_10578


namespace NUMINAMATH_GPT_average_mileage_correct_l105_10556

def total_distance : ℕ := 150 * 2
def sedan_mileage : ℕ := 25
def hybrid_mileage : ℕ := 50
def sedan_gas_used : ℕ := 150 / sedan_mileage
def hybrid_gas_used : ℕ := 150 / hybrid_mileage
def total_gas_used : ℕ := sedan_gas_used + hybrid_gas_used
def average_gas_mileage : ℚ := total_distance / total_gas_used

theorem average_mileage_correct :
  average_gas_mileage = 33 + 1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_average_mileage_correct_l105_10556


namespace NUMINAMATH_GPT_factorial_mod_5_l105_10580

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem factorial_mod_5 :
  (factorial 1 + factorial 2 + factorial 3 + factorial 4 + factorial 5 +
   factorial 6 + factorial 7 + factorial 8 + factorial 9 + factorial 10) % 5 = 3 :=
by
  sorry

end NUMINAMATH_GPT_factorial_mod_5_l105_10580


namespace NUMINAMATH_GPT_binary_to_base4_conversion_l105_10515

theorem binary_to_base4_conversion : ∀ (a b c d e : ℕ), 
  1101101101 = (11 * 2^8) + (01 * 2^6) + (10 * 2^4) + (11 * 2^2) + 01 -> 
  a = 3 -> b = 1 -> c = 2 -> d = 3 -> e = 1 -> 
  (a*10000 + b*1000 + c*100 + d*10 + e : ℕ) = 31131 :=
by
  -- proof will go here
  sorry

end NUMINAMATH_GPT_binary_to_base4_conversion_l105_10515


namespace NUMINAMATH_GPT_range_of_a_l105_10591

open Real

theorem range_of_a (a : ℝ) : 
  (¬ ∃ x : ℝ, x^2 + (a - 1) * x + 1 < 0) → -1 ≤ a ∧ a ≤ 3 :=
by
  intro h
  -- insert the actual proof here
  sorry

end NUMINAMATH_GPT_range_of_a_l105_10591


namespace NUMINAMATH_GPT_remembers_umbrella_prob_l105_10546

theorem remembers_umbrella_prob 
    (P_forgets : ℚ) 
    (h_forgets : P_forgets = 5 / 8) : 
    ∃ P_remembers : ℚ, P_remembers = 3 / 8 := 
by
    sorry

end NUMINAMATH_GPT_remembers_umbrella_prob_l105_10546


namespace NUMINAMATH_GPT_ethan_arianna_apart_l105_10539

def ethan_distance := 1000 -- the distance Ethan ran
def arianna_distance := 184 -- the distance Arianna ran

theorem ethan_arianna_apart : ethan_distance - arianna_distance = 816 := by
  sorry

end NUMINAMATH_GPT_ethan_arianna_apart_l105_10539


namespace NUMINAMATH_GPT_alice_age_l105_10529

theorem alice_age (a m : ℕ) (h1 : a = m - 18) (h2 : a + m = 50) : a = 16 := by
  sorry

end NUMINAMATH_GPT_alice_age_l105_10529


namespace NUMINAMATH_GPT_sqrt_fraction_expression_l105_10547

theorem sqrt_fraction_expression : 
  (Real.sqrt (9 / 4) - Real.sqrt (4 / 9) + (Real.sqrt (9 / 4) + Real.sqrt (4 / 9))^2) = (199 / 36) := 
by
  sorry

end NUMINAMATH_GPT_sqrt_fraction_expression_l105_10547


namespace NUMINAMATH_GPT_even_and_odd_functions_satisfying_equation_l105_10587

theorem even_and_odd_functions_satisfying_equation :
  ∀ (f g : ℝ → ℝ),
    (∀ x : ℝ, f (-x) = f x) →                      -- condition 1: f is even
    (∀ x : ℝ, g (-x) = -g x) →                    -- condition 2: g is odd
    (∀ x : ℝ, f x - g x = x^3 + x^2 + 1) →        -- condition 3: f(x) - g(x) = x^3 + x^2 + 1
    f 1 + g 1 = 1 :=                              -- question: proof of f(1) + g(1) = 1
by
  intros f g h_even h_odd h_eqn
  sorry

end NUMINAMATH_GPT_even_and_odd_functions_satisfying_equation_l105_10587


namespace NUMINAMATH_GPT_remainder_when_divided_by_x_minus_2_l105_10560

-- Define the polynomial
def f (x : ℕ) : ℕ := x^3 - x^2 + 4 * x - 1

-- Statement of the problem: Prove f(2) = 11 using the Remainder Theorem
theorem remainder_when_divided_by_x_minus_2 : f 2 = 11 :=
by
  sorry

end NUMINAMATH_GPT_remainder_when_divided_by_x_minus_2_l105_10560


namespace NUMINAMATH_GPT_diff_present_students_l105_10549

theorem diff_present_students (T A1 A2 A3 P1 P2 : ℕ) 
  (hT : T = 280)
  (h_total_absent : A1 + A2 + A3 = 240)
  (h_absent_ratio : A2 = 2 * A3)
  (h_absent_third_day : A3 = 280 / 7) 
  (hP1 : P1 = T - A1)
  (hP2 : P2 = T - A2) :
  P2 - P1 = 40 :=
sorry

end NUMINAMATH_GPT_diff_present_students_l105_10549


namespace NUMINAMATH_GPT_max_area_circle_between_parallel_lines_l105_10585

theorem max_area_circle_between_parallel_lines : 
  ∀ (l₁ l₂ : ℝ → ℝ → Prop), 
    (∀ x y, l₁ x y ↔ 3*x - 4*y = 0) → 
    (∀ x y, l₂ x y ↔ 3*x - 4*y - 20 = 0) → 
  ∃ A, A = 4 * Real.pi :=
by 
  sorry

end NUMINAMATH_GPT_max_area_circle_between_parallel_lines_l105_10585


namespace NUMINAMATH_GPT_nonagon_blue_quadrilateral_l105_10598

theorem nonagon_blue_quadrilateral :
  ∀ (vertices : Finset ℕ) (red blue : ℕ → ℕ → Prop),
    (vertices.card = 9) →
    (∀ a b, red a b ∨ blue a b) →
    (∀ a b c, (red a b ∧ red b c ∧ red c a) → False) →
    (∃ A B C D, blue A B ∧ blue B C ∧ blue C D ∧ blue D A ∧ blue A C ∧ blue B D) := 
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_nonagon_blue_quadrilateral_l105_10598


namespace NUMINAMATH_GPT_lollipops_left_l105_10555

def problem_conditions : Prop :=
  ∃ (lollipops_bought lollipops_eaten lollipops_left : ℕ),
    lollipops_bought = 12 ∧
    lollipops_eaten = 5 ∧
    lollipops_left = lollipops_bought - lollipops_eaten

theorem lollipops_left (lollipops_bought lollipops_eaten lollipops_left : ℕ) 
  (hb : lollipops_bought = 12) (he : lollipops_eaten = 5) (hl : lollipops_left = lollipops_bought - lollipops_eaten) : 
  lollipops_left = 7 := 
by 
  sorry

end NUMINAMATH_GPT_lollipops_left_l105_10555


namespace NUMINAMATH_GPT_right_triangle_angle_l105_10513

theorem right_triangle_angle {A B C : ℝ} (hABC : A + B + C = 180) (hC : C = 90) (hA : A = 70) : B = 20 :=
sorry

end NUMINAMATH_GPT_right_triangle_angle_l105_10513


namespace NUMINAMATH_GPT_min_value_of_g_function_l105_10541

noncomputable def g (x : Real) := x + (x + 1) / (x^2 + 1) + (x * (x + 3)) / (x^2 + 3) + (3 * (x + 1)) / (x * (x^2 + 3))

theorem min_value_of_g_function : ∀ x : ℝ, x > 0 → g x ≥ 3 := sorry

end NUMINAMATH_GPT_min_value_of_g_function_l105_10541


namespace NUMINAMATH_GPT_compare_M_N_l105_10503

variable (a : ℝ)

def M : ℝ := 2 * a^2 - 4 * a
def N : ℝ := a^2 - 2 * a - 3

theorem compare_M_N : M a > N a := by
  sorry

end NUMINAMATH_GPT_compare_M_N_l105_10503


namespace NUMINAMATH_GPT_special_pair_example_1_special_pair_example_2_special_pair_negation_l105_10592

-- Definition of "special rational number pair"
def is_special_rational_pair (a b : ℚ) : Prop := a + b = a * b - 1

-- Problem (1)
theorem special_pair_example_1 : is_special_rational_pair 5 (3 / 2) :=
  by sorry

-- Problem (2)
theorem special_pair_example_2 (a : ℚ) : is_special_rational_pair a 3 → a = 2 :=
  by sorry

-- Problem (3)
theorem special_pair_negation (m n : ℚ) : is_special_rational_pair m n → ¬ is_special_rational_pair (-n) (-m) :=
  by sorry

end NUMINAMATH_GPT_special_pair_example_1_special_pair_example_2_special_pair_negation_l105_10592


namespace NUMINAMATH_GPT_salary_May_l105_10544

theorem salary_May
  (J F M A M' : ℝ)
  (h1 : (J + F + M + A) / 4 = 8000)
  (h2 : (F + M + A + M') / 4 = 8400)
  (h3 : J = 4900) :
  M' = 6500 :=
  by
  sorry

end NUMINAMATH_GPT_salary_May_l105_10544


namespace NUMINAMATH_GPT_oranges_ratio_l105_10551

theorem oranges_ratio (T : ℕ) (h1 : 100 + T + 70 = 470) : T / 100 = 3 := by
  -- The solution steps are omitted.
  sorry

end NUMINAMATH_GPT_oranges_ratio_l105_10551


namespace NUMINAMATH_GPT_f_at_one_f_increasing_f_range_for_ineq_l105_10565

-- Define the function f with its properties
noncomputable def f : ℝ → ℝ := sorry

-- Properties of f
axiom f_domain : ∀ x, 0 < x → f x ≠ 0 
axiom f_property_additive : ∀ x y, f (x * y) = f x + f y
axiom f_property_positive : ∀ x, (1 < x) → (0 < f x)
axiom f_property_fract : f (1/3) = -1

-- Proofs to be completed
theorem f_at_one : f 1 = 0 :=
sorry

theorem f_increasing : ∀ (x₁ x₂ : ℝ), (0 < x₁) → (0 < x₂) → (x₁ < x₂) → (f x₁ < f x₂) :=
sorry

theorem f_range_for_ineq : {x : ℝ | 2 < x ∧ x ≤ 9/4} = {x : ℝ | f x - f (x - 2) ≥ 2} :=
sorry

end NUMINAMATH_GPT_f_at_one_f_increasing_f_range_for_ineq_l105_10565


namespace NUMINAMATH_GPT_value_of_v3_at_2_l105_10569

def f (x : ℝ) : ℝ := x^5 - 2 * x^4 + 3 * x^3 - 7 * x^2 + 6 * x - 3

def v3 (x : ℝ) := (x - 2) * x + 3 
def v3_eval_at_2 : ℝ := (2 - 2) * 2 + 3

theorem value_of_v3_at_2 : v3 2 - 7 = -1 := by
    sorry

end NUMINAMATH_GPT_value_of_v3_at_2_l105_10569


namespace NUMINAMATH_GPT_positive_difference_of_two_numbers_l105_10548

theorem positive_difference_of_two_numbers (x y : ℝ) (h1 : x + y = 10) (h2 : x^2 - y^2 = 40) : |x - y| = 4 :=
by
  sorry

end NUMINAMATH_GPT_positive_difference_of_two_numbers_l105_10548


namespace NUMINAMATH_GPT_find_special_number_l105_10508

theorem find_special_number:
  ∃ (n : ℕ), (n > 0) ∧ (∃ (k : ℕ), 2 * n = k^2)
           ∧ (∃ (m : ℕ), 3 * n = m^3)
           ∧ (∃ (p : ℕ), 5 * n = p^5)
           ∧ n = 1085 :=
by
  sorry

end NUMINAMATH_GPT_find_special_number_l105_10508


namespace NUMINAMATH_GPT_remainder_of_c_plus_d_l105_10506

-- Definitions based on conditions
def c (k : ℕ) : ℕ := 60 * k + 53
def d (m : ℕ) : ℕ := 40 * m + 29

-- Statement of the problem
theorem remainder_of_c_plus_d (k m : ℕ) :
  ((c k + d m) % 20) = 2 :=
by
  unfold c
  unfold d
  sorry

end NUMINAMATH_GPT_remainder_of_c_plus_d_l105_10506


namespace NUMINAMATH_GPT_find_profit_percentage_l105_10519

theorem find_profit_percentage (h : (m + 8) / (1 - 0.08) = m + 10) : m = 15 := sorry

end NUMINAMATH_GPT_find_profit_percentage_l105_10519


namespace NUMINAMATH_GPT_average_math_score_l105_10581

theorem average_math_score (scores : Fin 4 → ℕ) (other_avg : ℕ) (num_students : ℕ) (num_other_students : ℕ)
  (h1 : scores 0 = 90) (h2 : scores 1 = 85) (h3 : scores 2 = 88) (h4 : scores 3 = 80)
  (h5 : other_avg = 82) (h6 : num_students = 30) (h7 : num_other_students = 26) :
  (90 + 85 + 88 + 80 + 26 * 82) / 30 = 82.5 :=
by
  sorry

end NUMINAMATH_GPT_average_math_score_l105_10581


namespace NUMINAMATH_GPT_minimum_x_y_l105_10583

theorem minimum_x_y (x y : ℝ) (hx_pos : x > 0) (hy_pos : y > 0) (h_eq : 2 * x + 8 * y = x * y) : x + y ≥ 18 :=
by sorry

end NUMINAMATH_GPT_minimum_x_y_l105_10583


namespace NUMINAMATH_GPT_problem_solution_l105_10530

-- Definitions of sets
def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 2}
def B : Set ℕ := {2, 3, 4}

-- Complement within U
def complement_U (A : Set ℕ) : Set ℕ := {x ∈ U | x ∉ A}

-- The proof goal
theorem problem_solution : (complement_U A) ∪ B = {2, 3, 4, 5} := by
  sorry

end NUMINAMATH_GPT_problem_solution_l105_10530


namespace NUMINAMATH_GPT_hyperbola_parabola_focus_l105_10534

open Classical

theorem hyperbola_parabola_focus :
  ∃ a : ℝ, (a > 0) ∧ (∃ c > 0, (c = 2) ∧ (a^2 + 3 = c^2)) → a = 1 :=
sorry

end NUMINAMATH_GPT_hyperbola_parabola_focus_l105_10534


namespace NUMINAMATH_GPT_length_of_side_divisible_by_4_l105_10500

theorem length_of_side_divisible_by_4 {m n : ℕ} 
  (h : ∀ k : ℕ, (m * k) + (n * k) % 4 = 0 ) : 
  m % 4 = 0 ∨ n % 4 = 0 :=
by
  sorry

end NUMINAMATH_GPT_length_of_side_divisible_by_4_l105_10500


namespace NUMINAMATH_GPT_circle_passes_through_fixed_point_l105_10524

theorem circle_passes_through_fixed_point :
  ∀ (C : ℝ × ℝ), (C.2 ^ 2 = 4 * C.1) ∧ (C.1 = -1 + (C.1 + 1)) → ∃ P : ℝ × ℝ, P = (1, 0) ∧
    (P.1 - C.1) ^ 2 + (P.2 - C.2) ^ 2 = (C.1 + 1) ^ 2 + (0 - C.2) ^ 2 :=
by
  sorry

end NUMINAMATH_GPT_circle_passes_through_fixed_point_l105_10524


namespace NUMINAMATH_GPT_balloon_arrangement_count_l105_10536

theorem balloon_arrangement_count :
  let n := 7
  let l := 2
  let o := 2
  n.factorial / (l.factorial * o.factorial) = 1260 :=
by
  sorry

end NUMINAMATH_GPT_balloon_arrangement_count_l105_10536


namespace NUMINAMATH_GPT_sum_of_solutions_eq_8_l105_10511

theorem sum_of_solutions_eq_8 :
    let a : ℝ := 1
    let b : ℝ := -8
    let c : ℝ := -26
    ∀ x1 x2 : ℝ, (a * x1^2 + b * x1 + c = 0) ∧ (a * x2^2 + b * x2 + c = 0) →
      x1 + x2 = 8 :=
sorry

end NUMINAMATH_GPT_sum_of_solutions_eq_8_l105_10511


namespace NUMINAMATH_GPT_first_pump_rate_is_180_l105_10562

-- Define the known conditions
variables (R : ℕ) -- The rate of the first pump in gallons per hour
def second_pump_rate : ℕ := 250 -- The rate of the second pump in gallons per hour
def second_pump_time : ℕ := 35 / 10 -- 3.5 hours represented as a fraction
def total_pump_time : ℕ := 60 / 10 -- 6 hours represented as a fraction
def total_volume : ℕ := 1325 -- Total volume pumped by both pumps in gallons

-- Define derived conditions from the problem
def second_pump_volume : ℕ := second_pump_rate * second_pump_time -- Volume pumped by the second pump
def first_pump_volume : ℕ := total_volume - second_pump_volume -- Volume pumped by the first pump
def first_pump_time : ℕ := total_pump_time - second_pump_time -- Time the first pump was used

-- The main theorem to prove that the rate of the first pump is 180 gallons per hour
theorem first_pump_rate_is_180 : R = 180 :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_GPT_first_pump_rate_is_180_l105_10562


namespace NUMINAMATH_GPT_xyz_value_l105_10558

theorem xyz_value (x y z : ℝ) 
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 40)
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) + 2 * x * y * z = 20) 
  : x * y * z = 20 := 
by
  sorry

end NUMINAMATH_GPT_xyz_value_l105_10558


namespace NUMINAMATH_GPT_ratio_a_to_b_l105_10532

theorem ratio_a_to_b (a b : ℝ) (h : (a - 3 * b) / (2 * a - b) = 0.14285714285714285) : a / b = 4 :=
by 
  -- The proof goes here
  sorry

end NUMINAMATH_GPT_ratio_a_to_b_l105_10532


namespace NUMINAMATH_GPT_mixed_number_expression_l105_10596

open Real

-- Definitions of the given mixed numbers
def mixed_number1 : ℚ := (37 / 7)
def mixed_number2 : ℚ := (18 / 5)
def mixed_number3 : ℚ := (19 / 6)
def mixed_number4 : ℚ := (9 / 4)

-- Main theorem statement
theorem mixed_number_expression :
  25 * (mixed_number1 - mixed_number2) / (mixed_number3 + mixed_number4) = 7 + 49 / 91 :=
by
  sorry

end NUMINAMATH_GPT_mixed_number_expression_l105_10596


namespace NUMINAMATH_GPT_binomial_odd_sum_l105_10537

theorem binomial_odd_sum (n : ℕ) (h : (2:ℕ)^(n - 1) = 64) : n = 7 :=
by
  sorry

end NUMINAMATH_GPT_binomial_odd_sum_l105_10537


namespace NUMINAMATH_GPT_purely_imaginary_complex_number_l105_10564

theorem purely_imaginary_complex_number (a : ℝ) 
  (h1 : (a^2 - 4 * a + 3 = 0))
  (h2 : a ≠ 1) 
  : a = 3 := 
sorry

end NUMINAMATH_GPT_purely_imaginary_complex_number_l105_10564


namespace NUMINAMATH_GPT_largest_prime_factor_always_37_l105_10563

-- We define the cyclic sequence conditions
def cyclic_shift (seq : List ℕ) : Prop :=
  ∀ i, seq.get! (i % seq.length) % 10 = seq.get! ((i + 1) % seq.length) / 100 ∧
       (seq.get! ((i + 1) % seq.length) / 10 % 10 = seq.get! ((i + 2) % seq.length) % 10) ∧
       (seq.get! ((i + 2) % seq.length) / 10 % 10 = seq.get! ((i + 3) % seq.length) / 100)

-- Summing all elements of a list
def sum (l : List ℕ) : ℕ := l.foldr (· + ·) 0

-- Prove that 37 is always a factor of the sum T
theorem largest_prime_factor_always_37 (seq : List ℕ) (h : cyclic_shift seq) : 
  37 ∣ sum seq := 
sorry

end NUMINAMATH_GPT_largest_prime_factor_always_37_l105_10563


namespace NUMINAMATH_GPT_find_number_l105_10517

theorem find_number (x : ℝ) (h : 2 * x - 2.6 * 4 = 10) : x = 10.2 :=
sorry

end NUMINAMATH_GPT_find_number_l105_10517


namespace NUMINAMATH_GPT_remainder_4873_div_29_l105_10542

theorem remainder_4873_div_29 : 4873 % 29 = 1 := 
by sorry

end NUMINAMATH_GPT_remainder_4873_div_29_l105_10542


namespace NUMINAMATH_GPT_sacks_after_days_l105_10589

-- Define the number of sacks harvested per day
def harvest_per_day : ℕ := 74

-- Define the number of sacks discarded per day
def discard_per_day : ℕ := 71

-- Define the days of harvest
def days_of_harvest : ℕ := 51

-- Define the number of sacks that are not discarded per day
def net_sacks_per_day : ℕ := harvest_per_day - discard_per_day

-- Define the total number of sacks after the specified days of harvest
def total_sacks : ℕ := days_of_harvest * net_sacks_per_day

theorem sacks_after_days :
  total_sacks = 153 := by
  sorry

end NUMINAMATH_GPT_sacks_after_days_l105_10589


namespace NUMINAMATH_GPT_determine_n_l105_10518

noncomputable def S : ℕ → ℝ := sorry -- define arithmetic series sum
noncomputable def a_1 : ℝ := sorry -- define first term
noncomputable def d : ℝ := sorry -- define common difference

axiom S_6 : S 6 = 36
axiom S_n {n : ℕ} (h : n > 0) : S n = 324
axiom S_n_minus_6 {n : ℕ} (h : n > 6) : S (n - 6) = 144

theorem determine_n (n : ℕ) (h : n > 0) : n = 18 := by {
  sorry
}

end NUMINAMATH_GPT_determine_n_l105_10518
