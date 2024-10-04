import Mathlib
import Mathlib.Algebra.Group.InjSurj
import Mathlib.Algebra.GroupWithOne
import Mathlib.Algebra.Module.Basic
import Mathlib.Algebra.Polynomial
import Mathlib.Algebra.Trigonometry
import Mathlib.Analysis.Calculus.Deriv
import Mathlib.Analysis.Calculus.Volume
import Mathlib.Analysis.SpecialFunctions.Trigonometric
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Analysis.Trigonometry.Basic
import Mathlib.Data.Binomial
import Mathlib.Data.Complex.Exponential
import Mathlib.Data.Finset
import Mathlib.Data.Fintype.Basic
import Mathlib.Data.List.Sort
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Nat.Basic
import Mathlib.Data.Nat.Prime
import Mathlib.Data.PNat.Basic
import Mathlib.Data.Polynomial
import Mathlib.Data.Polynomial.Basic
import Mathlib.Data.Rat.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Basic
import Mathlib.Data.Set.Finite
import Mathlib.Geometry.Euclidean.Affine
import Mathlib.Geometry.Euclidean.Triangle
import Mathlib.Init.Data.Bool.Basic
import Mathlib.NumberTheory.Prime
import Mathlib.NumberTheory.Primes
import Mathlib.PrePort
import Mathlib.Probability.ProbabilityMassFunction
import Mathlib.Tactic
import Mathlib.TopPort
import Mathlib.Topology.Algebra.InfiniteSum
import Mathlib.Topology.Basic
import ProbabilityTheory

namespace part1_l642_642916

-- Definitions
def int_part (m : ℝ) : ℕ := floor m
def sqrt10 := Real.sqrt 10

-- Main statement
theorem part1 : int_part (sqrt10 + 1) = 4 := sorry

end part1_l642_642916


namespace problem1_problem2_problem3_l642_642795

noncomputable def f (x : ℝ) : ℝ := if x >= 0 then x^2 - 2 * x else (abs x)^2 - 2 * abs x

-- Define the condition that f is an even function
def even_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f x = f (-x)

-- Problem 1: Prove the minimum value of f(x) is -1.
theorem problem1 (h_even : even_function f) : ∃ x : ℝ, f x = -1 :=
by
  sorry

-- Problem 2: Prove the solution set of f(x) > 0 is (-∞, -2) ∪ (2, +∞).
theorem problem2 (h_even : even_function f) : 
  { x : ℝ | f x > 0 } = { x : ℝ | x < -2 } ∪ { x : ℝ | x > 2 } :=
by
  sorry

-- Problem 3: Prove there exists a real number x such that f(x+2) + f(-x) = 0.
theorem problem3 (h_even : even_function f) : ∃ x : ℝ, f (x + 2) + f (-x) = 0 :=
by
  sorry

end problem1_problem2_problem3_l642_642795


namespace diff_of_percentages_l642_642022

def percentage (percent : ℝ) (amount : ℝ) : ℝ := (percent / 100) * amount

theorem diff_of_percentages : 
  percentage 25 37 - percentage 25 17 = 5 :=
by
  sorry

end diff_of_percentages_l642_642022


namespace angle_950_12_second_quadrant_angle_1575_third_quadrant_l642_642735

def coterminal_angle (angle : Real) : Real :=
  let remainder := angle % 360
  if remainder < 0 then remainder + 360 else remainder

def quadrant (angle : Real) : String :=
  let coterm := coterminal_angle angle
  if coterm > 0 ∧ coterm < 90 then "first"
  else if coterm > 90 ∧ coterm < 180 then "second"
  else if coterm > 180 ∧ coterm < 270 then "third"
  else if coterm > 270 ∧ coterm < 360 then "fourth"
  else "on_axis"

theorem angle_950_12_second_quadrant : quadrant (-950 - 12 / 60) = "second" := by
  sorry

theorem angle_1575_third_quadrant : quadrant (-1575) = "third" := by
  sorry

end angle_950_12_second_quadrant_angle_1575_third_quadrant_l642_642735


namespace max_powers_of_three_l642_642592

theorem max_powers_of_three (n : ℕ) (S : set ℝ) (h1 : S.card = n) (h2: ∀ x ∈ S, 0 < x) (h3 : ∀ x y ∈ S, x ≠ y → x ≠ y) (hn : 3 ≤ n) :
  let T := { k : ℤ | ∃ a b c ∈ S, a ≠ b ∧ a ≠ c ∧ b ≠ c ∧ a + b + c = (3 : ℝ)^k } in
  T.card ≤ n-2 :=
begin
  sorry
end

end max_powers_of_three_l642_642592


namespace convex_hexagon_diagonal_range_l642_642103

-- Define what it means to have a convex hexagon with all sides equal
structure ConvexHexagon (α : Type) [MetricSpace α] :=
  (A B C D E F : α)
  (convex : Convex α (set.insert A (set.insert B (set.insert C (set.insert D (set.insert E (set.insert F ∅)))))))
  (side_length_eq_one : ∀ P Q : α, (P, Q) ∈ {(A, B), (B, C), (C, D), (D, E), (E, F), (F, A)} → dist P Q = 1)

-- Function to get the distances of specific diagonals AD, BE, CF
def diagonals {α : Type} [MetricSpace α] (h : ConvexHexagon α) : Finset ℝ :=
  {dist h.A h.D, dist h.B h.E, dist h.C h.F}

-- Prove that the range is [1, 3]
theorem convex_hexagon_diagonal_range (α : Type) [MetricSpace α] (h : ConvexHexagon α) :
  let ds := diagonals h in
  Finset.min' ds (by sorry) = 1 ∧ Finset.max' ds (by sorry) = 3 := 
sorry

end convex_hexagon_diagonal_range_l642_642103


namespace major_premise_of_rectangle_triangle_l642_642338

theorem major_premise_of_rectangle_triangle :
  (∀ (R : Type) (P : Type), (R → P) → (¬(T : Type) (P : Type), ¬(T → P)) → (¬(T → R))) ↔ (∀ (R : Type) (P : Type), (R → P)) :=
by
  sorry

end major_premise_of_rectangle_triangle_l642_642338


namespace cos_150_eq_neg_sqrt3_div_2_l642_642259

theorem cos_150_eq_neg_sqrt3_div_2 :
  let theta := 30 * Real.pi / 180 in
  let Q := 150 * Real.pi / 180 in
  cos Q = - (Real.sqrt 3 / 2) := by
    sorry

end cos_150_eq_neg_sqrt3_div_2_l642_642259


namespace joy_sees_grandma_in_48_hours_l642_642502

def days_until_joy_sees_grandma : ℕ := 2
def hours_per_day : ℕ := 24

theorem joy_sees_grandma_in_48_hours :
  days_until_joy_sees_grandma * hours_per_day = 48 := 
by
  sorry

end joy_sees_grandma_in_48_hours_l642_642502


namespace probability_of_selecting_one_station_from_each_jurisdiction_probability_of_selecting_at_least_one_station_within_Xiaogan_l642_642917

-- Definitions of the problem
def WuhanStations : List String := ["Houhu", "Jinyintan", "Tianhe Airport", "Tianhe Street"]
def XiaoganStations : List String := ["Minji", "Maochen", "Huaiyin"]

def totalStations : Nat := WuhanStations.length + XiaoganStations.length

-- Define the events
def eventM : List (String × String) :=
  List.product WuhanStations XiaoganStations

def eventN : List (String × String) :=
  (WuhanStations.product WuhanStations).filter (λ s, s.1 ≠ s.2)

-- Total number of combinations of selecting 2 out of 7 stations
def totalCombinations (n : Nat) : Nat :=
  n * (n - 1) / 2

def totalEvents : Nat := totalCombinations totalStations

-- Probabilities
def PM : Real := eventM.length / totalEvents
def PN : Real := eventN.length / totalEvents
def PNComplement : Real := 1 - PN

theorem probability_of_selecting_one_station_from_each_jurisdiction :
  PM = 4 / 7 :=
by
  sorry

theorem probability_of_selecting_at_least_one_station_within_Xiaogan :
  PNComplement = 5 / 7 :=
by
  sorry

end probability_of_selecting_one_station_from_each_jurisdiction_probability_of_selecting_at_least_one_station_within_Xiaogan_l642_642917


namespace half_liar_today_is_saturday_l642_642348

theorem half_liar_today_is_saturday
  (alternates : ∀ n : ℕ, half_liar n = ¬ half_liar (n + 1))
  (last_week_false : half_liar 7 = false)
  (statement_today_1 : yesterday_is_friday = true)
  (statement_today_2 : tomorrow_is_day_off = true) 
  : today_is_saturday = true := sorry

end half_liar_today_is_saturday_l642_642348


namespace trigonometric_identity_l642_642782

noncomputable def cos_of_angle_in_third_quadrant (α : ℝ) : Prop :=
  α > π ∧ α < (3 * π / 2)

theorem trigonometric_identity 
  (α : ℝ) (hα : cos_of_angle_in_third_quadrant α) 
  (h : Real.cos (75 * Real.pi / 180 + α) = 1 / 3) : 
  Real.cos (105 * Real.pi / 180 - α) + Real.sin (α - 105 * Real.pi / 180) = (2 * Real.sqrt 2 - 1) / 3 :=
by
  sorry

end trigonometric_identity_l642_642782


namespace cos_150_deg_l642_642329

-- Define the conditions as variables
def angle1 := 150
def angle2 := 180
def angle3 := 30
def cos30 := Real.sqrt 3 / 2

-- The main statement to be proved
theorem cos_150_deg : Real.cos (150 * Real.pi / 180) = - (Real.sqrt 3 / 2) :=
by
  sorry

end cos_150_deg_l642_642329


namespace cos_150_eq_neg_sqrt3_over_2_l642_642218

theorem cos_150_eq_neg_sqrt3_over_2 :
  ∀ {deg cosQ1 cosQ2: ℝ},
    cosQ1 = 30 →
    cosQ2 = 180 - cosQ1 →
    cos 150 = - (cos cosQ1) →
    cos cosQ1 = sqrt 3 / 2 →
    cos 150 = -sqrt 3 / 2 :=
by
  intros deg cosQ1 cosQ2 h1 h2 h3 h4
  sorry

end cos_150_eq_neg_sqrt3_over_2_l642_642218


namespace find_remainder_l642_642585

def dividend : ℕ := 997
def divisor : ℕ := 23
def quotient : ℕ := 43

theorem find_remainder : ∃ r : ℕ, dividend = (divisor * quotient) + r ∧ r = 8 :=
by
  sorry

end find_remainder_l642_642585


namespace cos_150_eq_neg_sqrt3_div_2_l642_642294

theorem cos_150_eq_neg_sqrt3_div_2 :
  (cos (150 * real.pi / 180)) = - (sqrt 3 / 2) := sorry

end cos_150_eq_neg_sqrt3_div_2_l642_642294


namespace number_of_correct_propositions_l642_642684

-- Define the propositions as separate conditions
def prop1 : Prop := ∀ (P Q : Plane) (l : Line), (P || l) ∧ (Q || l) → (P || Q)
def prop2 : Prop := ∀ (P Q : Plane) (R : Plane), (P || R) ∧ (Q || R) → (P || Q)
def prop3 : Prop := ∀ (l1 l2 : Line) (l : Line), (l1 ⟂ l) ∧ (l2 ⟂ l) → (l1 || l2)
def prop4 : Prop := ∀ (l1 l2 : Line) (P : Plane), (l1 ⟂ P) ∧ (l2 ⟂ P) → (l1 || l2)

-- The problem statement
theorem number_of_correct_propositions : (¬ prop1) ∧ prop2 ∧ (¬ prop3) ∧ prop4 → (2 = 2) :=
by
  intro h
  sorry

end number_of_correct_propositions_l642_642684


namespace max_misery_ratio_le_9_over_8_l642_642619

noncomputable def max_misery_ratio (n m : ℕ) (bits : Fin m → ℕ) (room_assignments_balanced room_assignments_proposed : Fin m → Fin n) : ℚ :=
  let load_balanced (r : Fin n) := (Finset.univ.filter (λ (s : Fin m), room_assignments_balanced s = r)).sum (λ s, bits s)
  let load_proposed (r : Fin n) := (Finset.univ.filter (λ (s : Fin m), room_assignments_proposed s = r)).sum (λ s, bits s)
  let displeasure_balanced := (Finset.univ : Finset (Fin m)).sum (λ s, bits s * load_balanced (room_assignments_balanced s))
  let displeasure_proposed := (Finset.univ : Finset (Fin m)).sum (λ s, bits s * load_proposed (room_assignments_proposed s))
  (displeasure_balanced : ℚ) / displeasure_proposed

theorem max_misery_ratio_le_9_over_8 (n m : ℕ) (bits : Fin m → ℕ) (room_assignments_balanced room_assignments_proposed : Fin m → Fin n) :
  max_misery_ratio n m bits room_assignments_balanced room_assignments_proposed ≤ 9 / 8 := 
sorry

end max_misery_ratio_le_9_over_8_l642_642619


namespace power_inequality_l642_642395

open Nat

theorem power_inequality (a b : ℝ) (n : ℕ)
  (h1 : 0 < a) (h2 : 0 < b) (h3 : (1 / a) + (1 / b) = 1) :
  (a + b)^n - a^n - b^n ≥ 2^(2 * n) - 2^(n + 1) := 
  sorry

end power_inequality_l642_642395


namespace vector_calculation_l642_642120

open scoped Matrix

def given_vectors : Matrix (Fin 2) (Fin 1) ℤ :=
  ![![3], ![-8]]

def scalar_vector : Matrix (Fin 2) (Fin 1) ℤ :=
  ![![2], ![-3]]

def add_vector : Matrix (Fin 2) (Fin 1) ℤ :=
  ![[-1], ![4]]

def expected_result : Matrix (Fin 2) (Fin 1) ℤ :=
  ![[-8], ![11]]

theorem vector_calculation :
  given_vectors - (5 • scalar_vector) + add_vector = expected_result := by
  sorry

end vector_calculation_l642_642120


namespace f_seven_l642_642767

noncomputable def f : ℝ → ℝ := sorry

def f_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x
def f_periodic (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x
def f_interval (f : ℝ → ℝ) : Prop := ∀ x, 0 < x ∧ x < 2 → f x = x + 2

theorem f_seven : f_odd f ∧ f_periodic f 4 ∧ f_interval f → f 7 = -3 := 
by 
  intros,
  sorry

end f_seven_l642_642767


namespace negation_of_forall_statement_l642_642580

theorem negation_of_forall_statement :
  ¬ (∀ x : ℝ, x^2 + 2 * x ≥ 0) ↔ ∃ x : ℝ, x^2 + 2 * x < 0 := 
by
  sorry

end negation_of_forall_statement_l642_642580


namespace Jurgen_started_packing_time_l642_642503

theorem Jurgen_started_packing_time :
  ∀ (packing_time : ℕ) (walk_time : ℕ) (arrival_buffer : ℕ) (bus_leaves : ℕ),
    packing_time = 25 →
    walk_time = 35 →
    arrival_buffer = 60 →
    bus_leaves = 18 * 60 + 45 →
    let start_time := bus_leaves - (packing_time + walk_time + arrival_buffer) in
    start_time = 16 * 60 + 45 :=
by
  intros packing_time walk_time arrival_buffer bus_leaves hp hw ha hb
  simp [hp, hw, ha, hb]
  rfl

end Jurgen_started_packing_time_l642_642503


namespace violet_prob_l642_642595

noncomputable def total_candies := 8 + 5 + 9 + 10 + 6

noncomputable def prob_green_first := (8 : ℚ) / total_candies
noncomputable def prob_yellow_second := (10 : ℚ) / (total_candies - 1)
noncomputable def prob_pink_third := (6 : ℚ) / (total_candies - 2)

noncomputable def combined_prob := prob_green_first * prob_yellow_second * prob_pink_third

theorem violet_prob :
  combined_prob = (20 : ℚ) / 2109 := by
    sorry

end violet_prob_l642_642595


namespace circumference_diameter_linear_relationship_l642_642104

theorem circumference_diameter_linear_relationship :
  ∀ r : ℕ, r ∈ {2, 3, 4, 5, 6, 7} → ∃ k : ℝ, ∀ r : ℕ, C = k * D :=
by
  sorry

end circumference_diameter_linear_relationship_l642_642104


namespace chime2500_date_l642_642646

def chimes_in_first_day : ℕ := 13 + 90

def chimes_in_full_day : ℕ := 102

noncomputable def date_of_2500th_chime : String :=
let total_chimes_needed := 2500
let chimes_first_day := chimes_in_first_day
let chimes_remaining := total_chimes_needed - chimes_first_day
let full_days_needed := chimes_remaining / chimes_in_full_day
let remaining_chimes := chimes_remaining % chimes_in_full_day
if remaining_chimes == 0 then
  "March 21, 2003"
else
  "March 21, 2003"  -- assumed within the same day since 51 chimes would finish within March 21

theorem chime2500_date : date_of_2500th_chime = "March 21, 2003" :=
by
  let total_chimes_needed := 2500
  have chimes_first_day := chimes_in_first_day
  let chimes_remaining := total_chimes_needed - chimes_first_day
  have full_days_needed := chimes_remaining / chimes_in_full_day
  have remaining_chimes := chimes_remaining % chimes_in_full_day
  -- Calculation of days and chimes
  have date := if remaining_chimes == 0 then "March 21, 2003" else "March 21, 2003"
  exact Eq.refl date

end chime2500_date_l642_642646


namespace value_of_x_squared_plus_9y_squared_l642_642444

theorem value_of_x_squared_plus_9y_squared (x y : ℝ)
  (h1 : x + 3 * y = 5)
  (h2 : x * y = -8) : x^2 + 9 * y^2 = 73 :=
by
  sorry

end value_of_x_squared_plus_9y_squared_l642_642444


namespace paper_boat_travel_time_l642_642491

theorem paper_boat_travel_time :
  ∀ (length_of_embankment : ℝ) (length_of_motorboat : ℝ)
    (time_downstream : ℝ) (time_upstream : ℝ) (v_boat : ℝ) (v_current : ℝ),
  length_of_embankment = 50 →
  length_of_motorboat = 10 →
  time_downstream = 5 →
  time_upstream = 4 →
  v_boat + v_current = length_of_embankment / time_downstream →
  v_boat - v_current = length_of_embankment / time_upstream →
  let speed_paper_boat := v_current in
  let travel_time := length_of_embankment / speed_paper_boat in
  travel_time = 40 :=
by
  intros length_of_embankment length_of_motorboat time_downstream time_upstream v_boat v_current
  intros h_length_emb h_length_motor t_down t_up h_v_boat_plus_current h_v_boat_minus_current
  let speed_paper_boat := v_current
  let travel_time := length_of_embankment / speed_paper_boat
  sorry

end paper_boat_travel_time_l642_642491


namespace cos_150_eq_negative_cos_30_l642_642153

theorem cos_150_eq_negative_cos_30 :
  Real.cos (150 * Real.pi / 180) = - (Real.cos (30 * Real.pi / 180)) :=
by sorry

end cos_150_eq_negative_cos_30_l642_642153


namespace percentage_of_carnations_l642_642081

theorem percentage_of_carnations {F : ℕ} (hF : 0 < F) :
  let yellow_flowers := 0.3 * F,
      non_yellow_flowers := 0.7 * F,
      pink_flowers := 0.35 * F,
      red_flowers := 0.35 * F,
      pink_roses := (1/5) * pink_flowers,
      red_carnations := (1/2) * red_flowers,
      yellow_roses := (3/10) * yellow_flowers,
      pink_carnations := pink_flowers - pink_roses,
      yellow_carnations := yellow_flowers - yellow_roses,
      total_carnations := pink_carnations + red_carnations + yellow_carnations
  in (total_carnations / F) * 100 = 66.5 := 
sorry

end percentage_of_carnations_l642_642081


namespace cos_150_eq_neg_half_l642_642202

theorem cos_150_eq_neg_half :
  real.cos (150 * real.pi / 180) = -1 / 2 :=
sorry

end cos_150_eq_neg_half_l642_642202


namespace number_of_permutations_l642_642959

/-- The digits in the area code are distinct and are 9, 8, 7, and 6 --/
def digits := {9, 8, 7, 6}

/-- Define the length of the digits set --/
def length_of_digits := 4

/-- Define the factorial function --/
def factorial (n : Nat) : Nat :=
  if n = 0 then 1 else n * factorial (n - 1)

/-- Prove the number of permutations of the set of digits equals 24. --/
theorem number_of_permutations :
  factorial length_of_digits = 24 :=
by
  sorry

end number_of_permutations_l642_642959


namespace decision_box_shape_is_diamond_l642_642861

-- Define the concept of a flowchart
def flowchart : Type := sorry

-- Define what a decision box is in the context of a flowchart
def decision_box (f : flowchart) : Type := sorry

-- Define the shape of a decision box in a flowchart
def shape_of_decision_box (f : flowchart) : decision_box f := "diamond"

-- The statement to prove
theorem decision_box_shape_is_diamond (f : flowchart) : shape_of_decision_box f = "diamond" :=
sorry

end decision_box_shape_is_diamond_l642_642861


namespace right_triangle_sum_of_sides_l642_642507

theorem right_triangle_sum_of_sides (A B C P Q R K L M N : Point) (h1 : is_right_triangle A B C)
  (h2 : square APQR ∧ area APQR = 9 ∧ P ∈ [AC] ∧ Q ∈ [BC] ∧ R ∈ [AB])
  (h3 : square KLMN ∧ area KLMN = 8 ∧ N ∈ [BC] ∧ K ∈ [BC] ∧ M ∈ [AB] ∧ L ∈ [AC]):
  |AB| + |AC| = 12 :=
sorry

end right_triangle_sum_of_sides_l642_642507


namespace cost_of_fencing_l642_642628

open Real

theorem cost_of_fencing
  (ratio_length_width : ∃ x : ℝ, 3 * x * 2 * x = 3750)
  (cost_per_meter : ℝ := 0.50) :
  ∃ cost : ℝ, cost = 125 := by
  sorry

end cost_of_fencing_l642_642628


namespace sum_series_eq_one_l642_642714

noncomputable def series : ℝ := ∑' n : ℕ, (4 * (n + 1) - 3) / 3^(n + 1)

theorem sum_series_eq_one : series = 1 := 
by sorry

end sum_series_eq_one_l642_642714


namespace toby_steps_needed_l642_642009

noncomputable def total_steps_needed : ℕ := 10000 * 9

noncomputable def first_sunday_steps : ℕ := 10200
noncomputable def first_monday_steps : ℕ := 10400
noncomputable def tuesday_steps : ℕ := 9400
noncomputable def wednesday_steps : ℕ := 9100
noncomputable def thursday_steps : ℕ := 8300
noncomputable def friday_steps : ℕ := 9200
noncomputable def saturday_steps : ℕ := 8900
noncomputable def second_sunday_steps : ℕ := 9500

noncomputable def total_steps_walked := 
  first_sunday_steps + 
  first_monday_steps + 
  tuesday_steps + 
  wednesday_steps + 
  thursday_steps + 
  friday_steps + 
  saturday_steps + 
  second_sunday_steps

noncomputable def remaining_steps_needed := total_steps_needed - total_steps_walked

noncomputable def days_left : ℕ := 3

noncomputable def average_steps_needed := remaining_steps_needed / days_left

theorem toby_steps_needed : average_steps_needed = 5000 := by
  sorry

end toby_steps_needed_l642_642009


namespace stone_number_150_is_6_l642_642987

theorem stone_number_150_is_6 : 
  ∃ n : ℕ, n = 150 % 22 ∧ n = 6 :=
by
  have h1 : 150 % 22 = 18 := by norm_num
  existsi 18
  split
  exact h1
  norm_num
  exact rfl

end stone_number_150_is_6_l642_642987


namespace ice_cream_sales_l642_642727

def every_sixth_customer_free : Prop := ∀ (n : ℕ), n % 6 = 0 → gives_free_ice_cream (customer_at n)

def cone_cost : ℕ := 2

def free_cones : ℕ := 10

theorem ice_cream_sales (every_sixth_customer_free : every_sixth_customer_free) 
  (cone_cost : ℕ) (free_cones : ℕ) (cone_cost = 2) (free_cones = 10) :
  ∃ total_sales : ℕ, total_sales = 100 := 
  sorry

end ice_cream_sales_l642_642727


namespace boat_travels_125_km_downstream_l642_642068

/-- The speed of the boat in still water is 20 km/hr -/
def boat_speed_still_water : ℝ := 20

/-- The speed of the stream is 5 km/hr -/
def stream_speed : ℝ := 5

/-- The total time taken downstream is 5 hours -/
def total_time_downstream : ℝ := 5

/-- The effective speed of the boat downstream -/
def effective_speed_downstream : ℝ := boat_speed_still_water + stream_speed

/-- The distance the boat travels downstream -/
def distance_downstream : ℝ := effective_speed_downstream * total_time_downstream

/-- The boat travels 125 km downstream -/
theorem boat_travels_125_km_downstream :
  distance_downstream = 125 := 
sorry

end boat_travels_125_km_downstream_l642_642068


namespace cos_150_eq_neg_sqrt3_div_2_l642_642283

theorem cos_150_eq_neg_sqrt3_div_2 :
  (cos (150 * real.pi / 180)) = - (sqrt 3 / 2) := sorry

end cos_150_eq_neg_sqrt3_div_2_l642_642283


namespace lucy_current_fish_l642_642528

-- Definitions based on conditions in the problem
def total_fish : ℕ := 280
def fish_needed_to_buy : ℕ := 68

-- Proving the number of fish Lucy currently has
theorem lucy_current_fish : total_fish - fish_needed_to_buy = 212 :=
by
  sorry

end lucy_current_fish_l642_642528


namespace cos_150_eq_neg_sqrt3_div_2_l642_642161

theorem cos_150_eq_neg_sqrt3_div_2 : Real.cos (150 * Real.pi / 180) = - (Real.sqrt 3 / 2) :=
by
  sorry

end cos_150_eq_neg_sqrt3_div_2_l642_642161


namespace cyclists_meet_time_l642_642012

theorem cyclists_meet_time 
  (v1 v2 : ℕ) (C : ℕ) (h1 : v1 = 7) (h2 : v2 = 8) (hC : C = 675) : 
  C / (v1 + v2) = 45 :=
by
  sorry

end cyclists_meet_time_l642_642012


namespace meat_per_deer_is_200_l642_642006

namespace wolf_pack

def number_hunting_wolves : ℕ := 4
def number_additional_wolves : ℕ := 16
def meat_needed_per_day : ℕ := 8
def days : ℕ := 5

def total_wolves : ℕ := number_hunting_wolves + number_additional_wolves

def total_meat_needed : ℕ := total_wolves * meat_needed_per_day * days

def number_deer : ℕ := number_hunting_wolves

def meat_per_deer : ℕ := total_meat_needed / number_deer

theorem meat_per_deer_is_200 : meat_per_deer = 200 := by
  sorry

end wolf_pack

end meat_per_deer_is_200_l642_642006


namespace sum_of_roots_of_quadratic_l642_642751

theorem sum_of_roots_of_quadratic (a b c : ℚ) (h : a ≠ 0)
  (eq : a*(-48) + b*66 + c*195 = 0) : 
  let sum_of_roots := -(b / a)
  in sum_of_roots = (11 / 8) := by
sorry

end sum_of_roots_of_quadratic_l642_642751


namespace cost_price_of_cricket_bat_l642_642037

variable (CP_A CP_B SP_C : ℝ)

-- Conditions
def condition1 : CP_B = 1.20 * CP_A := sorry
def condition2 : SP_C = 1.25 * CP_B := sorry
def condition3 : SP_C = 234 := sorry

-- The statement to prove
theorem cost_price_of_cricket_bat : CP_A = 156 := sorry

end cost_price_of_cricket_bat_l642_642037


namespace chameleons_all_green_l642_642539

-- Problem Definition
def initial_state : ℕ × ℕ × ℕ := (7, 10, 17)

-- Total number of chameleons
def total_chameleons : ℕ := 34

-- Predicate to check if all chameleons are of the same color
def all_same_color (state : ℕ × ℕ × ℕ) : Prop :=
  state.1 = total_chameleons ∨ state.2 = total_chameleons ∨ state.3 = total_chameleons

-- The proof statement
theorem chameleons_all_green (state : ℕ × ℕ × ℕ) (h : state = initial_state) (h_same_color : all_same_color state) : 
  state.3 = total_chameleons :=
sorry

end chameleons_all_green_l642_642539


namespace no_false_statements_about_squares_l642_642072

noncomputable def square (s : Type) := s -> Prop

def equiangular (s : square) : Prop :=
  ∀ (a b c d : s), (a = 90 ∧ b = 90 ∧ c = 90 ∧ d = 90)

def rectangle (s : square) : Prop :=
  ∀ (a b c d : s), (a + b + c + d = 360 ∧ a = 90 ∧ b = 90 ∧ c = 90 ∧ d = 90)

def regular_polygon (s : square) : Prop :=
  ∀ (a b c d : s), (a = b ∧ b = c ∧ c = d ∧ d = a ∧ a = 90)

def equal_side_length (s : square) : Prop :=
  ∀ (a b c d : s), (a = b ∧ b = c ∧ c = d)

def congruent_if_same_side_length (s1 s2 : square) : Prop :=
  ∀ (a1 b1 c1 d1 a2 b2 c2 d2 : s1), (a1 = a2 ∧ b1 = b2 ∧ c1 = c2 ∧ d1 = d2)

theorem no_false_statements_about_squares
    (s : Type)
    [square s]
    (h1: equiangular s)
    (h2: rectangle s)
    (h3: regular_polygon s)
    (h4: equal_side_length s)
    (h5: ∀ s2 : square, congruent_if_same_side_length s s2) : false :=
sorry

end no_false_statements_about_squares_l642_642072


namespace balls_boxes_distribution_l642_642435

theorem balls_boxes_distribution:
  (number_of_distinct_partitions 7 4 = 11) :=
sorry

-- Helper definition
def number_of_distinct_partitions (n k : ℕ) : ℕ :=
∀ (p : List ℕ), p.sum = n ∧ p.length ≤ k ∧ p.sorted = p

end balls_boxes_distribution_l642_642435


namespace intersection_point_l642_642405

theorem intersection_point (n : ℕ) (h : 0 < n) : 
  (n, n^2) ∈ {p : ℝ × ℝ | p.2 = n * p.1} ∧
  (n, n^2) ∈ {p : ℝ × ℝ | p.2 = n^3 / p.1} :=
by sorry

end intersection_point_l642_642405


namespace cos_150_eq_neg_sqrt3_over_2_l642_642222

theorem cos_150_eq_neg_sqrt3_over_2 : Real.cos (150 * Real.pi / 180) = -Real.sqrt 3 / 2 := 
by 
  sorry

end cos_150_eq_neg_sqrt3_over_2_l642_642222


namespace pucelana_sequence_count_l642_642638

/-- A pucelana sequence is an increasing sequence of 16 consecutive odd numbers whose sum is a perfect cube.
    We need to prove that there are exactly 2 pucelana sequences with 3-digit numbers only. -/
theorem pucelana_sequence_count :
  let is_pucelana_sequence (seq : List ℤ) := List.Sort (<) seq ∧ List.length seq = 16 ∧ 
                                             List.sum seq = k^3 ∧ (List.last seq sorry - List.head seq sorry) = 30 in
  let three_digit (n : ℤ) := 100 ≤ n ∧ n ≤ 999 in
  let seqs := [seq | seq ∈ PucelanaSequences, is_pucelana_sequence seq] in
  (∃ (seq: List ℤ), 3 ≤ List.head seq sorry ∧ List.head seq sorry ≤ 6 ∧
                    three_digit (List.head seq sorry) ∧ 
                    (List.filter (λ seq, three_digit (List.head seq sorry)) seqs).length = 2) := sorry

end pucelana_sequence_count_l642_642638


namespace cosine_150_eq_neg_sqrt3_div_2_l642_642245

theorem cosine_150_eq_neg_sqrt3_div_2 :
  (∀ θ : ℝ, θ = real.pi / 6) →
  (∀ θ : ℝ, cos (real.pi - θ) = -cos θ) →
  cos (5 * real.pi / 6) = -real.sqrt 3 / 2 :=
by
  intros hθ hcos_identity
  sorry

end cosine_150_eq_neg_sqrt3_div_2_l642_642245


namespace min_value_f_solution_set_exists_x_f_eq_0_l642_642792

def f (x : ℝ) : ℝ := if x ≥ 0 then x^2 - 2*x else (-(x))^2 - 2*(-x)

theorem min_value_f : ∃ x : ℝ, f(x) = -1 := 
by sorry

theorem solution_set : {x : ℝ | f(x) > 0} = {x : ℝ | x < -2 ∨ x > 2} :=
by sorry

theorem exists_x_f_eq_0 : ∃ x : ℝ, f(x + 2) + f(-x) = 0 :=
by sorry

end min_value_f_solution_set_exists_x_f_eq_0_l642_642792


namespace least_number_div_condition_l642_642660

theorem least_number_div_condition (m : ℕ) : 
  (∃ k r : ℕ, m = 34 * k + r ∧ m = 5 * (r + 8) ∧ r < 34) → m = 162 := 
by
  sorry

end least_number_div_condition_l642_642660


namespace CD_square_length_l642_642543

noncomputable def parabola (x : ℝ) := -3 * x^2 + 2 * x + 5

theorem CD_square_length :
  (∃ (x : ℝ), parabola x = 2 * sqrt (5 / 3)) →
  (2 * sqrt (5 / 3))^2 + (2 * sqrt (5 / 3))^2 = 100 / 3 := by 
  sorry

end CD_square_length_l642_642543


namespace total_cost_price_of_fruits_l642_642670

theorem total_cost_price_of_fruits :
  (let SP_apple := 18
       SP_orange := 24
       SP_banana := 12
       CP_apple := SP_apple * 6 / 5
       CP_orange := SP_orange * 8 / 7
       CP_banana := SP_banana * 4 / 3
       totalCP_apples := 10 * CP_apple
       totalCP_oranges := 15 * CP_orange
       totalCP_bananas := 20 * CP_banana
   in totalCP_apples + totalCP_oranges + totalCP_bananas = 947.45) :=
by 
   sorry

end total_cost_price_of_fruits_l642_642670


namespace problem_part1_problem_part2_l642_642784

-- Define the sequences and conditions
variable {a b : ℕ → ℕ}
variable {S : ℕ → ℕ}
variable {T : ℕ → ℕ}
variable {d q : ℕ}
variable {b_initial : ℕ}

axiom geom_seq (n : ℕ) : b n = b_initial * q^n
axiom arith_seq (n : ℕ) : a n = a 1 + (n - 1) * d
axiom sum_seq (n : ℕ) : S n = n * (a 1 + a n) / 2

-- Problem conditions
axiom cond_geom_seq : b_initial = 2
axiom cond_geom_b2_b3 : b 2 + b 3 = 12
axiom cond_geom_ratio : q > 0
axiom cond_relation_b3_a4 : b 3 = a 4 - 2 * a 1
axiom cond_sum_S_11_b4 : S 11 = 11 * b 4

-- Theorem statement
theorem problem_part1 :
  (a n = 3 * n - 2) ∧ (b n = 2 ^ n) :=
  sorry

theorem problem_part2 :
  (T n = (3 * n - 2) / 3 * 4^(n + 1) + 8 / 3) :=
  sorry

end problem_part1_problem_part2_l642_642784


namespace pond_ratios_l642_642598

theorem pond_ratios (T A : ℕ) (h1 : T = 48) (h2 : A = 32) : A / (T - A) = 2 :=
by
  sorry

end pond_ratios_l642_642598


namespace paper_boat_time_proof_l642_642479

/-- A 50-meter long embankment exists along a river.
 - A motorboat that passes this embankment in 5 seconds while moving downstream.
 - The same motorboat passes this embankment in 4 seconds while moving upstream.
 - Determine the time in seconds it takes for a paper boat, which moves with the current, to travel the length of this embankment.
 -/
noncomputable def paper_boat_travel_time 
  (embankment_length : ℝ)
  (motorboat_length : ℝ)
  (time_downstream : ℝ)
  (time_upstream : ℝ) : ℝ :=
  let v_eff_downstream := embankment_length / time_downstream,
      v_eff_upstream := embankment_length / time_upstream,
      v_boat := (v_eff_downstream + v_eff_upstream) / 2,
      v_current := (v_eff_downstream - v_eff_upstream) / 2 in
  embankment_length / v_current

theorem paper_boat_time_proof :
  paper_boat_travel_time 50 10 5 4 = 40 := 
begin
  sorry,
end

end paper_boat_time_proof_l642_642479


namespace dogs_in_kennel_l642_642626

variable (C D : ℕ)

-- definition of the ratio condition 
def ratio_condition : Prop :=
  C * 4 = 3 * D

-- definition of the difference condition
def difference_condition : Prop :=
  C = D - 8

theorem dogs_in_kennel (h1 : ratio_condition C D) (h2 : difference_condition C D) : D = 32 :=
by 
  -- proof steps go here
  sorry

end dogs_in_kennel_l642_642626


namespace sum_series_eq_l642_642709

theorem sum_series_eq :
  ∑' n : ℕ, (4 * (n + 1) - 3) / 3^(n + 1) = 3 / 2 :=
sorry

end sum_series_eq_l642_642709


namespace sum_of_sides_of_triangle_l642_642605

theorem sum_of_sides_of_triangle : 
  ∀ (A B C : ℝ), 
    (∠A = 60 ∧ ∠B = 90 ∧ ∠C = 30) →
    (AC = 8 * Real.sqrt 3) →
    (sum_of_sides A B C = 24 * Real.sqrt 3 + 24) :=
begin
  intros A B C angle_cond AC_side,
  sorry
end

end sum_of_sides_of_triangle_l642_642605


namespace find_a1_l642_642855

-- Define the arithmetic sequence and conditions
variable (a_n S_n : ℕ → ℤ) (d a₁ : ℤ) (n : ℕ)

-- Given conditions
axiom Sum2016 : S_n 2016 = 2016
axiom DifferenceCondition : (S_n 2016 / 2016) - (S_n 16 / 16) = 2000

-- Prove the value of a₁
theorem find_a1 : a₁ = -2014 :=
by
  -- Define the relationship among sums, first term, and common difference in an arithmetic sequence
  have h₁ : ∀ n, (S_n n / n) = (a₁ + (↑n - 1) * d / 2), from sorry,
  -- Start deriving based on given conditions
  specialize h₁ 2016,
  rw Sum2016 at h₁,
  have : (a₁ + (2015 * d / 2)) = 1, from by
    rw ← h₁,
    ring,
    sorry,
  -- Solve for d, given the difference condition
  specialize h₁ 16,
  set d := 1, -- inferred from provided solution
  have h₂ : (S_n 2016) / 2016 - (S_n 16) / 16 = 2000, from DifferenceCondition,
  rw [Sum2016, ← h₁] at h₂,
  ring,
  sorry,
  -- Finally, prove a₁ = -2014,
  sorry

end find_a1_l642_642855


namespace cos_150_eq_neg_half_l642_642281

theorem cos_150_eq_neg_half : Real.cos (150 * Real.pi / 180) = -1 / 2 := 
by
  sorry

end cos_150_eq_neg_half_l642_642281


namespace cosine_of_angle_between_diagonals_l642_642662

noncomputable def vector_a : ℝ × ℝ × ℝ := (3, 1, 2)
noncomputable def vector_b : ℝ × ℝ × ℝ := (1, 2, -1)

def add_vectors (u v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (u.1 + v.1, u.2 + v.2, u.3 + v.3)

def subtract_vectors (u v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (u.1 - v.1, u.2 - v.2, u.3 - v.3)

def dot_product (u v : ℝ × ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2 + u.3 * v.3

def magnitude (u : ℝ × ℝ × ℝ) : ℝ :=
  Real.sqrt (u.1 ^ 2 + u.2 ^ 2 + u.3 ^ 2)

theorem cosine_of_angle_between_diagonals :
  let d1 := add_vectors vector_a vector_b in
  let d2 := subtract_vectors vector_b vector_a in
  Real.cos θ = dot_product d1 d2 / (magnitude d1 * magnitude d2) :=
sorry

end cosine_of_angle_between_diagonals_l642_642662


namespace find_value_l642_642410

variable (f : ℝ → ℝ)

-- Conditions
axiom odd_function : ∀ x, f (-x) = -f x
axiom periodic_function : ∀ x, f (x + 2) = f x
axiom explicit_form : ∀ (x : ℝ), 0 ≤ x ∧ x ≤ 1 → f x = 2 * x * (1 - x)

-- Theorem statement
theorem find_value : f (-5/2) = -1/2 :=
by
  -- Here would be the place to start the proof based on the above axioms
  sorry

end find_value_l642_642410


namespace average_price_of_pencil_correct_l642_642635

def average_price_of_pencil (n_pens n_pencils : ℕ) (total_cost pen_price : ℕ) : ℕ :=
  let pen_cost := n_pens * pen_price
  let pencil_cost := total_cost - pen_cost
  let avg_pencil_price := pencil_cost / n_pencils
  avg_pencil_price

theorem average_price_of_pencil_correct :
  average_price_of_pencil 30 75 450 10 = 2 :=
by
  -- The proof is omitted as per the instructions.
  sorry

end average_price_of_pencil_correct_l642_642635


namespace jack_piggy_bank_after_8_weeks_l642_642870

-- Conditions as definitions
def initial_amount : ℕ := 43
def weekly_allowance : ℕ := 10
def saved_fraction (x : ℕ) : ℕ := x / 2
def duration : ℕ := 8

-- Mathematical equivalent proof problem
theorem jack_piggy_bank_after_8_weeks : initial_amount + (duration * saved_fraction weekly_allowance) = 83 := by
  sorry

end jack_piggy_bank_after_8_weeks_l642_642870


namespace repeating_decimal_subtraction_l642_642026

theorem repeating_decimal_subtraction : 
  0.\overline{789} - 0.\overline{456} - 0.\overline{123} = (70 / 333 : ℚ) :=
sorry

end repeating_decimal_subtraction_l642_642026


namespace integral_solution_l642_642745

theorem integral_solution (y : ℝ) (h : y > 1) :
  (∫ x in 1..y, x * real.log x) = 1 / 4 → y = real.sqrt real.exp 1 := 
by 
  sorry

end integral_solution_l642_642745


namespace limit_solution_l642_642699

open Filter Real Topology

noncomputable def limit_problem (a x : ℝ) : Prop :=
  tendsto (λ h, (a^(x+h) + a^(x-h) - 2 * a^x) / h) (𝓝 0) (𝓝 0)

theorem limit_solution (a x : ℝ) : limit_problem a x :=
by sorry

end limit_solution_l642_642699


namespace cos_150_eq_neg_sqrt3_div_2_l642_642260

theorem cos_150_eq_neg_sqrt3_div_2 :
  let theta := 30 * Real.pi / 180 in
  let Q := 150 * Real.pi / 180 in
  cos Q = - (Real.sqrt 3 / 2) := by
    sorry

end cos_150_eq_neg_sqrt3_div_2_l642_642260


namespace num_girls_at_park_l642_642007

theorem num_girls_at_park (G : ℕ) (h1 : 11 + 50 + G = 3 * 25) : G = 14 := by
  sorry

end num_girls_at_park_l642_642007


namespace circle_center_and_radius_l642_642571

theorem circle_center_and_radius (x y : ℝ) : 
  (x^2 + y^2 - 6 * x = 0) → ((x - 3)^2 + (y - 0)^2 = 9) :=
by
  intro h
  -- The proof is left as an exercise.
  sorry

end circle_center_and_radius_l642_642571


namespace jack_piggy_bank_l642_642877

variable (initial_amount : ℕ) (weekly_allowance : ℕ) (weeks : ℕ)

-- Conditions
def initial_amount := 43
def weekly_allowance := 10
def weeks := 8

-- Weekly savings calculation: Jack saves half of his weekly allowance
def weekly_savings := weekly_allowance / 2

-- Total savings over the given period
def total_savings := weekly_savings * weeks

-- Final amount in the piggy bank after the given period
def final_amount := initial_amount + total_savings

-- Theorem to prove: Final amount in the piggy bank after 8 weeks is $83.00
theorem jack_piggy_bank : final_amount = 83 := by
  sorry

end jack_piggy_bank_l642_642877


namespace cos_150_eq_neg_sqrt3_div_2_l642_642309

open Real

theorem cos_150_eq_neg_sqrt3_div_2 :
  cos (150 * pi / 180) = - (sqrt 3 / 2) := 
sorry

end cos_150_eq_neg_sqrt3_div_2_l642_642309


namespace honey_remains_l642_642062

theorem honey_remains (h_total : ℝ) (h_eaten : ℝ) (h_remains : ℝ) (h_prod : h_total = 0.36) (h_bears : h_eaten = 0.05) : h_remains = 0.31 :=
by
  have h1 : h_remains = h_total - h_eaten := sorry
  rw [h_prod, h_bears] at h1
  exact h1

end honey_remains_l642_642062


namespace sum_f_eq_sqrt_three_l642_642768

def f (x : ℝ) : ℝ := 
  sin ((π / 3) * x + (π / 3)) - sqrt 3 * cos ((π / 3) * x + (π / 3))

theorem sum_f_eq_sqrt_three : 
  (∑ i in finset.range 2020, f (i + 1)) = sqrt 3 := 
sorry

end sum_f_eq_sqrt_three_l642_642768


namespace reciprocal_self_reciprocal_neg_self_l642_642620

theorem reciprocal_self (x : ℝ) : (x = 1 ∨ x = -1) ↔ (x = 1 / x) :=
by sorry

theorem reciprocal_neg_self (y : ℂ) : (y = Complex.i ∨ y = -Complex.i) ↔ (y = -1 / y) :=
by sorry

end reciprocal_self_reciprocal_neg_self_l642_642620


namespace area_A_I1_I2_l642_642517

noncomputable def midpoint (a b : ℝ × ℝ) : ℝ × ℝ :=
((a.1 + b.1) / 2, (a.2 + b.2) / 2)

noncomputable def incenter (a b c : ℝ × ℝ) : ℝ × ℝ :=
-- Calculate the incenter of a triangle from its vertices.
sorry

noncomputable def area_triangle (a b c : ℝ × ℝ) : ℝ :=
-- Calculate the area of a triangle from its vertices.
sorry

theorem area_A_I1_I2 :
  let A := (0, 0)
  let B := (13, 0)
  let C := (15 * (14 / 15), 14)
  let X := midpoint B C
  let I1 := incenter A B X
  let I2 := incenter A C X
  area_triangle A I1 I2 = 5.424 :=
sorry

end area_A_I1_I2_l642_642517


namespace indistinguishable_balls_ways_l642_642433

theorem indistinguishable_balls_ways :
  ∃ ways, ways = 11 ∧ ways = finset.card { p : multiset ℕ // multiset.card p ≤ 4 ∧ multiset.sum p = 7 } :=
by
  use 11
  split
  . refl
  . sorry

end indistinguishable_balls_ways_l642_642433


namespace functional_eq_solution_l642_642353

noncomputable def f (x : ℝ) : ℝ :=
  if h : x ≥ 2 then 0 else 2 / (2 - x)

theorem functional_eq_solution :
  (∀ x : ℝ, f(2) = 0) ∧
  (∀ x : ℝ, 0 ≤ x ∧ x < 2 → f(x) ≠ 0) ∧
  (∀ x y : ℝ, f(x * f(y)) * f(y) = f(x + y)) :=
by {
  -- Proof should go here, but it is omitted as per instructions.
  sorry
}

end functional_eq_solution_l642_642353


namespace trigonometric_identity_l642_642545

theorem trigonometric_identity :
  (cos 66 * cos 6 + cos 84 * cos 24) / (cos 65 * cos 5 + cos 85 * cos 25) = 1 := 
sorry

end trigonometric_identity_l642_642545


namespace cos_150_eq_neg_sqrt3_div_2_l642_642131

theorem cos_150_eq_neg_sqrt3_div_2 :
  ∃ θ : ℝ, θ = 150 ∧
           0 < θ ∧ θ < 180 ∧
           ∃ φ : ℝ, φ = 180 - θ ∧
           cos φ = Real.sqrt 3 / 2 ∧
           cos θ = - cos φ :=
  sorry

end cos_150_eq_neg_sqrt3_div_2_l642_642131


namespace circles_intersect_and_common_chord_l642_642423

theorem circles_intersect_and_common_chord :
  (∃ P : ℝ × ℝ, P.1 ^ 2 + P.2 ^ 2 - P.1 + P.2 - 2 = 0 ∧
                P.1 ^ 2 + P.2 ^ 2 = 5) ∧
  (∀ x y : ℝ, (x ^ 2 + y ^ 2 - x + y - 2 = 0 ∧ x ^ 2 + y ^ 2 = 5) →
              x - y - 3 = 0) ∧
  (∃ A B : ℝ × ℝ, A.1 ^ 2 + A.2 ^ 2 - A.1 + A.2 - 2 = 0 ∧
                   A.1 ^ 2 + A.2 ^ 2 = 5 ∧
                   B.1 ^ 2 + B.2 ^ 2 - B.1 + B.2 - 2 = 0 ∧
                   B.1 ^ 2 + B.2 ^ 2 = 5 ∧
                   (A.1 - B.1) ^ 2 + (A.2 - B.2) ^ 2 = 2) := sorry

end circles_intersect_and_common_chord_l642_642423


namespace cos_150_eq_neg_half_l642_642191

theorem cos_150_eq_neg_half :
  real.cos (150 * real.pi / 180) = -1 / 2 :=
sorry

end cos_150_eq_neg_half_l642_642191


namespace cos_150_eq_neg_sqrt3_div_2_l642_642254

theorem cos_150_eq_neg_sqrt3_div_2 :
  let theta := 30 * Real.pi / 180 in
  let Q := 150 * Real.pi / 180 in
  cos Q = - (Real.sqrt 3 / 2) := by
    sorry

end cos_150_eq_neg_sqrt3_div_2_l642_642254


namespace leak_empties_tank_in_14_hours_l642_642108

def tank_filling (time_to_fill : ℝ) : ℝ := 1 / time_to_fill

noncomputable def tank_leaking_time (pump_fill_time leak_fill_time : ℝ) : ℝ :=
  let P := tank_filling pump_fill_time in
  let L := P - tank_filling leak_fill_time in
  1 / L

theorem leak_empties_tank_in_14_hours :
  tank_leaking_time 7 14 = 14 :=
by
  sorry

end leak_empties_tank_in_14_hours_l642_642108


namespace ratio_of_areas_l642_642844

noncomputable theory

open EuclideanGeometry

def midpoint (B C M : Point) := dist B M = dist C M

theorem ratio_of_areas (A B C D O M E : Point) (h_convex : ConvexQuadrilateral A B C D)
  (h_intersect : collinear A C O ∧ collinear B D O) 
  (h_midpoint : midpoint B C M)
  (h_intersect': line_through M O ∧ line_through A D)
  (h_intersect_ME_AD : E ∈ line_through M O ∧ E ∈  line_through A D ) 
  : (dist A E / dist E D) = (area A B O / area C D O) :=
sorry

end ratio_of_areas_l642_642844


namespace triangles_are_similar_l642_642921

-- Definitions for points A, B, C, U, V represented by complex numbers a, b, c, u, v respectively
def point (α : Type*) := α

-- Given five points on a complex plane
variables (a b c u v : ℂ)

-- Conditions stating the similarity of triangles ΔAUV, ΔVBU, and ΔUVC
def directly_similar (p1 p2 p3 q1 q2 q3 : point ℂ) : Prop :=
  ∃ k : ℂ, (k ≠ 0) ∧ ((q1 - q2) = k * (p1 - p2)) ∧ ((q2 - q3) = k * (p2 - p3)) ∧ ((q3 - q1) = k * (p3 - p1))

axiom similarity_cond1 : directly_similar a u v u v c
axiom similarity_cond2 : directly_similar v b u u v c
axiom similarity_cond3 : directly_similar u v c u v c

-- Prove that ΔABC is directly similar to ΔAUV (implying to the directly similar triangles in conditions)
theorem triangles_are_similar :
  directly_similar a b c a u v :=
sorry

end triangles_are_similar_l642_642921


namespace inverse_function_fixed_point_l642_642577

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^(x-2) - 1

theorem inverse_function_fixed_point
  (a : ℝ) (h1 : 0 < a) (h2 : a ≠ 1) :
  ∃ g : ℝ → ℝ, (∀ y : ℝ, g (f a y) = y) ∧ g 0 = 2 :=
sorry

end inverse_function_fixed_point_l642_642577


namespace car_passing_kth_intersection_l642_642087

open_locale classical

noncomputable theory

def probability_passing_kth_intersection (n k : ℕ) : ℚ :=
  (2 * k * n - 2 * k^2 + 2 * k - 1 : ℚ) / (n^2 : ℚ)

theorem car_passing_kth_intersection (n k : ℕ) (h₁ : n > 0) (h₂ : k > 0) (h₃ : k ≤ n) :
  probability_passing_kth_intersection n k = (2 * k * n - 2 * k^2 + 2 * k - 1 : ℚ) / (n^2 : ℚ) :=
  sorry

end car_passing_kth_intersection_l642_642087


namespace cos_150_eq_neg_half_l642_642195

theorem cos_150_eq_neg_half :
  real.cos (150 * real.pi / 180) = -1 / 2 :=
sorry

end cos_150_eq_neg_half_l642_642195


namespace tennis_to_soccer_ratio_l642_642935

theorem tennis_to_soccer_ratio
  (total_balls : ℕ)
  (soccer_balls : ℕ)
  (basketball_offset : ℕ)
  (baseball_offset : ℕ)
  (volleyballs : ℕ)
  (tennis_balls : ℕ)
  (total_balls_eq : total_balls = 145)
  (soccer_balls_eq : soccer_balls = 20)
  (basketball_count : soccer_balls + basketball_offset = 20 + 5)
  (baseball_count : soccer_balls + baseball_offset = 20 + 10)
  (volleyballs_eq : volleyballs = 30)
  (accounted_balls : soccer_balls + (soccer_balls + basketball_offset) + (soccer_balls + baseball_offset) + volleyballs = 105)
  (tennis_balls_eq : tennis_balls = 145 - 105) :
  tennis_balls / soccer_balls = 2 :=
sorry

end tennis_to_soccer_ratio_l642_642935


namespace chessboard_selection_possible_l642_642564

theorem chessboard_selection_possible :
  ∀ (f : Fin 8 → Fin 8 → Fin 32), (∀ x : Fin 8, ∃ y₁ y₂ : Fin 8, y₁ ≠ y₂ ∧ f x y₁ = f x y₂) →
  ∃ (sel : Fin 8 → Fin 8 → Bool), 
    (∀ x y, sel x y = true → (∃ n : Fin 32, f x y = n ∧ ∀ (x' y'), x' ≠ x ∧ y' ≠ y → f x' y' ≠ n)) ∧
    (∀ x, ∃ y, sel x y = true) ∧ (∀ y, ∃ x, sel x y = true) :=
begin
  sorry
end

end chessboard_selection_possible_l642_642564


namespace vectors_parallel_iff_abs_x_eq_2_l642_642817

noncomputable def a : ℝ × ℝ := (1, x)
noncomputable def b : ℝ × ℝ := (x^2, 4x)

theorem vectors_parallel_iff_abs_x_eq_2 (x : ℝ) (h₀ : a ≠ (0, 0)) (h₁ : b ≠ (0, 0)) : 
  (|x| = 2) ↔ (x^2 / 1 = 4x / x) := 
sorry

end vectors_parallel_iff_abs_x_eq_2_l642_642817


namespace parallel_segments_lengths_l642_642994

theorem parallel_segments_lengths
  (AB CD EF GH : ℝ)
  (h_parallel : AB = CD ∧ CD = EF ∧ EF = GH)
  (h_AB_length : AB = 120)
  (h_CD_length : CD = 80)
  (h_GH_length : GH = 140)
  (h_similarity : \triangle_xyz_similar \triangle_abc \triangle_efc \triangle_gfc)
  : EF = 80 ∧ GH = 140 := 
by
  sorry

end parallel_segments_lengths_l642_642994


namespace paper_boat_time_proof_l642_642476

/-- A 50-meter long embankment exists along a river.
 - A motorboat that passes this embankment in 5 seconds while moving downstream.
 - The same motorboat passes this embankment in 4 seconds while moving upstream.
 - Determine the time in seconds it takes for a paper boat, which moves with the current, to travel the length of this embankment.
 -/
noncomputable def paper_boat_travel_time 
  (embankment_length : ℝ)
  (motorboat_length : ℝ)
  (time_downstream : ℝ)
  (time_upstream : ℝ) : ℝ :=
  let v_eff_downstream := embankment_length / time_downstream,
      v_eff_upstream := embankment_length / time_upstream,
      v_boat := (v_eff_downstream + v_eff_upstream) / 2,
      v_current := (v_eff_downstream - v_eff_upstream) / 2 in
  embankment_length / v_current

theorem paper_boat_time_proof :
  paper_boat_travel_time 50 10 5 4 = 40 := 
begin
  sorry,
end

end paper_boat_time_proof_l642_642476


namespace earnings_per_view_correct_l642_642993

variable (Voltaire_daily_avg Leila_weekly_earning : ℕ)
variable (twice : ℕ → ℕ)

def viewers_per_day := 50
def Leila_earnings := 350
def days_per_week := 7

def Voltaire_per_week (Voltaire_daily_avg : ℕ) (days_per_week : ℕ) : ℕ :=
  Voltaire_daily_avg * days_per_week

def Leila_per_week (twice : ℕ → ℕ) (Voltaire_per_week : ℕ) : ℕ :=
  twice Voltaire_per_week

def earnings_per_view (Leila_weekly_earning : ℕ) (Leila_per_week : ℕ) : ℕ :=
  Leila_weekly_earning / Leila_per_week

theorem earnings_per_view_correct : 
  earnings_per_view Leila_earnings (Leila_per_week (λ x, 2 * x) (Voltaire_per_week viewers_per_day days_per_week)) = 0.5 :=
by
  sorry

end earnings_per_view_correct_l642_642993


namespace sum_of_digits_of_greatest_prime_divisor_of_9999_l642_642998

theorem sum_of_digits_of_greatest_prime_divisor_of_9999 :
  let n : ℕ := 9999 in
  let p : ℕ := 101 in
  let sum_of_digits := (p / 100) + ((p / 10) % 10) + (p % 10) in
  (9,999 = 101 * 99) → (nat.is_prime 101) → sum_of_digits = 2 :=
by
  sorry

end sum_of_digits_of_greatest_prime_divisor_of_9999_l642_642998


namespace cos_150_eq_neg_sqrt3_div_2_l642_642256

theorem cos_150_eq_neg_sqrt3_div_2 :
  let theta := 30 * Real.pi / 180 in
  let Q := 150 * Real.pi / 180 in
  cos Q = - (Real.sqrt 3 / 2) := by
    sorry

end cos_150_eq_neg_sqrt3_div_2_l642_642256


namespace chandler_needs_34_weeks_to_afford_laptop_l642_642702

-- Define the given conditions
def laptop_cost : ℕ := 800
def birthday_money : ℕ := 60 + 40 + 25
def weekly_earnings : ℕ := 20

-- Define the main statement
theorem chandler_needs_34_weeks_to_afford_laptop : ∃ (x : ℕ), (birthday_money + x * weekly_earnings >= laptop_cost) ∧ (x = 34) :=
by 
    use 34
    split
    -- Show that birthday_money + 34 * weekly_earnings is at least laptop_cost
    {
        sorry
    }
    -- Show that x equals 34
    {
        rfl
    }

end chandler_needs_34_weeks_to_afford_laptop_l642_642702


namespace robert_balls_l642_642549

theorem robert_balls :
  ∀ (r_initial t j : ℕ),
    r_initial = 25 →
    t = 40 →
    j = 60 →
    let r_after_tim := r_initial + t / 2 in
    let r_after_jenny := r_after_tim + j / 3 in
    r_after_jenny = 65 :=
begin
  intros r_initial t j H_r_initial H_t H_j,
  let r_after_tim := r_initial + t / 2,
  let r_after_jenny := r_after_tim + j / 3,
  rw [H_r_initial, H_t, H_j],
  norm_num,
  sorry,
end

end robert_balls_l642_642549


namespace maximum_area_triangle_l642_642769

-- Definitions of points and lines based on conditions
def pointA : ℝ × ℝ := (0, 0)
def pointB : ℝ × ℝ := (1, 3)

-- Line passing through A
def lineA (m : ℝ) (x y : ℝ) : Prop := x + m * y = 0

-- Line passing through B
def lineB (m : ℝ) (x y : ℝ) : Prop := m * x - y - m + 3 = 0

-- Intersection point P does not coincide with A and B
def notCoincideA (x y : ℝ) : Prop := (x, y) ≠ pointA
def notCoincideB (x y : ℝ) : Prop := (x, y) ≠ pointB

-- Area of triangle PAB
def area_triangle (PA PB : ℝ) : ℝ := 1/2 * PA^2

-- Prove the maximum area of triangle PAB is 5/2
theorem maximum_area_triangle
  (m : ℝ)
  (P : ℝ × ℝ)
  (h1 : lineA m P.1 P.2)
  (h2 : lineB m P.1 P.2)
  (h3 : notCoincideA P.1 P.2)
  (h4 : notCoincideB P.1 P.2) :
  ∃ PA PB : ℝ, PA = sqrt 5 ∧ PB = sqrt 5 ∧ area_triangle PA PB = 5/2 :=
sorry

end maximum_area_triangle_l642_642769


namespace next_bell_ringing_time_l642_642094

theorem next_bell_ringing_time (post_office_interval train_station_interval town_hall_interval start_time : ℕ)
  (h1 : post_office_interval = 18)
  (h2 : train_station_interval = 24)
  (h3 : town_hall_interval = 30)
  (h4 : start_time = 9) :
  let lcm := Nat.lcm post_office_interval (Nat.lcm train_station_interval town_hall_interval)
  lcm + start_time = 15 := by
  sorry

end next_bell_ringing_time_l642_642094


namespace cos_150_degree_l642_642173

theorem cos_150_degree : Real.cos (150 * Real.pi / 180) = -Real.sqrt 3 / 2 :=
by
  sorry

end cos_150_degree_l642_642173


namespace operation_1_even_if_input_even_operation_2_even_if_input_even_operation_3_even_if_input_even_operation_4_not_always_even_if_input_even_operation_5_not_always_even_if_input_even_l642_642346

-- Define what an even integer is
def is_even (a : ℤ) : Prop := ∃ k : ℤ, a = 2 * k

-- Define the operations
def add_four (a : ℤ) := a + 4
def subtract_six (a : ℤ) := a - 6
def multiply_by_eight (a : ℤ) := a * 8
def divide_by_two_add_two (a : ℤ) := a / 2 + 2
def average_with_ten (a : ℤ) := (a + 10) / 2

-- The proof statements
theorem operation_1_even_if_input_even (a : ℤ) (h : is_even a) : is_even (add_four a) := sorry
theorem operation_2_even_if_input_even (a : ℤ) (h : is_even a) : is_even (subtract_six a) := sorry
theorem operation_3_even_if_input_even (a : ℤ) (h : is_even a) : is_even (multiply_by_eight a) := sorry
theorem operation_4_not_always_even_if_input_even (a : ℤ) (h : is_even a) : ¬ is_even (divide_by_two_add_two a) := sorry
theorem operation_5_not_always_even_if_input_even (a : ℤ) (h : is_even a) : ¬ is_even (average_with_ten a) := sorry

end operation_1_even_if_input_even_operation_2_even_if_input_even_operation_3_even_if_input_even_operation_4_not_always_even_if_input_even_operation_5_not_always_even_if_input_even_l642_642346


namespace batsman_average_after_17th_inning_l642_642035

theorem batsman_average_after_17th_inning
  (A : ℕ)
  (h1 : (16 * A + 88) / 17 = A + 3) :
  37 + 3 = 40 :=
by sorry

end batsman_average_after_17th_inning_l642_642035


namespace measure_angle_PMN_is_66_degrees_l642_642858

/--
In a labeled diagram, we are given:
1. An isosceles triangle \( \triangle PQR \) with \( \angle PQR = 48^\circ \).
2. A second isosceles triangle \( \triangle PMN \) with \( PM = PN \).
Prove that \( \angle PMN = 66^\circ \).
-/

theorem measure_angle_PMN_is_66_degrees
    (P Q R M N : Point)
    (h1 : angle P Q R = 48)
    (h2 : dist P R = dist R Q)
    (h3 : dist P M = dist P N)
    : angle P M N = 66 :=
sorry

end measure_angle_PMN_is_66_degrees_l642_642858


namespace position_of_21_over_19_in_sequence_l642_642418

def sequence_term (n : ℕ) : ℚ := (n + 3) / (n + 1)

theorem position_of_21_over_19_in_sequence :
  ∃ n : ℕ, sequence_term n = 21 / 19 ∧ n = 18 :=
by sorry

end position_of_21_over_19_in_sequence_l642_642418


namespace range_of_a_l642_642408

noncomputable def f (x a : ℝ) : ℝ := log 2 (x^2 - a*x + 3*a)

theorem range_of_a (a : ℝ) : 
  (∀ x Δx : ℝ, x ≥ 2 → Δx > 0 → f (x + Δx) a > f x a) → -4 < a ∧ a ≤ 4 := 
by
  intros h
  sorry

end range_of_a_l642_642408


namespace find_m_intersect_l642_642527

noncomputable def f (x : ℝ) : ℝ := x^3 - (1/2)^(x-2)

theorem find_m_intersect :
  (∃ (x0 : ℝ), f(x0) = 0 ∧ 1 < x0 ∧ x0 < 2) →
  (∃ (m : ℤ), (x0 : ℝ) →  m ≤ x0 ∧ x0 < m + 1) →
  m = 1 := by
  sorry

end find_m_intersect_l642_642527


namespace max_elements_subset_property_l642_642809

theorem max_elements_subset_property :
  let S := { n : ℕ | 1 ≤ n ∧ n ≤ 2001 }
  ∃ (T : set ℕ), T ⊆ S ∧ (∀ x y z ∈ T, x + y ≠ z) ∧ (∀ U : set ℕ, U ⊆ S ∧ (∀ x y z ∈ U, x + y ≠ z) → #U ≤ 1001) :=
by
  let S := { n : ℕ | 1 ≤ n ∧ n ≤ 2001 }
  exists ({ n : ℕ | 1001 ≤ n ∧ n ≤ 2001 })
  sorry

end max_elements_subset_property_l642_642809


namespace cyclic_quadrilaterals_exist_l642_642723

variable (C1 C2 C3 C4 : Circle)

theorem cyclic_quadrilaterals_exist : ∃ (infinitely_many : Set (Quadrilateral)), 
  (∀ q ∈ infinitely_many, q.isCyclic ∧
  q.sides 1 ∈ tangentLines C1 ∧
  q.sides 2 ∈ tangentLines C2 ∧
  q.sides 3 ∈ tangentLines C3 ∧
  q.sides 4 ∈ tangentLines C4) ∧ 
  infinite infinitely_many :=
by
  sorry

end cyclic_quadrilaterals_exist_l642_642723


namespace cos_150_eq_neg_sqrt3_div_2_l642_642307

open Real

theorem cos_150_eq_neg_sqrt3_div_2 :
  cos (150 * pi / 180) = - (sqrt 3 / 2) := 
sorry

end cos_150_eq_neg_sqrt3_div_2_l642_642307


namespace cos_150_eq_neg_sqrt3_over_2_l642_642229

theorem cos_150_eq_neg_sqrt3_over_2 : Real.cos (150 * Real.pi / 180) = -Real.sqrt 3 / 2 := 
by 
  sorry

end cos_150_eq_neg_sqrt3_over_2_l642_642229


namespace ratio_of_radii_l642_642955

theorem ratio_of_radii
  (R r : ℝ)
  (h₁ : (4 * real.pi * r^2) / (4 * real.pi * R^2) = 1 / 5) :
  R / r = real.sqrt 5 :=
by
  sorry

end ratio_of_radii_l642_642955


namespace math_paths_count_l642_642851

-- Definitions for grids, positions, and paths
def triangular_grid :=
  [
    [# ' ', ' ', ' ', 'M', ' ', ' ', ' '],
    [' ', ' ', 'A', 'T', 'A', ' ', ' '],
    ['H', 'T', 'H', 'T', 'H', 'T', 'H'],
    ['M', 'A', 'T', 'H', 'A', 'T', 'M']
  ]

def start_positions : list (nat × nat) := [(3, 0), (3, 6), (0, 3), (3, 3)]

def end_position : nat × nat := (3, 3)

def valid_move (from to : nat × nat) : bool :=
  let dx := (from.1.abs - to.1.abs),
      dy := (from.2.abs - to.2.abs)
  in (dx = 1 ∧ dy = 0) ∨ (dx = 0 ∧ dy = 1)

def valid_path (path : list (nat × nat)) : bool :=
  list.all (list.zip_with valid_move path (path.tail ++ [end_position])) id

noncomputable def num_paths : nat :=
  start_positions.foldl (λ acc start_pos =>
    acc + if valid_path [start_pos, (start_pos.1 - 1, start_pos.2), (start_pos.1 - 1, start_pos.2 + 1), end_position] then 1 else 0
  ) 0

theorem math_paths_count : num_paths = 32 := sorry

end math_paths_count_l642_642851


namespace centroid_triangle_condition_l642_642883

variables {A B C D E I : Type*}
variables [convex_quadrilateral A B C D]
variables [AD_not_parallel_BC A D B C]
variables [E := intersection AD BC]
variables [I := intersection AC BD]

theorem centroid_triangle_condition
  (h1 : parallel AB CD)
  (h2 : IC ^ 2 = IA * AC) :
  (centroid EDC = centroid IAB) ↔ (h1 ∧ h2) := 
sorry

end centroid_triangle_condition_l642_642883


namespace fill_25_cans_in_5_hours_l642_642664

def fill_rate (volume : ℕ) (time : ℕ) : ℕ :=
  volume / time

def total_volume_per_can (fill_fraction : ℚ) (can_capacity : ℕ) : ℕ :=
  (fill_fraction * can_capacity).toNat

def total_volume_filled (num_cans : ℕ) (volume_per_can : ℕ) : ℕ :=
  num_cans * volume_per_can

def time_to_fill_cans (total_volume : ℕ) (fill_rate : ℕ) : ℕ :=
  total_volume / fill_rate

theorem fill_25_cans_in_5_hours :
  let can_capacity := 8
  let num_cans_filled := 20
  let fill_fraction := 3 / 4
  let time_filled := 3
  let cans_to_fill := 25
  let full_can_volume := can_capacity
  let rate := fill_rate (total_volume_filled num_cans_filled (total_volume_per_can fill_fraction can_capacity)) time_filled
  time_to_fill_cans (total_volume_filled cans_to_fill full_can_volume) rate = 5 := by
  sorry

end fill_25_cans_in_5_hours_l642_642664


namespace solve_f_eq_x_l642_642368

noncomputable def f : ℝ → ℝ := sorry
noncomputable def f_inv : ℝ → ℝ := sorry

axiom f_inv_domain : ∀ (x : ℝ), 0 ≤ x ∧ x < 1 → 1 ≤ f_inv x ∧ f_inv x < 2
axiom f_inv_range : ∀ (x : ℝ), 2 < x ∧ x ≤ 4 → 0 ≤ f_inv x ∧ f_inv x < 1
-- Assumption that f is invertible on [0, 3]
axiom f_inv_exists : ∀ (x : ℝ), 0 ≤ x ∧ x ≤ 3 → ∃ y : ℝ, f y = x

theorem solve_f_eq_x : ∃ x : ℝ, 0 ≤ x ∧ x ≤ 3 ∧ f x = x → x = 2 :=
by
  sorry

end solve_f_eq_x_l642_642368


namespace sum_even_coefficients_l642_642825

theorem sum_even_coefficients : 
  ∀ (a : ℕ → ℕ), (∃ x : ℝ, (2 * x + 1)^6 = ∑ k in finset.range 7, (a k) * x ^ k) → 
  a 2 + a 4 + a 6 = 364 :=
by
  sorry

end sum_even_coefficients_l642_642825


namespace miner_distance_when_heard_blast_l642_642658

theorem miner_distance_when_heard_blast :
  ∀ (t_blast : ℕ) (miner_rate_yd_ps : ℕ) (sound_rate_ft_ps : ℕ),
  t_blast = 25 →
  miner_rate_yd_ps = 10 →
  sound_rate_ft_ps = 1120 →
  let t_miner := 28000 / (1090 : ℚ) in
  let dist_ft := miner_rate_yd_ps * 3 * t_miner in
  let dist_yd := dist_ft / 3 in
  dist_yd ≈ 257 :=
by
  intros
  have t_miner_eq : t_miner = 28000 / (1090 : ℚ) := rfl
  have dist_ft_eq : dist_ft = miner_rate_yd_ps * 3 * t_miner := rfl
  have dist_yd_eq : dist_yd = dist_ft / 3 := rfl
  sorry

end miner_distance_when_heard_blast_l642_642658


namespace solution_exists_l642_642559

theorem solution_exists (x : ℝ) :
  (|2 * x - 3| ≤ 3 ∧ (1 / x) < 1 ∧ x ≠ 0) ↔ (1 < x ∧ x ≤ 3) :=
by
  sorry

end solution_exists_l642_642559


namespace car_P_takes_less_time_l642_642590

noncomputable def VR : ℝ := 56.44102863722254
noncomputable def VP : ℝ := VR + 10
noncomputable def Distance : ℝ := 750
noncomputable def TR := Distance / VR
noncomputable def TP := Distance / VP
noncomputable def DeltaT := TR - TP

theorem car_P_takes_less_time :
  DeltaT ≈ 2.005257093475442 :=
by
  sorry

end car_P_takes_less_time_l642_642590


namespace initial_positions_2048_l642_642886

noncomputable def number_of_initial_positions (n : ℕ) : ℤ :=
  2 ^ n - 2

theorem initial_positions_2048 : number_of_initial_positions 2048 = 2 ^ 2048 - 2 :=
by
  sorry

end initial_positions_2048_l642_642886


namespace cos_150_eq_negative_cos_30_l642_642139

theorem cos_150_eq_negative_cos_30 :
  Real.cos (150 * Real.pi / 180) = - (Real.cos (30 * Real.pi / 180)) :=
by sorry

end cos_150_eq_negative_cos_30_l642_642139


namespace solve_quadratic_equation_l642_642799

theorem solve_quadratic_equation (x : ℝ) :
  2 * x * (x + 1) = x + 1 ↔ (x = -1 ∨ x = 1 / 2) :=
by
  sorry

end solve_quadratic_equation_l642_642799


namespace limit_of_series_l642_642050

theorem limit_of_series :
  (∃ s, (∀ n, s n = ∑ i in finset.range (n + 1), 2 * (i + 1)) ∧
       (∀ n, s n = n * (n + 1))) →
  (tendsto (λ n, (s n / (n + 3) - n)) at_top (𝓝 (-2))) :=
by
  sorry

end limit_of_series_l642_642050


namespace necessary_but_not_sufficient_condition_l642_642054

theorem necessary_but_not_sufficient_condition (x : ℝ) : (|x - 1| < 1 → 0 < x ∧ x < 2) ∧ (¬ (x > 0 → 0 < x ∧ x < 2)) :=
by
  split
  -- First part: Prove that |x - 1| < 1 implies 0 < x < 2
  intro h
  dsimp at h
  rw abs_lt at h
  cases h
  split
  linarith
  linarith
  -- Second part: Prove that x > 0 does not imply 0 < x < 2
  intro h
  dsimp at h
  linarith

end necessary_but_not_sufficient_condition_l642_642054


namespace math_problem_proof_l642_642424

noncomputable def circle_M_equation : (ℝ × ℝ) → Prop :=
  λ (p : ℝ × ℝ), (p.1 + 1)^2 + p.2^2 = 1 / 4

noncomputable def circle_N_equation : (ℝ × ℝ) → Prop :=
  λ (p : ℝ × ℝ), (p.1 - 1)^2 + p.2^2 = 49 / 4

noncomputable def curve_E_equation : (ℝ × ℝ) → Prop :=
  λ (p : ℝ × ℝ), (p.1^2 / 4) + (p.2^2 / 3) = 1

noncomputable def hyperbola_C_equation : (ℝ × ℝ) → Prop :=
  λ (p : ℝ × ℝ), p.1^2 - (p.2^2 / 3) = 1

axiom PQ_lambda_equation (λ₁ λ₂ k : ℝ) :
  (16 - k^2) * λ₁^2 + 32 * λ₁ + 16 - (16 / 3) * k^2 = 0 ∧
  (16 - k^2) * λ₂^2 + 32 * λ₂ + 16 - (16 / 3) * k^2 = 0 ∧
  λ₁ + λ₂ = -8 / 3

noncomputable def point_Q (x : ℝ) : (ℝ × ℝ) → Prop :=
  λ (q : ℝ × ℝ), q = (x, 0) ∨ q = (-x, 0)

noncomputable def problem_statement : Prop :=
  (∀ (p : ℝ × ℝ), circle_M_equation p → circle_N_equation p → curve_E_equation p) ∧
  (∀ (p : ℝ × ℝ), hyperbola_C_equation p) ∧
  ∃ k λ₁ λ₂, PQ_lambda_equation λ₁ λ₂ k ∧ (∀ (q : ℝ × ℝ), point_Q 2 q)
  
theorem math_problem_proof : problem_statement := sorry

end math_problem_proof_l642_642424


namespace circumradius_independence_of_AB_l642_642606

theorem circumradius_independence_of_AB (R r : ℝ) (l : line) (A B : point) (circle1 : circle) (circle2 : circle) (C D : point) 
  (h1 : circle1.radius = R) (h2 : circle2.radius = r) 
  (h3 : circle1.tangent_with_line l A) (h4 : circle2.tangent_with_line l B)
  (h5 : circle1.intersects circle2 C) (h6 : circle1.intersects circle2 D) :
  ∃ (ρ : ℝ), ρ = sqrt (R * r) := 
begin
  sorry
end

end circumradius_independence_of_AB_l642_642606


namespace maximize_championship_rounds_l642_642008

theorem maximize_championship_rounds :
  ∃ k : ℕ, 1 ≤ k ∧ k ≤ 8 ∧
    (∀ m : ℕ, 1 ≤ m ∧ m ≤ 8 → 
    (num_configurations m 8 8 8 < num_configurations k 8 8 8) ↔ m ≠ k) ∧ 
    k = 6 :=
by sorry

noncomputable def num_configurations (k n : ℕ) : ℕ :=
(n.factorial ^ 2) / (((n - k).factorial ^ 2) * k.factorial)

end maximize_championship_rounds_l642_642008


namespace problem1_problem2_problem3_l642_642114

noncomputable def f (x k : ℝ) := x * Real.exp (k * x)

theorem problem1 (k : ℝ) (h : k ≠ 0) : tangent_line_eq f (0, f 0 k) = fun x => x := 
sorry

theorem problem2 (k : ℝ) (h : k ≠ 0) :
  ((0 < k → monotonic_on (fun x => f x k) (Iio (-1 / k)) (Ioi (-1 / k))) ∧
   (0 > k → monotonic_on (fun x => f x k) (Iio (-1 / k)) (Ioi (-1 / k)))) :=
sorry

theorem problem3 (k : ℝ) (h : k ≠ 0) (h_mono : ∀ x ∈ Ioo (-1 : ℝ) 1, monotonic (fun x => f x k)) :
  k ∈ Icc (-1:ℝ) 0 ∪ Icc (0 : ℝ) 1 :=
sorry

end problem1_problem2_problem3_l642_642114


namespace find_positive_integer_l642_642354

def product_of_digits (n : Nat) : Nat :=
  -- Function to compute product of digits, assume it is defined correctly
  sorry

theorem find_positive_integer (x : Nat) (h : x > 0) :
  product_of_digits x = x * x - 10 * x - 22 ↔ x = 12 :=
by
  sorry

end find_positive_integer_l642_642354


namespace sampling_methods_correct_l642_642021

def condition1 : Prop :=
  ∃ yogurt_boxes : ℕ, yogurt_boxes = 10 ∧ ∃ sample_boxes : ℕ, sample_boxes = 3

def condition2 : Prop :=
  ∃ rows seats_per_row attendees sample_size : ℕ,
    rows = 32 ∧ seats_per_row = 40 ∧ attendees = rows * seats_per_row ∧ sample_size = 32

def condition3 : Prop :=
  ∃ liberal_arts_classes science_classes total_classes sample_size : ℕ,
    liberal_arts_classes = 4 ∧ science_classes = 8 ∧ total_classes = liberal_arts_classes + science_classes ∧ sample_size = 50

def simple_random_sampling (s : Prop) : Prop := sorry -- definition for simple random sampling
def systematic_sampling (s : Prop) : Prop := sorry -- definition for systematic sampling
def stratified_sampling (s : Prop) : Prop := sorry -- definition for stratified sampling

theorem sampling_methods_correct :
  (condition1 → simple_random_sampling condition1) ∧
  (condition2 → systematic_sampling condition2) ∧
  (condition3 → stratified_sampling condition3) :=
by {
  sorry
}

end sampling_methods_correct_l642_642021


namespace simplify_fraction_l642_642554

theorem simplify_fraction (b : ℝ) (h : b ≠ 1) : 
  (b - 1) / (b + b / (b - 1)) = (b - 1) ^ 2 / b ^ 2 := 
by {
  sorry
}

end simplify_fraction_l642_642554


namespace common_tangent_range_l642_642403

open Real

theorem common_tangent_range (a : ℝ) (hₐ : 0 < a) :
  (∃ (x₁ x₂ : ℝ), y₁ = a * x₁^2 ∧ y₂ = exp x₂ ∧ deriv (λ x, a * x^2) x₁ = deriv (exp) x₂ ∧ y₁ = y₂) →
  a ∈ set.Ici (exp 2 / 4) :=
by sorry

end common_tangent_range_l642_642403


namespace surface_area_of_circumscribed_sphere_of_tetrahedron_l642_642400

theorem surface_area_of_circumscribed_sphere_of_tetrahedron 
  (a : ℝ) (h : a = real.sqrt 2) : 
  let r := real.sqrt 3 / 2 in
  4 * real.pi * r^2 = 3 * real.pi :=
by
  sorry

end surface_area_of_circumscribed_sphere_of_tetrahedron_l642_642400


namespace phone_number_fraction_l642_642113

theorem phone_number_fraction : 
  let total_valid_numbers := 6 * (10^6)
  let valid_numbers_with_conditions := 10^5
  valid_numbers_with_conditions / total_valid_numbers = 1 / 60 :=
by sorry

end phone_number_fraction_l642_642113


namespace speed_of_goods_train_l642_642036

theorem speed_of_goods_train 
  (t₁ t₂ v_express : ℝ)
  (h1 : v_express = 90) 
  (h2 : t₁ = 6) 
  (h3 : t₂ = 4)
  (h4 : v_express * t₂ = v * (t₁ + t₂)) : 
  v = 36 :=
by
  sorry

end speed_of_goods_train_l642_642036


namespace find_DG_l642_642934

theorem find_DG (a b S k l DG BC : ℕ) (h1: S = 17 * (a + b)) (h2: S % a = 0) (h3: S % b = 0) (h4: a = S / k) (h5: b = S / l) (h6: BC = 17) (h7: (k - 17) * (l - 17) = 289) : DG = 306 :=
by
  sorry

end find_DG_l642_642934


namespace pyramid_base_edge_length_l642_642565

-- Prove that the edge-length of the base of the pyramid is as specified
theorem pyramid_base_edge_length
  (r h : ℝ)
  (hemisphere_radius : r = 3)
  (pyramid_height : h = 8)
  (tangency_condition : true) : true :=
by
  sorry

end pyramid_base_edge_length_l642_642565


namespace cos_150_eq_neg_sqrt3_div_2_l642_642129

theorem cos_150_eq_neg_sqrt3_div_2 :
  ∃ θ : ℝ, θ = 150 ∧
           0 < θ ∧ θ < 180 ∧
           ∃ φ : ℝ, φ = 180 - θ ∧
           cos φ = Real.sqrt 3 / 2 ∧
           cos θ = - cos φ :=
  sorry

end cos_150_eq_neg_sqrt3_div_2_l642_642129


namespace cos_150_eq_neg_sqrt3_div_2_l642_642137

theorem cos_150_eq_neg_sqrt3_div_2 :
  ∃ θ : ℝ, θ = 150 ∧
           0 < θ ∧ θ < 180 ∧
           ∃ φ : ℝ, φ = 180 - θ ∧
           cos φ = Real.sqrt 3 / 2 ∧
           cos θ = - cos φ :=
  sorry

end cos_150_eq_neg_sqrt3_div_2_l642_642137


namespace correct_statements_l642_642111

-- Definitions and conditions
variables {P A B C E F : Type}
variables {line_PA : P ⟶ A}
variables {plane_circle_O : Type}
variables {circle_O : Type}
variables {diameter_AB : A = B}
variables {point_C_on_circle_O : C}
variables {perpendicular_AE_PC : perpendicular A E}
variables {perpendicular_AF_PB : perpendicular A F}

-- Statements in the problem
def AE_perpendicular_BC : Prop := perpendicular A E B C
def EF_perpendicular_PB : Prop := perpendicular E F P B
def AE_perpendicular_plane_PBC : Prop := perpendicular A E (plane P B C)

-- Proof goals
theorem correct_statements :
  AE_perpendicular_BC ∧ EF_perpendicular_PB ∧ AE_perpendicular_plane_PBC :=
by
  sorry

end correct_statements_l642_642111


namespace cyclist_trip_time_l642_642649

variable (a v : ℝ)
variable (h1 : a / v = 5)

theorem cyclist_trip_time
  (increase_factor : ℝ := 1.25) :
  (a / (2 * v) + a / (2 * (increase_factor * v)) = 4.5) :=
sorry

end cyclist_trip_time_l642_642649


namespace solve_for_n_l642_642940

theorem solve_for_n (n : ℕ) (h : 9^n * 9^n * 9^n * 9^n = 81^n) : n = 0 :=
by
  sorry

end solve_for_n_l642_642940


namespace correct_propositions_l642_642106

-- Definitions of the conditions
def proposition1 (a b c : ℝ) : Prop := (ac^2 > bc^2) → (a > b)
def proposition2 (α β : ℝ) : Prop := (sin α = sin β) → (α = β)
def are_lines_parallel (a : ℝ) : Prop := ∀ (x y : ℝ), (x - 2 * a * y = 1) → (2 * x - 2 * a * y = 1)
def is_even_function (f : ℝ → ℝ) : Prop := ∀x : ℝ, f (|x|) = f x

-- Logarithm base 2 function
def log_base_2 (x : ℝ) : ℝ := Real.log x / Real.log 2

-- Propositions provided in the problem
def prop3 (a : ℝ) : Prop := (a = 0) ↔ are_lines_parallel a
def prop4 (x : ℝ) : Prop := is_even_function log_base_2

-- Theorem stating the correct proposition numbers are 3 and 4
theorem correct_propositions : prop3 ∧ prop4 :=
by
  sorry -- Proof to be filled in

end correct_propositions_l642_642106


namespace matrix_solution_correct_l642_642356

open Matrix

def N : Matrix (Fin 2) (Fin 2) ℚ := ![![3, -7/3], ![4, -1/3]]

def v1 : Fin 2 → ℚ := ![4, 0]
def v2 : Fin 2 → ℚ := ![2, 3]

def result1 : Fin 2 → ℚ := ![12, 16]
def result2 : Fin 2 → ℚ := ![-1, 7]

theorem matrix_solution_correct :
  (mulVec N v1 = result1) ∧ 
  (mulVec N v2 = result2) := by
  sorry

end matrix_solution_correct_l642_642356


namespace largest_n_employees_in_same_quarter_l642_642843

theorem largest_n_employees_in_same_quarter (n : ℕ) (h1 : 72 % 4 = 0) (h2 : 72 / 4 = 18) : 
  n = 18 :=
sorry

end largest_n_employees_in_same_quarter_l642_642843


namespace tangent_line_at_a_minus_two_range_of_a_e_to_the_a_minus_two_vs_a_to_the_e_minus_two_l642_642800

-- Statement for (Ⅰ)
theorem tangent_line_at_a_minus_two : 
  ∀ (x : ℝ), 
  (∀ (a : ℝ), a = -2 → (f : ℝ → ℝ) (f x = log x + x - 1) → (∀ (f'(x) = 1/x + 1)), 
  (∀ (x = 1), ∀ (k = 2), ( ∀ (y = 2 * x - 2) → tangent_line_eq_y))
:= sorry

-- Statement for (Ⅱ)(i)
theorem range_of_a :
  ∀ (x : ℝ), 
  (∀ (a : ℝ), 
    (∀ (f x = log x - (1/2) * a * (x - 1)), 
    (∀ (x ∈ (1 : ℝ) + ∞), f x < 0) → a ∈ [2, ∞))
:= sorry

-- Statement for (Ⅱ)(ii)
theorem e_to_the_a_minus_two_vs_a_to_the_e_minus_two :
  ∀ (a : ℝ), 
    a ∈ [2, +∞) →
    (if a ∈ [2, e], e^(a-2) < a^(e-2)) ∧
    (if a = e, e^(a-2) = a^(e-2)) ∧
    (if a ∈ (e, +∞), e^(a-2) > a^(e-2))
:= sorry

end tangent_line_at_a_minus_two_range_of_a_e_to_the_a_minus_two_vs_a_to_the_e_minus_two_l642_642800


namespace point_not_in_third_quadrant_l642_642448

theorem point_not_in_third_quadrant (A : ℝ × ℝ) (h : A.snd = -A.fst + 8) : ¬ (A.fst < 0 ∧ A.snd < 0) :=
sorry

end point_not_in_third_quadrant_l642_642448


namespace measure_of_angle_XPM_l642_642866

-- Definitions based on given conditions
variables (X Y Z L M N P : Type)
variables (a b c : ℝ) -- Angles are represented in degrees
variables [DecidableEq X] [DecidableEq Y] [DecidableEq Z]

-- Triangle XYZ with angle bisectors XL, YM, and ZN meeting at incenter P
-- Given angle XYZ in degrees
def angle_XYZ : ℝ := 46

-- Incenter angle properties
axiom angle_bisector_XL (angle_XYP : ℝ) : angle_XYP = angle_XYZ / 2
axiom angle_bisector_YM (angle_YXP : ℝ) : ∃ (angle_YXZ : ℝ), angle_YXP = angle_YXZ / 2

-- The proposition we need to prove
theorem measure_of_angle_XPM : ∃ (angle_XPM : ℝ), angle_XPM = 67 := 
by {
  sorry
}

end measure_of_angle_XPM_l642_642866


namespace cos_150_eq_neg_sqrt3_div_2_l642_642126

theorem cos_150_eq_neg_sqrt3_div_2 :
  ∃ θ : ℝ, θ = 150 ∧
           0 < θ ∧ θ < 180 ∧
           ∃ φ : ℝ, φ = 180 - θ ∧
           cos φ = Real.sqrt 3 / 2 ∧
           cos θ = - cos φ :=
  sorry

end cos_150_eq_neg_sqrt3_div_2_l642_642126


namespace sequence_a_is_permutation_l642_642972

-- Define the sequence {a_n}
noncomputable def a (n : ℕ) : ℕ :=
  if h : n = 1 then 1 else 
  have ih : Π k < n, a k < n -> k ≠ n, from sorry,
  let S : finset ℕ := finset.univ.filter (λ x : ℕ, 
    ∀ j < n, (x ≠ a j) ∧ is_int (finset.sum (finset.range (n-1)).image 
      (λ k, √(a k + √(a k.pred + ... + √(a 2 + √(a 1))))))) in
  finset.min' S sorry

-- Define what it means for a sequence to be a permutation of all positive integers
def is_permutation_of_nat (seq : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, ∃ m : ℕ, seq m = n

-- State the theorem
theorem sequence_a_is_permutation : is_permutation_of_nat a :=
sorry

end sequence_a_is_permutation_l642_642972


namespace eigenvalues_are_one_l642_642506

variables {n : ℕ}
variables (A : Matrix (Fin n) (Fin n) ℂ)
variable (invertibleA : invertible A)

-- Additional condition definition
def special_condition (m : ℕ) : Prop :=
  ∃ (k_m : ℕ) (B_m : Matrix (Fin n) (Fin n) ℂ), invertible B_m ∧ A^(k_m * m) = B_m * A * B_m⁻¹

theorem eigenvalues_are_one 
  (condition : ∀ (m : ℕ), m > 0 → special_condition A m) :
  ∀ λ : ℂ, λ ∈ (A.eigenvalues) → λ = 1 :=
sorry

end eigenvalues_are_one_l642_642506


namespace cos_150_eq_neg_half_l642_642274

theorem cos_150_eq_neg_half : Real.cos (150 * Real.pi / 180) = -1 / 2 := 
by
  sorry

end cos_150_eq_neg_half_l642_642274


namespace find_eccentricity_l642_642332

variables {a b : ℝ} (h : a > 0 ∧ b > 0)
def hyperbola : set (ℝ × ℝ) := {p | let x := p.1, y := p.2 in (x^2) / (a^2) - (y^2) / (b^2) = 1}
def isOnHyperbola (p : ℝ × ℝ) : Prop := p ∈ hyperbola h

theorem find_eccentricity (p : ℝ × ℝ) (center : p ∈ hyperbola h) 
(tangent : let M := p in let F := (a * sqrt(1 + 1/2), 0) in M.2 = F.1)
(intersect_y : let M := p in let PQ := λ y, y = M.2 → M.1 = a * sqrt(1/3 * 1)) 
(equilateral : ∀ PQ, intersection_points PQ = { P | (P,M,P.1) is equiateral △ }):
∃ e : ℝ, e = √3 :=
sorry

end find_eccentricity_l642_642332


namespace cos_150_eq_neg_half_l642_642196

theorem cos_150_eq_neg_half :
  real.cos (150 * real.pi / 180) = -1 / 2 :=
sorry

end cos_150_eq_neg_half_l642_642196


namespace cos_150_eq_neg_sqrt3_over_2_l642_642228

theorem cos_150_eq_neg_sqrt3_over_2 : Real.cos (150 * Real.pi / 180) = -Real.sqrt 3 / 2 := 
by 
  sorry

end cos_150_eq_neg_sqrt3_over_2_l642_642228


namespace problem_solution_l642_642618

theorem problem_solution : 
  (1 * 2 * 3 * 4 * 5 * 6 * 7 * 10) / (1^2 + 2^2 + 3^2 + 4^2 + 5^2 + 6^2 + 7^2) = 360 :=
by
  sorry

end problem_solution_l642_642618


namespace paper_boat_travel_time_l642_642480

-- Defining the conditions as constants
def distance_embankment : ℝ := 50
def speed_downstream : ℝ := 10
def speed_upstream : ℝ := 12.5

-- Definitions for the speeds of the boat and current
noncomputable def v_boat : ℝ := (speed_upstream + speed_downstream) / 2
noncomputable def v_current : ℝ := (speed_downstream - speed_upstream) / 2

-- Statement to prove the time taken for the paper boat
theorem paper_boat_travel_time :
  (distance_embankment / v_current) = 40 := by
  sorry

end paper_boat_travel_time_l642_642480


namespace functional_equation_solution_l642_642962

theorem functional_equation_solution {f : ℝ → ℝ} (h : ∀ x ≠ 1, (x - 1) * f (x + 1) - f x = x) :
    ∀ x, f x = 1 + 2 * x :=
by
  sorry

end functional_equation_solution_l642_642962


namespace waiter_serves_all_guests_if_and_only_if_even_l642_642458

theorem waiter_serves_all_guests_if_and_only_if_even {n : ℕ} (h : n ≥ 2) :
  (∃ start : ℕ, ∀ i : ℕ, i < n → ∃ k : ℕ, (start - k * card_order i).nat_abs % n = i) ↔ even n :=
sorry

end waiter_serves_all_guests_if_and_only_if_even_l642_642458


namespace determine_a_l642_642900

variable (a : ℝ)

def f (x : ℝ) : ℝ := real.exp x - a * real.exp (-x)

def f_prime (x : ℝ) : ℝ := real.exp x + a * real.exp (-x)

-- Given: f is a twice differentiable function
-- Given: f'' is an odd function
theorem determine_a (h : ∀ x, f_prime a (-x) = - f_prime a x) : a = -1 :=
sorry

end determine_a_l642_642900


namespace minimum_c_is_1501_l642_642544

noncomputable def findMinimumC (a b c : ℕ) : ℕ :=
if a < b ∧ b < c ∧ a + b = c 
   ∧ (∃ x y : ℤ, 3 * x + y = 3005 ∧ y = abs (x - a) + abs (x - b) + abs (x - c)
   ∧ ∀ z, z ≠ x → (3 * z + abs (z - a) + abs (z - b) + abs (z - c)) ≠ 3005) then c
else 0

theorem minimum_c_is_1501 : ∃ a b c : ℕ, a < b ∧ b < c ∧ a + b = c 
  ∧ (∃ x y : ℤ, 3 * x + y = 3005 ∧ y = abs (x - a) + abs (x - b) + abs (x - c)
  ∧ ∀ z, z ≠ x → (3 * z + abs (z - a) + abs (z - b) + abs (z - c)) ≠ 3005)
  ∧ findMinimumC a b c = 1501 :=
sorry

end minimum_c_is_1501_l642_642544


namespace nth_operation_2011_l642_642805

-- Definitions of the operations results
def operation1 : ℕ := 2^3 + 5^3
def operation2 : ℕ := 1^3 + 3^3 + 3^3
def operation3 : ℕ := 5^3 + 5^3

-- Definitions for cyclic operations
def cycle : ℕ → ℕ
| 0     := operation1
| 1     := operation2
| 2     := operation3
| (n+3) := cycle n

-- Theorem to state the proof goal
theorem nth_operation_2011 : cycle 2010 = 133 :=
by 
  -- skipping the proof with sorry
  sorry

end nth_operation_2011_l642_642805


namespace probability_top_card_is_joker_l642_642100

def deck_size : ℕ := 54
def joker_count : ℕ := 2

theorem probability_top_card_is_joker :
  (joker_count : ℝ) / (deck_size : ℝ) = 1 / 27 :=
by
  sorry

end probability_top_card_is_joker_l642_642100


namespace length_PM_l642_642456

variable {P Q R M : Type} [MetricSpace P] [MetricSpace Q] [MetricSpace R]

def triangle_PQR (P Q R : Type) [MetricSpace P] [MetricSpace Q] [MetricSpace R] : Prop :=
  dist P Q = 46 ∧ dist Q R = 46 ∧ dist P R = 40

def midpoint (M Q R : Type) [MetricSpace Q] [MetricSpace R] [AddGroup M] : Prop :=
  dist Q M = dist R M

theorem length_PM {P Q R M : Type} [MetricSpace P] [MetricSpace Q] [MetricSpace R] :
  triangle_PQR P Q R →
  midpoint M Q R →
  dist P M = Real.sqrt 1587 :=
by
  intros hPQR hMid
  sorry

end length_PM_l642_642456


namespace probability_of_special_draw_l642_642977

theorem probability_of_special_draw 
    (deck : Finset Card) 
    (cards_dealt : List Card)
    (first_card_is_5 : card_is_5 (cards_dealt.nth 0))
    (second_card_is_heart : card_is_heart (cards_dealt.nth 1))
    (third_card_is_ace : card_is_ace (cards_dealt.nth 2)) :
    probability_special_draw(deck, cards_dealt) = 1 / 650 := by
  sorry

-- Define predicates for card characteristics
def Card := nat -- This is a simplification. You would define a card type with suits and ranks.
def card_is_5 (c : Card) : Prop := sorry -- Define the condition for the card to be a 5.
def card_is_heart (c : Card) : Prop := sorry -- Define the condition for the card to be a heart suit.
def card_is_ace (c : Card) : Prop := sorry -- Define the condition for the card to be an ace.

-- Define the function to calculate the probability
def probability_special_draw (deck : Finset Card) (cards_dealt : List Card) : ℚ := sorry


end probability_of_special_draw_l642_642977


namespace ellipse_equation_and_isosceles_triangle_l642_642777

def ellipse (C : ℝ → ℝ → Prop) : Prop := ∃ a b : ℝ, a > b ∧ b > 0 ∧ ∀ x y, C x y ↔ (x^2) / (a^2) + (y^2) / (b^2) = 1

def line_through_point (l : ℝ → ℝ → Prop) (m : ℝ × ℝ) : Prop := ∃ a b : ℝ, ∀ x y, l x y ↔ b * x - a * y = 0 ∧ l (m.1) (m.2)

def parallel_lines (l m : ℝ → ℝ → Prop) : Prop :=
∃ a b t : ℝ, ∀ x y, l x y ↔ b * x - a * y = 0 ∧ m x y ↔ y = (1/2) * x + t

def points_intersect (m : ℝ → ℝ → Prop) (C : ℝ → ℝ → Prop) (A B : ℝ × ℝ) : Prop :=
C (A.1) (A.2) ∧ m (A.1) (A.2) ∧ C (B.1) (B.2) ∧ m (B.1) (B.2)

def isosceles_triangle (E F M : ℝ × ℝ) : Prop :=
(E.2 = 0) ∧ (F.2 = 0) ∧ (M.2 ≠ 0) ∧ (E.1 - M.1 = M.1 - F.1)

theorem ellipse_equation_and_isosceles_triangle :
  (∃ (C : ℝ → ℝ → Prop) (l : ℝ → ℝ → Prop) (m : ℝ → ℝ → Prop) (M : ℝ × ℝ) (A B E F : ℝ × ℝ),
    ellipse C ∧ line_through_point l M ∧ M = (2 * real.sqrt 2, real.sqrt 2) ∧
    parallel_lines l m ∧ points_intersect m C A B ∧
    (∃ a b : ℝ, a = 2 * b ∧ a^2 = 16 ∧ b^2 = 4 ∧ C = λ x y, (x^2) / 16 + (y^2) / 4 = 1) ∧
    (isosceles_triangle E F M)
  ) :=
sorry

end ellipse_equation_and_isosceles_triangle_l642_642777


namespace min_value_f_solution_set_exists_x_f_eq_0_l642_642793

def f (x : ℝ) : ℝ := if x ≥ 0 then x^2 - 2*x else (-(x))^2 - 2*(-x)

theorem min_value_f : ∃ x : ℝ, f(x) = -1 := 
by sorry

theorem solution_set : {x : ℝ | f(x) > 0} = {x : ℝ | x < -2 ∨ x > 2} :=
by sorry

theorem exists_x_f_eq_0 : ∃ x : ℝ, f(x + 2) + f(-x) = 0 :=
by sorry

end min_value_f_solution_set_exists_x_f_eq_0_l642_642793


namespace translate_point_correct_l642_642986

-- Define initial point
def initial_point : ℝ × ℝ := (0, 1)

-- Define translation downward
def translate_down (point : ℝ × ℝ) (units : ℝ) : ℝ × ℝ :=
  (point.1, point.2 - units)

-- Define translation to the left
def translate_left (point : ℝ × ℝ) (units : ℝ) : ℝ × ℝ :=
  (point.1 - units, point.2)

-- Define the expected resulting point
def expected_point : ℝ × ℝ := (-4, -1)

-- Lean statement to prove the equivalence
theorem translate_point_correct :
  (translate_left (translate_down initial_point 2) 4) = expected_point :=
by 
  -- Here, we would prove it step by step if required
  sorry

end translate_point_correct_l642_642986


namespace circumcircle_fixed_point_l642_642969

theorem circumcircle_fixed_point (A P B C : Point) (M : Point)
  (angle_ABP_eq_angle_MAB : ∠ABP = ∠MAB)
  (angle_ACP_eq_angle_MAC : ∠ACP = ∠MAC)
  (midpoint_M : M = midpoint B C)
  (fixed_point_P : P ≠ A) :
  ∀ B C, ∃ O r, circumcircle O r A B C ∧ circumcircle O r A B C (fixed_point_P) :=
by 
  sorry

end circumcircle_fixed_point_l642_642969


namespace twice_perimeter_of_square_l642_642672

theorem twice_perimeter_of_square (s : ℝ) (h : s^2 = 625) : 2 * 4 * s = 200 :=
by sorry

end twice_perimeter_of_square_l642_642672


namespace erase_sum_not_divisible_by_11_l642_642541

theorem erase_sum_not_divisible_by_11 :
  let seq := list.range 11 |>.map (λ n, 10 * n + 4)
  let S := seq.sum
  (S % 11 = 0) →
  ¬∃ (erase_step : list ℕ → list (list ℕ → list ℕ)),
    erase_step.length = 4 ∧ 
    ∀ (i : ℕ) (l : list ℕ), i < 4 → (S - ((erase_step.take (i+1)).map (λ f, (f seq).sum)).sum) % 11 = 0 :=
by
  let seq := list.range 11 |>.map (λ n, 10 * n + 4)
  let S := seq.sum
  intros
  sorry

end erase_sum_not_divisible_by_11_l642_642541


namespace cos_150_degree_l642_642183

theorem cos_150_degree : Real.cos (150 * Real.pi / 180) = -Real.sqrt 3 / 2 :=
by
  sorry

end cos_150_degree_l642_642183


namespace anna_lemonade_earnings_l642_642110

theorem anna_lemonade_earnings (glasses_plain : ℕ) (price_per_glass : ℝ) (total_strawberry : ℝ) :
  glasses_plain = 36 → price_per_glass = 0.75 → total_strawberry = 16 → 
  glasses_plain * price_per_glass - total_strawberry = 11 :=
by
  assume h1 : glasses_plain = 36
  assume h2 : price_per_glass = 0.75
  assume h3 : total_strawberry = 16
  sorry

end anna_lemonade_earnings_l642_642110


namespace volume_of_pyramid_SPQR_l642_642925

variable (P Q R S : Type)
variable (SP SQ SR : ℝ)
variable (is_perpendicular_SP_SQ : SP * SQ = 0)
variable (is_perpendicular_SQ_SR : SQ * SR = 0)
variable (is_perpendicular_SR_SP : SR * SP = 0)
variable (SP_eq_9 : SP = 9)
variable (SQ_eq_8 : SQ = 8)
variable (SR_eq_7 : SR = 7)

theorem volume_of_pyramid_SPQR : 
  ∃ V : ℝ, V = 84 := by
  -- Conditions and assumption
  sorry

end volume_of_pyramid_SPQR_l642_642925


namespace find_incorrect_value_l642_642563

def average_of_incorrect (X : ℕ) : ℕ :=
  20

def average_of_correct : ℕ :=
  26

def incorrect_value (incorrect_average : ℕ) (correct_average : ℕ) (sum_incorrect : ℕ) (sum_correct : ℕ) : ℕ :=
  86 - (sum_correct - sum_incorrect)

theorem find_incorrect_value :
  let X := 86 - (10 * average_of_correct - 10 * average_of_incorrect 0)
  X = 26 :=
by
  unfold average_of_incorrect average_of_correct
  have h1 : 10 * average_of_incorrect 0 = 200 := by rfl
  have h2 : 10 * average_of_correct = 260 := by rfl
  have h3 : 260 - 200 = 60 := by rfl
  have h4 : 86 - 60 = 26 := by rfl
  show X = 26 from h4
  sorry

end find_incorrect_value_l642_642563


namespace count_n_even_sequences_l642_642772

/-- Definition of an "n-even sequence" --/
def is_n_even_sequence (n : ℕ) (a : list ℕ) : Prop :=
  a.sum = n ∧ (∃ k : ℕ, k % 2 = 0 ∧ k = (list.enum a).countp (λ ij, (ij.1 < ij.2 ∧ ij.2 > ij.1)))

/-- Main theorem statement to find the number of n-even sequences --/
theorem count_n_even_sequences (n : ℕ) : 
  ∃ c : ℕ, ∀ (a : list ℕ), is_n_even_sequence n a → a.length = c :=
begin
  let res := 2^(n-2) + 2^(nat.floor (n / 2) - 1),
  use res,
  sorry
end

end count_n_even_sequences_l642_642772


namespace piggy_bank_after_8_weeks_l642_642873

-- Define initial amount in the piggy bank
def initial_amount : ℝ := 43

-- Define weekly allowance amount
def weekly_allowance : ℝ := 10

-- Define fraction of allowance Jack saves
def saving_fraction : ℝ := 0.5

-- Define number of weeks
def number_of_weeks : ℕ := 8

-- Define weekly savings amount
def weekly_savings : ℝ := saving_fraction * weekly_allowance

-- Define total savings after a given number of weeks
def total_savings (weeks : ℕ) : ℝ := weeks * weekly_savings

-- Define the final amount in the piggy bank after a given number of weeks
def final_amount (weeks : ℕ) : ℝ := initial_amount + total_savings weeks

-- Theorem: Prove that final amount in piggy bank after 8 weeks is $83
theorem piggy_bank_after_8_weeks : final_amount number_of_weeks = 83 := by
  sorry

end piggy_bank_after_8_weeks_l642_642873


namespace find_probability_l642_642593

open ProbabilityTheory

def urns : List (List (Fin 6)) := replicate 9 [0, 0, 1, 1] ++ [[0, 0, 0, 0, 0, 1]]

def urn_prob : fin 10 → ℝ
| ⟨0, _⟩ := 0.9
| ⟨1, _⟩ := 0.1
| _ := 0  -- uninhabited cases, we only care about the first two

def ball_prob (urn : fin 10) (ball : Fin 6) : ℝ :=
if urn.val < 9 
then if ball.val < 2 then 0.5 else 0 
else if ball.val < 5 then 0.83334 else 0.16666

def total_prob : ℝ :=
(0.9 * 0.5) + (0.1 * 0.83334)

noncomputable def bayes_result : ℝ :=
(0.1 * 0.83334) / total_prob

theorem find_probability :
  bayes_result ≈ 0.15625 :=
sorry

end find_probability_l642_642593


namespace segment_PM_n_exists_l642_642336

-- Define the existence of two lines intersecting at point M (off the paper)
axiom two_lines_intersect_at_M (A B C D P : Point) (line1 : Line P A) (line2 : Line P B) 
  (line1_intersect_M : LineIntersect line1 line2 = M) : Prop

-- Define the point P on the paper
axiom point_P_on_paper (P : Point) : Prop

-- We aim to prove the existence of a segment PM_n that is part of the line PM intersecting the paper
theorem segment_PM_n_exists (A B C D P M : Point) (line1 : Line A B) (line2 : Line C D)
  (intersect_M : LineIntersect line1 line2 = M) (on_paper : point_P_on_paper P) :
  ∃ (PM_n : Segment), lies_on_paper PM_n ∧ (is_part_of_line PM_n (Line P M)) :=
begin
  sorry,
end

end segment_PM_n_exists_l642_642336


namespace hawks_points_l642_642841

theorem hawks_points (x y z : ℤ) 
  (h_total_points: x + y = 82)
  (h_margin: x - y = 18)
  (h_eagles_points: x = 12 + z) : 
  y = 32 := 
sorry

end hawks_points_l642_642841


namespace problem_statement_l642_642773

-- Definitions and conditions
def a (n : ℕ) : ℕ :=
if n = 0 then 1 else n

lemma a_one : a 1 = 1 := rfl

lemma a_rec (n i : ℕ) : a (n + i) - a n = i :=
begin
  cases n,
  { simp [a], },
  { simp [a], },
end

def b (n : ℕ) : ℚ :=
if n = 0 then 1 else 2 / (n * (n + 1))

lemma b_one : b 1 = 1 := rfl

lemma b_ratio (n : ℕ) (h : n ≠ 0) : 
  b (n + 1) / b n = a n / a (n + 2) :=
begin
  cases n,
  { contradiction },
  { simp [a, b], field_simp, norm_cast, simp },
end

noncomputable def S (n : ℕ) : ℚ := 
∑ k in Finset.range (n + 1), b k

lemma sum_b (n : ℕ) (h : n ≠ 0) : 
  S n = (2 * n) / (n + 1) :=
begin
  sorry -- This would contain the proof, but it is not required per instructions
end

-- Problem statement
theorem problem_statement (a : ℕ → ℕ) (b : ℕ → ℚ) (S : ℕ → ℚ) :
  (a 1 = 1) →
  (∀ (n i : ℕ), a (n + i) - a n = i) →
  (b 1 = 1) →
  (∀ (n : ℕ) (h : n ≠ 0), b (n + 1) / b n = a n / a (n + 2)) →
  (∀ (n : ℕ) (h : n ≠ 0), S n = (2 * n) / (n + 1)) →
  (a = (λ n, if n = 0 then 1 else n)) ∧
  (∀ (n : ℕ) (a_n : ℕ → ℕ) (b_n : ℕ → ℚ) S, 
    S = (λ n, ∑ k in Finset.range (n + 1), b k) →
    S n = (2 * n) / (n + 1)) :=
by {intros, split, {ext n, exact a}, {intros, exact a_4 n a_3}}

end problem_statement_l642_642773


namespace length_of_AB_l642_642926

-- Define the distances given as conditions
def AC : ℝ := 5
def BD : ℝ := 6
def CD : ℝ := 3

-- Define the linear relationship of points A, B, C, D on the line
def points_on_line_in_order := true -- This is just a placeholder

-- Main theorem to prove
theorem length_of_AB : AB = 2 :=
by
  -- Apply the conditions and the linear relationships
  have BC : ℝ := BD - CD
  have AB : ℝ := AC - BC
  -- This would contain the actual proof using steps, but we skip it here
  sorry

end length_of_AB_l642_642926


namespace distinct_real_roots_of_sqrt_eq_l642_642758

theorem distinct_real_roots_of_sqrt_eq 
  (a b c : ℝ) (h1 : a ≠ b) (h2 : a ≠ c) (h3 : b ≠ c) :
  ∃ (x1 x2 : ℝ), x1 ≠ x2 ∧ 
  (∀ x : ℝ, (∃ (sign1 sign2 sign3 : bool), 
    (if sign1 then (sqrt (x - a)) else (-sqrt (x - a))) +
    (if sign2 then (sqrt (x - b)) else (-sqrt (x - b))) +
    (if sign3 then (sqrt (x - c)) else (-sqrt (x - c))) = 0)
    ↔ x = x1 ∨ x = x2) :=
sorry

end distinct_real_roots_of_sqrt_eq_l642_642758


namespace cos_150_eq_neg_sqrt3_div_2_l642_642291

theorem cos_150_eq_neg_sqrt3_div_2 :
  (cos (150 * real.pi / 180)) = - (sqrt 3 / 2) := sorry

end cos_150_eq_neg_sqrt3_div_2_l642_642291


namespace cos_150_eq_neg_sqrt3_over_2_l642_642225

theorem cos_150_eq_neg_sqrt3_over_2 : Real.cos (150 * Real.pi / 180) = -Real.sqrt 3 / 2 := 
by 
  sorry

end cos_150_eq_neg_sqrt3_over_2_l642_642225


namespace sum_series_eq_l642_642710

theorem sum_series_eq :
  ∑' n : ℕ, (4 * (n + 1) - 3) / 3^(n + 1) = 3 / 2 :=
sorry

end sum_series_eq_l642_642710


namespace probability_of_special_draw_l642_642978

theorem probability_of_special_draw 
    (deck : Finset Card) 
    (cards_dealt : List Card)
    (first_card_is_5 : card_is_5 (cards_dealt.nth 0))
    (second_card_is_heart : card_is_heart (cards_dealt.nth 1))
    (third_card_is_ace : card_is_ace (cards_dealt.nth 2)) :
    probability_special_draw(deck, cards_dealt) = 1 / 650 := by
  sorry

-- Define predicates for card characteristics
def Card := nat -- This is a simplification. You would define a card type with suits and ranks.
def card_is_5 (c : Card) : Prop := sorry -- Define the condition for the card to be a 5.
def card_is_heart (c : Card) : Prop := sorry -- Define the condition for the card to be a heart suit.
def card_is_ace (c : Card) : Prop := sorry -- Define the condition for the card to be an ace.

-- Define the function to calculate the probability
def probability_special_draw (deck : Finset Card) (cards_dealt : List Card) : ℚ := sorry


end probability_of_special_draw_l642_642978


namespace find_y_l642_642947

theorem find_y (t : ℝ) (x : ℝ) (y : ℝ)
  (hx : x = 3 - 2 * t)
  (hy : y = 3 * t + 6)
  (hx_cond : x = -6) :
  y = 19.5 :=
by
  sorry

end find_y_l642_642947


namespace totalKidsInLawrenceCounty_l642_642504

-- Constants representing the number of kids in each category
def kidsGoToCamp : ℕ := 629424
def kidsStayHome : ℕ := 268627

-- Statement of the total number of kids in Lawrence county
theorem totalKidsInLawrenceCounty : kidsGoToCamp + kidsStayHome = 898051 := by
  sorry

end totalKidsInLawrenceCounty_l642_642504


namespace partition_coins_possible_l642_642380

noncomputable def partition_coins (n : ℕ) (coins : Fin 4n → ℕ) (colors : Fin 4n → Fin n) 
    (weights : Fin 4n → ℕ)  : Prop :=
∃ (S T : Finset (Fin 4n)),
  (S ∪ T = Finset.univ) ∧ 
  (S ∩ T = ∅) ∧ 
  (∀ c : Fin n, (Finset.card (S.filter (λ i, colors i = c)) = 2 ∧ 
                  Finset.card (T.filter (λ i, colors i = c)) = 2)) ∧ 
  (Finset.sum S weights = Finset.sum T weights)

theorem partition_coins_possible (n : ℕ) (h_n : 0 < n) :
  ∀ (coins : Fin 4n → ℕ) (colors : Fin 4n → Fin n) (weights : Fin 4n → ℕ),
  (∀ i, coins i = i + 1) ∧ 
  (∀ c : Fin n, Finset.card (Finset.filter (λ i, colors i = c) Finset.univ) = 4) →
  partition_coins n coins colors weights := 
by
  sorry

end partition_coins_possible_l642_642380


namespace diagonals_of_convex_polygon_l642_642822

theorem diagonals_of_convex_polygon (n : ℕ) (hn : n = 24) : 
  let num_diagonals := (n * (n - 3)) / 2 
  in num_diagonals = 126 :=
by
  rfl

end diagonals_of_convex_polygon_l642_642822


namespace num_possible_arrangements_l642_642552

theorem num_possible_arrangements : 
  ∀ (a b c d : ℕ), set.insert a (set.insert b (set.insert c (set.insert d ∅))) = {3, 4, 5, 6} →
  (a = 3 ∨ a = 4 ∨ a = 5 ∨ a = 6) →
  (b = 3 ∨ b = 4 ∨ b = 5 ∨ b = 6) →
  (c = 3 ∨ c = 4 ∨ c = 5 ∨ c = 6) →
  (d = 3 ∨ d = 4 ∨ d = 5 ∨ d = 6) →
  (is_distinct_list [a, b, c, d]) →
  (finset.card (perm.num_perms_to_finset [a,b,c,d])) = 24 :=
begin
  sorry
end

end num_possible_arrangements_l642_642552


namespace general_terms_and_sum_l642_642416

section 
-- Definitions based on given conditions
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, (a 1 = 2) ∧ (a 4 = 16) ∧ (∀ n : ℕ, a (n + 1) = q * a n)

def arithmetic_sequence (b : ℕ → ℝ) : Prop :=
  (b 2 = 3) ∧ (b 3 = 5) ∧ (∀ n : ℕ, b (n + 1) = b n + 2)

def sum_first_n_terms (c : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, c n = 1 / ((2 * n - 1) * (2 * n + 1)) ∧ S n = ∑ i in range n, c i

-- Theorem statement for general term formulas and sum S_n
theorem general_terms_and_sum (a b c : ℕ → ℝ) (S : ℕ → ℝ) :
  geometric_sequence a →
  arithmetic_sequence b →
  sum_first_n_terms c S →
  (∀ n : ℕ, a n = 2 ^ n) ∧
  (∀ n : ℕ, b n = 2 * n - 1) ∧
  (∀ n : ℕ, S n = n / (2 * n + 1)) :=
by 
  intros h1 h2 h3
  sorry
end

end general_terms_and_sum_l642_642416


namespace time_to_cross_approx_3_minutes_l642_642657

def km_per_hr_to_m_per_min (s : ℝ) := (s * 1000) / 60

noncomputable def time_to_cross_bridge (distance speed : ℝ) := distance / speed

theorem time_to_cross_approx_3_minutes :
  ∀ (speed_km_hr : ℝ) (bridge_length_m : ℝ),
    bridge_length_m = 500 ∧ km_per_hr_to_m_per_min speed_km_hr = 166.67 →
    time_to_cross_bridge bridge_length_m 166.67 ≈ 3 :=
by
  intros speed_km_hr bridge_length_m h
  sorry

end time_to_cross_approx_3_minutes_l642_642657


namespace football_games_per_month_l642_642600

theorem football_games_per_month :
  let total_games := 5491
  let months := 17.0
  total_games / months = 323 := 
by
  let total_games := 5491
  let months := 17.0
  -- This is where the actual computation would happen if we were to provide a proof
  sorry

end football_games_per_month_l642_642600


namespace trip_duration_17_hours_l642_642984

theorem trip_duration_17_hours :
  ∃ T : ℝ, 
    (∀ d₁ d₂ : ℝ,
      (d₁ / 30 + 1 + (150 - d₁) / 4 = T) ∧ 
      (d₁ / 30 + d₂ / 30 + (150 - (d₁ - d₂)) / 30 = T) ∧ 
      ((d₁ - d₂) / 4 + (150 - (d₁ - d₂)) / 30 = T))
  → T = 17 :=
by
  sorry

end trip_duration_17_hours_l642_642984


namespace fan_rotations_l642_642540

-- Conditions in the problem
def slow_rate : ℕ := 100
def medium_rate : ℕ := 2 * slow_rate
def high_rate : ℕ := 2 * medium_rate

-- Hypothesis
def time_minutes : ℕ := 15

-- Theorem statement
theorem fan_rotations : high_rate * time_minutes = 6000 := by
  -- since the solution involves calculation, we use a proof by calculation
  calc
    high_rate * time_minutes
      = (2 * (2 * slow_rate)) * time_minutes : by rfl
    ... = 4 * slow_rate * time_minutes : by ring
    ... = 4 * 100 * 15 : by rw slow_rate
    ... = 6000 : by norm_num

end fan_rotations_l642_642540


namespace LCM_30_45_l642_642098

theorem LCM_30_45 : Nat.lcm 30 45 = 90 := by
  sorry

end LCM_30_45_l642_642098


namespace tan_A_in_right_triangle_l642_642492

theorem tan_A_in_right_triangle 
  {A B C : Type} [metric_space ABC]
  (hC : ∠ C = 90) (h_AB : dist A B = 13) (h_BC : dist B C = 5) :
  tan ∠ A = 5 / 12 :=
by
  have h_AC : dist A C = 12 := by 
    -- Use the Pythagorean theorem to derive this value, skipping proof for brevity
    sorry
  -- Proof that tan A = BC / AC, skipping proof for brevity
  sorry

end tan_A_in_right_triangle_l642_642492


namespace max_red_stones_guarantee_l642_642729

-- Define a position on the Cartesian plane where x and y are positive integers not exceeding 20
structure Position :=
  (x : ℕ)
  (y : ℕ)
  (hx : x > 0 ∧ x ≤ 20)
  (hy : y > 0 ∧ y ≤ 20)

-- Define the distance function between two positions
def distance (p1 p2 : Position) : ℝ :=
  real.sqrt ((p1.x - p2.x) ^ 2 + (p1.y - p2.y) ^ 2)

-- Define the conditions where a new red stone can be placed
def valid_red_placement (red_positions : set Position) (new_pos : Position) : Prop :=
  ∀ pos ∈ red_positions, distance pos new_pos ≠ real.sqrt 5

-- Define the game setup and the maximum K such that A can always place at least K red stones
theorem max_red_stones_guarantee : ∃ K : ℕ, K = 100 ∧ ∀ blue_positions : set Position,
  ∃ red_positions : set Position, 
    (∀ pos ∈ red_positions, valid_red_placement (red_positions \ {pos}) pos) ∧ 
    ∀ pos ∈ red_positions, pos ∉ blue_positions ∧
    red_positions ≠ ∅ ∧ red_positions.size ≥ K :=
sorry

end max_red_stones_guarantee_l642_642729


namespace base_prime_representation_441_l642_642732

theorem base_prime_representation_441 : 
  base_prime_representation 441 = "0220" :=
sorry

end base_prime_representation_441_l642_642732


namespace mean_and_variance_shift_l642_642774

open_locale big_operators

variables {n : ℕ} (x : Fin n → ℝ) (c : ℝ) (h : c ≠ 0)

def mean (x : Fin n → ℝ) : ℝ := (1 / n) * ∑ i, x i

def variance (x : Fin n → ℝ) : ℝ := (1 / n) * ∑ i, (x i - mean x) ^ 2

theorem mean_and_variance_shift (x : Fin n → ℝ) (c : ℝ) (h : c ≠ 0) :
  mean (λ i, x i + c) = mean x + c ∧ variance (λ i, x i + c) = variance x :=
by
  sorry

end mean_and_variance_shift_l642_642774


namespace probability_correct_l642_642980

-- Defining the conditions of the problem
def num_cards : ℕ := 52
def first_card_five : ℕ → Prop
| 5 := true
| _ := false

def second_card_heart (card : ℕ) : Prop :=
card = 11 -- Assuming 11 represents a heart in a simplified model

def third_card_ace : ℕ → Prop
| 1  := true  -- Assuming 1 represents an Ace in this simplified model
| _ := false

-- The required theorem
theorem probability_correct :
  (3/52) * (12/51) * (4/50) + (3/52) * (1/51) * (3/50) + 
  (1/52) * (11/51) * (4/50) + (1/52) * (1/51) * (3/50) 
  = (1 / 663) :=
begin
  -- Proof steps would go here
  sorry
end

end probability_correct_l642_642980


namespace exterior_angle_of_regular_octagon_l642_642848

def sum_of_interior_angles (n : ℕ) : ℕ := 180 * (n - 2)
def interior_angle (s : ℕ) (n : ℕ) : ℕ := sum_of_interior_angles n / s
def exterior_angle (ia : ℕ) : ℕ := 180 - ia

theorem exterior_angle_of_regular_octagon : 
    exterior_angle (interior_angle 8 8) = 45 := 
by 
  sorry

end exterior_angle_of_regular_octagon_l642_642848


namespace derivative_of_f_l642_642801

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x

theorem derivative_of_f (x : ℝ) (h : 0 < x) :
    deriv f x = (1 - Real.log x) / (x ^ 2) := 
sorry

end derivative_of_f_l642_642801


namespace modulus_of_z_l642_642770

-- Define the complex number z
def z : ℂ := ((1 : ℂ) + (Complex.i))^2 / ((1 : ℂ) - (Complex.i))

-- Prove that the modulus of z is sqrt(2)
theorem modulus_of_z : Complex.abs z = Real.sqrt 2 := by
  sorry

end modulus_of_z_l642_642770


namespace miriam_tuesday_pushups_l642_642532

-- Definitions for conditions
def monday_pushups : ℕ := 5
def tuesday_pushups : ℕ
def wednesday_pushups : ℕ := 2 * tuesday_pushups
def thursday_pushups : ℕ := (monday_pushups + tuesday_pushups + wednesday_pushups) / 2
def friday_pushups : ℕ := monday_pushups + tuesday_pushups + wednesday_pushups + thursday_pushups

-- The main theorem stating that given the conditions, the number of push-ups on Tuesday is 7.
theorem miriam_tuesday_pushups : tuesday_pushups = 7 :=
  -- Provided condition that on Friday the total is equal to 39 push-ups.
  have h1 : friday_pushups = 39,
    by sorry
  -- Use the derived equation from the solution to finally prove tuesday_pushups = 7
  sorry

end miriam_tuesday_pushups_l642_642532


namespace hyperbola_range_l642_642832

theorem hyperbola_range (m : ℝ) : 
  (∃ (x y : ℝ), (x^2 / (|m| - 1)) + (y^2 / (2 - m)) = 1 ∧ (|m| - 1) * (2 - m) < 0) ↔ 
  (m ∈ set.Ioo (-1 : ℝ) 1 ∨ m ∈ set.Ioi (2 : ℝ)) := 
by 
  sorry

end hyperbola_range_l642_642832


namespace correlation_relationships_l642_642685

-- Let's define the relationships as conditions
def volume_cube_edge_length (v e : ℝ) : Prop := v = e^3
def yield_fertilizer (yield fertilizer : ℝ) : Prop := True -- Assume linear correlation within a certain range
def height_age (height age : ℝ) : Prop := True -- Assume linear correlation within a certain age range
def expenses_income (expenses income : ℝ) : Prop := True -- Assume linear correlation
def electricity_consumption_price (consumption price unit_price : ℝ) : Prop := price = consumption * unit_price

-- We want to prove that the answers correspond correctly to the conditions:
theorem correlation_relationships :
  ∀ (v e yield fertilizer height age expenses income consumption price unit_price : ℝ),
  ¬ volume_cube_edge_length v e ∧ yield_fertilizer yield fertilizer ∧ height_age height age ∧ expenses_income expenses income ∧ ¬ electricity_consumption_price consumption price unit_price → 
  "D" = "②③④" :=
by
  intros
  sorry

end correlation_relationships_l642_642685


namespace construct_1_degree_l642_642033

def canConstruct1DegreeUsing19Degree : Prop :=
  ∃ (n : ℕ), n * 19 = 360 + 1

theorem construct_1_degree (h : ∃ (x : ℕ), x * 19 = 360 + 1) : canConstruct1DegreeUsing19Degree := by
  sorry

end construct_1_degree_l642_642033


namespace cos_150_eq_neg_sqrt3_div_2_l642_642159

theorem cos_150_eq_neg_sqrt3_div_2 : Real.cos (150 * Real.pi / 180) = - (Real.sqrt 3 / 2) :=
by
  sorry

end cos_150_eq_neg_sqrt3_div_2_l642_642159


namespace all_points_T_collinear_l642_642508

open Set

-- Conditions
variables {O : ℕ → Circle}
variables {P : ℕ → ℕ → Point}
variables {T : Set Point}
variables {S : Set Point}
variables {i j k l : ℕ}

-- Assumptions
axiom O_distinct_radii : ∀ i j, i ≠ j → O i ≠ O j
axiom P_tangent_points :
  ∀ i j : ℕ, 1 ≤ i ∧ i < j ∧ j ≤ 2012 → P i j ∈ T
axiom S_subset_T : S ⊆ T
axiom S_collinear : Collinear S
axiom S_size : S.card = 2021056
axiom T_size : T.card = 2023066

-- To Prove
theorem all_points_T_collinear : Collinear T :=
sorry

end all_points_T_collinear_l642_642508


namespace exists_member_in_4_committees_l642_642003

open Finset

def club := Fin 11  -- There are 11 committees.
def member := {m : ℕ // m < 55} -- Total members since each committee could be represented by different members.

/-- Each committee has 5 members -/
def committee_members (c : club) : Finset member := sorry

/-- Every two committees have a member in common -/
axiom common_member 
  (c1 c2 : club) (h : c1 ≠ c2) : (∃ m : member, m ∈ committee_members c1 ∧ m ∈ committee_members c2)

theorem exists_member_in_4_committees :
  ∃ m : member, (∑ (c : club), ite (m ∈ committee_members c) 1 0) ≥ 4 :=
sorry

end exists_member_in_4_committees_l642_642003


namespace ana_workshop_percentage_l642_642109

variable (workday_minutes : Nat) (first_workshop : Nat) (second_workshop : Nat)

def percent_workday_in_workshops (total_workday : Nat) (workshop_time : Nat) := 
  (workshop_time.toRat / total_workday.toRat) * 100

theorem ana_workshop_percentage :
  let workday_minutes := 8 * 60
  let first_workshop := 35
  let second_workshop := 3 * first_workshop
  let total_workshop_time := first_workshop + second_workshop
  percent_workday_in_workshops workday_minutes total_workshop_time ≈ 29 :=
by
  sorry

end ana_workshop_percentage_l642_642109


namespace cos_150_eq_neg_half_l642_642275

theorem cos_150_eq_neg_half : Real.cos (150 * Real.pi / 180) = -1 / 2 := 
by
  sorry

end cos_150_eq_neg_half_l642_642275


namespace cos_150_eq_neg_sqrt3_div_2_l642_642167

theorem cos_150_eq_neg_sqrt3_div_2 : Real.cos (150 * Real.pi / 180) = - (Real.sqrt 3 / 2) :=
by
  sorry

end cos_150_eq_neg_sqrt3_div_2_l642_642167


namespace cos_150_eq_neg_half_l642_642192

theorem cos_150_eq_neg_half :
  real.cos (150 * real.pi / 180) = -1 / 2 :=
sorry

end cos_150_eq_neg_half_l642_642192


namespace oxygen_atom_diameter_in_scientific_notation_l642_642569

theorem oxygen_atom_diameter_in_scientific_notation :
  0.000000000148 = 1.48 * 10^(-10) :=
sorry

end oxygen_atom_diameter_in_scientific_notation_l642_642569


namespace total_area_covered_by_strips_l642_642367

def length := 15
def width := 2
def num_strips := 5
def intersection_width := 2

theorem total_area_covered_by_strips : 
  let area_one_strip := length * width in
  let total_area_without_overlaps := num_strips * area_one_strip in
  let area_one_intersection := intersection_width * intersection_width in
  let total_intersections := 10 * 2 in
  let total_intersection_area := total_intersections * area_one_intersection in
  total_area_without_overlaps - total_intersection_area = 70 := 
by sorry

end total_area_covered_by_strips_l642_642367


namespace angle_BAC_15_degrees_l642_642981

noncomputable def angleBAC : ℝ := 15

theorem angle_BAC_15_degrees :
  ∀ (O A B C : Type) 
  (area1 area2 area3 : ℝ)
  (r1 r2 r3 : ℝ)
  (h_area1 : area1 = 2 * π) 
  (h_area2 : area2 = 3 * π)
  (h_area3 : area3 = 4 * π) 
  (h_radius1 : r1 = sqrt 2)
  (h_radius2 : r2 = sqrt 3)
  (h_radius3 : r3 = 2),
  -- Conditions for circles' center O and points A, B, C with tangents
  (measure_angle A B C = angleBAC) :=
by
  sorry

end angle_BAC_15_degrees_l642_642981


namespace range_of_f_log_gt_zero_l642_642339

open Real

noncomputable def f (x : ℝ) : ℝ := -- Placeholder function definition
  sorry

theorem range_of_f_log_gt_zero :
  (∀ x, f x = f (-x)) ∧
  (∀ x y, 0 ≤ x → x ≤ y → f x ≤ f y) ∧
  (f (1 / 3) = 0) →
  {x : ℝ | f ((log x) / (log (1 / 8))) > 0} = 
    (Set.Ioo 0 (1 / 2) ∪ Set.Ioi 2) :=
  sorry

end range_of_f_log_gt_zero_l642_642339


namespace cost_price_of_computer_table_l642_642046

noncomputable def cost_price (paid : ℝ) (markup : ℝ) : ℝ :=
  paid / markup

theorem cost_price_of_computer_table :
  let paid := 7350
  let markup := 1.10
  cost_price paid markup = 6681.82 :=
by
  -- main condition and equation derived from it
  let paid := 7350
  let markup := 1.10
  have h : cost_price paid markup = paid / markup := rfl
  -- solving that equation led to the answer
  have h_solution : paid / markup = 6681.82 :=
    by norm_num
  show cost_price paid markup = 6681.82
  from eq.trans h h_solution

end cost_price_of_computer_table_l642_642046


namespace probability_of_C_and_D_are_equal_l642_642082

theorem probability_of_C_and_D_are_equal (h1 : Prob_A = 1/4) (h2 : Prob_B = 1/3) (h3 : total_prob = 1) (h4 : Prob_C = Prob_D) : 
  Prob_C = 5/24 ∧ Prob_D = 5/24 := by
  sorry

end probability_of_C_and_D_are_equal_l642_642082


namespace trajectory_eq_chord_length_MN_l642_642475

-- Define the conditions and the required theorem statements
def point (x : ℝ) (y : ℝ) := (x, y)

def A := point 3 0
def B := point (-1) 0
def line_l (x y : ℝ) := x - y + 3 = 0

-- Define vector operations involving points
def vector (p1 p2 : (ℝ × ℝ)) := ((p2.1 - p1.1), (p2.2 - p1.2))

def dot_product (v1 v2 : (ℝ × ℝ)) := v1.1 * v2.1 + v1.2 * v2.2

-- Prove that the trajectory of point C (question 1)
theorem trajectory_eq :
  ∀ (C : ℝ × ℝ), dot_product (vector A C) (vector B C) = 5 ↔ (C.1 - 1) ^ 2 + C.2 ^ 2 = 9 := by
  sorry

-- Prove the length of |MN| (question 2)
theorem chord_length_MN :
  ∀ (M N : ℝ × ℝ), line_l M.1 M.2 ∧ line_l N.1 N.2 ∧ ((M.1 - 1)^2 + M.2^2 = 9) ∧ ((N.1 - 1)^2 + N.2^2 = 9) →
  (dist M N = 2) := by
  sorry

end trajectory_eq_chord_length_MN_l642_642475


namespace cos_150_eq_negative_cos_30_l642_642154

theorem cos_150_eq_negative_cos_30 :
  Real.cos (150 * Real.pi / 180) = - (Real.cos (30 * Real.pi / 180)) :=
by sorry

end cos_150_eq_negative_cos_30_l642_642154


namespace cos_150_degree_l642_642178

theorem cos_150_degree : Real.cos (150 * Real.pi / 180) = -Real.sqrt 3 / 2 :=
by
  sorry

end cos_150_degree_l642_642178


namespace number_of_solutions_eq_7_l642_642572

theorem number_of_solutions_eq_7 :
  ∃! x ∈ (-10 : ℝ)..(10 : ℝ), 10 * Real.sin (x + Real.pi / 6) = x := sorry

end number_of_solutions_eq_7_l642_642572


namespace cos_150_degree_l642_642172

theorem cos_150_degree : Real.cos (150 * Real.pi / 180) = -Real.sqrt 3 / 2 :=
by
  sorry

end cos_150_degree_l642_642172


namespace sequence_a6_value_l642_642384

theorem sequence_a6_value :
  let S : ℕ → ℕ := λ n, 2^n - 1 in
  let a : ℕ → ℕ := λ n, S n - S (n - 1) in
  a 6 = 32 :=
by
  let S := λ n, 2^n - 1
  let a := λ n, S n - S (n - 1)
  show a 6 = 32
  sorry

end sequence_a6_value_l642_642384


namespace angle_A_in_triangle_l642_642867

theorem angle_A_in_triangle (a b : ℝ) (cosB : ℝ) (h_a : a = 7) (h_b : b = 8) (h_cosB : cosB = 1/7) :
  ∠A = π/3 :=
sorry

end angle_A_in_triangle_l642_642867


namespace identify_inequality_l642_642029

-- Define the options
inductive Option
| A
| B
| C
| D

-- Define the condition that checks if a given option is an inequality
def is_inequality : Option → Bool
| Option.A => True  -- "-x > 1" is an inequality
| _         => False -- All other options are not inequalities

-- The main statement
theorem identify_inequality : ∃ (opt : Option), is_inequality opt = True :=
by
  exists Option.A
  -- Option.A: "-x > 1", which is an inequality
  sorry

end identify_inequality_l642_642029


namespace area_enclosed_between_altitude_and_bisector_l642_642868

def side_AB : Nat := 26
def side_BC : Nat := 30
def side_AC : Nat := 28

theorem area_enclosed_between_altitude_and_bisector :
    enclosed_area side_AB side_BC side_AC = 36 := by
  sorry


end area_enclosed_between_altitude_and_bisector_l642_642868


namespace eight_people_lineup_two_windows_l642_642963

theorem eight_people_lineup_two_windows :
  (2 ^ 8) * (Nat.factorial 8) = 10321920 := by
  sorry

end eight_people_lineup_two_windows_l642_642963


namespace find_eccentricity_l642_642401

def parabola_focus : ℝ × ℝ :=
  (-sqrt 5, 0)

def ellipse_equation (x y a : ℝ) : Prop :=
  (x^2) / (a^2) + (y^2) / 4 = 1

def eccentricity (c a : ℝ) : ℝ :=
  c / a

theorem find_eccentricity (a : ℝ) (h₁ : a > 0) (h₂ : parabola_focus = (-sqrt 5, 0)) (h₃ : ellipse_equation (-sqrt 5) 0 a) :
  eccentricity (sqrt 5) a = sqrt 5 / 3 :=
by
  sorry

end find_eccentricity_l642_642401


namespace max_abs_equals_a1_and_inequality_holds_l642_642523

theorem max_abs_equals_a1_and_inequality_holds
    (n : ℕ) 
    (a : ℕ → ℝ)
    (h1 : ∀ i, a i ≥ a (i + 1)) 
    (h2 : ∀ k : ℕ, 0 < k → 0 ≤ ∑ i in finset.range n, (a i) ^ k)
    (p : ℝ)
    (h3 : p = max (finset.image (λ i, |a i|) (finset.range n))) :
    p = a 0 ∧ ∀ x : ℝ, x > a 0 → (∏ i in finset.range n, (x - a i)) ≤ x ^ n - (a 0) ^ n := 
by
  sorry

end max_abs_equals_a1_and_inequality_holds_l642_642523


namespace cube_relation_l642_642787

theorem cube_relation (x : ℝ) (h : x - 1/x = 5) : x^3 - 1/x^3 = 140 :=
by
  sorry

end cube_relation_l642_642787


namespace comparison_of_a_b_c_l642_642887

noncomputable def a : ℝ := Real.log 2 / Real.log 0.2
noncomputable def b : ℝ := 0.2^2
noncomputable def c : ℝ := 2^0.2

theorem comparison_of_a_b_c : a < b ∧ b < c := by
  sorry

end comparison_of_a_b_c_l642_642887


namespace avg_and_var_of_b1_b2_l642_642450

variables {n : ℕ} (x : ℕ → ℝ)
def average (x : ℕ → ℝ) (n : ℕ) := (∑ i in finset.range n, x i) / n
def variance (x : ℕ → ℝ) (n : ℕ) := (∑ i in finset.range n, (x i - average x n) ^ 2) / n

theorem avg_and_var_of_b1_b2 :
  average (λ i, x i + 1) n = 17 →
  variance (λ i, x i + 1) n = 2 →
  average (λ i, x i + 2) n = 18 ∧
  variance (λ i, x i + 2) n = 2 :=
by
  assume h_avg h_var
  sorry

end avg_and_var_of_b1_b2_l642_642450


namespace area_of_R2_l642_642780

theorem area_of_R2
  (a b : ℝ)
  (h1 : b = 3 * a)
  (h2 : a^2 + b^2 = 225) :
  a * b = 135 / 2 :=
by
  sorry

end area_of_R2_l642_642780


namespace find_pqr_l642_642515

noncomputable def vector_problem (a b c: ℝ^3) (p q r: ℝ) : Prop :=
  (a ≠ 0) ∧ (b ≠ 0) ∧ (c ≠ 0) ∧
  (a ⬝ b = 0) ∧ (b ⬝ c = 0) ∧ (c ⬝ a = 0) ∧
  (a ⬝ a = 1) ∧ (b ⬝ b = 1) ∧ (c ⬝ c = 1) ∧
  (a = p^2 • (a × b) + q^2 • (b × c) + r^2 • (c × a) + b) ∧
  (a ⬝ (b × c) = 2)

theorem find_pqr 
  {a b c : ℝ^3} {p q r : ℝ}
  (h : vector_problem a b c p q r) : p^2 + q^2 + r^2 = 1 / 2 := 
sorry

end find_pqr_l642_642515


namespace intersection_of_sets_l642_642422

def set_M : Set ℝ := { x : ℝ | (x + 2) * (x - 1) < 0 }
def set_N : Set ℝ := { x : ℝ | x + 1 < 0 }
def intersection (A B : Set ℝ) : Set ℝ := { x : ℝ | x ∈ A ∧ x ∈ B }

theorem intersection_of_sets :
  intersection set_M set_N = { x : ℝ | -2 < x ∧ x < -1 } := 
by
  sorry

end intersection_of_sets_l642_642422


namespace limit_solution_l642_642700

open Filter Real Topology

noncomputable def limit_problem (a x : ℝ) : Prop :=
  tendsto (λ h, (a^(x+h) + a^(x-h) - 2 * a^x) / h) (𝓝 0) (𝓝 0)

theorem limit_solution (a x : ℝ) : limit_problem a x :=
by sorry

end limit_solution_l642_642700


namespace cos_150_eq_neg_half_l642_642279

theorem cos_150_eq_neg_half : Real.cos (150 * Real.pi / 180) = -1 / 2 := 
by
  sorry

end cos_150_eq_neg_half_l642_642279


namespace cos_150_eq_neg_sqrt3_div_2_l642_642127

theorem cos_150_eq_neg_sqrt3_div_2 :
  ∃ θ : ℝ, θ = 150 ∧
           0 < θ ∧ θ < 180 ∧
           ∃ φ : ℝ, φ = 180 - θ ∧
           cos φ = Real.sqrt 3 / 2 ∧
           cos θ = - cos φ :=
  sorry

end cos_150_eq_neg_sqrt3_div_2_l642_642127


namespace hybrid_buses_cost_l642_642558

theorem hybrid_buses_cost (a b : ℕ) (x : ℕ)
  (h1 : a - b = 20)
  (h2 : 3 * b - 2 * a = 60)
  (h3 : 2.4 * x + 2 * (10 - x) = 22.4):
  a = 120 ∧ b = 100 ∧ (6 * 120 + 4 * 100 = 1120) :=
by
  sorry

end hybrid_buses_cost_l642_642558


namespace solution_l642_642051

noncomputable def problem_statement (n : ℕ) : Prop :=
  ∀ P : Polynomial ℤ,
  P.degree = n →
  ∃ a b : ℕ, a ≠ b ∧ (P.eval a + P.eval b) % (a + b) = 0

theorem solution :
  ∀ n : ℕ, ( even n ) → problem_statement n :=
by
  intros n hn_even
  sorry

end solution_l642_642051


namespace value_of_polynomial_at_2_l642_642383

def f (x : ℝ) : ℝ := 4 * x^5 + 2 * x^4 + 3 * x^3 - 2 * x^2 - 2500 * x + 434

theorem value_of_polynomial_at_2 : f 2 = -3390 := by
  -- proof would go here
  sorry

end value_of_polynomial_at_2_l642_642383


namespace sum_inf_series_l642_642704

theorem sum_inf_series :
  (\sum_{n=1}^{\infty} \frac{(4 * n) - 3}{3^n}) = 1 :=
by
  sorry

end sum_inf_series_l642_642704


namespace minimum_value_of_fraction_sum_l642_642526

open Real

theorem minimum_value_of_fraction_sum (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) (h : a + b + c + d = 2) : 
    6 ≤ (1 / (a + b) + 1 / (a + c) + 1 / (a + d) + 1 / (b + c) + 1 / (b + d) + 1 / (c + d)) := by 
  sorry

end minimum_value_of_fraction_sum_l642_642526


namespace trig_eq_solution_count_l642_642890

def g_n (n : ℕ) (x : ℝ) : ℝ := (Real.sin x) ^ n + (Real.cos x) ^ n

theorem trig_eq_solution_count :
  (Finset.card {x ∈ Finset.Icc 0 (2*Real.pi) | 
    8 * g_n 5 x - 5 * g_n 3 x = 3 * g_n 1 x}) = 3 := 
sorry

end trig_eq_solution_count_l642_642890


namespace cos_150_eq_negative_cos_30_l642_642140

theorem cos_150_eq_negative_cos_30 :
  Real.cos (150 * Real.pi / 180) = - (Real.cos (30 * Real.pi / 180)) :=
by sorry

end cos_150_eq_negative_cos_30_l642_642140


namespace curve_standard_equation_range_of_y_div_x_range_of_2x_plus_y_l642_642899

-- Define the parametric equations for the curve C with the parameter θ
def P (x y : ℝ) (θ : ℝ) : Prop := 
  (x = -2 + cos θ) ∧ (y = sin θ) ∧ (0 ≤ θ ∧ θ < 2 * Real.pi)

-- Prove the standard form of the curve
theorem curve_standard_equation (x y : ℝ) (θ : ℝ) (h : P x y θ) : 
  (x + 2) ^ 2 + y ^ 2 = 1 := 
sorry

-- Prove the range of y/x
theorem range_of_y_div_x (x y : ℝ) (θ : ℝ) (h : P x y θ) : 
  -Real.sqrt 3 / 3 ≤ y / x ∧ y / x ≤ Real.sqrt 3 / 3 :=
sorry

-- Prove the range of 2x + y
theorem range_of_2x_plus_y (x y : ℝ) (θ : ℝ) (h : P x y θ) : 
  -4 - Real.sqrt 5 ≤ 2 * x + y ∧ 2 * x + y ≤ -4 + Real.sqrt 5 :=
sorry

end curve_standard_equation_range_of_y_div_x_range_of_2x_plus_y_l642_642899


namespace volume_of_inscribed_cube_l642_642849

noncomputable def cube_volume_in_pyramid (a : ℝ) : ℝ :=
  let x := (a * (Real.sqrt 2 - 1)) / 3 in
  x^3

theorem volume_of_inscribed_cube (a : ℝ) :
  cube_volume_in_pyramid a = ((a * (Real.sqrt 2 - 1)) / 3)^3 :=
by
  sorry

end volume_of_inscribed_cube_l642_642849


namespace beaker_volume_l642_642737

theorem beaker_volume {a b c d e f g h i j : ℝ} (h₁ : a = 7) (h₂ : b = 4) (h₃ : c = 5)
                      (h₄ : d = 4) (h₅ : e = 6) (h₆ : f = 8) (h₇ : g = 7)
                      (h₈ : h = 3) (h₉ : i = 9) (h₁₀ : j = 6) :
  (a + b + c + d + e + f + g + h + i + j) / 5 = 11.8 :=
by
  sorry

end beaker_volume_l642_642737


namespace video_per_disc_correct_l642_642076

def video_per_disc (total_video : ℝ) (disc_capacity : ℝ) : ℝ :=
  total_video / (Real.ceil (total_video / disc_capacity))

theorem video_per_disc_correct :
  video_per_disc 495 65 = 61.875 :=
by
  -- We skip the proof here
  sorry

end video_per_disc_correct_l642_642076


namespace tv_factory_production_l642_642060

theorem tv_factory_production :
  ∀ (planned_production planned_days days_ahead_of_schedule : ℕ),
    planned_production = 560 →
    planned_days = 16 →
    days_ahead_of_schedule = 2 →
    planned_production / (planned_days - days_ahead_of_schedule) = 40 :=
by
  intros planned_production planned_days days_ahead_of_schedule
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end tv_factory_production_l642_642060


namespace exists_set_with_conditions_l642_642929

theorem exists_set_with_conditions (n : ℕ) (hn : 0 < n) : 
  ∃ (S : Finset ℕ), S.card = n ∧ (∀ a b ∈ S, a ≠ b → (a - b) ∣ a ∧ (a - b) ∣ b ∧ ∀ c ∈ S, c ≠ a → (a - b) ∣ c = False) :=
sorry

end exists_set_with_conditions_l642_642929


namespace sum_infinite_series_l642_642716

theorem sum_infinite_series : ∑' n : ℕ, (4 * (n + 1) - 3) / (3 ^ (n + 1)) = 3 / 2 := by
    sorry

end sum_infinite_series_l642_642716


namespace merge_sort_comparisons_l642_642385

theorem merge_sort_comparisons (n : ℕ) (m : ℕ) 
  (h₁ : m - 1 < Real.log 2 n) 
  (h₂ : Real.log 2 n ≤ m) : 
  (∃ c : ℕ, c ≤ m * n - 2^m + 1) := 
by 
  apply Exists.intro (m * n - 2^m + 1) 
  assumption 
  sorry

end merge_sort_comparisons_l642_642385


namespace prime_has_property_p_l642_642369

theorem prime_has_property_p (n : ℕ) (hn : Prime n) (a : ℕ) (h : n ∣ a^n - 1) : n^2 ∣ a^n - 1 := by
  sorry

end prime_has_property_p_l642_642369


namespace cos_150_eq_neg_half_l642_642197

theorem cos_150_eq_neg_half :
  real.cos (150 * real.pi / 180) = -1 / 2 :=
sorry

end cos_150_eq_neg_half_l642_642197


namespace lucy_current_fish_l642_642529

-- Definitions based on conditions in the problem
def total_fish : ℕ := 280
def fish_needed_to_buy : ℕ := 68

-- Proving the number of fish Lucy currently has
theorem lucy_current_fish : total_fish - fish_needed_to_buy = 212 :=
by
  sorry

end lucy_current_fish_l642_642529


namespace quadrilateral_equal_division_points_sum_is_58_l642_642901

theorem quadrilateral_equal_division_points_sum_is_58 :
  let A := (0,0)
  let B := (2,3)
  let C := (5,4)
  let D := (6,0)
  let intersection := (41, 8, 7, 2)
  (intersection.1 + intersection.2 + intersection.3 + intersection.4 = 58) :=
  sorry

end quadrilateral_equal_division_points_sum_is_58_l642_642901


namespace planes_parallel_or_intersect_l642_642811

-- Given three different planes α, β, γ
variables (α β γ : ℝ^3 → Prop)
-- α and β are different planes
hypothesis (H1 : α ≠ β)
-- α and γ are different planes
hypothesis (H2 : α ≠ γ)
-- β and γ are different planes
hypothesis (H3 : β ≠ γ)
-- α is perpendicular to γ
hypothesis (H4 : ∀ x, γ x → ¬ α x)
-- β is perpendicular to γ
hypothesis (H5 : ∀ x, γ x → ¬ β x)

-- Prove that α is either parallel to β or they intersect
theorem planes_parallel_or_intersect : (∀ x y, α x → β y → false) ∨ 
                                       (∃ x, α x ∧ β x) :=
sorry

end planes_parallel_or_intersect_l642_642811


namespace cos_150_eq_neg_half_l642_642190

theorem cos_150_eq_neg_half :
  real.cos (150 * real.pi / 180) = -1 / 2 :=
sorry

end cos_150_eq_neg_half_l642_642190


namespace nine_is_possible_l642_642919

theorem nine_is_possible (N : ℕ) (hN : N ≥ 9) (numbers : Fin N → ℝ)
  (h_distinct : ∀ i j, i ≠ j → numbers i ≠ numbers j)
  (h_range : ∀ i, 0 ≤ numbers i ∧ numbers i < 1)
  (h_sum_integer : ∀ (eight : Finset (Fin N)), eight.card = 8 → ∃ x, x ∉ eight ∧ (eight.sum (λ i => numbers i) + numbers x).isInteger) :
  N = 9 :=
by
  sorry

end nine_is_possible_l642_642919


namespace jack_piggy_bank_after_8_weeks_l642_642871

-- Conditions as definitions
def initial_amount : ℕ := 43
def weekly_allowance : ℕ := 10
def saved_fraction (x : ℕ) : ℕ := x / 2
def duration : ℕ := 8

-- Mathematical equivalent proof problem
theorem jack_piggy_bank_after_8_weeks : initial_amount + (duration * saved_fraction weekly_allowance) = 83 := by
  sorry

end jack_piggy_bank_after_8_weeks_l642_642871


namespace hyperbola_eccentricity_proof_l642_642415

variables {a b c e : Real}
noncomputable def hyperbola_eccentricity (a b : Real) (h1 : a > b) (h2 : b > 0) (h3 : (2 * sqrt (a ^ 2 + b ^ 2)) = c) : Real :=
  sqrt (1 + (b ^ 2 / a ^ 2)) 

theorem hyperbola_eccentricity_proof (a b c e : Real) 
  (h1 : a > b) 
  (h2 : b > 0) 
  (h3 : c = sqrt (a ^ 2 + b ^ 2))
  (h4 : abs ((b * (-a)) + (a * 0) - (a * b)) / sqrt (b ^ 2 + a ^ 2) = (2 * sqrt 2 / 3) * c):
  e = sqrt 6 / 2 :=
sorry

end hyperbola_eccentricity_proof_l642_642415


namespace sum_inf_series_l642_642706

theorem sum_inf_series :
  (\sum_{n=1}^{\infty} \frac{(4 * n) - 3}{3^n}) = 1 :=
by
  sorry

end sum_inf_series_l642_642706


namespace distance_sum_l642_642806

noncomputable def Cartesian_curve (x y : ℝ) := (x - 2)^2 + 4 * y^2 - 4 = 0
noncomputable def line_l (t : ℝ) := (1 + (real.sqrt 3 / 2) * t, (1 / 2) * t)
noncomputable def transformed_C (x y : ℝ) := (x - 2)^2 + y^2 - 4 = 0

theorem distance_sum (t1 t2 : ℝ)
  (h1 : (1 + (real.sqrt 3 / 2) * t1 - 2)^2 + ((1 / 2) * t1)^2 - 4 = 0)
  (h2 : (1 + (real.sqrt 3 / 2) * t2 - 2)^2 + ((1 / 2) * t2)^2 - 4 = 0)
  (h_sum : t1 + t2 = real.sqrt 3)
  (h_prod : t1 * t2 = -3) :
  real.abs t1 + real.abs t2 = real.sqrt 15 :=
by sorry

end distance_sum_l642_642806


namespace Mark_less_than_Craig_l642_642728

-- Definitions for the conditions
def Dave_weight : ℕ := 175
def Dave_bench_press : ℕ := Dave_weight * 3
def Craig_bench_press : ℕ := (20 * Dave_bench_press) / 100
def Mark_bench_press : ℕ := 55

-- The theorem to be proven
theorem Mark_less_than_Craig : Craig_bench_press - Mark_bench_press = 50 :=
by
  sorry

end Mark_less_than_Craig_l642_642728


namespace fraction_of_track_Scottsdale_to_Forest_Grove_l642_642566

def distance_between_Scottsdale_and_Sherbourne : ℝ := 200
def round_trip_duration : ℝ := 5
def time_Harsha_to_Sherbourne : ℝ := 2

theorem fraction_of_track_Scottsdale_to_Forest_Grove :
  ∃ f : ℝ, f = 1/5 ∧
    ∀ (d : ℝ) (t : ℝ) (h : ℝ),
    d = distance_between_Scottsdale_and_Sherbourne →
    t = round_trip_duration →
    h = time_Harsha_to_Sherbourne →
    (2.5 - h) / t = f :=
sorry

end fraction_of_track_Scottsdale_to_Forest_Grove_l642_642566


namespace min_value_sum_l642_642343

theorem min_value_sum (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) : 
  ∃ x, (∑ i in ([3, 5, 6] : List ℕ), (if i = 3 then a/3 else if i = 5 then b/5 else c/6)) = x → x ≥ 3 / Real.cbrt 90 := 
sorry

end min_value_sum_l642_642343


namespace trajectory_of_point_C_and_chord_length_l642_642472

-- Define the points A, B, C
def pointA : ℝ × ℝ := (3, 0)
def pointB : ℝ × ℝ := (-1, 0)
variable (x y : ℝ)

-- Define the conditions
def AC := (x - 3, y)
def BC := (x + 1, y)

def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

-- The given condition for the dot product
def condition := dot_product (AC x y) (BC x y) = 5

-- The equation of the trajectory of point C
def trajectory_eq := (x - 1)^2 + y^2 = 9

-- Define the line l
def line_l (x y : ℝ) : Prop := x - y + 3 = 0

-- Find the distance from point P (the center of the circle) to line l
def distance_P_to_l (P : ℝ × ℝ) (x y : ℝ) : ℝ :=
  abs (P.1 * 1 + P.2 * (-1) + 3) / sqrt (1^2 + (-1)^2)

def centerP : ℝ × ℝ := (1, 0)
def radius : ℝ := 3
def d := distance_P_to_l centerP x y

-- The length of the chord |MN|
def chord_length := 2 * sqrt (radius^2 - d^2)

-- The theorem to prove
theorem trajectory_of_point_C_and_chord_length :
  (condition x y) → trajectory_eq x y ∧ (line_l x y → chord_length = 2) :=
by
  sorry

end trajectory_of_point_C_and_chord_length_l642_642472


namespace cos_150_eq_neg_sqrt3_div_2_l642_642310

open Real

theorem cos_150_eq_neg_sqrt3_div_2 :
  cos (150 * pi / 180) = - (sqrt 3 / 2) := 
sorry

end cos_150_eq_neg_sqrt3_div_2_l642_642310


namespace longer_train_length_l642_642991

def length_of_longer_train
  (speed_train1 : ℝ) (speed_train2 : ℝ)
  (length_shorter_train : ℝ) (time_to_clear : ℝ)
  (relative_speed : ℝ := (speed_train1 + speed_train2) * 1000 / 3600)
  (total_distance : ℝ := relative_speed * time_to_clear) : ℝ :=
  total_distance - length_shorter_train

theorem longer_train_length :
  length_of_longer_train 80 55 121 7.626056582140095 = 164.9771230827526 :=
by
  unfold length_of_longer_train
  norm_num
  sorry  -- This placeholder is used to avoid writing out the full proof.

end longer_train_length_l642_642991


namespace smallest_integer_proof_l642_642614

noncomputable def smallest_integer_x : ℤ :=
7

theorem smallest_integer_proof : 
  ∀ x : ℤ, (x < 3 * x - 12 ∧ x > 0) ↔ x >= 7 :=
by
  -- We need to show the equivalence
  intro x
  constructor

  -- for the first direction:
  -- assume that x < 3 * x - 12 and x > 0
  -- we need to show x ≥ 7
  {
    intro h
    have h1 := h.1
    have h2 := h.2
    -- proving using subtraction and division operations involves details skipped here.
    sorry
  }

  -- for the reverse direction:
  -- assume x ≥ 7
  -- we need to show that x < 3 * x -12 and x > 0
  {
    intro h
    split
    {
      -- showing that x < 3 * x - 12 follows by simple algebra
      sorry
    }
    {
      -- showing that x > 0 directly
      sorry
  }

end smallest_integer_proof_l642_642614


namespace card_selection_problem_l642_642004

theorem card_selection_problem : 
  let cards := (List.range (84)).map (λ n, if n = 0 then 1 else 3 ^ n).append (List.range (84)).map (λ n, if n = 0 then 1 else 11 ^ n) in
  167 = cards.length →
  (∃ c1 c2 c3 ∈ cards, (c1 * c2 * c3) % 33 = 0 ∧ 
   ∃ k: ℕ, (c1 * c2 * c3) = 33 * k^2) → 
  (count_cards_for_square_33 cards = 139523) :=
by
  -- Here we need some definitions which might be invoked in this context
  def count_cards_for_square_33 (cards : List ℕ) : ℕ := sorry
  
  intro cards h_len h_product_exists
  -- The remaining proof is omitted as it is not required
  sorry

end card_selection_problem_l642_642004


namespace sum_of_v_values_is_zero_l642_642333

noncomputable def v (x : ℝ) : ℝ := x^2 * Real.sin (π * x)

theorem sum_of_v_values_is_zero :
  v (-1.5) + v (-0.5) + v (0.5) + v (1.5) = 0 :=
by
  -- (Function definition ensuring v(x) = x^2 * sin(π * x) and the symmetry property)
  
  have symmetry : ∀ x : ℝ, v (-x) = -v (x) := 
    by
      intro x
      unfold v
      rw [neg_square, Real.sin_neg]
      ring,
  
  -- Using symmetry property to simplify sums
  rw [symmetry 1.5, symmetry 0.5, neg_add_eq_zero, neg_add_eq_zero],
  -- Proof is completed
  refl,
  sorry

end sum_of_v_values_is_zero_l642_642333


namespace stamping_possible_l642_642464

/-- The problem states that Petya uses a wooden square stamp with 102 cells painted black, and presses it 100 times on a white sheet of paper.
  We need to prove that it is possible to cover a 101 × 101 square such that every cell except one is black. -/
theorem stamping_possible : 
  ∀ (s : Nat) (count : Nat), s = 102 → count = 100 → 
  let n := 101 in
  ∃ (grid : Fin n × Fin n → Prop), 
  (∀ (p : Fin n × Fin n), p ≠ (0, 0) → grid p) ∧ 
  (∀ (stamp : Fin 102 → Fin n × Fin n), True) :=
begin
  sorry,
end

end stamping_possible_l642_642464


namespace sum_of_values_b_for_quadratic_l642_642736

theorem sum_of_values_b_for_quadratic :
  (∑ b, 3 * x^2 + (b + 6) * x + 4 = 0 → b = -6 + 4 * real.sqrt 3 ∨ b = -6 - 4 * real.sqrt 3) →
  (b + 6 - 4 * real.sqrt 3) + (b + 6 + 4 * real.sqrt 3) = -12 :=
by
  sorry

end sum_of_values_b_for_quadratic_l642_642736


namespace max_n_inequality_l642_642419

noncomputable theory

def sequence (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | (n + 1) => sequence n + 2

theorem max_n_inequality (a : ℕ → ℕ) (h₁ : a 1 = 1) (h₂ : ∀ n, a (n + 1) = a n + 2)
  (h₃ : (finset.range n).sum (λ k, 1 / ((a k) * (a (k + 1)))) ≤ 1009 / 2019)
  : n ≤ 1009 :=
begin
  sorry
end

end max_n_inequality_l642_642419


namespace midpoint_halfway_l642_642747

theorem midpoint_halfway (A B : ℝ × ℝ) (hA : A = (2, 9)) (hB : B = (6, -3)) : 
  let M := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  in M = (4, 3) :=
by
  -- Assuming the coordinates for the calculation here
  have hA1 : A.1 = 2 := by rw [hA]; rfl
  have hA2 : A.2 = 9 := by rw [hA]; rfl
  have hB1 : B.1 = 6 := by rw [hB]; rfl
  have hB2 : B.2 = -3 := by rw [hB]; rfl
  -- Define M
  let M := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  -- Simplify M values
  have Mx : M.1 = (A.1 + B.1) / 2 := rfl
  have My : M.2 = (A.2 + B.2) / 2 := rfl
  -- Substitute A.1, A.2, B.1, B.2
  rw [hA1, hA2, hB1, hB2] at Mx My
  -- Verify M equals to (4, 3)
  exact ⟨Mx, My⟩
  sorry

end midpoint_halfway_l642_642747


namespace mary_regular_hours_l642_642911

theorem mary_regular_hours :
  ∃ (x : ℕ), x * 8 + (50 - x) * 10 = 460 ∧ x ≤ 50 ∧ x = 20 := by
s sorry

end mary_regular_hours_l642_642911


namespace mean_median_mode_l642_642374

theorem mean_median_mode (m n : ℕ) (h1 : 0 < m) (h2 : 0 < n) 
  (h3 : m + 7 < n) 
  (h4 : (m + (m + 3) + (m + 7) + n + (n + 5) + (2 * n - 1)) / 6 = n)
  (h5 : ((m + 7) + n) / 2 = n)
  (h6 : (m+3 < m+7 ∧ m+7 = n ∧ n < n+5 ∧ n+5 < 2*n - 1 )) :
  m+n = 2*n := by
  sorry

end mean_median_mode_l642_642374


namespace cos_150_eq_neg_half_l642_642282

theorem cos_150_eq_neg_half : Real.cos (150 * Real.pi / 180) = -1 / 2 := 
by
  sorry

end cos_150_eq_neg_half_l642_642282


namespace find_certain_number_l642_642661

theorem find_certain_number (x : ℕ) : 
  220050 = (x + 445) * (2 * (x - 445)) + 50 → x = 555 := 
begin
  sorry
end

end find_certain_number_l642_642661


namespace cos_150_degree_l642_642175

theorem cos_150_degree : Real.cos (150 * Real.pi / 180) = -Real.sqrt 3 / 2 :=
by
  sorry

end cos_150_degree_l642_642175


namespace aira_fewer_bands_than_joe_l642_642937

-- Define initial conditions
variables (samantha_bands aira_bands joe_bands : ℕ)
variables (shares_each : ℕ) (total_bands: ℕ)

-- Conditions from the problem
axiom h1 : shares_each = 6
axiom h2 : samantha_bands = aira_bands + 5
axiom h3 : total_bands = shares_each * 3
axiom h4 : aira_bands = 4
axiom h5 : samantha_bands + aira_bands + joe_bands = total_bands

-- The statement to be proven
theorem aira_fewer_bands_than_joe : joe_bands - aira_bands = 1 :=
sorry

end aira_fewer_bands_than_joe_l642_642937


namespace find_chlorine_atoms_l642_642075

def num_of_chlorine_atoms (molecular_weight : ℝ) (atomic_weight_Ba : ℝ) (atomic_weight_Cl : ℝ) : ℕ :=
  let n := (molecular_weight - atomic_weight_Ba) / atomic_weight_Cl
  n.round

theorem find_chlorine_atoms :
  ∀ (molecular_weight : ℝ) (atomic_weight_Ba : ℝ) (atomic_weight_Cl : ℝ),
  molecular_weight = 207 → atomic_weight_Ba = 137.33 → atomic_weight_Cl = 35.45 →
  num_of_chlorine_atoms molecular_weight atomic_weight_Ba atomic_weight_Cl = 2 :=
begin
  intros,
  sorry
end

end find_chlorine_atoms_l642_642075


namespace cos_150_eq_neg_sqrt3_over_2_l642_642213

theorem cos_150_eq_neg_sqrt3_over_2 :
  ∀ {deg cosQ1 cosQ2: ℝ},
    cosQ1 = 30 →
    cosQ2 = 180 - cosQ1 →
    cos 150 = - (cos cosQ1) →
    cos cosQ1 = sqrt 3 / 2 →
    cos 150 = -sqrt 3 / 2 :=
by
  intros deg cosQ1 cosQ2 h1 h2 h3 h4
  sorry

end cos_150_eq_neg_sqrt3_over_2_l642_642213


namespace ellipse_formula_line_formula_l642_642791

-- Definition for the equation of the ellipse C
def ellipse_eq (a b x y : ℝ) : Prop := (x^2 / a^2) + (y^2 / b^2) = 1

-- Conditions provided in the problem.
structure EllipseCondition :=
  (origin : (0, 0) ∈ ℝ × ℝ)
  (foci_axes : ∃ c, c > 0 ∧ (∀ (x y : ℝ), (x, y)^2 = 1 ↔ (x^2 / c^2) + (y^2 / (c^2 - 1)) = 1))
  (line_intersects_M : ∃ x y : ℝ, y = (3 / 2) * x ∧ ellipse_eq 2 (sqrt 3) x y)
  (proj_M_on_x : ∃ c : ℝ, c > 0 ∧ (x, y) = (c, (3 / 2) * c))
  (foci_in_x_coordinate : (1, 0) ∈ ℝ × ℝ ∧ (-1, 0) ∈ ℝ × ℝ)
  (dot_product_gives : ∀ (x y x' y' : ℝ), x * x' + y * y' = 9 / 4)

-- Define the proof problem in Lean.
theorem ellipse_formula (cond : EllipseCondition) : ellipse_eq 2 (sqrt 3) 
   := sorry

-- Definitions for the second part of the problem.
structure LineCondition :=
  (A : (2, 0) ∈ ℝ × ℝ)
  (perpendicular_bisector : ∃ P : ℝ × ℝ, ∠ PQ ∧ ∠ BFS)
  (intersection_with_C : ∃ B : ℝ × ℝ, is_intersection (B) (ellipse_eq 2 (sqrt 3) B))
  (Q_perpendicular_to_y : ∃ Q : ℝ × ℝ, Q ⊥ y)

theorem line_formula (cond : LineCondition) : ∃ k : ℝ, y = k * (x - 2) ∧ (k = sqrt(6)/4 ∨ k = -sqrt(6)/4) 
  := sorry

end ellipse_formula_line_formula_l642_642791


namespace minimum_f_l642_642753

noncomputable theory

def C_n (n : ℕ) : set (set (ℝ × ℝ)) :=
  { C | ∃ c ∈ C, ∀ x y z ∈ C, x ≠ y ∧ y ≠ z ∧ z ≠ x → 
    let a : ℝ × ℝ := x in
    let b : ℝ × ℝ := y in
    let c : ℝ × ℝ := z in
    let area := (1 / 2) * abs ((b.1 - a.1) * (c.2 - a.2) - (b.2 - a.2) * (c.1 - a.1)) in
    area > 1 }

def f (n : ℕ) (C : set (ℝ × ℝ)) : ℕ :=
  max { k | ∃ (A : set (ℝ × ℝ)), A ⊆ C ∧ A.card = k ∧ ∀ x y ∈ A, x ≠ y → 
    dist x y > 2 }

def f_min (n : ℕ) : ℕ :=
  min { f n C | C ∈ C_n n }

theorem minimum_f (n : ℕ) (h : n ≥ 4) : f_min n = (n + 2) / 3 :=
by sorry

end minimum_f_l642_642753


namespace cos_150_eq_neg_sqrt3_over_2_l642_642207

theorem cos_150_eq_neg_sqrt3_over_2 :
  ∀ {deg cosQ1 cosQ2: ℝ},
    cosQ1 = 30 →
    cosQ2 = 180 - cosQ1 →
    cos 150 = - (cos cosQ1) →
    cos cosQ1 = sqrt 3 / 2 →
    cos 150 = -sqrt 3 / 2 :=
by
  intros deg cosQ1 cosQ2 h1 h2 h3 h4
  sorry

end cos_150_eq_neg_sqrt3_over_2_l642_642207


namespace cos_150_deg_l642_642321

-- Define the conditions as variables
def angle1 := 150
def angle2 := 180
def angle3 := 30
def cos30 := Real.sqrt 3 / 2

-- The main statement to be proved
theorem cos_150_deg : Real.cos (150 * Real.pi / 180) = - (Real.sqrt 3 / 2) :=
by
  sorry

end cos_150_deg_l642_642321


namespace cos_150_eq_neg_sqrt3_div_2_l642_642313

open Real

theorem cos_150_eq_neg_sqrt3_div_2 :
  cos (150 * pi / 180) = - (sqrt 3 / 2) := 
sorry

end cos_150_eq_neg_sqrt3_div_2_l642_642313


namespace total_birds_seen_l642_642066

theorem total_birds_seen (M T W : ℕ) 
  (hM : M = 70)
  (hT : T = M / 2)
  (hW : W = T + 8) : 
  M + T + W = 148 :=
by
  subst hM
  subst hT
  subst hW
  have hT' : 35 = 70 / 2 := by norm_num
  have hW' : 43 = 35 + 8 := by norm_num
  rw [hT', hW']
  norm_num

end total_birds_seen_l642_642066


namespace bottom_right_corner_is_one_l642_642739

-- Define the 3x3 grid as a function from coordinates to values
def Grid := Array (Array ℕ)

-- The problem conditions
constant grid : Grid
axiom unique_numbers (i j : ℕ) (h1 : i ∈ finset.range 3) (h2 : j ∈ finset.range 3) :
  ∃ k, k ∈ finset.range 1 10 ∧ ∀ l (h3 : l ∈ finset.range 1 10), ∀ i2 j2, (i2 ≠ i ∨ j2 ≠ j) → grid[i][j] ≠ k

axiom consecutive_adjacent (n m i j : ℕ) (h1 : grid[i][j] = n) (h2 : abs (n - m) = 1) :
  ∃ i2 j2, (i2, j2) ∈ [(i+1, j), (i-1, j), (i, j+1), (i, j-1)] ∧ grid[i2][j2] = m

axiom corners_sum_to_24 :
  grid[0][0] + grid[0][2] + grid[2][0] + grid[2][2] = 24

axiom center_and_neighbors_sum_to_25 :
  grid[1][1] + grid[0][1] + grid[1][0] + grid[1][2] + grid[2][1] = 25

-- The theorem to prove
theorem bottom_right_corner_is_one : grid[2][2] = 1 :=
sorry

end bottom_right_corner_is_one_l642_642739


namespace cos_150_degree_l642_642174

theorem cos_150_degree : Real.cos (150 * Real.pi / 180) = -Real.sqrt 3 / 2 :=
by
  sorry

end cos_150_degree_l642_642174


namespace vasya_made_a_mistake_l642_642020

theorem vasya_made_a_mistake (A B V G D E : ℕ)
  (h1 : A ≠ B)
  (h2 : V ≠ G)
  (h3 : (10 * A + B) * (10 * V + G) = 1000 * D + 100 * D + 10 * E + E)
  (h4 : ∀ {X Y : ℕ}, X ≠ Y → D ≠ E) :
  False :=
by
  -- Proof goes here (skipped)
  sorry

end vasya_made_a_mistake_l642_642020


namespace prob_normal_5_6_l642_642797

open ProbabilityTheory

namespace Proof

constant x : ℝ
constant μ : ℝ := 4
constant σ : ℝ := 1

axiom normal_dist : ∀ x, x ∼ Normal μ σ

theorem prob_normal_5_6
  (h_mean : μ = 4)
  (h_stddev : σ = 1)
  (h_P_2sigma : ∀ (X : ℝ), P (μ - 2 * σ < X ∧ X ≤ μ + 2 * σ) = 0.9544)
  (h_P_sigma : ∀ (X : ℝ), P (μ - σ < X ∧ X ≤ μ + σ) = 0.6826) :
  P (5 < x ∧ x < 6) = 0.1359 :=
sorry

end Proof

end prob_normal_5_6_l642_642797


namespace length_of_paving_stone_l642_642983

theorem length_of_paving_stone (courtyard_length courtyard_width : ℝ)
  (num_paving_stones : ℕ) (paving_stone_width : ℝ) (total_area : ℝ)
  (paving_stone_length : ℝ) : 
  courtyard_length = 70 ∧ courtyard_width = 16.5 ∧ num_paving_stones = 231 ∧ paving_stone_width = 2 ∧ total_area = courtyard_length * courtyard_width ∧ total_area = num_paving_stones * paving_stone_length * paving_stone_width → 
  paving_stone_length = 2.5 :=
by
  sorry

end length_of_paving_stone_l642_642983


namespace anthony_pencils_l642_642689

theorem anthony_pencils (P : Nat) (h : P + 56 = 65) : P = 9 :=
by
  sorry

end anthony_pencils_l642_642689


namespace NES_original_price_l642_642602

-- Definitions and conditions
def SNES_value : ℝ := 150
def SNES_credit_rate : ℝ := 0.80
def Gameboy_value : ℝ := 50
def Gameboy_credit_rate : ℝ := 0.75
def PS2_value : ℝ := 100
def PS2_credit_rate : ℝ := 0.60
def payment_given : ℝ := 100
def change_received : ℝ := 12
def discount_rate : ℝ := 0.20
def sales_tax_rate : ℝ := 0.08

-- Calculated intermediate credits
def SNES_credit := SNES_credit_rate * SNES_value
def Gameboy_credit := Gameboy_credit_rate * Gameboy_value
def PS2_credit := PS2_credit_rate * PS2_value

-- Total credit calculation
def total_credit := SNES_credit + Gameboy_credit + PS2_credit

-- Payment amount after applying change
def net_payment := payment_given - change_received

-- Sale price before tax
def sale_price_before_tax := net_payment / (1 + sales_tax_rate)

-- Original price before discount
def original_NES_price := sale_price_before_tax / (1 - discount_rate)

-- Main theorem to prove
theorem NES_original_price :
  original_NES_price = 101.85 :=
begin
  sorry -- proof to be completed
end

end NES_original_price_l642_642602


namespace infinite_triples_sum_of_squares_all_three_representable_l642_642932

theorem infinite_triples_sum_of_squares :
  ∃ᶠ n : ℕ in at_top, 
  (∃ a b : ℕ, n = a^2 + b^2) ∧ ∀ m ∈ {n-1, n+1}, ¬(∃ a b : ℕ, m = a^2 + b^2) :=
sorry

theorem all_three_representable :
  ∃ᶠ n : ℕ in at_top, 
  ∀ m ∈ {(n-1), n, (n+1)}, (∃ a b : ℕ, m = a^2 + b^2) :=
sorry

end infinite_triples_sum_of_squares_all_three_representable_l642_642932


namespace cos_150_eq_neg_sqrt3_div_2_l642_642308

open Real

theorem cos_150_eq_neg_sqrt3_div_2 :
  cos (150 * pi / 180) = - (sqrt 3 / 2) := 
sorry

end cos_150_eq_neg_sqrt3_div_2_l642_642308


namespace cos_150_eq_neg_sqrt3_over_2_l642_642206

theorem cos_150_eq_neg_sqrt3_over_2 :
  ∀ {deg cosQ1 cosQ2: ℝ},
    cosQ1 = 30 →
    cosQ2 = 180 - cosQ1 →
    cos 150 = - (cos cosQ1) →
    cos cosQ1 = sqrt 3 / 2 →
    cos 150 = -sqrt 3 / 2 :=
by
  intros deg cosQ1 cosQ2 h1 h2 h3 h4
  sorry

end cos_150_eq_neg_sqrt3_over_2_l642_642206


namespace inconsistent_probabilities_l642_642042

-- Definitions from conditions
def P (event : Type) : ℝ := sorry

axiom P_A : P A = 6/17
axiom P_B : P B = 5/17
axiom P_A_union_B : P (A ∪ B) = 4/17

-- Theorem statement
theorem inconsistent_probabilities : ¬ (P (A ∪ B) ≤ P A ∧ P (A ∪ B) ≤ P B) :=
sorry

end inconsistent_probabilities_l642_642042


namespace cos_150_eq_negative_cos_30_l642_642145

theorem cos_150_eq_negative_cos_30 :
  Real.cos (150 * Real.pi / 180) = - (Real.cos (30 * Real.pi / 180)) :=
by sorry

end cos_150_eq_negative_cos_30_l642_642145


namespace midpoint_BD_on_common_tangent_l642_642078

-- Definitions and conditions
variables (A B C D : Point) (quadrilateral_convex : ConvexQuadrilateral A B C D)
variables (angle_B_eq_angle_D : angle A B C = angle A D C)

-- To Prove: The midpoint of the diagonal BD lies on the common internal tangent to the incircles of triangles △ABC and △ACD.
theorem midpoint_BD_on_common_tangent (midpoint_N : Point)
  (mid_BD : midpoint B D = midpoint_N) :
  lies_on_common_internal_tangent midpoint_N 
    (incircle_triangle A B C) (incircle_triangle A C D) :=
sorry

end midpoint_BD_on_common_tangent_l642_642078


namespace field_length_to_width_ratio_l642_642578

theorem field_length_to_width_ratio
(W L : ℕ) (P : ℕ) 
(hW : W = 80)
(hP : P = 384)
(hP_def : P = 2 * L + 2 * W) :
  (L : ℚ) / W = 7 / 5 :=
by {
  -- Definitions from problem conditions
  have hW_def : W = 80 := hW,
  have hP_def' : P = 384 := hP,
  have hP_eq : P = 2 * L + 2 * W := hP_def,
  
  -- Defining lengths and solving for ratio
  sorry
}

end field_length_to_width_ratio_l642_642578


namespace max_infinite_occurrences_sequence_l642_642948

theorem max_infinite_occurrences_sequence (N : ℕ) (a : ℕ → ℕ) 
  (h : ∀ n, n ≥ N → a n = { i : ℕ | i < n ∧ a i + i ≥ n }.card) : 
  ∃ s, (∀ n, s a n) ∧ (∀ k ∈ s, ∀₀ n, s.contains k a n) → s.card ≤ 2 :=
sorry

end max_infinite_occurrences_sequence_l642_642948


namespace cos_150_eq_neg_sqrt3_div_2_l642_642305

open Real

theorem cos_150_eq_neg_sqrt3_div_2 :
  cos (150 * pi / 180) = - (sqrt 3 / 2) := 
sorry

end cos_150_eq_neg_sqrt3_div_2_l642_642305


namespace number_exceeds_its_3_over_8_part_by_20_l642_642624

theorem number_exceeds_its_3_over_8_part_by_20 (x : ℝ) (h : x = (3 / 8) * x + 20) : x = 32 :=
by
  sorry

end number_exceeds_its_3_over_8_part_by_20_l642_642624


namespace part_1_part_3_500_units_part_3_1000_units_l642_642080

/-- Define the pricing function P as per the given conditions -/
def P (x : ℕ) : ℝ :=
  if 0 < x ∧ x ≤ 100 then 60
  else if 100 < x ∧ x <= 550 then 62 - 0.02 * x
  else 51

/-- Verify that ordering 550 units results in a per-unit price of 51 yuan -/
theorem part_1 : P 550 = 51 := sorry

/-- Compute profit for given order quantities -/
def profit (x : ℕ) : ℝ :=
  x * (P x - 40)

/-- Verify that an order of 500 units results in a profit of 6000 yuan -/
theorem part_3_500_units : profit 500 = 6000 := sorry

/-- Verify that an order of 1000 units results in a profit of 11000 yuan -/
theorem part_3_1000_units : profit 1000 = 11000 := sorry

end part_1_part_3_500_units_part_3_1000_units_l642_642080


namespace chord_length_is_sqrt6_line_pq_is_correct_l642_642854

section Problem1

-- Definitions for problem (part 1)
def circle_parametric (θ : ℝ) : ℝ × ℝ := (real.sqrt 2 * real.cos θ, real.sqrt 2 * real.sin θ)
def line_parametric (t : ℝ) : ℝ × ℝ := (1 + t, 2 + t)
def center_circle : ℝ × ℝ := (0, 0)
def radius_circle : ℝ := real.sqrt 2
def distance_from_center_to_line : ℝ := (1 : ℝ) / (real.sqrt 2)
def chord_length : ℝ := 2 * real.sqrt ((radius_circle ^ 2) - (distance_from_center_to_line ^ 2))

-- Proving the length of the chord AB
theorem chord_length_is_sqrt6 : chord_length = real.sqrt 6 := by
  sorry

end Problem1

section Problem2

-- Definitions for problem (part 2)
def polar_to_cartesian_equation (ρ θ : ℝ) : ℝ × ℝ := (ρ * real.cos θ, ρ * real.sin θ)
def circle_polar (θ : ℝ) : ℝ := 2 * real.cos θ + 2 * real.sin θ * real.sqrt 3
def cartesian_equation_circle2 (x y : ℝ) : Prop := (x ^ 2 + y ^ 2 = 2 * x + 2 * real.sqrt 3 * y)
def cartesian_equation_circleO (x y : ℝ) : Prop := (x ^ 2 + y ^ 2 = 2)
def line_pq (x y : ℝ) : Prop := (x + real.sqrt 3 * y - 1 = 0)

-- Proving the equation of the line containing PQ
theorem line_pq_is_correct : ∀ x y : ℝ, cartesian_equation_circleO x y → cartesian_equation_circle2 x y → line_pq x y := by
  sorry

end Problem2

end chord_length_is_sqrt6_line_pq_is_correct_l642_642854


namespace arithmetic_expression_evaluation_l642_642611

theorem arithmetic_expression_evaluation : (8 / 2 - 3 * 2 + 5^2 / 5) = 3 := by
  sorry

end arithmetic_expression_evaluation_l642_642611


namespace closest_percentage_change_l642_642909

/--
Problem Statement: Mary is about to pay for five items at the grocery store. 
The prices of the items are $7.99$, $4.99$, $2.99$, $1.99$, and $0.99$. 
Mary will pay with a twenty-dollar bill. Which of the following is closest 
to the percentage of the $20.00 that she will receive in change? 
Options are 5%, 10%, 15%, 20%, 25%. 

Proof: The calculated total price of the items is $18.95$. Mary will receive 
$1.05 in change. The percentage of change with respect to $20.00 is $5.25\%$. 
The closest option is 5%. 
-/
theorem closest_percentage_change :
  let prices := [7.99, 4.99, 2.99, 1.99, 0.99]
  let total := prices.foldl (· + ·) 0
  let paid := 20.00
  let change := paid - total
  let percentage_change := (change / paid) * 100
  abs (percentage_change - 5.0) < abs (percentage_change - 10.0) ∧
  abs (percentage_change - 5.0) < abs (percentage_change - 15.0) ∧
  abs (percentage_change - 5.0) < abs (percentage_change - 20.0) ∧
  abs (percentage_change - 5.0) < abs (percentage_change - 25.0) :=
by
  sorry

end closest_percentage_change_l642_642909


namespace total_volume_calculation_l642_642617

noncomputable def total_volume_of_four_cubes (edge_length_in_feet : ℝ) (conversion_factor : ℝ) : ℝ :=
  let edge_length_in_meters := edge_length_in_feet * conversion_factor
  let volume_of_one_cube := edge_length_in_meters^3
  4 * volume_of_one_cube

theorem total_volume_calculation :
  total_volume_of_four_cubes 5 0.3048 = 14.144 :=
by
  -- Proof needs to be filled in.
  sorry

end total_volume_calculation_l642_642617


namespace solve_equation_l642_642588

theorem solve_equation (x : ℝ) : 4 ^ x + 2 ^ x - 2 = 0 ↔ x = 0 := by
  sorry

end solve_equation_l642_642588


namespace no_parallelepiped_bricks_3x5x7_no_parallelepiped_bricks_2x5x6_l642_642493

theorem no_parallelepiped_bricks_3x5x7 
    (V_wall : ℕ) 
    (V_brick : ℕ) 
    (dimensions_wall : V_wall = 27 * 16 * 15) 
    (dimensions_brick : V_brick = 3 * 5 * 7) : 
    ¬ V_wall % V_brick = 0 :=
by {
    rw [dimensions_wall, dimensions_brick],
    sorry
}

theorem no_parallelepiped_bricks_2x5x6 
    (A_face : ℕ) 
    (brick_face1 : ℕ) 
    (brick_face2 : ℕ) 
    (brick_face3 : ℕ) 
    (dimensions_face : A_face = 27 * 15) 
    (dimension_face1 : brick_face1 = 2 * 5) 
    (dimension_face2 : brick_face2 = 5 * 6) 
    (dimension_face3 : brick_face3 = 2 * 6) : 
    ¬ (A_face % brick_face1 = 0 ∨ A_face % brick_face2 = 0 ∨ A_face % brick_face3 = 0) :=
by {
    rw [dimensions_face, dimension_face1, dimension_face2, dimension_face3],
    sorry
}

end no_parallelepiped_bricks_3x5x7_no_parallelepiped_bricks_2x5x6_l642_642493


namespace find_f_g_of_neg_2_l642_642397

variable (f g : ℝ → ℝ)

-- Function definition
def f_def : ℝ → ℝ :=
  λ x, if x > 0 then 2^x - 3 else g x

-- Odd function property
axiom f_odd (x : ℝ) : f_def f g (-x) = -f_def f g x

-- Goal
theorem find_f_g_of_neg_2 : f_def f g (g (-2)) = -1 :=
sorry

end find_f_g_of_neg_2_l642_642397


namespace total_birds_seen_l642_642067

theorem total_birds_seen (M T W : ℕ) 
  (hM : M = 70)
  (hT : T = M / 2)
  (hW : W = T + 8) : 
  M + T + W = 148 :=
by
  subst hM
  subst hT
  subst hW
  have hT' : 35 = 70 / 2 := by norm_num
  have hW' : 43 = 35 + 8 := by norm_num
  rw [hT', hW']
  norm_num

end total_birds_seen_l642_642067


namespace min_n_1_add_i_pow_n_real_l642_642824

theorem min_n_1_add_i_pow_n_real : ∃ n : ℕ, 0 < n ∧ (1 + complex.i)^n ∈ ℝ ∧ (∀ m : ℕ, 0 < m ∧ (1 + complex.i)^m ∈ ℝ → n ≤ m) :=
sorry

end min_n_1_add_i_pow_n_real_l642_642824


namespace octal_subtraction_541_276_l642_642744

def octal_subtraction : ℕ → ℕ → ℕ 
| a b := sorry  -- Dummy function definition for the subtraction in octal

-- Definitions for the octal numbers as natural numbers
def base8_541 : ℕ := 5 * 8^2 + 4 * 8 + 1
def base8_276 : ℕ := 2 * 8^2 + 7 * 8 + 6
def base8_243 : ℕ := 2 * 8^2 + 4 * 8 + 3

theorem octal_subtraction_541_276 : octal_subtraction base8_541 base8_276 = base8_243 :=
by
  sorry

end octal_subtraction_541_276_l642_642744


namespace abs_inequality_solution_set_l642_642587

theorem abs_inequality_solution_set {x : ℝ} : (|x - 2| < 1) ↔ (1 < x ∧ x < 3) :=
sorry

end abs_inequality_solution_set_l642_642587


namespace max_pages_copied_l642_642498

-- Definitions based on conditions
def cents_per_page := 7 / 4
def budget_cents := 1500

-- The theorem to prove
theorem max_pages_copied (c : ℝ) (budget : ℝ) (h₁ : c = cents_per_page) (h₂ : budget = budget_cents) : 
  ⌊(budget / c)⌋ = 857 :=
sorry

end max_pages_copied_l642_642498


namespace cos_150_eq_neg_sqrt3_div_2_l642_642164

theorem cos_150_eq_neg_sqrt3_div_2 : Real.cos (150 * Real.pi / 180) = - (Real.sqrt 3 / 2) :=
by
  sorry

end cos_150_eq_neg_sqrt3_div_2_l642_642164


namespace triangle_inequality_l642_642930

theorem triangle_inequality (a b c s R r : ℝ) (h_s : s = (a + b + c) / 2) 
  (h_abc : a > 0 ∧ b > 0 ∧ c > 0 ∧ R > 0 ∧ r > 0) :
  (a / (s - a)) + (b / (s - b)) + (c / (s - c)) ≥ (3 * R / r) :=
begin
  sorry
end

end triangle_inequality_l642_642930


namespace no_finite_set_S_l642_642731

theorem no_finite_set_S (S : Finset ℕ) (hS : ∀ s ∈ S, Nat.Prime s) : ¬(∀ n ≥ 2, ∃ s ∈ S, (∑ i in Finset.range (n + 1), i^2) - 1 % s = 0) :=
begin
  sorry
end

end no_finite_set_S_l642_642731


namespace subset_probability_exactly_consecutive_l642_642421

open Set

-- Define the set M
def M : Set ℕ := {1, 2, 3, 4, 5}

-- Define a function to check if the elements of a set are exactly consecutive integers
def exactly_consecutive (S : Set ℕ) : Prop := 
  ∀ (a ∈ S) (b ∈ S), a < b → b = a + card S - 1

-- Define the condition that a subset must have at least two elements
def at_least_two_elements (S : Set ℕ) : Prop :=
  2 ≤ S.card

-- Define the main statement to prove the probability is 5/13
theorem subset_probability_exactly_consecutive :
  (∑ S in powerset M, if at_least_two_elements S ∧ exactly_consecutive S then 1 else 0 : ℚ) / 
  (∑ S in powerset M, if at_least_two_elements S then 1 else 0 : ℚ) = 5 / 13 :=
by sorry

end subset_probability_exactly_consecutive_l642_642421


namespace letters_into_mailboxes_l642_642933

theorem letters_into_mailboxes (letters mailboxes : ℕ) (h_letters : letters = 5) (h_mailboxes : mailboxes = 3) :
  (mailboxes ^ letters) = 3 ^ 5 :=
by
  rw [h_letters, h_mailboxes]
  exact rfl

end letters_into_mailboxes_l642_642933


namespace large_planks_need_15_nails_l642_642501

-- Definitions based on given conditions
def total_nails : ℕ := 20
def small_planks_nails : ℕ := 5

-- Question: How many nails do the large planks need together?
-- Prove that the large planks need 15 nails together given the conditions.
theorem large_planks_need_15_nails : total_nails - small_planks_nails = 15 :=
by
  sorry

end large_planks_need_15_nails_l642_642501


namespace one_percent_as_decimal_l642_642821

theorem one_percent_as_decimal : (1 / 100 : ℝ) = 0.01 := by
  sorry

end one_percent_as_decimal_l642_642821


namespace factorial_division_l642_642331

theorem factorial_division : (10.fact / (6.fact * 4.fact) = 210) :=
by
  sorry

end factorial_division_l642_642331


namespace cos_150_eq_neg_sqrt3_over_2_l642_642204

theorem cos_150_eq_neg_sqrt3_over_2 :
  ∀ {deg cosQ1 cosQ2: ℝ},
    cosQ1 = 30 →
    cosQ2 = 180 - cosQ1 →
    cos 150 = - (cos cosQ1) →
    cos cosQ1 = sqrt 3 / 2 →
    cos 150 = -sqrt 3 / 2 :=
by
  intros deg cosQ1 cosQ2 h1 h2 h3 h4
  sorry

end cos_150_eq_neg_sqrt3_over_2_l642_642204


namespace angle_between_α_β_l642_642398

variable (A B C : EucVec) (angleA : RealAngle) (angleC : RealAngle) (angleB : RealAngle)
variable (BA CA BC CB : Vector)  
variable (cosA cosB cosC : ℝ)

def α := (BA / (∥BA∥ * cosA)) + (BC / (∥BC∥ * cosC))
def β := (CA / (∥CA∥ * cosA)) + (CB / (∥CB∥ * cosB))

theorem angle_between_α_β 
  (hA : angleA = 120)
  (hCA_0 : α • CA = 0) 
  (hBA_0 : β • BA = 0) 
  (cos_pos : cosB * cosC > 0) : 
  ∠α β = 60 :=
sorry


end angle_between_α_β_l642_642398


namespace number_of_ordered_pairs_l642_642720

def cycle_i (n : ℤ) : ℂ :=
  match n % 4 with
  | 0 => 1
  | 1 => complex.I
  | 2 => -1
  | 3 => -complex.I
  | _ => 0 -- this line should never be reached due to % 4

theorem number_of_ordered_pairs :
  (∃ pairs : list (ℤ × ℤ), 
     (∀ (x y : ℤ), (x, y) ∈ pairs → 25 ≤ x ∧ x < y ∧ y ≤ 50 ∧ cycle_i x + cycle_i y = 0)
     ∧ pairs.length = 11) :=
by
  sorry

end number_of_ordered_pairs_l642_642720


namespace total_cubes_on_highlighted_cells_l642_642537

-- Conditions
def board_size := 4
def cubes_on_cells := [2, 2, 2, 2, 2, 2, 2, 2] -- Represents 2 cells per height from 1 to 8 cubes

-- Theorem statement
theorem total_cubes_on_highlighted_cells : 
  let highlighted_cells := [some_identified_position_1, some_identified_position_2] in
  count_cubes highlighted_cells = 13 :=
sorry

end total_cubes_on_highlighted_cells_l642_642537


namespace intersection_spheres_integer_points_count_l642_642334

theorem intersection_spheres_integer_points_count :
  let sphere1 := {p : ℝ × ℝ × ℝ | p.1^2 + p.2^2 + (p.3 - 10)^2 ≤ 64}
  let sphere2 := {p : ℝ × ℝ × ℝ | p.1^2 + p.2^2 + (p.3 - 2)^2 ≤ 25}
  let integer_points := {p : ℤ × ℤ × ℤ | (p.1, p.2, p.3) ∈ sphere1 ∧ (p.1, p.2, p.3) ∈ sphere2}
  {p : integer_points}.card = 294 := sorry

end intersection_spheres_integer_points_count_l642_642334


namespace cos_150_eq_neg_sqrt3_div_2_l642_642157

theorem cos_150_eq_neg_sqrt3_div_2 : Real.cos (150 * Real.pi / 180) = - (Real.sqrt 3 / 2) :=
by
  sorry

end cos_150_eq_neg_sqrt3_div_2_l642_642157


namespace cos_150_eq_neg_sqrt3_over_2_l642_642216

theorem cos_150_eq_neg_sqrt3_over_2 :
  ∀ {deg cosQ1 cosQ2: ℝ},
    cosQ1 = 30 →
    cosQ2 = 180 - cosQ1 →
    cos 150 = - (cos cosQ1) →
    cos cosQ1 = sqrt 3 / 2 →
    cos 150 = -sqrt 3 / 2 :=
by
  intros deg cosQ1 cosQ2 h1 h2 h3 h4
  sorry

end cos_150_eq_neg_sqrt3_over_2_l642_642216


namespace find_f_2013_l642_642944

noncomputable def f : ℝ → ℝ := sorry

theorem find_f_2013 (x : ℝ) (h1 : ∀ x : ℝ, f(x+6) ≤ f(x+2) + 4) 
                     (h2 : ∀ x : ℝ, f(x+4) ≥ f(x+2) + 2) 
                     (h3 : f 1 = 1) : 
                     f 2013 = 2013 := 
by
  sorry

end find_f_2013_l642_642944


namespace certain_number_divisible_l642_642447

theorem certain_number_divisible (x : ℤ) (n : ℤ) (h1 : 0 < n ∧ n < 11) (h2 : x - n = 11 * k) (h3 : n = 1) : x = 12 :=
by sorry

end certain_number_divisible_l642_642447


namespace cube_face_sum_l642_642582

-- Define the number list and cube properties
def numbers : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9]
def is_cube_face_sum_equal (f : ℕ → ℕ → ℕ → ℕ → Prop) : Prop :=
  ∀ (a b c d e f g h i : ℕ), 
    {a, b, c, d, e, f, g, h, i} = {1, 2, 3, 4, 5, 6, 7, 8, 9} → 
    let S := (45 : ℕ) in
    3 * S = 135 ∧ 
    6 * (b : ℕ -> ℕ -> ℕ -> ℕ -> ℕ) = 22.5 * 6

theorem cube_face_sum (f : ℕ → ℕ → ℕ → ℕ → Prop) 
  (h : is_cube_face_sum_equal f) : 
  ∃ S, 
    S = 22.5 := 
by 
  sorry

end cube_face_sum_l642_642582


namespace sum_mean_median_mode_l642_642616

theorem sum_mean_median_mode : 
  let data := [2, 5, 1, 5, 2, 6, 1, 5, 0, 2]
  let ordered_data := [0, 1, 1, 2, 2, 2, 5, 5, 5, 6]
  let mean := (0 + 1 + 1 + 2 + 2 + 2 + 5 + 5 + 5 + 6) / 10
  let median := (2 + 2) / 2
  let mode := 5
  mean + median + mode = 9.9 := by
  sorry

end sum_mean_median_mode_l642_642616


namespace probability_three_common_l642_642535

open Nat

noncomputable def binom (n k : ℕ) := choose n k

theorem probability_three_common (n k m : ℕ) (h1 : n = 12) (h2 : k = 7) (h3 : m = 3)
    (h: (binom n k * binom n k) ≠ 0) :
    (binom n m * binom (n - m) (k - m) * binom (n - m) (k - m)) / (binom n k * binom n k) = 3502800 / 627264 :=
by
  rw [h1, h2, h3]
  have : binom 12 3 = 220 := by decide
  have : binom 9 4 = 126 := by decide
  have : binom 12 7 = 792 := by decide
  field_simp [this]
  norm_num
  have : 220 * 126 * 126 = 3502800 := by norm_num
  have : 792 * 792 = 627264 := by norm_num
  rw [this]
  norm_num
  sorry

end probability_three_common_l642_642535


namespace cos_150_eq_neg_sqrt3_div_2_l642_642311

open Real

theorem cos_150_eq_neg_sqrt3_div_2 :
  cos (150 * pi / 180) = - (sqrt 3 / 2) := 
sorry

end cos_150_eq_neg_sqrt3_div_2_l642_642311


namespace cos_150_eq_neg_sqrt3_div_2_l642_642134

theorem cos_150_eq_neg_sqrt3_div_2 :
  ∃ θ : ℝ, θ = 150 ∧
           0 < θ ∧ θ < 180 ∧
           ∃ φ : ℝ, φ = 180 - θ ∧
           cos φ = Real.sqrt 3 / 2 ∧
           cos θ = - cos φ :=
  sorry

end cos_150_eq_neg_sqrt3_div_2_l642_642134


namespace tan_2theta_eq_4_over_3_l642_642394

variable (θ : ℝ)
variable (h_obtuse : π / 2 < θ ∧ θ < π)
variable (h_condition : cos (2 * θ) - sin (2 * θ) = cos (θ) ^ 2)

theorem tan_2theta_eq_4_over_3 (θ : ℝ) (h_obtuse : π / 2 < θ ∧ θ < π) (h_condition : cos (2 * θ) - sin (2 * θ) = cos (θ) ^ 2) : 
  tan (2 * θ) = 4 / 3 := 
by 
  sorry

end tan_2theta_eq_4_over_3_l642_642394


namespace tetrahedron_volume_is_20_l642_642951

noncomputable def volume_of_tetrahedron (a b c d e f : ℝ) : ℝ :=
  let B := matrix.of_fn ![
    ![0, 1, 1, 1, 1],
    ![1, 0, a^2, b^2, d^2],
    ![1, a^2, 0, c^2, e^2],
    ![1, b^2, c^2, 0, f^2],
    ![1, d^2, e^2, f^2, 0]
  ]
  in (1 / 6) * real.sqrt (-matrix.det B)

theorem tetrahedron_volume_is_20 :
  volume_of_tetrahedron 6 4 5 5 4 3 = 20 := sorry

end tetrahedron_volume_is_20_l642_642951


namespace expression_meaningful_range_l642_642833

theorem expression_meaningful_range (a : ℝ) : (∃ x, x = (a + 3) ^ (1/2) / (a - 1)) ↔ (a ≥ -3 ∧ a ≠ 1) :=
by
  sorry

end expression_meaningful_range_l642_642833


namespace cos_150_eq_neg_sqrt3_div_2_l642_642166

theorem cos_150_eq_neg_sqrt3_div_2 : Real.cos (150 * Real.pi / 180) = - (Real.sqrt 3 / 2) :=
by
  sorry

end cos_150_eq_neg_sqrt3_div_2_l642_642166


namespace cos_150_eq_neg_sqrt3_over_2_l642_642209

theorem cos_150_eq_neg_sqrt3_over_2 :
  ∀ {deg cosQ1 cosQ2: ℝ},
    cosQ1 = 30 →
    cosQ2 = 180 - cosQ1 →
    cos 150 = - (cos cosQ1) →
    cos cosQ1 = sqrt 3 / 2 →
    cos 150 = -sqrt 3 / 2 :=
by
  intros deg cosQ1 cosQ2 h1 h2 h3 h4
  sorry

end cos_150_eq_neg_sqrt3_over_2_l642_642209


namespace correct_choice_l642_642686

def statement1 (a : ℚ) : Prop := -a = -a -- True by definition of negation
def statement2 (m : ℚ) (h : m ≠ 0) : Prop := m * (1 / m) = 1 -- True for m ≠ 0
def statement3 (x : ℝ) : Prop := (|x| = x) ↔ (x ≥ 0) -- True for x ≥ 0, not just x = 0
def statement4 (x y : ℝ) : Prop := (x < y) ↔ (y > x) -- True by definition of < and >

theorem correct_choice : (statement1 ∧ statement4) ↔ True :=
by
  sorry

end correct_choice_l642_642686


namespace cosine_angle_eq_neg_one_l642_642359

noncomputable def vector_AB := (-3, 3, -3)
noncomputable def vector_AC := (6, -6, 6)

noncomputable def dot_product (u v : ℝ × ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2 + u.3 * v.3

noncomputable def magnitude (u : ℝ × ℝ × ℝ) : ℝ :=
  Real.sqrt (u.1^2 + u.2^2 + u.3^2)

noncomputable def cos_angle (u v : ℝ × ℝ × ℝ) : ℝ :=
  dot_product u v / (magnitude u * magnitude v)

theorem cosine_angle_eq_neg_one : cos_angle vector_AB vector_AC = -1 := 
  by
  sorry

end cosine_angle_eq_neg_one_l642_642359


namespace quantity_of_pure_milk_remaining_l642_642677

-- Defining initial conditions for the problem
def initial_capacity : ℝ := 50
def milk_removed_1 : ℝ := 12
def solution_removed_2 : ℝ := 8
def solution_removed_3 : ℝ := 10
def solution_removed_4 : ℝ := 6

-- Function to calculate remaining milk after each step
noncomputable def remaining_milk_after_steps (initial_capacity milk_removed_1 solution_removed_2 solution_removed_3 solution_removed_4 : ℝ) : ℝ :=
let remaining_milk_1 := initial_capacity - milk_removed_1 in
let milk_after_step_1 := remaining_milk_1 * (initial_capacity - solution_removed_2) / initial_capacity in
let remaining_milk_2 := milk_after_step_1 - (solution_removed_2 * milk_after_step_1 / initial_capacity) in
let milk_after_step_2 := remaining_milk_2 * (initial_capacity - solution_removed_3) / initial_capacity in
let remaining_milk_3 := milk_after_step_2 - (solution_removed_3 * milk_after_step_2 / initial_capacity) in
let milk_after_step_3 := remaining_milk_3 * (initial_capacity - solution_removed_4) / initial_capacity in
let remaining_milk_4 := milk_after_step_3 - (solution_removed_4 * milk_after_step_3 / initial_capacity) in
remaining_milk_4

-- Statement to be proved
theorem quantity_of_pure_milk_remaining :
  remaining_milk_after_steps initial_capacity milk_removed_1 solution_removed_2 solution_removed_3 solution_removed_4 = 22.47 :=
  by sorry

end quantity_of_pure_milk_remaining_l642_642677


namespace big_island_counties_l642_642053

-- Defining the problem conditions
def island (counties : ℕ) : Prop :=
  ∃ (g1 g2 : Fin 2), counties = 2 * g1.val + g2.val

def road_pattern (counties : ℕ) : Prop :=
  island(counties) ∧ (counties % 2 = 1)

-- Statement to be proven
theorem big_island_counties : road_pattern 9 :=
by
  sorry

end big_island_counties_l642_642053


namespace seventh_root_binomial_expansion_l642_642730

theorem seventh_root_binomial_expansion : 
  (∃ (n : ℕ), n = 137858491849 ∧ (∃ (k : ℕ), n = (10 + 1) ^ k)) →
  (∃ a, a = 11 ∧ 11 ^ 7 = 137858491849) := 
by {
  sorry 
}

end seventh_root_binomial_expansion_l642_642730


namespace cos_150_eq_neg_half_l642_642272

theorem cos_150_eq_neg_half : Real.cos (150 * Real.pi / 180) = -1 / 2 := 
by
  sorry

end cos_150_eq_neg_half_l642_642272


namespace triangle_perimeter_l642_642576

-- Definitions for the conditions
def inscribed_circle_of_triangle_tangent_at (radius : ℝ) (DP : ℝ) (PE : ℝ) : Prop :=
  radius = 27 ∧ DP = 29 ∧ PE = 33

-- Perimeter calculation theorem
theorem triangle_perimeter (r DP PE : ℝ) (h : inscribed_circle_of_triangle_tangent_at r DP PE) : 
  ∃ perimeter : ℝ, perimeter = 774 :=
by
  sorry

end triangle_perimeter_l642_642576


namespace cos_150_eq_neg_sqrt3_over_2_l642_642221

theorem cos_150_eq_neg_sqrt3_over_2 : Real.cos (150 * Real.pi / 180) = -Real.sqrt 3 / 2 := 
by 
  sorry

end cos_150_eq_neg_sqrt3_over_2_l642_642221


namespace number_of_cds_l642_642505

-- Define the constants
def total_money : ℕ := 37
def cd_price : ℕ := 14
def cassette_price : ℕ := 9

theorem number_of_cds (total_money cd_price cassette_price : ℕ) (h_total_money : total_money = 37) (h_cd_price : cd_price = 14) (h_cassette_price : cassette_price = 9) :
  ∃ n : ℕ, n * cd_price + cassette_price = total_money ∧ n = 2 :=
by {
  -- Placeholder for the actual proof
  sorry
}

end number_of_cds_l642_642505


namespace piggy_bank_after_8_weeks_l642_642875

-- Define initial amount in the piggy bank
def initial_amount : ℝ := 43

-- Define weekly allowance amount
def weekly_allowance : ℝ := 10

-- Define fraction of allowance Jack saves
def saving_fraction : ℝ := 0.5

-- Define number of weeks
def number_of_weeks : ℕ := 8

-- Define weekly savings amount
def weekly_savings : ℝ := saving_fraction * weekly_allowance

-- Define total savings after a given number of weeks
def total_savings (weeks : ℕ) : ℝ := weeks * weekly_savings

-- Define the final amount in the piggy bank after a given number of weeks
def final_amount (weeks : ℕ) : ℝ := initial_amount + total_savings weeks

-- Theorem: Prove that final amount in piggy bank after 8 weeks is $83
theorem piggy_bank_after_8_weeks : final_amount number_of_weeks = 83 := by
  sorry

end piggy_bank_after_8_weeks_l642_642875


namespace find_k_range_of_a_exists_m_l642_642414

noncomputable def f (x k : ℝ) := Real.log (4^(4 * x) + 1) / Real.log 4 + k * x

-- Part 1: Find the value of k
theorem find_k (k : ℝ) (h : f (-1) k = f 1 k) : k = -1 / 2 :=
sorry

-- Part 2: Range of a such that y=f(x) has no intersections with y=(1/2)x + a
noncomputable def g (x : ℝ) := Real.log (4^(4 * x) + 1) / Real.log 4 - x

theorem range_of_a (a : ℝ) : (∀ x : ℝ, g x ≠ a) → a ∈ Set.Iic 0 :=
sorry

-- Part 3: Finding m such that the minimum value of h(x)=0
noncomputable def h (x k m : ℝ) := 4^(f x k + (1/2) * x) + m * 2^x - 1

theorem exists_m (h_min : ∃ m : ℝ, (∀ x ∈ Set.Icc 0 (Real.log 2 3), h x (-1 / 2) m = 0)) : ∃ m = -1 :=
sorry

end find_k_range_of_a_exists_m_l642_642414


namespace cos_150_eq_neg_sqrt3_div_2_l642_642301

open Real

theorem cos_150_eq_neg_sqrt3_div_2 :
  cos (150 * pi / 180) = - (sqrt 3 / 2) := 
sorry

end cos_150_eq_neg_sqrt3_div_2_l642_642301


namespace problem1_problem2_l642_642864

-- Definitions of the conditions and theorems
noncomputable def triangle_ABC_max_area (B : ℝ) (b : ℝ) (A C : ℝ) : ℝ :=
  let α := (A - C) / 2 in
  (b * b * sin (C)) / 2

theorem problem1 (B : ℝ) (A C : ℝ) (b : ℝ) (h1 : A + C = 120) (h2 : B = 60) 
                 (h3 : b = 2) : max (triangle_ABC_max_area B b A C) = 1 := sorry

noncomputable def cos_half_angle_diff (A C : ℝ) : ℝ :=
  cos ((A - C) / 2)

theorem problem2 (A C B : ℝ) (h1 : 1 / cos A + 1 / cos C = -sqrt(2) / cos B )
                 (h2 : B = 60) : cos_half_angle_diff A C = sqrt(2) / 2 := sorry

end problem1_problem2_l642_642864


namespace triangle_area_six_parts_l642_642982

theorem triangle_area_six_parts (S S₁ S₂ S₃ : ℝ) (h₁ : S₁ ≥ 0) (h₂ : S₂ ≥ 0) (h₃ : S₃ ≥ 0) :
  S = (Real.sqrt S₁ + Real.sqrt S₂ + Real.sqrt S₃) ^ 2 := 
sorry

end triangle_area_six_parts_l642_642982


namespace original_price_of_shirts_l642_642586

theorem original_price_of_shirts 
  (final_price : ℝ) (discount1: ℝ) (discount2: ℝ) (P: ℝ)
  (h_final_price : final_price = 830)
  (h_discount1 : discount1 = 0.15)
  (h_discount2 : discount2 = 0.02)
  (h_final_eq : final_price = P * (1 - discount1) * (1 - discount2)) :
  P ≈ 996.40 :=
  sorry

end original_price_of_shirts_l642_642586


namespace cos_150_eq_neg_sqrt3_over_2_l642_642210

theorem cos_150_eq_neg_sqrt3_over_2 :
  ∀ {deg cosQ1 cosQ2: ℝ},
    cosQ1 = 30 →
    cosQ2 = 180 - cosQ1 →
    cos 150 = - (cos cosQ1) →
    cos cosQ1 = sqrt 3 / 2 →
    cos 150 = -sqrt 3 / 2 :=
by
  intros deg cosQ1 cosQ2 h1 h2 h3 h4
  sorry

end cos_150_eq_neg_sqrt3_over_2_l642_642210


namespace evaluate_expression_l642_642999

theorem evaluate_expression : 5^2 + 15 / 3 - (3 * 2)^2 = -6 := 
by
  sorry

end evaluate_expression_l642_642999


namespace framed_painting_ratio_l642_642090

theorem framed_painting_ratio 
  (painting_width painting_height frame_ratio: ℝ) 
  (frame_side_width : ℝ) 
  (frame_area_twice_painting_area: Bool) 
  (total_width total_height : ℝ) :
  painting_width = 20 →
  painting_height = 30 →
  frame_ratio = 3 →
  total_width = painting_width + 2 * frame_side_width →
  total_height = painting_height + 2 * frame_ratio * frame_side_width →
  frame_area_twice_painting_area = true →
  (total_width * total_height - painting_width * painting_height) = 2 * (painting_width * painting_height) →
  (total_width < total_height) →
  (total_width / total_height = 1 / 2) :=
begin
  intros h1 h2 h3 h4 h5 h6 h7 h8,
  sorry
end

end framed_painting_ratio_l642_642090


namespace max_value_of_f_l642_642734

def f (x : ℝ) : ℝ := 2 * (Real.sin x)^2 + Real.sin (2 * x)

theorem max_value_of_f :
  ∃ x : ℝ, f x = 1 + Real.sqrt 2 :=
sorry

end max_value_of_f_l642_642734


namespace cosine_150_eq_neg_sqrt3_div_2_l642_642244

theorem cosine_150_eq_neg_sqrt3_div_2 :
  (∀ θ : ℝ, θ = real.pi / 6) →
  (∀ θ : ℝ, cos (real.pi - θ) = -cos θ) →
  cos (5 * real.pi / 6) = -real.sqrt 3 / 2 :=
by
  intros hθ hcos_identity
  sorry

end cosine_150_eq_neg_sqrt3_div_2_l642_642244


namespace terminating_decimal_count_l642_642756

theorem terminating_decimal_count :
  (finset.filter (λ n : ℕ, 1 ≤ n ∧ n ≤ 1600 ∧ (∃ k : ℕ, 2310 * k = d * (231:ℕ) ∧ ∀ p ∈ integer.primes, p | (2310 / d) → p = 2 ∨ p = 5))
  (finset.range 1601)).card = 6 :=
sorry

end terminating_decimal_count_l642_642756


namespace ratio_Florence_Rene_l642_642531

theorem ratio_Florence_Rene :
  ∀ (I F R : ℕ), R = 300 → F = k * R → I = 1/3 * (F + R + I) → F + R + I = 1650 → F / R = 3 / 2 := 
by 
  sorry

end ratio_Florence_Rene_l642_642531


namespace TotalPriceSupplies_l642_642533

theorem TotalPriceSupplies (drum30_price drum20_price : ℕ) (drums30 drums20 : ℕ) (H1 : drum30_price = 30) (H2 : drum20_price = 20) (H3 : drums30 = 2) (H4 : drums20 = 5) :
  drums30 * drum30_price + drums20 * drum20_price = 160 :=
by
  rw H1
  rw H2
  rw H3
  rw H4
  norm_num
  sorry  -- Proof goes here, but it's skipped for this example

end TotalPriceSupplies_l642_642533


namespace how_many_lassis_l642_642121

def lassis_per_mango : ℕ := 15 / 3

def lassis15mangos : ℕ := 15

theorem how_many_lassis (H : lassis_per_mango = 5) : lassis15mangos * lassis_per_mango = 75 :=
by
  rw [H]
  sorry

end how_many_lassis_l642_642121


namespace det_B_eq_two_l642_642514

variable (p q : ℝ)
def B : Matrix (Fin 2) (Fin 2) ℝ := ![![p, 3], ![-4, q]]
noncomputable def B_inv : Matrix (Fin 2) (Fin 2) ℝ := (1 / (p * q + 12)) • ![![q, -3], ![4, p]]

theorem det_B_eq_two (h : B + 2 * B_inv = 0) : det B = 2 :=
by
  sorry

end det_B_eq_two_l642_642514


namespace smallest_value_l642_642897

theorem smallest_value 
  (x1 x2 x3 : ℝ) 
  (hx1 : 0 < x1) 
  (hx2 : 0 < x2) 
  (hx3 : 0 < x3)
  (h : 2 * x1 + 3 * x2 + 4 * x3 = 100) : 
  x1^2 + x2^2 + x3^2 = 10000 / 29 := by
  sorry

end smallest_value_l642_642897


namespace trapezoid_division_areas_l642_642976

theorem trapezoid_division_areas (a b m : ℝ) :
  let t1 := (m / 18) * (5 * a + b),
      t2 := (m / 6) * (a + b),
      t3 := (m / 18) * (a + 5 * b)
  in (∃ t1 t2 t3, t1 = (m / 18) * (5 * a + b) ∧
                   t2 = (m / 6) * (a + b) ∧
                   t3 = (m / 18) * (a + 5 * b)) :=
by {
    sorry
}

end trapezoid_division_areas_l642_642976


namespace sum_S_of_divisors_l642_642631

-- Define S(m) and necessary conditions
def S (m : ℕ) : ℤ := match m with
  | 1 => 1
  | _ => let factors := (factorsList m) -- hypothetical function to get prime factors and exponents as a list
         let exponents_sum := factors.foldl (λ sum (_, exp) => sum + exp) 0
         (-1)^(exponents_sum)

-- The main theorem to prove
theorem sum_S_of_divisors (n : ℕ) : 
  (∑ d in divisors n, S d) = if (isPerfectSquare n) then 1 else 0 := 
  sorry

end sum_S_of_divisors_l642_642631


namespace amount_left_after_spending_l642_642495

-- Definitions based on conditions
def initial_amount : ℕ := 204
def amount_spent_on_toy (initial : ℕ) : ℕ := initial / 2
def remaining_after_toy (initial : ℕ) : ℕ := initial - amount_spent_on_toy initial
def amount_spent_on_book (remaining : ℕ) : ℕ := remaining / 2
def remaining_after_book (remaining : ℕ) : ℕ := remaining - amount_spent_on_book remaining

-- Proof statement
theorem amount_left_after_spending : 
  remaining_after_book (remaining_after_toy initial_amount) = 51 :=
sorry

end amount_left_after_spending_l642_642495


namespace certain_power_l642_642830

theorem certain_power (p q : ℕ) (hp : p.prime) (hq : q.prime) (x : ℕ) (h : (4 + 1) * (x + 1) = 50) : x = 9 := 
by 
  sorry

end certain_power_l642_642830


namespace largest_perimeter_l642_642761

-- Definitions based on problem statement
def side_length : ℝ := 20
def total_area : ℝ := side_length ^ 2
def pieces : ℝ := 5
def area_per_piece : ℝ := total_area / pieces
def point_P : ℝ × ℝ := (8, 9)
def point_Q : ℝ := 7

-- Proof statement for each scenario
theorem largest_perimeter (a b c d : ℝ) (ha : a = 18 + 5 / 9) (hb : b = 15 + 4 / 9) (hc : c = 13 + 4 / 11) (hd : d = 14 + 6 / 11) (he : e = 18 + 1 / 11) :
  a >= b ∧ a >= c ∧ a >= d ∧ a >= e :=
  sorry

end largest_perimeter_l642_642761


namespace pure_imaginary_implies_x_eq_zero_l642_642439

theorem pure_imaginary_implies_x_eq_zero (x : ℝ) (h : (x^2 - x) + (x - 1) * complex.i ∈ set.range (λ y, y * complex.i)) : x = 0 :=
by {
  sorry
}

end pure_imaginary_implies_x_eq_zero_l642_642439


namespace arc_length_polar_coordinate_l642_642118

theorem arc_length_polar_coordinate :
  let rho := λ φ : ℝ, (√2) * Real.exp φ in
  ∫ φ in 0..(Real.pi / 3), (√(rho φ)^2 + (rho φ)^2) = 2 * (Real.exp (Real.pi / 3) - 1) :=
by
  sorry

end arc_length_polar_coordinate_l642_642118


namespace minimum_value_of_function_l642_642965

theorem minimum_value_of_function : 
  ∀ (x : ℝ), ∃ y : ℝ, y = -cos x ^ 2 + 2 * sin x + 2 ∧ y ≥ 0 :=
begin
  assume x,
  use (-cos x ^ 2 + 2 * sin x + 2),
  split,
  { 
    simp, -- simplifies the condition 
    sorry  -- skips the proof
  },
  { 
    sorry -- skips the proof 
  }
end

end minimum_value_of_function_l642_642965


namespace cyclist_stop_time_l642_642652

/-- The problem setup:
- The hiker's speed: 5 miles per hour
- The cyclist's speed: 20 miles per hour
- The cyclist waits for 15 minutes
You need to prove that the cyclist stopped 3.75 minutes after passing the hiker.
--/
def hiker_speed : ℝ := 5 -- miles per hour
def cyclist_speed : ℝ := 20 -- miles per hour
def wait_time : ℝ := 15 / 60 -- hours (converted from minutes)

theorem cyclist_stop_time :
  let t : ℝ := (1.25 / cyclist_speed) * 60 in
  t = 3.75 :=
by
  sorry

end cyclist_stop_time_l642_642652


namespace cos_150_eq_neg_sqrt3_div_2_l642_642255

theorem cos_150_eq_neg_sqrt3_div_2 :
  let theta := 30 * Real.pi / 180 in
  let Q := 150 * Real.pi / 180 in
  cos Q = - (Real.sqrt 3 / 2) := by
    sorry

end cos_150_eq_neg_sqrt3_div_2_l642_642255


namespace isosceles_triangles_count_l642_642922

structure Point :=
  (x : ℝ)
  (y : ℝ)

def dist (A B : Point) : ℝ :=
  real.sqrt ((B.x - A.x)^2 + (B.y - A.y)^2)

def is_isosceles (A B C : Point) : Prop :=
  (dist A B = dist A C) ∨ (dist A B = dist B C) ∨ (dist A C = dist B C)

def triangle1_isosceles : Prop :=
  is_isosceles ⟨0, 0⟩ ⟨4, 0⟩ ⟨2, 3⟩

def triangle2_isosceles : Prop :=
  is_isosceles ⟨1, 1⟩ ⟨1, 4⟩ ⟨4, 1⟩

def triangle3_isosceles : Prop :=
  is_isosceles ⟨3, 0⟩ ⟨6, 0⟩ ⟨4, 3⟩

def triangle4_isosceles : Prop :=
  is_isosceles ⟨5, 2⟩ ⟨8, 2⟩ ⟨7, 5⟩

theorem isosceles_triangles_count : 
  (Ω : Type) (triangle1_isosceles : Prop) (triangle2_isosceles : Prop) (triangle3_isosceles : Prop) (triangle4_isosceles : Prop) : 
  (ite triangle1_isosceles 1 0 + ite triangle2_isosceles 1 0 + ite triangle3_isosceles 1 0 + ite triangle4_isosceles 1 0 = 2) := sorry

end isosceles_triangles_count_l642_642922


namespace minimize_fees_at_5_l642_642656

noncomputable def minimize_costs (x : ℝ) (y1 y2 : ℝ) : Prop :=
  let k1 := 40
  let k2 := 8 / 5
  y1 = k1 / x ∧ y2 = k2 * x ∧ (∀ x, y1 + y2 ≥ 16 ∧ (y1 + y2 = 16 ↔ x = 5))

theorem minimize_fees_at_5 :
  minimize_costs 5 4 16 :=
sorry

end minimize_fees_at_5_l642_642656


namespace sequence_a_n_l642_642862

-- Definition of the sequence S_n
def S (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  if h : n > 0 then 4 - a n - 1 / (2 ^ (n - 2)) 
  else 0

-- Conjecture that we need to prove: a_n = n / 2^(n-1) 
theorem sequence_a_n (a : ℕ → ℝ) (n : ℕ) 
  (h_S : ∀ n > 0, S a n = 4 - a n - 1 / (2 ^ (n - 2))):
  a n = n / (2 ^ (n - 1)) :=
sorry

end sequence_a_n_l642_642862


namespace find_b_l642_642924

theorem find_b (a b c d : ℝ)
  (h1 : f (-1) = 0)
  (h2 : f 1 = 0)
  (h3 : f 0 = 2)
  (f : ℝ → ℝ := λ x, a * x^3 + b * x^2 + c * x + d)
  : b = -2 := 
by
  have eq1 := h1
  have eq2 := h2
  have eq3 := h3
  -- further steps would go here to show that b = -2
  sorry

end find_b_l642_642924


namespace cos_150_eq_neg_sqrt3_div_2_l642_642158

theorem cos_150_eq_neg_sqrt3_div_2 : Real.cos (150 * Real.pi / 180) = - (Real.sqrt 3 / 2) :=
by
  sorry

end cos_150_eq_neg_sqrt3_div_2_l642_642158


namespace cos_150_deg_l642_642318

-- Define the conditions as variables
def angle1 := 150
def angle2 := 180
def angle3 := 30
def cos30 := Real.sqrt 3 / 2

-- The main statement to be proved
theorem cos_150_deg : Real.cos (150 * Real.pi / 180) = - (Real.sqrt 3 / 2) :=
by
  sorry

end cos_150_deg_l642_642318


namespace problem1_correct_problem2_correct_problem3_correct_problem4_correct_l642_642696

noncomputable def problem1 : ℤ :=
12 - (-18) + (-11) - 15

theorem problem1_correct : problem1 = 4 := by
  sorry

noncomputable def problem2 : ℚ :=
-(2^2) ÷ (4/9) * (-((2/3)^2))

theorem problem2_correct : problem2 = -4 := by
  sorry

noncomputable def problem3 : ℚ :=
(-3) * (-7/5) + (-1 - (3/7))

theorem problem3_correct : problem3 = 97/35 := by
  sorry

noncomputable def problem4 : ℚ :=
(2 + 1/3) - (2 + 3/5) + (5 + 2/3) - (4 + 2/5)

theorem problem4_correct : problem4 = 1 := by
  sorry

end problem1_correct_problem2_correct_problem3_correct_problem4_correct_l642_642696


namespace cos_150_eq_neg_sqrt3_over_2_l642_642220

theorem cos_150_eq_neg_sqrt3_over_2 : Real.cos (150 * Real.pi / 180) = -Real.sqrt 3 / 2 := 
by 
  sorry

end cos_150_eq_neg_sqrt3_over_2_l642_642220


namespace abs_ab_eq_2128_l642_642946

theorem abs_ab_eq_2128 (a b : ℤ) (h1 : a ≠ 0) (h2 : b ≠ 0)
  (h3 : ∃ r s : ℤ, r ≠ s ∧ ∃ r' : ℤ, r' = r ∧ 
          (x^3 + a * x^2 + b * x + 16 * a = (x - r)^2 * (x - s) ∧ r * r * s = -16 * a)) :
  |a * b| = 2128 :=
sorry

end abs_ab_eq_2128_l642_642946


namespace cos_150_eq_neg_sqrt3_div_2_l642_642298

theorem cos_150_eq_neg_sqrt3_div_2 :
  (cos (150 * real.pi / 180)) = - (sqrt 3 / 2) := sorry

end cos_150_eq_neg_sqrt3_div_2_l642_642298


namespace cos_150_eq_neg_sqrt3_over_2_l642_642223

theorem cos_150_eq_neg_sqrt3_over_2 : Real.cos (150 * Real.pi / 180) = -Real.sqrt 3 / 2 := 
by 
  sorry

end cos_150_eq_neg_sqrt3_over_2_l642_642223


namespace rhombus_overlap_area_l642_642607

theorem rhombus_overlap_area (beta : ℝ) (hbeta : beta ≠ 0) :
    let width := 2 in
    let diagonal1 := width in
    let diagonal2 := width / sin beta in
    (1/2) * diagonal1 * diagonal2 = 2 / (sin beta) :=
by
  intros
  let width := 2
  let diagonal1 := width
  let diagonal2 := width / sin beta
  calc
    (1/2) * diagonal1 * diagonal2
        = (1/2) * 2 * (2 / sin beta) : sorry  -- replace 'sorry' with the actual calculation
    ... = 2 / (sin beta) : sorry  -- replace 'sorry' with the actual calculation

end rhombus_overlap_area_l642_642607


namespace perimeter_of_hexagon_l642_642010

-- Define the points and distances
variables (X Y Z W V T : Type)
variables (d : ℝ)
variables [equilateral_triangle : ∀ (A B C : Type), (∀ (AB BC CA : ℝ), AB = BC ∧ BC = CA) → Prop]

-- The known conditions
axiom XY_eq_6 : d = 6
axiom midpoints : ∀ (U V : Type), (d/2 * 2 = d)

-- The condition that the triangles are equilateral
axiom XYZ_equilateral : equilateral_triangle X Y Z (λ AB BC CA, AB = BC ∧ BC = CA)
axiom XWV_equilateral : equilateral_triangle X W V (λ AB BC CA, AB = BC ∧ BC = CA)
axiom VUT_equilateral : equilateral_triangle V U T (λ AB BC CA, AB = BC ∧ BC = CA)

theorem perimeter_of_hexagon : d = 30 :=
by 
  -- We will skip the proof here
  sorry

end perimeter_of_hexagon_l642_642010


namespace cos_150_degree_l642_642179

theorem cos_150_degree : Real.cos (150 * Real.pi / 180) = -Real.sqrt 3 / 2 :=
by
  sorry

end cos_150_degree_l642_642179


namespace cos_150_eq_neg_half_l642_642269

theorem cos_150_eq_neg_half : Real.cos (150 * Real.pi / 180) = -1 / 2 := 
by
  sorry

end cos_150_eq_neg_half_l642_642269


namespace determine_coefficients_l642_642733

noncomputable def quadratic_polynomial (a b c x : ℝ) : ℝ :=
  a * x^2 + b * x + c

def is_quad_max (f : ℝ → ℝ) (k x₀ : ℝ) : Prop :=
  ∀ x, f x ≤ k

def sum_cubes_eq (f : ℝ → ℝ) (sum_cubes : ℝ) : Prop :=
  ∑ (root : ℝ) in finset.univ.filter (λ x, f x = 0), root^3 = sum_cubes

theorem determine_coefficients :
  ∃ a b c : ℝ, quadratic_polynomial a b c = (λ x, a * x^2 + b * x + c) ∧
  is_quad_max (quadratic_polynomial a b c) 25 (1 / 2) ∧
  sum_cubes_eq (quadratic_polynomial a b c) 19 ∧
  a = -4 ∧ b = 4 ∧ c = 24 :=
begin
  sorry
end

end determine_coefficients_l642_642733


namespace sum_first_9_terms_l642_642469

noncomputable theory

def a5 : ℝ := ∫ x in -1..1, (x + 1)

def S (n : ℕ) (a₁ : ℝ) (a₂ : ℝ) : ℝ :=
  n * (a₁ + a₂) / 2

theorem sum_first_9_terms (a_1 a_9 : ℝ) (h : a_1 + a_9 = 2 * a5) :
  S 9 a_1 a_9 = 18 :=
by
  sorry

end sum_first_9_terms_l642_642469


namespace pizzas_ordered_l642_642011

theorem pizzas_ordered (n s p : ℕ) (hn : n = 12) (hs : s = 8) (hp : p = 2) : n * p / s = 3 := by
  rw [hn, hs, hp]
  norm_num

end pizzas_ordered_l642_642011


namespace vectors_parallel_iff_abs_x_eq_two_l642_642814

theorem vectors_parallel_iff_abs_x_eq_two (x: ℝ) : 
  ((1, x) : ℝ × ℝ) ∥ (x^2, 4 * x) ↔ |x| = 2 := sorry

end vectors_parallel_iff_abs_x_eq_two_l642_642814


namespace _l642_642853

open Classical

variables {E F G H J K U V L : Type} 
variable [MetricSpace E] [MetricSpace F] [MetricSpace G] [MetricSpace H] [MetricSpace J] [MetricSpace K]
variable [MetricSpace U] [MetricSpace V] [MetricSpace L]
noncomputable def is_right_angle (a b c : E) : Prop :=
  ∃ d : ℝ, dist a b * dist b c = d * d

noncomputable theorem rectangle_and_right_triangles (E F G H J K U V L : E)
  (h1 : ∃ a b, ∠ E F G = a ∧ ∠ J E H = b ∧ b = 90)
  (h2 : dist E J = 15)
  (h3 : dist E K = 20)
  (h4 : dist J K = 25)
  (h5 : UV ⊥ FG)
  (h6 : U = UJ ∧ FU = UJ)
  (h7 : JH ∩ UV = K)
  : dist F J = 12 ∧ dist U K = Real.sqrt 385 := sorry

end _l642_642853


namespace cos_150_deg_l642_642320

-- Define the conditions as variables
def angle1 := 150
def angle2 := 180
def angle3 := 30
def cos30 := Real.sqrt 3 / 2

-- The main statement to be proved
theorem cos_150_deg : Real.cos (150 * Real.pi / 180) = - (Real.sqrt 3 / 2) :=
by
  sorry

end cos_150_deg_l642_642320


namespace paper_boat_travel_time_l642_642481

-- Defining the conditions as constants
def distance_embankment : ℝ := 50
def speed_downstream : ℝ := 10
def speed_upstream : ℝ := 12.5

-- Definitions for the speeds of the boat and current
noncomputable def v_boat : ℝ := (speed_upstream + speed_downstream) / 2
noncomputable def v_current : ℝ := (speed_downstream - speed_upstream) / 2

-- Statement to prove the time taken for the paper boat
theorem paper_boat_travel_time :
  (distance_embankment / v_current) = 40 := by
  sorry

end paper_boat_travel_time_l642_642481


namespace cos_150_eq_neg_sqrt3_div_2_l642_642290

theorem cos_150_eq_neg_sqrt3_div_2 :
  (cos (150 * real.pi / 180)) = - (sqrt 3 / 2) := sorry

end cos_150_eq_neg_sqrt3_div_2_l642_642290


namespace compare_logs_l642_642058

noncomputable def a : ℝ := Real.logd 2 3.6
noncomputable def b : ℝ := Real.logd 4 3.2
noncomputable def c : ℝ := Real.logd 4 3.6

theorem compare_logs : a > c ∧ c > b := by
  sorry

end compare_logs_l642_642058


namespace find_ABVG_l642_642956

noncomputable theory

def distinct_digits (A B V G : ℕ) : Prop :=
  A ≠ B ∧ A ≠ V ∧ A ≠ G ∧ B ≠ V ∧ B ≠ G ∧ V ≠ G

def valid_digits (A B V G : ℕ) : Prop :=
  A > 0 ∧ A < 10 ∧ B > 0 ∧ B < 10 ∧ V < 10 ∧ G < 10

def quadratic_has_real_roots (a b c : ℕ) : Prop :=
  (b * b) - 4 * a * c ≥ 0

theorem find_ABVG (A B V G : ℕ) :
  distinct_digits A B V G →
  valid_digits A B V G →
  quadratic_has_real_roots A B (10 * V + G) →
  quadratic_has_real_roots A (B * V) G →
  quadratic_has_real_roots A B G →
  A = 1 ∧ B = 9 ∧ V = 2 ∧ G = 0 :=
begin
  intros h_distinct h_valid h_q1 h_q2 h_q3,
  sorry
end

end find_ABVG_l642_642956


namespace closest_to_P1991_l642_642441

def T (n : ℕ) : ℕ := (n * (n + 1)) / 2

noncomputable def P (n : ℕ) : ℚ := ∏ i in (finset.range (n + 1)).filter (λ x, x ≥ 2), (T i : ℚ) / (T i - 1)

theorem closest_to_P1991 : abs (P 1991 - 2.9) < 0.1 :=
by
  sorry

end closest_to_P1991_l642_642441


namespace paper_boat_travel_time_l642_642489

theorem paper_boat_travel_time :
  ∀ (length_of_embankment : ℝ) (length_of_motorboat : ℝ)
    (time_downstream : ℝ) (time_upstream : ℝ) (v_boat : ℝ) (v_current : ℝ),
  length_of_embankment = 50 →
  length_of_motorboat = 10 →
  time_downstream = 5 →
  time_upstream = 4 →
  v_boat + v_current = length_of_embankment / time_downstream →
  v_boat - v_current = length_of_embankment / time_upstream →
  let speed_paper_boat := v_current in
  let travel_time := length_of_embankment / speed_paper_boat in
  travel_time = 40 :=
by
  intros length_of_embankment length_of_motorboat time_downstream time_upstream v_boat v_current
  intros h_length_emb h_length_motor t_down t_up h_v_boat_plus_current h_v_boat_minus_current
  let speed_paper_boat := v_current
  let travel_time := length_of_embankment / speed_paper_boat
  sorry

end paper_boat_travel_time_l642_642489


namespace minimum_value_set_monotonic_interval_l642_642819

noncomputable def a (x : ℝ) : ℝ × ℝ := (sqrt 3 * Real.cos x, 0)
noncomputable def b (x : ℝ) : ℝ × ℝ := (0, Real.sin x)
noncomputable def f (x : ℝ) : ℝ :=
  let ab := (a x).1 + (b x).1, (a x).2 + (b x).2 in
  ab.1 ^ 2 + ab.2 ^ 2

omit

-- (I) Prove that the minimum value of f(x) is 0 and find the set of x values when the minimum occurs.
theorem minimum_value_set (x : ℝ) (k : ℤ) :
  0 ≤ f x ∧ (f x = 0 ↔ ∃ k : ℤ, x = -Real.pi / 3 + k * Real.pi) :=
sorry

-- (II) Prove the interval where the function f(x) is monotonically increasing.
theorem monotonic_interval (k : ℤ) :
  ∀ x ∈ Set.Icc (-Real.pi / 3 + k * Real.pi) (Real.pi / 6 + k * Real.pi), 
  ∀ y ∈ Set.Icc (-Real.pi / 3 + k * Real.pi) (Real.pi / 6 + k * Real.pi), x < y → f x ≤ f y :=
sorry

end minimum_value_set_monotonic_interval_l642_642819


namespace extracurricular_books_l642_642971

theorem extracurricular_books (a b c d : ℕ) 
  (h1 : b + c + d = 110)
  (h2 : a + c + d = 108)
  (h3 : a + b + d = 104)
  (h4 : a + b + c = 119) :
  a = 37 ∧ b = 39 ∧ c = 43 ∧ d = 28 :=
by {
  -- Proof to be done here
  sorry
}

end extracurricular_books_l642_642971


namespace total_houses_l642_642846

theorem total_houses (d c dc : ℕ) (h1 : d = 40) (h2 : c = 30) (h3 : dc = 10) : d + c - dc = 60 := by
  rw [h1, h2, h3]
  simp
  sorry

end total_houses_l642_642846


namespace cos_150_eq_neg_sqrt3_div_2_l642_642123

theorem cos_150_eq_neg_sqrt3_div_2 :
  ∃ θ : ℝ, θ = 150 ∧
           0 < θ ∧ θ < 180 ∧
           ∃ φ : ℝ, φ = 180 - θ ∧
           cos φ = Real.sqrt 3 / 2 ∧
           cos θ = - cos φ :=
  sorry

end cos_150_eq_neg_sqrt3_div_2_l642_642123


namespace problem_statement_l642_642766

theorem problem_statement (a : ℝ) (h : a^2 + a⁻¹^2 = 3) : a + a⁻¹ = sqrt 5 ∨ a + a⁻¹ = -sqrt 5 :=
by
  sorry

end problem_statement_l642_642766


namespace mean_minus_median_l642_642375

-- Define days missed and number of students for each day
def days_missed_counts : List (Nat × Nat) := [(0, 4), (1, 2), (2, 5), (3, 2), (4, 3), (5, 4)]

-- Function to calculate mean from the list
def mean (l : List (Nat × Nat)) : Rat := 
  l.foldl (fun acc (d : Nat × Nat) => acc + (d.1 * d.2)) 0 / l.foldl (fun acc (d : Nat × Nat) => acc + d.2) 0

-- Function to calculate median from the list
def median (l : List (Nat × Nat)) : Rat := 
  let sorted_list := l.foldl (fun acc (d : Nat × Nat) => acc ++ List.replicate d.2 d.1) []
  let n := sorted_list.length
  if n % 2 = 0 then (sorted_list.get (n / 2 - 1) + sorted_list.get (n / 2)) / 2
  else sorted_list.get (n / 2)

noncomputable def mean_days_missed : Rat := mean days_missed_counts
noncomputable def median_days_missed : Rat := median days_missed_counts

theorem mean_minus_median : 
  mean_days_missed - median_days_missed = (7 / 10 : Rat) :=
by
  sorry

end mean_minus_median_l642_642375


namespace correct_statement_is_D_l642_642107

-- Define statements as propositions
def statement_A : Prop := ∀ r, Reasonable r → Correct r
def statement_B : Prop := ∀ r, Reasonable r → Inductive r
def statement_C : Prop := ∀ r, Inductive r → General_to_Specific r
def statement_D : Prop := ∀ r, Analogical r → Specific_to_Specific r

-- Define the conditions for Reasonable, Correct, Inductive, Analogical, General_to_Specific, and Specific_to_Specific.
axiom Reasonable (r : Reasoning) : Prop
axiom Correct (r : Reasoning) : Prop
axiom Inductive (r : Reasoning) : Prop
axiom Analogical (r : Reasoning) : Prop
axiom General_to_Specific (r : Reasoning) : Prop
axiom Specific_to_Specific (r : Reasoning) : Prop

-- Prove that the correct statement is D
theorem correct_statement_is_D : statement_D :=
by
 sorry

end correct_statement_is_D_l642_642107


namespace farmer_hay_bales_l642_642085

noncomputable def hay_bales_left : ℤ :=
  let bales_per_acre := 560 / 5 in
  let total_acres := 5 + 7 in
  let total_bales := (bales_per_acre * total_acres : ℤ) in
  let horses := 9 in
  let bales_per_horse_per_day := 3 in
  let days_per_month := 30 in
  let months := 4 in
  let daily_consumption := (bales_per_horse_per_day * horses : ℤ) in
  let monthly_consumption := daily_consumption * days_per_month in
  let total_consumption := monthly_consumption * months in
  total_bales - total_consumption

theorem farmer_hay_bales : hay_bales_left = -1896 := by
  sorry

end farmer_hay_bales_l642_642085


namespace cos_150_eq_neg_sqrt3_div_2_l642_642264

theorem cos_150_eq_neg_sqrt3_div_2 :
  let theta := 30 * Real.pi / 180 in
  let Q := 150 * Real.pi / 180 in
  cos Q = - (Real.sqrt 3 / 2) := by
    sorry

end cos_150_eq_neg_sqrt3_div_2_l642_642264


namespace smallest_integer_inequality_l642_642750

theorem smallest_integer_inequality :
  (∃ n : ℤ, ∀ x y z : ℝ, (x + y + z)^2 ≤ (n:ℝ) * (x^2 + y^2 + z^2)) ∧
  ∀ m : ℤ, (∀ x y z : ℝ, (x + y + z)^2 ≤ (m:ℝ) * (x^2 + y^2 + z^2)) → 3 ≤ m :=
  sorry

end smallest_integer_inequality_l642_642750


namespace cos_150_degree_l642_642181

theorem cos_150_degree : Real.cos (150 * Real.pi / 180) = -Real.sqrt 3 / 2 :=
by
  sorry

end cos_150_degree_l642_642181


namespace area_of_triangle_MNF_is_4sqrt3_l642_642655

-- Let p be a positive real number
variables (p : ℝ) (hp : 0 < p)

-- Define the parabola C
def parabola (x y : ℝ) : Prop := y^2 = 2 * p * x

-- Define the line with slope sqrt(3) passing through the focus of the parabola
def line_through_focus (F : ℝ × ℝ) (M : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, F.snd = k * F.fst ∧ M.snd = k * M.fst ∧ k = real.sqrt 3

-- Define the directrix of the parabola
def directrix (x : ℝ) : Prop := x = -p / 2

-- Define the condition MN ⊥ l
def perpendicular (M N : ℝ × ℝ) : Prop := M.fst = N.fst

-- Define the perimeter condition |MN + NF + FM| = 12
def perimeter_condition (M N F : ℝ × ℝ) : Prop :=
  real.dist M N + real.dist N F + real.dist F M = 12

-- Define the area of the triangle MNF
def area_MNF (M N F : ℝ × ℝ) : ℝ :=
  let a := real.dist M N in
  let b := real.dist N F in
  let c := real.dist F M in
  real.sqrt (s * (s - a) * (s - b) * (s - c))
  where s = (a + b + c) / 2

-- Prove that the area of the triangle MNF is 4√3
theorem area_of_triangle_MNF_is_4sqrt3
  (F M N : ℝ × ℝ)
  (hF : parabola F.fst F.snd)
  (hM : parabola M.fst M.snd)
  (hline : line_through_focus F M)
  (hdirectrix : directrix N.fst)
  (hperpendicular : perpendicular M N)
  (hperimeter : perimeter_condition M N F) :
  area_MNF M N F = 4 * real.sqrt 3 :=
sorry

end area_of_triangle_MNF_is_4sqrt3_l642_642655


namespace cos_150_eq_neg_sqrt3_div_2_l642_642303

open Real

theorem cos_150_eq_neg_sqrt3_div_2 :
  cos (150 * pi / 180) = - (sqrt 3 / 2) := 
sorry

end cos_150_eq_neg_sqrt3_div_2_l642_642303


namespace moles_of_CaCl2_l642_642362

theorem moles_of_CaCl2 (HCl moles_of_HCl : ℕ) (CaCO3 moles_of_CaCO3 : ℕ) 
  (reaction : (CaCO3 = 1) → (HCl = 2) → (moles_of_HCl = 6) → (moles_of_CaCO3 = 3)) :
  ∃ moles_of_CaCl2 : ℕ, moles_of_CaCl2 = 3 :=
by
  sorry

end moles_of_CaCl2_l642_642362


namespace piggy_bank_after_8_weeks_l642_642874

-- Define initial amount in the piggy bank
def initial_amount : ℝ := 43

-- Define weekly allowance amount
def weekly_allowance : ℝ := 10

-- Define fraction of allowance Jack saves
def saving_fraction : ℝ := 0.5

-- Define number of weeks
def number_of_weeks : ℕ := 8

-- Define weekly savings amount
def weekly_savings : ℝ := saving_fraction * weekly_allowance

-- Define total savings after a given number of weeks
def total_savings (weeks : ℕ) : ℝ := weeks * weekly_savings

-- Define the final amount in the piggy bank after a given number of weeks
def final_amount (weeks : ℕ) : ℝ := initial_amount + total_savings weeks

-- Theorem: Prove that final amount in piggy bank after 8 weeks is $83
theorem piggy_bank_after_8_weeks : final_amount number_of_weeks = 83 := by
  sorry

end piggy_bank_after_8_weeks_l642_642874


namespace cos_150_eq_neg_sqrt3_over_2_l642_642215

theorem cos_150_eq_neg_sqrt3_over_2 :
  ∀ {deg cosQ1 cosQ2: ℝ},
    cosQ1 = 30 →
    cosQ2 = 180 - cosQ1 →
    cos 150 = - (cos cosQ1) →
    cos cosQ1 = sqrt 3 / 2 →
    cos 150 = -sqrt 3 / 2 :=
by
  intros deg cosQ1 cosQ2 h1 h2 h3 h4
  sorry

end cos_150_eq_neg_sqrt3_over_2_l642_642215


namespace cos_150_eq_negative_cos_30_l642_642142

theorem cos_150_eq_negative_cos_30 :
  Real.cos (150 * Real.pi / 180) = - (Real.cos (30 * Real.pi / 180)) :=
by sorry

end cos_150_eq_negative_cos_30_l642_642142


namespace cosine_150_eq_neg_sqrt3_div_2_l642_642250

theorem cosine_150_eq_neg_sqrt3_div_2 :
  (∀ θ : ℝ, θ = real.pi / 6) →
  (∀ θ : ℝ, cos (real.pi - θ) = -cos θ) →
  cos (5 * real.pi / 6) = -real.sqrt 3 / 2 :=
by
  intros hθ hcos_identity
  sorry

end cosine_150_eq_neg_sqrt3_div_2_l642_642250


namespace max_annual_profit_l642_642597

theorem max_annual_profit :
  ∃ x : ℝ, 0 ≤ x ∧ x ≤ 5 ∧ 
  let S := x / 5 in
  let T := (2 * Real.sqrt (5 - x)) / 5 in
  S + T = 1.2 :=
sorry

end max_annual_profit_l642_642597


namespace problem_statement_l642_642960

variable (f : ℝ → ℝ)

-- Assuming that f is differentiable on ℝ and 2f(x) - f'(x) > 0 holds.
axiom differentiable_on_R : ∀ x, differentiable_at ℝ f x
axiom condition : ∀ x, 2 * f x - deriv f x > 0

-- Proof of the statement f(1) > f(2) / e^2
theorem problem_statement : f(1) > f(2) / Real.exp 2 := by
  sorry

end problem_statement_l642_642960


namespace cos_150_deg_l642_642317

-- Define the conditions as variables
def angle1 := 150
def angle2 := 180
def angle3 := 30
def cos30 := Real.sqrt 3 / 2

-- The main statement to be proved
theorem cos_150_deg : Real.cos (150 * Real.pi / 180) = - (Real.sqrt 3 / 2) :=
by
  sorry

end cos_150_deg_l642_642317


namespace total_visitors_count_l642_642860

def initial_morning_visitors : ℕ := 500
def noon_departures : ℕ := 119
def additional_afternoon_arrivals : ℕ := 138

def afternoon_arrivals : ℕ := noon_departures + additional_afternoon_arrivals
def total_visitors : ℕ := initial_morning_visitors + afternoon_arrivals

theorem total_visitors_count : total_visitors = 757 := 
by sorry

end total_visitors_count_l642_642860


namespace trajectory_envelope_l642_642630

/-- A material point is thrown from the origin with an initial velocity v₀ in a fixed vertical plane.
    The position of the point at time t is given by x(t) = v₀ * cos(α) * t and y(t) = v₀ * sin(α) * t - (g * t^2) / 2.
    We need to show that the envelope of the trajectories y = (v₀^2) / (2g) - (g / (2v₀^2)) * x^2.
 -/
theorem trajectory_envelope (v₀ g t : ℝ) (α : ℝ) (hα : α = Float.pi / 2) :
  let x := v₀ * cos α * t,
      y := v₀ * sin α * t - (g * t^2) / 2
  in y = (v₀^2) / (2g) - (g / (2 * v₀^2)) * x^2 := sorry

end trajectory_envelope_l642_642630


namespace expression_equivalent_l642_642562

theorem expression_equivalent (m n : ℤ) (R S : ℕ) (hR : R = 16^m) (hS : S = 9^n) :
  36^(m+n) = R^n * S^m :=
by
  sorry

end expression_equivalent_l642_642562


namespace range_b_x2_bx_b_positive_l642_642755

theorem range_b_x2_bx_b_positive :
  (∀ x : ℝ, x^2 + b * x + b > 0) → (0 < b ∧ b < 4) :=
begin
  intro h,
  sorry
end

end range_b_x2_bx_b_positive_l642_642755


namespace cos_150_eq_neg_sqrt3_over_2_l642_642211

theorem cos_150_eq_neg_sqrt3_over_2 :
  ∀ {deg cosQ1 cosQ2: ℝ},
    cosQ1 = 30 →
    cosQ2 = 180 - cosQ1 →
    cos 150 = - (cos cosQ1) →
    cos cosQ1 = sqrt 3 / 2 →
    cos 150 = -sqrt 3 / 2 :=
by
  intros deg cosQ1 cosQ2 h1 h2 h3 h4
  sorry

end cos_150_eq_neg_sqrt3_over_2_l642_642211


namespace cos_150_eq_neg_sqrt3_div_2_l642_642292

theorem cos_150_eq_neg_sqrt3_div_2 :
  (cos (150 * real.pi / 180)) = - (sqrt 3 / 2) := sorry

end cos_150_eq_neg_sqrt3_div_2_l642_642292


namespace right_triangle_AC_l642_642466

theorem right_triangle_AC (A B C : Type) [A B C] (angle_B : ∠B = 90°)
  (tan_A : tan(A) = 3/4) (AB : real := 6) : AC = 7.5 :=
by
  sorry 

end right_triangle_AC_l642_642466


namespace vaccination_target_failure_l642_642850

noncomputable def percentage_vaccination_target_failed (original_target : ℕ) (first_year : ℕ) (second_year_increase_rate : ℚ) (third_year : ℕ) : ℚ :=
  let second_year := first_year + second_year_increase_rate * first_year
  let total_vaccinated := first_year + second_year + third_year
  let shortfall := original_target - total_vaccinated
  (shortfall / original_target) * 100

theorem vaccination_target_failure :
  percentage_vaccination_target_failed 720 60 (65/100 : ℚ) 150 = 57.11 := 
  by sorry

end vaccination_target_failure_l642_642850


namespace find_point_on_curve_l642_642341

theorem find_point_on_curve :
  ∃ P : ℝ × ℝ, (P.1^3 - P.1 + 3 = P.2) ∧ (3 * P.1^2 - 1 = 2) ∧ (P = (1, 3) ∨ P = (-1, 3)) :=
sorry

end find_point_on_curve_l642_642341


namespace kindergarteners_line_up_probability_l642_642095

theorem kindergarteners_line_up_probability :
  let total_line_up := Nat.choose 20 9
  let first_scenario := Nat.choose 14 9
  let second_scenario_single := Nat.choose 13 8
  let second_scenario := 6 * second_scenario_single
  let valid_arrangements := first_scenario + second_scenario
  valid_arrangements / total_line_up = 9724 / 167960 := by
  sorry

end kindergarteners_line_up_probability_l642_642095


namespace cos_150_degree_l642_642182

theorem cos_150_degree : Real.cos (150 * Real.pi / 180) = -Real.sqrt 3 / 2 :=
by
  sorry

end cos_150_degree_l642_642182


namespace max_b_satisfies_conditions_l642_642654

noncomputable def maximum_b : ℚ := 17 / 50

theorem max_b_satisfies_conditions :
  ∀ (m : ℚ), (1 / 3 < m → m < maximum_b) →
  ∀ (x y : ℤ), (0 < x ∧ x ≤ 50 ∧ y = m * x + 3) → ¬(x, y).is_lattice_point :=
  sorry

def lattice_point (x y : ℤ) := (x, y)

extension (x y : ℤ) : lattice_point (x, y) :=
{x = int, y = int, }

end max_b_satisfies_conditions_l642_642654


namespace cosine_150_eq_neg_sqrt3_div_2_l642_642235

theorem cosine_150_eq_neg_sqrt3_div_2 :
  (∀ θ : ℝ, θ = real.pi / 6) →
  (∀ θ : ℝ, cos (real.pi - θ) = -cos θ) →
  cos (5 * real.pi / 6) = -real.sqrt 3 / 2 :=
by
  intros hθ hcos_identity
  sorry

end cosine_150_eq_neg_sqrt3_div_2_l642_642235


namespace cos_150_eq_neg_sqrt3_div_2_l642_642288

theorem cos_150_eq_neg_sqrt3_div_2 :
  (cos (150 * real.pi / 180)) = - (sqrt 3 / 2) := sorry

end cos_150_eq_neg_sqrt3_div_2_l642_642288


namespace cos_150_eq_negative_cos_30_l642_642148

theorem cos_150_eq_negative_cos_30 :
  Real.cos (150 * Real.pi / 180) = - (Real.cos (30 * Real.pi / 180)) :=
by sorry

end cos_150_eq_negative_cos_30_l642_642148


namespace cos_150_eq_neg_sqrt3_div_2_l642_642286

theorem cos_150_eq_neg_sqrt3_div_2 :
  (cos (150 * real.pi / 180)) = - (sqrt 3 / 2) := sorry

end cos_150_eq_neg_sqrt3_div_2_l642_642286


namespace smallest_palindrome_addition_l642_642997

def is_palindrome (n : ℕ) : Bool :=
  let s := n.toString
  s = s.reverse

theorem smallest_palindrome_addition (n addend : ℕ) (h₁ : n = 40317) (h₂ : addend := 40404) :
  is_palindrome addend = true ∧ addend > n ∧ ∀ m, is_palindrome (n + m) → m ≥ 87 := 
by
  sorry

end smallest_palindrome_addition_l642_642997


namespace perimeter_of_triangle_KLM_l642_642099

noncomputable def 𝐷ₘ𝒾𝑫₀ₙₑ𝟸 : ℝ :=
  41 * (sqrt(105) / 55 + sqrt(3) / 10 + 3 / 22)

-- Given conditions
variable (A B C D K L M : ℝ)
variable (AD BC BD AB CD : ℝ)
variable (O : Point) (p₁ p₂ p₃ : Plane)

axiom AD_eq_10 : AD = 10
axiom BC_BDratio : BC / BD = 3 / 2
axiom AB_CDratio : AB / CD = 4 * sqrt(3) / 11
axiom midpoint_projections :
  midpoint_projection(O p₁ p₂ p₃) ->
  midpoint_projection(O AB BC CD)
axiom distance_midpoints_AB_CD_eq_13 :
  distance(midpoint(AB) midpoint(CD)) = 13

theorem perimeter_of_triangle_KLM :
  perimeter(K L M) = 𝐷ₘ𝒾𝑫₀ₙₑ𝟸 :=
  sorry

end perimeter_of_triangle_KLM_l642_642099


namespace cos_150_eq_neg_sqrt3_div_2_l642_642125

theorem cos_150_eq_neg_sqrt3_div_2 :
  ∃ θ : ℝ, θ = 150 ∧
           0 < θ ∧ θ < 180 ∧
           ∃ φ : ℝ, φ = 180 - θ ∧
           cos φ = Real.sqrt 3 / 2 ∧
           cos θ = - cos φ :=
  sorry

end cos_150_eq_neg_sqrt3_div_2_l642_642125


namespace sufficient_but_not_necessary_l642_642633

theorem sufficient_but_not_necessary (x : ℝ) :
  (x > 1 → x > 0) ∧ ¬ (x > 0 → x > 1) :=
by
  sorry

end sufficient_but_not_necessary_l642_642633


namespace product_of_two_numbers_l642_642952

theorem product_of_two_numbers 
  (LCM HCF : ℕ) 
  (h_lcm : LCM = 750) 
  (h_hcf : HCF = 25) :
  LCM * HCF = 18750 :=
by
  -- Definitions and assumptions from conditions
  have h1 : LCM = 750 := h_lcm,
  have h2 : HCF = 25 := h_hcf,
  -- Given the LCM and HCF
  calc 
    LCM * HCF = 750 * 25     : by rw [h1, h2] -- Substitute LCM and HCF
          ... = 18750         : by norm_num -- Calculate product

end product_of_two_numbers_l642_642952


namespace green_green_pairs_count_l642_642116

-- Given Conditions
def blue_students := 65
def green_students := 95
def total_students := 160
def total_pairs := 80
def blue_blue_pairs := 25

-- Theorem to prove the number of green-green pairs 
theorem green_green_pairs_count : 
  let blue_students_in_blue_blue_pairs := blue_blue_pairs * 2 in
  let mixed_pairs_blue_students := blue_students - blue_students_in_blue_blue_pairs in
  let mixed_pairs_green_students := mixed_pairs_blue_students in
  let green_students_in_green_green_pairs := green_students - mixed_pairs_green_students in
  green_students_in_green_green_pairs / 2 = 40 :=
by
  sorry

end green_green_pairs_count_l642_642116


namespace tangents_form_rectangle_l642_642812

-- Define the first ellipse
def ellipse1 (a b x y : ℝ) : Prop := x^2 / a^4 + y^2 / b^4 = 1

-- Define the second ellipse
def ellipse2 (a b x y : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Define conjugate diameters through lines
def conjugate_diameters (a b m : ℝ) : Prop := True -- (You might want to further define what conjugate diameters imply here)

-- Prove the main statement
theorem tangents_form_rectangle
  (a b m : ℝ)
  (x1 y1 x2 y2 k1 k2 : ℝ)
  (h1 : ellipse1 a b x1 y1)
  (h2 : ellipse1 a b x2 y2)
  (h3 : ellipse2 a b x1 y1)
  (h4 : ellipse2 a b x2 y2)
  (conj1 : conjugate_diameters a b m)
  (tangent_slope1 : k1 = -b^2 / a^2 * (1 / m))
  (conj2 : conjugate_diameters a b (-b^4/a^4 * 1/m))
  (tangent_slope2 : k2 = -b^4 / a^4 * (1 / (-b^4/a^4 * (1/m))))
: k1 * k2 = -1 :=
sorry

end tangents_form_rectangle_l642_642812


namespace cos_150_eq_neg_sqrt3_over_2_l642_642205

theorem cos_150_eq_neg_sqrt3_over_2 :
  ∀ {deg cosQ1 cosQ2: ℝ},
    cosQ1 = 30 →
    cosQ2 = 180 - cosQ1 →
    cos 150 = - (cos cosQ1) →
    cos cosQ1 = sqrt 3 / 2 →
    cos 150 = -sqrt 3 / 2 :=
by
  intros deg cosQ1 cosQ2 h1 h2 h3 h4
  sorry

end cos_150_eq_neg_sqrt3_over_2_l642_642205


namespace positive_number_property_l642_642668

theorem positive_number_property (x : ℝ) (h : (100 - x) / 100 * x = 16) :
  x = 40 ∨ x = 60 :=
sorry

end positive_number_property_l642_642668


namespace sum_of_series_l642_642119

theorem sum_of_series : (∑ n in Finset.range 5, (1 / (↑(n + 2) * (↑(n + 3))))) = (5 / 14) :=
by
  sorry

end sum_of_series_l642_642119


namespace power_function_condition_l642_642807

noncomputable def k := 1
variable (α : ℝ)

theorem power_function_condition (h : (k : ℝ) * 2 ^ α = 4) : k + α = 3 := by
  have h₁ : (k : ℝ) = 1 := rfl
  rw [h₁, one_mul] at h
  have h₂ : 2 ^ α = 4 := h
  norm_num at h₂ -- solve 2 ^ α = 4 to α = 2
  exact h₁ ▸ h₂ ▸ rfl

end power_function_condition_l642_642807


namespace distance_between_planes_l642_642748

-- Define the planes using their equations
def plane1 (x y z : ℝ) : Prop := 3 * x - 2 * y + 4 * z = 12
def plane2 (x y z : ℝ) : Prop := 6 * x - 4 * y + 8 * z = 5

-- Prove that the distance between the planes plane1 and plane2 is expected_dist
def expected_dist : ℝ := 7 * Real.sqrt 29 / 29

theorem distance_between_planes : 
  ∃ (d : ℝ), d = expected_dist ∧ 
             ∀ x y z : ℝ, plane1 x y z → 
                           ∀ x' y' z' : ℝ, plane2 x' y' z' → dist (x, y, z) (x', y', z') = d := 
sorry

end distance_between_planes_l642_642748


namespace lattice_points_in_intersection_l642_642335

noncomputable def inside_first_sphere (x y z : ℤ) : Prop :=
  x^2 + y^2 + (z - 14)^2 ≤ 49

noncomputable def inside_second_sphere (x y z : ℤ) : Prop :=
  x^2 + y^2 + (z - 1)^2 ≤ 121 / 4

theorem lattice_points_in_intersection : 
  { p : ℤ × ℤ × ℤ // inside_first_sphere p.1 p.2 p.3 ∧ inside_second_sphere p.1 p.2 p.3 } = 13 :=
by 
  unfold inside_first_sphere inside_second_sphere
  sorry

end lattice_points_in_intersection_l642_642335


namespace train_X_distance_at_meet_l642_642017

-- Define constants for distances and times
constant distance : ℝ := 160
constant time_X : ℝ := 5
constant time_Y : ℝ := 3

-- Define the speeds of Train X and Train Y based on given times and distance
def speed_X : ℝ := distance / time_X
def speed_Y : ℝ := distance / time_Y

-- Define the combined speed when both trains move towards each other
def combined_speed : ℝ := speed_X + speed_Y

-- Define the time taken to meet each other
def meet_time : ℝ := distance / combined_speed

-- Proposition: Train X travels 60 km when it meets Train Y.
theorem train_X_distance_at_meet : speed_X * meet_time = 60 := by
  sorry

end train_X_distance_at_meet_l642_642017


namespace sum_infinite_series_l642_642718

theorem sum_infinite_series : ∑' n : ℕ, (4 * (n + 1) - 3) / (3 ^ (n + 1)) = 3 / 2 := by
    sorry

end sum_infinite_series_l642_642718


namespace cos_150_eq_neg_sqrt3_over_2_l642_642230

theorem cos_150_eq_neg_sqrt3_over_2 : Real.cos (150 * Real.pi / 180) = -Real.sqrt 3 / 2 := 
by 
  sorry

end cos_150_eq_neg_sqrt3_over_2_l642_642230


namespace cos_150_deg_l642_642316

-- Define the conditions as variables
def angle1 := 150
def angle2 := 180
def angle3 := 30
def cos30 := Real.sqrt 3 / 2

-- The main statement to be proved
theorem cos_150_deg : Real.cos (150 * Real.pi / 180) = - (Real.sqrt 3 / 2) :=
by
  sorry

end cos_150_deg_l642_642316


namespace cos_150_eq_neg_sqrt3_div_2_l642_642262

theorem cos_150_eq_neg_sqrt3_div_2 :
  let theta := 30 * Real.pi / 180 in
  let Q := 150 * Real.pi / 180 in
  cos Q = - (Real.sqrt 3 / 2) := by
    sorry

end cos_150_eq_neg_sqrt3_div_2_l642_642262


namespace cos_150_eq_neg_sqrt3_div_2_l642_642300

open Real

theorem cos_150_eq_neg_sqrt3_div_2 :
  cos (150 * pi / 180) = - (sqrt 3 / 2) := 
sorry

end cos_150_eq_neg_sqrt3_div_2_l642_642300


namespace cos_150_eq_neg_sqrt3_div_2_l642_642135

theorem cos_150_eq_neg_sqrt3_div_2 :
  ∃ θ : ℝ, θ = 150 ∧
           0 < θ ∧ θ < 180 ∧
           ∃ φ : ℝ, φ = 180 - θ ∧
           cos φ = Real.sqrt 3 / 2 ∧
           cos θ = - cos φ :=
  sorry

end cos_150_eq_neg_sqrt3_div_2_l642_642135


namespace last_locker_opened_l642_642640

-- Define the predicate indicating if a locker is open or closed after a given pass
def locker_state (n : ℕ) (lockers : Finset ℕ) : List ℕ :=
  ((list.range n).filter (λ k, k ∈ lockers))

-- Define the conditions provided in the problem
def initial_lockers : Finset ℕ := (Finset.range 768).filter (λ k, k % 2 = 0)

def subsequent_lockers (lockers : Finset ℕ) : Finset ℕ := -- This needs to follow the described pattern and require a proper definition
 
-- The theorem we want to prove
theorem last_locker_opened :
  (∀ n ∈ initial_lockers, locker_state n subsequent_lockers) →
  ∃ i : ℕ, i = 257 := sorry

end last_locker_opened_l642_642640


namespace cos_150_eq_neg_sqrt3_div_2_l642_642266

theorem cos_150_eq_neg_sqrt3_div_2 :
  let theta := 30 * Real.pi / 180 in
  let Q := 150 * Real.pi / 180 in
  cos Q = - (Real.sqrt 3 / 2) := by
    sorry

end cos_150_eq_neg_sqrt3_div_2_l642_642266


namespace sin_theta_is_one_l642_642828

theorem sin_theta_is_one (θ : ℝ) (h1 : 10 * tan θ = 5 * cos θ) (h2 : 0 < θ ∧ θ < π / 2) : sin θ = 1 := 
by
  sorry

end sin_theta_is_one_l642_642828


namespace minimum_time_to_reconnect_chain_l642_642070

theorem minimum_time_to_reconnect_chain (parts rings : ℕ) (time_per_cut : ℕ) :
  parts = 5 → rings = 3 → time_per_cut = 2 →
  ∃ min_time, min_time = 6 :=
by
  intros h_parts h_rings h_time_per_cut
  use 6
  -- skipping actual proof for the minimal time
  sorry

end minimum_time_to_reconnect_chain_l642_642070


namespace finding_x_value_inversely_proportional_to_square_l642_642048

theorem finding_x_value_inversely_proportional_to_square (k : ℝ) (y : ℝ) (x : ℝ)
  (h1 : x = k / y^2) (h2 : y = 2) (h3 : x = 1) : 
  (∀ y, y = 6 → x = 4 / 36) := 
  by
    intros
    rw h1
    sorry

end finding_x_value_inversely_proportional_to_square_l642_642048


namespace number_of_chlorine_atoms_l642_642749

def molecular_weight_of_aluminum : ℝ := 26.98
def molecular_weight_of_chlorine : ℝ := 35.45
def molecular_weight_of_compound : ℝ := 132.0

theorem number_of_chlorine_atoms :
  ∃ n : ℕ, molecular_weight_of_compound = molecular_weight_of_aluminum + n * molecular_weight_of_chlorine ∧ n = 3 :=
by
  sorry

end number_of_chlorine_atoms_l642_642749


namespace problem_one_problem_two_l642_642783

variable {α : ℝ}

theorem problem_one (h : Real.tan (π + α) = -1 / 2) :
  (2 * Real.cos (π - α) - 3 * Real.sin (π + α)) / (4 * Real.cos (α - 2 * π) + Real.sin (4 * π - α)) = -7 / 9 :=
sorry

theorem problem_two (h : Real.tan (π + α) = -1 / 2) :
  Real.sin (α - 7 * π) * Real.cos (α + 5 * π) = -2 / 5 :=
sorry

end problem_one_problem_two_l642_642783


namespace nate_distance_after_resting_l642_642536

variables (length_of_field total_distance : ℕ)

def distance_before_resting (length_of_field : ℕ) := 4 * length_of_field

def distance_after_resting (total_distance length_of_field : ℕ) : ℕ := 
  total_distance - distance_before_resting length_of_field

theorem nate_distance_after_resting
  (length_of_field_val : length_of_field = 168)
  (total_distance_val : total_distance = 1172) :
  distance_after_resting total_distance length_of_field = 500 :=
by
  -- Proof goes here
  sorry

end nate_distance_after_resting_l642_642536


namespace length_EC_l642_642055

-- Define Trapezoid and its properties
def trapezoid (ABCD : Type) :=
  ∀ (A B C D: Point) (E: Point),
  parallel (A B) (C D) ∧
  (length (A B) = 3 * length (C D)) ∧
  diagonalsIntersectAt E A C B D ∧
  (length (A C) = 15)

-- Define the Points and lengths
variable (ABCD : Type)
variable [trapezoid ABCD]
variable (A B C D E : Point)
variable (length_AC : length A C = 15)

-- The proof statement asserting the length of EC
theorem length_EC :
  length E C = 15 / 4 :=
sorry -- proof to be provided here

end length_EC_l642_642055


namespace number_of_purple_shells_l642_642950

def tamtam_collected_65_shells : ℕ := 65
def pink_shells : ℕ := 8
def yellow_shells : ℕ := 18
def blue_shells : ℕ := 12
def orange_shells : ℕ := 14

theorem number_of_purple_shells  (total_shells pink_shells yellow_shells blue_shells orange_shells : ℕ) :
  total_shells = 65 → pink_shells = 8 → yellow_shells = 18 → blue_shells = 12 → orange_shells = 14 →
  (∃ purple_shells, total_shells = pink_shells + yellow_shells + blue_shells + orange_shells + purple_shells ∧ purple_shells = 13) :=
by
  intros h1 h2 h3 h4 h5
  use ![65 - (8 + 18 + 12 + 14), rfl]
  }
  sorry

end number_of_purple_shells_l642_642950


namespace color_of_85th_bead_l642_642869

inductive Color
| red
| orange
| yellow
| green
| blue

open Color

def bead_sequence : List Color := 
  [red, red, orange, yellow, yellow, yellow, green, blue, blue]

def color_of_nth_bead (n : Nat) : Color :=
  bead_sequence.get! (n % bead_sequence.length)

theorem color_of_85th_bead : 
  color_of_nth_bead 84 = yellow :=
by
  sorry

end color_of_85th_bead_l642_642869


namespace cosine_150_eq_neg_sqrt3_div_2_l642_642242

theorem cosine_150_eq_neg_sqrt3_div_2 :
  (∀ θ : ℝ, θ = real.pi / 6) →
  (∀ θ : ℝ, cos (real.pi - θ) = -cos θ) →
  cos (5 * real.pi / 6) = -real.sqrt 3 / 2 :=
by
  intros hθ hcos_identity
  sorry

end cosine_150_eq_neg_sqrt3_div_2_l642_642242


namespace time_to_fill_cans_l642_642666

def capacity_cans_filled (n : ℕ) (d : nat) (cap : ℕ) (frac : ℕ) (h : ℕ) : ℕ :=
  (cap / frac) * d * frac

theorem time_to_fill_cans (num_cans_init : ℕ) (num_cans_final : ℕ) (capacity : ℕ) (fraction : ℕ)
  (time_initial : ℕ) (time_final : ℕ) (rate : ℕ) :
  capacity_cans_filled 20 6 8 4 3 = 120 → 
  rate = 40 → 
  time_final = (num_cans_final * capacity) / rate → 
  time_final = 5 :=
by
  intro h1 h2 h3
  rw [h1, h2] at *
  sorry

end time_to_fill_cans_l642_642666


namespace cos_150_eq_neg_sqrt3_div_2_l642_642258

theorem cos_150_eq_neg_sqrt3_div_2 :
  let theta := 30 * Real.pi / 180 in
  let Q := 150 * Real.pi / 180 in
  cos Q = - (Real.sqrt 3 / 2) := by
    sorry

end cos_150_eq_neg_sqrt3_div_2_l642_642258


namespace speed_of_faster_train_l642_642608

theorem speed_of_faster_train (V_s : ℝ) (V_s_value : V_s = 32) (time_pass : ℝ) (time_pass_value : time_pass = 15) 
  (length_train : ℝ) (length_train_value : length_train = 75) : 
  ∃ V_f : ℝ, V_f = 50 := 
by
  have rel_speed : ℝ := length_train / time_pass
  have rel_speed_value : rel_speed = 75 / 15 := by rw [length_train_value, time_pass_value]
  have rel_speed_kmh : ℝ := rel_speed * 3.6
  have rel_speed_kmh_value : rel_speed_kmh = (75 / 15) * 3.6 := by rw rel_speed_value
  have rel_speed_kmh_final : rel_speed_kmh = 18 := by norm_num [rel_speed_kmh_value]
  let V_f := V_s + rel_speed_kmh
  have V_f_value : V_f = 32 + 18 := by rw [V_s_value, rel_speed_kmh_final]
  have V_f_final : V_f = 50 := by norm_num [V_f_value]
  use V_f
  exact V_f_final

end speed_of_faster_train_l642_642608


namespace negation_of_universal_l642_642579

variable {f g : ℝ → ℝ}

theorem negation_of_universal :
  ¬ (∀ x : ℝ, f x * g x ≠ 0) ↔ ∃ x₀ : ℝ, f x₀ = 0 ∨ g x₀ = 0 :=
by
  sorry

end negation_of_universal_l642_642579


namespace largest_reciprocal_l642_642621

/--
Theorem: Among the given numbers, the number with the smallest value has the largest reciprocal.

Given:
- n1 : ℚ := 3/7
- n2 : ℚ := 1/2
- n3 : ℚ := 3/4
- n4 : ℝ := 4
- n5 : ℝ := 100

Prove:
The reciprocal of the smallest number is the largest among the reciprocals.
-/
theorem largest_reciprocal :
  let n1 := (3 : ℚ) / 7,
      n2 := (1 : ℚ) / 2,
      n3 := (3 : ℚ) / 4,
      n4 := (4 : ℝ),
      n5 := (100 : ℝ)
  in 1 / n2 < 1 / n1 ∧ 1 / n3 < 1 / n1 ∧ 1 / n4 < 1 / n1 ∧ 1 / n5 < 1 / n1 :=
by
  sorry

end largest_reciprocal_l642_642621


namespace jack_piggy_bank_after_8_weeks_l642_642872

-- Conditions as definitions
def initial_amount : ℕ := 43
def weekly_allowance : ℕ := 10
def saved_fraction (x : ℕ) : ℕ := x / 2
def duration : ℕ := 8

-- Mathematical equivalent proof problem
theorem jack_piggy_bank_after_8_weeks : initial_amount + (duration * saved_fraction weekly_allowance) = 83 := by
  sorry

end jack_piggy_bank_after_8_weeks_l642_642872


namespace airplane_altitude_is_correct_l642_642679

noncomputable def airplane_altitude (d : ℝ) (angle_alice : ℝ) (angle_bob : ℝ) : ℝ :=
  let h_alice := d * (Real.sin angle_bob / Real.sin (angle_alice + angle_bob))
  in Real.sqrt (d^2 + h_alice^2)

theorem airplane_altitude_is_correct :
  ∀ (d : ℝ) (angle_alice angle_bob : ℝ),
    d = 10 ∧ angle_alice = Real.pi / 6 ∧ angle_bob = Real.pi / 3 →
    airplane_altitude d angle_alice angle_bob ≈ 5.5 :=
by
  intros d angle_alice angle_bob h
  obtain ⟨h₁, h₂, h₃⟩ := h
  rw [h₁, h₂, h₃]
  sorry

end airplane_altitude_is_correct_l642_642679


namespace count_irrationals_in_set_l642_642687

def is_rational (x : ℝ) : Prop := ∃ (a b : ℤ), b ≠ 0 ∧ x = a / b

def is_irrational (x : ℝ) : Prop := ¬ is_rational x

def num_irrationals (l : List ℝ) : Nat :=
  l.countp is_irrational

theorem count_irrationals_in_set :
  num_irrationals [-4, 23 / 7, 3.1415, -3 * Real.pi, 3.030030003] = 2 :=
by
  sorry

end count_irrationals_in_set_l642_642687


namespace student_ticket_cost_l642_642601

theorem student_ticket_cost :
  ∀ (S : ℤ),
  (525 - 388) * S + 388 * 6 = 2876 → S = 4 :=
by
  sorry

end student_ticket_cost_l642_642601


namespace sum_of_sequence_l642_642344

theorem sum_of_sequence (n : ℕ) :
  (∑ k in Finset.range (n + 1), 3 * k + 2) + 5 = (3 * n^2 + 7 * n + 14) / 2 :=
by
  sorry

end sum_of_sequence_l642_642344


namespace time_to_fill_cans_l642_642665

def capacity_cans_filled (n : ℕ) (d : nat) (cap : ℕ) (frac : ℕ) (h : ℕ) : ℕ :=
  (cap / frac) * d * frac

theorem time_to_fill_cans (num_cans_init : ℕ) (num_cans_final : ℕ) (capacity : ℕ) (fraction : ℕ)
  (time_initial : ℕ) (time_final : ℕ) (rate : ℕ) :
  capacity_cans_filled 20 6 8 4 3 = 120 → 
  rate = 40 → 
  time_final = (num_cans_final * capacity) / rate → 
  time_final = 5 :=
by
  intro h1 h2 h3
  rw [h1, h2] at *
  sorry

end time_to_fill_cans_l642_642665


namespace cos_150_eq_neg_sqrt3_over_2_l642_642203

theorem cos_150_eq_neg_sqrt3_over_2 :
  ∀ {deg cosQ1 cosQ2: ℝ},
    cosQ1 = 30 →
    cosQ2 = 180 - cosQ1 →
    cos 150 = - (cos cosQ1) →
    cos cosQ1 = sqrt 3 / 2 →
    cos 150 = -sqrt 3 / 2 :=
by
  intros deg cosQ1 cosQ2 h1 h2 h3 h4
  sorry

end cos_150_eq_neg_sqrt3_over_2_l642_642203


namespace cos_150_eq_neg_half_l642_642199

theorem cos_150_eq_neg_half :
  real.cos (150 * real.pi / 180) = -1 / 2 :=
sorry

end cos_150_eq_neg_half_l642_642199


namespace paper_boat_travel_time_l642_642482

-- Defining the conditions as constants
def distance_embankment : ℝ := 50
def speed_downstream : ℝ := 10
def speed_upstream : ℝ := 12.5

-- Definitions for the speeds of the boat and current
noncomputable def v_boat : ℝ := (speed_upstream + speed_downstream) / 2
noncomputable def v_current : ℝ := (speed_downstream - speed_upstream) / 2

-- Statement to prove the time taken for the paper boat
theorem paper_boat_travel_time :
  (distance_embankment / v_current) = 40 := by
  sorry

end paper_boat_travel_time_l642_642482


namespace blue_to_white_ratio_l642_642671

/-- 
Given that in a large arrangement of regular hexagonal tiles:
1. Each blue tile is surrounded by 6 white tiles.
2. Each white tile is surrounded by 3 white and 3 blue tiles.

Prove that the ratio of the number of blue tiles (B) to the number of white tiles (W) is 1:2. 
-/
theorem blue_to_white_ratio : (B W : ℕ) → (h1 : ∀ b, surrounded_by b 6) → (h2 : ∀ w, surrounded_by w 3 3) → (W = 2 * B) :=
by
  sorry

end blue_to_white_ratio_l642_642671


namespace ratio_of_ages_l642_642089

-- Definitions based on given mathematical problem conditions
def man_age := 36
def son_age := 12

-- The statement to be proved
theorem ratio_of_ages : man_age / son_age = 3 := by
  simp [man_age, son_age]
  sorry

end ratio_of_ages_l642_642089


namespace cos_150_eq_neg_sqrt3_div_2_l642_642170

theorem cos_150_eq_neg_sqrt3_div_2 : Real.cos (150 * Real.pi / 180) = - (Real.sqrt 3 / 2) :=
by
  sorry

end cos_150_eq_neg_sqrt3_div_2_l642_642170


namespace domain_of_f_l642_642360

def f (x : ℝ) : ℝ := real.sqrt (x - 1) + real.cbrt (8 - x)

theorem domain_of_f :
  {x : ℝ | 1 ≤ x ∧ x ≤ 8} = {x : ℝ | x ∈ set.Icc 1 8} := sorry

end domain_of_f_l642_642360


namespace point_in_fourth_quadrant_l642_642468

-- Definitions of the quadrants as provided in the conditions
def first_quadrant (x y : ℝ) : Prop := x > 0 ∧ y > 0
def second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0
def third_quadrant (x y : ℝ) : Prop := x < 0 ∧ y < 0
def fourth_quadrant (x y : ℝ) : Prop := x > 0 ∧ y < 0

-- Given point
def point : ℝ × ℝ := (1, -2)

-- Theorem statement
theorem point_in_fourth_quadrant : fourth_quadrant point.fst point.snd :=
sorry

end point_in_fourth_quadrant_l642_642468


namespace exactly_five_valid_values_for_N_l642_642511

noncomputable def isValidN (N : ℕ) : Prop :=
  ∃ a b : ℕ, N = 10 * a + b ∧ 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧
  (let diff := N - (10 * b + a) in diff > 16 ∧ ∃ k : ℕ, k^2 = diff)

theorem exactly_five_valid_values_for_N :
  {N : ℕ | isValidN N}.toFinset.card = 5 := sorry

end exactly_five_valid_values_for_N_l642_642511


namespace minimize_sum_of_squares_l642_642898

theorem minimize_sum_of_squares :
  ∃ (a b c d : ℝ), 
    (a + 3 * b + 5 * c + 7 * d = 14 ∧ 
    a^2 + b^2 + c^2 + d^2 = 7 / 3 ∧ 
    a = 1 / 6 ∧ 
    b = 1 / 2 ∧ 
    c = 5 / 6 ∧ 
    d = 7 / 6) :=
begin
  sorry
end

end minimize_sum_of_squares_l642_642898


namespace sin_value_l642_642340

theorem sin_value (x : ℝ) (h : Real.sec x + Real.tan x = 5/4) : Real.sin x = 1/41 := sorry

end sin_value_l642_642340


namespace quadratic_function_f_inequality_g_solution_l642_642836

noncomputable def f (x : ℝ) : ℝ := (-1/2) * x^2 - 2 * x

noncomputable def g (x : ℝ) : ℝ := x * Real.log x + f x

theorem quadratic_function_f (x : ℝ) :
  (f (x + 1) + f x = (-x^2 - 5*x - (5 / 2))) →
  (f x = (-1/2) * x^2 - 2 * x) :=
by
  sorry

theorem inequality_g_solution (x : ℝ) :
  (g(x) = x * Real.log x + (-1/2) * x^2 - 2 * x) →
  (g (x^2 + x) ≥ g 2 ↔ (x ∈ Set.Icc (-2) (-1) ∪ Set.Icc 0 1)) :=
by
  sorry

end quadratic_function_f_inequality_g_solution_l642_642836


namespace sum_of_octahedron_faces_l642_642967

theorem sum_of_octahedron_faces (n : ℕ) :
  n + (n+1) + (n+2) + (n+3) + (n+4) + (n+5) + (n+6) + (n+7) = 8 * n + 28 :=
by
  sorry

end sum_of_octahedron_faces_l642_642967


namespace total_questions_on_test_l642_642674

theorem total_questions_on_test :
  ∀ (correct incorrect score : ℕ),
  (score = correct - 2 * incorrect) →
  (score = 76) →
  (correct = 92) →
  (correct + incorrect = 100) :=
by
  intros correct incorrect score grading_system score_eq correct_eq
  sorry

end total_questions_on_test_l642_642674


namespace ricsi_win_probability_l642_642936

theorem ricsi_win_probability :
  let P1 := 3/4 -- Probability that Ricsi loses when Dénes and Attila play against him
  let P2 := 1/2 -- Probability that Ricsi loses when Dénes plays against Ricsi and Attila
  let P3 := 2/3 -- Probability that Ricsi loses when Attila plays against Ricsi and Dénes
  let always_lose := (P1 * P1) * (P2 * P2) * (P3 * P3)
  let win_at_least_once := 1 - always_lose in
  win_at_least_once = 15/16 :=
by
  -- calculations to be done here
  sorry

end ricsi_win_probability_l642_642936


namespace trajectory_of_Q_l642_642512

variable (x y : ℝ)

def ellipse (x y : ℝ) : Prop := (x^2) / 7 + (y^2) / 3 = 1
def foci_left : ℝ × ℝ := (-2, 0)
def radius := 2 * Real.sqrt 7

theorem trajectory_of_Q :
  (∀ (P : ℝ × ℝ), ellipse P.1 P.2 → ∃ (Q : ℝ × ℝ), 
    (Q.1 + 2)^2 + Q.2^2 = 28) :=
by
  sorry

end trajectory_of_Q_l642_642512


namespace digit_inequality_solution_l642_642943

theorem digit_inequality_solution : (Finset.card (Finset.filter (λ d, 2 + d / 100 + 5 / 10000 > 2.015) (Finset.range 10))) = 8 := 
by
  sorry

end digit_inequality_solution_l642_642943


namespace paper_boat_time_proof_l642_642477

/-- A 50-meter long embankment exists along a river.
 - A motorboat that passes this embankment in 5 seconds while moving downstream.
 - The same motorboat passes this embankment in 4 seconds while moving upstream.
 - Determine the time in seconds it takes for a paper boat, which moves with the current, to travel the length of this embankment.
 -/
noncomputable def paper_boat_travel_time 
  (embankment_length : ℝ)
  (motorboat_length : ℝ)
  (time_downstream : ℝ)
  (time_upstream : ℝ) : ℝ :=
  let v_eff_downstream := embankment_length / time_downstream,
      v_eff_upstream := embankment_length / time_upstream,
      v_boat := (v_eff_downstream + v_eff_upstream) / 2,
      v_current := (v_eff_downstream - v_eff_upstream) / 2 in
  embankment_length / v_current

theorem paper_boat_time_proof :
  paper_boat_travel_time 50 10 5 4 = 40 := 
begin
  sorry,
end

end paper_boat_time_proof_l642_642477


namespace probability_fourth_quadrant_l642_642392

open Set

def A : Set ℤ := {-2, 1, 2}
def B : Set ℤ := {-1, 1, 3}

def passes_fourth_quadrant (a b : ℤ) : Prop :=
  (a < 0 ∧ b > 0) ∨ (a < 0 ∧ b < 0) ∨ (a > 0 ∧ b < 0)

def successful_events : Finset (ℤ × ℤ) :=
  {(a, b) | a ∈ A ∧ b ∈ B ∧ passes_fourth_quadrant a b}.to_finset

def total_events : ℕ := (A.to_finset.card) * (B.to_finset.card)

def successful_event_count : ℕ := successful_events.card

theorem probability_fourth_quadrant :
  (successful_event_count : ℚ) / total_events = 5 / 9 := by
  sorry

end probability_fourth_quadrant_l642_642392


namespace range_of_a_l642_642391

def A : Set ℝ := { x | x^2 - x - 2 > 0 }
def B (a : ℝ) : Set ℝ := { x | abs (x - a) < 3 }

theorem range_of_a (a : ℝ) :
  (A ∪ B a = Set.univ) → a ∈ Set.Ioo (-1 : ℝ) 2 :=
by
  sorry

end range_of_a_l642_642391


namespace transaction_loss_l642_642088

theorem transaction_loss :
  let house_sale_price := 10000
  let store_sale_price := 15000
  let house_loss_percentage := 0.25
  let store_gain_percentage := 0.25
  let h := house_sale_price / (1 - house_loss_percentage)
  let s := store_sale_price / (1 + store_gain_percentage)
  let total_cost_price := h + s
  let total_selling_price := house_sale_price + store_sale_price
  let difference := total_selling_price - total_cost_price
  difference = -1000 / 3 :=
by
  sorry

end transaction_loss_l642_642088


namespace find_water_bottles_l642_642642

def water_bottles (W A : ℕ) :=
  A = W + 6 ∧ W + A = 54 → W = 24

theorem find_water_bottles (W A : ℕ) (h1 : A = W + 6) (h2 : W + A = 54) : W = 24 :=
by sorry

end find_water_bottles_l642_642642


namespace hazel_walked_distance_l642_642428

theorem hazel_walked_distance
  (first_hour_distance : ℕ)
  (second_hour_distance : ℕ)
  (h1 : first_hour_distance = 2)
  (h2 : second_hour_distance = 2 * first_hour_distance) :
  (first_hour_distance + second_hour_distance = 6) :=
by {
  sorry
}

end hazel_walked_distance_l642_642428


namespace cos_150_degree_l642_642180

theorem cos_150_degree : Real.cos (150 * Real.pi / 180) = -Real.sqrt 3 / 2 :=
by
  sorry

end cos_150_degree_l642_642180


namespace sin_double_angle_cos_angle_difference_l642_642781

noncomputable def given_angles : Prop :=
  ∃ (α β : ℝ), (0 < α ∧ α < π / 2) ∧ (0 < β ∧ β < π / 2) ∧ 
  (sin α = 2 * real.sqrt 5 / 5) ∧ (cos (π - β) = - real.sqrt 2 / 10)

theorem sin_double_angle (h : given_angles) :
  ∃ (α : ℝ), (0 < α ∧ α < π / 2) ∧ (sin α = 2 * real.sqrt 5 / 5) → sin (2 * α) = 4 / 5 :=
sorry

theorem cos_angle_difference (h : given_angles) :
  ∃ (α β : ℝ), (0 < α ∧ α < π / 2) ∧ (0 < β ∧ β < π / 2) ∧ 
  (sin α = 2 * real.sqrt 5 / 5) ∧ (cos (π - β) = - real.sqrt 2 / 10) → cos (α - β) = 3 * real.sqrt 10 / 10 :=
sorry

end sin_double_angle_cos_angle_difference_l642_642781


namespace union_of_P_and_Q_l642_642453

theorem union_of_P_and_Q {α : Type*} (P Q : Set α) :
  (P = {-2, 2} ∧ Q = {-1, 0, 2, 3}) → (P ∪ Q = {-2, -1, 0, 2, 3}) :=
by
  assume h
  sorry

end union_of_P_and_Q_l642_642453


namespace slope_of_tangent_line_l642_642382

theorem slope_of_tangent_line :
  ∀ (P : ℝ × ℝ), P = (-2, 0) → (∀ (x y : ℝ), x^2 + y^2 = 1 → (∃ k : ℝ, (y = k * (x + 2)) ∧ (k = ±(√3 / 3)))) :=
by 
  -- proof should be inserted here
  sorry

end slope_of_tangent_line_l642_642382


namespace xiwangbei_jiushi_hao_l642_642859

def chinese_to_digit (ch : Char) : ℕ := sorry

noncomputable def solution : ℕ :=
  let X := 256000 + 410 in
  let lhs := (X * 1000 + 410) * 8
  let rhs := (410 * 1000 + 256) * 5
  if lhs = rhs then X else 0

theorem xiwangbei_jiushi_hao : solution = 256410 :=
  sorry

end xiwangbei_jiushi_hao_l642_642859


namespace inequality_proof_l642_642522

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a + 3 * c) / (a + 2 * b + c) + (4 * b) / (a + b + 2 * c) - (8 * c) / (a + b + 3 * c) ≥ -17 + 12 * Real.sqrt 2 :=
  sorry

end inequality_proof_l642_642522


namespace cos_150_eq_neg_half_l642_642200

theorem cos_150_eq_neg_half :
  real.cos (150 * real.pi / 180) = -1 / 2 :=
sorry

end cos_150_eq_neg_half_l642_642200


namespace find_boat_speed_l642_642752

variables (a c x : ℝ)

noncomputable def boat_speed_in_still_water (a c : ℝ) : ℝ :=
  (3 * a + 2 * c + Real.sqrt (9 * a^2 - 4 * a * c + 4 * c^2)) / 4

theorem find_boat_speed
  (h1 : a > 0)
  (h2 : c ≥ 0)
  (h3 : ∀ x, (x > c → ((a / x) + (a / (2 * (x - c))) = 1))) :
  x = boat_speed_in_still_water a c :=
sorry

end find_boat_speed_l642_642752


namespace find_x_l642_642837

theorem find_x : 
  let A := 720 * Real.sqrt 1152 in
  let B := Real.cbrt 15625 in
  let C := 7932 / (3^2 - Real.sqrt 196) in
  ∃ x, (x = -39660 - 17280 * Real.sqrt 2) ∧ ((x + A) / B = C) :=
by
  sorry

end find_x_l642_642837


namespace normal_guitar_strings_l642_642880

theorem normal_guitar_strings : 
  (∃ (x : ℕ), let basses := 3, bass_strings := 4, 
    guitars := 2 * basses, 
    guitar_strings := x, 
    eight_string_guitars := guitars - 3, 
    eight_string_guitar_strings := 8 
    in let total_strings := (basses * bass_strings) + (guitars * guitar_strings) + (eight_string_guitars * eight_string_guitar_strings) 
    in total_strings = 72 ∧ guitar_strings = 6) :=
begin
  sorry
end

end normal_guitar_strings_l642_642880


namespace order_of_magnitude_l642_642785

def a : ℝ := (0.2 : ℝ) ^ (0.3 : ℝ)
def b : ℝ := Real.log 1 / Real.log 0.2
def c : ℝ := Real.log 4 / Real.log 0.2

theorem order_of_magnitude : a > b ∧ b > c :=
by
  sorry

end order_of_magnitude_l642_642785


namespace f1_is_not_D_function_f2_is_D_function_F_is_D_function_number_of_D_functions_is_odd_l642_642903

def A : Set ℕ := {0, 1, 2, 3, 4, 5, 6}

def f1 (i : ℕ) : ℕ :=
  match i with
  | 0 => 0
  | 1 => 4
  | 2 => 6
  | 3 => 5
  | 4 => 1
  | 5 => 3
  | 6 => 2
  | _ => 0  -- not in A

def f2 (i : ℕ) : ℕ :=
  match i with
  | 0 => 1
  | 1 => 6
  | 2 => 4
  | 3 => 2
  | 4 => 0
  | 5 => 5
  | 6 => 3
  | _ => 0  -- not in A

def is_D_function (f : ℕ → ℕ) : Prop :=
  let d_i : ℕ → ℕ := λ i => (i - f i) % 7
  ∀ i j, i ∈ A → j ∈ A → i ≠ j → d_i i ≠ d_i j

theorem f1_is_not_D_function : ¬ is_D_function f1 :=
  by sorry

theorem f2_is_D_function: is_D_function f2 :=
  by sorry

theorem F_is_D_function (f : ℕ → ℕ) (h : is_D_function f) : 
  is_D_function (λ i => (i - f i) % 7) :=
  by sorry

theorem number_of_D_functions_is_odd : ∃ n : ℕ, is_D_function f ∧ 2 * n + 1 = nat.pred (nat.factorial 7) := 
  by sorry

end f1_is_not_D_function_f2_is_D_function_F_is_D_function_number_of_D_functions_is_odd_l642_642903


namespace cos_150_eq_neg_half_l642_642194

theorem cos_150_eq_neg_half :
  real.cos (150 * real.pi / 180) = -1 / 2 :=
sorry

end cos_150_eq_neg_half_l642_642194


namespace cos_150_eq_neg_sqrt3_over_2_l642_642226

theorem cos_150_eq_neg_sqrt3_over_2 : Real.cos (150 * Real.pi / 180) = -Real.sqrt 3 / 2 := 
by 
  sorry

end cos_150_eq_neg_sqrt3_over_2_l642_642226


namespace constant_term_binomial_expansion_l642_642831

theorem constant_term_binomial_expansion (a : ℝ) (h : 15 * a^2 = 120) : a = 2 * Real.sqrt 2 :=
sorry

end constant_term_binomial_expansion_l642_642831


namespace function_is_even_l642_642425

variables {G : Type*} [inner_product_space ℝ G]

def f (a b : G) (x : ℝ) : ℝ :=
  ∥(x • a + b)∥ ^ 2

theorem function_is_even (a b : G) (h_nonzero_a : a ≠ 0) (h_nonzero_b : b ≠ 0) (h_orth : ⟪a, b⟫ = 0) :
  ∀ x : ℝ, f a b x = f a b (-x) :=
by sorry

end function_is_even_l642_642425


namespace problem_part1_problem_part2_l642_642404

def binom (n k : ℕ) : ℕ := sorry  -- binomial coefficient function placeholder

def sum_of_binom_coefficients (n : ℕ) : ℕ :=
  2^n

def coefficient_second_term (n : ℕ) : ℕ :=
  2 * n

noncomputable def a_n (n : ℕ) : ℕ :=
  sum_of_binom_coefficients n

noncomputable def b_n (n : ℕ) : ℕ :=
  coefficient_second_term n

noncomputable def S_n (n : ℕ) : ℕ :=
  (n - 1) * 2^(n + 2) + 4

theorem problem_part1 (n : ℕ) : a_n n = 2^n ∧ b_n n = 2 * n :=
by sorry

theorem problem_part2 (n : ℕ) : (finset.range n).sum (λ k, a_n k * b_n k) = S_n n :=
by sorry

end problem_part1_problem_part2_l642_642404


namespace contractor_fine_l642_642077

theorem contractor_fine
  (days_worked : ℕ)
  (wage_per_day : ℕ)
  (days_absent : ℕ)
  (total_received : ℕ)
  (fine_per_day_absent : ℚ)
  (total_days : ℕ)
  (wage_received : ℚ)
  (fine_received : ℚ)
  (equation : wage_received - fine_received = total_received) :
  fine_per_day_absent = 7.5 :=
by
  let days_worked := total_days - days_absent
  let wage_received := days_worked * wage_per_day
  let fine_received := days_absent * fine_per_day_absent
  have h : equation := by sorry
  have total_days := 30
  have wage_per_day := 25
  have total_received := 685
  have days_absent := 2
  exact sorry

end contractor_fine_l642_642077


namespace cos_150_deg_l642_642325

-- Define the conditions as variables
def angle1 := 150
def angle2 := 180
def angle3 := 30
def cos30 := Real.sqrt 3 / 2

-- The main statement to be proved
theorem cos_150_deg : Real.cos (150 * Real.pi / 180) = - (Real.sqrt 3 / 2) :=
by
  sorry

end cos_150_deg_l642_642325


namespace cos_150_eq_neg_sqrt3_over_2_l642_642234

theorem cos_150_eq_neg_sqrt3_over_2 : Real.cos (150 * Real.pi / 180) = -Real.sqrt 3 / 2 := 
by 
  sorry

end cos_150_eq_neg_sqrt3_over_2_l642_642234


namespace taylor_series_coefficients_zero_or_nonzero_l642_642961

theorem taylor_series_coefficients_zero_or_nonzero {b c : ℝ} (hb : b > 0) (hc : c > 0) :
    let a_n := λ n : ℕ, (1 / n!) * (Real.sqrt (b^2 + c^2) ^ n * Real.cos (n * Real.arctan (c / b)))
    ∃ m : ℕ, a_n m = 0 → ∀ m, n : ℕ, a_n n = 0 :=
begin
    sorry
end

end taylor_series_coefficients_zero_or_nonzero_l642_642961


namespace cos_150_eq_neg_sqrt3_div_2_l642_642295

theorem cos_150_eq_neg_sqrt3_div_2 :
  (cos (150 * real.pi / 180)) = - (sqrt 3 / 2) := sorry

end cos_150_eq_neg_sqrt3_div_2_l642_642295


namespace area_of_triangle_MDA_l642_642460

-- Definitions of the given conditions
variable {r : ℝ} (O A B M D : EuclideanGeometry.Point ℝ)

-- Given conditions
axiom h1 : EuclideanGeometry.Circle.center_radius_eq O (2*r)
axiom h2 : EuclideanGeometry.Segment.length_eq AB (2*r)
axiom h3 : EuclideanGeometry.CenterPerpendicularBisectsChord O AB M
axiom h4 : EuclideanGeometry.PerpendicularFromMtoOA M OA D

-- Statement of the problem
theorem area_of_triangle_MDA (r : ℝ) (O A B M D : EuclideanGeometry.Point ℝ)
  [EuclideanGeometry.Circle O (2 * r)]
  [EuclideanGeometry.Segment AB = 2 * r]
  [EuclideanGeometry.CenterPerpendicularBisectsChord O AB M]
  [EuclideanGeometry.PerpendicularFromMtoOA M OA D] :
  EuclideanGeometry.TriangleArea M D A = (r^2 * Real.sqrt 3) / 8 :=
  sorry

end area_of_triangle_MDA_l642_642460


namespace remaining_volume_of_cube_after_cylinder_removal_l642_642648

-- Conditions
def side_length_of_cube := 6
def radius_of_cylinder := 3
def height_of_cylinder := side_length_of_cube

-- Calculations
def volume_of_cube := side_length_of_cube ^ 3
def volume_of_cylinder := Mathlib.pi * (radius_of_cylinder ^ 2) * height_of_cylinder
def remaining_volume := volume_of_cube - volume_of_cylinder

-- Proof Problem
theorem remaining_volume_of_cube_after_cylinder_removal :
  remaining_volume = 216 - 54 * Mathlib.pi := 
sorry

end remaining_volume_of_cube_after_cylinder_removal_l642_642648


namespace oxygen_atom_diameter_in_scientific_notation_l642_642570

theorem oxygen_atom_diameter_in_scientific_notation :
  0.000000000148 = 1.48 * 10^(-10) :=
sorry

end oxygen_atom_diameter_in_scientific_notation_l642_642570


namespace board_clear_if_and_only_if_divisible_by_3_l642_642895

-- We need to define the constraints and the main assertion
def L_tromino_move (n : ℕ) : Prop :=
  ∀ (grid : ℕ × ℕ → bool), -- representing an n x n grid 
    (∀ i j, i < n → j < n → grid (i, j) = false) → -- initially empty
    ∃ seq, -- there exists a sequence of moves
      (∀ s, s ∈ seq → (EXISTS_L_SHAPED_TROMINO grid s) ∨
           (EXISTS_FULLY_OCCUPIED_COLUMN grid s) ∨ 
           (EXISTS_FULLY_OCCUPIED_ROW grid s)) →
      (∀ i j, i < n → j < n → grid (i, j) = false)

theorem board_clear_if_and_only_if_divisible_by_3 (n : ℕ) (h : 2 ≤ n) :
  L_tromino_move n ↔ 3 ∣ n :=
sorry

end board_clear_if_and_only_if_divisible_by_3_l642_642895


namespace edges_in_5_face_prism_l642_642847

theorem edges_in_5_face_prism : ∀ (P : Type) [Prism P] (h : faces_count P = 5), edges_count P = 9 :=
by
  intro P _ h
  sorry


end edges_in_5_face_prism_l642_642847


namespace population_in_2001_l642_642581

-- Define the populations at specific years
def pop_2000 := 50
def pop_2002 := 146
def pop_2003 := 350

-- Define the population difference condition
def pop_condition (n : ℕ) (pop : ℕ → ℕ) :=
  pop (n + 3) - pop n = 3 * pop (n + 2)

-- Given that the population condition holds, and specific populations are known,
-- the population in the year 2001 is 100
theorem population_in_2001 :
  (∃ (pop : ℕ → ℕ), pop 2000 = pop_2000 ∧ pop 2002 = pop_2002 ∧ pop 2003 = pop_2003 ∧ 
    pop_condition 2000 pop) → ∃ (pop : ℕ → ℕ), pop 2001 = 100 :=
by
  -- Placeholder for the actual proof
  sorry

end population_in_2001_l642_642581


namespace cos_150_eq_neg_sqrt3_div_2_l642_642130

theorem cos_150_eq_neg_sqrt3_div_2 :
  ∃ θ : ℝ, θ = 150 ∧
           0 < θ ∧ θ < 180 ∧
           ∃ φ : ℝ, φ = 180 - θ ∧
           cos φ = Real.sqrt 3 / 2 ∧
           cos θ = - cos φ :=
  sorry

end cos_150_eq_neg_sqrt3_div_2_l642_642130


namespace pentagon_intersection_l642_642771

structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define points A, B, C, and C1 on the cube given edge length 1
def A := Point3D.mk 0 0 0
def B := Point3D.mk 1 0 0
def C := Point3D.mk 1 1 0
def C1 := Point3D.mk 1 1 1

-- Define point P on edge BC, parameterized by y with 0 <= y <= 1
def P (y : ℝ) (h : 0 ≤ y ∧ y ≤ 1) := Point3D.mk 1 y 0

-- Define point Q as the midpoint of edge CC1
def Q := Point3D.mk 1 1 (1/2)

-- Define the plane passing through points A, P, and Q
def plane (y : ℝ) (h : 0 ≤ y ∧ y ≤ 1) : Prop :=
  ∃ a b c d, 
    a * A.x + b * A.y + c * A.z + d = 0 ∧
    a * (P y h).x + b * (P y h).y + c * (P y h).z + d = 0 ∧
    a * Q.x + b * Q.y + c * Q.z + d = 0

-- We need to prove that the plane described above intersects the cube 
-- to form a pentagon if and only if BP ∈ (1/2, 1)

theorem pentagon_intersection (y : ℝ) (h : 0 ≤ y ∧ y ≤ 1) :
  plane y h → (1/2 < y ∧ y < 1) :=
begin
  sorry
end

end pentagon_intersection_l642_642771


namespace choir_robe_costs_l642_642669

theorem choir_robe_costs:
  ∀ (total_robes needed_robes total_cost robe_cost : ℕ),
  total_robes = 30 →
  needed_robes = 30 - 12 →
  total_cost = 36 →
  total_cost = needed_robes * robe_cost →
  robe_cost = 2 :=
by
  intros total_robes needed_robes total_cost robe_cost
  intro h_total_robes h_needed_robes h_total_cost h_cost_eq
  sorry

end choir_robe_costs_l642_642669


namespace place_mat_length_l642_642097

theorem place_mat_length (r : ℝ) (n : ℕ) (w : ℝ) (x : ℝ)
  (table_is_round : r = 3)
  (number_of_mats : n = 8)
  (mat_width : w = 1)
  (mat_length : ∀ (k: ℕ), 0 ≤ k ∧ k < n → (2 * r * Real.sin (Real.pi / n) = x)) :
  x = (3 * Real.sqrt 35) / 10 + 1 / 2 :=
sorry

end place_mat_length_l642_642097


namespace second_polygon_sides_l642_642015

theorem second_polygon_sides (s : ℝ) (h : s > 0) : 
  ∀ (n : ℕ), 
  let first_polygon_sides := 50 in
  let first_polygon_side_length := 3 * s in
  let second_polygon_side_length := s in
  let first_polygon_perimeter := first_polygon_sides * first_polygon_side_length in
  let second_polygon_perimeter := n * second_polygon_side_length in
  (first_polygon_perimeter = second_polygon_perimeter) → n = 150 :=
by
  intros n h_eq
  sorry

end second_polygon_sides_l642_642015


namespace locus_of_P_l642_642818

open Real

theorem locus_of_P (x y : ℝ) (hx : x^2 + y^2 = 3) (h_diff_neg : ∀ (P : ℝ × ℝ), 
    let M := (-1, 0) in
    let N := (1, 0) in
    let MP := (P.1 + 1, P.2) in
    let MN := (2, 0) in
    let NM := (-2, 0) in
    let PN := (P.1 - 1, P.2) in
    let diff := ((MP.fst * MN.fst + MP.snd * MN.snd) + (NM.fst * PN.fst + NM.snd * PN.snd)) / 2 - (MP.fst * MN.fst + MP.snd * MN.snd) in
    diff < 0
  ) : x > 0 :=
sorry

end locus_of_P_l642_642818


namespace find_matrix_N_l642_642746

variables {a b c d : ℝ}

def matrix_N : Matrix (Fin 2) (Fin 2) ℝ :=
  ![[a, b], [c, d]]

variables (a b c d)

def cond1 : Prop := matrix.vecMul (λ x y => x * y) (matrix_N a b c d) ![2,1] = ![5, -3]

def cond2 : Prop := matrix.vecMul (λ x y => x * y) (matrix_N a b c d) ![0,4] = ![20, -12]

theorem find_matrix_N (h1 : cond1 a b c d) (h2 : cond2 a b c d) : 
  matrix_N a b c d = ![[0, 5], [0, -3]] :=
by
  sorry -- Proof can be developed here

end find_matrix_N_l642_642746


namespace cos_150_eq_neg_sqrt3_div_2_l642_642251

theorem cos_150_eq_neg_sqrt3_div_2 :
  let theta := 30 * Real.pi / 180 in
  let Q := 150 * Real.pi / 180 in
  cos Q = - (Real.sqrt 3 / 2) := by
    sorry

end cos_150_eq_neg_sqrt3_div_2_l642_642251


namespace P_on_hyperbola_distance_to_focus_l642_642667

def distance_to_other_focus (x y : ℝ) : ℝ :=
  -- assuming the point P is such that the hyperbola condition is met
  let a := 5 in
  let b := real.sqrt 24 in
  let c := real.sqrt (a^2 + b^2) in -- c = sqrt(25 + 24) = sqrt(49) = 7
  if x ∈ { p : ℝ | (p / a) ^ 2 - (y / b) ^ 2 = 1 } then
    let d₁ := 11 in -- distance to one focus
    let d₂ := abs (d₁ - 2 * c) in -- distance to the other focus
    if d₂ < min (abs (c - a)) (a + c) then
      21 -- the only valid distance as per the hyperbola properties
    else 
      0
  else 
    0

theorem P_on_hyperbola_distance_to_focus
  (x y : ℝ) (h : ((x^2) / 25 - (y^2) / 24 = 1)) (d₁ : ℝ)
  (h₁ : d₁ = 11) :
  distance_to_other_focus x y = 21 := by
  sorry

end P_on_hyperbola_distance_to_focus_l642_642667


namespace seat_arrangement_count_l642_642647

def total_seats := 10
def num_armchairs := 7
def num_benches := 3

theorem seat_arrangement_count : nat.choose total_seats num_benches = 120 := by
  sorry

end seat_arrangement_count_l642_642647


namespace sum_series_eq_l642_642711

theorem sum_series_eq :
  ∑' n : ℕ, (4 * (n + 1) - 3) / 3^(n + 1) = 3 / 2 :=
sorry

end sum_series_eq_l642_642711


namespace fraction_of_students_who_walk_home_l642_642692

theorem fraction_of_students_who_walk_home (bus auto bikes scooters : ℚ) 
  (hbus : bus = 2/5) (hauto : auto = 1/5) 
  (hbikes : bikes = 1/10) (hscooters : scooters = 1/10) : 
  1 - (bus + auto + bikes + scooters) = 1/5 :=
by 
  rw [hbus, hauto, hbikes, hscooters]
  sorry

end fraction_of_students_who_walk_home_l642_642692


namespace find_cost_price_l642_642039

variable (CP : ℝ) -- cost price
variable (SP_loss SP_gain : ℝ) -- selling prices

-- Conditions
def loss_condition := SP_loss = 0.9 * CP
def gain_condition := SP_gain = 1.04 * CP
def difference_condition := SP_gain - SP_loss = 190

-- Theorem to prove
theorem find_cost_price (h_loss : loss_condition CP SP_loss)
                        (h_gain : gain_condition CP SP_gain)
                        (h_diff : difference_condition SP_loss SP_gain) :
  CP = 1357.14 := 
sorry

end find_cost_price_l642_642039


namespace dot_product_computation_l642_642790

variables (a b : EuclideanSpace ℝ 2)

def angle_between (a b : EuclideanSpace ℝ 2) : ℝ := real.arccos ((a.dot_product b) / (∥a∥ * ∥b∥))

def magnitude (v : EuclideanSpace ℝ 2) : ℝ := ∥v∥

theorem dot_product_computation
  (a b : EuclideanSpace ℝ 2)
  (h1 : angle_between a b = real.pi / 3)
  (h2 : magnitude a = 2)
  (h3 : magnitude b = 1) :
  a.dot_product (a + 2 • b) = 6 :=
by
  sorry

end dot_product_computation_l642_642790


namespace paper_boat_travel_time_l642_642488

theorem paper_boat_travel_time :
  ∀ (length_of_embankment : ℝ) (length_of_motorboat : ℝ)
    (time_downstream : ℝ) (time_upstream : ℝ) (v_boat : ℝ) (v_current : ℝ),
  length_of_embankment = 50 →
  length_of_motorboat = 10 →
  time_downstream = 5 →
  time_upstream = 4 →
  v_boat + v_current = length_of_embankment / time_downstream →
  v_boat - v_current = length_of_embankment / time_upstream →
  let speed_paper_boat := v_current in
  let travel_time := length_of_embankment / speed_paper_boat in
  travel_time = 40 :=
by
  intros length_of_embankment length_of_motorboat time_downstream time_upstream v_boat v_current
  intros h_length_emb h_length_motor t_down t_up h_v_boat_plus_current h_v_boat_minus_current
  let speed_paper_boat := v_current
  let travel_time := length_of_embankment / speed_paper_boat
  sorry

end paper_boat_travel_time_l642_642488


namespace intersection_of_M_and_N_l642_642904

-- Define the sets based on the given conditions
def M := {x : ℝ | 4 ≤ 2^(x) ∧ 2^(x) ≤ 16}
def N := {x : ℝ | x * (x - 3) < 0}

-- The proof problem statement
theorem intersection_of_M_and_N : M ∩ N = {x : ℝ | 2 ≤ x ∧ x < 3} :=
by
  sorry

end intersection_of_M_and_N_l642_642904


namespace number_of_family_members_l642_642044

-- Define the number of legs for each type of animal.
def bird_legs : ℕ := 2
def dog_legs : ℕ := 4
def cat_legs : ℕ := 4

-- Define the number of animals.
def birds : ℕ := 4
def dogs : ℕ := 3
def cats : ℕ := 18

-- Define the total number of legs of all animals.
def total_animal_feet : ℕ := birds * bird_legs + dogs * dog_legs + cats * cat_legs

-- Define the total number of heads of all animals.
def total_animal_heads : ℕ := birds + dogs + cats

-- Main theorem: If the total number of feet in the house is 74 more than the total number of heads, find the number of family members.
theorem number_of_family_members (F : ℕ) (h : total_animal_feet + 2 * F = total_animal_heads + F + 74) : F = 7 :=
by
  sorry

end number_of_family_members_l642_642044


namespace sum_of_coordinates_of_D_l642_642927

/--
Given points A = (4,8), B = (2,4), C = (6,6), and D = (a,b) in the first quadrant, if the quadrilateral formed by joining the midpoints of the segments AB, BC, CD, and DA is a square with sides inclined at 45 degrees to the x-axis, then the sum of the coordinates of point D is 6.
-/
theorem sum_of_coordinates_of_D 
  (a b : ℝ)
  (h_quadrilateral : ∃ A B C D : Prod ℝ ℝ, 
    A = (4, 8) ∧ B = (2, 4) ∧ C = (6, 6) ∧ D = (a, b) ∧ 
    ∃ M1 M2 M3 M4 : Prod ℝ ℝ,
    M1 = ((4 + 2) / 2, (8 + 4) / 2) ∧ M2 = ((2 + 6) / 2, (4 + 6) / 2) ∧ 
    M3 = (M2.1 + 1, M2.2 - 1) ∧ M4 = (M3.1 + 1, M3.2 + 1) ∧ 
    M3 = ((a + 6) / 2, (b + 6) / 2) ∧ M4 = ((a + 4) / 2, (b + 8) / 2)
  ) : 
  a + b = 6 := sorry

end sum_of_coordinates_of_D_l642_642927


namespace find_a_l642_642835

theorem find_a (a : ℝ) :
  (∀ x y : ℝ, ax + 2y + 1 = 0 → x + y - 2 = 0 → (a / -2) * (-1) = -1) → a = -2 :=
by
  intro h
  sorry

end find_a_l642_642835


namespace common_root_of_polynomials_l642_642557

theorem common_root_of_polynomials :
  let f := (2 : ℚ) * X^3 - (5 : ℚ) * X^2 + (6 : ℚ) * X - 2,
      g := (6 : ℚ) * X^3 - (3 : ℚ) * X^2 - (2 : ℚ) * X + 1,
      common_root := 1 / 2 in
  is_root f common_root ∧ is_root g common_root :=
by {
  -- Add your proof here
  sorry
}

end common_root_of_polynomials_l642_642557


namespace simplify_sqrt_288_add_2_l642_642939

noncomputable def problem_statement : Prop :=
  sqrt 288 + 2 = 12 * sqrt 2 + 2

theorem simplify_sqrt_288_add_2 : problem_statement := 
  sorry

end simplify_sqrt_288_add_2_l642_642939


namespace cos_150_eq_negative_cos_30_l642_642149

theorem cos_150_eq_negative_cos_30 :
  Real.cos (150 * Real.pi / 180) = - (Real.cos (30 * Real.pi / 180)) :=
by sorry

end cos_150_eq_negative_cos_30_l642_642149


namespace overall_average_speed_l642_642907

-- Define the conditions for Mark's travel
def time_cycling : ℝ := 1
def speed_cycling : ℝ := 20
def time_walking : ℝ := 2
def speed_walking : ℝ := 3

-- Define the total distance and total time
def total_distance : ℝ :=
  (time_cycling * speed_cycling) + (time_walking * speed_walking)

def total_time : ℝ :=
  time_cycling + time_walking

-- Define the proved statement for the average speed
theorem overall_average_speed : total_distance / total_time = 8.67 :=
by
  sorry

end overall_average_speed_l642_642907


namespace cos_150_eq_neg_sqrt3_div_2_l642_642168

theorem cos_150_eq_neg_sqrt3_div_2 : Real.cos (150 * Real.pi / 180) = - (Real.sqrt 3 / 2) :=
by
  sorry

end cos_150_eq_neg_sqrt3_div_2_l642_642168


namespace sin_x_div_2021pi_solution_count_l642_642573

theorem sin_x_div_2021pi_solution_count (a : ℝ) (h_eq : a = 2021 * Real.pi) :
  ∃ n : ℕ, (∀ x : ℝ, -a ≤ x ∧ x ≤ a → sin x = x / a → True) ∧ n = 4043 := sorry

end sin_x_div_2021pi_solution_count_l642_642573


namespace triple_factorial_has_at_least_1000_digits_triple_factorial_trailing_zeros_l642_642931

open Nat

def three_factorial : ℕ := 3!
def three_double_factorial : ℕ := three_factorial!
def three_triple_factorial : ℕ := three_double_factorial!

theorem triple_factorial_has_at_least_1000_digits (n : ℕ) (h1 : 3! = n) (h2 : n! = 720) : log 10 (720!) > 999 := by
  sorry

theorem triple_factorial_trailing_zeros (n : ℕ) (h1 : 3! = n) (h2 : n! = 720) : Nat.factorial_trailing_zeros 720 = 178 := by
  sorry

end triple_factorial_has_at_least_1000_digits_triple_factorial_trailing_zeros_l642_642931


namespace abcd_solution_l642_642373

-- Define the problem statement
theorem abcd_solution (a b c d : ℤ) (h1 : a + c = -2) (h2 : a * c + b + d = 3) (h3 : a * d + b * c = 4) (h4 : b * d = -10) : 
  a + b + c + d = 1 := by 
  sorry

end abcd_solution_l642_642373


namespace billiard_ball_weight_l642_642002

theorem billiard_ball_weight (w_box w_box_with_balls : ℝ) (h_w_box : w_box = 0.5) 
(h_w_box_with_balls : w_box_with_balls = 1.82) : 
    let total_weight_balls := w_box_with_balls - w_box;
    let weight_one_ball := total_weight_balls / 6;
    weight_one_ball = 0.22 :=
by
  sorry

end billiard_ball_weight_l642_642002


namespace normal_distribution_95_conf_interval_l642_642973

noncomputable def normalCDF95 : ℝ := 1.96

theorem normal_distribution_95_conf_interval :
  ∀ (X : ℝ), 
    (normalPDF X 16 2) →
    ∀ x : set ℝ, x = { y : ℝ | 12.08 ≤ y ∧ y ≤ 19.92 } →
    ∃ y : ℝ, μ = 16 ∧ σ = 2 → normPis X [12.08, 19.92] = 0.95 := 
by
  sorry

end normal_distribution_95_conf_interval_l642_642973


namespace jack_piggy_bank_l642_642876

variable (initial_amount : ℕ) (weekly_allowance : ℕ) (weeks : ℕ)

-- Conditions
def initial_amount := 43
def weekly_allowance := 10
def weeks := 8

-- Weekly savings calculation: Jack saves half of his weekly allowance
def weekly_savings := weekly_allowance / 2

-- Total savings over the given period
def total_savings := weekly_savings * weeks

-- Final amount in the piggy bank after the given period
def final_amount := initial_amount + total_savings

-- Theorem to prove: Final amount in the piggy bank after 8 weeks is $83.00
theorem jack_piggy_bank : final_amount = 83 := by
  sorry

end jack_piggy_bank_l642_642876


namespace total_surface_area_l642_642954

theorem total_surface_area (r h : ℝ) (pi : ℝ) (area_base : ℝ) (curved_area_hemisphere : ℝ) (lateral_area_cylinder : ℝ) :
  (pi * r^2 = 144 * pi) ∧ (h = 10) ∧ (curved_area_hemisphere = 2 * pi * r^2) ∧ (lateral_area_cylinder = 2 * pi * r * h) →
  (curved_area_hemisphere + lateral_area_cylinder + area_base = 672 * pi) :=
by
  sorry

end total_surface_area_l642_642954


namespace differential_equation_for_lines_one_unit_from_origin_l642_642376

-- Define the problem conditions
theorem differential_equation_for_lines_one_unit_from_origin
  (α : ℝ) (x y : ℝ) (h : x * cos α + y * sin α = 1) :
  ∃ y', y = x * y' + sqrt (1 + y' ^ 2) :=
by
  sorry

end differential_equation_for_lines_one_unit_from_origin_l642_642376


namespace box_surface_area_l642_642589

variables (a b c : ℝ)

noncomputable def sum_edges : ℝ := 4 * (a + b + c)
noncomputable def diagonal_length : ℝ := Real.sqrt (a^2 + b^2 + c^2)
noncomputable def surface_area : ℝ := 2 * (a * b + b * c + c * a)

/- The problem states that the sum of the lengths of the edges and the diagonal length gives us these values. -/
theorem box_surface_area (h1 : sum_edges a b c = 168) (h2 : diagonal_length a b c = 25) : surface_area a b c = 1139 :=
sorry

end box_surface_area_l642_642589


namespace rectangle_area_l642_642857

noncomputable def area_of_rectangle (z : ℂ) : ℝ :=
  complex.abs z ^ 5

theorem rectangle_area (z : ℂ) (h : ∃ a b c d : ℂ, set_of (λ w, w = z ∨ w = z ^ 2 + z ∨ w = z ^ 3 + z ^ 2 ∨ w = z ^ 4)
  = set_of (λ w, w = a ∨ w = b ∨ w = c ∨ w = d) ∧
  (a - b) * complex.conj(a - b) + (c - d) * complex.conj(c - d) = (b - c) * complex.conj(b - c) + (a - d) * complex.conj(a - d)) :
  ∃ A : ℝ, A = area_of_rectangle z := 
sorry

end rectangle_area_l642_642857


namespace smallest_integer_greater_than_sqrt3_plus1_pow2m_divisible_by_2mplus1_l642_642914

/- Define the conditions -/
def I (m : ℕ) : ℝ := (sqrt 3 + 1)^(2 * m) + (sqrt 3 - 1)^(2 * m)

/- Statement of the proof problem -/
theorem smallest_integer_greater_than_sqrt3_plus1_pow2m_divisible_by_2mplus1 
(m : ℕ) (h : 0 < m) :
  ∃ k : ℕ, k > (sqrt 3 + 1)^(2 * m) ∧ (k % 2^(m + 1) = 0) :=
sorry

end smallest_integer_greater_than_sqrt3_plus1_pow2m_divisible_by_2mplus1_l642_642914


namespace cos_150_degree_l642_642184

theorem cos_150_degree : Real.cos (150 * Real.pi / 180) = -Real.sqrt 3 / 2 :=
by
  sorry

end cos_150_degree_l642_642184


namespace cosine_150_eq_neg_sqrt3_div_2_l642_642241

theorem cosine_150_eq_neg_sqrt3_div_2 :
  (∀ θ : ℝ, θ = real.pi / 6) →
  (∀ θ : ℝ, cos (real.pi - θ) = -cos θ) →
  cos (5 * real.pi / 6) = -real.sqrt 3 / 2 :=
by
  intros hθ hcos_identity
  sorry

end cosine_150_eq_neg_sqrt3_div_2_l642_642241


namespace positive_integer_divisibility_l642_642759

theorem positive_integer_divisibility (n : ℕ) (h_pos : 0 < n) (h_div : (n^2 + 1) ∣ (n + 1)) : n = 1 :=
begin
  sorry
end

end positive_integer_divisibility_l642_642759


namespace tangent_line_equation_l642_642957

noncomputable def f (x : ℝ) : ℝ := x^3 - 3*x^2 + x

noncomputable def f' (x : ℝ) : ℝ := 3*x^2 - 6*x + 1

theorem tangent_line_equation :
  let x₁ := 1
      y₁ := f x₁
      m := f' x₁
  in 2*x₁ + y₁ - 1 = 0 :=
by
  sorry

end tangent_line_equation_l642_642957


namespace cos_150_eq_neg_sqrt3_div_2_l642_642133

theorem cos_150_eq_neg_sqrt3_div_2 :
  ∃ θ : ℝ, θ = 150 ∧
           0 < θ ∧ θ < 180 ∧
           ∃ φ : ℝ, φ = 180 - θ ∧
           cos φ = Real.sqrt 3 / 2 ∧
           cos θ = - cos φ :=
  sorry

end cos_150_eq_neg_sqrt3_div_2_l642_642133


namespace cos_150_eq_negative_cos_30_l642_642143

theorem cos_150_eq_negative_cos_30 :
  Real.cos (150 * Real.pi / 180) = - (Real.cos (30 * Real.pi / 180)) :=
by sorry

end cos_150_eq_negative_cos_30_l642_642143


namespace least_faces_l642_642014

theorem least_faces (a b : ℕ) (h1 : a ≥ 8) (h2 : b ≥ 8)
  (h3 : 8 * 5 / 6 = 10)
  (h4 : (a * b) * (1 / 10) = 0.1 * (a * b)) :
  a + b = 20 :=
sorry

end least_faces_l642_642014


namespace max_value_expression_l642_642449

theorem max_value_expression (a b c : ℝ) (h : a^2 + b^2 + c^2 = 9) : 
  (a - b)^2 + (b - c)^2 + (c - a)^2 ≤ 27 := 
by sorry

end max_value_expression_l642_642449


namespace relationship_among_a_b_c_l642_642396

variable a b c : ℝ
variable a_def : a = 1.9^0.4
variable b_def : b = Real.logBase 0.4 1.9
variable c_def : c = 0.4^1.9

theorem relationship_among_a_b_c : a > c ∧ c > b :=
by
  rw [a_def, b_def, c_def]
  sorry

end relationship_among_a_b_c_l642_642396


namespace coffee_temp_proof_l642_642974
noncomputable def coffeeTemp (A T : ℝ) : ℝ := 120 * 2^(-A * T) + 60

theorem coffee_temp_proof (A : ℝ) :
    coffeeTemp A 30 = 60.00000011175871 →
    coffeeTemp A 2 = 120 * 2^(-A * 2) + 60 :=
by
  assume h : coffeeTemp A 30 = 60.00000011175871
  sorry

end coffee_temp_proof_l642_642974


namespace cos_150_eq_neg_half_l642_642271

theorem cos_150_eq_neg_half : Real.cos (150 * Real.pi / 180) = -1 / 2 := 
by
  sorry

end cos_150_eq_neg_half_l642_642271


namespace repeating_decimal_fraction_l642_642352

noncomputable def repeating_decimal := 7 + ((789 : ℚ) / (10^4 - 1))

theorem repeating_decimal_fraction :
  repeating_decimal = (365 : ℚ) / 85 :=
by
  sorry

end repeating_decimal_fraction_l642_642352


namespace trig_equation_solution_l642_642622

theorem trig_equation_solution (x : ℝ) (k : ℤ) :
  8.483 * tan x - sin (2 * x) - cos (2 * x) + 2 * (2 * cos x - (1 / cos x)) = 0 ↔
  ∃ (k : ℤ), x = (π / 4) * (2 * k + 1) :=
by
  sorry

end trig_equation_solution_l642_642622


namespace max_and_min_values_part_range_of_a_for_monotonicity_l642_642412

-- Function f(x) definition
def f (a x : ℝ) : ℝ := -x^2 + a * x

-- Problem Part 1: Proving maximum and minimum values
theorem max_and_min_values_part (a : ℝ) (h : a = 3) :
  ∃ (max_x : ℝ) (max_value : ℝ) (min_x : ℝ) (min_value : ℝ),
    (max_x ∈ set.Icc (1/2) 2) ∧ (min_x ∈ set.Icc (1/2) 2) ∧
    (f a max_x = max_value) ∧ (f a min_x = min_value) ∧
    (∀ x ∈ set.Icc (1/2) 2, f a x ≤ max_value) ∧
    (∀ x ∈ set.Icc (1/2) 2, f a x ≥ min_value) ∧
    max_value = 9/4 ∧ min_value = 5/4 := 
begin
  sorry,
end

-- Problem Part 2: Proving range of a for monotonicity
theorem range_of_a_for_monotonicity :
  ∀ a : ℝ, (∀ x1 x2, (x1 ∈ set.Ioc (1/2) 2) → (x2 ∈ set.Ioc (1/2) 2) → x1 < x2 → f a x1 ≤ f a x2) ∨
            (∀ x1 x2, (x1 ∈ set.Ioc (1/2) 2) → (x2 ∈ set.Ioc (1/2) 2) → x1 < x2 → f a x1 ≥ f a x2) ↔
            a ≤ 1 ∨ a ≥ 4 :=
begin
  sorry,
end

end max_and_min_values_part_range_of_a_for_monotonicity_l642_642412


namespace number_of_sets_A_l642_642808

def M : Set ℤ := { m : ℤ | ∃ x y : ℤ, x * y = -36 ∧ m = -(x + y) }

theorem number_of_sets_A :
  let S := { A : Set ℤ | A ⊆ M ∧ (∀ a ∈ A, -a ∈ A) ∧ A ≠ ∅ } in
  S.card = 31 := sorry

end number_of_sets_A_l642_642808


namespace intersection_points_of_C_and_l_when_a_is_neg_1_max_distance_from_C_to_l_l642_642467

def C_ : set (ℝ × ℝ) := {p | ∃ θ : ℝ, p = (3 * Real.cos θ, Real.sin θ)}
def l (a : ℝ) : set (ℝ × ℝ) := {p | ∃ t : ℝ, p = (a + 4 * t, 1 - t)}

theorem intersection_points_of_C_and_l_when_a_is_neg_1 :
  ∀ x y, (x, y) ∈ C_ ∧ (x, y) ∈ l (-1) → (x = 3 ∧ y = 0) ∨ (x = -21 / 25 ∧ y = 24 / 25) :=
begin
  sorry
end

theorem max_distance_from_C_to_l :
  ∀ a, ∃ θ, (a = -16 ∨ a = 8) ∧
    (∥ 3 * Real.cos θ + 4 * Real.sin θ - a - 4 ∥ = 17) :=
begin
  sorry,
end

end intersection_points_of_C_and_l_when_a_is_neg_1_max_distance_from_C_to_l_l642_642467


namespace complex_number_problem_l642_642798

def z : ℂ := 1 - 2 * complex.i

theorem complex_number_problem : (1 / (complex.conj z)) = (1 / 5 - 2 / 5 * complex.i) :=
by sorry

end complex_number_problem_l642_642798


namespace trigonometric_expression_eval_l642_642393

-- Conditions
variable (α : Real) (h1 : ∃ x : Real, 3 * x^2 - x - 2 = 0 ∧ x = Real.cos α) (h2 : α > π ∧ α < 3 * π / 2)

-- Question and expected answer
theorem trigonometric_expression_eval :
  (Real.sin (-α + 3 * π / 2) * Real.cos (3 * π / 2 + α) * Real.tan (π - α)^2) /
  (Real.cos (π / 2 + α) * Real.sin (π / 2 - α)) = 5 / 4 := sorry

end trigonometric_expression_eval_l642_642393


namespace cos_150_eq_neg_half_l642_642267

theorem cos_150_eq_neg_half : Real.cos (150 * Real.pi / 180) = -1 / 2 := 
by
  sorry

end cos_150_eq_neg_half_l642_642267


namespace distance_between_intersection_points_l642_642001

noncomputable def cube_vertices : List (ℝ × ℝ × ℝ) :=
  [(0, 0, 0), (0, 0, 6), (0, 6, 0), (0, 6, 6), 
   (6, 0, 0), (6, 0, 6), (6, 6, 0), (6, 6, 6)]

def P : ℝ × ℝ × ℝ := (0, 3, 0)
def Q : ℝ × ℝ × ℝ := (2, 0, 0)
def R : ℝ × ℝ × ℝ := (2, 6, 6)

theorem distance_between_intersection_points :
  let normal_vector := (P.1 - Q.1, P.2 - Q.2, P.3 - Q.3) ×ᵥ (P.1 - R.1, P.2 - R.2, P.3 - R.3)
  let scaled_normal_vector := (3, 2, -1) -- scaled manually based on solution information
  let plane_equation := (3, 2, -1)
  let S := (4, 0, 6)
  let T := (0, 6, 6)
  dist S T = 2 * Real.sqrt 13 := 
sorry

end distance_between_intersection_points_l642_642001


namespace cos_150_eq_neg_sqrt3_div_2_l642_642293

theorem cos_150_eq_neg_sqrt3_div_2 :
  (cos (150 * real.pi / 180)) = - (sqrt 3 / 2) := sorry

end cos_150_eq_neg_sqrt3_div_2_l642_642293


namespace inradius_scalene_triangle_l642_642516

theorem inradius_scalene_triangle (ABC : Triangle) (A B C I : Point)
  (h1 : ABC.scalene)
  (h2 : AB ≠ AC)
  (h3 : dist B C = 40)
  (h4 : dist I C = 24)
  (h5 : ABC.incenter = I)
  (h6 : dist A B > dist A C) :
  inradius ABC = 4 * Real.sqrt 11 :=
sorry

end inradius_scalene_triangle_l642_642516


namespace split_trout_equally_l642_642915

-- Definitions for conditions
def Total_trout : ℕ := 18
def People : ℕ := 2

-- Statement we need to prove
theorem split_trout_equally 
(H1 : Total_trout = 18)
(H2 : People = 2) : 
  (Total_trout / People = 9) :=
by
  sorry

end split_trout_equally_l642_642915


namespace train_length_l642_642676

theorem train_length (s : ℝ) (t : ℝ) (h_s : s = 60) (h_t : t = 10) :
  ∃ L : ℝ, L = 166.7 := by
  sorry

end train_length_l642_642676


namespace cos_150_degree_l642_642177

theorem cos_150_degree : Real.cos (150 * Real.pi / 180) = -Real.sqrt 3 / 2 :=
by
  sorry

end cos_150_degree_l642_642177


namespace length_of_chord_l642_642071

open Real

/-- Given a circle with radius 4 and a distance of 3 from center to a chord, the length of the chord is 2√7 -/
theorem length_of_chord (r : ℝ) (d : ℝ) (a b : ℝ) (h₁ : r = 4) (h₂ : d = 3) (r_ne_zero : r ≠ 0)
  (hx : a^2 = r^2 - d^2) (h_eq_chord : b = 2 * a) : b = 2 * sqrt 7 :=
by 
  rw [hx, h₁, h₂] at h_eq_chord;
  sorry

end length_of_chord_l642_642071


namespace average_price_paid_per_book_l642_642547

theorem average_price_paid_per_book
  (books1 books2 : ℕ) (cost1 cost2 : ℕ)
  (total_books total_cost : ℕ) (avg_price : ℝ)
  (h_books : books1 = 27) (h_books2 : books2 = 20)
  (h_cost1 : cost1 = 581) (h_cost2 : cost2 = 594)
  (h_total_books : total_books = books1 + books2)
  (h_total_cost : total_cost = cost1 + cost2)
  (h_avg_price : avg_price = total_cost / total_books) :
  avg_price ≈ 25 := sorry

end average_price_paid_per_book_l642_642547


namespace percentage_profit_without_discount_l642_642038

variable (CP : ℝ) (discountRate profitRate noDiscountProfitRate : ℝ)

theorem percentage_profit_without_discount 
  (hCP : CP = 100)
  (hDiscount : discountRate = 0.04)
  (hProfit : profitRate = 0.26)
  (hNoDiscountProfit : noDiscountProfitRate = 0.3125) :
  let SP := CP * (1 + profitRate)
  let MP := SP / (1 - discountRate)
  noDiscountProfitRate = (MP - CP) / CP :=
by
  sorry

end percentage_profit_without_discount_l642_642038


namespace find_x_l642_642112

def magic_constant (a b c d e f g h i : ℤ) : Prop :=
  a + b + c = d + e + f ∧ d + e + f = g + h + i ∧
  a + d + g = b + e + h ∧ b + e + h = c + f + i ∧
  a + e + i = c + e + g

def given_magic_square (x : ℤ) : Prop :=
  magic_constant (4017) (2012) (0) 
                 (4015) (x - 2003) (11) 
                 (2014) (9) (x)

theorem find_x (x : ℤ) (h : given_magic_square x) : x = 4003 :=
by {
  sorry
}

end find_x_l642_642112


namespace max_value_ad_bc_l642_642765

theorem max_value_ad_bc (a b c d : ℤ) (h₁ : a ∈ ({-1, 1, 2} : Set ℤ))
                          (h₂ : b ∈ ({-1, 1, 2} : Set ℤ))
                          (h₃ : c ∈ ({-1, 1, 2} : Set ℤ))
                          (h₄ : d ∈ ({-1, 1, 2} : Set ℤ)) :
  ad - bc ≤ 6 :=
by sorry

end max_value_ad_bc_l642_642765


namespace john_flips_9_coins_l642_642827

noncomputable def coin_flip_probability : ℕ → ℚ
| 0     := 0
| n + 1 := if (n + 1) % 2 = 1 then
              (∑ k in finset.range ((n + 1) / 2 + 1), nat.choose (n + 1) k) / 2^(n + 1)
           else
              0

theorem john_flips_9_coins :
  coin_flip_probability 9 = 1 / 2 :=
by
  sorry

end john_flips_9_coins_l642_642827


namespace valerie_needs_small_bulbs_l642_642019

def cost_of_small_bulb := 8
def cost_of_large_bulb := 12
def total_money := 60
def money_left := 24
def num_large_bulbs := 1

theorem valerie_needs_small_bulbs : 
    ∃ (S : ℕ), 
    let spent := total_money - money_left in
    let amount_spent_on_large := num_large_bulbs * cost_of_large_bulb in
    let amount_spent_on_small := spent - amount_spent_on_large in
    S * cost_of_small_bulb = amount_spent_on_small :=
begin
    use 3,
    sorry
end

end valerie_needs_small_bulbs_l642_642019


namespace polynomial_evaluation_l642_642513

theorem polynomial_evaluation :
  ∃ (R : ℕ → ℕ → ℕ), (∀ i, 0 ≤ R i (i + 1) ∧ R i (i + 1) < 4) ∧ (∃ n, R 2 2 =  26 + 19 * 2 ∧ R 3 3 = 683) :=
by {
  -- placeholder for the proof
  sorry
}

end polynomial_evaluation_l642_642513


namespace min_moves_to_unify_numbers_l642_642690

theorem min_moves_to_unify_numbers (n : ℕ) :
  let divisors := (n + 1) * (n + 1),
      move_count := nat.gcd.default
  in ∃ moves : ℕ, (∀ (a b : ℕ), a ≠ b → a ≤ divisors ∧ b ≤ divisors → move_count) ∧ moves = n^2 + n :=
sorry

end min_moves_to_unify_numbers_l642_642690


namespace paper_boat_travel_time_l642_642484

-- Given conditions
def embankment_length : ℝ := 50
def boat_length : ℝ := 10
def downstream_time : ℝ := 5
def upstream_time : ℝ := 4

-- Derived conditions from the given problem
def downstream_speed := embankment_length / downstream_time
def upstream_speed := embankment_length / upstream_time

-- Prove that the paper boat's travel time is 40 seconds
theorem paper_boat_travel_time :
  let v_boat := (downstream_speed + upstream_speed) / 2 in
  let v_current := (downstream_speed - upstream_speed) / 2 in
  let travel_time := embankment_length / v_current in
  travel_time = 40 := 
  sorry

end paper_boat_travel_time_l642_642484


namespace min_edges_in_graph_l642_642361

theorem min_edges_in_graph (n : ℕ) (h : n ≥ 5) :
  ∃ G : SimpleGraph (Fin (2 * n)), 
    (∀ {x y z : Fin (2 * n)}, ¬ (G.Adj x y ∧ G.Adj y z ∧ G.Adj z x)) ∧ 
    (¬ ∃ H : SimpleGraph (Fin n), ∀ x y : Fin n, H.Adj x y) ∧
    G.edgeFinSet.card = n + 5 :=
sorry

end min_edges_in_graph_l642_642361


namespace median_salary_is_28000_l642_642117

-- Define salary data as a list of pairs (number of employees, salary)
def salary_data : List (Nat × Nat) := [
  (1, 145000),
  (7, 95000),
  (15, 78000),
  (12, 53000),
  (36, 28000)
]

-- Function to calculate median from a list of salaries
def median_salary (data : List (Nat × Nat)) : Nat :=
  let all_salaries := data.bind (λ (p : Nat × Nat), List.repeat p.2 p.1)
  let sorted_salaries := all_salaries.qsort (· ≤ ·)
  sorted_salaries.get! (sorted_salaries.length / 2)

-- The theorem stating the median salary
theorem median_salary_is_28000 : median_salary salary_data = 28000 := by
  sorry

end median_salary_is_28000_l642_642117


namespace cricket_bat_profit_l642_642079

variable (SP : ℝ) (PPercent : ℝ)
variable (CP : ℝ) (P : ℝ)

-- Given conditions
def selling_price (SP: ℝ) : Prop := SP = 850
def profit_percentage (PPercent : ℝ) : Prop := PPercent = 42.857142857142854 / 100

-- Define the cost price (CP)
def cost_price (SP PPercent : ℝ) : ℝ := SP / (1 + PPercent)

-- Define the profit (P) calculation
def calculated_profit (SP CP : ℝ) : ℝ := SP - CP

-- Prove that the calculated profit matches the given profit
theorem cricket_bat_profit 
  (hSP : selling_price SP) 
  (hPPercent : profit_percentage PPercent) : 
  P = 255 := by
  have h1 : CP = cost_price SP PPercent, from sorry,
  have h2 : P = calculated_profit SP CP, from sorry,
  sorry

end cricket_bat_profit_l642_642079


namespace magnitude_c_l642_642762

open Real

variables {a b c : EuclideanSpace ℝ (Fin 3)}

-- Condition 1: |a| = 2
axiom norm_a : ∥a∥ = 2

-- Condition 2: |b| = 3
axiom norm_b : ∥b∥ = 3

-- Condition 3: The angle between a and b is 2π/3
axiom angle_a_b : inner a b = ∥a∥ * ∥b∥ * cos (2 * π / 3)

-- Condition 4: a + b + c = 0
axiom zero_vector : a + b + c = 0

theorem magnitude_c : ∥c∥ = sqrt 7 := 
sorry

end magnitude_c_l642_642762


namespace determine_x_l642_642754

theorem determine_x : ∃ (x : ℕ), 
  (3 * x > 91 ∧ x < 120 ∧ 4 * x > 37 ∧ 2 * x ≥ 21 ∧ ¬(x > 7) ∧ 
   (3 * x > 91 ∨ x < 120 ∨ 4 * x > 37 ∨ 2 * x ≥ 21 ∨ x > 7)) ∨
  (3 * x > 91 ∧ x < 120 ∧ 4 * x > 37 ∧ ¬(2 * x ≥ 21) ∧ x > 7 ∧ 
   (3 * x > 91 ∨ x < 120 ∨ 4 * x > 37 ∨ 2 * x ≥ 21 ∨ x > 7)) ∨
  (3 * x > 91 ∧ x < 120 ∧ ¬(4 * x > 37) ∧ 2 * x ≥ 21 ∧ x > 7 ∧ 
   (3 * x > 91 ∨ x < 120 ∨ 4 * x > 37 ∨ 2 * x ≥ 21 ∨ x > 7)) ∨
  (3 * x > 91 ∧ ¬(x < 120) ∧ 4 * x > 37 ∧ 2 * x ≥ 21 ∧ x > 7 ∧ 
   (3 * x > 91 ∨ x < 120 ∨ 4 * x > 37 ∨ 2 * x ≥ 21 ∨ x > 7)) ∨
  (¬(3 * x > 91) ∧ x < 120 ∧ 4 * x > 37 ∧ 2 * x ≥ 21 ∧ x > 7 ∧ 
   (3 * x > 91 ∨ x < 120 ∨ 4 * x > 37 ∨ 2 * x ≥ 21 ∨ x > 7)) ∧
  x = 10 :=
sorry

end determine_x_l642_642754


namespace hank_route_length_l642_642918

-- Definitions of the problem conditions.
def route_length (d : ℝ) : Prop :=
  let travel_time_difference := (d / 70 - d / 75) in
  travel_time_difference = 1 / 30

-- The statement of the theorem to prove that the length of Hank's route is 35 km given the conditions.
theorem hank_route_length : ∃ d : ℝ, route_length d ∧ d = 35 :=
by 
  sorry

end hank_route_length_l642_642918


namespace airplane_altitude_l642_642680

theorem airplane_altitude (A B : Type) (dist_AB : ℝ) (angle_Alice elevation_Alice angle_Bob elevation_Bob : ℝ) (h : ℝ) :
  dist_AB = 12 ∧
  angle_Alice = 0 ∧
  elevation_Alice = 45 ∧
  angle_Bob = 45 ∧
  elevation_Bob = 30 →
  h = 6 * real.sqrt 2 :=
by 
  intro h₀
  have dist_AB_eq := h₀.1
  have angle_Alice_eq := h₀.2.1
  have elevation_Alice_eq := h₀.2.2.1
  have angle_Bob_eq := h₀.2.2.2.1
  have elevation_Bob_eq := h₀.2.2.2.2
  sorry

end airplane_altitude_l642_642680


namespace cos_150_eq_neg_sqrt3_div_2_l642_642306

open Real

theorem cos_150_eq_neg_sqrt3_div_2 :
  cos (150 * pi / 180) = - (sqrt 3 / 2) := 
sorry

end cos_150_eq_neg_sqrt3_div_2_l642_642306


namespace find_enclosed_area_l642_642645

/-- Given a circle of radius 4 centered at O, and square OABC with side length √3.
The sides AB and BC are extended to form equilateral triangles ABD and BCE with D and E 
lying on the circle. We want to find the area of the region enclosed by segments BD, BE, 
and the minor arc DE. -/
noncomputable def enclosed_area : ℝ :=
  let R := 4 in
  let s := real.sqrt 3 in
  let θ := real.arccos (31 / 32) in
  (θ / (real.pi / 180)) / 360 * real.pi * R^2 - (3 * real.sqrt 3 / 4)

theorem find_enclosed_area :
  enclosed_area = (real.arccos (31 / 32) / (real.pi / 180)) / 360 * real.pi * 16 - (3 * real.sqrt 3 / 4) :=
sorry

end find_enclosed_area_l642_642645


namespace cos_150_eq_neg_sqrt3_div_2_l642_642160

theorem cos_150_eq_neg_sqrt3_div_2 : Real.cos (150 * Real.pi / 180) = - (Real.sqrt 3 / 2) :=
by
  sorry

end cos_150_eq_neg_sqrt3_div_2_l642_642160


namespace trajectory_eq_C_dot_product_range_l642_642905

noncomputable theory

-- Definitions
def vec_a : ℝ × ℝ := (0, 2)
def vec_b : ℝ × ℝ := (1, 0)
def point_A : ℝ × ℝ := (0, -2)
def point_B : ℝ × ℝ := (0, 2)
def point_E : ℝ × ℝ := (1, 0)

-- Equation of the trajectory C of point P
def trajectory_C (x y : ℝ) : Prop := 8 * x^2 + y^2 = 4

-- Prove trajectory_C
theorem trajectory_eq_C : ∀ (x y : ℝ),
  (∃ (λ : ℝ), 2 * x - λ * y - 2 * λ = 0 ∧ 4 * λ * x + y - 2 = 0) →
  trajectory_C x y :=
sorry

-- Dot product range
def vec_EM (M : ℝ × ℝ) : ℝ × ℝ := (M.1 - 1, M.2)
def vec_EN (N : ℝ × ℝ) : ℝ × ℝ := (N.1 - 1, N.2)
def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

-- Prove the range of values for EM · EN
theorem dot_product_range : 
  ∀ (k x1 x2 y1 y2 : ℝ),
  0 ≤ k^2 ∧ k^2 < 8 ∧ y1 = k * (x1 - 1) ∧ y2 = k * (x2 - 1) ∧
  trajectory_C x1 y1 ∧ trajectory_C x2 y2 →
  let EM := vec_EM (x1, y1)
      EN := vec_EN (x2, y2) in
  ∃ (d : ℝ), d = dot_product EM EN ∧ d ∈ set.Ico (1 / 2) (9 / 4) :=
sorry

end trajectory_eq_C_dot_product_range_l642_642905


namespace minimum_value_fraction_l642_642426

theorem minimum_value_fraction (m n : ℝ) (h_m_pos : 0 < m) (h_n_pos : 0 < n) 
  (h_parallel : m / (4 - n) = 1 / 2) : 
  (1 / m + 8 / n) ≥ 9 / 2 :=
by
  sorry

end minimum_value_fraction_l642_642426


namespace construct_rectangle_similar_to_given_l642_642724

noncomputable def rectangle_in_square_similar_to_given_rect (α : ℝ) (sq : set (ℝ × ℝ)) (rec : set (ℝ × ℝ)) : Prop :=
∃ A B C D A₁ B₁ C₁ D₁ : (ℝ × ℝ),
  (A, B, C, D ∈ sq) ∧
  (A₁, B₁, C₁, D₁ ∈ sq) ∧
  is_rectangle A B C D ∧
  is_rectangle A₁ B₁ C₁ D₁ ∧
  similar A B C D rec ∧
  similar A₁ B₁ C₁ D₁ rec ∧
  let diagonals := intersection_of_diagonals sq in
  let lines_through_diagonals := lines_with_angle α diagonals in
  intersects_sides lines_through_diagonals sq A C A₁ C₁ ∧
  parallel_lines_from A A₁ sq B D B₁ D₁

-- Conditions of the problem
variables (α : ℝ) (sq : set (ℝ × ℝ)) (rec : set (ℝ × ℝ))

-- Defining the problem statement in Lean
theorem construct_rectangle_similar_to_given (H1 : 0 < α ∧ α < 90 ∧ is_square sq) :
  rectangle_in_square_similar_to_given_rect α sq rec :=
sorry

end construct_rectangle_similar_to_given_l642_642724


namespace cos_150_deg_l642_642327

-- Define the conditions as variables
def angle1 := 150
def angle2 := 180
def angle3 := 30
def cos30 := Real.sqrt 3 / 2

-- The main statement to be proved
theorem cos_150_deg : Real.cos (150 * Real.pi / 180) = - (Real.sqrt 3 / 2) :=
by
  sorry

end cos_150_deg_l642_642327


namespace hazel_walked_distance_l642_642429

theorem hazel_walked_distance
  (first_hour_distance : ℕ)
  (second_hour_distance : ℕ)
  (h1 : first_hour_distance = 2)
  (h2 : second_hour_distance = 2 * first_hour_distance) :
  (first_hour_distance + second_hour_distance = 6) :=
by {
  sorry
}

end hazel_walked_distance_l642_642429


namespace cos_150_eq_neg_sqrt3_div_2_l642_642257

theorem cos_150_eq_neg_sqrt3_div_2 :
  let theta := 30 * Real.pi / 180 in
  let Q := 150 * Real.pi / 180 in
  cos Q = - (Real.sqrt 3 / 2) := by
    sorry

end cos_150_eq_neg_sqrt3_div_2_l642_642257


namespace cos_150_eq_neg_half_l642_642201

theorem cos_150_eq_neg_half :
  real.cos (150 * real.pi / 180) = -1 / 2 :=
sorry

end cos_150_eq_neg_half_l642_642201


namespace probability_two_random_n_domino_tiles_match_calculate_probability_two_random_n_domino_tiles_match_domino_tile_probability_final_answer_l642_642688

theorem probability_two_random_n_domino_tiles_match :
  ∀ n : ℕ, (0 < n) →
    (∑ i in range(n+1), i) * (n+1) = ((n+1) * (n+1)) * n :=
  sorry

theorem calculate_probability_two_random_n_domino_tiles_match :
  ∀ n : ℕ, (0 < n) →
    (∑ i in range(n+1), i) * (n+1) = 4 * (n+1) * ((n + 3) * (n + 2)) :=
  sorry

theorem domino_tile_probability_final_answer :
  ∀ n : ℕ, (0 < n) → 
    probability_two_random_n_domino_tiles_match n = calculate_probability_two_random_n_domino_tiles_match n := 
  sorry

end probability_two_random_n_domino_tiles_match_calculate_probability_two_random_n_domino_tiles_match_domino_tile_probability_final_answer_l642_642688


namespace trajectory_of_point_C_and_chord_length_l642_642473

-- Define the points A, B, C
def pointA : ℝ × ℝ := (3, 0)
def pointB : ℝ × ℝ := (-1, 0)
variable (x y : ℝ)

-- Define the conditions
def AC := (x - 3, y)
def BC := (x + 1, y)

def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

-- The given condition for the dot product
def condition := dot_product (AC x y) (BC x y) = 5

-- The equation of the trajectory of point C
def trajectory_eq := (x - 1)^2 + y^2 = 9

-- Define the line l
def line_l (x y : ℝ) : Prop := x - y + 3 = 0

-- Find the distance from point P (the center of the circle) to line l
def distance_P_to_l (P : ℝ × ℝ) (x y : ℝ) : ℝ :=
  abs (P.1 * 1 + P.2 * (-1) + 3) / sqrt (1^2 + (-1)^2)

def centerP : ℝ × ℝ := (1, 0)
def radius : ℝ := 3
def d := distance_P_to_l centerP x y

-- The length of the chord |MN|
def chord_length := 2 * sqrt (radius^2 - d^2)

-- The theorem to prove
theorem trajectory_of_point_C_and_chord_length :
  (condition x y) → trajectory_eq x y ∧ (line_l x y → chord_length = 2) :=
by
  sorry

end trajectory_of_point_C_and_chord_length_l642_642473


namespace additional_charge_atlantic_call_l642_642992

-- Variables representing the base rate and per minute charge for both companies
def unitedBaseRate : ℝ := 9.0
def unitedPerMinute : ℝ := 0.25
def atlanticBaseRate : ℝ := 12.0
variable (atlanticPerMinute : ℝ)

-- The time in minutes
def minutes : ℕ := 60

-- Total cost equation for United Telephone
def unitedCost : ℝ := unitedBaseRate + unitedPerMinute * minutes

-- Total cost equation for Atlantic Call
def atlanticCost : ℝ := atlanticBaseRate + atlanticPerMinute * minutes

-- Proof statement
theorem additional_charge_atlantic_call :
  unitedCost = atlanticCost → atlanticPerMinute = 0.2 := by
  sorry

end additional_charge_atlantic_call_l642_642992


namespace difference_between_first_and_third_l642_642627

variable (x : ℕ)

-- Condition 1: The first number is twice the second.
def first_number : ℕ := 2 * x

-- Condition 2: The first number is three times the third.
def third_number : ℕ := first_number x / 3

-- Condition 3: The average of the three numbers is 88.
def average_condition : Prop := (first_number x + x + third_number x) / 3 = 88

-- Prove that the difference between first and third number is 96.
theorem difference_between_first_and_third 
  (h : average_condition x) : first_number x - third_number x = 96 :=
by
  sorry -- Proof omitted

end difference_between_first_and_third_l642_642627


namespace opposite_sides_of_line_l642_642968

theorem opposite_sides_of_line (a : ℝ) (h1 : 0 < a) (h2 : a < 2) : (-a) * (2 - a) < 0 :=
sorry

end opposite_sides_of_line_l642_642968


namespace evaluate_ratio_l642_642742

theorem evaluate_ratio : (2^2003 * 3^2002) / (6^2002) = 2 := 
by {
  sorry
}

end evaluate_ratio_l642_642742


namespace fill_25_cans_in_5_hours_l642_642663

def fill_rate (volume : ℕ) (time : ℕ) : ℕ :=
  volume / time

def total_volume_per_can (fill_fraction : ℚ) (can_capacity : ℕ) : ℕ :=
  (fill_fraction * can_capacity).toNat

def total_volume_filled (num_cans : ℕ) (volume_per_can : ℕ) : ℕ :=
  num_cans * volume_per_can

def time_to_fill_cans (total_volume : ℕ) (fill_rate : ℕ) : ℕ :=
  total_volume / fill_rate

theorem fill_25_cans_in_5_hours :
  let can_capacity := 8
  let num_cans_filled := 20
  let fill_fraction := 3 / 4
  let time_filled := 3
  let cans_to_fill := 25
  let full_can_volume := can_capacity
  let rate := fill_rate (total_volume_filled num_cans_filled (total_volume_per_can fill_fraction can_capacity)) time_filled
  time_to_fill_cans (total_volume_filled cans_to_fill full_can_volume) rate = 5 := by
  sorry

end fill_25_cans_in_5_hours_l642_642663


namespace binary_1101_to_decimal_l642_642725

theorem binary_1101_to_decimal : (1 * 2^3 + 1 * 2^2 + 0 * 2^1 + 1 * 2^0) = 13 := by
  -- To convert a binary number to its decimal equivalent, we multiply each digit by its corresponding power of 2 based on its position and then sum the results.
  sorry

end binary_1101_to_decimal_l642_642725


namespace abs_diff_of_pq_eq_6_and_pq_sum_7_l642_642896

variable (p q : ℝ)

noncomputable def abs_diff (a b : ℝ) := |a - b|

theorem abs_diff_of_pq_eq_6_and_pq_sum_7 (hpq : p * q = 6) (hpq_sum : p + q = 7) : abs_diff p q = 5 :=
by
  sorry

end abs_diff_of_pq_eq_6_and_pq_sum_7_l642_642896


namespace mean_books_read_club_l642_642534

theorem mean_books_read_club :
  let total_books := 2 * 1 + 4 * 2 + 3 * 3 + 5 * 4 + 1 * 5 + 4 * 6
  let total_members := 2 + 4 + 3 + 5 + 1 + 4
  let mean_books := (total_books : ℚ) / total_members
  (Real.toRat (Float.ofRat mean_books.toReal).roundTo 100) = 358 / 100 :=
by {
  sorry
}

end mean_books_read_club_l642_642534


namespace outfit_choices_l642_642437

-- Define the numbers of shirts, pants, and hats.
def num_shirts : ℕ := 6
def num_pants : ℕ := 7
def num_hats : ℕ := 6

-- Define the number of colors and the constraints.
def num_colors : ℕ := 6

-- The total number of outfits without restrictions.
def total_outfits : ℕ := num_shirts * num_pants * num_hats

-- Number of outfits where all items are the same color.
def same_color_outfits : ℕ := num_colors

-- Number of outfits where the shirt and pants are the same color.
def same_shirt_pants_color_outfits : ℕ := num_colors + 1  -- accounting for the extra pair of pants

-- The total number of valid outfits calculated.
def valid_outfits : ℕ :=
  total_outfits - same_color_outfits - same_shirt_pants_color_outfits

-- The theorem statement asserting the correct answer.
theorem outfit_choices : valid_outfits = 239 := by
  sorry

end outfit_choices_l642_642437


namespace cos_150_eq_neg_sqrt3_div_2_l642_642289

theorem cos_150_eq_neg_sqrt3_div_2 :
  (cos (150 * real.pi / 180)) = - (sqrt 3 / 2) := sorry

end cos_150_eq_neg_sqrt3_div_2_l642_642289


namespace nine_div_one_plus_four_div_x_eq_one_l642_642826

theorem nine_div_one_plus_four_div_x_eq_one (x : ℝ) (h : x = 0.5) : 9 / (1 + 4 / x) = 1 := by
  sorry

end nine_div_one_plus_four_div_x_eq_one_l642_642826


namespace cos_150_eq_neg_sqrt3_div_2_l642_642284

theorem cos_150_eq_neg_sqrt3_div_2 :
  (cos (150 * real.pi / 180)) = - (sqrt 3 / 2) := sorry

end cos_150_eq_neg_sqrt3_div_2_l642_642284


namespace cosine_150_eq_neg_sqrt3_div_2_l642_642246

theorem cosine_150_eq_neg_sqrt3_div_2 :
  (∀ θ : ℝ, θ = real.pi / 6) →
  (∀ θ : ℝ, cos (real.pi - θ) = -cos θ) →
  cos (5 * real.pi / 6) = -real.sqrt 3 / 2 :=
by
  intros hθ hcos_identity
  sorry

end cosine_150_eq_neg_sqrt3_div_2_l642_642246


namespace find_y_eq_1_l642_642789

noncomputable def find_y (y : ℝ) : Prop :=
  let θ := Real.arcsin (sqrt 5 / 5) in
  let P := (-2, y) in
  sin θ = sqrt 5 / 5
  ∧ -2 / sqrt (4 + y^2) = cos θ
  ∧ sin θ = y / sqrt (4 + y^2)

theorem find_y_eq_1 : ∃ y : ℝ, find_y y ∧ y = 1 :=
begin
  use 1,
  dsimp [find_y],
  split,
  { sorry }, -- This would be where you prove the conditions hold for y = 1.
  { refl }
end

end find_y_eq_1_l642_642789


namespace solve_positive_integer_x_l642_642556

theorem solve_positive_integer_x : ∃ (x : ℕ), 4 * x^2 - 16 * x - 60 = 0 ∧ x = 6 :=
by
  sorry

end solve_positive_integer_x_l642_642556


namespace part1_part2_l642_642411

noncomputable def f (x ω : ℝ) : ℝ := (3/2) * sin (ω * x) + (sqrt 3 / 2) * cos (ω * x)

theorem part1 (ω : ℝ) (hω : ω > 0) (T : ℝ) (hT : T = 3 * π)
  (h_period: ∀ x, f x ω = f (x + T) ω) :
  ω = 2 / 3 ∧ ∀ k : ℤ, -π + 3 * k * π ≤ x ∧ x ≤ π / 2 + 3 * k * π → f x (2 / 3) = (sqrt 3) * sin ((2 / 3) * x + π / 6) :=
sorry

theorem part2 (α ω : ℝ) (hω : ω = 2) (hα : 0 < α ∧ α < π) (h_val : f α ω = 3/2) :
  α = π / 4 ∨ α = π / 12 :=
sorry

end part1_part2_l642_642411


namespace cos_150_eq_neg_half_l642_642280

theorem cos_150_eq_neg_half : Real.cos (150 * Real.pi / 180) = -1 / 2 := 
by
  sorry

end cos_150_eq_neg_half_l642_642280


namespace cos_150_eq_neg_sqrt3_div_2_l642_642165

theorem cos_150_eq_neg_sqrt3_div_2 : Real.cos (150 * Real.pi / 180) = - (Real.sqrt 3 / 2) :=
by
  sorry

end cos_150_eq_neg_sqrt3_div_2_l642_642165


namespace cosine_150_eq_neg_sqrt3_div_2_l642_642247

theorem cosine_150_eq_neg_sqrt3_div_2 :
  (∀ θ : ℝ, θ = real.pi / 6) →
  (∀ θ : ℝ, cos (real.pi - θ) = -cos θ) →
  cos (5 * real.pi / 6) = -real.sqrt 3 / 2 :=
by
  intros hθ hcos_identity
  sorry

end cosine_150_eq_neg_sqrt3_div_2_l642_642247


namespace mean_score_of_all_students_l642_642913

-- Conditions
def M : ℝ := 90
def A : ℝ := 75
def ratio (m a : ℝ) : Prop := m / a = 2 / 3

-- Question and correct answer
theorem mean_score_of_all_students (m a : ℝ) (hm : ratio m a) : (60 * a + 75 * a) / (5 * a / 3) = 81 := by
  sorry

end mean_score_of_all_students_l642_642913


namespace cheese_partition_possible_l642_642005

-- Conditions
def weights : List ℝ := [w_1, w_2, ..., w_25] -- Assume distinct weights
def total_weight : ℝ := weights.sum

-- Question statement
theorem cheese_partition_possible :
  ∃ (k : ℕ) (part_1 part_2 : ℝ) (bag1 bag2 : List ℝ),
    k < 25 ∧
    part_1 + part_2 = weights.get k ∧
    weights.removeNth k = bag1 ++ bag2 ∧
    bag1.length = 12 ∧
    bag2.length = 12 ∧
    bag1.sum + part_1 = bag2.sum + part_2 :=
sorry

end cheese_partition_possible_l642_642005


namespace length_sum_of_vectors_l642_642546

open EuclideanSpace

theorem length_sum_of_vectors 
  {n : Type*} [DecidableEq n] [Fintype n] [NormedAddCommGroup (PiLp 2 (λ i : n, ℝ))]
  (a1 a2 a3 a4 a5 : PiLp 2 (λ i : n, ℝ)) :
  ∃ (v1 v2 v3 v4 v5 : PiLp 2 (λ i : n, ℝ)),
  v1 = a1 ∧ v2 = a2 ∧ v3 = a3 ∧ v4 = a4 ∧ v5 = a5 ∧
  (‖v1 + v2‖ ≤ ‖v3 + v4 + v5‖ ∨ 
   ‖v1 + v3‖ ≤ ‖v2 + v4 + v5‖ ∨ 
   ‖v1 + v4‖ ≤ ‖v2 + v3 + v5‖ ∨ 
   ‖v1 + v5‖ ≤ ‖v2 + v3 + v4‖ ∨ 
   ‖v2 + v3‖ <= ‖v1 + v4 + v5‖ ∨
   ‖v2 + v4‖ <= ‖v1 + v3 + v5‖ ∨
   ‖v2 + v5‖ <= ‖v1 + v3 + v4‖ ∨
   ‖v3 + v4‖ <= ‖v1 + v2 + v5‖ ∨
   ‖v3 + v5‖ <= ‖v1 + v2 + v4‖ ∨
   ‖v4 + v5‖ <= ‖v1 + v2 + v3‖) :=
sorry

end length_sum_of_vectors_l642_642546


namespace three_digit_numbers_divisible_by_17_l642_642432

theorem three_digit_numbers_divisible_by_17 : 
  let is_three_digit (n: ℕ) := 100 ≤ n ∧ n ≤ 999 in
  let divisible_by_17 (n: ℕ) := n % 17 = 0 in
  (∃ s: Finset ℕ, ∀ x ∈ s, is_three_digit x ∧ divisible_by_17 x ∧ s.card = 53) :=
sorry

end three_digit_numbers_divisible_by_17_l642_642432


namespace john_total_amount_to_pay_l642_642879

-- Define constants for the problem
def total_cost : ℝ := 6650
def rebate_percentage : ℝ := 0.06
def sales_tax_percentage : ℝ := 0.10

-- The main theorem to prove the final amount John needs to pay
theorem john_total_amount_to_pay : total_cost * (1 - rebate_percentage) * (1 + sales_tax_percentage) = 6876.10 := by
  sorry    -- Proof skipped

end john_total_amount_to_pay_l642_642879


namespace eggs_per_box_l642_642906

theorem eggs_per_box (num_boxes : ℕ) (total_eggs : ℕ) (h1 : num_boxes = 3) (h2 : total_eggs = 21) : (total_eggs / num_boxes) = 7 :=
by
  -- Using the provided conditions to solve the problem
  rw [h1, h2]
  norm_num
  sorry

end eggs_per_box_l642_642906


namespace exists_point_P_l642_642052

noncomputable def P := (4, Real.sqrt 15)
def F1 := (-3, 0 : ℝ)
def F2 := (3, 0 : ℝ)
def hyperbola (x y : ℝ) := x^2 / 4 - y^2 / 5 = 1
def first_quadrant (x y : ℝ) := x > 0 ∧ y > 0
def G (x y : ℝ) := (x / 3, y / 3)
def centroid (P F1 F2 : Prod ℝ ℝ × Prod ℝ ℝ × Prod ℝ ℝ) := G P.fst P.snd

theorem exists_point_P (P : Prod ℝ ℝ) :
  hyperbola P.fst P.snd → 
  first_quadrant P.fst P.snd →
  (∃ P, centroid P F1 F2 = (4/3, sqrt(15)/3)) := 
by {
  unfold hyperbola first_quadrant,
  simp [P],
  use P,
  sorry
}

end exists_point_P_l642_642052


namespace cos_150_degree_l642_642186

theorem cos_150_degree : Real.cos (150 * Real.pi / 180) = -Real.sqrt 3 / 2 :=
by
  sorry

end cos_150_degree_l642_642186


namespace cos_150_eq_neg_sqrt3_div_2_l642_642156

theorem cos_150_eq_neg_sqrt3_div_2 : Real.cos (150 * Real.pi / 180) = - (Real.sqrt 3 / 2) :=
by
  sorry

end cos_150_eq_neg_sqrt3_div_2_l642_642156


namespace run_to_cafe_time_l642_642823

theorem run_to_cafe_time (h_speed_const : ∀ t1 t2 d1 d2 : ℝ, (t1 / d1) = (t2 / d2))
  (h_store_time : 24 = 3 * (24 / 3))
  (h_cafe_halfway : ∀ d : ℝ, d = 1.5) :
  ∃ t : ℝ, t = 12 :=
by
  sorry

end run_to_cafe_time_l642_642823


namespace cos_150_eq_neg_sqrt3_div_2_l642_642265

theorem cos_150_eq_neg_sqrt3_div_2 :
  let theta := 30 * Real.pi / 180 in
  let Q := 150 * Real.pi / 180 in
  cos Q = - (Real.sqrt 3 / 2) := by
    sorry

end cos_150_eq_neg_sqrt3_div_2_l642_642265


namespace maxwell_distance_traveled_l642_642045

theorem maxwell_distance_traveled
    (distance_between_homes : ℕ)
    (maxwell_speed : ℕ)
    (brad_speed : ℕ)
    (starting_condition : Prop)
    (meeting_time : ℕ) :
    distance_between_homes = 40 →
    maxwell_speed = 3 →
    brad_speed = 5 →
    starting_condition →
    meeting_time = distance_between_homes / (maxwell_speed + brad_speed) →
    maxwell_speed * meeting_time = 15 :=
by
  intros h_dist h_max_speed h_brad_speed h_start_cond h_meet_time
  rw [h_dist, h_max_speed, h_brad_speed] at h_meet_time
  rw [div_eq_iff (by norm_num : (0 : ℕ) < 8)] at h_meet_time
  norm_num at h_meet_time
  rw [h_meet_time, h_max_speed]
  norm_num
  sorry

end maxwell_distance_traveled_l642_642045


namespace cos_150_eq_neg_half_l642_642278

theorem cos_150_eq_neg_half : Real.cos (150 * Real.pi / 180) = -1 / 2 := 
by
  sorry

end cos_150_eq_neg_half_l642_642278


namespace odd_function_condition_l642_642451

noncomputable def f (x a : ℝ) : ℝ := (Real.sin x) / ((x - a) * (x + 1))

theorem odd_function_condition (a : ℝ) (h : ∀ x : ℝ, f x a = - f (-x) a) : a = 1 := 
sorry

end odd_function_condition_l642_642451


namespace cos_150_deg_l642_642328

-- Define the conditions as variables
def angle1 := 150
def angle2 := 180
def angle3 := 30
def cos30 := Real.sqrt 3 / 2

-- The main statement to be proved
theorem cos_150_deg : Real.cos (150 * Real.pi / 180) = - (Real.sqrt 3 / 2) :=
by
  sorry

end cos_150_deg_l642_642328


namespace katie_smoothies_l642_642881

theorem katie_smoothies (smoothies_per_3_bananas : ℕ) (bananas_used : ℕ) : smoothies_per_3_bananas = 13 → bananas_used = 15 → (smoothies_per_3_bananas * bananas_used) / 3 = 65 :=
by
  intros h1 h2
  rw [h1, h2]
  sorry

end katie_smoothies_l642_642881


namespace cos_150_eq_neg_sqrt3_div_2_l642_642299

open Real

theorem cos_150_eq_neg_sqrt3_div_2 :
  cos (150 * pi / 180) = - (sqrt 3 / 2) := 
sorry

end cos_150_eq_neg_sqrt3_div_2_l642_642299


namespace avg_fishes_caught_per_lake_l642_642427

-- Definitions based on conditions
def fish_caught_Lake_Marion := 38
def fish_caught_Lake_Norman := 52
def fish_caught_Lake_Wateree := 27
def fish_caught_Lake_Wylie := 45
def fish_caught_Lake_Keowee := 64
def number_of_lakes := 5

-- Proof statement that the average number of fishes caught per lake is 45.2
theorem avg_fishes_caught_per_lake : 
  (fish_caught_Lake_Marion + fish_caught_Lake_Norman + fish_caught_Lake_Wateree + fish_caught_Lake_Wylie + fish_caught_Lake_Keowee) / number_of_lakes = 45.2 := 
by
  sorry

end avg_fishes_caught_per_lake_l642_642427


namespace extreme_points_range_of_a_inequality_l642_642889

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := log x + (1/2)*x^2 + a*x
def g (x : ℝ) : ℝ := exp x + (3/2)*x^2

theorem extreme_points (a : ℝ) :
  (∀ x > 0, f x a < g x) →
  (if a ∈ set.Ici (-2) then
    ∀ x > 0, f' x a ≥ 0
   else
    ∃ x1 x2 > 0, f' x1 a = 0 ∧ f' x2 a = 0
  ) :=
sorry

theorem range_of_a (a : ℝ) : (∀ x > 0, f x a ≤ g x) → a ≤ real.exp 1 + 1 :=
sorry

theorem inequality (x : ℝ) : x > 0 → exp x + x^2 - (exp 1 + 1)*x + exp 1 / x > 2 :=
sorry

end extreme_points_range_of_a_inequality_l642_642889


namespace infinitely_many_not_clean_l642_642892

theorem infinitely_many_not_clean (S : Set ℕ) (hS : S.Nonempty) :
  ∃ᶠ n in at_top, ¬∃! (l : Finset ℕ), (l ⊆ S) ∧ (l.sum id = n) ∧ (l.card % 2 = 1) :=
by
  sorry

end infinitely_many_not_clean_l642_642892


namespace sum_first_70_odd_eq_4900_l642_642047

theorem sum_first_70_odd_eq_4900 (h : (70 * (70 + 1) = 4970)) :
  (70 * 70 = 4900) :=
by
  sorry

end sum_first_70_odd_eq_4900_l642_642047


namespace parabola_solution_l642_642092

noncomputable def parabola_equation (p x0 : ℝ) : Prop :=
  let y := -4 * real.sqrt 2 in
  let directrix_distance := 6 in
  y^2 = 2 * p * x0 ∧ distance (x0, y) (x0, 0) = directrix_distance

theorem parabola_solution (p : ℝ) (h_pos : p > 0) :
  let y := -4 * real.sqrt 2 in
  let x0 := 16 / p in
  let directrix_distance := 6 in
  parabola_equation p x0 →
  p = 4 ∨ p = 8 → (y^2 = 8 * (16 / p) ∨ y^2 = 16 * (16 / p)) :=
by {
  intros,
  sorry
}

end parabola_solution_l642_642092


namespace smallest_int_solution_l642_642025

theorem smallest_int_solution : ∃ y : ℤ, y = 6 ∧ ∀ z : ℤ, z > 5 → y ≤ z := sorry

end smallest_int_solution_l642_642025


namespace line_intersects_circle_min_chord_length_l642_642389

open Real

noncomputable def CircleC : AffinePlane ℝ := {x | x.1^2 + x.2^2 - 4 * x.1 - 2 * x.2 - 20 = 0}
noncomputable def LineL (m : ℝ) : AffinePlane ℝ := {x | m * x.1 - x.2 - m + 3 = 0}

theorem line_intersects_circle (m : ℝ) : ∃ (P : ℝ × ℝ), P ∈ CircleC ∧ P ∈ LineL m :=
sorry

theorem min_chord_length : ∃ m : ℝ, ∃ (l : AffinePlane ℝ), LineL l = {x | x.1 - 2 * x.2 + 5 = 0} ∧
      (∀ (A B : ℝ × ℝ), A ∈ CircleC ∧ B ∈ CircleC ∧ A ∈ l ∧ B ∈ l → dist A B = 4 * sqrt 5) :=
sorry

end line_intersects_circle_min_chord_length_l642_642389


namespace quarter_circle_limit_l642_642568

theorem quarter_circle_limit 
  (D : ℝ) (n : ℕ) (hD_pos : 0 < D) (hn_pos : 0 < n) :
  (∑ i in finset.range n, (π * D / (4 * n))) = π * D / 4 :=
by
  sorry

end quarter_circle_limit_l642_642568


namespace mary_baking_cups_l642_642910

-- Conditions
def flour_needed : ℕ := 9
def sugar_needed : ℕ := 11
def flour_added : ℕ := 4
def sugar_added : ℕ := 0

-- Statement to prove
theorem mary_baking_cups : sugar_needed - (flour_needed - flour_added) = 6 := by
  sorry

end mary_baking_cups_l642_642910


namespace incorrect_method_B_l642_642030

def Point : Type := sorry
def Locus : set Point := sorry
def Conditions : Point → Prop := sorry

-- Statement A
def correctA (p : Point) : Prop := (p ∈ Locus ↔ Conditions p)

-- Statement B (we need to show this is incorrect)
def incorrectB (p : Point) : Prop := (¬(Conditions p) → ¬(p ∈ Locus)) ∧ (p ∈ Locus → Conditions p)

-- Theorem: Statement B is incorrect for defining a locus
theorem incorrect_method_B: ¬(∀ p : Point, incorrectB p) :=
sorry

end incorrect_method_B_l642_642030


namespace find_length_b_l642_642786

variable (a b : EuclideanSpace ℝ (Fin 2))

-- Given conditions
axiom length_a : ‖a‖ = 5
axiom length_a_add_b : ‖a + b‖ = 5 * Real.sqrt 2
axiom a_perp_b : inner a b = 0

-- Proof goal
theorem find_length_b : ‖b‖ = 5 := by
  sorry

end find_length_b_l642_642786


namespace max_xy_l642_642056

theorem max_xy (x y : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_eq : 3 * x + 2 * y = 12) : 
  xy ≤ 6 :=
sorry

end max_xy_l642_642056


namespace cos_150_eq_neg_sqrt3_div_2_l642_642162

theorem cos_150_eq_neg_sqrt3_div_2 : Real.cos (150 * Real.pi / 180) = - (Real.sqrt 3 / 2) :=
by
  sorry

end cos_150_eq_neg_sqrt3_div_2_l642_642162


namespace tall_mirror_passes_l642_642349

theorem tall_mirror_passes (T : ℕ)
    (s_tall_ref : ℕ)
    (s_wide_ref : ℕ)
    (e_tall_ref : ℕ)
    (e_wide_ref : ℕ)
    (wide_passes : ℕ)
    (total_reflections : ℕ)
    (H1 : s_tall_ref = 10)
    (H2 : s_wide_ref = 5)
    (H3 : e_tall_ref = 6)
    (H4 : e_wide_ref = 3)
    (H5 : wide_passes = 5)
    (H6 : s_tall_ref * T + s_wide_ref * wide_passes + e_tall_ref * T + e_wide_ref * wide_passes = 88) : 
    T = 3 := 
by sorry

end tall_mirror_passes_l642_642349


namespace cos_150_eq_negative_cos_30_l642_642144

theorem cos_150_eq_negative_cos_30 :
  Real.cos (150 * Real.pi / 180) = - (Real.cos (30 * Real.pi / 180)) :=
by sorry

end cos_150_eq_negative_cos_30_l642_642144


namespace cos_150_eq_neg_sqrt3_over_2_l642_642212

theorem cos_150_eq_neg_sqrt3_over_2 :
  ∀ {deg cosQ1 cosQ2: ℝ},
    cosQ1 = 30 →
    cosQ2 = 180 - cosQ1 →
    cos 150 = - (cos cosQ1) →
    cos cosQ1 = sqrt 3 / 2 →
    cos 150 = -sqrt 3 / 2 :=
by
  intros deg cosQ1 cosQ2 h1 h2 h3 h4
  sorry

end cos_150_eq_neg_sqrt3_over_2_l642_642212


namespace no_sum_76_l642_642637

theorem no_sum_76 (n : ℕ) (hn1 : n ≥ 10^11) (hn2 : n < 10^12) (hn3 : ∀ (i : ℕ), (nat.digits 10 n).nth i ∈ [1, 5, 9]) (hn4 : 37 ∣ n) :
  (nat.digits 10 n).sum ≠ 76 :=
sorry

end no_sum_76_l642_642637


namespace minimum_total_number_of_balls_l642_642641

theorem minimum_total_number_of_balls (x y z t : ℕ) 
  (h1 : x ≥ 4)
  (h2 : x ≥ 3 ∧ y ≥ 1)
  (h3 : x ≥ 2 ∧ y ≥ 1 ∧ z ≥ 1)
  (h4 : x ≥ 1 ∧ y ≥ 1 ∧ z ≥ 1 ∧ t ≥ 1) :
  x + y + z + t = 21 :=
  sorry

end minimum_total_number_of_balls_l642_642641


namespace solve_for_m_l642_642804

noncomputable section

-- Define the hyperbola equation and related properties.

def hyperbola (x y m : ℝ) : Prop := (x^2 / m) - y^2 = 1

def eccentricity (m : ℝ) : ℝ := Real.sqrt(1 + 1 / m)

def real_axis_length_twice_eccentricity (m : ℝ) : Prop :=
  2 * (eccentricity m) = 2 * Real.sqrt(m)

-- The theorem that we want to prove.
theorem solve_for_m (m : ℝ) (h : hyperbola x y m) (h_cond : real_axis_length_twice_eccentricity m) :
  m = (1 + Real.sqrt 5) / 2 :=
by
  sorry

end solve_for_m_l642_642804


namespace cos_150_eq_neg_sqrt3_over_2_l642_642219

theorem cos_150_eq_neg_sqrt3_over_2 : Real.cos (150 * Real.pi / 180) = -Real.sqrt 3 / 2 := 
by 
  sorry

end cos_150_eq_neg_sqrt3_over_2_l642_642219


namespace propositions_2_and_3_are_true_l642_642105

open_locale classical

noncomputable def is_hyperbola_locus (A B P : ℝ → ℝ) (s : ℝ) : Prop :=
abs (dist P A - dist P B) = s

noncomputable def maximum_value_PA (A B P : ℝ → ℝ) : Prop :=
let PA := dist P A in let PB := dist P B in
  let a := 5 in let c := 3 in
  |PA| = 10 - |PB| ∧ |AB| = 6 ∧ PA = 8

noncomputable def roots_as_eccentricities : Prop :=
let roots := [2, 1/2] in
  ∀ r ∈ roots, (r > 1 ∨ r < 1)

noncomputable def different_foci : Prop :=
  let foci_hyperbola := (5, 0) in let foci_ellipse := (0, sqrt 35) in
  foci_hyperbola ≠ foci_ellipse

theorem propositions_2_and_3_are_true : 
  ∀ (A B P : ℝ → ℝ),
  maximum_value_PA A B P ∧ roots_as_eccentricities :=
by {
  -- Proof would go here
  sorry
}

end propositions_2_and_3_are_true_l642_642105


namespace cos_150_deg_l642_642315

-- Define the conditions as variables
def angle1 := 150
def angle2 := 180
def angle3 := 30
def cos30 := Real.sqrt 3 / 2

-- The main statement to be proved
theorem cos_150_deg : Real.cos (150 * Real.pi / 180) = - (Real.sqrt 3 / 2) :=
by
  sorry

end cos_150_deg_l642_642315


namespace typing_speed_ratio_l642_642031

theorem typing_speed_ratio (T M : ℝ) (h1 : T + M = 12) (h2 : T + 1.25 * M = 14) : M / T = 2 :=
by
  sorry

end typing_speed_ratio_l642_642031


namespace probability_correct_l642_642979

-- Defining the conditions of the problem
def num_cards : ℕ := 52
def first_card_five : ℕ → Prop
| 5 := true
| _ := false

def second_card_heart (card : ℕ) : Prop :=
card = 11 -- Assuming 11 represents a heart in a simplified model

def third_card_ace : ℕ → Prop
| 1  := true  -- Assuming 1 represents an Ace in this simplified model
| _ := false

-- The required theorem
theorem probability_correct :
  (3/52) * (12/51) * (4/50) + (3/52) * (1/51) * (3/50) + 
  (1/52) * (11/51) * (4/50) + (1/52) * (1/51) * (3/50) 
  = (1 / 663) :=
begin
  -- Proof steps would go here
  sorry
end

end probability_correct_l642_642979


namespace birds_in_house_l642_642461

theorem birds_in_house (B : ℕ) :
  let dogs := 3
  let cats := 18
  let humans := 7
  let total_heads := B + dogs + cats + humans
  let total_feet := 2 * B + 4 * dogs + 4 * cats + 2 * humans
  total_feet = total_heads + 74 → B = 4 :=
by
  intros dogs cats humans total_heads total_feet condition
  -- We assume the condition and work towards the proof.
  sorry

end birds_in_house_l642_642461


namespace sum_inf_series_l642_642705

theorem sum_inf_series :
  (\sum_{n=1}^{\infty} \frac{(4 * n) - 3}{3^n}) = 1 :=
by
  sorry

end sum_inf_series_l642_642705


namespace find_m_l642_642390

theorem find_m (m : ℝ)
  (sample_points : List (ℝ × ℝ))
  (regression_eq : ∀ x y, (x, y) ∈ sample_points → y = 2 * x + 1) 
  (mean_formula_x : ∀ (sample_points : List (ℝ × ℝ)), ∑ x in sample_points.map Prod.fst, x / sample_points.length = 5 / 2)
  (mean_formula_y : ∀ (y_sum : ℝ), (2.98 + 5.01 + m + 9) / 4 = y_sum)
  (regression_passing_point : ∀ y_sum, y_sum = 6) : 
  m = 7.01 :=
  sorry

end find_m_l642_642390


namespace find_sets_of_four_numbers_l642_642355

variables {x1 x2 x3 x4 : ℝ} {t : ℝ}

theorem find_sets_of_four_numbers (h1 : x1 + (x2 * x3 * x4) = 2)
                                 (h2 : x2 + (x1 * x3 * x4) = 2)
                                 (h3 : x3 + (x1 * x2 * x4) = 2)
                                 (h4 : x4 + (x1 * x2 * x3) = 2) :
    {x1, x2, x3, x4} = {1, 1, 1, 1} ∨ {x1, x2, x3, x4} = {3, -1, -1, -1} :=
by
  sorry

end find_sets_of_four_numbers_l642_642355


namespace volume_is_correct_l642_642366

noncomputable def volume_of_solid : ℝ :=
  let solid := {p : ℝ × ℝ × ℝ | sqrt (p.1^2 + p.2^2) + abs p.3 ≤ 1} in
  volume solid

theorem volume_is_correct :
  volume_of_solid = (2 * real.pi) / 3 :=
sorry

end volume_is_correct_l642_642366


namespace two_solutions_for_positive_integer_m_l642_642757

theorem two_solutions_for_positive_integer_m :
  ∃ k : ℕ, k = 2 ∧ (∀ m : ℕ, 0 < m → 990 % (m^2 - 2) = 0 → m = 2 ∨ m = 3) := 
sorry

end two_solutions_for_positive_integer_m_l642_642757


namespace sum_series_eq_one_l642_642712

noncomputable def series : ℝ := ∑' n : ℕ, (4 * (n + 1) - 3) / 3^(n + 1)

theorem sum_series_eq_one : series = 1 := 
by sorry

end sum_series_eq_one_l642_642712


namespace distance_between_wheels_l642_642583

theorem distance_between_wheels 
  (D : ℕ) 
  (back_perimeter : ℕ) (front_perimeter : ℕ) 
  (more_revolutions : ℕ)
  (h1 : back_perimeter = 9)
  (h2 : front_perimeter = 7)
  (h3 : more_revolutions = 10)
  (h4 : D / front_perimeter = D / back_perimeter + more_revolutions) : 
  D = 315 :=
by
  sorry

end distance_between_wheels_l642_642583


namespace cos_150_eq_neg_sqrt3_div_2_l642_642261

theorem cos_150_eq_neg_sqrt3_div_2 :
  let theta := 30 * Real.pi / 180 in
  let Q := 150 * Real.pi / 180 in
  cos Q = - (Real.sqrt 3 / 2) := by
    sorry

end cos_150_eq_neg_sqrt3_div_2_l642_642261


namespace parametric_equation_of_midpoint_Q_l642_642542

variable (θ : ℝ)

/- P is a point moving on the circle (x - 6)^2 + (y - 4)^2 = 4 -/
def point_P_x : ℝ := 6 + 2 * Real.cos θ
def point_P_y : ℝ := 4 + 2 * Real.sin θ

/- O is the origin, by default (0, 0) -/
def point_O_x : ℝ := 0
def point_O_y : ℝ := 0

/- Midpoint Q of OP -/
def midpoint_Q_x : ℝ := (point_P_x θ + point_O_x) / 2
def midpoint_Q_y : ℝ := (point_P_y θ + point_O_y) / 2

theorem parametric_equation_of_midpoint_Q :
  midpoint_Q_x θ = 3 + Real.cos θ ∧
  midpoint_Q_y θ = 2 + Real.sin θ :=
by
  -- Proof steps go here
  sorry

end parametric_equation_of_midpoint_Q_l642_642542


namespace cos_150_eq_neg_sqrt3_div_2_l642_642163

theorem cos_150_eq_neg_sqrt3_div_2 : Real.cos (150 * Real.pi / 180) = - (Real.sqrt 3 / 2) :=
by
  sorry

end cos_150_eq_neg_sqrt3_div_2_l642_642163


namespace cosine_150_eq_neg_sqrt3_div_2_l642_642243

theorem cosine_150_eq_neg_sqrt3_div_2 :
  (∀ θ : ℝ, θ = real.pi / 6) →
  (∀ θ : ℝ, cos (real.pi - θ) = -cos θ) →
  cos (5 * real.pi / 6) = -real.sqrt 3 / 2 :=
by
  intros hθ hcos_identity
  sorry

end cosine_150_eq_neg_sqrt3_div_2_l642_642243


namespace pages_filled_with_images_ratio_l642_642639

theorem pages_filled_with_images_ratio (total_pages intro_pages text_pages : ℕ) 
  (h_total : total_pages = 98)
  (h_intro : intro_pages = 11)
  (h_text : text_pages = 19)
  (h_blank : 2 * text_pages = total_pages - intro_pages - 2 * text_pages) :
  (total_pages - intro_pages - text_pages - text_pages) / total_pages = 1 / 2 :=
by
  sorry

end pages_filled_with_images_ratio_l642_642639


namespace sliced_meat_cost_per_type_with_rush_shipping_l642_642377

theorem sliced_meat_cost_per_type_with_rush_shipping:
  let original_cost := 40.0
  let rush_delivery_percentage := 0.3
  let num_types := 4
  let rush_delivery_cost := rush_delivery_percentage * original_cost
  let total_cost := original_cost + rush_delivery_cost
  let cost_per_type := total_cost / num_types
  cost_per_type = 13.0 :=
by
  sorry

end sliced_meat_cost_per_type_with_rush_shipping_l642_642377


namespace sum_not_complete_residue_system_l642_642524

theorem sum_not_complete_residue_system {n : ℕ} (hn_even : Even n)
    (a b : Fin n → ℕ) (ha : ∀ k, a k < n) (hb : ∀ k, b k < n) 
    (h_complete_a : ∀ x : Fin n, ∃ k : Fin n, a k = x) 
    (h_complete_b : ∀ y : Fin n, ∃ k : Fin n, b k = y) :
    ¬ (∀ z : Fin n, ∃ k : Fin n, ∃ l : Fin n, z = (a k + b l) % n) :=
by
  sorry

end sum_not_complete_residue_system_l642_642524


namespace problem1_problem2_problem3_l642_642794

noncomputable def f (x : ℝ) : ℝ := if x >= 0 then x^2 - 2 * x else (abs x)^2 - 2 * abs x

-- Define the condition that f is an even function
def even_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f x = f (-x)

-- Problem 1: Prove the minimum value of f(x) is -1.
theorem problem1 (h_even : even_function f) : ∃ x : ℝ, f x = -1 :=
by
  sorry

-- Problem 2: Prove the solution set of f(x) > 0 is (-∞, -2) ∪ (2, +∞).
theorem problem2 (h_even : even_function f) : 
  { x : ℝ | f x > 0 } = { x : ℝ | x < -2 } ∪ { x : ℝ | x > 2 } :=
by
  sorry

-- Problem 3: Prove there exists a real number x such that f(x+2) + f(-x) = 0.
theorem problem3 (h_even : even_function f) : ∃ x : ℝ, f (x + 2) + f (-x) = 0 :=
by
  sorry

end problem1_problem2_problem3_l642_642794


namespace cos_150_eq_neg_half_l642_642273

theorem cos_150_eq_neg_half : Real.cos (150 * Real.pi / 180) = -1 / 2 := 
by
  sorry

end cos_150_eq_neg_half_l642_642273


namespace sum_reciprocals_bound_l642_642018

theorem sum_reciprocals_bound (n : ℕ) (hn : n ≥ 1) :
  (∑ i in finset.range (2 * n + 1) \ finset.range n, (1 : ℝ) / (i + n + 1)) > 25 / 24 := sorry

end sum_reciprocals_bound_l642_642018


namespace projected_increase_l642_642908

theorem projected_increase (R : ℝ) (P : ℝ) 
  (h1 : ∃ P, ∀ (R : ℝ), 0.9 * R = 0.75 * (R + (P / 100) * R)) 
  (h2 : ∀ (R : ℝ), R > 0) :
  P = 20 :=
by
  sorry

end projected_increase_l642_642908


namespace num_valid_triplets_l642_642431

-- Define the original set S
def S : Finset ℕ := Finset.range 16   -- {0, 1, 2, ..., 15}
def S' : Finset ℕ := S.erase 0         -- {1, 2, 3, ..., 15}

-- Define the condition for valid triplets
def is_valid_triplet (t : Finset ℕ) : Prop :=
  t.card = 3 ∧ (S'.sum - t.sum) / (S'.card - 3) = 8

-- Define the proof statement
theorem num_valid_triplets :
  (Finset.filter is_valid_triplet S'.powerset).card = 3 :=
begin
  sorry
end

end num_valid_triplets_l642_642431


namespace cos_150_deg_l642_642322

-- Define the conditions as variables
def angle1 := 150
def angle2 := 180
def angle3 := 30
def cos30 := Real.sqrt 3 / 2

-- The main statement to be proved
theorem cos_150_deg : Real.cos (150 * Real.pi / 180) = - (Real.sqrt 3 / 2) :=
by
  sorry

end cos_150_deg_l642_642322


namespace OrthographicProjections_l642_642027

theorem OrthographicProjections (
  main_view top_view left_view : orthographic_projection
) : 
  aligned_length main_view top_view ∧
  level_height main_view left_view ∧
  equal_width left_view top_view :=
sorry

end OrthographicProjections_l642_642027


namespace isabel_piggy_bank_l642_642496

theorem isabel_piggy_bank:
  ∀ (initial_amount spent_on_toy spent_on_book remaining_amount : ℕ),
  initial_amount = 204 →
  spent_on_toy = initial_amount / 2 →
  remaining_amount = initial_amount - spent_on_toy →
  spent_on_book = remaining_amount / 2 →
  remaining_amount - spent_on_book = 51 :=
by
  sorry

end isabel_piggy_bank_l642_642496


namespace sum_integers_from_neg25_to_10_l642_642615

theorem sum_integers_from_neg25_to_10 : 
  ∑ i in Finset.range (10 - (-25) + 1), (i - 25) = -270 :=
by
  sorry

end sum_integers_from_neg25_to_10_l642_642615


namespace paper_boat_travel_time_l642_642485

-- Given conditions
def embankment_length : ℝ := 50
def boat_length : ℝ := 10
def downstream_time : ℝ := 5
def upstream_time : ℝ := 4

-- Derived conditions from the given problem
def downstream_speed := embankment_length / downstream_time
def upstream_speed := embankment_length / upstream_time

-- Prove that the paper boat's travel time is 40 seconds
theorem paper_boat_travel_time :
  let v_boat := (downstream_speed + upstream_speed) / 2 in
  let v_current := (downstream_speed - upstream_speed) / 2 in
  let travel_time := embankment_length / v_current in
  travel_time = 40 := 
  sorry

end paper_boat_travel_time_l642_642485


namespace possible_S_values_l642_642920

theorem possible_S_values :
  ∀ (a : Fin 2012 → ℤ),
    (∀ i, a i = 1 ∨ a i = -1) →
    (∀ i : Fin 2012,
      ∑ (j : Fin 10), a ((i + j) % 2012) ≠ 0) →
    ∃ (S : ℤ), S = ∑ i, a i ∧
    (S % 2 = 0) ∧ -- ensure even sum
    (-2012 ≤ S ∧ S ≤ -404 ∨ 404 ≤ S ∧ S ≤ 2012)
:=
by
  sorry

end possible_S_values_l642_642920


namespace cos_150_eq_neg_sqrt3_div_2_l642_642132

theorem cos_150_eq_neg_sqrt3_div_2 :
  ∃ θ : ℝ, θ = 150 ∧
           0 < θ ∧ θ < 180 ∧
           ∃ φ : ℝ, φ = 180 - θ ∧
           cos φ = Real.sqrt 3 / 2 ∧
           cos θ = - cos φ :=
  sorry

end cos_150_eq_neg_sqrt3_div_2_l642_642132


namespace cos_150_eq_neg_sqrt3_over_2_l642_642231

theorem cos_150_eq_neg_sqrt3_over_2 : Real.cos (150 * Real.pi / 180) = -Real.sqrt 3 / 2 := 
by 
  sorry

end cos_150_eq_neg_sqrt3_over_2_l642_642231


namespace cos_150_eq_neg_sqrt3_div_2_l642_642252

theorem cos_150_eq_neg_sqrt3_div_2 :
  let theta := 30 * Real.pi / 180 in
  let Q := 150 * Real.pi / 180 in
  cos Q = - (Real.sqrt 3 / 2) := by
    sorry

end cos_150_eq_neg_sqrt3_div_2_l642_642252


namespace number_of_people_speaking_both_languages_l642_642625

theorem number_of_people_speaking_both_languages
  (total : ℕ) (L : ℕ) (F : ℕ) (N : ℕ) (B : ℕ) :
  total = 25 → L = 13 → F = 15 → N = 6 → total = L + F - B + N → B = 9 :=
by
  intros h_total h_L h_F h_N h_inclusion_exclusion
  sorry

end number_of_people_speaking_both_languages_l642_642625


namespace complex_modulus_solution_l642_642966

def complex_modulus_example : Complex := (1 : Complex) / (Complex.i - 1)

theorem complex_modulus_solution : Complex.abs complex_modulus_example = Real.sqrt 2 / 2 := by
  sorry

end complex_modulus_solution_l642_642966


namespace paper_boat_travel_time_l642_642490

theorem paper_boat_travel_time :
  ∀ (length_of_embankment : ℝ) (length_of_motorboat : ℝ)
    (time_downstream : ℝ) (time_upstream : ℝ) (v_boat : ℝ) (v_current : ℝ),
  length_of_embankment = 50 →
  length_of_motorboat = 10 →
  time_downstream = 5 →
  time_upstream = 4 →
  v_boat + v_current = length_of_embankment / time_downstream →
  v_boat - v_current = length_of_embankment / time_upstream →
  let speed_paper_boat := v_current in
  let travel_time := length_of_embankment / speed_paper_boat in
  travel_time = 40 :=
by
  intros length_of_embankment length_of_motorboat time_downstream time_upstream v_boat v_current
  intros h_length_emb h_length_motor t_down t_up h_v_boat_plus_current h_v_boat_minus_current
  let speed_paper_boat := v_current
  let travel_time := length_of_embankment / speed_paper_boat
  sorry

end paper_boat_travel_time_l642_642490


namespace cos_150_eq_neg_sqrt3_over_2_l642_642227

theorem cos_150_eq_neg_sqrt3_over_2 : Real.cos (150 * Real.pi / 180) = -Real.sqrt 3 / 2 := 
by 
  sorry

end cos_150_eq_neg_sqrt3_over_2_l642_642227


namespace log_base_4_30_l642_642378

variables (a c d : ℝ)
variables (log10 : ℝ → ℝ)

-- Given conditions
def log_10_2 := log10 2
def log_10_5 := log10 5

-- Definitions of a and c
def a := log_10_2
def c := log_10_5
def d := log10 3

theorem log_base_4_30 (log10 : ℝ → ℝ) (log_10_2_eq : log10 2 = a) (log_10_5_eq : log10 5 = c) (log_10_3_eq : log10 3 = d) :
  (log10 30 / log10 4) = (a + d + c) / (2 * a) :=
by
  sorry

end log_base_4_30_l642_642378


namespace triangle_third_side_length_l642_642463

noncomputable def length_of_third_side (AB AC : ℝ) (B : ℝ) (C : ℝ) (a : ℝ) : Prop :=
  (∀ (α β γ : ℝ), α + β + γ = π ∧ α = B ∧ β = 3 * C ∧ γ = C → a = 34)

theorem triangle_third_side_length :
  length_of_third_side 26 12 3C C 34 := sorry

end triangle_third_side_length_l642_642463


namespace limit_of_function_l642_642697

theorem limit_of_function (a x : ℝ) (h : ℝ) (ha : 0 < a) :
  ( ∀ ε > 0, ∃ δ > 0, ∀ h, abs h < δ → abs ( ( a ^ ( x + h ) + a ^ ( x - h ) - 2 * a ^ x ) / h ) < ε ) → 
  ( ∀ ε > 0, ∃ δ > 0, ∀ h, abs h < δ → abs ( a ^ x * log a ) < ε ) := 
sorry

end limit_of_function_l642_642697


namespace equal_distances_from_G_l642_642575

theorem equal_distances_from_G
  {A B C E F G R S : Type} [add_comm_group E] [module ℝ E]
  (h_triangle : triangle A B C)
  (h_incircle : incircle A B C E F)
  (h_intersect : line_through G (ray E B) (ray F C))
  (h_parallelogram1 : parallelogram B C E R)
  (h_parallelogram2 : parallelogram B C S F)
:
  dist G R = dist G S := 
sorry

end equal_distances_from_G_l642_642575


namespace cos_150_eq_negative_cos_30_l642_642141

theorem cos_150_eq_negative_cos_30 :
  Real.cos (150 * Real.pi / 180) = - (Real.cos (30 * Real.pi / 180)) :=
by sorry

end cos_150_eq_negative_cos_30_l642_642141


namespace choose_starters_with_twins_l642_642086

theorem choose_starters_with_twins :
  let total_players := 12
  let num_starters := 5
  let twins_num := 2
  let total_ways := Nat.choose total_players num_starters
  let without_twins := Nat.choose (total_players - twins_num) num_starters
  total_ways - without_twins = 540 := 
by
  let total_players := 12
  let num_starters := 5
  let twins_num := 2
  let total_ways := Nat.choose total_players num_starters
  let without_twins := Nat.choose (total_players - twins_num) num_starters
  exact Nat.sub_eq_of_eq_add sorry -- here we will need the exact proof steps which we skip

end choose_starters_with_twins_l642_642086


namespace number_of_tangents_from_origin_tangents_from_origin_unique_number_of_unique_tangents_from_origin_l642_642803

noncomputable def f (a x : ℝ) : ℝ := a*x^3 - 3*x^2 + 1

theorem number_of_tangents_from_origin (a : ℝ) (h : 2 * f a a = f a (-a) + f a (3*a))
  (h_nonzero : a ≠ 0) : 
  a = 1 ∨ a = -1 :=
sorry

theorem tangents_from_origin_unique (a : ℝ) (ha : a = 1 ∨ a = -1) :
  ∃! (x₀ : ℝ),
  let y₀ := f a x₀ in
  (y₀ = f a x₀) ∧ 
  (- y₀ = derivatives f a x₀ * x₀) :=
sorry

-- Combining the results to conclude the number of unique tangents
theorem number_of_unique_tangents_from_origin (a : ℝ) (h : 2 * f a a = f a (-a) + f a (3*a))
  (h_nonzero : a ≠ 0) : 
  (∃! (x₀ : ℝ), let y₀ := f a x₀ in 
  (y₀ = f a x₀) ∧ (- y₀ = derivatives f a x₀ * x₀)) ∧
  (a = 1 ∨ a = -1) :=
begin
  have := number_of_tangents_from_origin a h,
  sorry
end

end number_of_tangents_from_origin_tangents_from_origin_unique_number_of_unique_tangents_from_origin_l642_642803


namespace find_remainder_l642_642365

noncomputable def f (r : ℤ) := r^15 - r^3 + 1

theorem find_remainder :
  let r := 1 in (f r) = 1 := by
  sorry

end find_remainder_l642_642365


namespace radius_comparison_l642_642386

theorem radius_comparison 
  (a b c : ℝ)
  (da db dc r ρ : ℝ)
  (h₁ : da ≤ r)
  (h₂ : db ≤ r)
  (h₃ : dc ≤ r)
  (h₄ : 1 / 2 * (a * da + b * db + c * dc) = ρ * ((a + b + c) / 2)) :
  r ≥ ρ := 
sorry

end radius_comparison_l642_642386


namespace diagonal_length_rect_prism_l642_642069

theorem diagonal_length_rect_prism (x y z a : ℝ) (h₁ : x^2 + y^2 = 2) (h₂ : x^2 + z^2 = 2) (h₃ : y^2 + z^2 = 2) :
  a = √3 :=
by sorry

end diagonal_length_rect_prism_l642_642069


namespace jack_piggy_bank_l642_642878

variable (initial_amount : ℕ) (weekly_allowance : ℕ) (weeks : ℕ)

-- Conditions
def initial_amount := 43
def weekly_allowance := 10
def weeks := 8

-- Weekly savings calculation: Jack saves half of his weekly allowance
def weekly_savings := weekly_allowance / 2

-- Total savings over the given period
def total_savings := weekly_savings * weeks

-- Final amount in the piggy bank after the given period
def final_amount := initial_amount + total_savings

-- Theorem to prove: Final amount in the piggy bank after 8 weeks is $83.00
theorem jack_piggy_bank : final_amount = 83 := by
  sorry

end jack_piggy_bank_l642_642878


namespace cos_150_eq_neg_sqrt3_div_2_l642_642297

theorem cos_150_eq_neg_sqrt3_div_2 :
  (cos (150 * real.pi / 180)) = - (sqrt 3 / 2) := sorry

end cos_150_eq_neg_sqrt3_div_2_l642_642297


namespace part1_part2_l642_642409

-- Define the function f(x) = x^2 + a * x + 1
def f (x a : ℝ) : ℝ := x^2 + a * x + 1

-- Part 1: Solve the inequality f(x) > 0 for different ranges of a
theorem part1 (a x : ℝ) : 
(a > 2 ∨ a < -2 → (f x a > 0) ↔ (x ∈ (Iio ((-a - sqrt (a^2 - 4))/2) ∪ Ioi ((-a + sqrt (a^2 - 4))/2)))) ∧
(a = 2 → (f x a > 0) ↔ (x ∈ (Iio (-1) ∪ Ioi (-1)))) ∧
(a = -2 → (f x a > 0) ↔ (x ∈ (Iio 1 ∪ Ioi 1))) ∧
(-2 < a ∧ a < 2 → (f x a > 0) ∀ x, f x a > 0) :=
sorry

-- Part 2: When x > 0, f(x) ≥ 0 holds true identically. Find the range of values for a.
theorem part2 (a : ℝ) : 
(∀ x > 0, (f x a ≥ 0)) ↔ (a ∈ Ici (-2)) :=
sorry

end part1_part2_l642_642409


namespace cannot_connect_more_than_150_cities_l642_642842

variables (Cities : Type) (Airlines : Type)
variables [fintype Cities] [fintype Airlines]

variable (Flights : Cities → Cities → Airlines)

theorem cannot_connect_more_than_150_cities (hC : fintype.card Cities = 450)
  (hA : fintype.card Airlines = 6) :
  ¬ ∃ (a : Airlines) (S : finset Cities), S.card > 150 ∧ 
  ∀ (s1 s2 : Cities), s1 ∈ S → s2 ∈ S → 
  (exists (path : list Cities), path.head = s1 ∧ path.last = some s2 ∧ ∀ (c1 c2 : Cities), (c1, c2) ∈ path.zip path.tail → Flights c1 c2 = Flights s1 s2) :=
begin
  sorry
end

end cannot_connect_more_than_150_cities_l642_642842


namespace inequality_neg_multiplication_l642_642443

theorem inequality_neg_multiplication (m n : ℝ) (h : m > n) : -2 * m < -2 * n :=
by {
  sorry
}

end inequality_neg_multiplication_l642_642443


namespace quilt_longer_side_l642_642603

theorem quilt_longer_side (x : ℝ) : 
  (∀ A1 A2 : ℝ, A1 = 6 * x ∧ A2 = 12 * 12 → A1 = A2) → x = 24 := 
by 
  intro h
  have h_eq := h 6 * x (12 * 12)
  sorry

end quilt_longer_side_l642_642603


namespace age_problem_l642_642623

-- Define variables for ages
variables (a b c : ℕ)

-- Define the conditions as hypotheses
def proof_problem : Prop :=
  (b = 2 * c) ∧ (a = b + 2) ∧ (a + b + c = 47) → b = 18

-- The statement to be proven
theorem age_problem : proof_problem :=
begin
  -- Lean proof goes here but we use sorry for now
  sorry
end

end age_problem_l642_642623


namespace midpoint_trajectory_parallelogram_l642_642721

noncomputable def midpoint (p1 p2 : ℝ × ℝ) : ℝ × ℝ := 
  ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

def trajectory_is_rhombus : Prop :=
  ∃ z₁ z₂ z₃ z₄ : ℝ × ℝ, 
    -- conditions on the points forming a rhombus
    (z₁ ≠ z₂ ∧ z₂ ≠ z₃ ∧ z₃ ≠ z₄ ∧ z₄' ≠ z₁) ∧
    (midpoint z₁ z₃ = midpoint z₂ z₄) ∧ 
    (∃ d : ℝ, d > 0 ∧ 
      (dist z₁ z₂ = d ∧ dist z₂ z₃ = d ∧ dist z₃ z₄ = d ∧ dist z₄ z₁ = d)) ∧
    -- ensuring the motion description and trajectory requirements are met 
    true

theorem midpoint_trajectory_parallelogram (A B C D A' B' C' D' : ℝ × ℝ) 
  (h1 : A.1 = 0 ∧ A.2 = 0) 
  (h2 : B.1 = 1 ∧ B.2 = 0) 
  (h3 : C.1 = 1 ∧ C.2 = 1) 
  (h4 : D.1 = 0 ∧ D.2 = 1)
  (h5 : A'.1 = 0 ∧ A'.2 = 1) 
  (h6 : B'.1 = 1 ∧ B'.2 = 1)
  (h7 : C'.1 = 1 ∧ C'.2 = 2) 
  (h8 : D'.1 = 0 ∧ D'.2 = 2) 
  (hx : ∀ t ∈ [0,1], (∃ v : ℝ, (x (v t) = A + t * (B - A)) ∨ (x (v t) = B + t * (C - B)) ∨ ...)
  (hy : ∀ t ∈ [0,1], (∃ w : ℝ, (y (w t) = B' + t * (C' - B')) ∨ (y (w t) = C' + t * (C - C')) ∨ ...)
  (start_same_time : x 0 = A ∧ y 0 = B') :
  trajectory_is_rhombus :=
sorry

end midpoint_trajectory_parallelogram_l642_642721


namespace painting_falls_condition_l642_642681

-- Define the types of nails and conditions
inductive Nail
| Blue : ℕ → Nail
| Red : ℕ → Nail

open Nail

def BlueNails := [Blue 1, Blue 2, Blue 3]
def RedNails := [Red 4, Red 5, Red 6]

-- Define the sequence W
def sequence_A := [Blue 1, Blue 2, Blue 1, Blue 2, Blue 3, Blue 2, Blue 1, Blue 2, Blue 1, Blue 3]
def sequence_B := [Red 4, Red 5, Red 6, Red 4, Red 5, Red 6]

-- Defining inverses for the sequences
def inverse (seq : List (Nail)) : List (Nail) := seq.reverse

def W := sequence_A ++ sequence_B ++ inverse sequence_A ++ inverse sequence_B

-- Defining the condition for the painting to fall
def paintingFalls (nails: List Nail) (nail: Nail) (W: List Nail): Prop :=
  ∃ seq, seq.length > 0 ∧ (nail ∉ nails ++ inverse nails)

-- The theorem that needs to be proved in Lean
theorem painting_falls_condition :
  (∀ b ∈ BlueNails, paintingFalls (BlueNails ++ RedNails) b W) ∧
  (∀ r1 r2 ∈ RedNails, r1 ≠ r2 → paintingFalls (BlueNails ++ RedNails) r1 W ∧ paintingFalls (BlueNails ++ RedNails) r2 W) ∧
  (∀ r ∈ RedNails, ¬ paintingFalls (BlueNails ++ RedNails) r W) :=
by {
  sorry
}

end painting_falls_condition_l642_642681


namespace probability_between_lines_is_correct_l642_642813

noncomputable def probability_between_lines_in_1st_quadrant
  (l m : ℝ → ℝ) (l_eq : ∀ x, l x = -2 * x + 8) (m_eq : ∀ x, m x = -3 * x + 9) :
  ℝ :=
  let area_l := 1 / 2 * 4 * 8 in
  let area_m := 1 / 2 * 3 * 9 in
  let area_between := area_l - area_m in
  let probability := area_between / area_l in
  probability

theorem probability_between_lines_is_correct :
  probability_between_lines_in_1st_quadrant (λ x, -2 * x + 8) (λ x, -3 * x + 9)
    (by simp) (by simp) = 0.16 :=
  sorry

end probability_between_lines_is_correct_l642_642813


namespace ratio_proof_l642_642945

theorem ratio_proof (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : (4 * x + y) / (x - 4 * y) = 3) :
    (x + 4 * y) / (4 * x - y) = 9 / 53 :=
  sorry

end ratio_proof_l642_642945


namespace cos_150_eq_neg_sqrt3_div_2_l642_642302

open Real

theorem cos_150_eq_neg_sqrt3_div_2 :
  cos (150 * pi / 180) = - (sqrt 3 / 2) := 
sorry

end cos_150_eq_neg_sqrt3_div_2_l642_642302


namespace general_term_sum_first_n_terms_b_monotonically_increasing_range_l642_642388

-- Definitions and conditions
def is_arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

def Sn (a : ℕ → ℤ) (n : ℕ) : ℤ := (finset.range (n + 1)).sum a

-- Problem 1: Prove the general term formula of the arithmetic sequence
theorem general_term (a : ℕ → ℤ) (S : ℕ → ℤ)
  (h_arithmetic : is_arithmetic_sequence a)
  (h_first_term : a 1 = -9)
  (h_second_term_integer : ∃ k : ℤ, a 2 = k)
  (h_sum_condition : ∀ n : ℕ, Sn a n ≥ Sn a 5) :
  ∀ n : ℕ, a n = 2 * n - 11 :=
sorry

-- Problem 2: Prove the sum of the first n terms of {b_n}
def b (a : ℕ → ℤ) (n : ℕ) : ℚ :=
  if n = 0 then 4 / 3
  else if n % 2 = 1 then a (n / 2)
  else -b a (n - 1) + (-2) ^ n

def Tn (b : ℕ → ℚ) (n : ℕ) : ℚ := (finset.range n).sum b

theorem sum_first_n_terms_b (a : ℕ → ℤ) (b : ℕ → ℚ)
  (h_b1 : b 0 = 4 / 3)
  (h_b_def : ∀ n, b (n + 1) = if n % 2 = 1 then a (n / 2) else -b n + (-2) ^ (n + 1)) :
  ∀ n : ℕ, Tn b n = 
    if n % 2 = 1 then 2 ^ (n + 1) / 3
    else 2 ^ n / 3 + 2 * n - 13 :=
sorry

-- Problem 3: Prove the range of lambda for the sequence {c_n} to be monotonically increasing
def c (b : ℕ → ℚ) (a : ℕ → ℤ) (λ : ℝ) (n : ℕ) : ℝ :=
  b (2 * n) + b (2 * n + 1) + λ * (-1) ^ n * (1 / 2) ^ (a n + 5)

theorem monotonically_increasing_range (b : ℕ → ℚ) (a : ℕ → ℤ)
  (h_b1 : b 0 = 4 / 3)
  (h_b_def : ∀ n, b (n + 1) = if n % 2 = 1 then a (n / 2) else -b n + (-2) ^ (n + 1))
  (h_a : ∀ n : ℕ, a n = 2 * n - 11) :
  ∃ λ : ℝ, ∀ n : ℕ, c b a λ (n + 1) > c b a λ n ↔ λ ∈ set.Ioo (-3 / 5) (48 / 5) :=
sorry

end general_term_sum_first_n_terms_b_monotonically_increasing_range_l642_642388


namespace cos_150_eq_neg_half_l642_642268

theorem cos_150_eq_neg_half : Real.cos (150 * Real.pi / 180) = -1 / 2 := 
by
  sorry

end cos_150_eq_neg_half_l642_642268


namespace smarties_modulo_l642_642351

theorem smarties_modulo (m : ℕ) (h : m % 11 = 5) : (4 * m) % 11 = 9 :=
by
  sorry

end smarties_modulo_l642_642351


namespace sequence_not_distinct_l642_642510

-- We will state the theorem

theorem sequence_not_distinct (a0 : ℕ) (h : 0 < a0) : 
  ∃ (n m : ℕ), n ≠ m ∧ 
   (let a : ℕ → ℕ := λ b, 
      let ⟨digits, _⟩ := nat.digits 10 b in
      digits.sum (λ c, c ^ 2005)
    in a^[n] a0 = a^[m] a0) :=
begin
  -- The proof will be written here
  sorry
end

end sequence_not_distinct_l642_642510


namespace count_irrational_numbers_l642_642683

theorem count_irrational_numbers : 
  let numbers := [22 / 7, -3.5, 0, Real.sqrt 8, Real.pi, 0.1010010001...]
  let irrational_count := (numbers.filter (λ x, ¬ ∃ (a b : ℤ), b ≠ 0 ∧ x = a / b)).length
  irrational_count = 3 :=
by {
  sorry
}

end count_irrational_numbers_l642_642683


namespace find_val_of_f_of_f_neg1_l642_642407

def f (x : ℝ) : ℝ :=
if x < 0 then 2^(-x)
else Real.log2(x + 6)

theorem find_val_of_f_of_f_neg1 : f (f (-1)) = 3 := 
by {
  sorry
}

end find_val_of_f_of_f_neg1_l642_642407


namespace multiplication_correct_l642_642695

theorem multiplication_correct (a b c d e f: ℤ) (h₁: a * b = c) (h₂: d * e = f): 
    (63 * 14 = c) → (68 * 14 = f) → c = 882 ∧ f = 952 :=
by sorry

end multiplication_correct_l642_642695


namespace cos_150_eq_negative_cos_30_l642_642147

theorem cos_150_eq_negative_cos_30 :
  Real.cos (150 * Real.pi / 180) = - (Real.cos (30 * Real.pi / 180)) :=
by sorry

end cos_150_eq_negative_cos_30_l642_642147


namespace find_a_l642_642399

variable (a : ℝ)

def setA := {1, -2, a^2 - 1}
def setB := {1, a^2 - 3a, 0}

theorem find_a (h : setA a = setB a) : a = 1 :=
  sorry

end find_a_l642_642399


namespace common_year_has_52_weeks_1_day_leap_year_has_52_weeks_2_days_next_year_starts_on_wednesday_next_year_starts_on_thursday_l642_642073

-- a) Prove the statements about the number of weeks and extra days
theorem common_year_has_52_weeks_1_day: 
  ∀ (days_in_common_year : ℕ), 
  days_in_common_year = 365 → 
  (days_in_common_year / 7 = 52 ∧ days_in_common_year % 7 = 1)
:= by
  sorry

theorem leap_year_has_52_weeks_2_days: 
  ∀ (days_in_leap_year : ℕ), 
  days_in_leap_year = 366 → 
  (days_in_leap_year / 7 = 52 ∧ days_in_leap_year % 7 = 2)
:= by
  sorry

-- b) If a common year starts on a Tuesday, prove the following year starts on a Wednesday
theorem next_year_starts_on_wednesday: 
  ∀ (start_day : ℕ), 
  start_day = 2 ∧ (365 % 7 = 1) → 
  ((start_day + 365 % 7) % 7 = 3)
:= by
  sorry

-- c) If a leap year starts on a Tuesday, prove the following year starts on a Thursday
theorem next_year_starts_on_thursday: 
  ∀ (start_day : ℕ), 
  start_day = 2 ∧ (366 % 7 = 2) →
  ((start_day + 366 % 7) % 7 = 4)
:= by
  sorry

end common_year_has_52_weeks_1_day_leap_year_has_52_weeks_2_days_next_year_starts_on_wednesday_next_year_starts_on_thursday_l642_642073


namespace second_player_wins_l642_642923

-- Define the initial condition of the game
def initial_coins : Nat := 2016

-- Define the set of moves a player can make
def valid_moves : Finset Nat := {1, 2, 3}

-- Define the winning condition
def winning_player (coins : Nat) : String :=
  if coins % 4 = 0 then "second player"
  else "first player"

-- The theorem stating that second player has a winning strategy given the initial condition
theorem second_player_wins : winning_player initial_coins = "second player" :=
by
  sorry

end second_player_wins_l642_642923


namespace cos_150_eq_neg_half_l642_642270

theorem cos_150_eq_neg_half : Real.cos (150 * Real.pi / 180) = -1 / 2 := 
by
  sorry

end cos_150_eq_neg_half_l642_642270


namespace nested_sqrt_eq_l642_642741

theorem nested_sqrt_eq :
  let y := sqrt (18 + sqrt (18 + sqrt (18 + sqrt (18 + ...))))
  in y = (1 + sqrt 73) / 2 :=
by sorry

end nested_sqrt_eq_l642_642741


namespace cosine_150_eq_neg_sqrt3_div_2_l642_642249

theorem cosine_150_eq_neg_sqrt3_div_2 :
  (∀ θ : ℝ, θ = real.pi / 6) →
  (∀ θ : ℝ, cos (real.pi - θ) = -cos θ) →
  cos (5 * real.pi / 6) = -real.sqrt 3 / 2 :=
by
  intros hθ hcos_identity
  sorry

end cosine_150_eq_neg_sqrt3_div_2_l642_642249


namespace probability_range_l642_642096

noncomputable def c : ℝ := 5 / 4
def P (k : ℕ) : ℝ := if k ∈ {1, 2, 3, 4} then c / (k * (k + 1)) else 0

theorem probability_range : P (½ < ξ < 5/2) = 5 / 6 :=
by
  sorry

end probability_range_l642_642096


namespace container_fill_fraction_correct_l642_642063

/-- The capacity of the container in liters. --/
def container_capacity : ℝ := 80

/-- The container is initially 50% full, i.e., it contains 40 liters of water initially. --/
def initial_water_volume : ℝ := container_capacity * 0.5

/-- The amount of water added to the container. --/
def added_water_volume : ℝ := 20

/-- The total water volume in the container after adding the water. --/
def total_water_volume : ℝ := initial_water_volume + added_water_volume

/-- The fraction of the container that is full after adding the water. --/
def fill_fraction : ℝ := total_water_volume / container_capacity

/--
Proof that the fraction of the container that is full after adding the 
20 liters of water is 3/4.
--/
theorem container_fill_fraction_correct : fill_fraction = 3 / 4 := by
  sorry

end container_fill_fraction_correct_l642_642063


namespace cosine_150_eq_neg_sqrt3_div_2_l642_642237

theorem cosine_150_eq_neg_sqrt3_div_2 :
  (∀ θ : ℝ, θ = real.pi / 6) →
  (∀ θ : ℝ, cos (real.pi - θ) = -cos θ) →
  cos (5 * real.pi / 6) = -real.sqrt 3 / 2 :=
by
  intros hθ hcos_identity
  sorry

end cosine_150_eq_neg_sqrt3_div_2_l642_642237


namespace cos_150_eq_neg_half_l642_642189

theorem cos_150_eq_neg_half :
  real.cos (150 * real.pi / 180) = -1 / 2 :=
sorry

end cos_150_eq_neg_half_l642_642189


namespace ten_term_sequence_l642_642057
open Real

theorem ten_term_sequence (a b : ℝ) 
    (h₁ : a + b = 1)
    (h₂ : a^2 + b^2 = 3)
    (h₃ : a^3 + b^3 = 4)
    (h₄ : a^4 + b^4 = 7)
    (h₅ : a^5 + b^5 = 11) :
    a^10 + b^10 = 123 :=
  sorry

end ten_term_sequence_l642_642057


namespace indistinguishable_balls_ways_l642_642434

theorem indistinguishable_balls_ways :
  ∃ ways, ways = 11 ∧ ways = finset.card { p : multiset ℕ // multiset.card p ≤ 4 ∧ multiset.sum p = 7 } :=
by
  use 11
  split
  . refl
  . sorry

end indistinguishable_balls_ways_l642_642434


namespace side_length_S2_l642_642074

theorem side_length_S2 (r s : ℕ) (h1 : 2 * r + s = 2260) (h2 : 2 * r + 3 * s = 3782) : s = 761 :=
by
  -- proof omitted
  sorry

end side_length_S2_l642_642074


namespace boys_to_girls_ratio_l642_642043

theorem boys_to_girls_ratio (S G B : ℕ) (h1 : 1 / 2 * G = 1 / 3 * S) (h2 : S = B + G) : B / G = 1 / 2 :=
by
  -- Placeholder for the actual proof
  sorry

end boys_to_girls_ratio_l642_642043


namespace square_roots_equal_49_l642_642455

theorem square_roots_equal_49 (x a : ℝ) (hx1 : (2 * x - 3)^2 = a) (hx2 : (5 - x)^2 = a) (ha_pos: a > 0) : a = 49 := 
by 
  sorry

end square_roots_equal_49_l642_642455


namespace unique_solution_for_system_l642_642760

theorem unique_solution_for_system
  (a : ℝ)
  (h : a = ℝ.arcsin ((4 * real.sqrt 2 + 1)/7) + ((2 * k - 1) * Real.pi / 4) ∨ 
      a = - ℝ.arcsin ((4 * real.sqrt 2 + 1)/7) + ((2 * k - 1) * Real.pi / 4))
  (k : ℤ) :
  ∀ (x y : ℝ), 
  (x - 7 * real.cos a) ^ 2 + (y - 7 * real.sin a) ^ 2 = 1 ∧ 
  abs x + abs y = 8 → 
    x = 7 * real.cos a + 1 ∧ y = 7 * real.sin a + abs ((4√2 + 1)/7 - 7 * real.sin a) ∨ 
    x = 7 * real.cos a - 1 ∧ y = 7 * real.sin a - abs ((4√2 + 1)/7 - 7 * real.sin a) :=
sorry


end unique_solution_for_system_l642_642760


namespace part_1_part_2_l642_642802

noncomputable def f (a x : ℝ) : ℝ := a^x + x^2 + cos x - x * log a

-- Conditions
variable {a : ℝ}
variable (h1 : 0 < a)
variable (h2 : a ≠ 1)

-- Part (1): Analyze the monotonicity of f(x)
theorem part_1 (x : ℝ) : 
  (x < 0 → f a x is_monotonically_decreasing) ∧ 
  (x > 0 → f a x is_monotonically_increasing) :=
by 
  sorry

-- Part (2): Find the range of values for a given condition on f(x)
theorem part_2 (x1 x2 : ℝ) (h3 : x1 ∈ Icc (-1 : ℝ) 1) (h4 : x2 ∈ Icc (-1 : ℝ) 1) 
  (h5 : |f a x1 - f a x2| ≥ cos 1 + exp 1 - 2) : 
    a ∈ Ioc 0 (1/exp 1) ∨ a ∈ Ici (exp 1) :=
by 
  sorry

end part_1_part_2_l642_642802


namespace cos_150_eq_neg_sqrt3_div_2_l642_642124

theorem cos_150_eq_neg_sqrt3_div_2 :
  ∃ θ : ℝ, θ = 150 ∧
           0 < θ ∧ θ < 180 ∧
           ∃ φ : ℝ, φ = 180 - θ ∧
           cos φ = Real.sqrt 3 / 2 ∧
           cos θ = - cos φ :=
  sorry

end cos_150_eq_neg_sqrt3_div_2_l642_642124


namespace isabel_piggy_bank_l642_642497

theorem isabel_piggy_bank:
  ∀ (initial_amount spent_on_toy spent_on_book remaining_amount : ℕ),
  initial_amount = 204 →
  spent_on_toy = initial_amount / 2 →
  remaining_amount = initial_amount - spent_on_toy →
  spent_on_book = remaining_amount / 2 →
  remaining_amount - spent_on_book = 51 :=
by
  sorry

end isabel_piggy_bank_l642_642497


namespace find_number_l642_642942

theorem find_number :
  ∃ x : ℝ, (x - 1.9) * 1.5 + 32 / 2.5 = 20 ∧ x = 13.9 :=
by
  sorry

end find_number_l642_642942


namespace cos_150_eq_neg_sqrt3_div_2_l642_642169

theorem cos_150_eq_neg_sqrt3_div_2 : Real.cos (150 * Real.pi / 180) = - (Real.sqrt 3 / 2) :=
by
  sorry

end cos_150_eq_neg_sqrt3_div_2_l642_642169


namespace cos_150_degree_l642_642176

theorem cos_150_degree : Real.cos (150 * Real.pi / 180) = -Real.sqrt 3 / 2 :=
by
  sorry

end cos_150_degree_l642_642176


namespace estimated_population_value_correlation_coefficient_value_l642_642102

section AnimalPopulation

open Real

variables {n : ℕ}
variables (xi yi : Fin n → ℝ)
variables (n = 20) (xi yi)

definition sum_xi : ℝ := ∑ i in Finset.range 20, xi i
definition sum_yi : ℝ := ∑ i in Finset.range 20, yi i
definition mean_x : ℝ := (sum_xi xi) / 20
definition mean_y : ℝ := (sum_yi yi) / 20
definition sum_sq_diff_x : ℝ := ∑ i in Finset.range 20, (xi i - mean_x xi) ^ 2
definition sum_sq_diff_y : ℝ := ∑ i in Finset.range 20, (yi i - mean_y yi) ^ 2
definition sum_xy_diff : ℝ := ∑ i in Finset.range 20, (xi i - mean_x xi) * (yi i - mean_y yi)

-- Given conditions
axiom sum_xi_eq : sum_xi xi = 60
axiom sum_yi_eq : sum_yi yi = 1200
axiom sum_sq_diff_x_eq : sum_sq_diff_x xi = 80
axiom sum_sq_diff_y_eq : sum_sq_diff_y yi = 9000
axiom sum_xy_diff_eq : sum_xy_diff xi yi = 800

noncomputable def estimated_population (num_plots : ℝ) : ℝ := mean_y yi * num_plots

theorem estimated_population_value :
  estimated_population yi 200 = 12000 :=
by 
  have h1 : mean_y yi = 60 := by 
    rw [mean_y, sum_yi_eq]
    norm_num
  rw [estimated_population, h1]
  norm_num

noncomputable def correlation_coefficient : ℝ :=
  sum_xy_diff xi yi / sqrt (sum_sq_diff_x xi * sum_sq_diff_y yi)

theorem correlation_coefficient_value :
  abs (correlation_coefficient xi yi - 0.94) < 0.01 :=
by 
  have h2 : correlation_coefficient xi yi 
           = 800 / (sqrt (80 * 9000)) := by 
    rw [correlation_coefficient, sum_xy_diff_eq, sum_sq_diff_x_eq, sum_sq_diff_y_eq]
  rw [h2]
  have h3 : 800 / (600 * sqrt 2) 
            ≈ 0.94 := sorry  -- This part involves numerical approximations
  simp [h3]
  norm_num

end AnimalPopulation

end estimated_population_value_correlation_coefficient_value_l642_642102


namespace probability_of_two_absent_one_present_is_correct_l642_642845

noncomputable def probability_two_absent_one_present (absent_per_days_total : ℚ) (total_days : ℚ) : ℚ :=
  let p_absent := absent_per_days_total / total_days in
  let p_present := 1 - p_absent in
  (3 * (p_present * p_absent^2))

theorem probability_of_two_absent_one_present_is_correct :
  probability_two_absent_one_present 2 25 * 100 = 1.8 :=
by
  sorry

end probability_of_two_absent_one_present_is_correct_l642_642845


namespace four_element_subsets_count_l642_642509

open Finset

noncomputable def X : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}

noncomputable def A : Finset ℕ := {1, 2, 3, 4}

theorem four_element_subsets_count :
  (Finset.card {Y : Finset ℕ | Y ⊆ X ∧ Y.card = 4 ∧ 10 ∈ Y ∧ Y ∩ A ≠ ∅}) = 74 := 
by {
  sorry
}

end four_element_subsets_count_l642_642509


namespace roger_shelves_l642_642550

theorem roger_shelves (total_books : ℕ) (books_taken : ℕ) (books_per_shelf : ℕ) : 
  total_books = 24 → 
  books_taken = 3 → 
  books_per_shelf = 4 → 
  Nat.ceil ((total_books - books_taken) / books_per_shelf) = 6 :=
by
  intros h_total h_taken h_per_shelf
  rw [h_total, h_taken, h_per_shelf]
  sorry

end roger_shelves_l642_642550


namespace domain_sqrt_2_minus_x_l642_642342

def y (x : ℝ) := real.sqrt (2 - x)

theorem domain_sqrt_2_minus_x :
  { x : ℝ | 2 - x ≥ 0 } = { x : ℝ | x ≤ 2 } :=
by
  sorry

end domain_sqrt_2_minus_x_l642_642342


namespace range_of_eccentricity_l642_642778

theorem range_of_eccentricity (a b : ℝ) (h1 : a > b) (h2 : b > 0) (x y : ℝ) 
  (h_ellipse : x^2 / a^2 + y^2 / b^2 = 1) (c : ℝ := Real.sqrt (a^2 - b^2)) 
  (h_dot_product : ∀ (x y: ℝ) (h_point : x^2 / a^2 + y^2 / b^2 = 1), 
    let PF1 : ℝ × ℝ := (-c - x, -y)
    let PF2 : ℝ × ℝ := (c - x, -y)
    PF1.1 * PF2.1 + PF1.2 * PF2.2 ≤ a * c) : 
  ∀ (e : ℝ := c / a), (Real.sqrt 5 - 1) / 2 ≤ e ∧ e < 1 := 
by 
  sorry

end range_of_eccentricity_l642_642778


namespace frank_composes_problems_l642_642693

theorem frank_composes_problems (bill_problems : ℕ) (ryan_problems : ℕ) (frank_problems : ℕ) 
  (h1 : bill_problems = 20)
  (h2 : ryan_problems = 2 * bill_problems)
  (h3 : frank_problems = 3 * ryan_problems)
  : frank_problems / 4 = 30 :=
by
  sorry

end frank_composes_problems_l642_642693


namespace polynomial_coeff_sum_l642_642438

theorem polynomial_coeff_sum :
  (∀ (x : ℂ), (1 - 2 * x) ^ 7 = ∑ (n : ℕ) in finset.range 8, (a n) * x ^ n) →
  (∑ (n : ℕ) in finset.range 8, a n) = -1 :=
by
  sorry

end polynomial_coeff_sum_l642_642438


namespace cos_150_eq_neg_sqrt3_over_2_l642_642217

theorem cos_150_eq_neg_sqrt3_over_2 :
  ∀ {deg cosQ1 cosQ2: ℝ},
    cosQ1 = 30 →
    cosQ2 = 180 - cosQ1 →
    cos 150 = - (cos cosQ1) →
    cos cosQ1 = sqrt 3 / 2 →
    cos 150 = -sqrt 3 / 2 :=
by
  intros deg cosQ1 cosQ2 h1 h2 h3 h4
  sorry

end cos_150_eq_neg_sqrt3_over_2_l642_642217


namespace symmetric_points_sum_l642_642788

theorem symmetric_points_sum (a b : ℝ) (hA1 : A = (a, 1)) (hB1 : B = (5, b))
    (h_symmetric : (a, 1) = -(5, b)) : a + b = -6 :=
by
  sorry

end symmetric_points_sum_l642_642788


namespace visible_factor_numbers_200_to_250_l642_642659

def is_visible_factor_number (n : ℕ) : Prop :=
  let digits := List.filter_map (fun c => if c = '0' then none else some (c.to_nat - '0'.to_nat)) (n.to_string.data)
  ∀ (d : ℕ), d ∈ digits → d ∣ n

def count_visible_factor_numbers (a b : ℕ) : ℕ :=
  (List.range' a (b - a + 1)).filter (is_visible_factor_number).length

theorem visible_factor_numbers_200_to_250 : count_visible_factor_numbers 200 250 = 22 := 
by
  sorry

end visible_factor_numbers_200_to_250_l642_642659


namespace APB_area_l642_642673

-- Definition of the points and distances
def point := (ℝ × ℝ)
def A : point := (0, 0)
def B : point := (8, 0)
def D : point := (4, 0)
def P : point := (4, 4)

lemma midpoint_AB : (A.1 + B.1) / 2 = D.1 :=
by sorry

lemma PD_perpendicular_to_AB : P.2 - D.2 = 4 ∧ P.1 = D.1 :=
by sorry

lemma distances_equal : 
  let PA := (P.1 - A.1)^2 + (P.2 - A.2)^2,
      PB := (P.1 - B.1)^2 + (P.2 - B.2)^2,
      PD := (P.1 - D.1)^2 + (P.2 - D.2)^2 in
  PA = PB ∧ PB = PD :=
by sorry

lemma area_triangle_APB : 
  let base := 8,
      height := 4 in
  (1 / 2) * base * height = 16 :=
by sorry

theorem APB_area : 
  let A := (0, 0),
      B := (8, 0),
      D := (4, 0),
      P := (4, 4) in
  ∃ (area : ℝ), area = 16 :=
by 
  use (1 / 2) * 8 * 4
  show (1 / 2) * 8 * 4 = 16
  by sorry

end APB_area_l642_642673


namespace paper_boat_time_proof_l642_642478

/-- A 50-meter long embankment exists along a river.
 - A motorboat that passes this embankment in 5 seconds while moving downstream.
 - The same motorboat passes this embankment in 4 seconds while moving upstream.
 - Determine the time in seconds it takes for a paper boat, which moves with the current, to travel the length of this embankment.
 -/
noncomputable def paper_boat_travel_time 
  (embankment_length : ℝ)
  (motorboat_length : ℝ)
  (time_downstream : ℝ)
  (time_upstream : ℝ) : ℝ :=
  let v_eff_downstream := embankment_length / time_downstream,
      v_eff_upstream := embankment_length / time_upstream,
      v_boat := (v_eff_downstream + v_eff_upstream) / 2,
      v_current := (v_eff_downstream - v_eff_upstream) / 2 in
  embankment_length / v_current

theorem paper_boat_time_proof :
  paper_boat_travel_time 50 10 5 4 = 40 := 
begin
  sorry,
end

end paper_boat_time_proof_l642_642478


namespace cos_150_eq_neg_half_l642_642193

theorem cos_150_eq_neg_half :
  real.cos (150 * real.pi / 180) = -1 / 2 :=
sorry

end cos_150_eq_neg_half_l642_642193


namespace cos_150_deg_l642_642330

-- Define the conditions as variables
def angle1 := 150
def angle2 := 180
def angle3 := 30
def cos30 := Real.sqrt 3 / 2

-- The main statement to be proved
theorem cos_150_deg : Real.cos (150 * Real.pi / 180) = - (Real.sqrt 3 / 2) :=
by
  sorry

end cos_150_deg_l642_642330


namespace proof_problem_l642_642902

def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

def given_function (f : ℝ → ℝ) : Prop :=
  odd_function f ∧ f (-3) = -2

theorem proof_problem (f : ℝ → ℝ) (h : given_function f) : f 3 + f 0 = -2 :=
by sorry

end proof_problem_l642_642902


namespace infinitely_many_numbers_with_finitely_many_primes_l642_642691

theorem infinitely_many_numbers_with_finitely_many_primes (a : ℕ → ℕ) :
  (∀ i : ℕ, ∃ ai : ℕ, a i = ai) ∧
  (∀ k : ℤ, set.finite {i | nat.prime (a i + int.to_nat k)}) ↔
  ∃ S : set ℕ, set.infinite S ∧ (∀ i ∈ S, a i ∈ ℕ) :=
by sorry

end infinitely_many_numbers_with_finitely_many_primes_l642_642691


namespace fish_caught_in_second_catch_l642_642459

theorem fish_caught_in_second_catch {N x : ℕ} (hN : N = 1750) (hx1 : 70 * x = 2 * N) : x = 50 :=
by
  sorry

end fish_caught_in_second_catch_l642_642459


namespace brett_total_miles_l642_642694

def miles_per_hour : ℕ := 75
def hours_driven : ℕ := 12

theorem brett_total_miles : miles_per_hour * hours_driven = 900 := 
by 
  sorry

end brett_total_miles_l642_642694


namespace cos_150_deg_l642_642326

-- Define the conditions as variables
def angle1 := 150
def angle2 := 180
def angle3 := 30
def cos30 := Real.sqrt 3 / 2

-- The main statement to be proved
theorem cos_150_deg : Real.cos (150 * Real.pi / 180) = - (Real.sqrt 3 / 2) :=
by
  sorry

end cos_150_deg_l642_642326


namespace red_paper_larger_than_smaller_part_l642_642953

theorem red_paper_larger_than_smaller_part :
  ∀ (x : ℝ),
    (yell_paper_area = 2 * x) ∧
    (larger_part = (4 / 3) * x) ∧
    (smaller_part = 2 * x - larger_part) →
    ∃ (percentage : ℝ), percentage = 50 ∧
    (x - smaller_part) / smaller_part * 100 = percentage :=
by
  intros x hyp,
  cases hyp with hp yol_paper_area_eq,
  cases hp with hp larger_part_eq,
  cases hyp with sc smaller_part_eq,
  use 50,
  sorry

end red_paper_larger_than_smaller_part_l642_642953


namespace solution_set_l642_642888

noncomputable def f : ℝ → ℝ := sorry
noncomputable def f' (x : ℝ) : ℝ := sorry

axiom f_def (x : ℝ) : f(x) + f'(x) > 1
axiom f_at_zero : f(0) = 2017

theorem solution_set (x : ℝ) : (e ^ x * f(x) > e ^ x + 2016) ↔ (x > 0) :=
by
  sorry

end solution_set_l642_642888


namespace angle_bisector_median_inequality_l642_642928

theorem angle_bisector_median_inequality 
  (A B C : Type) [EuclideanGeometry A B C] 
  (l_a l_b l_c m_a m_b m_c : ℝ) 
  (h₁ : IsAngleBisector l_a A B C)
  (h₂ : IsAngleBisector l_b B C A)
  (h₃ : IsAngleBisector l_c C A B)
  (h₄ : IsMedian m_a A B C)
  (h₅ : IsMedian m_b B C A)
  (h₆ : IsMedian m_c C A B) : 
  (l_a / m_a + l_b / m_b + l_c / m_c) > 1 :=
sorry

end angle_bisector_median_inequality_l642_642928


namespace minimum_students_in_class_l642_642457

def min_number_of_students (b g : ℕ) : ℕ :=
  b + g

theorem minimum_students_in_class
  (b g : ℕ)
  (h1 : b = 2 * g / 3)
  (h2 : ∃ k : ℕ, g = 3 * k)
  (h3 : ∃ k : ℕ, 1 / 2 < (2 / 3) * g / b) :
  min_number_of_students b g = 5 :=
sorry

end minimum_students_in_class_l642_642457


namespace cos_150_eq_neg_sqrt3_div_2_l642_642314

open Real

theorem cos_150_eq_neg_sqrt3_div_2 :
  cos (150 * pi / 180) = - (sqrt 3 / 2) := 
sorry

end cos_150_eq_neg_sqrt3_div_2_l642_642314


namespace Sam_bags_filled_l642_642551

theorem Sam_bags_filled (total_cans bags_Sunday cans_per_bag : ℕ) (H1 : total_cans = 42)
  (H2 : bags_Sunday = 3) (H3 : cans_per_bag = 6) : 
  let bags_Saturday := (total_cans - (bags_Sunday * cans_per_bag)) / cans_per_bag in
  bags_Saturday = 4 :=
by
  intros total_cans bags_Sunday cans_per_bag H1 H2 H3
  let bags_Saturday := (total_cans - (bags_Sunday * cans_per_bag)) / cans_per_bag
  show bags_Saturday = 4
  sorry

end Sam_bags_filled_l642_642551


namespace coefficient_xy2_in_expansion_l642_642358
noncomputable theory

def binomial_coefficient (n k : ℕ) : ℕ :=
  nat.choose n k

theorem coefficient_xy2_in_expansion :
  let expr := (1 + x)^6 * (1 + y)^4 in
    (binomial_coefficient 6 1) * (binomial_coefficient 4 2) = 36 :=
by
  sorry

end coefficient_xy2_in_expansion_l642_642358


namespace graph_through_point_l642_642796

variable {α : Type*} [Field α]

-- Assumptions based on given conditions
variable (f : α → α)
variable (f_inv : α → α)
variable (h_inv : ∀ x, f (f_inv x) = x ∧ f_inv (f x) = x)
variable (h_graph : f 1 - 1 = 2)

theorem graph_through_point :
  ∃ y, y = (1 : α)/3 + f_inv 3 ∧ y = 4/3 :=
by
  -- Using the given information without proof steps, just hypotheses and final conclusion
  have h1 : f 1 = 3 :=
    by
      calc
        f 1 = 2 + 1 := by rw [sub_eq_add_neg, add_comm 2 (-1), neg_add_cancel_right]
             ... = 3   := by norm_num
  have h2 : f_inv 3 = 1 :=
    by rw [←h_inv 1, h1]
  use (1 : α)/3 + f_inv 3
  split
  · rfl
  · rw [h2, div_eq_mul_one_div, one_div, mul_one]

sörry

end graph_through_point_l642_642796


namespace smallest_m_value_l642_642985

def original_function (x : ℝ) : ℝ :=
  (sqrt 3) * cos x - sin x

def transformed_function (x m : ℝ) : ℝ :=
  (sqrt 3) * cos (x + m) - sin (x + m)

def is_symmetric_about_y_axis (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

theorem smallest_m_value :
  (m : ℝ) (h : 0 < m) (hm : is_symmetric_about_y_axis (transformed_function · m)) :
    m = (5 * pi) / 6 :=
sorry

end smallest_m_value_l642_642985


namespace symmedian_from_A_l642_642885

-- Definitions based on conditions
variables {A B C S : Type*}
variables [is_triangle ABC]
variables [is_direct_similarity_center S A B C]

-- Problem statement
theorem symmedian_from_A (A B C S : Type*) [is_triangle ABC] [is_direct_similarity_center S A B C] :
  is_symmedian_from A ABC (A S) :=
sorry

end symmedian_from_A_l642_642885


namespace length_AE_l642_642726

theorem length_AE (AB CD AC AE ratio : ℝ) 
  (h_AB : AB = 10) 
  (h_CD : CD = 15) 
  (h_AC : AC = 18) 
  (h_ratio : ratio = 2 / 3) 
  (h_areas : ∀ (areas : ℝ), areas = 2 / 3)
  : AE = 7.2 := 
sorry

end length_AE_l642_642726


namespace total_birds_from_monday_to_wednesday_l642_642064

def birds_monday := 70
def birds_tuesday := birds_monday / 2
def birds_wednesday := birds_tuesday + 8
def total_birds := birds_monday + birds_tuesday + birds_wednesday

theorem total_birds_from_monday_to_wednesday : total_birds = 148 :=
by
  -- sorry is used here to skip the actual proof
  sorry

end total_birds_from_monday_to_wednesday_l642_642064


namespace paper_boat_travel_time_l642_642483

-- Defining the conditions as constants
def distance_embankment : ℝ := 50
def speed_downstream : ℝ := 10
def speed_upstream : ℝ := 12.5

-- Definitions for the speeds of the boat and current
noncomputable def v_boat : ℝ := (speed_upstream + speed_downstream) / 2
noncomputable def v_current : ℝ := (speed_downstream - speed_upstream) / 2

-- Statement to prove the time taken for the paper boat
theorem paper_boat_travel_time :
  (distance_embankment / v_current) = 40 := by
  sorry

end paper_boat_travel_time_l642_642483


namespace range_f_range_a_l642_642779

noncomputable def f (x: ℝ) : ℝ := x + 4 / x
noncomputable def g (x a: ℝ) : ℝ := 2^x + a

theorem range_f : set.range (λ x, f x) ∩ set.Icc (5 : ℝ) (17/2 : ℝ) = set.Icc (5 : ℝ) (17/2 : ℝ) :=
sorry

theorem range_a (a : ℝ) : (∀ x1 ∈ set.Icc (1 / 2) 1, ∃ x2 ∈ set.Icc 2 3, f x1 ≥ g x2 a) ↔ a ≤ 1 :=
sorry

end range_f_range_a_l642_642779


namespace luke_trays_l642_642530

theorem luke_trays 
  (carries_per_trip : ℕ)
  (trips : ℕ)
  (second_table_trays : ℕ)
  (total_trays : carries_per_trip * trips = 36)
  (second_table_value : second_table_trays = 16) : 
  carries_per_trip * trips - second_table_trays = 20 :=
by sorry

end luke_trays_l642_642530


namespace sum_geq_sqrt3_l642_642764

theorem sum_geq_sqrt3 (a b c : ℝ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0) (h_eq : a * b + b * c + c * a = 1) : a + b + c ≥ real.sqrt 3 :=
sorry

end sum_geq_sqrt3_l642_642764


namespace gas_station_distances_correct_l642_642083

open Real

-- Let the current distances be defined as follows:
def current_distance_A := 2.34 -- in km
def current_distance_B := 2.16 -- in km

-- First proposal condition
def first_proposal_condition := (current_distance_A / 6 = current_distance_B / 8 + 0.12)

-- Second proposal condition
def second_proposal_condition := (current_distance_B / 6 = current_distance_A / 7 + 0.18)

theorem gas_station_distances_correct : 
  first_proposal_condition ∧ second_proposal_condition := 
by 
  show first_proposal_condition ∧ second_proposal_condition,
  sorry -- proof goes here

end gas_station_distances_correct_l642_642083


namespace largest_number_is_870_l642_642596

-- Define the set of digits {8, 7, 0}
def digits : Set ℕ := {8, 7, 0}

-- Define the largest number that can be made by arranging these digits
def largest_number (s : Set ℕ) : ℕ := 870

-- Statement to prove
theorem largest_number_is_870 : largest_number digits = 870 :=
by
  -- Proof is omitted
  sorry

end largest_number_is_870_l642_642596


namespace stop_shooting_after_2nd_scoring_5_points_eq_l642_642738

/-
Define the conditions and problem statement in Lean:
- Each person can shoot up to 10 times.
- Student A's shooting probability for each shot is 2/3.
- If student A stops shooting at the nth consecutive shot, they score 12-n points.
- We need to prove the probability that student A stops shooting right after the 2nd shot and scores 5 points is 8/729.
-/
def student_shoot_probability (shots : List Bool) (p : ℚ) : ℚ :=
  shots.foldr (λ s acc => if s then p * acc else (1 - p) * acc) 1

def stop_shooting_probability : ℚ :=
  let shots : List Bool := [false, true, false, false, false, true, true] -- represents misses and hits
  student_shoot_probability shots (2/3)

theorem stop_shooting_after_2nd_scoring_5_points_eq :
  stop_shooting_probability = (8 / 729) :=
sorry

end stop_shooting_after_2nd_scoring_5_points_eq_l642_642738


namespace fractional_part_of_students_who_walk_home_l642_642115

def fraction_bus := 1 / 3
def fraction_automobile := 1 / 5
def fraction_bicycle := 1 / 8
def fraction_scooter := 1 / 10

theorem fractional_part_of_students_who_walk_home :
  (1 : ℚ) - (fraction_bus + fraction_automobile + fraction_bicycle + fraction_scooter) = 29 / 120 :=
by
  sorry

end fractional_part_of_students_who_walk_home_l642_642115


namespace extremum_at_one_eq_a_monotonic_intervals_minimum_value_range_l642_642413

noncomputable def f (a x : ℝ) : ℝ :=
  Real.log (a * x + 1) + (1 - x) / (1 + x)

noncomputable def f_prime (a x : ℝ) : ℝ :=
  (a * x ^ 2 + a - 2) / ((a * x + 1) * (1 + x) ^ 2)

-- (1)
theorem extremum_at_one_eq_a (a : ℝ) (h : f_prime a 1 = 0) : a = 1 := sorry

-- (2)
theorem monotonic_intervals (a : ℝ) (x : ℝ) (h : 0 < a) :
  (a ≥ 2 → (∀ x ≥ 0, f_prime a x > 0)) ∧
  ((0 < a ∧ a < 2) → (∀ x, 0 ≤ x ∧ x < Real.sqrt((2 - a) / a) → f_prime a x < 0)
                    ∧ (∀ x, x > Real.sqrt((2 - a) / a) → f_prime a x > 0)) := sorry

-- (3)
theorem minimum_value_range (a : ℝ) (h1 : f a 0 = 1) (h2 : ∀ x, x ≥ 0 → f a x ≥ 1) : 2 ≤ a := sorry

end extremum_at_one_eq_a_monotonic_intervals_minimum_value_range_l642_642413


namespace sum_of_inserted_numbers_in_geometric_and_arithmetic_progressions_l642_642599

theorem sum_of_inserted_numbers_in_geometric_and_arithmetic_progressions :
  ∃ (a b : ℕ), (4 < a ∧ a < b ∧ b < 16) ∧
  (∃ r : ℚ, a = 4 * r ∧ b = 4 * r * r) ∧
  (a + b = 2 * b - a + 16) ∧
  a + b = 24 :=
by
  sorry

end sum_of_inserted_numbers_in_geometric_and_arithmetic_progressions_l642_642599


namespace cones_and_ball_radius_l642_642538

noncomputable def find_r (h₁ h₂ : ℕ) (r1 r2 r3 : ℝ) (R : ℝ) : Prop :=
  ∃ r, r1 = 2 * r ∧ r2 = 3 * r ∧ r3 = 10 * r ∧ R = 2 ∧ ( ∀ (P : ℝ), P ≠ C → dist O P = r  -> r = 1)

theorem cones_and_ball_radius :
  find_r  h₁ h₂ (2 * r) (3 * r) (10 * r) 2 :=
sorry

end cones_and_ball_radius_l642_642538


namespace vectors_parallel_iff_abs_x_eq_2_l642_642816

noncomputable def a : ℝ × ℝ := (1, x)
noncomputable def b : ℝ × ℝ := (x^2, 4x)

theorem vectors_parallel_iff_abs_x_eq_2 (x : ℝ) (h₀ : a ≠ (0, 0)) (h₁ : b ≠ (0, 0)) : 
  (|x| = 2) ↔ (x^2 / 1 = 4x / x) := 
sorry

end vectors_parallel_iff_abs_x_eq_2_l642_642816


namespace abcd_sum_eq_nine_l642_642372

theorem abcd_sum_eq_nine
  (a b c d : ℤ)
  (h : (λ x : ℝ, (x^2 + a * x + b) * (x^2 + c * x + d)) = (λ x : ℝ, x^4 + x^3 - 2 * x^2 + 17 * x + 15)) :
  a + b + c + d = 9 :=
sorry

end abcd_sum_eq_nine_l642_642372


namespace donovan_lap_time_l642_642347

-- Definitions based on problem conditions
def lap_time_michael := 40  -- Michael's lap time in seconds
def laps_michael := 9       -- Laps completed by Michael to pass Donovan
def laps_donovan := 8       -- Laps completed by Donovan in the same time

-- Condition based on the solution
def race_duration := laps_michael * lap_time_michael

-- define the conjecture
theorem donovan_lap_time : 
  (race_duration = laps_donovan * 45) := 
sorry

end donovan_lap_time_l642_642347


namespace total_cost_of_apples_l642_642591

theorem total_cost_of_apples (cost_per_kg : ℝ) (packaging_fee : ℝ) (weight : ℝ) :
  cost_per_kg = 15.3 →
  packaging_fee = 0.25 →
  weight = 2.5 →
  (weight * (cost_per_kg + packaging_fee) = 38.875) :=
by
  intros h1 h2 h3
  sorry

end total_cost_of_apples_l642_642591


namespace estimate_log_l642_642740

theorem estimate_log : 
  ∀ (log₅ : ℝ → ℝ), 
  (log₅ 625 = 4) → 
  (log₅ 3125 = 5) →
  round (log₅ 1561) = 4 :=
begin
  intros log₅ h1 h2,
  -- insert actual proof here
  sorry
end

end estimate_log_l642_642740


namespace cos_150_eq_neg_sqrt3_div_2_l642_642287

theorem cos_150_eq_neg_sqrt3_div_2 :
  (cos (150 * real.pi / 180)) = - (sqrt 3 / 2) := sorry

end cos_150_eq_neg_sqrt3_div_2_l642_642287


namespace total_pay_per_week_l642_642013

variable (X Y : ℝ)
variable (hx : X = 1.2 * Y)
variable (hy : Y = 240)

theorem total_pay_per_week : X + Y = 528 := by
  sorry

end total_pay_per_week_l642_642013


namespace cos_150_eq_neg_half_l642_642276

theorem cos_150_eq_neg_half : Real.cos (150 * Real.pi / 180) = -1 / 2 := 
by
  sorry

end cos_150_eq_neg_half_l642_642276


namespace cos_150_eq_neg_sqrt3_div_2_l642_642155

theorem cos_150_eq_neg_sqrt3_div_2 : Real.cos (150 * Real.pi / 180) = - (Real.sqrt 3 / 2) :=
by
  sorry

end cos_150_eq_neg_sqrt3_div_2_l642_642155


namespace sum_series_eq_one_l642_642713

noncomputable def series : ℝ := ∑' n : ℕ, (4 * (n + 1) - 3) / 3^(n + 1)

theorem sum_series_eq_one : series = 1 := 
by sorry

end sum_series_eq_one_l642_642713


namespace distances_equal_l642_642775

open EuclideanGeometry

-- Defining the conditions from the problem
variables (A B C D K L X Y : Point)
variables (ω_A ω_B : Sphere)
variables [Midpoint M : Point]

-- Defining points and spheres and specific tangency properties
variable (h_tetrahedron : Tetrahedron A B C D)
variable (h_ωA_tangent_BCD : IsTangentToFace ω_A ⟨A, B, C⟩ ⟨B, C, D⟩)
variable (h_ωA_tangent_planes : IsTangentToPlanes ω_A ⟨A, B, D⟩ ⟨A, C, D⟩ 
  ∧ OutsidePlane ω_A ⟨A, B, D⟩ ⟨A, C, D⟩) 
variable (h_ωB_tangent_ACD : IsTangentToFace ω_B ⟨B, A, C⟩ ⟨A, C, D⟩)
variable (h_ωB_tangent_planes : IsTangentToPlanes ω_B ⟨A, B, D⟩ ⟨B, C, D⟩ 
  ∧ OutsidePlane ω_B ⟨A, B, D⟩ ⟨B, C, D⟩)
variable (h_k_tangency_AKD : PointOfTangency ω_A ⟨K, A, K, D⟩)
variable (h_l_tangency_BLD : PointOfTangency ω_B ⟨L, B, L, D⟩)

-- Defining angle conditions
variable (h_angle_CKD : Angle (C, K, D) = Angle (C, X, D) + Angle (C, B, D))
variable (h_angle_CLD : Angle (C, L, D) = Angle (C, Y, D) + Angle (C, A, D))

-- Statement to prove from given conditions
theorem distances_equal
  (h_tetrahedron : Tetrahedron A B C D)
  (h_ωA_tangent_BCD : IsTangentToFace ω_A ⟨A, B, C⟩ ⟨B, C, D⟩)
  (h_ωA_tangent_planes : IsTangentToPlanes ω_A ⟨A, B, D⟩ ⟨A, C, D⟩ 
    ∧ OutsidePlane ω_A ⟨A, B, D⟩ ⟨A, C, D⟩) 
  (h_ωB_tangent_ACD : IsTangentToFace ω_B ⟨B, A, C⟩ ⟨A, C, D⟩)
  (h_ωB_tangent_planes : IsTangentToPlanes ω_B ⟨A, B, D⟩ ⟨B, C, D⟩ 
    ∧ OutsidePlane ω_B ⟨A, B, D⟩ ⟨B, C, D⟩)
  (h_k_tangency_AKD : PointOfTangency ω_A ⟨K, A, K, D⟩)
  (h_l_tangency_BLD : PointOfTangency ω_B ⟨L, B, L, D⟩)
  (h_angle_CKD : Angle (C, K, D) = Angle (C, X, D) + Angle (C, B, D))
  (h_angle_CLD : Angle (C, L, D) = Angle (C, Y, D) + Angle (C, A, D))
  (M : Midpoint C D) :
  Distance X M = Distance Y M := 
sorry

end distances_equal_l642_642775


namespace coefficient_a_neg_one_is_28_l642_642471

noncomputable def coefficient_of_a_neg_one_in_expansion : ℤ :=
  ∑ k in Finset.range (8 + 1), if (8 - k - k / 2 = -1) then (Nat.choose 8 k) * (-1)^k else 0

theorem coefficient_a_neg_one_is_28 : coefficient_of_a_neg_one_in_expansion = 28 :=
by
  sorry

end coefficient_a_neg_one_is_28_l642_642471


namespace solve_problem_l642_642024

def problem_statement : Prop := (245245 % 35 = 0)

theorem solve_problem : problem_statement :=
by
  sorry

end solve_problem_l642_642024


namespace sin_cos_identity_count_l642_642371

theorem sin_cos_identity_count :
  { n : ℕ | 1 ≤ n ∧ n ≤ 500 ∧ ∀ t : ℝ, (sin t + complex.I * cos t)^(-n) = sin (-n * t) + complex.I * cos (-n * t) }.card = 125 :=
by
  sorry

end sin_cos_identity_count_l642_642371


namespace cos_150_eq_neg_sqrt3_div_2_l642_642253

theorem cos_150_eq_neg_sqrt3_div_2 :
  let theta := 30 * Real.pi / 180 in
  let Q := 150 * Real.pi / 180 in
  cos Q = - (Real.sqrt 3 / 2) := by
    sorry

end cos_150_eq_neg_sqrt3_div_2_l642_642253


namespace cos8_minus_sin8_l642_642034

theorem cos8_minus_sin8 (α : ℝ) : 
  cos(α) ^ 8 - sin(α) ^ 8 = cos(2 * α) * (3 + cos(4 * α)) / 4 :=
by sorry

end cos8_minus_sin8_l642_642034


namespace mike_went_to_last_year_l642_642912

def this_year_games : ℕ := 15
def games_missed_this_year : ℕ := 41
def total_games_attended : ℕ := 54
def last_year_games : ℕ := total_games_attended - this_year_games

theorem mike_went_to_last_year :
  last_year_games = 39 :=
  by sorry

end mike_went_to_last_year_l642_642912


namespace coefficient_x99_is_zero_l642_642970

open Polynomial

noncomputable def P (x : ℤ) : Polynomial ℤ := sorry
noncomputable def Q (x : ℤ) : Polynomial ℤ := sorry

theorem coefficient_x99_is_zero : 
    (P 0 = 1) → 
    ((P x)^2 = 1 + x + x^100 * Q x) → 
    (Polynomial.coeff ((P x + 1)^100) 99 = 0) :=
by
    -- Proof omitted
    sorry

end coefficient_x99_is_zero_l642_642970


namespace find_N_l642_642829

theorem find_N (N : ℕ) :
  ((5 + 6 + 7 + 8) / 4 = (2014 + 2015 + 2016 + 2017) / N) → N = 1240 :=
by
  sorry

end find_N_l642_642829


namespace arithmetic_sequence_term_13_l642_642856

variable {a : ℕ → ℝ}
variable {d : ℝ}

def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + d

theorem arithmetic_sequence_term_13 (h_arith : arithmetic_sequence a d)
  (h_a5 : a 5 = 3)
  (h_a9 : a 9 = 6) :
  a 13 = 9 := 
by 
  sorry

end arithmetic_sequence_term_13_l642_642856


namespace find_max_c_not_in_range_l642_642612

theorem find_max_c_not_in_range :
  ∃ (c : ℤ), 5 ∉ (λ x : ℝ, x^2 + (c : ℝ) * x + 20) '' set.univ ∧ ∀ (d : ℤ), 5 ∉ (λ x : ℝ, x^2 + (d : ℝ) * x + 20) '' set.univ → d ≤ 7 :=
begin
  sorry
end

end find_max_c_not_in_range_l642_642612


namespace required_fabric_l642_642059

noncomputable def total_fabric :=
  let a := 2011 in
  let r := (4 : ℚ) / 5 in
  a / (1 - r)

theorem required_fabric:
  total_fabric = 10055 := by
  have a : ℚ := 2011
  have r : ℚ := 4 / 5
  have h1 : a / (1 - r) = 10055
  calc
    a / (1 - r)
    = 2011 * 5 : by rw [div_eq_mul_inv, inv_of_15, mul_comm, mul_assoc, div_self, mul_one]
  exact h1

end required_fabric_l642_642059


namespace tony_comics_average_l642_642604

theorem tony_comics_average :
  let a1 := 10
  let d := 6
  let n := 8
  let a_n (n : ℕ) := a1 + (n - 1) * d
  let S_n (n : ℕ) := n / 2 * (a1 + a_n n)
  (S_n n) / n = 31 := by
  sorry

end tony_comics_average_l642_642604


namespace locus_M_l642_642776

theorem locus_M (A B C M : Point) (midpoint_bc : Point)
  (h_midpoint : midpoint_bc = midpoint B C)
  (h_perpendicular : LineThrough M midpoint_bc ⊥ LineThrough B C)
  (h_intersect : LineThrough M A ∩ LineThrough C A = {M})
  (h_AB : dist A B = c)
  (h_AC : dist A C = b)
  : 
  (c < b → ∀ C', C' ≠ A → (M = ellipse A B (dist A C'))) ∧
  (c > b → ∀ C', C' ≠ A → (M = hyperbola A B (dist A C'))) ∧
  (c = b → M = A) :=
by
  sorry

end locus_M_l642_642776


namespace number_of_students_l642_642941

theorem number_of_students 
  (number_of_teachers : ℕ)
  (student_ticket_cost : ℕ)
  (teacher_ticket_cost : ℕ)
  (total_ticket_cost : ℕ) 
  (teachers_eq : number_of_teachers = 4) 
  (student_cost_eq : student_ticket_cost = 1) 
  (teacher_cost_eq : teacher_ticket_cost = 3)
  (total_cost_eq : total_ticket_cost = 24) :
  ∃ (S : ℕ), S = 12 :=
by
  -- Definitions from the conditions
  have T : ℕ := number_of_teachers,
  have student_ticket_price : ℕ := student_ticket_cost,
  have teacher_ticket_price : ℕ := teacher_ticket_cost,
  have total_cost : ℕ := total_ticket_cost,
  rw [teachers_eq, student_cost_eq, teacher_cost_eq, total_cost_eq] at *,
  -- Define the cost equation
  have h : S * student_ticket_price + T * teacher_ticket_price = total_cost,
  sorry

end number_of_students_l642_642941


namespace amount_left_after_spending_l642_642494

-- Definitions based on conditions
def initial_amount : ℕ := 204
def amount_spent_on_toy (initial : ℕ) : ℕ := initial / 2
def remaining_after_toy (initial : ℕ) : ℕ := initial - amount_spent_on_toy initial
def amount_spent_on_book (remaining : ℕ) : ℕ := remaining / 2
def remaining_after_book (remaining : ℕ) : ℕ := remaining - amount_spent_on_book remaining

-- Proof statement
theorem amount_left_after_spending : 
  remaining_after_book (remaining_after_toy initial_amount) = 51 :=
sorry

end amount_left_after_spending_l642_642494


namespace career_preference_angles_l642_642584

theorem career_preference_angles (m f : ℕ) (total_degrees : ℕ) (one_fourth_males one_half_females : ℚ) (male_ratio female_ratio : ℚ) :
  total_degrees = 360 → male_ratio = 2/3 → female_ratio = 3/3 →
  m = 2 * f / 3 → one_fourth_males = 1/4 * m → one_half_females = 1/2 * f →
  (one_fourth_males + one_half_females) / (m + f) * total_degrees = 144 :=
by
  sorry

end career_preference_angles_l642_642584


namespace cos_150_deg_l642_642319

-- Define the conditions as variables
def angle1 := 150
def angle2 := 180
def angle3 := 30
def cos30 := Real.sqrt 3 / 2

-- The main statement to be proved
theorem cos_150_deg : Real.cos (150 * Real.pi / 180) = - (Real.sqrt 3 / 2) :=
by
  sorry

end cos_150_deg_l642_642319


namespace time_to_run_square_field_l642_642041

theorem time_to_run_square_field :
  ∀ (side_meters : ℕ) (speed_km_per_hr : ℕ),
    side_meters = 60 →
    speed_km_per_hr = 9 →
    let perimeter_meters := 4 * side_meters in
    let speed_mps := (speed_km_per_hr * 1000) / 3600 in
    let time_seconds := perimeter_meters / speed_mps in
    time_seconds = 96 :=
by
  intros side_meters speed_km_per_hr h_side h_speed
  rw [h_side, h_speed]
  let perimeter_meters := 4 * 60
  let speed_mps := (9 * 1000) / 3600
  let time_seconds := perimeter_meters / speed_mps
  have : time_seconds = 96 := by
    -- Proof steps are omitted, just using sorry to complete the statement
    sorry
  exact this

end time_to_run_square_field_l642_642041


namespace cos_150_eq_neg_half_l642_642198

theorem cos_150_eq_neg_half :
  real.cos (150 * real.pi / 180) = -1 / 2 :=
sorry

end cos_150_eq_neg_half_l642_642198


namespace festival_minimum_days_l642_642634

theorem festival_minimum_days
  (n : ℕ) (h_n : n = 11) 
  (P : ℕ → Prop) -- P(d) represents that it's possible to schedule the festival in d days
  (H0 : ∀ d < 6, ¬P(d)) -- Less than 6 days is not enough
  (H1 : P(6)) -- 6 days is adequate
  : 6 = d :=
by sorry

end festival_minimum_days_l642_642634


namespace sum_infinite_series_l642_642717

theorem sum_infinite_series : ∑' n : ℕ, (4 * (n + 1) - 3) / (3 ^ (n + 1)) = 3 / 2 := by
    sorry

end sum_infinite_series_l642_642717


namespace tetrahedron_circumsphere_radius_l642_642675

-- Conditions
def PA : ℝ := Real.sqrt 6
def BC : ℝ := Real.sqrt 6
def PB : ℝ := Real.sqrt 8
def AC : ℝ := Real.sqrt 8
def PC : ℝ := Real.sqrt 10
def AB : ℝ := Real.sqrt 10

-- Statement
theorem tetrahedron_circumsphere_radius : 
  PA = BC ∧ PB = AC ∧ PC = AB → 
  ∃ R : ℝ, R = Real.sqrt 3 :=
by
  sorry

end tetrahedron_circumsphere_radius_l642_642675


namespace cos_150_eq_neg_sqrt3_div_2_l642_642296

theorem cos_150_eq_neg_sqrt3_div_2 :
  (cos (150 * real.pi / 180)) = - (sqrt 3 / 2) := sorry

end cos_150_eq_neg_sqrt3_div_2_l642_642296


namespace orchard_sections_l642_642958

variable (total_sacks : ℕ)
variable (sacks_per_section : ℕ)
variable (num_sections : ℕ)

theorem orchard_sections (h1 : total_sacks = 360) (h2 : sacks_per_section = 45) :
  num_sections = total_sacks / sacks_per_section :=
by
  -- we'll need to conclude that the number of sections is 8
  sorry

example : num_sections = 8 := by
  apply orchard_sections
  assume h1 : 360 = 360
  assume h2 : 45 = 45
  have : 360 / 45 = 8 := by
    norm_num
  exact this

end orchard_sections_l642_642958


namespace contrapositive_inequality_l642_642567

theorem contrapositive_inequality (a b : ℝ) :
  (a > b → a - 5 > b - 5) ↔ (a - 5 ≤ b - 5 → a ≤ b) := by
sorry

end contrapositive_inequality_l642_642567


namespace cos_150_degree_l642_642185

theorem cos_150_degree : Real.cos (150 * Real.pi / 180) = -Real.sqrt 3 / 2 :=
by
  sorry

end cos_150_degree_l642_642185


namespace remainder_when_x_plus_4uy_div_y_l642_642561

theorem remainder_when_x_plus_4uy_div_y (x y u v : ℕ) (h₀: x = u * y + v) (h₁: 0 ≤ v) (h₂: v < y) : 
  ((x + 4 * u * y) % y) = v := 
by 
  sorry

end remainder_when_x_plus_4uy_div_y_l642_642561


namespace cos_150_eq_negative_cos_30_l642_642146

theorem cos_150_eq_negative_cos_30 :
  Real.cos (150 * Real.pi / 180) = - (Real.cos (30 * Real.pi / 180)) :=
by sorry

end cos_150_eq_negative_cos_30_l642_642146


namespace jill_commute_time_l642_642337

theorem jill_commute_time :
  let dave_steps_per_min := 80
  let dave_cm_per_step := 70
  let dave_time_min := 20
  let dave_speed :=
    dave_steps_per_min * dave_cm_per_step
  let dave_distance :=
    dave_speed * dave_time_min
  let jill_steps_per_min := 120
  let jill_cm_per_step := 50
  let jill_speed :=
    jill_steps_per_min * jill_cm_per_step
  let jill_time :=
    dave_distance / jill_speed
  jill_time = 18 + 2 / 3 := by
  sorry

end jill_commute_time_l642_642337


namespace total_birds_from_monday_to_wednesday_l642_642065

def birds_monday := 70
def birds_tuesday := birds_monday / 2
def birds_wednesday := birds_tuesday + 8
def total_birds := birds_monday + birds_tuesday + birds_wednesday

theorem total_birds_from_monday_to_wednesday : total_birds = 148 :=
by
  -- sorry is used here to skip the actual proof
  sorry

end total_birds_from_monday_to_wednesday_l642_642065


namespace sum_inf_series_l642_642707

theorem sum_inf_series :
  (\sum_{n=1}^{\infty} \frac{(4 * n) - 3}{3^n}) = 1 :=
by
  sorry

end sum_inf_series_l642_642707


namespace initial_music_files_eq_sixteen_l642_642610

theorem initial_music_files_eq_sixteen (M : ℕ) :
  (M + 48 - 30 = 34) → (M = 16) :=
by
  sorry

end initial_music_files_eq_sixteen_l642_642610


namespace complex_roots_sum_l642_642379

def quadratic_with_real_coefficients (α β : ℂ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ α * β = c + 0i ∧ α + β = -b + 0i

theorem complex_roots_sum
  (a b : ℝ)
  (h1 : quadratic_with_real_coefficients (3 + a * complex.I) (b - 2 * complex.I))
  : a + b = 1 :=
by
  sorry

end complex_roots_sum_l642_642379


namespace grid_impossible_one_black_square_l642_642084

theorem grid_impossible_one_black_square (grid : Fin 8 → Fin 8 → Bool) 
  (initial_black_squares : ∀ (r c: Fin 8), grid r c = true → 32 = (∑ r c, if grid r c then 1 else 0)) : 
  ¬(∃ (f : Fin 8 → Fin 8 → Bool) (op : Fin 8 ⊕ Fin 8 → Fin 8 → Fin 8 → Bool)
    (all_flips_valid : ∀ x, (∀ r c, grid r c ≠ op x r c) ↔ (∑ r c, if op x r c then 1 else 0) % 2 = 0)
    (final_state : Fin 8 → Fin 8 → Bool),
    ∀ r c, final_state r c = grid r c ∧ 
    1 = (∑ r c, if final_state r c then 1 else 0)) :=
by
  sorry

end grid_impossible_one_black_square_l642_642084


namespace find_x_in_triangle_l642_642865

noncomputable def triangle_x (y z : ℝ) (cos_YZ : ℝ) : ℝ := 
    let cosX: ℝ := 17/42 in
    let x_squared := 58 - 42 * cosX in
    real.sqrt x_squared

theorem find_x_in_triangle (y z : ℝ) (cos_YZ : ℝ) (h_y : y = 7) (h_z : z = 3) (h_cos_YZ : cos_YZ = 17 / 32) :
    triangle_x y z cos_YZ ^ 2 = 41 :=
by
    sorry

end find_x_in_triangle_l642_642865


namespace value_of_a_plus_b_l642_642446

noncomputable def x := 2 + Real.sqrt 21

theorem value_of_a_plus_b :
  (a b : ℕ) (h₀ : x = a + Real.sqrt b) (h₁ : x^2 + 5 * x + 5 / x + 1 / x^2 = 38)
  ⇒
  a + b = 23 :=
by
  sorry

end value_of_a_plus_b_l642_642446


namespace cos_150_degree_l642_642171

theorem cos_150_degree : Real.cos (150 * Real.pi / 180) = -Real.sqrt 3 / 2 :=
by
  sorry

end cos_150_degree_l642_642171


namespace vectors_parallel_iff_abs_x_eq_two_l642_642815

theorem vectors_parallel_iff_abs_x_eq_two (x: ℝ) : 
  ((1, x) : ℝ × ℝ) ∥ (x^2, 4 * x) ↔ |x| = 2 := sorry

end vectors_parallel_iff_abs_x_eq_two_l642_642815


namespace total_digits_in_numbering_pages_l642_642629

theorem total_digits_in_numbering_pages (n : ℕ) (h : n = 100000) : 
  let digits1 := 9 * 1
  let digits2 := (99 - 10 + 1) * 2
  let digits3 := (999 - 100 + 1) * 3
  let digits4 := (9999 - 1000 + 1) * 4
  let digits5 := (99999 - 10000 + 1) * 5
  let digits6 := 6
  (digits1 + digits2 + digits3 + digits4 + digits5 + digits6) = 488895 :=
by
  sorry

end total_digits_in_numbering_pages_l642_642629


namespace angle_APC_eq_120_l642_642852

theorem angle_APC_eq_120 (A B C D E P : Point) 
  (hABC_eq : equilateral_triangle A B C) 
  (hD_on_AB : D ∈ segment A B)
  (hE_on_BC : E ∈ segment B C)
  (hAD_eq_BE : segment_length A D = segment_length B E)
  (hP_on_AE : P ∈ line_through_points A E)
  (hP_on_CD : P ∈ line_through_points C D) : 
  angle A P C = 120 := 
sorry

end angle_APC_eq_120_l642_642852


namespace seating_arrangements_l642_642465

theorem seating_arrangements (men women : Fin 6 → Type) :
  (∀ i, men i ≠ women i) →
  6.factorial * 6.factorial = 518400 :=
by sorry

end seating_arrangements_l642_642465


namespace part1_part2_part3_l642_642370

def is_beautiful_point (x y : ℝ) (a b : ℝ) : Prop :=
  a = -x ∧ b = x - y

def beautiful_points (x y : ℝ) : (ℝ × ℝ) × (ℝ × ℝ) :=
  let a := -x
  let b := x - y
  ((a, b), (b, a))

theorem part1 (x y : ℝ) (h : (x, y) = (4, 1)) :
  beautiful_points x y = ((-4, 3), (3, -4)) := by
  sorry

theorem part2 (x y : ℝ) (h : x = 2) (h' : (-x = 2 - y)) :
  y = 4 := by
  sorry

theorem part3 (x y : ℝ) (h : ((-x, x-y) = (-2, 7)) ∨ ((x-y, -x) = (-2, 7))) :
  (x = 2 ∧ y = -5) ∨ (x = -7 ∧ y = -5) := by
  sorry

end part1_part2_part3_l642_642370


namespace volume_region_between_concentric_spheres_l642_642989

open Real

theorem volume_region_between_concentric_spheres (r1 r2 : ℝ) (h_r1 : r1 = 4) (h_r2 : r2 = 8) :
  (4 / 3 * π * r2^3 - 4 / 3 * π * r1^3) = 1792 / 3 * π :=
by
  sorry

end volume_region_between_concentric_spheres_l642_642989


namespace cosine_150_eq_neg_sqrt3_div_2_l642_642248

theorem cosine_150_eq_neg_sqrt3_div_2 :
  (∀ θ : ℝ, θ = real.pi / 6) →
  (∀ θ : ℝ, cos (real.pi - θ) = -cos θ) →
  cos (5 * real.pi / 6) = -real.sqrt 3 / 2 :=
by
  intros hθ hcos_identity
  sorry

end cosine_150_eq_neg_sqrt3_div_2_l642_642248


namespace prob_at_least_two_fever_reactions_is_correct_l642_642101

-- Define the probability of a fever reaction after vaccination
def prob_fever_reaction : ℝ := 0.80

-- Define the scenario where 3 people are vaccinated
def num_people : ℕ := 3

-- Define the desired probability calculation
def prob_at_least_two_fever_reactions := 
  let prob_two_fever := (3.choose 2) * (prob_fever_reaction ^ 2) * (1 - prob_fever_reaction)
  let prob_three_fever := prob_fever_reaction ^ 3
  prob_two_fever + prob_three_fever

-- Proving the desired probability
theorem prob_at_least_two_fever_reactions_is_correct : prob_at_least_two_fever_reactions = 0.896 := by
  sorry

end prob_at_least_two_fever_reactions_is_correct_l642_642101


namespace adam_candy_pieces_l642_642678

theorem adam_candy_pieces (initial_boxes given_away boxes_after giving_per_box total_pieces: ℕ) 
  (h1: initial_boxes = 13) 
  (h2: given_away = 7) 
  (h3: boxes_after = initial_boxes - given_away)
  (h4: giving_per_box = 6)
  (h5: total_pieces = boxes_after * giving_per_box) 
  : total_pieces = 36 := 
by
  rw [h1, h2, h4, h3, h5]
  sorry

end adam_candy_pieces_l642_642678


namespace calculate_ratio_l642_642440

variables (M Q P N R : ℝ)

-- Definitions of conditions
def M_def : M = 0.40 * Q := by sorry
def Q_def : Q = 0.30 * P := by sorry
def N_def : N = 0.60 * P := by sorry
def R_def : R = 0.20 * P := by sorry

-- Statement of the proof problem
theorem calculate_ratio (hM : M = 0.40 * Q) (hQ : Q = 0.30 * P)
  (hN : N = 0.60 * P) (hR : R = 0.20 * P) : 
  (M + R) / N = 8 / 15 := by
  sorry

end calculate_ratio_l642_642440


namespace integral_arctg_sqrt_l642_642049

noncomputable def indefinite_integral (f : ℝ → ℝ) : ℝ → ℝ := λ (F : ℝ → ℝ), (∀ x, deriv F x = f x)

theorem integral_arctg_sqrt :
  ∃ C : ℝ, indefinite_integral (λ x, arctan (sqrt (5 * x - 1))) (λ x, x * arctan (sqrt (5 * x - 1)) - (sqrt (5 * x - 1) / 5) + C) :=
sorry

end integral_arctg_sqrt_l642_642049


namespace cos_150_eq_neg_sqrt3_over_2_l642_642233

theorem cos_150_eq_neg_sqrt3_over_2 : Real.cos (150 * Real.pi / 180) = -Real.sqrt 3 / 2 := 
by 
  sorry

end cos_150_eq_neg_sqrt3_over_2_l642_642233


namespace count_ways_to_achieve_5_points_l642_642644

def points_from_results (wins draws losses : ℕ) : ℝ :=
  wins * 1 + draws * 0.5 + losses * 0

theorem count_ways_to_achieve_5_points :
  (∑ wins draws losses, 
    (wins + draws + losses = 7 ∧ points_from_results wins draws losses = 5) 
    → (nat.choose 7 wins) * (nat.choose (7 - wins) draws)) = 161 := sorry

end count_ways_to_achieve_5_points_l642_642644


namespace area_below_line_l642_642995

noncomputable def circle_eqn (x y : ℝ) := 
  x^2 + 2 * x + (y^2 - 6 * y) + 50 = 0

noncomputable def line_eqn (x y : ℝ) := 
  y = x + 1

theorem area_below_line : 
  (∃ (x y : ℝ), circle_eqn x y ∧ y < x + 1) →
  ∃ (a : ℝ), a = 20 * π :=
by
  sorry

end area_below_line_l642_642995


namespace arithmetic_sequence_general_term_and_max_tn_m_l642_642387

theorem arithmetic_sequence_general_term_and_max_tn_m (a_n : ℕ → ℝ) (b_n : ℕ → ℝ) (T_n : ℕ → ℝ):
  (∀ n, a_n = (1 / 3)^n) →
  (∀ n, b_n = log 3 (1 / (a_n n))) →
  (∀ n, T_n n = (∑ i in Finset.range n, 1 / (b_n i * b_n (i + 1)))) →
  ∃ m : ℕ, (∀ n ∈ ℕ, T_n n > m / 16) ∧ m = 7 :=
by
  sorry

end arithmetic_sequence_general_term_and_max_tn_m_l642_642387


namespace cosine_150_eq_neg_sqrt3_div_2_l642_642239

theorem cosine_150_eq_neg_sqrt3_div_2 :
  (∀ θ : ℝ, θ = real.pi / 6) →
  (∀ θ : ℝ, cos (real.pi - θ) = -cos θ) →
  cos (5 * real.pi / 6) = -real.sqrt 3 / 2 :=
by
  intros hθ hcos_identity
  sorry

end cosine_150_eq_neg_sqrt3_div_2_l642_642239


namespace sum_of_digits_of_M_l642_642345

/-- Define the sum of the digits function. -/
def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

noncomputable def M : ℕ :=
  let sixteen_power_81 := 16 ^ 81
  let eighty_one_power_16 := 81 ^ 16
  Int.toNat (Real.sqrt (sixteen_power_81 * eighty_one_power_16))

theorem sum_of_digits_of_M :
  sum_of_digits M = /* The correct answer to be computed */
  sorry

end sum_of_digits_of_M_l642_642345


namespace cos_150_eq_neg_sqrt3_over_2_l642_642232

theorem cos_150_eq_neg_sqrt3_over_2 : Real.cos (150 * Real.pi / 180) = -Real.sqrt 3 / 2 := 
by 
  sorry

end cos_150_eq_neg_sqrt3_over_2_l642_642232


namespace factorize_cube_diff_l642_642743

theorem factorize_cube_diff (a b : ℝ) : a^3 - 8b^3 = (a - 2b) * (a^2 + 2 * a * b + 4 * b^2) :=
by
  sorry

end factorize_cube_diff_l642_642743


namespace second_polygon_sides_l642_642016

/--
Given two regular polygons where:
- The first polygon has 42 sides.
- Each side of the first polygon is three times the length of each side of the second polygon.
- The perimeters of both polygons are equal.
Prove that the second polygon has 126 sides.
-/
theorem second_polygon_sides
  (s : ℝ) -- the side length of the second polygon
  (h1 : ∃ n : ℕ, n = 42) -- the first polygon has 42 sides
  (h2 : ∃ m : ℝ, m = 3 * s) -- the side length of the first polygon is three times the side length of the second polygon
  (h3 : ∃ k : ℕ, k * (3 * s) = n * s) -- the perimeters of both polygons are equal
  : ∃ n2 : ℕ, n2 = 126 := 
by
  sorry

end second_polygon_sides_l642_642016


namespace solve_equation_l642_642454

theorem solve_equation (a : ℝ) (x : ℝ) : (2 * a * x + 3) / (a - x) = 3 / 4 → x = 1 → a = -3 :=
by
  intros h h1
  rw [h1] at h
  sorry

end solve_equation_l642_642454


namespace cos_150_eq_neg_sqrt3_div_2_l642_642285

theorem cos_150_eq_neg_sqrt3_div_2 :
  (cos (150 * real.pi / 180)) = - (sqrt 3 / 2) := sorry

end cos_150_eq_neg_sqrt3_div_2_l642_642285


namespace nandan_earnings_l642_642882

theorem nandan_earnings (N T : ℝ) (h : 1.83 * N * T = 26000) : 0.15 * N * T = 2131.15 :=
by
  have hNT : N * T = 26000 / 1.83 := by sorry
  rw hNT
  field_simp
  norm_num
  sorry

end nandan_earnings_l642_642882


namespace ratio_of_areas_l642_642988

theorem ratio_of_areas (AB CD AC BD : Real)
  (h_square_side_len : AB = 1)
  (h_first_circle_tangent : ∃ r1 : Real, r1 = 1/2)
  (h_second_circle_tangent : ∃ r2 : Real, r2 = 1/2) :
  let A1 := π * (1/2)^2,
      A2 := π * (1/2)^2 in
  A2 / A1 = 1 :=
by
  sorry

end ratio_of_areas_l642_642988


namespace cos_150_eq_neg_sqrt3_div_2_l642_642128

theorem cos_150_eq_neg_sqrt3_div_2 :
  ∃ θ : ℝ, θ = 150 ∧
           0 < θ ∧ θ < 180 ∧
           ∃ φ : ℝ, φ = 180 - θ ∧
           cos φ = Real.sqrt 3 / 2 ∧
           cos θ = - cos φ :=
  sorry

end cos_150_eq_neg_sqrt3_div_2_l642_642128


namespace geometric_sequence_arithmetic_special_l642_642381

variable {a : ℕ → ℝ}

def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) :=
  ∀ n, a (n + 1) = a n * q

def is_arithmetic_sequence (b : ℕ → ℝ) :=
  ∀ n, b n - b (n + 1) = b (n + 1) - b (n + 2)

theorem geometric_sequence_arithmetic_special 
  (q : ℝ) (a : ℕ → ℝ)
  (h1 : a 1 = 4)
  (h2 : q ≠ 1)
  (h3 : is_geometric_sequence a q)
  (h4 : is_arithmetic_sequence (λ n, if n = 0 then 4 * a 1 else if n = 1 then a 5 else -2 * a 3)) :
  q = -1 ∧ (∀ n, (finset.range n).sum (λ i, a (2 * (i + 1))) = -4 * n) :=
by 
  sorry

end geometric_sequence_arithmetic_special_l642_642381


namespace cos_150_eq_negative_cos_30_l642_642151

theorem cos_150_eq_negative_cos_30 :
  Real.cos (150 * Real.pi / 180) = - (Real.cos (30 * Real.pi / 180)) :=
by sorry

end cos_150_eq_negative_cos_30_l642_642151


namespace sub_one_inequality_l642_642442

theorem sub_one_inequality (a b : ℝ) (h : a < b) : a - 1 < b - 1 :=
sorry

end sub_one_inequality_l642_642442


namespace event_after_adding_5_red_balls_l642_642840

-- Definition of the initial conditions
def initial_red_balls : ℕ := 10
def initial_black_balls : ℕ := 5

-- Definition of the drawing process
def add_red_ball(n: ℕ) : ℕ := n + 1

-- Definition of the event of interest
def event_adding_back_5_red_balls (ξ : ℕ) : Prop :=
  ξ = 6

-- Statement to prove
theorem event_after_adding_5_red_balls (ξ: ℕ) :
  (∀ b in finset.range(5), add_red_ball(b) = b + 1) →
  event_adding_back_5_red_balls ξ :=
sorry  -- Proof to be completed

end event_after_adding_5_red_balls_l642_642840


namespace sin_cos_15_is_1_over_4_l642_642975

-- Given constants: sin 30°, sin 15°, and cos 15°
constant sin : ℝ → ℝ
constant cos : ℝ → ℝ

axiom sin_30 : sin (30 * π / 180) = 1 / 2
axiom cos_15 : cos (15 * π / 180) = sqrt 3 / 2
axiom sin_15 : sin (15 * π / 180) = 1 / 2

-- The theorem we want to prove
theorem sin_cos_15_is_1_over_4 : sin (15 * π / 180) * cos (15 * π / 180) = 1 / 4 :=
by
  sorry

end sin_cos_15_is_1_over_4_l642_642975


namespace sum_series_eq_one_l642_642715

noncomputable def series : ℝ := ∑' n : ℕ, (4 * (n + 1) - 3) / 3^(n + 1)

theorem sum_series_eq_one : series = 1 := 
by sorry

end sum_series_eq_one_l642_642715


namespace slower_train_passes_faster_driver_in_12_seconds_l642_642990

noncomputable def time_to_pass_driver (l v_f v_s : ℕ) : ℕ :=
  let relative_speed := (v_f + v_s) * 5 / 18 -- Converting km/hr to m/s
  l / relative_speed

theorem slower_train_passes_faster_driver_in_12_seconds :
  ∀ (l v_f v_s : ℕ),
    l = 250 ∧ v_f = 45 ∧ v_s = 30 →
    time_to_pass_driver l v_f v_s = 12 :=
by
  intros
  cases h with hl hv
  cases hv with hvf hvs
  rw [hl, hvf, hvs]
  dsimp [time_to_pass_driver]
  -- the detailed proof steps calculating m/s, etc., are omitted.
  -- Assume correct calculation leads to the final answer 12 seconds.
  sorry

end slower_train_passes_faster_driver_in_12_seconds_l642_642990


namespace cos_150_eq_negative_cos_30_l642_642150

theorem cos_150_eq_negative_cos_30 :
  Real.cos (150 * Real.pi / 180) = - (Real.cos (30 * Real.pi / 180)) :=
by sorry

end cos_150_eq_negative_cos_30_l642_642150


namespace cos_150_eq_neg_sqrt3_div_2_l642_642136

theorem cos_150_eq_neg_sqrt3_div_2 :
  ∃ θ : ℝ, θ = 150 ∧
           0 < θ ∧ θ < 180 ∧
           ∃ φ : ℝ, φ = 180 - θ ∧
           cos φ = Real.sqrt 3 / 2 ∧
           cos θ = - cos φ :=
  sorry

end cos_150_eq_neg_sqrt3_div_2_l642_642136


namespace initial_pounds_of_coffee_l642_642651

variable (x : ℝ) (h1 : 0.25 * x = d₀) (h2 : 0.60 * 100 = d₁) 
          (h3 : (d₀ + d₁) / (x + 100) = 0.32)

theorem initial_pounds_of_coffee (d₀ d₁ : ℝ) : 
  x = 400 :=
by
  -- Given conditions
  have h1 : d₀ = 0.25 * x := sorry
  have h2 : d₁ = 0.60 * 100 := sorry
  have h3 : 0.32 = (d₀ + d₁) / (x + 100) := sorry
  
  -- Additional steps to solve for x
  sorry

end initial_pounds_of_coffee_l642_642651


namespace QO_perpendicular_BC_l642_642893

-- Define the points and conditions given in the problem
variables (A B C M N Q P O : Type)
-- Define the triangle ∆ABC
noncomputable def triangle_ABC := ∃ (A B C : Type), ∃ (M N : Type), ∃ (Q P O : Type),

-- Define the conditions of the proof
let median_AM : ∀ A B C M, ∃ M, true := sorry,
let angle_bisector_AN : ∀ A B C N, ∃ N, true := sorry,
let perpendicular_NA_MA : ∀ N A Q, ∃ Q, true := sorry,
let perpendicular_NA_BA : ∀ N A P, ∃ P, true := sorry,
let perpendicular_PA_AN : ∀ P A O, ∃ O, true := sorry,

-- Define the proof statement
theorem QO_perpendicular_BC : QO ⟂ BC :=
begin
  sorry,
end

end QO_perpendicular_BC_l642_642893


namespace max_a_value_l642_642521

theorem max_a_value (a b c : ℝ) (h_sum : a + b + c = 6) (h_product : ab + ac + bc = 11) :
  a ≤ 2 + 2 * real.sqrt 15 / 3 :=
sorry

end max_a_value_l642_642521


namespace paper_boat_travel_time_l642_642487

-- Given conditions
def embankment_length : ℝ := 50
def boat_length : ℝ := 10
def downstream_time : ℝ := 5
def upstream_time : ℝ := 4

-- Derived conditions from the given problem
def downstream_speed := embankment_length / downstream_time
def upstream_speed := embankment_length / upstream_time

-- Prove that the paper boat's travel time is 40 seconds
theorem paper_boat_travel_time :
  let v_boat := (downstream_speed + upstream_speed) / 2 in
  let v_current := (downstream_speed - upstream_speed) / 2 in
  let travel_time := embankment_length / v_current in
  travel_time = 40 := 
  sorry

end paper_boat_travel_time_l642_642487


namespace cos_150_eq_neg_sqrt3_over_2_l642_642224

theorem cos_150_eq_neg_sqrt3_over_2 : Real.cos (150 * Real.pi / 180) = -Real.sqrt 3 / 2 := 
by 
  sorry

end cos_150_eq_neg_sqrt3_over_2_l642_642224


namespace cos_150_eq_neg_sqrt3_div_2_l642_642138

theorem cos_150_eq_neg_sqrt3_div_2 :
  ∃ θ : ℝ, θ = 150 ∧
           0 < θ ∧ θ < 180 ∧
           ∃ φ : ℝ, φ = 180 - θ ∧
           cos φ = Real.sqrt 3 / 2 ∧
           cos θ = - cos φ :=
  sorry

end cos_150_eq_neg_sqrt3_div_2_l642_642138


namespace balls_boxes_distribution_l642_642436

theorem balls_boxes_distribution:
  (number_of_distinct_partitions 7 4 = 11) :=
sorry

-- Helper definition
def number_of_distinct_partitions (n k : ℕ) : ℕ :=
∀ (p : List ℕ), p.sum = n ∧ p.length ≤ k ∧ p.sorted = p

end balls_boxes_distribution_l642_642436


namespace paper_boat_travel_time_l642_642486

-- Given conditions
def embankment_length : ℝ := 50
def boat_length : ℝ := 10
def downstream_time : ℝ := 5
def upstream_time : ℝ := 4

-- Derived conditions from the given problem
def downstream_speed := embankment_length / downstream_time
def upstream_speed := embankment_length / upstream_time

-- Prove that the paper boat's travel time is 40 seconds
theorem paper_boat_travel_time :
  let v_boat := (downstream_speed + upstream_speed) / 2 in
  let v_current := (downstream_speed - upstream_speed) / 2 in
  let travel_time := embankment_length / v_current in
  travel_time = 40 := 
  sorry

end paper_boat_travel_time_l642_642486


namespace cos_150_eq_negative_cos_30_l642_642152

theorem cos_150_eq_negative_cos_30 :
  Real.cos (150 * Real.pi / 180) = - (Real.cos (30 * Real.pi / 180)) :=
by sorry

end cos_150_eq_negative_cos_30_l642_642152


namespace find_f_inv_sum_l642_642894

noncomputable def f (x : ℝ) : ℝ :=
if x >= 0 then x^3 else -x^3

def f_inv_9 : ℝ := real.cbrt 9
def f_inv_n27 : ℝ := -real.cbrt 27

theorem find_f_inv_sum : f_inv_9 + f_inv_n27 = -0.92 := sorry

end find_f_inv_sum_l642_642894


namespace unique_four_digit_square_l642_642091

theorem unique_four_digit_square (n : ℕ) : 
  1000 ≤ n ∧ n < 10000 ∧ 
  (n % 10 = (n / 10) % 10) ∧ 
  ((n / 100) % 10 = (n / 1000) % 10) ∧ 
  (∃ k : ℕ, n = k^2) ↔ n = 7744 := 
by 
  sorry

end unique_four_digit_square_l642_642091


namespace frustum_volume_l642_642722

-- Define the constants
def base_length : ℝ := 10
def slant_length : ℝ := 12
def cut_height_1 : ℝ := 4
def cut_height_2 : ℝ := 6

-- Calculate derived values
def height := real.sqrt (slant_length^2 - (base_length/2)^2)
def scale_factor_1 := (height - cut_height_1) / height
def scale_factor_2 := (height - cut_height_2) / height
def base_side_1 := base_length * scale_factor_1
def base_side_2 := base_length * scale_factor_2

-- Calculate areas
def area_1 := base_side_1^2
def area_2 := base_side_2^2

-- Calculate volume of the frustum
def volume_frustum := (2 / 3) * (area_1 + real.sqrt(area_1 * area_2) + area_2)

-- The main theorem statement
theorem frustum_volume : 
  volume_frustum = 
    (2 / 3) * (real.sqrt (slant_length^2 - (base_length / 2)^2 - cut_height_1)^2 +
               real.sqrt ((real.sqrt (slant_length^2 - (base_length / 2)^2 - cut_height_1)^2 * 
                           real.sqrt (slant_length^2 - (base_length / 2)^2 - cut_height_2)^2)) +
               real.sqrt (slant_length^2 - (base_length / 2)^2 - cut_height_2)^2) := sorry

end frustum_volume_l642_642722


namespace problem_proof_l642_642519

noncomputable def integer_part (a : ℝ) : ℤ :=
  if h : ∃ k : ℤ, (k : ℝ) ≤ a ∧ a < (k + 1 : ℝ) then Classical.choose h else 0

noncomputable def fractional_part (a : ℝ) : ℝ :=
  a - (integer_part a)

theorem problem_proof :
  let x := integer_part (8 - Real.sqrt 11)
  let y := fractional_part (8 - Real.sqrt 11)
  (3 : ℝ) < Real.sqrt 11 ∧ Real.sqrt 11 < (4 : ℝ) →
  2 * x * y - y^2 = 5 :=
by
  sorry

end problem_proof_l642_642519


namespace inequality_holds_for_a_range_l642_642364

theorem inequality_holds_for_a_range (a : ℝ) :
  (∀ (x : ℝ) (θ : ℝ), (0 ≤ θ ∧ θ ≤ π / 2) →
    (x + 3 + 2 * sin θ * cos θ)^2 + (x + a * sin θ + a * cos θ)^2 ≥ 1 / 8) ↔
  (a ∈ Iio (sqrt 6 / 2) ∪ Icc (7 / 2) ⊤) :=
sorry

end inequality_holds_for_a_range_l642_642364


namespace total_amount_paid_l642_642040

variables (kg_grapes kg_mangoes rate_grapes rate_mangoes : ℝ)

def cost_grapes := kg_grapes * rate_grapes
def cost_mangoes := kg_mangoes * rate_mangoes
def total_cost := cost_grapes + cost_mangoes

theorem total_amount_paid (h_grapes : kg_grapes = 3) (h_rate_grapes : rate_grapes = 70)
  (h_mangoes : kg_mangoes = 9) (h_rate_mangoes : rate_mangoes = 55) :
  total_cost kg_grapes kg_mangoes rate_grapes rate_mangoes = 705 :=
by
  sorry

end total_amount_paid_l642_642040


namespace total_fruits_l642_642032

theorem total_fruits (A O W : ℕ) (H1 : O * 0.5 = O / 2)
  (H2 : W * 4 = 4 * W) 
  (H_eq : A = O ∧ O = W ∧ W = A) 
  (H_cost : (A * 1) + (O * 0.5) + (W * 4) = 66) : 
  A + O + W = 36 :=
by
  sorry

end total_fruits_l642_642032


namespace induction_expression_addition_l642_642028

def algebraic_expression_addition (n : ℕ) : ℕ :=
  (n + 1) * (n + 2) * ... * (n + n) - 2^n * 1 * 2 * ... * (2 * n - 1)

theorem induction_expression_addition (k : ℕ) (hk : 0 < k) :
  algebraic_expression_addition (k + 1) = 2 * (2 * k + 1) :=
by
  sorry

end induction_expression_addition_l642_642028


namespace cubic_expression_identity_l642_642445

theorem cubic_expression_identity (x : ℝ) (hx : x + 1/x = 8) : 
  x^3 + 1/x^3 = 332 :=
sorry

end cubic_expression_identity_l642_642445


namespace prime_condition_l642_642553

theorem prime_condition (p : ℕ) (hp : Nat.Prime p) (h2p : Nat.Prime (p + 2)) : p = 3 ∨ 6 ∣ (p + 1) := 
sorry

end prime_condition_l642_642553


namespace range_of_a_l642_642406

noncomputable def proposition_p (a : ℝ) : Prop :=
  ∃ x : ℝ, x^2 - a*x + 1 = 0

noncomputable def proposition_q (a : ℝ) : Prop :=
  ∀ x : ℝ, x^2 - 2*x + a > 0

theorem range_of_a (a : ℝ) (hp : proposition_p a) (hq : proposition_q a) : a ≥ 2 :=
sorry

end range_of_a_l642_642406


namespace cos_150_eq_neg_sqrt3_over_2_l642_642208

theorem cos_150_eq_neg_sqrt3_over_2 :
  ∀ {deg cosQ1 cosQ2: ℝ},
    cosQ1 = 30 →
    cosQ2 = 180 - cosQ1 →
    cos 150 = - (cos cosQ1) →
    cos cosQ1 = sqrt 3 / 2 →
    cos 150 = -sqrt 3 / 2 :=
by
  intros deg cosQ1 cosQ2 h1 h2 h3 h4
  sorry

end cos_150_eq_neg_sqrt3_over_2_l642_642208


namespace range_of_x_l642_642594

theorem range_of_x (total_students math_club chemistry_club : ℕ) (h_total : total_students = 45) 
(h_math : math_club = 28) (h_chemistry : chemistry_club = 21) (x : ℕ) :
  4 ≤ x ∧ x ≤ 21 ↔ (28 + 21 - x ≤ 45) :=
by sorry

end range_of_x_l642_642594


namespace cos_150_deg_l642_642323

-- Define the conditions as variables
def angle1 := 150
def angle2 := 180
def angle3 := 30
def cos30 := Real.sqrt 3 / 2

-- The main statement to be proved
theorem cos_150_deg : Real.cos (150 * Real.pi / 180) = - (Real.sqrt 3 / 2) :=
by
  sorry

end cos_150_deg_l642_642323


namespace distance_from_Q_equal_l642_642632

theorem distance_from_Q_equal (O₁ O₂ P Q A B : Type) 
  (h₁ : intersecting_circles O₁ O₂ P Q)
  (h₂ : circle_through_three_points P O₁ O₂)
  (h₃ : intersection_on_circle A O₁ B O₂ P O₁ O₂):
  distances_equal_from_point_to_lines Q P A B :=
begin
  sorry
end

end distance_from_Q_equal_l642_642632


namespace probability_divisible_by_8_l642_642609

-- Define the problem conditions
def is_8_sided_die (n : ℕ) : Prop := n = 6
def roll_dice (m : ℕ) : Prop := m = 8

-- Define the main proof statement
theorem probability_divisible_by_8 (n m : ℕ) (hn : is_8_sided_die n) (hm : roll_dice m) :  
  (35 : ℚ) / 36 = 
  (1 - ((1/2) ^ m + 28 * ((1/n) ^ 2 * ((1/2) ^ 6))) : ℚ) :=
by
  sorry

end probability_divisible_by_8_l642_642609


namespace find_m_n_sum_l642_642703

theorem find_m_n_sum :
  ∃ (m n : ℕ), (∃ x : ℝ, let rD := 3*x in let rE := x in
    (rD = real.sqrt m - n) ∧
    (2 - 3*x)^2 = 2 * (( -1/2)^2 + ((1/(3*x))^2 + 2*(1/x)^2)) ∧
    ( - 1/2 + 1/(3*x) + 2*(1/x))^2 = 2 * (( - 1/2)^2 + (1/(3*x))^2 + 2*(1/3*x)^2)) ∧
    m + n = 254 :=
sorry

end find_m_n_sum_l642_642703


namespace cos_150_eq_neg_sqrt3_div_2_l642_642263

theorem cos_150_eq_neg_sqrt3_div_2 :
  let theta := 30 * Real.pi / 180 in
  let Q := 150 * Real.pi / 180 in
  cos Q = - (Real.sqrt 3 / 2) := by
    sorry

end cos_150_eq_neg_sqrt3_div_2_l642_642263


namespace cos_150_deg_l642_642324

-- Define the conditions as variables
def angle1 := 150
def angle2 := 180
def angle3 := 30
def cos30 := Real.sqrt 3 / 2

-- The main statement to be proved
theorem cos_150_deg : Real.cos (150 * Real.pi / 180) = - (Real.sqrt 3 / 2) :=
by
  sorry

end cos_150_deg_l642_642324


namespace trajectory_eq_chord_length_MN_l642_642474

-- Define the conditions and the required theorem statements
def point (x : ℝ) (y : ℝ) := (x, y)

def A := point 3 0
def B := point (-1) 0
def line_l (x y : ℝ) := x - y + 3 = 0

-- Define vector operations involving points
def vector (p1 p2 : (ℝ × ℝ)) := ((p2.1 - p1.1), (p2.2 - p1.2))

def dot_product (v1 v2 : (ℝ × ℝ)) := v1.1 * v2.1 + v1.2 * v2.2

-- Prove that the trajectory of point C (question 1)
theorem trajectory_eq :
  ∀ (C : ℝ × ℝ), dot_product (vector A C) (vector B C) = 5 ↔ (C.1 - 1) ^ 2 + C.2 ^ 2 = 9 := by
  sorry

-- Prove the length of |MN| (question 2)
theorem chord_length_MN :
  ∀ (M N : ℝ × ℝ), line_l M.1 M.2 ∧ line_l N.1 N.2 ∧ ((M.1 - 1)^2 + M.2^2 = 9) ∧ ((N.1 - 1)^2 + N.2^2 = 9) →
  (dist M N = 2) := by
  sorry

end trajectory_eq_chord_length_MN_l642_642474


namespace limit_of_function_l642_642698

theorem limit_of_function (a x : ℝ) (h : ℝ) (ha : 0 < a) :
  ( ∀ ε > 0, ∃ δ > 0, ∀ h, abs h < δ → abs ( ( a ^ ( x + h ) + a ^ ( x - h ) - 2 * a ^ x ) / h ) < ε ) → 
  ( ∀ ε > 0, ∃ δ > 0, ∀ h, abs h < δ → abs ( a ^ x * log a ) < ε ) := 
sorry

end limit_of_function_l642_642698


namespace simplify_fraction_l642_642555

theorem simplify_fraction :
  (1 / (1 / (1 / 2) ^ 1 + 1 / (1 / 2) ^ 2 + 1 / (1 / 2) ^ 3)) = (1 / 14) :=
by 
  sorry

end simplify_fraction_l642_642555


namespace min_value_fraction_l642_642763

theorem min_value_fraction (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : (3:ℝ) ^ (1/2) = real.sqrt ((3:ℝ) ^ a * (3:ℝ) ^ (2 * b))) : 
  (2 / a + 1 / b) = 8 :=
sorry

end min_value_fraction_l642_642763


namespace max_y_coordinate_on_curve_l642_642023

theorem max_y_coordinate_on_curve :
  (∀ (x y : ℝ), ((x - 3)^2 / 25 + (y - 2)^2 / 9 = 0) → y ≤ 2) ∧
  (∃ (x : ℝ), (x - 3)^2 / 25 + (2 - 2)^2 / 9 = 0) :=
begin
  sorry
end

end max_y_coordinate_on_curve_l642_642023


namespace all_have_perp_property_l642_642420

def M₁ : Set (ℝ × ℝ) := {p | ∃ x, p = (x, x^3 - 2 * x^2 + 3)}
def M₂ : Set (ℝ × ℝ) := {p | ∃ x, p = (x, Real.log (2 - x) / Real.log 2)}
def M₃ : Set (ℝ × ℝ) := {p | ∃ x, p = (x, 2 - 2^x)}
def M₄ : Set (ℝ × ℝ) := {p | ∃ x, p = (x, 1 - Real.sin x)}

def perp_property (M : Set (ℝ × ℝ)) : Prop :=
∀ p ∈ M, ∃ q ∈ M, p.1 * q.1 + p.2 * q.2 = 0

-- Theorem statement
theorem all_have_perp_property :
  perp_property M₁ ∧ perp_property M₂ ∧ perp_property M₃ ∧ perp_property M₄ :=
sorry

end all_have_perp_property_l642_642420


namespace factorial_division_l642_642636

theorem factorial_division :
  (Nat.factorial 4) / (Nat.factorial (4 - 3)) = 24 :=
by
  sorry

end factorial_division_l642_642636


namespace joeys_age_digit_sum_l642_642500

noncomputable def joeys_age_multiple_of_zoes_age (C : ℕ) (J : ℕ) (Z : ℕ) (k : ℕ) : ℕ :=
  let m := if (k > 1) then (k * (Z + 40 - C) - (J + 40 - C)) else 0 in
  J + m

noncomputable def sum_of_digits (n : ℕ) : ℕ :=
  (n / 10) + (n % 10)

theorem joeys_age_digit_sum : 
  ∀ C : ℕ, ∀ J : ℕ, ∀ Z : ℕ, 
  C + 1 = J ∧ Z = 2 ∧ 40 = (40 / Z) * Z →
  sum_of_digits (joeys_age_multiple_of_zoes_age C J Z 13) = 13 :=
by
  intros C J Z h
  sorry

end joeys_age_digit_sum_l642_642500


namespace angle_between_line_and_plane_is_30_degrees_l642_642820

variables {V : Type*} [inner_product_space ℝ V]

/-- Variables m and n are vectors such that m is the direction vector of line l and n is
the normal vector of plane α, while cos <m, n> = -1/2. -/
def angle_between_line_and_plane (l : set V) (α : set V) (m n : V)
  (h1 : is_direction_vector_of_line m l)
  (h2 : is_normal_vector_of_plane n α)
  (h_cos_m_n : ⟪m, n⟫ = -1/2) : real.angle := 30 -- degrees

/-- The theorem states that the angle formed by the line l and the plane α is 30 degrees
given the above conditions. -/
theorem angle_between_line_and_plane_is_30_degrees
  (l : set V) (α : set V) (m n : V)
  (h1 : is_direction_vector_of_line m l)
  (h2 : is_normal_vector_of_plane n α)
  (h_cos_m_n : ⟪m, n⟫ = -1/2) : angle_between_line_and_plane l α m n h1 h2 h_cos_m_n = 30 :=
by
  sorry

end angle_between_line_and_plane_is_30_degrees_l642_642820


namespace direct_below_inverse_first_quadrant_l642_642650

variable (b x : ℝ)
variable (hb : b > 0)
def f : ℝ → ℝ := λ x, b * x / 2
def g : ℝ → ℝ := λ x, 2 * b / x

theorem direct_below_inverse_first_quadrant (x : ℝ) (hx : 0 < x) : f b x < g b x ↔ 0 < x ∧ x < 2 := by
  sorry

end direct_below_inverse_first_quadrant_l642_642650


namespace collinear_of_intersections_l642_642470

open Geometry Complex.angle

theorem collinear_of_intersections (A B C A1 B1 C1 P D E F : Point) (O : Circumcircle)
  (hA1 : A1 ∈ O) (hB1 : B1 ∈ O) (hC1 : C1 ∈ O)
  (h_parallel : parallel (line_through A A1) (line_through B B1) ∧ 
                parallel (line_through B B1) (line_through C C1) ∧
                parallel (line_through A A1) (line_through C C1))
  (hP : P ∈ O)
  (hD : D = line_through P A1 ∩ line_through B C)
  (hE : E = line_through P B1 ∩ line_through A C)
  (hF : F = line_through P C1 ∩ line_through A B) :
  collinear {D, E, F} :=
by
  sorry

end collinear_of_intersections_l642_642470


namespace cos_150_eq_neg_sqrt3_div_2_l642_642304

open Real

theorem cos_150_eq_neg_sqrt3_div_2 :
  cos (150 * pi / 180) = - (sqrt 3 / 2) := 
sorry

end cos_150_eq_neg_sqrt3_div_2_l642_642304


namespace expression_for_x_5_expression_for_x_minus1_expressions_for_x_l642_642838

-- Define the condition
def quadratic_eq (x : ℝ) : Prop := x^2 - 4 * x = 5

-- Two expressions involving x that equal 3 given the equation 
theorem expression_for_x_5 :
  quadratic_eq 5 → 5 - 2 = 3 :=
by
  intros h
  have h1 : 5^2 - 4 * 5 = 25 - 20 := by norm_num
  rw [h1] at h
  norm_num at h
  exact h

theorem expression_for_x_minus1 :
  quadratic_eq (-1) → -1 + 4 = 3 :=
by
  intros h
  have h1 : (-1)^2 - 4 * (-1) = 1 + 4 := by norm_num
  rw [h1] at h
  norm_num at h
  exact h

-- Alternative formulation for a more general proof
theorem expressions_for_x (x : ℝ) :
  quadratic_eq x → (x = 5 → x - 2 = 3) ∧ (x = -1 → x + 4 = 3) :=
by
  intros h
  split
  {
    intros hx
    rw [hx] at h
    have h1 : 5^2 - 4 * 5 = 25 - 20 := by norm_num
    rw [h1] at h
    norm_num at h
    exact h
  }
  {
    intros hx
    rw [hx] at h
    have h1 : (-1)^2 - 4 * (-1) = 1 + 4 := by norm_num
    rw [h1] at h
    norm_num at h
    exact h
  }

end expression_for_x_5_expression_for_x_minus1_expressions_for_x_l642_642838


namespace mean_variance_transformation_l642_642964

variable (n : ℕ)
variable (x : Fin n → ℝ)
variable (mean_original variance_original : ℝ)
variable (meam_new variance_new : ℝ)
variable (offset : ℝ)

theorem mean_variance_transformation (hmean : mean_original = 2.8) (hvariance : variance_original = 3.6) 
  (hoffset : offset = 60) : 
  (mean_new = mean_original + offset) ∧ (variance_new = variance_original) :=
  sorry

end mean_variance_transformation_l642_642964


namespace amount_per_can_l642_642499

theorem amount_per_can:
    ∀ (x : ℝ), (80 * 0.10 + 140 * x = 15) → x = 0.05 :=
by
  assume x,
  sorry

end amount_per_can_l642_642499


namespace cone_lateral_area_l642_642402

theorem cone_lateral_area (r l : ℝ) (h_r : r = 3) (h_l : l = 5) : 
  let c := 2 * Real.pi * r in
  let A := 1 / 2 * c * l in
  A = 15 * Real.pi :=
by
  let c := 2 * Real.pi * r
  have : c = 6 * Real.pi := by rw [h_r, ←mul_assoc, mul_comm 3, mul_assoc]
  let A := 1 / 2 * c * l
  have : A = 15 * Real.pi := by rw [this, h_l, div_mul_cancel (6 * Real.pi) (2), mul_assoc]; ring
  exact this

end cone_lateral_area_l642_642402


namespace sum_series_eq_l642_642708

theorem sum_series_eq :
  ∑' n : ℕ, (4 * (n + 1) - 3) / 3^(n + 1) = 3 / 2 :=
sorry

end sum_series_eq_l642_642708


namespace cos_150_eq_neg_half_l642_642277

theorem cos_150_eq_neg_half : Real.cos (150 * Real.pi / 180) = -1 / 2 := 
by
  sorry

end cos_150_eq_neg_half_l642_642277


namespace cos_150_eq_neg_sqrt3_over_2_l642_642214

theorem cos_150_eq_neg_sqrt3_over_2 :
  ∀ {deg cosQ1 cosQ2: ℝ},
    cosQ1 = 30 →
    cosQ2 = 180 - cosQ1 →
    cos 150 = - (cos cosQ1) →
    cos cosQ1 = sqrt 3 / 2 →
    cos 150 = -sqrt 3 / 2 :=
by
  intros deg cosQ1 cosQ2 h1 h2 h3 h4
  sorry

end cos_150_eq_neg_sqrt3_over_2_l642_642214


namespace cosine_150_eq_neg_sqrt3_div_2_l642_642236

theorem cosine_150_eq_neg_sqrt3_div_2 :
  (∀ θ : ℝ, θ = real.pi / 6) →
  (∀ θ : ℝ, cos (real.pi - θ) = -cos θ) →
  cos (5 * real.pi / 6) = -real.sqrt 3 / 2 :=
by
  intros hθ hcos_identity
  sorry

end cosine_150_eq_neg_sqrt3_div_2_l642_642236


namespace molecular_weight_of_compound_l642_642613

theorem molecular_weight_of_compound : 
  let atomic_weight_H := 1.008
  let atomic_weight_Cr := 51.996
  let atomic_weight_O := 15.999
  let atomic_weight_N := 14.007
  let atomic_weight_C := 12.011
  let num_H := 4
  let num_Cr := 2
  let num_O := 4
  let num_N := 3
  let num_C := 5
  let total_weight := 
    num_H * atomic_weight_H +
    num_Cr * atomic_weight_Cr +
    num_O * atomic_weight_O +
    num_N * atomic_weight_N +
    num_C * atomic_weight_C
  in total_weight = 274.096 :=
sorry

end molecular_weight_of_compound_l642_642613


namespace cos_150_eq_neg_sqrt3_div_2_l642_642312

open Real

theorem cos_150_eq_neg_sqrt3_div_2 :
  cos (150 * pi / 180) = - (sqrt 3 / 2) := 
sorry

end cos_150_eq_neg_sqrt3_div_2_l642_642312


namespace length_of_BQ_l642_642863

-- Define the triangle and related segments
variables {A B C E Q : Type}
variables {AB BC CA : ℝ}
variables {BE EC BQ : ℝ}

-- Assume the lengths of the sides of triangle ABC
def triangle_ABC :=
  AB = 13 ∧ BC = 15 ∧ CA = 14

-- Assume E lies on BC such that BE = EC
def point_E :=
  E ∈ segment BC ∧ BE = EC

-- The question to prove with given conditions: length of BQ
theorem length_of_BQ
  (h1 : triangle_ABC)
  (h2 : point_E)
  (hBQ : ∃ Q, Q ∈ circumcircle B I_B E ∧ Q ≠ E) :
  BQ = 13 * (real.sqrt 2) / 2 :=
by sorry

end length_of_BQ_l642_642863


namespace probability_avg_two_l642_642810

theorem probability_avg_two (s : set ℕ) (h : s = {1, 2, 3, 6}) : 
  (∃ (l : list ℕ), l.nodup ∧ ↑l ⊆ s ∧ (l.length = 3) ∧ ((l.sum : ℚ) / 3 = 2) )  = (ℚ.of_int 1 / 4) :=
by 
  sorry

end probability_avg_two_l642_642810


namespace solution_complex_b_l642_642452

theorem solution_complex_b (b : ℝ) :
  let z := (2 - b*complex.i) / (1 + 2*complex.i) in
  real := z.re,
  imag := z.im
  (real = -imag) → b = -2 / 3 :=
begin
  sorry
end

end solution_complex_b_l642_642452


namespace negation_of_p_l642_642949

def p := ∃ n : ℕ, n^2 > 2 * n - 1

theorem negation_of_p : ¬ p ↔ ∀ n : ℕ, n^2 ≤ 2 * n - 1 :=
by sorry

end negation_of_p_l642_642949


namespace solutions_diff_squared_l642_642518

theorem solutions_diff_squared (a b : ℝ) (h : 5 * a^2 - 6 * a - 55 = 0 ∧ 5 * b^2 - 6 * b - 55 = 0) :
  (a - b)^2 = 1296 / 25 := by
  sorry

end solutions_diff_squared_l642_642518


namespace statement_1_incorrect_statement_3_incorrect_statement_4_incorrect_l642_642682

-- Define the notion of line and plane
def Line := Type
def Plane := Type

-- Define the relations: parallel, contained-in, and intersection
def parallel (a b : Line) : Prop := sorry
def contained_in (a : Line) (α : Plane) : Prop := sorry
def intersects_at (a : Line) (α : Plane) (P : Type) : Prop := sorry

-- Conditions translated into Lean
def cond1 (a : Line) (α : Plane) (b : Line) : Prop := parallel a α ∧ contained_in b α → parallel a b
def cond2 (a : Line) (α : Plane) (b : Line) {P : Type} : Prop := intersects_at a α P ∧ contained_in b α → ¬ parallel a b
def cond3 (a : Line) (α : Plane) : Prop := ¬ contained_in a α → parallel a α
def cond4 (a : Line) (α : Plane) (b : Line) : Prop := parallel a α ∧ parallel b α → parallel a b

-- The statements that need to be proved incorrect
theorem statement_1_incorrect (a : Line) (α : Plane) (b : Line) : ¬ (cond1 a α b) := sorry
theorem statement_3_incorrect (a : Line) (α : Plane) : ¬ (cond3 a α) := sorry
theorem statement_4_incorrect (a : Line) (α : Plane) (b : Line) : ¬ (cond4 a α b) := sorry

end statement_1_incorrect_statement_3_incorrect_statement_4_incorrect_l642_642682


namespace monotonic_decreasing_range_l642_642834

def is_decreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x ≥ f y

noncomputable def f (a : ℝ) : ℝ → ℝ :=
λ x, if x < 1 then (5 * a - 4) * x + 7 * a - 3 else (2 * a - 1) ^ x

theorem monotonic_decreasing_range {a : ℝ} :
  (is_decreasing (f a) → (3 / 5) ≤ a ∧ a < (4 / 5)) :=
sorry

end monotonic_decreasing_range_l642_642834


namespace tan_squared_sum_geq_three_eighths_l642_642884

theorem tan_squared_sum_geq_three_eighths 
  (α β γ : ℝ) 
  (hα : 0 < α ∧ α < π / 2) 
  (hβ : 0 < β ∧ β < π / 2) 
  (hγ : 0 < γ ∧ γ < π / 2) 
  (h_sum_sin : sin α + sin β + sin γ = 1) :
  tan α ^ 2 + tan β ^ 2 + tan γ ^ 2 ≥ 3 / 8 := 
by 
  sorry

end tan_squared_sum_geq_three_eighths_l642_642884


namespace max_pieces_from_cake_l642_642996

theorem max_pieces_from_cake (large_cake_area small_piece_area : ℕ) 
  (h_large_cake : large_cake_area = 15 * 15) 
  (h_small_piece : small_piece_area = 5 * 5) :
  large_cake_area / small_piece_area = 9 := 
by
  sorry

end max_pieces_from_cake_l642_642996


namespace inequality_proof_l642_642000

variable (a b c l1 l2 l3 p : ℝ)

-- Conditions
def triangle_abc (a b c : ℝ) := a > 0 ∧ b > 0 ∧ c > 0
def semiperimeter (a b c : ℝ) := (a + b + c) / 2
def angle_bisectors (l1 l2 l3 : ℝ) := l1 > 0 ∧ l2 > 0 ∧ l3 > 0

-- The statement we want to prove
theorem inequality_proof (h_triangle : triangle_abc a b c) 
    (h_semiperimeter : p = semiperimeter a b c) 
    (h_angle_bisectors : angle_bisectors l1 l2 l3) :
    1 / l1^2 + 1 / l2^2 + 1 / l3^2 ≥ 81 / p^2 :=
by sorry

end inequality_proof_l642_642000


namespace probability_sum_of_square_divisors_of_15_factorial_l642_642093

theorem probability_sum_of_square_divisors_of_15_factorial :
  let m := 1
  let n := 84
  m + n = 85 :=
by
  -- Establish the necessary factorization for the proof context.
  let factorial_15_prime_factors := 2^11 * 3^6 * 5^3 * 7^2 * 11^1 * 13^1

  -- Total number of divisors computation based on exponents of prime factors.
  have total_divisors : factorial_15_prime_factors.factors.prod
    ((11 + 1) * (6 + 1) * (3 + 1) * (2 + 1) * (1 + 1) * (1 + 1)) = 4032 := sorry

  -- Count the perfect square divisors using permissible exponent combinations.
  have perf_sq_divisors : (6 choices of 2) * (4 choices of 3) * (2 choices of 5) * (1 choice of 7, 11, 13) = 48 := sorry

  -- Probability as simplified fraction.
  have prob_simplified : 48 / 4032 = 1 / 84
    let m := 1
    let n := 84 := sorry

  -- Verifying query requirement.
  have answer : m + n = 85 := sorry
  exact answer

end probability_sum_of_square_divisors_of_15_factorial_l642_642093


namespace replace_digits_correct_l642_642548

def digits_eq (a b c d e : ℕ) : Prop :=
  5 * 10 + a + (b * 100) + (c * 10) + 3 = (d * 1000) + (e * 100) + 1

theorem replace_digits_correct :
  ∃ (a b c d e : ℕ), 0 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 ∧ 0 ≤ d ∧ d ≤ 9 ∧ 0 ≤ e ∧ e ≤ 9 ∧
    digits_eq a b c d e ∧ a = 1 ∧ b = 1 ∧ c = 4 ∧ d = 1 ∧ e = 4 :=
by
  sorry

end replace_digits_correct_l642_642548


namespace sum_infinite_series_l642_642719

theorem sum_infinite_series : ∑' n : ℕ, (4 * (n + 1) - 3) / (3 ^ (n + 1)) = 3 / 2 := by
    sorry

end sum_infinite_series_l642_642719


namespace min_value_expression_l642_642525

theorem min_value_expression {x y : ℝ} (h1 : x^2 + y^2 = 2) (h2 : |x| ≠ |y|) : 
  min (frac 1 (x + y)^2 + frac 1 (x - y)^2) 1 :=
sorry

end min_value_expression_l642_642525


namespace cos_150_eq_neg_half_l642_642188

theorem cos_150_eq_neg_half :
  real.cos (150 * real.pi / 180) = -1 / 2 :=
sorry

end cos_150_eq_neg_half_l642_642188


namespace charlotte_can_walk_poodles_on_tuesday_l642_642122

def hours_per_day (day: String) : Nat :=
  if day ∈ ["Saturday", "Sunday"] then 4 else 8

def hours_per_week : Nat :=
  5 * hours_per_day "Monday" + 2 * hours_per_day "Saturday"

def hours_monday : Nat :=
  4 * 2 + 2 * 1 + 1 * 4

def hours_wednesday : Nat :=
  4 * 3

def hours_walking_dog (dog: String) : Nat :=
  match dog with
  | "Poodle" => 2
  | "Chihuahua" => 1
  | "Labrador" => 3
  | "GoldenRetriever" => 4
  | _ => 0

def remaining_hours (total: Nat) (monday: Nat) (wednesday: Nat): Nat :=
  total - (monday + wednesday)

theorem charlotte_can_walk_poodles_on_tuesday :
  let available_hours := hours_per_week
  let monday_hours := hours_monday
  let wednesday_hours := hours_wednesday
  let tuesday_hours := hours_per_day "Tuesday"
  let used_tuesday := 2 * hours_walking_dog "Chihuahua" + hours_walking_dog "GoldenRetriever"
  let remaining := tuesday_hours - used_tuesday
  remaining / hours_walking_dog "Poodle" = 1 :=
by
  have available_hours := hours_per_week
  have monday_hours := hours_monday
  have wednesday_hours := hours_wednesday
  have tuesday_hours := remaining_hours available_hours monday_hours wednesday_hours
  have used_tuesday := 2 * hours_walking_dog "Chihuahua" + hours_walking_dog "GoldenRetriever"
  have remaining := 8 - used_tuesday
  have result := remaining / hours_walking_dog "Poodle"
  show result = 1, from sorry

end charlotte_can_walk_poodles_on_tuesday_l642_642122


namespace number_of_valid_points_l642_642643

-- Define the points A and B
def A : ℝ × ℝ := (-2, 3)
def B : ℝ × ℝ := (4, -3)

-- Define the condition for a point (x, y) to be on a valid path
def on_valid_path (x y : ℤ) : Prop :=
  abs (x + 2) + abs (x - 4) + abs (y - 3) + abs (y + 3) ≤ 24

-- Define the set of integer coordinates that lie on at least one valid path
def valid_points : finset (ℤ × ℤ) :=
  {p | ∃ (x y : ℤ), p = (x, y) ∧ on_valid_path x y}.to_finset

-- The theorem to prove
theorem number_of_valid_points : valid_points.card = 243 := by
  sorry

end number_of_valid_points_l642_642643


namespace u_g_7_eq_l642_642891

noncomputable def u (x : ℝ) : ℝ := real.sqrt (4 * x + 2)
noncomputable def g (x : ℝ) : ℝ := 7 - u x

theorem u_g_7_eq : u (g 7) = real.sqrt (30 - 4 * real.sqrt 30) := by
sorry

end u_g_7_eq_l642_642891


namespace minimum_bailing_rate_l642_642560

theorem minimum_bailing_rate
  (distance_from_shore : Real := 1.5)
  (rowing_speed : Real := 3)
  (water_intake_rate : Real := 12)
  (max_water : Real := 45) :
  (distance_from_shore / rowing_speed) * 60 * water_intake_rate - max_water / ((distance_from_shore / rowing_speed) * 60) >= 10.5 :=
by
  -- Provide the units are consistent and the calculations agree with the given numerical data
  sorry

end minimum_bailing_rate_l642_642560


namespace count_two_digit_multiples_of_12_l642_642430

open Set

def is_twodigit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

def is_multiple_of_12 (n : ℕ) : Prop := ∃ k, n = 12 * k

theorem count_two_digit_multiples_of_12 : 
  ∃! (count : ℕ), 
  count = Finset.card 
    {n : ℕ | is_twodigit n ∧ is_multiple_of_12 n}.toFinset 
  ∧ count = 8 := 
by 
  sorry

end count_two_digit_multiples_of_12_l642_642430


namespace jordan_original_seat_l642_642462

/-- 
In a row of six seats, initially occupied by six friends. One friend, Jordan, leaves, causing some friends to shift positions:

- Alex moves one seat to the right.
- Sam moves two seats to the left.
- Taylor and Casey exchange seats.
- Dani moves one seat to the left.
  
Jordan returns to a seat that is not the leftmost seat (not seat 1).

We need to prove that Jordan must have originally occupied seat 4.
-/
theorem jordan_original_seat :
  -- Define the initial positions
  ∃ (seats : Fin 6 → String),
  seats 0 ≠ "Jordan" →
  (∀ j : Fin 6, j ≠ 0 → j ≠ 4 → seats (displace j) = "Jordan") →
  seats 4 = "Jordan" :=
begin
  sorry
end

-- Define the displacement function based on provided movements
noncomputable def displace : Fin 6 → Fin 6
| 0 => 1
| 1 => 3
| 2 => 4
| 3 => 2
| 4 => 0
| 5 => 2

end jordan_original_seat_l642_642462


namespace sum_of_polynomials_l642_642701

theorem sum_of_polynomials (d : ℕ) :
  let expr1 := 15 * d + 17 + 16 * d ^ 2
  let expr2 := 3 * d + 2
  let sum_expr := expr1 + expr2
  let a := 16
  let b := 18
  let c := 19
  sum_expr = a * d ^ 2 + b * d + c ∧ a + b + c = 53 := by
    sorry

end sum_of_polynomials_l642_642701


namespace coefficient_x2_in_polynomial_l642_642357

theorem coefficient_x2_in_polynomial :
  let p := 5 * (λ x : ℝ, x - 6) + 6 * (λ x : ℝ, 9 - 3 * x^2 + 2 * x) - 9 * (λ x : ℝ, 3 * x^2 - 2)
  (∀ (x : ℝ), p x → coeff_of_x2 p = 9) :=
by
  let p := 5 * (λ x : ℝ, x - 6) + 6 * (λ x : ℝ, 9 - 3 * x^2 + 2 * x) - 9 * (λ x : ℝ, 3 * x^2 - 2)
  intro x hp
  have h1 : coeff_of_x2 (λ x : ℝ, 5 * x - 30) = 0, sorry
  have h2 : coeff_of_x2 (λ x : ℝ, 6 * (9 - 3 * x^2 + 2 * x)) = -18, sorry
  have h3 : coeff_of_x2 (λ x : ℝ, -9 * (3 * x^2 - 2)) = 27, sorry
  rw [coeff_of_x2_add, h1, h2, h3]
  simp
  norm_num
  sorry

end coefficient_x2_in_polynomial_l642_642357


namespace cosine_150_eq_neg_sqrt3_div_2_l642_642240

theorem cosine_150_eq_neg_sqrt3_div_2 :
  (∀ θ : ℝ, θ = real.pi / 6) →
  (∀ θ : ℝ, cos (real.pi - θ) = -cos θ) →
  cos (5 * real.pi / 6) = -real.sqrt 3 / 2 :=
by
  intros hθ hcos_identity
  sorry

end cosine_150_eq_neg_sqrt3_div_2_l642_642240


namespace marble_problem_l642_642653

theorem marble_problem (total_marbles red_marbles blue_marbles : ℕ) 
  (_ : total_marbles = 160) 
  (_ : red_marbles = 40) 
  (_ : blue_marbles = 16) :
  let new_blue_marbles := blue_marbles + red_marbles / 3
  in new_blue_marbles = 29 :=
by
  have h1 : total_marbles = 160 := rfl
  have h2 : red_marbles = 40 := rfl
  have h3 : blue_marbles = 16 := rfl
  let new_blue_marbles := blue_marbles + red_marbles / 3
  have h_new_blue_marbles : new_blue_marbles = 29 := by
    rw [h2, h3]
    norm_num
  exact h_new_blue_marbles

end marble_problem_l642_642653


namespace intersection_max_difference_eq_zero_l642_642574

-- Definitions from the conditions
def f (x : ℝ) : ℝ := 5 - 2 * x^2 + x^4
def g (x : ℝ) : ℝ := 3 + 2 * x^2 + x^4

-- Definition of the maximum difference between y-coordinates of intersection points
def maxDifference : ℝ := 
  let x1 := sqrt (1/2)
  let y1 := f x1
  let x2 := -sqrt (1/2)
  let y2 := f x2
  abs (y1 - y2)

-- Statement to prove
theorem intersection_max_difference_eq_zero : maxDifference = 0 := by
  sorry

end intersection_max_difference_eq_zero_l642_642574


namespace cos_150_eq_neg_half_l642_642187

theorem cos_150_eq_neg_half :
  real.cos (150 * real.pi / 180) = -1 / 2 :=
sorry

end cos_150_eq_neg_half_l642_642187


namespace fx_root_and_decreasing_l642_642520

noncomputable def f (x : ℝ) : ℝ := (1 / 3) ^ x - Real.log x / Real.log 2

theorem fx_root_and_decreasing (a x0 : ℝ) (h0 : 0 < a) (hx0 : 0 < x0) (h_cond : a < x0) (hf_root : f x0 = 0) 
  (hf_decreasing : ∀ x y : ℝ, x < y → f y < f x) : f a > 0 := 
sorry

end fx_root_and_decreasing_l642_642520


namespace cosine_150_eq_neg_sqrt3_div_2_l642_642238

theorem cosine_150_eq_neg_sqrt3_div_2 :
  (∀ θ : ℝ, θ = real.pi / 6) →
  (∀ θ : ℝ, cos (real.pi - θ) = -cos θ) →
  cos (5 * real.pi / 6) = -real.sqrt 3 / 2 :=
by
  intros hθ hcos_identity
  sorry

end cosine_150_eq_neg_sqrt3_div_2_l642_642238


namespace number_of_coins_l642_642061

theorem number_of_coins (x : ℝ) (h : x + 0.50 * x + 0.25 * x + 0.10 * x + 0.05 * x = 315) : x ≈ 166 :=
sorry

end number_of_coins_l642_642061


namespace curve1_cartesian_curve2_cartesian_intersection_reciprocal_sum_l642_642417

def curve1 (t : ℝ) : ℝ × ℝ :=
  (1 + 1/2 * t, (sqrt 3)/2 * t)

def curve2 (rho theta : ℝ) : Prop :=
  rho^2 = 12 / (3 + sin theta^2)

def cartesian_curve1 (x y : ℝ) : Prop :=
  y = sqrt 3 * (x - 1)

def cartesian_curve2 (x y : ℝ) : Prop :=
  (x^2) / 4 + (y^2) / 3 = 1

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  (p1.1 - p2.1)^2 + (p1.2 - p2.2)^2

def pointF : ℝ × ℝ := (1, 0)

theorem curve1_cartesian (t : ℝ) :
  let p := curve1 t in cartesian_curve1 p.1 p.2 :=
by sorry

theorem curve2_cartesian (rho theta : ℝ) :
  curve2 rho theta → cartesian_curve2 (rho * cos theta) (rho * sin theta) :=
by sorry

theorem intersection_reciprocal_sum (t1 t2 : ℝ) (h1 : 5 * t1^2 + 4 * t1 - 12 = 0)
  (h2 : 5 * t2^2 + 4 * t2 - 12 = 0) :
  let A := curve1 t1 in
  let B := curve1 t2 in
  (1 / distance pointF A + 1 / distance pointF B) = 4 / 3 :=
by sorry

end curve1_cartesian_curve2_cartesian_intersection_reciprocal_sum_l642_642417


namespace number_of_n_l642_642363

theorem number_of_n (n : ℕ) (h1 : n > 0) (h2 : n ≤ 1200) (h3 : ∃ k : ℕ, 12 * n = k^2) :
  ∃ m : ℕ, m = 10 :=
by { sorry }

end number_of_n_l642_642363


namespace modulus_of_product_l642_642350

theorem modulus_of_product (z1 z2 : ℂ) (h1 : z1 = 7 - 4 * complex.I) (h2 : z2 = 3 + 2 * complex.I) :
  complex.abs (z1 * (complex.conj z2)) = real.sqrt 845 :=
by
  sorry

end modulus_of_product_l642_642350


namespace circumcircle_area_of_triangle_l642_642839

theorem circumcircle_area_of_triangle (A B C : Type) [EuclideanGeometry A] [Real B] [Real C] 
  (angle_A : Real) (side_b : Real) (area_ABC : Real) : 
  angle_A = 60 * (Real.pi / 180) →
  side_b = 2 →
  area_ABC = 2 * Real.sqrt 3 →
  let circumcircle_area : Real := (2 * Real.pi)^2 in
  circumcircle_area = 4 * Real.pi
  sorry

end circumcircle_area_of_triangle_l642_642839


namespace even_sum_probability_l642_642938

def numbers : List ℕ := [1, 2, 3, 5]

def total_combinations := Nat.choose (List.length numbers) 2

def even_sum_combinations := (numbers.combinations 2).count (λ l, l.sum % 2 = 0)

def probability_even_sum := even_sum_combinations / total_combinations

theorem even_sum_probability : probability_even_sum = 1 / 2 := 
by
  sorry

end even_sum_probability_l642_642938
