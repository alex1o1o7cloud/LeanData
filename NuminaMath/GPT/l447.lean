import Mathlib

namespace NUMINAMATH_GPT_rope_cut_prob_l447_44778

theorem rope_cut_prob (x : ℝ) (hx : 0 < x) : 
  (∃ (a b : ℝ), a + b = 1 ∧ min a b ≤ max a b / x) → 
  (1 / (x + 1) * 2) = 2 / (x + 1) :=
sorry

end NUMINAMATH_GPT_rope_cut_prob_l447_44778


namespace NUMINAMATH_GPT_product_of_solutions_eq_neg_nine_product_of_solutions_l447_44766

theorem product_of_solutions_eq_neg_nine :
  ∀ (x : ℝ), (|x| = 3 * (|x| - 2)) → x = 3 ∨ x = -3 :=
by
  sorry

theorem product_of_solutions :
  (∀ (x : ℝ), (|x| = 3 * (|x| - 2)) → (∃ (a b : ℝ), x = a ∨ x = b ∧ a * b = -9)) :=
by
  sorry

end NUMINAMATH_GPT_product_of_solutions_eq_neg_nine_product_of_solutions_l447_44766


namespace NUMINAMATH_GPT_complement_union_l447_44799

open Set

universe u

def U : Set ℕ := {1, 2, 3, 4}
def A : Set ℕ := {2, 4}
def B : Set ℕ := {1, 4}

theorem complement_union (U A B : Set ℕ) (hU : U = {1, 2, 3, 4}) (hA : A = {2, 4}) (hB : B = {1, 4}) :
  (U \ (A ∪ B)) = {3} :=
by
  simp [hU, hA, hB]
  sorry

end NUMINAMATH_GPT_complement_union_l447_44799


namespace NUMINAMATH_GPT_carlos_earnings_l447_44756

theorem carlos_earnings (h1 : ∃ w, 18 * w = w * 18) (h2 : ∃ w, 30 * w = w * 30) (h3 : ∀ w, 30 * w - 18 * w = 54) : 
  ∃ w, 18 * w + 30 * w = 216 := 
sorry

end NUMINAMATH_GPT_carlos_earnings_l447_44756


namespace NUMINAMATH_GPT_wind_velocity_determination_l447_44724

theorem wind_velocity_determination (ρ : ℝ) (P1 P2 : ℝ) (A1 A2 : ℝ) (V1 V2 : ℝ) (k : ℝ) :
  ρ = 1.2 →
  P1 = 0.75 →
  A1 = 2 →
  V1 = 12 →
  P1 = ρ * k * A1 * V1^2 →
  P2 = 20.4 →
  A2 = 10.76 →
  P2 = ρ * k * A2 * V2^2 →
  V2 = 27 := 
by sorry

end NUMINAMATH_GPT_wind_velocity_determination_l447_44724


namespace NUMINAMATH_GPT_intersection_A_complement_is_2_4_l447_44747

-- Declare the universal set U, set A, and set B
def U : Set ℕ := { 1, 2, 3, 4, 5, 6, 7 }
def A : Set ℕ := { 2, 4, 5 }
def B : Set ℕ := { 1, 3, 5, 7 }

-- Define the complement of set B with respect to U
def complement_U_B : Set ℕ := { x ∈ U | x ∉ B }

-- Define the intersection of set A and the complement of set B
def intersection_A_complement_U_B : Set ℕ := { x ∈ A | x ∈ complement_U_B }

-- State the theorem
theorem intersection_A_complement_is_2_4 : 
  intersection_A_complement_U_B = { 2, 4 } := 
by
  sorry

end NUMINAMATH_GPT_intersection_A_complement_is_2_4_l447_44747


namespace NUMINAMATH_GPT_battery_difference_l447_44707

def flashlights_batteries := 2
def toys_batteries := 15
def difference := 13

theorem battery_difference : toys_batteries - flashlights_batteries = difference :=
by
  sorry

end NUMINAMATH_GPT_battery_difference_l447_44707


namespace NUMINAMATH_GPT_blake_initial_money_l447_44735

theorem blake_initial_money (amount_spent_oranges amount_spent_apples amount_spent_mangoes change_received initial_amount : ℕ)
  (h1 : amount_spent_oranges = 40)
  (h2 : amount_spent_apples = 50)
  (h3 : amount_spent_mangoes = 60)
  (h4 : change_received = 150)
  (h5 : initial_amount = (amount_spent_oranges + amount_spent_apples + amount_spent_mangoes) + change_received) :
  initial_amount = 300 :=
by
  sorry

end NUMINAMATH_GPT_blake_initial_money_l447_44735


namespace NUMINAMATH_GPT_remaining_area_exclude_smaller_rectangles_l447_44734

-- Conditions from part a)
variables (x : ℕ)
def large_rectangle_area := (x + 8) * (x + 6)
def small1_rectangle_area := (2 * x - 1) * (x - 1)
def small2_rectangle_area := (x - 3) * (x - 5)

-- Proof statement from part c)
theorem remaining_area_exclude_smaller_rectangles :
  large_rectangle_area x - (small1_rectangle_area x - small2_rectangle_area x) = 25 * x + 62 :=
by
  sorry

end NUMINAMATH_GPT_remaining_area_exclude_smaller_rectangles_l447_44734


namespace NUMINAMATH_GPT_interval_length_l447_44701

theorem interval_length (a b : ℝ) (h : ∀ x : ℝ, a + 1 ≤ 3 * x + 6 ∧ 3 * x + 6 ≤ b - 2) :
  (b - a = 57) :=
sorry

end NUMINAMATH_GPT_interval_length_l447_44701


namespace NUMINAMATH_GPT_factorize_equivalence_l447_44731

-- declaring that the following definition may not be computable
noncomputable def factorize_expression (x y : ℝ) : Prop :=
  x^2 * y + x * y^2 = x * y * (x + y)

-- theorem to state the proof problem
theorem factorize_equivalence (x y : ℝ) : factorize_expression x y :=
sorry

end NUMINAMATH_GPT_factorize_equivalence_l447_44731


namespace NUMINAMATH_GPT_Megan_not_lead_actress_l447_44788

-- Define the conditions: total number of plays and lead actress percentage
def totalPlays : ℕ := 100
def leadActressPercentage : ℕ := 80

-- Define what we need to prove: the number of times Megan was not the lead actress
theorem Megan_not_lead_actress (totalPlays: ℕ) (leadActressPercentage: ℕ) : 
  (totalPlays * (100 - leadActressPercentage)) / 100 = 20 :=
by
  -- proof omitted
  sorry

end NUMINAMATH_GPT_Megan_not_lead_actress_l447_44788


namespace NUMINAMATH_GPT_grade_assignment_ways_l447_44728

theorem grade_assignment_ways (n_students : ℕ) (n_grades : ℕ) (h_students : n_students = 12) (h_grades : n_grades = 4) :
  (n_grades ^ n_students) = 16777216 := by
  rw [h_students, h_grades]
  rfl

end NUMINAMATH_GPT_grade_assignment_ways_l447_44728


namespace NUMINAMATH_GPT_max_stamps_l447_44784

def price_of_stamp : ℕ := 25  -- Price of one stamp in cents
def total_money : ℕ := 4000   -- Total money available in cents

theorem max_stamps : ∃ n : ℕ, price_of_stamp * n ≤ total_money ∧ (∀ m : ℕ, price_of_stamp * m ≤ total_money → m ≤ n) :=
by
  use 160
  sorry

end NUMINAMATH_GPT_max_stamps_l447_44784


namespace NUMINAMATH_GPT_rick_books_total_l447_44709

theorem rick_books_total 
  (N : ℕ)
  (h : N / 16 = 25) : 
  N = 400 := 
  sorry

end NUMINAMATH_GPT_rick_books_total_l447_44709


namespace NUMINAMATH_GPT_expected_participants_2008_l447_44761

theorem expected_participants_2008 (initial_participants : ℕ) (annual_increase_rate : ℝ) :
  initial_participants = 1000 ∧ annual_increase_rate = 1.25 →
  (initial_participants * annual_increase_rate ^ 3) = 1953.125 :=
by
  sorry

end NUMINAMATH_GPT_expected_participants_2008_l447_44761


namespace NUMINAMATH_GPT_spring_length_increase_l447_44785

-- Define the weight (x) and length (y) data points
def weights : List ℝ := [0, 1, 2, 3, 4, 5]
def lengths : List ℝ := [20, 20.5, 21, 21.5, 22, 22.5]

-- Prove that for each increase of 1 kg in weight, the length of the spring increases by 0.5 cm
theorem spring_length_increase (h : weights.length = lengths.length) :
  ∀ i, i < weights.length - 1 → (lengths.get! (i+1) - lengths.get! i) = 0.5 :=
by
  -- Proof goes here, omitted for now
  sorry

end NUMINAMATH_GPT_spring_length_increase_l447_44785


namespace NUMINAMATH_GPT_distributi_l447_44752

def number_of_distributions (spots : ℕ) (classes : ℕ) (min_spot_per_class : ℕ) : ℕ :=
  Nat.choose (spots - min_spot_per_class * classes + (classes - 1)) (classes - 1)

theorem distributi.on_of_10_spots (A B C : ℕ) (hA : A ≥ 1) (hB : B ≥ 1) (hC : C ≥ 1) 
(h_total : A + B + C = 10) : number_of_distributions 10 3 1 = 36 :=
by
  sorry

end NUMINAMATH_GPT_distributi_l447_44752


namespace NUMINAMATH_GPT_area_product_equal_no_consecutive_integers_l447_44781

open Real

-- Define the areas of the triangles for quadrilateral ABCD
variables {A B C D O : Point} 
variables {S1 S2 S3 S4 : Real}  -- Areas of triangles ABO, BCO, CDO, DAO

-- Given conditions
variables (h_intersection : lies_on_intersection O AC BD)
variables (h_areas : S1 = 1 / 2 * (|AO| * |BM|) ∧ S2 = 1 / 2 * (|CO| * |BM|) ∧ S3 = 1 / 2 * (|CO| * |DN|) ∧ S4 = 1 / 2 * (|AO| * |DN|))

-- Theorem for part (a)
theorem area_product_equal : S1 * S3 = S2 * S4 :=
by sorry

-- Theorem for part (b)
theorem no_consecutive_integers : ¬∃ (n : ℕ), S1 = n ∧ S2 = n + 1 ∧ S3 = n + 2 ∧ S4 = n + 3 :=
by sorry

end NUMINAMATH_GPT_area_product_equal_no_consecutive_integers_l447_44781


namespace NUMINAMATH_GPT_money_distribution_l447_44714

variable (A B C : ℕ)

theorem money_distribution :
  A + B + C = 500 →
  B + C = 360 →
  C = 60 →
  A + C = 200 :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_money_distribution_l447_44714


namespace NUMINAMATH_GPT_children_difference_l447_44712

-- Axiom definitions based on conditions
def initial_children : ℕ := 36
def first_stop_got_off : ℕ := 45
def first_stop_got_on : ℕ := 25
def second_stop_got_off : ℕ := 68
def final_children : ℕ := 12

-- Mathematical formulation of the problem and its proof statement
theorem children_difference :
  ∃ (x : ℕ), 
    initial_children - first_stop_got_off + first_stop_got_on - second_stop_got_off + x = final_children ∧ 
    (first_stop_got_off + second_stop_got_off) - (first_stop_got_on + x) = 24 :=
by 
  sorry

end NUMINAMATH_GPT_children_difference_l447_44712


namespace NUMINAMATH_GPT_width_of_roads_l447_44737

-- Definitions for the conditions
def length_of_lawn := 80 
def breadth_of_lawn := 60 
def total_cost := 5200 
def cost_per_sq_m := 4 

-- Derived condition: total area based on cost
def total_area_by_cost := total_cost / cost_per_sq_m 

-- Statement to prove: width of each road w is 65/7
theorem width_of_roads (w : ℚ) : (80 * w) + (60 * w) = total_area_by_cost → w = 65 / 7 :=
by
  sorry

end NUMINAMATH_GPT_width_of_roads_l447_44737


namespace NUMINAMATH_GPT_rectangle_area_l447_44798

theorem rectangle_area (x : ℝ) (h1 : x > 0) (h2 : x * 4 = 28) : x = 7 :=
sorry

end NUMINAMATH_GPT_rectangle_area_l447_44798


namespace NUMINAMATH_GPT_unique_solution_condition_l447_44764

theorem unique_solution_condition (a b c : ℝ) : 
  (∀ x : ℝ, 4 * x - 7 + a = (b + 1) * x + c) ↔ b ≠ 3 :=
by
  sorry

end NUMINAMATH_GPT_unique_solution_condition_l447_44764


namespace NUMINAMATH_GPT_find_difference_l447_44750

variable (a b c d e f : ℝ)

-- Conditions
def cond1 : Prop := a - b = c + d + 9
def cond2 : Prop := a + b = c - d - 3
def cond3 : Prop := e = a^2 + b^2
def cond4 : Prop := f = c^2 + d^2
def cond5 : Prop := f - e = 5 * a + 2 * b + 3 * c + 4 * d

-- Problem Statement
theorem find_difference (h1 : cond1 a b c d) (h2 : cond2 a b c d) (h3 : cond3 a b e) (h4 : cond4 c d f) (h5 : cond5 a b c d e f) : a - c = 3 :=
sorry

end NUMINAMATH_GPT_find_difference_l447_44750


namespace NUMINAMATH_GPT_remainder_3_pow_9_div_5_l447_44777

theorem remainder_3_pow_9_div_5 : (3^9) % 5 = 3 := by
  sorry

end NUMINAMATH_GPT_remainder_3_pow_9_div_5_l447_44777


namespace NUMINAMATH_GPT_bathroom_square_footage_l447_44732

theorem bathroom_square_footage
  (tiles_width : ℕ)
  (tiles_length : ℕ)
  (tile_size_inches : ℕ)
  (inches_per_foot : ℕ)
  (h1 : tiles_width = 10)
  (h2 : tiles_length = 20)
  (h3 : tile_size_inches = 6)
  (h4 : inches_per_foot = 12)
: (tiles_length * tile_size_inches / inches_per_foot) * (tiles_width * tile_size_inches / inches_per_foot) = 50 := 
by
  sorry

end NUMINAMATH_GPT_bathroom_square_footage_l447_44732


namespace NUMINAMATH_GPT_avg_waiting_time_is_1_point_2_minutes_l447_44716

/--
Assume that a Distracted Scientist immediately pulls out and recasts the fishing rod upon a bite,
doing so instantly. After this, he waits again. Consider a 6-minute interval.
During this time, the first rod receives 3 bites on average, and the second rod receives 2 bites
on average. Therefore, on average, there are 5 bites on both rods together in these 6 minutes.

We need to prove that the average waiting time for the first bite is 1.2 minutes.
-/
theorem avg_waiting_time_is_1_point_2_minutes :
  let first_rod_bites := 3
  let second_rod_bites := 2
  let total_time := 6 -- in minutes
  let total_bites := first_rod_bites + second_rod_bites
  let avg_rate := total_bites / total_time
  let avg_waiting_time := 1 / avg_rate
  avg_waiting_time = 1.2 := by
  sorry

end NUMINAMATH_GPT_avg_waiting_time_is_1_point_2_minutes_l447_44716


namespace NUMINAMATH_GPT_sequence_general_term_and_sum_l447_44789

theorem sequence_general_term_and_sum (a_n : ℕ → ℕ) (b_n S_n : ℕ → ℕ) :
  (∀ n, a_n n = 2 ^ n) ∧ (∀ n, b_n n = a_n n * (Real.logb 2 (a_n n)) ∧
  S_n n = (n - 1) * 2 ^ (n + 1) + 2) :=
by
  sorry

end NUMINAMATH_GPT_sequence_general_term_and_sum_l447_44789


namespace NUMINAMATH_GPT_first_wing_hall_rooms_l447_44767

theorem first_wing_hall_rooms
    (total_rooms : ℕ) (first_wing_floors : ℕ) (first_wing_halls_per_floor : ℕ)
    (second_wing_floors : ℕ) (second_wing_halls_per_floor : ℕ) (second_wing_rooms_per_hall : ℕ)
    (hotel_total_rooms : ℕ) (first_wing_total_rooms : ℕ) :
    hotel_total_rooms = total_rooms →
    first_wing_floors = 9 →
    first_wing_halls_per_floor = 6 →
    second_wing_floors = 7 →
    second_wing_halls_per_floor = 9 →
    second_wing_rooms_per_hall = 40 →
    hotel_total_rooms = first_wing_total_rooms + (second_wing_floors * second_wing_halls_per_floor * second_wing_rooms_per_hall) →
    first_wing_total_rooms = first_wing_floors * first_wing_halls_per_floor * 32 :=
by
  sorry

end NUMINAMATH_GPT_first_wing_hall_rooms_l447_44767


namespace NUMINAMATH_GPT_value_of_f_at_2_l447_44729

def f (x : ℝ) := x^2 + 2 * x - 3

theorem value_of_f_at_2 : f 2 = 5 :=
by
  sorry

end NUMINAMATH_GPT_value_of_f_at_2_l447_44729


namespace NUMINAMATH_GPT_find_z_l447_44746

theorem find_z (z : ℚ) : (7 + 11 + 23) / 3 = (15 + z) / 2 → z = 37 / 3 :=
by
  sorry

end NUMINAMATH_GPT_find_z_l447_44746


namespace NUMINAMATH_GPT_largest_number_in_sequence_l447_44758

noncomputable def largest_in_sequence (s : Fin 8 → ℝ) : ℝ :=
  max (s 0) (max (s 1) (max (s 2) (max (s 3) (max (s 4) (max (s 5) (max (s 6) (s 7)))))))

theorem largest_number_in_sequence (s : Fin 8 → ℝ)
  (h1 : ∀ i j : Fin 8, i < j → s i < s j)
  (h2 : ∃ i : Fin 5, (∃ d : ℝ, d = 4 ∨ d = 36) ∧ (∀ j : ℕ, j < 3 → s (i+j) + d = s (i+j+1)))
  (h3 : ∃ i : Fin 5, ∃ r : ℝ, (∀ j : ℕ, j < 3 → s (i+j) * r = s (i+j+1))) :
  largest_in_sequence s = 126 ∨ largest_in_sequence s = 6 :=
sorry

end NUMINAMATH_GPT_largest_number_in_sequence_l447_44758


namespace NUMINAMATH_GPT_geometric_progression_sum_of_cubes_l447_44792

theorem geometric_progression_sum_of_cubes :
  ∃ (a r : ℕ) (seq : Fin 6 → ℕ), (seq 0 = a) ∧ (seq 1 = a * r) ∧ (seq 2 = a * r^2) ∧ (seq 3 = a * r^3) ∧ (seq 4 = a * r^4) ∧ (seq 5 = a * r^5) ∧
  (∀ i, 0 ≤ seq i ∧ seq i < 100) ∧
  (seq 0 + seq 1 + seq 2 + seq 3 + seq 4 + seq 5 = 326) ∧
  (∃ T : ℕ, (∀ i, ∃ k, seq i = k^3 → k * k * k = seq i) ∧ T = 64) :=
sorry

end NUMINAMATH_GPT_geometric_progression_sum_of_cubes_l447_44792


namespace NUMINAMATH_GPT_no_such_graph_exists_l447_44748

noncomputable def vertex_degrees (n : ℕ) (deg : ℕ → ℕ) : Prop :=
  n ≥ 8 ∧
  ∃ (deg : ℕ → ℕ),
    (deg 0 = 4) ∧ (deg 1 = 5) ∧ ∀ i, 2 ≤ i ∧ i < n - 7 → deg i = i + 4 ∧
    (deg (n-7) = n-2) ∧ (deg (n-6) = n-2) ∧ (deg (n-5) = n-2) ∧
    (deg (n-4) = n-1) ∧ (deg (n-3) = n-1) ∧ (deg (n-2) = n-1)   

theorem no_such_graph_exists (n : ℕ) (deg : ℕ → ℕ) : 
  n ≥ 10 → ¬vertex_degrees n deg := 
by
  sorry

end NUMINAMATH_GPT_no_such_graph_exists_l447_44748


namespace NUMINAMATH_GPT_product_divisible_by_12_l447_44774

theorem product_divisible_by_12 (a b c d : ℤ) : 
  12 ∣ ((b - a) * (c - a) * (d - a) * (b - c) * (d - c) * (d - b)) :=
  sorry

end NUMINAMATH_GPT_product_divisible_by_12_l447_44774


namespace NUMINAMATH_GPT_union_of_sets_l447_44711

def setA : Set ℕ := {0, 1}
def setB : Set ℕ := {0, 2}

theorem union_of_sets : setA ∪ setB = {0, 1, 2} := 
sorry

end NUMINAMATH_GPT_union_of_sets_l447_44711


namespace NUMINAMATH_GPT_parabola_vertex_sum_l447_44768

theorem parabola_vertex_sum (p q r : ℝ) (h1 : ∀ x : ℝ, x = p * (x - 3)^2 + 2 → y) (h2 : p * (1 - 3)^2 + 2 = 6) :
  p + q + r = 6 :=
sorry

end NUMINAMATH_GPT_parabola_vertex_sum_l447_44768


namespace NUMINAMATH_GPT_trigon_expr_correct_l447_44719

noncomputable def trigon_expr : ℝ :=
  1 / Real.sin (Real.pi / 6) - 4 * Real.sin (Real.pi / 3)

theorem trigon_expr_correct : trigon_expr = 2 - 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_GPT_trigon_expr_correct_l447_44719


namespace NUMINAMATH_GPT_Paige_team_players_l447_44700

/-- Paige's team won their dodgeball game and scored 41 points total.
    If Paige scored 11 points and everyone else scored 6 points each,
    prove that the total number of players on the team was 6. -/
theorem Paige_team_players (total_points paige_points other_points : ℕ) (x : ℕ) (H1 : total_points = 41) (H2 : paige_points = 11) (H3 : other_points = 6) (H4 : paige_points + other_points * x = total_points) : x + 1 = 6 :=
by {
  sorry
}

end NUMINAMATH_GPT_Paige_team_players_l447_44700


namespace NUMINAMATH_GPT_original_population_correct_l447_44783

def original_population_problem :=
  let original_population := 6731
  let final_population := 4725
  let initial_disappeared := 0.10 * original_population
  let remaining_after_disappearance := original_population - initial_disappeared
  let left_after_remaining := 0.25 * remaining_after_disappearance
  let remaining_after_leaving := remaining_after_disappearance - left_after_remaining
  let disease_affected := 0.05 * original_population
  let disease_died := 0.02 * disease_affected
  let disease_migrated := 0.03 * disease_affected
  let remaining_after_disease := remaining_after_leaving - (disease_died + disease_migrated)
  let moved_to_village := 0.04 * remaining_after_disappearance
  let total_after_moving := remaining_after_disease + moved_to_village
  let births := 0.008 * original_population
  let deaths := 0.01 * original_population
  let final_population_calculated := total_after_moving + (births - deaths)
  final_population_calculated = final_population

theorem original_population_correct :
  original_population_problem ↔ True :=
by
  sorry

end NUMINAMATH_GPT_original_population_correct_l447_44783


namespace NUMINAMATH_GPT_quadratic_equivalence_statement_l447_44771

noncomputable def quadratic_in_cos (a b c x : ℝ) : Prop := 
  a * (Real.cos x)^2 + b * Real.cos x + c = 0

noncomputable def transform_to_cos2x (a b c : ℝ) : Prop := 
  (4*a^2) * (Real.cos (2*a))^2 + (2*a^2 + 4*a*c - 2*b^2) * Real.cos (2*a) + a^2 + 4*a*c - 2*b^2 + 4*c^2 = 0

theorem quadratic_equivalence_statement (a b c : ℝ) (h : quadratic_in_cos 4 2 (-1) a) :
  transform_to_cos2x 16 12 (-4) :=
sorry

end NUMINAMATH_GPT_quadratic_equivalence_statement_l447_44771


namespace NUMINAMATH_GPT_sum_of_squares_l447_44754

theorem sum_of_squares (x y : ℝ) (h1 : x + y = 22) (h2 : x * y = 40) : x^2 + y^2 = 404 :=
  sorry

end NUMINAMATH_GPT_sum_of_squares_l447_44754


namespace NUMINAMATH_GPT_range_of_x_when_m_is_4_range_of_m_l447_44762

-- Define the conditions for p and q
def p (x : ℝ) : Prop := x^2 - 7 * x + 10 < 0
def q (x m : ℝ) : Prop := x^2 - 4 * m * x + 3 * m^2 < 0
def neg_p (x : ℝ) : Prop := x ≤ 2 ∨ x ≥ 5
def neg_q (x m : ℝ) : Prop := x ≤ m ∨ x ≥ 3 * m

-- Define the conditions for the values of m
def cond_m_pos (m : ℝ) : Prop := m > 0
def cond_sufficient (m : ℝ) : Prop := cond_m_pos m ∧ m ≤ 2 ∧ 3 * m ≥ 5

-- Problem 1
theorem range_of_x_when_m_is_4 (x : ℝ) : p x ∧ q x 4 → 4 < x ∧ x < 5 :=
sorry

-- Problem 2
theorem range_of_m (m : ℝ) : (∀ x : ℝ, neg_q x m → neg_p x) → 5 / 3 ≤ m ∧ m ≤ 2 :=
sorry

end NUMINAMATH_GPT_range_of_x_when_m_is_4_range_of_m_l447_44762


namespace NUMINAMATH_GPT_train_capacity_l447_44782

theorem train_capacity (T : ℝ) (h : 2 * (T / 6) = 40) : T = 120 :=
sorry

end NUMINAMATH_GPT_train_capacity_l447_44782


namespace NUMINAMATH_GPT_sun_radius_scientific_notation_l447_44776

theorem sun_radius_scientific_notation : 
  (369000 : ℝ) = 3.69 * 10^5 :=
by
  sorry

end NUMINAMATH_GPT_sun_radius_scientific_notation_l447_44776


namespace NUMINAMATH_GPT_problem1_problem2_l447_44772

variables (x a : ℝ)

-- Proposition definitions
def proposition_p (a : ℝ) (x : ℝ) : Prop :=
  a > 0 ∧ (-x^2 + 4*a*x - 3*a^2) > 0

def proposition_q (x : ℝ) : Prop :=
  (x - 3) / (x - 2) < 0

-- Problems
theorem problem1 : (proposition_p 1 x ∧ proposition_q x) ↔ 2 < x ∧ x < 3 :=
by sorry

theorem problem2 : (¬ ∃ x, proposition_p a x) → (∀ x, ¬ proposition_q x) →
  1 ≤ a ∧ a ≤ 2 :=
by sorry

end NUMINAMATH_GPT_problem1_problem2_l447_44772


namespace NUMINAMATH_GPT_gasoline_distribution_impossible_l447_44706

theorem gasoline_distribution_impossible
  (x1 x2 x3 : ℝ)
  (h1 : x1 + x2 + x3 = 50)
  (h2 : x1 = x2 + 10)
  (h3 : x3 + 26 = x2) : false :=
by {
  sorry
}

end NUMINAMATH_GPT_gasoline_distribution_impossible_l447_44706


namespace NUMINAMATH_GPT_sum_of_four_digits_l447_44796

theorem sum_of_four_digits (EH OY AY OH : ℕ) (h1 : EH = 4 * OY) (h2 : AY = 4 * OH) : EH + OY + AY + OH = 150 :=
sorry

end NUMINAMATH_GPT_sum_of_four_digits_l447_44796


namespace NUMINAMATH_GPT_train_speed_including_stoppages_l447_44797

noncomputable def trainSpeedExcludingStoppages : ℝ := 45
noncomputable def stoppageTimePerHour : ℝ := 20 / 60 -- 20 minutes per hour converted to hours
noncomputable def runningTimePerHour : ℝ := 1 - stoppageTimePerHour

theorem train_speed_including_stoppages (speed : ℝ) (stoppage : ℝ) (running_time : ℝ) : 
  speed = 45 → stoppage = 20 / 60 → running_time = 1 - stoppage → 
  (speed * running_time) / 1 = 30 :=
by sorry

end NUMINAMATH_GPT_train_speed_including_stoppages_l447_44797


namespace NUMINAMATH_GPT_renu_suma_work_together_l447_44705

-- Define the time it takes for Renu to do the work by herself
def renu_days : ℕ := 6

-- Define the time it takes for Suma to do the work by herself
def suma_days : ℕ := 12

-- Define the work rate for Renu
def renu_work_rate : ℚ := 1 / renu_days

-- Define the work rate for Suma
def suma_work_rate : ℚ := 1 / suma_days

-- Define the combined work rate
def combined_work_rate : ℚ := renu_work_rate + suma_work_rate

-- Define the days it takes for both Renu and Suma to complete the work together
def days_to_complete_together : ℚ := 1 / combined_work_rate

-- The theorem stating that Renu and Suma can complete the work together in 4 days
theorem renu_suma_work_together : days_to_complete_together = 4 :=
by
  have h1 : renu_days = 6 := rfl
  have h2 : suma_days = 12 := rfl
  have h3 : renu_work_rate = 1 / 6 := by simp [renu_work_rate, h1]
  have h4 : suma_work_rate = 1 / 12 := by simp [suma_work_rate, h2]
  have h5 : combined_work_rate = 1 / 6 + 1 / 12 := by simp [combined_work_rate, h3, h4]
  have h6 : combined_work_rate = 1 / 4 := by norm_num [h5]
  have h7 : days_to_complete_together = 1 / (1 / 4) := by simp [days_to_complete_together, h6]
  have h8 : days_to_complete_together = 4 := by norm_num [h7]
  exact h8

end NUMINAMATH_GPT_renu_suma_work_together_l447_44705


namespace NUMINAMATH_GPT_value_of_X_is_one_l447_44713

-- Problem: Given the numbers 28 at the start of a row, 17 in the middle, and -15 in the same column as X,
-- we show the value of X must be 1 because the sequences are arithmetic.

theorem value_of_X_is_one (d : ℤ) (X : ℤ) :
  -- Conditions
  (17 - X = d) ∧ 
  (X - (-15) = d) ∧ 
  (d = 16) →
  -- Conclusion: X must be 1
  X = 1 :=
by 
  sorry

end NUMINAMATH_GPT_value_of_X_is_one_l447_44713


namespace NUMINAMATH_GPT_equation_solution_l447_44744

theorem equation_solution :
  ∃ a b c d : ℤ, a > 0 ∧ (∀ x : ℝ, (64 * x^2 + 96 * x - 36) = (a * x + b)^2 + d) ∧ c = -36 ∧ a + b + c + d = -94 :=
by sorry

end NUMINAMATH_GPT_equation_solution_l447_44744


namespace NUMINAMATH_GPT_ages_when_john_is_50_l447_44773

variable (age_john age_alice age_mike : ℕ)

-- Given conditions:
-- John is 10 years old
def john_is_10 : age_john = 10 := by sorry

-- Alice is twice John's age
def alice_is_twice_john : age_alice = 2 * age_john := by sorry

-- Mike is 4 years younger than Alice
def mike_is_4_years_younger : age_mike = age_alice - 4 := by sorry

-- Prove that when John is 50 years old, Alice will be 60 years old, and Mike will be 56 years old
theorem ages_when_john_is_50 : age_john = 50 → age_alice = 60 ∧ age_mike = 56 := 
by 
  intro h
  sorry

end NUMINAMATH_GPT_ages_when_john_is_50_l447_44773


namespace NUMINAMATH_GPT_olympiad_scores_above_18_l447_44757

theorem olympiad_scores_above_18 
  (n : Nat) 
  (scores : Fin n → ℕ) 
  (h_diff_scores : ∀ i j : Fin n, i ≠ j → scores i ≠ scores j) 
  (h_score_sum : ∀ i j k : Fin n, i ≠ j ∧ i ≠ k ∧ j ≠ k → scores i < scores j + scores k) 
  (h_n : n = 20) : 
  ∀ i : Fin n, scores i > 18 := 
by 
  -- See the proof for the detailed steps.
  sorry

end NUMINAMATH_GPT_olympiad_scores_above_18_l447_44757


namespace NUMINAMATH_GPT_value_depletion_rate_l447_44722

theorem value_depletion_rate (V_initial V_final : ℝ) (t : ℝ) (r : ℝ) :
  V_initial = 900 → V_final = 729 → t = 2 → V_final = V_initial * (1 - r)^t → r = 0.1 :=
by sorry

end NUMINAMATH_GPT_value_depletion_rate_l447_44722


namespace NUMINAMATH_GPT_art_piece_increase_l447_44790

theorem art_piece_increase (initial_price : ℝ) (multiplier : ℝ) (future_increase : ℝ) (h1 : initial_price = 4000) (h2 : multiplier = 3) :
  future_increase = (multiplier * initial_price) - initial_price :=
by
  rw [h1, h2]
  norm_num
  sorry

end NUMINAMATH_GPT_art_piece_increase_l447_44790


namespace NUMINAMATH_GPT_Sandy_change_l447_44742

theorem Sandy_change (pants shirt sweater shoes total paid change : ℝ)
  (h1 : pants = 13.58) (h2 : shirt = 10.29) (h3 : sweater = 24.97) (h4 : shoes = 39.99) (h5 : total = pants + shirt + sweater + shoes) (h6 : paid = 100) (h7 : change = paid - total) :
  change = 11.17 := 
sorry

end NUMINAMATH_GPT_Sandy_change_l447_44742


namespace NUMINAMATH_GPT_nancy_shoes_l447_44779

theorem nancy_shoes (boots_slippers_relation : ∀ (boots slippers : ℕ), slippers = boots + 9)
                    (heels_relation : ∀ (boots slippers heels : ℕ), heels = 3 * (boots + slippers)) :
                    ∃ (total_individual_shoes : ℕ), total_individual_shoes = 168 :=
by
  let boots := 6
  let slippers := boots + 9
  let total_pairs := boots + slippers
  let heels := 3 * total_pairs
  let total_pairs_shoes := boots + slippers + heels
  let total_individual_shoes := 2 * total_pairs_shoes
  use total_individual_shoes
  exact sorry

end NUMINAMATH_GPT_nancy_shoes_l447_44779


namespace NUMINAMATH_GPT_remainder_3_pow_20_mod_5_l447_44795

theorem remainder_3_pow_20_mod_5 : (3 ^ 20) % 5 = 1 := by
  sorry

end NUMINAMATH_GPT_remainder_3_pow_20_mod_5_l447_44795


namespace NUMINAMATH_GPT_xiaoxiao_types_faster_l447_44787

-- Defining the characters typed and time taken by both individuals
def characters_typed_taoqi : ℕ := 200
def time_taken_taoqi : ℕ := 5
def characters_typed_xiaoxiao : ℕ := 132
def time_taken_xiaoxiao : ℕ := 3

-- Calculating typing speeds
def speed_taoqi : ℕ := characters_typed_taoqi / time_taken_taoqi
def speed_xiaoxiao : ℕ := characters_typed_xiaoxiao / time_taken_xiaoxiao

-- Proving that 笑笑 types faster
theorem xiaoxiao_types_faster : speed_xiaoxiao > speed_taoqi := by
  -- Given calculations:
  -- speed_taoqi = 40
  -- speed_xiaoxiao = 44
  sorry

end NUMINAMATH_GPT_xiaoxiao_types_faster_l447_44787


namespace NUMINAMATH_GPT_gas_cost_per_gallon_l447_44786

theorem gas_cost_per_gallon (mpg : ℝ) (miles_per_day : ℝ) (days : ℝ) (total_cost : ℝ) : 
  mpg = 50 ∧ miles_per_day = 75 ∧ days = 10 ∧ total_cost = 45 → 
  (total_cost / ((miles_per_day * days) / mpg)) = 3 :=
by
  sorry

end NUMINAMATH_GPT_gas_cost_per_gallon_l447_44786


namespace NUMINAMATH_GPT_fred_earnings_l447_44725

-- Conditions as definitions
def initial_amount : ℕ := 23
def final_amount : ℕ := 86

-- Theorem to prove
theorem fred_earnings : final_amount - initial_amount = 63 := by
  sorry

end NUMINAMATH_GPT_fred_earnings_l447_44725


namespace NUMINAMATH_GPT_smallest_set_handshakes_l447_44739

-- Define the number of people
def num_people : Nat := 36

-- Define a type for people
inductive Person : Type
| a : Fin num_people → Person

-- Define the handshake relationship
def handshake (p1 p2 : Person) : Prop :=
  match p1, p2 with
  | Person.a i, Person.a j => i.val = (j.val + 1) % num_people ∨ j.val = (i.val + 1) % num_people

-- Define the problem statement
theorem smallest_set_handshakes :
  ∃ s : Finset Person, (∀ p : Person, p ∈ s ∨ ∃ q ∈ s, handshake p q) ∧ s.card = 18 :=
sorry

end NUMINAMATH_GPT_smallest_set_handshakes_l447_44739


namespace NUMINAMATH_GPT_solve_for_x_l447_44794

-- Assume x is a positive integer
def pos_integer (x : ℕ) : Prop := 0 < x

-- Assume the equation holds for some x
def equation (x : ℕ) : Prop :=
  1^(x+2) + 2^(x+1) + 3^(x-1) + 4^x = 1170

-- Proposition stating that if x satisfies the equation then x must be 5
theorem solve_for_x (x : ℕ) (h1 : pos_integer x) (h2 : equation x) : x = 5 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_x_l447_44794


namespace NUMINAMATH_GPT_triangle_side_length_l447_44703

   theorem triangle_side_length
   (A B C D E F : Type)
   (angle_bac angle_edf : Real)
   (AB AC DE DF : Real)
   (h1 : angle_bac = angle_edf)
   (h2 : AB = 5)
   (h3 : AC = 4)
   (h4 : DE = 2.5)
   (area_eq : (1 / 2) * AB * AC * Real.sin angle_bac = (1 / 2) * DE * DF * Real.sin angle_edf):
   DF = 8 :=
   by
   sorry
   
end NUMINAMATH_GPT_triangle_side_length_l447_44703


namespace NUMINAMATH_GPT_number_of_rectangles_containing_cell_l447_44740

theorem number_of_rectangles_containing_cell (m n p q : ℕ) (hp : 1 ≤ p ∧ p ≤ m) (hq : 1 ≤ q ∧ q ≤ n) :
    ∃ count : ℕ, count = p * q * (m - p + 1) * (n - q + 1) := 
    sorry

end NUMINAMATH_GPT_number_of_rectangles_containing_cell_l447_44740


namespace NUMINAMATH_GPT_product_of_two_numbers_l447_44751

theorem product_of_two_numbers :
  ∀ x y: ℝ, 
  ((x - y)^2) / ((x + y)^3) = 4 / 27 → 
  x + y = 5 * (x - y) + 3 → 
  x * y = 15.75 :=
by 
  intro x y
  sorry

end NUMINAMATH_GPT_product_of_two_numbers_l447_44751


namespace NUMINAMATH_GPT_distinct_three_digit_numbers_count_l447_44702

theorem distinct_three_digit_numbers_count : 
  ∃! n : ℕ, n = 5 * 4 * 3 :=
by
  use 60
  sorry

end NUMINAMATH_GPT_distinct_three_digit_numbers_count_l447_44702


namespace NUMINAMATH_GPT_total_coffee_blend_cost_l447_44720

-- Define the cost per pound of coffee types A and B
def cost_per_pound_A := 4.60
def cost_per_pound_B := 5.95

-- Given the pounds of coffee for Type A and the blend condition for Type B
def pounds_A := 67.52
def pounds_B := 2 * pounds_A

-- Total cost calculation
def total_cost := (pounds_A * cost_per_pound_A) + (pounds_B * cost_per_pound_B)

-- Theorem statement: The total cost of the coffee blend is $1114.08
theorem total_coffee_blend_cost : total_cost = 1114.08 := by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_total_coffee_blend_cost_l447_44720


namespace NUMINAMATH_GPT_jordans_greatest_average_speed_l447_44745

noncomputable def is_palindrome (n : ℕ) : Prop :=
  let s := n.digits 10
  s.reverse = s

theorem jordans_greatest_average_speed :
  ∃ (v : ℕ), 
  ∃ (d : ℕ), 
  ∃ (end_reading : ℕ), 
  is_palindrome 72327 ∧ 
  is_palindrome end_reading ∧ 
  72327 < end_reading ∧ 
  end_reading - 72327 = d ∧ 
  d ≤ 240 ∧ 
  end_reading ≤ 72327 + 240 ∧ 
  v = d / 4 ∧ 
  v = 50 :=
sorry

end NUMINAMATH_GPT_jordans_greatest_average_speed_l447_44745


namespace NUMINAMATH_GPT_greatest_ABCBA_l447_44769

/-
We need to prove that the greatest possible integer of the form AB,CBA 
that is both divisible by 11 and by 3, with A, B, and C being distinct digits, is 96569.
-/

theorem greatest_ABCBA (A B C : ℕ) 
  (h1 : A ≠ B ∧ B ≠ C ∧ A ≠ C) 
  (h2 : 1 ≤ A ∧ A ≤ 9 ∧ 0 ≤ B ∧ B ≤ 9 ∧ 0 ≤ C ∧ C ≤ 9) 
  (h3 : 10001 * A + 1010 * B + 100 * C < 100000) 
  (h4 : 2 * A - 2 * B + C ≡ 0 [MOD 11])
  (h5 : (2 * A + 2 * B + C) % 3 = 0) : 
  10001 * A + 1010 * B + 100 * C ≤ 96569 :=
sorry

end NUMINAMATH_GPT_greatest_ABCBA_l447_44769


namespace NUMINAMATH_GPT_two_vectors_less_than_45_deg_angle_l447_44753

theorem two_vectors_less_than_45_deg_angle (n : ℕ) (h : n = 30) (v : Fin n → ℝ → ℝ → ℝ) :
  ∃ i j : Fin n, i ≠ j ∧ ∃ θ : ℝ, θ < (45 * Real.pi / 180) :=
  sorry

end NUMINAMATH_GPT_two_vectors_less_than_45_deg_angle_l447_44753


namespace NUMINAMATH_GPT_vector_decomposition_l447_44710

def x : ℝ×ℝ×ℝ := (8, 0, 5)
def p : ℝ×ℝ×ℝ := (2, 0, 1)
def q : ℝ×ℝ×ℝ := (1, 1, 0)
def r : ℝ×ℝ×ℝ := (4, 1, 2)

theorem vector_decomposition :
  x = (1:ℝ) • p + (-2:ℝ) • q + (2:ℝ) • r :=
by
  sorry

end NUMINAMATH_GPT_vector_decomposition_l447_44710


namespace NUMINAMATH_GPT_lines_do_not_intersect_l447_44775

theorem lines_do_not_intersect (b : ℝ) :
  ∀ s v : ℝ,
    (2 + 3 * s = 5 + 6 * v) →
    (1 + 4 * s = 3 + 3 * v) →
    (b + 5 * s = 1 + 2 * v) →
    b ≠ -4/5 :=
by
  intros s v h1 h2 h3
  sorry

end NUMINAMATH_GPT_lines_do_not_intersect_l447_44775


namespace NUMINAMATH_GPT_range_of_m_l447_44759

theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, (x < m - 1 ∨ x > m + 1) → (x^2 - 2*x - 3 > 0)) ∧ 
  ¬(∀ x : ℝ, (x < m - 1 ∨ x > m + 1) ↔ (x^2 - 2*x - 3 > 0)) 
  ↔ 0 ≤ m ∧ m ≤ 2 :=
by 
  sorry

end NUMINAMATH_GPT_range_of_m_l447_44759


namespace NUMINAMATH_GPT_average_birds_seen_correct_l447_44708

-- Define the number of birds seen by each person
def birds_seen_by_marcus : ℕ := 7
def birds_seen_by_humphrey : ℕ := 11
def birds_seen_by_darrel : ℕ := 9

-- Define the number of people
def number_of_people : ℕ := 3

-- Calculate the total number of birds seen
def total_birds_seen : ℕ := birds_seen_by_marcus + birds_seen_by_humphrey + birds_seen_by_darrel

-- Calculate the average number of birds seen
def average_birds_seen : ℕ := total_birds_seen / number_of_people

-- Proof statement
theorem average_birds_seen_correct :
  average_birds_seen = 9 :=
by
  -- Leaving the proof out as instructed
  sorry

end NUMINAMATH_GPT_average_birds_seen_correct_l447_44708


namespace NUMINAMATH_GPT_interval_length_implies_difference_l447_44793

variable (c d : ℝ)

theorem interval_length_implies_difference (h1 : ∀ x : ℝ, c ≤ 3 * x + 5 ∧ 3 * x + 5 ≤ d) (h2 : (d - c) / 3 = 15) : d - c = 45 := 
sorry

end NUMINAMATH_GPT_interval_length_implies_difference_l447_44793


namespace NUMINAMATH_GPT_pet_store_cages_l447_44726

theorem pet_store_cages (initial_puppies sold_puppies puppies_per_cage : ℕ) 
  (h1 : initial_puppies = 78) (h2 : sold_puppies = 30) (h3 : puppies_per_cage = 8) : 
  (initial_puppies - sold_puppies) / puppies_per_cage = 6 := 
by 
  sorry

end NUMINAMATH_GPT_pet_store_cages_l447_44726


namespace NUMINAMATH_GPT_car_fuel_tanks_l447_44717

theorem car_fuel_tanks {x X p : ℝ}
  (h1 : x + X = 70)            -- Condition: total capacity is 70 liters
  (h2 : x * p = 45)            -- Condition: cost to fill small car's tank
  (h3 : X * (p + 0.29) = 68)   -- Condition: cost to fill large car's tank
  : x = 30 ∧ X = 40            -- Conclusion: capacities of the tanks
  :=
by {
  sorry
}

end NUMINAMATH_GPT_car_fuel_tanks_l447_44717


namespace NUMINAMATH_GPT_jan_drove_more_l447_44755

variables (d t s : ℕ)
variables (h h_ans : ℕ)
variables (ha_speed j_speed : ℕ)
variables (j d_plus : ℕ)

-- Ian's equation
def ian_distance (s t : ℕ) : ℕ := s * t

-- Han's additional conditions
def han_distance (s t : ℕ) (h_speed : ℕ)
    (d_plus : ℕ) : Prop :=
  d_plus + 120 = (s + h_speed) * (t + 2)

-- Jan's conditions and equation
def jan_distance (s t : ℕ) (j_speed : ℕ) : ℕ :=
  (s + j_speed) * (t + 3)

-- Proof statement
theorem jan_drove_more (d t s h_ans : ℕ)
    (h_speed j_speed : ℕ) (d_plus : ℕ)
    (h_dist_cond : han_distance s t h_speed d_plus)
    (j_dist_cond : jan_distance s t j_speed = h_ans) :
  h_ans = 195 :=
sorry

end NUMINAMATH_GPT_jan_drove_more_l447_44755


namespace NUMINAMATH_GPT_tickets_difference_l447_44743

def tickets_used_for_clothes : ℝ := 85
def tickets_used_for_accessories : ℝ := 45.5
def tickets_used_for_food : ℝ := 51
def tickets_used_for_toys : ℝ := 58

theorem tickets_difference : 
  (tickets_used_for_clothes + tickets_used_for_food + tickets_used_for_accessories) - tickets_used_for_toys = 123.5 := 
by
  sorry

end NUMINAMATH_GPT_tickets_difference_l447_44743


namespace NUMINAMATH_GPT_nancy_flooring_area_l447_44770

def area_of_rectangle (length : ℕ) (width : ℕ) : ℕ :=
  length * width

theorem nancy_flooring_area :
  let central_area_length := 10
  let central_area_width := 10
  let hallway_length := 6
  let hallway_width := 4
  let central_area := area_of_rectangle central_area_length central_area_width
  let hallway_area := area_of_rectangle hallway_length hallway_width
  let total_area := central_area + hallway_area
  total_area = 124 :=
by
  rfl  -- This is where the proof would go.

end NUMINAMATH_GPT_nancy_flooring_area_l447_44770


namespace NUMINAMATH_GPT_bridget_initial_skittles_l447_44718

theorem bridget_initial_skittles (b : ℕ) (h : b + 4 = 8) : b = 4 :=
by {
  sorry
}

end NUMINAMATH_GPT_bridget_initial_skittles_l447_44718


namespace NUMINAMATH_GPT_uncle_taller_than_james_l447_44723

def james_initial_height (uncle_height : ℕ) : ℕ := (2 * uncle_height) / 3

def james_final_height (initial_height : ℕ) (growth_spurt : ℕ) : ℕ := initial_height + growth_spurt

theorem uncle_taller_than_james (uncle_height : ℕ) (growth_spurt : ℕ) :
  uncle_height = 72 →
  growth_spurt = 10 →
  uncle_height - (james_final_height (james_initial_height uncle_height) growth_spurt) = 14 :=
by
  intros h1 h2
  rw [h1, h2]
  sorry

end NUMINAMATH_GPT_uncle_taller_than_james_l447_44723


namespace NUMINAMATH_GPT_simplify_expression_l447_44763

theorem simplify_expression : 0.4 * 0.5 + 0.3 * 0.2 = 0.26 := by
  sorry

end NUMINAMATH_GPT_simplify_expression_l447_44763


namespace NUMINAMATH_GPT_pascal_row_20_fifth_sixth_sum_l447_44727

-- Conditions from the problem
def pascal_element (n k : ℕ) : ℕ := Nat.choose n k

-- Question translated to a Lean theorem
theorem pascal_row_20_fifth_sixth_sum :
  pascal_element 20 4 + pascal_element 20 5 = 20349 :=
by
  sorry

end NUMINAMATH_GPT_pascal_row_20_fifth_sixth_sum_l447_44727


namespace NUMINAMATH_GPT_scenario1_scenario2_scenario3_l447_44704

noncomputable def scenario1_possible_situations : Nat :=
  12

noncomputable def scenario2_possible_situations : Nat :=
  144

noncomputable def scenario3_possible_situations : Nat :=
  50

theorem scenario1 (shots : Nat) (hits : Nat) (consecutive_hits : Nat) (remaining_hits : Nat) (not_consecutive : Prop) :
  shots = 10 ∧ hits = 7 ∧ consecutive_hits = 5 ∧ remaining_hits = 2 ∧ not_consecutive → 
  scenario1_possible_situations = 12 := by
  sorry

theorem scenario2 (shots : Nat) (hits : Nat) (consecutive_hits : Nat) (remaining_hits : Nat) :
  shots = 10 ∧ hits = 7 ∧ consecutive_hits = 4 ∧ remaining_hits = 3 → 
  scenario2_possible_situations = 144 := by
  sorry

theorem scenario3 (shots : Nat) (hits : Nat) (consecutive_hits : Nat) (remaining_hits : Nat) :
  shots = 10 ∧ hits = 6 ∧ consecutive_hits = 4 ∧ remaining_hits = 2 → 
  scenario3_possible_situations = 50 := by
  sorry

end NUMINAMATH_GPT_scenario1_scenario2_scenario3_l447_44704


namespace NUMINAMATH_GPT_figure_100_squares_l447_44721

theorem figure_100_squares :
  ∀ (f : ℕ → ℕ),
    (f 0 = 1) →
    (f 1 = 6) →
    (f 2 = 17) →
    (f 3 = 34) →
    f 100 = 30201 :=
by
  intros f h0 h1 h2 h3
  sorry

end NUMINAMATH_GPT_figure_100_squares_l447_44721


namespace NUMINAMATH_GPT_milk_butterfat_problem_l447_44738

-- Define the values given in the problem
def b1 : ℝ := 0.35  -- butterfat percentage of initial milk
def v1 : ℝ := 8     -- volume of initial milk in gallons
def b2 : ℝ := 0.10  -- butterfat percentage of milk to be added
def bf : ℝ := 0.20  -- desired butterfat percentage of the final mixture

-- Define the proof statement
theorem milk_butterfat_problem :
  ∃ x : ℝ, (2.8 + 0.1 * x) / (v1 + x) = bf ↔ x = 12 :=
by {
  sorry
}

end NUMINAMATH_GPT_milk_butterfat_problem_l447_44738


namespace NUMINAMATH_GPT_penguins_remaining_to_get_fish_l447_44736

def total_penguins : Nat := 36
def fed_penguins : Nat := 19

theorem penguins_remaining_to_get_fish : (total_penguins - fed_penguins = 17) :=
by
  sorry

end NUMINAMATH_GPT_penguins_remaining_to_get_fish_l447_44736


namespace NUMINAMATH_GPT_total_length_correct_l447_44730

-- Definitions for the first area's path length and scale.
def first_area_scale : ℕ := 500
def first_area_path_length_inches : ℕ := 6
def first_area_path_length_feet : ℕ := first_area_scale * first_area_path_length_inches

-- Definitions for the second area's path length and scale.
def second_area_scale : ℕ := 1000
def second_area_path_length_inches : ℕ := 3
def second_area_path_length_feet : ℕ := second_area_scale * second_area_path_length_inches

-- Total length represented by both paths in feet.
def total_path_length_feet : ℕ :=
  first_area_path_length_feet + second_area_path_length_feet

-- The Lean theorem proving that the total length is 6000 feet.
theorem total_length_correct : total_path_length_feet = 6000 := by
  sorry

end NUMINAMATH_GPT_total_length_correct_l447_44730


namespace NUMINAMATH_GPT_three_digit_integers_211_421_l447_44733

def is_one_more_than_multiple_of (n k : ℕ) : Prop :=
  ∃ m : ℕ, n = m * k + 1

theorem three_digit_integers_211_421
  (n : ℕ) (h1 : (100 ≤ n) ∧ (n ≤ 999))
  (h2 : is_one_more_than_multiple_of n 2)
  (h3 : is_one_more_than_multiple_of n 3)
  (h4 : is_one_more_than_multiple_of n 5)
  (h5 : is_one_more_than_multiple_of n 7) :
  n = 211 ∨ n = 421 :=
sorry

end NUMINAMATH_GPT_three_digit_integers_211_421_l447_44733


namespace NUMINAMATH_GPT_alex_silver_tokens_l447_44760

theorem alex_silver_tokens :
  ∃ x y : ℕ, 
    (100 - 3 * x + y ≤ 2) ∧ 
    (50 + 2 * x - 4 * y ≤ 3) ∧
    (x + y = 74) :=
by
  sorry

end NUMINAMATH_GPT_alex_silver_tokens_l447_44760


namespace NUMINAMATH_GPT_circles_tangent_internally_l447_44715

theorem circles_tangent_internally 
  (x y : ℝ) 
  (h : x^4 - 16 * x^2 + 2 * x^2 * y^2 - 16 * y^2 + y^4 = 4 * x^3 + 4 * x * y^2 - 64 * x) :
  ∃ c₁ c₂ : ℝ × ℝ, 
    (c₁ = (0, 0)) ∧ (c₂ = (2, 0)) ∧ 
    ((x - c₁.1)^2 + (y - c₁.2)^2 = 16) ∧ 
    ((x - c₂.1)^2 + (y - c₂.2)^2 = 4) ∧
    dist c₁ c₂ = 2 := 
sorry

end NUMINAMATH_GPT_circles_tangent_internally_l447_44715


namespace NUMINAMATH_GPT_remainder_when_concat_numbers_1_to_54_div_55_l447_44765

def concat_numbers (n : ℕ) : ℕ :=
  let digits x := x.digits 10
  (List.range n).bind digits |> List.reverse |> List.foldl (λ acc x => acc * 10 + x) 0

theorem remainder_when_concat_numbers_1_to_54_div_55 :
  let M := concat_numbers 55
  M % 55 = 44 :=
by
  sorry

end NUMINAMATH_GPT_remainder_when_concat_numbers_1_to_54_div_55_l447_44765


namespace NUMINAMATH_GPT_quadrilateral_angle_difference_l447_44791

theorem quadrilateral_angle_difference (h_ratio : ∀ (a b c d : ℕ), a = 3 * d ∧ b = 4 * d ∧ c = 5 * d ∧ d = 6 * d) 
  (h_sum : ∀ (a b c d : ℕ), a + b + c + d = 360) : 
  ∃ (x : ℕ), 6 * x - 3 * x = 60 := 
by 
  sorry

end NUMINAMATH_GPT_quadrilateral_angle_difference_l447_44791


namespace NUMINAMATH_GPT_train_platform_ratio_l447_44741

noncomputable def speed_km_per_hr := 216 -- condition 1
noncomputable def crossing_time_sec := 60 -- condition 2
noncomputable def train_length_m := 1800 -- condition 3

noncomputable def speed_m_per_s := speed_km_per_hr * 1000 / 3600
noncomputable def total_distance_m := speed_m_per_s * crossing_time_sec
noncomputable def platform_length_m := total_distance_m - train_length_m
noncomputable def ratio := train_length_m / platform_length_m

theorem train_platform_ratio : ratio = 1 := by
    sorry

end NUMINAMATH_GPT_train_platform_ratio_l447_44741


namespace NUMINAMATH_GPT_max_knights_on_island_l447_44749

theorem max_knights_on_island :
  ∃ n x, (n * (n - 1) = 90) ∧ (x * (10 - x) = 24) ∧ (x ≤ n) ∧ (∀ y, y * (10 - y) = 24 → y ≤ x) := sorry

end NUMINAMATH_GPT_max_knights_on_island_l447_44749


namespace NUMINAMATH_GPT_evaluate_expression_121point5_l447_44780

theorem evaluate_expression_121point5 :
  let x := (2 / 3 : ℝ)
  let y := (9 / 2 : ℝ)
  (1 / 3) * x^4 * y^5 = 121.5 :=
by
  let x := (2 / 3 : ℝ)
  let y := (9 / 2 : ℝ)
  sorry

end NUMINAMATH_GPT_evaluate_expression_121point5_l447_44780
