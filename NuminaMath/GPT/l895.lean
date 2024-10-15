import Mathlib

namespace NUMINAMATH_GPT_chemical_transformations_correct_l895_89505

def ethylbenzene : String := "C6H5CH2CH3"
def brominate (A : String) : String := "C6H5CH(Br)CH3"
def hydrolyze (B : String) : String := "C6H5CH(OH)CH3"
def dehydrate (C : String) : String := "C6H5CH=CH2"
def oxidize (D : String) : String := "C6H5COOH"
def brominate_with_catalyst (E : String) : String := "m-C6H4(Br)COOH"

def sequence_of_transformations : Prop :=
  ethylbenzene = "C6H5CH2CH3" ∧
  brominate ethylbenzene = "C6H5CH(Br)CH3" ∧
  hydrolyze (brominate ethylbenzene) = "C6H5CH(OH)CH3" ∧
  dehydrate (hydrolyze (brominate ethylbenzene)) = "C6H5CH=CH2" ∧
  oxidize (dehydrate (hydrolyze (brominate ethylbenzene))) = "C6H5COOH" ∧
  brominate_with_catalyst (oxidize (dehydrate (hydrolyze (brominate ethylbenzene)))) = "m-C6H4(Br)COOH"

theorem chemical_transformations_correct : sequence_of_transformations :=
by
  -- proof would go here
  sorry

end NUMINAMATH_GPT_chemical_transformations_correct_l895_89505


namespace NUMINAMATH_GPT_stools_count_l895_89581

theorem stools_count : ∃ x y : ℕ, 3 * x + 4 * y = 39 ∧ x = 3 := 
by
  sorry

end NUMINAMATH_GPT_stools_count_l895_89581


namespace NUMINAMATH_GPT_bus_stops_per_hour_l895_89542

theorem bus_stops_per_hour 
  (bus_speed_without_stoppages : Float)
  (bus_speed_with_stoppages : Float)
  (bus_stops_per_hour_in_minutes : Float) :
  bus_speed_without_stoppages = 60 ∧ 
  bus_speed_with_stoppages = 45 → 
  bus_stops_per_hour_in_minutes = 15 := by
  sorry

end NUMINAMATH_GPT_bus_stops_per_hour_l895_89542


namespace NUMINAMATH_GPT_find_c_l895_89598

theorem find_c (y : ℝ) (c : ℝ) (h1 : y > 0) (h2 : (6 * y / 20) + (c * y / 10) = 0.6 * y) : c = 3 :=
by 
  -- Skipping the proof
  sorry

end NUMINAMATH_GPT_find_c_l895_89598


namespace NUMINAMATH_GPT_hyperbola_focus_distance_l895_89583

theorem hyperbola_focus_distance :
  ∀ (x y : ℝ), (x^2 / 4 - y^2 / 3 = 1) → ∀ (F₁ F₂ : ℝ × ℝ), ∃ P : ℝ × ℝ, dist P F₁ = 3 → dist P F₂ = 7 :=
by
  sorry

end NUMINAMATH_GPT_hyperbola_focus_distance_l895_89583


namespace NUMINAMATH_GPT_lumber_cut_length_l895_89597

-- Define lengths of the pieces
def length_W : ℝ := 5
def length_X : ℝ := 3
def length_Y : ℝ := 5
def length_Z : ℝ := 4

-- Define distances from line M to the left end of the pieces
def distance_X : ℝ := 3
def distance_Y : ℝ := 2
def distance_Z : ℝ := 1.5

-- Define the total length of the pieces
def total_length : ℝ := 17

-- Define the length per side when cut by L
def length_per_side : ℝ := 8.5

theorem lumber_cut_length :
    (∃ (d : ℝ), 4 * d - 6.5 = 8.5 ∧ d = 3.75) :=
by
  sorry

end NUMINAMATH_GPT_lumber_cut_length_l895_89597


namespace NUMINAMATH_GPT_alien_home_planet_people_count_l895_89572

noncomputable def alien_earth_abduction (total_abducted returned_percentage taken_to_other_planet : ℕ) : ℕ :=
  let returned := total_abducted * returned_percentage / 100
  let remaining := total_abducted - returned
  remaining - taken_to_other_planet

theorem alien_home_planet_people_count :
  alien_earth_abduction 200 80 10 = 30 :=
by
  sorry

end NUMINAMATH_GPT_alien_home_planet_people_count_l895_89572


namespace NUMINAMATH_GPT_households_with_car_l895_89509

theorem households_with_car {H_total H_neither H_both H_bike_only : ℕ} 
    (cond1 : H_total = 90)
    (cond2 : H_neither = 11)
    (cond3 : H_both = 22)
    (cond4 : H_bike_only = 35) : 
    H_total - H_neither - (H_bike_only + H_both - H_both) + H_both = 44 := by
  sorry

end NUMINAMATH_GPT_households_with_car_l895_89509


namespace NUMINAMATH_GPT_problem_1_problem_2_l895_89540

noncomputable def a (k : ℝ) : ℝ × ℝ := (2, k)
noncomputable def b : ℝ × ℝ := (1, 1)
noncomputable def a_minus_3b (k : ℝ) : ℝ × ℝ := (2 - 3 * 1, k - 3 * 1)

-- First problem: Prove that k = 4 given vectors a and b, and the condition that b is perpendicular to (a - 3b)
theorem problem_1 (k : ℝ) (h : b.1 * (a_minus_3b k).1 + b.2 * (a_minus_3b k).2 = 0) : k = 4 :=
sorry

noncomputable def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1 * v.1 + v.2 * v.2)
noncomputable def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2
noncomputable def cosine (v w : ℝ × ℝ) : ℝ := dot_product v w / (magnitude v * magnitude w)

-- Second problem: Prove that the cosine value of the angle between a and b is 3√10/10 when k is 4
theorem problem_2 (k : ℝ) (hk : k = 4) : cosine (a k) b = 3 * Real.sqrt 10 / 10 :=
sorry

end NUMINAMATH_GPT_problem_1_problem_2_l895_89540


namespace NUMINAMATH_GPT_find_number_l895_89525

theorem find_number (x : ℕ) (h : 5 * x = 100) : x = 20 :=
by
  sorry

end NUMINAMATH_GPT_find_number_l895_89525


namespace NUMINAMATH_GPT_triangle_sides_inequality_l895_89571

theorem triangle_sides_inequality (a b c : ℝ) (h1 : a + b > c) (h2 : a + c > b) (h3 : b + c > a) :
  a^2 - 2 * a * b + b^2 - c^2 < 0 :=
by
  sorry

end NUMINAMATH_GPT_triangle_sides_inequality_l895_89571


namespace NUMINAMATH_GPT_existence_of_intersection_l895_89553

def setA (m : ℝ) : Set (ℝ × ℝ) := { p | ∃ (x y : ℝ), p = (x, y) ∧ (x^2 + m * x - y + 2 = 0) }
def setB : Set (ℝ × ℝ) := { p | ∃ (x y : ℝ), p = (x, y) ∧ (x - y + 1 = 0) ∧ (0 ≤ x ∧ x ≤ 2) }

theorem existence_of_intersection (m : ℝ) : (∃ (p : ℝ × ℝ), p ∈ (setA m ∩ setB)) ↔ m ≤ -1 := 
sorry

end NUMINAMATH_GPT_existence_of_intersection_l895_89553


namespace NUMINAMATH_GPT_measure_weights_l895_89538

theorem measure_weights (w1 w3 w7 : Nat) (h1 : w1 = 1) (h3 : w3 = 3) (h7 : w7 = 7) :
  ∃ s : Finset Nat, s.card = 7 ∧ 
    (1 ∈ s) ∧ (3 ∈ s) ∧ (7 ∈ s) ∧
    (4 ∈ s) ∧ (8 ∈ s) ∧ (10 ∈ s) ∧ 
    (11 ∈ s) := 
by
  sorry

end NUMINAMATH_GPT_measure_weights_l895_89538


namespace NUMINAMATH_GPT_composite_10201_in_all_bases_greater_than_two_composite_10101_in_all_bases_l895_89584

-- Definition for part (a)
def composite_base_greater_than_two (n : ℕ) : Prop :=
  ∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ (n^4 + 2*n^2 + 1) = a * b

-- Proof statement for part (a)
theorem composite_10201_in_all_bases_greater_than_two (n : ℕ) (h : n > 2) : composite_base_greater_than_two n :=
by sorry

-- Definition for part (b)
def composite_in_all_bases (n : ℕ) : Prop :=
  ∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ (n^4 + n^2 + 1) = a * b

-- Proof statement for part (b)
theorem composite_10101_in_all_bases (n : ℕ) : composite_in_all_bases n :=
by sorry

end NUMINAMATH_GPT_composite_10201_in_all_bases_greater_than_two_composite_10101_in_all_bases_l895_89584


namespace NUMINAMATH_GPT_largest_multiple_of_7_less_than_neg50_l895_89568

theorem largest_multiple_of_7_less_than_neg50 : ∃ x, (∃ k : ℤ, x = 7 * k) ∧ x < -50 ∧ ∀ y, (∃ m : ℤ, y = 7 * m) → y < -50 → y ≤ x :=
sorry

end NUMINAMATH_GPT_largest_multiple_of_7_less_than_neg50_l895_89568


namespace NUMINAMATH_GPT_correct_option_D_l895_89558

variables {a b m : Type}
variables {α β : Type}

axiom parallel (x y : Type) : Prop
axiom perpendicular (x y : Type) : Prop

variables (a_parallel_b : parallel a b)
variables (a_parallel_alpha : parallel a α)

variables (alpha_perpendicular_beta : perpendicular α β)
variables (a_parallel_alpha : parallel a α)

variables (alpha_parallel_beta : parallel α β)
variables (m_perpendicular_alpha : perpendicular m α)

theorem correct_option_D : parallel α β ∧ perpendicular m α → perpendicular m β := sorry

end NUMINAMATH_GPT_correct_option_D_l895_89558


namespace NUMINAMATH_GPT_remainder_of_polynomial_division_is_88_l895_89536

def p (x : ℝ) : ℝ := 4*x^5 - 3*x^4 + 5*x^3 - 7*x^2 + 3*x - 10

theorem remainder_of_polynomial_division_is_88 :
  p 2 = 88 :=
by
  sorry

end NUMINAMATH_GPT_remainder_of_polynomial_division_is_88_l895_89536


namespace NUMINAMATH_GPT_determine_m_range_l895_89575

theorem determine_m_range (m : ℝ) (h : (∃ (x y : ℝ), x^2 + y^2 + 2 * m * x + 2 = 0) ∧ 
                                    (∃ (r : ℝ) (h_r : r^2 = m^2 - 2), π * r^2 ≥ 4 * π)) :
  (m ≤ -Real.sqrt 6 ∨ m ≥ Real.sqrt 6) :=
by
  sorry

end NUMINAMATH_GPT_determine_m_range_l895_89575


namespace NUMINAMATH_GPT_joeys_votes_l895_89508

theorem joeys_votes
  (M B J : ℕ) 
  (h1 : M = 66) 
  (h2 : M = 3 * B) 
  (h3 : B = 2 * (J + 3)) : 
  J = 8 := 
by 
  sorry

end NUMINAMATH_GPT_joeys_votes_l895_89508


namespace NUMINAMATH_GPT_average_home_runs_correct_l895_89537

-- Define the number of players hitting specific home runs
def players_5_hr : ℕ := 3
def players_7_hr : ℕ := 2
def players_9_hr : ℕ := 1
def players_11_hr : ℕ := 2
def players_13_hr : ℕ := 1

-- Calculate the total number of home runs and total number of players
def total_hr : ℕ := 5 * players_5_hr + 7 * players_7_hr + 9 * players_9_hr + 11 * players_11_hr + 13 * players_13_hr
def total_players : ℕ := players_5_hr + players_7_hr + players_9_hr + players_11_hr + players_13_hr

-- Calculate the average number of home runs
def average_home_runs : ℚ := total_hr / total_players

-- The theorem we need to prove
theorem average_home_runs_correct : average_home_runs = 73 / 9 :=
by
  sorry

end NUMINAMATH_GPT_average_home_runs_correct_l895_89537


namespace NUMINAMATH_GPT_sam_drove_distance_l895_89556

theorem sam_drove_distance (marguerite_distance : ℝ) (marguerite_time : ℝ) (sam_time : ℝ) :
  marguerite_distance = 150 ∧ marguerite_time = 3 ∧ sam_time = 4 →
  (sam_time * (marguerite_distance / marguerite_time) = 200) :=
by
  sorry

end NUMINAMATH_GPT_sam_drove_distance_l895_89556


namespace NUMINAMATH_GPT_sara_change_l895_89592

def cost_of_first_book : ℝ := 5.5
def cost_of_second_book : ℝ := 6.5
def amount_given : ℝ := 20.0
def total_cost : ℝ := cost_of_first_book + cost_of_second_book
def change : ℝ := amount_given - total_cost

theorem sara_change : change = 8 :=
by
  have total_cost_correct : total_cost = 12.0 := by sorry
  have change_correct : change = amount_given - total_cost := by sorry
  show change = 8
  sorry

end NUMINAMATH_GPT_sara_change_l895_89592


namespace NUMINAMATH_GPT_solve_quad_linear_system_l895_89594

theorem solve_quad_linear_system :
  (∃ x y : ℝ, x^2 - 6 * x + 8 = 0 ∧ y + 2 * x = 12 ∧ ((x, y) = (4, 4) ∨ (x, y) = (2, 8))) :=
sorry

end NUMINAMATH_GPT_solve_quad_linear_system_l895_89594


namespace NUMINAMATH_GPT_not_prime_4k4_plus_1_not_prime_k4_plus_4_l895_89564

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem not_prime_4k4_plus_1 (k : ℕ) (hk : k > 0) : ¬ is_prime (4 * k^4 + 1) :=
by sorry

theorem not_prime_k4_plus_4 (k : ℕ) (hk : k > 0) : ¬ is_prime (k^4 + 4) :=
by sorry

end NUMINAMATH_GPT_not_prime_4k4_plus_1_not_prime_k4_plus_4_l895_89564


namespace NUMINAMATH_GPT_right_triangles_sides_l895_89589

theorem right_triangles_sides (a b c p S r DH FC FH: ℝ)
  (h₁ : a = 10)
  (h₂ : b = 10)
  (h₃ : c = 12)
  (h₄ : p = (a + b + c) / 2)
  (h₅ : S = Real.sqrt (p * (p - a) * (p - b) * (p - c)))
  (h₆ : r = S / p)
  (h₇ : DH = (c / 2) - r)
  (h₈ : FC = (a * r) / DH)
  (h₉ : FH = Real.sqrt (FC^2 - DH^2))
: FC = 3 ∧ DH = 4 ∧ FH = 5 := by
  sorry

end NUMINAMATH_GPT_right_triangles_sides_l895_89589


namespace NUMINAMATH_GPT_downloaded_data_l895_89516

/-- 
  Mason is trying to download a 880 MB game to his phone. After downloading some amount, his Internet
  connection slows to 3 MB/minute. It will take him 190 more minutes to download the game. Prove that 
  Mason has downloaded 310 MB before his connection slowed down. 
-/
theorem downloaded_data (total_size : ℕ) (speed : ℕ) (time_remaining : ℕ) (remaining_data : ℕ) (downloaded : ℕ) :
  total_size = 880 ∧
  speed = 3 ∧
  time_remaining = 190 ∧
  remaining_data = speed * time_remaining ∧
  downloaded = total_size - remaining_data →
  downloaded = 310 := 
by 
  sorry

end NUMINAMATH_GPT_downloaded_data_l895_89516


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_l895_89591

theorem sufficient_but_not_necessary (a : ℝ) : a = 1 → |a| = 1 ∧ (|a| = 1 → a = 1 → false) :=
by
  sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_l895_89591


namespace NUMINAMATH_GPT_total_oranges_is_correct_l895_89501

-- Definitions based on the problem's conditions
def layer_count : ℕ := 6
def base_length : ℕ := 9
def base_width : ℕ := 6

-- Function to compute the number of oranges in a layer given the current dimensions
def oranges_in_layer (length width : ℕ) : ℕ :=
  length * width

-- Function to compute the total number of oranges in the stack
def total_oranges_in_stack (base_length base_width : ℕ) : ℕ :=
  oranges_in_layer base_length base_width +
  oranges_in_layer (base_length - 1) (base_width - 1) +
  oranges_in_layer (base_length - 2) (base_width - 2) +
  oranges_in_layer (base_length - 3) (base_width - 3) +
  oranges_in_layer (base_length - 4) (base_width - 4) +
  oranges_in_layer (base_length - 5) (base_width - 5)

-- The theorem to be proved
theorem total_oranges_is_correct : total_oranges_in_stack 9 6 = 154 := by
  sorry

end NUMINAMATH_GPT_total_oranges_is_correct_l895_89501


namespace NUMINAMATH_GPT_num_terms_arithmetic_seq_l895_89550

theorem num_terms_arithmetic_seq (a d l : ℝ) (n : ℕ)
  (h1 : a = 3.25) 
  (h2 : d = 4)
  (h3 : l = 55.25)
  (h4 : l = a + (↑n - 1) * d) :
  n = 14 :=
by
  sorry

end NUMINAMATH_GPT_num_terms_arithmetic_seq_l895_89550


namespace NUMINAMATH_GPT_ann_trip_longer_than_mary_l895_89590

-- Define constants for conditions
def mary_hill_length : ℕ := 630
def mary_speed : ℕ := 90
def ann_hill_length : ℕ := 800
def ann_speed : ℕ := 40

-- Define a theorem to express the question and correct answer
theorem ann_trip_longer_than_mary : 
  (ann_hill_length / ann_speed - mary_hill_length / mary_speed) = 13 :=
by
  -- Now insert sorry to leave the proof unfinished
  sorry

end NUMINAMATH_GPT_ann_trip_longer_than_mary_l895_89590


namespace NUMINAMATH_GPT_lunks_for_apples_l895_89548

theorem lunks_for_apples : 
  (∀ (a : ℕ) (b : ℕ) (k : ℕ), 3 * b * k = 5 * a → 15 * k = 9 * a ∧ 2 * a * 9 = 4 * b * 9 → 15 * 2 * a / 4 = 18) :=
by
  intro a b k h1 h2
  sorry

end NUMINAMATH_GPT_lunks_for_apples_l895_89548


namespace NUMINAMATH_GPT_acute_angle_condition_l895_89528

theorem acute_angle_condition 
  (m : ℝ) 
  (a : ℝ × ℝ := (2,1))
  (b : ℝ × ℝ := (m,6)) 
  (dot_product := a.1 * b.1 + a.2 * b.2)
  (magnitude_a := Real.sqrt (a.1 * a.1 + a.2 * a.2))
  (magnitude_b := Real.sqrt (b.1 * b.1 + b.2 * b.2))
  (cos_angle := dot_product / (magnitude_a * magnitude_b))
  (acute_angle : cos_angle > 0) : -3 < m ∧ m ≠ 12 :=
sorry

end NUMINAMATH_GPT_acute_angle_condition_l895_89528


namespace NUMINAMATH_GPT_triangle_inequality_proof_l895_89578

theorem triangle_inequality_proof (a b c : ℝ) (h1 : a + b > c) (h2 : b + c > a) (h3 : c + a > b) :
  a^2 * b * (a - b) + b^2 * c * (b - c) + c^2 * a * (c - a) ≥ 0 :=
sorry

end NUMINAMATH_GPT_triangle_inequality_proof_l895_89578


namespace NUMINAMATH_GPT_find_number_l895_89576

theorem find_number (x : ℝ) (h : 100 - x = x + 40) : x = 30 :=
sorry

end NUMINAMATH_GPT_find_number_l895_89576


namespace NUMINAMATH_GPT_marcy_drinks_in_250_minutes_l895_89545

-- Define a function to represent that Marcy takes n minutes to drink x liters of water.
def time_to_drink (minutes_per_sip : ℕ) (sip_volume_ml : ℕ) (total_volume_liters : ℕ) : ℕ :=
  let total_volume_ml := total_volume_liters * 1000
  let sips := total_volume_ml / sip_volume_ml
  sips * minutes_per_sip

theorem marcy_drinks_in_250_minutes :
  time_to_drink 5 40 2 = 250 :=
  by
    -- The function definition and its application will show this value holds.
    sorry

end NUMINAMATH_GPT_marcy_drinks_in_250_minutes_l895_89545


namespace NUMINAMATH_GPT_shares_of_valuable_stock_l895_89588

theorem shares_of_valuable_stock 
  (price_val : ℕ := 78)
  (price_oth : ℕ := 39)
  (shares_oth : ℕ := 26)
  (total_asset : ℕ := 2106)
  (x : ℕ) 
  (h_val_stock : total_asset = 78 * x + 39 * 26) : 
  x = 14 :=
by
  sorry

end NUMINAMATH_GPT_shares_of_valuable_stock_l895_89588


namespace NUMINAMATH_GPT_diamond_expression_evaluation_l895_89504

def diamond (a b : ℚ) : ℚ := a - (1 / b)

theorem diamond_expression_evaluation :
  ((diamond (diamond 2 3) 4) - (diamond 2 (diamond 3 4))) = -29 / 132 :=
by {
    sorry
}

end NUMINAMATH_GPT_diamond_expression_evaluation_l895_89504


namespace NUMINAMATH_GPT_smallest_n_13n_congruent_456_mod_5_l895_89595

theorem smallest_n_13n_congruent_456_mod_5 : ∃ n : ℕ, (n > 0) ∧ (13 * n ≡ 456 [MOD 5]) ∧ (∀ m : ℕ, (m > 0 ∧ 13 * m ≡ 456 [MOD 5]) → n ≤ m) :=
by
  sorry

end NUMINAMATH_GPT_smallest_n_13n_congruent_456_mod_5_l895_89595


namespace NUMINAMATH_GPT_max_area_of_cone_l895_89523

noncomputable def max_cross_sectional_area (l θ : ℝ) : ℝ := (1/2) * l^2 * Real.sin θ

theorem max_area_of_cone :
  (∀ θ, 0 ≤ θ ∧ θ ≤ (2 * Real.pi / 3) → max_cross_sectional_area 3 θ ≤ (9 / 2))
  ∧ (∃ θ, 0 ≤ θ ∧ θ ≤ (2 * Real.pi / 3) ∧ max_cross_sectional_area 3 θ = (9 / 2)) := 
by
  sorry

end NUMINAMATH_GPT_max_area_of_cone_l895_89523


namespace NUMINAMATH_GPT_number_of_unit_fraction_pairs_l895_89534

/-- 
 The number of ways that 1/2007 can be expressed as the sum of two distinct positive unit fractions is 7.
-/
theorem number_of_unit_fraction_pairs : 
  ∃ (pairs : Finset (ℕ × ℕ)), 
    (∀ p ∈ pairs, p.1 ≠ p.2 ∧ (1 : ℚ) / 2007 = 1 / ↑p.1 + 1 / ↑p.2) ∧ 
    pairs.card = 7 :=
sorry

end NUMINAMATH_GPT_number_of_unit_fraction_pairs_l895_89534


namespace NUMINAMATH_GPT_at_most_one_true_l895_89563

theorem at_most_one_true (p q : Prop) (h : ¬(p ∧ q)) : ¬(p ∧ q ∧ ¬(¬p ∧ ¬q)) :=
by
  sorry

end NUMINAMATH_GPT_at_most_one_true_l895_89563


namespace NUMINAMATH_GPT_max_value_of_expression_l895_89502

theorem max_value_of_expression {a x1 x2 : ℝ}
  (h1 : x1^2 + a * x1 + a = 2)
  (h2 : x2^2 + a * x2 + a = 2)
  (h1_ne_x2 : x1 ≠ x2) :
  ∃ a : ℝ, (x1 - 2 * x2) * (x2 - 2 * x1) = -63 / 8 :=
by
  sorry

end NUMINAMATH_GPT_max_value_of_expression_l895_89502


namespace NUMINAMATH_GPT_equal_playtime_l895_89532

theorem equal_playtime (children : ℕ) (total_minutes : ℕ) (simultaneous_players : ℕ) (equal_playtime_per_child : ℕ)
  (h1 : children = 12) (h2 : total_minutes = 120) (h3 : simultaneous_players = 2) (h4 : equal_playtime_per_child = (simultaneous_players * total_minutes) / children) :
  equal_playtime_per_child = 20 := 
by sorry

end NUMINAMATH_GPT_equal_playtime_l895_89532


namespace NUMINAMATH_GPT_sum_last_two_digits_9_pow_23_plus_11_pow_23_l895_89521

theorem sum_last_two_digits_9_pow_23_plus_11_pow_23 :
  (9^23 + 11^23) % 100 = 60 :=
by
  sorry

end NUMINAMATH_GPT_sum_last_two_digits_9_pow_23_plus_11_pow_23_l895_89521


namespace NUMINAMATH_GPT_x_equals_y_squared_plus_2y_minus_1_l895_89574

theorem x_equals_y_squared_plus_2y_minus_1 (x y : ℝ) (h : x / (x - 1) = (y^2 + 2 * y - 1) / (y^2 + 2 * y - 2)) : 
  x = y^2 + 2 * y - 1 :=
sorry

end NUMINAMATH_GPT_x_equals_y_squared_plus_2y_minus_1_l895_89574


namespace NUMINAMATH_GPT_correct_operation_l895_89524

noncomputable def valid_operation (n : ℕ) (a b : ℕ) (c d : ℤ) (x : ℚ) : Prop :=
  match n with
  | 0 => (x ^ a / x ^ b = x ^ (a - b))
  | 1 => (x ^ a * x ^ b = x ^ (a + b))
  | 2 => (c * x ^ a + d * x ^ a = (c + d) * x ^ a)
  | 3 => ((c * x ^ a) ^ b = c ^ b * x ^ (a * b))
  | _ => False

theorem correct_operation (x : ℚ) : valid_operation 1 2 3 0 0 x :=
by sorry

end NUMINAMATH_GPT_correct_operation_l895_89524


namespace NUMINAMATH_GPT_squirrel_acorns_l895_89530

theorem squirrel_acorns :
  ∃ (c s r : ℕ), (4 * c = 5 * s) ∧ (3 * r = 4 * c) ∧ (r = s + 3) ∧ (5 * s = 40) :=
by
  sorry

end NUMINAMATH_GPT_squirrel_acorns_l895_89530


namespace NUMINAMATH_GPT_find_number_l895_89554

theorem find_number (x : ℝ) (h : 61 + 5 * 12 / (180 / x) = 62): x = 3 :=
by
  sorry

end NUMINAMATH_GPT_find_number_l895_89554


namespace NUMINAMATH_GPT_cos_beta_value_l895_89551

theorem cos_beta_value
  (α β : ℝ)
  (hαβ : 0 < α ∧ α < π ∧ 0 < β ∧ β < π)
  (h1 : Real.sin (α + β) = 5 / 13)
  (h2 : Real.tan (α / 2) = 1 / 2) :
  Real.cos β = -16 / 65 := 
by 
  sorry

end NUMINAMATH_GPT_cos_beta_value_l895_89551


namespace NUMINAMATH_GPT_g_value_at_2_over_9_l895_89570

theorem g_value_at_2_over_9 (g : ℝ → ℝ) 
  (hg0 : g 0 = 0)
  (hgmono : ∀ ⦃x y⦄, 0 ≤ x → x < y → y ≤ 1 → g x ≤ g y)
  (hg_symm : ∀ x, 0 ≤ x → x ≤ 1 → g (1 - x) = 1 - g x)
  (hg_frac : ∀ x, 0 ≤ x → x ≤ 1 → g (x / 4) = g x / 3) :
  g (2 / 9) = 8 / 27 :=
sorry

end NUMINAMATH_GPT_g_value_at_2_over_9_l895_89570


namespace NUMINAMATH_GPT_stream_speed_l895_89541

theorem stream_speed (v : ℝ) : 
  (∀ (speed_boat_in_still_water distance time : ℝ), 
    speed_boat_in_still_water = 25 ∧ distance = 90 ∧ time = 3 →
    distance = (speed_boat_in_still_water + v) * time) →
  v = 5 :=
by
  intro h
  have h1 := h 25 90 3 ⟨rfl, rfl, rfl⟩
  sorry

end NUMINAMATH_GPT_stream_speed_l895_89541


namespace NUMINAMATH_GPT_third_factorial_is_7_l895_89520

-- Define the factorial function
def factorial : ℕ → ℕ
| 0       => 1
| (n + 1) => (n + 1) * factorial n

-- Problem conditions
def b : ℕ := 9
def factorial_b_minus_2 : ℕ := factorial (b - 2)
def factorial_b_plus_1 : ℕ := factorial (b + 1)
def GCD_value : ℕ := Nat.gcd (Nat.gcd factorial_b_minus_2 factorial_b_plus_1) (factorial 7)

-- Theorem statement
theorem third_factorial_is_7 :
  Nat.gcd (Nat.gcd (factorial (b - 2)) (factorial (b + 1))) (factorial 7) = 5040 →
  ∃ k : ℕ, factorial k = 5040 ∧ k = 7 :=
by
  sorry

end NUMINAMATH_GPT_third_factorial_is_7_l895_89520


namespace NUMINAMATH_GPT_evaluate_expression_l895_89518

theorem evaluate_expression (x : ℤ) (z : ℤ) (hx : x = 4) (hz : z = -2) : z * (z - 4 * x) = 36 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l895_89518


namespace NUMINAMATH_GPT_original_number_l895_89569

/-- Proof that the original three-digit number abc equals 118 under the given conditions. -/
theorem original_number (N : ℕ) (hN : N = 4332) (a b c : ℕ)
  (h : 100 * a + 10 * b + c = 118) :
  100 * a + 10 * b + c = 118 :=
by
  sorry

end NUMINAMATH_GPT_original_number_l895_89569


namespace NUMINAMATH_GPT_digit_sum_eq_21_l895_89560

theorem digit_sum_eq_21 (A B C D: ℕ) (h1: A ≠ 0) 
    (h2: (A * 10 + B) * 100 + (C * 10 + D) = (C * 10 + D)^2 - (A * 10 + B)^2) 
    (hA: A < 10) (hB: B < 10) (hC: C < 10) (hD: D < 10) : 
    A + B + C + D = 21 :=
by 
  sorry

end NUMINAMATH_GPT_digit_sum_eq_21_l895_89560


namespace NUMINAMATH_GPT_scientific_notation_0_056_l895_89577

theorem scientific_notation_0_056 :
  (0.056 = 5.6 * 10^(-2)) :=
by
  sorry

end NUMINAMATH_GPT_scientific_notation_0_056_l895_89577


namespace NUMINAMATH_GPT_cost_price_represents_articles_l895_89562

theorem cost_price_represents_articles (C S : ℝ) (N : ℕ)
  (h1 : N * C = 16 * S)
  (h2 : S = C * 1.125) :
  N = 18 :=
by
  sorry

end NUMINAMATH_GPT_cost_price_represents_articles_l895_89562


namespace NUMINAMATH_GPT_fraction_to_terminating_decimal_l895_89573

theorem fraction_to_terminating_decimal :
  (47 / (2^3 * 5^4) : ℝ) = 0.0094 := by
  sorry

end NUMINAMATH_GPT_fraction_to_terminating_decimal_l895_89573


namespace NUMINAMATH_GPT_car_b_speed_l895_89555

/--
A car A going at 30 miles per hour set out on an 80-mile trip at 9:00 a.m.
Exactly 10 minutes later, a car B left from the same place and followed the same route.
Car B caught up with car A at 10:30 a.m.
Prove that the speed of car B is 33.75 miles per hour.
-/
theorem car_b_speed
    (v_a : ℝ) (t_start_a t_start_b t_end : ℝ) (v_b : ℝ)
    (h1 : v_a = 30) 
    (h2 : t_start_a = 9) 
    (h3 : t_start_b = 9 + (10 / 60)) 
    (h4 : t_end = 10.5) 
    (h5 : t_end - t_start_b = (4 / 3))
    (h6 : v_b * (t_end - t_start_b) = v_a * (t_end - t_start_a) + (v_a * (10 / 60))) :
  v_b = 33.75 := 
sorry

end NUMINAMATH_GPT_car_b_speed_l895_89555


namespace NUMINAMATH_GPT_find_smaller_number_l895_89552

theorem find_smaller_number (x y : ℤ) (h1 : x + y = 15) (h2 : 3 * x = 5 * y - 11) : x = 8 :=
by
  sorry

end NUMINAMATH_GPT_find_smaller_number_l895_89552


namespace NUMINAMATH_GPT_possible_values_of_expression_l895_89539

theorem possible_values_of_expression (a b c d : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0) :
  (∃ v : ℝ, v = (a / |a| + b / |b| + c / |c| + d / |d| + (a * b * c * d) / |a * b * c * d|) ∧ 
            (v = 5 ∨ v = 1 ∨ v = -3 ∨ v = -5)) :=
by
  sorry

end NUMINAMATH_GPT_possible_values_of_expression_l895_89539


namespace NUMINAMATH_GPT_equation_of_line_l895_89526

theorem equation_of_line (P : ℝ × ℝ) (m : ℝ) : 
  P = (3, 3) → m = 2 * 1 → ∃ b : ℝ, ∀ x : ℝ, P.2 = m * (x - P.1) + b ↔ y = 2 * x - 3 := 
by {
  sorry
}

end NUMINAMATH_GPT_equation_of_line_l895_89526


namespace NUMINAMATH_GPT_perimeter_of_ABFCDE_l895_89565

-- Define the problem parameters
def square_perimeter : ℤ := 60
def side_length (p : ℤ) : ℤ := p / 4
def equilateral_triangle_side (l : ℤ) : ℤ := l
def new_shape_sides : ℕ := 6
def new_perimeter (s : ℤ) : ℤ := new_shape_sides * s

-- Define the theorem to be proved
theorem perimeter_of_ABFCDE (p : ℤ) (s : ℕ) (len : ℤ) : len = side_length p → len = equilateral_triangle_side len →
  new_perimeter len = 90 :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_perimeter_of_ABFCDE_l895_89565


namespace NUMINAMATH_GPT_open_parking_spots_fourth_level_l895_89544

theorem open_parking_spots_fourth_level :
  ∀ (n_first n_total : ℕ)
    (n_second_diff n_third_diff : ℕ),
    n_first = 4 →
    n_second_diff = 7 →
    n_third_diff = 6 →
    n_total = 46 →
    ∃ (n_first n_second n_third n_fourth : ℕ),
      n_second = n_first + n_second_diff ∧
      n_third = n_second + n_third_diff ∧
      n_first + n_second + n_third + n_fourth = n_total ∧
      n_fourth = 14 := by
  sorry

end NUMINAMATH_GPT_open_parking_spots_fourth_level_l895_89544


namespace NUMINAMATH_GPT_pages_needed_l895_89514

theorem pages_needed (packs : ℕ) (cards_per_pack : ℕ) (cards_per_page : ℕ) (total_packs : packs = 60) (cards_in_pack : cards_per_pack = 7) (capacity_per_page : cards_per_page = 10) : (packs * cards_per_pack) / cards_per_page = 42 := 
by
  -- Utilize the conditions
  have H1 : packs = 60 := total_packs
  have H2 : cards_per_pack = 7 := cards_in_pack
  have H3 : cards_per_page = 10 := capacity_per_page
  -- Use these to simplify and prove the target expression 
  sorry

end NUMINAMATH_GPT_pages_needed_l895_89514


namespace NUMINAMATH_GPT_quantiville_jacket_junction_l895_89567

theorem quantiville_jacket_junction :
  let sales_tax_rate := 0.07
  let original_price := 120.0
  let discount := 0.25
  let amy_total := (original_price * (1 + sales_tax_rate)) * (1 - discount)
  let bob_total := (original_price * (1 - discount)) * (1 + sales_tax_rate)
  let carla_total := ((original_price * (1 + sales_tax_rate)) * (1 - discount)) * (1 + sales_tax_rate)
  (carla_total - amy_total) = 6.744 :=
by
  sorry

end NUMINAMATH_GPT_quantiville_jacket_junction_l895_89567


namespace NUMINAMATH_GPT_find_k_l895_89527

def condition (k : ℝ) : Prop := 24 / k = 4

theorem find_k (k : ℝ) (h : condition k) : k = 6 :=
sorry

end NUMINAMATH_GPT_find_k_l895_89527


namespace NUMINAMATH_GPT_contradiction_proof_l895_89561

theorem contradiction_proof (a b : ℕ) (h : a + b ≥ 3) : (a ≥ 2) ∨ (b ≥ 2) :=
sorry

end NUMINAMATH_GPT_contradiction_proof_l895_89561


namespace NUMINAMATH_GPT_students_per_group_l895_89593

def total_students : ℕ := 30
def number_of_groups : ℕ := 6

theorem students_per_group :
  total_students / number_of_groups = 5 :=
by
  sorry

end NUMINAMATH_GPT_students_per_group_l895_89593


namespace NUMINAMATH_GPT_rightmost_four_digits_of_7_pow_2045_l895_89519

theorem rightmost_four_digits_of_7_pow_2045 : (7^2045 % 10000) = 6807 :=
by
  sorry

end NUMINAMATH_GPT_rightmost_four_digits_of_7_pow_2045_l895_89519


namespace NUMINAMATH_GPT_value_of_expression_l895_89557

theorem value_of_expression (x y : ℝ) (h1 : x = 12) (h2 : y = 18) : 3 * (x - y) * (x + y) = -540 :=
by
  rw [h1, h2]
  sorry

end NUMINAMATH_GPT_value_of_expression_l895_89557


namespace NUMINAMATH_GPT_opposite_of_neg3_is_3_l895_89579

theorem opposite_of_neg3_is_3 : -(-3) = 3 := by
  sorry

end NUMINAMATH_GPT_opposite_of_neg3_is_3_l895_89579


namespace NUMINAMATH_GPT_present_worth_proof_l895_89549

-- Define the conditions
def banker's_gain (BG : ℝ) : Prop := BG = 16
def true_discount (TD : ℝ) : Prop := TD = 96

-- Define the relationship from the problem
def relationship (BG TD PW : ℝ) : Prop := BG = TD - PW

-- Define the present worth of the sum
def present_worth : ℝ := 80

-- Theorem stating that the present worth of the sum is Rs. 80 given the conditions
theorem present_worth_proof (BG TD PW : ℝ)
  (hBG : banker's_gain BG)
  (hTD : true_discount TD)
  (hRelation : relationship BG TD PW) :
  PW = present_worth := by
  sorry

end NUMINAMATH_GPT_present_worth_proof_l895_89549


namespace NUMINAMATH_GPT_expression_value_l895_89580

theorem expression_value : (5^2 - 5) * (6^2 - 6) - (7^2 - 7) = 558 := by
  sorry

end NUMINAMATH_GPT_expression_value_l895_89580


namespace NUMINAMATH_GPT_sum_cyc_geq_one_l895_89500

theorem sum_cyc_geq_one (a b c : ℝ) (hpos : a > 0 ∧ b > 0 ∧ c > 0) (hcond : a * b + b * c + c * a = a * b * c) :
  (a^4 / (b * (b^4 + c^3)) + b^4 / (c * (c^3 + a^4)) + c^4 / (a * (a^4 + b^3))) ≥ 1 :=
sorry

end NUMINAMATH_GPT_sum_cyc_geq_one_l895_89500


namespace NUMINAMATH_GPT_last_digit_1989_1989_last_digit_1989_1992_last_digit_1992_1989_last_digit_1992_1992_l895_89543

noncomputable def last_digit (n : ℕ) : ℕ := n % 10

theorem last_digit_1989_1989:
  last_digit (1989 ^ 1989) = 9 := 
sorry

theorem last_digit_1989_1992:
  last_digit (1989 ^ 1992) = 1 := 
sorry

theorem last_digit_1992_1989:
  last_digit (1992 ^ 1989) = 2 := 
sorry

theorem last_digit_1992_1992:
  last_digit (1992 ^ 1992) = 6 := 
sorry

end NUMINAMATH_GPT_last_digit_1989_1989_last_digit_1989_1992_last_digit_1992_1989_last_digit_1992_1992_l895_89543


namespace NUMINAMATH_GPT_cube_roots_not_arithmetic_progression_l895_89503

theorem cube_roots_not_arithmetic_progression
  (p q r : ℕ) 
  (hp : Prime p) 
  (hq : Prime q) 
  (hr : Prime r) 
  (h_distinct: p ≠ q ∧ q ≠ r ∧ p ≠ r) : 
  ¬ ∃ (d : ℝ) (m n : ℤ), (n ≠ m) ∧ (↑q)^(1/3 : ℝ) = (↑p)^(1/3 : ℝ) + (m : ℝ) * d ∧ (↑r)^(1/3 : ℝ) = (↑p)^(1/3 : ℝ) + (n : ℝ) * d :=
by sorry

end NUMINAMATH_GPT_cube_roots_not_arithmetic_progression_l895_89503


namespace NUMINAMATH_GPT_range_of_a_l895_89531

open Real

noncomputable def A (x : ℝ) : Prop := (x + 1) / (x - 2) ≥ 0
noncomputable def B (x a : ℝ) : Prop := x^2 - (2 * a + 1) * x + a^2 + a ≥ 0

theorem range_of_a :
  (∀ x, A x → B x a) ↔ -1 ≤ a ∧ a ≤ 1 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l895_89531


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_l895_89585

theorem arithmetic_sequence_sum (a : ℕ → ℕ) (S : ℕ → ℕ) (b : ℕ → ℚ) (T : ℕ → ℚ) 
  (h1 : a 3 = 7) (h2 : a 5 + a 7 = 26) :
  (∀ n, a n = 2 * n + 1) ∧
  (∀ n, S n = n^2 + 2 * n) ∧
  (∀ n, b n = 1 / ((2 * n + 1)^2 - 1)) ∧
  (∀ n, T n = n / (4 * (n + 1))) :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_l895_89585


namespace NUMINAMATH_GPT_initial_ratio_l895_89529

variable (A B : ℕ) (a b : ℕ)
variable (h1 : B = 6)
variable (h2 : (A + 2) / (B + 2) = 3 / 2)

theorem initial_ratio (A B : ℕ) (h1 : B = 6) (h2 : (A + 2) / (B + 2) = 3 / 2) : A / B = 5 / 3 := 
by 
    sorry

end NUMINAMATH_GPT_initial_ratio_l895_89529


namespace NUMINAMATH_GPT_range_of_a_l895_89587

theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, x^2 + (1 - a) * x + 1 < 0) → a < -1 ∨ a > 3 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l895_89587


namespace NUMINAMATH_GPT_smallest_prime_that_is_6_more_than_perfect_square_and_9_less_than_next_perfect_square_l895_89582

theorem smallest_prime_that_is_6_more_than_perfect_square_and_9_less_than_next_perfect_square :
  ∃ p : ℕ, Prime p ∧ (∃ k m : ℤ, k^2 = p - 6 ∧ m^2 = p + 9 ∧ m^2 - k^2 = 15) ∧ p = 127 :=
sorry

end NUMINAMATH_GPT_smallest_prime_that_is_6_more_than_perfect_square_and_9_less_than_next_perfect_square_l895_89582


namespace NUMINAMATH_GPT_range_of_t_l895_89599

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x

theorem range_of_t (t : ℝ) :  
  (∀ a > 0, ∀ x₀ y₀, 
    (a - a * Real.log x₀) / x₀^2 = 1 / 2 ∧ 
    y₀ = (a * Real.log x₀) / x₀ ∧ 
    x₀ = 2 * y₀ ∧ 
    a = Real.exp 1 ∧ 
    f (f x) = t -> t = 0) :=
by
  sorry

end NUMINAMATH_GPT_range_of_t_l895_89599


namespace NUMINAMATH_GPT_find_a_b_l895_89512

theorem find_a_b (a b : ℝ)
  (h1 : (0 - a)^2 + (-12 - b)^2 = 36)
  (h2 : (0 - a)^2 + (0 - b)^2 = 36) :
  a = 0 ∧ b = -6 :=
by
  sorry

end NUMINAMATH_GPT_find_a_b_l895_89512


namespace NUMINAMATH_GPT_lcm_of_lap_times_l895_89533

theorem lcm_of_lap_times :
  Nat.lcm (Nat.lcm 5 8) 10 = 40 := by
  sorry

end NUMINAMATH_GPT_lcm_of_lap_times_l895_89533


namespace NUMINAMATH_GPT_num_marked_cells_at_least_num_cells_in_one_square_l895_89522

-- Defining the total number of squares
def num_squares : ℕ := 2009

-- A square covers a cell if it is within its bounds.
-- A cell is marked if it is covered by an odd number of squares.
-- We have to show that the number of marked cells is at least the number of cells in one square.
theorem num_marked_cells_at_least_num_cells_in_one_square (side_length : ℕ) : 
  side_length * side_length ≤ (num_squares : ℕ) :=
sorry

end NUMINAMATH_GPT_num_marked_cells_at_least_num_cells_in_one_square_l895_89522


namespace NUMINAMATH_GPT_arithmetic_sequence_S10_l895_89586

def arithmetic_sequence (a d : ℤ) (n : ℕ) : ℤ :=
  a + n * d

def Sn (a d : ℤ) (n : ℕ) : ℤ :=
  n * a + (n * (n - 1)) / 2 * d

theorem arithmetic_sequence_S10 :
  ∃ (a d : ℤ), d ≠ 0 ∧ Sn a d 8 = 16 ∧
  (arithmetic_sequence a d 3)^2 = (arithmetic_sequence a d 2) * (arithmetic_sequence a d 6) ∧
  Sn a d 10 = 30 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_S10_l895_89586


namespace NUMINAMATH_GPT_exists_sequence_for_k_l895_89547

variable (n : ℕ) (k : ℕ)

noncomputable def exists_sequence (n k : ℕ) : Prop :=
  ∃ (x : ℕ → ℕ), ∀ i : ℕ, i < n → x i < x (i + 1)

theorem exists_sequence_for_k (n k : ℕ) (h1 : 1 ≤ k) (h2 : k ≤ n) :
  exists_sequence n k :=
  sorry

end NUMINAMATH_GPT_exists_sequence_for_k_l895_89547


namespace NUMINAMATH_GPT_least_number_of_square_tiles_l895_89566

theorem least_number_of_square_tiles
  (length_cm : ℕ) (width_cm : ℕ)
  (h1 : length_cm = 816) (h2 : width_cm = 432) :
  ∃ tile_count : ℕ, tile_count = 153 :=
by
  sorry

end NUMINAMATH_GPT_least_number_of_square_tiles_l895_89566


namespace NUMINAMATH_GPT_joan_total_seashells_l895_89535

-- Definitions of the conditions
def joan_initial_seashells : ℕ := 79
def mike_additional_seashells : ℕ := 63

-- Definition of the proof problem statement
theorem joan_total_seashells : joan_initial_seashells + mike_additional_seashells = 142 :=
by
  -- Proof would go here
  sorry

end NUMINAMATH_GPT_joan_total_seashells_l895_89535


namespace NUMINAMATH_GPT_no_tangent_line_l895_89513

-- Define the function f(x) = x^3 - 3ax
def f (a x : ℝ) : ℝ := x^3 - 3 * a * x

-- Define the derivative of f
def f' (a x : ℝ) : ℝ := 3 * x^2 - 3 * a

-- Proposition stating no b exists in ℝ such that y = -x + b is tangent to f
theorem no_tangent_line (a : ℝ) (H : ∀ b : ℝ, ¬ ∃ x : ℝ, f' a x = -1) : a < 1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_no_tangent_line_l895_89513


namespace NUMINAMATH_GPT_first_pack_weight_l895_89510

variable (hiking_rate : ℝ) (hours_per_day : ℝ) (days : ℝ)
variable (pounds_per_mile : ℝ) (first_resupply_percentage : ℝ) (second_resupply_percentage : ℝ)

theorem first_pack_weight (hiking_rate : ℝ) (hours_per_day : ℝ) (days : ℝ)
    (pounds_per_mile : ℝ) (first_resupply_percentage : ℝ) (second_resupply_percentage : ℝ) :
    hiking_rate = 2.5 →
    hours_per_day = 9 →
    days = 7 →
    pounds_per_mile = 0.6 →
    first_resupply_percentage = 0.30 →
    second_resupply_percentage = 0.20 →
    ∃ first_pack : ℝ, first_pack = 47.25 :=
by
  intro h1 h2 h3 h4 h5 h6
  let total_distance := hiking_rate * hours_per_day * days
  let total_supplies := pounds_per_mile * total_distance
  let first_resupply := total_supplies * first_resupply_percentage
  let second_resupply := total_supplies * second_resupply_percentage
  let first_pack := total_supplies - (first_resupply + second_resupply)
  use first_pack
  sorry

end NUMINAMATH_GPT_first_pack_weight_l895_89510


namespace NUMINAMATH_GPT_candy_comparison_l895_89515

variable (skittles_bryan : ℕ)
variable (gummy_bears_bryan : ℕ)
variable (chocolate_bars_bryan : ℕ)
variable (mms_ben : ℕ)
variable (jelly_beans_ben : ℕ)
variable (lollipops_ben : ℕ)

def bryan_total_candies := skittles_bryan + gummy_bears_bryan + chocolate_bars_bryan
def ben_total_candies := mms_ben + jelly_beans_ben + lollipops_ben

def difference_skittles_mms := skittles_bryan - mms_ben
def difference_gummy_jelly := jelly_beans_ben - gummy_bears_bryan
def difference_choco_lollipops := chocolate_bars_bryan - lollipops_ben

def sum_of_differences := difference_skittles_mms + difference_gummy_jelly + difference_choco_lollipops

theorem candy_comparison
  (h_bryan_skittles : skittles_bryan = 50)
  (h_bryan_gummy_bears : gummy_bears_bryan = 25)
  (h_bryan_choco_bars : chocolate_bars_bryan = 15)
  (h_ben_mms : mms_ben = 20)
  (h_ben_jelly_beans : jelly_beans_ben = 30)
  (h_ben_lollipops : lollipops_ben = 10) :
  bryan_total_candies = 90 ∧
  ben_total_candies = 60 ∧
  bryan_total_candies > ben_total_candies ∧
  difference_skittles_mms = 30 ∧
  difference_gummy_jelly = 5 ∧
  difference_choco_lollipops = 5 ∧
  sum_of_differences = 40 := by
  sorry

end NUMINAMATH_GPT_candy_comparison_l895_89515


namespace NUMINAMATH_GPT_no_solutions_l895_89546

theorem no_solutions
  (x y z : ℤ)
  (h : x^2 + y^2 = 4 * z - 1) : False :=
sorry

end NUMINAMATH_GPT_no_solutions_l895_89546


namespace NUMINAMATH_GPT_find_x_plus_y_l895_89517

theorem find_x_plus_y (x y : ℝ) (hx : abs x - x + y = 6) (hy : x + abs y + y = 16) : x + y = 10 :=
sorry

end NUMINAMATH_GPT_find_x_plus_y_l895_89517


namespace NUMINAMATH_GPT_time_period_for_investment_l895_89506

variable (P R₁₅ R₁₀ I₁₅ I₁₀ : ℝ)
variable (T : ℝ)

noncomputable def principal := 8400
noncomputable def rate15 := 15
noncomputable def rate10 := 10
noncomputable def interestDifference := 840

theorem time_period_for_investment :
  ∀ (T : ℝ),
    P = principal →
    R₁₅ = rate15 →
    R₁₀ = rate10 →
    I₁₅ = P * (R₁₅ / 100) * T →
    I₁₀ = P * (R₁₀ / 100) * T →
    (I₁₅ - I₁₀) = interestDifference →
    T = 2 :=
  sorry

end NUMINAMATH_GPT_time_period_for_investment_l895_89506


namespace NUMINAMATH_GPT_total_employees_l895_89507

theorem total_employees (female_employees managers male_associates female_managers : ℕ)
  (h_female_employees : female_employees = 90)
  (h_managers : managers = 40)
  (h_male_associates : male_associates = 160)
  (h_female_managers : female_managers = 40) :
  female_employees - female_managers + male_associates + managers = 250 :=
by {
  sorry
}

end NUMINAMATH_GPT_total_employees_l895_89507


namespace NUMINAMATH_GPT_max_number_of_squares_with_twelve_points_l895_89559

-- Define the condition: twelve marked points in a grid
def twelve_points_marked_on_grid : Prop := 
  -- Assuming twelve specific points represented in a grid-like structure
  -- (This will be defined concretely in the proof implementation context)
  sorry

-- Define the problem statement to be proved
theorem max_number_of_squares_with_twelve_points : 
  twelve_points_marked_on_grid → (∃ n, n = 11) :=
by 
  sorry

end NUMINAMATH_GPT_max_number_of_squares_with_twelve_points_l895_89559


namespace NUMINAMATH_GPT_eds_weight_l895_89596

variable (Al Ben Carl Ed : ℕ)

def weight_conditions : Prop :=
  Carl = 175 ∧ Ben = Carl - 16 ∧ Al = Ben + 25 ∧ Ed = Al - 38

theorem eds_weight (h : weight_conditions Al Ben Carl Ed) : Ed = 146 :=
by
  -- Conditions
  have h1 : Carl = 175    := h.1
  have h2 : Ben = Carl - 16 := h.2.1
  have h3 : Al = Ben + 25   := h.2.2.1
  have h4 : Ed = Al - 38    := h.2.2.2
  -- Proof itself is omitted, sorry placeholder
  sorry

end NUMINAMATH_GPT_eds_weight_l895_89596


namespace NUMINAMATH_GPT_initial_amount_A_correct_l895_89511

noncomputable def initial_amount_A :=
  let a := 21
  let b := 5
  let c := 9

  -- After A gives B and C
  let b_after_A := b + 5
  let c_after_A := c + 9
  let a_after_A := a - (5 + 9)

  -- After B gives A and C
  let a_after_B := a_after_A + (a_after_A / 2)
  let c_after_B := c_after_A + (c_after_A / 2)
  let b_after_B := b_after_A - (a_after_A / 2 + c_after_A / 2)

  -- After C gives A and B
  let a_final := a_after_B + 3 * a_after_B
  let b_final := b_after_B + 3 * b_after_B
  let c_final := c_after_B - (3 * a_final + b_final)

  (a_final = 24) ∧ (b_final = 16) ∧ (c_final = 8)

theorem initial_amount_A_correct : initial_amount_A := 
by
  -- Skipping proof details
  sorry

end NUMINAMATH_GPT_initial_amount_A_correct_l895_89511
