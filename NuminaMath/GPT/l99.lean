import Mathlib

namespace NUMINAMATH_GPT_Annabelle_saved_12_dollars_l99_9952

def weekly_allowance : ℕ := 30
def spent_on_junk_food : ℕ := weekly_allowance / 3
def spent_on_sweets : ℕ := 8
def total_spent : ℕ := spent_on_junk_food + spent_on_sweets
def saved_amount : ℕ := weekly_allowance - total_spent

theorem Annabelle_saved_12_dollars : saved_amount = 12 := by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_Annabelle_saved_12_dollars_l99_9952


namespace NUMINAMATH_GPT_acute_angle_l99_9913

variables (x : ℝ)

def a : ℝ × ℝ := (2, x)
def b : ℝ × ℝ := (1, 3)

def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

theorem acute_angle (x : ℝ) : 
  (-2 / 3 < x) → x ≠ -2 / 3 → dot_product (2, x) (1, 3) > 0 :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_acute_angle_l99_9913


namespace NUMINAMATH_GPT_find_a_l99_9945

variables {a b c : ℤ}

theorem find_a (h1 : a + b = c) (h2 : b + c = 7) (h3 : c = 5) : a = 3 :=
by
  sorry

end NUMINAMATH_GPT_find_a_l99_9945


namespace NUMINAMATH_GPT_largest_c_l99_9929

theorem largest_c (c : ℝ) : (∃ x : ℝ, x^2 + 4 * x + c = -3) → c ≤ 1 :=
by
  sorry

end NUMINAMATH_GPT_largest_c_l99_9929


namespace NUMINAMATH_GPT_count_integers_congruent_to_7_mod_13_l99_9983

theorem count_integers_congruent_to_7_mod_13 : 
  (∃ (n : ℕ), ∀ x, (1 ≤ x ∧ x < 500 ∧ x % 13 = 7) → x = 7 + 13 * n ∧ n < 38) :=
sorry

end NUMINAMATH_GPT_count_integers_congruent_to_7_mod_13_l99_9983


namespace NUMINAMATH_GPT_tank_filling_time_with_leaks_l99_9966

theorem tank_filling_time_with_leaks (pump_time : ℝ) (leak1_time : ℝ) (leak2_time : ℝ) (leak3_time : ℝ) (fill_time : ℝ)
  (h1 : pump_time = 2)
  (h2 : fill_time = 3)
  (h3 : leak1_time = 6)
  (h4 : leak2_time = 8)
  (h5 : leak3_time = 12) :
  fill_time = 8 := 
sorry

end NUMINAMATH_GPT_tank_filling_time_with_leaks_l99_9966


namespace NUMINAMATH_GPT_number_of_graphing_calculators_in_class_l99_9984

-- Define a structure for the problem
structure ClassData where
  num_boys : ℕ
  num_girls : ℕ
  num_scientific_calculators : ℕ
  num_girls_with_calculators : ℕ
  num_graphing_calculators : ℕ
  no_overlap : Prop

-- Instantiate the problem using given conditions
def mrs_anderson_class : ClassData :=
{
  num_boys := 20,
  num_girls := 18,
  num_scientific_calculators := 30,
  num_girls_with_calculators := 15,
  num_graphing_calculators := 10,
  no_overlap := true
}

-- Lean statement for the proof problem
theorem number_of_graphing_calculators_in_class (data : ClassData) :
  data.num_graphing_calculators = 10 :=
by
  sorry

end NUMINAMATH_GPT_number_of_graphing_calculators_in_class_l99_9984


namespace NUMINAMATH_GPT_round_trip_time_l99_9944

theorem round_trip_time (current_speed : ℝ) (boat_speed_still : ℝ) (distance_upstream : ℝ) (total_time : ℝ) :
  current_speed = 4 → 
  boat_speed_still = 18 → 
  distance_upstream = 85.56 →
  total_time = 10 :=
by
  intros h_current h_boat h_distance
  sorry

end NUMINAMATH_GPT_round_trip_time_l99_9944


namespace NUMINAMATH_GPT_jeremy_age_l99_9956

theorem jeremy_age (A J C : ℕ) (h1 : A + J + C = 132) (h2 : A = J / 3) (h3 : C = 2 * A) : J = 66 :=
by
  sorry

end NUMINAMATH_GPT_jeremy_age_l99_9956


namespace NUMINAMATH_GPT_diane_age_proof_l99_9935

noncomputable def diane_age (A Al D : ℕ) : Prop :=
  ((A + (30 - D) = 60) ∧ (Al + (30 - D) = 15) ∧ (A + Al = 47)) → (D = 16)

theorem diane_age_proof : ∃ (D : ℕ), ∃ (A Al : ℕ), diane_age A Al D :=
by {
  sorry
}

end NUMINAMATH_GPT_diane_age_proof_l99_9935


namespace NUMINAMATH_GPT_trajectory_of_P_below_x_axis_l99_9934

theorem trajectory_of_P_below_x_axis (x y : ℝ) (P_below_x_axis : y < 0)
    (tangent_to_parabola : ∃ A B: ℝ × ℝ, A.1^2 = 2 * A.2 ∧ B.1^2 = 2 * B.2 ∧ (x^2 + y^2 = 1))
    (AB_tangent_to_circle : ∀ (x0 y0 : ℝ), x0^2 + y0^2 = 1 → x0 * x + y0 * y = 1) :
    y^2 - x^2 = 1 :=
sorry

end NUMINAMATH_GPT_trajectory_of_P_below_x_axis_l99_9934


namespace NUMINAMATH_GPT_division_of_negatives_l99_9985

theorem division_of_negatives : (-500 : ℤ) / (-50 : ℤ) = 10 := by
  sorry

end NUMINAMATH_GPT_division_of_negatives_l99_9985


namespace NUMINAMATH_GPT_find_a_of_cool_frog_meeting_l99_9940

-- Question and conditions
def frogs : ℕ := 16
def friend_probability : ℚ := 1 / 2
def cool_condition (f: ℕ → ℕ) : Prop := ∀ i, f i % 4 = 0

-- Example theorem where we need to find 'a'
theorem find_a_of_cool_frog_meeting :
  let a := 1167
  let b := 2 ^ 41
  ∀ (f: ℕ → ℕ), ∀ (p: ℚ) (h: p = friend_probability),
    (cool_condition f) →
    (∃ a b, a / b = p ∧ a % gcd a b = 0 ∧ gcd a b = 1) ∧ a = 1167 :=
by
  sorry

end NUMINAMATH_GPT_find_a_of_cool_frog_meeting_l99_9940


namespace NUMINAMATH_GPT_find_number_l99_9943

theorem find_number (x : ℤ) (h : (((55 + x) / 7 + 40) * 5 = 555)) : x = 442 :=
sorry

end NUMINAMATH_GPT_find_number_l99_9943


namespace NUMINAMATH_GPT_percent_of_volume_filled_by_cubes_l99_9933

theorem percent_of_volume_filled_by_cubes :
  let box_width := 8
  let box_height := 6
  let box_length := 12
  let cube_size := 2
  let box_volume := box_width * box_height * box_length
  let cube_volume := cube_size ^ 3
  let num_cubes := (box_width / cube_size) * (box_height / cube_size) * (box_length / cube_size)
  let cubes_volume := num_cubes * cube_volume
  (cubes_volume / box_volume : ℝ) * 100 = 100 := by
  sorry

end NUMINAMATH_GPT_percent_of_volume_filled_by_cubes_l99_9933


namespace NUMINAMATH_GPT_union_A_B_eq_neg2_neg1_0_l99_9930

def setA : Set ℤ := {x : ℤ | (x + 2) * (x - 1) < 0}
def setB : Set ℤ := {-2, -1}

theorem union_A_B_eq_neg2_neg1_0 : (setA ∪ setB) = ({-2, -1, 0} : Set ℤ) :=
by
  sorry

end NUMINAMATH_GPT_union_A_B_eq_neg2_neg1_0_l99_9930


namespace NUMINAMATH_GPT_power_function_passes_through_fixed_point_l99_9978

theorem power_function_passes_through_fixed_point 
  (a : ℝ) (ha : 0 < a ∧ a ≠ 1)
  (P : ℝ × ℝ) (hP : P = (4, 2))
  (f : ℝ → ℝ) (hf : f x = x ^ a) : ∀ x, f x = x ^ (1 / 2) :=
by
  sorry

end NUMINAMATH_GPT_power_function_passes_through_fixed_point_l99_9978


namespace NUMINAMATH_GPT_compute_F_2_f_3_l99_9976

def f (a : ℝ) : ℝ := a^2 - 3 * a + 2
def F (a b : ℝ) : ℝ := b + a^3

theorem compute_F_2_f_3 : F 2 (f 3) = 10 :=
by
  sorry

end NUMINAMATH_GPT_compute_F_2_f_3_l99_9976


namespace NUMINAMATH_GPT_mean_of_set_eq_10point6_l99_9960

open Real -- For real number operations

theorem mean_of_set_eq_10point6 (n : ℝ)
  (h : n + 7 = 11) :
  (4 + 7 + 11 + 13 + 18) / 5 = 10.6 :=
by
  have h1 : n = 4 := by linarith
  sorry -- skip the proof part

end NUMINAMATH_GPT_mean_of_set_eq_10point6_l99_9960


namespace NUMINAMATH_GPT_intersection_chord_line_eq_l99_9916

noncomputable def circle1 (x y : ℝ) : Prop := (x - 2)^2 + (y - 1)^2 = 10
noncomputable def circle2 (x y : ℝ) : Prop := (x + 6)^2 + (y + 3)^2 = 50

theorem intersection_chord_line_eq (x y : ℝ) (h1 : circle1 x y) (h2 : circle2 x y) : 
  2 * x + y = 0 :=
sorry

end NUMINAMATH_GPT_intersection_chord_line_eq_l99_9916


namespace NUMINAMATH_GPT_totalBooksOnShelves_l99_9991

-- Define the conditions
def numShelves : Nat := 150
def booksPerShelf : Nat := 15

-- Define the statement to be proved
theorem totalBooksOnShelves : numShelves * booksPerShelf = 2250 :=
by
  -- Skipping the proof
  sorry

end NUMINAMATH_GPT_totalBooksOnShelves_l99_9991


namespace NUMINAMATH_GPT_integer_solutions_l99_9919

-- Define the problem statement in Lean
theorem integer_solutions :
  {p : ℤ × ℤ | ∃ x y : ℤ, p = (x, y) ∧ x^2 + x = y^4 + y^3 + y^2 + y} =
  {(-1, -1), (0, -1), (-1, 0), (0, 0), (5, 2), (-6, 2)} :=
by
  sorry

end NUMINAMATH_GPT_integer_solutions_l99_9919


namespace NUMINAMATH_GPT_x_intercept_is_7_0_l99_9905

-- Define the given line equation
def line_eq (x y : ℚ) : Prop := 4 * x + 7 * y = 28

-- State the theorem we want to prove
theorem x_intercept_is_7_0 :
  ∃ x : ℚ, ∃ y : ℚ, line_eq x y ∧ y = 0 ∧ x = 7 :=
by
  sorry

end NUMINAMATH_GPT_x_intercept_is_7_0_l99_9905


namespace NUMINAMATH_GPT_cost_proof_l99_9902

-- Given conditions
def total_cost : Int := 190
def working_days : Int := 19
def trips_per_day : Int := 2
def total_trips : Int := working_days * trips_per_day

-- Define the problem to prove
def cost_per_trip : Int := 5

theorem cost_proof : (total_cost / total_trips = cost_per_trip) := 
by 
  -- This is a placeholder to indicate that we're skipping the proof
  sorry

end NUMINAMATH_GPT_cost_proof_l99_9902


namespace NUMINAMATH_GPT_charity_distribution_l99_9928

theorem charity_distribution 
  (X : ℝ) (Y : ℝ) (Z : ℝ) (W : ℝ) (A : ℝ)
  (h1 : X > 0) (h2 : Y > 0) (h3 : Y < 100) (h4 : Z > 0) (h5 : W > 0) (h6 : A > 0)
  (h7 : W * A = X * (100 - Y) / 100) :
  (Y * X) / (100 * Z) = A * W * Y / (100 * Z) :=
by 
  sorry

end NUMINAMATH_GPT_charity_distribution_l99_9928


namespace NUMINAMATH_GPT_reach_any_composite_from_4_l99_9962

/-- 
Prove that starting from the number \( 4 \), it is possible to reach any given composite number 
through repeatedly adding one of its divisors, different from itself and one. 
-/
theorem reach_any_composite_from_4:
  ∀ n : ℕ, Prime (n) → n ≥ 4 → (∃ k d : ℕ, d ∣ k ∧ k = k + d ∧ k = n) := 
by 
  sorry


end NUMINAMATH_GPT_reach_any_composite_from_4_l99_9962


namespace NUMINAMATH_GPT_number_of_valid_numbers_l99_9911

def is_valid_number (N : ℕ) : Prop :=
  N ≥ 1000 ∧ N < 10000 ∧ ∃ a x : ℕ, 1 ≤ a ∧ a ≤ 9 ∧ x < 1000 ∧ 
  N = 1000 * a + x ∧ x = N / 9

theorem number_of_valid_numbers : ∃ (n : ℕ), n = 7 ∧ ∀ N, is_valid_number N → N < 1000 * (n + 2) := 
sorry

end NUMINAMATH_GPT_number_of_valid_numbers_l99_9911


namespace NUMINAMATH_GPT_cannot_determine_E1_l99_9918

variable (a b c d : ℝ)

theorem cannot_determine_E1 (h1 : a + b - c - d = 5) (h2 : (b - d)^2 = 16) : 
  ¬ ∃ e : ℝ, e = a - b - c + d :=
by
  sorry

end NUMINAMATH_GPT_cannot_determine_E1_l99_9918


namespace NUMINAMATH_GPT_sequence_max_length_l99_9951

theorem sequence_max_length (x : ℕ) :
  (2000 - 2 * x > 0) ∧ (3 * x - 2000 > 0) ∧ (4000 - 5 * x > 0) ∧ 
  (8 * x - 6000 > 0) ∧ (10000 - 13 * x > 0) ∧ (21 * x - 16000 > 0) → x = 762 :=
by
  sorry

end NUMINAMATH_GPT_sequence_max_length_l99_9951


namespace NUMINAMATH_GPT_largest_share_of_partner_l99_9948

theorem largest_share_of_partner 
    (ratios : List ℕ := [2, 3, 4, 4, 6])
    (total_profit : ℕ := 38000) :
    let total_parts := ratios.sum
    let part_value := total_profit / total_parts
    let largest_share := List.maximum ratios * part_value
    largest_share = 12000 :=
by
    let total_parts := ratios.sum
    let part_value := total_profit / total_parts
    let largest_share := List.maximum ratios * part_value
    have h1 : total_parts = 19 := by
        sorry
    have h2 : part_value = 2000 := by
        sorry
    have h3 : List.maximum ratios = 6 := by
        sorry
    have h4 : largest_share = 12000 := by
        sorry
    exact h4


end NUMINAMATH_GPT_largest_share_of_partner_l99_9948


namespace NUMINAMATH_GPT_find_x_of_total_area_l99_9914

theorem find_x_of_total_area 
  (x : Real)
  (h_triangle : (1/2) * (4 * x) * (3 * x) = 6 * x^2)
  (h_square1 : (3 * x)^2 = 9 * x^2)
  (h_square2 : (6 * x)^2 = 36 * x^2)
  (h_total : 6 * x^2 + 9 * x^2 + 36 * x^2 = 700) :
  x = Real.sqrt (700 / 51) :=
by {
  sorry
}

end NUMINAMATH_GPT_find_x_of_total_area_l99_9914


namespace NUMINAMATH_GPT_distance_between_A_and_B_l99_9915

theorem distance_between_A_and_B 
  (v1 v2: ℝ) (s: ℝ)
  (h1 : (s - 8) / v1 = s / v2)
  (h2 : s / (2 * v1) = (s - 15) / v2)
  (h3: s = 40) : 
  s = 40 := 
sorry

end NUMINAMATH_GPT_distance_between_A_and_B_l99_9915


namespace NUMINAMATH_GPT_angle_sum_155_l99_9989

theorem angle_sum_155
  (AB AC DE DF : ℝ)
  (h1 : AB = AC)
  (h2 : DE = DF)
  (angle_BAC angle_EDF : ℝ)
  (h3 : angle_BAC = 20)
  (h4 : angle_EDF = 30) :
  ∃ (angle_DAC angle_ADE : ℝ), angle_DAC + angle_ADE = 155 :=
by
  sorry

end NUMINAMATH_GPT_angle_sum_155_l99_9989


namespace NUMINAMATH_GPT_total_songs_time_l99_9982

-- Definitions of durations for each radio show
def duration_show1 : ℕ := 180
def duration_show2 : ℕ := 240
def duration_show3 : ℕ := 120

-- Definitions of talking segments for each show
def talking_segments_show1 : ℕ := 3 * 10  -- 3 segments, 10 minutes each
def talking_segments_show2 : ℕ := 4 * 15  -- 4 segments, 15 minutes each
def talking_segments_show3 : ℕ := 2 * 8   -- 2 segments, 8 minutes each

-- Definitions of ad breaks for each show
def ad_breaks_show1 : ℕ := 5 * 5  -- 5 breaks, 5 minutes each
def ad_breaks_show2 : ℕ := 6 * 4  -- 6 breaks, 4 minutes each
def ad_breaks_show3 : ℕ := 3 * 6  -- 3 breaks, 6 minutes each

-- Function to calculate time spent on songs for a given show
def time_spent_on_songs (duration talking ad_breaks : ℕ) : ℕ :=
  duration - talking - ad_breaks

-- Total time spent on songs for all three shows
def total_time_spent_on_songs : ℕ :=
  time_spent_on_songs duration_show1 talking_segments_show1 ad_breaks_show1 +
  time_spent_on_songs duration_show2 talking_segments_show2 ad_breaks_show2 +
  time_spent_on_songs duration_show3 talking_segments_show3 ad_breaks_show3

-- The theorem we want to prove
theorem total_songs_time : total_time_spent_on_songs = 367 := 
  sorry

end NUMINAMATH_GPT_total_songs_time_l99_9982


namespace NUMINAMATH_GPT_total_legs_of_collection_l99_9924

theorem total_legs_of_collection (spiders ants : ℕ) (legs_per_spider legs_per_ant : ℕ)
  (h_spiders : spiders = 8) (h_ants : ants = 12)
  (h_legs_per_spider : legs_per_spider = 8) (h_legs_per_ant : legs_per_ant = 6) :
  (spiders * legs_per_spider + ants * legs_per_ant) = 136 :=
by
  sorry

end NUMINAMATH_GPT_total_legs_of_collection_l99_9924


namespace NUMINAMATH_GPT_rectangular_floor_length_l99_9909

theorem rectangular_floor_length
    (cost_per_square : ℝ)
    (total_cost : ℝ)
    (carpet_length : ℝ)
    (carpet_width : ℝ)
    (floor_width : ℝ)
    (floor_area : ℝ) 
    (H1 : cost_per_square = 15)
    (H2 : total_cost = 225)
    (H3 : carpet_length = 2)
    (H4 : carpet_width = 2)
    (H5 : floor_width = 6)
    (H6 : floor_area = floor_width * carpet_length * carpet_width * 15): 
    floor_area / floor_width = 10 :=
by
  sorry

end NUMINAMATH_GPT_rectangular_floor_length_l99_9909


namespace NUMINAMATH_GPT_car_speed_in_mph_l99_9959

-- Defining the given conditions
def fuel_efficiency : ℚ := 56 -- kilometers per liter
def gallons_to_liters : ℚ := 3.8 -- liters per gallon
def kilometers_to_miles : ℚ := 1 / 1.6 -- miles per kilometer
def fuel_decrease_gallons : ℚ := 3.9 -- gallons
def time_hours : ℚ := 5.7 -- hours

-- Using definitions to compute the speed
theorem car_speed_in_mph :
  (fuel_decrease_gallons * gallons_to_liters * fuel_efficiency * kilometers_to_miles) / time_hours = 91 :=
sorry

end NUMINAMATH_GPT_car_speed_in_mph_l99_9959


namespace NUMINAMATH_GPT_calculate_chord_length_l99_9949

noncomputable def chord_length_of_tangent (r1 r2 : ℝ) (c : ℝ) : Prop :=
  r1^2 - r2^2 = 18 ∧ (c / 2)^2 = 18

theorem calculate_chord_length (r1 r2 : ℝ) (h : chord_length_of_tangent r1 r2 (6 * Real.sqrt 2)) :
  (6 * Real.sqrt 2) = 6 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_calculate_chord_length_l99_9949


namespace NUMINAMATH_GPT_time_reduced_fraction_l99_9942

theorem time_reduced_fraction
  (T : ℝ)
  (V : ℝ)
  (hV : V = 42)
  (D : ℝ)
  (hD_1 : D = V * T)
  (V' : ℝ)
  (hV' : V' = V + 21)
  (T' : ℝ)
  (hD_2 : D = V' * T') :
  (T - T') / T = 1 / 3 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_time_reduced_fraction_l99_9942


namespace NUMINAMATH_GPT_savings_on_cheapest_flight_l99_9958

theorem savings_on_cheapest_flight :
  let delta_price := 850
  let delta_discount := 0.20
  let united_price := 1100
  let united_discount := 0.30
  let delta_final_price := delta_price - delta_price * delta_discount
  let united_final_price := united_price - united_price * united_discount
  delta_final_price < united_final_price →
  united_final_price - delta_final_price = 90 :=
by
  sorry

end NUMINAMATH_GPT_savings_on_cheapest_flight_l99_9958


namespace NUMINAMATH_GPT_math_problem_l99_9950

theorem math_problem (a b c : ℝ) (h1 : a + b + c = 12) (h2 : ab + ac + bc = 30) :
  a^3 + b^3 + c^3 - 3 * a * b * c + 2 * (a + b + c) = 672 :=
by
  sorry

end NUMINAMATH_GPT_math_problem_l99_9950


namespace NUMINAMATH_GPT_no_solutions_a_l99_9907

theorem no_solutions_a (x y : ℤ) : x^2 + y^2 ≠ 2003 := 
sorry

end NUMINAMATH_GPT_no_solutions_a_l99_9907


namespace NUMINAMATH_GPT_anthony_path_shortest_l99_9917

noncomputable def shortest_distance (A B C D M : ℝ) : ℝ :=
  4 + 2 * Real.sqrt 3

theorem anthony_path_shortest {A B C D : ℝ} (M : ℝ) (side_length : ℝ) (h : side_length = 4) : 
  shortest_distance A B C D M = 4 + 2 * Real.sqrt 3 :=
by 
  sorry

end NUMINAMATH_GPT_anthony_path_shortest_l99_9917


namespace NUMINAMATH_GPT_rectangle_area_ratio_l99_9995

-- Define points in complex plane or as tuples (for 2D geometry)
structure Point where
  x : ℝ
  y : ℝ

-- Rectangle vertices
def A : Point := {x := 0, y := 0}
def B : Point := {x := 1, y := 0}
def C : Point := {x := 1, y := 2}
def D : Point := {x := 0, y := 2}

-- Centroid of triangle BCD
def E : Point := {x := 1.0, y := 1.333}

-- Point F such that DF = 1/4 * DA
def F : Point := {x := 1.5, y := 0}

-- Calculate areas of triangles and quadrilateral
noncomputable def area_triangle (P Q R : Point) : ℝ :=
  0.5 * abs ((Q.x - P.x) * (R.y - P.y) - (Q.y - P.y) * (R.x - P.x))

noncomputable def area_rectangle : ℝ :=
  2.0  -- Area of rectangle ABCD (1 * 2)

noncomputable def problem_statement : Prop :=
  let area_DFE := area_triangle D F E
  let area_ABEF := area_rectangle - area_triangle A B F - area_triangle D A F
  area_DFE / area_ABEF = 1 / 10.5

theorem rectangle_area_ratio :
  problem_statement :=
by
  sorry

end NUMINAMATH_GPT_rectangle_area_ratio_l99_9995


namespace NUMINAMATH_GPT_tg_plus_ctg_l99_9947

theorem tg_plus_ctg (x : ℝ) (h : 1 / Real.cos x - 1 / Real.sin x = Real.sqrt 15) :
  Real.tan x + (1 / Real.tan x) = -3 ∨ Real.tan x + (1 / Real.tan x) = 5 :=
sorry

end NUMINAMATH_GPT_tg_plus_ctg_l99_9947


namespace NUMINAMATH_GPT_equal_heights_of_cylinder_and_cone_l99_9904

theorem equal_heights_of_cylinder_and_cone
  (r h : ℝ)
  (hc : h > 0)
  (hr : r > 0)
  (V_cylinder V_cone : ℝ)
  (V_cylinder_eq : V_cylinder = π * r ^ 2 * h)
  (V_cone_eq : V_cone = 1/3 * π * r ^ 2 * h)
  (volume_ratio : V_cylinder / V_cone = 3) :
h = h := -- Since we are given that the heights are initially the same
sorry

end NUMINAMATH_GPT_equal_heights_of_cylinder_and_cone_l99_9904


namespace NUMINAMATH_GPT_probability_of_exactly_9_correct_matches_is_zero_l99_9961

theorem probability_of_exactly_9_correct_matches_is_zero :
  ∀ (n : ℕ) (translate : Fin n → Fin n),
    (n = 10) → 
    (∀ i : Fin n, translate i ≠ i) → 
    (∃ (k : ℕ), (k < n ∧ k ≠ n-1) → false ) :=
by
  sorry

end NUMINAMATH_GPT_probability_of_exactly_9_correct_matches_is_zero_l99_9961


namespace NUMINAMATH_GPT_boxes_given_to_brother_l99_9954

-- Definitions
def total_boxes : ℝ := 14.0
def pieces_per_box : ℝ := 6.0
def pieces_remaining : ℝ := 42.0

-- Theorem stating the problem
theorem boxes_given_to_brother : 
  (total_boxes * pieces_per_box - pieces_remaining) / pieces_per_box = 7.0 := 
by
  sorry

end NUMINAMATH_GPT_boxes_given_to_brother_l99_9954


namespace NUMINAMATH_GPT_problem_solution_l99_9965

-- Define the given circle equation C
def circle_C_eq (x y : ℝ) : Prop := x^2 + y^2 - 4 * x + 3 = 0

-- Define the line of symmetry
def line_symmetry_eq (x y : ℝ) : Prop := y = -x - 4

-- Define the symmetric circle equation
def sym_circle_eq (x y : ℝ) : Prop := (x + 4)^2 + (y + 6)^2 = 1

theorem problem_solution (x y : ℝ)
  (H1 : circle_C_eq x y)
  (H2 : line_symmetry_eq x y) :
  sym_circle_eq x y :=
sorry

end NUMINAMATH_GPT_problem_solution_l99_9965


namespace NUMINAMATH_GPT_symmetric_point_proof_l99_9964

def symmetric_point (P : ℝ × ℝ) (line : ℝ → ℝ) : ℝ × ℝ := sorry

theorem symmetric_point_proof :
  symmetric_point (2, 5) (λ x => 1 - x) = (-4, -1) := sorry

end NUMINAMATH_GPT_symmetric_point_proof_l99_9964


namespace NUMINAMATH_GPT_total_weight_is_1kg_total_weight_in_kg_eq_1_l99_9900

theorem total_weight_is_1kg 
  (weight_msg : ℕ := 80)
  (weight_salt : ℕ := 500)
  (weight_detergent : ℕ := 420) :
  (weight_msg + weight_salt + weight_detergent) = 1000 := by
sorry

theorem total_weight_in_kg_eq_1 
  (total_weight_g : ℕ := weight_msg + weight_salt + weight_detergent) :
  (total_weight_g = 1000) → (total_weight_g / 1000 = 1) := by
sorry

end NUMINAMATH_GPT_total_weight_is_1kg_total_weight_in_kg_eq_1_l99_9900


namespace NUMINAMATH_GPT_johns_father_age_l99_9953

variable {Age : Type} [OrderedRing Age]
variables (J M F : Age)

def john_age := J
def mother_age := M
def father_age := F

def john_younger_than_father (F J : Age) : Prop := F = 2 * J
def father_older_than_mother (F M : Age) : Prop := F = M + 4
def age_difference_between_john_and_mother (M J : Age) : Prop := M = J + 16

-- The question to be proved in Lean:
theorem johns_father_age :
  john_younger_than_father F J →
  father_older_than_mother F M →
  age_difference_between_john_and_mother M J →
  F = 40 := 
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_johns_father_age_l99_9953


namespace NUMINAMATH_GPT_value_of_expression_l99_9922

theorem value_of_expression : 8 * (6 - 4) + 2 = 18 := by
  sorry

end NUMINAMATH_GPT_value_of_expression_l99_9922


namespace NUMINAMATH_GPT_complete_square_form_l99_9908

noncomputable def quadratic_expr (x : ℝ) : ℝ := x^2 - 10 * x + 15

theorem complete_square_form (b c : ℤ) (h : ∀ x : ℝ, quadratic_expr x = 0 ↔ (x + b)^2 = c) :
  b + c = 5 :=
sorry

end NUMINAMATH_GPT_complete_square_form_l99_9908


namespace NUMINAMATH_GPT_find_m_l99_9977

open Real

noncomputable def curve_equation (x y : ℝ) : Prop :=
  (x - 1)^2 + y^2 = 1

noncomputable def line_equation (m t x y : ℝ) : Prop :=
  x = (sqrt 3 / 2) * t + m ∧ y = (1 / 2) * t

noncomputable def dist (x1 y1 x2 y2 : ℝ) : ℝ :=
  sqrt ((x2 - x1)^2 + (y2 - y1)^2)

theorem find_m (m : ℝ) (h_nonneg : 0 ≤ m) :
  (∀ (t1 t2 : ℝ), (∀ x y, line_equation m t1 x y → curve_equation x y) → 
                   (∀ x y, line_equation m t2 x y → curve_equation x y) →
                   (dist m 0 x1 y1) * (dist m 0 x2 y2) = 1) →
  m = 1 ∨ m = 1 + sqrt 2 :=
sorry

end NUMINAMATH_GPT_find_m_l99_9977


namespace NUMINAMATH_GPT_fraction_evaluation_l99_9938

def h (x : ℤ) : ℤ := 3 * x + 4
def k (x : ℤ) : ℤ := 4 * x - 3

theorem fraction_evaluation :
  (h (k (h 3))) / (k (h (k 3))) = 151 / 121 :=
by sorry

end NUMINAMATH_GPT_fraction_evaluation_l99_9938


namespace NUMINAMATH_GPT_Kvi_wins_race_l99_9963

/-- Define the frogs and their properties --/
structure Frog :=
  (name : String)
  (jump_distance_in_dm : ℕ) /-- jump distance in decimeters --/
  (jumps_per_cycle : ℕ) /-- number of jumps per cycle (unit time of reference) --/

def FrogKva : Frog := ⟨"Kva", 6, 2⟩
def FrogKvi : Frog := ⟨"Kvi", 4, 3⟩

/-- Define the conditions for the race --/
def total_distance_in_m : ℕ := 40
def total_distance_in_dm := total_distance_in_m * 10

/-- Racing function to determine winner --/
def race_winner (f1 f2 : Frog) (total_distance : ℕ) : String :=
  if (total_distance % (f1.jump_distance_in_dm * f1.jumps_per_cycle) < total_distance % (f2.jump_distance_in_dm * f2.jumps_per_cycle))
  then f1.name
  else f2.name

/-- Proving Kvi wins under the given conditions --/
theorem Kvi_wins_race :
  race_winner FrogKva FrogKvi total_distance_in_dm = "Kvi" :=
by
  sorry

end NUMINAMATH_GPT_Kvi_wins_race_l99_9963


namespace NUMINAMATH_GPT_rectangle_width_l99_9998

theorem rectangle_width (x y : ℝ) (h1 : x * y = 12) (h2 : x^2 + y^2 = 25) : y = 3 := 
by 
  sorry

end NUMINAMATH_GPT_rectangle_width_l99_9998


namespace NUMINAMATH_GPT_monotonic_iff_a_range_l99_9946

noncomputable def f (a x : ℝ) : ℝ := x^3 + a * x^2 + (a + 6) * x + 1

theorem monotonic_iff_a_range (a : ℝ) : 
  (∀ x1 x2, x1 ≤ x2 → f a x1 ≤ f a x2 ∨ f a x1 ≥ f a x2) ↔ (-3 < a ∧ a < 6) :=
by 
  sorry

end NUMINAMATH_GPT_monotonic_iff_a_range_l99_9946


namespace NUMINAMATH_GPT_distinct_schedules_l99_9997

-- Define the problem setting and assumptions
def subjects := ["Chinese", "Mathematics", "Politics", "English", "Physical Education", "Art"]

-- Given conditions
def math_in_first_three_periods (schedule : List String) : Prop :=
  ∃ k, (k < 3) ∧ (schedule.get! k = "Mathematics")

def english_not_in_sixth_period (schedule : List String) : Prop :=
  schedule.get! 5 ≠ "English"

-- Define the proof problem
theorem distinct_schedules : 
  ∃! (schedules : List (List String)), 
  (∀ schedule ∈ schedules, 
    math_in_first_three_periods schedule ∧ 
    english_not_in_sixth_period schedule) ∧
  schedules.length = 288 :=
by
  sorry

end NUMINAMATH_GPT_distinct_schedules_l99_9997


namespace NUMINAMATH_GPT_simplify_expression_l99_9932

theorem simplify_expression (x : ℝ) : 3 * x + 6 * x + 9 * x + 12 * x + 15 * x + 18 = 45 * x + 18 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l99_9932


namespace NUMINAMATH_GPT_circle_radius_zero_l99_9923

theorem circle_radius_zero (x y : ℝ) :
  4 * x^2 - 8 * x + 4 * y^2 + 16 * y + 20 = 0 → ∃ c : ℝ × ℝ, ∃ r : ℝ, (x - c.1)^2 + (y - c.2)^2 = r^2 ∧ r = 0 :=
by
  sorry

end NUMINAMATH_GPT_circle_radius_zero_l99_9923


namespace NUMINAMATH_GPT_polynomial_sum_of_squares_l99_9967

theorem polynomial_sum_of_squares (P : Polynomial ℝ) (hP : ∀ x : ℝ, 0 ≤ P.eval x) :
  ∃ (Q R : Polynomial ℝ), P = Q^2 + R^2 :=
sorry

end NUMINAMATH_GPT_polynomial_sum_of_squares_l99_9967


namespace NUMINAMATH_GPT_cos_sin_identity_l99_9988

theorem cos_sin_identity : 
  (Real.cos (14 * Real.pi / 180) * Real.cos (59 * Real.pi / 180) + 
   Real.sin (14 * Real.pi / 180) * Real.sin (121 * Real.pi / 180)) = (Real.sqrt 2 / 2) :=
by
  sorry

end NUMINAMATH_GPT_cos_sin_identity_l99_9988


namespace NUMINAMATH_GPT_no_positive_alpha_exists_l99_9927

theorem no_positive_alpha_exists :
  ¬ ∃ α > 0, ∀ x : ℝ, |Real.cos x| + |Real.cos (α * x)| > Real.sin x + Real.sin (α * x) :=
by
  sorry

end NUMINAMATH_GPT_no_positive_alpha_exists_l99_9927


namespace NUMINAMATH_GPT_quadratic_inequality_solution_l99_9955

theorem quadratic_inequality_solution:
  (∃ p : ℝ, ∀ x : ℝ, x^2 + p * x - 6 < 0 ↔ -3 < x ∧ x < 2) → ∃ p : ℝ, p = 1 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_quadratic_inequality_solution_l99_9955


namespace NUMINAMATH_GPT_value_of_ab_over_cd_l99_9973

theorem value_of_ab_over_cd (a b c d : ℚ) (h₁ : a / b = 2 / 3) (h₂ : c / b = 1 / 5) (h₃ : c / d = 7 / 15) : (a * b) / (c * d) = 140 / 9 :=
by
  sorry

end NUMINAMATH_GPT_value_of_ab_over_cd_l99_9973


namespace NUMINAMATH_GPT_average_score_first_2_matches_l99_9990

theorem average_score_first_2_matches (A : ℝ) 
  (h1 : 3 * 40 = 120) 
  (h2 : 5 * 36 = 180) 
  (h3 : 2 * A + 120 = 180) : 
  A = 30 := 
by 
  have hA : 2 * A = 60 := by linarith [h3]
  have hA2 : A = 30 := by linarith [hA]
  exact hA2

end NUMINAMATH_GPT_average_score_first_2_matches_l99_9990


namespace NUMINAMATH_GPT_carson_can_ride_giant_slide_exactly_twice_l99_9921

noncomputable def Carson_Carnival : Prop := 
  let total_time_available := 240
  let roller_coaster_time := 30
  let tilt_a_whirl_time := 60
  let giant_slide_time := 15
  let vortex_time := 45
  let bumper_cars_time := 25
  let roller_coaster_rides := 4
  let tilt_a_whirl_rides := 2
  let vortex_rides := 1
  let bumper_cars_rides := 3

  let total_time_spent := 
    roller_coaster_time * roller_coaster_rides +
    tilt_a_whirl_time * tilt_a_whirl_rides +
    vortex_time * vortex_rides +
    bumper_cars_time * bumper_cars_rides

  total_time_available - (total_time_spent + giant_slide_time * 2) = 0

theorem carson_can_ride_giant_slide_exactly_twice : Carson_Carnival :=
by
  unfold Carson_Carnival
  sorry -- proof will be provided here

end NUMINAMATH_GPT_carson_can_ride_giant_slide_exactly_twice_l99_9921


namespace NUMINAMATH_GPT_elvins_fixed_monthly_charge_l99_9992

-- Definition of the conditions
def january_bill (F C_J : ℝ) : Prop := F + C_J = 48
def february_bill (F C_J : ℝ) : Prop := F + 2 * C_J = 90

theorem elvins_fixed_monthly_charge (F C_J : ℝ) (h_jan : january_bill F C_J) (h_feb : february_bill F C_J) : F = 6 :=
by
  sorry

end NUMINAMATH_GPT_elvins_fixed_monthly_charge_l99_9992


namespace NUMINAMATH_GPT_minimize_f_sin_65_sin_40_l99_9980

noncomputable def f (x : ℝ) (m : ℝ) (n : ℝ) : ℝ := m^2 * x^2 + (n + 1) * x + 1

theorem minimize_f_sin_65_sin_40 (m n : ℝ) (h₁ : m = Real.sin (65 * Real.pi / 180))
  (h₂ : n = Real.sin (40 * Real.pi / 180)) : 
  ∃ x, x = -1 ∧ (∀ y, f y m n ≥ f (-1) m n) :=
by
  -- Proof to be completed
  sorry

end NUMINAMATH_GPT_minimize_f_sin_65_sin_40_l99_9980


namespace NUMINAMATH_GPT_buffy_breath_time_l99_9903

theorem buffy_breath_time (k : ℕ) (b : ℕ) (f : ℕ) 
  (h1 : k = 3 * 60) 
  (h2 : b = k - 20) 
  (h3 : f = b - 40) :
  f = 120 :=
by {
  sorry
}

end NUMINAMATH_GPT_buffy_breath_time_l99_9903


namespace NUMINAMATH_GPT_fraction_numerator_less_denominator_l99_9986

theorem fraction_numerator_less_denominator (x : ℝ) (h : -3 ≤ x ∧ x ≤ 3) :
  (8 * x - 3 < 9 + 5 * x) ↔ (-3 ≤ x ∧ x < 3) :=
by sorry

end NUMINAMATH_GPT_fraction_numerator_less_denominator_l99_9986


namespace NUMINAMATH_GPT_mike_total_payment_l99_9939

def camera_initial_cost : ℝ := 4000
def camera_increase_rate : ℝ := 0.30
def lens_initial_cost : ℝ := 400
def lens_discount : ℝ := 200
def sales_tax_rate : ℝ := 0.08

def new_camera_cost := camera_initial_cost * (1 + camera_increase_rate)
def discounted_lens_cost := lens_initial_cost - lens_discount
def total_purchase_before_tax := new_camera_cost + discounted_lens_cost
def sales_tax := total_purchase_before_tax * sales_tax_rate
def total_purchase_with_tax := total_purchase_before_tax + sales_tax

theorem mike_total_payment : total_purchase_with_tax = 5832 := by
  sorry

end NUMINAMATH_GPT_mike_total_payment_l99_9939


namespace NUMINAMATH_GPT_seeds_in_fourth_pot_l99_9906

theorem seeds_in_fourth_pot (total_seeds : ℕ) (total_pots : ℕ) (seeds_per_pot : ℕ) (first_three_pots : ℕ)
  (h1 : total_seeds = 10) (h2 : total_pots = 4) (h3 : seeds_per_pot = 3) (h4 : first_three_pots = 3) : 
  (total_seeds - (seeds_per_pot * first_three_pots)) = 1 :=
by
  sorry

end NUMINAMATH_GPT_seeds_in_fourth_pot_l99_9906


namespace NUMINAMATH_GPT_increasing_interval_l99_9994

-- Define the function f(x) = x^2 + 2*(a - 1)*x
def f (x : ℝ) (a : ℝ) : ℝ := x^2 + 2*(a - 1)*x

-- Define the condition for f(x) being increasing on [4, +∞)
def is_increasing_on_interval (a : ℝ) : Prop := 
  ∀ x y : ℝ, 4 ≤ x → x ≤ y → 
    f x a ≤ f y a

-- Define the main theorem that we need to prove
theorem increasing_interval (a : ℝ) (h : is_increasing_on_interval a) : -3 ≤ a :=
by 
  sorry -- proof is required, but omitted as per the instruction.

end NUMINAMATH_GPT_increasing_interval_l99_9994


namespace NUMINAMATH_GPT_sum_of_cubes_l99_9987

theorem sum_of_cubes (x y : ℝ) (h₁ : x + y = -1) (h₂ : x * y = -1) : x^3 + y^3 = -4 := by
  sorry

end NUMINAMATH_GPT_sum_of_cubes_l99_9987


namespace NUMINAMATH_GPT_factor_expression_l99_9912

theorem factor_expression (x : ℝ) : 3 * x^2 - 75 = 3 * (x + 5) * (x - 5) :=
by
  sorry

end NUMINAMATH_GPT_factor_expression_l99_9912


namespace NUMINAMATH_GPT_cylinder_volume_ratio_l99_9901

theorem cylinder_volume_ratio
  (S1 S2 : ℝ) (v1 v2 : ℝ)
  (lateral_area_equal : 2 * Real.pi * S1.sqrt = 2 * Real.pi * S2.sqrt)
  (base_area_ratio : S1 / S2 = 16 / 9) :
  v1 / v2 = 4 / 3 :=
by
  sorry

end NUMINAMATH_GPT_cylinder_volume_ratio_l99_9901


namespace NUMINAMATH_GPT_lcm_18_20_25_l99_9937

-- Lean 4 statement to prove the smallest positive integer divisible by 18, 20, and 25 is 900
theorem lcm_18_20_25 : Nat.lcm (Nat.lcm 18 20) 25 = 900 :=
by
  sorry

end NUMINAMATH_GPT_lcm_18_20_25_l99_9937


namespace NUMINAMATH_GPT_max_value_2x_minus_y_l99_9972

theorem max_value_2x_minus_y (x y : ℝ) (h₁ : x + y - 1 < 0) (h₂ : x - y ≤ 0) (h₃ : 0 ≤ x) :
  ∃ z, (z = 2 * x - y) ∧ (z ≤ (1 / 2)) :=
sorry

end NUMINAMATH_GPT_max_value_2x_minus_y_l99_9972


namespace NUMINAMATH_GPT_count_valid_four_digit_numbers_l99_9981

theorem count_valid_four_digit_numbers : 
  let valid_first_digits := (4*5 + 4*4)
  let valid_last_digits := (5*5 + 4*4)
  valid_first_digits * valid_last_digits = 1476 :=
by
  sorry

end NUMINAMATH_GPT_count_valid_four_digit_numbers_l99_9981


namespace NUMINAMATH_GPT_root_interval_sum_l99_9920

def f (x : ℝ) : ℝ := x^3 - x + 1

theorem root_interval_sum (a b : ℤ) (h1 : b - a = 1) (h2 : ∃! x : ℝ, a < x ∧ x < b ∧ f x = 0) : a + b = -3 :=
by
  sorry

end NUMINAMATH_GPT_root_interval_sum_l99_9920


namespace NUMINAMATH_GPT_cone_radius_l99_9968

noncomputable def radius_of_cone (V : ℝ) (h : ℝ) : ℝ := 
  3 / Real.sqrt (Real.pi)

theorem cone_radius :
  ∀ (V h : ℝ), V = 12 → h = 4 → radius_of_cone V h = 3 / Real.sqrt (Real.pi) :=
by
  intros V h hV hv
  sorry

end NUMINAMATH_GPT_cone_radius_l99_9968


namespace NUMINAMATH_GPT_perpendicular_lines_l99_9910

theorem perpendicular_lines (m : ℝ) : 
    (∀ x y : ℝ, x - 2 * y + 5 = 0) ∧ (∀ x y : ℝ, 2 * x + m * y - 6 = 0) → m = -1 :=
by
  sorry

end NUMINAMATH_GPT_perpendicular_lines_l99_9910


namespace NUMINAMATH_GPT_anne_speed_l99_9975

-- Definition of distance and time
def distance : ℝ := 6
def time : ℝ := 3

-- Statement to prove
theorem anne_speed : distance / time = 2 := by
  sorry

end NUMINAMATH_GPT_anne_speed_l99_9975


namespace NUMINAMATH_GPT_tan_of_alpha_l99_9957

noncomputable def point_P : ℝ × ℝ := (1, -2)

theorem tan_of_alpha (α : ℝ) (h : ∃ (P : ℝ × ℝ), P = point_P ∧ P.2 / P.1 = -2) :
  Real.tan α = -2 :=
sorry

end NUMINAMATH_GPT_tan_of_alpha_l99_9957


namespace NUMINAMATH_GPT_Mina_has_2_25_cent_coins_l99_9925

def MinaCoinProblem : Prop :=
  ∃ (x y z : ℕ), -- number of 5-cent, 10-cent, and 25-cent coins
  x + y + z = 15 ∧
  (74 - 4 * x - 3 * y = 30) ∧ -- corresponds to 30 different values can be obtained
  z = 2

theorem Mina_has_2_25_cent_coins : MinaCoinProblem :=
by 
  sorry

end NUMINAMATH_GPT_Mina_has_2_25_cent_coins_l99_9925


namespace NUMINAMATH_GPT_ellipse_properties_l99_9970

noncomputable def ellipse (x y a b : ℝ) : Prop :=
  (x^2 / a^2 + y^2 / b^2 = 1)

noncomputable def slopes_condition (x1 y1 x2 y2 : ℝ) (k_ab k_oa k_ob : ℝ) : Prop :=
  (k_ab^2 = k_oa * k_ob)

variables {x y : ℝ}

theorem ellipse_properties :
  (ellipse x y 2 1) ∧ -- Given ellipse equation
  (∃ (x1 y1 x2 y2 k_ab k_oa k_ob : ℝ), slopes_condition x1 y1 x2 y2 k_ab k_oa k_ob) →
  (∃ (OA OB : ℝ), OA^2 + OB^2 = 5) ∧ -- Prove sum of squares is constant
  (∃ (m : ℝ), (m = 1 → ∃ (line_eq : ℝ → ℝ), ∀ x, line_eq x = (1 / 2) * x + m)) -- Maximum area of triangle AOB

:= sorry

end NUMINAMATH_GPT_ellipse_properties_l99_9970


namespace NUMINAMATH_GPT_range_of_m_l99_9996

theorem range_of_m (m : ℝ) (h1 : 0 < m) (h2 : ∀ x : ℝ, 0 < x ∧ x ≤ 1 → |m * x^3 - Real.log x| ≥ 1) : m ≥ (1 / 3) * Real.exp 2 :=
sorry

end NUMINAMATH_GPT_range_of_m_l99_9996


namespace NUMINAMATH_GPT_two_fruits_probability_l99_9931

noncomputable def prob_exactly_two_fruits : ℚ := 10 / 9

theorem two_fruits_probability :
  (∀ (f : ℕ → ℝ), (f 0 = 1/3) ∧ (f 1 = 1/3) ∧ (f 2 = 1/3) ∧
   (∃ f1 f2 f3, f1 ≠ f2 ∧ f1 ≠ f3 ∧ f2 ≠ f3 ∧
    (f1 + f2 + f3 = prob_exactly_two_fruits))) :=
sorry

end NUMINAMATH_GPT_two_fruits_probability_l99_9931


namespace NUMINAMATH_GPT_black_balls_count_l99_9993

theorem black_balls_count :
  ∀ (r k : ℕ), r = 10 -> (2 : ℚ) / 7 = r / (r + k : ℚ) -> k = 25 := by
  intros r k hr hprob
  sorry

end NUMINAMATH_GPT_black_balls_count_l99_9993


namespace NUMINAMATH_GPT_sqrt_20_19_18_17_plus_1_eq_341_l99_9926

theorem sqrt_20_19_18_17_plus_1_eq_341 :
  Real.sqrt ((20: ℝ) * 19 * 18 * 17 + 1) = 341 := by
sorry

end NUMINAMATH_GPT_sqrt_20_19_18_17_plus_1_eq_341_l99_9926


namespace NUMINAMATH_GPT_factorize_expression_l99_9979

variable (a : ℝ)

theorem factorize_expression : a^3 - 2 * a^2 = a^2 * (a - 2) :=
by
  sorry

end NUMINAMATH_GPT_factorize_expression_l99_9979


namespace NUMINAMATH_GPT_multiply_decimals_l99_9936

theorem multiply_decimals :
  0.25 * 0.08 = 0.02 :=
sorry

end NUMINAMATH_GPT_multiply_decimals_l99_9936


namespace NUMINAMATH_GPT_factorization_of_expression_l99_9969

theorem factorization_of_expression
  (a b c : ℝ)
  (expansion : (b+c)*(c+a)*(a+b) + abc = (a+b+c)*(ab+ac+bc)) : 
  ∃ (m l : ℝ), (m = 0 ∧ l = a + b + c ∧ 
  (b+c)*(c+a)*(a+b) + abc = m*(a^2 + b^2 + c^2) + l*(ab + ac + bc)) :=
by
  sorry

end NUMINAMATH_GPT_factorization_of_expression_l99_9969


namespace NUMINAMATH_GPT_breakfast_problem_probability_l99_9999

def are_relatively_prime (m n : ℕ) : Prop :=
  Nat.gcd m n = 1

theorem breakfast_problem_probability : 
  ∃ m n : ℕ, are_relatively_prime m n ∧ 
  (1 / 1 * 9 / 11 * 6 / 10 * 1 / 3) * 1 = 9 / 55 ∧ m + n = 64 :=
by
  sorry

end NUMINAMATH_GPT_breakfast_problem_probability_l99_9999


namespace NUMINAMATH_GPT_problem_proof_l99_9971

def is_multiple_of (a b : ℕ) : Prop := ∃ k : ℕ, b = k * a

def y := 80 + 120 + 160 + 200 + 360 + 440 + 4040

theorem problem_proof :
  is_multiple_of 5 y ∧
  is_multiple_of 10 y ∧
  is_multiple_of 20 y ∧
  is_multiple_of 40 y := 
by
  sorry

end NUMINAMATH_GPT_problem_proof_l99_9971


namespace NUMINAMATH_GPT_geometric_sum_S15_l99_9941

noncomputable def S (n : ℕ) : ℝ := sorry  -- Assume S is defined for the sequence sum

theorem geometric_sum_S15 (S_5 S_10 : ℝ) (h1 : S_5 = 5) (h2 : S_10 = 30) : 
    S 15 = 155 := 
by 
  -- Placeholder for geometric sequence proof
  sorry

end NUMINAMATH_GPT_geometric_sum_S15_l99_9941


namespace NUMINAMATH_GPT_inequality1_solution_inequality2_solution_l99_9974

variables (x a : ℝ)

theorem inequality1_solution : (∀ x : ℝ, (2 * x) / (x + 1) < 1 ↔ -1 < x ∧ x < 1) :=
by
  sorry

theorem inequality2_solution (a : ℝ) : 
  (∀ x : ℝ, x^2 + (2 - a) * x - 2 * a ≥ 0 ↔ 
    (a = -2 → true) ∧ 
    (a > -2 → (x ≤ -2 ∨ x ≥ a)) ∧ 
    (a < -2 → (x ≤ a ∨ x ≥ -2))) :=
by
  sorry

end NUMINAMATH_GPT_inequality1_solution_inequality2_solution_l99_9974
