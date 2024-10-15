import Mathlib

namespace NUMINAMATH_GPT_total_cost_of_commodities_l166_16667

theorem total_cost_of_commodities (a b : ℕ) (h₁ : a = 477) (h₂ : a - b = 127) : a + b = 827 :=
by
  sorry

end NUMINAMATH_GPT_total_cost_of_commodities_l166_16667


namespace NUMINAMATH_GPT_graph_translation_l166_16624

theorem graph_translation (f : ℝ → ℝ) (x : ℝ) (h : f 1 = -1) :
  f (x - 1) - 1 = -2 :=
by
  sorry

end NUMINAMATH_GPT_graph_translation_l166_16624


namespace NUMINAMATH_GPT_boat_distance_downstream_l166_16657

-- Let v_s be the speed of the stream in km/h
-- Condition 1: In one hour, a boat goes 5 km against the stream.
-- Condition 2: The speed of the boat in still water is 8 km/h.

theorem boat_distance_downstream (v_s : ℝ) :
  (8 - v_s = 5) →
  (distance : ℝ) →
  8 + v_s = distance →
  distance = 11 := by
  sorry

end NUMINAMATH_GPT_boat_distance_downstream_l166_16657


namespace NUMINAMATH_GPT_no_natural_pairs_exist_l166_16696

theorem no_natural_pairs_exist (n m : ℕ) : ¬(n + 1) * (2 * n + 1) = 18 * m ^ 2 :=
by
  sorry

end NUMINAMATH_GPT_no_natural_pairs_exist_l166_16696


namespace NUMINAMATH_GPT_largest_possible_A_l166_16648

theorem largest_possible_A (A B : ℕ) (h1 : A = 5 * 2 + B) (h2 : B < 5) : A ≤ 14 :=
by
  have h3 : A ≤ 10 + 4 := sorry
  exact h3

end NUMINAMATH_GPT_largest_possible_A_l166_16648


namespace NUMINAMATH_GPT_arithmetic_sequence_fifth_term_l166_16629

theorem arithmetic_sequence_fifth_term :
  ∀ (a₁ d n : ℕ), a₁ = 3 → d = 4 → n = 5 → a₁ + (n - 1) * d = 19 :=
by
  intros a₁ d n ha₁ hd hn
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_fifth_term_l166_16629


namespace NUMINAMATH_GPT_length_of_XY_correct_l166_16678

noncomputable def length_of_XY (XZ : ℝ) (angleY : ℝ) (angleZ : ℝ) :=
  if angleZ = 90 ∧ angleY = 30 then 8 * Real.sqrt 3 else panic! "Invalid triangle angles"

theorem length_of_XY_correct : length_of_XY 12 30 90 = 8 * Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_length_of_XY_correct_l166_16678


namespace NUMINAMATH_GPT_poly_has_integer_roots_iff_a_eq_one_l166_16616

-- Definition: a positive real number
def pos_real (a : ℝ) : Prop := a > 0

-- The polynomial
def p (a : ℝ) (x : ℝ) : ℝ := a^3 * x^3 + a^2 * x^2 + a * x + a

-- The main theorem
theorem poly_has_integer_roots_iff_a_eq_one (a : ℝ) (x : ℤ) :
  (pos_real a ∧ ∃ x : ℤ, p a x = 0) ↔ a = 1 :=
by sorry

end NUMINAMATH_GPT_poly_has_integer_roots_iff_a_eq_one_l166_16616


namespace NUMINAMATH_GPT_surface_area_of_cube_edge_8_l166_16687

-- Definition of surface area of a cube
def surface_area_of_cube (edge_length : ℕ) : ℕ :=
  6 * (edge_length * edge_length)

-- Theorem to prove the surface area for a cube with edge length of 8 cm is 384 cm²
theorem surface_area_of_cube_edge_8 : surface_area_of_cube 8 = 384 :=
by
  -- The proof will be inserted here. We use sorry to indicate the missing proof.
  sorry

end NUMINAMATH_GPT_surface_area_of_cube_edge_8_l166_16687


namespace NUMINAMATH_GPT_extra_sweets_l166_16654

theorem extra_sweets (S : ℕ) (h1 : ∀ n: ℕ, S = 120 * 38) : 
    (38 - (S / 190) = 14) :=
by
  -- Here we will provide the proof 
  sorry

end NUMINAMATH_GPT_extra_sweets_l166_16654


namespace NUMINAMATH_GPT_distinct_nonzero_reals_product_l166_16635

theorem distinct_nonzero_reals_product 
  (x y : ℝ) 
  (hx : x ≠ 0)
  (hy : y ≠ 0)
  (hxy: x ≠ y)
  (h : x + 3 / x = y + 3 / y) :
  x * y = 3 :=
sorry

end NUMINAMATH_GPT_distinct_nonzero_reals_product_l166_16635


namespace NUMINAMATH_GPT_game_a_greater_than_game_c_l166_16634

-- Definitions of probabilities for heads and tails
def prob_heads : ℚ := 3 / 4
def prob_tails : ℚ := 1 / 4

-- Define the probabilities for Game A and Game C based on given conditions
def prob_game_a : ℚ := (prob_heads ^ 4) + (prob_tails ^ 4)
def prob_game_c : ℚ :=
  (prob_heads ^ 5) +
  (prob_tails ^ 5) +
  (prob_heads ^ 3 * prob_tails ^ 2) +
  (prob_tails ^ 3 * prob_heads ^ 2)

-- Define the difference
def prob_difference : ℚ := prob_game_a - prob_game_c

-- The theorem to be proved
theorem game_a_greater_than_game_c :
  prob_difference = 3 / 64 :=
by
  sorry

end NUMINAMATH_GPT_game_a_greater_than_game_c_l166_16634


namespace NUMINAMATH_GPT_right_triangle_angle_l166_16695

theorem right_triangle_angle (x : ℝ) (h1 : x + 5 * x = 90) : 5 * x = 75 :=
by
  sorry

end NUMINAMATH_GPT_right_triangle_angle_l166_16695


namespace NUMINAMATH_GPT_percentage_of_alcohol_in_first_vessel_l166_16672

variable (x : ℝ) -- percentage of alcohol in the first vessel in decimal form, i.e., x% is represented as x/100

-- conditions
variable (v1_capacity : ℝ := 2)
variable (v2_capacity : ℝ := 6)
variable (v2_alcohol_concentration : ℝ := 0.5)
variable (total_capacity : ℝ := 10)
variable (new_concentration : ℝ := 0.37)

theorem percentage_of_alcohol_in_first_vessel :
  (x / 100) * v1_capacity + v2_alcohol_concentration * v2_capacity = new_concentration * total_capacity -> x = 35 := 
by
  sorry

end NUMINAMATH_GPT_percentage_of_alcohol_in_first_vessel_l166_16672


namespace NUMINAMATH_GPT_exists_plane_perpendicular_l166_16641

-- Definitions of line, plane and perpendicularity intersection etc.
variables (Point : Type) (Line Plane : Type)
variables (l : Line) (α : Plane) (intersects : Line → Plane → Prop)
variables (perpendicular : Line → Plane → Prop) (perpendicular_planes : Plane → Plane → Prop)
variables (β : Plane) (subset : Line → Plane → Prop)

-- Conditions
axiom line_intersects_plane (h1 : intersects l α) : Prop
axiom line_not_perpendicular_plane (h2 : ¬perpendicular l α) : Prop

-- The main statement to prove
theorem exists_plane_perpendicular (h1 : intersects l α) (h2 : ¬perpendicular l α) :
  ∃ (β : Plane), (subset l β) ∧ (perpendicular_planes β α) :=
sorry

end NUMINAMATH_GPT_exists_plane_perpendicular_l166_16641


namespace NUMINAMATH_GPT_simplify_rationalize_expr_l166_16666

theorem simplify_rationalize_expr :
  (1 / (2 + 1 / (Real.sqrt 5 + 2))) = (Real.sqrt 5 / 5) :=
by
  sorry

end NUMINAMATH_GPT_simplify_rationalize_expr_l166_16666


namespace NUMINAMATH_GPT_train_speed_is_36_kph_l166_16692

-- Define the given conditions
def distance_meters : ℕ := 1800
def time_minutes : ℕ := 3

-- Convert distance from meters to kilometers
def distance_kilometers : ℕ -> ℕ := fun d => d / 1000
-- Convert time from minutes to hours
def time_hours : ℕ -> ℚ := fun t => (t : ℚ) / 60

-- Calculate speed in kilometers per hour
def speed_kph (d : ℕ) (t : ℕ) : ℚ :=
  let d_km := d / 1000
  let t_hr := (t : ℚ) / 60
  d_km / t_hr

-- The theorem to prove the speed
theorem train_speed_is_36_kph :
  speed_kph distance_meters time_minutes = 36 := by
  sorry

end NUMINAMATH_GPT_train_speed_is_36_kph_l166_16692


namespace NUMINAMATH_GPT_shelves_in_room_l166_16674

theorem shelves_in_room
  (n_action_figures_per_shelf : ℕ)
  (total_action_figures : ℕ)
  (h1 : n_action_figures_per_shelf = 10)
  (h2 : total_action_figures = 80) :
  total_action_figures / n_action_figures_per_shelf = 8 := by
  sorry

end NUMINAMATH_GPT_shelves_in_room_l166_16674


namespace NUMINAMATH_GPT_eval_x_power_x_power_x_at_3_l166_16602

theorem eval_x_power_x_power_x_at_3 : (3^3)^(3^3) = 27^27 := by
    sorry

end NUMINAMATH_GPT_eval_x_power_x_power_x_at_3_l166_16602


namespace NUMINAMATH_GPT_find_f_5_l166_16619

section
variables (f : ℝ → ℝ)

-- Given condition
def functional_equation (x : ℝ) : Prop := x * f x = 2 * f (1 - x) + 1

-- Prove that f(5) = 1/12 given the condition
theorem find_f_5 (h : ∀ x, functional_equation f x) : f 5 = 1 / 12 :=
sorry
end

end NUMINAMATH_GPT_find_f_5_l166_16619


namespace NUMINAMATH_GPT_identify_translation_l166_16625

def phenomenon (x : String) : Prop :=
  x = "translational"

def option_A : Prop := phenomenon "rotational"
def option_B : Prop := phenomenon "rotational"
def option_C : Prop := phenomenon "translational"
def option_D : Prop := phenomenon "rotational"

theorem identify_translation :
  (¬ option_A) ∧ (¬ option_B) ∧ option_C ∧ (¬ option_D) :=
  by {
    sorry
  }

end NUMINAMATH_GPT_identify_translation_l166_16625


namespace NUMINAMATH_GPT_sum_of_eight_numbers_l166_16680

theorem sum_of_eight_numbers (avg : ℝ) (num_of_items : ℕ) (h_avg : avg = 5.3) (h_items : num_of_items = 8) :
  avg * num_of_items = 42.4 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_eight_numbers_l166_16680


namespace NUMINAMATH_GPT_find_larger_number_l166_16614

theorem find_larger_number :
  ∃ (L S : ℕ), L - S = 1365 ∧ L = 6 * S + 15 ∧ L = 1635 :=
sorry

end NUMINAMATH_GPT_find_larger_number_l166_16614


namespace NUMINAMATH_GPT_find_x_values_l166_16689

theorem find_x_values (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0)
  (h1 : x + 1 / y = 12) (h2 : y + 1 / x = 3 / 8) :
  x = 4 ∨ x = 8 :=
by
  sorry

end NUMINAMATH_GPT_find_x_values_l166_16689


namespace NUMINAMATH_GPT_second_train_speed_l166_16627

theorem second_train_speed :
  ∃ v : ℝ, 
  (∀ t : ℝ, 20 * t = v * t + 50) ∧
  (∃ t : ℝ, 20 * t + v * t = 450) →
  v = 16 :=
by
  sorry

end NUMINAMATH_GPT_second_train_speed_l166_16627


namespace NUMINAMATH_GPT_sum_of_ages_l166_16665

variables (S F : ℕ)

theorem sum_of_ages
  (h1 : F - 18 = 3 * (S - 18))
  (h2 : F = 2 * S) :
  S + F = 108 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_ages_l166_16665


namespace NUMINAMATH_GPT_paving_time_together_l166_16650

/-- Define the rate at which Mary alone paves the driveway -/
noncomputable def Mary_rate : ℝ := 1 / 4

/-- Define the rate at which Hillary alone paves the driveway -/
noncomputable def Hillary_rate : ℝ := 1 / 3

/-- Define the increased rate of Mary when working together -/
noncomputable def Mary_rate_increased := Mary_rate + (0.3333 * Mary_rate)

/-- Define the decreased rate of Hillary when working together -/
noncomputable def Hillary_rate_decreased := Hillary_rate - (0.5 * Hillary_rate)

/-- Combine their rates when working together -/
noncomputable def combined_rate := Mary_rate_increased + Hillary_rate_decreased

/-- Prove that the time taken to pave the driveway together is approximately 2 hours -/
theorem paving_time_together : abs ((1 / combined_rate) - 2) < 0.0001 :=
by
  sorry

end NUMINAMATH_GPT_paving_time_together_l166_16650


namespace NUMINAMATH_GPT_gain_percent_is_50_l166_16644

theorem gain_percent_is_50
  (C : ℕ) (S : ℕ) (hC : C = 10) (hS : S = 15) : ((S - C) / C : ℚ) * 100 = 50 := by
  sorry

end NUMINAMATH_GPT_gain_percent_is_50_l166_16644


namespace NUMINAMATH_GPT_kim_money_l166_16649

theorem kim_money (S P K : ℝ) (h1 : K = 1.40 * S) (h2 : S = 0.80 * P) (h3 : S + P = 1.80) : K = 1.12 :=
by sorry

end NUMINAMATH_GPT_kim_money_l166_16649


namespace NUMINAMATH_GPT_total_bouquets_sold_l166_16677

-- Define the conditions as variables
def monday_bouquets : ℕ := 12
def tuesday_bouquets : ℕ := 3 * monday_bouquets
def wednesday_bouquets : ℕ := tuesday_bouquets / 3

-- The statement to prove
theorem total_bouquets_sold : 
  monday_bouquets + tuesday_bouquets + wednesday_bouquets = 60 :=
by
  -- The proof is omitted using sorry
  sorry

end NUMINAMATH_GPT_total_bouquets_sold_l166_16677


namespace NUMINAMATH_GPT_seven_distinct_integers_exist_pair_l166_16611

theorem seven_distinct_integers_exist_pair (a : Fin 7 → ℕ) (h_distinct : Function.Injective a)
  (h_bound : ∀ i, 1 ≤ a i ∧ a i ≤ 126) :
  ∃ i j : Fin 7, i ≠ j ∧ (1 / 2 : ℚ) ≤ (a i : ℚ) / a j ∧ (a i : ℚ) / a j ≤ 2 := sorry

end NUMINAMATH_GPT_seven_distinct_integers_exist_pair_l166_16611


namespace NUMINAMATH_GPT_no_integer_roots_l166_16652

def cubic_polynomial (a b c d x : ℤ) : ℤ :=
  a * x^3 + b * x^2 + c * x + d

theorem no_integer_roots (a b c d : ℤ) (h1 : cubic_polynomial a b c d 1 = 2015) (h2 : cubic_polynomial a b c d 2 = 2017) :
  ∀ x : ℤ, cubic_polynomial a b c d x ≠ 2016 :=
by
  sorry

end NUMINAMATH_GPT_no_integer_roots_l166_16652


namespace NUMINAMATH_GPT_find_initial_quantities_l166_16612

/-- 
Given:
- x + y = 92
- (2/5) * x + (1/4) * y = 26

Prove:
- x = 20
- y = 72
-/
theorem find_initial_quantities (x y : ℝ) (h1 : x + y = 92) (h2 : (2/5) * x + (1/4) * y = 26) :
  x = 20 ∧ y = 72 :=
sorry

end NUMINAMATH_GPT_find_initial_quantities_l166_16612


namespace NUMINAMATH_GPT_gcd_4004_10010_l166_16645

theorem gcd_4004_10010 : Nat.gcd 4004 10010 = 2002 :=
by
  have h1 : 4004 = 4 * 1001 := by norm_num
  have h2 : 10010 = 10 * 1001 := by norm_num
  sorry

end NUMINAMATH_GPT_gcd_4004_10010_l166_16645


namespace NUMINAMATH_GPT_possible_landing_l166_16610

-- There are 1985 airfields
def num_airfields : ℕ := 1985

-- 50 airfields where planes could potentially land
def num_land_airfields : ℕ := 50

-- Define the structure of the problem
structure AirfieldSetup :=
  (airfields : Fin num_airfields → Fin num_land_airfields)

-- There exists a configuration such that the conditions are met
theorem possible_landing : ∃ (setup : AirfieldSetup), 
  (∀ i : Fin num_airfields, -- For each airfield
    ∃ j : Fin num_land_airfields, -- There exists a landing airfield
    setup.airfields i = j) -- The plane lands at this airfield.
:=
sorry

end NUMINAMATH_GPT_possible_landing_l166_16610


namespace NUMINAMATH_GPT_axis_of_symmetry_l166_16639

-- Definitions for conditions
variable (ω : ℝ) (φ : ℝ) (A B : ℝ)
noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (ω * x + φ)

-- Hypotheses
axiom ω_pos : ω > 0
axiom φ_bound : 0 ≤ φ ∧ φ < Real.pi
axiom even_func : ∀ x, f x = f (-x)
axiom dist_AB : abs (B - A) = 4 * Real.sqrt 2

-- Proof statement
theorem axis_of_symmetry : ∃ x : ℝ, x = 4 := 
sorry

end NUMINAMATH_GPT_axis_of_symmetry_l166_16639


namespace NUMINAMATH_GPT_find_b_c_d_sum_l166_16642

theorem find_b_c_d_sum :
  ∃ (b c d : ℤ), (∀ n : ℕ, n > 0 → 
    a_n = b * (⌊(n : ℝ)^(1/3)⌋.natAbs : ℤ) + d ∧
    b = 2 ∧ c = 0 ∧ d = 0) ∧ (b + c + d = 2) :=
sorry

end NUMINAMATH_GPT_find_b_c_d_sum_l166_16642


namespace NUMINAMATH_GPT_decreasing_exponential_iff_l166_16668

theorem decreasing_exponential_iff {a : ℝ} :
  (∀ x y : ℝ, x < y → (a - 1)^y < (a - 1)^x) ↔ (1 < a ∧ a < 2) :=
by 
  sorry

end NUMINAMATH_GPT_decreasing_exponential_iff_l166_16668


namespace NUMINAMATH_GPT_evaluate_expression_l166_16603

theorem evaluate_expression : 
  (3000^3 - 2999 * 3000^2 - 2999^2 * 3000 + 2999^3 = 26991001) :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l166_16603


namespace NUMINAMATH_GPT_no_snow_probability_l166_16638

theorem no_snow_probability (p1 p2 p3 p4 : ℚ) 
  (h1 : p1 = 2 / 3) 
  (h2 : p2 = 3 / 4) 
  (h3 : p3 = 5 / 6) 
  (h4 : p4 = 1 / 2) : 
  (1 - p1) * (1 - p2) * (1 - p3) * (1 - p4) = 1 / 144 :=
by
  sorry

end NUMINAMATH_GPT_no_snow_probability_l166_16638


namespace NUMINAMATH_GPT_road_trip_mileage_base10_l166_16646

-- Defining the base 8 number 3452
def base8_to_base10 (n : Nat) : Nat :=
  3 * 8^3 + 4 * 8^2 + 5 * 8^1 + 2 * 8^0

-- Stating the problem as a theorem
theorem road_trip_mileage_base10 : base8_to_base10 3452 = 1834 := by
  sorry

end NUMINAMATH_GPT_road_trip_mileage_base10_l166_16646


namespace NUMINAMATH_GPT_polygon_sum_13th_position_l166_16609

theorem polygon_sum_13th_position :
  let sum_n : ℕ := (100 * 101) / 2;
  2 * sum_n = 10100 :=
by
  sorry

end NUMINAMATH_GPT_polygon_sum_13th_position_l166_16609


namespace NUMINAMATH_GPT_correct_average_marks_l166_16683

theorem correct_average_marks 
  (avg_marks : ℝ) 
  (num_students : ℕ) 
  (incorrect_marks : ℕ → (ℝ × ℝ)) :
  avg_marks = 85 →
  num_students = 50 →
  incorrect_marks 0 = (95, 45) →
  incorrect_marks 1 = (78, 58) →
  incorrect_marks 2 = (120, 80) →
  (∃ corrected_avg_marks : ℝ, corrected_avg_marks = 82.8) :=
by
  sorry

end NUMINAMATH_GPT_correct_average_marks_l166_16683


namespace NUMINAMATH_GPT_B_work_rate_l166_16669

theorem B_work_rate (A_rate C_rate combined_rate : ℝ) (B_days : ℝ) (hA : A_rate = 1 / 4) (hC : C_rate = 1 / 8) (hCombined : A_rate + 1 / B_days + C_rate = 1 / 2) : B_days = 8 :=
by
  sorry

end NUMINAMATH_GPT_B_work_rate_l166_16669


namespace NUMINAMATH_GPT_ratio_problem_l166_16606

theorem ratio_problem (x : ℕ) : (20 / 1 : ℝ) = (x / 10 : ℝ) → x = 200 := by
  sorry

end NUMINAMATH_GPT_ratio_problem_l166_16606


namespace NUMINAMATH_GPT_video_game_cost_l166_16661

theorem video_game_cost :
  let september_saving : ℕ := 50
  let october_saving : ℕ := 37
  let november_saving : ℕ := 11
  let mom_gift : ℕ := 25
  let remaining_money : ℕ := 36
  let total_savings : ℕ := september_saving + october_saving + november_saving
  let total_with_gift : ℕ := total_savings + mom_gift
  let game_cost : ℕ := total_with_gift - remaining_money
  game_cost = 87 :=
by
  sorry

end NUMINAMATH_GPT_video_game_cost_l166_16661


namespace NUMINAMATH_GPT_probability_of_heads_or_five_tails_is_one_eighth_l166_16637

namespace coin_flip

def num_heads_or_at_least_five_tails : ℕ :=
1 + 6 + 1

def total_outcomes : ℕ :=
2^6

def probability_heads_or_five_tails : ℚ :=
num_heads_or_at_least_five_tails / total_outcomes

theorem probability_of_heads_or_five_tails_is_one_eighth :
  probability_heads_or_five_tails = 1 / 8 := by
  sorry

end coin_flip

end NUMINAMATH_GPT_probability_of_heads_or_five_tails_is_one_eighth_l166_16637


namespace NUMINAMATH_GPT_finite_points_outside_unit_circle_l166_16630

noncomputable def centroid (x y z : ℝ × ℝ) : ℝ × ℝ := 
  ((x.1 + y.1 + z.1) / 3, (x.2 + y.2 + z.2) / 3)

theorem finite_points_outside_unit_circle
  (A₁ B₁ C₁ D₁ : ℝ × ℝ)
  (A : ℕ → ℝ × ℝ)
  (B : ℕ → ℝ × ℝ)
  (C : ℕ → ℝ × ℝ)
  (D : ℕ → ℝ × ℝ)
  (hA : ∀ n, A (n + 1) = centroid (B n) (C n) (D n))
  (hB : ∀ n, B (n + 1) = centroid (A n) (C n) (D n))
  (hC : ∀ n, C (n + 1) = centroid (A n) (B n) (D n))
  (hD : ∀ n, D (n + 1) = centroid (A n) (B n) (C n))
  (h₀ : A 1 = A₁ ∧ B 1 = B₁ ∧ C 1 = C₁ ∧ D 1 = D₁)
  : ∃ N : ℕ, ∀ n > N, (A n).1 * (A n).1 + (A n).2 * (A n).2 ≤ 1 :=
sorry

end NUMINAMATH_GPT_finite_points_outside_unit_circle_l166_16630


namespace NUMINAMATH_GPT_max_y_for_f_eq_0_l166_16685

-- Define f(x, y, z) as the remainder when (x - y)! is divided by (x + z).
def f (x y z : ℕ) : ℕ :=
  Nat.factorial (x - y) % (x + z)

-- Conditions given in the problem
variable (x y z : ℕ)
variable (hx : x = 100)
variable (hz : z = 50)

theorem max_y_for_f_eq_0 : 
  f x y z = 0 → y ≤ 75 :=
by
  rw [hx, hz]
  sorry

end NUMINAMATH_GPT_max_y_for_f_eq_0_l166_16685


namespace NUMINAMATH_GPT_prob_B_given_A_l166_16600

theorem prob_B_given_A (P_A P_B P_A_and_B : ℝ) (h1 : P_A = 0.06) (h2 : P_B = 0.08) (h3 : P_A_and_B = 0.02) :
  (P_A_and_B / P_A) = (1 / 3) :=
by
  -- substitute values
  sorry

end NUMINAMATH_GPT_prob_B_given_A_l166_16600


namespace NUMINAMATH_GPT_find_f_and_min_g_l166_16691

theorem find_f_and_min_g (f g : ℝ → ℝ) (a : ℝ)
  (h1 : ∀ x : ℝ, f (2 * x - 3) = 4 * x^2 + 2 * x + 1)
  (h2 : ∀ x : ℝ, g x = f (x + a) - 7 * x):
  
  (∀ x : ℝ, f x = x^2 + 7 * x + 13) ∧
  
  (∀ a : ℝ, 
    ∀ x : ℝ, 
      (x = 1 → (a ≥ -1 → g x = a^2 + 9 * a + 14)) ∧
      (-3 < a ∧ a < -1 → g (-a) = 7 * a + 13) ∧
      (x = 3 → (a ≤ -3 → g x = a^2 + 13 * a + 22))) :=
by
  sorry

end NUMINAMATH_GPT_find_f_and_min_g_l166_16691


namespace NUMINAMATH_GPT_complex_quadrant_l166_16626

def z1 := Complex.mk 1 (-2)
def z2 := Complex.mk 2 1
def z := z1 * z2

theorem complex_quadrant : z = Complex.mk 4 (-3) ∧ z.re > 0 ∧ z.im < 0 :=
by
  sorry

end NUMINAMATH_GPT_complex_quadrant_l166_16626


namespace NUMINAMATH_GPT_jimin_rank_l166_16617

theorem jimin_rank (seokjin_rank : ℕ) (h1 : seokjin_rank = 4) (h2 : ∃ jimin_rank, jimin_rank = seokjin_rank + 1) : 
  ∃ jimin_rank, jimin_rank = 5 := 
by
  sorry

end NUMINAMATH_GPT_jimin_rank_l166_16617


namespace NUMINAMATH_GPT_minimum_single_discount_l166_16607

theorem minimum_single_discount (n : ℕ) :
  (∀ x : ℝ, 0 < x → 
    ((1 - n / 100) * x < (1 - 0.18) * (1 - 0.18) * x) ∧
    ((1 - n / 100) * x < (1 - 0.12) * (1 - 0.12) * (1 - 0.12) * x) ∧
    ((1 - n / 100) * x < (1 - 0.28) * (1 - 0.07) * x))
  ↔ n = 34 :=
by
  sorry

end NUMINAMATH_GPT_minimum_single_discount_l166_16607


namespace NUMINAMATH_GPT_least_number_to_add_l166_16673

theorem least_number_to_add (n : ℕ) (h : n = 28523) : 
  ∃ x, x + n = 29560 ∧ 3 ∣ (x + n) ∧ 5 ∣ (x + n) ∧ 7 ∣ (x + n) ∧ 8 ∣ (x + n) :=
by 
  sorry

end NUMINAMATH_GPT_least_number_to_add_l166_16673


namespace NUMINAMATH_GPT_original_price_of_apples_l166_16662

-- Define the conditions and problem
theorem original_price_of_apples 
  (discounted_price : ℝ := 0.60 * original_price)
  (total_cost : ℝ := 30)
  (weight : ℝ := 10) :
  original_price = 5 :=
by
  -- This is the point where the proof steps would go.
  sorry

end NUMINAMATH_GPT_original_price_of_apples_l166_16662


namespace NUMINAMATH_GPT_kyunghoon_time_to_go_down_l166_16682

theorem kyunghoon_time_to_go_down (d : ℕ) (t_up t_down total_time : ℕ) : 
  ((t_up = d / 3) ∧ (t_down = (d + 2) / 4) ∧ (total_time = 4) → (t_up + t_down = total_time) → (t_down = 2)) := 
by
  sorry

end NUMINAMATH_GPT_kyunghoon_time_to_go_down_l166_16682


namespace NUMINAMATH_GPT_percentage_refund_l166_16643

theorem percentage_refund
  (initial_amount : ℕ)
  (sweater_cost : ℕ)
  (tshirt_cost : ℕ)
  (shoes_cost : ℕ)
  (amount_left_after_refund : ℕ)
  (refund_percentage : ℕ) :
  initial_amount = 74 →
  sweater_cost = 9 →
  tshirt_cost = 11 →
  shoes_cost = 30 →
  amount_left_after_refund = 51 →
  refund_percentage = 90 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_percentage_refund_l166_16643


namespace NUMINAMATH_GPT_emir_needs_more_money_l166_16651

def dictionary_cost : ℕ := 5
def dinosaur_book_cost : ℕ := 11
def cookbook_cost : ℕ := 5
def saved_money : ℕ := 19
def total_cost : ℕ := dictionary_cost + dinosaur_book_cost + cookbook_cost
def additional_money_needed : ℕ := total_cost - saved_money

theorem emir_needs_more_money : additional_money_needed = 2 := by
  sorry

end NUMINAMATH_GPT_emir_needs_more_money_l166_16651


namespace NUMINAMATH_GPT_initial_tomatoes_l166_16622

/-- 
Given the conditions:
  - The farmer picked 134 tomatoes yesterday.
  - The farmer picked 30 tomatoes today.
  - The farmer will have 7 tomatoes left after today.
Prove that the initial number of tomatoes in the farmer's garden was 171.
--/

theorem initial_tomatoes (picked_yesterday : ℕ) (picked_today : ℕ) (left_tomatoes : ℕ)
  (h1 : picked_yesterday = 134)
  (h2 : picked_today = 30)
  (h3 : left_tomatoes = 7) :
  (picked_yesterday + picked_today + left_tomatoes) = 171 :=
by 
  sorry

end NUMINAMATH_GPT_initial_tomatoes_l166_16622


namespace NUMINAMATH_GPT_complex_magnitude_l166_16694

variable (a b : ℝ)

theorem complex_magnitude :
  ((1 + 2 * a * Complex.I) * Complex.I = 1 - b * Complex.I) →
  Complex.normSq (a + b * Complex.I) = 5/4 :=
by
  intro h
  -- Add missing logic to transform assumption to the norm result
  sorry

end NUMINAMATH_GPT_complex_magnitude_l166_16694


namespace NUMINAMATH_GPT_max_stamps_l166_16640

-- Definitions based on conditions
def price_of_stamp := 28 -- in cents
def total_money := 3600 -- in cents

-- The theorem statement
theorem max_stamps (price_of_stamp total_money : ℕ) : (total_money / price_of_stamp) = 128 := by
  sorry

end NUMINAMATH_GPT_max_stamps_l166_16640


namespace NUMINAMATH_GPT_find_k_l166_16623

theorem find_k (k : ℝ) :
  (∀ x, x^2 + k*x + 10 = 0 → (∃ r s : ℝ, x = r ∨ x = s) ∧ r + s = -k ∧ r * s = 10) ∧
  (∀ x, x^2 - k*x + 10 = 0 → (∃ r s : ℝ, x = r + 4 ∨ x = s + 4) ∧ (r + 4) + (s + 4) = k) → 
  k = 4 :=
by
  sorry

end NUMINAMATH_GPT_find_k_l166_16623


namespace NUMINAMATH_GPT_profit_days_l166_16604

theorem profit_days (total_days : ℕ) (mean_profit_month first_half_days second_half_days : ℕ)
  (mean_profit_first_half mean_profit_second_half : ℕ)
  (h1 : mean_profit_month * total_days = (mean_profit_first_half * first_half_days + mean_profit_second_half * second_half_days))
  (h2 : first_half_days + second_half_days = total_days)
  (h3 : mean_profit_month = 350)
  (h4 : mean_profit_first_half = 225)
  (h5 : mean_profit_second_half = 475)
  (h6 : total_days = 30) : 
  first_half_days = 15 ∧ second_half_days = 15 := 
by 
  sorry

end NUMINAMATH_GPT_profit_days_l166_16604


namespace NUMINAMATH_GPT_arc_length_of_circle_l166_16601

section circle_arc_length

def diameter (d : ℝ) : Prop := d = 4
def central_angle_deg (θ_d : ℝ) : Prop := θ_d = 36

theorem arc_length_of_circle
  (d : ℝ) (θ_d : ℝ) (r : ℝ := d / 2) (θ : ℝ := θ_d * (π / 180)) (l : ℝ := θ * r) :
  diameter d → central_angle_deg θ_d → l = 2 * π / 5 :=
by
  intros h1 h2
  sorry

end circle_arc_length

end NUMINAMATH_GPT_arc_length_of_circle_l166_16601


namespace NUMINAMATH_GPT_calculate_expression_l166_16656

-- Defining the main theorem to prove
theorem calculate_expression (a b : ℝ) : 
  3 * a + 2 * b - 2 * (a - b) = a + 4 * b :=
by 
  sorry

end NUMINAMATH_GPT_calculate_expression_l166_16656


namespace NUMINAMATH_GPT_solution_x_y_zero_l166_16636

theorem solution_x_y_zero (x y : ℤ) (h : x^2 * y^2 = x^2 + y^2) : x = 0 ∧ y = 0 :=
by
sorry

end NUMINAMATH_GPT_solution_x_y_zero_l166_16636


namespace NUMINAMATH_GPT_solve_equation_and_find_c_d_l166_16628

theorem solve_equation_and_find_c_d : 
  ∃ (c d : ℕ), (∃ x : ℝ, x^2 + 14 * x = 84 ∧ x = Real.sqrt c - d) ∧ c + d = 140 := 
sorry

end NUMINAMATH_GPT_solve_equation_and_find_c_d_l166_16628


namespace NUMINAMATH_GPT_triangle_DEF_area_l166_16679

theorem triangle_DEF_area (DE height : ℝ) (hDE : DE = 12) (hHeight : height = 15) : 
  (1/2) * DE * height = 90 :=
by
  rw [hDE, hHeight]
  norm_num

end NUMINAMATH_GPT_triangle_DEF_area_l166_16679


namespace NUMINAMATH_GPT_same_solutions_a_value_l166_16664

theorem same_solutions_a_value (a x : ℝ) (h1 : 2 * x + 1 = 3) (h2 : 3 - (a - x) / 3 = 1) : a = 7 := by
  sorry

end NUMINAMATH_GPT_same_solutions_a_value_l166_16664


namespace NUMINAMATH_GPT_quadrilateral_area_ratio_l166_16618

noncomputable def area_of_octagon (a : ℝ) : ℝ := 2 * a^2 * (1 + Real.sqrt 2)

noncomputable def area_of_square (s : ℝ) : ℝ := s^2

theorem quadrilateral_area_ratio (a : ℝ) (s : ℝ)
    (h1 : s = a * Real.sqrt (2 + Real.sqrt 2))
    : (area_of_square s) / (area_of_octagon a) = Real.sqrt 2 / 2 :=
by
  sorry

end NUMINAMATH_GPT_quadrilateral_area_ratio_l166_16618


namespace NUMINAMATH_GPT_exponent_division_l166_16660

variable (a : ℝ) (m n : ℝ)
-- Conditions
def condition1 : Prop := a^m = 2
def condition2 : Prop := a^n = 16

-- Theorem Statement
theorem exponent_division (h1 : condition1 a m) (h2 : condition2 a n) : a^(m - n) = 1 / 8 := by
  sorry

end NUMINAMATH_GPT_exponent_division_l166_16660


namespace NUMINAMATH_GPT_rectangular_region_area_l166_16633

theorem rectangular_region_area :
  ∀ (s : ℝ), 18 * s * s = (15 * Real.sqrt 2) * (7.5 * Real.sqrt 2) :=
by
  intro s
  have h := 5 ^ 2 = 2 * s ^ 2
  have s := Real.sqrt (25 / 2)
  exact sorry

end NUMINAMATH_GPT_rectangular_region_area_l166_16633


namespace NUMINAMATH_GPT_probability_of_choosing_perfect_square_is_0_08_l166_16655

-- Definitions for the conditions
def n : ℕ := 100
def p : ℚ := 1 / 200
def probability (m : ℕ) : ℚ := if m ≤ 50 then p else 3 * p
def perfect_squares_before_50 : Finset ℕ := {1, 4, 9, 16, 25, 36, 49}
def perfect_squares_between_51_and_100 : Finset ℕ := {64, 81, 100}
def total_perfect_squares : Finset ℕ := perfect_squares_before_50 ∪ perfect_squares_between_51_and_100

-- Statement to prove that the probability of selecting a perfect square is 0.08
theorem probability_of_choosing_perfect_square_is_0_08 :
  (perfect_squares_before_50.card * p + perfect_squares_between_51_and_100.card * 3 * p) = 0.08 := 
by
  -- Adding sorry to skip the proof
  sorry

end NUMINAMATH_GPT_probability_of_choosing_perfect_square_is_0_08_l166_16655


namespace NUMINAMATH_GPT_find_x_for_parallel_l166_16663

-- Definitions for vector components and parallel condition.
def a : ℝ × ℝ := (1, 2)
def b (x : ℝ) : ℝ × ℝ := (x, -3)

def parallel (v w : ℝ × ℝ) : Prop := v.1 * w.2 = v.2 * w.1

theorem find_x_for_parallel :
  ∃ x : ℝ, parallel a (b x) ∧ x = -3 / 2 :=
by
  -- The statement to be proven
  sorry

end NUMINAMATH_GPT_find_x_for_parallel_l166_16663


namespace NUMINAMATH_GPT_upper_bound_expression_4n_plus_7_l166_16699

theorem upper_bound_expression_4n_plus_7 (U : ℤ) :
  (∃ (n : ℕ),  4 * n + 7 > 1) ∧
  (∀ (n : ℕ), 4 * n + 7 < U → ∃ (k : ℕ), k ≤ 19 ∧ k = n) ∧
  (∃ (n_min n_max : ℕ), n_max = n_min + 19 ∧ 4 * n_max + 7 < U) →
  U = 84 := sorry

end NUMINAMATH_GPT_upper_bound_expression_4n_plus_7_l166_16699


namespace NUMINAMATH_GPT_projection_matrix_ratio_l166_16686

theorem projection_matrix_ratio
  (x y : ℚ)
  (h1 : (4/29) * x - (10/29) * y = x)
  (h2 : -(10/29) * x + (25/29) * y = y) :
  y / x = -5/2 :=
by
  sorry

end NUMINAMATH_GPT_projection_matrix_ratio_l166_16686


namespace NUMINAMATH_GPT_first_tap_fill_time_l166_16605

theorem first_tap_fill_time (T : ℝ) (h1 : T > 0) (h2 : 12 > 0) 
  (h3 : 1/T - 1/12 = 1/12) : T = 6 :=
sorry

end NUMINAMATH_GPT_first_tap_fill_time_l166_16605


namespace NUMINAMATH_GPT_sum_to_fraction_l166_16631

theorem sum_to_fraction :
  (2 / 10) + (3 / 100) + (4 / 1000) + (6 / 10000) + (7 / 100000) = 23467 / 100000 :=
by
  sorry

end NUMINAMATH_GPT_sum_to_fraction_l166_16631


namespace NUMINAMATH_GPT_part1_part2_l166_16647

variables {R : Type} [LinearOrderedField R]

def setA := {x : R | -1 < x ∧ x ≤ 5}
def setB (m : R) := {x : R | x^2 - 2*x - m < 0}
def complementB (m : R) := {x : R | x ≤ -1 ∨ x ≥ 3}

theorem part1 : 
  {x : R | 6 / (x + 1) ≥ 1} = setA := 
by 
  sorry

theorem part2 (m : R) (hm : m = 3) : 
  setA ∩ complementB m = {x : R | 3 ≤ x ∧ x ≤ 5} := 
by 
  sorry

end NUMINAMATH_GPT_part1_part2_l166_16647


namespace NUMINAMATH_GPT_average_of_first_40_results_l166_16684

theorem average_of_first_40_results 
  (A : ℝ)
  (avg_other_30 : ℝ := 40)
  (avg_all_70 : ℝ := 34.285714285714285) : A = 30 :=
by 
  let sum1 := A * 40
  let sum2 := avg_other_30 * 30
  let combined_sum := sum1 + sum2
  let combined_avg := combined_sum / 70
  have h1 : combined_avg = avg_all_70 := by sorry
  have h2 : combined_avg = 34.285714285714285 := by sorry
  have h3 : combined_sum = (A * 40) + (40 * 30) := by sorry
  have h4 : (A * 40) + 1200 = 2400 := by sorry
  have h5 : A * 40 = 1200 := by sorry
  have h6 : A = 1200 / 40 := by sorry
  have h7 : A = 30 := by sorry
  exact h7

end NUMINAMATH_GPT_average_of_first_40_results_l166_16684


namespace NUMINAMATH_GPT_total_money_made_from_jerseys_l166_16615

def price_per_jersey : ℕ := 76
def jerseys_sold : ℕ := 2

theorem total_money_made_from_jerseys : price_per_jersey * jerseys_sold = 152 := 
by
  -- The actual proof steps will go here
  sorry

end NUMINAMATH_GPT_total_money_made_from_jerseys_l166_16615


namespace NUMINAMATH_GPT_sum_of_integers_with_product_5_pow_4_l166_16690

theorem sum_of_integers_with_product_5_pow_4 :
  ∃ (a b c d : ℕ), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
  a * b * c * d = 5^4 ∧
  a + b + c + d = 156 :=
by sorry

end NUMINAMATH_GPT_sum_of_integers_with_product_5_pow_4_l166_16690


namespace NUMINAMATH_GPT_max_product_price_l166_16693

/-- Conditions: 
1. Company C sells 50 products.
2. The average retail price of the products is $2,500.
3. No product sells for less than $800.
4. Exactly 20 products sell for less than $2,000.
Goal:
Prove that the greatest possible selling price of the most expensive product is $51,000.
-/
theorem max_product_price (n : ℕ) (avg_price : ℝ) (min_price : ℝ) (threshold_price : ℝ) (num_below_threshold : ℕ) :
  n = 50 → 
  avg_price = 2500 → 
  min_price = 800 → 
  threshold_price = 2000 → 
  num_below_threshold = 20 → 
  ∃ max_price : ℝ, max_price = 51000 :=
by 
  sorry

end NUMINAMATH_GPT_max_product_price_l166_16693


namespace NUMINAMATH_GPT_sum_of_solutions_eq_zero_l166_16653

theorem sum_of_solutions_eq_zero :
  let p := 6
  let q := 150
  (∃ x1 x2 : ℝ, p * x1 = q / x1 ∧ p * x2 = q / x2 ∧ x1 ≠ x2 ∧ x1 + x2 = 0) :=
sorry

end NUMINAMATH_GPT_sum_of_solutions_eq_zero_l166_16653


namespace NUMINAMATH_GPT_sample_size_proof_l166_16659

theorem sample_size_proof (p : ℝ) (N : ℤ) (n : ℤ) (h1 : N = 200) (h2 : p = 0.25) : n = 50 :=
by
  sorry

end NUMINAMATH_GPT_sample_size_proof_l166_16659


namespace NUMINAMATH_GPT_inequality_no_solution_iff_a_le_neg3_l166_16613

theorem inequality_no_solution_iff_a_le_neg3 (a : ℝ) :
  (∀ x : ℝ, ¬ (|x - 1| - |x + 2| < a)) ↔ a ≤ -3 := 
sorry

end NUMINAMATH_GPT_inequality_no_solution_iff_a_le_neg3_l166_16613


namespace NUMINAMATH_GPT_jason_optimal_reroll_probability_l166_16620

-- Define the probability function based on the three dice roll problem
def probability_of_rerolling_two_dice : ℚ := 
  -- As per the problem, the computed and fixed probability is 7/64.
  7 / 64

-- Prove that Jason's optimal strategy leads to rerolling exactly two dice with a probability of 7/64.
theorem jason_optimal_reroll_probability : probability_of_rerolling_two_dice = 7 / 64 := 
  sorry

end NUMINAMATH_GPT_jason_optimal_reroll_probability_l166_16620


namespace NUMINAMATH_GPT_proof_f_1_add_g_2_l166_16658

def f (x : ℤ) : ℤ := 3 * x - 4
def g (x : ℤ) : ℤ := x + 1

theorem proof_f_1_add_g_2 : f (1 + g 2) = 8 := by
  sorry

end NUMINAMATH_GPT_proof_f_1_add_g_2_l166_16658


namespace NUMINAMATH_GPT_solve_for_nabla_l166_16676

theorem solve_for_nabla (nabla mu : ℤ) (h1 : 5 * (-3) = nabla + mu - 3) (h2 : mu = 4) : 
  nabla = -16 := 
by
  sorry

end NUMINAMATH_GPT_solve_for_nabla_l166_16676


namespace NUMINAMATH_GPT_simplify_and_evaluate_l166_16608

theorem simplify_and_evaluate (a : ℝ) (h₁ : a^2 - 4 * a + 3 = 0) (h₂ : a ≠ 3) : 
  ( (a^2 - 9) / (a^2 - 3 * a) / ( (a^2 + 9) / a + 6 ) = 1 / 4 ) :=
by 
  sorry

end NUMINAMATH_GPT_simplify_and_evaluate_l166_16608


namespace NUMINAMATH_GPT_asian_population_percentage_in_west_is_57_l166_16688

variable (NE MW South West : ℕ)

def total_asian_population (NE MW South West : ℕ) : ℕ :=
  NE + MW + South + West

def west_asian_population_percentage
  (NE MW South West : ℕ) (total_asian_population : ℕ) : ℚ :=
  (West : ℚ) / (total_asian_population : ℚ) * 100

theorem asian_population_percentage_in_west_is_57 :
  total_asian_population 2 3 4 12 = 21 →
  west_asian_population_percentage 2 3 4 12 21 = 57 :=
by
  intros
  sorry

end NUMINAMATH_GPT_asian_population_percentage_in_west_is_57_l166_16688


namespace NUMINAMATH_GPT_arithmetic_expression_l166_16670

theorem arithmetic_expression :
  10 + 4 * (5 + 3)^3 = 2058 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_expression_l166_16670


namespace NUMINAMATH_GPT_integer_part_M_is_4_l166_16632

-- Define the variables and conditions based on the problem statement
variable (a b c : ℝ)

-- This non-computable definition includes the main mathematical expression we need to evaluate
noncomputable def M (a b c : ℝ) := Real.sqrt (3 * a + 1) + Real.sqrt (3 * b + 1) + Real.sqrt (3 * c + 1)

-- The theorem we need to prove
theorem integer_part_M_is_4 (h₁ : a + b + c = 1) (h₂ : 0 < a) (h₃ : 0 < b) (h₄ : 0 < c) : 
  ⌊M a b c⌋ = 4 := 
by 
  sorry

end NUMINAMATH_GPT_integer_part_M_is_4_l166_16632


namespace NUMINAMATH_GPT_remaining_laps_l166_16698

def track_length : ℕ := 9
def initial_laps : ℕ := 6
def total_distance : ℕ := 99

theorem remaining_laps : (total_distance - (initial_laps * track_length)) / track_length = 5 := by
  sorry

end NUMINAMATH_GPT_remaining_laps_l166_16698


namespace NUMINAMATH_GPT_point_coordinates_l166_16681

theorem point_coordinates (m : ℝ) 
  (h1 : dist (0 : ℝ) (Real.sqrt m) = 4) : 
  (-m, Real.sqrt m) = (-16, 4) := 
by
  -- The proof will use the conditions and solve for m to find the coordinates
  sorry

end NUMINAMATH_GPT_point_coordinates_l166_16681


namespace NUMINAMATH_GPT_geometric_probability_l166_16697

theorem geometric_probability :
  let total_length := (1/2 - 0)
  let favorable_length := (1/3 - 0)
  total_length > 0 ∧ favorable_length > 0 →
  (favorable_length / total_length) = 2 / 3 :=
by
  intros _ _ _
  sorry

end NUMINAMATH_GPT_geometric_probability_l166_16697


namespace NUMINAMATH_GPT_correct_option_C_l166_16671

noncomputable def f (x : ℝ) : ℝ := (2^x - 1) / (2^x + 1)

theorem correct_option_C : ∀ (x1 x2 : ℝ), 0 < x1 → x1 < x2 → x1 * f x1 < x2 * f x2 :=
by
  intro x1 x2 hx1 hx12
  sorry

end NUMINAMATH_GPT_correct_option_C_l166_16671


namespace NUMINAMATH_GPT_random_events_l166_16621

/-- Definition of what constitutes a random event --/
def is_random_event (e : String) : Prop :=
  e = "Drawing 3 first-quality glasses out of 10 glasses (8 first-quality, 2 substandard)" ∨
  e = "Forgetting the last digit of a phone number, randomly pressing and it is correct" ∨
  e = "Winning the first prize in a sports lottery"

/-- Define the specific events --/
def event_1 := "Drawing 3 first-quality glasses out of 10 glasses (8 first-quality, 2 substandard)"
def event_2 := "Forgetting the last digit of a phone number, randomly pressing and it is correct"
def event_3 := "Opposite electric charges attract each other"
def event_4 := "Winning the first prize in a sports lottery"

/-- Lean 4 statement for the proof problem --/
theorem random_events :
  (is_random_event event_1) ∧
  (is_random_event event_2) ∧
  ¬(is_random_event event_3) ∧
  (is_random_event event_4) :=
by 
  sorry

end NUMINAMATH_GPT_random_events_l166_16621


namespace NUMINAMATH_GPT_word_sum_problems_l166_16675

theorem word_sum_problems (J M O I : Fin 10) (h_distinct : J ≠ M ∧ J ≠ O ∧ J ≠ I ∧ M ≠ O ∧ M ≠ I ∧ O ≠ I) 
  (h_nonzero_J : J ≠ 0) (h_nonzero_I : I ≠ 0) :
  let JMO := 100 * J + 10 * M + O
  let IMO := 100 * I + 10 * M + O
  (JMO + JMO + JMO = IMO) → 
  (JMO = 150 ∧ IMO = 450) ∨ (JMO = 250 ∧ IMO = 750) :=
sorry

end NUMINAMATH_GPT_word_sum_problems_l166_16675
