import Mathlib

namespace NUMINAMATH_GPT_sum_of_distinct_prime_divisors_1728_l1317_131752

theorem sum_of_distinct_prime_divisors_1728 : 
  (2 + 3 = 5) :=
sorry

end NUMINAMATH_GPT_sum_of_distinct_prime_divisors_1728_l1317_131752


namespace NUMINAMATH_GPT_athlete_stable_performance_l1317_131781

theorem athlete_stable_performance 
  (A_var : ℝ) (B_var : ℝ) (C_var : ℝ) (D_var : ℝ)
  (avg_score : ℝ)
  (hA_var : A_var = 0.019)
  (hB_var : B_var = 0.021)
  (hC_var : C_var = 0.020)
  (hD_var : D_var = 0.022)
  (havg : avg_score = 13.2) :
  A_var < B_var ∧ A_var < C_var ∧ A_var < D_var :=
by {
  sorry
}

end NUMINAMATH_GPT_athlete_stable_performance_l1317_131781


namespace NUMINAMATH_GPT_fraction_addition_l1317_131791

theorem fraction_addition (a b c d : ℚ) (h1 : a = 3/4) (h2 : b = 5/9) : a + b = 47/36 :=
by
  rw [h1, h2]
  sorry

end NUMINAMATH_GPT_fraction_addition_l1317_131791


namespace NUMINAMATH_GPT_sqrt_12_times_sqrt_27_div_sqrt_3_eq_6_sqrt_3_l1317_131735

noncomputable def calculation (x y z : ℝ) : ℝ :=
  (Real.sqrt x * Real.sqrt y) / Real.sqrt z

theorem sqrt_12_times_sqrt_27_div_sqrt_3_eq_6_sqrt_3 :
  calculation 12 27 3 = 6 * Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_sqrt_12_times_sqrt_27_div_sqrt_3_eq_6_sqrt_3_l1317_131735


namespace NUMINAMATH_GPT_problem_T8_l1317_131724

noncomputable def a : Nat → ℚ
| 0     => 1/2
| (n+1) => a n / (1 + 3 * a n)

noncomputable def T (n : Nat) : ℚ :=
  (Finset.range n).sum (λ i => 1 / a (i + 1))

theorem problem_T8 : T 8 = 100 :=
sorry

end NUMINAMATH_GPT_problem_T8_l1317_131724


namespace NUMINAMATH_GPT_truck_sand_at_arrival_l1317_131740

-- Definitions based on conditions in part a)
def initial_sand : ℝ := 4.1
def lost_sand : ℝ := 2.4

-- Theorem statement corresponding to part c)
theorem truck_sand_at_arrival : initial_sand - lost_sand = 1.7 :=
by
  -- "sorry" placeholder to skip the proof
  sorry

end NUMINAMATH_GPT_truck_sand_at_arrival_l1317_131740


namespace NUMINAMATH_GPT_questionnaire_visitors_l1317_131776

theorem questionnaire_visitors
  (V : ℕ)
  (E U : ℕ)
  (h1 : ∀ v : ℕ, v ∈ { x : ℕ | x ≠ E ∧ x ≠ U } → v = 110)
  (h2 : E = U)
  (h3 : 3 * V = 4 * (E + U - 110))
  : V = 440 :=
by
  sorry

end NUMINAMATH_GPT_questionnaire_visitors_l1317_131776


namespace NUMINAMATH_GPT_find_complex_number_l1317_131779

open Complex

theorem find_complex_number (z : ℂ) (h : z * (1 - I) = 2) : z = 1 + I :=
sorry

end NUMINAMATH_GPT_find_complex_number_l1317_131779


namespace NUMINAMATH_GPT_find_A_l1317_131704

def A : ℕ := 7 * 5 + 3

theorem find_A : A = 38 :=
by
  sorry

end NUMINAMATH_GPT_find_A_l1317_131704


namespace NUMINAMATH_GPT_number_of_tiles_per_row_l1317_131738

theorem number_of_tiles_per_row : 
  ∀ (side_length_in_feet room_area_in_sqft : ℕ) (tile_width_in_inches : ℕ), 
  room_area_in_sqft = 256 → tile_width_in_inches = 8 → 
  side_length_in_feet * side_length_in_feet = room_area_in_sqft → 
  12 * side_length_in_feet / tile_width_in_inches = 24 := 
by
  intros side_length_in_feet room_area_in_sqft tile_width_in_inches h_area h_tile_width h_side_length
  sorry

end NUMINAMATH_GPT_number_of_tiles_per_row_l1317_131738


namespace NUMINAMATH_GPT_problem_statement_l1317_131742

noncomputable def U : Set ℝ := {x | -3 ≤ x ∧ x ≤ 3}
noncomputable def M : Set ℝ := {x | -1 < x ∧ x < 1}
noncomputable def N : Set ℝ := {x | 0 < x ∧ x < 2}
noncomputable def complement_U (s : Set ℝ) : Set ℝ := {x | x ∈ U ∧ x ∉ s}
noncomputable def intersection (A B : Set ℝ) : Set ℝ := {x | x ∈ A ∧ x ∈ B}

theorem problem_statement : intersection N (complement_U M) = {x | 1 ≤ x ∧ x < 2} :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l1317_131742


namespace NUMINAMATH_GPT_elevator_people_count_l1317_131767

theorem elevator_people_count (weight_limit : ℕ) (excess_weight : ℕ) (avg_weight : ℕ) (total_weight : ℕ) (n : ℕ) 
  (h1 : weight_limit = 1500)
  (h2 : excess_weight = 100)
  (h3 : avg_weight = 80)
  (h4 : total_weight = weight_limit + excess_weight)
  (h5 : total_weight = n * avg_weight) :
  n = 20 :=
sorry

end NUMINAMATH_GPT_elevator_people_count_l1317_131767


namespace NUMINAMATH_GPT_correct_option_l1317_131786

-- Define the four conditions as propositions
def option_A (a b : ℝ) : Prop := (a + b) ^ 2 = a ^ 2 + b ^ 2
def option_B (a : ℝ) : Prop := 2 * a ^ 2 + a = 3 * a ^ 3
def option_C (a : ℝ) : Prop := a ^ 3 * a ^ 2 = a ^ 5
def option_D (a : ℝ) (h : a ≠ 0) : Prop := 2 * a⁻¹ = 1 / (2 * a)

-- Prove which operation is the correct one
theorem correct_option (a b : ℝ) (h : a ≠ 0) : option_C a :=
by {
  -- Placeholder for actual proofs, each option needs to be verified
  sorry
}

end NUMINAMATH_GPT_correct_option_l1317_131786


namespace NUMINAMATH_GPT_ratio_cube_sphere_surface_area_l1317_131710

theorem ratio_cube_sphere_surface_area (R : ℝ) (h1 : R > 0) :
  let Scube := 24 * R^2
  let Ssphere := 4 * Real.pi * R^2
  (Scube / Ssphere) = (6 / Real.pi) :=
by
  sorry

end NUMINAMATH_GPT_ratio_cube_sphere_surface_area_l1317_131710


namespace NUMINAMATH_GPT_ana_bonita_age_gap_l1317_131734

theorem ana_bonita_age_gap (A B n : ℚ) (h1 : A = 2 * B + 3) (h2 : A - 2 = 6 * (B - 2)) (h3 : A = B + n) : n = 6.25 :=
by
  sorry

end NUMINAMATH_GPT_ana_bonita_age_gap_l1317_131734


namespace NUMINAMATH_GPT_find_a_l1317_131729

theorem find_a (a x : ℝ)
    (h1 : 6 * (x + 8) = 18 * x)
    (h2 : 6 * x - 2 * (a - x) = 2 * a + x) :
    a = 7 :=
  sorry

end NUMINAMATH_GPT_find_a_l1317_131729


namespace NUMINAMATH_GPT_jade_and_julia_total_money_l1317_131718

theorem jade_and_julia_total_money (x : ℕ) : 
  let jade_initial := 38 
  let julia_initial := jade_initial / 2 
  let jade_after := jade_initial + x 
  let julia_after := julia_initial + x 
  jade_after + julia_after = 57 + 2 * x := by
  sorry

end NUMINAMATH_GPT_jade_and_julia_total_money_l1317_131718


namespace NUMINAMATH_GPT_James_age_is_11_l1317_131798

-- Define the ages of Julio and James.
def Julio_age := 36

-- The age condition in 14 years.
def Julio_age_in_14_years := Julio_age + 14

-- James' age in 14 years and the relation as per the condition.
def James_age_in_14_years (J : ℕ) := J + 14

-- The main proof statement.
theorem James_age_is_11 (J : ℕ) 
  (h1 : Julio_age_in_14_years = 2 * James_age_in_14_years J) : J = 11 :=
by
  sorry

end NUMINAMATH_GPT_James_age_is_11_l1317_131798


namespace NUMINAMATH_GPT_time_descend_hill_l1317_131726

-- Definitions
def time_to_top : ℝ := 4
def avg_speed_whole_journey : ℝ := 3
def avg_speed_uphill : ℝ := 2.25

-- Theorem statement
theorem time_descend_hill (t : ℝ) 
  (h1 : time_to_top = 4) 
  (h2 : avg_speed_whole_journey = 3) 
  (h3 : avg_speed_uphill = 2.25) : 
  t = 2 := 
sorry

end NUMINAMATH_GPT_time_descend_hill_l1317_131726


namespace NUMINAMATH_GPT_petals_per_rose_l1317_131785

theorem petals_per_rose
    (roses_per_bush : ℕ)
    (bushes : ℕ)
    (bottles : ℕ)
    (oz_per_bottle : ℕ)
    (petals_per_oz : ℕ)
    (petals : ℕ)
    (ounces : ℕ := bottles * oz_per_bottle)
    (total_petals : ℕ := ounces * petals_per_oz)
    (petals_per_bush : ℕ := total_petals / bushes)
    (petals_per_rose : ℕ := petals_per_bush / roses_per_bush) :
    petals_per_oz = 320 →
    roses_per_bush = 12 →
    bushes = 800 →
    bottles = 20 →
    oz_per_bottle = 12 →
    petals_per_rose = 8 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_petals_per_rose_l1317_131785


namespace NUMINAMATH_GPT_sale_in_fifth_month_l1317_131749

-- Define the sales for the first four months and the required sale for the sixth month
def sale_month1 : ℕ := 5124
def sale_month2 : ℕ := 5366
def sale_month3 : ℕ := 5808
def sale_month4 : ℕ := 5399
def sale_month6 : ℕ := 4579

-- Define the target average sale and number of months
def target_average_sale : ℕ := 5400
def number_of_months : ℕ := 6

-- Define the total sales calculation using the provided information
def total_sales : ℕ := target_average_sale * number_of_months
def total_sales_first_four_months : ℕ := sale_month1 + sale_month2 + sale_month3 + sale_month4

-- Prove the sale in the fifth month
theorem sale_in_fifth_month : 
  sale_month1 + sale_month2 + sale_month3 + sale_month4 + (total_sales - 
  (total_sales_first_four_months + sale_month6)) + sale_month6 = total_sales :=
by
  sorry

end NUMINAMATH_GPT_sale_in_fifth_month_l1317_131749


namespace NUMINAMATH_GPT_value_of_x_pow_12_l1317_131778

theorem value_of_x_pow_12 (x : ℝ) (hx : x + 1/x = Real.sqrt 5) : x^12 = 439 := sorry

end NUMINAMATH_GPT_value_of_x_pow_12_l1317_131778


namespace NUMINAMATH_GPT_longer_strap_length_l1317_131794

theorem longer_strap_length (S L : ℕ) 
  (h1 : L = S + 72) 
  (h2 : S + L = 348) : 
  L = 210 := 
sorry

end NUMINAMATH_GPT_longer_strap_length_l1317_131794


namespace NUMINAMATH_GPT_negation_proposition_l1317_131753

theorem negation_proposition : 
  (¬ (∀ x : ℝ, x^3 - x^2 + 1 ≤ 0)) ↔ (∃ x : ℝ, x^3 - x^2 + 1 > 0) :=
by
  sorry

end NUMINAMATH_GPT_negation_proposition_l1317_131753


namespace NUMINAMATH_GPT_polygon_sides_l1317_131732

theorem polygon_sides (perimeter side_length : ℕ) (h₁ : perimeter = 150) (h₂ : side_length = 15): 
  (perimeter / side_length) = 10 := 
by
  -- Here goes the proof part
  sorry

end NUMINAMATH_GPT_polygon_sides_l1317_131732


namespace NUMINAMATH_GPT_f_at_2008_l1317_131702

noncomputable def f : ℝ → ℝ := sorry
noncomputable def finv : ℝ → ℝ := sorry

axiom f_inverse : ∀ x, f (finv x) = x ∧ finv (f x) = x
axiom f_at_9 : f 9 = 18

theorem f_at_2008 : f 2008 = -1981 :=
by
  sorry

end NUMINAMATH_GPT_f_at_2008_l1317_131702


namespace NUMINAMATH_GPT_not_rectangle_determined_by_angle_and_side_l1317_131799

axiom parallelogram_determined_by_two_sides_and_angle : Prop
axiom equilateral_triangle_determined_by_area : Prop
axiom square_determined_by_perimeter_and_side : Prop
axiom rectangle_determined_by_two_diagonals : Prop
axiom rectangle_determined_by_angle_and_side : Prop

theorem not_rectangle_determined_by_angle_and_side : ¬rectangle_determined_by_angle_and_side := 
sorry

end NUMINAMATH_GPT_not_rectangle_determined_by_angle_and_side_l1317_131799


namespace NUMINAMATH_GPT_range_of_a_l1317_131771

variable (A B : Set ℝ)
variable (a : ℝ)

def setA : Set ℝ := {x | x < -1 ∨ x ≥ 1}
def setB (a : ℝ) : Set ℝ := {x | x ≤ 2 * a ∨ x ≥ a + 1}

theorem range_of_a (a : ℝ) :
  (compl (setB a) ⊆ setA) ↔ (a ≤ -2 ∨ (1 / 2 ≤ a ∧ a < 1)) :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1317_131771


namespace NUMINAMATH_GPT_simplify_expression_l1317_131746

variable (b c : ℝ)

theorem simplify_expression :
  (1 : ℝ) * (-2 * b) * (3 * b^2) * (-4 * c^3) * (5 * c^4) = -120 * b^3 * c^7 :=
by sorry

end NUMINAMATH_GPT_simplify_expression_l1317_131746


namespace NUMINAMATH_GPT_minimum_value_ge_100_minimum_value_eq_100_l1317_131744

noncomputable def minimum_value_expression (α β : ℝ) : ℝ :=
  (3 * Real.cos α + 4 * Real.sin β - 7)^2 + (3 * Real.sin α + 4 * Real.cos β - 10)^2

theorem minimum_value_ge_100 (α β : ℝ) : minimum_value_expression α β ≥ 100 :=
  sorry

theorem minimum_value_eq_100 (α β : ℝ)
  (hα : 3 * Real.cos α + 4 * Real.sin β = 7)
  (hβ : 3 * Real.sin α + 4 * Real.cos β = 10) :
  minimum_value_expression α β = 100 :=
  sorry

end NUMINAMATH_GPT_minimum_value_ge_100_minimum_value_eq_100_l1317_131744


namespace NUMINAMATH_GPT_no_solution_fraction_eq_l1317_131758

theorem no_solution_fraction_eq (x : ℝ) : 
  (1 / (x - 2) = (1 - x) / (2 - x) - 3) → False := 
by 
  sorry

end NUMINAMATH_GPT_no_solution_fraction_eq_l1317_131758


namespace NUMINAMATH_GPT_intersection_of_planes_is_line_l1317_131755

-- Define the conditions as Lean 4 statements
def plane1 (x y z : ℝ) : Prop := 2 * x + 3 * y + z - 8 = 0
def plane2 (x y z : ℝ) : Prop := x - 2 * y - 2 * z + 1 = 0

-- Define the canonical form of the line as a Lean 4 proposition
def canonical_line (x y z : ℝ) : Prop := 
  (x - 3) / -4 = y / 5 ∧ y / 5 = (z - 2) / -7

-- The theorem to state equivalence between conditions and canonical line equations
theorem intersection_of_planes_is_line :
  ∀ (x y z : ℝ), plane1 x y z → plane2 x y z → canonical_line x y z :=
by
  intros x y z h1 h2
  -- TODO: Insert proof here
  sorry

end NUMINAMATH_GPT_intersection_of_planes_is_line_l1317_131755


namespace NUMINAMATH_GPT_problem_statement_l1317_131772

variable {Point Line Plane : Type}

-- Definitions for perpendicular and parallel
def perpendicular (l1 l2 : Line) : Prop := sorry
def parallel (l1 l2 : Line) : Prop := sorry
def perp_to_plane (l : Line) (p : Plane) : Prop := sorry

-- Given variables
variable (a b c d : Line) (α β : Plane)

-- Conditions
axiom a_perp_b : perpendicular a b
axiom c_perp_d : perpendicular c d
axiom a_perp_alpha : perp_to_plane a α
axiom c_perp_alpha : perp_to_plane c α

-- Required proof
theorem problem_statement : perpendicular c b :=
by sorry

end NUMINAMATH_GPT_problem_statement_l1317_131772


namespace NUMINAMATH_GPT_g_at_10_l1317_131780

-- Definitions and conditions
def f : ℝ → ℝ := sorry
axiom f_at_1 : f 1 = 10
axiom f_inequality_1 : ∀ x : ℝ, f (x + 20) ≥ f x + 20
axiom f_inequality_2 : ∀ x : ℝ, f (x + 1) ≤ f x + 1
def g (x : ℝ) : ℝ := f x - x + 1

-- Proof statement (no proof required)
theorem g_at_10 : g 10 = 10 := sorry

end NUMINAMATH_GPT_g_at_10_l1317_131780


namespace NUMINAMATH_GPT_base8_to_base10_4532_l1317_131743

theorem base8_to_base10_4532 : 
    (4 * 8^3 + 5 * 8^2 + 3 * 8^1 + 2 * 8^0) = 2394 := 
by sorry

end NUMINAMATH_GPT_base8_to_base10_4532_l1317_131743


namespace NUMINAMATH_GPT_walnut_trees_currently_in_park_l1317_131720

-- Definitions from the conditions
def total_trees : ℕ := 77
def trees_to_be_planted : ℕ := 44

-- Statement to prove: number of current trees = 33
theorem walnut_trees_currently_in_park : total_trees - trees_to_be_planted = 33 :=
by
  sorry

end NUMINAMATH_GPT_walnut_trees_currently_in_park_l1317_131720


namespace NUMINAMATH_GPT_arithmetic_sequence_8th_term_l1317_131788

theorem arithmetic_sequence_8th_term :
  ∃ (a1 a15 n : ℕ) (d a8 : ℝ),
  a1 = 3 ∧ a15 = 48 ∧ n = 15 ∧
  d = (a15 - a1) / (n - 1) ∧
  a8 = a1 + 7 * d ∧
  a8 = 25.5 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_8th_term_l1317_131788


namespace NUMINAMATH_GPT_length_of_bridge_l1317_131756

-- Define the conditions
def length_of_train : ℝ := 750
def speed_of_train_kmh : ℝ := 120
def crossing_time : ℝ := 45
def wind_resistance_factor : ℝ := 0.10

-- Define the conversion from km/hr to m/s
def kmh_to_ms (v : ℝ) : ℝ := v * 0.27778

-- Define the actual speed considering wind resistance
def actual_speed_ms (v : ℝ) (resistance : ℝ) : ℝ := (kmh_to_ms v) * (1 - resistance)

-- Define the total distance covered
def total_distance (speed : ℝ) (time : ℝ) : ℝ := speed * time

-- Theorem: Length of the bridge
theorem length_of_bridge : total_distance (actual_speed_ms speed_of_train_kmh wind_resistance_factor) crossing_time - length_of_train = 600 := by
  sorry

end NUMINAMATH_GPT_length_of_bridge_l1317_131756


namespace NUMINAMATH_GPT_complex_number_simplification_l1317_131766

theorem complex_number_simplification (i : ℂ) (h : i^2 = -1) : i * (1 - i) - 1 = i := 
by
  sorry

end NUMINAMATH_GPT_complex_number_simplification_l1317_131766


namespace NUMINAMATH_GPT_complex_square_l1317_131716

theorem complex_square (z : ℂ) (i : ℂ) (h₁ : z = 5 - 3 * i) (h₂ : i * i = -1) : z^2 = 16 - 30 * i :=
by
  rw [h₁]
  sorry

end NUMINAMATH_GPT_complex_square_l1317_131716


namespace NUMINAMATH_GPT_original_ratio_of_flour_to_baking_soda_l1317_131728

-- Define the conditions
def sugar_to_flour_ratio_5_to_5 (sugar flour : ℕ) : Prop :=
  sugar = 2400 ∧ sugar = flour

def baking_soda_mass_condition (flour : ℕ) (baking_soda : ℕ) : Prop :=
  flour = 2400 ∧ (∃ b : ℕ, baking_soda = b ∧ flour / (b + 60) = 8)

-- The theorem statement we need to prove
theorem original_ratio_of_flour_to_baking_soda :
  ∃ flour baking_soda : ℕ,
  sugar_to_flour_ratio_5_to_5 2400 flour ∧
  baking_soda_mass_condition flour baking_soda →
  flour / baking_soda = 10 :=
by
  sorry

end NUMINAMATH_GPT_original_ratio_of_flour_to_baking_soda_l1317_131728


namespace NUMINAMATH_GPT_belle_biscuits_l1317_131762

-- Define the conditions
def cost_per_rawhide_bone : ℕ := 1
def num_rawhide_bones_per_evening : ℕ := 2
def cost_per_biscuit : ℚ := 0.25
def total_weekly_cost : ℚ := 21
def days_in_week : ℕ := 7

-- Define the number of biscuits Belle eats every evening
def num_biscuits_per_evening : ℚ := 4

-- Define the statement that encapsulates the problem
theorem belle_biscuits :
  (total_weekly_cost = days_in_week * (num_rawhide_bones_per_evening * cost_per_rawhide_bone + num_biscuits_per_evening * cost_per_biscuit)) :=
sorry

end NUMINAMATH_GPT_belle_biscuits_l1317_131762


namespace NUMINAMATH_GPT_quotient_of_division_l1317_131722

theorem quotient_of_division 
  (dividend divisor remainder : ℕ) 
  (h_dividend : dividend = 265) 
  (h_divisor : divisor = 22) 
  (h_remainder : remainder = 1) 
  (h_div : dividend = divisor * (dividend / divisor) + remainder) : 
  (dividend / divisor) = 12 := 
by
  sorry

end NUMINAMATH_GPT_quotient_of_division_l1317_131722


namespace NUMINAMATH_GPT_no_real_pairs_for_same_lines_l1317_131703

theorem no_real_pairs_for_same_lines : ¬ ∃ (a b : ℝ), (∀ x y : ℝ, 2 * x + a * y + b = 0 ↔ b * x - 3 * y + 15 = 0) :=
by {
  sorry
}

end NUMINAMATH_GPT_no_real_pairs_for_same_lines_l1317_131703


namespace NUMINAMATH_GPT_positive_integer_sixk_l1317_131717

theorem positive_integer_sixk (n : ℕ) :
  (∃ d1 d2 d3 : ℕ, d1 < d2 ∧ d2 < d3 ∧ d1 + d2 + d3 = n ∧ d1 ∣ n ∧ d2 ∣ n ∧ d3 ∣ n) ↔ (∃ k : ℕ, n = 6 * k) :=
by
  sorry

end NUMINAMATH_GPT_positive_integer_sixk_l1317_131717


namespace NUMINAMATH_GPT_angle_B_in_progression_l1317_131775

theorem angle_B_in_progression (A B C a b c : ℝ) (h1: A ≠ 0 ∧ B ≠ 0 ∧ C ≠ 0) 
(h2: B - A = C - B) (h3: b^2 - a^2 = a * c) (h4: A + B + C = Real.pi) : 
B = 2 * Real.pi / 7 := sorry

end NUMINAMATH_GPT_angle_B_in_progression_l1317_131775


namespace NUMINAMATH_GPT_problem_statement_l1317_131759

noncomputable def f (x : ℝ) (A : ℝ) (ϕ : ℝ) : ℝ := A * Real.cos (2 * x + ϕ)

theorem problem_statement {A ϕ : ℝ} (hA : A > 0) (hϕ : |ϕ| < π / 2)
  (h1 : f (-π / 4) A ϕ = 2 * Real.sqrt 2)
  (h2 : f 0 A ϕ = 2 * Real.sqrt 6)
  (h3 : f (π / 12) A ϕ = 2 * Real.sqrt 2)
  (h4 : f (π / 4) A ϕ = -2 * Real.sqrt 2)
  (h5 : f (π / 3) A ϕ = -2 * Real.sqrt 6) :
  ϕ = π / 6 ∧ f (5 * π / 12) A ϕ = -4 * Real.sqrt 2 := 
sorry

end NUMINAMATH_GPT_problem_statement_l1317_131759


namespace NUMINAMATH_GPT_determine_q_l1317_131784

theorem determine_q (p q : ℝ) (hp : p > 1) (hq : q > 1) (h1 : 1 / p + 1 / q = 1) (h2 : p * q = 4) : q = 2 := 
sorry

end NUMINAMATH_GPT_determine_q_l1317_131784


namespace NUMINAMATH_GPT_madeline_money_l1317_131737

variable (M B : ℝ)

theorem madeline_money :
  B = 1/2 * M →
  M + B = 72 →
  M = 48 :=
  by
    intros h1 h2
    sorry

end NUMINAMATH_GPT_madeline_money_l1317_131737


namespace NUMINAMATH_GPT_midpoint_trajectory_l1317_131796

theorem midpoint_trajectory (x y : ℝ) (x0 y0 : ℝ)
  (h_circle : x0^2 + y0^2 = 4)
  (h_tangent : x0 * x + y0 * y = 4)
  (h_x0 : x0 = 2 / x)
  (h_y0 : y0 = 2 / y) :
  x^2 * y^2 = x^2 + y^2 :=
sorry

end NUMINAMATH_GPT_midpoint_trajectory_l1317_131796


namespace NUMINAMATH_GPT_probability_A_correct_l1317_131701

-- Definitions of probabilities
variable (P_A P_B : Prop)
variable (P_AB : Prop := P_A ∧ P_B)
variable (prob_AB : ℝ := 2 / 3)
variable (prob_B_given_A : ℝ := 8 / 9)

-- Lean statement of the mathematical problem
theorem probability_A_correct :
  (P_AB → P_A ∧ P_B) →
  (prob_AB = (2 / 3)) →
  (prob_B_given_A = (2 / 3) / prob_A) →
  (∃ prob_A : ℝ, prob_A = 3 / 4) :=
by
  sorry

end NUMINAMATH_GPT_probability_A_correct_l1317_131701


namespace NUMINAMATH_GPT_henry_has_more_than_500_seeds_on_saturday_l1317_131787

theorem henry_has_more_than_500_seeds_on_saturday :
  (∃ k : ℕ, 5 * 3^k > 500 ∧ k + 1 = 6) :=
sorry

end NUMINAMATH_GPT_henry_has_more_than_500_seeds_on_saturday_l1317_131787


namespace NUMINAMATH_GPT_smallest_side_of_triangle_l1317_131777

theorem smallest_side_of_triangle (a b c : ℝ) (h : a^2 + b^2 > 5 * c^2) : 
  a > c ∧ b > c :=
by
  sorry

end NUMINAMATH_GPT_smallest_side_of_triangle_l1317_131777


namespace NUMINAMATH_GPT_geometric_sequence_common_ratio_l1317_131725

theorem geometric_sequence_common_ratio {a : ℕ+ → ℝ} (q : ℝ) (h_geom : ∀ n, a (n + 1) = q * a n) 
  (h_a3 : a 3 = 1) (h_a5 : a 5 = 4) : q = 2 ∨ q = -2 :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_common_ratio_l1317_131725


namespace NUMINAMATH_GPT_milk_cans_l1317_131736

theorem milk_cans (x y : ℕ) (h : 10 * x + 17 * y = 206) : x = 7 ∧ y = 8 := sorry

end NUMINAMATH_GPT_milk_cans_l1317_131736


namespace NUMINAMATH_GPT_find_X_l1317_131715

theorem find_X :
  let N := 90
  let X := (1 / 15) * N - (1 / 2 * 1 / 3 * 1 / 5 * N)
  X = 3 := by
  sorry

end NUMINAMATH_GPT_find_X_l1317_131715


namespace NUMINAMATH_GPT_constant_term_of_first_equation_l1317_131751

theorem constant_term_of_first_equation
  (y z : ℤ)
  (h1 : 2 * 20 - y - z = 40)
  (h2 : 3 * 20 + y - z = 20)
  (hx : 20 = 20) :
  4 * 20 + y + z = 80 := 
sorry

end NUMINAMATH_GPT_constant_term_of_first_equation_l1317_131751


namespace NUMINAMATH_GPT_abs_val_neg_three_l1317_131739

-- Definition section: stating the conditions
def abs_val (x : Int) : Int := if x < 0 then -x else x

-- Statement of the proof problem
theorem abs_val_neg_three : abs_val (-3) = 3 := by
  sorry

end NUMINAMATH_GPT_abs_val_neg_three_l1317_131739


namespace NUMINAMATH_GPT_original_difference_in_books_l1317_131730

theorem original_difference_in_books 
  (x y : ℕ) 
  (h1 : x + y = 5000) 
  (h2 : (1 / 2 : ℚ) * (x - 400) - (y + 400) = 400) : 
  x - y = 3000 := 
by 
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_original_difference_in_books_l1317_131730


namespace NUMINAMATH_GPT_num_comfortable_butterflies_final_state_l1317_131782

noncomputable def num_comfortable_butterflies (n : ℕ) : ℕ :=
  if h : 0 < n then
    n
  else
    0

theorem num_comfortable_butterflies_final_state {n : ℕ} (h : 0 < n):
  num_comfortable_butterflies n = n := by
  sorry

end NUMINAMATH_GPT_num_comfortable_butterflies_final_state_l1317_131782


namespace NUMINAMATH_GPT_part1_part2_l1317_131795

theorem part1 (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 1) (h4 : a^2 + 4 * b^2 + c^2 - 2 * c = 2) : 
  a + 2 * b + c ≤ 4 :=
sorry

theorem part2 (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 1) (h4 : a^2 + 4 * b^2 + c^2 - 2 * c = 2) (h5 : a = 2 * b) : 
  1 / b + 1 / (c - 1) ≥ 3 :=
sorry

end NUMINAMATH_GPT_part1_part2_l1317_131795


namespace NUMINAMATH_GPT_marching_band_total_weight_l1317_131748

def weight_trumpet : ℕ := 5
def weight_clarinet : ℕ := 5
def weight_trombone : ℕ := 10
def weight_tuba : ℕ := 20
def weight_drummer : ℕ := 15
def weight_percussionist : ℕ := 8

def uniform_trumpet : ℕ := 3
def uniform_clarinet : ℕ := 3
def uniform_trombone : ℕ := 4
def uniform_tuba : ℕ := 5
def uniform_drummer : ℕ := 6
def uniform_percussionist : ℕ := 3

def count_trumpet : ℕ := 6
def count_clarinet : ℕ := 9
def count_trombone : ℕ := 8
def count_tuba : ℕ := 3
def count_drummer : ℕ := 2
def count_percussionist : ℕ := 4

def total_weight_band : ℕ :=
  (count_trumpet * (weight_trumpet + uniform_trumpet)) +
  (count_clarinet * (weight_clarinet + uniform_clarinet)) +
  (count_trombone * (weight_trombone + uniform_trombone)) +
  (count_tuba * (weight_tuba + uniform_tuba)) +
  (count_drummer * (weight_drummer + uniform_drummer)) +
  (count_percussionist * (weight_percussionist + uniform_percussionist))

theorem marching_band_total_weight : total_weight_band = 393 :=
  by
  sorry

end NUMINAMATH_GPT_marching_band_total_weight_l1317_131748


namespace NUMINAMATH_GPT_inequality_proof_l1317_131768

variable (a b c : ℝ)

-- Conditions
def conditions : Prop := a > 0 ∧ b > 0 ∧ c > 0 ∧ a^2 + b^2 + c^2 = 14

-- Statement to prove
theorem inequality_proof (h : conditions a b c) : 
  a^5 + (1/8) * b^5 + (1/27) * c^5 ≥ 14 := 
sorry

end NUMINAMATH_GPT_inequality_proof_l1317_131768


namespace NUMINAMATH_GPT_sophomore_spaghetti_tortellini_ratio_l1317_131706

theorem sophomore_spaghetti_tortellini_ratio
    (total_students : ℕ)
    (spaghetti_lovers : ℕ)
    (tortellini_lovers : ℕ)
    (grade_levels : ℕ)
    (spaghetti_sophomores : ℕ)
    (tortellini_sophomores : ℕ)
    (h1 : total_students = 800)
    (h2 : spaghetti_lovers = 300)
    (h3 : tortellini_lovers = 120)
    (h4 : grade_levels = 4)
    (h5 : spaghetti_sophomores = spaghetti_lovers / grade_levels)
    (h6 : tortellini_sophomores = tortellini_lovers / grade_levels) :
    (spaghetti_sophomores : ℚ) / (tortellini_sophomores : ℚ) = 5 / 2 := by
  sorry

end NUMINAMATH_GPT_sophomore_spaghetti_tortellini_ratio_l1317_131706


namespace NUMINAMATH_GPT_exponent_property_l1317_131714

theorem exponent_property :
  4^4 * 9^4 * 4^9 * 9^9 = 36^13 :=
by
  -- Add the proof here
  sorry

end NUMINAMATH_GPT_exponent_property_l1317_131714


namespace NUMINAMATH_GPT_volume_of_pyramid_l1317_131792

-- Definitions based on conditions
def regular_quadrilateral_pyramid (h r : ℝ) := 
  ∃ a : ℝ, ∃ S : ℝ, ∃ V : ℝ,
  a = 2 * h * ((h^2 - r^2) / r^2).sqrt ∧
  S = (2 * h * ((h^2 - r^2) / r^2).sqrt)^2 ∧
  V = (4 * h^5 - 4 * h^3 * r^2) / (3 * r^2)

-- Lean 4 theorem statement
theorem volume_of_pyramid (h r : ℝ) (h_pos : h > 0) (r_pos : r > 0) : 
  ∃ V : ℝ, V = (4 * h^5 - 4 * h^3 * r^2) / (3 * r^2) :=
sorry

end NUMINAMATH_GPT_volume_of_pyramid_l1317_131792


namespace NUMINAMATH_GPT_car_distance_covered_by_car_l1317_131750

theorem car_distance_covered_by_car
  (V : ℝ)                               -- Initial speed of the car
  (D : ℝ)                               -- Distance covered by the car
  (h1 : D = V * 6)                      -- The car takes 6 hours to cover the distance at speed V
  (h2 : D = 56 * 9)                     -- The car takes 9 hours to cover the distance at speed 56
  : D = 504 :=                          -- Prove that the distance D is 504 kilometers
by
  sorry

end NUMINAMATH_GPT_car_distance_covered_by_car_l1317_131750


namespace NUMINAMATH_GPT_tan_product_l1317_131763

noncomputable def tan : ℝ → ℝ := sorry

theorem tan_product :
  (tan (Real.pi / 8)) * (tan (3 * Real.pi / 8)) * (tan (5 * Real.pi / 8)) = 2 * Real.sqrt 7 :=
by
  sorry

end NUMINAMATH_GPT_tan_product_l1317_131763


namespace NUMINAMATH_GPT_original_number_l1317_131765

theorem original_number (x : ℝ) (hx : 100000 * x = 5 * (1 / x)) : x = 0.00707 := 
by
  sorry

end NUMINAMATH_GPT_original_number_l1317_131765


namespace NUMINAMATH_GPT_m_leq_nine_l1317_131700

theorem m_leq_nine (m : ℝ) : (∀ x : ℝ, (x^2 - 4*x + 3 < 0) → (x^2 - 6*x + 8 < 0) → (2*x^2 - 9*x + m < 0)) → m ≤ 9 :=
by
sorry

end NUMINAMATH_GPT_m_leq_nine_l1317_131700


namespace NUMINAMATH_GPT_maximum_cookies_andy_could_have_eaten_l1317_131793

theorem maximum_cookies_andy_could_have_eaten :
  ∃ x : ℤ, (x ≥ 0 ∧ 2 * x + (x - 3) + x = 30) ∧ (∀ y : ℤ, 0 ≤ y ∧ 2 * y + (y - 3) + y = 30 → y ≤ 8) :=
by {
  sorry
}

end NUMINAMATH_GPT_maximum_cookies_andy_could_have_eaten_l1317_131793


namespace NUMINAMATH_GPT_integer_multiplied_by_b_l1317_131745

variable (a b : ℤ) (x : ℤ)

theorem integer_multiplied_by_b (h1 : -11 * a < 0) (h2 : x < 0) (h3 : (-11 * a * x) * (x * b) + a * b = 89) :
  x = -1 :=
by
  sorry

end NUMINAMATH_GPT_integer_multiplied_by_b_l1317_131745


namespace NUMINAMATH_GPT_no_integer_solutions_m2n_eq_2mn_minus_3_l1317_131789

theorem no_integer_solutions_m2n_eq_2mn_minus_3 :
  ∀ (m n : ℤ), m + 2 * n ≠ 2 * m * n - 3 := 
sorry

end NUMINAMATH_GPT_no_integer_solutions_m2n_eq_2mn_minus_3_l1317_131789


namespace NUMINAMATH_GPT_urn_probability_four_each_l1317_131774

def number_of_sequences := Nat.choose 6 3

def probability_of_sequence := (1/3) * (1/2) * (3/5) * (1/2) * (4/7) * (5/8)

def total_probability := number_of_sequences * probability_of_sequence

theorem urn_probability_four_each :
  total_probability = 5 / 14 := by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_urn_probability_four_each_l1317_131774


namespace NUMINAMATH_GPT_sqrt_simplify_l1317_131757

theorem sqrt_simplify (p : ℝ) :
  (Real.sqrt (12 * p) * Real.sqrt (7 * p^3) * Real.sqrt (15 * p^5)) =
  6 * p^4 * Real.sqrt (35 * p) :=
by
  sorry

end NUMINAMATH_GPT_sqrt_simplify_l1317_131757


namespace NUMINAMATH_GPT_Linda_original_savings_l1317_131769

-- Definition of the problem with all conditions provided.
theorem Linda_original_savings (S : ℝ) (TV_cost : ℝ) (TV_tax_rate : ℝ) (refrigerator_rate : ℝ) (furniture_discount_rate : ℝ) :
  let furniture_cost := (3 / 4) * S
  let TV_cost_with_tax := TV_cost + TV_cost * TV_tax_rate
  let refrigerator_cost := TV_cost + TV_cost * refrigerator_rate
  let remaining_savings := TV_cost_with_tax + refrigerator_cost
  let furniture_cost_after_discount := furniture_cost - furniture_cost * furniture_discount_rate
  (remaining_savings = (1 / 4) * S) →
  S = 1898.40 :=
by
  sorry


end NUMINAMATH_GPT_Linda_original_savings_l1317_131769


namespace NUMINAMATH_GPT_largest_divisor_l1317_131760

theorem largest_divisor (n : ℤ) (h1 : n > 0) (h2 : n % 2 = 1) : 
  (∃ k : ℤ, k > 0 ∧ (∀ n : ℤ, n > 0 → n % 2 = 1 → k ∣ (n * (n + 2) * (n + 4) * (n + 6) * (n + 8)))) → 
  k = 15 :=
by
  sorry

end NUMINAMATH_GPT_largest_divisor_l1317_131760


namespace NUMINAMATH_GPT_calculator_display_exceeds_1000_after_three_presses_l1317_131747

-- Define the operation of pressing the squaring key
def square_key (n : ℕ) : ℕ := n * n

-- Define the initial display number
def initial_display : ℕ := 3

-- Prove that after pressing the squaring key 3 times, the display is greater than 1000.
theorem calculator_display_exceeds_1000_after_three_presses : 
  square_key (square_key (square_key initial_display)) > 1000 :=
by
  sorry

end NUMINAMATH_GPT_calculator_display_exceeds_1000_after_three_presses_l1317_131747


namespace NUMINAMATH_GPT_time_in_future_is_4_l1317_131727

def current_time := 5
def future_hours := 1007
def modulo := 12
def future_time := (current_time + future_hours) % modulo

theorem time_in_future_is_4 : future_time = 4 := by
  sorry

end NUMINAMATH_GPT_time_in_future_is_4_l1317_131727


namespace NUMINAMATH_GPT_intersection_M_N_l1317_131705

def M : Set ℝ := { x : ℝ | 0 < x ∧ x < 4 }
def N : Set ℝ := { x : ℝ | (1 / 3) ≤ x ∧ x ≤ 5 }

theorem intersection_M_N : M ∩ N = { x : ℝ | (1 / 3) ≤ x ∧ x < 4 } :=
by
  sorry

end NUMINAMATH_GPT_intersection_M_N_l1317_131705


namespace NUMINAMATH_GPT_part1_part2_l1317_131708

noncomputable def cost_prices (x y : ℕ) : Prop := 
  8800 / (y + 4) = 2 * (4000 / x) ∧ 
  x = 40 ∧ 
  y = 44

theorem part1 : ∃ x y : ℕ, cost_prices x y := sorry

noncomputable def minimum_lucky_rabbits (m : ℕ) : Prop := 
  26 * m + 20 * (200 - m) ≥ 4120 ∧ 
  m = 20

theorem part2 : ∃ m : ℕ, minimum_lucky_rabbits m := sorry

end NUMINAMATH_GPT_part1_part2_l1317_131708


namespace NUMINAMATH_GPT_combined_ratio_l1317_131711

theorem combined_ratio (cayley_students fermat_students : ℕ) 
                       (cayley_ratio_boys cayley_ratio_girls fermat_ratio_boys fermat_ratio_girls : ℕ) 
                       (h_cayley : cayley_students = 400) 
                       (h_cayley_ratio : (cayley_ratio_boys, cayley_ratio_girls) = (3, 2)) 
                       (h_fermat : fermat_students = 600) 
                       (h_fermat_ratio : (fermat_ratio_boys, fermat_ratio_girls) = (2, 3)) :
  (480 : ℚ) / 520 = 12 / 13 := 
by 
  sorry

end NUMINAMATH_GPT_combined_ratio_l1317_131711


namespace NUMINAMATH_GPT_max_ak_at_k_125_l1317_131790

def binomial_coefficient (n k : ℕ) : ℕ :=
  Nat.choose n k

def ak (k : ℕ) : ℚ :=
  binomial_coefficient 500 k * (0.3)^k

theorem max_ak_at_k_125 : 
  ∀ k : ℕ, k ∈ Finset.range 501 → (ak k ≤ ak 125) :=
by sorry

end NUMINAMATH_GPT_max_ak_at_k_125_l1317_131790


namespace NUMINAMATH_GPT_geometric_sequence_a4_l1317_131721

theorem geometric_sequence_a4 (x a_4 : ℝ) (h1 : 2*x + 2 = (3*x + 3) * (2*x + 2) / x)
  (h2 : x = -4 ∨ x = -1) (h3 : x = -4) : a_4 = -27 / 2 :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_a4_l1317_131721


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_l1317_131713

theorem arithmetic_sequence_sum
  (a : ℕ → ℤ)
  (d : ℤ)
  (h_arith : ∀ n, a (n + 1) = a n + d)
  (h1 : a 1 + a 4 + a 7 = 45)
  (h2 : a 2 + a 5 + a 8 = 39) :
  a 3 + a 6 + a 9 = 33 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_l1317_131713


namespace NUMINAMATH_GPT_sqrt13_decomposition_ten_plus_sqrt3_decomposition_l1317_131754

-- For the first problem
theorem sqrt13_decomposition :
  let a := 3
  let b := Real.sqrt 13 - 3
  a^2 + b - Real.sqrt 13 = 6 := by
sorry

-- For the second problem
theorem ten_plus_sqrt3_decomposition :
  let x := 11
  let y := Real.sqrt 3 - 1
  x - y = 12 - Real.sqrt 3 := by
sorry

end NUMINAMATH_GPT_sqrt13_decomposition_ten_plus_sqrt3_decomposition_l1317_131754


namespace NUMINAMATH_GPT_find_a_l1317_131731

noncomputable def f (x : Real) (a : Real) : Real :=
if h : 0 < x ∧ x < 2 then (Real.log x - a * x) 
else 
if h' : -2 < x ∧ x < 0 then sorry
else 
   sorry

theorem find_a (a : Real) : (∀ x : Real, f x a = - f (-x) a) → (∀ x: Real, (0 < x ∧ x < 2) → f x a = Real.log x - a * x) → a > (1 / 2) → (∀ x: Real, (-2 < x ∧ x < 0) → f x a ≥ 1) → a = 1 := 
sorry

end NUMINAMATH_GPT_find_a_l1317_131731


namespace NUMINAMATH_GPT_max_value_a4b3c2_l1317_131741

theorem max_value_a4b3c2 (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a + b + c = 1) :
  a^4 * b^3 * c^2 ≤ 1 / 6561 :=
sorry

end NUMINAMATH_GPT_max_value_a4b3c2_l1317_131741


namespace NUMINAMATH_GPT_george_speed_to_school_l1317_131707

theorem george_speed_to_school :
  ∀ (d1 d2 v1 v2 v_arrive : ℝ), 
  d1 = 1.0 → d2 = 0.5 → v1 = 3.0 → v2 * (d1 / v1 + d2 / v2) = (d1 + d2) / 4.0 → v_arrive = 12.0 :=
by sorry

end NUMINAMATH_GPT_george_speed_to_school_l1317_131707


namespace NUMINAMATH_GPT_rectangular_field_area_l1317_131761

theorem rectangular_field_area (w l : ℝ) (h1 : w = l / 3) (h2 : 2 * (w + l) = 72) : w * l = 243 := by
  sorry

end NUMINAMATH_GPT_rectangular_field_area_l1317_131761


namespace NUMINAMATH_GPT_correct_sum_of_integers_l1317_131733

theorem correct_sum_of_integers (a b : ℕ) (h1 : a - b = 4) (h2 : a * b = 63) : a + b = 18 := 
  sorry

end NUMINAMATH_GPT_correct_sum_of_integers_l1317_131733


namespace NUMINAMATH_GPT_number_of_correct_inequalities_l1317_131770

variable {a b : ℝ}

theorem number_of_correct_inequalities (h₁ : a > 0) (h₂ : 0 > b) (h₃ : a + b > 0) :
  (ite (a^2 > b^2) 1 0) + (ite (1/a > 1/b) 1 0) + (ite (a^3 < ab^2) 1 0) + (ite (a^2 * b < b^3) 1 0) = 3 := 
sorry

end NUMINAMATH_GPT_number_of_correct_inequalities_l1317_131770


namespace NUMINAMATH_GPT_price_difference_is_correct_l1317_131764

-- Define the conditions
def original_price : ℝ := 1200
def increase_percentage : ℝ := 0.10
def decrease_percentage : ℝ := 0.15

-- Define the intermediate values
def increased_price : ℝ := original_price * (1 + increase_percentage)
def final_price : ℝ := increased_price * (1 - decrease_percentage)
def price_difference : ℝ := original_price - final_price

-- State the theorem to prove
theorem price_difference_is_correct : price_difference = 78 := 
by 
  sorry

end NUMINAMATH_GPT_price_difference_is_correct_l1317_131764


namespace NUMINAMATH_GPT_value_of_f1_l1317_131773

noncomputable def f (x : ℝ) (m : ℝ) := 2 * x^2 - m * x + 3

theorem value_of_f1 (m : ℝ) (h_increasing : ∀ x : ℝ, x ≥ -2 → 2 * x^2 - m * x + 3 ≤ 2 * (x + 1)^2 - m * (x + 1) + 3)
  (h_decreasing : ∀ x : ℝ, x ≤ -2 → 2 * (x - 1)^2 - m * (x - 1) + 3 ≤ 2 * x^2 - m * x + 3) : 
  f 1 (-8) = 13 := 
sorry

end NUMINAMATH_GPT_value_of_f1_l1317_131773


namespace NUMINAMATH_GPT_find_abc_l1317_131783

open Polynomial

noncomputable def my_gcd_lcm_problem (a b c : ℤ) : Prop :=
  gcd (X^2 + (C a * X) + C b) (X^2 + (C b * X) + C c) = X + 1 ∧
  lcm (X^2 + (C a * X) + C b) (X^2 + (C b * X) + C c) = X^3 - 5*X^2 + 7*X - 3

theorem find_abc : ∀ (a b c : ℤ),
  my_gcd_lcm_problem a b c → a + b + c = -3 :=
by
  intros a b c h
  sorry

end NUMINAMATH_GPT_find_abc_l1317_131783


namespace NUMINAMATH_GPT_distinct_values_of_expression_l1317_131797

variable {u v x y z : ℝ}

theorem distinct_values_of_expression (hu : u + u⁻¹ = x) (hv : v + v⁻¹ = y)
  (hx_distinct : x ≠ y) (hx_abs : |x| ≥ 2) (hy_abs : |y| ≥ 2) :
  (∃ z1 z2 : ℝ, z1 ≠ z2 ∧ (z = u * v + (u * v)⁻¹)) →
  ∃ n, n = 2 := by 
    sorry

end NUMINAMATH_GPT_distinct_values_of_expression_l1317_131797


namespace NUMINAMATH_GPT_isosceles_triangle_perimeter_l1317_131723

theorem isosceles_triangle_perimeter (a b c : ℕ) (h_eq_triangle : a + b + c = 60) (h_eq_sides : a = b) 
  (isosceles_base : c = 15) (isosceles_side1_eq : a = 20) : a + b + c = 55 :=
by
  sorry

end NUMINAMATH_GPT_isosceles_triangle_perimeter_l1317_131723


namespace NUMINAMATH_GPT_no_real_solutions_l1317_131709

theorem no_real_solutions :
  ¬ ∃ (a b c d : ℝ), 
  (a^3 + c^3 = 2) ∧ 
  (a^2 * b + c^2 * d = 0) ∧ 
  (b^3 + d^3 = 1) ∧ 
  (a * b^2 + c * d^2 = -6) := 
by
  sorry

end NUMINAMATH_GPT_no_real_solutions_l1317_131709


namespace NUMINAMATH_GPT_max_value_of_fraction_l1317_131719

theorem max_value_of_fraction (x y z : ℕ) (hx : 10 ≤ x ∧ x < 100) (hy : 10 ≤ y ∧ y < 100) (hz : 10 ≤ z ∧ z < 100) (h : x + y + z = 180) : 
  (x + y) / z ≤ 17 :=
sorry

end NUMINAMATH_GPT_max_value_of_fraction_l1317_131719


namespace NUMINAMATH_GPT_value_of_y_l1317_131712

theorem value_of_y (y : ℝ) (h : (3 * y - 9) / 3 = 18) : y = 21 :=
sorry

end NUMINAMATH_GPT_value_of_y_l1317_131712
