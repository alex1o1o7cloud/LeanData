import Mathlib

namespace NUMINAMATH_GPT_store_A_total_cost_store_B_total_cost_cost_effective_store_l1455_145566

open Real

def total_cost_store_A (x : ℝ) : ℝ :=
  110 * x + 210 * (100 - x)

def total_cost_store_B (x : ℝ) : ℝ :=
  120 * x + 202 * (100 - x)

theorem store_A_total_cost (x : ℝ) :
  total_cost_store_A x = -100 * x + 21000 :=
by
  sorry

theorem store_B_total_cost (x : ℝ) :
  total_cost_store_B x = -82 * x + 20200 :=
by
  sorry

theorem cost_effective_store (x : ℝ) (h : x = 60) :
  total_cost_store_A x < total_cost_store_B x :=
by
  rw [h]
  sorry

end NUMINAMATH_GPT_store_A_total_cost_store_B_total_cost_cost_effective_store_l1455_145566


namespace NUMINAMATH_GPT_normal_level_shortage_l1455_145525

theorem normal_level_shortage
  (T : ℝ) (Normal_level : ℝ)
  (h1 : 0.75 * T = 30)
  (h2 : 30 = 2 * Normal_level) :
  T - Normal_level = 25 := 
by
  sorry

end NUMINAMATH_GPT_normal_level_shortage_l1455_145525


namespace NUMINAMATH_GPT_area_of_triangle_with_given_sides_l1455_145576

variable (a b c : ℝ)
variable (s : ℝ := (a + b + c) / 2)
variable (area : ℝ := Real.sqrt (s * (s - a) * (s - b) * (s - c)))

theorem area_of_triangle_with_given_sides (ha : a = 65) (hb : b = 60) (hc : c = 25) :
  area = 750 := by
  sorry

end NUMINAMATH_GPT_area_of_triangle_with_given_sides_l1455_145576


namespace NUMINAMATH_GPT_min_cos_beta_l1455_145571

open Real

theorem min_cos_beta (α β : ℝ)
  (hα : 0 < α ∧ α < π / 2)
  (hβ : 0 < β ∧ β < π / 2)
  (h_eq : sin (2 * α + β) = (3 / 2) * sin β) :
  cos β = sqrt 5 / 3 := 
sorry

end NUMINAMATH_GPT_min_cos_beta_l1455_145571


namespace NUMINAMATH_GPT_range_of_a_l1455_145554

theorem range_of_a (a : ℝ) :
  (∀ x : ℤ, a ≤ x ∧ (x : ℝ) < 2 → x = -1 ∨ x = 0 ∨ x = 1) ↔ (-2 < a ∧ a ≤ -1) :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1455_145554


namespace NUMINAMATH_GPT_sum_of_powers_of_i_l1455_145524

noncomputable def i : Complex := Complex.I

theorem sum_of_powers_of_i :
  (Finset.range 2011).sum (λ n => i^(n+1)) = -1 := by
  sorry

end NUMINAMATH_GPT_sum_of_powers_of_i_l1455_145524


namespace NUMINAMATH_GPT_candy_weight_reduction_l1455_145572

theorem candy_weight_reduction:
  ∀ (W P : ℝ), (33.333333333333314 / 100) * (P / W) = (P / (W - (1/4) * W)) →
  (1 - (W - (1/4) * W) / W) * 100 = 25 :=
by
  intros W P h
  sorry

end NUMINAMATH_GPT_candy_weight_reduction_l1455_145572


namespace NUMINAMATH_GPT_find_a_b_c_sum_l1455_145552

-- Define the necessary conditions and constants
def radius : ℝ := 10  -- tower radius in feet
def rope_length : ℝ := 30  -- length of the rope in feet
def unicorn_height : ℝ := 6  -- height of the unicorn from ground in feet
def rope_end_distance : ℝ := 6  -- distance from the unicorn to the nearest point on the tower

def a : ℕ := 30
def b : ℕ := 900
def c : ℕ := 10  -- assuming c is not necessarily prime for the purpose of this exercise

-- The theorem we want to prove
theorem find_a_b_c_sum : a + b + c = 940 :=
by
  sorry

end NUMINAMATH_GPT_find_a_b_c_sum_l1455_145552


namespace NUMINAMATH_GPT_jean_candy_count_l1455_145580

theorem jean_candy_count : ∃ C : ℕ, 
  C - 7 = 16 ∧ 
  (C - 7 + 7 = C) ∧ 
  (C - 7 = 16) ∧ 
  (C + 0 = C) ∧
  (C - 7 = 16) :=
by 
  sorry 

end NUMINAMATH_GPT_jean_candy_count_l1455_145580


namespace NUMINAMATH_GPT_Rockets_won_38_games_l1455_145586

-- Definitions for each team and their respective wins
variables (Sharks Dolphins Rockets Wolves Comets : ℕ)
variables (wins : Finset ℕ)
variables (shArks_won_more_than_Dolphins : Sharks > Dolphins)
variables (rockets_won_more_than_Wolves : Rockets > Wolves)
variables (rockets_won_fewer_than_Comets : Rockets < Comets)
variables (Wolves_won_more_than_25_games : Wolves > 25)
variables (possible_wins : wins = {28, 33, 38, 43})

-- Statement that the Rockets won 38 games given the conditions
theorem Rockets_won_38_games
  (shArks_won_more_than_Dolphins : Sharks > Dolphins)
  (rockets_won_more_than_Wolves : Rockets > Wolves)
  (rockets_won_fewer_than_Comets : Rockets < Comets)
  (Wolves_won_more_than_25_games : Wolves > 25)
  (possible_wins : wins = {28, 33, 38, 43}) :
  Rockets = 38 :=
sorry

end NUMINAMATH_GPT_Rockets_won_38_games_l1455_145586


namespace NUMINAMATH_GPT_vitamin_d3_total_days_l1455_145579

def vitamin_d3_days (capsules_per_bottle : ℕ) (daily_serving_size : ℕ) (bottles_needed : ℕ) : ℕ :=
  (capsules_per_bottle / daily_serving_size) * bottles_needed

theorem vitamin_d3_total_days :
  vitamin_d3_days 60 2 6 = 180 :=
by
  sorry

end NUMINAMATH_GPT_vitamin_d3_total_days_l1455_145579


namespace NUMINAMATH_GPT_olympic_medals_l1455_145518

theorem olympic_medals :
  ∃ (a b c : ℕ),
    (a + b + c = 100) ∧
    (3 * a - 153 = 0) ∧
    (c - b = 7) ∧
    (a = 51) ∧
    (a - 13 = 38) ∧
    (c = 28) :=
by
  sorry

end NUMINAMATH_GPT_olympic_medals_l1455_145518


namespace NUMINAMATH_GPT_total_team_players_l1455_145526

-- Conditions
def team_percent_boys : ℚ := 0.6
def team_percent_girls := 1 - team_percent_boys
def junior_girls_count : ℕ := 10
def total_girls := junior_girls_count * 2
def girl_percentage_as_decimal := team_percent_girls

-- Problem
theorem total_team_players : (total_girls : ℚ) / girl_percentage_as_decimal = 50 := 
by 
    sorry

end NUMINAMATH_GPT_total_team_players_l1455_145526


namespace NUMINAMATH_GPT_investment_compound_half_yearly_l1455_145582

theorem investment_compound_half_yearly
  (P : ℝ) (r : ℝ) (n : ℕ) (A : ℝ) (t : ℝ) 
  (h1 : P = 6000) 
  (h2 : r = 0.10) 
  (h3 : n = 2) 
  (h4 : A = 6615) :
  t = 1 :=
by
  sorry

end NUMINAMATH_GPT_investment_compound_half_yearly_l1455_145582


namespace NUMINAMATH_GPT_cost_of_fencing_l1455_145534

/-- Define given conditions: -/
def sides_ratio (length width : ℕ) : Prop := length = 3 * width / 2

def park_area : ℕ := 3750

def paise_to_rupees (paise : ℕ) : ℕ := paise / 100

/-- Prove that the cost of fencing the park is 150 rupees: -/
theorem cost_of_fencing 
  (length width : ℕ) 
  (h : sides_ratio length width) 
  (h_area : length * width = park_area) 
  (cost_per_meter_paise : ℕ := 60) : 
  (length + width) * 2 * (paise_to_rupees cost_per_meter_paise) = 150 :=
by sorry

end NUMINAMATH_GPT_cost_of_fencing_l1455_145534


namespace NUMINAMATH_GPT_find_g2_l1455_145506

noncomputable def g : ℝ → ℝ := sorry

axiom functional_eq (x : ℝ) (hx : x ≠ 0) : 4 * g x - 3 * g (1 / x) = x^2

theorem find_g2 : g 2 = 67 / 28 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_g2_l1455_145506


namespace NUMINAMATH_GPT_expression_equals_4096_l1455_145587

noncomputable def calculate_expression : ℕ :=
  ((16^15 / 16^14)^3 * 8^3) / 2^9

theorem expression_equals_4096 : calculate_expression = 4096 :=
by {
  -- proof would go here
  sorry
}

end NUMINAMATH_GPT_expression_equals_4096_l1455_145587


namespace NUMINAMATH_GPT_total_height_of_buildings_l1455_145522

-- Definitions based on the conditions
def tallest_building : ℤ := 100
def second_tallest_building : ℤ := tallest_building / 2
def third_tallest_building : ℤ := second_tallest_building / 2
def fourth_tallest_building : ℤ := third_tallest_building / 5

-- Use the definitions to state the theorem
theorem total_height_of_buildings : 
  tallest_building + second_tallest_building + third_tallest_building + fourth_tallest_building = 180 := by
  sorry

end NUMINAMATH_GPT_total_height_of_buildings_l1455_145522


namespace NUMINAMATH_GPT_exponent_division_l1455_145517

-- We need to reformulate the given condition into Lean definitions
def twenty_seven_is_three_cubed : Prop := 27 = 3^3

-- Using the condition to state the problem
theorem exponent_division (h : twenty_seven_is_three_cubed) : 
  3^15 / 27^3 = 729 :=
by
  sorry

end NUMINAMATH_GPT_exponent_division_l1455_145517


namespace NUMINAMATH_GPT_hypotenuse_of_45_45_90_triangle_l1455_145536

noncomputable def leg_length : ℝ := 15
noncomputable def angle_opposite_leg : ℝ := Real.pi / 4  -- 45 degrees in radians

theorem hypotenuse_of_45_45_90_triangle (h_leg : ℝ) (h_angle : ℝ) 
  (h_leg_cond : h_leg = leg_length) (h_angle_cond : h_angle = angle_opposite_leg) :
  ∃ h_hypotenuse : ℝ, h_hypotenuse = h_leg * Real.sqrt 2 :=
sorry

end NUMINAMATH_GPT_hypotenuse_of_45_45_90_triangle_l1455_145536


namespace NUMINAMATH_GPT_total_weight_of_nuts_l1455_145551

theorem total_weight_of_nuts:
  let almonds := 0.14
  let pecans := 0.38
  let walnuts := 0.22
  let cashews := 0.47
  let pistachios := 0.29
  almonds + pecans + walnuts + cashews + pistachios = 1.50 :=
by
  sorry

end NUMINAMATH_GPT_total_weight_of_nuts_l1455_145551


namespace NUMINAMATH_GPT_last_three_digits_of_7_pow_120_l1455_145503

theorem last_three_digits_of_7_pow_120 :
  7^120 % 1000 = 681 :=
by
  sorry

end NUMINAMATH_GPT_last_three_digits_of_7_pow_120_l1455_145503


namespace NUMINAMATH_GPT_rick_division_steps_l1455_145590

theorem rick_division_steps (initial_books : ℕ) (final_books : ℕ) 
  (h_initial : initial_books = 400) (h_final : final_books = 25) : 
  (∀ n : ℕ, (initial_books / (2^n) = final_books) → n = 4) :=
by
  sorry

end NUMINAMATH_GPT_rick_division_steps_l1455_145590


namespace NUMINAMATH_GPT_sufficient_not_necessary_condition_l1455_145593

variables (a b c : ℝ)

theorem sufficient_not_necessary_condition (h1 : c < b) (h2 : b < a) :
  (ac < 0 → ab > ac) ∧ (ab > ac → ac < 0) → false :=
sorry

end NUMINAMATH_GPT_sufficient_not_necessary_condition_l1455_145593


namespace NUMINAMATH_GPT_ants_in_field_l1455_145520

-- Defining constants
def width_feet : ℕ := 500
def length_feet : ℕ := 600
def ants_per_square_inch : ℕ := 4
def inches_per_foot : ℕ := 12

-- Converting dimensions from feet to inches
def width_inches : ℕ := width_feet * inches_per_foot
def length_inches : ℕ := length_feet * inches_per_foot

-- Calculating the area of the field in square inches
def field_area_square_inches : ℕ := width_inches * length_inches

-- Calculating the total number of ants
def total_ants : ℕ := ants_per_square_inch * field_area_square_inches

-- Theorem statement
theorem ants_in_field : total_ants = 172800000 :=
by
  -- Proof is skipped
  sorry

end NUMINAMATH_GPT_ants_in_field_l1455_145520


namespace NUMINAMATH_GPT_find_number_l1455_145565

theorem find_number (x : ℝ) : 
  220050 = (555 + x) * (2 * (x - 555)) + 50 ↔ x = 425.875 ∨ x = -980.875 := 
by 
  sorry

end NUMINAMATH_GPT_find_number_l1455_145565


namespace NUMINAMATH_GPT_max_non_intersecting_circles_tangent_max_intersecting_circles_without_center_containment_max_intersecting_circles_without_center_containment_2_l1455_145550

-- Definitions and conditions related to the given problem
def unit_circle (r : ℝ) : Prop := r = 1

-- Maximum number of non-intersecting circles of radius 1 tangent to a unit circle.
theorem max_non_intersecting_circles_tangent (r : ℝ) (K : ℝ) 
  (h_r : unit_circle r) (h_K : unit_circle K) : 
  ∃ n, n = 6 := sorry

-- Maximum number of circles of radius 1 intersecting a given unit circle without intersecting centers.
theorem max_intersecting_circles_without_center_containment (r : ℝ) (K : ℝ) 
  (h_r : unit_circle r) (h_K : unit_circle K) : 
  ∃ n, n = 12 := sorry

-- Maximum number of circles of radius 1 intersecting a unit circle K without containing the center of K or any other circle's center.
theorem max_intersecting_circles_without_center_containment_2 (r : ℝ) (K : ℝ)
  (h_r : unit_circle r) (h_K : unit_circle K) :
  ∃ n, n = 18 := sorry

end NUMINAMATH_GPT_max_non_intersecting_circles_tangent_max_intersecting_circles_without_center_containment_max_intersecting_circles_without_center_containment_2_l1455_145550


namespace NUMINAMATH_GPT_length_of_faster_train_l1455_145539

/-- 
Let the faster train have a speed of 144 km per hour, the slower train a speed of 
72 km per hour, and the time taken for the faster train to cross a man in the 
slower train be 19 seconds. Then the length of the faster train is 380 meters.
-/
theorem length_of_faster_train 
  (speed_faster_train : ℝ) (speed_slower_train : ℝ) (time_to_cross : ℝ)
  (h_speed_faster_train : speed_faster_train = 144) 
  (h_speed_slower_train : speed_slower_train = 72) 
  (h_time_to_cross : time_to_cross = 19) :
  (speed_faster_train - speed_slower_train) * (5 / 18) * time_to_cross = 380 :=
by
  sorry

end NUMINAMATH_GPT_length_of_faster_train_l1455_145539


namespace NUMINAMATH_GPT_twenty_yuan_banknotes_count_l1455_145516

theorem twenty_yuan_banknotes_count (x y z : ℕ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
                                    (total_banknotes : x + y + z = 24)
                                    (total_amount : 10 * x + 20 * y + 50 * z = 1000) :
                                    y = 4 := 
sorry

end NUMINAMATH_GPT_twenty_yuan_banknotes_count_l1455_145516


namespace NUMINAMATH_GPT_rectangle_perimeters_l1455_145521

theorem rectangle_perimeters (length width : ℕ) (h1 : length = 7) (h2 : width = 5) :
  (∃ (L1 L2 : ℕ), L1 = 4 * width ∧ L2 = length ∧ 2 * (L1 + L2) = 54) ∧
  (∃ (L3 L4 : ℕ), L3 = 4 * length ∧ L4 = width ∧ 2 * (L3 + L4) = 66) ∧
  (∃ (L5 L6 : ℕ), L5 = 2 * length ∧ L6 = 2 * width ∧ 2 * (L5 + L6) = 48) :=
by
  sorry

end NUMINAMATH_GPT_rectangle_perimeters_l1455_145521


namespace NUMINAMATH_GPT_total_sandwiches_prepared_l1455_145535

def num_people := 219.0
def sandwiches_per_person := 3.0

theorem total_sandwiches_prepared : num_people * sandwiches_per_person = 657.0 :=
by
  sorry

end NUMINAMATH_GPT_total_sandwiches_prepared_l1455_145535


namespace NUMINAMATH_GPT_power_function_passing_through_point_l1455_145545

theorem power_function_passing_through_point :
  ∃ (α : ℝ), (2:ℝ)^α = 4 := by
  sorry

end NUMINAMATH_GPT_power_function_passing_through_point_l1455_145545


namespace NUMINAMATH_GPT_part_a_l1455_145546

theorem part_a (c : ℤ) : (∃ x : ℤ, x + (x / 2) = c) ↔ (c % 3 ≠ 2) :=
sorry

end NUMINAMATH_GPT_part_a_l1455_145546


namespace NUMINAMATH_GPT_triangle_area_l1455_145510

theorem triangle_area {a b : ℝ} (h : a ≠ 0) :
  (∃ x y : ℝ, 3 * x + a * y = 12) → b = 24 / a ↔ (∃ x y : ℝ, x = 4 ∧ y = 12 / a ∧ b = (1/2) * 4 * (12 / a)) :=
by
  sorry

end NUMINAMATH_GPT_triangle_area_l1455_145510


namespace NUMINAMATH_GPT_find_angle_B_l1455_145505

noncomputable def angle_B (A B C a b c : ℝ): Prop := 
  a * Real.cos B - b * Real.cos A = b ∧ 
  C = Real.pi / 5

theorem find_angle_B (a b c A B C : ℝ) (h : angle_B A B C a b c) : 
  B = 4 * Real.pi / 15 :=
by
  sorry

end NUMINAMATH_GPT_find_angle_B_l1455_145505


namespace NUMINAMATH_GPT_value_of_1_minus_a_l1455_145559

theorem value_of_1_minus_a (a : ℤ) (h : a = -(-6)) : 1 - a = -5 := 
by 
  sorry

end NUMINAMATH_GPT_value_of_1_minus_a_l1455_145559


namespace NUMINAMATH_GPT_sravan_distance_l1455_145532

theorem sravan_distance {D : ℝ} :
  (D / 90 + D / 60 = 15) ↔ (D = 540) :=
by sorry

end NUMINAMATH_GPT_sravan_distance_l1455_145532


namespace NUMINAMATH_GPT_gretchen_flavors_l1455_145549

/-- 
Gretchen's local ice cream shop offers 100 different flavors. She tried a quarter of the flavors 2 years ago and double that amount last year. Prove how many more flavors she needs to try this year to have tried all 100 flavors.
-/
theorem gretchen_flavors (F T2 T1 T R : ℕ) (h1 : F = 100)
  (h2 : T2 = F / 4)
  (h3 : T1 = 2 * T2)
  (h4 : T = T2 + T1)
  (h5 : R = F - T) : R = 25 :=
sorry

end NUMINAMATH_GPT_gretchen_flavors_l1455_145549


namespace NUMINAMATH_GPT_baseball_card_value_decrease_l1455_145515

theorem baseball_card_value_decrease (initial_value : ℝ) :
  (1 - 0.70 * 0.90) * 100 = 37 := 
by sorry

end NUMINAMATH_GPT_baseball_card_value_decrease_l1455_145515


namespace NUMINAMATH_GPT_cos_double_angle_l1455_145583

theorem cos_double_angle (α : ℝ) (h : Real.sin α = (Real.sqrt 3) / 2) : 
  Real.cos (2 * α) = -1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_cos_double_angle_l1455_145583


namespace NUMINAMATH_GPT_simplify_and_evaluate_l1455_145500

-- Define the condition as a predicate
def condition (a b : ℝ) : Prop := (a + 1/2)^2 + |b - 2| = 0

-- The simplified expression
def simplified_expression (a b : ℝ) : ℝ := 12 * a^2 * b - 6 * a * b^2

-- Statement: Given the condition, prove that the simplified expression evaluates to 18
theorem simplify_and_evaluate : ∀ (a b : ℝ), condition a b → simplified_expression a b = 18 :=
by
  intros a b hc
  sorry  -- Proof omitted

end NUMINAMATH_GPT_simplify_and_evaluate_l1455_145500


namespace NUMINAMATH_GPT_fixed_fee_rental_l1455_145594

theorem fixed_fee_rental (F C h : ℕ) (hC : C = F + 7 * h) (hC80 : C = 80) (hh9 : h = 9) : F = 17 :=
by
  sorry

end NUMINAMATH_GPT_fixed_fee_rental_l1455_145594


namespace NUMINAMATH_GPT_cevians_concurrent_circumscribable_l1455_145553

-- Define the problem
variables {A B C D X Y Z : Type}

-- Define concurrent cevians
def cevian_concurrent (A B C X Y Z D : Type) : Prop := true

-- Define circumscribable quadrilaterals
def circumscribable (A B C D : Type) : Prop := true

-- The theorem statement
theorem cevians_concurrent_circumscribable (h_conc: cevian_concurrent A B C X Y Z D) 
(h1: circumscribable D Y A Z) (h2: circumscribable D Z B X) : circumscribable D X C Y :=
sorry

end NUMINAMATH_GPT_cevians_concurrent_circumscribable_l1455_145553


namespace NUMINAMATH_GPT_mean_age_Mendez_children_l1455_145585

def Mendez_children_ages : List ℕ := [5, 5, 10, 12, 15]

theorem mean_age_Mendez_children : 
  (5 + 5 + 10 + 12 + 15) / 5 = 9.4 := 
by
  sorry

end NUMINAMATH_GPT_mean_age_Mendez_children_l1455_145585


namespace NUMINAMATH_GPT_sum_of_possible_areas_of_square_in_xy_plane_l1455_145537

theorem sum_of_possible_areas_of_square_in_xy_plane (x1 x2 x3 : ℝ) (A : ℝ)
    (h1 : x1 = 2 ∨ x1 = 0 ∨ x1 = 18)
    (h2 : x2 = 2 ∨ x2 = 0 ∨ x2 = 18)
    (h3 : x3 = 2 ∨ x3 = 0 ∨ x3 = 18) :
  A = 1168 := sorry

end NUMINAMATH_GPT_sum_of_possible_areas_of_square_in_xy_plane_l1455_145537


namespace NUMINAMATH_GPT_tetrahedron_sphere_relations_l1455_145514

theorem tetrahedron_sphere_relations 
  (ρ ρ1 ρ2 ρ3 ρ4 m1 m2 m3 m4 : ℝ)
  (hρ_pos : ρ > 0)
  (hρ1_pos : ρ1 > 0)
  (hρ2_pos : ρ2 > 0)
  (hρ3_pos : ρ3 > 0)
  (hρ4_pos : ρ4 > 0)
  (hm1_pos : m1 > 0)
  (hm2_pos : m2 > 0)
  (hm3_pos : m3 > 0)
  (hm4_pos : m4 > 0) : 
  (2 / ρ = 1 / ρ1 + 1 / ρ2 + 1 / ρ3 + 1 / ρ4) ∧
  (1 / ρ = 1 / m1 + 1 / m2 + 1 / m3 + 1 / m4) ∧
  ( 1 / ρ1 = -1 / m1 + 1 / m2 + 1 / m3 + 1 / m4 ) := sorry

end NUMINAMATH_GPT_tetrahedron_sphere_relations_l1455_145514


namespace NUMINAMATH_GPT_ab_eq_neg_two_l1455_145555

theorem ab_eq_neg_two (a b : ℝ) (h : |a - 1| + (b + 2)^2 = 0) : a * b^a = -2 :=
by
  sorry

end NUMINAMATH_GPT_ab_eq_neg_two_l1455_145555


namespace NUMINAMATH_GPT_f_f_3_eq_651_over_260_l1455_145544

def f (x : ℚ) : ℚ := x⁻¹ + (x⁻¹ / (2 + x⁻¹))

/-- Prove that f(f(3)) = 651/260 -/
theorem f_f_3_eq_651_over_260 : f (f (3)) = 651 / 260 := 
sorry

end NUMINAMATH_GPT_f_f_3_eq_651_over_260_l1455_145544


namespace NUMINAMATH_GPT_cost_of_trip_per_student_l1455_145511

def raised_fund : ℕ := 50
def contribution_per_student : ℕ := 5
def num_students : ℕ := 20
def remaining_fund : ℕ := 10

theorem cost_of_trip_per_student :
  ((raised_fund - remaining_fund) / num_students) = 2 := by
  sorry

end NUMINAMATH_GPT_cost_of_trip_per_student_l1455_145511


namespace NUMINAMATH_GPT_unique_solution_abs_eq_l1455_145513

theorem unique_solution_abs_eq (x : ℝ) : (|x - 9| = |x + 3| + 2) ↔ x = 2 :=
by
  sorry

end NUMINAMATH_GPT_unique_solution_abs_eq_l1455_145513


namespace NUMINAMATH_GPT_problem_statement_l1455_145589

-- Define y as the sum of the given terms
def y : ℤ := 128 + 192 + 256 + 320 + 576 + 704 + 6464

-- The theorem to prove that y is a multiple of 8, 16, 32, and 64
theorem problem_statement : 
  (8 ∣ y) ∧ (16 ∣ y) ∧ (32 ∣ y) ∧ (64 ∣ y) :=
by sorry

end NUMINAMATH_GPT_problem_statement_l1455_145589


namespace NUMINAMATH_GPT_find_line_equation_l1455_145509

theorem find_line_equation :
  ∃ (a b c : ℝ), (a * -5 + b * -1 = c) ∧ (a * 1 + b * 1 = c + 2) ∧ (b ≠ 0) ∧ (a * 2 + b = 0) → (∃ (a b c : ℝ), a = 1 ∧ b = -2 ∧ c = -5) :=
by
  sorry

end NUMINAMATH_GPT_find_line_equation_l1455_145509


namespace NUMINAMATH_GPT_fill_cistern_time_l1455_145556

theorem fill_cistern_time (A B C : ℕ) (hA : A = 10) (hB : B = 12) (hC : C = 50) :
    1 / (1 / A + 1 / B - 1 / C) = 300 / 49 :=
by
  sorry

end NUMINAMATH_GPT_fill_cistern_time_l1455_145556


namespace NUMINAMATH_GPT_sum_six_smallest_multiples_of_eleven_l1455_145573

theorem sum_six_smallest_multiples_of_eleven : 
  (11 + 22 + 33 + 44 + 55 + 66) = 231 :=
by
  sorry

end NUMINAMATH_GPT_sum_six_smallest_multiples_of_eleven_l1455_145573


namespace NUMINAMATH_GPT_larger_investment_value_l1455_145548

-- Definitions of the conditions given in the problem
def investment_value_1 : ℝ := 500
def yearly_return_rate_1 : ℝ := 0.07
def yearly_return_rate_2 : ℝ := 0.27
def combined_return_rate : ℝ := 0.22

-- Stating the proof problem
theorem larger_investment_value :
  ∃ X : ℝ, X = 1500 ∧ 
    yearly_return_rate_1 * investment_value_1 + yearly_return_rate_2 * X = combined_return_rate * (investment_value_1 + X) :=
by {
  sorry -- Proof is omitted as per instructions
}

end NUMINAMATH_GPT_larger_investment_value_l1455_145548


namespace NUMINAMATH_GPT_graph_translation_l1455_145528

variable (f : ℝ → ℝ)

theorem graph_translation (h : f 1 = 3) : f (-1) + 1 = 4 :=
sorry

end NUMINAMATH_GPT_graph_translation_l1455_145528


namespace NUMINAMATH_GPT_circle_properties_radius_properties_l1455_145588

theorem circle_properties (m x y : ℝ) :
  (x^2 + y^2 - 2*(m + 3)*x + 2*(1 - 4*m^2)*y + 16*m^4 + 9 = 0) ↔
    (-((1 : ℝ) / (7 : ℝ)) < m ∧ m < (1 : ℝ)) :=
sorry

theorem radius_properties (m : ℝ) (h : -((1 : ℝ) / (7 : ℝ)) < m ∧ m < (1 : ℝ)) :
  ∃ r : ℝ, (0 < r ∧ r ≤ (4 / Real.sqrt 7)) :=
sorry

end NUMINAMATH_GPT_circle_properties_radius_properties_l1455_145588


namespace NUMINAMATH_GPT_decision_represented_by_D_l1455_145577

-- Define the basic symbols in the flowchart
inductive BasicSymbol
| Start
| Process
| Decision
| End

open BasicSymbol

-- Define the meaning of each basic symbol
def meaning_of (sym : BasicSymbol) : String :=
  match sym with
  | Start => "start"
  | Process => "process"
  | Decision => "decision"
  | End => "end"

-- The theorem stating that the Decision symbol represents a decision
theorem decision_represented_by_D : meaning_of Decision = "decision" :=
by sorry

end NUMINAMATH_GPT_decision_represented_by_D_l1455_145577


namespace NUMINAMATH_GPT_total_customers_served_l1455_145578

-- Definitions for the hours worked by Ann, Becky, and Julia
def hours_ann : ℕ := 8
def hours_becky : ℕ := 8
def hours_julia : ℕ := 6

-- Definition for the number of customers served per hour
def customers_per_hour : ℕ := 7

-- Total number of customers served by Ann, Becky, and Julia
def total_customers : ℕ :=
  (hours_ann * customers_per_hour) + 
  (hours_becky * customers_per_hour) + 
  (hours_julia * customers_per_hour)

theorem total_customers_served : total_customers = 154 :=
  by 
    -- This is where the proof would go, but we'll use sorry to indicate it's incomplete
    sorry

end NUMINAMATH_GPT_total_customers_served_l1455_145578


namespace NUMINAMATH_GPT_remainder_when_divided_by_13_l1455_145547

theorem remainder_when_divided_by_13 (N : ℕ) (k : ℕ) : (N = 39 * k + 17) → (N % 13 = 4) := by
  sorry

end NUMINAMATH_GPT_remainder_when_divided_by_13_l1455_145547


namespace NUMINAMATH_GPT_min_count_to_ensure_multiple_of_5_l1455_145563

theorem min_count_to_ensure_multiple_of_5 (n : ℕ) (S : Finset ℕ) (hS : S = Finset.range 31) :
  25 ≤ S.card ∧ (∀ (T : Finset ℕ), T ⊆ S → T.card = 24 → ↑(∃ x ∈ T, x % 5 = 0)) :=
by sorry

end NUMINAMATH_GPT_min_count_to_ensure_multiple_of_5_l1455_145563


namespace NUMINAMATH_GPT_nitin_borrowed_amount_l1455_145562

theorem nitin_borrowed_amount (P : ℝ) (interest_paid : ℝ) 
  (rate1 rate2 rate3 : ℝ) (time1 time2 time3 : ℝ) 
  (h_rates1 : rate1 = 0.06) (h_rates2 : rate2 = 0.09) 
  (h_rates3 : rate3 = 0.13) (h_time1 : time1 = 3) 
  (h_time2 : time2 = 5) (h_time3 : time3 = 3)
  (h_interest : interest_paid = 8160) :
  P * (rate1 * time1 + rate2 * time2 + rate3 * time3) = interest_paid → 
  P = 8000 := 
by 
  sorry

end NUMINAMATH_GPT_nitin_borrowed_amount_l1455_145562


namespace NUMINAMATH_GPT_value_of_z_l1455_145569

theorem value_of_z :
  let mean_of_4_16_20 := (4 + 16 + 20) / 3
  let mean_of_8_z := (8 + z) / 2
  ∀ z : ℚ, mean_of_4_16_20 = mean_of_8_z → z = 56 / 3 := 
by
  intro z mean_eq
  sorry

end NUMINAMATH_GPT_value_of_z_l1455_145569


namespace NUMINAMATH_GPT_shaded_trapezium_area_l1455_145531

theorem shaded_trapezium_area :
  let side1 := 3
  let side2 := 5
  let side3 := 8
  let p := 3 / 2
  let q := 4
  let height := 5
  let area := (1 / 2) * (p + q) * height
  area = 55 / 4 :=
by
  let side1 := 3
  let side2 := 5
  let side3 := 8
  let p := 3 / 2
  let q := 4
  let height := 5
  let area := (1 / 2) * (p + q) * height
  show area = 55 / 4
  sorry

end NUMINAMATH_GPT_shaded_trapezium_area_l1455_145531


namespace NUMINAMATH_GPT_farmer_planning_problem_l1455_145570

theorem farmer_planning_problem
  (A : ℕ) (D : ℕ)
  (h1 : A = 120 * D)
  (h2 : ∀ t : ℕ, t = 85 * (D + 5) + 40)
  (h3 : 85 * (D + 5) + 40 = 120 * D) : 
  A = 1560 ∧ D = 13 := 
by
  sorry

end NUMINAMATH_GPT_farmer_planning_problem_l1455_145570


namespace NUMINAMATH_GPT_prove_k_eq_one_l1455_145529

theorem prove_k_eq_one 
  (n m k : ℕ) 
  (h_positive : 0 < n)  -- implies n, and hence n-1, n+1, are all positive
  (h_eq : (n-1) * n * (n+1) = m^k): 
  k = 1 := 
sorry

end NUMINAMATH_GPT_prove_k_eq_one_l1455_145529


namespace NUMINAMATH_GPT_determine_e_l1455_145574

-- Define the polynomial Q(x)
def Q (x : ℝ) (d e f : ℝ) : ℝ := 3 * x^3 + d * x^2 + e * x + f

-- Define the problem statement
theorem determine_e (d e f : ℝ)
  (h1 : f = 9)
  (h2 : (d * (d + 9)) - 168 = 0)
  (h3 : d^2 - 6 * e = 12 + d + e)
  : e = -24 ∨ e = 20 :=
by
  sorry

end NUMINAMATH_GPT_determine_e_l1455_145574


namespace NUMINAMATH_GPT_percentage_of_loss_l1455_145560

-- Define the conditions as given in the problem
def original_selling_price : ℝ := 720
def gain_selling_price : ℝ := 880
def gain_percentage : ℝ := 0.10

-- Define the main theorem
theorem percentage_of_loss : ∀ (CP : ℝ),
  (1.10 * CP = gain_selling_price) → 
  ((CP - original_selling_price) / CP * 100 = 10) :=
by
  intro CP
  intro h
  have h1 : CP = gain_selling_price / 1.10 := by sorry
  have h2 : (CP - original_selling_price) = 80 := by sorry -- Intermediate step to show loss
  have h3 : ((80 / CP) * 100 = 10) := by sorry -- Calculation of percentage of loss
  sorry

end NUMINAMATH_GPT_percentage_of_loss_l1455_145560


namespace NUMINAMATH_GPT_sec_150_eq_neg_two_sqrt_three_over_three_l1455_145561

-- Definitions to match the problem conditions
noncomputable def sec (θ : ℝ) : ℝ := 1 / Real.cos θ

-- Stating the proposition to be proved
theorem sec_150_eq_neg_two_sqrt_three_over_three : sec (150 * Real.pi / 180) = -2 * Real.sqrt 3 / 3 := 
sorry

end NUMINAMATH_GPT_sec_150_eq_neg_two_sqrt_three_over_three_l1455_145561


namespace NUMINAMATH_GPT_roots_condition_l1455_145501

theorem roots_condition (m : ℝ) : 
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x1 > 3 ∧ x2 < 3 ∧ x1^2 - m * x1 + 2 * m = 0 ∧ x2^2 - m * x2 + 2 * m = 0) ↔ m > 9 :=
by sorry

end NUMINAMATH_GPT_roots_condition_l1455_145501


namespace NUMINAMATH_GPT_weight_of_second_new_player_l1455_145527

theorem weight_of_second_new_player
  (number_of_original_players : ℕ)
  (average_weight_of_original_players : ℝ)
  (weight_of_first_new_player : ℝ)
  (new_average_weight : ℝ)
  (total_number_of_players : ℕ)
  (total_weight_of_9_players : ℝ)
  (combined_weight_of_original_and_first_new : ℝ)
  (weight_of_second_new_player : ℝ)
  (h1 : number_of_original_players = 7)
  (h2 : average_weight_of_original_players = 103)
  (h3 : weight_of_first_new_player = 110)
  (h4 : new_average_weight = 99)
  (h5 : total_number_of_players = 9)
  (h6 : total_weight_of_9_players = total_number_of_players * new_average_weight)
  (h7 : combined_weight_of_original_and_first_new = number_of_original_players * average_weight_of_original_players + weight_of_first_new_player)
  (h8 : total_weight_of_9_players - combined_weight_of_original_and_first_new = weight_of_second_new_player) :
  weight_of_second_new_player = 60 :=
by
  sorry

end NUMINAMATH_GPT_weight_of_second_new_player_l1455_145527


namespace NUMINAMATH_GPT_new_weight_l1455_145575

-- Conditions
def avg_weight_increase (n : ℕ) (avg_increase : ℝ) : ℝ := n * avg_increase
def weight_replacement (initial_weight : ℝ) (total_increase : ℝ) : ℝ := initial_weight + total_increase

-- Problem Statement: Proving the weight of the new person
theorem new_weight {n : ℕ} {avg_increase initial_weight W : ℝ} 
  (h_n : n = 8) (h_avg_increase : avg_increase = 2.5) (h_initial_weight : initial_weight = 65) (h_W : W = 85) :
  weight_replacement initial_weight (avg_weight_increase n avg_increase) = W :=
by 
  rw [h_n, h_avg_increase, h_initial_weight, h_W]
  sorry

end NUMINAMATH_GPT_new_weight_l1455_145575


namespace NUMINAMATH_GPT_probability_of_closer_to_D_in_triangle_DEF_l1455_145543

noncomputable def triangle_area (a b c : ℝ × ℝ) : ℝ :=
  0.5 * abs ((b.1 - a.1) * (c.2 - a.2) - (c.1 - a.1) * (b.2 - a.2))

theorem probability_of_closer_to_D_in_triangle_DEF :
  let D := (0, 0)
  let E := (0, 6)
  let F := (8, 0)
  let M := ((D.1 + F.1) / 2, (D.2 + F.2) / 2)
  let N := ((D.1 + E.1) / 2, (D.2 + E.2) / 2)
  let area_DEF := triangle_area D E F
  let area_DMN := triangle_area D M N
  area_DMN / area_DEF = 1 / 4 := by
    sorry

end NUMINAMATH_GPT_probability_of_closer_to_D_in_triangle_DEF_l1455_145543


namespace NUMINAMATH_GPT_payment_per_minor_character_l1455_145504

noncomputable def M : ℝ := 285000 / 19 

theorem payment_per_minor_character
    (num_main_characters : ℕ := 5)
    (num_minor_characters : ℕ := 4)
    (total_payment : ℝ := 285000)
    (payment_ratio : ℝ := 3)
    (eq1 : 5 * 3 * M + 4 * M = total_payment) :
    M = 15000 :=
by
  sorry

end NUMINAMATH_GPT_payment_per_minor_character_l1455_145504


namespace NUMINAMATH_GPT_infinite_series_sum_l1455_145597

theorem infinite_series_sum :
  ∑' (k : ℕ), (k + 1) / 4^(k + 1) = 4 / 9 :=
sorry

end NUMINAMATH_GPT_infinite_series_sum_l1455_145597


namespace NUMINAMATH_GPT_scientific_notation_correct_l1455_145581

theorem scientific_notation_correct : 657000 = 6.57 * 10^5 :=
by
  sorry

end NUMINAMATH_GPT_scientific_notation_correct_l1455_145581


namespace NUMINAMATH_GPT_total_score_is_correct_l1455_145523

def dad_points : ℕ := 7
def olaf_points : ℕ := 3 * dad_points
def total_points : ℕ := dad_points + olaf_points

theorem total_score_is_correct : total_points = 28 := by
  sorry

end NUMINAMATH_GPT_total_score_is_correct_l1455_145523


namespace NUMINAMATH_GPT_max_number_of_band_members_l1455_145599

-- Conditions definitions
def num_band_members (r x : ℕ) : ℕ := r * x + 3

def num_band_members_new (r x : ℕ) : ℕ := (r - 1) * (x + 2)

-- The main statement
theorem max_number_of_band_members :
  ∃ (r x : ℕ), num_band_members r x = 231 ∧ num_band_members_new r x = 231 
  ∧ ∀ (r' x' : ℕ), (num_band_members r' x' < 120 ∧ num_band_members_new r' x' = num_band_members r' x') → (num_band_members r' x' ≤ 231) :=
sorry

end NUMINAMATH_GPT_max_number_of_band_members_l1455_145599


namespace NUMINAMATH_GPT_a_minus_b_ge_one_l1455_145567

def a : ℕ := 19^91
def b : ℕ := (999991)^19

theorem a_minus_b_ge_one : a - b ≥ 1 :=
by
  sorry

end NUMINAMATH_GPT_a_minus_b_ge_one_l1455_145567


namespace NUMINAMATH_GPT_money_left_l1455_145557

-- Conditions
def initial_savings : ℤ := 6000
def spent_on_flight : ℤ := 1200
def spent_on_hotel : ℤ := 800
def spent_on_food : ℤ := 3000

-- Total spent
def total_spent : ℤ := spent_on_flight + spent_on_hotel + spent_on_food

-- Prove that the money left is $1,000
theorem money_left (h1 : initial_savings = 6000)
                   (h2 : spent_on_flight = 1200)
                   (h3 : spent_on_hotel = 800)
                   (h4 : spent_on_food = 3000) :
                   initial_savings - total_spent = 1000 :=
by
  -- Insert proof steps here
  sorry

end NUMINAMATH_GPT_money_left_l1455_145557


namespace NUMINAMATH_GPT_minute_hand_angle_is_pi_six_minute_hand_arc_length_is_2pi_third_l1455_145502

theorem minute_hand_angle_is_pi_six (radius : ℝ) (fast_min : ℝ) (h1 : radius = 4) (h2 : fast_min = 5) :
  (fast_min / 60 * 2 * Real.pi = Real.pi / 6) :=
by sorry

theorem minute_hand_arc_length_is_2pi_third (radius : ℝ) (angle : ℝ) (fast_min : ℝ) (h1 : radius = 4) (h2 : angle = Real.pi / 6) (h3 : fast_min = 5) :
  (radius * angle = 2 * Real.pi / 3) :=
by sorry

end NUMINAMATH_GPT_minute_hand_angle_is_pi_six_minute_hand_arc_length_is_2pi_third_l1455_145502


namespace NUMINAMATH_GPT_shaniqua_income_per_haircut_l1455_145568

theorem shaniqua_income_per_haircut (H : ℝ) :
  (8 * H + 5 * 25 = 221) → (H = 12) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_shaniqua_income_per_haircut_l1455_145568


namespace NUMINAMATH_GPT_speed_conversion_l1455_145584

-- Define the conversion factor
def conversion_factor := 3.6

-- Define the given speed in meters per second
def speed_mps := 16.668

-- Define the expected speed in kilometers per hour
def expected_speed_kmph := 60.0048

-- The theorem to prove that the given speed in m/s converts to the expected speed in km/h
theorem speed_conversion : speed_mps * conversion_factor = expected_speed_kmph := 
  by
    sorry

end NUMINAMATH_GPT_speed_conversion_l1455_145584


namespace NUMINAMATH_GPT_tetrahedron_max_volume_l1455_145596

noncomputable def tetrahedron_volume (AC AB BD CD : ℝ) : ℝ :=
  let x := (2 : ℝ) * (Real.sqrt 3) / 3
  let m := Real.sqrt (1 - x^2 / 4)
  let α := Real.pi / 2 -- Maximize with sin α = 1
  x * m^2 * Real.sin α / 6

theorem tetrahedron_max_volume : ∀ (AC AB BD CD : ℝ),
  AC = 1 → AB = 1 → BD = 1 → CD = 1 →
  tetrahedron_volume AC AB BD CD = 2 * Real.sqrt 3 / 27 :=
by
  intros AC AB BD CD hAC hAB hBD hCD
  rw [hAC, hAB, hBD, hCD]
  dsimp [tetrahedron_volume]
  norm_num
  sorry

end NUMINAMATH_GPT_tetrahedron_max_volume_l1455_145596


namespace NUMINAMATH_GPT_problem_equivalent_l1455_145508

theorem problem_equivalent (a c : ℕ) (h : (3 * 100 + a * 10 + 7) + 214 = 5 * 100 + c * 10 + 1) (h5c1_div3 : (5 + c + 1) % 3 = 0) : a + c = 4 :=
sorry

end NUMINAMATH_GPT_problem_equivalent_l1455_145508


namespace NUMINAMATH_GPT_domain_of_f_l1455_145530

noncomputable def f (x : ℝ) := Real.sqrt (x - 1) + (1 / (x - 2))

theorem domain_of_f : { x : ℝ | x ≥ 1 ∧ x ≠ 2 } = { x : ℝ | ∃ (y : ℝ), f x = y } :=
sorry

end NUMINAMATH_GPT_domain_of_f_l1455_145530


namespace NUMINAMATH_GPT_sum_quotient_dividend_divisor_l1455_145591

theorem sum_quotient_dividend_divisor (n : ℕ) (d : ℕ) (h : n = 45) (h1 : d = 3) : 
  (n / d) + n + d = 63 :=
by
  sorry

end NUMINAMATH_GPT_sum_quotient_dividend_divisor_l1455_145591


namespace NUMINAMATH_GPT_total_legs_in_room_l1455_145507

def count_legs : Nat :=
  let tables_4_legs := 4 * 4
  let sofas_legs := 1 * 4
  let chairs_4_legs := 2 * 4
  let tables_3_legs := 3 * 3
  let tables_1_leg := 1 * 1
  let rocking_chair_legs := 1 * 2
  tables_4_legs + sofas_legs + chairs_4_legs + tables_3_legs + tables_1_leg + rocking_chair_legs

theorem total_legs_in_room : count_legs = 40 := by
  sorry

end NUMINAMATH_GPT_total_legs_in_room_l1455_145507


namespace NUMINAMATH_GPT_problem1_problem2_problem3_problem4_l1455_145564

def R : Set ℝ := Set.univ
def A : Set ℝ := {x | 1 < x ∧ x < 5}
def B : Set ℝ := {x | 3 ≤ x ∧ x ≤ 6}

theorem problem1 : A ∩ B = {x | 3 ≤ x ∧ x < 5} := sorry

theorem problem2 : A ∪ B = {x | 1 < x ∧ x ≤ 6} := sorry

theorem problem3 : (Set.compl A) ∩ B = {x | 5 ≤ x ∧ x ≤ 6} :=
sorry

theorem problem4 : Set.compl (A ∩ B) = {x | x < 3 ∨ x ≥ 5} := sorry

end NUMINAMATH_GPT_problem1_problem2_problem3_problem4_l1455_145564


namespace NUMINAMATH_GPT_exists_common_point_l1455_145538

-- Definitions: Rectangle and the problem conditions
structure Rectangle :=
(x_min y_min x_max y_max : ℝ)
(h_valid : x_min ≤ x_max ∧ y_min ≤ y_max)

def rectangles_intersect (R1 R2 : Rectangle) : Prop :=
¬(R1.x_max < R2.x_min ∨ R2.x_max < R1.x_min ∨ R1.y_max < R2.y_min ∨ R2.y_max < R1.y_min)

def all_rectangles_intersect (rects : List Rectangle) : Prop :=
∀ (R1 R2 : Rectangle), R1 ∈ rects → R2 ∈ rects → rectangles_intersect R1 R2

-- Theorem: Existence of a common point
theorem exists_common_point (rects : List Rectangle) (h_intersect : all_rectangles_intersect rects) : 
  ∃ (T : ℝ × ℝ), ∀ (R : Rectangle), R ∈ rects → 
    R.x_min ≤ T.1 ∧ T.1 ≤ R.x_max ∧ 
    R.y_min ≤ T.2 ∧ T.2 ≤ R.y_max := 
sorry

end NUMINAMATH_GPT_exists_common_point_l1455_145538


namespace NUMINAMATH_GPT_bryan_total_books_magazines_l1455_145541

-- Conditions as definitions
def novels : ℕ := 90
def comics : ℕ := 160
def rooms : ℕ := 12
def x := (3 / 4 : ℚ) * novels
def y := (6 / 5 : ℚ) * comics
def z := (1 / 2 : ℚ) * rooms

-- Calculations based on conditions
def books_per_shelf := 27 * x
def magazines_per_shelf := 80 * y
def total_shelves := 23 * z
def total_books := books_per_shelf * total_shelves
def total_magazines := magazines_per_shelf * total_shelves
def grand_total := total_books + total_magazines

-- Theorem to prove
theorem bryan_total_books_magazines :
  grand_total = 2371275 := by
  sorry

end NUMINAMATH_GPT_bryan_total_books_magazines_l1455_145541


namespace NUMINAMATH_GPT_race_outcomes_l1455_145558

def participants : List String := ["Abe", "Bobby", "Charles", "Devin", "Edwin", "Fiona"]

theorem race_outcomes (h : ¬ "Fiona" ∈ ["Abe", "Bobby", "Charles", "Devin", "Edwin"]) : 
  (participants.length - 1) * (participants.length - 2) * (participants.length - 3) = 60 :=
by
  sorry

end NUMINAMATH_GPT_race_outcomes_l1455_145558


namespace NUMINAMATH_GPT_find_other_number_l1455_145533

theorem find_other_number (A B : ℕ) (h_lcm : Nat.lcm A B = 192) (h_hcf : Nat.gcd A B = 16) (h_A : A = 48) : B = 64 :=
by
  sorry

end NUMINAMATH_GPT_find_other_number_l1455_145533


namespace NUMINAMATH_GPT_equation_of_parallel_line_l1455_145519

theorem equation_of_parallel_line (l : ℝ → ℝ → Prop) (P : ℝ × ℝ)
  (x y : ℝ) (m : ℝ) (H_1 : P = (1, 2)) (H_2 : ∀ x y m, l x y ↔ (2 * x + y + m = 0) )
  (H_3 : l x y) : 
  l 2 (y - 4) := 
  sorry

end NUMINAMATH_GPT_equation_of_parallel_line_l1455_145519


namespace NUMINAMATH_GPT_domain_of_function_l1455_145540

noncomputable def domain_f (x : ℝ) : Prop :=
  -x^2 + 2 * x + 3 > 0 ∧ 1 - x > 0 ∧ x ≠ 0

theorem domain_of_function :
  {x : ℝ | domain_f x} = {x : ℝ | -1 < x ∧ x < 1 ∧ x ≠ 0} :=
by
  sorry

end NUMINAMATH_GPT_domain_of_function_l1455_145540


namespace NUMINAMATH_GPT_line_passes_through_center_l1455_145542

theorem line_passes_through_center (a : ℝ) :
  (∀ x y : ℝ, x^2 + y^2 + 2 * x - 4 * y = 0 → 3 * x + y + a = 0) → a = 1 :=
by
  sorry

end NUMINAMATH_GPT_line_passes_through_center_l1455_145542


namespace NUMINAMATH_GPT_min_value_of_one_over_a_and_one_over_b_l1455_145592

noncomputable def minValue (a b : ℝ) : ℝ :=
  if 2 * a + 3 * b = 1 then 1 / a + 1 / b else 0

theorem min_value_of_one_over_a_and_one_over_b :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ 2 * a + 3 * b = 1 ∧ minValue a b = 65 / 6 :=
by
  sorry

end NUMINAMATH_GPT_min_value_of_one_over_a_and_one_over_b_l1455_145592


namespace NUMINAMATH_GPT_sum_of_products_is_70_l1455_145598

theorem sum_of_products_is_70 (a b c : ℝ) (h1 : a^2 + b^2 + c^2 = 149) (h2 : a + b + c = 17) :
  a * b + b * c + c * a = 70 :=
by
  sorry 

end NUMINAMATH_GPT_sum_of_products_is_70_l1455_145598


namespace NUMINAMATH_GPT_cost_of_pen_is_51_l1455_145512

-- Definitions of variables and conditions
variables {p q : ℕ}
variables (h1 : 6 * p + 2 * q = 348)
variables (h2 : 3 * p + 4 * q = 234)

-- Goal: Prove the cost of a pen (p) is 51 cents
theorem cost_of_pen_is_51 : p = 51 :=
by
  -- placeholder for the proof
  sorry

end NUMINAMATH_GPT_cost_of_pen_is_51_l1455_145512


namespace NUMINAMATH_GPT_jellybeans_needed_l1455_145595

-- Define the initial conditions as constants
def jellybeans_per_large_glass := 50
def jellybeans_per_small_glass := jellybeans_per_large_glass / 2
def number_of_large_glasses := 5
def number_of_small_glasses := 3

-- Calculate the total number of jellybeans needed
def total_jellybeans : ℕ :=
  (number_of_large_glasses * jellybeans_per_large_glass) + 
  (number_of_small_glasses * jellybeans_per_small_glass)

-- Prove that the total number of jellybeans needed is 325
theorem jellybeans_needed : total_jellybeans = 325 :=
sorry

end NUMINAMATH_GPT_jellybeans_needed_l1455_145595
