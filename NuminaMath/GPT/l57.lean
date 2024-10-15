import Mathlib

namespace NUMINAMATH_GPT_number_of_moles_H2SO4_formed_l57_5700

-- Define the moles of reactants
def initial_moles_SO2 : ℕ := 1
def initial_moles_H2O2 : ℕ := 1

-- Given the balanced chemical reaction
-- SO2 + H2O2 → H2SO4
def balanced_reaction := (1, 1) -- Representing the reactant coefficients for SO2 and H2O2

-- Define the number of moles of product formed
def moles_H2SO4 (moles_SO2 moles_H2O2 : ℕ) : ℕ :=
moles_SO2 -- Since according to balanced equation, 1 mole of each reactant produces 1 mole of product

theorem number_of_moles_H2SO4_formed :
  moles_H2SO4 initial_moles_SO2 initial_moles_H2O2 = 1 := by
  sorry

end NUMINAMATH_GPT_number_of_moles_H2SO4_formed_l57_5700


namespace NUMINAMATH_GPT_find_b_value_l57_5768

theorem find_b_value
    (k1 k2 b : ℝ)
    (y1 y2 : ℝ → ℝ)
    (a n : ℝ)
    (h1 : ∀ x, y1 x = k1 / x)
    (h2 : ∀ x, y2 x = k2 * x + b)
    (intersection_A : y1 1 = 4)
    (intersection_B : y2 a = 1 ∧ y1 a = 1)
    (translated_C_y1 : y1 (-1) = n + 6)
    (translated_C_y2 : y2 1 = n)
    (k1k2_nonzero : k1 ≠ 0 ∧ k2 ≠ 0)
    (sum_k1_k2 : k1 + k2 = 0) :
  b = -6 :=
sorry

end NUMINAMATH_GPT_find_b_value_l57_5768


namespace NUMINAMATH_GPT_toothbrush_count_l57_5769

theorem toothbrush_count (T A : ℕ) (h1 : 53 + 67 + 46 = 166)
  (h2 : 67 - 36 = 31) (h3 : A = 31) (h4 : T = 166 + 2 * A) :
  T = 228 :=
  by 
  -- Using Lean's sorry keyword to skip the proof
  sorry

end NUMINAMATH_GPT_toothbrush_count_l57_5769


namespace NUMINAMATH_GPT_find_58th_digit_in_fraction_l57_5770

def decimal_representation_of_fraction : ℕ := sorry

theorem find_58th_digit_in_fraction:
  (decimal_representation_of_fraction = 4) := sorry

end NUMINAMATH_GPT_find_58th_digit_in_fraction_l57_5770


namespace NUMINAMATH_GPT_min_value_x_squared_plus_6x_l57_5707

theorem min_value_x_squared_plus_6x : ∀ x : ℝ, ∃ y : ℝ, y ≤ x^2 + 6*x ∧ y = -9 := 
sorry

end NUMINAMATH_GPT_min_value_x_squared_plus_6x_l57_5707


namespace NUMINAMATH_GPT_boxes_per_day_l57_5713

theorem boxes_per_day (apples_per_box fewer_apples_per_day total_apples_two_weeks : ℕ)
  (h1 : apples_per_box = 40)
  (h2 : fewer_apples_per_day = 500)
  (h3 : total_apples_two_weeks = 24500) :
  (∃ x : ℕ, (7 * apples_per_box * x + 7 * (apples_per_box * x - fewer_apples_per_day) = total_apples_two_weeks) ∧ x = 50) := 
sorry

end NUMINAMATH_GPT_boxes_per_day_l57_5713


namespace NUMINAMATH_GPT_truck_travel_l57_5717

/-- If a truck travels 150 miles using 5 gallons of diesel, then it will travel 210 miles using 7 gallons of diesel. -/
theorem truck_travel (d1 d2 g1 g2 : ℕ) (h1 : d1 = 150) (h2 : g1 = 5) (h3 : g2 = 7) (h4 : d2 = d1 * g2 / g1) : d2 = 210 := by
  sorry

end NUMINAMATH_GPT_truck_travel_l57_5717


namespace NUMINAMATH_GPT_exists_point_P_equal_distance_squares_l57_5767

-- Define the points in the plane representing the vertices of the triangles
variables {A1 A2 A3 B1 B2 B3 C1 C2 C3 : ℝ × ℝ}
-- Define the function that calculates the square distance between two points
def sq_distance (P Q : ℝ × ℝ) : ℝ :=
  (P.1 - Q.1) ^ 2 + (P.2 - Q.2) ^ 2

-- Define the proof statement
theorem exists_point_P_equal_distance_squares :
  ∃ P : ℝ × ℝ,
    sq_distance P A1 + sq_distance P A2 + sq_distance P A3 =
    sq_distance P B1 + sq_distance P B2 + sq_distance P B3 ∧
    sq_distance P A1 + sq_distance P A2 + sq_distance P A3 =
    sq_distance P C1 + sq_distance P C2 + sq_distance P C3 := sorry

end NUMINAMATH_GPT_exists_point_P_equal_distance_squares_l57_5767


namespace NUMINAMATH_GPT_angle_sum_and_relation_l57_5792

variable {A B : ℝ}

theorem angle_sum_and_relation (h1 : A + B = 180) (h2 : A = 5 * B) : A = 150 := by
  sorry

end NUMINAMATH_GPT_angle_sum_and_relation_l57_5792


namespace NUMINAMATH_GPT_find_b_l57_5790

-- Definitions
def quadratic (x b c : ℝ) : ℝ := x^2 + b * x + c

theorem find_b (b c : ℝ) 
  (h_diff : ∀ x : ℝ, 1 ≤ x ∧ x ≤ 7 → (∀ y : ℝ, 1 ≤ y ∧ y ≤ 7 → quadratic x b c - quadratic y b c = 25)) :
  b = -4 ∨ b = -12 :=
by sorry

end NUMINAMATH_GPT_find_b_l57_5790


namespace NUMINAMATH_GPT_ratio_of_female_contestants_l57_5793

theorem ratio_of_female_contestants (T M F : ℕ) (hT : T = 18) (hM : M = 12) (hF : F = T - M) :
  F / T = 1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_female_contestants_l57_5793


namespace NUMINAMATH_GPT_duration_of_investment_l57_5785

-- Define the constants as given in the conditions
def Principal : ℝ := 7200
def Rate : ℝ := 17.5
def SimpleInterest : ℝ := 3150

-- Define the time variable we want to prove
def Time : ℝ := 2.5

-- Prove that the calculated time matches the expected value
theorem duration_of_investment :
  SimpleInterest = (Principal * Rate * Time) / 100 :=
sorry

end NUMINAMATH_GPT_duration_of_investment_l57_5785


namespace NUMINAMATH_GPT_proof_rewritten_eq_and_sum_l57_5736

-- Define the given equation
def given_eq (x : ℝ) : Prop := 64 * x^2 + 80 * x - 72 = 0

-- Define the rewritten form of the equation
def rewritten_eq (x : ℝ) : Prop := (8 * x + 5)^2 = 97

-- Define the correctness of rewriting the equation
def correct_rewrite (x : ℝ) : Prop :=
  given_eq x → rewritten_eq x

-- Define the correct value of a + b + c
def correct_sum : Prop :=
  8 + 5 + 97 = 110

-- The final theorem statement
theorem proof_rewritten_eq_and_sum (x : ℝ) : correct_rewrite x ∧ correct_sum :=
by
  sorry

end NUMINAMATH_GPT_proof_rewritten_eq_and_sum_l57_5736


namespace NUMINAMATH_GPT_distance_between_houses_l57_5722

theorem distance_between_houses
  (alice_speed : ℕ) (bob_speed : ℕ) (alice_distance : ℕ) 
  (alice_walk_time : ℕ) (bob_walk_time : ℕ)
  (alice_start : ℕ) (bob_start : ℕ)
  (bob_start_after_alice : bob_start = alice_start + 1)
  (alice_speed_eq : alice_speed = 5)
  (bob_speed_eq : bob_speed = 4)
  (alice_distance_eq : alice_distance = 25)
  (alice_walk_time_eq : alice_walk_time = alice_distance / alice_speed)
  (bob_walk_time_eq : bob_walk_time = alice_walk_time - 1)
  (bob_distance_eq : bob_walk_time = bob_walk_time * bob_speed)
  (total_distance : ℕ)
  (total_distance_eq : total_distance = alice_distance + bob_distance) :
  total_distance = 41 :=
by sorry

end NUMINAMATH_GPT_distance_between_houses_l57_5722


namespace NUMINAMATH_GPT_athlete_total_heartbeats_l57_5775

/-
  An athlete's heart rate starts at 140 beats per minute at the beginning of a race
  and increases by 5 beats per minute for each subsequent mile. How many times does
  the athlete's heart beat during a 10-mile race if the athlete runs at a pace of
  6 minutes per mile?
-/

def athlete_heartbeats (initial_rate : ℕ) (increase_rate : ℕ) (miles : ℕ) (minutes_per_mile : ℕ) : ℕ :=
  let n := miles
  let a := initial_rate
  let l := initial_rate + (increase_rate * (miles - 1))
  let S := (n * (a + l)) / 2
  S * minutes_per_mile

theorem athlete_total_heartbeats :
  athlete_heartbeats 140 5 10 6 = 9750 :=
sorry

end NUMINAMATH_GPT_athlete_total_heartbeats_l57_5775


namespace NUMINAMATH_GPT_muffins_division_l57_5725

theorem muffins_division (total_muffins total_people muffins_per_person : ℕ) 
  (h1 : total_muffins = 20) (h2 : total_people = 5) (h3 : muffins_per_person = total_muffins / total_people) : 
  muffins_per_person = 4 := 
by
  sorry

end NUMINAMATH_GPT_muffins_division_l57_5725


namespace NUMINAMATH_GPT_crumbs_triangle_area_l57_5729

theorem crumbs_triangle_area :
  ∀ (table_length table_width : ℝ) (crumbs : ℕ),
    table_length = 2 ∧ table_width = 1 ∧ crumbs = 500 →
    ∃ (triangle_area : ℝ), (triangle_area < 0.005 ∧ ∃ (a b c : Type), a ≠ b ∧ b ≠ c ∧ a ≠ c) :=
by
  sorry

end NUMINAMATH_GPT_crumbs_triangle_area_l57_5729


namespace NUMINAMATH_GPT_balls_in_drawers_l57_5760

theorem balls_in_drawers (n k : ℕ) (h_n : n = 5) (h_k : k = 2) : (k ^ n) = 32 :=
by
  rw [h_n, h_k]
  sorry

end NUMINAMATH_GPT_balls_in_drawers_l57_5760


namespace NUMINAMATH_GPT_combine_monomials_x_plus_y_l57_5798

theorem combine_monomials_x_plus_y : ∀ (x y : ℤ),
  7 * x = 2 - 4 * y →
  y + 7 = 2 * x →
  x + y = -1 :=
by
  intros x y h1 h2
  sorry

end NUMINAMATH_GPT_combine_monomials_x_plus_y_l57_5798


namespace NUMINAMATH_GPT_intersection_of_sets_A_B_l57_5742

def set_A : Set ℝ := { x : ℝ | x^2 - 2*x - 3 > 0 }
def set_B : Set ℝ := { x : ℝ | -2 < x ∧ x ≤ 2 }
def set_intersection : Set ℝ := { x : ℝ | -2 < x ∧ x < -1 }

theorem intersection_of_sets_A_B :
  (set_A ∩ set_B) = set_intersection :=
  sorry

end NUMINAMATH_GPT_intersection_of_sets_A_B_l57_5742


namespace NUMINAMATH_GPT_area_of_rhombus_is_375_l57_5763

-- define the given diagonals
def diagonal1 := 25
def diagonal2 := 30

-- define the formula for the area of a rhombus
def area_of_rhombus (d1 d2 : ℕ) : ℕ := (d1 * d2) / 2

-- state the theorem
theorem area_of_rhombus_is_375 : area_of_rhombus diagonal1 diagonal2 = 375 := 
by 
  -- The proof is omitted as per the requirement
  sorry

end NUMINAMATH_GPT_area_of_rhombus_is_375_l57_5763


namespace NUMINAMATH_GPT_solution_set_of_inequality_system_l57_5715

theorem solution_set_of_inequality_system (x : ℝ) :
  (x + 1 ≥ 0) ∧ (x - 2 < 0) ↔ (-1 ≤ x ∧ x < 2) :=
by
  sorry

end NUMINAMATH_GPT_solution_set_of_inequality_system_l57_5715


namespace NUMINAMATH_GPT_fraction_of_blueberry_tart_l57_5751

/-- Let total leftover tarts be 0.91.
    Let the tart filled with cherries be 0.08.
    Let the tart filled with peaches be 0.08.
    Prove that the fraction of the tart filled with blueberries is 0.75. --/
theorem fraction_of_blueberry_tart (H_total : Real) (H_cherry : Real) (H_peach : Real)
  (H1 : H_total = 0.91) (H2 : H_cherry = 0.08) (H3 : H_peach = 0.08) :
  (H_total - (H_cherry + H_peach)) = 0.75 :=
sorry

end NUMINAMATH_GPT_fraction_of_blueberry_tart_l57_5751


namespace NUMINAMATH_GPT_flour_already_put_in_l57_5796

theorem flour_already_put_in (total_flour flour_still_needed: ℕ) (h1: total_flour = 9) (h2: flour_still_needed = 6) : total_flour - flour_still_needed = 3 := 
by
  -- Here we will state the proof
  sorry

end NUMINAMATH_GPT_flour_already_put_in_l57_5796


namespace NUMINAMATH_GPT_find_sum_of_a_and_b_l57_5771

theorem find_sum_of_a_and_b (a b : ℝ) (h1 : 0.005 * a = 0.65) (h2 : 0.0125 * b = 1.04) : a + b = 213.2 :=
  sorry

end NUMINAMATH_GPT_find_sum_of_a_and_b_l57_5771


namespace NUMINAMATH_GPT_correlation_coefficient_is_one_l57_5704

noncomputable def correlation_coefficient (x_vals y_vals : List ℝ) : ℝ := sorry

theorem correlation_coefficient_is_one 
  (n : ℕ) 
  (x y : Fin n → ℝ) 
  (h1 : n ≥ 2) 
  (h2 : ∃ i j, i ≠ j ∧ x i ≠ x j) 
  (h3 : ∀ i, y i = 3 * x i + 1) : 
  correlation_coefficient (List.ofFn x) (List.ofFn y) = 1 := 
sorry

end NUMINAMATH_GPT_correlation_coefficient_is_one_l57_5704


namespace NUMINAMATH_GPT_concave_sequence_count_l57_5744

   theorem concave_sequence_count (m : ℕ) (h : 2 ≤ m) :
     ∀ b_0, (b_0 = 1 ∨ b_0 = 2) → 
     (∃ b : ℕ → ℕ, (∀ k, 2 ≤ k ∧ k ≤ m → b k + b (k - 2) ≤ 2 * b (k - 1)) → 
     (∃ S : ℕ, S ≤ 2^m)) :=
   by 
     sorry
   
end NUMINAMATH_GPT_concave_sequence_count_l57_5744


namespace NUMINAMATH_GPT_neg_p_is_true_neg_q_is_true_l57_5756

theorem neg_p_is_true : ∃ m : ℝ, ∀ x : ℝ, (x^2 + x - m = 0 → False) :=
sorry

theorem neg_q_is_true : ∀ x : ℝ, (x^2 + x + 1 > 0) :=
sorry

end NUMINAMATH_GPT_neg_p_is_true_neg_q_is_true_l57_5756


namespace NUMINAMATH_GPT_digit_b_divisible_by_5_l57_5774

theorem digit_b_divisible_by_5 (B : ℕ) (h : B = 0 ∨ B = 5) : 
  (∃ n : ℕ, (947 * 10 + B) = 5 * n) ↔ (B = 0 ∨ B = 5) :=
by {
  sorry
}

end NUMINAMATH_GPT_digit_b_divisible_by_5_l57_5774


namespace NUMINAMATH_GPT_smallest_m_for_no_real_solution_l57_5721

theorem smallest_m_for_no_real_solution : 
  (∀ x : ℝ, ∀ m : ℝ, (m * x^2 - 3 * x + 1 = 0) → false) ↔ (m ≥ 3) :=
by
  sorry

end NUMINAMATH_GPT_smallest_m_for_no_real_solution_l57_5721


namespace NUMINAMATH_GPT_hypotenuse_is_18_8_l57_5718

def right_triangle_hypotenuse_perimeter_area (a b c : ℝ) : Prop :=
  a + b + c = 40 ∧ (1/2 * a * b = 24) ∧ (a^2 + b^2 = c^2)

theorem hypotenuse_is_18_8 : ∃ (a b c : ℝ), right_triangle_hypotenuse_perimeter_area a b c ∧ c = 18.8 :=
by
  sorry

end NUMINAMATH_GPT_hypotenuse_is_18_8_l57_5718


namespace NUMINAMATH_GPT_find_b_l57_5702

theorem find_b (a b : ℕ) (h1 : a = 105) (h2 : a^3 = 21 * 25 * 45 * b) : b = 49 :=
by
  sorry

end NUMINAMATH_GPT_find_b_l57_5702


namespace NUMINAMATH_GPT_cost_price_of_watch_l57_5745

theorem cost_price_of_watch
  (C : ℝ)
  (h1 : 0.9 * C + 225 = 1.05 * C) :
  C = 1500 :=
by sorry

end NUMINAMATH_GPT_cost_price_of_watch_l57_5745


namespace NUMINAMATH_GPT_pyramid_base_edge_length_l57_5735

noncomputable def edge_length_of_pyramid_base : ℝ :=
  let R := 4 -- radius of the hemisphere
  let h := 12 -- height of the pyramid
  let base_length := 6 -- edge-length of the base of the pyramid to be proved
  -- assume necessary geometric configurations of the pyramid and sphere
  base_length

theorem pyramid_base_edge_length :
  ∀ R h base_length, R = 4 → h = 12 → edge_length_of_pyramid_base = base_length → base_length = 6 :=
by
  intros R h base_length hR hH hBaseLength
  have R_spec : R = 4 := hR
  have h_spec : h = 12 := hH
  have base_length_spec : edge_length_of_pyramid_base = base_length := hBaseLength
  sorry

end NUMINAMATH_GPT_pyramid_base_edge_length_l57_5735


namespace NUMINAMATH_GPT_find_y_of_pentagon_l57_5782

def y_coordinate (y : ℝ) : Prop :=
  let area_ABDE := 12
  let area_BCD := 2 * (y - 3)
  let total_area := area_ABDE + area_BCD
  total_area = 35

theorem find_y_of_pentagon :
  ∃ y : ℝ, y_coordinate y ∧ y = 14.5 :=
by
  sorry

end NUMINAMATH_GPT_find_y_of_pentagon_l57_5782


namespace NUMINAMATH_GPT_num_possible_lists_l57_5748

theorem num_possible_lists :
  let binA_balls := 8
  let binB_balls := 5
  let total_lists := binA_balls * binB_balls
  total_lists = 40 := by
{
  let binA_balls := 8
  let binB_balls := 5
  let total_lists := binA_balls * binB_balls
  show total_lists = 40
  exact rfl
}

end NUMINAMATH_GPT_num_possible_lists_l57_5748


namespace NUMINAMATH_GPT_square_of_leg_l57_5762

theorem square_of_leg (a c b : ℝ) (h1 : c = 2 * a + 1) (h2 : a^2 + b^2 = c^2) : b^2 = 3 * a^2 + 4 * a + 1 :=
by
  sorry

end NUMINAMATH_GPT_square_of_leg_l57_5762


namespace NUMINAMATH_GPT_find_x_value_l57_5749

theorem find_x_value (x : ℝ) (h : 0.75 / x = 5 / 6) : x = 0.9 := 
by 
  sorry

end NUMINAMATH_GPT_find_x_value_l57_5749


namespace NUMINAMATH_GPT_incorrect_option_D_l57_5739

variable {p q : Prop}

theorem incorrect_option_D (hp : ¬p) (hq : q) : ¬(¬q) := 
by 
  sorry  

end NUMINAMATH_GPT_incorrect_option_D_l57_5739


namespace NUMINAMATH_GPT_g_675_eq_42_l57_5709

-- Define the function g on positive integers
def g : ℕ → ℕ := sorry

-- State the conditions
axiom g_multiplicative : ∀ (x y : ℕ), g (x * y) = g x + g y
axiom g_15 : g 15 = 18
axiom g_45 : g 45 = 24

-- The theorem we want to prove
theorem g_675_eq_42 : g 675 = 42 := 
by 
  sorry

end NUMINAMATH_GPT_g_675_eq_42_l57_5709


namespace NUMINAMATH_GPT_max_ab_value_l57_5759

variable (a b c : ℝ)

-- Conditions
axiom h1 : 0 < a ∧ a < 1
axiom h2 : 0 < b ∧ b < 1
axiom h3 : 0 < c ∧ c < 1
axiom h4 : 3 * a + 2 * b = 1

-- Goal
theorem max_ab_value : ab = 1 / 24 :=
by
  sorry

end NUMINAMATH_GPT_max_ab_value_l57_5759


namespace NUMINAMATH_GPT_abs_five_minus_sqrt_pi_l57_5765

theorem abs_five_minus_sqrt_pi : |5 - Real.sqrt Real.pi| = 3.22755 := by
  sorry

end NUMINAMATH_GPT_abs_five_minus_sqrt_pi_l57_5765


namespace NUMINAMATH_GPT_perpendicular_lines_parallel_l57_5747

noncomputable def line := Type
noncomputable def plane := Type

variables (m n : line) (α : plane)

def parallel (l1 l2 : line) : Prop := sorry -- Definition of parallel lines
def perpendicular (l : line) (α : plane) : Prop := sorry -- Definition of perpendicular line to a plane

theorem perpendicular_lines_parallel (h1 : perpendicular m α) (h2 : perpendicular n α) : parallel m n :=
sorry

end NUMINAMATH_GPT_perpendicular_lines_parallel_l57_5747


namespace NUMINAMATH_GPT_area_of_triangles_l57_5730

theorem area_of_triangles
  (ABC_area : ℝ)
  (AD : ℝ)
  (DB : ℝ)
  (h_AD_DB : AD + DB = 7)
  (h_equal_areas : ABC_area = 12) :
  (∃ ABE_area : ℝ, ABE_area = 36 / 7) ∧ (∃ DBF_area : ℝ, DBF_area = 36 / 7) :=
by
  sorry

end NUMINAMATH_GPT_area_of_triangles_l57_5730


namespace NUMINAMATH_GPT_equivalent_statement_l57_5701

theorem equivalent_statement (x y z w : ℝ)
  (h : (2 * x + y) / (y + z) = (z + w) / (w + 2 * x)) :
  (x = z / 2 ∨ 2 * x + y + z + w = 0) :=
sorry

end NUMINAMATH_GPT_equivalent_statement_l57_5701


namespace NUMINAMATH_GPT_find_m_value_l57_5726

theorem find_m_value
  (x y : ℤ)
  (h1 : x = 2)
  (h2 : y = m)
  (h3 : 3 * x + 2 * y = 10) : 
  m = 2 :=
by
  sorry

end NUMINAMATH_GPT_find_m_value_l57_5726


namespace NUMINAMATH_GPT_min_value_expression_l57_5753

variable (a b c : ℝ)
variable (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
variable (h_eq : a * b * c = 64)

theorem min_value_expression :
  a^2 + 6 * a * b + 9 * b^2 + 3 * c^2 ≥ 192 :=
by {
  sorry
}

end NUMINAMATH_GPT_min_value_expression_l57_5753


namespace NUMINAMATH_GPT_cars_count_l57_5732

theorem cars_count
  (distance : ℕ)
  (time_between_cars : ℕ)
  (total_time_hours : ℕ)
  (cars_per_hour : ℕ)
  (expected_cars_count : ℕ) :
  distance = 3 →
  time_between_cars = 20 →
  total_time_hours = 10 →
  cars_per_hour = 3 →
  expected_cars_count = total_time_hours * cars_per_hour →
  expected_cars_count = 30 :=
by
  intros h1 h2 h3 h4 h5
  rw [h3, h4] at h5
  exact h5


end NUMINAMATH_GPT_cars_count_l57_5732


namespace NUMINAMATH_GPT_intersection_at_one_point_l57_5743

-- Define the quadratic equation derived from the intersection condition
def quadratic (y k : ℝ) : ℝ :=
  3 * y^2 - 2 * y + (k - 4)

-- Define the discriminant of the quadratic equation
def discriminant (k : ℝ) : ℝ :=
  (-2)^2 - 4 * 3 * (k - 4)

-- The statement of the problem in Lean
theorem intersection_at_one_point (k : ℝ) :
  (∃ y : ℝ, quadratic y k = 0 ∧ discriminant k = 0) ↔ k = 13 / 3 :=
by 
  sorry

end NUMINAMATH_GPT_intersection_at_one_point_l57_5743


namespace NUMINAMATH_GPT_quadratic_inequality_solution_l57_5776

theorem quadratic_inequality_solution (x : ℝ) : 
  (x - 2) * (x + 2) < 5 ↔ -3 < x ∧ x < 3 :=
by sorry

end NUMINAMATH_GPT_quadratic_inequality_solution_l57_5776


namespace NUMINAMATH_GPT_highest_a_value_l57_5746

theorem highest_a_value (a b : ℤ) (h1 : a > b) (h2 : b > 0) (h3 : a + b + a * b = 143) : a = 23 :=
sorry

end NUMINAMATH_GPT_highest_a_value_l57_5746


namespace NUMINAMATH_GPT_tom_books_l57_5779

theorem tom_books (books_may books_june books_july : ℕ) (h_may : books_may = 2) (h_june : books_june = 6) (h_july : books_july = 10) : 
books_may + books_june + books_july = 18 := by
sorry

end NUMINAMATH_GPT_tom_books_l57_5779


namespace NUMINAMATH_GPT_quadratic_has_real_root_l57_5727

theorem quadratic_has_real_root (p : ℝ) : 
  ∃ x : ℝ, 3 * (p + 2) * x^2 - p * x - (4 * p + 7) = 0 :=
sorry

end NUMINAMATH_GPT_quadratic_has_real_root_l57_5727


namespace NUMINAMATH_GPT_fifteenth_odd_multiple_of_5_is_145_l57_5783

def sequence_term (n : ℕ) : ℤ :=
  10 * n - 5

theorem fifteenth_odd_multiple_of_5_is_145 : sequence_term 15 = 145 :=
by
  sorry

end NUMINAMATH_GPT_fifteenth_odd_multiple_of_5_is_145_l57_5783


namespace NUMINAMATH_GPT_distance_between_points_l57_5766

def point1 : ℝ × ℝ := (3.5, -2)
def point2 : ℝ × ℝ := (7.5, 5)

theorem distance_between_points : 
  Real.sqrt ((point2.1 - point1.1)^2 + (point2.2 - point1.2)^2) = Real.sqrt 65 := by
  sorry

end NUMINAMATH_GPT_distance_between_points_l57_5766


namespace NUMINAMATH_GPT_checkerboard_corners_sum_l57_5786

theorem checkerboard_corners_sum : 
  let N : ℕ := 9 
  let corners := [1, 9, 73, 81]
  (corners.sum = 164) := by
  sorry

end NUMINAMATH_GPT_checkerboard_corners_sum_l57_5786


namespace NUMINAMATH_GPT_average_of_4_8_N_l57_5778

-- Define the condition for N
variable (N : ℝ) (cond : 7 < N ∧ N < 15)

-- State the theorem to prove
theorem average_of_4_8_N (N : ℝ) (h : 7 < N ∧ N < 15) :
  (frac12 + N) / 3 = 7 ∨ (12 + N) / 3 = 9 :=
sorry

end NUMINAMATH_GPT_average_of_4_8_N_l57_5778


namespace NUMINAMATH_GPT_min_value_of_sum_l57_5719

theorem min_value_of_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : (1 / (2 * a)) + (1 / b) = 1) :
  a + 2 * b = 9 / 2 :=
sorry

end NUMINAMATH_GPT_min_value_of_sum_l57_5719


namespace NUMINAMATH_GPT_other_investment_interest_rate_l57_5797

open Real

-- Definitions of the given conditions
def total_investment : ℝ := 22000
def investment_at_8_percent : ℝ := 17000
def total_interest : ℝ := 1710
def interest_rate_8_percent : ℝ := 0.08

-- Derived definitions from the conditions
def other_investment_amount : ℝ := total_investment - investment_at_8_percent
def interest_from_8_percent : ℝ := investment_at_8_percent * interest_rate_8_percent
def interest_from_other : ℝ := total_interest - interest_from_8_percent

-- Proof problem: Prove that the percentage of the other investment is 0.07 (or 7%).
theorem other_investment_interest_rate :
  interest_from_other / other_investment_amount = 0.07 := by
  sorry

end NUMINAMATH_GPT_other_investment_interest_rate_l57_5797


namespace NUMINAMATH_GPT_evaluate_f_x_l57_5711

def f (x : ℝ) : ℝ := x^5 + 3 * x^3 + 2 * x^2 + 4 * x

theorem evaluate_f_x : f 3 - f (-3) = 672 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_evaluate_f_x_l57_5711


namespace NUMINAMATH_GPT_g_increasing_g_multiplicative_g_special_case_g_18_value_l57_5705

def g (n : ℕ) : ℕ :=
sorry

theorem g_increasing : ∀ n : ℕ, n > 0 → g (n + 1) > g n :=
sorry

theorem g_multiplicative : ∀ m n : ℕ, m > 0 → n > 0 → g (m * n) = g m * g n :=
sorry

theorem g_special_case : ∀ m n : ℕ, m > 0 → n > 0 → m ≠ n → m ^ n = n ^ m → g m = n ∨ g n = m :=
sorry

theorem g_18_value : g 18 = 324 :=
sorry

end NUMINAMATH_GPT_g_increasing_g_multiplicative_g_special_case_g_18_value_l57_5705


namespace NUMINAMATH_GPT_prod_of_consecutive_nums_divisible_by_504_l57_5755

theorem prod_of_consecutive_nums_divisible_by_504
  (a : ℕ)
  (h : ∃ b : ℕ, a = b ^ 3) :
  (a^3 - 1) * a^3 * (a^3 + 1) % 504 = 0 := 
sorry

end NUMINAMATH_GPT_prod_of_consecutive_nums_divisible_by_504_l57_5755


namespace NUMINAMATH_GPT_lcm_gcd_product_l57_5708

theorem lcm_gcd_product (a b : ℕ) (ha : a = 36) (hb : b = 60) : 
  Nat.lcm a b * Nat.gcd a b = 2160 :=
by
  rw [ha, hb]
  sorry

end NUMINAMATH_GPT_lcm_gcd_product_l57_5708


namespace NUMINAMATH_GPT_intersection_of_A_and_B_l57_5728

def A : Set ℝ := { x | x^2 - x - 2 ≥ 0 }
def B : Set ℝ := { x | -2 ≤ x ∧ x < 2 }

theorem intersection_of_A_and_B :
  A ∩ B = { x | -2 ≤ x ∧ x ≤ -1 } := by
-- The proof would go here
sorry

end NUMINAMATH_GPT_intersection_of_A_and_B_l57_5728


namespace NUMINAMATH_GPT_sum_due_in_years_l57_5758

theorem sum_due_in_years 
  (D : ℕ)
  (S : ℕ)
  (r : ℚ)
  (H₁ : D = 168)
  (H₂ : S = 768)
  (H₃ : r = 14 / 100) :
  ∃ t : ℕ, t = 2 := 
by
  sorry

end NUMINAMATH_GPT_sum_due_in_years_l57_5758


namespace NUMINAMATH_GPT_inequality_solution_subset_l57_5731

theorem inequality_solution_subset {x a : ℝ} : (∀ x, |x| > a * x + 1 → x ≤ 0) ↔ a ≥ 1 :=
by sorry

end NUMINAMATH_GPT_inequality_solution_subset_l57_5731


namespace NUMINAMATH_GPT_range_of_a_l57_5706

open Real

theorem range_of_a (a : ℝ) :
  (∀ x > 0, ae^x + x + x * log x ≥ x^2) → a ≥ 1 / exp 2 :=
sorry

end NUMINAMATH_GPT_range_of_a_l57_5706


namespace NUMINAMATH_GPT_veggies_count_l57_5791

def initial_tomatoes := 500
def picked_tomatoes := 325
def initial_potatoes := 400
def picked_potatoes := 270
def initial_cucumbers := 300
def planted_cucumber_plants := 200
def cucumbers_per_plant := 2
def initial_cabbages := 100
def picked_cabbages := 50
def planted_cabbage_plants := 80
def cabbages_per_cabbage_plant := 3

noncomputable def remaining_tomatoes : Nat :=
  initial_tomatoes - picked_tomatoes

noncomputable def remaining_potatoes : Nat :=
  initial_potatoes - picked_potatoes

noncomputable def remaining_cucumbers : Nat :=
  initial_cucumbers + planted_cucumber_plants * cucumbers_per_plant

noncomputable def remaining_cabbages : Nat :=
  (initial_cabbages - picked_cabbages) + planted_cabbage_plants * cabbages_per_cabbage_plant

theorem veggies_count :
  remaining_tomatoes = 175 ∧
  remaining_potatoes = 130 ∧
  remaining_cucumbers = 700 ∧
  remaining_cabbages = 290 :=
by
  sorry

end NUMINAMATH_GPT_veggies_count_l57_5791


namespace NUMINAMATH_GPT_integer_solutions_of_prime_equation_l57_5788

theorem integer_solutions_of_prime_equation (p : ℕ) (hp : Prime p) :
  ∃ x y : ℤ, (p * (x + y) = x * y) ↔ 
    (x = (p * (p + 1)) ∧ y = (p + 1)) ∨ 
    (x = 2 * p ∧ y = 2 * p) ∨ 
    (x = 0 ∧ y = 0) ∨ 
    (x = p * (1 - p) ∧ y = (p - 1)) := 
sorry

end NUMINAMATH_GPT_integer_solutions_of_prime_equation_l57_5788


namespace NUMINAMATH_GPT_probability_of_two_mathematicians_living_contemporarily_l57_5772

noncomputable def probability_of_contemporary_lifespan : ℚ :=
  let total_area := 500 * 500
  let triangle_area := 0.5 * 380 * 380
  let non_overlap_area := 2 * triangle_area
  let overlap_area := total_area - non_overlap_area
  overlap_area / total_area

theorem probability_of_two_mathematicians_living_contemporarily :
  probability_of_contemporary_lifespan = 2232 / 5000 :=
by
  -- The actual proof would go here
  sorry

end NUMINAMATH_GPT_probability_of_two_mathematicians_living_contemporarily_l57_5772


namespace NUMINAMATH_GPT_water_wasted_per_hour_l57_5757

def drips_per_minute : ℝ := 10
def volume_per_drop : ℝ := 0.05

def drops_per_hour : ℝ := 60 * drips_per_minute
def total_volume : ℝ := drops_per_hour * volume_per_drop

theorem water_wasted_per_hour : total_volume = 30 :=
by
  sorry

end NUMINAMATH_GPT_water_wasted_per_hour_l57_5757


namespace NUMINAMATH_GPT_new_person_weight_l57_5720

variable {W : ℝ} -- Total weight of the original group of 15 people
variable {N : ℝ} -- Weight of the new person

theorem new_person_weight
  (avg_increase : (W - 90 + N) / 15 = (W - 90) / 14 + 3.7)
  : N = 55.5 :=
sorry

end NUMINAMATH_GPT_new_person_weight_l57_5720


namespace NUMINAMATH_GPT_breaks_difference_l57_5750

-- James works for 240 minutes
def total_work_time : ℕ := 240

-- He takes a water break every 20 minutes
def water_break_interval : ℕ := 20

-- He takes a sitting break every 120 minutes
def sitting_break_interval : ℕ := 120

-- Calculate the number of water breaks James takes
def number_of_water_breaks : ℕ := total_work_time / water_break_interval

-- Calculate the number of sitting breaks James takes
def number_of_sitting_breaks : ℕ := total_work_time / sitting_break_interval

-- Prove the difference between the number of water breaks and sitting breaks is 10
theorem breaks_difference :
  number_of_water_breaks - number_of_sitting_breaks = 10 :=
by
  -- calculate number_of_water_breaks = 12
  -- calculate number_of_sitting_breaks = 2
  -- check the difference 12 - 2 = 10
  sorry

end NUMINAMATH_GPT_breaks_difference_l57_5750


namespace NUMINAMATH_GPT_intersection_empty_l57_5710

def setA : Set ℝ := { x | x^2 - 2 * x > 0 }
def setB : Set ℝ := { x | |x + 1| < 0 }

theorem intersection_empty : setA ∩ setB = ∅ :=
by
  sorry

end NUMINAMATH_GPT_intersection_empty_l57_5710


namespace NUMINAMATH_GPT_largest_angle_in_pentagon_l57_5795

theorem largest_angle_in_pentagon {R S : ℝ} (h₁: R = S) 
  (h₂: (75 : ℝ) + 110 + R + S + (3 * R - 20) = 540) : 
  (3 * R - 20) = 217 :=
by {
  -- Given conditions are assigned and now we need to prove the theorem, the proof is omitted
  sorry
}

end NUMINAMATH_GPT_largest_angle_in_pentagon_l57_5795


namespace NUMINAMATH_GPT_books_sold_correct_l57_5723

-- Define the initial number of books, number of books added, and the final number of books.
def initial_books : ℕ := 41
def added_books : ℕ := 2
def final_books : ℕ := 10

-- Define the number of books sold.
def sold_books : ℕ := initial_books + added_books - final_books

-- The theorem we need to prove: the number of books sold is 33.
theorem books_sold_correct : sold_books = 33 := by
  sorry

end NUMINAMATH_GPT_books_sold_correct_l57_5723


namespace NUMINAMATH_GPT_find_x_collinear_l57_5787

theorem find_x_collinear (x : ℝ) (a : ℝ × ℝ := (2, -1)) (b : ℝ × ℝ := (x, 1)) 
  (h_collinear : ∃ k : ℝ, (2 * 2 + x) = k * x ∧ (2 * -1 + 1) = k * 1) : x = -2 :=
by
  sorry

end NUMINAMATH_GPT_find_x_collinear_l57_5787


namespace NUMINAMATH_GPT_cost_to_paint_cube_l57_5761

def side_length := 30 -- in feet
def cost_per_kg := 40 -- Rs. per kg
def coverage_per_kg := 20 -- sq. ft. per kg

def area_of_one_face := side_length * side_length
def total_surface_area := 6 * area_of_one_face
def paint_required := total_surface_area / coverage_per_kg
def total_cost := paint_required * cost_per_kg

theorem cost_to_paint_cube : total_cost = 10800 := 
by
  -- proof here would follow the solution steps provided in the solution part, which are omitted
  sorry

end NUMINAMATH_GPT_cost_to_paint_cube_l57_5761


namespace NUMINAMATH_GPT_employee_n_salary_l57_5773

theorem employee_n_salary (x : ℝ) (h : x + 1.2 * x = 583) : x = 265 := sorry

end NUMINAMATH_GPT_employee_n_salary_l57_5773


namespace NUMINAMATH_GPT_rectangle_area_ratio_l57_5764

theorem rectangle_area_ratio (a b c d : ℝ) 
  (h1 : a / c = 3 / 4) (h2 : b / d = 3 / 4) :
  (a * b) / (c * d) = 9 / 16 := 
  sorry

end NUMINAMATH_GPT_rectangle_area_ratio_l57_5764


namespace NUMINAMATH_GPT_find_lengths_of_segments_l57_5799

variable (b c : ℝ)

theorem find_lengths_of_segments (CK AK AB CT AC AT : ℝ)
  (h1 : CK = AK + AB)
  (h2 : CK = (b + c) / 2)
  (h3 : CT = AC - AT)
  (h4 : AC = b) :
  AT = (b + c) / 2 ∧ CT = (b - c) / 2 := 
sorry

end NUMINAMATH_GPT_find_lengths_of_segments_l57_5799


namespace NUMINAMATH_GPT_probability_at_least_one_8_l57_5789

theorem probability_at_least_one_8 (n : ℕ) (hn : n = 8) : 
  (1 - (7/8) * (7/8)) = 15 / 64 :=
by
  rw [← hn]
  sorry

end NUMINAMATH_GPT_probability_at_least_one_8_l57_5789


namespace NUMINAMATH_GPT_remainder_addition_l57_5780

theorem remainder_addition (m : ℕ) (k : ℤ) (h : m = 9 * k + 4) : (m + 2025) % 9 = 4 := by
  sorry

end NUMINAMATH_GPT_remainder_addition_l57_5780


namespace NUMINAMATH_GPT_order_y1_y2_y3_l57_5754

-- Defining the parabolic function and the points A, B, C
def parabola (a x : ℝ) : ℝ :=
  a * x^2 - 2 * a * x + 3

-- Points A, B, C
def y1 (a : ℝ) : ℝ := parabola a (-1)
def y2 (a : ℝ) : ℝ := parabola a 2
def y3 (a : ℝ) : ℝ := parabola a 4

-- Assumption: a > 0
variables (a : ℝ) (h : a > 0)

-- The theorem to prove
theorem order_y1_y2_y3 : 
  y2 a < y1 a ∧ y1 a < y3 a :=
sorry

end NUMINAMATH_GPT_order_y1_y2_y3_l57_5754


namespace NUMINAMATH_GPT_max_initial_value_seq_l57_5794

theorem max_initial_value_seq :
  ∀ (x : Fin 1996 → ℝ),
    (∀ i : Fin 1996, 1 ≤ x i) →
    (x 0 = x 1995) →
    (∀ i : Fin 1995, x i + 2 / x i = 2 * x (i + 1) + 1 / x (i + 1)) →
    x 0 ≤ 2 ^ 997 :=
sorry

end NUMINAMATH_GPT_max_initial_value_seq_l57_5794


namespace NUMINAMATH_GPT_chicken_nuggets_cost_l57_5703

theorem chicken_nuggets_cost :
  ∀ (nuggets_ordered boxes_cost : ℕ) (nuggets_per_box : ℕ),
  nuggets_ordered = 100 →
  nuggets_per_box = 20 →
  boxes_cost = 4 →
  (nuggets_ordered / nuggets_per_box) * boxes_cost = 20 :=
by
  intros nuggets_ordered boxes_cost nuggets_per_box h1 h2 h3
  sorry

end NUMINAMATH_GPT_chicken_nuggets_cost_l57_5703


namespace NUMINAMATH_GPT_find_c_l57_5734

-- Define the function f(x)
def f (x c : ℝ) : ℝ := x * (x - c) ^ 2

-- Define the first derivative of f(x)
def f_prime (x c : ℝ) : ℝ := 3 * x ^ 2 - 4 * c * x + c ^ 2

-- Define the condition that f(x) has a local maximum at x = 2
def is_local_max (f' : ℝ → ℝ) (x0 : ℝ) : Prop :=
  f' x0 = 0 ∧ (∀ x, x < x0 → f' x > 0) ∧ (∀ x, x > x0 → f' x < 0)

-- The main theorem stating the equivalent proof problem
theorem find_c (c : ℝ) : is_local_max (f_prime 2) 2 → c = 6 := 
  sorry

end NUMINAMATH_GPT_find_c_l57_5734


namespace NUMINAMATH_GPT_horse_revolutions_l57_5784

-- Defining the problem conditions
def radius_outer : ℝ := 30
def radius_inner : ℝ := 10
def revolutions_outer : ℕ := 25

-- The question we need to prove
theorem horse_revolutions :
  (revolutions_outer : ℝ) * (radius_outer / radius_inner) = 75 := 
by
  sorry

end NUMINAMATH_GPT_horse_revolutions_l57_5784


namespace NUMINAMATH_GPT_cos_double_angle_unit_circle_l57_5781

theorem cos_double_angle_unit_circle (α y₀ : ℝ) (h : (1/2)^2 + y₀^2 = 1) : 
  Real.cos (2 * α) = -1/2 :=
by 
  -- The proof is omitted
  sorry

end NUMINAMATH_GPT_cos_double_angle_unit_circle_l57_5781


namespace NUMINAMATH_GPT_calculate_value_l57_5752

theorem calculate_value : (245^2 - 225^2) / 20 = 470 :=
by
  sorry

end NUMINAMATH_GPT_calculate_value_l57_5752


namespace NUMINAMATH_GPT_solve_system_l57_5733

theorem solve_system : ∃ x y : ℝ, 2 * x - y = 3 ∧ 3 * x + 2 * y = 8 ∧ x = 2 ∧ y = 1 :=
by
  sorry

end NUMINAMATH_GPT_solve_system_l57_5733


namespace NUMINAMATH_GPT_p_necessary_not_sufficient_for_q_l57_5777

open Classical

variable (p q : Prop)

theorem p_necessary_not_sufficient_for_q (h1 : ¬(p → q)) (h2 : ¬q → ¬p) : (¬(p → q) ∧ (¬q → ¬p) ∧ (¬p → ¬q ∧ ¬(¬q → p))) := by
  sorry

end NUMINAMATH_GPT_p_necessary_not_sufficient_for_q_l57_5777


namespace NUMINAMATH_GPT_total_people_participated_l57_5712

theorem total_people_participated 
  (N f p : ℕ)
  (h1 : N = f * p)
  (h2 : N = (f - 10) * (p + 1))
  (h3 : N = (f - 25) * (p + 3)) : 
  N = 900 :=
by 
  sorry

end NUMINAMATH_GPT_total_people_participated_l57_5712


namespace NUMINAMATH_GPT_edward_garage_sale_games_l57_5738

variables
  (G_total : ℕ) -- total number of games
  (G_good : ℕ) -- number of good games
  (G_bad : ℕ) -- number of bad games
  (G_friend : ℕ) -- number of games bought from a friend
  (G_garage : ℕ) -- number of games bought at the garage sale

-- The conditions
def total_games (G_total : ℕ) (G_good : ℕ) (G_bad : ℕ) : Prop :=
  G_total = G_good + G_bad

def garage_sale_games (G_total : ℕ) (G_friend : ℕ) (G_garage : ℕ) : Prop :=
  G_total = G_friend + G_garage

-- The theorem to be proved
theorem edward_garage_sale_games
  (G_total : ℕ) 
  (G_good : ℕ) 
  (G_bad : ℕ)
  (G_friend : ℕ) 
  (G_garage : ℕ) 
  (h1 : total_games G_total G_good G_bad)
  (h2 : G_good = 24)
  (h3 : G_bad = 31)
  (h4 : G_friend = 41) :
  G_garage = 14 :=
by
  sorry

end NUMINAMATH_GPT_edward_garage_sale_games_l57_5738


namespace NUMINAMATH_GPT_min_value_of_expr_l57_5724

-- Define the expression
def expr (x y : ℝ) : ℝ := (x * y + 1)^2 + (x - y)^2

-- Statement to prove that the minimum value of the expression is 1
theorem min_value_of_expr : ∃ x y : ℝ, expr x y = 1 ∧ ∀ a b : ℝ, expr a b ≥ 1 :=
by
  -- Here the proof would be provided, but we leave it as sorry as per instructions.
  sorry

end NUMINAMATH_GPT_min_value_of_expr_l57_5724


namespace NUMINAMATH_GPT_fraction_ordering_l57_5714

theorem fraction_ordering :
  (4 / 13) < (12 / 37) ∧ (12 / 37) < (15 / 31) ∧ (4 / 13) < (15 / 31) :=
by sorry

end NUMINAMATH_GPT_fraction_ordering_l57_5714


namespace NUMINAMATH_GPT_range_of_m_l57_5716

theorem range_of_m (m : ℝ) : (∀ x : ℝ, ∀ y : ℝ, (2 ≤ x ∧ x ≤ 3) → (3 ≤ y ∧ y ≤ 6) → m * x^2 - x * y + y^2 ≥ 0) ↔ (m ≥ 0) :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l57_5716


namespace NUMINAMATH_GPT_frac_wx_l57_5741

theorem frac_wx (x y z w : ℚ) (h1 : x / y = 5) (h2 : y / z = 1 / 2) (h3 : z / w = 7) : w / x = 2 / 35 :=
by
  sorry

end NUMINAMATH_GPT_frac_wx_l57_5741


namespace NUMINAMATH_GPT_visitors_not_enjoyed_not_understood_l57_5737

theorem visitors_not_enjoyed_not_understood (V E U : ℕ) (hv_v : V = 520)
  (hu_e : E = U) (he : E = 3 * V / 4) : (V / 4) = 130 :=
by
  rw [hv_v] at he
  sorry

end NUMINAMATH_GPT_visitors_not_enjoyed_not_understood_l57_5737


namespace NUMINAMATH_GPT_oxygen_atom_count_l57_5740

-- Definitions and conditions
def molecular_weight_C : ℝ := 12.01
def molecular_weight_H : ℝ := 1.008
def molecular_weight_O : ℝ := 16.00

def num_carbon_atoms : ℕ := 4
def num_hydrogen_atoms : ℕ := 1
def total_molecular_weight : ℝ := 65.0

-- Theorem statement
theorem oxygen_atom_count : 
  ∃ (num_oxygen_atoms : ℕ), 
  num_oxygen_atoms * molecular_weight_O = total_molecular_weight - (num_carbon_atoms * molecular_weight_C + num_hydrogen_atoms * molecular_weight_H) 
  ∧ num_oxygen_atoms = 1 :=
by
  sorry

end NUMINAMATH_GPT_oxygen_atom_count_l57_5740
