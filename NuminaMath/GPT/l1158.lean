import Mathlib

namespace NUMINAMATH_GPT_min_value_of_quadratic_l1158_115844

theorem min_value_of_quadratic :
  ∀ x : ℝ, ∃ z : ℝ, z = 4 * x^2 + 8 * x + 16 ∧ z ≥ 12 ∧ (∀ z' : ℝ, (z' = 4 * x^2 + 8 * x + 16) → z' ≥ 12) :=
by
  sorry

end NUMINAMATH_GPT_min_value_of_quadratic_l1158_115844


namespace NUMINAMATH_GPT_total_cost_is_eight_x_l1158_115806

-- Definitions of cost variables based on conditions
variable (x : ℝ) -- Cost of shorts

-- Cost conditions
variable (shirt_cost : ℝ) (boot_cost : ℝ) (shin_guard_cost : ℝ)
variable (c1 : x + shirt_cost = 2 * x)
variable (c2 : x + boot_cost = 5 * x)
variable (c3 : x + shin_guard_cost = 3 * x)

-- To prove that the total cost is 8 times the cost of shorts
theorem total_cost_is_eight_x
  (c1 : x + shirt_cost = 2 * x)
  (c2 : x + boot_cost = 5 * x)
  (c3 : x + shin_guard_cost = 3 * x) :
  x + shirt_cost + boot_cost + shin_guard_cost = 8 * x := 
by
  sorry

end NUMINAMATH_GPT_total_cost_is_eight_x_l1158_115806


namespace NUMINAMATH_GPT_equation_solutions_l1158_115828

theorem equation_solutions (x : ℝ) : x * (2 * x + 1) = 2 * x + 1 ↔ x = -1 / 2 ∨ x = 1 :=
by
  sorry

end NUMINAMATH_GPT_equation_solutions_l1158_115828


namespace NUMINAMATH_GPT_total_dog_food_per_day_l1158_115878

-- Definitions based on conditions
def dog1_eats_per_day : ℝ := 0.125
def dog2_eats_per_day : ℝ := 0.125
def number_of_dogs : ℕ := 2

-- Mathematically equivalent proof problem statement
theorem total_dog_food_per_day : dog1_eats_per_day + dog2_eats_per_day = 0.25 := 
by
  sorry

end NUMINAMATH_GPT_total_dog_food_per_day_l1158_115878


namespace NUMINAMATH_GPT_sum_of_cubes_eq_neg2_l1158_115820

theorem sum_of_cubes_eq_neg2 (a b : ℝ) (h1 : a + b = 1) (h2 : a * b = 1) : a^3 + b^3 = -2 := 
sorry

end NUMINAMATH_GPT_sum_of_cubes_eq_neg2_l1158_115820


namespace NUMINAMATH_GPT_complement_of_union_in_U_l1158_115819

-- Define the universal set U
def U : Set ℕ := {x | x < 6 ∧ x > 0}

-- Define the sets A and B
def A : Set ℕ := {1, 3}
def B : Set ℕ := {3, 5}

-- The complement of A ∪ B in U
def complement_U_union_A_B : Set ℕ := {x | x ∈ U ∧ x ∉ (A ∪ B)}

theorem complement_of_union_in_U : complement_U_union_A_B = {2, 4} :=
by {
  -- Placeholder for the proof
  sorry
}

end NUMINAMATH_GPT_complement_of_union_in_U_l1158_115819


namespace NUMINAMATH_GPT_positive_root_exists_iff_m_eq_neg_one_l1158_115892

theorem positive_root_exists_iff_m_eq_neg_one :
  (∃ x : ℝ, x > 0 ∧ (x / (x - 1) - m / (1 - x) = 2)) ↔ m = -1 :=
by
  sorry

end NUMINAMATH_GPT_positive_root_exists_iff_m_eq_neg_one_l1158_115892


namespace NUMINAMATH_GPT_mixed_bag_cost_l1158_115838

def cost_per_pound_colombian : ℝ := 5.5
def cost_per_pound_peruvian : ℝ := 4.25
def total_weight : ℝ := 40
def weight_colombian : ℝ := 28.8

noncomputable def cost_per_pound_mixed_bag : ℝ :=
  (weight_colombian * cost_per_pound_colombian + (total_weight - weight_colombian) * cost_per_pound_peruvian) / total_weight

theorem mixed_bag_cost :
  cost_per_pound_mixed_bag = 5.15 :=
  sorry

end NUMINAMATH_GPT_mixed_bag_cost_l1158_115838


namespace NUMINAMATH_GPT_pow_mod_remainder_l1158_115836

theorem pow_mod_remainder (n : ℕ) : 5 ^ 2023 % 11 = 4 :=
by sorry

end NUMINAMATH_GPT_pow_mod_remainder_l1158_115836


namespace NUMINAMATH_GPT_smallest_integer_remainder_l1158_115889

theorem smallest_integer_remainder (n : ℕ) 
  (h5 : n ≡ 1 [MOD 5]) (h7 : n ≡ 1 [MOD 7]) (h8 : n ≡ 1 [MOD 8]) :
  80 < n ∧ n < 299 := 
sorry

end NUMINAMATH_GPT_smallest_integer_remainder_l1158_115889


namespace NUMINAMATH_GPT_minimum_abs_ab_l1158_115884

theorem minimum_abs_ab (a b : ℝ) (h : (a^2) * (b / (a^2 + 1)) = 1) : abs (a * b) = 2 := 
  sorry

end NUMINAMATH_GPT_minimum_abs_ab_l1158_115884


namespace NUMINAMATH_GPT_cash_realized_before_brokerage_l1158_115853

theorem cash_realized_before_brokerage (C : ℝ) (h1 : 0.25 / 100 * C = C / 400)
(h2 : C - C / 400 = 108) : C = 108.27 :=
by
  sorry

end NUMINAMATH_GPT_cash_realized_before_brokerage_l1158_115853


namespace NUMINAMATH_GPT_people_in_house_l1158_115885

theorem people_in_house 
  (charlie_and_susan : ℕ := 2)
  (sarah_and_friends : ℕ := 5)
  (living_room_people : ℕ := 8) :
  (charlie_and_susan + sarah_and_friends) + living_room_people = 15 := 
by
  sorry

end NUMINAMATH_GPT_people_in_house_l1158_115885


namespace NUMINAMATH_GPT_ratio_a_b_c_l1158_115874

theorem ratio_a_b_c (a b c : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) 
  (h4 : (a + b + c) / 3 = 42) (h5 : a = 28) : 
  ∃ y z : ℕ, a / 28 = 1 ∧ b / (ky) = 1 / k ∧ c / (kz) = 1 / k ∧ (b + c) = 98 :=
by sorry

end NUMINAMATH_GPT_ratio_a_b_c_l1158_115874


namespace NUMINAMATH_GPT_point_relationship_on_parabola_neg_x_plus_1_sq_5_l1158_115814

theorem point_relationship_on_parabola_neg_x_plus_1_sq_5
  (y_1 y_2 y_3 : ℝ) :
  (A : ℝ × ℝ) = (-2, y_1) →
  (B : ℝ × ℝ) = (1, y_2) →
  (C : ℝ × ℝ) = (2, y_3) →
  (A.2 = -(A.1 + 1)^2 + 5) →
  (B.2 = -(B.1 + 1)^2 + 5) →
  (C.2 = -(C.1 + 1)^2 + 5) →
  y_1 > y_2 ∧ y_2 > y_3 :=
by
  sorry

end NUMINAMATH_GPT_point_relationship_on_parabola_neg_x_plus_1_sq_5_l1158_115814


namespace NUMINAMATH_GPT_walk_two_dogs_for_7_minutes_l1158_115895

variable (x : ℕ)

def charge_per_dog : ℕ := 20
def charge_per_minute_per_dog : ℕ := 1
def total_earnings : ℕ := 171

def charge_one_dog := charge_per_dog + charge_per_minute_per_dog * 10
def charge_three_dogs := charge_per_dog * 3 + charge_per_minute_per_dog * 9 * 3
def charge_two_dogs (x : ℕ) := charge_per_dog * 2 + charge_per_minute_per_dog * x * 2

theorem walk_two_dogs_for_7_minutes 
  (h1 : charge_one_dog = 30)
  (h2 : charge_three_dogs = 87)
  (h3 : charge_one_dog + charge_three_dogs + charge_two_dogs x = total_earnings) : 
  x = 7 :=
by
  unfold charge_one_dog charge_three_dogs charge_per_dog charge_per_minute_per_dog total_earnings at *
  sorry

end NUMINAMATH_GPT_walk_two_dogs_for_7_minutes_l1158_115895


namespace NUMINAMATH_GPT_projection_of_AB_onto_CD_l1158_115855

noncomputable def A : ℝ × ℝ := (-1, 1)
noncomputable def B : ℝ × ℝ := (1, 2)
noncomputable def C : ℝ × ℝ := (-2, -1)
noncomputable def D : ℝ × ℝ := (3, 4)

noncomputable def vector_sub (p1 p2 : ℝ × ℝ) : ℝ × ℝ :=
  (p2.1 - p1.1, p2.2 - p1.2)

noncomputable def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

noncomputable def magnitude (v : ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1 * v.1 + v.2 * v.2)

theorem projection_of_AB_onto_CD :
  let AB := vector_sub A B
  let CD := vector_sub C D
  (magnitude AB) * (dot_product AB CD) / (magnitude CD) ^ 2 = 3 * Real.sqrt 2 / 2 :=
by
  sorry

end NUMINAMATH_GPT_projection_of_AB_onto_CD_l1158_115855


namespace NUMINAMATH_GPT_range_of_a_l1158_115839

theorem range_of_a (x : ℝ) (a : ℝ) (h₀ : x ∈ Set.Icc (-2 : ℝ) 3)
(h₁ : 2 * x - x ^ 2 ≥ a) : a ≤ 1 :=
sorry

end NUMINAMATH_GPT_range_of_a_l1158_115839


namespace NUMINAMATH_GPT_alice_speed_exceeds_l1158_115876

theorem alice_speed_exceeds (distance : ℕ) (v_bob : ℕ) (time_diff : ℕ) (v_alice : ℕ)
  (h_distance : distance = 220)
  (h_v_bob : v_bob = 40)
  (h_time_diff : time_diff = 1/2) : 
  v_alice > 44 := 
sorry

end NUMINAMATH_GPT_alice_speed_exceeds_l1158_115876


namespace NUMINAMATH_GPT_distance_from_P_to_AB_l1158_115880

-- Definitions of conditions
def is_point_in_triangle (P A B C : ℝ×ℝ) : Prop := sorry
def parallel_to_base (P A B C : ℝ×ℝ) : Prop := sorry
def divides_area_in_ratio (P A B C : ℝ×ℝ) (r1 r2 : ℕ) : Prop := sorry

theorem distance_from_P_to_AB (P A B C : ℝ×ℝ) 
  (H_in_triangle : is_point_in_triangle P A B C)
  (H_parallel : parallel_to_base P A B C)
  (H_area_ratio : divides_area_in_ratio P A B C 1 3)
  (H_altitude : ∃ h : ℝ, h = 1) :
  ∃ d : ℝ, d = 3/4 :=
by
  sorry

end NUMINAMATH_GPT_distance_from_P_to_AB_l1158_115880


namespace NUMINAMATH_GPT_reciprocal_check_C_l1158_115867

theorem reciprocal_check_C : 0.1 * 10 = 1 := 
by 
  sorry

end NUMINAMATH_GPT_reciprocal_check_C_l1158_115867


namespace NUMINAMATH_GPT_evaluate_expression_l1158_115861

-- Given variables x and y are non-zero
variables (x y : ℝ)

-- Condition
axiom xy_nonzero : x * y ≠ 0

-- Statement of the proof
theorem evaluate_expression (x y : ℝ) (h : x * y ≠ 0) :
  ((x^3 + 2) / x * (y^3 + 2) / y + (x^3 - 2) / y * (y^3 - 2) / x) = 2 * x * y * (x^2 * y^2) + 8 / (x * y) := 
by {
  sorry
}

end NUMINAMATH_GPT_evaluate_expression_l1158_115861


namespace NUMINAMATH_GPT_activities_equally_popular_l1158_115891

def Dodgeball_prefers : ℚ := 10 / 25
def ArtWorkshop_prefers : ℚ := 12 / 30
def MovieScreening_prefers : ℚ := 18 / 45
def QuizBowl_prefers : ℚ := 16 / 40

theorem activities_equally_popular :
  Dodgeball_prefers = ArtWorkshop_prefers ∧
  ArtWorkshop_prefers = MovieScreening_prefers ∧
  MovieScreening_prefers = QuizBowl_prefers :=
by
  sorry

end NUMINAMATH_GPT_activities_equally_popular_l1158_115891


namespace NUMINAMATH_GPT_arithmetic_seq_problem_l1158_115860

theorem arithmetic_seq_problem (S : ℕ → ℤ) (n : ℕ) (h1 : S 6 = 36) 
                               (h2 : S n = 324) (h3 : S (n - 6) = 144) (hn : n > 6) : 
  n = 18 := 
sorry

end NUMINAMATH_GPT_arithmetic_seq_problem_l1158_115860


namespace NUMINAMATH_GPT_part1_part2_l1158_115847

noncomputable def f (m x : ℝ) : ℝ := Real.exp (m * x) - Real.log x - 2

theorem part1 (t : ℝ) :
  (1 / 2 < t ∧ t < 1) →
  (∃! t : ℝ, f 1 t = 0) := sorry

theorem part2 :
  (∃ m : ℝ, 0 < m ∧ m < 1 ∧ ∀ x : ℝ, x > 0 → f m x > 0) := sorry

end NUMINAMATH_GPT_part1_part2_l1158_115847


namespace NUMINAMATH_GPT_sin_cos_15_sin_cos_18_l1158_115856

theorem sin_cos_15 (h45sin : Real.sin (45 * Real.pi / 180) = Real.sqrt 2 / 2)
                  (h45cos : Real.cos (45 * Real.pi / 180) = Real.sqrt 2 / 2)
                  (h30sin : Real.sin (30 * Real.pi / 180) = 1 / 2)
                  (h30cos : Real.cos (30 * Real.pi / 180) = Real.sqrt 3 / 2) :
  Real.sin (15 * Real.pi / 180) = (Real.sqrt 6 - Real.sqrt 2) / 4 ∧
  Real.cos (15 * Real.pi / 180) = (Real.sqrt 6 + Real.sqrt 2) / 4 := by
  sorry

theorem sin_cos_18 (h18sin : Real.sin (18 * Real.pi / 180) = (-1 + Real.sqrt 5) / 4)
                   (h36cos : Real.cos (36 * Real.pi / 180) = (Real.sqrt 5 + 1) / 4) :
  Real.cos (18 * Real.pi / 180) = Real.sqrt (10 + 2 * Real.sqrt 5) / 4 := by
  sorry

end NUMINAMATH_GPT_sin_cos_15_sin_cos_18_l1158_115856


namespace NUMINAMATH_GPT_horner_method_multiplications_and_additions_l1158_115882

noncomputable def f (x : ℕ) : ℕ :=
  12 * x ^ 6 + 5 * x ^ 5 + 11 * x ^ 2 + 2 * x + 5

theorem horner_method_multiplications_and_additions (x : ℕ) :
  let multiplications := 6
  let additions := 4
  multiplications = 6 ∧ additions = 4 :=
sorry

end NUMINAMATH_GPT_horner_method_multiplications_and_additions_l1158_115882


namespace NUMINAMATH_GPT_find_d_l1158_115830

noncomputable def single_point_graph (d : ℝ) : Prop :=
  ∃ x y : ℝ, 3 * x^2 + 2 * y^2 + 9 * x - 14 * y + d = 0

theorem find_d : single_point_graph 31.25 :=
sorry

end NUMINAMATH_GPT_find_d_l1158_115830


namespace NUMINAMATH_GPT_length_of_tracks_l1158_115831

theorem length_of_tracks (x y : ℕ) 
  (h1 : 6 * (x + 2 * y) = 5000)
  (h2 : 7 * (x + y) = 5000) : x = 5 * y :=
  sorry

end NUMINAMATH_GPT_length_of_tracks_l1158_115831


namespace NUMINAMATH_GPT_members_playing_both_l1158_115803

variable (N B T Neither BT : ℕ)

theorem members_playing_both (hN : N = 30) (hB : B = 17) (hT : T = 17) (hNeither : Neither = 2) 
  (hBT : BT = B + T - (N - Neither)) : BT = 6 := 
by 
  rw [hN, hB, hT, hNeither] at hBT
  exact hBT

end NUMINAMATH_GPT_members_playing_both_l1158_115803


namespace NUMINAMATH_GPT_sum_of_constants_l1158_115811

theorem sum_of_constants (x a b : ℤ) (h : x^2 - 10 * x + 15 = 0) 
    (h1 : (x + a)^2 = b) : a + b = 5 := 
sorry

end NUMINAMATH_GPT_sum_of_constants_l1158_115811


namespace NUMINAMATH_GPT_feb_03_2013_nine_day_l1158_115804

-- Definitions of the main dates involved
def dec_21_2012 : Nat := 0  -- Assuming day 0 is Dec 21, 2012
def feb_03_2013 : Nat := 45  -- 45 days after Dec 21, 2012

-- Definition to determine the Nine-day period
def nine_day_period (x : Nat) : (Nat × Nat) :=
  let q := x / 9
  let r := x % 9
  (q + 1, r + 1)

-- Theorem we want to prove
theorem feb_03_2013_nine_day : nine_day_period feb_03_2013 = (5, 9) :=
by
  sorry

end NUMINAMATH_GPT_feb_03_2013_nine_day_l1158_115804


namespace NUMINAMATH_GPT_base6_divisibility_13_l1158_115841

theorem base6_divisibility_13 (d : ℕ) (h : 0 ≤ d ∧ d ≤ 5) : (435 + 42 * d) % 13 = 0 ↔ d = 5 :=
by sorry

end NUMINAMATH_GPT_base6_divisibility_13_l1158_115841


namespace NUMINAMATH_GPT_certain_number_is_3_l1158_115898

-- Given conditions
variables (z x : ℤ)
variable (k : ℤ)
variable (n : ℤ)

-- Conditions
-- Remainder when z is divided by 9 is 6
def is_remainder_6 (z : ℤ) := ∃ k : ℤ, z = 9 * k + 6
-- (z + x) / 9 is an integer
def is_integer_division (z x : ℤ) := ∃ m : ℤ, (z + x) / 9 = m

-- Proof to show that x must be 3
theorem certain_number_is_3 (z : ℤ) (h1 : is_remainder_6 z) (h2 : is_integer_division z x) : x = 3 :=
sorry

end NUMINAMATH_GPT_certain_number_is_3_l1158_115898


namespace NUMINAMATH_GPT_eliminate_denominator_l1158_115894

theorem eliminate_denominator (x : ℝ) : 6 - (x - 2) / 2 = x → 12 - x + 2 = 2 * x :=
by
  intro h
  sorry

end NUMINAMATH_GPT_eliminate_denominator_l1158_115894


namespace NUMINAMATH_GPT_problem_equiv_l1158_115801

-- Definitions to match the conditions
def is_monomial (v : List ℤ) : Prop :=
  ∀ i ∈ v, True  -- Simplified; typically this would involve more specific definitions

def degree (e : String) : ℕ :=
  if e = "xy" then 2 else 0

noncomputable def coefficient (v : String) : ℤ :=
  if v = "m" then 1 else 0

-- Main fact to be proven
theorem problem_equiv :
  is_monomial [-3, 1, 5] :=
sorry

end NUMINAMATH_GPT_problem_equiv_l1158_115801


namespace NUMINAMATH_GPT_SharonOranges_l1158_115848

-- Define the given conditions
def JanetOranges : Nat := 9
def TotalOranges : Nat := 16

-- Define the statement that needs to be proven
theorem SharonOranges (J : Nat) (T : Nat) (S : Nat) (hJ : J = 9) (hT : T = 16) (hS : S = T - J) : S = 7 := by
  -- (proof to be filled in later)
  sorry

end NUMINAMATH_GPT_SharonOranges_l1158_115848


namespace NUMINAMATH_GPT_number_of_machines_in_first_group_l1158_115829

-- Define the initial conditions
def first_group_production_rate (x : ℕ) : ℚ :=
  20 / (x * 10)

def second_group_production_rate : ℚ :=
  180 / (20 * 22.5)

-- The theorem we aim to prove
theorem number_of_machines_in_first_group (x : ℕ) (h1 : first_group_production_rate x = second_group_production_rate) :
  x = 5 :=
by
  -- Placeholder for the proof steps
  sorry

end NUMINAMATH_GPT_number_of_machines_in_first_group_l1158_115829


namespace NUMINAMATH_GPT_player_a_winning_strategy_l1158_115857

theorem player_a_winning_strategy (P : ℝ) : 
  (∃ m n : ℕ, P = m / (2 ^ n) ∧ m < 2 ^ n)
  ∨ P = 0
  ∨ P = 1 ↔
  (∀ d : ℝ, ∃ d_direction : ℤ, 
    (P + (d * d_direction) = 0) ∨ (P + (d * d_direction) = 1)) :=
sorry

end NUMINAMATH_GPT_player_a_winning_strategy_l1158_115857


namespace NUMINAMATH_GPT_negation_of_proposition_l1158_115824

theorem negation_of_proposition :
  (¬ (∃ x₀ : ℝ, x₀ > 2 ∧ x₀^3 - 2 * x₀^2 < 0)) ↔ (∀ x : ℝ, x > 2 → x^3 - 2 * x^2 ≥ 0) := by
  sorry

end NUMINAMATH_GPT_negation_of_proposition_l1158_115824


namespace NUMINAMATH_GPT_find_number_l1158_115875

theorem find_number (x : ℝ) 
(h : x * 13.26 + x * 9.43 + x * 77.31 = 470) : 
x = 4.7 := 
sorry

end NUMINAMATH_GPT_find_number_l1158_115875


namespace NUMINAMATH_GPT_chloe_apples_l1158_115815

theorem chloe_apples :
  ∃ x : ℕ, (∃ y : ℕ, x = y + 8 ∧ y = x / 3) ∧ x = 12 := 
by
  sorry

end NUMINAMATH_GPT_chloe_apples_l1158_115815


namespace NUMINAMATH_GPT_find_n_tan_eq_l1158_115851

theorem find_n_tan_eq (n : ℤ) (h₁ : -180 < n) (h₂ : n < 180) 
  (h₃ : Real.tan (n * (Real.pi / 180)) = Real.tan (276 * (Real.pi / 180))) : 
  n = 96 :=
sorry

end NUMINAMATH_GPT_find_n_tan_eq_l1158_115851


namespace NUMINAMATH_GPT_envelope_weight_l1158_115890

theorem envelope_weight (E : ℝ) :
  (8 * (1 / 5) + E ≤ 2) ∧ (1 < 8 * (1 / 5) + E) ∧ (E ≥ 0) ↔ E = 2 / 5 :=
by
  sorry

end NUMINAMATH_GPT_envelope_weight_l1158_115890


namespace NUMINAMATH_GPT_personal_planner_cost_l1158_115866

variable (P : ℝ)
variable (C_spiral_notebook : ℝ := 15)
variable (total_cost_with_discount : ℝ := 112)
variable (discount_rate : ℝ := 0.20)
variable (num_spiral_notebooks : ℝ := 4)
variable (num_personal_planners : ℝ := 8)

theorem personal_planner_cost : (4 * C_spiral_notebook + 8 * P) * (1 - 0.20) = 112 → 
  P = 10 :=
by
  sorry

end NUMINAMATH_GPT_personal_planner_cost_l1158_115866


namespace NUMINAMATH_GPT_regular_polygon_exterior_angle_l1158_115869

theorem regular_polygon_exterior_angle (n : ℕ) (h : n > 2) (h_exterior : 36 = 360 / n) : n = 10 :=
sorry

end NUMINAMATH_GPT_regular_polygon_exterior_angle_l1158_115869


namespace NUMINAMATH_GPT_find_y_l1158_115805

theorem find_y (x y z : ℤ) (h₁ : x + y + z = 355) (h₂ : x - y = 200) (h₃ : x + z = 500) : y = -145 :=
by
  sorry

end NUMINAMATH_GPT_find_y_l1158_115805


namespace NUMINAMATH_GPT_total_pink_crayons_l1158_115810

def mara_crayons := 40
def mara_pink_percent := 10
def luna_crayons := 50
def luna_pink_percent := 20

def pink_crayons (total_crayons : ℕ) (percent_pink : ℕ) : ℕ :=
  (percent_pink * total_crayons) / 100

def mara_pink_crayons := pink_crayons mara_crayons mara_pink_percent
def luna_pink_crayons := pink_crayons luna_crayons luna_pink_percent

theorem total_pink_crayons : mara_pink_crayons + luna_pink_crayons = 14 :=
by
  -- Proof can be written here.
  sorry

end NUMINAMATH_GPT_total_pink_crayons_l1158_115810


namespace NUMINAMATH_GPT_total_rainfall_in_Springdale_l1158_115826

theorem total_rainfall_in_Springdale
    (rainfall_first_week rainfall_second_week : ℝ)
    (h1 : rainfall_second_week = 1.5 * rainfall_first_week)
    (h2 : rainfall_second_week = 12) :
    (rainfall_first_week + rainfall_second_week = 20) :=
by
  sorry

end NUMINAMATH_GPT_total_rainfall_in_Springdale_l1158_115826


namespace NUMINAMATH_GPT_distinct_triangles_count_l1158_115849

def num_combinations (n k : ℕ) : ℕ := n.choose k

def count_collinear_sets_in_grid (grid_size : ℕ) : ℕ :=
  let rows := grid_size
  let cols := grid_size
  let diagonals := 2
  rows + cols + diagonals

noncomputable def distinct_triangles_in_grid (grid_size n k : ℕ) : ℕ :=
  num_combinations n k - count_collinear_sets_in_grid grid_size

theorem distinct_triangles_count :
  distinct_triangles_in_grid 3 9 3 = 76 := 
by 
  sorry

end NUMINAMATH_GPT_distinct_triangles_count_l1158_115849


namespace NUMINAMATH_GPT_point_inside_circle_l1158_115859

theorem point_inside_circle (r OP : ℝ) (h₁ : r = 3) (h₂ : OP = 2) : OP < r :=
by
  sorry

end NUMINAMATH_GPT_point_inside_circle_l1158_115859


namespace NUMINAMATH_GPT_shirt_price_is_correct_l1158_115873

noncomputable def sweater_price (T : ℝ) : ℝ := T + 7.43 

def discounted_price (S : ℝ) : ℝ := S * 0.90

theorem shirt_price_is_correct :
  ∃ (T S : ℝ), T + discounted_price S = 80.34 ∧ T = S - 7.43 ∧ T = 38.76 :=
by
  sorry

end NUMINAMATH_GPT_shirt_price_is_correct_l1158_115873


namespace NUMINAMATH_GPT_power_expression_evaluation_l1158_115877

theorem power_expression_evaluation :
  (1 / 2) ^ 2016 * (-2) ^ 2017 * (-1) ^ 2017 = 2 := 
by
  sorry

end NUMINAMATH_GPT_power_expression_evaluation_l1158_115877


namespace NUMINAMATH_GPT_find_c_l1158_115802

noncomputable def f (x a b c : ℤ) := x^3 + a * x^2 + b * x + c

theorem find_c (a b c : ℤ) (h₀ : a ≠ 0) (h₁ : b ≠ 0) (h₂ : c ≠ 0) 
  (h₃ : f a a b c = a^3) (h₄ : f b a b c = b^3) : c = 16 :=
sorry

end NUMINAMATH_GPT_find_c_l1158_115802


namespace NUMINAMATH_GPT_impossible_to_form_16_unique_remainders_with_3_digits_l1158_115850

theorem impossible_to_form_16_unique_remainders_with_3_digits :
  ¬∃ (digits : Finset ℕ) (num_fun : Fin 16 → ℕ), digits.card = 3 ∧ 
  ∀ i j : Fin 16, i ≠ j → num_fun i % 16 ≠ num_fun j % 16 ∧ 
  ∀ n : ℕ, n ∈ (digits : Set ℕ) → 100 ≤ num_fun i ∧ num_fun i < 1000 :=
sorry

end NUMINAMATH_GPT_impossible_to_form_16_unique_remainders_with_3_digits_l1158_115850


namespace NUMINAMATH_GPT_eighty_five_percent_of_forty_greater_than_four_fifths_of_twenty_five_l1158_115845

theorem eighty_five_percent_of_forty_greater_than_four_fifths_of_twenty_five:
  (0.85 * 40) - (4 / 5 * 25) = 14 :=
by
  sorry

end NUMINAMATH_GPT_eighty_five_percent_of_forty_greater_than_four_fifths_of_twenty_five_l1158_115845


namespace NUMINAMATH_GPT_expression_simplification_l1158_115871

theorem expression_simplification (x : ℝ) (h : x < -2) : 1 - |1 + x| = -2 - x := 
by
  sorry

end NUMINAMATH_GPT_expression_simplification_l1158_115871


namespace NUMINAMATH_GPT_tesses_ride_is_longer_l1158_115883

noncomputable def tesses_total_distance : ℝ := 0.75 + 0.85 + 1.15
noncomputable def oscars_total_distance : ℝ := 0.25 + 1.35

theorem tesses_ride_is_longer :
  (tesses_total_distance - oscars_total_distance) = 1.15 := by
  sorry

end NUMINAMATH_GPT_tesses_ride_is_longer_l1158_115883


namespace NUMINAMATH_GPT_arithmetic_sequence_product_l1158_115868

noncomputable def arithmetic_sequence (n : ℕ) : ℝ := sorry

theorem arithmetic_sequence_product (a_1 a_6 a_7 a_4 a_9 : ℝ) (d : ℝ) :
  a_1 = 2 →
  a_6 = a_1 + 5 * d →
  a_7 = a_1 + 6 * d →
  a_6 * a_7 = 15 →
  a_4 = a_1 + 3 * d →
  a_9 = a_1 + 8 * d →
  a_4 * a_9 = 234 / 25 :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_product_l1158_115868


namespace NUMINAMATH_GPT_sharon_trip_distance_l1158_115842

noncomputable def usual_speed (x : ℝ) : ℝ := x / 200

noncomputable def reduced_speed (x : ℝ) : ℝ := x / 200 - 30 / 60

theorem sharon_trip_distance (x : ℝ) (h1 : (x / 3) / usual_speed x + (2 * x / 3) / reduced_speed x = 310) : 
x = 220 :=
by
  sorry

end NUMINAMATH_GPT_sharon_trip_distance_l1158_115842


namespace NUMINAMATH_GPT_years_required_l1158_115862

def num_stadiums := 30
def avg_cost_per_stadium := 900
def annual_savings := 1500
def total_cost := num_stadiums * avg_cost_per_stadium

theorem years_required : total_cost / annual_savings = 18 :=
by
  sorry

end NUMINAMATH_GPT_years_required_l1158_115862


namespace NUMINAMATH_GPT_find_25_percent_l1158_115899

theorem find_25_percent (x : ℝ) (h : x - (3/4) * x = 100) : (1/4) * x = 100 :=
by
  sorry

end NUMINAMATH_GPT_find_25_percent_l1158_115899


namespace NUMINAMATH_GPT_casey_saves_money_l1158_115832

def first_employee_hourly_wage : ℕ := 20
def second_employee_hourly_wage : ℕ := 22
def subsidy_per_hour : ℕ := 6
def weekly_work_hours : ℕ := 40

theorem casey_saves_money :
  let first_employee_weekly_cost := first_employee_hourly_wage * weekly_work_hours
  let second_employee_effective_hourly_wage := second_employee_hourly_wage - subsidy_per_hour
  let second_employee_weekly_cost := second_employee_effective_hourly_wage * weekly_work_hours
  let savings := first_employee_weekly_cost - second_employee_weekly_cost
  savings = 160 :=
by
  sorry

end NUMINAMATH_GPT_casey_saves_money_l1158_115832


namespace NUMINAMATH_GPT_tammy_speed_on_second_day_l1158_115843

variable (v₁ t₁ v₂ t₂ d₁ d₂ : ℝ)

theorem tammy_speed_on_second_day
  (h1 : t₁ + t₂ = 14)
  (h2 : t₂ = t₁ - 2)
  (h3 : d₁ + d₂ = 52)
  (h4 : v₂ = v₁ + 0.5)
  (h5 : d₁ = v₁ * t₁)
  (h6 : d₂ = v₂ * t₂)
  (h_eq : v₁ * t₁ + (v₁ + 0.5) * (t₁ - 2) = 52)
  : v₂ = 4 := 
sorry

end NUMINAMATH_GPT_tammy_speed_on_second_day_l1158_115843


namespace NUMINAMATH_GPT_monthly_avg_growth_rate_25_max_avg_tourists_next_10_days_l1158_115897

-- Definition of given conditions regarding tourists count in February and April
def tourists_in_february : ℕ := 16000
def tourists_in_april : ℕ := 25000

-- Theorem 1: Monthly average growth rate of tourists from February to April is 25%.
theorem monthly_avg_growth_rate_25 :
  (tourists_in_april : ℝ) = tourists_in_february * (1 + 0.25)^2 :=
sorry

-- Definition of given conditions for tourists count from May 1st to May 21st
def tourists_may_1_to_21 : ℕ := 21250
def max_total_tourists_may : ℕ := 31250 -- Expressed in thousands as 31.25 in millions

-- Theorem 2: Maximum average number of tourists per day in the next 10 days of May.
theorem max_avg_tourists_next_10_days :
  ∀ (a : ℝ), tourists_may_1_to_21 + 10 * a ≤ max_total_tourists_may →
  a ≤ 10000 :=
sorry

end NUMINAMATH_GPT_monthly_avg_growth_rate_25_max_avg_tourists_next_10_days_l1158_115897


namespace NUMINAMATH_GPT_mark_additional_inches_l1158_115879

theorem mark_additional_inches
  (mark_feet : ℕ)
  (mark_inches : ℕ)
  (mike_feet : ℕ)
  (mike_inches : ℕ)
  (foot_to_inches : ℕ)
  (mike_taller_than_mark : ℕ) :
  mark_feet = 5 →
  mike_feet = 6 →
  mike_inches = 1 →
  mike_taller_than_mark = 10 →
  foot_to_inches = 12 →
  5 * 12 + mark_inches + 10 = 6 * 12 + 1 →
  mark_inches = 3 :=
by
  intros
  sorry

end NUMINAMATH_GPT_mark_additional_inches_l1158_115879


namespace NUMINAMATH_GPT_frigate_catches_smuggler_at_five_l1158_115807

noncomputable def time_to_catch : ℝ :=
  2 + (12 / 4) -- Initial leading distance / Relative speed before storm
  
theorem frigate_catches_smuggler_at_five 
  (initial_distance : ℝ)
  (frigate_speed_before_storm : ℝ)
  (smuggler_speed_before_storm : ℝ)
  (time_before_storm : ℝ)
  (frigate_speed_after_storm : ℝ)
  (smuggler_speed_after_storm : ℝ) :
  initial_distance = 12 →
  frigate_speed_before_storm = 14 →
  smuggler_speed_before_storm = 10 →
  time_before_storm = 3 →
  frigate_speed_after_storm = 12 →
  smuggler_speed_after_storm = 9 →
  time_to_catch = 5 :=
by
{
  sorry
}

end NUMINAMATH_GPT_frigate_catches_smuggler_at_five_l1158_115807


namespace NUMINAMATH_GPT_find_m_n_l1158_115809

theorem find_m_n (m n : ℝ) : (∀ x : ℝ, -5 ≤ x ∧ x ≤ 1 → x^2 - m * x + n ≤ 0) → m = -4 ∧ n = -5 :=
by
  sorry

end NUMINAMATH_GPT_find_m_n_l1158_115809


namespace NUMINAMATH_GPT_least_number_to_add_l1158_115812

theorem least_number_to_add (x : ℕ) (h : 53 ∣ x ∧ 71 ∣ x) : 
  ∃ n : ℕ, x = 1357 + n ∧ n = 2406 :=
by sorry

end NUMINAMATH_GPT_least_number_to_add_l1158_115812


namespace NUMINAMATH_GPT_length_AB_is_correct_l1158_115863

noncomputable def length_of_AB (x y : ℚ) : ℚ :=
  let a := 3 * x
  let b := 2 * x
  let c := 4 * y
  let d := 5 * y
  let pq_distance := abs (c - a)
  if 5 * x = 9 * y ∧ pq_distance = 3 then 5 * x else 0

theorem length_AB_is_correct : 
  ∃ x y : ℚ, 5 * x = 9 * y ∧ (abs (4 * y - 3 * x)) = 3 ∧ length_of_AB x y = 135 / 7 := 
by
  sorry

end NUMINAMATH_GPT_length_AB_is_correct_l1158_115863


namespace NUMINAMATH_GPT_final_lights_on_l1158_115846

def lights_on_by_children : ℕ :=
  let total_lights := 200
  let flips_x := total_lights / 7
  let flips_y := total_lights / 11
  let lcm_xy := 77  -- since lcm(7, 11) = 7 * 11 = 77
  let flips_both := total_lights / lcm_xy
  flips_x + flips_y - flips_both

theorem final_lights_on : lights_on_by_children = 44 :=
by
  sorry

end NUMINAMATH_GPT_final_lights_on_l1158_115846


namespace NUMINAMATH_GPT_inequality_proof_l1158_115834

open Real

theorem inequality_proof (x y : ℝ) (hx : x > 1/2) (hy : y > 1) : 
  (4 * x^2) / (y - 1) + (y^2) / (2 * x - 1) ≥ 8 := 
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l1158_115834


namespace NUMINAMATH_GPT_A_B_together_l1158_115837

/-- This represents the problem of finding out the number of days A and B together 
can finish a piece of work given the conditions. -/
theorem A_B_together (A_rate B_rate: ℝ) (A_days B_days: ℝ) (work: ℝ) :
  A_rate = 1 / 8 →
  A_days = 4 →
  B_rate = 1 / 12 →
  B_days = 6 →
  work = 1 →
  (A_days * A_rate + B_days * B_rate = work / 2) →
  (24 / (A_rate + B_rate) = 4.8) :=
by
  intros hA_rate hA_days hB_rate hB_days hwork hwork_done
  sorry

end NUMINAMATH_GPT_A_B_together_l1158_115837


namespace NUMINAMATH_GPT_quadratic_inequality_solution_l1158_115886

theorem quadratic_inequality_solution : 
  {x : ℝ | x^2 - 5 * x + 6 > 0 ∧ x ≠ 3} = {x : ℝ | x < 2} ∪ {x : ℝ | x > 3} :=
by
  sorry

end NUMINAMATH_GPT_quadratic_inequality_solution_l1158_115886


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_condition_l1158_115823

theorem necessary_but_not_sufficient_condition (a b : ℝ) : 
  (a > b → a + 1 > b) ∧ (∃ a b : ℝ, a + 1 > b ∧ ¬ a > b) :=
by 
  sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_condition_l1158_115823


namespace NUMINAMATH_GPT_max_students_seated_l1158_115854

/-- Problem statement:
There are a total of 8 rows of desks.
The first row has 10 desks.
Each subsequent row has 2 more desks than the previous row.
We need to prove that the maximum number of students that can be seated in the class is 136.
-/
theorem max_students_seated : 
  let n := 8      -- number of rows
  let a1 := 10    -- desks in the first row
  let d := 2      -- common difference
  let an := a1 + (n - 1) * d  -- desks in the n-th row
  let S := n / 2 * (a1 + an)  -- sum of the arithmetic series
  S = 136 :=
by
  sorry

end NUMINAMATH_GPT_max_students_seated_l1158_115854


namespace NUMINAMATH_GPT_diff_of_squares_example_l1158_115888

theorem diff_of_squares_example : 535^2 - 465^2 = 70000 := by
  sorry

end NUMINAMATH_GPT_diff_of_squares_example_l1158_115888


namespace NUMINAMATH_GPT_power_sum_l1158_115881

theorem power_sum :
  (-1:ℤ)^53 + 2^(3^4 + 4^3 - 6 * 7) = 2^103 - 1 :=
by
  sorry

end NUMINAMATH_GPT_power_sum_l1158_115881


namespace NUMINAMATH_GPT_find_primes_satisfying_equation_l1158_115864

theorem find_primes_satisfying_equation :
  {p : ℕ | p.Prime ∧ ∃ x y : ℕ, x > 0 ∧ y > 0 ∧ x * (y^2 - p) + y * (x^2 - p) = 5 * p} = {2, 3, 7} :=
by
  sorry

end NUMINAMATH_GPT_find_primes_satisfying_equation_l1158_115864


namespace NUMINAMATH_GPT_num_int_values_not_satisfying_l1158_115896

theorem num_int_values_not_satisfying:
  (∃ n : ℕ, n = 7 ∧ (∃ x : ℤ, 7 * x^2 + 25 * x + 24 ≤ 30)) :=
sorry

end NUMINAMATH_GPT_num_int_values_not_satisfying_l1158_115896


namespace NUMINAMATH_GPT_average_speed_correct_l1158_115813

-- Define the conditions
def distance_first_hour := 90 -- in km
def distance_second_hour := 30 -- in km
def time_first_hour := 1 -- in hours
def time_second_hour := 1 -- in hours

-- Define the total distance and total time
def total_distance := distance_first_hour + distance_second_hour
def total_time := time_first_hour + time_second_hour

-- Define the average speed
def avg_speed := total_distance / total_time

-- State the theorem to prove the average speed is 60
theorem average_speed_correct :
  avg_speed = 60 := 
by 
  -- Placeholder for the actual proof
  sorry

end NUMINAMATH_GPT_average_speed_correct_l1158_115813


namespace NUMINAMATH_GPT_solve_equation_3x6_eq_3mx_div_xm1_l1158_115817

theorem solve_equation_3x6_eq_3mx_div_xm1 (x : ℝ) 
  (h1 : x ≠ 1)
  (h2 : x^2 + 5*x - 6 ≠ 0) :
  (3 * x + 6) / (x^2 + 5 * x - 6) = (3 - x) / (x - 1) ↔ (x = 3 ∨ x = -6) :=
by 
  sorry

end NUMINAMATH_GPT_solve_equation_3x6_eq_3mx_div_xm1_l1158_115817


namespace NUMINAMATH_GPT_g_is_zero_l1158_115818

noncomputable def g (x : Real) : Real := 
  Real.sqrt (Real.cos x ^ 4 + 4 * Real.sin x ^ 2) - 
  Real.sqrt (Real.sin x ^ 4 + 4 * Real.cos x ^ 2)

theorem g_is_zero : ∀ x : Real, g x = 0 := by
  sorry

end NUMINAMATH_GPT_g_is_zero_l1158_115818


namespace NUMINAMATH_GPT_problem_l1158_115821

theorem problem (n : ℝ) (h : n + 1 / n = 10) : n ^ 2 + 1 / n ^ 2 + 5 = 103 :=
by sorry

end NUMINAMATH_GPT_problem_l1158_115821


namespace NUMINAMATH_GPT_cosine_of_angle_between_vectors_l1158_115887

theorem cosine_of_angle_between_vectors (a1 b1 c1 a2 b2 c2 : ℝ) :
  let u := (a1, b1, c1)
  let v := (a2, b2, c2)
  let dot_product := a1 * a2 + b1 * b2 + c1 * c2
  let magnitude_u := Real.sqrt (a1^2 + b1^2 + c1^2)
  let magnitude_v := Real.sqrt (a2^2 + b2^2 + c2^2)
  dot_product / (magnitude_u * magnitude_v) = 
      (a1 * a2 + b1 * b2 + c1 * c2) / (Real.sqrt (a1^2 + b1^2 + c1^2) * Real.sqrt (a2^2 + b2^2 + c2^2)) :=
by
  let u := (a1, b1, c1)
  let v := (a2, b2, c2)
  let dot_product := a1 * a2 + b1 * b2 + c1 * c2
  let magnitude_u := Real.sqrt (a1^2 + b1^2 + c1^2)
  let magnitude_v := Real.sqrt (a2^2 + b2^2 + c2^2)
  sorry

end NUMINAMATH_GPT_cosine_of_angle_between_vectors_l1158_115887


namespace NUMINAMATH_GPT_music_track_duration_l1158_115827

theorem music_track_duration (minutes : ℝ) (seconds_per_minute : ℝ) (duration_in_minutes : minutes = 12.5) (seconds_per_minute_is_60 : seconds_per_minute = 60) : minutes * seconds_per_minute = 750 := by
  sorry

end NUMINAMATH_GPT_music_track_duration_l1158_115827


namespace NUMINAMATH_GPT_rectangular_coords_transformation_l1158_115822

noncomputable def sphericalToRectangular (ρ θ φ : ℝ) : ℝ × ℝ × ℝ :=
(ρ * Real.sin φ * Real.cos θ, ρ * Real.sin φ * Real.sin θ, ρ * Real.cos φ)

theorem rectangular_coords_transformation :
  let ρ := Real.sqrt (2 ^ 2 + (-3) ^ 2 + 6 ^ 2)
  let φ := Real.arccos (6 / ρ)
  let θ := Real.arctan (-3 / 2)
  sphericalToRectangular ρ (Real.pi + θ) φ = (-2, 3, 6) :=
by
  sorry

end NUMINAMATH_GPT_rectangular_coords_transformation_l1158_115822


namespace NUMINAMATH_GPT_geometric_arithmetic_sequence_l1158_115872

theorem geometric_arithmetic_sequence (a_n : ℕ → ℕ) (q : ℕ) (a1_eq : a_n 1 = 3)
  (an_geometric : ∀ n, a_n (n + 1) = a_n n * q)
  (arithmetic_condition : 4 * a_n 1 + a_n 3 = 8 * a_n 2) :
  a_n 3 + a_n 4 + a_n 5 = 84 := by
  sorry

end NUMINAMATH_GPT_geometric_arithmetic_sequence_l1158_115872


namespace NUMINAMATH_GPT_find_number_l1158_115858

theorem find_number (x : ℝ) (h : x^2 + 95 = (x - 20)^2) : x = 7.625 :=
sorry

end NUMINAMATH_GPT_find_number_l1158_115858


namespace NUMINAMATH_GPT_vec_sub_eq_l1158_115865

variables (a b : ℝ × ℝ)
def vec_a : ℝ × ℝ := (2, 1)
def vec_b : ℝ × ℝ := (-3, 4)

theorem vec_sub_eq : vec_a - vec_b = (5, -3) :=
by 
  -- You can fill in the proof steps here
  sorry

end NUMINAMATH_GPT_vec_sub_eq_l1158_115865


namespace NUMINAMATH_GPT_cannot_obtain_fraction_3_5_l1158_115840

theorem cannot_obtain_fraction_3_5 (n k : ℕ) :
  ¬ ∃ (a b : ℕ), (a = 5 + k ∧ b = 8 + k ∨ (∃ m : ℕ, a = m * 5 ∧ b = m * 8)) ∧ (a = 3 ∧ b = 5) :=
by
  sorry

end NUMINAMATH_GPT_cannot_obtain_fraction_3_5_l1158_115840


namespace NUMINAMATH_GPT_inequality_of_pos_real_product_l1158_115852

theorem inequality_of_pos_real_product
  (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hxyz : x * y * z = 1) :
  (x^3 / ((1 + y) * (1 + z)) + y^3 / ((1 + z) * (1 + x)) + z^3 / ((1 + x) * (1 + y))) ≥ (3 / 4) :=
sorry

end NUMINAMATH_GPT_inequality_of_pos_real_product_l1158_115852


namespace NUMINAMATH_GPT_find_functions_l1158_115800

theorem find_functions (f : ℝ → ℝ) (h : ∀ x : ℝ, f (2002 * x - f 0) = 2002 * x^2) :
  (∀ x, f x = (x^2) / 2002) ∨ (∀ x, f x = (x^2) / 2002 + 2 * x + 2002) :=
sorry

end NUMINAMATH_GPT_find_functions_l1158_115800


namespace NUMINAMATH_GPT_line_intersects_circle_l1158_115870

-- Definitions based on conditions
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 9 = 0
def line_eq (m x y : ℝ) : Prop := m*x + y + m - 2 = 0

-- Theorem statement based on question and correct answer
theorem line_intersects_circle (m : ℝ) :
  ∃ (x y : ℝ), circle_eq x y ∧ line_eq m x y :=
sorry

end NUMINAMATH_GPT_line_intersects_circle_l1158_115870


namespace NUMINAMATH_GPT_ratio_pat_to_mark_l1158_115893

theorem ratio_pat_to_mark (K P M : ℕ) 
  (h1 : P + K + M = 117) 
  (h2 : P = 2 * K) 
  (h3 : M = K + 65) : 
  P / Nat.gcd P M = 1 ∧ M / Nat.gcd P M = 3 := 
by
  sorry

end NUMINAMATH_GPT_ratio_pat_to_mark_l1158_115893


namespace NUMINAMATH_GPT_infinite_primes_of_form_4n_plus_3_l1158_115833

theorem infinite_primes_of_form_4n_plus_3 :
  ∀ (S : Finset ℕ), (∀ p ∈ S, Prime p ∧ p % 4 = 3) →
  ∃ q, Prime q ∧ q % 4 = 3 ∧ q ∉ S :=
by 
  sorry

end NUMINAMATH_GPT_infinite_primes_of_form_4n_plus_3_l1158_115833


namespace NUMINAMATH_GPT_subtracted_value_l1158_115808

theorem subtracted_value (s : ℕ) (h : s = 4) (x : ℕ) (h2 : (s + s^2 - x = 4)) : x = 16 :=
by
  sorry

end NUMINAMATH_GPT_subtracted_value_l1158_115808


namespace NUMINAMATH_GPT_sequence_expression_l1158_115835

theorem sequence_expression (s a : ℕ → ℝ) (h : ∀ n : ℕ, 1 ≤ n → s n = (3 / 2 * (a n - 1))) :
  ∀ n : ℕ, 1 ≤ n → a n = 3^n :=
by
  sorry

end NUMINAMATH_GPT_sequence_expression_l1158_115835


namespace NUMINAMATH_GPT_electricity_cost_one_kilometer_minimum_electricity_kilometers_l1158_115825

-- Part 1: Cost of traveling one kilometer using electricity only
theorem electricity_cost_one_kilometer (x : ℝ) (fuel_cost : ℝ) (electricity_cost : ℝ) 
  (total_fuel_cost : ℝ) (total_electricity_cost : ℝ) 
  (fuel_per_km_more_than_electricity : ℝ) (distance_fuel : ℝ) (distance_electricity : ℝ)
  (h1 : total_fuel_cost = distance_fuel * fuel_cost)
  (h2 : total_electricity_cost = distance_electricity * electricity_cost)
  (h3 : fuel_per_km_more_than_electricity = 0.5)
  (h4 : fuel_cost = electricity_cost + fuel_per_km_more_than_electricity)
  (h5 : distance_fuel = 76 / (electricity_cost + 0.5))
  (h6 : distance_electricity = 26 / electricity_cost) : 
  x = 0.26 :=
sorry

-- Part 2: Minimum kilometers traveled using electricity
theorem minimum_electricity_kilometers (total_trip_cost : ℝ) (electricity_per_km : ℝ) 
  (hybrid_total_km : ℝ) (max_total_cost : ℝ) (fuel_per_km : ℝ) (y : ℝ)
  (h1 : electricity_per_km = 0.26)
  (h2 : fuel_per_km = 0.26 + 0.5)
  (h3 : hybrid_total_km = 100)
  (h4 : max_total_cost = 39)
  (h5 : total_trip_cost = electricity_per_km * y + (hybrid_total_km - y) * fuel_per_km)
  (h6 : total_trip_cost ≤ max_total_cost) :
  y ≥ 74 :=
sorry

end NUMINAMATH_GPT_electricity_cost_one_kilometer_minimum_electricity_kilometers_l1158_115825


namespace NUMINAMATH_GPT_distinct_nonzero_reals_satisfy_equation_l1158_115816

open Real

theorem distinct_nonzero_reals_satisfy_equation
  (a b c : ℝ) (h₁ : a ≠ b) (h₂ : b ≠ c) (h₃ : c ≠ a) (h₄ : a ≠ 0) (h₅ : b ≠ 0) (h₆ : c ≠ 0)
  (h₇ : a + 2 / b = b + 2 / c) (h₈ : b + 2 / c = c + 2 / a) :
  (a + 2 / b) ^ 2 + (b + 2 / c) ^ 2 + (c + 2 / a) ^ 2 = 6 :=
sorry

end NUMINAMATH_GPT_distinct_nonzero_reals_satisfy_equation_l1158_115816
