import Mathlib

namespace exists_nat_square_starting_with_digits_l627_62787

theorem exists_nat_square_starting_with_digits (S : ℕ) : 
  ∃ (N k : ℕ), S * 10^k ≤ N^2 ∧ N^2 < (S + 1) * 10^k := 
by {
  sorry
}

end exists_nat_square_starting_with_digits_l627_62787


namespace triangle_with_incircle_radius_one_has_sides_5_4_3_l627_62770

variable {a b c : ℕ} (h1 : a ≥ b ∧ b ≥ c)
variable (h2 : ∃ (a b c : ℕ), (a + b + c) / 2 * 1 = (a + b + c) / 2 * ((a + b + c) / 2 - a) * ((a + b + c) / 2 - b) * ((a + b + c) / 2 - c))

theorem triangle_with_incircle_radius_one_has_sides_5_4_3 :
  a = 5 ∧ b = 4 ∧ c = 3 :=
by
    sorry

end triangle_with_incircle_radius_one_has_sides_5_4_3_l627_62770


namespace tangent_circles_m_values_l627_62745

noncomputable def is_tangent (m : ℝ) : Prop :=
  let o1_center := (m, 0)
  let o2_center := (-1, 2 * m)
  let distance := Real.sqrt ((m + 1)^2 + (2 * m)^2)
  (distance = 5 ∨ distance = 1)

theorem tangent_circles_m_values :
  {m : ℝ | is_tangent m} = {-12 / 5, -2 / 5, 0, 2} := by
  sorry

end tangent_circles_m_values_l627_62745


namespace reflection_over_line_y_eq_x_l627_62768

theorem reflection_over_line_y_eq_x {x y x' y' : ℝ} (h_c : (x, y) = (6, -5)) (h_reflect : (x', y') = (y, x)) :
  (x', y') = (-5, 6) :=
  by
    simp [h_c, h_reflect]
    sorry

end reflection_over_line_y_eq_x_l627_62768


namespace combined_avg_score_l627_62704

-- Define the average scores
def avg_score_u : ℕ := 65
def avg_score_b : ℕ := 80
def avg_score_c : ℕ := 77

-- Define the ratio of the number of students
def ratio_u : ℕ := 4
def ratio_b : ℕ := 6
def ratio_c : ℕ := 5

-- Prove the combined average score
theorem combined_avg_score : (ratio_u * avg_score_u + ratio_b * avg_score_b + ratio_c * avg_score_c) / (ratio_u + ratio_b + ratio_c) = 75 :=
by
  sorry

end combined_avg_score_l627_62704


namespace min_width_for_fence_area_least_200_l627_62781

theorem min_width_for_fence_area_least_200 (w : ℝ) (h : w * (w + 20) ≥ 200) : w ≥ 10 :=
sorry

end min_width_for_fence_area_least_200_l627_62781


namespace p_plus_q_identity_l627_62729

variable {α : Type*} [CommRing α]

-- Definitions derived from conditions
def p (x : α) : α := 3 * (x - 2)
def q (x : α) : α := (x + 2) * (x - 4)

-- Lean theorem stating the problem
theorem p_plus_q_identity (x : α) : p x + q x = x^2 + x - 14 :=
by
  unfold p q
  sorry

end p_plus_q_identity_l627_62729


namespace speed_in_still_water_l627_62795

-- Defining the terms as given conditions in the problem
def speed_downstream (v_m v_s : ℝ) : ℝ := v_m + v_s
def speed_upstream (v_m v_s : ℝ) : ℝ := v_m - v_s

-- Given conditions translated into Lean definitions
def downstream_condition : Prop :=
  ∃ (v_m v_s : ℝ), speed_downstream v_m v_s = 7

def upstream_condition : Prop :=
  ∃ (v_m v_s : ℝ), speed_upstream v_m v_s = 4

-- The problem statement to prove
theorem speed_in_still_water : 
  downstream_condition ∧ upstream_condition → ∃ v_m : ℝ, v_m = 5.5 :=
by 
  intros
  sorry

end speed_in_still_water_l627_62795


namespace integer_not_always_greater_decimal_l627_62753

-- Definitions based on conditions
def is_decimal (d : ℚ) : Prop :=
  ∃ (i : ℤ) (f : ℚ), 0 ≤ f ∧ f < 1 ∧ d = i + f

def is_greater (a : ℤ) (b : ℚ) : Prop :=
  (a : ℚ) > b

theorem integer_not_always_greater_decimal : ¬ ∀ n : ℤ, ∀ d : ℚ, is_decimal d → (is_greater n d) :=
by
  sorry

end integer_not_always_greater_decimal_l627_62753


namespace trip_time_difference_l627_62784

def travel_time (distance speed : ℕ) : ℕ :=
  distance / speed

theorem trip_time_difference
  (speed : ℕ)
  (speed_pos : 0 < speed)
  (distance1 : ℕ)
  (distance2 : ℕ)
  (time_difference : ℕ)
  (h1 : distance1 = 540)
  (h2 : distance2 = 600)
  (h_speed : speed = 60)
  (h_time_diff : time_difference = (travel_time distance2 speed) - (travel_time distance1 speed) * 60)
  : time_difference = 60 :=
by
  sorry

end trip_time_difference_l627_62784


namespace elizabeth_wedding_gift_cost_l627_62700

-- Defining the given conditions
def cost_steak_knife_set : ℝ := 80.00
def num_steak_knife_sets : ℝ := 2
def cost_dinnerware_set : ℝ := 200.00
def discount_rate : ℝ := 0.10
def sales_tax_rate : ℝ := 0.05

-- Calculating total expense
def total_cost (cost_steak_knife_set num_steak_knife_sets cost_dinnerware_set : ℝ) : ℝ :=
  (cost_steak_knife_set * num_steak_knife_sets) + cost_dinnerware_set

def discounted_price (total_cost discount_rate : ℝ) : ℝ :=
  total_cost - (total_cost * discount_rate)

def final_price (discounted_price sales_tax_rate : ℝ) : ℝ :=
  discounted_price + (discounted_price * sales_tax_rate)

def elizabeth_spends (cost_steak_knife_set num_steak_knife_sets cost_dinnerware_set discount_rate sales_tax_rate : ℝ) : ℝ :=
  final_price (discounted_price (total_cost cost_steak_knife_set num_steak_knife_sets cost_dinnerware_set) discount_rate) sales_tax_rate

theorem elizabeth_wedding_gift_cost
  (cost_steak_knife_set : ℝ)
  (num_steak_knife_sets : ℝ)
  (cost_dinnerware_set : ℝ)
  (discount_rate : ℝ)
  (sales_tax_rate : ℝ) :
  elizabeth_spends cost_steak_knife_set num_steak_knife_sets cost_dinnerware_set discount_rate sales_tax_rate = 340.20 := 
by
  sorry -- Proof is to be completed

end elizabeth_wedding_gift_cost_l627_62700


namespace p_is_contradictory_to_q_l627_62736

variable (a : ℝ)

def p := a > 0 → a^2 ≠ 0
def q := a ≤ 0 → a^2 = 0

theorem p_is_contradictory_to_q : (p a) ↔ ¬ (q a) :=
by
  sorry

end p_is_contradictory_to_q_l627_62736


namespace dihedral_angles_pyramid_l627_62707

noncomputable def dihedral_angles (a b : ℝ) : ℝ × ℝ :=
  let alpha := Real.arccos ((a * Real.sqrt 3) / Real.sqrt (4 * b ^ 2 - a ^ 2))
  let gamma := 2 * Real.arctan (b / Real.sqrt (4 * b ^ 2 - a ^ 2))
  (alpha, gamma)

theorem dihedral_angles_pyramid (a b alpha gamma : ℝ) (h1 : a > 0) (h2 : b > 0) :
  dihedral_angles a b = (alpha, gamma) :=
sorry

end dihedral_angles_pyramid_l627_62707


namespace total_cost_second_set_l627_62786

variable (A V : ℝ)

-- Condition declarations
axiom cost_video_cassette : V = 300
axiom cost_second_set : 7 * A + 3 * V = 1110

-- Proof goal
theorem total_cost_second_set :
  7 * A + 3 * V = 1110 :=
by
  sorry

end total_cost_second_set_l627_62786


namespace derivative_at_zero_l627_62738

noncomputable def f (x : ℝ) : ℝ :=
  if x ≠ 0 then (Real.log (1 + 2 * x^2 + x^3)) / x else 0

theorem derivative_at_zero : deriv f 0 = 2 := by
  sorry

end derivative_at_zero_l627_62738


namespace unique_tangent_lines_through_point_l627_62785

theorem unique_tangent_lines_through_point (P : ℝ × ℝ) (hP : P = (2, 4)) :
  ∃! l : ℝ × ℝ → Prop, (l P) ∧ (∀ p : ℝ × ℝ, l p → p ∈ {p : ℝ × ℝ | p.2 ^ 2 = 8 * p.1}) := sorry

end unique_tangent_lines_through_point_l627_62785


namespace molecular_weight_is_correct_l627_62798

noncomputable def molecular_weight_of_compound : ℝ :=
  3 * 39.10 + 2 * 51.996 + 7 * 15.999 + 4 * 1.008 + 1 * 14.007

theorem molecular_weight_is_correct : molecular_weight_of_compound = 351.324 := 
by
  sorry

end molecular_weight_is_correct_l627_62798


namespace union_sets_l627_62772

-- Define the sets A and B
def A : Set ℤ := {0, 1}
def B : Set ℤ := {x : ℤ | (x + 2) * (x - 1) < 0}

-- The theorem to be proven
theorem union_sets : A ∪ B = {-1, 0, 1} :=
by
  sorry

end union_sets_l627_62772


namespace factorization_correct_l627_62723

def expression (x : ℝ) : ℝ := 16 * x^3 + 4 * x^2
def factored_expression (x : ℝ) : ℝ := 4 * x^2 * (4 * x + 1)

theorem factorization_correct (x : ℝ) : expression x = factored_expression x := 
by 
  sorry

end factorization_correct_l627_62723


namespace operation_multiplication_in_P_l627_62714

-- Define the set P
def P : Set ℕ := {n | ∃ k : ℕ, n = k^2}

-- Define the operation "*" as multiplication within the set P
def operation (a b : ℕ) : ℕ := a * b

-- Define the property to be proved
theorem operation_multiplication_in_P (a b : ℕ)
  (ha : a ∈ P) (hb : b ∈ P) : operation a b ∈ P :=
sorry

end operation_multiplication_in_P_l627_62714


namespace river_flow_volume_l627_62765

noncomputable def river_depth : ℝ := 2
noncomputable def river_width : ℝ := 45
noncomputable def flow_rate_kmph : ℝ := 4
noncomputable def flow_rate_mpm := flow_rate_kmph * 1000 / 60
noncomputable def cross_sectional_area := river_depth * river_width
noncomputable def volume_per_minute := cross_sectional_area * flow_rate_mpm

theorem river_flow_volume :
  volume_per_minute = 6000.3 := by
  sorry

end river_flow_volume_l627_62765


namespace Seth_bought_20_cartons_of_ice_cream_l627_62782

-- Definitions from conditions
def ice_cream_cost_per_carton : ℕ := 6
def yogurt_cost_per_carton : ℕ := 1
def num_yogurt_cartons : ℕ := 2
def extra_amount_spent_on_ice_cream : ℕ := 118

-- Let x be the number of cartons of ice cream Seth bought
def num_ice_cream_cartons (x : ℕ) : Prop :=
  ice_cream_cost_per_carton * x = num_yogurt_cartons * yogurt_cost_per_carton + extra_amount_spent_on_ice_cream

-- The proof goal
theorem Seth_bought_20_cartons_of_ice_cream : num_ice_cream_cartons 20 :=
by
  unfold num_ice_cream_cartons
  unfold ice_cream_cost_per_carton yogurt_cost_per_carton num_yogurt_cartons extra_amount_spent_on_ice_cream
  sorry

end Seth_bought_20_cartons_of_ice_cream_l627_62782


namespace add_congruence_mul_congruence_l627_62711

namespace ModularArithmetic

-- Define the congruence relation mod m
def is_congruent_mod (a b m : ℤ) : Prop := ∃ k : ℤ, a - b = k * m

-- Part (a): Proving a + c ≡ b + d (mod m)
theorem add_congruence {a b c d m : ℤ}
  (h₁ : is_congruent_mod a b m)
  (h₂ : is_congruent_mod c d m) :
  is_congruent_mod (a + c) (b + d) m :=
  sorry

-- Part (b): Proving a ⋅ c ≡ b ⋅ d (mod m)
theorem mul_congruence {a b c d m : ℤ}
  (h₁ : is_congruent_mod a b m)
  (h₂ : is_congruent_mod c d m) :
  is_congruent_mod (a * c) (b * d) m :=
  sorry

end ModularArithmetic

end add_congruence_mul_congruence_l627_62711


namespace remainder_when_divided_by_7_l627_62731

theorem remainder_when_divided_by_7
  (x : ℤ) (k : ℤ) (h : x = 52 * k + 19) : x % 7 = 5 :=
sorry

end remainder_when_divided_by_7_l627_62731


namespace jason_remaining_pokemon_cards_l627_62773

theorem jason_remaining_pokemon_cards :
  (3 - 2) = 1 :=
by 
  sorry

end jason_remaining_pokemon_cards_l627_62773


namespace equation_has_real_roots_l627_62755

theorem equation_has_real_roots (k : ℝ) : ∀ (x : ℝ), 
  ∃ x, x = k^2 * (x - 1) * (x - 2) :=
by {
  sorry
}

end equation_has_real_roots_l627_62755


namespace equivalent_proof_problem_l627_62721

theorem equivalent_proof_problem (x : ℤ) (h : (x - 5) / 7 = 7) : (x - 14) / 10 = 4 :=
by
  sorry

end equivalent_proof_problem_l627_62721


namespace area_contained_by_graph_l627_62792

theorem area_contained_by_graph (x y : ℝ) :
  (|x + y| + |x - y| ≤ 6) → (36 = 36) := by
  sorry

end area_contained_by_graph_l627_62792


namespace original_number_l627_62778

theorem original_number (x : ℝ) (h : x * 1.5 = 105) : x = 70 :=
sorry

end original_number_l627_62778


namespace probability_at_least_one_trip_l627_62722

theorem probability_at_least_one_trip (p_A_trip : ℚ) (p_B_trip : ℚ)
  (h1 : p_A_trip = 1/4) (h2 : p_B_trip = 1/5) :
  (1 - ((1 - p_A_trip) * (1 - p_B_trip))) = 2/5 :=
by
  sorry

end probability_at_least_one_trip_l627_62722


namespace inequality_proof_l627_62728

theorem inequality_proof (x y z : ℝ) (hx : x < 0) (hy : y < 0) (hz : z < 0) :
    (x * y * z) / ((1 + 5 * x) * (4 * x + 3 * y) * (5 * y + 6 * z) * (z + 18)) ≤ (1 : ℝ) / 5120 := 
by
  sorry

end inequality_proof_l627_62728


namespace correct_factorization_A_l627_62760

theorem correct_factorization_A (x : ℝ) : x^2 - 4 * x + 4 = (x - 2)^2 :=
by sorry

end correct_factorization_A_l627_62760


namespace ratio_of_slices_l627_62777

theorem ratio_of_slices
  (initial_slices : ℕ)
  (slices_eaten_for_lunch : ℕ)
  (remaining_slices_after_lunch : ℕ)
  (slices_left_for_tomorrow : ℕ)
  (slices_eaten_for_dinner : ℕ)
  (ratio : ℚ) :
  initial_slices = 12 → 
  slices_eaten_for_lunch = initial_slices / 2 →
  remaining_slices_after_lunch = initial_slices - slices_eaten_for_lunch →
  slices_left_for_tomorrow = 4 →
  slices_eaten_for_dinner = remaining_slices_after_lunch - slices_left_for_tomorrow →
  ratio = (slices_eaten_for_dinner : ℚ) / remaining_slices_after_lunch →
  ratio = 1 / 3 :=
by sorry

end ratio_of_slices_l627_62777


namespace find_x_l627_62780

theorem find_x (x y : ℝ) (h1 : x - y = 8) (h2 : x + y = 10) : x = 9 := 
by 
sorry

end find_x_l627_62780


namespace solution_l627_62720

noncomputable def f (a b x : ℝ) : ℝ := a * Real.sin x + b * 3 * x + 4

def problem (a b : ℝ) (m : ℝ) (h1 : f a b m = 5) (h2 : m = Real.logb 3 10) : Prop :=
  f a b (-Real.logb 3 3) = 3

theorem solution (a b : ℝ) (m : ℝ) (h1 : f a b m = 5) (h2 : m = Real.logb 3 10) : problem a b m h1 h2 :=
sorry

end solution_l627_62720


namespace strategy_probabilities_l627_62708

noncomputable def P1 : ℚ := 1 / 3
noncomputable def P2 : ℚ := 1 / 2
noncomputable def P3 : ℚ := 2 / 3

theorem strategy_probabilities :
  (P1 < P2) ∧
  (P1 < P3) ∧
  (2 * P1 = P3) := by
  sorry

end strategy_probabilities_l627_62708


namespace sum_of_cube_angles_l627_62716

theorem sum_of_cube_angles (W X Y Z : Point) (cube : Cube)
  (angle_WXY angle_XYZ angle_YZW angle_ZWX : ℝ)
  (h₁ : angle_WXY = 90)
  (h₂ : angle_XYZ = 90)
  (h₃ : angle_YZW = 90)
  (h₄ : angle_ZWX = 60) :
  angle_WXY + angle_XYZ + angle_YZW + angle_ZWX = 330 := by
  sorry

end sum_of_cube_angles_l627_62716


namespace find_number_l627_62793

theorem find_number (x : ℝ) (h : 0.5 * x = 0.25 * x + 2) : x = 8 :=
by
  sorry

end find_number_l627_62793


namespace midpoint_of_five_points_on_grid_l627_62740

theorem midpoint_of_five_points_on_grid 
    (points : Fin 5 → ℤ × ℤ) :
    ∃ i j : Fin 5, i ≠ j ∧ ((points i).fst + (points j).fst) % 2 = 0 
    ∧ ((points i).snd + (points j).snd) % 2 = 0 :=
by sorry

end midpoint_of_five_points_on_grid_l627_62740


namespace solve_quadratic_l627_62705

theorem solve_quadratic (x : ℝ) (h1 : 2 * x^2 - 6 * x = 0) (h2 : x ≠ 0) : x = 3 := by
  sorry

end solve_quadratic_l627_62705


namespace find_c_l627_62715

theorem find_c (c : ℝ) (h : (-c / 4) + (-c / 7) = 22) : c = -56 :=
by
  sorry

end find_c_l627_62715


namespace equality_condition_l627_62737

theorem equality_condition (a b c : ℝ) :
  a + b + c = (a + b) * (a + c) → a = 1 ∧ b = 1 ∧ c = 1 :=
by
  sorry

end equality_condition_l627_62737


namespace find_other_number_l627_62726

theorem find_other_number 
  {A B : ℕ} 
  (h_A : A = 24)
  (h_hcf : Nat.gcd A B = 14)
  (h_lcm : Nat.lcm A B = 312) :
  B = 182 :=
by
  -- Proof skipped
  sorry

end find_other_number_l627_62726


namespace initial_avg_weight_l627_62744

theorem initial_avg_weight (A : ℝ) (h : 6 * A + 121 = 7 * 151) : A = 156 :=
by
sorry

end initial_avg_weight_l627_62744


namespace other_x_intercept_l627_62748

def foci1 := (0, -3)
def foci2 := (4, 0)
def x_intercept1 := (0, 0)

theorem other_x_intercept :
  (∃ x : ℝ, (|x - 4| + |-3| * x = 7)) → x = 11 / 4 := by
  sorry

end other_x_intercept_l627_62748


namespace moles_of_CO2_formed_l627_62796

variables (CH4 O2 C2H2 CO2 H2O : Type)
variables (nCH4 nO2 nC2H2 nCO2 : ℕ)
variables (reactsCompletely : Prop)

-- Balanced combustion equations
axiom combustion_methane : ∀ (mCH4 mO2 mCO2 mH2O : ℕ), mCH4 = 1 → mO2 = 2 → mCO2 = 1 → mH2O = 2 → Prop
axiom combustion_acetylene : ∀ (aC2H2 aO2 aCO2 aH2O : ℕ), aC2H2 = 2 → aO2 = 5 → aCO2 = 4 → aH2O = 2 → Prop

-- Given conditions
axiom conditions :
  nCH4 = 3 ∧ nO2 = 6 ∧ nC2H2 = 2 ∧ reactsCompletely

-- Prove the number of moles of CO2 formed
theorem moles_of_CO2_formed : 
  (nCH4 = 3 ∧ nO2 = 6 ∧ nC2H2 = 2 ∧ reactsCompletely) →
  nCO2 = 3
:= by
  intros h
  sorry

end moles_of_CO2_formed_l627_62796


namespace a_2023_value_l627_62733

theorem a_2023_value :
  ∀ (a : ℕ → ℚ),
  a 1 = 5 ∧
  a 2 = 5 / 11 ∧
  (∀ n, 3 ≤ n → a n = (a (n - 2)) * (a (n - 1)) / (3 * (a (n - 2)) - (a (n - 1)))) →
  a 2023 = 5 / 10114 ∧ 5 + 10114 = 10119 :=
by
  sorry

end a_2023_value_l627_62733


namespace number_of_taxis_l627_62724

-- Define the conditions explicitly
def number_of_cars : ℕ := 3
def people_per_car : ℕ := 4
def number_of_vans : ℕ := 2
def people_per_van : ℕ := 5
def people_per_taxi : ℕ := 6
def total_people : ℕ := 58

-- Define the number of people in cars and vans
def people_in_cars := number_of_cars * people_per_car
def people_in_vans := number_of_vans * people_per_van
def people_in_taxis := total_people - (people_in_cars + people_in_vans)

-- The theorem we need to prove
theorem number_of_taxis : people_in_taxis / people_per_taxi = 6 := by
  sorry

end number_of_taxis_l627_62724


namespace ashley_loan_least_months_l627_62797

theorem ashley_loan_least_months (t : ℕ) (principal : ℝ) (interest_rate : ℝ) (triple_principal : ℝ) : 
  principal = 1500 ∧ interest_rate = 0.06 ∧ triple_principal = 3 * principal → 
  1.06^t > triple_principal → t = 20 :=
by
  intro h h2
  sorry

end ashley_loan_least_months_l627_62797


namespace compute_binom_product_l627_62794

/-- Definition of binomial coefficient -/
def binomial (n k : ℕ) : ℕ := n.choose k

/-- The main theorem to prove -/
theorem compute_binom_product : binomial 10 3 * binomial 8 3 = 6720 :=
by
  sorry

end compute_binom_product_l627_62794


namespace current_time_l627_62734

theorem current_time (t : ℝ) 
  (h1 : 6 * (t + 10) - (90 + 0.5 * (t - 5)) = 90 ∨ 6 * (t + 10) - (90 + 0.5 * (t - 5)) = -90) :
  t = 3 + 11 / 60 := sorry

end current_time_l627_62734


namespace sqrt_meaningful_condition_l627_62735

theorem sqrt_meaningful_condition (x : ℝ) : (2 * x + 6 >= 0) ↔ (x >= -3) := by
  sorry

end sqrt_meaningful_condition_l627_62735


namespace prove_x_minus_y_squared_l627_62776

noncomputable section

variables {x y a b : ℝ}

theorem prove_x_minus_y_squared (h1 : x * y = b) (h2 : x / y + y / x = a) : (x - y) ^ 2 = a * b - 2 * b := 
  sorry

end prove_x_minus_y_squared_l627_62776


namespace spent_on_new_tires_is_correct_l627_62706

-- Conditions
def amount_spent_on_speakers : ℝ := 136.01
def amount_spent_on_cd_player : ℝ := 139.38
def total_amount_spent : ℝ := 387.85

-- Goal
def amount_spent_on_tires : ℝ := total_amount_spent - (amount_spent_on_speakers + amount_spent_on_cd_player)

theorem spent_on_new_tires_is_correct : 
  amount_spent_on_tires = 112.46 :=
by
  sorry

end spent_on_new_tires_is_correct_l627_62706


namespace general_term_is_correct_l627_62701

variable (a : ℕ → ℤ)
variable (n : ℕ)

def is_arithmetic_sequence := ∃ d a₁, ∀ n, a n = a₁ + d * (n - 1)

axiom a_10_eq_30 : a 10 = 30
axiom a_20_eq_50 : a 20 = 50

noncomputable def general_term (n : ℕ) : ℤ := 2 * n + 10

theorem general_term_is_correct (a: ℕ → ℤ)
  (h1 : is_arithmetic_sequence a)
  (h2 : a 10 = 30)
  (h3 : a 20 = 50)
  : ∀ n, a n = general_term n :=
sorry

end general_term_is_correct_l627_62701


namespace find_a4_l627_62713

-- Given expression of x^5
def polynomial_expansion (x a_0 a_1 a_2 a_3 a_4 a_5 : ℝ) : Prop :=
  x^5 = a_0 + a_1 * (x+1) + a_2 * (x+1)^2 + a_3 * (x+1)^3 + a_4 * (x+1)^4 + a_5 * (x+1)^5

theorem find_a4 (x a_0 a_1 a_2 a_3 a_4 a_5 : ℝ) (h : polynomial_expansion x a_0 a_1 a_2 a_3 a_4 a_5) : a_4 = -5 :=
  sorry

end find_a4_l627_62713


namespace tangent_line_to_circle_l627_62759

theorem tangent_line_to_circle :
  ∀ (x y : ℝ), x^2 + y^2 = 5 → (x = 2 → y = -1 → 2 * x - y - 5 = 0) :=
by
  intros x y h_circle hx hy
  sorry

end tangent_line_to_circle_l627_62759


namespace maximum_illuminated_surfaces_l627_62712

noncomputable def optimal_position (r R d : ℝ) (h : d > r + R) : ℝ :=
  d / (1 + Real.sqrt (R^3 / r^3))

theorem maximum_illuminated_surfaces (r R d : ℝ) (h : d > r + R) (h1 : r ≤ optimal_position r R d h) (h2 : optimal_position r R d h ≤ d - R) :
  (optimal_position r R d h = d / (1 + Real.sqrt (R^3 / r^3))) ∨ (optimal_position r R d h = r) :=
sorry

end maximum_illuminated_surfaces_l627_62712


namespace sin_C_eq_63_over_65_l627_62730

theorem sin_C_eq_63_over_65 (A B C : Real) (h₁ : 0 < A) (h₂ : A < π)
  (h₃ : 0 < B) (h₄ : B < π) (h₅ : 0 < C) (h₆ : C < π)
  (h₇ : A + B + C = π)
  (h₈ : Real.sin A = 5 / 13) (h₉ : Real.cos B = 3 / 5) : Real.sin C = 63 / 65 := 
by
  sorry

end sin_C_eq_63_over_65_l627_62730


namespace calculation_equals_106_25_l627_62750

noncomputable def calculation : ℝ := 2.5 * 8.5 * (5.2 - 0.2)

theorem calculation_equals_106_25 : calculation = 106.25 := 
by
  sorry

end calculation_equals_106_25_l627_62750


namespace geometric_sequence_third_term_l627_62766

theorem geometric_sequence_third_term :
  ∀ (a r : ℕ), a = 2 ∧ a * r ^ 3 = 162 → a * r ^ 2 = 18 :=
by
  intros a r
  intro h
  have ha : a = 2 := h.1
  have h_fourth_term : a * r ^ 3 = 162 := h.2
  sorry

end geometric_sequence_third_term_l627_62766


namespace additional_investment_l627_62761

-- Given the conditions
variables (x y : ℝ)
def interest_rate_1 := 0.02
def interest_rate_2 := 0.04
def invested_amount := 1000
def total_interest := 92

-- Theorem to prove
theorem additional_investment : 
  0.02 * invested_amount + 0.04 * (invested_amount + y) = total_interest → 
  y = 800 :=
by
  sorry

end additional_investment_l627_62761


namespace number_of_coins_l627_62763

-- Define the conditions
def equal_number_of_coins (x : ℝ) :=
  ∃ n : ℝ, n = x

-- Define the total value condition
def total_value (x : ℝ) :=
  x + 0.50 * x + 0.25 * x = 70

-- The theorem to be proved
theorem number_of_coins (x : ℝ) (h1 : equal_number_of_coins x) (h2 : total_value x) : x = 40 :=
by sorry

end number_of_coins_l627_62763


namespace sum_of_properly_paintable_numbers_l627_62719

-- Definitions based on conditions
def properly_paintable (a b c : ℕ) : Prop :=
  ∀ n : ℕ, (n % a = 0 ∧ n % b ≠ 1 ∧ n % c ≠ 3) ∨
           (n % a ≠ 0 ∧ n % b = 1 ∧ n % c ≠ 3) ∨
           (n % a ≠ 0 ∧ n % b ≠ 1 ∧ n % c = 3) → n < 100

-- Main theorem to prove
theorem sum_of_properly_paintable_numbers : 
  (properly_paintable 3 3 6) ∧ (properly_paintable 4 2 8) → 
  100 * 3 + 10 * 3 + 6 + 100 * 4 + 10 * 2 + 8 = 764 :=
by
  sorry  -- The proof goes here, but it's not required

-- Note: The actual condition checks in the definition of properly_paintable 
-- might need more detailed splits into depending on specific post visits and a 
-- more rigorous formalization to comply with the exact checking as done above. 
-- This definition is a simplified logical structure to represent the condition.


end sum_of_properly_paintable_numbers_l627_62719


namespace pythagorean_triple_correct_l627_62752

def is_pythagorean_triple (a b c : ℕ) : Prop :=
  a^2 + b^2 = c^2

theorem pythagorean_triple_correct : 
  is_pythagorean_triple 9 12 15 ∧ ¬ is_pythagorean_triple 3 4 6 ∧ ¬ is_pythagorean_triple 1 2 3 ∧ ¬ is_pythagorean_triple 6 12 13 :=
by
  sorry

end pythagorean_triple_correct_l627_62752


namespace band_gigs_count_l627_62769

-- Definitions of earnings per role and total earnings
def leadSingerEarnings := 30
def guitaristEarnings := 25
def bassistEarnings := 20
def drummerEarnings := 25
def keyboardistEarnings := 20
def backupSingerEarnings := 15
def totalEarnings := 2055

-- Calculate total per gig earnings
def totalPerGigEarnings :=
  leadSingerEarnings + guitaristEarnings + bassistEarnings + drummerEarnings + keyboardistEarnings + backupSingerEarnings

-- Statement to prove the number of gigs played is 15
theorem band_gigs_count :
  totalEarnings / totalPerGigEarnings = 15 := 
by { sorry }

end band_gigs_count_l627_62769


namespace joseph_total_cost_l627_62717

variable (cost_refrigerator cost_water_heater cost_oven : ℝ)

-- Conditions
axiom h1 : cost_refrigerator = 3 * cost_water_heater
axiom h2 : cost_oven = 500
axiom h3 : cost_oven = 2 * cost_water_heater

-- Theorem
theorem joseph_total_cost : cost_refrigerator + cost_water_heater + cost_oven = 1500 := by
  sorry

end joseph_total_cost_l627_62717


namespace car_dealership_l627_62725

variable (sportsCars : ℕ) (sedans : ℕ) (trucks : ℕ)

theorem car_dealership (h1 : 3 * sedans = 5 * sportsCars) 
  (h2 : 3 * trucks = 3 * sportsCars) 
  (h3 : sportsCars = 45) : 
  sedans = 75 ∧ trucks = 45 := by
  sorry

end car_dealership_l627_62725


namespace transformation_result_l627_62718

noncomputable def initial_function (x : ℝ) : ℝ := Real.sin (2 * x)

noncomputable def translate_left (f : ℝ → ℝ) (a : ℝ) : ℝ → ℝ :=
  λ x => f (x + a)

noncomputable def compress_horizontal (f : ℝ → ℝ) (k : ℝ) : ℝ → ℝ :=
  λ x => f (k * x)

theorem transformation_result :
  (compress_horizontal (translate_left initial_function (Real.pi / 3)) 2) x = Real.sin (4 * x + (2 * Real.pi / 3)) :=
sorry

end transformation_result_l627_62718


namespace highest_avg_speed_2_to_3_l627_62749

-- Define the time periods and distances traveled in those periods
def distance_8_to_9 : ℕ := 50
def distance_9_to_10 : ℕ := 70
def distance_10_to_11 : ℕ := 60
def distance_2_to_3 : ℕ := 80
def distance_3_to_4 : ℕ := 40

-- Define the average speed calculation for each period
def avg_speed (distance : ℕ) (hours : ℕ) : ℕ := distance / hours

-- Proposition stating that the highest average speed is from 2 pm to 3 pm
theorem highest_avg_speed_2_to_3 : 
  avg_speed distance_2_to_3 1 > avg_speed distance_8_to_9 1 ∧ 
  avg_speed distance_2_to_3 1 > avg_speed distance_9_to_10 1 ∧ 
  avg_speed distance_2_to_3 1 > avg_speed distance_10_to_11 1 ∧ 
  avg_speed distance_2_to_3 1 > avg_speed distance_3_to_4 1 := 
by 
  sorry

end highest_avg_speed_2_to_3_l627_62749


namespace d_divisibility_l627_62742

theorem d_divisibility (p d : ℕ) (h_p : 0 < p) (h_d : 0 < d)
  (h1 : Prime p) 
  (h2 : Prime (p + d)) 
  (h3 : Prime (p + 2 * d)) 
  (h4 : Prime (p + 3 * d)) 
  (h5 : Prime (p + 4 * d)) 
  (h6 : Prime (p + 5 * d)) : 
  (2 ∣ d) ∧ (3 ∣ d) ∧ (5 ∣ d) :=
by
  sorry

end d_divisibility_l627_62742


namespace solve_for_x_l627_62732

theorem solve_for_x (x : ℤ) (h : 2 / (x + 3) + 3 * x / (x + 3) - 5 / (x + 3) = 4) : 
  x = -15 := 
by
  sorry

end solve_for_x_l627_62732


namespace meeting_distance_from_top_l627_62767

section

def total_distance : ℝ := 12
def uphill_distance : ℝ := 6
def downhill_distance : ℝ := 6
def john_start_time : ℝ := 0.25
def john_uphill_speed : ℝ := 12
def john_downhill_speed : ℝ := 18
def jenny_uphill_speed : ℝ := 14
def jenny_downhill_speed : ℝ := 21

theorem meeting_distance_from_top : 
  ∃ (d : ℝ), d = 6 - 14 * ((0.25) + 6 / 14 - (1 / 2) - (6 - 18 * ((1 / 2) + d / 18))) / 14 ∧ d = 45 / 32 :=
sorry

end

end meeting_distance_from_top_l627_62767


namespace both_players_same_score_probability_l627_62790

theorem both_players_same_score_probability :
  let p_A_score := 0.6
  let p_B_score := 0.8
  let p_A_miss := 1 - p_A_score
  let p_B_miss := 1 - p_B_score
  (p_A_score * p_B_score + p_A_miss * p_B_miss = 0.56) :=
by
  sorry

end both_players_same_score_probability_l627_62790


namespace trim_hedges_purpose_l627_62774

-- Given possible answers
inductive Answer
| A : Answer
| B : Answer
| C : Answer
| D : Answer

-- Define the purpose of trimming hedges
def trimmingHedges : Answer :=
  Answer.B

-- Formal problem statement
theorem trim_hedges_purpose : trimmingHedges = Answer.B :=
  sorry

end trim_hedges_purpose_l627_62774


namespace exponentiation_rule_l627_62702

theorem exponentiation_rule (a : ℝ) : (a^4) * (a^4) = a^8 :=
by 
  sorry

end exponentiation_rule_l627_62702


namespace polygon_sides_l627_62754

theorem polygon_sides :
  ∀ (n : ℕ), (n > 2) → (n - 2) * 180 < 360 → n = 3 :=
by
  intros n hn1 hn2
  sorry

end polygon_sides_l627_62754


namespace largest_base5_to_base7_l627_62758

-- Define the largest four-digit number in base-5
def largest_base5_four_digit_number : ℕ := 4 * 5^3 + 4 * 5^2 + 4 * 5^1 + 4 * 5^0

-- Convert this number to base-7
def convert_to_base7 (n : ℕ) : ℕ := 
  let d3 := n / (7^3)
  let r3 := n % (7^3)
  let d2 := r3 / (7^2)
  let r2 := r3 % (7^2)
  let d1 := r2 / (7^1)
  let r1 := r2 % (7^1)
  let d0 := r1
  (d3 * 10^3) + (d2 * 10^2) + (d1 * 10^1) + d0

-- Theorem to prove m in base-7
theorem largest_base5_to_base7 : 
  convert_to_base7 largest_base5_four_digit_number = 1551 :=
by 
  -- skip the proof
  sorry

end largest_base5_to_base7_l627_62758


namespace barbata_interest_rate_l627_62709

theorem barbata_interest_rate
  (initial_investment: ℝ)
  (additional_investment: ℝ)
  (additional_rate: ℝ)
  (total_income_rate: ℝ)
  (total_income: ℝ)
  (h_total_investment_eq: initial_investment + additional_investment = 4800)
  (h_total_income_eq: 0.06 * (initial_investment + additional_investment) = total_income):
  (initial_investment * (r : ℝ) + additional_investment * additional_rate = total_income) →
  r = 0.04 := sorry

end barbata_interest_rate_l627_62709


namespace percentage_of_boys_currently_l627_62791

theorem percentage_of_boys_currently (B G : ℕ) (h1 : B + G = 50) (h2 : B + 50 = 95) : (B / 50) * 100 = 90 := by
  sorry

end percentage_of_boys_currently_l627_62791


namespace six_points_within_circle_l627_62743

/-- If six points are placed inside or on a circle with radius 1, then 
there always exist at least two points such that the distance between 
them is at most 1. -/
theorem six_points_within_circle : ∀ (points : Fin 6 → ℝ × ℝ), 
  (∀ i, (points i).1^2 + (points i).2^2 ≤ 1) → 
  ∃ i j, i ≠ j ∧ dist (points i) (points j) ≤ 1 :=
by
  -- Condition: Circle of radius 1
  intro points h_points
  sorry

end six_points_within_circle_l627_62743


namespace sum_powers_of_5_mod_8_l627_62762

theorem sum_powers_of_5_mod_8 :
  (List.sum (List.map (fun n => (5^n % 8)) (List.range 2011))) % 8 = 4 := 
  sorry

end sum_powers_of_5_mod_8_l627_62762


namespace molecular_weight_calc_l627_62727

namespace MolecularWeightProof

def atomic_weight_H : ℝ := 1.01
def atomic_weight_Br : ℝ := 79.90
def atomic_weight_O : ℝ := 16.00
def number_of_H : ℕ := 1
def number_of_Br : ℕ := 1
def number_of_O : ℕ := 3

theorem molecular_weight_calc :
  (number_of_H * atomic_weight_H + number_of_Br * atomic_weight_Br + number_of_O * atomic_weight_O) = 128.91 :=
by
  sorry

end MolecularWeightProof

end molecular_weight_calc_l627_62727


namespace length_AB_l627_62764

open Real

noncomputable def parabola_focus : (ℝ × ℝ) := (1, 0)

theorem length_AB (x1 y1 x2 y2 : ℝ) 
  (hA : y1^2 = 4 * x1) (hB : y2^2 = 4 * x2) 
  (hLine: (y2 - y1) * 1 = (x2 - x1) *0)
  (hSum : x1 + x2 = 6) : 
  dist (x1, y1) (x2, y2) = 8 := 
sorry

end length_AB_l627_62764


namespace exponent_sum_l627_62703

theorem exponent_sum : (-3)^3 + (-3)^2 + (-3)^1 + 3^1 + 3^2 + 3^3 = 18 := by
  sorry

end exponent_sum_l627_62703


namespace average_speed_for_trip_l627_62775

-- Define the total distance of the trip
def total_distance : ℕ := 850

--  Define the distance and speed for the first part of the trip
def distance1 : ℕ := 400
def speed1 : ℕ := 20

-- Define the distance and speed for the remaining part of the trip
def distance2 : ℕ := 450
def speed2 : ℕ := 15

-- Define the calculated average speed for the entire trip
def average_speed : ℕ := 17

theorem average_speed_for_trip 
  (d_total : ℕ)
  (d1 : ℕ) (s1 : ℕ)
  (d2 : ℕ) (s2 : ℕ)
  (hsum : d1 + d2 = d_total)
  (d1_eq : d1 = distance1)
  (s1_eq : s1 = speed1)
  (d2_eq : d2 = distance2)
  (s2_eq : s2 = speed2) :
  (d_total / ((d1 / s1) + (d2 / s2))) = average_speed := by
  sorry

end average_speed_for_trip_l627_62775


namespace choir_members_correct_l627_62799

def choir_members_condition (n : ℕ) : Prop :=
  150 < n ∧ n < 250 ∧ 
  n % 3 = 1 ∧ 
  n % 6 = 2 ∧ 
  n % 8 = 3

theorem choir_members_correct : ∃ n, choir_members_condition n ∧ (n = 195 ∨ n = 219) :=
by {
  sorry
}

end choir_members_correct_l627_62799


namespace cube_volume_ratio_l627_62783

theorem cube_volume_ratio (a b : ℝ) (h : (a^2 / b^2) = 9 / 25) :
  (b^3 / a^3) = 125 / 27 :=
by
  sorry

end cube_volume_ratio_l627_62783


namespace polygon_sides_l627_62739

theorem polygon_sides (n : ℕ) (h1 : (n - 2) * 180 - 180 = 2190) : n = 15 :=
sorry

end polygon_sides_l627_62739


namespace find_k_l627_62710

theorem find_k (x k : ℝ) :
  (∀ x, x ∈ Set.Ioo (-4 : ℝ) 3 ↔ x * (x^2 - 9) < k) → k = 0 :=
  by
  sorry

end find_k_l627_62710


namespace find_second_number_in_second_set_l627_62741

theorem find_second_number_in_second_set :
    (14 + 32 + 53) / 3 = 3 + (21 + x + 22) / 3 → x = 47 :=
by intro h
   sorry

end find_second_number_in_second_set_l627_62741


namespace expression_equals_one_l627_62779

variable {R : Type*} [Field R]
variables (x y z : R)

theorem expression_equals_one (h₁ : x ≠ y) (h₂ : x ≠ z) (h₃ : y ≠ z) :
    (x^2 / ((x - y) * (x - z)) + y^2 / ((y - x) * (y - z)) + z^2 / ((z - x) * (z - y))) = 1 :=
by sorry

end expression_equals_one_l627_62779


namespace algebraic_expression_value_l627_62756

theorem algebraic_expression_value (x y : ℝ) (h : x - 2 * y = -2) : 
  (2 * y - x) ^ 2 - 2 * x + 4 * y - 1 = 7 :=
by
  sorry

end algebraic_expression_value_l627_62756


namespace range_of_a_l627_62746

theorem range_of_a (a : ℝ) : 
  {x : ℝ | x^2 - 4 * x + 3 < 0} ⊆ {x : ℝ | 2^(1 - x) + a ≤ 0 ∧ x^2 - 2 * (a + 7) * x + 5 ≤ 0} → 
  -4 ≤ a ∧ a ≤ -1 := by
  sorry

end range_of_a_l627_62746


namespace george_borrow_amount_l627_62771

-- Define the conditions
def initial_fee_rate : ℝ := 0.05
def doubling_rate : ℝ := 2
def total_weeks : ℕ := 2
def total_fee : ℝ := 15

-- Define the problem statement
theorem george_borrow_amount : 
  ∃ (P : ℝ), (initial_fee_rate * P + initial_fee_rate * doubling_rate * P = total_fee) ∧ P = 100 :=
by
  -- Statement only, proof is skipped
  sorry

end george_borrow_amount_l627_62771


namespace greatest_number_that_divides_54_87_172_l627_62747

noncomputable def gcdThree (a b c : ℤ) : ℤ :=
  gcd (gcd a b) c

theorem greatest_number_that_divides_54_87_172
  (d r : ℤ)
  (h1 : 54 % d = r)
  (h2 : 87 % d = r)
  (h3 : 172 % d = r) :
  d = gcdThree 33 85 118 := by
  -- We would start the proof here, but it's omitted per instructions
  sorry

end greatest_number_that_divides_54_87_172_l627_62747


namespace option_A_two_solutions_l627_62788

theorem option_A_two_solutions :
    (∀ (a b : ℝ) (A : ℝ), 
    (a = 3 ∧ b = 4 ∧ A = 45) ∨ 
    (a = 7 ∧ b = 14 ∧ A = 30) ∨ 
    (a = 2 ∧ b = 7 ∧ A = 60) ∨ 
    (a = 8 ∧ b = 5 ∧ A = 135) →
    (∃ a b A : ℝ, a = 3 ∧ b = 4 ∧ A = 45 ∧ 2 = 2)) :=
by
  sorry

end option_A_two_solutions_l627_62788


namespace percentage_paid_to_A_l627_62757

theorem percentage_paid_to_A (A B : ℝ) (h1 : A + B = 550) (h2 : B = 220) : (A / B) * 100 = 150 := by
  -- Proof omitted
  sorry

end percentage_paid_to_A_l627_62757


namespace range_of_m_l627_62751

def A (x : ℝ) := x^2 - 3 * x - 10 ≤ 0
def B (x m : ℝ) := m + 1 ≤ x ∧ x ≤ 2 * m - 1

theorem range_of_m (m : ℝ) (h : ∀ x, B x m → A x) : m ≤ 3 := by
  sorry

end range_of_m_l627_62751


namespace find_H_over_G_l627_62789

variable (G H : ℤ)
variable (x : ℝ)

-- Conditions
def condition (G H : ℤ) (x : ℝ) : Prop :=
  x ≠ -7 ∧ x ≠ 0 ∧ x ≠ 6 ∧
  (↑G / (x + 7) + ↑H / (x * (x - 6)) = (x^2 - 3 * x + 15) / (x^3 + x^2 - 42 * x))

-- Theorem Statement
theorem find_H_over_G (G H : ℤ) (x : ℝ) (h : condition G H x) : (H : ℝ) / G = 15 / 7 :=
sorry

end find_H_over_G_l627_62789
