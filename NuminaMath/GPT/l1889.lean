import Mathlib

namespace NUMINAMATH_GPT_number_of_ways_to_put_7_balls_in_2_boxes_l1889_188923

theorem number_of_ways_to_put_7_balls_in_2_boxes :
  let distributions := [(7,0), (6,1), (5,2), (4,3)]
  let binom : (ℕ × ℕ) → ℕ := fun p => Nat.choose p.fst p.snd
  let counts := [1, binom (7,6), binom (7,5), binom (7,4)]
  counts.sum = 64 := by sorry

end NUMINAMATH_GPT_number_of_ways_to_put_7_balls_in_2_boxes_l1889_188923


namespace NUMINAMATH_GPT_reciprocals_of_product_one_l1889_188975

theorem reciprocals_of_product_one (x y : ℝ) (h : x * y = 1) : x = 1 / y ∧ y = 1 / x :=
by 
  sorry

end NUMINAMATH_GPT_reciprocals_of_product_one_l1889_188975


namespace NUMINAMATH_GPT_car_speed_first_hour_l1889_188907

theorem car_speed_first_hour 
  (x : ℝ)  -- Speed of the car in the first hour.
  (s2 : ℝ)  -- Speed of the car in the second hour is fixed at 40 km/h.
  (avg_speed : ℝ)  -- Average speed over two hours is 65 km/h.
  (h1 : s2 = 40)  -- speed in the second hour is 40 km/h.
  (h2 : avg_speed = 65)  -- average speed is 65 km/h
  (h3 : avg_speed = (x + s2) / 2)  -- definition of average speed
  : x = 90 := 
  sorry

end NUMINAMATH_GPT_car_speed_first_hour_l1889_188907


namespace NUMINAMATH_GPT_Angle_CNB_20_l1889_188996

theorem Angle_CNB_20 :
  ∀ (A B C N : Type) 
    (AC BC : Prop) 
    (angle_ACB : ℕ)
    (angle_NAC : ℕ)
    (angle_NCA : ℕ), 
    (AC ↔ BC) →
    angle_ACB = 98 →
    angle_NAC = 15 →
    angle_NCA = 21 →
    ∃ angle_CNB, angle_CNB = 20 :=
by
  sorry

end NUMINAMATH_GPT_Angle_CNB_20_l1889_188996


namespace NUMINAMATH_GPT_tan_add_pi_over_3_l1889_188972

theorem tan_add_pi_over_3 (x : ℝ) (h : Real.tan x = 3) : 
  Real.tan (x + Real.pi / 3) = - (6 + 5 * Real.sqrt 3) / 13 :=
sorry

end NUMINAMATH_GPT_tan_add_pi_over_3_l1889_188972


namespace NUMINAMATH_GPT_billy_used_54_tickets_l1889_188969

-- Definitions
def ferris_wheel_rides := 7
def bumper_car_rides := 3
def ferris_wheel_cost := 6
def bumper_car_cost := 4

-- Theorem Statement
theorem billy_used_54_tickets : 
  ferris_wheel_rides * ferris_wheel_cost + bumper_car_rides * bumper_car_cost = 54 := 
by
  sorry

end NUMINAMATH_GPT_billy_used_54_tickets_l1889_188969


namespace NUMINAMATH_GPT_largest_common_value_less_than_1000_l1889_188953

theorem largest_common_value_less_than_1000 :
  ∃ a : ℕ, a < 1000 ∧ (∃ n : ℤ, a = 4 + 5 * n) ∧ (∃ m : ℤ, a = 5 + 8 * m) ∧ 
            (∀ b : ℕ, (b < 1000 ∧ (∃ n : ℤ, b = 4 + 5 * n) ∧ (∃ m : ℤ, b = 5 + 8 * m)) → b ≤ a) :=
sorry

end NUMINAMATH_GPT_largest_common_value_less_than_1000_l1889_188953


namespace NUMINAMATH_GPT_find_edge_lengths_sum_l1889_188974

noncomputable def sum_edge_lengths (a d : ℝ) (volume surface_area : ℝ) : ℝ :=
  if (a - d) * a * (a + d) = volume ∧ 2 * ((a - d) * a + a * (a + d) + (a - d) * (a + d)) = surface_area then
    4 * ((a - d) + a + (a + d))
  else
    0

theorem find_edge_lengths_sum:
  (∃ a d : ℝ, (a - d) * a * (a + d) = 512 ∧ 2 * ((a - d) * a + a * (a + d) + (a - d) * (a + d)) = 352) →
  sum_edge_lengths (Real.sqrt 59) 1 512 352 = 12 * Real.sqrt 59 :=
by
  sorry

end NUMINAMATH_GPT_find_edge_lengths_sum_l1889_188974


namespace NUMINAMATH_GPT_tangent_length_from_A_to_circle_l1889_188949

noncomputable def point_A_polar : (ℝ × ℝ) := (6, Real.pi)
noncomputable def circle_eq_polar (θ : ℝ) : ℝ := -4 * Real.cos θ

theorem tangent_length_from_A_to_circle : 
  ∃ (length : ℝ), length = 2 * Real.sqrt 3 ∧ 
  (∃ (ρ θ : ℝ), point_A_polar = (6, Real.pi) ∧ ρ = circle_eq_polar θ) :=
sorry

end NUMINAMATH_GPT_tangent_length_from_A_to_circle_l1889_188949


namespace NUMINAMATH_GPT_geometric_sequence_a5_l1889_188985

theorem geometric_sequence_a5 (a : ℕ → ℝ) (r : ℝ) (h_geom : ∀ n, a (n+1) = a n * r)
  (h_eqn : ∃ x : ℝ, (x^2 + 7*x + 9 = 0) ∧ (a 3 = x) ∧ (a 7 = x)) :
  a 5 = 3 ∨ a 5 = -3 := 
sorry

end NUMINAMATH_GPT_geometric_sequence_a5_l1889_188985


namespace NUMINAMATH_GPT_alpha_tan_beta_gt_beta_tan_alpha_l1889_188999

theorem alpha_tan_beta_gt_beta_tan_alpha (α β : ℝ) (h1 : 0 < α) (h2 : α < β) (h3 : β < π / 2) 
: α * Real.tan β > β * Real.tan α := 
sorry

end NUMINAMATH_GPT_alpha_tan_beta_gt_beta_tan_alpha_l1889_188999


namespace NUMINAMATH_GPT_min_value_ge_8_min_value_8_at_20_l1889_188901

noncomputable def min_value (x : ℝ) (h : x > 4) : ℝ := (x + 12) / Real.sqrt (x - 4)

theorem min_value_ge_8 (x : ℝ) (h : x > 4) : min_value x h ≥ 8 := sorry

theorem min_value_8_at_20 : min_value 20 (by norm_num) = 8 := sorry

end NUMINAMATH_GPT_min_value_ge_8_min_value_8_at_20_l1889_188901


namespace NUMINAMATH_GPT_Yvonne_probability_of_success_l1889_188988

theorem Yvonne_probability_of_success
  (P_X : ℝ) (P_Z : ℝ) (P_XY_notZ : ℝ) :
  P_X = 1 / 3 →
  P_Z = 5 / 8 →
  P_XY_notZ = 0.0625 →
  ∃ P_Y : ℝ, P_Y = 0.5 :=
by
  intros hX hZ hXY_notZ
  existsi (0.5 : ℝ)
  sorry

end NUMINAMATH_GPT_Yvonne_probability_of_success_l1889_188988


namespace NUMINAMATH_GPT_find_x_squared_plus_y_squared_l1889_188970

variables (x y : ℝ)

theorem find_x_squared_plus_y_squared
  (h1 : x - y = 10)
  (h2 : x * y = 9) :
  x^2 + y^2 = 118 :=
sorry

end NUMINAMATH_GPT_find_x_squared_plus_y_squared_l1889_188970


namespace NUMINAMATH_GPT_common_ratio_is_2_l1889_188921

noncomputable def arithmetic_sequence_common_ratio (a : ℕ → ℝ) (q : ℝ) : Prop :=
∀ n, 2 * (a (n + 2) - a n) = 3 * a (n + 1)

theorem common_ratio_is_2 
  (a : ℕ → ℝ) 
  (q : ℝ) 
  (h1 : ∀ n, a (n + 1) = a n * q)
  (h2 : a 1 > 0)
  (h3 : arithmetic_sequence_common_ratio a q) :
  q = 2 :=
sorry

end NUMINAMATH_GPT_common_ratio_is_2_l1889_188921


namespace NUMINAMATH_GPT_inequality_holds_for_all_reals_l1889_188914

theorem inequality_holds_for_all_reals (x : ℝ) : 
  7 / 20 + |3 * x - 2 / 5| ≥ 1 / 4 :=
sorry

end NUMINAMATH_GPT_inequality_holds_for_all_reals_l1889_188914


namespace NUMINAMATH_GPT_average_time_for_relay_race_l1889_188905

noncomputable def average_leg_time (y_time z_time w_time x_time : ℕ) : ℚ :=
  (y_time + z_time + w_time + x_time) / 4

theorem average_time_for_relay_race :
  let y_time := 58
  let z_time := 26
  let w_time := 2 * z_time
  let x_time := 35
  average_leg_time y_time z_time w_time x_time = 42.75 := by
    sorry

end NUMINAMATH_GPT_average_time_for_relay_race_l1889_188905


namespace NUMINAMATH_GPT_obtuse_triangle_of_sin_cos_sum_l1889_188932

theorem obtuse_triangle_of_sin_cos_sum
  (A : ℝ) (hA : 0 < A ∧ A < π) 
  (h_eq : Real.sin A + Real.cos A = 12 / 25) :
  π / 2 < A ∧ A < π :=
sorry

end NUMINAMATH_GPT_obtuse_triangle_of_sin_cos_sum_l1889_188932


namespace NUMINAMATH_GPT_value_of_f_3x_minus_7_l1889_188927

def f (x : ℝ) : ℝ := 3 * x + 5

theorem value_of_f_3x_minus_7 (x : ℝ) : f (3 * x - 7) = 9 * x - 16 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_value_of_f_3x_minus_7_l1889_188927


namespace NUMINAMATH_GPT_percent_apple_juice_in_blend_l1889_188933

noncomputable def juice_blend_apple_percentage : ℚ :=
  let apple_juice_per_apple := 9 / 2
  let plum_juice_per_plum := 12 / 3
  let total_apple_juice := 4 * apple_juice_per_apple
  let total_plum_juice := 6 * plum_juice_per_plum
  let total_juice := total_apple_juice + total_plum_juice
  (total_apple_juice / total_juice) * 100

theorem percent_apple_juice_in_blend :
  juice_blend_apple_percentage = 43 :=
by
  sorry

end NUMINAMATH_GPT_percent_apple_juice_in_blend_l1889_188933


namespace NUMINAMATH_GPT_Evan_dog_weight_l1889_188967

-- Define the weights of the dogs as variables
variables (E I : ℕ)

-- Conditions given in the problem
def Evan_dog_weight_wrt_Ivan (I : ℕ) : ℕ := 7 * I
def dogs_total_weight (E I : ℕ) : Prop := E + I = 72

-- Correct answer we need to prove
theorem Evan_dog_weight (h1 : Evan_dog_weight_wrt_Ivan I = E)
                          (h2 : dogs_total_weight E I)
                          (h3 : I = 9) : E = 63 :=
by
  sorry

end NUMINAMATH_GPT_Evan_dog_weight_l1889_188967


namespace NUMINAMATH_GPT_proportion_estimation_chi_squared_test_l1889_188983

-- Definitions based on the conditions
def total_elders : ℕ := 500
def not_vaccinated_male : ℕ := 20
def not_vaccinated_female : ℕ := 10
def vaccinated_male : ℕ := 230
def vaccinated_female : ℕ := 240

-- Calculations based on the problem conditions
noncomputable def proportion_vaccinated : ℚ := (vaccinated_male + vaccinated_female) / total_elders

def chi_squared_statistic (a b c d n : ℕ) : ℚ :=
  n * (a * d - b * c) ^ 2 / ((a + b) * (c + d) * (a + c) * (b + d))

noncomputable def K2_value : ℚ :=
  chi_squared_statistic not_vaccinated_male not_vaccinated_female vaccinated_male vaccinated_female total_elders

-- Specify the critical value for 99% confidence
def critical_value_99 : ℚ := 6.635

-- Theorem statements (problems to prove)
theorem proportion_estimation : proportion_vaccinated = 94 / 100 := by
  sorry

theorem chi_squared_test : K2_value < critical_value_99 := by
  sorry

end NUMINAMATH_GPT_proportion_estimation_chi_squared_test_l1889_188983


namespace NUMINAMATH_GPT_marbles_total_l1889_188937

theorem marbles_total (yellow blue red total : ℕ)
  (hy : yellow = 5)
  (h_ratio : blue / red = 3 / 4)
  (h_red : red = yellow + 3)
  (h_total : total = yellow + blue + red) : total = 19 :=
by
  sorry

end NUMINAMATH_GPT_marbles_total_l1889_188937


namespace NUMINAMATH_GPT_triangle_angle_A_l1889_188956

theorem triangle_angle_A (C : ℝ) (c : ℝ) (a : ℝ) 
  (hC : C = 45) (hc : c = Real.sqrt 2) (ha : a = Real.sqrt 3) :
  (∃ A : ℝ, A = 60 ∨ A = 120) :=
by
  sorry

end NUMINAMATH_GPT_triangle_angle_A_l1889_188956


namespace NUMINAMATH_GPT_smallest_n_for_convex_100gon_l1889_188958

def isConvexPolygon (P : List (Real × Real)) : Prop := sorry -- Assumption for polygon convexity
def canBeIntersectedByTriangles (P : List (Real × Real)) (n : ℕ) : Prop := sorry -- Assumption for intersection by n triangles

theorem smallest_n_for_convex_100gon :
  ∀ (P : List (Real × Real)),
  isConvexPolygon P →
  List.length P = 100 →
  (∀ n, canBeIntersectedByTriangles P n → n ≥ 50) ∧ canBeIntersectedByTriangles P 50 :=
sorry

end NUMINAMATH_GPT_smallest_n_for_convex_100gon_l1889_188958


namespace NUMINAMATH_GPT_evaluate_expression_l1889_188987

theorem evaluate_expression : 
  -((5: ℤ) ^ 2) - (-(3: ℤ) ^ 3) * ((2: ℚ) / 9) - 9 * |((-(2: ℚ)) / 3)| = -25 := by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l1889_188987


namespace NUMINAMATH_GPT_average_cost_is_thirteen_l1889_188928

noncomputable def averageCostPerPen (pensCost shippingCost : ℝ) (totalPens : ℕ) : ℕ :=
  Nat.ceil ((pensCost + shippingCost) * 100 / totalPens)

theorem average_cost_is_thirteen :
  averageCostPerPen 29.85 8.10 300 = 13 :=
by
  sorry

end NUMINAMATH_GPT_average_cost_is_thirteen_l1889_188928


namespace NUMINAMATH_GPT_probability_after_6_passes_l1889_188960

noncomputable section

-- We define people
inductive Person
| A | B | C

-- Probability that person A has the ball after n passes
def P : ℕ → Person → ℚ
| 0, Person.A => 1
| 0, _ => 0
| n+1, Person.A => (P n Person.B + P n Person.C) / 2
| n+1, Person.B => (P n Person.A + P n Person.C) / 2
| n+1, Person.C => (P n Person.A + P n Person.B) / 2

theorem probability_after_6_passes :
  P 6 Person.A = 11 / 32 := by
  sorry

end NUMINAMATH_GPT_probability_after_6_passes_l1889_188960


namespace NUMINAMATH_GPT_a_eq_bn_l1889_188951

theorem a_eq_bn (a b n : ℕ) :
  (∀ k : ℕ, k ≠ b → ∃ m : ℕ, a - k^n = m * (b - k)) → a = b^n :=
by
  sorry

end NUMINAMATH_GPT_a_eq_bn_l1889_188951


namespace NUMINAMATH_GPT_negation_of_exists_sin_gt_one_equiv_forall_sin_le_one_l1889_188912

open Real

theorem negation_of_exists_sin_gt_one_equiv_forall_sin_le_one :
  (¬ (∃ x : ℝ, sin x > 1)) ↔ (∀ x : ℝ, sin x ≤ 1) :=
sorry

end NUMINAMATH_GPT_negation_of_exists_sin_gt_one_equiv_forall_sin_le_one_l1889_188912


namespace NUMINAMATH_GPT_exponent_form_l1889_188947

theorem exponent_form (x : ℕ) (k : ℕ) : (3^x) % 10 = 7 ↔ x = 4 * k + 3 :=
by
  sorry

end NUMINAMATH_GPT_exponent_form_l1889_188947


namespace NUMINAMATH_GPT_find_matrix_A_l1889_188936

-- Define the condition that A v = 3 v for all v in R^3
def satisfiesCondition (A : Matrix (Fin 3) (Fin 3) ℝ) : Prop :=
  ∀ (v : Fin 3 → ℝ), A.mulVec v = 3 • v

theorem find_matrix_A (A : Matrix (Fin 3) (Fin 3) ℝ) :
  satisfiesCondition A → A = 3 • 1 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_find_matrix_A_l1889_188936


namespace NUMINAMATH_GPT_equal_or_equal_exponents_l1889_188924

theorem equal_or_equal_exponents
  (a b c p q r : ℕ)
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_pos_p : 0 < p) (h_pos_q : 0 < q) (h_pos_r : 0 < r)
  (h1 : a^p + b^q + c^r = a^q + b^r + c^p)
  (h2 : a^q + b^r + c^p = a^r + b^p + c^q) :
  a = b ∧ b = c ∧ c = a ∨ p = q ∧ q = r ∧ r = p :=
  sorry

end NUMINAMATH_GPT_equal_or_equal_exponents_l1889_188924


namespace NUMINAMATH_GPT_pair_green_shirts_l1889_188978

/-- In a regional math gathering, 83 students wore red shirts, and 97 students wore green shirts. 
The 180 students are grouped into 90 pairs. Exactly 35 of these pairs consist of students both 
wearing red shirts. Prove that the number of pairs consisting solely of students wearing green shirts is 42. -/
theorem pair_green_shirts (r g total pairs rr: ℕ) (h_r : r = 83) (h_g : g = 97) (h_total : total = 180) 
    (h_pairs : pairs = 90) (h_rr : rr = 35) : 
    (g - (r - rr * 2)) / 2 = 42 := 
by 
  /- The proof is omitted. -/
  sorry

end NUMINAMATH_GPT_pair_green_shirts_l1889_188978


namespace NUMINAMATH_GPT_tangent_line_at_zero_l1889_188979

noncomputable def curve (x : ℝ) : ℝ := Real.exp (2 * x)

theorem tangent_line_at_zero :
  ∃ m b, (∀ x, (curve x) = m * x + b) ∧
    m = 2 ∧ b = 1 :=
by 
  sorry

end NUMINAMATH_GPT_tangent_line_at_zero_l1889_188979


namespace NUMINAMATH_GPT_sum_of_integers_l1889_188982

theorem sum_of_integers (x y : ℕ) (hxy_diff : x - y = 8) (hxy_prod : x * y = 240) (hx_gt_hy : x > y) : x + y = 32 := by
  sorry

end NUMINAMATH_GPT_sum_of_integers_l1889_188982


namespace NUMINAMATH_GPT_cosine_identity_l1889_188904

variable (α : ℝ)

theorem cosine_identity (h : Real.sin (Real.pi / 6 - α) = 1 / 3) : 
  Real.cos (2 * Real.pi / 3 + 2 * α) = -7 / 9 :=
by
  sorry

end NUMINAMATH_GPT_cosine_identity_l1889_188904


namespace NUMINAMATH_GPT_bus_full_people_could_not_take_l1889_188952

-- Definitions of the given conditions
def bus_capacity : ℕ := 80
def first_pickup_people : ℕ := (3 / 5) * bus_capacity
def people_exit_at_second_pickup : ℕ := 25
def people_waiting_at_second_pickup : ℕ := 90

-- The Lean statement to prove the number of people who could not take the bus
theorem bus_full_people_could_not_take (h1 : bus_capacity = 80)
                                       (h2 : first_pickup_people = 48)
                                       (h3 : people_exit_at_second_pickup = 25)
                                       (h4 : people_waiting_at_second_pickup = 90) :
  90 - (80 - (48 - 25)) = 33 :=
by
  sorry

end NUMINAMATH_GPT_bus_full_people_could_not_take_l1889_188952


namespace NUMINAMATH_GPT_distance_of_ladder_to_building_l1889_188917

theorem distance_of_ladder_to_building :
  ∀ (c a b : ℕ), c = 25 ∧ a = 20 ∧ (a^2 + b^2 = c^2) → b = 15 :=
by
  intros c a b h
  rcases h with ⟨hc, ha, hpyth⟩
  have h1 : c = 25 := hc
  have h2 : a = 20 := ha
  have h3 : a^2 + b^2 = c^2 := hpyth
  sorry

end NUMINAMATH_GPT_distance_of_ladder_to_building_l1889_188917


namespace NUMINAMATH_GPT_fraction_identity_l1889_188930

theorem fraction_identity (x y : ℚ) (h : x / y = 5 / 3) : y / (x - y) = 3 / 2 :=
by { sorry }

end NUMINAMATH_GPT_fraction_identity_l1889_188930


namespace NUMINAMATH_GPT_amy_spent_32_l1889_188938

theorem amy_spent_32 (x: ℝ) (h1: 0.15 * x + 1.6 * x + x = 55) : 1.6 * x = 32 :=
by
  sorry

end NUMINAMATH_GPT_amy_spent_32_l1889_188938


namespace NUMINAMATH_GPT_at_most_n_maximum_distance_pairs_l1889_188909

theorem at_most_n_maximum_distance_pairs (n : ℕ) (h : n > 2) 
(points : Fin n → ℝ × ℝ) :
  ∃ (maxDistPairs : Finset (Fin n × Fin n)), (maxDistPairs.card ≤ n) ∧ 
  ∀ (p1 p2 : Fin n), (p1, p2) ∈ maxDistPairs → 
  (∀ (q1 q2 : Fin n), dist (points q1) (points q2) ≤ dist (points p1) (points p2)) :=
sorry

end NUMINAMATH_GPT_at_most_n_maximum_distance_pairs_l1889_188909


namespace NUMINAMATH_GPT_sin_alpha_trig_expression_l1889_188998

theorem sin_alpha {α : ℝ} (hα : ∃ P : ℝ × ℝ, P = (4/5, -3/5)) :
  Real.sin α = -3/5 :=
sorry

theorem trig_expression {α : ℝ} 
  (hα : Real.sin α = -3/5) : 
  (Real.sin (π / 2 - α) / Real.sin (α + π)) - 
  (Real.tan (α - π) / Real.cos (3 * π - α)) = 19 / 48 :=
sorry

end NUMINAMATH_GPT_sin_alpha_trig_expression_l1889_188998


namespace NUMINAMATH_GPT_radius_of_third_circle_l1889_188939

open Real

theorem radius_of_third_circle (r : ℝ) :
  let r_large := 40
  let r_small := 25
  let area_large := π * r_large^2
  let area_small := π * r_small^2
  let region_area := area_large - area_small
  let half_region_area := region_area / 2
  let third_circle_area := π * r^2
  (third_circle_area = half_region_area) -> r = 15 * sqrt 13 :=
by
  sorry

end NUMINAMATH_GPT_radius_of_third_circle_l1889_188939


namespace NUMINAMATH_GPT_side_length_of_square_l1889_188911

variable (n : ℝ)

theorem side_length_of_square (h : n^2 = 9/16) : n = 3/4 :=
sorry

end NUMINAMATH_GPT_side_length_of_square_l1889_188911


namespace NUMINAMATH_GPT_evaluate_ceiling_neg_cubed_frac_l1889_188977

theorem evaluate_ceiling_neg_cubed_frac :
  (Int.ceil ((- (5 : ℚ) / 3) ^ 3 + 1) = -3) :=
sorry

end NUMINAMATH_GPT_evaluate_ceiling_neg_cubed_frac_l1889_188977


namespace NUMINAMATH_GPT_percent_defective_shipped_l1889_188989

-- Conditions given in the problem
def percent_defective (percent_total_defective: ℝ) : Prop := percent_total_defective = 0.08
def percent_shipped_defective (percent_defective_shipped: ℝ) : Prop := percent_defective_shipped = 0.04

-- The main theorem we want to prove
theorem percent_defective_shipped (percent_total_defective percent_defective_shipped : ℝ) 
  (h1 : percent_defective percent_total_defective) (h2 : percent_shipped_defective percent_defective_shipped) : 
  (percent_total_defective * percent_defective_shipped * 100) = 0.32 :=
by
  sorry

end NUMINAMATH_GPT_percent_defective_shipped_l1889_188989


namespace NUMINAMATH_GPT_alton_weekly_profit_l1889_188962

-- Definitions of the given conditions
def dailyEarnings : ℕ := 8
def daysInWeek : ℕ := 7
def weeklyRent : ℕ := 20

-- The proof problem: Prove that the total profit every week is $36
theorem alton_weekly_profit : (dailyEarnings * daysInWeek) - weeklyRent = 36 := by
  sorry

end NUMINAMATH_GPT_alton_weekly_profit_l1889_188962


namespace NUMINAMATH_GPT_solve_phi_eq_l1889_188955

noncomputable def φ := (1 + Real.sqrt 5) / 2
noncomputable def φ_hat := (1 - Real.sqrt 5) / 2
noncomputable def F : ℕ → ℤ
| n =>
  if n = 0 then 0
  else if n = 1 then 1
  else F (n - 1) + F (n - 2)

theorem solve_phi_eq (n : ℕ) :
  ∃ x y : ℤ, x * φ ^ (n + 1) + y * φ^n = 1 ∧ 
    x = (-1 : ℤ)^(n+1) * F n ∧ y = (-1 : ℤ)^n * F (n + 1) := by
  sorry

end NUMINAMATH_GPT_solve_phi_eq_l1889_188955


namespace NUMINAMATH_GPT_coffee_price_l1889_188926

theorem coffee_price (C : ℝ) :
  (7 * C) + (8 * 4) = 67 → C = 5 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_coffee_price_l1889_188926


namespace NUMINAMATH_GPT_daisies_multiple_of_4_l1889_188973

def num_roses := 8
def num_daisies (D : ℕ) := D
def num_marigolds := 48
def num_arrangements := 4

theorem daisies_multiple_of_4 (D : ℕ) 
  (h_roses_div_4 : num_roses % num_arrangements = 0)
  (h_marigolds_div_4 : num_marigolds % num_arrangements = 0)
  (h_total_div_4 : (num_roses + num_daisies D + num_marigolds) % num_arrangements = 0) :
  D % 4 = 0 :=
sorry

end NUMINAMATH_GPT_daisies_multiple_of_4_l1889_188973


namespace NUMINAMATH_GPT_frank_whack_a_mole_tickets_l1889_188980

variable (W : ℕ)
variable (skee_ball_tickets : ℕ := 9)
variable (candy_cost : ℕ := 6)
variable (candies_bought : ℕ := 7)
variable (total_tickets : ℕ := W + skee_ball_tickets)
variable (required_tickets : ℕ := candy_cost * candies_bought)

theorem frank_whack_a_mole_tickets : W + skee_ball_tickets = required_tickets → W = 33 := by
  sorry

end NUMINAMATH_GPT_frank_whack_a_mole_tickets_l1889_188980


namespace NUMINAMATH_GPT_number_of_distinct_configurations_l1889_188902

-- Definitions of the problem conditions
structure CubeConfig where
  white_cubes : Finset (Fin 8)
  blue_cubes : Finset (Fin 8)
  condition_1 : white_cubes.card = 5
  condition_2 : blue_cubes.card = 3
  condition_3 : ∀ x ∈ white_cubes, x ∉ blue_cubes

def distinctConfigCount (configs : Finset CubeConfig) : ℕ :=
  (configs.filter (λ config => 
    config.white_cubes.card = 5 ∧
    config.blue_cubes.card = 3 ∧
    (∀ x ∈ config.white_cubes, x ∉ config.blue_cubes)
  )).card

-- Theorem stating the correct number of distinct configurations
theorem number_of_distinct_configurations : distinctConfigCount ∅ = 5 := 
  sorry

end NUMINAMATH_GPT_number_of_distinct_configurations_l1889_188902


namespace NUMINAMATH_GPT_initial_deposit_l1889_188993

theorem initial_deposit :
  ∀ (P : ℝ) (r : ℝ) (n : ℕ) (t : ℝ),
    r = 0.05 → n = 1 → t = 2 → P * (1 + r / n) ^ (n * t) = 6615 → P = 6000 :=
by
  intros P r n t h_r h_n h_t h_eq
  rw [h_r, h_n, h_t] at h_eq
  norm_num at h_eq
  sorry

end NUMINAMATH_GPT_initial_deposit_l1889_188993


namespace NUMINAMATH_GPT_calculate_expression_l1889_188943

variable {a : ℝ}

theorem calculate_expression (h₁ : a ≠ 0) (h₂ : a ≠ 1) :
  (a - 1 / a) / ((a - 1) / a) = a + 1 := 
sorry

end NUMINAMATH_GPT_calculate_expression_l1889_188943


namespace NUMINAMATH_GPT_minimum_value_f_l1889_188908

noncomputable def f (x y : ℝ) : ℝ :=
  x^2 + y^2 + 16 / (x + y)^2

theorem minimum_value_f (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
  ∃ t ≥ 0, ∀ (x y : ℝ), 0 < x → 0 < y → f x y ≥ t ∧ t = 4 * Real.sqrt 2 :=
sorry

end NUMINAMATH_GPT_minimum_value_f_l1889_188908


namespace NUMINAMATH_GPT_white_water_addition_l1889_188954

theorem white_water_addition :
  ∃ (W H I T E A R : ℕ), 
  W ≠ H ∧ W ≠ I ∧ W ≠ T ∧ W ≠ E ∧ W ≠ A ∧ W ≠ R ∧
  H ≠ I ∧ H ≠ T ∧ H ≠ E ∧ H ≠ A ∧ H ≠ R ∧
  I ≠ T ∧ I ≠ E ∧ I ≠ A ∧ I ≠ R ∧
  T ≠ E ∧ T ≠ A ∧ T ≠ R ∧
  E ≠ A ∧ E ≠ R ∧
  A ≠ R ∧
  W = 8 ∧ I = 6 ∧ P = 1 ∧ C = 9 ∧ N = 0 ∧
  (10000 * W + 1000 * H + 100 * I + 10 * T + E) + 
  (10000 * W + 1000 * A + 100 * T + 10 * E + R) = 169069 :=
by 
  sorry

end NUMINAMATH_GPT_white_water_addition_l1889_188954


namespace NUMINAMATH_GPT_diagonal_crosses_700_cubes_l1889_188997

noncomputable def num_cubes_crossed (a b c : ℕ) : ℕ :=
  a + b + c - Nat.gcd a b - Nat.gcd b c - Nat.gcd c a + Nat.gcd (Nat.gcd a b) c

theorem diagonal_crosses_700_cubes :
  num_cubes_crossed 200 300 350 = 700 :=
sorry

end NUMINAMATH_GPT_diagonal_crosses_700_cubes_l1889_188997


namespace NUMINAMATH_GPT_find_m_positive_root_l1889_188903

theorem find_m_positive_root :
  (∃ x > 0, (x - 4) / (x - 3) - m - 4 = m / (3 - x)) → m = 1 :=
by
  sorry

end NUMINAMATH_GPT_find_m_positive_root_l1889_188903


namespace NUMINAMATH_GPT_arithmetic_sequence_properties_l1889_188957

noncomputable def arithmetic_sequence (n : ℕ) : ℕ :=
  4 * n - 3

noncomputable def sum_of_first_n_terms (n : ℕ) : ℕ :=
  2 * n^2 - n

noncomputable def sum_of_reciprocal_sequence (n : ℕ) : ℝ :=
  n / (4 * n + 1)

theorem arithmetic_sequence_properties :
  (arithmetic_sequence 3 = 9) →
  (arithmetic_sequence 8 = 29) →
  (∀ n, arithmetic_sequence n = 4 * n - 3) ∧
  (∀ n, sum_of_first_n_terms n = 2 * n^2 - n) ∧
  (∀ n, sum_of_reciprocal_sequence n = n / (4 * n + 1)) :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_properties_l1889_188957


namespace NUMINAMATH_GPT_dimes_difference_l1889_188965

theorem dimes_difference (a b c : ℕ) :
  a + b + c = 120 →
  5 * a + 10 * b + 25 * c = 1265 →
  c ≥ 10 →
  (max (b) - min (b)) = 92 :=
sorry

end NUMINAMATH_GPT_dimes_difference_l1889_188965


namespace NUMINAMATH_GPT_find_x_eq_728_l1889_188940

theorem find_x_eq_728 (n : ℕ) (x : ℕ) (hx : x = 9 ^ n - 1)
  (hprime_factors : ∃ (p q r : ℕ), (p ≠ q ∧ p ≠ r ∧ q ≠ r) ∧ (Nat.Prime p) ∧ (Nat.Prime q) ∧ (Nat.Prime r) ∧ (p * q * r) ∣ x)
  (h7 : 7 ∣ x) : x = 728 :=
sorry

end NUMINAMATH_GPT_find_x_eq_728_l1889_188940


namespace NUMINAMATH_GPT_geometric_sequence_a6_l1889_188976

theorem geometric_sequence_a6 (a : ℕ → ℝ) (geometric_seq : ∀ n, a (n + 1) = a n * a 1)
  (h1 : (a 4) * (a 8) = 9) (h2 : (a 4) + (a 8) = -11) : a 6 = -3 := by
  sorry

end NUMINAMATH_GPT_geometric_sequence_a6_l1889_188976


namespace NUMINAMATH_GPT_part_1_part_2_part_3_l1889_188915

def whiteHorseNumber (a b c : ℚ) : ℚ :=
  min (a - b) (min ((a - c) / 2) ((b - c) / 3))

theorem part_1 : 
  whiteHorseNumber (-2) (-4) 1 = -5/3 :=
by sorry

theorem part_2 : 
  max (whiteHorseNumber (-2) (-4) 1) (max (whiteHorseNumber (-2) 1 (-4)) 
  (max (whiteHorseNumber (-4) (-2) 1) (max (whiteHorseNumber (-4) 1 (-2)) 
  (max (whiteHorseNumber 1 (-4) (-2)) (whiteHorseNumber 1 (-2) (-4)) )))) = 2/3 :=
by sorry

theorem part_3 (x : ℚ) (h : ∃a b c : ℚ, a = -1 ∧ b = 6 ∧ c = x ∧ whiteHorseNumber a b c = 2) : 
  x = -7 ∨ x = 8 :=
by sorry

end NUMINAMATH_GPT_part_1_part_2_part_3_l1889_188915


namespace NUMINAMATH_GPT_reciprocal_of_x_l1889_188986

theorem reciprocal_of_x (x : ℝ) (h1 : x^3 - 2 * x^2 = 0) (h2 : x ≠ 0) : x = 2 → (1 / x = 1 / 2) :=
by {
  sorry
}

end NUMINAMATH_GPT_reciprocal_of_x_l1889_188986


namespace NUMINAMATH_GPT_conic_section_eccentricities_cubic_l1889_188931

theorem conic_section_eccentricities_cubic : 
  ∃ (e1 e2 e3 : ℝ), 
    (e1 = 1) ∧ 
    (0 < e2 ∧ e2 < 1) ∧ 
    (e3 > 1) ∧ 
    2 * e1^3 - 7 * e1^2 + 7 * e1 - 2 = 0 ∧
    2 * e2^3 - 7 * e2^2 + 7 * e2 - 2 = 0 ∧
    2 * e3^3 - 7 * e3^2 + 7 * e3 - 2 = 0 := 
by
  sorry

end NUMINAMATH_GPT_conic_section_eccentricities_cubic_l1889_188931


namespace NUMINAMATH_GPT_number_of_students_in_class_l1889_188922

theorem number_of_students_in_class
  (total_stickers : ℕ) (stickers_to_friends : ℕ) (stickers_left : ℝ) (students_each : ℕ → ℝ)
  (n_friends : ℕ) (remaining_stickers : ℝ) :
  total_stickers = 300 →
  stickers_to_friends = (n_friends * (n_friends + 1)) / 2 →
  stickers_left = 7.5 →
  ∀ n, n_friends = 10 →
  remaining_stickers = total_stickers - stickers_to_friends - (students_each n_friends) * (n - n_friends - 1) →
  (∃ n : ℕ, remaining_stickers = 7.5 ∧
              total_stickers - (stickers_to_friends + (students_each (n - n_friends - 1) * (n - n_friends - 1))) = 7.5) :=
by
  sorry

end NUMINAMATH_GPT_number_of_students_in_class_l1889_188922


namespace NUMINAMATH_GPT_solve_for_x_l1889_188942

theorem solve_for_x 
  (y : ℚ) (x : ℚ)
  (h : x / (x - 1) = (y^3 + 2 * y^2 - 2) / (y^3 + 2 * y^2 - 3)) :
  x = (y^3 + 2 * y^2 - 2) / 2 :=
sorry

end NUMINAMATH_GPT_solve_for_x_l1889_188942


namespace NUMINAMATH_GPT_min_value_of_f_l1889_188906

noncomputable def f (x : ℝ) := x + 2 * Real.cos x

theorem min_value_of_f :
  ∀ (x : ℝ), -Real.pi / 2 ≤ x ∧ x ≤ 0 → f x ≥ f (-Real.pi / 2) :=
by
  intro x hx
  -- conditions are given, statement declared, but proof is not provided
  sorry

end NUMINAMATH_GPT_min_value_of_f_l1889_188906


namespace NUMINAMATH_GPT_invertible_from_c_l1889_188925

-- Define the function f
def f (x : ℝ) : ℝ := (x - 3)^2 - 4

-- Define the condition for c and the statement to prove
theorem invertible_from_c (c : ℝ) (h : ∀ x1 x2 : ℝ, c ≤ x1 → c ≤ x2 → f x1 = f x2 → x1 = x2) : c = 3 :=
sorry

end NUMINAMATH_GPT_invertible_from_c_l1889_188925


namespace NUMINAMATH_GPT_part1_part2_l1889_188948

-- Definitions and conditions
variables {A B C a b c : ℝ}
variable (h1 : sin C * sin (A - B) = sin B * sin (C - A)) -- Given condition

-- Part (1): If A = 2B, then find C
theorem part1 (h2 : A = 2 * B) : C = (5 / 8) * π := by
  sorry

-- Part (2): Prove that 2a² = b² + c²
theorem part2 : 2 * a^2 = b^2 + c^2 := by
  sorry

end NUMINAMATH_GPT_part1_part2_l1889_188948


namespace NUMINAMATH_GPT_adam_more_apples_than_combined_l1889_188945

def adam_apples : Nat := 10
def jackie_apples : Nat := 2
def michael_apples : Nat := 5

theorem adam_more_apples_than_combined : 
  adam_apples - (jackie_apples + michael_apples) = 3 :=
by
  sorry

end NUMINAMATH_GPT_adam_more_apples_than_combined_l1889_188945


namespace NUMINAMATH_GPT_min_value_expr_l1889_188916

theorem min_value_expr (a : ℝ) (h₁ : 0 < a) (h₂ : a < 3) : 
  ∃ m : ℝ, (∀ x : ℝ, 0 < x → x < 3 → (1/x + 9/(3 - x)) ≥ m) ∧ m = 16 / 3 :=
sorry

end NUMINAMATH_GPT_min_value_expr_l1889_188916


namespace NUMINAMATH_GPT_time_spent_watching_movies_l1889_188966

def total_flight_time_minutes : ℕ := 11 * 60 + 20
def time_reading_minutes : ℕ := 2 * 60
def time_eating_dinner_minutes : ℕ := 30
def time_listening_radio_minutes : ℕ := 40
def time_playing_games_minutes : ℕ := 1 * 60 + 10
def time_nap_minutes : ℕ := 3 * 60

theorem time_spent_watching_movies :
  total_flight_time_minutes
  - time_reading_minutes
  - time_eating_dinner_minutes
  - time_listening_radio_minutes
  - time_playing_games_minutes
  - time_nap_minutes = 4 * 60 := by
  sorry

end NUMINAMATH_GPT_time_spent_watching_movies_l1889_188966


namespace NUMINAMATH_GPT_complex_numbers_count_l1889_188935

theorem complex_numbers_count (z : ℂ) (h1 : z^24 = 1) (h2 : ∃ r : ℝ, z^6 = r) : ℕ :=
  sorry -- Proof goes here

end NUMINAMATH_GPT_complex_numbers_count_l1889_188935


namespace NUMINAMATH_GPT_ten_percent_of_x_l1889_188990

variable (certain_value : ℝ)
variable (x : ℝ)

theorem ten_percent_of_x (h : 3 - (1 / 4) * 2 - (1 / 3) * 3 - (1 / 7) * x = certain_value) :
  0.1 * x = 0.7 * (1.5 - certain_value) := sorry

end NUMINAMATH_GPT_ten_percent_of_x_l1889_188990


namespace NUMINAMATH_GPT_total_number_of_items_l1889_188992

-- Define the conditions as equations in Lean
def model_cars_price := 5
def model_trains_price := 8
def total_amount := 31

-- Initialize the variable definitions for number of cars and trains
variables (c t : ℕ)

-- The proof problem: Show that given the equation, the sum of cars and trains is 5
theorem total_number_of_items : (model_cars_price * c + model_trains_price * t = total_amount) → (c + t = 5) := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_GPT_total_number_of_items_l1889_188992


namespace NUMINAMATH_GPT_face_value_stock_l1889_188981

-- Given conditions
variables (F : ℝ) (yield quoted_price dividend_rate : ℝ)
variables (h_yield : yield = 20) (h_quoted_price : quoted_price = 125)
variables (h_dividend_rate : dividend_rate = 0.25)

--Theorem to prove the face value of the stock is 100
theorem face_value_stock : (dividend_rate * F / quoted_price) * 100 = yield ↔ F = 100 :=
by
  sorry

end NUMINAMATH_GPT_face_value_stock_l1889_188981


namespace NUMINAMATH_GPT_painted_by_all_three_l1889_188929

/-
Statement: Given that 75% of the floor is painted red, 70% painted green, and 65% painted blue,
prove that at least 10% of the floor is painted with all three colors.
-/

def painted_by_red (floor : ℝ) : ℝ := 0.75 * floor
def painted_by_green (floor : ℝ) : ℝ := 0.70 * floor
def painted_by_blue (floor : ℝ) : ℝ := 0.65 * floor

theorem painted_by_all_three (floor : ℝ) :
  ∃ (x : ℝ), x = 0.10 * floor ∧
  (painted_by_red floor) + (painted_by_green floor) + (painted_by_blue floor) ≥ 2 * floor :=
sorry

end NUMINAMATH_GPT_painted_by_all_three_l1889_188929


namespace NUMINAMATH_GPT_difference_between_first_and_third_l1889_188994

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

end NUMINAMATH_GPT_difference_between_first_and_third_l1889_188994


namespace NUMINAMATH_GPT_expected_participants_in_2005_l1889_188919

open Nat

def initial_participants : ℕ := 500
def annual_increase_rate : ℚ := 1.2
def num_years : ℕ := 5
def expected_participants_2005 : ℚ := 1244

theorem expected_participants_in_2005 :
  (initial_participants : ℚ) * annual_increase_rate ^ num_years = expected_participants_2005 := by
  sorry

end NUMINAMATH_GPT_expected_participants_in_2005_l1889_188919


namespace NUMINAMATH_GPT_stratified_sampling_2nd_year_students_l1889_188941

theorem stratified_sampling_2nd_year_students
  (students_1st_year : ℕ) (students_2nd_year : ℕ) (students_3rd_year : ℕ) (total_sample_size : ℕ) :
  students_1st_year = 1000 ∧ students_2nd_year = 800 ∧ students_3rd_year = 700 ∧ total_sample_size = 100 →
  (students_2nd_year * total_sample_size / (students_1st_year + students_2nd_year + students_3rd_year) = 32) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_stratified_sampling_2nd_year_students_l1889_188941


namespace NUMINAMATH_GPT_sum_of_edges_l1889_188944

theorem sum_of_edges (a b c : ℝ)
  (h1 : a * b * c = 8)
  (h2 : 2 * (a * b + b * c + c * a) = 32)
  (h3 : b ^ 2 = a * c) :
  4 * (a + b + c) = 32 := 
sorry

end NUMINAMATH_GPT_sum_of_edges_l1889_188944


namespace NUMINAMATH_GPT_find_a_squared_plus_b_squared_l1889_188918

theorem find_a_squared_plus_b_squared 
  (a b : ℝ) 
  (h1 : a + b = 40) 
  (h2 : a * b = 104) : 
  a^2 + b^2 = 1392 := 
by 
  sorry

end NUMINAMATH_GPT_find_a_squared_plus_b_squared_l1889_188918


namespace NUMINAMATH_GPT_radar_placement_problem_l1889_188959

noncomputable def max_distance (n : ℕ) (coverage_radius : ℝ) (central_angle : ℝ) : ℝ :=
  coverage_radius / Real.sin (central_angle / 2)

noncomputable def ring_area (inner_radius : ℝ) (outer_radius : ℝ) : ℝ :=
  Real.pi * (outer_radius ^ 2 - inner_radius ^ 2)

theorem radar_placement_problem (r : ℝ := 13) (n : ℕ := 5) (width : ℝ := 10) :
  let angle := 2 * Real.pi / n
  let max_dist := max_distance n r angle
  let inner_radius := (r ^ 2 - (r - width) ^ 2) / Real.tan (angle / 2)
  let outer_radius := inner_radius + width
  max_dist = 12 / Real.sin (angle / 2) ∧
  ring_area inner_radius outer_radius = 240 * Real.pi / Real.tan (angle / 2) :=
by
  sorry

end NUMINAMATH_GPT_radar_placement_problem_l1889_188959


namespace NUMINAMATH_GPT_simplify_expression_l1889_188968

-- Define the expressions and the simplification statement
def expr1 (x : ℝ) := (3 * x - 6) * (x + 8)
def expr2 (x : ℝ) := (x + 6) * (3 * x - 2)
def simplified (x : ℝ) := 2 * x - 36

theorem simplify_expression (x : ℝ) : expr1 x - expr2 x = simplified x := by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1889_188968


namespace NUMINAMATH_GPT_find_number_l1889_188910

theorem find_number (x : ℝ) (h : (((x + 1.4) / 3 - 0.7) * 9 = 5.4)) : x = 2.5 :=
by 
  sorry

end NUMINAMATH_GPT_find_number_l1889_188910


namespace NUMINAMATH_GPT_number_of_terms_in_arithmetic_sequence_l1889_188946

theorem number_of_terms_in_arithmetic_sequence 
  (a d n l : ℤ) (h1 : a = 7) (h2 : d = 2) (h3 : l = 145) 
  (h4 : l = a + (n - 1) * d) : n = 70 := 
by sorry

end NUMINAMATH_GPT_number_of_terms_in_arithmetic_sequence_l1889_188946


namespace NUMINAMATH_GPT_tournament_participants_l1889_188971

theorem tournament_participants (n : ℕ) (h₁ : 2 * (n * (n - 1) / 2 + 4) - (n - 2) * (n - 3) - 16 = 124) : n = 13 :=
sorry

end NUMINAMATH_GPT_tournament_participants_l1889_188971


namespace NUMINAMATH_GPT_b_came_third_four_times_l1889_188995

variable (a b c N : ℕ)

theorem b_came_third_four_times
    (a_pos : a > 0) 
    (b_pos : b > 0) 
    (c_pos : c > 0)
    (a_gt_b : a > b) 
    (b_gt_c : b > c) 
    (a_b_c_sum : a + b + c = 8)
    (score_A : 4 * a + b = 26) 
    (score_B : a + 4 * c = 11) 
    (score_C : 3 * b + 2 * c = 11) 
    (B_won_first_event : a + b + c = 8) : 
    4 * c = 4 := 
sorry

end NUMINAMATH_GPT_b_came_third_four_times_l1889_188995


namespace NUMINAMATH_GPT_quadratic_root_neg3_l1889_188934

theorem quadratic_root_neg3 : ∃ x : ℝ, x^2 - 9 = 0 ∧ (x = -3) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_root_neg3_l1889_188934


namespace NUMINAMATH_GPT_range_of_a_l1889_188963

-- Definitions of position conditions in the 4th quadrant
def PosInFourthQuad (x y : ℝ) : Prop := (x > 0) ∧ (y < 0)

-- Statement to prove
theorem range_of_a (a : ℝ) (h : PosInFourthQuad (2 * a + 4) (3 * a - 6)) : -2 < a ∧ a < 2 :=
  sorry

end NUMINAMATH_GPT_range_of_a_l1889_188963


namespace NUMINAMATH_GPT_golden_ratio_eqn_value_of_ab_value_of_pq_n_l1889_188961

-- Part (1): Finding the golden ratio
theorem golden_ratio_eqn {x : ℝ} (h1 : x^2 + x - 1 = 0) : x = (-1 + Real.sqrt 5) / 2 ∨ x = (-1 - Real.sqrt 5) / 2 := 
sorry

-- Part (2): Finding the value of ab
theorem value_of_ab {a b m : ℝ} (h1 : a^2 + m * a = 1) (h2 : b^2 - 2 * m * b = 4) (h3 : b ≠ -2 * a) : a * b = 2 :=
sorry

-- Part (3): Finding the value of pq - n
theorem value_of_pq_n {p q n : ℝ} (h1 : p ≠ q) (eq1 : p^2 + n * p - 1 = q) (eq2 : q^2 + n * q - 1 = p) : p * q - n = 0 :=
sorry

end NUMINAMATH_GPT_golden_ratio_eqn_value_of_ab_value_of_pq_n_l1889_188961


namespace NUMINAMATH_GPT_min_value_of_f_in_interval_l1889_188920

def f (x k : ℝ) : ℝ := x^2 - k * x - 1

theorem min_value_of_f_in_interval (k : ℝ) :
  (f 1 k = -k ∧ k ≤ 2) ∨ 
  (∃ k', k' = 2 ∧ f (k'/2) k = - (k'^2) / 4 - 1 ∧ 2 < k ∧ k < 8) ∨ 
  (f 4 k = 15 - 4 * k ∧ k ≥ 8) :=
by sorry

end NUMINAMATH_GPT_min_value_of_f_in_interval_l1889_188920


namespace NUMINAMATH_GPT_Jorge_Giuliana_cakes_l1889_188900

theorem Jorge_Giuliana_cakes (C : ℕ) :
  (2 * 7 + 2 * C + 2 * 30 = 110) → (C = 18) :=
by
  sorry

end NUMINAMATH_GPT_Jorge_Giuliana_cakes_l1889_188900


namespace NUMINAMATH_GPT_pencil_partition_l1889_188991

theorem pencil_partition (total_length green_fraction green_length remaining_length white_fraction half_remaining white_length gold_length : ℝ)
  (h1 : green_fraction = 7 / 10)
  (h2 : total_length = 2)
  (h3 : green_length = green_fraction * total_length)
  (h4 : remaining_length = total_length - green_length)
  (h5 : white_fraction = 1 / 2)
  (h6 : white_length = white_fraction * remaining_length)
  (h7 : gold_length = remaining_length - white_length) :
  (gold_length / remaining_length) = 1 / 2 :=
sorry

end NUMINAMATH_GPT_pencil_partition_l1889_188991


namespace NUMINAMATH_GPT_motorist_spent_on_petrol_l1889_188913

def original_price_per_gallon : ℝ := 5.56
def reduction_percentage : ℝ := 0.10
def new_price_per_gallon := original_price_per_gallon - (0.10 * original_price_per_gallon)
def gallons_more_after_reduction : ℝ := 5

theorem motorist_spent_on_petrol (X : ℝ) 
  (h1 : new_price_per_gallon = original_price_per_gallon - (reduction_percentage * original_price_per_gallon))
  (h2 : (X / new_price_per_gallon) - (X / original_price_per_gallon) = gallons_more_after_reduction) :
  X = 250.22 :=
by
  sorry

end NUMINAMATH_GPT_motorist_spent_on_petrol_l1889_188913


namespace NUMINAMATH_GPT_factor_expression_l1889_188984

variable (x : ℝ)

def e : ℝ := (16 * x^6 + 36 * x^4 - 9) - (4 * x^6 - 6 * x^4 + 5)

theorem factor_expression : e x = 2 * (6 * x^6 + 21 * x^4 - 7) :=
by
  sorry

end NUMINAMATH_GPT_factor_expression_l1889_188984


namespace NUMINAMATH_GPT_max_abs_value_l1889_188964

theorem max_abs_value (x y : ℝ) (hx : |x - 1| ≤ 2) (hy : |y - 1| ≤ 2) : |x - 2 * y + 1| ≤ 6 :=
sorry

end NUMINAMATH_GPT_max_abs_value_l1889_188964


namespace NUMINAMATH_GPT_smallest_n_multiple_of_7_l1889_188950

theorem smallest_n_multiple_of_7 (x y n : ℤ) (h1 : x + 2 ≡ 0 [ZMOD 7]) (h2 : y - 2 ≡ 0 [ZMOD 7]) :
  x^2 + x * y + y^2 + n ≡ 0 [ZMOD 7] → n = 3 :=
by
  sorry

end NUMINAMATH_GPT_smallest_n_multiple_of_7_l1889_188950
