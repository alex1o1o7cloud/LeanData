import Mathlib

namespace NUMINAMATH_GPT_total_cubes_in_stack_l1522_152275

theorem total_cubes_in_stack :
  let bottom_layer := 4
  let middle_layer := 2
  let top_layer := 1
  bottom_layer + middle_layer + top_layer = 7 :=
by
  sorry

end NUMINAMATH_GPT_total_cubes_in_stack_l1522_152275


namespace NUMINAMATH_GPT_books_bought_l1522_152289

noncomputable def totalCost : ℤ :=
  let numFilms := 9
  let costFilm := 5
  let numCDs := 6
  let costCD := 3
  let costBook := 4
  let totalSpent := 79
  totalSpent - (numFilms * costFilm + numCDs * costCD)

theorem books_bought : ∃ B : ℤ, B * 4 = totalCost := by
  sorry

end NUMINAMATH_GPT_books_bought_l1522_152289


namespace NUMINAMATH_GPT_find_triples_tan_l1522_152231

open Real

theorem find_triples_tan (x y z : ℝ) :
  x * (1 - y^2) * (1 - z^2) + y * (1 - z^2) * (1 - x^2) + z * (1 - x^2) * (1 - y^2) = 4 * x * y * z → 
  ∃ (A B C : ℝ), x = tan A ∧ y = tan B ∧ z = tan C :=
by
  sorry

end NUMINAMATH_GPT_find_triples_tan_l1522_152231


namespace NUMINAMATH_GPT_michael_num_dogs_l1522_152242

variable (total_cost : ℕ)
variable (cost_per_animal : ℕ)
variable (num_cats : ℕ)
variable (num_dogs : ℕ)

-- Conditions
def michael_total_cost := total_cost = 65
def michael_num_cats := num_cats = 2
def michael_cost_per_animal := cost_per_animal = 13

-- Theorem to prove
theorem michael_num_dogs (h_total_cost : michael_total_cost total_cost)
                         (h_num_cats : michael_num_cats num_cats)
                         (h_cost_per_animal : michael_cost_per_animal cost_per_animal) :
  num_dogs = 3 :=
by
  sorry

end NUMINAMATH_GPT_michael_num_dogs_l1522_152242


namespace NUMINAMATH_GPT_determine_jug_capacity_l1522_152237

variable (jug_capacity : Nat)
variable (small_jug : Nat)

theorem determine_jug_capacity (h1 : jug_capacity = 5) (h2 : small_jug = 3 ∨ small_jug = 4):
  (∃ overflow_remains : Nat, 
    (overflow_remains = jug_capacity ∧ small_jug = 4) ∨ 
    (¬(overflow_remains = jug_capacity) ∧ small_jug = 3)) :=
by
  sorry

end NUMINAMATH_GPT_determine_jug_capacity_l1522_152237


namespace NUMINAMATH_GPT_minimum_value_x_plus_2y_l1522_152235

theorem minimum_value_x_plus_2y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + 2 * y = x * y) : x + 2 * y ≥ 8 :=
sorry

end NUMINAMATH_GPT_minimum_value_x_plus_2y_l1522_152235


namespace NUMINAMATH_GPT_sum_of_possible_values_CDF_l1522_152250

theorem sum_of_possible_values_CDF 
  (C D F : ℕ) 
  (hC: 0 ≤ C ∧ C ≤ 9)
  (hD: 0 ≤ D ∧ D ≤ 9)
  (hF: 0 ≤ F ∧ F ≤ 9)
  (hdiv: (C + 4 + 9 + 8 + D + F + 4) % 9 = 0) :
  C + D + F = 2 ∨ C + D + F = 11 → (2 + 11 = 13) :=
by sorry

end NUMINAMATH_GPT_sum_of_possible_values_CDF_l1522_152250


namespace NUMINAMATH_GPT_cad_to_jpy_l1522_152251

theorem cad_to_jpy (h : 2000 / 18 =  y / 5) : y = 556 := 
by 
  sorry

end NUMINAMATH_GPT_cad_to_jpy_l1522_152251


namespace NUMINAMATH_GPT_distance_traveled_l1522_152278

-- Definition of the velocity function
def velocity (t : ℝ) : ℝ := 2 * t - 3

-- Prove the integral statement
theorem distance_traveled : 
  (∫ t in (0 : ℝ)..(5 : ℝ), abs (velocity t)) = 29 / 2 := by 
{ sorry }

end NUMINAMATH_GPT_distance_traveled_l1522_152278


namespace NUMINAMATH_GPT_abs_diff_of_two_numbers_l1522_152227

theorem abs_diff_of_two_numbers (x y : ℝ) (h1 : x + y = 30) (h2 : x * y = 200) : |x - y| = 10 :=
by
  sorry

end NUMINAMATH_GPT_abs_diff_of_two_numbers_l1522_152227


namespace NUMINAMATH_GPT_problem_A_problem_B_problem_C_problem_D_l1522_152223

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / (Real.exp x)

theorem problem_A : ∀ x: ℝ, 0 < x ∧ x < 1 → f x < 0 := 
by sorry

theorem problem_B : ∃! (x : ℝ), ∃ c : ℝ, deriv f x = 0 := 
by sorry

theorem problem_C : ∀ (x : ℝ), ∃ c : ℝ, deriv f x = 0 → ¬∃ d : ℝ, d ≠ c ∧ deriv f d = 0 := 
by sorry

theorem problem_D : ¬ ∃ x₀ : ℝ, f x₀ = 1 / Real.exp 1 := 
by sorry

end NUMINAMATH_GPT_problem_A_problem_B_problem_C_problem_D_l1522_152223


namespace NUMINAMATH_GPT_negation_of_proposition_l1522_152286

variable (x : ℝ)
variable (p : Prop)

def proposition : Prop := ∀ x > 0, (x + 1) * Real.exp x > 1

theorem negation_of_proposition : ¬ proposition ↔ ∃ x > 0, (x + 1) * Real.exp x ≤ 1 :=
by
  sorry

end NUMINAMATH_GPT_negation_of_proposition_l1522_152286


namespace NUMINAMATH_GPT_remainder_19008_div_31_l1522_152210

theorem remainder_19008_div_31 :
  ∀ (n : ℕ), (n = 432 * 44) → n % 31 = 5 :=
by
  intro n h
  sorry

end NUMINAMATH_GPT_remainder_19008_div_31_l1522_152210


namespace NUMINAMATH_GPT_median_on_AB_eq_altitude_on_BC_eq_perp_bisector_on_AC_eq_l1522_152202

-- Definition of points A, B, and C
def A : ℝ × ℝ := (4, 0)
def B : ℝ × ℝ := (6, 7)
def C : ℝ × ℝ := (0, 3)

-- The problem statements as Lean theorems
theorem median_on_AB_eq : ∀ (A B : ℝ × ℝ), A = (4, 0) ∧ B = (6, 7) → ∃ (x y : ℝ), x - 10 * y + 30 = 0 := by
  intros
  sorry

theorem altitude_on_BC_eq : ∀ (B C : ℝ × ℝ), B = (6, 7) ∧ C = (0, 3) → ∃ (x y : ℝ), 3 * x + 2 * y - 12 = 0 := by
  intros
  sorry

theorem perp_bisector_on_AC_eq : ∀ (A C : ℝ × ℝ), A = (4, 0) ∧ C = (0, 3) → ∃ (x y : ℝ), 8 * x - 6 * y - 7 = 0 := by
  intros
  sorry

end NUMINAMATH_GPT_median_on_AB_eq_altitude_on_BC_eq_perp_bisector_on_AC_eq_l1522_152202


namespace NUMINAMATH_GPT_sam_has_two_nickels_l1522_152259

def average_value_initial (total_value : ℕ) (total_coins : ℕ) := total_value / total_coins = 15
def average_value_with_extra_dime (total_value : ℕ) (total_coins : ℕ) := (total_value + 10) / (total_coins + 1) = 16

theorem sam_has_two_nickels (total_value total_coins : ℕ) (h1 : average_value_initial total_value total_coins) (h2 : average_value_with_extra_dime total_value total_coins) : 
∃ (nickels : ℕ), nickels = 2 := 
by 
  sorry

end NUMINAMATH_GPT_sam_has_two_nickels_l1522_152259


namespace NUMINAMATH_GPT_speed_with_stream_l1522_152280

variable (V_m V_s : ℝ)

def against_speed : Prop := V_m - V_s = 13
def still_water_rate : Prop := V_m = 6

theorem speed_with_stream (h1 : against_speed V_m V_s) (h2 : still_water_rate V_m) : V_m + V_s = 13 := 
sorry

end NUMINAMATH_GPT_speed_with_stream_l1522_152280


namespace NUMINAMATH_GPT_two_digit_number_solution_l1522_152220

theorem two_digit_number_solution (N : ℕ) (x y : ℕ) :
  (10 * x + y = N) ∧ (4 * x + 2 * y = (10 * x + y) / 2) →
  N = 32 ∨ N = 64 ∨ N = 96 := 
sorry

end NUMINAMATH_GPT_two_digit_number_solution_l1522_152220


namespace NUMINAMATH_GPT_unique_roots_of_system_l1522_152258

theorem unique_roots_of_system {x y z : ℂ} 
  (h1 : x + y + z = 3) 
  (h2 : x^2 + y^2 + z^2 = 3) 
  (h3 : x^3 + y^3 + z^3 = 3) : 
  (x = 1 ∧ y = 1 ∧ z = 1) :=
sorry

end NUMINAMATH_GPT_unique_roots_of_system_l1522_152258


namespace NUMINAMATH_GPT_incorrect_statement_B_l1522_152253

open Set

-- Define the relevant events as described in the problem
def event_subscribe_at_least_one (ω : Type) (A B : Set ω) : Set ω := A ∪ B
def event_subscribe_at_most_one (ω : Type) (A B : Set ω) : Set ω := (A ∩ B)ᶜ

-- Define the problem statement
theorem incorrect_statement_B (ω : Type) (A B : Set ω) :
  ¬ (event_subscribe_at_least_one ω A B) = (event_subscribe_at_most_one ω A B)ᶜ :=
sorry

end NUMINAMATH_GPT_incorrect_statement_B_l1522_152253


namespace NUMINAMATH_GPT_express_a_b_find_a_b_m_n_find_a_l1522_152206

-- 1. Prove that a = m^2 + 5n^2 and b = 2mn given a + b√5 = (m + n√5)^2
theorem express_a_b (a b m n : ℤ) (h : a + b * Real.sqrt 5 = (m + n * Real.sqrt 5) ^ 2) :
  a = m ^ 2 + 5 * n ^ 2 ∧ b = 2 * m * n := sorry

-- 2. Prove there exists positive integers a = 6, b = 2, m = 1, and n = 1 such that 
-- a + b√5 = (m + n√5)^2.
theorem find_a_b_m_n : ∃ (a b m n : ℕ), a = 6 ∧ b = 2 ∧ m = 1 ∧ n = 1 ∧ 
  (a + b * Real.sqrt 5 = (m + n * Real.sqrt 5) ^ 2) := sorry

-- 3. Prove a = 46 or a = 14 given a + 6√5 = (m + n√5)^2 and a, m, n are positive integers.
theorem find_a (a m n : ℕ) (h : a + 6 * Real.sqrt 5 = (m + n * Real.sqrt 5) ^ 2) :
  a = 46 ∨ a = 14 := sorry

end NUMINAMATH_GPT_express_a_b_find_a_b_m_n_find_a_l1522_152206


namespace NUMINAMATH_GPT_girls_in_blue_dresses_answered_affirmatively_l1522_152217

theorem girls_in_blue_dresses_answered_affirmatively :
  ∃ (n : ℕ), n = 17 ∧
  ∀ (total_girls red_dresses blue_dresses answer_girls : ℕ),
  total_girls = 30 →
  red_dresses = 13 →
  blue_dresses = 17 →
  answer_girls = n →
  answer_girls = blue_dresses :=
sorry

end NUMINAMATH_GPT_girls_in_blue_dresses_answered_affirmatively_l1522_152217


namespace NUMINAMATH_GPT_gcd_of_lcm_and_ratio_l1522_152239

theorem gcd_of_lcm_and_ratio (X Y : ℕ) (h1 : Nat.lcm X Y = 180) (h2 : X * 5 = Y * 2) : Nat.gcd X Y = 18 :=
sorry

end NUMINAMATH_GPT_gcd_of_lcm_and_ratio_l1522_152239


namespace NUMINAMATH_GPT_trajectory_eq_ellipse_range_sum_inv_dist_l1522_152219

-- Conditions for circle M
def CircleM := { center : ℝ × ℝ // center = (-3, 0) }
def radiusM := 1

-- Conditions for circle N
def CircleN := { center : ℝ × ℝ // center = (3, 0) }
def radiusN := 9

-- Conditions for circle P
def CircleP (x y : ℝ) (r : ℝ) := 
  (dist (x, y) (-3, 0) = r + radiusM) ∧
  (dist (x, y) (3, 0) = radiusN - r)

-- Proof for the equation of the trajectory
theorem trajectory_eq_ellipse :
  ∃ (x y : ℝ), CircleP x y r → x^2 / 25 + y^2 / 16 = 1 :=
sorry

-- Proof for the range of 1/PM + 1/PN
theorem range_sum_inv_dist :
  ∃ (r PM PN : ℝ), 
    PM ∈ [2, 8] ∧ 
    PN = 10 - PM ∧ 
    CircleP (PM - radiusM) (PN - radiusN) r → 
    (2/5 ≤ (1/PM + 1/PN) ∧ (1/PM + 1/PN) ≤ 5/8) :=
sorry

end NUMINAMATH_GPT_trajectory_eq_ellipse_range_sum_inv_dist_l1522_152219


namespace NUMINAMATH_GPT_cats_in_shelter_l1522_152241

theorem cats_in_shelter (C D: ℕ) (h1 : 15 * D = 7 * C) 
                        (h2 : 15 * (D + 12) = 11 * C) :
    C = 45 := by
  sorry

end NUMINAMATH_GPT_cats_in_shelter_l1522_152241


namespace NUMINAMATH_GPT_megatek_manufacturing_percentage_l1522_152296

theorem megatek_manufacturing_percentage :
  ∀ (total_degrees manufacturing_degrees total_percentage : ℝ),
  total_degrees = 360 → manufacturing_degrees = 216 → total_percentage = 100 →
  (manufacturing_degrees / total_degrees) * total_percentage = 60 :=
by
  intros total_degrees manufacturing_degrees total_percentage H1 H2 H3
  rw [H1, H2, H3]
  sorry

end NUMINAMATH_GPT_megatek_manufacturing_percentage_l1522_152296


namespace NUMINAMATH_GPT_simplify_and_evaluate_l1522_152266

noncomputable def x : ℕ := 2023
noncomputable def y : ℕ := 2

theorem simplify_and_evaluate :
  (x + 2 * y)^2 - ((x^3 + 4 * x^2 * y) / x) = 16 :=
by
  sorry

end NUMINAMATH_GPT_simplify_and_evaluate_l1522_152266


namespace NUMINAMATH_GPT_workers_together_time_l1522_152271

-- Definition of the times taken by each worker to complete the job
def timeA : ℚ := 8
def timeB : ℚ := 10
def timeC : ℚ := 12

-- Definition of the rates based on the times
def rateA : ℚ := 1 / timeA
def rateB : ℚ := 1 / timeB
def rateC : ℚ := 1 / timeC

-- Definition of the total rate when working together
def total_rate : ℚ := rateA + rateB + rateC

-- Definition of the total time taken to complete the job when working together
def total_time : ℚ := 1 / total_rate

-- The final theorem we need to prove
theorem workers_together_time : total_time = 120 / 37 :=
by {
  -- structure of the proof will go here, but it is not required as per the instructions
  sorry
}

end NUMINAMATH_GPT_workers_together_time_l1522_152271


namespace NUMINAMATH_GPT_change_combinations_12_dollars_l1522_152243

theorem change_combinations_12_dollars :
  ∃ (solutions : Finset (ℕ × ℕ × ℕ)), 
  (∀ (n d q : ℕ), (n, d, q) ∈ solutions ↔ 5 * n + 10 * d + 25 * q = 1200 ∧ n ≥ 1 ∧ d ≥ 1 ∧ q ≥ 1) ∧ solutions.card = 61 :=
sorry

end NUMINAMATH_GPT_change_combinations_12_dollars_l1522_152243


namespace NUMINAMATH_GPT_compute_f_1_g_3_l1522_152297

def f (x : ℝ) := 3 * x - 5
def g (x : ℝ) := x + 1

theorem compute_f_1_g_3 : f (1 + g 3) = 10 := by
  sorry

end NUMINAMATH_GPT_compute_f_1_g_3_l1522_152297


namespace NUMINAMATH_GPT_solution_set_of_inequality_l1522_152225

theorem solution_set_of_inequality : 
  {x : ℝ | (x - 2) * (x + 1) < 0} = {x : ℝ | -1 < x ∧ x < 2} :=
by sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l1522_152225


namespace NUMINAMATH_GPT_range_of_x_l1522_152240

noncomputable def f (x : ℝ) : ℝ := Real.log x + 2^x + x - 1

theorem range_of_x (x : ℝ) (h : f (x^2 - 4) < 2) : 
  (-Real.sqrt 5 < x ∧ x < -2) ∨ (2 < x ∧ x < Real.sqrt 5) :=
sorry

end NUMINAMATH_GPT_range_of_x_l1522_152240


namespace NUMINAMATH_GPT_squares_centers_equal_perpendicular_l1522_152234

def Square (center : (ℝ × ℝ)) (side : ℝ) := {p : ℝ × ℝ // abs (p.1 - center.1) ≤ side / 2 ∧ abs (p.2 - center.2) ≤ side / 2}

theorem squares_centers_equal_perpendicular 
  (a b : ℝ)
  (O A B C : ℝ × ℝ)
  (hA : A = (a, a))
  (hB : B = (b, 2 * a + b))
  (hC : C = (- (a + b), a + b))
  (hO_vertex : O = (0, 0)) :
  dist O B = dist A C ∧ ∃ m₁ m₂ : ℝ, (B.2 - O.2) / (B.1 - O.1) = m₁ ∧ (C.2 - A.2) / (C.1 - A.1) = m₂ ∧ m₁ * m₂ = -1 := sorry

end NUMINAMATH_GPT_squares_centers_equal_perpendicular_l1522_152234


namespace NUMINAMATH_GPT_geometric_sequence_a_5_l1522_152254

noncomputable def a_n : ℕ → ℝ := sorry

theorem geometric_sequence_a_5 :
  (∀ n : ℕ, ∃ r : ℝ, a_n (n + 1) = r * a_n n) →  -- geometric sequence property
  (∃ x₁ x₂ : ℝ, x₁ + x₂ = -7 ∧ x₁ * x₂ = 9 ∧ a_n 3 = x₁ ∧ a_n 7 = x₂) →  -- roots of the quadratic equation and their assignments
  a_n 5 = -3 := sorry

end NUMINAMATH_GPT_geometric_sequence_a_5_l1522_152254


namespace NUMINAMATH_GPT_limsup_subset_l1522_152200

variable {Ω : Type*} -- assuming a universal sample space Ω for the events A_n and B_n

def limsup (A : ℕ → Set Ω) : Set Ω := 
  ⋂ k, ⋃ n ≥ k, A n

theorem limsup_subset {A B : ℕ → Set Ω} (h : ∀ n, A n ⊆ B n) : 
  limsup A ⊆ limsup B :=
by
  -- here goes the proof
  sorry

end NUMINAMATH_GPT_limsup_subset_l1522_152200


namespace NUMINAMATH_GPT_Penny_total_species_identified_l1522_152282

/-- Penny identified 35 species of sharks, 15 species of eels, and 5 species of whales.
    Prove that the total number of species identified is 55. -/
theorem Penny_total_species_identified :
  let sharks_species := 35
  let eels_species := 15
  let whales_species := 5
  sharks_species + eels_species + whales_species = 55 :=
by
  sorry

end NUMINAMATH_GPT_Penny_total_species_identified_l1522_152282


namespace NUMINAMATH_GPT_compare_abc_l1522_152284

noncomputable def a : ℝ := 2 * Real.log (21 / 20)
noncomputable def b : ℝ := Real.log (11 / 10)
noncomputable def c : ℝ := Real.sqrt 1.2 - 1

theorem compare_abc : a > b ∧ b < c ∧ a > c :=
by {
  sorry
}

end NUMINAMATH_GPT_compare_abc_l1522_152284


namespace NUMINAMATH_GPT_arithmetic_sum_l1522_152236

variable {a : ℕ → ℝ}

def is_arithmetic_seq (a : ℕ → ℝ) : Prop :=
∀ n, a (n + 1) - a n = a 1 - a 0

theorem arithmetic_sum :
  is_arithmetic_seq a →
  a 5 + a 6 + a 7 = 15 →
  a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9 = 35 :=
by
  intros
  sorry

end NUMINAMATH_GPT_arithmetic_sum_l1522_152236


namespace NUMINAMATH_GPT_lateral_surface_area_of_cylinder_l1522_152215

variable (m n : ℝ) (S : ℝ)

theorem lateral_surface_area_of_cylinder (h1 : S > 0) (h2 : m > 0) (h3 : n > 0) :
  ∃ (lateral_surface_area : ℝ),
    lateral_surface_area = (π * S) / (Real.sin (π * n / (m + n))) :=
sorry

end NUMINAMATH_GPT_lateral_surface_area_of_cylinder_l1522_152215


namespace NUMINAMATH_GPT_not_all_positive_l1522_152247

theorem not_all_positive (a b c : ℝ) (h1 : a + b + c = 4) (h2 : a^2 + b^2 + c^2 = 12) (h3 : a * b * c = 1) : a ≤ 0 ∨ b ≤ 0 ∨ c ≤ 0 :=
sorry

end NUMINAMATH_GPT_not_all_positive_l1522_152247


namespace NUMINAMATH_GPT_find_m_l1522_152283

noncomputable def distance (p1 p2 : (ℝ × ℝ)) : ℝ :=
  Real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

def point_on_x_axis_distance (x y : ℝ) : Prop :=
  y = 14

def point_distance_from_fixed_point (x y : ℝ) : Prop :=
  distance (x, y) (3, 8) = 8

def x_coordinate_condition (x : ℝ) : Prop :=
  x > 3

def m_distance (x y m : ℝ) : Prop :=
  distance (x, y) (0, 0) = m

theorem find_m (x y m : ℝ) 
  (h1 : point_on_x_axis_distance x y) 
  (h2 : point_distance_from_fixed_point x y) 
  (h3 : x_coordinate_condition x) :
  m_distance x y m → 
  m = Real.sqrt (233 + 12 * Real.sqrt 7) := by
  sorry

end NUMINAMATH_GPT_find_m_l1522_152283


namespace NUMINAMATH_GPT_sara_peaches_l1522_152279

theorem sara_peaches (initial_peaches : ℕ) (picked_peaches : ℕ) (total_peaches : ℕ) 
  (h1 : initial_peaches = 24) (h2 : picked_peaches = 37) : 
  total_peaches = 61 :=
by
  sorry

end NUMINAMATH_GPT_sara_peaches_l1522_152279


namespace NUMINAMATH_GPT_mary_investment_l1522_152269

noncomputable def compound_interest (P r : ℝ) (n t : ℕ) : ℝ :=
  P * (1 + r / n) ^ (n * t)

theorem mary_investment :
  ∃ (P : ℝ), P = 51346 ∧ compound_interest P 0.10 12 7 = 100000 :=
by
  sorry

end NUMINAMATH_GPT_mary_investment_l1522_152269


namespace NUMINAMATH_GPT_viola_final_jump_l1522_152276

variable (n : ℕ) (T : ℝ) (x : ℝ)

theorem viola_final_jump (h1 : T = 3.80 * n)
                        (h2 : (T + 3.99) / (n + 1) = 3.81)
                        (h3 : T + 3.99 + x = 3.82 * (n + 2)) : 
                        x = 4.01 :=
sorry

end NUMINAMATH_GPT_viola_final_jump_l1522_152276


namespace NUMINAMATH_GPT_no_integer_solution_for_expression_l1522_152262

theorem no_integer_solution_for_expression (x y z : ℤ) :
  x^4 + y^4 + z^4 - 2 * x^2 * y^2 - 2 * y^2 * z^2 - 2 * z^2 * x^2 ≠ 2000 :=
by sorry

end NUMINAMATH_GPT_no_integer_solution_for_expression_l1522_152262


namespace NUMINAMATH_GPT_joy_sees_grandma_in_48_hours_l1522_152261

def days_until_joy_sees_grandma : ℕ := 2
def hours_per_day : ℕ := 24

theorem joy_sees_grandma_in_48_hours :
  days_until_joy_sees_grandma * hours_per_day = 48 := 
by
  sorry

end NUMINAMATH_GPT_joy_sees_grandma_in_48_hours_l1522_152261


namespace NUMINAMATH_GPT_range_of_m_for_one_real_root_l1522_152213

def f (x : ℝ) (m : ℝ) : ℝ := x^3 - 3*x + m

theorem range_of_m_for_one_real_root :
  (∃! x : ℝ, f x m = 0) ↔ (m < -2 ∨ m > 2) := by
  sorry

end NUMINAMATH_GPT_range_of_m_for_one_real_root_l1522_152213


namespace NUMINAMATH_GPT_yellow_more_than_purple_l1522_152263
-- Import math library for necessary definitions.

-- Define the problem conditions in Lean
def num_purple_candies : ℕ := 10
def num_total_candies : ℕ := 36

axiom exists_yellow_and_green_candies 
  (Y G : ℕ) 
  (h1 : G = Y - 2) 
  (h2 : 10 + Y + G = 36) : True

-- The theorem to prove
theorem yellow_more_than_purple 
  (Y : ℕ) 
  (hY : exists (G : ℕ), G = Y - 2 ∧ 10 + Y + G = 36) : Y - num_purple_candies = 4 :=
by {
  sorry -- proof is not required
}

end NUMINAMATH_GPT_yellow_more_than_purple_l1522_152263


namespace NUMINAMATH_GPT_polygon_with_equal_angle_sums_is_quadrilateral_l1522_152248

theorem polygon_with_equal_angle_sums_is_quadrilateral 
    (n : ℕ)
    (h1 : (n - 2) * 180 = 360)
    (h2 : 360 = 360) :
  n = 4 := 
sorry

end NUMINAMATH_GPT_polygon_with_equal_angle_sums_is_quadrilateral_l1522_152248


namespace NUMINAMATH_GPT_average_of_class_is_49_5_l1522_152205

noncomputable def average_score_of_class : ℝ :=
  let total_students := 50
  let students_95 := 5
  let students_0 := 5
  let students_85 := 5
  let remaining_students := total_students - (students_95 + students_0 + students_85)
  let total_marks := (students_95 * 95) + (students_0 * 0) + (students_85 * 85) + (remaining_students * 45)
  total_marks / total_students

theorem average_of_class_is_49_5 : average_score_of_class = 49.5 := 
by sorry

end NUMINAMATH_GPT_average_of_class_is_49_5_l1522_152205


namespace NUMINAMATH_GPT_minimum_value_of_PA_PF_l1522_152273

noncomputable def ellipse_min_distance : ℝ :=
  let F := (1, 0)
  let A := (1, 1)
  let a : ℝ := 3
  let F1 := (-1, 0)
  let d_A_F1 : ℝ := Real.sqrt ((-1 - 1)^2 + (0 - 1)^2)
  6 - d_A_F1

theorem minimum_value_of_PA_PF :
  ellipse_min_distance = 6 - Real.sqrt 5 :=
by
  sorry

end NUMINAMATH_GPT_minimum_value_of_PA_PF_l1522_152273


namespace NUMINAMATH_GPT_part1_solution_set_part2_inequality_l1522_152255

noncomputable def f (x : ℝ) : ℝ := 
  x * Real.exp (x + 1)

theorem part1_solution_set (h : 0 < x) : 
  f x < 3 * Real.log 3 - 3 ↔ 0 < x ∧ x < Real.log 3 - 1 :=
sorry

theorem part2_inequality (h1 : f x1 = 3 * Real.exp x1 + 3 * Real.exp (Real.log x1)) 
    (h2 : f x2 = 3 * Real.exp x2 + 3 * Real.exp (Real.log x2)) (h_distinct : x1 ≠ x2) :
  x1 + x2 + Real.log (x1 * x2) > 2 :=
sorry

end NUMINAMATH_GPT_part1_solution_set_part2_inequality_l1522_152255


namespace NUMINAMATH_GPT_football_game_wristbands_l1522_152274

theorem football_game_wristbands (total_wristbands wristbands_per_person : Nat) (h1 : total_wristbands = 290) (h2 : wristbands_per_person = 2) :
  total_wristbands / wristbands_per_person = 145 :=
by
  sorry

end NUMINAMATH_GPT_football_game_wristbands_l1522_152274


namespace NUMINAMATH_GPT_pyramid_rhombus_side_length_l1522_152246

theorem pyramid_rhombus_side_length
  (α β S: ℝ) (hα : 0 < α) (hβ : 0 < β) (hS : 0 < S) :
  ∃ a : ℝ, a = 2 * Real.sqrt (2 * S * Real.cos β / Real.sin α) :=
by
  sorry

end NUMINAMATH_GPT_pyramid_rhombus_side_length_l1522_152246


namespace NUMINAMATH_GPT_eq_infinite_solutions_function_satisfies_identity_l1522_152293

-- First Part: Proving the equation has infinitely many positive integer solutions
theorem eq_infinite_solutions : ∃ (x y z t : ℕ), ∀ n : ℕ, x^2 + 2 * y^2 = z^2 + 2 * t^2 ∧ x > 0 ∧ y > 0 ∧ z > 0 ∧ t > 0 := 
sorry

-- Second Part: Finding and proving the function f
def f (n : ℕ) : ℕ := n

theorem function_satisfies_identity (f : ℕ → ℕ) (h : ∀ m n : ℕ, f (f n^2 + 2 * f m^2) = n^2 + 2 * m^2) : ∀ k : ℕ, f k = k :=
sorry

end NUMINAMATH_GPT_eq_infinite_solutions_function_satisfies_identity_l1522_152293


namespace NUMINAMATH_GPT_domino_tile_count_l1522_152230

theorem domino_tile_count (low high : ℕ) (tiles_standard_set : ℕ) (range_standard_set : ℕ) (range_new_set : ℕ) :
  range_standard_set = 6 → tiles_standard_set = 28 →
  low = 0 → high = 12 →
  range_new_set = 13 → 
  (∀ n, 0 ≤ n ∧ n ≤ range_standard_set → ∀ m, n ≤ m ∧ m ≤ range_standard_set → n ≤ m → true) →
  (∀ n, 0 ≤ n ∧ n ≤ range_new_set → ∀ m, n ≤ m ∧ m <= range_new_set → n <= m → true) →
  tiles_new_set = 91 :=
by
  intros h_range_standard h_tiles_standard h_low h_high h_range_new h_standard_pairs h_new_pairs
  --skipping the proof
  sorry

end NUMINAMATH_GPT_domino_tile_count_l1522_152230


namespace NUMINAMATH_GPT_isosceles_triangle_perimeter_l1522_152203

theorem isosceles_triangle_perimeter 
  (a b : ℕ) 
  (h_iso : a = b ∨ a = 3 ∨ b = 3) 
  (h_sides : a = 6 ∨ b = 6) 
  : a + b + 3 = 15 := by
  sorry

end NUMINAMATH_GPT_isosceles_triangle_perimeter_l1522_152203


namespace NUMINAMATH_GPT_geometric_series_sum_l1522_152244

theorem geometric_series_sum :
  let a := 3
  let r := 3
  let n := 9
  let last_term := a * r^(n - 1)
  last_term = 19683 →
  let S := a * (r^n - 1) / (r - 1)
  S = 29523 :=
by
  intros
  sorry

end NUMINAMATH_GPT_geometric_series_sum_l1522_152244


namespace NUMINAMATH_GPT_time_train_passes_jogger_l1522_152207

noncomputable def jogger_speed_kmph : ℝ := 9
noncomputable def jogger_speed_mps : ℝ := jogger_speed_kmph * 1000 / 3600

noncomputable def train_speed_kmph : ℝ := 45
noncomputable def train_speed_mps : ℝ := train_speed_kmph * 1000 / 3600

noncomputable def relative_speed_mps : ℝ := train_speed_mps - jogger_speed_mps

noncomputable def initial_lead_m : ℝ := 150
noncomputable def train_length_m : ℝ := 100

noncomputable def total_distance_to_cover_m : ℝ := initial_lead_m + train_length_m

noncomputable def time_to_pass_jogger_s : ℝ := total_distance_to_cover_m / relative_speed_mps

theorem time_train_passes_jogger : time_to_pass_jogger_s = 25 := by
  sorry

end NUMINAMATH_GPT_time_train_passes_jogger_l1522_152207


namespace NUMINAMATH_GPT_number_of_adult_tickets_l1522_152285

-- Define the parameters of the problem
def price_adult_ticket : ℝ := 5.50
def price_child_ticket : ℝ := 3.50
def total_tickets : ℕ := 21
def total_cost : ℝ := 83.50

-- Define the main theorem to be proven
theorem number_of_adult_tickets : 
  ∃ (A C : ℕ), A + C = total_tickets ∧ 
                (price_adult_ticket * A + price_child_ticket * C = total_cost) ∧ 
                 A = 5 :=
by
  -- The proof content will be filled in later
  sorry

end NUMINAMATH_GPT_number_of_adult_tickets_l1522_152285


namespace NUMINAMATH_GPT_sum_of_odd_function_at_points_l1522_152294

def is_odd_function (f : ℝ → ℝ) := ∀ x : ℝ, f (-x) = -f x

theorem sum_of_odd_function_at_points (f : ℝ → ℝ) (h : is_odd_function f) : 
  f (-2) + f (-1) + f 0 + f 1 + f 2 = 0 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_odd_function_at_points_l1522_152294


namespace NUMINAMATH_GPT_jill_speed_downhill_l1522_152298

theorem jill_speed_downhill 
  (up_speed : ℕ) (total_time : ℕ) (hill_distance : ℕ) 
  (up_time : ℕ) (down_time : ℕ) (down_speed : ℕ) 
  (h1 : up_speed = 9)
  (h2 : total_time = 175)
  (h3 : hill_distance = 900)
  (h4 : up_time = hill_distance / up_speed)
  (h5 : down_time = total_time - up_time)
  (h6 : down_speed = hill_distance / down_time) :
  down_speed = 12 := 
  by
    sorry

end NUMINAMATH_GPT_jill_speed_downhill_l1522_152298


namespace NUMINAMATH_GPT_compute_3X4_l1522_152221

def operation_X (a b : ℤ) : ℤ := b + 12 * a - a^2

theorem compute_3X4 : operation_X 3 4 = 31 := 
by
  sorry

end NUMINAMATH_GPT_compute_3X4_l1522_152221


namespace NUMINAMATH_GPT_solution_set_of_quadratic_inequality_l1522_152211

theorem solution_set_of_quadratic_inequality (a b : ℝ)
  (h : ∀ x : ℝ, ax^2 + bx - 2 > 0 ↔ x ∈ Set.Ioo (-4 : ℝ) 1) :
  a + b = 2 := 
sorry

end NUMINAMATH_GPT_solution_set_of_quadratic_inequality_l1522_152211


namespace NUMINAMATH_GPT_students_present_l1522_152287

theorem students_present (absent_students male_students female_student_diff : ℕ) 
  (h1 : absent_students = 18) 
  (h2 : male_students = 848) 
  (h3 : female_student_diff = 49) : 
  (male_students + (male_students - female_student_diff) - absent_students = 1629) := 

by 
  sorry

end NUMINAMATH_GPT_students_present_l1522_152287


namespace NUMINAMATH_GPT_ratio_of_percentages_l1522_152257

theorem ratio_of_percentages (x y : ℝ) (h : 0.10 * x = 0.20 * y) : x / y = 2 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_percentages_l1522_152257


namespace NUMINAMATH_GPT_ratio_of_girls_to_boys_l1522_152249

theorem ratio_of_girls_to_boys (total_students girls boys : ℕ) 
  (h1 : total_students = 26) 
  (h2 : girls = boys + 6) 
  (h3 : girls + boys = total_students) : 
  (girls : ℚ) / boys = 8 / 5 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_girls_to_boys_l1522_152249


namespace NUMINAMATH_GPT_midpoint_coordinates_l1522_152260

theorem midpoint_coordinates (xM yM xN yN : ℝ) (hM : xM = 3) (hM' : yM = -2) (hN : xN = -1) (hN' : yN = 0) :
  (xM + xN) / 2 = 1 ∧ (yM + yN) / 2 = -1 :=
by
  simp [hM, hM', hN, hN']
  sorry

end NUMINAMATH_GPT_midpoint_coordinates_l1522_152260


namespace NUMINAMATH_GPT_relation_y1_y2_y3_l1522_152218

-- Definition of being on the parabola
def on_parabola (x : ℝ) (y m : ℝ) : Prop := y = -3*x^2 - 12*x + m

-- The conditions given in the problem
variables {y1 y2 y3 m : ℝ}

-- The points (-3, y1), (-2, y2), (1, y3) are on the parabola given by the equation
axiom h1 : on_parabola (-3) y1 m
axiom h2 : on_parabola (-2) y2 m
axiom h3 : on_parabola (1) y3 m

-- We need to prove the relationship between y1, y2, and y3
theorem relation_y1_y2_y3 : y2 > y1 ∧ y1 > y3 :=
by { sorry }

end NUMINAMATH_GPT_relation_y1_y2_y3_l1522_152218


namespace NUMINAMATH_GPT_frequency_of_scoring_l1522_152201

def shots : ℕ := 80
def goals : ℕ := 50
def frequency : ℚ := goals / shots

theorem frequency_of_scoring : frequency = 0.625 := by
  sorry

end NUMINAMATH_GPT_frequency_of_scoring_l1522_152201


namespace NUMINAMATH_GPT_increasing_interval_l1522_152229

def my_function (x : ℝ) : ℝ := -(x - 3) * |x|

theorem increasing_interval : ∀ x y : ℝ, 0 ≤ x → x ≤ y → my_function x ≤ my_function y :=
by
  sorry

end NUMINAMATH_GPT_increasing_interval_l1522_152229


namespace NUMINAMATH_GPT_calculate_expression_l1522_152295

theorem calculate_expression : 
  (3.242 * (14 + 6) - 7.234 * 7) / 20 = 0.7101 :=
by
  sorry

end NUMINAMATH_GPT_calculate_expression_l1522_152295


namespace NUMINAMATH_GPT_arithmetic_sum_expression_zero_l1522_152232

theorem arithmetic_sum_expression_zero (a d : ℤ) (i j k : ℕ) (S_i S_j S_k : ℤ) :
  S_i = i * (a + (i - 1) * d / 2) →
  S_j = j * (a + (j - 1) * d / 2) →
  S_k = k * (a + (k - 1) * d / 2) →
  (S_i / i * (j - k) + S_j / j * (k - i) + S_k / k * (i - j) = 0) :=
by
  intros hS_i hS_j hS_k
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_arithmetic_sum_expression_zero_l1522_152232


namespace NUMINAMATH_GPT_no_solution_eq_l1522_152208

theorem no_solution_eq (k : ℝ) :
  (¬ ∃ x : ℝ, x ≠ 3 ∧ x ≠ 7 ∧ (x + 2) / (x - 3) = (x - k) / (x - 7)) ↔ k = 2 :=
by
  sorry

end NUMINAMATH_GPT_no_solution_eq_l1522_152208


namespace NUMINAMATH_GPT_train_speed_l1522_152252

theorem train_speed (distance : ℝ) (time : ℝ) (h_distance : distance = 400) (h_time : time = 40) : distance / time = 10 := by
  rw [h_distance, h_time]
  norm_num

end NUMINAMATH_GPT_train_speed_l1522_152252


namespace NUMINAMATH_GPT_scientific_notation_of_4600000000_l1522_152224

theorem scientific_notation_of_4600000000 :
  4.6 * 10^9 = 4600000000 := 
by
  sorry

end NUMINAMATH_GPT_scientific_notation_of_4600000000_l1522_152224


namespace NUMINAMATH_GPT_solution_set_inequality_l1522_152288

theorem solution_set_inequality (x : ℝ) : (-2 * x + 3 < 0) ↔ (x > 3 / 2) := by 
  sorry

end NUMINAMATH_GPT_solution_set_inequality_l1522_152288


namespace NUMINAMATH_GPT_jungkook_red_balls_l1522_152245

-- Definitions from conditions
def num_boxes : ℕ := 2
def red_balls_per_box : ℕ := 3

-- Theorem stating the problem
theorem jungkook_red_balls : (num_boxes * red_balls_per_box) = 6 :=
by sorry

end NUMINAMATH_GPT_jungkook_red_balls_l1522_152245


namespace NUMINAMATH_GPT_units_digit_of_product_of_seven_consecutive_integers_is_zero_l1522_152277

/-- Define seven consecutive positive integers and show the units digit of their product is 0 -/
theorem units_digit_of_product_of_seven_consecutive_integers_is_zero (n : ℕ) :
  ∃ (k : ℕ), k = (n * (n + 1) * (n + 2) * (n + 3) * (n + 4) * (n + 5) * (n + 6)) % 10 ∧ k = 0 :=
by {
  -- We state that the units digit k of the product of seven consecutive integers
  -- starting from n is 0
  sorry
}

end NUMINAMATH_GPT_units_digit_of_product_of_seven_consecutive_integers_is_zero_l1522_152277


namespace NUMINAMATH_GPT_stock_reaches_N_fourth_time_l1522_152214

noncomputable def stock_at_k (c0 a b : ℝ) (k : ℕ) : ℝ :=
  if k % 2 = 0 then c0 + (k / 2) * (a - b)
  else c0 + (k / 2 + 1) * a - (k / 2) * b

theorem stock_reaches_N_fourth_time (c0 a b N : ℝ) (hN3 : ∃ k1 k2 k3 : ℕ, k1 ≠ k2 ∧ k2 ≠ k3 ∧ k1 ≠ k3 ∧ stock_at_k c0 a b k1 = N ∧ stock_at_k c0 a b k2 = N ∧ stock_at_k c0 a b k3 = N) :
  ∃ k4 : ℕ, k4 ≠ k1 ∧ k4 ≠ k2 ∧ k4 ≠ k3 ∧ stock_at_k c0 a b k4 = N := 
sorry

end NUMINAMATH_GPT_stock_reaches_N_fourth_time_l1522_152214


namespace NUMINAMATH_GPT_one_third_of_1206_is_300_percent_of_134_l1522_152299

theorem one_third_of_1206_is_300_percent_of_134 :
  let number := 1206
  let fraction := 1 / 3
  let computed_one_third := fraction * number
  let whole := 134
  let expected_percent := 300
  let percent := (computed_one_third / whole) * 100
  percent = expected_percent := by
  let number := 1206
  let fraction := 1 / 3
  have computed_one_third : ℝ := fraction * number
  let whole := 134
  let expected_percent := 300
  have percent : ℝ := (computed_one_third / whole) * 100
  exact sorry

end NUMINAMATH_GPT_one_third_of_1206_is_300_percent_of_134_l1522_152299


namespace NUMINAMATH_GPT_cos_value_l1522_152268

theorem cos_value (α : ℝ) 
  (h1 : Real.sin (α + Real.pi / 12) = 1 / 3) : 
  Real.cos (α + 7 * Real.pi / 12) = -(1 + Real.sqrt 24) / 6 :=
sorry

end NUMINAMATH_GPT_cos_value_l1522_152268


namespace NUMINAMATH_GPT_mr_smiths_sixth_child_not_represented_l1522_152226

def car_plate_number := { n : ℕ // ∃ a b : ℕ, n = 1001 * a + 110 * b ∧ 1 ≤ a ∧ a ≤ 9 ∧ 1 ≤ b ∧ b ≤ 9 }
def mr_smith_is_45 (n : ℕ) := (n % 100) = 45
def divisible_by_children_ages (n : ℕ) : Prop := ∀ i : ℕ, 1 ≤ i ∧ i ≤ 9 → n % i = 0

theorem mr_smiths_sixth_child_not_represented :
    ∃ n : car_plate_number, mr_smith_is_45 n.val ∧ divisible_by_children_ages n.val → ¬ (6 ∣ n.val) :=
by
  sorry

end NUMINAMATH_GPT_mr_smiths_sixth_child_not_represented_l1522_152226


namespace NUMINAMATH_GPT_ratio_of_areas_l1522_152209

-- Define the squares and their side lengths
def Square (side_length : ℝ) := side_length * side_length

-- Define the side lengths of Square C and Square D
def side_C (x : ℝ) : ℝ := x
def side_D (x : ℝ) : ℝ := 3 * x

-- Define their areas
def area_C (x : ℝ) : ℝ := Square (side_C x)
def area_D (x : ℝ) : ℝ := Square (side_D x)

-- The statement to prove
theorem ratio_of_areas (x : ℝ) (hx : x ≠ 0) : area_C x / area_D x = 1 / 9 := by
  sorry

end NUMINAMATH_GPT_ratio_of_areas_l1522_152209


namespace NUMINAMATH_GPT_power_addition_l1522_152291

theorem power_addition (y : ℕ) (h : 3^y = 81) : 3^(y + 3) = 2187 := by
  sorry

end NUMINAMATH_GPT_power_addition_l1522_152291


namespace NUMINAMATH_GPT_symmetry_center_on_line_l1522_152290

def symmetry_center_curve :=
  ∃ θ : ℝ, (∃ x y : ℝ, (x = -1 + Real.cos θ ∧ y = 2 + Real.sin θ))

-- The main theorem to prove
theorem symmetry_center_on_line : 
  (∃ cx cy : ℝ, (symmetry_center_curve ∧ (cy = -2 * cx))) :=
sorry

end NUMINAMATH_GPT_symmetry_center_on_line_l1522_152290


namespace NUMINAMATH_GPT_solution_k_values_l1522_152212

theorem solution_k_values (k : ℕ) : 
  (∃ m n : ℕ, 0 < m ∧ 0 < n ∧ m * (m + k) = n * (n + 1)) 
  → k = 1 ∨ 4 ≤ k := 
by
  sorry

end NUMINAMATH_GPT_solution_k_values_l1522_152212


namespace NUMINAMATH_GPT_tomato_price_per_kilo_l1522_152270

theorem tomato_price_per_kilo 
  (initial_money: ℝ) (money_left: ℝ)
  (potato_price_per_kilo: ℝ) (potato_kilos: ℝ)
  (cucumber_price_per_kilo: ℝ) (cucumber_kilos: ℝ)
  (banana_price_per_kilo: ℝ) (banana_kilos: ℝ)
  (tomato_kilos: ℝ)
  (spent_on_potatoes: initial_money - money_left = potato_price_per_kilo * potato_kilos)
  (spent_on_cucumbers: initial_money - money_left = cucumber_price_per_kilo * cucumber_kilos)
  (spent_on_bananas: initial_money - money_left = banana_price_per_kilo * banana_kilos)
  (total_spent: initial_money - money_left = 74)
  : (74 - (potato_price_per_kilo * potato_kilos + cucumber_price_per_kilo * cucumber_kilos + banana_price_per_kilo * banana_kilos)) / tomato_kilos = 3 := 
sorry

end NUMINAMATH_GPT_tomato_price_per_kilo_l1522_152270


namespace NUMINAMATH_GPT_marble_group_l1522_152292

theorem marble_group (x : ℕ) (h1 : 144 % x = 0) (h2 : 144 % (x + 2) = (144 / x) - 1) : x = 16 :=
sorry

end NUMINAMATH_GPT_marble_group_l1522_152292


namespace NUMINAMATH_GPT_smallest_number_is_1111_in_binary_l1522_152222

theorem smallest_number_is_1111_in_binary :
  let a := 15   -- Decimal equivalent of 1111 in binary
  let b := 78   -- Decimal equivalent of 210 in base 6
  let c := 64   -- Decimal equivalent of 1000 in base 4
  let d := 65   -- Decimal equivalent of 101 in base 8
  a < b ∧ a < c ∧ a < d := 
by
  let a := 15
  let b := 78
  let c := 64
  let d := 65
  show a < b ∧ a < c ∧ a < d
  sorry

end NUMINAMATH_GPT_smallest_number_is_1111_in_binary_l1522_152222


namespace NUMINAMATH_GPT_unique_pyramid_formation_l1522_152265

theorem unique_pyramid_formation:
  ∀ (positions: Finset ℕ)
    (is_position_valid: ℕ → Prop),
    (positions.card = 5) → 
    (∀ n ∈ positions, n < 5) → 
    (∃! n, is_position_valid n) :=
by
  sorry

end NUMINAMATH_GPT_unique_pyramid_formation_l1522_152265


namespace NUMINAMATH_GPT_division_by_fraction_l1522_152264

theorem division_by_fraction : 5 / (1 / 5) = 25 := by
  sorry

end NUMINAMATH_GPT_division_by_fraction_l1522_152264


namespace NUMINAMATH_GPT_complement_A_in_U_l1522_152216

def U : Set ℝ := {x : ℝ | x > 0}
def A : Set ℝ := {x : ℝ | 1 < x ∧ x < 2}
def AC : Set ℝ := {x : ℝ | (0 < x ∧ x ≤ 1) ∨ (2 ≤ x)}

theorem complement_A_in_U : U \ A = AC := 
by 
  sorry

end NUMINAMATH_GPT_complement_A_in_U_l1522_152216


namespace NUMINAMATH_GPT_max_knights_seated_l1522_152238

theorem max_knights_seated (total_islanders : ℕ) (half_islanders : ℕ) 
  (knight_statement_half : ℕ) (liar_statement_half : ℕ) :
  total_islanders = 100 ∧ knight_statement_half = 50 
    ∧ liar_statement_half = 50 
    ∧ (∀ (k : ℕ), (knight_statement_half = k ∧ liar_statement_half = k)
    → (k ≤ 67)) →
  ∃ K : ℕ, K ≤ 67 :=
by
  -- the proof goes here
  sorry

end NUMINAMATH_GPT_max_knights_seated_l1522_152238


namespace NUMINAMATH_GPT_part1_part1_monotonicity_intervals_part2_l1522_152272

noncomputable def f (x a : ℝ) := x * Real.log x - a * (x - 1)^2 - x + 1

-- Part 1: Monotonicity and Extreme values when a = 0
theorem part1 (x : ℝ) : f x 0 = x * Real.log x - x + 1 := sorry

theorem part1_monotonicity_intervals (x : ℝ) :
  (∀ (x : ℝ), 0 < x ∧ x < 1 → f x 0 < f 1 0) ∧
  (∀ (x : ℝ), x > 1 → f 1 0 < f x 0) ∧ 
  (f 1 0 = 0) := sorry

-- Part 2: f(x) < 0 for x > 1 and a >= 1/2
theorem part2 (x a : ℝ) (hx : x > 1) (ha : a ≥ 1/2) : f x a < 0 := sorry

end NUMINAMATH_GPT_part1_part1_monotonicity_intervals_part2_l1522_152272


namespace NUMINAMATH_GPT_factor_expression_l1522_152267

theorem factor_expression (c : ℝ) : 180 * c ^ 2 + 36 * c = 36 * c * (5 * c + 1) := 
by
  sorry

end NUMINAMATH_GPT_factor_expression_l1522_152267


namespace NUMINAMATH_GPT_find_a_b_c_l1522_152256

noncomputable def x : ℝ := Real.sqrt ((Real.sqrt 37) / 2 + 3 / 2)

theorem find_a_b_c :
  ∃ a b c : ℕ, (x^80 = 2 * x^78 + 8 * x^76 + 9 * x^74 - x^40 + a * x^36 + b * x^34 + c * x^30) ∧ (a + b + c = 151) :=
by
  sorry

end NUMINAMATH_GPT_find_a_b_c_l1522_152256


namespace NUMINAMATH_GPT_books_about_sports_l1522_152233

theorem books_about_sports (total_books school_books sports_books : ℕ) 
  (h1 : total_books = 58)
  (h2 : school_books = 19) 
  (h3 : sports_books = total_books - school_books) :
  sports_books = 39 :=
by 
  rw [h1, h2] at h3 
  exact h3

end NUMINAMATH_GPT_books_about_sports_l1522_152233


namespace NUMINAMATH_GPT_find_second_number_l1522_152204

theorem find_second_number (x : ℝ) 
    (h : (14 + x + 53) / 3 = (21 + 47 + 22) / 3 + 3) : 
    x = 32 := 
by 
    sorry

end NUMINAMATH_GPT_find_second_number_l1522_152204


namespace NUMINAMATH_GPT_camel_height_in_feet_l1522_152281

theorem camel_height_in_feet (h_ht_14 : ℕ) (ratio : ℕ) (inch_to_ft : ℕ) : ℕ :=
  let hare_height := 14
  let camel_height_in_inches := hare_height * 24
  let camel_height_in_feet := camel_height_in_inches / 12
  camel_height_in_feet
#print camel_height_in_feet

example : camel_height_in_feet 14 24 12 = 28 := by sorry

end NUMINAMATH_GPT_camel_height_in_feet_l1522_152281


namespace NUMINAMATH_GPT_problem_statement_l1522_152228

theorem problem_statement (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : b < a) (f : ℝ → ℝ) 
  (h_f : ∀ x, f x = |Real.log x|) (h_eq : f a = f b) :
  a * b = 1 ∧ Real.exp a + Real.exp b > 2 * Real.exp 1 ∧ (1 / a)^2 - b + 5 / 4 ≥ 1 :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l1522_152228
