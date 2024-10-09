import Mathlib

namespace min_distance_circle_tangent_l2014_201445

theorem min_distance_circle_tangent
  (P : ℝ × ℝ)
  (hP: 3 * P.1 + 4 * P.2 = 11) :
  ∃ d : ℝ, d = 11 / 5 := 
sorry

end min_distance_circle_tangent_l2014_201445


namespace num_students_l2014_201466

theorem num_students (n : ℕ) 
    (average_marks_wrong : ℕ)
    (wrong_mark : ℕ)
    (correct_mark : ℕ)
    (average_marks_correct : ℕ) :
    average_marks_wrong = 100 →
    wrong_mark = 90 →
    correct_mark = 10 →
    average_marks_correct = 92 →
    n = 10 :=
by
  intros h1 h2 h3 h4
  sorry

end num_students_l2014_201466


namespace evaluate_expression_l2014_201451

theorem evaluate_expression (a b : ℤ) (h_a : a = 4) (h_b : b = -3) : -a - b^3 + a * b = 11 :=
by
  rw [h_a, h_b]
  sorry

end evaluate_expression_l2014_201451


namespace perpendicular_planes_parallel_l2014_201460

-- Define the lines m and n, and planes alpha and beta
def Line := Unit
def Plane := Unit

-- Define perpendicular and parallel relations
def perpendicular (l : Line) (p : Plane) : Prop := sorry
def parallel (p₁ p₂ : Plane) : Prop := sorry

-- The main theorem statement: If m ⊥ α and m ⊥ β, then α ∥ β
theorem perpendicular_planes_parallel (m : Line) (α β : Plane)
  (h₁ : perpendicular m α) (h₂ : perpendicular m β) : parallel α β :=
sorry

end perpendicular_planes_parallel_l2014_201460


namespace find_other_number_l2014_201408

-- Define the conditions
variable (B : ℕ)
variable (HCF : ℕ → ℕ → ℕ)
variable (LCM : ℕ → ℕ → ℕ)

axiom hcf_cond : HCF 24 B = 15
axiom lcm_cond : LCM 24 B = 312

-- The theorem statement
theorem find_other_number (B : ℕ) (HCF : ℕ → ℕ → ℕ) (LCM : ℕ → ℕ → ℕ) 
  (hcf_cond : HCF 24 B = 15) (lcm_cond : LCM 24 B = 312) : 
  B = 195 :=
sorry

end find_other_number_l2014_201408


namespace length_of_bridge_l2014_201476

def length_of_train : ℝ := 135  -- Length of the train in meters
def speed_of_train_km_per_hr : ℝ := 45  -- Speed of the train in km/hr
def speed_of_train_m_per_s : ℝ := 12.5  -- Speed of the train in m/s
def time_to_cross_bridge : ℝ := 30  -- Time to cross the bridge in seconds
def distance_covered : ℝ := speed_of_train_m_per_s * time_to_cross_bridge  -- Total distance covered

theorem length_of_bridge :
  distance_covered - length_of_train = 240 :=
by
  sorry

end length_of_bridge_l2014_201476


namespace distinct_rationals_count_l2014_201450

theorem distinct_rationals_count : ∃ N : ℕ, (N = 40) ∧ ∀ k : ℚ, (|k| < 100) → (∃ x : ℤ, 3 * x^2 + k * x + 8 = 0) :=
by
  sorry

end distinct_rationals_count_l2014_201450


namespace solve_equation_l2014_201417

theorem solve_equation :
  ∀ x : ℝ, 81 * (1 - x) ^ 2 = 64 ↔ x = 1 / 9 ∨ x = 17 / 9 :=
by
  sorry

end solve_equation_l2014_201417


namespace triangle_area_l2014_201464

noncomputable def vector (x y z : ℝ) : ℝ × ℝ × ℝ :=
(x, y, z)

noncomputable def cross_product (v w : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
(v.2.1 * w.2.2 - v.2.2 * w.2.1,
 v.2.2 * w.1 - v.1 * w.2.2,
 v.1 * w.2.1 - v.2.1 * w.1)

noncomputable def magnitude (v : ℝ × ℝ × ℝ) : ℝ :=
Real.sqrt (v.1^2 + v.2.1^2 + v.2.2^2)

theorem triangle_area :
  let A := vector 2 1 (-1)
  let B := vector 3 0 3
  let C := vector 7 3 2
  let AB := (B.1 - A.1, B.2.1 - A.2.1, B.2.2 - A.2.2)
  let AC := (C.1 - A.1, C.2.1 - A.2.1, C.2.2 - A.2.2)
  0.5 * magnitude (cross_product AB AC) = (1 / 2) * Real.sqrt 459 :=
by
  -- All the steps needed to prove the theorem here
  sorry

end triangle_area_l2014_201464


namespace loraine_total_wax_l2014_201427

-- Conditions
def large_animal_wax := 4
def small_animal_wax := 2
def small_animal_count := 12 / small_animal_wax
def large_animal_count := small_animal_count / 3
def total_wax := 12 + (large_animal_count * large_animal_wax)

-- The proof problem
theorem loraine_total_wax : total_wax = 20 := by
  sorry

end loraine_total_wax_l2014_201427


namespace soda_ratio_l2014_201430

theorem soda_ratio (total_sodas diet_sodas regular_sodas : ℕ) (h1 : total_sodas = 64) (h2 : diet_sodas = 28) (h3 : regular_sodas = total_sodas - diet_sodas) : regular_sodas / Nat.gcd regular_sodas diet_sodas = 9 ∧ diet_sodas / Nat.gcd regular_sodas diet_sodas = 7 :=
by
  sorry

end soda_ratio_l2014_201430


namespace contingency_table_proof_l2014_201481

noncomputable def probability_of_mistake (K_squared : ℝ) : ℝ :=
if K_squared > 3.841 then 0.05 else 1.0 -- placeholder definition to be refined

theorem contingency_table_proof :
  probability_of_mistake 4.013 ≤ 0.05 :=
by sorry

end contingency_table_proof_l2014_201481


namespace arc_length_150_deg_max_area_sector_l2014_201493

noncomputable def alpha := 150 * (Real.pi / 180)
noncomputable def r := 6
noncomputable def perimeter := 24

-- 1. Proving the arc length when α = 150° and r = 6
theorem arc_length_150_deg : alpha * r = 5 * Real.pi := by
  sorry

-- 2. Proving the maximum area and corresponding alpha given the perimeter of 24
theorem max_area_sector : ∃ (α : ℝ), α = 2 ∧ (1 / 2) * ((perimeter - 2 * r) * r) = 36 := by
  sorry

end arc_length_150_deg_max_area_sector_l2014_201493


namespace value_of_x_after_z_doubled_l2014_201475

theorem value_of_x_after_z_doubled (x y z : ℕ) (hz : z = 48) (hz_d : z_d = 2 * z) (hy : y = z / 4) (hx : x = y / 3) :
  x = 8 := by
  -- Proof goes here (skipped as instructed)
  sorry

end value_of_x_after_z_doubled_l2014_201475


namespace perimeter_of_sector_l2014_201456

theorem perimeter_of_sector (r : ℝ) (area : ℝ) (perimeter : ℝ) 
  (hr : r = 1) (ha : area = π / 3) : perimeter = (2 * π / 3) + 2 :=
by
  -- You can start the proof here
  sorry

end perimeter_of_sector_l2014_201456


namespace lowest_temperature_l2014_201434

-- Define the temperatures in the four cities.
def temp_Harbin := -20
def temp_Beijing := -10
def temp_Hangzhou := 0
def temp_Jinhua := 2

-- The proof statement asserting the lowest temperature.
theorem lowest_temperature :
  min temp_Harbin (min temp_Beijing (min temp_Hangzhou temp_Jinhua)) = -20 :=
by
  -- Proof omitted
  sorry

end lowest_temperature_l2014_201434


namespace index_card_area_l2014_201421

theorem index_card_area 
  (a b : ℕ)
  (ha : a = 5)
  (hb : b = 7)
  (harea : (a - 2) * b = 21) :
  (a * (b - 2) = 25) :=
by
  sorry

end index_card_area_l2014_201421


namespace calories_per_serving_is_120_l2014_201484

-- Define the conditions
def servings : ℕ := 3
def halfCalories : ℕ := 180
def totalCalories : ℕ := 2 * halfCalories

-- Define the target value
def caloriesPerServing : ℕ := totalCalories / servings

-- The proof goal
theorem calories_per_serving_is_120 : caloriesPerServing = 120 :=
by 
  sorry

end calories_per_serving_is_120_l2014_201484


namespace correct_product_of_a_and_b_l2014_201425

theorem correct_product_of_a_and_b
    (a b : ℕ)
    (h_a_pos : 0 < a)
    (h_b_pos : 0 < b)
    (h_a_two_digits : 10 ≤ a ∧ a < 100)
    (a' : ℕ)
    (h_a' : a' = (a % 10) * 10 + (a / 10))
    (h_product_erroneous : a' * b = 198) :
  a * b = 198 :=
sorry

end correct_product_of_a_and_b_l2014_201425


namespace ploughing_problem_l2014_201499

theorem ploughing_problem
  (hours_per_day_group1 : ℕ)
  (days_group1 : ℕ)
  (bulls_group1 : ℕ)
  (total_fields_group2 : ℕ)
  (hours_per_day_group2 : ℕ)
  (days_group2 : ℕ)
  (bulls_group2 : ℕ)
  (fields_group1 : ℕ)
  (fields_group2 : ℕ) :
    hours_per_day_group1 = 10 →
    days_group1 = 3 →
    bulls_group1 = 10 →
    hours_per_day_group2 = 8 →
    days_group2 = 2 →
    bulls_group2 = 30 →
    fields_group2 = 32 →
    480 * fields_group1 = 300 * fields_group2 →
    fields_group1 = 20 := by
  sorry

end ploughing_problem_l2014_201499


namespace maximum_x_plus_7y_exists_Q_locus_l2014_201468

noncomputable def Q_locus (x y : ℝ) : Prop :=
  (x - 1)^2 + (y - 1)^2 = 2

theorem maximum_x_plus_7y (M : ℝ × ℝ) (h : Q_locus M.fst M.snd) : 
  ∃ max_value, max_value = 18 :=
  sorry

theorem exists_Q_locus (x y : ℝ) : 
  (∃ (Q : ℝ × ℝ), Q_locus Q.fst Q.snd) :=
  sorry

end maximum_x_plus_7y_exists_Q_locus_l2014_201468


namespace two_legged_birds_count_l2014_201486

-- Definitions and conditions
variables {x y z : ℕ}
variables (heads_eq : x + y + z = 200) (legs_eq : 2 * x + 3 * y + 4 * z = 558)

-- The statement to prove
theorem two_legged_birds_count : x = 94 :=
sorry

end two_legged_birds_count_l2014_201486


namespace steve_speed_back_home_l2014_201496

-- Definitions based on conditions
def distance := 20 -- distance from house to work in km
def total_time := 6 -- total time on the road in hours
def speed_to_work (v : ℝ) := v -- speed to work in km/h
def speed_back_home (v : ℝ) := 2 * v -- speed back home in km/h

-- Theorem to assert the proof
theorem steve_speed_back_home (v : ℝ) (h : distance / v + distance / (2 * v) = total_time) :
  speed_back_home v = 10 := by
  -- Proof goes here but we just state sorry to skip it
  sorry

end steve_speed_back_home_l2014_201496


namespace inequality_proof_l2014_201409

variables {a b c : ℝ}

theorem inequality_proof (h1 : a > b) (h2 : b > c) (h3 : a + b + c = 0) : a * c < b * c :=
by
  sorry

end inequality_proof_l2014_201409


namespace probability_dice_sum_perfect_square_l2014_201429

def is_perfect_square (n : ℕ) : Prop :=
  n = 4 ∨ n = 9 ∨ n = 16

noncomputable def probability_perfect_square : ℚ :=
  12 / 64

theorem probability_dice_sum_perfect_square :
  -- Two standard 8-sided dice are rolled
  -- The probability that the sum rolled is a perfect square is 3/16
  (∃ dice1 dice2 : ℕ, 1 ≤ dice1 ∧ dice1 ≤ 8 ∧ 1 ≤ dice2 ∧ dice2 ≤ 8) →
  probability_perfect_square = 3 / 16 :=
sorry

end probability_dice_sum_perfect_square_l2014_201429


namespace isosceles_triangle_area_l2014_201462

theorem isosceles_triangle_area (a b h : ℝ) (h_eq : h = a / (2 * Real.sqrt 3)) :
  (1 / 2 * a * h) = (a^2 * Real.sqrt 3) / 12 :=
by
  -- Define the necessary parameters and conditions
  let area := (1 / 2) * a * h
  have h := h_eq
  -- Substitute and prove the calculated area
  sorry

end isosceles_triangle_area_l2014_201462


namespace geometric_sequence_a5_eq_8_l2014_201436

variable (a : ℕ → ℝ)
variable (n : ℕ)

-- Conditions
axiom pos (n : ℕ) : a n > 0
axiom prod_eq (a3 a7 : ℝ) : a 3 * a 7 = 64

-- Statement to prove
theorem geometric_sequence_a5_eq_8
  (pos : ∀ n, a n > 0)
  (prod_eq : a 3 * a 7 = 64) :
  a 5 = 8 :=
sorry

end geometric_sequence_a5_eq_8_l2014_201436


namespace tank_capacity_l2014_201489

theorem tank_capacity (w c: ℚ) (h1: w / c = 1 / 3) (h2: (w + 5) / c = 2 / 5) :
  c = 37.5 := 
by
  sorry

end tank_capacity_l2014_201489


namespace flower_beds_l2014_201485

theorem flower_beds (seeds_per_bed total_seeds flower_beds : ℕ) 
  (h1 : seeds_per_bed = 10) (h2 : total_seeds = 60) : 
  flower_beds = total_seeds / seeds_per_bed := by
  rw [h1, h2]
  sorry

end flower_beds_l2014_201485


namespace find_crew_members_l2014_201420

noncomputable def passengers_initial := 124
noncomputable def passengers_texas := passengers_initial - 58 + 24
noncomputable def passengers_nc := passengers_texas - 47 + 14
noncomputable def total_people_virginia := 67

theorem find_crew_members (passengers_initial passengers_texas passengers_nc total_people_virginia : ℕ) :
  passengers_initial = 124 →
  passengers_texas = passengers_initial - 58 + 24 →
  passengers_nc = passengers_texas - 47 + 14 →
  total_people_virginia = 67 →
  ∃ crew_members : ℕ, total_people_virginia = passengers_nc + crew_members ∧ crew_members = 10 :=
by
  sorry

end find_crew_members_l2014_201420


namespace spheres_max_min_dist_l2014_201401

variable {R_1 R_2 d : ℝ}

noncomputable def max_min_dist (R_1 R_2 d : ℝ) (sep : d > R_1 + R_2) :
  ℝ × ℝ :=
(d + R_1 + R_2, d - R_1 - R_2)

theorem spheres_max_min_dist {R_1 R_2 d : ℝ} (sep : d > R_1 + R_2) :
  max_min_dist R_1 R_2 d sep = (d + R_1 + R_2, d - R_1 - R_2) := by
sorry

end spheres_max_min_dist_l2014_201401


namespace hyperbola_asymptote_l2014_201431

theorem hyperbola_asymptote (a : ℝ) (h : a > 0) (hyp_eq : ∀ x y : ℝ, (x^2) / (a^2) - (y^2) / 81 = 1 → y = 3 * x) : a = 3 := 
by
  sorry

end hyperbola_asymptote_l2014_201431


namespace eggs_in_each_basket_l2014_201492

theorem eggs_in_each_basket :
  ∃ x : ℕ, x ∣ 30 ∧ x ∣ 42 ∧ x ≥ 5 ∧ x = 6 :=
by
  sorry

end eggs_in_each_basket_l2014_201492


namespace original_proposition_false_implies_negation_true_l2014_201437

-- Define the original proposition and its negation
def original_proposition (x y : ℝ) : Prop := (x + y > 0) → (x > 0 ∧ y > 0)
def negation (x y : ℝ) : Prop := ¬ original_proposition x y

-- Theorem statement
theorem original_proposition_false_implies_negation_true (x y : ℝ) : ¬ original_proposition x y → negation x y :=
by
  -- Since ¬ original_proposition x y implies the negation is true
  intro h
  exact h

end original_proposition_false_implies_negation_true_l2014_201437


namespace total_fiscal_revenue_scientific_notation_l2014_201454

theorem total_fiscal_revenue_scientific_notation : 
  ∃ a n, (1073 * 10^8 : ℝ) = a * 10^n ∧ (1 ≤ |a| ∧ |a| < 10) ∧ a = 1.07 ∧ n = 11 :=
by
  use 1.07, 11
  simp
  sorry

end total_fiscal_revenue_scientific_notation_l2014_201454


namespace contrapositive_of_a_gt_1_then_a_sq_gt_1_l2014_201433

theorem contrapositive_of_a_gt_1_then_a_sq_gt_1 : 
  (∀ a : ℝ, a > 1 → a^2 > 1) → (∀ a : ℝ, a^2 ≤ 1 → a ≤ 1) :=
by 
  sorry

end contrapositive_of_a_gt_1_then_a_sq_gt_1_l2014_201433


namespace mixed_groups_count_l2014_201473

-- Defining the conditions
def total_children : ℕ := 300
def groups_count : ℕ := 100
def group_size : ℕ := 3
def photographs_per_group : ℕ := group_size
def total_photographs : ℕ := groups_count * photographs_per_group
def boy_boy_photos : ℕ := 100
def girl_girl_photos : ℕ := 56
def mixed_photos : ℕ := total_photographs - boy_boy_photos - girl_girl_photos
def mixed_groups : ℕ := mixed_photos / 2

theorem mixed_groups_count : mixed_groups = 72 := by
  -- skipping the proof
  sorry

end mixed_groups_count_l2014_201473


namespace fraction_arithmetic_proof_l2014_201478

theorem fraction_arithmetic_proof :
  (7 / 6) + (5 / 4) - (3 / 2) = 11 / 12 :=
by sorry

end fraction_arithmetic_proof_l2014_201478


namespace existence_of_epsilon_and_u_l2014_201403

theorem existence_of_epsilon_and_u (n : ℕ) (h : 0 < n) :
  ∀ k ≥ 1, ∃ ε : ℝ, (0 < ε ∧ ε < 1 / k) ∧
  (∀ (a : Fin n → ℝ), (∀ i, 0 < a i) → ∃ u > 0, ∀ i, ε < (u * a i - ⌊u * a i⌋) ∧ (u * a i - ⌊u * a i⌋) < 1 / k) :=
by {
  sorry
}

end existence_of_epsilon_and_u_l2014_201403


namespace correct_factorization_l2014_201455

theorem correct_factorization :
  (∀ a b : ℝ, ¬ (a^2 + b^2 = (a + b) * (a - b))) ∧
  (∀ a : ℝ, ¬ (a^4 - 1 = (a^2 + 1) * (a^2 - 1))) ∧
  (∀ x : ℝ, ¬ (x^2 + 2 * x + 4 = (x + 2)^2)) ∧
  (∀ x : ℝ, x^2 - 3 * x + 2 = (x - 1) * (x - 2)) :=
by
  sorry

end correct_factorization_l2014_201455


namespace range_of_c_value_of_c_given_perimeter_l2014_201419

variables (a b c : ℝ)

-- Question 1: Proving the range of values for c
theorem range_of_c (h1 : a + b = 3 * c - 2) (h2 : a - b = 2 * c - 6) :
  1 < c ∧ c < 6 :=
sorry

-- Question 2: Finding the value of c for a given perimeter
theorem value_of_c_given_perimeter (h1 : a + b = 3 * c - 2) (h2 : a - b = 2 * c - 6) (h3 : a + b + c = 18) :
  c = 5 :=
sorry

end range_of_c_value_of_c_given_perimeter_l2014_201419


namespace median_of_first_fifteen_positive_integers_l2014_201424

theorem median_of_first_fifteen_positive_integers : ∃ m : ℝ, m = 7.5 :=
by
  let seq := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
  let median := (seq[6] + seq[7]) / 2
  use median
  sorry

end median_of_first_fifteen_positive_integers_l2014_201424


namespace chad_total_spend_on_ice_l2014_201459

-- Define the given conditions
def num_people : ℕ := 15
def pounds_per_person : ℕ := 2
def pounds_per_bag : ℕ := 1
def price_per_pack : ℕ := 300 -- Price in cents to avoid floating-point issues
def bags_per_pack : ℕ := 10

-- The main statement to prove
theorem chad_total_spend_on_ice : 
  (num_people * pounds_per_person * 100 / (pounds_per_bag * bags_per_pack) * price_per_pack / 100 = 9) :=
by sorry

end chad_total_spend_on_ice_l2014_201459


namespace complex_proof_problem_l2014_201488

theorem complex_proof_problem (i : ℂ) (h1 : i^2 = -1) :
  (i^2 + i^3 + i^4) / (1 - i) = (1 / 2) - (1 / 2) * i :=
by
  -- Proof will be provided here
  sorry

end complex_proof_problem_l2014_201488


namespace num_people_is_8_l2014_201498

-- Define the known conditions
def bill_amt : ℝ := 314.16
def person_amt : ℝ := 34.91
def total_amt : ℝ := 314.19

-- Prove that the number of people is 8
theorem num_people_is_8 : ∃ num_people : ℕ, num_people = total_amt / person_amt ∧ num_people = 8 :=
by
  sorry

end num_people_is_8_l2014_201498


namespace absolute_difference_probability_l2014_201444

-- Define the conditions
def num_red_marbles : ℕ := 1500
def num_black_marbles : ℕ := 2000
def total_marbles : ℕ := num_red_marbles + num_black_marbles

def P_s : ℚ :=
  let ways_to_choose_2_red := (num_red_marbles * (num_red_marbles - 1)) / 2
  let ways_to_choose_2_black := (num_black_marbles * (num_black_marbles - 1)) / 2
  let total_favorable_outcomes := ways_to_choose_2_red + ways_to_choose_2_black
  total_favorable_outcomes / (total_marbles * (total_marbles - 1) / 2)

def P_d : ℚ :=
  (num_red_marbles * num_black_marbles) / (total_marbles * (total_marbles - 1) / 2)

-- Prove the statement
theorem absolute_difference_probability : |P_s - P_d| = 1 / 50 := by
  sorry

end absolute_difference_probability_l2014_201444


namespace fractional_part_frustum_l2014_201432

noncomputable def base_edge : ℝ := 24
noncomputable def original_altitude : ℝ := 18
noncomputable def smaller_altitude : ℝ := original_altitude / 3

noncomputable def volume_pyramid (base_edge : ℝ) (altitude : ℝ) : ℝ :=
  (1 / 3) * (base_edge ^ 2) * altitude

noncomputable def volume_original : ℝ := volume_pyramid base_edge original_altitude
noncomputable def similarity_ratio : ℝ := (smaller_altitude / original_altitude) ^ 3
noncomputable def volume_smaller : ℝ := similarity_ratio * volume_original
noncomputable def volume_frustum : ℝ := volume_original - volume_smaller

noncomputable def fractional_volume_frustum : ℝ := volume_frustum / volume_original

theorem fractional_part_frustum : fractional_volume_frustum = 26 / 27 := by
  sorry

end fractional_part_frustum_l2014_201432


namespace tangent_line_to_ellipse_l2014_201415

variable (a b x y x₀ y₀ : ℝ)

-- Definitions
def is_ellipse (x y a b : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

def point_on_ellipse (x₀ y₀ a b : ℝ) : Prop :=
  x₀^2 / a^2 + y₀^2 / b^2 = 1

-- Theorem
theorem tangent_line_to_ellipse
  (h₁ : point_on_ellipse x₀ y₀ a b) :
    (x₀ * x) / (a^2) + (y₀ * y) / (b^2) = 1 :=
sorry

end tangent_line_to_ellipse_l2014_201415


namespace simplify_expression_l2014_201447

variable {R : Type*} [CommRing R]

theorem simplify_expression (x y : R) : (x + 2 * y) * (x - 2 * y) - y * (3 - 4 * y) = x^2 - 3 * y :=
by
  sorry

end simplify_expression_l2014_201447


namespace smallest_of_six_consecutive_even_numbers_l2014_201491

theorem smallest_of_six_consecutive_even_numbers (h : ∃ n : ℤ, (n - 4) + (n - 2) + n + (n + 2) + (n + 4) + (n + 6) = 390) : ∃ m : ℤ, m = 60 :=
by
  have ex : ∃ n : ℤ, 6 * n + 6 = 390 := by sorry
  obtain ⟨n, hn⟩ := ex
  use (n - 4)
  sorry

end smallest_of_six_consecutive_even_numbers_l2014_201491


namespace ratio_fraction_l2014_201487

variable (X Y Z : ℝ)
variable (k : ℝ) (hk : k > 0)

-- Given conditions
def ratio_condition := (3 * Y = 2 * X) ∧ (6 * Y = 2 * Z)

-- Statement
theorem ratio_fraction (h : ratio_condition X Y Z) : 
  (2 * X + 3 * Y) / (5 * Z - 2 * X) = 1 / 2 := by
  sorry

end ratio_fraction_l2014_201487


namespace neither_plaid_nor_purple_l2014_201453

-- Definitions and given conditions:
def total_shirts := 5
def total_pants := 24
def plaid_shirts := 3
def purple_pants := 5

-- Proof statement:
theorem neither_plaid_nor_purple : 
  (total_shirts - plaid_shirts) + (total_pants - purple_pants) = 21 := 
by 
  -- Mark proof steps with sorry
  sorry

end neither_plaid_nor_purple_l2014_201453


namespace superchess_no_attacks_l2014_201495

open Finset

theorem superchess_no_attacks (board_size : ℕ) (num_pieces : ℕ)  (attack_limit : ℕ) 
  (h_board_size : board_size = 100) (h_num_pieces : num_pieces = 20) 
  (h_attack_limit : attack_limit = 20) : 
  ∃ (placements : Finset (ℕ × ℕ)), placements.card = num_pieces ∧
  ∀ {p1 p2 : ℕ × ℕ}, p1 ≠ p2 → p1 ∈ placements → p2 ∈ placements → 
  ¬(∃ (attack_positions : Finset (ℕ × ℕ)), attack_positions.card ≤ attack_limit ∧ 
  ∃ piece_pos : ℕ × ℕ, piece_pos ∈ placements ∧ attack_positions ⊆ placements ∧ p1 ∈ attack_positions ∧ p2 ∈ attack_positions) :=
sorry

end superchess_no_attacks_l2014_201495


namespace remaining_customers_after_some_left_l2014_201412

-- Define the initial conditions and question (before proving it)
def initial_customers := 8
def new_customers := 99
def total_customers_after_new := 104

-- Define the hypothesis based on the total customers after new customers added
theorem remaining_customers_after_some_left (x : ℕ) (h : x + new_customers = total_customers_after_new) : x = 5 :=
by {
  -- Proof omitted
  sorry
}

end remaining_customers_after_some_left_l2014_201412


namespace mean_absolute_temperature_correct_l2014_201400

noncomputable def mean_absolute_temperature (temps : List ℝ) : ℝ :=
  (temps.map (λ x => |x|)).sum / temps.length

theorem mean_absolute_temperature_correct :
  mean_absolute_temperature [-6, -3, -3, -6, 0, 4, 3] = 25 / 7 :=
by
  sorry

end mean_absolute_temperature_correct_l2014_201400


namespace reciprocal_of_subtraction_l2014_201442

-- Defining the conditions
def x : ℚ := 1 / 9
def y : ℚ := 2 / 3

-- Defining the main theorem statement
theorem reciprocal_of_subtraction : (1 / (y - x)) = 9 / 5 :=
by
  sorry

end reciprocal_of_subtraction_l2014_201442


namespace least_positive_integer_to_multiple_of_5_l2014_201482

theorem least_positive_integer_to_multiple_of_5 : ∃ (n : ℕ), n > 0 ∧ (725 + n) % 5 = 0 ∧ ∀ m : ℕ, m > 0 ∧ (725 + m) % 5 = 0 → n ≤ m :=
by
  sorry

end least_positive_integer_to_multiple_of_5_l2014_201482


namespace find_number_l2014_201435

theorem find_number (x : ℚ) (h : 0.5 * x = (3/5) * x - 10) : x = 100 := 
sorry

end find_number_l2014_201435


namespace miles_walked_on_Tuesday_l2014_201463

theorem miles_walked_on_Tuesday (monday_miles total_miles : ℕ) (hmonday : monday_miles = 9) (htotal : total_miles = 18) :
  total_miles - monday_miles = 9 :=
by
  sorry

end miles_walked_on_Tuesday_l2014_201463


namespace denise_removed_bananas_l2014_201416

theorem denise_removed_bananas (initial_bananas remaining_bananas : ℕ) 
  (h_initial : initial_bananas = 46) (h_remaining : remaining_bananas = 41) : 
  initial_bananas - remaining_bananas = 5 :=
by
  sorry

end denise_removed_bananas_l2014_201416


namespace wholesale_cost_is_200_l2014_201457

variable (W R E : ℝ)

def retail_price (W : ℝ) : ℝ := 1.20 * W

def employee_price (R : ℝ) : ℝ := 0.75 * R

-- Main theorem stating that given the retail and employee price formulas and the employee paid amount,
-- the wholesale cost W is equal to 200.
theorem wholesale_cost_is_200
  (hR : R = retail_price W)
  (hE : E = employee_price R)
  (heq : E = 180) :
  W = 200 :=
by
  sorry

end wholesale_cost_is_200_l2014_201457


namespace qiuqiu_servings_l2014_201439

-- Define the volume metrics
def bottles : ℕ := 1
def cups_per_bottle_kangkang : ℕ := 4
def foam_expansion : ℕ := 3
def foam_fraction : ℚ := 1 / 2

-- Calculate the effective cup volume under Qiuqiu's serving method
def beer_fraction_per_cup_qiuqiu : ℚ := 1 / 2 + (1 / foam_expansion) * foam_fraction

-- Calculate the number of cups Qiuqiu can serve from one bottle
def qiuqiu_cups_from_bottle : ℚ := cups_per_bottle_kangkang / beer_fraction_per_cup_qiuqiu

-- The theorem statement
theorem qiuqiu_servings :
  qiuqiu_cups_from_bottle = 6 := by
  sorry

end qiuqiu_servings_l2014_201439


namespace product_of_consecutive_integers_l2014_201410

theorem product_of_consecutive_integers (n : ℕ) : 24 ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
sorry

end product_of_consecutive_integers_l2014_201410


namespace area_bounded_region_l2014_201465

theorem area_bounded_region :
  (∃ (x y : ℝ), y^2 + 2 * x * y + 50 * |x| = 500) →
  ∃ (area : ℝ), area = 1250 :=
by
  sorry

end area_bounded_region_l2014_201465


namespace coplanar_lines_l2014_201483

noncomputable def line1 (s k : ℝ) : ℝ × ℝ × ℝ :=
  (2 + s, 5 - k * s, 3 + k * s)

noncomputable def line2 (t : ℝ) : ℝ × ℝ × ℝ :=
  (2 * t, 4 + 2 * t, 6 - 2 * t)

theorem coplanar_lines (k : ℝ) :
  (exists s t : ℝ, line1 s k = line2 t) ∨ line1 1 k = (1, -k, k) ∧ line2 1 = (2, 2, -2) → k = -1 :=
by sorry

end coplanar_lines_l2014_201483


namespace right_triangle_ratio_l2014_201428

noncomputable def hypotenuse (a b : ℝ) : ℝ := Real.sqrt (a^2 + b^2)

noncomputable def radius_circumscribed_circle (hypo : ℝ) : ℝ := hypo / 2

noncomputable def radius_inscribed_circle (a b hypo : ℝ) : ℝ := (a + b - hypo) / 2

theorem right_triangle_ratio 
  (a b : ℝ) 
  (ha : a = 6) 
  (hb : b = 8) 
  (right_angle : a^2 + b^2 = hypotenuse a b ^ 2) :
  radius_inscribed_circle a b (hypotenuse a b) / radius_circumscribed_circle (hypotenuse a b) = 2 / 5 :=
by {
  -- The proof to be filled in
  sorry
}

end right_triangle_ratio_l2014_201428


namespace boy_current_age_l2014_201472

theorem boy_current_age (x : ℕ) (h : 5 ≤ x) (age_statement : x = 2 * (x - 5)) : x = 10 :=
by
  sorry

end boy_current_age_l2014_201472


namespace min_rubles_for_1001_l2014_201470

def min_rubles_needed (n : ℕ) : ℕ :=
  let side_cells := (n + 1) * 4
  let inner_cells := (n - 1) * (n - 1)
  let total := inner_cells * 4 + side_cells
  total / 2 -- since each side is shared by two cells

theorem min_rubles_for_1001 : min_rubles_needed 1001 = 503000 := by
  sorry

end min_rubles_for_1001_l2014_201470


namespace vector_orthogonality_solution_l2014_201479

theorem vector_orthogonality_solution :
  let a := (3, -2)
  let b := (x, 1)
  (a.1 * b.1 + a.2 * b.2 = 0) →
  x = 2 / 3 :=
by
  intro h
  sorry

end vector_orthogonality_solution_l2014_201479


namespace lcm_inequality_l2014_201414

theorem lcm_inequality (m n : ℕ) (h : n > m) :
  Nat.lcm m n + Nat.lcm (m + 1) (n + 1) ≥ 2 * m * n / Real.sqrt (n - m) := 
sorry

end lcm_inequality_l2014_201414


namespace fourth_grade_planted_89_l2014_201494

-- Define the number of trees planted by the fifth grade
def fifth_grade_trees : Nat := 114

-- Define the condition that the fifth grade planted twice as many trees as the third grade
def third_grade_trees : Nat := fifth_grade_trees / 2

-- Define the condition that the fourth grade planted 32 more trees than the third grade
def fourth_grade_trees : Nat := third_grade_trees + 32

-- Theorem to prove the number of trees planted by the fourth grade is 89
theorem fourth_grade_planted_89 : fourth_grade_trees = 89 := by
  sorry

end fourth_grade_planted_89_l2014_201494


namespace problem1_problem2_l2014_201448

def f (x b : ℝ) : ℝ := |x - b| + |x + b|

theorem problem1 (x : ℝ) : (∀ y, y = 1 → f x y ≤ x + 2) ↔ (0 ≤ x ∧ x ≤ 2) :=
sorry

theorem problem2 (a b : ℝ) (h : a ≠ 0) : (∀ y, y = 1 → f y b ≥ (|a + 1| - |2 * a - 1|) / |a|) ↔ (b ≤ -3 / 2 ∨ b ≥ 3 / 2) :=
sorry

end problem1_problem2_l2014_201448


namespace abc_cube_geq_abc_sum_l2014_201438

theorem abc_cube_geq_abc_sum (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  (a ^ a * b ^ b * c ^ c) ^ 3 ≥ (a * b * c) ^ (a + b + c) :=
by
  sorry

end abc_cube_geq_abc_sum_l2014_201438


namespace parabola_point_distance_to_focus_l2014_201440

theorem parabola_point_distance_to_focus :
  ∀ (x y : ℝ), (y^2 = 12 * x) → (∃ (xf : ℝ), xf = 3 ∧ 0 ≤ y) → (∃ (d : ℝ), d = 7) → x = 4 :=
by
  intros x y parabola_focus distance_to_focus distance
  sorry

end parabola_point_distance_to_focus_l2014_201440


namespace matthew_hotdogs_needed_l2014_201467

def total_hotdogs (ella_hotdogs emma_hotdogs : ℕ) (luke_multiplier hunter_multiplier : ℕ) : ℕ :=
  let total_sisters := ella_hotdogs + emma_hotdogs
  let luke := luke_multiplier * total_sisters
  let hunter := hunter_multiplier * total_sisters / 2  -- because 1.5 = 3/2 and hunter_multiplier = 3
  total_sisters + luke + hunter

theorem matthew_hotdogs_needed :
  total_hotdogs 2 2 2 3 = 18 := 
by
  -- This proof is correct given the calculations above
  sorry

end matthew_hotdogs_needed_l2014_201467


namespace factor_difference_of_cubes_l2014_201422

theorem factor_difference_of_cubes (t : ℝ) : 
  t^3 - 125 = (t - 5) * (t^2 + 5 * t + 25) :=
sorry

end factor_difference_of_cubes_l2014_201422


namespace proof_smallest_lcm_1_to_12_l2014_201406

noncomputable def smallest_lcm_1_to_12 : ℕ := 27720

theorem proof_smallest_lcm_1_to_12 :
  ∀ n : ℕ, (∀ i : ℕ, 1 ≤ i ∧ i ≤ 12 → i ∣ n) ↔ n = 27720 :=
by
  sorry

end proof_smallest_lcm_1_to_12_l2014_201406


namespace probability_at_least_75_cents_l2014_201461

theorem probability_at_least_75_cents (p n d q c50 : Prop) 
  (Hp : p = tt ∨ p = ff)
  (Hn : n = tt ∨ n = ff)
  (Hd : d = tt ∨ d = ff)
  (Hq : q = tt ∨ q = ff)
  (Hc50 : c50 = tt ∨ c50 = ff) :
  (1 / 2 : ℝ) = 
  ((if c50 = tt then (if q = tt then 1 else 0) else 0) + 
  (if c50 = tt then 2^3 else 0)) / 2^5 :=
by sorry

end probability_at_least_75_cents_l2014_201461


namespace leftover_space_desks_bookcases_l2014_201441

theorem leftover_space_desks_bookcases 
  (number_of_desks : ℕ) (number_of_bookcases : ℕ)
  (wall_length : ℝ) (desk_length : ℝ) (bookcase_length : ℝ) (space_between : ℝ)
  (equal_number : number_of_desks = number_of_bookcases)
  (wall_length_eq : wall_length = 15)
  (desk_length_eq : desk_length = 2)
  (bookcase_length_eq : bookcase_length = 1.5)
  (space_between_eq : space_between = 0.5) :
  ∃ k : ℝ, k = 3 := 
by
  sorry

end leftover_space_desks_bookcases_l2014_201441


namespace average_age_after_leaves_is_27_l2014_201411

def average_age_of_remaining_people (initial_avg_age : ℕ) (initial_people_count : ℕ) 
    (age_leave1 : ℕ) (age_leave2 : ℕ) (remaining_people_count : ℕ) : ℕ :=
  let initial_total_age := initial_avg_age * initial_people_count
  let new_total_age := initial_total_age - (age_leave1 + age_leave2)
  new_total_age / remaining_people_count

theorem average_age_after_leaves_is_27 :
  average_age_of_remaining_people 25 6 20 22 4 = 27 :=
by
  -- Proof is skipped
  sorry

end average_age_after_leaves_is_27_l2014_201411


namespace probability_of_5_blue_marbles_l2014_201474

/--
Jane has a bag containing 9 blue marbles and 6 red marbles. 
She draws a marble, records its color, returns it to the bag, and repeats this process 8 times. 
We aim to prove that the probability that she draws exactly 5 blue marbles is \(0.279\).
-/
theorem probability_of_5_blue_marbles :
  let blue_probability := 9 / 15 
  let red_probability := 6 / 15
  let single_combination_prob := (blue_probability^5) * (red_probability^3)
  let combinations := (Nat.choose 8 5)
  let total_probability := combinations * single_combination_prob
  (Float.round (total_probability.toFloat * 1000) / 1000) = 0.279 :=
by
  sorry

end probability_of_5_blue_marbles_l2014_201474


namespace polynomial_bound_l2014_201449

noncomputable def P (x : ℝ) : ℝ := sorry  -- Placeholder for the polynomial P(x)

theorem polynomial_bound (n : ℕ) (hP : ∀ y : ℝ, 0 ≤ y ∧ y ≤ 1 → |P y| ≤ 1) :
  P (-1 / n) ≤ 2^(n + 1) - 1 :=
sorry

end polynomial_bound_l2014_201449


namespace andrew_friends_brought_food_l2014_201443

theorem andrew_friends_brought_food (slices_per_friend total_slices : ℕ) (h1 : slices_per_friend = 4) (h2 : total_slices = 16) :
  total_slices / slices_per_friend = 4 :=
by
  sorry

end andrew_friends_brought_food_l2014_201443


namespace calculate_k_l2014_201452

variable (A B C D k : ℕ)

def workers_time : Prop :=
  (1 / (A : ℚ) + 1 / (B : ℚ) + 1 / (C : ℚ) + 1 / (D : ℚ)) = 
  (1 / (A - 8 : ℚ)) ∧
  (1 / (A : ℚ) + 1 / (B : ℚ) + 1 / (C : ℚ) + 1 / (D : ℚ)) = 
  (1 / (B - 2 : ℚ)) ∧
  (1 / (A : ℚ) + 1 / (B : ℚ) + 1 / (C : ℚ) + 1 / (D : ℚ)) = 
  (3 / (C : ℚ))

theorem calculate_k (h : workers_time A B C D) : k = 16 :=
  sorry

end calculate_k_l2014_201452


namespace sin_double_angle_plus_pi_div_two_l2014_201480

open Real

theorem sin_double_angle_plus_pi_div_two (θ : ℝ) (h₀ : 0 < θ) (h₁ : θ < π / 2) (h₂ : sin θ = 1 / 3) :
  sin (2 * θ + π / 2) = 7 / 9 :=
by
  sorry

end sin_double_angle_plus_pi_div_two_l2014_201480


namespace assumption_for_contradiction_l2014_201477

theorem assumption_for_contradiction (a b : ℕ) (a_pos : 0 < a) (b_pos : 0 < b) (h : 5 ∣ a * b) : 
  ¬ (¬ (5 ∣ a) ∧ ¬ (5 ∣ b)) := 
sorry

end assumption_for_contradiction_l2014_201477


namespace tile_ratio_l2014_201423

/-- Given the initial configuration and extension method, the ratio of black tiles to white tiles in the new design is 22/27. -/
theorem tile_ratio (initial_black : ℕ) (initial_white : ℕ) (border_black : ℕ) (border_white : ℕ) (total_tiles : ℕ)
  (h1 : initial_black = 10)
  (h2 : initial_white = 15)
  (h3 : border_black = 12)
  (h4 : border_white = 12)
  (h5 : total_tiles = 49) :
  (initial_black + border_black) / (initial_white + border_white) = 22 / 27 := 
by {
  /- 
     Here we would provide the proof steps if needed.
     This is a theorem stating that the ratio of black to white tiles 
     in the new design is 22 / 27 given the initial conditions.
  -/
  sorry 
}

end tile_ratio_l2014_201423


namespace election_winner_percentage_l2014_201402

theorem election_winner_percentage :
    let votes_candidate1 := 2500
    let votes_candidate2 := 5000
    let votes_candidate3 := 15000
    let total_votes := votes_candidate1 + votes_candidate2 + votes_candidate3
    let winning_votes := votes_candidate3
    (winning_votes / total_votes) * 100 = 75 := 
by 
    sorry

end election_winner_percentage_l2014_201402


namespace repaved_before_today_l2014_201490

variable (total_repaved today_repaved : ℕ)

theorem repaved_before_today (h1 : total_repaved = 4938) (h2 : today_repaved = 805) :
  total_repaved - today_repaved = 4133 :=
by 
  -- variables are integers and we are performing a subtraction
  sorry

end repaved_before_today_l2014_201490


namespace max_balls_of_clay_l2014_201405

theorem max_balls_of_clay (radius cube_side_length : ℝ) (V_cube : ℝ) (V_ball : ℝ) (num_balls : ℕ) :
  radius = 3 ->
  cube_side_length = 10 ->
  V_cube = cube_side_length ^ 3 ->
  V_ball = (4 / 3) * π * radius ^ 3 ->
  num_balls = ⌊ V_cube / V_ball ⌋ ->
  num_balls = 8 :=
by
  sorry

end max_balls_of_clay_l2014_201405


namespace find_positive_difference_l2014_201426

theorem find_positive_difference 
  (p1 p2 : ℝ × ℝ) (q1 q2 : ℝ × ℝ) 
  (h_p1 : p1 = (0, 8)) (h_p2 : p2 = (4, 0))
  (h_q1 : q1 = (0, 5)) (h_q2 : q2 = (10, 0))
  (y : ℝ) (hy : y = 20) :
  let m_p := (p2.2 - p1.2) / (p2.1 - p1.1)
  let b_p := p1.2 - m_p * p1.1
  let x_p := (y - b_p) / m_p
  let m_q := (q2.2 - q1.2) / (q2.1 - q1.1)
  let b_q := q1.2 - m_q * q1.1
  let x_q := (y - b_q) / m_q
  abs (x_p - x_q) = 24 :=
by
  sorry

end find_positive_difference_l2014_201426


namespace diagonal_BD_eq_diagonal_AD_eq_l2014_201404

structure Point where
  x : ℝ
  y : ℝ

def A : Point := ⟨-1, 2⟩
def C : Point := ⟨5, 4⟩
def line_AB (p : Point) : Prop := p.x - p.y + 3 = 0

theorem diagonal_BD_eq :
  (∃ M : Point, M = ⟨2, 3⟩) →
  (∃ BD : Point → Prop, (BD = fun p => 3*p.x + p.y - 9 = 0)) :=
by
  sorry

theorem diagonal_AD_eq :
  (∃ M : Point, M = ⟨2, 3⟩) →
  (∃ AD : Point → Prop, (AD = fun p => p.x + 7*p.y - 13 = 0)) :=
by
  sorry

end diagonal_BD_eq_diagonal_AD_eq_l2014_201404


namespace triangle_sides_l2014_201418

theorem triangle_sides
  (D E : ℝ) (DE EF : ℝ)
  (h1 : Real.cos (2 * D - E) + Real.sin (D + E) = 2)
  (hDE : DE = 6) :
  EF = 3 :=
by
  -- Proof is omitted
  sorry

end triangle_sides_l2014_201418


namespace solve_problem_l2014_201469

noncomputable def problem_statement (x : ℝ) : Prop :=
  1 + Real.sin x - Real.cos (5 * x) - Real.sin (7 * x) = 2 * Real.cos (3 * x / 2) ^ 2

theorem solve_problem (x : ℝ) :
  problem_statement x ↔
  (∃ k : ℤ, x = (Real.pi / 8) * (2 * k + 1)) ∨ 
  (∃ n : ℤ, x = (Real.pi / 4) * (4 * n - 1)) :=
by
  sorry

end solve_problem_l2014_201469


namespace value_range_of_a_l2014_201407

variable (a : ℝ)
variable (suff_not_necess : ∀ x, x ∈ ({3, a} : Set ℝ) → 2 * x^2 - 5 * x - 3 ≥ 0)

theorem value_range_of_a :
  (a ≤ -1/2 ∨ a > 3) :=
sorry

end value_range_of_a_l2014_201407


namespace problem1_solutionset_problem2_minvalue_l2014_201446

noncomputable def f (x : ℝ) : ℝ := 45 * abs (2 * x - 1)
noncomputable def g (x : ℝ) : ℝ := f x + f (x - 1)

theorem problem1_solutionset :
  {x : ℝ | 0 < x ∧ x < 2 / 3} = {x : ℝ | f x + abs (x + 1) < 2} :=
by
  sorry

theorem problem2_minvalue (a : ℝ) (m n : ℝ) (h : m + n = a ∧ m > 0 ∧ n > 0) :
  a = 2 → (4 / m + 1 / n) ≥ 9 / 2 :=
by
  sorry

end problem1_solutionset_problem2_minvalue_l2014_201446


namespace largest_integral_solution_l2014_201413

theorem largest_integral_solution : ∃ x : ℤ, (1 / 4 < x / 7 ∧ x / 7 < 3 / 5) ∧ ∀ y : ℤ, (1 / 4 < y / 7 ∧ y / 7 < 3 / 5) → y ≤ x := sorry

end largest_integral_solution_l2014_201413


namespace mrs_jensens_preschool_l2014_201471

theorem mrs_jensens_preschool (total_students students_with_both students_with_neither students_with_green_eyes students_with_red_hair : ℕ) 
(h1 : total_students = 40) 
(h2 : students_with_red_hair = 3 * students_with_green_eyes) 
(h3 : students_with_both = 8) 
(h4 : students_with_neither = 4) :
students_with_green_eyes = 12 := 
sorry

end mrs_jensens_preschool_l2014_201471


namespace triangle_inequality_for_f_l2014_201458

noncomputable def f (x m : ℝ) : ℝ :=
  x^3 - 3 * x + m

theorem triangle_inequality_for_f (a b c m : ℝ) (h₀ : 0 ≤ a) (h₁ : a ≤ 2) (h₂ : 0 ≤ b) (h₃ : b ≤ 2) (h₄ : 0 ≤ c) (h₅ : c ≤ 2) 
(h₆ : 6 < m) :
  ∃ u v w, u = f a m ∧ v = f b m ∧ w = f c m ∧ u + v > w ∧ u + w > v ∧ v + w > u := 
sorry

end triangle_inequality_for_f_l2014_201458


namespace mahmoud_gets_at_least_two_heads_l2014_201497

def probability_of_at_least_two_heads := 1 - ((1/2)^5 + 5 * (1/2)^5)

theorem mahmoud_gets_at_least_two_heads (n : ℕ) (hn : n = 5) :
  probability_of_at_least_two_heads = 13 / 16 :=
by
  simp only [probability_of_at_least_two_heads, hn]
  sorry

end mahmoud_gets_at_least_two_heads_l2014_201497
