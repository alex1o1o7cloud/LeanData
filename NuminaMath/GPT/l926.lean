import Mathlib

namespace ambulance_ride_cost_correct_l926_92608

noncomputable def total_bill : ℝ := 18000
noncomputable def medication_percentage : ℝ := 0.35
noncomputable def imaging_percentage : ℝ := 0.15
noncomputable def surgery_percentage : ℝ := 0.25
noncomputable def overnight_stays_percentage : ℝ := 0.10
noncomputable def doctors_fees_percentage : ℝ := 0.05

noncomputable def food_fee : ℝ := 300
noncomputable def consultation_fee : ℝ := 450
noncomputable def physical_therapy_fee : ℝ := 600

noncomputable def medication_cost : ℝ := medication_percentage * total_bill
noncomputable def imaging_cost : ℝ := imaging_percentage * total_bill
noncomputable def surgery_cost : ℝ := surgery_percentage * total_bill
noncomputable def overnight_stays_cost : ℝ := overnight_stays_percentage * total_bill
noncomputable def doctors_fees_cost : ℝ := doctors_fees_percentage * total_bill

noncomputable def percentage_based_costs : ℝ :=
  medication_cost + imaging_cost + surgery_cost + overnight_stays_cost + doctors_fees_cost

noncomputable def fixed_costs : ℝ :=
  food_fee + consultation_fee + physical_therapy_fee

noncomputable def total_known_costs : ℝ :=
  percentage_based_costs + fixed_costs

noncomputable def ambulance_ride_cost : ℝ :=
  total_bill - total_known_costs

theorem ambulance_ride_cost_correct :
  ambulance_ride_cost = 450 := by
  sorry

end ambulance_ride_cost_correct_l926_92608


namespace tan_150_deg_l926_92673

theorem tan_150_deg : Real.tan (150 * Real.pi / 180) = - (Real.sqrt 3) / 3 :=
by
  -- Conditions used for defining the theorem
  -- 1. 150^\circ = 180^\circ - 30^\circ
  -- 2. Coordinates of a point on the unit circle at angle θ are (cos θ, sin θ)
  -- 3. For 30^\circ, (cos 30^\circ, sin 30^\circ) = (√3/2, 1/2)
  -- 4. Reflect the point across the y-axis changes x-coordinate's sign
  -- 5. tan θ = y/x for a point (x, y) on the unit circle

  sorry

end tan_150_deg_l926_92673


namespace root_eq_neg_l926_92607

theorem root_eq_neg {a : ℝ} (h : 3 * a - 9 < 0) : (a - 4) * (a - 5) > 0 :=
by
  sorry

end root_eq_neg_l926_92607


namespace andrea_average_distance_per_day_l926_92606

theorem andrea_average_distance_per_day
  (total_distance : ℕ := 168)
  (fraction_completed : ℚ := 3/7)
  (total_days : ℕ := 6)
  (days_completed : ℕ := 3) :
  (total_distance * (1 - fraction_completed) / (total_days - days_completed)) = 32 :=
by sorry

end andrea_average_distance_per_day_l926_92606


namespace great_wall_scientific_notation_l926_92676

theorem great_wall_scientific_notation :
  6700000 = 6.7 * 10^6 :=
sorry

end great_wall_scientific_notation_l926_92676


namespace trees_planted_l926_92660

def initial_trees : ℕ := 150
def total_trees_after_planting : ℕ := 225

theorem trees_planted (number_of_trees_planted : ℕ) : 
  number_of_trees_planted = total_trees_after_planting - initial_trees → number_of_trees_planted = 75 :=
by 
  sorry

end trees_planted_l926_92660


namespace sin_30_plus_cos_60_l926_92694

-- Define the trigonometric evaluations as conditions
def sin_30_degree := 1 / 2
def cos_60_degree := 1 / 2

-- Lean statement for proving the sum of these values
theorem sin_30_plus_cos_60 : sin_30_degree + cos_60_degree = 1 := by
  sorry

end sin_30_plus_cos_60_l926_92694


namespace max_value_a_plus_b_l926_92622

theorem max_value_a_plus_b
  (a b : ℝ)
  (h1 : 4 * a + 3 * b ≤ 10)
  (h2 : 3 * a + 5 * b ≤ 11) :
  a + b ≤ 156 / 55 :=
sorry

end max_value_a_plus_b_l926_92622


namespace value_of_m_minus_n_over_n_l926_92645

theorem value_of_m_minus_n_over_n (m n : ℚ) (h : (2/3 : ℚ) * m = (5/6 : ℚ) * n) :
  (m - n) / n = 1 / 4 := 
sorry

end value_of_m_minus_n_over_n_l926_92645


namespace triangle_inequality_l926_92667

variable {a b c S n : ℝ}

theorem triangle_inequality (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) 
(habc : a + b > c) (habc' : a + c > b) (habc'' : b + c > a)
(hS : 2 * S = a + b + c) (hn : n ≥ 1) :
  (a^n / (b + c)) + (b^n / (c + a)) + (c^n / (a + b)) ≥ ((2 / 3)^(n - 2)) * S^(n - 1) :=
by
  sorry

end triangle_inequality_l926_92667


namespace February_March_Ratio_l926_92679

theorem February_March_Ratio (J F M : ℕ) (h1 : F = 2 * J) (h2 : M = 8800) (h3 : J + F + M = 12100) : F / M = 1 / 4 :=
by
  sorry

end February_March_Ratio_l926_92679


namespace determine_k_if_even_function_l926_92666

noncomputable def f (x k : ℝ) : ℝ := k * x^2 + (k - 1) * x + 2

theorem determine_k_if_even_function (k : ℝ) (h_even: ∀ x : ℝ, f x k = f (-x) k ) : k = 1 :=
by
  sorry

end determine_k_if_even_function_l926_92666


namespace quadratic_eq_has_distinct_real_roots_l926_92656

theorem quadratic_eq_has_distinct_real_roots (c : ℝ) (h : c = 0) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (x1 ^ 2 - 3 * x1 + c = 0) ∧ (x2 ^ 2 - 3 * x2 + c = 0)) :=
by {
  sorry
}

end quadratic_eq_has_distinct_real_roots_l926_92656


namespace converse_false_l926_92687

variable {a b : ℝ}

theorem converse_false : (¬ (∀ a b : ℝ, (ab = 0 → a = 0))) :=
by
  sorry

end converse_false_l926_92687


namespace time_for_embankments_l926_92684

theorem time_for_embankments (rate : ℚ) (t1 t2 : ℕ) (w1 w2 : ℕ)
    (h1 : w1 = 75) (h2 : w2 = 60) (h3 : t1 = 4)
    (h4 : rate = 1 / (w1 * t1 : ℚ)) 
    (h5 : t2 = 1 / (w2 * rate)) : 
    t1 + t2 = 9 :=
sorry

end time_for_embankments_l926_92684


namespace total_houses_in_neighborhood_l926_92619

-- Definition of the function f
def f (x : ℕ) : ℕ := x^2 + 3*x

-- Given conditions
def x := 40

-- The theorem states that the total number of houses in Mariam's neighborhood is 1760.
theorem total_houses_in_neighborhood : (x + f x) = 1760 :=
by
  sorry

end total_houses_in_neighborhood_l926_92619


namespace parabola_shifted_left_and_down_l926_92675

-- Define the initial parabola
def initial_parabola (x : ℝ) : ℝ :=
  3 * (x - 4) ^ 2 + 3

-- Define the transformation (shift 4 units to the left and 4 units down)
def transformed_parabola (x : ℝ) : ℝ :=
  initial_parabola (x + 4) - 4

-- Prove that after transformation the given parabola becomes y = 3x^2 - 1
theorem parabola_shifted_left_and_down :
  ∀ x : ℝ, transformed_parabola x = 3 * x ^ 2 - 1 := 
by 
  sorry

end parabola_shifted_left_and_down_l926_92675


namespace day_of_week_150th_day_previous_year_l926_92655

theorem day_of_week_150th_day_previous_year (N : ℕ) 
  (h1 : (275 % 7 = 4))  -- Thursday is 4th day of the week if starting from Sunday as 0
  (h2 : (215 % 7 = 4))  -- Similarly, Thursday is 4th day of the week
  : (150 % 7 = 6) :=     -- Proving the 150th day of year N-1 is a Saturday (Saturday as 6th day of the week)
sorry

end day_of_week_150th_day_previous_year_l926_92655


namespace pipes_fill_tank_in_1_5_hours_l926_92650

theorem pipes_fill_tank_in_1_5_hours :
  (1 / 3 + 1 / 9 + 1 / 18 + 1 / 6) = (2 / 3) →
  (1 / (2 / 3)) = (3 / 2) :=
by sorry

end pipes_fill_tank_in_1_5_hours_l926_92650


namespace find_pairs_l926_92669

def digit_sum (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem find_pairs (a b : ℕ) (h_a_pos : 0 < a) (h_b_pos : 0 < b) : 
  (digit_sum (a^(b+1)) = a^b) ↔ 
  ((a = 1) ∨ (a = 3 ∧ b = 2) ∨ (a = 9 ∧ b = 1)) :=
by
  sorry

end find_pairs_l926_92669


namespace symmetric_point_y_axis_l926_92647

-- Define the original point P
def P : ℝ × ℝ := (1, 6)

-- Define the reflection across the y-axis
def reflect_y_axis (point : ℝ × ℝ) : ℝ × ℝ :=
  (-point.fst, point.snd)

-- Define the symmetric point with respect to the y-axis
def symmetric_point := reflect_y_axis P

-- Statement to prove
theorem symmetric_point_y_axis : symmetric_point = (-1, 6) :=
by
  -- Proof omitted
  sorry

end symmetric_point_y_axis_l926_92647


namespace tax_diminished_by_16_percent_l926_92620

variables (T X : ℝ)

-- Condition: The new revenue is 96.6% of the original revenue
def new_revenue_effect : Prop :=
  (1.15 * (T - X) / 100) = (T / 100) * 0.966

-- Target: Prove that X is 16% of T
theorem tax_diminished_by_16_percent (h : new_revenue_effect T X) : X = 0.16 * T :=
sorry

end tax_diminished_by_16_percent_l926_92620


namespace sqrt_11_bounds_l926_92696

theorem sqrt_11_bounds : ∃ a : ℤ, a < Real.sqrt 11 ∧ Real.sqrt 11 < a + 1 ∧ a = 3 := 
by
  sorry

end sqrt_11_bounds_l926_92696


namespace find_s_is_neg4_l926_92630

def g (x s : ℝ) : ℝ := 3 * x^4 + 2 * x^3 - x^2 - 4 * x + s

theorem find_s_is_neg4 : (∃ s : ℝ, g (-1) s = 0) ↔ (s = -4) :=
sorry

end find_s_is_neg4_l926_92630


namespace sought_circle_equation_l926_92668

def circle_passing_through_point (D E F : ℝ) : Prop :=
  ∀ (x y : ℝ), (x = 0) → (y = 2) → x^2 + y^2 + D * x + E * y + F = 0

def chord_lies_on_line (D E F : ℝ) : Prop :=
  (D + 1) / 5 = (E - 2) / 2 ∧ (D + 1) / 5 = (F + 3)

theorem sought_circle_equation :
  ∃ (D E F : ℝ), 
  circle_passing_through_point D E F ∧ 
  chord_lies_on_line D E F ∧
  (D = -6) ∧ (E = 0) ∧ (F = -4) ∧ 
  ∀ x y : ℝ, x^2 + y^2 + D * x + E * y + F = 0 ↔ x^2 + y^2 - 6 * x - 4 = 0 :=
by
  sorry

end sought_circle_equation_l926_92668


namespace total_wheels_in_neighborhood_l926_92632

def cars_in_Jordan_driveway := 2
def wheels_per_car := 4
def spare_wheel := 1
def bikes_with_2_wheels := 3
def wheels_per_bike := 2
def bike_missing_rear_wheel := 1
def bike_with_training_wheel := 2 + 1
def trash_can_wheels := 2
def tricycle_wheels := 3
def wheelchair_main_wheels := 2
def wheelchair_small_wheels := 2
def wagon_wheels := 4
def roller_skates_total_wheels := 4
def roller_skates_missing_wheel := 1

def pickup_truck_wheels := 4
def boat_trailer_wheels := 2
def motorcycle_wheels := 2
def atv_wheels := 4

theorem total_wheels_in_neighborhood :
  (cars_in_Jordan_driveway * wheels_per_car + spare_wheel + bikes_with_2_wheels * wheels_per_bike + bike_missing_rear_wheel + bike_with_training_wheel + trash_can_wheels + tricycle_wheels + wheelchair_main_wheels + wheelchair_small_wheels + wagon_wheels + (roller_skates_total_wheels - roller_skates_missing_wheel)) +
  (pickup_truck_wheels + boat_trailer_wheels + motorcycle_wheels + atv_wheels) = 47 := by
  sorry

end total_wheels_in_neighborhood_l926_92632


namespace robot_paths_from_A_to_B_l926_92665

/-- Define a function that computes the number of distinct paths a robot can take -/
def distinctPaths (A B : ℕ × ℕ) : ℕ := sorry

/-- Proof statement: There are 556 distinct paths from A to B, given the movement conditions -/
theorem robot_paths_from_A_to_B (A B : ℕ × ℕ) (h_move : (A, B) = ((0, 0), (10, 10))) :
  distinctPaths A B = 556 :=
sorry

end robot_paths_from_A_to_B_l926_92665


namespace point_in_fourth_quadrant_l926_92631

def Point : Type := ℤ × ℤ

def in_fourth_quadrant (p : Point) : Prop :=
  p.1 > 0 ∧ p.2 < 0

def A : Point := (-3, 7)
def B : Point := (3, -7)
def C : Point := (3, 7)
def D : Point := (-3, -7)

theorem point_in_fourth_quadrant : in_fourth_quadrant B :=
by {
  -- skipping the proof steps for the purpose of this example
  sorry
}

end point_in_fourth_quadrant_l926_92631


namespace solve_system_of_equations_l926_92693

theorem solve_system_of_equations 
  (x y : ℝ) 
  (h1 : x / 3 - (y + 1) / 2 = 1) 
  (h2 : 4 * x - (2 * y - 5) = 11) : 
  x = 0 ∧ y = -3 :=
  sorry

end solve_system_of_equations_l926_92693


namespace number_of_members_l926_92635

theorem number_of_members (n : ℕ) (h : n * n = 4624) : n = 68 :=
sorry

end number_of_members_l926_92635


namespace mean_proportional_l926_92644

variable (a b c d : ℕ)
variable (x : ℕ)

def is_geometric_mean (a b : ℕ) (x : ℕ) := x = Int.sqrt (a * b)

theorem mean_proportional (h49 : a = 49) (h64 : b = 64) (h81 : d = 81)
  (h_geometric1 : x = 56) (h_geometric2 : c = 72) :
  c = 64 := sorry

end mean_proportional_l926_92644


namespace factors_of_expression_l926_92641

def total_distinct_factors : ℕ :=
  let a := 10
  let b := 3
  let c := 2
  (a + 1) * (b + 1) * (c + 1)

theorem factors_of_expression :
  total_distinct_factors = 132 :=
by 
  -- the proof goes here
  sorry

end factors_of_expression_l926_92641


namespace multiplication_correct_l926_92601

theorem multiplication_correct : 3795421 * 8634.25 = 32774670542.25 := by
  sorry

end multiplication_correct_l926_92601


namespace cubic_roots_c_over_d_l926_92639

theorem cubic_roots_c_over_d (a b c d : ℤ) (h : a ≠ 0)
  (h_roots : ∃ r1 r2 r3, r1 = -1 ∧ r2 = 3 ∧ r3 = 4 ∧ 
              a * r1 * r2 * r3 + b * (r1 * r2 + r2 * r3 + r3 * r1) + c * (r1 + r2 + r3) + d = 0)
  : (c : ℚ) / d = 5 / 12 := 
sorry

end cubic_roots_c_over_d_l926_92639


namespace paul_eats_sandwiches_l926_92692

theorem paul_eats_sandwiches (S : ℕ) (h : (S + 2 * S + 4 * S) * 2 = 28) : S = 2 :=
by
  sorry

end paul_eats_sandwiches_l926_92692


namespace sum_of_edge_lengths_of_truncated_octahedron_prism_l926_92691

-- Define the vertices, edge length, and the assumption of the prism being a truncated octahedron
def prism_vertices : ℕ := 24
def edge_length : ℕ := 5
def truncated_octahedron_edges : ℕ := 36

-- The Lean statement to prove the sum of edge lengths
theorem sum_of_edge_lengths_of_truncated_octahedron_prism :
  prism_vertices = 24 ∧ edge_length = 5 ∧ truncated_octahedron_edges = 36 →
  truncated_octahedron_edges * edge_length = 180 :=
by
  sorry

end sum_of_edge_lengths_of_truncated_octahedron_prism_l926_92691


namespace cos_theta_value_l926_92626

open Real

def a : ℝ × ℝ := (3, -1)
def b : ℝ × ℝ := (2, 0)

noncomputable def cos_theta (u v : ℝ × ℝ) : ℝ :=
  (u.1 * v.1 + u.2 * v.2) / (Real.sqrt (u.1 ^ 2 + u.2 ^ 2) * Real.sqrt (v.1 ^ 2 + v.2 ^ 2))

theorem cos_theta_value :
  cos_theta a b = 3 * Real.sqrt 10 / 10 :=
by
  sorry

end cos_theta_value_l926_92626


namespace length_of_bridge_l926_92670

theorem length_of_bridge (t : ℝ) (s : ℝ) (d : ℝ) : 
  (t = 24 / 60) ∧ (s = 10) ∧ (d = s * t) → d = 4 := by
  sorry

end length_of_bridge_l926_92670


namespace total_amount_paid_l926_92600

theorem total_amount_paid (sales_tax : ℝ) (tax_rate : ℝ) (cost_tax_free_items : ℝ) : 
  sales_tax = 1.28 → tax_rate = 0.08 → cost_tax_free_items = 12.72 → 
  (sales_tax / tax_rate + sales_tax + cost_tax_free_items) = 30.00 :=
by
  intros h1 h2 h3
  -- Proceed with the proof using h1, h2, and h3
  sorry

end total_amount_paid_l926_92600


namespace original_price_sarees_l926_92651

theorem original_price_sarees
  (P : ℝ)
  (h : 0.90 * 0.85 * P = 378.675) :
  P = 495 :=
sorry

end original_price_sarees_l926_92651


namespace each_student_contribution_l926_92623

-- Definitions for conditions in the problem
def numberOfStudents : ℕ := 30
def totalAmount : ℕ := 480
def numberOfFridaysInTwoMonths : ℕ := 8

-- Statement to prove
theorem each_student_contribution (numberOfStudents : ℕ) (totalAmount : ℕ) (numberOfFridaysInTwoMonths : ℕ) : 
  totalAmount / (numberOfFridaysInTwoMonths * numberOfStudents) = 2 := 
by
  sorry

end each_student_contribution_l926_92623


namespace toothpick_removal_l926_92677

noncomputable def removalStrategy : ℕ :=
  let numToothpicks := 60
  let numUpward1Triangles := 22
  let numDownward1Triangles := 14
  let numUpward2Triangles := 4

  -- minimum toothpicks to remove to achieve the goal
  15

theorem toothpick_removal :
  let numToothpicks := 60
  let numUpward1Triangles := 22
  let numDownward1Triangles := 14
  let numUpward2Triangles := 4
  removalStrategy = 15 := by
  sorry

end toothpick_removal_l926_92677


namespace Zenobius_more_descendants_l926_92674

/-- Total number of descendants in King Pafnutius' lineage --/
def descendants_Pafnutius : Nat :=
  2 + 60 * 2 + 20 * 1

/-- Total number of descendants in King Zenobius' lineage --/
def descendants_Zenobius : Nat :=
  4 + 35 * 3 + 35 * 1

theorem Zenobius_more_descendants : descendants_Zenobius > descendants_Pafnutius := by
  sorry

end Zenobius_more_descendants_l926_92674


namespace right_angled_isosceles_triangle_third_side_length_l926_92629

theorem right_angled_isosceles_triangle_third_side_length (a b c : ℝ) (h₀ : a = 50) (h₁ : b = 50) (h₂ : a + b + c = 160) : c = 60 :=
by
  -- TODO: Provide proof
  sorry

end right_angled_isosceles_triangle_third_side_length_l926_92629


namespace max_a2_plus_b2_l926_92614

theorem max_a2_plus_b2 (a b : ℝ) 
  (h : abs (a - 1) + abs (a - 6) + abs (b + 3) + abs (b - 2) = 10) : 
  (a^2 + b^2) ≤ 45 :=
sorry

end max_a2_plus_b2_l926_92614


namespace coal_consumption_rel_l926_92661

variables (Q a x y : ℝ)
variables (h₀ : 0 < x) (h₁ : x < a) (h₂ : Q ≠ 0) (h₃ : a ≠ 0) (h₄ : a - x ≠ 0)

theorem coal_consumption_rel :
  y = Q / (a - x) - Q / a :=
sorry

end coal_consumption_rel_l926_92661


namespace candy_remaining_l926_92604

def initial_candy : ℝ := 1012.5
def talitha_took : ℝ := 283.7
def solomon_took : ℝ := 398.2
def maya_took : ℝ := 197.6

theorem candy_remaining : initial_candy - (talitha_took + solomon_took + maya_took) = 133 := 
by
  sorry

end candy_remaining_l926_92604


namespace quadratic_roots_relation_l926_92688

theorem quadratic_roots_relation (a b c d : ℝ) (h : ∀ x : ℝ, (c * x^2 + d * x + a = 0) → 
  (a * (2007 * x)^2 + b * (2007 * x) + c = 0)) : b^2 = d^2 := 
sorry

end quadratic_roots_relation_l926_92688


namespace intersection_M_N_l926_92685

def M : Set ℤ := {-2, -1, 0, 1, 2}
def N : Set ℤ := {x | x^2 - x - 6 ≥ 0}

theorem intersection_M_N : M ∩ N = {-2} := by
  sorry

end intersection_M_N_l926_92685


namespace work_completion_time_l926_92602

theorem work_completion_time (A_work_rate B_work_rate C_work_rate : ℝ) 
  (hA : A_work_rate = 1 / 8) 
  (hB : B_work_rate = 1 / 16) 
  (hC : C_work_rate = 1 / 16) : 
  1 / (A_work_rate + B_work_rate + C_work_rate) = 4 :=
by
  -- Proof goes here
  sorry

end work_completion_time_l926_92602


namespace slower_speed_is_10_l926_92638

-- Define the problem conditions
def walked_distance (faster_speed slower_speed actual_distance extra_distance : ℕ) : Prop :=
  actual_distance / slower_speed = (actual_distance + extra_distance) / faster_speed

-- Define main statement to prove
theorem slower_speed_is_10 (actual_distance : ℕ) (extra_distance : ℕ) (faster_speed : ℕ) (slower_speed : ℕ) :
  walked_distance faster_speed slower_speed actual_distance extra_distance ∧ 
  faster_speed = 15 ∧ extra_distance = 15 ∧ actual_distance = 30 → slower_speed = 10 :=
by
  intro h
  sorry

end slower_speed_is_10_l926_92638


namespace Ali_money_left_l926_92654

theorem Ali_money_left (initial_money : ℕ) 
  (spent_on_food_ratio : ℚ) 
  (spent_on_glasses_ratio : ℚ) 
  (spent_on_food : ℕ) 
  (left_after_food : ℕ) 
  (spent_on_glasses : ℕ) 
  (final_left : ℕ) :
    initial_money = 480 →
    spent_on_food_ratio = 1 / 2 →
    spent_on_food = initial_money * spent_on_food_ratio →
    left_after_food = initial_money - spent_on_food →
    spent_on_glasses_ratio = 1 / 3 →
    spent_on_glasses = left_after_food * spent_on_glasses_ratio →
    final_left = left_after_food - spent_on_glasses →
    final_left = 160 :=
by
  sorry

end Ali_money_left_l926_92654


namespace lagrange_intermediate_value_l926_92605

open Set

variable {a b : ℝ} (f : ℝ → ℝ)

-- Ensure that a < b for the interval [a, b]
axiom hab : a < b

-- Assume f is differentiable on [a, b]
axiom differentiable_on_I : DifferentiableOn ℝ f (Icc a b)

theorem lagrange_intermediate_value :
  ∃ (x0 : ℝ), x0 ∈ Ioo a b ∧ (deriv f x0) = (f a - f b) / (a - b) :=
sorry

end lagrange_intermediate_value_l926_92605


namespace minimum_distance_l926_92652

section MinimumDistance
open Real

noncomputable def f (x : ℝ) : ℝ := exp x
noncomputable def g (x : ℝ) : ℝ := 2 * sqrt x
def t (x1 x2 : ℝ) := f x1 = g x2
def d (x1 x2 : ℝ) := abs (x2 - x1)

theorem minimum_distance : ∃ (x1 x2 : ℝ), t x1 x2 ∧ d x1 x2 = (1 - log 2) / 2 := 
sorry

end MinimumDistance

end minimum_distance_l926_92652


namespace problem_1_problem_3_problem_4_l926_92658

-- Definition of the function f(x)
def f (x : ℝ) (b c : ℝ) : ℝ := (|x| * x) + (b * x) + c

-- Prove that when b > 0, f(x) is monotonically increasing on ℝ
theorem problem_1 (b c : ℝ) (h : b > 0) : 
  ∀ x y : ℝ, x < y → f x b c < f y b c :=
sorry

-- Prove that the graph of f(x) is symmetric about the point (0, c) when b = 0
theorem problem_3 (b c : ℝ) (h : b = 0) :
  ∀ x : ℝ, f x b c = f (-x) b c :=
sorry

-- Prove that when b < 0, f(x) = 0 can have three real roots
theorem problem_4 (b c : ℝ) (h : b < 0) :
  ∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧ f x₁ b c = 0 ∧ f x₂ b c = 0 ∧ f x₃ b c = 0 :=
sorry

end problem_1_problem_3_problem_4_l926_92658


namespace total_trees_in_park_l926_92690

theorem total_trees_in_park (oak_planted_total maple_planted_total birch_planted_total : ℕ)
  (initial_oak initial_maple initial_birch : ℕ)
  (oak_removed_day2 maple_removed_day2 birch_removed_day2 : ℕ)
  (D1_oak_plant : ℕ) (D2_oak_plant : ℕ) (D1_maple_plant : ℕ) (D2_maple_plant : ℕ)
  (D1_birch_plant : ℕ) (D2_birch_plant : ℕ):
  initial_oak = 25 → initial_maple = 40 → initial_birch = 20 →
  oak_planted_total = 73 → maple_planted_total = 52 → birch_planted_total = 35 →
  D1_oak_plant = 29 → D2_oak_plant = 26 →
  D1_maple_plant = 26 → D2_maple_plant = 13 →
  D1_birch_plant = 10 → D2_birch_plant = 16 →
  oak_removed_day2 = 15 → maple_removed_day2 = 10 → birch_removed_day2 = 5 →
  (initial_oak + oak_planted_total - oak_removed_day2) +
  (initial_maple + maple_planted_total - maple_removed_day2) +
  (initial_birch + birch_planted_total - birch_removed_day2) = 215 :=
by
  intros h_initial_oak h_initial_maple h_initial_birch
         h_oak_planted_total h_maple_planted_total h_birch_planted_total
         h_D1_oak h_D2_oak h_D1_maple h_D2_maple h_D1_birch h_D2_birch
         h_oak_removed h_maple_removed h_birch_removed
  sorry

end total_trees_in_park_l926_92690


namespace no_divisors_in_range_l926_92609

theorem no_divisors_in_range : ¬ ∃ n : ℕ, 80 < n ∧ n < 90 ∧ n ∣ (3^40 - 1) :=
by sorry

end no_divisors_in_range_l926_92609


namespace a7_value_l926_92643

theorem a7_value
  (a : ℕ → ℝ)
  (hx2 : ∀ n, n > 0 → a n ≠ 0)
  (slope_condition : ∀ n, n ≥ 2 → 2 * a n = 2 * a (n - 1) + 1)
  (point_condition : a 1 * 4 = 8) :
  a 7 = 5 :=
by
  sorry

end a7_value_l926_92643


namespace math_problem_l926_92689

theorem math_problem (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x = 1 / y) :
  (x - 1 / x) * (y + 1 / y) = x^2 - y^2 :=
by
  sorry

end math_problem_l926_92689


namespace A_alone_days_l926_92681

theorem A_alone_days (A B C : ℝ) (hB: B = 9) (hC: C = 7.2) 
  (h: 1 / A + 1 / B + 1 / C = 1 / 2) : A = 2 :=
by
  rw [hB, hC] at h
  sorry

end A_alone_days_l926_92681


namespace gcd_360_504_l926_92671

theorem gcd_360_504 : Nat.gcd 360 504 = 72 := by
  sorry

end gcd_360_504_l926_92671


namespace roll_2_four_times_last_not_2_l926_92628

def probability_of_rolling_2_four_times_last_not_2 : ℚ :=
  (1/6)^4 * (5/6)

theorem roll_2_four_times_last_not_2 :
  probability_of_rolling_2_four_times_last_not_2 = 5 / 7776 := 
by
  sorry

end roll_2_four_times_last_not_2_l926_92628


namespace area_of_paper_l926_92683

theorem area_of_paper (L W : ℕ) (h1 : 2 * L + W = 34) (h2 : L + 2 * W = 38) : 
  L * W = 140 := 
by sorry

end area_of_paper_l926_92683


namespace hexagon_can_be_divided_into_congruent_triangles_l926_92621

section hexagon_division

-- Definitions
variables {H : Type} -- H represents the type for hexagon

-- Conditions
variables (is_hexagon : H → Prop) -- A predicate stating that a shape is a hexagon
variables (lies_on_grid : H → Prop) -- A predicate stating that the hexagon lies on the grid
variables (can_cut_along_grid_lines : H → Prop) -- A predicate stating that cuts can only be made along the grid lines
variables (identical_figures : Type u → Prop) -- A predicate stating that the obtained figures must be identical
variables (congruent_triangles : Type u → Prop) -- A predicate stating that the obtained figures are congruent triangles
variables (area_division : H → Prop) -- A predicate stating that the area of the hexagon is divided equally

-- Theorem statement
theorem hexagon_can_be_divided_into_congruent_triangles (h : H)
  (H_is_hexagon : is_hexagon h)
  (H_on_grid : lies_on_grid h)
  (H_cut : can_cut_along_grid_lines h) :
  ∃ (F : Type u), identical_figures F ∧ congruent_triangles F ∧ area_division h :=
sorry

end hexagon_division

end hexagon_can_be_divided_into_congruent_triangles_l926_92621


namespace last_four_digits_of_3_power_24000_l926_92646

theorem last_four_digits_of_3_power_24000 (h : 3^800 ≡ 1 [MOD 2000]) : 3^24000 ≡ 1 [MOD 2000] :=
  by sorry

end last_four_digits_of_3_power_24000_l926_92646


namespace rectangular_field_perimeter_l926_92695

theorem rectangular_field_perimeter
  (a b : ℝ)
  (diag_eq : a^2 + b^2 = 1156)
  (area_eq : a * b = 240)
  (side_relation : a = 2 * b) :
  2 * (a + b) = 91.2 :=
by
  sorry

end rectangular_field_perimeter_l926_92695


namespace tangent_line_at_P_l926_92678

/-- Define the center of the circle as the origin and point P --/
def center : ℝ × ℝ := (0, 0)

def P : ℝ × ℝ := (1, 2)

/-- Define the circle with radius squared r², where the radius passes through point P leading to r² = 5 --/
def circle_equation (x y : ℝ) : Prop := x * x + y * y = 5

/-- Define the condition that point P lies on the circle centered at the origin --/
def P_on_circle : Prop := circle_equation P.1 P.2

/-- Define what it means for a line to be the tangent at point P --/
def tangent_line (x y : ℝ) : Prop := x + 2 * y - 5 = 0

theorem tangent_line_at_P : P_on_circle → ∃ x y, tangent_line x y :=
by {
  sorry
}

end tangent_line_at_P_l926_92678


namespace Ashutosh_time_to_complete_job_l926_92625

noncomputable def SureshWorkRate : ℝ := 1 / 15
noncomputable def AshutoshWorkRate (A : ℝ) : ℝ := 1 / A
noncomputable def SureshWorkIn9Hours : ℝ := 9 * SureshWorkRate

theorem Ashutosh_time_to_complete_job (A : ℝ) :
  (1 - SureshWorkIn9Hours) * AshutoshWorkRate A = 14 / 35 →
  A = 35 :=
by
  sorry

end Ashutosh_time_to_complete_job_l926_92625


namespace find_ratio_l926_92603

-- Definitions
noncomputable def cost_per_gram_A : ℝ := 0.01
noncomputable def cost_per_gram_B : ℝ := 0.008
noncomputable def new_cost_per_gram_A : ℝ := 0.011
noncomputable def new_cost_per_gram_B : ℝ := 0.0072

def total_weight : ℝ := 1000

-- Theorem statement
theorem find_ratio (x y : ℝ) (h1 : x + y = total_weight)
    (h2 : cost_per_gram_A * x + cost_per_gram_B * y = new_cost_per_gram_A * x + new_cost_per_gram_B * y) :
    x / y = 4 / 5 :=
by
  sorry

end find_ratio_l926_92603


namespace power_function_passes_through_1_1_l926_92618

theorem power_function_passes_through_1_1 (a : ℝ) : (1 : ℝ) ^ a = 1 := 
by
  sorry

end power_function_passes_through_1_1_l926_92618


namespace union_of_sets_l926_92610

open Set

noncomputable def A (a : ℝ) : Set ℝ := {1, 2^a}
noncomputable def B (a b : ℝ) : Set ℝ := {a, b}

theorem union_of_sets (a b : ℝ) (h₁ : A a ∩ B a b = {1 / 2}) :
  A a ∪ B a b = {-1, 1 / 2, 1} :=
by
  sorry

end union_of_sets_l926_92610


namespace product_of_divisors_18_l926_92637

-- Definitions
def num := 18
def divisors := [1, 2, 3, 6, 9, 18]

-- The theorem statement
theorem product_of_divisors_18 : 
  (divisors.foldl (·*·) 1) = 104976 := 
by sorry

end product_of_divisors_18_l926_92637


namespace person_birth_date_l926_92664

theorem person_birth_date
  (x : ℕ)
  (h1 : 1937 - x = x^2 - x)
  (d m : ℕ)
  (h2 : 44 + m = d^2)
  (h3 : 0 < m ∧ m < 13)
  (h4 : d = 7 ∧ m = 5) :
  (x = 44 ∧ 1937 - (x + x^2) = 1892) ∧  d = 7 ∧ m = 5 :=
by
  sorry

end person_birth_date_l926_92664


namespace least_product_of_distinct_primes_greater_than_50_l926_92672

def is_prime (n : ℕ) : Prop := ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def distinct_primes_greater_than_50 (p q : ℕ) : Prop :=
  is_prime p ∧ is_prime q ∧ p ≠ q ∧ p > 50 ∧ q > 50

theorem least_product_of_distinct_primes_greater_than_50 :
  ∃ (p q : ℕ), distinct_primes_greater_than_50 p q ∧ p * q = 3127 :=
by
  sorry

end least_product_of_distinct_primes_greater_than_50_l926_92672


namespace root_polynomial_sum_l926_92680

theorem root_polynomial_sum {b c : ℝ} (hb : b^2 - b - 1 = 0) (hc : c^2 - c - 1 = 0) : 
  (1 / (1 - b)) + (1 / (1 - c)) = -1 := 
sorry

end root_polynomial_sum_l926_92680


namespace min_tosses_one_head_l926_92640

theorem min_tosses_one_head (n : ℕ) (P : ℝ) (h₁ : P = 1 - (1 / 2) ^ n) (h₂ : P ≥ 15 / 16) : n ≥ 4 :=
by
  sorry -- Proof to be filled in.

end min_tosses_one_head_l926_92640


namespace intersection_A_B_l926_92612

def A : Set ℝ := {x | x^2 - x = 0}
def B : Set ℝ := {x | x^2 + x = 0}

theorem intersection_A_B : A ∩ B = {0} :=
by
  sorry

end intersection_A_B_l926_92612


namespace complement_of_A_in_S_l926_92659

universe u

def S : Set ℕ := {x | 0 ≤ x ∧ x ≤ 5}
def A : Set ℕ := {x | 1 < x ∧ x < 5}

theorem complement_of_A_in_S : S \ A = {0, 1, 5} := 
by sorry

end complement_of_A_in_S_l926_92659


namespace weighted_average_correct_l926_92636

-- Define the marks and credits for each subject
def marks_english := 90
def marks_mathematics := 92
def marks_physics := 85
def marks_chemistry := 87
def marks_biology := 85

def credits_english := 3
def credits_mathematics := 4
def credits_physics := 4
def credits_chemistry := 3
def credits_biology := 2

-- Define the weighted sum and total credits
def weighted_sum := marks_english * credits_english + marks_mathematics * credits_mathematics + marks_physics * credits_physics + marks_chemistry * credits_chemistry + marks_biology * credits_biology
def total_credits := credits_english + credits_mathematics + credits_physics + credits_chemistry + credits_biology

-- Prove that the weighted average is 88.0625
theorem weighted_average_correct : (weighted_sum.toFloat / total_credits.toFloat) = 88.0625 :=
by 
  sorry

end weighted_average_correct_l926_92636


namespace shopkeeper_profit_percent_l926_92634

theorem shopkeeper_profit_percent (cost_price profit : ℝ) (h1 : cost_price = 960) (h2 : profit = 40) : 
  (profit / cost_price) * 100 = 4.17 :=
by
  sorry

end shopkeeper_profit_percent_l926_92634


namespace simplify_expression_l926_92617

theorem simplify_expression (x : ℝ) : 
  6 * (x - 7) * (2 * x + 15) + (3 * x - 4) * (x + 5) = 15 * x^2 + 17 * x - 650 :=
by
  sorry

end simplify_expression_l926_92617


namespace no_integers_satisfying_polynomials_l926_92698

theorem no_integers_satisfying_polynomials 
: ¬ ∃ (a b c d : ℤ), a * 19^3 + b * 19^2 + c * 19 + d = 1 ∧ a * 62^3 + b * 62^2 + c * 62 + d = 2 := 
by
  sorry

end no_integers_satisfying_polynomials_l926_92698


namespace price_of_case_l926_92633

variables (bottles_per_day : ℚ) (days : ℕ) (bottles_per_case : ℕ) (total_spent : ℚ)

def total_bottles_consumed (bottles_per_day : ℚ) (days : ℕ) : ℚ :=
  bottles_per_day * days

def cases_needed (total_bottles : ℚ) (bottles_per_case : ℕ) : ℚ :=
  total_bottles / bottles_per_case

def price_per_case (total_spent : ℚ) (cases : ℚ) : ℚ :=
  total_spent / cases

theorem price_of_case (h1 : bottles_per_day = 1/2)
                      (h2 : days = 240)
                      (h3 : bottles_per_case = 24)
                      (h4 : total_spent = 60) :
  price_per_case total_spent (cases_needed (total_bottles_consumed bottles_per_day days) bottles_per_case) = 12 := 
sorry

end price_of_case_l926_92633


namespace max_value_ratio_l926_92611

theorem max_value_ratio (a b c: ℝ) (h_pos: a > 0 ∧ b > 0 ∧ c > 0) (h_eq: a * (a + b + c) = b * c) :
  (a / (b + c) ≤ (Real.sqrt 2 - 1) / 2) :=
sorry -- proof omitted

end max_value_ratio_l926_92611


namespace log_10_850_consecutive_integers_l926_92682

theorem log_10_850_consecutive_integers : 
  (2:ℝ) < Real.log 850 / Real.log 10 ∧ Real.log 850 / Real.log 10 < (3:ℝ) →
  ∃ (a b : ℕ), (a = 2) ∧ (b = 3) ∧ (2 < Real.log 850 / Real.log 10) ∧ (Real.log 850 / Real.log 10 < 3) ∧ (a + b = 5) :=
by
  sorry

end log_10_850_consecutive_integers_l926_92682


namespace net_pay_rate_l926_92663

def travelTime := 3 -- hours
def speed := 50 -- miles per hour
def fuelEfficiency := 25 -- miles per gallon
def earningsRate := 0.6 -- dollars per mile
def gasolineCost := 3 -- dollars per gallon

theorem net_pay_rate
  (travelTime : ℕ)
  (speed : ℕ)
  (fuelEfficiency : ℕ)
  (earningsRate : ℚ)
  (gasolineCost : ℚ)
  (h_time : travelTime = 3)
  (h_speed : speed = 50)
  (h_fuelEfficiency : fuelEfficiency = 25)
  (h_earningsRate : earningsRate = 0.6)
  (h_gasolineCost : gasolineCost = 3) :
  (earningsRate * speed * travelTime - (speed * travelTime / fuelEfficiency) * gasolineCost) / travelTime = 24 :=
by
  sorry

end net_pay_rate_l926_92663


namespace trees_occupy_area_l926_92616

theorem trees_occupy_area
  (length : ℕ) (width : ℕ) (number_of_trees : ℕ)
  (h_length : length = 1000)
  (h_width : width = 2000)
  (h_trees : number_of_trees = 100000) :
  (length * width) / number_of_trees = 20 := 
by
  sorry

end trees_occupy_area_l926_92616


namespace jungsoo_number_is_correct_l926_92642

def J := (1 * 4) + (0.1 * 2) + (0.001 * 7)
def Y := 100 * J 
def S := Y + 0.05

theorem jungsoo_number_is_correct : S = 420.75 := by
  sorry

end jungsoo_number_is_correct_l926_92642


namespace mary_no_torn_cards_l926_92649

theorem mary_no_torn_cards
  (T : ℕ) -- number of Mary's initial torn baseball cards
  (initial_cards : ℕ := 18) -- initial baseball cards
  (fred_cards : ℕ := 26) -- baseball cards given by Fred
  (bought_cards : ℕ := 40) -- baseball cards bought
  (total_cards : ℕ := 84) -- total baseball cards Mary has now
  (h : initial_cards - T + fred_cards + bought_cards = total_cards)
  : T = 0 :=
by sorry

end mary_no_torn_cards_l926_92649


namespace train_speed_in_km_hr_l926_92613

noncomputable def train_length : ℝ := 320
noncomputable def crossing_time : ℝ := 7.999360051195905
noncomputable def speed_in_meter_per_sec : ℝ := train_length / crossing_time
noncomputable def meter_per_sec_to_km_hr (speed_mps : ℝ) : ℝ := speed_mps * 3.6
noncomputable def expected_speed : ℝ := 144.018001125

theorem train_speed_in_km_hr :
  meter_per_sec_to_km_hr speed_in_meter_per_sec = expected_speed := by
  sorry

end train_speed_in_km_hr_l926_92613


namespace shortest_chord_length_l926_92624

theorem shortest_chord_length
  (x y : ℝ)
  (hx : x^2 + y^2 - 6 * x - 8 * y = 0)
  (point_on_circle : (3, 5) = (x, y)) :
  ∃ (length : ℝ), length = 4 * Real.sqrt 6 := 
by
  sorry

end shortest_chord_length_l926_92624


namespace eccentricity_of_ellipse_l926_92686
-- Import the Mathlib library for mathematical tools and structures

-- Define the condition for the ellipse and the arithmetic sequence
variables {a b c : ℝ} (h1 : a > b) (h2 : b > 0) (h3 : 2 * b = a + c) (h4 : b^2 = a^2 - c^2)

-- State the theorem to prove
theorem eccentricity_of_ellipse : ∃ e : ℝ, e = 3 / 5 :=
by
  -- Proof would go here
  sorry

end eccentricity_of_ellipse_l926_92686


namespace set_intersection_l926_92615

theorem set_intersection (A B : Set ℝ)
  (hA : A = { x : ℝ | 1 < x ∧ x < 4 })
  (hB : B = { x : ℝ | x^2 - 2 * x - 3 ≤ 0 }) :
  A ∩ (Set.univ \ B) = { x : ℝ | 3 < x ∧ x < 4 } :=
by
  sorry

end set_intersection_l926_92615


namespace penny_paid_amount_l926_92657

-- Definitions based on conditions
def bulk_price : ℕ := 5
def minimum_spend : ℕ := 40
def tax_rate : ℕ := 1
def excess_pounds : ℕ := 32

-- Expression for total calculated cost
def total_pounds := (minimum_spend / bulk_price) + excess_pounds
def cost_before_tax := total_pounds * bulk_price
def total_tax := total_pounds * tax_rate
def total_cost := cost_before_tax + total_tax

-- Required proof statement
theorem penny_paid_amount : total_cost = 240 := 
by 
  sorry

end penny_paid_amount_l926_92657


namespace number_of_triangles_l926_92697

theorem number_of_triangles (x y : ℕ) (P Q : ℕ × ℕ) (O : ℕ × ℕ := (0,0)) (area : ℕ) :
  (P ≠ Q) ∧ (P.1 * 31 + P.2 = 2023) ∧ (Q.1 * 31 + Q.2 = 2023) ∧ 
  (P.1 ≠ Q.1 → P.1 - Q.1 = n ∧ 2023 * n % 6 = 0) → area = 165 :=
sorry

end number_of_triangles_l926_92697


namespace real_solution_count_l926_92627

noncomputable def f (x : ℝ) : ℝ :=
  (1/(x - 1)) + (2/(x - 2)) + (3/(x - 3)) + (4/(x - 4)) + 
  (5/(x - 5)) + (6/(x - 6)) + (7/(x - 7)) + (8/(x - 8)) + 
  (9/(x - 9)) + (10/(x - 10))

theorem real_solution_count : ∃ n : ℕ, n = 11 ∧ 
  ∃ x : ℝ, f x = x :=
sorry

end real_solution_count_l926_92627


namespace difference_of_squares_l926_92648

theorem difference_of_squares (a b : ℕ) (h₁ : a = 69842) (h₂ : b = 30158) :
  (a^2 - b^2) / (a - b) = 100000 :=
by
  rw [h₁, h₂]
  sorry

end difference_of_squares_l926_92648


namespace simplify_expression_l926_92662

theorem simplify_expression :
  (1 / (1 / ((1 / 3)^1) + 1 / ((1 / 3)^2) + 1 / ((1 / 3)^3))) = 1 / 39 :=
by
  sorry

end simplify_expression_l926_92662


namespace seokgi_walk_distance_correct_l926_92653

-- Definitions of distances as per conditions
def entrance_to_temple_km : ℕ := 4
def entrance_to_temple_m : ℕ := 436
def temple_to_summit_m : ℕ := 1999

-- Total distance Seokgi walked in kilometers
def total_walked_km : ℕ := 12870

-- Proof statement
theorem seokgi_walk_distance_correct :
  ((entrance_to_temple_km * 1000 + entrance_to_temple_m) + temple_to_summit_m) * 2 / 1000 = total_walked_km / 1000 :=
by
  -- We will fill this in with the proof steps
  sorry

end seokgi_walk_distance_correct_l926_92653


namespace arithmetic_sequence_a1_l926_92699

/-- In an arithmetic sequence {a_n],
given a_3 = -2, a_n = 3 / 2, and S_n = -15 / 2,
prove that the value of a_1 is -3 or -19 / 6.
-/
theorem arithmetic_sequence_a1 (a_n S_n : ℕ → ℚ)
  (h1 : a_n 3 = -2)
  (h2 : ∃ n : ℕ, a_n n = 3 / 2)
  (h3 : ∃ n : ℕ, S_n n = -15 / 2) :
  ∃ x : ℚ, x = -3 ∨ x = -19 / 6 :=
by 
  sorry

end arithmetic_sequence_a1_l926_92699
