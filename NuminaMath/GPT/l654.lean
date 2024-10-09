import Mathlib

namespace more_children_got_off_than_got_on_l654_65494

-- Define the initial number of children on the bus
def initial_children : ℕ := 36

-- Define the number of children who got off the bus
def children_got_off : ℕ := 68

-- Define the total number of children on the bus after changes
def final_children : ℕ := 12

-- Define the unknown number of children who got on the bus
def children_got_on : ℕ := sorry -- We will use the conditions to solve for this in the proof

-- The main proof statement
theorem more_children_got_off_than_got_on : (children_got_off - children_got_on = 24) :=
by
  -- Write the equation describing the total number of children after changes
  have h1 : initial_children - children_got_off + children_got_on = final_children := sorry
  -- Solve for the number of children who got on the bus (children_got_on)
  have h2 : children_got_on = final_children + (children_got_off - initial_children) := sorry
  -- Substitute to find the required difference
  have h3 : children_got_off - final_children - (children_got_off - initial_children) = 24 := sorry
  -- Conclude the proof
  exact sorry


end more_children_got_off_than_got_on_l654_65494


namespace exponentiation_of_squares_l654_65425

theorem exponentiation_of_squares :
  ((Real.sqrt 2 + 1)^2000 * (Real.sqrt 2 - 1)^2000 = 1) :=
by
  sorry

end exponentiation_of_squares_l654_65425


namespace shortest_side_of_right_triangle_l654_65477

theorem shortest_side_of_right_triangle (a b : ℝ) (ha : a = 5) (hb : b = 12) : 
  ∀ c, (c = 5 ∨ c = 12 ∨ c = (Real.sqrt (a^2 + b^2))) → c = 5 :=
by
  intros c h
  sorry

end shortest_side_of_right_triangle_l654_65477


namespace triangle_external_angle_properties_l654_65456

theorem triangle_external_angle_properties (A B C : ℝ) (hA : 0 < A ∧ A < 180) (hB : 0 < B ∧ B < 180) (hC : 0 < C ∧ C < 180) (h_sum : A + B + C = 180) :
  (∃ E1 E2 E3, E1 + E2 + E3 = 360 ∧ E1 > 90 ∧ E2 > 90 ∧ E3 <= 90) :=
by
  sorry

end triangle_external_angle_properties_l654_65456


namespace find_y1_l654_65474

noncomputable def y1_proof : Prop :=
∃ (y1 y2 y3 : ℝ), 0 ≤ y3 ∧ y3 ≤ y2 ∧ y2 ≤ y1 ∧ y1 ≤ 1 ∧
(1 - y1)^2 + (y1 - y2)^2 + (y2 - y3)^2 + y3^2 = 1 / 9 ∧
y1 = 1 / 2

-- Statement to be proven:
theorem find_y1 : y1_proof :=
sorry

end find_y1_l654_65474


namespace expectation_defective_items_variance_of_defective_items_l654_65418
-- Importing the necessary library from Mathlib

-- Define the conditions
def total_products : ℕ := 100
def defective_products : ℕ := 10
def selected_products : ℕ := 3

-- Define the expected number of defective items
def expected_defective_items : ℝ := 0.3

-- Define the variance of defective items
def variance_defective_items : ℝ := 0.2645

-- Lean statements to verify the conditions and results
theorem expectation_defective_items :
  let p := (defective_products: ℝ) / (total_products: ℝ)
  p * (selected_products: ℝ) = expected_defective_items := by sorry

theorem variance_of_defective_items :
  let p := (defective_products: ℝ) / (total_products: ℝ)
  let n := (selected_products: ℝ)
  n * p * (1 - p) * (total_products - n) / (total_products - 1) = variance_defective_items := by sorry

end expectation_defective_items_variance_of_defective_items_l654_65418


namespace solve_inequality_l654_65491

theorem solve_inequality {a x : ℝ} (h1 : a > 0) (h2 : a ≠ 1) :
  (∃ (x : ℝ), (a > 1 ∧ (a^(2/3) ≤ x ∧ x < a^(3/4) ∨ x > a)) ∨ (0 < a ∧ a < 1 ∧ (a^(3/4) < x ∧ x ≤ a^(2/3) ∨ 0 < x ∧ x < a))) :=
sorry

end solve_inequality_l654_65491


namespace find_ab_integer_l654_65489

theorem find_ab_integer (a b : ℕ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_neq : a ≠ b) :
    ∃ n : ℤ, (a^b + b^a) = n * (a^a - b^b) ↔ (a = 2 ∧ b = 1) ∨ (a = 1 ∧ b = 2) := 
sorry

end find_ab_integer_l654_65489


namespace power_expression_l654_65455

theorem power_expression (a b : ℕ) (h1 : a = 12) (h2 : b = 18) : (3^a * 3^b) = (243^6) :=
by
  let c := 3
  have h3 : a + b = 30 := by simp [h1, h2]
  have h4 : 3^(a + b) = 3^30 := by rw [h3]
  have h5 : 3^30 = 243^6 := by norm_num
  sorry  -- skip other detailed steps

end power_expression_l654_65455


namespace logarithmic_function_through_point_l654_65466

noncomputable def log_function_expression (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

theorem logarithmic_function_through_point (f : ℝ → ℝ) :
  (∀ x a : ℝ, a > 0 ∧ a ≠ 1 → f x = log_function_expression a x) ∧ f 4 = 2 →
  ∃ g : ℝ → ℝ, ∀ x : ℝ, g x = log_function_expression 2 x :=
by {
  sorry
}

end logarithmic_function_through_point_l654_65466


namespace ivy_baked_55_cupcakes_l654_65427

-- Definitions based on conditions
def cupcakes_morning : ℕ := 20
def cupcakes_afternoon : ℕ := cupcakes_morning + 15
def total_cupcakes : ℕ := cupcakes_morning + cupcakes_afternoon

-- Theorem statement that needs to be proved
theorem ivy_baked_55_cupcakes : total_cupcakes = 55 := by
    sorry

end ivy_baked_55_cupcakes_l654_65427


namespace carla_water_drank_l654_65438

theorem carla_water_drank (W S : ℝ) (h1 : W + S = 54) (h2 : S = 3 * W - 6) : W = 15 :=
by
  sorry

end carla_water_drank_l654_65438


namespace arithmetic_sequence_a5_l654_65476

noncomputable def a_n (n : ℕ) : ℝ := sorry  -- The terms of the arithmetic sequence

theorem arithmetic_sequence_a5 :
  (∀ (n : ℕ), a_n n = a_n 0 + n * (a_n 1 - a_n 0)) →
  a_n 1 = 1 →
  a_n 1 + a_n 3 = 16 →
  a_n 4 = 15 :=
by {
  -- Proof omission, ensure these statements are correct with sorry
  sorry
}

end arithmetic_sequence_a5_l654_65476


namespace f_is_odd_l654_65440

noncomputable def f (α : ℝ) (x : ℝ) : ℝ := (α - 2) * x ^ α

theorem f_is_odd (α : ℝ) (hα : α = 3) : ∀ x : ℝ, f α (-x) = -f α x :=
by sorry

end f_is_odd_l654_65440


namespace range_of_a_l654_65483

variable (a : ℝ)

theorem range_of_a
  (h : ∃ x : ℝ, x^2 + 2 * a * x + 1 < 0) :
  a < -1 ∨ a > 1 :=
by {
  sorry
}

end range_of_a_l654_65483


namespace isosceles_triangle_base_angle_l654_65479

theorem isosceles_triangle_base_angle (A B C : ℝ) (h_sum : A + B + C = 180) (h_iso : B = C) (h_one_angle : A = 80) : B = 50 :=
sorry

end isosceles_triangle_base_angle_l654_65479


namespace sixteen_is_sixtyfour_percent_l654_65435

theorem sixteen_is_sixtyfour_percent (x : ℝ) (h : 16 / x = 64 / 100) : x = 25 :=
by sorry

end sixteen_is_sixtyfour_percent_l654_65435


namespace ram_balance_speed_l654_65439

theorem ram_balance_speed
  (part_speed : ℝ)
  (balance_distance : ℝ)
  (total_distance : ℝ)
  (total_time : ℝ)
  (part_time : ℝ)
  (balance_speed : ℝ)
  (h1 : part_speed = 20)
  (h2 : total_distance = 400)
  (h3 : total_time = 8)
  (h4 : part_time = 3.2)
  (h5 : balance_distance = total_distance - part_speed * part_time)
  (h6 : balance_speed = balance_distance / (total_time - part_time)) :
  balance_speed = 70 :=
by
  simp [h1, h2, h3, h4, h5, h6]
  sorry

end ram_balance_speed_l654_65439


namespace ellipse_condition_sufficient_not_necessary_l654_65450

theorem ellipse_condition_sufficient_not_necessary (n : ℝ) :
  (-1 < n) ∧ (n < 2) → 
  (2 - n > 0) ∧ (n + 1 > 0) ∧ (2 - n > n + 1) :=
by
  intro h
  sorry

end ellipse_condition_sufficient_not_necessary_l654_65450


namespace solution_set_inequality_l654_65493

-- Definitions of the conditions
def is_odd_function (f : ℝ → ℝ) := ∀ x : ℝ, f (-x) = -f x

noncomputable def f : ℝ → ℝ
| x => if x > 0 then 2^x - 3 else - (2^(-x) - 3)

-- Statement to prove
theorem solution_set_inequality :
  is_odd_function f ∧ (∀ x > 0, f x = 2^x - 3)
  → {x : ℝ | f x ≤ -5} = {x : ℝ | x ≤ -3} := by
  sorry

end solution_set_inequality_l654_65493


namespace iPhone_savings_l654_65487

theorem iPhone_savings
  (costX costY : ℕ)
  (discount_same_model discount_mixed : ℝ)
  (h1 : costX = 600)
  (h2 : costY = 800)
  (h3 : discount_same_model = 0.05)
  (h4 : discount_mixed = 0.03) :
  (costX + costX + costY) - ((costX * (1 - discount_same_model)) * 2 + costY * (1 - discount_mixed)) = 84 :=
by
  sorry

end iPhone_savings_l654_65487


namespace min_2a_b_c_l654_65445

theorem min_2a_b_c (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : (a + b) * b * c = 5) :
  2 * a + b + c ≥ 2 * Real.sqrt 5 := sorry

end min_2a_b_c_l654_65445


namespace polynomial_value_l654_65459

theorem polynomial_value (x : ℝ) :
  let a := 2009 * x + 2008
  let b := 2009 * x + 2009
  let c := 2009 * x + 2010
  a^2 + b^2 + c^2 - a * b - b * c - a * c = 3 := by
  sorry

end polynomial_value_l654_65459


namespace dodecagon_diagonals_eq_54_dodecagon_triangles_eq_220_l654_65492

-- Define a regular dodecagon
def dodecagon_sides : ℕ := 12

-- Prove that the number of diagonals in a regular dodecagon is 54
theorem dodecagon_diagonals_eq_54 : (dodecagon_sides * (dodecagon_sides - 3)) / 2 = 54 :=
by sorry

-- Prove that the number of possible triangles formed from a regular dodecagon vertices is 220
theorem dodecagon_triangles_eq_220 : Nat.choose dodecagon_sides 3 = 220 :=
by sorry

end dodecagon_diagonals_eq_54_dodecagon_triangles_eq_220_l654_65492


namespace least_multiple_36_sum_digits_l654_65451

noncomputable def digit_sum (n : ℕ) : ℕ :=
  (n.digits 10).sum

theorem least_multiple_36_sum_digits :
  ∃ n : ℕ, n = 36 ∧ (36 ∣ n) ∧ (9 ∣ digit_sum n) ∧ (∀ m : ℕ, (36 ∣ m) ∧ (9 ∣ digit_sum m) → 36 ≤ m) :=
by sorry

end least_multiple_36_sum_digits_l654_65451


namespace three_digit_problem_l654_65462

theorem three_digit_problem :
  ∃ (M Γ U : ℕ), 
    M ≠ Γ ∧ M ≠ U ∧ Γ ≠ U ∧
    M ≤ 9 ∧ Γ ≤ 9 ∧ U ≤ 9 ∧
    100 * M + 10 * Γ + U = (M + Γ + U) * (M + Γ + U - 2) ∧
    100 * M + 10 * Γ + U = 195 :=
by
  sorry

end three_digit_problem_l654_65462


namespace ral_current_age_l654_65475

-- Definitions according to the conditions
def ral_three_times_suri (ral suri : ℕ) : Prop := ral = 3 * suri
def suri_in_6_years (suri : ℕ) : Prop := suri + 6 = 25

-- The proof problem statement
theorem ral_current_age (ral suri : ℕ) (h1 : ral_three_times_suri ral suri) (h2 : suri_in_6_years suri) : ral = 57 :=
by sorry

end ral_current_age_l654_65475


namespace greatest_possible_value_of_x_l654_65457

-- Define the function based on the given equation
noncomputable def f (x : ℝ) : ℝ := (4 * x - 16) / (3 * x - 4)

-- Statement to be proved
theorem greatest_possible_value_of_x : 
  (∀ x : ℝ, (f x)^2 + (f x) = 20) → 
  ∃ x : ℝ, (f x)^2 + (f x) = 20 ∧ x = 36 / 19 :=
by
  sorry

end greatest_possible_value_of_x_l654_65457


namespace find_weight_first_dog_l654_65498

noncomputable def weight_first_dog (x : ℕ) (y : ℕ) : Prop :=
  (x + 31 + 35 + 33) / 4 = (x + 31 + 35 + 33 + y) / 5

theorem find_weight_first_dog (x : ℕ) : weight_first_dog x 31 → x = 25 := by
  sorry

end find_weight_first_dog_l654_65498


namespace find_positive_n_unique_solution_l654_65411

theorem find_positive_n_unique_solution (n : ℝ) :
  (∃ x : ℝ, 4 * x^2 + n * x + 4 = 0) → n = 8 :=
by
  intro h
  sorry

end find_positive_n_unique_solution_l654_65411


namespace smallest_k_for_720_l654_65499

/-- Given a number 720, prove that the smallest positive integer k such that 720 * k is both a perfect square and a perfect cube is 1012500. -/
theorem smallest_k_for_720 (k : ℕ) : (∃ k > 0, 720 * k = (n : ℕ) ^ 6) -> k = 1012500 :=
by sorry

end smallest_k_for_720_l654_65499


namespace income_increase_percentage_l654_65447

theorem income_increase_percentage (I : ℝ) (P : ℝ) (h1 : 0 < I)
  (h2 : 0 ≤ P) (h3 : 0.75 * I + 0.075 * I = 0.825 * I) 
  (h4 : 1.5 * (0.25 * I) = ((I * (1 + P / 100)) - 0.825 * I)) 
  : P = 20 := by
sorry

end income_increase_percentage_l654_65447


namespace remainder_div_power10_l654_65460

theorem remainder_div_power10 (n : ℕ) (h : n > 0) : 
  ∃ k : ℕ, (10^n - 1) % 37 = k^2 := by
  sorry

end remainder_div_power10_l654_65460


namespace painting_perimeter_l654_65407

-- Definitions for the problem conditions
def frame_thickness : ℕ := 3
def frame_area : ℕ := 108

-- Declaration that expresses the given conditions and the problem's conclusion
theorem painting_perimeter {w h : ℕ} (h_frame : (w + 2 * frame_thickness) * (h + 2 * frame_thickness) - w * h = frame_area) :
  2 * (w + h) = 24 :=
by
  sorry

end painting_perimeter_l654_65407


namespace one_liter_fills_five_cups_l654_65442

-- Define the problem conditions and question in Lean 4
def one_liter_milliliters : ℕ := 1000
def cup_volume_milliliters : ℕ := 200

theorem one_liter_fills_five_cups : one_liter_milliliters / cup_volume_milliliters = 5 := 
by 
  sorry -- proof skipped

end one_liter_fills_five_cups_l654_65442


namespace sequence_becomes_negative_from_8th_term_l654_65430

def seq (n : ℕ) : ℤ := 21 + 4 * n - n ^ 2

theorem sequence_becomes_negative_from_8th_term :
  ∀ n, n ≥ 8 ↔ seq n < 0 :=
by
  -- proof goes here
  sorry

end sequence_becomes_negative_from_8th_term_l654_65430


namespace larger_number_l654_65452

theorem larger_number (x y : ℤ) (h1 : x + y = 47) (h2 : x - y = 7) : x = 27 :=
  sorry

end larger_number_l654_65452


namespace find_people_who_own_only_cats_l654_65471

variable (C : ℕ)

theorem find_people_who_own_only_cats
  (ownsOnlyDogs : ℕ)
  (ownsCatsAndDogs : ℕ)
  (ownsCatsDogsSnakes : ℕ)
  (totalPetOwners : ℕ)
  (h1 : ownsOnlyDogs = 15)
  (h2 : ownsCatsAndDogs = 5)
  (h3 : ownsCatsDogsSnakes = 3)
  (h4 : totalPetOwners = 59) :
  C = 36 :=
by
  sorry

end find_people_who_own_only_cats_l654_65471


namespace percentage_increase_from_1200_to_1680_is_40_l654_65414

theorem percentage_increase_from_1200_to_1680_is_40 :
  let initial_value := 1200
  let final_value := 1680
  let percentage_increase := ((final_value - initial_value) / initial_value) * 100
  percentage_increase = 40 := by
  let initial_value := 1200
  let final_value := 1680
  let percentage_increase := ((final_value - initial_value) / initial_value) * 100
  sorry

end percentage_increase_from_1200_to_1680_is_40_l654_65414


namespace secant_length_problem_l654_65490

theorem secant_length_problem (tangent_length : ℝ) (internal_segment_length : ℝ) (external_segment_length : ℝ) 
    (h1 : tangent_length = 18) (h2 : internal_segment_length = 27) : external_segment_length = 9 :=
by
  sorry

end secant_length_problem_l654_65490


namespace initial_catfish_count_l654_65401

theorem initial_catfish_count (goldfish : ℕ) (remaining_fish : ℕ) (disappeared_fish : ℕ) (catfish : ℕ) :
  goldfish = 7 → 
  remaining_fish = 15 → 
  disappeared_fish = 4 → 
  catfish + goldfish = 19 →
  catfish = 12 :=
by
  intros h1 h2 h3 h4
  sorry

end initial_catfish_count_l654_65401


namespace part_I_part_II_l654_65469

noncomputable def f (x a : ℝ) : ℝ := x^2 - 2 * a * x
noncomputable def g (x : ℝ) : ℝ := Real.log x
noncomputable def h (x a : ℝ) : ℝ := (f x a) + (g x)

theorem part_I (a : ℝ) :
  (∀ x > 0, f x a ≥ g x) → a ≤ 0.5 :=
by
  sorry

theorem part_II (a x1 x2 : ℝ) (hx1 : x1 > 0) (hx2 : x2 > 0) 
  (hx1_lt_half : x1 < 0.5) :
  (h x1 a = 2 * x1^2 + Real.log x1) →
  (h x2 a = 2 * x2^2 + Real.log x2) →
  (x1 * x2 = 0.5) →
  h x1 a - h x2 a > (3 / 4) - Real.log 2 :=
by
  sorry

end part_I_part_II_l654_65469


namespace find_point_A_l654_65421

theorem find_point_A (x : ℝ) (h : x + 7 - 4 = 0) : x = -3 :=
sorry

end find_point_A_l654_65421


namespace asthma_distribution_l654_65467

noncomputable def total_children := 490
noncomputable def boys := 280
noncomputable def general_asthma_ratio := 2 / 7
noncomputable def boys_asthma_ratio := 1 / 9

noncomputable def total_children_with_asthma := general_asthma_ratio * total_children
noncomputable def boys_with_asthma := boys_asthma_ratio * boys
noncomputable def girls_with_asthma := total_children_with_asthma - boys_with_asthma

theorem asthma_distribution
  (h_general_asthma: general_asthma_ratio = 2 / 7)
  (h_total_children: total_children = 490)
  (h_boys: boys = 280)
  (h_boys_asthma: boys_asthma_ratio = 1 / 9):
  boys_with_asthma = 31 ∧ girls_with_asthma = 109 :=
by
  sorry

end asthma_distribution_l654_65467


namespace moses_percentage_l654_65412

theorem moses_percentage (P : ℝ) (T : ℝ) (E : ℝ) (total_amount : ℝ) (moses_more : ℝ)
  (h1 : total_amount = 50)
  (h2 : moses_more = 5)
  (h3 : T = E)
  (h4 : P / 100 * total_amount = E + moses_more)
  (h5 : 2 * E = (1 - P / 100) * total_amount) :
  P = 40 :=
by
  sorry

end moses_percentage_l654_65412


namespace moles_of_ammonia_formed_l654_65436

def reaction (n_koh n_nh4i n_nh3 : ℕ) := 
  n_koh + n_nh4i + n_nh3 

theorem moles_of_ammonia_formed (n_koh : ℕ) :
  reaction n_koh 3 3 = n_koh + 3 + 3 := 
sorry

end moles_of_ammonia_formed_l654_65436


namespace unique_n0_exists_l654_65470

open Set

theorem unique_n0_exists 
  (a : ℕ → ℕ) 
  (S : ℕ → ℕ) 
  (h1 : ∀ n : ℕ, a n < a (n + 1)) 
  (h2 : ∀ n : ℕ, S (n + 1) = S n + a (n + 1))
  (h3 : ∀ n : ℕ, S 0 = a 0) :
  ∃! n_0 : ℕ, (S (n_0 + 1)) / n_0 > a (n_0 + 1)
             ∧ (S (n_0 + 1)) / n_0 ≤ a (n_0 + 2) := 
sorry

end unique_n0_exists_l654_65470


namespace abs_difference_of_two_numbers_l654_65406

theorem abs_difference_of_two_numbers (x y : ℝ) (h1 : x + y = 36) (h2 : x * y = 105) :
  |x - y| = 6 * Real.sqrt 24.333 := sorry

end abs_difference_of_two_numbers_l654_65406


namespace range_of_e_l654_65400

theorem range_of_e (a b c d e : ℝ)
  (h1 : a + b + c + d + e = 8)
  (h2 : a^2 + b^2 + c^2 + d^2 + e^2 = 16) :
  0 ≤ e ∧ e ≤ 16 / 5 :=
by
  sorry

end range_of_e_l654_65400


namespace parallel_vectors_l654_65433

noncomputable def vector_a : ℝ × ℝ := (2, 1)
noncomputable def vector_b (m : ℝ) : ℝ × ℝ := (m, -1)

theorem parallel_vectors {m : ℝ} (h : (∃ k : ℝ, vector_a = k • vector_b m)) : m = -2 :=
by
  sorry

end parallel_vectors_l654_65433


namespace travis_flight_cost_l654_65419

theorem travis_flight_cost 
  (cost_leg1 : ℕ := 1500) 
  (cost_leg2 : ℕ := 1000) 
  (discount_leg1 : ℕ := 25) 
  (discount_leg2 : ℕ := 35) : 
  cost_leg1 - (discount_leg1 * cost_leg1 / 100) + cost_leg2 - (discount_leg2 * cost_leg2 / 100) = 1775 :=
by
  sorry

end travis_flight_cost_l654_65419


namespace quadratic_roots_r12_s12_l654_65446

theorem quadratic_roots_r12_s12 (r s : ℝ) (h1 : r + s = 2 * Real.sqrt 3) (h2 : r * s = 1) :
  r^12 + s^12 = 940802 :=
sorry

end quadratic_roots_r12_s12_l654_65446


namespace num_distinct_remainders_of_prime_squared_mod_120_l654_65473

theorem num_distinct_remainders_of_prime_squared_mod_120:
  ∀ p : ℕ, Prime p → p > 5 → (p^2 % 120 = 1 ∨ p^2 % 120 = 49) := 
sorry

end num_distinct_remainders_of_prime_squared_mod_120_l654_65473


namespace ratio_a_over_3_to_b_over_2_l654_65468

theorem ratio_a_over_3_to_b_over_2 (a b c : ℝ) (h1 : 2 * a = 3 * b) (h2 : c ≠ 0) (h3 : 3 * a + 2 * b = c) :
  (a / 3) / (b / 2) = 1 :=
sorry

end ratio_a_over_3_to_b_over_2_l654_65468


namespace miles_driven_l654_65404

-- Definitions based on the conditions
def years : ℕ := 9
def months_in_a_year : ℕ := 12
def months_in_a_period : ℕ := 4
def miles_per_period : ℕ := 37000

-- The proof statement
theorem miles_driven : years * months_in_a_year / months_in_a_period * miles_per_period = 999000 := 
sorry

end miles_driven_l654_65404


namespace sufficient_drivers_and_correct_time_l654_65454

-- Conditions definitions
def one_way_minutes := 2 * 60 + 40  -- 2 hours 40 minutes in minutes
def round_trip_minutes := 2 * one_way_minutes  -- round trip in minutes
def rest_minutes := 60  -- mandatory rest period in minutes

-- Time checks for drivers
def driver_a_return := 12 * 60 + 40  -- Driver A returns at 12:40 PM in minutes
def driver_a_next_trip := driver_a_return + rest_minutes  -- Driver A's next trip time
def driver_d_departure := 13 * 60 + 5  -- Driver D departs at 13:05 in minutes

-- Verify sufficiency of four drivers and time correctness
theorem sufficient_drivers_and_correct_time : 
  4 = 4 ∧ (driver_a_next_trip + round_trip_minutes = 21 * 60 + 30) :=
by
  -- Explain the reasoning path that leads to this conclusion within this block
  sorry

end sufficient_drivers_and_correct_time_l654_65454


namespace smallest_nonneg_integer_l654_65485

theorem smallest_nonneg_integer (n : ℕ) (h : 0 ≤ n ∧ n < 53) :
  50 * n ≡ 47 [MOD 53] → n = 2 :=
by
  sorry

end smallest_nonneg_integer_l654_65485


namespace perpendicular_tangent_line_exists_and_correct_l654_65465

theorem perpendicular_tangent_line_exists_and_correct :
  ∃ L : ℝ → ℝ → Prop,
    (∀ x y, L x y ↔ 3 * x + y + 6 = 0) ∧
    (∀ x y, 2 * x - 6 * y + 1 = 0 → 3 * x + y + 6 ≠ 0) ∧
    (∃ a b : ℝ, 
       b = a^3 + 3*a^2 - 5 ∧ 
       (a, b) ∈ { p : ℝ × ℝ | ∃ f' : ℝ → ℝ, f' a = 3 * a^2 + 6 * a ∧ f' a * 3 + 1 = 0 } ∧
       L a b)
:= 
sorry

end perpendicular_tangent_line_exists_and_correct_l654_65465


namespace quadratic_inequality_solution_l654_65408

theorem quadratic_inequality_solution (d : ℝ) 
  (h1 : 0 < d) 
  (h2 : d < 16) : 
  ∃ x : ℝ, (x^2 - 8*x + d < 0) :=
  sorry

end quadratic_inequality_solution_l654_65408


namespace remaining_distance_l654_65464

-- Definitions for the given conditions
def total_distance : ℕ := 436
def first_stopover_distance : ℕ := 132
def second_stopover_distance : ℕ := 236

-- Prove that the remaining distance from the second stopover to the island is 68 miles.
theorem remaining_distance : total_distance - (first_stopover_distance + second_stopover_distance) = 68 := by
  -- The proof (details) will go here
  sorry

end remaining_distance_l654_65464


namespace solve_equation1_solve_equation2_solve_equation3_solve_equation4_l654_65453

theorem solve_equation1 (x : ℝ) : (x - 1)^2 = 4 ↔ (x = 3 ∨ x = -1) :=
by sorry

theorem solve_equation2 (x : ℝ) : (x - 2)^2 = 5 * (x - 2) ↔ (x = 2 ∨ x = 7) :=
by sorry

theorem solve_equation3 (x : ℝ) : x^2 - 5*x + 6 = 0 ↔ (x = 2 ∨ x = 3) :=
by sorry

theorem solve_equation4 (x : ℝ) : x^2 - 4*x - 1 = 0 ↔ (x = 2 + Real.sqrt 5 ∨ x = 2 - Real.sqrt 5) :=
by sorry

end solve_equation1_solve_equation2_solve_equation3_solve_equation4_l654_65453


namespace new_mean_after_adding_14_to_each_of_15_numbers_l654_65434

theorem new_mean_after_adding_14_to_each_of_15_numbers (avg : ℕ) (n : ℕ) (n_sum : ℕ) (new_sum : ℕ) :
  avg = 40 →
  n = 15 →
  n_sum = n * avg →
  new_sum = n_sum + n * 14 →
  new_sum / n = 54 :=
by
  intros h_avg h_n h_n_sum h_new_sum
  sorry

end new_mean_after_adding_14_to_each_of_15_numbers_l654_65434


namespace solve_quadratic_l654_65423

theorem solve_quadratic (x : ℝ) : 2 * x^2 - x = 2 ↔ x = (1 + Real.sqrt 17) / 4 ∨ x = (1 - Real.sqrt 17) / 4 := by
  sorry

end solve_quadratic_l654_65423


namespace inequality_proof_l654_65416

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h : a * b * c = 1/8) :
  a^2 + b^2 + c^2 + a^2 * b^2 + b^2 * c^2 + c^2 * a^2 ≥ 15 / 16 :=
by sorry

end inequality_proof_l654_65416


namespace xy_addition_l654_65448

theorem xy_addition (x y : ℕ) (h1 : x * y = 24) (h2 : x - y = 5) (hx_pos : 0 < x) (hy_pos : 0 < y) : x + y = 11 := 
sorry

end xy_addition_l654_65448


namespace fraction_of_fritz_money_l654_65496

theorem fraction_of_fritz_money
  (Fritz_money : ℕ)
  (total_amount : ℕ)
  (fraction : ℚ)
  (Sean_money : ℚ)
  (Rick_money : ℚ)
  (h1 : Fritz_money = 40)
  (h2 : total_amount = 96)
  (h3 : Sean_money = fraction * Fritz_money + 4)
  (h4 : Rick_money = 3 * Sean_money)
  (h5 : Rick_money + Sean_money = total_amount) :
  fraction = 1 / 2 :=
by
  sorry

end fraction_of_fritz_money_l654_65496


namespace general_term_sequence_x_l654_65417

-- Definitions used in Lean statement corresponding to the conditions.
noncomputable def sequence_a (n : ℕ) : ℝ := sorry

noncomputable def sequence_x (n : ℕ) : ℝ := sorry

axiom condition_1 : ∀ n : ℕ, 
  ((sequence_a (n + 2))⁻¹ = ((sequence_a n)⁻¹ + (sequence_a (n + 1))⁻¹) / 2)

axiom condition_2 {n : ℕ} : sequence_x n > 0

axiom condition_3 : sequence_x 1 = 3

axiom condition_4 : sequence_x 1 + sequence_x 2 + sequence_x 3 = 39

axiom condition_5 (n : ℕ) : (sequence_x n)^(sequence_a n) = 
  (sequence_x (n + 1))^(sequence_a (n + 1)) ∧ 
  (sequence_x (n + 1))^(sequence_a (n + 1)) = 
  (sequence_x (n + 2))^(sequence_a (n + 2))

-- Theorem stating that the general term of sequence {x_n} is 3^n.
theorem general_term_sequence_x : ∀ n : ℕ, sequence_x n = 3^n :=
by
  sorry

end general_term_sequence_x_l654_65417


namespace businessmen_neither_coffee_nor_tea_l654_65472

theorem businessmen_neither_coffee_nor_tea :
  ∀ (total_count coffee tea both neither : ℕ),
    total_count = 30 →
    coffee = 15 →
    tea = 13 →
    both = 6 →
    neither = total_count - (coffee + tea - both) →
    neither = 8 := 
by
  intros total_count coffee tea both neither ht hc ht2 hb hn
  rw [ht, hc, ht2, hb] at hn
  simp at hn
  exact hn

end businessmen_neither_coffee_nor_tea_l654_65472


namespace frank_total_cans_l654_65410

def cansCollectedSaturday : List Nat := [4, 6, 5, 7, 8]
def cansCollectedSunday : List Nat := [6, 5, 9]
def cansCollectedMonday : List Nat := [8, 8]

def totalCansCollected (lst1 lst2 lst3 : List Nat) : Nat :=
  lst1.sum + lst2.sum + lst3.sum

theorem frank_total_cans :
  totalCansCollected cansCollectedSaturday cansCollectedSunday cansCollectedMonday = 66 :=
by
  sorry

end frank_total_cans_l654_65410


namespace arithmetic_progression_no_rth_power_l654_65405

noncomputable def is_arith_sequence (a : ℕ → ℤ) : Prop := 
∀ n : ℕ, a n = 4 * (n : ℤ) - 2

theorem arithmetic_progression_no_rth_power (n : ℕ) :
  ∃ a : ℕ → ℤ, is_arith_sequence a ∧ 
  (∀ r : ℕ, 2 ≤ r ∧ r ≤ n → 
  ¬ (∃ k : ℤ, ∃ m : ℕ, m > 0 ∧ a m = k ^ r)) := 
sorry

end arithmetic_progression_no_rth_power_l654_65405


namespace peach_pies_l654_65420

theorem peach_pies (total_pies : ℕ) (apple_ratio blueberry_ratio peach_ratio : ℕ)
  (h_ratio : apple_ratio + blueberry_ratio + peach_ratio = 10)
  (h_total : total_pies = 30)
  (h_ratios : apple_ratio = 3 ∧ blueberry_ratio = 2 ∧ peach_ratio = 5) :
  total_pies / (apple_ratio + blueberry_ratio + peach_ratio) * peach_ratio = 15 :=
by
  sorry

end peach_pies_l654_65420


namespace necessarily_positive_y_plus_z_l654_65443

-- Given conditions
variables {x y z : ℝ}

-- Assert the conditions
axiom hx : 0 < x ∧ x < 1
axiom hy : -1 < y ∧ y < 0
axiom hz : 1 < z ∧ z < 2

-- Prove that y + z is necessarily positive
theorem necessarily_positive_y_plus_z : y + z > 0 :=
by
  sorry

end necessarily_positive_y_plus_z_l654_65443


namespace sin_theta_value_l654_65480

theorem sin_theta_value (θ : ℝ) (h₁ : θ ∈ Set.Icc (Real.pi / 4) (Real.pi / 2)) 
  (h₂ : Real.sin (2 * θ) = (3 * Real.sqrt 7) / 8) : Real.sin θ = 3 / 4 :=
sorry

end sin_theta_value_l654_65480


namespace Toby_second_part_distance_l654_65486

noncomputable def total_time_journey (distance_unloaded_second: ℝ) : ℝ :=
  18 + (distance_unloaded_second / 20) + 8 + 7

theorem Toby_second_part_distance:
  ∃ d : ℝ, total_time_journey d = 39 ∧ d = 120 :=
by
  use 120
  unfold total_time_journey
  sorry

end Toby_second_part_distance_l654_65486


namespace floor_sqrt_77_l654_65403

theorem floor_sqrt_77 : 8 < Real.sqrt 77 ∧ Real.sqrt 77 < 9 → Int.floor (Real.sqrt 77) = 8 :=
by
  sorry

end floor_sqrt_77_l654_65403


namespace canteen_distance_l654_65431

-- Given definitions
def G_to_road : ℝ := 450
def G_to_B : ℝ := 700

-- Proof statement
theorem canteen_distance :
  ∃ x : ℝ, (x ≠ 0) ∧ 
           (G_to_road^2 + (x - G_to_road)^2 = x^2) ∧ 
           (x = 538) := 
by {
  sorry
}

end canteen_distance_l654_65431


namespace hyperbola_eccentricity_l654_65481

variables (a b c e : ℝ) (a_pos : a > 0) (b_pos : b > 0)
          (c_eq : c = 4) (b_eq : b = 2 * Real.sqrt 3)
          (hyperbola_eq : c ^ 2 = a ^ 2 + b ^ 2)
          (projection_cond : 2 < (4 * (a * Real.sqrt (a ^ 2 + b ^ 2)) / (Real.sqrt (a ^ 2 + b ^ 2))) ∧ (4 * (a * Real.sqrt (a ^ 2 + b ^ 2)) / (Real.sqrt (a ^ 2 + b ^ 2))) ≤ 4)

theorem hyperbola_eccentricity : e = c / a := 
by
  sorry

end hyperbola_eccentricity_l654_65481


namespace sandra_socks_l654_65463

variables (x y z : ℕ)

theorem sandra_socks :
  x + y + z = 15 →
  2 * x + 3 * y + 5 * z = 36 →
  x ≤ 6 →
  y ≤ 6 →
  z ≤ 6 →
  x = 11 :=
by
  sorry

end sandra_socks_l654_65463


namespace percentage_transactions_anthony_handled_more_l654_65413

theorem percentage_transactions_anthony_handled_more (M A C J : ℕ) (P : ℚ)
  (hM : M = 90)
  (hJ : J = 83)
  (hCJ : J = C + 17)
  (hCA : C = (2 * A) / 3)
  (hP : P = ((A - M): ℚ) / M * 100) :
  P = 10 := by
  sorry

end percentage_transactions_anthony_handled_more_l654_65413


namespace ab_square_l654_65497

theorem ab_square (x y : ℝ) (hx : y = 4 * x^2 + 7 * x - 1) (hy : y = -4 * x^2 + 7 * x + 1) :
  (2 * x)^2 + (2 * y)^2 = 50 :=
by
  sorry

end ab_square_l654_65497


namespace binom_1450_2_eq_1050205_l654_65437

def binom_coefficient (n k : ℕ) : ℕ :=
  n.choose k

theorem binom_1450_2_eq_1050205 : binom_coefficient 1450 2 = 1050205 :=
by {
  sorry
}

end binom_1450_2_eq_1050205_l654_65437


namespace mrs_awesome_class_l654_65495

def num_students (b g : ℕ) : ℕ := b + g

theorem mrs_awesome_class (b g : ℕ) (h1 : b = g + 3) (h2 : 480 - (b * b + g * g) = 5) : num_students b g = 31 :=
by
  sorry

end mrs_awesome_class_l654_65495


namespace product_correlation_function_l654_65449

open ProbabilityTheory

/-
Theorem: Given two centered and uncorrelated random functions \( \dot{X}(t) \) and \( \dot{Y}(t) \),
the correlation function of their product \( Z(t) = \dot{X}(t) \dot{Y}(t) \) is the product of their correlation functions.
-/
theorem product_correlation_function 
  (X Y : ℝ → ℝ)
  (hX_centered : ∀ t, (∫ x, X t ∂x) = 0) 
  (hY_centered : ∀ t, (∫ y, Y t ∂y) = 0)
  (h_uncorrelated : ∀ t1 t2, ∫ x, X t1 * Y t2 ∂x = (∫ x, X t1 ∂x) * (∫ y, Y t2 ∂y)) :
  ∀ t1 t2, 
  (∫ x, (X t1 * Y t1) * (X t2 * Y t2) ∂x) = 
  (∫ x, X t1 * X t2 ∂x) * (∫ y, Y t1 * Y t2 ∂y) :=
by
  sorry

end product_correlation_function_l654_65449


namespace value_of_fraction_l654_65402

theorem value_of_fraction (x y z w : ℝ) 
  (h1 : x = 4 * y) 
  (h2 : y = 3 * z) 
  (h3 : z = 5 * w) : 
  (x * z) / (y * w) = 20 := 
by
  sorry

end value_of_fraction_l654_65402


namespace find_twentieth_special_number_l654_65484

theorem find_twentieth_special_number :
  ∃ n : ℕ, (n ≡ 2 [MOD 3]) ∧ (n ≡ 5 [MOD 8]) ∧ (∀ k < 20, ∃ m : ℕ, (m ≡ 2 [MOD 3]) ∧ (m ≡ 5 [MOD 8]) ∧ m < n) ∧ (n = 461) := 
sorry

end find_twentieth_special_number_l654_65484


namespace min_value_of_a_plus_2b_l654_65426

theorem min_value_of_a_plus_2b (a b : ℝ) (h_pos : a > 0 ∧ b > 0) (h_eq : 1 / a + 2 / b = 4) : a + 2 * b = 9 / 4 :=
by
  sorry

end min_value_of_a_plus_2b_l654_65426


namespace value_of_expression_l654_65488

-- defining the conditions
def in_interval (a : ℝ) : Prop := 1 < a ∧ a < 2

-- defining the algebraic expression
def algebraic_expression (a : ℝ) : ℝ := abs (a - 2) + abs (1 - a)

-- theorem to be proved
theorem value_of_expression (a : ℝ) (h : in_interval a) : algebraic_expression a = 1 :=
by
  -- proof will go here
  sorry

end value_of_expression_l654_65488


namespace find_n_l654_65482

theorem find_n (x n : ℝ) (h_x : x = 0.5) : (9 / (1 + n / x) = 1) → n = 4 := 
by
  intro h
  have h_x_eq : x = 0.5 := h_x
  -- Proof content here covering the intermediary steps
  sorry

end find_n_l654_65482


namespace third_beats_seventh_l654_65415

-- Definitions and conditions
variable (points : Fin 8 → ℕ)
variable (distinct_points : Function.Injective points)
variable (sum_last_four : points 1 = points 4 + points 5 + points 6 + points 7)

-- Proof statement
theorem third_beats_seventh 
  (h_distinct : ∀ i j, i ≠ j → points i ≠ points j)
  (h_sum : points 1 = points 4 + points 5 + points 6 + points 7) :
  points 2 > points 6 :=
sorry

end third_beats_seventh_l654_65415


namespace find_number_l654_65478

theorem find_number (x : ℝ) (h : 0.30 * x = 90 + 120) : x = 700 :=
by 
  sorry

end find_number_l654_65478


namespace exponents_problem_l654_65409

theorem exponents_problem :
  5000 * (5000^9) * 2^(1000) = 5000^(10) * 2^(1000) := by sorry

end exponents_problem_l654_65409


namespace calculate_sum_and_double_l654_65458

theorem calculate_sum_and_double :
  2 * (1324 + 4231 + 3124 + 2413) = 22184 :=
by
  sorry

end calculate_sum_and_double_l654_65458


namespace inequality_range_l654_65432

theorem inequality_range (a : ℝ) (h : ∀ x : ℝ, |x - 3| + |x + 1| > a) : a < 4 := by
  sorry

end inequality_range_l654_65432


namespace part1_part2_l654_65444

-- Given Definitions
variable (p : ℕ) [hp : Fact (p > 3)] [prime : Fact (Nat.Prime p)]
variable (A_l : ℕ → ℕ)

-- Assertions to Prove
theorem part1 (l : ℕ) (hl : 1 ≤ l ∧ l ≤ p - 2) : A_l l % p = 0 :=
sorry

theorem part2 (l : ℕ) (hl : 1 < l ∧ l < p ∧ l % 2 = 1) : A_l l % (p * p) = 0 :=
sorry

end part1_part2_l654_65444


namespace quadratic_roots_l654_65422

-- Definitions based on problem conditions
def sum_of_roots (p q : ℝ) : Prop := p + q = 12
def abs_diff_of_roots (p q : ℝ) : Prop := |p - q| = 4

-- The theorem we want to prove
theorem quadratic_roots : ∃ (a b c : ℝ), a ≠ 0 ∧ (∀ p q, sum_of_roots p q ∧ abs_diff_of_roots p q → a * (x - p) * (x - q) = x^2 - 12 * x + 32) := sorry

end quadratic_roots_l654_65422


namespace largest_product_of_three_l654_65428

-- Definitions of the numbers in the set
def numbers : List Int := [-5, 1, -3, 5, -2, 2]

-- Define a function to calculate the product of a list of three integers
def product_of_three (a b c : Int) : Int := a * b * c

-- Define a predicate to state that 75 is the largest product of any three numbers from the given list
theorem largest_product_of_three :
  ∃ (a b c : Int), a ∈ numbers ∧ b ∈ numbers ∧ c ∈ numbers ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ product_of_three a b c = 75 :=
sorry

end largest_product_of_three_l654_65428


namespace num_distinct_convex_polygons_on_12_points_l654_65429

theorem num_distinct_convex_polygons_on_12_points : 
  let num_subsets := 2 ^ 12
  let num_subsets_with_0_members := Nat.choose 12 0
  let num_subsets_with_1_member := Nat.choose 12 1
  let num_subsets_with_2_members := Nat.choose 12 2
  num_subsets - num_subsets_with_0_members - num_subsets_with_1_member - num_subsets_with_2_members = 4017 :=
by
  let num_subsets := 2 ^ 12
  let num_subsets_with_0_members := Nat.choose 12 0
  let num_subsets_with_1_member := Nat.choose 12 1
  let num_subsets_with_2_members := Nat.choose 12 2
  have h : num_subsets - num_subsets_with_0_members - num_subsets_with_1_member - num_subsets_with_2_members = 4017 := by sorry
  exact h

end num_distinct_convex_polygons_on_12_points_l654_65429


namespace inequality_proof_l654_65441

theorem inequality_proof (x y : ℝ) (hx : |x| < 1) (hy : |y| < 1) :
  1 / (1 - x^2) + 1 / (1 - y^2) ≥ 2 / (1 - x * y) :=
sorry

end inequality_proof_l654_65441


namespace find_a_l654_65424

variable {x n : ℝ}

theorem find_a (hx : x > 0) (hn : n > 0) :
    (∀ n > 0, x + n^n / x^n ≥ n + 1) ↔ (∀ n > 0, a = n^n) :=
sorry

end find_a_l654_65424


namespace problem_statement_l654_65461

theorem problem_statement (m : ℝ) (h : m^2 + m - 1 = 0) : m^3 + 2 * m^2 + 2004 = 2005 :=
sorry

end problem_statement_l654_65461
