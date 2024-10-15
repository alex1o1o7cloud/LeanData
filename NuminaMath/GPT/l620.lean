import Mathlib

namespace NUMINAMATH_GPT_largest_number_is_27_l620_62005

-- Define the condition as a predicate
def three_consecutive_multiples_sum_to (k : ℕ) (sum : ℕ) : Prop :=
  ∃ n : ℕ, (3 * n) + (3 * n + 3) + (3 * n + 6) = sum

-- Define the proof statement
theorem largest_number_is_27 : three_consecutive_multiples_sum_to 3 72 → 3 * 7 + 6 = 27 :=
by
  intro h
  cases' h with n h_eq
  sorry

end NUMINAMATH_GPT_largest_number_is_27_l620_62005


namespace NUMINAMATH_GPT_least_faces_combined_l620_62090

noncomputable def num_faces_dice_combined : ℕ :=
  let a := 11
  let b := 7
  a + b

/-- Given the conditions on the dice setups for sums of 8, 11, and 15,
the least number of faces on the two dice combined is 18. -/
theorem least_faces_combined (a b : ℕ) (h1 : 6 < a) (h2 : 6 < b)
  (h_sum_8 : ∃ (p : ℕ), p = 7)  -- 7 ways to roll a sum of 8
  (h_sum_11 : ∃ (q : ℕ), q = 14)  -- half probability means 14 ways to roll a sum of 11
  (h_sum_15 : ∃ (r : ℕ), r = 2) : a + b = 18 :=
by
  sorry

end NUMINAMATH_GPT_least_faces_combined_l620_62090


namespace NUMINAMATH_GPT_maximum_value_of_f_in_interval_l620_62079

noncomputable def f (x : ℝ) := (Real.sin x)^2 + (Real.sqrt 3) * Real.cos x - (3 / 4)

theorem maximum_value_of_f_in_interval : 
  ∃ x ∈ Set.Icc (0 : ℝ) (Real.pi / 2), f x = 1 := 
  sorry

end NUMINAMATH_GPT_maximum_value_of_f_in_interval_l620_62079


namespace NUMINAMATH_GPT_boat_travel_time_l620_62085

noncomputable def total_travel_time (stream_speed boat_speed distance_AB : ℝ) : ℝ :=
  let downstream_speed := boat_speed + stream_speed
  let upstream_speed := boat_speed - stream_speed
  let distance_BC := distance_AB / 2
  (distance_AB / downstream_speed) + (distance_BC / upstream_speed)

theorem boat_travel_time :
  total_travel_time 4 14 180 = 19 :=
by
  sorry

end NUMINAMATH_GPT_boat_travel_time_l620_62085


namespace NUMINAMATH_GPT_roots_quadratic_relation_l620_62026

theorem roots_quadratic_relation (a b c d A B : ℝ)
  (h1 : a^2 + A * a + 1 = 0)
  (h2 : b^2 + A * b + 1 = 0)
  (h3 : c^2 + B * c + 1 = 0)
  (h4 : d^2 + B * d + 1 = 0) :
  (a - c) * (b - c) * (a + d) * (b + d) = B^2 - A^2 :=
sorry

end NUMINAMATH_GPT_roots_quadratic_relation_l620_62026


namespace NUMINAMATH_GPT_total_oranges_for_philip_l620_62077

-- Define the initial conditions
def betty_oranges : ℕ := 15
def bill_oranges : ℕ := 12
def combined_oranges : ℕ := betty_oranges + bill_oranges
def frank_oranges : ℕ := 3 * combined_oranges
def seeds_planted : ℕ := 4 * frank_oranges
def successful_trees : ℕ := (3 / 4) * seeds_planted

-- The ratio of trees with different quantities of oranges
def ratio_parts : ℕ := 2 + 3 + 5
def trees_with_8_oranges : ℕ := (2 * successful_trees) / ratio_parts
def trees_with_10_oranges : ℕ := (3 * successful_trees) / ratio_parts
def trees_with_14_oranges : ℕ := (5 * successful_trees) / ratio_parts

-- Calculate the total number of oranges
def total_oranges : ℕ :=
  (trees_with_8_oranges * 8) +
  (trees_with_10_oranges * 10) +
  (trees_with_14_oranges * 14)

-- Statement to prove
theorem total_oranges_for_philip : total_oranges = 2798 :=
by
  sorry

end NUMINAMATH_GPT_total_oranges_for_philip_l620_62077


namespace NUMINAMATH_GPT_heather_distance_l620_62064

-- Definitions based on conditions
def distance_from_car_to_entrance (x : ℝ) : ℝ := x
def distance_from_entrance_to_rides (x : ℝ) : ℝ := x
def distance_from_rides_to_car : ℝ := 0.08333333333333333
def total_distance_walked : ℝ := 0.75

-- Lean statement to prove
theorem heather_distance (x : ℝ) (h : distance_from_car_to_entrance x + distance_from_entrance_to_rides x + distance_from_rides_to_car = total_distance_walked) :
  x = 0.33333333333333335 :=
by
  sorry

end NUMINAMATH_GPT_heather_distance_l620_62064


namespace NUMINAMATH_GPT_find_remainder_l620_62015

def p (x : ℝ) : ℝ := x^5 + 2 * x^2 + 1

theorem find_remainder : p 2 = 41 :=
by sorry

end NUMINAMATH_GPT_find_remainder_l620_62015


namespace NUMINAMATH_GPT_total_number_of_students_l620_62059

theorem total_number_of_students
  (ratio_girls_to_boys : ℕ) (ratio_boys_to_girls : ℕ)
  (num_girls : ℕ)
  (ratio_condition : ratio_girls_to_boys = 5 ∧ ratio_boys_to_girls = 8)
  (num_girls_condition : num_girls = 160)
  : (num_girls * (ratio_girls_to_boys + ratio_boys_to_girls) / ratio_girls_to_boys = 416) :=
by
  sorry

end NUMINAMATH_GPT_total_number_of_students_l620_62059


namespace NUMINAMATH_GPT_smallest_positive_integer_g_l620_62069

theorem smallest_positive_integer_g (g : ℕ) (h_pos : g > 0) (h_square : ∃ k : ℕ, 3150 * g = k^2) : g = 14 := 
  sorry

end NUMINAMATH_GPT_smallest_positive_integer_g_l620_62069


namespace NUMINAMATH_GPT_matchstick_triangle_sides_l620_62040

theorem matchstick_triangle_sides (a b c : ℕ) :
  a + b + c = 100 ∧ max a (max b c) = 3 * min a (min b c) ∧
  (a < b ∧ b < c ∨ a < c ∧ c < b ∨ b < a ∧ a < c) →
  (a = 15 ∧ b = 40 ∧ c = 45 ∨ a = 16 ∧ b = 36 ∧ c = 48) :=
by
  sorry

end NUMINAMATH_GPT_matchstick_triangle_sides_l620_62040


namespace NUMINAMATH_GPT_height_difference_zero_l620_62072

-- Define the problem statement and conditions
theorem height_difference_zero (a b : ℝ) (h1 : ∀ x, y = 2 * x^2)
  (h2 : b - a^2 = 1 / 4) : 
  ( b - 2 * a^2) = 0 :=
by
  sorry

end NUMINAMATH_GPT_height_difference_zero_l620_62072


namespace NUMINAMATH_GPT_percent_greater_than_fraction_l620_62037

theorem percent_greater_than_fraction : 
  (0.80 * 40) - (4/5) * 20 = 16 :=
by
  sorry

end NUMINAMATH_GPT_percent_greater_than_fraction_l620_62037


namespace NUMINAMATH_GPT_systematic_sampling_method_l620_62095

theorem systematic_sampling_method (k : ℕ) (n : ℕ) 
  (invoice_stubs : ℕ → ℕ) : 
  (k > 0) → 
  (n > 0) → 
  (invoice_stubs 15 = k) → 
  (∀ i : ℕ, invoice_stubs (15 + i * 50) = k + i * 50)
  → (sampling_method = "systematic") :=
by 
  intro h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_systematic_sampling_method_l620_62095


namespace NUMINAMATH_GPT_solve_inequality_l620_62057

theorem solve_inequality (x : ℝ) : x > 13 ↔ x^3 - 16 * x^2 + 73 * x > 84 :=
by
  sorry

end NUMINAMATH_GPT_solve_inequality_l620_62057


namespace NUMINAMATH_GPT_percent_singles_l620_62073

theorem percent_singles :
  ∀ (total_hits home_runs triples doubles : ℕ),
  total_hits = 50 →
  home_runs = 2 →
  triples = 4 →
  doubles = 10 →
  (total_hits - (home_runs + triples + doubles)) * 100 / total_hits = 68 :=
by
  sorry

end NUMINAMATH_GPT_percent_singles_l620_62073


namespace NUMINAMATH_GPT_problem1_inequality_problem2_inequality_l620_62080

theorem problem1_inequality (x : ℝ) (h1 : 2 * x + 10 ≤ 5 * x + 1) (h2 : 3 * (x - 1) > 9) : x > 4 := sorry

theorem problem2_inequality (x : ℝ) (h1 : 3 * (x + 2) ≥ 2 * x + 5) (h2 : 2 * x - (3 * x + 1) / 2 < 1) : -1 ≤ x ∧ x < 3 := sorry

end NUMINAMATH_GPT_problem1_inequality_problem2_inequality_l620_62080


namespace NUMINAMATH_GPT_jill_total_tax_percentage_l620_62025

theorem jill_total_tax_percentage (total_spent : ℝ) 
  (spent_clothing : ℝ) (spent_food : ℝ) (spent_other : ℝ)
  (tax_clothing_rate : ℝ) (tax_food_rate : ℝ) (tax_other_rate : ℝ)
  (h_clothing : spent_clothing = 0.45 * total_spent)
  (h_food : spent_food = 0.45 * total_spent)
  (h_other : spent_other = 0.10 * total_spent)
  (h_tax_clothing : tax_clothing_rate = 0.05)
  (h_tax_food : tax_food_rate = 0.0)
  (h_tax_other : tax_other_rate = 0.10) :
  ((spent_clothing * tax_clothing_rate + spent_food * tax_food_rate + spent_other * tax_other_rate) / total_spent) * 100 = 3.25 :=
by
  sorry

end NUMINAMATH_GPT_jill_total_tax_percentage_l620_62025


namespace NUMINAMATH_GPT_solve_card_trade_problem_l620_62062

def card_trade_problem : Prop :=
  ∃ V : ℕ, 
  (75 - V + 10 + 88 - 8 + V = 75 + 88 - 8 + 10 ∧ V + 15 = 35)

theorem solve_card_trade_problem : card_trade_problem :=
  sorry

end NUMINAMATH_GPT_solve_card_trade_problem_l620_62062


namespace NUMINAMATH_GPT_smallest_addition_to_make_multiple_of_5_l620_62067

theorem smallest_addition_to_make_multiple_of_5 : ∃ k : ℕ, k > 0 ∧ (729 + k) % 5 = 0 ∧ k = 1 := sorry

end NUMINAMATH_GPT_smallest_addition_to_make_multiple_of_5_l620_62067


namespace NUMINAMATH_GPT_find_value_of_expression_l620_62036

-- Given conditions
variable (a : ℝ)
variable (h_root : a^2 + 2 * a - 2 = 0)

-- Mathematically equivalent proof problem
theorem find_value_of_expression : 3 * a^2 + 6 * a + 2023 = 2029 :=
by
  sorry

end NUMINAMATH_GPT_find_value_of_expression_l620_62036


namespace NUMINAMATH_GPT_geometric_seq_term_positive_l620_62027

theorem geometric_seq_term_positive :
  ∃ (b : ℝ), 81 * (b / 81) = b ∧ b * (b / 81) = (8 / 27) ∧ b > 0 ∧ b = 2 * Real.sqrt 6 :=
by 
  use 2 * Real.sqrt 6
  sorry

end NUMINAMATH_GPT_geometric_seq_term_positive_l620_62027


namespace NUMINAMATH_GPT_common_ratio_eq_l620_62099

variables {x y z r : ℝ}

theorem common_ratio_eq (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) (hxy : x ≠ y) (hxz : x ≠ z) (hyz : y ≠ z)
  (hgp : x * (y - z) ≠ 0 ∧ y * (z - x) ≠ 0 ∧ z * (x - y) ≠ 0 ∧ 
          (y * (z - x)) / (x * (y - z)) = r ∧ (z * (x - y)) / (y * (z - x)) = r) :
  r^2 + r + 1 = 0 :=
sorry

end NUMINAMATH_GPT_common_ratio_eq_l620_62099


namespace NUMINAMATH_GPT_find_natural_number_pairs_l620_62052

theorem find_natural_number_pairs (a b q : ℕ) : 
  (a ∣ b^2 ∧ b ∣ a^2 ∧ (a + 1) ∣ (b^2 + 1)) ↔ 
  ((a = q^2 ∧ b = q) ∨ 
   (a = q^2 ∧ b = q^3) ∨ 
   (a = (q^2 - 1) * q^2 ∧ b = q * (q^2 - 1)^2)) :=
by
  sorry

end NUMINAMATH_GPT_find_natural_number_pairs_l620_62052


namespace NUMINAMATH_GPT_arithmetic_sum_property_l620_62010

variable {a : ℕ → ℤ} -- declare the sequence as a sequence of integers

-- Define the condition of the arithmetic sequence
def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∀ n, a (n + 1) - a n = a 2 - a 1

-- Given condition: sum of specific terms in the sequence equals 400
def sum_condition (a : ℕ → ℤ) : Prop :=
  a 3 + a 4 + a 5 + a 6 + a 7 = 400

-- The goal: if the sum_condition holds, then a_2 + a_8 = 160
theorem arithmetic_sum_property
  (h_sum : sum_condition a)
  (h_arith : arithmetic_sequence a) :
  a 2 + a 8 = 160 := by
  sorry

end NUMINAMATH_GPT_arithmetic_sum_property_l620_62010


namespace NUMINAMATH_GPT_geometric_sequence_find_a_n_l620_62049

variable {n m p : ℕ}
variable {a : ℕ → ℕ}
variable {S : ℕ → ℕ}

-- Given conditions
axiom h1 : ∀ n, 2 * S (n + 1) - 3 * S n = 2 * a 1
axiom h2 : a 1 ≠ 0
axiom h3 : ∀ n, S (n + 1) = S n + a (n + 1)

-- Part (1)
theorem geometric_sequence : ∃ r, ∀ n, a (n + 1) = r * a n :=
sorry

-- Part (2)
axiom p_geq_3 : 3 ≤ p
axiom a1_pos : 0 < a 1
axiom a_p_pos : 0 < a p
axiom constraint1 : a 1 ≥ m ^ (p - 1)
axiom constraint2 : a p ≤ (m + 1) ^ (p - 1)

theorem find_a_n : ∀ n, a n = 2 ^ (p - 1) * (3 / 2) ^ (n - 1) :=
sorry

end NUMINAMATH_GPT_geometric_sequence_find_a_n_l620_62049


namespace NUMINAMATH_GPT_ratio_of_areas_of_concentric_circles_eq_9_over_4_l620_62044

theorem ratio_of_areas_of_concentric_circles_eq_9_over_4
  (C1 C2 : ℝ)
  (h1 : ∃ Q : ℝ, true) -- Existence of point Q
  (h2 : (30 / 360) * C1 = (45 / 360) * C2) -- Arcs formed by 30-degree and 45-degree angles are equal in length
  : (π * (C1 / (2 * π))^2) / (π * (C2 / (2 * π))^2) = 9 / 4 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_areas_of_concentric_circles_eq_9_over_4_l620_62044


namespace NUMINAMATH_GPT_largest_whole_number_for_inequality_l620_62030

theorem largest_whole_number_for_inequality :
  ∀ n : ℕ, (1 : ℝ) / 4 + (n : ℝ) / 6 < 3 / 2 → n ≤ 7 :=
by
  admit  -- skip the proof

end NUMINAMATH_GPT_largest_whole_number_for_inequality_l620_62030


namespace NUMINAMATH_GPT_estimate_total_fish_l620_62092

theorem estimate_total_fish (m n k : ℕ) (hk : k ≠ 0) (hm : m ≠ 0) (hn : n ≠ 0):
  ∃ x : ℕ, x = (m * n) / k :=
by
  sorry

end NUMINAMATH_GPT_estimate_total_fish_l620_62092


namespace NUMINAMATH_GPT_water_difference_l620_62006

variables (S H : ℝ)

theorem water_difference 
  (h_diff_after : S - 0.43 - (H + 0.43) = 0.88)
  (h_seungmin_more : S > H) :
  S - H = 1.74 :=
by
  sorry

end NUMINAMATH_GPT_water_difference_l620_62006


namespace NUMINAMATH_GPT_evaluate_expression_l620_62089

theorem evaluate_expression (a b : ℕ) :
  a = 3 ^ 1006 →
  b = 7 ^ 1007 →
  (a + b)^2 - (a - b)^2 = 42 * 10^x :=
by
  intro h1 h2
  sorry

end NUMINAMATH_GPT_evaluate_expression_l620_62089


namespace NUMINAMATH_GPT_find_first_number_in_second_set_l620_62060

theorem find_first_number_in_second_set: 
  ∃ x: ℕ, (20 + 40 + 60) / 3 = (x + 80 + 15) / 3 + 5 ∧ x = 10 :=
by
  sorry

end NUMINAMATH_GPT_find_first_number_in_second_set_l620_62060


namespace NUMINAMATH_GPT_log3_infinite_nested_l620_62051

theorem log3_infinite_nested (x : ℝ) (h : x = Real.logb 3 (64 + x)) : x = 4 :=
by
  sorry

end NUMINAMATH_GPT_log3_infinite_nested_l620_62051


namespace NUMINAMATH_GPT_intersection_point_of_lines_l620_62086

theorem intersection_point_of_lines :
  ∃ (x y : ℚ), (2 * y = 3 * x - 6) ∧ (x + 5 * y = 10) ∧ (x = 50 / 17) ∧ (y = 24 / 17) :=
by
  sorry

end NUMINAMATH_GPT_intersection_point_of_lines_l620_62086


namespace NUMINAMATH_GPT_min_value_of_expression_l620_62012

noncomputable def f (x : ℝ) : ℝ :=
  2 / x + 9 / (1 - 2 * x)

theorem min_value_of_expression (x : ℝ) (h1 : 0 < x) (h2 : x < 1 / 2) : ∃ m, f x = m ∧ m = 25 :=
by
  sorry

end NUMINAMATH_GPT_min_value_of_expression_l620_62012


namespace NUMINAMATH_GPT_cylindrical_to_rectangular_multiplied_l620_62031

theorem cylindrical_to_rectangular_multiplied :
  let r := 7
  let θ := Real.pi / 4
  let z := -3
  let x := r * Real.cos θ
  let y := r * Real.sin θ
  (2 * x, 2 * y, 2 * z) = (7 * Real.sqrt 2, 7 * Real.sqrt 2, -6) := 
by
  sorry

end NUMINAMATH_GPT_cylindrical_to_rectangular_multiplied_l620_62031


namespace NUMINAMATH_GPT_rhombus_area_3cm_45deg_l620_62017

noncomputable def rhombusArea (a : ℝ) (theta : ℝ) : ℝ :=
  a * (a * Real.sin theta)

theorem rhombus_area_3cm_45deg :
  rhombusArea 3 (Real.pi / 4) = 9 * Real.sqrt 2 / 2 := 
by
  sorry

end NUMINAMATH_GPT_rhombus_area_3cm_45deg_l620_62017


namespace NUMINAMATH_GPT_tan_shift_symmetric_l620_62042

theorem tan_shift_symmetric :
  let f (x : ℝ) := Real.tan (2 * x + Real.pi / 6)
  let g (x : ℝ) := f (x + Real.pi / 6)
  g (Real.pi / 4) = 0 ∧ ∀ x, g (Real.pi / 2 - x) = -g (Real.pi / 2 + x) :=
by
  sorry

end NUMINAMATH_GPT_tan_shift_symmetric_l620_62042


namespace NUMINAMATH_GPT_total_recess_correct_l620_62035

-- Definitions based on the conditions
def base_recess : Int := 20
def recess_for_A (n : Int) : Int := n * 2
def recess_for_B (n : Int) : Int := n * 1
def recess_for_C (n : Int) : Int := n * 0
def recess_for_D (n : Int) : Int := -n * 1

def total_recess (a b c d : Int) : Int :=
  base_recess + recess_for_A a + recess_for_B b + recess_for_C c + recess_for_D d

-- The proof statement originally there would use these inputs
theorem total_recess_correct : total_recess 10 12 14 5 = 47 := by
  sorry

end NUMINAMATH_GPT_total_recess_correct_l620_62035


namespace NUMINAMATH_GPT_feet_per_inch_of_model_l620_62033

def height_of_statue := 75 -- in feet
def height_of_model := 5 -- in inches

theorem feet_per_inch_of_model : (height_of_statue / height_of_model) = 15 :=
by
  sorry

end NUMINAMATH_GPT_feet_per_inch_of_model_l620_62033


namespace NUMINAMATH_GPT_ratio_of_points_l620_62048

def Noa_points : ℕ := 30
def total_points : ℕ := 90

theorem ratio_of_points (Phillip_points : ℕ) (h1 : Phillip_points = 2 * Noa_points) (h2 : Noa_points + Phillip_points = total_points) : Phillip_points / Noa_points = 2 := 
by
  intros
  sorry

end NUMINAMATH_GPT_ratio_of_points_l620_62048


namespace NUMINAMATH_GPT_hyperbola_equation_sum_of_slopes_l620_62076

noncomputable def a : ℝ := 1
noncomputable def b : ℝ := Real.sqrt 3

theorem hyperbola_equation :
  ∀ (a b : ℝ) (H1 : a > 0) (H2 : b > 0) (H3 : (2^2) = a^2 + b^2)
    (H4 : ∀ (x₀ y₀ : ℝ), (x₀ ≠ -a) ∧ (x₀ ≠ a) → (y₀^2 = (b^2 / a^2) * (x₀^2 - a^2)) ∧ ((y₀ / (x₀ + a) * y₀ / (x₀ - a)) = 3)),
  (∀ x y : ℝ, (x^2 / a^2 - y^2 / b^2 = 1) → (x^2 - y^2 / 3 = 1)) :=
by
  intros a b H1 H2 H3 H4 x y Hxy
  sorry

theorem sum_of_slopes (m n : ℝ) (H1 : m < 1) :
  ∀ (k1 k2 : ℝ) (H2 : A ≠ B) (H3 : ((k1 ≠ k2) ∧ (1 + k1^2) / (3 - k1^2) = (1 + k2^2) / (3 - k2^2))),
  k1 + k2 = 0 :=
by
  intros k1 k2 H2 H3
  exact sorry

end NUMINAMATH_GPT_hyperbola_equation_sum_of_slopes_l620_62076


namespace NUMINAMATH_GPT_expression_evaluation_l620_62050

theorem expression_evaluation (m n : ℤ) (h : m * n = m + 3) : 2 * m * n + 3 * m - 5 * m * n - 10 = -19 := 
by 
  sorry

end NUMINAMATH_GPT_expression_evaluation_l620_62050


namespace NUMINAMATH_GPT_find_y_l620_62071

theorem find_y (y : ℤ) (h : (15 + 24 + y) / 3 = 23) : y = 30 :=
by
  sorry

end NUMINAMATH_GPT_find_y_l620_62071


namespace NUMINAMATH_GPT_find_loss_percentage_l620_62056

theorem find_loss_percentage (W : ℝ) (profit_percentage : ℝ) (remaining_percentage : ℝ)
  (overall_loss : ℝ) (stock_worth : ℝ) (L : ℝ) :
  W = 12499.99 →
  profit_percentage = 0.20 →
  remaining_percentage = 0.80 →
  overall_loss = -500 →
  0.04 * W - (L / 100) * (remaining_percentage * W) = overall_loss →
  L = 10 :=
by
  intro hW hprofit_percentage hremaining_percentage hoverall_loss heq
  -- We'll provide the proof here
  sorry

end NUMINAMATH_GPT_find_loss_percentage_l620_62056


namespace NUMINAMATH_GPT_train_length_l620_62002

-- Definitions and conditions based on the problem
def time : ℝ := 28.997680185585153
def bridge_length : ℝ := 150
def train_speed : ℝ := 10

-- The theorem to prove
theorem train_length : (train_speed * time) - bridge_length = 139.97680185585153 :=
by
  sorry

end NUMINAMATH_GPT_train_length_l620_62002


namespace NUMINAMATH_GPT_triangle_side_c_l620_62028

variable {A B C : ℝ} -- Angles of the triangle
variable {a b c : ℝ} -- Sides opposite to the respective angles

-- Conditions given
variable (h1 : Real.tan A = 2 * Real.tan B)
variable (h2 : a^2 - b^2 = (1 / 3) * c)

-- The proof problem
theorem triangle_side_c (h1 : Real.tan A = 2 * Real.tan B) (h2 : a^2 - b^2 = (1 / 3) * c) : c = 1 :=
by sorry

end NUMINAMATH_GPT_triangle_side_c_l620_62028


namespace NUMINAMATH_GPT_cube_sum_of_quadratic_roots_ratio_sum_of_quadratic_roots_l620_62091

theorem cube_sum_of_quadratic_roots (x₁ x₂ : ℝ) (h : x₁^2 - 3 * x₁ + 1 = 0) (h' : x₂^2 - 3 * x₂ + 1 = 0) :
  x₁^3 + x₂^3 = 18 :=
sorry

theorem ratio_sum_of_quadratic_roots (x₁ x₂ : ℝ) (h : x₁^2 - 3 * x₁ + 1 = 0) (h' : x₂^2 - 3 * x₂ + 1 = 0) :
  (x₂ / x₁) + (x₁ / x₂) = 7 :=
sorry

end NUMINAMATH_GPT_cube_sum_of_quadratic_roots_ratio_sum_of_quadratic_roots_l620_62091


namespace NUMINAMATH_GPT_blue_balls_taken_out_l620_62087

theorem blue_balls_taken_out :
  ∃ x : ℕ, (0 ≤ x ∧ x ≤ 7) ∧ (7 - x) / (15 - x) = 1 / 3 ∧ x = 3 :=
sorry

end NUMINAMATH_GPT_blue_balls_taken_out_l620_62087


namespace NUMINAMATH_GPT_quad_root_magnitude_l620_62065

theorem quad_root_magnitude (m : ℝ) :
  (∃ x : ℝ, x^2 - x + m^2 - 4 = 0 ∧ x = 1) → m = 2 ∨ m = -2 :=
by
  sorry

end NUMINAMATH_GPT_quad_root_magnitude_l620_62065


namespace NUMINAMATH_GPT_intersection_line_exists_unique_l620_62011

universe u

noncomputable section

structure Point (α : Type u) :=
(x y z : α)

structure Line (α : Type u) :=
(dir point : Point α)

variables {α : Type u} [Field α]

-- Define skew lines conditions
def skew_lines (l1 l2 : Line α) : Prop :=
¬ ∃ p : Point α, ∃ t1 t2 : α, 
  l1.point = p ∧ l1.dir ≠ (Point.mk 0 0 0) ∧ l2.point = p ∧ l2.dir ≠ (Point.mk 0 0 0) ∧
  l1.dir.x * t1 = l2.dir.x * t2 ∧
  l1.dir.y * t1 = l2.dir.y * t2 ∧
  l1.dir.z * t1 = l2.dir.z * t2

-- Define a point not on the lines
def point_not_on_lines (p : Point α) (l1 l2 : Line α) : Prop :=
  (∀ t1 : α, p ≠ Point.mk (l1.point.x + l1.dir.x * t1) (l1.point.y + l1.dir.y * t1) (l1.point.z + l1.dir.z * t1))
  ∧
  (∀ t2 : α, p ≠ Point.mk (l2.point.x + l2.dir.x * t2) (l2.point.y + l2.dir.y * t2) (l2.point.z + l2.dir.z * t2))

-- Main theorem: existence and typical uniqueness of the intersection line
theorem intersection_line_exists_unique {l1 l2 : Line α} {O : Point α}
  (h_skew : skew_lines l1 l2) (h_point_not_on_lines : point_not_on_lines O l1 l2) :
  ∃! l : Line α, l.point = O ∧ (
    ∃ t1 : α, ∃ t2 : α,
    Point.mk (O.x + l.dir.x * t1) (O.y + l.dir.y * t1) (O.z + l.dir.z * t1) = 
    Point.mk (l1.point.x + l1.dir.x * t1) (l1.point.y + l1.dir.y * t1) (l1.point.z + l1.dir.z * t1) ∧
    Point.mk (O.x + l.dir.x * t2) (O.y + l.dir.x * t2) (O.z + l.dir.z * t2) = 
    Point.mk (l2.point.x + l2.dir.x * t2) (l2.point.y + l2.dir.y * t2) (l2.point.z + l2.dir.z * t2)
  ) :=
by
  sorry

end NUMINAMATH_GPT_intersection_line_exists_unique_l620_62011


namespace NUMINAMATH_GPT_gcd_1234_2047_l620_62047

theorem gcd_1234_2047 : Nat.gcd 1234 2047 = 1 :=
by sorry

end NUMINAMATH_GPT_gcd_1234_2047_l620_62047


namespace NUMINAMATH_GPT_ellipse_value_l620_62063

noncomputable def a_c_ratio (a c : ℝ) : ℝ :=
  (a + c) / (a - c)

theorem ellipse_value (a b c : ℝ) 
  (h1 : a^2 = b^2 + c^2) 
  (h2 : a^2 + b^2 - 3 * c^2 = 0) :
  a_c_ratio a c = 3 + 2 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_ellipse_value_l620_62063


namespace NUMINAMATH_GPT_probability_of_black_given_not_white_l620_62098

variable (total_balls white_balls black_balls red_balls : ℕ)
variable (ball_is_not_white : Prop)

theorem probability_of_black_given_not_white 
  (h1 : total_balls = 10)
  (h2 : white_balls = 5)
  (h3 : black_balls = 3)
  (h4 : red_balls = 2)
  (h5 : ball_is_not_white) :
  (3 : ℚ) / 5 = (black_balls : ℚ) / (total_balls - white_balls) :=
by
  simp only [h1, h2, h3, h4]
  sorry

end NUMINAMATH_GPT_probability_of_black_given_not_white_l620_62098


namespace NUMINAMATH_GPT_range_of_a_minus_b_l620_62053

theorem range_of_a_minus_b (a b : ℝ) (h1 : -1 < a) (h2 : a < b) (h3 : b < 2) : -3 < a - b ∧ a - b < 0 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_minus_b_l620_62053


namespace NUMINAMATH_GPT_vector_addition_correct_l620_62094

def a : ℝ × ℝ := (-1, 6)
def b : ℝ × ℝ := (3, -2)
def c : ℝ × ℝ := (2, 4)

theorem vector_addition_correct : a + b = c := by
  sorry

end NUMINAMATH_GPT_vector_addition_correct_l620_62094


namespace NUMINAMATH_GPT_diagram_is_knowledge_structure_l620_62009

inductive DiagramType
| ProgramFlowchart
| ProcessFlowchart
| KnowledgeStructureDiagram
| OrganizationalStructureDiagram

axiom given_diagram : DiagramType
axiom diagram_is_one_of_them : 
  given_diagram = DiagramType.ProgramFlowchart ∨ 
  given_diagram = DiagramType.ProcessFlowchart ∨ 
  given_diagram = DiagramType.KnowledgeStructureDiagram ∨ 
  given_diagram = DiagramType.OrganizationalStructureDiagram

theorem diagram_is_knowledge_structure :
  given_diagram = DiagramType.KnowledgeStructureDiagram :=
sorry

end NUMINAMATH_GPT_diagram_is_knowledge_structure_l620_62009


namespace NUMINAMATH_GPT_angle_in_second_quadrant_l620_62061

-- Definitions of conditions
def sin2_pos : Prop := Real.sin 2 > 0
def cos2_neg : Prop := Real.cos 2 < 0

-- Statement of the problem
theorem angle_in_second_quadrant (h1 : sin2_pos) (h2 : cos2_neg) : 
    (∃ α, 0 < α ∧ α < π ∧ P = (Real.sin α, Real.cos α)) :=
by
  sorry

end NUMINAMATH_GPT_angle_in_second_quadrant_l620_62061


namespace NUMINAMATH_GPT_total_weight_lifted_l620_62041

-- Given definitions from the conditions
def weight_left_hand : ℕ := 10
def weight_right_hand : ℕ := 10

-- The proof problem statement
theorem total_weight_lifted : weight_left_hand + weight_right_hand = 20 := 
by 
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_total_weight_lifted_l620_62041


namespace NUMINAMATH_GPT_min_a_squared_plus_b_squared_l620_62023

theorem min_a_squared_plus_b_squared (a b : ℝ) (h : ∃ x : ℝ, x^4 + a * x^3 + b * x^2 + a * x + 1 = 0) : a^2 + b^2 ≥ 4 / 5 := 
sorry

end NUMINAMATH_GPT_min_a_squared_plus_b_squared_l620_62023


namespace NUMINAMATH_GPT_multiplication_in_A_l620_62043

def A : Set ℤ :=
  {x | ∃ a b : ℤ, x = a^2 + b^2}

theorem multiplication_in_A (x1 x2 : ℤ) (h1 : x1 ∈ A) (h2 : x2 ∈ A) :
  x1 * x2 ∈ A :=
sorry

end NUMINAMATH_GPT_multiplication_in_A_l620_62043


namespace NUMINAMATH_GPT_christian_sue_need_more_money_l620_62019

-- Definition of initial amounts
def christian_initial := 5
def sue_initial := 7

-- Definition of earnings from activities
def christian_per_yard := 5
def christian_yards := 4
def sue_per_dog := 2
def sue_dogs := 6

-- Definition of perfume cost
def perfume_cost := 50

-- Theorem statement for the math problem
theorem christian_sue_need_more_money :
  let christian_total := christian_initial + (christian_per_yard * christian_yards)
  let sue_total := sue_initial + (sue_per_dog * sue_dogs)
  let total_money := christian_total + sue_total
  total_money < perfume_cost → perfume_cost - total_money = 6 :=
by 
  intros
  let christian_total := christian_initial + (christian_per_yard * christian_yards)
  let sue_total := sue_initial + (sue_per_dog * sue_dogs)
  let total_money := christian_total + sue_total
  sorry

end NUMINAMATH_GPT_christian_sue_need_more_money_l620_62019


namespace NUMINAMATH_GPT_max_sin_angle_F1PF2_on_ellipse_l620_62000

theorem max_sin_angle_F1PF2_on_ellipse
  (x y : ℝ)
  (P : ℝ × ℝ)
  (F1 F2 : ℝ × ℝ)
  (h : P ∈ {Q | Q.1^2 / 9 + Q.2^2 / 5 = 1})
  (F1_is_focus : F1 = (-2, 0))
  (F2_is_focus : F2 = (2, 0)) :
  ∃ sin_max, sin_max = 4 * Real.sqrt 5 / 9 := 
sorry

end NUMINAMATH_GPT_max_sin_angle_F1PF2_on_ellipse_l620_62000


namespace NUMINAMATH_GPT_train_speed_l620_62029

theorem train_speed 
  (t1 : ℝ) (t2 : ℝ) (L : ℝ) (v : ℝ) 
  (h1 : t1 = 12) 
  (h2 : t2 = 44) 
  (h3 : L = v * 12)
  (h4 : L + 320 = v * 44) : 
  (v * 3.6 = 36) :=
by
  sorry

end NUMINAMATH_GPT_train_speed_l620_62029


namespace NUMINAMATH_GPT_unique_lottery_ticket_number_l620_62020

noncomputable def five_digit_sum_to_age (ticket : ℕ) (neighbor_age : ℕ) := 
  (ticket >= 10000 ∧ ticket <= 99999) ∧ 
  (neighbor_age = 5 * ((ticket / 10000) + (ticket % 10000 / 1000) + 
                        (ticket % 1000 / 100) + (ticket % 100 / 10) + 
                        (ticket % 10)))

theorem unique_lottery_ticket_number {ticket : ℕ} {neighbor_age : ℕ} 
    (h : five_digit_sum_to_age ticket neighbor_age) 
    (unique_solution : ∀ ticket1 ticket2, 
                        five_digit_sum_to_age ticket1 neighbor_age → 
                        five_digit_sum_to_age ticket2 neighbor_age → 
                        ticket1 = ticket2) : 
  ticket = 99999 :=
  sorry

end NUMINAMATH_GPT_unique_lottery_ticket_number_l620_62020


namespace NUMINAMATH_GPT_percentage_reduction_l620_62093

theorem percentage_reduction (y x z p q : ℝ) (hy : y ≠ 0) (h1 : x = y - 10) (h2 : z = y - 20) :
  p = 1000 / y ∧ q = 2000 / y := by
  sorry

end NUMINAMATH_GPT_percentage_reduction_l620_62093


namespace NUMINAMATH_GPT_polyhedron_edges_vertices_l620_62008

theorem polyhedron_edges_vertices (F : ℕ) (triangular_faces : Prop) (hF : F = 20) : ∃ S A : ℕ, S = 12 ∧ A = 30 :=
by
  -- stating the problem conditions and desired conclusion
  sorry

end NUMINAMATH_GPT_polyhedron_edges_vertices_l620_62008


namespace NUMINAMATH_GPT_find_sale_month4_l620_62084

-- Define sales for each month
def sale_month1 : ℕ := 5400
def sale_month2 : ℕ := 9000
def sale_month3 : ℕ := 6300
def sale_month5 : ℕ := 4500
def sale_month6 : ℕ := 1200
def avg_sale_per_month : ℕ := 5600

-- Define the total number of months
def num_months : ℕ := 6

-- Define the expression for total sales required
def total_sales_required : ℕ := avg_sale_per_month * num_months

-- Define the expression for total known sales
def total_known_sales : ℕ := sale_month1 + sale_month2 + sale_month3 + sale_month5 + sale_month6

-- State and prove the theorem:
theorem find_sale_month4 : sale_month1 = 5400 → sale_month2 = 9000 → sale_month3 = 6300 → 
                            sale_month5 = 4500 → sale_month6 = 1200 → avg_sale_per_month = 5600 →
                            num_months = 6 → (total_sales_required - total_known_sales = 8200) := 
by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end NUMINAMATH_GPT_find_sale_month4_l620_62084


namespace NUMINAMATH_GPT_absolute_value_bound_l620_62038

theorem absolute_value_bound (x : ℝ) (hx : |x| ≤ 2) : |3 * x - x^3| ≤ 2 := 
by
  sorry

end NUMINAMATH_GPT_absolute_value_bound_l620_62038


namespace NUMINAMATH_GPT_number_of_third_year_students_to_sample_l620_62013

theorem number_of_third_year_students_to_sample
    (total_students : ℕ)
    (first_year_students : ℕ)
    (second_year_students : ℕ)
    (third_year_students : ℕ)
    (total_to_sample : ℕ)
    (h_total : total_students = 1200)
    (h_first : first_year_students = 480)
    (h_second : second_year_students = 420)
    (h_third : third_year_students = 300)
    (h_sample : total_to_sample = 100) :
    third_year_students * total_to_sample / total_students = 25 :=
by
  sorry

end NUMINAMATH_GPT_number_of_third_year_students_to_sample_l620_62013


namespace NUMINAMATH_GPT_tan_of_obtuse_angle_l620_62018

theorem tan_of_obtuse_angle (α : ℝ) (h_cos : Real.cos α = -1/2) (h_obtuse : π/2 < α ∧ α < π) :
  Real.tan α = -Real.sqrt 3 :=
sorry

end NUMINAMATH_GPT_tan_of_obtuse_angle_l620_62018


namespace NUMINAMATH_GPT_sin_150_eq_half_l620_62016

noncomputable def calculate_sin_150 : ℝ := Real.sin (150 * Real.pi / 180)

theorem sin_150_eq_half : calculate_sin_150 = 1 / 2 :=
by
  -- We would include the detailed steps if not skipping the proof as per the instruction.
  sorry

end NUMINAMATH_GPT_sin_150_eq_half_l620_62016


namespace NUMINAMATH_GPT_polynomial_simplify_l620_62082

theorem polynomial_simplify (x : ℝ) :
  (2*x^5 + 3*x^3 - 5*x^2 + 8*x - 6) + (-6*x^5 + x^3 + 4*x^2 - 8*x + 7) = -4*x^5 + 4*x^3 - x^2 + 1 :=
  sorry

end NUMINAMATH_GPT_polynomial_simplify_l620_62082


namespace NUMINAMATH_GPT_union_condition_intersection_condition_l620_62088

def setA : Set ℝ := {x | x^2 - 5 * x + 6 ≤ 0}
def setB (a : ℝ) : Set ℝ := {x | a < x ∧ x ≤ 3}

theorem union_condition (a : ℝ) : setA ∪ setB a = setB a ↔ a < 2 := sorry

theorem intersection_condition (a : ℝ) : setA ∩ setB a = setB a ↔ a ≥ 2 := sorry

end NUMINAMATH_GPT_union_condition_intersection_condition_l620_62088


namespace NUMINAMATH_GPT_number_of_sheep_l620_62007

theorem number_of_sheep (S H : ℕ)
  (h1 : S / H = 4 / 7)
  (h2 : H * 230 = 12880) :
  S = 32 :=
by
  sorry

end NUMINAMATH_GPT_number_of_sheep_l620_62007


namespace NUMINAMATH_GPT_vehicle_speeds_l620_62021

theorem vehicle_speeds (d t: ℕ) (b_speed c_speed : ℕ) (h1 : d = 80) (h2 : c_speed = 3 * b_speed) (h3 : t = 3) (arrival_difference : ℕ) (h4 : arrival_difference = 1 / 3):
  b_speed = 20 ∧ c_speed = 60 :=
by
  sorry

end NUMINAMATH_GPT_vehicle_speeds_l620_62021


namespace NUMINAMATH_GPT_time_to_pass_trolley_l620_62034

/--
Conditions:
- Length of the train = 110 m
- Speed of the train = 60 km/hr
- Speed of the trolley = 12 km/hr

Prove that the time it takes for the train to pass the trolley completely is 5.5 seconds.
-/
theorem time_to_pass_trolley :
  ∀ (train_length : ℝ) (train_speed_kmh : ℝ) (trolley_speed_kmh : ℝ),
    train_length = 110 →
    train_speed_kmh = 60 →
    trolley_speed_kmh = 12 →
  train_length / ((train_speed_kmh + trolley_speed_kmh) * (1000 / 3600)) = 5.5 :=
by
  intros
  sorry

end NUMINAMATH_GPT_time_to_pass_trolley_l620_62034


namespace NUMINAMATH_GPT_point_on_line_l620_62081

theorem point_on_line (m : ℝ) : (2 = m - 1) → (m = 3) :=
by sorry

end NUMINAMATH_GPT_point_on_line_l620_62081


namespace NUMINAMATH_GPT_heat_production_example_l620_62045

noncomputable def heat_produced_by_current (R : ℝ) (I : ℝ → ℝ) (t1 t2 : ℝ) : ℝ :=
∫ (t : ℝ) in t1..t2, (I t)^2 * R

theorem heat_production_example :
  heat_produced_by_current 40 (λ t => 5 + 4 * t) 0 10 = 303750 :=
by
  sorry

end NUMINAMATH_GPT_heat_production_example_l620_62045


namespace NUMINAMATH_GPT_ship_total_distance_l620_62075

variables {v_r : ℝ} {t_total : ℝ} {a d : ℝ}

-- Given conditions
def conditions (v_r t_total a d : ℝ) :=
  v_r = 2 ∧ t_total = 3.2 ∧
  (∃ v : ℝ, ∀ t : ℝ, t = a/(v + v_r) + (a + d)/v + (a + 2*d)/(v - v_r)) 

-- The main statement to prove
theorem ship_total_distance (d_total : ℝ) :
  conditions 2 3.2 a d → d_total = 102 :=
by
  sorry

end NUMINAMATH_GPT_ship_total_distance_l620_62075


namespace NUMINAMATH_GPT_carol_allowance_problem_l620_62022

open Real

theorem carol_allowance_problem (w : ℝ) 
  (fixed_allowance : ℝ := 20) 
  (extra_earnings_per_week : ℝ := 22.5) 
  (total_money : ℝ := 425) :
  fixed_allowance * w + extra_earnings_per_week * w = total_money → w = 10 :=
by
  intro h
  -- Proof skipped
  sorry

end NUMINAMATH_GPT_carol_allowance_problem_l620_62022


namespace NUMINAMATH_GPT_even_square_minus_self_l620_62046

theorem even_square_minus_self (a : ℤ) : 2 ∣ (a^2 - a) :=
sorry

end NUMINAMATH_GPT_even_square_minus_self_l620_62046


namespace NUMINAMATH_GPT_triangle_formation_and_acuteness_l620_62083

variables {a b c : ℝ} {k n : ℕ}

theorem triangle_formation_and_acuteness (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (hn : 2 ≤ n) (hk : k < n) (hp : a^n + b^n = c^n) : 
  (a^k + b^k > c^k ∧ b^k + c^k > a^k ∧ c^k + a^k > b^k) ∧ (k < n / 2 → (a^k)^2 + (b^k)^2 > (c^k)^2) :=
sorry

end NUMINAMATH_GPT_triangle_formation_and_acuteness_l620_62083


namespace NUMINAMATH_GPT_sara_total_spent_l620_62054

-- Definitions based on the conditions
def ticket_price : ℝ := 10.62
def discount_rate : ℝ := 0.10
def rented_movie : ℝ := 1.59
def bought_movie : ℝ := 13.95
def snacks : ℝ := 7.50
def sales_tax_rate : ℝ := 0.05

-- Problem statement
theorem sara_total_spent : 
  let total_tickets := 2 * ticket_price
  let discount := total_tickets * discount_rate
  let discounted_tickets := total_tickets - discount
  let subtotal := discounted_tickets + rented_movie + bought_movie
  let sales_tax := subtotal * sales_tax_rate
  let total_with_tax := subtotal + sales_tax
  let total_amount := total_with_tax + snacks
  total_amount = 43.89 :=
by
  sorry

end NUMINAMATH_GPT_sara_total_spent_l620_62054


namespace NUMINAMATH_GPT_garden_table_ratio_l620_62003

theorem garden_table_ratio (x y : ℝ) (h₁ : x + y = 750) (h₂ : y = 250) : x / y = 2 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_garden_table_ratio_l620_62003


namespace NUMINAMATH_GPT_no_solution_fractional_eq_l620_62068

theorem no_solution_fractional_eq :
  ¬∃ x : ℝ, (1 - x) / (x - 2) = 1 / (2 - x) + 1 :=
by
  -- The proof is intentionally omitted.
  sorry

end NUMINAMATH_GPT_no_solution_fractional_eq_l620_62068


namespace NUMINAMATH_GPT_solve_quadratic_inequality_l620_62024

theorem solve_quadratic_inequality (a x : ℝ) :
  (x ^ 2 - (2 + a) * x + 2 * a < 0) ↔ 
  ((a < 2 ∧ a < x ∧ x < 2) ∨ (a = 2 ∧ false) ∨ 
   (a > 2 ∧ 2 < x ∧ x < a)) :=
by sorry

end NUMINAMATH_GPT_solve_quadratic_inequality_l620_62024


namespace NUMINAMATH_GPT_pow_addition_l620_62001

theorem pow_addition : (-2 : ℤ)^2 + (2 : ℤ)^2 = 8 :=
by
  sorry

end NUMINAMATH_GPT_pow_addition_l620_62001


namespace NUMINAMATH_GPT_milan_billed_minutes_l620_62014

theorem milan_billed_minutes (monthly_fee : ℝ) (cost_per_minute : ℝ) (total_bill : ℝ) (minutes : ℝ)
  (h1 : monthly_fee = 2)
  (h2 : cost_per_minute = 0.12)
  (h3 : total_bill = 23.36)
  (h4 : total_bill = monthly_fee + cost_per_minute * minutes)
  : minutes = 178 := 
sorry

end NUMINAMATH_GPT_milan_billed_minutes_l620_62014


namespace NUMINAMATH_GPT_total_medals_1996_l620_62058

variable (g s b : Nat)

theorem total_medals_1996 (h_g : g = 16) (h_s : s = 22) (h_b : b = 12) :
  g + s + b = 50 :=
by
  sorry

end NUMINAMATH_GPT_total_medals_1996_l620_62058


namespace NUMINAMATH_GPT_integral_solutions_l620_62070

/-- 
  Prove that the integral solutions to the equation 
  (m^2 - n^2)^2 = 1 + 16n are exactly (m, n) = (±1, 0), (±4, 3), (±4, 5). 
--/
theorem integral_solutions (m n : ℤ) :
  (m^2 - n^2)^2 = 1 + 16 * n ↔ (m = 1 ∧ n = 0) ∨ (m = -1 ∧ n = 0) ∨
                        (m = 4 ∧ n = 3) ∨ (m = -4 ∧ n = 3) ∨
                        (m = 4 ∧ n = 5) ∨ (m = -4 ∧ n = 5) :=
by
  sorry

end NUMINAMATH_GPT_integral_solutions_l620_62070


namespace NUMINAMATH_GPT_albert_brother_younger_l620_62039

variables (A B Y F M : ℕ)
variables (h1 : F = 48)
variables (h2 : M = 46)
variables (h3 : F - M = 4)
variables (h4 : Y = A - B)

theorem albert_brother_younger (h_cond : (F - M = 4) ∧ (F = 48) ∧ (M = 46) ∧ (Y = A - B)) : Y = 2 :=
by
  rcases h_cond with ⟨h_diff, h_father, h_mother, h_ages⟩
  -- Assuming that each step provided has correct assertive logic.
  sorry

end NUMINAMATH_GPT_albert_brother_younger_l620_62039


namespace NUMINAMATH_GPT_value_range_neg_x_squared_l620_62097

theorem value_range_neg_x_squared:
  (∀ y, (-9 ≤ y ∧ y ≤ 0) ↔ ∃ x, (-3 ≤ x ∧ x ≤ 1) ∧ y = -x^2) :=
by
  sorry

end NUMINAMATH_GPT_value_range_neg_x_squared_l620_62097


namespace NUMINAMATH_GPT_solutions_to_h_eq_1_l620_62066

noncomputable def h (x : ℝ) : ℝ :=
if x ≤ 0 then 5 * x + 10 else 3 * x - 5

theorem solutions_to_h_eq_1 : {x : ℝ | h x = 1} = {-9/5, 2} :=
by
  sorry

end NUMINAMATH_GPT_solutions_to_h_eq_1_l620_62066


namespace NUMINAMATH_GPT_football_field_area_l620_62004

-- Define the conditions
def fertilizer_spread : ℕ := 1200
def area_partial : ℕ := 3600
def fertilizer_partial : ℕ := 400

-- Define the expected result
def area_total : ℕ := 10800

-- Theorem to prove
theorem football_field_area :
  (fertilizer_spread / (fertilizer_partial / area_partial)) = area_total :=
by sorry

end NUMINAMATH_GPT_football_field_area_l620_62004


namespace NUMINAMATH_GPT_yellow_balls_count_l620_62078

theorem yellow_balls_count (x y z : ℕ) 
  (h1 : x + y + z = 68)
  (h2 : y = 2 * x)
  (h3 : 3 * z = 4 * y) : y = 24 :=
by {
  sorry
}

end NUMINAMATH_GPT_yellow_balls_count_l620_62078


namespace NUMINAMATH_GPT_sarah_must_solve_at_least_16_l620_62032

theorem sarah_must_solve_at_least_16
  (total_problems : ℕ)
  (problems_attempted : ℕ)
  (problems_unanswered : ℕ)
  (points_per_correct : ℕ)
  (points_per_unanswered : ℕ)
  (target_score : ℕ)
  (h1 : total_problems = 30)
  (h2 : points_per_correct = 7)
  (h3 : points_per_unanswered = 2)
  (h4 : problems_unanswered = 5)
  (h5 : problems_attempted = 25)
  (h6 : target_score = 120) :
  ∃ (correct_solved : ℕ), correct_solved ≥ 16 ∧ correct_solved ≤ problems_attempted ∧
    (correct_solved * points_per_correct) + (problems_unanswered * points_per_unanswered) ≥ target_score :=
by {
  sorry
}

end NUMINAMATH_GPT_sarah_must_solve_at_least_16_l620_62032


namespace NUMINAMATH_GPT_age_of_new_person_l620_62096

theorem age_of_new_person (T A : ℕ) (h1 : (T / 10 : ℤ) - 3 = (T - 40 + A) / 10) : A = 10 := 
sorry

end NUMINAMATH_GPT_age_of_new_person_l620_62096


namespace NUMINAMATH_GPT_abs_sum_le_abs_one_plus_mul_l620_62055

theorem abs_sum_le_abs_one_plus_mul {x y : ℝ} (hx : |x| ≤ 1) (hy : |y| ≤ 1) : 
  |x + y| ≤ |1 + x * y| :=
sorry

end NUMINAMATH_GPT_abs_sum_le_abs_one_plus_mul_l620_62055


namespace NUMINAMATH_GPT_find_n_l620_62074

-- Definitions based on the given conditions
def binomial_expectation (n : ℕ) (p : ℝ) : ℝ := n * p
def binomial_variance (n : ℕ) (p : ℝ) : ℝ := n * p * (1 - p)

-- The mathematically equivalent proof problem statement:
theorem find_n (n : ℕ) (p : ℝ) (h1 : binomial_expectation n p = 6) (h2 : binomial_variance n p = 3) : n = 12 :=
sorry

end NUMINAMATH_GPT_find_n_l620_62074
