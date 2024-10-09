import Mathlib

namespace number_of_boys_l341_34137

noncomputable def numGirls : Nat := 46
noncomputable def numGroups : Nat := 8
noncomputable def groupSize : Nat := 9
noncomputable def totalMembers : Nat := numGroups * groupSize
noncomputable def numBoys : Nat := totalMembers - numGirls

theorem number_of_boys :
  numBoys = 26 := by
  sorry

end number_of_boys_l341_34137


namespace cat_finishes_food_on_sunday_l341_34159

-- Define the constants and parameters
def daily_morning_consumption : ℚ := 2 / 5
def daily_evening_consumption : ℚ := 1 / 5
def total_food : ℕ := 8
def days_in_week : ℕ := 7

-- Define the total daily consumption
def total_daily_consumption : ℚ := daily_morning_consumption + daily_evening_consumption

-- Define the sum of consumptions over each day until the day when all food is consumed
def food_remaining_after_days (days : ℕ) : ℚ := total_food - days * total_daily_consumption

-- Proposition that the food is finished on Sunday
theorem cat_finishes_food_on_sunday :
  ∃ days : ℕ, (food_remaining_after_days days ≤ 0) ∧ days ≡ 7 [MOD days_in_week] :=
sorry

end cat_finishes_food_on_sunday_l341_34159


namespace verify_other_root_l341_34143

variable {a b c x : ℝ}

-- Given conditions
axiom distinct_non_zero_constants : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ a ≠ b ∧ b ≠ c ∧ c ≠ a

axiom root_two : a * 2^2 - (a + b + c) * 2 + (b + c) = 0

-- Function under test
noncomputable def other_root (a b c : ℝ) : ℝ :=
  (b + c - a) / a

-- The goal statement
theorem verify_other_root :
  ∀ (a b c : ℝ), (a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ a ≠ b ∧ b ≠ c ∧ c ≠ a) → (a * 2^2 - (a + b + c) * 2 + (b + c) = 0) → 
  (∀ x, (a * x^2 - (a + b + c) * x + (b + c) = 0) → (x = 2 ∨ x = (b + c - a) / a)) :=
by
  intros a b c h1 h2 x h3
  sorry

end verify_other_root_l341_34143


namespace everton_college_payment_l341_34103

theorem everton_college_payment :
  let num_sci_calculators := 20
  let num_graph_calculators := 25
  let price_sci_calculator := 10
  let price_graph_calculator := 57
  let total_payment := num_sci_calculators * price_sci_calculator + num_graph_calculators * price_graph_calculator
  total_payment = 1625 :=
by
  let num_sci_calculators := 20
  let num_graph_calculators := 25
  let price_sci_calculator := 10
  let price_graph_calculator := 57
  let total_payment := num_sci_calculators * price_sci_calculator + num_graph_calculators * price_graph_calculator
  sorry

end everton_college_payment_l341_34103


namespace car_arrives_first_and_earlier_l341_34104

-- Define the conditions
def total_intersections : ℕ := 11
def total_blocks : ℕ := 12
def green_time : ℕ := 3
def red_time : ℕ := 1
def car_block_time : ℕ := 1
def bus_block_time : ℕ := 2

-- Define the functions that compute the travel times
def car_travel_time (blocks : ℕ) : ℕ :=
  (blocks / 3) * (green_time + red_time) + (blocks % 3 * car_block_time)

def bus_travel_time (blocks : ℕ) : ℕ :=
  blocks * bus_block_time

-- Define the theorem to prove
theorem car_arrives_first_and_earlier :
  car_travel_time total_blocks < bus_travel_time total_blocks ∧
  bus_travel_time total_blocks - car_travel_time total_blocks = 9 := 
by
  sorry

end car_arrives_first_and_earlier_l341_34104


namespace part_I_extreme_values_part_II_three_distinct_real_roots_part_III_compare_sizes_l341_34162

noncomputable def f (x : ℝ) (p q : ℝ) : ℝ := (1 / 3) * x^3 + (1 / 2) * (p - 1) * x^2 + q * x

theorem part_I_extreme_values : 
  (∀ x, f x (-3) 3 = (1 / 3) * x^3 - 2 * x^2 + 3 * x) → 
  (f 1 (-3) 3 = f 3 (-3) 3) := 
sorry

theorem part_II_three_distinct_real_roots : 
  (∀ x, f x (-3) 3 = (1 / 3) * x^3 - 2 * x^2 + 3 * x) → 
  (∀ g : ℝ → ℝ, g x = f x (-3) 3 - 1 → 
  (∀ x, g x ≠ 0) → 
  ∃ a b c, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ g a = 0 ∧ g b = 0 ∧ g c = 0) :=
sorry

theorem part_III_compare_sizes (x1 x2 p a l q: ℝ) :
  f (x : ℝ) (-3) 3 = (1 / 3) * x^3 - 2 * x^2 + 3 * x → 
  x1 < x2 → 
  x2 - x1 > l → 
  x1 > a → 
  (a^2 + p * a + q) > x1 := 
sorry

end part_I_extreme_values_part_II_three_distinct_real_roots_part_III_compare_sizes_l341_34162


namespace functions_not_necessarily_equal_l341_34119

-- Define the domain and range
variables {α β : Type*}

-- Define two functions f and g with the same domain and range
variables (f g : α → β)

-- Lean statement for the given mathematical problem
theorem functions_not_necessarily_equal (h_domain : ∀ x : α, (∃ x : α, true))
  (h_range : ∀ y : β, (∃ y : β, true)) : ¬(f = g) :=
sorry

end functions_not_necessarily_equal_l341_34119


namespace avg_rest_students_l341_34133

/- Definitions based on conditions -/
def total_students : ℕ := 28
def students_scored_95 : ℕ := 4
def students_scored_0 : ℕ := 3
def avg_whole_class : ℚ := 47.32142857142857
def total_marks_95 : ℚ := students_scored_95 * 95
def total_marks_0 : ℚ := students_scored_0 * 0
def marks_whole_class : ℚ := total_students * avg_whole_class
def rest_students : ℕ := total_students - students_scored_95 - students_scored_0

/- Theorem to prove the average of the rest students given the conditions -/
theorem avg_rest_students : (total_marks_95 + total_marks_0 + rest_students * 45) = marks_whole_class :=
by
  sorry

end avg_rest_students_l341_34133


namespace max_erasers_l341_34174

theorem max_erasers (p n e : ℕ) (h₁ : p ≥ 1) (h₂ : n ≥ 1) (h₃ : e ≥ 1) (h₄ : 3 * p + 4 * n + 8 * e = 60) :
  e ≤ 5 :=
sorry

end max_erasers_l341_34174


namespace abs_neg_two_l341_34125

theorem abs_neg_two : abs (-2) = 2 :=
by
  sorry

end abs_neg_two_l341_34125


namespace tic_tac_toe_alex_wins_second_X_l341_34199

theorem tic_tac_toe_alex_wins_second_X :
  ∃ b : ℕ, b = 12 := 
sorry

end tic_tac_toe_alex_wins_second_X_l341_34199


namespace min_sum_of_grid_numbers_l341_34154

-- Definition of the 2x2 grid and the problem conditions
variables (a b c d : ℕ)
variables (a_pos : a > 0) (b_pos : b > 0) (c_pos : c > 0) (d_pos : d > 0)

-- Lean statement for the minimum sum proof problem
theorem min_sum_of_grid_numbers :
  a + b + c + d + a * b + c * d + a * c + b * d = 2015 → a + b + c + d = 88 :=
by
  sorry

end min_sum_of_grid_numbers_l341_34154


namespace difference_of_numbers_l341_34188

variables (x y : ℝ)

-- Definitions corresponding to the conditions
def sum_of_numbers (x y : ℝ) : Prop := x + y = 30
def product_of_numbers (x y : ℝ) : Prop := x * y = 200

-- The proof statement in Lean
theorem difference_of_numbers (x y : ℝ) 
  (h1: sum_of_numbers x y) 
  (h2: product_of_numbers x y) : x - y = 10 ∨ y - x = 10 :=
by
  sorry

end difference_of_numbers_l341_34188


namespace age_of_oldest_child_l341_34191

theorem age_of_oldest_child (a1 a2 a3 x : ℕ) (h1 : a1 = 5) (h2 : a2 = 7) (h3 : a3 = 10) (h_avg : (a1 + a2 + a3 + x) / 4 = 8) : x = 10 :=
by
  sorry

end age_of_oldest_child_l341_34191


namespace puppy_food_cost_l341_34134

theorem puppy_food_cost :
  let puppy_cost : ℕ := 10
  let days_in_week : ℕ := 7
  let total_number_of_weeks : ℕ := 3
  let cups_per_day : ℚ := 1 / 3
  let cups_per_bag : ℚ := 3.5
  let cost_per_bag : ℕ := 2
  let total_days := total_number_of_weeks * days_in_week
  let total_cups := total_days * cups_per_day
  let total_bags := total_cups / cups_per_bag
  let food_cost := total_bags * cost_per_bag
  let total_cost := puppy_cost + food_cost
  total_cost = 14 := by
  sorry

end puppy_food_cost_l341_34134


namespace number_of_members_in_league_l341_34173

-- Define the costs of the items considering the conditions
def sock_cost : ℕ := 6
def tshirt_cost : ℕ := sock_cost + 3
def shorts_cost : ℕ := sock_cost + 2

-- Define the total cost for one member
def total_cost_one_member : ℕ := 
  2 * (sock_cost + tshirt_cost + shorts_cost)

-- Given total expenditure
def total_expenditure : ℕ := 4860

-- Define the theorem to be proved
theorem number_of_members_in_league :
  total_expenditure / total_cost_one_member = 106 :=
by 
  sorry

end number_of_members_in_league_l341_34173


namespace avg_score_first_4_l341_34148

-- Definitions based on conditions
def average_score_all_7 : ℝ := 56
def total_matches : ℕ := 7
def average_score_last_3 : ℝ := 69.33333333333333
def matches_first : ℕ := 4
def matches_last : ℕ := 3

-- Calculation of total runs from average scores.
def total_runs_all_7 : ℝ := average_score_all_7 * total_matches
def total_runs_last_3 : ℝ := average_score_last_3 * matches_last

-- Total runs for the first 4 matches
def total_runs_first_4 : ℝ := total_runs_all_7 - total_runs_last_3

-- Prove the average score for the first 4 matches.
theorem avg_score_first_4 :
  (total_runs_first_4 / matches_first) = 46 := 
sorry

end avg_score_first_4_l341_34148


namespace geometric_sequence_reciprocals_sum_l341_34121

theorem geometric_sequence_reciprocals_sum :
  ∃ (a : ℕ → ℝ) (q : ℝ), 
    (a 1 = 2) ∧ 
    (a 1 + a 3 + a 5 = 14) ∧ 
    (∀ n : ℕ, a (n + 1) = a n * q) → 
      (1 / a 1 + 1 / a 3 + 1 / a 5 = 7 / 8) :=
sorry

end geometric_sequence_reciprocals_sum_l341_34121


namespace intersection_M_N_l341_34167

def M : Set ℝ := { x | -2 ≤ x ∧ x < 2 }
def N : Set ℝ := { x | x ≥ -2 }

theorem intersection_M_N : M ∩ N = { x | -2 ≤ x ∧ x < 2 } := by
  sorry

end intersection_M_N_l341_34167


namespace point_equal_distances_l341_34169

theorem point_equal_distances (x y : ℝ) (hx : y = x) (hxy : y - 4 = -x) (hline : x + y = 4) : x = 2 :=
by sorry

end point_equal_distances_l341_34169


namespace composite_for_all_n_greater_than_one_l341_34171

theorem composite_for_all_n_greater_than_one (n : ℕ) (h : n > 1) : ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ n^4 + 4^n = a * b :=
by
  sorry

end composite_for_all_n_greater_than_one_l341_34171


namespace maximize_revenue_l341_34152

-- Define the conditions
def price (p : ℝ) := p ≤ 30
def toys_sold (p : ℝ) : ℝ := 150 - 4 * p
def revenue (p : ℝ) := p * (toys_sold p)

-- State the theorem to solve the problem
theorem maximize_revenue : ∃ p : ℝ, price p ∧ 
  (∀ q : ℝ, price q → revenue q ≤ revenue p) ∧ p = 18.75 :=
by {
  sorry
}

end maximize_revenue_l341_34152


namespace factor_expression_l341_34101

theorem factor_expression (x : ℝ) : x * (x + 3) + 2 * (x + 3) = (x + 2) * (x + 3) :=
by
  sorry

end factor_expression_l341_34101


namespace extremum_range_l341_34147

noncomputable def f (a x : ℝ) : ℝ := x^3 + 2 * x^2 - a * x + 1

noncomputable def f_prime (a x : ℝ) : ℝ := 3 * x^2 + 4 * x - a

theorem extremum_range 
  (h : ∀ a : ℝ, (∃ (x : ℝ) (hx : -1 < x ∧ x < 1), f_prime a x = 0) → 
                (∀ x : ℝ, -1 < x ∧ x < 1 → f_prime a x ≠ 0)):
  ∀ a : ℝ, -1 < a ∧ a < 7 :=
sorry

end extremum_range_l341_34147


namespace find_ac_find_a_and_c_l341_34113

variables (A B C a b c : ℝ)

-- Condition: Angles A, B, C form an arithmetic sequence.
def arithmetic_sequence := 2 * B = A + C

-- Condition: Area of the triangle is sqrt(3)/2.
def area_triangle := (1/2) * a * c * (Real.sin B) = (Real.sqrt 3) / 2

-- Condition: b = sqrt(3)
def b_sqrt3 := b = Real.sqrt 3

-- Goal 1: To prove that ac = 2.
theorem find_ac (h1 : arithmetic_sequence A B C) (h2 : area_triangle a c B) : a * c = 2 :=
sorry

-- Goal 2: To prove a = 2 and c = 1 given the additional condition.
theorem find_a_and_c (h1 : arithmetic_sequence A B C) (h2 : area_triangle a c B) (h3 : b_sqrt3 b) (h4 : a > c) : a = 2 ∧ c = 1 :=
sorry

end find_ac_find_a_and_c_l341_34113


namespace marly_needs_3_bags_l341_34166

-- Definitions based on the problem conditions
def milk : ℕ := 2
def chicken_stock : ℕ := 3 * milk
def vegetables : ℕ := 1
def total_soup : ℕ := milk + chicken_stock + vegetables
def bag_capacity : ℕ := 3

-- The theorem to prove the number of bags required
theorem marly_needs_3_bags : total_soup / bag_capacity = 3 := 
sorry

end marly_needs_3_bags_l341_34166


namespace corresponding_angles_equal_l341_34116

-- Definition of corresponding angles (this should be previously defined, so here we assume it is just a predicate)
def CorrespondingAngles (a b : Angle) : Prop := sorry

-- The main theorem to be proven
theorem corresponding_angles_equal (a b : Angle) (h : CorrespondingAngles a b) : a = b := 
sorry

end corresponding_angles_equal_l341_34116


namespace middle_number_consecutive_even_l341_34105

theorem middle_number_consecutive_even (a b c : ℤ) 
  (h1 : a = b - 2) 
  (h2 : c = b + 2) 
  (h3 : a + b = 18) 
  (h4 : a + c = 22) 
  (h5 : b + c = 28) : 
  b = 11 :=
by sorry

end middle_number_consecutive_even_l341_34105


namespace geometric_sequence_ratio_l341_34190

-- Definitions and conditions from part a)
def q : ℚ := 1 / 2

def sum_of_first_n (a1 : ℚ) (n : ℕ) : ℚ :=
  a1 * (1 - q ^ n) / (1 - q)

def a_n (a1 : ℚ) (n : ℕ) : ℚ :=
  a1 * q ^ (n - 1)

-- Theorem representing the proof problem from part c)
theorem geometric_sequence_ratio (a1 : ℚ) : 
  (sum_of_first_n a1 4) / (a_n a1 3) = 15 / 2 := 
sorry

end geometric_sequence_ratio_l341_34190


namespace multiplication_result_l341_34185

theorem multiplication_result : 
  (500 * 2468 * 0.2468 * 100) = 30485120 :=
by
  sorry

end multiplication_result_l341_34185


namespace purely_imaginary_iff_l341_34122

theorem purely_imaginary_iff (a : ℝ) :
  (a^2 - a - 2 = 0 ∧ (|a - 1| - 1 ≠ 0)) ↔ a = -1 :=
by
  sorry

end purely_imaginary_iff_l341_34122


namespace taller_tree_height_l341_34164

-- Definitions and Variables
variables (h : ℝ)

-- Conditions as Definitions
def top_difference_condition := (h - 20) / h = 5 / 7

-- Proof Statement
theorem taller_tree_height (h : ℝ) (H : top_difference_condition h) : h = 70 := 
by {
  sorry
}

end taller_tree_height_l341_34164


namespace polyhedron_space_diagonals_l341_34142

theorem polyhedron_space_diagonals (V E F T P : ℕ) (total_pairs_of_vertices total_edges total_face_diagonals : ℕ)
  (hV : V = 30)
  (hE : E = 70)
  (hF : F = 40)
  (hT : T = 30)
  (hP : P = 10)
  (h_total_pairs_of_vertices : total_pairs_of_vertices = 30 * 29 / 2)
  (h_total_face_diagonals : total_face_diagonals = 5 * 10)
  :
  total_pairs_of_vertices - E - total_face_diagonals = 315 := 
by
  sorry

end polyhedron_space_diagonals_l341_34142


namespace ones_digit_of_largest_power_of_3_dividing_factorial_l341_34127

theorem ones_digit_of_largest_power_of_3_dividing_factorial (n : ℕ) (h : 27 = 3^3) : 
  (fun x => x % 10) (3^13) = 3 := by
  sorry

end ones_digit_of_largest_power_of_3_dividing_factorial_l341_34127


namespace insufficient_pharmacies_l341_34124

/-- Define the grid size and conditions --/
def grid_size : ℕ := 9
def total_intersections : ℕ := 100
def pharmacy_walking_distance : ℕ := 3
def number_of_pharmacies : ℕ := 12

/-- Prove that 12 pharmacies are not enough for the given conditions. --/
theorem insufficient_pharmacies : ¬ (number_of_pharmacies ≥ grid_size + 3) → 
  ∃(pharmacies : ℕ), pharmacies = number_of_pharmacies ∧
  ∀(x y : ℕ), (x ≤ grid_size ∧ y ≤ grid_size) → 
  ¬ exists (p : ℕ × ℕ), p.1 ≤ x + pharmacy_walking_distance ∧
  p.2 ≤ y + pharmacy_walking_distance ∧ 
  p.1 < total_intersections ∧ 
  p.2 < total_intersections := 
by sorry

end insufficient_pharmacies_l341_34124


namespace value_of_A_l341_34146

theorem value_of_A (A C : ℤ) (h₁ : 2 * A - C + 4 = 26) (h₂ : C = 6) : A = 14 :=
by sorry

end value_of_A_l341_34146


namespace curve_is_hyperbola_l341_34179

theorem curve_is_hyperbola (u : ℝ) (x y : ℝ) 
  (h1 : x = Real.cos u ^ 2)
  (h2 : y = Real.sin u ^ 4) : 
  ∃ (a b : ℝ), a ≠ 0 ∧  b ≠ 0 ∧ x / a ^ 2 - y / b ^ 2 = 1 := 
sorry

end curve_is_hyperbola_l341_34179


namespace cone_sphere_ratio_l341_34145

noncomputable def volume_of_sphere (r : ℝ) : ℝ :=
  (4 / 3) * Real.pi * r^3

noncomputable def volume_of_cone (r : ℝ) (h : ℝ) : ℝ :=
  (1 / 3) * Real.pi * (2 * r)^2 * h

theorem cone_sphere_ratio (r h : ℝ) (V_cone V_sphere : ℝ) (h_sphere : V_sphere = volume_of_sphere r)
  (h_cone : V_cone = volume_of_cone r h) (h_relation : V_cone = (1/3) * V_sphere) :
  (h / (2 * r) = 1 / 6) :=
by
  sorry

end cone_sphere_ratio_l341_34145


namespace company_production_l341_34158

theorem company_production (bottles_per_case number_of_cases total_bottles : ℕ)
  (h1 : bottles_per_case = 12)
  (h2 : number_of_cases = 10000)
  (h3 : total_bottles = number_of_cases * bottles_per_case) : 
  total_bottles = 120000 :=
by {
  -- Proof is omitted, add actual proof here
  sorry
}

end company_production_l341_34158


namespace number_difference_l341_34115

theorem number_difference:
  ∀ (number : ℝ), 0.30 * number = 63.0000000000001 →
  (3 / 7) * number - 0.40 * number = 6.00000000000006 := by
  sorry

end number_difference_l341_34115


namespace quadratic_j_value_l341_34187

theorem quadratic_j_value (a b c : ℝ) (h : a * (0 : ℝ)^2 + b * (0 : ℝ) + c = 5 * ((0 : ℝ) - 3)^2 + 15) :
  ∃ m j n, 4 * a * (0 : ℝ)^2 + 4 * b * (0 : ℝ) + 4 * c = m * ((0 : ℝ) - j)^2 + n ∧ j = 3 :=
by
  sorry

end quadratic_j_value_l341_34187


namespace slope_of_line_determined_by_solutions_l341_34149

theorem slope_of_line_determined_by_solutions :
  (∀ (x₁ y₁ x₂ y₂ : ℝ), (4 / x₁ + 6 / y₁ = 0) ∧ (4 / x₂ + 6 / y₂ = 0) →
    (y₂ - y₁) / (x₂ - x₁) = -3 / 2) :=
sorry

end slope_of_line_determined_by_solutions_l341_34149


namespace measure_of_U_is_120_l341_34130

variable {α β γ δ ε ζ : ℝ}
variable (h1 : α = γ) (h2 : α = ζ) (h3 : β + δ = 180) (h4 : ε + ζ = 180)

noncomputable def measure_of_U : ℝ :=
  let total_sum := 720
  have sum_of_angles : α + β + γ + δ + ζ + ε = total_sum := by
    sorry
  have subs_suppl_G_R : β + δ = 180 := h3
  have subs_suppl_E_U : ε + ζ = 180 := h4
  have congruent_F_I_U : α = γ ∧ α = ζ := ⟨h1, h2⟩
  let α : ℝ := sorry
  α

theorem measure_of_U_is_120 : measure_of_U h1 h2 h3 h4 = 120 :=
  sorry

end measure_of_U_is_120_l341_34130


namespace f_1_eq_2_f_6_plus_f_7_eq_15_f_2012_eq_3849_l341_34180

noncomputable def f : ℕ+ → ℕ+ := sorry

axiom f_properties (n : ℕ+) : f (f n) = 3 * n

axiom f_increasing (n : ℕ+) : f (n + 1) > f n

-- Proof for f(1)
theorem f_1_eq_2 : f 1 = 2 := 
by
sorry

-- Proof for f(6) + f(7)
theorem f_6_plus_f_7_eq_15 : f 6 + f 7 = 15 := 
by
sorry

-- Proof for f(2012)
theorem f_2012_eq_3849 : f 2012 = 3849 := 
by
sorry

end f_1_eq_2_f_6_plus_f_7_eq_15_f_2012_eq_3849_l341_34180


namespace point_symmetric_about_y_axis_l341_34182

theorem point_symmetric_about_y_axis (A B : ℝ × ℝ) 
  (hA : A = (1, -2)) 
  (hSym : B = (-A.1, A.2)) :
  B = (-1, -2) := 
by 
  sorry

end point_symmetric_about_y_axis_l341_34182


namespace xy_value_l341_34106

theorem xy_value (x y : ℝ) (h : x * (x - y) = x^2 - 6) : x * y = 6 := 
by 
  sorry

end xy_value_l341_34106


namespace part1_distance_part2_equation_l341_34161

noncomputable section

-- Define the conditions for Part 1
def hyperbola_C1 (x y : ℝ) : Prop := (x^2 / 4) - (y^2 / 12) = 1

-- Define the point M(3, t) existing on hyperbola C₁
def point_on_hyperbola_C1 (t : ℝ) : Prop := hyperbola_C1 3 t

-- Define the right focus of hyperbola C1
def right_focus_C1 : ℝ × ℝ := (4, 0)

-- Part 1: Distance from point M to the right focus
theorem part1_distance (t : ℝ) (h : point_on_hyperbola_C1 t) :  
  let distance := Real.sqrt ((3 - 4)^2 + (t - 0)^2)
  distance = 4 := sorry

-- Define the conditions for Part 2
def hyperbola_C2 (x y : ℝ) (m : ℝ) : Prop := (x^2 / 4) - (y^2 / 12) = m

-- Define the point (-3, 2√6) existing on hyperbola C₂
def point_on_hyperbola_C2 (m : ℝ) : Prop := hyperbola_C2 (-3) (2 * Real.sqrt 6) m

-- Part 2: The standard equation of hyperbola C₂
theorem part2_equation (h : point_on_hyperbola_C2 (1/4)) : 
  ∀ (x y : ℝ), hyperbola_C2 x y (1/4) ↔ (x^2 - (y^2 / 3) = 1) := sorry

end part1_distance_part2_equation_l341_34161


namespace max_value_expression_l341_34195

noncomputable def factorize_15000 := 2^3 * 3 * 5^4

theorem max_value_expression (x y : ℕ) (h1 : 6 * x^2 - 5 * x * y + y^2 = 0) (h2 : x ∣ factorize_15000) : 
  2 * x + 3 * y ≤ 60000 := sorry

end max_value_expression_l341_34195


namespace range_of_m_l341_34112

noncomputable def f (x m : ℝ) : ℝ := (1 / 4) * x^4 - (2 / 3) * x^3 + m

theorem range_of_m (m : ℝ) : (∀ x : ℝ, f x m + (1 / 3) ≥ 0) ↔ m ≥ 1 := 
sorry

end range_of_m_l341_34112


namespace geometric_progression_term_count_l341_34111

theorem geometric_progression_term_count
  (q : ℝ) (b4 : ℝ) (S : ℝ) (b1 : ℝ)
  (h1 : q = 1 / 3)
  (h2 : b4 = b1 * (q ^ 3))
  (h3 : S = b1 * (1 - q ^ 5) / (1 - q))
  (h4 : b4 = 1 / 54)
  (h5 : S = 121 / 162) :
  5 = 5 := sorry

end geometric_progression_term_count_l341_34111


namespace find_coeff_and_root_range_l341_34156

def f (x : ℝ) (a b : ℝ) : ℝ := a * x^3 - b * x + 4

theorem find_coeff_and_root_range (a b : ℝ)
  (h1 : f 2 a b = - (4/3))
  (h2 : deriv (λ x => f x a b) 2 = 0) :
  a = 1 / 3 ∧ b = 4 ∧ 
  (∀ k : ℝ, (∃ x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3 ∧ f x1 (1/3) 4 = k ∧ f x2 (1/3) 4 = k ∧ f x3 (1/3) 4 = k) ↔ - (4/3) < k ∧ k < 28/3) :=
sorry

end find_coeff_and_root_range_l341_34156


namespace two_trains_cross_time_l341_34178

/-- Definition for the two trains' parameters -/
structure Train :=
  (length : ℝ)  -- length in meters
  (speed : ℝ)  -- speed in km/hr

/-- The parameters of Train 1 and Train 2 -/
def train1 : Train := { length := 140, speed := 60 }
def train2 : Train := { length := 160, speed := 40 }

noncomputable def relative_speed_mps (t1 t2 : Train) : ℝ :=
  (t1.speed + t2.speed) * (5 / 18)

noncomputable def total_length (t1 t2 : Train) : ℝ :=
  t1.length + t2.length

noncomputable def time_to_cross (t1 t2 : Train) : ℝ :=
  total_length t1 t2 / relative_speed_mps t1 t2

theorem two_trains_cross_time :
  time_to_cross train1 train2 = 10.8 := by
  sorry

end two_trains_cross_time_l341_34178


namespace range_of_m_l341_34120

theorem range_of_m (x y m : ℝ) (h1 : 2 / x + 1 / y = 1) (h2 : x + y = 2 + 2 * m) : -4 < m ∧ m < 2 :=
sorry

end range_of_m_l341_34120


namespace complement_union_eq_l341_34176

variable (U : Set ℝ) (M N : Set ℝ)

noncomputable def complement_union (U M N : Set ℝ) : Set ℝ :=
  U \ (M ∪ N)

theorem complement_union_eq :
  U = Set.univ → 
  M = {x | |x| < 1} → 
  N = {y | ∃ x, y = 2^x} → 
  complement_union U M N = {x | x ≤ -1} :=
by
  intros hU hM hN
  unfold complement_union
  sorry

end complement_union_eq_l341_34176


namespace number_of_items_l341_34165

variable (s d : ℕ)
variable (total_money cost_sandwich cost_drink discount : ℝ)
variable (s_purchase_criterion : s > 5)
variable (total_money_value : total_money = 50.00)
variable (cost_sandwich_value : cost_sandwich = 6.00)
variable (cost_drink_value : cost_drink = 1.50)
variable (discount_value : discount = 5.00)

theorem number_of_items (h1 : total_money = 50.00)
(h2 : cost_sandwich = 6.00)
(h3 : cost_drink = 1.50)
(h4 : discount = 5.00)
(h5 : s > 5) :
  s + d = 9 :=
by
  sorry

end number_of_items_l341_34165


namespace perpendicular_lines_slope_l341_34153

theorem perpendicular_lines_slope {a : ℝ} :
  (∃ (a : ℝ), (∀ x y : ℝ, x + 2 * y - 1 = 0 → a * x - y - 1 = 0) ∧ (a * (-1 / 2)) = -1) → a = 2 :=
by sorry

end perpendicular_lines_slope_l341_34153


namespace percentage_gain_is_20_percent_l341_34140

theorem percentage_gain_is_20_percent (manufacturing_cost transportation_cost total_shoes selling_price : ℝ)
(h1 : manufacturing_cost = 220)
(h2 : transportation_cost = 500)
(h3 : total_shoes = 100)
(h4 : selling_price = 270) :
  let cost_per_shoe := manufacturing_cost + transportation_cost / total_shoes
  let profit_per_shoe := selling_price - cost_per_shoe
  let percentage_gain := (profit_per_shoe / cost_per_shoe) * 100
  percentage_gain = 20 :=
by
  sorry

end percentage_gain_is_20_percent_l341_34140


namespace find_x0_l341_34172

noncomputable def f (x : ℝ) : ℝ := 13 - 8 * x + x^2

theorem find_x0 :
  (∃ x0 : ℝ, deriv f x0 = 4) → ∃ x0 : ℝ, x0 = 6 :=
by
  sorry

end find_x0_l341_34172


namespace percentage_cats_less_dogs_l341_34151

theorem percentage_cats_less_dogs (C D F : ℕ) (h1 : C < D) (h2 : F = 2 * D) (h3 : C + D + F = 304) (h4 : F = 160) :
  ((D - C : ℕ) * 100 / D : ℕ) = 20 := 
sorry

end percentage_cats_less_dogs_l341_34151


namespace oxen_grazing_months_l341_34160

theorem oxen_grazing_months (a_oxen : ℕ) (a_months : ℕ) (b_oxen : ℕ) (c_oxen : ℕ) (c_months : ℕ) (total_rent : ℝ) (c_share_rent : ℝ) (x : ℕ) :
  a_oxen = 10 →
  a_months = 7 →
  b_oxen = 12 →
  c_oxen = 15 →
  c_months = 3 →
  total_rent = 245 →
  c_share_rent = 63 →
  (c_oxen * c_months) / ((a_oxen * a_months) + (b_oxen * x) + (c_oxen * c_months)) = c_share_rent / total_rent →
  x = 5 :=
sorry

end oxen_grazing_months_l341_34160


namespace quotient_remainder_div_by_18_l341_34150

theorem quotient_remainder_div_by_18 (M q : ℕ) (h : M = 54 * q + 37) : 
  ∃ k r, M = 18 * k + r ∧ r < 18 ∧ k = 3 * q + 2 ∧ r = 1 :=
by sorry

end quotient_remainder_div_by_18_l341_34150


namespace total_profit_is_8800_l341_34138

variable (A B C : Type) [CommRing A] [CommRing B] [CommRing C]

variable (investment_A investment_B investment_C : ℝ)
variable (total_profit : ℝ)

-- Conditions
def A_investment_three_times_B (investment_A investment_B : ℝ) : Prop :=
  investment_A = 3 * investment_B

def B_invest_two_thirds_C (investment_B investment_C : ℝ) : Prop :=
  investment_B = 2 / 3 * investment_C

def B_share_is_1600 (investment_B total_profit : ℝ) : Prop :=
  1600 = (2 / 11) * total_profit

theorem total_profit_is_8800 :
  A_investment_three_times_B investment_A investment_B →
  B_invest_two_thirds_C investment_B investment_C →
  B_share_is_1600 investment_B total_profit →
  total_profit = 8800 :=
by
  intros
  sorry

end total_profit_is_8800_l341_34138


namespace functional_eq_uniq_l341_34186

noncomputable def f (x : ℝ) : ℝ := sorry

theorem functional_eq_uniq (f : ℝ → ℝ) (h : ∀ x y : ℝ, (f x * f y - f (x * y)) / 4 = x^2 + y^2 + 2) : 
  ∀ x : ℝ, f x = x^2 + 3 :=
by 
  sorry

end functional_eq_uniq_l341_34186


namespace num_kids_eq_3_l341_34114

def mom_eyes : ℕ := 1
def dad_eyes : ℕ := 3
def kid_eyes : ℕ := 4
def total_eyes : ℕ := 16

theorem num_kids_eq_3 : ∃ k : ℕ, 1 + 3 + 4 * k = 16 ∧ k = 3 := by
  sorry

end num_kids_eq_3_l341_34114


namespace children_playing_both_sports_l341_34117

variable (total_children : ℕ) (T : ℕ) (S : ℕ) (N : ℕ)

theorem children_playing_both_sports 
  (h1 : total_children = 38) 
  (h2 : T = 19) 
  (h3 : S = 21) 
  (h4 : N = 10) : 
  (T + S) - (total_children - N) = 12 := 
by
  sorry

end children_playing_both_sports_l341_34117


namespace locker_number_problem_l341_34198

theorem locker_number_problem 
  (cost_per_digit : ℝ)
  (total_cost : ℝ)
  (one_digit_cost : ℝ)
  (two_digit_cost : ℝ)
  (three_digit_cost : ℝ) :
  cost_per_digit = 0.03 →
  one_digit_cost = 0.27 →
  two_digit_cost = 5.40 →
  three_digit_cost = 81.00 →
  total_cost = 206.91 →
  10 * cost_per_digit = six_cents →
  9 * cost_per_digit = three_cents →
  1 * 9 * cost_per_digit = one_digit_cost →
  2 * 45 * cost_per_digit = two_digit_cost →
  3 * 300 * cost_per_digit = three_digit_cost →
  (999 * 3 + x * 4 = 6880) →
  ∀ total_locker : ℕ, total_locker = 2001 := sorry

end locker_number_problem_l341_34198


namespace part1_part2_l341_34128

def f (x a : ℝ) : ℝ := abs (x - 1) + abs (x - a)

theorem part1 (x : ℝ) (h : f x 2 ≥ 2) : x ≤ 1/2 ∨ x ≥ 2.5 := by
  sorry

theorem part2 (a : ℝ) (h_even : ∀ x : ℝ, f (-x) a = f x a) : a = -1 := by
  sorry

end part1_part2_l341_34128


namespace probability_of_chosen_figure_is_circle_l341_34144

-- Define the total number of figures and number of circles.
def total_figures : ℕ := 12
def number_of_circles : ℕ := 5

-- Define the probability calculation.
def probability_of_circle (total : ℕ) (circles : ℕ) : ℚ := circles / total

-- State the theorem using the defined conditions.
theorem probability_of_chosen_figure_is_circle : 
  probability_of_circle total_figures number_of_circles = 5 / 12 :=
by
  sorry  -- Placeholder for the actual proof.

end probability_of_chosen_figure_is_circle_l341_34144


namespace find_n_l341_34155

theorem find_n (n k : ℕ) (h_pos : k > 0) (h_calls : ∀ (s : Finset (Fin n)), s.card = n-2 → (∃ (f : Finset (Fin n × Fin n)), f.card = 3^k ∧ ∀ (x y : Fin n), (x, y) ∈ f → x ≠ y)) : n = 5 := 
sorry

end find_n_l341_34155


namespace min_value_l341_34193

theorem min_value (a b : ℝ) (h_pos : 0 < a ∧ 0 < b) (h_ab : a * b = 1) (h_a_2b : a = 2 * b) :
  a + 2 * b = 2 * Real.sqrt 2 := by
  sorry

end min_value_l341_34193


namespace total_amount_shared_l341_34110

theorem total_amount_shared (a b c d : ℝ) (h1 : a = (1/3) * (b + c + d)) 
    (h2 : b = (2/7) * (a + c + d)) (h3 : c = (4/9) * (a + b + d)) 
    (h4 : d = (5/11) * (a + b + c)) (h5 : a = b + 20) (h6 : c = d - 15) 
    (h7 : (a + b + c + d) % 10 = 0) : a + b + c + d = 1330 :=
by
  sorry

end total_amount_shared_l341_34110


namespace laura_has_435_dollars_l341_34108

-- Define the monetary values and relationships
def darwin_money := 45
def mia_money := 2 * darwin_money + 20
def combined_money := mia_money + darwin_money
def laura_money := 3 * combined_money - 30

-- The theorem to prove: Laura's money is $435
theorem laura_has_435_dollars : laura_money = 435 := by
  sorry

end laura_has_435_dollars_l341_34108


namespace fourth_house_number_l341_34189

theorem fourth_house_number (sum: ℕ) (k x: ℕ) (h1: sum = 78) (h2: k ≥ 4)
  (h3: (k+1) * (x + k) = 78) : x + 6 = 14 :=
by
  sorry

end fourth_house_number_l341_34189


namespace prism_surface_area_l341_34118

theorem prism_surface_area (P : ℝ) (h : ℝ) (S : ℝ) (s: ℝ) 
  (hP : P = 4)
  (hh : h = 2) 
  (hs : s = 1) 
  (h_surf_top : S = s * s) 
  (h_lat : S = 8) : 
  S = 10 := 
sorry

end prism_surface_area_l341_34118


namespace simplify_complex_expr_l341_34192

theorem simplify_complex_expr : ∀ i : ℂ, i^2 = -1 → 3 * (4 - 2 * i) + 2 * i * (3 - i) = 14 :=
by 
  intro i 
  intro h
  sorry

end simplify_complex_expr_l341_34192


namespace ratio_h_w_l341_34129

-- Definitions from conditions
variables (h w : ℝ)
variables (XY YZ : ℝ)
variables (h_pos : 0 < h) (w_pos : 0 < w) -- heights and widths are positive
variables (XY_pos : 0 < XY) (YZ_pos : 0 < YZ) -- segment lengths are positive

-- Given that in the right-angled triangle ∆XYZ, YZ = 2 * XY
axiom YZ_eq_2XY : YZ = 2 * XY

-- Prove that h / w = 3 / 8
theorem ratio_h_w (H : XY / YZ = 4 * h / (3 * w)) : h / w = 3 / 8 :=
by {
  -- Use the axioms and given conditions here to prove H == ratio
  sorry
}

end ratio_h_w_l341_34129


namespace quarterly_business_tax_cost_l341_34135

theorem quarterly_business_tax_cost
    (price_federal : ℕ := 50)
    (price_state : ℕ := 30)
    (Q : ℕ)
    (num_federal : ℕ := 60)
    (num_state : ℕ := 20)
    (num_quart_business : ℕ := 10)
    (total_revenue : ℕ := 4400)
    (revenue_equation : num_federal * price_federal + num_state * price_state + num_quart_business * Q = total_revenue) :
    Q = 80 :=
by 
  sorry

end quarterly_business_tax_cost_l341_34135


namespace correct_mean_251_l341_34136

theorem correct_mean_251
  (n : ℕ) (incorrect_mean : ℕ) (wrong_val : ℕ) (correct_val : ℕ)
  (h1 : n = 30) (h2 : incorrect_mean = 250) (h3 : wrong_val = 135) (h4 : correct_val = 165) :
  ((incorrect_mean * n + (correct_val - wrong_val)) / n) = 251 :=
by
  sorry

end correct_mean_251_l341_34136


namespace chenny_candies_l341_34181

def friends_count : ℕ := 7
def candies_per_friend : ℕ := 2
def candies_have : ℕ := 10

theorem chenny_candies : 
    (friends_count * candies_per_friend - candies_have) = 4 := by
    sorry

end chenny_candies_l341_34181


namespace son_age_l341_34132

theorem son_age:
  ∃ S M : ℕ, 
  (M = S + 20) ∧ 
  (M + 2 = 2 * (S + 2)) ∧ 
  (S = 18) := 
by
  sorry

end son_age_l341_34132


namespace proper_subsets_B_l341_34102

theorem proper_subsets_B (A B : Set ℝ) (a : ℝ)
  (hA : A = {x | x^2 + 2*x + 1 = 0})
  (hA_singleton : A = {a})
  (hB : B = {x | x^2 + a*x = 0}) :
  a = -1 ∧ 
  B = {0, 1} ∧
  (∀ S, S ∈ ({∅, {0}, {1}} : Set (Set ℝ)) ↔ S ⊂ B) :=
by
  -- Proof not provided, only statement required.
  sorry

end proper_subsets_B_l341_34102


namespace gcd_18_30_l341_34126

-- Define the two numbers
def num1 : ℕ := 18
def num2 : ℕ := 30

-- State the theorem to find the gcd
theorem gcd_18_30 : Nat.gcd num1 num2 = 6 := by
  sorry

end gcd_18_30_l341_34126


namespace find_m_l341_34197

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := x + m
noncomputable def g (x : ℝ) : ℝ := 2 * x - 2

theorem find_m : 
  ∃ m : ℝ, ∀ x : ℝ, f m x = g x → m = -2 := by
  sorry

end find_m_l341_34197


namespace gcd_sum_abcde_edcba_l341_34157

-- Definition to check if digits are consecutive
def consecutive_digits (a b c d e : ℤ) : Prop :=
  b = a + 1 ∧ c = a + 2 ∧ d = a + 3 ∧ e = a + 4

-- Definition of the five-digit number in the form abcde
def abcde (a b c d e : ℤ) : ℤ :=
  10000 * a + 1000 * b + 100 * c + 10 * d + e

-- Definition of the five-digit number in the form edcba
def edcba (a b c d e : ℤ) : ℤ :=
  10000 * e + 1000 * d + 100 * c + 10 * b + a

-- Definition which sums both abcde and edcba
def sum_abcde_edcba (a b c d e : ℤ) : ℤ :=
  abcde a b c d e + edcba a b c d e

-- Lean theorem statement for the problem
theorem gcd_sum_abcde_edcba (a b c d e : ℤ) (h : consecutive_digits a b c d e) :
  Int.gcd (sum_abcde_edcba a b c d e) 11211 = 11211 :=
by
  sorry

end gcd_sum_abcde_edcba_l341_34157


namespace concert_songs_l341_34196

def total_songs (g : ℕ) : ℕ := (9 + 3 + 9 + g) / 3

theorem concert_songs 
  (g : ℕ) 
  (h1 : 9 + 3 + 9 + g = 3 * total_songs g) 
  (h2 : 3 + g % 4 = 0) 
  (h3 : 4 ≤ g ∧ g ≤ 9) 
  : total_songs g = 9 ∨ total_songs g = 10 := 
sorry

end concert_songs_l341_34196


namespace find_2x_2y_2z_l341_34107

theorem find_2x_2y_2z (x y z : ℝ) 
  (h1 : y + z = 10 - 2 * x)
  (h2 : x + z = -12 - 4 * y)
  (h3 : x + y = 5 - 2 * z) : 
  2 * x + 2 * y + 2 * z = 3 :=
by
  sorry

end find_2x_2y_2z_l341_34107


namespace hyperbola_ratio_l341_34184

theorem hyperbola_ratio (a b c : ℝ)
  (h1 : a > 0) (h2 : b > 0)
  (h_eq : a^2 - b^2 = 1)
  (h_ecc : 2 = c / a)
  (h_focus : c = 1) :
  a / b = Real.sqrt 3 / 3 := by
  have ha : a = 1 / 2 := sorry
  have hc : c = 1 := h_focus
  have hb : b = Real.sqrt 3 / 2 := sorry
  exact sorry

end hyperbola_ratio_l341_34184


namespace non_student_ticket_price_l341_34163

theorem non_student_ticket_price (x : ℕ) : 
  (∃ (n_student_ticket_price ticket_count total_revenue student_tickets : ℕ),
    n_student_ticket_price = 9 ∧
    ticket_count = 2000 ∧
    total_revenue = 20960 ∧
    student_tickets = 520 ∧
    (student_tickets * n_student_ticket_price + (ticket_count - student_tickets) * x = total_revenue)) -> 
  x = 11 := 
by
  -- placeholder for proof
  sorry

end non_student_ticket_price_l341_34163


namespace calculation_proof_l341_34123

theorem calculation_proof : 
  2 * Real.tan (Real.pi / 3) - (-2023) ^ 0 + (1 / 2) ^ (-1 : ℤ) + abs (Real.sqrt 3 - 1) = 3 * Real.sqrt 3 := 
by
  sorry

end calculation_proof_l341_34123


namespace yuan_exchange_l341_34139

theorem yuan_exchange : 
  ∃ (n : ℕ), n = 5 ∧ ∀ (x y : ℕ), x + 5 * y = 20 → x ≥ 0 ∧ y ≥ 0 :=
by {
  sorry
}

end yuan_exchange_l341_34139


namespace mitchell_more_than_antonio_l341_34177

-- Definitions based on conditions
def mitchell_pencils : ℕ := 30
def total_pencils : ℕ := 54

-- Definition of the main question
def antonio_pencils : ℕ := total_pencils - mitchell_pencils

-- The theorem to be proved
theorem mitchell_more_than_antonio : mitchell_pencils - antonio_pencils = 6 :=
by
-- Proof is omitted
sorry

end mitchell_more_than_antonio_l341_34177


namespace solve_inequality_system_simplify_expression_l341_34100

-- Part 1: System of Inequalities

theorem solve_inequality_system : 
  ∀ (x : ℝ), (x + 2) / 5 < 1 ∧ 3 * x - 1 ≥ 2 * x → 1 ≤ x ∧ x < 3 :=  by
  sorry

-- Part 2: Expression Simplification

theorem simplify_expression (m : ℝ) (hm : m ≠ 0) : 
  (m - 1 / m) * ((m^2 - m) / (m^2 - 2 * m + 1)) = m + 1 :=
  by
  sorry

end solve_inequality_system_simplify_expression_l341_34100


namespace value_of_g_at_3_l341_34141

theorem value_of_g_at_3 (g : ℕ → ℕ) (h : ∀ x, g (x + 2) = 2 * x + 3) : g 3 = 5 := by
  sorry

end value_of_g_at_3_l341_34141


namespace profit_percentage_A_is_20_l341_34175

-- Definitions of conditions
def cost_price_A := 156 -- Cost price of the cricket bat for A
def selling_price_C := 234 -- Selling price of the cricket bat to C
def profit_percent_B := 25 / 100 -- Profit percentage for B

-- Calculations
def cost_price_B := selling_price_C / (1 + profit_percent_B) -- Cost price of the cricket bat for B
def selling_price_A := cost_price_B -- Selling price of the cricket bat for A

-- Profit and profit percentage calculations
def profit_A := selling_price_A - cost_price_A -- Profit for A
def profit_percent_A := profit_A / cost_price_A * 100 -- Profit percentage for A

-- Statement to prove
theorem profit_percentage_A_is_20 : profit_percent_A = 20 :=
by
  sorry

end profit_percentage_A_is_20_l341_34175


namespace tiling_polygons_l341_34109

theorem tiling_polygons (n : ℕ) (h1 : 2 < n) (h2 : ∃ x : ℕ, x * (((n - 2) * 180 : ℝ) / n) = 360) :
  n = 3 ∨ n = 4 ∨ n = 6 := 
by
  sorry

end tiling_polygons_l341_34109


namespace range_of_a_l341_34183

theorem range_of_a (a : ℝ) : (∀ x > 0, a - x - |Real.log x| ≤ 0) → a ≤ 1 := by
  sorry

end range_of_a_l341_34183


namespace circle_diameter_and_circumference_l341_34168

theorem circle_diameter_and_circumference (A : ℝ) (hA : A = 225 * π) : 
  ∃ r d C, r = 15 ∧ d = 2 * r ∧ C = 2 * π * r ∧ d = 30 ∧ C = 30 * π :=
by
  sorry

end circle_diameter_and_circumference_l341_34168


namespace floor_sqrt_sum_eq_floor_sqrt_expr_l341_34131

-- Proof problem definition
theorem floor_sqrt_sum_eq_floor_sqrt_expr (n : ℕ) : 
  (Int.floor (Real.sqrt n + Real.sqrt (n + 1))) = (Int.floor (Real.sqrt (4 * n + 2))) := 
sorry

end floor_sqrt_sum_eq_floor_sqrt_expr_l341_34131


namespace nina_running_distance_l341_34170

theorem nina_running_distance (total_distance : ℝ) (initial_run : ℝ) (num_initial_runs : ℕ) :
  total_distance = 0.8333333333333334 →
  initial_run = 0.08333333333333333 →
  num_initial_runs = 2 →
  (total_distance - initial_run * num_initial_runs = 0.6666666666666667) :=
by
  intros h_total h_initial h_num
  sorry

end nina_running_distance_l341_34170


namespace calculation_result_l341_34194

theorem calculation_result :
  let a := 0.0088
  let b := 4.5
  let c := 0.05
  let d := 0.1
  let e := 0.008
  (a * b) / (c * d * e) = 990 :=
by
  sorry

end calculation_result_l341_34194
