import Mathlib

namespace yearly_return_of_1500_investment_is_27_percent_l1521_152129

-- Definitions based on conditions
def combined_yearly_return (x : ℝ) : Prop :=
  let investment1 := 500
  let investment2 := 1500
  let total_investment := investment1 + investment2
  let combined_return := 0.22 * total_investment
  let return_from_500 := 0.07 * investment1
  let return_from_1500 := combined_return - return_from_500
  x / 100 * investment2 = return_from_1500

-- Theorem statement to be proven
theorem yearly_return_of_1500_investment_is_27_percent : combined_yearly_return 27 :=
by sorry

end yearly_return_of_1500_investment_is_27_percent_l1521_152129


namespace symmetric_line_equation_l1521_152174

theorem symmetric_line_equation (x y : ℝ) : 
  (y = 2 * x + 1) → (-y = 2 * (-x) + 1) :=
by
  sorry

end symmetric_line_equation_l1521_152174


namespace water_volume_correct_l1521_152180

noncomputable def volume_of_water : ℝ :=
  let r := 4
  let h := 9
  let d := 2
  48 * Real.pi - 36 * Real.sqrt 3

theorem water_volume_correct :
  volume_of_water = 48 * Real.pi - 36 * Real.sqrt 3 := 
by sorry

end water_volume_correct_l1521_152180


namespace max_disjoint_regions_l1521_152153

theorem max_disjoint_regions {p : ℕ} (hp : Nat.Prime p) (hp_ge3 : 3 ≤ p) : ∃ R, R = 3 * p^2 - 3 * p + 1 :=
by
  sorry

end max_disjoint_regions_l1521_152153


namespace sequence_solution_l1521_152106

theorem sequence_solution :
  ∃ (a : ℕ → ℕ), a 1 = 2 ∧ (∀ n : ℕ, n ∈ (Set.Icc 1 9) → 
    (n * a (n + 1) = (n + 1) * a n + 2)) ∧ a 10 = 38 :=
by
  sorry

end sequence_solution_l1521_152106


namespace solve_triangle_l1521_152112

noncomputable def angle_A := 45
noncomputable def angle_B := 60
noncomputable def side_a := Real.sqrt 2

theorem solve_triangle {A B : ℕ} {a b : Real}
    (hA : A = angle_A)
    (hB : B = angle_B)
    (ha : a = side_a) :
    b = Real.sqrt 3 := 
by sorry

end solve_triangle_l1521_152112


namespace whole_numbers_between_sqrts_l1521_152167

theorem whole_numbers_between_sqrts :
  let lower_bound := Real.sqrt 50
  let upper_bound := Real.sqrt 200
  let start := Nat.ceil lower_bound
  let end_ := Nat.floor upper_bound
  ∃ n, n = end_ - start + 1 ∧ n = 7 := by
  sorry

end whole_numbers_between_sqrts_l1521_152167


namespace twelfth_term_arithmetic_sequence_l1521_152163

theorem twelfth_term_arithmetic_sequence (a d : ℤ) (h1 : a + 2 * d = 13) (h2 : a + 6 * d = 25) : a + 11 * d = 40 := 
sorry

end twelfth_term_arithmetic_sequence_l1521_152163


namespace distance_ratio_l1521_152173

-- Defining the conditions
def speedA : ℝ := 50 -- Speed of Car A in km/hr
def timeA : ℝ := 6 -- Time taken by Car A in hours

def speedB : ℝ := 100 -- Speed of Car B in km/hr
def timeB : ℝ := 1 -- Time taken by Car B in hours

-- Calculating the distances
def distanceA : ℝ := speedA * timeA -- Distance covered by Car A
def distanceB : ℝ := speedB * timeB -- Distance covered by Car B

-- Statement to prove the ratio of distances
theorem distance_ratio : (distanceA / distanceB) = 3 :=
by
  -- Calculations here might be needed, but we use sorry to indicate proof is pending
  sorry

end distance_ratio_l1521_152173


namespace ratio_accepted_to_rejected_l1521_152185

-- Let n be the total number of eggs processed per day
def eggs_per_day := 400

-- Let accepted_per_batch be the number of accepted eggs per batch
def accepted_per_batch := 96

-- Let rejected_per_batch be the number of rejected eggs per batch
def rejected_per_batch := 4

-- On a particular day, 12 additional eggs were accepted
def additional_accepted_eggs := 12

-- Normalize definitions to make our statements clearer
def accepted_batches := eggs_per_day / (accepted_per_batch + rejected_per_batch)
def normally_accepted_eggs := accepted_per_batch * accepted_batches
def normally_rejected_eggs := rejected_per_batch * accepted_batches
def total_accepted_eggs := normally_accepted_eggs + additional_accepted_eggs
def total_rejected_eggs := eggs_per_day - total_accepted_eggs

theorem ratio_accepted_to_rejected :
  (total_accepted_eggs / gcd total_accepted_eggs total_rejected_eggs) = 99 ∧
  (total_rejected_eggs / gcd total_accepted_eggs total_rejected_eggs) = 1 :=
by
  sorry

end ratio_accepted_to_rejected_l1521_152185


namespace weight_of_person_replaced_l1521_152154

theorem weight_of_person_replaced (W_new : ℝ) (h1 : W_new = 74) (h2 : (W_new - W_old) = 9) : W_old = 65 := 
by
  sorry

end weight_of_person_replaced_l1521_152154


namespace part1_eq_of_line_l_part2_eq_of_line_l1_l1521_152141

def intersection_point (m n : ℝ × ℝ × ℝ) : ℝ × ℝ := sorry

def line_through_point_eq_dists (P A B : ℝ × ℝ) : ℝ × ℝ × ℝ := sorry
def line_area_triangle (P : ℝ × ℝ) (triangle_area : ℝ) : ℝ × ℝ × ℝ := sorry

-- Conditions defined:
def m : ℝ × ℝ × ℝ := (2, -1, -3)
def n : ℝ × ℝ × ℝ := (1, 1, -3)
def P : ℝ × ℝ := intersection_point m n
def A : ℝ × ℝ := (1, 3)
def B : ℝ × ℝ := (3, 2)
def triangle_area : ℝ := 4

-- Questions translated into Lean 4 statements:
theorem part1_eq_of_line_l : ∃ l : ℝ × ℝ × ℝ, 
  (l = line_through_point_eq_dists P A B) := sorry

theorem part2_eq_of_line_l1 : ∃ l1 : ℝ × ℝ × ℝ,
  (l1 = line_area_triangle P triangle_area) := sorry

end part1_eq_of_line_l_part2_eq_of_line_l1_l1521_152141


namespace green_eyes_count_l1521_152145

noncomputable def people_count := 100
noncomputable def blue_eyes := 19
noncomputable def brown_eyes := people_count / 2
noncomputable def black_eyes := people_count / 4
noncomputable def green_eyes := people_count - (blue_eyes + brown_eyes + black_eyes)

theorem green_eyes_count : green_eyes = 6 := by
  sorry

end green_eyes_count_l1521_152145


namespace ratio_of_areas_l1521_152125

theorem ratio_of_areas 
  (t : ℝ) (q : ℝ)
  (h1 : t = 1 / 4)
  (h2 : q = 1 / 2) :
  q / t = 2 :=
by sorry

end ratio_of_areas_l1521_152125


namespace solve_system_of_inequalities_l1521_152195

theorem solve_system_of_inequalities (p : ℚ) : (19 * p < 10) ∧ (1 / 2 < p) → (1 / 2 < p ∧ p < 10 / 19) :=
by
  sorry

end solve_system_of_inequalities_l1521_152195


namespace quadratic_two_distinct_real_roots_l1521_152193

theorem quadratic_two_distinct_real_roots 
    (a b c : ℝ)
    (h1 : a > 0)
    (h2 : c < 0) : 
    ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ ax^2 + bx + c = 0 := 
sorry

end quadratic_two_distinct_real_roots_l1521_152193


namespace total_cost_l1521_152152

theorem total_cost (cost_sandwich cost_soda cost_cookie : ℕ)
    (num_sandwich num_soda num_cookie : ℕ) 
    (h1 : cost_sandwich = 4) 
    (h2 : cost_soda = 3) 
    (h3 : cost_cookie = 2) 
    (h4 : num_sandwich = 4) 
    (h5 : num_soda = 6) 
    (h6 : num_cookie = 7):
    cost_sandwich * num_sandwich + cost_soda * num_soda + cost_cookie * num_cookie = 48 :=
by
  sorry

end total_cost_l1521_152152


namespace probability_multiple_of_7_condition_l1521_152140

theorem probability_multiple_of_7_condition :
  ∃ (a b : ℕ), 1 ≤ a ∧ a ≤ 100 ∧ 1 ≤ b ∧ b ≤ 100 ∧ a ≠ b ∧ (ab + a + b + 1) % 7 = 0 → 
  (1295 / 4950 = 259 / 990) :=
sorry

end probability_multiple_of_7_condition_l1521_152140


namespace simplify_expression_evaluate_expression_l1521_152189

-- Definitions for the first part
variable (a b : ℝ)

theorem simplify_expression (ha : a ≠ 0) (hb : b ≠ 0) :
  (2 * a^(1/2) * b^(1/3)) * (a^(2/3) * b^(1/2)) / (1/3 * a^(1/6) * b^(5/6)) = 6 * a :=
by
  sorry

-- Definitions for the second part
theorem evaluate_expression :
  (9 / 16)^(1 / 2) + 10^(Real.log 9 / Real.log 10 - 2 * Real.log 2 / Real.log 10) + Real.log (4 * Real.exp 3) 
  - (Real.log 8 / Real.log 9) * (Real.log 33 / Real.log 4) = 7 / 2 :=
by 
  sorry

end simplify_expression_evaluate_expression_l1521_152189


namespace anna_not_lose_l1521_152161

theorem anna_not_lose :
  ∀ (cards : Fin 9 → ℕ),
    ∃ (A B C D : ℕ),
      (A + B ≥ C + D) :=
by
  sorry

end anna_not_lose_l1521_152161


namespace time_after_1456_minutes_l1521_152138

noncomputable def hours_in_minutes := 1456 / 60
noncomputable def minutes_remainder := 1456 % 60

def current_time : Nat := 6 * 60  -- 6:00 a.m. in minutes
def added_time : Nat := current_time + 1456

def six_sixteen_am : Nat := (6 * 60) + 16  -- 6:16 a.m. in minutes the next day

theorem time_after_1456_minutes : added_time % (24 * 60) = six_sixteen_am :=
by
  sorry

end time_after_1456_minutes_l1521_152138


namespace no_integers_six_digit_cyclic_permutation_l1521_152165

theorem no_integers_six_digit_cyclic_permutation (n : ℕ) (a b c d e f : ℕ) (h : 10 ≤ a ∧ a < 10) :
  ¬(n = 5 ∨ n = 6 ∨ n = 8 ∧
    n * (a * 10^5 + b * 10^4 + c * 10^3 + d * 10^2 + e * 10 + f) =
    b * 10^5 + c * 10^4 + d * 10^3 + e * 10^2 + f * 10 + a) :=
by sorry

end no_integers_six_digit_cyclic_permutation_l1521_152165


namespace conor_vegetables_per_week_l1521_152191

theorem conor_vegetables_per_week : 
  let eggplants_per_day := 12 
  let carrots_per_day := 9 
  let potatoes_per_day := 8 
  let work_days_per_week := 4 
  (eggplants_per_day + carrots_per_day + potatoes_per_day) * work_days_per_week = 116 := 
by 
  let eggplants_per_day := 12 
  let carrots_per_day := 9 
  let potatoes_per_day := 8 
  let work_days_per_week := 4 
  show (eggplants_per_day + carrots_per_day + potatoes_per_day) * work_days_per_week = 116 
  sorry

end conor_vegetables_per_week_l1521_152191


namespace solve_for_x_l1521_152156

def star (a b : ℤ) := a * b + 3 * b - a

theorem solve_for_x : ∃ x : ℤ, star 4 x = 46 := by
  sorry

end solve_for_x_l1521_152156


namespace time_for_train_to_pass_platform_is_190_seconds_l1521_152157

def trainLength : ℕ := 1200
def timeToCrossTree : ℕ := 120
def platformLength : ℕ := 700
def speed (distance time : ℕ) := distance / time
def distanceToCrossPlatform (trainLength platformLength : ℕ) := trainLength + platformLength
def timeToCrossPlatform (distance speed : ℕ) := distance / speed

theorem time_for_train_to_pass_platform_is_190_seconds
  (trainLength timeToCrossTree platformLength : ℕ) (h1 : trainLength = 1200) (h2 : timeToCrossTree = 120) (h3 : platformLength = 700) :
  timeToCrossPlatform (distanceToCrossPlatform trainLength platformLength) (speed trainLength timeToCrossTree) = 190 := by
  sorry

end time_for_train_to_pass_platform_is_190_seconds_l1521_152157


namespace correct_options_A_and_D_l1521_152113

noncomputable def problem_statement :=
  ∃ A B C D : Prop,
  (A = (∀ x : ℝ, x^2 + x + 1 > 0) ↔ (∃ x : ℝ, x^2 + x + 1 ≤ 0)) ∧ 
  (B = ∀ (a b c d : ℝ), a > b → c > d → ¬(a * c > b * d)) ∧
  (C = ∀ m : ℝ, ¬((∀ x : ℝ, x > 0 → (m^2 - m - 1) * x^(m^2 - 2 * m - 3) < 0) → (-1 < m ∧ m < 2))) ∧
  (D = ∀ a : ℝ, (∃ x₁ x₂ : ℝ, x₁ > 0 ∧ x₂ < 0 ∧ x₁ + x₂ = 3 - a ∧ x₁ * x₂ = a) → a < 0)

-- We need to prove that only A and D are true
theorem correct_options_A_and_D : problem_statement :=
  sorry

end correct_options_A_and_D_l1521_152113


namespace greatest_of_consecutive_integers_l1521_152135

theorem greatest_of_consecutive_integers (x y z : ℤ) (h1: y = x + 1) (h2: z = x + 2) (h3: x + y + z = 21) : z = 8 :=
by
  sorry

end greatest_of_consecutive_integers_l1521_152135


namespace paper_strip_total_covered_area_l1521_152194

theorem paper_strip_total_covered_area :
  let length := 12
  let width := 2
  let strip_count := 5
  let overlap_per_intersection := 4
  let intersection_count := 10
  let area_per_strip := length * width
  let total_area_without_overlap := strip_count * area_per_strip
  let total_overlap_area := intersection_count * overlap_per_intersection
  total_area_without_overlap - total_overlap_area = 80 := 
by
  sorry

end paper_strip_total_covered_area_l1521_152194


namespace problem1_problem2_l1521_152169

-- Problem (1)
theorem problem1 (a : ℚ) (h : a = -1/2) : 
  a * (a - 4) - (a + 6) * (a - 2) = 16 := by
  sorry

-- Problem (2)
theorem problem2 (x y : ℚ) (hx : x = 8) (hy : y = -8) :
  (x + 2 * y) * (x - 2 * y) - (2 * x - y) * (-2 * x - y) = 0 := by
  sorry

end problem1_problem2_l1521_152169


namespace largest_of_eight_consecutive_summing_to_5400_l1521_152170

theorem largest_of_eight_consecutive_summing_to_5400 :
  ∃ (n : ℕ), (n + (n+1) + (n+2) + (n+3) + (n+4) + (n+5) + (n+6) + (n+7) = 5400)
  → (n+7 = 678) :=
by 
  sorry

end largest_of_eight_consecutive_summing_to_5400_l1521_152170


namespace additional_cost_l1521_152192

theorem additional_cost (original_bales : ℕ) (previous_cost_per_bale : ℕ) (better_quality_cost_per_bale : ℕ) :
  let new_bales := original_bales * 2
  let original_cost := original_bales * previous_cost_per_bale
  let new_cost := new_bales * better_quality_cost_per_bale
  new_cost - original_cost = 210 :=
by sorry

end additional_cost_l1521_152192


namespace correct_factorization_l1521_152120

-- Define the expressions involved in the options
def option_A (x a b : ℝ) : Prop := x * (a - b) = a * x - b * x
def option_B (x y : ℝ) : Prop := x^2 - 1 + y^2 = (x - 1) * (x + 1) + y^2
def option_C (x : ℝ) : Prop := x^2 - 1 = (x + 1) * (x - 1)
def option_D (x a b c : ℝ) : Prop := a * x + b * x + c = x * (a + b) + c

-- Theorem stating that option C represents true factorization
theorem correct_factorization (x : ℝ) : option_C x := by
  sorry

end correct_factorization_l1521_152120


namespace root_expression_value_l1521_152131

theorem root_expression_value (m : ℝ) (h : 2 * m^2 + 3 * m - 1 = 0) : 4 * m^2 + 6 * m - 2019 = -2017 :=
by
  sorry

end root_expression_value_l1521_152131


namespace total_pages_allowed_l1521_152128

noncomputable def words_total := 48000
noncomputable def words_per_page_large := 1800
noncomputable def words_per_page_small := 2400
noncomputable def pages_large := 4
noncomputable def total_pages : ℕ := 21

theorem total_pages_allowed :
  pages_large * words_per_page_large + (total_pages - pages_large) * words_per_page_small = words_total :=
  by sorry

end total_pages_allowed_l1521_152128


namespace symmetric_point_of_M_neg2_3_l1521_152104

-- Conditions
def symmetric_point (M : ℝ × ℝ) : ℝ × ℝ :=
  (-M.1, -M.2)

-- Main statement
theorem symmetric_point_of_M_neg2_3 :
  symmetric_point (-2, 3) = (2, -3) := 
by
  -- Proof goes here
  sorry

end symmetric_point_of_M_neg2_3_l1521_152104


namespace certain_number_less_32_l1521_152115

theorem certain_number_less_32 (x : ℤ) (h : x - 48 = 22) : x - 32 = 38 :=
by
  sorry

end certain_number_less_32_l1521_152115


namespace total_notebooks_l1521_152114

theorem total_notebooks (num_boxes : ℕ) (parts_per_box : ℕ) (notebooks_per_part : ℕ) (h1 : num_boxes = 22)
  (h2 : parts_per_box = 6) (h3 : notebooks_per_part = 5) : 
  num_boxes * parts_per_box * notebooks_per_part = 660 := 
by
  sorry

end total_notebooks_l1521_152114


namespace percentage_calculation_l1521_152109

theorem percentage_calculation (P : ℕ) (h1 : 0.25 * 16 = 4) 
    (h2 : P / 100 * 40 = 6) : P = 15 :=
by 
    sorry

end percentage_calculation_l1521_152109


namespace parametric_eq_C1_rectangular_eq_C2_max_dist_PQ_l1521_152160

-- Curve C1 given by x^2 / 9 + y^2 = 1, prove its parametric form
theorem parametric_eq_C1 (α : ℝ) : 
  (∃ (x y : ℝ), x = 3 * Real.cos α ∧ y = Real.sin α ∧ (x ^ 2 / 9 + y ^ 2 = 1)) := 
sorry

-- Curve C2 given by ρ^2 - 8ρ sin θ + 15 = 0, prove its rectangular form
theorem rectangular_eq_C2 (ρ θ : ℝ) : 
  (∃ (x y : ℝ), x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ ∧ 
    (ρ ^ 2 - 8 * ρ * Real.sin θ + 15 = 0) ↔ (x ^ 2 + y ^ 2 - 8 * y + 15 = 0)) := 
sorry

-- Prove the maximum value of |PQ|
theorem max_dist_PQ : 
  (∃ (P Q : ℝ × ℝ), 
    (P = (3 * Real.cos α, Real.sin α)) ∧ 
    (Q = (0, 4)) ∧ 
    (∀ α : ℝ, Real.sqrt ((3 * Real.cos α) ^ 2 + (Real.sin α - 4) ^ 2) ≤ 8)) := 
sorry

end parametric_eq_C1_rectangular_eq_C2_max_dist_PQ_l1521_152160


namespace income_on_first_day_l1521_152103

theorem income_on_first_day (income : ℕ → ℚ) (h1 : income 10 = 18)
  (h2 : ∀ n, income (n + 1) = 2 * income n) :
  income 1 = 0.03515625 :=
by
  sorry

end income_on_first_day_l1521_152103


namespace power_function_is_x_cubed_l1521_152188

/-- Define the power function and its property -/
def power_function (a : ℕ) (x : ℝ) : ℝ := x ^ a

/-- The given condition that the function passes through the point (3, 27) -/
def passes_through_point (f : ℝ → ℝ) : Prop :=
  f 3 = 27

/-- Prove that the power function is x^3 -/
theorem power_function_is_x_cubed (f : ℝ → ℝ)
  (h : passes_through_point f) : 
  f = fun x => x ^ 3 := 
by
  sorry -- proof to be filled in

end power_function_is_x_cubed_l1521_152188


namespace batsman_average_after_17th_match_l1521_152172

theorem batsman_average_after_17th_match 
  (A : ℕ) 
  (h1 : (16 * A + 87) / 17 = A + 3) : 
  A + 3 = 39 := 
sorry

end batsman_average_after_17th_match_l1521_152172


namespace intersection_of_lines_l1521_152197

-- Definitions for the lines given by their equations
def line1 (x y : ℝ) : Prop := 5 * x - 3 * y = 9
def line2 (x y : ℝ) : Prop := x^2 + 4 * x - y = 10

-- The statement to prove
theorem intersection_of_lines :
  (line1 2 (1 / 3) ∧ line2 2 (1 / 3)) ∨ (line1 (-3.5) (-8.83) ∧ line2 (-3.5) (-8.83)) :=
by
  sorry

end intersection_of_lines_l1521_152197


namespace grooming_time_correct_l1521_152122

def time_to_groom_poodle : ℕ := 30
def time_to_groom_terrier : ℕ := time_to_groom_poodle / 2
def num_poodles : ℕ := 3
def num_terriers : ℕ := 8

def total_grooming_time : ℕ := 
  (num_poodles * time_to_groom_poodle) + (num_terriers * time_to_groom_terrier)

theorem grooming_time_correct : 
  total_grooming_time = 210 := by
  sorry

end grooming_time_correct_l1521_152122


namespace range_of_a_l1521_152181

noncomputable def cubic_function (x : ℝ) : ℝ := x^3 - 3 * x

theorem range_of_a (a : ℝ) : 
  (∃ x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3 ∧ (cubic_function x1 = a) ∧ (cubic_function x2 = a) ∧ (cubic_function x3 = a)) ↔ (-2 < a ∧ a < 2) :=
by
  sorry

end range_of_a_l1521_152181


namespace negate_exactly_one_even_l1521_152118

def is_even (n : ℕ) : Prop := n % 2 = 0
def is_odd (n : ℕ) : Prop := ¬ is_even n

theorem negate_exactly_one_even (a b c : ℕ) :
  ¬ ((is_even a ∧ is_odd b ∧ is_odd c) ∨ (is_odd a ∧ is_even b ∧ is_odd c) ∨ (is_odd a ∧ is_odd b ∧ is_even c)) ↔ 
  ((is_odd a ∧ is_odd b ∧ is_odd c) ∨ (is_even a ∧ is_even b) ∨ (is_even a ∧ is_even c) ∨ (is_even b ∧ is_even c)) :=
sorry

end negate_exactly_one_even_l1521_152118


namespace square_of_hypotenuse_product_eq_160_l1521_152182

noncomputable def square_of_product_of_hypotenuses (x y : ℝ) (h1 h2 : ℝ) : ℝ :=
  (h1 * h2) ^ 2

theorem square_of_hypotenuse_product_eq_160 :
  ∀ (x y h1 h2 : ℝ),
    (1 / 2) * x * (2 * y) = 4 →
    (1 / 2) * x * y = 8 →
    x^2 + (2 * y)^2 = h1^2 →
    x^2 + y^2 = h2^2 →
    square_of_product_of_hypotenuses x y h1 h2 = 160 :=
by
  intros x y h1 h2 area1 area2 pythagorean1 pythagorean2
  -- The detailed proof steps would go here
  sorry

end square_of_hypotenuse_product_eq_160_l1521_152182


namespace max_value_x_sq_y_l1521_152134

theorem max_value_x_sq_y (x y : ℝ) (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : x + 3 * y = 9) :
  x^2 * y ≤ 36 :=
sorry

end max_value_x_sq_y_l1521_152134


namespace polynomial_evaluation_l1521_152119

theorem polynomial_evaluation (x : ℝ) (h1 : x^2 - 4 * x - 12 = 0) (h2 : 0 < x) : x^3 - 4 * x^2 - 12 * x + 16 = 16 := 
by
  sorry

end polynomial_evaluation_l1521_152119


namespace average_height_males_l1521_152179

theorem average_height_males
  (M W H_m : ℝ)
  (h₀ : W ≠ 0)
  (h₁ : M = 2 * W)
  (h₂ : (M * H_m + W * 170) / (M + W) = 180) :
  H_m = 185 := 
sorry

end average_height_males_l1521_152179


namespace homework_time_decrease_l1521_152100

theorem homework_time_decrease (x : ℝ) :
  let T_initial := 100
  let T_final := 70
  T_initial * (1 - x) * (1 - x) = T_final :=
by
  sorry

end homework_time_decrease_l1521_152100


namespace simplify_complex_fraction_l1521_152147

-- Define the complex numbers involved
def numerator := 3 + 4 * Complex.I
def denominator := 5 - 2 * Complex.I

-- Define what we need to prove: the simplified form
theorem simplify_complex_fraction : 
    (numerator / denominator : Complex) = (7 / 29) + (26 / 29) * Complex.I := 
by
  -- Proof is omitted here
  sorry

end simplify_complex_fraction_l1521_152147


namespace like_terms_exponents_l1521_152159

theorem like_terms_exponents {m n : ℕ} (h1 : 4 * a * b^n = 4 * (a^1) * (b^n)) (h2 : -2 * a^m * b^4 = -2 * (a^m) * (b^4)) :
  (m = 1 ∧ n = 4) :=
by sorry

end like_terms_exponents_l1521_152159


namespace log_eq_l1521_152111

theorem log_eq {a b : ℝ} (h₁ : a = Real.log 256 / Real.log 4) (h₂ : b = Real.log 27 / Real.log 3) : 
  a = (4 / 3) * b :=
by
  sorry

end log_eq_l1521_152111


namespace additional_amount_per_10_cents_l1521_152155

-- Definitions of the given conditions
def expected_earnings_per_share : ℝ := 0.80
def dividend_ratio : ℝ := 0.5
def actual_earnings_per_share : ℝ := 1.10
def shares_owned : ℕ := 600
def total_dividend_paid : ℝ := 312

-- Proof statement
theorem additional_amount_per_10_cents (additional_amount : ℝ) :
  (total_dividend_paid - (shares_owned * (expected_earnings_per_share * dividend_ratio))) / shares_owned / 
  ((actual_earnings_per_share - expected_earnings_per_share) / 0.10) = additional_amount :=
sorry

end additional_amount_per_10_cents_l1521_152155


namespace value_of_x_l1521_152196

-- Let a and b be real numbers.
variable (a b : ℝ)

-- Given conditions
def cond_1 : 10 * a = 6 * b := sorry
def cond_2 : 120 * a * b = 800 := sorry

theorem value_of_x (x : ℝ) (h1 : 10 * a = x) (h2 : 6 * b = x) (h3 : 120 * a * b = 800) : x = 20 :=
sorry

end value_of_x_l1521_152196


namespace solve_exponent_problem_l1521_152158

theorem solve_exponent_problem
  (h : (1 / 8) * (2 ^ 36) = 8 ^ x) : x = 11 :=
by
  sorry

end solve_exponent_problem_l1521_152158


namespace buicks_count_l1521_152148

-- Definitions
def total_cars := 301
def ford_eqn (chevys : ℕ) := 3 + 2 * chevys
def buicks_eqn (chevys : ℕ) := 12 + 8 * chevys

-- Statement
theorem buicks_count (chevys : ℕ) (fords : ℕ) (buicks : ℕ) :
  total_cars = chevys + fords + buicks ∧
  fords = ford_eqn chevys ∧
  buicks = buicks_eqn chevys →
  buicks = 220 :=
by
  intros h
  sorry

end buicks_count_l1521_152148


namespace tape_recorder_cost_l1521_152166

theorem tape_recorder_cost (x y : ℕ) (h1 : 170 ≤ x * y) (h2 : x * y ≤ 195)
  (h3 : (y - 2) * (x + 1) = x * y) : x * y = 180 :=
by
  sorry

end tape_recorder_cost_l1521_152166


namespace simplify_and_rationalize_l1521_152133

theorem simplify_and_rationalize : 
  (1 / (2 + (1 / (Real.sqrt 5 + 2)))) = (Real.sqrt 5 / 5) :=
by
  sorry

end simplify_and_rationalize_l1521_152133


namespace nuts_eaten_condition_not_all_nuts_eaten_l1521_152127

/-- proof problem with conditions and questions --/

-- Let's define the initial setup and the conditions:

def anya_has_all_nuts (nuts : Nat) := nuts > 3

def distribution (a b c : ℕ → ℕ) (n : ℕ) := 
  ((a (n + 1) = b n + c n + (a n % 2)) ∧ 
   (b (n + 1) = a n / 2) ∧ 
   (c (n + 1) = a n / 2))

def nuts_eaten (a b c : ℕ → ℕ) (n : ℕ) := 
  (a n % 2 > 0 ∨ b n % 2 > 0 ∨ c n % 2 > 0)

-- Prove at least one nut will be eaten
theorem nuts_eaten_condition (a b c : ℕ → ℕ) (n : ℕ) :
  anya_has_all_nuts (a 0) → distribution a b c n → nuts_eaten a b c n :=
sorry

-- Prove not all nuts will be eaten
theorem not_all_nuts_eaten (a b c : ℕ → ℕ):
  anya_has_all_nuts (a 0) → distribution a b c n → 
  ¬∀ (n: ℕ), (a n = 0 ∧ b n = 0 ∧ c n = 0) :=
sorry

end nuts_eaten_condition_not_all_nuts_eaten_l1521_152127


namespace product_of_last_two_digits_l1521_152171

theorem product_of_last_two_digits (n A B : ℤ) 
  (h1 : n % 8 = 0) 
  (h2 : 10 * A + B = n % 100) 
  (h3 : A + B = 14) : 
  A * B = 48 := 
sorry

end product_of_last_two_digits_l1521_152171


namespace circle_tangent_radius_l1521_152186

noncomputable def R : ℝ := 4
noncomputable def r : ℝ := 3
noncomputable def O1O2 : ℝ := R + r
noncomputable def r_inscribed : ℝ := (R * r) / O1O2

theorem circle_tangent_radius :
  r_inscribed = (24 : ℝ) / 7 :=
by
  -- The proof would go here
  sorry

end circle_tangent_radius_l1521_152186


namespace consecutive_composites_l1521_152123

theorem consecutive_composites 
  (a t d r : ℕ) (h_a_comp : ∃ p q, p > 1 ∧ q > 1 ∧ a = p * q)
  (h_t_comp : ∃ p q, p > 1 ∧ q > 1 ∧ t = p * q)
  (h_d_comp : ∃ p q, p > 1 ∧ q > 1 ∧ d = p * q)
  (h_r_pos : r > 0) :
  ∃ n : ℕ, n > 0 ∧ ∀ k : ℕ, k < r → ∃ m : ℕ, m > 1 ∧ m ∣ (a * t^(n + k) + d) :=
  sorry

end consecutive_composites_l1521_152123


namespace carson_gold_stars_l1521_152101

theorem carson_gold_stars (yesterday_stars today_total_stars earned_today : ℕ) 
  (h1 : yesterday_stars = 6) 
  (h2 : today_total_stars = 15) 
  (h3 : earned_today = today_total_stars - yesterday_stars) 
  : earned_today = 9 :=
sorry

end carson_gold_stars_l1521_152101


namespace regular_polygons_constructible_l1521_152164

-- Define a right triangle where the smaller leg is half the length of the hypotenuse
structure RightTriangle30_60_90 :=
(smaller_leg hypotenuse : ℝ)
(ratio : smaller_leg = hypotenuse / 2)

-- Define the constructibility of polygons
def canConstructPolygon (n: ℕ) : Prop :=
n = 3 ∨ n = 4 ∨ n = 6 ∨ n = 12

theorem regular_polygons_constructible (T : RightTriangle30_60_90) :
  ∀ n : ℕ, canConstructPolygon n :=
by
  intro n
  sorry

end regular_polygons_constructible_l1521_152164


namespace find_finite_sets_l1521_152105

open Set

theorem find_finite_sets (X : Set ℝ) (h1 : X.Nonempty) (h2 : X.Finite)
  (h3 : ∀ x ∈ X, (x + |x|) ∈ X) :
  ∃ (F : Set ℝ), F.Finite ∧ (∀ x ∈ F, x < 0) ∧ X = insert 0 F :=
sorry

end find_finite_sets_l1521_152105


namespace determine_parallel_planes_l1521_152137

-- Definition of planes and lines with parallelism
structure Plane :=
  (points : Set (ℝ × ℝ × ℝ))

structure Line :=
  (point1 point2 : ℝ × ℝ × ℝ)
  (in_plane : Plane)

def parallel_planes (α β : Plane) : Prop :=
  ∀ (l1 : Line) (l2 : Line), l1.in_plane = α → l2.in_plane = β → (l1 = l2)

def parallel_lines (l1 l2 : Line) : Prop :=
  ∀ p1 p2, l1.point1 = p1 → l1.point2 = p2 → l2.point1 = p1 → l2.point2 = p2


theorem determine_parallel_planes (α β γ : Plane)
  (h1 : parallel_planes γ α)
  (h2 : parallel_planes γ β)
  (l1 l2 : Line)
  (l1_in_alpha : l1.in_plane = α)
  (l2_in_alpha : l2.in_plane = α)
  (parallel_l1_l2 : ¬ (l1 = l2) → parallel_lines l1 l2)
  (l1_parallel_beta : ∀ l, l.in_plane = β → parallel_lines l l1)
  (l2_parallel_beta : ∀ l, l.in_plane = β → parallel_lines l l2) :
  parallel_planes α β := 
sorry

end determine_parallel_planes_l1521_152137


namespace find_natural_numbers_l1521_152187

def LCM (a b : ℕ) : ℕ := (a * b) / (Nat.gcd a b)

theorem find_natural_numbers :
  ∃ a b : ℕ, a + b = 54 ∧ LCM a b - Nat.gcd a b = 114 ∧ (a = 24 ∧ b = 30 ∨ a = 30 ∧ b = 24) := by {
  sorry
}

end find_natural_numbers_l1521_152187


namespace point_P_coordinates_l1521_152117

theorem point_P_coordinates :
  ∃ (P : ℝ × ℝ), P.1 > 0 ∧ P.2 < 0 ∧ abs P.2 = 3 ∧ abs P.1 = 8 ∧ P = (8, -3) :=
sorry

end point_P_coordinates_l1521_152117


namespace total_time_correct_l1521_152143

-- Definitions for the conditions
def dean_time : ℕ := 9
def micah_time : ℕ := (2 * dean_time) / 3
def jake_time : ℕ := micah_time + micah_time / 3

-- Proof statement for the total time
theorem total_time_correct : micah_time + dean_time + jake_time = 23 := by
  sorry

end total_time_correct_l1521_152143


namespace johns_height_in_feet_l1521_152102

def initial_height := 66 -- John's initial height in inches
def growth_rate := 2      -- Growth rate in inches per month
def growth_duration := 3  -- Growth duration in months
def inches_per_foot := 12 -- Conversion factor from inches to feet

def total_growth : ℕ := growth_rate * growth_duration

def final_height_in_inches : ℕ := initial_height + total_growth

-- Now, proof that the final height in feet is 6
theorem johns_height_in_feet : (final_height_in_inches / inches_per_foot) = 6 :=
by {
  -- We would provide the detailed proof here
  sorry
}

end johns_height_in_feet_l1521_152102


namespace turtle_ran_while_rabbit_sleeping_l1521_152110

-- Define the constants and variables used in the problem
def total_distance : ℕ := 1000
def rabbit_speed_multiple : ℕ := 5
def rabbit_behind_distance : ℕ := 10

-- Define a function that represents the turtle's distance run while the rabbit is sleeping
def turtle_distance_while_rabbit_sleeping (total_distance : ℕ) (rabbit_speed_multiple : ℕ) (rabbit_behind_distance : ℕ) : ℕ :=
  total_distance - total_distance / (rabbit_speed_multiple + 1)

-- Prove that the turtle ran 802 meters while the rabbit was sleeping
theorem turtle_ran_while_rabbit_sleeping :
  turtle_distance_while_rabbit_sleeping total_distance rabbit_speed_multiple rabbit_behind_distance = 802 :=
by
  -- We reserve the proof and focus only on the statement
  sorry

end turtle_ran_while_rabbit_sleeping_l1521_152110


namespace unique_function_l1521_152162

def functional_equation (f : ℤ → ℤ) : Prop :=
  ∀ x y : ℤ, f (-f x - f y) = 1 - x - y

theorem unique_function :
  ∀ f : ℤ → ℤ, (functional_equation f) → (∀ x : ℤ, f x = x - 1) :=
by
  intros f h
  sorry

end unique_function_l1521_152162


namespace complex_values_l1521_152175

open Complex

theorem complex_values (z : ℂ) (h : z ^ 3 + z = 2 * (abs z) ^ 2) :
  z = 0 ∨ z = 1 ∨ z = -1 + 2 * Complex.I ∨ z = -1 - 2 * Complex.I :=
by sorry

end complex_values_l1521_152175


namespace value_of_a_value_of_sin_A_plus_pi_over_4_l1521_152126

section TriangleABC

variables {a b c A B : ℝ}
variables (h_b : b = 3) (h_c : c = 1) (h_A_eq_2B : A = 2 * B)

theorem value_of_a : a = 2 * Real.sqrt 3 :=
sorry

theorem value_of_sin_A_plus_pi_over_4 : Real.sin (A + π / 4) = (4 - Real.sqrt 2) / 6 :=
sorry

end TriangleABC

end value_of_a_value_of_sin_A_plus_pi_over_4_l1521_152126


namespace value_of_a_l1521_152108

theorem value_of_a (a : ℤ) (h0 : 0 ≤ a) (h1 : a < 13) (h2 : 13 ∣ 12^20 + a) : a = 12 :=
by sorry

end value_of_a_l1521_152108


namespace point_on_line_eq_l1521_152198

theorem point_on_line_eq (a b : ℝ) (h : b = -3 * a - 4) : b + 3 * a + 4 = 0 :=
by
  sorry

end point_on_line_eq_l1521_152198


namespace percentage_decrease_in_area_l1521_152130

variable (L B : ℝ)

def original_area (L B : ℝ) : ℝ := L * B
def new_length (L : ℝ) : ℝ := 0.70 * L
def new_breadth (B : ℝ) : ℝ := 0.85 * B
def new_area (L B : ℝ) : ℝ := new_length L * new_breadth B

theorem percentage_decrease_in_area (L B : ℝ) (hL : L > 0) (hB : B > 0) :
  ((original_area L B - new_area L B) / original_area L B) * 100 = 40.5 :=
by
  sorry

end percentage_decrease_in_area_l1521_152130


namespace transformed_roots_polynomial_l1521_152146

-- Given conditions
variables {a b c : ℝ}
variables (h : ∀ x, (x - a) * (x - b) * (x - c) = x^3 - 4 * x + 6)

-- Prove the equivalent polynomial with the transformed roots
theorem transformed_roots_polynomial :
  (∀ x, (x - (a - 3)) * (x - (b - 3)) * (x - (c - 3)) = x^3 + 9 * x^2 + 23 * x + 21) :=
sorry

end transformed_roots_polynomial_l1521_152146


namespace question1_question2_l1521_152116

section problem1

variable (a b : ℝ)

theorem question1 (h1 : a = 1) (h2 : b = 2) : 
  ∀ x : ℝ, abs (2 * x + 1) + abs (3 * x - 2) ≤ 5 ↔ 
  (-4 / 5 ≤ x ∧ x ≤ 6 / 5) :=
sorry

end problem1

section problem2

theorem question2 :
  (∀ x : ℝ, abs (x - 1) + abs (x + 2) ≥ m^2 - 3 * m + 5) → 
  ∃ (m : ℝ), m ≤ 2 :=
sorry

end problem2

end question1_question2_l1521_152116


namespace log_proof_l1521_152190

noncomputable def log_base (b x : ℝ) : ℝ :=
  Real.log x / Real.log b

theorem log_proof (x : ℝ) (h : log_base 7 (x + 6) = 2) : log_base 13 x = log_base 13 43 :=
by
  sorry

end log_proof_l1521_152190


namespace problem_statement_l1521_152176

theorem problem_statement (a b c : ℝ) (h : a * c^2 > b * c^2) (hc : c ≠ 0) : 
  a > b :=
by 
  sorry

end problem_statement_l1521_152176


namespace copy_pages_cost_l1521_152121

theorem copy_pages_cost :
  (7 : ℕ) * (n : ℕ) = 3500 * 4 / 7 → n = 2000 :=
by
  sorry

end copy_pages_cost_l1521_152121


namespace intersection_eq_neg1_l1521_152150

open Set

noncomputable def setA : Set Int := {x : Int | x^2 - 1 ≤ 0}
def setB : Set Int := {x : Int | x^2 - x - 2 = 0}

theorem intersection_eq_neg1 : setA ∩ setB = {-1} := by
  sorry

end intersection_eq_neg1_l1521_152150


namespace exponent_multiplication_rule_l1521_152178

theorem exponent_multiplication_rule :
  3000 * (3000 ^ 3000) = 3000 ^ 3001 := 
by {
  sorry
}

end exponent_multiplication_rule_l1521_152178


namespace two_digits_same_in_three_digit_numbers_l1521_152168

theorem two_digits_same_in_three_digit_numbers (h1 : (100 : ℕ) ≤ n) (h2 : n < 600) : 
  ∃ n, n = 140 := sorry

end two_digits_same_in_three_digit_numbers_l1521_152168


namespace quadrilateral_angle_W_l1521_152199

theorem quadrilateral_angle_W (W X Y Z : ℝ) 
  (h₁ : W = 3 * X) 
  (h₂ : W = 4 * Y) 
  (h₃ : W = 6 * Z) 
  (sum_angles : W + X + Y + Z = 360) : 
  W = 1440 / 7 := by
sorry

end quadrilateral_angle_W_l1521_152199


namespace count_triangles_l1521_152183

-- Define the problem conditions
def num_small_triangles : ℕ := 11
def num_medium_triangles : ℕ := 4
def num_large_triangles : ℕ := 1

-- Define the main statement asserting the total number of triangles
theorem count_triangles (small : ℕ) (medium : ℕ) (large : ℕ) :
  small = num_small_triangles →
  medium = num_medium_triangles →
  large = num_large_triangles →
  small + medium + large = 16 :=
by
  intros h_small h_medium h_large
  rw [h_small, h_medium, h_large]
  sorry

end count_triangles_l1521_152183


namespace sphere_surface_area_ratio_l1521_152139

axiom prism_has_circumscribed_sphere : Prop
axiom prism_has_inscribed_sphere : Prop

theorem sphere_surface_area_ratio 
  (h1 : prism_has_circumscribed_sphere)
  (h2 : prism_has_inscribed_sphere) : 
  ratio_surface_area_of_circumscribed_to_inscribed_sphere = 5 :=
sorry

end sphere_surface_area_ratio_l1521_152139


namespace polygon_sides_l1521_152132

theorem polygon_sides (n : ℕ) 
  (h1 : (n - 2) * 180 = 3 * 360)
  (h2 : n ≥ 3) : 
  n = 8 := 
by
  sorry

end polygon_sides_l1521_152132


namespace prime_factors_2310_l1521_152124

-- Define the number in question
def number : ℕ := 2310

-- The main theorem statement
theorem prime_factors_2310 : ∃ s : Finset ℕ, (∀ p ∈ s, Nat.Prime p) ∧ 
Finset.prod s id = number ∧ Finset.card s = 5 := by
  sorry

end prime_factors_2310_l1521_152124


namespace original_number_j_l1521_152144

noncomputable def solution (n : ℚ) : ℚ := (3 * (n + 3) - 5) / 3

theorem original_number_j { n : ℚ } (h : solution n = 10) : n = 26 / 3 :=
by
  sorry

end original_number_j_l1521_152144


namespace length_of_PS_l1521_152107

theorem length_of_PS
  (PT TR QT TS PQ : ℝ)
  (h1 : PT = 5)
  (h2 : TR = 7)
  (h3 : QT = 9)
  (h4 : TS = 4)
  (h5 : PQ = 7) :
  PS = Real.sqrt 66.33 := 
  sorry

end length_of_PS_l1521_152107


namespace math_problem_equivalence_l1521_152184

theorem math_problem_equivalence :
  (-3 : ℚ) / (-1 - 3 / 4) * (3 / 4) / (3 / 7) = 3 := 
by 
  sorry

end math_problem_equivalence_l1521_152184


namespace fraction_female_to_male_fraction_male_to_total_l1521_152149

-- Define the number of male and female students
def num_male_students : ℕ := 30
def num_female_students : ℕ := 24
def total_students : ℕ := num_male_students + num_female_students

-- Prove the fraction of female students to male students
theorem fraction_female_to_male :
  (num_female_students : ℚ) / num_male_students = 4 / 5 :=
by sorry

-- Prove the fraction of male students to total students
theorem fraction_male_to_total :
  (num_male_students : ℚ) / total_students = 5 / 9 :=
by sorry

end fraction_female_to_male_fraction_male_to_total_l1521_152149


namespace fraction_calculation_l1521_152136

theorem fraction_calculation : (4 / 9 + 1 / 9) / (5 / 8 - 1 / 8) = 10 / 9 := by
  sorry

end fraction_calculation_l1521_152136


namespace ellipse_condition_l1521_152151

variables (m n : ℝ)

-- Definition of the curve
def curve_eqn (x y : ℝ) := m * x^2 + n * y^2 = 1

-- Define the condition for being an ellipse
def is_ellipse (m n : ℝ) : Prop :=
  m > 0 ∧ n > 0 ∧ m ≠ n

def mn_positive (m n : ℝ) : Prop := m * n > 0

-- Prove that mn > 0 is a necessary but not sufficient condition
theorem ellipse_condition (m n : ℝ) : mn_positive m n → is_ellipse m n → False := sorry

end ellipse_condition_l1521_152151


namespace xy_equals_one_l1521_152142

-- Define the mathematical theorem
theorem xy_equals_one (x y : ℝ) (h : x + y = 1 / x + 1 / y) (h₂ : x + y ≠ 0) : x * y = 1 := 
by
  sorry

end xy_equals_one_l1521_152142


namespace cos_120_eq_neg_half_l1521_152177

theorem cos_120_eq_neg_half : Real.cos (120 * Real.pi / 180) = -1/2 := by
  sorry

end cos_120_eq_neg_half_l1521_152177
