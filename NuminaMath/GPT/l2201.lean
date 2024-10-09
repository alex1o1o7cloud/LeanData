import Mathlib

namespace relationship_between_A_and_p_l2201_220164

variable {x y p : ℝ}

theorem relationship_between_A_and_p (h1 : x ≠ 0) (h2 : y ≠ 0)
  (h3 : x ≠ y * 2) (h4 : x ≠ p * y)
  (A : ℝ) (hA : A = (x^2 - 3 * y^2) / (3 * x^2 + y^2))
  (hEq : (p * x * y) / (x^2 - (2 + p) * x * y + 2 * p * y^2) - y / (x - 2 * y) = 1 / 2) :
  A = (9 * p^2 - 3) / (27 * p^2 + 1) := 
sorry

end relationship_between_A_and_p_l2201_220164


namespace glee_club_female_members_l2201_220167

theorem glee_club_female_members (m f : ℕ) 
  (h1 : f = 2 * m) 
  (h2 : m + f = 18) : 
  f = 12 :=
by
  sorry

end glee_club_female_members_l2201_220167


namespace inscribed_circle_diameter_of_right_triangle_l2201_220114

theorem inscribed_circle_diameter_of_right_triangle (a b : ℕ) (hc : a = 8) (hb : b = 15) :
  2 * (60 / (a + b + Int.sqrt (a ^ 2 + b ^ 2))) = 6 :=
by
  sorry

end inscribed_circle_diameter_of_right_triangle_l2201_220114


namespace notebooks_multiple_of_3_l2201_220130

theorem notebooks_multiple_of_3 (N : ℕ) (h1 : ∃ k : ℕ, N = 3 * k) :
  ∃ k : ℕ, N = 3 * k :=
by
  sorry

end notebooks_multiple_of_3_l2201_220130


namespace last_three_digits_of_7_pow_120_l2201_220139

theorem last_three_digits_of_7_pow_120 :
  7^120 % 1000 = 681 :=
by
  sorry

end last_three_digits_of_7_pow_120_l2201_220139


namespace inequality_solution_set_l2201_220111

theorem inequality_solution_set (m n : ℝ) 
    (h₁ : ∀ x : ℝ, mx - n > 0 ↔ x < 1 / 3) 
    (h₂ : m + n < 0) 
    (h₃ : m = 3 * n) 
    (h₄ : n < 0) : 
    ∀ x : ℝ, (m + n) * x < n - m ↔ x > -1 / 2 :=
by
  sorry

end inequality_solution_set_l2201_220111


namespace base_b_for_three_digits_l2201_220199

theorem base_b_for_three_digits (b : ℕ) : b = 7 ↔ b^2 ≤ 256 ∧ 256 < b^3 := by
  sorry

end base_b_for_three_digits_l2201_220199


namespace roots_condition_l2201_220148

theorem roots_condition (m : ℝ) : 
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x1 > 3 ∧ x2 < 3 ∧ x1^2 - m * x1 + 2 * m = 0 ∧ x2^2 - m * x2 + 2 * m = 0) ↔ m > 9 :=
by sorry

end roots_condition_l2201_220148


namespace least_value_expression_l2201_220129

open Real

theorem least_value_expression (x : ℝ) : 
  let expr := (x + 1) * (x + 2) * (x + 4) * (x + 5) + 2023 + 3 * cos (2 * x)
  ∃ a : ℝ, expr = a ∧ ∀ b : ℝ, b < a → False :=
sorry

end least_value_expression_l2201_220129


namespace complex_division_l2201_220113

-- Define complex numbers and imaginary unit
def i : ℂ := Complex.I

theorem complex_division : (3 + 4 * i) / (1 + i) = (7 / 2) + (1 / 2) * i :=
by
  sorry

end complex_division_l2201_220113


namespace sum_reciprocal_inequality_l2201_220140

theorem sum_reciprocal_inequality (p q a b c d e : ℝ) (hp : 0 < p) (ha : p ≤ a) (hb : p ≤ b) (hc : p ≤ c) (hd : p ≤ d) (he : p ≤ e) (haq : a ≤ q) (hbq : b ≤ q) (hcq : c ≤ q) (hdq : d ≤ q) (heq : e ≤ q) :
  (a + b + c + d + e) * (1 / a + 1 / b + 1 / c + 1 / d + 1 / e) ≤ 25 + 6 * ((Real.sqrt (q / p) - Real.sqrt (p / q)) ^ 2) :=
by sorry

end sum_reciprocal_inequality_l2201_220140


namespace caltech_equilateral_triangles_l2201_220134

theorem caltech_equilateral_triangles (n : ℕ) (h : n = 900) :
  let total_triangles := (n * (n - 1) / 2) * 2
  let overcounted_triangles := n / 3
  total_triangles - overcounted_triangles = 808800 :=
by
  sorry

end caltech_equilateral_triangles_l2201_220134


namespace sneaker_final_price_l2201_220152

-- Definitions of the conditions
def original_price : ℝ := 120
def coupon_value : ℝ := 10
def discount_percent : ℝ := 0.1

-- The price after the coupon is applied
def price_after_coupon := original_price - coupon_value

-- The membership discount amount
def membership_discount := price_after_coupon * discount_percent

-- The final price the man will pay
def final_price := price_after_coupon - membership_discount

theorem sneaker_final_price : final_price = 99 := by
  sorry

end sneaker_final_price_l2201_220152


namespace smallest_base_l2201_220125

theorem smallest_base (b : ℕ) (n : ℕ) : (n = 512) → (b^3 ≤ n ∧ n < b^4) → ((n / b^3) % b + 1) % 2 = 0 → b = 6 := sorry

end smallest_base_l2201_220125


namespace find_sample_size_l2201_220102

theorem find_sample_size :
  ∀ (n : ℕ), 
    (∃ x : ℝ,
      2 * x + 3 * x + 4 * x + 6 * x + 4 * x + x = 1 ∧
      2 * n * x + 3 * n * x + 4 * n * x = 27) →
    n = 60 :=
by
  intro n
  rintro ⟨x, h1, h2⟩
  sorry

end find_sample_size_l2201_220102


namespace plant_supplier_money_left_correct_l2201_220156

noncomputable def plant_supplier_total_earnings : ℕ :=
  35 * 52 + 30 * 32 + 20 * 77 + 25 * 22 + 40 * 15

noncomputable def plant_supplier_total_expenses : ℕ :=
  3 * 65 + 2 * 45 + 280 + 150 + 100 + 125 + 225 + 550

noncomputable def plant_supplier_money_left : ℕ :=
  plant_supplier_total_earnings - plant_supplier_total_expenses

theorem plant_supplier_money_left_correct :
  plant_supplier_money_left = 3755 :=
by
  sorry

end plant_supplier_money_left_correct_l2201_220156


namespace function_inverse_overlap_form_l2201_220165

theorem function_inverse_overlap_form (a b c d : ℝ) (h : ¬(a = 0 ∧ c = 0)) : 
  (∀ x, (c * x + d) * (dx - b) = (a * x + b) * (-c * x + a)) → 
  (∃ f : ℝ → ℝ, (∀ x, f x = x ∨ f x = (a * x + b) / (c * x - a))) :=
by 
  sorry

end function_inverse_overlap_form_l2201_220165


namespace problem1_problem2_problem3_l2201_220126

-- Problem 1 Statement
theorem problem1 : (π - 3.14)^0 + (1 / 2)^(-1) + (-1)^(2023) = 2 :=
by {
  -- use tactic mode to assist the proof
  sorry
}

-- Problem 2 Statement
theorem problem2 (b : ℝ) : (-b)^2 * b + 6 * b^4 / (2 * b) + (-2 * b)^3 = -4 * b^3 :=
by {
  -- use tactic mode to assist the proof
  sorry
}

-- Problem 3 Statement
theorem problem3 (x : ℝ) : (x - 1)^2 - x * (x + 2) = -4 * x + 1 :=
by {
  -- use tactic mode to assist the proof
  sorry
}

end problem1_problem2_problem3_l2201_220126


namespace percentage_increase_first_job_percentage_increase_second_job_percentage_increase_third_job_l2201_220145

theorem percentage_increase_first_job :
  let old_salary := 65
  let new_salary := 70
  (new_salary - old_salary) / old_salary * 100 = 7.69 := by
  sorry

theorem percentage_increase_second_job :
  let old_salary := 120
  let new_salary := 138
  (new_salary - old_salary) / old_salary * 100 = 15 := by
  sorry

theorem percentage_increase_third_job :
  let old_salary := 200
  let new_salary := 220
  (new_salary - old_salary) / old_salary * 100 = 10 := by
  sorry

end percentage_increase_first_job_percentage_increase_second_job_percentage_increase_third_job_l2201_220145


namespace team_arrangements_l2201_220127

noncomputable def factorial : ℕ → ℕ
| 0       => 1
| (n + 1) => (n + 1) * factorial n

theorem team_arrangements :
  let num_players := 10
  let team_blocks := 4
  let cubs_players := 3
  let red_sox_players := 3
  let yankees_players := 2
  let dodgers_players := 2
  (factorial team_blocks) * (factorial cubs_players) * (factorial red_sox_players) * (factorial yankees_players) * (factorial dodgers_players) = 3456 := 
by
  -- Proof steps will be inserted here
  sorry

end team_arrangements_l2201_220127


namespace ihsan_children_l2201_220138

theorem ihsan_children :
  ∃ n : ℕ, (n + n^2 + n^3 + n^4 = 2800) ∧ (n = 7) :=
sorry

end ihsan_children_l2201_220138


namespace geometric_sequence_fifth_term_l2201_220109

theorem geometric_sequence_fifth_term (r : ℕ) (h₁ : 5 * r^3 = 405) : 5 * r^4 = 405 :=
sorry

end geometric_sequence_fifth_term_l2201_220109


namespace probability_four_friends_same_group_l2201_220118

-- Define the conditions of the problem
def total_students : ℕ := 900
def groups : ℕ := 5
def friends : ℕ := 4
def probability_per_group : ℚ := 1 / groups

-- Define the statement we need to prove
theorem probability_four_friends_same_group :
  (probability_per_group * probability_per_group * probability_per_group) = 1 / 125 :=
sorry

end probability_four_friends_same_group_l2201_220118


namespace cos_beta_value_l2201_220191

theorem cos_beta_value (α β : ℝ) (hα1 : 0 < α ∧ α < π/2) (hβ1 : 0 < β ∧ β < π/2) 
  (h1 : Real.sin α = 4/5) (h2 : Real.cos (α + β) = -12/13) : 
  Real.cos β = -16/65 := 
by 
  sorry

end cos_beta_value_l2201_220191


namespace cubes_with_no_colored_faces_l2201_220101

theorem cubes_with_no_colored_faces (width length height : ℕ) (total_cubes cube_side : ℕ) :
  width = 6 ∧ length = 5 ∧ height = 4 ∧ total_cubes = 120 ∧ cube_side = 1 →
  (width - 2) * (length - 2) * (height - 2) = 24 :=
by
  intros h
  sorry

end cubes_with_no_colored_faces_l2201_220101


namespace infinite_integer_triples_solution_l2201_220153

theorem infinite_integer_triples_solution (a b c : ℤ) : 
  ∃ (a b c : ℤ), ∀ n : ℤ, a^2 + b^2 = c^2 + 3 :=
sorry

end infinite_integer_triples_solution_l2201_220153


namespace smallest_solution_l2201_220168

theorem smallest_solution (x : ℝ) (h : x^4 - 16 * x^2 + 63 = 0) :
  x = -3 :=
sorry

end smallest_solution_l2201_220168


namespace passes_through_1_1_l2201_220147

theorem passes_through_1_1 (a : ℝ) (h_pos : a > 0) (h_ne : a ≠ 1) : (1, 1) ∈ {p : ℝ × ℝ | ∃ x : ℝ, p = (x, a^ (x - 1))} :=
by
  -- proof not required
  sorry

end passes_through_1_1_l2201_220147


namespace curve_symmetric_about_y_eq_x_l2201_220131

def curve_eq (x y : ℝ) : Prop := x * y * (x + y) = 1

theorem curve_symmetric_about_y_eq_x :
  ∀ (x y : ℝ), curve_eq x y ↔ curve_eq y x :=
by sorry

end curve_symmetric_about_y_eq_x_l2201_220131


namespace payment_per_minor_character_l2201_220144

noncomputable def M : ℝ := 285000 / 19 

theorem payment_per_minor_character
    (num_main_characters : ℕ := 5)
    (num_minor_characters : ℕ := 4)
    (total_payment : ℝ := 285000)
    (payment_ratio : ℝ := 3)
    (eq1 : 5 * 3 * M + 4 * M = total_payment) :
    M = 15000 :=
by
  sorry

end payment_per_minor_character_l2201_220144


namespace sum_of_digits_of_77_is_14_l2201_220171

-- Define the conditions given in the problem
def triangular_array_sum (N : ℕ) : ℕ := N * (N + 1) / 2

-- Define what it means to be the sum of the digits of a number
def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.foldl (· + ·) 0

-- The actual Lean theorem statement
theorem sum_of_digits_of_77_is_14 (N : ℕ) (h : triangular_array_sum N = 3003) : sum_of_digits N = 14 :=
by
  sorry  -- Proof to be completed here

end sum_of_digits_of_77_is_14_l2201_220171


namespace factors_of_2520_l2201_220180

theorem factors_of_2520 : (∃ (factors : Finset ℕ), factors.card = 48 ∧ ∀ d, d ∈ factors ↔ d > 0 ∧ 2520 % d = 0) :=
sorry

end factors_of_2520_l2201_220180


namespace lcm_of_9_12_15_l2201_220107

theorem lcm_of_9_12_15 : Nat.lcm (Nat.lcm 9 12) 15 = 180 := by
  sorry

end lcm_of_9_12_15_l2201_220107


namespace percent_singles_l2201_220176

theorem percent_singles (total_hits home_runs triples doubles : ℕ) 
  (h_total: total_hits = 50) 
  (h_hr: home_runs = 3) 
  (h_tr: triples = 2) 
  (h_double: doubles = 8) : 
  100 * (total_hits - (home_runs + triples + doubles)) / total_hits = 74 := 
by
  -- proofs
  sorry

end percent_singles_l2201_220176


namespace final_balance_is_60_million_l2201_220124

-- Define the initial conditions
def initial_balance : ℕ := 100
def earnings_from_selling_players : ℕ := 2 * 10
def cost_of_buying_players : ℕ := 4 * 15

-- Define the final balance calculation and state the theorem
theorem final_balance_is_60_million : initial_balance + earnings_from_selling_players - cost_of_buying_players = 60 := by
  sorry

end final_balance_is_60_million_l2201_220124


namespace shapes_values_correct_l2201_220122

-- Define variable types and conditions
variables (x y z w : ℕ)
variables (sum1 sum2 sum3 sum4 T : ℕ)

-- Define the conditions for the problem as given in (c)
axiom row_sum1 : x + y + z = sum1
axiom row_sum2 : y + z + w = sum2
axiom row_sum3 : z + w + x = sum3
axiom row_sum4 : w + x + y = sum4
axiom col_sum  : x + y + z + w = T

-- Define the variables with specific values as determined in the solution
def triangle := 2
def square := 0
def a_tilde := 6
def O_value := 1

-- Prove that the assigned values satisfy the conditions
theorem shapes_values_correct :
  x = triangle ∧ y = square ∧ z = a_tilde ∧ w = O_value :=
by { sorry }

end shapes_values_correct_l2201_220122


namespace ratio_of_dinner_to_lunch_l2201_220104

theorem ratio_of_dinner_to_lunch
  (dinner: ℕ) (lunch: ℕ) (breakfast: ℕ) (k: ℕ)
  (h1: dinner = 240)
  (h2: dinner = k * lunch)
  (h3: dinner = 6 * breakfast)
  (h4: breakfast + lunch + dinner = 310) :
  dinner / lunch = 8 :=
by
  -- Proof to be completed
  sorry

end ratio_of_dinner_to_lunch_l2201_220104


namespace arithmetic_sequence_common_difference_l2201_220157

variable {a : ℕ → ℝ}
variable {d : ℝ}

-- Statement of the problem
theorem arithmetic_sequence_common_difference
  (h1 : a 2 + a 6 = 8)
  (h2 : a 3 + a 4 = 3)
  (h_arith : ∀ n, a (n+1) = a n + d) :
  d = 5 := by
  sorry

end arithmetic_sequence_common_difference_l2201_220157


namespace triangle_height_l2201_220195

theorem triangle_height (base height : ℝ) (area : ℝ) (h1 : base = 2) (h2 : area = 3) (area_formula : area = (base * height) / 2) : height = 3 :=
by
  sorry

end triangle_height_l2201_220195


namespace ninth_grade_class_notification_l2201_220158

theorem ninth_grade_class_notification (n : ℕ) (h1 : 1 + n + n * n = 43) : n = 6 :=
by
  sorry

end ninth_grade_class_notification_l2201_220158


namespace intersection_complement_l2201_220108

open Set

variable (x : ℝ)

def M : Set ℝ := { x | -1 < x ∧ x < 2 }
def N : Set ℝ := { x | 1 ≤ x }

theorem intersection_complement :
  M ∩ (univ \ N) = { x | -1 < x ∧ x < 1 } := by
  sorry

end intersection_complement_l2201_220108


namespace slope_angle_y_eq_neg1_l2201_220169

theorem slope_angle_y_eq_neg1 : (∃ line : ℝ → ℝ, ∀ y: ℝ, line y = -1 → ∃ θ : ℝ, θ = 0) :=
by
  -- Sorry is used to skip the proof.
  sorry

end slope_angle_y_eq_neg1_l2201_220169


namespace max_volume_of_hollow_cube_l2201_220198

/-- 
We have 1000 solid cubes with edge lengths of 1 unit each. 
The small cubes can be glued together but not cut. 
The cube to be created is hollow with a wall thickness of 1 unit.
Prove that the maximum external volume of the cube we can create is 2197 cubic units.
--/

theorem max_volume_of_hollow_cube :
  ∃ x : ℕ, 6 * x^2 - 12 * x + 8 ≤ 1000 ∧ x^3 = 2197 :=
sorry

end max_volume_of_hollow_cube_l2201_220198


namespace simplify_expression_l2201_220177

theorem simplify_expression (b : ℝ) : 3 * b * (3 * b^2 + 2 * b - 4) - 2 * b^2 = 9 * b^3 + 4 * b^2 - 12 * b :=
by sorry

end simplify_expression_l2201_220177


namespace container_volume_ratio_l2201_220132

theorem container_volume_ratio (C D : ℝ) (hC: C > 0) (hD: D > 0)
  (h: (3/4) * C = (5/8) * D) : (C / D) = (5 / 6) :=
by
  sorry

end container_volume_ratio_l2201_220132


namespace least_number_to_subtract_l2201_220117

theorem least_number_to_subtract (n d : ℕ) (n_val : n = 13602) (d_val : d = 87) : 
  ∃ r, (n - r) % d = 0 ∧ r = 30 := by
  sorry

end least_number_to_subtract_l2201_220117


namespace p_is_sufficient_but_not_necessary_for_q_l2201_220188

-- Definitions and conditions
def p (x : ℝ) : Prop := (x = 1)
def q (x : ℝ) : Prop := (x^2 - 3 * x + 2 = 0)

-- Theorem statement
theorem p_is_sufficient_but_not_necessary_for_q : ∀ x : ℝ, (p x → q x) ∧ (¬ (q x → p x)) :=
by
  sorry

end p_is_sufficient_but_not_necessary_for_q_l2201_220188


namespace boys_at_reunion_l2201_220146

theorem boys_at_reunion (n : ℕ) (H : n * (n - 1) / 2 = 45) : n = 10 :=
by sorry

end boys_at_reunion_l2201_220146


namespace people_per_table_l2201_220112

theorem people_per_table (kids adults tables : ℕ) (h_kids : kids = 45) (h_adults : adults = 123) (h_tables : tables = 14) :
  ((kids + adults) / tables) = 12 :=
by
  -- Placeholder for proof
  sorry

end people_per_table_l2201_220112


namespace line_through_origin_and_intersection_eq_x_y_l2201_220170

theorem line_through_origin_and_intersection_eq_x_y :
  ∀ (x y : ℝ), (x - 2 * y + 2 = 0) ∧ (2 * x - y - 2 = 0) →
  ∃ m b : ℝ, m = 1 ∧ b = 0 ∧ (y = m * x + b) :=
by
  sorry

end line_through_origin_and_intersection_eq_x_y_l2201_220170


namespace minute_hand_angle_is_pi_six_minute_hand_arc_length_is_2pi_third_l2201_220160

theorem minute_hand_angle_is_pi_six (radius : ℝ) (fast_min : ℝ) (h1 : radius = 4) (h2 : fast_min = 5) :
  (fast_min / 60 * 2 * Real.pi = Real.pi / 6) :=
by sorry

theorem minute_hand_arc_length_is_2pi_third (radius : ℝ) (angle : ℝ) (fast_min : ℝ) (h1 : radius = 4) (h2 : angle = Real.pi / 6) (h3 : fast_min = 5) :
  (radius * angle = 2 * Real.pi / 3) :=
by sorry

end minute_hand_angle_is_pi_six_minute_hand_arc_length_is_2pi_third_l2201_220160


namespace valid_outfit_choices_l2201_220106

def shirts := 6
def pants := 6
def hats := 12
def patterned_hats := 6

theorem valid_outfit_choices : 
  (shirts * pants * hats) - shirts - (patterned_hats * shirts * (pants - 1)) = 246 := by
  sorry

end valid_outfit_choices_l2201_220106


namespace Alex_runs_faster_l2201_220142

def Rick_speed : ℚ := 5
def Jen_speed : ℚ := (3 / 4) * Rick_speed
def Mark_speed : ℚ := (4 / 3) * Jen_speed
def Alex_speed : ℚ := (5 / 6) * Mark_speed

theorem Alex_runs_faster : Alex_speed = 25 / 6 :=
by
  -- Proof is skipped
  sorry

end Alex_runs_faster_l2201_220142


namespace stratified_sampling_correct_l2201_220119

-- Defining the conditions
def first_grade_students : ℕ := 600
def second_grade_students : ℕ := 680
def third_grade_students : ℕ := 720
def total_sample_size : ℕ := 50
def total_students := first_grade_students + second_grade_students + third_grade_students

-- Expected number of students to be sampled from first, second, and third grades
def expected_first_grade_sample := total_sample_size * first_grade_students / total_students
def expected_second_grade_sample := total_sample_size * second_grade_students / total_students
def expected_third_grade_sample := total_sample_size * third_grade_students / total_students

-- Main theorem statement
theorem stratified_sampling_correct :
  expected_first_grade_sample = 15 ∧
  expected_second_grade_sample = 17 ∧
  expected_third_grade_sample = 18 := by
  sorry

end stratified_sampling_correct_l2201_220119


namespace dress_designs_count_l2201_220175

theorem dress_designs_count :
  let colors := 5
  let patterns := 4
  let sizes := 3
  colors * patterns * sizes = 60 :=
by
  let colors := 5
  let patterns := 4
  let sizes := 3
  have h : colors * patterns * sizes = 60 := by norm_num
  exact h

end dress_designs_count_l2201_220175


namespace at_least_4_stayed_l2201_220187

-- We define the number of people and their respective probabilities of staying.
def numPeople : ℕ := 8
def numCertain : ℕ := 5
def numUncertain : ℕ := 3
def probUncertainStay : ℚ := 1 / 3

-- We state the problem formally:
theorem at_least_4_stayed :
  (probUncertainStay ^ 3 * 3 + (probUncertainStay ^ 2 * (2 / 3) * 3) + (probUncertainStay * (2 / 3)^2 * 3)) = 19 / 27 :=
by
  sorry

end at_least_4_stayed_l2201_220187


namespace simplify_and_evaluate_l2201_220178

-- Define the condition as a predicate
def condition (a b : ℝ) : Prop := (a + 1/2)^2 + |b - 2| = 0

-- The simplified expression
def simplified_expression (a b : ℝ) : ℝ := 12 * a^2 * b - 6 * a * b^2

-- Statement: Given the condition, prove that the simplified expression evaluates to 18
theorem simplify_and_evaluate : ∀ (a b : ℝ), condition a b → simplified_expression a b = 18 :=
by
  intros a b hc
  sorry  -- Proof omitted

end simplify_and_evaluate_l2201_220178


namespace correct_number_of_arrangements_l2201_220190

def arrangements_with_conditions (n : ℕ) : ℕ := 
  if n = 6 then
    let case1 := 120  -- when B is at the far right
    let case2 := 96   -- when A is at the far right
    case1 + case2
  else 0

theorem correct_number_of_arrangements : arrangements_with_conditions 6 = 216 :=
by {
  -- The detailed proof is omitted here
  sorry
}

end correct_number_of_arrangements_l2201_220190


namespace remainder_when_divided_by_7_l2201_220193

-- Definitions based on conditions
def k_condition (k : ℕ) : Prop :=
(k % 5 = 2) ∧ (k % 6 = 5) ∧ (k < 38)

-- Theorem based on the question and correct answer
theorem remainder_when_divided_by_7 {k : ℕ} (h : k_condition k) : k % 7 = 3 :=
sorry

end remainder_when_divided_by_7_l2201_220193


namespace solve_for_n_l2201_220100

theorem solve_for_n (n : ℕ) (h : (8 ^ n) * (8 ^ n) * (8 ^ n) = 64 ^ 3) : n = 2 :=
by sorry

end solve_for_n_l2201_220100


namespace floor_sub_le_l2201_220103

theorem floor_sub_le : ∀ (x y : ℝ), ⌊x - y⌋ ≤ ⌊x⌋ - ⌊y⌋ :=
by sorry

end floor_sub_le_l2201_220103


namespace sin_double_angle_given_cos_identity_l2201_220173

theorem sin_double_angle_given_cos_identity (α : ℝ) 
  (h : Real.cos (α + π / 4) = Real.sqrt 2 / 4) : 
  Real.sin (2 * α) = 3 / 4 :=
by
  sorry

end sin_double_angle_given_cos_identity_l2201_220173


namespace depth_of_river_bank_l2201_220155

theorem depth_of_river_bank (top_width bottom_width area depth : ℝ) 
  (h₁ : top_width = 12)
  (h₂ : bottom_width = 8)
  (h₃ : area = 500)
  (h₄ : area = (1 / 2) * (top_width + bottom_width) * depth) :
  depth = 50 :=
sorry

end depth_of_river_bank_l2201_220155


namespace possible_combinations_l2201_220137

noncomputable def dark_chocolate_price : ℝ := 5
noncomputable def milk_chocolate_price : ℝ := 4.50
noncomputable def white_chocolate_price : ℝ := 6
noncomputable def sales_tax_rate : ℝ := 0.07
noncomputable def leonardo_money : ℝ := 4 + 0.59

noncomputable def total_money := leonardo_money

noncomputable def dark_chocolate_with_tax := dark_chocolate_price * (1 + sales_tax_rate)
noncomputable def milk_chocolate_with_tax := milk_chocolate_price * (1 + sales_tax_rate)
noncomputable def white_chocolate_with_tax := white_chocolate_price * (1 + sales_tax_rate)

theorem possible_combinations :
  total_money = 4.59 ∧ (total_money >= 0 ∧ total_money < dark_chocolate_with_tax ∧ total_money < white_chocolate_with_tax ∧
  total_money ≥ milk_chocolate_with_tax ∧ milk_chocolate_with_tax = 4.82) :=
by
  sorry

end possible_combinations_l2201_220137


namespace vector_subtraction_l2201_220174

-- Define the vectors a and b
def a : ℝ × ℝ := (-2, 1)
def b : ℝ × ℝ := (-3, -4)

-- Statement we want to prove: 2a - b = (-1, 6)
theorem vector_subtraction : 2 • a - b = (-1, 6) := by
  sorry

end vector_subtraction_l2201_220174


namespace at_least_one_not_less_than_two_l2201_220161

open Real

theorem at_least_one_not_less_than_two (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  ∃ x, (x = a + 1/b ∨ x = b + 1/c ∨ x = c + 1/a) ∧ 2 ≤ x :=
by
  sorry

end at_least_one_not_less_than_two_l2201_220161


namespace max_value_of_expression_l2201_220186

theorem max_value_of_expression (x y z : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) (h_sum : x + y + z = 1) :
  x + y^3 + z^4 ≤ 1 :=
sorry

end max_value_of_expression_l2201_220186


namespace total_nuts_correct_l2201_220192

-- Definitions for conditions
def w : ℝ := 0.25
def a : ℝ := 0.25
def p : ℝ := 0.15
def c : ℝ := 0.40

-- The theorem to be proven
theorem total_nuts_correct : w + a + p + c = 1.05 := by
  sorry

end total_nuts_correct_l2201_220192


namespace simplify_fraction_l2201_220162

theorem simplify_fraction : (270 / 18) * (7 / 140) * (9 / 4) = 27 / 16 :=
by sorry

end simplify_fraction_l2201_220162


namespace chicken_nugget_ratio_l2201_220151

theorem chicken_nugget_ratio (k d a t : ℕ) (h1 : a = 20) (h2 : t = 100) (h3 : k + d + a = t) : (k + d) / a = 4 :=
by
  sorry

end chicken_nugget_ratio_l2201_220151


namespace time_for_2km_l2201_220159

def distance_over_time (t : ℕ) : ℝ := 
  sorry -- Function representing the distance walked over time

theorem time_for_2km : ∃ t : ℕ, distance_over_time t = 2 ∧ t = 105 :=
by
  sorry

end time_for_2km_l2201_220159


namespace inequality_holds_for_interval_l2201_220172

theorem inequality_holds_for_interval (a : ℝ) : 
  (∀ x, 1 < x ∧ x < 5 → x^2 - 2 * (a - 2) * x + a < 0) → a ≥ 5 :=
by
  intros h
  sorry

end inequality_holds_for_interval_l2201_220172


namespace tank_saltwater_solution_l2201_220163

theorem tank_saltwater_solution (x : ℝ) :
  let water1 := 0.75 * x
  let water1_evaporated := (1/3) * water1
  let water2 := water1 - water1_evaporated
  let salt2 := 0.25 * x
  let water3 := water2 + 12
  let salt3 := salt2 + 24
  let step2_eq := (salt3 / (water3 + 24)) = 0.4
  let water4 := water3 - (1/4) * water3
  let salt4 := salt3
  let water5 := water4 + 15
  let salt5 := salt4 + 30
  let step4_eq := (salt5 / (water5 + 30)) = 0.5
  step2_eq ∧ step4_eq → x = 192 :=
by
  sorry

end tank_saltwater_solution_l2201_220163


namespace min_value_of_expr_l2201_220194

theorem min_value_of_expr (a b c : ℝ) (h1 : 0 < a ∧ a ≤ b ∧ b ≤ c) (h2 : a * b * c = 1) :
    (1 / (a ^ 3 * (b + c)) + 1 / (b ^ 3 * (a + c)) + 1 / (c ^ 3 * (a + b))) ≥ 3 / 2 := 
by
  sorry

end min_value_of_expr_l2201_220194


namespace probability_correct_l2201_220179

/-
  Problem statement:
  Consider a modified city map where a student walks from intersection A to intersection B, passing through C and D.
  The student always walks east or south and at each intersection, decides the direction to go with a probability of 1/2.
  The map requires 4 eastward and 3 southward moves to reach B from A. C is 2 east, 1 south move from A. D is 3 east, 2 south moves from A.
  Prove that the probability the student goes through both C and D is 12/35.
-/

noncomputable def probability_passing_C_and_D : ℚ :=
  let total_paths_A_to_B := Nat.choose 7 4
  let paths_A_to_C := Nat.choose 3 2
  let paths_C_to_D := Nat.choose 2 1
  let paths_D_to_B := Nat.choose 2 1
  (paths_A_to_C * paths_C_to_D * paths_D_to_B) / total_paths_A_to_B

theorem probability_correct :
  probability_passing_C_and_D = 12 / 35 :=
by
  sorry

end probability_correct_l2201_220179


namespace total_time_for_journey_l2201_220135

theorem total_time_for_journey (x : ℝ) : 
  let time_first_part := x / 50
  let time_second_part := 3 * x / 80
  time_first_part + time_second_part = 23 * x / 400 :=
by 
  sorry

end total_time_for_journey_l2201_220135


namespace x_gt_one_sufficient_but_not_necessary_for_abs_x_gt_one_l2201_220141

theorem x_gt_one_sufficient_but_not_necessary_for_abs_x_gt_one {x : ℝ} :
  (x > 1 → |x| > 1) ∧ (¬(|x| > 1 → x > 1)) :=
by
  sorry

end x_gt_one_sufficient_but_not_necessary_for_abs_x_gt_one_l2201_220141


namespace find_common_difference_l2201_220197

variable {a : ℕ → ℤ}  -- Define a sequence indexed by natural numbers, returning integers
variable (d : ℤ)  -- Define the common difference as an integer

-- The conditions: sequence is arithmetic, a_2 = 14, a_5 = 5
axiom arithmetic_sequence (n : ℕ) : a n = a 0 + n * d
axiom a_2_eq_14 : a 2 = 14
axiom a_5_eq_5 : a 5 = 5

-- The proof statement
theorem find_common_difference : d = -3 :=
by sorry

end find_common_difference_l2201_220197


namespace f_of_pi_over_6_l2201_220133

noncomputable def f (ω : ℝ) (ϕ : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x + ϕ)

theorem f_of_pi_over_6 (ω ϕ : ℝ) (h₀ : ω > 0) (h₁ : -Real.pi / 2 ≤ ϕ) (h₂ : ϕ < Real.pi / 2) 
  (transformed : ∀ x, f ω ϕ (x/2 - Real.pi/6) = Real.sin x) :
  f ω ϕ (Real.pi / 6) = Real.sqrt 2 / 2 :=
by
  sorry

end f_of_pi_over_6_l2201_220133


namespace range_of_a_l2201_220136

noncomputable def f (a x : ℝ) : ℝ := Real.log (x^2 - a * x - 3)

def monotonic_increasing (a : ℝ) : Prop :=
  ∀ x > 1, 2 * x - a > 0

def positive_argument (a : ℝ) : Prop :=
  ∀ x > 1, x^2 - a * x - 3 > 0

theorem range_of_a :
  {a : ℝ | monotonic_increasing a ∧ positive_argument a} = {a : ℝ | a ≤ -2} :=
sorry

end range_of_a_l2201_220136


namespace max_quotient_l2201_220116

theorem max_quotient (x y : ℝ) (h1 : -5 ≤ x) (h2 : x ≤ -3) (h3 : 3 ≤ y) (h4 : y ≤ 6) : 
  ∃ z, z = (x + y) / x ∧ ∀ w, w = (x + y) / x → w ≤ 0 :=
by
  sorry

end max_quotient_l2201_220116


namespace Julio_fish_catch_rate_l2201_220143

theorem Julio_fish_catch_rate (F : ℕ) : 
  (9 * F) - 15 = 48 → F = 7 :=
by
  intro h1
  --- proof
  sorry

end Julio_fish_catch_rate_l2201_220143


namespace average_monthly_balance_l2201_220110

theorem average_monthly_balance
  (jan feb mar apr may : ℕ) 
  (Hjan : jan = 200)
  (Hfeb : feb = 300)
  (Hmar : mar = 100)
  (Hapr : apr = 250)
  (Hmay : may = 150) :
  (jan + feb + mar + apr + may) / 5 = 200 := 
  by
  sorry

end average_monthly_balance_l2201_220110


namespace smallest_x_correct_l2201_220196

noncomputable def smallest_x (K : ℤ) : ℤ := 135000

theorem smallest_x_correct (K : ℤ) :
  (∃ x : ℤ, 180 * x = K ^ 5 ∧ x > 0) → smallest_x K = 135000 :=
by
  sorry

end smallest_x_correct_l2201_220196


namespace rectangle_length_l2201_220115

theorem rectangle_length
  (w l : ℝ)
  (h1 : l = 4 * w)
  (h2 : l * w = 100) :
  l = 20 :=
sorry

end rectangle_length_l2201_220115


namespace train_cross_bridge_time_l2201_220120

theorem train_cross_bridge_time
  (length_train : ℕ) (speed_train_kmph : ℕ) (length_bridge : ℕ) 
  (km_to_m : ℕ) (hour_to_s : ℕ)
  (h1 : length_train = 165) 
  (h2 : speed_train_kmph = 54) 
  (h3 : length_bridge = 720) 
  (h4 : km_to_m = 1000) 
  (h5 : hour_to_s = 3600) 
  : (length_train + length_bridge) / ((speed_train_kmph * km_to_m) / hour_to_s) = 59 := 
sorry

end train_cross_bridge_time_l2201_220120


namespace smallest_nonneg_integer_divisible_by_4_l2201_220128

theorem smallest_nonneg_integer_divisible_by_4 :
  ∃ n : ℕ, (7 * (n - 3)^5 - n^2 + 16 * n - 30) % 4 = 0 ∧ ∀ m : ℕ, m < n -> (7 * (m - 3)^5 - m^2 + 16 * m - 30) % 4 ≠ 0 :=
by
  use 1
  sorry

end smallest_nonneg_integer_divisible_by_4_l2201_220128


namespace find_avg_speed_l2201_220150

variables (v t : ℝ)

noncomputable def avg_speed_cond := 
  (v + Real.sqrt 15) * (t - Real.pi / 4) = v * t

theorem find_avg_speed (h : avg_speed_cond v t) : v = Real.sqrt 15 :=
by
  sorry

end find_avg_speed_l2201_220150


namespace find_fifth_month_sale_l2201_220149

theorem find_fifth_month_sale (s1 s2 s3 s4 s6 A : ℝ) (h1 : s1 = 800) (h2 : s2 = 900) (h3 : s3 = 1000) (h4 : s4 = 700) (h5 : s6 = 900) (h6 : A = 850) :
  ∃ s5 : ℝ, (s1 + s2 + s3 + s4 + s5 + s6) / 6 = A ∧ s5 = 800 :=
by
  sorry

end find_fifth_month_sale_l2201_220149


namespace f_eq_g_iff_l2201_220123

noncomputable def f (m n x : ℝ) := m * x^2 + n * x
noncomputable def g (p q x : ℝ) := p * x + q

theorem f_eq_g_iff (m n p q : ℝ) :
  (∀ x, f m n (g p q x) = g p q (f m n x)) ↔ 2 * m = n := by
  sorry

end f_eq_g_iff_l2201_220123


namespace angle_equiv_330_neg390_l2201_220181

theorem angle_equiv_330_neg390 : ∃ k : ℤ, 330 = -390 + 360 * k :=
by
  sorry

end angle_equiv_330_neg390_l2201_220181


namespace maria_correct_answers_l2201_220182

theorem maria_correct_answers (x : ℕ) (n c d s : ℕ) (h1 : n = 30) (h2 : c = 20) (h3 : d = 5) (h4 : s = 325)
  (h5 : n = x + (n - x)) : 20 * x - 5 * (30 - x) = 325 → x = 19 :=
by 
  intros h_eq
  sorry

end maria_correct_answers_l2201_220182


namespace find_vertex_l2201_220185

noncomputable def parabola_vertex (x y : ℝ) : Prop :=
  2 * y^2 + 8 * y - 3 * x + 6 = 0

theorem find_vertex :
  ∃ (x y : ℝ), parabola_vertex x y ∧ x = -14/3 ∧ y = -2 :=
by
  sorry

end find_vertex_l2201_220185


namespace smallest_whole_number_l2201_220184

theorem smallest_whole_number :
  ∃ x : ℕ, x % 3 = 2 ∧ x % 5 = 3 ∧ x % 7 = 4 ∧ x = 23 :=
sorry

end smallest_whole_number_l2201_220184


namespace henry_has_more_games_l2201_220154

-- Define the conditions and initial states
def initial_games_henry : ℕ := 33
def given_games_neil : ℕ := 5
def initial_games_neil : ℕ := 2

-- Define the number of games Henry and Neil have now
def games_henry_now : ℕ := initial_games_henry - given_games_neil
def games_neil_now : ℕ := initial_games_neil + given_games_neil

-- State the theorem to be proven
theorem henry_has_more_games : games_henry_now / games_neil_now = 4 :=
by
  sorry

end henry_has_more_games_l2201_220154


namespace problem_l2201_220189

variable (x y z w : ℚ)

theorem problem
  (h1 : x / y = 7)
  (h2 : z / y = 5)
  (h3 : z / w = 3 / 4) :
  w / x = 20 / 21 :=
by sorry

end problem_l2201_220189


namespace ellipse_equation_with_m_l2201_220166

theorem ellipse_equation_with_m (m : ℝ) : 
  (∃ x y : ℝ, m * (x^2 + y^2 + 2 * y + 1) = (x - 2 * y + 3)^2) → m ∈ Set.Ioi 5 := 
sorry

end ellipse_equation_with_m_l2201_220166


namespace beef_weight_loss_percentage_l2201_220105

theorem beef_weight_loss_percentage (weight_before weight_after weight_lost_percentage : ℝ) 
  (before_process : weight_before = 861.54)
  (after_process : weight_after = 560) 
  (weight_lost : (weight_before - weight_after) = 301.54)
  : weight_lost_percentage = 34.99 :=
by
  sorry

end beef_weight_loss_percentage_l2201_220105


namespace problem_statement_l2201_220121

variable (a b c : ℝ)

-- Conditions given in the problem
axiom h1 : (a * c) / (a + b) + (b * a) / (b + c) + (c * b) / (c + a) = -24
axiom h2 : (b * c) / (a + b) + (c * a) / (b + c) + (a * b) / (c + a) = 8

-- The Lean statement for the proof problem
theorem problem_statement (a b c : ℝ) (h1 : (a * c) / (a + b) + (b * a) / (b + c) + (c * b) / (c + a) = -24)
    (h2 : (b * c) / (a + b) + (c * a) / (b + c) + (a * b) / (c + a) = 8) :
    (b / (a + b) + c / (b + c) + a / (c + a)) = 19 / 2 :=
sorry

end problem_statement_l2201_220121


namespace contractor_earnings_l2201_220183

def total_days : ℕ := 30
def work_rate : ℝ := 25
def fine_rate : ℝ := 7.5
def absent_days : ℕ := 8
def worked_days : ℕ := total_days - absent_days
def total_earned : ℝ := worked_days * work_rate
def total_fine : ℝ := absent_days * fine_rate
def total_received : ℝ := total_earned - total_fine

theorem contractor_earnings : total_received = 490 :=
by
  sorry

end contractor_earnings_l2201_220183
