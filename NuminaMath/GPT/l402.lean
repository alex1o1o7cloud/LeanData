import Mathlib

namespace find_x_value_l402_40242

open Real

theorem find_x_value (x : ℝ) (h1 : 0 < x) (h2 : x < 180) 
(h3 : tan (150 * π / 180 - x * π / 180) = (sin (150 * π / 180) - sin (x * π / 180)) / (cos (150 * π / 180) - cos (x * π / 180))) :
x = 120 :=
sorry

end find_x_value_l402_40242


namespace rational_sum_zero_l402_40238

theorem rational_sum_zero (x1 x2 x3 x4 : ℚ)
  (h1 : x1 = x2 + x3 + x4)
  (h2 : x2 = x1 + x3 + x4)
  (h3 : x3 = x1 + x2 + x4)
  (h4 : x4 = x1 + x2 + x3) : 
  x1 = 0 ∧ x2 = 0 ∧ x3 = 0 ∧ x4 = 0 := 
sorry

end rational_sum_zero_l402_40238


namespace john_fixes_8_computers_l402_40292

theorem john_fixes_8_computers 
  (total_computers : ℕ)
  (unfixable_percentage : ℝ)
  (waiting_percentage : ℝ) 
  (h1 : total_computers = 20)
  (h2 : unfixable_percentage = 0.2)
  (h3 : waiting_percentage = 0.4) :
  let fixed_right_away := total_computers * (1 - unfixable_percentage - waiting_percentage)
  fixed_right_away = 8 :=
by
  sorry

end john_fixes_8_computers_l402_40292


namespace relationship_between_a_and_b_l402_40229

theorem relationship_between_a_and_b {a b : ℝ} (h1 : a > 0) (h2 : b > 0)
  (h3 : ∀ x : ℝ, |(2 * x + 2)| < a → |(x + 1)| < b) : b ≥ a / 2 :=
by
  -- The proof steps will be inserted here
  sorry

end relationship_between_a_and_b_l402_40229


namespace number_of_students_l402_40299

theorem number_of_students (S N : ℕ) (h1 : S = 15 * N)
                           (h2 : (8 * 14) = 112)
                           (h3 : (6 * 16) = 96)
                           (h4 : 17 = 17)
                           (h5 : S = 225) : N = 15 :=
by sorry

end number_of_students_l402_40299


namespace cards_distribution_l402_40208

theorem cards_distribution (total_cards : ℕ) (total_people : ℕ) (cards_per_person : ℕ) (extra_cards : ℕ) (people_with_extra_cards : ℕ) (people_with_fewer_cards : ℕ) :
  total_cards = 100 →
  total_people = 15 →
  total_cards / total_people = cards_per_person →
  total_cards % total_people = extra_cards →
  people_with_extra_cards = extra_cards →
  people_with_fewer_cards = total_people - people_with_extra_cards →
  people_with_fewer_cards = 5 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end cards_distribution_l402_40208


namespace max_naive_number_l402_40294

-- Define the digits and conditions for a naive number
variable (a b c d : ℕ)
variable (M : ℕ)
variable (h1 : b = c + 2)
variable (h2 : a = d + 6)
variable (h3 : M = 1000 * a + 100 * b + 10 * c + d)

-- Define P(M) and Q(M)
def P (a b c d : ℕ) : ℕ := 3 * (a + b) + c + d
def Q (a : ℕ) : ℕ := a - 5

-- Problem statement: Prove the maximum value of M satisfying the divisibility condition
theorem max_naive_number (div_cond : (P a b c d) % (Q a) = 0) (hq : Q a % 10 = 0) : M = 9313 := 
sorry

end max_naive_number_l402_40294


namespace nat_condition_l402_40200

theorem nat_condition (n : ℕ) (h : n ≥ 2) :
  (∀ i j : ℕ, 0 ≤ i → i ≤ j → j ≤ n → (i + j) % 2 = (Nat.choose n i + Nat.choose n j) % 2) ↔
  (∃ p : ℕ, n = 2^p - 2) :=
sorry

end nat_condition_l402_40200


namespace solve_y_l402_40291

theorem solve_y (y : ℝ) (h : (4 * y - 2) / (5 * y - 5) = 3 / 4) : y = -7 :=
by
  sorry

end solve_y_l402_40291


namespace intersection_M_N_l402_40222

noncomputable def M : Set ℝ := { x | -1 < x ∧ x < 3 }
noncomputable def N : Set ℝ := { x | ∃ y, y = Real.log (x - x^2) }
noncomputable def intersection (A B : Set ℝ) : Set ℝ := { x | x ∈ A ∧ x ∈ B }

theorem intersection_M_N : intersection M N = { x | 0 < x ∧ x < 1 } :=
by
  sorry

end intersection_M_N_l402_40222


namespace rectangle_dimensions_l402_40287

theorem rectangle_dimensions (l w : ℝ) (h1 : 2 * l + 2 * w = 240) (h2 : l * w = 2880) :
  (l = 86.833 ∧ w = 33.167) ∨ (l = 33.167 ∧ w = 86.833) :=
by
  sorry

end rectangle_dimensions_l402_40287


namespace free_endpoints_can_be_1001_l402_40269

variables (initial_segs : ℕ) (total_free_ends : ℕ) (k : ℕ)

-- Initial setup: one initial segment.
def initial_segment : ℕ := 1

-- Each time 5 segments are drawn from a point, the number of free ends increases by 4.
def free_ends_after_k_actions (k : ℕ) : ℕ := initial_segment + 4 * k

-- Question: Can the number of free endpoints be exactly 1001?
theorem free_endpoints_can_be_1001 : free_ends_after_k_actions 250 = 1001 := by
  sorry

end free_endpoints_can_be_1001_l402_40269


namespace parallel_and_equidistant_line_l402_40258

-- Define the two given lines
def line1 (x y : ℝ) : Prop := 3 * x + 2 * y - 6 = 0
def line2 (x y : ℝ) : Prop := 6 * x + 4 * y - 3 = 0

-- Define the desired property: a line parallel to line1 and line2, and equidistant from both
theorem parallel_and_equidistant_line :
  ∃ b : ℝ, ∀ x y : ℝ, (3 * x + 2 * y + b = 0) ∧
  (|-6 - b| / Real.sqrt (9 + 4) = |-3/2 - b| / Real.sqrt (9 + 4)) →
  (12 * x + 8 * y - 15 = 0) :=
by
  sorry

end parallel_and_equidistant_line_l402_40258


namespace solid_triangle_front_view_l402_40277

def is_triangle_front_view (solid : ℕ) : Prop :=
  solid = 1 ∨ solid = 2 ∨ solid = 3 ∨ solid = 5

theorem solid_triangle_front_view (s : ℕ) (h : s = 1 ∨ s = 2 ∨ s = 3 ∨ s = 4 ∨ s = 5 ∨ s = 6):
  is_triangle_front_view s ↔ (s = 1 ∨ s = 2 ∨ s = 3 ∨ s = 5) :=
by
  sorry

end solid_triangle_front_view_l402_40277


namespace rectangle_dimensions_l402_40235

-- Define the known shapes and their dimensions
def square (s : ℝ) : ℝ := s^2
def rectangle1 : ℝ := 10 * 24
def rectangle2 (a b : ℝ) : ℝ := a * b

-- The total area must match the area of a square of side length 24 cm
def total_area (s a b : ℝ) : ℝ := (2 * square s) + rectangle1 + rectangle2 a b

-- The problem statement
theorem rectangle_dimensions
  (s a b : ℝ)
  (h0 : a ∈ [2, 19, 34, 34, 14, 14, 24])
  (h1 : b ∈ [24, 17.68, 10, 44, 24, 17, 38])
  : (total_area s a b = 24^2) :=
by
  sorry

end rectangle_dimensions_l402_40235


namespace simplify_fraction_product_l402_40231

theorem simplify_fraction_product :
  (2 / 3) * (4 / 7) * (9 / 13) = 24 / 91 := by
  sorry

end simplify_fraction_product_l402_40231


namespace cos_8_minus_sin_8_l402_40252

theorem cos_8_minus_sin_8 (α m : ℝ) (h : Real.cos (2 * α) = m) :
  Real.cos α ^ 8 - Real.sin α ^ 8 = m * (1 + m^2) / 2 :=
by
  sorry

end cos_8_minus_sin_8_l402_40252


namespace plastering_cost_correct_l402_40244

def length : ℕ := 40
def width : ℕ := 18
def depth : ℕ := 10
def cost_per_sq_meter : ℚ := 1.25

def area_bottom (L W : ℕ) : ℕ := L * W
def perimeter_bottom (L W : ℕ) : ℕ := 2 * (L + W)
def area_walls (P D : ℕ) : ℕ := P * D
def total_area (A_bottom A_walls : ℕ) : ℕ := A_bottom + A_walls
def total_cost (A_total : ℕ) (cost_per_sq_meter : ℚ) : ℚ := A_total * cost_per_sq_meter

theorem plastering_cost_correct :
  total_cost (total_area (area_bottom length width)
                        (area_walls (perimeter_bottom length width) depth))
             cost_per_sq_meter = 2350 :=
by 
  sorry

end plastering_cost_correct_l402_40244


namespace new_person_weight_l402_40270

theorem new_person_weight (w : ℝ) (avg_increase : ℝ) (replaced_person_weight : ℝ) (num_people : ℕ) 
(H1 : avg_increase = 4.8) (H2 : replaced_person_weight = 62) (H3 : num_people = 12) : 
w = 119.6 :=
by
  -- We could provide the intermediate steps as definitions here but for the theorem statement, we just present the goal.
  sorry

end new_person_weight_l402_40270


namespace sum_of_squares_diagonals_cyclic_quadrilateral_l402_40216

theorem sum_of_squares_diagonals_cyclic_quadrilateral 
(a b c d : ℝ) (α : ℝ) 
(hc : c^2 = a^2 + b^2 + 2 * a * b * Real.cos α)
(hd : d^2 = a^2 + b^2 - 2 * a * b * Real.cos α) :
  c^2 + d^2 = 2 * a^2 + 2 * b^2 :=
by
  sorry

end sum_of_squares_diagonals_cyclic_quadrilateral_l402_40216


namespace sum_of_distinct_integers_l402_40290

theorem sum_of_distinct_integers (a b c d e : ℤ) (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e) 
(h_prod : (7 - a) * (7 - b) * (7 - c) * (7 - d) * (7 - e) = 120) : 
a + b + c + d + e = 33 := 
sorry

end sum_of_distinct_integers_l402_40290


namespace intersection_of_parabola_with_y_axis_l402_40274

theorem intersection_of_parabola_with_y_axis :
  ∃ y : ℝ, y = - (0 + 2)^2 + 6 ∧ (0, y) = (0, 2) :=
by
  sorry

end intersection_of_parabola_with_y_axis_l402_40274


namespace problem_solution_l402_40204

open Real

/-- If (y / 6) / 3 = 6 / (y / 3), then y is ±18. -/
theorem problem_solution (y : ℝ) (h : (y / 6) / 3 = 6 / (y / 3)) : y = 18 ∨ y = -18 :=
by
  sorry

end problem_solution_l402_40204


namespace speed_equivalence_l402_40230

def convert_speed (speed_kmph : ℚ) : ℚ :=
  speed_kmph * 0.277778

theorem speed_equivalence : convert_speed 162 = 45 :=
by
  sorry

end speed_equivalence_l402_40230


namespace solid2_solid4_views_identical_l402_40247

-- Define the solids and their orthographic views
structure Solid :=
  (top_view : String)
  (front_view : String)
  (side_view : String)

-- Given solids as provided by the problem
def solid1 : Solid := { top_view := "...", front_view := "...", side_view := "..." }
def solid2 : Solid := { top_view := "...", front_view := "...", side_view := "..." }
def solid3 : Solid := { top_view := "...", front_view := "...", side_view := "..." }
def solid4 : Solid := { top_view := "...", front_view := "...", side_view := "..." }

-- Function to compare two solids' views
def views_identical (s1 s2 : Solid) : Prop :=
  (s1.top_view = s2.top_view ∧ s1.front_view = s2.front_view) ∨
  (s1.top_view = s2.top_view ∧ s1.side_view = s2.side_view) ∨
  (s1.front_view = s2.front_view ∧ s1.side_view = s2.side_view)

-- Theorem statement
theorem solid2_solid4_views_identical : views_identical solid2 solid4 := 
sorry

end solid2_solid4_views_identical_l402_40247


namespace arithmetic_seq_sum_a4_a6_l402_40245

noncomputable def arithmetic_seq (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_seq_sum_a4_a6 (a : ℕ → ℝ)
  (h_arith : arithmetic_seq a)
  (h_root1 : a 3 ^ 2 - 3 * a 3 + 1 = 0)
  (h_root2 : a 7 ^ 2 - 3 * a 7 + 1 = 0) :
  a 4 + a 6 = 3 :=
sorry

end arithmetic_seq_sum_a4_a6_l402_40245


namespace multiplication_equivalence_l402_40205

theorem multiplication_equivalence :
    44 * 22 = 88 * 11 :=
by
  sorry

end multiplication_equivalence_l402_40205


namespace find_number_l402_40213

theorem find_number (x : ℤ) (h : x * 9999 = 806006795) : x = 80601 :=
sorry

end find_number_l402_40213


namespace equation_solutions_exist_l402_40286

theorem equation_solutions_exist (d x y : ℤ) (hx : Odd x) (hy : Odd y)
  (hxy : x^2 - d * y^2 = -4) : ∃ X Y : ℕ, X^2 - d * Y^2 = -1 :=
by
  sorry  -- Proof is omitted as per the instructions

end equation_solutions_exist_l402_40286


namespace f_at_10_l402_40233

variable (f : ℕ → ℝ)

-- Conditions
axiom f_1 : f 1 = 2
axiom f_relation : ∀ m n : ℕ, m ≥ n → f (m + n) + f (m - n) = (f (2 * m) + f (2 * n)) / 2 + 2 * n

-- Prove f(10) = 361
theorem f_at_10 : f 10 = 361 :=
by
  sorry

end f_at_10_l402_40233


namespace problem_statement_l402_40255

theorem problem_statement (x : ℝ) : 
  (∀ (x : ℝ), (100 / x = 80 / (x - 4)) → true) :=
by
  intro x
  sorry

end problem_statement_l402_40255


namespace inequality_solution_l402_40246

theorem inequality_solution (x : ℝ) : 
  (x - 3) / (x + 7) < 0 ↔ -7 < x ∧ x < 3 :=
by
  sorry

end inequality_solution_l402_40246


namespace num_aluminum_cans_l402_40295

def num_glass_bottles : ℕ := 10
def total_litter : ℕ := 18

theorem num_aluminum_cans : total_litter - num_glass_bottles = 8 :=
by
  sorry

end num_aluminum_cans_l402_40295


namespace last_two_digits_of_7_pow_2015_l402_40256

theorem last_two_digits_of_7_pow_2015 : ((7 ^ 2015) % 100) = 43 := 
by
  sorry

end last_two_digits_of_7_pow_2015_l402_40256


namespace triangle_C_squared_eq_b_a_plus_b_l402_40212

variables {A B C a b : ℝ}

theorem triangle_C_squared_eq_b_a_plus_b
  (h1 : C = 2 * B)
  (h2 : A ≠ B) :
  C^2 = b * (a + b) :=
sorry

end triangle_C_squared_eq_b_a_plus_b_l402_40212


namespace a_2005_l402_40271

noncomputable def a : ℕ → ℤ := sorry 

axiom a3 : a 3 = 5
axiom a5 : a 5 = 8
axiom exists_n : ∃ (n : ℕ), n > 0 ∧ a n + a (n + 1) + a (n + 2) = 7

theorem a_2005 : a 2005 = -6 := by {
  sorry
}

end a_2005_l402_40271


namespace sum_of_decimals_is_one_l402_40228

-- Define digits for each decimal place
def digit_a : ℕ := 2
def digit_b : ℕ := 3
def digit_c : ℕ := 2
def digit_d : ℕ := 2

-- Define the decimal numbers with these digits
def decimal1 : Rat := (digit_b * 10 + digit_a) / 100
def decimal2 : Rat := (digit_d * 10 + digit_c) / 100
def decimal3 : Rat := (2 * 10 + 2) / 100
def decimal4 : Rat := (2 * 10 + 3) / 100

-- The main theorem that states the sum of these decimals is 1
theorem sum_of_decimals_is_one : decimal1 + decimal2 + decimal3 + decimal4 = 1 := by
  sorry

end sum_of_decimals_is_one_l402_40228


namespace correct_choice_is_C_l402_40293

def is_opposite_number (a b : ℤ) : Prop := a + b = 0

def option_A : Prop := ¬is_opposite_number (2^3) (3^2)
def option_B : Prop := ¬is_opposite_number (-2) (-|-2|)
def option_C : Prop := is_opposite_number ((-3)^2) (-3^2)
def option_D : Prop := ¬is_opposite_number 2 (-(-2))

theorem correct_choice_is_C : option_C ∧ option_A ∧ option_B ∧ option_D :=
by
  sorry

end correct_choice_is_C_l402_40293


namespace jims_speed_l402_40234

variable (x : ℝ)

theorem jims_speed (bob_speed : ℝ) (bob_head_start : ℝ) (time : ℝ) (bob_distance : ℝ) :
  bob_speed = 6 →
  bob_head_start = 1 →
  time = 1 / 3 →
  bob_distance = bob_speed * time →
  (x * time = bob_distance + bob_head_start) →
  x = 9 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end jims_speed_l402_40234


namespace abc_positive_l402_40261

theorem abc_positive (a b c : ℝ) (h1 : a + b + c > 0) (h2 : ab + bc + ca > 0) (h3 : abc > 0) : a > 0 ∧ b > 0 ∧ c > 0 :=
by
  sorry

end abc_positive_l402_40261


namespace ratio_of_Jordyn_age_to_Zrinka_age_is_2_l402_40223

variable (Mehki_age : ℕ) (Jordyn_age : ℕ) (Zrinka_age : ℕ)

-- Conditions
def Mehki_is_10_years_older_than_Jordyn := Mehki_age = Jordyn_age + 10
def Zrinka_age_is_6 := Zrinka_age = 6
def Mehki_age_is_22 := Mehki_age = 22

-- Theorem statement: the ratio of Jordyn's age to Zrinka's age is 2.
theorem ratio_of_Jordyn_age_to_Zrinka_age_is_2
  (h1 : Mehki_is_10_years_older_than_Jordyn Mehki_age Jordyn_age)
  (h2 : Zrinka_age_is_6 Zrinka_age)
  (h3 : Mehki_age_is_22 Mehki_age) : Jordyn_age / Zrinka_age = 2 :=
by
  -- The proof would go here
  sorry

end ratio_of_Jordyn_age_to_Zrinka_age_is_2_l402_40223


namespace difference_of_squares_l402_40289

theorem difference_of_squares 
  (x y : ℝ) 
  (optionA := (-x + y) * (x + y))
  (optionB := (-x + y) * (x - y))
  (optionC := (x + 2) * (2 + x))
  (optionD := (2 * x + 3) * (3 * x - 2)) :
  optionA = -(x + y)^2 ∨ optionA = (x + y) * (y - x) :=
sorry

end difference_of_squares_l402_40289


namespace irrational_roots_of_odd_coeffs_l402_40282

theorem irrational_roots_of_odd_coeffs (a b c : ℤ) (ha : a % 2 = 1) (hb : b % 2 = 1) (hc : c % 2 = 1) : 
  ¬ ∃ (x : ℚ), a * x^2 + b * x + c = 0 := 
sorry

end irrational_roots_of_odd_coeffs_l402_40282


namespace parallel_vectors_tan_l402_40237

theorem parallel_vectors_tan (θ : ℝ)
  (a : ℝ × ℝ) (b : ℝ × ℝ)
  (h₀ : a = (2, Real.sin θ))
  (h₁ : b = (1, Real.cos θ))
  (h_parallel : a.1 * b.2 = a.2 * b.1) :
  Real.tan θ = 2 := 
sorry

end parallel_vectors_tan_l402_40237


namespace mr_lee_harvested_apples_l402_40248

theorem mr_lee_harvested_apples :
  let number_of_baskets := 19
  let apples_per_basket := 25
  (number_of_baskets * apples_per_basket = 475) :=
by
  sorry

end mr_lee_harvested_apples_l402_40248


namespace carousel_problem_l402_40207

theorem carousel_problem (n : ℕ) : 
  (∃ (f : Fin n → Fin n), 
    (∀ i, f (f i) = i) ∧ 
    (∀ i j, i ≠ j → f i ≠ f j) ∧ 
    (∀ i, f i < n)) ↔ 
  (Even n) := 
sorry

end carousel_problem_l402_40207


namespace least_number_to_add_l402_40227

theorem least_number_to_add (m n : ℕ) (h₁ : m = 1052) (h₂ : n = 23) : 
  ∃ k : ℕ, (m + k) % n = 0 ∧ k = 6 :=
by
  sorry

end least_number_to_add_l402_40227


namespace three_digit_number_divisible_by_eleven_l402_40278

theorem three_digit_number_divisible_by_eleven
  (x : ℕ) (n : ℕ)
  (units_digit_is_two : n % 10 = 2)
  (hundreds_digit_is_seven : n / 100 = 7)
  (tens_digit : n = 700 + x * 10 + 2)
  (divisibility_condition : (7 - x + 2) % 11 = 0) :
  n = 792 := by
  sorry

end three_digit_number_divisible_by_eleven_l402_40278


namespace one_eighth_percent_of_160_plus_half_l402_40239

theorem one_eighth_percent_of_160_plus_half :
  ((1 / 8) / 100 * 160) + 0.5 = 0.7 :=
  sorry

end one_eighth_percent_of_160_plus_half_l402_40239


namespace parallel_condition_coincide_condition_perpendicular_condition_l402_40263

-- Define the equations of the lines
def l1 (m : ℝ) (x y : ℝ) : Prop := (m + 3) * x + 4 * y = 5 - 3 * m
def l2 (m : ℝ) (x y : ℝ) : Prop := 2 * x + (m + 5) * y = 8

-- Parallel lines condition
theorem parallel_condition (m : ℝ) : (l1 m = l2 m ↔ m = -7) →
  (∀ x y : ℝ, l1 m x y ∧ l2 m x y) → False := sorry

-- Coincidence condition
theorem coincide_condition (m : ℝ) : 
  (l1 (-1) = l2 (-1)) :=
sorry

-- Perpendicular lines condition
theorem perpendicular_condition (m : ℝ) : 
  (m = - 13 / 3 ↔ (2 * (m + 3) + 4 * (m + 5) = 0)) :=
sorry

end parallel_condition_coincide_condition_perpendicular_condition_l402_40263


namespace total_books_in_school_l402_40241

theorem total_books_in_school (tables_A tables_B tables_C : ℕ)
  (books_per_table_A books_per_table_B books_per_table_C : ℕ → ℕ)
  (hA : tables_A = 750)
  (hB : tables_B = 500)
  (hC : tables_C = 850)
  (h_books_per_table_A : ∀ n, books_per_table_A n = 3 * n / 5)
  (h_books_per_table_B : ∀ n, books_per_table_B n = 2 * n / 5)
  (h_books_per_table_C : ∀ n, books_per_table_C n = n / 3) :
  books_per_table_A tables_A + books_per_table_B tables_B + books_per_table_C tables_C = 933 :=
by sorry

end total_books_in_school_l402_40241


namespace total_cups_of_mushroom_soup_l402_40280

def cups_team_1 : ℕ := 90
def cups_team_2 : ℕ := 120
def cups_team_3 : ℕ := 70

theorem total_cups_of_mushroom_soup :
  cups_team_1 + cups_team_2 + cups_team_3 = 280 :=
  by sorry

end total_cups_of_mushroom_soup_l402_40280


namespace lego_tower_levels_l402_40296

theorem lego_tower_levels (initial_pieces : ℕ) (pieces_per_level : ℕ) (pieces_left : ℕ) 
    (h1 : initial_pieces = 100) (h2 : pieces_per_level = 7) (h3 : pieces_left = 23) :
    (initial_pieces - pieces_left) / pieces_per_level = 11 := 
by
  sorry

end lego_tower_levels_l402_40296


namespace consecutive_odd_integers_expressions_l402_40273

theorem consecutive_odd_integers_expressions
  {p q : ℤ} (hpq : p + 2 = q ∨ p - 2 = q) (hp_odd : p % 2 = 1) (hq_odd : q % 2 = 1) :
  (2 * p + 5 * q) % 2 = 1 ∧ (5 * p - 2 * q) % 2 = 1 ∧ (2 * p * q + 5) % 2 = 1 :=
  sorry

end consecutive_odd_integers_expressions_l402_40273


namespace two_n_minus_one_lt_n_plus_one_squared_l402_40236

theorem two_n_minus_one_lt_n_plus_one_squared (n : ℕ) (h : n > 0) : 2 * n - 1 < (n + 1) ^ 2 := 
by
  sorry

end two_n_minus_one_lt_n_plus_one_squared_l402_40236


namespace simplify_division_l402_40298

theorem simplify_division (a b c d : ℕ) (h1 : a = 27) (h2 : b = 10^12) (h3 : c = 9) (h4 : d = 10^4) :
  ((a * b) / (c * d) = 300000000) :=
by {
  sorry
}

end simplify_division_l402_40298


namespace translate_parabola_l402_40260

-- Translating the parabola y = (x-2)^2 - 8 three units left and five units up
theorem translate_parabola (x y : ℝ) :
  y = (x - 2) ^ 2 - 8 →
  y = ((x + 3) - 2) ^ 2 - 8 + 5 →
  y = (x + 1) ^ 2 - 3 := by
sorry

end translate_parabola_l402_40260


namespace average_minutes_heard_l402_40214

theorem average_minutes_heard :
  let total_audience := 200
  let duration := 90
  let percent_entire := 0.15
  let percent_slept := 0.15
  let percent_half := 0.25
  let percent_one_fourth := 0.75
  let total_entire := total_audience * percent_entire
  let total_slept := total_audience * percent_slept
  let remaining := total_audience - total_entire - total_slept
  let total_half := remaining * percent_half
  let total_one_fourth := remaining * percent_one_fourth
  let minutes_entire := total_entire * duration
  let minutes_half := total_half * (duration / 2)
  let minutes_one_fourth := total_one_fourth * (duration / 4)
  let total_minutes_heard := minutes_entire + 0 + minutes_half + minutes_one_fourth
  let average_minutes := total_minutes_heard / total_audience
  average_minutes = 33 :=
by
  sorry

end average_minutes_heard_l402_40214


namespace efficiency_ratio_l402_40288

variable {A B : ℝ}

theorem efficiency_ratio (hA : A = 1 / 30) (hAB : A + B = 1 / 20) : A / B = 2 :=
by
  sorry

end efficiency_ratio_l402_40288


namespace students_play_football_l402_40257

theorem students_play_football 
  (total : ℕ) (C : ℕ) (B : ℕ) (Neither : ℕ) (F : ℕ) 
  (h_total : total = 420) 
  (h_C : C = 175) 
  (h_B : B = 130) 
  (h_Neither : Neither = 50) 
  (h_inclusion_exclusion : F + C - B = total - Neither) :
  F = 325 := 
sorry

end students_play_football_l402_40257


namespace max_value_x_y2_z3_l402_40217

theorem max_value_x_y2_z3 (x y z : ℝ) (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0) (h4 : x + y + z = 1) : 
  x + y^2 + z^3 ≤ 1 :=
by
  sorry

end max_value_x_y2_z3_l402_40217


namespace number_of_connections_l402_40251

-- Definitions based on conditions
def switches : ℕ := 15
def connections_per_switch : ℕ := 4

-- Theorem statement proving the correct number of connections
theorem number_of_connections : switches * connections_per_switch / 2 = 30 := by
  sorry

end number_of_connections_l402_40251


namespace exactly_one_is_multiple_of_5_l402_40219

theorem exactly_one_is_multiple_of_5 (a b : ℤ) (h: 24 * a^2 + 1 = b^2) : 
  (∃ k : ℤ, a = 5 * k) ∧ (∀ l : ℤ, b ≠ 5 * l) ∨ (∃ m : ℤ, b = 5 * m) ∧ (∀ n : ℤ, a ≠ 5 * n) :=
sorry

end exactly_one_is_multiple_of_5_l402_40219


namespace circle_equation_l402_40201

theorem circle_equation (x y : ℝ) :
    (x - 1) ^ 2 + (y - 1) ^ 2 = 1 ↔ (∃ (C : ℝ × ℝ), C = (1, 1) ∧ ∃ (r : ℝ), r = 1 ∧ (x - C.1) ^ 2 + (y - C.2) ^ 2 = r ^ 2) :=
by
  sorry

end circle_equation_l402_40201


namespace petya_second_race_finishes_first_l402_40210

variable (t v_P v_V : ℝ)
variable (h1 : v_P * t = 100)
variable (h2 : v_V * t = 90)
variable (d : ℝ)

theorem petya_second_race_finishes_first :
  v_V = 0.9 * v_P ∧
  d * v_P = 10 + d * (0.9 * v_P) →
  ∃ t2 : ℝ, t2 = 100 / v_P ∧ (v_V * t2 = 90) →
  ∃ t3 : ℝ, t3 = t2 + d / 10 ∧ (d * v_P = 100) →
  v_P * d / 10 - v_V * d / 10 = 1 :=
by
  sorry

end petya_second_race_finishes_first_l402_40210


namespace number_of_smaller_cubes_in_larger_cube_l402_40202

-- Defining the conditions
def volume_large_cube : ℝ := 125
def volume_small_cube : ℝ := 1
def surface_area_difference : ℝ := 600

-- Translating the question into a math proof problem
theorem number_of_smaller_cubes_in_larger_cube : 
  ∃ n : ℕ, n * 6 - 6 * (volume_large_cube^(1/3) ^ 2) = surface_area_difference :=
by
  sorry

end number_of_smaller_cubes_in_larger_cube_l402_40202


namespace simplify_expression_l402_40206

theorem simplify_expression :
  (1 / (1 / (1 / 3 : ℝ)^1 + 1 / (1 / 3)^2 + 1 / (1 / 3)^3)) = (1 / 39 : ℝ) :=
by
  sorry

end simplify_expression_l402_40206


namespace baker_sold_cakes_l402_40243

theorem baker_sold_cakes (S : ℕ) (h1 : 154 = S + 63) : S = 91 :=
by
  sorry

end baker_sold_cakes_l402_40243


namespace divide_80_into_two_parts_l402_40276

theorem divide_80_into_two_parts :
  ∃ a b : ℕ, a + b = 80 ∧ b / 2 = a + 10 ∧ a = 20 ∧ b = 60 :=
by
  sorry

end divide_80_into_two_parts_l402_40276


namespace other_group_less_garbage_l402_40262

theorem other_group_less_garbage :
  387 + (735 - 387) = 735 :=
by
  sorry

end other_group_less_garbage_l402_40262


namespace number_of_children_correct_l402_40240

def total_spectators : ℕ := 25000
def men_spectators : ℕ := 15320
def ratio_children_women : ℕ × ℕ := (7, 3)
def remaining_spectators : ℕ := total_spectators - men_spectators
def total_ratio_parts : ℕ := ratio_children_women.1 + ratio_children_women.2
def spectators_per_part : ℕ := remaining_spectators / total_ratio_parts

def children_spectators : ℕ := spectators_per_part * ratio_children_women.1

theorem number_of_children_correct : children_spectators = 6776 := by
  sorry

end number_of_children_correct_l402_40240


namespace negation_of_exists_gt_1_l402_40220

theorem negation_of_exists_gt_1 :
  (∀ x : ℝ, x ≤ 1) ↔ ¬ (∃ x : ℝ, x > 1) :=
sorry

end negation_of_exists_gt_1_l402_40220


namespace increasing_function_a_l402_40272

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
  if x ≥ 0 then
    x^2
  else
    x^3 - (a-1)*x + a^2 - 3*a - 4

theorem increasing_function_a (a : ℝ) :
  (∀ x y : ℝ, x < y → f x a ≤ f y a) ↔ -1 ≤ a ∧ a ≤ 1 :=
by
  sorry

end increasing_function_a_l402_40272


namespace weight_of_b_l402_40254

variable {a b c : ℝ}

theorem weight_of_b (h1 : (a + b + c) / 3 = 45)
                    (h2 : (a + b) / 2 = 40)
                    (h3 : (b + c) / 2 = 43) :
                    b = 31 := by
  sorry

end weight_of_b_l402_40254


namespace cube_edge_length_l402_40267

def radius := 2
def edge_length (r : ℕ) := 4 + 2 * r

theorem cube_edge_length :
  ∀ r : ℕ, r = radius → edge_length r = 8 :=
by
  intros r h
  rw [h, edge_length]
  rfl

end cube_edge_length_l402_40267


namespace problem_part1_problem_part2_problem_part3_l402_40268

noncomputable def S (n : ℕ) : ℕ := 2 * n^2 + n

noncomputable def a_n (n : ℕ) : ℕ :=
  if n = 1 then S n else S n - S (n - 1)

noncomputable def b_n (n : ℕ) : ℕ := 2^(n - 1)

noncomputable def T_n (n : ℕ) : ℕ :=
  (4 * n - 5) * 2^n + 5

theorem problem_part1 (n : ℕ) (h : n > 0) : n > 0 → a_n n = 4 * n - 1 := by
  sorry

theorem problem_part2 (n : ℕ) (h : n > 0) : n > 0 → b_n n = 2^(n - 1) := by
  sorry

theorem problem_part3 (n : ℕ) (h : n > 0) : n > 0 → T_n n = (4 * n - 5) * 2^n + 5 := by
  sorry

end problem_part1_problem_part2_problem_part3_l402_40268


namespace find_radius_l402_40215

theorem find_radius (a : ℝ) :
  (∃ (x y : ℝ), (x + 2) ^ 2 + (y - 2) ^ 2 = a ∧ x + y + 2 = 0) ∧
  (∃ (l : ℝ), l = 6 ∧ 2 * Real.sqrt (a - 2) = l) →
  a = 11 :=
by
  sorry

end find_radius_l402_40215


namespace johns_elevation_after_travel_l402_40211

-- Definitions based on conditions:
def initial_elevation : ℝ := 400
def downward_rate : ℝ := 10
def time_travelled : ℕ := 5

-- Proof statement:
theorem johns_elevation_after_travel:
  initial_elevation - (downward_rate * time_travelled) = 350 :=
by
  sorry

end johns_elevation_after_travel_l402_40211


namespace cell_survival_after_6_hours_l402_40203

def cell_sequence (a : ℕ → ℕ) : Prop :=
  (a 0 = 2) ∧ (∀ n, a (n + 1) = 2 * a n - 1)

theorem cell_survival_after_6_hours :
  ∃ a : ℕ → ℕ, cell_sequence a ∧ a 6 = 65 :=
by
  sorry

end cell_survival_after_6_hours_l402_40203


namespace number_of_animal_books_l402_40285

variable (A : ℕ)

theorem number_of_animal_books (h1 : 6 * 6 + 3 * 6 + A * 6 = 102) : A = 8 :=
sorry

end number_of_animal_books_l402_40285


namespace equality_of_costs_l402_40279

variable (x : ℝ)
def C1 : ℝ := 50 + 0.35 * (x - 500)
def C2 : ℝ := 75 + 0.45 * (x - 1000)

theorem equality_of_costs : C1 x = C2 x → x = 2500 :=
by
  intro h
  sorry

end equality_of_costs_l402_40279


namespace coconut_trees_per_sqm_l402_40253

def farm_area : ℕ := 20
def harvests : ℕ := 2
def total_earnings : ℝ := 240
def coconut_price : ℝ := 0.50
def coconuts_per_tree : ℕ := 6

theorem coconut_trees_per_sqm : 
  let total_coconuts := total_earnings / coconut_price / harvests
  let total_trees := total_coconuts / coconuts_per_tree 
  let trees_per_sqm := total_trees / farm_area 
  trees_per_sqm = 2 :=
by
  sorry

end coconut_trees_per_sqm_l402_40253


namespace find_ellipse_equation_l402_40297

noncomputable def ellipse_equation (a b : ℝ) : Prop :=
  ∃ c : ℝ, a > b ∧ b > 0 ∧ 4 * a = 16 ∧ |c| = 2 ∧ a^2 = b^2 + c^2

theorem find_ellipse_equation :
  (∃ (a b : ℝ), ellipse_equation a b) → (∃ b : ℝ, (a = 4) ∧ (b > 0) ∧ (b^2 = 12) ∧ (∀ x y : ℝ, (x^2 / 16) + (y^2 / 12) = 1)) :=
by {
  sorry
}

end find_ellipse_equation_l402_40297


namespace trapezium_other_parallel_side_l402_40264

theorem trapezium_other_parallel_side (a : ℝ) (b d : ℝ) (area : ℝ) 
  (h1 : a = 18) (h2 : d = 15) (h3 : area = 285) : b = 20 :=
by
  sorry

end trapezium_other_parallel_side_l402_40264


namespace determine_h_l402_40224

def h (x : ℝ) := -12 * x^4 - 4 * x^3 - 8 * x^2 + 3 * x - 1

theorem determine_h (x : ℝ) : 
  (12 * x^4 + 9 * x^3 - 3 * x + 1 + h x = 5 * x^3 - 8 * x^2 + 3) →
  h x = -12 * x^4 - 4 * x^3 - 8 * x^2 + 3 * x - 1 :=
by
  sorry

end determine_h_l402_40224


namespace sine_of_smaller_angle_and_k_domain_l402_40209

theorem sine_of_smaller_angle_and_k_domain (α : ℝ) (k : ℝ) (AD : ℝ) (h0 : 1 < k) 
  (h1 : CD = AD * Real.tan (2 * α)) (h2 : BD = AD * Real.tan α) 
  (h3 : k = CD / BD) :
  k > 2 ∧ Real.sin (Real.pi / 2 - 2 * α) = 1 / (k - 1) := by
  sorry

end sine_of_smaller_angle_and_k_domain_l402_40209


namespace diff_in_set_l402_40283

variable (A : Set Int)
variable (ha : ∃ a ∈ A, a > 0)
variable (hb : ∃ b ∈ A, b < 0)
variable (h : ∀ {a b : Int}, a ∈ A → b ∈ A → (2 * a) ∈ A ∧ (a + b) ∈ A)

theorem diff_in_set (x y : Int) (hx : x ∈ A) (hy : y ∈ A) : (x - y) ∈ A :=
  sorry

end diff_in_set_l402_40283


namespace find_divisor_l402_40226

-- Definitions of the conditions
def dividend : ℕ := 15968
def quotient : ℕ := 89
def remainder : ℕ := 37

-- The theorem stating the proof problem
theorem find_divisor (D : ℕ) (h : dividend = D * quotient + remainder) : D = 179 :=
sorry

end find_divisor_l402_40226


namespace complement_of_16deg51min_is_73deg09min_l402_40221

def complement_angle (A : ℝ) : ℝ := 90 - A

theorem complement_of_16deg51min_is_73deg09min :
  complement_angle 16.85 = 73.15 := by
  sorry

end complement_of_16deg51min_is_73deg09min_l402_40221


namespace product_xyz_l402_40250

theorem product_xyz (x y z : ℝ) (h1 : x + 1/y = 2) (h2 : y + 1/z = 2) : x * y * z = -1 :=
by
  sorry

end product_xyz_l402_40250


namespace tetrahedron_edge_square_sum_l402_40232

variable (A B C D : Point)
variable (AB AC AD BC BD CD : ℝ) -- Lengths of the edges
variable (m₁ m₂ m₃ : ℝ) -- Distances between the midpoints of the opposite edges

theorem tetrahedron_edge_square_sum:
  (AB ^ 2 + AC ^ 2 + AD ^ 2 + BC ^ 2 + BD ^ 2 + CD ^ 2) =
  4 * (m₁ ^ 2 + m₂ ^ 2 + m₃ ^ 2) :=
  sorry

end tetrahedron_edge_square_sum_l402_40232


namespace relationship_between_f_x1_and_f_x2_l402_40249

variable (f : ℝ → ℝ)
variable (x1 x2 : ℝ)

-- Conditions:
variable (h_even : ∀ x, f x = f (-x))          -- f is even
variable (h_increasing : ∀ a b, 0 < a → a < b → f a < f b)  -- f is increasing on (0, +∞)
variable (h_x1_neg : x1 < 0)                   -- x1 < 0
variable (h_x2_pos : 0 < x2)                   -- x2 > 0
variable (h_abs : |x1| > |x2|)                 -- |x1| > |x2|

-- Goal:
theorem relationship_between_f_x1_and_f_x2 : f x1 > f x2 :=
by
  sorry

end relationship_between_f_x1_and_f_x2_l402_40249


namespace cubic_root_sum_eq_constant_term_divided_l402_40284

theorem cubic_root_sum_eq_constant_term_divided 
  (a b c : ℝ) 
  (h_roots : (24 * a^3 - 36 * a^2 + 14 * a - 1 = 0) 
           ∧ (24 * b^3 - 36 * b^2 + 14 * b - 1 = 0) 
           ∧ (24 * c^3 - 36 * c^2 + 14 * c - 1 = 0))
  (h_bounds : 0 < a ∧ a < 1 ∧ 0 < b ∧ b < 1 ∧ 0 < c ∧ c < 1) 
  : (1 / (1 - a) + 1 / (1 - b) + 1 / (1 - c)) = (158 / 73) := 
sorry

end cubic_root_sum_eq_constant_term_divided_l402_40284


namespace quadratic_function_vertex_and_comparison_l402_40259

theorem quadratic_function_vertex_and_comparison
  (a b c : ℝ)
  (A_conds : 4 * a - 2 * b + c = 9)
  (B_conds : c = 3)
  (C_conds : 16 * a + 4 * b + c = 3) :
  (a = 1/2 ∧ b = -2 ∧ c = 3) ∧
  (∀ (m : ℝ) (y₁ y₂ : ℝ),
     y₁ = 1/2 * m^2 - 2 * m + 3 ∧
     y₂ = 1/2 * (m + 1)^2 - 2 * (m + 1) + 3 →
     (m < 3/2 → y₁ > y₂) ∧
     (m = 3/2 → y₁ = y₂) ∧
     (m > 3/2 → y₁ < y₂)) :=
by
  sorry

end quadratic_function_vertex_and_comparison_l402_40259


namespace total_students_at_competition_l402_40218
-- Import necessary Lean libraries for arithmetic and logic

-- Define the conditions as variables and expressions
namespace ScienceFair

variables (K KKnowItAll KarenHigh NovelCoronaHigh : ℕ)
variables (hK : KKnowItAll = 50)
variables (hKH : KarenHigh = 3 * KKnowItAll / 5)
variables (hNCH : NovelCoronaHigh = 2 * (KKnowItAll + KarenHigh))

-- Define the proof problem
theorem total_students_at_competition (KKnowItAll KarenHigh NovelCoronaHigh : ℕ)
  (hK : KKnowItAll = 50)
  (hKH : KarenHigh = 3 * KKnowItAll / 5)
  (hNCH : NovelCoronaHigh = 2 * (KKnowItAll + KarenHigh)) :
  KKnowItAll + KarenHigh + NovelCoronaHigh = 240 := by
  sorry

end ScienceFair

end total_students_at_competition_l402_40218


namespace minimum_value_l402_40265

theorem minimum_value (a b : ℝ) (h₀ : 0 ≤ a) (h₁ : 0 ≤ b) (h₂ : a + b = 1) : 
  (∃ (x : ℝ), x = a + 2*b) → (∃ (y : ℝ), y = 2*a + b) → 
  (∀ (x y : ℝ), x + y = 3 → (1/x + 4/y) ≥ 3) :=
by
  sorry

end minimum_value_l402_40265


namespace larger_square_side_length_l402_40225

theorem larger_square_side_length :
  ∃ (a : ℕ), ∃ (b : ℕ), a^2 = b^2 + 2001 ∧ (a = 1001 ∨ a = 335 ∨ a = 55 ∨ a = 49) :=
by
  sorry

end larger_square_side_length_l402_40225


namespace ratio_of_poets_to_novelists_l402_40281

-- Define the conditions
def total_people : ℕ := 24
def novelists : ℕ := 15
def poets := total_people - novelists

-- Theorem asserting the ratio of poets to novelists
theorem ratio_of_poets_to_novelists (h1 : poets = total_people - novelists) : poets / novelists = 3 / 5 := by
  sorry

end ratio_of_poets_to_novelists_l402_40281


namespace f_25_over_11_neg_l402_40266

variable (f : ℚ → ℚ)
axiom f_mul : ∀ a b : ℚ, a > 0 → b > 0 → f (a * b) = f a + f b
axiom f_prime : ∀ p : ℕ, Prime p → f p = p

theorem f_25_over_11_neg : f (25 / 11) < 0 :=
by
  -- You can prove the necessary steps during interactive proof:
  -- Using primes 25 = 5^2 and 11 itself,
  -- f (25/11) = f 25 - f 11. Thus, f (25) = 2f(5) = 2 * 5 = 10 and f(11) = 11
  -- Therefore, f (25/11) = 10 - 11 = -1 < 0.
  sorry

end f_25_over_11_neg_l402_40266


namespace tom_tim_typing_ratio_l402_40275

variable (T M : ℝ)

theorem tom_tim_typing_ratio (h1 : T + M = 12) (h2 : T + 1.3 * M = 15) : M / T = 5 :=
sorry

end tom_tim_typing_ratio_l402_40275
