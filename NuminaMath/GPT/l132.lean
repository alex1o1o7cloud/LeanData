import Mathlib

namespace charlie_has_54_crayons_l132_13205

theorem charlie_has_54_crayons
  (crayons_Billie : ℕ)
  (crayons_Bobbie : ℕ)
  (crayons_Lizzie : ℕ)
  (crayons_Charlie : ℕ)
  (h1 : crayons_Billie = 18)
  (h2 : crayons_Bobbie = 3 * crayons_Billie)
  (h3 : crayons_Lizzie = crayons_Bobbie / 2)
  (h4 : crayons_Charlie = 2 * crayons_Lizzie) : 
  crayons_Charlie = 54 := 
sorry

end charlie_has_54_crayons_l132_13205


namespace find_N_l132_13248

-- Definitions and conditions directly appearing in the problem
variable (X Y Z N : ℝ)

axiom condition1 : 0.15 * X = 0.25 * N + Y
axiom condition2 : X + Y = Z

-- The theorem to prove
theorem find_N : N = 4.6 * X - 4 * Z := by
  sorry

end find_N_l132_13248


namespace math_problem_l132_13297

open Nat

-- Given conditions
def S (n : ℕ) : ℕ := n * (n + 1)

-- Definitions for the terms a_n, b_n, c_n, and the sum T_n
def a_n (n : ℕ) (h : n ≠ 0) : ℕ := if n = 1 then 2 else 2 * n
def b_n (n : ℕ) (h : n ≠ 0) : ℕ := 2 * (3^n + 1)
def c_n (n : ℕ) (h : n ≠ 0) : ℕ := a_n n h * b_n n h / 4
def T (n : ℕ) (h : 0 < n) : ℕ := 
  (2 * n - 1) * 3^(n + 1) / 4 + 3 / 4 + n * (n + 1) / 2

-- Main theorem to establish the solution
theorem math_problem (n : ℕ) (h : n ≠ 0) : 
  S n = n * (n + 1) →
  a_n n h = 2 * n ∧ 
  b_n n h = 2 * (3^n + 1) ∧ 
  T n (Nat.pos_of_ne_zero h) = (2 * n - 1) * 3^(n + 1) / 4 + 3 / 4 + n * (n + 1) / 2 := 
by
  intros hS
  sorry

end math_problem_l132_13297


namespace seventy_fifth_elem_in_s_l132_13222

-- Define the set s
def s : Set ℕ := {x | ∃ n : ℕ, x = 8 * n + 5}

-- State the main theorem
theorem seventy_fifth_elem_in_s : (∃ n : ℕ, n = 74 ∧ (8 * n + 5) = 597) :=
by
  -- The proof is skipped using sorry
  sorry

end seventy_fifth_elem_in_s_l132_13222


namespace digit_sum_10_pow_93_minus_937_l132_13291

-- Define a function to compute the sum of digits of a number
def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem digit_sum_10_pow_93_minus_937 :
  sum_of_digits (10^93 - 937) = 819 :=
by
  sorry

end digit_sum_10_pow_93_minus_937_l132_13291


namespace compelling_quadruples_l132_13286
   
   def isCompellingQuadruple (a b c d : ℕ) : Prop :=
     1 ≤ a ∧ a < b ∧ b < c ∧ c < d ∧ d ≤ 10 ∧ a + d < b + c 

   def compellingQuadruplesCount (count : ℕ) : Prop :=
     count = 80
   
   theorem compelling_quadruples :
     ∃ count, compellingQuadruplesCount count :=
   by
     use 80
     sorry
   
end compelling_quadruples_l132_13286


namespace pages_revised_twice_l132_13257

theorem pages_revised_twice
  (x : ℕ)
  (h1 : ∀ x, x > 30 → 1000 + 100 + 10 * x ≠ 1400)
  (h2 : ∀ x, x < 30 → 1000 + 100 + 10 * x ≠ 1400)
  (h3 : 1000 + 100 + 10 * 30 = 1400) :
  x = 30 :=
by
  sorry

end pages_revised_twice_l132_13257


namespace contrapositive_l132_13271

theorem contrapositive (x y : ℝ) : (¬ (x = 0 ∧ y = 0)) → (x^2 + y^2 ≠ 0) :=
by
  intro h
  sorry

end contrapositive_l132_13271


namespace possible_sets_B_l132_13260

def A : Set ℤ := {-1}

def isB (B : Set ℤ) : Prop :=
  A ∪ B = {-1, 3}

theorem possible_sets_B : ∀ B : Set ℤ, isB B → B = {3} ∨ B = {-1, 3} :=
by
  intros B hB
  sorry

end possible_sets_B_l132_13260


namespace additional_payment_is_65_l132_13253

def installments (n : ℕ) : ℤ := 65
def first_payment : ℕ := 20
def first_amount : ℤ := 410
def remaining_payment (x : ℤ) : ℕ := 45
def remaining_amount (x : ℤ) : ℤ := 410 + x
def average_amount : ℤ := 455

-- Define the total amount paid using both methods
def total_amount (x : ℤ) : ℤ := (20 * 410) + (45 * (410 + x))
def total_average : ℤ := 65 * 455

theorem additional_payment_is_65 :
  total_amount 65 = total_average :=
sorry

end additional_payment_is_65_l132_13253


namespace range_of_f_l132_13227

noncomputable def f (x : ℝ) : ℝ := (Real.arccos x) ^ 3 + (Real.arcsin x) ^ 3

theorem range_of_f : 
  ∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → 
           ∃ y : ℝ, y = f x ∧ (y ≥ (Real.pi ^ 3) / 32) ∧ (y ≤ (7 * (Real.pi ^ 3)) / 8) :=
sorry

end range_of_f_l132_13227


namespace fran_travel_time_l132_13251

theorem fran_travel_time (joann_speed fran_speed : ℝ) (joann_time joann_distance : ℝ) :
  joann_speed = 15 → joann_time = 4 → joann_distance = joann_speed * joann_time →
  fran_speed = 20 → fran_time = joann_distance / fran_speed →
  fran_time = 3 :=
by 
  intros h1 h2 h3 h4 h5
  sorry

end fran_travel_time_l132_13251


namespace smallest_model_length_l132_13211

theorem smallest_model_length 
  (full_size_length : ℕ)
  (mid_size_ratio : ℚ)
  (smallest_size_ratio : ℚ)
  (H1 : full_size_length = 240)
  (H2 : mid_size_ratio = 1/10)
  (H3 : smallest_size_ratio = 1/2) 
  : full_size_length * mid_size_ratio * smallest_size_ratio = 12 :=
by
  sorry

end smallest_model_length_l132_13211


namespace cannot_be_right_angle_triangle_l132_13217

-- Definition of the converse of the Pythagorean theorem
def is_right_angle_triangle (a b c : ℕ) : Prop :=
  a ^ 2 + b ^ 2 = c ^ 2

-- Definition to check if a given set of sides cannot form a right-angled triangle
def cannot_form_right_angle_triangle (a b c : ℕ) : Prop :=
  ¬ is_right_angle_triangle a b c

-- Given sides of the triangle option D
theorem cannot_be_right_angle_triangle : cannot_form_right_angle_triangle 3 4 6 :=
  by sorry

end cannot_be_right_angle_triangle_l132_13217


namespace triangle_area_triangle_perimeter_l132_13230

noncomputable def area_of_triangle (A B C : ℝ) (a b c : ℝ) := 
  1/2 * b * c * (Real.sin A)

theorem triangle_area (A B C a b c : ℝ) 
  (h1 : b^2 + c^2 - a^2 = bc) 
  (h2 : A = Real.pi / 3) : 
  area_of_triangle A B C a b c = Real.sqrt 3 / 4 := 
  sorry

theorem triangle_perimeter (A B C a b c : ℝ) 
  (h1 : b^2 + c^2 - a^2 = bc) 
  (h2 : 4 * Real.cos B * Real.cos C - 1 = 0) 
  (h3 : b + c = 2)
  (h4 : a = 1) :
  a + b + c = 3 :=
  sorry

end triangle_area_triangle_perimeter_l132_13230


namespace certain_number_x_l132_13233

theorem certain_number_x :
  ∃ x : ℤ, (287 * 287 + 269 * 269 - x * (287 * 269) = 324) ∧ (x = 2) := 
by {
  use 2,
  sorry
}

end certain_number_x_l132_13233


namespace difference_cubics_divisible_by_24_l132_13242

theorem difference_cubics_divisible_by_24 
    (a b : ℤ) (h : ∃ k : ℤ, a - b = 3 * k) : 
    ∃ k : ℤ, (2 * a + 1)^3 - (2 * b + 1)^3 = 24 * k :=
by
  sorry

end difference_cubics_divisible_by_24_l132_13242


namespace chord_length_of_circle_intersected_by_line_l132_13209

open Real

-- Definitions for the conditions given in the problem
def line_eqn (x y : ℝ) : Prop := x - y - 1 = 0
def circle_eqn (x y : ℝ) : Prop := x^2 - 4 * x + y^2 = 4

-- The proof statement (problem) in Lean 4
theorem chord_length_of_circle_intersected_by_line :
  ∀ (x y : ℝ), circle_eqn x y → line_eqn x y → ∃ L : ℝ, L = sqrt 17 := by
  sorry

end chord_length_of_circle_intersected_by_line_l132_13209


namespace log_base_10_of_2_bounds_l132_13224

theorem log_base_10_of_2_bounds :
  (10^3 = 1000) ∧ (10^4 = 10000) ∧ (2^11 = 2048) ∧ (2^14 = 16384) →
  (3 / 11 : ℝ) < Real.log 2 / Real.log 10 ∧ Real.log 2 / Real.log 10 < (2 / 7 : ℝ) :=
by
  sorry

end log_base_10_of_2_bounds_l132_13224


namespace greatest_number_of_quarters_l132_13223

def eva_has_us_coins : ℝ := 4.80
def quarters_and_dimes_have_same_count (q : ℕ) : Prop := (0.25 * q + 0.10 * q = eva_has_us_coins)

theorem greatest_number_of_quarters : ∃ (q : ℕ), quarters_and_dimes_have_same_count q ∧ q = 13 :=
sorry

end greatest_number_of_quarters_l132_13223


namespace capacity_of_second_bucket_l132_13288

theorem capacity_of_second_bucket (c1 : ∃ (tank_capacity : ℕ), tank_capacity = 12 * 49) (c2 : ∃ (bucket_count : ℕ), bucket_count = 84) :
  ∃ (bucket_capacity : ℕ), bucket_capacity = 7 :=
by
  -- Extract the total capacity of the tank from condition 1
  obtain ⟨tank_capacity, htank⟩ := c1
  -- Extract the number of buckets from condition 2
  obtain ⟨bucket_count, hcount⟩ := c2
  -- Use the given relations to calculate the capacity of each bucket
  use tank_capacity / bucket_count
  -- Provide the necessary calculations
  sorry

end capacity_of_second_bucket_l132_13288


namespace quadrilateral_ABCD_pq_sum_l132_13243

noncomputable def AB_pq_sum : ℕ :=
  let p : ℕ := 9
  let q : ℕ := 141
  p + q

theorem quadrilateral_ABCD_pq_sum (BC CD AD : ℕ) (m_angle_A m_angle_B : ℕ) (hBC : BC = 8) (hCD : CD = 12) (hAD : AD = 10) (hAngleA : m_angle_A = 60) (hAngleB : m_angle_B = 60) : AB_pq_sum = 150 := by sorry

end quadrilateral_ABCD_pq_sum_l132_13243


namespace problem_statement_l132_13264

-- Mathematical Definitions
def num_students : ℕ := 6
def num_boys : ℕ := 4
def num_girls : ℕ := 2
def num_selected : ℕ := 3

def event_A : Prop := ∃ (boyA : ℕ), boyA < num_boys
def event_B : Prop := ∃ (girlB : ℕ), girlB < num_girls

def C (n k : ℕ) : ℕ := Nat.choose n k

-- Total number of ways to select 3 out of 6 students
def total_ways : ℕ := C num_students num_selected

-- Probability of event A
def P_A : ℚ := C (num_students - 1) (num_selected - 1) / total_ways

-- Probability of events A and B
def P_AB : ℚ := C (num_students - 2) (num_selected - 2) / total_ways

-- Conditional probability P(B|A)
def P_B_given_A : ℚ := P_AB / P_A

theorem problem_statement : P_B_given_A = 2 / 5 := sorry

end problem_statement_l132_13264


namespace rectangle_vertices_complex_plane_l132_13270

theorem rectangle_vertices_complex_plane (b : ℝ) :
  (∀ (z : ℂ), z^4 - 10*z^3 + (16*b : ℂ)*z^2 - 2*(3*b^2 - 5*b + 4 : ℂ)*z + 6 = 0 →
    (∃ (w₁ w₂ : ℂ), z = w₁ ∨ z = w₂)) →
  (b = 5 / 3 ∨ b = 2) :=
sorry

end rectangle_vertices_complex_plane_l132_13270


namespace UF_opponent_score_l132_13275

theorem UF_opponent_score 
  (total_points : ℕ)
  (games_played : ℕ)
  (previous_points_avg : ℕ)
  (championship_score : ℕ)
  (opponent_score : ℕ)
  (total_points_condition : total_points = 720)
  (games_played_condition : games_played = 24)
  (previous_points_avg_condition : previous_points_avg = total_points / games_played)
  (championship_score_condition : championship_score = previous_points_avg / 2 - 2)
  (loss_by_condition : opponent_score = championship_score - 2) :
  opponent_score = 11 :=
by
  sorry

end UF_opponent_score_l132_13275


namespace b_should_pay_l132_13287

-- Definitions for the number of horses and their duration in months
def horses_of_a := 12
def months_of_a := 8

def horses_of_b := 16
def months_of_b := 9

def horses_of_c := 18
def months_of_c := 6

-- Total rent
def total_rent := 870

-- Shares in horse-months for each person
def share_of_a := horses_of_a * months_of_a
def share_of_b := horses_of_b * months_of_b
def share_of_c := horses_of_c * months_of_c

-- Total share in horse-months
def total_share := share_of_a + share_of_b + share_of_c

-- Fraction for b
def fraction_for_b := share_of_b / total_share

-- Amount b should pay
def amount_for_b := total_rent * fraction_for_b

-- Theorem to verify the amount b should pay
theorem b_should_pay : amount_for_b = 360 := by
  -- The steps of the proof would go here
  sorry

end b_should_pay_l132_13287


namespace num_possible_values_a_l132_13241

theorem num_possible_values_a (a : ℕ) :
  (9 ∣ a) ∧ (a ∣ 18) ∧ (0 < a) → ∃ n : ℕ, n = 2 :=
by
  sorry

end num_possible_values_a_l132_13241


namespace curve_symmetry_l132_13295

theorem curve_symmetry :
  ∃ θ : ℝ, θ = 5 * Real.pi / 6 ∧
  ∀ (ρ θ' : ℝ), ρ = 4 * Real.sin (θ' - Real.pi / 3) ↔ ρ = 4 * Real.sin ((θ - θ') - Real.pi / 3) :=
sorry

end curve_symmetry_l132_13295


namespace fraction_equality_l132_13239

theorem fraction_equality : (16 : ℝ) / (8 * 17) = (1.6 : ℝ) / (0.8 * 17) := 
sorry

end fraction_equality_l132_13239


namespace length_of_adjacent_side_l132_13231

variable (a b : ℝ)

theorem length_of_adjacent_side (area : ℝ) (side : ℝ) :
  area = 6 * a^3 + 9 * a^2 - 3 * a * b →
  side = 3 * a →
  (area / side = 2 * a^2 + 3 * a - b) :=
by
  intro h_area
  intro h_side
  sorry

end length_of_adjacent_side_l132_13231


namespace xy_product_eq_two_l132_13236

theorem xy_product_eq_two (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hxy : x ≠ y) (h : x + 2 / x = y + 2 / y) : x * y = 2 := 
sorry

end xy_product_eq_two_l132_13236


namespace sum_of_two_numbers_l132_13247

-- Define the two numbers and conditions
variables {x y : ℝ}
axiom prod_eq : x * y = 120
axiom sum_squares_eq : x^2 + y^2 = 289

-- The statement we want to prove
theorem sum_of_two_numbers (x y : ℝ) (prod_eq : x * y = 120) (sum_squares_eq : x^2 + y^2 = 289) : x + y = 23 :=
sorry

end sum_of_two_numbers_l132_13247


namespace polynomial_evaluation_l132_13262

theorem polynomial_evaluation (p : Polynomial ℚ) 
  (hdeg : p.degree = 7)
  (h : ∀ n : ℕ, n ≤ 7 → p.eval (2^n) = 1 / 2^(n + 1)) : 
  p.eval 0 = 255 / 2^28 := 
sorry

end polynomial_evaluation_l132_13262


namespace unique_array_count_l132_13252

theorem unique_array_count (n m : ℕ) (h_conds : n * m = 49 ∧ n ≥ 2 ∧ m ≥ 2 ∧ n = m) :
  ∃! (n m : ℕ), (n * m = 49 ∧ n ≥ 2 ∧ m ≥ 2 ∧ n = m) :=
by
  sorry

end unique_array_count_l132_13252


namespace root_quadratic_expression_value_l132_13281

theorem root_quadratic_expression_value (m : ℝ) (h : m^2 - m - 3 = 0) : 2023 - m^2 + m = 2020 := 
by 
  sorry

end root_quadratic_expression_value_l132_13281


namespace find_length_AE_l132_13246

theorem find_length_AE (AB BC CD DE AC CE AE : ℕ) 
  (h1 : AB = 2) 
  (h2 : BC = 2) 
  (h3 : CD = 5) 
  (h4 : DE = 7)
  (h5 : AC > 2) 
  (h6 : AC < 4) 
  (h7 : CE > 2) 
  (h8 : CE < 5)
  (h9 : AC ≠ CE)
  (h10 : AC ≠ AE)
  (h11 : CE ≠ AE)
  : AE = 5 :=
sorry

end find_length_AE_l132_13246


namespace rectangle_placement_l132_13298

theorem rectangle_placement (a b c d : ℝ)
  (h1 : a < c)
  (h2 : c < d)
  (h3 : d < b)
  (h4 : a * b < c * d) :
  (b^2 - a^2)^2 ≤ (b * d - a * c)^2 + (b * c - a * d)^2 :=
sorry

end rectangle_placement_l132_13298


namespace div_1959_l132_13213

theorem div_1959 (n : ℕ) : ∃ k : ℤ, 5^(8 * n) - 2^(4 * n) * 7^(2 * n) = k * 1959 := 
by 
  sorry

end div_1959_l132_13213


namespace complex_solution_l132_13278

theorem complex_solution (z : ℂ) (h : z^2 = -5 - 12 * Complex.I) :
  z = 2 - 3 * Complex.I ∨ z = -2 + 3 * Complex.I := 
sorry

end complex_solution_l132_13278


namespace num_female_students_l132_13282

theorem num_female_students (F : ℕ) (h1: 8 * 85 + F * 92 = (8 + F) * 90) : F = 20 := 
by
  sorry

end num_female_students_l132_13282


namespace sin_eq_sqrt3_div_2_range_l132_13228

theorem sin_eq_sqrt3_div_2_range :
  {x | 0 ≤ x ∧ x ≤ 2 * Real.pi ∧ Real.sin x ≥ Real.sqrt 3 / 2} = 
  {x | Real.pi / 3 ≤ x ∧ x ≤ 2 * Real.pi / 3} :=
sorry

end sin_eq_sqrt3_div_2_range_l132_13228


namespace quadratic_min_value_l132_13266

theorem quadratic_min_value (p r : ℝ) (f : ℝ → ℝ) (h₀ : ∀ x, f x = x^2 + 2 * p * x + r) (h₁ : ∃ x₀, f x₀ = 1 ∧ ∀ x, f x₀ ≤ f x) : r = p^2 + 1 :=
by
  sorry

end quadratic_min_value_l132_13266


namespace sum_of_reciprocal_squares_l132_13244

theorem sum_of_reciprocal_squares
  (p q r : ℝ)
  (h1 : p + q + r = 9)
  (h2 : p * q + q * r + r * p = 8)
  (h3 : p * q * r = -2) :
  (1 / p ^ 2 + 1 / q ^ 2 + 1 / r ^ 2) = 25 := by
  sorry

end sum_of_reciprocal_squares_l132_13244


namespace price_per_pound_of_rocks_l132_13215

def number_of_rocks : ℕ := 10
def average_weight_per_rock : ℝ := 1.5
def total_amount_made : ℝ := 60

theorem price_per_pound_of_rocks:
  (total_amount_made / (number_of_rocks * average_weight_per_rock)) = 4 := 
by
  sorry

end price_per_pound_of_rocks_l132_13215


namespace red_grapes_in_salad_l132_13203

theorem red_grapes_in_salad {G R B : ℕ} 
  (h1 : R = 3 * G + 7)
  (h2 : B = G - 5)
  (h3 : G + R + B = 102) : R = 67 :=
sorry

end red_grapes_in_salad_l132_13203


namespace expand_and_simplify_l132_13232

theorem expand_and_simplify (x : ℝ) : (2*x + 6)*(x + 9) = 2*x^2 + 24*x + 54 :=
by
  sorry

end expand_and_simplify_l132_13232


namespace chen_recording_l132_13226

variable (standard xia_steps chen_steps : ℕ)
variable (xia_record : ℤ)

-- Conditions: 
-- standard = 5000
-- Xia walked 6200 steps, recorded as +1200 steps
def met_standard (s : ℕ) : Prop :=
  s >= 5000

def xia_condition := (xia_steps = 6200) ∧ (xia_record = 1200) ∧ (xia_record = (xia_steps : ℤ) - 5000)

-- Question and solution combined into a statement: 
-- Chen walked 4800 steps, recorded as -200 steps
def chen_condition := (chen_steps = 4800) ∧ (met_standard chen_steps = false) → (((standard : ℤ) - chen_steps) * -1 = -200)

-- Proof goal:
theorem chen_recording (h₁ : standard = 5000) (h₂ : xia_condition xia_steps xia_record):
  chen_condition standard chen_steps :=
by
  sorry

end chen_recording_l132_13226


namespace ratio_of_intercepts_l132_13216

theorem ratio_of_intercepts (b s t : ℝ) (h1 : s = -2 * b / 5) (h2 : t = -3 * b / 7) :
  s / t = 14 / 15 :=
by
  sorry

end ratio_of_intercepts_l132_13216


namespace evaluate_g_at_neg2_l132_13274

def g (x : ℝ) : ℝ := x^3 - 3 * x^2 + 4

theorem evaluate_g_at_neg2 : g (-2) = -16 := by
  sorry

end evaluate_g_at_neg2_l132_13274


namespace symmetric_points_add_l132_13210

theorem symmetric_points_add (a b : ℝ) : 
  (P : ℝ × ℝ) → (Q : ℝ × ℝ) →
  P = (a-1, 5) →
  Q = (2, b-1) →
  (P.fst = Q.fst) →
  P.snd = -Q.snd →
  a + b = -1 :=
by
  sorry

end symmetric_points_add_l132_13210


namespace hilt_books_difference_l132_13200

noncomputable def original_price : ℝ := 11
noncomputable def discount_rate : ℝ := 0.20
noncomputable def discount_price (price : ℝ) (rate : ℝ) : ℝ := price * (1 - rate)
noncomputable def quantity : ℕ := 15
noncomputable def sale_price : ℝ := 25
noncomputable def tax_rate : ℝ := 0.10
noncomputable def price_with_tax (price : ℝ) (rate : ℝ) : ℝ := price * (1 + rate)

noncomputable def total_cost : ℝ := discount_price original_price discount_rate * quantity
noncomputable def total_revenue : ℝ := price_with_tax sale_price tax_rate * quantity
noncomputable def profit : ℝ := total_revenue - total_cost

theorem hilt_books_difference : profit = 280.50 :=
by
  sorry

end hilt_books_difference_l132_13200


namespace cricket_team_players_l132_13234

theorem cricket_team_players (P N : ℕ) (h1 : 37 = 37) 
  (h2 : (57 - 37) = 20) 
  (h3 : ∀ N, (2 / 3 : ℚ) * N = 20 → N = 30) 
  (h4 : P = 37 + 30) : P = 67 := 
by
  -- Proof steps will go here
  sorry

end cricket_team_players_l132_13234


namespace eighteen_women_time_l132_13206

theorem eighteen_women_time (h : ∀ (n : ℕ), n = 6 → ∀ (t : ℕ), t = 60 → true) : ∀ (n : ℕ), n = 18 → ∀ (t : ℕ), t = 20 → true :=
by
  sorry

end eighteen_women_time_l132_13206


namespace ratio_of_pond_to_field_area_l132_13254

theorem ratio_of_pond_to_field_area
  (l w : ℕ)
  (field_area pond_area : ℕ)
  (h1 : l = 2 * w)
  (h2 : l = 36)
  (h3 : pond_area = 9 * 9)
  (field_area_def : field_area = l * w)
  (pond_area_def : pond_area = 81) :
  pond_area / field_area = 1 / 8 := 
sorry

end ratio_of_pond_to_field_area_l132_13254


namespace max_value_of_a_l132_13256

theorem max_value_of_a (a : ℝ) :
  (∀ x : ℝ, x^2 - a * x + a ≥ 0) → a ≤ 4 := 
by {
  sorry
}

end max_value_of_a_l132_13256


namespace value_of_polynomial_l132_13290

variable {R : Type} [CommRing R]

theorem value_of_polynomial 
  (m : R) 
  (h : 2 * m^2 - 3 * m - 1 = 0) : 
  6 * m^2 - 9 * m + 2019 = 2022 := by
  sorry

end value_of_polynomial_l132_13290


namespace fraction_day_crew_loaded_l132_13280

variable (D W : ℕ)  -- D: Number of boxes loaded by each worker on the day crew, W: Number of workers on the day crew

-- Condition 1: Each worker on the night crew loaded 3/4 as many boxes as each worker on the day crew
def boxes_loaded_night_worker : ℕ := 3 * D / 4
-- Condition 2: The night crew has 5/6 as many workers as the day crew
def workers_night : ℕ := 5 * W / 6

-- Question: Fraction of all the boxes loaded by the day crew
theorem fraction_day_crew_loaded :
  (D * W : ℚ) / ((D * W) + (3 * D / 4) * (5 * W / 6)) = (8 / 13) := by
  sorry

end fraction_day_crew_loaded_l132_13280


namespace p_implies_q_l132_13202

theorem p_implies_q (x : ℝ) :
  (|2*x - 3| < 1) → (x*(x - 3) < 0) :=
by
  intros hp
  sorry

end p_implies_q_l132_13202


namespace average_speed_last_segment_l132_13289

variable (total_distance : ℕ := 120)
variable (total_minutes : ℕ := 120)
variable (first_segment_minutes : ℕ := 40)
variable (first_segment_speed : ℕ := 50)
variable (second_segment_minutes : ℕ := 40)
variable (second_segment_speed : ℕ := 55)
variable (third_segment_speed : ℕ := 75)

theorem average_speed_last_segment :
  let total_hours := total_minutes / 60
  let average_speed := total_distance / total_hours
  let speed_first_segment := first_segment_speed * (first_segment_minutes / 60)
  let speed_second_segment := second_segment_speed * (second_segment_minutes / 60)
  let speed_third_segment := third_segment_speed * (third_segment_minutes / 60)
  average_speed = (speed_first_segment + speed_second_segment + speed_third_segment) / 3 →
  third_segment_speed = 75 :=
by
  sorry

end average_speed_last_segment_l132_13289


namespace middle_aged_employees_participating_l132_13265

-- Define the total number of employees and the ratio
def total_employees : ℕ := 1200
def ratio_elderly : ℕ := 1
def ratio_middle_aged : ℕ := 5
def ratio_young : ℕ := 6

-- Define the number of employees chosen for the performance
def chosen_employees : ℕ := 36

-- Calculate the number of middle-aged employees participating in the performance
theorem middle_aged_employees_participating : (36 * ratio_middle_aged / (ratio_elderly + ratio_middle_aged + ratio_young)) = 15 :=
by
  sorry

end middle_aged_employees_participating_l132_13265


namespace ratio_of_15th_term_l132_13207

theorem ratio_of_15th_term (a d b e : ℤ) :
  (∀ n : ℕ, (n * (2 * a + (n - 1) * d)) / (n * (2 * b + (n - 1) * e)) = (7 * n^2 + 1) / (4 * n^2 + 27)) →
  (a + 14 * d) / (b + 14 * e) = 7 / 4 :=
by sorry

end ratio_of_15th_term_l132_13207


namespace necessary_but_not_sufficient_condition_l132_13214

-- Define the condition p: x^2 - x < 0
def p (x : ℝ) : Prop := x^2 - x < 0

-- Define the necessary but not sufficient condition
def necessary_but_not_sufficient (x : ℝ) : Prop := -1 < x ∧ x < 1

-- State the theorem
theorem necessary_but_not_sufficient_condition :
  ∀ x : ℝ, p x → necessary_but_not_sufficient x :=
sorry

end necessary_but_not_sufficient_condition_l132_13214


namespace solve_for_x_l132_13267

theorem solve_for_x :
  ∃ x : ℝ, 5 ^ (Real.logb 5 15) = 7 * x + 2 ∧ x = 13 / 7 :=
by
  sorry

end solve_for_x_l132_13267


namespace max_possible_intersections_l132_13277

theorem max_possible_intersections : 
  let num_x := 12
  let num_y := 6
  let intersections := (num_x * (num_x - 1) / 2) * (num_y * (num_y - 1) / 2)
  intersections = 990 := 
by 
  sorry

end max_possible_intersections_l132_13277


namespace intersection_of_A_and_B_l132_13261

def A : Set ℝ := {x | 0 ≤ x ∧ x ≤ 2}
def B : Set ℝ := {x | -1 < x ∧ x ≤ 1}

theorem intersection_of_A_and_B : A ∩ B = {x | 0 ≤ x ∧ x ≤ 1} := by
  sorry

end intersection_of_A_and_B_l132_13261


namespace min_value_of_ellipse_l132_13235

noncomputable def min_m_plus_n (a b : ℝ) (h_ab_nonzero : a * b ≠ 0) (h_abs_diff : |a| ≠ |b|) : ℝ :=
(a ^ (2/3) + b ^ (2/3)) ^ (3/2)

theorem min_value_of_ellipse (m n a b : ℝ) (h1 : m > n) (h2 : n > 0) (h_ellipse : (a^2 / m^2) + (b^2 / n^2) = 1) (h_ab_nonzero : a * b ≠ 0) (h_abs_diff : |a| ≠ |b|) :
  (m + n) = min_m_plus_n a b h_ab_nonzero h_abs_diff :=
sorry

end min_value_of_ellipse_l132_13235


namespace pineapples_sold_l132_13245

/-- 
There were initially 86 pineapples in the store. After selling some pineapples,
9 of the remaining pineapples were rotten and were discarded. Given that there 
are 29 fresh pineapples left, prove that the number of pineapples sold is 48.
-/
theorem pineapples_sold (initial_pineapples : ℕ) (rotten_pineapples : ℕ) (remaining_fresh_pineapples : ℕ)
  (h_init : initial_pineapples = 86)
  (h_rotten : rotten_pineapples = 9)
  (h_fresh : remaining_fresh_pineapples = 29) :
  initial_pineapples - (remaining_fresh_pineapples + rotten_pineapples) = 48 :=
sorry

end pineapples_sold_l132_13245


namespace triangle_ABC_problem_l132_13212

noncomputable def perimeter_of_triangle (a b c : ℝ) : ℝ := a + b + c

theorem triangle_ABC_problem 
  (a b c : ℝ) (A B C : ℝ) 
  (h1 : a = 3) 
  (h2 : B = π / 3) 
  (area : ℝ)
  (h3 : (1/2) * a * c * Real.sin B = 6 * Real.sqrt 3) :

  perimeter_of_triangle a b c = 18 ∧ 
  Real.sin (2 * A) = 39 * Real.sqrt 3 / 98 := 
by 
  sorry

end triangle_ABC_problem_l132_13212


namespace kaleb_tickets_l132_13250

variable (T : Nat)
variable (tickets_left : Nat) (ticket_cost : Nat) (total_spent : Nat)

theorem kaleb_tickets : tickets_left = 3 → ticket_cost = 9 → total_spent = 27 → T = 6 :=
by
  sorry

end kaleb_tickets_l132_13250


namespace Q_coordinates_l132_13268

structure Point where
  x : ℝ
  y : ℝ

def O : Point := ⟨0, 0⟩
def P : Point := ⟨0, 3⟩
def R : Point := ⟨5, 0⟩

def isRectangle (A B C D : Point) : Prop :=
  -- replace this with the actual implementation of rectangle properties
  sorry

theorem Q_coordinates :
  ∃ Q : Point, isRectangle O P Q R ∧ Q.x = 5 ∧ Q.y = 3 :=
by
  -- replace this with the actual proof
  sorry

end Q_coordinates_l132_13268


namespace find_q_l132_13258

theorem find_q (q x : ℝ) (h1 : x = 2) (h2 : q * x - 3 = 11) : q = 7 :=
by
  sorry

end find_q_l132_13258


namespace NumberOfStudentsEnrolledOnlyInEnglish_l132_13276

-- Definition of the problem's variables and conditions
variables (TotalStudents BothEnglishAndGerman TotalGerman OnlyEnglish OnlyGerman : ℕ)
variables (h1 : TotalStudents = 52)
variables (h2 : BothEnglishAndGerman = 12)
variables (h3 : TotalGerman = 22)
variables (h4 : TotalStudents = OnlyEnglish + OnlyGerman + BothEnglishAndGerman)
variables (h5 : OnlyGerman = TotalGerman - BothEnglishAndGerman)

-- Theorem to prove the number of students enrolled only in English
theorem NumberOfStudentsEnrolledOnlyInEnglish : OnlyEnglish = 30 :=
by
  -- Insert the necessary proof steps here to derive the number of students enrolled only in English from the given conditions
  sorry

end NumberOfStudentsEnrolledOnlyInEnglish_l132_13276


namespace Janice_age_l132_13204

theorem Janice_age (x : ℝ) (h : x + 12 = 8 * (x - 2)) : x = 4 := by
  sorry

end Janice_age_l132_13204


namespace candy_bars_eaten_l132_13219

theorem candy_bars_eaten (calories_per_candy : ℕ) (total_calories : ℕ) (h1 : calories_per_candy = 31) (h2 : total_calories = 341) :
  total_calories / calories_per_candy = 11 :=
by
  sorry

end candy_bars_eaten_l132_13219


namespace non_square_solution_equiv_l132_13220

theorem non_square_solution_equiv 
  (a b : ℤ) (h1 : ¬∃ k : ℤ, a = k^2) (h2 : ¬∃ k : ℤ, b = k^2) :
  (∃ x y z w : ℤ, x^2 - a * y^2 - b * z^2 + a * b * w^2 = 0 ∧ (x, y, z, w) ≠ (0, 0, 0, 0)) ↔
  (∃ x y z : ℤ, x^2 - a * y^2 - b * z^2 = 0 ∧ (x, y, z) ≠ (0, 0, 0)) :=
by sorry

end non_square_solution_equiv_l132_13220


namespace opposite_of_neg_2022_eq_2022_l132_13284

-- Define what it means to find the opposite of a number
def opposite (n : Int) : Int := -n

-- State the theorem that needs to be proved
theorem opposite_of_neg_2022_eq_2022 : opposite (-2022) = 2022 :=
by
  -- Proof would go here but we skip it with sorry
  sorry

end opposite_of_neg_2022_eq_2022_l132_13284


namespace find_quotient_l132_13221

theorem find_quotient (dividend divisor remainder quotient : ℕ) 
  (h1 : dividend = 23) (h2 : divisor = 4) (h3 : remainder = 3)
  (h4 : dividend = (divisor * quotient) + remainder) : quotient = 5 :=
sorry

end find_quotient_l132_13221


namespace min_value_expression_l132_13237

theorem min_value_expression (x : ℝ) (h : x > 10) : (x^2) / (x - 10) ≥ 40 :=
sorry

end min_value_expression_l132_13237


namespace no_arithmetic_seq_with_sum_n_cubed_l132_13240

theorem no_arithmetic_seq_with_sum_n_cubed (a1 d : ℕ) :
  ¬ (∀ (n : ℕ), (n > 0) → (n / 2) * (2 * a1 + (n - 1) * d) = n^3) :=
sorry

end no_arithmetic_seq_with_sum_n_cubed_l132_13240


namespace line_intersects_x_axis_at_point_l132_13225

theorem line_intersects_x_axis_at_point (x1 y1 x2 y2 : ℝ) 
  (h1 : (x1, y1) = (7, -3))
  (h2 : (x2, y2) = (3, 1)) : 
  ∃ x, (x, 0) = (4, 0) :=
by
  -- sorry serves as a placeholder for the actual proof
  sorry

end line_intersects_x_axis_at_point_l132_13225


namespace equivalent_operation_l132_13293

theorem equivalent_operation (x : ℚ) : 
  (x * (2 / 3)) / (4 / 7) = x * (7 / 6) :=
by sorry

end equivalent_operation_l132_13293


namespace range_of_a_l132_13269

noncomputable def p (a : ℝ) : Prop :=
  ∃ x : ℝ, -1 ≤ x ∧ x ≤ 1 ∧ a^2 * x^2 + a * x - 2 = 0

noncomputable def q (a : ℝ) : Prop :=
  ∃ x : ℝ, x < 0 ∧ a * x^2 + 2 * x + 1 = 0

theorem range_of_a (a : ℝ) :
  (p a ∨ q a) ∧ ¬(p a ∧ q a) → (1 < a ∨ -1 < a ∧ a < 1) :=
by sorry

end range_of_a_l132_13269


namespace factorize_one_factorize_two_l132_13255

variable (x a b : ℝ)

-- Problem 1: Prove that 4x^2 - 64 = 4(x + 4)(x - 4)
theorem factorize_one : 4 * x^2 - 64 = 4 * (x + 4) * (x - 4) :=
sorry

-- Problem 2: Prove that 4ab^2 - 4a^2b - b^3 = -b(2a - b)^2
theorem factorize_two : 4 * a * b^2 - 4 * a^2 * b - b^3 = -b * (2 * a - b)^2 :=
sorry

end factorize_one_factorize_two_l132_13255


namespace cost_of_dozen_pens_l132_13208

theorem cost_of_dozen_pens 
  (x : ℝ)
  (hx_pos : 0 < x)
  (h1 : 3 * (5 * x) + 5 * x = 150)
  (h2 : 5 * x / x = 5): 
  12 * (5 * x) = 450 :=
by
  sorry

end cost_of_dozen_pens_l132_13208


namespace geometric_series_cubes_sum_l132_13294

theorem geometric_series_cubes_sum (b s : ℝ) (h : -1 < s ∧ s < 1) :
  ∑' n : ℕ, (b * s^n)^3 = b^3 / (1 - s^3) := 
sorry

end geometric_series_cubes_sum_l132_13294


namespace time_for_new_circle_l132_13201

theorem time_for_new_circle 
  (rounds : ℕ) (time : ℕ) (k : ℕ) (original_time_per_round new_time_per_round : ℝ) 
  (h1 : rounds = 8) 
  (h2 : time = 40) 
  (h3 : k = 10) 
  (h4 : original_time_per_round = time / rounds)
  (h5 : new_time_per_round = original_time_per_round * k) :
  new_time_per_round = 50 :=
by {
  sorry
}

end time_for_new_circle_l132_13201


namespace new_cost_relation_l132_13299

def original_cost (k t b : ℝ) : ℝ :=
  k * (t * b)^4

def new_cost (k t b : ℝ) : ℝ :=
  k * ((2 * b) * (0.75 * t))^4

theorem new_cost_relation (k t b : ℝ) (C : ℝ) 
  (hC : C = original_cost k t b) :
  new_cost k t b = 25.63 * C := sorry

end new_cost_relation_l132_13299


namespace jasons_shelves_l132_13292

theorem jasons_shelves (total_books : ℕ) (number_of_shelves : ℕ) (h_total_books : total_books = 315) (h_number_of_shelves : number_of_shelves = 7) : (total_books / number_of_shelves) = 45 := 
by
  sorry

end jasons_shelves_l132_13292


namespace flu_epidemic_infection_rate_l132_13249

theorem flu_epidemic_infection_rate : 
  ∃ x : ℝ, 1 + x + x * (1 + x) = 100 ∧ x = 9 := 
by
  sorry

end flu_epidemic_infection_rate_l132_13249


namespace necessary_and_sufficient_condition_l132_13285

variables {a : ℕ → ℝ}
-- Define the arithmetic sequence condition
def is_arithmetic_sequence (a : ℕ → ℝ) :=
  ∃ d, ∀ n, a (n + 1) = a n + d

-- Define the monotonically increasing condition
def is_monotonically_increasing (a : ℕ → ℝ) :=
  ∀ n, a (n + 1) > a n

-- Define the specific statement
theorem necessary_and_sufficient_condition (a : ℕ → ℝ) :
  is_arithmetic_sequence a →
  (a 1 < a 3 ↔ is_monotonically_increasing a) :=
by sorry

end necessary_and_sufficient_condition_l132_13285


namespace emily_remainder_l132_13218

theorem emily_remainder (c d : ℤ) (h1 : c % 60 = 53) (h2 : d % 42 = 35) : (c + d) % 21 = 4 :=
by
  sorry

end emily_remainder_l132_13218


namespace ursula_purchases_total_cost_l132_13263

variable (T C B Br : ℝ)
variable (hT : T = 10) (hTC : T = 2 * C) (hB : B = 0.8 * C) (hBr : Br = B / 2)

theorem ursula_purchases_total_cost : T + C + B + Br = 21 := by
  sorry

end ursula_purchases_total_cost_l132_13263


namespace winning_candidate_votes_l132_13259

theorem winning_candidate_votes (V : ℝ) (h1 : 0.62 * V - 0.38 * V = 336): 0.62 * V = 868 :=
by
  sorry

end winning_candidate_votes_l132_13259


namespace walking_rate_ratio_l132_13273

theorem walking_rate_ratio (R R' : ℝ) (usual_time early_time : ℝ) (H1 : usual_time = 42) (H2 : early_time = 36) 
(H3 : R * usual_time = R' * early_time) : (R' / R = 7 / 6) :=
by
  -- proof to be completed
  sorry

end walking_rate_ratio_l132_13273


namespace harry_basketball_points_l132_13283

theorem harry_basketball_points :
  ∃ (x y : ℕ), 
    (x < 15) ∧ 
    (y < 15) ∧ 
    (62 + x) % 11 = 0 ∧ 
    (62 + x + y) % 12 = 0 ∧ 
    (x * y = 24) :=
by
  sorry

end harry_basketball_points_l132_13283


namespace fraction_still_missing_l132_13279

theorem fraction_still_missing (x : ℕ) (hx : x > 0) :
  let lost := (1/3 : ℚ) * x
  let found := (2/3 : ℚ) * lost
  let remaining := x - lost + found
  (x - remaining) / x = (1/9 : ℚ) :=
by
  let lost := (1/3 : ℚ) * x
  let found := (2/3 : ℚ) * lost
  let remaining := x - lost + found
  have h_fraction_still_missing : (x - remaining) / x = (1/9 : ℚ) := sorry
  exact h_fraction_still_missing

end fraction_still_missing_l132_13279


namespace arithmetic_expression_evaluation_l132_13272

theorem arithmetic_expression_evaluation :
  1325 + (180 / 60) * 3 - 225 = 1109 :=
by
  sorry -- To be filled with the proof steps

end arithmetic_expression_evaluation_l132_13272


namespace tammy_total_miles_l132_13296

noncomputable def miles_per_hour : ℝ := 1.527777778
noncomputable def hours_driven : ℝ := 36.0
noncomputable def total_miles := miles_per_hour * hours_driven

theorem tammy_total_miles : abs (total_miles - 55.0) < 1e-5 :=
by
  sorry

end tammy_total_miles_l132_13296


namespace tree_height_at_2_years_l132_13229

theorem tree_height_at_2_years (h : ℕ → ℕ) 
  (h_growth : ∀ n, h (n + 1) = 3 * h n) 
  (h_5 : h 5 = 243) : 
  h 2 = 9 := 
sorry

end tree_height_at_2_years_l132_13229


namespace intersection_with_x_axis_l132_13238

theorem intersection_with_x_axis (a : ℝ) (h : 2 * a - 4 = 0) : a = 2 := by
  sorry

end intersection_with_x_axis_l132_13238
