import Mathlib

namespace equation_has_roots_l1750_175043

theorem equation_has_roots (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (a^2 * (x₁ - 2) + a * (39 - 20 * x₁) + 20 = 0) 
                         ∧ (a^2 * (x₂ - 2) + a * (39 - 20 * x₂) + 20 = 0)) ↔ 
  a = 20 :=
by sorry

end equation_has_roots_l1750_175043


namespace solve_system_of_equations_l1750_175055

theorem solve_system_of_equations (x y : ℝ) (h1 : y^2 + 2 * x * y + x^2 - 6 * y - 6 * x + 5 = 0)
  (h2 : y - x + 1 = x^2 - 3 * x) : 
  ((x = 2 ∧ y = -1) ∨ (x = -1 ∧ y = 2) ∨ (x = -2 ∧ y = 7)) ∧ x ≠ 0 ∧ x ≠ 3 :=
by 
  sorry

end solve_system_of_equations_l1750_175055


namespace bernoulli_inequality_gt_bernoulli_inequality_lt_l1750_175007

theorem bernoulli_inequality_gt (h : ℝ) (x : ℝ) (hx1 : h > -1) (hx2 : x > 1 ∨ x < 0) : (1 + h)^x > 1 + h * x := sorry

theorem bernoulli_inequality_lt (h : ℝ) (x : ℝ) (hx1 : h > -1) (hx2 : 0 < x) (hx3 : x < 1) : (1 + h)^x < 1 + h * x := sorry

end bernoulli_inequality_gt_bernoulli_inequality_lt_l1750_175007


namespace tan_half_angle_sin_cos_expression_l1750_175040

-- Proof Problem 1: If α is an angle in the third quadrant and sin α = -5/13, then tan (α / 2) = -5.
theorem tan_half_angle (α : ℝ) (h1 : Real.sin α = -5/13) (h2 : 3 * π / 2 < α ∧ α < 2 * π) : 
  Real.tan (α / 2) = -5 := 
by 
  sorry

-- Proof Problem 2: If tan α = 2, then sin²(π - α) + 2sin(3π/2 + α)cos(π/2 + α) = 8/5.
theorem sin_cos_expression (α : ℝ) (h : Real.tan α = 2) : 
  Real.sin (π - α) ^ 2 + 2 * Real.sin (3 * π / 2 + α) * Real.cos (π / 2 + α) = 8 / 5 :=
by 
  sorry

end tan_half_angle_sin_cos_expression_l1750_175040


namespace problem1_problem2_problem3_l1750_175035

-- Proof Problem 1
theorem problem1 : -12 - (-18) + (-7) = -1 := 
by {
  sorry
}

-- Proof Problem 2
theorem problem2 : ((4 / 7) - (1 / 9) + (2 / 21)) * (-63) = -35 := 
by {
  sorry
}

-- Proof Problem 3
theorem problem3 : ((-4) ^ 2) / 2 + 9 * (-1 / 3) - abs (3 - 4) = 4 := 
by {
  sorry
}

end problem1_problem2_problem3_l1750_175035


namespace ratio_angela_jacob_l1750_175016

-- Definitions for the conditions
def deans_insects := 30
def jacobs_insects := 5 * deans_insects
def angelas_insects := 75

-- The proof statement proving the ratio
theorem ratio_angela_jacob : angelas_insects / jacobs_insects = 1 / 2 :=
by
  -- Sorry is used here to indicate that the proof is skipped
  sorry

end ratio_angela_jacob_l1750_175016


namespace least_number_to_subtract_l1750_175060

theorem least_number_to_subtract (x : ℕ) (h : 5026 % 5 = x) : x = 1 :=
by sorry

end least_number_to_subtract_l1750_175060


namespace no_three_times_age_ago_l1750_175010

theorem no_three_times_age_ago (F D : ℕ) (h₁ : F = 40) (h₂ : D = 40) (h₃ : F = 2 * D) :
  ¬ ∃ x, F - x = 3 * (D - x) :=
by
  sorry

end no_three_times_age_ago_l1750_175010


namespace product_of_positive_integer_solutions_l1750_175029

theorem product_of_positive_integer_solutions (p : ℕ) (hp : Nat.Prime p) :
  ∀ n : ℕ, (n^2 - 47 * n + 660 = p) → False :=
by
  -- Placeholder for proof, based on the problem conditions.
  sorry

end product_of_positive_integer_solutions_l1750_175029


namespace shirts_sold_l1750_175042

theorem shirts_sold (initial final : ℕ) (h : initial = 49) (h1 : final = 28) : initial - final = 21 :=
sorry

end shirts_sold_l1750_175042


namespace average_age_of_cricket_team_l1750_175053

theorem average_age_of_cricket_team :
  let captain_age := 28
  let ages_sum := 28 + (28 + 4) + (28 - 2) + (28 + 6)
  let remaining_players := 15 - 4
  let total_sum := ages_sum + remaining_players * (A - 1)
  let total_players := 15
  total_sum / total_players = 27.25 := 
by 
  sorry

end average_age_of_cricket_team_l1750_175053


namespace problem1_arithmetic_sequence_problem2_geometric_sequence_l1750_175005

-- Problem (1)
variable (S : Nat → Int)
variable (a : Nat → Int)

axiom S10_eq_50 : S 10 = 50
axiom S20_eq_300 : S 20 = 300
axiom S_def : (∀ n : Nat, n > 0 → S n = n * a 1 + (n * (n-1) / 2) * (a 2 - a 1))

theorem problem1_arithmetic_sequence (n : Nat) : a n = 2 * n - 6 := sorry

-- Problem (2)
variable (a : Nat → Int)

axiom S3_eq_a2_plus_10a1 : S 3 = a 2 + 10 * a 1
axiom a5_eq_81 : a 5 = 81
axiom positive_terms : ∀ n, a n > 0

theorem problem2_geometric_sequence (n : Nat) : S n = (3 ^ n - 1) / 2 := sorry

end problem1_arithmetic_sequence_problem2_geometric_sequence_l1750_175005


namespace probability_of_two_jacob_one_isaac_l1750_175052

-- Definition of the problem conditions
def jacob_letters := 5
def isaac_letters := 5
def total_cards := 12
def cards_drawn := 3

-- Combination function
def C (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Probability calculation
def probability_two_jacob_one_isaac : ℚ :=
  (C jacob_letters 2 * C isaac_letters 1 : ℚ) / (C total_cards cards_drawn : ℚ)

-- The statement of the problem
theorem probability_of_two_jacob_one_isaac :
  probability_two_jacob_one_isaac = 5 / 22 :=
  by sorry

end probability_of_two_jacob_one_isaac_l1750_175052


namespace solution_set_for_inequality_l1750_175045

def f (x : ℝ) : ℝ := x^3 + x

theorem solution_set_for_inequality {a : ℝ} (h : -2 < a ∧ a < 2) :
  f a + f (a^2 - 2) < 0 ↔ -2 < a ∧ a < 0 ∨ 0 < a ∧ a < 1 := sorry

end solution_set_for_inequality_l1750_175045


namespace find_a_l1750_175024

open Complex

theorem find_a (a : ℝ) (h : (2 + Complex.I * a) / (1 + Complex.I * Real.sqrt 2) = -Complex.I * Real.sqrt 2) :
  a = Real.sqrt 2 := by
  sorry

end find_a_l1750_175024


namespace cary_initial_wage_l1750_175085

noncomputable def initial_hourly_wage (x : ℝ) : Prop :=
  let first_year_wage := 1.20 * x
  let second_year_wage := 0.75 * first_year_wage
  second_year_wage = 9

theorem cary_initial_wage : ∃ x : ℝ, initial_hourly_wage x ∧ x = 10 := 
by
  use 10
  unfold initial_hourly_wage
  simp
  sorry

end cary_initial_wage_l1750_175085


namespace tangent_line_through_point_l1750_175012

theorem tangent_line_through_point (x y : ℝ) (h : (x - 2)^2 + y^2 = 1) : 
  (∃ k : ℝ, 15 * x - 8 * y - 13 = 0) ∨ x = 3 := sorry

end tangent_line_through_point_l1750_175012


namespace problem_inequality_l1750_175018

open Real

theorem problem_inequality 
  (p q r x y theta: ℝ) :
  p * x ^ (q - y) + q * x ^ (r - y) + r * x ^ (y - theta)  ≥ p + q + r :=
sorry

end problem_inequality_l1750_175018


namespace fraction_of_5100_l1750_175094

theorem fraction_of_5100 (x : ℝ) (h : ((3 / 4) * x * (2 / 5) * 5100 = 765.0000000000001)) : x = 0.5 :=
by
  sorry

end fraction_of_5100_l1750_175094


namespace point_symmetric_second_quadrant_l1750_175058

theorem point_symmetric_second_quadrant (m : ℝ) 
  (symmetry : ∃ x y : ℝ, P = (-m, m-3) ∧ (-x, -y) = (x, y)) 
  (second_quadrant : ∃ x y : ℝ, P = (-m, m-3) ∧ x < 0 ∧ y > 0) : 
  m < 0 := 
sorry

end point_symmetric_second_quadrant_l1750_175058


namespace geometric_sequence_sum_l1750_175041

def a (n : ℕ) : ℕ := 3 * (2 ^ (n - 1))

theorem geometric_sequence_sum :
  a 1 = 3 → a 4 = 24 → (a 3 + a 4 + a 5) = 84 :=
by
  intros h1 h4
  sorry

end geometric_sequence_sum_l1750_175041


namespace longest_side_of_triangle_l1750_175070

theorem longest_side_of_triangle :
  ∀ (A B C a b : ℝ),
    B = 2 * π / 3 →
    C = π / 6 →
    a = 5 →
    A = π - B - C →
    (b / (Real.sin B) = a / (Real.sin A)) →
    b = 5 * Real.sqrt 3 :=
by
  intros A B C a b hB hC ha hA h_sine_ratio
  sorry

end longest_side_of_triangle_l1750_175070


namespace percentage_of_absent_students_l1750_175061

theorem percentage_of_absent_students (total_students boys girls : ℕ) (fraction_boys_absent fraction_girls_absent : ℚ)
  (total_students_eq : total_students = 180)
  (boys_eq : boys = 120)
  (girls_eq : girls = 60)
  (fraction_boys_absent_eq : fraction_boys_absent = 1/6)
  (fraction_girls_absent_eq : fraction_girls_absent = 1/4) :
  let boys_absent := fraction_boys_absent * boys
  let girls_absent := fraction_girls_absent * girls
  let total_absent := boys_absent + girls_absent
  let absent_percentage := (total_absent / total_students) * 100
  abs (absent_percentage - 19) < 1 :=
by
  sorry

end percentage_of_absent_students_l1750_175061


namespace probability_cs_majors_consecutive_l1750_175069

def total_ways_to_choose_5_out_of_12 : ℕ :=
  Nat.choose 12 5

def number_of_ways_cs_majors_consecutive : ℕ :=
  12

theorem probability_cs_majors_consecutive :
  (number_of_ways_cs_majors_consecutive : ℚ) / (total_ways_to_choose_5_out_of_12 : ℚ) = 1 / 66 := by
  sorry

end probability_cs_majors_consecutive_l1750_175069


namespace Vitya_catches_mother_l1750_175015

theorem Vitya_catches_mother (s : ℕ) : 
    let distance := 20 * s
    let relative_speed := 4 * s
    let time := distance / relative_speed
    time = 5 :=
by
  sorry

end Vitya_catches_mother_l1750_175015


namespace min_k_value_l1750_175065

-- Definition of the problem's conditions
def remainder_condition (n k : ℕ) : Prop :=
  ∀ i, 2 ≤ i → i ≤ k → n % i = i - 1

def in_range (x a b : ℕ) : Prop :=
  a < x ∧ x < b

-- The statement of the proof problem in Lean 4
theorem min_k_value (n k : ℕ) (h1 : remainder_condition n k) (hn_range : in_range n 2000 3000) :
  k = 9 :=
sorry

end min_k_value_l1750_175065


namespace percentage_return_l1750_175054

theorem percentage_return (income investment : ℝ) (h_income : income = 680) (h_investment : investment = 8160) :
  (income / investment) * 100 = 8.33 :=
by
  rw [h_income, h_investment]
  -- The rest of the proof is omitted.
  sorry

end percentage_return_l1750_175054


namespace plane_intersects_unit_cubes_l1750_175014

def unitCubeCount (side_length : ℕ) : ℕ :=
  side_length ^ 3

def intersectionCount (num_unitCubes : ℕ) (side_length : ℕ) : ℕ :=
  if side_length = 4 then 32 else 0 -- intersection count only applies for side_length = 4

theorem plane_intersects_unit_cubes
  (side_length : ℕ)
  (num_unitCubes : ℕ)
  (cubeArrangement : num_unitCubes = unitCubeCount side_length)
  (planeCondition : True) -- the plane is perpendicular to the diagonal and bisects it
  : intersectionCount num_unitCubes side_length = 32 := by
  sorry

end plane_intersects_unit_cubes_l1750_175014


namespace fraction_cubed_sum_l1750_175021

theorem fraction_cubed_sum (x y : ℤ) (h1 : x = 3) (h2 : y = 4) :
  (x^3 + 3 * y^3) / 7 = 31 + 3 / 7 := by
  sorry

end fraction_cubed_sum_l1750_175021


namespace domain_of_lg_abs_x_minus_1_l1750_175030

theorem domain_of_lg_abs_x_minus_1 (x : ℝ) : 
  (|x| - 1 > 0) ↔ (x < -1 ∨ x > 1) := 
by
  sorry

end domain_of_lg_abs_x_minus_1_l1750_175030


namespace sin_beta_value_l1750_175023

open Real

theorem sin_beta_value (α β : ℝ) 
  (hα : 0 < α ∧ α < π / 2) 
  (hβ : 0 < β ∧ β < π / 2) 
  (h1 : sin α = 5 / 13) 
  (h2 : cos (α + β) = -4 / 5) : 
  sin β = 56 / 65 := 
sorry

end sin_beta_value_l1750_175023


namespace real_solutions_equation_l1750_175093

theorem real_solutions_equation :
  ∃! x : ℝ, 9 * x^2 - 90 * ⌊ x ⌋ + 99 = 0 :=
sorry

end real_solutions_equation_l1750_175093


namespace jess_remaining_blocks_l1750_175077

-- Define the number of blocks for each segment of Jess's errands
def blocks_to_post_office : Nat := 24
def blocks_to_store : Nat := 18
def blocks_to_gallery : Nat := 15
def blocks_to_library : Nat := 14
def blocks_to_work : Nat := 22
def blocks_already_walked : Nat := 9

-- Calculate the total blocks to be walked
def total_blocks : Nat :=
  blocks_to_post_office + blocks_to_store + blocks_to_gallery + blocks_to_library + blocks_to_work

-- The remaining blocks Jess needs to walk
def blocks_remaining : Nat :=
  total_blocks - blocks_already_walked

-- The statement to be proved
theorem jess_remaining_blocks : blocks_remaining = 84 :=
by
  sorry

end jess_remaining_blocks_l1750_175077


namespace sum_of_first_50_primes_is_5356_l1750_175017

open Nat

-- Define the first 50 prime numbers
def first_50_primes : List Nat := 
  [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 
   83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 
   179, 181, 191, 193, 197, 199, 211, 223, 227, 229]

-- Calculate their sum
def sum_first_50_primes : Nat := List.foldr (Nat.add) 0 first_50_primes

-- Now we state the theorem we want to prove
theorem sum_of_first_50_primes_is_5356 : 
  sum_first_50_primes = 5356 := 
by
  -- Placeholder for proof
  sorry

end sum_of_first_50_primes_is_5356_l1750_175017


namespace base_equivalence_l1750_175026

theorem base_equivalence : 
  ∀ (b : ℕ), (b^3 + 3*b^2 + 4)^2 = 9*b^4 + 9*b^3 + 2*b^2 + 2*b + 5 ↔ b = 10 := 
by
  sorry

end base_equivalence_l1750_175026


namespace family_percentage_eaten_after_dinner_l1750_175090

theorem family_percentage_eaten_after_dinner
  (total_brownies : ℕ)
  (children_percentage : ℚ)
  (left_over_brownies : ℕ)
  (lorraine_extra_brownie : ℕ)
  (remaining_percentage : ℚ) :
  total_brownies = 16 →
  children_percentage = 0.25 →
  lorraine_extra_brownie = 1 →
  left_over_brownies = 5 →
  remaining_percentage = 50 := by
  sorry

end family_percentage_eaten_after_dinner_l1750_175090


namespace ratio_of_side_lengths_sum_l1750_175050

theorem ratio_of_side_lengths_sum (a b c : ℕ) (ha : a = 4) (hb : b = 15) (hc : c = 25) :
  a + b + c = 44 := 
by
  sorry

end ratio_of_side_lengths_sum_l1750_175050


namespace molecular_weight_K2Cr2O7_l1750_175082

/--
K2Cr2O7 consists of:
- 2 K atoms
- 2 Cr atoms
- 7 O atoms

Atomic weights:
- K: 39.10 g/mol
- Cr: 52.00 g/mol
- O: 16.00 g/mol

We need to prove that the molecular weight of 4 moles of K2Cr2O7 is 1176.80 g/mol.
-/
theorem molecular_weight_K2Cr2O7 :
  let weight_K := 39.10
  let weight_Cr := 52.00
  let weight_O := 16.00
  let mol_weight_K2Cr2O7 := (2 * weight_K) + (2 * weight_Cr) + (7 * weight_O)
  (4 * mol_weight_K2Cr2O7) = 1176.80 :=
by
  sorry

end molecular_weight_K2Cr2O7_l1750_175082


namespace rewrite_expression_l1750_175084

theorem rewrite_expression (k : ℝ) :
  ∃ d r s : ℝ, (8 * k^2 - 12 * k + 20 = d * (k + r)^2 + s) ∧ (r + s = 14.75) := 
sorry

end rewrite_expression_l1750_175084


namespace books_per_shelf_l1750_175002

theorem books_per_shelf (total_distance : ℕ) (total_shelves : ℕ) (one_way_distance : ℕ) 
  (h1 : total_distance = 3200) (h2 : total_shelves = 4) (h3 : one_way_distance = total_distance / 2) 
  (h4 : one_way_distance = 1600) :
  ∀ books_per_shelf : ℕ, books_per_shelf = one_way_distance / total_shelves := 
by
  sorry

end books_per_shelf_l1750_175002


namespace negation_of_exists_l1750_175001

theorem negation_of_exists (p : Prop) :
  (∃ x : ℝ, x^2 + 2 * x < 0) ↔ ¬ (∀ x : ℝ, x^2 + 2 * x >= 0) :=
sorry

end negation_of_exists_l1750_175001


namespace solve_problem_l1750_175003

open Nat

theorem solve_problem :
  ∃ (n p : ℕ), p.Prime ∧ n > 0 ∧ ∃ k : ℤ, p^2 + 7^n = k^2 ∧ (n, p) = (1, 3) := 
by
  sorry

end solve_problem_l1750_175003


namespace f_odd_function_l1750_175047

noncomputable def f : ℝ → ℝ := sorry

axiom f_additive (a b : ℝ) : f (a + b) = f a + f b

theorem f_odd_function : ∀ x : ℝ, f (-x) = -f x := by
  intro x
  sorry

end f_odd_function_l1750_175047


namespace student_courses_last_year_l1750_175057

variable (x : ℕ)
variable (courses_last_year : ℕ := x)
variable (avg_grade_last_year : ℕ := 100)
variable (courses_year_before : ℕ := 5)
variable (avg_grade_year_before : ℕ := 60)
variable (avg_grade_two_years : ℕ := 81)

theorem student_courses_last_year (h1 : avg_grade_last_year = 100)
                                   (h2 : courses_year_before = 5)
                                   (h3 : avg_grade_year_before = 60)
                                   (h4 : avg_grade_two_years = 81)
                                   (hc : ((5 * avg_grade_year_before) + (courses_last_year * avg_grade_last_year)) / (courses_year_before + courses_last_year) = avg_grade_two_years) :
                                   courses_last_year = 6 := by
  sorry

end student_courses_last_year_l1750_175057


namespace custom_op_neg2_neg3_l1750_175056

  def custom_op (a b : ℤ) : ℤ := b^2 - a

  theorem custom_op_neg2_neg3 : custom_op (-2) (-3) = 11 :=
  by
    sorry
  
end custom_op_neg2_neg3_l1750_175056


namespace retail_women_in_LA_l1750_175037

/-
Los Angeles has 6 million people living in it. If half the population is women 
and 1/3 of the women work in retail, how many women work in retail in Los Angeles?
-/

theorem retail_women_in_LA 
  (total_population : ℕ)
  (half_population_women : total_population / 2 = women_population)
  (third_women_retail : women_population / 3 = retail_women)
  : total_population = 6000000 → retail_women = 1000000 :=
by
  sorry

end retail_women_in_LA_l1750_175037


namespace range_of_c_l1750_175099

def p (c : ℝ) := (0 < c) ∧ (c < 1)
def q (c : ℝ) := (1 - 2 * c < 0)

theorem range_of_c (c : ℝ) : (p c ∨ q c) ∧ ¬ (p c ∧ q c) ↔ (0 < c ∧ c ≤ 1/2) ∨ (1 < c) :=
by sorry

end range_of_c_l1750_175099


namespace maryville_population_increase_l1750_175071

def average_people_added_per_year (P2000 P2005 : ℕ) (period : ℕ) : ℕ :=
  (P2005 - P2000) / period
  
theorem maryville_population_increase :
  let P2000 := 450000
  let P2005 := 467000
  let period := 5
  average_people_added_per_year P2000 P2005 period = 3400 :=
by
  sorry

end maryville_population_increase_l1750_175071


namespace min_y_value_l1750_175078

noncomputable def y (a x : ℝ) : ℝ := (Real.exp x - a)^2 + (Real.exp (-x) - a)^2

theorem min_y_value (a : ℝ) (h : a ≠ 0) : 
  (a ≥ 2 → ∃ x, y a x = a^2 - 2) ∧ (a < 2 → ∃ x, y a x = 2*(a-1)^2) :=
sorry

end min_y_value_l1750_175078


namespace weak_multiple_l1750_175019

def is_weak (a b n : ℕ) : Prop :=
  ∀ (x y : ℕ), n ≠ a * x + b * y

theorem weak_multiple (a b n : ℕ) (h_coprime : Nat.gcd a b = 1) (h_weak : is_weak a b n) (h_bound : n < a * b / 6) : 
  ∃ k ≥ 2, is_weak a b (k * n) :=
by
  sorry

end weak_multiple_l1750_175019


namespace radius_first_field_l1750_175095

theorem radius_first_field (r_2 : ℝ) (h_r2 : r_2 = 10) (h_area : ∃ A_2, ∃ A_1, A_1 = 0.09 * A_2 ∧ A_2 = π * r_2^2) : ∃ r_1 : ℝ, r_1 = 3 :=
by
  sorry

end radius_first_field_l1750_175095


namespace sarahs_total_problems_l1750_175079

def math_pages : ℕ := 4
def reading_pages : ℕ := 6
def science_pages : ℕ := 5
def math_problems_per_page : ℕ := 4
def reading_problems_per_page : ℕ := 4
def science_problems_per_page : ℕ := 6

def total_math_problems : ℕ := math_pages * math_problems_per_page
def total_reading_problems : ℕ := reading_pages * reading_problems_per_page
def total_science_problems : ℕ := science_pages * science_problems_per_page

def total_problems : ℕ := total_math_problems + total_reading_problems + total_science_problems

theorem sarahs_total_problems :
  total_problems = 70 :=
by
  -- proof will be inserted here
  sorry

end sarahs_total_problems_l1750_175079


namespace jordan_buys_rice_l1750_175086

variables (r l : ℝ)

theorem jordan_buys_rice
  (price_rice : ℝ := 1.20)
  (price_lentils : ℝ := 0.60)
  (total_pounds : ℝ := 30)
  (total_cost : ℝ := 27.00)
  (eq1 : r + l = total_pounds)
  (eq2 : price_rice * r + price_lentils * l = total_cost) :
  r = 15.0 :=
by
  sorry

end jordan_buys_rice_l1750_175086


namespace ammonium_iodide_required_l1750_175073

theorem ammonium_iodide_required
  (KOH_moles NH3_moles KI_moles H2O_moles : ℕ)
  (hn : NH3_moles = 3) (hk : KOH_moles = 3) (hi : KI_moles = 3) (hw : H2O_moles = 3) :
  ∃ NH4I_moles, NH3_moles = 3 ∧ KI_moles = 3 ∧ H2O_moles = 3 ∧ KOH_moles = 3 ∧ NH4I_moles = 3 :=
by
  sorry

end ammonium_iodide_required_l1750_175073


namespace keys_missing_l1750_175089

theorem keys_missing (vowels := 5) (consonants := 21)
  (missing_consonants := consonants / 7) (missing_vowels := 2) :
  missing_consonants + missing_vowels = 5 := by
  sorry

end keys_missing_l1750_175089


namespace Nancy_antacid_consumption_l1750_175025

theorem Nancy_antacid_consumption :
  let antacids_per_month : ℕ :=
    let antacids_per_day_indian := 3
    let antacids_per_day_mexican := 2
    let antacids_per_day_other := 1
    let days_indian_per_week := 3
    let days_mexican_per_week := 2
    let days_total_per_week := 7
    let weeks_per_month := 4

    let antacids_per_week_indian := antacids_per_day_indian * days_indian_per_week
    let antacids_per_week_mexican := antacids_per_day_mexican * days_mexican_per_week
    let days_other_per_week := days_total_per_week - days_indian_per_week - days_mexican_per_week
    let antacids_per_week_other := antacids_per_day_other * days_other_per_week

    let antacids_per_week_total := antacids_per_week_indian + antacids_per_week_mexican + antacids_per_week_other

    antacids_per_week_total * weeks_per_month
    
  antacids_per_month = 60 := sorry

end Nancy_antacid_consumption_l1750_175025


namespace find_w_l1750_175096

variables (w x y z : ℕ)

-- conditions
def condition1 : Prop := x = w / 2
def condition2 : Prop := y = w + x
def condition3 : Prop := z = 400
def condition4 : Prop := w + x + y + z = 1000

-- problem to prove
theorem find_w (h1 : condition1 w x) (h2 : condition2 w x y) (h3 : condition3 z) (h4 : condition4 w x y z) : w = 200 :=
by sorry

end find_w_l1750_175096


namespace passengers_got_on_in_Texas_l1750_175062

theorem passengers_got_on_in_Texas (start_pax : ℕ) 
  (texas_depart_pax : ℕ) 
  (nc_depart_pax : ℕ) 
  (nc_board_pax : ℕ) 
  (virginia_total_people : ℕ) 
  (crew_members : ℕ) 
  (final_pax_virginia : ℕ) 
  (X : ℕ) :
  start_pax = 124 →
  texas_depart_pax = 58 →
  nc_depart_pax = 47 →
  nc_board_pax = 14 →
  virginia_total_people = 67 →
  crew_members = 10 →
  final_pax_virginia = virginia_total_people - crew_members →
  X + 33 = final_pax_virginia →
  X = 24 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end passengers_got_on_in_Texas_l1750_175062


namespace dampening_factor_l1750_175022

theorem dampening_factor (s r : ℝ) 
  (h1 : s / (1 - r) = 16) 
  (h2 : s * r / (1 - r^2) = -6) :
  r = -3 / 11 := 
sorry

end dampening_factor_l1750_175022


namespace sum_of_squares_twice_square_sum_of_fourth_powers_twice_fourth_power_l1750_175028

-- Definitions
def a (t : ℤ) := 4 * t
def b (t : ℤ) := 3 - 2 * t - t^2
def c (t : ℤ) := 3 + 2 * t - t^2

-- Theorem for sum of squares
theorem sum_of_squares_twice_square (t : ℤ) : 
  a t ^ 2 + b t ^ 2 + c t ^ 2 = 2 * ((3 + t^2) ^ 2) :=
by 
  sorry

-- Theorem for sum of fourth powers
theorem sum_of_fourth_powers_twice_fourth_power (t : ℤ) : 
  a t ^ 4 + b t ^ 4 + c t ^ 4 = 2 * ((3 + t^2) ^ 4) :=
by 
  sorry

end sum_of_squares_twice_square_sum_of_fourth_powers_twice_fourth_power_l1750_175028


namespace rectangle_length_l1750_175068

theorem rectangle_length (P W : ℝ) (hP : P = 30) (hW : W = 10) :
  ∃ (L : ℝ), 2 * (L + W) = P ∧ L = 5 :=
by
  sorry

end rectangle_length_l1750_175068


namespace infinite_primes_dividing_S_l1750_175049

noncomputable def infinite_set_of_pos_integers (S : Set ℕ) : Prop :=
  (∀ n : ℕ, ∃ m : ℕ, m > n ∧ m ∈ S) ∧ ∀ n : ℕ, n ∈ S → n > 0

def set_of_sums (S : Set ℕ) : Set ℕ :=
  {t | ∃ x y, x ∈ S ∧ y ∈ S ∧ x ≠ y ∧ t = x + y}

noncomputable def finitely_many_primes_condition (S : Set ℕ) (T : Set ℕ) : Prop :=
  {p : ℕ | Prime p ∧ p % 4 = 1 ∧ (∃ t ∈ T, p ∣ t)}.Finite

theorem infinite_primes_dividing_S (S : Set ℕ) (T := set_of_sums S)
  (hS : infinite_set_of_pos_integers S)
  (hT : finitely_many_primes_condition S T) :
  {p : ℕ | Prime p ∧ ∃ s ∈ S, p ∣ s}.Infinite := 
sorry

end infinite_primes_dividing_S_l1750_175049


namespace carnations_count_l1750_175083

-- Define the conditions 
def vase_capacity : Nat := 9
def number_of_vases : Nat := 3
def number_of_roses : Nat := 23
def total_flowers : Nat := number_of_vases * vase_capacity

-- Define the number of carnations
def number_of_carnations : Nat := total_flowers - number_of_roses

-- Assertion that should be proved
theorem carnations_count : number_of_carnations = 4 := by
  sorry

end carnations_count_l1750_175083


namespace correct_equation_l1750_175092

def initial_investment : ℝ := 2500
def expected_investment : ℝ := 6600
def growth_rate (x : ℝ) : ℝ := x

theorem correct_equation (x : ℝ) : 
  initial_investment * (1 + growth_rate x) + initial_investment * (1 + growth_rate x)^2 = expected_investment :=
by
  sorry

end correct_equation_l1750_175092


namespace find_down_payment_l1750_175059

noncomputable def purchasePrice : ℝ := 118
noncomputable def monthlyPayment : ℝ := 10
noncomputable def numberOfMonths : ℝ := 12
noncomputable def interestRate : ℝ := 0.15254237288135593
noncomputable def totalPayments : ℝ := numberOfMonths * monthlyPayment -- total amount paid through installments
noncomputable def interestPaid : ℝ := purchasePrice * interestRate -- total interest paid
noncomputable def totalPaid : ℝ := purchasePrice + interestPaid -- total amount paid including interest

theorem find_down_payment : ∃ D : ℝ, D + totalPayments = totalPaid ∧ D = 16 :=
by sorry

end find_down_payment_l1750_175059


namespace height_of_stack_of_pots_l1750_175076

-- Definitions corresponding to problem conditions
def pot_thickness : ℕ := 1

def top_pot_diameter : ℕ := 16

def bottom_pot_diameter : ℕ := 4

def diameter_decrement : ℕ := 2

-- Number of pots calculation
def num_pots : ℕ := (top_pot_diameter - bottom_pot_diameter) / diameter_decrement + 1

-- The total vertical distance from the bottom of the lowest pot to the top of the highest pot
def total_vertical_distance : ℕ := 
  let inner_heights := num_pots * (top_pot_diameter - pot_thickness + bottom_pot_diameter - pot_thickness) / 2
  let total_thickness := num_pots * pot_thickness
  inner_heights + total_thickness

theorem height_of_stack_of_pots : total_vertical_distance = 65 := 
sorry

end height_of_stack_of_pots_l1750_175076


namespace largest_vertex_sum_of_parabola_l1750_175067

theorem largest_vertex_sum_of_parabola 
  (a T : ℤ)
  (hT : T ≠ 0)
  (h1 : 0 = a * 0^2 + b * 0 + c)
  (h2 : 0 = a * (2 * T) ^ 2 + b * (2 * T) + c)
  (h3 : 36 = a * (2 * T + 2) ^ 2 + b * (2 * T + 2) + c) :
  ∃ N : ℚ, N = -5 / 4 :=
sorry

end largest_vertex_sum_of_parabola_l1750_175067


namespace find_ages_l1750_175074

theorem find_ages (x : ℕ) (h : (x + 1) * (x + 1) = x * x + 5) : x = 2 := 
sorry

end find_ages_l1750_175074


namespace ratio_area_of_rectangle_to_square_l1750_175081

theorem ratio_area_of_rectangle_to_square (s : ℝ) :
  (1.2 * s * 0.8 * s) / (s * s) = 24 / 25 :=
by
  sorry

end ratio_area_of_rectangle_to_square_l1750_175081


namespace part1_part2_l1750_175032

def f (x : ℝ) (a : ℝ) : ℝ := |2 * x - 1| + |2 * x - a|

theorem part1 (x : ℝ) : (f x 2 < 2) ↔ (1/4 < x ∧ x < 5/4) := by
  sorry
  
theorem part2 (a : ℝ) (hx : ∀ x : ℝ, f x a ≥ 3 * a + 2) :
  (-3/2 ≤ a ∧ a ≤ -1/4) := by
  sorry

end part1_part2_l1750_175032


namespace arithmetic_sequence_a7_l1750_175013

theorem arithmetic_sequence_a7 (S_13 : ℕ → ℕ → ℕ) (n : ℕ) (a7 : ℕ) (h1: S_13 13 52 = 52) (h2: S_13 13 a7 = 13 * a7):
  a7 = 4 :=
by
  sorry

end arithmetic_sequence_a7_l1750_175013


namespace marigolds_sold_second_day_l1750_175048

theorem marigolds_sold_second_day (x : ℕ) (h1 : 14 ≤ x)
  (h2 : 2 * x + 14 + x = 89) : x = 25 :=
by
  sorry

end marigolds_sold_second_day_l1750_175048


namespace probability_without_replacement_probability_with_replacement_l1750_175039

-- Definition for without replacement context
def without_replacement_total_outcomes : ℕ := 6
def without_replacement_favorable_outcomes : ℕ := 3
def without_replacement_prob : ℚ :=
  without_replacement_favorable_outcomes / without_replacement_total_outcomes

-- Theorem stating that the probability of selecting two consecutive integers without replacement is 1/2
theorem probability_without_replacement : 
  without_replacement_prob = 1 / 2 := by
  sorry

-- Definition for with replacement context
def with_replacement_total_outcomes : ℕ := 16
def with_replacement_favorable_outcomes : ℕ := 3
def with_replacement_prob : ℚ :=
  with_replacement_favorable_outcomes / with_replacement_total_outcomes

-- Theorem stating that the probability of selecting two consecutive integers with replacement is 3/16
theorem probability_with_replacement : 
  with_replacement_prob = 3 / 16 := by
  sorry

end probability_without_replacement_probability_with_replacement_l1750_175039


namespace find_difference_l1750_175075

-- Define the problem conditions in Lean
theorem find_difference (a b : ℕ) (hrelprime : Nat.gcd a b = 1)
                        (hpos : a > b) 
                        (hfrac : (a^3 - b^3) / (a - b)^3 = 73 / 3) :
    a - b = 3 :=
by
    sorry

end find_difference_l1750_175075


namespace domain_of_expression_l1750_175044

theorem domain_of_expression (x : ℝ) 
  (h1 : 3 * x - 6 ≥ 0) 
  (h2 : 7 - 2 * x ≥ 0) 
  (h3 : 7 - 2 * x > 0) : 
  2 ≤ x ∧ x < 7 / 2 := by
sorry

end domain_of_expression_l1750_175044


namespace hari_contribution_l1750_175066

theorem hari_contribution 
    (P_investment : ℕ) (P_time : ℕ) (H_time : ℕ) (profit_ratio : ℚ)
    (investment_ratio : P_investment * P_time / (Hari_contribution * H_time) = profit_ratio) :
    Hari_contribution = 10080 :=
by
    have P_investment := 3920
    have P_time := 12
    have H_time := 7
    have profit_ratio := (2 : ℚ) / 3
    sorry

end hari_contribution_l1750_175066


namespace tens_digit_23_1987_l1750_175072

theorem tens_digit_23_1987 : (23 ^ 1987 % 100) / 10 % 10 = 4 :=
by
  -- The proof goes here
  sorry

end tens_digit_23_1987_l1750_175072


namespace compute_expr_l1750_175038

theorem compute_expr :
  ((π - 3.14)^0 + (-0.125)^2008 * 8^2008) = 2 := 
by 
  sorry

end compute_expr_l1750_175038


namespace decreasing_interval_l1750_175046

noncomputable def f (x : ℝ) : ℝ := x^3 + 3 * x^2 + 2

theorem decreasing_interval : ∀ x : ℝ, (-2 < x ∧ x < 0) → (deriv f x < 0) := 
by
  sorry

end decreasing_interval_l1750_175046


namespace fraction_of_original_water_after_four_replacements_l1750_175008

-- Define the initial condition and process
def initial_water_volume : ℚ := 10
def initial_alcohol_volume : ℚ := 10
def initial_total_volume : ℚ := initial_water_volume + initial_alcohol_volume

def fraction_remaining_after_removal (fraction_remaining : ℚ) : ℚ :=
  fraction_remaining * (initial_total_volume - 5) / initial_total_volume

-- Define the function counting the iterations process
def fraction_after_replacements (n : ℕ) (fraction_remaining : ℚ) : ℚ :=
  Nat.iterate fraction_remaining_after_removal n fraction_remaining

-- We have 4 replacements, start with 1 (because initially half of tank is water, 
-- fraction is 1 means we start with all original water)
def fraction_of_original_water_remaining : ℚ := (fraction_after_replacements 4 1)

-- Our goal in proof form
theorem fraction_of_original_water_after_four_replacements :
  fraction_of_original_water_remaining = (81 / 256) := by
  sorry

end fraction_of_original_water_after_four_replacements_l1750_175008


namespace total_people_going_to_zoo_l1750_175027

def cars : ℝ := 3.0
def people_per_car : ℝ := 63.0

theorem total_people_going_to_zoo : cars * people_per_car = 189.0 :=
by 
  sorry

end total_people_going_to_zoo_l1750_175027


namespace how_many_integers_satisfy_l1750_175004

theorem how_many_integers_satisfy {n : ℤ} : ((n - 3) * (n + 5) < 0) ↔ (n = -4 ∨ n = -3 ∨ n = -2 ∨ n = -1 ∨ n = 0 ∨ n = 1 ∨ n = 2) := sorry

end how_many_integers_satisfy_l1750_175004


namespace supplementary_angle_l1750_175088

theorem supplementary_angle {α β : ℝ} (angle_supplementary : α + β = 180) (angle_1_eq : α = 80) : β = 100 :=
by
  sorry

end supplementary_angle_l1750_175088


namespace principal_sum_l1750_175011

theorem principal_sum (A1 A2 : ℝ) (I P : ℝ) 
  (hA1 : A1 = 1717) 
  (hA2 : A2 = 1734) 
  (hI : I = A2 - A1)
  (h_simple_interest : A1 = P + I) : P = 1700 :=
by
  sorry

end principal_sum_l1750_175011


namespace sum_of_midpoints_l1750_175020

theorem sum_of_midpoints {a b c : ℝ} (h : a + b + c = 10) :
  (a + b) / 2 + (a + c) / 2 + (b + c) / 2 = 10 :=
by
  sorry

end sum_of_midpoints_l1750_175020


namespace problem_l1750_175006

variable {x y : ℝ}

theorem problem (h1 : (x + y)^2 = 81) (h2 : x * y = 10) : (x - y)^2 = 41 := 
by
  sorry

end problem_l1750_175006


namespace P_investment_calculation_l1750_175087

variable {P_investment : ℝ}
variable (Q_investment : ℝ := 36000)
variable (total_profit : ℝ := 18000)
variable (Q_profit : ℝ := 6001.89)

def P_profit : ℝ := total_profit - Q_profit

theorem P_investment_calculation :
  P_investment = (P_profit * Q_investment) / Q_profit :=
by
  sorry

end P_investment_calculation_l1750_175087


namespace total_pigs_correct_l1750_175098

def initial_pigs : Float := 64.0
def incoming_pigs : Float := 86.0
def total_pigs : Float := 150.0

theorem total_pigs_correct : initial_pigs + incoming_pigs = total_pigs := by 
  sorry

end total_pigs_correct_l1750_175098


namespace true_converses_count_l1750_175051

-- Definitions according to the conditions
def parallel_lines (L1 L2 : Prop) : Prop := L1 ↔ L2
def congruent_triangles (T1 T2 : Prop) : Prop := T1 ↔ T2
def vertical_angles (A1 A2 : Prop) : Prop := A1 = A2
def squares_equal (m n : ℝ) : Prop := m = n → (m^2 = n^2)

-- Propositions with their converses
def converse_parallel (L1 L2 : Prop) : Prop := parallel_lines L1 L2 → parallel_lines L2 L1
def converse_congruent (T1 T2 : Prop) : Prop := congruent_triangles T1 T2 → congruent_triangles T2 T1
def converse_vertical (A1 A2 : Prop) : Prop := vertical_angles A1 A2 → vertical_angles A2 A1
def converse_squares (m n : ℝ) : Prop := (m^2 = n^2) → (m = n)

-- Proving the number of true converses
theorem true_converses_count : 
  (∃ L1 L2, converse_parallel L1 L2) →
  (∃ T1 T2, ¬converse_congruent T1 T2) →
  (∃ A1 A2, converse_vertical A1 A2) →
  (∃ m n : ℝ, ¬converse_squares m n) →
  (2 = 2) := by
  intros _ _ _ _
  sorry

end true_converses_count_l1750_175051


namespace smallest_k_condition_exists_l1750_175000

theorem smallest_k_condition_exists (k : ℕ) :
    k > 1 ∧ (k % 13 = 1) ∧ (k % 8 = 1) ∧ (k % 3 = 1) → k = 313 :=
by
  sorry

end smallest_k_condition_exists_l1750_175000


namespace johnny_hourly_wage_l1750_175080

-- Definitions based on conditions
def hours_worked : ℕ := 6
def total_earnings : ℝ := 28.5

-- Theorem statement
theorem johnny_hourly_wage : total_earnings / hours_worked = 4.75 :=
by
  sorry

end johnny_hourly_wage_l1750_175080


namespace roberto_valid_outfits_l1750_175097

-- Definitions based on the conditions
def total_trousers : ℕ := 6
def total_shirts : ℕ := 8
def total_jackets : ℕ := 4
def restricted_jacket : ℕ := 1
def restricted_shirts : ℕ := 2

-- Theorem statement
theorem roberto_valid_outfits : 
  total_trousers * total_shirts * total_jackets - total_trousers * restricted_shirts * restricted_jacket = 180 := 
by
  sorry

end roberto_valid_outfits_l1750_175097


namespace fractions_integer_or_fractional_distinct_l1750_175091

theorem fractions_integer_or_fractional_distinct (a b : Fin 6 → ℕ) (h_pos : ∀ i, 0 < a i ∧ 0 < b i)
  (h_irreducible : ∀ i, Nat.gcd (a i) (b i) = 1)
  (h_sum_eq : (Finset.univ : Finset (Fin 6)).sum a = (Finset.univ : Finset (Fin 6)).sum b) :
  ¬ ∀ i j : Fin 6, i ≠ j → ((a i / b i = a j / b j) ∨ (a i % b i / b i = a j % b j / b j)) :=
sorry

end fractions_integer_or_fractional_distinct_l1750_175091


namespace vertex_on_x_axis_l1750_175009

theorem vertex_on_x_axis (d : ℝ) : 
  (∃ x : ℝ, x^2 - 6 * x + d = 0) ↔ d = 9 :=
by
  sorry

end vertex_on_x_axis_l1750_175009


namespace new_person_weight_l1750_175034

theorem new_person_weight (avg_weight_increase : ℝ) (old_weight new_weight : ℝ) (n : ℕ)
    (weight_increase_per_person : avg_weight_increase = 3.5)
    (number_of_persons : n = 8)
    (replaced_person_weight : old_weight = 62) :
    new_weight = 90 :=
by
  sorry

end new_person_weight_l1750_175034


namespace correct_calculation_l1750_175031

variable {a b : ℝ}

theorem correct_calculation : 
  (2 * a^3 + 2 * a ≠ 2 * a^4) ∧
  ((a - 2 * b)^2 ≠ a^2 - 4 * b^2) ∧
  (-5 * (2 * a - b) ≠ -10 * a - 5 * b) ∧
  ((-2 * a^2 * b)^3 = -8 * a^6 * b^3) :=
by
  sorry

end correct_calculation_l1750_175031


namespace combined_annual_income_after_expenses_l1750_175064

noncomputable def brady_monthly_incomes : List ℕ := [150, 200, 250, 300, 200, 150, 180, 220, 240, 270, 300, 350]
noncomputable def dwayne_monthly_incomes : List ℕ := [100, 150, 200, 250, 150, 120, 140, 190, 180, 230, 260, 300]
def brady_annual_expense : ℕ := 450
def dwayne_annual_expense : ℕ := 300

def annual_income (monthly_incomes : List ℕ) : ℕ :=
  monthly_incomes.foldr (· + ·) 0

theorem combined_annual_income_after_expenses :
  (annual_income brady_monthly_incomes - brady_annual_expense) +
  (annual_income dwayne_monthly_incomes - dwayne_annual_expense) = 3930 :=
by
  sorry

end combined_annual_income_after_expenses_l1750_175064


namespace quadratic_relationship_l1750_175036

theorem quadratic_relationship :
  ∀ (x z : ℕ), (x = 1 ∧ z = 5) ∨ (x = 2 ∧ z = 12) ∨ (x = 3 ∧ z = 23) ∨ (x = 4 ∧ z = 38) ∨ (x = 5 ∧ z = 57) →
  z = 2 * x^2 + x + 2 :=
by
  sorry

end quadratic_relationship_l1750_175036


namespace a_8_value_l1750_175063

variable {n : ℕ}
def S (n : ℕ) : ℕ := n^2
def a (n : ℕ) : ℕ := S n - S (n - 1)

theorem a_8_value : a 8 = 15 := by
  sorry

end a_8_value_l1750_175063


namespace mary_more_than_marco_l1750_175033

def marco_initial : ℕ := 24
def mary_initial : ℕ := 15
def half_marco : ℕ := marco_initial / 2
def mary_after_give : ℕ := mary_initial + half_marco
def mary_after_spend : ℕ := mary_after_give - 5
def marco_final : ℕ := marco_initial - half_marco

theorem mary_more_than_marco :
  mary_after_spend - marco_final = 10 := by
  sorry

end mary_more_than_marco_l1750_175033
