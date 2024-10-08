import Mathlib

namespace dividend_calculation_l86_86079

theorem dividend_calculation 
  (D : ℝ) (Q : ℕ) (R : ℕ) 
  (hD : D = 164.98876404494382)
  (hQ : Q = 89)
  (hR : R = 14) :
  ⌈D * Q + R⌉ = 14698 :=
sorry

end dividend_calculation_l86_86079


namespace girls_more_than_boys_l86_86166

/-- 
In a class with 42 students, where the ratio of boys to girls is 3:4, 
prove that there are 6 more girls than boys.
-/
theorem girls_more_than_boys (students total_students : ℕ) (boys girls : ℕ) (ratio_boys_girls : 3 * girls = 4 * boys)
  (total_students_count : boys + girls = total_students)
  (total_students_value : total_students = 42) : girls - boys = 6 :=
by
  sorry

end girls_more_than_boys_l86_86166


namespace largest_integer_is_190_l86_86925

theorem largest_integer_is_190 (A B C D : ℤ) 
  (h1 : A < B) (h2 : B < C) (h3 : C < D) 
  (h4 : (A + B + C + D) / 4 = 76) 
  (h5 : A = 37) 
  (h6 : B = 38) 
  (h7 : C = 39) : 
  D = 190 := 
sorry

end largest_integer_is_190_l86_86925


namespace vertical_asymptote_l86_86027

theorem vertical_asymptote (x : ℝ) : 4 * x - 9 = 0 → x = 9 / 4 := by
  sorry

end vertical_asymptote_l86_86027


namespace problem_statement_l86_86331

theorem problem_statement (x : ℤ) (h : 3 - x = -2) : x + 1 = 6 := 
by {
  -- Proof would be provided here
  sorry
}

end problem_statement_l86_86331


namespace basketball_free_throws_l86_86743

theorem basketball_free_throws (a b x : ℕ) 
  (h1 : 3 * b = 2 * a)
  (h2 : b = a - 2)
  (h3 : 2 * a + 3 * b + x = 68) : x = 44 :=
by
  sorry

end basketball_free_throws_l86_86743


namespace quarters_needed_to_buy_items_l86_86359

-- Define the costs of each item in cents
def cost_candy_bar : ℕ := 25
def cost_chocolate : ℕ := 75
def cost_juice : ℕ := 50

-- Define the quantities of each item
def num_candy_bars : ℕ := 3
def num_chocolates : ℕ := 2
def num_juice_packs : ℕ := 1

-- Define the value of a quarter in cents
def value_of_quarter : ℕ := 25

-- Define the total cost of the items
def total_cost : ℕ := (num_candy_bars * cost_candy_bar) + (num_chocolates * cost_chocolate) + (num_juice_packs * cost_juice)

-- Calculate the number of quarters needed
def num_quarters_needed : ℕ := total_cost / value_of_quarter

-- The theorem to prove that the number of quarters needed is 11
theorem quarters_needed_to_buy_items : num_quarters_needed = 11 := by
  -- Proof omitted
  sorry

end quarters_needed_to_buy_items_l86_86359


namespace A_inter_complement_B_eq_l86_86785

-- Define set A
def set_A : Set ℝ := {x | -3 < x ∧ x < 6}

-- Define set B
def set_B : Set ℝ := {x | 2 < x ∧ x < 7}

-- Define the complement of set B in the real numbers
def complement_B : Set ℝ := {x | x ≤ 2 ∨ x ≥ 7}

-- Define the intersection of set A with the complement of set B
def A_inter_complement_B : Set ℝ := set_A ∩ complement_B

-- Stating the theorem to prove
theorem A_inter_complement_B_eq : A_inter_complement_B = {x | -3 < x ∧ x ≤ 2} :=
by
  -- Proof goes here
  sorry

end A_inter_complement_B_eq_l86_86785


namespace a_1995_eq_l86_86900

def a_3 : ℚ := (2 + 3) / (1 + 6)

def a (n : ℕ) : ℚ :=
  if n = 3 then a_3
  else if n ≥ 4 then
    let a_n_minus_1 := a (n - 1)
    (a_n_minus_1 + n) / (1 + n * a_n_minus_1)
  else
    0 -- We only care about n ≥ 3 in this problem

-- The problem itself
theorem a_1995_eq :
  a 1995 = 1991009 / 1991011 :=
by
  sorry

end a_1995_eq_l86_86900


namespace point_on_y_axis_is_zero_l86_86837

-- Given conditions
variables (m : ℝ) (y : ℝ)
-- \( P(m, 2) \) lies on the y-axis
def point_on_y_axis (m y : ℝ) : Prop := (m = 0)

-- Proof statement: Prove that if \( P(m, 2) \) lies on the y-axis, then \( m = 0 \)
theorem point_on_y_axis_is_zero (h : point_on_y_axis m 2) : m = 0 :=
by 
  -- the proof would go here
  sorry

end point_on_y_axis_is_zero_l86_86837


namespace dot_product_is_constant_l86_86363

-- Define the trajectory C as the parabola given by the equation y^2 = 4x
def trajectory (x y : ℝ) : Prop := y^2 = 4 * x

-- Prove the range of k for the line passing through point (-1, 0) and intersecting trajectory C
def valid_slope (k : ℝ) : Prop := (-1 < k ∧ k < 0) ∨ (0 < k ∧ k < 1)

-- Prove that ∀ D ≠ A, B on the parabola y^2 = 4x, and lines DA and DB intersect vertical line through (1, 0) on points P, Q, OP ⋅ OQ = 5
theorem dot_product_is_constant (D A B P Q : ℝ × ℝ) 
  (hD : trajectory D.1 D.2)
  (hA : trajectory A.1 A.2)
  (hB : trajectory B.1 B.2)
  (hDiff : D ≠ A ∧ D ≠ B)
  (hP : P = (1, (D.2 * A.2 + 4) / (D.2 + A.2))) 
  (hQ : Q = (1, (D.2 * B.2 + 4) / (D.2 + B.2))) :
  (1 + (D.2 * A.2 + 4) / (D.2 + A.2)) * (1 + (D.2 * B.2 + 4) / (D.2 + B.2)) = 5 :=
sorry

end dot_product_is_constant_l86_86363


namespace degrees_of_interior_angles_l86_86164

-- Definitions for the problem conditions
variables {a b c h_a h_b S : ℝ} 
variables (ABC : Triangle) 
variables (height_to_bc height_to_ac : ℝ)
variables (le_a_ha : a ≤ height_to_bc)
variables (le_b_hb : b ≤ height_to_ac)
variables (area : S = 1 / 2 * a * height_to_bc)
variables (area_eq : S = 1 / 2 * b * height_to_ac)
variables (ha_eq : height_to_bc = 2 * S / a)
variables (hb_eq : height_to_ac = 2 * S / b)
variables (height_pos : 0 < 2 * S)
variables (length_pos : 0 < a ∧ 0 < b ∧ 0 < c)

-- Conclude the degrees of the interior angles
theorem degrees_of_interior_angles : 
  ∃ A B C : ℝ, A = 45 ∧ B = 45 ∧ C = 90 :=
sorry

end degrees_of_interior_angles_l86_86164


namespace cone_volume_divided_by_pi_l86_86065

theorem cone_volume_divided_by_pi : 
  let r := 15
  let l := 20
  let h := 5 * Real.sqrt 7
  let V := (1/3:ℝ) * Real.pi * r^2 * h
  (V / Real.pi = 1125 * Real.sqrt 7) := sorry

end cone_volume_divided_by_pi_l86_86065


namespace gcd_three_digit_palindromes_l86_86538

theorem gcd_three_digit_palindromes : 
  ∀ (a b : ℕ), (1 ≤ a ∧ a ≤ 9) → (1 ≤ b ∧ b ≤ 9) → 
  ∃ d : ℕ, d = 1 ∧ ∀ n m : ℕ, (n = 101 * a + 10 * b) → (m = 101 * a + 10 * b) → gcd n m = d := 
by sorry

end gcd_three_digit_palindromes_l86_86538


namespace inequality_abc_ge_1_sqrt_abcd_l86_86635

theorem inequality_abc_ge_1_sqrt_abcd
  (a b c d : ℝ)
  (h₀ : 0 ≤ a) (h₁ : 0 ≤ b) (h₂ : 0 ≤ c) (h₃ : 0 ≤ d)
  (h_sum : a^2 + b^2 + c^2 + d^2 = 4) :
  (a + b + c + d) / 2 ≥ 1 + Real.sqrt (a * b * c * d) :=
by
  sorry

end inequality_abc_ge_1_sqrt_abcd_l86_86635


namespace inequality_incorrect_l86_86714

theorem inequality_incorrect (a b : ℝ) (h : a > b) : ¬(-a + 2 > -b + 2) :=
by
  sorry

end inequality_incorrect_l86_86714


namespace solve_for_x_l86_86263

theorem solve_for_x {x : ℝ} (h_pos : x > 0) 
  (h_eq : Real.sqrt (12 * x) * Real.sqrt (15 * x) * Real.sqrt (4 * x) * Real.sqrt (10 * x) = 20) :
  x = 2^(1/4) / Real.sqrt 3 :=
by
  -- proof omitted
  sorry

end solve_for_x_l86_86263


namespace question_2024_polynomials_l86_86704

open Polynomial

noncomputable def P (x : ℝ) : Polynomial ℝ := sorry
noncomputable def Q (x : ℝ) : Polynomial ℝ := sorry

-- Main statement
theorem question_2024_polynomials (P Q : Polynomial ℝ) (hP : P.degree = 2024) (hQ : Q.degree = 2024)
    (hPm : P.leadingCoeff = 1) (hQm : Q.leadingCoeff = 1) (h : ∀ x : ℝ, P.eval x ≠ Q.eval x) :
    ∀ (α : ℝ), α ≠ 0 → ∃ x : ℝ, P.eval (x - α) = Q.eval (x + α) :=
by
  sorry

end question_2024_polynomials_l86_86704


namespace Ava_watched_television_for_240_minutes_l86_86729

-- Define the conditions
def hours (h : ℕ) := h = 4

-- Define the conversion factor from hours to minutes
def convert_hours_to_minutes (h : ℕ) : ℕ := h * 60

-- State the theorem
theorem Ava_watched_television_for_240_minutes (h : ℕ) (hh : hours h) : convert_hours_to_minutes h = 240 :=
by
  -- The proof goes here but is skipped
  sorry

end Ava_watched_television_for_240_minutes_l86_86729


namespace jerome_contact_list_count_l86_86349

theorem jerome_contact_list_count :
  (let classmates := 20
   let out_of_school_friends := classmates / 2
   let family := 3 -- two parents and one sister
   let total_contacts := classmates + out_of_school_friends + family
   total_contacts = 33) :=
by
  let classmates := 20
  let out_of_school_friends := classmates / 2
  let family := 3
  let total_contacts := classmates + out_of_school_friends + family
  show total_contacts = 33
  sorry

end jerome_contact_list_count_l86_86349


namespace area_comparison_l86_86033

-- Define the side lengths of the triangles
def a₁ := 17
def b₁ := 17
def c₁ := 12

def a₂ := 17
def b₂ := 17
def c₂ := 16

-- Define the semiperimeters
def s₁ := (a₁ + b₁ + c₁) / 2
def s₂ := (a₂ + b₂ + c₂) / 2

-- Define the areas using Heron's formula
noncomputable def area₁ := (s₁ * (s₁ - a₁) * (s₁ - b₁) * (s₁ - c₁)).sqrt
noncomputable def area₂ := (s₂ * (s₂ - a₂) * (s₂ - b₂) * (s₂ - c₂)).sqrt

-- The theorem to prove
theorem area_comparison : area₁ < area₂ := sorry

end area_comparison_l86_86033


namespace sum_of_x_l86_86505

-- define the function f as an even function
def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

-- define the function f as strictly monotonic on the interval (0, +∞)
def is_strictly_monotonic_on_positive (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 < x → 0 < y → x < y → f x < f y

-- define the main problem statement
theorem sum_of_x (f : ℝ → ℝ) (x : ℝ) (h1 : is_even_function f) (h2 : is_strictly_monotonic_on_positive f) (h3 : x ≠ 0)
  (hx : f (x^2 - 2*x - 1) = f (x + 1)) : 
  ∃ (x1 x2 x3 x4 : ℝ), (x1 + x2 + x3 + x4 = 4) ∧
                        (x1^2 - 3*x1 - 2 = 0) ∧
                        (x2^2 - 3*x2 - 2 = 0) ∧
                        (x3^2 - x3 = 0) ∧
                        (x4^2 - x4 = 0) :=
sorry

end sum_of_x_l86_86505


namespace audrey_peaches_l86_86305

variable (A : ℕ)
variable (P : ℕ := 48)
variable (D : ℕ := 22)

theorem audrey_peaches : A - P = D → A = 70 :=
by
  intro h
  sorry

end audrey_peaches_l86_86305


namespace find_first_term_arithmetic_progression_l86_86565

theorem find_first_term_arithmetic_progression
  (a1 a2 a3 : ℝ)
  (h1 : a1 + a2 + a3 = 12)
  (h2 : a1 * a2 * a3 = 48)
  (h3 : a2 = a1 + d)
  (h4 : a3 = a1 + 2 * d)
  (h5 : a1 < a2 ∧ a2 < a3) :
  a1 = 2 :=
by
  sorry

end find_first_term_arithmetic_progression_l86_86565


namespace katya_classmates_l86_86236

-- Let N be the number of Katya's classmates
variable (N : ℕ)

-- Let K be the number of candies Artyom initially received
variable (K : ℕ)

-- Condition 1: After distributing some candies, Katya had 10 more candies left than Artyom
def condition_1 := K + 10

-- Condition 2: Katya gave each child, including herself, one more candy, so she gave out N + 1 candies in total
def condition_2 := N + 1

-- Condition 3: After giving out these N + 1 candies, everyone in the class has the same number of candies.
def condition_3 : Prop := (K + 1) = (condition_1 K - condition_2 N) / (N + 1)


-- Goal: Prove the number of Katya's classmates N is 9.
theorem katya_classmates : N = 9 :=
by
  -- Restate the conditions in Lean
  
  -- Apply the conditions to find that the only viable solution is N = 9
  sorry

end katya_classmates_l86_86236


namespace set_union_example_l86_86585

open Set

/-- Given sets A = {1, 2, 3} and B = {-1, 1}, prove that A ∪ B = {-1, 1, 2, 3} -/
theorem set_union_example : 
  let A := ({1, 2, 3} : Set ℤ)
  let B := ({-1, 1} : Set ℤ)
  A ∪ B = ({-1, 1, 2, 3} : Set ℤ) :=
by
  let A := ({1, 2, 3} : Set ℤ)
  let B := ({-1, 1} : Set ℤ)
  show A ∪ B = ({-1, 1, 2, 3} : Set ℤ)
  -- Proof to be provided here
  sorry

end set_union_example_l86_86585


namespace sequence_result_l86_86478

theorem sequence_result :
  (1 + 2)^2 + 1 = 10 ∧
  (2 + 3)^2 + 1 = 26 ∧
  (4 + 5)^2 + 1 = 82 →
  (3 + 4)^2 + 1 = 50 :=
by sorry

end sequence_result_l86_86478


namespace cost_price_of_toy_l86_86013

theorem cost_price_of_toy (x : ℝ) (selling_price_per_toy : ℝ) (gain : ℝ) 
  (sale_price : ℝ) (number_of_toys : ℕ) (selling_total : ℝ) (gain_condition : ℝ) :
  (selling_total = number_of_toys * selling_price_per_toy) →
  (selling_price_per_toy = x + gain) →
  (gain = gain_condition / number_of_toys) → 
  (gain_condition = 3 * x) →
  selling_total = 25200 → number_of_toys = 18 → x = 1200 :=
by
  sorry

end cost_price_of_toy_l86_86013


namespace total_blocks_l86_86647

theorem total_blocks (red_blocks yellow_blocks blue_blocks : ℕ) 
  (h1 : red_blocks = 18) 
  (h2 : yellow_blocks = red_blocks + 7) 
  (h3 : blue_blocks = red_blocks + 14) : 
  red_blocks + yellow_blocks + blue_blocks = 75 := 
by
  sorry

end total_blocks_l86_86647


namespace percentage_of_democrats_l86_86705

variable (D R : ℝ)

theorem percentage_of_democrats (h1 : D + R = 100) (h2 : 0.75 * D + 0.20 * R = 53) :
  D = 60 :=
by
  sorry

end percentage_of_democrats_l86_86705


namespace quadratic_roots_and_signs_l86_86245

theorem quadratic_roots_and_signs :
  (∃ x1 x2 : ℝ, (x1^2 - 13*x1 + 40 = 0) ∧ (x2^2 - 13*x2 + 40 = 0) ∧ x1 = 5 ∧ x2 = 8 ∧ 0 < x1 ∧ 0 < x2) :=
by
  sorry

end quadratic_roots_and_signs_l86_86245


namespace average_rainfall_is_4_l86_86707

namespace VirginiaRainfall

def march_rainfall : ℝ := 3.79
def april_rainfall : ℝ := 4.5
def may_rainfall : ℝ := 3.95
def june_rainfall : ℝ := 3.09
def july_rainfall : ℝ := 4.67

theorem average_rainfall_is_4 :
  (march_rainfall + april_rainfall + may_rainfall + june_rainfall + july_rainfall) / 5 = 4 := by
  sorry

end VirginiaRainfall

end average_rainfall_is_4_l86_86707


namespace sum_reciprocals_of_roots_l86_86354

-- Problem statement: Prove that the sum of the reciprocals of the roots of the quadratic equation x^2 - 11x + 6 = 0 is 11/6.
theorem sum_reciprocals_of_roots : 
  ∀ (p q : ℝ), p + q = 11 → p * q = 6 → (1 / p + 1 / q = 11 / 6) :=
by
  intro p q hpq hprod
  sorry

end sum_reciprocals_of_roots_l86_86354


namespace commute_time_l86_86882

theorem commute_time (d w t : ℝ) (x : ℝ) (h_distance : d = 1.5) (h_walking_speed : w = 3) (h_train_speed : t = 20)
  (h_extra_time : 30 = 4.5 + x + 2) : x = 25.5 :=
by {
  -- Add the statement of the proof
  sorry
}

end commute_time_l86_86882


namespace dot_product_a_b_l86_86497

-- Define the vectors a and b
def vector_a : ℝ × ℝ := (2, 3)
def vector_b : ℝ × ℝ := (4, -3)

-- Define the dot product function
def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

-- Statement of the theorem to prove
theorem dot_product_a_b : dot_product vector_a vector_b = -1 := 
by sorry

end dot_product_a_b_l86_86497


namespace original_design_ratio_built_bridge_ratio_l86_86246

-- Definitions
variables (v1 v2 r1 r2 : ℝ)

-- Conditions as per the problem
def original_height_relation : Prop := v1 = 3 * v2
def built_radius_relation : Prop := r2 = 2 * r1

-- Prove the required ratios
theorem original_design_ratio (h1 : original_height_relation v1 v2) (h2 : built_radius_relation r1 r2) : (v1 / r1 = 3 / 4) := sorry

theorem built_bridge_ratio (h1 : original_height_relation v1 v2) (h2 : built_radius_relation r1 r2) : (v2 / r2 = 1 / 8) := sorry

end original_design_ratio_built_bridge_ratio_l86_86246


namespace origin_not_in_A_point_M_in_A_l86_86074

def set_A : Set (ℝ × ℝ) := { p | ∃ x y : ℝ, p = (x, y) ∧ x + 2 * y - 1 ≥ 0 ∧ y ≤ x + 2 ∧ 2 * x + y - 5 ≤ 0}

theorem origin_not_in_A : (0, 0) ∉ set_A := by
  sorry

theorem point_M_in_A : (1, 1) ∈ set_A := by
  sorry

end origin_not_in_A_point_M_in_A_l86_86074


namespace system_of_equations_solution_l86_86249

theorem system_of_equations_solution
  (x1 x2 x3 x4 x5 : ℝ)
  (h1 : x1 + 2 * x2 + 2 * x3 + 2 * x4 + 2 * x5 = 1)
  (h2 : x1 + 3 * x2 + 4 * x3 + 4 * x4 + 4 * x5 = 2)
  (h3 : x1 + 3 * x2 + 5 * x3 + 6 * x4 + 6 * x5 = 3)
  (h4 : x1 + 3 * x2 + 5 * x3 + 7 * x4 + 8 * x5 = 4)
  (h5 : x1 + 3 * x2 + 5 * x3 + 7 * x4 + 9 * x5 = 5) :
  x1 = 1 ∧ x2 = -1 ∧ x3 = 1 ∧ x4 = -1 ∧ x5 = 1 :=
by {
  -- proof steps go here
  sorry
}

end system_of_equations_solution_l86_86249


namespace eval_expression_l86_86113

theorem eval_expression : 8 - (6 / (4 - 2)) = 5 := 
sorry

end eval_expression_l86_86113


namespace f_at_2_l86_86965

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 3 * x + 4

-- State the theorem that we need to prove
theorem f_at_2 : f 2 = 2 := by
  -- the proof will go here
  sorry

end f_at_2_l86_86965


namespace polar_curve_symmetry_l86_86045

theorem polar_curve_symmetry :
  ∀ (ρ θ : ℝ), ρ = 4 * Real.sin (θ - π / 3) → 
  ∃ k : ℤ, θ = 5 * π / 6 + k * π :=
sorry

end polar_curve_symmetry_l86_86045


namespace transformation_matrix_correct_l86_86366
noncomputable def M : Matrix (Fin 2) (Fin 2) ℝ := ![
  ![0, 3],
  ![-3, 0]
]

theorem transformation_matrix_correct :
  let R : Matrix (Fin 2) (Fin 2) ℝ := ![
    ![0, 1],
    ![-1, 0]
  ];
  let S : ℝ := 3;
  M = S • R :=
by
  sorry

end transformation_matrix_correct_l86_86366


namespace coat_price_reduction_l86_86409

variable (original_price reduction : ℝ)

theorem coat_price_reduction
  (h_orig : original_price = 500)
  (h_reduct : reduction = 350)
  : reduction / original_price * 100 = 70 := 
sorry

end coat_price_reduction_l86_86409


namespace find_inscription_l86_86878

-- Definitions for the conditions
def identical_inscriptions (box1 box2 : String) : Prop :=
  box1 = box2

def conclusion_same_master (box : String) : Prop :=
  (∀ (made_by : String → Prop), made_by "Bellini" ∨ made_by "Cellini") ∧
  ¬∀ (made_by : String → Prop), made_by "Bellini" ∧ made_by "Cellini"

def cannot_identify_master (box : String) : Prop :=
  ¬(∀ (made_by : String → Prop), made_by "Bellini") ∧
  ¬(∀ (made_by : String → Prop), made_by "Cellini")

def single_casket_indeterminate (box : String) : Prop :=
  (∀ (made_by : String → Prop), made_by "Bellini" ∨ made_by "Cellini") ∧
  ¬(∀ (made_by : String → Prop), made_by "Bellini" ∧ made_by "Cellini") ∧
  ¬(∀ (made_by : String → Prop), made_by "Bellini")

-- Inscription on the boxes
def inscription := "At least one of these boxes was made by Cellini's son."

-- The Lean statement for the proof
theorem find_inscription (box1 box2 : String)
  (h1 : identical_inscriptions box1 box2)
  (h2 : conclusion_same_master box1)
  (h3 : cannot_identify_master box1)
  (h4 : single_casket_indeterminate box1) :
  box1 = inscription :=
sorry

end find_inscription_l86_86878


namespace value_of_m_l86_86100

theorem value_of_m (m : ℝ) (h₁ : m^2 - 9 * m + 19 = 1) (h₂ : 2 * m^2 - 7 * m - 9 ≤ 0) : m = 3 :=
sorry

end value_of_m_l86_86100


namespace hyperbola_eccentricity_l86_86076

theorem hyperbola_eccentricity (a b : ℝ) (h : ∃ P : ℝ × ℝ, ∃ A : ℝ × ℝ, ∃ F : ℝ × ℝ, 
  (∃ c : ℝ, F = (c, 0) ∧ A = (-a, 0) ∧ P.1 ^ 2 / a ^ 2 - P.2 ^ 2 / b ^ 2 = 1 ∧ 
  (F.fst - P.fst) ^ 2 + P.snd ^ 2 = (F.fst + a) ^ 2 ∧ (F.fst - A.fst) ^ 2 + (F.snd - A.snd) ^ 2 = (F.fst + a) ^ 2 ∧ 
  (P.snd = F.snd) ∧ (abs (F.fst - A.fst) = abs (F.fst - P.fst)))) : 
∃ e : ℝ, e = 2 :=
by
  sorry

end hyperbola_eccentricity_l86_86076


namespace zoo_rabbits_count_l86_86173

theorem zoo_rabbits_count (parrots rabbits : ℕ) (h_ratio : parrots * 4 = rabbits * 3) (h_parrots_count : parrots = 21) : rabbits = 28 :=
by
  sorry

end zoo_rabbits_count_l86_86173


namespace right_triangle_48_55_l86_86998

def right_triangle_properties (a b : ℕ) (ha : a = 48) (hb : b = 55) : Prop :=
  let area := 1 / 2 * a * b
  let hypotenuse := Real.sqrt (a ^ 2 + b ^ 2)
  area = 1320 ∧ hypotenuse = 73

theorem right_triangle_48_55 : right_triangle_properties 48 55 (by rfl) (by rfl) :=
  sorry

end right_triangle_48_55_l86_86998


namespace club_members_after_four_years_l86_86723

theorem club_members_after_four_years
  (b : ℕ → ℕ)
  (h_initial : b 0 = 20)
  (h_recursive : ∀ k, b (k + 1) = 3 * (b k) - 10) :
  b 4 = 1220 :=
sorry

end club_members_after_four_years_l86_86723


namespace sam_age_two_years_ago_l86_86450

theorem sam_age_two_years_ago (J S : ℕ) (h1 : J = 3 * S) (h2 : J + 9 = 2 * (S + 9)) : S - 2 = 7 :=
sorry

end sam_age_two_years_ago_l86_86450


namespace units_digit_of_147_pow_is_7_some_exponent_units_digit_l86_86682

theorem units_digit_of_147_pow_is_7 (n : ℕ) : (147 ^ 25) % 10 = 7 % 10 :=
by
  sorry

theorem some_exponent_units_digit (n : ℕ) (hn : n % 4 = 2) : ((147 ^ 25) ^ n) % 10 = 9 :=
by
  have base_units_digit := units_digit_of_147_pow_is_7 25
  sorry

end units_digit_of_147_pow_is_7_some_exponent_units_digit_l86_86682


namespace primes_or_prime_squares_l86_86098

theorem primes_or_prime_squares (n : ℕ) (h1 : n > 1)
  (h2 : ∀ d, d ∣ n → d > 1 → (d - 1) ∣ (n - 1)) : 
  (∃ p, Nat.Prime p ∧ (n = p ∨ n = p * p)) :=
by
  sorry

end primes_or_prime_squares_l86_86098


namespace function_relation_l86_86681

theorem function_relation (f : ℝ → ℝ) 
  (h0 : ∀ x, f (-x) = f x)
  (h1 : ∀ x, f (x + 2) = f x)
  (h2 : ∀ x y, 0 ≤ x ∧ x < y ∧ y ≤ 1 → f x < f y) :
  f 0 < f (-6.5) ∧ f (-6.5) < f (-1) := 
by
  sorry

end function_relation_l86_86681


namespace ac_lt_bc_of_a_gt_b_and_c_lt_0_l86_86561

theorem ac_lt_bc_of_a_gt_b_and_c_lt_0 {a b c : ℝ} (h1 : a > b) (h2 : c < 0) : a * c < b * c :=
  sorry

end ac_lt_bc_of_a_gt_b_and_c_lt_0_l86_86561


namespace min_floor_sum_l86_86547

theorem min_floor_sum (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  ∃ (n : ℕ), n = 4 ∧ n = 
  ⌊(2 * a + b) / c⌋ + ⌊(b + 2 * c) / a⌋ + ⌊(2 * c + a) / b⌋ := 
sorry

end min_floor_sum_l86_86547


namespace num_divisors_count_l86_86733

theorem num_divisors_count (n : ℕ) (m : ℕ) (H : m = 32784) :
  (∃ S : Finset ℕ, (∀ x ∈ S, x ∈ (Finset.range 10) ∧ m % x = 0) ∧ S.card = n) ↔ n = 7 :=
by
  sorry

end num_divisors_count_l86_86733


namespace number_of_sturgeons_l86_86935

def number_of_fishes := 145
def number_of_pikes := 30
def number_of_herrings := 75

theorem number_of_sturgeons : (number_of_fishes - (number_of_pikes + number_of_herrings) = 40) :=
  by
  sorry

end number_of_sturgeons_l86_86935


namespace number_of_blue_butterflies_l86_86735

theorem number_of_blue_butterflies 
  (total_butterflies : ℕ)
  (B Y : ℕ)
  (H1 : total_butterflies = 11)
  (H2 : B = 2 * Y)
  (H3 : total_butterflies = B + Y + 5) : B = 4 := 
sorry

end number_of_blue_butterflies_l86_86735


namespace linear_equation_m_not_eq_4_l86_86061

theorem linear_equation_m_not_eq_4 (m x y : ℝ) :
  (m * x + 3 * y = 4 * x - 1) → m ≠ 4 :=
by
  sorry

end linear_equation_m_not_eq_4_l86_86061


namespace pow_two_gt_cube_l86_86141

theorem pow_two_gt_cube (n : ℕ) (h : 10 ≤ n) : 2^n > n^3 := sorry

end pow_two_gt_cube_l86_86141


namespace problem_statement_l86_86522

theorem problem_statement (a b : ℝ) (h : (1 / a + 1 / b) / (1 / a - 1 / b) = 2023) : (a + b) / (a - b) = 2023 :=
by
  sorry

end problem_statement_l86_86522


namespace part1_part2_l86_86414

theorem part1 (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (h3 : (1 - 4 / (2 * a^0 + a)) = 0) : a = 2 :=
sorry

theorem part2 (k : ℝ) (h : ∃ x : ℝ, (2^x + 1) * (1 - 2 / (2^x + 1)) + k = 0) : k < 1 :=
sorry

end part1_part2_l86_86414


namespace jordan_book_pages_l86_86955

theorem jordan_book_pages (avg_first_4_days : ℕ)
                           (avg_next_2_days : ℕ)
                           (pages_last_day : ℕ)
                           (total_pages : ℕ) :
  avg_first_4_days = 42 → 
  avg_next_2_days = 38 → 
  pages_last_day = 20 → 
  total_pages = 4 * avg_first_4_days + 2 * avg_next_2_days + pages_last_day →
  total_pages = 264 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  exact h4

end jordan_book_pages_l86_86955


namespace find_function_l86_86499

theorem find_function (f : ℝ → ℝ) (h : ∀ x : ℝ, f x + (0.5 + x) * f (1 - x) = 1) :
  ∀ x : ℝ, f x = if x ≠ 0.5 then 1 / (0.5 - x) else 0.5 :=
by
  sorry

end find_function_l86_86499


namespace find_a_for_exponential_function_l86_86124

theorem find_a_for_exponential_function (a : ℝ) :
  a - 2 = 1 ∧ a > 0 ∧ a ≠ 1 → a = 3 :=
by
  intro h
  obtain ⟨h1, h2, h3⟩ := h
  sorry

end find_a_for_exponential_function_l86_86124


namespace farmer_goats_sheep_unique_solution_l86_86475

theorem farmer_goats_sheep_unique_solution:
  ∃ g h : ℕ, 0 < g ∧ 0 < h ∧ 28 * g + 30 * h = 1200 ∧ h > g :=
by
  sorry

end farmer_goats_sheep_unique_solution_l86_86475


namespace eccentricity_proof_l86_86888

variables (a b c : ℝ) (h1 : a > b) (h2 : b > 0)
def ellipse_eq (x y: ℝ) := (x^2 / a^2) + (y^2 / b^2) = 1
def circle_eq (x y: ℝ) := x^2 + y^2 = b^2

-- Conditions
def a_eq_3b : Prop := a = 3 * b
def major_minor_axis_relation : Prop := a^2 = b^2 + c^2

-- To prove
theorem eccentricity_proof 
  (h3 : a_eq_3b a b)
  (h4 : major_minor_axis_relation a b c) :
  (c / a) = (2 * Real.sqrt 2 / 3) := 
  sorry

end eccentricity_proof_l86_86888


namespace problem_solution_l86_86966

variable (x y z : ℝ)

theorem problem_solution
  (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (h1 : x + y + x * y = 8)
  (h2 : y + z + y * z = 15)
  (h3 : z + x + z * x = 35) :
  x + y + z + x * y = 15 :=
sorry

end problem_solution_l86_86966


namespace quadratic_has_two_distinct_roots_l86_86857

theorem quadratic_has_two_distinct_roots (a b c : ℝ) (h : 2016 + a^2 + a * c < a * b) : 
  (b^2 - 4 * a * c) > 0 :=
by {
  sorry
}

end quadratic_has_two_distinct_roots_l86_86857


namespace smallest_percentage_owning_90_percent_money_l86_86997

theorem smallest_percentage_owning_90_percent_money
  (P M : ℝ)
  (h1 : 0.2 * P = 0.8 * M) :
  (∃ x : ℝ, x = 0.6 * P ∧ 0.9 * M <= (0.2 * P + (x - 0.2 * P))) :=
sorry

end smallest_percentage_owning_90_percent_money_l86_86997


namespace max_value_quadratic_max_value_quadratic_attained_l86_86929

def quadratic (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem max_value_quadratic : ∀ (x : ℝ), quadratic (-8) 32 (-1) x ≤ 31 :=
by
  sorry

theorem max_value_quadratic_attained : 
  quadratic (-8) 32 (-1) 2 = 31 :=
by
  sorry

end max_value_quadratic_max_value_quadratic_attained_l86_86929


namespace range_of_a_l86_86312

theorem range_of_a (a : ℝ) :
  (∀ x ≥ 0, ∃ y ∈ Set.Ici a, y = (x^2 + 2*x + a) / (x + 1)) ↔ a ≤ 2 :=
by
  sorry

end range_of_a_l86_86312


namespace least_k_divisible_by_2160_l86_86889

theorem least_k_divisible_by_2160 (k : ℤ) : k^3 ∣ 2160 → k ≥ 60 := by
  sorry

end least_k_divisible_by_2160_l86_86889


namespace max_negatives_l86_86351

theorem max_negatives (a b c d e f : ℤ) (h : ab + cdef < 0) : ∃ w : ℤ, w = 4 := 
sorry

end max_negatives_l86_86351


namespace tiles_needed_l86_86446

def hallway_length : ℕ := 14
def hallway_width : ℕ := 20
def border_tile_side : ℕ := 2
def interior_tile_side : ℕ := 3

theorem tiles_needed :
  let border_length_tiles := ((hallway_length - 2 * border_tile_side) / border_tile_side) * 2
  let border_width_tiles := ((hallway_width - 2 * border_tile_side) / border_tile_side) * 2
  let corner_tiles := 4
  let total_border_tiles := border_length_tiles + border_width_tiles + corner_tiles
  let interior_length := hallway_length - 2 * border_tile_side
  let interior_width := hallway_width - 2 * border_tile_side
  let interior_area := interior_length * interior_width
  let interior_tiles_needed := (interior_area + interior_tile_side * interior_tile_side - 1) / (interior_tile_side * interior_tile_side)
  total_border_tiles + interior_tiles_needed = 48 := 
by {
  sorry
}

end tiles_needed_l86_86446


namespace smallest_h_l86_86972

theorem smallest_h (h : ℕ) : 
  (∀ k, h = k → (k + 5) % 8 = 0 ∧ 
        (k + 5) % 11 = 0 ∧ 
        (k + 5) % 24 = 0) ↔ h = 259 :=
by
  sorry

end smallest_h_l86_86972


namespace fraction_to_decimal_l86_86006

theorem fraction_to_decimal : (3 : ℚ) / 40 = 0.075 :=
by
  sorry

end fraction_to_decimal_l86_86006


namespace ordering_of_a_b_c_l86_86423

theorem ordering_of_a_b_c (a b c : ℝ)
  (ha : a = Real.exp (1 / 2))
  (hb : b = Real.log (1 / 2))
  (hc : c = Real.sin (1 / 2)) :
  a > c ∧ c > b :=
by sorry

end ordering_of_a_b_c_l86_86423


namespace can_still_row_probability_l86_86125

/-- Define the probabilities for the left and right oars --/
def P_left1_work : ℚ := 3 / 5
def P_left2_work : ℚ := 2 / 5
def P_right1_work : ℚ := 4 / 5 
def P_right2_work : ℚ := 3 / 5

/-- Define the probabilities of the failures as complementary probabilities --/
def P_left1_fail : ℚ := 1 - P_left1_work
def P_left2_fail : ℚ := 1 - P_left2_work
def P_right1_fail : ℚ := 1 - P_right1_work
def P_right2_fail : ℚ := 1 - P_right2_work

/-- Define the probability of both left oars failing --/
def P_both_left_fail : ℚ := P_left1_fail * P_left2_fail

/-- Define the probability of both right oars failing --/
def P_both_right_fail : ℚ := P_right1_fail * P_right2_fail

/-- Define the probability of all four oars failing --/
def P_all_fail : ℚ := P_both_left_fail * P_both_right_fail

/-- Calculate the probability that at least one oar on each side works --/
def P_can_row : ℚ := 1 - (P_both_left_fail + P_both_right_fail - P_all_fail)

theorem can_still_row_probability :
  P_can_row = 437 / 625 :=
by {
  -- The proof is to be completed
  sorry
}

end can_still_row_probability_l86_86125


namespace simplify_expression_l86_86492

variable (p q r : ℝ)
variable (hp : p ≠ 2)
variable (hq : q ≠ 3)
variable (hr : r ≠ 4)

theorem simplify_expression : 
  (p^2 - 4) / (4 - r^2) * (q^2 - 9) / (2 - p^2) * (r^2 - 16) / (3 - q^2) = -1 :=
by
  -- Skipping the proof using sorry
  sorry

end simplify_expression_l86_86492


namespace smallest_seating_l86_86841

theorem smallest_seating (N : ℕ) (h: ∀ (chairs : ℕ) (occupants : ℕ), 
  chairs = 100 ∧ occupants = 25 → 
  ∃ (adjacent_occupied: ℕ), adjacent_occupied > 0 ∧ adjacent_occupied < chairs ∧
  adjacent_occupied ≠ occupants) : 
  N = 25 :=
sorry

end smallest_seating_l86_86841


namespace mr_blues_yard_expectation_l86_86586

noncomputable def calculate_expected_harvest (length_steps : ℕ) (width_steps : ℕ) (step_length : ℝ) (yield_per_sqft : ℝ) : ℝ :=
  let length_feet := length_steps * step_length
  let width_feet := width_steps * step_length
  let area := length_feet * width_feet
  let total_yield := area * yield_per_sqft
  total_yield

theorem mr_blues_yard_expectation : calculate_expected_harvest 18 25 2.5 (3 / 4) = 2109.375 :=
by
  sorry

end mr_blues_yard_expectation_l86_86586


namespace geom_seq_find_b3_l86_86863

-- Given conditions
def is_geometric_seq (b : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, b (n + 1) = b n * r

def geom_seq_condition (b : ℕ → ℝ) : Prop :=
  is_geometric_seq b ∧ b 2 * b 3 * b 4 = 8

-- Proof statement: We need to prove that b 3 = 2
theorem geom_seq_find_b3 (b : ℕ → ℝ) (h : geom_seq_condition b) : b 3 = 2 :=
  sorry

end geom_seq_find_b3_l86_86863


namespace find_constants_and_min_value_l86_86567

noncomputable def f (a b x : ℝ) := a * Real.exp x + b * x * Real.log x
noncomputable def f' (a b x : ℝ) := a * Real.exp x + b * Real.log x + b
noncomputable def g (a b x : ℝ) := f a b x - Real.exp 1 * x^2

theorem find_constants_and_min_value :
  (∀ (a b : ℝ),
    -- Condition for the derivative at x = 1 and the given tangent line slope
    (f' a b 1 = 2 * Real.exp 1) ∧
    -- Condition for the function value at x = 1
    (f a b 1 = Real.exp 1) →
    -- Expected results for a and b
    (a = 1 ∧ b = Real.exp 1)) ∧

  -- Evaluating the minimum value of the function g(x)
  (∀ (x : ℝ), 0 < x →
    -- Given the minimum occurs at x = 1
    g 1 (Real.exp 1) 1 = 0 ∧
    (∀ (x : ℝ), 0 < x →
      (g 1 (Real.exp 1) x ≥ 0))) :=
sorry

end find_constants_and_min_value_l86_86567


namespace products_not_all_greater_than_one_quarter_l86_86029

theorem products_not_all_greater_than_one_quarter
  (a b c : ℝ)
  (ha : 0 < a ∧ a < 1)
  (hb : 0 < b ∧ b < 1)
  (hc : 0 < c ∧ c < 1) :
  ¬ ((1 - a) * b > 1 / 4 ∧ (1 - b) * c > 1 / 4 ∧ (1 - c) * a > 1 / 4) :=
by
  sorry

end products_not_all_greater_than_one_quarter_l86_86029


namespace pq_sum_of_harmonic_and_geometric_sequences_l86_86201

theorem pq_sum_of_harmonic_and_geometric_sequences
  (x y z : ℝ)
  (h1 : (1 / x - 1 / y) / (1 / y - 1 / z) = 1)
  (h2 : 3 * x * y = 7 * z) :
  ∃ p q : ℕ, (Nat.gcd p q = 1) ∧ p + q = 79 :=
by
  sorry

end pq_sum_of_harmonic_and_geometric_sequences_l86_86201


namespace quadratic_has_real_roots_l86_86460

theorem quadratic_has_real_roots (m : ℝ) : (∃ x : ℝ, (m - 1) * x^2 - 2 * x + 1 = 0) ↔ (m ≤ 2 ∧ m ≠ 1) := 
by 
  sorry

end quadratic_has_real_roots_l86_86460


namespace no_counterexample_exists_l86_86992

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem no_counterexample_exists : ∀ n : ℕ, sum_of_digits n % 9 = 0 → n % 9 = 0 :=
by
  intro n h
  sorry

end no_counterexample_exists_l86_86992


namespace deer_distribution_l86_86575

theorem deer_distribution :
  ∃ a : ℕ → ℚ,
    (a 1 + a 2 + a 3 + a 4 + a 5 = 5) ∧
    (a 4 = 2 / 3) ∧ 
    (a 3 = 1) ∧ 
    (a 1 = 5 / 3) :=
by
  sorry

end deer_distribution_l86_86575


namespace eunseo_change_correct_l86_86180

-- Define the given values
def r : ℕ := 3
def p_r : ℕ := 350
def b : ℕ := 2
def p_b : ℕ := 180
def P : ℕ := 2000

-- Define the total cost of candies and the change
def total_cost := r * p_r + b * p_b
def change := P - total_cost

-- Theorem statement
theorem eunseo_change_correct : change = 590 := by
  -- proof not required, so using sorry
  sorry

end eunseo_change_correct_l86_86180


namespace number_of_girls_in_school_l86_86148

theorem number_of_girls_in_school (total_students : ℕ) (sample_size : ℕ) (x : ℕ) :
  total_students = 2400 →
  sample_size = 200 →
  2 * x + 10 = sample_size →
  (95 / 200 : ℚ) * (total_students : ℚ) = 1140 :=
by
  intros h_total h_sample h_sampled
  rw [h_total, h_sample] at *
  sorry

end number_of_girls_in_school_l86_86148


namespace trig_solution_l86_86195

noncomputable def solve_trig_system (x y : ℝ) : Prop :=
  (3 * Real.cos x + 4 * Real.sin x = -1.4) ∧ 
  (13 * Real.cos x - 41 * Real.cos y = -45) ∧ 
  (13 * Real.sin x + 41 * Real.sin y = 3)

theorem trig_solution :
  solve_trig_system (112.64 * Real.pi / 180) (347.32 * Real.pi / 180) ∧ 
  solve_trig_system (239.75 * Real.pi / 180) (20.31 * Real.pi / 180) :=
by {
    repeat { sorry }
  }

end trig_solution_l86_86195


namespace juice_left_l86_86267

theorem juice_left (total consumed : ℚ) (h_total : total = 1) (h_consumed : consumed = 4 / 6) :
  total - consumed = 2 / 6 ∨ total - consumed = 1 / 3 :=
by
  sorry

end juice_left_l86_86267


namespace rectangle_length_difference_l86_86760

variable (s l w : ℝ)

-- Conditions
def condition1 : Prop := 2 * (l + w) = 4 * s + 4
def condition2 : Prop := w = s - 2

-- Theorem to prove
theorem rectangle_length_difference
  (s l w : ℝ)
  (h1 : condition1 s l w)
  (h2 : condition2 s w) : l = s + 4 :=
by
sorry

end rectangle_length_difference_l86_86760


namespace height_of_box_l86_86355

theorem height_of_box (h : ℝ) :
  (∃ (h : ℝ),
    (∀ (x y z : ℝ), (x = 3) ∧ (y = 3) ∧ (z = h / 2) → true) ∧
    (∀ (x y z : ℝ), (x = 1) ∧ (y = 1) ∧ (z = 1) → true) ∧
    h = 6) :=
sorry

end height_of_box_l86_86355


namespace sufficient_but_not_necessary_condition_l86_86995

theorem sufficient_but_not_necessary_condition (a : ℝ) :
  (∀ x : ℝ, (-2 ≤ x ∧ x ≤ 2) → (x ≤ a))
  → (∃ x : ℝ, (x ≤ a ∧ ¬((-2 ≤ x ∧ x ≤ 2))))
  → (a ≥ 2) :=
by
  intros h1 h2
  sorry

end sufficient_but_not_necessary_condition_l86_86995


namespace intersection_is_correct_l86_86346

def M : Set ℤ := {-2, 1, 2}
def N : Set ℤ := {1, 2, 4}

theorem intersection_is_correct : M ∩ N = {1, 2} := 
by {
  sorry
}

end intersection_is_correct_l86_86346


namespace find_multiple_of_sum_l86_86873

-- Define the conditions and the problem statement in Lean
theorem find_multiple_of_sum (a b m : ℤ) 
  (h1 : b = 8) 
  (h2 : b - a = 3) 
  (h3 : a * b = 14 + m * (a + b)) : 
  m = 2 :=
by
  sorry

end find_multiple_of_sum_l86_86873


namespace medium_stores_in_sample_l86_86051

theorem medium_stores_in_sample :
  let total_stores := 300
  let large_stores := 30
  let medium_stores := 75
  let small_stores := 195
  let sample_size := 20
  sample_size * (medium_stores/total_stores) = 5 :=
by
  sorry

end medium_stores_in_sample_l86_86051


namespace solution_part_for_a_l86_86218

noncomputable def find_k (k x y n : ℕ) : Prop :=
  gcd x y = 1 ∧ 
  x > 0 ∧ y > 0 ∧ 
  k % (x^2) = 0 ∧ 
  k % (y^2) = 0 ∧ 
  k / (x^2) = n ∧ 
  k / (y^2) = n + 148

theorem solution_part_for_a (k x y n : ℕ) (h : find_k k x y n) : k = 467856 :=
sorry

end solution_part_for_a_l86_86218


namespace arithmetic_mean_of_reciprocals_of_first_five_primes_l86_86198

theorem arithmetic_mean_of_reciprocals_of_first_five_primes :
  (1 / 2 + 1 / 3 + 1 / 5 + 1 / 7 + 1 / 11) / 5 = 2927 / 11550 := 
sorry

end arithmetic_mean_of_reciprocals_of_first_five_primes_l86_86198


namespace cookie_count_l86_86170

theorem cookie_count (C : ℕ) 
  (h1 : 3 * C / 4 + 1 * (C / 4) / 5 + 1 * (C / 4) * 4 / 20 = 10) 
  (h2: 1 * (5 * 4 / 20) / 10 = 1): 
  C = 100 :=
by 
sorry

end cookie_count_l86_86170


namespace product_of_number_and_sum_of_digits_l86_86527

-- Definitions according to the conditions
def units_digit (a b : ℕ) : Prop := b = a + 2
def number_equals_24 (a b : ℕ) : Prop := 10 * a + b = 24

-- The main statement to prove the product of the number and the sum of its digits
theorem product_of_number_and_sum_of_digits :
  ∃ (a b : ℕ), units_digit a b ∧ number_equals_24 a b ∧ (24 * (a + b) = 144) :=
sorry

end product_of_number_and_sum_of_digits_l86_86527


namespace parabola_focus_l86_86183

theorem parabola_focus (x y : ℝ) : (y^2 = -8 * x) → (x, y) = (-2, 0) :=
by
  sorry

end parabola_focus_l86_86183


namespace factor_equivalence_l86_86910

noncomputable def given_expression (x : ℝ) :=
  (3 * x^3 + 70 * x^2 - 5) - (-4 * x^3 + 2 * x^2 - 5)

noncomputable def target_form (x : ℝ) :=
  7 * x^2 * (x + 68 / 7)

theorem factor_equivalence (x : ℝ) : given_expression x = target_form x :=
by
  sorry

end factor_equivalence_l86_86910


namespace smallest_positive_integer_x_for_2520x_eq_m_cubed_l86_86612

theorem smallest_positive_integer_x_for_2520x_eq_m_cubed :
  ∃ (M x : ℕ), x > 0 ∧ 2520 * x = M^3 ∧ (∀ y, y > 0 ∧ 2520 * y = M^3 → x ≤ y) :=
sorry

end smallest_positive_integer_x_for_2520x_eq_m_cubed_l86_86612


namespace line_equation_l86_86352

variable (t : ℝ)
variable (x y : ℝ)

def param_x (t : ℝ) : ℝ := 3 * t + 2
def param_y (t : ℝ) : ℝ := 5 * t - 7

theorem line_equation :
  ∃ m b : ℝ, ∀ t : ℝ, y = param_y t ∧ x = param_x t → y = m * x + b := by
  use (5 / 3)
  use (-31 / 3)
  sorry

end line_equation_l86_86352


namespace book_cost_in_cny_l86_86116

-- Conditions
def usd_to_nad : ℝ := 7      -- One US dollar to Namibian dollar
def usd_to_cny : ℝ := 6      -- One US dollar to Chinese yuan
def book_cost_nad : ℝ := 168 -- Cost of the book in Namibian dollars

-- Statement to prove
theorem book_cost_in_cny : book_cost_nad * (usd_to_cny / usd_to_nad) = 144 :=
sorry

end book_cost_in_cny_l86_86116


namespace sum_of_all_possible_values_of_N_with_equation_l86_86661

def satisfiesEquation (N : ℝ) : Prop :=
  N * (N - 4) = -7

theorem sum_of_all_possible_values_of_N_with_equation :
  (∀ N, satisfiesEquation N → N + (4 - N) = 4) :=
sorry

end sum_of_all_possible_values_of_N_with_equation_l86_86661


namespace boat_speed_l86_86069

theorem boat_speed (v : ℝ) : 
  let rate_current := 7
  let distance := 35.93
  let time := 44 / 60
  (v + rate_current) * time = distance → v = 42 :=
by
  intro h
  sorry

end boat_speed_l86_86069


namespace light_flash_time_l86_86744

/--
A light flashes every few seconds. In 3/4 of an hour, it flashes 300 times.
Prove that it takes 9 seconds for the light to flash once.
-/
theorem light_flash_time : 
  (3 / 4 * 60 * 60) / 300 = 9 :=
by
  sorry

end light_flash_time_l86_86744


namespace f_at_one_f_extremes_l86_86373

noncomputable def f : ℝ → ℝ := sorry
axiom f_domain : ∀ x : ℝ, x > 0 → f x = f x
axiom f_multiplicative : ∀ x y : ℝ, x > 0 → y > 0 → f (x * y) = f x + f y
axiom f_positive : ∀ x : ℝ, x > 1 → f x > 0

theorem f_at_one : f 1 = 0 := sorry

theorem f_extremes (hf_sub_one_fifth : f (1 / 5) = -1) :
  ∃ c d : ℝ, (∀ x : ℝ, 1 / 25 ≤ x ∧ x ≤ 125 → c ≤ f x ∧ f x ≤ d) ∧
  c = -2 ∧ d = 3 := sorry

end f_at_one_f_extremes_l86_86373


namespace profit_value_l86_86016

variable (P : ℝ) -- Total profit made by the business in that year.
variable (MaryInvestment : ℝ) -- Mary's investment
variable (MikeInvestment : ℝ) -- Mike's investment
variable (MaryExtra : ℝ) -- Extra money received by Mary

-- Conditions
axiom mary_investment : MaryInvestment = 900
axiom mike_investment : MikeInvestment = 100
axiom mary_received_more : MaryExtra = 1600
axiom profit_shared_equally : (P / 3) / 2 + (MaryInvestment / (MaryInvestment + MikeInvestment)) * (2 * P / 3) 
                           = MikeInvestment / (MaryInvestment + MikeInvestment) * (2 * P / 3) + MaryExtra

-- Statement
theorem profit_value : P = 4000 :=
by
  sorry

end profit_value_l86_86016


namespace lanes_on_road_l86_86753

theorem lanes_on_road (num_lanes : ℕ)
  (h1 : ∀ trucks_per_lane cars_per_lane total_vehicles, 
          cars_per_lane = 2 * (trucks_per_lane * num_lanes) ∧
          trucks_per_lane = 60 ∧
          total_vehicles = num_lanes * (trucks_per_lane + cars_per_lane) ∧
          total_vehicles = 2160) :
  num_lanes = 12 :=
by
  sorry

end lanes_on_road_l86_86753


namespace star_polygon_internal_angles_sum_l86_86072

-- Define the core aspects of the problem using type defintions and axioms.
def n_star_polygon_total_internal_angle_sum (n : ℕ) : ℝ :=
  180 * (n - 4)

theorem star_polygon_internal_angles_sum (n : ℕ) (h : n ≥ 6) :
  n_star_polygon_total_internal_angle_sum n = 180 * (n - 4) :=
by
  -- This step would involve the formal proof using Lean
  sorry

end star_polygon_internal_angles_sum_l86_86072


namespace find_value_l86_86907

variable (N : ℝ)

def condition : Prop := (1 / 4) * (1 / 3) * (2 / 5) * N = 16

theorem find_value (h : condition N) : (1 / 3) * (2 / 5) * N = 64 :=
sorry

end find_value_l86_86907


namespace quadratic_polynomial_l86_86357

theorem quadratic_polynomial (x y : ℝ) (hx : x + y = 12) (hy : x * (3 * y) = 108) : 
  (t : ℝ) → t^2 - 12 * t + 36 = 0 :=
by 
  sorry

end quadratic_polynomial_l86_86357


namespace trapezoid_base_difference_is_10_l86_86988

noncomputable def trapezoid_base_difference (AD BC AB : ℝ) (angle_BAD angle_ADC : ℝ) : ℝ :=
if angle_BAD = 60 ∧ angle_ADC = 30 ∧ AB = 5 then AD - BC else 0

theorem trapezoid_base_difference_is_10 (AD BC : ℝ) (angle_BAD angle_ADC : ℝ) (h_BAD : angle_BAD = 60)
(h_ADC : angle_ADC = 30) (h_AB : AB = 5) : trapezoid_base_difference AD BC AB angle_BAD angle_ADC = 10 :=
sorry

end trapezoid_base_difference_is_10_l86_86988


namespace rational_solutions_iff_k_equals_8_l86_86336

theorem rational_solutions_iff_k_equals_8 {k : ℕ} (hk : k > 0) :
  (∃ (x : ℚ), k * x^2 + 16 * x + k = 0) ↔ k = 8 :=
by
  sorry

end rational_solutions_iff_k_equals_8_l86_86336


namespace sin_cos_sixth_l86_86031

theorem sin_cos_sixth (θ : ℝ) (h : Real.sin (2 * θ) = 1 / 3) :
  Real.sin θ ^ 6 + Real.cos θ ^ 6 = 11 / 12 :=
sorry

end sin_cos_sixth_l86_86031


namespace smallest_k_l86_86265

theorem smallest_k (k : ℕ) : 
  (∀ x, x ∈ [13, 7, 3, 5] → k % x = 1) ∧ k > 1 → k = 1366 :=
by
  sorry

end smallest_k_l86_86265


namespace toms_dog_age_is_twelve_l86_86950

-- Definitions based on given conditions
def toms_cat_age : ℕ := 8
def toms_rabbit_age : ℕ := toms_cat_age / 2
def toms_dog_age : ℕ := 3 * toms_rabbit_age

-- The statement to be proved
theorem toms_dog_age_is_twelve : toms_dog_age = 12 := by
  sorry

end toms_dog_age_is_twelve_l86_86950


namespace pie_piece_cost_l86_86484

theorem pie_piece_cost (pieces_per_pie : ℕ) (pies_per_hour : ℕ) (total_earnings : ℝ) :
  pieces_per_pie = 3 → pies_per_hour = 12 → total_earnings = 138 →
  (total_earnings / (pieces_per_pie * pies_per_hour)) = 3.83 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  sorry

end pie_piece_cost_l86_86484


namespace domain_shift_l86_86237

theorem domain_shift (f : ℝ → ℝ) :
  {x : ℝ | 1 ≤ x ∧ x ≤ 2} = {x | -2 ≤ x ∧ x ≤ -1} →
  {x : ℝ | ∃ y : ℝ, x = y - 1 ∧ 1 ≤ y ∧ y ≤ 2} =
  {x : ℝ | ∃ y : ℝ, x = y + 2 ∧ -2 ≤ y ∧ y ≤ -1} :=
by
  sorry

end domain_shift_l86_86237


namespace speed_in_still_water_l86_86508

-- Define the given conditions
def upstream_speed : ℝ := 32
def downstream_speed : ℝ := 48

-- State the theorem to be proven
theorem speed_in_still_water : (upstream_speed + downstream_speed) / 2 = 40 := by
  -- Proof omitted
  sorry

end speed_in_still_water_l86_86508


namespace pen_tip_movement_l86_86431

-- Definitions for the conditions
def condition_a := "Point movement becomes a line"
def condition_b := "Line movement becomes a surface"
def condition_c := "Surface movement becomes a solid"
def condition_d := "Intersection of surfaces results in a line"

-- The main statement we need to prove
theorem pen_tip_movement (phenomenon : String) : 
  phenomenon = "the pen tip quickly sliding on the paper to write the number 6" →
  condition_a = "Point movement becomes a line" :=
by
  intros
  sorry

end pen_tip_movement_l86_86431


namespace car_speed_first_hour_l86_86670

-- Definitions based on the conditions in the problem
noncomputable def speed_second_hour := 30
noncomputable def average_speed := 45
noncomputable def total_time := 2

-- Assertion based on the problem's question and correct answer
theorem car_speed_first_hour: ∃ (x : ℕ), (average_speed * total_time) = (x + speed_second_hour) ∧ x = 60 :=
by
  sorry

end car_speed_first_hour_l86_86670


namespace clea_ride_escalator_time_l86_86605

def clea_time_not_walking (x k y : ℝ) : Prop :=
  60 * x = y ∧ 24 * (x + k) = y ∧ 1.5 * x = k ∧ 40 = y / k

theorem clea_ride_escalator_time :
  ∀ (x y k : ℝ), 60 * x = y → 24 * (x + k) = y → (1.5 * x = k) → 40 = y / k :=
by
  intros x y k H1 H2 H3
  sorry

end clea_ride_escalator_time_l86_86605


namespace smallest_positive_number_is_correct_l86_86871

noncomputable def smallest_positive_number : ℝ := 20 - 5 * Real.sqrt 15

theorem smallest_positive_number_is_correct :
  ∀ n,
    (n = 12 - 3 * Real.sqrt 12 ∨ n = 3 * Real.sqrt 12 - 11 ∨ n = 20 - 5 * Real.sqrt 15 ∨ n = 55 - 11 * Real.sqrt 30 ∨ n = 11 * Real.sqrt 30 - 55) →
    n > 0 → smallest_positive_number ≤ n :=
by
  sorry

end smallest_positive_number_is_correct_l86_86871


namespace LanceCents_l86_86874

noncomputable def MargaretCents : ℕ := 75
noncomputable def GuyCents : ℕ := 60
noncomputable def BillCents : ℕ := 60
noncomputable def TotalCents : ℕ := 265

theorem LanceCents (lanceCents : ℕ) :
  MargaretCents + GuyCents + BillCents + lanceCents = TotalCents → lanceCents = 70 :=
by
  intros
  sorry

end LanceCents_l86_86874


namespace jan_keeps_on_hand_l86_86650

theorem jan_keeps_on_hand (total_length : ℕ) (section_length : ℕ) (friend_fraction : ℚ) (storage_fraction : ℚ) 
  (total_sections : ℕ) (sections_to_friend : ℕ) (remaining_sections : ℕ) (sections_in_storage : ℕ) (sections_on_hand : ℕ) :
  total_length = 1000 → section_length = 25 → friend_fraction = 1 / 4 → storage_fraction = 1 / 2 →
  total_sections = total_length / section_length →
  sections_to_friend = friend_fraction * total_sections →
  remaining_sections = total_sections - sections_to_friend →
  sections_in_storage = storage_fraction * remaining_sections →
  sections_on_hand = remaining_sections - sections_in_storage →
  sections_on_hand = 15 :=
by sorry

end jan_keeps_on_hand_l86_86650


namespace arithmetic_sequence_product_l86_86630

theorem arithmetic_sequence_product
  (a d : ℤ)
  (h1 : a + 5 * d = 17)
  (h2 : d = 2) :
  (a + 2 * d) * (a + 3 * d) = 143 :=
by
  sorry

end arithmetic_sequence_product_l86_86630


namespace card_stack_partition_l86_86211

theorem card_stack_partition (n k : ℕ) (cards : Multiset ℕ) (h1 : ∀ x ∈ cards, x ∈ Finset.range (n + 1)) (h2 : cards.sum = k * n!) :
  ∃ stacks : List (Multiset ℕ), stacks.length = k ∧ ∀ stack ∈ stacks, stack.sum = n! :=
sorry

end card_stack_partition_l86_86211


namespace three_in_A_even_not_in_A_l86_86213

def A : Set ℤ := {x | ∃ m n : ℤ, x = m^2 - n^2}

-- (1) Prove that 3 ∈ A
theorem three_in_A : 3 ∈ A :=
sorry

-- (2) Prove that ∀ k ∈ ℤ, 4k - 2 ∉ A
theorem even_not_in_A (k : ℤ) : (4 * k - 2) ∉ A :=
sorry

end three_in_A_even_not_in_A_l86_86213


namespace substitution_result_l86_86434

theorem substitution_result (x y : ℝ) (h1 : y = 2 * x + 1) (h2 : 5 * x - 2 * y = 7) : 5 * x - 4 * x - 2 = 7 :=
by
  sorry

end substitution_result_l86_86434


namespace product_neg_int_add_five_l86_86304

theorem product_neg_int_add_five:
  let x := -11 
  let y := -8 
  x * y + 5 = 93 :=
by
  -- Proof omitted
  sorry

end product_neg_int_add_five_l86_86304


namespace rick_group_division_l86_86154

theorem rick_group_division :
  ∀ (total_books : ℕ), total_books = 400 → 
  (∃ n : ℕ, (∀ (books_per_category : ℕ) (divisions : ℕ), books_per_category = total_books / (2 ^ divisions) → books_per_category = 25 → divisions = n) ∧ n = 4) :=
by
  sorry

end rick_group_division_l86_86154


namespace germs_left_after_sprays_l86_86011

-- Define the percentages as real numbers
def S1 : ℝ := 0.50 -- 50%
def S2 : ℝ := 0.35 -- 35%
def S3 : ℝ := 0.20 -- 20%
def S4 : ℝ := 0.10 -- 10%

-- Define the overlaps as real numbers
def overlap12 : ℝ := 0.10 -- between S1 and S2
def overlap23 : ℝ := 0.07 -- between S2 and S3
def overlap34 : ℝ := 0.05 -- between S3 and S4
def overlap13 : ℝ := 0.03 -- between S1 and S3
def overlap14 : ℝ := 0.02 -- between S1 and S4

theorem germs_left_after_sprays :
  let total_killed := S1 + S2 + S3 + S4
  let total_overlap := overlap12 + overlap23 + overlap34 + overlap13 + overlap14
  let adjusted_overlap := overlap12 + overlap23 + overlap34
  let effective_killed := total_killed - adjusted_overlap
  let percentage_left := 1.0 - effective_killed
  percentage_left = 0.07 := by
  -- proof steps to be inserted here
  sorry

end germs_left_after_sprays_l86_86011


namespace length_AB_l86_86248

-- Definitions and conditions
variables (R r a : ℝ) (hR : R > r) (BC_eq_a : BC = a) (r_eq_4 : r = 4)

-- Length of AB
theorem length_AB (AB : ℝ) : AB = a * Real.sqrt (R / (R - 4)) :=
sorry

end length_AB_l86_86248


namespace value_of_fraction_l86_86598

theorem value_of_fraction (a b c d e f : ℚ) (h1 : a / b = 1 / 3) (h2 : c / d = 1 / 3) (h3 : e / f = 1 / 3) :
  (3 * a - 2 * c + e) / (3 * b - 2 * d + f) = 1 / 3 :=
by
  sorry

end value_of_fraction_l86_86598


namespace a_finishes_race_in_t_seconds_l86_86560

theorem a_finishes_race_in_t_seconds 
  (time_B : ℝ := 45)
  (dist_B : ℝ := 100)
  (dist_A_wins_by : ℝ := 20)
  (total_dist : ℝ := 100)
  : ∃ t : ℝ, t = 36 := 
  sorry

end a_finishes_race_in_t_seconds_l86_86560


namespace prob_sum_to_3_three_dice_correct_l86_86299

def prob_sum_to_3_three_dice (sum : ℕ) (dice_count : ℕ) (dice_faces : Finset ℕ) : ℚ :=
  if sum = 3 ∧ dice_count = 3 ∧ dice_faces = {1, 2, 3, 4, 5, 6} then (1 : ℚ) / 216 else 0

theorem prob_sum_to_3_three_dice_correct :
  prob_sum_to_3_three_dice 3 3 {1, 2, 3, 4, 5, 6} = (1 : ℚ) / 216 := 
by
  sorry

end prob_sum_to_3_three_dice_correct_l86_86299


namespace flavors_needed_this_year_l86_86648

def num_flavors_total : ℕ := 100

def num_flavors_two_years_ago : ℕ := num_flavors_total / 4

def num_flavors_last_year : ℕ := 2 * num_flavors_two_years_ago

def num_flavors_tried_so_far : ℕ := num_flavors_two_years_ago + num_flavors_last_year

theorem flavors_needed_this_year : 
  (num_flavors_total - num_flavors_tried_so_far) = 25 := by {
  sorry
}

end flavors_needed_this_year_l86_86648


namespace abs_diff_kth_power_l86_86881

theorem abs_diff_kth_power (k : ℕ) (a b : ℤ) (x y : ℤ)
  (hk : 2 ≤ k)
  (ha : a ≠ 0) (hb : b ≠ 0)
  (hab_odd : (a + b) % 2 = 1)
  (hxy : 0 < |x - y| ∧ |x - y| ≤ 2)
  (h_eq : a^k * x - b^k * y = a - b) :
  ∃ m : ℤ, |a - b| = m^k :=
sorry

end abs_diff_kth_power_l86_86881


namespace find_a_l86_86196

open Set

noncomputable def A : Set ℝ := {x | x^2 - 2 * x - 8 = 0}
noncomputable def B (a : ℝ) : Set ℝ := {x | x^2 + a * x + a^2 - 12 = 0}

theorem find_a (a : ℝ) : (A ∪ (B a) = A) ↔ (a = -2 ∨ a ≥ 4 ∨ a < -4) := by
  sorry

end find_a_l86_86196


namespace parallel_lines_solution_l86_86517

theorem parallel_lines_solution (a : ℝ) :
  (∀ x y : ℝ, (x + a * y + 6 = 0) → (a - 2) * x + 3 * y + 2 * a = 0) → (a = -1) :=
by
  intro h
  -- Add more formal argument insights if needed
  sorry

end parallel_lines_solution_l86_86517


namespace problem_statement_l86_86536

theorem problem_statement :
  (1 / 3 * 1 / 6 * P = (1 / 4 * 1 / 8 * 64) + (1 / 5 * 1 / 10 * 100)) → 
  P = 72 :=
by
  sorry

end problem_statement_l86_86536


namespace fraction_value_l86_86546

theorem fraction_value : (4 * 5) / 10 = 2 := by
  sorry

end fraction_value_l86_86546


namespace units_digit_of_product_l86_86843

-- Definitions for units digit patterns for powers of 5 and 7
def units_digit (n : ℕ) : ℕ := n % 10

def power5_units_digit := 5
def power7_units_cycle := [7, 9, 3, 1]

-- Statement of the problem
theorem units_digit_of_product :
  units_digit ((5 ^ 3) * (7 ^ 52)) = 5 :=
by
  sorry

end units_digit_of_product_l86_86843


namespace cupric_cyanide_formed_l86_86017

-- Definition of the problem
def formonitrile : ℕ := 6
def copper_sulfate : ℕ := 3
def sulfuric_acid : ℕ := 3

-- Stoichiometry from the balanced equation
def stoichiometry (hcn mol_multiplier: ℕ): ℕ := 
  (hcn / mol_multiplier)

theorem cupric_cyanide_formed :
  stoichiometry formonitrile 2 = 3 := 
sorry

end cupric_cyanide_formed_l86_86017


namespace residue_mod_neg_935_mod_24_l86_86566

theorem residue_mod_neg_935_mod_24 : (-935) % 24 = 1 :=
by
  sorry

end residue_mod_neg_935_mod_24_l86_86566


namespace cab_company_charge_l86_86886

-- Defining the conditions
def total_cost : ℝ := 23
def base_price : ℝ := 3
def distance_to_hospital : ℝ := 5

-- Theorem stating the cost per mile
theorem cab_company_charge : 
  (total_cost - base_price) / distance_to_hospital = 4 :=
by
  -- Proof is omitted
  sorry

end cab_company_charge_l86_86886


namespace min_value_y1_minus_4y2_l86_86550

/-- 
Suppose a parabola C : y^2 = 4x intersects at points A(x1, y1) and B(x2, y2) with a line 
passing through its focus. Given that A is in the first quadrant, 
the minimum value of |y1 - 4y2| is 8.
--/
theorem min_value_y1_minus_4y2 (x1 y1 x2 y2 : ℝ) 
  (h1 : y1^2 = 4 * x1) 
  (h2 : y2^2 = 4 * x2)
  (h3 : x1 > 0) (h4 : y1 > 0) 
  (focus : (1, 0) ∈ {(x, y) | y^2 = 4 * x}) : 
  (|y1 - 4 * y2|) ≥ 8 :=
sorry

end min_value_y1_minus_4y2_l86_86550


namespace area_of_rectangle_l86_86798

theorem area_of_rectangle (a b : ℝ) (area : ℝ) 
(h1 : a = 5.9) 
(h2 : b = 3) 
(h3 : area = a * b) : 
area = 17.7 := 
by 
  -- proof goes here
  sorry

-- Definitions and conditions alignment:
-- a represents one side of the rectangle.
-- b represents the other side of the rectangle.
-- area represents the area of the rectangle.
-- h1: a = 5.9 corresponds to the first condition.
-- h2: b = 3 corresponds to the second condition.
-- h3: area = a * b connects the conditions to the formula to find the area.
-- The goal is to show that area = 17.7, which matches the correct answer.

end area_of_rectangle_l86_86798


namespace james_sheets_of_paper_l86_86855

noncomputable def sheets_of_paper (books : ℕ) (pages_per_book : ℕ) (pages_per_side : ℕ) (sides_per_sheet : ℕ) : ℕ :=
  (books * pages_per_book) / (pages_per_side * sides_per_sheet)

theorem james_sheets_of_paper :
  sheets_of_paper 2 600 4 2 = 150 :=
by
  sorry

end james_sheets_of_paper_l86_86855


namespace not_prime_sum_l86_86969

theorem not_prime_sum (a b c : ℕ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_square : ∃ k : ℕ, a^2 - b * c = k^2) : ¬ Nat.Prime (2 * a + b + c) := 
sorry

end not_prime_sum_l86_86969


namespace additional_charge_per_2_5_mile_l86_86094

theorem additional_charge_per_2_5_mile (x : ℝ) : 
  (∀ (total_charge distance charge_per_segment initial_fee : ℝ),
    total_charge = 5.65 →
    initial_fee = 2.5 →
    distance = 3.6 →
    charge_per_segment = (3.6 / (2/5)) →
    total_charge = initial_fee + charge_per_segment * x → 
    x = 0.35) :=
by
  intros total_charge distance charge_per_segment initial_fee
  intros h_total_charge h_initial_fee h_distance h_charge_per_segment h_eq
  sorry

end additional_charge_per_2_5_mile_l86_86094


namespace square_area_l86_86587

-- Definition of the vertices' coordinates
def y_coords := ({-3, 2, 2, -3} : Set ℤ)
def x_coords_when_y2 := ({0, 5} : Set ℤ)

-- The statement we need to prove
theorem square_area (h1 : y_coords = {-3, 2, 2, -3}) 
                     (h2 : x_coords_when_y2 = {0, 5}) : 
                     ∃ s : ℤ, s^2 = 25 :=
by
  sorry

end square_area_l86_86587


namespace sufficient_not_necessary_condition_l86_86792

theorem sufficient_not_necessary_condition (x y : ℝ) : 
  (x - y) * x^4 < 0 → x < y ∧ ¬(x < y → (x - y) * x^4 < 0) := 
sorry

end sufficient_not_necessary_condition_l86_86792


namespace sand_removal_l86_86137

theorem sand_removal :
  let initial_weight := (8 / 3 : ℚ)
  let first_removal := (1 / 4 : ℚ)
  let second_removal := (5 / 6 : ℚ)
  initial_weight - (first_removal + second_removal) = (13 / 12 : ℚ) := by
  -- sorry is used here to skip the proof as instructed
  sorry

end sand_removal_l86_86137


namespace total_income_in_june_l86_86228

-- Establishing the conditions
def daily_production : ℕ := 200
def days_in_june : ℕ := 30
def price_per_gallon : ℝ := 3.55

-- Defining total milk production in June as a function of daily production and days in June
def total_milk_production_in_june : ℕ :=
  daily_production * days_in_june

-- Defining total income as a function of milk production and price per gallon
def total_income (milk_production : ℕ) (price : ℝ) : ℝ :=
  milk_production * price

-- Stating the theorem that we need to prove
theorem total_income_in_june :
  total_income total_milk_production_in_june price_per_gallon = 21300 := 
sorry

end total_income_in_june_l86_86228


namespace trip_total_time_l86_86909

theorem trip_total_time 
  (x : ℕ) 
  (h1 : 30 * 5 = 150) 
  (h2 : 42 * x + 150 = 38 * (x + 5)) 
  (h3 : 38 = (150 + 42 * x) / (5 + x)) : 
  5 + x = 15 := by
  sorry

end trip_total_time_l86_86909


namespace discount_is_10_percent_l86_86884

variable (C : ℝ)  -- Cost of the item
variable (S S' : ℝ)  -- Selling prices with and without discount

-- Conditions
def condition1 : Prop := S = 1.20 * C
def condition2 : Prop := S' = 1.30 * C

-- The proposition to prove
theorem discount_is_10_percent (h1 : condition1 C S) (h2 : condition2 C S') : S' - S = 0.10 * C := by
  sorry

end discount_is_10_percent_l86_86884


namespace cypress_tree_price_l86_86037

def amount_per_cypress_tree (C : ℕ) : Prop :=
  let cabin_price := 129000
  let cash := 150
  let cypress_count := 20
  let pine_count := 600
  let maple_count := 24
  let pine_price := 200
  let maple_price := 300
  let leftover_cash := 350
  let total_amount_raised := cabin_price - cash + leftover_cash
  let total_pine_maple := (pine_count * pine_price) + (maple_count * maple_price)
  let total_cypress := total_amount_raised - total_pine_maple
  let cypress_sale_price := total_cypress / cypress_count
  cypress_sale_price = C

theorem cypress_tree_price : amount_per_cypress_tree 100 :=
by {
  -- Proof skipped
  sorry
}

end cypress_tree_price_l86_86037


namespace total_number_of_animals_l86_86912

-- Definitions based on conditions
def number_of_females : ℕ := 35
def males_outnumber_females_by : ℕ := 7
def number_of_males : ℕ := number_of_females + males_outnumber_females_by

-- Theorem to prove the total number of animals
theorem total_number_of_animals :
  number_of_females + number_of_males = 77 := by
  sorry

end total_number_of_animals_l86_86912


namespace area_three_layers_l86_86392

def total_area_rugs : ℝ := 200
def floor_covered_area : ℝ := 140
def exactly_two_layers_area : ℝ := 24

theorem area_three_layers : (2 * (200 - 140 - 24) / 2 = 2 * 18) := 
by admit -- since we're instructed to skip the proof

end area_three_layers_l86_86392


namespace max_abs_sum_l86_86462

theorem max_abs_sum (x y : ℝ) (h : x^2 + y^2 = 4) : |x| + |y| ≤ 2 * Real.sqrt 2 :=
by
  sorry

end max_abs_sum_l86_86462


namespace farm_field_area_l86_86697

variable (A D : ℕ)

theorem farm_field_area
  (h1 : 160 * D = A)
  (h2 : 85 * (D + 2) + 40 = A) :
  A = 480 :=
by
  sorry

end farm_field_area_l86_86697


namespace inequality_proof_l86_86035

theorem inequality_proof (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : x + y ≤ (y^2 / x) + (x^2 / y) :=
sorry

end inequality_proof_l86_86035


namespace find_radius_l86_86291

theorem find_radius (abbc: ℝ) (adbd: ℝ) (bccc: ℝ) (dcdd: ℝ) (R: ℝ)
  (h1: abbc = 4) (h2: adbd = 4) (h3: bccc = 2) (h4: dcdd = 1) :
  R = 5 :=
sorry

end find_radius_l86_86291


namespace olivia_card_value_l86_86639

theorem olivia_card_value (x : ℝ) (hx1 : 90 < x ∧ x < 180)
  (h_sin_pos : Real.sin x > 0) (h_cos_neg : Real.cos x < 0) (h_tan_neg : Real.tan x < 0)
  (h_olivia_distinguish : ∀ (a b c : ℝ), 
    (a = Real.sin x ∨ a = Real.cos x ∨ a = Real.tan x) →
    (b = Real.sin x ∨ b = Real.cos x ∨ b = Real.tan x) →
    (c = Real.sin x ∨ c = Real.cos x ∨ c = Real.tan x) →
    (a ≠ b ∧ b ≠ c ∧ c ≠ a) →
    (a = Real.sin x ∨ a = Real.cos x ∨ a = Real.tan x) →
    (b = Real.sin x ∨ b = Real.cos x ∨ b = Real.tan x) →
    (c = Real.sin x ∨ c = Real.cos x ∨ c = Real.tan x) →
    (∃! a, a = Real.sin x ∨ a = Real.cos x ∨ a = Real.tan x)) :
  Real.sin 135 = Real.cos 45 := 
sorry

end olivia_card_value_l86_86639


namespace alice_travel_time_l86_86523

theorem alice_travel_time (distance_AB : ℝ) (bob_speed : ℝ) (alice_speed : ℝ) (max_time_diff_hr : ℝ) (time_conversion : ℝ) :
  distance_AB = 60 →
  bob_speed = 40 →
  alice_speed = 60 →
  max_time_diff_hr = 0.5 →
  time_conversion = 60 →
  max_time_diff_hr * time_conversion = 30 :=
by
  intros
  sorry

end alice_travel_time_l86_86523


namespace sum_binomial_coefficients_l86_86090

theorem sum_binomial_coefficients (a b : ℕ) (h1 : a = 2^3) (h2 : b = (2 + 1)^3) : a + b = 35 :=
by
  sorry

end sum_binomial_coefficients_l86_86090


namespace Jason_toys_correct_l86_86303

variable (R Jn Js : ℕ)

def Rachel_toys : ℕ := 1

def John_toys (R : ℕ) : ℕ := R + 6

def Jason_toys (Jn : ℕ) : ℕ := 3 * Jn

theorem Jason_toys_correct (hR : R = 1) (hJn : Jn = John_toys R) (hJs : Js = Jason_toys Jn) : Js = 21 :=
by
  sorry

end Jason_toys_correct_l86_86303


namespace true_statement_l86_86540

def statement_i (i : ℕ) (n : ℕ) : Prop := 
  (i = (n - 1))

theorem true_statement :
  ∃! n : ℕ, (n ≤ 100 ∧ ∀ i, (i ≠ n - 1) → statement_i i n = false) ∧ statement_i (n - 1) n = true :=
by
  sorry

end true_statement_l86_86540


namespace sequence_term_l86_86755

noncomputable def a_n (n : ℕ) : ℕ :=
  if n = 1 then 3 else 4 * n - 2

def S_n (n : ℕ) : ℕ :=
  2 * n^2 + 1

theorem sequence_term (n : ℕ) : a_n n = if n = 1 then S_n 1 else S_n n - S_n (n - 1) :=
by 
  sorry

end sequence_term_l86_86755


namespace percentage_of_students_who_received_certificates_l86_86669

theorem percentage_of_students_who_received_certificates
  (total_boys : ℕ)
  (total_girls : ℕ)
  (perc_boys_certificates : ℝ)
  (perc_girls_certificates : ℝ)
  (h1 : total_boys = 30)
  (h2 : total_girls = 20)
  (h3 : perc_boys_certificates = 0.1)
  (h4 : perc_girls_certificates = 0.2)
  : (3 + 4) / (30 + 20) * 100 = 14 :=
by
  sorry

end percentage_of_students_who_received_certificates_l86_86669


namespace non_degenerate_ellipse_l86_86838

theorem non_degenerate_ellipse (x y k : ℝ) : (∃ k, (2 * x^2 + 9 * y^2 - 12 * x - 27 * y = k) → k > -135 / 4) := sorry

end non_degenerate_ellipse_l86_86838


namespace sin2_cos3_tan4_lt_zero_l86_86782

theorem sin2_cos3_tan4_lt_zero (h1 : Real.sin 2 > 0) (h2 : Real.cos 3 < 0) (h3 : Real.tan 4 > 0) : Real.sin 2 * Real.cos 3 * Real.tan 4 < 0 :=
sorry

end sin2_cos3_tan4_lt_zero_l86_86782


namespace knights_statements_l86_86507

theorem knights_statements (r ℓ : Nat) (hr : r ≥ 2) (hℓ : ℓ ≥ 2)
  (h : 2 * r * ℓ = 230) :
  (r + ℓ) * (r + ℓ - 1) - 230 = 526 :=
by
  sorry

end knights_statements_l86_86507


namespace largest_sum_36_l86_86260

theorem largest_sum_36 : ∃ n : ℕ, ∃ a : ℕ, (n * a + (n * (n - 1)) / 2 = 36) ∧ ∀ m : ℕ, (m * a + (m * (m - 1)) / 2 = 36) → m ≤ 8 :=
by
  sorry

end largest_sum_36_l86_86260


namespace product_sequence_eq_l86_86577

theorem product_sequence_eq :
  let seq := [ (1 : ℚ) / 2, 4 / 1, 1 / 8, 16 / 1, 1 / 32, 64 / 1,
               1 / 128, 256 / 1, 1 / 512, 1024 / 1, 1 / 2048, 4096 / 1 ]
  (seq.prod) * (3 / 4) = 1536 := by 
  -- expand and simplify the series of products
  sorry 

end product_sequence_eq_l86_86577


namespace secondary_spermatocytes_can_contain_two_y_chromosomes_l86_86264

-- Definitions corresponding to the conditions
def primary_spermatocytes_first_meiotic_division_contains_y (n : Nat) : Prop := n = 1
def spermatogonia_metaphase_mitosis_contains_y (n : Nat) : Prop := n = 1
def secondary_spermatocytes_second_meiotic_division_contains_y (n : Nat) : Prop := n = 0 ∨ n = 2
def spermatogonia_prophase_mitosis_contains_y (n : Nat) : Prop := n = 1

-- The theorem statement equivalent to the given math problem
theorem secondary_spermatocytes_can_contain_two_y_chromosomes :
  ∃ n, (secondary_spermatocytes_second_meiotic_division_contains_y n ∧ n = 2) :=
sorry

end secondary_spermatocytes_can_contain_two_y_chromosomes_l86_86264


namespace unique_solution_l86_86223

-- Given conditions in the problem:
def prime (p : ℕ) : Prop := Nat.Prime p
def is_solution (p n k : ℕ) : Prop :=
  3 ^ p + 4 ^ p = n ^ k ∧ k > 1 ∧ prime p

-- The only solution:
theorem unique_solution (p n k : ℕ) :
  is_solution p n k → (p, n, k) = (2, 5, 2) := 
by
  sorry

end unique_solution_l86_86223


namespace frequency_of_8th_group_l86_86186

theorem frequency_of_8th_group :
  let sample_size := 100
  let freq1 := 15
  let freq2 := 17
  let freq3 := 11
  let freq4 := 13
  let freq_5_to_7 := 0.32 * sample_size
  let total_freq_1_to_4 := freq1 + freq2 + freq3 + freq4
  let remaining_freq := sample_size - total_freq_1_to_4
  let freq8 := remaining_freq - freq_5_to_7
  (freq8 / sample_size = 0.12) :=
by
  sorry

end frequency_of_8th_group_l86_86186


namespace tan_difference_of_angle_l86_86345

noncomputable def point_on_terminal_side (θ : ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ (2, 3) = (k * Real.cos θ, k * Real.sin θ)

theorem tan_difference_of_angle (θ : ℝ) (hθ : point_on_terminal_side θ) :
  Real.tan (θ - Real.pi / 4) = 1 / 5 :=
sorry

end tan_difference_of_angle_l86_86345


namespace Adam_smiley_count_l86_86652

theorem Adam_smiley_count :
  ∃ (adam mojmir petr pavel : ℕ), adam + mojmir + petr + pavel = 52 ∧
  petr + pavel = 33 ∧ adam >= 1 ∧ mojmir >= 1 ∧ petr >= 1 ∧ pavel >= 1 ∧
  mojmir > max petr pavel ∧ adam = 1 :=
by
  sorry

end Adam_smiley_count_l86_86652


namespace range_of_reciprocal_sum_l86_86913

theorem range_of_reciprocal_sum {x y : ℝ} (hx : 0 < x) (hy : 0 < y) (h : x + 4 * y + 1 / x + 1 / y = 10) :
  1 ≤ 1 / x + 1 / y ∧ 1 / x + 1 / y ≤ 9 := 
sorry

end range_of_reciprocal_sum_l86_86913


namespace complementary_angles_positive_difference_l86_86861

theorem complementary_angles_positive_difference :
  ∀ (θ₁ θ₂ : ℝ), (θ₁ + θ₂ = 90) → (θ₁ = 3 * θ₂) → (|θ₁ - θ₂| = 45) :=
by
  intros θ₁ θ₂ h₁ h₂
  sorry

end complementary_angles_positive_difference_l86_86861


namespace sum_of_possible_values_l86_86666

theorem sum_of_possible_values (x y : ℝ) (h : x * y - x / y^3 - y / x^3 = 2) :
  (x - 2) * (y - 2) = 6 ∨ (x - 2) * (y - 2) = 9 →
  (if (x - 2) * (y - 2) = 6 then 6 else 0) + (if (x - 2) * (y - 2) = 9 then 9 else 0) = 15 :=
by
  sorry

end sum_of_possible_values_l86_86666


namespace simplify_expression1_simplify_expression2_l86_86182

-- Problem 1 statement
theorem simplify_expression1 (a b : ℤ) : 2 * (2 * b - 3 * a) + 3 * (2 * a - 3 * b) = -5 * b :=
  by
  sorry

-- Problem 2 statement
theorem simplify_expression2 (a b : ℤ) : 4 * a^2 + 2 * (3 * a * b - 2 * a^2) - (7 * a * b - 1) = -a * b + 1 :=
  by
  sorry

end simplify_expression1_simplify_expression2_l86_86182


namespace infinite_solutions_ax2_by2_eq_z3_l86_86902

theorem infinite_solutions_ax2_by2_eq_z3 
  (a b : ℤ) 
  (coprime_ab : Int.gcd a b = 1) :
  ∃ (x y z : ℤ), (∀ n : ℤ, ∃ (x y z : ℤ), a * x^2 + b * y^2 = z^3 
  ∧ Int.gcd x y = 1) := 
sorry

end infinite_solutions_ax2_by2_eq_z3_l86_86902


namespace range_of_a_for_inequality_l86_86879

theorem range_of_a_for_inequality : 
  ∃ a : ℝ, (∀ x : ℤ, (a * x - 1) ^ 2 < x ^ 2) ↔ 
    (a > -3 / 2 ∧ a ≤ -4 / 3) ∨ (4 / 3 ≤ a ∧ a < 3 / 2) :=
by
  sorry

end range_of_a_for_inequality_l86_86879


namespace inequality_solution_l86_86751

theorem inequality_solution (x : ℝ) : (2 * x - 1) / 3 ≥ 1 → x ≥ 2 := by
  sorry

end inequality_solution_l86_86751


namespace find_g_of_7_l86_86282

theorem find_g_of_7 (g : ℝ → ℝ) (h : ∀ x : ℝ, g (3 * x - 8) = 2 * x + 11) : g 7 = 21 :=
by
  sorry

end find_g_of_7_l86_86282


namespace angle_complement_supplement_l86_86847

theorem angle_complement_supplement (x : ℝ) (h1 : 180 - x = 4 * (90 - x)) : x = 60 :=
sorry

end angle_complement_supplement_l86_86847


namespace value_of_c_in_base8_perfect_cube_l86_86315

theorem value_of_c_in_base8_perfect_cube (c : ℕ) (h : 0 ≤ c ∧ c < 8) :
  4 * 8^2 + c * 8 + 3 = x^3 → c = 0 := by
  sorry

end value_of_c_in_base8_perfect_cube_l86_86315


namespace determinant_A_l86_86057

def A : Matrix (Fin 3) (Fin 3) ℤ :=
  ![![2, -1, 5], ![0, 4, -2], ![3, 0, 1]]

theorem determinant_A : Matrix.det A = -46 := by
  sorry

end determinant_A_l86_86057


namespace domain_of_log_function_l86_86510

noncomputable def f (x : ℝ) : ℝ := Real.log (2 * x - 1)

theorem domain_of_log_function : {
  x : ℝ // ∃ y : ℝ, f y = x
} = { x : ℝ | x > 1 / 2 } := by
sorry

end domain_of_log_function_l86_86510


namespace correct_options_l86_86191

theorem correct_options (a b : ℝ) (h : a > 0) (ha : a^2 = 4 * b) :
  ((a^2 - b^2 ≤ 4) ∧ (a^2 + 1 / b ≥ 4) ∧ (¬ (∃ x1 x2, x1 * x2 > 0 ∧ x^2 + a * x - b < 0)) ∧ 
  (∀ (x1 x2 : ℝ), |x1 - x2| = 4 → x^2 + a * x + b < 4 → 4 = 4)) :=
sorry

end correct_options_l86_86191


namespace parabola_shifted_l86_86190

-- Define the original parabola
def originalParabola (x : ℝ) : ℝ := (x + 2)^2 + 3

-- Shift the parabola by 3 units to the right
def shiftedRight (x : ℝ) : ℝ := originalParabola (x - 3)

-- Then shift the parabola 2 units down
def shiftedRightThenDown (x : ℝ) : ℝ := shiftedRight x - 2

-- The problem asks to prove that the final expression is equal to (x - 1)^2 + 1
theorem parabola_shifted (x : ℝ) : shiftedRightThenDown x = (x - 1)^2 + 1 :=
by
  sorry

end parabola_shifted_l86_86190


namespace line_passes_through_3_1_l86_86746

open Classical

noncomputable def line_passes_through_fixed_point (m x y : ℝ) : Prop :=
  (2 * m + 1) * x + (m + 1) * y - 7 * m - 4 = 0

theorem line_passes_through_3_1 (m : ℝ) :
  line_passes_through_fixed_point m 3 1 :=
by
  sorry

end line_passes_through_3_1_l86_86746


namespace perpendicular_vectors_m_value_l86_86321

theorem perpendicular_vectors_m_value
  (a : ℝ × ℝ := (1, 2))
  (b : ℝ × ℝ)
  (h_perpendicular : (a.1 * b.1 + a.2 * b.2) = 0) :
  b = (-2, 1) :=
by
  sorry

end perpendicular_vectors_m_value_l86_86321


namespace planting_rate_l86_86904

theorem planting_rate (total_acres : ℕ) (days : ℕ) (initial_tractors : ℕ) (initial_days : ℕ) (additional_tractors : ℕ) (additional_days : ℕ) :
  total_acres = 1700 →
  days = 5 →
  initial_tractors = 2 →
  initial_days = 2 →
  additional_tractors = 7 →
  additional_days = 3 →
  (total_acres / ((initial_tractors * initial_days) + (additional_tractors * additional_days))) = 68 :=
by
  sorry

end planting_rate_l86_86904


namespace depth_of_melted_ice_cream_l86_86463

theorem depth_of_melted_ice_cream
  (r_sphere : ℝ) (r_cylinder : ℝ) (V_sphere : ℝ)
  (h : ℝ)
  (sphere_volume_eq : V_sphere = (4 / 3) * Real.pi * r_sphere^3)
  (cylinder_volume_eq : V_sphere = Real.pi * r_cylinder^2 * h)
  (r_sphere_eq : r_sphere = 3)
  (r_cylinder_eq : r_cylinder = 9)
  : h = 4 / 9 :=
by
  -- Proof is omitted
  sorry

end depth_of_melted_ice_cream_l86_86463


namespace sum_of_D_coordinates_l86_86108

noncomputable def sum_of_coordinates_of_D (D : ℝ × ℝ) (M C : ℝ × ℝ) : ℝ :=
  D.1 + D.2

theorem sum_of_D_coordinates (D M C : ℝ × ℝ) (H_M_midpoint : M = ((C.1 + D.1) / 2, (C.2 + D.2) / 2)) 
                             (H_M_value : M = (5, 9)) (H_C_value : C = (11, 5)) : 
                             sum_of_coordinates_of_D D M C = 12 :=
sorry

end sum_of_D_coordinates_l86_86108


namespace smallest_three_digit_number_l86_86731

theorem smallest_three_digit_number (x : ℤ) (h1 : x - 7 % 7 = 0) (h2 : x - 8 % 8 = 0) (h3 : x - 9 % 9 = 0) : x = 504 := 
sorry

end smallest_three_digit_number_l86_86731


namespace angle_sum_triangle_l86_86535

theorem angle_sum_triangle (A B C : Type) (angle_A angle_B angle_C : ℝ) 
(h1 : angle_A = 45) (h2 : angle_B = 25) 
(h3 : angle_A + angle_B + angle_C = 180) : 
angle_C = 110 := 
sorry

end angle_sum_triangle_l86_86535


namespace circle_area_increase_l86_86283

theorem circle_area_increase (r : ℝ) (h : r > 0) :
  let new_radius := 1.5 * r
  let original_area := π * r^2
  let new_area := π * (new_radius)^2
  let increase := new_area - original_area
  let percentage_increase := (increase / original_area) * 100
  percentage_increase = 125 := 
by {
  -- The proof will be written here.
  sorry
}

end circle_area_increase_l86_86283


namespace number_power_eq_l86_86143

theorem number_power_eq (x : ℕ) (h : x^10 = 16^5) : x = 4 :=
by {
  -- Add supporting calculations here if needed
  sorry
}

end number_power_eq_l86_86143


namespace escalator_steps_l86_86708

theorem escalator_steps (T : ℝ) (E : ℝ) (N : ℝ) (h1 : N - 11 = 2 * (N - 29)) : N = 47 :=
by
  sorry

end escalator_steps_l86_86708


namespace find_number_l86_86553

theorem find_number (x : ℝ) (h : (x / 6) * 12 = 13) : x = 6.5 :=
by
  sorry

end find_number_l86_86553


namespace A_intersection_B_complement_l86_86654

noncomputable
def universal_set : Set ℝ := Set.univ

def set_A : Set ℝ := {x | x > 1}

def set_B : Set ℝ := {y | -1 < y ∧ y < 2}

def B_complement : Set ℝ := {y | y <= -1 ∨ y >= 2}

def intersection : Set ℝ := {x | x >= 2}

theorem A_intersection_B_complement :
  (set_A ∩ B_complement) = intersection :=
  sorry

end A_intersection_B_complement_l86_86654


namespace team_team_count_correct_l86_86448

/-- Number of ways to select a team of three students from 20,
    one for each subject: math, Russian language, and informatics. -/
def ways_to_form_team (n : ℕ) : ℕ :=
  if n ≥ 3 then n * (n - 1) * (n - 2) else 0

theorem team_team_count_correct : ways_to_form_team 20 = 6840 :=
by sorry

end team_team_count_correct_l86_86448


namespace circle_radius_5_l86_86442

theorem circle_radius_5 (k x y : ℝ) : x^2 + 8 * x + y^2 + 10 * y - k = 0 → (x + 4) ^ 2 + (y + 5) ^ 2 = 25 → k = -16 :=
by
  sorry

end circle_radius_5_l86_86442


namespace number_of_seniors_in_statistics_l86_86161

theorem number_of_seniors_in_statistics (total_students : ℕ) (half_enrolled_in_statistics : ℕ) (percentage_seniors : ℚ) (students_in_statistics seniors_in_statistics : ℕ) 
(h1 : total_students = 120)
(h2 : half_enrolled_in_statistics = total_students / 2)
(h3 : students_in_statistics = half_enrolled_in_statistics)
(h4 : percentage_seniors = 0.90)
(h5 : seniors_in_statistics = students_in_statistics * percentage_seniors) : 
seniors_in_statistics = 54 := 
by sorry

end number_of_seniors_in_statistics_l86_86161


namespace polygon_diagonals_l86_86529

theorem polygon_diagonals (n : ℕ) (h : n - 3 = 4) : n = 7 :=
sorry

end polygon_diagonals_l86_86529


namespace largest_and_smallest_values_quartic_real_roots_l86_86301

noncomputable def function_y (a b x : ℝ) : ℝ :=
  (4 * a^2 * x^2 + b^2 * (x^2 - 1)^2) / (x^2 + 1)^2

theorem largest_and_smallest_values (a b : ℝ) (h : a > b) :
  ∃ x y, function_y a b x = y^2 ∧ y = a ∧ y = b :=
by
  sorry

theorem quartic_real_roots (a b y : ℝ) (h₁ : a > b) (h₂ : y > b) (h₃ : y < a) :
  ∃ x₀ x₁ x₂ x₃, function_y a b x₀ = y^2 ∧ function_y a b x₁ = y^2 ∧ function_y a b x₂ = y^2 ∧ function_y a b x₃ = y^2 :=
by
  sorry

end largest_and_smallest_values_quartic_real_roots_l86_86301


namespace primes_sum_divisible_by_60_l86_86470

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem primes_sum_divisible_by_60 (p q r s : ℕ) 
  (hp : is_prime p) 
  (hq : is_prime q) 
  (hr : is_prime r) 
  (hs : is_prime s) 
  (h_cond1 : 5 < p) 
  (h_cond2 : p < q) 
  (h_cond3 : q < r) 
  (h_cond4 : r < s) 
  (h_cond5 : s < p + 10) : 
  (p + q + r + s) % 60 = 0 :=
sorry

end primes_sum_divisible_by_60_l86_86470


namespace quadratic_completion_l86_86591

noncomputable def sum_of_r_s (r s : ℝ) : ℝ := r + s

theorem quadratic_completion (x r s : ℝ) (h : 16 * x^2 - 64 * x - 144 = 0) :
  ((x + r)^2 = s) → sum_of_r_s r s = -7 :=
by
  sorry

end quadratic_completion_l86_86591


namespace greatest_integer_equality_l86_86696

theorem greatest_integer_equality (m : ℝ) (h : m ≥ 3) :
  Int.floor ((m * (m + 1)) / (2 * (2 * m - 1))) = Int.floor ((m + 1) / 4) :=
  sorry

end greatest_integer_equality_l86_86696


namespace tan_difference_l86_86930

theorem tan_difference (α β : ℝ) (h1 : Real.tan α = 3) (h2 : Real.tan β = 4 / 3) :
  Real.tan (α - β) = 1 / 3 :=
by
  sorry

end tan_difference_l86_86930


namespace sugar_percentage_in_new_solution_l86_86777

open Real

noncomputable def original_volume : ℝ := 450
noncomputable def original_sugar_percentage : ℝ := 20 / 100
noncomputable def added_sugar : ℝ := 7.5
noncomputable def added_water : ℝ := 20
noncomputable def added_kola : ℝ := 8.1
noncomputable def added_flavoring : ℝ := 2.3

noncomputable def original_sugar_amount : ℝ := original_volume * original_sugar_percentage
noncomputable def total_sugar_amount : ℝ := original_sugar_amount + added_sugar
noncomputable def new_total_volume : ℝ := original_volume + added_water + added_kola + added_flavoring + added_sugar
noncomputable def new_sugar_percentage : ℝ := (total_sugar_amount / new_total_volume) * 100

theorem sugar_percentage_in_new_solution : abs (new_sugar_percentage - 19.97) < 0.01 := sorry

end sugar_percentage_in_new_solution_l86_86777


namespace ratio_of_e_to_l_l86_86415

-- Define the conditions
def e (S : ℕ) : ℕ := 4 * S
def l (S : ℕ) : ℕ := 8 * S

-- Prove the main statement
theorem ratio_of_e_to_l (S : ℕ) (h_e : e S = 4 * S) (h_l : l S = 8 * S) : e S / gcd (e S) (l S) / l S / gcd (e S) (l S) = 1 / 2 := by
  sorry

end ratio_of_e_to_l_l86_86415


namespace amount_used_to_pay_l86_86162

noncomputable def the_cost_of_football : ℝ := 9.14
noncomputable def the_cost_of_baseball : ℝ := 6.81
noncomputable def the_change_received : ℝ := 4.05

theorem amount_used_to_pay : 
    (the_cost_of_football + the_cost_of_baseball + the_change_received) = 20.00 := 
by
  sorry

end amount_used_to_pay_l86_86162


namespace complex_expression_evaluation_l86_86226

theorem complex_expression_evaluation (z : ℂ) (h : z = 1 - I) :
  (z^2 - 2 * z) / (z - 1) = -2 * I :=
by
  sorry

end complex_expression_evaluation_l86_86226


namespace bruce_money_left_l86_86911

-- Definitions for the given values
def initial_amount : ℕ := 71
def shirt_cost : ℕ := 5
def number_of_shirts : ℕ := 5
def pants_cost : ℕ := 26

-- The theorem that Bruce has $20 left
theorem bruce_money_left : initial_amount - (shirt_cost * number_of_shirts + pants_cost) = 20 :=
by
  sorry

end bruce_money_left_l86_86911


namespace find_center_angle_l86_86456

noncomputable def pi : ℝ := Real.pi
/-- Given conditions from the math problem -/
def radius : ℝ := 12
def area : ℝ := 67.88571428571429

theorem find_center_angle (θ : ℝ) 
  (area_def : area = (θ / 360) * pi * radius ^ 2) : 
  θ = 54 :=
sorry

end find_center_angle_l86_86456


namespace find_m_l86_86836

theorem find_m (m : ℝ) : (∀ x : ℝ, x - m > 5 ↔ x > 2) → m = -3 :=
by
  sorry

end find_m_l86_86836


namespace temperature_difference_l86_86845

-- Define the temperatures given in the problem.
def T_noon : ℝ := 10
def T_midnight : ℝ := -150

-- State the theorem to prove the temperature difference.
theorem temperature_difference :
  T_noon - T_midnight = 160 :=
by
  -- We skip the proof and add sorry.
  sorry

end temperature_difference_l86_86845


namespace toys_secured_in_25_minutes_l86_86146

def net_toy_gain_per_minute (toys_mom_puts : ℕ) (toys_mia_takes : ℕ) : ℕ :=
  toys_mom_puts - toys_mia_takes

def total_minutes (total_toys : ℕ) (toys_mom_puts : ℕ) (toys_mia_takes : ℕ) : ℕ :=
  (total_toys - 1) / net_toy_gain_per_minute toys_mom_puts toys_mia_takes + 1

theorem toys_secured_in_25_minutes :
  total_minutes 50 5 3 = 25 :=
by
  sorry

end toys_secured_in_25_minutes_l86_86146


namespace eval_fraction_expression_l86_86479
noncomputable def inner_expr := 2 + 2
noncomputable def middle_expr := 2 + (1 / inner_expr)
noncomputable def outer_expr := 2 + (1 / middle_expr)

theorem eval_fraction_expression : outer_expr = 22 / 9 := by
  sorry

end eval_fraction_expression_l86_86479


namespace find_y_values_l86_86688

variable (x y : ℝ)

theorem find_y_values 
    (h1 : 3 * x^2 + 9 * x + 4 * y - 2 = 0)
    (h2 : 3 * x + 2 * y - 6 = 0) : 
    y^2 - 13 * y + 26 = 0 := by
  sorry

end find_y_values_l86_86688


namespace inverse_of_f_at_neg2_l86_86762

-- Define the function f
def f (x : ℝ) : ℝ := 2 * x + 1

-- Define the property of the inverse function we need to prove
theorem inverse_of_f_at_neg2 : f (-(3/2)) = -2 :=
  by
    -- Placeholder for the proof
    sorry

end inverse_of_f_at_neg2_l86_86762


namespace number_multiplied_by_approx_l86_86917

variable (X : ℝ)

theorem number_multiplied_by_approx (h : (0.0048 * X) / (0.05 * 0.1 * 0.004) = 840) : X = 3.5 :=
by
  sorry

end number_multiplied_by_approx_l86_86917


namespace period_pi_omega_l86_86188

noncomputable def f (x : ℝ) (ω : ℝ) : ℝ :=
  3 * (Real.sin (ω * x)) * (Real.cos (ω * x)) - 4 * (Real.cos (ω * x))^2

theorem period_pi_omega (ω : ℝ) (hω : ω > 0) (period_condition : ∀ x, f x ω = f (x + π) ω)
  (theta : ℝ) (h_f_theta : f theta ω = 1 / 2) :
  f (theta + π / 2) ω + f (theta - π / 4) ω = -13 / 2 :=
by
  sorry

end period_pi_omega_l86_86188


namespace expected_winnings_is_correct_l86_86204

noncomputable def peculiar_die_expected_winnings : ℝ :=
  (1/4) * 2 + (1/2) * 5 + (1/4) * (-10)

theorem expected_winnings_is_correct :
  peculiar_die_expected_winnings = 0.5 := by
  sorry

end expected_winnings_is_correct_l86_86204


namespace problem_probability_ao_drawn_second_l86_86932

def is_ao_drawn_second (pair : ℕ × ℕ) : Bool :=
  pair.snd = 3

def random_pairs : List (ℕ × ℕ) := [
  (1, 3), (2, 4), (1, 2), (3, 2), (4, 3), (1, 4), (2, 4), (3, 2), (3, 1), (2, 1), 
  (2, 3), (1, 3), (3, 2), (2, 1), (2, 4), (4, 2), (1, 3), (3, 2), (2, 1), (3, 4)
]

def count_ao_drawn_second : ℕ :=
  (random_pairs.filter is_ao_drawn_second).length

def probability_ao_drawn_second : ℚ :=
  count_ao_drawn_second / random_pairs.length

theorem problem_probability_ao_drawn_second :
  probability_ao_drawn_second = 1 / 4 :=
by
  sorry

end problem_probability_ao_drawn_second_l86_86932


namespace probability_of_success_l86_86774

def prob_successful_attempt := 0.5

def prob_unsuccessful_attempt := 1 - prob_successful_attempt

def all_fail_prob := prob_unsuccessful_attempt ^ 4

def at_least_one_success_prob := 1 - all_fail_prob

theorem probability_of_success :
  at_least_one_success_prob = 0.9375 :=
by
  -- Proof would be here
  sorry

end probability_of_success_l86_86774


namespace corn_plants_multiple_of_nine_l86_86919

theorem corn_plants_multiple_of_nine 
  (num_sunflowers : ℕ) (num_tomatoes : ℕ) (num_corn : ℕ) (max_plants_per_row : ℕ)
  (h1 : num_sunflowers = 45) (h2 : num_tomatoes = 63) (h3 : max_plants_per_row = 9)
  : ∃ k : ℕ, num_corn = 9 * k :=
by
  sorry

end corn_plants_multiple_of_nine_l86_86919


namespace apples_per_pie_l86_86473

theorem apples_per_pie (total_apples : ℕ) (apples_given : ℕ) (pies : ℕ) : 
  total_apples = 47 ∧ apples_given = 27 ∧ pies = 5 →
  (total_apples - apples_given) / pies = 4 :=
by
  intros h
  sorry

end apples_per_pie_l86_86473


namespace find_values_l86_86724

theorem find_values (h t u : ℕ) 
  (h0 : u = h - 5) 
  (h1 : (h * 100 + t * 10 + u) - (h * 100 + u * 10 + t) = 96)
  (hu : h < 10 ∧ t < 10 ∧ u < 10) :
  h = 5 ∧ t = 9 ∧ u = 0 :=
by 
  sorry

end find_values_l86_86724


namespace impossible_sequence_l86_86326

theorem impossible_sequence (a : ℕ → ℝ) (c : ℝ) (a1 : ℝ)
  (h_periodic : ∀ n, a (n + 3) = a n)
  (h_det : ∀ n, a n * a (n + 3) - a (n + 1) * a (n + 2) = c)
  (ha1 : a 1 = 2) (hc : c = 2) : false :=
by
  sorry

end impossible_sequence_l86_86326


namespace solve_inequality_l86_86796

theorem solve_inequality (x : ℝ) :
  (0 ≤ x^2 - x - 2 ∧ x^2 - x - 2 ≤ 4) ↔
  (-2 ≤ x ∧ x ≤ -1) ∨ (2 ≤ x ∧ x ≤ 3) :=
by sorry

end solve_inequality_l86_86796


namespace scientific_notation_of_12000_l86_86519

theorem scientific_notation_of_12000 : 12000 = 1.2 * 10^4 := 
by sorry

end scientific_notation_of_12000_l86_86519


namespace continuity_at_three_l86_86145

noncomputable def f (x : ℝ) : ℝ := -2 * x ^ 2 - 4

theorem continuity_at_three (ε : ℝ) (hε : 0 < ε) :
  ∃ δ > 0, ∀ x : ℝ, |x - 3| < δ → |f x - f 3| < ε :=
sorry

end continuity_at_three_l86_86145


namespace find_x_such_that_fraction_eq_l86_86311

theorem find_x_such_that_fraction_eq 
  (x : ℚ) (h₁ : x ≠ 1) (h₂ : x ≠ 5) : 
  (x^2 - 4 * x + 3) / (x^2 - 6 * x + 5) = (x^2 - 3 * x - 10) / (x^2 - 2 * x - 15) ↔ 
  x = -19 / 3 :=
sorry

end find_x_such_that_fraction_eq_l86_86311


namespace pow_two_greater_than_square_l86_86275

theorem pow_two_greater_than_square (n : ℕ) (h : n ≥ 5) : 2 ^ n > n ^ 2 :=
  sorry

end pow_two_greater_than_square_l86_86275


namespace fraction_transformation_half_l86_86581

theorem fraction_transformation_half (a b : ℝ) (a_ne_zero : a ≠ 0) (b_ne_zero : b ≠ 0) :
  ((2 * a + 2 * b) / (4 * a^2 + 4 * b^2)) = (1 / 2) * ((a + b) / (a^2 + b^2)) :=
by sorry

end fraction_transformation_half_l86_86581


namespace yogurt_combinations_l86_86185

theorem yogurt_combinations (f : ℕ) (t : ℕ) (h_f : f = 4) (h_t : t = 6) :
  (f * (t.choose 2) = 60) :=
by
  rw [h_f, h_t]
  sorry

end yogurt_combinations_l86_86185


namespace effective_writing_speed_is_750_l86_86597

-- Definitions based on given conditions in problem part a)
def total_words : ℕ := 60000
def total_hours : ℕ := 100
def break_hours : ℕ := 20
def effective_hours : ℕ := total_hours - break_hours
def effective_writing_speed : ℕ := total_words / effective_hours

-- Statement to be proved
theorem effective_writing_speed_is_750 : effective_writing_speed = 750 := by
  sorry

end effective_writing_speed_is_750_l86_86597


namespace min_value_of_seq_l86_86323

theorem min_value_of_seq 
  (a : ℕ → ℝ)
  (S : ℕ → ℝ)
  (m a₁ : ℝ)
  (h1 : ∀ n, a n + a (n + 1) = n * (-1) ^ ((n * (n + 1)) / 2))
  (h2 : m + S 2015 = -1007)
  (h3 : a₁ * m > 0) :
  ∃ x, x = (1 / a₁) + (4 / m) ∧ x = 9 :=
by
  sorry

end min_value_of_seq_l86_86323


namespace quadratic_inequality_solution_l86_86424

theorem quadratic_inequality_solution (x : ℝ) : (x^2 + 5*x + 6 > 0) ↔ (x < -3 ∨ x > -2) :=
  by
    sorry

end quadratic_inequality_solution_l86_86424


namespace total_games_played_l86_86894

-- Define the number of teams
def num_teams : ℕ := 12

-- Define the number of games each team plays with each other team
def games_per_pair : ℕ := 4

-- The theorem stating the total number of games played
theorem total_games_played : num_teams * (num_teams - 1) / 2 * games_per_pair = 264 :=
by
  sorry

end total_games_played_l86_86894


namespace flour_amount_second_combination_l86_86556

-- Define given conditions as parameters
variables {sugar_cost flour_cost : ℝ} (sugar_per_pound flour_per_pound : ℝ)
variable (cost1 cost2 : ℝ)

axiom cost1_eq :
  40 * sugar_per_pound + 16 * flour_per_pound = cost1

axiom cost2_eq :
  30 * sugar_per_pound + flour_cost = cost2

axiom sugar_rate :
  sugar_per_pound = 0.45

axiom flour_rate :
  flour_per_pound = 0.45

-- Define the target theorem
theorem flour_amount_second_combination : ∃ flour_amount : ℝ, flour_amount = 28 := by
  sorry

end flour_amount_second_combination_l86_86556


namespace number_of_sodas_in_pack_l86_86769

/-- Billy has twice as many brothers as sisters -/
def twice_as_many_brothers_as_sisters (brothers sisters : ℕ) : Prop :=
  brothers = 2 * sisters

/-- Billy has 2 sisters -/
def billy_has_2_sisters : Prop :=
  ∃ sisters : ℕ, sisters = 2

/-- Billy can give 2 sodas to each of his siblings if he wants to give out the entire pack while giving each sibling the same number of sodas -/
def divide_sodas_evenly (total_sodas siblings sodas_per_sibling : ℕ) : Prop :=
  total_sodas = siblings * sodas_per_sibling

/-- Determine the total number of sodas in the pack given the conditions -/
theorem number_of_sodas_in_pack : 
  ∃ (sisters brothers total_sodas : ℕ), 
    (twice_as_many_brothers_as_sisters brothers sisters) ∧ 
    (billy_has_2_sisters) ∧ 
    (divide_sodas_evenly total_sodas (sisters + brothers + 1) 2) ∧
    (total_sodas = 12) :=
by
  sorry

end number_of_sodas_in_pack_l86_86769


namespace moles_of_H2O_formed_l86_86477

theorem moles_of_H2O_formed
  (moles_H2SO4 : ℕ)
  (moles_H2O : ℕ)
  (H : moles_H2SO4 = 3)
  (H' : moles_H2O = 3) :
  moles_H2O = 3 :=
by
  sorry

end moles_of_H2O_formed_l86_86477


namespace range_of_a_l86_86371

def set_A (a : ℝ) : Set ℝ := {x : ℝ | a - 1 < x ∧ x < 2 * a + 1}
def set_B : Set ℝ := {x : ℝ | 0 < x ∧ x < 1}

theorem range_of_a (a : ℝ) : (set_A a ∩ set_B = ∅) ↔ (a ≤ -2 ∨ (a > -2 ∧ a ≤ -1/2) ∨ a ≥ 2) := by
  sorry

end range_of_a_l86_86371


namespace slices_left_per_person_is_2_l86_86279

variables (phil_slices andre_slices small_pizza_slices large_pizza_slices : ℕ)
variables (total_slices_eaten total_slices_left slices_per_person : ℕ)

-- Conditions
def conditions : Prop :=
  phil_slices = 9 ∧
  andre_slices = 9 ∧
  small_pizza_slices = 8 ∧
  large_pizza_slices = 14 ∧
  total_slices_eaten = phil_slices + andre_slices ∧
  total_slices_left = (small_pizza_slices + large_pizza_slices) - total_slices_eaten ∧
  slices_per_person = total_slices_left / 2

theorem slices_left_per_person_is_2 (h : conditions phil_slices andre_slices small_pizza_slices large_pizza_slices total_slices_eaten total_slices_left slices_per_person) :
  slices_per_person = 2 :=
sorry

end slices_left_per_person_is_2_l86_86279


namespace amount_of_p_l86_86790

theorem amount_of_p (p q r : ℝ) (h1 : q = (1 / 6) * p) (h2 : r = (1 / 6) * p) 
  (h3 : p = (q + r) + 32) : p = 48 :=
by
  sorry

end amount_of_p_l86_86790


namespace find_x_l86_86531

theorem find_x (x : ℝ) (h : x ^ 2 ∈ ({1, 0, x} : Set ℝ)) : x = -1 := 
sorry

end find_x_l86_86531


namespace line_always_passes_fixed_point_l86_86009

theorem line_always_passes_fixed_point:
  ∀ a x y, x = 5 → y = -3 → (a * x + (2 * a - 1) * y + a - 3 = 0) :=
by
  intros a x y h1 h2
  rw [h1, h2]
  sorry

end line_always_passes_fixed_point_l86_86009


namespace storage_temperature_overlap_l86_86818

theorem storage_temperature_overlap (T_A_min T_A_max T_B_min T_B_max : ℝ) 
  (hA : T_A_min = 0)
  (hA' : T_A_max = 5)
  (hB : T_B_min = 2)
  (hB' : T_B_max = 7) : 
  (max T_A_min T_B_min, min T_A_max T_B_max) = (2, 5) := by 
{
  sorry -- The proof is omitted as per instructions.
}

end storage_temperature_overlap_l86_86818


namespace reaction_produces_nh3_l86_86360

-- Define the Chemical Equation as a structure
structure Reaction where
  reagent1 : ℕ -- moles of NH4NO3
  reagent2 : ℕ -- moles of NaOH
  product  : ℕ -- moles of NH3

-- Given conditions
def reaction := Reaction.mk 2 2 2

-- Theorem stating that given 2 moles of NH4NO3 and 2 moles of NaOH,
-- the number of moles of NH3 formed is 2 moles.
theorem reaction_produces_nh3 (r : Reaction) (h1 : r.reagent1 = 2)
  (h2 : r.reagent2 = 2) : r.product = 2 := by
  sorry

end reaction_produces_nh3_l86_86360


namespace expression_not_equal_33_l86_86292

theorem expression_not_equal_33 (x y : ℤ) :
  x^5 + 3 * x^4 * y - 5 * x^3 * y^2 - 15 * x^2 * y^3 + 4 * x * y^4 + 12 * y^5 ≠ 33 := 
sorry

end expression_not_equal_33_l86_86292


namespace vat_percentage_is_15_l86_86563

def original_price : ℝ := 1700
def final_price : ℝ := 1955
def tax_amount := final_price - original_price

theorem vat_percentage_is_15 :
  (tax_amount / original_price) * 100 = 15 := 
sorry

end vat_percentage_is_15_l86_86563


namespace total_number_of_fleas_l86_86872

theorem total_number_of_fleas :
  let G_fleas := 10
  let O_fleas := G_fleas / 2
  let M_fleas := 5 * O_fleas
  G_fleas + O_fleas + M_fleas = 40 := rfl

end total_number_of_fleas_l86_86872


namespace simplify_expression_l86_86084

variables (a b : ℝ)

theorem simplify_expression (h₁ : a = 2) (h₂ : b = -1) :
  (2 * a^2 - a * b - b^2) - 2 * (a^2 - 2 * a * b + b^2) = -5 :=
by
  sorry

end simplify_expression_l86_86084


namespace trig_identity_proof_l86_86174

theorem trig_identity_proof
  (α : ℝ)
  (h : Real.sin (α - π / 6) = 3 / 5) :
  Real.cos (2 * π / 3 - α) = 3 / 5 :=
sorry

end trig_identity_proof_l86_86174


namespace num_green_balls_l86_86791

theorem num_green_balls (G : ℕ) (h : (3 * 2 : ℚ) / ((5 + G) * (4 + G)) = 1/12) : G = 4 :=
by
  sorry

end num_green_balls_l86_86791


namespace remaining_pages_l86_86362

theorem remaining_pages (total_pages : ℕ) (science_project_percentage : ℕ) (math_homework_pages : ℕ)
  (h1 : total_pages = 120)
  (h2 : science_project_percentage = 25) 
  (h3 : math_homework_pages = 10) : 
  total_pages - (total_pages * science_project_percentage / 100) - math_homework_pages = 80 := by
  sorry

end remaining_pages_l86_86362


namespace area_of_room_in_square_inches_l86_86828

-- Defining the conversion from feet to inches
def feet_to_inches (ft : ℕ) : ℕ := ft * 12

-- Given conditions
def length_in_feet : ℕ := 10
def width_in_feet : ℕ := 10

-- Calculate length and width in inches
def length_in_inches := feet_to_inches length_in_feet
def width_in_inches := feet_to_inches width_in_feet

-- Calculate area in square inches
def area_in_square_inches := length_in_inches * width_in_inches

-- Theorem statement
theorem area_of_room_in_square_inches
  (h1 : length_in_feet = 10)
  (h2 : width_in_feet = 10)
  (conversion : feet_to_inches 1 = 12) :
  area_in_square_inches = 14400 :=
sorry

end area_of_room_in_square_inches_l86_86828


namespace Caden_total_money_l86_86640

theorem Caden_total_money (p n d q : ℕ) (hp : p = 120)
    (hn : p = 3 * n) 
    (hd : n = 5 * d)
    (hq : q = 2 * d) :
    (p * 1 / 100 + n * 5 / 100 + d * 10 / 100 + q * 25 / 100) = 8 := 
by
  sorry

end Caden_total_money_l86_86640


namespace number_of_sheep_l86_86649

theorem number_of_sheep (s d : ℕ) 
  (h1 : s + d = 15)
  (h2 : 4 * s + 2 * d = 22 + 2 * (s + d)) : 
  s = 11 :=
by
  sorry

end number_of_sheep_l86_86649


namespace protein_in_steak_is_correct_l86_86736

-- Definitions of the conditions
def collagen_protein_per_scoop : ℕ := 18 / 2 -- 9 grams
def protein_powder_per_scoop : ℕ := 21 -- 21 grams

-- Define the total protein consumed
def total_protein (collagen_scoops protein_scoops : ℕ) (protein_from_steak : ℕ) : ℕ :=
  collagen_protein_per_scoop * collagen_scoops + protein_powder_per_scoop * protein_scoops + protein_from_steak

-- Condition in the problem
def total_protein_consumed : ℕ := 86

-- Prove that the protein in the steak is 56 grams
theorem protein_in_steak_is_correct : 
  total_protein 1 1 56 = total_protein_consumed :=
sorry

end protein_in_steak_is_correct_l86_86736


namespace no_such_function_exists_l86_86480

theorem no_such_function_exists :
  ¬ ∃ f : ℕ → ℕ, ∀ n : ℕ, f (f n) = n + 2015 := 
by
  sorry

end no_such_function_exists_l86_86480


namespace max_triangle_area_l86_86087

theorem max_triangle_area (a b c : ℝ) (h1 : b + c = 8) (h2 : a + b > c)
  (h3 : a + c > b) (h4 : b + c > a) :
  (a - b + c) * (a + b - c) ≤ 64 / 17 :=
by sorry

end max_triangle_area_l86_86087


namespace sum_m_n_l86_86683

-- Define the conditions and the result

def probabilityOfNo3x3RedSquare : ℚ :=
  65408 / 65536

def gcd_65408_65536 := Nat.gcd 65408 65536

def simplifiedProbability : ℚ :=
  probabilityOfNo3x3RedSquare / gcd_65408_65536

def m : ℕ :=
  511

def n : ℕ :=
  512

theorem sum_m_n : m + n = 1023 := by
  sorry

end sum_m_n_l86_86683


namespace ball_in_78th_position_is_green_l86_86651

-- Definition of colors in the sequence
inductive Color
| red
| yellow
| green
| blue
| violet

open Color

-- Function to compute the color of a ball at a given position within a cycle
def ball_color (n : Nat) : Color :=
  match n % 5 with
  | 0 => red    -- 78 % 5 == 3, hence 3 + 1 == 4 ==> Using 0 for red to 4 for violet
  | 1 => yellow
  | 2 => green
  | 3 => blue
  | 4 => violet
  | _ => red  -- default case, should not be reached

-- Theorem stating the desired proof problem
theorem ball_in_78th_position_is_green : ball_color 78 = green :=
by
  sorry

end ball_in_78th_position_is_green_l86_86651


namespace least_number_of_roots_l86_86099

variable (g : ℝ → ℝ) -- Declare the function g with domain ℝ and codomain ℝ

-- Define the conditions as assumptions.
variable (h1 : ∀ x : ℝ, g (3 + x) = g (3 - x))
variable (h2 : ∀ x : ℝ, g (8 + x) = g (8 - x))
variable (h3 : g 0 = 0)

-- State the theorem to prove the necessary number of roots.
theorem least_number_of_roots : ∀ a b : ℝ, a ≤ -2000 ∧ b ≥ 2000 → ∃ n ≥ 668, ∃ x : ℝ, g x = 0 ∧ a ≤ x ∧ x ≤ b :=
by
  -- To be filled in with the logic to prove the theorem.
  sorry

end least_number_of_roots_l86_86099


namespace regular_pentagon_diagonal_square_l86_86973

variable (a d : ℝ)
def is_regular_pentagon (a d : ℝ) : Prop :=
d ^ 2 = a ^ 2 + a * d

theorem regular_pentagon_diagonal_square :
  is_regular_pentagon a d :=
sorry

end regular_pentagon_diagonal_square_l86_86973


namespace find_a_plus_b_l86_86272

variable (r a b : ℝ)
variable (seq : ℕ → ℝ)

-- Conditions on the sequence
axiom seq_def : seq 0 = 4096
axiom seq_rule : ∀ n, seq (n + 1) = seq n * r

-- Given value
axiom r_value : r = 1 / 4

-- Given intermediate positions in the sequence
axiom seq_a : seq 3 = a
axiom seq_b : seq 4 = b
axiom seq_5 : seq 5 = 4

-- Theorem to prove
theorem find_a_plus_b : a + b = 80 := by
  sorry

end find_a_plus_b_l86_86272


namespace people_who_came_to_game_l86_86266

def total_seats : Nat := 92
def people_with_banners : Nat := 38
def empty_seats : Nat := 45

theorem people_who_came_to_game : (total_seats - empty_seats = 47) :=
by 
  sorry

end people_who_came_to_game_l86_86266


namespace cube_volume_equality_l86_86454

open BigOperators Real

-- Definitions
def initial_volume : ℝ := 1

def removed_volume (x : ℝ) : ℝ := x^2

def removed_volume_with_overlap (x y : ℝ) : ℝ := x^2 - (x^2 * y)

def remaining_volume (a b c : ℝ) : ℝ := 
  initial_volume - removed_volume c - removed_volume_with_overlap b c - removed_volume_with_overlap a c - removed_volume_with_overlap a b + (c^2 * b)

-- Main theorem to prove
theorem cube_volume_equality (c b a : ℝ) (hcb : c < b) (hba : b < a) (ha1 : a < 1):
  (c = 1 / 2) ∧ 
  (b = (1 + Real.sqrt 17) / 8) ∧ 
  (a = (17 + Real.sqrt 17 + Real.sqrt (1202 - 94 * Real.sqrt 17)) / 64) :=
sorry

end cube_volume_equality_l86_86454


namespace bulb_illumination_l86_86278

theorem bulb_illumination (n : ℕ) (h : n = 6) : 
  (2^n - 1) = 63 := by {
  sorry
}

end bulb_illumination_l86_86278


namespace range_of_a_l86_86026

theorem range_of_a (a : ℝ) (A : Set ℝ) (hA : ∀ x, x ∈ A ↔ a / (x - 1) < 1) (h_not_in : 2 ∉ A) : a ≥ 1 := 
sorry

end range_of_a_l86_86026


namespace range_of_m_l86_86119

theorem range_of_m (m : ℝ) (h : ∃ x : ℝ, x^2 - x - m = 0) : m ≥ -1/4 :=
by
  sorry

end range_of_m_l86_86119


namespace rectangle_area_ratio_k_l86_86439

theorem rectangle_area_ratio_k (d : ℝ) (l w : ℝ) (h1 : l / w = 5 / 2) (h2 : d^2 = l^2 + w^2) :
  ∃ k : ℝ, k = 10 / 29 ∧ (l * w = k * d^2) :=
by {
  -- proof steps will go here
  sorry
}

end rectangle_area_ratio_k_l86_86439


namespace smallest_value_l86_86945

theorem smallest_value 
  (x1 x2 x3 : ℝ) 
  (hx1 : 0 < x1) 
  (hx2 : 0 < x2) 
  (hx3 : 0 < x3)
  (h : 2 * x1 + 3 * x2 + 4 * x3 = 100) : 
  x1^2 + x2^2 + x3^2 = 10000 / 29 := by
  sorry

end smallest_value_l86_86945


namespace find_n_l86_86717

theorem find_n : ∃ n : ℕ, 2^7 * 3^3 * 5 * n = Nat.factorial 12 ∧ n = 27720 :=
by
  use 27720
  have h1 : 2^7 * 3^3 * 5 * 27720 = Nat.factorial 12 :=
  sorry -- This will be the place to prove the given equation eventually.
  exact ⟨h1, rfl⟩

end find_n_l86_86717


namespace proof_problem_l86_86273

-- Definitions
def U : Set ℕ := {x | x < 7 ∧ x > 0}
def A : Set ℕ := {1, 2, 5}
def B : Set ℕ := {2, 3, 4, 5}

-- The equality proof statement
theorem proof_problem :
  (A ∩ B = {2, 5}) ∧
  ({x | x ∈ U ∧ ¬ (x ∈ A)} = {3, 4, 6}) ∧
  (A ∪ {x | x ∈ U ∧ ¬ (x ∈ B)} = {1, 2, 5, 6}) :=
by
  sorry

end proof_problem_l86_86273


namespace area_of_gray_region_l86_86494

theorem area_of_gray_region :
  (radius_smaller = (2 : ℝ) / 2) →
  (radius_larger = 4 * radius_smaller) →
  (gray_area = π * radius_larger ^ 2 - π * radius_smaller ^ 2) →
  gray_area = 15 * π :=
by
  intro h1 h2 h3
  rw [h1, h2] at h3
  sorry

end area_of_gray_region_l86_86494


namespace cubic_sum_l86_86787

theorem cubic_sum (x y z : ℝ) (h1 : x + y + z = 2) (h2 : x * y + x * z + y * z = -5) (h3 : x * y * z = -6) :
  x^3 + y^3 + z^3 = 18 :=
by
  sorry

end cubic_sum_l86_86787


namespace waiter_tips_earned_l86_86466

theorem waiter_tips_earned (total_customers tips_left no_tip_customers tips_per_customer : ℕ) :
  no_tip_customers + tips_left = total_customers ∧ tips_per_customer = 3 ∧ no_tip_customers = 5 ∧ total_customers = 7 → 
  tips_left * tips_per_customer = 6 :=
by
  intro h
  sorry

end waiter_tips_earned_l86_86466


namespace smallest_positive_divisible_by_111_has_last_digits_2004_l86_86314

theorem smallest_positive_divisible_by_111_has_last_digits_2004 :
  ∃ (X : ℕ), (∃ (A : ℕ), X = A * 10^4 + 2004) ∧ 111 ∣ X ∧ X = 662004 := by
  sorry

end smallest_positive_divisible_by_111_has_last_digits_2004_l86_86314


namespace max_value_of_f_l86_86959

noncomputable def f (x : ℝ) : ℝ := Real.exp x - x

theorem max_value_of_f : ∃ max, max ∈ Set.image f (Set.Icc (-1 : ℝ) 1) ∧ max = Real.exp 1 - 1 :=
by
  sorry

end max_value_of_f_l86_86959


namespace max_integer_value_of_k_l86_86939

theorem max_integer_value_of_k :
  ∀ x y k : ℤ,
    x - 4 * y = k - 1 →
    2 * x + y = k →
    x - y ≤ 0 →
    k ≤ 0 :=
by
  intros x y k h1 h2 h3
  sorry

end max_integer_value_of_k_l86_86939


namespace problem_2023_divisible_by_consecutive_integers_l86_86876

theorem problem_2023_divisible_by_consecutive_integers :
  ∃ (n : ℕ), (n = 2022 ∨ n = 2023 ∨ n = 2024) ∧ (2023^2023 - 2023^2021) % n = 0 :=
sorry

end problem_2023_divisible_by_consecutive_integers_l86_86876


namespace branches_number_l86_86956

-- Conditions (converted into Lean definitions)
def total_leaves : ℕ := 12690
def twigs_per_branch : ℕ := 90
def leaves_per_twig_percentage_4 : ℝ := 0.3
def leaves_per_twig_percentage_5 : ℝ := 0.7
def leaves_per_twig_4 : ℕ := 4
def leaves_per_twig_5 : ℕ := 5

-- The goal
theorem branches_number (B : ℕ) 
  (h1 : twigs_per_branch = 90) 
  (h2 : leaves_per_twig_percentage_4 = 0.3) 
  (h3 : leaves_per_twig_percentage_5 = 0.7) 
  (h4 : leaves_per_twig_4 = 4) 
  (h5 : leaves_per_twig_5 = 5) 
  (h6 : total_leaves = 12690) :
  B = 30 := 
sorry

end branches_number_l86_86956


namespace solution_correct_l86_86699

noncomputable def quadratic_inequality_solution (x : ℝ) : Prop :=
  x^2 - 36 * x + 320 ≤ 16

theorem solution_correct (x : ℝ) : quadratic_inequality_solution x ↔ 16 ≤ x ∧ x ≤ 19 :=
by sorry

end solution_correct_l86_86699


namespace solve_quadratic_eq_l86_86826

theorem solve_quadratic_eq : (x : ℝ) → (x^2 - 4 = 0) → (x = 2 ∨ x = -2) :=
by
  sorry

end solve_quadratic_eq_l86_86826


namespace find_f_10_l86_86418

noncomputable def f : ℤ → ℤ := sorry

axiom cond1 : f 1 + 1 > 0
axiom cond2 : ∀ x y : ℤ, f (x + y) - x * f y - y * f x = f x * f y - x - y + x * y
axiom cond3 : ∀ x : ℤ, 2 * f x = f (x + 1) - x + 1

theorem find_f_10 : f 10 = 1014 := 
by
  sorry 

end find_f_10_l86_86418


namespace sixtieth_term_of_arithmetic_sequence_l86_86528

theorem sixtieth_term_of_arithmetic_sequence (a1 a15 : ℚ) (d : ℚ) (h1 : a1 = 7) (h2 : a15 = 37)
  (h3 : a15 = a1 + 14 * d) : a1 + 59 * d = 134.5 := by
  sorry

end sixtieth_term_of_arithmetic_sequence_l86_86528


namespace sets_B_C_D_represent_same_function_l86_86224

theorem sets_B_C_D_represent_same_function :
  (∀ x : ℝ, (2 * x = 2 * (x ^ (3 : ℝ) ^ (1 / 3)))) ∧
  (∀ x t : ℝ, (x ^ 2 + x + 3 = t ^ 2 + t + 3)) ∧
  (∀ x : ℝ, (x ^ 2 = (x ^ 4) ^ (1 / 2))) :=
by
  sorry

end sets_B_C_D_represent_same_function_l86_86224


namespace total_population_estimate_l86_86199

def average_population_min : ℕ := 3200
def average_population_max : ℕ := 3600
def towns : ℕ := 25

theorem total_population_estimate : 
    ∃ x : ℕ, average_population_min ≤ x ∧ x ≤ average_population_max ∧ towns * x = 85000 :=
by 
  sorry

end total_population_estimate_l86_86199


namespace quotient_remainder_difference_l86_86593

theorem quotient_remainder_difference (N Q P R k : ℕ) (h1 : N = 75) (h2 : N = 5 * Q) (h3 : N = 34 * P + R) (h4 : Q = R + k) (h5 : k > 0) :
  Q - R = 8 :=
sorry

end quotient_remainder_difference_l86_86593


namespace benny_number_of_days_worked_l86_86370

-- Define the conditions
def total_hours_worked : ℕ := 18
def hours_per_day : ℕ := 3

-- Define the problem statement in Lean
theorem benny_number_of_days_worked : (total_hours_worked / hours_per_day) = 6 := 
by
  sorry

end benny_number_of_days_worked_l86_86370


namespace BD_value_l86_86564

noncomputable def triangleBD (AC BC AD CD : ℝ) : ℝ :=
  let θ := Real.arccos ((3 ^ 2 + 9 ^ 2 - 7 ^ 2) / (2 * 3 * 9))
  let ψ := Real.pi - θ
  let cosψ := Real.cos ψ
  let x := (-1.026 + Real.sqrt ((1.026 ^ 2) + 4 * 40)) / 2
  if x > 0 then x else 5.8277 -- confirmed manually as positive root.

theorem BD_value : (triangleBD 7 7 9 3) = 5.8277 :=
by
  apply sorry

end BD_value_l86_86564


namespace gravel_per_truckload_l86_86285

def truckloads_per_mile : ℕ := 3
def miles_day1 : ℕ := 4
def miles_day2 : ℕ := 2 * miles_day1 - 1
def total_paved_miles : ℕ := miles_day1 + miles_day2
def total_road_length : ℕ := 16
def miles_remaining : ℕ := total_road_length - total_paved_miles
def remaining_truckloads : ℕ := miles_remaining * truckloads_per_mile
def barrels_needed : ℕ := 6
def gravel_per_pitch : ℕ := 5
def P : ℚ := barrels_needed / remaining_truckloads
def G : ℚ := gravel_per_pitch * P

theorem gravel_per_truckload :
  G = 2 :=
by
  sorry

end gravel_per_truckload_l86_86285


namespace profit_percent_calc_l86_86498

theorem profit_percent_calc (SP CP : ℝ) (h : CP = 0.25 * SP) : (SP - CP) / CP * 100 = 300 :=
by
  sorry

end profit_percent_calc_l86_86498


namespace ellipse_same_foci_l86_86573

-- Definitions related to the problem
variables {x y p q : ℝ}

-- Condition
def represents_hyperbola (p q : ℝ) : Prop :=
  (p * q > 0) ∧ (∀ x y : ℝ, (x^2 / -p + y^2 / q = 1))

-- Proof Statement
theorem ellipse_same_foci (p q : ℝ) (hpq : p * q > 0)
  (h : ∀ x y : ℝ, x^2 / -p + y^2 / q = 1) :
  (∀ x y : ℝ, x^2 / (2*p + q) + y^2 / p = -1) :=
sorry -- Proof goes here

end ellipse_same_foci_l86_86573


namespace minimum_value_expression_l86_86440

theorem minimum_value_expression {a b c : ℝ} (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_prod : a * b * c = 1) : 
  a^2 + 4 * a * b + 9 * b^2 + 3 * b * c + c^2 ≥ 18 :=
by
  sorry

end minimum_value_expression_l86_86440


namespace insurance_payment_yearly_l86_86991

noncomputable def quarterly_payment : ℝ := 378
noncomputable def quarters_per_year : ℕ := 12 / 3
noncomputable def annual_payment : ℝ := quarterly_payment * quarters_per_year

theorem insurance_payment_yearly : annual_payment = 1512 := by
  sorry

end insurance_payment_yearly_l86_86991


namespace cyclic_proportion_l86_86298

variable {A B C p q r : ℝ}

theorem cyclic_proportion (h1 : A / B = p) (h2 : B / C = q) (h3 : C / A = r) :
  ∃ x y z, A = x ∧ B = y ∧ C = z ∧ x / y = p ∧ y / z = q ∧ z / x = r ∧
  x = (p^2 * q / r)^(1/3:ℝ) ∧ y = (q^2 * r / p)^(1/3:ℝ) ∧ z = (r^2 * p / q)^(1/3:ℝ) :=
by sorry

end cyclic_proportion_l86_86298


namespace remainder_when_divided_by_x_minus_3_l86_86465

open Polynomial

noncomputable def p : ℝ[X] := 4 * X^3 - 12 * X^2 + 16 * X - 20

theorem remainder_when_divided_by_x_minus_3 : eval 3 p = 28 := by
  sorry

end remainder_when_divided_by_x_minus_3_l86_86465


namespace odd_prime_power_condition_l86_86542

noncomputable def is_power_of (a b : ℕ) : Prop :=
  ∃ t : ℕ, b = a ^ t

theorem odd_prime_power_condition (n p x y k : ℕ) (hn : 1 < n) (hp_prime : Prime p) 
  (hp_odd : p % 2 = 1) (hx : x ≠ 0) (hy : y ≠ 0) (hk : k ≠ 0) (hx_odd : x % 2 ≠ 0) 
  (hy_odd : y % 2 ≠ 0) (h_eq : x^n + y^n = p^k) :
  is_power_of p n :=
sorry

end odd_prime_power_condition_l86_86542


namespace race_order_l86_86642

inductive Position where
| First | Second | Third | Fourth | Fifth
deriving DecidableEq, Repr

structure Statements where
  amy1 : Position → Prop
  amy2 : Position → Prop
  bruce1 : Position → Prop
  bruce2 : Position → Prop
  chris1 : Position → Prop
  chris2 : Position → Prop
  donna1 : Position → Prop
  donna2 : Position → Prop
  eve1 : Position → Prop
  eve2 : Position → Prop

def trueStatements : Statements := {
  amy1 := fun p => p = Position.Second,
  amy2 := fun p => p = Position.Third,
  bruce1 := fun p => p = Position.Second,
  bruce2 := fun p => p = Position.Fourth,
  chris1 := fun p => p = Position.First,
  chris2 := fun p => p = Position.Second,
  donna1 := fun p => p = Position.Third,
  donna2 := fun p => p = Position.Fifth,
  eve1 := fun p => p = Position.Fourth,
  eve2 := fun p => p = Position.First,
}

theorem race_order (f : Statements) :
  f.amy1 Position.Second ∧ f.amy2 Position.Third ∧
  f.bruce1 Position.First ∧ f.bruce2 Position.Fourth ∧
  f.chris1 Position.Fifth ∧ f.chris2 Position.Second ∧
  f.donna1 Position.Fourth ∧ f.donna2 Position.Fifth ∧
  f.eve1 Position.Fourth ∧ f.eve2 Position.First :=
by
  sorry

end race_order_l86_86642


namespace ab_equiv_l86_86737

theorem ab_equiv (a b : ℝ) (hb : b ≠ 0) (h : (a - b) / b = 3 / 7) : a / b = 10 / 7 :=
by
  sorry

end ab_equiv_l86_86737


namespace ratio_debt_manny_to_annika_l86_86780

-- Define the conditions
def money_jericho_has : ℕ := 30
def debt_to_annika : ℕ := 14
def remaining_money_after_debts : ℕ := 9

-- Define the amount Jericho owes Manny
def debt_to_manny : ℕ := money_jericho_has - debt_to_annika - remaining_money_after_debts

-- Prove the ratio of amount Jericho owes Manny to the amount he owes Annika is 1:2
theorem ratio_debt_manny_to_annika :
  debt_to_manny * 2 = debt_to_annika :=
by
  -- Proof goes here
  sorry

end ratio_debt_manny_to_annika_l86_86780


namespace cricket_average_increase_l86_86039

theorem cricket_average_increase (runs_mean : ℕ) (innings : ℕ) (runs : ℕ) (new_runs : ℕ) (x : ℕ) :
  runs_mean = 35 → innings = 10 → runs = 79 → (total_runs : ℕ) = runs_mean * innings → 
  (new_total : ℕ) = total_runs + runs → (new_mean : ℕ) = new_total / (innings + 1) ∧ new_mean = runs_mean + x → x = 4 :=
by
  sorry

end cricket_average_increase_l86_86039


namespace ballet_class_members_l86_86290

theorem ballet_class_members (large_groups : ℕ) (members_per_large_group : ℕ) (total_members : ℕ) 
    (h1 : large_groups = 12) (h2 : members_per_large_group = 7) (h3 : total_members = large_groups * members_per_large_group) : 
    total_members = 84 :=
sorry

end ballet_class_members_l86_86290


namespace correct_calculation_result_l86_86933

theorem correct_calculation_result (x : ℝ) (h : x / 12 = 8) : 12 * x = 1152 :=
sorry

end correct_calculation_result_l86_86933


namespace mul_97_97_eq_9409_l86_86488

theorem mul_97_97_eq_9409 : 97 * 97 = 9409 := 
  sorry

end mul_97_97_eq_9409_l86_86488


namespace tom_saves_promotion_l86_86489

open Nat

theorem tom_saves_promotion (price : ℕ) (disc_percent : ℕ) (discount_amount : ℕ) 
    (promotion_x_cost second_pair_cost_promo_x promotion_y_cost promotion_savings : ℕ) 
    (h1 : price = 50)
    (h2 : disc_percent = 40)
    (h3 : discount_amount = 15)
    (h4 : second_pair_cost_promo_x = price - (price * disc_percent / 100))
    (h5 : promotion_x_cost = price + second_pair_cost_promo_x)
    (h6 : promotion_y_cost = price + (price - discount_amount))
    (h7 : promotion_savings = promotion_y_cost - promotion_x_cost) :
  promotion_savings = 5 :=
by
  sorry

end tom_saves_promotion_l86_86489


namespace Amanda_tickets_third_day_l86_86151

theorem Amanda_tickets_third_day :
  (let total_tickets := 80
   let first_day_tickets := 5 * 4
   let second_day_tickets := 32

   total_tickets - (first_day_tickets + second_day_tickets) = 28) :=
by
  sorry

end Amanda_tickets_third_day_l86_86151


namespace max_value_M_l86_86627

theorem max_value_M : 
  ∃ t : ℝ, (t = (3 / (4 ^ (1 / 3)))) ∧ 
    (∀ (a b c : ℝ), 0 < a → 0 < b → 0 < c → 
      a^3 + b^3 + c^3 - 3 * a * b * c ≥ t * (a * b^2 + b * c^2 + c * a^2 - 3 * a * b * c)) :=
sorry

end max_value_M_l86_86627


namespace y_intercept_of_line_l86_86934

theorem y_intercept_of_line (m : ℝ) (x₀ y₀ : ℝ) (h₁ : m = -3) (h₂ : x₀ = 7) (h₃ : y₀ = 0) :
  ∃ (b : ℝ), (0, b) = (0, 21) :=
by
  -- Our goal is to prove the y-intercept is (0, 21)
  sorry

end y_intercept_of_line_l86_86934


namespace sum_not_equals_any_l86_86741

-- Define the nine special natural numbers a1 to a9
def a1 (k : ℕ) : ℕ := (10^k - 1) / 9
def a2 (m : ℕ) : ℕ := 2 * (10^m - 1) / 9
def a3 (p : ℕ) : ℕ := 3 * (10^p - 1) / 9
def a4 (q : ℕ) : ℕ := 4 * (10^q - 1) / 9
def a5 (r : ℕ) : ℕ := 5 * (10^r - 1) / 9
def a6 (s : ℕ) : ℕ := 6 * (10^s - 1) / 9
def a7 (t : ℕ) : ℕ := 7 * (10^t - 1) / 9
def a8 (u : ℕ) : ℕ := 8 * (10^u - 1) / 9
def a9 (v : ℕ) : ℕ := 9 * (10^v - 1) / 9

-- Statement of the problem
theorem sum_not_equals_any (k m p q r s t u v : ℕ) :
  ¬ (a1 k = a2 m + a3 p + a4 q + a5 r + a6 s + a7 t + a8 u + a9 v) ∧
  ¬ (a2 m = a1 k + a3 p + a4 q + a5 r + a6 s + a7 t + a8 u + a9 v) ∧
  ¬ (a3 p = a1 k + a2 m + a4 q + a5 r + a6 s + a7 t + a8 u + a9 v) ∧
  ¬ (a4 q = a1 k + a2 m + a3 p + a5 r + a6 s + a7 t + a8 u + a9 v) ∧
  ¬ (a5 r = a1 k + a2 m + a3 p + a4 q + a6 s + a7 t + a8 u + a9 v) ∧
  ¬ (a6 s = a1 k + a2 m + a3 p + a4 q + a5 r + a7 t + a8 u + a9 v) ∧
  ¬ (a7 t = a1 k + a2 m + a3 p + a4 q + a5 r + a6 s + a8 u + a9 v) ∧
  ¬ (a8 u = a1 k + a2 m + a3 p + a4 q + a5 r + a6 s + a7 t + a9 v) ∧
  ¬ (a9 v = a1 k + a2 m + a3 p + a4 q + a5 r + a6 s + a7 t + a8 u) :=
  sorry

end sum_not_equals_any_l86_86741


namespace negation_of_p_is_correct_l86_86129

variable (c : ℝ)

-- Proposition p defined as: there exists c > 0 such that x^2 - x + c = 0 has a solution
def proposition_p : Prop :=
  ∃ c > 0, ∃ x : ℝ, x^2 - x + c = 0

-- Negation of proposition p
def neg_proposition_p : Prop :=
  ∀ c > 0, ¬ ∃ x : ℝ, x^2 - x + c = 0

-- The Lean statement to prove
theorem negation_of_p_is_correct :
  neg_proposition_p ↔ (∀ c > 0, ¬ ∃ x : ℝ, x^2 - x + c = 0) :=
by
  sorry

end negation_of_p_is_correct_l86_86129


namespace alice_savings_l86_86205

noncomputable def commission (sales : ℝ) : ℝ := 0.02 * sales
noncomputable def totalEarnings (basic_salary commission : ℝ) : ℝ := basic_salary + commission
noncomputable def savings (total_earnings : ℝ) : ℝ := 0.10 * total_earnings

theorem alice_savings (sales basic_salary : ℝ) (commission_rate savings_rate : ℝ) :
  commission_rate = 0.02 →
  savings_rate = 0.10 →
  sales = 2500 →
  basic_salary = 240 →
  savings (totalEarnings basic_salary (commission_rate * sales)) = 29 :=
by
  intros h1 h2 h3 h4
  sorry

end alice_savings_l86_86205


namespace num_math_not_science_l86_86968

-- Definitions as conditions
def students_total : ℕ := 30
def both_clubs : ℕ := 2
def math_to_science_ratio : ℕ := 3

-- The proof we need to show
theorem num_math_not_science :
  ∃ x y : ℕ, (x + y + both_clubs = students_total) ∧ (y = math_to_science_ratio * (x + both_clubs) - 2 * (math_to_science_ratio - 1)) ∧ (y - both_clubs = 20) :=
by
  sorry

end num_math_not_science_l86_86968


namespace cistern_fill_time_l86_86740

-- Define the problem conditions
def pipe_p_fill_time : ℕ := 10
def pipe_q_fill_time : ℕ := 15
def joint_filling_time : ℕ := 2
def remaining_fill_time : ℕ := 10 -- This is the answer we need to prove

-- Prove that the remaining fill time is equal to 10 minutes
theorem cistern_fill_time :
  (joint_filling_time * (1 / pipe_p_fill_time + 1 / pipe_q_fill_time) + (remaining_fill_time / pipe_q_fill_time)) = 1 :=
sorry

end cistern_fill_time_l86_86740


namespace arccos_of_half_eq_pi_over_three_l86_86768

theorem arccos_of_half_eq_pi_over_three : Real.arccos (1 / 2) = Real.pi / 3 :=
sorry

end arccos_of_half_eq_pi_over_three_l86_86768


namespace range_of_m_l86_86476

variables (m : ℝ)

def p : Prop := ∀ x : ℝ, 0 < x → (1/2 : ℝ)^x + m - 1 < 0
def q : Prop := ∃ x : ℝ, 0 < x ∧ m * x^2 + 4 * x - 1 = 0

theorem range_of_m (h : p m ∧ q m) : -4 ≤ m ∧ m ≤ 0 :=
sorry

end range_of_m_l86_86476


namespace defective_units_shipped_for_sale_l86_86850

theorem defective_units_shipped_for_sale (d p : ℝ) (h1 : d = 0.09) (h2 : p = 0.04) : (d * p * 100 = 0.36) :=
by 
  -- Assuming some calculation steps 
  sorry

end defective_units_shipped_for_sale_l86_86850


namespace smallest_n_divisible_by_2022_l86_86893

theorem smallest_n_divisible_by_2022 (n : ℕ) (h1 : n > 1) (h2 : (n^7 - 1) % 2022 = 0) : n = 79 :=
sorry

end smallest_n_divisible_by_2022_l86_86893


namespace factorization_of_difference_of_squares_l86_86846

theorem factorization_of_difference_of_squares (m : ℝ) : 
  m^2 - 16 = (m + 4) * (m - 4) := 
by 
  sorry

end factorization_of_difference_of_squares_l86_86846


namespace roots_opposite_eq_minus_one_l86_86118

theorem roots_opposite_eq_minus_one (k : ℝ) 
  (h_real_roots : ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁ + x₂ = 0 ∧ x₁ * x₂ = k + 1) :
  k = -1 :=
by
  sorry

end roots_opposite_eq_minus_one_l86_86118


namespace quadratic_inequality_false_iff_l86_86668

open Real

theorem quadratic_inequality_false_iff (a : ℝ) : 
  (¬ ∃ x : ℝ, x^2 + (a - 1) * x + 1 ≤ 0) ↔ -1 < a ∧ a < 3 :=
sorry

end quadratic_inequality_false_iff_l86_86668


namespace total_earnings_correct_l86_86348

-- Define the earnings of Terrence
def TerrenceEarnings : ℕ := 30

-- Define the difference in earnings between Jermaine and Terrence
def JermaineEarningsDifference : ℕ := 5

-- Define the earnings of Jermaine
def JermaineEarnings : ℕ := TerrenceEarnings + JermaineEarningsDifference

-- Define the earnings of Emilee
def EmileeEarnings : ℕ := 25

-- Define the total earnings
def TotalEarnings : ℕ := TerrenceEarnings + JermaineEarnings + EmileeEarnings

theorem total_earnings_correct : TotalEarnings = 90 := by
  sorry

end total_earnings_correct_l86_86348


namespace arithmetic_progression_11th_term_l86_86604

theorem arithmetic_progression_11th_term:
  ∀ (a d : ℝ), (15 / 2) * (2 * a + 14 * d) = 56.25 → a + 6 * d = 3.25 → a + 10 * d = 5.25 :=
by
  intros a d h_sum h_7th
  sorry

end arithmetic_progression_11th_term_l86_86604


namespace sum_of_first_11_terms_l86_86232

theorem sum_of_first_11_terms (a1 d : ℝ) (h : 2 * a1 + 10 * d = 8) : 
  (11 / 2) * (2 * a1 + 10 * d) = 44 := 
by sorry

end sum_of_first_11_terms_l86_86232


namespace mairiad_distance_ratio_l86_86200

open Nat

theorem mairiad_distance_ratio :
  ∀ (x : ℕ),
  let miles_run := 40
  let miles_walked := 3 * miles_run / 5
  let total_distance := miles_run + miles_walked + x * miles_run
  total_distance = 184 →
  24 + x * 40 = 144 →
  (24 + 3 * 40) / 40 = 3.6 := 
sorry

end mairiad_distance_ratio_l86_86200


namespace power_function_inequality_l86_86032

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^a

theorem power_function_inequality (x : ℝ) (a : ℝ) : (x > 1) → (f x a < x) ↔ (a < 1) :=
by
  sorry

end power_function_inequality_l86_86032


namespace sandy_books_from_first_shop_l86_86103

theorem sandy_books_from_first_shop 
  (cost_first_shop : ℕ)
  (books_second_shop : ℕ)
  (cost_second_shop : ℕ)
  (average_price : ℕ)
  (total_cost : ℕ)
  (total_books : ℕ)
  (num_books_first_shop : ℕ) :
  cost_first_shop = 1480 →
  books_second_shop = 55 →
  cost_second_shop = 920 →
  average_price = 20 →
  total_cost = cost_first_shop + cost_second_shop →
  total_books = total_cost / average_price →
  num_books_first_shop + books_second_shop = total_books →
  num_books_first_shop = 65 :=
by 
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end sandy_books_from_first_shop_l86_86103


namespace graph_two_intersecting_lines_l86_86374

theorem graph_two_intersecting_lines (x y : ℝ) : (x + y)^2 = x^2 + y^2 + 3 * x * y ↔ x = 0 ∨ y = 0 :=
by
  -- Placeholder for the proof
  sorry

end graph_two_intersecting_lines_l86_86374


namespace evaluate_expression_l86_86063

theorem evaluate_expression : 7^3 - 3 * 7^2 + 3 * 7 - 1 = 216 := by
  sorry

end evaluate_expression_l86_86063


namespace find_limpet_shells_l86_86420

variable (L L_shells E_shells J_shells totalShells : ℕ)

def Ed_and_Jacob_initial_shells := 2
def Ed_oyster_shells := 2
def Ed_conch_shells := 4
def Jacob_more_shells := 2
def total_shells := 30

def Ed_total_shells := L + Ed_oyster_shells + Ed_conch_shells
def Jacob_total_shells := Ed_total_shells + Jacob_more_shells

theorem find_limpet_shells
  (H : Ed_and_Jacob_initial_shells + Ed_total_shells + Jacob_total_shells = total_shells) :
  L = 7 :=
by
  sorry

end find_limpet_shells_l86_86420


namespace arithmetic_seq_k_l86_86132

theorem arithmetic_seq_k (a : ℕ → ℝ) (S : ℕ → ℝ) (k : ℕ) 
  (h1 : a 1 = -3)
  (h2 : a (k + 1) = 3 / 2)
  (h3 : S k = -12)
  (h4 : ∀ n, S n = n * (a 1 + a (n+1)) / 2):
  k = 13 :=
sorry

end arithmetic_seq_k_l86_86132


namespace quadratic_inequality_solution_range_l86_86114

theorem quadratic_inequality_solution_range (a : ℝ) :
  (¬ ∃ x : ℝ, 4 * x^2 + (a - 2) * x + 1 / 4 ≤ 0) ↔ 0 < a ∧ a < 4 :=
by
  sorry

end quadratic_inequality_solution_range_l86_86114


namespace missing_dimension_of_soap_box_l86_86734

theorem missing_dimension_of_soap_box 
  (volume_carton : ℕ) 
  (volume_soap_box : ℕ)
  (number_of_boxes : ℕ)
  (x : ℕ) 
  (h1 : volume_carton = 25 * 48 * 60) 
  (h2 : volume_soap_box = x * 6 * 5)
  (h3: number_of_boxes = 300)
  (h4 : number_of_boxes * volume_soap_box = volume_carton) : 
  x = 8 := by 
  sorry

end missing_dimension_of_soap_box_l86_86734


namespace find_A_l86_86157

theorem find_A (A : ℤ) (h : A + 10 = 15) : A = 5 :=
sorry

end find_A_l86_86157


namespace sector_angle_l86_86764

theorem sector_angle (r : ℝ) (S : ℝ) (α : ℝ) (h₁ : r = 10) (h₂ : S = 50 * π / 3) (h₃ : S = 1 / 2 * r^2 * α) : 
  α = π / 3 :=
by
  sorry

end sector_angle_l86_86764


namespace cubic_solution_identity_l86_86159

theorem cubic_solution_identity {a b c : ℕ} 
  (h1 : a + b + c = 6) 
  (h2 : ab + bc + ca = 11) 
  (h3 : abc = 6) : 
  (ab / c) + (bc / a) + (ca / b) = 49 / 6 := 
by 
  sorry

end cubic_solution_identity_l86_86159


namespace digits_of_2_pow_100_last_three_digits_of_2_pow_100_l86_86800

-- Prove that 2^100 has 31 digits.
theorem digits_of_2_pow_100 : (10^30 ≤ 2^100) ∧ (2^100 < 10^31) :=
by
  sorry

-- Prove that the last three digits of 2^100 are 376.
theorem last_three_digits_of_2_pow_100 : 2^100 % 1000 = 376 :=
by
  sorry

end digits_of_2_pow_100_last_three_digits_of_2_pow_100_l86_86800


namespace arccos_one_over_sqrt_two_eq_pi_over_four_l86_86464

theorem arccos_one_over_sqrt_two_eq_pi_over_four : Real.arccos (1 / Real.sqrt 2) = Real.pi / 4 :=
by
  sorry

end arccos_one_over_sqrt_two_eq_pi_over_four_l86_86464


namespace traffic_safety_team_eq_twice_fire_l86_86775

-- Define initial members in the teams
def t0 : ℕ := 8
def f0 : ℕ := 7

-- Define the main theorem
theorem traffic_safety_team_eq_twice_fire (x : ℕ) : t0 + x = 2 * (f0 - x) :=
by sorry

end traffic_safety_team_eq_twice_fire_l86_86775


namespace sophie_one_dollar_bills_l86_86979

theorem sophie_one_dollar_bills (x y z : ℕ) 
  (h1 : x + y + z = 55) 
  (h2 : x + 2 * y + 5 * z = 126) 
  : x = 18 := by
  sorry

end sophie_one_dollar_bills_l86_86979


namespace inequality_positive_numbers_l86_86261

theorem inequality_positive_numbers (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (x / (x + 2 * y + 3 * z)) + (y / (y + 2 * z + 3 * x)) + (z / (z + 2 * x + 3 * y)) ≤ 4 / 3 :=
by
  sorry

end inequality_positive_numbers_l86_86261


namespace single_digit_solution_l86_86644

theorem single_digit_solution :
  ∃ A : ℕ, A < 10 ∧ A^3 = 210 + A ∧ A = 6 :=
by
  existsi 6
  sorry

end single_digit_solution_l86_86644


namespace order_of_activities_l86_86469

noncomputable def fraction_liking_activity_dodgeball : ℚ := 8 / 24
noncomputable def fraction_liking_activity_barbecue : ℚ := 10 / 30
noncomputable def fraction_liking_activity_archery : ℚ := 9 / 18

theorem order_of_activities :
  (fraction_liking_activity_archery > fraction_liking_activity_dodgeball) ∧
  (fraction_liking_activity_archery > fraction_liking_activity_barbecue) ∧
  (fraction_liking_activity_dodgeball = fraction_liking_activity_barbecue) :=
by
  sorry

end order_of_activities_l86_86469


namespace find_ruv_l86_86620

theorem find_ruv (u v : ℝ) : 
  (∃ u v : ℝ, 
    (3 + 8 * u + 5, 1 - 4 * u + 2) = (4 + -3 * v + 5, 2 + 4 * v + 2)) →
  (u = -1/2 ∧ v = -1) :=
by
  intros H
  sorry

end find_ruv_l86_86620


namespace factor_expression_l86_86172

theorem factor_expression (x : ℝ) : 
  4 * x * (x - 5) + 6 * (x - 5) = (4 * x + 6) * (x - 5) :=
by 
  sorry

end factor_expression_l86_86172


namespace max_ab_bc_cd_l86_86112

theorem max_ab_bc_cd {a b c d : ℝ} (h_nonneg : 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ 0 ≤ d)
  (h_sum : a + b + c + d = 200) (h_a : a = 2 * d) : 
  ab + bc + cd ≤ 14166.67 :=
sorry

end max_ab_bc_cd_l86_86112


namespace original_number_is_correct_l86_86128

theorem original_number_is_correct (x : ℝ) (h : 10 * x = x + 34.65) : x = 3.85 :=
sorry

end original_number_is_correct_l86_86128


namespace equation_of_chord_line_l86_86722

theorem equation_of_chord_line (m n s t : ℝ)
  (h₀ : m > 0) (h₁ : n > 0) (h₂ : s > 0) (h₃ : t > 0)
  (h₄ : m + n = 3)
  (h₅ : m / s + n / t = 1)
  (h₆ : m < n)
  (h₇ : s + t = 3 + 2 * Real.sqrt 2)
  (h₈ : ∃ x1 x2 y1 y2 : ℝ, 
        (x1 + x2) / 2 = m ∧ (y1 + y2) / 2 = n ∧
        x1 ^ 2 / 4 + y1 ^ 2 / 16 = 1 ∧
        x2 ^ 2 / 4 + y2 ^ 2 / 16 = 1) 
  : 2 * m + n - 4 = 0 := sorry

end equation_of_chord_line_l86_86722


namespace consecutive_tree_distance_l86_86788

theorem consecutive_tree_distance (yard_length : ℕ) (num_trees : ℕ) (distance : ℚ)
  (h1 : yard_length = 520) 
  (h2 : num_trees = 40) :
  distance = yard_length / (num_trees - 1) :=
by
  -- Proof steps would go here
  sorry

end consecutive_tree_distance_l86_86788


namespace remaining_milk_correct_l86_86832

def arranged_milk : ℝ := 21.52
def sold_milk : ℝ := 12.64
def remaining_milk (total : ℝ) (sold : ℝ) : ℝ := total - sold

theorem remaining_milk_correct :
  remaining_milk arranged_milk sold_milk = 8.88 :=
by
  sorry

end remaining_milk_correct_l86_86832


namespace q_minus_r_l86_86594

noncomputable def problem (x : ℝ) : Prop :=
  (5 * x - 15) / (x^2 + x - 20) = x + 3

def q_and_r (q r : ℝ) : Prop :=
  q ≠ r ∧ problem q ∧ problem r ∧ q > r

theorem q_minus_r (q r : ℝ) (h : q_and_r q r) : q - r = 2 :=
  sorry

end q_minus_r_l86_86594


namespace inequality_solution_set_l86_86251

theorem inequality_solution_set (x : ℝ) : (3 - 2 * x) * (x + 1) ≤ 0 ↔ (x < -1) ∨ (x ≥ 3 / 2) :=
  sorry

end inequality_solution_set_l86_86251


namespace sin_double_angle_identity_l86_86496

theorem sin_double_angle_identity: 2 * Real.sin (15 * Real.pi / 180) * Real.cos (15 * Real.pi / 180) = 1 / 2 :=
by
  sorry

end sin_double_angle_identity_l86_86496


namespace gcd_256_180_600_l86_86673

theorem gcd_256_180_600 : Nat.gcd (Nat.gcd 256 180) 600 = 4 :=
by
  -- This proof is marked as 'sorry' to indicate it is a place holder
  sorry

end gcd_256_180_600_l86_86673


namespace bank_exceeds_1600cents_in_9_days_after_Sunday_l86_86977

theorem bank_exceeds_1600cents_in_9_days_after_Sunday
  (a : ℕ)
  (r : ℕ)
  (initial_deposit : ℕ)
  (days_after_sunday : ℕ)
  (geometric_series : ℕ -> ℕ)
  (sum_geometric_series : ℕ -> ℕ)
  (geo_series_definition : ∀(n : ℕ), geometric_series n = 5 * 2^n)
  (sum_geo_series_definition : ∀(n : ℕ), sum_geometric_series n = 5 * (2^n - 1))
  (exceeds_condition : ∀(n : ℕ), sum_geometric_series n > 1600 -> n >= 9) :
  days_after_sunday = 9 → a = 5 → r = 2 → initial_deposit = 5 → days_after_sunday = 9 → geometric_series 1 = 10 → sum_geometric_series 9 > 1600 :=
by sorry

end bank_exceeds_1600cents_in_9_days_after_Sunday_l86_86977


namespace solve_system_l86_86203

variable {R : Type*} [CommRing R] {a b c x y z : R}

theorem solve_system (h : a ≠ b ∧ b ≠ c ∧ c ≠ a) 
  (h₁ : z + a*y + a^2*x + a^3 = 0) 
  (h₂ : z + b*y + b^2*x + b^3 = 0) 
  (h₃ : z + c*y + c^2*x + c^3 = 0) :
  x = -(a + b + c) ∧ y = (a * b + a * c + b * c) ∧ z = -(a * b * c) := 
sorry

end solve_system_l86_86203


namespace matrix_pow_sub_l86_86754

open Matrix

noncomputable def B : Matrix (Fin 2) (Fin 2) ℚ := !![3, 4; 0, 2]

theorem matrix_pow_sub : 
  B^10 - 3 • B^9 = !![0, 4; 0, -1] := 
by
  sorry

end matrix_pow_sub_l86_86754


namespace smallest_multiple_of_7_not_particular_l86_86402

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.foldr (λ d acc => acc + d) 0

def is_particular_integer (n : ℕ) : Prop :=
  n % (sum_of_digits n) ^ 2 = 0

theorem smallest_multiple_of_7_not_particular :
  ∃ n, n > 0 ∧ n % 7 = 0 ∧ ¬ is_particular_integer n ∧ ∀ m, m > 0 ∧ m % 7 = 0 ∧ ¬ is_particular_integer m → n ≤ m :=
  by
    use 7
    sorry

end smallest_multiple_of_7_not_particular_l86_86402


namespace paint_needed_for_snake_l86_86816

open Nat

def total_paint (paint_per_segment segments additional_paint : Nat) : Nat :=
  paint_per_segment * segments + additional_paint

theorem paint_needed_for_snake :
  total_paint 240 336 20 = 80660 :=
by
  sorry

end paint_needed_for_snake_l86_86816


namespace decimal_to_fraction_l86_86126

theorem decimal_to_fraction : 2.36 = 59 / 25 :=
by
  sorry

end decimal_to_fraction_l86_86126


namespace mod_inverse_3_40_l86_86433

theorem mod_inverse_3_40 : 3 * 27 % 40 = 1 := by
  sorry

end mod_inverse_3_40_l86_86433


namespace find_seventh_value_l86_86066

theorem find_seventh_value (x₁ x₂ x₃ x₄ x₅ x₆ : ℝ)
  (h₁ : x₁ + 3*x₂ + 5*x₃ + 7*x₄ + 9*x₅ + 11*x₆ = 0)
  (h₂ : 3*x₁ + 5*x₂ + 7*x₃ + 9*x₄ + 11*x₅ + 13*x₆ = 10)
  (h₃ : 5*x₁ + 7*x₂ + 9*x₃ + 11*x₄ + 13*x₅ + 15*x₆ = 100) :
  7*x₁ + 9*x₂ + 11*x₃ + 13*x₄ + 15*x₅ + 17*x₆ = 210 :=
sorry

end find_seventh_value_l86_86066


namespace num_ordered_pairs_eq_seven_l86_86310

theorem num_ordered_pairs_eq_seven : ∃ n, n = 7 ∧ ∀ (x y : ℕ), (x * y = 64) → (x > 0 ∧ y > 0) → n = 7 :=
by
  sorry

end num_ordered_pairs_eq_seven_l86_86310


namespace additional_money_needed_l86_86748

-- Define the initial conditions as assumptions
def initial_bales : ℕ := 15
def previous_cost_per_bale : ℕ := 20
def multiplier : ℕ := 3
def new_cost_per_bale : ℕ := 27

-- Define the problem statement
theorem additional_money_needed :
  let initial_cost := initial_bales * previous_cost_per_bale 
  let new_bales := initial_bales * multiplier
  let new_cost := new_bales * new_cost_per_bale
  new_cost - initial_cost = 915 :=
by
  sorry

end additional_money_needed_l86_86748


namespace alloy_problem_l86_86500

theorem alloy_problem (x : ℝ) (h1 : 0.12 * x + 0.08 * 30 = 0.09333333333333334 * (x + 30)) : x = 15 :=
by
  sorry

end alloy_problem_l86_86500


namespace intersecting_x_value_l86_86384

theorem intersecting_x_value : 
  (∃ x y : ℝ, y = 3 * x - 17 ∧ 3 * x + y = 103) → 
  (∃ x : ℝ, x = 20) :=
by
  sorry

end intersecting_x_value_l86_86384


namespace average_of_original_set_l86_86804

-- Average of 8 numbers is some value A and the average of the new set where each number is 
-- multiplied by 8 is 168. We need to show that the original average A is 21.

theorem average_of_original_set (A : ℝ) (h1 : (64 * A) / 8 = 168) : A = 21 :=
by {
  -- This is the theorem statement, we add the proof next
  sorry -- proof placeholder
}

end average_of_original_set_l86_86804


namespace honors_students_count_l86_86177

variable {total_students : ℕ}
variable {total_girls total_boys : ℕ}
variable {honors_girls honors_boys : ℕ}

axiom class_size_constraint : total_students < 30
axiom prob_girls_honors : (honors_girls : ℝ) / total_girls = 3 / 13
axiom prob_boys_honors : (honors_boys : ℝ) / total_boys = 4 / 11
axiom total_students_eq : total_students = total_girls + total_boys
axiom honors_girls_value : honors_girls = 3
axiom honors_boys_value : honors_boys = 4

theorem honors_students_count : 
  honors_girls + honors_boys = 7 :=
by
  sorry

end honors_students_count_l86_86177


namespace borrowed_sheets_l86_86071

theorem borrowed_sheets (sheets borrowed: ℕ) (average_page : ℝ) 
  (total_pages : ℕ := 80) (pages_per_sheet : ℕ := 2) (total_sheets : ℕ := 40) 
  (h1 : borrowed ≤ total_sheets)
  (h2 : sheets = total_sheets - borrowed)
  (h3 : average_page = 26) : borrowed = 17 :=
sorry 

end borrowed_sheets_l86_86071


namespace diagonals_in_eight_sided_polygon_l86_86609

-- Definitions based on the conditions
def n := 8  -- Number of sides
def right_angles := 2  -- Number of right angles

-- Calculating the number of diagonals using the formula
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

-- Lean statement for the problem
theorem diagonals_in_eight_sided_polygon : num_diagonals n = 20 :=
by
  -- Substitute n = 8 into the formula and simplify
  sorry

end diagonals_in_eight_sided_polygon_l86_86609


namespace weight_of_empty_carton_l86_86280

theorem weight_of_empty_carton
    (half_full_carton_weight : ℕ)
    (full_carton_weight : ℕ)
    (h1 : half_full_carton_weight = 5)
    (h2 : full_carton_weight = 8) :
  full_carton_weight - 2 * (full_carton_weight - half_full_carton_weight) = 2 :=
by
  sorry

end weight_of_empty_carton_l86_86280


namespace smallest_number_of_students_l86_86548

theorem smallest_number_of_students (n9 n7 n8 : ℕ) (h7 : 9 * n7 = 7 * n9) (h8 : 5 * n8 = 9 * n9) :
  n9 + n7 + n8 = 134 :=
by
  -- Skipping proof with sorry
  sorry

end smallest_number_of_students_l86_86548


namespace find_m_value_l86_86727

theorem find_m_value (x m : ℝ)
  (h1 : -3 * x = -5 * x + 4)
  (h2 : m^x - 9 = 0) :
  m = 3 ∨ m = -3 := 
sorry

end find_m_value_l86_86727


namespace intersection_value_of_a_l86_86005

theorem intersection_value_of_a (a : ℝ) (A B : Set ℝ) 
  (hA : A = {0, 1, 3})
  (hB : B = {a + 1, a^2 + 2})
  (h_inter : A ∩ B = {1}) : 
  a = 0 :=
by
  sorry

end intersection_value_of_a_l86_86005


namespace problem_b_amount_l86_86240

theorem problem_b_amount (a b : ℝ) (h1 : a + b = 1210) (h2 : (4/5) * a = (2/3) * b) : b = 453.75 :=
sorry

end problem_b_amount_l86_86240


namespace garden_roller_diameter_l86_86040

theorem garden_roller_diameter 
  (length : ℝ) 
  (total_area : ℝ) 
  (num_revolutions : ℕ) 
  (pi : ℝ) 
  (A : length = 2)
  (B : total_area = 37.714285714285715)
  (C : num_revolutions = 5)
  (D : pi = 22 / 7) : 
  ∃ d : ℝ, d = 1.2 :=
by
  sorry

end garden_roller_diameter_l86_86040


namespace meat_purchase_l86_86253

theorem meat_purchase :
  ∃ x y : ℕ, 16 * x = y + 25 ∧ 8 * x = y - 15 ∧ y / x = 11 :=
by
  sorry

end meat_purchase_l86_86253


namespace number_of_b_values_l86_86931

theorem number_of_b_values (b : ℤ) :
  (∃ (x1 x2 x3 : ℤ), ∀ (x : ℤ), x^2 + b * x + 6 ≤ 0 ↔ x = x1 ∨ x = x2 ∨ x = x3) ↔ (b = -6 ∨ b = -5 ∨ b = 5 ∨ b = 6) :=
by
  sorry

end number_of_b_values_l86_86931


namespace chips_probability_l86_86757

/-- A bag contains 4 green, 3 orange, and 5 blue chips. If the 12 chips are randomly drawn from
    the bag, one at a time and without replacement, the probability that the chips are drawn such
    that the 4 green chips are drawn consecutively, the 3 orange chips are drawn consecutively,
    and the 5 blue chips are drawn consecutively, but not necessarily in the green-orange-blue
    order, is 1/4620. -/
theorem chips_probability :
  let total_chips := 12
  let factorial := Nat.factorial
  let favorable_outcomes := (factorial 3) * (factorial 4) * (factorial 3) * (factorial 5)
  let total_outcomes := factorial total_chips
  favorable_outcomes / total_outcomes = 1 / 4620 :=
by
  -- proof goes here, but we skip it
  sorry

end chips_probability_l86_86757


namespace polygon_interior_exterior_relation_l86_86996

theorem polygon_interior_exterior_relation (n : ℕ) 
  (h1 : (n-2) * 180 = 3 * 360) 
  (h2 : n ≥ 3) :
  n = 8 :=
by
  sorry

end polygon_interior_exterior_relation_l86_86996


namespace intersection_is_correct_l86_86322

noncomputable def A : Set ℝ := {x | -2 < x ∧ x < 2}

noncomputable def B : Set ℝ := {x | x^2 - 5 * x - 6 < 0}

theorem intersection_is_correct : A ∩ B = {x | -1 < x ∧ x < 2} := 
by { sorry }

end intersection_is_correct_l86_86322


namespace bruce_goals_l86_86133

theorem bruce_goals (B M : ℕ) (h1 : M = 3 * B) (h2 : B + M = 16) : B = 4 :=
by {
  -- Omitted proof
  sorry
}

end bruce_goals_l86_86133


namespace waffle_bowl_more_scoops_l86_86695

-- Definitions based on conditions
def single_cone_scoops : ℕ := 1
def banana_split_scoops : ℕ := 3 * single_cone_scoops
def double_cone_scoops : ℕ := 2 * single_cone_scoops
def total_scoops : ℕ := 10
def remaining_scoops : ℕ := total_scoops - (banana_split_scoops + single_cone_scoops + double_cone_scoops)

-- Question: Prove that the waffle bowl has 1 more scoop than the banana split
theorem waffle_bowl_more_scoops : remaining_scoops - banana_split_scoops = 1 := by
  have h1 : single_cone_scoops = 1 := rfl
  have h2 : banana_split_scoops = 3 * single_cone_scoops := rfl
  have h3 : double_cone_scoops = 2 * single_cone_scoops := rfl
  have h4 : total_scoops = 10 := rfl
  have h5 : remaining_scoops = total_scoops - (banana_split_scoops + single_cone_scoops + double_cone_scoops) := rfl
  sorry

end waffle_bowl_more_scoops_l86_86695


namespace volume_of_prism_l86_86426

theorem volume_of_prism (x y z : ℝ) (hx : x * y = 28) (hy : x * z = 45) (hz : y * z = 63) : x * y * z = 282 := by
  sorry

end volume_of_prism_l86_86426


namespace number_of_valid_pairs_is_343_l86_86842

-- Define the given problem conditions
def given_number : Nat := 1003003001

-- Define the expression for LCM calculation
def LCM (x y : Nat) : Nat := (x * y) / (Nat.gcd x y)

-- Define the prime factorization of the given number
def is_prime_factorization_correct : Prop :=
  given_number = 7^3 * 11^3 * 13^3

-- Define x and y form as described
def is_valid_form (x y : Nat) : Prop :=
  ∃ (a b c d e f : ℕ), x = 7^a * 11^b * 13^c ∧ y = 7^d * 11^e * 13^f

-- Define the LCM condition for the ordered pairs
def meets_lcm_condition (x y : Nat) : Prop :=
  LCM x y = given_number

-- State the theorem to prove an equivalent problem
theorem number_of_valid_pairs_is_343 : is_prime_factorization_correct →
  (∃ (n : ℕ), n = 343 ∧ 
    (∀ (x y : ℕ), is_valid_form x y → meets_lcm_condition x y → x > 0 → y > 0 → True)
  ) :=
by
  intros h
  use 343
  sorry

end number_of_valid_pairs_is_343_l86_86842


namespace relationship_between_a_b_l86_86892

theorem relationship_between_a_b (a b x : ℝ) (h1 : 2 * x = a + b) (h2 : 2 * x^2 = a^2 - b^2) : 
  a = -b ∨ a = 3 * b :=
  sorry

end relationship_between_a_b_l86_86892


namespace combined_river_length_estimate_l86_86134

def river_length_GSA := 402 
def river_error_GSA := 0.5 
def river_prob_error_GSA := 0.04 

def river_length_AWRA := 403 
def river_error_AWRA := 0.5 
def river_prob_error_AWRA := 0.04 

/-- 
Given the measurements from GSA and AWRA, 
the combined estimate of the river's length, Rio-Coralio, is 402.5 km,
and the probability of error for this combined estimate is 0.04.
-/
theorem combined_river_length_estimate :
  ∃ l : ℝ, l = 402.5 ∧ ∀ p : ℝ, (p = 0.04) :=
sorry

end combined_river_length_estimate_l86_86134


namespace time_difference_between_car_and_minivan_arrival_l86_86895

variable (car_speed : ℝ := 40)
variable (minivan_speed : ℝ := 50)
variable (pass_time : ℝ := 1 / 6) -- in hours

theorem time_difference_between_car_and_minivan_arrival :
  (60 * (1 / 6 - (20 / 3 / 50))) = 2 := sorry

end time_difference_between_car_and_minivan_arrival_l86_86895


namespace emily_olivia_books_l86_86447

theorem emily_olivia_books (shared_books total_books_emily books_olivia_not_in_emily : ℕ)
  (h1 : shared_books = 15)
  (h2 : total_books_emily = 23)
  (h3 : books_olivia_not_in_emily = 8) : (total_books_emily - shared_books + books_olivia_not_in_emily = 16) :=
by
  sorry

end emily_olivia_books_l86_86447


namespace isosceles_triangle_l86_86086

noncomputable def triangle_is_isosceles (A B C a b c : ℝ) (h_triangle : a = 2 * b * Real.cos C) : Prop :=
  ∃ (A B C : ℝ), (B = C) ∧ (a = 2 * b * Real.cos C)

theorem isosceles_triangle
  (A B C a b c : ℝ)
  (h_sides : a = 2 * b * Real.cos C)
  (h_triangle : ∃ (A B C : ℝ), (B = C) ∧ (a = 2 * b * Real.cos C)) :
  B = C :=
sorry

end isosceles_triangle_l86_86086


namespace expand_expression_l86_86022

variable (x y z : ℕ)

theorem expand_expression (x y z: ℕ) : 
  (x + 10) * (3 * y + 5 * z + 15) = 3 * x * y + 5 * x * z + 15 * x + 30 * y + 50 * z + 150 :=
by
  sorry

end expand_expression_l86_86022


namespace lottery_probability_l86_86702

theorem lottery_probability (p: ℝ) :
  (∀ n, 1 ≤ n ∧ n ≤ 15 → p = 2/3) →
  (true → p = 0.6666666666666666) →
  p = 2/3 :=
by
  intros h h'
  sorry

end lottery_probability_l86_86702


namespace algebraic_expression_value_l86_86646

-- Define the equation and its roots.
def quadratic_eq (x : ℝ) : Prop := 2 * x^2 - 3 * x + 1 = 0

def is_root (x : ℝ) : Prop := quadratic_eq x

-- The main theorem.
theorem algebraic_expression_value (x1 x2 : ℝ) (h1 : is_root x1) (h2 : is_root x2) :
  (x1 + x2) / (1 + x1 * x2) = 1 :=
sorry

end algebraic_expression_value_l86_86646


namespace coin_probability_l86_86319

theorem coin_probability :
  let PA := 3/4
  let PB := 1/2
  let PC := 1/4
  (PA * PB * (1 - PC)) = 9/32 :=
by
  sorry

end coin_probability_l86_86319


namespace jackson_fishes_per_day_l86_86014

def total_fishes : ℕ := 90
def jonah_per_day : ℕ := 4
def george_per_day : ℕ := 8
def competition_days : ℕ := 5

def jackson_per_day (J : ℕ) : Prop :=
  (total_fishes - (jonah_per_day * competition_days + george_per_day * competition_days)) / competition_days = J

theorem jackson_fishes_per_day : jackson_per_day 6 :=
  by
    sorry

end jackson_fishes_per_day_l86_86014


namespace f_eq_2x_pow_5_l86_86075

def f (x : ℝ) : ℝ := (2*x + 1)^5 - 5*(2*x + 1)^4 + 10*(2*x + 1)^3 - 10*(2*x + 1)^2 + 5*(2*x + 1) - 1

theorem f_eq_2x_pow_5 (x : ℝ) : f x = (2*x)^5 :=
by
  sorry

end f_eq_2x_pow_5_l86_86075


namespace solve_diophantine_l86_86140

theorem solve_diophantine : ∃ (x y : ℕ) (t : ℤ), x = 4 - 43 * t ∧ y = 6 - 65 * t ∧ t ≤ 0 ∧ 65 * x - 43 * y = 2 :=
by
  sorry

end solve_diophantine_l86_86140


namespace min_x2_y2_eq_16_then_product_zero_l86_86590

theorem min_x2_y2_eq_16_then_product_zero
  (x y : ℝ)
  (h1 : ∃ x y : ℝ, (x^2 + y^2 = 16 ∧ ∀ a b : ℝ, a^2 + b^2 ≥ 16) ) :
  (x + 4) * (y - 4) = 0 := 
sorry

end min_x2_y2_eq_16_then_product_zero_l86_86590


namespace option_B_equals_six_l86_86435

theorem option_B_equals_six :
  (3 - (-3)) = 6 :=
by
  sorry

end option_B_equals_six_l86_86435


namespace winning_candidate_percentage_l86_86194

noncomputable def votes : List ℝ := [15236.71, 20689.35, 12359.23, 30682.49, 25213.17, 18492.93]

theorem winning_candidate_percentage :
  (List.foldr max 0 votes / (List.foldr (· + ·) 0 votes) * 100) = 25.01 :=
by
  sorry

end winning_candidate_percentage_l86_86194


namespace original_cost_of_car_l86_86896

noncomputable def original_cost (C : ℝ) : ℝ :=
  if h : C + 13000 ≠ 0 then (60900 - (C + 13000)) / (C + 13000) * 100 else 0

theorem original_cost_of_car 
  (C : ℝ) 
  (h1 : original_cost C = 10.727272727272727)
  (h2 : 60900 - (C + 13000) > 0) :
  C = 433500 :=
by
  sorry

end original_cost_of_car_l86_86896


namespace ratio_expression_value_l86_86220

theorem ratio_expression_value (p q s u : ℚ) (h1 : p / q = 5 / 2) (h2 : s / u = 11 / 7) : 
  (5 * p * s - 3 * q * u) / (7 * q * u - 2 * p * s) = -233 / 12 :=
by {
  -- Proof will be provided here.
  sorry
}

end ratio_expression_value_l86_86220


namespace large_planks_need_15_nails_l86_86509

-- Definitions based on given conditions
def total_nails : ℕ := 20
def small_planks_nails : ℕ := 5

-- Question: How many nails do the large planks need together?
-- Prove that the large planks need 15 nails together given the conditions.
theorem large_planks_need_15_nails : total_nails - small_planks_nails = 15 :=
by
  sorry

end large_planks_need_15_nails_l86_86509


namespace ball_hits_ground_at_correct_time_l86_86953

def initial_velocity : ℝ := 7
def initial_height : ℝ := 10

-- The height function as given by the condition
def height_function (t : ℝ) : ℝ := -4.9 * t^2 + initial_velocity * t + initial_height

-- Statement
theorem ball_hits_ground_at_correct_time :
  ∃ t : ℝ, height_function t = 0 ∧ t = 2313 / 1000 :=
by
  sorry

end ball_hits_ground_at_correct_time_l86_86953


namespace diamond_calculation_l86_86369

def diamond (a b : ℚ) : ℚ := (a - b) / (1 + a * b)

theorem diamond_calculation : diamond 1 (diamond 2 (diamond 3 (diamond 4 5))) = 87 / 59 :=
by
  sorry

end diamond_calculation_l86_86369


namespace base_case_n_equals_1_l86_86242

variable {a : ℝ}
variable {n : ℕ}

theorem base_case_n_equals_1 (h1 : a ≠ 1) (h2 : n = 1) : 1 + a = 1 + a :=
by
  sorry

end base_case_n_equals_1_l86_86242


namespace find_a5_l86_86095

-- Define the geometric sequence and the given conditions
def geom_sequence (a : ℕ → ℝ) (r : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * r

-- Define the conditions for our problem
def conditions (a : ℕ → ℝ) :=
  geom_sequence a 2 ∧ (∀ n, 0 < a n) ∧ a 3 * a 11 = 16

-- Our goal is to prove that a_5 = 1
theorem find_a5 (a : ℕ → ℝ) (h : conditions a) : a 5 = 1 := 
by 
  sorry

end find_a5_l86_86095


namespace range_of_p_l86_86922

noncomputable def proof_problem (p : ℝ) : Prop :=
  (∀ x : ℝ, (4 * x + p < 0) → (x < -1 ∨ x > 2)) → (p ≥ 4)

theorem range_of_p (p : ℝ) : proof_problem p :=
by
  intros h
  sorry

end range_of_p_l86_86922


namespace exactly_1_male_and_exactly_2_female_mutually_exclusive_not_complementary_l86_86584

-- Definitions based on the given conditions
def male_students := 3
def female_students := 2
def total_students := male_students + female_students

def at_least_1_male_event := ∃ (n : ℕ), n ≥ 1 ∧ n ≤ male_students
def all_female_event := ∀ (n : ℕ), n ≤ female_students
def at_least_1_female_event := ∃ (n : ℕ), n ≥ 1 ∧ n ≤ female_students
def all_male_event := ∀ (n : ℕ), n ≤ male_students
def exactly_1_male_event := ∃ (n : ℕ), n = 1 ∧ n ≤ male_students
def exactly_2_female_event := ∃ (n : ℕ), n = 2 ∧ n ≤ female_students

def mutually_exclusive (e1 e2 : Prop) : Prop := ¬ (e1 ∧ e2)
def complementary (e1 e2 : Prop) : Prop := e1 ∧ ¬ e2 ∨ ¬ e1 ∧ e2

-- Statement of the problem
theorem exactly_1_male_and_exactly_2_female_mutually_exclusive_not_complementary :
  mutually_exclusive exactly_1_male_event exactly_2_female_event ∧ 
  ¬ complementary exactly_1_male_event exactly_2_female_event :=
by
  sorry

end exactly_1_male_and_exactly_2_female_mutually_exclusive_not_complementary_l86_86584


namespace actual_time_when_car_clock_shows_10PM_l86_86943

def car_clock_aligned (aligned_time wristwatch_time : ℕ) : Prop :=
  aligned_time = wristwatch_time

def car_clock_time (rate: ℚ) (hours_elapsed_real_time hours_elapsed_car_time : ℚ) : Prop :=
  rate = hours_elapsed_car_time / hours_elapsed_real_time

def actual_time (current_car_time car_rate : ℚ) : ℚ :=
  current_car_time / car_rate

theorem actual_time_when_car_clock_shows_10PM :
  let accurate_start_time := 9 -- 9:00 AM
  let car_start_time := 9 -- Synchronized at 9:00 AM
  let wristwatch_time_wristwatch := 13 -- 1:00 PM in hours
  let car_time_car := 13 + 48 / 60 -- 1:48 PM in hours
  let rate := car_time_car / wristwatch_time_wristwatch
  let current_car_time := 22 -- 10:00 PM in hours
  let real_time := actual_time current_car_time rate
  real_time = 19.8333 := -- which converts to 7:50 PM (Option B)
sorry

end actual_time_when_car_clock_shows_10PM_l86_86943


namespace bucket_weight_l86_86276

theorem bucket_weight (x y p q : ℝ) 
  (h1 : x + (3 / 4) * y = p) 
  (h2 : x + (1 / 3) * y = q) :
  x + (5 / 6) * y = (6 * p - q) / 5 :=
sorry

end bucket_weight_l86_86276


namespace problem_correctness_l86_86684

variable (f : ℝ → ℝ)
variable (h₀ : ∀ x, f x > 0)
variable (h₁ : ∀ a b, f a * f b = f (a + b))

theorem problem_correctness :
  (f 0 = 1) ∧
  (∀ a, f (-a) = 1 / f a) ∧
  (∀ a, f a = (f (3 * a)) ^ (1 / 3)) :=
by 
  -- Using the hypotheses provided
  sorry

end problem_correctness_l86_86684


namespace group_division_l86_86097

theorem group_division (total_students groups_per_group : ℕ) (h1 : total_students = 30) (h2 : groups_per_group = 5) : 
  (total_students / groups_per_group) = 6 := 
by 
  sorry

end group_division_l86_86097


namespace fish_to_apples_l86_86227

variable {Fish Loaf Rice Apple : Type}
variable (f : Fish → ℝ) (l : Loaf → ℝ) (r : Rice → ℝ) (a : Apple → ℝ)
variable (F : Fish) (L : Loaf) (A : Apple) (R : Rice)

-- Conditions
axiom cond1 : 4 * f F = 3 * l L
axiom cond2 : l L = 5 * r R
axiom cond3 : r R = 2 * a A

-- Proof statement
theorem fish_to_apples : f F = 7.5 * a A :=
by
  sorry

end fish_to_apples_l86_86227


namespace range_of_x_l86_86834

theorem range_of_x (x p : ℝ) (h₀ : 0 ≤ p ∧ p ≤ 4) :
  x^2 + p * x > 4 * x + p - 3 → (x < 1 ∨ x > 3) :=
sorry

end range_of_x_l86_86834


namespace max_squares_on_checkerboard_l86_86658

theorem max_squares_on_checkerboard (n : ℕ) (h1 : n = 7) (h2 : ∀ s : ℕ, s = 2) : ∃ max_squares : ℕ, max_squares = 18 := sorry

end max_squares_on_checkerboard_l86_86658


namespace find_b_l86_86330

-- Define the conditions as hypotheses
def f (b : ℝ) (x : ℝ) : ℝ := x^3 + 2*x^2 + b*x - 3

theorem find_b (x₁ x₂ b : ℝ) (h₁ : x₁ ≠ x₂)
  (h₂ : 3 * x₁^2 + 4 * x₁ + b = 0)
  (h₃ : 3 * x₂^2 + 4 * x₂ + b = 0)
  (h₄ : x₁^2 + x₂^2 = 34 / 9) :
  b = -3 :=
by
  -- Proof will be inserted here
  sorry

end find_b_l86_86330


namespace find_first_term_l86_86284

noncomputable def first_term_of_arithmetic_sequence : ℝ := -19.2

theorem find_first_term
  (a d : ℝ)
  (h1 : 50 * (2 * a + 99 * d) = 1050)
  (h2 : 50 * (2 * a + 199 * d) = 4050) :
  a = first_term_of_arithmetic_sequence :=
by
  -- Given conditions
  have h1' : 2 * a + 99 * d = 21 := by sorry
  have h2' : 2 * a + 199 * d = 81 := by sorry
  -- Solve for d
  have hd : d = 0.6 := by sorry
  -- Substitute d into h1'
  have h_subst : 2 * a + 99 * 0.6 = 21 := by sorry
  -- Solve for a
  have ha : a = -19.2 := by sorry
  exact ha

end find_first_term_l86_86284


namespace tamia_bell_pepper_pieces_l86_86864

def total_pieces (n k p : Nat) : Nat :=
  let slices := n * k
  let half_slices := slices / 2
  let smaller_pieces := half_slices * p
  let total := half_slices + smaller_pieces
  total

theorem tamia_bell_pepper_pieces :
  total_pieces 5 20 3 = 200 :=
by
  sorry

end tamia_bell_pepper_pieces_l86_86864


namespace fill_question_mark_l86_86829

def sudoku_grid : Type := 
  List (List (Option ℕ))

def initial_grid : sudoku_grid := 
  [ [some 3, none, none, none],
    [none, none, none, some 1], 
    [none, none, some 2, none], 
    [some 1, none, none, none] ]

def valid_sudoku (grid : sudoku_grid) : Prop :=
  -- Ensure the grid is a valid 4x4 Sudoku grid
  -- Adding necessary constraints for rows, columns and 2x2 subgrids.
  sorry

def solve_sudoku (grid : sudoku_grid) : sudoku_grid :=
  -- Function that solves the Sudoku (not implemented for this proof statement)
  sorry

theorem fill_question_mark : solve_sudoku initial_grid = 
  [ [some 3, some 2, none, none],
    [none, none, none, some 1], 
    [none, none, some 2, none], 
    [some 1, none, none, none] ] :=
  sorry

end fill_question_mark_l86_86829


namespace sum_f_eq_28743_l86_86329

def f (n : ℕ) : ℕ := 4 * n ^ 3 - 6 * n ^ 2 + 4 * n + 13

theorem sum_f_eq_28743 : (Finset.range 13).sum (λ n => f (n + 1)) = 28743 :=
by
  -- Placeholder for actual proof
  sorry

end sum_f_eq_28743_l86_86329


namespace prime_p4_minus_one_sometimes_divisible_by_48_l86_86192

theorem prime_p4_minus_one_sometimes_divisible_by_48 (p : ℕ) (hp : Nat.Prime p) (hge : p ≥ 7) : 
  ∃ k : ℕ, k ≥ 1 ∧ 48 ∣ p^4 - 1 :=
sorry

end prime_p4_minus_one_sometimes_divisible_by_48_l86_86192


namespace ratio_M_N_l86_86368

variables {M Q P N R : ℝ}

-- Conditions
def condition1 : M = 0.40 * Q := sorry
def condition2 : Q = 0.25 * P := sorry
def condition3 : N = 0.75 * R := sorry
def condition4 : R = 0.60 * P := sorry

-- Theorem to prove
theorem ratio_M_N : M / N = 2 / 9 := sorry

end ratio_M_N_l86_86368


namespace river_current_speed_l86_86545

/--
Given conditions:
- The rower realized the hat was missing 15 minutes after passing under the bridge.
- The rower caught the hat 15 minutes later.
- The total distance the hat traveled from the bridge is 1 kilometer.
Prove that the speed of the river current is 2 km/h.
-/
theorem river_current_speed (t1 t2 d : ℝ) (h_t1 : t1 = 15 / 60) (h_t2 : t2 = 15 / 60) (h_d : d = 1) : 
  d / (t1 + t2) = 2 := by
sorry

end river_current_speed_l86_86545


namespace motorcyclist_average_speed_BC_l86_86725

theorem motorcyclist_average_speed_BC :
  ∀ (d_AB : ℝ) (theta : ℝ) (d_BC_half_d_AB : ℝ) (avg_speed_trip : ℝ)
    (time_ratio_AB_BC : ℝ) (total_speed : ℝ) (t_AB : ℝ) (t_BC : ℝ),
    d_AB = 120 →
    theta = 10 →
    d_BC_half_d_AB = 1 / 2 →
    avg_speed_trip = 30 →
    time_ratio_AB_BC = 3 →
    t_AB = 4.5 →
    t_BC = 1.5 →
    t_AB = time_ratio_AB_BC * t_BC →
    avg_speed_trip = total_speed →
    total_speed = (d_AB + (d_AB * d_BC_half_d_AB)) / (t_AB + t_BC) →
    t_AB / 3 = t_BC →
    ((d_AB * d_BC_half_d_AB) / t_BC = 40) :=
by
  intros d_AB theta d_BC_half_d_AB avg_speed_trip time_ratio_AB_BC total_speed
        t_AB t_BC h1 h2 h3 h4 h5 h6 h7 h8 h9 h10 h11
  sorry

end motorcyclist_average_speed_BC_l86_86725


namespace compute_f_at_2012_l86_86342

noncomputable def B := { x : ℚ | x ≠ -1 ∧ x ≠ 0 ∧ x ≠ 2 }

noncomputable def h (x : ℚ) : ℚ := 2 - (1 / x)

noncomputable def f (x : B) : ℝ := sorry  -- As a placeholder since the definition isn't given directly

-- Main theorem
theorem compute_f_at_2012 : 
  (∀ x : B, f x + f ⟨h x, sorry⟩ = Real.log (abs (2 * (x : ℚ)))) →
  f ⟨2012, sorry⟩ = Real.log ((4024 : ℚ) / (4023 : ℚ)) :=
sorry

end compute_f_at_2012_l86_86342


namespace math_competition_rankings_l86_86883

noncomputable def rankings (n : ℕ) : ℕ → Prop := sorry

theorem math_competition_rankings :
  (∀ (A B C D E : ℕ), 
    A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧
    B ≠ C ∧ B ≠ D ∧ B ≠ E ∧
    C ≠ D ∧ C ≠ E ∧
    D ≠ E ∧
    
    -- A's guesses
    (rankings A 1 → rankings B 3 ∧ rankings C 5) →
    -- B's guesses
    (rankings B 2 → rankings E 4 ∧ rankings D 5) →
    -- C's guesses
    (rankings C 3 → rankings A 1 ∧ rankings E 4) →
    -- D's guesses
    (rankings D 4 → rankings C 1 ∧ rankings D 2) →
    -- E's guesses
    (rankings E 5 → rankings A 3 ∧ rankings D 4) →
    -- Condition that each position is guessed correctly by someone
    (∃ i, rankings A i) ∧
    (∃ i, rankings B i) ∧
    (∃ i, rankings C i) ∧
    (∃ i, rankings D i) ∧
    (∃ i, rankings E i) →
    
    -- The actual placing according to derived solution
    rankings A 1 ∧ 
    rankings D 2 ∧ 
    rankings B 3 ∧ 
    rankings E 4 ∧ 
    rankings C 5) :=
sorry

end math_competition_rankings_l86_86883


namespace calculate_expression_l86_86150

theorem calculate_expression (y : ℤ) (hy : y = 2) : (3 * y + 4)^2 = 100 :=
by
  sorry

end calculate_expression_l86_86150


namespace percentage_loss_l86_86274

theorem percentage_loss (selling_price_with_loss : ℝ)
    (desired_selling_price_for_profit : ℝ)
    (profit_percentage : ℝ) (actual_selling_price : ℝ)
    (calculated_loss_percentage : ℝ) :
    selling_price_with_loss = 16 →
    desired_selling_price_for_profit = 21.818181818181817 →
    profit_percentage = 20 →
    actual_selling_price = 18.181818181818182 →
    calculated_loss_percentage = 12 → 
    calculated_loss_percentage = (actual_selling_price - selling_price_with_loss) / actual_selling_price * 100 := 
sorry

end percentage_loss_l86_86274


namespace window_width_l86_86601

theorem window_width (h_pane_height : ℕ) (h_to_w_ratio_num : ℕ) (h_to_w_ratio_den : ℕ) (gaps : ℕ) 
(border : ℕ) (columns : ℕ) 
(panes_per_row : ℕ) (pane_height : ℕ) 
(heights_equal : h_pane_height = pane_height)
(ratio : h_to_w_ratio_num * pane_height = h_to_w_ratio_den * panes_per_row)
: columns * (h_to_w_ratio_den * pane_height / h_to_w_ratio_num) + 
  gaps + 2 * border = 57 := sorry

end window_width_l86_86601


namespace arithmetic_sequence_sum_l86_86340

theorem arithmetic_sequence_sum (a : ℕ → ℝ) (a4_eq_3 : a 4 = 3) :
  a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 = 21 :=
by
  sorry

end arithmetic_sequence_sum_l86_86340


namespace find_sum_of_money_l86_86059

theorem find_sum_of_money (P : ℝ) (H1 : P * 0.18 * 2 - P * 0.12 * 2 = 840) : P = 7000 :=
by
  sorry

end find_sum_of_money_l86_86059


namespace valid_third_side_length_l86_86643

theorem valid_third_side_length (x : ℝ) : 4 < x ∧ x < 14 ↔ (((5 : ℝ) + 9 > x) ∧ (x + 5 > 9) ∧ (x + 9 > 5)) :=
by 
  sorry

end valid_third_side_length_l86_86643


namespace total_pieces_of_clothing_l86_86631

def number_of_pieces_per_drawer : ℕ := 2
def number_of_drawers : ℕ := 4

theorem total_pieces_of_clothing : 
  (number_of_pieces_per_drawer * number_of_drawers = 8) :=
by sorry

end total_pieces_of_clothing_l86_86631


namespace complement_union_A_B_l86_86007

def is_element_of_set_A (x : ℤ) : Prop := ∃ k : ℤ, x = 3 * k + 1
def is_element_of_set_B (x : ℤ) : Prop := ∃ k : ℤ, x = 3 * k + 2
def is_element_of_complement_U (x : ℤ) : Prop := ∃ k : ℤ, x = 3 * k

theorem complement_union_A_B :
  {x : ℤ | ¬ (is_element_of_set_A x ∨ is_element_of_set_B x)} = {x : ℤ | is_element_of_complement_U x} :=
by
  sorry

end complement_union_A_B_l86_86007


namespace ratio_second_part_l86_86432

theorem ratio_second_part (first_part second_part total : ℕ) 
  (h_ratio_percent : 50 = 100 * first_part / total) 
  (h_first_part : first_part = 10) : 
  second_part = 10 := by
  have h_total : total = 2 * first_part := by sorry
  sorry

end ratio_second_part_l86_86432


namespace line_intersects_parabola_exactly_once_at_m_l86_86493

theorem line_intersects_parabola_exactly_once_at_m :
  (∃ y : ℝ, -3 * y^2 - 4 * y + 7 = m) → (∃! m : ℝ, m = 25 / 3) :=
by
  intro h
  sorry

end line_intersects_parabola_exactly_once_at_m_l86_86493


namespace roots_of_unity_cubic_l86_86115

noncomputable def countRootsOfUnityCubic (c d e : ℤ) : ℕ := sorry

theorem roots_of_unity_cubic :
  ∃ (z : ℂ) (n : ℕ), (z^n = 1) ∧ (∃ (c d e : ℤ), z^3 + c * z^2 + d * z + e = 0)
  ∧ countRootsOfUnityCubic c d e = 12 :=
sorry

end roots_of_unity_cubic_l86_86115


namespace line_b_y_intercept_l86_86721

theorem line_b_y_intercept :
  ∃ c : ℝ, (∀ x : ℝ, (-3) * x + c = -3 * x + 7) ∧ ∃ p : ℝ × ℝ, (p = (5, -2)) → -3 * 5 + c = -2 →
  c = 13 :=
by
  sorry

end line_b_y_intercept_l86_86721


namespace hyperbola_same_foci_l86_86082

-- Define the conditions for the ellipse and hyperbola
def ellipse (x y : ℝ) : Prop := (x^2 / 12) + (y^2 / 4) = 1
def hyperbola (x y m : ℝ) : Prop := (x^2 / m) - y^2 = 1

-- Statement to be proved in Lean 4
theorem hyperbola_same_foci : ∃ m : ℝ, ∀ x y : ℝ, ellipse x y → hyperbola x y m :=
by
  have a_squared := 12
  have b_squared := 4
  have c_squared := a_squared - b_squared
  have c := Real.sqrt c_squared
  have c_value : c = 2 * Real.sqrt 2 := by sorry
  let m := c^2 - 1
  exact ⟨m, by sorry⟩

end hyperbola_same_foci_l86_86082


namespace smallest_y_l86_86865

noncomputable def x : ℕ := 3 * 40 * 75

def is_perfect_cube (n : ℕ) : Prop :=
  ∃ (k : ℕ), k^3 = n

theorem smallest_y (y : ℕ) (hy : y = 3) :
  ∀ (x : ℕ), x = 3 * 40 * 75 → is_perfect_cube (x * y) :=
by
  intro x hx
  unfold is_perfect_cube
  exists 5 -- This is just a placeholder value; the proof would find the correct k
  sorry

end smallest_y_l86_86865


namespace land_area_l86_86599

theorem land_area (x : ℝ) (h : (70 * x - 800) / 1.2 * 1.6 + 800 = 80 * x) : x = 20 :=
by
  sorry

end land_area_l86_86599


namespace boat_speed_in_still_water_l86_86316

theorem boat_speed_in_still_water (b s : ℝ) (h1 : b + s = 11) (h2 : b - s = 5) : b = 8 := 
by
  /- The proof steps would go here -/
  sorry

end boat_speed_in_still_water_l86_86316


namespace ball_bounce_height_l86_86503

theorem ball_bounce_height :
  ∃ k : ℕ, (20 * (3 / 4 : ℝ)^k < 2) ∧ ∀ n < k, ¬ (20 * (3 / 4 : ℝ)^n < 2) :=
sorry

end ball_bounce_height_l86_86503


namespace inequality_not_always_correct_l86_86025

variables (x y z : ℝ) (h₁ : x > 0) (h₂ : y > 0) (h₃ : x > y) (h₄ : z > 0)

theorem inequality_not_always_correct :
  ¬ ∀ z > 0, (xz^2 / z > yz^2 / z) :=
sorry

end inequality_not_always_correct_l86_86025


namespace find_number_l86_86833

theorem find_number (x : ℝ) (h : 0.15 * 0.30 * 0.50 * x = 126) : x = 5600 := 
by
  -- Proof goes here
  sorry

end find_number_l86_86833


namespace negation_example_l86_86558

open Real

theorem negation_example : 
  ¬(∀ x : ℝ, ∃ n : ℕ, 0 < n ∧ n ≥ x^2) ↔ ∃ x : ℝ, ∀ n : ℕ, 0 < n → n < x^2 := 
  sorry

end negation_example_l86_86558


namespace weight_of_replaced_person_is_correct_l86_86905

-- Define a constant representing the number of persons in the group.
def num_people : ℕ := 10
-- Define a constant representing the weight of the new person.
def new_person_weight : ℝ := 110
-- Define a constant representing the increase in average weight when the new person joins.
def avg_weight_increase : ℝ := 5
-- Define the weight of the person who was replaced.
noncomputable def replaced_person_weight : ℝ :=
  new_person_weight - num_people * avg_weight_increase

-- Prove that the weight of the replaced person is 60 kg.
theorem weight_of_replaced_person_is_correct : replaced_person_weight = 60 :=
by
  -- Skip the detailed proof steps.
  sorry

end weight_of_replaced_person_is_correct_l86_86905


namespace calendar_sum_multiple_of_4_l86_86020

theorem calendar_sum_multiple_of_4 (a : ℕ) : 
  let top_left := a - 1
  let bottom_left := a + 6
  let bottom_right := a + 7
  top_left + a + bottom_left + bottom_right = 4 * (a + 3) :=
by
  sorry

end calendar_sum_multiple_of_4_l86_86020


namespace sector_angle_degree_measure_l86_86156

-- Define the variables and conditions
variables (θ r : ℝ)
axiom h1 : (1 / 2) * θ * r^2 = 1
axiom h2 : 2 * r + θ * r = 4

-- Define the theorem to be proved
theorem sector_angle_degree_measure (θ r : ℝ) (h1 : (1 / 2) * θ * r^2 = 1) (h2 : 2 * r + θ * r = 4) : θ = 2 :=
sorry

end sector_angle_degree_measure_l86_86156


namespace jellybeans_in_jar_l86_86486

theorem jellybeans_in_jar (num_kids_normal : ℕ) (num_absent : ℕ) (num_jellybeans_each : ℕ) (num_leftover : ℕ) 
  (h1 : num_kids_normal = 24) (h2 : num_absent = 2) (h3 : num_jellybeans_each = 3) (h4 : num_leftover = 34) : 
  (num_kids_normal - num_absent) * num_jellybeans_each + num_leftover = 100 :=
by sorry

end jellybeans_in_jar_l86_86486


namespace part1_part2_l86_86295

noncomputable def A : Set ℝ := {x | x^2 + 4 * x = 0}
noncomputable def B (a : ℝ) : Set ℝ := {x | x^2 + 2 * (a + 1) * x + a^2 - 1 = 0}

theorem part1 (a : ℝ) : A ∪ B a = B a ↔ a = 1 :=
by
  sorry

theorem part2 (a : ℝ) : A ∩ B a = B a ↔ a ≤ -1 ∨ a = 1 :=
by
  sorry

end part1_part2_l86_86295


namespace roots_cubic_eq_sum_fraction_l86_86653

theorem roots_cubic_eq_sum_fraction (p q r : ℝ)
  (h1 : p + q + r = 8)
  (h2 : p * q + p * r + q * r = 10)
  (h3 : p * q * r = 3) :
  p / (q * r + 2) + q / (p * r + 2) + r / (p * q + 2) = 8 / 69 := 
sorry

end roots_cubic_eq_sum_fraction_l86_86653


namespace lisa_max_non_a_quizzes_l86_86663

def lisa_goal : ℕ := 34
def quizzes_total : ℕ := 40
def quizzes_taken_first : ℕ := 25
def quizzes_with_a_first : ℕ := 20
def remaining_quizzes : ℕ := quizzes_total - quizzes_taken_first
def additional_a_needed : ℕ := lisa_goal - quizzes_with_a_first

theorem lisa_max_non_a_quizzes : 
  additional_a_needed ≤ remaining_quizzes → 
  remaining_quizzes - additional_a_needed ≤ 1 :=
by
  sorry

end lisa_max_non_a_quizzes_l86_86663


namespace find_k_l86_86906

-- Defining the quadratic function
def quadratic (x k : ℝ) := x^2 + (2 * k + 1) * x + k^2 + 1

-- Condition 1: The roots are distinct, implies discriminant > 0
def discriminant_positive (k : ℝ) := (2 * k + 1)^2 - 4 * (k^2 + 1) > 0

-- Condition 2: Product of roots given as 5
def product_of_roots (k : ℝ) := k^2 + 1 = 5

-- Main theorem
theorem find_k (k : ℝ) (hk1 : discriminant_positive k) (hk2 : product_of_roots k) : k = 2 := by
  sorry

end find_k_l86_86906


namespace calculation_equivalence_l86_86152

theorem calculation_equivalence : 3000 * (3000 ^ 2999) = 3000 ^ 3000 := 
by
  sorry

end calculation_equivalence_l86_86152


namespace bucket_full_weight_l86_86525

theorem bucket_full_weight (p q : ℝ) (x y : ℝ) 
    (h1 : x + (3/4) * y = p) 
    (h2 : x + (1/3) * y = q) : 
    x + y = (8 * p - 3 * q) / 5 :=
sorry

end bucket_full_weight_l86_86525


namespace total_money_difference_l86_86356

-- Define the number of quarters each sibling has
def quarters_Karen : ℕ := 32
def quarters_Christopher : ℕ := 64
def quarters_Emily : ℕ := 20
def quarters_Michael : ℕ := 12

-- Define the value of each quarter
def value_per_quarter : ℚ := 0.25

-- Prove that the total money difference between the pairs of siblings is $16.00
theorem total_money_difference : 
  (quarters_Karen - quarters_Emily) * value_per_quarter + 
  (quarters_Christopher - quarters_Michael) * value_per_quarter = 16 := by
sorry

end total_money_difference_l86_86356


namespace solve_modular_equation_l86_86390

theorem solve_modular_equation (x : ℤ) :
  (15 * x + 2) % 18 = 7 % 18 ↔ x % 6 = 1 % 6 := by
  sorry

end solve_modular_equation_l86_86390


namespace complex_number_z_l86_86742

theorem complex_number_z (z : ℂ) (h : (3 + 1 * I) * z = 4 - 2 * I) : z = 1 - I :=
by
  sorry

end complex_number_z_l86_86742


namespace calculate_neg2_add3_l86_86680

theorem calculate_neg2_add3 : (-2) + 3 = 1 :=
  sorry

end calculate_neg2_add3_l86_86680


namespace Jake_has_fewer_peaches_l86_86135

def Steven_peaches := 14
def Jill_peaches := 5
def Jake_peaches := Jill_peaches + 3

theorem Jake_has_fewer_peaches : Steven_peaches - Jake_peaches = 6 :=
by
  sorry

end Jake_has_fewer_peaches_l86_86135


namespace probability_of_selecting_cooking_l86_86393

noncomputable def probability_course_selected (num_courses : ℕ) (selected_course : ℕ) : ℚ :=
  selected_course / num_courses

theorem probability_of_selecting_cooking :
  let courses := 4
  let cooking := 1
  probability_course_selected courses cooking = 1 / 4 :=
by
  sorry

end probability_of_selecting_cooking_l86_86393


namespace min_period_f_and_max_value_g_l86_86869

open Real

noncomputable def f (x : ℝ) : ℝ := abs (sin x) + abs (cos x)
noncomputable def g (x : ℝ) : ℝ := sin x ^ 3 - sin x

theorem min_period_f_and_max_value_g :
  (∀ m : ℝ, (∀ x : ℝ, f (x + m) = f x) -> m = π / 2) ∧ 
  (∃ n : ℝ, ∀ x : ℝ, g x ≤ n ∧ (∃ x : ℝ, g x = n)) ∧ 
  (∃ mn : ℝ, mn = (π / 2) * (2 * sqrt 3 / 9)) := 
by sorry

end min_period_f_and_max_value_g_l86_86869


namespace garden_area_increase_l86_86749

noncomputable def original_garden_length : ℝ := 60
noncomputable def original_garden_width : ℝ := 20
noncomputable def original_garden_area : ℝ := original_garden_length * original_garden_width
noncomputable def original_garden_perimeter : ℝ := 2 * (original_garden_length + original_garden_width)

noncomputable def circle_radius : ℝ := original_garden_perimeter / (2 * Real.pi)
noncomputable def circle_area : ℝ := Real.pi * (circle_radius ^ 2)

noncomputable def area_increase : ℝ := circle_area - original_garden_area

theorem garden_area_increase :
  area_increase = (6400 / Real.pi) - 1200 :=
by 
  sorry -- proof goes here

end garden_area_increase_l86_86749


namespace largest_m_l86_86858

noncomputable def max_min_ab_bc_ca (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h1 : a + b + c = 9) (h2 : ab + bc + ca = 27) : ℝ :=
  min (a * b) (min (b * c) (c * a))

theorem largest_m (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (h1 : a + b + c = 9) (h2 : ab + bc + ca = 27) : max_min_ab_bc_ca a b c ha hb hc h1 h2 = 6.75 :=
by
  sorry

end largest_m_l86_86858


namespace find_q_l86_86809

theorem find_q (P J T : ℝ) (Q : ℝ) (q : ℚ) 
  (h1 : J = 0.75 * P)
  (h2 : J = 0.80 * T)
  (h3 : T = P * (1 - Q))
  (h4 : Q = q / 100) :
  q = 6.25 := 
by
  sorry

end find_q_l86_86809


namespace express_as_terminating_decimal_l86_86123

section terminating_decimal

theorem express_as_terminating_decimal
  (a b : ℚ)
  (h1 : a = 125)
  (h2 : b = 144)
  (h3 : b = 2^4 * 3^2): 
  a / b = 0.78125 := 
by 
  sorry

end terminating_decimal

end express_as_terminating_decimal_l86_86123


namespace Chris_age_l86_86521

theorem Chris_age (a b c : ℚ) 
  (h1 : a + b + c = 30)
  (h2 : c - 5 = 2 * a)
  (h3 : b = (3/4) * a - 1) :
  c = 263/11 := by
  sorry

end Chris_age_l86_86521


namespace days_to_use_up_one_bag_l86_86300

def rice_kg : ℕ := 11410
def bags : ℕ := 3260
def rice_per_day : ℚ := 0.25
def rice_per_bag : ℚ := rice_kg / bags

theorem days_to_use_up_one_bag : (rice_per_bag / rice_per_day) = 14 := by
  sorry

end days_to_use_up_one_bag_l86_86300


namespace spring_bud_cup_eq_289_l86_86483

theorem spring_bud_cup_eq_289 (x : ℕ) (h : x + x = 578) : x = 289 :=
sorry

end spring_bud_cup_eq_289_l86_86483


namespace max_result_l86_86656

-- Define the expressions as Lean definitions
def expr1 : Int := 2 + (-2)
def expr2 : Int := 2 - (-2)
def expr3 : Int := 2 * (-2)
def expr4 : Int := 2 / (-2)

-- State the theorem
theorem max_result : 
  (expr2 = 4) ∧ (expr2 > expr1) ∧ (expr2 > expr3) ∧ (expr2 > expr4) :=
by
  sorry

end max_result_l86_86656


namespace find_constants_l86_86534

theorem find_constants (a b c : ℝ) (h_neg : a < 0) (h_amp : |a| = 3) (h_period : b > 0 ∧ (2 * π / b) = 8 * π) : 
a = -3 ∧ b = 0.5 :=
by
  sorry

end find_constants_l86_86534


namespace isosceles_triangle_range_l86_86645

theorem isosceles_triangle_range (x : ℝ) (h1 : 0 < x) (h2 : 2 * x + (10 - 2 * x) = 10):
  (5 / 2) < x ∧ x < 5 :=
by
  sorry

end isosceles_triangle_range_l86_86645


namespace complex_roots_circle_radius_l86_86578

theorem complex_roots_circle_radius (z : ℂ) (h : (z + 2)^4 = 16 * z^4) :
  ∃ r : ℝ, (∀ z, (z + 2)^4 = 16 * z^4 → (z - (2/3))^2 + y^2 = r) ∧ r = 1 :=
sorry

end complex_roots_circle_radius_l86_86578


namespace emma_prob_at_least_one_correct_l86_86341

-- Define the probability of getting a question wrong
def prob_wrong : ℚ := 4 / 5

-- Define the probability of getting all five questions wrong
def prob_all_wrong : ℚ := prob_wrong ^ 5

-- Define the probability of getting at least one question correct
def prob_at_least_one_correct : ℚ := 1 - prob_all_wrong

-- Define the main theorem to be proved
theorem emma_prob_at_least_one_correct : prob_at_least_one_correct = 2101 / 3125 := by
  sorry  -- This is where the proof would go

end emma_prob_at_least_one_correct_l86_86341


namespace semicircle_perimeter_l86_86712

/-- The perimeter of a semicircle with radius 6.3 cm is approximately 32.382 cm. -/
theorem semicircle_perimeter (r : ℝ) (h : r = 6.3) : 
  (π * r + 2 * r = 32.382) :=
by
  sorry

end semicircle_perimeter_l86_86712


namespace book_has_125_pages_l86_86795

-- Define the number of pages in each chapter
def chapter1_pages : ℕ := 66
def chapter2_pages : ℕ := 35
def chapter3_pages : ℕ := 24

-- Define the total number of pages in the book
def total_pages : ℕ := chapter1_pages + chapter2_pages + chapter3_pages

-- State the theorem to prove that the total number of pages is 125
theorem book_has_125_pages : total_pages = 125 := 
by 
  -- The proof is omitted for the purpose of this task
  sorry

end book_has_125_pages_l86_86795


namespace abs_inequality_solution_l86_86671

theorem abs_inequality_solution (x : ℝ) : |2 * x - 5| > 1 ↔ x < 2 ∨ x > 3 := sorry

end abs_inequality_solution_l86_86671


namespace total_annual_interest_l86_86461

def total_amount : ℝ := 4000
def P1 : ℝ := 2800
def Rate1 : ℝ := 0.03
def Rate2 : ℝ := 0.05

def P2 : ℝ := total_amount - P1
def I1 : ℝ := P1 * Rate1
def I2 : ℝ := P2 * Rate2
def I_total : ℝ := I1 + I2

theorem total_annual_interest : I_total = 144 := by
  sorry

end total_annual_interest_l86_86461


namespace cakes_and_bread_weight_l86_86840

theorem cakes_and_bread_weight 
  (B : ℕ)
  (cake_weight : ℕ := B + 100)
  (h1 : 4 * cake_weight = 800)
  : 3 * cake_weight + 5 * B = 1100 := by
  sorry

end cakes_and_bread_weight_l86_86840


namespace sat_production_correct_highest_lowest_diff_correct_total_weekly_wage_correct_l86_86827

def avg_daily_production := 400
def weekly_planned_production := 2800
def daily_deviations := [15, -5, 21, 16, -7, 0, -8]
def total_weekly_deviation := 80

-- Calculation for sets produced on Saturday
def sat_production_exceeds_plan := total_weekly_deviation - (daily_deviations.take (daily_deviations.length - 1)).sum
def sat_production := avg_daily_production + sat_production_exceeds_plan

-- Calculation for the difference between the max and min production days
def max_deviation := max sat_production_exceeds_plan (daily_deviations.maximum.getD 0)
def min_deviation := min sat_production_exceeds_plan (daily_deviations.minimum.getD 0)
def highest_lowest_diff := max_deviation - min_deviation

-- Calculation for the weekly wage for each worker
def workers := 20
def daily_wage := 200
def basic_weekly_wage := daily_wage * 7
def additional_wage := (15 + 21 + 16 + sat_production_exceeds_plan) * 10 - (5 + 7 + 8) * 15
def total_bonus := additional_wage / workers
def total_weekly_wage := basic_weekly_wage + total_bonus

theorem sat_production_correct : sat_production = 448 := by
  sorry

theorem highest_lowest_diff_correct : highest_lowest_diff = 56 := by
  sorry

theorem total_weekly_wage_correct : total_weekly_wage = 1435 := by
  sorry

end sat_production_correct_highest_lowest_diff_correct_total_weekly_wage_correct_l86_86827


namespace Sahil_purchase_price_l86_86459

theorem Sahil_purchase_price :
  ∃ P : ℝ, (1.5 * (P + 6000) = 25500) → P = 11000 :=
sorry

end Sahil_purchase_price_l86_86459


namespace desks_in_classroom_l86_86952

theorem desks_in_classroom (d c : ℕ) (h1 : c = 4 * d) (h2 : 4 * c + 6 * d = 728) : d = 33 :=
by
  -- The proof is omitted, this placeholder is to indicate that it is required to complete the proof.
  sorry

end desks_in_classroom_l86_86952


namespace squares_circles_intersections_l86_86607

noncomputable def number_of_intersections (p1 p2 : (ℤ × ℤ)) (square_side : ℚ) (circle_radius : ℚ) : ℕ :=
sorry -- function definition placeholder

theorem squares_circles_intersections :
  let p1 := (0, 0)
  let p2 := (1009, 437)
  let square_side := (1 : ℚ) / 4
  let circle_radius := (1 : ℚ) / 8
  (number_of_intersections p1 p2 square_side circle_radius) = 526 := by
  sorry

end squares_circles_intersections_l86_86607


namespace jane_trail_mix_chocolate_chips_l86_86353

theorem jane_trail_mix_chocolate_chips (c₁ : ℝ) (c₂ : ℝ) (c₃ : ℝ) (c₄ : ℝ) (c₅ : ℝ) :
  (c₁ = 0.30) → (c₂ = 0.70) → (c₃ = 0.45) → (c₄ = 0.35) → (c₅ = 0.60) →
  c₄ = 0.35 ∧ (c₅ - c₁) * 2 = 0.40 := 
by
  intros h₁ h₂ h₃ h₄ h₅
  sorry

end jane_trail_mix_chocolate_chips_l86_86353


namespace intersection_A_B_l86_86012

def set_A : Set ℝ := {x : ℝ | |x| = x}
def set_B : Set ℝ := {x : ℝ | x^2 + x ≥ 0}
def set_intersection : Set ℝ := {x : ℝ | 0 ≤ x}

theorem intersection_A_B :
  (set_A ∩ set_B) = set_intersection :=
by
  sorry

-- You can verify if the Lean code builds successfully using Lean 4 environment.

end intersection_A_B_l86_86012


namespace servant_received_amount_l86_86482

def annual_salary := 900
def uniform_price := 100
def fraction_of_year_served := 3 / 4

theorem servant_received_amount :
  annual_salary * fraction_of_year_served + uniform_price = 775 := by
  sorry

end servant_received_amount_l86_86482


namespace unique_function_satisfying_conditions_l86_86307

theorem unique_function_satisfying_conditions :
  ∀ f : ℚ → ℚ, (f 1 = 2) → (∀ x y : ℚ, f (x * y) = f x * f y - f (x + y) + 1) → (∀ x : ℚ, f x = x + 1) :=
by
  intro f h1 hCond
  sorry

end unique_function_satisfying_conditions_l86_86307


namespace Vasya_numbers_l86_86422

theorem Vasya_numbers (x y : ℝ) (h1 : x + y = x * y) (h2 : x * y = x / y) : x = 1/2 ∧ y = -1 :=
by
  sorry

end Vasya_numbers_l86_86422


namespace drawing_two_black_balls_probability_equals_half_l86_86711

noncomputable def total_number_of_events : ℕ := 6

noncomputable def number_of_black_draw_events : ℕ := 3

noncomputable def probability_of_drawing_two_black_balls : ℚ :=
  number_of_black_draw_events / total_number_of_events

theorem drawing_two_black_balls_probability_equals_half :
  probability_of_drawing_two_black_balls = 1 / 2 :=
by
  sorry

end drawing_two_black_balls_probability_equals_half_l86_86711


namespace added_number_is_nine_l86_86239

theorem added_number_is_nine (y : ℤ) : 
  3 * (2 * 4 + y) = 51 → y = 9 :=
by
  sorry

end added_number_is_nine_l86_86239


namespace possible_integer_roots_l86_86557

-- Define the general polynomial
def polynomial (b2 b1 : ℤ) (x : ℤ) : ℤ := x ^ 3 + b2 * x ^ 2 + b1 * x - 30

-- Statement: Prove the set of possible integer roots includes exactly the divisors of -30
theorem possible_integer_roots (b2 b1 : ℤ) :
  {r : ℤ | polynomial b2 b1 r = 0} = 
  {-30, -15, -10, -6, -5, -3, -2, -1, 1, 2, 3, 5, 6, 10, 15, 30} :=
sorry

end possible_integer_roots_l86_86557


namespace sum_gcd_lcm_l86_86490

theorem sum_gcd_lcm (a b : ℕ) (h_a : a = 75) (h_b : b = 4500) :
  Nat.gcd a b + Nat.lcm a b = 4575 := by
  sorry

end sum_gcd_lcm_l86_86490


namespace square_area_less_than_circle_area_l86_86618

theorem square_area_less_than_circle_area (a : ℝ) (ha : 0 < a) :
    let S1 := (a / 4) ^ 2
    let r := a / (2 * Real.pi)
    let S2 := Real.pi * r^2
    (S1 < S2) := by
sorry

end square_area_less_than_circle_area_l86_86618


namespace evaluate_expression_l86_86458

open Nat

theorem evaluate_expression : 
  (3 * 4 * 5 * 6) * (1 / 3 + 1 / 4 + 1 / 5 + 1 / 6) = 342 := by
  sorry

end evaluate_expression_l86_86458


namespace parabola_equation_l86_86579

-- Define the conditions and the claim
theorem parabola_equation (p : ℝ) (hp : p > 0) (h_symmetry : -p / 2 = -1 / 2) : 
  (∀ x y : ℝ, x^2 = 2 * p * y ↔ x^2 = 2 * y) :=
by 
  sorry

end parabola_equation_l86_86579


namespace scientific_notation_of_graphene_l86_86083

theorem scientific_notation_of_graphene :
  0.00000000034 = 3.4 * 10^(-10) :=
sorry

end scientific_notation_of_graphene_l86_86083


namespace max_good_triplets_l86_86136

-- Define the problem's conditions
variables (k : ℕ) (h_pos : 0 < k)

-- The statement to be proven
theorem max_good_triplets : ∃ T, T = 12 * k ^ 4 := 
sorry

end max_good_triplets_l86_86136


namespace prime_p_in_range_l86_86657

theorem prime_p_in_range (p : ℕ) (prime_p : Nat.Prime p) 
    (h : ∃ a b : ℤ, a * b = -530 * p ∧ a + b = p) : 43 < p ∧ p ≤ 53 := 
sorry

end prime_p_in_range_l86_86657


namespace twenty_four_point_game_l86_86862

theorem twenty_four_point_game : (9 + 7) * 3 / 2 = 24 := by
  sorry -- Proof to be provided

end twenty_four_point_game_l86_86862


namespace determine_OP_squared_l86_86259

-- Define the given conditions
variable (O P : Point) -- Points: center O and intersection point P
variable (r : ℝ) (AB CD : ℝ) (E F : Point) -- radius, lengths of chords, midpoints of chords
variable (OE OF : ℝ) -- Distances from center to midpoints of chords
variable (EF : ℝ) -- Distance between midpoints
variable (OP : ℝ) -- Distance from center to intersection point

-- Conditions as given
axiom circle_radius : r = 30
axiom chord_AB_length : AB = 40
axiom chord_CD_length : CD = 14
axiom distance_midpoints : EF = 15
axiom distance_OE : OE = 20
axiom distance_OF : OF = 29

-- The proof problem: determine that OP^2 = 733 given the conditions
theorem determine_OP_squared :
  OP^2 = 733 :=
sorry

end determine_OP_squared_l86_86259


namespace determine_coordinates_of_M_l86_86233

def point_in_fourth_quadrant (M : ℝ × ℝ) : Prop :=
  M.1 > 0 ∧ M.2 < 0

def distance_to_x_axis (M : ℝ × ℝ) (d : ℝ) : Prop :=
  |M.2| = d

def distance_to_y_axis (M : ℝ × ℝ) (d : ℝ) : Prop :=
  |M.1| = d

theorem determine_coordinates_of_M :
  ∃ M : ℝ × ℝ, point_in_fourth_quadrant M ∧ distance_to_x_axis M 3 ∧ distance_to_y_axis M 4 ∧ M = (4, -3) :=
by
  sorry

end determine_coordinates_of_M_l86_86233


namespace correct_operation_C_l86_86923

theorem correct_operation_C (m : ℕ) : m^7 / m^3 = m^4 := by
  sorry

end correct_operation_C_l86_86923


namespace roots_of_equation_l86_86437

theorem roots_of_equation :
  (∃ (x_1 x_2 : ℝ), x_1 > x_2 ∧ (∀ x, x^2 - |x-1| - 1 = 0 ↔ x = x_1 ∨ x = x_2)) :=
sorry

end roots_of_equation_l86_86437


namespace problem_part1_problem_part2_l86_86388

theorem problem_part1 (x y : ℝ) (h1 : x - 2 * y = 3) (h2 : x^2 - 2 * x * y + 4 * y^2 = 11) :
  x * y = 1 :=
sorry

theorem problem_part2 (x y : ℝ) (h1 : x - 2 * y = 3) (h2 : x^2 - 2 * x * y + 4 * y^2 = 11) :
  x^2 * y - 2 * x * y^2 = 3 :=
sorry

end problem_part1_problem_part2_l86_86388


namespace sum_is_integer_l86_86686

theorem sum_is_integer (x y z : ℝ) (h1 : x ^ 2 = y + 2) (h2 : y ^ 2 = z + 2) (h3 : z ^ 2 = x + 2) : ∃ n : ℤ, x + y + z = n :=
  sorry

end sum_is_integer_l86_86686


namespace value_of_a_sub_b_l86_86994

theorem value_of_a_sub_b (a b : ℝ) (h1 : abs a = 8) (h2 : abs b = 5) (h3 : a > 0) (h4 : b < 0) : a - b = 13 := 
  sorry

end value_of_a_sub_b_l86_86994


namespace total_lives_l86_86555

theorem total_lives (initial_players additional_players lives_per_player : ℕ) (h1 : initial_players = 4) (h2 : additional_players = 5) (h3 : lives_per_player = 3) :
  (initial_players + additional_players) * lives_per_player = 27 :=
by
  sorry

end total_lives_l86_86555


namespace evaluate_expression_l86_86254

theorem evaluate_expression :
  (2^1 - 3 + 5^3 - 2)⁻¹ * 3 = (3 : ℚ) / 122 :=
by
  -- proof goes here
  sorry

end evaluate_expression_l86_86254


namespace max_b_integer_l86_86835

theorem max_b_integer (b : ℤ) : (∀ x : ℝ, x^2 + (b : ℝ) * x + 20 ≠ -10) → b ≤ 10 :=
by
  sorry

end max_b_integer_l86_86835


namespace jacqueline_guavas_l86_86524

theorem jacqueline_guavas 
  (G : ℕ) 
  (plums : ℕ := 16) 
  (apples : ℕ := 21) 
  (given : ℕ := 40) 
  (remaining : ℕ := 15) 
  (initial_fruits : ℕ := plums + G + apples)
  (total_fruits_after_given : ℕ := remaining + given) : 
  initial_fruits = total_fruits_after_given → G = 18 := 
by
  intro h
  sorry

end jacqueline_guavas_l86_86524


namespace largest_integer_m_l86_86986

theorem largest_integer_m (m n : ℕ) (h1 : ∀ n ≤ m, (2 * n + 1) / (3 * n + 8) < (Real.sqrt 5 - 1) / 2) 
(h2 : ∀ n ≤ m, (Real.sqrt 5 - 1) / 2 < (n + 7) / (2 * n + 1)) : 
  m = 27 :=
sorry

end largest_integer_m_l86_86986


namespace monthly_interest_rate_l86_86622

-- Define the principal amount (initial amount).
def principal : ℝ := 200

-- Define the final amount after 2 months (A).
def amount_after_two_months : ℝ := 222

-- Define the number of months (n).
def months : ℕ := 2

-- Define the monthly interest rate (r) we need to prove.
def interest_rate : ℝ := 0.053

-- Main statement to prove
theorem monthly_interest_rate :
  amount_after_two_months = principal * (1 + interest_rate)^months :=
sorry

end monthly_interest_rate_l86_86622


namespace units_digit_of_33_pow_33_mul_7_pow_7_l86_86395

theorem units_digit_of_33_pow_33_mul_7_pow_7 : (33 ^ (33 * (7 ^ 7))) % 10 = 7 := 
  sorry

end units_digit_of_33_pow_33_mul_7_pow_7_l86_86395


namespace area_of_enclosed_figure_l86_86716

theorem area_of_enclosed_figure:
  ∫ (x : ℝ) in (1/2)..2, x⁻¹ = 2 * Real.log 2 :=
by
  sorry

end area_of_enclosed_figure_l86_86716


namespace number_of_women_l86_86050

theorem number_of_women
    (n : ℕ) -- number of men
    (d_m : ℕ) -- number of dances each man had
    (d_w : ℕ) -- number of dances each woman had
    (total_men : n = 15) -- there are 15 men
    (each_man_dances : d_m = 4) -- each man danced with 4 women
    (each_woman_dances : d_w = 3) -- each woman danced with 3 men
    (total_dances : n * d_m = w * d_w): -- total dances are the same when counted from both sides
  w = 20 := sorry -- There should be exactly 20 women.


end number_of_women_l86_86050


namespace rope_segment_equation_l86_86167

theorem rope_segment_equation (x : ℝ) (h1 : 2 - x > 0) :
  x^2 = 2 * (2 - x) :=
by
  sorry

end rope_segment_equation_l86_86167


namespace eccentricity_of_hyperbola_l86_86568

noncomputable def hyperbola_eccentricity (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : b^2 = 2 * a^2) : ℝ :=
  (1 + b^2 / a^2) ^ (1/2)

theorem eccentricity_of_hyperbola (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : b^2 = 2 * a^2) :
  hyperbola_eccentricity a b h1 h2 h3 = Real.sqrt 3 := 
by
  unfold hyperbola_eccentricity
  rw [h3]
  simp
  sorry

end eccentricity_of_hyperbola_l86_86568


namespace purchased_both_books_l86_86541

theorem purchased_both_books: 
  ∀ (A B AB C : ℕ), A = 2 * B → AB = 2 * (B - AB) → C = 1000 → C = A - AB → AB = 500 := 
by
  intros A B AB C h1 h2 h3 h4
  sorry

end purchased_both_books_l86_86541


namespace total_length_of_scale_l86_86825

theorem total_length_of_scale (num_parts : ℕ) (length_per_part : ℕ) 
  (h1: num_parts = 4) (h2: length_per_part = 20) : 
  num_parts * length_per_part = 80 := by
  sorry

end total_length_of_scale_l86_86825


namespace evaluate_sum_of_squares_l86_86530

theorem evaluate_sum_of_squares 
  (x y : ℝ) 
  (h1 : 3 * x + 2 * y = 20) 
  (h2 : 4 * x + y = 25) : (x + y)^2 = 49 :=
  sorry

end evaluate_sum_of_squares_l86_86530


namespace sum_of_min_max_l86_86606

-- Define the necessary parameters and conditions
variables (n k : ℕ)
  (h_pos_nk : 0 < n ∧ 0 < k)
  (f : ℕ → ℕ)
  (h_toppings : ∀ t, (0 ≤ f t ∧ f t ≤ n) ∧ (f t + f (t + k) % (2 * k) = n))
  (m M : ℕ)
  (h_m : ∀ t, m ≤ f t)
  (h_M : ∀ t, f t ≤ M)
  (h_min_max : ∃ t_min t_max, m = f t_min ∧ M = f t_max)

-- The goal is to prove that the sum of m and M equals n
theorem sum_of_min_max (n k : ℕ) (h_pos_nk : 0 < n ∧ 0 < k)
  (f : ℕ → ℕ) (h_toppings : ∀ t, (0 ≤ f t ∧ f t ≤ n) ∧ (f t + f (t + k) % (2 * k) = n))
  (m M : ℕ) (h_m : ∀ t, m ≤ f t)
  (h_M : ∀ t, f t ≤ M)
  (h_min_max : ∃ t_min t_max, m = f t_min ∧ M = f t_max) :
  m + M = n := 
sorry

end sum_of_min_max_l86_86606


namespace remainder_of_7529_div_by_9_is_not_divisible_by_11_l86_86559

theorem remainder_of_7529_div_by_9 : 7529 % 9 = 5 := by
  sorry

theorem is_not_divisible_by_11 : ¬ (7529 % 11 = 0) := by
  sorry

end remainder_of_7529_div_by_9_is_not_divisible_by_11_l86_86559


namespace sum_of_common_ratios_of_sequences_l86_86058

def arithmetico_geometric_sequence (a b c : ℕ → ℝ) (r : ℝ) (d : ℝ) : Prop :=
∀ n, a (n + 1) = r * a n + d ∧ b (n + 1) = r * b n + d

theorem sum_of_common_ratios_of_sequences {m n : ℝ}
    {a1 a2 a3 b1 b2 b3 : ℝ}
    (p q : ℝ)
    (h_a1 : a1 = m)
    (h_a2 : a2 = m * p + 5)
    (h_a3 : a3 = m * p^2 + 5 * p + 5)
    (h_b1 : b1 = n)
    (h_b2 : b2 = n * q + 5)
    (h_b3 : b3 = n * q^2 + 5 * q + 5)
    (h_cond : a3 - b3 = 3 * (a2 - b2)) :
    p + q = 4 :=
by
  sorry

end sum_of_common_ratios_of_sequences_l86_86058


namespace nonneg_reals_ineq_l86_86216

theorem nonneg_reals_ineq 
  (a b x y : ℝ)
  (ha : 0 ≤ a) (hb : 0 ≤ b)
  (hx : 0 ≤ x) (hy : 0 ≤ y)
  (hab : a^5 + b^5 ≤ 1)
  (hxy : x^5 + y^5 ≤ 1) :
  a^2 * x^3 + b^2 * y^3 ≤ 1 :=
sorry

end nonneg_reals_ineq_l86_86216


namespace sum_of_consecutive_page_numbers_l86_86399

theorem sum_of_consecutive_page_numbers (n : ℕ) (h : n * (n + 1) = 20412) : n + (n + 1) = 287 :=
sorry

end sum_of_consecutive_page_numbers_l86_86399


namespace fraction_is_seventh_l86_86001

-- Definition of the condition on x being greater by a certain percentage
def x_greater := 1125.0000000000002 / 100

-- Definition of x in terms of the condition
def x := (4 / 7) * (1 + x_greater)

-- Definition of the fraction f
def f := 1 / x

-- Lean theorem statement to prove the fraction is 1/7
theorem fraction_is_seventh (x_greater: ℝ) : (1 / ((4 / 7) * (1 + x_greater))) = 1 / 7 :=
by
  sorry

end fraction_is_seventh_l86_86001


namespace range_of_function_l86_86241

noncomputable def function_range (x : ℝ) : ℝ :=
    (1 / 2) ^ (-x^2 + 2 * x)

theorem range_of_function : 
    (Set.range function_range) = Set.Ici (1 / 2) :=
by
    sorry

end range_of_function_l86_86241


namespace tetrahedron_BC_squared_l86_86713

theorem tetrahedron_BC_squared (AB AC BC R r : ℝ) 
  (h1 : AB = 1) 
  (h2 : AC = 1) 
  (h3 : 1 ≤ BC) 
  (h4 : R = 4 * r) 
  (concentric : AB = AC ∧ R > 0 ∧ r > 0) :
  BC^2 = 1 + Real.sqrt (7 / 15) := 
by 
sorry

end tetrahedron_BC_squared_l86_86713


namespace divisibility_by_7_l86_86781

theorem divisibility_by_7 (m a : ℤ) (h : 0 ≤ a ∧ a ≤ 9) (B : ℤ) (hB : B = m - 2 * a) (h7 : B % 7 = 0) : (10 * m + a) % 7 = 0 := 
sorry

end divisibility_by_7_l86_86781


namespace charles_pictures_after_work_l86_86491

variable (initial_papers : ℕ)
variable (draw_today : ℕ)
variable (draw_yesterday_morning : ℕ)
variable (papers_left : ℕ)

theorem charles_pictures_after_work :
    initial_papers = 20 →
    draw_today = 6 →
    draw_yesterday_morning = 6 →
    papers_left = 2 →
    initial_papers - (draw_today + draw_yesterday_morning + 6) = papers_left →
    6 = (initial_papers - draw_today - draw_yesterday_morning - papers_left) := 
by
  intros h1 h2 h3 h4 h5
  exact sorry

end charles_pictures_after_work_l86_86491


namespace hyperbola_asymptotes_n_l86_86867

theorem hyperbola_asymptotes_n {y x : ℝ} (n : ℝ) (H : ∀ x y, (y^2 / 16) - (x^2 / 9) = 1 → y = n * x ∨ y = -n * x) : n = 4/3 :=
  sorry

end hyperbola_asymptotes_n_l86_86867


namespace fraction_comparison_l86_86616

theorem fraction_comparison (a b c d : ℝ) 
  (h1 : (a / b) < (c / d))
  (h2 : b > d) (h3 : d > 0) :
  (a + c) / (b + d) < (1 / 2) * ((a / b) + (c / d)) :=
by
  sorry

end fraction_comparison_l86_86616


namespace height_difference_l86_86877

-- Definitions of the terms and conditions
variables {b h : ℝ} -- base and height of Triangle B
variables {b' h' : ℝ} -- base and height of Triangle A

-- Given conditions:
-- Triangle A's base is 10% greater than Triangle B's base
def base_relation (b' : ℝ) (b : ℝ) := b' = 1.10 * b

-- The area of Triangle A is 1% less than the area of Triangle B
def area_relation (b h b' h' : ℝ) := (1 / 2) * b' * h' = (1 / 2) * b * h - 0.01 * (1 / 2) * b * h

-- Proof statement
theorem height_difference (b h b' h' : ℝ) (H_base: base_relation b' b) (H_area: area_relation b h b' h') :
  h' = 0.9 * h := 
sorry

end height_difference_l86_86877


namespace smallest_A_is_144_l86_86552

noncomputable def smallest_A (B : ℕ) := B * 28 + 4

theorem smallest_A_is_144 :
  ∃ (B : ℕ), smallest_A B = 144 ∧ ∀ (B' : ℕ), B' * 28 + 4 < 144 → false :=
by
  sorry

end smallest_A_is_144_l86_86552


namespace one_minus_repeating_six_l86_86700

noncomputable def repeating_six : Real := 2 / 3

theorem one_minus_repeating_six : 1 - repeating_six = 1 / 3 :=
by
  sorry

end one_minus_repeating_six_l86_86700


namespace salary_increase_correct_l86_86987

noncomputable def old_average_salary : ℕ := 1500
noncomputable def number_of_employees : ℕ := 24
noncomputable def manager_salary : ℕ := 11500
noncomputable def new_total_salary := (number_of_employees * old_average_salary) + manager_salary
noncomputable def new_number_of_people := number_of_employees + 1
noncomputable def new_average_salary := new_total_salary / new_number_of_people
noncomputable def salary_increase := new_average_salary - old_average_salary

theorem salary_increase_correct : salary_increase = 400 := by
sorry

end salary_increase_correct_l86_86987


namespace range_of_a_l86_86389

noncomputable def f (a x : ℝ) : ℝ := (Real.log (x^2 - a * x + 5)) / (Real.log a)

theorem range_of_a (a : ℝ) (x₁ x₂ : ℝ) 
  (ha0 : 0 < a) (ha1 : a ≠ 1) 
  (hx₁x₂ : x₁ < x₂) (hx₂ : x₂ ≤ a / 2) 
  (hf : (f a x₂ - f a x₁ < 0)) : 
  1 < a ∧ a < 2 * Real.sqrt 5 := 
sorry

end range_of_a_l86_86389


namespace length_of_de_l86_86515

theorem length_of_de
  {a b c d e : ℝ} 
  (h1 : b - a = 5) 
  (h2 : c - a = 11) 
  (h3 : e - a = 22) 
  (h4 : c - b = 2 * (d - c)) :
  e - d = 8 :=
by 
  sorry

end length_of_de_l86_86515


namespace max_frac_sum_l86_86394

theorem max_frac_sum {n : ℕ} (h_n : n > 1) :
  ∀ (a b c d : ℕ), (a + c ≤ n) ∧ (b > 0) ∧ (d > 0) ∧
  (a * d + b * c < b * d) → 
  ↑a / ↑b + ↑c / ↑d ≤ (1 - 1 / ( ⌊(2*n : ℝ)/3 + 1/6⌋₊ + 1) * ( ⌊(2*n : ℝ)/3 + 1/6⌋₊ * (n - ⌊(2*n : ℝ)/3 + 1/6⌋₊) + 1)) :=
by sorry

end max_frac_sum_l86_86394


namespace find_m_minus_n_l86_86380

-- Define line equations, parallelism, and perpendicularity
def line1 (x y : ℝ) : Prop := 3 * x - 6 * y + 1 = 0
def line2 (x y : ℝ) (m : ℝ) : Prop := x - m * y + 2 = 0
def line3 (x y : ℝ) (n : ℝ) : Prop := n * x + y + 3 = 0

def parallel (m1 m2 : ℝ) : Prop := m1 = m2
def perpendicular (m1 m2 : ℝ) : Prop := m1 * m2 = -1

theorem find_m_minus_n (m n : ℝ) (h_parallel : parallel (1/2) (1/m)) (h_perpendicular: perpendicular (1/2) (-1/n)) : m - n = 0 :=
sorry

end find_m_minus_n_l86_86380


namespace age_ratio_l86_86571

theorem age_ratio (A B C : ℕ) (h1 : A = B + 2) (h2 : A + B + C = 27) (h3 : B = 10) : B / C = 2 :=
by
  sorry

end age_ratio_l86_86571


namespace ratio_of_diagonals_l86_86822

theorem ratio_of_diagonals (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : (4 * b) / (4 * a) = 11) : (b * Real.sqrt 2) / (a * Real.sqrt 2) = 11 := 
by 
  sorry

end ratio_of_diagonals_l86_86822


namespace calculate_expression_l86_86222

theorem calculate_expression : -Real.sqrt 9 - 4 * (-2) + 2 * Real.cos (Real.pi / 3) = 6 :=
by
  sorry

end calculate_expression_l86_86222


namespace contradiction_assumption_l86_86238

theorem contradiction_assumption (x y : ℝ) (h1 : x > y) : ¬ (x^3 ≤ y^3) := 
by
  sorry

end contradiction_assumption_l86_86238


namespace find_x_value_l86_86908

noncomputable def solve_some_number (x : ℝ) : Prop :=
  let expr := (x - (8 / 7) * 5 + 10)
  expr = 13.285714285714286

theorem find_x_value : ∃ x : ℝ, solve_some_number x ∧ x = 9 := by
  sorry

end find_x_value_l86_86908


namespace hyperbola_asymptote_distance_l86_86313

section
open Function Real

variables (O P : ℝ × ℝ) (C : ℝ × ℝ → Prop) (M : ℝ × ℝ)
          (dist_asymptote : ℝ)

-- Conditions
def is_origin (O : ℝ × ℝ) : Prop := O = (0, 0)
def on_hyperbola (P : ℝ × ℝ) : Prop := P.1 ^ 2 / 9 - P.2 ^ 2 / 16 = 1
def unit_circle (M : ℝ × ℝ) : Prop := sqrt (M.1 ^ 2 + M.2 ^ 2) = 1
def orthogonal (O M P : ℝ × ℝ) : Prop := O.1 * P.1 + O.2 * P.2 = 0
def min_PM (dist : ℝ) : Prop := dist = 1 -- The minimum distance when |PM| is minimized

-- Proof problem
theorem hyperbola_asymptote_distance :
  is_origin O → 
  on_hyperbola P → 
  unit_circle M → 
  orthogonal O M P → 
  min_PM (sqrt ((P.1 - M.1) ^ 2 + (P.2 - M.2) ^ 2)) → 
  dist_asymptote = 12 / 5 :=
sorry
end

end hyperbola_asymptote_distance_l86_86313


namespace cat_food_sufficiency_l86_86332

theorem cat_food_sufficiency (L S : ℝ) (h : L + 4 * S = 14) : L + 3 * S ≥ 11 :=
sorry

end cat_food_sufficiency_l86_86332


namespace call_duration_l86_86287

def initial_credit : ℝ := 30
def cost_per_minute : ℝ := 0.16
def remaining_credit : ℝ := 26.48

theorem call_duration :
  (initial_credit - remaining_credit) / cost_per_minute = 22 := 
sorry

end call_duration_l86_86287


namespace ratio_e_to_f_l86_86679

theorem ratio_e_to_f {a b c d e f : ℝ}
  (h1 : a / b = 1 / 3)
  (h2 : b / c = 2)
  (h3 : c / d = 1 / 2)
  (h4 : d / e = 3)
  (h5 : a * b * c / (d * e * f) = 0.75) :
  e / f = 0.5 :=
sorry

end ratio_e_to_f_l86_86679


namespace Brenda_bakes_cakes_l86_86984

theorem Brenda_bakes_cakes 
  (cakes_per_day : ℕ)
  (days : ℕ)
  (sell_fraction : ℚ)
  (total_cakes_baked : ℕ := cakes_per_day * days)
  (cakes_left : ℚ := total_cakes_baked * sell_fraction)
  (h1 : cakes_per_day = 20)
  (h2 : days = 9)
  (h3 : sell_fraction = 1 / 2) :
  cakes_left = 90 := 
by 
  -- Proof to be filled in later
  sorry

end Brenda_bakes_cakes_l86_86984


namespace largest_4_digit_divisible_by_88_and_prime_gt_100_l86_86010

noncomputable def is_4_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999

noncomputable def is_divisible_by (n d : ℕ) : Prop :=
  d ∣ n

noncomputable def is_prime (p : ℕ) : Prop :=
  Nat.Prime p

noncomputable def lcm (a b : ℕ) : ℕ :=
  Nat.lcm a b

theorem largest_4_digit_divisible_by_88_and_prime_gt_100 (p : ℕ) (hp : is_prime p) (h1 : 100 < p):
  ∃ n, is_4_digit n ∧ is_divisible_by n 88 ∧ is_divisible_by n p ∧
       (∀ m, is_4_digit m ∧ is_divisible_by m 88 ∧ is_divisible_by m p → m ≤ n) :=
sorry

end largest_4_digit_divisible_by_88_and_prime_gt_100_l86_86010


namespace farmer_animals_l86_86225

theorem farmer_animals : 
  ∃ g s : ℕ, 
    35 * g + 40 * s = 2000 ∧ 
    g = 2 * s ∧ 
    (0 < g ∧ 0 < s) ∧ 
    g = 36 ∧ s = 18 := 
by 
  sorry

end farmer_animals_l86_86225


namespace intersection_M_N_l86_86252

def M : Set ℝ := {x : ℝ | -4 < x ∧ x < 4}
def N : Set ℝ := {x : ℝ | x ≥ -1 / 3}

theorem intersection_M_N :
  M ∩ N = {x : ℝ | -1 / 3 ≤ x ∧ x < 4} :=
sorry

end intersection_M_N_l86_86252


namespace anna_candy_division_l86_86948

theorem anna_candy_division : 
  ∀ (total_candies friends : ℕ), 
  total_candies = 30 → 
  friends = 4 → 
  ∃ (candies_to_remove : ℕ), 
  candies_to_remove = 2 ∧ 
  (total_candies - candies_to_remove) % friends = 0 := 
by
  sorry

end anna_candy_division_l86_86948


namespace gcd_50403_40302_l86_86277

theorem gcd_50403_40302 : Nat.gcd 50403 40302 = 1 :=
by
  sorry

end gcd_50403_40302_l86_86277


namespace sarah_score_l86_86257

theorem sarah_score
  (hunter_score : ℕ)
  (john_score : ℕ)
  (grant_score : ℕ)
  (sarah_score : ℕ)
  (h1 : hunter_score = 45)
  (h2 : john_score = 2 * hunter_score)
  (h3 : grant_score = john_score + 10)
  (h4 : sarah_score = grant_score - 5) :
  sarah_score = 95 :=
by
  sorry

end sarah_score_l86_86257


namespace percentage_solution_l86_86383

variable (x y : ℝ)
variable (P : ℝ)

-- Conditions
axiom cond1 : 0.20 * (x - y) = (P / 100) * (x + y)
axiom cond2 : y = (1 / 7) * x

-- Theorem statement
theorem percentage_solution : P = 15 :=
by 
  -- Sorry means skipping the proof
  sorry

end percentage_solution_l86_86383


namespace parallelogram_area_l86_86207

theorem parallelogram_area (base height : ℝ) (h_base : base = 20) (h_height : height = 16) :
  base * height = 320 :=
by
  sorry

end parallelogram_area_l86_86207


namespace camila_bikes_more_l86_86028

-- Definitions based on conditions
def camila_speed : ℝ := 15
def daniel_speed_initial : ℝ := 15
def daniel_speed_after_3hours : ℝ := 10
def biking_time : ℝ := 6
def time_before_decrease : ℝ := 3
def time_after_decrease : ℝ := biking_time - time_before_decrease

def distance_camila := camila_speed * biking_time
def distance_daniel := (daniel_speed_initial * time_before_decrease) + (daniel_speed_after_3hours * time_after_decrease)

-- The statement to prove: Camila has biked 15 more miles than Daniel
theorem camila_bikes_more : distance_camila - distance_daniel = 15 := 
by
  sorry

end camila_bikes_more_l86_86028


namespace harmonic_mean_pairs_count_l86_86807

open Nat

theorem harmonic_mean_pairs_count :
  ∃! n : ℕ, (∀ x y : ℕ, x < y ∧ x > 0 ∧ y > 0 ∧ (2 * x * y) / (x + y) = 4^15 → n = 29) :=
sorry

end harmonic_mean_pairs_count_l86_86807


namespace measure_of_angle_A_l86_86343

noncomputable def angle_A (angle_B : ℝ) := 3 * angle_B - 40

theorem measure_of_angle_A (x : ℝ) (angle_A_parallel_B : true) (h : ∃ k : ℝ, (k = x ∧ (angle_A x = x ∨ angle_A x + x = 180))) :
  angle_A x = 20 ∨ angle_A x = 125 :=
by
  sorry

end measure_of_angle_A_l86_86343


namespace upper_bound_exists_l86_86234

theorem upper_bound_exists (U : ℤ) :
  (∀ n : ℤ, 1 < 4 * n + 7 ∧ 4 * n + 7 < U) →
  (∃ n_min n_max : ℤ, n_max = n_min + 29 ∧ 4 * n_max + 7 < U ∧ 4 * n_min + 7 > 1) →
  (U = 120) :=
by
  intros h1 h2
  sorry

end upper_bound_exists_l86_86234


namespace find_k_when_root_is_zero_l86_86189

-- Define the quadratic equation and what it implies
theorem find_k_when_root_is_zero (k : ℝ) (h : (k-1) * 0^2 + 6 * 0 + k^2 - k = 0) :
  k = 0 :=
by
  -- The proof steps would go here, but we're skipping it as instructed
  sorry

end find_k_when_root_is_zero_l86_86189


namespace boy_age_proof_l86_86773

theorem boy_age_proof (P X : ℕ) (hP : P = 16) (hcond : P - X = (P + 4) / 2) : X = 6 :=
by
  sorry

end boy_age_proof_l86_86773


namespace ladder_rung_length_l86_86839

noncomputable def ladder_problem : Prop :=
  let total_height_ft := 50
  let spacing_in := 6
  let wood_ft := 150
  let feet_to_inches(ft : ℕ) : ℕ := ft * 12
  let total_height_in := feet_to_inches total_height_ft
  let wood_in := feet_to_inches wood_ft
  let number_of_rungs := total_height_in / spacing_in
  let length_of_each_rung := wood_in / number_of_rungs
  length_of_each_rung = 18

theorem ladder_rung_length : ladder_problem := sorry

end ladder_rung_length_l86_86839


namespace sequence_periodic_l86_86514

def sequence (a : ℕ → ℝ) : Prop :=
  a 1 = -2 ∧ ∀ n, a (n + 1) = (1 + a n) / (1 - a n)

theorem sequence_periodic :
  ∃ a : ℕ → ℝ, sequence a ∧ a 2016 = 3 :=
by
  sorry

end sequence_periodic_l86_86514


namespace isosceles_triangle_of_cosine_condition_l86_86324

theorem isosceles_triangle_of_cosine_condition
  (A B C : ℝ)
  (h : 2 * Real.cos A * Real.cos B = 1 - Real.cos C) :
  A = B ∨ A = π - B :=
  sorry

end isosceles_triangle_of_cosine_condition_l86_86324


namespace greatest_value_of_n_l86_86961

theorem greatest_value_of_n (n : ℤ) (h : 101 * n ^ 2 ≤ 3600) : n ≤ 5 :=
by
  sorry

end greatest_value_of_n_l86_86961


namespace sets_equal_l86_86981

-- Defining the sets and proving their equality
theorem sets_equal : { x : ℝ | x^2 + 1 = 0 } = (∅ : Set ℝ) :=
  sorry

end sets_equal_l86_86981


namespace find_number_l86_86675

theorem find_number (x : ℝ) (h : (3.242 * 16) / x = 0.051871999999999995) : x = 1000 :=
by
  sorry

end find_number_l86_86675


namespace resulting_figure_perimeter_l86_86485

def original_square_side : ℕ := 100

def original_square_area : ℕ := original_square_side * original_square_side

def rect1_side1 : ℕ := original_square_side
def rect1_side2 : ℕ := original_square_side / 2

def rect2_side1 : ℕ := original_square_side
def rect2_side2 : ℕ := original_square_side / 2

def new_figure_perimeter : ℕ :=
  3 * original_square_side + 4 * (original_square_side / 2)

theorem resulting_figure_perimeter :
  new_figure_perimeter = 500 :=
by {
    sorry
}

end resulting_figure_perimeter_l86_86485


namespace picked_tomatoes_eq_53_l86_86799

-- Definitions based on the conditions
def initial_tomatoes : ℕ := 177
def initial_potatoes : ℕ := 12
def items_left : ℕ := 136

-- Define what we need to prove
theorem picked_tomatoes_eq_53 : initial_tomatoes + initial_potatoes - items_left = 53 :=
by sorry

end picked_tomatoes_eq_53_l86_86799


namespace total_pokemon_cards_l86_86887

-- Definitions based on conditions
def jenny_cards : ℕ := 6
def orlando_cards : ℕ := jenny_cards + 2
def richard_cards : ℕ := 3 * orlando_cards

-- The theorem stating the total number of cards
theorem total_pokemon_cards : jenny_cards + orlando_cards + richard_cards = 38 :=
by
  sorry

end total_pokemon_cards_l86_86887


namespace combined_class_average_score_l86_86054

theorem combined_class_average_score
  (avg_A : ℕ := 65) (avg_B : ℕ := 90) (avg_C : ℕ := 77)
  (ratio_A : ℕ := 4) (ratio_B : ℕ := 6) (ratio_C : ℕ := 5) :
  ((avg_A * ratio_A + avg_B * ratio_B + avg_C * ratio_C) / (ratio_A + ratio_B + ratio_C) = 79) :=
by 
  sorry

end combined_class_average_score_l86_86054


namespace distinct_convex_quadrilaterals_l86_86924

open Nat

noncomputable def combinations (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem distinct_convex_quadrilaterals (n : ℕ) (h : n > 4) 
  (no_three_collinear : ℕ → Prop) :
  ∃ k, k ≥ combinations n 5 / (n - 4) :=
by
  sorry

end distinct_convex_quadrilaterals_l86_86924


namespace point_Q_in_third_quadrant_l86_86897

theorem point_Q_in_third_quadrant (m : ℝ) :
  (2 * m + 4 = 0 → (m - 3, m).fst < 0 ∧ (m - 3, m).snd < 0) :=
by
  sorry

end point_Q_in_third_quadrant_l86_86897


namespace tomatoes_left_l86_86678

theorem tomatoes_left (initial_tomatoes : ℕ) (birds : ℕ) (fraction : ℕ) (E1 : initial_tomatoes = 21) 
  (E2 : birds = 2) (E3 : fraction = 3) : 
  initial_tomatoes - initial_tomatoes / fraction = 14 :=
by 
  sorry

end tomatoes_left_l86_86678


namespace largest_non_sum_217_l86_86665

def is_prime (n : ℕ) : Prop := sorry -- This would be some definition of primality

noncomputable def largest_non_sum_of_composite (n : ℕ) : Prop :=
  ∀ (a b : ℕ), 0 ≤ b ∧ b < 30 ∧ (∀ i : ℕ, i < a → is_prime (b + 30 * i)) →
  n ≤ 30 * a + b

theorem largest_non_sum_217 : ∃ n, largest_non_sum_of_composite n ∧ n = 217 := sorry

end largest_non_sum_217_l86_86665


namespace cube_properties_l86_86554

theorem cube_properties (s y : ℝ) (h1 : s^3 = 8 * y) (h2 : 6 * s^2 = 6 * y) : y = 64 := by
  sorry

end cube_properties_l86_86554


namespace TV_height_l86_86516

theorem TV_height (Area Width Height : ℝ) (h_area : Area = 21) (h_width : Width = 3) (h_area_def : Area = Width * Height) : Height = 7 := 
by
  sorry

end TV_height_l86_86516


namespace giselle_initial_doves_l86_86687

theorem giselle_initial_doves (F : ℕ) (h1 : ∀ F, F > 0) (h2 : 3 * F * 3 / 4 + F = 65) : F = 20 :=
sorry

end giselle_initial_doves_l86_86687


namespace correlation_coefficient_l86_86512

theorem correlation_coefficient (variation_explained_by_height : ℝ)
    (variation_explained_by_errors : ℝ)
    (total_variation : variation_explained_by_height + variation_explained_by_errors = 1)
    (percentage_explained_by_height : variation_explained_by_height = 0.71) :
  variation_explained_by_height = 0.71 := 
by
  sorry

end correlation_coefficient_l86_86512


namespace minimum_value_of_fraction_l86_86520

theorem minimum_value_of_fraction (x : ℝ) (hx : x > 10) : ∃ m, m = 30 ∧ ∀ y > 10, (y * y) / (y - 10) ≥ m :=
by 
  sorry

end minimum_value_of_fraction_l86_86520


namespace range_m_if_neg_p_implies_neg_q_range_x_if_m_is_5_and_p_or_q_true_p_and_q_false_l86_86210

-- Question 1
def prop_p (x : ℝ) : Prop := (x + 1) * (x - 5) ≤ 0
def prop_q (x m : ℝ) : Prop := 1 - m ≤ x + 1 ∧ x + 1 < 1 + m ∧ m > 0
def neg_p (x : ℝ) : Prop := ¬ prop_p x
def neg_q (x m : ℝ) : Prop := ¬ prop_q x m

theorem range_m_if_neg_p_implies_neg_q : 
  (∀ x, neg_p x → neg_q x m) → 0 < m ∧ m ≤ 1 :=
by
  sorry

-- Question 2
theorem range_x_if_m_is_5_and_p_or_q_true_p_and_q_false : 
  (∀ x, (prop_p x ∨ prop_q x 5) ∧ ¬ (prop_p x ∧ prop_q x 5)) → 
  ∀ x, (x = 5 ∨ (-5 ≤ x ∧ x < -1)) :=
by
  sorry

end range_m_if_neg_p_implies_neg_q_range_x_if_m_is_5_and_p_or_q_true_p_and_q_false_l86_86210


namespace range_of_a_l86_86430

theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, x^2 + a * x + 4 < 0) ↔ (a < -4 ∨ a > 4) :=
by 
sorry

end range_of_a_l86_86430


namespace min_value_f_l86_86600

theorem min_value_f
  (a b c : ℝ)
  (α β γ : ℤ)
  (hα : α = 1 ∨ α = -1)
  (hβ : β = 1 ∨ β = -1)
  (hγ : γ = 1 ∨ γ = -1)
  (h : a * α + b * β + c * γ = 0) :
  (∃ f_min : ℝ, f_min = ( ((a ^ 3 + b ^ 3 + c ^ 3) / (a * b * c)) ^ 2) ∧ f_min = 9) :=
sorry

end min_value_f_l86_86600


namespace extremum_condition_l86_86621

theorem extremum_condition (a b : ℝ) (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = x^3 + a * x^2 + b * x + a^2)
  (h2 : f 1 = 10)
  (h3 : deriv f 1 = 0) :
  a + b = -7 :=
sorry

end extremum_condition_l86_86621


namespace eval_expression_l86_86339

theorem eval_expression (x : ℝ) (h₀ : x = 3) :
  let initial_expr : ℝ := (2 * x + 2) / (x - 2)
  let replaced_expr : ℝ := (2 * initial_expr + 2) / (initial_expr - 2)
  replaced_expr = 8 :=
by
  sorry

end eval_expression_l86_86339


namespace genuine_items_count_l86_86778

def total_purses : ℕ := 26
def total_handbags : ℕ := 24
def fake_purses : ℕ := total_purses / 2
def fake_handbags : ℕ := total_handbags / 4
def genuine_purses : ℕ := total_purses - fake_purses
def genuine_handbags : ℕ := total_handbags - fake_handbags

theorem genuine_items_count : genuine_purses + genuine_handbags = 31 := by
  sorry

end genuine_items_count_l86_86778


namespace positive_difference_eq_six_l86_86347

theorem positive_difference_eq_six (x y : ℝ) (h1 : x + y = 8) (h2 : x ^ 2 - y ^ 2 = 48) : |x - y| = 6 := by
  sorry

end positive_difference_eq_six_l86_86347


namespace units_digit_sum_of_factorials_l86_86983

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def ones_digit (n : ℕ) : ℕ :=
  n % 10

theorem units_digit_sum_of_factorials :
  ones_digit (factorial 1 + factorial 2 + factorial 3 + factorial 4 + factorial 5 +
              factorial 6 + factorial 7 + factorial 8 + factorial 9 + factorial 10) = 3 := 
sorry

end units_digit_sum_of_factorials_l86_86983


namespace cauliflower_production_proof_l86_86147

theorem cauliflower_production_proof (x y : ℕ) 
  (h1 : y^2 - x^2 = 401)
  (hx : x > 0)
  (hy : y > 0) :
  y^2 = 40401 :=
by
  sorry

end cauliflower_production_proof_l86_86147


namespace inequality_selection_l86_86928

theorem inequality_selection (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a ≠ b) :
  4 * (a^3 + b^3) > (a + b)^3 := 
by sorry

end inequality_selection_l86_86928


namespace total_games_in_season_l86_86468

theorem total_games_in_season (teams: ℕ) (division_teams: ℕ) (intra_division_games: ℕ) (inter_division_games: ℕ) (total_games: ℕ) : 
  teams = 18 → division_teams = 9 → intra_division_games = 3 → inter_division_games = 2 → total_games = 378 :=
by
  sorry

end total_games_in_season_l86_86468


namespace simplify_expression_l86_86709

variables (x y : ℝ)

theorem simplify_expression :
  (3 * x)^4 + (4 * x) * (x^3) + (5 * y)^2 = 85 * x^4 + 25 * y^2 :=
by
  sorry

end simplify_expression_l86_86709


namespace cost_price_approx_l86_86387

noncomputable def cost_price (selling_price : ℝ) (profit_percent : ℝ) : ℝ :=
  selling_price / (1 + profit_percent / 100)

theorem cost_price_approx :
  ∀ (selling_price profit_percent : ℝ),
  selling_price = 2552.36 →
  profit_percent = 6 →
  abs (cost_price selling_price profit_percent - 2407.70) < 0.01 :=
by
  intros selling_price profit_percent h1 h2
  sorry

end cost_price_approx_l86_86387


namespace exhibition_admission_fees_ratio_l86_86030

theorem exhibition_admission_fees_ratio
  (a c : ℕ)
  (h1 : 30 * a + 15 * c = 2925)
  (h2 : a % 5 = 0)
  (h3 : c % 5 = 0) :
  (a / 5 = c / 5) :=
by
  sorry

end exhibition_admission_fees_ratio_l86_86030


namespace bananas_left_l86_86328

theorem bananas_left (original_bananas : ℕ) (bananas_eaten : ℕ) 
  (h1 : original_bananas = 12) (h2 : bananas_eaten = 4) : 
  original_bananas - bananas_eaten = 8 := 
by
  sorry

end bananas_left_l86_86328


namespace percent_gain_on_transaction_l86_86551

theorem percent_gain_on_transaction
  (c : ℝ) -- cost per sheep
  (price_750_sold : ℝ := 800 * c) -- price at which 750 sheep were sold in total
  (price_per_sheep_750 : ℝ := price_750_sold / 750)
  (price_per_sheep_50 : ℝ := 1.1 * price_per_sheep_750)
  (revenue_750 : ℝ := price_per_sheep_750 * 750)
  (revenue_50 : ℝ := price_per_sheep_50 * 50)
  (total_revenue : ℝ := revenue_750 + revenue_50)
  (total_cost : ℝ := 800 * c)
  (profit : ℝ := total_revenue - total_cost)
  (percent_gain : ℝ := (profit / total_cost) * 100) :
  percent_gain = 14 :=
sorry

end percent_gain_on_transaction_l86_86551


namespace final_number_independent_of_order_l86_86092

theorem final_number_independent_of_order 
  (p q r : ℕ) : 
  ∃ k : ℕ, 
    (p % 2 ≠ 0 ∨ q % 2 ≠ 0 ∨ r % 2 ≠ 0) ∧ 
    (∀ (p' q' r' : ℕ), 
       p' + q' + r' = p + q + r → 
       p' % 2 = p % 2 ∧ q' % 2 = q % 2 ∧ r' % 2 = r % 2 → 
       (p' = 1 ∧ q' = 0 ∧ r' = 0 ∨ 
        p' = 0 ∧ q' = 1 ∧ r' = 0 ∨ 
        p' = 0 ∧ q' = 0 ∧ r' = 1) → 
       k = p ∨ k = q ∨ k = r) := 
sorry

end final_number_independent_of_order_l86_86092


namespace eval_expression_l86_86692

theorem eval_expression : 
  (20-19 + 18-17 + 16-15 + 14-13 + 12-11 + 10-9 + 8-7 + 6-5 + 4-3 + 2-1) / 
  (1-2 + 3-4 + 5-6 + 7-8 + 9-10 + 11-12 + 13-14 + 15-16 + 17-18 + 19-20) = -1 := by
  sorry

end eval_expression_l86_86692


namespace num_integers_satisfy_inequality_l86_86537

theorem num_integers_satisfy_inequality : ∃ (s : Finset ℤ), (∀ x ∈ s, |7 * x - 5| ≤ 15) ∧ s.card = 5 :=
by
  sorry

end num_integers_satisfy_inequality_l86_86537


namespace find_number_l86_86104

theorem find_number (x : ℝ) (h : 0.45 * x = 162) : x = 360 :=
sorry

end find_number_l86_86104


namespace brick_height_l86_86916

theorem brick_height (length width : ℕ) (num_bricks : ℕ) (wall_length wall_width wall_height : ℕ) (h : ℕ) :
  length = 20 ∧ width = 10 ∧ num_bricks = 25000 ∧ wall_length = 2500 ∧ wall_width = 200 ∧ wall_height = 75 ∧
  ( 20 * 10 * h = (wall_length * wall_width * wall_height) / 25000 ) -> 
  h = 75 :=
by
  sorry

end brick_height_l86_86916


namespace find_x_when_y_neg_five_l86_86662

-- Definitions based on the conditions provided
variable (x y : ℝ)
def inversely_proportional (x y : ℝ) := ∃ (k : ℝ), x * y = k

-- Proving the main result
theorem find_x_when_y_neg_five (h_prop : inversely_proportional x y) (hx4 : x = 4) (hy2 : y = 2) :
    (y = -5) → x = - 8 / 5 := by
  sorry

end find_x_when_y_neg_five_l86_86662


namespace sin_15_deg_eq_l86_86718

theorem sin_15_deg_eq : 
  Real.sin (15 * Real.pi / 180) = (Real.sqrt 6 - Real.sqrt 2) / 4 := 
by
  -- conditions
  have h1 : Real.sin (45 * Real.pi / 180) = Real.sqrt 2 / 2 := by sorry
  have h2 : Real.cos (45 * Real.pi / 180) = Real.sqrt 2 / 2 := by sorry
  have h3 : Real.sin (30 * Real.pi / 180) = 1 / 2 := by sorry
  have h4 : Real.cos (30 * Real.pi / 180) = Real.sqrt 3 / 2 := by sorry
  
  -- proof
  sorry

end sin_15_deg_eq_l86_86718


namespace sum_of_transformed_numbers_l86_86495

theorem sum_of_transformed_numbers (a b S : ℝ) (h : a + b = S) :
    3 * (a + 5) + 3 * (b + 5) = 3 * S + 30 :=
by
  sorry

end sum_of_transformed_numbers_l86_86495


namespace union_A_B_interval_l86_86811

def setA (x : ℝ) : Prop := x ≥ -1
def setB (y : ℝ) : Prop := y ≥ 1

theorem union_A_B_interval :
  {x | setA x} ∪ {y | setB y} = {z : ℝ | z ≥ -1} :=
by
  sorry

end union_A_B_interval_l86_86811


namespace max_b_plus_c_triangle_l86_86756

theorem max_b_plus_c_triangle (a b c : ℝ) (A : ℝ) 
  (h₁ : a = 4) (h₂ : A = Real.pi / 3) (h₃ : a^2 = b^2 + c^2 - 2 * b * c * Real.cos A) :
  b + c ≤ 8 :=
by
  -- sorry is added to skip the proof for now.
  sorry

end max_b_plus_c_triangle_l86_86756


namespace line_equation_l86_86372

open Real

theorem line_equation (x y : Real) : 
  (3 * x + 2 * y - 1 = 0) ↔ (y = (-(3 / 2)) * x + 2.5) :=
by
  sorry

end line_equation_l86_86372


namespace find_m_l86_86885

noncomputable def f (x a : ℝ) : ℝ := x - a

theorem find_m (a m : ℝ) :
  (∀ x, 0 ≤ x ∧ x ≤ 4 → f x a ≤ 2) →
  (∃ x, -2 ≤ x ∧ x ≤ 4 ∧ -1 - f (x + 1) a ≤ m) :=
sorry

end find_m_l86_86885


namespace least_positive_angle_l86_86024

theorem least_positive_angle (θ : ℝ) (h : Real.cos (10 * Real.pi / 180) = Real.sin (15 * Real.pi / 180) + Real.sin θ) :
  θ = 32.5 * Real.pi / 180 := 
sorry

end least_positive_angle_l86_86024


namespace number_of_three_digit_multiples_of_6_l86_86445

theorem number_of_three_digit_multiples_of_6 : 
  let lower_bound := 100
  let upper_bound := 999
  let multiple := 6
  let smallest_n := Nat.ceil (100 / multiple)
  let largest_n := Nat.floor (999 / multiple)
  let count_multiples := largest_n - smallest_n + 1
  count_multiples = 150 := by
  sorry

end number_of_three_digit_multiples_of_6_l86_86445


namespace no_int_coeffs_l86_86562

def P (a b c d : ℤ) (x : ℤ) : ℤ := a * x^3 + b * x^2 + c * x + d

theorem no_int_coeffs (a b c d : ℤ) : 
  ¬ (P a b c d 19 = 1 ∧ P a b c d 62 = 2) :=
by sorry

end no_int_coeffs_l86_86562


namespace number_of_people_in_group_l86_86209

theorem number_of_people_in_group :
  ∃ (N : ℕ), (∀ (avg_weight : ℝ), 
  ∃ (new_person_weight : ℝ) (replaced_person_weight : ℝ),
  new_person_weight = 85 ∧ replaced_person_weight = 65 ∧
  avg_weight + 2.5 = ((N * avg_weight + (new_person_weight - replaced_person_weight)) / N) ∧ 
  N = 8) :=
by
  sorry

end number_of_people_in_group_l86_86209


namespace value_of_y_l86_86805

theorem value_of_y (x y : ℤ) (h1 : x^2 = y - 8) (h2 : x = -7) : y = 57 :=
sorry

end value_of_y_l86_86805


namespace stratified_sampling_third_grade_l86_86964

theorem stratified_sampling_third_grade (total_students : ℕ) (first_grade_students : ℕ)
  (second_grade_students : ℕ) (third_grade_students : ℕ) (sample_size : ℕ)
  (h_total : total_students = 270000) (h_first : first_grade_students = 99000)
  (h_second : second_grade_students = 90000) (h_third : third_grade_students = 81000)
  (h_sample : sample_size = 3000) :
  third_grade_students * (sample_size / total_students) = 900 := 
by {
  sorry
}

end stratified_sampling_third_grade_l86_86964


namespace ratio_of_teaspoons_to_knives_is_2_to_1_l86_86989

-- Define initial conditions based on the problem
def initial_knives : ℕ := 24
def initial_teaspoons (T : ℕ) : Prop := 
  initial_knives + T + (1 / 3 : ℚ) * initial_knives + (2 / 3 : ℚ) * T = 112

-- Define the ratio to be proved
def ratio_teaspoons_to_knives (T : ℕ) : Prop :=
  initial_teaspoons T ∧ T = 48 ∧ 48 / initial_knives = 2

theorem ratio_of_teaspoons_to_knives_is_2_to_1 : ∃ T, ratio_teaspoons_to_knives T :=
by
  -- Proof would follow here
  sorry

end ratio_of_teaspoons_to_knives_is_2_to_1_l86_86989


namespace dart_lands_in_center_square_l86_86256

theorem dart_lands_in_center_square (s : ℝ) (h : 0 < s) :
  let center_square_area := (s / 2) ^ 2
  let triangle_area := 1 / 2 * (s / 2) ^ 2
  let total_triangle_area := 4 * triangle_area
  let total_board_area := center_square_area + total_triangle_area
  let probability := center_square_area / total_board_area
  probability = 1 / 3 :=
by
  sorry

end dart_lands_in_center_square_l86_86256


namespace diff_PA_AQ_const_l86_86271

open Real

def point := (ℝ × ℝ)

noncomputable def distance (p1 p2 : point) : ℝ :=
  sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem diff_PA_AQ_const (a : ℝ) (h : 0 ≤ a ∧ a ≤ 1) :
  let P := (0, -sqrt 2)
  let Q := (0, sqrt 2)
  let A := (a, sqrt (a^2 + 1))
  distance P A - distance A Q = 2 := 
sorry

end diff_PA_AQ_const_l86_86271


namespace book_cost_proof_l86_86504

variable (C1 C2 : ℝ)

theorem book_cost_proof (h1 : C1 + C2 = 460)
                        (h2 : C1 * 0.85 = C2 * 1.19) :
    C1 = 268.53 := by
  sorry

end book_cost_proof_l86_86504


namespace necessary_but_not_sufficient_l86_86457

variable (a b : ℝ)

theorem necessary_but_not_sufficient : 
  ¬ (a ≠ 1 ∨ b ≠ 2 → a + b ≠ 3) ∧ (a + b ≠ 3 → a ≠ 1 ∨ b ≠ 2) :=
by
  sorry

end necessary_but_not_sufficient_l86_86457


namespace number_of_straight_A_students_l86_86694

-- Define the initial conditions and numbers
variables {x y : ℕ}

-- Define the initial student count and conditions on percentages
def initial_student_count := 25
def new_student_count := 7
def total_student_count := initial_student_count + new_student_count
def initial_percentage (x : ℕ) := (x : ℚ) / initial_student_count * 100
def new_percentage (x y : ℕ) := ((x + y : ℚ) / total_student_count) * 100

theorem number_of_straight_A_students
  (x y : ℕ)
  (h : initial_percentage x + 10 = new_percentage x y) :
  (x + y = 16) :=
sorry

end number_of_straight_A_students_l86_86694


namespace find_sets_l86_86088

theorem find_sets (a b c d : ℕ) (h₁ : 1 < a) (h₂ : a < b) (h₃ : b < c) (h₄ : c < d)
  (h₅ : (abcd - 1) % ((a-1) * (b-1) * (c-1) * (d-1)) = 0) :
  (a = 3 ∧ b = 5 ∧ c = 17 ∧ d = 255) ∨ (a = 2 ∧ b = 4 ∧ c = 10 ∧ d = 80) :=
by
  sorry

end find_sets_l86_86088


namespace david_distance_to_airport_l86_86052

theorem david_distance_to_airport (t : ℝ) (d : ℝ) :
  (35 * (t + 1) = d) ∧ (d - 35 = 50 * (t - 1.5)) → d = 210 :=
by
  sorry

end david_distance_to_airport_l86_86052


namespace ceil_mul_eq_225_l86_86971

theorem ceil_mul_eq_225 {x : ℝ} (h₁ : ⌈x⌉ * x = 225) (h₂ : x > 0) : x = 15 :=
sorry

end ceil_mul_eq_225_l86_86971


namespace k_eq_1_l86_86078

theorem k_eq_1 
  (n m k : ℕ) 
  (hn : n > 0) 
  (hm : m > 0) 
  (hk : k > 0) 
  (h : (n - 1) * n * (n + 1) = m^k) : 
  k = 1 := 
sorry

end k_eq_1_l86_86078


namespace combined_earnings_l86_86511

theorem combined_earnings (dwayne_earnings brady_earnings : ℕ) (h1 : dwayne_earnings = 1500) (h2 : brady_earnings = dwayne_earnings + 450) : 
  dwayne_earnings + brady_earnings = 3450 :=
by 
  rw [h1, h2]
  sorry

end combined_earnings_l86_86511


namespace proof_problem_l86_86830

variables {a b c : ℝ}

theorem proof_problem (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : 4 * a^2 + b^2 + 16 * c^2 = 1) :
  (0 < a * b ∧ a * b < 1 / 4) ∧ (1 / a^2 + 1 / b^2 + 1 / (4 * a * b * c^2) > 49) :=
by
  sorry

end proof_problem_l86_86830


namespace problem_conditions_l86_86333

theorem problem_conditions (m : ℝ) (hf_pow : m^2 - m - 1 = 1) (hf_inc : m > 0) : m = 2 :=
sorry

end problem_conditions_l86_86333


namespace condition_A_is_necessary_but_not_sufficient_for_condition_B_l86_86165

-- Define conditions
variables (a b : ℝ)

-- Condition A: ab > 0
def condition_A : Prop := a * b > 0

-- Condition B: a > 0 and b > 0
def condition_B : Prop := a > 0 ∧ b > 0

-- Prove that condition_A is a necessary but not sufficient condition for condition_B
theorem condition_A_is_necessary_but_not_sufficient_for_condition_B :
  (condition_A a b → condition_B a b) ∧ ¬(condition_B a b → condition_A a b) :=
by
  sorry

end condition_A_is_necessary_but_not_sufficient_for_condition_B_l86_86165


namespace race_distance_l86_86171

/-- Given that Sasha, Lesha, and Kolya start a 100m race simultaneously and run at constant velocities,
when Sasha finishes, Lesha is 10m behind, and when Lesha finishes, Kolya is 10m behind.
Prove that the distance between Sasha and Kolya when Sasha finishes is 19 meters. -/
theorem race_distance
    (v_S v_L v_K : ℝ)
    (h1 : 100 / v_S - 100 / v_L = 10 / v_L)
    (h2 : 100 / v_L - 100 / v_K = 10 / v_K) :
    100 - 81 = 19 :=
by
  sorry

end race_distance_l86_86171


namespace integer_multiple_of_ten_l86_86999

theorem integer_multiple_of_ten (x : ℤ) :
  10 * x = 30 ↔ x = 3 :=
by
  sorry

end integer_multiple_of_ten_l86_86999


namespace population_net_increase_l86_86046

-- Define conditions
def birth_rate : ℚ := 5 / 2    -- 5 people every 2 seconds
def death_rate : ℚ := 3 / 2    -- 3 people every 2 seconds
def one_day_in_seconds : ℕ := 86400   -- Number of seconds in one day

-- Define the net increase per second
def net_increase_per_second := birth_rate - death_rate

-- Prove that the net increase in one day is 86400 people given the conditions
theorem population_net_increase :
  net_increase_per_second * one_day_in_seconds = 86400 :=
sorry

end population_net_increase_l86_86046


namespace find_a_value_l86_86993

theorem find_a_value 
  (a : ℝ)
  (h : abs (1 - (-1 / (4 * a))) = 2) :
  a = 1 / 4 ∨ a = -1 / 12 :=
sorry

end find_a_value_l86_86993


namespace find_j_of_scaled_quadratic_l86_86120

/- Define the given condition -/
def quadratic_expressed (p q r : ℝ) : Prop :=
  ∀ x : ℝ, p * x^2 + q * x + r = 5 * (x - 3)^2 + 15

/- State the theorem to be proved -/
theorem find_j_of_scaled_quadratic (p q r m j l : ℝ) (h_quad : quadratic_expressed p q r) :
  (∀ x : ℝ, 2 * p * x^2 + 2 * q * x + 2 * r = m * (x - j)^2 + l) → j = 3 :=
by
  intro h
  sorry

end find_j_of_scaled_quadratic_l86_86120


namespace range_function_1_l86_86920

theorem range_function_1 (y : ℝ) : 
  (∃ x : ℝ, x ≥ -1 ∧ y = (1/3) ^ x) ↔ (0 < y ∧ y ≤ 3) :=
sorry

end range_function_1_l86_86920


namespace no_three_digit_whole_number_solves_log_eq_l86_86003

noncomputable def log_function (b : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log b

theorem no_three_digit_whole_number_solves_log_eq :
  ¬ ∃ n : ℤ, (100 ≤ n ∧ n < 1000) ∧ log_function (3 * n) 10 + log_function (7 * n) 10 = 1 :=
by
  sorry

end no_three_digit_whole_number_solves_log_eq_l86_86003


namespace ratio_of_larger_to_smaller_l86_86958

theorem ratio_of_larger_to_smaller
  (x y : ℝ) (h₁ : 0 < y) (h₂ : y < x) (h3 : x + y = 6 * (x - y)) :
  x / y = 7 / 5 :=
by sorry

end ratio_of_larger_to_smaller_l86_86958


namespace only_odd_integer_option_l86_86258

theorem only_odd_integer_option : 
  (6 ^ 2 = 36 ∧ Even 36) ∧ 
  (23 - 17 = 6 ∧ Even 6) ∧ 
  (9 * 24 = 216 ∧ Even 216) ∧ 
  (96 / 8 = 12 ∧ Even 12) ∧ 
  (9 * 41 = 369 ∧ Odd 369)
:= by
  sorry

end only_odd_integer_option_l86_86258


namespace find_primes_a_l86_86142

theorem find_primes_a :
  ∀ (a : ℕ), (∀ n : ℕ, n < a → Nat.Prime (4 * n * n + a)) → (a = 3 ∨ a = 7) :=
by
  sorry

end find_primes_a_l86_86142


namespace jawbreakers_in_package_correct_l86_86732

def jawbreakers_ate : Nat := 20
def jawbreakers_left : Nat := 4
def jawbreakers_in_package : Nat := jawbreakers_ate + jawbreakers_left

theorem jawbreakers_in_package_correct : jawbreakers_in_package = 24 := by
  sorry

end jawbreakers_in_package_correct_l86_86732


namespace jim_miles_remaining_l86_86217

theorem jim_miles_remaining (total_miles : ℕ) (miles_driven : ℕ) (total_miles_eq : total_miles = 1200) (miles_driven_eq : miles_driven = 384) :
  total_miles - miles_driven = 816 :=
by
  sorry

end jim_miles_remaining_l86_86217


namespace increase_in_area_is_44_percent_l86_86767

-- Let's define the conditions first
variables {r : ℝ} -- radius of the medium pizza
noncomputable def radius_large (r : ℝ) := 1.2 * r
noncomputable def area (r : ℝ) := Real.pi * r ^ 2

-- Now we state the Lean theorem that expresses the problem
theorem increase_in_area_is_44_percent (r : ℝ) : 
  (area (radius_large r) - area r) / area r * 100 = 44 :=
by
  sorry

end increase_in_area_is_44_percent_l86_86767


namespace farmer_plants_rows_per_bed_l86_86206

theorem farmer_plants_rows_per_bed 
    (bean_seedlings : ℕ) (beans_per_row : ℕ)
    (pumpkin_seeds : ℕ) (pumpkins_per_row : ℕ)
    (radishes : ℕ) (radishes_per_row : ℕ)
    (plant_beds : ℕ)
    (h1 : bean_seedlings = 64)
    (h2 : beans_per_row = 8)
    (h3 : pumpkin_seeds = 84)
    (h4 : pumpkins_per_row = 7)
    (h5 : radishes = 48)
    (h6 : radishes_per_row = 6)
    (h7 : plant_beds = 14) : 
    (bean_seedlings / beans_per_row + pumpkin_seeds / pumpkins_per_row + radishes / radishes_per_row) / plant_beds = 2 :=
by
  sorry

end farmer_plants_rows_per_bed_l86_86206


namespace total_charge_for_first_4_minutes_under_plan_A_is_0_60_l86_86813

def planA_charges (X : ℝ) (minutes : ℕ) : ℝ :=
  if minutes <= 4 then X
  else X + (minutes - 4) * 0.06

def planB_charges (minutes : ℕ) : ℝ :=
  minutes * 0.08

theorem total_charge_for_first_4_minutes_under_plan_A_is_0_60
  (X : ℝ)
  (h : planA_charges X 18 = planB_charges 18) :
  X = 0.60 :=
by
  sorry

end total_charge_for_first_4_minutes_under_plan_A_is_0_60_l86_86813


namespace beads_per_package_eq_40_l86_86592

theorem beads_per_package_eq_40 (b r : ℕ) (x : ℕ) (total_beads : ℕ) 
(h1 : b = 3) (h2 : r = 5) (h3 : total_beads = 320) (h4 : total_beads = (b + r) * x) :
  x = 40 := by
  sorry

end beads_per_package_eq_40_l86_86592


namespace total_string_length_l86_86543

theorem total_string_length 
  (circumference1 : ℝ) (height1 : ℝ) (loops1 : ℕ)
  (circumference2 : ℝ) (height2 : ℝ) (loops2 : ℕ)
  (h1 : circumference1 = 6) (h2 : height1 = 20) (h3 : loops1 = 5)
  (h4 : circumference2 = 3) (h5 : height2 = 10) (h6 : loops2 = 3)
  : (loops1 * Real.sqrt (circumference1 ^ 2 + (height1 / loops1) ^ 2) + loops2 * Real.sqrt (circumference2 ^ 2 + (height2 / loops2) ^ 2)) = (5 * Real.sqrt 52 + 3 * Real.sqrt 19.89) := 
by {
  sorry
}

end total_string_length_l86_86543


namespace angle_value_l86_86703

theorem angle_value (α : ℝ) (h1 : 0 ≤ α) (h2 : α < 360) 
(h3 : (Real.sin 215 * π / 180, Real.cos 215 * π / 180) = (Real.sin α, Real.cos α)) :
α = 235 :=
sorry

end angle_value_l86_86703


namespace reeya_third_subject_score_l86_86761

theorem reeya_third_subject_score
  (score1 score2 score4 : ℕ)
  (avg_score : ℕ)
  (num_subjects : ℕ)
  (total_score : ℕ)
  (score3 : ℕ) :
  score1 = 65 →
  score2 = 67 →
  score4 = 85 →
  avg_score = 75 →
  num_subjects = 4 →
  total_score = avg_score * num_subjects →
  score1 + score2 + score3 + score4 = total_score →
  score3 = 83 :=
by
  intros h1 h2 h4 h5 h6 h7 h8
  sorry

end reeya_third_subject_score_l86_86761


namespace find_P20_l86_86214

theorem find_P20 (a b : ℝ) (P : ℝ → ℝ) (hP : ∀ x, P x = x^2 + a * x + b) 
  (h_condition : P 10 + P 30 = 40) : P 20 = -80 :=
by {
  -- Additional statements to structure the proof can go here
  sorry
}

end find_P20_l86_86214


namespace lines_intersect_at_point_l86_86382

noncomputable def line1 (s : ℚ) : ℚ × ℚ :=
  (1 + 2 * s, 4 - 3 * s)

noncomputable def line2 (v : ℚ) : ℚ × ℚ :=
  (3 + 3 * v, 2 - v)

theorem lines_intersect_at_point :
  ∃ s v : ℚ,
    line1 s = (15 / 7, 16 / 7) ∧
    line2 v = (15 / 7, 16 / 7) ∧
    s = 4 / 7 ∧
    v = -2 / 7 := by
  sorry

end lines_intersect_at_point_l86_86382


namespace circle_radius_square_l86_86202

-- Definition of the problem setup
variables {EF GH ER RF GS SH R S : ℝ}

-- Given conditions
def condition1 : ER = 23 := by sorry
def condition2 : RF = 23 := by sorry
def condition3 : GS = 31 := by sorry
def condition4 : SH = 15 := by sorry

-- Circle radius to be proven
def radius_squared : ℝ := 706

-- Lean 4 theorem statement
theorem circle_radius_square (h1 : ER = 23) (h2 : RF = 23) (h3 : GS = 31) (h4 : SH = 15) :
  (r : ℝ) ^ 2 = 706 := sorry

end circle_radius_square_l86_86202


namespace line_parallel_xaxis_l86_86444

theorem line_parallel_xaxis (x y : ℝ) : y = 2 ↔ (∃ a b : ℝ, a = 4 ∧ b = 2 ∧ y = 2) :=
by 
  sorry

end line_parallel_xaxis_l86_86444


namespace existential_proposition_l86_86667

theorem existential_proposition :
  (∃ x y : ℝ, x + y > 1) ∧ (∀ P : Prop, (∃ x y : ℝ, x + y > 1 → P) → P) :=
sorry

end existential_proposition_l86_86667


namespace perimeter_of_rectangle_l86_86779

-- Define the conditions
def area (l w : ℝ) : Prop := l * w = 180
def length_three_times_width (l w : ℝ) : Prop := l = 3 * w

-- Define the problem
theorem perimeter_of_rectangle (l w : ℝ) (h₁ : area l w) (h₂ : length_three_times_width l w) : 
  2 * (l + w) = 16 * Real.sqrt 15 := 
sorry

end perimeter_of_rectangle_l86_86779


namespace surface_area_of_4cm_cube_after_corner_removal_l86_86130

noncomputable def surface_area_after_corner_removal (cube_side original_surface_length corner_cube_side : ℝ) : ℝ := 
  let num_faces : ℕ := 6
  let num_corners : ℕ := 8
  let surface_area_one_face := cube_side * cube_side
  let original_surface_area := num_faces * surface_area_one_face
  let corner_surface_area_one_face := 3 * (corner_cube_side * corner_cube_side)
  let exposed_surface_area_one_face := 3 * (corner_cube_side * corner_cube_side)
  let net_change_per_corner_cube := -corner_surface_area_one_face + exposed_surface_area_one_face
  let total_change := num_corners * net_change_per_corner_cube
  original_surface_area + total_change

theorem surface_area_of_4cm_cube_after_corner_removal : 
  ∀ (cube_side original_surface_length corner_cube_side : ℝ), 
  cube_side = 4 ∧ original_surface_length = 4 ∧ corner_cube_side = 2 →
  surface_area_after_corner_removal cube_side original_surface_length corner_cube_side = 96 :=
by
  intros cube_side original_surface_length corner_cube_side h
  rcases h with ⟨hs, ho, hc⟩
  rw [hs, ho, hc]
  sorry

end surface_area_of_4cm_cube_after_corner_removal_l86_86130


namespace highest_number_paper_l86_86976

theorem highest_number_paper (n : ℕ) (h : (1 : ℝ) / n = 0.010526315789473684) : n = 95 :=
sorry

end highest_number_paper_l86_86976


namespace parabola_focus_line_ratio_l86_86676

noncomputable def ratio_AF_BF : ℝ := (Real.sqrt 5 + 3) / 2

theorem parabola_focus_line_ratio :
  ∀ (F A B : ℝ × ℝ), 
    F = (1, 0) ∧ 
    (A.2 = 2 * A.1 - 2 ∧ A.2^2 = 4 * A.1 ) ∧ 
    (B.2 = 2 * B.1 - 2 ∧ B.2^2 = 4 * B.1) ∧ 
    A.2 > 0 -> 
  |(A.1 - F.1) / (B.1 - F.1)| = ratio_AF_BF :=
by
  sorry

end parabola_focus_line_ratio_l86_86676


namespace valid_parameterizations_l86_86770

open Real

def is_scalar_multiple (v1 v2 : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, v1 = (k * v2.1, k * v2.2)

def lies_on_line (p : ℝ × ℝ) : Prop :=
  p.2 = 2 * p.1 - 7

def valid_parametrization (p d : ℝ × ℝ) : Prop :=
  lies_on_line p ∧ is_scalar_multiple d (1, 2)

theorem valid_parameterizations :
  valid_parametrization (4, 1) (-2, -4) ∧ 
  ¬ valid_parametrization (12, 17) (5, 10) ∧ 
  valid_parametrization (3.5, 0) (1, 2) ∧ 
  valid_parametrization (-2, -11) (0.5, 1) ∧ 
  valid_parametrization (0, -7) (10, 20) :=
by {
  sorry
}

end valid_parameterizations_l86_86770


namespace abs_ineq_l86_86110

theorem abs_ineq (x : ℝ) (h : |x + 1| > 3) : x < -4 ∨ x > 2 :=
  sorry

end abs_ineq_l86_86110


namespace proof_6_times_15_times_5_eq_2_l86_86408

noncomputable def given_condition (a b c : ℝ) : Prop :=
  a * b * c = (Real.sqrt ((a + 2) * (b + 3))) / (c + 1)

theorem proof_6_times_15_times_5_eq_2 : 
  given_condition 6 15 5 → 6 * 15 * 5 = 2 :=
by
  sorry

end proof_6_times_15_times_5_eq_2_l86_86408


namespace remainder_when_divided_l86_86404

theorem remainder_when_divided (N : ℤ) (k : ℤ) (h : N = 296 * k + 75) : N % 37 = 1 :=
by
  sorry

end remainder_when_divided_l86_86404


namespace pipe_A_fill_time_l86_86641

theorem pipe_A_fill_time (t : ℝ) (h1 : t > 0) (h2 : ∃ tA tB, tA = t ∧ tB = t / 6 ∧ (tA + tB) = 3) : t = 21 :=
by
  sorry

end pipe_A_fill_time_l86_86641


namespace Rams_monthly_salary_l86_86719

variable (R S A : ℝ)
variable (annual_salary : ℝ)
variable (monthly_salary_conversion : annual_salary / 12 = A)
variable (ram_shyam_condition : 0.10 * R = 0.08 * S)
variable (shyam_abhinav_condition : S = 2 * A)
variable (abhinav_annual_salary : annual_salary = 192000)

theorem Rams_monthly_salary 
  (annual_salary : ℝ)
  (ram_shyam_condition : 0.10 * R = 0.08 * S)
  (shyam_abhinav_condition : S = 2 * A)
  (abhinav_annual_salary : annual_salary = 192000)
  (monthly_salary_conversion: annual_salary / 12 = A): 
  R = 25600 := by
  sorry

end Rams_monthly_salary_l86_86719


namespace p_div_q_is_12_l86_86738

-- Definition of binomials and factorials required for the proof
open Nat

/-- Define the number of ways to distribute balls for configuration A -/
def config_A : ℕ :=
  @choose 5 1 * @choose 4 2 * @choose 2 1 * (factorial 20) / (factorial 2 * factorial 4 * factorial 4 * factorial 3 * factorial 7)

/-- Define the number of ways to distribute balls for configuration B -/
def config_B : ℕ :=
  @choose 5 2 * @choose 3 3 * (factorial 20) / (factorial 3 * factorial 3 * factorial 4 * factorial 4 * factorial 4)

/-- The ratio of probabilities p/q for the given distributions of balls into bins is 12 -/
theorem p_div_q_is_12 : config_A / config_B = 12 :=
by
  sorry

end p_div_q_is_12_l86_86738


namespace simplify_expression_l86_86181

theorem simplify_expression : ( (2^8 + 4^5) * (2^3 - (-2)^3) ^ 8 ) = 0 := 
by sorry

end simplify_expression_l86_86181


namespace card_prob_ace_of_hearts_l86_86153

def problem_card_probability : Prop :=
  let deck_size := 52
  let draw_size := 2
  let ace_hearts := 1
  let total_combinations := Nat.choose deck_size draw_size
  let favorable_combinations := deck_size - ace_hearts
  let probability := favorable_combinations / total_combinations
  probability = 1 / 26

theorem card_prob_ace_of_hearts : problem_card_probability := by
  sorry

end card_prob_ace_of_hearts_l86_86153


namespace common_property_of_rhombus_and_rectangle_l86_86824

structure Rhombus :=
  (bisect_perpendicular : ∀ d₁ d₂ : ℝ, ∃ p : ℝ × ℝ, p = (0, 0))
  (diagonals_not_equal : ∀ d₁ d₂ : ℝ, ¬(d₁ = d₂))

structure Rectangle :=
  (bisect_each_other : ∀ d₁ d₂ : ℝ, ∃ p : ℝ × ℝ, p = (0, 0))
  (diagonals_equal : ∀ d₁ d₂ : ℝ, d₁ = d₂)

theorem common_property_of_rhombus_and_rectangle (R : Rhombus) (S : Rectangle) :
  ∀ d₁ d₂ : ℝ, ∃ p : ℝ × ℝ, p = (0, 0) :=
by
  -- Assuming the properties of Rhombus R and Rectangle S
  sorry

end common_property_of_rhombus_and_rectangle_l86_86824


namespace marble_cut_in_third_week_l86_86797

def percentage_cut_third_week := 
  let initial_weight : ℝ := 250 
  let final_weight : ℝ := 105
  let percent_cut_first_week : ℝ := 0.30
  let percent_cut_second_week : ℝ := 0.20
  let weight_after_first_week := initial_weight * (1 - percent_cut_first_week)
  let weight_after_second_week := weight_after_first_week * (1 - percent_cut_second_week)
  (weight_after_second_week - final_weight) / weight_after_second_week * 100 = 25

theorem marble_cut_in_third_week :
  percentage_cut_third_week = true :=
by
  sorry

end marble_cut_in_third_week_l86_86797


namespace marathon_speed_ratio_l86_86506

theorem marathon_speed_ratio (M D : ℝ) (J : ℝ) (H1 : D = 9) (H2 : J = 4/3 * M) (H3 : M + J + D = 23) :
  D / M = 3 / 2 :=
by
  sorry

end marathon_speed_ratio_l86_86506


namespace intersection_empty_condition_l86_86789

-- Define the sets M and N under the given conditions
def M : Set (ℝ × ℝ) := { p | p.1^2 + 2 * p.2^2 = 3 }

def N (m b : ℝ) : Set (ℝ × ℝ) := { p | p.2 = m * p.1 + b }

-- The theorem that we need to prove based on the problem statement
theorem intersection_empty_condition (b : ℝ) :
  (∀ m : ℝ, M ∩ N m b = ∅) ↔ (b^2 > 6 * m^2 + 2) := sorry

end intersection_empty_condition_l86_86789


namespace find_y_l86_86309

open Complex

theorem find_y (y : ℝ) (h₁ : (3 : ℂ) + (↑y : ℂ) * I = z₁) 
  (h₂ : (2 : ℂ) - I = z₂) 
  (h₃ : z₁ / z₂ = 1 + I) 
  (h₄ : z₁ = (3 : ℂ) + (↑y : ℂ) * I) 
  (h₅ : z₂ = (2 : ℂ) - I)
  : y = 1 :=
sorry


end find_y_l86_86309


namespace distance_between_trees_l86_86942

theorem distance_between_trees (l : ℕ) (n : ℕ) (d : ℕ) (h_length : l = 225) (h_trees : n = 26) (h_segments : n - 1 = 25) : d = 9 :=
sorry

end distance_between_trees_l86_86942


namespace proof_u_g_3_l86_86375

noncomputable def u (x : ℝ) : ℝ := Real.sqrt (5 * x + 2)

noncomputable def g (x : ℝ) : ℝ := 7 - u x

theorem proof_u_g_3 :
  u (g 3) = Real.sqrt (37 - 5 * Real.sqrt 17) :=
sorry

end proof_u_g_3_l86_86375


namespace initial_green_marbles_l86_86730

theorem initial_green_marbles (m g' : ℕ) (h_m : m = 23) (h_g' : g' = 9) : (g' + m = 32) :=
by
  subst h_m
  subst h_g'
  rfl

end initial_green_marbles_l86_86730


namespace number_of_dogs_with_both_tags_and_collars_l86_86962

-- Defining the problem
def total_dogs : ℕ := 80
def dogs_with_tags : ℕ := 45
def dogs_with_collars : ℕ := 40
def dogs_with_neither : ℕ := 1

-- Statement: Prove the number of dogs with both tags and collars
theorem number_of_dogs_with_both_tags_and_collars : 
  (dogs_with_tags + dogs_with_collars - total_dogs + dogs_with_neither) = 6 :=
by
  sorry

end number_of_dogs_with_both_tags_and_collars_l86_86962


namespace exists_bound_for_expression_l86_86720

theorem exists_bound_for_expression :
  ∃ (C : ℝ), (∀ (k : ℤ), abs ((k^8 - 2*k + 1 : ℤ) / (k^4 - 3 : ℤ)) < C) := 
sorry

end exists_bound_for_expression_l86_86720


namespace solve_for_nabla_l86_86438

theorem solve_for_nabla : (∃ (nabla : ℤ), 5 * (-3) + 4 = nabla + 7) → (∃ (nabla : ℤ), nabla = -18) :=
by
  sorry

end solve_for_nabla_l86_86438


namespace vertex_parabola_shape_l86_86944

theorem vertex_parabola_shape
  (a d : ℕ) (ha : 0 < a) (hd : 0 < d) :
  ∃ (P : ℝ → ℝ → Prop), 
  (∀ t : ℝ, ∃ (x y : ℝ), P x y ∧ (x = (-t / (2 * a))) ∧ (y = -a * (x^2) + d)) ∧
  (∀ x y : ℝ, P x y ↔ (y = -a * (x^2) + d)) :=
by
  sorry

end vertex_parabola_shape_l86_86944


namespace compute_fraction_l86_86175

theorem compute_fraction : 
  (1 - 2 + 4 - 8 + 16 - 32 + 64) / (2 - 4 + 8 - 16 + 32 - 64 + 128) = 1 / 2 := 
by
  sorry

end compute_fraction_l86_86175


namespace peaches_total_l86_86023

theorem peaches_total (n P : ℕ) (h1 : P - 6 * n = 57) (h2 : P = 9 * (n - 6) + 3) : P = 273 :=
by
  sorry

end peaches_total_l86_86023


namespace cylindrical_to_rectangular_l86_86726

theorem cylindrical_to_rectangular (r θ z : ℝ) (h₁ : r = 10) (h₂ : θ = Real.pi / 6) (h₃ : z = 2) :
  (r * Real.cos θ, r * Real.sin θ, z) = (5 * Real.sqrt 3, 5, 2) := 
by
  sorry

end cylindrical_to_rectangular_l86_86726


namespace cone_section_volume_ratio_l86_86963

theorem cone_section_volume_ratio :
  ∀ (r h : ℝ), (h > 0 ∧ r > 0) →
  let V1 := ((75 / 3) * π * r^2 * h - (64 / 3) * π * r^2 * h)
  let V2 := ((64 / 3) * π * r^2 * h - (27 / 3) * π * r^2 * h)
  V2 / V1 = 37 / 11 :=
by
  intros r h h_pos
  sorry

end cone_section_volume_ratio_l86_86963


namespace total_sampled_students_l86_86379

-- Define the total number of students in each grade
def students_in_grade12 : ℕ := 700
def students_in_grade11 : ℕ := 700
def students_in_grade10 : ℕ := 800

-- Define the number of students sampled from grade 10
def sampled_from_grade10 : ℕ := 80

-- Define the total number of students in the school
def total_students : ℕ := students_in_grade12 + students_in_grade11 + students_in_grade10

-- Prove that the total number of students sampled (x) is equal to 220
theorem total_sampled_students : 
  (sampled_from_grade10 : ℚ) / (students_in_grade10 : ℚ) * (total_students : ℚ) = 220 := 
by
  sorry

end total_sampled_students_l86_86379


namespace difference_of_squares_l86_86739

theorem difference_of_squares (x y : ℕ) (h1 : x + y = 26) (h2 : x * y = 168) : x^2 - y^2 = 52 := by
  sorry

end difference_of_squares_l86_86739


namespace tangent_through_points_l86_86449

theorem tangent_through_points :
  ∀ (x₁ x₂ : ℝ),
    (∀ y₁ y₂ : ℝ, y₁ = x₁^2 + 1 → y₂ = x₂^2 + 1 → 
    (2 * x₁ * (x₂ - x₁) + y₁ = 0 → x₂ = -x₁) ∧ 
    (2 * x₂ * (x₁ - x₂) + y₂ = 0 → x₁ = -x₂)) →
  (x₁ = 1 / Real.sqrt 3 ∧ x₂ = -1 / Real.sqrt 3 ∧
   (x₁^2 + 1 = (1 / 3) + 1) ∧ (x₂^2 + 1 = (1 / 3) + 1)) :=
by
  sorry

end tangent_through_points_l86_86449


namespace find_m_n_l86_86513

open Nat

-- Define binomial coefficient
def binom (n k : ℕ) : ℕ := n.choose k

theorem find_m_n (m n : ℕ) (h1 : binom (n+1) (m+1) / binom (n+1) m = 5 / 3) 
  (h2 : binom (n+1) m / binom (n+1) (m-1) = 5 / 3) : m = 3 ∧ n = 6 :=
  sorry

end find_m_n_l86_86513


namespace value_of_N_l86_86403

theorem value_of_N (N : ℕ) (x y z w s : ℕ) (h_pos_x : 0 < x) (h_pos_y : 0 < y)
    (h_pos_z : 0 < z) (h_pos_w : 0 < w) (h_pos_s : 0 < s) (h_sum : x + y + z + w + s = N)
    (h_comb : Nat.choose N 4 = 3003) : N = 18 := 
by
  sorry

end value_of_N_l86_86403


namespace ratio_of_third_to_second_is_four_l86_86327

theorem ratio_of_third_to_second_is_four
  (x y z k : ℕ)
  (h1 : y = 2 * x)
  (h2 : z = k * y)
  (h3 : (x + y + z) / 3 = 165)
  (h4 : y = 90) :
  z / y = 4 :=
by
  sorry

end ratio_of_third_to_second_is_four_l86_86327


namespace inequality_solution_l86_86337

theorem inequality_solution (x : ℝ) : 
  (x + 10) / (x^2 + 2 * x + 5) ≥ 0 ↔ x ∈ Set.Ici (-10) :=
sorry

end inequality_solution_l86_86337


namespace sin_cos_difference_theorem_tan_theorem_l86_86982

open Real

noncomputable def sin_cos_difference (x : ℝ) : Prop :=
  -π / 2 < x ∧ x < 0 ∧ (sin x + cos x = 1 / 5) ∧ (sin x - cos x = - 7 / 5)

theorem sin_cos_difference_theorem (x : ℝ) (h : sin_cos_difference x) : 
  sin x - cos x = - 7 / 5 := by
  sorry

noncomputable def sin_cos_ratio (x : ℝ) : Prop :=
  -π / 2 < x ∧ x < 0 ∧ (sin x + cos x = 1 / 5) ∧ (sin x - cos x = - 7 / 5) ∧ (tan x = -3 / 4)

theorem tan_theorem (x : ℝ) (h : sin_cos_ratio x) :
  tan x = -3 / 4 := by
  sorry

end sin_cos_difference_theorem_tan_theorem_l86_86982


namespace point_not_in_region_l86_86614

-- Define the inequality
def inequality (x y : ℝ) : Prop := 3 * x + 2 * y < 6

-- Points definition
def point := ℝ × ℝ

-- Points to be checked
def p1 : point := (0, 0)
def p2 : point := (1, 1)
def p3 : point := (0, 2)
def p4 : point := (2, 0)

-- Conditions stating that certain points satisfy the inequality
axiom h1 : inequality p1.1 p1.2
axiom h2 : inequality p2.1 p2.2
axiom h3 : inequality p3.1 p3.2

-- Goal: Prove that point (2,0) does not satisfy the inequality
theorem point_not_in_region : ¬ inequality p4.1 p4.2 :=
sorry -- Proof omitted

end point_not_in_region_l86_86614


namespace total_rainfall_correct_l86_86689

-- Define the individual rainfall amounts
def rainfall_mon1 : ℝ := 0.17
def rainfall_wed1 : ℝ := 0.42
def rainfall_fri : ℝ := 0.08
def rainfall_mon2 : ℝ := 0.37
def rainfall_wed2 : ℝ := 0.51

-- Define the total rainfall
def total_rainfall : ℝ := rainfall_mon1 + rainfall_wed1 + rainfall_fri + rainfall_mon2 + rainfall_wed2

-- Theorem statement to prove the total rainfall is 1.55 cm
theorem total_rainfall_correct : total_rainfall = 1.55 :=
by
  -- Proof goes here
  sorry

end total_rainfall_correct_l86_86689


namespace crayons_given_correct_l86_86102

def crayons_lost : ℕ := 161
def additional_crayons : ℕ := 410
def crayons_given (lost : ℕ) (additional : ℕ) : ℕ := lost + additional

theorem crayons_given_correct : crayons_given crayons_lost additional_crayons = 571 :=
by
  sorry

end crayons_given_correct_l86_86102


namespace ratio_b_c_l86_86706

theorem ratio_b_c (a b c d e f : ℝ) 
  (h1 : a / b = 1 / 3)
  (h2 : a * b * c / (d * e * f) = 0.1875)
  (h3 : c / d = 1 / 2)
  (h4 : d / e = 3)
  (h5 : e / f = 1 / 8) : 
  b / c = 3 :=
sorry

end ratio_b_c_l86_86706


namespace table_legs_l86_86417

theorem table_legs (total_tables : ℕ) (total_legs : ℕ) (four_legged_tables : ℕ) (four_legged_count : ℕ) 
  (other_legged_tables : ℕ) (other_legged_count : ℕ) :
  total_tables = 36 →
  total_legs = 124 →
  four_legged_tables = 16 →
  four_legged_count = 4 →
  other_legged_tables = total_tables - four_legged_tables →
  total_legs = (four_legged_tables * four_legged_count) + (other_legged_tables * other_legged_count) →
  other_legged_count = 3 := 
by
  sorry

end table_legs_l86_86417


namespace resulting_figure_has_25_sides_l86_86728

/-- Consider a sequential construction starting with an isosceles triangle, adding a rectangle 
    on one side, then a regular hexagon on a non-adjacent side of the rectangle, followed by a
    regular heptagon, another regular hexagon, and finally, a regular nonagon. -/
def sides_sequence : List ℕ := [3, 4, 6, 7, 6, 9]

/-- The number of sides exposed to the outside in the resulting figure. -/
def exposed_sides (sides : List ℕ) : ℕ :=
  let total_sides := sides.sum
  let adjacent_count := 2 + 2 + 2 + 2 + 1
  total_sides - adjacent_count

theorem resulting_figure_has_25_sides :
  exposed_sides sides_sequence = 25 := 
by
  sorry

end resulting_figure_has_25_sides_l86_86728


namespace smallest_x_2_abs_eq_24_l86_86752

theorem smallest_x_2_abs_eq_24 : ∃ x : ℝ, (2 * |x - 10| = 24) ∧ (∀ y : ℝ, (2 * |y - 10| = 24) -> x ≤ y) := 
sorry

end smallest_x_2_abs_eq_24_l86_86752


namespace correct_quotient_is_32_l86_86160

-- Definitions based on the conditions
def incorrect_divisor := 12
def correct_divisor := 21
def incorrect_quotient := 56
def dividend := incorrect_divisor * incorrect_quotient -- Given as 672

-- Statement of the theorem
theorem correct_quotient_is_32 :
  dividend / correct_divisor = 32 :=
by
  -- skip the proof
  sorry

end correct_quotient_is_32_l86_86160


namespace only_pairs_satisfying_conditions_l86_86455

theorem only_pairs_satisfying_conditions (a b : ℕ) (a_pos : 0 < a) (b_pos : 0 < b) :
  (b^2 + b + 1) % a = 0 ∧ (a^2 + a + 1) % b = 0 → a = 1 ∧ b = 1 :=
by
  sorry

end only_pairs_satisfying_conditions_l86_86455


namespace exists_100_digit_number_divisible_by_sum_of_digits_l86_86690

-- Definitions
def is_100_digit_number (n : ℕ) : Prop :=
  10^99 ≤ n ∧ n < 10^100

def no_zero_digits (n : ℕ) : Prop :=
  ∀ d ∈ n.digits 10, d ≠ 0

def sum_of_digits (n : ℕ) : ℕ :=
  (n.digits 10).sum

def is_divisible_by_sum_of_digits (n : ℕ) : Prop :=
  n % sum_of_digits n = 0

-- Main theorem statement
theorem exists_100_digit_number_divisible_by_sum_of_digits :
  ∃ n : ℕ, is_100_digit_number n ∧ no_zero_digits n ∧ is_divisible_by_sum_of_digits n :=
sorry

end exists_100_digit_number_divisible_by_sum_of_digits_l86_86690


namespace quadratic_real_roots_l86_86903

variable (a b : ℝ)

theorem quadratic_real_roots (h : ∀ a : ℝ, ∃ x : ℝ, x^2 - 2*a*x - a + 2*b = 0) : b ≤ -1/8 :=
by
  sorry

end quadratic_real_roots_l86_86903


namespace minimum_value_polynomial_l86_86288

def polynomial (x y : ℝ) : ℝ := 5 * x^2 - 4 * x * y + 4 * y^2 + 12 * x + 25

theorem minimum_value_polynomial : ∃ (m : ℝ), (∀ (x y : ℝ), polynomial x y ≥ m) ∧ m = 16 :=
by
  sorry

end minimum_value_polynomial_l86_86288


namespace total_opponents_points_is_36_l86_86268
-- Import the Mathlib library

-- Define the conditions as Lean definitions
def game_scores : List ℕ := [3, 5, 6, 7, 8, 9, 11, 12]

def lost_by_two (n : ℕ) : Prop := n + 2 ∈ game_scores

def three_times_as_many (n : ℕ) : Prop := n * 3 ∈ game_scores

-- State the problem
theorem total_opponents_points_is_36 : 
  (∃ l1 l2 l3 w1 w2 w3 w4 w5 : ℕ, 
    game_scores = [l1, l2, l3, w1, w2, w3, w4, w5] ∧
    lost_by_two l1 ∧ lost_by_two l2 ∧ lost_by_two l3 ∧
    three_times_as_many w1 ∧ three_times_as_many w2 ∧ 
    three_times_as_many w3 ∧ three_times_as_many w4 ∧ 
    three_times_as_many w5 ∧ 
    l1 + 2 + l2 + 2 + l3 + 2 + ((w1 / 3) + (w2 / 3) + (w3 / 3) + (w4 / 3) + (w5 / 3)) = 36) :=
sorry

end total_opponents_points_is_36_l86_86268


namespace fifth_term_in_geometric_sequence_l86_86077

variable (y : ℝ)

def geometric_sequence : ℕ → ℝ
| 0       => 3
| (n + 1) => geometric_sequence n * (3 * y)

theorem fifth_term_in_geometric_sequence (y : ℝ) : 
  geometric_sequence y 4 = 243 * y^4 :=
sorry

end fifth_term_in_geometric_sequence_l86_86077


namespace min_value_x_l86_86229

open Real 

variable (x : ℝ)

theorem min_value_x (hx_pos : 0 < x) 
    (ineq : log x ≥ 2 * log 3 + (1 / 3) * log x + 1) : 
    x ≥ 27 * exp (3 / 2) :=
by 
  sorry

end min_value_x_l86_86229


namespace frame_interior_edge_sum_l86_86163

theorem frame_interior_edge_sum (y : ℝ) :
  ( ∀ outer_edge1 : ℝ, outer_edge1 = 7 →
    ∀ frame_width : ℝ, frame_width = 2 →
    ∀ frame_area : ℝ, frame_area = 30 →
    7 * y - (3 * (y - 4)) = 30) → 
  (7 * y - (4 * y - 12) ) / 4 = 4.5 → 
  (3 + (y - 4)) * 2 = 7 :=
sorry

end frame_interior_edge_sum_l86_86163


namespace minimum_point_translation_l86_86941

noncomputable def f (x : ℝ) : ℝ := |x| - 2

theorem minimum_point_translation :
  let minPoint := (0, f 0)
  let newMinPoint := (minPoint.1 + 4, minPoint.2 + 5)
  newMinPoint = (4, 3) :=
by
  sorry

end minimum_point_translation_l86_86941


namespace find_years_lent_to_B_l86_86472

def principal_B := 5000
def principal_C := 3000
def rate := 8
def time_C := 4
def total_interest := 1760

-- Interest calculation for B
def interest_B (n : ℕ) := (principal_B * rate * n) / 100

-- Interest calculation for C (constant time of 4 years)
def interest_C := (principal_C * rate * time_C) / 100

-- Total interest received
def total_interest_received (n : ℕ) := interest_B n + interest_C

theorem find_years_lent_to_B (n : ℕ) (h : total_interest_received n = total_interest) : n = 2 :=
by
  sorry

end find_years_lent_to_B_l86_86472


namespace correct_description_of_sperm_l86_86085

def sperm_carries_almost_no_cytoplasm (sperm : Type) : Prop := sorry

theorem correct_description_of_sperm : sperm_carries_almost_no_cytoplasm sperm := 
sorry

end correct_description_of_sperm_l86_86085


namespace pipe_A_fill_time_l86_86901

theorem pipe_A_fill_time (B C : ℝ) (hB : B = 8) (hC : C = 14.4) (hB_not_zero : B ≠ 0) (hC_not_zero : C ≠ 0) :
  ∃ (A : ℝ), (1 / A + 1 / B = 1 / C) ∧ A = 24 :=
by
  sorry

end pipe_A_fill_time_l86_86901


namespace roots_geometric_progression_condition_l86_86289

theorem roots_geometric_progression_condition 
  (a b c : ℝ) 
  (x1 x2 x3 : ℝ)
  (h1 : x1 + x2 + x3 = -a)
  (h2 : x1 * x2 + x2 * x3 + x1 * x3 = b)
  (h3 : x1 * x2 * x3 = -c)
  (h4 : x2^2 = x1 * x3) :
  a^3 * c = b^3 :=
sorry

end roots_geometric_progression_condition_l86_86289


namespace unique_solution_l86_86638

def is_valid_func (f : ℕ → ℕ) : Prop :=
  ∀ n, f (f n) + f n = 2 * n + 2001 ∨ f (f n) + f n = 2 * n + 2002

theorem unique_solution (f : ℕ → ℕ) (hf : is_valid_func f) :
  ∀ n, f n = n + 667 :=
sorry

end unique_solution_l86_86638


namespace cookies_per_bag_l86_86549

-- Definitions of the given conditions
def c1 := 23  -- number of chocolate chip cookies
def c2 := 25  -- number of oatmeal cookies
def b := 8    -- number of baggies

-- Statement to prove
theorem cookies_per_bag : (c1 + c2) / b = 6 :=
by 
  sorry

end cookies_per_bag_l86_86549


namespace total_marks_l86_86365

variable (marks_in_music marks_in_maths marks_in_arts marks_in_social_studies : ℕ)

def marks_conditions : Prop :=
  marks_in_maths = marks_in_music - (1/10) * marks_in_music ∧
  marks_in_maths = marks_in_arts - 20 ∧
  marks_in_social_studies = marks_in_music + 10 ∧
  marks_in_music = 70

theorem total_marks 
  (h : marks_conditions marks_in_music marks_in_maths marks_in_arts marks_in_social_studies) :
  marks_in_music + marks_in_maths + marks_in_arts + marks_in_social_studies = 296 :=
by
  sorry

end total_marks_l86_86365


namespace find_a_solve_inequality_intervals_of_monotonicity_l86_86623

-- Problem 1: Prove a = 2 given conditions
theorem find_a (a : ℝ) (h₀ : a > 0) (h₁ : a ≠ 1) (h₂ : Real.log 3 / Real.log a > Real.log 2 / Real.log a) 
    (h₃ : Real.log (2 * a) / Real.log a - Real.log a / Real.log a = 1) : a = 2 := 
  by
  sorry

-- Problem 2: Prove the solution interval for inequality
theorem solve_inequality (x a : ℝ) (h₀ : 1 < x) (h₁ : x < 3 / 2) : 
    Real.log (x - 1) / Real.log (1 / 3) > Real.log (a - x) / Real.log (1 / 3) :=
  by
  have ha : a = 2 := sorry
  sorry

-- Problem 3: Prove intervals of monotonicity for g(x)
theorem intervals_of_monotonicity (x : ℝ) : 
  (∀ x : ℝ, 0 < x → x ≤ 2 → (|Real.log x / Real.log 2 - 1| : ℝ) = 1 - Real.log x / Real.log 2) ∧ 
  (∀ x : ℝ, x > 2 → (|Real.log x / Real.log 2 - 1| : ℝ) = Real.log x / Real.log 2 - 1) :=
  by
  sorry

end find_a_solve_inequality_intervals_of_monotonicity_l86_86623


namespace chebyshev_birth_year_l86_86471

theorem chebyshev_birth_year :
  ∃ (a b : ℕ),
  a > b ∧ 
  a + b = 3 ∧ 
  (1821 = 1800 + 10 * a + 1 * b) ∧
  (1821 + 73) < 1900 :=
by sorry

end chebyshev_birth_year_l86_86471


namespace value_of_c_plus_d_l86_86286

theorem value_of_c_plus_d (a b c d : ℝ) (h1 : a + b = 5) (h2 : b + c = 6) (h3 : a + d = 2) : c + d = 3 :=
by
  sorry

end value_of_c_plus_d_l86_86286


namespace least_M_bench_sections_l86_86044

/--
A single bench section at a community event can hold either 8 adults, 12 children, or 10 teenagers. 
We are to find the smallest positive integer M such that when M bench sections are connected end to end,
an equal number of adults, children, and teenagers seated together will occupy all the bench space.
-/
theorem least_M_bench_sections
  (M : ℕ)
  (hM_pos : M > 0)
  (adults_capacity : ℕ := 8 * M)
  (children_capacity : ℕ := 12 * M)
  (teenagers_capacity : ℕ := 10 * M)
  (h_equal_capacity : adults_capacity = children_capacity ∧ children_capacity = teenagers_capacity) :
  M = 15 := 
sorry

end least_M_bench_sections_l86_86044


namespace spring_summer_work_hours_l86_86949

def john_works_spring_summer : Prop :=
  ∀ (work_hours_winter_week : ℕ) (weeks_winter : ℕ) (earnings_winter : ℕ)
    (weeks_spring_summer : ℕ) (earnings_spring_summer : ℕ) (hourly_rate : ℕ),
    work_hours_winter_week = 40 →
    weeks_winter = 8 →
    earnings_winter = 3200 →
    weeks_spring_summer = 24 →
    earnings_spring_summer = 4800 →
    hourly_rate = earnings_winter / (work_hours_winter_week * weeks_winter) →
    (earnings_spring_summer / hourly_rate) / weeks_spring_summer = 20

theorem spring_summer_work_hours : john_works_spring_summer :=
  sorry

end spring_summer_work_hours_l86_86949


namespace power_sums_fifth_l86_86358

noncomputable def compute_power_sums (α β γ : ℂ) : ℂ :=
  α^5 + β^5 + γ^5

theorem power_sums_fifth (α β γ : ℂ)
  (h1 : α + β + γ = 2)
  (h2 : α^2 + β^2 + γ^2 = 5)
  (h3 : α^3 + β^3 + γ^3 = 10) :
  compute_power_sums α β γ = 47.2 :=
sorry

end power_sums_fifth_l86_86358


namespace discard_sacks_l86_86772

theorem discard_sacks (harvested_sacks_per_day : ℕ) (oranges_per_day : ℕ) (oranges_per_sack : ℕ) :
  harvested_sacks_per_day = 76 → oranges_per_day = 600 → oranges_per_sack = 50 → 
  harvested_sacks_per_day - oranges_per_day / oranges_per_sack = 64 :=
by
  intros h1 h2 h3
  -- Automatically passes the proof as a placeholder
  sorry

end discard_sacks_l86_86772


namespace not_possible_to_color_plane_l86_86786

theorem not_possible_to_color_plane :
  ¬ ∃ (color : ℕ → ℕ × ℕ → ℕ) (c : ℕ), 
    (c = 2016) ∧
    (∀ (A B C : ℕ × ℕ), (A ≠ B ∧ B ≠ C ∧ C ≠ A) → 
                        (color c A = color c B) ∨ (color c B = color c C) ∨ (color c C = color c A)) :=
by
  sorry

end not_possible_to_color_plane_l86_86786


namespace trigonometric_identity_l86_86093

theorem trigonometric_identity (α β : ℝ) : 
  ((Real.tan α + Real.tan β) / Real.tan (α + β)) 
  + ((Real.tan α - Real.tan β) / Real.tan (α - β)) 
  + 2 * (Real.tan α) ^ 2 
 = 2 / (Real.cos α) ^ 2 :=
  sorry

end trigonometric_identity_l86_86093


namespace not_777_integers_l86_86421

theorem not_777_integers (p : ℕ) (hp : Nat.Prime p) :
  ¬ (∃ count : ℕ, count = 777 ∧ ∀ n : ℕ, ∃ k : ℕ, (n ^ 3 + n * p + 1 = k * (n + p + 1))) :=
by
  sorry

end not_777_integers_l86_86421


namespace fourth_term_geometric_progression_l86_86854

theorem fourth_term_geometric_progression
  (x : ℝ)
  (h : ∀ n : ℕ, n ≥ 0 → (3 * x * (n : ℝ) + 3 * (n : ℝ)) = (6 * x * ((n - 1) : ℝ) + 6 * ((n - 1) : ℝ))) :
  (((3*x + 3)^2 = (6*x + 6) * x) ∧ x = -3) → (∀ n : ℕ, n = 4 → (2^(n-3) * (6*x + 6)) = -24) :=
by
  sorry

end fourth_term_geometric_progression_l86_86854


namespace smallest_sum_of_five_consecutive_primes_divisible_by_three_l86_86396

-- Definition of the conditions
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def consecutive_primes (a b c d e : ℕ) : Prop :=
  is_prime a ∧ is_prime b ∧ is_prime c ∧ is_prime d ∧ is_prime e ∧
  (b = a + 1 ∨ b = a + 2) ∧ (c = b + 1 ∨ c = b + 2) ∧
  (d = c + 1 ∨ d = c + 2) ∧ (e = d + 1 ∨ e = d + 2)

theorem smallest_sum_of_five_consecutive_primes_divisible_by_three :
  ∃ a b c d e, consecutive_primes a b c d e ∧ a + b + c + d + e = 39 ∧ 39 % 3 = 0 :=
sorry

end smallest_sum_of_five_consecutive_primes_divisible_by_three_l86_86396


namespace second_polygon_sides_l86_86381

theorem second_polygon_sides (s : ℝ) (n : ℝ) (h1 : 50 * 3 * s = n * s) : n = 150 := 
by
  sorry

end second_polygon_sides_l86_86381


namespace crayons_total_l86_86067

theorem crayons_total (Wanda Dina Jacob: ℕ) (hW: Wanda = 62) (hD: Dina = 28) (hJ: Jacob = Dina - 2) :
  Wanda + Dina + Jacob = 116 :=
by
  sorry

end crayons_total_l86_86067


namespace sheila_hourly_wage_l86_86851

-- Definition of conditions
def hours_per_day_mon_wed_fri := 8
def days_mon_wed_fri := 3
def hours_per_day_tue_thu := 6
def days_tue_thu := 2
def weekly_earnings := 432

-- Variables derived from conditions
def total_hours_mon_wed_fri := hours_per_day_mon_wed_fri * days_mon_wed_fri
def total_hours_tue_thu := hours_per_day_tue_thu * days_tue_thu
def total_hours_per_week := total_hours_mon_wed_fri + total_hours_tue_thu

-- Proof statement
theorem sheila_hourly_wage : (weekly_earnings / total_hours_per_week) = 12 := 
sorry

end sheila_hourly_wage_l86_86851


namespace different_testing_methods_1_different_testing_methods_2_l86_86823

-- Definitions used in Lean 4 statement should be derived from the conditions in a).
def total_products := 10
def defective_products := 4
def non_defective_products := total_products - defective_products
def choose (n k : Nat) : Nat := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Statement (1)
theorem different_testing_methods_1 :
  let first_defective := 5
  let last_defective := 10
  let non_defective_in_first_4 := choose 6 4
  let defective_in_middle_5 := choose 5 3
  let total_methods := non_defective_in_first_4 * defective_in_middle_5 * Nat.factorial 5 * Nat.factorial 4
  total_methods = 103680 := sorry

-- Statement (2)
theorem different_testing_methods_2 :
  let first_defective := 5
  let remaining_defective := 4
  let non_defective_in_first_4 := choose 6 4
  let total_methods := non_defective_in_first_4 * Nat.factorial 5
  total_methods = 576 := sorry

end different_testing_methods_1_different_testing_methods_2_l86_86823


namespace shortest_track_length_l86_86532

open Nat

def Melanie_track_length := 8
def Martin_track_length := 20

theorem shortest_track_length :
  Nat.lcm Melanie_track_length Martin_track_length = 40 :=
by
  sorry

end shortest_track_length_l86_86532


namespace domain_of_log_function_l86_86808

theorem domain_of_log_function (x : ℝ) :
  (-1 < x ∧ x < 1) ↔ (1 - x) / (1 + x) > 0 :=
by sorry

end domain_of_log_function_l86_86808


namespace unique_largest_negative_integer_l86_86866

theorem unique_largest_negative_integer :
  ∃! x : ℤ, x = -1 ∧ (∀ y : ℤ, y < 0 → x ≥ y) :=
by
  sorry

end unique_largest_negative_integer_l86_86866


namespace bags_on_monday_l86_86946

/-- Define the problem conditions -/
def t : Nat := 8  -- total number of bags
def f : Nat := 4  -- number of bags found the next day

-- Define the statement to be proven
theorem bags_on_monday : t - f = 4 := by
  -- Sorry to skip the proof
  sorry

end bags_on_monday_l86_86946


namespace cos_4theta_l86_86407

theorem cos_4theta (θ : ℝ) (h : Real.cos θ = 1 / 4) : Real.cos (4 * θ) = 17 / 32 :=
sorry

end cos_4theta_l86_86407


namespace ian_investment_percentage_change_l86_86335

theorem ian_investment_percentage_change :
  let initial_investment := 200
  let first_year_loss := 0.10
  let second_year_gain := 0.25
  let amount_after_loss := initial_investment * (1 - first_year_loss)
  let amount_after_gain := amount_after_loss * (1 + second_year_gain)
  let percentage_change := (amount_after_gain - initial_investment) / initial_investment * 100
  percentage_change = 12.5 := 
by
  sorry

end ian_investment_percentage_change_l86_86335


namespace fraction_reduction_l86_86436

theorem fraction_reduction (x y : ℝ) : 
  (4 * x - 4 * y) / (4 * x * 4 * y) = (1 / 4) * ((x - y) / (x * y)) := 
by 
  sorry

end fraction_reduction_l86_86436


namespace original_team_players_l86_86794

theorem original_team_players (n : ℕ) (W : ℝ)
    (h1 : W = n * 76)
    (h2 : (W + 110 + 60) / (n + 2) = 78) : n = 7 :=
  sorry

end original_team_players_l86_86794


namespace double_inequality_solution_l86_86320

open Set

theorem double_inequality_solution (x : ℝ) :
  -1 < (x^2 - 16 * x + 24) / (x^2 - 4 * x + 8) ∧
  (x^2 - 16 * x + 24) / (x^2 - 4 * x + 8) < 1 ↔
  x ∈ Ioo (3 / 2) 4 ∪ Ioi 8 :=
by
  sorry

end double_inequality_solution_l86_86320


namespace max_area_rectangle_l86_86412

theorem max_area_rectangle (P : ℝ) (hP : P = 60) (a b : ℝ) (h1 : b = 3 * a) (h2 : 2 * a + 2 * b = P) : a * b = 168.75 :=
by
  sorry

end max_area_rectangle_l86_86412


namespace hotel_accommodation_arrangements_l86_86582

theorem hotel_accommodation_arrangements :
  let triple_room := 1
  let double_rooms := 2
  let adults := 3
  let children := 2
  (∀ (triple_room : ℕ) (double_rooms : ℕ) (adults : ℕ) (children : ℕ),
    children ≤ adults ∧ double_rooms + triple_room ≥ 1 →
    (∃ (arrangements : ℕ),
      arrangements = 60)) :=
sorry

end hotel_accommodation_arrangements_l86_86582


namespace original_length_before_sharpening_l86_86602

/-- Define the current length of the pencil after sharpening -/
def current_length : ℕ := 14

/-- Define the length of the pencil that was sharpened off -/
def sharpened_off_length : ℕ := 17

/-- Prove that the original length of the pencil before sharpening was 31 inches -/
theorem original_length_before_sharpening : current_length + sharpened_off_length = 31 := by
  sorry

end original_length_before_sharpening_l86_86602


namespace scientific_notation_example_l86_86187

theorem scientific_notation_example :
  284000000 = 2.84 * 10^8 :=
by
  sorry

end scientific_notation_example_l86_86187


namespace at_least_one_zero_l86_86047

theorem at_least_one_zero (a b : ℝ) : (¬ (a ≠ 0 ∧ b ≠ 0)) → (a = 0 ∨ b = 0) := by
  intro h
  have h' : ¬ ((a ≠ 0) ∧ (b ≠ 0)) := h
  sorry

end at_least_one_zero_l86_86047


namespace avg_mark_excluded_students_l86_86230

-- Define the given conditions
variables (n : ℕ) (A A_remaining : ℕ) (excluded_count : ℕ)
variable (T : ℕ := n * A)
variable (T_remaining : ℕ := (n - excluded_count) * A_remaining)
variable (T_excluded : ℕ := T - T_remaining)

-- Define the problem statement
theorem avg_mark_excluded_students (h1: n = 14) (h2: A = 65) (h3: A_remaining = 90) (h4: excluded_count = 5) :
   T_excluded / excluded_count = 20 :=
by
  sorry

end avg_mark_excluded_students_l86_86230


namespace part1_part2_part3_part4_l86_86937

section QuadraticFunction

variable {x : ℝ} {y : ℝ} 

-- 1. Prove that if a quadratic function y = x^2 + bx - 3 intersects the x-axis at (3, 0), 
-- then b = -2 and the other intersection point is (-1, 0).
theorem part1 (b : ℝ) : 
  ((3:ℝ) ^ 2 + b * (3:ℝ) - 3 = 0) → 
  b = -2 ∧ ∃ x : ℝ, (x = -1 ∧ x^2 + b * x - 3 = 0) := 
  sorry

-- 2. For the function y = x^2 + bx - 3 where b = -2, 
-- prove that when 0 < y < 5, x is in -2 < x < -1 or 3 < x < 4.
theorem part2 (b : ℝ) :
  b = -2 → 
  (0 < y ∧ y < 5 → ∃ x : ℝ, (x^2 + b * x - 3 = y) → (-2 < x ∧ x < -1) ∨ (3 < x ∧ x < 4)) :=
  sorry

-- 3. Prove that the value t such that y = x^2 + bx - 3 and y > t always holds for all x
-- is t < -((b ^ 2 + 12) / 4).
theorem part3 (b t : ℝ) :
  (∀ x : ℝ, (x ^ 2 + b * x - 3 > t)) → t < -(b ^ 2 + 12) / 4 :=
  sorry

-- 4. Given y = x^2 - 3x - 3 and 1 < x < 2, 
-- prove that m < y < n with n = -5, b = -3, and m ≤ -21 / 4.
theorem part4 (m n : ℝ) :
  (1 < x ∧ x < 2 → m < x^2 - 3 * x - 3 ∧ x^2 - 3 * x - 3 < n) →
  n = -5 ∧ -21 / 4 ≤ m :=
  sorry

end QuadraticFunction

end part1_part2_part3_part4_l86_86937


namespace option_C_is_correct_l86_86367

theorem option_C_is_correct :
  (-3 - (-2) ≠ -5) ∧
  (-|(-1:ℝ)/3| + 1 ≠ 4/3) ∧
  (4 - 4 / 2 = 2) ∧
  (3^2 / 6 * (1/6) ≠ 9) :=
by
  -- Proof omitted
  sorry

end option_C_is_correct_l86_86367


namespace distance_CD_l86_86293

theorem distance_CD (C D : ℝ × ℝ) (r₁ r₂ : ℝ) (φ₁ φ₂ : ℝ) 
  (hC : C = (r₁, φ₁)) (hD : D = (r₂, φ₂)) (r₁_eq_5 : r₁ = 5) (r₂_eq_12 : r₂ = 12)
  (angle_diff : φ₁ - φ₂ = π / 3) : dist C D = Real.sqrt 109 :=
  sorry

end distance_CD_l86_86293


namespace polynomial_factor_pq_l86_86428

theorem polynomial_factor_pq (p q : ℝ) (h : ∀ x : ℝ, (x^2 + 2*x + 5) ∣ (x^4 + p*x^2 + q)) : p + q = 31 :=
sorry

end polynomial_factor_pq_l86_86428


namespace f_at_neg2_l86_86817

noncomputable def f (x : ℝ) : ℝ :=
if x > 0 then 2^x - Real.log (x^2 - 3*x + 5) / Real.log 3 
else -2^(-x) + Real.log ((-x)^2 + 3*(-x) + 5) / Real.log 3 

theorem f_at_neg2 : f (-2) = -3 := by
  sorry

end f_at_neg2_l86_86817


namespace remaining_puppies_l86_86758

def initial_puppies : Nat := 8
def given_away_puppies : Nat := 4

theorem remaining_puppies : initial_puppies - given_away_puppies = 4 := 
by 
  sorry

end remaining_puppies_l86_86758


namespace infinite_geometric_sum_l86_86967

noncomputable def geometric_sequence (n : ℕ) : ℝ := 3 * (-1 / 2)^(n - 1)

theorem infinite_geometric_sum :
  ∑' n, geometric_sequence n = 2 :=
sorry

end infinite_geometric_sum_l86_86967


namespace part1_part2_l86_86574

def f (x a : ℝ) := |x - a| + 2 * |x + 1|

-- Part 1: Solve the inequality f(x) > 4 when a = 2
theorem part1 (x : ℝ) : f x 2 > 4 ↔ (x < -4/3 ∨ x > 0) := by
  sorry

-- Part 2: If the solution set of the inequality f(x) < 3x + 4 is {x | x > 2}, find the value of a.
theorem part2 (a : ℝ) : (∀ x : ℝ, (f x a < 3 * x + 4 ↔ x > 2)) → a = 6 := by
  sorry

end part1_part2_l86_86574


namespace number_of_workers_l86_86634

theorem number_of_workers (W C : ℕ) (h1 : W * C = 300000) (h2 : W * (C + 50) = 350000) : W = 1000 :=
sorry

end number_of_workers_l86_86634


namespace values_of_x_l86_86938

theorem values_of_x (x : ℝ) : (x+2)*(x-9) < 0 ↔ -2 < x ∧ x < 9 := 
by
  sorry

end values_of_x_l86_86938


namespace total_amount_245_l86_86008

-- Define the conditions and the problem
theorem total_amount_245 (a : ℝ) (x y z : ℝ) (h1 : y = 0.45 * a) (h2 : z = 0.30 * a) (h3 : y = 63) :
  x + y + z = 245 := 
by
  -- Starting the proof (proof steps are unnecessary as per the procedure)
  sorry

end total_amount_245_l86_86008


namespace part1_monotonicity_part2_find_range_l86_86848

noncomputable def f (x a : ℝ) : ℝ := Real.exp x + a * x^2 - x

-- Part (1): Monotonicity when a = 1
theorem part1_monotonicity : 
  ∀ x : ℝ, 
    ( f x 1 > f (x - 1) 1 ∧ x > 0 ) ∨ 
    ( f x 1 < f (x + 1) 1 ∧ x < 0 ) :=
  sorry

-- Part (2): Finding the range of a when x ≥ 0
theorem part2_find_range (x a : ℝ) (h : 0 ≤ x) (ineq : f x a ≥ 1/2 * x^3 + 1) : 
  a ≥ (7 - Real.exp 2) / 4 :=
  sorry

end part1_monotonicity_part2_find_range_l86_86848


namespace new_class_mean_l86_86954

theorem new_class_mean 
  (n1 n2 : ℕ) 
  (mean1 mean2 : ℝ)
  (students_total : ℕ)
  (total_score1 total_score2 : ℝ)
  (h1 : n1 = 45)
  (h2 : n2 = 5)
  (h3 : mean1 = 80)
  (h4 : mean2 = 90)
  (h5 : students_total = 50)
  (h6 : total_score1 = n1 * mean1)
  (h7 : total_score2 = n2 * mean2) :
  (total_score1 + total_score2) / students_total = 81 :=
by
  sorry

end new_class_mean_l86_86954


namespace find_x_y_l86_86453

theorem find_x_y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) 
  (h1 : (x * y / 7) ^ (3 / 2) = x) 
  (h2 : (x * y / 7) = y) : 
  x = 7 ∧ y = 7 ^ (2 / 3) :=
by
  sorry

end find_x_y_l86_86453


namespace number_of_possible_values_b_l86_86570

theorem number_of_possible_values_b : 
  ∃ n : ℕ, n = 2 ∧ 
    (∀ b : ℕ, b ≥ 2 → (b^3 ≤ 256) ∧ (256 < b^4) ↔ (b = 5 ∨ b = 6)) :=
by {
  sorry
}

end number_of_possible_values_b_l86_86570


namespace knights_probability_l86_86467

theorem knights_probability :
  let knights : Nat := 30
  let chosen : Nat := 4
  let probability (n k : Nat) := 1 - (((n - k + 1) * (n - k - 1) * (n - k - 3) * (n - k - 5)) / 
                                      ((n - 0) * (n - 1) * (n - 2) * (n - 3)))
  probability knights chosen = (389 / 437) := sorry

end knights_probability_l86_86467


namespace triple_complement_angle_l86_86042

theorem triple_complement_angle (x : ℝ) (h1 : x = 3 * (90 - x)) : x = 67.5 :=
sorry

end triple_complement_angle_l86_86042


namespace zhou_yu_age_at_death_l86_86376

theorem zhou_yu_age_at_death (x : ℕ) (h₁ : 1 ≤ x ∧ x ≤ 9)
    (h₂ : ∃ age : ℕ, age = 10 * (x - 3) + x)
    (h₃ : x^2 = 10 * (x - 3) + x) :
    x^2 = 10 * (x - 3) + x :=
by
  sorry

end zhou_yu_age_at_death_l86_86376


namespace num_regions_of_lines_l86_86427

theorem num_regions_of_lines (R : ℕ → ℕ) :
  R 1 = 2 ∧ 
  (∀ n, R (n + 1) = R n + (n + 1)) →
  (∀ n, R n = (n * (n + 1)) / 2 + 1) :=
by
  intro h
  sorry

end num_regions_of_lines_l86_86427


namespace alex_cakes_l86_86875

theorem alex_cakes :
  let slices_first_cake := 8
  let slices_second_cake := 12
  let given_away_friends_first := slices_first_cake / 4
  let remaining_after_friends_first := slices_first_cake - given_away_friends_first
  let given_away_family_first := remaining_after_friends_first / 2
  let remaining_after_family_first := remaining_after_friends_first - given_away_family_first
  let stored_in_freezer_first := remaining_after_family_first / 4
  let remaining_after_freezer_first := remaining_after_family_first - stored_in_freezer_first
  let remaining_after_eating_first := remaining_after_freezer_first - 2
  
  let given_away_friends_second := slices_second_cake / 3
  let remaining_after_friends_second := slices_second_cake - given_away_friends_second
  let given_away_family_second := remaining_after_friends_second / 6
  let remaining_after_family_second := remaining_after_friends_second - given_away_family_second
  let stored_in_freezer_second := remaining_after_family_second / 4
  let remaining_after_freezer_second := remaining_after_family_second - stored_in_freezer_second
  let remaining_after_eating_second := remaining_after_freezer_second - 1

  remaining_after_eating_first + stored_in_freezer_first + remaining_after_eating_second + stored_in_freezer_second = 7 :=
by
  -- Proof goes here
  sorry

end alex_cakes_l86_86875


namespace solve_expression_l86_86270

theorem solve_expression (x y z : ℚ)
  (h1 : 2 * x + 3 * y + z = 20)
  (h2 : x + 2 * y + 3 * z = 26)
  (h3 : 3 * x + y + 2 * z = 29) :
  12 * x^2 + 22 * x * y + 12 * y^2 + 12 * x * z + 12 * y * z + 12 * z^2 = (computed_value : ℚ) :=
by
  sorry

end solve_expression_l86_86270


namespace gcd_8251_6105_l86_86002

theorem gcd_8251_6105 : Nat.gcd 8251 6105 = 37 := by
  sorry

end gcd_8251_6105_l86_86002


namespace harmon_high_school_proof_l86_86243

noncomputable def harmon_high_school : Prop :=
  ∃ (total_players players_physics players_both players_chemistry : ℕ),
    total_players = 18 ∧
    players_physics = 10 ∧
    players_both = 3 ∧
    players_chemistry = (total_players - players_physics + players_both)

theorem harmon_high_school_proof : harmon_high_school :=
  sorry

end harmon_high_school_proof_l86_86243


namespace coloring_points_l86_86975

theorem coloring_points
  (A : ℤ × ℤ) (B : ℤ × ℤ) (C : ℤ × ℤ)
  (hA : A.fst % 2 = 1 ∧ A.snd % 2 = 1)
  (hB : (B.fst % 2 = 1 ∧ B.snd % 2 = 0) ∨ (B.fst % 2 = 0 ∧ B.snd % 2 = 1))
  (hC : C.fst % 2 = 0 ∧ C.snd % 2 = 0) :
  ∃ D : ℤ × ℤ,
    (D.fst % 2 = 1 ∧ D.snd % 2 = 0) ∨ (D.fst % 2 = 0 ∧ D.snd % 2 = 1) ∧
    (A.fst + C.fst = B.fst + D.fst) ∧
    (A.snd + C.snd = B.snd + D.snd) := 
sorry

end coloring_points_l86_86975


namespace correct_statement_about_Digital_Earth_l86_86776

-- Definitions of the statements
def statement_A : Prop :=
  "Digital Earth is a reflection of the real Earth through digital means" = "Correct statement about Digital Earth"

def statement_B : Prop :=
  "Digital Earth is an extension of GIS technology" = "Correct statement about Digital Earth"

def statement_C : Prop :=
  "Digital Earth can only achieve global information sharing through the internet" = "Correct statement about Digital Earth"

def statement_D : Prop :=
  "The core idea of Digital Earth is to use digital means to uniformly address Earth's issues" = "Correct statement about Digital Earth"

-- Theorem that needs to be proved 
theorem correct_statement_about_Digital_Earth : statement_C :=
by 
  sorry

end correct_statement_about_Digital_Earth_l86_86776


namespace time_interval_for_7_students_l86_86610

-- Definitions from conditions
def students_per_ride : ℕ := 7
def total_students : ℕ := 21
def total_time : ℕ := 15

-- Statement of the problem
theorem time_interval_for_7_students : (total_time / (total_students / students_per_ride)) = 5 := 
by sorry

end time_interval_for_7_students_l86_86610


namespace candidate_failed_by_25_marks_l86_86626

-- Define the given conditions
def maximum_marks : ℝ := 127.27
def passing_percentage : ℝ := 0.55
def marks_secured : ℝ := 45

-- Define the minimum passing marks
def minimum_passing_marks : ℝ := passing_percentage * maximum_marks

-- Define the number of failing marks the candidate missed
def failing_marks : ℝ := minimum_passing_marks - marks_secured

-- Define the main theorem to prove the candidate failed by 25 marks
theorem candidate_failed_by_25_marks :
  failing_marks = 25 := 
by
  sorry

end candidate_failed_by_25_marks_l86_86626


namespace arccos_cos_three_l86_86544

theorem arccos_cos_three : Real.arccos (Real.cos 3) = 3 - 2 * Real.pi :=
  sorry

end arccos_cos_three_l86_86544


namespace cube_expansion_l86_86410

theorem cube_expansion : 101^3 + 3 * 101^2 + 3 * 101 + 1 = 1061208 :=
by
  sorry

end cube_expansion_l86_86410


namespace jumps_correct_l86_86308

def R : ℕ := 157
def X : ℕ := 86
def total_jumps (R X : ℕ) : ℕ := R + (R + X)

theorem jumps_correct : total_jumps R X = 400 := by
  sorry

end jumps_correct_l86_86308


namespace sum_of_a2_and_a3_l86_86481

theorem sum_of_a2_and_a3 (S : ℕ → ℕ) (hS : ∀ n, S n = 3^n + 1) :
  S 3 - S 1 = 24 :=
by
  sorry

end sum_of_a2_and_a3_l86_86481


namespace extra_flowers_correct_l86_86168

variable (pickedTulips : ℕ) (pickedRoses : ℕ) (usedFlowers : ℕ)

def totalFlowers : ℕ := pickedTulips + pickedRoses
def extraFlowers : ℕ := totalFlowers pickedTulips pickedRoses - usedFlowers

theorem extra_flowers_correct : 
  pickedTulips = 39 → pickedRoses = 49 → usedFlowers = 81 → extraFlowers pickedTulips pickedRoses usedFlowers = 7 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end extra_flowers_correct_l86_86168


namespace sum_squares_l86_86588

theorem sum_squares (a b c : ℝ) (h1 : a + b + c = 22) (h2 : a * b + b * c + c * a = 116) : 
  (a^2 + b^2 + c^2 = 252) :=
by
  sorry

end sum_squares_l86_86588


namespace part1_max_min_part2_cos_value_l86_86294

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (Real.pi * x + Real.pi / 6)

theorem part1_max_min (x : ℝ) (hx : -1/2 ≤ x ∧ x ≤ 1/2) : 
  (∃ xₘ, (xₘ ∈ Set.Icc (-1/2) (1/2)) ∧ f xₘ = 2) ∧ 
  (∃ xₘ, (xₘ ∈ Set.Icc (-1/2) (1/2)) ∧ f xₘ = -Real.sqrt 3) :=
sorry

theorem part2_cos_value (α : ℝ) (h : f (α / (2 * Real.pi)) = 1/4) : 
  Real.cos (2 * Real.pi / 3 - α) = -31/32 :=
sorry

end part1_max_min_part2_cos_value_l86_86294


namespace garden_area_l86_86451

def radius : ℝ := 0.6
def pi_approx : ℝ := 3
def circle_area (r : ℝ) (π : ℝ) := π * r^2

theorem garden_area : circle_area radius pi_approx = 1.08 :=
by
  sorry

end garden_area_l86_86451


namespace infinitely_many_gt_sqrt_l86_86980

open Real

noncomputable def sequences := ℕ → ℕ × ℕ

def strictly_increasing_ratios (seq : sequences) : Prop :=
  ∀ n : ℕ, 0 < n → (seq (n + 1)).2 / (seq (n + 1)).1 > (seq n).2 / (seq n).1

theorem infinitely_many_gt_sqrt (seq : sequences) 
  (positive_integers : ∀ n : ℕ, (seq n).1 > 0 ∧ (seq n).2 > 0) 
  (inc_ratios : strictly_increasing_ratios seq) :
  ∃ᶠ n in at_top, (seq n).2 > sqrt n :=
sorry

end infinitely_many_gt_sqrt_l86_86980


namespace max_ab_is_nine_l86_86338

noncomputable def f (a b x : ℝ) : ℝ := 4 * x^3 - a * x^2 - 2 * b * x + 2

/-- If a > 0, b > 0, and the function f(x) = 4x^3 - ax^2 - 2bx + 2 has an extremum at x = 1, then the maximum value of ab is 9. -/
theorem max_ab_is_nine {a b : ℝ}
  (ha : a > 0) (hb : b > 0)
  (extremum_x1 : deriv (f a b) 1 = 0) :
  a * b ≤ 9 :=
sorry

end max_ab_is_nine_l86_86338


namespace num_choices_l86_86990

theorem num_choices (classes scenic_spots : ℕ) (h_classes : classes = 4) (h_scenic_spots : scenic_spots = 3) :
  (scenic_spots ^ classes) = 81 :=
by
  -- The detailed proof goes here
  sorry

end num_choices_l86_86990


namespace age_difference_l86_86918

/-- 
The overall age of x and y is some years greater than the overall age of y and z. Z is 12 years younger than X.
Prove: The overall age of x and y is 12 years greater than the overall age of y and z.
-/
theorem age_difference {X Y Z : ℕ} (h1: X + Y > Y + Z) (h2: Z = X - 12) : 
  (X + Y) - (Y + Z) = 12 :=
by 
  -- proof goes here
  sorry

end age_difference_l86_86918


namespace S_40_eq_150_l86_86038

variable {R : Type*} [Field R]

-- Define the sum function for geometric sequences.
noncomputable def geom_sum (a q : R) (n : ℕ) : R :=
  a * (1 - q^n) / (1 - q)

-- Given conditions from the problem.
axiom S_10_eq : ∀ {a q : R}, geom_sum a q 10 = 10
axiom S_30_eq : ∀ {a q : R}, geom_sum a q 30 = 70

-- The main theorem stating S40 = 150 under the given conditions.
theorem S_40_eq_150 {a q : R} (h10 : geom_sum a q 10 = 10) (h30 : geom_sum a q 30 = 70) :
  geom_sum a q 40 = 150 :=
sorry

end S_40_eq_150_l86_86038


namespace arithmetic_sequence_ratio_l86_86215

def arithmetic_sequence (a : ℕ → ℝ) :=
  ∃ (d : ℝ), ∀ n, a (n + 1) = a n + d

variable {a b : ℕ → ℝ}
variable {S T : ℕ → ℝ}

noncomputable def sum_first_n_terms (a : ℕ → ℝ) (n : ℕ) : ℝ :=
(n + 1) * (a 0 + a n) / 2

variable (S_eq_k_mul_n_plus_2 : ∀ n, S n = (n + 2) * (S 0 / (n + 2)))
variable (T_eq_k_mul_n_plus_1 : ∀ n, T n = (n + 1) * (T 0 / (n + 1)))

theorem arithmetic_sequence_ratio (h₁ : arithmetic_sequence a) (h₂ : arithmetic_sequence b)
  (h₃ : ∀ n, S n = sum_first_n_terms a n)
  (h₄ : ∀ n, T n = sum_first_n_terms b n)
  (h₅ : ∀ n, (S n) / (T n) = (n + 2) / (n + 1))
  : a 6 / b 8 = 13 / 16 := 
sorry

end arithmetic_sequence_ratio_l86_86215


namespace students_with_no_preference_l86_86107

def total_students : ℕ := 210
def prefer_mac : ℕ := 60
def equally_prefer_both (x : ℕ) : ℕ := x / 3

def no_preference_students : ℕ :=
  total_students - (prefer_mac + equally_prefer_both prefer_mac)

theorem students_with_no_preference :
  no_preference_students = 130 :=
by
  sorry

end students_with_no_preference_l86_86107


namespace function_passes_through_A_l86_86262

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 4 + Real.log x / Real.log a

theorem function_passes_through_A 
  (a : ℝ) 
  (h1 : 0 < a) 
  (h2 : a ≠ 1)
  : f a 2 = 4 := sorry

end function_passes_through_A_l86_86262


namespace points_difference_l86_86759

-- Define the given data
def points_per_touchdown : ℕ := 7
def brayden_gavin_touchdowns : ℕ := 7
def cole_freddy_touchdowns : ℕ := 9

-- Define the theorem to prove the difference in points
theorem points_difference :
  (points_per_touchdown * cole_freddy_touchdowns) - 
  (points_per_touchdown * brayden_gavin_touchdowns) = 14 :=
  by sorry

end points_difference_l86_86759


namespace percentOfNonUnionWomenIs90_l86_86617

variable (totalEmployees : ℕ) (percentMen : ℚ) (percentUnionized : ℚ) (percentUnionizedMen : ℚ)

noncomputable def percentNonUnionWomen : ℚ :=
  let numberOfMen := percentMen * totalEmployees
  let numberOfUnionEmployees := percentUnionized * totalEmployees
  let numberOfUnionMen := percentUnionizedMen * numberOfUnionEmployees
  let numberOfNonUnionEmployees := totalEmployees - numberOfUnionEmployees
  let numberOfNonUnionMen := numberOfMen - numberOfUnionMen
  let numberOfNonUnionWomen := numberOfNonUnionEmployees - numberOfNonUnionMen
  (numberOfNonUnionWomen / numberOfNonUnionEmployees) * 100

theorem percentOfNonUnionWomenIs90
  (h1 : percentMen = 46 / 100)
  (h2 : percentUnionized = 60 / 100)
  (h3 : percentUnionizedMen = 70 / 100) : percentNonUnionWomen 100 46 60 70 = 90 :=
sorry

end percentOfNonUnionWomenIs90_l86_86617


namespace a_equals_b_l86_86247

theorem a_equals_b (a b : ℕ) (h : a^3 + a + 4 * b^2 = 4 * a * b + b + b * a^2) : a = b := 
sorry

end a_equals_b_l86_86247


namespace rugby_team_new_avg_weight_l86_86036

noncomputable def new_average_weight (original_players : ℕ) (original_avg_weight : ℕ) 
  (new_player_weights : List ℕ) : ℚ :=
  let total_original_weight := original_players * original_avg_weight
  let total_new_weight := new_player_weights.foldl (· + ·) 0
  let new_total_weight := total_original_weight + total_new_weight
  let new_total_players := original_players + new_player_weights.length
  (new_total_weight : ℚ) / (new_total_players : ℚ)

theorem rugby_team_new_avg_weight :
  new_average_weight 20 180 [210, 220, 230] = 185.22 := by
  sorry

end rugby_team_new_avg_weight_l86_86036


namespace squirrels_in_tree_l86_86814

theorem squirrels_in_tree (nuts : ℕ) (squirrels : ℕ) (h1 : nuts = 2) (h2 : squirrels = nuts + 2) : squirrels = 4 :=
by
    rw [h1] at h2
    exact h2

end squirrels_in_tree_l86_86814


namespace common_chord_line_l86_86413

def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 2 * x - 8 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 + 2 * x - 4 * y - 4 = 0

theorem common_chord_line : 
  ∀ x y : ℝ, (circle1 x y ∧ circle2 x y) ↔ (x - y + 1 = 0) := 
by sorry

end common_chord_line_l86_86413


namespace cubic_inches_in_two_cubic_feet_l86_86055

theorem cubic_inches_in_two_cubic_feet (conv : 1 = 12) : 2 * (12 * 12 * 12) = 3456 :=
by
  sorry

end cubic_inches_in_two_cubic_feet_l86_86055


namespace polynomial_p_l86_86674

variable {a b c : ℝ}

theorem polynomial_p (a b c : ℝ) : 
  a * (b - c)^2 + b * (c - a)^2 + c * (a - b)^2 = (a - b) * (b - c) * (c - a) * 2 :=
by
  sorry

end polynomial_p_l86_86674


namespace hilt_has_2_pennies_l86_86715

-- Define the total value of coins each person has without considering Mrs. Hilt's pennies
def dimes : ℕ := 2
def nickels : ℕ := 2
def hilt_base_amount : ℕ := dimes * 10 + nickels * 5 -- 30 cents

def jacob_pennies : ℕ := 4
def jacob_nickels : ℕ := 1
def jacob_dimes : ℕ := 1
def jacob_amount : ℕ := jacob_pennies * 1 + jacob_nickels * 5 + jacob_dimes * 10 -- 19 cents

def difference : ℕ := 13
def hilt_pennies : ℕ := 2 -- The solution's correct answer

theorem hilt_has_2_pennies : hilt_base_amount - jacob_amount + hilt_pennies = difference := by sorry

end hilt_has_2_pennies_l86_86715


namespace limestone_amount_l86_86411

theorem limestone_amount (L S : ℝ) (h1 : L + S = 100) (h2 : 3 * L + 5 * S = 425) : L = 37.5 :=
by
  -- Proof will go here
  sorry

end limestone_amount_l86_86411


namespace min_value_expression_l86_86131

theorem min_value_expression (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  4 * x^3 + 8 * y^3 + 18 * z^3 + 1 / (6 * x * y * z) ≥ 4 := by
  sorry

end min_value_expression_l86_86131


namespace more_likely_condition_l86_86429

-- Definitions for the problem
def total_placements (n : ℕ) := n * n * (n * n - 1)

def not_same_intersection_placements (n : ℕ) := n * n * (n * n - 1)

def same_row_or_column_exclusions (n : ℕ) := 2 * n * (n - 1) * n

def not_same_street_placements (n : ℕ) := total_placements n - same_row_or_column_exclusions n

def probability_not_same_intersection (n : ℕ) := not_same_intersection_placements n / total_placements n

def probability_not_same_street (n : ℕ) := not_same_street_placements n / total_placements n

-- Main proposition
theorem more_likely_condition (n : ℕ) (h : n = 7) :
  probability_not_same_intersection n > probability_not_same_street n := 
by 
  sorry

end more_likely_condition_l86_86429


namespace geometric_sequence_condition_l86_86625

theorem geometric_sequence_condition (a : ℕ → ℝ) :
  (∀ n ≥ 2, a n = 2 * a (n-1)) → 
  (∃ r, r = 2 ∧ ∀ n ≥ 2, a n = r * a (n-1)) ∧ 
  (∃ b, b ≠ 0 ∧ ∀ n, a n = 0) :=
sorry

end geometric_sequence_condition_l86_86625


namespace find_water_and_bucket_weight_l86_86533

-- Define the original amount of water (x) and the weight of the bucket (y)
variables (x y : ℝ)

-- Given conditions described as hypotheses
def conditions (x y : ℝ) : Prop :=
  4 * x + y = 16 ∧ 6 * x + y = 22

-- The goal is to prove the values of x and y
theorem find_water_and_bucket_weight (h : conditions x y) : x = 3 ∧ y = 4 :=
by
  sorry

end find_water_and_bucket_weight_l86_86533


namespace Elaine_rent_increase_l86_86664

noncomputable def Elaine_rent_percent (E: ℝ) : ℝ :=
  let last_year_rent := 0.20 * E
  let this_year_earnings := 1.25 * E
  let this_year_rent := 0.30 * this_year_earnings
  let ratio := (this_year_rent / last_year_rent) * 100
  ratio

theorem Elaine_rent_increase (E : ℝ) : Elaine_rent_percent E = 187.5 :=
by 
  -- The proof would go here.
  sorry

end Elaine_rent_increase_l86_86664


namespace number_is_2250_l86_86049

-- Question: Prove that x = 2250 given the condition.
theorem number_is_2250 (x : ℕ) (h : x / 5 = 75 + x / 6) : x = 2250 := 
sorry

end number_is_2250_l86_86049


namespace sum_of_ages_l86_86056

variables (K T1 T2 : ℕ)

theorem sum_of_ages (h1 : K * T1 * T2 = 72) (h2 : T1 = T2) (h3 : T1 < K) : K + T1 + T2 = 14 :=
sorry

end sum_of_ages_l86_86056


namespace complement_A_U_l86_86386

-- Define the universal set U and set A as given in the problem.
def U : Set ℕ := { x | x ≥ 3 }
def A : Set ℕ := { x | x * x ≥ 10 }

-- Prove that the complement of A with respect to U is {3}.
theorem complement_A_U :
  (U \ A) = {3} :=
by
  sorry

end complement_A_U_l86_86386


namespace family_ages_l86_86583

theorem family_ages :
  ∃ (x j b m F M : ℕ), 
    (b = j - x) ∧
    (m = j - 2 * x) ∧
    (j * b = F) ∧
    (b * m = M) ∧
    (j + b + m + F + M = 90) ∧
    (F = M + x ∨ F = M - x) ∧
    (j = 6) ∧ 
    (b = 6) ∧ 
    (m = 6) ∧ 
    (F = 36) ∧ 
    (M = 36) :=
sorry

end family_ages_l86_86583


namespace work_completed_in_5_days_l86_86629

-- Define the rates of work for A, B, and C
def rateA : ℚ := 1 / 15
def rateB : ℚ := 1 / 14
def rateC : ℚ := 1 / 16

-- Summing their rates to get the combined rate
def combined_rate : ℚ := rateA + rateB + rateC

-- This is the statement we need to prove, i.e., the time required for A, B, and C to finish the work together is 5 days.
theorem work_completed_in_5_days (hA : rateA = 1 / 15) (hB : rateB = 1 / 14) (hC : rateC = 1 / 16) :
  (1 / combined_rate) = 5 :=
by
  sorry

end work_completed_in_5_days_l86_86629


namespace ratio_x_y_l86_86250

theorem ratio_x_y (x y : ℚ) (h : (14 * x - 5 * y) / (17 * x - 3 * y) = 2 / 3) : x / y = 1 / 23 := by
  sorry

end ratio_x_y_l86_86250


namespace arithmetic_sequence_common_difference_l86_86073

theorem arithmetic_sequence_common_difference (a : ℕ → ℝ) (h : ∀ n, 4 * a (n + 1) - 4 * a n - 9 = 0) :
  ∃ d, (∀ n, a (n + 1) - a n = d) ∧ d = 9 / 4 := 
  sorry

end arithmetic_sequence_common_difference_l86_86073


namespace person_a_work_days_l86_86659

theorem person_a_work_days (x : ℝ) :
  (2 * (1 / x + 1 / 45) = 1 / 9) → (x = 30) :=
by
  sorry

end person_a_work_days_l86_86659


namespace x4_y4_value_l86_86940

theorem x4_y4_value (x y : ℝ) (h1 : x^4 + x^2 = 3) (h2 : y^4 - y^2 = 3) : x^4 + y^4 = 7 := by
  sorry

end x4_y4_value_l86_86940


namespace problem8x_eq_5_200timesreciprocal_l86_86947

theorem problem8x_eq_5_200timesreciprocal (x : ℚ) (h : 8 * x = 5) : 200 * (1 / x) = 320 := 
by 
  sorry

end problem8x_eq_5_200timesreciprocal_l86_86947


namespace measured_percentage_weight_loss_l86_86208

variable (W : ℝ) -- W is the starting weight.
variable (weight_loss_percent : ℝ := 0.12) -- 12% weight loss.
variable (clothes_weight_percent : ℝ := 0.03) -- 3% clothes weight addition.
variable (beverage_weight_percent : ℝ := 0.005) -- 0.5% beverage weight addition.

theorem measured_percentage_weight_loss : 
  (W - ((0.88 * W) + (clothes_weight_percent * W) + (beverage_weight_percent * W))) / W * 100 = 8.5 :=
by
  sorry

end measured_percentage_weight_loss_l86_86208


namespace richard_more_pins_than_patrick_l86_86951

theorem richard_more_pins_than_patrick :
  ∀ (R P R2 P2 : ℕ), 
    P = 70 → 
    R > P →
    P2 = 2 * R →
    R2 = P2 - 3 → 
    (R + R2) = (P + P2) + 12 → 
    R = 70 + 15 := 
by 
  intros R P R2 P2 hP hRp hP2 hR2 hTotal
  sorry

end richard_more_pins_than_patrick_l86_86951


namespace inequality_solution_set_l86_86860

theorem inequality_solution_set :
  {x : ℝ | (1/2 - x) * (x - 1/3) > 0} = {x : ℝ | 1/3 < x ∧ x < 1/2} := 
sorry

end inequality_solution_set_l86_86860


namespace scientific_notation_eight_million_l86_86212

theorem scientific_notation_eight_million :
  ∃ a n, 8000000 = a * 10 ^ n ∧ 1 ≤ a ∧ a < 10 ∧ a = 8 ∧ n = 6 :=
by
  use 8
  use 6
  sorry

end scientific_notation_eight_million_l86_86212


namespace weeks_per_month_l86_86771

-- Define the given conditions
def num_employees_initial : Nat := 500
def additional_employees : Nat := 200
def hourly_wage : Nat := 12
def daily_work_hours : Nat := 10
def weekly_work_days : Nat := 5
def total_monthly_pay : Nat := 1680000

-- Calculate the total number of employees after hiring
def total_employees : Nat := num_employees_initial + additional_employees

-- Calculate the pay rates
def daily_pay_per_employee : Nat := hourly_wage * daily_work_hours
def weekly_pay_per_employee : Nat := daily_pay_per_employee * weekly_work_days

-- Calculate the total weekly pay for all employees
def total_weekly_pay : Nat := weekly_pay_per_employee * total_employees

-- Define the statement to be proved
theorem weeks_per_month
  (h1 : total_employees = num_employees_initial + additional_employees)
  (h2 : daily_pay_per_employee = hourly_wage * daily_work_hours)
  (h3 : weekly_pay_per_employee = daily_pay_per_employee * weekly_work_days)
  (h4 : total_weekly_pay = weekly_pay_per_employee * total_employees)
  (h5 : total_monthly_pay = 1680000) :
  total_monthly_pay / total_weekly_pay = 4 :=
by sorry

end weeks_per_month_l86_86771


namespace calc_delta_l86_86244

noncomputable def delta (a b : ℝ) : ℝ :=
  (a^2 + b^2) / (1 + a * b)

-- Definition of the main problem as a Lean 4 statement
theorem calc_delta (h1 : 2 > 0) (h2 : 3 > 0) (h3 : 4 > 0) :
  delta (delta 2 3) 4 = 6661 / 2891 :=
by
  sorry

end calc_delta_l86_86244


namespace solution_l86_86914

def is_prime (n : ℕ) : Prop := ∀ (m : ℕ), m ∣ n → m = 1 ∨ m = n

noncomputable def find_pairs : Prop :=
  ∃ a b : ℕ, a ≠ b ∧ a > 0 ∧ b > 0 ∧ is_prime (a * b^2 / (a + b)) ∧ ((a = 6 ∧ b = 2) ∨ (a = 2 ∧ b = 6))

theorem solution :
  find_pairs := sorry

end solution_l86_86914


namespace even_function_a_value_l86_86750

def f (x a : ℝ) : ℝ := (x + 1) * (x - a)

theorem even_function_a_value (a : ℝ) (h : ∀ x : ℝ, f x a = f (-x) a) : a = 1 := by
  sorry

end even_function_a_value_l86_86750


namespace selection_methods_l86_86936

-- Conditions
def volunteers : ℕ := 5
def friday_slots : ℕ := 1
def saturday_slots : ℕ := 2
def sunday_slots : ℕ := 1

-- Function to calculate combinatorial n choose k
def choose (n k : ℕ) : ℕ :=
(n.factorial) / ((k.factorial) * ((n - k).factorial))

-- Function to calculate permutations of n P k
def perm (n k : ℕ) : ℕ :=
(n.factorial) / ((n - k).factorial)

-- The target proposition
theorem selection_methods : choose volunteers saturday_slots * perm (volunteers - saturday_slots) (friday_slots + sunday_slots) = 60 :=
by
  -- assumption here leads to the property required, usually this would be more detailed computation.
  sorry

end selection_methods_l86_86936


namespace find_complex_number_purely_imaginary_l86_86660

theorem find_complex_number_purely_imaginary :
  ∃ z : ℂ, (∃ b : ℝ, b ≠ 0 ∧ z = 1 + b * I) ∧ (∀ a b : ℝ, z = a + b * I → a^2 - b^2 + 3 = 0) :=
by
  -- Proof will go here
  sorry

end find_complex_number_purely_imaginary_l86_86660


namespace perpendicular_lines_l86_86615

theorem perpendicular_lines (a : ℝ) : 
  (3 * y + x + 4 = 0) → 
  (4 * y + a * x + 5 = 0) → 
  (∀ x y, x ≠ 0 ∧ y ≠ 0 → - (1 / 3 : ℝ) * - (a / 4 : ℝ) = -1) → 
  a = -12 := 
by
  intros h1 h2 h_perpendicularity
  sorry

end perpendicular_lines_l86_86615


namespace percentage_temporary_workers_l86_86105

-- Definitions based on the given conditions
def total_workers : ℕ := 100
def percentage_technicians : ℝ := 0.9
def percentage_non_technicians : ℝ := 0.1
def percentage_permanent_technicians : ℝ := 0.9
def percentage_permanent_non_technicians : ℝ := 0.1

-- Statement to prove that the percentage of temporary workers is 18%
theorem percentage_temporary_workers :
  100 * (1 - (percentage_permanent_technicians * percentage_technicians +
              percentage_permanent_non_technicians * percentage_non_technicians)) = 18 :=
by sorry

end percentage_temporary_workers_l86_86105


namespace certain_event_idiom_l86_86391

theorem certain_event_idiom : 
  ∃ (idiom : String), idiom = "Catching a turtle in a jar" ∧ 
  ∀ (option : String), 
    option = "Catching a turtle in a jar" ∨ 
    option = "Carving a boat to find a sword" ∨ 
    option = "Waiting by a tree stump for a rabbit" ∨ 
    option = "Fishing for the moon in the water" → 
    (option = idiom ↔ (option = "Catching a turtle in a jar")) := 
by
  sorry

end certain_event_idiom_l86_86391


namespace match_Tile_C_to_Rectangle_III_l86_86127

-- Define the structure for a Tile
structure Tile where
  top : ℕ
  right : ℕ
  bottom : ℕ
  left : ℕ

-- Define the given tiles
def Tile_A : Tile := { top := 5, right := 3, bottom := 7, left := 2 }
def Tile_B : Tile := { top := 3, right := 6, bottom := 2, left := 8 }
def Tile_C : Tile := { top := 7, right := 9, bottom := 1, left := 3 }
def Tile_D : Tile := { top := 1, right := 8, bottom := 5, left := 9 }

-- The proof problem: Prove that Tile C should be matched to Rectangle III
theorem match_Tile_C_to_Rectangle_III : (Tile_C = { top := 7, right := 9, bottom := 1, left := 3 }) → true := 
by
  intros
  sorry

end match_Tile_C_to_Rectangle_III_l86_86127


namespace find_constant_l86_86155

theorem find_constant (N : ℝ) (C : ℝ) (h1 : N = 12.0) (h2 : C + 0.6667 * N = 0.75 * N) : C = 0.9996 :=
by
  sorry

end find_constant_l86_86155


namespace picnic_weather_condition_l86_86701

variables (P Q : Prop)

theorem picnic_weather_condition (h : ¬P → ¬Q) : Q → P := 
by sorry

end picnic_weather_condition_l86_86701


namespace difference_of_sums_l86_86745

def even_numbers_sum (n : ℕ) : ℕ := (n * (n + 1))
def odd_numbers_sum (n : ℕ) : ℕ := n^2

theorem difference_of_sums : 
  even_numbers_sum 3003 - odd_numbers_sum 3003 = 7999 := 
by {
  sorry 
}

end difference_of_sums_l86_86745


namespace average_age_of_women_l86_86572

theorem average_age_of_women (A : ℕ) (W1 W2 : ℕ) 
  (h1 : 7 * A - 26 - 30 + W1 + W2 = 7 * (A + 4)) : 
  (W1 + W2) / 2 = 42 := 
by 
  sorry

end average_age_of_women_l86_86572


namespace michael_total_fish_l86_86921

-- Definitions based on conditions
def michael_original_fish : ℕ := 31
def ben_fish_given : ℕ := 18

-- Theorem to prove the total number of fish Michael has now
theorem michael_total_fish : (michael_original_fish + ben_fish_given) = 49 :=
by sorry

end michael_total_fish_l86_86921


namespace Ryan_has_28_marbles_l86_86398

theorem Ryan_has_28_marbles :
  ∃ R : ℕ, (12 + R) - (1/4 * (12 + R)) * 2 = 20 ∧ R = 28 :=
by
  sorry

end Ryan_has_28_marbles_l86_86398


namespace find_y_l86_86859

theorem find_y (x y : ℝ) (h1 : x + 2 * y = 12) (h2 : x = 6) : y = 3 :=
by
  sorry

end find_y_l86_86859


namespace maxValue_is_6084_over_17_l86_86121

open Real

noncomputable def maxValue (x y : ℝ) (h : x + y = 5) : ℝ :=
  x^4 * y + x^3 * y + x^2 * y + x * y + x * y^2 + x * y^3 + x * y^4

theorem maxValue_is_6084_over_17 (x y : ℝ) (h : x + y = 5) :
  maxValue x y h ≤ 6084 / 17 := 
sorry

end maxValue_is_6084_over_17_l86_86121


namespace initial_sugar_amount_l86_86269

-- Definitions based on the conditions
def packs : ℕ := 12
def weight_per_pack : ℕ := 250
def leftover_sugar : ℕ := 20

-- Theorem statement
theorem initial_sugar_amount : packs * weight_per_pack + leftover_sugar = 3020 :=
by
  sorry

end initial_sugar_amount_l86_86269


namespace average_cost_per_trip_is_correct_l86_86763

def oldest_pass_cost : ℕ := 100
def second_oldest_pass_cost : ℕ := 90
def third_oldest_pass_cost : ℕ := 80
def youngest_pass_cost : ℕ := 70

def oldest_trips : ℕ := 35
def second_oldest_trips : ℕ := 25
def third_oldest_trips : ℕ := 20
def youngest_trips : ℕ := 15

def total_cost : ℕ := oldest_pass_cost + second_oldest_pass_cost + third_oldest_pass_cost + youngest_pass_cost
def total_trips : ℕ := oldest_trips + second_oldest_trips + third_oldest_trips + youngest_trips

def average_cost_per_trip : ℚ := total_cost / total_trips

theorem average_cost_per_trip_is_correct : average_cost_per_trip = 340 / 95 :=
by sorry

end average_cost_per_trip_is_correct_l86_86763


namespace symmetric_abs_necessary_not_sufficient_l86_86637

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def y_axis_symmetric (f : ℝ → ℝ) : Prop :=
  ∀ x, |f (-x)| = |f x|

theorem symmetric_abs_necessary_not_sufficient (f : ℝ → ℝ) :
  is_odd_function f → y_axis_symmetric f := sorry

end symmetric_abs_necessary_not_sufficient_l86_86637


namespace solve_x_squared_solve_x_cubed_l86_86281

-- Define the first problem with its condition and prove the possible solutions
theorem solve_x_squared {x : ℝ} (h : (x + 1)^2 = 9) : x = 2 ∨ x = -4 :=
sorry

-- Define the second problem with its condition and prove the possible solution
theorem solve_x_cubed {x : ℝ} (h : -2 * (x^3 - 1) = 18) : x = -2 :=
sorry

end solve_x_squared_solve_x_cubed_l86_86281


namespace total_students_l86_86000

theorem total_students (m n : ℕ) (hm : m ≥ 3) (hn : n ≥ 3) (hshake : (2 * m * n - m - n) = 252) : m * n = 72 :=
  sorry

end total_students_l86_86000


namespace condition_is_necessary_but_not_sufficient_l86_86443

noncomputable def sequence_satisfies_condition (a : ℕ → ℤ) : Prop :=
  a 3 + a 7 = 2 * a 5

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a n = a 1 + (n - 1) * d

theorem condition_is_necessary_but_not_sufficient (a : ℕ → ℤ) :
  (sequence_satisfies_condition a ∧ (¬ arithmetic_sequence a)) ∨
  (arithmetic_sequence a → sequence_satisfies_condition a) :=
sorry

end condition_is_necessary_but_not_sufficient_l86_86443


namespace slower_speed_l86_86405

theorem slower_speed (x : ℝ) (h_walk_faster : 12 * (100 / x) - 100 = 20) : x = 10 :=
by sorry

end slower_speed_l86_86405


namespace initial_distance_proof_l86_86122

noncomputable def initial_distance (V_A V_B T : ℝ) : ℝ :=
  (V_A * T) + (V_B * T)

theorem initial_distance_proof 
  (V_A V_B : ℝ) 
  (T : ℝ) 
  (h1 : V_A / V_B = 5 / 6)
  (h2 : V_B = 90)
  (h3 : T = 8 / 15) :
  initial_distance V_A V_B T = 88 := 
by
  -- proof goes here
  sorry

end initial_distance_proof_l86_86122


namespace proof_l_squared_l86_86603

noncomputable def longest_line_segment (diameter : ℝ) (sectors : ℕ) : ℝ :=
  let R := diameter / 2
  let theta := (2 * Real.pi) / sectors
  2 * R * (Real.sin (theta / 2))

theorem proof_l_squared :
  let diameter := 18
  let sectors := 4
  let l := longest_line_segment diameter sectors
  l^2 = 162 := by
  let diameter := 18
  let sectors := 4
  let l := longest_line_segment diameter sectors
  have h : l^2 = 162 := sorry
  exact h

end proof_l_squared_l86_86603


namespace b_2023_equals_one_fifth_l86_86820

theorem b_2023_equals_one_fifth (b : ℕ → ℚ) (h1 : b 1 = 4) (h2 : b 2 = 5)
    (h_rec : ∀ (n : ℕ), n ≥ 3 → b n = b (n - 1) / b (n - 2)) :
    b 2023 = 1 / 5 := by
  sorry

end b_2023_equals_one_fifth_l86_86820


namespace find_abc_of_N_l86_86868

theorem find_abc_of_N :
  ∃ N : ℕ, (N % 10000) = (N + 2) % 10000 ∧ 
            (N % 16 = 15 ∧ (N + 2) % 16 = 1) ∧ 
            ∃ abc : ℕ, (100 ≤ abc ∧ abc < 1000) ∧ 
            (N % 1000) = 100 * abc + 99 := sorry

end find_abc_of_N_l86_86868


namespace joan_books_l86_86219

theorem joan_books (initial_books sold_books result_books : ℕ) 
  (h_initial : initial_books = 33) 
  (h_sold : sold_books = 26) 
  (h_result : initial_books - sold_books = result_books) : 
  result_books = 7 := 
by
  sorry

end joan_books_l86_86219


namespace wire_cut_l86_86068

theorem wire_cut (x : ℝ) (h1 : x + (100 - x) = 100) (h2 : x = (7/13) * (100 - x)) : x = 35 :=
sorry

end wire_cut_l86_86068


namespace balls_in_boxes_l86_86611

theorem balls_in_boxes : 
  let balls := 5
  let boxes := 4
  (∀ (ball : Fin balls), Fin boxes) → (4^5 = 1024) := 
by
  intro h
  sorry

end balls_in_boxes_l86_86611


namespace line_through_intersection_and_origin_l86_86870

theorem line_through_intersection_and_origin :
  ∃ (x y : ℝ), (2*x + y = 3) ∧ (x + 4*y = 2) ∧ (x - 10*y = 0) :=
by
  sorry

end line_through_intersection_and_origin_l86_86870


namespace tablets_of_medicine_A_l86_86317

-- Given conditions as definitions
def B_tablets : ℕ := 16

def min_extracted_tablets : ℕ := 18

-- Question and expected answer encapsulated in proof statement
theorem tablets_of_medicine_A (A_tablets : ℕ) (h : A_tablets + B_tablets - 2 >= min_extracted_tablets) : A_tablets = 3 :=
sorry

end tablets_of_medicine_A_l86_86317


namespace min_value_x_l86_86539

theorem min_value_x (a b x : ℝ) (ha : 0 < a) (hb : 0 < b)
(hcond : 4 * a + b * (1 - a) = 0)
(hineq : ∀ a b, 0 < a → 0 < b → 4 * a + b * (1 - a) = 0 → (1 / (a ^ 2) + 16 / (b ^ 2) ≥ 1 + x / 2 - x ^ 2)) :
  x = 1 :=
sorry

end min_value_x_l86_86539


namespace sum_of_operations_l86_86004

noncomputable def triangle (a b c : ℕ) : ℕ :=
  a + 2 * b - c

theorem sum_of_operations :
  triangle 3 5 7 + triangle 6 1 8 = 6 :=
by
  sorry

end sum_of_operations_l86_86004


namespace complement_intersection_empty_l86_86344

open Set

-- Given definitions and conditions
def U : Set ℕ := {1, 2, 3}
def A : Set ℕ := {1, 2}
def B : Set ℕ := {1, 3}

-- Complement operation with respect to U
def C_U (X : Set ℕ) : Set ℕ := U \ X

-- The proof statement to be shown
theorem complement_intersection_empty :
  (C_U A ∩ C_U B) = ∅ := by sorry

end complement_intersection_empty_l86_86344


namespace age_intervals_l86_86632

theorem age_intervals (A1 A2 A3 A4 A5 : ℝ) (x : ℝ) (h1 : A1 = 7)
  (h2 : A2 = A1 + x) (h3 : A3 = A1 + 2 * x) (h4 : A4 = A1 + 3 * x) (h5 : A5 = A1 + 4 * x)
  (sum_ages : A1 + A2 + A3 + A4 + A5 = 65) :
  x = 3.7 :=
by
  -- Sketch a proof or leave 'sorry' for completeness
  sorry

end age_intervals_l86_86632


namespace dogwood_tree_count_l86_86899

theorem dogwood_tree_count (n d1 d2 d3 d4 d5: ℕ) 
  (h1: n = 39)
  (h2: d1 = 24)
  (h3: d2 = d1 / 2)
  (h4: d3 = 4 * d2)
  (h5: d4 = 5)
  (h6: d5 = 15):
  n + d1 + d2 + d3 + d4 + d5 = 143 :=
by
  sorry

end dogwood_tree_count_l86_86899


namespace car_speed_l86_86693

theorem car_speed (v : ℝ) (h1 : 1 / 900 * 3600 = 4) (h2 : 1 / v * 3600 = 6) : v = 600 :=
by
  sorry

end car_speed_l86_86693


namespace value_computation_l86_86890

theorem value_computation (N : ℝ) (h1 : 1.20 * N = 2400) : 0.20 * N = 400 := 
by
  sorry

end value_computation_l86_86890


namespace nearest_integer_to_expression_correct_l86_86852

noncomputable def nearest_integer_to_expression : ℤ :=
  Int.floor ((3 + Real.sqrt 2) ^ 6)

theorem nearest_integer_to_expression_correct : nearest_integer_to_expression = 7414 :=
by
  sorry

end nearest_integer_to_expression_correct_l86_86852


namespace swim_back_distance_l86_86801

variables (swimming_speed_still_water : ℝ) (water_speed : ℝ) (time_back : ℝ) (distance_back : ℝ)

theorem swim_back_distance :
  swimming_speed_still_water = 12 → 
  water_speed = 10 → 
  time_back = 4 →
  distance_back = (swimming_speed_still_water - water_speed) * time_back →
  distance_back = 8 :=
by
  intros swimming_speed_still_water_eq water_speed_eq time_back_eq distance_back_eq
  have swim_speed : (swimming_speed_still_water - water_speed) = 2 := by sorry
  rw [swim_speed, time_back_eq] at distance_back_eq
  sorry

end swim_back_distance_l86_86801


namespace cos_alpha_plus_pi_over_2_l86_86184

theorem cos_alpha_plus_pi_over_2 (α : ℝ) (h : Real.sin α = 1/3) : 
    Real.cos (α + Real.pi / 2) = -(1/3) :=
by
  sorry

end cos_alpha_plus_pi_over_2_l86_86184


namespace cone_volume_l86_86613

theorem cone_volume (l : ℝ) (S_side : ℝ) (h r V : ℝ)
  (hl : l = 10)
  (hS : S_side = 60 * Real.pi)
  (hr : S_side = π * r * l)
  (hh : h = Real.sqrt (l^2 - r^2))
  (hV : V = (1/3) * π * r^2 * h) :
  V = 96 * Real.pi := 
sorry

end cone_volume_l86_86613


namespace x_squared_minus_y_squared_l86_86502

-- Define the given conditions as Lean definitions
def x_plus_y : ℚ := 8 / 15
def x_minus_y : ℚ := 1 / 45

-- State the proof problem in Lean 4
theorem x_squared_minus_y_squared : (x_plus_y * x_minus_y = 8 / 675) := 
by
  sorry

end x_squared_minus_y_squared_l86_86502


namespace m_over_n_eq_l86_86302

variables (m n : ℝ)
variables (x y x1 y1 x2 y2 x0 y0 : ℝ)

-- Ellipse equation
axiom ellipse_eq : m * x^2 + n * y^2 = 1

-- Line equation
axiom line_eq : x + y = 1

-- Points M and N on the ellipse
axiom M_point : m * x1^2 + n * y1^2 = 1
axiom N_point : m * x2^2 + n * y2^2 = 1

-- Midpoint of MN is P
axiom P_midpoint : x0 = (x1 + x2) / 2 ∧ y0 = (y1 + y2) / 2

-- Slope of OP
axiom slope_OP : y0 / x0 = (Real.sqrt 2) / 2

theorem m_over_n_eq : m / n = (Real.sqrt 2) / 2 :=
sorry

end m_over_n_eq_l86_86302


namespace prove_ax5_by5_l86_86856

variables {a b x y : ℝ}

theorem prove_ax5_by5 (h1 : a * x + b * y = 5)
                      (h2 : a * x^2 + b * y^2 = 11)
                      (h3 : a * x^3 + b * y^3 = 30)
                      (h4 : a * x^4 + b * y^4 = 85) :
  a * x^5 + b * y^5 = 7025 / 29 :=
sorry

end prove_ax5_by5_l86_86856


namespace find_x_l86_86452

theorem find_x (x : ℕ) : (x % 9 = 0) ∧ (x^2 > 144) ∧ (x < 30) → (x = 18 ∨ x = 27) :=
by 
  sorry

end find_x_l86_86452


namespace chords_from_nine_points_l86_86526

theorem chords_from_nine_points (n : ℕ) (h : n = 9) : (n * (n - 1)) / 2 = 36 := by
  sorry

end chords_from_nine_points_l86_86526


namespace compare_a_b_l86_86596

def a := 1 / 3 + 1 / 4
def b := 1 / 5 + 1 / 6 + 1 / 7

theorem compare_a_b : a > b := 
  sorry

end compare_a_b_l86_86596


namespace village_population_500_l86_86361

variable (n : ℝ) -- Define the variable for population increase
variable (initial_population : ℝ) -- Define the variable for the initial population

-- Conditions from the problem
def first_year_increase : Prop := initial_population * (3 : ℝ) = n
def initial_population_def : Prop := initial_population = n / 3
def second_year_increase_def := ((n / 3 + n) * (n / 100 )) = 300

-- Define the final population formula
def population_after_two_years : ℝ := (initial_population + n + 300)

theorem village_population_500 (n : ℝ) (initial_population: ℝ) :
  first_year_increase n initial_population →
  initial_population_def n initial_population →
  second_year_increase_def n →
  population_after_two_years n initial_population = 500 :=
by sorry

#check village_population_500

end village_population_500_l86_86361


namespace vacation_books_pair_count_l86_86091

/-- 
Given three distinct mystery novels, three distinct fantasy novels, and three distinct biographies,
we want to prove that the number of possible pairs of books of different genres is 27.
-/

theorem vacation_books_pair_count :
  let mystery_books := 3
  let fantasy_books := 3
  let biography_books := 3
  let total_books := mystery_books + fantasy_books + biography_books
  let pairs := (total_books * (total_books - 3)) / 2
  pairs = 27 := 
by
  sorry

end vacation_books_pair_count_l86_86091


namespace Cooper_age_l86_86891

variable (X : ℕ)
variable (Dante : ℕ)
variable (Maria : ℕ)

theorem Cooper_age (h1 : Dante = 2 * X) (h2 : Maria = 2 * X + 1) (h3 : X + Dante + Maria = 31) : X = 6 :=
by
  -- Proof is omitted as indicated
  sorry

end Cooper_age_l86_86891


namespace simplify_expression_l86_86397

theorem simplify_expression (a : ℝ) (h₁ : a ≠ 1) (h₂ : a ≠ 1 / 2) :
    1 - 1 / (1 - a / (1 - a)) = -a / (1 - 2 * a) := by
  sorry

end simplify_expression_l86_86397


namespace set_intersection_complement_l86_86853

def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8}
def A : Set ℕ := {2, 5, 8}
def B : Set ℕ := {1, 3, 5, 7}

theorem set_intersection_complement :
  ((U \ A) ∩ B) = {1, 3, 7} :=
by
  sorry

end set_intersection_complement_l86_86853


namespace probability_remainder_is_4_5_l86_86158

def probability_remainder_1 (N : ℕ) : Prop :=
  N ≥ 1 ∧ N ≤ 2020 → (N^16 % 5 = 1)

theorem probability_remainder_is_4_5 : 
  ∀ N, N ≥ 1 ∧ N ≤ 2020 → (N^16 % 5 = 1) → (number_of_successful_outcomes / total_outcomes = 4 / 5) :=
sorry

end probability_remainder_is_4_5_l86_86158


namespace katie_earnings_l86_86974

def bead_necklaces : Nat := 4
def gemstone_necklaces : Nat := 3
def cost_per_necklace : Nat := 3

theorem katie_earnings : bead_necklaces + gemstone_necklaces * cost_per_necklace = 21 := 
by
  sorry

end katie_earnings_l86_86974


namespace large_integer_value_l86_86633

theorem large_integer_value :
  (2 + 3) * (2^2 + 3^2) * (2^4 - 3^4) * (2^8 + 3^8) * (2^16 - 3^16) * (2^32 + 3^32) * (2^64 - 3^64)
  > 0 := 
by
  sorry

end large_integer_value_l86_86633


namespace coeff_x4_in_expansion_l86_86595

open Nat

noncomputable def binom (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

noncomputable def coefficient_x4_term : ℕ := binom 9 4

noncomputable def constant_term : ℕ := 243 * 4

theorem coeff_x4_in_expansion : coefficient_x4_term * 972 * Real.sqrt 2 = 122472 * Real.sqrt 2 :=
by
  sorry

end coeff_x4_in_expansion_l86_86595


namespace largest_possible_perimeter_l86_86784

theorem largest_possible_perimeter (x : ℕ) (h1 : 1 < x) (h2 : x < 11) : 
    5 + 6 + x ≤ 21 := 
  sorry

end largest_possible_perimeter_l86_86784


namespace range_of_a_l86_86810

theorem range_of_a (a : ℝ) :
  (∀ x, (3 ≤ x → 2*a*x + 4 ≤ 2*a*(x+1) + 4) ∧ (2 < x ∧ x < 3 → (a + (2*a + 2)/(x-2) ≤ a + (2*a + 2)/(x-1))) ) →
  -1 < a ∧ a ≤ -2/3 :=
by
  intros h
  sorry

end range_of_a_l86_86810


namespace revenue_from_full_price_tickets_l86_86898

theorem revenue_from_full_price_tickets (f h p : ℕ) 
    (h1 : f + h = 160) 
    (h2 : f * p + h * (p / 2) = 2400) 
    (h3 : h = 160 - f)
    (h4 : 2 * 2400 = 4800) :
  f * p = 800 := 
sorry

end revenue_from_full_price_tickets_l86_86898


namespace max_A_k_value_l86_86880

noncomputable def A_k (k : ℕ) : ℝ := (19^k + 66^k) / k.factorial

theorem max_A_k_value : 
  ∃ k : ℕ, (∀ m : ℕ, (A_k m ≤ A_k k)) ∧ k = 65 :=
by
  sorry

end max_A_k_value_l86_86880


namespace coterminal_angle_l86_86806

theorem coterminal_angle (α : ℤ) : 
  ∃ k : ℤ, α = k * 360 + 283 ↔ ∃ k : ℤ, α = k * 360 - 437 :=
sorry

end coterminal_angle_l86_86806


namespace geometric_sum_S6_l86_86101

open Real

-- Define a geometric sequence
noncomputable def geometric_sequence (a : ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  a * q ^ (n - 1)

-- Define the sum of the first n terms of a geometric sequence
noncomputable def sum_geometric (a : ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  if q = 1 then a * n else a * (1 - q ^ n) / (1 - q)

-- Given conditions
variables (a q : ℝ) (n : ℕ)
variable (S3 : ℝ)
variable (q : ℝ) (h_q : q = 2)
variable (h_S3 : S3 = 7)

theorem geometric_sum_S6 :
  sum_geometric a 2 6 = 63 :=
  by
    sorry

end geometric_sum_S6_l86_86101


namespace total_time_of_flight_l86_86915

variables {V_0 g t t_1 H : ℝ}  -- Define variables

-- Define conditions
def initial_condition (V_0 g t_1 H : ℝ) : Prop :=
H = (1/2) * g * t_1^2

def return_condition (V_0 g t : ℝ) : Prop :=
t = 2 * (V_0 / g)

theorem total_time_of_flight
  (V_0 g : ℝ)
  (h1 : initial_condition V_0 g (V_0 / g) (1/2 * g * (V_0 / g)^2))
  : return_condition V_0 g (2 * V_0 / g) :=
by
  sorry

end total_time_of_flight_l86_86915


namespace max_full_marks_probability_l86_86501

-- Define the total number of mock exams
def total_mock_exams : ℕ := 20
-- Define the number of full marks scored in mock exams
def full_marks_in_mocks : ℕ := 8

-- Define the probability of event A (scoring full marks in the first test)
def P_A : ℚ := full_marks_in_mocks / total_mock_exams

-- Define the probability of not scoring full marks in the first test
def P_neg_A : ℚ := 1 - P_A

-- Define the probability of event B (scoring full marks in the second test)
def P_B : ℚ := 1 / 2

-- Define the maximum probability of scoring full marks in either the first or the second test
def max_probability : ℚ := P_A + P_neg_A * P_B

-- The main theorem conjecture
theorem max_full_marks_probability :
  max_probability = 7 / 10 :=
by
  -- Inserting placeholder to skip the proof for now
  sorry

end max_full_marks_probability_l86_86501


namespace find_equation_of_tangent_line_l86_86926

def is_tangent_at_point (l : ℝ → ℝ → Prop) (x₀ y₀ : ℝ) := 
  ∃ x y, (x - 1)^2 + (y + 2)^2 = 1 ∧ l x₀ y₀ ∧ l x y

def equation_of_line (l : ℝ → ℝ → Prop) := 
  ∀ x y, l x y ↔ (x = 2 ∨ 12 * x - 5 * y - 9 = 0)

theorem find_equation_of_tangent_line : 
  ∀ (l : ℝ → ℝ → Prop),
  (∀ x y, l x y ↔ (x - 1)^2 + (y + 2)^2 ≠ 1 ∧ (x, y) = (2,3))
  → is_tangent_at_point l 2 3
  → equation_of_line l := 
sorry

end find_equation_of_tangent_line_l86_86926


namespace solution_set_of_inequality_l86_86021

open Set

theorem solution_set_of_inequality :
  {x : ℝ | (x ≠ -2) ∧ (x ≠ -8) ∧ (2 / (x + 2) + 4 / (x + 8) ≥ 4 / 5)} =
  {x : ℝ | (-8 < x ∧ x < -2) ∨ (-2 < x ∧ x ≤ 4)} :=
by
  sorry

end solution_set_of_inequality_l86_86021


namespace find_t_l86_86255

variables {a b c r s t : ℝ}

theorem find_t (h1 : a + b + c = -3)
             (h2 : a * b + b * c + c * a = 4)
             (h3 : a * b * c = -1)
             (h4 : ∀ x, x^3 + 3*x^2 + 4*x + 1 = 0 → (x = a ∨ x = b ∨ x = c))
             (h5 : ∀ y, y^3 + r*y^2 + s*y + t = 0 → (y = a + b ∨ y = b + c ∨ y = c + a))
             : t = 11 :=
sorry

end find_t_l86_86255


namespace overall_average_score_l86_86636

variables (average_male average_female sum_male sum_female total_sum : ℕ)
variables (count_male count_female total_count : ℕ)

def average_score (sum : ℕ) (count : ℕ) : ℕ := sum / count

theorem overall_average_score
  (average_male : ℕ := 84)
  (count_male : ℕ := 8)
  (average_female : ℕ := 92)
  (count_female : ℕ := 24)
  (sum_male : ℕ := count_male * average_male)
  (sum_female : ℕ := count_female * average_female)
  (total_sum : ℕ := sum_male + sum_female)
  (total_count : ℕ := count_male + count_female) :
  average_score total_sum total_count = 90 := 
sorry

end overall_average_score_l86_86636


namespace min_value_expr_l86_86139

theorem min_value_expr (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a / b) + (b / c) + (c / a) + (a / c) ≥ 4 := 
sorry

end min_value_expr_l86_86139


namespace sum_S_17_33_50_l86_86070

def S (n : ℕ) : ℤ :=
  if n % 2 = 0 then - (n / 2)
  else (n / 2) + 1

theorem sum_S_17_33_50 : (S 17) + (S 33) + (S 50) = 1 := by
  sorry

end sum_S_17_33_50_l86_86070


namespace rankings_are_correct_l86_86149

-- Define teams:
inductive Team
| A | B | C | D

-- Define the type for ranking
structure Ranking :=
  (first : Team)
  (second : Team)
  (third : Team)
  (last : Team)

-- Define the predictions of Jia, Yi, and Bing
structure Predictions := 
  (Jia : Ranking)
  (Yi : Ranking)
  (Bing : Ranking)

-- Define the condition that each prediction is half right, half wrong
def isHalfRightHalfWrong (pred : Ranking) (actual : Ranking) : Prop :=
  (pred.first = actual.first ∨ pred.second = actual.second ∨ pred.third = actual.third ∨ pred.last = actual.last) ∧
  (pred.first ≠ actual.first ∨ pred.second ≠ actual.second ∨ pred.third ≠ actual.third ∨ pred.last ≠ actual.last)

-- Define the actual rankings
def actualRanking : Ranking := { first := Team.C, second := Team.A, third := Team.D, last := Team.B }

-- Define Jia's Predictions 
def JiaPrediction : Ranking := { first := Team.C, second := Team.C, third := Team.D, last := Team.D }

-- Define Yi's Predictions 
def YiPrediction : Ranking := { first := Team.B, second := Team.A, third := Team.C, last := Team.D }

-- Define Bing's Predictions 
def BingPrediction : Ranking := { first := Team.C, second := Team.B, third := Team.A, last := Team.D }

-- Create an instance of predictions
def pred : Predictions := { Jia := JiaPrediction, Yi := YiPrediction, Bing := BingPrediction }

-- The theorem to be proved
theorem rankings_are_correct :
  isHalfRightHalfWrong pred.Jia actualRanking ∧ 
  isHalfRightHalfWrong pred.Yi actualRanking ∧ 
  isHalfRightHalfWrong pred.Bing actualRanking →
  actualRanking.first = Team.C ∧ actualRanking.second = Team.A ∧ actualRanking.third = Team.D ∧ 
  actualRanking.last = Team.B :=
by
  sorry -- Proof is not required.

end rankings_are_correct_l86_86149


namespace taozi_is_faster_than_xiaoxiao_l86_86377

theorem taozi_is_faster_than_xiaoxiao : 
  let taozi_speed := 210
  let xiaoxiao_distance := 500
  let xiaoxiao_time := 3
  let xiaoxiao_speed := xiaoxiao_distance / xiaoxiao_time
  taozi_speed > xiaoxiao_speed
:= by
  let taozi_speed := 210
  let xiaoxiao_distance := 500
  let xiaoxiao_time := 3
  let xiaoxiao_speed := xiaoxiao_distance / xiaoxiao_time
  sorry

end taozi_is_faster_than_xiaoxiao_l86_86377


namespace mul_equiv_l86_86350

theorem mul_equiv :
  (213 : ℝ) * 16 = 3408 →
  (16 : ℝ) * 21.3 = 340.8 :=
by
  sorry

end mul_equiv_l86_86350


namespace additional_wolves_in_pack_l86_86765

-- Define the conditions
def wolves_out_hunting : ℕ := 4
def meat_per_wolf_per_day : ℕ := 8
def hunting_days : ℕ := 5
def meat_per_deer : ℕ := 200

-- Calculate total meat per wolf for hunting days
def meat_per_wolf_total : ℕ := meat_per_wolf_per_day * hunting_days

-- Calculate wolves fed per deer
def wolves_fed_per_deer : ℕ := meat_per_deer / meat_per_wolf_total

-- Calculate total deer killed by wolves out hunting
def total_deers_killed : ℕ := wolves_out_hunting

-- Calculate total meat provided by hunting wolves
def total_meat_provided : ℕ := total_deers_killed * meat_per_deer

-- Calculate number of wolves fed by total meat provided
def total_wolves_fed : ℕ := total_meat_provided / meat_per_wolf_total

-- Define the main theorem to prove the answer
theorem additional_wolves_in_pack (total_wolves_fed wolves_out_hunting : ℕ) : 
  total_wolves_fed - wolves_out_hunting = 16 :=
by
  sorry

end additional_wolves_in_pack_l86_86765


namespace factorize_expression_l86_86518

theorem factorize_expression (a x : ℝ) : a * x^2 - 2 * a * x + a = a * (x - 1)^2 :=
by sorry

end factorize_expression_l86_86518


namespace smallest_yellow_marbles_l86_86844

-- Definitions for given conditions
def total_marbles (n : ℕ): Prop := n > 0
def blue_marbles (n : ℕ) : ℕ := n / 4
def red_marbles (n : ℕ) : ℕ := n / 6
def green_marbles : ℕ := 7
def yellow_marbles (n : ℕ) : ℕ := n - (blue_marbles n + red_marbles n + green_marbles)

-- Lean statement that verifies the smallest number of yellow marbles is 0
theorem smallest_yellow_marbles (n : ℕ) (h : total_marbles n) : yellow_marbles n = 0 :=
  sorry

end smallest_yellow_marbles_l86_86844


namespace cannot_tile_surface_square_hexagon_l86_86691

-- Definitions of internal angles of the tile shapes
def internal_angle_triangle := 60
def internal_angle_square := 90
def internal_angle_hexagon := 120
def internal_angle_octagon := 135

-- The theorem to prove that square and hexagon cannot tile a surface without gaps or overlaps
theorem cannot_tile_surface_square_hexagon : ∀ (m n : ℕ), internal_angle_square * m + internal_angle_hexagon * n ≠ 360 := 
by sorry

end cannot_tile_surface_square_hexagon_l86_86691


namespace range_of_m_l86_86179

theorem range_of_m (m : ℝ) :
  (∀ x : ℝ, x^2 + (m + 2) * x + (m + 5) = 0 → 0 < x) → (-5 < m ∧ m ≤ -4) :=
by
  sorry

end range_of_m_l86_86179


namespace apple_price_33kg_l86_86710

theorem apple_price_33kg
  (l q : ℝ)
  (h1 : 10 * l = 3.62)
  (h2 : 30 * l + 6 * q = 12.48) :
  30 * l + 3 * q = 11.67 :=
by
  sorry

end apple_price_33kg_l86_86710


namespace amy_music_files_l86_86803

-- Define the number of total files on the flash drive
def files_on_flash_drive := 48.0

-- Define the number of video files on the flash drive
def video_files := 21.0

-- Define the number of picture files on the flash drive
def picture_files := 23.0

-- Define the number of music files, derived from the conditions
def music_files := files_on_flash_drive - (video_files + picture_files)

-- The theorem we need to prove
theorem amy_music_files : music_files = 4.0 := by
  sorry

end amy_music_files_l86_86803


namespace number_of_buses_l86_86655

-- Definitions based on the given conditions
def vans : ℕ := 6
def people_per_van : ℕ := 6
def people_per_bus : ℕ := 18
def total_people : ℕ := 180

-- Theorem to prove the number of buses
theorem number_of_buses : 
  ∃ buses : ℕ, buses = (total_people - (vans * people_per_van)) / people_per_bus ∧ buses = 8 :=
by
  sorry

end number_of_buses_l86_86655


namespace problems_per_page_l86_86041

theorem problems_per_page (total_problems finished_problems pages_left problems_per_page : ℕ)
  (h1 : total_problems = 40)
  (h2 : finished_problems = 26)
  (h3 : pages_left = 2)
  (h4 : total_problems - finished_problems = pages_left * problems_per_page) :
  problems_per_page = 7 :=
by
  sorry

end problems_per_page_l86_86041


namespace solution_set_of_inequality_l86_86364

theorem solution_set_of_inequality (x : ℝ) : 
  (3 * x - 4 > 2) → (x > 2) :=
by
  intro h
  sorry

end solution_set_of_inequality_l86_86364


namespace valid_digit_cancel_fractions_l86_86802

def digit_cancel_fraction (a b c d : ℕ) : Prop :=
  10 * a + b == 0 ∧ 10 * c + d == 0 ∧ 
  (b == d ∨ b == c ∨ a == d ∨ a == c) ∧
  (b ≠ a ∨ d ≠ c) ∧
  ((10 * a + b) ≠ (10 * c + d)) ∧
  ((10 * a + b) * d == (10 * c + d) * a)

theorem valid_digit_cancel_fractions :
  ∀ (a b c d : ℕ), 
  digit_cancel_fraction a b c d → 
  (10 * a + b == 26 ∧ 10 * c + d == 65) ∨
  (10 * a + b == 16 ∧ 10 * c + d == 64) ∨
  (10 * a + b == 19 ∧ 10 * c + d == 95) ∨
  (10 * a + b == 49 ∧ 10 * c + d == 98) :=
by {sorry}

end valid_digit_cancel_fractions_l86_86802


namespace girls_joined_l86_86080

theorem girls_joined (initial_girls : ℕ) (boys : ℕ) (girls_more_than_boys : ℕ) (G : ℕ) :
  initial_girls = 632 →
  boys = 410 →
  girls_more_than_boys = 687 →
  initial_girls + G = boys + girls_more_than_boys →
  G = 465 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  linarith

end girls_joined_l86_86080


namespace evaluate_expression_l86_86783

theorem evaluate_expression (x y z : ℤ) (hx : x = 25) (hy : y = 33) (hz : z = 7) :
    (x - (y - z)) - ((x - y) - z) = 14 := by 
  sorry

end evaluate_expression_l86_86783


namespace jeans_cost_before_sales_tax_l86_86400

-- Defining conditions
def original_cost : ℝ := 49
def summer_discount : ℝ := 0.50
def wednesday_discount : ℝ := 10

-- The mathematical equivalent proof problem
theorem jeans_cost_before_sales_tax :
  let discount_price := original_cost * (1 - summer_discount)
  let wednesday_price := discount_price - wednesday_discount
  wednesday_price = 14.50 :=
by
  let discount_price := original_cost * (1 - summer_discount)
  let wednesday_price := discount_price - wednesday_discount
  sorry

end jeans_cost_before_sales_tax_l86_86400


namespace binary_mul_correct_l86_86812

def bin_to_nat (l : List ℕ) : ℕ :=
  l.foldl (λ n b => 2 * n + b) 0

def p : List ℕ := [1,0,1,1,0,1]
def q : List ℕ := [1,1,0,1]
def r : List ℕ := [1,0,0,0,1,0,0,0,1,1]

theorem binary_mul_correct :
  bin_to_nat p * bin_to_nat q = bin_to_nat r := by
  sorry

end binary_mul_correct_l86_86812


namespace boat_speed_in_still_water_l86_86169

theorem boat_speed_in_still_water 
  (rate_of_current : ℝ) 
  (time_in_hours : ℝ) 
  (distance_downstream : ℝ)
  (h_rate : rate_of_current = 5) 
  (h_time : time_in_hours = 15 / 60) 
  (h_distance : distance_downstream = 6.25) : 
  ∃ x : ℝ, (distance_downstream = (x + rate_of_current) * time_in_hours) ∧ x = 20 :=
by 
  -- Main theorem statement, proof omitted for brevity.
  sorry

end boat_speed_in_still_water_l86_86169


namespace number_of_buses_l86_86793

theorem number_of_buses (total_students : ℕ) (students_per_bus : ℕ) (students_in_cars : ℕ) (buses : ℕ)
  (h1 : total_students = 375)
  (h2 : students_per_bus = 53)
  (h3 : students_in_cars = 4)
  (h4 : buses = (total_students - students_in_cars + students_per_bus - 1) / students_per_bus) :
  buses = 8 := by
  -- We will demonstrate that the number of buses indeed equals 8 under the given conditions.
  sorry

end number_of_buses_l86_86793


namespace find_x_value_l86_86672

-- Let's define the conditions
def equation (x y : ℝ) : Prop := x^2 - 4 * x + y = 0
def y_value : ℝ := 4

-- Define the theorem which states that x = 2 satisfies the conditions
theorem find_x_value (x : ℝ) (h : equation x y_value) : x = 2 :=
by
  sorry

end find_x_value_l86_86672


namespace original_selling_price_l86_86018

theorem original_selling_price (C : ℝ) (h : 1.60 * C = 2560) : 1.40 * C = 2240 :=
by
  sorry

end original_selling_price_l86_86018


namespace jellybean_removal_l86_86111

theorem jellybean_removal 
    (initial_count : ℕ) 
    (first_removal : ℕ) 
    (added_back : ℕ) 
    (final_count : ℕ)
    (initial_count_eq : initial_count = 37)
    (first_removal_eq : first_removal = 15)
    (added_back_eq : added_back = 5)
    (final_count_eq : final_count = 23) :
    (initial_count - first_removal + added_back - final_count) = 4 :=
by 
    sorry

end jellybean_removal_l86_86111


namespace even_function_l86_86685

-- Definition of f and F with the given conditions
variable (f : ℝ → ℝ)
variable (a : ℝ)
variable (x : ℝ)

-- Condition that x is in the interval (-a, a)
def in_interval (a x : ℝ) : Prop := x > -a ∧ x < a

-- Definition of F(x)
def F (x : ℝ) : ℝ := f x + f (-x)

-- The proposition that we want to prove
theorem even_function (h : in_interval a x) : F f x = F f (-x) :=
by
  unfold F
  sorry

end even_function_l86_86685


namespace correct_operation_l86_86144

theorem correct_operation :
  (3 * m^2 + 4 * m^2 ≠ 7 * m^4) ∧
  (4 * m^3 * 5 * m^3 ≠ 20 * m^3) ∧
  ((-2 * m)^3 ≠ -6 * m^3) ∧
  (m^10 / m^5 = m^5) :=
by
  sorry

end correct_operation_l86_86144


namespace find_constant_a_l86_86197

theorem find_constant_a (a : ℝ) : 
  (∃ x : ℝ, -3 ≤ x ∧ x ≤ 2 ∧ ax^2 + 2 * a * x + 1 = 9) → (a = -8 ∨ a = 1) :=
by
  sorry

end find_constant_a_l86_86197


namespace result_of_fractions_mult_l86_86608

theorem result_of_fractions_mult (a b c d : ℚ) (x : ℕ) :
  a = 3 / 4 →
  b = 1 / 2 →
  c = 2 / 5 →
  d = 5100 →
  a * b * c * d = 765 := by
  sorry

end result_of_fractions_mult_l86_86608


namespace pq_r_sum_l86_86015

theorem pq_r_sum (p q r : ℝ) (h1 : p^3 - 18 * p^2 + 27 * p - 72 = 0) 
                 (h2 : 27 * q^3 - 243 * q^2 + 729 * q - 972 = 0)
                 (h3 : 3 * r = 9) : p + q + r = 18 :=
by
  sorry

end pq_r_sum_l86_86015


namespace rectangle_is_square_l86_86576

theorem rectangle_is_square
  (a b: ℝ)  -- rectangle side lengths
  (h: a ≠ b)  -- initial assumption: rectangle not a square
  (shift_perpendicular: ∀ (P Q R S: ℝ × ℝ), (P ≠ Q → Q ≠ R → R ≠ S → S ≠ P) → (∀ (shift: ℝ × ℝ → ℝ × ℝ), ∀ (P₁: ℝ × ℝ), shift P₁ = P₁ + (0, 1) ∨ shift P₁ = P₁ + (1, 0)) → false):
  False := sorry

end rectangle_is_square_l86_86576


namespace positive_integer_triples_satisfying_conditions_l86_86048

theorem positive_integer_triples_satisfying_conditions :
  ∀ (a b c : ℕ), a^2 + b^2 + c^2 = 2005 ∧ a ≤ b ∧ b ≤ c →
  (a, b, c) = (23, 24, 30) ∨
  (a, b, c) = (12, 30, 31) ∨
  (a, b, c) = (9, 30, 32) ∨
  (a, b, c) = (4, 30, 33) ∨
  (a, b, c) = (15, 22, 36) ∨
  (a, b, c) = (9, 18, 40) ∨
  (a, b, c) = (4, 15, 42) :=
sorry

end positive_integer_triples_satisfying_conditions_l86_86048


namespace f_not_monotonic_l86_86193

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-(x:ℝ)) = -f x

def is_not_monotonic (f : ℝ → ℝ) : Prop :=
  ¬ ( (∀ x y, x < y → f x ≤ f y) ∨ (∀ x y, x < y → f y ≤ f x) )

variable (f : ℝ → ℝ)

axiom periodicity : ∀ x, f (x + 3/2) = -f x 
axiom odd_shifted : is_odd_function (λ x => f (x - 3/4))

theorem f_not_monotonic : is_not_monotonic f := by
  sorry

end f_not_monotonic_l86_86193


namespace european_fraction_is_one_fourth_l86_86831

-- Define the total number of passengers
def P : ℕ := 108

-- Define the fractions and the number of passengers from each continent
def northAmerica := (1 / 12) * P
def africa := (1 / 9) * P
def asia := (1 / 6) * P
def otherContinents := 42

-- Define the total number of non-European passengers
def totalNonEuropean := northAmerica + africa + asia + otherContinents

-- Define the number of European passengers
def european := P - totalNonEuropean

-- Define the fraction of European passengers
def europeanFraction := european / P

-- Prove that the fraction of European passengers is 1/4
theorem european_fraction_is_one_fourth : europeanFraction = 1 / 4 := 
by
  unfold europeanFraction european totalNonEuropean northAmerica africa asia P
  sorry

end european_fraction_is_one_fourth_l86_86831


namespace circle_properties_l86_86960

noncomputable def circle_center_and_radius (x y : ℝ) : ℝ × ℝ × ℝ :=
  let eq1 := x^2 - 4 * y - 18
  let eq2 := -y^2 + 6 * x + 26
  let lhs := x^2 - 6 * x + y^2 - 4 * y
  let rhs := 44
  let center_x := 3
  let center_y := 2
  let radius := Real.sqrt 57
  let target := 5 + radius
  (center_x, center_y, target)

theorem circle_properties
  (x y : ℝ) :
  let (a, b, r) := circle_center_and_radius x y 
  a + b + r = 5 + Real.sqrt 57 :=
by
  sorry

end circle_properties_l86_86960


namespace abs_discriminant_inequality_l86_86406

theorem abs_discriminant_inequality 
  (a b c A B C : ℝ) 
  (ha : a ≠ 0) 
  (hA : A ≠ 0) 
  (h : ∀ x : ℝ, |a * x^2 + b * x + c| ≤ |A * x^2 + B * x + C|) : 
  |b^2 - 4 * a * c| ≤ |B^2 - 4 * A * C| :=
sorry

end abs_discriminant_inequality_l86_86406


namespace product_of_equal_numbers_l86_86766

theorem product_of_equal_numbers (a b c d : ℕ) (h1 : (a + b + c + d) / 4 = 20) (h2 : a = 12) (h3 : b = 22) 
(h4 : c = d) : c * d = 529 := 
by
  sorry

end product_of_equal_numbers_l86_86766


namespace cos_17pi_over_4_eq_sqrt2_over_2_l86_86089

theorem cos_17pi_over_4_eq_sqrt2_over_2 : Real.cos (17 * Real.pi / 4) = Real.sqrt 2 / 2 := by
  sorry

end cos_17pi_over_4_eq_sqrt2_over_2_l86_86089


namespace factor_correct_l86_86487

noncomputable def factor_fraction (a b c : ℝ) : ℝ :=
  ((a^3 - b^3)^3 + (b^3 - c^3)^3 + (c^3 - a^3)^3) / ((a - b)^3 + (b - c)^3 + (c - a)^3)

theorem factor_correct (a b c : ℝ) : 
  factor_fraction a b c = (a^2 + a*b + b^2) * (b^2 + b*c + c^2) * (c^2 + c*a + a^2) :=
by
  sorry

end factor_correct_l86_86487


namespace same_color_probability_l86_86306

-- Define the total number of balls
def total_balls : ℕ := 4 + 6 + 5

-- Define the number of each color of balls
def white_balls : ℕ := 4
def black_balls : ℕ := 6
def red_balls : ℕ := 5

-- Define the events and probabilities
def pr_event (n : ℕ) (total : ℕ) : ℚ := n / total
def pr_cond_event (n : ℕ) (total : ℕ) : ℚ := n / total

-- Define the probabilities for each compound event
def pr_C1 : ℚ := pr_event white_balls total_balls * pr_cond_event (white_balls - 1) (total_balls - 1)
def pr_C2 : ℚ := pr_event black_balls total_balls * pr_cond_event (black_balls - 1) (total_balls - 1)
def pr_C3 : ℚ := pr_event red_balls total_balls * pr_cond_event (red_balls - 1) (total_balls - 1)

-- Define the total probability
def pr_C : ℚ := pr_C1 + pr_C2 + pr_C3

-- The goal is to prove that the total probability pr_C is equal to 31 / 105
theorem same_color_probability : pr_C = 31 / 105 := 
  by sorry

end same_color_probability_l86_86306


namespace sqrt_extraction_count_l86_86619

theorem sqrt_extraction_count (p : ℕ) [Fact p.Prime] : 
    ∃ k, k = (p + 1) / 2 ∧ ∀ n < p, ∃ x < p, x^2 ≡ n [MOD p] ↔ n < k := 
by
  sorry

end sqrt_extraction_count_l86_86619


namespace matrix_operation_value_l86_86060

theorem matrix_operation_value : 
  let p := 4 
  let q := 5
  let r := 2
  let s := 3 
  (p * s - q * r) = 2 :=
by
  sorry

end matrix_operation_value_l86_86060


namespace initial_population_l86_86419

theorem initial_population (P : ℝ) (h : 0.72 * P = 3168) : P = 4400 :=
sorry

end initial_population_l86_86419


namespace NewYearSeasonMarkup_is_25percent_l86_86053

variable (C N : ℝ)
variable (h1 : N >= 0)
variable (h2 : 0.92 * (1 + N) * 1.20 * C = 1.38 * C)

theorem NewYearSeasonMarkup_is_25percent : N = 0.25 :=
  by
  sorry

end NewYearSeasonMarkup_is_25percent_l86_86053


namespace canoe_row_probability_l86_86698

theorem canoe_row_probability :
  let p_left_works := 3 / 5
  let p_right_works := 3 / 5
  let p_left_breaks := 1 - p_left_works
  let p_right_breaks := 1 - p_right_works
  let p_can_still_row := (p_left_works * p_right_works) + (p_left_works * p_right_breaks) + (p_left_breaks * p_right_works)
  p_can_still_row = 21 / 25 :=
by
  sorry

end canoe_row_probability_l86_86698


namespace largest_of_decimals_l86_86081

theorem largest_of_decimals :
  let a := 0.993
  let b := 0.9899
  let c := 0.990
  let d := 0.989
  let e := 0.9909
  a > b ∧ a > c ∧ a > d ∧ a > e :=
by
  sorry

end largest_of_decimals_l86_86081


namespace new_socks_bought_l86_86425

-- Given conditions:
def initial_socks : ℕ := 11
def socks_thrown_away : ℕ := 4
def final_socks : ℕ := 33

-- Theorem proof statement:
theorem new_socks_bought : (final_socks - (initial_socks - socks_thrown_away)) = 26 :=
by
  sorry

end new_socks_bought_l86_86425


namespace solve_for_t_l86_86034

theorem solve_for_t (t : ℝ) (h1 : x = 1 - 4 * t) (h2 : y = 2 * t - 2) : x = y → t = 1/2 :=
by
  sorry

end solve_for_t_l86_86034


namespace vector_magnitude_l86_86019

noncomputable def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

noncomputable def magnitude (v : ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1 ^ 2 + v.2 ^ 2)

theorem vector_magnitude : 
  let AB := (-1, 2)
  let BC := (x, -5)
  let AC := (AB.1 + BC.1, AB.2 + BC.2)
  dot_product AB BC = -7 → magnitude AC = 5 :=
by sorry

end vector_magnitude_l86_86019


namespace triangle_ineq_l86_86231

noncomputable def TriangleSidesProof (AB AC BC : ℝ) :=
  AB = AC ∧ BC = 10 ∧ 2 * AB + BC ≤ 44 → 5 < AB ∧ AB ≤ 17

-- Statement for the proof problem
theorem triangle_ineq (AB AC BC : ℝ) (h1 : AB = AC) (h2 : BC = 10) (h3 : 2 * AB + BC ≤ 44) :
  5 < AB ∧ AB ≤ 17 :=
sorry

end triangle_ineq_l86_86231


namespace exist_pairs_sum_and_diff_l86_86677

theorem exist_pairs_sum_and_diff (N : ℕ) : ∃ a b c d : ℕ, 
  (a + b = c + d) ∧ (a * b + N = c * d ∨ a * b = c * d + N) := sorry

end exist_pairs_sum_and_diff_l86_86677


namespace find_A_l86_86325

theorem find_A (A : ℝ) (h : 4 * A + 5 = 33) : A = 7 :=
  sorry

end find_A_l86_86325


namespace geometric_sequence_150th_term_l86_86628

-- Given conditions
def a1 : ℤ := 5
def a2 : ℤ := -10

-- Computation of common ratio
def r : ℤ := a2 / a1

-- Definition of the n-th term in geometric sequence
def nth_term (n : ℕ) : ℤ :=
  a1 * r^(n-1)

-- Statement to prove
theorem geometric_sequence_150th_term :
  nth_term 150 = -5 * 2^149 :=
by
  sorry

end geometric_sequence_150th_term_l86_86628


namespace ones_digit_exponent_73_l86_86318

theorem ones_digit_exponent_73 (n : ℕ) : 
  (73 ^ n) % 10 = 7 ↔ n % 4 = 3 := 
sorry

end ones_digit_exponent_73_l86_86318


namespace least_N_no_square_l86_86589

theorem least_N_no_square (N : ℕ) : 
  (∀ k, (1000 * N) ≤ k ∧ k ≤ (1000 * N + 999) → 
  ∃ m, ¬ (k = m^2)) ↔ N = 282 :=
by
  sorry

end least_N_no_square_l86_86589


namespace percent_yield_hydrogen_gas_l86_86221

theorem percent_yield_hydrogen_gas
  (moles_fe : ℝ) (moles_h2so4 : ℝ) (actual_yield_h2 : ℝ) (theoretical_yield_h2 : ℝ) :
  moles_fe = 3 →
  moles_h2so4 = 4 →
  actual_yield_h2 = 1 →
  theoretical_yield_h2 = moles_fe →
  (actual_yield_h2 / theoretical_yield_h2) * 100 = 33.33 :=
by
  intros h_moles_fe h_moles_h2so4 h_actual_yield_h2 h_theoretical_yield_h2
  sorry

end percent_yield_hydrogen_gas_l86_86221


namespace expenditure_increase_l86_86474

theorem expenditure_increase (x : ℝ) (h₁ : 3 * x / (3 * x + 2 * x) = 3 / 5)
  (h₂ : 2 * x / (3 * x + 2 * x) = 2 / 5)
  (h₃ : ((5 * x) + 0.15 * (5 * x)) = 5.75 * x) 
  (h₄ : (2 * x + 0.06 * 2 * x) = 2.12 * x) 
  : ((3.63 * x - 3 * x) / (3 * x) * 100) = 21 := 
  by
  sorry

end expenditure_increase_l86_86474


namespace paidAmount_Y_l86_86849

theorem paidAmount_Y (X Y : ℝ) (h1 : X + Y = 638) (h2 : X = 1.2 * Y) : Y = 290 :=
by
  sorry

end paidAmount_Y_l86_86849


namespace peter_pizza_fraction_l86_86385

def pizza_slices : ℕ := 16
def peter_slices_alone : ℕ := 2
def shared_slice : ℚ := 1 / 2

theorem peter_pizza_fraction :
  let fraction_alone := peter_slices_alone * (1 / pizza_slices)
  let fraction_shared := shared_slice * (1 / pizza_slices)
  let total_fraction := fraction_alone + fraction_shared
  total_fraction = 5 / 32 :=
by
  let fraction_alone := peter_slices_alone * (1 / pizza_slices)
  let fraction_shared := shared_slice * (1 / pizza_slices)
  let total_fraction := fraction_alone + fraction_shared
  sorry

end peter_pizza_fraction_l86_86385


namespace inequality_solution_l86_86176

theorem inequality_solution :
  ∀ x : ℝ, ( (x - 3) / ( (x - 2) ^ 2 ) < 0 ) ↔ ( x < 2 ∨ (2 < x ∧ x < 3) ) :=
by
  sorry

end inequality_solution_l86_86176


namespace consecutive_even_number_difference_l86_86109

theorem consecutive_even_number_difference (x : ℤ) (h : x^2 - (x - 2)^2 = 2012) : x = 504 :=
sorry

end consecutive_even_number_difference_l86_86109


namespace probability_of_event_A_l86_86401

/-- The events A and B are independent, and it is given that:
  1. P(A) > 0
  2. P(A) = 2 * P(B)
  3. P(A or B) = 8 * P(A and B)

We need to prove that P(A) = 1/3. 
-/
theorem probability_of_event_A (P_A P_B : ℝ) (hP_indep : P_A * P_B = P_A) 
  (hP_A_pos : P_A > 0) (hP_A_eq_2P_B : P_A = 2 * P_B) 
  (hP_or_eq_8P_and : P_A + P_B - P_A * P_B = 8 * P_A * P_B) : 
  P_A = 1 / 3 := 
by
  sorry

end probability_of_event_A_l86_86401


namespace inequality_proof_l86_86062

theorem inequality_proof (a b : ℝ) : 
  a^2 + b^2 + 2 * (a - 1) * (b - 1) ≥ 1 := 
by 
  sorry

end inequality_proof_l86_86062


namespace negation_of_square_zero_l86_86064

variable {m : ℝ}

def is_positive (m : ℝ) : Prop := m > 0
def square_is_zero (m : ℝ) : Prop := m^2 = 0

theorem negation_of_square_zero (h : ∀ m, is_positive m → square_is_zero m) :
  ∀ m, ¬ is_positive m → ¬ square_is_zero m := 
sorry

end negation_of_square_zero_l86_86064


namespace green_peaches_sum_l86_86117

theorem green_peaches_sum (G1 G2 G3 : ℕ) : 
  (4 + G1) + (4 + G2) + (3 + G3) = 20 → G1 + G2 + G3 = 9 :=
by
  intro h
  sorry

end green_peaches_sum_l86_86117


namespace lisa_needs_change_probability_l86_86821

theorem lisa_needs_change_probability :
  let quarters := 16
  let toy_prices := List.range' 2 10 |> List.map (fun n => n * 25) -- List of toy costs: (50,75,...,300)
  let favorite_toy_price := 275
  let factorial := Nat.factorial
  let favorable := (factorial 9) + 9 * (factorial 8)
  let total_permutations := factorial 10
  let p_no_change := (favorable.toFloat / total_permutations.toFloat) -- Convert to Float for probability calculations
  let p_change_needed := Float.round ((1.0 - p_no_change) * 100.0) / 100.0
  p_change_needed = 4.0 / 5.0 := sorry

end lisa_needs_change_probability_l86_86821


namespace least_odd_prime_factor_of_2023_pow_8_add_1_l86_86747

theorem least_odd_prime_factor_of_2023_pow_8_add_1 :
  ∃ (p : ℕ), Prime p ∧ (2023^8 + 1) % p = 0 ∧ p % 2 = 1 ∧ p = 97 :=
by
  sorry

end least_odd_prime_factor_of_2023_pow_8_add_1_l86_86747


namespace sphere_radius_in_cube_l86_86580

theorem sphere_radius_in_cube (r : ℝ) (n : ℕ) (side_length : ℝ) 
  (h1 : side_length = 2) 
  (h2 : n = 16)
  (h3 : ∀ (i : ℕ), i < n → (center_distance : ℝ) = 2 * r)
  (h4: ∀ (i : ℕ), i < n → (face_distance : ℝ) = r) : 
  r = 1 :=
by
  sorry

end sphere_radius_in_cube_l86_86580


namespace margo_walks_total_distance_l86_86957

theorem margo_walks_total_distance :
  let time_to_house := 15
  let time_to_return := 25
  let total_time_minutes := time_to_house + time_to_return
  let total_time_hours := (total_time_minutes : ℝ) / 60
  let avg_rate := 3  -- units: miles per hour
  (avg_rate * total_time_hours = 2) := 
sorry

end margo_walks_total_distance_l86_86957


namespace total_tomato_seeds_l86_86416

theorem total_tomato_seeds (morn_mike morn_morning ted_morning sarah_morning : ℕ)
    (aft_mike aft_ted aft_sarah : ℕ)
    (H1 : morn_mike = 50)
    (H2 : ted_morning = 2 * morn_mike)
    (H3 : sarah_morning = morn_mike + 30)
    (H4 : aft_mike = 60)
    (H5 : aft_ted = aft_mike - 20)
    (H6 : aft_sarah = sarah_morning + 20) :
    morn_mike + aft_mike + ted_morning + aft_ted + sarah_morning + aft_sarah = 430 :=
by
  rw [H1, H2, H3, H4, H5, H6]
  sorry

end total_tomato_seeds_l86_86416


namespace remaining_shape_perimeter_l86_86985

def rectangle_perimeter (L W : ℕ) : ℕ := 2 * (L + W)

theorem remaining_shape_perimeter (L W S : ℕ) (hL : L = 12) (hW : W = 5) (hS : S = 2) :
  rectangle_perimeter L W = 34 :=
by
  rw [hL, hW]
  rfl

end remaining_shape_perimeter_l86_86985


namespace line_through_points_l86_86441

theorem line_through_points (m n p : ℝ) 
  (h1 : m = 4 * n + 5) 
  (h2 : m + 2 = 4 * (n + p) + 5) : 
  p = 1 / 2 := 
by 
  sorry

end line_through_points_l86_86441


namespace vector_magnitude_parallel_l86_86297

theorem vector_magnitude_parallel (x : ℝ) 
  (h1 : 4 / x = 2 / 1) :
  ( Real.sqrt ((4 + x) ^ 2 + (2 + 1) ^ 2) ) = 3 * Real.sqrt 5 := 
sorry

end vector_magnitude_parallel_l86_86297


namespace inequality_tangents_l86_86334

def f (x : ℝ) (a b : ℝ) : ℝ := x^3 - a * x - b

theorem inequality_tangents (a b : ℝ) (h1 : 0 < a)
  (h2 : ∃ x0 : ℝ, 2 * x0^3 - 3 * a * x0^2 + a^2 + 2 * b = 0): 
  -a^2 / 2 < b ∧ b < f a a b :=
by
  sorry

end inequality_tangents_l86_86334


namespace ellipse_focal_distance_l86_86138

theorem ellipse_focal_distance :
  let a := 9
  let b := 5
  let c := Real.sqrt (a^2 - b^2)
  2 * c = 4 * Real.sqrt 14 :=
by
  sorry

end ellipse_focal_distance_l86_86138


namespace find_f_prime_zero_l86_86978

variable {f : ℝ → ℝ}
variable {f' : ℝ → ℝ}

-- Condition given in the problem.
def f_def : ∀ x : ℝ, f x = x^2 + 2 * x * f' 1 := 
sorry

-- Statement we want to prove.
theorem find_f_prime_zero : f' 0 = -4 := 
sorry

end find_f_prime_zero_l86_86978


namespace man_is_older_by_l86_86096

theorem man_is_older_by :
  ∀ (M S : ℕ), S = 22 → (M + 2) = 2 * (S + 2) → (M - S) = 24 :=
by
  intros M S h1 h2
  sorry

end man_is_older_by_l86_86096


namespace smallest_x_for_square_l86_86970

theorem smallest_x_for_square (N : ℕ) (h1 : ∃ x : ℕ, x > 0 ∧ 1260 * x = N^2) : ∃ x : ℕ, x = 35 :=
by
  sorry

end smallest_x_for_square_l86_86970


namespace servings_in_box_l86_86106

theorem servings_in_box (total_cereal : ℕ) (serving_size : ℕ) (total_cereal_eq : total_cereal = 18) (serving_size_eq : serving_size = 2) :
  total_cereal / serving_size = 9 :=
by
  sorry

end servings_in_box_l86_86106


namespace number_of_boys_l86_86378

theorem number_of_boys (x : ℕ) (boys girls : ℕ)
  (initialRatio : girls / boys = 5 / 6)
  (afterLeavingRatio : (girls - 20) / boys = 2 / 3) :
  boys = 120 := by
  -- Proof is omitted
  sorry

end number_of_boys_l86_86378


namespace range_of_m_l86_86569

theorem range_of_m 
  (h : ∀ x, -1 < x ∧ x < 4 → x > 2 * (m: ℝ)^2 - 3)
  : ∀ (m: ℝ), -1 ≤ m ∧ m ≤ 1 :=
by 
  sorry

end range_of_m_l86_86569


namespace squares_are_equal_l86_86178

theorem squares_are_equal (a b c d : ℝ) (h₀ : a ≠ 0) (h₁ : b ≠ 0) (h₂ : c ≠ 0) (h₃ : d ≠ 0) 
    (h₄ : a * (b + c + d) = b * (a + c + d)) 
    (h₅ : a * (b + c + d) = c * (a + b + d)) 
    (h₆ : a * (b + c + d) = d * (a + b + c)) : 
    a^2 = b^2 ∧ b^2 = c^2 ∧ c^2 = d^2 := 
by
  sorry

end squares_are_equal_l86_86178


namespace range_of_a_l86_86819

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
if x >= 0 then -x + 3 * a else x^2 - a * x + 1

theorem range_of_a (a : ℝ) : (∀ x1 x2 : ℝ, x1 < x2 → f a x1 ≥ f a x2) ↔ (0 <= a ∧ a <= 1/3) :=
by
  sorry

end range_of_a_l86_86819


namespace acrobat_count_range_l86_86927

def animal_legs (elephants monkeys acrobats : ℕ) : ℕ :=
  4 * elephants + 2 * monkeys + 2 * acrobats

def animal_heads (elephants monkeys acrobats : ℕ) : ℕ :=
  elephants + monkeys + acrobats

theorem acrobat_count_range (e m a : ℕ) (h1 : animal_heads e m a = 18)
  (h2 : animal_legs e m a = 50) : 0 ≤ a ∧ a ≤ 11 :=
by {
  sorry
}

end acrobat_count_range_l86_86927


namespace problem_sequence_inequality_l86_86815

def a (n : ℕ) : ℚ := 15 + (n - 1 : ℚ) * (-(2 / 3))

theorem problem_sequence_inequality :
  ∃ k : ℕ, (a k) * (a (k + 1)) < 0 ∧ k = 23 :=
by {
  use 23,
  sorry
}

end problem_sequence_inequality_l86_86815


namespace cubic_of_cubic_roots_correct_l86_86296

variable (a b c : ℝ) (α β γ : ℝ)

-- Vieta's formulas conditions
axiom vieta1 : α + β + γ = -a
axiom vieta2 : α * β + β * γ + γ * α = b
axiom vieta3 : α * β * γ = -c

-- Define the polynomial whose roots are α³, β³, and γ³
def cubic_of_cubic_roots (x : ℝ) : ℝ :=
  x^3 + (a^3 - 3*a*b + 3*c)*x^2 + (b^3 + 3*c^2 - 3*a*b*c)*x + c^3

-- Prove that this polynomial has α³, β³, γ³ as roots
theorem cubic_of_cubic_roots_correct :
  ∀ x : ℝ, cubic_of_cubic_roots a b c x = 0 ↔ (x = α^3 ∨ x = β^3 ∨ x = γ^3) :=
sorry

end cubic_of_cubic_roots_correct_l86_86296


namespace number_of_unique_four_digit_numbers_from_2004_l86_86624

-- Definitions representing the conditions
def is_four_digit_number (n : ℕ) : Prop := 1000 ≤ n ∧ n < 10000
def uses_digits_from_2004 (n : ℕ) : Prop := 
  ∀ d ∈ [n / 1000, (n / 100) % 10, (n / 10) % 10, n % 10], d ∈ [0, 2, 4]

-- The proposition we need to prove
theorem number_of_unique_four_digit_numbers_from_2004 :
  ∃ n : ℕ, is_four_digit_number n ∧ uses_digits_from_2004 n ∧ n = 6 := 
sorry

end number_of_unique_four_digit_numbers_from_2004_l86_86624


namespace cube_root_neg_eight_l86_86043

theorem cube_root_neg_eight : ∃ x : ℝ, x^3 = -8 ∧ x = -2 :=
by {
  sorry
}

end cube_root_neg_eight_l86_86043


namespace total_sticks_of_gum_in_12_brown_boxes_l86_86235

-- Definitions based on the conditions
def packs_per_carton := 7
def sticks_per_pack := 5
def cartons_in_full_box := 6
def cartons_in_partial_box := 3
def num_brown_boxes := 12
def num_partial_boxes := 2

-- Calculation definitions
def sticks_per_carton := packs_per_carton * sticks_per_pack
def sticks_per_full_box := cartons_in_full_box * sticks_per_carton
def sticks_per_partial_box := cartons_in_partial_box * sticks_per_carton
def num_full_boxes := num_brown_boxes - num_partial_boxes

-- Final total sticks of gum
def total_sticks_of_gum := (num_full_boxes * sticks_per_full_box) + (num_partial_boxes * sticks_per_partial_box)

-- The theorem to be proved
theorem total_sticks_of_gum_in_12_brown_boxes :
  total_sticks_of_gum = 2310 :=
by
  -- The proof is omitted.
  sorry

end total_sticks_of_gum_in_12_brown_boxes_l86_86235
