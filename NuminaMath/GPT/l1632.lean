import Mathlib

namespace NUMINAMATH_GPT_range_of_x_l1632_163286

theorem range_of_x (x : ℝ) : (x ≠ -3) ∧ (x ≤ 4) ↔ (x ≤ 4) ∧ (x ≠ -3) :=
by { sorry }

end NUMINAMATH_GPT_range_of_x_l1632_163286


namespace NUMINAMATH_GPT_tangent_line_circle_l1632_163250

theorem tangent_line_circle (r : ℝ) (h : 0 < r) :
  (∀ x y : ℝ, x + y = r → x * x + y * y ≠ 4 * r) →
  r = 8 :=
by
  sorry

end NUMINAMATH_GPT_tangent_line_circle_l1632_163250


namespace NUMINAMATH_GPT_total_volume_is_10_l1632_163292

noncomputable def total_volume_of_final_mixture (V : ℝ) : ℝ :=
  2.5 + V

theorem total_volume_is_10 :
  ∃ (V : ℝ), 
  (0.30 * 2.5 + 0.50 * V = 0.45 * (2.5 + V)) ∧ 
  total_volume_of_final_mixture V = 10 :=
by
  sorry

end NUMINAMATH_GPT_total_volume_is_10_l1632_163292


namespace NUMINAMATH_GPT_max_green_socks_l1632_163255

theorem max_green_socks (g y : ℕ) (h1 : g + y ≤ 2025)
  (h2 : (g * (g - 1))/(g + y) * (g + y - 1) = 1/3) : 
  g ≤ 990 := 
sorry

end NUMINAMATH_GPT_max_green_socks_l1632_163255


namespace NUMINAMATH_GPT_original_wage_l1632_163296

theorem original_wage (W : ℝ) 
  (h1: 1.40 * W = 28) : 
  W = 20 :=
sorry

end NUMINAMATH_GPT_original_wage_l1632_163296


namespace NUMINAMATH_GPT_find_number_l1632_163249

theorem find_number 
  (x : ℚ) 
  (h : (3 / 4) * x - (8 / 5) * x + 63 = 12) : 
  x = 60 := 
by
  sorry

end NUMINAMATH_GPT_find_number_l1632_163249


namespace NUMINAMATH_GPT_first_duck_fraction_l1632_163203

-- Definitions based on the conditions
variable (total_bread : ℕ) (left_bread : ℕ) (second_duck_bread : ℕ) (third_duck_bread : ℕ)

-- Given values
def given_values : Prop :=
  total_bread = 100 ∧ left_bread = 30 ∧ second_duck_bread = 13 ∧ third_duck_bread = 7

-- Proof statement
theorem first_duck_fraction (h : given_values total_bread left_bread second_duck_bread third_duck_bread) :
  (total_bread - left_bread) - (second_duck_bread + third_duck_bread) = 1/2 * total_bread := by 
  sorry

end NUMINAMATH_GPT_first_duck_fraction_l1632_163203


namespace NUMINAMATH_GPT_length_less_than_twice_width_l1632_163267

def length : ℝ := 24
def width : ℝ := 13.5

theorem length_less_than_twice_width : 2 * width - length = 3 := by
  sorry

end NUMINAMATH_GPT_length_less_than_twice_width_l1632_163267


namespace NUMINAMATH_GPT_shortest_tree_height_proof_l1632_163299

def tallest_tree_height : ℕ := 150
def middle_tree_height : ℕ := (2 * tallest_tree_height) / 3
def shortest_tree_height : ℕ := middle_tree_height / 2

theorem shortest_tree_height_proof : shortest_tree_height = 50 := by
  sorry

end NUMINAMATH_GPT_shortest_tree_height_proof_l1632_163299


namespace NUMINAMATH_GPT_melanie_plums_l1632_163226

variable (initialPlums : ℕ) (givenPlums : ℕ)

theorem melanie_plums :
  initialPlums = 7 → givenPlums = 3 → initialPlums - givenPlums = 4 :=
by
  intro h1 h2
  -- proof omitted
  exact sorry

end NUMINAMATH_GPT_melanie_plums_l1632_163226


namespace NUMINAMATH_GPT_relationship_abc_d_l1632_163221

theorem relationship_abc_d : 
  ∀ (a b c d : ℝ), 
  a < b → 
  d < c → 
  (c - a) * (c - b) < 0 → 
  (d - a) * (d - b) > 0 → 
  d < a ∧ a < c ∧ c < b :=
by
  intros a b c d a_lt_b d_lt_c h1 h2
  sorry

end NUMINAMATH_GPT_relationship_abc_d_l1632_163221


namespace NUMINAMATH_GPT_symmetric_point_with_respect_to_y_eq_x_l1632_163260

theorem symmetric_point_with_respect_to_y_eq_x :
  ∃ x₀ y₀ : ℝ, (∃ (M : ℝ × ℝ), M = (3, 1) ∧
  ((x₀ + 3) / 2 = (y₀ + 1) / 2) ∧
  ((y₀ - 1) / (x₀ - 3) = -1)) ∧
  (x₀ = 1 ∧ y₀ = 3) :=
by
  sorry

end NUMINAMATH_GPT_symmetric_point_with_respect_to_y_eq_x_l1632_163260


namespace NUMINAMATH_GPT_smallest_four_digit_divisible_by_9_l1632_163251

theorem smallest_four_digit_divisible_by_9 
    (n : ℕ) 
    (h1 : 1000 ≤ n ∧ n < 10000) 
    (h2 : n % 9 = 0)
    (h3 : n % 10 % 2 = 1)
    (h4 : (n / 1000) % 2 = 1)
    (h5 : (n / 10) % 10 % 2 = 0)
    (h6 : (n / 100) % 10 % 2 = 0) :
  n = 3609 :=
sorry

end NUMINAMATH_GPT_smallest_four_digit_divisible_by_9_l1632_163251


namespace NUMINAMATH_GPT_convert_yahs_to_bahs_l1632_163252

theorem convert_yahs_to_bahs :
  (∀ (bahs rahs yahs : ℝ), (10 * bahs = 18 * rahs) 
    ∧ (6 * rahs = 10 * yahs) 
    → (1500 * yahs / (10 / 6) / (18 / 10) = 500 * bahs)) :=
by
  intros bahs rahs yahs h
  sorry

end NUMINAMATH_GPT_convert_yahs_to_bahs_l1632_163252


namespace NUMINAMATH_GPT_complement_union_l1632_163258

def R := Set ℝ

def A : Set ℝ := {x | x ≥ 1}

def B : Set ℝ := {y | ∃ x, x ≥ 1 ∧ y = Real.exp x}

theorem complement_union (R : Set ℝ) (A : Set ℝ) (B : Set ℝ) :
  (A ∪ B)ᶜ = {x | x < 1} := by
  sorry

end NUMINAMATH_GPT_complement_union_l1632_163258


namespace NUMINAMATH_GPT_ensure_two_of_each_l1632_163213

theorem ensure_two_of_each {A B : ℕ} (hA : A = 10) (hB : B = 10) :
  ∃ n : ℕ, n = 12 ∧
  ∀ (extracted : ℕ → ℕ),
    (extracted 0 + extracted 1 = n) →
    (extracted 0 ≥ 2 ∧ extracted 1 ≥ 2) :=
by
  sorry

end NUMINAMATH_GPT_ensure_two_of_each_l1632_163213


namespace NUMINAMATH_GPT_distance_to_line_l1632_163248

theorem distance_to_line (a : ℝ) (d : ℝ)
  (h1 : d = 6)
  (h2 : |3 * a + 6| / 5 = d) :
  a = 8 ∨ a = -12 :=
by
  sorry

end NUMINAMATH_GPT_distance_to_line_l1632_163248


namespace NUMINAMATH_GPT_Kamal_biology_marks_l1632_163234

theorem Kamal_biology_marks 
  (E : ℕ) (M : ℕ) (P : ℕ) (C : ℕ) (A : ℕ) (N : ℕ) (B : ℕ) 
  (hE : E = 66)
  (hM : M = 65)
  (hP : P = 77)
  (hC : C = 62)
  (hA : A = 69)
  (hN : N = 5)
  (h_total : N * A = E + M + P + C + B) 
  : B = 75 :=
by
  sorry

end NUMINAMATH_GPT_Kamal_biology_marks_l1632_163234


namespace NUMINAMATH_GPT_complex_multiplication_l1632_163224

theorem complex_multiplication (i : ℂ) (hi : i^2 = -1) : (1 + i) * (1 - i) = 1 := 
by
  sorry

end NUMINAMATH_GPT_complex_multiplication_l1632_163224


namespace NUMINAMATH_GPT_ball_bounces_l1632_163211

theorem ball_bounces (k : ℕ) :
  1500 * (2 / 3 : ℝ)^k < 2 ↔ k ≥ 19 :=
sorry

end NUMINAMATH_GPT_ball_bounces_l1632_163211


namespace NUMINAMATH_GPT_proof_equivalence_l1632_163243

variables {A B C D : Type} [AddGroup A] [AddGroup B] [AddGroup C] [AddGroup D]
variables {α β γ δ : ℝ} -- angles are real numbers

-- Definition of cyclic quadrilateral
def cyclic_quadrilateral (α β γ δ : ℝ) : Prop :=
α + γ = 180 ∧ β + δ = 180

-- Definition of the problem statements
def statement1 (α γ : ℝ) : Prop :=
α = γ → α = 90

def statement3 (α γ : ℝ) : Prop :=
180 - α + 180 - γ = 180

def statement2 (α β : ℝ) (ψ χ : ℝ) : Prop := 
α = β → cyclic_quadrilateral α β ψ χ → ψ = χ ∨ (α = β ∧ α = ψ ∧ α = χ)

def statement4 (α β γ δ : ℝ) : Prop :=
1*α + 2*β + 3*γ + 4*δ = 360

-- Theorem statement
theorem proof_equivalence (α β γ δ : ℝ) :
  cyclic_quadrilateral α β γ δ →
  (statement1 α γ) ∧ (statement3 α γ) ∧ ¬(statement2 α β γ δ) ∧ ¬(statement4 α β γ δ) :=
by
  sorry

end NUMINAMATH_GPT_proof_equivalence_l1632_163243


namespace NUMINAMATH_GPT_find_a9_l1632_163218

theorem find_a9 (a_1 a_2 : ℕ) (a : ℕ → ℕ) 
  (h1 : ∀ n ≥ 1, a (n + 2) = a (n + 1) + a n)
  (h2 : a 7 = 210)
  (h3 : a 1 = a_1)
  (h4 : a 2 = a_2) : 
  a 9 = 550 := by
  sorry

end NUMINAMATH_GPT_find_a9_l1632_163218


namespace NUMINAMATH_GPT_interest_rate_calculation_l1632_163254

theorem interest_rate_calculation (P : ℝ) (r : ℝ) (h1 : P * (1 + r / 100)^3 = 800) (h2 : P * (1 + r / 100)^4 = 820) :
  r = 2.5 := 
  sorry

end NUMINAMATH_GPT_interest_rate_calculation_l1632_163254


namespace NUMINAMATH_GPT_square_side_increase_l1632_163269

theorem square_side_increase (s : ℝ) :
  let new_side := 1.5 * s
  let new_area := new_side^2
  let original_area := s^2
  let new_perimeter := 4 * new_side
  let original_perimeter := 4 * s
  let new_diagonal := new_side * Real.sqrt 2
  let original_diagonal := s * Real.sqrt 2
  (new_area - original_area) / original_area * 100 = 125 ∧
  (new_perimeter - original_perimeter) / original_perimeter * 100 = 50 ∧
  (new_diagonal - original_diagonal) / original_diagonal * 100 = 50 :=
by
  sorry

end NUMINAMATH_GPT_square_side_increase_l1632_163269


namespace NUMINAMATH_GPT_solve_exp_l1632_163209

theorem solve_exp (x : ℕ) : 8^x = 2^9 → x = 3 :=
by
  sorry

end NUMINAMATH_GPT_solve_exp_l1632_163209


namespace NUMINAMATH_GPT_find_ab_l1632_163288

def f (a b : ℝ) (x : ℝ) : ℝ := a * x^2 - 2 * a * x + 2 + b

theorem find_ab (a b : ℝ) (h₀ : a ≠ 0) (h₁ : f a b 2 = 2) (h₂ : f a b 3 = 5) :
    (a = 1 ∧ b = 0) ∨ (a = -1 ∧ b = 3) :=
by 
  sorry

end NUMINAMATH_GPT_find_ab_l1632_163288


namespace NUMINAMATH_GPT_function_machine_output_l1632_163287

-- Define the initial input
def input : ℕ := 12

-- Define the function machine steps
def functionMachine (x : ℕ) : ℕ :=
  if x * 3 <= 20 then (x * 3) / 2
  else (x * 3) - 2

-- State the property we want to prove
theorem function_machine_output : functionMachine 12 = 34 :=
by
  -- Skip the proof
  sorry

end NUMINAMATH_GPT_function_machine_output_l1632_163287


namespace NUMINAMATH_GPT_liam_savings_per_month_l1632_163216

theorem liam_savings_per_month (trip_cost bill_cost left_after_bills : ℕ) 
                               (months_in_two_years : ℕ) (total_savings_per_month : ℕ) :
  trip_cost = 7000 →
  bill_cost = 3500 →
  left_after_bills = 8500 →
  months_in_two_years = 24 →
  total_savings_per_month = 19000 →
  total_savings_per_month / months_in_two_years = 79167 / 100 :=
by
  intros
  sorry

end NUMINAMATH_GPT_liam_savings_per_month_l1632_163216


namespace NUMINAMATH_GPT_opposite_of_neg2016_l1632_163238

theorem opposite_of_neg2016 : -(-2016) = 2016 := 
by 
  sorry

end NUMINAMATH_GPT_opposite_of_neg2016_l1632_163238


namespace NUMINAMATH_GPT_parallelogram_opposite_sides_equal_l1632_163207

-- Given definitions and properties of a parallelogram
structure Parallelogram (α : Type*) [Add α] [AddCommGroup α] [Module ℝ α] :=
(a b c d : α) 
(parallel_a : a + b = c + d)
(parallel_b : b + c = d + a)
(parallel_c : c + d = a + b)
(parallel_d : d + a = b + c)

open Parallelogram

-- Define problem statement to prove opposite sides are equal
theorem parallelogram_opposite_sides_equal {α : Type*} [Add α] [AddCommGroup α] [Module ℝ α] 
  (p : Parallelogram α) : 
  p.a = p.c ∧ p.b = p.d :=
sorry -- Proof goes here

end NUMINAMATH_GPT_parallelogram_opposite_sides_equal_l1632_163207


namespace NUMINAMATH_GPT_sandwich_price_l1632_163277

-- Definitions based on conditions
def price_of_soda : ℝ := 0.87
def total_cost : ℝ := 6.46
def num_soda : ℝ := 4
def num_sandwich : ℝ := 2

-- The key equation based on conditions
def total_cost_equation (S : ℝ) : Prop := 
  num_sandwich * S + num_soda * price_of_soda = total_cost

theorem sandwich_price :
  ∃ S : ℝ, total_cost_equation S ∧ S = 1.49 :=
by
  sorry

end NUMINAMATH_GPT_sandwich_price_l1632_163277


namespace NUMINAMATH_GPT_Walter_age_in_2010_l1632_163275

-- Define Walter's age in 2005 as y
def Walter_age_2005 (y : ℕ) : Prop :=
  (2005 - y) + (2005 - 3 * y) = 3858

-- Define Walter's age in 2010
theorem Walter_age_in_2010 (y : ℕ) (hy : Walter_age_2005 y) : y + 5 = 43 :=
by
  sorry

end NUMINAMATH_GPT_Walter_age_in_2010_l1632_163275


namespace NUMINAMATH_GPT_sum_of_bases_l1632_163259

theorem sum_of_bases (F1 F2 : ℚ) (R1 R2 : ℕ) (hF1_R1 : F1 = (3 * R1 + 7) / (R1^2 - 1) ∧ F2 = (7 * R1 + 3) / (R1^2 - 1))
    (hF1_R2 : F1 = (2 * R2 + 5) / (R2^2 - 1) ∧ F2 = (5 * R2 + 2) / (R2^2 - 1)) : 
    R1 + R2 = 19 := 
sorry

end NUMINAMATH_GPT_sum_of_bases_l1632_163259


namespace NUMINAMATH_GPT_factorization_example_l1632_163208

theorem factorization_example :
  (x : ℝ) → (x^2 + 6 * x + 9 = (x + 3)^2) :=
by
  sorry

end NUMINAMATH_GPT_factorization_example_l1632_163208


namespace NUMINAMATH_GPT_students_with_dogs_l1632_163278

theorem students_with_dogs (total_students : ℕ) (half_students : total_students / 2 = 50)
  (percent_girls_with_dogs : ℕ → ℚ) (percent_boys_with_dogs : ℕ → ℚ)
  (girls_with_dogs : ∀ (total_girls: ℕ), percent_girls_with_dogs total_girls = 0.2)
  (boys_with_dogs : ∀ (total_boys: ℕ), percent_boys_with_dogs total_boys = 0.1) :
  ∀ (total_girls total_boys students_with_dogs: ℕ),
  total_students = 100 →
  total_girls = total_students / 2 →
  total_boys = total_students / 2 →
  total_girls = 50 →
  total_boys = 50 →
  students_with_dogs = (percent_girls_with_dogs (total_students / 2) * (total_students / 2) + 
                        percent_boys_with_dogs (total_students / 2) * (total_students / 2)) →
  students_with_dogs = 15 :=
by
  intros total_girls total_boys students_with_dogs h1 h2 h3 h4 h5 h6
  sorry

end NUMINAMATH_GPT_students_with_dogs_l1632_163278


namespace NUMINAMATH_GPT_ratio_of_divisor_to_quotient_l1632_163263

noncomputable def r : ℕ := 5
noncomputable def n : ℕ := 113

-- Assuming existence of k and quotient Q
axiom h1 : ∃ (k Q : ℕ), (3 * r + 3 = k * Q) ∧ (n = (3 * r + 3) * Q + r)

theorem ratio_of_divisor_to_quotient : ∃ (D Q : ℕ), (D = 3 * r + 3) ∧ (n = D * Q + r) ∧ (D / Q = 3) :=
  by sorry

end NUMINAMATH_GPT_ratio_of_divisor_to_quotient_l1632_163263


namespace NUMINAMATH_GPT_mary_earns_per_home_l1632_163201

theorem mary_earns_per_home :
  let total_earned := 12696
  let homes_cleaned := 276.0
  total_earned / homes_cleaned = 46 :=
by
  sorry

end NUMINAMATH_GPT_mary_earns_per_home_l1632_163201


namespace NUMINAMATH_GPT_t_shirt_cost_calculation_l1632_163237

variables (initial_amount ticket_cost food_cost money_left t_shirt_cost : ℕ)

axiom h1 : initial_amount = 75
axiom h2 : ticket_cost = 30
axiom h3 : food_cost = 13
axiom h4 : money_left = 9

theorem t_shirt_cost_calculation : 
  t_shirt_cost = initial_amount - (ticket_cost + food_cost) - money_left :=
sorry

end NUMINAMATH_GPT_t_shirt_cost_calculation_l1632_163237


namespace NUMINAMATH_GPT_maple_trees_planted_plant_maple_trees_today_l1632_163270

-- Define the initial number of maple trees
def initial_maple_trees : ℕ := 2

-- Define the number of maple trees the park will have after planting
def final_maple_trees : ℕ := 11

-- Define the number of popular trees, though it is irrelevant for the proof
def initial_popular_trees : ℕ := 5

-- The main statement to prove: number of maple trees planted today
theorem maple_trees_planted : ℕ :=
  final_maple_trees - initial_maple_trees

-- Prove that the number of maple trees planted today is 9
theorem plant_maple_trees_today :
  maple_trees_planted = 9 :=
by
  sorry

end NUMINAMATH_GPT_maple_trees_planted_plant_maple_trees_today_l1632_163270


namespace NUMINAMATH_GPT_complex_seventh_root_of_unity_l1632_163242

theorem complex_seventh_root_of_unity (r : ℂ) (h1 : r^7 = 1) (h2: r ≠ 1) : 
  (r - 1) * (r^2 - 1) * (r^3 - 1) * (r^4 - 1) * (r^5 - 1) * (r^6 - 1) = 7 :=
by
  sorry

end NUMINAMATH_GPT_complex_seventh_root_of_unity_l1632_163242


namespace NUMINAMATH_GPT_a_neg_half_not_bounded_a_bounded_range_l1632_163241

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
  1 + a * (1/3)^x + (1/9)^x

theorem a_neg_half_not_bounded (a : ℝ) :
  a = -1/2 → ¬(∃ M > 0, ∀ x < 0, |f x a| ≤ M) :=
by
  sorry

theorem a_bounded_range (a : ℝ) : 
  (∀ x ≥ 0, |f x a| ≤ 4) → -6 ≤ a ∧ a ≤ 2 :=
by
  sorry

end NUMINAMATH_GPT_a_neg_half_not_bounded_a_bounded_range_l1632_163241


namespace NUMINAMATH_GPT_total_students_surveyed_l1632_163204

variable (T : ℕ)
variable (F : ℕ)

theorem total_students_surveyed :
  (F = 20 + 60) → (F = 40 * (T / 100)) → (T = 200) :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_total_students_surveyed_l1632_163204


namespace NUMINAMATH_GPT_birds_on_fence_l1632_163206

theorem birds_on_fence (B S : ℕ): 
  S = 3 →
  S + 6 = B + 5 →
  B = 4 :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_birds_on_fence_l1632_163206


namespace NUMINAMATH_GPT_f_10_l1632_163262

namespace MathProof

variable (f : ℤ → ℤ)

-- Condition 1: f(1) + 1 > 0
axiom cond1 : f 1 + 1 > 0

-- Condition 2: f(x + y) - x * f(y) - y * f(x) = f(x) * f(y) - x - y + x * y for any x, y ∈ ℤ
axiom cond2 : ∀ x y : ℤ, f (x + y) - x * f y - y * f x = f x * f y - x - y + x * y

-- Condition 3: 2 * f(x) = f(x + 1) - x + 1 for any x ∈ ℤ
axiom cond3 : ∀ x : ℤ, 2 * f x = f (x + 1) - x + 1

-- We need to prove f(10) = 1014
theorem f_10 : f 10 = 1014 :=
by
  sorry

end MathProof

end NUMINAMATH_GPT_f_10_l1632_163262


namespace NUMINAMATH_GPT_equilateral_is_cute_specific_triangle_is_cute_find_AB_length_l1632_163256

-- Definition of a cute triangle
def is_cute_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = 2 * c^2 ∨ a^2 + c^2 = 2 * b^2 ∨ b^2 + c^2 = 2 * a^2

-- 1. Prove an equilateral triangle is a cute triangle
theorem equilateral_is_cute (a : ℝ) : is_cute_triangle a a a :=
by
  sorry

-- 2. Prove the triangle with sides 4, 2√6, and 2√5 is a cute triangle
theorem specific_triangle_is_cute : is_cute_triangle 4 (2*Real.sqrt 6) (2*Real.sqrt 5) :=
by
  sorry

-- 3. Prove the length of AB for the given right triangle is 2√6 or 2√3
theorem find_AB_length (AB BC : ℝ) (AC : ℝ := 2*Real.sqrt 2) (h_cute : is_cute_triangle AB BC AC) : AB = 2*Real.sqrt 6 ∨ AB = 2*Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_equilateral_is_cute_specific_triangle_is_cute_find_AB_length_l1632_163256


namespace NUMINAMATH_GPT_response_rate_is_60_percent_l1632_163253

-- Definitions based on conditions
def responses_needed : ℕ := 900
def questionnaires_mailed : ℕ := 1500

-- Derived definition
def response_rate_percentage : ℚ := (responses_needed : ℚ) / (questionnaires_mailed : ℚ) * 100

-- The theorem stating the problem
theorem response_rate_is_60_percent :
  response_rate_percentage = 60 := 
sorry

end NUMINAMATH_GPT_response_rate_is_60_percent_l1632_163253


namespace NUMINAMATH_GPT_interest_groups_ranges_l1632_163285

variable (A B C : Finset ℕ)

-- Given conditions
axiom card_A : A.card = 5
axiom card_B : B.card = 4
axiom card_C : C.card = 7
axiom card_A_inter_B : (A ∩ B).card = 3
axiom card_A_inter_B_inter_C : (A ∩ B ∩ C).card = 2

-- Mathematical statement to be proved
theorem interest_groups_ranges :
  2 ≤ ((A ∪ B) ∩ C).card ∧ ((A ∪ B) ∩ C).card ≤ 5 ∧
  8 ≤ (A ∪ B ∪ C).card ∧ (A ∪ B ∪ C).card ≤ 11 := by
  sorry

end NUMINAMATH_GPT_interest_groups_ranges_l1632_163285


namespace NUMINAMATH_GPT_arithmetic_sequence_max_sum_proof_l1632_163265

noncomputable def arithmetic_sequence_max_sum (a_1 d : ℝ) (n : ℕ) : ℝ :=
  n * a_1 + (n * (n - 1)) / 2 * d

theorem arithmetic_sequence_max_sum_proof (a_1 d : ℝ) 
  (h1 : 3 * a_1 + 6 * d = 9)
  (h2 : a_1 + 5 * d = -9) :
  ∃ n : ℕ, n = 3 ∧ arithmetic_sequence_max_sum a_1 d n = 21 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_max_sum_proof_l1632_163265


namespace NUMINAMATH_GPT_pratt_certificate_space_bound_l1632_163212

-- Define the Pratt certificate space function λ(p)
noncomputable def pratt_space (p : ℕ) : ℝ := sorry

-- Define the log_2 function (if not already available in Mathlib)
noncomputable def log2 (x : ℝ) : ℝ := sorry

-- Assuming that p is a prime number
variable {p : ℕ} (hp : Nat.Prime p)

-- The proof problem
theorem pratt_certificate_space_bound (hp : Nat.Prime p) :
  pratt_space p ≤ 6 * (log2 p) ^ 2 := 
sorry

end NUMINAMATH_GPT_pratt_certificate_space_bound_l1632_163212


namespace NUMINAMATH_GPT_natural_solution_unique_l1632_163280

theorem natural_solution_unique (n : ℕ) (h : (2 * n - 1) / n^5 = 3 - 2 / n) : n = 1 := by
  sorry

end NUMINAMATH_GPT_natural_solution_unique_l1632_163280


namespace NUMINAMATH_GPT_x_is_perfect_square_l1632_163230

theorem x_is_perfect_square {x y : ℕ} (hx : x > 0) (hy : y > 0) (h : (x^2 + y^2 - x) % (2 * x * y) = 0) : ∃ z : ℕ, x = z^2 :=
by
  -- The proof will proceed here
  sorry

end NUMINAMATH_GPT_x_is_perfect_square_l1632_163230


namespace NUMINAMATH_GPT_triangle_angle_A_l1632_163264

theorem triangle_angle_A (A B C : ℝ) (h1 : C = 3 * B) (h2 : B = 30) (h3 : A + B + C = 180) : A = 60 := by
  sorry

end NUMINAMATH_GPT_triangle_angle_A_l1632_163264


namespace NUMINAMATH_GPT_inequality_multiplication_l1632_163210

theorem inequality_multiplication (a b : ℝ) (h : a > b) : 3 * a > 3 * b :=
sorry

end NUMINAMATH_GPT_inequality_multiplication_l1632_163210


namespace NUMINAMATH_GPT_arithmetic_sequence_inequality_l1632_163268

variable {α : Type*} [OrderedRing α]

theorem arithmetic_sequence_inequality
  (a : ℕ → α) (d : α)
  (h_arith_seq : ∀ n, a (n + 1) = a n + d)
  (h_d_pos : d > 0)
  (n : ℕ)
  (h_n_gt_1 : n > 1) :
  a 1 * a (n + 1) < a 2 * a n := 
sorry

end NUMINAMATH_GPT_arithmetic_sequence_inequality_l1632_163268


namespace NUMINAMATH_GPT_at_least_one_has_two_distinct_roots_l1632_163244

theorem at_least_one_has_two_distinct_roots
  (p q1 q2 : ℝ)
  (h : p = q1 + q2 + 1) :
  (1 - 4 * q1 > 0) ∨ ((q1 + q2 + 1) ^ 2 - 4 * q2 > 0) :=
by sorry

end NUMINAMATH_GPT_at_least_one_has_two_distinct_roots_l1632_163244


namespace NUMINAMATH_GPT_asymptotes_of_hyperbola_l1632_163235

theorem asymptotes_of_hyperbola (b : ℝ) (h_focus : 2 * Real.sqrt 2 ≠ 0) :
  2 * Real.sqrt 2 = Real.sqrt ((2 * 2) + b^2) → 
  (∀ (x y : ℝ), ((x^2 / 4) - (y^2 / b^2) = 1 → x^2 - y^2 = 4)) → 
  (∀ (x y : ℝ), ((x^2 - y^2 = 4) → y = x ∨ y = -x)) := 
  sorry

end NUMINAMATH_GPT_asymptotes_of_hyperbola_l1632_163235


namespace NUMINAMATH_GPT_value_of_phi_l1632_163298

theorem value_of_phi { φ : ℝ } (hφ1 : 0 < φ) (hφ2 : φ < π)
  (symm_condition : ∃ k : ℤ, -π / 8 + φ = k * π + π / 2) : φ = 3 * π / 4 := 
by 
  sorry

end NUMINAMATH_GPT_value_of_phi_l1632_163298


namespace NUMINAMATH_GPT_dragons_total_games_l1632_163247

theorem dragons_total_games (y x : ℕ) (h1 : x = 60 * y / 100) (h2 : (x + 8) = 55 * (y + 11) / 100) : y + 11 = 50 :=
by
  sorry

end NUMINAMATH_GPT_dragons_total_games_l1632_163247


namespace NUMINAMATH_GPT_factorize_difference_of_squares_l1632_163290

theorem factorize_difference_of_squares (y : ℝ) : y^2 - 4 = (y + 2) * (y - 2) := 
by
  sorry

end NUMINAMATH_GPT_factorize_difference_of_squares_l1632_163290


namespace NUMINAMATH_GPT_union_complement_eq_l1632_163219

open Set

variable (U A B : Set ℕ)

def complement (U A : Set ℕ) : Set ℕ := { x | x ∈ U ∧ x ∉ A }

theorem union_complement_eq (U A B : Set ℕ) (hU : U = {0, 1, 2, 3, 4}) (hA : A = {1, 2, 3}) (hB : B = {2, 4}) :
  (complement U A) ∪ B = {0, 2, 4} :=
by
  rw [hU, hA, hB]
  sorry

end NUMINAMATH_GPT_union_complement_eq_l1632_163219


namespace NUMINAMATH_GPT_min_z_value_l1632_163282

theorem min_z_value (x y z : ℝ) (h1 : 2 * x + y = 1) (h2 : z = 4 ^ x + 2 ^ y) : z ≥ 2 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_min_z_value_l1632_163282


namespace NUMINAMATH_GPT_arctan_sum_pi_div_two_l1632_163294

noncomputable def arctan_sum : Real :=
  Real.arctan (3 / 4) + Real.arctan (4 / 3)

theorem arctan_sum_pi_div_two : arctan_sum = Real.pi / 2 := 
by sorry

end NUMINAMATH_GPT_arctan_sum_pi_div_two_l1632_163294


namespace NUMINAMATH_GPT_volume_of_A_is_2800_l1632_163289

-- Define the dimensions of the fishbowl and water heights
def fishbowl_side_length : ℝ := 20
def height_with_A : ℝ := 16
def height_without_A : ℝ := 9

-- Compute the volume of water with and without object (A)
def volume_with_A : ℝ := fishbowl_side_length ^ 2 * height_with_A
def volume_without_A : ℝ := fishbowl_side_length ^ 2 * height_without_A

-- The volume of object (A)
def volume_A : ℝ := volume_with_A - volume_without_A

-- Prove that this volume is 2800 cubic centimeters
theorem volume_of_A_is_2800 :
  volume_A = 2800 := by
  sorry

end NUMINAMATH_GPT_volume_of_A_is_2800_l1632_163289


namespace NUMINAMATH_GPT_find_m_of_transformed_point_eq_l1632_163279

theorem find_m_of_transformed_point_eq (m : ℝ) (h : m + 1 = 5) : m = 4 :=
by
  sorry

end NUMINAMATH_GPT_find_m_of_transformed_point_eq_l1632_163279


namespace NUMINAMATH_GPT_proof_f_derivative_neg1_l1632_163240

noncomputable def f (x : ℝ) (a b c : ℝ) : ℝ :=
  a * x ^ 4 + b * x ^ 2 + c

noncomputable def f_derivative (x : ℝ) (a b : ℝ) : ℝ :=
  4 * a * x ^ 3 + 2 * b * x

theorem proof_f_derivative_neg1
  (a b c : ℝ) (h : f_derivative 1 a b = 2) :
  f_derivative (-1) a b = -2 :=
by
  sorry

end NUMINAMATH_GPT_proof_f_derivative_neg1_l1632_163240


namespace NUMINAMATH_GPT_toys_ratio_l1632_163215

-- Definitions of given conditions
variables (rabbits : ℕ) (toys_monday toys_wednesday toys_friday toys_saturday total_toys : ℕ)
variables (h_rabbits : rabbits = 16)
variables (h_toys_monday : toys_monday = 6)
variables (h_toys_friday : toys_friday = 4 * toys_monday)
variables (h_toys_saturday : toys_saturday = toys_wednesday / 2)
variables (h_total_toys : total_toys = rabbits * 3)

-- Define the Lean theorem to state the problem conditions and prove the ratio
theorem toys_ratio (h : toys_monday + toys_wednesday + toys_friday + toys_saturday = total_toys) :
  (if (2 * toys_wednesday = 12) then 2 else 1) = 2 :=
by 
  sorry

end NUMINAMATH_GPT_toys_ratio_l1632_163215


namespace NUMINAMATH_GPT_range_of_a_l1632_163231

variable (a : ℝ)

def p (a : ℝ) : Prop := 3/2 < a ∧ a < 5/2
def q (a : ℝ) : Prop := 2 ≤ a ∧ a ≤ 4

theorem range_of_a (h₁ : ¬(p a ∧ q a)) (h₂ : p a ∨ q a) : (3/2 < a ∧ a < 2) ∨ (5/2 ≤ a ∧ a ≤ 4) :=
sorry

end NUMINAMATH_GPT_range_of_a_l1632_163231


namespace NUMINAMATH_GPT_value_of_f_neg2_l1632_163271

def f (x : ℤ) : ℤ := x^2 - 3 * x + 1

theorem value_of_f_neg2 : f (-2) = 11 := by
  sorry

end NUMINAMATH_GPT_value_of_f_neg2_l1632_163271


namespace NUMINAMATH_GPT_fixed_point_of_inverse_l1632_163229

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a ^ (x - 1) + 4

theorem fixed_point_of_inverse (a : ℝ) (ha_pos : 0 < a) (ha_ne_one : a ≠ 1) :
  f a (5) = 1 :=
by
  unfold f
  sorry

end NUMINAMATH_GPT_fixed_point_of_inverse_l1632_163229


namespace NUMINAMATH_GPT_tangent_line_eq_l1632_163222

theorem tangent_line_eq (x y: ℝ):
  (x^2 + y^2 = 4) → ((2, 3) = (x, y)) →
  (x = 2 ∨ 5 * x - 12 * y + 26 = 0) :=
by
  sorry

end NUMINAMATH_GPT_tangent_line_eq_l1632_163222


namespace NUMINAMATH_GPT_find_number_of_folders_l1632_163284

theorem find_number_of_folders :
  let price_pen := 1
  let price_notebook := 3
  let price_folder := 5
  let pens_bought := 3
  let notebooks_bought := 4
  let bill := 50
  let change := 25
  let total_cost_pens_notebooks := pens_bought * price_pen + notebooks_bought * price_notebook
  let amount_spent := bill - change
  let amount_spent_on_folders := amount_spent - total_cost_pens_notebooks
  let number_of_folders := amount_spent_on_folders / price_folder
  number_of_folders = 2 :=
by
  sorry

end NUMINAMATH_GPT_find_number_of_folders_l1632_163284


namespace NUMINAMATH_GPT_points_lie_on_parabola_l1632_163257

noncomputable def lies_on_parabola (t : ℝ) : Prop :=
  let x := Real.cos t ^ 2
  let y := Real.sin t * Real.cos t
  y ^ 2 = x * (1 - x)

-- Statement to prove
theorem points_lie_on_parabola : ∀ t : ℝ, lies_on_parabola t :=
by
  intro t
  sorry

end NUMINAMATH_GPT_points_lie_on_parabola_l1632_163257


namespace NUMINAMATH_GPT_quadratic_real_roots_k_eq_one_l1632_163225

theorem quadratic_real_roots_k_eq_one 
  (k : ℕ) 
  (h_nonneg : k ≥ 0) 
  (h_real_roots : ∃ x : ℝ, k * x^2 - 2 * x + 1 = 0) : 
  k = 1 := 
sorry

end NUMINAMATH_GPT_quadratic_real_roots_k_eq_one_l1632_163225


namespace NUMINAMATH_GPT_baskets_picked_l1632_163293

theorem baskets_picked
  (B : ℕ) -- How many baskets did her brother pick?
  (S : ℕ := 15) -- Each basket contains 15 strawberries
  (H1 : (8 * B * S) + (B * S) + ((8 * B * S) - 93) = 4 * 168) -- Total number of strawberries when divided equally
  (H2 : S = 15) -- Number of strawberries in each basket
: B = 3 :=
sorry

end NUMINAMATH_GPT_baskets_picked_l1632_163293


namespace NUMINAMATH_GPT_non_empty_subsets_count_l1632_163200

def odd_set : Finset ℕ := {1, 3, 5, 7, 9}
def even_set : Finset ℕ := {2, 4, 6, 8}

noncomputable def num_non_empty_subsets_odd : ℕ := 2 ^ odd_set.card - 1
noncomputable def num_non_empty_subsets_even : ℕ := 2 ^ even_set.card - 1

theorem non_empty_subsets_count :
  num_non_empty_subsets_odd + num_non_empty_subsets_even = 46 :=
by sorry

end NUMINAMATH_GPT_non_empty_subsets_count_l1632_163200


namespace NUMINAMATH_GPT_variance_scaled_l1632_163217

-- Let V represent the variance of the set of data
def original_variance : ℝ := 3
def scale_factor : ℝ := 3

-- Prove that the new variance is 27 
theorem variance_scaled (V : ℝ) (s : ℝ) (hV : V = 3) (hs : s = 3) : s^2 * V = 27 := by
  sorry

end NUMINAMATH_GPT_variance_scaled_l1632_163217


namespace NUMINAMATH_GPT_cannot_achieve_1_5_percent_salt_solution_l1632_163220

-- Define the initial concentrations and volumes
def initial_state (V1 V2 : ℝ) (C1 C2 : ℝ) : Prop :=
  V1 = 1 ∧ C1 = 0 ∧ V2 = 1 ∧ C2 = 0.02

-- Define the transfer and mixing operation
noncomputable def transfer_and_mix (V1_old V2_old C1_old C2_old : ℝ) (amount_to_transfer : ℝ)
  (new_V1 new_V2 new_C1 new_C2 : ℝ) : Prop :=
  amount_to_transfer ≤ V2_old ∧
  new_V1 = V1_old + amount_to_transfer ∧
  new_V2 = V2_old - amount_to_transfer ∧
  new_C1 = (V1_old * C1_old + amount_to_transfer * C2_old) / new_V1 ∧
  new_C2 = (V2_old * C2_old - amount_to_transfer * C2_old) / new_V2

-- Prove that it is impossible to achieve a 1.5% salt concentration in container 1
theorem cannot_achieve_1_5_percent_salt_solution :
  ∀ V1 V2 C1 C2, initial_state V1 V2 C1 C2 →
  ¬ ∃ V1' V2' C1' C2', transfer_and_mix V1 V2 C1 C2 0.5 V1' V2' C1' C2' ∧ C1' = 0.015 :=
by
  intros
  sorry

end NUMINAMATH_GPT_cannot_achieve_1_5_percent_salt_solution_l1632_163220


namespace NUMINAMATH_GPT_range_of_a_l1632_163291

theorem range_of_a (p q : Set ℝ) (a : ℝ) (h1 : ∀ x, 2 * x^2 - 3 * x + 1 ≤ 0 → x ∈ p) 
                             (h2 : ∀ x, x^2 - (2 * a + 1) * x + a^2 + a ≤ 0 → x ∈ q)
                             (h3 : ∀ x, p x → q x ∧ ∃ x, ¬p x ∧ q x) : 
  0 ≤ a ∧ a ≤ 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1632_163291


namespace NUMINAMATH_GPT_usual_time_is_180_l1632_163246

variable (D S1 T : ℝ)

-- Conditions
def usual_time : Prop := T = D / S1
def reduced_speed : Prop := ∃ S2 : ℝ, S2 = 5 / 6 * S1
def total_delay : Prop := 6 + 12 + 18 = 36
def total_time_reduced_speed_stops : Prop := ∃ T' : ℝ, T' + 36 = 6 / 5 * T
def time_equation : Prop := T + 36 = 6 / 5 * T

-- Proof problem statement
theorem usual_time_is_180 (h1 : usual_time D S1 T)
                          (h2 : reduced_speed S1)
                          (h3 : total_delay)
                          (h4 : total_time_reduced_speed_stops T)
                          (h5 : time_equation T) :
                          T = 180 := by
  sorry

end NUMINAMATH_GPT_usual_time_is_180_l1632_163246


namespace NUMINAMATH_GPT_value_of_x_squared_plus_9y_squared_l1632_163261

theorem value_of_x_squared_plus_9y_squared (x y : ℝ)
  (h1 : x + 3 * y = 5)
  (h2 : x * y = -8) : x^2 + 9 * y^2 = 73 :=
by
  sorry

end NUMINAMATH_GPT_value_of_x_squared_plus_9y_squared_l1632_163261


namespace NUMINAMATH_GPT_extremum_condition_l1632_163281

noncomputable def quadratic_polynomial (a x : ℝ) : ℝ := a * x^2 + x + 1

theorem extremum_condition (a : ℝ) :
  (∃ x : ℝ, ∃ f' : ℝ → ℝ, 
     (f' = (fun x => 2 * a * x + 1)) ∧ 
     (f' x = 0) ∧ 
     (∃ (f'' : ℝ → ℝ), (f'' = (fun x => 2 * a)) ∧ (f'' x ≠ 0))) ↔ a < 0 := 
sorry

end NUMINAMATH_GPT_extremum_condition_l1632_163281


namespace NUMINAMATH_GPT_number_of_draw_matches_eq_points_difference_l1632_163236

-- Definitions based on the conditions provided
def teams : ℕ := 16
def matches_per_round : ℕ := 8
def rounds : ℕ := 16
def total_points : ℕ := 222
def total_matches : ℕ := matches_per_round * rounds
def hypothetical_points : ℕ := total_matches * 2
def points_difference : ℕ := hypothetical_points - total_points

-- Theorem stating the equivalence to be proved
theorem number_of_draw_matches_eq_points_difference : 
  points_difference = 34 := 
by
  sorry

end NUMINAMATH_GPT_number_of_draw_matches_eq_points_difference_l1632_163236


namespace NUMINAMATH_GPT_joe_egg_count_l1632_163205

theorem joe_egg_count : 
  let clubhouse : ℕ := 12
  let park : ℕ := 5
  let townhall : ℕ := 3
  clubhouse + park + townhall = 20 :=
by
  sorry

end NUMINAMATH_GPT_joe_egg_count_l1632_163205


namespace NUMINAMATH_GPT_remainder_when_divided_by_18_l1632_163245

theorem remainder_when_divided_by_18 (n : ℕ) (r3 r6 r9 : ℕ)
  (hr3 : r3 = n % 3)
  (hr6 : r6 = n % 6)
  (hr9 : r9 = n % 9)
  (h_sum : r3 + r6 + r9 = 15) :
  n % 18 = 17 := sorry

end NUMINAMATH_GPT_remainder_when_divided_by_18_l1632_163245


namespace NUMINAMATH_GPT_violet_has_27_nails_l1632_163227

def nails_tickletoe : ℕ := 12  -- T
def nails_violet : ℕ := 2 * nails_tickletoe + 3

theorem violet_has_27_nails (h : nails_tickletoe + nails_violet = 39) : nails_violet = 27 :=
by
  sorry

end NUMINAMATH_GPT_violet_has_27_nails_l1632_163227


namespace NUMINAMATH_GPT_Dabbie_spends_99_dollars_l1632_163274

noncomputable def total_cost_turkeys (w1 w2 w3 w4 : ℝ) (cost_per_kg : ℝ) : ℝ :=
  (w1 + w2 + w3 + w4) * cost_per_kg

theorem Dabbie_spends_99_dollars :
  let w1 := 6
  let w2 := 9
  let w3 := 2 * w2
  let w4 := (w1 + w2 + w3) / 2
  let cost_per_kg := 2
  total_cost_turkeys w1 w2 w3 w4 cost_per_kg = 99 := 
by
  sorry

end NUMINAMATH_GPT_Dabbie_spends_99_dollars_l1632_163274


namespace NUMINAMATH_GPT_common_difference_is_minus_3_l1632_163239

variable (a_n : ℕ → ℤ) (a1 d : ℤ)

-- Definitions expressing the conditions of the problem
def arithmetic_prog : Prop := ∀ (n : ℕ), a_n n = a1 + (n - 1) * d

def condition1 : Prop := a1 + (a1 + 6 * d) = -8

def condition2 : Prop := a1 + d = 2

-- The statement we need to prove
theorem common_difference_is_minus_3 :
  arithmetic_prog a_n a1 d ∧ condition1 a1 d ∧ condition2 a1 d → d = -3 :=
by {
  -- The proof would go here
  sorry
}

end NUMINAMATH_GPT_common_difference_is_minus_3_l1632_163239


namespace NUMINAMATH_GPT_john_bought_more_than_ray_l1632_163233

variable (R_c R_d M_c M_d J_c J_d : ℕ)

-- Define the conditions
def conditions : Prop :=
  (R_c = 10) ∧
  (R_d = 3) ∧
  (M_c = R_c + 6) ∧
  (M_d = R_d + 1) ∧
  (J_c = M_c + 5) ∧
  (J_d = M_d + 2)

-- Define the question
def john_more_chickens_and_ducks (J_c R_c J_d R_d : ℕ) : ℕ :=
  (J_c - R_c) + (J_d - R_d)

-- The proof problem statement
theorem john_bought_more_than_ray :
  conditions R_c R_d M_c M_d J_c J_d → john_more_chickens_and_ducks J_c R_c J_d R_d = 14 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_john_bought_more_than_ray_l1632_163233


namespace NUMINAMATH_GPT_range_of_a_l1632_163276

variable (f : ℝ → ℝ)
variable (a : ℝ)

def is_even (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def is_increasing_on_nonneg (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 ≤ x → x ≤ y → f x ≤ f y

def holds_on_interval (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x, (1/2) ≤ x ∧ x ≤ 1 → f (a*x + 1) ≤ f (x - 2)

theorem range_of_a (h1 : is_even f)
                   (h2 : is_increasing_on_nonneg f)
                   (h3 : holds_on_interval f a) :
  -2 ≤ a ∧ a ≤ 0 := 
sorry

end NUMINAMATH_GPT_range_of_a_l1632_163276


namespace NUMINAMATH_GPT_number_eq_180_l1632_163223

theorem number_eq_180 (x : ℝ) (h : 64 + 5 * 12 / (x / 3) = 65) : x = 180 :=
sorry

end NUMINAMATH_GPT_number_eq_180_l1632_163223


namespace NUMINAMATH_GPT_largest_among_four_l1632_163283

theorem largest_among_four (a b : ℝ) (ha : a > 0) (hb : b < 0) :
  max (max a (max (a + b) (a - b))) (ab) = a - b :=
by {
  sorry
}

end NUMINAMATH_GPT_largest_among_four_l1632_163283


namespace NUMINAMATH_GPT_range_of_m_l1632_163228

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := 2 * m * x ^ 2 - 2 * (4 - m) * x + 1
def g (m : ℝ) (x : ℝ) : ℝ := m * x

theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, f m x > 0 ∨ g m x > 0) ↔ (0 < m ∧ m < 8) :=
sorry

end NUMINAMATH_GPT_range_of_m_l1632_163228


namespace NUMINAMATH_GPT_calculate_expression_l1632_163202

theorem calculate_expression : 
  (1 / 2) ^ (-2: ℤ) - 3 * Real.tan (Real.pi / 6) - abs (Real.sqrt 3 - 2) = 2 := 
by
  sorry

end NUMINAMATH_GPT_calculate_expression_l1632_163202


namespace NUMINAMATH_GPT_maximize_probability_l1632_163266

def numbers_list : List Int := [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]

def pairs_summing_to_12 (l : List Int) : List (Int × Int) :=
  List.filter (fun (p : Int × Int) => p.1 + p.2 = 12) (List.product l l)

def distinct_pairs (pairs : List (Int × Int)) : List (Int × Int) :=
  List.filter (fun (p : Int × Int) => p.1 ≠ p.2) pairs

def valid_pairs (l : List Int) : List (Int × Int) :=
  distinct_pairs (pairs_summing_to_12 l)

def count_valid_pairs (l : List Int) : Nat :=
  List.length (valid_pairs l)

def remove_and_check (x : Int) : List Int :=
  List.erase numbers_list x

theorem maximize_probability :
  ∀ x : Int, count_valid_pairs (remove_and_check 6) ≥ count_valid_pairs (remove_and_check x) :=
sorry

end NUMINAMATH_GPT_maximize_probability_l1632_163266


namespace NUMINAMATH_GPT_one_fourth_of_six_point_eight_eq_seventeen_tenths_l1632_163295

theorem one_fourth_of_six_point_eight_eq_seventeen_tenths :  (6.8 / 4) = (17 / 10) :=
by
  -- Skipping proof
  sorry

end NUMINAMATH_GPT_one_fourth_of_six_point_eight_eq_seventeen_tenths_l1632_163295


namespace NUMINAMATH_GPT_c_completes_in_three_days_l1632_163214

variables (r_A r_B r_C : ℝ)
variables (h1 : r_A + r_B = 1/3)
variables (h2 : r_B + r_C = 1/3)
variables (h3 : r_A + r_C = 2/3)

theorem c_completes_in_three_days : 1 / r_C = 3 :=
by sorry

end NUMINAMATH_GPT_c_completes_in_three_days_l1632_163214


namespace NUMINAMATH_GPT_total_flour_amount_l1632_163297

-- Define the initial amount of flour in the bowl
def initial_flour : ℝ := 2.75

-- Define the amount of flour added by the baker
def added_flour : ℝ := 0.45

-- Prove that the total amount of flour is 3.20 kilograms
theorem total_flour_amount : initial_flour + added_flour = 3.20 :=
by
  sorry

end NUMINAMATH_GPT_total_flour_amount_l1632_163297


namespace NUMINAMATH_GPT_find_p_l1632_163272

noncomputable def parabola_focus (p : ℝ) : ℝ × ℝ :=
  (p / 2, 0)

def hyperbola_focus : ℝ × ℝ :=
  (2, 0)

theorem find_p (p : ℝ) (h : p > 0) (hp : parabola_focus p = hyperbola_focus) : p = 4 :=
by
  sorry

end NUMINAMATH_GPT_find_p_l1632_163272


namespace NUMINAMATH_GPT_fraction_of_students_with_mentor_l1632_163273

theorem fraction_of_students_with_mentor (s n : ℕ) (h : n / 2 = s / 3) :
  (n / 2 + s / 3 : ℚ) / (n + s : ℚ) = 2 / 5 := by
  sorry

end NUMINAMATH_GPT_fraction_of_students_with_mentor_l1632_163273


namespace NUMINAMATH_GPT_line_passes_through_2nd_and_4th_quadrants_l1632_163232

theorem line_passes_through_2nd_and_4th_quadrants (b : ℝ) :
  (∀ x : ℝ, x > 0 → -2 * x + b < 0) ∧ (∀ x : ℝ, x < 0 → -2 * x + b > 0) :=
by
  sorry

end NUMINAMATH_GPT_line_passes_through_2nd_and_4th_quadrants_l1632_163232
