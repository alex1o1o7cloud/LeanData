import Mathlib

namespace NUMINAMATH_GPT_find_P_l1112_111250

theorem find_P (P Q R S : ℕ) (h1: P ≠ Q) (h2: R ≠ S) (h3: P * Q = 72) (h4: R * S = 72) (h5: P - Q = R + S) :
  P = 18 := 
  sorry

end NUMINAMATH_GPT_find_P_l1112_111250


namespace NUMINAMATH_GPT_find_N_l1112_111227

theorem find_N (a b c : ℤ) (N : ℤ)
  (h1 : a + b + c = 105)
  (h2 : a - 5 = N)
  (h3 : b + 10 = N)
  (h4 : 5 * c = N) : 
  N = 50 :=
by
  sorry

end NUMINAMATH_GPT_find_N_l1112_111227


namespace NUMINAMATH_GPT_area_of_triangle_is_24_l1112_111270

open Real

-- Define the coordinates of the vertices
def A : ℝ × ℝ := (-2, 3)
def B : ℝ × ℝ := (6, 1)
def C : ℝ × ℝ := (10, 6)

-- Define the vectors from point C
def v : ℝ × ℝ := (A.1 - C.1, A.2 - C.2)
def w : ℝ × ℝ := (B.1 - C.1, B.2 - C.2)

-- Define the determinant for the parallelogram area
def parallelogram_area : ℝ :=
  abs (v.1 * w.2 - v.2 * w.1)

-- Prove the area of the triangle
theorem area_of_triangle_is_24 : (parallelogram_area / 2) = 24 := by
  sorry

end NUMINAMATH_GPT_area_of_triangle_is_24_l1112_111270


namespace NUMINAMATH_GPT_count_coin_distributions_l1112_111288

-- Mathematical conditions
def coin_denominations : Finset ℕ := {1, 2, 3, 5}
def number_of_boys : ℕ := 6

-- Theorem statement
theorem count_coin_distributions : (coin_denominations.card ^ number_of_boys) = 4096 :=
by
  sorry

end NUMINAMATH_GPT_count_coin_distributions_l1112_111288


namespace NUMINAMATH_GPT_find_a_pow_b_l1112_111212

theorem find_a_pow_b (a b : ℝ) (h : (a - 2)^2 + |b + 1| = 0) : a^b = 1 / 2 := 
sorry

end NUMINAMATH_GPT_find_a_pow_b_l1112_111212


namespace NUMINAMATH_GPT_factorize_poly1_min_value_poly2_l1112_111251

-- Define the polynomials
def poly1 := fun (x : ℝ) => x^2 + 2 * x - 3
def factored_poly1 := fun (x : ℝ) => (x - 1) * (x + 3)

def poly2 := fun (x : ℝ) => x^2 + 4 * x + 5
def min_value := 1

-- State the theorems without providing proofs
theorem factorize_poly1 : ∀ x : ℝ, poly1 x = factored_poly1 x := 
by { sorry }

theorem min_value_poly2 : ∀ x : ℝ, poly2 x ≥ min_value := 
by { sorry }

end NUMINAMATH_GPT_factorize_poly1_min_value_poly2_l1112_111251


namespace NUMINAMATH_GPT_smallest_value_x_l1112_111228

theorem smallest_value_x : 
  (∃ x : ℝ, ((5*x - 20)/(4*x - 5))^2 + ((5*x - 20)/(4*x - 5)) = 6 ∧ 
  (∀ y : ℝ, ((5*y - 20)/(4*y - 5))^2 + ((5*y - 20)/(4*y - 5)) = 6 → x ≤ y)) → 
  x = 35 / 17 :=
by 
  sorry

end NUMINAMATH_GPT_smallest_value_x_l1112_111228


namespace NUMINAMATH_GPT_num_perfect_square_factors_of_450_l1112_111225

theorem num_perfect_square_factors_of_450 :
  ∃ n : ℕ, n = 4 ∧ ∀ d : ℕ, d ∣ 450 → (∃ k : ℕ, d = k * k) → d = 1 ∨ d = 25 ∨ d = 9 ∨ d = 225 :=
by
  sorry

end NUMINAMATH_GPT_num_perfect_square_factors_of_450_l1112_111225


namespace NUMINAMATH_GPT_ed_marbles_l1112_111267

theorem ed_marbles (doug_initial_marbles : ℕ) (marbles_lost : ℕ) (ed_doug_difference : ℕ) 
  (h1 : doug_initial_marbles = 22) (h2 : marbles_lost = 3) (h3 : ed_doug_difference = 5) : 
  (doug_initial_marbles + ed_doug_difference) = 27 :=
by
  sorry

end NUMINAMATH_GPT_ed_marbles_l1112_111267


namespace NUMINAMATH_GPT_percentage_increase_l1112_111273

-- defining the given values
def Z := 150
def total := 555
def x_from_y (Y : ℝ) := 1.25 * Y

-- defining the condition that x gets 25% more than y and z out of 555 is Rs. 150
def condition1 (X Y : ℝ) := X = x_from_y Y
def condition2 (X Y : ℝ) := X + Y + Z = total

-- theorem to prove
theorem percentage_increase (Y : ℝ) :
  condition1 (x_from_y Y) Y →
  condition2 (x_from_y Y) Y →
  ((Y - Z) / Z) * 100 = 20 :=
by
  sorry

end NUMINAMATH_GPT_percentage_increase_l1112_111273


namespace NUMINAMATH_GPT_inequality_solution_set_l1112_111295

theorem inequality_solution_set : 
  {x : ℝ | -x^2 + 4*x + 5 < 0} = {x : ℝ | x < -1 ∨ x > 5} := 
by
  sorry

end NUMINAMATH_GPT_inequality_solution_set_l1112_111295


namespace NUMINAMATH_GPT_compound_interest_rate_l1112_111215

theorem compound_interest_rate
  (P : ℝ) (r : ℝ) :
  (3000 = P * (1 + r / 100)^3) →
  (3600 = P * (1 + r / 100)^4) →
  r = 20 :=
by
  sorry

end NUMINAMATH_GPT_compound_interest_rate_l1112_111215


namespace NUMINAMATH_GPT_option_c_same_function_l1112_111269

theorem option_c_same_function :
  ∀ (x : ℝ), x ≠ 0 → (1 + (1 / x) = u ↔ u = 1 + (1 / (1 + 1 / x))) :=
by sorry

end NUMINAMATH_GPT_option_c_same_function_l1112_111269


namespace NUMINAMATH_GPT_box_area_ratio_l1112_111263

theorem box_area_ratio 
  (l w h : ℝ)
  (V : l * w * h = 5184)
  (A1 : w * h = (1/2) * l * w)
  (A2 : l * h = 288):
  (l * w) / (l * h) = 3 / 2 := 
by
  sorry

end NUMINAMATH_GPT_box_area_ratio_l1112_111263


namespace NUMINAMATH_GPT_exists_3x3_grid_l1112_111237

theorem exists_3x3_grid : 
  ∃ (a₁₂ a₂₁ a₂₃ a₃₂ : ℕ), 
  a₁₂ ≠ a₂₁ ∧ a₁₂ ≠ a₂₃ ∧ a₁₂ ≠ a₃₂ ∧ 
  a₂₁ ≠ a₂₃ ∧ a₂₁ ≠ a₃₂ ∧ 
  a₂₃ ≠ a₃₂ ∧ 
  a₁₂ ≤ 25 ∧ a₂₁ ≤ 25 ∧ a₂₃ ≤ 25 ∧ a₃₂ ≤ 25 ∧ 
  a₁₂ > 0 ∧ a₂₁ > 0 ∧ a₂₃ > 0 ∧ a₃₂ > 0 ∧
  (∃ (a₁₁ a₁₃ a₃₁ a₃₃ a₂₂ : ℕ),
  a₁₁ ≤ 25 ∧ a₁₃ ≤ 25 ∧ a₃₁ ≤ 25 ∧ a₃₃ ≤ 25 ∧ a₂₂ ≤ 25 ∧
  a₁₁ > 0 ∧ a₁₃ > 0 ∧ a₃₁ > 0 ∧ a₃₃ > 0 ∧ a₂₂ > 0 ∧
  a₁₁ ≠ a₁₂ ∧ a₁₁ ≠ a₂₁ ∧ a₁₁ ≠ a₁₃ ∧ a₁₁ ≠ a₃₁ ∧ 
  a₁₃ ≠ a₃₃ ∧ a₁₃ ≠ a₂₃ ∧ a₂₁ ≠ a₃₁ ∧ a₃₁ ≠ a₃₂ ∧ 
  a₃₃ ≠ a₂₂ ∧ a₃₃ ≠ a₃₂ ∧ a₂₂ = 1 ∧
  (a₁₂ % a₂₂ = 0 ∨ a₂₂ % a₁₂ = 0) ∧
  (a₂₁ % a₂₂ = 0 ∨ a₂₂ % a₂₁ = 0) ∧
  (a₂₃ % a₂₂ = 0 ∨ a₂₂ % a₂₃ = 0) ∧
  (a₃₂ % a₂₂ = 0 ∨ a₂₂ % a₃₂ = 0) ∧
  (a₁₁ % a₁₂ = 0 ∨ a₁₂ % a₁₁ = 0) ∧
  (a₁₁ % a₂₁ = 0 ∨ a₂₁ % a₁₁ = 0) ∧
  (a₁₃ % a₁₂ = 0 ∨ a₁₂ % a₁₃ = 0) ∧
  (a₁₃ % a₂₃ = 0 ∨ a₂₃ % a₁₃ = 0) ∧
  (a₃₁ % a₂₁ = 0 ∨ a₂₁ % a₃₁ = 0) ∧
  (a₃₁ % a₃₂ = 0 ∨ a₃₂ % a₃₁ = 0) ∧
  (a₃₃ % a₂₃ = 0 ∨ a₂₃ % a₃₃ = 0) ∧
  (a₃₃ % a₃₂ = 0 ∨ a₃₂ % a₃₃ = 0)) 
  :=
sorry

end NUMINAMATH_GPT_exists_3x3_grid_l1112_111237


namespace NUMINAMATH_GPT_correct_statements_l1112_111243

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (2 * x - 13 / 4 * Real.pi)

theorem correct_statements :
    (f (Real.pi / 8) = 0) ∧ 
    (∀ x, 2 * Real.sin (2 * (x - 5 / 8 * Real.pi)) = f x) :=
by
  sorry

end NUMINAMATH_GPT_correct_statements_l1112_111243


namespace NUMINAMATH_GPT_b_95_mod_49_l1112_111289

-- Define the sequence b_n
def b (n : ℕ) : ℕ := 7^n + 9^n

-- Goal: Prove that the remainder when b 95 is divided by 49 is 28
theorem b_95_mod_49 : b 95 % 49 = 28 := 
by
  sorry

end NUMINAMATH_GPT_b_95_mod_49_l1112_111289


namespace NUMINAMATH_GPT_unpainted_cubes_count_l1112_111202

/- Definitions of the conditions -/
def total_cubes : ℕ := 6 * 6 * 6
def painted_faces_per_face : ℕ := 4
def total_faces : ℕ := 6
def painted_faces : ℕ := painted_faces_per_face * total_faces
def overlapped_painted_faces : ℕ := 4 -- Each center four squares on one face corresponds to a center square on the opposite face.
def unique_painted_cubes : ℕ := painted_faces / 2

/- Lean Theorem statement that corresponds to proving the question asked in the problem -/
theorem unpainted_cubes_count : 
  total_cubes - unique_painted_cubes = 208 :=
  by
    sorry

end NUMINAMATH_GPT_unpainted_cubes_count_l1112_111202


namespace NUMINAMATH_GPT_odd_number_diff_squares_unique_l1112_111223

theorem odd_number_diff_squares_unique (n : ℕ) (h : 0 < n) : 
  ∃! (x y : ℤ), (2 * n + 1) = x^2 - y^2 :=
by {
  sorry
}

end NUMINAMATH_GPT_odd_number_diff_squares_unique_l1112_111223


namespace NUMINAMATH_GPT_value_of_a1_l1112_111233

def seq (a : ℕ → ℚ) (a_8 : ℚ) : Prop :=
  ∀ n : ℕ, (a (n + 1) = 1 / (1 - a n)) ∧ a 8 = 2

theorem value_of_a1 (a : ℕ → ℚ) (h : seq a 2) : a 1 = 1 / 2 :=
  sorry

end NUMINAMATH_GPT_value_of_a1_l1112_111233


namespace NUMINAMATH_GPT_smallest_positive_x_for_palindrome_l1112_111232

def is_palindrome (n : ℕ) : Prop :=
  let s := n.digits 10
  s = s.reverse

theorem smallest_positive_x_for_palindrome :
  ∃ x : ℕ, x > 0 ∧ is_palindrome (x + 1234) ∧ (∀ y : ℕ, y > 0 → is_palindrome (y + 1234) → x ≤ y) ∧ x = 97 := 
sorry

end NUMINAMATH_GPT_smallest_positive_x_for_palindrome_l1112_111232


namespace NUMINAMATH_GPT_village_population_l1112_111244

-- Defining the variables and the condition
variable (P : ℝ) (h : 0.9 * P = 36000)

-- Statement of the theorem to prove
theorem village_population : P = 40000 :=
by sorry

end NUMINAMATH_GPT_village_population_l1112_111244


namespace NUMINAMATH_GPT_perimeter_of_large_rectangle_l1112_111259

-- We are bringing in all necessary mathematical libraries, no specific submodules needed.
theorem perimeter_of_large_rectangle
  (small_rectangle_longest_side : ℝ)
  (number_of_small_rectangles : ℕ)
  (length_of_large_rectangle : ℝ)
  (height_of_large_rectangle : ℝ)
  (perimeter_of_large_rectangle : ℝ) :
  small_rectangle_longest_side = 10 ∧ number_of_small_rectangles = 9 →
  length_of_large_rectangle = 2 * small_rectangle_longest_side →
  height_of_large_rectangle = 5 * (small_rectangle_longest_side / 2) →
  perimeter_of_large_rectangle = 2 * (length_of_large_rectangle + height_of_large_rectangle) →
  perimeter_of_large_rectangle = 76 := by
  sorry

end NUMINAMATH_GPT_perimeter_of_large_rectangle_l1112_111259


namespace NUMINAMATH_GPT_profit_percentage_is_20_l1112_111224

noncomputable def selling_price : ℝ := 200
noncomputable def cost_price : ℝ := 166.67
noncomputable def profit : ℝ := selling_price - cost_price

theorem profit_percentage_is_20 :
  (profit / cost_price) * 100 = 20 := by
  sorry

end NUMINAMATH_GPT_profit_percentage_is_20_l1112_111224


namespace NUMINAMATH_GPT_mod_2021_2022_2023_2024_eq_zero_mod_7_l1112_111200

theorem mod_2021_2022_2023_2024_eq_zero_mod_7 :
  (2021 * 2022 * 2023 * 2024) % 7 = 0 := by
  sorry

end NUMINAMATH_GPT_mod_2021_2022_2023_2024_eq_zero_mod_7_l1112_111200


namespace NUMINAMATH_GPT_inequality_hold_l1112_111229

theorem inequality_hold (x y z : ℝ) :
  (x^2 + 2 * y^2 + 2 * z^2) / (x^2 + y * z) + 
  (y^2 + 2 * z^2 + 2 * x^2) / (y^2 + z * x) + 
  (z^2 + 2 * x^2 + 2 * y^2) / (z^2 + x * y) > 6 :=
by sorry

end NUMINAMATH_GPT_inequality_hold_l1112_111229


namespace NUMINAMATH_GPT_evaluate_expression_l1112_111268

theorem evaluate_expression (b : ℕ) (h : b = 2) : b^3 * b^4 = 128 :=
by
  -- sorry is used to skip the proof
  sorry

end NUMINAMATH_GPT_evaluate_expression_l1112_111268


namespace NUMINAMATH_GPT_Calvin_mistake_correct_l1112_111281

theorem Calvin_mistake_correct (a : ℕ) : 37 + 31 * a = 37 * 31 + a → a = 37 :=
sorry

end NUMINAMATH_GPT_Calvin_mistake_correct_l1112_111281


namespace NUMINAMATH_GPT_simplify_175_sub_57_sub_43_simplify_128_sub_64_sub_36_simplify_156_sub_49_sub_51_l1112_111240

theorem simplify_175_sub_57_sub_43 : 175 - 57 - 43 = 75 :=
by
  sorry

theorem simplify_128_sub_64_sub_36 : 128 - 64 - 36 = 28 :=
by
  sorry

theorem simplify_156_sub_49_sub_51 : 156 - 49 - 51 = 56 :=
by
  sorry

end NUMINAMATH_GPT_simplify_175_sub_57_sub_43_simplify_128_sub_64_sub_36_simplify_156_sub_49_sub_51_l1112_111240


namespace NUMINAMATH_GPT_total_amount_shared_l1112_111257

-- Define the variables
variables (a b c : ℕ)

-- Define the conditions
axiom condition1 : a = (1 / 3 : ℝ) * (b + c)
axiom condition2 : b = (2 / 7 : ℝ) * (a + c)
axiom condition3 : a = b + 15

-- The proof statement
theorem total_amount_shared : a + b + c = 540 :=
by
  -- We assume these axioms are declared and noncontradictory
  sorry

end NUMINAMATH_GPT_total_amount_shared_l1112_111257


namespace NUMINAMATH_GPT_earnings_correct_l1112_111254

def price_8inch : ℝ := 5
def price_12inch : ℝ := 2.5 * price_8inch
def price_16inch : ℝ := 3 * price_8inch
def price_20inch : ℝ := 4 * price_8inch
def price_24inch : ℝ := 5.5 * price_8inch

noncomputable def earnings_monday : ℝ :=
  3 * price_8inch + 2 * price_12inch + 1 * price_16inch + 2 * price_20inch + 1 * price_24inch

noncomputable def earnings_tuesday : ℝ :=
  5 * price_8inch + 1 * price_12inch + 4 * price_16inch + 2 * price_24inch

noncomputable def earnings_wednesday : ℝ :=
  4 * price_8inch + 3 * price_12inch + 3 * price_16inch + 1 * price_20inch

noncomputable def earnings_thursday : ℝ :=
  2 * price_8inch + 2 * price_12inch + 2 * price_16inch + 1 * price_20inch + 3 * price_24inch

noncomputable def earnings_friday : ℝ :=
  6 * price_8inch + 4 * price_12inch + 2 * price_16inch + 2 * price_20inch

noncomputable def earnings_saturday : ℝ :=
  1 * price_8inch + 3 * price_12inch + 3 * price_16inch + 4 * price_20inch + 2 * price_24inch

noncomputable def earnings_sunday : ℝ :=
  3 * price_8inch + 2 * price_12inch + 4 * price_16inch + 3 * price_20inch + 1 * price_24inch

noncomputable def total_earnings : ℝ :=
  earnings_monday + earnings_tuesday + earnings_wednesday + earnings_thursday + earnings_friday + earnings_saturday + earnings_sunday

theorem earnings_correct : total_earnings = 1025 := by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_earnings_correct_l1112_111254


namespace NUMINAMATH_GPT_change_is_4_25_l1112_111283

-- Define the conditions
def apple_cost : ℝ := 0.75
def amount_paid : ℝ := 5.00

-- State the theorem
theorem change_is_4_25 : amount_paid - apple_cost = 4.25 :=
by
  sorry

end NUMINAMATH_GPT_change_is_4_25_l1112_111283


namespace NUMINAMATH_GPT_tangent_lengths_identity_l1112_111236

theorem tangent_lengths_identity
  (a b c BC AC AB : ℝ)
  (sqrt_a sqrt_b sqrt_c : ℝ)
  (h1 : sqrt_a^2 = a)
  (h2 : sqrt_b^2 = b)
  (h3 : sqrt_c^2 = c) :
  a * BC + c * AB - b * AC = BC * AC * AB :=
sorry

end NUMINAMATH_GPT_tangent_lengths_identity_l1112_111236


namespace NUMINAMATH_GPT_rectangle_width_is_pi_l1112_111211

theorem rectangle_width_is_pi (w : ℝ) (h1 : real_w ≠ 0)
    (h2 : ∀ w, ∃ length, length = 2 * w)
    (h3 : ∀ w, 2 * (length + w) = 6 * w)
    (h4 : 2 * (2 * w + w) = 6 * π) : 
    w = π :=
by {
  sorry -- The proof would go here.
}

end NUMINAMATH_GPT_rectangle_width_is_pi_l1112_111211


namespace NUMINAMATH_GPT_grade_A_probability_l1112_111210

theorem grade_A_probability
  (P_B : ℝ) (P_C : ℝ)
  (hB : P_B = 0.05)
  (hC : P_C = 0.03) :
  1 - P_B - P_C = 0.92 :=
by
  sorry

end NUMINAMATH_GPT_grade_A_probability_l1112_111210


namespace NUMINAMATH_GPT_functional_square_for_all_n_l1112_111271

theorem functional_square_for_all_n (f : ℕ → ℕ) :
  (∀ m n : ℕ, ∃ k : ℕ, (f m + n) * (m + f n) = k ^ 2) ↔ ∃ c : ℕ, ∀ n : ℕ, f n = n + c := 
sorry

end NUMINAMATH_GPT_functional_square_for_all_n_l1112_111271


namespace NUMINAMATH_GPT_trigonometric_identity_l1112_111216

open Real

theorem trigonometric_identity :
  let cos_18 := (sqrt 5 + 1) / 4
  let sin_18 := (sqrt 5 - 1) / 4
  4 * cos_18 ^ 2 - 1 = 1 / (4 * sin_18 ^ 2) :=
by
  let cos_18 := (sqrt 5 + 1) / 4
  let sin_18 := (sqrt 5 - 1) / 4
  sorry

end NUMINAMATH_GPT_trigonometric_identity_l1112_111216


namespace NUMINAMATH_GPT_largest_four_digit_number_l1112_111264

theorem largest_four_digit_number :
  ∃ a b c d : ℕ, 
    9 < 1000 * a + 100 * b + 10 * c + d ∧ 
    1000 * a + 100 * b + 10 * c + d < 10000 ∧ 
    c = a + b ∧ 
    d = b + c ∧ 
    1000 * a + 100 * b + 10 * c + d = 9099 :=
by {
  sorry
}

end NUMINAMATH_GPT_largest_four_digit_number_l1112_111264


namespace NUMINAMATH_GPT_solve_equation_l1112_111276

theorem solve_equation (x : ℝ) (h : x * (x - 3) = 10) : x = 5 ∨ x = -2 :=
by sorry

end NUMINAMATH_GPT_solve_equation_l1112_111276


namespace NUMINAMATH_GPT_paint_cost_decrease_l1112_111292

variables (C P : ℝ)
variable (cost_decrease_canvas : ℝ := 0.40)
variable (total_cost_decrease : ℝ := 0.56)
variable (paint_to_canvas_ratio : ℝ := 4)

theorem paint_cost_decrease (x : ℝ) : 
  P = 4 * C ∧ 
  P * (1 - x) + C * (1 - cost_decrease_canvas) = (1 - total_cost_decrease) * (P + C) → 
  x = 0.60 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_paint_cost_decrease_l1112_111292


namespace NUMINAMATH_GPT_rectangle_area_decrease_l1112_111258

noncomputable def rectangle_area_change (L B : ℝ) (hL : L > 0) (hB : B > 0) : ℝ :=
  let L' := 1.10 * L
  let B' := 0.90 * B
  let A  := L * B
  let A' := L' * B'
  A'

theorem rectangle_area_decrease (L B : ℝ) (hL : L > 0) (hB : B > 0) :
  rectangle_area_change L B hL hB = 0.99 * (L * B) := by
  sorry

end NUMINAMATH_GPT_rectangle_area_decrease_l1112_111258


namespace NUMINAMATH_GPT_hyperbola_center_l1112_111201

theorem hyperbola_center (x1 y1 x2 y2 : ℝ) (f1 : x1 = 3) (f2 : y1 = -2) (f3 : x2 = 11) (f4 : y2 = 6) :
    (x1 + x2) / 2 = 7 ∧ (y1 + y2) / 2 = 2 :=
by
  sorry

end NUMINAMATH_GPT_hyperbola_center_l1112_111201


namespace NUMINAMATH_GPT_expression_evaluation_l1112_111247

theorem expression_evaluation :
  (8 / 4 - 3^2 + 4 * 5) = 13 :=
by sorry

end NUMINAMATH_GPT_expression_evaluation_l1112_111247


namespace NUMINAMATH_GPT_division_by_reciprocal_l1112_111291

theorem division_by_reciprocal :
  (10 / 3) / (1 / 5) = 50 / 3 := 
sorry

end NUMINAMATH_GPT_division_by_reciprocal_l1112_111291


namespace NUMINAMATH_GPT_estimation_correct_l1112_111235

-- Definitions corresponding to conditions.
def total_population : ℕ := 10000
def surveyed_population : ℕ := 200
def aware_surveyed : ℕ := 125

-- The proportion step: 125/200 = x/10000
def proportion (aware surveyed total_pop : ℕ) : ℕ :=
  (aware * total_pop) / surveyed

-- Using this to define our main proof goal
def estimated_aware := proportion aware_surveyed surveyed_population total_population

-- Final proof statement
theorem estimation_correct :
  estimated_aware = 6250 :=
sorry

end NUMINAMATH_GPT_estimation_correct_l1112_111235


namespace NUMINAMATH_GPT_find_triples_l1112_111246

theorem find_triples (a m n : ℕ) (h1 : a ≥ 2) (h2 : m ≥ 2) :
  a^n + 203 ∣ a^(m * n) + 1 → ∃ (k : ℕ), (k ≥ 1) := 
sorry

end NUMINAMATH_GPT_find_triples_l1112_111246


namespace NUMINAMATH_GPT_distribution_of_balls_l1112_111252

theorem distribution_of_balls :
  ∃ (P : ℕ → ℕ → ℕ), P 6 4 = 9 := 
by
  sorry

end NUMINAMATH_GPT_distribution_of_balls_l1112_111252


namespace NUMINAMATH_GPT_Angela_insect_count_l1112_111265

variables (Angela Jacob Dean : ℕ)
-- Conditions
def condition1 : Prop := Angela = Jacob / 2
def condition2 : Prop := Jacob = 5 * Dean
def condition3 : Prop := Dean = 30

-- Theorem statement proving Angela's insect count
theorem Angela_insect_count (h1 : condition1 Angela Jacob) (h2 : condition2 Jacob Dean) (h3 : condition3 Dean) : Angela = 75 :=
by
  sorry

end NUMINAMATH_GPT_Angela_insect_count_l1112_111265


namespace NUMINAMATH_GPT_median_length_of_right_triangle_l1112_111242

noncomputable def length_of_median (a b c : ℕ) : ℝ := 
  if a * a + b * b = c * c then c / 2 else 0

theorem median_length_of_right_triangle :
  length_of_median 9 12 15 = 7.5 :=
by
  -- Insert the proof here
  sorry

end NUMINAMATH_GPT_median_length_of_right_triangle_l1112_111242


namespace NUMINAMATH_GPT_triangle_inequality_equality_iff_equilateral_l1112_111299

theorem triangle_inequality (a b c : ℝ) (h1 : a + b > c) (h2 : b + c > a) (h3 : c + a > b) : 
  a^2 * b * (a - b) + b^2 * c * (b - c) + c^2 * a * (c - a) ≥ 0 := 
sorry

theorem equality_iff_equilateral (a b c : ℝ) (h1 : a + b > c) (h2 : b + c > a) (h3 : c + a > b) : 
  a^2 * b * (a - b) + b^2 * c * (b - c) + c^2 * a * (c - a) = 0 ↔ a = b ∧ b = c := 
sorry

end NUMINAMATH_GPT_triangle_inequality_equality_iff_equilateral_l1112_111299


namespace NUMINAMATH_GPT_side_of_rhombus_l1112_111284

variable (d : ℝ) (K : ℝ) 

-- Conditions
def shorter_diagonal := d
def longer_diagonal := 3 * d
def area_rhombus := K = (1 / 2) * d * (3 * d)

-- Proof Statement
theorem side_of_rhombus (h1 : K = (3 / 2) * d^2) : (∃ s : ℝ, s = Real.sqrt (5 * K / 3)) := 
  sorry

end NUMINAMATH_GPT_side_of_rhombus_l1112_111284


namespace NUMINAMATH_GPT_largest_square_perimeter_is_28_l1112_111294

-- Definitions and assumptions
def rect_length : ℝ := 10
def rect_width : ℝ := 7

-- Define the largest possible square
def largest_square_side := rect_width

-- Define the perimeter of a square
def perimeter_of_square (side : ℝ) : ℝ := 4 * side

-- Proving statement
theorem largest_square_perimeter_is_28 :
  perimeter_of_square largest_square_side = 28 := 
  by 
    -- sorry is used to skip the proof
    sorry

end NUMINAMATH_GPT_largest_square_perimeter_is_28_l1112_111294


namespace NUMINAMATH_GPT_binary_101_to_decimal_l1112_111208

theorem binary_101_to_decimal : (1 * 2^2 + 0 * 2^1 + 1 * 2^0) = 5 := by
  sorry

end NUMINAMATH_GPT_binary_101_to_decimal_l1112_111208


namespace NUMINAMATH_GPT_total_yellow_leaves_l1112_111262

noncomputable def calculate_yellow_leaves (total : ℕ) (percent_brown : ℕ) (percent_green : ℕ) : ℕ :=
  let brown_leaves := (total * percent_brown + 50) / 100
  let green_leaves := (total * percent_green + 50) / 100
  total - (brown_leaves + green_leaves)

theorem total_yellow_leaves :
  let t_yellow := calculate_yellow_leaves 15 25 40
  let f_yellow := calculate_yellow_leaves 22 30 20
  let s_yellow := calculate_yellow_leaves 30 15 50
  t_yellow + f_yellow + s_yellow = 26 :=
by
  sorry

end NUMINAMATH_GPT_total_yellow_leaves_l1112_111262


namespace NUMINAMATH_GPT_owls_joined_l1112_111249

theorem owls_joined (initial_owls : ℕ) (total_owls : ℕ) (join_owls : ℕ) 
  (h_initial : initial_owls = 3) (h_total : total_owls = 5) : join_owls = 2 :=
by {
  -- Sorry is used to skip the proof
  sorry
}

end NUMINAMATH_GPT_owls_joined_l1112_111249


namespace NUMINAMATH_GPT_least_value_of_fourth_integer_l1112_111261

theorem least_value_of_fourth_integer :
  ∃ (A B C D : ℕ), 
    A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D ∧
    A + B + C + D = 64 ∧ 
    A = 3 * B ∧ B = C - 2 ∧ 
    D = 52 := sorry

end NUMINAMATH_GPT_least_value_of_fourth_integer_l1112_111261


namespace NUMINAMATH_GPT_order_xyz_l1112_111226

theorem order_xyz (x : ℝ) (h1 : 0.8 < x) (h2 : x < 0.9) :
  let y := x^x
  let z := x^(x^x)
  x < z ∧ z < y :=
by
  sorry

end NUMINAMATH_GPT_order_xyz_l1112_111226


namespace NUMINAMATH_GPT_correct_option_is_B_l1112_111290

-- Definitions and conditions based on the problem
def is_monomial (t : String) : Prop :=
  t = "1"

def coefficient (expr : String) : Int :=
  if expr = "x" then 1
  else if expr = "-3x" then -3
  else 0

def degree (term : String) : Int :=
  if term = "5x^2y" then 3
  else 0

-- Proof statement
theorem correct_option_is_B : 
  is_monomial "1" ∧ ¬ (coefficient "x" = 0) ∧ ¬ (coefficient "-3x" = 3) ∧ ¬ (degree "5x^2y" = 2) := 
by
  -- Proof steps will go here
  sorry

end NUMINAMATH_GPT_correct_option_is_B_l1112_111290


namespace NUMINAMATH_GPT_bulls_on_farm_l1112_111296

theorem bulls_on_farm (C B : ℕ) (h1 : C / B = 10 / 27) (h2 : C + B = 555) : B = 405 :=
sorry

end NUMINAMATH_GPT_bulls_on_farm_l1112_111296


namespace NUMINAMATH_GPT_geometric_sequence_value_l1112_111238

variable {α : Type*} [LinearOrderedField α] (a : ℕ → α)
variable (r : α)
variable (a_pos : ∀ n, a n > 0)
variable (h1 : a 1 = 2)
variable (h99 : a 99 = 8)
variable (geom_seq : ∀ n, a (n + 1) = r * a n)

theorem geometric_sequence_value :
  a 20 * a 50 * a 80 = 64 := by
  sorry

end NUMINAMATH_GPT_geometric_sequence_value_l1112_111238


namespace NUMINAMATH_GPT_boat_speed_still_water_l1112_111217

variable (V_b V_s t : ℝ)

-- Conditions given in the problem
axiom speedOfStream : V_s = 13
axiom timeRelation : ∀ t, (V_b + V_s) * t = 2 * (V_b - V_s) * t

-- The statement to be proved
theorem boat_speed_still_water : V_b = 39 :=
by
  sorry

end NUMINAMATH_GPT_boat_speed_still_water_l1112_111217


namespace NUMINAMATH_GPT_negation_of_at_most_three_l1112_111239

theorem negation_of_at_most_three (x : ℕ) : ¬ (x ≤ 3) ↔ x > 3 :=
by sorry

end NUMINAMATH_GPT_negation_of_at_most_three_l1112_111239


namespace NUMINAMATH_GPT_min_value_frac_l1112_111222

theorem min_value_frac (x y : ℝ) (h₁ : x + y = 1) (h₂ : x > 0) (h₃ : y > 0) : 
  ∃ c, (∀ (a b : ℝ), (a + b = 1) → (a > 0) → (b > 0) → (1/a + 4/b) ≥ c) ∧ c = 9 :=
by
  sorry

end NUMINAMATH_GPT_min_value_frac_l1112_111222


namespace NUMINAMATH_GPT_find_a_l1112_111260

theorem find_a {a : ℝ} (h : {x : ℝ | (1/2 : ℝ) < x ∧ x < 2} = {x : ℝ | 0 < ax^2 + 5 * x - 2}) : a = -2 :=
sorry

end NUMINAMATH_GPT_find_a_l1112_111260


namespace NUMINAMATH_GPT_can_cabinet_be_moved_out_through_door_l1112_111256

/-
Definitions for the problem:
- Length, width, and height of the room
- Width, height, and depth of the cabinet
- Width and height of the door
-/

structure Dimensions where
  length : ℝ
  width : ℝ
  height : ℝ

def room : Dimensions := { length := 4, width := 2.5, height := 2.3 }
def cabinet : Dimensions := { length := 0.6, width := 1.8, height := 2.1 }
def door : Dimensions := { length := 0.8, height := 1.9, width := 0 }

theorem can_cabinet_be_moved_out_through_door : 
  (cabinet.length ≤ door.length ∧ cabinet.width ≤ door.height) ∨ 
  (cabinet.width ≤ door.length ∧ cabinet.length ≤ door.height) 
∧ 
cabinet.height ≤ room.height ∧ cabinet.width ≤ room.width ∧ 
cabinet.length ≤ room.length → True :=
by
  sorry

end NUMINAMATH_GPT_can_cabinet_be_moved_out_through_door_l1112_111256


namespace NUMINAMATH_GPT_ellipse_non_degenerate_l1112_111230

noncomputable def non_degenerate_ellipse_condition (b : ℝ) : Prop := b > -13

theorem ellipse_non_degenerate (b : ℝ) :
  (∃ x y : ℝ, 4*x^2 + 9*y^2 - 16*x + 18*y + 12 = b) → non_degenerate_ellipse_condition b :=
by
  sorry

end NUMINAMATH_GPT_ellipse_non_degenerate_l1112_111230


namespace NUMINAMATH_GPT_volunteer_selection_probability_l1112_111275

theorem volunteer_selection_probability :
  ∀ (students total_students remaining_students selected_volunteers : ℕ),
    total_students = 2018 →
    remaining_students = total_students - 18 →
    selected_volunteers = 50 →
    (selected_volunteers : ℚ) / total_students = (25 : ℚ) / 1009 :=
by
  intros students total_students remaining_students selected_volunteers
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_volunteer_selection_probability_l1112_111275


namespace NUMINAMATH_GPT_tetrahedron_volume_eq_three_l1112_111297

noncomputable def volume_of_tetrahedron : ℝ :=
  let PQ := 3
  let PR := 4
  let PS := 5
  let QR := 5
  let QS := Real.sqrt 34
  let RS := Real.sqrt 41
  have := (PQ = 3) ∧ (PR = 4) ∧ (PS = 5) ∧ (QR = 5) ∧ (QS = Real.sqrt 34) ∧ (RS = Real.sqrt 41)
  3

theorem tetrahedron_volume_eq_three : volume_of_tetrahedron = 3 := 
by { sorry }

end NUMINAMATH_GPT_tetrahedron_volume_eq_three_l1112_111297


namespace NUMINAMATH_GPT_line_through_point_and_intersects_circle_with_chord_length_8_l1112_111205

theorem line_through_point_and_intersects_circle_with_chord_length_8 :
  ∃ (l : ℝ → ℝ), (∀ (x : ℝ), l x = 0 ↔ x = 5) ∨ 
  (∀ (x y : ℝ), 7 * x + 24 * y = 35) ↔ 
  (∃ (x : ℝ), x = 5) ∨ 
  (∀ (x y : ℝ), 7 * x + 24 * y = 35) := 
by
  sorry

end NUMINAMATH_GPT_line_through_point_and_intersects_circle_with_chord_length_8_l1112_111205


namespace NUMINAMATH_GPT_gcd_of_powers_of_two_l1112_111272

def m : ℕ := 2^2100 - 1
def n : ℕ := 2^2000 - 1

theorem gcd_of_powers_of_two :
  Nat.gcd m n = 2^100 - 1 := sorry

end NUMINAMATH_GPT_gcd_of_powers_of_two_l1112_111272


namespace NUMINAMATH_GPT_burger_meal_cost_l1112_111274

theorem burger_meal_cost 
  (x : ℝ) 
  (h : 5 * (x + 1) = 35) : 
  x = 6 := 
sorry

end NUMINAMATH_GPT_burger_meal_cost_l1112_111274


namespace NUMINAMATH_GPT_number_of_bad_carrots_l1112_111241

-- Definitions for conditions
def olivia_picked : ℕ := 20
def mother_picked : ℕ := 14
def good_carrots : ℕ := 19

-- Sum of total carrots picked
def total_carrots : ℕ := olivia_picked + mother_picked

-- Theorem stating the number of bad carrots
theorem number_of_bad_carrots : total_carrots - good_carrots = 15 :=
by
  sorry

end NUMINAMATH_GPT_number_of_bad_carrots_l1112_111241


namespace NUMINAMATH_GPT_nesbitt_inequality_nesbitt_inequality_eq_l1112_111206

variable {a b c : ℝ}

theorem nesbitt_inequality (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a / (b + c)) + (b / (a + c)) + (c / (a + b)) ≥ (3 / 2) :=
sorry

theorem nesbitt_inequality_eq (ha : a > 0) (hb : b > 0) (hc : c > 0) : 
  ((a / (b + c)) + (b / (a + c)) + (c / (a + b)) = (3 / 2)) ↔ (a = b ∧ b = c) :=
sorry

end NUMINAMATH_GPT_nesbitt_inequality_nesbitt_inequality_eq_l1112_111206


namespace NUMINAMATH_GPT_find_m_l1112_111234

theorem find_m (
  x : ℚ 
) (m : ℚ) 
  (h1 : 4 * x + 2 * m = 3 * x + 1) 
  (h2 : 3 * x + 2 * m = 6 * x + 1) 
: m = 1/2 := 
  sorry

end NUMINAMATH_GPT_find_m_l1112_111234


namespace NUMINAMATH_GPT_smallest_n_for_divisibility_l1112_111266

theorem smallest_n_for_divisibility (a₁ a₂ : ℕ) (n : ℕ) (h₁ : a₁ = 5 / 8) (h₂ : a₂ = 25) :
  (∃ n : ℕ, n ≥ 1 ∧ (a₁ * (40 ^ (n - 1)) % 2000000 = 0)) → (n = 7) :=
by
  sorry

end NUMINAMATH_GPT_smallest_n_for_divisibility_l1112_111266


namespace NUMINAMATH_GPT_tracy_initial_candies_l1112_111231

theorem tracy_initial_candies (x : ℕ) (consumed_candies : ℕ) (remaining_candies_given_rachel : ℕ) (remaining_candies_given_monica : ℕ) (candies_eaten_by_tracy : ℕ) (candies_eaten_by_mom : ℕ) 
  (brother_candies_taken : ℕ) (final_candies : ℕ) (h_consume : consumed_candies = 2 / 5 * x) (h_remaining1 : remaining_candies_given_rachel = 1 / 3 * (3 / 5 * x)) 
  (h_remaining2 : remaining_candies_given_monica = 1 / 6 * (3 / 5 * x)) (h_left_after_friends : 3 / 5 * x - (remaining_candies_given_rachel + remaining_candies_given_monica) = 3 / 10 * x)
  (h_candies_left : 3 / 10 * x - (candies_eaten_by_tracy + candies_eaten_by_mom) = final_candies + brother_candies_taken) (h_eaten_tracy : candies_eaten_by_tracy = 10)
  (h_eaten_mom : candies_eaten_by_mom = 10) (h_final : final_candies = 6) (h_brother_bound : 2 ≤ brother_candies_taken ∧ brother_candies_taken ≤ 6) : x = 100 := 
by 
  sorry

end NUMINAMATH_GPT_tracy_initial_candies_l1112_111231


namespace NUMINAMATH_GPT_max_difference_in_flour_masses_l1112_111282

/--
Given three brands of flour with the following mass ranges:
1. Brand A: (48 ± 0.1) kg
2. Brand B: (48 ± 0.2) kg
3. Brand C: (48 ± 0.3) kg

Prove that the maximum difference in mass between any two bags of these different brands is 0.5 kg.
-/
theorem max_difference_in_flour_masses :
  (∀ (a b : ℝ), ((47.9 ≤ a ∧ a ≤ 48.1) ∧ (47.8 ≤ b ∧ b ≤ 48.2)) →
    |a - b| ≤ 0.5) ∧
  (∀ (a c : ℝ), ((47.9 ≤ a ∧ a ≤ 48.1) ∧ (47.7 ≤ c ∧ c ≤ 48.3)) →
    |a - c| ≤ 0.5) ∧
  (∀ (b c : ℝ), ((47.8 ≤ b ∧ b ≤ 48.2) ∧ (47.7 ≤ c ∧ c ≤ 48.3)) →
    |b - c| ≤ 0.5) := 
sorry

end NUMINAMATH_GPT_max_difference_in_flour_masses_l1112_111282


namespace NUMINAMATH_GPT_cone_base_circumference_l1112_111278

theorem cone_base_circumference (V : ℝ) (h : ℝ) (C : ℝ) (r : ℝ) :
  V = 18 * Real.pi →
  h = 6 →
  (V = (1 / 3) * Real.pi * r^2 * h) →
  C = 2 * Real.pi * r →
  C = 6 * Real.pi :=
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_cone_base_circumference_l1112_111278


namespace NUMINAMATH_GPT_cuboid_volume_l1112_111285

/-- Given a cuboid with edges 6 cm, 5 cm, and 6 cm, the volume of the cuboid
    is 180 cm³. -/
theorem cuboid_volume (a b c : ℕ) (h1 : a = 6) (h2 : b = 5) (h3 : c = 6) :
  a * b * c = 180 := by
  sorry

end NUMINAMATH_GPT_cuboid_volume_l1112_111285


namespace NUMINAMATH_GPT_problem_l1112_111298

variables (A B C D E : ℝ)

-- Conditions
def condition1 := A > C
def condition2 := E > B ∧ B > D
def condition3 := D > A
def condition4 := C > B

-- Proof goal: Dana (D) and Beth (B) have the same amount of money
theorem problem (h1 : condition1 A C) (h2 : condition2 E B D) (h3 : condition3 D A) (h4 : condition4 C B) : D = B :=
sorry

end NUMINAMATH_GPT_problem_l1112_111298


namespace NUMINAMATH_GPT_associates_hired_l1112_111293

variable (partners : ℕ) (associates initial_associates hired_associates : ℕ)
variable (initial_ratio : partners / initial_associates = 2 / 63)
variable (final_ratio : partners / (initial_associates + hired_associates) = 1 / 34)
variable (partners_count : partners = 18)

theorem associates_hired : hired_associates = 45 :=
by
  -- Insert solution steps here...
  sorry

end NUMINAMATH_GPT_associates_hired_l1112_111293


namespace NUMINAMATH_GPT_plane_divides_pyramid_l1112_111209

noncomputable def volume_of_parts (a h KL KK1: ℝ): ℝ × ℝ :=
  -- Define the pyramid and prism structure and the conditions
  let volume_total := (1/3) * (a^2) * h
  let volume_part1 := 512/15
  let volume_part2 := volume_total - volume_part1
  (⟨volume_part1, volume_part2⟩ : ℝ × ℝ)

theorem plane_divides_pyramid (a h KL KK1: ℝ) 
  (h₁ : a = 8 * Real.sqrt 2) 
  (h₂ : h = 4) 
  (h₃ : KL = 2) 
  (h₄ : KK1 = 1):
  volume_of_parts a h KL KK1 = (512/15, 2048/15) := 
by 
  sorry

end NUMINAMATH_GPT_plane_divides_pyramid_l1112_111209


namespace NUMINAMATH_GPT_rational_division_example_l1112_111203

theorem rational_division_example : (3 / 7) / 5 = 3 / 35 := by
  sorry

end NUMINAMATH_GPT_rational_division_example_l1112_111203


namespace NUMINAMATH_GPT_coat_price_reduction_l1112_111253

theorem coat_price_reduction (original_price reduction_amount : ℝ) (h : original_price = 500) (h_red : reduction_amount = 150) :
  ((reduction_amount / original_price) * 100) = 30 :=
by
  rw [h, h_red]
  norm_num

end NUMINAMATH_GPT_coat_price_reduction_l1112_111253


namespace NUMINAMATH_GPT_range_f_when_a_1_range_of_a_values_l1112_111255

noncomputable def f (x a : ℝ) : ℝ := |x - a| + |x + 4|

theorem range_f_when_a_1 : 
  (∀ x : ℝ, f x 1 ≥ 5) :=
sorry

theorem range_of_a_values :
  (∀ x, f x a ≥ 1) → (a ∈ Set.union (Set.Iic (-5)) (Set.Ici (-3))) :=
sorry

end NUMINAMATH_GPT_range_f_when_a_1_range_of_a_values_l1112_111255


namespace NUMINAMATH_GPT_original_total_cost_l1112_111214

-- Definitions based on the conditions
def price_jeans : ℝ := 14.50
def price_shirt : ℝ := 9.50
def price_jacket : ℝ := 21.00

def jeans_count : ℕ := 2
def shirts_count : ℕ := 4
def jackets_count : ℕ := 1

-- The proof statement
theorem original_total_cost :
  (jeans_count * price_jeans) + (shirts_count * price_shirt) + (jackets_count * price_jacket) = 88 := 
by
  sorry

end NUMINAMATH_GPT_original_total_cost_l1112_111214


namespace NUMINAMATH_GPT_remainder_correct_l1112_111219

def dividend : ℕ := 165
def divisor : ℕ := 18
def quotient : ℕ := 9
def remainder : ℕ := 3

theorem remainder_correct {d q r : ℕ} (h1 : d = dividend) (h2 : q = quotient) (h3 : r = divisor * q) : d = 165 → q = 9 → 165 = 162 + remainder :=
by { sorry }

end NUMINAMATH_GPT_remainder_correct_l1112_111219


namespace NUMINAMATH_GPT_problem_statement_l1112_111277

noncomputable def angle_between_vectors (a b : EuclideanSpace ℝ (Fin 3)) : ℝ :=
Real.arccos (inner a b / (‖a‖ * ‖b‖))

theorem problem_statement
  (a b : EuclideanSpace ℝ (Fin 3))
  (h_angle_ab : angle_between_vectors a b = Real.pi / 3)
  (h_norm_a : ‖a‖ = 2)
  (h_norm_b : ‖b‖ = 1) :
  angle_between_vectors a (a + 2 • b) = Real.pi / 6 :=
sorry

end NUMINAMATH_GPT_problem_statement_l1112_111277


namespace NUMINAMATH_GPT_correct_polynomial_l1112_111207

noncomputable def p : Polynomial ℤ :=
  Polynomial.C 1 * Polynomial.X^6 - Polynomial.C 8 * Polynomial.X^4 - Polynomial.C 2 * Polynomial.X^3 + Polynomial.C 13 * Polynomial.X^2 - Polynomial.C 10 * Polynomial.X - Polynomial.C 1

theorem correct_polynomial (r t : ℝ) :
  (r^3 - r - 1 = 0) → (t = r + Real.sqrt 2) → Polynomial.aeval t p = 0 :=
by
  sorry

end NUMINAMATH_GPT_correct_polynomial_l1112_111207


namespace NUMINAMATH_GPT_robot_swap_eventually_non_swappable_l1112_111220

theorem robot_swap_eventually_non_swappable (n : ℕ) (a : Fin n → ℕ) :
  ∃ t : ℕ, ∀ i : Fin (n - 1), ¬ (a (⟨i, sorry⟩ : Fin n) > a (⟨i + 1, sorry⟩ : Fin n)) ↔ n > 1 :=
sorry

end NUMINAMATH_GPT_robot_swap_eventually_non_swappable_l1112_111220


namespace NUMINAMATH_GPT_find_J_l1112_111280

variables (J S B : ℕ)

-- Conditions
def condition1 : Prop := J - 20 = 2 * S
def condition2 : Prop := B = J / 2
def condition3 : Prop := J + S + B = 330
def condition4 : Prop := (J - 20) + S + B = 318

-- Theorem to prove
theorem find_J (h1 : condition1 J S) (h2 : condition2 J B) (h3 : condition3 J S B) (h4 : condition4 J S B) :
  J = 170 :=
sorry

end NUMINAMATH_GPT_find_J_l1112_111280


namespace NUMINAMATH_GPT_lines_from_equation_l1112_111204

-- Definitions for the conditions
def satisfies_equation (x y : ℝ) : Prop :=
  2 * x^2 + y^2 + 3 * x * y + 3 * x + y = 2

-- Equivalent Lean statement to the proof problem
theorem lines_from_equation :
  (∀ x y : ℝ, satisfies_equation x y → (y = -x - 2) ∨ (y = -2 * x + 1)) :=
by
  intros x y h
  sorry

end NUMINAMATH_GPT_lines_from_equation_l1112_111204


namespace NUMINAMATH_GPT_find_principal_l1112_111279

variable (SI : ℝ) (R : ℝ) (T : ℝ) (P : ℝ)

theorem find_principal (h1 : SI = 4020.75) (h2 : R = 0.0875) (h3 : T = 5.5) (h4 : SI = P * R * T) : 
  P = 8355.00 :=
sorry

end NUMINAMATH_GPT_find_principal_l1112_111279


namespace NUMINAMATH_GPT_sequence_general_term_l1112_111286

theorem sequence_general_term (a : ℕ → ℕ) (h1 : a 1 = 1)
  (h2 : ∀ n : ℕ, 3 * (Finset.range (n + 1)).sum a = (n + 2) * a n) :
  ∀ n : ℕ, a n = n :=
by
  sorry

end NUMINAMATH_GPT_sequence_general_term_l1112_111286


namespace NUMINAMATH_GPT_solve_floor_equation_l1112_111245

theorem solve_floor_equation (x : ℝ) (h : ⌊x * ⌊x⌋⌋ = 20) : 5 ≤ x ∧ x < 5.25 := by
  sorry

end NUMINAMATH_GPT_solve_floor_equation_l1112_111245


namespace NUMINAMATH_GPT_remainder_a25_div_26_l1112_111221

def concatenate_numbers (n : ℕ) : ℕ :=
  -- Placeholder function for concatenating numbers from 1 to n
  sorry

theorem remainder_a25_div_26 :
  let a_25 := concatenate_numbers 25
  a_25 % 26 = 13 :=
by sorry

end NUMINAMATH_GPT_remainder_a25_div_26_l1112_111221


namespace NUMINAMATH_GPT_find_value_of_n_l1112_111248

def is_prime (p : ℕ) : Prop := Nat.Prime p

theorem find_value_of_n
  (a b c n : ℕ)
  (ha : is_prime a)
  (hb : is_prime b)
  (hc : is_prime c)
  (h1 : 2 * a + 3 * b = c)
  (h2 : 4 * a + c + 1 = 4 * b)
  (h3 : n = a * b * c)
  (h4 : n < 10000) :
  n = 1118 :=
by
  sorry

end NUMINAMATH_GPT_find_value_of_n_l1112_111248


namespace NUMINAMATH_GPT_john_gives_to_stud_owner_l1112_111287

variable (initial_puppies : ℕ) (puppies_given_away : ℕ) (puppies_kept : ℕ) (price_per_puppy : ℕ) (profit : ℕ)

theorem john_gives_to_stud_owner
  (h1 : initial_puppies = 8)
  (h2 : puppies_given_away = initial_puppies / 2)
  (h3 : puppies_kept = 1)
  (h4 : price_per_puppy = 600)
  (h5 : profit = 1500) :
  let puppies_left_to_sell := initial_puppies - puppies_given_away - puppies_kept
  let total_sales := puppies_left_to_sell * price_per_puppy
  total_sales - profit = 300 :=
by
  intro puppies_left_to_sell
  intro total_sales
  sorry

end NUMINAMATH_GPT_john_gives_to_stud_owner_l1112_111287


namespace NUMINAMATH_GPT_find_number_divided_by_6_l1112_111213

theorem find_number_divided_by_6 (x : ℤ) (h : (x + 17) / 5 = 25) : x / 6 = 18 :=
by
  sorry

end NUMINAMATH_GPT_find_number_divided_by_6_l1112_111213


namespace NUMINAMATH_GPT_dora_knows_coin_position_l1112_111218

-- Definitions
def R_is_dime_or_nickel (R : ℕ) (L : ℕ) : Prop := 
  (R = 10 ∧ L = 5) ∨ (R = 5 ∧ L = 10)

-- Theorem statement
theorem dora_knows_coin_position (R : ℕ) (L : ℕ) 
  (h : R_is_dime_or_nickel R L) :
  (3 * R + 2 * L) % 2 = 0 ↔ (R = 10 ∧ L = 5) :=
by
  sorry

end NUMINAMATH_GPT_dora_knows_coin_position_l1112_111218
