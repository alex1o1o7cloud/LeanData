import Mathlib

namespace NUMINAMATH_GPT_intersection_point_of_lines_PQ_RS_l882_88226

def point := ℝ × ℝ × ℝ

def P : point := (4, -3, 6)
def Q : point := (1, 10, 11)
def R : point := (3, -4, 2)
def S : point := (-1, 5, 16)

theorem intersection_point_of_lines_PQ_RS :
  let line_PQ (u : ℝ) := (4 - 3 * u, -3 + 13 * u, 6 + 5 * u)
  let line_RS (v : ℝ) := (3 - 4 * v, -4 + 9 * v, 2 + 14 * v)
  ∃ u v : ℝ,
    line_PQ u = line_RS v →
    line_PQ u = (19 / 5, 44 / 3, 23 / 3) :=
by
  sorry

end NUMINAMATH_GPT_intersection_point_of_lines_PQ_RS_l882_88226


namespace NUMINAMATH_GPT_identity_completion_factorize_polynomial_equilateral_triangle_l882_88201

-- Statement 1: Prove that a^3 - b^3 + a^2 b - ab^2 = (a - b)(a + b)^2 
theorem identity_completion (a b : ℝ) : a^3 - b^3 + a^2 * b - a * b^2 = (a - b) * (a + b)^2 :=
sorry

-- Statement 2: Prove that 4x^2 - 2x - y^2 - y = (2x + y)(2x - y - 1)
theorem factorize_polynomial (x y : ℝ) : 4 * x^2 - 2 * x - y^2 - y = (2 * x + y) * (2 * x - y - 1) :=
sorry

-- Statement 3: Given a^2 + b^2 + 2c^2 - 2ac - 2bc = 0, Prove that triangle ABC is equilateral
theorem equilateral_triangle (a b c : ℝ) (h : a^2 + b^2 + 2 * c^2 - 2 * a * c - 2 * b * c = 0) : a = b ∧ b = c :=
sorry

end NUMINAMATH_GPT_identity_completion_factorize_polynomial_equilateral_triangle_l882_88201


namespace NUMINAMATH_GPT_find_x_l882_88206

theorem find_x (x : ℝ) (h : (20 + 30 + 40 + x) / 4 = 35) : x = 50 := by
  sorry

end NUMINAMATH_GPT_find_x_l882_88206


namespace NUMINAMATH_GPT_tan_sum_l882_88267

-- Define the conditions as local variables
variables {α β : ℝ} (h₁ : Real.tan α = -2) (h₂ : Real.tan β = 5)

-- The statement to prove
theorem tan_sum : Real.tan (α + β) = 3 / 11 :=
by 
  -- Proof goes here, using 'sorry' as placeholder
  sorry

end NUMINAMATH_GPT_tan_sum_l882_88267


namespace NUMINAMATH_GPT_max_correct_answers_l882_88255

theorem max_correct_answers (c w b : ℕ) (h1 : c + w + b = 30) (h2 : 4 * c - w = 85) : c ≤ 23 :=
  sorry

end NUMINAMATH_GPT_max_correct_answers_l882_88255


namespace NUMINAMATH_GPT_bicycle_helmet_savings_l882_88295

theorem bicycle_helmet_savings :
  let bicycle_regular_price := 320
  let bicycle_discount := 0.2
  let helmet_regular_price := 80
  let helmet_discount := 0.1
  let bicycle_savings := bicycle_regular_price * bicycle_discount
  let helmet_savings := helmet_regular_price * helmet_discount
  let total_savings := bicycle_savings + helmet_savings
  let total_regular_price := bicycle_regular_price + helmet_regular_price
  let percentage_savings := (total_savings / total_regular_price) * 100
  percentage_savings = 18 := 
by sorry

end NUMINAMATH_GPT_bicycle_helmet_savings_l882_88295


namespace NUMINAMATH_GPT_tan_alpha_value_complex_expression_value_l882_88297

theorem tan_alpha_value (α : ℝ) (h : Real.tan (π / 4 + α) = 1 / 2) : Real.tan α = -1 / 3 :=
sorry

theorem complex_expression_value 
(α : ℝ) 
(h1 : Real.tan (π / 4 + α) = 1 / 2) 
(h2 : Real.tan α = -1 / 3) : 
Real.sin (2 * α + 2 * π) - (Real.sin (π / 2 - α))^2 / 
(1 - Real.cos (π - 2 * α) + (Real.sin α)^2) = -15 / 19 :=
sorry

end NUMINAMATH_GPT_tan_alpha_value_complex_expression_value_l882_88297


namespace NUMINAMATH_GPT_vector_problem_l882_88240

noncomputable def t_value : ℝ :=
  (-5 - Real.sqrt 13) / 2

theorem vector_problem 
  (t : ℝ)
  (a : ℝ × ℝ := (1, 1))
  (b : ℝ × ℝ := (2, t))
  (h : Real.sqrt ((1 - 2)^2 + (1 - t)^2) = (1 * 2 + 1 * t)) :
  t = t_value := 
sorry

end NUMINAMATH_GPT_vector_problem_l882_88240


namespace NUMINAMATH_GPT_scalene_triangle_cannot_be_divided_into_two_congruent_triangles_l882_88280

-- Definitions and Conditions
structure Triangle :=
(a b c : ℝ) (h1 : a ≠ b) (h2 : b ≠ c) (h3 : a ≠ c)

-- Statement of the problem
theorem scalene_triangle_cannot_be_divided_into_two_congruent_triangles (T : Triangle) :
  ¬(∃ (D : ℝ) (ABD ACD : Triangle), ABD.a = ACD.a ∧ ABD.b = ACD.b ∧ ABD.c = ACD.c) :=
sorry

end NUMINAMATH_GPT_scalene_triangle_cannot_be_divided_into_two_congruent_triangles_l882_88280


namespace NUMINAMATH_GPT_probability_two_boys_l882_88204

def number_of_students : ℕ := 5
def number_of_boys : ℕ := 2
def number_of_girls : ℕ := 3
def total_pairs : ℕ := Nat.choose number_of_students 2
def boys_pairs : ℕ := Nat.choose number_of_boys 2

theorem probability_two_boys :
  number_of_students = 5 →
  number_of_boys = 2 →
  number_of_girls = 3 →
  (boys_pairs : ℝ) / (total_pairs : ℝ) = 1 / 10 :=
by
  sorry

end NUMINAMATH_GPT_probability_two_boys_l882_88204


namespace NUMINAMATH_GPT_find_xy_solution_l882_88281

theorem find_xy_solution (x y : ℕ) (hx : x > 0) (hy : y > 0) 
    (h : 3^x + x^4 = y.factorial + 2019) : 
    (x = 6 ∧ y = 3) :=
by {
  sorry
}

end NUMINAMATH_GPT_find_xy_solution_l882_88281


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_l882_88296

/-
The sum of the first 20 terms of the arithmetic sequence 8, 5, 2, ... is -410.
-/

theorem arithmetic_sequence_sum :
  let a : ℤ := 8
  let d : ℤ := -3
  let n : ℤ := 20
  let S_n : ℤ := n * a + (d * n * (n - 1)) / 2
  S_n = -410 := by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_l882_88296


namespace NUMINAMATH_GPT_volume_of_mixture_removed_replaced_l882_88291

noncomputable def volume_removed (initial_mixture: ℝ) (initial_milk: ℝ) (final_concentration: ℝ): ℝ :=
  (1 - final_concentration / initial_milk) * initial_mixture

theorem volume_of_mixture_removed_replaced (initial_mixture: ℝ) (initial_milk: ℝ) (final_concentration: ℝ) (V: ℝ):
  initial_mixture = 100 →
  initial_milk = 36 →
  final_concentration = 9 →
  V = 50 →
  volume_removed initial_mixture initial_milk final_concentration = V :=
by
  intros h1 h2 h3 h4
  have h5 : initial_mixture = 100 := h1
  have h6 : initial_milk = 36 := h2
  have h7 : final_concentration = 9 := h3
  rw [h5, h6, h7]
  sorry

end NUMINAMATH_GPT_volume_of_mixture_removed_replaced_l882_88291


namespace NUMINAMATH_GPT_min_k_intersects_circle_l882_88292

def circle_eq (x y : ℝ) := (x + 2)^2 + y^2 = 4
def line_eq (x y k : ℝ) := k * x - y - 2 * k = 0

theorem min_k_intersects_circle :
  (∀ k : ℝ, (∃ x y : ℝ, circle_eq x y ∧ line_eq x y k) → k ≥ - (Real.sqrt 3) / 3) :=
sorry

end NUMINAMATH_GPT_min_k_intersects_circle_l882_88292


namespace NUMINAMATH_GPT_jessies_weight_loss_l882_88252

-- Definitions based on the given conditions
def initial_weight : ℝ := 74
def weight_loss_rate_even_days : ℝ := 0.2 + 0.15
def weight_loss_rate_odd_days : ℝ := 0.3
def total_exercise_days : ℕ := 25
def even_days : ℕ := (total_exercise_days - 1) / 2
def odd_days : ℕ := even_days + 1

-- The goal is to prove the total weight loss is 8.1 kg
theorem jessies_weight_loss : 
  (even_days * weight_loss_rate_even_days + odd_days * weight_loss_rate_odd_days) = 8.1 := 
by
  sorry

end NUMINAMATH_GPT_jessies_weight_loss_l882_88252


namespace NUMINAMATH_GPT_friends_belong_special_team_l882_88293

-- Define a type for students
universe u
variable {Student : Type u}

-- Assume a friendship relation among students
variable (friend : Student → Student → Prop)

-- Assume the conditions as given in the problem
variable (S : Student → Set (Set Student))
variable (students : Set Student)
variable (S_non_empty : ∀ v : Student, S v ≠ ∅)
variable (friendship_condition : 
  ∀ u v : Student, friend u v → 
    (∃ w : Student, S u ∩ S v ⊇ S w))
variable (special_team : ∀ (T : Set Student),
  (∃ v ∈ T, ∀ w : Student, w ∈ T → friend v w) ↔
  (∃ v ∈ T, ∀ w : Student, friend v w → w ∈ T))

-- Prove that any two friends belong to some special team
theorem friends_belong_special_team :
  ∀ u v : Student, friend u v → 
    (∃ T : Set Student, T ∈ S u ∩ S v ∧ 
      (∃ w ∈ T, ∀ x : Student, friend w x → x ∈ T)) :=
by
  sorry  -- Proof omitted


end NUMINAMATH_GPT_friends_belong_special_team_l882_88293


namespace NUMINAMATH_GPT_taylor_pets_count_l882_88256

noncomputable def totalPetsTaylorFriends (T : ℕ) (x1 : ℕ) (x2 : ℕ) : ℕ :=
  T + 3 * x1 + 2 * x2

theorem taylor_pets_count (T : ℕ) (x1 x2 : ℕ) (h1 : x1 = 2 * T) (h2 : x2 = 2) (h3 : totalPetsTaylorFriends T x1 x2 = 32) :
  T = 4 :=
by
  sorry

end NUMINAMATH_GPT_taylor_pets_count_l882_88256


namespace NUMINAMATH_GPT_johnson_potatoes_l882_88207

/-- Given that Johnson has a sack of 300 potatoes, 
    gives some to Gina, twice that amount to Tom, and 
    one-third of the amount given to Tom to Anne,
    and has 47 potatoes left, we prove that 
    Johnson gave Gina 69 potatoes. -/
theorem johnson_potatoes : 
  ∃ G : ℕ, 
  ∀ (Gina Tom Anne total : ℕ), 
    total = 300 ∧ 
    total - (Gina + Tom + Anne) = 47 ∧ 
    Tom = 2 * Gina ∧ 
    Anne = (1 / 3 : ℚ) * Tom ∧ 
    (Gina + Tom + (Anne : ℕ)) = (11 / 3 : ℚ) * Gina ∧ 
    (Gina + Tom + Anne) = 253 
    ∧ total = Gina + Tom + Anne + 47 
    → Gina = 69 := sorry


end NUMINAMATH_GPT_johnson_potatoes_l882_88207


namespace NUMINAMATH_GPT_non_fiction_vs_fiction_diff_l882_88243

def total_books : Nat := 35
def fiction_books : Nat := 5
def picture_books : Nat := 11
def autobiography_books : Nat := 2 * fiction_books

def accounted_books : Nat := fiction_books + autobiography_books + picture_books
def non_fiction_books : Nat := total_books - accounted_books

theorem non_fiction_vs_fiction_diff :
  non_fiction_books - fiction_books = 4 := by 
  sorry

end NUMINAMATH_GPT_non_fiction_vs_fiction_diff_l882_88243


namespace NUMINAMATH_GPT_tangent_line_circle_l882_88269

theorem tangent_line_circle (r : ℝ) (h : r > 0) :
  (∀ x y : ℝ, (x - 1)^2 + (y - 1)^2 = r^2 → x + y = 2 * r) ↔ r = 2 + Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_tangent_line_circle_l882_88269


namespace NUMINAMATH_GPT_regular_polygon_sides_l882_88216

theorem regular_polygon_sides (h_regular: ∀(n : ℕ), (360 / n) = 18) : ∃ n : ℕ, n = 20 :=
by
  -- So that the statement can be built successfully, we assume the result here
  sorry

end NUMINAMATH_GPT_regular_polygon_sides_l882_88216


namespace NUMINAMATH_GPT_functional_eq_solutions_l882_88299

-- Define the conditions for the problem
def func_equation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x * f y - y * f x) = f (x * y) - x * y

-- Define the two solutions to be proven correct
def f1 (x : ℝ) : ℝ := x
def f2 (x : ℝ) : ℝ := |x|

-- State the main theorem to be proven
theorem functional_eq_solutions (f : ℝ → ℝ) (h : func_equation f) : f = f1 ∨ f = f2 :=
sorry

end NUMINAMATH_GPT_functional_eq_solutions_l882_88299


namespace NUMINAMATH_GPT_min_value_b_minus_a_l882_88228

noncomputable def f (x : ℝ) : ℝ := Real.exp x
noncomputable def g (x : ℝ) : ℝ := Real.log (x / 2) + 1 / 2

theorem min_value_b_minus_a :
  ∀ (a : ℝ), ∃ (b : ℝ), b > 0 ∧ f a = g b ∧ ∀ (y : ℝ), b - a = 2 * Real.exp (y - 1 / 2) - Real.log y → y = 1 / 2 → b - a = 2 + Real.log 2 := by
  sorry

end NUMINAMATH_GPT_min_value_b_minus_a_l882_88228


namespace NUMINAMATH_GPT_students_both_courses_l882_88284

-- Definitions from conditions
def total_students : ℕ := 87
def students_french : ℕ := 41
def students_german : ℕ := 22
def students_neither : ℕ := 33

-- The statement we need to prove
theorem students_both_courses : (students_french + students_german - 9 + students_neither = total_students) → (9 = 96 - total_students) :=
by
  -- The proof would go here, but we leave it as sorry for now
  sorry

end NUMINAMATH_GPT_students_both_courses_l882_88284


namespace NUMINAMATH_GPT_center_square_is_15_l882_88262

noncomputable def center_square_value : ℤ :=
  let d1 := (15 - 3) / 2
  let d3 := (33 - 9) / 2
  let middle_first_row := 3 + d1
  let middle_last_row := 9 + d3
  let d2 := (middle_last_row - middle_first_row) / 2
  middle_first_row + d2

theorem center_square_is_15 : center_square_value = 15 := by
  sorry

end NUMINAMATH_GPT_center_square_is_15_l882_88262


namespace NUMINAMATH_GPT_O_is_incenter_l882_88272

variable {n : ℕ}
variable (A : Fin n → ℝ × ℝ)
variable (O : ℝ × ℝ)

-- Conditions
def inside_convex_ngon (O : ℝ × ℝ) (A : Fin n → ℝ × ℝ) : Prop := sorry
def angles_acute (O : ℝ × ℝ) (A : Fin n → ℝ × ℝ) : Prop := sorry
def angles_inequality (O : ℝ × ℝ) (A : Fin n → ℝ × ℝ) : Prop := sorry

-- This is the statement that we need to prove.
theorem O_is_incenter 
  (h1 : inside_convex_ngon O A)
  (h2 : angles_acute O A) 
  (h3 : angles_inequality O A) 
: sorry := sorry

end NUMINAMATH_GPT_O_is_incenter_l882_88272


namespace NUMINAMATH_GPT_expected_adjacent_black_pairs_60_cards_l882_88236

noncomputable def expected_adjacent_black_pairs 
(deck_size : ℕ) (black_cards : ℕ) (red_cards : ℕ) : ℚ :=
  if h : deck_size = black_cards + red_cards 
  then (black_cards:ℚ) * (black_cards - 1) / (deck_size - 1) 
  else 0

theorem expected_adjacent_black_pairs_60_cards :
  expected_adjacent_black_pairs 60 36 24 = 1260 / 59 := by
  sorry

end NUMINAMATH_GPT_expected_adjacent_black_pairs_60_cards_l882_88236


namespace NUMINAMATH_GPT_partial_fraction_product_l882_88263

theorem partial_fraction_product : 
  (∃ A B C : ℚ, 
    (∀ x : ℚ, x ≠ 3 ∧ x ≠ -3 ∧ x ≠ 5 → 
      (x^2 - 21) / ((x - 3) * (x + 3) * (x - 5)) = A / (x - 3) + B / (x + 3) + C / (x - 5))
      ∧ (A * B * C = -1/16)) := 
    sorry

end NUMINAMATH_GPT_partial_fraction_product_l882_88263


namespace NUMINAMATH_GPT_expression_value_l882_88298

theorem expression_value (x y : ℝ) (h : x - y = 1) :
  x^4 - x * y^3 - x^3 * y - 3 * x^2 * y + 3 * x * y^2 + y^4 = 1 :=
by
  sorry

end NUMINAMATH_GPT_expression_value_l882_88298


namespace NUMINAMATH_GPT_min_distance_line_curve_l882_88259

/-- 
  Given line l with parametric equations:
    x = 1 + t * cos α,
    y = t * sin α,
  and curve C with the polar equation:
    ρ * sin^2 θ = 4 * cos θ,
  prove:
    1. The Cartesian coordinate equation of C is y^2 = 4x.
    2. The minimum value of the distance |AB|, where line l intersects curve C, is 4.
-/
theorem min_distance_line_curve {t α θ ρ x y : ℝ} 
  (h_line_x: x = 1 + t * Real.cos α)
  (h_line_y: y = t * Real.sin α)
  (h_curve_polar: ρ * (Real.sin θ)^2 = 4 * Real.cos θ)
  (h_alpha_range: 0 < α ∧ α < Real.pi) : 
  (∀ {x y}, y^2 = 4 * x) ∧ (min_value_of_AB = 4) :=
sorry

end NUMINAMATH_GPT_min_distance_line_curve_l882_88259


namespace NUMINAMATH_GPT_Andy_collects_16_balls_l882_88224

-- Define the number of balls collected by Andy, Roger, and Maria.
variables (x : ℝ) (r : ℝ) (m : ℝ)

-- Define the conditions
def Andy_twice_as_many_as_Roger : Prop := r = x / 2
def Andy_five_more_than_Maria : Prop := m = x - 5
def Total_balls : Prop := x + r + m = 35

-- Define the main theorem to prove Andy's number of balls
theorem Andy_collects_16_balls (h1 : Andy_twice_as_many_as_Roger x r) 
                               (h2 : Andy_five_more_than_Maria x m) 
                               (h3 : Total_balls x r m) : 
                               x = 16 := 
by 
  sorry

end NUMINAMATH_GPT_Andy_collects_16_balls_l882_88224


namespace NUMINAMATH_GPT_calc1_calc2_calc3_l882_88231

theorem calc1 : 1 - 2 + 3 - 4 + 5 = 3 := by sorry
theorem calc2 : - (4 / 7) / (8 / 49) = - (7 / 2) := by sorry
theorem calc3 : ((1 / 2) - (3 / 5) + (2 / 3)) * (-15) = - (17 / 2) := by sorry

end NUMINAMATH_GPT_calc1_calc2_calc3_l882_88231


namespace NUMINAMATH_GPT_marco_paint_fraction_l882_88289

theorem marco_paint_fraction (W : ℝ) (M : ℝ) (minutes_paint : ℝ) (fraction_paint : ℝ) :
  M = 60 ∧ W = 1 ∧ minutes_paint = 12 ∧ fraction_paint = 1/5 → 
  (minutes_paint / M) * W = fraction_paint := 
by
  sorry

end NUMINAMATH_GPT_marco_paint_fraction_l882_88289


namespace NUMINAMATH_GPT_greatest_possible_y_l882_88276

theorem greatest_possible_y
  (x y : ℤ)
  (h : x * y + 7 * x + 6 * y = -14) : y ≤ 21 :=
sorry

end NUMINAMATH_GPT_greatest_possible_y_l882_88276


namespace NUMINAMATH_GPT_elise_initial_money_l882_88222

theorem elise_initial_money :
  ∃ (X : ℤ), X + 13 - 2 - 18 = 1 ∧ X = 8 :=
by
  sorry

end NUMINAMATH_GPT_elise_initial_money_l882_88222


namespace NUMINAMATH_GPT_hyperbola_parabola_intersection_l882_88249

open Real

theorem hyperbola_parabola_intersection :
  let A := (4, 4)
  let B := (4, -4)
  |dist A B| = 8 :=
by
  let hyperbola_asymptote (x y: ℝ) := x^2 - y^2 = 1
  let parabola_equation (x y : ℝ) := y^2 = 4 * x
  sorry

end NUMINAMATH_GPT_hyperbola_parabola_intersection_l882_88249


namespace NUMINAMATH_GPT_sequence_elements_are_prime_l882_88205

variable {a : ℕ → ℕ} {p : ℕ → ℕ}

def increasing_seq (f : ℕ → ℕ) : Prop :=
  ∀ i j, i < j → f i < f j

def divisible_by_prime (a p : ℕ → ℕ) : Prop :=
  ∀ n, Prime (p n) ∧ p n ∣ a n

def satisfies_condition (a p : ℕ → ℕ) : Prop :=
  ∀ n k, a n - a k = p n - p k

theorem sequence_elements_are_prime (h1 : increasing_seq a) 
    (h2 : divisible_by_prime a p) 
    (h3 : satisfies_condition a p) :
    ∀ n, Prime (a n) :=
by 
  sorry

end NUMINAMATH_GPT_sequence_elements_are_prime_l882_88205


namespace NUMINAMATH_GPT_add_base3_numbers_l882_88230

-- Definitions to represent the numbers in base 3
def base3_num1 := (2 : ℕ) -- 2_3
def base3_num2 := (2 * 3 + 2 : ℕ) -- 22_3
def base3_num3 := (2 * 3^2 + 0 * 3 + 2 : ℕ) -- 202_3
def base3_num4 := (2 * 3^3 + 0 * 3^2 + 2 * 3 + 2 : ℕ) -- 2022_3

-- Summing the numbers in base 10 first
def sum_base10 := base3_num1 + base3_num2 + base3_num3 + base3_num4

-- Expected result in base 10 for 21010_3
def result_base10 := 2 * 3^4 + 1 * 3^3 + 0 * 3^2 + 1 * 3 + 0

-- Proof statement
theorem add_base3_numbers : sum_base10 = result_base10 :=
by {
  -- Proof not required, so we skip it using sorry
  sorry
}

end NUMINAMATH_GPT_add_base3_numbers_l882_88230


namespace NUMINAMATH_GPT_CarrieSpent_l882_88251

variable (CostPerShirt NumberOfShirts : ℝ)

def TotalCost (CostPerShirt NumberOfShirts : ℝ) : ℝ :=
  CostPerShirt * NumberOfShirts

theorem CarrieSpent {CostPerShirt NumberOfShirts : ℝ} 
  (h1 : CostPerShirt = 9.95) 
  (h2 : NumberOfShirts = 20) : 
  TotalCost CostPerShirt NumberOfShirts = 199.00 :=
by
  sorry

end NUMINAMATH_GPT_CarrieSpent_l882_88251


namespace NUMINAMATH_GPT_aquarium_length_l882_88278

theorem aquarium_length {L : ℝ} (W H : ℝ) (final_volume : ℝ)
  (hW : W = 6) (hH : H = 3) (h_final_volume : final_volume = 54)
  (h_volume_relation : final_volume = 3 * (1/4 * L * W * H)) :
  L = 4 := by
  -- Mathematically translate the problem given conditions and resulting in L = 4.
  sorry

end NUMINAMATH_GPT_aquarium_length_l882_88278


namespace NUMINAMATH_GPT_circle_properties_l882_88211

theorem circle_properties :
  ∃ (c d s : ℝ), (∀ x y : ℝ, x^2 - 4 * y - 25 = -y^2 + 10 * x + 49 → (x - 5)^2 + (y - 2)^2 = s^2) ∧
  c = 5 ∧ d = 2 ∧ s = Real.sqrt 103 ∧ c + d + s = 7 + Real.sqrt 103 :=
by
  sorry

end NUMINAMATH_GPT_circle_properties_l882_88211


namespace NUMINAMATH_GPT_sum_and_count_l882_88247

def sum_of_integers (a b : ℕ) : ℕ :=
  (b - a + 1) * (a + b) / 2

def count_even_integers (a b : ℕ) : ℕ :=
  (b - a) / 2 + 1

theorem sum_and_count (x y : ℕ) (hx : x = sum_of_integers 30 50) (hy : y = count_even_integers 30 50) :
  x + y = 851 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_sum_and_count_l882_88247


namespace NUMINAMATH_GPT_matrix_unique_solution_l882_88260

-- Definitions for the conditions given in the problem
def vec_i : Fin 3 → ℤ := ![1, 0, 0]
def vec_j : Fin 3 → ℤ := ![0, 1, 0]
def vec_k : Fin 3 → ℤ := ![0, 0, 1]

def matrix_M : Matrix (Fin 3) (Fin 3) ℤ := ![
  ![5, -3, 8],
  ![4, 6, -2],
  ![-9, 0, 5]
]

-- Define the target vectors
def target_i : Fin 3 → ℤ := ![5, 4, -9]
def target_j : Fin 3 → ℤ := ![-3, 6, 0]
def target_k : Fin 3 → ℤ := ![8, -2, 5]

-- The statement of the proof
theorem matrix_unique_solution : 
  (matrix_M.mulVec vec_i = target_i) ∧
  (matrix_M.mulVec vec_j = target_j) ∧
  (matrix_M.mulVec vec_k = target_k) :=
  by {
    sorry
  }

end NUMINAMATH_GPT_matrix_unique_solution_l882_88260


namespace NUMINAMATH_GPT_rectangular_coords_of_neg_theta_l882_88203

theorem rectangular_coords_of_neg_theta 
  (x y z : ℝ) 
  (rho theta phi : ℝ)
  (hx : x = 8)
  (hy : y = 6)
  (hz : z = -3)
  (h_rho : rho = Real.sqrt (x^2 + y^2 + z^2))
  (h_cos_phi : Real.cos phi = z / rho)
  (h_sin_phi : Real.sin phi = Real.sqrt (1 - (Real.cos phi)^2))
  (h_tan_theta : Real.tan theta = y / x) :
  (rho * Real.sin phi * Real.cos (-theta), rho * Real.sin phi * Real.sin (-theta), rho * Real.cos phi) = (8, -6, -3) := 
  sorry

end NUMINAMATH_GPT_rectangular_coords_of_neg_theta_l882_88203


namespace NUMINAMATH_GPT_prove_fraction_identity_l882_88218

theorem prove_fraction_identity (x y : ℂ) (h : (x + y) / (x - y) + (x - y) / (x + y) = 1) : 
  (x^4 + y^4) / (x^4 - y^4) + (x^4 - y^4) / (x^4 + y^4) = 41 / 20 := 
by 
  sorry

end NUMINAMATH_GPT_prove_fraction_identity_l882_88218


namespace NUMINAMATH_GPT_wall_building_problem_l882_88234

theorem wall_building_problem 
    (num_workers_1 : ℕ) (length_wall_1 : ℕ) (days_1 : ℕ)
    (num_workers_2 : ℕ) (length_wall_2 : ℕ) (days_2 : ℕ) :
    num_workers_1 = 8 → length_wall_1 = 140 → days_1 = 42 →
    num_workers_2 = 30 → length_wall_2 = 100 →
    (work_done : ℕ → ℕ → ℕ) → 
    (work_done length_wall_1 days_1 = num_workers_1 * days_1 * length_wall_1) →
    (work_done length_wall_2 days_2 = num_workers_2 * days_2 * length_wall_2) →
    (days_2 = 8) :=
by
  intros h1 h2 h3 h4 h5 wf wlen1 wlen2
  sorry

end NUMINAMATH_GPT_wall_building_problem_l882_88234


namespace NUMINAMATH_GPT_find_rate_percent_l882_88213

theorem find_rate_percent 
  (P : ℝ) 
  (r : ℝ) 
  (h1 : 2420 = P * (1 + r / 100)^2) 
  (h2 : 3025 = P * (1 + r / 100)^3) : 
  r = 25 :=
by
  sorry

end NUMINAMATH_GPT_find_rate_percent_l882_88213


namespace NUMINAMATH_GPT_total_time_required_l882_88225

noncomputable def walking_speed_flat : ℝ := 4
noncomputable def walking_speed_uphill : ℝ := walking_speed_flat * 0.8

noncomputable def running_speed_flat : ℝ := 8
noncomputable def running_speed_uphill : ℝ := running_speed_flat * 0.7

noncomputable def distance_walked_uphill : ℝ := 2
noncomputable def distance_run_uphill : ℝ := 1
noncomputable def distance_run_flat : ℝ := 1

noncomputable def time_walk_uphill := distance_walked_uphill / walking_speed_uphill
noncomputable def time_run_uphill := distance_run_uphill / running_speed_uphill
noncomputable def time_run_flat := distance_run_flat / running_speed_flat

noncomputable def total_time := time_walk_uphill + time_run_uphill + time_run_flat

theorem total_time_required :
  total_time = 0.9286 := by
  sorry

end NUMINAMATH_GPT_total_time_required_l882_88225


namespace NUMINAMATH_GPT_opposite_of_pi_l882_88264

theorem opposite_of_pi : -1 * Real.pi = -Real.pi := 
by sorry

end NUMINAMATH_GPT_opposite_of_pi_l882_88264


namespace NUMINAMATH_GPT_ashley_family_spending_l882_88294

theorem ashley_family_spending:
  let child_ticket := 4.25
  let adult_ticket := child_ticket + 3.50
  let senior_ticket := adult_ticket - 1.75
  let morning_discount := 0.10
  let total_morning_tickets := 2 * adult_ticket + 4 * child_ticket + senior_ticket
  let morning_tickets_after_discount := total_morning_tickets * (1 - morning_discount)
  let buy_2_get_1_free_discount := child_ticket
  let discount_for_5_or_more := 4.00
  let total_tickets_after_vouchers := morning_tickets_after_discount - buy_2_get_1_free_discount - discount_for_5_or_more
  let popcorn := 5.25
  let soda := 3.50
  let candy := 4.00
  let concession_total := 3 * popcorn + 2 * soda + candy
  let concession_discount := concession_total * 0.10
  let concession_after_discount := concession_total - concession_discount
  let final_total := total_tickets_after_vouchers + concession_after_discount
  final_total = 50.47 := by
  sorry

end NUMINAMATH_GPT_ashley_family_spending_l882_88294


namespace NUMINAMATH_GPT_sum_of_vars_l882_88250

theorem sum_of_vars 
  (x y z : ℝ) 
  (h1 : x + y = 4) 
  (h2 : y + z = 6) 
  (h3 : z + x = 8) : 
  x + y + z = 9 := 
by 
  sorry

end NUMINAMATH_GPT_sum_of_vars_l882_88250


namespace NUMINAMATH_GPT_find_positive_n_l882_88275

theorem find_positive_n (n x : ℝ) (h : 16 * x ^ 2 + n * x + 4 = 0) : n = 16 :=
by
  sorry

end NUMINAMATH_GPT_find_positive_n_l882_88275


namespace NUMINAMATH_GPT_lowest_height_l882_88274

noncomputable def length_A : ℝ := 2.4
noncomputable def length_B : ℝ := 3.2
noncomputable def length_C : ℝ := 2.8

noncomputable def height_Eunji : ℝ := 8 * length_A
noncomputable def height_Namjoon : ℝ := 4 * length_B
noncomputable def height_Hoseok : ℝ := 5 * length_C

theorem lowest_height :
  height_Namjoon = 12.8 ∧ 
  height_Namjoon < height_Eunji ∧ 
  height_Namjoon < height_Hoseok :=
by
  sorry

end NUMINAMATH_GPT_lowest_height_l882_88274


namespace NUMINAMATH_GPT_translated_coordinates_of_B_l882_88265

-- Definitions and conditions
def pointA : ℝ × ℝ := (-2, 3)

def translate_right (x : ℝ) (units : ℝ) : ℝ := x + units
def translate_down (y : ℝ) (units : ℝ) : ℝ := y - units

-- Theorem statement
theorem translated_coordinates_of_B :
  let Bx := translate_right (-2) 3
  let By := translate_down 3 5
  (Bx, By) = (1, -2) :=
by
  -- This is where the proof would go, but we're using sorry to skip the proof steps.
  sorry

end NUMINAMATH_GPT_translated_coordinates_of_B_l882_88265


namespace NUMINAMATH_GPT_min_value_geq_4_plus_2sqrt2_l882_88200

theorem min_value_geq_4_plus_2sqrt2
  (a b c : ℝ)
  (h1: a > 0)
  (h2: b > 0)
  (h3: c > 1)
  (h4: a + b = 1) :
  ( ( (a^2 + 1) / (a * b) - 2 ) * c + (Real.sqrt 2) / (c - 1) ) ≥ (4 + 2 * (Real.sqrt 2)) :=
sorry

end NUMINAMATH_GPT_min_value_geq_4_plus_2sqrt2_l882_88200


namespace NUMINAMATH_GPT_stock_decrease_required_l882_88208

theorem stock_decrease_required (x : ℝ) (h : x > 0) : 
  (∃ (p : ℝ), (1 - p) * 1.40 * x = x ∧ p * 100 = 28.57) :=
sorry

end NUMINAMATH_GPT_stock_decrease_required_l882_88208


namespace NUMINAMATH_GPT_radius_of_sphere_l882_88261

-- Define the conditions.
def radius_wire : ℝ := 8
def length_wire : ℝ := 36

-- Given the volume of the metallic sphere is equal to the volume of the wire,
-- Prove that the radius of the sphere is 12 cm.
theorem radius_of_sphere (r_wire : ℝ) (h_wire : ℝ) (r_sphere : ℝ) : 
    r_wire = radius_wire → h_wire = length_wire →
    (π * r_wire^2 * h_wire = (4/3) * π * r_sphere^3) → 
    r_sphere = 12 :=
by
  intros h₁ h₂ h₃
  -- Add proof steps here.
  sorry

end NUMINAMATH_GPT_radius_of_sphere_l882_88261


namespace NUMINAMATH_GPT_interval_monotonically_increasing_range_g_l882_88227

noncomputable def f (x : ℝ) : ℝ :=
  2 * (Real.sqrt 3) * Real.sin (x + (Real.pi / 4)) * Real.cos (x + (Real.pi / 4)) + Real.sin (2 * x) - 1

noncomputable def g (x : ℝ) : ℝ :=
  2 * Real.sin (2 * x + (2 * Real.pi / 3)) - 1

theorem interval_monotonically_increasing :
  ∃ (k : ℤ), ∀ (x : ℝ), (k * Real.pi - (5 * Real.pi / 12) ≤ x ∧ x ≤ k * Real.pi + (Real.pi / 12)) → 0 ≤ deriv f x :=
sorry

theorem range_g (m : ℝ) : 
  ∃ (x : ℝ), (0 ≤ x ∧ x ≤ Real.pi / 2) → g x = m ↔ -3 ≤ m ∧ m ≤ Real.sqrt 3 - 1 :=
sorry

end NUMINAMATH_GPT_interval_monotonically_increasing_range_g_l882_88227


namespace NUMINAMATH_GPT_small_triangle_count_l882_88286

theorem small_triangle_count (n : ℕ) (h : n = 2009) : (2 * n + 1) = 4019 := 
by {
    sorry
}

end NUMINAMATH_GPT_small_triangle_count_l882_88286


namespace NUMINAMATH_GPT_calculation_correct_l882_88290

theorem calculation_correct : 
  (3 * 4 * 5) * ((1 / 3) + (1 / 4) + (1 / 5)) = 47 := 
by
  sorry

end NUMINAMATH_GPT_calculation_correct_l882_88290


namespace NUMINAMATH_GPT_dependent_variable_is_temperature_l882_88237

-- Define the variables involved in the problem
variables (intensity_of_sunlight : ℝ)
variables (temperature_of_water : ℝ)
variables (duration_of_exposure : ℝ)
variables (capacity_of_heater : ℝ)

-- Define the conditions
def changes_with_duration (temp: ℝ) (duration: ℝ) : Prop :=
  ∃ f : ℝ → ℝ, (∀ d, temp = f d) ∧ ∀ d₁ d₂, d₁ ≠ d₂ → f d₁ ≠ f d₂

-- The theorem we need to prove
theorem dependent_variable_is_temperature :
  changes_with_duration temperature_of_water duration_of_exposure → 
  (∀ t, ∃ d, temperature_of_water = t → duration_of_exposure = d) :=
sorry

end NUMINAMATH_GPT_dependent_variable_is_temperature_l882_88237


namespace NUMINAMATH_GPT_volume_of_inequality_region_l882_88270

-- Define the inequality condition as a predicate
def region (x y z : ℝ) : Prop :=
  |4 * x - 20| + |3 * y + 9| + |z - 2| ≤ 6

-- Define the volume calculation for the region
def volume_of_region := 36

-- The proof statement
theorem volume_of_inequality_region : 
  (∃ x y z : ℝ, region x y z) → volume_of_region = 36 :=
by
  sorry

end NUMINAMATH_GPT_volume_of_inequality_region_l882_88270


namespace NUMINAMATH_GPT_fishbowl_count_l882_88221

def number_of_fishbowls (total_fish : ℕ) (fish_per_bowl : ℕ) : ℕ :=
  total_fish / fish_per_bowl

theorem fishbowl_count (h1 : 23 > 0) (h2 : 6003 % 23 = 0) :
  number_of_fishbowls 6003 23 = 261 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_fishbowl_count_l882_88221


namespace NUMINAMATH_GPT_room_length_l882_88282

theorem room_length
  (width : ℝ)
  (cost_rate : ℝ)
  (total_cost : ℝ)
  (h_width : width = 4)
  (h_cost_rate : cost_rate = 850)
  (h_total_cost : total_cost = 18700) :
  ∃ L : ℝ, L = 5.5 ∧ total_cost = cost_rate * (L * width) :=
by
  sorry

end NUMINAMATH_GPT_room_length_l882_88282


namespace NUMINAMATH_GPT_power_cycle_i_l882_88217

theorem power_cycle_i (i : ℂ) (h1 : i^1 = i) (h2 : i^2 = -1) (h3 : i^3 = -i) (h4 : i^4 = 1) :
  i^23 + i^75 = -2 * i :=
by
  sorry

end NUMINAMATH_GPT_power_cycle_i_l882_88217


namespace NUMINAMATH_GPT_combined_soldiers_correct_l882_88244

-- Define the parameters for the problem
def interval : ℕ := 5
def wall_length : ℕ := 7300
def soldiers_per_tower : ℕ := 2

-- Calculate the number of towers and the total number of soldiers
def num_towers : ℕ := wall_length / interval
def combined_soldiers : ℕ := num_towers * soldiers_per_tower

-- Prove that the combined number of soldiers is as expected
theorem combined_soldiers_correct : combined_soldiers = 2920 := 
by
  sorry

end NUMINAMATH_GPT_combined_soldiers_correct_l882_88244


namespace NUMINAMATH_GPT_paige_mp3_player_songs_l882_88277

/--
Paige had 11 songs on her mp3 player.
She deleted 9 old songs.
She added 8 new songs.

We are to prove:
- The final number of songs on her mp3 player is 10.
-/
theorem paige_mp3_player_songs (initial_songs deleted_songs added_songs final_songs : ℕ)
  (h₁ : initial_songs = 11)
  (h₂ : deleted_songs = 9)
  (h₃ : added_songs = 8) :
  final_songs = initial_songs - deleted_songs + added_songs :=
by
  sorry

end NUMINAMATH_GPT_paige_mp3_player_songs_l882_88277


namespace NUMINAMATH_GPT_f_no_zero_point_l882_88287

noncomputable def f (x : ℝ) : ℝ := Real.exp x - x

theorem f_no_zero_point (x : ℝ) (h : x > 0) : f x ≠ 0 :=
by 
  sorry

end NUMINAMATH_GPT_f_no_zero_point_l882_88287


namespace NUMINAMATH_GPT_Ted_age_48_l882_88239

/-- Given ages problem:
 - t is Ted's age
 - s is Sally's age
 - a is Alex's age 
 - The following conditions hold:
   1. t = 2s + 17 
   2. a = s / 2
   3. t + s + a = 72
 - Prove that Ted's age (t) is 48.
-/ 
theorem Ted_age_48 {t s a : ℕ} (h1 : t = 2 * s + 17) (h2 : a = s / 2) (h3 : t + s + a = 72) : t = 48 := by
  sorry

end NUMINAMATH_GPT_Ted_age_48_l882_88239


namespace NUMINAMATH_GPT_largest_multiple_of_8_less_than_neg_63_l882_88241

theorem largest_multiple_of_8_less_than_neg_63 : 
  ∃ n : ℤ, (n < -63) ∧ (∃ k : ℤ, n = 8 * k) ∧ (∀ m : ℤ, (m < -63) ∧ (∃ l : ℤ, m = 8 * l) → m ≤ n) :=
sorry

end NUMINAMATH_GPT_largest_multiple_of_8_less_than_neg_63_l882_88241


namespace NUMINAMATH_GPT_determine_numbers_l882_88279

theorem determine_numbers (a b c : ℕ) (h₁ : a + b + c = 15) 
  (h₂ : (1 / (a : ℝ)) + (1 / (b : ℝ)) + (1 / (c : ℝ)) = 71 / 105) : 
  (a = 3 ∧ b = 5 ∧ c = 7) ∨ (a = 3 ∧ b = 7 ∧ c = 5) ∨ (a = 5 ∧ b = 3 ∧ c = 7) ∨ 
  (a = 5 ∧ b = 7 ∧ c = 3) ∨ (a = 7 ∧ b = 3 ∧ c = 5) ∨ (a = 7 ∧ b = 5 ∧ c = 3) :=
sorry

end NUMINAMATH_GPT_determine_numbers_l882_88279


namespace NUMINAMATH_GPT_sum_of_perimeters_of_squares_l882_88219

theorem sum_of_perimeters_of_squares (x y : ℕ)
  (h1 : x^2 - y^2 = 19) : 4 * x + 4 * y = 76 := 
by
  sorry

end NUMINAMATH_GPT_sum_of_perimeters_of_squares_l882_88219


namespace NUMINAMATH_GPT_kelly_games_giveaway_l882_88209

theorem kelly_games_giveaway (n m g : ℕ) (h_current: n = 50) (h_left: m = 35) : g = n - m :=
by
  sorry

end NUMINAMATH_GPT_kelly_games_giveaway_l882_88209


namespace NUMINAMATH_GPT_pet_store_cages_l882_88202

theorem pet_store_cages (init_puppies sold_puppies puppies_per_cage : ℕ)
  (h1 : init_puppies = 18)
  (h2 : sold_puppies = 3)
  (h3 : puppies_per_cage = 5) :
  (init_puppies - sold_puppies) / puppies_per_cage = 3 :=
by
  sorry

end NUMINAMATH_GPT_pet_store_cages_l882_88202


namespace NUMINAMATH_GPT_todd_ratio_boss_l882_88235

theorem todd_ratio_boss
  (total_cost : ℕ)
  (boss_contribution : ℕ)
  (employees_contribution : ℕ)
  (num_employees : ℕ)
  (each_employee_pay : ℕ) 
  (total_contributed : ℕ)
  (todd_contribution : ℕ) :
  total_cost = 100 →
  boss_contribution = 15 →
  num_employees = 5 →
  each_employee_pay = 11 →
  total_contributed = num_employees * each_employee_pay + boss_contribution →
  todd_contribution = total_cost - total_contributed →
  (todd_contribution : ℚ) / (boss_contribution : ℚ) = 2 := by
  sorry

end NUMINAMATH_GPT_todd_ratio_boss_l882_88235


namespace NUMINAMATH_GPT_probability_ratio_l882_88273

-- Defining the total number of cards and each number's frequency
def total_cards := 60
def each_number_frequency := 4
def distinct_numbers := 15

-- Defining probability p' and q'
def p' := (15: ℕ) * (Nat.choose 4 4) / (Nat.choose 60 4)
def q' := 210 * (Nat.choose 4 3) * (Nat.choose 4 1) / (Nat.choose 60 4)

-- Prove the value of q'/p'
theorem probability_ratio : (q' / p') = 224 := by
  sorry

end NUMINAMATH_GPT_probability_ratio_l882_88273


namespace NUMINAMATH_GPT_first_investment_percentage_l882_88246

theorem first_investment_percentage :
  let total_inheritance := 4000
  let invested_6_5 := 1800
  let interest_rate_6_5 := 0.065
  let total_interest := 227
  let remaining_investment := total_inheritance - invested_6_5
  let interest_from_6_5 := invested_6_5 * interest_rate_6_5
  let interest_from_remaining := total_interest - interest_from_6_5
  let P := interest_from_remaining / remaining_investment
  P = 0.05 :=
by 
  sorry

end NUMINAMATH_GPT_first_investment_percentage_l882_88246


namespace NUMINAMATH_GPT_greatest_m_value_l882_88271

theorem greatest_m_value (x y m : ℝ) 
  (h₁: x^2 + y^2 = 1)
  (h₂ : |x^3 - y^3| + |x - y| = m^3) : 
  m ≤ 2^(1/3) :=
sorry

end NUMINAMATH_GPT_greatest_m_value_l882_88271


namespace NUMINAMATH_GPT_range_of_m_l882_88268

noncomputable def f (x : ℝ) : ℝ := Real.exp x + Real.exp (-x)

theorem range_of_m (m : ℝ) : (∀ x > 0, m * f x ≤ Real.exp (-x) + m - 1) ↔ m ≤ -1/3 :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l882_88268


namespace NUMINAMATH_GPT_max_sum_of_squares_l882_88258

theorem max_sum_of_squares :
  ∃ m n : ℕ, (m ∈ Finset.range 101) ∧ (n ∈ Finset.range 101) ∧ ((n^2 - n * m - m^2)^2 = 1) ∧ (m^2 + n^2 = 10946) :=
by
  sorry

end NUMINAMATH_GPT_max_sum_of_squares_l882_88258


namespace NUMINAMATH_GPT_monotonically_decreasing_when_a_half_l882_88248

noncomputable def f (x a : ℝ) : ℝ := x * (Real.log x - a * x)

theorem monotonically_decreasing_when_a_half :
  ∀ x : ℝ, 0 < x → (f x (1 / 2)) ≤ 0 :=
by
  sorry

end NUMINAMATH_GPT_monotonically_decreasing_when_a_half_l882_88248


namespace NUMINAMATH_GPT_quadratic_inequality_solution_l882_88229

theorem quadratic_inequality_solution (a : ℝ) :
  (∀ x : ℝ, ax^2 + 2*x + a > 0 ↔ x ≠ -1/a) → a = 1 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_inequality_solution_l882_88229


namespace NUMINAMATH_GPT_books_remaining_in_library_l882_88253

def initial_books : ℕ := 250
def books_taken_out_Tuesday : ℕ := 120
def books_returned_Wednesday : ℕ := 35
def books_withdrawn_Thursday : ℕ := 15

theorem books_remaining_in_library :
  initial_books
  - books_taken_out_Tuesday
  + books_returned_Wednesday
  - books_withdrawn_Thursday = 150 :=
by
  sorry

end NUMINAMATH_GPT_books_remaining_in_library_l882_88253


namespace NUMINAMATH_GPT_sum_first_6_is_correct_l882_88257

namespace ProofProblem

def sequence (a : ℕ → ℚ) : Prop :=
  (a 1 = 1) ∧ ∀ n : ℕ, n ≥ 2 → a (n - 1) = 2 * a n

def sum_first_6 (a : ℕ → ℚ) : ℚ :=
  a 1 + a 2 + a 3 + a 4 + a 5 + a 6

theorem sum_first_6_is_correct (a : ℕ → ℚ) (h : sequence a) :
  sum_first_6 a = 63 / 32 :=
sorry

end ProofProblem

end NUMINAMATH_GPT_sum_first_6_is_correct_l882_88257


namespace NUMINAMATH_GPT_find_x_l882_88220

theorem find_x (x : ℝ) : (∀ y : ℝ, 10 * x * y - 15 * y + 3 * x - y^2 - 4.5 = 0) → x = 1.5 := 
by 
  sorry

end NUMINAMATH_GPT_find_x_l882_88220


namespace NUMINAMATH_GPT_a_2016_is_neg1_l882_88233

noncomputable def a : ℕ → ℤ
| 0     => 0 -- Arbitrary value for n = 0 since sequences generally start from 1 in Lean
| 1     => 1
| 2     => 2
| n + 1 => a n - a (n - 1)

theorem a_2016_is_neg1 : a 2016 = -1 := sorry

end NUMINAMATH_GPT_a_2016_is_neg1_l882_88233


namespace NUMINAMATH_GPT_domain_of_f_l882_88266

noncomputable def f (x : ℝ) : ℝ := 1 / (Real.sqrt (1 - x^2)) + x^0

theorem domain_of_f :
  {x : ℝ | 1 - x^2 > 0 ∧ x ≠ 0} = {x : ℝ | -1 < x ∧ x < 1 ∧ x ≠ 0} :=
by
  sorry

end NUMINAMATH_GPT_domain_of_f_l882_88266


namespace NUMINAMATH_GPT_price_difference_l882_88223

-- Define the prices of commodity X and Y in the year 2001 + n.
def P_X (n : ℕ) (a : ℝ) : ℝ := 4.20 + 0.45 * n + a * n
def P_Y (n : ℕ) (b : ℝ) : ℝ := 6.30 + 0.20 * n + b * n

-- Define the main theorem to prove
theorem price_difference (n : ℕ) (a b : ℝ) :
  P_X n a = P_Y n b + 0.65 ↔ (0.25 + a - b) * n = 2.75 :=
by
  sorry

end NUMINAMATH_GPT_price_difference_l882_88223


namespace NUMINAMATH_GPT_x_eq_1_sufficient_but_not_necessary_l882_88214

theorem x_eq_1_sufficient_but_not_necessary (x : ℝ) : x^2 - 3 * x + 2 = 0 → (x = 1 ↔ true) ∧ (x ≠ 1 → ∃ y : ℝ, y ≠ x ∧ y^2 - 3 * y + 2 = 0) :=
by
  sorry

end NUMINAMATH_GPT_x_eq_1_sufficient_but_not_necessary_l882_88214


namespace NUMINAMATH_GPT_interval_length_l882_88245

theorem interval_length (c d : ℝ) (h : (d - 5) / 3 - (c - 5) / 3 = 15) : d - c = 45 :=
sorry

end NUMINAMATH_GPT_interval_length_l882_88245


namespace NUMINAMATH_GPT_wall_cost_equal_l882_88283

theorem wall_cost_equal (A B C : ℝ) (d_1 d_2 : ℝ) (h1 : A = B) (h2 : B = C) : d_1 = d_2 :=
by
  -- sorry is used to skip the proof
  sorry

end NUMINAMATH_GPT_wall_cost_equal_l882_88283


namespace NUMINAMATH_GPT_fraction_product_l882_88242

theorem fraction_product (a b c d e : ℝ) (h1 : a = 1/2) (h2 : b = 1/3) (h3 : c = 1/4) (h4 : d = 1/6) (h5 : e = 144) :
  a * b * c * d * e = 1 := 
by
  -- Given the conditions h1 to h5, we aim to prove the product is 1
  sorry

end NUMINAMATH_GPT_fraction_product_l882_88242


namespace NUMINAMATH_GPT_equation_solution_l882_88232

noncomputable def solve_equation : Prop :=
∃ (x : ℝ), x^6 + (3 - x)^6 = 730 ∧ (x = 1.5 + Real.sqrt 5 ∨ x = 1.5 - Real.sqrt 5)

theorem equation_solution : solve_equation :=
sorry

end NUMINAMATH_GPT_equation_solution_l882_88232


namespace NUMINAMATH_GPT_number_of_valid_m_values_l882_88212

noncomputable def polynomial (m : ℤ) (x : ℤ) : ℤ := 
  2 * (m - 1) * x ^ 2 - (m ^ 2 - m + 12) * x + 6 * m

noncomputable def discriminant (m : ℤ) : ℤ :=
  (m ^ 2 - m + 12) ^ 2 - 4 * 2 * (m - 1) * 6 * m

def is_perfect_square (n : ℤ) : Prop :=
  ∃ (k : ℤ), k * k = n

def has_integral_roots (m : ℤ) : Prop :=
  ∃ (r1 r2 : ℤ), polynomial m r1 = 0 ∧ polynomial m r2 = 0

def valid_m_values (m : ℤ) : Prop :=
  (discriminant m) > 0 ∧ is_perfect_square (discriminant m) ∧ has_integral_roots m

theorem number_of_valid_m_values : 
  (∃ M : List ℤ, (∀ m ∈ M, valid_m_values m) ∧ M.length = 4) :=
  sorry

end NUMINAMATH_GPT_number_of_valid_m_values_l882_88212


namespace NUMINAMATH_GPT_find_a7_a8_a9_l882_88285

variable {α : Type*} [LinearOrderedField α]

-- Definitions of the conditions
def is_arithmetic_sequence (a : ℕ → α) : Prop :=
  ∃ d, ∀ n, a (n + 1) = a n + d

def sum_of_arithmetic_sequence (a : ℕ → α) (n : ℕ) : α :=
  n * a 1 + (n * (n - 1) / 2) * (a 2 - a 1)

variables {a : ℕ → α}
variables (S : ℕ → α)
variables (S_3 S_6 : α)

-- Given conditions
axiom is_arith_seq : is_arithmetic_sequence a
axiom S_def : ∀ n, S n = sum_of_arithmetic_sequence a n
axiom S_3_eq : S 3 = 9
axiom S_6_eq : S 6 = 36

-- Theorem to prove
theorem find_a7_a8_a9 : a 7 + a 8 + a 9 = 45 :=
sorry

end NUMINAMATH_GPT_find_a7_a8_a9_l882_88285


namespace NUMINAMATH_GPT_expand_and_simplify_l882_88215

theorem expand_and_simplify (x : ℝ) : 
  (2 * x + 6) * (x + 10) = 2 * x^2 + 26 * x + 60 :=
sorry

end NUMINAMATH_GPT_expand_and_simplify_l882_88215


namespace NUMINAMATH_GPT_f_g_2_eq_neg_19_l882_88288

def f (x : ℝ) : ℝ := 5 - 4 * x

def g (x : ℝ) : ℝ := x^2 + 2

theorem f_g_2_eq_neg_19 : f (g 2) = -19 := 
by
  -- The proof is omitted
  sorry

end NUMINAMATH_GPT_f_g_2_eq_neg_19_l882_88288


namespace NUMINAMATH_GPT_tax_free_amount_l882_88238

theorem tax_free_amount (X : ℝ) (total_value : ℝ) (tax_paid : ℝ) 
    (tax_rate : ℝ) (exceeds_value : ℝ) :
    total_value = 1720 → 
    tax_rate = 0.11 → 
    tax_paid = 123.2 → 
    total_value - X = exceeds_value → 
    tax_paid = tax_rate * exceeds_value → 
    X = 600 := 
by
  intros h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_tax_free_amount_l882_88238


namespace NUMINAMATH_GPT_kyle_gas_and_maintenance_expense_l882_88210

def monthly_income : ℝ := 3200
def rent : ℝ := 1250
def utilities : ℝ := 150
def retirement_savings : ℝ := 400
def groceries_eating_out : ℝ := 300
def insurance : ℝ := 200
def miscellaneous_expenses : ℝ := 200
def car_payment : ℝ := 350

def total_bills : ℝ := rent + utilities + retirement_savings + groceries_eating_out + insurance + miscellaneous_expenses

theorem kyle_gas_and_maintenance_expense :
  monthly_income - total_bills - car_payment = 350 :=
by
  sorry

end NUMINAMATH_GPT_kyle_gas_and_maintenance_expense_l882_88210


namespace NUMINAMATH_GPT_solution_set_inequality_l882_88254

theorem solution_set_inequality (x : ℝ) : 
  (∃ x, (x-1)/((x^2) - x - 30) > 0) ↔ (x > -5 ∧ x < 1) ∨ (x > 6) :=
by
  sorry

end NUMINAMATH_GPT_solution_set_inequality_l882_88254
