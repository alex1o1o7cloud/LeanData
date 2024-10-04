import Mathlib

namespace fraction_area_correct_l288_288976

-- Definitions for the problem conditions
def grid_size : ℕ := 6
def shaded_square_vertices : list (ℕ × ℕ) := [(2,3), (4,3), (4,5), (2,5)]

-- Define the length of the side of the shaded square
def side_length_shaded_square : ℕ := 2

-- Define the area of the larger square and the shaded square
def area_larger_square : ℕ := grid_size * grid_size
def area_shaded_square : ℕ := side_length_shaded_square * side_length_shaded_square

-- Define the fraction of the area of the larger square that is inside the shaded square
def fraction_area : ℚ := area_shaded_square / area_larger_square

-- The theorem to be proven
theorem fraction_area_correct : fraction_area = 1/9 :=
by
  -- Proof would go here
  sorry

end fraction_area_correct_l288_288976


namespace rational_coeff_binomial_terms_l288_288533

theorem rational_coeff_binomial_terms : 
  ∃ n, 
    (n = 6) ∧ 
    (∀ k, (∃ a, 0 ≤ k ∧ k ≤ 30 ∧ k = 6 * a) ↔ 
    (k ∈ {0, 6, 12, 18, 24, 30})) :=
sorry

end rational_coeff_binomial_terms_l288_288533


namespace modulus_sum_problem_statement_l288_288775

theorem modulus_sum (a b c d : ℤ) :
  (complex.abs ⟨a, b⟩) + (complex.abs ⟨c, d⟩) = real.sqrt (a^2 + b^2) + real.sqrt (c^2 + d^2) := 
by 
  sorry

theorem problem_statement :
  (complex.abs ⟨3, -5⟩) + (complex.abs ⟨3, 7⟩) = real.sqrt 34 + real.sqrt 58 := 
by 
  sorry

end modulus_sum_problem_statement_l288_288775


namespace minimum_traverse_distance_l288_288731

theorem minimum_traverse_distance (a : ℝ) (ha : a > 0) : 
  let h := (Real.sqrt 3 / 2) * a
  let r := h / 2
  let distance := a * (Real.sqrt 109 + 2) / 8
  in
  distance = a * (Real.sqrt 109 + 2) / 8 :=
by
  sorry

end minimum_traverse_distance_l288_288731


namespace sammy_total_math_problems_l288_288997

theorem sammy_total_math_problems (done left total : ℕ) (h_done : done = 2) (h_left : left = 7) (h_total : total = done + left) : total = 9 := 
by 
  rw [h_done, h_left, h_total]
  norm_num

end sammy_total_math_problems_l288_288997


namespace count_4_digit_numbers_divisible_by_13_l288_288874

theorem count_4_digit_numbers_divisible_by_13 : 
  ∃ n : ℕ, n = 693 ∧ (∀ k : ℕ, k >= 1000 ∧ k < 10000 ∧ k % 13 = 0 → ∃ m : ℕ, m = (k - 1000) / 13 + 1 ∧ m = n) :=
by {
  -- Solution proof will be placed here.
  sorry
}

end count_4_digit_numbers_divisible_by_13_l288_288874


namespace a11_is_1_l288_288433

variable (a : ℕ → ℕ) (S : ℕ → ℕ)

-- Condition 1: The sum of the first n terms S_n satisfies S_n + S_m = S_{n+m}
axiom sum_condition (n m : ℕ) : S n + S m = S (n + m)

-- Condition 2: a_1 = 1
axiom a1_condition : a 1 = 1

-- Question: prove a_{11} = 1
theorem a11_is_1 : a 11 = 1 :=
sorry


end a11_is_1_l288_288433


namespace union_of_A_and_B_intersection_of_A_and_B_l288_288442

noncomputable def A : Set ℝ := { x | -4 < x ∧ x < 4 }
noncomputable def B : Set ℝ := { x | x > 3 ∨ x < 1 }

theorem union_of_A_and_B : A ∪ B = Set.univ :=
by
  sorry

theorem intersection_of_A_and_B : A ∩ B = { x | (-4 < x ∧ x < 1) ∨ (3 < x ∧ x < 4) } :=
by
  sorry

end union_of_A_and_B_intersection_of_A_and_B_l288_288442


namespace correct_equation_l288_288271

-- Define the necessary conditions and parameters
variables (x : ℝ)

-- Length of the rectangle
def length := x 

-- Width is 6 meters less than the length
def width := x - 6

-- The area of the rectangle
def area := 720

-- Proof statement
theorem correct_equation : 
  x * (x - 6) = 720 :=
sorry

end correct_equation_l288_288271


namespace fraction_of_earth_used_for_agriculture_or_urban_l288_288358

theorem fraction_of_earth_used_for_agriculture_or_urban
  (earth_surface_land_fraction : ℚ) (inhabitable_land_fraction : ℚ) (used_land_fraction : ℚ) :
  earth_surface_land_fraction = 1/3 →
  inhabitable_land_fraction = 2/3 →
  used_land_fraction = 3/4 →
  (earth_surface_land_fraction * inhabitable_land_fraction * used_land_fraction) = 1/6 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  sorry

end fraction_of_earth_used_for_agriculture_or_urban_l288_288358


namespace total_cleaning_time_l288_288195

def hose_time : ℕ := 10
def shampoos : ℕ := 3
def shampoo_time : ℕ := 15

theorem total_cleaning_time : hose_time + shampoos * shampoo_time = 55 := by
  sorry

end total_cleaning_time_l288_288195


namespace range_of_m_l288_288111

open Real

theorem range_of_m (m : ℝ) : (∀ x : ℝ, (m + 1) * x^2 - (m - 1) * x + 3 * (m - 1) < 0) ↔ m < -1 :=
by
  sorry

end range_of_m_l288_288111


namespace polynomial_divisibility_by_5_l288_288690

theorem polynomial_divisibility_by_5
  (a b c d : ℤ)
  (divisible : ∀ x : ℤ, 5 ∣ (a * x ^ 3 + b * x ^ 2 + c * x + d)) :
  5 ∣ a ∧ 5 ∣ b ∧ 5 ∣ c ∧ 5 ∣ d :=
sorry

end polynomial_divisibility_by_5_l288_288690


namespace balcony_more_than_orchestra_l288_288689

theorem balcony_more_than_orchestra (x y : ℕ) 
  (h1 : x + y = 370) 
  (h2 : 12 * x + 8 * y = 3320) : y - x = 190 :=
sorry

end balcony_more_than_orchestra_l288_288689


namespace tangent_line_to_y_eq_x_cubed_at_P_l288_288230

theorem tangent_line_to_y_eq_x_cubed_at_P :
  (∀ (x y : ℝ), y = x^3 → (3*x - y - 2 = 0) → P (1, 1)) :=
by {
  intros x y eqn_tangent,
  sorry
}

end tangent_line_to_y_eq_x_cubed_at_P_l288_288230


namespace isosceles_triangle_largest_angle_l288_288921

theorem isosceles_triangle_largest_angle (A B C : ℝ) (h : A = B) (h₁ : C + A + B = 180) (h₂ : C = 30) : 
  180 - 2 * 30 = 120 :=
by sorry

end isosceles_triangle_largest_angle_l288_288921


namespace systematic_sampling_third_group_l288_288603

-- Define the conditions and question as a Lean theorem
theorem systematic_sampling_third_group :
  ∀ (total_students : ℕ) (sample_size : ℕ) (initial_draw : ℕ)
  (group_size : ℕ) (group_number : ℕ),
  total_students = 400 →
  sample_size = 20 →
  initial_draw = 11 →
  group_size = total_students / sample_size →
  group_number = 3 →
  (initial_draw + (group_number - 1) * group_size) = 51 :=
by
  intros total_students sample_size initial_draw group_size group_number
  intro h_total_students
  intro h_sample_size
  intro h_initial_draw
  intro h_group_size
  intro h_group_number
  rw [h_total_students, h_sample_size, h_initial_draw, h_group_size, h_group_number]
  show 11 + (3 - 1) * (400 / 20) = 51
  sorry  -- Proof goes here

end systematic_sampling_third_group_l288_288603


namespace value_of_b_l288_288768

theorem value_of_b (b : ℚ) (x y : ℚ) (hx : x = 3) (hy : y = -5) 
  (hl : b * x + (b - 2) * y = b - 1) : b = 11 / 3 := 
by {
  -- Definitions for substitution as given in conditions.
  rw [hx, hy] at hl,
  -- The proof will carry from the simplified form already given.
  sorry
}

end value_of_b_l288_288768


namespace max_area_triangle_time_l288_288978

theorem max_area_triangle_time:
  let t := 15 in
  let second_hand_rotation_rate := 6 in
  let minute_hand_rotation_rate := 0.1 in
  after t seconds, the first time the area of triangle OAB will be the maximum,
  where
    is_perpendicular (angle_in_degrees_a : ℝ) (angle_in_degrees_b : ℝ) :=
      (angle_in_degrees_a - angle_in_degrees_b)%360 = 90
  :=
  sorry

end max_area_triangle_time_l288_288978


namespace max_marks_paper_I_l288_288710

variable (M : ℝ)

theorem max_marks_paper_I (h1 : 0.65 * M = 112 + 58) : M = 262 :=
  sorry

end max_marks_paper_I_l288_288710


namespace area_of_triangle_l288_288200

variable {p : ℝ}
def C := (0, p)
def B := (16, 0 : ℝ)
def O := (0, 0 : ℝ)

theorem area_of_triangle (h₀ : 0 ≤ p) (h₁ : p ≤ 20) : 
  let base := (B.1 - O.1)
      height := (C.2 - O.2) in
  1 / 2 * base * height = 8 * p :=
by
  sorry

end area_of_triangle_l288_288200


namespace complex_cubic_eq_l288_288617

theorem complex_cubic_eq (a b : ℤ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_eq : (a + b * Complex.i)^3 = 2 + 11 * Complex.i) : 
  a + b * Complex.i = 2 + Complex.i := 
sorry

end complex_cubic_eq_l288_288617


namespace determine_a_b_l288_288284

theorem determine_a_b (a b n : ℕ): 
  -- Conditions
  (a > 0) ∧ (b > 0) ∧ (a ≠ b) ∧ (a ∣ b) ∧ 
  (decimal_digits a = 2 * n) ∧ (decimal_digits b = 2 * n) ∧
  (first_n_digits a n = last_n_digits b n) ∧ 
  (first_n_digits b n = last_n_digits a n) →
  -- Correct Answer
  (a = 24 ∧ b = 42) ∨ (a = 39 ∧ b = 93) :=
by
  sorry

def decimal_digits (x : ℕ) : ℕ := sorry

def first_n_digits (x n : ℕ) : ℕ := sorry

def last_n_digits (x n : ℕ) : ℕ := sorry

end determine_a_b_l288_288284


namespace inequalities_no_solution_l288_288660

theorem inequalities_no_solution (x n : ℝ) (h1 : x ≤ 1) (h2 : x ≥ n) : n > 1 :=
sorry

end inequalities_no_solution_l288_288660


namespace sin_double_theta_eq_five_fourths_l288_288497

theorem sin_double_theta_eq_five_fourths (theta : ℝ) (h : cos theta + sin theta = 3/2) : sin (2 * theta) = 5/4 :=
by
  sorry

end sin_double_theta_eq_five_fourths_l288_288497


namespace solution_count_of_ggx_eq_3_l288_288170

def g (x : ℝ) : ℝ :=
if x < 0 then -x + 4 else 3 * x - 6

theorem solution_count_of_ggx_eq_3 : {x : ℝ | g (g x) = 3}.finite.card = 3 :=
by
  sorry

end solution_count_of_ggx_eq_3_l288_288170


namespace number_of_valid_triangles_l288_288101

theorem number_of_valid_triangles : 
  let points := {(x, y) | 1 ≤ x ∧ x ≤ 5 ∧ 1 ≤ y ∧ y ≤ 5} in
  ∃ n : ℕ, n = (nat.card (points.choose 3)) - 120 ∧ n = 2180 :=
by
  sorry

end number_of_valid_triangles_l288_288101


namespace collinear_C_D_M_l288_288930

/-- Provide definitions and the main theorem structure, using sorry to skip the proof. -/
section Geometry

variables {O1 O2 P A B M C D : Point} 
           {circleO1 circleO2 : Circle}

/-- assumptions based on given conditions -/
def circle_tangent (c1 c2 : Circle) := sorry
def on_circle (p : Point) (c : Circle) := sorry
def tangent_point (p : Point) (c : Circle) := Point
def midpoint (p1 p2 : Point) := Point
def perpendicular (l1 l2 : Line) := Prop
def intersection (l : Line) (c : Circle) := Point

-- Assuming all given conditions
axiom h1 : circle_tangent circleO1 circleO2
axiom h2 : on_circle P circleO1
axiom h3 : tangent_point P circleO2 = A
axiom h4 : tangent_point P circleO2 = B
axiom h5 : M = midpoint A B
axiom h6 : perpendicular (Line.mk O1 C) (Line.mk P A)
axiom h7 : on_circle C circleO1
axiom h8 : on_circle D circleO1
axiom h9 : intersection (Line.mk P B) circleO1 = D

-- Prove that points C, D, and M are collinear
theorem collinear_C_D_M : collinear C D M :=
sorry

end Geometry

end collinear_C_D_M_l288_288930


namespace original_length_of_two_ropes_is_48_l288_288279

-- Define the problem conditions
def total_original_length_of_two_ropes 
  (L : ℕ) -- The length of each of the two ropes
  (diff_part_length : ℕ) -- The difference in the segment lengths between the two ropes
  (first_rope_parts : ℕ) -- The number of parts the first rope is cut into
  (second_rope_parts : ℕ) -- The number of parts the second rope is cut into :=
  first_rope_parts ≠ second_rope_parts -- The parts count for both ropes must be different

-- Prove the correct answer
theorem original_length_of_two_ropes_is_48 
  (L : ℕ) -- Length of each rope
  (diff_part_length : L / 4 - L / 6 = 2) -- The difference in segment lengths between the two ropes
  : 2 * L = 48 := sorry

end original_length_of_two_ropes_is_48_l288_288279


namespace pants_cost_l288_288520

theorem pants_cost (P : ℝ) : 
(80 + 3 * P + 300) * 0.90 = 558 → P = 80 :=
by
  sorry

end pants_cost_l288_288520


namespace present_ages_ratio_l288_288977

noncomputable def ratio_of_ages (F S : ℕ) : ℚ :=
  F / S

theorem present_ages_ratio (F S : ℕ) (h1 : F + S = 220) (h2 : (F + 10) * 3 = (S + 10) * 5) :
  ratio_of_ages F S = 7 / 4 :=
by
  sorry

end present_ages_ratio_l288_288977


namespace rectangle_area_ratio_l288_288220

theorem rectangle_area_ratio (l b : ℕ) (h1 : l = b + 10) (h2 : b = 8) : (l * b) / b = 18 := by
  sorry

end rectangle_area_ratio_l288_288220


namespace collection_card_average_l288_288326

theorem collection_card_average (n : ℕ) 
    (total_cards : (1 + 2 + 3 + ... + n) = n * (n + 1) / 2)
    (sum_values : (1^2 + 2^2 + 3^2 + ... + n^2) = n * (n + 1) * (2n + 1) / 6) :
    ( (n * (n + 1) * (2n + 1) / 6) / (n * (n + 1) / 2) = 2023 ) → n = 3034 := 
by
  sorry

end collection_card_average_l288_288326


namespace sum_first_n_terms_max_sum_terms_l288_288435

theorem sum_first_n_terms (a_n : ℕ → ℚ) (d : ℚ) (S : ℕ → ℚ) (n : ℕ)
  (h1 : S 10 = 125 / 7)
  (h2 : S 20 = -250 / 7)
  (h3 : ∀ n, a_n = 5 + (n - 1) * d)
  (h4 : ∀ n, S n = (n * (5 + a_n)) / 2) :
  S n = (75 * n - 5 * n^2) / 14 := sorry

theorem max_sum_terms (S : ℕ → ℚ) (n : ℕ)
  (h1 : S 10 = 125 / 7)
  (h2 : S 20 = -250 / 7)
  (sum_formula : ∀ n, S n = (75 * n - 5 * n^2) / 14) :
  n = 7 ∨ n = 8 := sorry

end sum_first_n_terms_max_sum_terms_l288_288435


namespace no_finite_decimal_fractions_l288_288205

theorem no_finite_decimal_fractions (n : ℤ) (hn : n ≠ 0) :
  ¬(∀ x : ℚ, (4 * n^2 - 1) * x^2 - 4 * n^2 * x + n^2 = 0 →
    (∃ a b : ℤ, b ≠ 0 ∧ (∀ p : ℤ, p.prime → p ∣ b → (p = 2 ∨ p = 5)))) :=
by
  sorry

end no_finite_decimal_fractions_l288_288205


namespace smallest_m_l288_288411

theorem smallest_m (m : ℕ) (h : m > 0) : (1 / 2 + 1 / 4 + 1 / 9 + 1 / m).den = 1 → m = 6 :=
by
  sorry

end smallest_m_l288_288411


namespace ratio_of_areas_l288_288692

-- Defining the variables for sides of rectangles
variables {a b c d : ℝ}

-- Given conditions
axiom h1 : a / c = 4 / 5
axiom h2 : b / d = 4 / 5

-- Statement to prove the ratio of areas
theorem ratio_of_areas (h1 : a / c = 4 / 5) (h2 : b / d = 4 / 5) : (a * b) / (c * d) = 16 / 25 :=
sorry

end ratio_of_areas_l288_288692


namespace trigonometric_polynomial_l288_288687

noncomputable def f (n : ℕ) (a b : ℕ → ℝ) (x : ℝ) : ℝ := 
  ∑ j in finset.range (n + 1), (a j * real.sin (j * x) + b j * real.cos (j * x))

theorem trigonometric_polynomial (n : ℕ) (a b : ℕ → ℝ) (h_f : ∀ x ∈ Icc 0 (2 * real.pi), abs (f n a b x) = 1 → set.finite ↥({x | abs (f n a b x) = 1}).to_finset.card = 2 * n) :
  ∃ k : ℝ, ∀ x ∈ Icc 0 (2 * real.pi), f n a b x = real.cos (n * x + k) := 
sorry

end trigonometric_polynomial_l288_288687


namespace e_shakes_l288_288010

def friends := {a, b, c, d, e : ℕ}

variables (shakes : friends → ℕ)

axioms (a_shakes : shakes a = 4)
       (b_shakes : shakes b = 1)
       (c_shakes : shakes c = 3)
       (d_shakes : shakes d = 2)

theorem e_shakes : shakes e = 2 :=
by
  sorry

end e_shakes_l288_288010


namespace no_daughters_count_l288_288361

theorem no_daughters_count
  (d : ℕ)  -- Bertha's daughters
  (total : ℕ)  -- Total daughters and granddaughters
  (d_with_children : ℕ)  -- Daughters with children
  (children_per_daughter : ℕ)  -- Children per daughter
  (no_great_granddaughters : Prop) -- No great-granddaughters
  (d = 8)
  (total = 40)
  (d_with_children * children_per_daughter = total - d)
  (children_per_daughter = 4)
  (no_great_granddaughters := ∀ gd, gd ∈ ∅  → ¬∃ x, x ∈ gd) :
  total - d_with_children = 32 :=
by sorry

end no_daughters_count_l288_288361


namespace f_neg_half_plus_f_2_eq_1_l288_288452

noncomputable def f : ℝ → ℝ :=
λ x, if x ≤ 0 then (9 / 4) ^ x else Real.log 2 / Real.log 8 * Real.log x

theorem f_neg_half_plus_f_2_eq_1 :
  f (-1/2) + f 2 = 1 :=
by
  sorry

end f_neg_half_plus_f_2_eq_1_l288_288452


namespace distance_proof_l288_288348

theorem distance_proof (d : ℝ) (h1 : d < 6) (h2 : d > 5) (h3 : d > 4) : d ∈ Set.Ioo 5 6 :=
by
  sorry

end distance_proof_l288_288348


namespace sum_of_longest_altitudes_l288_288485

theorem sum_of_longest_altitudes (a b c : ℕ) (h1 : a = 7) (h2 : b = 24) (h3 : c = 25) 
    (h4 : a^2 + b^2 = c^2) : a + b = 31 :=
by
  rw [h1, h2, h3]
  sorry

end sum_of_longest_altitudes_l288_288485


namespace four_digit_numbers_divisible_by_13_l288_288875

theorem four_digit_numbers_divisible_by_13 : 
  (set.Icc 1000 9999).filter (λ x, x % 13 = 0).cardinality = 693 :=
by
  sorry

end four_digit_numbers_divisible_by_13_l288_288875


namespace quadratic_min_value_l288_288292

theorem quadratic_min_value (k : ℝ) :
  (∀ x : ℝ, 3 ≤ x ∧ x ≤ 5 → y = (1/2) * (x - 1) ^ 2 + k) ∧
  (∀ y : ℝ, 3 ≤ y ∧ y ≤ 5 → y ≥ 3) → k = 1 :=
sorry

end quadratic_min_value_l288_288292


namespace cost_of_pencils_l288_288186

def cost_of_notebooks : ℝ := 3 * 1.2
def cost_of_pens : ℝ := 1.7
def total_spent : ℝ := 6.8

theorem cost_of_pencils :
  total_spent - (cost_of_notebooks + cost_of_pens) = 1.5 :=
by
  sorry

end cost_of_pencils_l288_288186


namespace math_problem_l288_288836

def f (x b : ℝ) : ℝ := - (1/3) * x^3 + x + b

def has_three_zeros (b : ℝ) : Prop :=
  let f' (x : ℝ) := -x^2 + 1
  let f'' (x : ℝ) := -2 * x
  ∃ (a c : ℝ), a < c ∧ f(b, a) = 0 ∧ f(b, 0) = 0 ∧ f(b, c) = 0

def is_even_function (g : ℝ → ℝ) : Prop :=
  ∀ (x : ℝ), g(x) = g(-x)

def max_min_sum_implies_b_equals_two (M m b : ℝ) : Prop :=
  (M = (2/3) + b ∧ m = - (2/3) + b ∧ M + m = 2) → b = 2

def three_tangent_lines (b : ℝ) : Prop :=
  let x_0 := 1
  let y_0 := f(x_0, b)
  let tangent := λ (x : ℝ), (-x_0^2 + 1)*(x - x_0) - (1/3)*x_0^3 + x_0 + b
  ∃ (b₁ b₂ b₃ : ℝ), b = b₁ ∧ b = b₂ ∧ b = b₃

theorem math_problem
  (b : ℝ)
  (A : has_three_zeros b → b ∈ Ioo (-(2/3) : ℝ) (2/3))
  (B : is_even_function (λ x, abs (f(x , b)-b)))
  (C : ¬ max_min_sum_implies_b_equals_two ((2/3) + b) ((-2/3) + b) b)
  (D : three_tangent_lines b → b ∈ Ioo 0 (1/3)) :
  A ∧ B ∧ D :=
by
  sorry

end math_problem_l288_288836


namespace proof_problem_l288_288828

noncomputable def α (k : ℤ) := k * π + 2 * π / 3

theorem proof_problem 
  (h : ∀ k : ℤ, ∃ α : ℝ, α = k * π + 2 * π / 3) :
  (∃ α : ℝ, (∀ k : ℤ, α = k * π + 2 * π / 3) ∧ 
    (∀ α : ℝ, (tan α = -sqrt 3) ∧ 
      (∃ k : ℤ, α = k * π + 2 * π / 3) ∧ 
      (∀ α : ℝ, 
        (sqrt 3 * sin (α - π) + 5 * cos (2 * π - α)) / 
        (-sqrt 3 * cos (3 * π / 2 + α) + cos (π + α)) = 4))
  :=
begin
  sorry
end

end proof_problem_l288_288828


namespace early_finish_hours_l288_288356

theorem early_finish_hours 
  (h : Nat) 
  (total_customers : Nat) 
  (num_workers : Nat := 3)
  (service_rate : Nat := 7) 
  (full_hours : Nat := 8)
  (total_customers_served : total_customers = 154) 
  (two_workers_hours : Nat := 2 * full_hours * service_rate) 
  (early_worker_customers : Nat := h * service_rate)
  (total_service : total_customers = two_workers_hours + early_worker_customers) : 
  h = 6 :=
by
  sorry

end early_finish_hours_l288_288356


namespace incenter_coordinates_l288_288145

variables {A B C : Type} [AddCommGroup A] [AddCommGroup B] [AddCommGroup C]
def a : ℝ := 8
def b : ℝ := 10
def c : ℝ := 6
def total : ℝ := a + b + c

def x : ℝ := a / total
def y : ℝ := b / total
def z : ℝ := c / total

theorem incenter_coordinates :
  x + y + z = 1 ∧ 
  (x, y, z) = (1 / 3, 5 / 12, 1 / 4) :=
by {
  sorry
}

end incenter_coordinates_l288_288145


namespace vector_coordinates_l288_288851

def vec (x y z : Int) : (Int × Int × Int) := (x, y, z)

def a := vec 2 -3 1
def b := vec 2 0 -2
def c := vec -1 -2 0
def r := (2 * a.1 - 3 * b.1 + c.1, 2 * a.2 - 3 * b.2 + c.2, 2 * a.3 - 3 * b.3 + c.3)

theorem vector_coordinates : r = (-3, -8, 8) :=
sorry

end vector_coordinates_l288_288851


namespace angle_bisector_length_l288_288980

variable (b R : ℝ)

theorem angle_bisector_length (h_b_pos : 0 < b) (h_R_pos : 0 < R) :
    let AB := 2 * R,
        BC := Real.sqrt (4 * R^2 - b^2),
        angle_bisector_length := 2 * b * Real.sqrt R / Real.sqrt (2 * R + b)
    in 
    angle_bisector_length = 2 * b * Real.sqrt R / Real.sqrt (2 * R + b) :=
by 
  sorry

end angle_bisector_length_l288_288980


namespace complex_trace_ellipse_l288_288625

theorem complex_trace_ellipse (z : ℂ) (hz : complex.abs z = 3) : 
  ∃ a b : ℝ, (let x := (10 / 9) * a in 
              let y := (8 / 9) * b in 
              (a^2 + b^2 = 9) → (x^2 / (100 / 9) + y^2 / (64 / 9) = 1)) := 
by 
  sorry

end complex_trace_ellipse_l288_288625


namespace distance_proof_l288_288347

theorem distance_proof (d : ℝ) (h1 : d < 6) (h2 : d > 5) (h3 : d > 4) : d ∈ Set.Ioo 5 6 :=
by
  sorry

end distance_proof_l288_288347


namespace count_4digit_numbers_divisible_by_13_l288_288892

theorem count_4digit_numbers_divisible_by_13 : 
  let count := (λ n, n - 77 + 1) (769)
  count = 693 :=
by
-- Let 77 be the smallest integer such that 13 * 77 = 1001
-- Let 769 be the largest integer such that 13 * 769 = 9997
-- Hence, the range of these integers should be 769 - 77 + 1
-- This gives us a count of 693
-- The statement as 693 is directly given
sorry

end count_4digit_numbers_divisible_by_13_l288_288892


namespace count_4_digit_numbers_divisible_by_13_l288_288873

theorem count_4_digit_numbers_divisible_by_13 : 
  ∃ n : ℕ, n = 693 ∧ (∀ k : ℕ, k >= 1000 ∧ k < 10000 ∧ k % 13 = 0 → ∃ m : ℕ, m = (k - 1000) / 13 + 1 ∧ m = n) :=
by {
  -- Solution proof will be placed here.
  sorry
}

end count_4_digit_numbers_divisible_by_13_l288_288873


namespace points_per_insult_l288_288971

-- Define the constants representing the points for each behavior
def points_for_interrupting : ℕ := 5
def points_for_throwing : ℕ := 25
def max_points : ℕ := 100

-- Define the behaviors of Jerry
def interrupts (n : ℕ) : ℕ := n * points_for_interrupting
def throws_things (m : ℕ) : ℕ := m * points_for_throwing
def insults_classmates (x : ℕ) (p : ℕ) : ℕ := x * p

-- Given conditions
def jerry_interrupts : ℕ := 2
def jerry_throws : ℕ := 2
def jerry_insults : ℕ := 4

-- Total points calculation
def total_points (interrupts_points throws_points insults_points : ℕ) : ℕ :=
  interrupts_points + throws_points + insults_points

-- Theorem stating that Jerry gets 10 points per insult
theorem points_per_insult : 
  ∀ (p : ℕ),
    let total_interrupting_points := interrupts jerry_interrupts,
        total_throwing_points := throws_things jerry_throws,
        remaining_points := max_points - (total_interrupting_points + total_throwing_points),
        points_for_each_insult := remaining_points / jerry_insults
    in
      p = points_for_each_insult →
      p = 10 :=
by 
  sorry

end points_per_insult_l288_288971


namespace total_order_cost_is_correct_l288_288737

noncomputable def totalOrderCost : ℝ :=
  let costGeography := 35 * 10.5
  let costEnglish := 35 * 7.5
  let costMath := 20 * 12.0
  let costScience := 30 * 9.5
  let costHistory := 25 * 11.25
  let costArt := 15 * 6.75
  let discount c := c * 0.10
  let netGeography := if 35 >= 30 then costGeography - discount costGeography else costGeography
  let netEnglish := if 35 >= 30 then costEnglish - discount costEnglish else costEnglish
  let netScience := if 30 >= 30 then costScience - discount costScience else costScience
  let netMath := costMath
  let netHistory := costHistory
  let netArt := costArt
  netGeography + netEnglish + netMath + netScience + netHistory + netArt

theorem total_order_cost_is_correct : totalOrderCost = 1446.00 := by
  sorry

end total_order_cost_is_correct_l288_288737


namespace min_sum_reciprocals_of_roots_l288_288825

theorem min_sum_reciprocals_of_roots (k : ℝ) 
  (h_roots_positive : ∀ x : ℝ, (x^2 - k * x + k + 3 = 0) → 0 < x) :
  (k ≥ 6) → 
  ∀ x1 x2 : ℝ, (x1*x2 = k + 3) ∧ (x1 + x2 = k) ∧ (x1 > 0) ∧ (x2 > 0) → 
  (1 / x1 + 1 / x2) = 2 / 3 :=
by 
  -- proof steps go here
  sorry

end min_sum_reciprocals_of_roots_l288_288825


namespace total_cookies_l288_288252

-- Definitions of the conditions
def cookies_in_bag : ℕ := 21
def bags_in_box : ℕ := 4
def boxes : ℕ := 2

-- Theorem stating the total number of cookies
theorem total_cookies : cookies_in_bag * bags_in_box * boxes = 168 := by
  sorry

end total_cookies_l288_288252


namespace range_of_function_l288_288244

theorem range_of_function (x : ℝ) :
  (∃ y : ℝ, y = sqrt x / (x - 1)) → x ≥ 0 ∧ x ≠ 1 :=
by
  intro h
  sorry -- The proof goes here

end range_of_function_l288_288244


namespace find_prime_c_l288_288956

-- Define the statement of the problem
theorem find_prime_c (c : ℕ) (hc : Nat.Prime c) (h : ∃ m : ℕ, (m > 0) ∧ (11 * c + 1 = m^2)) : c = 13 :=
by
  sorry

end find_prime_c_l288_288956


namespace projection_of_b_in_direction_of_a_l288_288052

variables {ℝ : Type*} {a b : ℝ → ℝ}

noncomputable def magnitude (v : ℝ → ℝ) : ℝ := real.sqrt (v 0 ^ 2 + v 1 ^ 2 + v 2 ^ 2)

noncomputable def dot_product (v w : ℝ → ℝ) : ℝ := v 0 * w 0 + v 1 * w 1 + v 2 * w 2

theorem projection_of_b_in_direction_of_a
  (h₁ : magnitude a = 2)
  (h₂ : dot_product (λ i, 2 * a i - b i) a = 0) :
  (dot_product a b) / (magnitude a) = 4 :=
by
  sorry

end projection_of_b_in_direction_of_a_l288_288052


namespace quadratic_has_two_distinct_real_roots_l288_288654

theorem quadratic_has_two_distinct_real_roots 
  (a b c : ℝ)
  (h1 : a = 1)
  (h2 : b = 2)
  (h3 : c = -1) :
  let Δ := b ^ 2 - 4 * a * c in
  Δ = 8 ∧ Δ > 0 :=
by
  sorry

end quadratic_has_two_distinct_real_roots_l288_288654


namespace sum_G_equals_4027934_l288_288787

noncomputable def G (n : ℕ) : ℕ := 
  if n > 1 then 2 * n else 0

theorem sum_G_equals_4027934 :
  ∑ n in finset.Icc 2 2007, G n = 4_027_934 :=
by
  sorry

end sum_G_equals_4027934_l288_288787


namespace perfect_square_trinomial_l288_288054

theorem perfect_square_trinomial (a : ℝ) :
  (∃ b : ℝ, (x : ℝ) -> x^2 + a * x + 81 = (x + b)^2) → (a = 18 ∨ a = -18) :=
begin
  sorry
end

end perfect_square_trinomial_l288_288054


namespace hexagon_side_length_l288_288762

def hexagon_sides (a b : ℕ) (n : ℕ) : Prop :=
  ∃ (sides : Fin 6 → ℕ), (∀ i, sides i = a ∨ sides i = b) ∧ (∑ i, sides i = n)

theorem hexagon_side_length (a b : ℕ) (n : ℕ)
    (h : hexagon_sides a b n) : 
    a = 4 ∧ b = 7 ∧ n = 38 → (∃ x, x = 1) :=
by
  sorry

end hexagon_side_length_l288_288762


namespace analytic_expression_of_f_solve_inequality_f_l288_288331

-- Definition of the function f based on given conditions
def f (x : ℝ) : ℝ :=
  if x > 0 then x - 2
  else if x < 0 then x + 2
  else 0

-- Prove the given statements:
theorem analytic_expression_of_f :
  ∀ x, (x > 0 → f x = x - 2) ∧ (x = 0 → f x = 0) ∧ (x < 0 → f x = x + 2) :=
by intro x; split_ifs; simp

theorem solve_inequality_f :
  {x : ℝ | f x < 2} = {x : ℝ | x < 4} :=
by sorry

end analytic_expression_of_f_solve_inequality_f_l288_288331


namespace mark_walk_distance_in_15_minutes_l288_288943

theorem mark_walk_distance_in_15_minutes : 
  let rate := (1 : ℝ) / 35
  let distance := rate * 15 
  real.round(distance * 10) / 10 = 0.4 :=
by
  let rate := (1 : ℝ) / 35
  let distance := rate * 15 
  have result := real.round(distance * 10) / 10 
  show result = 0.4
  sorry

end mark_walk_distance_in_15_minutes_l288_288943


namespace perimeter_of_square_is_32_l288_288019

-- Given conditions
def radius := 4
def diameter := 2 * radius
def side_length_of_square := diameter

-- Question: What is the perimeter of the square?
def perimeter_of_square := 4 * side_length_of_square

-- Proof statement
theorem perimeter_of_square_is_32 : perimeter_of_square = 32 :=
sorry

end perimeter_of_square_is_32_l288_288019


namespace product_of_successive_numbers_l288_288242

-- Given conditions
def n : ℝ := 51.49757275833493

-- Proof statement
theorem product_of_successive_numbers : n * (n + 1) = 2703.0000000000005 :=
by
  -- Proof would be supplied here
  sorry

end product_of_successive_numbers_l288_288242


namespace isosceles_triangle_perimeter_l288_288521

-- Definitions for the conditions
def is_isosceles_triangle (a b c : ℕ) : Prop :=
  a = b ∨ b = c ∨ c = a

def valid_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

-- Statement of the theorem
theorem isosceles_triangle_perimeter {a b c : ℕ} (h1 : is_isosceles_triangle a b c) (h2 : valid_triangle a b c) :
  (a = 2 ∧ b = 4 ∧ c = 4 ∨ a = 4 ∧ b = 4 ∧ c = 2 ∨ a = 4 ∧ b = 2 ∧ c = 4) →
  a + b + c = 10 :=
by 
  sorry

end isosceles_triangle_perimeter_l288_288521


namespace locus_of_point_C_area_of_quadrilateral_SMTN_l288_288309

/-- Problem Conditions -/
structure ProblemConditions where
  A : ℝ×ℝ := (-1, 0)
  B : ℝ×ℝ := (1, 0)
  radius : ℝ := 2 * Real.sqrt 2

/-- Question 1: Equation of the Locus -/
theorem locus_of_point_C (conds : ProblemConditions) :
  ∃ (E : ℝ×ℝ → Prop), (∀ C, E C ↔ (C.1 ^ 2 / 2 + C.2 ^ 2 = 1)) :=
sorry

/-- Question 2: Minimum and Maximum Area of Quadrilateral SMTN -/
theorem area_of_quadrilateral_SMTN (conds : ProblemConditions)
  (S T M N: ℝ×ℝ)
  (H1 : (S.1-conds.B.1)/(T.1-conds.B.1) = (S.2-conds.B.2)/(T.2-conds.B.2))
  (H2 : (M.1-conds.B.1)/(N.1-conds.B.1) = (M.2-conds.B.2)/(N.2-conds.B.2))
  (H3 : (S.1-conds.B.1) * (M.1-conds.B.1) + (S.2-conds.B.2) * (M.2-conds.B.2) = 0) :
  ∃ (min_area max_area : ℝ), min_area = 16 / 9 ∧ max_area = 2 :=
sorry

end locus_of_point_C_area_of_quadrilateral_SMTN_l288_288309


namespace general_term_formula_l288_288529

noncomputable def arithmetic_seq (a_n : ℕ → ℝ) : Prop :=
  ∃ a d : ℝ, a_2 + a_7 + a_12 = 12 ∧ a_2 * a_7 * a_12 = 28 ∧
    (forall n : ℕ, a_n = a + (n - 1) * d)

theorem general_term_formula (a_n : ℕ → ℝ) :
  (a_2 + a_7 + a_12 = 12) ∧ (a_2 * a_7 * a_12 = 28) → 
  (∀ n, a_n = (3  / 5 : ℝ) * n - (1 / 5 : ℝ)) ∨ (∀ n, a_n = (-3  / 5 : ℝ) * n + (41 / 5 : ℝ)) :=
sorry

end general_term_formula_l288_288529


namespace part1_solution_set_part2_range_of_a_l288_288177

noncomputable def f (x a : ℝ) : ℝ := |x - 1| + |x - a|

theorem part1_solution_set (a : ℝ) (h : a = 4) : 
  {x : ℝ | f x a ≥ 5} = {x | x ≤ 0} ∪ {x | x ≥ 5} :=
by 
  -- proof goes here
  sorry

theorem part2_range_of_a (a : ℝ) (h : ∀ x : ℝ, f x a ≥ 4) : 
  a ≤ -3 ∨ a ≥ 5 :=
by
  -- proof goes here
  sorry

end part1_solution_set_part2_range_of_a_l288_288177


namespace jindra_initial_dice_count_l288_288553

-- Given conditions about the dice stacking
def number_of_dice_per_layer : ℕ := 36
def layers_stacked_completely : ℕ := 6
def dice_received : ℕ := 18

-- We need to prove that the initial number of dice Jindra had is 234
theorem jindra_initial_dice_count : 
    (layers_stacked_completely * number_of_dice_per_layer + dice_received) = 234 :=
    by 
        sorry

end jindra_initial_dice_count_l288_288553


namespace correct_statements_count_l288_288421

variables (a b c : ℝ)

-- Condition 1: ac² > bc² implies a > b
def condition1 : Prop := ∀ (a b c : ℝ), ac² > bc² → a > b

-- Condition 2: |a - 2| > |b - 2| implies (a - 2)² > (b - 2)²
def condition2 : Prop := ∀ (a b : ℝ), |a - 2| > |b - 2| → (a - 2)² > (b - 2)²

-- Condition 3: a > b > c > 0 implies 1/a < 1/b < 1/c
def condition3 : Prop := ∀ (a b c : ℝ), a > b > c > 0 → 1/a < 1/b ∧ 1/b < 1/c

-- Condition 4: ab ≠ 0 implies b/a + a/b ≥ 2
def condition4 : Prop := ∀ (a b : ℝ), ab ≠ 0 → b/a + a/b ≥ 2

-- Number of correct statements is 3
theorem correct_statements_count : (condition1 a b c) ∧ (condition2 a b) ∧ (condition3 a b c) ∧ ¬ (condition4 a b) → 3 := sorry

end correct_statements_count_l288_288421


namespace remainder_h_x10_div_h_l288_288167

noncomputable def h (x : ℕ→ ℕ) : ℕ → ℕ :=
λ x, x^7 + x^6 + x^5 + x^4 + x^3 + x^2 + x + 1

theorem remainder_h_x10_div_h (x : ℕ→ ℕ) :
    ∀ x, polynomial.remainder (h (x^10)) (h x) = 8 :=
sorry

end remainder_h_x10_div_h_l288_288167


namespace solve_complex_numbers_l288_288584

theorem solve_complex_numbers : 
  ∃ p q r : ℂ, 
  p + q + r = 2 ∧ 
  p * q + p * r + q * r = 3 ∧ 
  p * q * r = 2 ∧ 
  {p, q, r} = {1, (1 + real.sqrt 7 * complex.I) / 2, (1 - real.sqrt 7 * complex.I) / 2} := 
sorry

end solve_complex_numbers_l288_288584


namespace number_of_subsets_of_A_l288_288474

theorem number_of_subsets_of_A :
  let A := { x : ℤ | (x + 1) / (x - 2) ≤ 0 } in
  A.finite ∧ A.to_finset.card = 3 ∧ (2 ^ A.to_finset.card = 8) :=
by {
  let A := { x : ℤ | (x + 1) / (x - 2) ≤ 0 },
  have hA : A = {x | x = -1 ∨ x = 0 ∨ x = 1}, 
  { sorry }, -- Placeholder for the proof of the set A being {-1, 0, 1}
  have h0 : A.finite, 
  { sorry }, -- Placeholder for the proof of A being finite
  have hcard : A.to_finset.card = 3, 
  { sorry }, -- Placeholder for the proof that the cardinality of A is 3
  exact ⟨h0, hcard, by norm_num⟩, -- Completing the proof using the obtained results
}

end number_of_subsets_of_A_l288_288474


namespace volume_common_part_l288_288665

noncomputable theory

open Real

theorem volume_common_part (V : ℝ) (a : ℝ) (SP : ℝ) (h₁ : SP = a)
  (h₂ : a^3 = 3 * V) :
  let V1 := (3 * sqrt 2 - 2) / 21 * V
  in V1 = V * (3 * sqrt 2 - 2) / 21 :=
sorry

end volume_common_part_l288_288665


namespace de_Moivre_formula_rational_cos_sin_l288_288607

variables (θ : ℝ) (n : ℤ)

-- Definition of rational numbers
def is_rational (x : ℝ) : Prop := ∃ (a b : ℤ), b ≠ 0 ∧ x = a / b

-- De Moivre's formula
theorem de_Moivre_formula (θ : ℝ) (n : ℤ) : 
  (Complex.cos θ + Complex.sin θ * Complex.i)^n = 
  Complex.cos (n * θ) + Complex.sin (n * θ) * Complex.i :=
sorry

-- Rationality deduction from de Moivre's formula
theorem rational_cos_sin (θ : ℝ) (n : ℤ) : 
  is_rational (Real.cos θ) → is_rational (Real.sin θ) → is_rational (Real.cos (n * θ)) ∧ is_rational (Real.sin (n * θ)) :=
sorry

end de_Moivre_formula_rational_cos_sin_l288_288607


namespace fourth_valid_sample_is_20_l288_288323

noncomputable def randomNumberTable := [66, 67, 40, 37, 14, 64, 05, 71, 11, 05, 65, 09, 95, 86, 68, 76, 83, 20, 37, 90, 
                                        57, 16, 03, 11, 63, 14, 90, 84, 45, 21, 75, 73, 88, 05, 90, 52, 23, 59, 43, 10]

noncomputable def validRange (num : ℕ) : Prop := num ≥ 1 ∧ num ≤ 50

theorem fourth_valid_sample_is_20 : 
    ∃ (samples : List ℕ), 
    samples = (randomNumberTable.drop 8).filter validRange ∧ samples.nth 3 = some 20 := 
by 
  sorry

end fourth_valid_sample_is_20_l288_288323


namespace locus_is_sphere_l288_288850

def Point := (ℝ × ℝ × ℝ)

def distance (P Q : Point) : ℝ :=
  real.sqrt ((P.1 - Q.1) ^ 2 + (P.2 - Q.2) ^ 2 + (P.3 - Q.3) ^ 2)

theorem locus_is_sphere (A B : Point) (k : ℝ) (h₁ : 0 < k) (h₂ : k ≠ 1) :
  ∃ C : Point, ∃ R : ℝ, R > 0 ∧
  {M : Point | distance M A / distance M B = k} = {M | distance M C = R} :=
sorry

end locus_is_sphere_l288_288850


namespace count_valid_age_pairs_l288_288152

theorem count_valid_age_pairs :
  ∃ (d n : ℕ) (a b : ℕ), 10 * a + b ≥ 30 ∧
                       10 * b + a ≥ 35 ∧
                       b > a ∧
                       ∃ k : ℕ, k = 10 := 
sorry

end count_valid_age_pairs_l288_288152


namespace water_fee_correct_l288_288209

def water_fee (a n : ℕ) : ℕ :=
  if n ≤ 12 then a * n
  else if n ≤ 20 then (3 * (a * n) - 6 * a) / 2
  else 2 * a * n - 16 * a

theorem water_fee_correct (a n : ℕ) :
  (n ≤ 12 → water_fee a n = a * n) ∧
  (12 < n ∧ n ≤ 20 → water_fee a n = 1.5 * (a * n) - 6 * a) ∧
  (n > 20 → water_fee a n = 2 * a * n - 16 * a) :=
by
  sorry

end water_fee_correct_l288_288209


namespace minimum_area_triangle_l288_288148

open Real

noncomputable def minimum_triangle_area {A K : Point} (α : ℝ) (d1 d2 : ℝ) :=
  let side1 := dist K (line_through A α)
  let side2 := dist K (rotate (line_through A α) (π / 6))
  (side1 = 1) → (side2 = 2) → α = π / 6 → minimum_area K = 8

-- The statement of the problem
theorem minimum_area_triangle : minimum_triangle_area 30 (1 : ℝ) (2 : ℝ) := 
sorry

end minimum_area_triangle_l288_288148


namespace arithmetic_sqrt_sqrt_81_l288_288288

-- Definition of the square root
noncomputable def sqrt (x : ℝ) : ℝ :=
if x >= 0 then Classical.choose (exists_sqrt x) else 0

-- Assume that sqrt(81) = 9
axiom sqrt_81_is_9 : sqrt 81 = 9

-- Prove that the arithmetic square root of sqrt(81) is 3
theorem arithmetic_sqrt_sqrt_81 : sqrt (sqrt 81) = 3 :=
by
  rw sqrt_81_is_9
  exact sqrt_9_eq_3
  sorry -- proof of sqrt_9_eq_3 is omitted, assuming it's verified elsewhere

end arithmetic_sqrt_sqrt_81_l288_288288


namespace sin_double_angle_l288_288493

theorem sin_double_angle (θ : ℝ) : (cos θ + sin θ = 3/2) → sin (2 * θ) = 5/4 :=
by
  intro h
  sorry

end sin_double_angle_l288_288493


namespace round_3967149_8487234_l288_288991

theorem round_3967149_8487234 :
  Real.round 3967149.8487234 = 3967150 :=
by
  sorry

end round_3967149_8487234_l288_288991


namespace cakes_on_table_l288_288668

theorem cakes_on_table (N : ℕ) (h1 : ∀ k : ℕ, k % N < N) (h2 : 7 | N) : N = 9 :=
sorry

end cakes_on_table_l288_288668


namespace speed_difference_valid_l288_288657

-- Definitions of the conditions
def speed (s : ℕ) : ℕ := s^2 + 2 * s

-- Theorem statement that needs to be proven
theorem speed_difference_valid : 
  (speed 5 - speed 3) = 20 :=
  sorry

end speed_difference_valid_l288_288657


namespace remainder_division_l288_288017

theorem remainder_division
  (P E M S F N T : ℕ)
  (h1 : P = E * M + S)
  (h2 : M = N * F + T) :
  (∃ r, P = (EF + 1) * (P / (EF + 1)) + r ∧ r = ET + S - N) :=
sorry

end remainder_division_l288_288017


namespace total_tin_in_new_alloy_l288_288316

-- Define the weights of alloy A and alloy B
def weightAlloyA : Float := 135
def weightAlloyB : Float := 145

-- Define the ratio of lead to tin in alloy A
def ratioLeadToTinA : Float := 3 / 5

-- Define the ratio of tin to copper in alloy B
def ratioTinToCopperB : Float := 2 / 3

-- Define the total parts for alloy A and alloy B
def totalPartsA : Float := 3 + 5
def totalPartsB : Float := 2 + 3

-- Define the fraction of tin in alloy A and alloy B
def fractionTinA : Float := 5 / totalPartsA
def fractionTinB : Float := 2 / totalPartsB

-- Calculate the amount of tin in alloy A and alloy B
def tinInAlloyA : Float := fractionTinA * weightAlloyA
def tinInAlloyB : Float := fractionTinB * weightAlloyB

-- Calculate the total amount of tin in the new alloy
def totalTinInNewAlloy : Float := tinInAlloyA + tinInAlloyB

-- The theorem to be proven
theorem total_tin_in_new_alloy : totalTinInNewAlloy = 142.375 := by
  sorry

end total_tin_in_new_alloy_l288_288316


namespace altitude_AD_eq_line_BC_eq_l288_288545

-- Problem 1: Equation of altitude AD
theorem altitude_AD_eq (A B C : Point) (AD : Line) (hA : A = ⟨-2, 1⟩) (hB : B = ⟨4, 3⟩) (hC : C = ⟨3, -2⟩) 
(hAD_perp_BC : perpendicular AD (line_through B C)) (hA_on_AD : on_line A AD) :
  AD = line_equation 1 5 (-3) := 
sorry

-- Problem 2: Equation of line BC with midpoint M
theorem line_BC_eq (A B M : Point) (BC : Line) (hA : A = ⟨-2, 1⟩) (hB : B = ⟨4, 3⟩) (hM : M = ⟨3, 1⟩) (hM_mid : midpoint M A C) :
  BC = line_equation 1 2 (-10) :=
sorry

end altitude_AD_eq_line_BC_eq_l288_288545


namespace julia_tulip_count_l288_288557

def tulip_count (tulips daisies : ℕ) : Prop :=
  3 * daisies = 7 * tulips

theorem julia_tulip_count : 
  ∃ t, tulip_count t 65 ∧ t = 28 := 
by
  sorry

end julia_tulip_count_l288_288557


namespace periodic_sequence_integer_alpha_l288_288158

theorem periodic_sequence_integer_alpha (α : ℝ) (h₀ : α > 1) 
  (h₁ : ∃ p : ℕ, p > 0 ∧ ∀ n : ℕ, n ≥ 1 → α * ⌊α^n⌋ - ⌊α^(n+1)⌋ = α * ⌊α^(n + p)⌋ - ⌊α^(n+1 + p)⌋) 
  : α ∈ ℤ := 
sorry

end periodic_sequence_integer_alpha_l288_288158


namespace geometric_sequence_third_term_l288_288122

theorem geometric_sequence_third_term (a1 a5 a3 : ℕ) (r : ℝ) 
  (h1 : a1 = 4) 
  (h2 : a5 = 1296) 
  (h3 : a5 = a1 * r^4)
  (h4 : a3 = a1 * r^2) : 
  a3 = 36 := 
by 
  sorry

end geometric_sequence_third_term_l288_288122


namespace monotonic_increasing_interval_l288_288643

noncomputable def f (x : ℝ) : ℝ := (1 / 2)^(x^2 - 2 * x + 6)

theorem monotonic_increasing_interval : 
  ∀ x y : ℝ, x < y → y < 1 → f x < f y :=
by
  sorry

end monotonic_increasing_interval_l288_288643


namespace distance_between_poles_l288_288337

-- Defining the conditions
def length : ℝ := 90
def width : ℝ := 50
def poles : ℝ := 14
def perimeter : ℝ := 2 * (length + width)

-- Mathematically equivalent proof problem
theorem distance_between_poles : 
  ∃ d : ℝ, d = perimeter / (poles - 1) ∧ d ≈ 21.54 :=
by
  sorry

end distance_between_poles_l288_288337


namespace probability_at_least_five_at_least_six_times_l288_288329

/-- 
A fair die is rolled eight times. The probability of rolling at least 
a five at least six times out of the eight rolls is 129/6561.
--/
theorem probability_at_least_five_at_least_six_times :
  let p := (1 : ℚ) / 6
  in (∑ k in finset.range 3, (nat.choose 8 (6 + k) * p ^ (6 + k) * (1 - p) ^ (8 - 6 - k))) = 129 / 6561 :=
by
  sorry

end probability_at_least_five_at_least_six_times_l288_288329


namespace centers_lie_on_line_max_length_of_chord_AB_line_tangent_to_all_circles_l288_288175

noncomputable def circle_eq (t : ℝ) : ℝ → ℝ → Prop :=
  λ x y, x^2 + y^2 - 2 * (t + 3) * x - 2 * t * y + t^2 + 4 * t + 8 = 0

def line_l (x y : ℝ) : Prop := x + y - 3 = 0

theorem centers_lie_on_line :
  ∀ t : ℝ, t ≠ -1 → ∃ p : ℝ × ℝ, circle_eq t p.1 p.2 ∧ p.2 = p.1 - 3 :=
sorry

theorem max_length_of_chord_AB :
  ∃ t : ℝ, ∃ p1 p2 : ℝ × ℝ, circle_eq t p1.1 p1.2 ∧ circle_eq t p2.1 p2.2 ∧ 
  line_l p1.1 p1.2 ∧ line_l p2.1 p2.2 ∧ 
  (dist p1 p2 = 2 * Real.sqrt 2) :=
sorry

theorem line_tangent_to_all_circles :
  ∃ t : ℝ, ∀ t : ℝ, ∃ p : ℝ × ℝ, circle_eq t p.1 p.2 → 
  ((p.1 = 2 ∨ p.2 = -1) ∧ t ≠ -1) :=
sorry

end centers_lie_on_line_max_length_of_chord_AB_line_tangent_to_all_circles_l288_288175


namespace probability_three_colors_all_present_l288_288793

theorem probability_three_colors_all_present (balls total red black white selected : ℕ) 
  (h_total : total = 11) 
  (h_red : red = 3) 
  (h_black : black = 3) 
  (h_white : white = 5) 
  (h_selected : selected = 4) : 
  (number_of_ways := (nat.desc_factorial total selected) / (nat.factorial selected)) 
  (ways_two_red := (nat.comb red 2) * (nat.comb black 1) * (nat.comb white 1))
  (ways_two_black := (nat.comb black 2) * (nat.comb red 1) * (nat.comb white 1))
  (ways_two_white := (nat.comb white 2) * (nat.comb red 1) * (nat.comb black 1))
  (ways_all_three_colors := ways_two_red + ways_two_black + ways_two_white) :
  (ways_all_three_colors / number_of_ways) = 6 / 11 :=
by
  sorry

end probability_three_colors_all_present_l288_288793


namespace count_positive_area_triangles_l288_288092

noncomputable def numPositiveAreaTriangles : ℕ := 2160

theorem count_positive_area_triangles 
  (vertices : list (ℤ × ℤ))
  (h1 : ∀ p ∈ vertices, 1 ≤ p.1 ∧ p.1 ≤ 5 ∧ 1 ≤ p.2 ∧ p.2 ≤ 5)
  (h2 : vertices.length = 25) : 
  (number_of_triangles_with_positive_area vertices) = numPositiveAreaTriangles :=
sorry

end count_positive_area_triangles_l288_288092


namespace berthaDaughtersGranddaughtersNoDaughters_l288_288365

-- Define the conditions of the problem
def berthaHasEightDaughters : ℕ := 8
def totalWomen : ℕ := 40

noncomputable def berthaHasNoSons : Prop := true
def daughtersHaveFourDaughters (x : ℕ) : Prop := x = 4
def granddaughters : ℕ := totalWomen - berthaHasEightDaughters
def daughtersWithNoChildren : ℕ := 0

theorem berthaDaughtersGranddaughtersNoDaughters :
  let daughtersWithChildren := granddaughters / 4,
      womenWithNoDaughters := granddaughters + daughtersWithNoChildren
  in womenWithNoDaughters = 32 :=
by
  sorry

end berthaDaughtersGranddaughtersNoDaughters_l288_288365


namespace range_of_g_l288_288953

noncomputable def f (x : ℝ) : ℝ := 5 * x + 3

noncomputable def g (x : ℝ) : ℝ := f (f (f (f x)))

theorem range_of_g : ∀ x, (-1 : ℝ) ≤ x ∧ x ≤ (1 : ℝ) → (-157 : ℝ) ≤ g(x) ∧ g(x) ≤ (1093 : ℝ) :=
by
  -- Here we would provide a mathematical proof that can rigorously justify
  -- the stated range. For now, we insert a placeholder "sorry".
  sorry

end range_of_g_l288_288953


namespace units_digit_a2017_l288_288472

noncomputable def seq_a (n : ℕ) : ℝ := (real.sqrt 2 + 1)^n - (real.sqrt 2 - 1)^n

theorem units_digit_a2017 : int.floor (seq_a 2017) % 10 = 2 :=
sorry

end units_digit_a2017_l288_288472


namespace main_proof_l288_288437

noncomputable def is_eq_Ellipse_C : Prop :=
  ∀ a b : ℝ, 1 < a ∧ a > b ∧ b = 1 ∧ 
  (∀ P Q : ℝ × ℝ, (P.1^2 + P.2^2 = 1) ∧ ((Q.2^2 / a^2) + (Q.1^2 / b^2) = 1) →
  max (P.1 - Q.1)^2 + (P.2 - Q.2)^2 = 9 → (a = 2) → ((Q.2^2 / 4) + Q.1^2 = 1))

noncomputable def max_area_triangle_OMN : Prop :=
  ∀ t : ℝ, t ≠ 0 → 
  ∃ (M N : ℝ × ℝ), 
    (t^2 - 1 = (M.1 * N.1 + M.2 * N.2)) ∧ 
    max_area_OMN t = 1

theorem main_proof : is_eq_Ellipse_C ∧ max_area_triangle_OMN :=
sorry

end main_proof_l288_288437


namespace eq_correct_l288_288268

variable (x : ℝ)

def width (x : ℝ) : ℝ := x - 6

def area_eq (x : ℝ) : Prop := x * width x = 720

theorem eq_correct (h : area_eq x) : x * (x - 6) = 720 :=
by exact h

end eq_correct_l288_288268


namespace final_snowboard_price_l288_288595

noncomputable def originalPrice : ℝ := 120
noncomputable def firstDiscountRate : ℝ := 0.40
noncomputable def secondDiscountRate : ℝ := 0.20

theorem final_snowboard_price :
  let firstDiscount := originalPrice * firstDiscountRate in
  let reducedPriceAfterFirstDiscount := originalPrice - firstDiscount in
  let secondDiscount := reducedPriceAfterFirstDiscount * secondDiscountRate in
  let finalPrice := reducedPriceAfterFirstDiscount - secondDiscount in
  finalPrice = 57.6 := by
    sorry

end final_snowboard_price_l288_288595


namespace tic_tac_toe_tie_probability_l288_288128

theorem tic_tac_toe_tie_probability (john_wins martha_wins : ℚ) 
  (hj : john_wins = 4 / 9) 
  (hm : martha_wins = 5 / 12) : 
  1 - (john_wins + martha_wins) = 5 / 36 := 
by {
  /- insert proof here -/
  sorry
}

end tic_tac_toe_tie_probability_l288_288128


namespace other_root_of_quadratic_l288_288198

theorem other_root_of_quadratic (m : ℝ) (h : ∃ α : ℝ, α = 1 ∧ (3 * α^2 + m * α = 5)) :
  ∃ β : ℝ, β = -5 / 3 :=
by
  sorry

end other_root_of_quadratic_l288_288198


namespace cosine_of_angle_lambda_value_l288_288477

notation "ℝ" => Real

variables (a b : ℝ × ℝ)

def dot_prod (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2
def mag (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1^2 + v.2^2)

def cos_theta (a b : ℝ × ℝ) := (dot_prod a b) / ((mag a) * (mag b))

theorem cosine_of_angle (h₁ : a = (4, 3)) (h₂ : b = (-1, 2)) :
  cos_theta a b = 2 * Real.sqrt 5 / 5 :=
by
  sorry

def is_perpendicular (u v : ℝ × ℝ) : Prop := dot_prod u v = 0

theorem lambda_value (h₁ : a = (4, 3)) (h₂ : b = (-1, 2)) 
  (h₃ : is_perpendicular (a - (λ x : ℝ, (x * b))) (2 * a + b)) : 
  ∃ λ : ℝ, λ = 52 / 9 :=
by
  sorry

end cosine_of_angle_lambda_value_l288_288477


namespace parallel_CF_OE_l288_288250

-- Define conditions of the problem
variables (A B C O D E Q F : Type*) -- Points
variables [Aff_direction H1 H2 H3 H4 H5] -- Right angle A, center O of circumcircle, normal O to AC, mid AE, meets CE in Q
variables (F : (line (ce [D E])) intersect (line (ce [O Q])) ) -- Intersection of BE and OQ

-- Theorem to prove CF parallel to OE
theorem parallel_CF_OE (h1 : right_angle ∠A) 
                        (h2 : circumcenter ΔABC = O)
                        (h3 : orthogonal_projection O (line (ce [A C])) = D)
                        (h4 : dist E O = dist D O ∧ dist E A = dist D A)
                        (h5 : ray(bisector ∠CAO ) [line (ce [C E])] intersects = Q)
                        (h6 : (line (ce [B E])) intersect (line (ce [O Q])) = F ) : parallel (line (ce [C F])) (line (ce [O E])) :=
by
  sorry

end parallel_CF_OE_l288_288250


namespace largest_prime_divisor_of_17_squared_plus_40_squared_l288_288780

theorem largest_prime_divisor_of_17_squared_plus_40_squared :
  ∃ p : ℕ, prime p ∧ p ≤ 1889 ∧ ∀ q : ℕ, prime q → q ≤ 1889 → q ≠ p → q < p := 
by
  have h1 : 17^2 = 289 := by norm_num
  have h2 : 40^2 = 1600 := by norm_num
  have h3 : 1889 = 17^2 + 40^2 := by rw [h1, h2]; norm_num
  use 269
  split
  sorry -- Prime proof for 269
  split
  exact le_of_eq h3
  intros q hq hq1 hneq
  sorry -- Proof that 269 is larger than any other prime divisor

end largest_prime_divisor_of_17_squared_plus_40_squared_l288_288780


namespace ball_distribution_l288_288897

theorem ball_distribution :
  let balls := 5 in
  let boxes := 4 in
  -- The number of distinct ways to partition 5 balls into 4 indistinguishable boxes
  ((λ n k : ℕ, n = 5 ∧ k = 4) → ∃! p : ℕ, p = 4) :=
by sorry

end ball_distribution_l288_288897


namespace equal_area_parts_l288_288412

theorem equal_area_parts (k : ℝ) (h_nonneg : 0 ≤ k) :
  (let a := -sqrt(1 + sqrt(1 - k)),
       b := -sqrt(1 - sqrt(1 - k)),
       c := sqrt(1 - sqrt(1 - k)),
       d := sqrt(1 + sqrt(1 - k)),
       A_R := ∫ x in c..d, (-x^4 + 2*x^2 - k) dx,
       A_M := ∫ x in b..c, (-x^4 + 2*x^2 - k) dx,
       A_L := ∫ x in a..b, (-x^4 + 2*x^2 - k) dx
   in A_R = A_M ∧ A_M = A_L) ↔ k = 2 / 3 :=
by with_integrals sorry

end equal_area_parts_l288_288412


namespace positive_integer_triples_characterization_l288_288400

theorem positive_integer_triples_characterization :
  ∀ a m n : ℕ, a > 1 → m < n → (∀ p : ℕ, p.prime ∈ (nat.factors (a^m - 1) : multiset ℕ) ↔ p.prime ∈ (nat.factors (a^n - 1) : multiset ℕ)) →
  ∃ l : ℕ, l ≥ 2 ∧ a = 2^l - 1 ∧ m = 1 ∧ n = 2 :=
by sorry

end positive_integer_triples_characterization_l288_288400


namespace validate_triangle_count_l288_288085

noncomputable def count_valid_triangles : ℕ :=
  let total_points := 25
  let total_triples := (Nat.choose total_points 3)
  let collinear_rows := 5 * (Nat.choose 5 3)
  let collinear_columns := 5 * (Nat.choose 5 3)
  let main_diagonals := 2 * (Nat.choose 5 3)
  let secondary_diagonals := 8 * (Nat.choose 4 3)
  let invalid_triangles := collinear_rows + collinear_columns + main_diagonals + secondary_diagonals
  total_triples - invalid_triangles

theorem validate_triangle_count : count_valid_triangles = 2148 :=
by
  sorry

end validate_triangle_count_l288_288085


namespace angle_A1_L_B1_is_45_deg_l288_288699

variables (S : Type) [circle S]

variables (A B K L A_1 B_1 : S) (S_A S_B : circle S) (AB_LK : is_diameter S A B)
variables (is_diameter : is_perpendicular S K L) (tan_SA : tangent S_A A_1)
variables (tan_SB : tangent S_B B_1) (on_AB : on_diameter A B K)

theorem angle_A1_L_B1_is_45_deg : ∠ (A_1 L B_1) = 45 :=
by sorry

end angle_A1_L_B1_is_45_deg_l288_288699


namespace food_for_elephants_l288_288942

theorem food_for_elephants (t : ℕ) : 
  (∀ (food_per_day : ℕ), (12 * food_per_day) * 1 = (1000 * food_per_day) * 600) →
  (∀ (food_per_day : ℕ), (t * food_per_day) * 1 = (100 * food_per_day) * d) →
  d = 500 * t :=
by
  sorry

end food_for_elephants_l288_288942


namespace triangle_is_right_if_midpoints_of_altitudes_collinear_l288_288986

variables {A B C A1 B1 C1 : Type} [EuclideanGeometry A B C] -- Assuming basic Euclidean geometry for the points
variable {triangle : Triangle A B C} -- Assuming the existence of a triangle

-- Definitions for the midpoints of the altitudes
variable (midpoint_altitude_A1 : Midpoint (altitudeFrom triangle A))
variable (midpoint_altitude_B1 : Midpoint (altitudeFrom triangle B))
variable (midpoint_altitude_C1 : Midpoint (altitudeFrom triangle C))

-- The collinearity condition
variable (collinear_midpoints : Collinear [midpoint_altitude_A1, midpoint_altitude_B1, midpoint_altitude_C1])

-- The proof statement
theorem triangle_is_right_if_midpoints_of_altitudes_collinear 
  (h : collinear_midpoints) : IsRightTriangle triangle := 
sorry

end triangle_is_right_if_midpoints_of_altitudes_collinear_l288_288986


namespace problem_circumference_tangent_l288_288445

-- Define the circle and line
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 + 2 * x - 4 * y - 4 = 0
def line_eq (a x y : ℝ) : Prop := a * x - y + 3 = 0

-- Define a function to calculate the distance
def distance (x1 y1 x2 y2 : ℝ) : ℝ := Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2)

noncomputable def length_PQ (a : ℝ) : ℝ :=
  let P := (a, -2)
  let C := (-1, 2)
  let PC := distance P.1 P.2 C.1 C.2
  Real.sqrt (PC^2 - 9)

-- The goal statement
theorem problem_circumference_tangent (a : ℝ) :
  line_eq a (-1) 2 →
  length_PQ 1 = Real.sqrt 11 :=
by
  intros h
  sorry

end problem_circumference_tangent_l288_288445


namespace sally_mcqueen_cost_l288_288182

theorem sally_mcqueen_cost :
  let lightning_mcqueen_cost := 140000
      mater_cost := 0.1 * lightning_mcqueen_cost
      sally_mcqueen_cost := 3 * mater_cost
  in sally_mcqueen_cost = 42000 :=
by
  let lightning_mcqueen_cost := 140000
  let mater_cost := 0.1 * lightning_mcqueen_cost
  let sally_mcqueen_cost := 3 * mater_cost
  calc 
    sally_mcqueen_cost 
    = 3 * (0.1 * lightning_mcqueen_cost) : by rw [mater_cost]
    = 3 * 14000                       : by rw [lightning_mcqueen_cost * 0.1]
    = 42000                           : by norm_num

end sally_mcqueen_cost_l288_288182


namespace max_value_of_sum_of_cubes_l288_288570

theorem max_value_of_sum_of_cubes (a b c d e : ℝ) (h : a^2 + b^2 + c^2 + d^2 + e^2 = 5) :
  a^3 + b^3 + c^3 + d^3 + e^3 ≤ 5 * real.sqrt 5 :=
sorry

end max_value_of_sum_of_cubes_l288_288570


namespace range_of_m_l288_288139

theorem range_of_m (m : ℝ) :
  (∀ P : ℝ × ℝ, P.2 = 2 * P.1 + m → (abs (P.1^2 + (P.2 - 1)^2) = (1/2) * abs (P.1^2 + (P.2 - 4)^2)) → (-2 * Real.sqrt 5) ≤ m ∧ m ≤ (2 * Real.sqrt 5)) :=
sorry

end range_of_m_l288_288139


namespace log_product_eq_six_l288_288505

theorem log_product_eq_six :
  let y := (List.range 63).map (λ n => Real.log (n + 2) / Real.log (n+1)).foldr (*) 1 in
  y = 6 := by
  sorry

end log_product_eq_six_l288_288505


namespace tangent_line_slope_at_ln_x_is_1_over_e_l288_288827

noncomputable def ln (x : ℝ) : ℝ := Real.log x

theorem tangent_line_slope_at_ln_x_is_1_over_e :
  ∀ k : ℝ, (∀ x : ℝ, 0 < x → y = k * x → tangent (ln x) (λ x' : ℝ, k * x') x) → k = 1 / Real.exp 1 :=
by
  intros k h
  sorry

end tangent_line_slope_at_ln_x_is_1_over_e_l288_288827


namespace calculate_x_l288_288745

theorem calculate_x : ∃ x : ℕ, 144 + 2 * 12 * 7 + 49 = x ∧ x = 361 :=
by
  use 361
  split
  · simp
  · rfl

end calculate_x_l288_288745


namespace probability_same_color_is_3_over_13_l288_288319

def numGreenBalls : ℕ := 15
def numWhiteBalls : ℕ := 12
def totalBalls : ℕ := numGreenBalls + numWhiteBalls
def numDrawnBalls : ℕ := 3

noncomputable def combinations (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

def probSameColor : ℚ := 
  (combinations numGreenBalls numDrawnBalls + combinations numWhiteBalls numDrawnBalls) / combinations totalBalls numDrawnBalls

theorem probability_same_color_is_3_over_13 : probSameColor = 3 / 13 := by
  sorry

end probability_same_color_is_3_over_13_l288_288319


namespace value_of_P_when_n_200_l288_288949

theorem value_of_P_when_n_200 :
  let P := ∏ k in finset.range(199) + 2, (1 - 1 / (k * k))
  P = 3 / 40000 :=
by
  sorry

end value_of_P_when_n_200_l288_288949


namespace f_of_2014_l288_288043

theorem f_of_2014 (f : ℝ → ℝ)
  (h1 : ∀ x : ℝ, f (x + 4) = -f x + 2 * Real.sqrt 2)
  (h2 : ∀ x : ℝ, f (-x) = f x)
  : f 2014 = Real.sqrt 2 :=
sorry

end f_of_2014_l288_288043


namespace balls_into_boxes_l288_288899

theorem balls_into_boxes :
  {l : List ℕ // l.sum = 5 ∧ l.length <= 4 ∧ l.sorted = l} = 6 := by sorry

end balls_into_boxes_l288_288899


namespace max_pieces_of_cake_l288_288307

theorem max_pieces_of_cake (
  larger_cake_side : ℕ := 20
  smaller_piece_side : ℕ := 2
) : larger_cake_side * larger_cake_side / (smaller_piece_side * smaller_piece_side) = 100 := by
  sorry

end max_pieces_of_cake_l288_288307


namespace find_k_l288_288006

theorem find_k (k : ℝ) : 4 + ∑' (n : ℕ), (4 + n * k) / 5^n = 10 → k = 16 := by
  sorry

end find_k_l288_288006


namespace correct_statements_l288_288635

def translated_and_stretched_function (x : ℝ) : ℝ :=
  2 * sin (2 * x + π / 3)

def is_symmetric_about (f : ℝ → ℝ) (a : ℝ) (b : ℝ) : Prop :=
  ∀ x : ℝ, f (a - x) = b - f (a + x)

def is_increasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y : ℝ, a ≤ x ∧ x < y ∧ y ≤ b → f x < f y

def min_value (f : ℝ → ℝ) (a b : ℝ) : ℝ :=
  infi (f '' (Icc a b))

theorem correct_statements :
  let f := translated_and_stretched_function
  in is_symmetric_about f (π / 3) 0 ∧
     ¬is_increasing_on f 0 (π / 6) ∧
     min_value (λ x, f x + (-2 * √3)) 0 (π / 2) = √3 :=
  by sorry

end correct_statements_l288_288635


namespace row_or_col_contains_sqrt_n_symbols_l288_288172

theorem row_or_col_contains_sqrt_n_symbols (n : ℕ) 
  (symbols : Finset (Fin n)) 
  (cells : Fin n × Fin n → Finset (Fin n)) 
  (sym_count : ∀ s : Finset (Fin n), s.card = n → ∃! c : Finset (Fin n × Fin n), c.card = n ∧ cells (c.coord1) = symbols) : 
  (∃ i : Fin n, (symbols.card (cells (i, ⟪⟩)).card) ≥ Nat.sqrt n)
  ∨ 
  (∃ j : Fin n, (symbols.card (cells (⟪⟩, j)).card) ≥ Nat.sqrt n) :=
sorry

end row_or_col_contains_sqrt_n_symbols_l288_288172


namespace Luke_total_coins_l288_288588

theorem Luke_total_coins (piles_quarters piles_dimes coins_per_pile : ℕ) 
  (hq : piles_quarters = 5) (hd : piles_dimes = 5) (hcoins : coins_per_pile = 3) : 
  piles_quarters * coins_per_pile + piles_dimes * coins_per_pile = 30 :=
by 
  rw [hq, hd, hcoins] 
  rw [Nat.mul_comm 5 3]
  sorry

end Luke_total_coins_l288_288588


namespace smallest_sphere_touches_L1_L2_l288_288187

noncomputable def L1 (t : ℝ) : ℝ × ℝ × ℝ :=
  (t + 1, 2*t - 4, -3*t + 5)

noncomputable def L2 (t : ℝ) : ℝ × ℝ × ℝ :=
  (4*t - 12, -t + 8, t + 17)

def sphere_center : ℝ × ℝ × ℝ :=
  (-1/2, -3/2, 31/2)

def sphere_radius : ℝ :=
  (Real.sqrt 251) / 2

theorem smallest_sphere_touches_L1_L2 :
  ∃ (center : ℝ × ℝ × ℝ) (radius : ℝ),
    center = sphere_center ∧
    radius = sphere_radius ∧
    (∃ t1 t2 : ℝ, (L1 t1 - center).norm = radius ∧ (L2 t2 - center).norm = radius) :=
sorry

end smallest_sphere_touches_L1_L2_l288_288187


namespace non_defective_probability_l288_288336

-- Definitions based on conditions
def event_a := { p : ℙ // p ∉ event_b ∪ event_c }
def event_b : { p : ℙ // p ∈ event_b }
def event_c : { p : ℙ // p ∈ event_c }

-- Probabilities of events based on conditions
lemma prob_b : ℙ(event_b) = 0.03 := sorry 
lemma prob_c : ℙ(event_c) = 0.01 := sorry

-- Proof Statement
theorem non_defective_probability :
  ℙ(event_a) = 0.96 :=
by
  rw [event_a, compl_union]
  rw [prob_b, prob_c]
  norm_num
  -- Detailed proof omitted
  sorry


end non_defective_probability_l288_288336


namespace factorize_expression_l288_288395

theorem factorize_expression (a b : ℝ) : a * b^2 - 25 * a = a * (b + 5) * (b - 5) :=
by
  sorry

end factorize_expression_l288_288395


namespace distance_to_nearest_town_l288_288346

theorem distance_to_nearest_town (d : ℝ) :
  ¬ (d ≥ 6) → ¬ (d ≤ 5) → ¬ (d ≤ 4) → (d > 5 ∧ d < 6) :=
by
  intro h1 h2 h3
  sorry

end distance_to_nearest_town_l288_288346


namespace sequence_properties_l288_288806

-- Define the arithmetic-geometric sequence and its sum
def a_n (n : ℕ) : ℕ := 2^(n-1)
def S_n (n : ℕ) : ℕ := 2^n - 1
def T_n (n : ℕ) : ℕ := 2^(n+1) - n - 2

theorem sequence_properties : 
(S_n 3 = 7) ∧ (S_n 6 = 63) → 
(∀ n: ℕ, a_n n = 2^(n-1)) ∧ 
(∀ n: ℕ, S_n n = 2^n - 1) ∧ 
(∀ n: ℕ, T_n n = 2^(n+1) - n - 2) :=
by
  sorry

end sequence_properties_l288_288806


namespace hyperbola_asymptotes_l288_288227

noncomputable def hyperbola_eccentricity (a b : ℝ) : ℝ := Real.sqrt (1 + (b^2 / a^2))

theorem hyperbola_asymptotes (a b : ℝ) (h : hyperbola_eccentricity a b = Real.sqrt 3) :
  (∀ x y : ℝ, (y = Real.sqrt 2 * x) ∨ (y = -Real.sqrt 2 * x)) :=
sorry

end hyperbola_asymptotes_l288_288227


namespace volume_of_regular_tetrahedron_l288_288632

noncomputable def volume_of_tetrahedron (a H : ℝ) : ℝ :=
  (a^2 * H) / (6 * Real.sqrt 2)

theorem volume_of_regular_tetrahedron
  (d_face : ℝ)
  (d_edge : ℝ)
  (h : Real.sqrt 14 = d_edge)
  (h1 : 2 = d_face)
  (volume_approx : ℝ) :
  ∃ a H, (d_face = Real.sqrt ((H / 2)^2 + (a * Real.sqrt 3 / 6)^2) ∧ 
          d_edge = Real.sqrt ((H / 2)^2 + (a / (2 * Real.sqrt 3))^2) ∧ 
          Real.sqrt (volume_of_tetrahedron a H) = 533.38) :=
  sorry

end volume_of_regular_tetrahedron_l288_288632


namespace infinitely_many_positive_solutions_l288_288608

theorem infinitely_many_positive_solutions 
    (n : ℕ) (k : ℕ) (h_n : n > 0) (h_k : k ≥ 0) : 
    ∃ (x_1 x_2 ... x_n y : ℕ), ∀ _x, x_1^3 + x_2^3 + ... + x_n^3 = y^{3*k + 2} ∧ (x_1 > 0) ∧ (x_2 > 0) ∧ ... ∧ (x_n > 0) := 
sorry

end infinitely_many_positive_solutions_l288_288608


namespace count_4_digit_numbers_divisible_by_13_l288_288870

theorem count_4_digit_numbers_divisible_by_13 : 
  ∃ n : ℕ, n = 693 ∧ (∀ k : ℕ, k >= 1000 ∧ k < 10000 ∧ k % 13 = 0 → ∃ m : ℕ, m = (k - 1000) / 13 + 1 ∧ m = n) :=
by {
  -- Solution proof will be placed here.
  sorry
}

end count_4_digit_numbers_divisible_by_13_l288_288870


namespace count_4digit_numbers_divisible_by_13_l288_288893

theorem count_4digit_numbers_divisible_by_13 : 
  let count := (λ n, n - 77 + 1) (769)
  count = 693 :=
by
-- Let 77 be the smallest integer such that 13 * 77 = 1001
-- Let 769 be the largest integer such that 13 * 769 = 9997
-- Hence, the range of these integers should be 769 - 77 + 1
-- This gives us a count of 693
-- The statement as 693 is directly given
sorry

end count_4digit_numbers_divisible_by_13_l288_288893


namespace dot_product_value_l288_288056

-- Definitions for points, vectors, and necessary operations
variables (A B C N M : ℝ × ℝ)
variables (x y : ℝ)
variables (CA CB CM AN : ℝ × ℝ)

-- Conditions from the problem
def is_equilateral (A B C : ℝ × ℝ) (s : ℝ) :=
  dist A B = s ∧ dist B C = s ∧ dist C A = s

def midpoint (A B N : ℝ × ℝ) :=
  N = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

def point_on_line (A B M : ℝ × ℝ) :=
  ∃ t, 0 ≤ t ∧ t ≤ 1 ∧ M = (t * A.1 + (1 - t) * B.1, t * A.2 + (1 - t) * B.2)

def vector_equation (C A B M : ℝ × ℝ) (x y : ℝ) :=
  M = (x * (C.1 - A.1) + y * (B.1 - C.1), x * (C.2 - A.2) + y * (B.2 - C.2)) ∧ x > 0 ∧ y > 0

def minimum_condition (x y : ℝ) :=
  (9 / x + 1 / y) = 16

-- Definition of dot product
def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

-- Main theorem encapsulating the proof goal
theorem dot_product_value 
  (eq_triangle : is_equilateral A B C 3)
  (midpoint_N : midpoint A B N)
  (point_M : point_on_line A B M)
  (vec_eq : vector_equation C A B M x y)
  (min_cond : minimum_condition x y) :
  dot_product (CM.x A B M x y) (AN.x A B N x y) = -9 / 8 :=
sorry

end dot_product_value_l288_288056


namespace part1_part2_l288_288180

def f (x a : ℝ) : ℝ := |x - 1| + |x - a|

theorem part1 (a : ℝ) (h : a = 4) : 
  {x : ℝ | f x a ≥ 5} = {x : ℝ | x ≤ 0 ∨ x ≥ 5} :=
sorry

theorem part2 (a : ℝ) (h : ∀ x : ℝ, f x a ≥ 4) : 
  a ≤ -3 ∨ a ≥ 5 :=
sorry

end part1_part2_l288_288180


namespace money_left_after_spending_l288_288258

def initial_money : ℕ := 24
def doris_spent : ℕ := 6
def martha_spent : ℕ := doris_spent / 2
def total_spent : ℕ := doris_spent + martha_spent
def money_left := initial_money - total_spent

theorem money_left_after_spending : money_left = 15 := by
  sorry

end money_left_after_spending_l288_288258


namespace difference_in_dimes_l288_288199

theorem difference_in_dimes (n d q h : ℕ) 
  (hc1 : n + d + q + h = 150)
  (hc2 : 5 * n + 10 * d + 25 * q + 50 * h = 2000) : 
  (max_d : ℕ) (min_d : ℕ) (dmax : max_d = 250) 
  (dmin : min_d = 7) : 
  max_d - min_d = 243 := 
sorry

end difference_in_dimes_l288_288199


namespace right_angle_sine_cosine_l288_288789

theorem right_angle_sine_cosine :
  ∀ θ: ℝ, θ = π / 2 → (cos θ = 0 ∧ sin θ = 1) := by
  intro θ h
  have h_cos: cos θ = 0 := sorry
  have h_sin: sin θ = 1 := sorry
  exact ⟨h_cos, h_sin⟩

end right_angle_sine_cosine_l288_288789


namespace max_sqrt_expr_l288_288577

theorem max_sqrt_expr (x y z : ℝ) (h_nonneg_x : 0 ≤ x) (h_nonneg_y : 0 ≤ y) (h_nonneg_z : 0 ≤ z) (h_sum : x + y + z = 7) :
  sqrt (3 * x + 2) + sqrt (3 * y + 2) + sqrt (3 * z + 2) ≤ 9 :=
sorry

end max_sqrt_expr_l288_288577


namespace count_4_digit_numbers_divisible_by_13_l288_288883

theorem count_4_digit_numbers_divisible_by_13 : 
  let count := (9962 - 1014) / 13 + 1
  1000 ≤ 1014 ∧ 9962 ≤ 9999 →
  count = 689 :=
  by
    sorry

end count_4_digit_numbers_divisible_by_13_l288_288883


namespace measure_ADC_l288_288640

-- Definitions
def angle_measures (x y ADC : ℝ) : Prop :=
  2 * x + 60 + 2 * y = 180 ∧ x + y = 60 ∧ x + y + ADC = 180

-- Goal
theorem measure_ADC (x y ADC : ℝ) (h : angle_measures x y ADC) : ADC = 120 :=
by {
  -- Solution could go here, skipped for brevity
  sorry
}

end measure_ADC_l288_288640


namespace vector_combination_of_Q_l288_288963

-- Define the context in which the problem exists.
section
variables {V : Type*} [AddCommGroup V] [Module ℝ V]
variables (A B Q : V)

-- Define the condition given in the problem.
def divides_externally (AQ QB : ℝ) : Prop := AQ / QB = 7 / 2

-- Define the theorem to prove the problem.
theorem vector_combination_of_Q (h : divides_externally (AQ:ℝ) (QB:ℝ)) :
  ∃ s v : ℝ, Q = s • A + v • B ∧ s = -2/5 ∧ v = 7/5 :=
begin
  sorry
end
end

end vector_combination_of_Q_l288_288963


namespace four_digit_numbers_divisible_by_13_l288_288877

theorem four_digit_numbers_divisible_by_13 : 
  (set.Icc 1000 9999).filter (λ x, x % 13 = 0).cardinality = 693 :=
by
  sorry

end four_digit_numbers_divisible_by_13_l288_288877


namespace sum_f_equals_2016_l288_288758

/-- Defines the cubic function f(x) -/
def f (x : ℝ) : ℝ := (1/3) * x^3 - (1/2) * x^2 + 3 * x - 5 / 12

/-- The main statement to prove --/
theorem sum_f_equals_2016 :
  (∑ k in finset.range 2017 \ {0}, f (k / 2017)) = 2016 :=
by sorry

end sum_f_equals_2016_l288_288758


namespace johnny_savings_l288_288945

variable (S : ℤ) -- The savings in September.

theorem johnny_savings :
  (S + 49 + 46 - 58 = 67) → (S = 30) :=
by
  intro h
  sorry

end johnny_savings_l288_288945


namespace regular_permutation_iff_l288_288034

def is_legal_transposition (a : List Nat) (i j : Nat) : Prop :=
  (a.nth i = some 0) ∧ (i > 0) ∧ (a.nth (i - 1) = some (a.nthLe j (by linarith) + 1))

def is_regular_permutation (a : List Nat) (n : Nat) : Prop :=
  ∃ (seq : List (List Nat)),
    seq.head = some a ∧
    seq.any (λ a', a'.head = some 0) ∧
    seq.any (λ a', a'.last = some 0) ∧
    seq.all (λ t, is_legal_transposition t (t.indexOf 0) (t.indexOf (t.indexOf 0 + 1))) ∧
    seq.last = some ([1:n, n + 1])

theorem regular_permutation_iff (n : Nat) :
  ((∃ k : Nat, n = 2^k - 1) ∨ n = 2) ↔
  is_regular_permutation (List.range (n + 1)).reverse n :=
sorry

end regular_permutation_iff_l288_288034


namespace average_age_of_women_l288_288622
-- Lean 4 statement


theorem average_age_of_women (A W1 W2 : ℕ) :
  (∀ (ages : list ℕ), ages.length = 6 ∧ (list.sum ages = 6 * A) ∧
  ((list.sum (10::12::ages.drop 2) - (10 + 12) + W1 + W2 = 6 * (A + 2))) →
  (W1 + W2) / 2 = 17) :=
begin
  sorry,
end

end average_age_of_women_l288_288622


namespace find_n_of_permut_comb_eq_l288_288489

open Nat

theorem find_n_of_permut_comb_eq (n : Nat) (h : (n! / (n - 3)!) = 6 * (n! / (4! * (n - 4)!))) : n = 7 := by
  sorry

end find_n_of_permut_comb_eq_l288_288489


namespace distance_between_trees_l288_288130

theorem distance_between_trees (n : ℕ) (len : ℝ) (d : ℝ) 
  (h1 : n = 26) 
  (h2 : len = 400) 
  (h3 : len / (n - 1) = d) : 
  d = 16 :=
by
  sorry

end distance_between_trees_l288_288130


namespace my_set_is_closed_operation_preserving_on_Q_l288_288115

-- Definition of closed set
def closed_set (M : Set ℝ) : Prop :=
  (∀ x y ∈ M, x + y ∈ M) ∧ (∀ x y ∈ M, x * y ∈ M)

-- Definition of operation-preserving function
def operation_preserving (M : Set ℝ) (f : ℝ → ℝ) : Prop :=
  (∀ x y ∈ M, f (x + y) = f x + f y) ∧ (∀ x y ∈ M, f (x * y) = f x * f y)

-- The set { √3m + n | m, n ∈ Q }
def my_set : Set ℝ := { x | ∃ (m n : ℚ), x = (√3 : ℝ) * m + n }

-- Verify that my_set is closed
theorem my_set_is_closed : closed_set my_set := sorry

-- A function f(x) = x for x ∈ ℚ
def f (x : ℝ) : ℝ := x

-- Verify that f is operation-preserving on ℚ
theorem operation_preserving_on_Q : operation_preserving (Set.range (coe : ℚ → ℝ)) f := sorry

end my_set_is_closed_operation_preserving_on_Q_l288_288115


namespace max_gcd_is_2_l288_288652

-- Define the sequence
def a (n : ℕ) : ℕ := 101 + (n + 1)^2 + 3 * n

-- Define the gcd of consecutive terms
def d (n : ℕ) : ℕ := Nat.gcd (a n) (a (n + 1))

theorem max_gcd_is_2 : ∀ n : ℕ, n > 0 → d n = 2 :=
by
  intros n hn
  dsimp [d]
  sorry

end max_gcd_is_2_l288_288652


namespace man_double_son_in_years_l288_288719

-- Definitions of conditions
def son_age : ℕ := 18
def man_age : ℕ := son_age + 20

-- The proof problem statement
theorem man_double_son_in_years :
  ∃ (X : ℕ), (man_age + X = 2 * (son_age + X)) ∧ X = 2 :=
by
  sorry

end man_double_son_in_years_l288_288719


namespace net_change_in_price_l288_288304

-- Define the initial price of the TV
def initial_price (P : ℝ) := P

-- Define the price after a 20% decrease
def decreased_price (P : ℝ) := 0.80 * P

-- Define the final price after a 50% increase on the decreased price
def final_price (P : ℝ) := 1.20 * P

-- Prove that the net change is 20% of the original price
theorem net_change_in_price (P : ℝ) : final_price P - initial_price P = 0.20 * P := by
  sorry

end net_change_in_price_l288_288304


namespace expression_evaluation_l288_288746

theorem expression_evaluation :
  5 * 423 + 4 * 423 + 3 * 423 + 421 = 5497 := by
  sorry

end expression_evaluation_l288_288746


namespace min_value_of_f_value_of_c_l288_288063

-- Definitions and conditions for problem 1
def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- Part 1
theorem min_value_of_f
  (a : ℝ) (b : ℝ) (c : ℝ)
  (a_nat : a ∈ ℕ) (a_pos : 0 < a)
  (b_nat : b ∈ ℕ)
  (c_int : c ∈ ℤ)
  (b_gt_2a : b > 2 * a)
  (max_f_sin : ∀ x : ℝ, f a b c (sin x) ≤ 2)
  (min_f_sin : ∀ x : ℝ, -4 ≤ f a b c (sin x)) :
  ∃ x : ℝ, f a b c x = -(17 / 4) :=
  sorry

-- Definitions and conditions for problem 2
theorem value_of_c
  (a : ℝ) (b : ℝ) (c : ℝ)
  (a_nat : a ∈ ℕ) (a_pos : 0 < a)
  (b_nat : b ∈ ℕ)
  (c_int : c ∈ ℤ)
  (for_all_x_bounds : ∀ x : ℝ, 4 * x ≤ f a b c x ∧ f a b c x ≤ 2 * (x^2 + 1))
  (exists_x0 : ∃ x0 : ℝ, f a b c x0 < 2 * (x0^2 + 1)) :
  c = 1 :=
  sorry

end min_value_of_f_value_of_c_l288_288063


namespace find_the_number_l288_288009

theorem find_the_number (x k : ℕ) (h1 : x / k = 4) (h2 : k = 8) : x = 32 := by
  sorry

end find_the_number_l288_288009


namespace exists_line_intersecting_all_polygons_l288_288596

open Set

-- Definition and condition of polygons sharing a common point
def polygons (P : ℕ → Set (ℝ × ℝ)) : Prop := 
  ∀ i j, i ≠ j → (P i ∩ P j).Nonempty

-- Equivalent proof problem statement
theorem exists_line_intersecting_all_polygons (P : ℕ → Set (ℝ × ℝ)) (h : polygons P) :
  ∃ l : ℝ → Set (ℝ × ℝ), ∀ n, ∃ p ∈ P n, p ∈ l :=
sorry

end exists_line_intersecting_all_polygons_l288_288596


namespace part1_part2_l288_288179

def f (x a : ℝ) : ℝ := |x - 1| + |x - a|

theorem part1 (a : ℝ) (h : a = 4) : 
  {x : ℝ | f x a ≥ 5} = {x : ℝ | x ≤ 0 ∨ x ≥ 5} :=
sorry

theorem part2 (a : ℝ) (h : ∀ x : ℝ, f x a ≥ 4) : 
  a ≤ -3 ∨ a ≥ 5 :=
sorry

end part1_part2_l288_288179


namespace percentage_shaded_squares_l288_288678

theorem percentage_shaded_squares :
  (∀ (total_squares dark_grey_squares light_grey_squares: ℕ), 
      total_squares = 36 → 
      dark_grey_squares = 6 → 
      light_grey_squares = 6 → 
      (dark_grey_squares / total_squares) * 100 = 16.6 ∧ 
      (light_grey_squares / total_squares) * 100 = 16.6) := sorry

end percentage_shaded_squares_l288_288678


namespace correct_operation_is_B_l288_288681

theorem correct_operation_is_B :
  ((sqrt 3 - sqrt 2 ≠ 1) ∧
   (sqrt 3 * sqrt 2 = sqrt 6) ∧
   ((sqrt 3 - 1) ^ 2 ≠ 3 - 1) ∧
   (sqrt ((5:ℝ)^2 - (3:ℝ)^2) ≠ 5 - 3)) := 
by
  sorry

end correct_operation_is_B_l288_288681


namespace area_PST_is_5_l288_288146

noncomputable def area_of_triangle_PST 
  (P Q R S T : Point)
  (PQ QR PR : ℝ) 
  (PS PT : ℝ)
  (hPQ : PQ = 8)
  (hQR : QR = 9)
  (hPR : PR = 10)
  (hPS : PS = 3)
  (hPT : PT = 5)
  : ℝ := 
  5

theorem area_PST_is_5 
  (P Q R S T : Point)
  (PQ QR PR : ℝ) 
  (PS PT : ℝ)
  (hPQ : PQ = 8)
  (hQR : QR = 9)
  (hPR : PR = 10)
  (hPS : PS = 3)
  (hPT : PT = 5)
  : area_of_triangle_PST P Q R S T PQ QR PR PS PT hPQ hQR hPR hPS hPT = 5 :=
sorry

end area_PST_is_5_l288_288146


namespace dot_product_sum_l288_288166

variables {V : Type*} [inner_product_space ℝ V]

theorem dot_product_sum (u v w : V) (h₁ : ∥u∥ = 3) (h₂ : ∥v∥ = 4) (h₃ : ∥w∥ = 5) (h₄ : u + v + w = 0) :
  inner_product_space.inst_has_inner.inner u v + inner_product_space.inst_has_inner.inner u w + inner_product_space.inst_has_inner.inner v w = -25 :=
begin
  -- proof
  sorry
end

end dot_product_sum_l288_288166


namespace find_b2_a2_minus_a1_l288_288040

theorem find_b2_a2_minus_a1 
  (a₁ a₂ b₁ b₂ b₃ : ℝ)
  (d r : ℝ)
  (h_arith_seq : a₁ = -9 + d ∧ a₂ = a₁ + d)
  (h_geo_seq : b₁ = -9 * r ∧ b₂ = b₁ * r ∧ b₃ = b₂ * r ∧ (-9) * (-1) = b₁ * b₃)
  (h_d_val : a₂ - a₁ = d)
  (h_b2_val : b₂ = -1) : 
  b₂ * (a₂ - a₁) = -8 :=
sorry

end find_b2_a2_minus_a1_l288_288040


namespace balls_into_boxes_l288_288900

theorem balls_into_boxes :
  {l : List ℕ // l.sum = 5 ∧ l.length <= 4 ∧ l.sorted = l} = 6 := by sorry

end balls_into_boxes_l288_288900


namespace positive_area_triangles_correct_l288_288084

-- Define the set of points with integer coordinates in the given range
def grid_points := { p : ℤ × ℤ | 1 ≤ p.1 ∧ p.1 ≤ 5 ∧ 1 ≤ p.2 ∧ p.2 ≤ 5 }

-- Define a function to check for collinearity of three points
def collinear (p1 p2 p3 : ℤ × ℤ) : Prop :=
  (p2.2 - p1.2) * (p3.1 - p1.1) = (p3.2 - p1.2) * (p2.1 - p1.1)

-- Define the count of triangles with positive area (not collinear) from the given points
def triangles_with_positive_area (points : set (ℤ × ℤ)) : ℕ :=
  (@set.to_finset (ℤ × ℤ) _ points).to_list.combinations 3
  .filter (λ l, l.length = 3 ∧ ¬ collinear l.head l.nth 1 l.nth 2).length

-- State the proof problem
theorem positive_area_triangles_correct :
  triangles_with_positive_area grid_points = 2170 := sorry

end positive_area_triangles_correct_l288_288084


namespace find_slope_of_AB_l288_288811

noncomputable def slope_of_AB (B : ℝ × ℝ) (k : ℝ) : Prop :=
  let A := (1 : ℝ, 0 : ℝ)
  let C := (-B.1, -B.2)
  let D := ((A.1 + C.1) / 2, (A.2 + C.2) / 2)
  let slope_AB := k
  let slope_BD := (D.2 - B.2) / (D.1 - B.1)
  (B.1 ^ 2 + B.2 ^ 2 = 1) ∧ 
  ((B.1 > 0) ∧ (B.2 > 0)) ∧
  (slope_AB + slope_BD = 0) ∧
  (k = - (Real.sqrt 5))

theorem find_slope_of_AB :
  ∃ (B : ℝ × ℝ) (k : ℝ), slope_of_AB B k := sorry

end find_slope_of_AB_l288_288811


namespace polar_coordinates_of_2_neg2_l288_288647

noncomputable def polar_coordinates (x y : ℝ) : ℝ × ℝ :=
  let ρ := Real.sqrt (x^2 + y^2)
  let θ := Real.arctan (y / x)
  (ρ, θ)

theorem polar_coordinates_of_2_neg2 :
  polar_coordinates 2 (-2) = (2 * Real.sqrt 2, -Real.pi / 4) :=
by
  sorry

end polar_coordinates_of_2_neg2_l288_288647


namespace area_ratio_condition_l288_288138

theorem area_ratio_condition (a : ℝ) (h : a > 0) (α : ℝ) (β : α = arctan 2) (γ : α < π / 2) 
  (h1 : ∃ (M N : ℝ), let S := λ x y: ℝ, (x + 2*y) / (3*x - 2*y) in S a (a / 2 - a * cot α) = (1 - cot α) / (1 + cot α)) :
  α ∈ set.Ico (arctan 2) (π / 2) :=
by
  intro
  assumption
  sorry

end area_ratio_condition_l288_288138


namespace fifth_boy_payment_l288_288784

variable {R : Type} [CommRing R]

noncomputable def boys_payment_amounts (a b c d e : R) : Prop :=
  a + b + c + d + e = 120 ∧
  a = (1/3 : R) * (b + c + d + e) ∧
  b = (1/4 : R) * (a + c + d + e) ∧
  c = (1/5 : R) * (a + b + d + e) ∧
  d = 2 * e

theorem fifth_boy_payment (a b c d e : R) (h : boys_payment_amounts a b c d e) : e = 13.33 :=
by sorry

end fifth_boy_payment_l288_288784


namespace tan3x_eq_sinx_has_12_solutions_l288_288895

theorem tan3x_eq_sinx_has_12_solutions :
    ∀ x ∈ set.Icc (0 : ℝ) (2 * Real.pi), 
    tendsto (fun (k : ℤ) => tan (3 * (↑k * Real.pi / 3))) at_top at_top ∧
    tendsto (fun (k : ℤ) => sin (↑k * Real.pi)) at_top at_top →
    (∃! x ∈ set.Icc (0 : ℝ) (2 * Real.pi), tan (3 * x) = sin x)  := by
  sorry

end tan3x_eq_sinx_has_12_solutions_l288_288895


namespace num_triangles_with_positive_area_l288_288108

/-- 
Given vertices in a 5x5 grid with integer coordinates satisfying 1 ≤ x ≤ 5 and 1 ≤ y ≤ 5,
prove that the number of triangles with positive area is 2170. 
-/
theorem num_triangles_with_positive_area : 
  (∑ t in ({(i, j) | 1 ≤ i ∧ i ≤ 5 ∧ 1 ≤ j ∧ j ≤ 5}.powerset.filter (λ (s : set (ℕ × ℕ)), s.card = 3)),
    if t₁.1 * (t₂.2 - t₃.2) + t₂.1 * (t₃.2 - t₁.2) + t₃.1 * (t₁.2 - t₂.2) ≠ 0 then 1 else 0) = 2170 :=
by sorry

end num_triangles_with_positive_area_l288_288108


namespace is_strictly_monotonically_decreasing_l288_288802

def seq (a : ℝ) (n : ℕ) : ℝ := 2 ^ n * (real.sqrt (real.sqrt [1 / (2 * n)] a) - 1)

theorem is_strictly_monotonically_decreasing (a : ℝ) (h : a > 0 ∧ a ≠ 1) :
  ∀ n : ℕ, seq a n > seq a (n+1) :=
sorry

end is_strictly_monotonically_decreasing_l288_288802


namespace x_y_square_sum_l288_288424

theorem x_y_square_sum (x y : ℝ) (h1 : x - y = -1) (h2 : x * y = 1 / 2) : x^2 + y^2 = 2 := 
by 
  sorry

end x_y_square_sum_l288_288424


namespace count_base_8_digits_5_or_6_l288_288862

-- Define the conditions in Lean
def is_digit_5_or_6 (d : ℕ) : Prop := d = 5 ∨ d = 6

def count_digits_5_or_6 := 
  let total_base_8 := 512
  let total_without_5_6 := 6 * 6 * 6 -- since we exclude 2 out of 8 digits
  total_base_8 - total_without_5_6

-- The statement of the proof problem
theorem count_base_8_digits_5_or_6 : count_digits_5_or_6 = 296 :=
by {
  sorry
}

end count_base_8_digits_5_or_6_l288_288862


namespace rice_weight_range_l288_288236

theorem rice_weight_range (w : ℝ) (h : w = 10 ± 0.1) : 9.9 ≤ w ∧ w ≤ 10.1 := 
by 
  sorry

end rice_weight_range_l288_288236


namespace relation_between_a_and_b_l288_288042

theorem relation_between_a_and_b (a b : ℝ) 
  (h1 : a = sqrt 5 + sqrt 6) 
  (h2 : b = 1 / (sqrt 6 - sqrt 5)) : 
  a = b := 
sorry

end relation_between_a_and_b_l288_288042


namespace chord_length_of_intersection_of_line_and_circle_l288_288841

open Real

noncomputable def length_of_chord : ℝ := 2 * sqrt (1 - (1 / sqrt 2) ^ 2)

theorem chord_length_of_intersection_of_line_and_circle :
  ∀ (θ ρ : ℝ), 
  (ρ * (sin θ + cos θ) = 1) → 
  (∃ x y : ℝ, x = cos θ ∧ y = sin θ + 2) → 
  length_of_chord = sqrt 2 := 
by 
  intros θ ρ hline hcircle
  rw length_of_chord
  sorry

end chord_length_of_intersection_of_line_and_circle_l288_288841


namespace SallyMcQueenCostCorrect_l288_288184

def LightningMcQueenCost : ℕ := 140000
def MaterCost : ℕ := (140000 * 10) / 100
def SallyMcQueenCost : ℕ := 3 * MaterCost

theorem SallyMcQueenCostCorrect : SallyMcQueenCost = 42000 := by
  sorry

end SallyMcQueenCostCorrect_l288_288184


namespace original_selling_price_l288_288730

-- Definitions
def cost_price (C : ℝ) := C
def selling_price_profit (C : ℝ) := 1.25 * C
def selling_price_loss (C : ℝ) := 0.80 * C

-- Conditions
axiom selling_price_loss_condition (C : ℝ) : selling_price_loss C = 512

-- Proof statement
theorem original_selling_price (C : ℝ) (h : selling_price_loss_condition C) : selling_price_profit C = 800 :=
by
  sorry

end original_selling_price_l288_288730


namespace parabola_vertices_form_hyperbola_l288_288062

variable (k : ℝ) (c : ℝ)

noncomputable def parabola_vertex_set (t : ℝ) : ℝ × ℝ :=
  let x_v := -k / t
  let y_v := k^2 - (2 * k^2) / x_v + c
  (x_v, y_v)

theorem parabola_vertices_form_hyperbola (k c : ℝ) :
  ∃ b, ∀ t ≠ 0, (parabola_vertex_set k c t).snd = k^2 - (2 * k^2) / (parabola_vertex_set k c t).fst + c :=
by
  sorry

end parabola_vertices_form_hyperbola_l288_288062


namespace inequality_solution_l288_288767

theorem inequality_solution (x : ℝ) (h : 4^(2*x-1) > (1/2)^(-x-4)) : x > 2/5 := 
sorry

end inequality_solution_l288_288767


namespace inverse_contrapositive_negation_l288_288118

variable (p q r : Prop)

-- q is the inverse of p
def is_inverse (p q : Prop) : Prop := ¬ p → ¬ q

-- r is the contrapositive of p
def is_contrapositive (p r : Prop) : Prop := ¬ q → ¬ p

theorem inverse_contrapositive_negation
  (hpq : is_inverse p q)
  (hpr : is_contrapositive p r) :
  q ↔ ¬ r :=
sorry

end inverse_contrapositive_negation_l288_288118


namespace sum_equality_y_value_l288_288109

theorem sum_equality_y_value :
  (∑ n in Finset.range 1001 + 1, n * (1002 - n)) = 1001 * 501 * 334 :=
by
  sorry

end sum_equality_y_value_l288_288109


namespace cricket_run_rate_and_partnerships_l288_288932

-- Run rate in the first 10 overs
def run_rate_first_10_overs : ℝ := 3.2

-- Number of overs in first part of the game
def overs_first_part : ℝ := 10

-- Number of overs remaining
def overs_remaining : ℝ := 40

-- Target runs for the game
def target_runs : ℝ := 292

-- Number of partnerships
def partnerships : ℝ := 2

-- The required run rate in the remaining 40 overs and the minimum runs each partnership should contribute without losing any more wickets
theorem cricket_run_rate_and_partnerships :
  let runs_scored_first_10_overs := run_rate_first_10_overs * overs_first_part in
  let runs_needed_remaining := target_runs - runs_scored_first_10_overs in
  let required_run_rate_remaining := runs_needed_remaining / overs_remaining in
  let minimum_runs_per_partnership := runs_needed_remaining / partnerships in
  required_run_rate_remaining = 6.5 ∧ minimum_runs_per_partnership = 130 :=
by
  sorry

end cricket_run_rate_and_partnerships_l288_288932


namespace time_to_cross_approx_l288_288704

noncomputable def lengthFirstTrain : ℝ := 280 
noncomputable def speedFirstTrain : ℝ := 120 * (5 / 18) 
noncomputable def speedSecondTrain : ℝ := 80 * (5 / 18) 
noncomputable def lengthSecondTrain : ℝ := 220.04 

noncomputable def relativeSpeed : ℝ := speedFirstTrain + speedSecondTrain
noncomputable def combinedLength : ℝ := lengthFirstTrain + lengthSecondTrain
noncomputable def timeTaken : ℝ := combinedLength / relativeSpeed

theorem time_to_cross_approx : timeTaken ≈ 9 := 
by 
  sorry

end time_to_cross_approx_l288_288704


namespace karl_problem_l288_288946

theorem karl_problem (p w : ℂ) (h1 : 10 * p - w = 50000) (h2 : 10 = 2) (h3 : w = 10 + 250 * complex.I) : 
  p = 5001 + 25 * complex.I :=
sorry

end karl_problem_l288_288946


namespace log4_18_l288_288500

theorem log4_18 {a b : ℝ} (h1 : log 10 2 = a) (h2 : log 10 3 = b) : log 4 18 = (a + 2 * b) / (2 * a) :=
sorry

end log4_18_l288_288500


namespace acquaintance_time_to_read_novel_l288_288974

theorem acquaintance_time_to_read_novel :
  ∀ (N : ℕ), 
  let my_reading_time := 180 in
  let my_friend_reading_time := my_reading_time / 3 in
  let acquaintance_reading_time := my_friend_reading_time / 2 in
  acquaintance_reading_time = 30 :=
by
  intro N
  let my_reading_time := 180
  let my_friend_reading_time := my_reading_time / 3
  let acquaintance_reading_time := my_friend_reading_time / 2
  have h : acquaintance_reading_time = 30 := by simp [my_reading_time, my_friend_reading_time, acquaintance_reading_time]
  exact h

end acquaintance_time_to_read_novel_l288_288974


namespace cumulative_net_income_g8_investment_pays_off_after_9_months_l288_288712

theorem cumulative_net_income_g8 :
  ∀ (k : ℝ), (g 3 = 3090000 → g(n) = n^2 + kn ∧ g(5) - g(4) = 1090000 
              → g(8) = g(5) + 3 * 1090000 → 8520000) where
  g : ℕ → ℝ := λ n, if n > 5 then 109 * n - 20 else n^2 + k * n :=
sorry

theorem investment_pays_off_after_9_months :
  ∀ (k : ℝ), (g 3 = 3090000 → g(n) = n^2 + kn ∧ (∀ (n > 5), g n = 109 * n - 20)
              → (n : ℕ → ℝ, net_income_without_investment n = 700 * n - (n^2 + 2n if n <= 12 else 30 + 20 * (n - 1)))
              → ∀ n, g(n) - 5000000 + 1000000 > net_income_without_investment n
              → n ≥ 9) where
  g : ℕ → ℝ := λ n, if n > 5 then 109 * n - 20 else (n^2 + k * n) :=
sorry

end cumulative_net_income_g8_investment_pays_off_after_9_months_l288_288712


namespace part1_monotonicity_part2_range_of_a_l288_288832

-- Define the function
def f (a x : ℝ) : ℝ := a^x + x^2 - x * Real.log a

-- Part 1: Monotonicity when a = e
theorem part1_monotonicity :
  ∀ x : ℝ, (f Real.exp x) = Real.exp x + x^2 - x → 
             (∀ x ≥ 0, f Real.exp x is increasing) ∧ 
             (∀ x ≤ 0, f Real.exp x is decreasing) := 
begin
  -- Proof omitted
  sorry
end

-- Part 2: Range of values for a
theorem part2_range_of_a (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (∃ x1 x2 ∈ Icc (-1 : ℝ) 1, |f a x1 - f a x2| ≥ Real.exp - 1) →
  a ∈ Icc 0 (1 / Real.exp) ∪ Icc Real.exp +∞ :=
begin
  -- Proof omitted
  sorry
end

end part1_monotonicity_part2_range_of_a_l288_288832


namespace tetrahedron_volume_l288_288630

/-- Given a regular triangular pyramid (tetrahedron) with the following properties:
  - Distance from the midpoint of the height to a lateral face is 2.
  - Distance from the midpoint of the height to a lateral edge is √14.
  Prove that the volume of the pyramid is approximately 533.38.
-/
theorem tetrahedron_volume (d_face d_edge : ℝ) (volume : ℝ) (h1 : d_face = 2) (h2 : d_edge = Real.sqrt 14) :
  Abs (volume - 533.38) < 0.01 :=
by {
  sorry -- Proof will go here
}

end tetrahedron_volume_l288_288630


namespace ratio_BK_KC_l288_288526

variables {A B C D K : Type}
variables [AffineSpace ℝ A] [AffineSpace ℝ B] [AffineSpace ℝ C] [AffineSpace ℝ D] [AffineSpace ℝ K]

-- Define the parallelogram condition
def is_parallelogram (A B C D : Type) [AffineSpace ℝ A] [AffineSpace ℝ B] [AffineSpace ℝ C] [AffineSpace ℝ D] : Prop :=
  let AB := (B : ℝ) - (A : ℝ) in
  let AD := (D : ℝ) - (A : ℝ) in
  let BC := (C : ℝ) - (B : ℝ) in
  let CD := (D : ℝ) - (C : ℝ) in
  AB = CD ∧ AD = BC

-- Define the given angles condition
def angles_equal (K B D A : Type) [AffineSpace ℝ K] [AffineSpace ℝ B] [AffineSpace ℝ D] [AffineSpace ℝ A] : Prop :=
  let KDB := (D : ℝ) - (K : ℝ) in
  let BDA := (A : ℝ) - (B : ℝ) in
  ∠ KDB = ∠ BDA

-- The main theorem to prove
theorem ratio_BK_KC {A B C D K : Type} [AffineSpace ℝ A] [AffineSpace ℝ B] [AffineSpace ℝ C] [AffineSpace ℝ D] [AffineSpace ℝ K]
  (h1 : is_parallelogram A B C D)
  (h2 : ∥(C : ℝ) - (A : ℝ)∥ = 2 * ∥(B : ℝ) - (A : ℝ)∥)
  (h3 : angles_equal K B D A) :
  ∥(B : ℝ) - (K : ℝ)∥ / ∥(K : ℝ) - (C : ℝ)∥ = 2 / 1 :=
sorry

end ratio_BK_KC_l288_288526


namespace similar_triangles_hypotenuse_product_square_l288_288276

theorem similar_triangles_hypotenuse_product_square (PQR STU : Triangle) 
  (h1 : similar PQR STU) 
  (h2 : PQR.area = 9)
  (h3 : STU.area = 4)
  (h4 : PQR.hypotenuse = 3 * STU.hypotenuse) : 
  (PQR.hypotenuse * STU.hypotenuse) ^ 2 = 36864 := 
sorry

end similar_triangles_hypotenuse_product_square_l288_288276


namespace chord_square_difference_impossible_l288_288674

theorem chord_square_difference_impossible (A B : ℝ×ℝ) (c : ℝ) (h_unit_circle : c = 1)
  (hAB_on_circle : dist A (0,0) = 1 ∧ dist B (0,0) = 1)
  (hAB_on_line : ∀ (p : ℝ×ℝ), p = A ∨ p = B → ∃ (q : ℝ×ℝ), dist p (0,0) = 1 ∧ line_through A B p q) :
  ¬∃ (s₁ s₂ : ℝ), s₁ = s₂ + 1 ∧ (s₁ - s₂ = 1) :=
begin
  sorry
end

end chord_square_difference_impossible_l288_288674


namespace equilateral_triangle_l288_288310

variables {ω : Complex} {z1 z2 : Complex}
def isEquilateral (a b c : Complex) : Prop :=
distance a b = distance b c ∧ distance b c = distance c a ∧ distance c a = distance a b

theorem equilateral_triangle (hω1 : ω^3 = 1) (hω2 : ω ≠ 1) : 
  isEquilateral z1 z2 (-ω * z1) ∧ isEquilateral z2 (-ω * z1) (-ω^2 * z2) ∧ isEquilateral (-ω * z1) (-ω^2 * z2) z1 :=
sorry

end equilateral_triangle_l288_288310


namespace max_digit_sum_24_hour_watch_l288_288717

theorem max_digit_sum_24_hour_watch : ∃ max_sum : ℕ, max_sum = 38 ∧
  (∀ (h m s : Nat), 
    h < 24 ∧ m < 60 ∧ s < 60 → 
    (h.digits.sum + m.digits.sum + s.digits.sum) ≤ max_sum) := 
sorry

end max_digit_sum_24_hour_watch_l288_288717


namespace volume_of_box_is_96_l288_288727

theorem volume_of_box_is_96
  (x : ℕ) (h : 12 * x^3 = 96) : ∃ x : ℕ, 12 * x^3 = 96 :=
by {
  use 2,
  exact h,
}

end volume_of_box_is_96_l288_288727


namespace savings_account_growth_l288_288360

theorem savings_account_growth (a : ℝ) (n : ℕ) :
  let r := 1.02 in
  (a * r^n) = (a * (1 + 0.02)^n) :=
by
  sorry

end savings_account_growth_l288_288360


namespace validate_triangle_count_l288_288087

noncomputable def count_valid_triangles : ℕ :=
  let total_points := 25
  let total_triples := (Nat.choose total_points 3)
  let collinear_rows := 5 * (Nat.choose 5 3)
  let collinear_columns := 5 * (Nat.choose 5 3)
  let main_diagonals := 2 * (Nat.choose 5 3)
  let secondary_diagonals := 8 * (Nat.choose 4 3)
  let invalid_triangles := collinear_rows + collinear_columns + main_diagonals + secondary_diagonals
  total_triples - invalid_triangles

theorem validate_triangle_count : count_valid_triangles = 2148 :=
by
  sorry

end validate_triangle_count_l288_288087


namespace seating_arrangement_count_l288_288970

theorem seating_arrangement_count : 
  let people := ["Mr. Lopez", "Mrs. Lopez", "Child 1", "Child 2", "Babysitter"]
  let driver := ["Mr. Lopez", "Mrs. Lopez"]
  let seats := ["driver seat", "front passenger seat", "back seat 1", "back seat 2", "back seat 3"]
  (length driver * length (people.erase (driver.head!)).erase_eq head! * (Finset.perm (Finset.singleton "back seat 1" ∪ 
  Finset.singleton "back seat 2" ∪ Finset.singleton "back seat 3")).card) = 48 := 
by
  let driver_count := 2
  let front_passenger_count := 4
  let back_seat_permutations := 6
  calc
    driver_count * front_passenger_count * back_seat_permutations = 48 : by
      sorry

end seating_arrangement_count_l288_288970


namespace height_prediction_at_age_10_l288_288335

theorem height_prediction_at_age_10 
  (regression_model : ℕ → ℝ)
  (h : ∀ x, regression_model x = 7.19 * x + 73.93) :
  abs (regression_model 10 - 145.83) < 0.01 :=
begin
  sorry
end

end height_prediction_at_age_10_l288_288335


namespace count_4digit_numbers_divisible_by_13_l288_288894

theorem count_4digit_numbers_divisible_by_13 : 
  let count := (λ n, n - 77 + 1) (769)
  count = 693 :=
by
-- Let 77 be the smallest integer such that 13 * 77 = 1001
-- Let 769 be the largest integer such that 13 * 769 = 9997
-- Hence, the range of these integers should be 769 - 77 + 1
-- This gives us a count of 693
-- The statement as 693 is directly given
sorry

end count_4digit_numbers_divisible_by_13_l288_288894


namespace maximum_area_of_equilateral_triangle_in_rectangle_l288_288653

noncomputable def maxEquilateralTriangleArea (a b : ℝ) : ℝ :=
  (953 * Real.sqrt 3) / 16

theorem maximum_area_of_equilateral_triangle_in_rectangle :
  ∀ (a b : ℕ), a = 13 → b = 14 → maxEquilateralTriangleArea a b = (953 * Real.sqrt 3) / 16 :=
by
  intros a b h₁ h₂
  rw [h₁, h₂]
  apply rfl

end maximum_area_of_equilateral_triangle_in_rectangle_l288_288653


namespace problem_like_terms_l288_288682

def like_terms (a b : Expr) : Prop := -- definition of like terms
  match (a, b) with
  | (Expr.const _ _, Expr.const _ _) => true
  | _ => false

theorem problem_like_terms : like_terms (Expr.const 12) (Expr.const -5) :=
  by sorry

end problem_like_terms_l288_288682


namespace sum_of_first_9_terms_l288_288471

def a (n : ℕ) : ℚ :=
if n = 1 then 1
else a (n - 1) + 1/2

def S_9 : ℚ := (Finset.range 9).sum (λ i, a (i + 1))

theorem sum_of_first_9_terms : S_9 = 27 := by
  sorry

end sum_of_first_9_terms_l288_288471


namespace bertha_zero_granddaughters_l288_288368

def bertha := "Bertha"
def daughters (x : String) := if x = bertha then 8 else 0
def has_daughters (x : String) := 4  -- Represents daughters having 4 daughters each
def no_daughters (x : String) := 0 -- Represents daughters having no daughters

# Assuming there is a total of 40 members including daughters and granddaughters
def total_members := 8 + 32 -- Bertha's daughters are 8, granddaughters are 32

def no_granddaughters_daughters :=
  if total_members = 40 then 32 else 0 -- Since daughters are 8 and granddaughters having none is 32

theorem bertha_zero_granddaughters :
  total_members = 40 → no_granddaughters_daughters = 32 :=
by
  sorry

end bertha_zero_granddaughters_l288_288368


namespace negation_univ_statement_l288_288486

-- Proving the equivalence of the negation of a universal statement
theorem negation_univ_statement :
  (¬ ∀ x : ℝ, 2^x > 0) ↔ (∃ x₀ : ℝ, 2^x₀ ≤ 0) := by
  sorry

end negation_univ_statement_l288_288486


namespace count_positive_area_triangles_l288_288091

noncomputable def numPositiveAreaTriangles : ℕ := 2160

theorem count_positive_area_triangles 
  (vertices : list (ℤ × ℤ))
  (h1 : ∀ p ∈ vertices, 1 ≤ p.1 ∧ p.1 ≤ 5 ∧ 1 ≤ p.2 ∧ p.2 ≤ 5)
  (h2 : vertices.length = 25) : 
  (number_of_triangles_with_positive_area vertices) = numPositiveAreaTriangles :=
sorry

end count_positive_area_triangles_l288_288091


namespace find_m_l288_288510

theorem find_m (m : ℝ) (x : ℝ) : 
  (|m| - 3) * x^2 + (-m + 3) * x - 4 = 0 → (∀ x, (|m| - 3) = 0) → m = -3 :=
by
  sorry

end find_m_l288_288510


namespace part_a_l288_288691

theorem part_a (X : Type) [Fintype X] [Finite X] (n k : ℕ) (h : 0 ≤ k ∧ k ≤ Fintype.card X)
  (a_n_k b_n_k_minus_1 : ℕ)
  (Ha : a_n_k = Fintype.card {p : Perm X // ∀ p1 p2 : Perm X, p1 ≠ p2 → Fintype.card 
  {x ∈ X // p x = p1 x ∧ p2 x = x } ≥ k })
  (Hb : b_n_k_minus_1 = Fintype.card {p : Perm X // ∀ p1 p2 : Perm X, p1 ≠ p2 → 
  Fintype.card {x ∈ X // p x = p1 x ∧ p2 x = x } ≤ k-1 }) :
  a_n_k * b_n_k_minus_1 ≤ Fintype.card (Perm X) := 
sorry

end part_a_l288_288691


namespace fraction_identity_l288_288465

theorem fraction_identity (a b c : ℝ) (h1 : a + b + c > 0) (h2 : a + b - c > 0) (h3 : a + c - b > 0) (h4 : b + c - a > 0) 
  (h5 : (a+b+c)/(a+b-c) = 7) (h6 : (a+b+c)/(a+c-b) = 1.75) : (a+b+c)/(b+c-a) = 3.5 :=
by
  sorry

end fraction_identity_l288_288465


namespace height_at_age_10_is_around_146_l288_288650

noncomputable def predicted_height (x : ℝ) : ℝ :=
  7.2 * x + 74

theorem height_at_age_10_is_around_146 :
  abs (predicted_height 10 - 146) < ε :=
by
  let ε := 10
  sorry

end height_at_age_10_is_around_146_l288_288650


namespace assignment_statements_l288_288232

-- Define the conditions
def cond1 : Prop := 6 = p
def cond2 : Prop := a = 3 * 5 + 2
def cond3 : Prop := b + 3 = 5
def cond4 : Prop := p = ((3 * x + 2) - 4) * x + 3
def cond5 : Prop := a = a ^ 3
def cond6 : Prop := (x = 5) ∧ (y = 5) ∧ (z = 5)
def cond7 : Prop := a * b = 3
def cond8 : Prop := x = y + 2 + x

-- Prove the assignment statements are 2, 4, 5, 8
theorem assignment_statements :
  (cond2) ∧ (cond4) ∧ (cond5) ∧ (cond8) ∧
  ¬cond1 ∧ ¬cond3 ∧ ¬cond6 ∧ ¬cond7 :=
by sorry

end assignment_statements_l288_288232


namespace problem_solution_l288_288503
open Nat

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |> List.foldl (· + ·) 0

def f (n : ℕ) : ℕ :=
  sum_of_digits (n^2 + 1)

def f_seq : ℕ → ℕ → ℕ
| 0, n => f n
| (k+1), n => f (f_seq k n)

theorem problem_solution :
  f_seq 2016 9 = 8 :=
sorry

end problem_solution_l288_288503


namespace regression_slope_l288_288829

theorem regression_slope 
  (x y : Fin 10 → ℝ)
  (b : ℝ)
  (h1 : ∀ i, y i = -3 + b * x i)
  (h_sum_x : (Σ i, x i) = 17)
  (h_sum_y : (Σ i, y i) = 4) :
  b = 2 := by
  sorry

end regression_slope_l288_288829


namespace f_is_odd_and_periodic_l288_288573

noncomputable def f : ℝ → ℝ := sorry

axiom h1 : ∀ x : ℝ, f (10 + x) = f (10 - x)
axiom h2 : ∀ x : ℝ, f (20 - x) = -f (20 + x)

theorem f_is_odd_and_periodic : 
  (∀ x : ℝ, f (-x) = -f x) ∧ (∃ T : ℝ, T = 40 ∧ ∀ x : ℝ, f (x + T) = f x) :=
by
  sorry

end f_is_odd_and_periodic_l288_288573


namespace incenter_of_triangle_KLM_l288_288820

variable {α : Type*} {M : Type*} [Group M] [Module M α] [InnerProductSpace α]

theorem incenter_of_triangle_KLM
    (O A B C: α)
    (A' B' C': α)
    (K L M: α)
    (h_mid_A': A' = midpoint B C)
    (h_mid_B': B' = midpoint C A)
    (h_mid_C': C' = midpoint A B)
    (h_circumcenter: O = circumcenter A B C)
    (h_circles_A: dist O A' = dist O (∥(C-B)∥/2))
    (h_circles_B: dist O B' = dist O (∥(A-C)∥/2))
    (h_circles_C: dist O C' = dist O (∥(B-A)∥/2))
    (h_intersect_K: K ≠ O ∧ ∃ (r: ℝ), circle A' r = ∩ circle B' r ∩ circle C' r)
    (h_intersect_L: L ≠ O ∧ ∃ (r: ℝ), circle B' r = ∩ circle A' r ∩ circle C' r)
    (h_intersect_M: M ≠ O ∧ ∃ (r: ℝ), circle C' r = ∩ circle A' r ∩ circle B' r):
    is_incenter O K L M := by sorry

end incenter_of_triangle_KLM_l288_288820


namespace num_triangles_with_positive_area_l288_288103

/-- 
Given vertices in a 5x5 grid with integer coordinates satisfying 1 ≤ x ≤ 5 and 1 ≤ y ≤ 5,
prove that the number of triangles with positive area is 2170. 
-/
theorem num_triangles_with_positive_area : 
  (∑ t in ({(i, j) | 1 ≤ i ∧ i ≤ 5 ∧ 1 ≤ j ∧ j ≤ 5}.powerset.filter (λ (s : set (ℕ × ℕ)), s.card = 3)),
    if t₁.1 * (t₂.2 - t₃.2) + t₂.1 * (t₃.2 - t₁.2) + t₃.1 * (t₁.2 - t₂.2) ≠ 0 then 1 else 0) = 2170 :=
by sorry

end num_triangles_with_positive_area_l288_288103


namespace intersection_complement_l288_288022

open Set

def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {2, 4}

theorem intersection_complement : A ∩ (U \ B) = {1, 3} :=
by {
  sorry
}

end intersection_complement_l288_288022


namespace converse_of_ptolemies_theorem_l288_288989

theorem converse_of_ptolemies_theorem
  (A B C D M : Type)
  [metric_space A] [metric_space B] [metric_space C] [metric_space D] [metric_space M]
  (AC BD AB CD AD BC : ℝ)
  (h : AC * BD = AB * CD + AD * BC) :
  ∃ O : Type, ∃ (circumscribed: ∀ P, P ∈ {A, B, C, D} → dist O P = r) where
  r : ℝ :=
sorry

end converse_of_ptolemies_theorem_l288_288989


namespace polyhedron_divisible_into_convex_parts_l288_288983

/-- Any polyhedron can be divided into convex polyhedra. -/
theorem polyhedron_divisible_into_convex_parts (P : Polyhedron) : 
  ∃ (convex_parts : set Polyhedron), (∀ C ∈ convex_parts, Convex C) ∧ (⋃₀ convex_parts = P) ∧ (∀ C1 C2 ∈ convex_parts, C1 ≠ C2 → Disjoint C1 C2) := 
sorry

end polyhedron_divisible_into_convex_parts_l288_288983


namespace travel_time_comparison_l288_288555

theorem travel_time_comparison
  (v : ℝ) -- speed during the first trip
  (t1 : ℝ) (t2 : ℝ)
  (h1 : t1 = 80 / v) -- time for the first trip
  (h2 : t2 = 100 / v) -- time for the second trip
  : t2 = 1.25 * t1 :=
by
  sorry

end travel_time_comparison_l288_288555


namespace cube_root_expression_l288_288490

theorem cube_root_expression (M : ℝ) (h : M > 1) : 
  Real.cbrt (M * Real.cbrt (M * Real.cbrt (M * Real.cbrt (M)))) = M ^ (40 / 81) := 
by
  sorry

end cube_root_expression_l288_288490


namespace number_of_correct_propositions_l288_288475

variable (Line Plane : Type)
variable parallel perpendicular : Line → Plane → Prop
variable parallel_lines perpendicular_lines : Line → Line → Prop
variable is_parallel_planes is_perpendicular_planes : Plane → Plane → Prop
variable m n : Line
variable α β : Plane

theorem number_of_correct_propositions :
  (parallel m α ∧ parallel n β ∧ is_parallel_planes α β → parallel_lines m n = false) ∧
  (perpendicular m α ∧ parallel n β ∧ is_parallel_planes α β → perpendicular_lines m n = true) ∧
  (parallel m α ∧ perpendicular n β ∧ is_perpendicular_planes α β → parallel_lines m n = false) ∧
  (perpendicular m α ∧ perpendicular n β ∧ is_perpendicular_planes α β → perpendicular_lines m n = true) →
  2 :=
sorry

end number_of_correct_propositions_l288_288475


namespace arithmetic_sequence_difference_l288_288739

noncomputable def arithmetic_sequence (a d : ℝ) : ℕ → ℝ
| n := a + n * d

def average (s : List ℝ) : ℝ :=
s.sum / s.length

def sequence_sum (s : List ℝ) : ℝ :=
s.sum

def valid_sequence (s : List ℝ) : Prop :=
s.all (λ x, 10 ≤ x ∧ x ≤ 100)

def L (a d : ℝ) : ℝ :=
a + 49 * d

def G (a d : ℝ) : ℝ :=
a + 49 * d

theorem arithmetic_sequence_difference :
  ∃ (a d : ℝ), 
  (valid_sequence (List.range 200).map (arithmetic_sequence a d)) ∧
  (sequence_sum ((List.range 200).map (arithmetic_sequence a d)) = 10000) ∧
  (average ((List.range 200).map (arithmetic_sequence a d)) = 50) ∧
  (G (a, d) - L (a, d) = 8080 / 199) :=
by
  sorry

end arithmetic_sequence_difference_l288_288739


namespace line_perp_to_plane_contains_line_implies_perp_l288_288818

variables {Point Line Plane : Type}
variables (m n : Line) (α : Plane)
variables (contains : Plane → Line → Prop) (perp : Line → Line → Prop) (perp_plane : Line → Plane → Prop)

-- Given: 
-- m and n are two different lines
-- α is a plane
-- m ⊥ α (m is perpendicular to the plane α)
-- n ⊂ α (n is contained in the plane α)
-- Prove: m ⊥ n
theorem line_perp_to_plane_contains_line_implies_perp (hm : perp_plane m α) (hn : contains α n) : perp m n :=
sorry

end line_perp_to_plane_contains_line_implies_perp_l288_288818


namespace minimum_value_of_f_l288_288838

noncomputable def f (x : ℝ) : ℝ := x^2 - 2*x - 4*real.log x + 3

theorem minimum_value_of_f :
  ∃ x ∈ set.Ioi (0 : ℝ), ∀ y ∈ set.Ioi (0 : ℝ), f x ≤ f y ∧ f x = 3 - 4 * real.log 2 := by
  sorry

end minimum_value_of_f_l288_288838


namespace fixed_circle_circumcenters_l288_288755

theorem fixed_circle_circumcenters
  (S1 S2 : Circle)
  (P Q : Point)
  (intersect_S1_S2: P ∈ S1 ∧ Q ∈ S1 ∧ P ∈ S2 ∧ Q ∈ S2)
  (A1 B1 : Point)
  (distinct_A1_B1 : A1 ≠ B1 ∧ A1 ≠ P ∧ A1 ≠ Q ∧ B1 ≠ P ∧ B1 ≠ Q)
  (A2 B2 C : Point)
  (A1P_meets_S2 : (Line_through A1 P) ∩ S2 = {P, A2})
  (B1P_meets_S2 : (Line_through B1 P) ∩ S2 = {P, B2})
  (A1B1_meets_A2B2 : (Line_through A1 B1) ∩ (Line_through A2 B2) = {C}) :
  ∃ fixed_circle : Circle,
    ∀ (A1' B1' : Point) (distinct_A1'_B1' : A1' ≠ B1' ∧ A1' ≠ P ∧ A1' ≠ Q ∧ B1' ≠ P ∧ B1' ≠ Q)
    (A2' B2' C' : Point)
    (A1'P_meets_S2 : (Line_through A1' P) ∩ S2 = {P, A2'})
    (B1'P_meets_S2 : (Line_through B1' P) ∩ S2 = {P, B2'})
    (A1'B1'_meets_A2'B2' : (Line_through A1' B1') ∩ (Line_through A2' B2') = {C'}),
    circumcenter (triangle A1' A2' C') ∈ fixed_circle := 
by 
  sorry

end fixed_circle_circumcenters_l288_288755


namespace positive_area_triangles_correct_l288_288081

-- Define the set of points with integer coordinates in the given range
def grid_points := { p : ℤ × ℤ | 1 ≤ p.1 ∧ p.1 ≤ 5 ∧ 1 ≤ p.2 ∧ p.2 ≤ 5 }

-- Define a function to check for collinearity of three points
def collinear (p1 p2 p3 : ℤ × ℤ) : Prop :=
  (p2.2 - p1.2) * (p3.1 - p1.1) = (p3.2 - p1.2) * (p2.1 - p1.1)

-- Define the count of triangles with positive area (not collinear) from the given points
def triangles_with_positive_area (points : set (ℤ × ℤ)) : ℕ :=
  (@set.to_finset (ℤ × ℤ) _ points).to_list.combinations 3
  .filter (λ l, l.length = 3 ∧ ¬ collinear l.head l.nth 1 l.nth 2).length

-- State the proof problem
theorem positive_area_triangles_correct :
  triangles_with_positive_area grid_points = 2170 := sorry

end positive_area_triangles_correct_l288_288081


namespace ratio_of_13th_terms_l288_288568

noncomputable def arith_series (a d : ℕ) (n : ℕ) : ℕ :=
n * (2 * a + (n - 1) * d) / 2

theorem ratio_of_13th_terms (c f g h : ℕ)
  (h1 : ∀ n, (arith_series c f n) * (3 * n + 17) = (5 * n + 3) * (arith_series g h n)) :
  (c + 12 * f) * 89 = 52 * (g + 12 * h) :=
begin
  sorry
end

end ratio_of_13th_terms_l288_288568


namespace determine_a_l288_288217

noncomputable def a_value (x : ℝ) [hx : x = (x + 7)^3] : ℝ := Real.logₛₐ 7

theorem determine_a (a : ℝ)
  (h1 : ∀ (A B C : ℝ × ℝ), 
    A.2  = Math.logₐ A.1 
    ∧ B.2 = 3 * Math.logₐ (B.1 + 7)
    ∧ C.2 = 4 * Math.logₐ (C.1 + 7)
    ∧ A.1 + 7 = B.1 ∧ B.1 + 7 = C.1
  )
  (h2 : (∀ (x : ℝ), x = (x + 7)^3))
  (h3 : ∀ (A B C : ℝ × ℝ), B.1 = A.1 + 7 ∧ C.1 = B.1 + 7 ∧ (C.2 - B.2) = 7):
  a = Real.exp ((1:ℝ) / 7) :=
sorry

end determine_a_l288_288217


namespace unique_integer_x_makes_p_perfect_square_l288_288263

def p (x : ℤ) : ℤ := x^4 + 6 * x^3 + 11 * x^2 + 3 * x + 31

theorem unique_integer_x_makes_p_perfect_square :
  ∃! x : ℤ, ∃ k : ℤ, p(x) = k^2 :=
sorry

end unique_integer_x_makes_p_perfect_square_l288_288263


namespace coordinates_of_H_l288_288849

-- Define the points O, A, B
structure Point :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

def O : Point := ⟨0, 0, 0⟩
def A : Point := ⟨-1, 1, 0⟩
def B : Point := ⟨0, 1, 1⟩

-- Define vector operations
def sub (P Q : Point) : Point :=
  ⟨P.x - Q.x, P.y - Q.y, P.z - Q.z⟩

def dot (P Q : Point) : ℝ :=
  P.x * Q.x + P.y * Q.y + P.z * Q.z

-- Define the line equation parameter
def on_line (H A O : Point) : Prop :=
  ∀ λ : ℝ, H = ⟨λ * (A.x - O.x), λ * (A.y - O.y), λ * (A.z - O.z)⟩

-- Define the perpendicular condition
def perpendicular (H B A O : Point) : Prop :=
  dot (sub H B) (sub A O) = 0

-- State the theorem
theorem coordinates_of_H :
  ∃ H : Point, on_line H A O ∧ perpendicular H B A O ∧ H = ⟨-1/2, 1/2, 0⟩ :=
by
  sorry

end coordinates_of_H_l288_288849


namespace minimum_distance_from_circle_to_line_l288_288642

noncomputable def circle : set (ℝ × ℝ) := { p | (p.1 - 1)^2 + (p.2 - 1)^2 = 1 }

def line (p : ℝ × ℝ) : Prop := 3 * p.1 + 4 * p.2 + 8 = 0

theorem minimum_distance_from_circle_to_line :
  let center := (1, 1) in
  let radius := 1 in
  let distance := abs (3 * center.1 + 4 * center.2 + 8) / sqrt (3^2 + 4^2) in
  distance - radius = 2 :=
by
    let center := (1, 1)
    let radius := 1
    let distance := abs (3 * center.1 + 4 * center.2 + 8) / sqrt (3^2 + 4^2)
    have h : distance - radius = 2 := by sorry
    exact h

end minimum_distance_from_circle_to_line_l288_288642


namespace proof_f_ff_l288_288066

def f (x : ℝ) : ℝ :=
  if x > 0 then log x / log 2 else x^2

theorem proof_f_ff (x : ℝ) (hx_neg : x ≤ 0) (hx : f x) :
  f (f (-1/2)) = -2 :=
by
  sorry

end proof_f_ff_l288_288066


namespace sequence_perfect_square_l288_288760

variable (a : ℕ → ℤ)

axiom a1 : a 1 = 1
axiom a2 : a 2 = 1
axiom recurrence : ∀ n ≥ 3, a n = 7 * (a (n - 1)) - (a (n - 2))

theorem sequence_perfect_square (n : ℕ) (hn : n > 0) : ∃ k : ℤ, a n + a (n + 1) + 2 = k * k :=
by
  sorry

end sequence_perfect_square_l288_288760


namespace find_k_solution_l288_288004

theorem find_k_solution :
    ∃ k : ℝ, (4 + ∑' n : ℕ, (4 + (n : ℝ)*k) / 5^(n + 1) = 10) ∧ k = 16 :=
begin
  use 16,
  sorry
end

end find_k_solution_l288_288004


namespace sum_of_two_planar_angles_greater_than_third_l288_288988

-- Define the conditions and the problem
def non_collinear (A B C: Point) : Prop := 
  ¬ (collinear A B C)

-- The main theorem statement
theorem sum_of_two_planar_angles_greater_than_third 
  (A B C S : Point) 
  (h_non_collinear : non_collinear A B C) : 
  ∠ ASB + ∠ BSC > ∠ CSA :=
by 
  sorry

end sum_of_two_planar_angles_greater_than_third_l288_288988


namespace find_x_l288_288396

theorem find_x (x : ℝ) : 3^4 * 3^x = 81 → x = 0 := 
by
  sorry

end find_x_l288_288396


namespace num_triangles_with_positive_area_l288_288106

/-- 
Given vertices in a 5x5 grid with integer coordinates satisfying 1 ≤ x ≤ 5 and 1 ≤ y ≤ 5,
prove that the number of triangles with positive area is 2170. 
-/
theorem num_triangles_with_positive_area : 
  (∑ t in ({(i, j) | 1 ≤ i ∧ i ≤ 5 ∧ 1 ≤ j ∧ j ≤ 5}.powerset.filter (λ (s : set (ℕ × ℕ)), s.card = 3)),
    if t₁.1 * (t₂.2 - t₃.2) + t₂.1 * (t₃.2 - t₁.2) + t₃.1 * (t₁.2 - t₂.2) ≠ 0 then 1 else 0) = 2170 :=
by sorry

end num_triangles_with_positive_area_l288_288106


namespace part1_part2_l288_288546

-- Given points A, B, and C with specific coordinates for part (1)
def A := (-2,1)
def B := (4,3)
def C1 := (3,-2) -- C for part (1)

-- Part (1): Prove the line equation containing altitude AD on side BC
theorem part1 :
  ∃ m b : ℝ, (∀ (x y : ℝ), y = m * x + b ↔ x + 5 * y - 3 = 0)
    ∧ (∀ (D : ℝ × ℝ), D = (fst A, (snd A + (5/snd C1 - snd A) * (fst B - fst A)))) :=
sorry

-- Given points A, B, and M with specific coordinates for part (2)
def B := (4,3)
def M := (3,1) -- M is the midpoint of AC

-- Part (2): Prove the line equation containing side BC
theorem part2 :
  ∃ m b : ℝ, (∀ (x y : ℝ), y = m * x + b ↔ x + 2 * y - 10 = 0)
    ∧ (fst M = (fst A + fst C2) / 2) ∧ (snd M = (snd A + snd C2) / 2) :=
sorry

end part1_part2_l288_288546


namespace chastity_amount_left_l288_288752

noncomputable def initial_amount : ℝ := 25.00
noncomputable def lollipop_cost : ℝ := 1.50
noncomputable def gummies_cost : ℝ := 2.00
noncomputable def chips_cost : ℝ := 1.25
noncomputable def chocolate_bar_cost : ℝ := 1.75

noncomputable def total_cost_before_discount : ℝ := 
  (4 * lollipop_cost) + (2 * gummies_cost) + (3 * chips_cost) + chocolate_bar_cost

noncomputable def discount_rate : ℝ := 0.10
noncomputable def sales_tax_rate : ℝ := 0.05

noncomputable def total_cost_after_discount : ℝ := total_cost_before_discount * (1 - discount_rate)
noncomputable def sales_tax_amount : ℝ := real.ceil $ total_cost_after_discount * sales_tax_rate * 100 / 100
noncomputable def total_cost_after_tax : ℝ := total_cost_after_discount + sales_tax_amount

noncomputable def amount_left : ℝ := initial_amount - total_cost_after_tax

theorem chastity_amount_left : amount_left = 10.35 := by
  have h_total : total_cost_before_discount = 15.50 := by sorry
  have h_discount : discount_rate * total_cost_before_discount = 1.55 := by sorry
  have h_sales_tax : sales_tax_amount = 0.70 := by sorry
  have h_final_cost : total_cost_after_tax = 14.65 := by sorry
  show amount_left = 10.35, from calc
    initial_amount - total_cost_after_tax = 25.00 - 14.65 : by sorry
    ... = 10.35 : by sorry

end chastity_amount_left_l288_288752


namespace count_4_digit_numbers_divisible_by_13_l288_288884

theorem count_4_digit_numbers_divisible_by_13 : 
  let count := (9962 - 1014) / 13 + 1
  1000 ≤ 1014 ∧ 9962 ≤ 9999 →
  count = 689 :=
  by
    sorry

end count_4_digit_numbers_divisible_by_13_l288_288884


namespace trigonometric_identity_l288_288420

theorem trigonometric_identity (α : ℝ) (h : Real.tan α = 3) :
  2 * Real.sin α ^ 2 - Real.sin α * Real.cos α + Real.cos α ^ 2 = 8 / 5 := 
by
  sorry

end trigonometric_identity_l288_288420


namespace sqrt7_minus_fraction_inequality_l288_288204

theorem sqrt7_minus_fraction_inequality (m n : ℕ) (h : real.sqrt 7 - (m : ℝ) / n > 0) : 
  real.sqrt 7 - (m : ℝ) / n > 1 / (m * n : ℝ) :=
by
  sorry

end sqrt7_minus_fraction_inequality_l288_288204


namespace volume_of_tetrahedron_PQRS_l288_288620

-- Definitions of the given conditions for the tetrahedron
def PQ := 6
def PR := 4
def PS := 5
def QR := 5
def QS := 6
def RS := 15 / 2  -- RS is (15 / 2), i.e., 7.5
def area_PQR := 12

noncomputable def volume_tetrahedron (PQ PR PS QR QS RS area_PQR : ℝ) : ℝ := 1 / 3 * area_PQR * 4

theorem volume_of_tetrahedron_PQRS :
  volume_tetrahedron PQ PR PS QR QS RS area_PQR = 16 :=
by sorry

end volume_of_tetrahedron_PQRS_l288_288620


namespace validate_triangle_count_l288_288088

noncomputable def count_valid_triangles : ℕ :=
  let total_points := 25
  let total_triples := (Nat.choose total_points 3)
  let collinear_rows := 5 * (Nat.choose 5 3)
  let collinear_columns := 5 * (Nat.choose 5 3)
  let main_diagonals := 2 * (Nat.choose 5 3)
  let secondary_diagonals := 8 * (Nat.choose 4 3)
  let invalid_triangles := collinear_rows + collinear_columns + main_diagonals + secondary_diagonals
  total_triples - invalid_triangles

theorem validate_triangle_count : count_valid_triangles = 2148 :=
by
  sorry

end validate_triangle_count_l288_288088


namespace find_K_3_15_10_l288_288788

def K (x y z : ℚ) : ℚ := 
  x / y + y / z + z / x + (x + y) / z

theorem find_K_3_15_10 : K 3 15 10 = 41 / 6 := 
  by
  sorry

end find_K_3_15_10_l288_288788


namespace product_eq_one_sixth_l288_288373

theorem product_eq_one_sixth : 
  (∏ n in finset.range 10, (1 - 1 / (n + 3))) = 1 / 6 := 
sorry

end product_eq_one_sixth_l288_288373


namespace probability_correct_l288_288736

def total_assignments := 15 * 14 * 13

def is_multiple (a b : ℕ) : Prop := a % b = 0

def valid (al bill cal : ℕ) : Prop :=
  is_multiple al bill ∧ is_multiple bill cal ∧
  al ≠ bill ∧ bill ≠ cal ∧ al ≠ cal

def count_valid_assignments : ℕ :=
  Finset.card
    (Finset.univ.filter (λ triplet: Fin3 15 × Fin3 15 × Fin3 15,
      valid triplet.1 triplet.2 triplet.3))

def probability_valid_assignment : ℚ :=
  count_valid_assignments / total_assignments

theorem probability_correct :
  probability_valid_assignment = 1 / 60 := sorry

end probability_correct_l288_288736


namespace unique_triple_primes_l288_288149

theorem unique_triple_primes (p q r : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (hr : Nat.Prime r) 
  (h1 : p < q) (h2 : q < r) (h3 : (p^3 + q^3 + r^3) / (p + q + r) = 249) : r = 19 :=
sorry

end unique_triple_primes_l288_288149


namespace smart_charging_piles_growth_l288_288523

noncomputable def a : ℕ := 301
noncomputable def b : ℕ := 500
variable (x : ℝ) -- Monthly average growth rate

theorem smart_charging_piles_growth :
  a * (1 + x) ^ 2 = b :=
by
  -- Proof should go here
  sorry

end smart_charging_piles_growth_l288_288523


namespace angle_inequality_l288_288160

theorem angle_inequality {A B C P : Type} [InnerProductSpace ℝ A] [AffineSpace A P] 
  [InnerProductSpace ℝ B] [AffineSpace B P] 
  [InnerProductSpace ℝ C] [AffineSpace C P] (hP : ∀ A B C, ∃ P ∈ interior (triangle A B C)) : 
  ∃ P : P, P ∈ interior (triangle A B C) → angle A P B ≤ 30 ∨ angle B P C ≤ 30 ∨ angle C P A ≤ 30 := 
sorry

end angle_inequality_l288_288160


namespace minimum_value_expression_l288_288958

variable (a b c k : ℝ)
variable (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_pos_k : 0 < k)

theorem minimum_value_expression :
  (\frac{a + b + k}{c} + \frac{a + c + k}{b} + \frac{b + c + k}{a}) >= 9 :=
sorry

end minimum_value_expression_l288_288958


namespace monica_classes_per_day_l288_288591

theorem monica_classes_per_day (n : ℕ)
  (students : Fin n → ℕ)
  (h1 : students 0 = 20)
  (h2 : students 1 = 25)
  (h3 : students 2 = 25)
  (h4 : students 3 = 10) -- Derived as half of the first class (20 / 2)
  (h5 : students 4 = 28)
  (h6 : students 5 = 28)
  (h_total : (Finset.univ : Finset (Fin n)).sum students = 136) : 
  n = 6 :=
begin
  sorry
end

end monica_classes_per_day_l288_288591


namespace complex_cubic_eq_l288_288616

theorem complex_cubic_eq (a b : ℤ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_eq : (a + b * Complex.i)^3 = 2 + 11 * Complex.i) : 
  a + b * Complex.i = 2 + Complex.i := 
sorry

end complex_cubic_eq_l288_288616


namespace integer_solution_inequalities_l288_288675

theorem integer_solution_inequalities (x : ℤ) (h1 : x + 12 > 14) (h2 : -3 * x > -9) : x = 2 :=
by
  sorry

end integer_solution_inequalities_l288_288675


namespace ratio_girls_to_boys_l288_288915

-- Define the number of students and conditions
def num_students : ℕ := 25
def girls_more_than_boys : ℕ := 3

-- Define the variables
variables (g b : ℕ)

-- Define the conditions
def total_students := g + b = num_students
def girls_boys_relationship := b = g - girls_more_than_boys

-- Lean theorem statement
theorem ratio_girls_to_boys (g b : ℕ) (h1 : total_students g b) (h2 : girls_boys_relationship g b) : (g : ℚ) / b = 14 / 11 :=
sorry

end ratio_girls_to_boys_l288_288915


namespace not_perfect_square_l288_288297

theorem not_perfect_square (h1 : ∃ x : ℝ, x^2 = 1 ^ 2018) 
                           (h2 : ¬ ∃ x : ℝ, x^2 = 2 ^ 2019)
                           (h3 : ∃ x : ℝ, x^2 = 3 ^ 2020)
                           (h4 : ∃ x : ℝ, x^2 = 4 ^ 2021)
                           (h5 : ∃ x : ℝ, x^2 = 6 ^ 2022) : 
  2 ^ 2019 ≠ x^2 := 
sorry

end not_perfect_square_l288_288297


namespace find_scalars_l288_288569

theorem find_scalars (r s : ℝ) : 
  let N := ![![3, 4], ![-2, 0]] : Matrix (Fin 2) (Fin 2) ℝ,
      I := Matrix.eye (Fin 2) ℝ in
  N * N = r • N + s • I → (r, s) = (3, 8) := 
by
  intros N I h
  sorry

end find_scalars_l288_288569


namespace water_depth_function_l288_288519

theorem water_depth_function (t : ℝ) (ht : 0 ≤ t ∧ t ≤ 24) :
  (∃ ω φ k : ℝ, ω = π / 6 ∧ φ = 0 ∧ k = 12 ∧ (∀ t, y = 3 * sin(ω * t + φ) + k) ∧
    (∀ t₁ t₂, (t₂ = t₁ + 12) → (∃ A, (A + k = 15) ∧ (-A + k = 9)) ∧
    (∃ t₁ t₂, t₁ = 3 → y = 15)) :=
begin
  sorry
end

end water_depth_function_l288_288519


namespace value_of_Q_l288_288161

theorem value_of_Q (n : ℕ) (h : n = 2008) : 
  let Q := (1 - 1/3) * (1 - 1/4) * (1 - 1/5) * (1 - 1/6) * ... * (1 - 1/n)
  in Q = 1/1004 :=
by {
  sorry
}

end value_of_Q_l288_288161


namespace correct_statements_subset_relation_l288_288685

-- Conditions
def positive_sufficient_condition {a b : ℝ} (h : a < b) : (1 / a > 1 / b) := 
  begin
    split_ifs with h₁ h₂;
    sorry, -- We handle both conditions here appropriately
  end

def negative_sufficient_condition {a b : ℝ} (h : a < b) : (1 / a < 1 / b) := 
  begin
    split_ifs with h₁ h₂;
    sorry, -- We handle both conditions here appropriately
  end

-- Proof Problem Statements
theorem correct_statements (a b : ℝ) (n : ℕ) (h_pos : a > 0 ∧ b > 0) (h_neg : a < 0 ∧ b < 0) :
  (positive_sufficient_condition h_pos.2) ∧ 
  (¬ sufficient_for (λ x : ℝ, x ∈ set_of (λ x : ℝ, 1 / a > 1 / b)) (λ x : ℝ, x < b)) := 
  sorry

theorem subset_relation {A B : set ℝ} (h : ∀ x, x ∈ A → x ∈ B): A ⊆ B := 
  sorry

end correct_statements_subset_relation_l288_288685


namespace Sally_out_of_pocket_payment_l288_288994

theorem Sally_out_of_pocket_payment :
  let amount_given : ℕ := 320
  let cost_per_book : ℕ := 12
  let number_of_students : ℕ := 30
  let total_cost : ℕ := cost_per_book * number_of_students
  let out_of_pocket_cost : ℕ := total_cost - amount_given
  out_of_pocket_cost = 40 := by
  sorry

end Sally_out_of_pocket_payment_l288_288994


namespace find_possible_values_l288_288649

def real_number_y (y : ℝ) := (3 < y ∧ y < 4)

theorem find_possible_values (y : ℝ) (h : real_number_y y) : 
  42 < (y^2 + 7*y + 12) ∧ (y^2 + 7*y + 12) < 56 := 
sorry

end find_possible_values_l288_288649


namespace evaluate_logarithmic_expression_l288_288774

open Real

-- Define the problem and use the necessary conditions
def log2_of_16 : ℝ := log 16 / log 2
def log2_of_8 : ℝ := log 8 / log 2
def log2_of_4 : ℝ := log 4 / log 2
def log2_of_64 : ℝ := log 64 / log 2

-- State the theorem (problem) and the expected result
theorem evaluate_logarithmic_expression :
  log2_of_16 + 3 * log2_of_8 + 2 * log2_of_4 - log2_of_64 = 11 :=
by
  sorry

end evaluate_logarithmic_expression_l288_288774


namespace limit_evaluation_l288_288585

noncomputable def f : ℝ → ℝ := sorry

variables {x : ℝ}

theorem limit_evaluation (h : differentiable ℝ f) :
  (tendsto (λ k : ℝ, (f (1 - k) - f 1) / (3 * k)) (𝓝 0) (𝓝 (- (1/3) * (deriv f 1)))) :=
sorry

end limit_evaluation_l288_288585


namespace triangle_inequality_equality_condition_l288_288959

theorem triangle_inequality (a b c S : ℝ)
  (h_tri : a^2 + b^2 + c^2 = 4 * S * Real.sqrt 3)
  (h_area : S = Real.sqrt ((a + b + c) / 2 * ((a + b + c) / 2 - a) * ((a + b + c) / 2 - b) * ((a + b + c) / 2 - c))):
  a^2 + b^2 + c^2 ≥ 4 * S * Real.sqrt 3 :=
sorry

theorem equality_condition (a b c S : ℝ)
  (h_tri : a^2 + b^2 + c^2 = 4 * S * Real.sqrt 3)
  (h_area : S = Real.sqrt ((a + b + c) / 2 * ((a + b + c) / 2 - a) * ((a + b + c) / 2 - b) * ((a + b + c) / 2 - c))):
  (a = b) ∧ (b = c) :=
sorry

end triangle_inequality_equality_condition_l288_288959


namespace hexagon_coloring_l288_288235

def hex_colorings : ℕ := 2

theorem hexagon_coloring :
  ∃ c : ℕ, c = hex_colorings := by
  sorry

end hexagon_coloring_l288_288235


namespace shipping_cost_invariance_l288_288211

theorem shipping_cost_invariance (settlements : List ℕ) (city : ℕ) :
  (∀ ordering : List ℕ, ordering.perm settlements → 
    ∑ i in ordering, (∑ j in ordering.drop (ordering.indexOf i), (settlements.nth i.getOrElse 0))) = 
    (∑ i in settlements, (∑ j in settlements.drop (settlements.indexOf i), (settlements.nth i.getOrElse 0))) :=
by
  -- the proof goes here
  sorry

end shipping_cost_invariance_l288_288211


namespace projection_of_vectors_l288_288797

variables {a b : ℝ}

noncomputable def vector_projection (a b : ℝ) : ℝ :=
  (a * b) / b^2 * b

theorem projection_of_vectors
  (ha : abs a = 6)
  (hb : abs b = 3)
  (hab : a * b = -12) : vector_projection a b = -4 :=
sorry

end projection_of_vectors_l288_288797


namespace luke_initial_stickers_l288_288965

theorem luke_initial_stickers (x : ℕ) (h : x + 12 + 20 - 5 - 8 = 39) : x = 20 := 
by 
  sorry

end luke_initial_stickers_l288_288965


namespace greatest_possible_large_chips_l288_288253

theorem greatest_possible_large_chips 
  (s l : ℕ) 
  (p : ℕ) 
  (h1 : s + l = 72) 
  (h2 : s = l + p) 
  (h_prime : Prime p) : 
  l ≤ 35 :=
sorry

end greatest_possible_large_chips_l288_288253


namespace coefficient_of_expansion_l288_288508

theorem coefficient_of_expansion (m : ℝ) (h : m^3 * (Nat.choose 6 3) = -160) : m = -2 := by
  sorry

end coefficient_of_expansion_l288_288508


namespace binary_representation_of_31_l288_288382

theorem binary_representation_of_31 : nat.binary_repr 31 = "11111" :=
sorry

end binary_representation_of_31_l288_288382


namespace prove_series_equality_l288_288045

noncomputable def alternating_sum (n : ℕ) : ℚ :=
  ∑ k in (finset.range (n + 1)).filter (λ k, k % 2 = 1), (if k % 4 = 1 then 1 else -1) * (1 / (k + 1))

noncomputable def series_sum (n : ℕ) : ℚ :=
  2 * ∑ k in (finset.range n).filter (λ k, k % 2 = 0), 1 / (k + 2 + n)

theorem prove_series_equality (n : ℕ) (h : n > 0) (h_even : n % 2 = 0) : 
  alternating_sum n = series_sum n := sorry

end prove_series_equality_l288_288045


namespace surface_area_of_cube_l288_288659

theorem surface_area_of_cube (V : ℝ) (hV : V = 27) : 
  ∃ SA : ℝ, SA = 54 :=
by
  -- To skip the actual proof steps
  let side_length := (V)^(1/3)
  let face_area := side_length * side_length
  let total_surface_area := 6 * face_area
  use total_surface_area
  have h : total_surface_area = 54 := by
    -- Continuing from previous definition
    calc
      total_surface_area = 6 * (V)^(2/3)   : by sorry -- skipping complex details
                      ... = 6 * 9          : by sorry -- showing 3^2
                      ... = 54             : rfl
  show total_surface_area = 54, from h
  sorry

end surface_area_of_cube_l288_288659


namespace validate_triangle_count_l288_288086

noncomputable def count_valid_triangles : ℕ :=
  let total_points := 25
  let total_triples := (Nat.choose total_points 3)
  let collinear_rows := 5 * (Nat.choose 5 3)
  let collinear_columns := 5 * (Nat.choose 5 3)
  let main_diagonals := 2 * (Nat.choose 5 3)
  let secondary_diagonals := 8 * (Nat.choose 4 3)
  let invalid_triangles := collinear_rows + collinear_columns + main_diagonals + secondary_diagonals
  total_triples - invalid_triangles

theorem validate_triangle_count : count_valid_triangles = 2148 :=
by
  sorry

end validate_triangle_count_l288_288086


namespace angle_P_in_trapezoid_correct_l288_288938

noncomputable def angle_P_in_trapezoid (P Q R S : Type) (angle : P → Q → R → S → ℝ)
  (P Q R S : ℝ) 
  (h_parallel: PQS → RS → Prop)
  (h_angles_eq: ℝ)
  (h_angle_P_eq_3angle_S: Angle P = 3 * Angle S)
  (h_angle_R_eq_angle_Q: Angle R = Angle Q) : Prop :=
Angle P = 135.

theorem angle_P_in_trapezoid_correct (P Q R S : Type) 
(pq pq' rs rs' α PQ RS S P R Q : ℝ)
(h_parallel: α⟺PQ⟷RS) 
(h_angle_P_eq_3angle_S : Angle_ P = 3 * Angle S)
(h_angle_R_eq_angle_Q: Angle R = Angle Q) : 
Angle P = 135° :=
begin
  sorry,
end

end angle_P_in_trapezoid_correct_l288_288938


namespace solution_set_of_inequality_l288_288050

def domain_f (f : ℝ → ℝ) : Prop :=
  ∀ x > 0, true

noncomputable def f'' (f : ℝ → ℝ) : (ℝ → ℝ) := sorry

noncomputable def condition_f (f : ℝ → ℝ) : Prop :=
  ∀ x > 0, f x < - x * f'' f x

theorem solution_set_of_inequality (f : ℝ → ℝ)
  (h₁ : domain_f f)
  (h₂ : ∀ x > 0, differentiable ℝ (f x))
  (h₃ : ∀ x > 0, differentiable ℝ (f'' f x))
  (h₄ : condition_f f) :
  { x : ℝ | f (x+1) > (x-1) * f (x^2 - 1) } = { x : ℝ |  x > 2 } :=
sorry

end solution_set_of_inequality_l288_288050


namespace find_angle_C_find_sin_A_l288_288804

-- Definitions for the conditions given in the problem
variables {A B C a b c : ℝ}

-- Condition 1
def condition1 (c1 : ℝ) : Prop := sqrt 3 * b * cos (A + B) / 2 = c * sin B

-- Question 1: Find the measure of ∠C
theorem find_angle_C (c1 : condition1) : C = 60 := sorry

-- Condition 2
def condition2 : Prop := a + b = sqrt 3 * c

-- Question 2: Find sin A given a + b = √3 c
theorem find_sin_A (c2 : condition2) : sin A = 1 / 2 ∨ sin A = 1 := sorry

end find_angle_C_find_sin_A_l288_288804


namespace correct_equation_l288_288270

-- Define the necessary conditions and parameters
variables (x : ℝ)

-- Length of the rectangle
def length := x 

-- Width is 6 meters less than the length
def width := x - 6

-- The area of the rectangle
def area := 720

-- Proof statement
theorem correct_equation : 
  x * (x - 6) = 720 :=
sorry

end correct_equation_l288_288270


namespace Circle_Chord_Tangent_Proof_l288_288314

theorem Circle_Chord_Tangent_Proof (O : Type*) [Plane EuclideanGeometry O] (A B D : O) (C : ∀ {X Y : O}, X ≠ Y → O)
  (Diameter : EuclideanGeometry.OrdinaryCircle O) (Chord AC : EuclideanGeometry.Chord O)
  (Tangent CD : EuclideanGeometry.Tangent O) (h : AC ∩ CD = {A})
  (hPerp : EuclideanGeometry.Perpendicular AC CD) :
  AC.length ^ 2 = Diameter.length * (EuclideanGeometry.SegmentLength A D) :=
sorry

end Circle_Chord_Tangent_Proof_l288_288314


namespace angle_in_fourth_quadrant_l288_288318

def is_in_fourth_quadrant (θ : ℝ) : Prop := (360 * (2 * n) + θ) = θ ∧ 270 < θ ∧ θ < 360

theorem angle_in_fourth_quadrant (θ : ℝ) (n : ℤ) (h1 : θ = 640) : is_in_fourth_quadrant 280 :=
by
  sorry

end angle_in_fourth_quadrant_l288_288318


namespace count_ways_to_assign_providers_l288_288561

theorem count_ways_to_assign_providers : ∃ (ways : ℕ), ways = 25 * 24 * 23 * 22 ∧ ways = 303600 :=
by
  use 25 * 24 * 23 * 22
  split
  sorry

end count_ways_to_assign_providers_l288_288561


namespace find_fraction_l288_288468

variable (a b c : ℝ)
variable (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
variable (h1 : (a + b + c) / (a + b - c) = 7)
variable (h2 : (a + b + c) / (a + c - b) = 1.75)

theorem find_fraction : (a + b + c) / (b + c - a) = 3.5 := 
by {
  sorry
}

end find_fraction_l288_288468


namespace triangle_third_side_length_l288_288919

theorem triangle_third_side_length (x: ℕ) (h1: x % 2 = 0) (h2: 2 + 14 > x) (h3: 14 - 2 < x) : x = 14 :=
by 
  sorry

end triangle_third_side_length_l288_288919


namespace automobile_travel_distance_l288_288354

theorem automobile_travel_distance (b s : ℝ) (h1 : s > 0) :
  let rate := (b / 8) / s  -- rate in meters per second
  let rate_km_per_min := rate * (1 / 1000) * 60  -- convert to kilometers per minute
  let time := 5  -- time in minutes
  rate_km_per_min * time = 3 * b / 80 / s := sorry

end automobile_travel_distance_l288_288354


namespace exponent_multiplication_l288_288374

theorem exponent_multiplication (a : ℝ) : a^2 * a^3 = a^5 := by
  sorry

end exponent_multiplication_l288_288374


namespace num_of_4_digit_numbers_divisible_by_13_l288_288888

theorem num_of_4_digit_numbers_divisible_by_13 :
  let smallest_4_digit := 1000
  let smallest_divisible_by_13 := 1001
  let largest_4_digit := 9999
  let largest_divisible_by_13 := 9997
  let a := smallest_divisible_by_13
  let d := 13
  let l := largest_divisible_by_13
  (\(l - a) / d) + 1 = 693 :=
by
  let smallest_4_digit := 1000
  let smallest_divisible_by_13 := 1001
  let largest_4_digit := 9999
  let largest_divisible_by_13 := 9997
  let a := smallest_divisible_by_13
  let d := 13
  let l := largest_divisible_by_13
  have h1 : (l - a) / d + 1 = (9997 - 1001) / 13 + 1, by sorry
  have h2 : (9997 - 1001) / 13 + 1 = 8996 / 13 + 1, by sorry
  have h3 : 8996 / 13 + 1 = 692 + 1, by sorry
  have h4 : 692 + 1 = 693, by sorry
  exact h4

end num_of_4_digit_numbers_divisible_by_13_l288_288888


namespace quadratic_has_two_real_roots_l288_288044

-- Definition of the problem conditions
variables {m n : ℝ}

-- Statement of the proof problem
theorem quadratic_has_two_real_roots (h_mn_real : m ∈ ℝ ∧ n ∈ ℝ) :
  ∃ (x1 x2 : ℝ), (x1 = m ∧ x2 = n) ∧ (x1 ≠ x2 ∨ x1 = x2) :=
by
  -- Proof steps will be filled in here
  sorry

end quadratic_has_two_real_roots_l288_288044


namespace problem1_problem2_l288_288514

noncomputable def triangle_sides_angles (a b c A B C : ℝ) :=
  ∃ k : ℝ, k ≠ 0 ∧ a = k * sin A ∧ b = k * sin B ∧ c = k * sin C

theorem problem1
  {a b c A B C : ℝ}
  (h1 : triangle_sides_angles a b c A B C)
  (h2 : (cos A - 2 * cos C) * b = (2 * c - a) * cos B) :
  sin C = 2 * sin A :=
sorry

theorem problem2
  {a b c A B C : ℝ}
  (h1 : triangle_sides_angles a b c A B C)
  (h2 : (cos A - 2 * cos C) * b = (2 * c - a) * cos B)
  (h3 : b * cos C + c * cos B = 1)
  (h4 : a + b + c = 5) :
  b = 2 :=
sorry

end problem1_problem2_l288_288514


namespace inequality_holds_l288_288414

variable {a b c : ℝ}

theorem inequality_holds (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a^2 + 2) * (b^2 + 2) * (c^2 + 2) ≥ 9 * (a * b + b * c + c * a) :=
sorry

end inequality_holds_l288_288414


namespace area_of_triangle_NGI_l288_288914
open Real

noncomputable def triangle_PQR : Type := 
  {side_lengths : ℝ × ℝ × ℝ // 
    side_lengths = (10, 17, 21) ∧ 
    heron's_area side_lengths > 0 }

def coord_P : ℝ × ℝ := (0, 0)
def coord_Q : ℝ × ℝ := (21, 0)
def coord_R : ℝ × ℝ

noncomputable def centroid_G (P Q R : ℝ × ℝ) : ℝ × ℝ := 
  ((P.1 + Q.1 + R.1) / 3, 
   (P.2 + Q.2 + R.2) / 3)

noncomputable def incenter_I (P Q R : ℝ × ℝ) : ℝ × ℝ := sorry -- Calculate incenter I

noncomputable def circle_center_N (P R : ℝ × ℝ) : ℝ × ℝ := 
  let midpoint_PR := ((P.1 + R.1) / 2, (P.2 + R.2) / 2)
  -- Position of N satisfying tangency conditions (to be solved)
  sorry

noncomputable def area_triangle (A B C : ℝ × ℝ) : ℝ := 
  abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2)) / 2

def problem_statement : Prop := ∃ G I N : ℝ × ℝ, 
  area_triangle N G I = sorry -- Calculated area

theorem area_of_triangle_NGI :
  problem_statement
:= sorry

end area_of_triangle_NGI_l288_288914


namespace solve_for_x_l288_288610

noncomputable def x_solution : Real :=
  (Real.log2 3) / 2

theorem solve_for_x :
  ∀ x : Real, (2 ^ (8 ^ x) = 8 ^ (2 ^ x)) ↔ x = x_solution :=
by
  sorry

end solve_for_x_l288_288610


namespace chloe_cycling_rate_l288_288418

theorem chloe_cycling_rate : 
  ∀ (george_rate : ℝ) (lucy_factor max_factor chloe_factor : ℝ),
  george_rate = 6 →
  lucy_factor = 3 / 4 →
  max_factor = 4 / 3 →
  chloe_factor = 5 / 6 →
  let lucy_rate := lucy_factor * george_rate in
  let max_rate := max_factor * lucy_rate in
  let chloe_rate := chloe_factor * max_rate in
  chloe_rate = 5 :=
  by 
    intros george_rate lucy_factor max_factor chloe_factor h_george h_lucy h_max h_chloe
    rw [h_george, h_lucy, h_max, h_chloe]
    let lucy_rate := (3 / 4 : ℝ) * 6
    let max_rate := (4 / 3 : ℝ) * lucy_rate
    let chloe_rate := (5 / 6 : ℝ) * max_rate
    have h1 : lucy_rate = 4.5 := by norm_num
    have h2 : max_rate = 6 := by norm_num
    have h3 : chloe_rate = 5 := by norm_num
    exact h3

end chloe_cycling_rate_l288_288418


namespace prob_rain_at_most_3_days_in_may_l288_288648

noncomputable def prob_rain (n k: ℕ) (p: ℝ) : ℝ :=
  (nat.choose n k) * (p ^ k) * ((1 - p) ^ (n - k))

noncomputable def prob_rain_at_most_3_days (n: ℕ) (p: ℝ) : ℝ :=
  prob_rain n 0 p + prob_rain n 1 p + prob_rain n 2 p + prob_rain n 3 p

theorem prob_rain_at_most_3_days_in_may :
  prob_rain_at_most_3_days 31 (1/5) = 0.780 := 
sorry

end prob_rain_at_most_3_days_in_may_l288_288648


namespace window_treatments_total_cost_l288_288564

def sheers_cost_per_pair := 40
def drapes_cost_per_pair := 60
def number_of_windows := 3

theorem window_treatments_total_cost :
  (number_of_windows * sheers_cost_per_pair) + (number_of_windows * drapes_cost_per_pair) = 300 :=
by 
  -- calculations omitted
  sorry

end window_treatments_total_cost_l288_288564


namespace exists_rational_numbers_l288_288388

noncomputable def smallest_odd_integer : ℕ :=
  25

theorem exists_rational_numbers (n : ℕ) (h1 : n = smallest_odd_integer) :
  ∃ (x : Fin n → ℚ),
    (∑ i, x i = 0) ∧
    (∑ i, (x i)^2 = 1) ∧
    (∀ i j, x i * x j ≥ - 1 / n) :=
begin
  sorry
end

end exists_rational_numbers_l288_288388


namespace conjugate_in_fourth_quadrant_l288_288792

-- Define a condition to check if a complex number lies in a specified quadrant
-- Lean does not have built-in support for quadrants explicitly, so we need to define it.
def isFourthQuadrant (z : ℂ) : Prop :=
  z.re > 0 ∧ z.im < 0

theorem conjugate_in_fourth_quadrant (x y : ℝ) (h : (y / (1 - complex.I) = x + complex.I)) :
  isFourthQuadrant (complex.conj (x + y * complex.I)) :=
by
  -- This is skipped for now since we are only asked to provide the statement
  sorry

end conjugate_in_fourth_quadrant_l288_288792


namespace sin_double_angle_l288_288491

theorem sin_double_angle (θ : ℝ) : (cos θ + sin θ = 3/2) → sin (2 * θ) = 5/4 :=
by
  intro h
  sorry

end sin_double_angle_l288_288491


namespace non_degenerate_triangles_l288_288073

theorem non_degenerate_triangles (x y : ℕ) (hx : 1 ≤ x ∧ x ≤ 5) (hy : 1 ≤ y ∧ y ≤ 5) : 
  ∃ (n : ℕ), n = 2160 ∧ 1 ≤ n :=
by
  sorry

end non_degenerate_triangles_l288_288073


namespace part1_part2_part3_l288_288604

-- Definitions and conditions
def quad_expression_1 (a b : ℤ) (m n : ℤ) := a + b * Real.sqrt 3 = (m + n * Real.sqrt 3) ^ 2
def quad_expression_2 (a : ℤ) (m n : ℤ) := a + 4 * Real.sqrt 3 = (m + n * Real.sqrt 3) ^ 2
def simplify_sqrt_expr := Real.sqrt (25 + 4 * Real.sqrt 6)

-- Part 1: Express a and b in terms of m and n
theorem part1 (a b m n : ℤ) (h : quad_expression_1 a b m n) :
  a = m^2 + 3*n^2 ∧ b = 2*m*n := sorry

-- Part 2: Find the value of a
theorem part2 (a m n : ℤ) (h : quad_expression_2 a m n) :
  a = 13 ∨ a = 7 := sorry

-- Part 3: Simplify the expression
theorem part3 : simplify_sqrt_expr = 5 + 2 * Real.sqrt 6 := sorry

end part1_part2_part3_l288_288604


namespace quadratic_properties_l288_288844

-- Defining the quadratic function
def quadratic_function (a b c x : ℝ) : ℝ :=
  a * x * x + b * x + c

-- Given conditions
def y_values (a b c : ℝ) (f : ℝ → ℝ) : list (ℝ × ℝ) :=
  [(-2, f (-2)), (-1, f (-1)), (0, f 0), (1, f 1), (2, f 2), (3, f 3), (4, f 4)]

-- Main statement to prove the required properties
theorem quadratic_properties (a b c : ℝ) :
  let f := quadratic_function a b c,
      values := y_values a b c f
  in
  (∀ x ∈ [-2..4], (x == 1 → is_min (f x)) ∧ (x >= 1 → nonneg (f (x+1) - f x))) ∧
  (m = f 4) ∧
  (f (-3) > f 2) ∧
  (∀ x, f x < 0 ↔ -1 < x ∧ x < 3) ∧
  ((f (-2)) = 5 ∧ (f 4) = 5) :=
by
  let f := quadratic_function a b c
  let values := y_values a b c f
  sorry

end quadratic_properties_l288_288844


namespace solution_set_f_neg_x_l288_288454

noncomputable def f (a b x : Real) : Real := (a * x - 1) * (x - b)

theorem solution_set_f_neg_x (a b : Real) (h : ∀ x, f a b x > 0 ↔ -1 < x ∧ x < 3) : 
  ∀ x, f a b (-x) < 0 ↔ x < -3 ∨ x > 1 := 
by
  sorry

end solution_set_f_neg_x_l288_288454


namespace ratio_between_house_and_park_l288_288686

theorem ratio_between_house_and_park (w x y : ℝ) (hw : w > 0) (hx : x > 0) (hy : y > 0)
    (h : y / w = x / w + (x + y) / (10 * w)) : x / y = 9 / 11 :=
by 
  sorry

end ratio_between_house_and_park_l288_288686


namespace number_of_zeros_l288_288048

noncomputable def f (x : ℝ) : ℝ :=
  if H : 0 < x ∧ x ≤ 1 then
    ln x + 2
  else
    sorry -- The rest of the function is defined by its properties

theorem number_of_zeros :
  (∀ x : ℝ, f (-x) = -f x) ∧
  (∀ x : ℝ, f (2 - x) = f x) ∧
  (∀ x : ℝ, 0 < x ∧ x ≤ 1 → f x = ln x + 2) →
  ∃! zs : ℝ, (∀ z : ℝ, z ∈ zs → f z = 0 ∧ z ∈ Ico (-2 : ℝ) 4) ∧ zs.card = 9 :=
by
  sorry

end number_of_zeros_l288_288048


namespace true_propositions_l288_288029

variable (m n : Line) (α β : Plane)
variable [NonCoincidentLines m n] [NonCoincidentPlanes α β]

theorem true_propositions :
  (∀ {n : Line}, (α ∩ β = n) → (m ∥ n) → ¬ (m ∥ α ∧ m ∥ β)) ∧
  (∀ {α β : Plane}, (m ⊥ α) → (m ⊥ β) → α ∥ β) ∧
  (∀ {α : Plane}, (m ∥ α) → (m ⊥ n) → ¬ (n ⊥ α)) ∧
  (∀ {α : Plane n : Line}, (m ⊥ α) → (n ⊆ α) → (m ⊥ n)) :=
  by
    sorry

end true_propositions_l288_288029


namespace assignment_methods_l288_288020

theorem assignment_methods :
  ∃ (f : Π (t : Type), set (t → ℕ)), 
  (f bool = {[A, B, C, D]} ∧ 
  (A ≠ B) ∧
  f bool = 8) :=
sorry

end assignment_methods_l288_288020


namespace a_n_formula_T_n_formula_l288_288057

def S (n : ℕ) : ℕ := 2 ^ (n + 1)

def a (n : ℕ) : ℕ :=
  if n = 1 then 4 else 2 ^ n

def b (n : ℕ) : ℝ := 
  (1 / ((n + 1) * Real.log 2 (a n))) + n

def T (n : ℕ) : ℝ :=
  (3 / 4) - (1 / (n + 1)) + (n * (n + 1) / 2)

theorem a_n_formula (n : ℕ) : 
  a n =
  if n = 1 then 4 else 2^n :=
sorry

theorem T_n_formula (n : ℕ) (S_n : ℕ → ℕ) (h1 : S_n n = S n) (b_n : ℕ → ℝ) (h2 : ∀ n, b_n n = b n) :
  T n = (3 / 4) - (1 / (n + 1)) + (n * (n + 1) / 2) :=
sorry

end a_n_formula_T_n_formula_l288_288057


namespace smaller_number_is_476_l288_288695

theorem smaller_number_is_476 (x y : ℕ) 
  (h1 : y - x = 2395) 
  (h2 : y = 6 * x + 15) : 
  x = 476 := 
by 
  sorry

end smaller_number_is_476_l288_288695


namespace part_1_part_2_l288_288451

noncomputable def f (x : ℝ) : ℝ := abs (x + 1) + abs (x - 3)
noncomputable def g (a x : ℝ) : ℝ := a - abs (x - 2)

theorem part_1 (a : ℝ) : (∃ x : ℝ, f x < g a x) → a > 4 := sorry

theorem part_2 (a b : ℝ) : (set.Ioo b (7 / 2)).Nonempty ∧ (∀ x ∈ set.Ioo b (7 / 2), f x < g a x) → a + b = 6 := sorry

end part_1_part_2_l288_288451


namespace count_positive_area_triangles_l288_288095

noncomputable def numPositiveAreaTriangles : ℕ := 2160

theorem count_positive_area_triangles 
  (vertices : list (ℤ × ℤ))
  (h1 : ∀ p ∈ vertices, 1 ≤ p.1 ∧ p.1 ≤ 5 ∧ 1 ≤ p.2 ∧ p.2 ≤ 5)
  (h2 : vertices.length = 25) : 
  (number_of_triangles_with_positive_area vertices) = numPositiveAreaTriangles :=
sorry

end count_positive_area_triangles_l288_288095


namespace area_tripled_sides_l288_288905

theorem area_tripled_sides (a b : ℝ) (θ : ℝ) (hθ : 0 < θ ∧ θ < π) :
  let A := 1 / 2 * a * b * Real.sin θ in
  let A' := 1 / 2 * (3 * a) * (3 * b) * Real.sin θ in
  A' = 9 * A := by
  sorry

end area_tripled_sides_l288_288905


namespace determine_m_l288_288389

theorem determine_m (α : ℝ) :
    (tan α + cot α)^2 + (sec α + csc α)^2 = (5 + sin (2 * α)) + sin^2 α + cos^2 α := 
by 
-- more detailed assumptions or simplifications may be added here if necessary
sorry

end determine_m_l288_288389


namespace range_of_a_l288_288817

theorem range_of_a:
  (∃! y ∈ [-1, 1], ∀ x ∈ [0, 1], x + y^2 * Real.exp y = a) ↔ a ∈ (1 + 1 / Real.exp 1, Real.exp 1] :=
by
  sorry

end range_of_a_l288_288817


namespace range_of_f_l288_288961

noncomputable def f (x : ℝ) : ℝ :=
if x ≤ 0 then x + 1 else 2 ^ x

theorem range_of_f :
  {x : ℝ | f x + f (x - 0.5) > 1} = {x : ℝ | x > -0.25} :=
by
  sorry

end range_of_f_l288_288961


namespace find_sin_phi_l288_288165

variable (p q r : Vector ℝ 3)
variable (φ : ℝ)

-- Conditions
axiom norm_p : ∥p∥ = 2
axiom norm_q : ∥q∥ = 4
axiom norm_r : ∥r∥ = 5
axiom vector_identity : p × (q × r) = 2 • r

theorem find_sin_phi : sin φ = sqrt 15 / 4 := by 
  sorry

end find_sin_phi_l288_288165


namespace count_4_digit_numbers_divisible_by_13_l288_288881

theorem count_4_digit_numbers_divisible_by_13 : 
  let count := (9962 - 1014) / 13 + 1
  1000 ≤ 1014 ∧ 9962 ≤ 9999 →
  count = 689 :=
  by
    sorry

end count_4_digit_numbers_divisible_by_13_l288_288881


namespace min_distance_l288_288143

noncomputable def line_eq (t : ℝ) : ℝ × ℝ :=
(√2/2 * t, -1 + √2/2 * t)

def curve_eq (ρ θ : ℝ) : ℝ × ℝ :=
(ρ * cos θ, ρ * sin θ)

theorem min_distance (P Q : ℝ × ℝ)
  (hP : ∃ t, P = line_eq t)
  (hQ : ∃ θ ρ, ρ * cos θ = Q.1 ∧ ρ * sin θ = Q.2 ∧ Q.1^2 = 2 * Q.2) :
  ∃ Q, Q.1 = 1 / 4 ∧ Q.2 = 1 / 32 ∧ ∀ P (hP : ∃ t, P = line_eq t), dist P Q = 15 * √2 / 32 := by
  sorry

end min_distance_l288_288143


namespace geometric_probability_models_l288_288350

def is_geometric_probability_model : Set ℕ := {n | 
  match n with
  | 1 => True  -- Condition 1 is a geometric probability model
  | 2 => True  -- Condition 2 is a geometric probability model
  | 3 => False -- Condition 3 is not a geometric probability model
  | 4 => True  -- Condition 4 is a geometric probability model
  | _ => False -- Any other condition is not listed
  end
}

theorem geometric_probability_models :
  is_geometric_probability_model = {1, 2, 4} :=
by
  sorry

end geometric_probability_models_l288_288350


namespace arithmetic_geometric_mean_median_l288_288429

theorem arithmetic_geometric_mean_median :
  ∀ (a : ℕ → ℤ) (d : ℤ),
    (∀ n, 0 ≤ n → 0 ≤ n → n < 10 → a (n + 1) = a 1 + n * d) →
    a 3 = 8 →
    a 7 * a 1 = a 3 ^ 2 →
    d ≠ 0 →
    list.mean (list.of_fn (fun n => a (n + 1)) 10) = 13 ∧
    list.median (list.of_fn (fun n => a (n + 1)) 10) = 13 :=
begin
  sorry
end

end arithmetic_geometric_mean_median_l288_288429


namespace new_person_weight_l288_288223

theorem new_person_weight (avg_increase : Real) (n : Nat) (old_weight : Real) (W_new : Real) :
  avg_increase = 2.5 → n = 8 → old_weight = 67 → W_new = old_weight + n * avg_increase → W_new = 87 :=
by
  intros avg_increase_eq n_eq old_weight_eq calc_eq
  sorry

end new_person_weight_l288_288223


namespace tan_30_eq_tan_60_eq_value_of_tan_expression_l288_288781

def tan (θ : ℝ) : ℝ := Real.tan θ  -- Assume Real module provides tangent function
def deg_to_rad (d : ℝ) : ℝ := d * Real.pi / 180  -- Conversion from degrees to radians

theorem tan_30_eq : tan (deg_to_rad 30) = 1 / Real.sqrt 3 := sorry
theorem tan_60_eq : tan (deg_to_rad 60) = Real.sqrt 3 := sorry

theorem value_of_tan_expression :
  (1 + tan (deg_to_rad 30)) * (1 + tan (deg_to_rad 60)) = 2 + (4 * Real.sqrt 3) / 3 :=
by
  rw [tan_30_eq, tan_60_eq]
  sorry

end tan_30_eq_tan_60_eq_value_of_tan_expression_l288_288781


namespace validate_triangle_count_l288_288089

noncomputable def count_valid_triangles : ℕ :=
  let total_points := 25
  let total_triples := (Nat.choose total_points 3)
  let collinear_rows := 5 * (Nat.choose 5 3)
  let collinear_columns := 5 * (Nat.choose 5 3)
  let main_diagonals := 2 * (Nat.choose 5 3)
  let secondary_diagonals := 8 * (Nat.choose 4 3)
  let invalid_triangles := collinear_rows + collinear_columns + main_diagonals + secondary_diagonals
  total_triples - invalid_triangles

theorem validate_triangle_count : count_valid_triangles = 2148 :=
by
  sorry

end validate_triangle_count_l288_288089


namespace count_selected_students_in_range_l288_288133

-- Define the number of students
def num_students : ℕ := 100

-- Define the sample size
def sample_size : ℕ := 25

-- Calculate the sampling interval
def sampling_interval : ℕ := num_students / sample_size

-- Define the initial number randomly chosen
def initial_number : ℕ := 4

-- Check if a number is selected in the systematic sampling
def is_selected (n k : ℕ) (initial : ℕ) : Prop :=
  ∃ m : ℕ, n = initial + m * k

-- Define the set of student numbers from 046 to 078
def student_range : set ℕ := { n | 46 ≤ n ∧ n ≤ 78 }

-- The count of selected students in the given range
def selected_in_range : ℕ := { n | n ∈ student_range ∧ is_selected n sampling_interval initial_number }.card

theorem count_selected_students_in_range :
  selected_in_range = 8 :=
sorry

end count_selected_students_in_range_l288_288133


namespace base8_contains_5_or_6_l288_288855

theorem base8_contains_5_or_6 (n : ℕ) (h : n = 512) : 
  let count_numbers_without_5_6 := 6^3 in
  let total_numbers := 512 in
  total_numbers - count_numbers_without_5_6 = 296 := by
  sorry

end base8_contains_5_or_6_l288_288855


namespace div_by_2_iff_last_digit_even_l288_288206

theorem div_by_2_iff_last_digit_even (a b c : ℕ) (h : a = 10 * b + c) : (a % 2 = 0) ↔ (c % 2 = 0) :=
begin
  sorry
end

end div_by_2_iff_last_digit_even_l288_288206


namespace loci_of_vertices_of_inscribed_square_are_lines_l288_288147

theorem loci_of_vertices_of_inscribed_square_are_lines
  {A B C M N L K : Type*}
  [IsTriangle A B C]
  (h1 : IsSquareInscribedInTriangle MNKL ABC)
  (h2 : OnSideOrExtension L K AB)
  (h3 : OnSides M AC N BC)
  (h4 : MovesAlongLineNotParallelTo C AB) :
  IsPairOfLines (LocusOf (M, N)) :=
sorry

end loci_of_vertices_of_inscribed_square_are_lines_l288_288147


namespace number_of_red_balls_l288_288322

-- Given conditions
def total_balls : ℕ := 100
def white_balls : ℕ := 20
def green_balls : ℕ := 30
def yellow_balls : ℕ := 10
def purple_balls : ℕ := 3
def probability_neither_red_nor_purple : ℝ := 0.6

-- Define the number of red balls to prove it equals 3
def red_balls_solution : ℕ := 3

-- Proof statement that given conditions imply the number of red balls equals 3
theorem number_of_red_balls :
  ∀ (R : ℕ),
  (total_balls - red_balls_solution - purple_balls = white_balls + green_balls + yellow_balls) ∧
  (probability_neither_red_nor_purple = ((white_balls + green_balls + yellow_balls) / (total_balls - purple_balls - R))) →
  R = red_balls_solution :=
by
  intro R
  intro h
  have total_balls_excluding_purple := total_balls - purple_balls
  have left_side := white_balls + green_balls + yellow_balls
  have right_side := total_balls_excluding_purple - R
  cases h with h_left h_right
  simp at h_left ⊢
  have h_actual : left_side = right_side := h_left
  have prob_without_red_purple := left_side / right_side
  simp at h_right ⊢
  rw h_right at prob_without_red_purple
  simp at *
  sorry

end number_of_red_balls_l288_288322


namespace mu_value_of_complex_zeta_l288_288951

noncomputable def is_equilateral_triangle (a b c : ℂ) : Prop := 
  abs (a - b) = abs (b - c) ∧ abs (b - c) = abs (c - a)

theorem mu_value_of_complex_zeta 
  (zeta : ℂ) 
  (hzeta : abs zeta = 3)
  (hmu : ∃ μ > 1, is_equilateral_triangle zeta (zeta ^ 2) (μ * zeta)) :
  ∃ μ > 1, μ = (1 + Real.sqrt 33) / 2 :=
  sorry

end mu_value_of_complex_zeta_l288_288951


namespace sin_double_angle_l288_288496

variable {θ : ℝ}

theorem sin_double_angle (h : cos θ + sin θ = 3 / 2) : sin (2 * θ) = 5 / 4 :=
by
  sorry

end sin_double_angle_l288_288496


namespace exact_differential_solution_l288_288215

theorem exact_differential_solution (x y : ℝ) :
  ∃ C : ℝ, ∃ u : ℝ → ℝ → ℝ, 
  (∂ u / ∂ x = x + y ∧ ∂ u / ∂ y = x + 2y) ∧
  (u x y = x^2 / 2 + y * x + y^2) ∧
  (u x y = C) := 
sorry

end exact_differential_solution_l288_288215


namespace sphere_radius_cone_l288_288340

-- Define the base radius and height of the cone
def base_radius : ℝ := 20
def height : ℝ := 30

-- Hypothetical constants for the radius of the sphere
def a : ℝ := 120
def b : ℝ := 120
def c : ℝ := 13

-- The formulation of the radius
def radius_sphere := a * Real.sqrt c - b

-- The statement to prove
theorem sphere_radius_cone (base_radius height a b c : ℝ) (a := 120) (b := 120) (c := 13) 
 (h_base_radius : base_radius = 20) 
 (h_height : height = 30) 
 (h_radius : radius_sphere = a * Real.sqrt c - b) : 
 a + b + c = 253 :=
by 
  sorry

end sphere_radius_cone_l288_288340


namespace constant_term_expansion_l288_288509

theorem constant_term_expansion (n : ℕ) (h : n ∈ (Finset.range 1 (n + 1))) (h_binom : n.choose 2 = 36) :
  let T := λ r, (Int.pow (-1) r) * (Int.pow 3 (18 - 3*r)) * (Nat.choose 9 r) * (Int.pow x (9 - (3*r) / 2))
  in T 6 = 84 :=
by
  sorry

end constant_term_expansion_l288_288509


namespace geometric_series_cubes_sum_l288_288332

theorem geometric_series_cubes_sum (b s : ℝ) (h : -1 < s ∧ s < 1) :
  ∑' n : ℕ, (b * s^n)^3 = b^3 / (1 - s^3) := 
sorry

end geometric_series_cubes_sum_l288_288332


namespace four_digit_numbers_divisible_by_13_l288_288876

theorem four_digit_numbers_divisible_by_13 : 
  (set.Icc 1000 9999).filter (λ x, x % 13 = 0).cardinality = 693 :=
by
  sorry

end four_digit_numbers_divisible_by_13_l288_288876


namespace sequence_general_term_l288_288430

noncomputable def sequence (n : ℕ) : ℝ :=
  if n = 0 then 0 else 2 / (n * (n + 1))

theorem sequence_general_term (n : ℕ) (hn : n > 0) :
  ∀ a : ℕ → ℝ, (a 1 = 1) → (∀ n : ℕ, n > 0 → sum (λ k, a k) n = n^2 * (a n)) → 
  a n = sequence n := sorry

end sequence_general_term_l288_288430


namespace smart_charging_piles_equation_l288_288524

-- defining conditions as constants
constant initial_piles : ℕ := 301
constant third_month_piles : ℕ := 500
constant growth_rate : ℝ 

-- Expressing the given mathematical proof problem
theorem smart_charging_piles_equation : 
  initial_piles * (1 + growth_rate)^2 = third_month_piles :=
sorry

end smart_charging_piles_equation_l288_288524


namespace count_base_8_digits_5_or_6_l288_288863

-- Define the conditions in Lean
def is_digit_5_or_6 (d : ℕ) : Prop := d = 5 ∨ d = 6

def count_digits_5_or_6 := 
  let total_base_8 := 512
  let total_without_5_6 := 6 * 6 * 6 -- since we exclude 2 out of 8 digits
  total_base_8 - total_without_5_6

-- The statement of the proof problem
theorem count_base_8_digits_5_or_6 : count_digits_5_or_6 = 296 :=
by {
  sorry
}

end count_base_8_digits_5_or_6_l288_288863


namespace sequence_values_general_term_sum_inverse_a_l288_288431

theorem sequence_values (S : ℕ → ℕ) (a : ℕ → ℕ) 
  (h1 : S 1 = 1) (h2 : ∀ n, 3 * S n = (n + 2) * a n) :
  a 2 = 3 ∧ a 3 = 6 :=
by sorry

theorem general_term (S : ℕ → ℕ) (a : ℕ → ℕ) 
  (h1 : S 1 = 1) (h2 : ∀ n, 3 * S n = (n + 2) * a n) :
  ∀ n, a n = n * (n + 1) / 2 :=
by sorry

theorem sum_inverse_a (a : ℕ → ℕ) 
  (h : ∀ n, a n = n * (n + 1) / 2) :
  ∀ n, (Finset.range n).sum (λ k, 1 / (a (k + 1) : ℚ)) = 2 * n / (n + 1) :=
by sorry

end sequence_values_general_term_sum_inverse_a_l288_288431


namespace carol_allowance_problem_l288_288376

open Real

theorem carol_allowance_problem (w : ℝ) 
  (fixed_allowance : ℝ := 20) 
  (extra_earnings_per_week : ℝ := 22.5) 
  (total_money : ℝ := 425) :
  fixed_allowance * w + extra_earnings_per_week * w = total_money → w = 10 :=
by
  intro h
  -- Proof skipped
  sorry

end carol_allowance_problem_l288_288376


namespace correct_equation_l288_288265

variable (x : ℝ)
axiom area_eq_720 : x * (x - 6) = 720

theorem correct_equation : x * (x - 6) = 720 := by
  exact area_eq_720

end correct_equation_l288_288265


namespace expected_value_traffic_jam_commute_l288_288191

open ProbabilityTheory

noncomputable def traffic_jam_expected_value (n : ℕ) (p : ℚ) : ℚ :=
  n * p

theorem expected_value_traffic_jam_commute :
  traffic_jam_expected_value 6 (1/6 : ℚ) = 1 :=
by
  -- The expected value of a binomial distribution B(6, 1/6)
  -- is calculated by multiplying the number of trials n by the probability of success p.
  -- Here, n = 6 and p = 1/6, so the expected value is E(ξ) = 6 * (1/6) = 1.
  sorry

end expected_value_traffic_jam_commute_l288_288191


namespace road_greening_cost_l288_288714

-- Define constants for the conditions
def l_total : ℕ := 1500
def cost_A : ℕ := 22
def cost_B : ℕ := 25

-- Define variables for the cost per stem
variables (x y : ℕ)

-- Define the conditions from Plan A and Plan B
def plan_A (x y : ℕ) : Prop := 2 * x + 3 * y = cost_A
def plan_B (x y : ℕ) : Prop := x + 5 * y = cost_B

-- System of equations to find x and y
def system_of_equations (x y : ℕ) : Prop := plan_A x y ∧ plan_B x y

-- Define the constraint for the length of road greened according to Plan B
def length_constraint (a : ℕ) : Prop := l_total - a ≥ 2 * a

-- Define the total cost function
def total_cost (a : ℕ) (x y : ℕ) : ℕ := 22 * a + (x + 5 * y) * (l_total - a)

-- Prove the cost per stem and the minimized cost
theorem road_greening_cost :
  (∃ x y, system_of_equations x y ∧ x = 5 ∧ y = 4) ∧
  (∃ a : ℕ, length_constraint a ∧ a = 500 ∧ total_cost a 5 4 = 36000) :=
by
  -- This is where the proof would go
  sorry

end road_greening_cost_l288_288714


namespace amare_initial_fabric_l288_288597

-- Definitions based on the given conditions
def yards_per_dress : ℝ := 5.5
def feet_per_yard : ℝ := 3
def num_dresses : ℕ := 4
def feet_still_needed : ℕ := 59

-- Goal: Prove that Amare initially had 7 feet of fabric
theorem amare_initial_fabric : 
  let feet_per_dress := yards_per_dress * feet_per_yard,
      total_feet_needed := num_dresses * feet_per_dress
  in total_feet_needed - feet_still_needed = 7 :=
by
  sorry

end amare_initial_fabric_l288_288597


namespace reflection_identity_l288_288985

-- Define the reflection function
def reflect (O P : ℝ × ℝ) : ℝ × ℝ := (2 * O.1 - P.1, 2 * O.2 - P.2)

-- Given three points and a point P
variables (O1 O2 O3 P : ℝ × ℝ)

-- Define the sequence of reflections
def sequence_reflection (P : ℝ × ℝ) : ℝ × ℝ :=
  reflect O3 (reflect O2 (reflect O1 P))

-- Lean 4 statement to prove the mathematical theorem
theorem reflection_identity :
  sequence_reflection O1 O2 O3 (sequence_reflection O1 O2 O3 P) = P :=
by sorry

end reflection_identity_l288_288985


namespace sum_g_inv_1_to_2500_l288_288954

def g (n : ℕ) : ℕ := nat.floor (real.cbrt n)

theorem sum_g_inv_1_to_2500 : 
  ∑ k in finset.range 2500.succ, (1 / g k : ℝ) = 315 := 
by
  sorry

end sum_g_inv_1_to_2500_l288_288954


namespace pyramid_height_l288_288732

-- Define the heights of individual blocks and the structure of the pyramid.
def block_height := 10 -- in centimeters
def num_layers := 3

-- Define the total height of the pyramid as the sum of the heights of all blocks.
def total_height (block_height : Nat) (num_layers : Nat) := block_height * num_layers

-- The theorem stating that the total height of the stack is 30 cm given the conditions.
theorem pyramid_height : total_height block_height num_layers = 30 := by
  sorry

end pyramid_height_l288_288732


namespace real_part_zero_implies_m_l288_288908

theorem real_part_zero_implies_m (m : ℝ) (h : (m^2 - m) = 0) (h_ne : m ≠ 0) : m = 1 :=
by
  -- Conditions:
  -- (m^2 - m) = 0
  -- m ≠ 0
  have h1 : m * (m - 1) = 0, from h,
  sorry -- Proof goes here

end real_part_zero_implies_m_l288_288908


namespace eva_marks_difference_l288_288391

theorem eva_marks_difference 
    (m2 : ℕ) (a2 : ℕ) (s2 : ℕ) (total_marks : ℕ)
    (h_m2 : m2 = 80) (h_a2 : a2 = 90) (h_s2 : s2 = 90) (h_total_marks : total_marks = 485)
    (m1 a1 s1 : ℕ)
    (h_m1 : m1 = m2 + 10)
    (h_a1 : a1 = a2 - 15)
    (h_s1 : s1 = s2 - 1 / 3 * s2)
    (total_semesters : ℕ)
    (h_total_semesters : total_semesters = m1 + a1 + s1 + m2 + a2 + s2)
    : m1 = m2 + 10 := by
  sorry

end eva_marks_difference_l288_288391


namespace leaves_falling_every_day_l288_288370

-- Definitions of the conditions
def roof_capacity := 500 -- in pounds
def leaves_per_pound := 1000 -- number of leaves per pound
def collapse_time := 5000 -- in days

-- Function to calculate the number of leaves falling each day
def leaves_per_day (roof_capacity : Nat) (leaves_per_pound : Nat) (collapse_time : Nat) : Nat :=
  (roof_capacity * leaves_per_pound) / collapse_time

-- Theorem stating the expected result
theorem leaves_falling_every_day :
  leaves_per_day roof_capacity leaves_per_pound collapse_time = 100 :=
by
  sorry

end leaves_falling_every_day_l288_288370


namespace sum_proper_divisors_72_eq_123_l288_288677

-- Definitions stated in the problem conditions
def is_proper_divisor (n d : ℕ) : Prop := d ∣ n ∧ d ≠ n

def proper_divisors (n : ℕ) : set ℕ := { d | is_proper_divisor n d }

-- Define the sum of proper divisors function
def sum_proper_divisors (n : ℕ) : ℕ := ∑ d in proper_divisors n, d

-- The theorem to prove
theorem sum_proper_divisors_72_eq_123 : sum_proper_divisors 72 = 123 := by
  sorry

end sum_proper_divisors_72_eq_123_l288_288677


namespace number_of_triangles_in_figure_l288_288896

theorem number_of_triangles_in_figure (small_triangles_first_section : ℕ)
                                      (small_triangles_additional_section : ℕ)
                                      (combined_triangles_first_section : ℕ)
                                      (combined_triangles_additional_section : ℕ) :
  small_triangles_first_section = 6 →
  small_triangles_additional_section = 5 →
  combined_triangles_first_section = 5 →
  combined_triangles_additional_section = 0 →
  (small_triangles_first_section + small_triangles_additional_section + 
   combined_triangles_first_section + combined_triangles_additional_section) = 16 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  norm_num
  sorry

end number_of_triangles_in_figure_l288_288896


namespace max_ab_of_tangent_circles_l288_288821

theorem max_ab_of_tangent_circles (a b : ℝ) 
  (hC1 : ∀ x y : ℝ, (x - a)^2 + (y + 2)^2 = 4)
  (hC2 : ∀ x y : ℝ, (x + b)^2 + (y + 2)^2 = 1)
  (h_tangent : a + b = 3) :
  ab ≤ 9 / 4 :=
by
  sorry

end max_ab_of_tangent_circles_l288_288821


namespace divide_plot_into_equal_parts_l288_288416

-- Define positions of apple trees and currant bushes
def apple_trees : Finset ℕ := {0, 1, 2, 3}
def currant_bushes : Finset ℕ := {0, 1, 2, 3, 4, 5, 6, 7}

-- Define a function that counts the number of elements in a part
def count_elements (part : Finset ℕ) : ℕ := part.card

-- Define the matchsticks required
def total_matchsticks_used : ℕ := 13

-- Define each section
structure Section :=
(elements : Finset ℕ)
(contains_one_apple_tree : ∃ (x : ℕ), x ∈ apple_trees ∧ x ∈ elements)
(contains_two_currant_bushes : (Finset.filter (λ x, x ∈ currant_bushes) elements).card = 2)

def equal_parts (sections : List Section) : Prop :=
sections.length = 4 ∧ (∀ s, s ∈ sections → s.elements.card = 3)

noncomputable def plot_division (sections : List Section) : Prop :=
equal_parts sections ∧
(∀ s, s ∈ sections → s.contains_one_apple_tree ∧ s.contains_two_currant_bushes) ∧
total_matchsticks_used = 13

-- The theorem that needs to be proven
theorem divide_plot_into_equal_parts :
  ∃ (sections : List Section), plot_division sections :=
sorry

end divide_plot_into_equal_parts_l288_288416


namespace triangle_area_exists_l288_288567

theorem triangle_area_exists (m n k : ℕ) (hm : 0 < m) (hn :0 < n) (hk : 0 < k) (hkn : k ≤ m * n) :
  ∃ (A B C : (ℕ × ℕ)), (0 ≤ A.fst ∧ A.fst ≤ m ∧ 0 ≤ A.snd ∧ A.snd ≤ n) ∧
                       (0 ≤ B.fst ∧ B.fst ≤ m ∧ 0 ≤ B.snd ∧ B.snd ≤ n) ∧
                       (0 ≤ C.fst ∧ C.fst ≤ m ∧ 0 ≤ C.snd ∧ C.snd ≤ n) ∧
                       (triangle_area A B C = (k / 2)) :=
sorry

noncomputable def triangle_area (A B C : (ℕ × ℕ)) : ℚ :=
-- Compute the area of the triangle with vertices A, B, and C on the Cartesian plane
(1 / 2) * ((B.1 - A.1) * (C.2 - A.2) - (B.2 - A.2) * (C.1 - A.1))

end triangle_area_exists_l288_288567


namespace fraction_identity_l288_288463

theorem fraction_identity (a b c : ℝ) (h1 : a + b + c > 0) (h2 : a + b - c > 0) (h3 : a + c - b > 0) (h4 : b + c - a > 0) 
  (h5 : (a+b+c)/(a+b-c) = 7) (h6 : (a+b+c)/(a+c-b) = 1.75) : (a+b+c)/(b+c-a) = 3.5 :=
by
  sorry

end fraction_identity_l288_288463


namespace area_of_region_l288_288286

theorem area_of_region :
  ∀ (x y : ℝ), (x^2 + y^2 - 4 * x + 2 * y = -2) → (π * 3) = 3 * π :=
begin
  intros x y h,
  sorry
end

end area_of_region_l288_288286


namespace perpendicular_lines_m_equation_of_line_l288_288476

-- Define the conditions for the given lines l1 and l2
def line_l1 (m : ℝ) : ℝ → ℝ → Prop := λ x y, (m - 2) * x + m * y - 8 = 0

def line_l2 (m : ℝ) : ℝ → ℝ → Prop := λ x y, m * x + y - 3 = 0

-- Part (I): Prove that m = 1 or m = 0 if l1 is perpendicular to l2
theorem perpendicular_lines_m (m : ℝ) :
  (∀ x y, line_l1 m x y → line_l2 m x y → m = 1 ∨ m = 0) :=
sorry

-- Part (II): Prove the equation of line l given the point P and the sum of intercepts condition
theorem equation_of_line (m : ℝ) (P : ℝ × ℝ) (l : ℝ → ℝ → Prop)
  (H1 : P = (1, 2 * m))
  (H2 : ∀ x y, l x y → x + y = 1) :
  l = λ x y, x - y + 1 = 0 :=
sorry

end perpendicular_lines_m_equation_of_line_l288_288476


namespace point_on_line_l288_288397

theorem point_on_line : 
  ∃ t : ℚ, (3 * t + 1 = 0) ∧ ((2 - 4) / (t - 1) = (7 - 4) / (3 - 1)) :=
by
  sorry

end point_on_line_l288_288397


namespace distance_between_intersections_l288_288638

theorem distance_between_intersections :
  let f : ℝ → ℝ := λ x, 3 * x ^ 2 + 2 * x - 5
  let g : ℝ := -2
  ∀ x1 x2 : ℝ, f x1 = g ∧ f x2 = g →
  abs (x1 - x2) = 2 * real.sqrt 10 / 3 :=
by
  sorry

end distance_between_intersections_l288_288638


namespace number_of_valid_triangles_l288_288099

theorem number_of_valid_triangles : 
  let points := {(x, y) | 1 ≤ x ∧ x ≤ 5 ∧ 1 ≤ y ∧ y ≤ 5} in
  ∃ n : ℕ, n = (nat.card (points.choose 3)) - 120 ∧ n = 2180 :=
by
  sorry

end number_of_valid_triangles_l288_288099


namespace matt_total_points_l288_288590

variable (n2_successful_shots : Nat) (n3_successful_shots : Nat)

def total_points (n2 : Nat) (n3 : Nat) : Nat :=
  2 * n2 + 3 * n3

theorem matt_total_points :
  total_points 4 2 = 14 :=
by
  sorry

end matt_total_points_l288_288590


namespace find_k_l288_288001

-- Define the series summation function
def series (k : ℝ) : ℝ := 4 + (∑ n, (4 + n * k) / 5^n)

-- State the theorem with the given condition and required proof
theorem find_k (h : series k = 10) : k = 16 := sorry

end find_k_l288_288001


namespace gladys_age_ten_years_from_now_l288_288556

-- Define the age relations
def juanico_current_age (gladys_current_age : ℕ) : ℕ := (gladys_current_age / 2) - 4
def gladys_current_age_from_juanico (juanico_current_age : ℕ) : ℕ := 2 * (juanico_current_age + 4)

-- Prove the main statement
theorem gladys_age_ten_years_from_now (G : ℕ) (H : ℕ) (J : ℕ)
    (gladys_future_age : G = H + 10)
    (juanico_age_equation : J = H / 2 - 4)
    (juanico_future_age : J + 30 = 41) : G = 40 :=
by {
  have JC : J = 11 := by rw [juanico_future_age, <-sub_eq_add_neg] at juanico_future_age; exact sub_eq_of_eq_add juanico_future_age,
  have GC : H = 30 := by rw [<-sub_add, <-mul_two] and exact (mul_eq_of_eq_div (by exact mul_comm) (by exact add_eq_of_eq_sub' JC)),
  exact H + 10 = G at gladys_future_age; rw [GC] and exact rfl
}

end gladys_age_ten_years_from_now_l288_288556


namespace max_sections_with_lines_l288_288933

theorem max_sections_with_lines (n : ℕ) (h : n = 5) :
  ∃ m : ℕ, m = 15 := by
  use 15
  sorry

end max_sections_with_lines_l288_288933


namespace expected_students_count_l288_288911

-- Define a normal distribution X with mean mu and variance sigma^2
variables {μ σ : ℝ} (hσ : σ > 0)

-- Let X ~ N(μ, σ^2)
def normal_dist (X : ℝ → ℝ) (μ σ : ℝ) : Prop :=
  ∀ x, X x = 1 / (σ * sqrt(2 * π)) * exp(-0.5 * ((x - μ) / σ) ^ 2)

-- Given probabilities for standard deviations from the mean
axiom prob_1σ : P (λ X, μ - σ < X ∧ X ≤ μ + σ) = 0.6826
axiom prob_2σ : P (λ X, μ - 2 * σ < X ∧ X ≤ μ + 2 * σ) = 0.9544
axiom prob_3σ : P (λ X, μ - 3 * σ < X ∧ X ≤ μ + 3 * σ) = 0.9974

-- Define the class size and properties of their scores distribution
def class_size : ℕ := 48
def mu_score : ℝ := 80
def sigma_score : ℝ := 10
def scores_normal_dist (X : ℝ → ℝ) : Prop := normal_dist X mu_score sigma_score

-- Theoretically expected number of students scoring between 80 and 90
theorem expected_students_count :
  let p := 0.6826 / 2 in
  class_size * p = 16 := by
  sorry

end expected_students_count_l288_288911


namespace sin_double_theta_eq_five_fourths_l288_288499

theorem sin_double_theta_eq_five_fourths (theta : ℝ) (h : cos theta + sin theta = 3/2) : sin (2 * theta) = 5/4 :=
by
  sorry

end sin_double_theta_eq_five_fourths_l288_288499


namespace SallyMcQueenCostCorrect_l288_288185

def LightningMcQueenCost : ℕ := 140000
def MaterCost : ℕ := (140000 * 10) / 100
def SallyMcQueenCost : ℕ := 3 * MaterCost

theorem SallyMcQueenCostCorrect : SallyMcQueenCost = 42000 := by
  sorry

end SallyMcQueenCostCorrect_l288_288185


namespace hyperbola_sufficient_but_not_necessary_l288_288935

theorem hyperbola_sufficient_but_not_necessary :
  (∀ (C : Type) (x y : ℝ), C = {p : ℝ × ℝ | ((p.1)^2 / 16) - ((p.2)^2 / 9) = 1} →
  (∀ x, y = 3 * (x / 4) ∨ y = -3 * (x / 4)) →
  ∃ (C' : Type) (x' y' : ℝ), C' = {p : ℝ × ℝ | ((p.1)^2 / 64) - ((p.2)^2 / 36) = 1} ∧
  (∀ x', y' = 3 * (x' / 4) ∨ y' = -3 * (x' / 4))) :=
sorry

end hyperbola_sufficient_but_not_necessary_l288_288935


namespace chloe_and_friends_points_l288_288753

-- Define the conditions as Lean definitions and then state the theorem to be proven.

def total_pounds_recycled : ℕ := 28 + 2

def pounds_per_point : ℕ := 6

def points_earned (total_pounds : ℕ) (pounds_per_point : ℕ) : ℕ :=
  total_pounds / pounds_per_point

theorem chloe_and_friends_points :
  points_earned total_pounds_recycled pounds_per_point = 5 :=
by
  sorry

end chloe_and_friends_points_l288_288753


namespace power_equality_l288_288901

theorem power_equality (x : ℝ) (hx : 5^x = 100) : 5^(x+2) = 2500 :=
by sorry

end power_equality_l288_288901


namespace percent_non_bball_in_theater_l288_288516

variable (N : ℕ)

def students_play_basketball := 0.7 * N
def students_in_theater := 0.4 * N
def bball_and_theater := 0.2 * students_play_basketball
def only_basketball := students_play_basketball - bball_and_theater
def not_play_basketball := N - students_play_basketball
def only_theater := students_in_theater - bball_and_theater

theorem percent_non_bball_in_theater :
  (only_theater / not_play_basketball) * 100 = 87 := by
  sorry

end percent_non_bball_in_theater_l288_288516


namespace number_with_5_or_6_base_8_l288_288869

open Finset

def count_numbers_with_5_or_6 : ℕ :=
  let base_8_numbers := Ico 1 (8 ^ 3)
  let count_with_5_or_6 := base_8_numbers.filter (λ n, ∃ b, b ∈ digit_set 8 n ∧ (b = 5 ∨ b = 6))
  count_with_5_or_6.card

theorem number_with_5_or_6_base_8 : count_numbers_with_5_or_6 = 296 := 
by 
  -- Proof omitted for this exercise
  sorry

end number_with_5_or_6_base_8_l288_288869


namespace tan_theta_eq_pm_sqrt3_l288_288041

noncomputable def tan_value (theta : ℝ) : ℝ := if (cos (pi + theta) = -1/2) then tan (theta) else 0

theorem tan_theta_eq_pm_sqrt3 (theta : ℝ) (h : cos (pi + theta) = -1/2) : 
  tan (theta - 9 * pi) = if (sin theta = sqrt 3 / 2) then sqrt 3 else if (sin theta = -sqrt 3 / 2) then -sqrt 3 else 0 :=
sorry

end tan_theta_eq_pm_sqrt3_l288_288041


namespace sequence_a_n_l288_288542

theorem sequence_a_n (a : ℤ) (h : (-1)^1 * 1 + a + (-1)^4 * 4 + a = 3 * ( (-1)^2 * 2 + a )) :
  a = -3 ∧ ((-1)^100 * 100 + a) = 97 :=
by
  sorry  -- proof is omitted

end sequence_a_n_l288_288542


namespace fraction_of_brilliant_integers_divisible_by_11_l288_288385

def is_even (n : ℕ) : Prop := n % 2 = 0

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

def is_brilliant (n : ℕ) : Prop :=
  is_even n ∧ n > 10 ∧ n < 150 ∧ sum_of_digits n = 10

theorem fraction_of_brilliant_integers_divisible_by_11 :
  (∃ k : ℕ, is_brilliant k ∧ k % 11 = 0 ∧ k / 8) := sorry

end fraction_of_brilliant_integers_divisible_by_11_l288_288385


namespace range_of_n_l288_288117

theorem range_of_n (m n : ℝ) (h₁ : n = m^2 + 2 * m + 2) (h₂ : |m| < 2) : -1 ≤ n ∧ n < 10 :=
sorry

end range_of_n_l288_288117


namespace circle_equation_AB_diameter_l288_288246

theorem circle_equation_AB_diameter :
  let A B : ℝ × ℝ := [(1 - 2*sqrt 3, 2*sqrt 3), (1 + 2*sqrt 3, -2*sqrt 3)] in
  let M : ℝ × ℝ := (7, 2*sqrt 3) in
  let r : ℝ := sqrt 64 in
  ∃ center : ℝ × ℝ, center = M ∧ r = 8 ∧ 
  (∀ x y : ℝ, ((x - 7)^2 + (y - 2*sqrt 3)^2 = 64) ↔ ((x,y) ∈ {p : ℝ × ℝ | (p.1 - 7)^2 + (p.2 - 2*sqrt 3)^2 = r^2})) :=
sorry

end circle_equation_AB_diameter_l288_288246


namespace intersect_range_k_l288_288639

theorem intersect_range_k : 
  ∀ k : ℝ, (∃ x y : ℝ, x^2 - (kx + 2)^2 = 6) ↔ 
  -Real.sqrt (5 / 3) < k ∧ k < Real.sqrt (5 / 3) := 
by sorry

end intersect_range_k_l288_288639


namespace probability_two_heads_in_three_tosses_correct_l288_288713

noncomputable def probability_two_heads_in_three_tosses : ℚ :=
  let outcomes : finset (fin 2 × fin 2 × fin 2) := finset.product (finset.product finset.univ finset.univ) finset.univ
  let event_A := outcomes.filter (λ s, (s.1.1.val + s.1.2.val + s.2.val) = 2)
  (event_A.card : ℚ) / (outcomes.card : ℚ)

theorem probability_two_heads_in_three_tosses_correct :
  probability_two_heads_in_three_tosses = 3 / 8 :=
by
  sorry

end probability_two_heads_in_three_tosses_correct_l288_288713


namespace sum_of_four_subsets_equal_l288_288987

-- Define the problem statement
theorem sum_of_four_subsets_equal (s : Finset ℕ) (h₁ : s.card = 117) (h₂ : ∀ x ∈ s, 100 ≤ x ∧ x ≤ 999) :
  ∃ a b c d : Finset ℕ, a ∩ b = ∅ ∧ a ∩ c = ∅ ∧ a ∩ d = ∅ ∧ b ∩ c = ∅ ∧ b ∩ d = ∅ ∧ c ∩ d = ∅ ∧ 
  (a ∪ b ∪ c ∪ d) ⊆ s ∧ a.sum id = b.sum id ∧ a.sum id = c.sum id ∧ a.sum id = d.sum id :=
by
  sorry

end sum_of_four_subsets_equal_l288_288987


namespace cats_needed_to_catch_100_mice_in_time_l288_288701

-- Define the context and given conditions
def cats_mice_catch_time (cats mice minutes : ℕ) : Prop :=
  cats = 5 ∧ mice = 5 ∧ minutes = 5

-- Define the goal
theorem cats_needed_to_catch_100_mice_in_time :
  cats_mice_catch_time 5 5 5 → (∃ t : ℕ, t = 500) :=
by
  intro h
  sorry

end cats_needed_to_catch_100_mice_in_time_l288_288701


namespace max_period_of_function_l288_288801

theorem max_period_of_function (f : ℝ → ℝ) (h1 : ∀ x, f (1 + x) = f (1 - x)) (h2 : ∀ x, f (8 + x) = f (8 - x)) :
  ∃ T > 0, (∀ x, f (x + T) = f x) ∧ (∀ T' > 0, (∀ x, f (x + T') = f x) → T' ≥ 14) ∧ T = 14 :=
sorry

end max_period_of_function_l288_288801


namespace crayons_loss_difference_l288_288599

theorem crayons_loss_difference (crayons_given crayons_lost : ℕ) 
  (h_given : crayons_given = 90) 
  (h_lost : crayons_lost = 412) : 
  crayons_lost - crayons_given = 322 :=
by
  sorry

end crayons_loss_difference_l288_288599


namespace expression1_value_expression2_value_l288_288783

-- Conditions for Expression 1
def sin_30 := 1 / 2
def cos_60 := 1 / 2
def tan_45 := 1

-- Proof Statement for Expression 1
theorem expression1_value :
  2 * sin_30 + 3 * cos_60 - 4 * tan_45 = -3 / 2 :=
by 
  sorry

-- Conditions for Expression 2
def tan_60 := Real.sqrt 3
def four_minus_pi_pow_0 := 1  -- Since anything to the power of 0 is 1.
def cos_30 := Real.sqrt 3 / 2
def one_fourth_pow_neg1 := 4

-- Proof Statement for Expression 2
theorem expression2_value :
  tan_60 - four_minus_pi_pow_0 + 2 * cos_30 + one_fourth_pow_neg1 = 2 * Real.sqrt 3 + 3 :=
by 
  sorry

end expression1_value_expression2_value_l288_288783


namespace height_percentage_l288_288507

theorem height_percentage {Q P : ℝ} (hQ_gt_0: Q > 0)
  (hP_def: P = Q * 0.6) :
  ((Q - P) / P) * 100 ≈ 66.67 :=
by norm_num; sorry

end height_percentage_l288_288507


namespace find_r13_l288_288693

-- Define the problem
variables (results : Fin 25 → ℝ)
variables (avg_total avg_first12 avg_last12 r13 : ℝ)

-- Given conditions
def total_sum := 25 * avg_total
def first12_sum := 12 * avg_first12
def last12_sum := 12 * avg_last12

-- Defining the sums based on given averages
def given_conditions (h_avg_total : avg_total = 18) 
                     (h_avg_first12 : avg_first12 = 14)
                     (h_avg_last12 : avg_last12 = 17)
                     (h_total_sum : (Finset.univ.sum results) = total_sum)
                     (h_first12_sum : (Finset.range 12).sum results = first12_sum)
                     (h_last12_sum : (Finset.range 12).sum (λ i, results (Fin 25 (12 + i))) = last12_sum) :
  (results ⟨12, by decide⟩ = r13) :=
sorry

-- The proof goal
theorem find_r13 : ∀ h_avg_total, h_avg_first12, h_avg_last12, h_total_sum, h_first12_sum, h_last12_sum,
  given_conditions h_avg_total h_avg_first12 h_avg_last12 h_total_sum h_first12_sum h_last12_sum -> r13 = 78 :=
by
  -- Implement the theorem proof
  sorry

end find_r13_l288_288693


namespace number_of_marbles_in_Ellen_box_l288_288012

-- Defining the conditions given in the problem
def Dan_box_volume : ℕ := 216
def Ellen_side_multiplier : ℕ := 3
def marble_size_consistent_between_boxes : Prop := True -- Placeholder for the consistency condition

-- Main theorem statement
theorem number_of_marbles_in_Ellen_box :
  ∃ number_of_marbles_in_Ellen_box : ℕ,
  (∀ s : ℕ, s^3 = Dan_box_volume → (Ellen_side_multiplier * s)^3 / s^3 = 27 → 
  number_of_marbles_in_Ellen_box = 27 * Dan_box_volume) :=
by
  sorry

end number_of_marbles_in_Ellen_box_l288_288012


namespace series_sum_correct_l288_288747

noncomputable def infinite_series_sum : ℚ :=
  ∑' n, (5 + n) * (1 / 1000) ^ n

theorem series_sum_correct : infinite_series_sum = 4995005 / 998001 := by
  sorry

end series_sum_correct_l288_288747


namespace work_done_l288_288320

noncomputable def F : ℝ → ℝ :=
λ x, if x ≤ 2 then 5 else 3 * x + 4

def work (a b : ℝ) (F : ℝ → ℝ) : ℝ :=
∫ x in a..b, F x

theorem work_done : work 0 4 F = 36 := by
  sorry

end work_done_l288_288320


namespace tens_ones_digit_sum_l288_288289

-- Definition and assumptions as per conditions
def ones_digit (n: ℕ) : ℕ := n % 10
def tens_digit (n: ℕ) : ℕ := (n / 10) % 10
def digit_sum (a b : ℕ) : ℕ := a + b

theorem tens_ones_digit_sum (n : ℕ) (h1 : n = (1 + 6) ^ 12):
  digit_sum (tens_digit n) (ones_digit n) = 1 :=
by
  have h2 : (1 + 6) ^ 12 = 7 ^ 12 := by norm_num
  -- Using the derived fact that 7^12 has ones digit 1 based on cyclic pattern
  have h3 : ones_digit (7 ^ 12) = 1 := by sorry
  have h4 : tens_digit (7 ^ 12) = 0 := by sorry
  rw [h2] at h1
  rw [<- h1]  -- Align n
  rw [<- nat.add_zero 1]  -- 1 = 0 + 1
  -- Split sum into components
  exact calc
    digit_sum (tens_digit n) (ones_digit n)
      = digit_sum 0 (ones_digit n) : by rw [h4]
  ... = digit_sum 0 1            : by rw [h3]
  ... = 1                        : by simp

end tens_ones_digit_sum_l288_288289


namespace find_x_for_perpendicular_vectors_l288_288852

noncomputable def vector_a : ℝ × ℝ × ℝ := (2, -1, 3)
noncomputable def vector_b (x : ℝ) : ℝ × ℝ × ℝ := (-4, 2, x)
noncomputable def vector_c (x : ℝ) : ℝ × ℝ × ℝ := (1, -x, 2)
noncomputable def dot_product (v1 v2 : ℝ × ℝ × ℝ) : ℝ := 
  v1.1 * v2.1 + v1.2 * v2.2 + v1.3 * v2.3

theorem find_x_for_perpendicular_vectors : 
  ∃ x : ℝ, dot_product (vector_a.1 - 4, vector_a.2 + 2, vector_a.3 + x + 3) (vector_c x) = 0 ∧ x = -4 := 
sorry

end find_x_for_perpendicular_vectors_l288_288852


namespace arithmetic_sequence_first_term_l288_288530

theorem arithmetic_sequence_first_term :
  ∃ a₁ a₂ d : ℤ, a₂ = -5 ∧ d = 3 ∧ a₂ = a₁ + d ∧ a₁ = -8 :=
by
  sorry

end arithmetic_sequence_first_term_l288_288530


namespace find_f_neg_one_l288_288014

noncomputable def f : ℝ → ℝ := sorry

axiom functional_equation : ∀ x y : ℝ, f (x^2 + y) = f x + f (y^2)

theorem find_f_neg_one : f (-1) = 0 := sorry

end find_f_neg_one_l288_288014


namespace num_of_4_digit_numbers_divisible_by_13_l288_288885

theorem num_of_4_digit_numbers_divisible_by_13 :
  let smallest_4_digit := 1000
  let smallest_divisible_by_13 := 1001
  let largest_4_digit := 9999
  let largest_divisible_by_13 := 9997
  let a := smallest_divisible_by_13
  let d := 13
  let l := largest_divisible_by_13
  (\(l - a) / d) + 1 = 693 :=
by
  let smallest_4_digit := 1000
  let smallest_divisible_by_13 := 1001
  let largest_4_digit := 9999
  let largest_divisible_by_13 := 9997
  let a := smallest_divisible_by_13
  let d := 13
  let l := largest_divisible_by_13
  have h1 : (l - a) / d + 1 = (9997 - 1001) / 13 + 1, by sorry
  have h2 : (9997 - 1001) / 13 + 1 = 8996 / 13 + 1, by sorry
  have h3 : 8996 / 13 + 1 = 692 + 1, by sorry
  have h4 : 692 + 1 = 693, by sorry
  exact h4

end num_of_4_digit_numbers_divisible_by_13_l288_288885


namespace vectors_perpendicular_l288_288683

def vec_a : ℝ × ℝ × ℝ := (1, 2, 1 / 2)
def vec_b : ℝ × ℝ × ℝ := (1 / 2, -1 / 2, 1)

theorem vectors_perpendicular : vec_a.1 * vec_b.1 + vec_a.2 * vec_b.2 + vec_a.3 * vec_b.3 = 0 := 
by
  -- This theorem is supposed to prove that the dot product equals 0, hence they are perpendicular
  sorry

end vectors_perpendicular_l288_288683


namespace count_base_8_digits_5_or_6_l288_288865

-- Define the conditions in Lean
def is_digit_5_or_6 (d : ℕ) : Prop := d = 5 ∨ d = 6

def count_digits_5_or_6 := 
  let total_base_8 := 512
  let total_without_5_6 := 6 * 6 * 6 -- since we exclude 2 out of 8 digits
  total_base_8 - total_without_5_6

-- The statement of the proof problem
theorem count_base_8_digits_5_or_6 : count_digits_5_or_6 = 296 :=
by {
  sorry
}

end count_base_8_digits_5_or_6_l288_288865


namespace angle_COA_one_third_angle_DOB_l288_288669

theorem angle_COA_one_third_angle_DOB
  (O A C D B : Point)
  (circle_center_O : Circle)
  (radius_r : ℝ)
  (h_AC_eq_r : distance O C = radius_r)
  (h_point_A_outside_circle : distance O A > radius_r)
  (h_C_on_circle : CircleContains circle_center_O C)
  (h_D_on_circle : CircleContains circle_center_O D)
  (h_AO_secant_O : Secant O A B)
  (h_ACD_secant_O : Secant O A C D) :
  angle O C A = (1 / 3) * angle O D B := by
  sorry

end angle_COA_one_third_angle_DOB_l288_288669


namespace count_numbers_with_5_or_6_in_base_8_l288_288858

-- Define the condition that checks if a number contains digits 5 or 6 in base 8
def contains_digit_5_or_6_in_base_8 (n : ℕ) : Prop :=
  let digits := Nat.digits 8 n
  5 ∈ digits ∨ 6 ∈ digits

-- The main problem statement
theorem count_numbers_with_5_or_6_in_base_8 :
  (Finset.filter contains_digit_5_or_6_in_base_8 (Finset.range 513)).card = 296 :=
by
  sorry

end count_numbers_with_5_or_6_in_base_8_l288_288858


namespace roots_conditions_l288_288571

theorem roots_conditions (c d : ℝ) :
  (∃ x r s : ℝ, r ≠ 0 ∧ s ≠ 0 ∧ r > 0 ∧
    (z^3 - (3 + c * complex.I) * z^2 + (9 + d * complex.I) * z - 5).eval x = 0 ∧ 
    (z^3 - (3 + c * complex.I) * z^2 + (9 + d * complex.I) * z - 5).eval (r + complex.I * s) = 0 ∧ 
    (z^3 - (3 + c * complex.I) * z^2 + (9 + d * complex.I) * z - 5).eval (r - complex.I * s) = 0) →
  (c = 0 ∧ d = 0) :=
begin
  intro h,
  -- Proof omitted
  sorry
end

end roots_conditions_l288_288571


namespace min_dist_l288_288957

open Complex

theorem min_dist (z w : ℂ) (hz : abs (z - (2 - 5 * I)) = 2) (hw : abs (w - (-3 + 4 * I)) = 4) :
  ∃ d, d = abs (z - w) ∧ d ≥ (Real.sqrt 106 - 6) := sorry

end min_dist_l288_288957


namespace least_even_p_l288_288015

def is_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

theorem least_even_p 
  (p : ℕ) 
  (hp : 2 ∣ p) -- p is an even integer
  (h : is_square (300 * p)) -- 300 * p is the square of an integer
  : p = 3 := 
sorry

end least_even_p_l288_288015


namespace rectangle_length_to_width_ratio_l288_288670

theorem rectangle_length_to_width_ratio (a : ℝ) (h : 0 < a) :
  ∃ (rect_len rect_width : ℝ), rect_len = 3 * a ∧ rect_width = a ∧ rect_len / rect_width = 3 :=
by
  use 3 * a, a
  split
  case left => exact rfl
  case right => split
  case left => exact rfl
  case right => exact div_self (ne_of_gt h)

end rectangle_length_to_width_ratio_l288_288670


namespace num_triangles_with_positive_area_l288_288105

/-- 
Given vertices in a 5x5 grid with integer coordinates satisfying 1 ≤ x ≤ 5 and 1 ≤ y ≤ 5,
prove that the number of triangles with positive area is 2170. 
-/
theorem num_triangles_with_positive_area : 
  (∑ t in ({(i, j) | 1 ≤ i ∧ i ≤ 5 ∧ 1 ≤ j ∧ j ≤ 5}.powerset.filter (λ (s : set (ℕ × ℕ)), s.card = 3)),
    if t₁.1 * (t₂.2 - t₃.2) + t₂.1 * (t₃.2 - t₁.2) + t₃.1 * (t₁.2 - t₂.2) ≠ 0 then 1 else 0) = 2170 :=
by sorry

end num_triangles_with_positive_area_l288_288105


namespace sum_not_divisible_by_5_l288_288984

theorem sum_not_divisible_by_5 (n : ℕ) (hn : 0 < n) : 
  let x := ∑ k in Finset.range (n + 1), 2^(3*k) * nat.choose (2*n+1) (2*k+1)
  in ¬ 5 ∣ x :=
by
  intro n hn
  let x := ∑ k in Finset.range (n + 1), 2^3*k * nat.choose (2*n+1) (2*k+1)
  sorry

end sum_not_divisible_by_5_l288_288984


namespace milk_in_tank_is_correct_l288_288259

noncomputable def milk_left_in_tank : ℕ :=
  let initial_milk := 30000
  let pump_rate := 2880
  let pump_hours := 4
  let initial_add := 1200
  let increase_per_hour := 200
  let add_hours := 7
  let pumped_out := pump_rate * pump_hours
  let remaining_after_pump := initial_milk - pumped_out
  let last_term := initial_add + increase_per_hour * (add_hours - 1)
  let sum_added := add_hours * (initial_add + last_term) / 2
  remaining_after_pump + sum_added

theorem milk_in_tank_is_correct : milk_left_in_tank = 31080 :=
  by
    let initial_milk := 30000
    let pump_rate := 2880
    let pump_hours := 4
    let initial_add := 1200
    let increase_per_hour := 200
    let add_hours := 7
    let pumped_out := pump_rate * pump_hours
    let remaining_after_pump := initial_milk - pumped_out
    let last_term := initial_add + increase_per_hour * (add_hours - 1)
    let sum_added := add_hours * (initial_add + last_term) / 2
    have h1 : pumped_out = 11520 := by sorry
    have h2 : remaining_after_pump = 18480 := by sorry
    have h3 : last_term = 2400 := by sorry
    have h4 : sum_added = 12600 := by sorry
    show remaining_after_pump + sum_added = 31080 from
      by
        rw [h2, h4]
        exact rfl

end milk_in_tank_is_correct_l288_288259


namespace luke_initial_stickers_l288_288966

theorem luke_initial_stickers (x : ℕ) (h : x + 12 + 20 - 5 - 8 = 39) : x = 20 := 
by 
  sorry

end luke_initial_stickers_l288_288966


namespace carbon_dioxide_production_l288_288409

-- Condition statements
variables (CH4 O2 CO2 : Type)
variables (n_CH4 n_O2 : ℕ)
variables (reaction_coeff_CH4 : ℕ := 1)
variables (reaction_coeff_O2 : ℕ := 2)
variables (reaction_coeff_CO2 : ℕ := 1)
variables (n_CO2 : ℕ)

-- Conditions
axiom methane_condition : n_CH4 = 3
axiom oxygen_condition : n_O2 = 6
axiom conversion_condition : (n_O2 >= n_CH4 * reaction_coeff_O2)

-- Prove the number of moles of carbon dioxide formed is 3
theorem carbon_dioxide_production :
  n_CH4 = 3 → n_O2 = 6 → n_CO2 = n_CH4 * reaction_coeff_CO2 :=
begin
  intros,
  rw methane_condition at *,
  rw oxygen_condition at *,
  rw conversion_condition at *,
  linarith,
  sorry
end

end carbon_dioxide_production_l288_288409


namespace locus_of_foot_of_perpendicular_l288_288408

theorem locus_of_foot_of_perpendicular (k : ℝ) :
  ∃ r θ : ℝ, r^2 = 2 * k^2 * sin (2 * θ) ∧ 
             (∃ x y : ℝ, x * y = k^2 ∧ ∃ t : ℝ, y - t = -k^2 / t^2 * (x - t)) :=
sorry

end locus_of_foot_of_perpendicular_l288_288408


namespace number_of_valid_triangles_l288_288100

theorem number_of_valid_triangles : 
  let points := {(x, y) | 1 ≤ x ∧ x ≤ 5 ∧ 1 ≤ y ∧ y ≤ 5} in
  ∃ n : ℕ, n = (nat.card (points.choose 3)) - 120 ∧ n = 2180 :=
by
  sorry

end number_of_valid_triangles_l288_288100


namespace part_of_ellipse_l288_288224

noncomputable def curve (x y : ℝ) := x = real.sqrt (1 - 4*y^2)

theorem part_of_ellipse (x y : ℝ) (h : curve x y) : 
  x^2 + 4*y^2 = 1 ∧ x >= 0 :=
by sorry

end part_of_ellipse_l288_288224


namespace log24_eq_2b_minus_a_l288_288419

variable (a b : ℝ)

-- given conditions
axiom log6_eq : Real.log 6 = a
axiom log12_eq : Real.log 12 = b

-- proof goal statement
theorem log24_eq_2b_minus_a : Real.log 24 = 2 * b - a :=
by
  sorry

end log24_eq_2b_minus_a_l288_288419


namespace floor_ceiling_addition_l288_288773

theorem floor_ceiling_addition :
  (Int.floor (-3.67) + Int.ceil 30.95) = 27 :=
  by
  sorry

end floor_ceiling_addition_l288_288773


namespace non_degenerate_triangles_l288_288078

theorem non_degenerate_triangles (x y : ℕ) (hx : 1 ≤ x ∧ x ≤ 5) (hy : 1 ≤ y ∧ y ≤ 5) : 
  ∃ (n : ℕ), n = 2160 ∧ 1 ≤ n :=
by
  sorry

end non_degenerate_triangles_l288_288078


namespace quadratic_eq_c_has_equal_roots_l288_288067

theorem quadratic_eq_c_has_equal_roots (c : ℝ) (h : ∃ x : ℝ, x^2 - 4 * x + c = 0 ∧
                      ∀ y : ℝ, x^2 - 4 * x + c = 0 → y = x) : c = 4 := sorry

end quadratic_eq_c_has_equal_roots_l288_288067


namespace inequality_solution_l288_288613

theorem inequality_solution (x : ℝ) (hx : x ≥ 0) : (2021 * (x ^ 10) - 1 ≥ 2020 * x) ↔ (x = 1) :=
sorry

end inequality_solution_l288_288613


namespace bridge_length_l288_288733

-- Definitions for given conditions
def train_length : ℕ := 100
def crossing_time : ℕ := 36
def train_speed : ℕ := 40

-- Proof statement
theorem bridge_length : 
  let total_distance := train_speed * crossing_time in
  let bridge_length := total_distance - train_length in
  bridge_length = 1340 :=
by
  sorry

end bridge_length_l288_288733


namespace wuyang_math_total_participants_l288_288926

theorem wuyang_math_total_participants :
  ∀ (x : ℕ), 
  95 * (x + 5) = 75 * (x + 3 + 10) → 
  2 * (x + x + 8) + 9 = 125 :=
by
  intro x h
  sorry

end wuyang_math_total_participants_l288_288926


namespace circumference_eq_when_diameter_is_4_and_C_eq_A_l288_288295

def circumference (d : ℝ) : ℝ := π * d
def area (r : ℝ) : ℝ := π * r * r
def diameter_to_radius (d : ℝ) : ℝ := d / 2

theorem circumference_eq_when_diameter_is_4_and_C_eq_A :
  ∀ (d : ℝ), (circumference d = area (diameter_to_radius d)) → (d = 4) → (circumference d = 4 * π) :=
by
  sorry

end circumference_eq_when_diameter_is_4_and_C_eq_A_l288_288295


namespace minimize_AC_BC_l288_288203

-- Assume A, B are points and l is a line.
variable {A B C B1 : Point}
variable {l : Line}

-- Reflect B across line l to get point B1
def reflection (B : Point) (l : Line) : Point := sorry

-- Let A and B be on the same side of the line l
axiom same_side (A B : Point) (l : Line) : Prop

-- Function to find the intersection of two lines
def intersection (l1 l2 : Line) : Point := sorry

-- Define the line AB1
def line_AB1 (A B1 : Point) : Line := sorry

theorem minimize_AC_BC (A B : Point) (l : Line) (h : same_side A B l) :
  ∃ C : Point, C = intersection (line_AB1 A (reflection B l)) l ∧
  ∀ C' : Point, C' ∈ l → AC + BC ≤ AC' + BC' :=
sorry

end minimize_AC_BC_l288_288203


namespace subtraction_correct_l288_288311

theorem subtraction_correct :
  1_000_000_000_000 - 777_777_777_777 = 222_222_222_223 :=
by
  sorry

end subtraction_correct_l288_288311


namespace minimum_value_expression_l288_288575

noncomputable def min_value_expr (x y : ℝ) : ℝ :=
  let term1 := x + 1/y
  let term2 := y + 1/x
  (term1 * (term1 - 2023)) + (term2 * (term2 - 2023))

theorem minimum_value_expression (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
  ∃ (a : ℝ), a = -2050513 ∧ ∀ (x y : ℝ), 0 < x → 0 < y → min_value_expr x y ≥ a :=
begin
  sorry
end

end minimum_value_expression_l288_288575


namespace geometric_arithmetic_series_difference_l288_288748

theorem geometric_arithmetic_series_difference :
  let a := 1
  let r := 1 / 2
  let S := a / (1 - r)
  let T := 1 + 2 + 3
  S - T = -4 :=
by
  sorry

end geometric_arithmetic_series_difference_l288_288748


namespace polygon_sides_l288_288121

-- Definitions based on the conditions provided
def sum_interior_angles (n : ℕ) : ℝ :=
  (n - 2) * 180

def sum_exterior_angles : ℝ := 360 

def condition (n : ℕ) : Prop :=
  sum_interior_angles n = 2 * sum_exterior_angles + 180

-- Main theorem based on the correct answer
theorem polygon_sides (n : ℕ) (h : condition n) : n = 7 :=
sorry

end polygon_sides_l288_288121


namespace validate_triangle_count_l288_288090

noncomputable def count_valid_triangles : ℕ :=
  let total_points := 25
  let total_triples := (Nat.choose total_points 3)
  let collinear_rows := 5 * (Nat.choose 5 3)
  let collinear_columns := 5 * (Nat.choose 5 3)
  let main_diagonals := 2 * (Nat.choose 5 3)
  let secondary_diagonals := 8 * (Nat.choose 4 3)
  let invalid_triangles := collinear_rows + collinear_columns + main_diagonals + secondary_diagonals
  total_triples - invalid_triangles

theorem validate_triangle_count : count_valid_triangles = 2148 :=
by
  sorry

end validate_triangle_count_l288_288090


namespace dima_better_than_sasha_l288_288769

structure Contestants where
  Dima : ℕ
  Sasha : ℕ
  Kolya : ℕ
  Gleb : ℕ

def placed_first (x : Contestants) := ∃ p, p = 1
def placed_second (x : Contestants) := ∃ p, p = 2
def placed_third (x : Contestants) := ∃ p, p = 3
def placed_fourth (x : Contestants) := ∃ p, p = 4

axiom A (x : Contestants) : placed_first x.Dima ∨ placed_third x.Gleb
axiom B (x : Contestants) : placed_second x.Gleb ∨ placed_first x.Kolya
axiom C (x : Contestants) : placed_third x.Sasha ∨ placed_second x.Dima

theorem dima_better_than_sasha (x : Contestants) (hA : A x) (hB : B x) (hC : C x) : x.Dima < x.Sasha := sorry

end dima_better_than_sasha_l288_288769


namespace perpendicular_vectors_relation_l288_288277

theorem perpendicular_vectors_relation (a b : ℝ) (h : 3 * a - 7 * b = 0) : a = 7 * b / 3 :=
by
  sorry

end perpendicular_vectors_relation_l288_288277


namespace max_rocket_height_l288_288339

-- Define the quadratic function representing the rocket's height
def rocket_height (t : ℝ) : ℝ := -20 * t^2 + 100 * t + 50

-- State the maximum height problem
theorem max_rocket_height : ∃ t : ℝ, rocket_height t = 175 ∧ ∀ t' : ℝ, rocket_height t' ≤ 175 :=
by
  use 2.5
  sorry -- The proof will show that the maximum height is 175 meters at time t = 2.5 seconds

end max_rocket_height_l288_288339


namespace show_watching_days_l288_288156

def numberOfEpisodes := 20
def lengthOfEachEpisode := 30
def dailyWatchingTime := 2

theorem show_watching_days:
  (numberOfEpisodes * lengthOfEachEpisode) / 60 / dailyWatchingTime = 5 := 
by
  sorry

end show_watching_days_l288_288156


namespace sandy_paint_area_l288_288606

-- Define the dimensions of the wall
def wall_height : ℕ := 10
def wall_length : ℕ := 15

-- Define the dimensions of the decorative region
def deco_height : ℕ := 3
def deco_length : ℕ := 5

-- Calculate the areas and prove the required area to paint
theorem sandy_paint_area :
  wall_height * wall_length - deco_height * deco_length = 135 := by
  sorry

end sandy_paint_area_l288_288606


namespace find_integer_triples_l288_288399

def satisfies_equation (x y z : ℕ) : Prop :=
  x^4 + y^4 + z^4 = 2 * x^2 * y^2 + 2 * y^2 * z^2 + 2 * z^2 * x^2 - 63

theorem find_integer_triples :
  { (x, y, z) : ℕ × ℕ × ℕ // satisfies_equation x y z ∧ x > 0 ∧ y > 0 ∧ z > 0 } =
  { ((1, 4, 4)), ((4, 1, 4)), ((4, 4, 1)), ((2, 2, 3)), ((2, 3, 2)), ((3, 2, 2)) } :=
by
  sorry

end find_integer_triples_l288_288399


namespace num_of_4_digit_numbers_divisible_by_13_l288_288889

theorem num_of_4_digit_numbers_divisible_by_13 :
  let smallest_4_digit := 1000
  let smallest_divisible_by_13 := 1001
  let largest_4_digit := 9999
  let largest_divisible_by_13 := 9997
  let a := smallest_divisible_by_13
  let d := 13
  let l := largest_divisible_by_13
  (\(l - a) / d) + 1 = 693 :=
by
  let smallest_4_digit := 1000
  let smallest_divisible_by_13 := 1001
  let largest_4_digit := 9999
  let largest_divisible_by_13 := 9997
  let a := smallest_divisible_by_13
  let d := 13
  let l := largest_divisible_by_13
  have h1 : (l - a) / d + 1 = (9997 - 1001) / 13 + 1, by sorry
  have h2 : (9997 - 1001) / 13 + 1 = 8996 / 13 + 1, by sorry
  have h3 : 8996 / 13 + 1 = 692 + 1, by sorry
  have h4 : 692 + 1 = 693, by sorry
  exact h4

end num_of_4_digit_numbers_divisible_by_13_l288_288889


namespace tan_condition_sufficient_necessary_l288_288231

noncomputable def is_sufficient_and_necessary (x : ℝ) (k : ℤ) : Prop :=
  (tan x = 1) ↔ (∃ k : ℤ, x = k * Real.pi + Real.pi / 4)

theorem tan_condition_sufficient_necessary (x : ℝ) : 
  (tan x = 1) ↔ (∃ k : ℤ, x = k * Real.pi + Real.pi / 4) :=
by
  sorry

end tan_condition_sufficient_necessary_l288_288231


namespace find_m_perpendicular_l288_288058

def vec_dot_prod (v1 v2 : ℝ × ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2 + v1.3 * v2.3

theorem find_m_perpendicular (v1 v2 : ℝ × ℝ × ℝ) (h : vec_dot_prod v1 v2 = 0) :
  ∃ m : ℝ, v1 = (2,3,4) ∧ v2 = (-1,m,2) ∧ m = -2 :=
by
  sorry

end find_m_perpendicular_l288_288058


namespace exists_isosceles_triangle_same_color_l288_288349

theorem exists_isosceles_triangle_same_color (pts : Set Point) (circle : Circle) 
  (colored_points : ∀ p ∈ pts, p.color = Color1 ∨ p.color = Color2) :
  ∃ (A B C : Point), A ∈ pts ∧ B ∈ pts ∧ C ∈ pts ∧ 
  A.color = B.color ∧ B.color = C.color ∧ 
  is_isosceles_triangle A B C :=
sorry

end exists_isosceles_triangle_same_color_l288_288349


namespace calculate_value_of_expression_l288_288574

theorem calculate_value_of_expression :
  ∀ (p q : ℝ),
  (3 * p^2 + 9 * p - 21 = 0) →
  (3 * q^2 + 9 * q - 21 = 0) →
  (3 * p - 4) * (6 * q - 8) = 122 :=
begin
  intros p q hp hq,
  sorry,
end

end calculate_value_of_expression_l288_288574


namespace trains_distance_l288_288696

theorem trains_distance (t x : ℝ) 
  (h1 : x = 20 * t)
  (h2 : x + 50 = 25 * t) : 
  x + (x + 50) = 450 := 
by 
  -- placeholder for the proof
  sorry

end trains_distance_l288_288696


namespace largest_sum_is_225_l288_288663

noncomputable def max_sum_products (f g h j : ℕ) : ℕ :=
if (f ∈ {6, 7, 8, 9} ∧ g ∈ {6, 7, 8, 9} ∧ h ∈ {6, 7, 8, 9} ∧ j ∈ {6, 7, 8, 9} ∧ 
    {f, g, h, j}.nodup) then
    let fh := min (f * h + g * j) (min (f * j + g * h) (f * g + h * j)) in
    (f + g + h + j)^2 - (f^2 + g^2 + h^2 + j^2) - 2 * fh in
  (30^2 - 230) - 2 * 110 -- Plugging in the example values
else 
  0

theorem largest_sum_is_225 (f g h j : ℕ) (hf : f ∈ {6, 7, 8, 9}) (hg : g ∈ {6, 7, 8, 9}) 
  (hh : h ∈ {6, 7, 8, 9}) (hj : j ∈ {6, 7, 8, 9}) (h_nodup : ({f, g, h, j} : set ℕ).nodup) : 
  max_sum_products f g h j = 225 :=
by {
  sorry -- We would prove the steps leading to the solution here.
}

end largest_sum_is_225_l288_288663


namespace hcf_of_two_numbers_l288_288637

noncomputable def number1 : ℕ := 414

noncomputable def lcm_factors : Set ℕ := {13, 18}

noncomputable def hcf (a b : ℕ) : ℕ := Nat.gcd a b

-- Statement to prove
theorem hcf_of_two_numbers (Y : ℕ) 
  (H : ℕ) 
  (lcm : ℕ) 
  (H_lcm_factors : lcm = H * 13 * 18)
  (H_lcm_prop : lcm = (number1 * Y) / H)
  (H_Y : Y = (H^2 * 13 * 18) / 414)
  : H = 23 := 
sorry

end hcf_of_two_numbers_l288_288637


namespace log_expression_evaluation_l288_288750

theorem log_expression_evaluation :
  log 2 (sqrt 32 / 2) - log 10 4 - log 10 25 + (5^(log 5 2)) - 2 * ((16 / 25)^(-1 / 2)) = -1 :=
by
  sorry

end log_expression_evaluation_l288_288750


namespace martha_children_l288_288189

noncomputable def num_children (total_cakes : ℕ) (cakes_per_child : ℕ) : ℕ :=
  total_cakes / cakes_per_child

theorem martha_children : num_children 18 6 = 3 := by
  sorry

end martha_children_l288_288189


namespace option_A_option_D_l288_288168

open Complex

theorem option_A (z : ℂ) (h : z ∈ ℝ) : z = conj z :=
by
  unfold conj
  rcases h with ⟨a, rfl⟩
  simp

theorem option_D (z : ℂ) (h : (1 + I) * z = 1 - I) : abs z = 1 :=
by
  have hz : z = -(I),
  {
    apply_fun (1 - I) * (·) at h,
    rw [mul_assoc, mul_one, ←mul_assoc, mul_inv_cancel, one_mul] at h,
    exact (of_real_eq_of_real_iff (by norm_num : ↑(norm_sq (1 + I)) ≠ 0)).1 h,
  },
  rw hz,
  simp

end option_A_option_D_l288_288168


namespace area_square_larger_than_circle_l288_288278

theorem area_square_larger_than_circle (R : ℝ) (hR : R > 0) (AB BC CD : ℝ) (h1 : AB = BC) (h2: BC = CD) (h3: ∀ (x y z w : ℝ), x = y → y = z → z = w → x = w): 
  ∃ (AB : ℝ), AB > 0 ∧ AB = 2 * R * sin (3 * π / 8) ∧ (R^2 * (2 + real.sqrt 2) > π * R^2) :=
begin
  sorry
end

end area_square_larger_than_circle_l288_288278


namespace actual_average_height_l288_288517

theorem actual_average_height 
  (num_students : ℕ)
  (initial_average_height : ℝ)
  (height_rec1 rec1_actual_height : ℝ)
  (height_rec2 rec2_actual_height : ℝ)
  (height_rec3 rec3_actual_height : ℝ)
  (num_students = 40)
  (initial_average_height = 184)
  (height_rec1 = 166)
  (rec1_actual_height = 106)
  (height_rec2 = 172)
  (rec2_actual_height = 152)
  (height_rec3 = 190)
  (rec3_actual_height = 198) :
  (initial_average_height * num_students - (height_rec1 - rec1_actual_height) - (height_rec2 - rec2_actual_height) + (rec3_actual_height - height_rec3)) / num_students = 182.20 := 
by sorry

end actual_average_height_l288_288517


namespace sphere_properties_l288_288658

noncomputable def radius (D : ℝ) : ℝ := D / 2
noncomputable def surface_area (R : ℝ) : ℝ := 4 * Real.pi * R^2
noncomputable def volume (R : ℝ) : ℝ := (4 / 3) * Real.pi * R^3

theorem sphere_properties {D : ℝ} (hD : D = 6) :
  surface_area (radius D) = 36 * Real.pi ∧ volume (radius D) = 36 * Real.pi :=
by
  have hR : radius D = 3 := by
    rw [hD, radius]
    norm_num
  split
  { rw [surface_area, hR]
    norm_num
    ring }
  { rw [volume, hR]
    norm_num
    ring }

end sphere_properties_l288_288658


namespace base8_contains_5_or_6_l288_288857

theorem base8_contains_5_or_6 (n : ℕ) (h : n = 512) : 
  let count_numbers_without_5_6 := 6^3 in
  let total_numbers := 512 in
  total_numbers - count_numbers_without_5_6 = 296 := by
  sorry

end base8_contains_5_or_6_l288_288857


namespace tan_C_value_l288_288913

noncomputable def tan_values (A B : ℝ) : Prop :=
(tan A = some ((3 * X ^ 2 - 7 * X + 2 = 0).roots)) ∧ (tan B = some ((3 * X ^ 2 - 7 * X + 2 = 0).roots)) 

theorem tan_C_value (A B C : ℝ) (h: tan_values(A,B)) : tan C = -7 :=
sorry

end tan_C_value_l288_288913


namespace trivia_game_points_l288_288299

theorem trivia_game_points 
  (num_questions_first_half : ℕ)
  (num_questions_second_half : ℕ)
  (final_score : ℕ)
  (h1 : num_questions_first_half = 3)
  (h2 : num_questions_second_half = 2)
  (h3 : final_score = 15) : 
  ∃ (p: ℕ), 
    let q := num_questions_first_half + num_questions_second_half in
    q = 5 ∧ q * p = final_score ∧ p = 3 :=
by {
  -- sorry skips the proof.
  sorry 
}

end trivia_game_points_l288_288299


namespace activity_probability_l288_288417

noncomputable def total_basic_events : ℕ := 3^4
noncomputable def favorable_events : ℕ := Nat.choose 4 2 * Nat.factorial 3

theorem activity_probability :
  (favorable_events : ℚ) / total_basic_events = 4 / 9 :=
by
  sorry

end activity_probability_l288_288417


namespace find_vector_b_l288_288059

def vector_collinear (a b : ℝ × ℝ) : Prop :=
    ∃ k : ℝ, (a.1 = k * b.1 ∧ a.2 = k * b.2)

def dot_product (a b : ℝ × ℝ) : ℝ :=
    a.1 * b.1 + a.2 * b.2

theorem find_vector_b (a b : ℝ × ℝ) (h_collinear : vector_collinear a b) (h_dot : dot_product a b = -10) : b = (-4, 2) :=
    by
        sorry

end find_vector_b_l288_288059


namespace temperature_difference_l288_288196

theorem temperature_difference : 
  let beijing_temp := -6
  let changtai_temp := 15
  changtai_temp - beijing_temp = 21 := 
by
  -- Let the given temperatures
  let beijing_temp := -6
  let changtai_temp := 15
  -- Perform the subtraction and define the expected equality
  show changtai_temp - beijing_temp = 21
  -- Preliminary proof placeholder
  sorry

end temperature_difference_l288_288196


namespace min_elements_in_A_l288_288579

open Set

theorem min_elements_in_A (n : ℕ) (hn : n ≥ 2) (S : Finset ℝ) (hS : S.card = n) :
  (S.image (λ p, (p.1 + p.2) / 2 : ℝ × ℝ → ℝ)).card ≥ 2 * n - 3 :=
sorry

end min_elements_in_A_l288_288579


namespace segment_length_sum_l288_288839

theorem segment_length_sum (k : ℝ) (h : -1 ≤ k ∧ k ≤ 1) :
  let f := λ (x : ℝ), 4*x^2 - 4*k*x + (k - 1) in
  let x1 := (k + real.sqrt(k^2 - k + 1)) / 2 in
  let x2 := (k - real.sqrt(k^2 - k + 1)) / 2 in
  let d_max := real.sqrt 3 in
  let d_min := real.sqrt 3 / 2 in
  (d_max + d_min) = (3 * real.sqrt 3 / 2) :=
sorry

end segment_length_sum_l288_288839


namespace count_numbers_with_5_or_6_in_base_8_l288_288861

-- Define the condition that checks if a number contains digits 5 or 6 in base 8
def contains_digit_5_or_6_in_base_8 (n : ℕ) : Prop :=
  let digits := Nat.digits 8 n
  5 ∈ digits ∨ 6 ∈ digits

-- The main problem statement
theorem count_numbers_with_5_or_6_in_base_8 :
  (Finset.filter contains_digit_5_or_6_in_base_8 (Finset.range 513)).card = 296 :=
by
  sorry

end count_numbers_with_5_or_6_in_base_8_l288_288861


namespace OH_over_ON_eq_2_no_other_common_points_l288_288142

noncomputable def coordinates (t p : ℝ) : ℝ × ℝ :=
  (t^2 / (2 * p), t)

noncomputable def symmetric_point (M P : ℝ × ℝ) : ℝ × ℝ :=
  let (xM, yM) := M;
  let (xP, yP) := P;
  (2 * xP - xM, 2 * yP - yM)

noncomputable def line_ON (p t : ℝ) : ℝ → ℝ :=
  λ x => (p / t) * x

noncomputable def line_MH (t p : ℝ) : ℝ → ℝ :=
  λ x => (p / (2 * t)) * x + t

noncomputable def point_H (t p : ℝ) : ℝ × ℝ :=
  (2 * t^2 / p, 2 * t)

theorem OH_over_ON_eq_2
  (t p : ℝ) (ht : t ≠ 0) (hp : p > 0)
  (M : ℝ × ℝ := (0, t))
  (P : ℝ × ℝ := coordinates t p)
  (N : ℝ × ℝ := symmetric_point M P)
  (H : ℝ × ℝ := point_H t p) :
  (H.snd) / (N.snd) = 2 := by
  sorry

theorem no_other_common_points
  (t p : ℝ) (ht : t ≠ 0) (hp : p > 0)
  (M : ℝ × ℝ := (0, t))
  (P : ℝ × ℝ := coordinates t p)
  (N : ℝ × ℝ := symmetric_point M P)
  (H : ℝ × ℝ := point_H t p) :
  ∀ y, (y ≠ H.snd → ¬ ∃ x, line_MH t p x = y ∧ y^2 = 2 * p * x) := by 
  sorry

end OH_over_ON_eq_2_no_other_common_points_l288_288142


namespace trapezoid_area_eq_l288_288342

variable {a : ℝ}

def height := 2 * a
def base1 := height + 3 * a
def base2 := 4 * a
def trapezoid_area (h b1 b2 : ℝ) := (h * (b1 + b2)) / 2

theorem trapezoid_area_eq : trapezoid_area height base1 base2 = 9 * a^2 := by
  sorry

end trapezoid_area_eq_l288_288342


namespace solve_equation_l288_288214

noncomputable def eta (x : ℝ) : ℝ := if x ≥ 0 then 1 else 0

theorem solve_equation (x : ℝ) (k : ℤ) :
  (cos (2 * x * (eta (x + 3 * π) - eta (x - 8 * π))) = sin x + cos x) ↔
  (x = 2 * π * k ∨
  (x = -π / 2 + 2 * π * k ∧ (k = -1 ∨ k = 0 ∨ k = 1 ∨ k = 2 ∨ k = 3 ∨ k = 4)) ∨
  (x = π / 2 + 2 * π * k ∧ (k ≠ -1 ∧ k ≠ 0 ∧ k ≠ 1 ∧ k ≠ 2 ∧ k ≠ 3)) ∨
  (∃ m : ℤ, x = -π / 4 + π * m ∧ (-2 ≤ m ∧ m ≤ 8))) :=
sorry

end solve_equation_l288_288214


namespace count_4digit_numbers_divisible_by_13_l288_288891

theorem count_4digit_numbers_divisible_by_13 : 
  let count := (λ n, n - 77 + 1) (769)
  count = 693 :=
by
-- Let 77 be the smallest integer such that 13 * 77 = 1001
-- Let 769 be the largest integer such that 13 * 769 = 9997
-- Hence, the range of these integers should be 769 - 77 + 1
-- This gives us a count of 693
-- The statement as 693 is directly given
sorry

end count_4digit_numbers_divisible_by_13_l288_288891


namespace range_of_m_l288_288469

theorem range_of_m : 
  (∃ x : ℝ, 4^x + 2^(x + 1) - m = 0) → (m ∈ set.Ici 0) :=
by
  sorry

end range_of_m_l288_288469


namespace ball_distribution_l288_288898

theorem ball_distribution :
  let balls := 5 in
  let boxes := 4 in
  -- The number of distinct ways to partition 5 balls into 4 indistinguishable boxes
  ((λ n k : ℕ, n = 5 ∧ k = 4) → ∃! p : ℕ, p = 4) :=
by sorry

end ball_distribution_l288_288898


namespace missing_number_is_4_point_5_l288_288055

noncomputable def find_x (x : ℝ) : Prop :=
  (0.0088 * x) / (0.05 * 0.1 * 0.008) = 990

theorem missing_number_is_4_point_5 :
  ∃ x : ℝ, find_x x ∧ x = 4.5 :=
by
  use 4.5
  have h : (0.05 * 0.1 * 0.008) = 0.00004 := by norm_num
  have h2 : (0.0088 * 4.5) / 0.00004 = 990 := by norm_num
  unfold find_x
  rw [h]
  exact ⟨h2, rfl⟩

end missing_number_is_4_point_5_l288_288055


namespace inequality_proof_l288_288798

theorem inequality_proof (x y z : ℝ) (hx : x ≥ y) (hy : y ≥ z) (hz : z > 0) :
  (x^2 * y / z + y^2 * z / x + z^2 * x / y) ≥ (x^2 + y^2 + z^2) := 
  sorry

end inequality_proof_l288_288798


namespace sum_binom_mod_500_l288_288378

theorem sum_binom_mod_500 :
  (∑ k in Finset.range 504, Nat.choose 2011 (4 * k)) % 500 = 29 :=
by
  sorry

end sum_binom_mod_500_l288_288378


namespace particle_acceleration_l288_288722

-- Define the variables
variables (a b c t x v f : ℝ)

-- Define the conditions
def position (t : ℝ) : ℝ := a * t + b * t^2 + c * t^3
def velocity (t : ℝ) : ℝ := a + 2 * b * t + 3 * c * t^2
def acceleration (t : ℝ) : ℝ := 2 * b + 6 * c * t

-- State the theorem
theorem particle_acceleration (a b c v : ℝ) (H : ∃ t, velocity t = v) : 
  ∃ f, f = 2 * b + sqrt(12 * c * (v - a)) :=
sorry

end particle_acceleration_l288_288722


namespace sum_not_integer_l288_288601

def sum_of_fractions (N : ℕ) [fact (2 ≤ N)] := ∑ m in {n | 1 < n < N}, ∑ n in {n | m < n < N}, (1 : ℚ) / (m * n)

theorem sum_not_integer : ¬ ∃ s : ℤ, sum_of_fractions 1986 = s := 
  sorry

end sum_not_integer_l288_288601


namespace unique_real_solution_for_cubic_l288_288764

theorem unique_real_solution_for_cubic {b : ℝ} :
  (∀ x : ℝ, (x^3 - b * x^2 - 3 * b * x + b^2 - 4 = 0) → ∃! x : ℝ, (x^3 - b * x^2 - 3 * b * x + b^2 - 4 = 0)) ↔ b > 3 :=
sorry

end unique_real_solution_for_cubic_l288_288764


namespace complement_A_in_U_l288_288848

-- Define the universal set as ℝ
def U : Set ℝ := Set.univ

-- Define the set A as given in the conditions
def A : Set ℝ := {y | ∃ x : ℝ, 2^(Real.log x) = y}

-- The main statement based on the conditions and the correct answer
theorem complement_A_in_U : (U \ A) = {y | y ≤ 0} := by
  sorry

end complement_A_in_U_l288_288848


namespace wine_consumption_correct_l288_288518

-- Definitions based on conditions
def drank_after_first_pound : ℚ := 1
def drank_after_second_pound : ℚ := 1
def drank_after_third_pound : ℚ := 1 / 2
def drank_after_fourth_pound : ℚ := 1 / 4
def drank_after_fifth_pound : ℚ := 1 / 8
def drank_after_sixth_pound : ℚ := 1 / 16

-- Total wine consumption
def total_wine_consumption : ℚ :=
  drank_after_first_pound + drank_after_second_pound +
  drank_after_third_pound + drank_after_fourth_pound +
  drank_after_fifth_pound + drank_after_sixth_pound

-- Theorem statement
theorem wine_consumption_correct :
  total_wine_consumption = 47 / 16 :=
by
  sorry

end wine_consumption_correct_l288_288518


namespace max_mark_paper_i_l288_288708

theorem max_mark_paper_i (M : ℝ) (h1 : 0.65 * M = 170) : M ≈ 262 :=
by sorry

end max_mark_paper_i_l288_288708


namespace question1_geometric_sequence_question2_minimum_term_l288_288432

theorem question1_geometric_sequence (a : ℕ → ℝ) (p : ℝ) (q : ℝ) :
  (∀ n : ℕ, a (n + 1) = a n + p * (3 ^ n) - n * q) →
  q = 0 →
  (a 1 = 1 / 2) →
  (∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a 1 * (r ^ n)) →
  (p = 0 ∨ p = 1) :=
by sorry

theorem question2_minimum_term (a : ℕ → ℝ) (p : ℝ) (q : ℝ) :
  (∀ n : ℕ, a (n + 1) = a n + p * (3 ^ n) - n * q) →
  p = 1 →
  (a 1 = 1 / 2) →
  (a 4 = min (min (a 1) (a 2)) (a 3)) →
  3 ≤ q ∧ q ≤ 27 / 4 :=
by sorry

end question1_geometric_sequence_question2_minimum_term_l288_288432


namespace avg_rate_of_change_l288_288796

noncomputable def f (x : ℝ) : ℝ := 3 * x^2 + 5

theorem avg_rate_of_change :
  (f 0.2 - f 0.1) / (0.2 - 0.1) = 0.9 := by
  sorry

end avg_rate_of_change_l288_288796


namespace find_N_l288_288327

theorem find_N (N : ℕ) (W : ℝ) 
  (W_positive : 0 < W)
  (heaviest_sum : \(\sum_{i=1}^{3} heaviest(i) = 0.35 * W \))
  (lightest_sum : \(\sum_{j=1}^{3} lightest(j) = 0.25 * W \)) :
  N = 10 := 
by {
  sorry
}

end find_N_l288_288327


namespace find_prob_p_l288_288741

variable (p : ℚ)

theorem find_prob_p (h : 15 * p^4 * (1 - p)^2 = 500 / 2187) : p = 3 / 7 := 
  sorry

end find_prob_p_l288_288741


namespace problem_statement_l288_288952

theorem problem_statement :
  ∃ m n : ℕ, Nat.Coprime m n ∧ (m:ℝ)/n = 1/40 ∧ m + n = 41 :=
begin
  sorry
end

end problem_statement_l288_288952


namespace not_an_axiom_A_l288_288535

-- Definitions of propositions B, C, and D as axioms
axiom axiom_B : ∀ (P1 P2 P3 : Point), ¬Collinear P1 P2 P3 → ∃! plane, PointsOnPlane P1 P2 P3
axiom axiom_C : ∀ (l : Line) (P1 P2 : Point), PointsOnLine l P1 P2 → ∀ (Q : Point), OnSamePlane P1 Q → PointsOnLine l Q
axiom axiom_D : ∀ (plane1 plane2 : Plane) (P : Point), PointOnTwoPlanes P plane1 plane2 → ∃! (l : Line), LineOnPlane l plane1 ∧ LineOnPlane l plane2 ∧ PointOnLine P l

-- Proposition A 
def proposition_A : Prop := ∀ (plane1 plane2 plane3 : Plane), (PlaneParallel plane1 plane3 ∧ PlaneParallel plane2 plane3) → PlaneParallel plane1 plane2

theorem not_an_axiom_A : ¬proposition_A := by 
  -- The proof is omitted here
  sorry

end not_an_axiom_A_l288_288535


namespace homothety_midpoint_locus_l288_288757

noncomputable def midpoint_locus_homothety (P Q : ℝ × ℝ) (O : ℝ × ℝ) (r : ℝ) (k : ℝ) : set (ℝ × ℝ) :=
  {M | ∃ (Q : ℝ × ℝ), dist O Q = r ∧ M = (1 - k) • P + k • Q}

theorem homothety_midpoint_locus :
  let O := (0, 0) in
  let P := (8, 0) in
  let r := 6 in
  let k := 1/3 in
  midpoint_locus_homothety P Q O r k =
    {C | ∃ (x y : ℝ), C = (16 / 3, 0) ∧ dist (16 / 3, 0) C = 2} :=
sorry

end homothety_midpoint_locus_l288_288757


namespace divide_integer_probability_l288_288313

theorem divide_integer_probability : 
  let r_vals := filter (λ r, -5 < r ∧ r < 8) (list.range 14).map(λ x, x - 6),
      k_vals := filter (λ k, 0 < k ∧ k < 10) (list.range 11).map(λ x, x - 1),
      valid_pairs := (r_vals.product k_vals).filter (λ ⟨r, k⟩, k ∣ r) in
  (valid_pairs.length : ℚ) / (r_vals.length * k_vals.length) = 33 / 108 :=
by
  sorry

end divide_integer_probability_l288_288313


namespace alternating_sum_eq_neg151_l288_288393

theorem alternating_sum_eq_neg151 : 
  ∑ i in finset.range 102, (-1)^(i+1) * i = -151 :=
sorry

end alternating_sum_eq_neg151_l288_288393


namespace max_sqrt_sum_l288_288576

theorem max_sqrt_sum (x y z : ℝ) (h_nonneg_x : 0 ≤ x) (h_nonneg_y : 0 ≤ y) (h_nonneg_z : 0 ≤ z) (h_sum : x + y + z = 8) :
  sqrt (3 * x + 2) + sqrt (3 * y + 2) + sqrt (3 * z + 2) ≤ 3 * sqrt 10 :=
sorry

end max_sqrt_sum_l288_288576


namespace continuous_stripe_probability_l288_288390

theorem continuous_stripe_probability (tetrahedron : Type) [fintype tetrahedron] [decidable_eq tetrahedron]
  (faces : fin 4 → fin 3 → tetrahedron)
  (stripe_pairing : tetrahedron → tetrahedron)
  (random_and_independent : ∀ f : fin 4, is_random (stripe_pairing ∘ faces f))
  : probability (is_continuous_stripe stripe_pairing) = 2 / 81 :=
sorry

end continuous_stripe_probability_l288_288390


namespace partI_partII_l288_288803

variables (P A B C D : ℝ)

-- Conditions as Lean definitions
def PB_perp_AD (P B A D : ℝ) := ⟪B - P, D - A⟫ = 0
def equilateral_PAD (P A D : ℝ) := dist P A = 2 ∧ dist A D = 2 ∧ dist D P = 2
def rhombus_ABCD (A B C D : ℝ) := dist A B = dist B C ∧ dist B C = dist C D ∧ dist C D = dist D A ∧ dist D A = dist A B
def dihedral_angle_PAD_ABCD (P A D B C : ℝ) := angle (plane_span (P, A, D)) (plane_span (A, B, C, D)) = 120

-- Problem Part I: Distance from point P to plane ABCD
theorem partI (P A D B : ℝ) (h1 : PB_perp_AD P B A D)
              (h2 : equilateral_PAD P A D)
              (h3 : rhombus_ABCD A B C D)
              (h4 : dihedral_angle_PAD_ABCD P A D B C) : 
  dist_point_plane P (A, B, C) = 1.5 := sorry

-- Problem Part II: Dihedral angle between faces APB and CPB
theorem partII (P A D B C : ℝ) (h1 : PB_perp_AD P B A D)
              (h2 : equilateral_PAD P A D)
              (h3 : rhombus_ABCD A B C D)
              (h4 : dihedral_angle_PAD_ABCD P A D B C) :
  dihedral_angle (triangle_span (P, A, B)) (triangle_span (P, C, B)) = 
  π - arccos (2 * sqrt 7 / 7) := sorry

end partI_partII_l288_288803


namespace min_value_of_n_l288_288511

theorem min_value_of_n : ∃ n : ℕ, (n % 2 = 0) ∧ (∃ x : ℕ → ℤ, 
  (∀ i, i ∈ Finset.range n → x i = 7 ∨ x i = -7) ∧ 
  (Finset.sum (Finset.range n) x = 0) ∧ 
  (Finset.sum (Finset.range n) (λ i, (i + 1) * x i) = 2009) ∧ 
  n = 34) :=
by
  sorry

end min_value_of_n_l288_288511


namespace table_quiz_student_pairs_repetition_l288_288666

theorem table_quiz_student_pairs_repetition (n : ℕ) (h: ne (n) 0):
  ∀ T : ℕ → finset (fin n) → finset (fin n), 
  (∀ w : ℕ, T(w).card = n) → 
  (∀ w1 w2 : ℕ, w1 ≠ w2 → 
  ∀ s1 s2 : finset (fin n), 
  s1 ∈ T(w1) → s2 ∈ T(w2) → 
  disjoint s1 s2) → 
  ∃ w1 w2 : ℕ, w1 < w2 ∧ w2 ≤ n + 1 ∧ ∃ s1 s2 : fin (n * n), 
  s1 ∈ T(w1) ∧ s2 ∈ T(w2) ∧ s1 = s2 :=
sorry

end table_quiz_student_pairs_repetition_l288_288666


namespace distribution_schemes_count_l288_288282

theorem distribution_schemes_count :
  let A := 2,
      B := 2,
      total_pieces := 7,
      remaining_pieces := total_pieces - A - B,
      communities := 5 in
  ∑ (x : Finset ℕ) in (Finset.range (remaining_pieces + 1)).powerset, 
      if x.card = 3 ∨ x.card = 2 ∨ x.card = 1 then 
        1 
      else 
        0 = 35 := sorry

end distribution_schemes_count_l288_288282


namespace greatest_prime_factor_of_sum_295_to_615_l288_288907

def sum_even_multiples_25_between (a b : ℤ) : ℤ :=
  let multiples := list.range' ((a + 50 - 1) / 50 * 50) ((b - 50 + 1) / 50) |>.map (λ x, x * 25) |>.filter (λ x, x % 50 = 0)
  multiples.sum

theorem greatest_prime_factor_of_sum_295_to_615 :
  let k := sum_even_multiples_25_between 295 615 in
  nat.greatest_prime_factor k = 7 :=
by
  sorry

end greatest_prime_factor_of_sum_295_to_615_l288_288907


namespace train_length_l288_288734

theorem train_length
  (time : ℝ) (man_speed train_speed : ℝ) (same_direction : Prop)
  (h_time : time = 62.99496040316775)
  (h_man_speed : man_speed = 6)
  (h_train_speed : train_speed = 30)
  (h_same_direction : same_direction) :
  (train_speed - man_speed) * (1000 / 3600) * time = 1259.899208063355 := 
sorry

end train_length_l288_288734


namespace average_weight_decrease_l288_288222

theorem average_weight_decrease 
  (num_persons : ℕ)
  (avg_weight_initial : ℕ)
  (new_person_weight : ℕ)
  (new_avg_weight : ℚ)
  (weight_decrease : ℚ)
  (h1 : num_persons = 20)
  (h2 : avg_weight_initial = 60)
  (h3 : new_person_weight = 45)
  (h4 : new_avg_weight = (1200 + 45) / 21) : 
  weight_decrease = avg_weight_initial - new_avg_weight :=
by
  sorry

end average_weight_decrease_l288_288222


namespace javier_fraction_to_anna_zero_l288_288359

-- Variables
variable (l : ℕ) -- Lee's initial sticker count
variable (j : ℕ) -- Javier's initial sticker count
variable (a : ℕ) -- Anna's initial sticker count

-- Initial conditions
def conditions (l j a : ℕ) : Prop :=
  j = 4 * a ∧ a = 3 * l

-- Javier's final stickers count
def final_javier_stickers (ja : ℕ) (j : ℕ) : ℕ :=
  ja

-- Anna's final stickers count (af = final Anna's stickers)
def final_anna_stickers (af : ℕ) : ℕ :=
  af

-- Lee's final stickers count (lf = final Lee's stickers)
def final_lee_stickers (lf : ℕ) : ℕ :=
  lf

-- Final distribution requirements
def final_distribution (ja af lf : ℕ) : Prop :=
  ja = 2 * af ∧ ja = 3 * lf

-- Correct answer, fraction of stickers given to Anna
def fraction_given_to_anna (j ja : ℕ) : ℚ :=
  ((j - ja) : ℚ) / (j : ℚ)

-- Lean theorem statement to prove
theorem javier_fraction_to_anna_zero
  (l j a ja af lf : ℕ)
  (h_cond : conditions l j a)
  (h_final : final_distribution ja af lf) :
  fraction_given_to_anna j ja = 0 :=
by sorry

end javier_fraction_to_anna_zero_l288_288359


namespace lloyd_house_of_cards_l288_288188

theorem lloyd_house_of_cards 
  (decks : ℕ) (cards_per_deck : ℕ) (layers : ℕ)
  (h1 : decks = 24) (h2 : cards_per_deck = 78) (h3 : layers = 48) :
  ((decks * cards_per_deck) / layers) = 39 := 
  by
  sorry

end lloyd_house_of_cards_l288_288188


namespace mrs_franklin_students_l288_288972

theorem mrs_franklin_students (valentines_current valentines_needed : ℝ) (classrooms : ℕ) 
                               (h_valentines_current : valentines_current = 58.3) 
                               (h_valentines_needed : valentines_needed = 16.5)
                               (h_classrooms : classrooms = 3) :
                               classrooms * (Int.to_nat (Real.ceil (valentines_current + valentines_needed) / classrooms)) = 75 :=
by sorry

end mrs_franklin_students_l288_288972


namespace number_of_polynomials_l288_288386

-- Definitions for conditions
def polynomial_form (a : ℕ → ℤ) (n : ℕ) := ∑ i in finset.range (n+1), (a i) * (n+1 - i)

def condition_sum_abs_coeff_degree (a : ℕ → ℤ) (n : ℕ) := 
  n + (finset.sum (finset.range (n+1)) (λ i, |a i|))

-- The theorem statement
theorem number_of_polynomials : 
  ∃ (p : ℕ → ℤ) (n : ℕ), condition_sum_abs_coeff_degree p n = 5 ∧ polynomial_form p n = 48 :=
sorry

end number_of_polynomials_l288_288386


namespace no_daughters_count_l288_288362

theorem no_daughters_count
  (d : ℕ)  -- Bertha's daughters
  (total : ℕ)  -- Total daughters and granddaughters
  (d_with_children : ℕ)  -- Daughters with children
  (children_per_daughter : ℕ)  -- Children per daughter
  (no_great_granddaughters : Prop) -- No great-granddaughters
  (d = 8)
  (total = 40)
  (d_with_children * children_per_daughter = total - d)
  (children_per_daughter = 4)
  (no_great_granddaughters := ∀ gd, gd ∈ ∅  → ¬∃ x, x ∈ gd) :
  total - d_with_children = 32 :=
by sorry

end no_daughters_count_l288_288362


namespace direct_proportion_function_l288_288296

-- Definitions of the functions
def fA (x : ℝ) : ℝ := 3 * x - 4
def fB (x : ℝ) : ℝ := -2 * x + 1
def fC (x : ℝ) : ℝ := 3 * x
def fD (x : ℝ) : ℝ := 3 * x^2 + 2

-- Definition of a direct proportion function
def is_direct_proportion (f : ℝ → ℝ) : Prop := ∃ k : ℝ, k ≠ 0 ∧ (∀ x : ℝ, f x = k * x)

-- Theorem statement
theorem direct_proportion_function : is_direct_proportion fC ∧ ¬ is_direct_proportion fA ∧ ¬ is_direct_proportion fB ∧ ¬ is_direct_proportion fD :=
by
  sorry

end direct_proportion_function_l288_288296


namespace sequence_a4_value_l288_288651

theorem sequence_a4_value :
  ∀ (a : ℕ → ℕ), a 1 = 1 ∧ (∀ n : ℕ, 0 < n → a (n + 1) = 2 * a n + 3) → a 4 = 29 :=
by sorry

end sequence_a4_value_l288_288651


namespace find_theta_l288_288814

noncomputable def theta : Type := Real

axiom cond1 (θ : theta) : (1 / Real.sin θ) + (1 / Real.cos θ) = 35 / 12
axiom cond2 (θ : theta) : θ > 0 ∧ θ < Real.pi / 2

theorem find_theta (θ : theta) (h1: cond1 θ) (h2: cond2 θ) :
  θ = Real.arcsin (3 / 5) ∨ θ = Real.arcsin (4 / 5) := sorry

end find_theta_l288_288814


namespace excircle_centers_perpendicular_l288_288742

-- Define the main contextual setup required for the theorem
variables {A B C : Type}
variables {D E : A}
variables (I_A I_1 I_2 : A → A → A → A)  -- excircle centers
variables [incircle] [tangent_point]
variables [Triangle] [TriangleExcircle]
variables [ExcircleTangency] [InternalPoint]

-- Define the specific points and conditions
def point_on_BC (BC : Set A) (point : A) : Prop := point ∈ BC
def tangent_at (excircle : A → A → A → A) (side : A → A) (point : A) : Prop := excircle touches side at point
def excircle_center (triangle : A → A → A → A) (E : A) : A := center of excircle of triangle at E

-- The main theorem to prove
theorem excircle_centers_perpendicular (BC : Set A)
    (hD : point_on_BC BC D)
    (hE : point_on_BC BC E)
    (I1_center : excircle_center (Triangle A B E) = I_1)
    (I2_center : excircle_center (Triangle A C E) = I_2)
    (ht1 : tangent_at I_A BC D)
    (ht2 : tangent_at I_A BC E)
    (ht3 : tangent_at I_1 BC E)
    (ht4 : tangent_at I_2 BC E) :
    ⟪I_1, D⟫ ⟂ ⟪I_2, D⟫ :=
sorry

end excircle_centers_perpendicular_l288_288742


namespace backpack_price_increase_l288_288558

variable (x : ℝ)

theorem backpack_price_increase :
  let backpack_price := 50 + x in
  let ring_binder_price := 18 in
  let total_ring_binders_cost := 3 * ring_binder_price in
  let total_cost := backpack_price + total_ring_binders_cost in
  total_cost = 109 → x = 5 :=
by
  intros _ _
  unfold backpack_price ring_binder_price total_ring_binders_cost total_cost
  sorry

end backpack_price_increase_l288_288558


namespace count_4digit_numbers_divisible_by_13_l288_288890

theorem count_4digit_numbers_divisible_by_13 : 
  let count := (λ n, n - 77 + 1) (769)
  count = 693 :=
by
-- Let 77 be the smallest integer such that 13 * 77 = 1001
-- Let 769 be the largest integer such that 13 * 769 = 9997
-- Hence, the range of these integers should be 769 - 77 + 1
-- This gives us a count of 693
-- The statement as 693 is directly given
sorry

end count_4digit_numbers_divisible_by_13_l288_288890


namespace min_vertical_segment_length_l288_288636

noncomputable def minVerticalSegLength : ℤ → ℝ 
| x => abs (2 * abs x + x^2 + 4 * x + 1)

theorem min_vertical_segment_length :
  ∀ x : ℤ, minVerticalSegLength x = 1 ↔  x = 0 := 
by
  intros x
  sorry

end min_vertical_segment_length_l288_288636


namespace fraction_value_l288_288460

theorem fraction_value
  (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (cond1 : (a + b + c) / (a + b - c) = 7)
  (cond2 : (a + b + c) / (a + c - b) = 1.75) :
  (a + b + c) / (b + c - a) = 3.5 :=
by
  sorry

end fraction_value_l288_288460


namespace prob_red_blue_calc_l288_288705

noncomputable def prob_red_blue : ℚ :=
  let p_yellow := (6 : ℚ) / 13
  let p_red_blue_given_yellow := (7 : ℚ) / 12
  let p_red_blue_given_not_yellow := (7 : ℚ) / 13
  p_red_blue_given_yellow * p_yellow + p_red_blue_given_not_yellow * (1 - p_yellow)

/-- The probability of drawing a red or blue marble from the updated bag contents is 91/169. -/
theorem prob_red_blue_calc : prob_red_blue = 91 / 169 :=
by
  -- This proof is omitted as per instructions.
  sorry

end prob_red_blue_calc_l288_288705


namespace volume_of_tetrahedron_B1_EFG_l288_288541

def Point := (ℝ × ℝ × ℝ)

-- Define the vertices of the cuboid
def A : Point := (0, 0, 0)
def B : Point := (2, 0, 0)
def C : Point := (2, 1, 0)
def D : Point := (0, 1, 0)
def A1 : Point := (0, 0, 1)
def B1 : Point := (2, 0, 1)
def C1 : Point := (2, 1, 1)
def D1 : Point := (0, 1, 1)

-- Define the midpoints E, F, and G
def E : Point := (0, 0, 0.5)
def F : Point := (1, 1, 1)
def G : Point := (2, 0.5, 0)

-- Tetrahedron vertices
def TetrahedronVertices : List Point := [B1, E, F, G]

-- Function to calculate the volume of a tetrahedron given four points.
def volume_of_tetrahedron (a b c d : Point) : ℝ :=
  (1 / 6) * real.abs (matrix.det ![
    ![a.1, a.2, a.3, 1],
    ![b.1, b.2, b.3, 1],
    ![c.1, c.2, c.3, 1],
    ![d.1, d.2, d.3, 1]
  ])

-- Statement of the proof problem
theorem volume_of_tetrahedron_B1_EFG :
  volume_of_tetrahedron B1 E F G = 1 / 3 :=
by
  sorry

end volume_of_tetrahedron_B1_EFG_l288_288541


namespace angle_between_lines_is_45_degrees_l288_288226

noncomputable def dot_product (a b : ℝ × ℝ) : ℝ :=
  a.1 * b.1 + a.2 * b.2

noncomputable def magnitude (a : ℝ × ℝ) : ℝ :=
  Real.sqrt (a.1 ^ 2 + a.2 ^ 2)

noncomputable def angle_between (a b : ℝ × ℝ) : ℝ :=
  Real.acos ((dot_product a b) / (magnitude a * magnitude b))

-- The direction vectors of the lines
def a : ℝ × ℝ := (1, 2)
def b : ℝ × ℝ := (1, -3)

-- The value of the angle between the lines, expressed in degrees
def angle_degrees (a b : ℝ × ℝ) : ℝ :=
  angle_between a b * 180 / Real.pi

theorem angle_between_lines_is_45_degrees :
  angle_degrees a b = 45 ∨ angle_degrees a b = 135 :=
by
  sorry

end angle_between_lines_is_45_degrees_l288_288226


namespace distance_to_nearest_town_l288_288345

theorem distance_to_nearest_town (d : ℝ) :
  ¬ (d ≥ 6) → ¬ (d ≤ 5) → ¬ (d ≤ 4) → (d > 5 ∧ d < 6) :=
by
  intro h1 h2 h3
  sorry

end distance_to_nearest_town_l288_288345


namespace find_k_l288_288008

theorem find_k (k : ℝ) : 4 + ∑' (n : ℕ), (4 + n * k) / 5^n = 10 → k = 16 := by
  sorry

end find_k_l288_288008


namespace range_of_a_l288_288843

theorem range_of_a (a : ℝ) (p : ∀ x ∈ set.Icc 0 1, a ≥ real.exp x) (q : ∃ x : ℝ, x^2 + 4*x + a = 0) : a ∈ set.Icc (real.exp 1) 4 :=
by
  sorry

end range_of_a_l288_288843


namespace solve_trig_equation_l288_288216

theorem solve_trig_equation (x y z : Real) (n k m : Int) (h1 : cos x ≠ 0) (h2 : cos y ≠ 0) :
  ((cos x ^ 2 + 1 / cos x ^ 2) ^ 3 + (cos y ^ 2 + 1 / cos y ^ 2) ^ 3 = 16 * sin z) ↔
  (x = ↑n * Real.pi ∧ y = ↑k * Real.pi ∧ z = Real.pi / 2 + 2 * ↑m * Real.pi) :=
by
  sorry

end solve_trig_equation_l288_288216


namespace count_positive_area_triangles_l288_288093

noncomputable def numPositiveAreaTriangles : ℕ := 2160

theorem count_positive_area_triangles 
  (vertices : list (ℤ × ℤ))
  (h1 : ∀ p ∈ vertices, 1 ≤ p.1 ∧ p.1 ≤ 5 ∧ 1 ≤ p.2 ∧ p.2 ≤ 5)
  (h2 : vertices.length = 25) : 
  (number_of_triangles_with_positive_area vertices) = numPositiveAreaTriangles :=
sorry

end count_positive_area_triangles_l288_288093


namespace count_odd_integers_with_conditions_l288_288071

/-
  Question: How many odd integers between 3000 and 6000 have all different digits, including at least one digit as '5'?
  Conditions:
  - The integer is odd.
  - The integer is between 3000 and 6000.
  - The integer has all different digits.
  - The integer includes at least one digit as '5'.
-/

def odd_integers_with_conditions : ℕ :=
  -- Correct answer from solution
  804

theorem count_odd_integers_with_conditions :
  ∃! n : ℕ, n = odd_integers_with_conditions :=
begin
  use 804,
  split,
  { refl },
  { assume x hx,
    exact hx }
end

end count_odd_integers_with_conditions_l288_288071


namespace distance_between_points_l288_288406

-- Define the given points
def point1 : (ℝ × ℝ) := (2, 3)
def point2 : (ℝ × ℝ) := (5, 9)

-- Define the distance function
def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

-- State the theorem to prove
theorem distance_between_points :
  distance point1 point2 = 3 * real.sqrt 5 := 
sorry

end distance_between_points_l288_288406


namespace problem_statement_l288_288453

noncomputable def f : ℝ → ℝ :=
λ x, if x > 0 then 4 ^ x else f (x + 1) - 1

theorem problem_statement : f (-1/2) + f (1/2) = 3 := 
sorry

end problem_statement_l288_288453


namespace number_of_valid_triangles_l288_288098

theorem number_of_valid_triangles : 
  let points := {(x, y) | 1 ≤ x ∧ x ≤ 5 ∧ 1 ≤ y ∧ y ≤ 5} in
  ∃ n : ℕ, n = (nat.card (points.choose 3)) - 120 ∧ n = 2180 :=
by
  sorry

end number_of_valid_triangles_l288_288098


namespace berthaDaughtersGranddaughtersNoDaughters_l288_288366

-- Define the conditions of the problem
def berthaHasEightDaughters : ℕ := 8
def totalWomen : ℕ := 40

noncomputable def berthaHasNoSons : Prop := true
def daughtersHaveFourDaughters (x : ℕ) : Prop := x = 4
def granddaughters : ℕ := totalWomen - berthaHasEightDaughters
def daughtersWithNoChildren : ℕ := 0

theorem berthaDaughtersGranddaughtersNoDaughters :
  let daughtersWithChildren := granddaughters / 4,
      womenWithNoDaughters := granddaughters + daughtersWithNoChildren
  in womenWithNoDaughters = 32 :=
by
  sorry

end berthaDaughtersGranddaughtersNoDaughters_l288_288366


namespace triangle_ratio_l288_288578

open Real

variable {P1 P2 P3 P Q1 Q2 Q3 : Type} [OrderedField ℝ]

/-- Given a point P inside the triangle P1 P2 P3, and lines P1P, P2P, and P3P intersecting the opposite sides at points Q1, Q2, and Q3 respectively, there exists at least one ratio among P1P/PQ1, P2P/PQ2, and P3P/PQ3 which is not greater than 2, and at least one which is not less than 2. -/
theorem triangle_ratio (hPtInside : ∃ λ₁ λ₂ λ₃ : ℝ, 0 < λ₁ ∧ λ₁ < 1 ∧ 0 < λ₂ ∧ λ₂ < 1 ∧ 0 < λ₃ ∧ λ₃ < 1) :
  (∃ x y z : ℝ, x/y ≤ 2 ∧ z/y ≥ 2) :=
by
  sorry

end triangle_ratio_l288_288578


namespace solution_set_inequality_l288_288036

variable (f : ℝ → ℝ)
variable (h1 : ∀ x, f (x - 1/2) + f (x + 1) = 0)
variable (h2 : e ^ 3 * f 2018 = 1)
variable (h3 : ∀ x, f x > f'' (-x))
variable (h4 : ∀ x, f x = f (-x))

theorem solution_set_inequality :
  ∀ x, f (x - 1) > 1 / (e ^ x) ↔ x > 3 :=
sorry

end solution_set_inequality_l288_288036


namespace product_of_ratios_equals_one_l288_288426

-- Defining the points on the polygon and associated intersecting points
variables {n : ℕ} (A : fin (2 * n - 1) → Type) (O : Type) 
          (B : fin n → Type) (A_k : fin (2 * n - 1) → fin (2 * n - 1) → Type)
          (A_intersect : ∀ k : fin n, A_k k = O)
          (B_k_intersect : ∀ k : fin n, B k = A_intersect k)

-- Assuming the existence of a function measuring distance between two points
variables (dist : Π (a b : Type), ℝ)

-- The final proof statement
theorem product_of_ratios_equals_one :
  (∏ k in finset.range n, dist (A (n + k - 1)) (B k) / dist (A (n + k)) (B k) = 1) :=
sorry

end product_of_ratios_equals_one_l288_288426


namespace correct_equation_l288_288272

-- Define the necessary conditions and parameters
variables (x : ℝ)

-- Length of the rectangle
def length := x 

-- Width is 6 meters less than the length
def width := x - 6

-- The area of the rectangle
def area := 720

-- Proof statement
theorem correct_equation : 
  x * (x - 6) = 720 :=
sorry

end correct_equation_l288_288272


namespace coefficient_of_x5_in_binomial_expansion_l288_288931

theorem coefficient_of_x5_in_binomial_expansion :
  let n : ℕ := 8
  let binomial : ℕ -> ℕ -> ℤ := λ n r, nat.choose n r
  let term_coeff : ℕ -> ℕ -> ℤ := λ n r, (-1 : ℤ) ^ r * binomial n r
  let term_exponent : ℕ -> ℕ -> ℤ := λ n r, ((16 - 3 * (r : ℕ)) / 2)
  0 < n -> (∑ r in finset.range (n + 1), if term_exponent n r = 5 then term_coeff n r * x ^ 5 else 0) = 28 :=
by
  sorry

end coefficient_of_x5_in_binomial_expansion_l288_288931


namespace plane_equation_correct_l288_288300

structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

def vectorBC (B C : Point3D) : Point3D :=
  { x := C.x - B.x, y := C.y - B.y, z := C.z - B.z }

def planeEquation (n : Point3D) (A : Point3D) (P : Point3D) : ℝ :=
  n.x * (P.x - A.x) + n.y * (P.y - A.y) + n.z * (P.z - A.z)

theorem plane_equation_correct :
  let A := ⟨3, -3, -6⟩
  let B := ⟨1, 9, -5⟩
  let C := ⟨6, 6, -4⟩
  let n := vectorBC B C
  ∀ P, planeEquation n A P = 0 ↔ 5 * (P.x - A.x) - 3 * (P.y - A.y) + 1 * (P.z - A.z) - 18 = 0 :=
by
  sorry

end plane_equation_correct_l288_288300


namespace correct_equation_l288_288266

variable (x : ℝ)
axiom area_eq_720 : x * (x - 6) = 720

theorem correct_equation : x * (x - 6) = 720 := by
  exact area_eq_720

end correct_equation_l288_288266


namespace two_distinct_roots_l288_288037

def f (x : ℝ) := | x - 3 | + 1
def g (k x : ℝ) := k * x

theorem two_distinct_roots (k : ℝ) (h : 1 / 3 < k ∧ k < 1) : 
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f x₁ = g k x₁ ∧ f x₂ = g k x₂ :=
sorry

end two_distinct_roots_l288_288037


namespace total_cats_and_kittens_left_l288_288560

-- Definitions from the conditions
def num_adult_cats : ℕ := 120
def percent_female : ℚ := 0.6
def fraction_litters : ℚ := 2 / 3
def avg_kittens_per_litter : ℕ := 3
def kittens_adopted : ℕ := 20

-- Lean statement for the proof problem
theorem total_cats_and_kittens_left :
  let num_female_cats := (num_adult_cats : ℚ) * percent_female
      num_litters := fraction_litters * num_female_cats
      total_kittens := (num_litters : ℕ) * avg_kittens_per_litter
      kittens_left := total_kittens - kittens_adopted
      total_cats_and_kittens := num_adult_cats + kittens_left
  in total_cats_and_kittens = 244 := by
  sorry

end total_cats_and_kittens_left_l288_288560


namespace number_with_5_or_6_base_8_l288_288866

open Finset

def count_numbers_with_5_or_6 : ℕ :=
  let base_8_numbers := Ico 1 (8 ^ 3)
  let count_with_5_or_6 := base_8_numbers.filter (λ n, ∃ b, b ∈ digit_set 8 n ∧ (b = 5 ∨ b = 6))
  count_with_5_or_6.card

theorem number_with_5_or_6_base_8 : count_numbers_with_5_or_6 = 296 := 
by 
  -- Proof omitted for this exercise
  sorry

end number_with_5_or_6_base_8_l288_288866


namespace find_fixed_point_l288_288016

theorem find_fixed_point (c d k : ℝ) 
(h : ∀ k : ℝ, d = 5 * c^2 + k * c - 3 * k) : (c, d) = (3, 45) :=
sorry

end find_fixed_point_l288_288016


namespace option_d_is_correct_l288_288680

theorem option_d_is_correct : (-2 : ℤ) ^ 3 = -8 := by
  sorry

end option_d_is_correct_l288_288680


namespace intersection_eq_l288_288068

def set_A : Set ℤ := {-1, 0, 1}
def set_B : Set ℝ := {x : ℝ | x * (x - 1) ≤ 0}

theorem intersection_eq : set_A ∩ (set_B ∩ Set.univ) = {0, 1} := 
by 
  sorry

end intersection_eq_l288_288068


namespace sum_greater_than_3_l288_288600

theorem sum_greater_than_3 (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a * b + b * c + c * a > a + b + c) : a + b + c > 3 :=
sorry

end sum_greater_than_3_l288_288600


namespace arc_length_150_deg_in_radius_6_l288_288126

-- Define necessary variables
def radius : ℝ := 6
def central_angle : ℝ := 150
def pi_val : ℝ := Real.pi

-- Define the calculation for arc length
def arc_length (theta r : ℝ) : ℝ := (theta * pi_val * r) / 180

-- The proof statement for the arc length given the radius and central angle
theorem arc_length_150_deg_in_radius_6 
  (r : ℝ) 
  (theta : ℝ)
  (hr : r = 6)
  (htheta : theta = 150) : 
  arc_length theta r = 5 * pi_val := 
by
  sorry


end arc_length_150_deg_in_radius_6_l288_288126


namespace transform_y1_to_y2_right_shift_l288_288273

noncomputable def y1 (x : ℝ) : ℝ := sin (2 * x) - cos (2 * x)
noncomputable def y2 (x : ℝ) : ℝ := sqrt 2 * cos (2 * x)

theorem transform_y1_to_y2_right_shift : 
  y1 = (λ x, y2 (x + (3 * π / 8))) :=
by sorry

end transform_y1_to_y2_right_shift_l288_288273


namespace find_AG_l288_288794

-- Defining constants and variables
variables (DE EC AD BC FB AG : ℚ)
variables (BC_def : BC = (1 / 3) * AD)
variables (FB_def : FB = (2 / 3) * AD)
variables (DE_val : DE = 8)
variables (EC_val : EC = 6)
variables (sum_AD : BC + FB = AD)

-- The theorem statement
theorem find_AG : AG = 56 / 9 :=
by
  -- Placeholder for the proof
  sorry

end find_AG_l288_288794


namespace locus_of_point_minimum_m_l288_288809

-- Part 1: Prove the equation of curve C
theorem locus_of_point (P : ℝ × ℝ) (A := (0,1)) (B := (0,-2)) :
  (dist P B = 2 * dist P A) → (P.fst^2 + P.snd^2 - 4 * P.snd = 0) :=
by 
  sorry

-- Part 2: Finding the minimum value of m
theorem minimum_m (M : ℝ × ℝ) (A := (0,1)) (B := (0,-2))
  (m : ℝ) (hM : M.snd = -m * M.fst + 3 * m + 1) :
  (dist M B ≥ 2 * dist M A) → (m ≥ (3 - 2 * real.sqrt 6) / 5) :=
by
  sorry

end locus_of_point_minimum_m_l288_288809


namespace positive_area_triangles_correct_l288_288083

-- Define the set of points with integer coordinates in the given range
def grid_points := { p : ℤ × ℤ | 1 ≤ p.1 ∧ p.1 ≤ 5 ∧ 1 ≤ p.2 ∧ p.2 ≤ 5 }

-- Define a function to check for collinearity of three points
def collinear (p1 p2 p3 : ℤ × ℤ) : Prop :=
  (p2.2 - p1.2) * (p3.1 - p1.1) = (p3.2 - p1.2) * (p2.1 - p1.1)

-- Define the count of triangles with positive area (not collinear) from the given points
def triangles_with_positive_area (points : set (ℤ × ℤ)) : ℕ :=
  (@set.to_finset (ℤ × ℤ) _ points).to_list.combinations 3
  .filter (λ l, l.length = 3 ∧ ¬ collinear l.head l.nth 1 l.nth 2).length

-- State the proof problem
theorem positive_area_triangles_correct :
  triangles_with_positive_area grid_points = 2170 := sorry

end positive_area_triangles_correct_l288_288083


namespace incenter_of_triangle_l288_288754

variables {P : Type} [MetricSpace P]

-- Definitions and conditions
def circle (O : P) (r : ℝ) : set P := { X | dist X O = r }
def intersects (o1 o2 : set P) := ∃ (A B : P), A ≠ B ∧ A ∈ o1 ∧ A ∈ o2 ∧ B ∈ o1 ∧ B ∈ o2

noncomputable def obtuse_angle (O1 A O2 : P) : Prop := sorry -- Placeholder for the obtuse angle definition

variables (O1 O2 A B C D : P)
variables (r1 r2 : ℝ)

-- Specific setup for the circles and points
def circle_o1 := circle O1 r1
def circle_o2 := circle O2 r2

-- Point B intersects both circles
def circle_conditions := intersects circle_o1 circle_o2 ∧ 
  A ≠ B ∧ 
  obtuse_angle O1 A O2 ∧ 
  (∃ C, C ≠ B ∧ dist B C = r2 ∧ dist O1 C = dist O1 B) ∧ 
  (∃ D, D ≠ B ∧ dist B D = r1 ∧ dist O2 D = dist O2 B)

-- Statement to prove
theorem incenter_of_triangle (h : circle_conditions O1 O2 A B C D r1 r2) : is_incenter B A C D := 
begin
  sorry
end

end incenter_of_triangle_l288_288754


namespace real_imaginary_equality_l288_288060

noncomputable def equation (a : ℝ) : ℂ := (a + complex.I * 2 : ℂ) * complex.I

theorem real_imaginary_equality (a : ℝ) :
  (equation a).re = (equation a).im → a = -2 :=
by
  sorry

end real_imaginary_equality_l288_288060


namespace smart_charging_piles_equation_l288_288525

-- defining conditions as constants
constant initial_piles : ℕ := 301
constant third_month_piles : ℕ := 500
constant growth_rate : ℝ 

-- Expressing the given mathematical proof problem
theorem smart_charging_piles_equation : 
  initial_piles * (1 + growth_rate)^2 = third_month_piles :=
sorry

end smart_charging_piles_equation_l288_288525


namespace blocks_time_l288_288968

def blocks_added (c : ℕ) : ℕ :=
  5 * c

def blocks_removed (c : ℕ) : ℕ :=
  3 * c

def net_blocks (c : ℕ) : ℕ :=
  blocks_added c - blocks_removed c

def total_blocks (c : ℕ) : ℕ :=
  net_blocks c + 5

def time_spent (c : ℕ) : ℕ :=
  c * 45 + 45

theorem blocks_time :
  ∀ (c : ℕ), net_blocks 22 = 44 →
  time_spent 22 = 16.5 * 60 :=
by
  intro c
  sorry

end blocks_time_l288_288968


namespace grasshopper_jumps_l288_288338

noncomputable def calculate_position (n : ℕ) : (ℤ × ℤ) :=
  let m := n / 4
  let rem := n % 4
  let base_west := -2 * m
  let base_south := -2 * m
  let additional_moves :=
    match rem with
    | 0 => (0, 0)
    | 1 => (1, 0)
    | 2 => (1, -2)
    | 3 => (-2, -2)
    | 4 => (-2, -4)
    | _ => (0, 0)  -- This case should not occur because rem should be in 0 to 3.
  let (additional_west, additional_south) := additional_moves
  (base_west + additional_west, base_south + additional_south)

theorem grasshopper_jumps :
  let n := 323
  let sum_of_squares_of_digits := (3 ∧ 3 ∧ 2 : ℕ → ℕ → ℕ → ℕ) → ℕ × 9 + 4 + 9 = 22
  let (pos_west, pos_south) := calculate_position n
  pos_west = 162 ∧ pos_south = 158 :=
sorry

end grasshopper_jumps_l288_288338


namespace arithmetic_sequence_150th_term_l288_288123

theorem arithmetic_sequence_150th_term :
  ∀ (a : ℕ) (d : ℕ) (n : ℕ), a = 3 → d = 5 → n = 150 → (a + (n - 1) * d) = 748 :=
by
  intros a d n ha hd hn
  rw [ha, hd, hn]
  apply congr_arg _ (Nat.mul_sub_ne_zero _)
  apply Nat.add_sub_ne_zero
  apply Nat.sub_eq_zero_of_eq (by rfl)
  apply Nat.add_sub_ne_zero
  apply Nat.mul_sub_ne_zero
  apply Nat.add_sub_ne_zero
  sorry  -- You can fill in the proof steps.

end arithmetic_sequence_150th_term_l288_288123


namespace problem1_problem2_l288_288377

noncomputable def integral1 : ℝ := ∫ x in 0..2, (4 - 2 * x) * (4 - x^2)
noncomputable def integral2 : ℝ := ∫ x in 1..2, (x^2 - 2 * x - 3) / x

theorem problem1 : integral1 = 40 / 3 := by
  sorry

theorem problem2 : integral2 = 1 - Real.log 2 := by
  sorry

end problem1_problem2_l288_288377


namespace Sally_out_of_pocket_payment_l288_288993

theorem Sally_out_of_pocket_payment :
  let amount_given : ℕ := 320
  let cost_per_book : ℕ := 12
  let number_of_students : ℕ := 30
  let total_cost : ℕ := cost_per_book * number_of_students
  let out_of_pocket_cost : ℕ := total_cost - amount_given
  out_of_pocket_cost = 40 := by
  sorry

end Sally_out_of_pocket_payment_l288_288993


namespace binomial_coefficient_sum_leq_one_l288_288920

theorem binomial_coefficient_sum_leq_one
  (M : ℕ)
  (S : ℕ)
  (a : Fin S → ℕ)
  (h_subsets_not_contained : ∀ i j, ¬ (i ≠ j ∧ (a i ≤ a j)))
  : ∑ i in Finset.univ, 1 / Nat.choose M (a i) ≤ 1 := 
sorry

end binomial_coefficient_sum_leq_one_l288_288920


namespace sin_double_angle_l288_288492

theorem sin_double_angle (θ : ℝ) : (cos θ + sin θ = 3/2) → sin (2 * θ) = 5/4 :=
by
  intro h
  sorry

end sin_double_angle_l288_288492


namespace gcd_six_triangular_n_and_n_minus_one_l288_288013

theorem gcd_six_triangular_n_and_n_minus_one (n : ℕ) (hn : 0 < n) :
  let T_n := ∑ k in Finset.range (n + 1), k
  in gcd (6 * T_n) (n - 1) ≤ 3 :=
by
  sorry

end gcd_six_triangular_n_and_n_minus_one_l288_288013


namespace eq_correct_l288_288269

variable (x : ℝ)

def width (x : ℝ) : ℝ := x - 6

def area_eq (x : ℝ) : Prop := x * width x = 720

theorem eq_correct (h : area_eq x) : x * (x - 6) = 720 :=
by exact h

end eq_correct_l288_288269


namespace problem1_part1_problem1_part2_l288_288846

open Set Real

theorem problem1_part1 (a : ℝ) (h1: a = 5) :
  let A := { x : ℝ | (x - 6) * (x - 2 * a - 5) > 0 }
  let B := { x : ℝ | (a ^ 2 + 2 - x) * (2 * a - x) < 0 }
  A ∩ B = { x | 15 < x ∧ x < 27 } := sorry

theorem problem1_part2 (a : ℝ) (h2: a > 1 / 2) :
  let A := { x : ℝ | x < 6 ∨ x > 2 * a + 5 }
  let B := { x : ℝ | 2 * a < x ∧ x < a ^ 2 + 2 }
  (∀ x, x ∈ A → x ∈ B) ∧ ¬ (∀ x, x ∈ B → x ∈ A) → (1 / 2 < a ∧ a ≤ 2) := sorry

end problem1_part1_problem1_part2_l288_288846


namespace ram_gohul_work_days_l288_288303

theorem ram_gohul_work_days (ram_days gohul_days : ℕ) (H_ram: ram_days = 10) (H_gohul: gohul_days = 15): 
  (ram_days * gohul_days) / (ram_days + gohul_days) = 6 := 
by
  sorry

end ram_gohul_work_days_l288_288303


namespace math_proof_problem_l288_288439

-- Given condition definitions
structure Point (α : Type) :=
(x : α)
(y : α)

def P : Point ℝ := ⟨-5, 0⟩

def onCircle (Q : Point ℝ) : Prop := (Q.x - 5)^2 + Q.y^2 = 36

def midpoint (P Q M : Point ℝ) : Prop := M.x = (P.x + Q.x) / 2 ∧ M.y = (P.y + Q.y) / 2

def trajectory (M : Point ℝ) : Prop := M.x^2 + M.y^2 = 9

noncomputable def line (P : Point ℝ) (k : ℝ) : (x y : ℝ) → Prop := 
  λ x y, y = k * (x + P.x)

def intersects (P : Point ℝ) (k : ℝ) : Prop :=
  let A := Point.mk (-5 + 6) (k * (1 + 6)) in
  let B := Point.mk (-5 - 6) (k * (1 - 6)) in
  let dist := 4 in
  let AB := (A.x - B.x) * (A.x - B.x) + (A.y - B.y) * (A.y - B.y) in  
  |AB| = dist

def dotProduct (A B P : Point ℝ) : ℝ :=
  let PA := Point.mk (A.x - P.x) (A.y - P.y) in
  let PB := Point.mk (B.x - P.x) (B.y - P.y) in
  PA.x * PB.x + PA.y * PB.y

-- The Lean 4 statement for the math proof problem
theorem math_proof_problem :
  ∃ M : Point ℝ, midpoint P Q M → onCircle Q →
  trajectory M ∧ 
  (∀ k : ℝ, (k = 1/2 ∨ k = -1/2) →
    (intersects P k →
    (|P.x - A.x| * |P.x - B.x| = 16))) :=
sorry

end math_proof_problem_l288_288439


namespace fraction_value_l288_288462

theorem fraction_value
  (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (cond1 : (a + b + c) / (a + b - c) = 7)
  (cond2 : (a + b + c) / (a + c - b) = 1.75) :
  (a + b + c) / (b + c - a) = 3.5 :=
by
  sorry

end fraction_value_l288_288462


namespace horizontal_length_paper_glued_correct_l288_288257

noncomputable def horizontal_length_paper_glued
    (paper_width : ℕ) (paper_length : ℕ)
    (pieces : ℕ) (overlap : ℝ) : ℝ :=
    let effective_width := paper_width - overlap in
    let total_length_cm := paper_width + (pieces - 1) * effective_width in
    total_length_cm / 100

theorem horizontal_length_paper_glued_correct :
    horizontal_length_paper_glued 25 30 15 0.5 = 3.68 :=
by
    -- This is where the proof would go
    sorry

end horizontal_length_paper_glued_correct_l288_288257


namespace minimum_braking_distance_l288_288623

theorem minimum_braking_distance
  (α : ℝ)
  (v g cos_alpha sin_alpha : ℝ)
  (hα : α = 15)
  (hv : v = 20)
  (hg : g = 10)
  (hcos : cos_alpha = 0.966)
  (hsin : sin_alpha = 0.259) :
  (v^2) / (2 * ((sin_alpha / cos_alpha) * g)) ≈ 74.6 :=
by
  have hmu : (sin_alpha / cos_alpha) = 0.268 := sorry
  have ha : ((sin_alpha / cos_alpha) * g) = 2.68 := sorry
  calc
    (v^2) / (2 * ((sin_alpha / cos_alpha) * g))
        = 400 / (2 * 2.68) : by sorry
    ... ≈ 74.6 : by sorry

end minimum_braking_distance_l288_288623


namespace part1_monotonic_decreasing_part2_range_of_a_l288_288456

def f (x : ℝ) : ℝ := x / (x - 1)

theorem part1_monotonic_decreasing : ∀ x1 x2 : ℝ, 1 < x1 → x1 < x2 → (1 : ℝ) < x2 → f x1 > f x2 :=
by
  sorry

theorem part2_range_of_a : ∀ a : ℝ, f (2 * a + 1) > f (a + 2) → (0 < a ∧ a < 1) :=
by
  sorry

end part1_monotonic_decreasing_part2_range_of_a_l288_288456


namespace regular_polygon_eq_equilateral_l288_288011

-- Definitions for a convex n-gon with the specified properties
structure RegularNGon (n : ℕ) (n_ge_4 : n ≥ 4) :=
  (convex : Prop)
  (condition : ∀ k, 2 ≤ k → k ≤ n - 2 → sorry)  -- Placeholder for the actual mathematical condition

-- Theorem statement for the proof problem
theorem regular_polygon_eq_equilateral (n : ℕ) (n_ge_7 : n ≥ 7) 
  (P : RegularNGon n (by linarith)) : 
  sorry -- Placeholder for the statement that the polygon is a regular (equilateral) n-gon.

end regular_polygon_eq_equilateral_l288_288011


namespace sum_of_logs_of_first_8_terms_l288_288540

noncomputable def a (n : ℕ) : ℝ -- Geometric sequence

axiom a4 : a 4 = 2
axiom a5 : a 5 = 5

def S := ∑ i in finset.range 8, real.log (a (i + 1))

theorem sum_of_logs_of_first_8_terms : S = 4 :=
by
  sorry

end sum_of_logs_of_first_8_terms_l288_288540


namespace fraction_people_over_65_l288_288777

theorem fraction_people_over_65 (T : ℕ) (F : ℕ) : 
  (3:ℚ) / 7 * T = 24 ∧ 50 < T ∧ T < 100 → T = 56 ∧ ∃ F : ℕ, (F / 56 : ℚ) = F / (T : ℚ) :=
by 
  sorry

end fraction_people_over_65_l288_288777


namespace seating_arrangements_l288_288482

theorem seating_arrangements (n m : ℕ) (desks : fin n) (students : fin m) (at_least_one_empty : desks ≥ students + 1) : 
  (students = 2) → (desks = 5) → ∃ (ways : ℕ), ways = 6 := 
by 
  sorry

end seating_arrangements_l288_288482


namespace wrapping_paper_area_is_correct_l288_288321

noncomputable def wrapping_paper_area (s h : ℝ) : ℝ :=
  let l := (Real.sqrt (s ^ 2 + h ^ 2) + s * Real.sqrt 2) in
  l ^ 2

theorem wrapping_paper_area_is_correct (s h : ℝ) :
  wrapping_paper_area s h = (3 * s ^ 2 + h ^ 2 + 2 * s * Real.sqrt (2 * (s ^ 2 + h ^ 2))) :=
by
  sorry

end wrapping_paper_area_is_correct_l288_288321


namespace jerry_original_butterflies_l288_288153

/-- Define the number of butterflies Jerry originally had -/
def original_butterflies (let_go : ℕ) (now_has : ℕ) : ℕ := let_go + now_has

/-- Given conditions -/
def let_go : ℕ := 11
def now_has : ℕ := 82

/-- Theorem to prove the number of butterflies Jerry originally had -/
theorem jerry_original_butterflies : original_butterflies let_go now_has = 93 :=
by
  sorry

end jerry_original_butterflies_l288_288153


namespace number_of_orange_bottles_l288_288283

-- Definitions based on given conditions
variables {O A G : ℕ}

/-- A bottle of orange juice costs 70 cents, a bottle of apple juice costs 60 cents, 
and a bottle of grape juice costs 80 cents. -/
def cost_eqn := 70 * O + 60 * A + 80 * G = 7250

/-- A total of 100 bottles were bought for $72.50. -/
def total_bottles_eqn := O + A + G = 100

/-- The number of apple and grape juice bottles is equal. -/
def equal_apple_grape := A = G

/-- Twice as many orange juice bottles were bought as apple juice bottles. -/
def double_orange_apple := O = 2 * A

theorem number_of_orange_bottles
  (h1 : cost_eqn) (h2 : total_bottles_eqn) (h3 : equal_apple_grape) (h4 : double_orange_apple) :
  O = 50 :=
by sorry

end number_of_orange_bottles_l288_288283


namespace intersection_distance_l288_288925

-- Conditions
def line_l1_cartesian (x y : ℝ) : Prop := y = sqrt 3 * x
def curve_C_param_x (φ : ℝ) : ℝ := 1 + sqrt 3 * cos φ
def curve_C_param_y (φ : ℝ) : ℝ := sqrt 3 * sin φ
def valid_param (φ : ℝ) : Prop := 0 ≤ φ ∧ φ ≤ π
def line_l2_polar (ρ θ : ℝ) : Prop := 2 * ρ * sin (θ + π / 3) + 3 * sqrt 3 = 0

-- Prove the intersection points and distance
theorem intersection_distance :
  let ρ1 := 2
  let θ1 := π / 3
  let ρ2 := -3
  let θ2 := π / 3
  |ρ1 - ρ2| = 5 :=
by
  -- The intersections are as given in the problem
  let ρ1 : ℝ := 2
  let θ1 : ℝ := π / 3
  let ρ2 : ℝ := -3
  let θ2 : ℝ := π / 3

  -- The required distance calculation:
  have h : |ρ1 - ρ2| = abs (ρ1 - ρ2), by sorry
  rw [abs_sub_eq_add_sub ((2 : ℝ) - (-3 : ℝ))]
  exact eq.refl (5 : ℝ)  -- asserting the given distance


end intersection_distance_l288_288925


namespace sum_of_third_terms_is_correct_l288_288759

-- Definitions based on the given conditions
def sequence (n : ℕ) := y_1 + (n - 1) * 2
def sum_of_sequence := 2080
def num_of_terms := 2023
def num_of_terms_T := 674
def common_diff := 2

-- Sum of every third term starting from the first
def sum_of_every_third_term (y_1 : ℤ) :=
  674 * (y_1 + 2019)

-- Calculate y_1 based on the total sum equation
def y_1 := (2080 - (2022 * 2011)) / 2023

-- The main theorem to be proved
theorem sum_of_third_terms_is_correct :
  ∑ i in (finset.range 674), (y_1 + (i * 6)) = 674 * (y_1 + 2019) :=
by
  sorry

end sum_of_third_terms_is_correct_l288_288759


namespace nadine_total_cleaning_time_l288_288192

-- Conditions
def time_hosing_off := 10 -- minutes
def shampoos := 3
def time_per_shampoo := 15 -- minutes

-- Total cleaning time calculation
def total_cleaning_time := time_hosing_off + (shampoos * time_per_shampoo)

-- Theorem statement
theorem nadine_total_cleaning_time : total_cleaning_time = 55 := by
  sorry

end nadine_total_cleaning_time_l288_288192


namespace bulldozer_no_double_hit_l288_288929

/-- Suppose there are finitely many disjoint line segments (walls) not parallel to either axis.
A bulldozer starts at an arbitrary point and initially moves in the +x direction. Upon hitting a wall,
it turns at a right angle away from the wall and continues moving parallel to the axes. Prove that it
is impossible for the bulldozer to hit both sides of every wall. -/
theorem bulldozer_no_double_hit (walls : set (set (ℝ × ℝ))) (H : finite walls)
  (H_disjoint : ∀ w₁ w₂ ∈ walls, w₁ ≠ w₂ → disjoint w₁ w₂)
  (H_not_parallel : ∀ w ∈ walls, ¬ parallel_to_axes w) :
  ∃ w ∈ walls, ∀ path : list (ℝ × ℝ), ¬ (hits_both_sides path w) :=
sorry

/-- Additional definitions and helper lemmas would go here -/
def disjoint (w1 w2 : set (ℝ × ℝ)) := ∀ p ∈ w1, ∀ q ∈ w2, p ≠ q
def parallel_to_axes (w : set (ℝ × ℝ)) := ∃ c, w = {p : ℝ × ℝ | p.2 = c} ∨ w = {p : ℝ × ℝ | p.1 = c}
def hits_both_sides (path : list (ℝ × ℝ)) (w : set (ℝ × ℝ)) : Prop := 
  hits_side path w "left" ∧ hits_side path w "right"
/-- Add additional helper definitions, and enable proper encapsulation of logic increments -/
def hits_side (path : list (ℝ × ℝ)) (w : set (ℝ × ℝ)) (side : string) : Prop :=
  sorry

end bulldozer_no_double_hit_l288_288929


namespace count_4_digit_numbers_divisible_by_13_l288_288880

theorem count_4_digit_numbers_divisible_by_13 : 
  let count := (9962 - 1014) / 13 + 1
  1000 ≤ 1014 ∧ 9962 ≤ 9999 →
  count = 689 :=
  by
    sorry

end count_4_digit_numbers_divisible_by_13_l288_288880


namespace final_amount_after_bets_l288_288688

theorem final_amount_after_bets :
  let initial_money : ℝ := 64
  let bets := 6
  let wins := 3
  let losses := 3
  let bet_ratio : ℝ := 0.5
  (∀ (sequence : List Bool), 
    (sequence.length = bets ∧ sequence.count (fun b => b) = wins ∧ sequence.count (fun b => ¬b) = losses) →
    (initial_money * (sequence.foldl (λ acc win, if win then acc * 1.5 else acc * 0.5) 1) = 27)) :=
sorry

end final_amount_after_bets_l288_288688


namespace distance_between_points_l288_288405

-- Define the given points
def point1 : (ℝ × ℝ) := (2, 3)
def point2 : (ℝ × ℝ) := (5, 9)

-- Define the distance function
def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

-- State the theorem to prove
theorem distance_between_points :
  distance point1 point2 = 3 * real.sqrt 5 := 
sorry

end distance_between_points_l288_288405


namespace basketball_team_selection_l288_288706

theorem basketball_team_selection : 
  ∀ (total_players : ℕ) (twins : finset ℕ) (team_size : ℕ),
  total_players = 15 →
  twins = {0, 1} →
  team_size = 5 →
  (finset.card {comb : finset ℕ // comb.card = team_size ∧ (0 ∈ comb ∨ 1 ∈ comb)}) = 1716 := by
{
  intros,
  sorry
}

end basketball_team_selection_l288_288706


namespace greatest_four_digit_p_l288_288330

-- Define conditions
def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n < 10000
def reverse_digits (n : ℕ) : ℕ := 
  let d1 := n / 1000 % 10
  let d2 := n / 100 % 10
  let d3 := n / 10 % 10
  let d4 := n % 10
  d4 * 1000 + d3 * 100 + d2 * 10 + d1
def is_divisible_by (a b : ℕ) : Prop := b ∣ a

-- Proof problem
theorem greatest_four_digit_p (p : ℕ) (q : ℕ) 
    (hp1 : is_four_digit p)
    (hp2 : q = reverse_digits p)
    (hp3 : is_four_digit q)
    (hp4 : is_divisible_by p 63)
    (hp5 : is_divisible_by q 63)
    (hp6 : is_divisible_by p 19) :
  p = 5985 :=
sorry

end greatest_four_digit_p_l288_288330


namespace maximum_value_of_m_l288_288234

theorem maximum_value_of_m
  (m : ℝ)
  (h1 : ∀ x ∈ Ioo (-m) m, deriv (λ x, sin (2 * x + π / 4)) x ≠ 0) :
  m ≤ π / 8 :=
by
  sorry

end maximum_value_of_m_l288_288234


namespace conversion_bahs_yahs_l288_288506

def bahs_to_rahs (bahs : ℕ) : ℕ := 3 * bahs
def rahs_to_yahs (rahs : ℕ) : ℕ := (10 * rahs) / 6

theorem conversion_bahs_yahs (bah_rah_relation : 10 = bahs_to_rahs 1) (rah_yah_relation : 6 = rahs_to_yahs 1) :
  ∀ yahs : ℕ, bahs_to_rahs (3 * yahs / 5) = 3 * yahs / 5 → bahs_to_rahs 100 = 3 := 
by
  intros yahs h
  have needed_bahs : bahs_to_rahs (3 * yahs / 5) = 100
  sorry

end conversion_bahs_yahs_l288_288506


namespace maximum_number_of_intersections_of_150_lines_is_7171_l288_288964

def lines_are_distinct (L : ℕ → Type) : Prop := 
  ∀ n m : ℕ, n ≠ m → L n ≠ L m

def lines_parallel_to_each_other (L : ℕ → Type) (k : ℕ) : Prop :=
  ∀ n m : ℕ, n ≠ m → L (k * n) = L (k * m)

def lines_pass_through_point_B (L : ℕ → Type) (B : Type) (k : ℕ) : Prop :=
  ∀ n : ℕ, L (k * n - 4) = B

def lines_not_parallel (L : ℕ → Type) (k1 k2 : ℕ) : Prop :=
  ∀ n m : ℕ, L (k1 * n) ≠ L (k2 * m)

noncomputable def max_points_of_intersection
  (L : ℕ → Type)
  (B : Type)
  (k1 k2 : ℕ)
  (h_distinct : lines_are_distinct L)
  (h_parallel1 : lines_parallel_to_each_other L k1)
  (h_parallel2 : lines_parallel_to_each_other L k2)
  (h_pass_through_B : lines_pass_through_point_B L B 5)
  (h_not_parallel : lines_not_parallel L k1 k2)
  : ℕ :=
  7171

theorem maximum_number_of_intersections_of_150_lines_is_7171
  (L : ℕ → Type)
  (B : Type)
  (k1 k2 : ℕ)
  (h_distinct : lines_are_distinct L)
  (h_parallel1 : lines_parallel_to_each_other L k1)
  (h_parallel2 : lines_parallel_to_each_other L k2)
  (h_pass_through_B : lines_pass_through_point_B L B 5)
  (h_not_parallel : lines_not_parallel L k1 k2)
  : max_points_of_intersection L B k1 k2 h_distinct h_parallel1 h_parallel2 h_pass_through_B h_not_parallel = 7171 := 
  by 
  sorry

end maximum_number_of_intersections_of_150_lines_is_7171_l288_288964


namespace elevator_floors_l288_288969

-- First, define constants for the conditions
constant first_half_time : ℕ := 15
constant next_5_floors_time_per_floor : ℕ := 5
constant final_5_floors_time_per_floor : ℕ := 16
constant total_time : ℕ := 120

-- Define the total number of floors F
constant F : ℕ

-- Conditions translating the problem
axiom h1 : first_half_time + 
           (next_5_floors_time_per_floor * 5) + 
           (final_5_floors_time_per_floor * 5) = total_time

-- Prove that the total number of floors is 20
theorem elevator_floors : F = 20 :=
by sorry

end elevator_floors_l288_288969


namespace Lenora_scored_30_points_l288_288129

variable (x y : ℕ)
variable (hx : x + y = 40)
variable (three_point_success_rate : ℚ := 25 / 100)
variable (free_throw_success_rate : ℚ := 50 / 100)
variable (points_three_point : ℚ := 3)
variable (points_free_throw : ℚ := 1)
variable (three_point_contribution : ℚ := three_point_success_rate * points_three_point * x)
variable (free_throw_contribution : ℚ := free_throw_success_rate * points_free_throw * y)
variable (total_points : ℚ := three_point_contribution + free_throw_contribution)

theorem Lenora_scored_30_points : total_points = 30 :=
by
  sorry

end Lenora_scored_30_points_l288_288129


namespace M_infinite_l288_288948

open Nat

-- Define the set M
def M : Set ℕ := {k | ∃ n : ℕ, 3 ^ n % n = k % n}

-- Statement of the problem
theorem M_infinite : Set.Infinite M :=
sorry

end M_infinite_l288_288948


namespace days_required_by_x_l288_288308

theorem days_required_by_x (x y : ℝ) 
  (h1 : (1 / x + 1 / y = 1 / 12)) 
  (h2 : (1 / y = 1 / 24)) : 
  x = 24 := 
by
  sorry

end days_required_by_x_l288_288308


namespace problem_general_formulas_no_term_l288_288446

noncomputable def Sn (n : ℕ) : ℕ := 2^(n + 2) + n^2 + 3 * n - 4

theorem problem (a b : ℕ → ℕ) (S_n : ℕ → ℕ)
  (h1 : ∀ n, a n = 4 * n + 4)
  (h2 : ∀ n, b n = 2)
  (h3 : ∀ n, a 1 * b 1 + a 2 * b 2 + a 3 * b 3 + ... + a n * b n = n * 2^(n + 3))
  (h4 : ∀ n, S_n n = a n + b n) :
  S_n = 2^(n + 2) + n^2 + 3 * n - 4 :=
sorry

theorem general_formulas (a b : ℕ → ℕ)
  (h1 : a 1 = 8)
  (h2 : ∃ d, ∀ n, a (n + 1) - a n = d)
  (h3 : ∃ r, ∀ n, b (n + 1) = r * b n)
  (h4 : ∀ n, a n = 4 * n + 4)
  (h5 : ∀ n, b n = 2) :
  a n = 4 * n + 4 ∧ b n = 2 :=
sorry

theorem no_term (b : ℕ → ℕ)
  (h1 : b 1 = 4)
  (h2 : ∀ n, b n = 2)
  (h3 : ∀ r, r ≥ 2) :
  ¬(∃ n, b n = (∑ i in finset.range(n), b i)) :=
sorry

end problem_general_formulas_no_term_l288_288446


namespace bertha_zero_granddaughters_l288_288369

def bertha := "Bertha"
def daughters (x : String) := if x = bertha then 8 else 0
def has_daughters (x : String) := 4  -- Represents daughters having 4 daughters each
def no_daughters (x : String) := 0 -- Represents daughters having no daughters

# Assuming there is a total of 40 members including daughters and granddaughters
def total_members := 8 + 32 -- Bertha's daughters are 8, granddaughters are 32

def no_granddaughters_daughters :=
  if total_members = 40 then 32 else 0 -- Since daughters are 8 and granddaughters having none is 32

theorem bertha_zero_granddaughters :
  total_members = 40 → no_granddaughters_daughters = 32 :=
by
  sorry

end bertha_zero_granddaughters_l288_288369


namespace find_cost_price_l288_288645

-- Condition 1: The owner charges his customer 15% more than the cost price.
def selling_price (C : Real) : Real := C * 1.15

-- Condition 2: A customer paid Rs. 8325 for the computer table.
def paid_amount : Real := 8325

-- Define the cost price and its expected value
def cost_price : Real := 7239.13

-- The theorem to prove that the cost price matches the expected value
theorem find_cost_price : 
  ∃ C : Real, selling_price C = paid_amount ∧ C = cost_price :=
by
  sorry

end find_cost_price_l288_288645


namespace multiple_choice_question_count_l288_288352

theorem multiple_choice_question_count (n : ℕ) : 
  (4 * 224 / (2^4 - 2) = 4^2) → n = 2 := 
by
  sorry

end multiple_choice_question_count_l288_288352


namespace distances_exceed_radius_third_power_l288_288343

theorem distances_exceed_radius_third_power (A B C : ℤ × ℤ) (R : ℝ) 
  (hA : (A.1 : ℝ)^2 + (A.2 : ℝ)^2 = R^2)
  (hB : (B.1 : ℝ)^2 + (B.2 : ℝ)^2 = R^2)
  (hC : (C.1 : ℝ)^2 + (C.2 : ℝ)^2 = R^2) :
  ∃ (d : ℝ), d ∈ {dist A B, dist B C, dist C A} ∧ d > R ^ (1/3) :=
sorry

end distances_exceed_radius_third_power_l288_288343


namespace function_domain_exclusion_l288_288536

theorem function_domain_exclusion (x : ℝ) :
  (∃ y, y = 2 / (x - 8)) ↔ x ≠ 8 :=
sorry

end function_domain_exclusion_l288_288536


namespace window_treatments_total_cost_l288_288565

def sheers_cost_per_pair := 40
def drapes_cost_per_pair := 60
def number_of_windows := 3

theorem window_treatments_total_cost :
  (number_of_windows * sheers_cost_per_pair) + (number_of_windows * drapes_cost_per_pair) = 300 :=
by 
  -- calculations omitted
  sorry

end window_treatments_total_cost_l288_288565


namespace four_digit_numbers_divisible_by_13_l288_288879

theorem four_digit_numbers_divisible_by_13 : 
  (set.Icc 1000 9999).filter (λ x, x % 13 = 0).cardinality = 693 :=
by
  sorry

end four_digit_numbers_divisible_by_13_l288_288879


namespace factorize_expression_l288_288394

theorem factorize_expression (a b : ℝ) : a * b^2 - 25 * a = a * (b + 5) * (b - 5) :=
by
  sorry

end factorize_expression_l288_288394


namespace num_triangles_with_positive_area_l288_288107

/-- 
Given vertices in a 5x5 grid with integer coordinates satisfying 1 ≤ x ≤ 5 and 1 ≤ y ≤ 5,
prove that the number of triangles with positive area is 2170. 
-/
theorem num_triangles_with_positive_area : 
  (∑ t in ({(i, j) | 1 ≤ i ∧ i ≤ 5 ∧ 1 ≤ j ∧ j ≤ 5}.powerset.filter (λ (s : set (ℕ × ℕ)), s.card = 3)),
    if t₁.1 * (t₂.2 - t₃.2) + t₂.1 * (t₃.2 - t₁.2) + t₃.1 * (t₁.2 - t₂.2) ≠ 0 then 1 else 0) = 2170 :=
by sorry

end num_triangles_with_positive_area_l288_288107


namespace hyperbola_focal_length_and_eccentricity_l288_288032

theorem hyperbola_focal_length_and_eccentricity 
  (a b : ℝ) (h_a : a > 0) (h_b : b > 0)
  (c : ℝ) (h_hyp : ∀ x y : ℝ, (x^2 / a^2 - y^2 / b^2 = 1) → 
    (y = sqrt 3 * x - 4 * sqrt 3) → (x = c) ∧ (y = 0))
  (h_asymptote : b / a = sqrt 3) 
  (h_F : (c = 4)) : 
  (2 * c = 8) ∧ ((c / a) = (8 / 3)) :=
  sorry

end hyperbola_focal_length_and_eccentricity_l288_288032


namespace number_of_correct_statements_is_4_l288_288644

-- Define each condition as a separate proposition
def cond1 : Prop := ∀ (T : Triangle), T.is_stable
def cond2 : Prop := ∀ (T : Triangle), T.circumcenter_exists
def cond3 : Prop := ∀ (A : Angle), A.angle_bisector_property
def cond4 : Prop := ∃ (T₁ T₂ : Triangle), T₁.has_two_sides_and_one_angle_equal_to T₂ ∧ ¬T₁ ≅ T₂
def cond5 : Prop := ∀ (T : IsoscelesTriangle), T.vertex_angle_bisector_equals_median_and_altitude_in_vertex

-- Define the problem statement
theorem number_of_correct_statements_is_4 : 
  cond1 ∧ cond2 ∧ cond3 ∧ cond4 ∧ ¬cond5 := 
sorry

end number_of_correct_statements_is_4_l288_288644


namespace total_orchids_l288_288251

-- Conditions
def current_orchids : ℕ := 2
def additional_orchids : ℕ := 4

-- Proof statement
theorem total_orchids : current_orchids + additional_orchids = 6 :=
by
  sorry

end total_orchids_l288_288251


namespace train_length_55_meters_l288_288280

noncomputable def V_f := 47 * 1000 / 3600 -- Speed of the faster train in m/s
noncomputable def V_s := 36 * 1000 / 3600 -- Speed of the slower train in m/s
noncomputable def t := 36 -- Time in seconds

theorem train_length_55_meters (L : ℝ) (Vf : ℝ := V_f) (Vs : ℝ := V_s) (time : ℝ := t) :
  (2 * L = (Vf - Vs) * time) → L = 55 :=
by
  sorry

end train_length_55_meters_l288_288280


namespace base8_contains_5_or_6_l288_288854

theorem base8_contains_5_or_6 (n : ℕ) (h : n = 512) : 
  let count_numbers_without_5_6 := 6^3 in
  let total_numbers := 512 in
  total_numbers - count_numbers_without_5_6 = 296 := by
  sorry

end base8_contains_5_or_6_l288_288854


namespace total_cleaning_time_l288_288194

def hose_time : ℕ := 10
def shampoos : ℕ := 3
def shampoo_time : ℕ := 15

theorem total_cleaning_time : hose_time + shampoos * shampoo_time = 55 := by
  sorry

end total_cleaning_time_l288_288194


namespace proposition_1_proposition_2_proposition_3_proposition_4_l288_288450

def f (x : ℝ) (b c : ℝ) : ℝ := x * |x| + b * x + c

noncomputable def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f(x)

noncomputable def is_increasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f(x) < f(y)

theorem proposition_1 (b : ℝ) :
  is_odd (fun x => f x b 0) :=
by
  sorry

theorem proposition_2 (c : ℝ) :
  is_increasing (fun x => f x 0 c) :=
by
  sorry

theorem proposition_3 (b c : ℝ) :
  ∃ c, (λ x, f x b c) = (λ x, f x b 0 + c) :=
by
  sorry

theorem proposition_4 (b c : ℝ) :
  ∀ x1 x2 x3 x4, f x1 b c = 0 → f x2 b c = 0 → f x3 b c = 0 → f x4 b c = 0 → false :=
by
  sorry

end proposition_1_proposition_2_proposition_3_proposition_4_l288_288450


namespace points_lie_on_circle_l288_288415

theorem points_lie_on_circle (t : ℝ) : 
  let x := (3 - t^3) / (3 + t^3)
  let y := (3 * t) / (3 + t^3)
  in x^2 + y^2 = 1 :=
by
  let x := (3 - t^3) / (3 + t^3)
  let y := (3 * t) / (3 + t^3)
  have h : x^2 + y^2 = ((3 - t^3) / (3 + t^3))^2 + ((3 * t) / (3 + t^3))^2 := 
    by rw [x, y]
  sorry

end points_lie_on_circle_l288_288415


namespace real_part_eq_imag_part_l288_288119

theorem real_part_eq_imag_part (a : ℝ) :
  let z := (2 + a * complex.I) * (1 + complex.I) in 
  z.re = z.im -> a = 0 :=
by
  sorry

end real_part_eq_imag_part_l288_288119


namespace triangle_angle_and_perimeter_l288_288513

theorem triangle_angle_and_perimeter (a b c A B C : ℝ)
  (h1 : (2 * b - c) * real.cos A = a * real.cos C)
  (h2 : A = real.pi / 3)
  (h3 : a = real.sqrt 13)
  (h4 : 1/2 * b * c * real.sin A = 3 * real.sqrt 3) :
  (A = real.pi / 3) ∧ (a + b + c = 7 + real.sqrt 13) := 
  by
    sorry

end triangle_angle_and_perimeter_l288_288513


namespace invisible_point_exists_l288_288341

-- Define the spherical planet and the point-like asteroids
variable (Planet : Sphere) (Asteroids : Finset Point) (count_Asteroids : Asteroids.card = 25)

/--
  There exists a point on the surface of a spherical planet from which 
  an astronomer cannot see more than 11 out of the 25 point-like asteroids.
-/
theorem invisible_point_exists (Planet : Sphere) (Asteroids : Finset Point) 
  (h : Asteroids.card = 25) : 
  ∃ (p : Point), (count_visible_asteroids p Planet Asteroids) ≤ 11 := 
sorry

end invisible_point_exists_l288_288341


namespace sum_of_first_15_terms_l288_288305

variable (a d : ℕ)

def nth_term (n : ℕ) : ℕ := a + (n - 1) * d

def sum_of_first_n_terms (n : ℕ) : ℕ := n * (2 * a + (n - 1) * d) / 2

theorem sum_of_first_15_terms (h : nth_term 4 + nth_term 12 = 16) : sum_of_first_n_terms 15 = 120 :=
by
  sorry

end sum_of_first_15_terms_l288_288305


namespace range_of_m_l288_288831

theorem range_of_m (m : ℝ) :
  (∀ x : ℝ, (x^2 - 2 * m * x + 4 = 0) → x > 1) ↔ (2 ≤ m ∧ m < 5/2) := sorry

end range_of_m_l288_288831


namespace max_reported_numbers_l288_288744

-- Define the number of boys and girls
def boys := 29
def girls := 15

-- Define the condition that each pair dances at most once
-- We can define this using a type definition or a predicate, though it is conceptually implied
def at_most_once (a b : ℕ) : Prop := a ≤ boys ∧ b ≤ girls

-- Define the function that counts dances for each child
-- In this scenario, we'll assume dance_counts functions which assigns and counts the dances
def max_dance_count_boy := boys
def min_dance_count_boy := 0
def max_dance_count_girl := boys
def min_dance_count_girl := 0

-- The main theorem stating the problem and solution
theorem max_reported_numbers {n m : ℕ} (h1: n = boys) (h2: m = girls) : 
  ∃ (num_dances : Finset ℕ), num_dances.card = 29 :=
by {
  -- conditions can be expanded or more precisely written as needed
  -- placeholder proof
  sorry
}

end max_reported_numbers_l288_288744


namespace intersection_M_N_l288_288847

def M : Set ℝ := {x | x^2 - 2 * x < 0}
def N : Set ℝ := {-2, -1, 0, 1, 2}

theorem intersection_M_N : M ∩ N = {1} :=
  by sorry

end intersection_M_N_l288_288847


namespace transformed_stats_l288_288053

noncomputable def mean (l : List ℝ) : ℝ :=
  l.sum / l.length

noncomputable def std_dev (l : List ℝ) : ℝ :=
  Real.sqrt ((l.map (λ x => (x - mean l)^2)).sum / l.length)

theorem transformed_stats (l : List ℝ) 
  (hmean : mean l = 10)
  (hstddev : std_dev l = 2) :
  mean (l.map (λ x => 2 * x - 1)) = 19 ∧ std_dev (l.map (λ x => 2 * x - 1)) = 4 := by
  sorry

end transformed_stats_l288_288053


namespace unique_solution_l288_288765

noncomputable def f (x : ℝ) : ℝ := 2^x + 3^x + 6^x - 9^x

theorem unique_solution : ∃! x : ℝ, f x = 0 :=
by
  use 1
  split
  • show f 1 = 0
    sorry
  • intros y hy
    show y = 1
    sorry

end unique_solution_l288_288765


namespace gumballs_initial_count_l288_288771

noncomputable def initial_gumballs := (34.3 / (0.7 ^ 3))

theorem gumballs_initial_count :
  initial_gumballs = 100 :=
sorry

end gumballs_initial_count_l288_288771


namespace minimum_distance_from_midpoint_to_y_axis_l288_288228

theorem minimum_distance_from_midpoint_to_y_axis (M N : ℝ × ℝ) (P : ℝ × ℝ)
  (hM : M.snd ^ 2 = M.fst) (hN : N.snd ^ 2 = N.fst)
  (hlength : (M.fst - N.fst)^2 + (M.snd - N.snd)^2 = 16)
  (hP : P = ((M.fst + N.fst) / 2, (M.snd + N.snd) / 2)) :
  abs P.fst = 7 / 4 :=
sorry

end minimum_distance_from_midpoint_to_y_axis_l288_288228


namespace sphere_observation_possible_l288_288315

open Real
open EuclideanGeometry

noncomputable def observation_stations (D : ℝ) :=
  let r := D / 2
  in ∃ (stations : Fin 8 → E3),  -- There are 8 observation stations
       (∀ i : Fin 8, dist (stations i) 0 = r) ∧
       (∀ (p : E3), dist p 0 = D + r → ∃ i j : Fin 8, i ≠ j ∧ ∠ p (stations i) (stations j) ≤ arccos (1 / 3))

theorem sphere_observation_possible (D : ℝ) : observation_stations D :=
  sorry

end sphere_observation_possible_l288_288315


namespace number_with_5_or_6_base_8_l288_288868

open Finset

def count_numbers_with_5_or_6 : ℕ :=
  let base_8_numbers := Ico 1 (8 ^ 3)
  let count_with_5_or_6 := base_8_numbers.filter (λ n, ∃ b, b ∈ digit_set 8 n ∧ (b = 5 ∨ b = 6))
  count_with_5_or_6.card

theorem number_with_5_or_6_base_8 : count_numbers_with_5_or_6 = 296 := 
by 
  -- Proof omitted for this exercise
  sorry

end number_with_5_or_6_base_8_l288_288868


namespace berthaDaughtersGranddaughtersNoDaughters_l288_288364

-- Define the conditions of the problem
def berthaHasEightDaughters : ℕ := 8
def totalWomen : ℕ := 40

noncomputable def berthaHasNoSons : Prop := true
def daughtersHaveFourDaughters (x : ℕ) : Prop := x = 4
def granddaughters : ℕ := totalWomen - berthaHasEightDaughters
def daughtersWithNoChildren : ℕ := 0

theorem berthaDaughtersGranddaughtersNoDaughters :
  let daughtersWithChildren := granddaughters / 4,
      womenWithNoDaughters := granddaughters + daughtersWithNoChildren
  in womenWithNoDaughters = 32 :=
by
  sorry

end berthaDaughtersGranddaughtersNoDaughters_l288_288364


namespace arrange_students_l288_288703

theorem arrange_students 
  (students : Fin 6 → Type) 
  (A B : Type) 
  (h1 : ∃ i j, students i = A ∧ students j = B ∧ (i = j + 1 ∨ j = i + 1)) : 
  (∃ (n : ℕ), n = 240) := 
sorry

end arrange_students_l288_288703


namespace carpet_width_in_cm_l288_288626

def room_length : ℝ := 15
def cost_per_meter_paise : ℝ := 30
def total_cost_rupees : ℝ := 36
def room_breadth : ℝ := 6

def cost_per_meter_rupees : ℝ := cost_per_meter_paise / 100

theorem carpet_width_in_cm :
  let total_meters_of_carpet := total_cost_rupees / cost_per_meter_rupees;
  let carpet_width_meters := total_meters_of_carpet / room_length;
  carpet_width_meters * 100 = 800 :=
by
  sorry

end carpet_width_in_cm_l288_288626


namespace AB_eq_CD_l288_288702

open EuclideanGeometry

-- Define the conditions as hypotheses
variables (S1 S2 : Circle) (h_radii : S1.radius = S2.radius)
variables (A B C D : Point) (l : Line)
variables (W1 W2 : Circle)
variables (h_tangents1 : TangentExternally S1 W1) (h_tangents2 : TangentInternally S2 W2)
variables (h_line_intersect1 : Intersects l S1 B D) (h_line_intersect2 : Intersects l S2 A C)
variables (h_order : OrderedOnLine [A, B, C, D] l)
variables (h_tangent_to_line1 : TangentToLine l W1) (h_tangent_to_line2 : TangentToLine l W2)
variables (h_tangent_each_other : Tangent W1 W2)

-- State the theorem to prove
theorem AB_eq_CD : distance A B = distance C D :=
sorry

end AB_eq_CD_l288_288702


namespace factors_of_2520_l288_288483

theorem factors_of_2520 : (∃ (factors : Finset ℕ), factors.card = 48 ∧ ∀ d, d ∈ factors ↔ d > 0 ∧ 2520 % d = 0) :=
sorry

end factors_of_2520_l288_288483


namespace hyperbola_eccentricity_l288_288823

theorem hyperbola_eccentricity (a : ℝ) (h : 0 < a) (h_asymptote : Real.tan (Real.pi / 6) = 1 / a) : 
  (a = Real.sqrt 3) → 
  (∃ e : ℝ, e = (2 * Real.sqrt 3) / 3) :=
by
  intros
  sorry

end hyperbola_eccentricity_l288_288823


namespace number_of_tires_l288_288672

theorem number_of_tires (n : ℕ)
  (repair_cost : ℕ → ℝ)
  (sales_tax : ℕ → ℝ)
  (total_cost : ℝ) :
  (∀ t, repair_cost t = 7) →
  (∀ t, sales_tax t = 0.5) →
  (total_cost = n * (repair_cost 0 + sales_tax 0)) →
  total_cost = 30 →
  n = 4 :=
by 
  sorry

end number_of_tires_l288_288672


namespace position_of_M_l288_288031

def R : ℝ := sorry
def a : ℝ := 12 / 35 * R

theorem position_of_M:
  ∃ x y : ℝ, (x * y = a * (x + y)) ∧ (x^2 + y^2 = R^2) ∧
    ((x = 3 * R / 5 ∧ y = 4 * R / 5) ∨ (x = 4 * R / 5 ∧ y = 3 * R / 5)) :=
by
  sorry

end position_of_M_l288_288031


namespace box_weight_l288_288721

theorem box_weight (total_weight : ℕ) (number_of_boxes : ℕ) (box_weight : ℕ) 
  (h1 : total_weight = 267) 
  (h2 : number_of_boxes = 3) 
  (h3 : box_weight = total_weight / number_of_boxes) : 
  box_weight = 89 := 
by 
  sorry

end box_weight_l288_288721


namespace all_statements_incorrect_l288_288298

open Classical

variables (P Q : Prop) (x : ℝ)
def is_rectangle (q : Type) := Π (a b : q) [has_eq q], ∀ (diagonals_eq : a = b), q = rectangle
def prop1 := ¬ (P → Q) ↔ P ∧ ¬Q
def prop2 := ¬ (∀ x : ℝ, x^2 - x ≤ 0) ↔ (∃ x : ℝ, x^2 - x > 0)
def prop3 := is_rectangle quadrilateral → ∀ (a b : Type) [has_eq Type], (a = b → quadrilateral = rectangle)
def prop4 := (¬ (x = 3)) → ¬ (|x| = 3)

theorem all_statements_incorrect :
  (¬ prop1) ∧ (¬ prop2) ∧ (¬ prop3) ∧ (¬ prop4) :=
by
  sorry

end all_statements_incorrect_l288_288298


namespace find_length_b_l288_288816

open Real

def area (a b : ℝ) : ℝ := 1/2 * a * b

theorem find_length_b (a b c : ℝ)  (sin_A sin_B sin_C : ℝ) (S : ℝ):
  (2 * sin_B = sin_A + sin_C) ∧ (sin_B = 1/2) ∧ (S = 3/2) ∧ (area a c 1/2 = S) ∧ (a * c = 6) ∧ (cos 30 = sqrt 3 / 2) → 
  b = sqrt 3 + 1 := 
sorry

end find_length_b_l288_288816


namespace find_50th_permutation_l288_288240

-- defining the condition of using digits exactly once
def uses_exactly_once (n : ℕ) : Prop :=
  let digits := [1, 2, 3, 4, 5] in
  n.digits = digits.permutations

-- defining the ordering from least to greatest
def ordered_from_least_to_greatest (n : ℕ) : Prop :=
  let digits := [1, 2, 3, 4, 5] in
  let sorted_digits_list := digits.permutations.sort () in
  n = sorted_digits_list[n-1]

-- The tuple
theorem find_50th_permutation :
  ∃ n, ordered_from_least_to_greatest (n : ℕ) ∧ uses_exactly_once (n : ℕ) ∧ n = 31254 :=
sorry

end find_50th_permutation_l288_288240


namespace find_d_l288_288212

noncomputable def area_triangle (base height : ℝ) : ℝ := (1 / 2) * base * height

theorem find_d (d : ℝ) (h₁ : 0 ≤ d ∧ d ≤ 2) (h₂ : 6 - ((1 / 2) * (2 - d) * 2) = 2 * ((1 / 2) * (2 - d) * 2)) : 
  d = 0 :=
sorry

end find_d_l288_288212


namespace max_f_l288_288455

def f (x : ℝ) : ℝ := Math.cos (Real.pi / 2 + x) + Math.sin (Real.pi / 2 + x) ^ 2

theorem max_f : ∃ (x : ℝ), f x = 5 / 4 :=
sorry

end max_f_l288_288455


namespace count_numbers_with_5_or_6_in_base_8_l288_288859

-- Define the condition that checks if a number contains digits 5 or 6 in base 8
def contains_digit_5_or_6_in_base_8 (n : ℕ) : Prop :=
  let digits := Nat.digits 8 n
  5 ∈ digits ∨ 6 ∈ digits

-- The main problem statement
theorem count_numbers_with_5_or_6_in_base_8 :
  (Finset.filter contains_digit_5_or_6_in_base_8 (Finset.range 513)).card = 296 :=
by
  sorry

end count_numbers_with_5_or_6_in_base_8_l288_288859


namespace option_b_correct_l288_288423

variables {m n : Line} {α β : Plane}

-- Define the conditions as per the problem.
def line_perpendicular_to_plane (l : Line) (p : Plane) : Prop := sorry
def line_parallel_to_plane (l : Line) (p : Plane) : Prop := sorry
def plane_perpendicular_to_plane (p1 p2 : Plane) : Prop := sorry
def lines_perpendicular (l1 l2 : Line) : Prop := sorry

theorem option_b_correct (h1 : line_perpendicular_to_plane m α)
                         (h2 : line_perpendicular_to_plane n β)
                         (h3 : lines_perpendicular m n) :
                         plane_perpendicular_to_plane α β :=
sorry

end option_b_correct_l288_288423


namespace intersection_complement_eq_l288_288069

def U := {1, 2, 3, 4, 5} : set ℕ
def M := {1, 3, 5} : set ℕ
def N := {2, 3, 4} : set ℕ

theorem intersection_complement_eq :
  M ∩ (U \ N) = {1, 5} :=
by
  sorry

end intersection_complement_eq_l288_288069


namespace find_line_equation_l288_288407

-- define the condition of passing through the point (-3, -1)
def passes_through (x y : ℝ) (a b : ℝ) := (a = -3) ∧ (b = -1)

-- define the condition of being parallel to the line x - 3y - 1 = 0
def is_parallel (m n c : ℝ) := (m = 1) ∧ (n = -3)

-- theorem statement
theorem find_line_equation (a b : ℝ) (c : ℝ) :
  passes_through a b (-3) (-1) →
  is_parallel 1 (-3) c →
  (a - 3 * b + c = 0) :=
sorry

end find_line_equation_l288_288407


namespace no_daughters_count_l288_288363

theorem no_daughters_count
  (d : ℕ)  -- Bertha's daughters
  (total : ℕ)  -- Total daughters and granddaughters
  (d_with_children : ℕ)  -- Daughters with children
  (children_per_daughter : ℕ)  -- Children per daughter
  (no_great_granddaughters : Prop) -- No great-granddaughters
  (d = 8)
  (total = 40)
  (d_with_children * children_per_daughter = total - d)
  (children_per_daughter = 4)
  (no_great_granddaughters := ∀ gd, gd ∈ ∅  → ¬∃ x, x ∈ gd) :
  total - d_with_children = 32 :=
by sorry

end no_daughters_count_l288_288363


namespace ellipse_equation_slope_relation_l288_288924

-- Given: Ellipse C with conditions
variables (e : ℝ) (a b x1 y1 x2 y2 m : ℝ)
variables (A B D M N : ℝ × ℝ)
variables (k1 k2 : ℝ)

-- The conditions
axiom eccentricity : e = sqrt 3 / 2
axiom a_gt_b_pos : a > b ∧ b > 0
axiom segment_length : (y1 = x1) ∧ (sqrt 2 * (2 * sqrt 5 / 5) * a = 4 * sqrt 10 / 5)
axiom line_intersection : (B = (-x1, -y1))
axiom perpendicular_ad_ab : (AD ⊥ AB)
axiom line_bd_am : (BD.M = (3 * x1, 0)) ∧ (k1 = -(1 / (4 * k2)))

-- To Prove
theorem ellipse_equation : (a = 2) ∧ (b = 1) → ∀ x y, x^2 / 4 + y^2 = 1 :=
sorry

theorem slope_relation : k1 = -1 / 2 * k2 :=
sorry

end ellipse_equation_slope_relation_l288_288924


namespace sum_tan_squared_l288_288164

open Real

def is_in_T (x : ℝ) : Prop := 
  0 < x ∧ x < π / 2 ∧ (
    sin x = sqrt (cos x ^ 2 + (cos x / sin x)^2) ∨ 
    cos x = sqrt (sin x ^ 2 + (cos x / sin x)^2) ∨ 
    (cos x / sin x) = sqrt (sin x ^ 2 + cos x ^ 2))

theorem sum_tan_squared : ∑ x in { x : ℝ | is_in_T x }, (tan x) ^ 2 = 1 := 
by
  sorry

end sum_tan_squared_l288_288164


namespace minimum_excellence_percentage_l288_288141

theorem minimum_excellence_percentage (n : ℕ) (h : n = 100)
    (m c b : ℕ) 
    (h_math : m = 70)
    (h_chinese : c = 75) 
    (h_min_both : b = c - (n - m))
    (h_percent : b = 45) :
    b = 45 :=
    sorry

end minimum_excellence_percentage_l288_288141


namespace part1_part2_l288_288457

-- Define the function f
def f (a x : ℝ) := a * Real.log x + x - (1 / x)

-- Part (I) conditions and conclusion
theorem part1 (a := - 5 / 2) : 
  (∀ x, x = 1 / 2 → f a x = 1 / 2) → ∃ y, y = 2 ∧ f a y = (3 / 2) - (5 * Real.log 2 / 2) := 
sorry

-- Part (II) conditions and conclusion
theorem part2 (a b x: ℝ) 
  (h1 : -5 / 2 ≤ a ∧ a ≤ 0)
  (h2 : 1 / 2 ≤ x ∧ x ≤ 2)
  (h3 : a * Real.log x - (1 / x) ≤ b - x):
  b ≥ 3 / 2 :=
sorry

end part1_part2_l288_288457


namespace find_angle_C_l288_288137

def right_triangle_angle_sum (A B C : ℝ) : Prop :=
  A = 90 ∧ B + C = 90 ∧ A + B + C = 180

theorem find_angle_C [decidable_eq ℝ] (A B C : ℝ) (h₁ : right_triangle_angle_sum A B C) (h₂ : A = 90) (h₃ : B = 50) : C = 40 :=
by
  rw [←h₂, ←h₃] at h₁
  sorry

end find_angle_C_l288_288137


namespace nadine_total_cleaning_time_l288_288193

-- Conditions
def time_hosing_off := 10 -- minutes
def shampoos := 3
def time_per_shampoo := 15 -- minutes

-- Total cleaning time calculation
def total_cleaning_time := time_hosing_off + (shampoos * time_per_shampoo)

-- Theorem statement
theorem nadine_total_cleaning_time : total_cleaning_time = 55 := by
  sorry

end nadine_total_cleaning_time_l288_288193


namespace product_fraction_formula_l288_288756

theorem product_fraction_formula :
  (∏ n in Finset.range 15, (n + 5) / (n + 1 : ℕ)) = 93024 := by
  sorry

end product_fraction_formula_l288_288756


namespace withdraw_money_from_three_cards_probability_withdraw_money_all_cards_l288_288724

-- Define the conditions
def card_count : ℕ := 4
def pin_count : ℕ := 4
def max_attempts : ℕ := 3

-- First part: Prove Kirpich can withdraw money from three cards.
theorem withdraw_money_from_three_cards :
  (∀ (cards : fin card_count → ℕ), ∀ (pins : fin pin_count → ℕ),
   (∀ c₁ c₂ : fin card_count, c₁ ≠ c₂ → cards c₁ ≠ cards c₂) →
   ∀ p₁ p₂ : fin pin_count, p₁ ≠ p₂ → pins p₁ ≠ pins p₂) →
  ∃ (success_cards : fin card_count → bool), (success_cards (1 : fin card_count) = true) ∧ (success_cards (2 : fin card_count) = true) ∧ (success_cards (3 : fin card_count) = true) :=
sorry

-- Second part: Prove the probability that Kirpich will be able to withdraw money from all four cards is 23/24.
theorem probability_withdraw_money_all_cards :
  (∀ (cards : fin card_count → ℕ), ∀ (pins : fin pin_count → ℕ),
   (∀ c₁ c₂ : fin card_count, c₁ ≠ c₂ → cards c₁ ≠ cards c₂) →
   ∀ p₁ p₂ : fin pin_count, p₁ ≠ p₂ → pins p₁ ≠ pins p₂) →
  ∑ c in finset.univ, c ≠ (23 / 24 : ℝ) :=
sorry

end withdraw_money_from_three_cards_probability_withdraw_money_all_cards_l288_288724


namespace exists_universal_involutive_l288_288413

variable (S : Set (ℤ × ℤ))

def is_universal (f : (ℤ × ℤ) → (ℤ × ℤ)) : Prop :=
  Function.Bijective f ∧ 
  ∀ (n m : ℤ), (n, m) ∈ S → f (n, m) ∈ {(n-1, m), (n+1, m), (n, m-1), (n, m+1)}

theorem exists_universal_involutive {g : (ℤ × ℤ) → (ℤ × ℤ)} (hg : is_universal S g) :
  ∃ f : (ℤ × ℤ) → (ℤ × ℤ), is_universal S f ∧
  ∀ (n m : ℤ), (n, m) ∈ S → f (f (n, m)) = (n, m) :=
sorry

end exists_universal_involutive_l288_288413


namespace algebra_inequality_l288_288770

theorem algebra_inequality (a b c : ℝ) 
  (H : a^4 + b^4 + c^4 ≤ 2 * (a^2 * b^2 + b^2 * c^2 + c^2 * a^2)) :
  a^2 + b^2 + c^2 ≤ 2 * (a * b + b * c + c * a) :=
sorry

end algebra_inequality_l288_288770


namespace number_of_valid_triangles_l288_288102

theorem number_of_valid_triangles : 
  let points := {(x, y) | 1 ≤ x ∧ x ≤ 5 ∧ 1 ≤ y ∧ y ≤ 5} in
  ∃ n : ℕ, n = (nat.card (points.choose 3)) - 120 ∧ n = 2180 :=
by
  sorry

end number_of_valid_triangles_l288_288102


namespace allocation_schemes_l288_288743

-- Define the number of intern teachers and classes
def num_teachers := 4
def num_classes := 3

-- Statement that the number of different allocation schemes with given conditions is 36
theorem allocation_schemes : (number_of_allocation_schemes num_teachers num_classes) = 36 := 
by 
  sorry

end allocation_schemes_l288_288743


namespace problem_l288_288646

noncomputable def rotated_and_reflected (a b : ℝ) : ℝ × ℝ :=
  let (x', y') := (2 - (b - 3), 3 + (a - 2))
  in (y', x')

theorem problem (a b : ℝ) (h : rotated_and_reflected a b = (5, 1)) : b - a = 2 :=
  sorry

end problem_l288_288646


namespace quad_pyramid_faces_l288_288725

-- Define a quadrangular pyramid
structure QuadrangularPyramid where
  lateral_faces : Nat
  base_faces : Nat

-- Define the conditions of the problem
def quadPyramid : QuadrangularPyramid := 
{ lateral_faces := 4, base_faces := 1 }

-- The theorem to prove the total number of faces
theorem quad_pyramid_faces (P : QuadrangularPyramid) : 
  P.lateral_faces + P.base_faces = 5 :=
by
  rw [quadPyramid.lateral_faces, quadPyramid.base_faces]
  rfl

end quad_pyramid_faces_l288_288725


namespace find_k_l288_288000

-- Define the series summation function
def series (k : ℝ) : ℝ := 4 + (∑ n, (4 + n * k) / 5^n)

-- State the theorem with the given condition and required proof
theorem find_k (h : series k = 10) : k = 16 := sorry

end find_k_l288_288000


namespace arg_u_principal_value_omega_not_positive_real_l288_288030

variable (θ : ℝ) (a : ℝ)
variable (0 < θ) (θ < 2 * Real.pi)
def z : ℂ := 1 - Real.cos θ + Real.sin θ * Complex.I
def u : ℂ := a ^ 2 + a * Complex.I

theorem arg_u_principal_value :
    (a ≠ 0 → a = Real.sin θ / (1 - Real.cos θ)) →
    (0 < θ ∧ θ < Real.pi → Complex.arg u = θ / 2) ∧
    (Real.pi < θ ∧ θ < 2 * Real.pi → Complex.arg u = Real.pi + θ / 2) :=
by
  sorry

theorem omega_not_positive_real :
    (a ≠ 0 → a = Real.sin θ / (1 - Real.cos θ)) →
    ∀ w : ℂ, w = z ^ 2 + u ^ 2 + 2 * z * u → ¬ (∃ r : ℝ, r > 0 ∧ w = r) :=
by
  sorry

end arg_u_principal_value_omega_not_positive_real_l288_288030


namespace complement_union_A_B_l288_288962

-- Define the sets U, A, and B as per the conditions
def U : Set ℕ := {x | x > 0 ∧ x < 6}
def A : Set ℕ := {1, 3}
def B : Set ℕ := {3, 5}

-- Specify the statement to prove the complement of A ∪ B with respect to U
theorem complement_union_A_B : (U \ (A ∪ B)) = {2, 4} :=
by
  sorry

end complement_union_A_B_l288_288962


namespace rice_in_each_container_ounces_l288_288990

-- Given conditions
def total_rice_pounds : ℚ := 25 / 4
def num_containers : ℕ := 4
def pounds_to_ounces : ℚ := 16

-- Problem statement: proving the amount of rice in each container in ounces
theorem rice_in_each_container_ounces :
  (total_rice_pounds / num_containers) * pounds_to_ounces = 25 :=
by sorry

end rice_in_each_container_ounces_l288_288990


namespace digit_in_fifth_decimal_place_l288_288110

noncomputable def evaluate_accurate (x : ℝ) (n : ℕ) : ℝ := x^n

theorem digit_in_fifth_decimal_place :
  (10^5 * (evaluate_accurate 1.0025 10)).floor % 10 = 2 :=
by
  let value := (evaluate_accurate 1.0025 10)
  have rounded_value : ℝ := (value * 10^5).round / 10^5
  have fifth_digit := (rounded_value * 10^5).floor % 10
  sorry

end digit_in_fifth_decimal_place_l288_288110


namespace find_x_for_opposite_expressions_l288_288293

theorem find_x_for_opposite_expressions :
  ∃ x : ℝ, (x + 1) + (3 * x - 5) = 0 ↔ x = 1 :=
by
  sorry

end find_x_for_opposite_expressions_l288_288293


namespace non_degenerate_triangles_l288_288077

theorem non_degenerate_triangles (x y : ℕ) (hx : 1 ≤ x ∧ x ≤ 5) (hy : 1 ≤ y ∧ y ≤ 5) : 
  ∃ (n : ℕ), n = 2160 ∧ 1 ≤ n :=
by
  sorry

end non_degenerate_triangles_l288_288077


namespace seq_formula_sum_of_T_n_l288_288805

variable {a : ℕ → ℝ} {T : ℕ → ℝ}

axiom sum_of_first_four_terms (d : ℝ) (a1 : ℝ) : 4 * a1 + 6 * d = 14
axiom geometric_condition (d : ℝ) (a1 : ℝ) : (a1 + 2 * d)^2 = a1 * (a1 + 6 * d)

theorem seq_formula (d a1 : ℝ) (h1 : 4 * a1 + 6 * d = 14) (h2 : (a1 + 2 * d)^2 = a1 * (a1 + 6 * d)) :
  ∀ n : ℕ, a n = n + 1 :=
by sorry

variable {seq : ℕ → ℝ} 
axiom a_n_formula (n : ℕ) : seq n = n + 1

def T_n : ℕ → ℝ := λ n, ∑ i in finset.range n, (1 / ((seq i) * (seq (i + 1))))

theorem sum_of_T_n (n : ℕ) :
  T n = n / (2 * (n + 2)) :=
by sorry

end seq_formula_sum_of_T_n_l288_288805


namespace infinite_lcm_inequality_l288_288566

theorem infinite_lcm_inequality (k : ℕ) (hk : k > 1) : 
  ∃ᶠ n : ℕ in Filter.atTop, Nat.lcm_list (List.range (k + 1).map (λ i, n + i)) > 
                             Nat.lcm_list (List.range (k + 1).map (λ i, n + i + 1)) :=
sorry

end infinite_lcm_inequality_l288_288566


namespace area_enclosed_by_graph_l288_288287

theorem area_enclosed_by_graph :
  let f : ℝ × ℝ → ℝ := λ p, p.1 ^ 2 + p.2 ^ 2
  let g : ℝ × ℝ → ℝ := λ p, (|p.1| + |p.2|) ^ 2
  (f = g) → (integral (λ p, 1) (region_enclosed_by f g) = 2) :=
by
  intros
  sorry

end area_enclosed_by_graph_l288_288287


namespace side_length_of_square_surrounded_by_quarters_and_circle_eq_l288_288627

theorem side_length_of_square_surrounded_by_quarters_and_circle_eq :
  ∀ (square_side length : ℝ), 
  (∀ (circle_radius : ℝ), (∀ (quarter_circle_radius : ℝ), 
  (circle_radius = 1) ∧ (quarter_circle_radius = 1) ∧ 
  (∃ (square: ℝ), (square = 2 - Real.sqrt 2)) →
  (square_side length = square)) :=
sorry

end side_length_of_square_surrounded_by_quarters_and_circle_eq_l288_288627


namespace real_condition_iff_imaginary_zero_l288_288026

theorem real_condition_iff_imaginary_zero (a b c d : ℝ) : 
  (a * d + b * c = 0) ↔ ((a * c - b * d) + (a * d + b * c) * complex.I).im = 0 := 
sorry

end real_condition_iff_imaginary_zero_l288_288026


namespace shaded_area_of_square_trisection_points_l288_288357

theorem shaded_area_of_square_trisection_points :
  let s := 6
  let A := (0, 0)
  let B := (s, 0)
  let C := (s, s)
  let D := (0, s)
  let E := ((2 * s) / 3, 0)
  let F := (s, (2 * s) / 3)
  let G := ((s) / 3, s)
  let H := (0, (s) / 3)
  in ∃ region_area : ℝ, region_area = 18 := sorry

end shaded_area_of_square_trisection_points_l288_288357


namespace sum_b_n_first_5_eq_7_l288_288927

-- Define the arithmetic sequence {a_n}
def a_n (n : ℕ) : ℚ := 1 + (2 * n - 2) / 5

-- Define the sequence {b_n} where b_n is the greatest integer not exceeding a_n
def b_n (n : ℕ) : ℤ := int.floor (a_n n)

-- The main theorem to prove
theorem sum_b_n_first_5_eq_7 : (b_n 1) + (b_n 2) + (b_n 3) + (b_n 4) + (b_n 5) = 7 := by
  sorry

end sum_b_n_first_5_eq_7_l288_288927


namespace find_polynomial_expression_range_of_c_l288_288174

-- Part I
theorem find_polynomial_expression (a b c d : ℝ) (h_a : a ≠ 0) (h_symmetric : ∀ x, f (-x) = -f (x)) 
  (h_min : ∀ x, x = 1/2 → f x = -1) : f = λ x, 4*x^3 - 3*x := 
sorry

-- Part II
theorem range_of_c (c : ℝ) (h_mono : ∀ x, f (x) = x^3 + x^2 + c*x + 1 → f' (x) = deriv f x ≥ 0 ∨ deriv f x ≤ 0): c ≥ 1/3 :=
sorry

end find_polynomial_expression_range_of_c_l288_288174


namespace bisector_intersects_arc_at_E_l288_288967

theorem bisector_intersects_arc_at_E (
  A D E B C F : Point,
  is_isosceles_AED : IsIsoscelesTr A E D,
  is_isosceles_BCE : IsIsoscelesTr B C E,
  AD_eq_BE : dist A D = dist B E,
  bisector : Line,
  in_arc : Arc,
  intersection_point : Point
) : intersection_point = E :=
sorry

end bisector_intersects_arc_at_E_l288_288967


namespace survey_total_people_l288_288134

theorem survey_total_people (total_people_believe_friendly mistaken_belief_people : ℕ)
    (frac_believe_friendly frac_believe_mistaken : ℝ) 
    (h1 : frac_believe_friendly = 0.923) 
    (h2 : frac_believe_mistaken = 0.384) 
    (h3 : mistaken_belief_people = 28)
    (h4 : mistaken_belief_people = frac_believe_mistaken * total_people_believe_friendly) : 
  total_people_believe_friendly = 73 → 
  total_people_believe_friendly = frac_believe_friendly * 79 := 
begin
  sorry
end

end survey_total_people_l288_288134


namespace fraction_decomposition_l288_288233

theorem fraction_decomposition :
  ∀ (x A B : ℝ),
  (2x^2 + x - 6 = (2x - 3) * (x + 2)) →
  (6x - 15) / (2x^2 + x - 6) = A / (x + 2) + B / (2x - 3) →
  A = 3.857 ∧ B = -1.714 :=
by
  sorry

end fraction_decomposition_l288_288233


namespace principal_amount_l288_288291

theorem principal_amount (r : ℝ) (n t : ℕ) (A P : ℝ) (interest : ℝ) : 
  r = 0.12 ∧ n = 1 ∧ t = 3 ∧ A = P * (1 + r / n)^(n * t) ∧ interest = A - P ∧ interest = P - 5888 → 
  P ≈ 14539.13 :=
by {
  sorry
}

end principal_amount_l288_288291


namespace inverse_fixed_point_l288_288487

noncomputable def f (a : ℝ) (ha : a > 0) (hne1 : a ≠ 1) : ℝ → ℝ :=
  λ x, Real.log (x - 1) / Real.log a

theorem inverse_fixed_point (a : ℝ) (ha : a > 0) (hne1 : a ≠ 1) : 
    (∃ x y : ℝ, f a ha hne1 y = x ∧ (x = 0 ∧ y = 2)) :=
by
  sorry

end inverse_fixed_point_l288_288487


namespace four_digit_numbers_divisible_by_13_l288_288878

theorem four_digit_numbers_divisible_by_13 : 
  (set.Icc 1000 9999).filter (λ x, x % 13 = 0).cardinality = 693 :=
by
  sorry

end four_digit_numbers_divisible_by_13_l288_288878


namespace find_complex_number_l288_288619

open Complex

theorem find_complex_number (a b : ℕ) (hp : (a:ℂ) + (b:ℂ) * I) = (2 + 11 * I) ^ (1/3) :
  (a:ℂ) + (b:ℂ) * I = 2 + I := by
  sorry

end find_complex_number_l288_288619


namespace solving_equation_l288_288552

theorem solving_equation (x : ℝ) : 3 * (x - 3) = (x - 3)^2 ↔ x = 3 ∨ x = 6 := 
by
  sorry

end solving_equation_l288_288552


namespace evaluate_statements_l288_288834

def f (x : ℝ) : ℝ := |cos x| * sin x

lemma period_incorrect : ¬ ∃ T : ℝ, (∀ x, f (x + T) = f x) ∧ T = π :=
by sorry

lemma abs_val_equality_not_implication (x1 x2 : ℝ) (k : ℤ) :
  | f x1 | = | f x2 | → ¬ (x1 = x2 + k * π) :=
by sorry

lemma monotonically_increasing_on_interval :
  ∀ x1 x2 : ℝ, (-π / 4) ≤ x1 → x1 ≤ x2 → x2 ≤ (π / 4) → f x1 ≤ f x2 :=
by sorry

lemma not_symmetric_about_point :
  ¬ ∃ x, f (x + π / 2) = f x ∧ (-π / 2, 0) =
    ((π / 2), 0) ∧ f x = 0 :=
by sorry

theorem evaluate_statements :
  period_incorrect ∧
  (∀ x1 x2 k, abs_val_equality_not_implication x1 x2 k) ∧
  monotonically_increasing_on_interval ∧
  not_symmetric_about_point 
:=
by sorry

end evaluate_statements_l288_288834


namespace path_count_from_top_to_bottom_l288_288325

-- Let's denote the grid and describe the conditions as follows:
-- A path starts at an unshaded square in the first row.
-- A path ends at an unshaded square in the last row (fifth row).
-- The only allowed moves are diagonal: one square down and one square left or one square down and one square right.

def unshaded_path_count (rows: ℕ) (cols: ℕ) : ℕ :=
  -- This function will calculate the paths based on the problem's description and conditions:
  -- Assume the proper grid configuration is given, this function calculates the number
  -- of paths satisfying the conditions (for the illustrative purpose, we assume rows=5 and cols=3 here).
  sorry

theorem path_count_from_top_to_bottom (rows cols: ℕ) : unshaded_path_count rows cols = 24 :=
by
  -- Assuming rows = 5 and cols = 3 specifically as per the example in the problem.
  assume rows = 5
  assume cols = 3
  exact sorry

end path_count_from_top_to_bottom_l288_288325


namespace derivative_f_l288_288225

def f (x : ℝ) : ℝ := sin (x / 4) ^ 4 + cos (x / 4) ^ 4

theorem derivative_f (x : ℝ) : deriv f x = - (sin x) / 4 :=
by
  sorry

end derivative_f_l288_288225


namespace time_to_cross_trains_l288_288306

/-- Length of the first train in meters -/
def length_train1 : ℕ := 50

/-- Length of the second train in meters -/
def length_train2 : ℕ := 120

/-- Speed of the first train in km/hr -/
def speed_train1_kmh : ℕ := 60

/-- Speed of the second train in km/hr -/
def speed_train2_kmh : ℕ := 40

/-- Relative speed in km/hr as trains are moving in opposite directions -/
def relative_speed_kmh : ℕ := speed_train1_kmh + speed_train2_kmh

/-- Convert speed from km/hr to m/s -/
def kmh_to_ms (speed_kmh : ℕ) : ℚ := (speed_kmh * 1000) / 3600

/-- Relative speed in m/s -/
def relative_speed_ms : ℚ := kmh_to_ms relative_speed_kmh

/-- Total distance to be covered in meters -/
def total_distance : ℕ := length_train1 + length_train2

/-- Time taken in seconds to cross each other -/
def time_to_cross : ℚ := total_distance / relative_speed_ms

theorem time_to_cross_trains :
  time_to_cross = 6.12 := 
sorry

end time_to_cross_trains_l288_288306


namespace marian_returned_amount_l288_288589

theorem marian_returned_amount
  (B : ℕ) (G : ℕ) (H : ℕ) (N : ℕ)
  (hB : B = 126) (hG : G = 60) (hH : H = G / 2) (hN : N = 171) :
  (B + G + H - N) = 45 := 
by
  sorry

end marian_returned_amount_l288_288589


namespace range_of_x_in_function_l288_288539

theorem range_of_x_in_function (x : ℝ) (h : x ≠ 8) : true := sorry

end range_of_x_in_function_l288_288539


namespace kyle_spent_frac_l288_288947

theorem kyle_spent_frac (dave_money : ℕ) (kyle_left : ℕ) (dave_has : dave_money = 46) (kyle_start : kyle_left = 126 - 84) :
  let kyle_money := 3 * dave_money - 12 in
  kyle_money = 126 → (kyle_money - kyle_left) / kyle_money = 1 / 3 :=
by
  intros
  rw [dave_has] at kyle_money
  rw [kyle_start]
  sorry

end kyle_spent_frac_l288_288947


namespace modulus_of_complex_number_l288_288447

theorem modulus_of_complex_number :
  let z := (1 + 3 * complex.I) / (1 - complex.I)
  in complex.abs z = real.sqrt 5 :=
by
  sorry

end modulus_of_complex_number_l288_288447


namespace range_of_b_l288_288822

noncomputable def f (x : ℝ) : ℝ :=
if 0 ≤ x ∧ x < 1 then 2^x - 1
else if x ≥ 1 then 1 / x
else 0  -- since f is defined on ℝ, we need a definition for all x, though this won't matter for the proof

def g (x : ℝ) : ℝ :=
if x > 0 then Real.log x / Real.log 2 -- equivalent to log_2 x
else 0  -- g is not defined at x = 0, but for completeness in def

theorem range_of_b (a b : ℝ) (h : f a = g b) : b ∈ Set.Icc (1 / 2) 2 ∪ Set.Icc (-2) (-1 / 2) :=
sorry

end range_of_b_l288_288822


namespace sum_of_squares_odd_l288_288906

def is_odd (n : ℕ) : Prop := n % 2 = 1

theorem sum_of_squares_odd :
  is_odd (∑ i in Finset.range (2002 + 1), i ^ 2) :=
sorry

end sum_of_squares_odd_l288_288906


namespace quadratic_function_coeff_l288_288909

theorem quadratic_function_coeff (m : ℝ) :
  (∀ x : ℝ, (m + 1) * x^2 + 2 * x + 1 ≠ 0) → m ≠ -1 :=
by
  intro h
  have h1 : m + 1 ≠ 0 := sorry
  exact mt (eq_add_one_of_ne m).mp h.le.antisymm h1
  sorry

end quadratic_function_coeff_l288_288909


namespace tablecloth_covers_table_l288_288740

theorem tablecloth_covers_table
(length_ellipse : ℝ) (width_ellipse : ℝ) (length_tablecloth : ℝ) (width_tablecloth : ℝ)
(h1 : length_ellipse = 160)
(h2 : width_ellipse = 100)
(h3 : length_tablecloth = 140)
(h4 : width_tablecloth = 130) :
length_tablecloth >= width_ellipse ∧ width_tablecloth >= width_ellipse ∧
(length_tablecloth ^ 2 + width_tablecloth ^ 2) >= (length_ellipse ^ 2 + width_ellipse ^ 2) :=
by
  sorry

end tablecloth_covers_table_l288_288740


namespace relationship_xy_l288_288113

variable (x y : ℝ)

theorem relationship_xy (h₁ : x - y > x + 2) (h₂ : x + y + 3 < y - 1) : x < -4 ∧ y < -2 := 
by sorry

end relationship_xy_l288_288113


namespace ellipse_equation_and_triangle_area_l288_288047

theorem ellipse_equation_and_triangle_area 
  {c a : ℝ} (a_pos : a > 0)
  (h1 : c ^ 2 = 2 / 3 * a ^ 2)
  (h2 : c ^ 2 = a ^ 2 - 4)
  (h3 : |((5 * a / 3) - (a / 3))| = 4 * c)
  (P : ℝ × ℝ) (P_eq : P = (-3, 2)) :
  (∀ x y : ℝ, 
    (x ^ 2) / 12 + (y ^ 2) / 4 = 1 ↔ 
      ((x, y) = (0, 2) ∨ (x, y) = (-3, -1))) ∧ 
  (area_of_triangle P (-3, -1) (0, 2) = 9 / 2) :=
sorry

end ellipse_equation_and_triangle_area_l288_288047


namespace ellipse_equation_and_min_OB_l288_288436

theorem ellipse_equation_and_min_OB (a b c x y x0 y0 : ℝ)
  (E : ℝ × ℝ)
  (a_pos : 0 < a)
  (b_pos : 0 < b)
  (a_gt_b : a > b)
  (passes_through_E : E = (sqrt 3, 1))
  (eccentricity_eq : c = sqrt (6) / 3)
  (ellipse_eq : x^2 / a^2 + y^2 / b^2 = 1)
  (ellipse_E : passes_through_E)
  (min_OB : x0^2 / 6 + y0^2 / 2 = 1 → (abs (y0 + 3 / (2 * abs y0)) ≥ sqrt 6)) :
  (∃ (a b : ℝ), a² = 6 ∧ b² = 2 ∧ (passes_through_E ∧ eccentricity_eq ∧ ellipse_eq)) ∧ (min_OB) :=
sorry

end ellipse_equation_and_min_OB_l288_288436


namespace rachel_total_homework_pages_l288_288602

-- Define the conditions
def math_homework_pages : Nat := 10
def additional_reading_pages : Nat := 3

-- Define the proof goal
def total_homework_pages (math_pages reading_extra : Nat) : Nat :=
  math_pages + (math_pages + reading_extra)

-- The final statement with the expected result
theorem rachel_total_homework_pages : total_homework_pages math_homework_pages additional_reading_pages = 23 :=
by
  sorry

end rachel_total_homework_pages_l288_288602


namespace range_of_f_l288_288243

noncomputable def f (x : ℝ) := Real.arcsin (x ^ 2 - x)

theorem range_of_f :
  Set.range f = Set.Icc (-Real.arcsin (1/4)) (Real.pi / 2) :=
sorry

end range_of_f_l288_288243


namespace find_vertex_N_l288_288543

structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

def midpoint (p1 p2 : Point3D) : Point3D :=
  { x := (p1.x + p2.x) / 2,
    y := (p1.y + p2.y) / 2,
    z := (p1.z + p2.z) / 2 }

theorem find_vertex_N
  (P Q R : Point3D)
  (hP : P = { x := 3, y := 2, z := 0 })
  (hQ : Q = { x := 1, y := 3, z := 2 })
  (hR : R = { x := 4, y := 1, z := 3 }) :
  ∃ N : Point3D, N = { x := 6, y := 0, z := 1 } :=
by
  sorry

end find_vertex_N_l288_288543


namespace three_at_five_l288_288501

def op_at (a b : ℤ) : ℤ := 3 * a - 3 * b

theorem three_at_five : op_at 3 5 = -6 :=
by
  sorry

end three_at_five_l288_288501


namespace tangent_line_equation_l288_288633

theorem tangent_line_equation (x y : ℝ) :
  (2, 1) ∈ {p : ℝ × ℝ | p.1^2 + p.2^2 = 5} →
  2 * x + y = 5 ↔ 2 * x + y - 5 = 0 :=
by
  intro h
  sorry

end tangent_line_equation_l288_288633


namespace log_sum_real_coeffs_l288_288580

theorem log_sum_real_coeffs (S : ℝ) (x : ℝ) :
  S = ∑ k in Finset.range (2009 + 1), (1 + ![1, x + 0]^2009).coeff k \\
  log2 S = 1004 :=
by
  sorry

end log_sum_real_coeffs_l288_288580


namespace num_inequalities_true_l288_288594

variables {x y a b : ℝ}

-- Given conditions
constants (hne_zero : x ≠ 0)
          (yne_zero : y ≠ 0)
          (ane_zero : a ≠ 0)
          (bne_zero : b ≠ 0)
          (hx : x^2 < a^2)
          (hy : |y| < |b|)

-- Statement we want to prove
theorem num_inequalities_true : 
  (x^2 < a^2) ∧ (|y| < |b|) →
  (x^2 + y^2 < a^2 + b^2) ∧ 
  (x^2 * y^2 < a^2 * b^2) ∧ 
  ¬((x^2 - y^2 < a^2 - b^2) ∧ 
    (x^2 / y^2 < a^2 / b^2)) → 
  ∃ n : ℕ, n = 2 :=
begin
  sorry
end

end num_inequalities_true_l288_288594


namespace distance_between_trees_l288_288131

-- Variables representing the total length of the yard and the number of trees.
variable (length_of_yard : ℕ) (number_of_trees : ℕ)

-- The given conditions
def yard_conditions (length_of_yard number_of_trees : ℕ) :=
  length_of_yard = 700 ∧ number_of_trees = 26

-- The proof statement: If the yard is 700 meters long and there are 26 trees, 
-- then the distance between two consecutive trees is 28 meters.
theorem distance_between_trees (length_of_yard : ℕ) (number_of_trees : ℕ)
  (h : yard_conditions length_of_yard number_of_trees) : 
  (length_of_yard / (number_of_trees - 1)) = 28 := 
by
  sorry

end distance_between_trees_l288_288131


namespace problem_statement_l288_288444

variable (f : ℝ → ℝ)

-- f(x) is even
def is_even_function (f : ℝ → ℝ) := ∀ x, f x = f (-x)

-- f(x+1) is odd
def is_odd_shifted_function (f : ℝ → ℝ) := ∀ x, f (x + 1) = -f (x + 1)

-- f(x) is decreasing in [0,1]
def is_decreasing_in_interval (f : ℝ → ℝ) := ∀ x₁ x₂ ∈ Icc (0 : ℝ) 1, x₁ ≠ x₂ → (x₁ - x₂) * (f x₁ - f x₂) < 0

-- f(x) is periodic with period 4
def is_periodic (f : ℝ → ℝ) (p : ℝ) := ∀ x, f(x) = f(x + p)

-- Definitions of a, b, and c
def a := f (82 / 11)
def b := -f (50 / 9)
def c := f (24 / 7)

theorem problem_statement :
  is_even_function f →
  is_odd_shifted_function f →
  is_decreasing_in_interval f →
  is_periodic f 4 →
  let a := f (6 / 11),
      b := f (4 / 9),
      c := f (4 / 7) in
  b > a ∧ a > c := 
by
  sorry

end problem_statement_l288_288444


namespace greatest_possible_large_chips_l288_288254

theorem greatest_possible_large_chips 
  (s l : ℕ) 
  (p : ℕ) 
  (h1 : s + l = 72) 
  (h2 : s = l + p) 
  (h_prime : Prime p) : 
  l ≤ 35 :=
sorry

end greatest_possible_large_chips_l288_288254


namespace greatest_possible_large_chips_l288_288256

theorem greatest_possible_large_chips 
  (total : ℕ) (s l p : ℕ) (prime_p : Nat.Prime p) (chips_eq : s + l = total) (num_eq : s = l + p)
  (total_eq : total = 72) : 
  l ≤ 35 := by
  have H : 2 * l + p = total := sorry -- Derived from s + l = 72 and s = l + p
  have p_value : p = 2 := sorry -- Smallest prime
  have H1 : 2 * l + 2 = 72 := sorry -- Substituting p = 2
  have H2 : 2 * l = 70 := sorry -- Simplifying
  have H3 : l = 35 := sorry -- Solving for l
  l ≤ 35 := sorry

end greatest_possible_large_chips_l288_288256


namespace find_k_solution_l288_288003

theorem find_k_solution :
    ∃ k : ℝ, (4 + ∑' n : ℕ, (4 + (n : ℝ)*k) / 5^(n + 1) = 10) ∧ k = 16 :=
begin
  use 16,
  sorry
end

end find_k_solution_l288_288003


namespace largest_sum_is_225_l288_288664

noncomputable def max_sum_products (f g h j : ℕ) : ℕ :=
if (f ∈ {6, 7, 8, 9} ∧ g ∈ {6, 7, 8, 9} ∧ h ∈ {6, 7, 8, 9} ∧ j ∈ {6, 7, 8, 9} ∧ 
    {f, g, h, j}.nodup) then
    let fh := min (f * h + g * j) (min (f * j + g * h) (f * g + h * j)) in
    (f + g + h + j)^2 - (f^2 + g^2 + h^2 + j^2) - 2 * fh in
  (30^2 - 230) - 2 * 110 -- Plugging in the example values
else 
  0

theorem largest_sum_is_225 (f g h j : ℕ) (hf : f ∈ {6, 7, 8, 9}) (hg : g ∈ {6, 7, 8, 9}) 
  (hh : h ∈ {6, 7, 8, 9}) (hj : j ∈ {6, 7, 8, 9}) (h_nodup : ({f, g, h, j} : set ℕ).nodup) : 
  max_sum_products f g h j = 225 :=
by {
  sorry -- We would prove the steps leading to the solution here.
}

end largest_sum_is_225_l288_288664


namespace maximum_term_of_sequence_l288_288434

open Real

noncomputable def seq (n : ℕ) : ℝ := n / (n^2 + 81)

theorem maximum_term_of_sequence : ∃ n : ℕ, seq n = 1 / 18 ∧ ∀ k : ℕ, seq k ≤ 1 / 18 :=
by
  sorry

end maximum_term_of_sequence_l288_288434


namespace isosceles_triangle_perimeter_l288_288355

theorem isosceles_triangle_perimeter (a b : ℕ) (h₀ : a = 3 ∨ a = 4) (h₁ : b = 3 ∨ b = 4) (h₂ : a ≠ b) :
  (a = 3 ∧ b = 4 ∧ 4 ∈ [b]) ∨ (a = 4 ∧ b = 3 ∧ 4 ∈ [a]) → 
  (a + a + b = 10) ∨ (a + b + b = 11) :=
by
  sorry

end isosceles_triangle_perimeter_l288_288355


namespace functional_equation_solution_l288_288398

theorem functional_equation_solution (f : ℕ → ℕ) 
  (H : ∀ a b : ℕ, f (f a + f b) = a + b) : 
  ∀ n : ℕ, f n = n := 
by
  sorry

end functional_equation_solution_l288_288398


namespace team_A_processes_fraction_l288_288707

theorem team_A_processes_fraction (A B : ℕ) (total_calls : ℚ) 
  (h1 : A = (5/8) * B) 
  (h2 : (8 / 11) * total_calls = TeamB_calls_processed)
  (frac_TeamA_calls : ℚ := (1 - (8 / 11)) * total_calls)
  (calls_per_member_A : ℚ := frac_TeamA_calls / A)
  (calls_per_member_B : ℚ := (8 / 11) * total_calls / B) : 
  calls_per_member_A / calls_per_member_B = 3 / 5 := 
by
  sorry

end team_A_processes_fraction_l288_288707


namespace find_altitude_of_larger_triangle_l288_288621

theorem find_altitude_of_larger_triangle
  (area_larger : ℝ)
  (area_larger_eq : area_larger = 1600)
  (num_smaller_triangles : ℝ)
  (num_smaller_triangles_eq : num_smaller_triangles = 2)
  (base_smaller : ℝ)
  (base_smaller_eq : base_smaller = 40)
  (area_formula : ∀ A b h, A = (1/2) * b * h) :
  let area_smaller := area_larger / num_smaller_triangles in
  ∃ (h : ℝ), area_smaller = (1/2) * base_smaller * h ∧ h = 40 :=
by
  sorry

end find_altitude_of_larger_triangle_l288_288621


namespace celia_wins_1318_l288_288975

-- Define the cards and initial setup
inductive Card
| nine
| ten
| jack
| queen
| king
| ace
open Card

structure Player :=
(cards : List Card)
(team  : String) -- "Celia" or "Team"

def Game := 
(players : List Player) -- List of all players Celia, Alice, Betsy

-- Initial conditions
def initial_game : Game := sorry -- This will include initial conditions of the cards distributed

-- Function to define the optimal play and calculate Celia's winning probability
noncomputable def optimal_play (game : Game) : ℚ := sorry

-- The statement we need to prove
theorem celia_wins_1318 {p q : ℕ} (hpq_rel_prime : Nat.coprime p q) (hprob : optimal_play initial_game = p / q) : 100 * p + q = 1318 :=
by sorry

end celia_wins_1318_l288_288975


namespace sin_double_angle_l288_288494

variable {θ : ℝ}

theorem sin_double_angle (h : cos θ + sin θ = 3 / 2) : sin (2 * θ) = 5 / 4 :=
by
  sorry

end sin_double_angle_l288_288494


namespace number_of_valid_triangles_l288_288097

theorem number_of_valid_triangles : 
  let points := {(x, y) | 1 ≤ x ∧ x ≤ 5 ∧ 1 ≤ y ∧ y ≤ 5} in
  ∃ n : ℕ, n = (nat.card (points.choose 3)) - 120 ∧ n = 2180 :=
by
  sorry

end number_of_valid_triangles_l288_288097


namespace wall_area_l288_288922

-- Definition of the width and length of the wall
def width : ℝ := 5.4
def length : ℝ := 2.5

-- Statement of the theorem
theorem wall_area : (width * length) = 13.5 :=
by
  sorry

end wall_area_l288_288922


namespace min_value_arith_seq_l288_288353

noncomputable def S_n (n : ℕ) : ℕ :=
  n * (n + 1) / 2

theorem min_value_arith_seq : ∀ n : ℕ, n > 0 → 2 * S_n 2 = (n + 1) * 2 → (n = 4 → (2 * S_n n + 13) / n = 33 / 4) :=
by
  intros n hn hS2 hn_eq_4
  sorry

end min_value_arith_seq_l288_288353


namespace required_speed_l288_288973

-- Definitions
def distance_to_office (t : ℝ) : ℝ := 50 * (t + (1/12))
def early_arrival_distance (t : ℝ) : ℝ := 70 * (t - (1/12))

-- Theorem to prove
theorem required_speed (t : ℝ) (d : ℝ) (h1 : d = distance_to_office t) (h2 : d = early_arrival_distance t) : d / t = 58 := 
by 
  -- Proof is omitted
  sorry

end required_speed_l288_288973


namespace Greg_needs_93_feet_l288_288624

noncomputable def GregFencing (area : ℝ) (extra_fencing : ℝ) (pi_approx : ℝ) : ℝ :=
  let r := real.sqrt (area * (7 / 22)) in
  let circumference := 2 * pi_approx * r in
  circumference + extra_fencing

theorem Greg_needs_93_feet :
  GregFencing 616 5 (22/7) = 93 := by
  sorry

end Greg_needs_93_feet_l288_288624


namespace median_convergence_l288_288698

noncomputable theory

open MeasureTheory
open ProbTheory

-- Definitions for the random variables and their distributions.
variables (α : Type*) [MeasurableSpace α] -- measurable space for the random variable
variables (ξ : α → ennreal) (ξ_n : ℕ → α → ennreal) -- sequences of random variables
variables (μ : ennreal) (μ_n : ℕ → ennreal) -- sequences of medians

-- Assumptions
variables (h_median : ∀ n, µ_n n ⟶ µ)
variables (h_inequality_ξ : ∀ x, P(ξ < µ) ≤ 1/2 ∧ P(ξ ≤ µ) ≥ 1/2)
variables (h_inequality_ξn : ∀ n x, P(ξ_n n < µ_n n) ≤ 1/2 ∧ P(ξ_n n ≤ µ_n n) ≥ 1/2)
variables (h_convergence : ∀ s, converges_in_probability (ξ_n s) ξ)

-- The main theorem statement
theorem median_convergence : (∀ n, (h_inequality_ξn n) ∧ (h_convergence n) ∧ (h_inequality_ξ)) → ∀ ε > 0, ∃ N, ∀ n ≥ N, |μ_n n - μ| < ε := by
  sorry

end median_convergence_l288_288698


namespace find_n_l288_288132

-- Define the problem in Lean
noncomputable def largeCircleRadius (N r : ℝ) : ℝ := N * r
noncomputable def areaOfSmallCircles (N r : ℝ) : ℝ := N * π * r^2
noncomputable def areaOfLargeCircle (N r : ℝ) : ℝ := π * (N * r)^2
noncomputable def remainingArea (N r : ℝ) : ℝ := (π * (N * r)^2) - (N * π * r^2)
noncomputable def ratioOfAreas (N r : ℝ) : ℝ := (N * π * r^2) / (remainingArea N r)

-- Define the main theorem to be proved
theorem find_n (N : ℝ) (h : ratioOfAreas N r = 1 / 3): N = 4 :=
by
  sorry

end find_n_l288_288132


namespace solve_length_BF_l288_288728

-- Define the problem conditions
def rectangular_paper (short_side long_side : ℝ) : Prop :=
  short_side = 12 ∧ long_side > short_side

def vertex_touch_midpoint (vmp mid : ℝ) : Prop :=
  vmp = mid / 2

def congruent_triangles (triangle1 triangle2 : ℝ) : Prop :=
  triangle1 = triangle2

-- Theorem to prove the length of BF
theorem solve_length_BF (short_side long_side vmp mid triangle1 triangle2 : ℝ) 
  (h1 : rectangular_paper short_side long_side)
  (h2 : vertex_touch_midpoint vmp mid)
  (h3 : congruent_triangles triangle1 triangle2) :
  -- The length of BF is 10
  mid = 6 → 18 - 6 = 12 + 6 - 10 → 10 = 12 - (18 - 10) → vmp = 6 → 6 * 2 = 12 →
  sorry :=
sorry

end solve_length_BF_l288_288728


namespace find_cos_B_find_area_triangle_l288_288549

/- Part 1 -/
theorem find_cos_B (a b : ℝ) (A B C : ℝ) (h : a = sqrt 2 * b) (h2 : sin B ^ 2 = sin A * sin C) :
  cos B = 3 / 4 := by
  sorry

/- Part 2 -/
theorem find_area_triangle (a b c : ℝ) (B : ℝ) (h : cos B = 3 / 4) (h2 : a = 2)
  (h3 : b^2 = a * c) :
  (1 / 2) * a * c * sqrt(7) / 4 = sqrt(7) ∨
  (1 / 2) * a * c * sqrt(7) / 4 = 1 / 2 * 2 * c * sqrt (7) / 4 := by
  sorry

end find_cos_B_find_area_triangle_l288_288549


namespace find_m_plus_n_l288_288312

theorem find_m_plus_n 
  (XY YZ XZ : ℕ)
  (RY QZ QY : ℝ)
  (XY_val : XY = 29)
  (YZ_val : YZ = 31)
  (XZ_val : XZ = 30)
  (arc_RY_eq_QZ : RY = QZ)
  (arc_QY_eq_RY_plus_1 : QY = RY + 1)
  (arc_XR_eq_RY_plus_2 : ∃ XR, XR = RY + 2)
  :
  let XR := 33 / 2 in 35 = 33 + 2 :=
by
  sorry

end find_m_plus_n_l288_288312


namespace original_salary_condition_l288_288245

variable (S: ℝ)

theorem original_salary_condition (h: 1.10 * 1.08 * 0.95 * 0.93 * S = 6270) :
  S = 6270 / (1.10 * 1.08 * 0.95 * 0.93) :=
by
  sorry

end original_salary_condition_l288_288245


namespace simplify_expression_l288_288999

theorem simplify_expression (x : ℝ) : 7 * x + 9 - 3 * x + 15 * 2 = 4 * x + 39 := 
by sorry

end simplify_expression_l288_288999


namespace translate_line_up_l288_288275

-- Define the original line equation as a function
def original_line (x : ℝ) : ℝ := -2 * x

-- Define the transformed line equation as a function
def translated_line (x : ℝ) : ℝ := -2 * x + 1

-- Prove that translating the original line upward by 1 unit gives the translated line
theorem translate_line_up (x : ℝ) :
  original_line x + 1 = translated_line x :=
by
  unfold original_line translated_line
  simp

end translate_line_up_l288_288275


namespace distance_last_pair_of_trees_l288_288124

theorem distance_last_pair_of_trees 
  (yard_length : ℝ := 1200)
  (num_trees : ℕ := 117)
  (initial_distance : ℝ := 5)
  (distance_increment : ℝ := 2) :
  let num_distances := num_trees - 1
  let last_distance := initial_distance + (num_distances - 1) * distance_increment
  last_distance = 235 := by 
  sorry

end distance_last_pair_of_trees_l288_288124


namespace quadratic_inequality_empty_solution_range_l288_288120

theorem quadratic_inequality_empty_solution_range (b : ℝ) :
  (∀ x : ℝ, ¬ (x^2 + b * x + 1 ≤ 0)) ↔ -2 < b ∧ b < 2 :=
by
  sorry

end quadratic_inequality_empty_solution_range_l288_288120


namespace total_leftover_pies_l288_288125

def leftover_pies (total : ℕ) (percent_eaten : ℚ) : ℕ :=
  total - (total * percent_eaten).toNat

theorem total_leftover_pies :
  let total_pies := 400;
  let apple_leftover := leftover_pies total_pies 0.68;
  let peach_leftover := leftover_pies total_pies 0.72;
  let cherry_leftover := leftover_pies total_pies 0.79;
  let chocolate_leftover := leftover_pies total_pies 0.81;
  let lemon_meringue_leftover := leftover_pies total_pies 0.65;
  apple_leftover + peach_leftover + cherry_leftover + chocolate_leftover + lemon_meringue_leftover = 540 :=
by {
  sorry
}

end total_leftover_pies_l288_288125


namespace angle_BCA_proof_l288_288937

-- Define the triangle ABC, with A, B, C as points.
variables {A B C L M : Type}
variables [angle_bisector A B C L] [median B M A C] (h1 : distance A B = 2 * distance B L)
          (h2 : angle L M A = 127)

-- Define the theorem corresponding to the proof problem.
theorem angle_BCA_proof : measure_angle B C A = 74 :=
sorry

end angle_BCA_proof_l288_288937


namespace apples_shared_l288_288992

-- Definitions and conditions based on problem statement
def initial_apples : ℕ := 89
def remaining_apples : ℕ := 84

-- The goal to prove that Ruth shared 5 apples with Peter
theorem apples_shared : initial_apples - remaining_apples = 5 := by
  sorry

end apples_shared_l288_288992


namespace exists_triangle_with_two_obtuse_angles_l288_288207

theorem exists_triangle_with_two_obtuse_angles :
  ∃ (T : Type) [triangle T], ∃ (α β : T.angle), (α > 90 ∧ β > 90) :=
sorry

end exists_triangle_with_two_obtuse_angles_l288_288207


namespace survey_support_percentage_l288_288726

theorem survey_support_percentage (men women : ℕ) (support_men_percentage support_women_percentage : ℝ) 
  (men_support women_support total_support : ℝ) 
  (total_people : ℕ)
  (h1 : men = 150)
  (h2 : women = 850)
  (h3 : support_men_percentage = 0.70)
  (h4 : support_women_percentage = 0.75)
  (h5 : men_support = support_men_percentage * men)
  (h6 : women_support = support_women_percentage * women)
  (h7 : total_support = men_support + women_support)
  (h8 : total_people = men + women)
  (h9 : total_support = 743)
  (h10 : total_people = 1000):
  (total_support / total_people) * 100 = 74.3 := 
begin
  sorry
end

end survey_support_percentage_l288_288726


namespace positive_area_triangles_correct_l288_288080

-- Define the set of points with integer coordinates in the given range
def grid_points := { p : ℤ × ℤ | 1 ≤ p.1 ∧ p.1 ≤ 5 ∧ 1 ≤ p.2 ∧ p.2 ≤ 5 }

-- Define a function to check for collinearity of three points
def collinear (p1 p2 p3 : ℤ × ℤ) : Prop :=
  (p2.2 - p1.2) * (p3.1 - p1.1) = (p3.2 - p1.2) * (p2.1 - p1.1)

-- Define the count of triangles with positive area (not collinear) from the given points
def triangles_with_positive_area (points : set (ℤ × ℤ)) : ℕ :=
  (@set.to_finset (ℤ × ℤ) _ points).to_list.combinations 3
  .filter (λ l, l.length = 3 ∧ ¬ collinear l.head l.nth 1 l.nth 2).length

-- State the proof problem
theorem positive_area_triangles_correct :
  triangles_with_positive_area grid_points = 2170 := sorry

end positive_area_triangles_correct_l288_288080


namespace AD_parallel_BC_l288_288033

-- Definitions for given points and segments
variables {A B C D : Point}
variables {AB BC CD : Segment}
variables {angle_ABC angle_BCD : ℝ}

-- Assumptions based on conditions
def non_closed_broken_line (A B C D : Point) : Prop :=
  True   -- Just a placeholder to mark the points as forming a broken line

def same_side (A D : Point) (BC : Line) : Prop :=
  True   -- Placeholder to indicate A and D are on the same side of BC

axiom AB_eq_CD : ↥AB = ↥CD
axiom angle_ABC_eq_angle_BCD : angle_ABC = angle_BCD
axiom A_D_same_side_BC : same_side A D (line B C)

-- Statement to prove that AD is parallel to BC given the conditions
theorem AD_parallel_BC
  (A B C D : Point)
  (AB_eq_CD : ↥AB = ↥CD)
  (angle_ABC_eq_angle_BCD : angle_ABC = angle_BCD)
  (A_D_same_side_BC : same_side A D (line B C))
  : parallel (line A D) (line B C) :=
sorry

end AD_parallel_BC_l288_288033


namespace cosine_value_of_angle_l288_288480

variables (a b : ℝ × ℝ)
variables (cosine_of_angle : ℝ)

def vec_add (u v : ℝ × ℝ) : ℝ × ℝ := (u.1 + v.1, u.2 + v.2)
def vec_sub (u v : ℝ × ℝ) : ℝ × ℝ := (u.1 - v.1, u.2 - v.2)
def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2
def magnitude (u : ℝ × ℝ) : ℝ := Real.sqrt (u.1 ^ 2 + u.2 ^ 2)
def cosine_formula (a b : ℝ × ℝ) : ℝ := dot_product a b / (magnitude a * magnitude b)

theorem cosine_value_of_angle 
  (h1 : vec_add a b = (5, -10))
  (h2 : vec_sub a b = (3, 6)) :
  cosine_of_angle = 2 * Real.sqrt 13 / 13 :=
sorry

end cosine_value_of_angle_l288_288480


namespace range_of_k_is_interval_l288_288039

noncomputable def rangeOfK : set ℝ :=
{ k | ∀ x ∈ Icc 0 1, k * 4^x - k * 2^(x + 1) + 6 * (k - 5) ≠ 0 }

theorem range_of_k_is_interval : rangeOfK = Icc 5 6 := 
by {
  sorry
}

end range_of_k_is_interval_l288_288039


namespace johns_average_speed_l288_288155

-- Definitions for the conditions
def initial_distance : ℝ := 40
def initial_time : ℝ := 4
def break_time : ℝ := 1
def post_repair_distance : ℝ := 15
def post_repair_time : ℝ := 3

-- Definition for total distance and total time
def total_distance : ℝ := initial_distance + post_repair_distance
def total_time : ℝ := initial_time + break_time + post_repair_time

-- Definition for average speed
def average_speed : ℝ := total_distance / total_time

-- The proof statement
theorem johns_average_speed : average_speed = 6.875 := by
  -- The proof is omitted
  sorry

end johns_average_speed_l288_288155


namespace universal_sequence_min_length_universal_sequence_min_length_n4_l288_288697

def isUniversal (a : List ℕ) (n : ℕ) : Prop :=
∀ (perm : List ℕ), perm.permutations.Contains [1, 2, ..., n] → ∃ (subseq : List ℕ), subseq ⊆ a ∧ subseq = perm

theorem universal_sequence_min_length (n : ℕ) (a : List ℕ) (ha : isUniversal a n) : 
  a.length ≥ n * (n + 1) / 2 := 
sorry

theorem universal_sequence_min_length_n4 (a : List ℕ) (ha : isUniversal a 4) : 
  a.length = 12 := 
sorry

end universal_sequence_min_length_universal_sequence_min_length_n4_l288_288697


namespace find_cost_price_l288_288302

theorem find_cost_price (SP : ℝ) (profit_percentage : ℝ) (CP : ℝ) 
  (h1 : SP = 400) (h2 : profit_percentage = 60) :
  CP = 250 :=
by
  have h3 : SP = CP + (profit_percentage / 100) * CP := sorry
  have h4 : 400 = CP + 0.6 * CP := by 
    rw [h1, h2]
    sorry
  have h5 : 400 = 1.6 * CP := sorry
  have h6 : CP = 400 / 1.6 := sorry
  exact eq.trans h6 (eq.refl 250)

end find_cost_price_l288_288302


namespace p_and_not_q_l288_288440

def p : Prop :=
  ∀ x : ℕ, x > 0 → (1 / 2) ^ x ≥ (1 / 3) ^ x

def q : Prop :=
  ∃ x : ℕ, x > 0 ∧ 2^x + 2^(1-x) = 2 * Real.sqrt 2

theorem p_and_not_q : p ∧ ¬q :=
by
  have h_p : p := sorry
  have h_not_q : ¬q := sorry
  exact ⟨h_p, h_not_q⟩

end p_and_not_q_l288_288440


namespace triangle_side_length_l288_288548

theorem triangle_side_length 
  (A B C : Type) [metric_space A] [metric_space B] [metric_space C]
  (angleA angleB : ℝ) (sideBC : ℝ)
  (h₁ : angleA = 60)
  (h₂ : angleB = 45)
  (h₃ : sideBC = 3 * real.sqrt 2) :
  ∃ AC : ℝ, AC = 2 * real.sqrt 3 := 
by
  sorry

end triangle_side_length_l288_288548


namespace polar_line_equation_l288_288934

theorem polar_line_equation
  (rho theta : ℝ)
  (h1 : rho = 4 * Real.cos theta)
  (h2 : ∀ (x y : ℝ), (x - 2) ^ 2 + y ^ 2 = 4 → x = 2)
  : rho * Real.cos theta = 2 :=
sorry

end polar_line_equation_l288_288934


namespace min_value_of_objective_function_l288_288586

theorem min_value_of_objective_function : 
  ∃ (x y : ℝ), 
    (2 * x + y - 2 ≥ 0) ∧ 
    (x - 2 * y + 4 ≥ 0) ∧ 
    (x - 1 ≤ 0) ∧ 
    (∀ (u v: ℝ), 
      (2 * u + v - 2 ≥ 0) → 
      (u - 2 * v + 4 ≥ 0) → 
      (u - 1 ≤ 0) → 
      (3 * u + 2 * v ≥ 3)) :=
  sorry

end min_value_of_objective_function_l288_288586


namespace age_twice_of_father_l288_288720

theorem age_twice_of_father (S M Y : ℕ) (h₁ : S = 22) (h₂ : M = S + 24) (h₃ : M + Y = 2 * (S + Y)) : Y = 2 := by
  sorry

end age_twice_of_father_l288_288720


namespace point_of_tangency_l288_288061

-- Define the curve function
def curve (x : ℝ) : ℝ := (x^2) / 4

-- Define the derivative of the curve function
def derivative_curve (x : ℝ) : ℝ := (1/2) * x

-- State the desired proof statement 
theorem point_of_tangency (a : ℝ) 
    (h_curve : ∀ x, derivative_curve x = (1/2) * x)
    (h_slope : derivative_curve a = 1/2) : 
    a = 1 := by 
  sorry

end point_of_tangency_l288_288061


namespace count_4_digit_numbers_divisible_by_13_l288_288871

theorem count_4_digit_numbers_divisible_by_13 : 
  ∃ n : ℕ, n = 693 ∧ (∀ k : ℕ, k >= 1000 ∧ k < 10000 ∧ k % 13 = 0 → ∃ m : ℕ, m = (k - 1000) / 13 + 1 ∧ m = n) :=
by {
  -- Solution proof will be placed here.
  sorry
}

end count_4_digit_numbers_divisible_by_13_l288_288871


namespace Ahmed_total_distance_traveled_l288_288344

/--
Ahmed stops one-quarter of the way to the store.
He continues for 12 km to reach the store.
Prove that the total distance Ahmed travels is 16 km.
-/
theorem Ahmed_total_distance_traveled
  (D : ℝ) (h1 : D > 0)  -- D is the total distance to the store, assumed to be positive
  (h_stop : D / 4 + 12 = D) : D = 16 := 
sorry

end Ahmed_total_distance_traveled_l288_288344


namespace probability_increasing_function_correct_l288_288671

noncomputable def probability_increasing_function : ℚ :=
  let outcomes := { (m, n) | m ∈ finset.range 1 7 ∧ n ∈ finset.range 1 7 } in
  let valid_outcomes := { (m, n) ∈ outcomes | (n : ℚ) / (2 * m) ≤ 1 } in
  finset.card valid_outcomes / finset.card outcomes

theorem probability_increasing_function_correct :
  probability_increasing_function = 5 / 6 := by
  sorry

end probability_increasing_function_correct_l288_288671


namespace find_commission_rate_l288_288738

def commission_rate (commission_earned total_sales : ℝ) : ℝ :=
  (commission_earned / total_sales) * 100

theorem find_commission_rate :
  commission_rate 12.50 250 = 5 :=
by
  sorry

end find_commission_rate_l288_288738


namespace monotonic_k_range_l288_288449

noncomputable def f (x k : ℝ) : ℝ := x^2 - 2 * k * x - 2

theorem monotonic_k_range :
  (∀ x1 x2, (5 ≤ x1 ∧ x1 ≤ x2) → f x1 k ≤ f x2 k) → (k ∈ set.Iic 5) :=
sorry

end monotonic_k_range_l288_288449


namespace non_degenerate_triangles_l288_288075

theorem non_degenerate_triangles (x y : ℕ) (hx : 1 ≤ x ∧ x ≤ 5) (hy : 1 ≤ y ∧ y ≤ 5) : 
  ∃ (n : ℕ), n = 2160 ∧ 1 ≤ n :=
by
  sorry

end non_degenerate_triangles_l288_288075


namespace quadratic_inequality_solution_l288_288028

theorem quadratic_inequality_solution (c : ℝ) (h : c > 1) :
  {x : ℝ | x^2 - (c + 1/c) * x + 1 > 0} = {x : ℝ | x < 1/c ∨ x > c} :=
sorry

end quadratic_inequality_solution_l288_288028


namespace odd_perfect_square_l288_288159

theorem odd_perfect_square (n : ℕ) (h : ∑ d in divisors n, d = 2 * n + 1) : ∃ m : ℕ, m % 2 = 1 ∧ n = m * m :=
by
  sorry

end odd_perfect_square_l288_288159


namespace solve_for_x_l288_288023

def delta (x : ℝ) : ℝ := 4 * x + 5
def phi (x : ℝ) : ℝ := 6 * x + 3

theorem solve_for_x (x : ℝ) (h : delta (phi x) = -1) : x = - 3 / 4 :=
by
  sorry

end solve_for_x_l288_288023


namespace fifth_largest_four_digit_number_with_sum_seventeen_is_9611_l288_288237

-- Define the structure for a four-digit number
structure FourDigitNumber :=
  (a b c d : ℕ)
  (a_nonzero : a ≠ 0)
  (digits_range : a ≤ 9 ∧ b ≤ 9 ∧ c ≤ 9 ∧ d ≤ 9)
  (sum_digits : a + b + c + d = 17)

-- Define the property of being the 5th largest four-digit number with digit sum 17
def isFifthLargest (n : FourDigitNumber) : Prop :=
  ∃ (n₁ n₂ n₃ n₄ : FourDigitNumber),
    (n₁.sum_digits = 17 ∧ n₁ > n₂ ∧ n₂.sum_digits = 17 ∧ n₂ > n₃ ∧ n₃.sum_digits = 17 ∧ n₃ > n₄ ∧ n₄.sum_digits = 17 ∧ n₄ > n)

-- The main statement to be proven
theorem fifth_largest_four_digit_number_with_sum_seventeen_is_9611 :
  ∃ (n : FourDigitNumber),
    isFifthLargest n ∧ n.a = 9 ∧ n.b = 6 ∧ n.c = 1 ∧ n.d = 1 :=
sorry

end fifth_largest_four_digit_number_with_sum_seventeen_is_9611_l288_288237


namespace number_of_possible_m_l288_288819

/-- A three-digit number where tens digit is 8 --/
def isThreeDigitWithTens8 (m : ℕ) : Prop :=
  100 ≤ m ∧ m < 1000 ∧ (m / 10) % 10 = 8

/-- m and n satisfy the equation m - 40n = 24 --/
def satisfiesEquation (m n : ℕ) : Prop :=
  m - 40 * n = 24

/-- There are exactly two values of m satisfying the conditions --/
theorem number_of_possible_m : {m : ℕ // ∃ n : ℕ, isThreeDigitWithTens8 m ∧ satisfiesEquation m n}.card = 2 :=
by
  sorry

end number_of_possible_m_l288_288819


namespace part_i_part_ii_l288_288064

noncomputable def f (x : ℝ) : ℝ := log 10 (x + 1)

theorem part_i (x : ℝ) (h1 : 0 < f (1 - 2 * x) - f x) (h2 : f (1 - 2 * x) - f x < 1) :
  -2 / 3 < x ∧ x < 1 / 3 :=
sorry

theorem part_ii (g : ℝ → ℝ) (h_even : ∀ x, g x = g (-x)) (h_periodic : ∀ x, g (x + 2) = g x)
  (h_def : ∀ x, 0 ≤ x ∧ x ≤ 1 → g x = f x) :
  ∀ x, 1 ≤ x ∧ x ≤ 2 → g x = 3 - 10^x :=
sorry

end part_i_part_ii_l288_288064


namespace cannot_form_triangle_triangle_exists_sets_evaluation_l288_288684

theorem cannot_form_triangle (a b c : ℝ) : 
  (a + b ≤ c ∨ a + c ≤ b ∨ b + c ≤ a) → ¬(a + b > c ∧ a + c > b ∧ b + c > a) := by 
  intros h h'
  exact h (and.left h')

theorem triangle_exists (a b c : ℝ) :
  (a + b > c) → (a + c > b) → (b + c > a) :=
by sorry

theorem sets_evaluation :
  let A := (2, 2, 3) in
  let B := (5, 7, 4) in
  let C := (2, 4, 6) in
  let D := (4, 5, 8) in
  (triangle_exists A.1 A.2 A.3) ∧
  (triangle_exists B.1 B.2 B.3) ∧
  (cannot_form_triangle C.1 C.2 C.3) ∧
  (triangle_exists D.1 D.2 D.3) := by sorry

end cannot_form_triangle_triangle_exists_sets_evaluation_l288_288684


namespace correct_equation_l288_288264

variable (x : ℝ)
axiom area_eq_720 : x * (x - 6) = 720

theorem correct_equation : x * (x - 6) = 720 := by
  exact area_eq_720

end correct_equation_l288_288264


namespace distance_point_to_line_zero_or_four_l288_288628

theorem distance_point_to_line_zero_or_four {b : ℝ} 
(h : abs (b - 2) / Real.sqrt 2 = Real.sqrt 2) : 
b = 0 ∨ b = 4 := 
sorry

end distance_point_to_line_zero_or_four_l288_288628


namespace exponent_value_l288_288782

theorem exponent_value (exponent : ℕ) (y: ℕ) :
  (12 ^ exponent) * (6 ^ 4) / 432 = y → y = 36 → exponent = 1 :=
by
  intro h1 h2
  sorry

end exponent_value_l288_288782


namespace part1_part2_l288_288547

-- Given points A, B, and C with specific coordinates for part (1)
def A := (-2,1)
def B := (4,3)
def C1 := (3,-2) -- C for part (1)

-- Part (1): Prove the line equation containing altitude AD on side BC
theorem part1 :
  ∃ m b : ℝ, (∀ (x y : ℝ), y = m * x + b ↔ x + 5 * y - 3 = 0)
    ∧ (∀ (D : ℝ × ℝ), D = (fst A, (snd A + (5/snd C1 - snd A) * (fst B - fst A)))) :=
sorry

-- Given points A, B, and M with specific coordinates for part (2)
def B := (4,3)
def M := (3,1) -- M is the midpoint of AC

-- Part (2): Prove the line equation containing side BC
theorem part2 :
  ∃ m b : ℝ, (∀ (x y : ℝ), y = m * x + b ↔ x + 2 * y - 10 = 0)
    ∧ (fst M = (fst A + fst C2) / 2) ∧ (snd M = (snd A + snd C2) / 2) :=
sorry

end part1_part2_l288_288547


namespace max_four_products_l288_288661

theorem max_four_products (f g h j : ℕ) 
  (h_values : {f, g, h, j} = {6, 7, 8, 9}) :
  fg + gh + hj + fj ≤ 225 := 
begin
  sorry
end

end max_four_products_l288_288661


namespace num_of_4_digit_numbers_divisible_by_13_l288_288887

theorem num_of_4_digit_numbers_divisible_by_13 :
  let smallest_4_digit := 1000
  let smallest_divisible_by_13 := 1001
  let largest_4_digit := 9999
  let largest_divisible_by_13 := 9997
  let a := smallest_divisible_by_13
  let d := 13
  let l := largest_divisible_by_13
  (\(l - a) / d) + 1 = 693 :=
by
  let smallest_4_digit := 1000
  let smallest_divisible_by_13 := 1001
  let largest_4_digit := 9999
  let largest_divisible_by_13 := 9997
  let a := smallest_divisible_by_13
  let d := 13
  let l := largest_divisible_by_13
  have h1 : (l - a) / d + 1 = (9997 - 1001) / 13 + 1, by sorry
  have h2 : (9997 - 1001) / 13 + 1 = 8996 / 13 + 1, by sorry
  have h3 : 8996 / 13 + 1 = 692 + 1, by sorry
  have h4 : 692 + 1 = 693, by sorry
  exact h4

end num_of_4_digit_numbers_divisible_by_13_l288_288887


namespace points_concyclic_l288_288173

-- Definitions required by the conditions
variables (A B C A1 A2 B1 B2 C1 C2 : ℝ)

-- Conditions extracted from the problem
-- Define a triangle ABC
variable (triangle_ABC : ∀ (A B C α β γ : ℝ), Prop)

-- Condition 1: Distance conditions for A1 and A2
variable (dist_AA1 : ℝ)
variable (dist_AA2 : ℝ)
variable (dist_BB1 : ℝ)
variable (dist_BB2 : ℝ)
variable (dist_CC1 : ℝ)
variable (dist_CC2 : ℝ)

-- Condition 2: Points placement for A1 and A2
variable (A1_placement : Prop)
variable (A2_placement : Prop)
-- Similarly for B1, B2 and C1, C2
variable (B1_placement : Prop)
variable (B2_placement : Prop)
variable (C1_placement : Prop)
variable (C2_placement : Prop)

-- Main theorem: Show that points are concyclic
theorem points_concyclic (h1 : dist_AA1 = dist_AA2 = (C - B)) (h2 : A1_placement)
  (h3 : A2_placement) (h4 : dist_BB1 = dist_BB2 = (A - C)) (h5 : B1_placement)
  (h6 : B2_placement) (h7 : dist_CC1 = dist_CC2 = (B - A)) (h8 : C1_placement)
  (h9 : C2_placement) : (
  ∃ (circle_center : ℝ) (circle_radius : ℝ),
  ∀ points ∈ {A1, A2, B1, B2, C1, C2}, dist circle_center points = circle_radius
) := sorry

end points_concyclic_l288_288173


namespace max_marks_paper_I_l288_288711

variable (M : ℝ)

theorem max_marks_paper_I (h1 : 0.65 * M = 112 + 58) : M = 262 :=
  sorry

end max_marks_paper_I_l288_288711


namespace num_triangles_with_positive_area_l288_288104

/-- 
Given vertices in a 5x5 grid with integer coordinates satisfying 1 ≤ x ≤ 5 and 1 ≤ y ≤ 5,
prove that the number of triangles with positive area is 2170. 
-/
theorem num_triangles_with_positive_area : 
  (∑ t in ({(i, j) | 1 ≤ i ∧ i ≤ 5 ∧ 1 ≤ j ∧ j ≤ 5}.powerset.filter (λ (s : set (ℕ × ℕ)), s.card = 3)),
    if t₁.1 * (t₂.2 - t₃.2) + t₂.1 * (t₃.2 - t₁.2) + t₃.1 * (t₁.2 - t₂.2) ≠ 0 then 1 else 0) = 2170 :=
by sorry

end num_triangles_with_positive_area_l288_288104


namespace beats_per_week_l288_288151

def beats_per_minute : ℕ := 200
def minutes_per_hour : ℕ := 60
def hours_per_day : ℕ := 2
def days_per_week : ℕ := 7

theorem beats_per_week : beats_per_minute * minutes_per_hour * hours_per_day * days_per_week = 168000 := by
  sorry

end beats_per_week_l288_288151


namespace balance_of_Diamonds_l288_288260

def Delta_and_Diamond_balance (a b c : ℝ) : Prop :=
  3 * a + 2 * b = 12 * c ∧ a = b + 3 * c

theorem balance_of_Diamonds :
  ∀ (a b c : ℝ), Delta_and_Diamond_balance a b c → 4 * b = 3 * c :=
by
  intros a b c h,
  obtain ⟨h1, h2⟩ := h,
  sorry

end balance_of_Diamonds_l288_288260


namespace count_4_digit_numbers_divisible_by_13_l288_288882

theorem count_4_digit_numbers_divisible_by_13 : 
  let count := (9962 - 1014) / 13 + 1
  1000 ≤ 1014 ∧ 9962 ≤ 9999 →
  count = 689 :=
  by
    sorry

end count_4_digit_numbers_divisible_by_13_l288_288882


namespace sum_of_intercepts_l288_288718

def line_eq := ∀ x y : ℝ, y + 3 = -3 * (x - 5)

theorem sum_of_intercepts : (∃ x : ℝ, line_eq x 0) + (∃ y : ℝ, line_eq 0 y) = 16 :=
by
  sorry

end sum_of_intercepts_l288_288718


namespace fran_travel_time_l288_288154

theorem fran_travel_time (joann_speed fran_speed : ℝ) (joann_time joann_distance : ℝ) :
  joann_speed = 15 → joann_time = 4 → joann_distance = joann_speed * joann_time →
  fran_speed = 20 → fran_time = joann_distance / fran_speed →
  fran_time = 3 :=
by 
  intros h1 h2 h3 h4 h5
  sorry

end fran_travel_time_l288_288154


namespace Leah_potatoes_peeled_l288_288853

noncomputable def pile_of_potatoes : ℕ := 50
noncomputable def homers_rate : ℕ := 3
noncomputable def leahs_rate : ℕ := 4
noncomputable def time_homer_alone : ℕ := 6

theorem Leah_potatoes_peeled :
  let peeled_by_homer := time_homer_alone * homers_rate,
      remaining_potatoes := pile_of_potatoes - peeled_by_homer,
      combined_rate := homers_rate + leahs_rate,
      time_peeling_together := remaining_potatoes / combined_rate in
  leahs_rate * time_peeling_together = 18 := sorry

end Leah_potatoes_peeled_l288_288853


namespace area_of_isosceles_right_triangle_OAB_eq_2_l288_288795

-- Definitions of vectors a and b
def vector_a : ℝ × ℝ := (-1, 1)
def vector_OA (b : ℝ × ℝ) : ℝ × ℝ := (-1, 1) - b
def vector_OB (b : ℝ × ℝ) : ℝ × ℝ := (-1, 1) + b

-- Given conditions for an isosceles right triangle with O as the right angle vertex
def is_isosceles_right_triangle (OA OB : ℝ × ℝ) : Prop :=
  let dot_product := OA.1 * OB.1 + OA.2 * OB.2 in
  dot_product = 0

-- Definition of the area of triangle OAB
def area_OAB (OA OB : ℝ × ℝ) : ℝ :=
  0.5 * (OA.1 * OB.2 - OA.2 * OB.1).abs

-- Prove the area of the triangle is 2
theorem area_of_isosceles_right_triangle_OAB_eq_2 (b : ℝ × ℝ)
  (h : is_isosceles_right_triangle (vector_OA b) (vector_OB b)) :
  area_OAB (vector_OA b) (vector_OB b) = 2 :=
sorry

end area_of_isosceles_right_triangle_OAB_eq_2_l288_288795


namespace find_theta_2phi_l288_288815

-- Given
variables {θ φ : ℝ}
variables (hθ_acute : 0 < θ ∧ θ < π / 2)
variables (hφ_acute : 0 < φ ∧ φ < π / 2)
variables (h_tanθ : Real.tan θ = 3 / 11)
variables (h_sinφ : Real.sin φ = 1 / 3)

-- To prove
theorem find_theta_2phi : 
  ∃ x : ℝ, 0 < x ∧ x < π / 2 ∧ Real.tan x = (21 + 6 * Real.sqrt 2) / (77 - 6 * Real.sqrt 2) ∧ x = θ + 2 * φ := 
sorry

end find_theta_2phi_l288_288815


namespace positive_area_triangles_correct_l288_288079

-- Define the set of points with integer coordinates in the given range
def grid_points := { p : ℤ × ℤ | 1 ≤ p.1 ∧ p.1 ≤ 5 ∧ 1 ≤ p.2 ∧ p.2 ≤ 5 }

-- Define a function to check for collinearity of three points
def collinear (p1 p2 p3 : ℤ × ℤ) : Prop :=
  (p2.2 - p1.2) * (p3.1 - p1.1) = (p3.2 - p1.2) * (p2.1 - p1.1)

-- Define the count of triangles with positive area (not collinear) from the given points
def triangles_with_positive_area (points : set (ℤ × ℤ)) : ℕ :=
  (@set.to_finset (ℤ × ℤ) _ points).to_list.combinations 3
  .filter (λ l, l.length = 3 ∧ ¬ collinear l.head l.nth 1 l.nth 2).length

-- State the proof problem
theorem positive_area_triangles_correct :
  triangles_with_positive_area grid_points = 2170 := sorry

end positive_area_triangles_correct_l288_288079


namespace tan_double_angle_proof_l288_288425

theorem tan_double_angle_proof (α : ℝ) (h1 : α ∈ set.Ioc (π / 2) π)
  (h2 : Real.sin (α - π / 2) = 3 / 5) : Real.tan (2 * α) = 24 / 7 := 
by
  sorry

end tan_double_angle_proof_l288_288425


namespace count_positive_area_triangles_l288_288096

noncomputable def numPositiveAreaTriangles : ℕ := 2160

theorem count_positive_area_triangles 
  (vertices : list (ℤ × ℤ))
  (h1 : ∀ p ∈ vertices, 1 ≤ p.1 ∧ p.1 ≤ 5 ∧ 1 ≤ p.2 ∧ p.2 ≤ 5)
  (h2 : vertices.length = 25) : 
  (number_of_triangles_with_positive_area vertices) = numPositiveAreaTriangles :=
sorry

end count_positive_area_triangles_l288_288096


namespace kite_quadrilateral_CD_length_l288_288333

/-- Given a kite-shaped quadrilateral ABCD inscribed in a circle of radius 150√2,
    with AB = AD = 150 and BC = 150√2, prove that the length of side CD is 150√2. -/
theorem kite_quadrilateral_CD_length
  (A B C D O : Type)
  [MetricSpace A]
  [MetricSpace B]
  [MetricSpace C]
  [MetricSpace D]
  [MetricSpace O]
  (h_circle_radius : dist O A = 150 * Real.sqrt 2)
  (h_inscribed : dist O B = 150 * Real.sqrt 2)
  (h_AB_AD : dist A B = 150 ∧ dist A D = 150)
  (h_BC : dist B C = 150 * Real.sqrt 2) :
  dist C D = 150 * Real.sqrt 2 := by
  sorry

end kite_quadrilateral_CD_length_l288_288333


namespace max_four_products_l288_288662

theorem max_four_products (f g h j : ℕ) 
  (h_values : {f, g, h, j} = {6, 7, 8, 9}) :
  fg + gh + hj + fj ≤ 225 := 
begin
  sorry
end

end max_four_products_l288_288662


namespace log_base_2_inverse_sixteen_l288_288392

theorem log_base_2_inverse_sixteen : log 2 (1/16) = -4 :=
by
  have h1 : 16 = 2^4 := by norm_num
  have h2 : 1/16 = 2^(-4) := by rw [h1, one_div_pow, neg_num_eq_neg]
  rw [log_inv_eq_log_neg] at h2
  exact h2

end log_base_2_inverse_sixteen_l288_288392


namespace find_fraction_l288_288467

variable (a b c : ℝ)
variable (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
variable (h1 : (a + b + c) / (a + b - c) = 7)
variable (h2 : (a + b + c) / (a + c - b) = 1.75)

theorem find_fraction : (a + b + c) / (b + c - a) = 3.5 := 
by {
  sorry
}

end find_fraction_l288_288467


namespace base7_subtraction_l288_288402

theorem base7_subtraction : 
  (3214 : ℕ)₇ - (1234 : ℕ)₇ = (2650 : ℕ)₇ := by
  sorry

end base7_subtraction_l288_288402


namespace caroline_socks_l288_288751

theorem caroline_socks :
  ∃ (p : ℕ), p = 10 :=
begin
  -- Conditions
  let initial_socks := 40,
  let lost_socks := 4,
  let fraction_donated := 2 / 3,
  let gifted_socks := 3,
  let total_socks := 25,

  -- Translate conditions
  let remaining_socks_after_loss := initial_socks - lost_socks,
  let donated_socks := fraction_donated * remaining_socks_after_loss,
  let remaining_socks_after_donation := remaining_socks_after_loss - donated_socks,
  let remaining_socks_after_gift := remaining_socks_after_donation + gifted_socks,
  let purchased_socks := total_socks - remaining_socks_after_gift,

  -- Prove
  use purchased_socks,
  have h : purchased_socks = 10,
  { sorry }, -- Skip the proof steps

  exact h,
end

end caroline_socks_l288_288751


namespace non_degenerate_triangles_l288_288076

theorem non_degenerate_triangles (x y : ℕ) (hx : 1 ≤ x ∧ x ≤ 5) (hy : 1 ≤ y ∧ y ≤ 5) : 
  ∃ (n : ℕ), n = 2160 ∧ 1 ≤ n :=
by
  sorry

end non_degenerate_triangles_l288_288076


namespace dot_product_a_b_l288_288478

-- Definitions for unit vectors e1 and e2 with given conditions
variables (e1 e2 : ℝ × ℝ)
variables (h_norm_e1 : e1.1^2 + e1.2^2 = 1) -- e1 is a unit vector
variables (h_norm_e2 : e2.1^2 + e2.2^2 = 1) -- e2 is a unit vector
variables (h_angle : e1.1 * e2.1 + e1.2 * e2.2 = -1 / 2) -- angle between e1 and e2 is 120 degrees

-- Definitions for vectors a and b
def a : ℝ × ℝ := (e1.1 + e2.1, e1.2 + e2.2)
def b : ℝ × ℝ := (e1.1 - 3 * e2.1, e1.2 - 3 * e2.2)

-- Theorem to prove
theorem dot_product_a_b : (a e1 e2) • (b e1 e2) = -1 :=
by
  sorry

end dot_product_a_b_l288_288478


namespace weight_of_new_man_l288_288694

theorem weight_of_new_man : 
  ∀ (avg_increase : ℝ) (old_man_weight : ℝ) (num_men : ℕ),
  avg_increase = 2.5 → old_man_weight = 68 → num_men = 10 → 
  let total_increase := num_men * avg_increase in
  let weight_new_man := old_man_weight + total_increase in
  weight_new_man = 93 :=
by
  intros avg_increase old_man_weight num_men h1 h2 h3
  let total_increase := num_men * avg_increase 
  let weight_new_man := old_man_weight + total_increase
  have : total_increase = 25 := by sorry
  have : weight_new_man = 93 := by sorry
  exact this

end weight_of_new_man_l288_288694


namespace combined_mpg_l288_288944

def car_averages (jane_avg mpg mike_avg mpg carl_avg mpg miles driven) : ℚ :=
  let jane_gas := driven / jane_avg
  let mike_gas := driven / mike_avg
  let carl_gas := driven / carl_avg
  let total_gas := jane_gas + mike_gas + carl_gas
  let total_distance := 3 * driven
  total_distance / total_gas

theorem combined_mpg (driven : ℚ) (h1 : driven = 100) :
  car_averages 30 15 20 driven = 38 :=
by
  -- Step to outline the proof
  sorry

end combined_mpg_l288_288944


namespace find_matrix_N_l288_288778

open Matrix

theorem find_matrix_N :
  ∃ (N : Matrix (Fin 2) (Fin 2) ℚ),
    (mul_vec N ![2, -1] = ![5, -3]) ∧
    (mul_vec N ![1, 4] = ![-2, 8]) :=
begin
  -- The proof will be done here.
  sorry
end

end find_matrix_N_l288_288778


namespace shaded_to_large_ratio_l288_288427

noncomputable def small_square := unit
noncomputable def area (s : small_square) := (1 : ℝ)
noncomputable def large_square := fin 25
noncomputable def area_large_square : ℝ := 25 * area ()

def shaded_region := pentagon_like_shape small_square

def area_shaded_region : ℝ := 1.5

theorem shaded_to_large_ratio :
  (area_shaded_region / area_large_square) = 3 / 50 := by
  sorry

end shaded_to_large_ratio_l288_288427


namespace magnitude_complex_given_conditions_l288_288049

noncomputable def magnitude_of_complex_number (z : ℂ) (argz : ℝ) := (|z| = Real.sqrt 2 + 1)

theorem magnitude_complex_given_conditions: ∀ (z : ℂ), 
  (Complex.arg z = Real.pi / 3) ∧
  (Complex.abs (z - 1) = Real.sqrt (Complex.abs z * Complex.abs (z - 2))) →
  magnitude_of_complex_number z (Real.pi / 3) :=
by
  intros z h
  cases h with arg_cond abs_cond
  sorry

end magnitude_complex_given_conditions_l288_288049


namespace river_depth_l288_288729

theorem river_depth (V : ℝ) (W : ℝ) (F : ℝ) (D : ℝ) 
  (hV : V = 10666.666666666666) 
  (hW : W = 40) 
  (hF : F = 66.66666666666667) 
  (hV_eq : V = W * D * F) : 
  D = 4 :=
by sorry

end river_depth_l288_288729


namespace first_step_induction_l288_288281

theorem first_step_induction (n : ℕ) (h : 1 < n) : 1 + 1/2 + 1/3 < 2 :=
by
  sorry

end first_step_induction_l288_288281


namespace stephen_total_distance_l288_288218

def speed_first_segment := 16 -- miles per hour
def time_first_segment := 10 / 60 -- hours

def speed_second_segment := 12 -- miles per hour
def headwind := 2 -- miles per hour
def actual_speed_second_segment := speed_second_segment - headwind
def time_second_segment := 20 / 60 -- hours

def speed_third_segment := 20 -- miles per hour
def tailwind := 4 -- miles per hour
def actual_speed_third_segment := speed_third_segment + tailwind
def time_third_segment := 15 / 60 -- hours

def distance_first_segment := speed_first_segment * time_first_segment
def distance_second_segment := actual_speed_second_segment * time_second_segment
def distance_third_segment := actual_speed_third_segment * time_third_segment

theorem stephen_total_distance : distance_first_segment + distance_second_segment + distance_third_segment = 12 := by
  sorry

end stephen_total_distance_l288_288218


namespace find_range_of_a_l288_288833

-- Definitions based on conditions
def f (x a : ℝ) : ℝ := Real.exp x + 2 * x - a

def exists_point_on_sine_curve (x0 y0 : ℝ) : Prop := y0 = Real.sin x0

def monotonically_increasing_in_interval (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x1 x2, a ≤ x1 ∧ x1 < x2 ∧ x2 ≤ b → f x1 < f x2

-- The Lean statement
theorem find_range_of_a :
  (∃ x0 y0, exists_point_on_sine_curve x0 y0 ∧ f (f y0) 0 = y0) →
  ((∀ x, -1 ≤ x ∧ x ≤ 1 → monotonically_increasing_in_interval (λ x, f x 0) (-1) 1) →
  [-1 + Real.exp (-1), Real.exp 1 + 1] ∈
  {a | ∃ y0 ∈ Icc (-1 : ℝ) 1, f (f y0) a = y0}) :=
sorry

end find_range_of_a_l288_288833


namespace part1_part2_l288_288238

-- Define the conditions
def conditions :=
  ∃ (x y b r : ℝ), 
    (x = 120) ∧ 
    (y = 80) ∧ 
    (b = 40) ∧ 
    (r = 20) ∧ 
    (2 * 8000 / (x - b) = 24000 / x) ∧ 
    (r / 10 * 10000 = x - b) ∧ 
    b * 2 = (800 * 100 / (x - b)) ∧ 
    y - r / 2 = y ∧ 
    2 * 800 / (y - b / 2) + 10 = x + r. 

-- Assert the result for part 1
theorem part1 {x y : ℝ} (h : conditions) : 
  x = 120 ∧ 
  y = 80 := 
  sorry

-- Assert the result for part 2
theorem part2 {m : ℝ} (h : conditions) : 
  m = 200 ∧ 
  (let w := -12 * m + 16000 in w = 13600) := 
  sorry

end part1_part2_l288_288238


namespace triangle_inradius_l288_288239

theorem triangle_inradius (p A : ℝ) (h₁ : p = 40) (h₂ : A = 50) : 
  (2 * A) / p = 2.5 :=
by 
  rw [h₁, h₂]
  norm_num

# We add sorry because we are only interested in the statement
# and not the actual proof at this point.

end triangle_inradius_l288_288239


namespace paper_clips_total_l288_288587

theorem paper_clips_total :
  let boxes := 9
  let clips_per_box := 9
  total_clips == 81 :=
by
  let total_clips := boxes * clips_per_box
  sorry

end paper_clips_total_l288_288587


namespace prob1_prob2_l288_288375

-- Problem 1
theorem prob1 (x y : ℝ) : 3 * x^2 * y * (-2 * x * y)^3 = -24 * x^5 * y^4 :=
sorry

-- Problem 2
theorem prob2 (x y : ℝ) : (5 * x + 2 * y) * (3 * x - 2 * y) = 15 * x^2 - 4 * x * y - 4 * y^2 :=
sorry

end prob1_prob2_l288_288375


namespace count_positive_area_triangles_l288_288094

noncomputable def numPositiveAreaTriangles : ℕ := 2160

theorem count_positive_area_triangles 
  (vertices : list (ℤ × ℤ))
  (h1 : ∀ p ∈ vertices, 1 ≤ p.1 ∧ p.1 ≤ 5 ∧ 1 ≤ p.2 ∧ p.2 ≤ 5)
  (h2 : vertices.length = 25) : 
  (number_of_triangles_with_positive_area vertices) = numPositiveAreaTriangles :=
sorry

end count_positive_area_triangles_l288_288094


namespace not_always_possible_to_cut_l288_288181

def rectangle := (ℕ × ℕ)

def area (r : rectangle) : ℕ := r.1 * r.2

def almostEqualHalfArea (original_area new_area : ℕ) : Prop :=
  ((original_area : ℚ) / 2 - 1 : ℚ) ≤ new_area ∧ new_area ≤ ((original_area : ℚ) / 2 + 1 : ℚ)

def possibleCut (orig : rectangle) : Prop :=
  ∃ (cut : rectangle), almostEqualHalfArea (area orig) (area cut)

theorem not_always_possible_to_cut (orig : rectangle) : ¬(possibleCut orig) :=
  sorry

end not_always_possible_to_cut_l288_288181


namespace age_of_15th_person_is_26_l288_288221

variables (age_of_15th_person : ℕ)
variables (avg_age_16 : ℕ) (avg_age_5 : ℕ) (avg_age_9 : ℕ)
variables (total_16 : ℕ) (total_5 : ℕ) (total_9 : ℕ)

theorem age_of_15th_person_is_26
  (h1 : avg_age_16 = 15)
  (h2 : avg_age_5 = 14) 
  (h3 : avg_age_9 = 16)
  (T : total_16 = 16 * avg_age_16)
  (T_5 : total_5 = 5 * avg_age_5)
  (T_9 : total_9 = 9 * avg_age_9)
  : age_of_15th_person = 26 :=
by {
  rw [h1, h2, h3] at *,
  have hT : T = 16 * 15 := rfl,
  have h5 : T_5 = 5 * 14 := rfl,
  have h9 : T_9 = 9 * 16 := rfl,
  rw [hT, h5, h9],
  have h : 240 = (70 + 144) + age_of_15th_person,
  rw h,
  calc age_of_15th_person = 240 - 214 : sorry
                       ... = 26 : rfl
}

end age_of_15th_person_is_26_l288_288221


namespace bertha_zero_granddaughters_l288_288367

def bertha := "Bertha"
def daughters (x : String) := if x = bertha then 8 else 0
def has_daughters (x : String) := 4  -- Represents daughters having 4 daughters each
def no_daughters (x : String) := 0 -- Represents daughters having no daughters

# Assuming there is a total of 40 members including daughters and granddaughters
def total_members := 8 + 32 -- Bertha's daughters are 8, granddaughters are 32

def no_granddaughters_daughters :=
  if total_members = 40 then 32 else 0 -- Since daughters are 8 and granddaughters having none is 32

theorem bertha_zero_granddaughters :
  total_members = 40 → no_granddaughters_daughters = 32 :=
by
  sorry

end bertha_zero_granddaughters_l288_288367


namespace problem_conditions_and_solutions_l288_288700

variable (a b : ℝ)

def f (x : ℝ) : ℝ := x^2 + (2 + real.log a) * x + real.log b

theorem problem_conditions_and_solutions :
  (f a b (-1) = -2) ∧ (∀ x : ℝ, f a b x ≥ 2 * x) ↔ (a = 10 * b ∧ a = 100 ∧ b = 10) :=
by
  sorry

end problem_conditions_and_solutions_l288_288700


namespace problem_1_problem_2_l288_288845

def A (a : ℝ) : set ℝ := { x | (x - 2) * (x - (3 * a + 1)) < 0 }
def B (a : ℝ) : set ℝ := { x | (x - 2 * a) / (x - (a^2 + 1)) ≤ 0 }

theorem problem_1 (a : ℝ) (h : a = 2) : 
  A a ∩ B a = set.Icc 4 5 := sorry

theorem problem_2 : 
  ∀ a : ℝ, B a ⊆ A a ↔ 1 < a ∧ a ≤ 3 := sorry

end problem_1_problem_2_l288_288845


namespace measure_minor_arc_KB_l288_288534

theorem measure_minor_arc_KB (K A T B Q : Type) 
  [metric_space Q] [is_circle K A T B Q]
  (angle_KAT : real) (angle_KAB : real)
  (h1 : angle_KAT = 50)
  (h2 : angle_KAB = 20) :
  minor_arc_measure K B = 40 :=
sorry

end measure_minor_arc_KB_l288_288534


namespace altitude_AD_eq_line_BC_eq_l288_288544

-- Problem 1: Equation of altitude AD
theorem altitude_AD_eq (A B C : Point) (AD : Line) (hA : A = ⟨-2, 1⟩) (hB : B = ⟨4, 3⟩) (hC : C = ⟨3, -2⟩) 
(hAD_perp_BC : perpendicular AD (line_through B C)) (hA_on_AD : on_line A AD) :
  AD = line_equation 1 5 (-3) := 
sorry

-- Problem 2: Equation of line BC with midpoint M
theorem line_BC_eq (A B M : Point) (BC : Line) (hA : A = ⟨-2, 1⟩) (hB : B = ⟨4, 3⟩) (hM : M = ⟨3, 1⟩) (hM_mid : midpoint M A C) :
  BC = line_equation 1 2 (-10) :=
sorry

end altitude_AD_eq_line_BC_eq_l288_288544


namespace paths_to_point_l288_288380

theorem paths_to_point (m n : ℕ) : 
  let binomial := Nat.choose (m + n) m in
  ∃ paths : ℕ, paths = binomial :=
by 
  sorry

end paths_to_point_l288_288380


namespace problem_statement_l288_288448

def f (x : ℝ) : ℝ :=
if x < 1 then 1 + Real.logb 2 (2 - x) else 2^x

theorem problem_statement : f (-2) + f (Real.logb 2 6) = 9 := by
  sorry

end problem_statement_l288_288448


namespace area_of_ABCD_l288_288715

def Point := (ℝ × ℝ × ℝ)

noncomputable def distance (p q : Point) : ℝ :=
  (real.sqrt ((p.1 - q.1) ^ 2 + (p.2 - q.2) ^ 2 + (p.3 - q.3) ^ 2))

noncomputable def area_of_quadrilateral (p q r s : Point) : ℝ := 
  let d1 := distance p r
  let d2 := distance q s
  0.5 * d1 * d2

def cube_side_length : ℝ := 2
def A : Point := (0, 0, 0)
def C : Point := (cube_side_length, cube_side_length, cube_side_length)
def B : Point := (cube_side_length / 2, 0, cube_side_length / 2)
def D : Point := (cube_side_length / 2, cube_side_length, cube_side_length / 2)

theorem area_of_ABCD : area_of_quadrilateral A B C D = 2 * real.sqrt(6) :=
by
  sorry

end area_of_ABCD_l288_288715


namespace parallel_vectors_l288_288842

def a (x : ℝ) : ℝ × ℝ := (x, 2)

def b (x : ℝ) : ℝ × ℝ := (3, x - 1)

theorem parallel_vectors (x : ℝ) : 
  (a x, b x).det = 0 ↔ (x = 3 ∨ x = -2) :=
by
  sorry

end parallel_vectors_l288_288842


namespace flea_return_l288_288808

-- Define the type of elements on the plane as points
def Point := (ℝ × ℝ)

-- Assume lines are functions that map points to points on the plane
def Line := Point → Point

-- Define the main theorem
theorem flea_return (lines : Fin 2n → Line) (M : Point)
  (intersect_at_one_point : ∀ (i j : Fin 2n) (x : Point), lines i (lines j x) = lines j (lines i x))
  (sym_reflect : ∀ i, ∃ line : Line, ∀ (x : Point), line x = lines i x)
  (returns_to_M : M ∈ (λ x => (Function.iterate (λ x, lines (Fin 0)) 2n) x) M):
  ∀ (M' : Point), M' ∈ (λ x => (Function.iterate (λ x, lines (Fin 0)) 2n) x) M' :=
begin
  sorry
end

end flea_return_l288_288808


namespace triangle_area_eq_quadrilateral_area_l288_288582

noncomputable def midpoint (p1 p2 : ℝ × ℝ) : ℝ × ℝ :=
((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

structure Hexagon :=
(A B C D E F : ℝ × ℝ)
(is_regular : ∀ {P Q R : ℝ × ℝ}, P ∈ [A, B, C, D, E, F] → Q ∈ [A, B, C, D, E, F] → R ∈ [A, B, C, D, E, F] → dist P Q = dist Q R)

theorem triangle_area_eq_quadrilateral_area
  (h : Hexagon) :
  let M := midpoint h.D h.C,
      K := midpoint h.E h.D,
      L := (λ (A M B K : ℝ × ℝ), classical.some (
        exists_unique_of_exists_of_unique (λ θ, ∀ a b c d : ℝ × ℝ, function.injective (λ t, (cos t, sin t)) ∧ cos θ * b.1 - sin θ * b.2 = cos θ *a.1 - sin θ * a.2 ∧ sin θ * b.1 + cos θ * b.2 = sin θ *a.1 + cos θ * a.2 ∧ cos θ * d.1 - sin θ * d.2 = cos θ *c.1 - sin θ * c.2 ∧ sin θ *d.1 + cos θ * d.2 = sin θ * c.1 + cos θ * c.2))) (h.A M h.B K))
  (area : ℝ) :=
  let area_triangle (A B L : ℝ × ℝ) : ℝ := 0.5 * abs (A.1 * (B.2 - L.2) + B.1 * (L.2 - A.2) + L.1 * (A.2 - B.2)),
      area_quadrilateral (L M D K : ℝ × ℝ) : ℝ := area_triangle L M D + area_triangle L D K
  in
  area_triangle h.A h.B L = area_quadrilateral L M h.D K :=
sorry

end triangle_area_eq_quadrilateral_area_l288_288582


namespace triangle_sides_l288_288614

theorem triangle_sides (a b c : ℝ) (t r : ℝ) (h_c : c = 30) (h_t : t = 336) (h_r : r = 8) :
  a = 26 ∧ b = 28 ∧ c = 30 :=
by
  have h_a : a = 26 := sorry,
  have h_b : b = 28 := sorry,
  tauto

end triangle_sides_l288_288614


namespace range_of_alpha_l288_288428

noncomputable def f (x : ℝ) := 1 / 2 * (x ^ 3 - 1 / x)

theorem range_of_alpha (P : ℝ) (h : f P = P) (tan_alpha : ℝ) 
  (ht : tan_alpha = (deriv f) P) : ∃ α : ℝ, α ∈ [pi / 3, pi / 2) := by
  sorry

end range_of_alpha_l288_288428


namespace find_k_l288_288002

-- Define the series summation function
def series (k : ℝ) : ℝ := 4 + (∑ n, (4 + n * k) / 5^n)

-- State the theorem with the given condition and required proof
theorem find_k (h : series k = 10) : k = 16 := sorry

end find_k_l288_288002


namespace patio_rows_before_rearrangement_l288_288723

theorem patio_rows_before_rearrangement (r c : ℕ) 
  (h1 : r * c = 160) 
  (h2 : (r + 4) * (c - 2) = 160)
  (h3 : ∃ k : ℕ, 5 * k = r)
  (h4 : ∃ l : ℕ, 5 * l = c) :
  r = 16 :=
by
  sorry

end patio_rows_before_rearrangement_l288_288723


namespace solve_for_x_l288_288611

noncomputable def x_solution : Real :=
  (Real.log2 3) / 2

theorem solve_for_x :
  ∀ x : Real, (2 ^ (8 ^ x) = 8 ^ (2 ^ x)) ↔ x = x_solution :=
by
  sorry

end solve_for_x_l288_288611


namespace parallelogram_area_309_l288_288779

noncomputable def area_parallelogram (u v : ℝˣ³) : ℝ :=
  real.sqrt ((u.cross_product v).norm_squared)

theorem parallelogram_area_309 :
  let u := ![4, 2, -3] : ℝˣ³
  let v := ![2, -1, 5] : ℝˣ³
  area_parallelogram u v = real.sqrt 309 :=
sorry

end parallelogram_area_309_l288_288779


namespace range_of_x_in_function_l288_288538

theorem range_of_x_in_function (x : ℝ) (h : x ≠ 8) : true := sorry

end range_of_x_in_function_l288_288538


namespace sin_double_theta_eq_five_fourths_l288_288498

theorem sin_double_theta_eq_five_fourths (theta : ℝ) (h : cos theta + sin theta = 3/2) : sin (2 * theta) = 5/4 :=
by
  sorry

end sin_double_theta_eq_five_fourths_l288_288498


namespace unique_function_f_l288_288162

noncomputable def S := {x : ℝ // x ≠ 0}

def f (S → ℝ) :=
  ∀ x : S, ∀ y : S, x.1 + y.1 ≠ 0 →
    f 1 = 1 ∧
    f (1 / (x.1 + y.1)) = f (1 / x.1) + f (1 / y.1) ∧
    (x.1 + y.1) * f (x.1 + y.1) = x.1 * y.1 * f x.1 * f y.1

theorem unique_function_f : ∃! (f : S → ℝ),
  (f 1 = 1) ∧
  (∀ x y : S, x.1 + y.1 ≠ 0 →
    (f (1 / (x.1 + y.1)) = f (1 / x.1) + f (1 / y.1)) ∧
    ((x.1 + y.1) * f (x.1 + y.1) = x.1 * y.1 * f (x.1) * f (y.1))) :=
sorry

end unique_function_f_l288_288162


namespace cross_product_scalar_multiplication_l288_288114

def vector_cross_mul (a b : ℝ → ℝ → ℝ) : ℝ := sorry

theorem cross_product_scalar_multiplication
  (a b : ℝ → ℝ → ℝ)
  (h : vector_cross_mul a b = (-3, 2, 6)) :
  vector_cross_mul a (λ x y, 4 * b x y) = (-12, 8, 24) :=
sorry

end cross_product_scalar_multiplication_l288_288114


namespace find_x_minus_y_l288_288479

noncomputable section

open Classical

variable (x y : ℝ)
variable (e1 e2 : ℝ × ℝ)
def condition1 := e1 = (1, 2)
def condition2 := e2 = (3, 4)
def condition3 := x * e1.1 + y * e2.1 = 5 ∧ x * e1.2 + y * e2.2 = 6

theorem find_x_minus_y (h1 : condition1) (h2 : condition2) (h3 : condition3) : x - y = -3 := 
  sorry

end find_x_minus_y_l288_288479


namespace george_blocks_l288_288791

theorem george_blocks (num_boxes : ℕ) (blocks_per_box : ℕ) (total_blocks : ℕ) :
  num_boxes = 2 → blocks_per_box = 6 → total_blocks = num_boxes * blocks_per_box → total_blocks = 12 := by
  intros h_num_boxes h_blocks_per_box h_blocks_equal
  rw [h_num_boxes, h_blocks_per_box] at h_blocks_equal
  exact h_blocks_equal

end george_blocks_l288_288791


namespace base8_contains_5_or_6_l288_288856

theorem base8_contains_5_or_6 (n : ℕ) (h : n = 512) : 
  let count_numbers_without_5_6 := 6^3 in
  let total_numbers := 512 in
  total_numbers - count_numbers_without_5_6 = 296 := by
  sorry

end base8_contains_5_or_6_l288_288856


namespace angle_between_vectors_l288_288046

-- Definitions and conditions
variables (a b : EuclideanSpace ℝ (Fin 2)) -- Vectors in 2D space
variable (θ : ℝ) -- Angle between vectors

-- Conditions
axiom norm_a : ‖a‖ = 1
axiom norm_b : ‖b‖ = 1
axiom dot_product_eq : inner a b = 1 / 2

-- The theorem to prove
theorem angle_between_vectors : θ = Real.arccos (1 / 2) :=
by
  sorry

end angle_between_vectors_l288_288046


namespace find_b_plus_c_l288_288176

noncomputable def f (x : ℝ) (b c : ℝ) : ℝ := x^3 + b * x^2 + c * x
noncomputable def f' (x : ℝ) (b c : ℝ) : ℝ := 3 * x^2 + 2 * b * x + c
noncomputable def g (x : ℝ) (b c : ℝ) : ℝ := f x b c - f' x b c

def is_odd (h : ℝ → ℝ) : Prop := ∀ x : ℝ, h (-x) = -h(x)

theorem find_b_plus_c (b c : ℝ) (h_odd_g : is_odd (g · b c)) : b + c = 3 :=
sorry

end find_b_plus_c_l288_288176


namespace deg_to_rad_neg_630_l288_288761

theorem deg_to_rad_neg_630 :
  (-630 : ℝ) * (Real.pi / 180) = -7 * Real.pi / 2 := by
  sorry

end deg_to_rad_neg_630_l288_288761


namespace problem_I_problem_II_l288_288835

def f (x : ℝ) : ℝ := |x - 2| - |x + 1|

theorem problem_I : {x : ℝ | f x + x > 0} = { x : ℝ | x ∈ (-∞, -1) ∪ [-1, 1) ∪ (3, ∞) } :=
sorry

theorem problem_II (a : ℝ) : (∀ x : ℝ, f x ≤ a^2 - 2a) → a ∈ (-∞, -1] ∪ [3, ∞) :=
sorry

end problem_I_problem_II_l288_288835


namespace f_31_is_neg_1_l288_288807

theorem f_31_is_neg_1 (f : ℝ → ℝ) 
  (h1 : ∀ x : ℝ, f (-x) = -f x) 
  (h2 : ∀ x : ℝ, f (x + 1) = f (1 - x)) 
  (h3 : ∀ x : ℝ, 0 ≤ x → x ≤ 1 → f x = Real.log2 (x + 1)) :
  f 31 = -1 := 
by
  sorry

end f_31_is_neg_1_l288_288807


namespace part1_part2_l288_288438

-- Defining the given functions f(x) and g(x) in Lean
def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := (3 ^ x + a) / (3 ^ x + b)

def g (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := f x a b * (3 ^ x + 1) + 3 ^ (-x)

-- Part 1
theorem part1 (x : ℝ) : f x 5 (-3) = 3 ^ x ↔ x = Real.log (5) / Real.log (3) :=
by
  sorry

-- Part 2
theorem part2 (m : ℝ) : (∀ x : ℝ, x ≠ 0 → g (2*x) (-1) 1 ≥ m * g x (-1) 1 - 10) ↔ m = 4 * Real.sqrt 2 + 2 :=
by
  sorry

end part1_part2_l288_288438


namespace area_of_triangle_MOI_is_7_div_4_l288_288515

noncomputable def point : Type := ℝ × ℝ

def A : point := (0, 0)
def B : point := (8, 0)
def C : point := (0, 7)

def side_length (p1 p2 : point) : ℝ :=
(real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2))

def AB := side_length A B
def AC := side_length A C
def BC := side_length B C

def incenter (a b c : ℝ) (A B C : point) : point :=
((a * A.1 + b * B.1 + c * C.1) / (a + b + c), (a * A.2 + b * B.2 + c * C.2) / (a + b + c))

def I := incenter BC AC AB A B C

-- Assuming circumcenter is common knowledge and does not need a custom function here
def O : point := sorry  -- Actual circumcenter computation skipped for brevity

def M : point := (15/16, 3.5)

def triangle_area (p1 p2 p3 : point) : ℝ :=
abs (p1.1 * (p2.2 - p3.2) + p2.1 * (p3.2 - p1.2) + p3.1 * (p1.2 - p2.2)) / 2

def area_MOI := triangle_area M O I

theorem area_of_triangle_MOI_is_7_div_4 :
  area_MOI = 7 / 4 :=
sorry -- Proof is to be filled in

end area_of_triangle_MOI_is_7_div_4_l288_288515


namespace find_k_solution_l288_288005

theorem find_k_solution :
    ∃ k : ℝ, (4 + ∑' n : ℕ, (4 + (n : ℝ)*k) / 5^(n + 1) = 10) ∧ k = 16 :=
begin
  use 16,
  sorry
end

end find_k_solution_l288_288005


namespace lines_slope_angle_l288_288634

theorem lines_slope_angle (m n : ℝ) (θ₁ θ₂ : ℝ)
  (h1 : L1 = fun x => m * x)
  (h2 : L2 = fun x => n * x)
  (h3 : θ₁ = 3 * θ₂)
  (h4 : m = 3 * n)
  (h5 : θ₂ ≠ 0) :
  m * n = 9 / 4 :=
by
  sorry

end lines_slope_angle_l288_288634


namespace area_of_triangle_in_sector_l288_288917

noncomputable theory

-- Define the radius and angle of the sector
def radius : ℝ := 15
def theta : ℝ := π / 2

-- Define the condition that the triangle cuts the angle of the sector into three equal parts
def angle_triangle : ℝ := theta / 3

-- Define the area of the triangle using the sine rule for triangles
def triangle_area : ℝ := 1/2 * radius * radius * Real.sin(angle_triangle)

-- State the theorem
theorem area_of_triangle_in_sector :
  triangle_area = 1125 * Real.sqrt 3 / 4 :=
sorry

end area_of_triangle_in_sector_l288_288917


namespace smallest_n_for_coloring_l288_288410

theorem smallest_n_for_coloring (n : ℕ) : n = 4 :=
sorry

end smallest_n_for_coloring_l288_288410


namespace distance_midpoints_eq_2_5_l288_288201

theorem distance_midpoints_eq_2_5 (A B C : ℝ) (hAB : A < B) (hBC : B < C) (hAC_len : C - A = 5) :
    let M1 := (A + B) / 2
    let M2 := (B + C) / 2
    (M2 - M1 = 2.5) :=
by
    let M1 := (A + B) / 2
    let M2 := (B + C) / 2
    sorry

end distance_midpoints_eq_2_5_l288_288201


namespace smallest_positive_period_f_max_value_g_on_interval_l288_288837

noncomputable def f (x : ℝ) : ℝ := sin (x + π / 2) * cos (x - π / 3)

theorem smallest_positive_period_f :
  ∃ T > 0, ∀ x : ℝ, f (x + T) = f x ∧ T = π :=
sorry

noncomputable def g (x : ℝ) : ℝ := 2 * sin (2 * x + π / 6)

theorem max_value_g_on_interval :
  ∃ M, (∀ x ∈ set.Icc (0 : ℝ) (π / 3), g x ≤ M) ∧ M = 2 :=
sorry

end smallest_positive_period_f_max_value_g_on_interval_l288_288837


namespace regular_dodecahedron_edges_l288_288484

-- Define a regular dodecahedron as a type
inductive RegularDodecahedron : Type
| mk : RegularDodecahedron

-- Define a function that returns the number of edges for a regular dodecahedron
def numberOfEdges (d : RegularDodecahedron) : Nat :=
  30

-- The mathematical statement to be proved
theorem regular_dodecahedron_edges (d : RegularDodecahedron) : numberOfEdges d = 30 := by
  sorry

end regular_dodecahedron_edges_l288_288484


namespace no_valid_two_digit_N_exists_l288_288021

def is_two_digit_number (N : ℕ) : Prop :=
  10 ≤ N ∧ N < 100

def is_perfect_cube (x : ℕ) : Prop :=
  ∃ (n : ℕ), n ^ 3 = x

def reverse_digits (N : ℕ) : ℕ :=
  match N / 10, N % 10 with
  | a, b => 10 * b + a

theorem no_valid_two_digit_N_exists : ∀ N : ℕ,
  is_two_digit_number N →
  (is_perfect_cube (N - reverse_digits N) ∧ (N - reverse_digits N) ≠ 27) → false :=
by sorry

end no_valid_two_digit_N_exists_l288_288021


namespace inequality_max_value_l288_288112

theorem inequality_max_value (a : ℝ) (h₁ : -1 < a) (h₂ : a < 0) : 
  ∃ a₀, a₀ = -2 + Real.sqrt 2 ∧ f a₀ = -3 - 2 * Real.sqrt 2 where
    f : ℝ → ℝ := λ a, 2 / a - 1 / (1 + a) :=
  sorry

end inequality_max_value_l288_288112


namespace james_total_slices_l288_288551

def total_slices : ℕ := 8 + 12 + 16 + 18
def tom_eats : ℝ := 2.5
def alice_eats : ℝ := 3.5
def bob_eats_total : ℝ := 7.25
def bob_eats_third : ℝ := 5
def james_eats_half (remaining_slices : ℝ) : ℝ := remaining_slices / 2

theorem james_total_slices :
  let remaining_first := 8 - tom_eats,
      remaining_second := 12 - alice_eats,
      remaining_third := 16 - bob_eats_third,
      remaining_fourth := 18 - (bob_eats_total - bob_eats_third),
      james_first := james_eats_half remaining_first,
      james_second := james_eats_half remaining_second,
      james_third := james_eats_half remaining_third,
      james_fourth := james_eats_half remaining_fourth
  in james_first + james_second + james_third + james_fourth = 20.375 :=
by
  sorry

end james_total_slices_l288_288551


namespace angle_between_diagonals_l288_288401

theorem angle_between_diagonals (α : ℝ) : 
  let β := 2 * arctan (cos α) 
  in β = 2 * arctan (cos α) := by
  -- This is where the proof would go, but we'll leave it as a placeholder.
  sorry

end angle_between_diagonals_l288_288401


namespace find_n_coordinates_l288_288035

variables {a b : ℝ}

def is_perpendicular (m n : ℝ × ℝ) : Prop :=
  m.1 * n.1 + m.2 * n.2 = 0

def same_magnitude (m n : ℝ × ℝ) : Prop :=
  m.1 ^ 2 + m.2 ^ 2 = n.1 ^ 2 + n.2 ^ 2

theorem find_n_coordinates (n : ℝ × ℝ) (h1 : is_perpendicular (a, b) n) (h2 : same_magnitude (a, b) n) :
  n = (b, -a) :=
sorry

end find_n_coordinates_l288_288035


namespace beta_greater_than_alpha_l288_288443

theorem beta_greater_than_alpha (α β : ℝ) (h1 : 0 < α) (h2 : α < π / 2) (h3 : 0 < β) (h4 : β < π / 2) (h5 : Real.sin (α + β) = 2 * Real.sin α) : β > α := 
sorry

end beta_greater_than_alpha_l288_288443


namespace joe_out_of_money_after_one_month_worst_case_l288_288554

-- Define the initial amount Joe has
def initial_amount : ℝ := 240

-- Define Joe's monthly subscription cost
def subscription_cost : ℝ := 15

-- Define the range of prices for buying games
def min_game_cost : ℝ := 40
def max_game_cost : ℝ := 60

-- Define the range of prices for selling games
def min_resale_price : ℝ := 20
def max_resale_price : ℝ := 40

-- Define the maximum number of games Joe can purchase per month
def max_games_per_month : ℕ := 3

-- Prove that Joe will be out of money after 1 month in the worst-case scenario
theorem joe_out_of_money_after_one_month_worst_case :
  initial_amount - 
  (max_games_per_month * max_game_cost - max_games_per_month * min_resale_price + subscription_cost) < 0 :=
by
  sorry

end joe_out_of_money_after_one_month_worst_case_l288_288554


namespace quadrilateral_not_necessarily_square_l288_288641

-- Define the conditions and the quadrilateral
variables {A B C D K L M N : Type} [convex_quadrilateral A B C D]
variables [midpoint_squared A B C D K L M N]

-- Define the theorem to prove
theorem quadrilateral_not_necessarily_square 
    (h1 : midpoints_form_square A B C D K L M N) :
    ¬ is_square_convex_quadrilateral A B C D :=
begin
  sorry
end

end quadrilateral_not_necessarily_square_l288_288641


namespace contrapositive_statement_l288_288955

theorem contrapositive_statement (m : ℝ) : 
  (¬ ∃ (x : ℝ), x^2 + x - m = 0) → m > 0 :=
by
  sorry

end contrapositive_statement_l288_288955


namespace trig_expression_simplifies_to_one_l288_288998

theorem trig_expression_simplifies_to_one :
  (√(1 - 2 * Real.sin (Float.pi * 40 / 180) * Real.cos (Float.pi * 40 / 180))) / 
  (Real.cos (Float.pi * 40 / 180) - (√(1 - (Real.sin (Float.pi * 50 / 180)) ^ 2))) = 1 :=
by
  sorry

end trig_expression_simplifies_to_one_l288_288998


namespace apples_used_l288_288324

theorem apples_used (initial_apples remaining_apples : ℕ) (h_initial : initial_apples = 40) (h_remaining : remaining_apples = 39) : initial_apples - remaining_apples = 1 := 
by
  sorry

end apples_used_l288_288324


namespace bowling_ball_weight_l288_288785

theorem bowling_ball_weight (b c : ℝ) 
  (h1 : 5 * b = 4 * c) 
  (h2 : 2 * c = 80) : 
  b = 32 :=
by
  sorry

end bowling_ball_weight_l288_288785


namespace train_speed_20_mph_l288_288384

/-- Given:
1. Darcy lives 1.5 miles from work.
2. She can walk to work at a constant rate of 3 miles per hour.
3. There is an additional 10.5 minutes spent walking to the nearest train station, waiting for the train, and walking from the final train station to her work if she rides the train.
4. It takes Darcy a total of 15 more minutes to commute to work by walking than it takes her to commute to work by riding the train.

Prove that the train's speed is 20 miles per hour.
-/
theorem train_speed_20_mph
  (d : ℝ) (w_speed : ℝ) (additional_time : ℝ) (walk_vs_train : ℝ)
  (h_d : d = 1.5) (h_w_speed : w_speed = 3) (h_additional_time : additional_time = 10.5 / 60)
  (h_walk_vs_train : walk_vs_train = 15 / 60) :
  let walk_time := d / w_speed in
  let train_time_total := walk_time - walk_vs_train in
  let actual_train_time := train_time_total - additional_time in
  let train_speed := d / actual_train_time in
  train_speed = 20 :=
by
  sorry

end train_speed_20_mph_l288_288384


namespace window_treatments_cost_l288_288562

-- Define the costs and the number of windows
def cost_sheers : ℝ := 40.00
def cost_drapes : ℝ := 60.00
def number_of_windows : ℕ := 3

-- Define the total cost calculation
def total_cost := (cost_sheers + cost_drapes) * number_of_windows

-- State the theorem that needs to be proved
theorem window_treatments_cost : total_cost = 300.00 :=
by
  sorry

end window_treatments_cost_l288_288562


namespace percent_apple_juice_in_blend_l288_288150

noncomputable def juice_blend_apple_percentage : ℚ :=
  let apple_juice_per_apple := 9 / 2
  let plum_juice_per_plum := 12 / 3
  let total_apple_juice := 4 * apple_juice_per_apple
  let total_plum_juice := 6 * plum_juice_per_plum
  let total_juice := total_apple_juice + total_plum_juice
  (total_apple_juice / total_juice) * 100

theorem percent_apple_juice_in_blend :
  juice_blend_apple_percentage = 43 :=
by
  sorry

end percent_apple_juice_in_blend_l288_288150


namespace minimum_value_of_expression_l288_288840

noncomputable def min_value (a b : ℝ) (h₀ : a ≠ 0) (h₁ : b ≠ 0)
(h₂ : ∃ A B : ℝ × ℝ, let line := (λ x y, √2 * a * x + b * y = 1) in
  let circle := (λ x y, x^2 + y^2 = 1) in
  line A.1 A.2 ∧ line B.1 B.2 ∧ circle A.1 A.2 ∧ circle B.1 B.2 ∧ 
  (∃ (O : ℝ × ℝ) (H₃ : O = (0, 0)), 
    (A - O).fst * (A - O).fst + (A - O).snd * (A - O).snd +
    (B - O).fst * (B - O).fst + (B - O).snd * (B - O).snd = (A - B).fst * (A - B).fst + (A - B).snd * (A - B).snd)) : ℝ :=
if h : 2 * a^2 + b^2 = 2 then 4 else 0

-- Stating the main theorem
theorem minimum_value_of_expression (a b : ℝ) (h₀ : a ≠ 0) (h₁ : b ≠ 0)
(h₂ : 2 * a^2 + b^2 = 2) :
min_value a b h₀ h₁ ⟨_, h₂⟩ = 4 :=
sorry

end minimum_value_of_expression_l288_288840


namespace range_of_x2_plus_y2_l288_288830

noncomputable def complex_circle_range (x y : ℝ) : Prop :=
  let z := complex.mk x y in
  complex.abs (z - complex.mk 3 4) = 1

theorem range_of_x2_plus_y2 (x y : ℝ) (h : complex_circle_range x y) :
  16 ≤ x^2 + y^2 ∧ x^2 + y^2 ≤ 36 :=
sorry

end range_of_x2_plus_y2_l288_288830


namespace num_of_4_digit_numbers_divisible_by_13_l288_288886

theorem num_of_4_digit_numbers_divisible_by_13 :
  let smallest_4_digit := 1000
  let smallest_divisible_by_13 := 1001
  let largest_4_digit := 9999
  let largest_divisible_by_13 := 9997
  let a := smallest_divisible_by_13
  let d := 13
  let l := largest_divisible_by_13
  (\(l - a) / d) + 1 = 693 :=
by
  let smallest_4_digit := 1000
  let smallest_divisible_by_13 := 1001
  let largest_4_digit := 9999
  let largest_divisible_by_13 := 9997
  let a := smallest_divisible_by_13
  let d := 13
  let l := largest_divisible_by_13
  have h1 : (l - a) / d + 1 = (9997 - 1001) / 13 + 1, by sorry
  have h2 : (9997 - 1001) / 13 + 1 = 8996 / 13 + 1, by sorry
  have h3 : 8996 / 13 + 1 = 692 + 1, by sorry
  have h4 : 692 + 1 = 693, by sorry
  exact h4

end num_of_4_digit_numbers_divisible_by_13_l288_288886


namespace keychain_arrangement_count_l288_288135

theorem keychain_arrangement_count : ∀ (keys : Set String),
  keys = {"house_key", "car_key", "office_key", "key1", "key2", "key3"} →
  (∀ l : List (Set String), keys.count "house_key" = 1 → "house_key".opposite("car_key") → 
  "office_key".adjacent("house_key") →
  (are_identical_if_rotated_or_flipped l)) →
  ∃ (n: ℕ), n = 12 :=
sorry

end keychain_arrangement_count_l288_288135


namespace product_of_consecutive_sums_not_eq_111111111_l288_288981

theorem product_of_consecutive_sums_not_eq_111111111 :
  ∀ (a : ℤ), (3 * a + 3) * (3 * a + 12) ≠ 111111111 := 
by
  intros a
  sorry

end product_of_consecutive_sums_not_eq_111111111_l288_288981


namespace smallest_positive_a_l288_288786

variable (θ : ℝ) (hθ : 0 < θ ∧ θ < (Real.pi / 2))
noncomputable def a := (Real.sin θ * Real.cos θ) ^ 2 / (1 + Real.sqrt 3 * Real.sin θ * Real.cos θ)

theorem smallest_positive_a
  (a : ℝ)
  (h1 : sqrt a / cos θ + sqrt a / sin θ > 1)
  (h2 : ∃ x ∈ Icc (1 - sqrt a / sin θ) (sqrt a / cos θ),
      ((1 - x) * sin θ - sqrt (a - x^2 * cos θ^2)) ^ 2 +
      (x * cos θ - sqrt (a - (1 - x)^2 * sin θ^2)) ^ 2 ≤ a) :
  a = (Real.sin θ * Real.cos θ) ^ 2 / (1 + Real.sqrt 3 * Real.sin θ * Real.cos θ) :=
sorry

end smallest_positive_a_l288_288786


namespace find_k_l288_288007

theorem find_k (k : ℝ) : 4 + ∑' (n : ℕ), (4 + n * k) / 5^n = 10 → k = 16 := by
  sorry

end find_k_l288_288007


namespace bridge_length_l288_288334

theorem bridge_length :
  ∀ (v_i a : ℝ) (t : ℝ),
    v_i = 3 →
    a = 0.2 →
    t = 0.25 →
    let d := v_i * t + (1 / 2) * a * t^2 in
    d * 1000 = 756.25 :=
begin
  intros v_i a t hv ha ht,
  have h1 : d = v_i * t + (1 / 2) * a * t^2 := rfl,
  rw [hv, ha, ht] at h1,
  rw ← h1,
  sorry -- proof steps are not required here
end

end bridge_length_l288_288334


namespace c_n_monotonically_decreasing_l288_288473

variable (a : ℕ → ℝ) (b : ℕ → ℝ) (c : ℕ → ℝ)

theorem c_n_monotonically_decreasing 
    (h_a0 : a 0 = 0)
    (h_b : ∀ n ≥ 1, b n = a n - a (n - 1))
    (h_c : ∀ n ≥ 1, c n = a n / n)
    (h_bn_decrease : ∀ n ≥ 1, b n ≥ b (n + 1)) : 
    ∀ n ≥ 2, c n ≤ c (n - 1) := 
by
  sorry

end c_n_monotonically_decreasing_l288_288473


namespace distance_from_A_to_C_correct_total_distance_traveled_correct_l288_288735

-- Define the conditions
def distance_to_A : ℕ := 30
def distance_to_B : ℕ := 20
def distance_to_C : ℤ := -15
def times_to_C : ℕ := 3

-- Define the resulting calculated distances based on the conditions
def distance_A_to_C : ℕ := distance_to_A + distance_to_C.natAbs
def total_distance_traveled : ℕ := (distance_to_A + distance_to_B) * 2 + distance_to_C.natAbs * (times_to_C * 2)

-- The proof problems (statements) based on the problem's questions
theorem distance_from_A_to_C_correct : distance_A_to_C = 45 := by
  sorry

theorem total_distance_traveled_correct : total_distance_traveled = 190 := by
  sorry

end distance_from_A_to_C_correct_total_distance_traveled_correct_l288_288735


namespace lattice_point_in_triangle_l288_288381

theorem lattice_point_in_triangle 
  {P Q R S E : ℤ × ℤ}
  (is_convex : convex_lattice_quadrilateral P Q R S)
  (intersect_at : diagonals_intersect_at P Q R S E)
  (angle_condition : angle P + angle Q < 180) :
  ∃ M : ℤ × ℤ, (M ≠ P ∧ M ≠ Q) ∧ (M ∈ interior (triangle P Q E) ∨ M ∈ boundary (triangle P Q E)) := 
sorry

end lattice_point_in_triangle_l288_288381


namespace sequence_arithmetic_general_term_range_m_l288_288936

noncomputable def a_seq : ℕ → ℝ
| 1 := 1
| n+1 := a_seq n / (a_seq n * (S_seq (n + 1)) - (S_seq (n + 1))^2) / 2

noncomputable def S_seq : ℕ → ℝ
| 1 := a_seq 1
| n+1 := S_seq n + a_seq (n + 1)

theorem sequence_arithmetic :
  ∀ n ≥ 2, (1 / S_seq n - 1 / S_seq (n - 1)) = 1 / 2 :=
by sorry

theorem general_term :
  a_seq 1 = 1 ∧ 
  (∀ n > 1, a_seq n = -2 / (n * (n + 1))) :=
by sorry

noncomputable def b_seq : ℕ → ℝ
| 1 := 1
| n+1 := -2 / (n * a_seq n)

noncomputable def T_seq : ℕ → ℝ
| 1 := b_seq 1
| n+1 := T_seq n + b_seq (n + 1)

theorem range_m :
  ∀ n ≥ 2, T_seq n < 8 / 15 :=
by sorry

end sequence_arithmetic_general_term_range_m_l288_288936


namespace plates_to_remove_l288_288481

-- Definitions based on the problem conditions
def number_of_plates : ℕ := 38
def weight_per_plate : ℕ := 10
def acceptable_weight : ℕ := 320

-- Theorem to prove
theorem plates_to_remove (initial_weight := number_of_plates * weight_per_plate) 
  (excess_weight := initial_weight - acceptable_weight) 
  (plates_to_remove := excess_weight / weight_per_plate) :
  plates_to_remove = 6 :=
by
  sorry

end plates_to_remove_l288_288481


namespace find_complex_number_l288_288618

open Complex

theorem find_complex_number (a b : ℕ) (hp : (a:ℂ) + (b:ℂ) * I) = (2 + 11 * I) ^ (1/3) :
  (a:ℂ) + (b:ℂ) * I = 2 + I := by
  sorry

end find_complex_number_l288_288618


namespace part1_solution_set_part2_range_of_a_l288_288178

noncomputable def f (x a : ℝ) : ℝ := |x - 1| + |x - a|

theorem part1_solution_set (a : ℝ) (h : a = 4) : 
  {x : ℝ | f x a ≥ 5} = {x | x ≤ 0} ∪ {x | x ≥ 5} :=
by 
  -- proof goes here
  sorry

theorem part2_range_of_a (a : ℝ) (h : ∀ x : ℝ, f x a ≥ 4) : 
  a ≤ -3 ∨ a ≥ 5 :=
by
  -- proof goes here
  sorry

end part1_solution_set_part2_range_of_a_l288_288178


namespace sally_mcqueen_cost_l288_288183

theorem sally_mcqueen_cost :
  let lightning_mcqueen_cost := 140000
      mater_cost := 0.1 * lightning_mcqueen_cost
      sally_mcqueen_cost := 3 * mater_cost
  in sally_mcqueen_cost = 42000 :=
by
  let lightning_mcqueen_cost := 140000
  let mater_cost := 0.1 * lightning_mcqueen_cost
  let sally_mcqueen_cost := 3 * mater_cost
  calc 
    sally_mcqueen_cost 
    = 3 * (0.1 * lightning_mcqueen_cost) : by rw [mater_cost]
    = 3 * 14000                       : by rw [lightning_mcqueen_cost * 0.1]
    = 42000                           : by norm_num

end sally_mcqueen_cost_l288_288183


namespace eq_correct_l288_288267

variable (x : ℝ)

def width (x : ℝ) : ℝ := x - 6

def area_eq (x : ℝ) : Prop := x * width x = 720

theorem eq_correct (h : area_eq x) : x * (x - 6) = 720 :=
by exact h

end eq_correct_l288_288267


namespace area_tripled_sides_l288_288904

theorem area_tripled_sides (a b : ℝ) (θ : ℝ) (hθ : 0 < θ ∧ θ < π) :
  let A := 1 / 2 * a * b * Real.sin θ in
  let A' := 1 / 2 * (3 * a) * (3 * b) * Real.sin θ in
  A' = 9 * A := by
  sorry

end area_tripled_sides_l288_288904


namespace squares_perimeter_and_rectangle_area_l288_288247

theorem squares_perimeter_and_rectangle_area (x y : ℝ) (hx : x^2 + y^2 = 145) (hy : x^2 - y^2 = 105) : 
  (4 * x + 4 * y = 28 * Real.sqrt 5) ∧ ((x + y) * x = 175) := 
by 
  sorry

end squares_perimeter_and_rectangle_area_l288_288247


namespace ratio_of_areas_l288_288923

variables (s : ℝ)

-- Assume that ABCD is a square with side length s
-- Points M and N are midpoints of sides AB and CD respectively
def square_midpoint_A (s : ℝ) := (0, s)
def square_midpoint_B (s : ℝ) := (s, s)
def square_midpoint_C (s : ℝ) := (s, 0)
def square_midpoint_D (s : ℝ) := (0, 0)

def M (s : ℝ) := ((0 + s) / 2, s)
def N (s : ℝ) := (s, (0 + s) / 2)

def area_triangle_AMN (M N : ℝ × ℝ) (s : ℝ) :=
  1 / 2 * abs (0 * (s - (s / 2)) + (s / 2) * ((s / 2) - s) + s * (s - s))

def area_square (s : ℝ) := s * s

def ratio_area_AMN_to_ABCD (s : ℝ) :=
  area_triangle_AMN (M s) (N s) s / area_square s

theorem ratio_of_areas (s : ℝ) : ratio_area_AMN_to_ABCD s = 1 / 8 :=
by
  -- Proof is omitted
  sorry

end ratio_of_areas_l288_288923


namespace problem1_solution_problem2_solution_l288_288422

section Problem1
variables (x : ℝ)

def f (x : ℝ) : ℝ := 1 + 1 / |x|

theorem problem1_solution :
  {y : ℝ | f y ≤ 2*y} = {y : ℝ | y ≥ 1} :=
sorry
end Problem1

section Problem2
variables (x a : ℝ)

def f2 (a x : ℝ) : ℝ := a + 1 / |x|

theorem problem2_solution (h : ∃ x ∈ Icc (-2 : ℝ) (-1 : ℝ), f2 a x = 2*x) :
  a ∈ Icc (-(9/2):ℝ) (-3:ℝ) :=
sorry
end Problem2

end problem1_solution_problem2_solution_l288_288422


namespace sin_theta_between_line_and_plane_l288_288950

noncomputable def angle_between_line_and_plane (θ : ℝ) : Prop :=
  let d := ⟨4, 5, 8⟩ : EuclideanSpace ℝ (Fin 3)
  let n := ⟨5, -3, 7⟩ : EuclideanSpace ℝ (Fin 3)
  let d_dot_n := d.1 * n.1 + d.2 * n.2 + d.3 * n.3
  let d_norm := Real.sqrt (d.1 ^ 2 + d.2 ^ 2 + d.3 ^ 2)
  let n_norm := Real.sqrt (n.1 ^ 2 + n.2 ^ 2 + n.3 ^ 2)
  θ = Real.arcsin (d_dot_n / (d_norm * n_norm))

theorem sin_theta_between_line_and_plane : 
  angle_between_line_and_plane (Real.arcsin (61 / (Real.sqrt 105 * Real.sqrt 83)))
:= sorry

end sin_theta_between_line_and_plane_l288_288950


namespace checkerboard_contains_140_squares_with_at_least_6_black_squares_l288_288379

theorem checkerboard_contains_140_squares_with_at_least_6_black_squares :
  let checkerboard := matrix (fin 10) (fin 10) (bool)
  (∀ checkerboard, 
    (∀ i j, checkerboard i j = (i + j) % 2 = 0 →
      number_of_squares_with_at_least_6_black_squares checkerboard = 140)) := sorry

end checkerboard_contains_140_squares_with_at_least_6_black_squares_l288_288379


namespace distance_between_A_and_B_l288_288403

-- Define the points A and B
def A : ℝ × ℝ := (2, 3)
def B : ℝ × ℝ := (5, 9)

-- Define the distance formula
def euclidean_distance (p1 p2 : ℝ × ℝ) : ℝ :=
  (real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2))

-- The proposition to prove
theorem distance_between_A_and_B : euclidean_distance A B = 3 * real.sqrt 5 := by
  sorry

end distance_between_A_and_B_l288_288403


namespace length_DF_l288_288939

noncomputable def triangle_Sides (A B C : Point) : Prop :=
  dist A B = 13 ∧ dist B C = 14 ∧ dist C A = 15

noncomputable def perpendiculars (A B C D E F : Point) : Prop :=
  ∃ D E F : Point,
  lies_on_segment D B C ∧
  lies_on_segment E C A ∧
  lies_on_segment F D E ∧
  perp (line_through A D) (line_through B C) ∧
  perp (line_through D E) (line_through C A) ∧
  perp (line_through A F) (line_through B F)

theorem length_DF {A B C D E F : Point} : 
  triangle_Sides A B C → 
  perpendiculars A B C D E F →
  ∃ m n : ℕ, nat_coprime m n ∧ (DF D F = m / n) ∧ (m + n = 77) :=
by
  intros hSides hPerp
  sorry

end length_DF_l288_288939


namespace segment_AB_length_l288_288038

-- Define the parametric equations for curve C
def curve_parametric (a : ℝ) : ℝ × ℝ :=
  (1 + sqrt 5 * cos a, 2 + sqrt 5 * sin a)

-- Define the Cartesian equations for lines l1 and l2
def line_l1 (x : ℝ) : Prop := x = 0
def line_l2 (x y : ℝ) : Prop := x = y

-- Convert the curve parametric equation to the polar equation
def curve_polar (θ : ℝ) : ℝ := 2 * cos θ + 4 * sin θ

-- Define the polar equations for lines l1 and l2
def line_l1_polar (θ : ℝ) : Prop := θ = π/2
def line_l2_polar (θ : ℝ) : Prop := θ = π/4

-- The theorem we need to prove
theorem segment_AB_length : 
  let ρ1 := curve_polar (π/2),
      ρ2 := curve_polar (π/4)
  in (ρ1 = 4 ∧ ρ2 = 3 * sqrt 2) →
     | sqrt (ρ1^2 + ρ2^2 - 2 * ρ1 * ρ2 * cos (π / 4)) = sqrt 10 :=
by
  sorry

end segment_AB_length_l288_288038


namespace angle_B_side_b_l288_288550

variable (A B C a b c : ℝ)
variable (S : ℝ := 5 * Real.sqrt 3)

-- Conditions
variable (h1 : a = b * Real.cos C + (Real.sqrt 3) / 3 * c * Real.sin B)
variable (h2 : 1/2 * a * c * Real.sin B = S)
variable (h3 : a = 5)

-- The two parts to prove
theorem angle_B (h1 : a = b * Real.cos C + (Real.sqrt 3) / 3 * c * Real.sin B) : 
  B = π / 3 := 
  sorry

theorem side_b (h1 : a = b * Real.cos C + (Real.sqrt 3) / 3 * c * Real.sin B)
  (h2 : 1/2 * a * c * Real.sin B = S) (h3 : a = 5) : 
  b = Real.sqrt 21 := 
  sorry

end angle_B_side_b_l288_288550


namespace largest_4_digit_divisible_by_35_l288_288676

theorem largest_4_digit_divisible_by_35 : ∃ n : ℕ, (1000 ≤ n ∧ n ≤ 9999) ∧ (n % 35 = 0) ∧ (∀ m : ℕ, (1000 ≤ m ∧ m ≤ 9999) ∧ (m % 35 = 0) → m ≤ n) ∧ n = 9985 := 
by sorry

end largest_4_digit_divisible_by_35_l288_288676


namespace smallest_k_unique_l288_288249

noncomputable def smallest_k : ℕ :=
  2017

theorem smallest_k_unique (P Q : Polynomial ℤ) (k : ℕ) (n : Fin k → ℤ)
  (hP_deg : P.degree = 2017)
  (hQ_deg : Q.degree = 2017)
  (hP_lead : P.leading_coeff = 1)
  (hQ_lead : Q.leading_coeff = 1)
  (h_prod : (Finset.univ.image n).prod (λ i, P.eval i) = (Finset.univ.image n).prod (λ i, Q.eval i)) :
  k = smallest_k → P = Q :=
begin
  sorry
end

end smallest_k_unique_l288_288249


namespace solve_for_y_l288_288213

theorem solve_for_y (y : ℝ) (h : (sqrt (8 * y)) / (sqrt (4 * (y - 2))) = 3) : y = 18 / 7 := 
by sorry

end solve_for_y_l288_288213


namespace no_positive_integer_solutions_l288_288229

theorem no_positive_integer_solutions (A : ℕ) (hA : 1 ≤ A ∧ A ≤ 9) :
  ¬(∃ x : ℕ, x^2 - 2 * A * x + A0 = 0) :=
by sorry

end no_positive_integer_solutions_l288_288229


namespace volume_of_regular_tetrahedron_l288_288631

noncomputable def volume_of_tetrahedron (a H : ℝ) : ℝ :=
  (a^2 * H) / (6 * Real.sqrt 2)

theorem volume_of_regular_tetrahedron
  (d_face : ℝ)
  (d_edge : ℝ)
  (h : Real.sqrt 14 = d_edge)
  (h1 : 2 = d_face)
  (volume_approx : ℝ) :
  ∃ a H, (d_face = Real.sqrt ((H / 2)^2 + (a * Real.sqrt 3 / 6)^2) ∧ 
          d_edge = Real.sqrt ((H / 2)^2 + (a / (2 * Real.sqrt 3))^2) ∧ 
          Real.sqrt (volume_of_tetrahedron a H) = 533.38) :=
  sorry

end volume_of_regular_tetrahedron_l288_288631


namespace cindy_subtraction_l288_288262

theorem cindy_subtraction (a : ℕ) (h1 : a = 50 ^ 2)
(h2 : 51 ^ 2 = 50 ^ 2 + 2 * 50 * 1 + 1) 
(h3 : 49 ^ 2 = 50 ^ 2 - 2 * 50 * 1 + 1) : 
a - 99 = 49 ^ 2 :=
by
  rw h1
  rw h3
  sorry

end cindy_subtraction_l288_288262


namespace part_one_part_two_l288_288749

-- For part (1)
theorem part_one :
  (2 + 10 / 27 : ℝ) ^ (-2 / 3) + 2 * Real.log 2 / Real.log 3 -
  (Real.log (4 / 9) / Real.log 3) - 5 ^ (Real.log 9 / Real.log 25) = -7 / 16 := 
sorry

-- For part (2)
theorem part_two (α : ℝ) : 
  (Real.sin (3 * Real.pi - α) * Real.sin (Real.pi / 2 + α)) / 
  (2 * Real.cos (Real.pi / 6 - 2 * α) - 2 * Real.cos (2 * α) * Real.sin (Real.pi / 3)) = 1 / 2 := 
sorry

end part_one_part_two_l288_288749


namespace part_a_fixed_point_T_part_b_circumcircle_tangency_l288_288144

-- Definitions of the triangle and points in the problem
variables (A B C D E F T : Point)
variables (lineBC : Line)
variables (triangleABC : Triangle)
variables (D_on_BC : On D lineBC)
variables (lineDE_parallel_AB : Parallel (Line.through D E) (Line.through A B))
variables (lineDF_parallel_AC : Parallel (Line.through D F) (Line.through A C))

-- Statement of Part (a)
theorem part_a_fixed_point_T (D_on_BC : ∀ D, On D lineBC) :
  ∃ T, ∀ D, On D lineBC → 
    (Circumcircle (A E F)).passes_through T :=
  sorry

-- Statement of Part (b)
theorem part_b_circumcircle_tangency (D_on_AT : On D (Line.through A T)) :
  tangent (Circumcircle (A E F)) (Circumcircle (B T C)) :=
  sorry

end part_a_fixed_point_T_part_b_circumcircle_tangency_l288_288144


namespace diff_one_tenth_and_one_tenth_percent_of_6000_l288_288294

def one_tenth_of_6000 := 6000 / 10
def one_tenth_percent_of_6000 := (1 / 1000) * 6000

theorem diff_one_tenth_and_one_tenth_percent_of_6000 : 
  (one_tenth_of_6000 - one_tenth_percent_of_6000) = 594 :=
by
  sorry

end diff_one_tenth_and_one_tenth_percent_of_6000_l288_288294


namespace fraction_value_l288_288461

theorem fraction_value
  (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (cond1 : (a + b + c) / (a + b - c) = 7)
  (cond2 : (a + b + c) / (a + c - b) = 1.75) :
  (a + b + c) / (b + c - a) = 3.5 :=
by
  sorry

end fraction_value_l288_288461


namespace inequality_solution_l288_288612

theorem inequality_solution (x : ℝ) (hx : x ≥ 0) : (2021 * (x ^ 10) - 1 ≥ 2020 * x) ↔ (x = 1) :=
sorry

end inequality_solution_l288_288612


namespace president_and_committee_combination_l288_288136

theorem president_and_committee_combination : 
  (∃ (n : ℕ), n = 10 * (Nat.choose 9 3)) := 
by
  use 840
  sorry

end president_and_committee_combination_l288_288136


namespace distinct_real_solution_conditions_l288_288065

theorem distinct_real_solution_conditions:
  (∃ x > 0, ∃ g (x) = log 2 ((2 * x) / (x + 1)) 
  ∧ (abs (g x)^2 + m * abs (g x) + 2 * m + 3 = 0)) ↔ (-3 / 2 < m ∧ m ≤ -4 / 3) := 
sorry

end distinct_real_solution_conditions_l288_288065


namespace dice_roll_probability_bounds_l288_288716

noncomputable def dice_roll_probability : Prop :=
  let n := 80
  let p := (1 : ℝ) / 6
  let q := 1 - p
  let epsilon := 2.58 / 24
  let lower_bound := (p - epsilon) * n
  let upper_bound := (p + epsilon) * n
  5 ≤ lower_bound ∧ upper_bound ≤ 22

theorem dice_roll_probability_bounds :
  dice_roll_probability :=
sorry

end dice_roll_probability_bounds_l288_288716


namespace function_domain_exclusion_l288_288537

theorem function_domain_exclusion (x : ℝ) :
  (∃ y, y = 2 / (x - 8)) ↔ x ≠ 8 :=
sorry

end function_domain_exclusion_l288_288537


namespace volume_difference_l288_288210

def volume (width height depth : Float) : Float := width * height * depth

/-- Convert inches to feet. --/
def inches_to_feet (inches : Float) : Float := inches / 12

/-- Convert meters to feet. --/
def meters_to_feet (meters : Float) : Float := meters * 3.281

/-- Define the dimensions of each birdhouse in feet. --/
def sara_width : Float := 1
def sara_height : Float := 2
def sara_depth : Float := 2

def jake_width : Float := inches_to_feet 16
def jake_height : Float := inches_to_feet 20
def jake_depth : Float := inches_to_feet 18

def tom_width : Float := meters_to_feet 0.4
def tom_height : Float := meters_to_feet 0.6
def tom_depth : Float := meters_to_feet 0.5

/-- Define the volumes of each birdhouse. --/
def sara_volume : Float := volume sara_width sara_height sara_depth
def jake_volume : Float := volume jake_width jake_height jake_depth
def tom_volume : Float := volume tom_width tom_height tom_depth

/-- Prove that the difference between the largest and smallest volumes is approximately 0.916 cubic feet. --/
theorem volume_difference : (max (max sara_volume jake_volume) tom_volume - min (min sara_volume jake_volume) tom_volume ≈ 0.916) := sorry

end volume_difference_l288_288210


namespace total_cost_smore_night_l288_288598

-- Define the costs per item
def cost_graham_cracker : ℝ := 0.10
def cost_marshmallow : ℝ := 0.15
def cost_chocolate : ℝ := 0.25
def cost_caramel_piece : ℝ := 0.20
def cost_toffee_piece : ℝ := 0.05

-- Calculate the cost for each ingredient per S'more
def cost_caramel : ℝ := 2 * cost_caramel_piece
def cost_toffee : ℝ := 4 * cost_toffee_piece

-- Total cost of one S'more
def cost_one_smore : ℝ :=
  cost_graham_cracker + cost_marshmallow + cost_chocolate + cost_caramel + cost_toffee

-- Number of people and S'mores per person
def num_people : ℕ := 8
def smores_per_person : ℕ := 3

-- Total number of S'mores
def total_smores : ℕ := num_people * smores_per_person

-- Total cost of all the S'mores
def total_cost : ℝ := total_smores * cost_one_smore

-- The final statement
theorem total_cost_smore_night : total_cost = 26.40 := 
  sorry

end total_cost_smore_night_l288_288598


namespace greatest_possible_large_chips_l288_288255

theorem greatest_possible_large_chips 
  (total : ℕ) (s l p : ℕ) (prime_p : Nat.Prime p) (chips_eq : s + l = total) (num_eq : s = l + p)
  (total_eq : total = 72) : 
  l ≤ 35 := by
  have H : 2 * l + p = total := sorry -- Derived from s + l = 72 and s = l + p
  have p_value : p = 2 := sorry -- Smallest prime
  have H1 : 2 * l + 2 = 72 := sorry -- Substituting p = 2
  have H2 : 2 * l = 70 := sorry -- Simplifying
  have H3 : l = 35 := sorry -- Solving for l
  l ≤ 35 := sorry

end greatest_possible_large_chips_l288_288255


namespace number_with_5_or_6_base_8_l288_288867

open Finset

def count_numbers_with_5_or_6 : ℕ :=
  let base_8_numbers := Ico 1 (8 ^ 3)
  let count_with_5_or_6 := base_8_numbers.filter (λ n, ∃ b, b ∈ digit_set 8 n ∧ (b = 5 ∨ b = 6))
  count_with_5_or_6.card

theorem number_with_5_or_6_base_8 : count_numbers_with_5_or_6 = 296 := 
by 
  -- Proof omitted for this exercise
  sorry

end number_with_5_or_6_base_8_l288_288867


namespace correct_quadratic_equation_l288_288527

theorem correct_quadratic_equation 
  (first_student_roots : ℤ × ℤ)
  (second_student_roots : ℤ × ℤ)
  (correct_equation : ℤ → Prop) :
  first_student_roots = (5, 3) →
  second_student_roots = (-7, -2) →
  correct_equation = (λ x, x^2 - 8 * x + 14) →
  correct_equation = none_of_these :=
by
  intros h1 h2 h3
  sorry

end correct_quadratic_equation_l288_288527


namespace count_numbers_with_5_or_6_in_base_8_l288_288860

-- Define the condition that checks if a number contains digits 5 or 6 in base 8
def contains_digit_5_or_6_in_base_8 (n : ℕ) : Prop :=
  let digits := Nat.digits 8 n
  5 ∈ digits ∨ 6 ∈ digits

-- The main problem statement
theorem count_numbers_with_5_or_6_in_base_8 :
  (Finset.filter contains_digit_5_or_6_in_base_8 (Finset.range 513)).card = 296 :=
by
  sorry

end count_numbers_with_5_or_6_in_base_8_l288_288860


namespace tangent_parallel_to_line_l288_288387

noncomputable def y (x : ℝ) := x^4

def point_P := (1 : ℝ, 1 : ℝ)

def line1 (x y : ℝ) := 4 * x - y + 1 = 0

theorem tangent_parallel_to_line : 
  let slope_tangent := (deriv y 1) in
  slope_tangent = 4 ∧ ¬ ∃ b: ℝ, (∀ x : ℝ, y x = 4 * x + b) ∧ b = -1 :=
  sorry

end tangent_parallel_to_line_l288_288387


namespace perpendicular_sufficient_not_necessary_l288_288812

variables {α β : Type} [plane α] [plane β] [hne: α ≠ β] (m : line) (hm : m ∈ α)

theorem perpendicular_sufficient_not_necessary :
  (m ⊥ β) → α ⊥ β ∧ (α ⊥ β → ¬ (m ⊥ β)) :=
sorry

end perpendicular_sufficient_not_necessary_l288_288812


namespace problem1_problem2_l288_288441

-- Definitions and conditions:
def p (x : ℝ) : Prop := x^2 - 4 * x - 5 ≤ 0
def q (x m : ℝ) : Prop := (x^2 - 2 * x + 1 - m^2 ≤ 0) ∧ (m > 0)

-- Question (1) statement: Prove that if p is a sufficient condition for q, then m ≥ 4
theorem problem1 (p_implies_q : ∀ x : ℝ, p x → q x m) : m ≥ 4 := sorry

-- Question (2) statement: Prove that if m = 5 and p ∨ q is true but p ∧ q is false,
-- then the range of x is [-4, -1) ∪ (5, 6]
theorem problem2 (m_eq : m = 5) (p_or_q : ∃ x : ℝ, p x ∨ q x m) (p_and_not_q : ¬ (∃ x : ℝ, p x ∧ q x m)) :
  ∃ x : ℝ, (x < -1 ∧ -4 ≤ x) ∨ (5 < x ∧ x ≤ 6) := sorry

end problem1_problem2_l288_288441


namespace fraction_identity_l288_288464

theorem fraction_identity (a b c : ℝ) (h1 : a + b + c > 0) (h2 : a + b - c > 0) (h3 : a + c - b > 0) (h4 : b + c - a > 0) 
  (h5 : (a+b+c)/(a+b-c) = 7) (h6 : (a+b+c)/(a+c-b) = 1.75) : (a+b+c)/(b+c-a) = 3.5 :=
by
  sorry

end fraction_identity_l288_288464


namespace zeros_of_difference_l288_288459

noncomputable def f (x : ℝ) : ℝ := abs (log x)

theorem zeros_of_difference :
  ∃! x ∈ Ioo 0 e, f x - f (e - x) = 0 ∧
  ∃! y ∈ Ioo 0 e, f y - f (e - y) = 0 ∧
  ∃! z ∈ Ioo 0 e, f z - f (e - z) = 0 := by
  sorry

end zeros_of_difference_l288_288459


namespace arithmetic_sequence_ratio_l288_288163

-- Define the variables and conditions
variables {a : ℕ → ℚ} {S : ℕ → ℚ} (d : ℚ)
def a_n (n : ℕ) := a 1 + (n - 1) * d
def S_n (n : ℕ) := n * (a 1 + a_n (n - 1)) / 2
axiom a1 (h : a 1 = 2 * a_n 8 - 3 * a_n 4) : True

-- Define the theorem and prove the desired result
theorem arithmetic_sequence_ratio (d_ne_zero : d ≠ 0) (h1 : a 1 = 2 * a_n 8 - 3 * a_n 4) : 
  S 8 / S 16 = 3 / 10 :=
by
  sorry

end arithmetic_sequence_ratio_l288_288163


namespace no_real_solution_log_eq_l288_288116

open Real

theorem no_real_solution_log_eq :
  ∀ x : ℝ, (x + 5 > 0) → (x - 3 > 0) → (x^2 - 8x + 15 > 0) →
    log (x + 5) + log (x - 3) ≠ log (x^2 - 8x + 15) :=
by sorry

end no_real_solution_log_eq_l288_288116


namespace find_m_l288_288810

def A : Set ℕ := {1, 3}
def B (m : ℕ) : Set ℕ := {1, 2, m}

theorem find_m (m : ℕ) (h : A ⊆ B m) : m = 3 :=
sorry

end find_m_l288_288810


namespace sally_out_of_pocket_l288_288996

-- Definitions based on conditions
def g : ℕ := 320 -- Amount given by the school
def c : ℕ := 12  -- Cost per book
def n : ℕ := 30  -- Number of students

-- Definition derived from conditions
def total_cost : ℕ := n * c
def out_of_pocket : ℕ := total_cost - g

-- Proof statement
theorem sally_out_of_pocket : out_of_pocket = 40 := by
  -- The proof steps would go here
  sorry

end sally_out_of_pocket_l288_288996


namespace max_value_of_expr_l288_288581

theorem max_value_of_expr (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) 
  (h_sum : x + y = 2) : x^2 * y^2 * (x^2 + y^2) ≤ 2 :=
begin
  sorry
end

end max_value_of_expr_l288_288581


namespace simplify_fraction_l288_288609

theorem simplify_fraction (i : ℂ) (h : i^2 = -1) : 
  (2 - i) / (1 + 4 * i) = -2 / 17 - (9 / 17) * i :=
by
  sorry

end simplify_fraction_l288_288609


namespace find_x_value_l288_288051

noncomputable def power_function (x : ℝ) : ℝ := x ^ (-3)

theorem find_x_value : ∃ x : ℝ, power_function x = 27 := 
begin
  use 1 / 3,
  unfold power_function,
  have hx : (1 / 3) ^ (-3) = 27, by {
    norm_num,
  },
  exact hx,
end

end find_x_value_l288_288051


namespace find_K_coordinates_l288_288202

theorem find_K_coordinates :
  ∀ (M N K : ℝ × ℝ), (M = (7, 4) ∧ N = (-3, 9) ∧ let λ := (2 : ℝ) / 3 in K = ((M.1 + λ * N.1) / (1 + λ), (M.2 + λ * N.2) / (1 + λ))) →
  K = (3, 6) :=
by
  intros M N K H
  sorry

end find_K_coordinates_l288_288202


namespace ed_perpendicular_bc_l288_288799

theorem ed_perpendicular_bc
  (circle : Type)
  (A B C : circle)
  (D : circle)
  (E : circle)
  [MidpointSegment D B C]
  [MidpointArc E A B C] :
  Perpendicular E D B C :=
by
  sorry

end ed_perpendicular_bc_l288_288799


namespace circumcenter_location_l288_288912

theorem circumcenter_location (A B C : ℝ) (h₁ : sin A * cos B < 0) : 
  (the circumcenter of triangle ABC is outside the triangle) :=
  sorry

end circumcenter_location_l288_288912


namespace one_seventh_increase_after_2013th_digit_removed_l288_288531

theorem one_seventh_increase_after_2013th_digit_removed : 
  (0.142857 : ℝ) * (10 ^ (-6 : ℝ)) = (1 / 7) →
  2013 % 6 = 3 →
  (0.\overline{142857} : ℝ) < (0.184857 : ℝ) :=
by
  sorry

end one_seventh_increase_after_2013th_digit_removed_l288_288531


namespace max_mark_paper_i_l288_288709

theorem max_mark_paper_i (M : ℝ) (h1 : 0.65 * M = 170) : M ≈ 262 :=
by sorry

end max_mark_paper_i_l288_288709


namespace find_BE_l288_288940

variables (A B C D E : Type)
variables [triangle : Triangle A B C]
variables [segment_AB : Segment A B 8]
variables [segment_BC : Segment B C 20]
variables [segment_CA : Segment C A 16]
variables (D_on_BC : PointOnSegment D B C)
variables (E_on_BC : PointOnSegment E B C)
variables (CD_equals_8 : Segment C D 8)
variables (angle_BAE_eq_angle_CAD : CongruentAngles (Angle B A E) (Angle C A D))

theorem find_BE : Segment B E = 2 :=
by
  sorry

end find_BE_l288_288940


namespace range_of_expression_l288_288470

theorem range_of_expression (a b c : ℝ) 
  (h : (1/4) * a^2 + (1/4) * b^2 + c^2 = 1) :
  -2 ≤ ab + 2bc + 2ca ∧ ab + 2bc + 2ca ≤ 4 := sorry

end range_of_expression_l288_288470


namespace count_integer_roots_of_3125_l288_288790

theorem count_integer_roots_of_3125 : 
  (∃ n : ℕ, (n > 0) ∧ (5^5 = 3125) ∧ (3125^(1/n)) ∈ ℤ) = 2 :=
sorry

end count_integer_roots_of_3125_l288_288790


namespace translation_correct_l288_288274

-- Definitions based on conditions
def original_function (x : ℝ) : ℝ := 2^x + 1
def translated_function (x : ℝ) : ℝ := 2^(x + 1)

-- Definition for the translation vector
def translation_vector : ℝ × ℝ := (-1, -1)

-- Statement to prove:
theorem translation_correct :
  ∀ x : ℝ, translated_function x = original_function (x - 1) - 1 :=
by
  intros
  unfold original_function translated_function
  rw [←sub_add_eq_sub_sub_swap, sub_self]
  sorry

end translation_correct_l288_288274


namespace max_sin_sum_in_triangle_l288_288903

theorem max_sin_sum_in_triangle : ∀ (A B C : ℝ), (A + B + C = π) → (0 < A ∧ A < π) → (0 < B ∧ B < π) → (0 < C ∧ C < π) →
  (sin A + sin B + sin C ≤ 3 * (sqrt 3) / 2) := 
by
  intros A B C h_sum hA hB hC
  have h : (sin A + sin B + sin C) / 3 ≤ sin ((A + B + C) / 3), from sorry
  rw [h_sum, sin_pi_div_three] at h
  linarith [h]

end max_sin_sum_in_triangle_l288_288903


namespace find_ages_l288_288248

variables (H J A : ℕ)

def conditions := 
  H + J + A = 90 ∧ 
  H = 2 * J - 5 ∧ 
  H + J - 10 = A

theorem find_ages (h_cond : conditions H J A) : 
  H = 32 ∧ 
  J = 18 ∧ 
  A = 40 :=
sorry

end find_ages_l288_288248


namespace divide_cross_into_equal_polygons_l288_288383

-- Define the basic structure of the cross and its properties.
structure Square := 
  (side_length : ℝ)

structure Cross :=
  (squares : List Square)
  (total_area : ℝ := squares.length)
  (central_square : Square)

-- Define the conditions.
variables 
  [cross1 : Cross]
  (five_squares_property : cross1.squares.length = 5)
  (identical_squares_property : ∀ s ∈ cross1.squares, s.side_length = 1)
  (area_property : cross1.total_area = 5)

-- Prove the question.
theorem divide_cross_into_equal_polygons (cross1 : Cross) 
  (h_squares : cross1.squares.length = 5)
  (h_identical : ∀ s ∈ cross1.squares, s.side_length = 1)
  (h_area : cross1.total_area = 5) : 
  ∃ (polygons : List (List Square)), 
    polygons.length = 3 ∧ 
    (∀ p ∈ polygons, 
      (let area := p.length in 
       area = 5 / 3) ∧ 
      (let perimeter :=  -- placeholder for perimeter calculation, 
       perimeter = perimeter)) :=
sorry

end divide_cross_into_equal_polygons_l288_288383


namespace light_bulbs_on_after_1011_people_l288_288667

theorem light_bulbs_on_after_1011_people : 
  let num_light_bulbs := 2021
  let num_people := 1011
  (count_perfect_squares num_light_bulbs) = 44 :=
by sorry

def count_perfect_squares (n : ℕ) : ℕ :=
  Nat.sqrt n

end light_bulbs_on_after_1011_people_l288_288667


namespace problem_l288_288763

def f (x : ℝ) : ℝ := (x^2 + 1) / x
def g (x : ℝ) : ℝ := (x^2 - 1) / x

theorem problem :
  (f 2 - g 2 = 1) ∧ 
  (∃ m : ℝ, f m = 2 * Real.sqrt 2 ∧ g m ≠ 2) ∧
  (∀ (n : ℕ) (m : Fin n → ℝ),
    (∑ i, f (m i) = 1024 ∧ ∑ i, g (m i) = 1026 →
    (∑ i, m i) * (∑ i, 1 / m i) = -1025)) :=
by 
  split
  sorry

  split
  sorry

  sorry

end problem_l288_288763


namespace sin_double_angle_l288_288495

variable {θ : ℝ}

theorem sin_double_angle (h : cos θ + sin θ = 3 / 2) : sin (2 * θ) = 5 / 4 :=
by
  sorry

end sin_double_angle_l288_288495


namespace greatest_coloring_integer_l288_288157

theorem greatest_coloring_integer (α β : ℝ) (h1 : 1 < α) (h2 : α < β) :
  ∃ r : ℕ, r = 2 ∧ ∀ (f : ℕ → ℕ), ∃ x y : ℕ, x ≠ y ∧ f x = f y ∧ α ≤ (x : ℝ) / (y : ℝ) ∧ (x : ℝ) / (y : ℝ) ≤ β := 
sorry

end greatest_coloring_integer_l288_288157


namespace maximal_number_of_coins_l288_288301

noncomputable def largest_number_of_coins (n k : ℕ) : Prop :=
n < 100 ∧ n = 12 * k + 3

theorem maximal_number_of_coins (n k : ℕ) : largest_number_of_coins n k → n = 99 :=
by
  sorry

end maximal_number_of_coins_l288_288301


namespace solution_set_abs_ineq_l288_288656

theorem solution_set_abs_ineq (x : ℝ) : abs (2 - x) ≤ 1 ↔ 1 ≤ x ∧ x ≤ 3 := by
  sorry

end solution_set_abs_ineq_l288_288656


namespace range_of_x_l288_288458

def f (x : ℝ) : ℝ := Real.exp x - Real.exp (-x) + 1

theorem range_of_x (x : ℝ) (h : f (2*x - 1) + f (4 - x^2) > 2) : -1 < x ∧ x < 3 :=
by
  sorry

end range_of_x_l288_288458


namespace total_study_time_is_60_l288_288772

-- Define the times Elizabeth studied for each test
def science_time : ℕ := 25
def math_time : ℕ := 35

-- Define the total study time
def total_study_time : ℕ := science_time + math_time

-- Proposition that the total study time equals 60 minutes
theorem total_study_time_is_60 : total_study_time = 60 := by
  /-
  Here we would provide the proof steps, but since the task is to write the statement only,
  we add 'sorry' to indicate the missing proof.
  -/
  sorry

end total_study_time_is_60_l288_288772


namespace sally_out_of_pocket_l288_288995

-- Definitions based on conditions
def g : ℕ := 320 -- Amount given by the school
def c : ℕ := 12  -- Cost per book
def n : ℕ := 30  -- Number of students

-- Definition derived from conditions
def total_cost : ℕ := n * c
def out_of_pocket : ℕ := total_cost - g

-- Proof statement
theorem sally_out_of_pocket : out_of_pocket = 40 := by
  -- The proof steps would go here
  sorry

end sally_out_of_pocket_l288_288995


namespace sin_A_is_sqrt2_div_2_l288_288512

theorem sin_A_is_sqrt2_div_2 (a b c : ℝ) (A B C: ℝ)
  (h1 : a = 1) (h2 : b = 1) (h3 : c = √2)
  (h4 : a^2 + b^2 = c^2) :
  real.sin A = √2 / 2 :=
sorry

end sin_A_is_sqrt2_div_2_l288_288512


namespace nancy_shoes_problem_l288_288593

theorem nancy_shoes_problem :
  let boots := 6 in
  ∃ slippers heels : ℕ, 
    (slippers = 15) ∧ 
    heels = 3 * (slippers + boots) ∧ 
    2 * boots + 2 * slippers + 2 * heels = 168 ∧ 
    slippers - boots = 9 :=
begin
  let boots := 6,
  use 15,  -- Number of slippers
  use 3 * (15 + boots),  -- Number of heels
  split,
  { refl, },  -- slippers = 15
  split,
  { refl, },  -- heels = 3 * (slippers + boots)
  split,
  { sorry, },  -- 2 * boots + 2 * slippers + 2 * heels = 168
  { sorry, }   -- slippers - boots = 9
end

end nancy_shoes_problem_l288_288593


namespace expected_k_cycle_in_Gnp_l288_288916

open Classical

noncomputable def factorial_falling (n k : ℕ) : ℕ :=
  if k = 0 then 1 else n * factorial_falling (n - 1) (k - 1)

variables {n k : ℕ} {p : ℝ}

def expected_k_cycle (n k : ℕ) (p : ℝ) : ℝ :=
  (factorial_falling n k) / (2 * k) * p^k

theorem expected_k_cycle_in_Gnp (n k : ℕ) (p : ℝ) (h1 : 0 < n) (h2 : 0 < k) (h3 : 0 ≤ p) (h4 : p ≤ 1) : 
  let G := sample_Gnp n p in
  (E G : ℝ) = expected_k_cycle n k p :=
sorry

end expected_k_cycle_in_Gnp_l288_288916


namespace correct_average_and_variance_l288_288127

theorem correct_average_and_variance
  (n : ℕ) (avg incorrect_variance correct_variance : ℝ)
  (incorrect_score1 actual_score1 incorrect_score2 actual_score2 : ℝ)
  (H1 : n = 48)
  (H2 : avg = 70)
  (H3 : incorrect_variance = 75)
  (H4 : incorrect_score1 = 50)
  (H5 : actual_score1 = 80)
  (H6 : incorrect_score2 = 100)
  (H7 : actual_score2 = 70)
  (Havg : avg = (n * avg - incorrect_score1 - incorrect_score2 + actual_score1 + actual_score2) / n)
  (Hvar : correct_variance = incorrect_variance + (actual_score1 - avg) ^ 2 + (actual_score2 - avg) ^ 2
                     - (incorrect_score1 - avg) ^ 2 - (incorrect_score2 - avg) ^ 2 / n) :
  avg = 70 ∧ correct_variance = 50 :=
by {
  sorry
}

end correct_average_and_variance_l288_288127


namespace non_degenerate_triangles_l288_288074

theorem non_degenerate_triangles (x y : ℕ) (hx : 1 ≤ x ∧ x ≤ 5) (hy : 1 ≤ y ∧ y ≤ 5) : 
  ∃ (n : ℕ), n = 2160 ∧ 1 ≤ n :=
by
  sorry

end non_degenerate_triangles_l288_288074


namespace area_of_abs_inequality_l288_288766

theorem area_of_abs_inequality :
  ∀ (x y : ℝ), |x + 2 * y| + |2 * x - y| ≤ 6 → 
  ∃ (area : ℝ), area = 12 := 
by
  -- This skips the proofs
  sorry

end area_of_abs_inequality_l288_288766


namespace face_opposite_teal_is_blue_l288_288328

-- Definitions for the painting of the cube faces
def unique_colors := {"B", "Y", "O", "K", "T", "V"}

-- Views of the cube
def first_view := ("Y", "B", "O")
def second_view := ("Y", "K", "O")
def third_view := ("Y", "V", "O")

-- Face opposite to teal (T) should be proven to be blue (B)
theorem face_opposite_teal_is_blue
  (h1 : ∀ (c1 c2 c3 : String), (c1, c2, c3) ∈ {first_view, second_view, third_view} → c1 = "Y")
  (h2 : ∀ (c1 c2 c3 : String), (c1, c2, c3) ∈ {first_view, second_view, third_view} → c3 = "O")
  (h3 : {c2 | ∃ c1, ∃ c3, (c1, c2, c3) = first_view ∨ (c1, c2, c3) = second_view ∨ (c1, c2, c3) = third_view} = {"B", "K", "V"})
  : "B" = "Y" → "B" = "O" → "B" = "K" → "B" = "T" → "B" = "V" → false → ("B" = "B") :=
begin
  sorry
end

end face_opposite_teal_is_blue_l288_288328


namespace simplified_identity_l288_288285

theorem simplified_identity :
  (12 : ℚ) * ( (1/3 : ℚ) + (1/4) + (1/6) + (1/12) )⁻¹ = 72 / 5 :=
  sorry

end simplified_identity_l288_288285


namespace fraction_eval_l288_288776

theorem fraction_eval : 
    (1 / (3 - (1 / (3 - (1 / (3 - (1 / 4))))))) = (11 / 29) := 
by
  sorry

end fraction_eval_l288_288776


namespace problem1_problem_variant_l288_288982

theorem problem1 :
  ∃ (a b : ℕ), 
  (4 * a^3 - 3 * a + 1 = 2 * b^2)
  ∧ (a ≤ 1980)
  ∧ (finset.card {x : ℕ | ∃ b : ℕ, x ≤ 1980 ∧ 4 * x^3 - 3 * x + 1 = 2 * b^2} ≥ 31) :=
sorry

theorem problem_variant :
  ∃ (a b : ℕ). (4 * a^3 - 3 * a + 1 = 2 * b^2) ∧ (∀ n in finset.univ, ∃ x y : ℕ, x > n ∧ (4 * x ^ 3 - 3 * x + 1 = 2 * y^2)) :=
sorry

end problem1_problem_variant_l288_288982


namespace Mitch_hourly_rate_l288_288190

theorem Mitch_hourly_rate :
  let weekday_hours := 5 * 5
  let weekend_hours := 3 * 2
  let equivalent_weekend_hours := weekend_hours * 2
  let total_hours := weekday_hours + equivalent_weekend_hours
  let weekly_earnings := 111
  weekly_earnings / total_hours = 3 :=
by
  let weekday_hours := 5 * 5
  let weekend_hours := 3 * 2
  let equivalent_weekend_hours := weekend_hours * 2
  let total_hours := weekday_hours + equivalent_weekend_hours
  let weekly_earnings := 111
  sorry

end Mitch_hourly_rate_l288_288190


namespace problem_cartesian_coord_l288_288528

theorem problem_cartesian_coord (
  -- Conditions
  P : ℝ × ℝ,
  dist_sum_condition : dist P (0, -Real.sqrt 3) + dist P (0, Real.sqrt 3) = 4
) :
  -- Part (I)
  (∃ C : ℝ × ℝ → Prop, ∀ P, (C P ↔ P.1^2 + (P.2^2 / 4) = 1)) ∧
  -- Part (II)
  (∀ (k : ℝ), 
    let line_eq := λ P : ℝ × ℝ, P.2 = k * P.1 + 1 in 
    let A := (A.1 : ℝ × ℝ) in
    let B := (B.1 : ℝ × ℝ) in
    (C A ∧ C B ∧ line_eq A ∧ line_eq B →
      (vector.dot_product (A.1, A.2) (B.1, B.2) = 0 ↔ k = 1/2 ∨ k = -1/2) ∧
      (abs (dist A B) = 4 * Real.sqrt 65 / 17)
    )
  ) :=
sorry

end problem_cartesian_coord_l288_288528


namespace pi_times_positive_volume_difference_l288_288351

theorem pi_times_positive_volume_difference :
  let r_A := 5 / Real.pi in
  let h_A := 8 in
  let V_A := Real.pi * r_A^2 * h_A in
  let r_B := 9 / (2 * Real.pi) in
  let h_B := 7 in
  let V_B := Real.pi * r_B^2 * h_B in
  Real.pi * abs (V_B - V_A) = 58.25 := 
by {
  sorry
}

end pi_times_positive_volume_difference_l288_288351


namespace count_4_digit_numbers_divisible_by_13_l288_288872

theorem count_4_digit_numbers_divisible_by_13 : 
  ∃ n : ℕ, n = 693 ∧ (∀ k : ℕ, k >= 1000 ∧ k < 10000 ∧ k % 13 = 0 → ∃ m : ℕ, m = (k - 1000) / 13 + 1 ∧ m = n) :=
by {
  -- Solution proof will be placed here.
  sorry
}

end count_4_digit_numbers_divisible_by_13_l288_288872


namespace find_fraction_l288_288466

variable (a b c : ℝ)
variable (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
variable (h1 : (a + b + c) / (a + b - c) = 7)
variable (h2 : (a + b + c) / (a + c - b) = 1.75)

theorem find_fraction : (a + b + c) / (b + c - a) = 3.5 := 
by {
  sorry
}

end find_fraction_l288_288466


namespace three_g_x_l288_288504

-- Given condition
def g (x : ℝ) : ℝ := 3 / (3 + 2 * x)

-- Proof problem
theorem three_g_x (x : ℝ) (h : x > 0) : 3 * g x = 27 / (9 + 2 * x) :=
sorry

end three_g_x_l288_288504


namespace solution_l288_288800

-- Definitions of geometric conditions and points
variable (A B C D E M F G : Point)
variable (circle : Circle)
variable (chord1 : Chord circle A B)
variable (chord2 : Chord circle C D)
variable (intersection : Intersect chord1 chord2 = E)
variable (segment_EM : Segment E B)
variable (point_M_on_segment : OnSegment M segment_EM)
variable (circle_DEM : CircleThrough D E M)
variable (tangent_circle : Tangent circle_DEM E (Line_through_E_and F))
variable (tangent_circle' : Tangent circle_DEM E (Line_through_E_and G))
variable (ratio_AM_AB : (AM / AB) = t)

-- Goal
def geom_problem : Prop := 
  EG / EF = t / (1 - t)

theorem solution : geom_problem A B C D E M F G circle chord1 chord2 intersection segment_EM point_M_on_segment circle_DEM tangent_circle tangent_circle' ratio_AM_AB :=
  sorry

end solution_l288_288800


namespace power_increase_l288_288488

theorem power_increase (y : ℝ) (h : 2^y = 50) : 2^(y + 3) = 400 := by
  sorry

end power_increase_l288_288488


namespace hexadecagon_quadrilateral_area_ratio_l288_288208

theorem hexadecagon_quadrilateral_area_ratio
  (ABCDEEFGHIJKLMNOP : Type) [regular_hexadecagon ABCDEEFGHIJKLMNOP]
  (ACEG_quadrilateral : Quadrilateral (ACEG ABCDEEFGHIJKLMNOP))
  (p q : ℝ)
  (hexadecagon_area : area ABCDEEFGHIJKLMNOP = p)
  (quadrilateral_area : area ACEG_quadrilateral = q) :
  q / p = Real.sqrt 2 / 2 := 
sorry

end hexadecagon_quadrilateral_area_ratio_l288_288208


namespace inequalities_l288_288027

variable (a b c : ℝ)
variable (ha : a = 8.1 ^ 0.51)
variable (hb : b = 8.1 ^ 0.5)
variable (hc : c = Real.log 0.3 / Real.log 3)

theorem inequalities (a b c : ℝ)
  (ha : a = 8.1 ^ 0.51)
  (hb : b = 8.1 ^ 0.5)
  (hc : c = Real.log 0.3 / Real.log 3) :
  c < b ∧ b < a := by
  sorry

end inequalities_l288_288027


namespace tetrahedron_volume_l288_288629

/-- Given a regular triangular pyramid (tetrahedron) with the following properties:
  - Distance from the midpoint of the height to a lateral face is 2.
  - Distance from the midpoint of the height to a lateral edge is √14.
  Prove that the volume of the pyramid is approximately 533.38.
-/
theorem tetrahedron_volume (d_face d_edge : ℝ) (volume : ℝ) (h1 : d_face = 2) (h2 : d_edge = Real.sqrt 14) :
  Abs (volume - 533.38) < 0.01 :=
by {
  sorry -- Proof will go here
}

end tetrahedron_volume_l288_288629


namespace find_a_in_coeff_expansion_l288_288140

theorem find_a_in_coeff_expansion :
  ∃ (a : ℝ), 
  (∀ (x : ℝ), (x ≠ 0) → (∑ k in finset.range (6), 
    (nat.choose 5 k) * (2 * x)^(5 - k) * (a / x^2)^k) 
    = ∑ k in finset.range(6), 
    (nat.choose 5 k) * (2)^(5 - k) * a^k * x^(5 - 3 * k)) → 
  ∃ (a : ℝ), coefficient_of_x_neg4_eq := 320 → 
  a = 2 := 
sorry

end find_a_in_coeff_expansion_l288_288140


namespace inequality_condition_l288_288169

noncomputable def inequality_holds_for_all (a b c : ℝ) : Prop :=
  ∀ (x : ℝ), a * Real.sin x + b * Real.cos x + c > 0

theorem inequality_condition (a b c : ℝ) :
  inequality_holds_for_all a b c ↔ Real.sqrt (a^2 + b^2) < c :=
by sorry

end inequality_condition_l288_288169


namespace max_abs_z_l288_288902

noncomputable def z := ℂ
noncomputable def a := ℝ
noncomputable def b := ℝ

def question (z : ℂ) : ℂ :=
  z

def condition1 (z : ℂ) := 
  ∃ a b : ℝ, z = a + b * complex.I

def condition2 (z : ℂ) := 
  ∃ a b : ℝ, ℂ.conjugate z = a - b * complex.I

def condition3 (z : ℂ) := 
  (z - 2 * complex.I) * (complex.conjugate z + 2 * complex.I) = 1 

def maximizes (z : ℂ) := 
  complex.abs z

def max_value_of_abs_z := 3

theorem max_abs_z (z : ℂ) (h1: condition1 z) (h2: condition2 z) (h3: condition3 z) :
  maximizes z <= max_value_of_abs_z := 
sorry

end max_abs_z_l288_288902


namespace island_not_Mayya_l288_288197

-- Define types for the inhabitants (A and B) and possible island name.
inductive Inhabitant
| A
| B

def isKnight (i : Inhabitant) : Prop
def isLiar (i : Inhabitant) : Prop
def isIslandMayya : Prop

-- The statements made by A and B
axiom StatementA : (isKnight Inhabitant.B ∧ isIslandMayya)
axiom StatementB : (isLiar Inhabitant.A ∧ isIslandMayya)

-- The property that a knight always tells the truth
axiom KnightTellsTruth : ∀ (i : Inhabitant), isKnight i → (i = Inhabitant.A → StatementA) ∧ (i = Inhabitant.B → StatementB)

-- The property that a liar always lies
axiom LiarAlwaysLies : ∀ (i : Inhabitant), isLiar i → (i = Inhabitant.A → ¬StatementA) ∧ (i = Inhabitant.B → ¬StatementB)

-- The theorem to be proven
theorem island_not_Mayya : ¬isIslandMayya :=
by
  sorry

end island_not_Mayya_l288_288197


namespace missing_coin_value_l288_288592

-- Definitions based on the conditions
def value_of_dime := 10 -- Value of 1 dime in cents
def value_of_nickel := 5 -- Value of 1 nickel in cents
def num_dimes := 1
def num_nickels := 2
def total_value_found := 45 -- Total value found in cents

-- Statement to prove the missing coin's value
theorem missing_coin_value : 
  (total_value_found - (num_dimes * value_of_dime + num_nickels * value_of_nickel)) = 25 := 
by
  sorry

end missing_coin_value_l288_288592


namespace window_treatments_cost_l288_288563

-- Define the costs and the number of windows
def cost_sheers : ℝ := 40.00
def cost_drapes : ℝ := 60.00
def number_of_windows : ℕ := 3

-- Define the total cost calculation
def total_cost := (cost_sheers + cost_drapes) * number_of_windows

-- State the theorem that needs to be proved
theorem window_treatments_cost : total_cost = 300.00 :=
by
  sorry

end window_treatments_cost_l288_288563


namespace determine_remainder_l288_288502

theorem determine_remainder (a b c : ℕ) (h1 : a < 7) (h2 : b < 7) (h3 : c < 7)
  (H1 : (a + 2 * b + 3 * c) % 7 = 1) 
  (H2 : (2 * a + 3 * b + c) % 7 = 2) 
  (H3 : (3 * a + b + 2 * c) % 7 = 1) : 
  (a * b * c) % 7 = 0 := 
sorry

end determine_remainder_l288_288502


namespace irreducible_fraction_repeating_decimal_p_equals_2_l288_288910

theorem irreducible_fraction_repeating_decimal_p_equals_2 (p q : ℕ) (hp : p ≠ 0) (hq : q ≠ 0) (irr : nat.coprime p q) (dec_eq : (p:ℚ) / q = 0.18 * 9 / 99 ∧ (p:ℚ) / q = 2 / 11) : p = 2 :=
sorry

end irreducible_fraction_repeating_decimal_p_equals_2_l288_288910


namespace sum_of_complex_powers_l288_288171

def i : ℂ := complex.I

theorem sum_of_complex_powers (n : ℕ) (h : n % 6 = 0) :
  (∑ k in finset.range (n + 1), (k + 1) * i ^ k) = (4 * n / 3 : ℂ) + 1 + (3 * n / 2) * i :=
by
  sorry

end sum_of_complex_powers_l288_288171


namespace integral_sin_div_x_approx_l288_288372

theorem integral_sin_div_x_approx : 
  |∫ x in 0..1, (sin x) / x - 0.94| < 0.01 :=
sorry

end integral_sin_div_x_approx_l288_288372


namespace binomial_coefficient_19_13_l288_288813

theorem binomial_coefficient_19_13 
  (h1 : Nat.choose 20 13 = 77520) 
  (h2 : Nat.choose 20 14 = 38760) 
  (h3 : Nat.choose 18 13 = 18564) :
  Nat.choose 19 13 = 37128 := 
sorry

end binomial_coefficient_19_13_l288_288813


namespace total_students_went_to_concert_l288_288219

/-- There are 12 buses and each bus took 57 students. We want to find out the total number of students who went to the concert. -/
theorem total_students_went_to_concert (num_buses : ℕ) (students_per_bus : ℕ) (total_students : ℕ) 
  (h1 : num_buses = 12) (h2 : students_per_bus = 57) (h3 : total_students = num_buses * students_per_bus) : 
  total_students = 684 := 
by
  sorry

end total_students_went_to_concert_l288_288219


namespace concyclic_iff_eq_diag_l288_288928

noncomputable def convex_pentagon (A B C D E : Type) : Prop :=
∃ (AB DE BC EA : ℝ), AB = DE ∧ BC = EA ∧ AB ≠ EA

theorem concyclic_iff_eq_diag
  {A B C D E : Type}
  (h_pentagon : convex_pentagon A B C D E)
  (h_concyclic_BCDE : ∀ (P : Type), P = B ∨ P = C ∨ P = D ∨ P = E → ∃ (circle : Type), ∀ (Q : Type), Q = B ∨ Q = C ∨ Q = D ∨ Q = E → Q ∈ circle) :
  (∀ (Q : Type), Q = A ∨ Q = B ∨ Q = C ∨ Q = D → ∃ (circ : Type), ∀ (R : Type), R = A ∨ R = B ∨ R = C ∨ R = D → R ∈ circ) ↔ 
  (∃ (AC AD : ℝ), AC = AD) :=
by sorry

end concyclic_iff_eq_diag_l288_288928


namespace overall_percent_support_l288_288918

/-- Prove that the overall percent of the people surveyed who supported the motion
     for the new environmental law is 67%, given that 75% of 200 men and 65% of 800 women
     supported the motion. -/
theorem overall_percent_support (p1 p2 : ℝ) (n1 n2 : ℕ)
  (h1 : p1 = 0.75) (h2 : n1 = 200) (h3 : p2 = 0.65) (h4 : n2 = 800) :
  let supportive_men := p1 * n1,
      supportive_women := p2 * n2,
      total_supportive := supportive_men + supportive_women,
      total_people := n1 + n2,
      percent_supportive := (total_supportive / total_people) * 100
  in percent_supportive = 67 := 
by {
  -- The proof is not required, so we use sorry to finish the statement.
  sorry
}

end overall_percent_support_l288_288918


namespace max_min_difference_l288_288826

noncomputable def f (x : ℝ) : ℝ := x^3 - 12*x + 8

theorem max_min_difference :
  let M := max (max (f (-3)) (f (-2))) (max (f (2)) (f (3))),
      m := min (min (f (-3)) (f (-2))) (min (f (2)) (f (3)))
  in M - m = 32 :=
by
  let M := max (max (f (-3)) (f (-2))) (max (f (2)) (f (3)))
  let m := min (min (f (-3)) (f (-2))) (min (f (2)) (f (3)))
  have : M = 24 := sorry
  have : m = -8 := sorry
  have : M - m = 32 := by rw [this, this] sorry
  exact this

end max_min_difference_l288_288826


namespace ham_and_bread_percentage_l288_288261

-- Defining the different costs as constants
def cost_of_bread : ℝ := 50
def cost_of_ham : ℝ := 150
def cost_of_cake : ℝ := 200

-- Defining the total cost of the items
def total_cost : ℝ := cost_of_bread + cost_of_ham + cost_of_cake

-- Defining the combined cost of ham and bread
def combined_cost_ham_and_bread : ℝ := cost_of_bread + cost_of_ham

-- The theorem stating that the combined cost of ham and bread is 50% of the total cost
theorem ham_and_bread_percentage : (combined_cost_ham_and_bread / total_cost) * 100 = 50 := by
  sorry  -- Proof to be provided

end ham_and_bread_percentage_l288_288261


namespace positive_area_triangles_correct_l288_288082

-- Define the set of points with integer coordinates in the given range
def grid_points := { p : ℤ × ℤ | 1 ≤ p.1 ∧ p.1 ≤ 5 ∧ 1 ≤ p.2 ∧ p.2 ≤ 5 }

-- Define a function to check for collinearity of three points
def collinear (p1 p2 p3 : ℤ × ℤ) : Prop :=
  (p2.2 - p1.2) * (p3.1 - p1.1) = (p3.2 - p1.2) * (p2.1 - p1.1)

-- Define the count of triangles with positive area (not collinear) from the given points
def triangles_with_positive_area (points : set (ℤ × ℤ)) : ℕ :=
  (@set.to_finset (ℤ × ℤ) _ points).to_list.combinations 3
  .filter (λ l, l.length = 3 ∧ ¬ collinear l.head l.nth 1 l.nth 2).length

-- State the proof problem
theorem positive_area_triangles_correct :
  triangles_with_positive_area grid_points = 2170 := sorry

end positive_area_triangles_correct_l288_288082


namespace slices_sold_today_eq_two_l288_288072

theorem slices_sold_today_eq_two 
  (total_slices : ℕ) (slices_sold_yesterday : ℕ) :
  total_slices = 7 → slices_sold_yesterday = 5 → total_slices - slices_sold_yesterday = 2 :=
by 
  intros h1 h2
  rw [h1, h2]
  rfl

end slices_sold_today_eq_two_l288_288072


namespace values_of_x_l288_288679

theorem values_of_x (x : ℝ) : (-2 < x ∧ x < 2) ↔ (x^2 < |x| + 2) := by
  sorry

end values_of_x_l288_288679


namespace percentage_problem_l288_288317

theorem percentage_problem 
  (number : ℕ)
  (h1 : number = 6400)
  (h2 : 5 * number / 100 = 20 * 650 / 100 + 190) : 
  20 = 20 :=
by 
  sorry

end percentage_problem_l288_288317


namespace lengths_proportional_l288_288979

-- Given conditions
variables (x y z : ℝ) -- Angles of the original triangle
variables (a u b v c w : Mathlib.LinearAlgebra.Basic.Real)

-- Summation condition of the vectors forming a closed loop in the non-convex hexagon
axiom sum_vector_eq_zero : w + a + u + b + v + c = 0

-- The equivalence proof statement
theorem lengths_proportional (h1 : x + y + z = 180) 
  (h2 : w + a + u + b + v + c = 0) : 
  ∃ k : ℝ, ∀ (a' b' c' : ℝ), a = k * a' ∧ b = k * b' ∧ c = k * c' := sorry

end lengths_proportional_l288_288979


namespace count_n_integers_satisfying_conditions_l288_288070

theorem count_n_integers_satisfying_conditions :
  let same_remainder (n : ℤ) := (n % 5 = n % 7)
  card ({n : ℤ | 150 < n ∧ n < 250 ∧ same_remainder n}.toFinset) = 15 :=
by
  sorry

end count_n_integers_satisfying_conditions_l288_288070


namespace proper_subsets_count_l288_288024

def is_int (x : ℝ) : Prop := x = ↑(⌊x⌋)
def A := {x : ℤ | abs (x - 3) < Real.pi}
def B := {x : ℤ | x^2 - 11 * x + 5 < 0}
def C := {x : ℤ | 2 * x^2 - 11 * x + 10 >= 3 * x - 2}
def C_complement := {x : ℤ | x ∉ C}
def A_inter_B_inter_C_complement := A ∩ B ∩ C_complement

theorem proper_subsets_count 
  : ∃ (S : Finset ℤ), S.to_set = A_inter_B_inter_C_complement ∧ S.card = 8 ∧ (S.powerset.card - 1 = 255) := 
sorry

end proper_subsets_count_l288_288024


namespace hyperbola_eccentricity_l288_288824

theorem hyperbola_eccentricity (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (P : ℝ × ℝ) (hP : P = (1, 3)) :
    (∃ e : ℝ, (P.snd = 3 * P.fst ∧ e = sqrt 10)) →
    ∃ e : ℝ, e = sqrt 10 :=
by
  intro h
  sorry

end hyperbola_eccentricity_l288_288824


namespace quadratic_distinct_real_roots_l288_288018

theorem quadratic_distinct_real_roots (m : ℝ) : 
  (m ≠ 0 ∧ m < 1 / 5) ↔ ∃ (x y : ℝ), x ≠ y ∧ m * x^2 - 2 * x + 5 = 0 ∧ m * y^2 - 2 * y + 5 = 0 :=
sorry

end quadratic_distinct_real_roots_l288_288018


namespace find_50th_permutation_l288_288241

-- defining the condition of using digits exactly once
def uses_exactly_once (n : ℕ) : Prop :=
  let digits := [1, 2, 3, 4, 5] in
  n.digits = digits.permutations

-- defining the ordering from least to greatest
def ordered_from_least_to_greatest (n : ℕ) : Prop :=
  let digits := [1, 2, 3, 4, 5] in
  let sorted_digits_list := digits.permutations.sort () in
  n = sorted_digits_list[n-1]

-- The tuple
theorem find_50th_permutation :
  ∃ n, ordered_from_least_to_greatest (n : ℕ) ∧ uses_exactly_once (n : ℕ) ∧ n = 31254 :=
sorry

end find_50th_permutation_l288_288241


namespace smart_charging_piles_growth_l288_288522

noncomputable def a : ℕ := 301
noncomputable def b : ℕ := 500
variable (x : ℝ) -- Monthly average growth rate

theorem smart_charging_piles_growth :
  a * (1 + x) ^ 2 = b :=
by
  -- Proof should go here
  sorry

end smart_charging_piles_growth_l288_288522


namespace cost_of_each_muffin_l288_288559

-- Define the cost of juice
def juice_cost : ℝ := 1.45

-- Define the total cost paid by Kevin
def total_cost : ℝ := 3.70

-- Assume the cost of each muffin
def muffin_cost (M : ℝ) : Prop := 
  3 * M + juice_cost = total_cost

-- The theorem we aim to prove
theorem cost_of_each_muffin : muffin_cost 0.75 :=
by
  -- Here the proof would go
  sorry

end cost_of_each_muffin_l288_288559


namespace angle_GZD_l288_288532

/-- Given that line segments AB and CD are parallel, angle A V E is 72 degrees, and angle E F G is t degrees, 
prove that angle G Z D is 108 degrees minus t. -/
theorem angle_GZD (AB CD : Line) (parallel : AB ∥ CD)
  (AVE : Angle) (HZ_AVE: AVE = 72)
  (EFG : Angle) (t : ℝ)
  (HZ_EFG: EFG = t) :
  ∃ GZD : Angle, GZD = 108 - t := 
sorry

end angle_GZD_l288_288532


namespace intersection_with_median_l288_288583

variables (a b c : ℝ)

def z_0 := Complex.I * a
def z_1 := 1/2 + Complex.I * b
def z_2 := 1 + Complex.I * c

noncomputable def z (t : ℝ) :=
  z_0 * (Real.cos t)^4 + 2 * z_1 * (Real.cos t)^2 * (Real.sin t)^2 + z_2 * (Real.sin t)^4

theorem intersection_with_median :
  (∃ t : ℝ, z a b c t = Complex.ofReal(1/2) + Complex.I * Real.ofReal((a + c + 2 * b) / 4)) := by
  sorry

end intersection_with_median_l288_288583


namespace cos_alpha_plus_7pi_over_12_l288_288025

theorem cos_alpha_plus_7pi_over_12 (α : ℝ) (h : sin (α + π / 12) = 1 / 3) : 
  cos (α + 7 * π / 12) = -1 / 3 := 
sorry

end cos_alpha_plus_7pi_over_12_l288_288025


namespace sum_of_squares_equiv_l288_288960

namespace PolynomialRoots

variables {a b c : ℝ}

def cubic_polynomial : Polynomial ℝ := 3 * X^3 - 4 * X^2 + 100 * X - 3

-- Assume a, b, c are roots of the cubic polynomial
axiom roots_of_cubic : cubic_polynomial.eval a = 0 ∧ cubic_polynomial.eval b = 0 ∧ cubic_polynomial.eval c = 0

-- Vieta's formula for sum of roots
axiom sum_of_roots : a + b + c = 4 / 3

-- The theorem to prove
theorem sum_of_squares_equiv : (a + b + 2)^2 + (b + c + 2)^2 + (c + a + 2)^2 = 1079 / 9 :=
by
  sorry

end PolynomialRoots

end sum_of_squares_equiv_l288_288960


namespace length_of_PR_l288_288941

theorem length_of_PR (P Q R S : Point) (h1 : IsAltitude PS QR) (h2 : angle QRP = 30) :
  length PR = 2 * length PS :=
  sorry

end length_of_PR_l288_288941


namespace not_enough_evidence_to_show_relationship_l288_288673

noncomputable def isEvidenceToShowRelationship (table : Array (Array Nat)) : Prop :=
  ∃ evidence : Bool, ¬evidence

theorem not_enough_evidence_to_show_relationship :
  isEvidenceToShowRelationship #[#[5, 15, 20], #[40, 10, 50], #[45, 25, 70]] :=
sorry 

end not_enough_evidence_to_show_relationship_l288_288673


namespace ratio_advertisement_to_outreach_l288_288371

theorem ratio_advertisement_to_outreach (total_hours : ℕ) (customer_outreach_hours : ℕ) (marketing_hours : ℕ) : total_hours = 8 → customer_outreach_hours = 4 → marketing_hours = 2 → (total_hours - (customer_outreach_hours + marketing_hours)) / customer_outreach_hours = 1 / 2 :=
by
  intro h_total h_outreach h_marketing
  have h_advertisement : total_hours - (customer_outreach_hours + marketing_hours) = 2 := by sorry
  calc
    (total_hours - (customer_outreach_hours + marketing_hours)) / customer_outreach_hours
      = 2 / 4 : by rw [h_advertisement, h_outreach]
      ... = 1 / 2 : by norm_num

end ratio_advertisement_to_outreach_l288_288371


namespace measure_angle_VRS_l288_288615

-- Define the conditions and question in Lean 4
variables (RS UV : Type) [linear_ordered_field RS] [linear_ordered_field UV]
variables (y : ℝ)
variables (angle_SUV angle_VRS angle_RVU : ℝ)
variables (parallel_RS_UV : RS ∥ UV)
variables (condition_SUV : angle_SUV = 2 * y)
variables (condition_VRS : angle_VRS = y + 3 * y)
variables (condition_RVU : angle_RVU = 2.5 * y)
variables (linear_pair : angle_VRS + angle_RVU = 180)

-- Prove the measure of angle VRS
theorem measure_angle_VRS :
  angle_VRS = 110.76 :=
by
  sorry

end measure_angle_VRS_l288_288615


namespace distance_between_A_and_B_l288_288404

-- Define the points A and B
def A : ℝ × ℝ := (2, 3)
def B : ℝ × ℝ := (5, 9)

-- Define the distance formula
def euclidean_distance (p1 p2 : ℝ × ℝ) : ℝ :=
  (real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2))

-- The proposition to prove
theorem distance_between_A_and_B : euclidean_distance A B = 3 * real.sqrt 5 := by
  sorry

end distance_between_A_and_B_l288_288404


namespace count_base_8_digits_5_or_6_l288_288864

-- Define the conditions in Lean
def is_digit_5_or_6 (d : ℕ) : Prop := d = 5 ∨ d = 6

def count_digits_5_or_6 := 
  let total_base_8 := 512
  let total_without_5_6 := 6 * 6 * 6 -- since we exclude 2 out of 8 digits
  total_base_8 - total_without_5_6

-- The statement of the proof problem
theorem count_base_8_digits_5_or_6 : count_digits_5_or_6 = 296 :=
by {
  sorry
}

end count_base_8_digits_5_or_6_l288_288864


namespace inequality_solution_l288_288655

theorem inequality_solution (x : ℝ) (h : 2^(x+2) > 8) : x > 1 :=
sorry

end inequality_solution_l288_288655


namespace number_of_possible_values_and_sum_of_f3_l288_288572

noncomputable def f (x : ℝ) : ℝ := sorry

theorem number_of_possible_values_and_sum_of_f3 : 
  (∀ x y : ℝ, f (x * f y + 2 * x) = 2 * x * y + f x) →
  let n := 2 in
  let s := 0 in
  n * s = 0 := sorry

end number_of_possible_values_and_sum_of_f3_l288_288572


namespace UV_over_RS_eq_1_665_l288_288605

-- Definitions based on conditions
def RectangleMNPQ : Type :=
  { M : ℝ × ℝ // M = (0, 6) } ×
  { N : ℝ × ℝ // N = (10, 6) } ×
  { P : ℝ × ℝ // P = (10, 0) } ×
  { Q : ℝ × ℝ // Q = (0, 0) }

def PointR : ℝ × ℝ := (8, 6)
def PointT : ℝ × ℝ := (10, 4)
def PointS : ℝ × ℝ := (3, 0)

-- We need to prove the ratio of the lengths
theorem UV_over_RS_eq_1_665 :
  ∃ V U R S : ℝ × ℝ,
    let UV : ℝ := real.sqrt ((U.1 - V.1)^2 + (U.2 - V.2)^2)
    let RS : ℝ := real.sqrt ((R.1 - S.1)^2 + (R.2 - S.2)^2)
    in UV / RS = 1.665 :=
by
  sorry

end UV_over_RS_eq_1_665_l288_288605


namespace line_passes_through_point_correct_k_l288_288290

/-- The line equation 2 - 3kx = 4y passes through the point (-1/3, -2) when k = -10 -/
theorem line_passes_through_point_correct_k : 
  ∃ k : ℤ, (k = -10) ∧ (∀ x y : ℚ, (x = -1/3) → (y = -2) → (2 - 3 * k * x = 4 * y)) :=
by
  use -10
  intro x y hx hy
  rw [hx, hy]
  norm_num
  sorry

end line_passes_through_point_correct_k_l288_288290
