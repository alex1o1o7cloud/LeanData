import Mathlib

namespace assignment_plans_l116_116920

theorem assignment_plans (students locations : ℕ) (library science_museum nursing_home : ℕ) 
  (students_eq : students = 5) (locations_eq : locations = 3) 
  (lib_gt0 : library > 0) (sci_gt0 : science_museum > 0) (nur_gt0 : nursing_home > 0) 
  (lib_science_nursing : library + science_museum + nursing_home = students) : 
  ∃ (assignments : ℕ), assignments = 150 :=
by
  sorry

end assignment_plans_l116_116920


namespace cylindrical_to_cartesian_l116_116313

theorem cylindrical_to_cartesian :
  ∀ (r θ z : ℝ), r = 2 → θ = π / 3 → z = 2 → 
  (r * Real.cos θ, r * Real.sin θ, z) = (1, Real.sqrt 3, 2) :=
by
  intros r θ z hr hθ hz
  sorry

end cylindrical_to_cartesian_l116_116313


namespace find_x_l116_116765

theorem find_x (x y : ℕ) (h1 : x / y = 8 / 3) (h2 : y = 27) : x = 72 := by
  sorry

end find_x_l116_116765


namespace quadrilateral_perimeter_l116_116336

theorem quadrilateral_perimeter (a b : ℝ) (h₁ : a = 10) (h₂ : b = 15)
  (h₃ : ∀ (ABD BCD ABC ACD : ℝ), ABD = BCD ∧ ABC = ACD) : a + a + b + b = 50 :=
by
  rw [h₁, h₂]
  linarith


end quadrilateral_perimeter_l116_116336


namespace angle_in_second_quadrant_l116_116448

open Real

-- Define the fourth quadrant condition
def isFourthQuadrant (α : ℝ) (k : ℤ) : Prop :=
  2 * k * π - π / 2 < α ∧ α < 2 * k * π

-- Define the second quadrant condition
def isSecondQuadrant (β : ℝ) (k : ℤ) : Prop :=
  2 * k * π + π / 2 < β ∧ β < 2 * k * π + π

-- The main theorem to prove
theorem angle_in_second_quadrant (α : ℝ) (k : ℤ) :
  isFourthQuadrant α k → isSecondQuadrant (π + α) k :=
sorry

end angle_in_second_quadrant_l116_116448


namespace intersection_points_l116_116662

noncomputable def even_function (f : ℝ → ℝ) :=
  ∀ x, f (-x) = f x

def monotonically_increasing (f : ℝ → ℝ) :=
  ∀ x y, 0 < x ∧ x < y → f x < f y

theorem intersection_points (f : ℝ → ℝ)
  (h_even : even_function f)
  (h_mono_inc : monotonically_increasing f)
  (h_sign_change : f 1 * f 2 < 0) :
  ∃! x1 x2 : ℝ, f x1 = 0 ∧ f x2 = 0 ∧ x1 ≠ x2 :=
sorry

end intersection_points_l116_116662


namespace fraction_books_sold_l116_116785

theorem fraction_books_sold :
  (∃ B F : ℝ, 3.50 * (B - 40) = 280.00000000000006 ∧ B ≠ 0 ∧ F = ((B - 40) / B) ∧ B = 120) → (F = 2 / 3) :=
by
  intro h
  obtain ⟨B, F, h1, h2, e⟩ := h
  sorry

end fraction_books_sold_l116_116785


namespace abs_sub_abs_eq_six_l116_116795

theorem abs_sub_abs_eq_six
  (a b : ℝ)
  (h₁ : |a| = 4)
  (h₂ : |b| = 2)
  (h₃ : a * b < 0) :
  |a - b| = 6 :=
sorry

end abs_sub_abs_eq_six_l116_116795


namespace smallest_integer_2023m_54321n_l116_116485

theorem smallest_integer_2023m_54321n : ∃ (m n : ℤ), 2023 * m + 54321 * n = 1 :=
sorry

end smallest_integer_2023m_54321n_l116_116485


namespace egg_rolls_total_l116_116372

def total_egg_rolls (omar_rolls : ℕ) (karen_rolls : ℕ) : ℕ :=
  omar_rolls + karen_rolls

theorem egg_rolls_total :
  total_egg_rolls 219 229 = 448 :=
by
  sorry

end egg_rolls_total_l116_116372


namespace math_problem_l116_116501

-- Define the individual numbers
def a : Int := 153
def b : Int := 39
def c : Int := 27
def d : Int := 21

-- Define the entire expression and its expected result
theorem math_problem : (a + b + c + d) * 2 = 480 := by
  sorry

end math_problem_l116_116501


namespace satisify_absolute_value_inequality_l116_116447

theorem satisify_absolute_value_inequality :
  ∃ (t : Finset ℤ), t.card = 2 ∧ ∀ y ∈ t, |7 * y + 4| ≤ 10 :=
by
  sorry

end satisify_absolute_value_inequality_l116_116447


namespace range_of_a_l116_116824

theorem range_of_a (a : ℝ) (h : a ≥ 0) :
  ∃ a, (2 * Real.sqrt 3 ≤ a ∧ a ≤ 4 * Real.sqrt 2) ↔
  (∀ x y : ℝ, 
    ((x - a)^2 + y^2 = 1) ∧ (x^2 + (y - 2)^2 = 25)) :=
sorry

end range_of_a_l116_116824


namespace polynomial_roots_l116_116207

theorem polynomial_roots :
  ∀ x, (3 * x^4 + 16 * x^3 - 36 * x^2 + 8 * x = 0) ↔ 
       (x = 0 ∨ x = 1 / 3 ∨ x = -3 + 2 * Real.sqrt 17 ∨ x = -3 - 2 * Real.sqrt 17) :=
by
  sorry

end polynomial_roots_l116_116207


namespace gary_money_after_sale_l116_116960

theorem gary_money_after_sale :
  let initial_money := 73.0
  let sale_amount := 55.0
  initial_money + sale_amount = 128.0 :=
by
  let initial_money := 73.0
  let sale_amount := 55.0
  show initial_money + sale_amount = 128.0
  sorry

end gary_money_after_sale_l116_116960


namespace part_a_part_b_l116_116132

-- Define the functions K_m and K_4
def K (m : ℕ) (x y z : ℝ) : ℝ :=
  x * (x - y)^m * (x - z)^m + y * (y - x)^m * (y - z)^m + z * (z - x)^m * (z - y)^m

-- Define M
def M (x y z : ℝ) : ℝ :=
  (x - y)^2 * (y - z)^2 * (z - x)^2

-- The proof goals:
-- 1. Prove K_m >= 0 for odd positive integer m
theorem part_a (m : ℕ) (hm : m % 2 = 1) (x y z : ℝ) : 
  0 ≤ K m x y z := 
sorry

-- 2. Prove K_7 + M^2 * K_1 >= M * K_4
theorem part_b (x y z : ℝ) : 
  K 7 x y z + (M x y z)^2 * K 1 x y z ≥ M x y z * K 4 x y z := 
sorry

end part_a_part_b_l116_116132


namespace initial_volume_of_mixture_l116_116858

variable (V : ℝ)
variable (H1 : 0.2 * V + 12 = 0.25 * (V + 12))

theorem initial_volume_of_mixture (H : 0.2 * V + 12 = 0.25 * (V + 12)) : V = 180 := by
  sorry

end initial_volume_of_mixture_l116_116858


namespace forty_percent_of_number_l116_116130

theorem forty_percent_of_number (N : ℝ) 
  (h : (1 / 4) * (1 / 3) * (2 / 5) * N = 17) : 
  0.40 * N = 204 :=
sorry

end forty_percent_of_number_l116_116130


namespace swimming_time_l116_116643

theorem swimming_time (c t : ℝ) 
  (h1 : 10.5 + c ≠ 0)
  (h2 : 10.5 - c ≠ 0)
  (h3 : t = 45 / (10.5 + c))
  (h4 : t = 18 / (10.5 - c)) :
  t = 3 := 
by
  sorry

end swimming_time_l116_116643


namespace quadratic_equation_problems_l116_116830

noncomputable def quadratic_has_real_roots (m : ℝ) : Prop :=
  let a := m
  let b := -(3 * m - 1)
  let c := 2 * m - 2
  let Δ := b ^ 2 - 4 * a * c
  Δ ≥ 0

noncomputable def valid_m_values (m : ℝ) : Prop :=
  let a := m
  let b := -(3 * m - 1)
  let c := 2 * m - 2
  let Δ := b ^ 2 - 4 * a * c
  1 = m ∨ -1 / 3 = m

theorem quadratic_equation_problems (m : ℝ) :
  quadratic_has_real_roots m ∧
  (∀ x1 x2 : ℝ, 
      (x1 ≠ x2) →
      x1 + x2 = -(3 * m - 1) / m →
      x1 * x2 = (2 * m - 2) / m →
      abs (x1 - x2) = 2 →
      valid_m_values m) :=
by 
  sorry

end quadratic_equation_problems_l116_116830


namespace inequality_proof_l116_116085

theorem inequality_proof (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a * b * c = 1) : 
  1 / (a^3 * (b + c)) + 1 / (b^3 * (c + a)) + 1 / (c^3 * (a + b)) ≥ 3 / 2 :=
by
  sorry

end inequality_proof_l116_116085


namespace variance_of_data_set_is_4_l116_116469

/-- The data set for which we want to calculate the variance --/
def data_set : List ℝ := [2, 4, 5, 6, 8]

/-- The mean of the data set --/
noncomputable def mean (l : List ℝ) : ℝ :=
  (l.sum) / (l.length)

-- Calculation of the variance of a list given its mean
noncomputable def variance (l : List ℝ) (μ : ℝ) : ℝ :=
  (l.map (λ x => (x - μ) ^ 2)).sum / l.length

theorem variance_of_data_set_is_4 :
  variance data_set (mean data_set) = 4 :=
by
  sorry

end variance_of_data_set_is_4_l116_116469


namespace problem_R_l116_116524

noncomputable def R (g S h : ℝ) : ℝ := g * S + h

theorem problem_R {g h : ℝ} (h_h : h = 6 - 4 * g) :
  R g 14 h = 56 :=
by
  sorry

end problem_R_l116_116524


namespace platform_length_l116_116316

variable (L : ℝ) -- The length of the platform
variable (train_length : ℝ := 300) -- The length of the train
variable (time_pole : ℝ := 26) -- Time to cross the signal pole
variable (time_platform : ℝ := 39) -- Time to cross the platform

theorem platform_length :
  (train_length / time_pole) = (train_length + L) / time_platform → L = 150 := sorry

end platform_length_l116_116316


namespace abs_mult_example_l116_116754

theorem abs_mult_example : (|(-3)| * 2) = 6 := by
  have h1 : |(-3)| = 3 := by
    exact abs_of_neg (show -3 < 0 by norm_num)
  rw [h1]
  exact mul_eq_mul_left_iff.mpr (Or.inl rfl)

end abs_mult_example_l116_116754


namespace determine_b_l116_116905

noncomputable def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

theorem determine_b (a b c m1 m2 : ℝ) (h1 : a > b) (h2 : b > c) (h3 : f a b c 1 = 0)
  (h4 : a^2 + (f a b c m1 + f a b c m2) * a + (f a b c m1) * (f a b c m2) = 0) : 
  b ≥ 0 := 
by
  -- Proof logic goes here
  sorry

end determine_b_l116_116905


namespace inverse_sum_l116_116277

def f (x : ℝ) : ℝ := x * |x|

theorem inverse_sum (h1 : ∃ x : ℝ, f x = 9) (h2 : ∃ x : ℝ, f x = -81) :
  ∃ a b: ℝ, f a = 9 ∧ f b = -81 ∧ a + b = -6 :=
by
  sorry

end inverse_sum_l116_116277


namespace negation_of_universal_l116_116295

theorem negation_of_universal :
  ¬ (∀ x : ℝ, x^3 - x^2 + 1 ≤ 0) ↔ ∃ x : ℝ, x^3 - x^2 + 1 > 0 :=
by
  sorry    -- Proof is not required, just the statement.

end negation_of_universal_l116_116295


namespace eval_imaginary_expression_l116_116371

theorem eval_imaginary_expression :
  ∀ (i : ℂ), i^2 = -1 → i^2022 + i^2023 + i^2024 + i^2025 = 0 :=
by
  sorry

end eval_imaginary_expression_l116_116371


namespace calculation_l116_116779

theorem calculation : (-6)^6 / 6^4 + 4^3 - 7^2 * 2 = 2 :=
by
  -- We add "sorry" here to indicate where the proof would go.
  sorry

end calculation_l116_116779


namespace solve_for_x_l116_116976

theorem solve_for_x (x : ℤ) (h : 3 * x - 5 = 4 * x + 10) : x = -15 :=
sorry

end solve_for_x_l116_116976


namespace adoption_days_l116_116733

theorem adoption_days (P0 P_in P_adopt_rate : Nat) (P_total : Nat) (hP0 : P0 = 3) (hP_in : P_in = 3) (hP_adopt_rate : P_adopt_rate = 3) (hP_total : P_total = P0 + P_in) :
  P_total / P_adopt_rate = 2 := 
by
  sorry

end adoption_days_l116_116733


namespace compound_interest_rate_l116_116165

theorem compound_interest_rate (P r : ℝ) (h1 : 17640 = P * (1 + r / 100)^8)
                                (h2 : 21168 = P * (1 + r / 100)^12) :
  4 * (r / 100) = 18.6 :=
by
  sorry

end compound_interest_rate_l116_116165


namespace consecutive_numbers_count_l116_116431

theorem consecutive_numbers_count (n : ℕ) 
(avg : ℝ) 
(largest : ℕ) 
(h_avg : avg = 20) 
(h_largest : largest = 23) 
(h_eq : (largest + (largest - (n - 1))) / 2 = avg) : 
n = 7 := 
by 
  sorry

end consecutive_numbers_count_l116_116431


namespace simplify_expression_l116_116929

theorem simplify_expression (x y z : ℝ) :
  3 * (x - (2 * y - 3 * z)) - 2 * ((3 * x - 2 * y) - 4 * z) = -3 * x - 2 * y + 17 * z :=
by
  sorry

end simplify_expression_l116_116929


namespace sum_of_greatest_values_l116_116625

theorem sum_of_greatest_values (b : ℝ) (h : 4 * b ^ 4 - 41 * b ^ 2 + 100 = 0) :
  b = 2.5 ∨ b = 2 → 2.5 + 2 = 4.5 :=
by sorry

end sum_of_greatest_values_l116_116625


namespace simplify_expression_l116_116416

variable (a b c : ℝ)

theorem simplify_expression :
  (-32 * a^4 * b^5 * c) / ((-2 * a * b)^3) * (-3 / 4 * a * c) = -3 * a^2 * b^2 * c^2 :=
  by
    sorry

end simplify_expression_l116_116416


namespace tan_theta_sub_9pi_l116_116198

theorem tan_theta_sub_9pi (θ : ℝ) (h : Real.cos (Real.pi + θ) = -1 / 2) : 
  Real.tan (θ - 9 * Real.pi) = Real.sqrt 3 :=
by
  sorry

end tan_theta_sub_9pi_l116_116198


namespace no_arrangement_of_1_to_1978_coprime_l116_116893

theorem no_arrangement_of_1_to_1978_coprime :
  ¬ ∃ (a : Fin 1978 → ℕ), 
    (∀ i : Fin 1977, Nat.gcd (a i) (a (i + 1)) = 1) ∧ 
    (∀ i : Fin 1976, Nat.gcd (a i) (a (i + 2)) = 1) ∧ 
    (∀ i : Fin 1978, 1 ≤ a i ∧ a i ≤ 1978 ∧ ∀ j : Fin 1978, (i ≠ j → a i ≠ a j)) :=
sorry

end no_arrangement_of_1_to_1978_coprime_l116_116893


namespace total_surface_area_of_cylinder_l116_116956

theorem total_surface_area_of_cylinder 
  (r h : ℝ) 
  (hr : r = 3) 
  (hh : h = 8) : 
  2 * Real.pi * r * h + 2 * Real.pi * r^2 = 66 * Real.pi := by
  sorry

end total_surface_area_of_cylinder_l116_116956


namespace batsman_average_after_17th_inning_l116_116978

theorem batsman_average_after_17th_inning
  (A : ℕ)  -- average after the 16th inning
  (h1 : 16 * A + 300 = 17 * (A + 10)) :
  A + 10 = 140 :=
by
  sorry

end batsman_average_after_17th_inning_l116_116978


namespace roots_cubic_eq_l116_116106

theorem roots_cubic_eq (r s p q : ℝ) (h1 : r + s = p) (h2 : r * s = q) :
    r^3 + s^3 = p^3 - 3 * q * p :=
by
    -- Placeholder for proof
    sorry

end roots_cubic_eq_l116_116106


namespace days_to_complete_work_l116_116672

-- Conditions
def work_rate_A : ℚ := 1 / 8
def work_rate_B : ℚ := 1 / 16
def combined_work_rate := work_rate_A + work_rate_B

-- Statement to prove
theorem days_to_complete_work : 1 / combined_work_rate = 16 / 3 := by
  sorry

end days_to_complete_work_l116_116672


namespace sales_tax_difference_l116_116907

theorem sales_tax_difference :
  let price : ℝ := 50
  let tax_rate1 : ℝ := 0.075
  let tax_rate2 : ℝ := 0.0625
  let tax1 := price * tax_rate1
  let tax2 := price * tax_rate2
  let difference := tax1 - tax2
  difference = 0.625 :=
by
  sorry

end sales_tax_difference_l116_116907


namespace halfway_fraction_l116_116306

theorem halfway_fraction (a b : ℚ) (h₁ : a = 3/4) (h₂ : b = 5/6) : (a + b) / 2 = 19 / 24 :=
by
  sorry

end halfway_fraction_l116_116306


namespace ratio_elyse_to_rick_l116_116774

-- Define the conditions
def Elyse_initial_gum : ℕ := 100
def Shane_leftover_gum : ℕ := 14
def Shane_chewed_gum : ℕ := 11

-- Theorem stating the ratio of pieces Elyse gave to Rick to the total number of pieces Elyse had
theorem ratio_elyse_to_rick :
  let total_gum := Elyse_initial_gum
  let Shane_initial_gum := Shane_leftover_gum + Shane_chewed_gum
  let Rick_initial_gum := 2 * Shane_initial_gum
  let Elyse_given_to_Rick := Rick_initial_gum
  (Elyse_given_to_Rick : ℚ) / total_gum = 1 / 2 :=
by
  sorry

end ratio_elyse_to_rick_l116_116774


namespace division_example_l116_116906

theorem division_example : ∃ A B : ℕ, 23 = 6 * A + B ∧ A = 3 ∧ B < 6 := 
by sorry

end division_example_l116_116906


namespace hexagon_coloring_unique_l116_116308

-- Define the coloring of the hexagon using enumeration
inductive Color
  | green
  | blue
  | orange

-- Assume we have a function that represents the coloring of the hexagons
-- with the constraints given in the problem
def is_valid_coloring (coloring : ℕ → ℕ → Color) : Prop :=
  ∀ x y : ℕ, -- For all hexagons
  (coloring x y = Color.green ∧ x = 0 ∧ y = 0) ∨ -- The labeled hexagon G is green
  (coloring x y ≠ coloring (x + 1) y ∧ -- No two hexagons with a common side have the same color
   coloring x y ≠ coloring (x - 1) y ∧ 
   coloring x y ≠ coloring x (y + 1) ∧
   coloring x y ≠ coloring x (y - 1))

-- The problem is to prove there are exactly 2 valid colorings of the hexagon grid
theorem hexagon_coloring_unique :
  ∃ (count : ℕ), count = 2 ∧
  ∀ coloring : (ℕ → ℕ → Color), is_valid_coloring coloring → count = 2 :=
by
  sorry

end hexagon_coloring_unique_l116_116308


namespace absolute_value_simplify_l116_116615

variable (a : ℝ)

theorem absolute_value_simplify
  (h : a < 3) : |a - 3| = 3 - a := sorry

end absolute_value_simplify_l116_116615


namespace sufficient_not_necessary_condition_l116_116010

theorem sufficient_not_necessary_condition (x : ℝ) :
  (x > 2) → ((x + 1) * (x - 2) > 0) ∧ ¬(∀ y, (y + 1) * (y - 2) > 0 → y > 2) := 
sorry

end sufficient_not_necessary_condition_l116_116010


namespace scooter_price_l116_116274

theorem scooter_price (total_cost: ℝ) (h: 0.20 * total_cost = 240): total_cost = 1200 :=
by
  sorry

end scooter_price_l116_116274


namespace area_of_sine_triangle_l116_116538

-- We define the problem conditions and the statement we want to prove
theorem area_of_sine_triangle (A B C : Real) (area_ABC : ℝ) (unit_circle : ℝ) :
  unit_circle = 1 → area_ABC = 1 / 2 →
  let a := 2 * Real.sin A
  let b := 2 * Real.sin B
  let c := 2 * Real.sin C
  let s := (a + b + c) / 2
  let area_sine_triangle := 
    (s * (s - a) * (s - b) * (s - c)).sqrt / 4 
  area_sine_triangle = 1 / 8 :=
by
  intros
  sorry -- Proof is left as an exercise

end area_of_sine_triangle_l116_116538


namespace probability_one_out_of_three_l116_116499

def probability_passing_exactly_one (p : ℚ) (n k : ℕ) :=
  (Nat.choose n k) * (p^k) * ((1 - p)^(n - k))

theorem probability_one_out_of_three :
  probability_passing_exactly_one (1/3) 3 1 = 4/9 :=
by sorry

end probability_one_out_of_three_l116_116499


namespace symmetric_sum_l116_116811

theorem symmetric_sum (a b : ℤ) (h1 : a = -4) (h2 : b = -3) : a + b = -7 := by
  sorry

end symmetric_sum_l116_116811


namespace point_on_x_axis_coordinates_l116_116201

theorem point_on_x_axis_coordinates (a : ℝ) (P : ℝ × ℝ) (h : P = (a - 1, a + 2)) (hx : P.2 = 0) : P = (-3, 0) :=
by
  -- Replace this with the full proof
  sorry

end point_on_x_axis_coordinates_l116_116201


namespace angles_in_interval_l116_116882

-- Define the main statement we need to prove
theorem angles_in_interval (theta : ℝ) (h1 : 0 ≤ theta) (h2 : theta ≤ 2 * Real.pi) :
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → x^2 * Real.cos theta - x * (1 - x) + (1-x)^2 * Real.sin theta < 0) →
  (Real.pi / 2 < theta ∧ theta < 3 * Real.pi / 2) :=
by
  sorry

end angles_in_interval_l116_116882


namespace triangle_is_isosceles_l116_116514

theorem triangle_is_isosceles (α β γ δ ε : ℝ)
  (h1 : α + β + γ = 180) 
  (h2 : α + β = δ) 
  (h3 : β + γ = ε) : 
  α = γ ∨ β = γ ∨ α = β := 
sorry

end triangle_is_isosceles_l116_116514


namespace largest_triangle_angle_l116_116732

theorem largest_triangle_angle (y : ℝ) (h1 : 45 + 60 + y = 180) : y = 75 :=
by { sorry }

end largest_triangle_angle_l116_116732


namespace sequence_general_term_l116_116596

theorem sequence_general_term (n : ℕ) (h : n > 0) : 
  ∃ a : ℕ → ℚ, (∀ n, a n = 1 / n) :=
by
  sorry

end sequence_general_term_l116_116596


namespace interest_rate_correct_l116_116921

theorem interest_rate_correct :
  let SI := 155
  let P := 810
  let T := 4
  let R := SI * 100 / (P * T)
  R = 155 * 100 / (810 * 4) := 
sorry

end interest_rate_correct_l116_116921


namespace log_sum_nine_l116_116402

-- Define that {a_n} is a geometric sequence and satisfies the given conditions.
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a n = a 1 * r ^ (n - 1)

-- Given conditions
axiom a_pos (a : ℕ → ℝ) : (∀ n, a n > 0)      -- All terms are positive
axiom a2a8_eq_4 (a : ℕ → ℝ) : a 2 * a 8 = 4    -- a₂a₈ = 4

theorem log_sum_nine (a : ℕ → ℝ) 
  (geo_seq : geometric_sequence a) 
  (pos : ∀ n, a n > 0)
  (eq4 : a 2 * a 8 = 4) :
  (Real.logb 2 (a 1) + Real.logb 2 (a 2) + Real.logb 2 (a 3) + Real.logb 2 (a 4)
  + Real.logb 2 (a 5) + Real.logb 2 (a 6) + Real.logb 2 (a 7) + Real.logb 2 (a 8)
  + Real.logb 2 (a 9)) = 9 :=
by
  sorry

end log_sum_nine_l116_116402


namespace crayons_total_l116_116272

def crayons_per_child := 6
def number_of_children := 12
def total_crayons := 72

theorem crayons_total :
  crayons_per_child * number_of_children = total_crayons := by
  sorry

end crayons_total_l116_116272


namespace number_of_people_in_village_l116_116299

variable (P : ℕ) -- Define the total number of people in the village

def people_not_working : ℕ := 50
def people_with_families : ℕ := 25
def people_singing_in_shower : ℕ := 75
def max_people_overlap : ℕ := 50

theorem number_of_people_in_village :
  P - people_not_working + P - people_with_families + P - people_singing_in_shower - max_people_overlap = P → 
  P = 100 :=
by
  sorry

end number_of_people_in_village_l116_116299


namespace fraction_shaded_in_cube_l116_116803

theorem fraction_shaded_in_cube :
  let side_length := 2
  let face_area := side_length * side_length
  let total_surface_area := 6 * face_area
  let shaded_faces := 3
  let shaded_face_area := face_area / 2
  let total_shaded_area := shaded_faces * shaded_face_area
  total_shaded_area / total_surface_area = 1 / 4 :=
by
  sorry

end fraction_shaded_in_cube_l116_116803


namespace direct_proportion_increases_inverse_proportion_increases_l116_116208

-- Question 1: Prove y=2x increases as x increases.
theorem direct_proportion_increases (x1 x2 : ℝ) (h : x1 < x2) : 
  2 * x1 < 2 * x2 := by sorry

-- Question 2: Prove y=-2/x increases as x increases when x > 0.
theorem inverse_proportion_increases (x1 x2 : ℝ) (h1 : 0 < x1) (h2 : x1 < x2) : 
  - (2 / x1) < - (2 / x2) := by sorry

end direct_proportion_increases_inverse_proportion_increases_l116_116208


namespace dimension_proof_l116_116951

noncomputable def sports_field_dimensions (x y: ℝ) : Prop :=
  -- Given conditions
  x^2 + y^2 = 185^2 ∧
  (x - 4) * (y - 4) = x * y - 1012 ∧
  -- Seeking to prove dimensions
  ((x = 153 ∧ y = 104) ∨ (x = 104 ∧ y = 153))

theorem dimension_proof : ∃ x y: ℝ, sports_field_dimensions x y := by
  sorry

end dimension_proof_l116_116951


namespace trajectory_of_Q_is_parabola_l116_116246

/--
Given a point P (x, y) moves on a unit circle centered at the origin,
prove that the trajectory of point Q (u, v) defined by u = x + y and v = xy 
satisfies u^2 - 2v = 1 and is thus a parabola.
-/
theorem trajectory_of_Q_is_parabola 
  (x y u v : ℝ) 
  (h1 : x^2 + y^2 = 1) 
  (h2 : u = x + y) 
  (h3 : v = x * y) :
  u^2 - 2 * v = 1 :=
sorry

end trajectory_of_Q_is_parabola_l116_116246


namespace principal_amount_l116_116077

theorem principal_amount (P : ℝ) (r t : ℝ) (d : ℝ) 
  (h1 : r = 7)
  (h2 : t = 2)
  (h3 : d = 49)
  (h4 : P * ((1 + r / 100) ^ t - 1) - P * (r * t / 100) = d) :
  P = 10000 :=
by sorry

end principal_amount_l116_116077


namespace common_difference_of_arithmetic_sequence_l116_116829

theorem common_difference_of_arithmetic_sequence
  (a : ℕ → ℝ) (d : ℝ)
  (h1 : a 2 = 9)
  (h2 : a 5 = 33)
  (h_arith_seq : ∀ n : ℕ, a (n + 1) = a n + d) :
  d = 8 :=
sorry

end common_difference_of_arithmetic_sequence_l116_116829


namespace minimum_value_f_l116_116619

noncomputable def f (x : ℝ) (f1 f2 : ℝ) : ℝ :=
  f1 * x + f2 / x - 2

theorem minimum_value_f (f1 f2 : ℝ) (h1 : f2 = 2) (h2 : f1 = 3 / 2) :
  ∃ x > 0, f x f1 f2 = 2 * Real.sqrt 3 - 2 :=
by
  sorry

end minimum_value_f_l116_116619


namespace fish_worth_bags_of_rice_l116_116088

variable (f l a r : ℝ)

theorem fish_worth_bags_of_rice
    (h1 : 5 * f = 3 * l)
    (h2 : l = 6 * a)
    (h3 : 2 * a = r) :
    1 / f = 9 / (5 * r) :=
by
  sorry

end fish_worth_bags_of_rice_l116_116088


namespace converse_negation_contrapositive_l116_116315

variable {x : ℝ}

def P (x : ℝ) : Prop := x^2 - 3 * x + 2 ≠ 0
def Q (x : ℝ) : Prop := x ≠ 1 ∧ x ≠ 2

theorem converse (h : Q x) : P x := by
  sorry

theorem negation (h : ¬ P x) : ¬ Q x := by
  sorry

theorem contrapositive (h : ¬ Q x) : ¬ P x := by
  sorry

end converse_negation_contrapositive_l116_116315


namespace cub_eqn_root_sum_l116_116872

noncomputable def cos_x := Real.cos (Real.pi / 5)

theorem cub_eqn_root_sum (a b : ℕ) (h1 : a > 0) (h2 : b > 0)
(h3 : a * cos_x ^ 3 - b * cos_x - 1 = 0) : a + b = 12 :=
sorry

end cub_eqn_root_sum_l116_116872


namespace solution_to_equation_l116_116825

theorem solution_to_equation : 
    (∃ x : ℤ, (x = 2 ∨ x = -2 ∨ x = 1 ∨ x = -1) ∧ (2 * x - 3 = -1)) → x = 1 :=
by
  sorry

end solution_to_equation_l116_116825


namespace find_c_l116_116541

open Real

noncomputable def triangle_side_c (a b c : ℝ) (A B C : ℝ) :=
  A = (π / 4) ∧
  2 * b * sin B - c * sin C = 2 * a * sin A ∧
  (1/2) * b * c * (sqrt 2)/2 = 3 →
  c = 2 * sqrt 2
  
theorem find_c {a b c A B C : ℝ} (h : triangle_side_c a b c A B C) : c = 2 * sqrt 2 :=
sorry

end find_c_l116_116541


namespace max_books_borrowed_l116_116385

theorem max_books_borrowed (students_total : ℕ) (students_no_books : ℕ) 
  (students_1_book : ℕ) (students_2_books : ℕ) (students_at_least_3_books : ℕ) 
  (average_books_per_student : ℝ) (H1 : students_total = 60) 
  (H2 : students_no_books = 4) 
  (H3 : students_1_book = 18) 
  (H4 : students_2_books = 20) 
  (H5 : students_at_least_3_books = students_total - (students_no_books + students_1_book + students_2_books)) 
  (H6 : average_books_per_student = 2.5) : 
  ∃ max_books : ℕ, max_books = 41 :=
by
  sorry

end max_books_borrowed_l116_116385


namespace jon_weekly_speed_gain_l116_116942

-- Definitions based on the conditions
def initial_speed : ℝ := 80
def speed_increase_percentage : ℝ := 0.20
def training_sessions : ℕ := 4
def weeks_per_session : ℕ := 4
def total_training_duration : ℕ := training_sessions * weeks_per_session

-- The calculated final speed
def final_speed : ℝ := initial_speed + initial_speed * speed_increase_percentage

theorem jon_weekly_speed_gain : 
  (final_speed - initial_speed) / total_training_duration = 1 :=
by
  -- This is the statement we want to prove
  sorry

end jon_weekly_speed_gain_l116_116942


namespace books_shelves_l116_116298

def initial_books : ℝ := 40.0
def additional_books : ℝ := 20.0
def books_per_shelf : ℝ := 4.0

theorem books_shelves :
  (initial_books + additional_books) / books_per_shelf = 15 :=
by 
  sorry

end books_shelves_l116_116298


namespace mira_weekly_distance_l116_116105

noncomputable def total_distance_jogging : ℝ :=
  let monday_distance := 4 * 2
  let thursday_distance := 5 * 1.5
  monday_distance + thursday_distance

noncomputable def total_distance_swimming : ℝ :=
  2 * 1

noncomputable def total_distance_cycling : ℝ :=
  12 * 1

noncomputable def total_distance : ℝ :=
  total_distance_jogging + total_distance_swimming + total_distance_cycling

theorem mira_weekly_distance : total_distance = 29.5 := by
  unfold total_distance
  unfold total_distance_jogging
  unfold total_distance_swimming
  unfold total_distance_cycling
  sorry

end mira_weekly_distance_l116_116105


namespace seq_composite_l116_116605

-- Define the sequence recurrence relation
def seq (a : ℕ → ℕ) : Prop :=
  ∀ (k : ℕ), k ≥ 1 → a (k+2) = a (k+1) * a k + 1

-- Prove that for k ≥ 9, a_k - 22 is composite
theorem seq_composite (a : ℕ → ℕ) (h_seq : seq a) :
  ∀ (k : ℕ), k ≥ 9 → ∃ d, d > 1 ∧ d < a k ∧ d ∣ (a k - 22) :=
by
  sorry

end seq_composite_l116_116605


namespace intersection_A_B_l116_116719

open Set

variable (x : ℝ)

def A : Set ℝ := {x | x^2 - 3 * x - 4 < 0}
def B : Set ℝ := {-4, 1, 3, 5}

theorem intersection_A_B :
  A ∩ B = { 1, 3 } :=
sorry

end intersection_A_B_l116_116719


namespace arithmetic_sequence_probability_l116_116054

theorem arithmetic_sequence_probability (n p : ℕ) (h_cond : n + p = 2008) (h_neg : n = 161) (h_pos : p = 2008 - 161) :
  ∃ a b : ℕ, (a = 1715261 ∧ b = 2016024 ∧ a + b = 3731285) ∧ (a / b = 1715261 / 2016024) := by
  sorry

end arithmetic_sequence_probability_l116_116054


namespace product_of_good_numbers_is_good_l116_116800

def is_good (n : ℕ) : Prop :=
  ∃ (a b c x y : ℤ), n = a * x * x + b * x * y + c * y * y ∧ b * b - 4 * a * c = -20

theorem product_of_good_numbers_is_good {n1 n2 : ℕ} (h1 : is_good n1) (h2 : is_good n2) : is_good (n1 * n2) :=
sorry

end product_of_good_numbers_is_good_l116_116800


namespace ratio_a7_b7_l116_116938

-- Definitions of the conditions provided in the problem
variables {a b : ℕ → ℝ}   -- Arithmetic sequences {a_n} and {b_n}
variables {S T : ℕ → ℝ}   -- Sums of the first n terms of {a_n} and {b_n}

-- Condition: For any positive integer n, S_n / T_n = (3n + 5) / (2n + 3)
axiom condition_S_T (n : ℕ) (hn : 0 < n) : S n / T n = (3 * n + 5) / (2 * n + 3)

-- Goal: Prove that a_7 / b_7 = 44 / 29
theorem ratio_a7_b7 : a 7 / b 7 = 44 / 29 := 
sorry

end ratio_a7_b7_l116_116938


namespace prove_ordered_triple_l116_116139

theorem prove_ordered_triple (x y z : ℝ) (h1 : x > 2) (h2 : y > 2) (h3 : z > 2)
  (h4 : (x + 3)^2 / (y + z - 3) + (y + 5)^2 / (z + x - 5) + (z + 7)^2 / (x + y - 7) = 45) : 
  (x, y, z) = (13, 11, 6) :=
sorry

end prove_ordered_triple_l116_116139


namespace project_completion_time_l116_116513

def process_duration (a b c d e f : Nat) : Nat :=
  let duration_c := max a b + c
  let duration_d := duration_c + d
  let duration_e := duration_c + e
  let duration_f := max duration_d duration_e + f
  duration_f

theorem project_completion_time :
  ∀ (a b c d e f : Nat), a = 2 → b = 3 → c = 2 → d = 5 → e = 4 → f = 1 →
  process_duration a b c d e f = 11 := by
  intros
  subst_vars
  sorry

end project_completion_time_l116_116513


namespace fair_coin_flip_probability_difference_l116_116915

noncomputable def choose : ℕ → ℕ → ℕ
| _, 0 => 1
| 0, _ => 0
| n + 1, r + 1 => choose n r + choose n (r + 1)

theorem fair_coin_flip_probability_difference :
  let p := 1 / 2
  let p_heads_4_out_of_5 := choose 5 4 * p^4 * p
  let p_heads_5_out_of_5 := p^5
  p_heads_4_out_of_5 - p_heads_5_out_of_5 = 1 / 8 :=
by
  sorry

end fair_coin_flip_probability_difference_l116_116915


namespace geometric_sequence_a3_is_15_l116_116055

noncomputable def geometric_sequence (a1 q : ℝ) (n : ℕ) : ℝ :=
a1 * q^(n - 1)

theorem geometric_sequence_a3_is_15 (q : ℝ) (a1 : ℝ) (a5 : ℝ) 
  (h1 : a1 = 3) (h2 : a5 = 75) (h_seq : ∀ n, a5 = geometric_sequence a1 q n) :
  geometric_sequence a1 q 3 = 15 :=
by 
  sorry

end geometric_sequence_a3_is_15_l116_116055


namespace tangent_line_at_x0_minimum_value_on_interval_minimum_value_on_interval_high_minimum_value_on_interval_mid_l116_116154

noncomputable def f (x a : ℝ) : ℝ := (x - a) * Real.exp x

theorem tangent_line_at_x0 (a : ℝ) (h : a = 2) : 
    (∃ m b : ℝ, (∀ x : ℝ, f x a = m * x + b) ∧ m = -1 ∧ b = -2) :=
by 
    sorry

theorem minimum_value_on_interval (a : ℝ) :
    (1 ≤ a) → (a ≤ 2) → f 1 a = (1 - a) * Real.exp 1 :=
by 
    sorry

theorem minimum_value_on_interval_high (a : ℝ) :
    (a ≥ 3) → f 2 a = (2 - a) * Real.exp 2 :=
by 
    sorry

theorem minimum_value_on_interval_mid (a : ℝ) :
    (2 < a) → (a < 3) → f (a - 1) a = -(Real.exp (a - 1)) :=
by 
    sorry

end tangent_line_at_x0_minimum_value_on_interval_minimum_value_on_interval_high_minimum_value_on_interval_mid_l116_116154


namespace legs_sum_of_right_triangle_with_hypotenuse_41_l116_116190

noncomputable def right_triangle_legs_sum (x : ℕ) : ℕ := x + (x + 1)

theorem legs_sum_of_right_triangle_with_hypotenuse_41 :
  ∃ x : ℕ, (x * x + (x + 1) * (x + 1) = 41 * 41) ∧ right_triangle_legs_sum x = 57 := by
sorry

end legs_sum_of_right_triangle_with_hypotenuse_41_l116_116190


namespace solve_inequality_prove_inequality_l116_116751

open Real

-- Problem 1: Solve the inequality
theorem solve_inequality (x : ℝ) : (x - 1) / (2 * x + 1) ≤ 0 ↔ (-1 / 2) < x ∧ x ≤ 1 :=
sorry

-- Problem 2: Prove the inequality given positive a, b, and c
theorem prove_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
    (a + b + c) * (1 / a + 1 / (b + c)) ≥ 4 :=
sorry

end solve_inequality_prove_inequality_l116_116751


namespace binom_difference_30_3_2_l116_116857

-- Define the binomial coefficient function.
def binom (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Theorem statement: binom(30, 3) - binom(30, 2) = 3625
theorem binom_difference_30_3_2 : binom 30 3 - binom 30 2 = 3625 := by
  sorry

end binom_difference_30_3_2_l116_116857


namespace solution_of_abs_square_eq_zero_l116_116879

-- Define the given conditions as hypotheses
variables {x y : ℝ}
theorem solution_of_abs_square_eq_zero (h : |x + 2| + (y - 1)^2 = 0) : x = -2 ∧ y = 1 :=
sorry

end solution_of_abs_square_eq_zero_l116_116879


namespace prism_diagonal_correct_l116_116117

open Real

noncomputable def prism_diagonal_1 := 2 * sqrt 6
noncomputable def prism_diagonal_2 := sqrt 66

theorem prism_diagonal_correct (length width : ℝ) (h1 : length = 8) (h2 : width = 4) :
  (prism_diagonal_1 = 2 * sqrt 6 ∧ prism_diagonal_2 = sqrt 66) :=
by
  sorry

end prism_diagonal_correct_l116_116117


namespace positive_difference_abs_eq_15_l116_116079

theorem positive_difference_abs_eq_15 :
  ∃ (x1 x2 : ℝ), (|x1 - 3| = 15) ∧ (|x2 - 3| = 15) ∧ (x1 ≠ x2) ∧ (|x1 - x2| = 30) :=
by
  sorry

end positive_difference_abs_eq_15_l116_116079


namespace find_width_of_room_l116_116844

theorem find_width_of_room (length room_cost cost_per_sqm total_cost width W : ℕ) 
  (h1 : length = 13)
  (h2 : cost_per_sqm = 12)
  (h3 : total_cost = 1872)
  (h4 : room_cost = length * W * cost_per_sqm)
  (h5 : total_cost = room_cost) : 
  W = 12 := 
by sorry

end find_width_of_room_l116_116844


namespace mark_has_seven_butterfingers_l116_116487

/-
  Mark has 12 candy bars in total between Mars bars, Snickers, and Butterfingers.
  He has 3 Snickers and 2 Mars bars.
  Prove that he has 7 Butterfingers.
-/

noncomputable def total_candy_bars : Nat := 12
noncomputable def snickers : Nat := 3
noncomputable def mars_bars : Nat := 2
noncomputable def butterfingers : Nat := total_candy_bars - (snickers + mars_bars)

theorem mark_has_seven_butterfingers : butterfingers = 7 := by
  sorry

end mark_has_seven_butterfingers_l116_116487


namespace toys_of_Jason_l116_116554

theorem toys_of_Jason (R J Jason : ℕ) 
  (hR : R = 1) 
  (hJ : J = R + 6) 
  (hJason : Jason = 3 * J) : 
  Jason = 21 :=
by
  sorry

end toys_of_Jason_l116_116554


namespace kolya_pays_90_rubles_l116_116136

theorem kolya_pays_90_rubles {x y : ℝ} 
  (h1 : x + 3 * y = 78) 
  (h2 : x + 8 * y = 108) :
  x + 5 * y = 90 :=
by sorry

end kolya_pays_90_rubles_l116_116136


namespace log_sin_decrease_interval_l116_116870

open Real

noncomputable def interval_of_decrease (x : ℝ) : Prop :=
  ∃ (k : ℤ), (k * π + π / 8 < x ∧ x ≤ k * π + 3 * π / 8)

theorem log_sin_decrease_interval (x : ℝ) :
  interval_of_decrease x ↔ ∃ (k : ℤ), (k * π + π / 8 < x ∧ x ≤ k * π + 3 * π / 8) :=
by
  sorry

end log_sin_decrease_interval_l116_116870


namespace calculate_taxi_fare_l116_116579

theorem calculate_taxi_fare :
  ∀ (f_80 f_120: ℝ), f_80 = 160 ∧ f_80 = 20 + (80 * (140/80)) →
                      f_120 = 20 + (120 * (140/80)) →
                      f_120 = 230 :=
by
  intro f_80 f_120
  rintro ⟨h80, h_proportional⟩ h_120
  sorry

end calculate_taxi_fare_l116_116579


namespace students_who_like_both_apple_pie_and_chocolate_cake_l116_116536

def total_students := 50
def students_who_like_apple_pie := 22
def students_who_like_chocolate_cake := 20
def students_who_like_neither := 10
def students_who_like_only_cookies := 5

theorem students_who_like_both_apple_pie_and_chocolate_cake :
  (students_who_like_apple_pie + students_who_like_chocolate_cake - (total_students - students_who_like_neither - students_who_like_only_cookies)) = 7 := 
by
  sorry

end students_who_like_both_apple_pie_and_chocolate_cake_l116_116536


namespace vector_line_equation_l116_116637

open Real

noncomputable def vector_projection (u v : ℝ × ℝ) : ℝ × ℝ :=
  let numer := (u.1 * v.1 + u.2 * v.2)
  let denom := (v.1 * v.1 + v.2 * v.2)
  (numer * v.1 / denom, numer * v.2 / denom)

theorem vector_line_equation (x y : ℝ) :
  vector_projection (x, y) (3, 4) = (-3, -4) → 
  y = -3 / 4 * x - 25 / 4 :=
  sorry

end vector_line_equation_l116_116637


namespace spadesuit_evaluation_l116_116727

-- Define the operation
def spadesuit (a b : ℝ) : ℝ := (a + b) * (a - b)

-- The theorem to prove
theorem spadesuit_evaluation : spadesuit 4 (spadesuit 5 (-2)) = -425 :=
by
  sorry

end spadesuit_evaluation_l116_116727


namespace original_calculation_l116_116314

theorem original_calculation
  (x : ℝ)
  (h : ((x * 3) + 14) * 2 = 946) :
  ((x / 3) + 14) * 2 = 130 :=
sorry

end original_calculation_l116_116314


namespace period_start_time_l116_116963

/-- A period of time had 4 hours of rain and 5 hours without rain, ending at 5 pm. 
Prove that the period started at 8 am. -/
theorem period_start_time :
  let end_time := 17 -- 5 pm in 24-hour format
  let rainy_hours := 4
  let non_rainy_hours := 5
  let total_hours := rainy_hours + non_rainy_hours
  let start_time := end_time - total_hours
  start_time = 8 :=
by
  sorry

end period_start_time_l116_116963


namespace scrabble_letter_values_l116_116814

-- Definitions based on conditions
def middle_letter_value : ℕ := 8
def final_score : ℕ := 30

-- The theorem we need to prove
theorem scrabble_letter_values (F T : ℕ)
  (h1 : 3 * (F + middle_letter_value + T) = final_score) :
  F = 1 ∧ T = 1 :=
sorry

end scrabble_letter_values_l116_116814


namespace min_abs_sum_l116_116125

theorem min_abs_sum (x : ℝ) : ∃ x : ℝ, (∀ y, abs (y + 3) + abs (y - 2) ≥ abs (x + 3) + abs (x - 2)) ∧ (abs (x + 3) + abs (x - 2) = 5) := sorry

end min_abs_sum_l116_116125


namespace food_last_after_join_l116_116152

-- Define the conditions
def initial_men := 760
def additional_men := 2280
def initial_days := 22
def days_before_join := 2
def initial_food := initial_men * initial_days
def remaining_food := initial_food - (initial_men * days_before_join)
def total_men := initial_men + additional_men

-- Define the goal to prove
theorem food_last_after_join :
  (remaining_food / total_men) = 5 :=
by
  sorry

end food_last_after_join_l116_116152


namespace reformulate_and_find_product_l116_116320

theorem reformulate_and_find_product (a b x y : ℝ)
  (h : a^8 * x * y - a^7 * y - a^6 * x = a^5 * (b^5 - 2)) :
  ∃ m' n' p' : ℤ, (a^m' * x - a^n') * (a^p' * y - a^3) = a^5 * b^5 ∧ m' * n' * p' = 48 :=
by
  sorry

end reformulate_and_find_product_l116_116320


namespace evaluate_exponentiation_l116_116574

theorem evaluate_exponentiation : (3 ^ 3) ^ 4 = 531441 := by
  sorry

end evaluate_exponentiation_l116_116574


namespace sum_of_roots_l116_116169

theorem sum_of_roots (r s t : ℝ) (hroots : 3 * (r^3 + s^3 + t^3) + 9 * (r^2 + s^2 + t^2) - 36 * (r + s + t) + 12 = 0) :
  r + s + t = -3 :=
sorry

end sum_of_roots_l116_116169


namespace speed_of_car_B_is_correct_l116_116219

def carB_speed : ℕ := 
  let speedA := 50 -- Car A's speed in km/hr
  let timeA := 6 -- Car A's travel time in hours
  let ratio := 3 -- The ratio of distances between Car A and Car B
  let distanceA := speedA * timeA -- Calculate Car A's distance
  let timeB := 1 -- Car B's travel time in hours
  let distanceB := distanceA / ratio -- Calculate Car B's distance
  distanceB / timeB -- Calculate Car B's speed

theorem speed_of_car_B_is_correct : carB_speed = 100 := by
  sorry

end speed_of_car_B_is_correct_l116_116219


namespace value_added_to_each_number_is_11_l116_116521

-- Given definitions and conditions
def initial_average : ℝ := 40
def number_count : ℕ := 15
def new_average : ℝ := 51

-- Mathematically equivalent proof statement
theorem value_added_to_each_number_is_11 (x : ℝ) 
  (h1 : number_count * initial_average = 600)
  (h2 : (600 + number_count * x) / number_count = new_average) : 
  x = 11 := 
by 
  sorry

end value_added_to_each_number_is_11_l116_116521


namespace hexagon_chord_problem_l116_116954

-- Define the conditions of the problem
structure Hexagon :=
  (circumcircle : Type*)
  (inscribed : Prop)
  (AB BC CD : ℕ)
  (DE EF FA : ℕ)
  (chord_length_fraction_form : ℚ) 

-- Define the unique problem from given conditions and correct answer
theorem hexagon_chord_problem (hex : Hexagon) 
  (h1 : hex.inscribed)
  (h2 : hex.AB = 3) (h3 : hex.BC = 3) (h4 : hex.CD = 3)
  (h5 : hex.DE = 5) (h6 : hex.EF = 5) (h7 : hex.FA = 5)
  (h8 : hex.chord_length_fraction_form = 360 / 49) :
  let m := 360
  let n := 49
  m + n = 409 :=
by
  sorry

end hexagon_chord_problem_l116_116954


namespace division_remainder_l116_116214

theorem division_remainder :
  let p := fun x : ℝ => 5 * x^4 - 9 * x^3 + 3 * x^2 - 7 * x - 30
  let q := 3 * x - 9
  p 3 % q = 138 :=
by
  sorry

end division_remainder_l116_116214


namespace g_max_value_l116_116341

def g (n : ℕ) : ℕ :=
if n < 15 then n + 15 else g (n - 7)

theorem g_max_value : ∃ N : ℕ, ∀ n : ℕ, g n ≤ N ∧ N = 29 := 
by 
  sorry

end g_max_value_l116_116341


namespace sum_of_ages_l116_116598

-- Define ages of Kiana and her twin brothers
variables (kiana_age : ℕ) (twin_age : ℕ)

-- Define conditions
def age_product_condition : Prop := twin_age * twin_age * kiana_age = 162
def age_less_than_condition : Prop := kiana_age < 10
def twins_older_condition : Prop := twin_age > kiana_age

-- The main problem statement
theorem sum_of_ages (h1 : age_product_condition twin_age kiana_age) (h2 : age_less_than_condition kiana_age) (h3 : twins_older_condition twin_age kiana_age) :
  twin_age * 2 + kiana_age = 20 :=
sorry

end sum_of_ages_l116_116598


namespace quadratic_function_coefficient_nonzero_l116_116383

theorem quadratic_function_coefficient_nonzero (m : ℝ) :
  (y = (m + 2) * x * x + m) ↔ (m ≠ -2 ∧ (m^2 + m - 2 = 0) → m = 1) := by
  sorry

end quadratic_function_coefficient_nonzero_l116_116383


namespace greatest_integer_less_than_or_equal_to_frac_l116_116216

theorem greatest_integer_less_than_or_equal_to_frac (a b c d : ℝ)
  (ha : a = 4^100) (hb : b = 3^100) (hc : c = 4^95) (hd : d = 3^95) :
  ⌊(a + b) / (c + d)⌋ = 1023 := 
by
  sorry

end greatest_integer_less_than_or_equal_to_frac_l116_116216


namespace andrew_eggs_count_l116_116446

def cost_of_toast (num_toasts : ℕ) : ℕ :=
  num_toasts * 1

def cost_of_eggs (num_eggs : ℕ) : ℕ :=
  num_eggs * 3

def total_cost (num_toasts : ℕ) (num_eggs : ℕ) : ℕ :=
  cost_of_toast num_toasts + cost_of_eggs num_eggs

theorem andrew_eggs_count (E : ℕ) (H1 : total_cost 2 2 = 8)
                       (H2 : total_cost 1 E + 8 = 15) : E = 2 := by
  sorry

end andrew_eggs_count_l116_116446


namespace joan_gave_apples_l116_116049

theorem joan_gave_apples (initial_apples : ℕ) (remaining_apples : ℕ) (given_apples : ℕ) 
  (h1 : initial_apples = 43) (h2 : remaining_apples = 16) : given_apples = 27 :=
by
  -- Show that given_apples is obtained by subtracting remaining_apples from initial_apples
  sorry

end joan_gave_apples_l116_116049


namespace problem_statement_l116_116146

variables {x y z w p q : Prop}

theorem problem_statement (h1 : x = y → z ≠ w) (h2 : z = w → p ≠ q) : x ≠ y → p ≠ q :=
by
  sorry

end problem_statement_l116_116146


namespace consecutive_green_balls_l116_116768

theorem consecutive_green_balls : ∃ (fill_ways : ℕ), fill_ways = 21 ∧ 
  (∃ (boxes : Fin 6 → Bool), 
    (∀ i, boxes i = true → 
      (∀ j, boxes j = true → (i ≤ j ∨ j ≤ i)) ∧ 
      ∃ k, boxes k = true)) :=
by
  sorry

end consecutive_green_balls_l116_116768


namespace simplify_expr_l116_116091

theorem simplify_expr (x y : ℝ) : 
  (3 * x - 2 * y - 4) * (x + y + 5) - (x + 2 * y + 5) * (3 * x - y - 1) = -4 * x * y - 3 * x - 7 * y - 15 := 
by 
  sorry

end simplify_expr_l116_116091


namespace range_of_a_l116_116729

noncomputable def f (a b x : ℝ) : ℝ := Real.log x - (1/2) * a * x^2 - b * x

theorem range_of_a (a b x : ℝ) (h1 : ∀ x > 0, (1/x) - a * x - b ≠ 0) (h2 : ∀ x > 0, x = 1 → (1/x) - a * x - b = 0) : 
  (1 - a) = b ∧ a > -1 :=
by
  sorry

end range_of_a_l116_116729


namespace topics_assignment_l116_116391

theorem topics_assignment (students groups arrangements : ℕ) (h1 : students = 6) (h2 : groups = 3) (h3 : arrangements = 90) :
  let T := arrangements / (students * (students - 1) / 2 * (4 * 3 / 2 * 1))
  T = 1 :=
by
  sorry

end topics_assignment_l116_116391


namespace single_cone_scoops_l116_116377

theorem single_cone_scoops (banana_split_scoops : ℕ) (waffle_bowl_scoops : ℕ) (single_cone_scoops : ℕ) (double_cone_scoops : ℕ)
  (h1 : banana_split_scoops = 3 * single_cone_scoops)
  (h2 : waffle_bowl_scoops = banana_split_scoops + 1)
  (h3 : double_cone_scoops = 2 * single_cone_scoops)
  (h4 : single_cone_scoops + banana_split_scoops + waffle_bowl_scoops + double_cone_scoops = 10) :
  single_cone_scoops = 1 :=
by
  sorry

end single_cone_scoops_l116_116377


namespace num_vec_a_exists_l116_116121

-- Define the vectors and the conditions
def vec_a (x y : ℝ) : (ℝ × ℝ) := (x, y)
def vec_b (x y : ℝ) : (ℝ × ℝ) := (x^2, y^2)
def vec_c : (ℝ × ℝ) := (1, 1)

-- Define the dot product
def dot_prod (v1 v2 : ℝ × ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

-- Define the conditions
def cond_1 (x y : ℝ) : Prop := (x + y = 1)
def cond_2 (x y : ℝ) : Prop := (x^2 / 4 + (1 - x)^2 / 9 = 1)

-- The proof problem statement
theorem num_vec_a_exists : ∃! (x y : ℝ), cond_1 x y ∧ cond_2 x y := by
  sorry

end num_vec_a_exists_l116_116121


namespace heavy_rain_duration_l116_116304

-- Define the conditions as variables and constants
def initial_volume := 100 -- Initial volume in liters
def final_volume := 280   -- Final volume in liters
def flow_rate := 2        -- Flow rate in liters per minute

-- Define the duration query as a theorem to be proved
theorem heavy_rain_duration : 
  (final_volume - initial_volume) / flow_rate = 90 := 
by
  sorry

end heavy_rain_duration_l116_116304


namespace multiple_of_area_l116_116118

-- Define the given conditions
def perimeter (s : ℝ) : ℝ := 4 * s
def area (s : ℝ) : ℝ := s * s

theorem multiple_of_area (m s a p : ℝ) 
  (h1 : p = perimeter s)
  (h2 : a = area s)
  (h3 : m * a = 10 * p + 45)
  (h4 : p = 36) : m = 5 :=
by 
  sorry

end multiple_of_area_l116_116118


namespace men_dropped_out_l116_116400

theorem men_dropped_out (x : ℕ) : 
  (∀ (days_half days_full men men_remaining : ℕ),
    days_half = 15 ∧ days_full = 25 ∧ men = 5 ∧ men_remaining = men - x ∧ 
    (men * (2 * days_half)) = ((men_remaining) * days_full)) -> x = 1 :=
by
  intros h
  sorry

end men_dropped_out_l116_116400


namespace packs_of_cake_l116_116964

-- Given conditions
def total_grocery_packs : ℕ := 27
def cookie_packs : ℕ := 23

-- Question: How many packs of cake did Lucy buy?
-- Mathematically equivalent problem: Proving that cake_packs is 4
theorem packs_of_cake : (total_grocery_packs - cookie_packs) = 4 :=
by
  -- Proof goes here. Using sorry to skip the proof.
  sorry

end packs_of_cake_l116_116964


namespace steve_annual_salary_l116_116522

variable (S : ℝ)

theorem steve_annual_salary :
  (0.70 * S - 800 = 27200) → (S = 40000) :=
by
  intro h
  sorry

end steve_annual_salary_l116_116522


namespace tan_square_B_eq_tan_A_tan_C_range_l116_116126

theorem tan_square_B_eq_tan_A_tan_C_range (A B C : ℝ) (h_triangle : 0 < A ∧ A < π / 2 ∧ 0 < B ∧ B < π / 2 ∧ 0 < C ∧ C < π / 2 ∧ A + B + C = π) 
  (h_tan : Real.tan B * Real.tan B = Real.tan A * Real.tan C) : (π / 3) ≤ B ∧ B < (π / 2) :=
by
  sorry

end tan_square_B_eq_tan_A_tan_C_range_l116_116126


namespace green_peaches_in_each_basket_l116_116675

theorem green_peaches_in_each_basket (G : ℕ) 
  (h1 : ∀ B : ℕ, B = 15) 
  (h2 : ∀ R : ℕ, R = 19) 
  (h3 : ∀ P : ℕ, P = 345) 
  (h_eq : 345 = 15 * (19 + G)) : 
  G = 4 := by
  sorry

end green_peaches_in_each_basket_l116_116675


namespace new_fish_received_l116_116899

def initial_fish := 14
def added_fish := 2
def eaten_fish := 6
def final_fish := 11

def current_fish := initial_fish + added_fish - eaten_fish
def returned_fish := 2
def exchanged_fish := final_fish - current_fish

theorem new_fish_received : exchanged_fish = 1 := by
  sorry

end new_fish_received_l116_116899


namespace length_of_path_along_arrows_l116_116206

theorem length_of_path_along_arrows (s : List ℝ) (h : s.sum = 73) :
  (3 * s.sum = 219) :=
by
  sorry

end length_of_path_along_arrows_l116_116206


namespace find_larger_number_l116_116798

theorem find_larger_number (L S : ℕ) (h1 : L - S = 2500) (h2 : L = 6 * S + 15) : L = 2997 :=
sorry

end find_larger_number_l116_116798


namespace solve_equation_l116_116862

theorem solve_equation (x : ℝ) (h : x ≠ 1) : 
  1 / (x - 1) + 1 = 3 / (2 * x - 2) ↔ x = 3 / 2 := by
  sorry

end solve_equation_l116_116862


namespace cone_base_radius_l116_116991

theorem cone_base_radius (angle : ℝ) (sector_radius : ℝ) (base_radius : ℝ) 
(h1 : angle = 216)
(h2 : sector_radius = 15)
(h3 : 2 * π * base_radius = (3 / 5) * 2 * π * sector_radius) :
base_radius = 9 := 
sorry

end cone_base_radius_l116_116991


namespace distinct_arrangements_BOOKKEEPER_l116_116722

theorem distinct_arrangements_BOOKKEEPER :
  let n := 9
  let nO := 2
  let nK := 2
  let nE := 3
  ∃ arrangements : ℕ,
  arrangements = Nat.factorial n / (Nat.factorial nO * Nat.factorial nK * Nat.factorial nE) ∧
  arrangements = 15120 :=
by { sorry }

end distinct_arrangements_BOOKKEEPER_l116_116722


namespace negation_of_divisible_by_2_even_l116_116347

theorem negation_of_divisible_by_2_even :
  (¬ ∀ n : ℤ, (∃ k, n = 2 * k) → (∃ k, n = 2 * k ∧ n % 2 = 0)) ↔
  ∃ n : ℤ, (∃ k, n = 2 * k) ∧ ¬ (n % 2 = 0) :=
by
  sorry

end negation_of_divisible_by_2_even_l116_116347


namespace min_value_x_l116_116631

theorem min_value_x (a : ℝ) (h : ∀ a > 0, x^2 ≤ 1 + a) : ∃ x, ∀ a > 0, -1 ≤ x ∧ x ≤ 1 := 
sorry

end min_value_x_l116_116631


namespace combined_area_of_tracts_l116_116908

theorem combined_area_of_tracts :
  let length1 := 300
  let width1 := 500
  let length2 := 250
  let width2 := 630
  let area1 := length1 * width1
  let area2 := length2 * width2
  let combined_area := area1 + area2
  combined_area = 307500 :=
by
  sorry

end combined_area_of_tracts_l116_116908


namespace abs_eq_two_implies_l116_116133

theorem abs_eq_two_implies (x : ℝ) (h : |x - 3| = 2) : x = 5 ∨ x = 1 := 
sorry

end abs_eq_two_implies_l116_116133


namespace exactly_one_pair_probability_l116_116699

def four_dice_probability : ℚ :=
  sorry  -- Here we skip the actual computation and proof

theorem exactly_one_pair_probability : four_dice_probability = 5/9 := by {
  -- Placeholder for proof, explanation, and calculation
  sorry
}

end exactly_one_pair_probability_l116_116699


namespace shuai_fen_ratio_l116_116114

theorem shuai_fen_ratio 
  (C : ℕ) (B_and_D : ℕ) (a : ℕ) (x : ℚ) 
  (hC : C = 36) (hB_and_D : B_and_D = 75) :
  (x = 0.25) ∧ (a = 175) := 
by {
  -- This is where the proof steps would go
  sorry
}

end shuai_fen_ratio_l116_116114


namespace no_prime_numbers_divisible_by_91_l116_116757

-- Define the concept of a prime number.
def is_prime (n : ℕ) : Prop :=
  1 < n ∧ (∀ m : ℕ, m ∣ n → m = 1 ∨ m = n)

-- Define the factors of 91.
def factors_of_91 (n : ℕ) : Prop :=
  n = 7 ∨ n = 13

-- State the problem formally: there are no prime numbers divisible by 91.
theorem no_prime_numbers_divisible_by_91 :
  ∀ p : ℕ, is_prime p → ¬ (91 ∣ p) :=
by
  intros p prime_p div91
  sorry

end no_prime_numbers_divisible_by_91_l116_116757


namespace car_win_probability_l116_116119

-- Definitions from conditions
def total_cars : ℕ := 12
def p_X : ℚ := 1 / 6
def p_Y : ℚ := 1 / 10
def p_Z : ℚ := 1 / 8

-- Proof statement: The probability that one of the cars X, Y, or Z will win is 47/120
theorem car_win_probability : p_X + p_Y + p_Z = 47 / 120 := by
  sorry

end car_win_probability_l116_116119


namespace servant_cash_received_l116_116878

theorem servant_cash_received (salary_cash : ℕ) (turban_value : ℕ) (months_worked : ℕ) (total_months : ℕ)
  (h_salary_cash : salary_cash = 90) (h_turban_value : turban_value = 70) (h_months_worked : months_worked = 9)
  (h_total_months : total_months = 12) : 
  salary_cash * months_worked / total_months + (turban_value * months_worked / total_months) - turban_value = 50 := by
sorry

end servant_cash_received_l116_116878


namespace intersection_of_sets_l116_116323

def A : Set ℝ := {0, 1, 2, 3}
def B : Set ℝ := {1, 3, 4}
def C : Set ℝ := {x | x > 2 ∨ x < 1}

theorem intersection_of_sets :
  (A ∪ B) ∩ C = {0, 3, 4} :=
by
  sorry

end intersection_of_sets_l116_116323


namespace prob1_prob2_prob3_prob4_l116_116648

theorem prob1 : (-20) + (-14) - (-18) - 13 = -29 := sorry

theorem prob2 : (-24) * (-1/2 + 3/4 - 1/3) = 2 := sorry

theorem prob3 : (- (49 + 24/25)) * 10 = -499.6 := sorry

theorem prob4 :
  -3^2 + ((-1/3) * (-3) - 8/5 / 2^2) = -8 - 2/5 := sorry

end prob1_prob2_prob3_prob4_l116_116648


namespace ellipse_ratio_squared_l116_116376

theorem ellipse_ratio_squared (a b c : ℝ) 
    (h1 : b / a = a / c) 
    (h2 : c^2 = a^2 - b^2) : (b / a)^2 = 1 / 2 :=
by
  sorry

end ellipse_ratio_squared_l116_116376


namespace cans_of_soda_l116_116902

variable (T R E : ℝ)

theorem cans_of_soda (hT: T > 0) (hR: R > 0) (hE: E > 0) : 5 * E * T / R = (5 * E) / R * T :=
by
  sorry

end cans_of_soda_l116_116902


namespace fish_worth_rice_l116_116024

variables (f l r : ℝ)

-- Conditions based on the problem statement
def fish_for_bread : Prop := 3 * f = 2 * l
def bread_for_rice : Prop := l = 4 * r

-- Statement to be proven
theorem fish_worth_rice (h₁ : fish_for_bread f l) (h₂ : bread_for_rice l r) : f = (8 / 3) * r :=
  sorry

end fish_worth_rice_l116_116024


namespace cary_net_calorie_deficit_is_250_l116_116072

-- Define the conditions
def miles_walked : ℕ := 3
def candy_bar_calories : ℕ := 200
def calories_per_mile : ℕ := 150

-- Define the function to calculate total calories burned
def total_calories_burned (miles : ℕ) (calories_per_mile : ℕ) : ℕ :=
  miles * calories_per_mile

-- Define the function to calculate net calorie deficit
def net_calorie_deficit (total_calories : ℕ) (candy_calories : ℕ) : ℕ :=
  total_calories - candy_calories

-- The statement to be proven
theorem cary_net_calorie_deficit_is_250 :
  net_calorie_deficit (total_calories_burned miles_walked calories_per_mile) candy_bar_calories = 250 :=
  by sorry

end cary_net_calorie_deficit_is_250_l116_116072


namespace bread_left_in_pond_l116_116326

theorem bread_left_in_pond (total_bread : ℕ) 
                           (half_bread_duck : ℕ)
                           (second_duck_bread : ℕ)
                           (third_duck_bread : ℕ)
                           (total_bread_thrown : total_bread = 100)
                           (half_duck_eats : half_bread_duck = total_bread / 2)
                           (second_duck_eats : second_duck_bread = 13)
                           (third_duck_eats : third_duck_bread = 7) :
                           total_bread - (half_bread_duck + second_duck_bread + third_duck_bread) = 30 :=
    by
    sorry

end bread_left_in_pond_l116_116326


namespace specific_natural_numbers_expr_l116_116310

theorem specific_natural_numbers_expr (a b c : ℕ) 
  (h1 : Nat.gcd a b = 1) (h2 : Nat.gcd b c = 1) (h3 : Nat.gcd c a = 1) : 
  ∃ n : ℕ, (n = 7 ∨ n = 8) ∧ (n = (a + b) / c + (b + c) / a + (c + a) / b) :=
by sorry

end specific_natural_numbers_expr_l116_116310


namespace negation_of_p_l116_116417

theorem negation_of_p (p : Prop) :
  (¬ (∀ (a : ℝ), a ≥ 0 → a^4 + a^2 ≥ 0)) ↔ (∃ (a : ℝ), a ≥ 0 ∧ a^4 + a^2 < 0) := 
by
  sorry

end negation_of_p_l116_116417


namespace find_g2_l116_116901

theorem find_g2 (g : ℝ → ℝ) (h : ∀ x : ℝ, x ≠ 0 → g x - 3 * g (1 / x) = 3 ^ x) : 
  g 2 = (9 - 3 * Real.sqrt 3) / 8 := 
sorry

end find_g2_l116_116901


namespace leo_weight_l116_116607

-- Definitions from the conditions
variable (L K J M : ℝ)

-- Conditions 
def condition1 : Prop := L + 15 = 1.60 * K
def condition2 : Prop := L + 15 = 0.40 * J
def condition3 : Prop := J = K + 25
def condition4 : Prop := M = K - 20
def condition5 : Prop := L + K + J + M = 350

-- Final statement to prove based on the conditions
theorem leo_weight (h1 : condition1 L K) (h2 : condition2 L J) (h3 : condition3 J K) 
                   (h4 : condition4 M K) (h5 : condition5 L K J M) : L = 110.22 :=
by 
  sorry

end leo_weight_l116_116607


namespace length_of_solution_set_l116_116977

variable {a b : ℝ}

theorem length_of_solution_set (h : ∀ x : ℝ, a ≤ 3 * x + 4 ∧ 3 * x + 4 ≤ b → 12 = (b - a) / 3) : b - a = 36 :=
sorry

end length_of_solution_set_l116_116977


namespace expected_value_is_150_l116_116174

noncomputable def expected_value_of_winnings : ℝ :=
  let p := (1:ℝ)/8
  let winnings := [0, 2, 3, 5, 7]
  let losses := [4, 6]
  let extra := 5
  let win_sum := (winnings.sum : ℝ)
  let loss_sum := (losses.sum : ℝ)
  let E := p * 0 + p * win_sum - p * loss_sum + p * extra
  E

theorem expected_value_is_150 : expected_value_of_winnings = 1.5 := 
by sorry

end expected_value_is_150_l116_116174


namespace todd_savings_l116_116941

-- Define the initial conditions
def original_price : ℝ := 125
def sale_discount : ℝ := 0.20
def coupon : ℝ := 10
def card_discount : ℝ := 0.10

-- Define the resulting values after applying discounts
def sale_price := original_price * (1 - sale_discount)
def after_coupon := sale_price - coupon
def final_price := after_coupon * (1 - card_discount)

-- Define the total savings
def savings := original_price - final_price

-- The proof statement
theorem todd_savings : savings = 44 := by
  sorry

end todd_savings_l116_116941


namespace probability_at_least_eight_stayed_correct_l116_116350

noncomputable def probability_at_least_eight_stayed (n : ℕ) (c : ℕ) (p : ℚ) : ℚ :=
  let certain_count := c
  let unsure_count := n - c
  let k := 3
  let prob_eight := 
    (Nat.choose unsure_count k : ℚ) * (p^k) * ((1 - p)^(unsure_count - k))
  let prob_nine := p^unsure_count
  prob_eight + prob_nine

theorem probability_at_least_eight_stayed_correct :
  probability_at_least_eight_stayed 9 5 (3/7) = 513 / 2401 :=
by
  sorry

end probability_at_least_eight_stayed_correct_l116_116350


namespace y_days_worked_l116_116181

theorem y_days_worked 
  ( W : ℝ )
  ( x_rate : ℝ := W / 21 )
  ( y_rate : ℝ := W / 15 )
  ( d : ℝ )
  ( y_work_done : ℝ := d * y_rate )
  ( x_work_done_after_y_leaves : ℝ := 14 * x_rate )
  ( total_work_done : y_work_done + x_work_done_after_y_leaves = W ) :
  d = 5 := 
sorry

end y_days_worked_l116_116181


namespace shaded_areas_equal_l116_116002

theorem shaded_areas_equal (φ : ℝ) (hφ1 : 0 < φ) (hφ2 : φ < π / 4) : 
  (Real.tan φ) = 2 * φ :=
sorry

end shaded_areas_equal_l116_116002


namespace problem_a_problem_b_unique_solution_l116_116604

-- Problem (a)

theorem problem_a (a b c n : ℤ) (hnat : 0 ≤ n) (h : a * n^2 + b * n + c = 0) : n ∣ c :=
sorry

-- Problem (b)

theorem problem_b_unique_solution : ∀ n : ℕ, n^5 - 2 * n^4 - 7 * n^2 - 7 * n + 3 = 0 → n = 3 :=
sorry

end problem_a_problem_b_unique_solution_l116_116604


namespace algebra_problem_l116_116780

noncomputable def expression (a b : ℝ) : ℝ :=
  (3 * a + b / 3)⁻¹ * ((3 * a)⁻¹ + (b / 3)⁻¹)

theorem algebra_problem (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  expression a b = (a * b)⁻¹ :=
by
  sorry

end algebra_problem_l116_116780


namespace simplify_expression_l116_116617

theorem simplify_expression (b : ℝ) (h : b ≠ 1 / 2) : 1 - (2 / (1 + (b / (1 - 2 * b)))) = (3 * b - 1) / (1 - b) :=
by
    sorry

end simplify_expression_l116_116617


namespace opposite_of_B_is_I_l116_116740

inductive Face
| A | B | C | D | E | F | G | H | I

open Face

def opposite_face (f : Face) : Face :=
  match f with
  | A => G
  | B => I
  | C => H
  | D => F
  | E => E
  | F => F
  | G => A
  | H => C
  | I => B

theorem opposite_of_B_is_I : opposite_face B = I :=
  by
    sorry

end opposite_of_B_is_I_l116_116740


namespace equal_costs_at_45_students_l116_116060

def ticket_cost_option1 (x : ℕ) : ℝ :=
  x * 30 * 0.8

def ticket_cost_option2 (x : ℕ) : ℝ :=
  (x - 5) * 30 * 0.9

theorem equal_costs_at_45_students : ∀ x : ℕ, ticket_cost_option1 x = ticket_cost_option2 x ↔ x = 45 := 
by
  intro x
  sorry

end equal_costs_at_45_students_l116_116060


namespace directrix_of_parabola_l116_116490

theorem directrix_of_parabola (x y : ℝ) (h : y = x^2) : y = -1 / 4 :=
sorry

end directrix_of_parabola_l116_116490


namespace max_5_cent_coins_l116_116792

theorem max_5_cent_coins :
  ∃ (x y z : ℕ), 
  x + y + z = 25 ∧ 
  x + 2*y + 5*z = 60 ∧
  (∀ y' z' : ℕ, y' + 4*z' = 35 → z' ≤ 8) ∧
  y + 4*z = 35 ∧ z = 8 := 
sorry

end max_5_cent_coins_l116_116792


namespace alex_not_read_probability_l116_116911

def probability_reads : ℚ := 5 / 8
def probability_not_reads : ℚ := 3 / 8

theorem alex_not_read_probability : (1 - probability_reads) = probability_not_reads := 
by
  sorry

end alex_not_read_probability_l116_116911


namespace scientific_notation_of_10760000_l116_116558

theorem scientific_notation_of_10760000 : 
  (10760000 : ℝ) = 1.076 * 10^7 := 
sorry

end scientific_notation_of_10760000_l116_116558


namespace units_digit_of_45_pow_125_plus_7_pow_87_l116_116528

theorem units_digit_of_45_pow_125_plus_7_pow_87 :
  (45 ^ 125 + 7 ^ 87) % 10 = 8 :=
by
  -- sorry to skip the proof
  sorry

end units_digit_of_45_pow_125_plus_7_pow_87_l116_116528


namespace team_combinations_l116_116239

/-- 
The math club at Walnutridge High School has five girls and seven boys. 
How many different teams, comprising two girls and two boys, can be formed 
if one boy on each team must also be designated as the team leader?
-/
theorem team_combinations (girls boys : ℕ) (h_girls : girls = 5) (h_boys : boys = 7) :
  ∃ n, n = 420 :=
by
  sorry

end team_combinations_l116_116239


namespace original_cost_price_l116_116407

theorem original_cost_price (C : ℝ) : 
  (0.89 * C * 1.20 = 54000) → C = 50561.80 :=
by
  sorry

end original_cost_price_l116_116407


namespace pipe_A_time_to_fill_l116_116228

theorem pipe_A_time_to_fill (T_B : ℝ) (T_combined : ℝ) (T_A : ℝ): 
  T_B = 75 → T_combined = 30 → 
  (1 / T_B + 1 / T_A = 1 / T_combined) → T_A = 50 :=
by
  -- Placeholder proof
  intro h1 h2 h3
  have h4 : T_B = 75 := h1
  have h5 : T_combined = 30 := h2
  have h6 : 1 / T_B + 1 / T_A = 1 / T_combined := h3
  sorry

end pipe_A_time_to_fill_l116_116228


namespace destiny_cookies_divisible_l116_116115

theorem destiny_cookies_divisible (C : ℕ) (h : C % 6 = 0) : ∃ k : ℕ, C = 6 * k :=
by {
  sorry
}

end destiny_cookies_divisible_l116_116115


namespace weekly_earnings_l116_116430

theorem weekly_earnings :
  let hours_Monday := 2
  let minutes_Tuesday := 75
  let start_Thursday := (15, 10) -- 3:10 PM in (hour, minute) format
  let end_Thursday := (17, 45) -- 5:45 PM in (hour, minute) format
  let minutes_Saturday := 45

  let pay_rate_weekday := 4 -- \$4 per hour
  let pay_rate_weekend := 5 -- \$5 per hour

  -- Convert time to hours
  let hours_Tuesday := minutes_Tuesday / 60.0
  let Thursday_work_minutes := (end_Thursday.1 * 60 + end_Thursday.2) - (start_Thursday.1 * 60 + start_Thursday.2)
  let hours_Thursday := Thursday_work_minutes / 60.0
  let hours_Saturday := minutes_Saturday / 60.0

  -- Calculate earnings
  let earnings_Monday := hours_Monday * pay_rate_weekday
  let earnings_Tuesday := hours_Tuesday * pay_rate_weekday
  let earnings_Thursday := hours_Thursday * pay_rate_weekday
  let earnings_Saturday := hours_Saturday * pay_rate_weekend

  -- Total earnings
  let total_earnings := earnings_Monday + earnings_Tuesday + earnings_Thursday + earnings_Saturday

  total_earnings = 27.08 := by sorry

end weekly_earnings_l116_116430


namespace sum_of_solutions_of_absolute_value_l116_116816

theorem sum_of_solutions_of_absolute_value (n1 n2 : ℚ) (h1 : 3*n1 - 8 = 5) (h2 : 3*n2 - 8 = -5) :
  n1 + n2 = 16 / 3 :=
by
  sorry

end sum_of_solutions_of_absolute_value_l116_116816


namespace find_k_max_product_l116_116095

theorem find_k_max_product : 
  (∃ k : ℝ, (3 : ℝ) * (x ^ 2) - 4 * x + k = 0 ∧ 16 - 12 * k ≥ 0 ∧ (∀ x1 x2 : ℝ, x1 * x2 = k / 3 → x1 + x2 = 4 / 3 → x1 * x2 ≤ (2 / 3) ^ 2)) →
  k = 4 / 3 :=
by 
  sorry

end find_k_max_product_l116_116095


namespace percent_calculation_l116_116912

theorem percent_calculation (x : ℝ) (h : 0.30 * 0.40 * x = 24) : 0.20 * 0.60 * x = 24 := 
by
  sorry

end percent_calculation_l116_116912


namespace largest_base5_three_digits_is_124_l116_116863

noncomputable def largest_base5_three_digits_to_base10 : ℕ :=
  4 * 5^2 + 4 * 5^1 + 4 * 5^0

theorem largest_base5_three_digits_is_124 :
  largest_base5_three_digits_to_base10 = 124 :=
by
  -- calculating 4 * 5^2 + 4 * 5^1 + 4 * 5^0 = 124
  sorry

end largest_base5_three_digits_is_124_l116_116863


namespace xiao_li_place_l116_116576

def guess_A (place : String) : Prop :=
  place ≠ "first" ∧ place ≠ "second"

def guess_B (place : String) : Prop :=
  place ≠ "first" ∧ place = "third"

def guess_C (place : String) : Prop :=
  place ≠ "third" ∧ place = "first"

def correct_guesses (guess : String → Prop) (place : String) : Prop :=
  guess place

def half_correct_guesses (guess : String → Prop) (place : String) : Prop :=
  (guess "first" = (place = "first")) ∨
  (guess "second" = (place = "second")) ∨
  (guess "third" = (place = "third"))

theorem xiao_li_place :
  ∃ (place : String),
  (correct_guesses guess_A place ∧
   half_correct_guesses guess_B place ∧
   ¬ correct_guesses guess_C place) ∨
  (correct_guesses guess_B place ∧
   half_correct_guesses guess_A place ∧
   ¬ correct_guesses guess_C place) ∨
  (correct_guesses guess_C place ∧
   half_correct_guesses guess_A place ∧
   ¬ correct_guesses guess_B place) ∨
  (correct_guesses guess_C place ∧
   half_correct_guesses guess_B place ∧
   ¬ correct_guesses guess_A place) :=
sorry

end xiao_li_place_l116_116576


namespace find_second_expression_l116_116317

theorem find_second_expression (a : ℕ) (x : ℕ) 
  (h1 : (2 * a + 16 + x) / 2 = 74) (h2 : a = 28) : x = 76 := 
by
  sorry

end find_second_expression_l116_116317


namespace problem1_problem2_l116_116362

variable (α : ℝ) (tan_alpha_eq_one_over_three : Real.tan α = 1 / 3)

-- For the first proof problem
theorem problem1 : (Real.sin α + 3 * Real.cos α) / (Real.sin α - Real.cos α) = -5 :=
by sorry

-- For the second proof problem
theorem problem2 : Real.cos α ^ 2 - Real.sin (2 * α) = 3 / 10 :=
by sorry

end problem1_problem2_l116_116362


namespace recurring_decimal_division_l116_116432

noncomputable def recurring_decimal_fraction : ℚ :=
  let frac_81 := (81 : ℚ) / 99
  let frac_36 := (36 : ℚ) / 99
  frac_81 / frac_36

theorem recurring_decimal_division :
  recurring_decimal_fraction = 9 / 4 :=
by
  sorry

end recurring_decimal_division_l116_116432


namespace find_value_divided_by_4_l116_116593

theorem find_value_divided_by_4 (x : ℝ) (h : 812 / x = 25) : x / 4 = 8.12 :=
by
  sorry

end find_value_divided_by_4_l116_116593


namespace find_b_l116_116566

-- Given conditions
def varies_inversely (a b : ℝ) := ∃ K : ℝ, K = a * b
def constant_a (a : ℝ) := a = 1500
def constant_b (b : ℝ) := b = 0.25

-- The theorem to prove
theorem find_b (a : ℝ) (b : ℝ) (h_inv: varies_inversely a b)
  (h_a: constant_a a) (h_b: constant_b b): b = 0.125 := 
sorry

end find_b_l116_116566


namespace solve_problem_l116_116532

noncomputable def problem_statement : Prop :=
  let a := Real.arcsin (4/5)
  let b := Real.arccos (1/2)
  Real.sin (a + b) = (4 + 3 * Real.sqrt 3) / 10

theorem solve_problem : problem_statement :=
by
  sorry

end solve_problem_l116_116532


namespace max_profit_l116_116187

noncomputable def C (x : ℝ) : ℝ :=
  if x < 80 then (1/2) * x^2 + 40 * x
  else 101 * x + 8100 / x - 2180

noncomputable def profit (x : ℝ) : ℝ :=
  if x < 80 then 100 * x - C x - 500
  else 100 * x - C x - 500

theorem max_profit :
  (∀ x, (0 < x ∧ x < 80) → profit x = - (1/2) * x^2 + 60 * x - 500) ∧
  (∀ x, (80 ≤ x) → profit x = 1680 - (x + 8100 / x)) ∧
  (∃ x, x = 90 ∧ profit x = 1500) :=
by {
  -- Proof here
  sorry
}

end max_profit_l116_116187


namespace constant_term_expansion_l116_116674

-- Define the binomial coefficient
noncomputable def binomial_coeff (n k : ℕ) : ℕ := Nat.choose n k

-- Define the general term in the binomial expansion
noncomputable def general_term (r n : ℕ) (x : ℝ) : ℝ := 
  (2:ℝ)^r * binomial_coeff n r * x^((n-5*r)/2)

-- Given problem conditions
def n := 10
def largest_binomial_term_index := 5  -- Represents the sixth term (r = 5)

-- Statement to prove the constant term equals 180
theorem constant_term_expansion {x : ℝ} : 
  general_term 2 n 1 = 180 :=
by {
  sorry
}

end constant_term_expansion_l116_116674


namespace problem_3_problem_4_l116_116251

open Classical

section
  variable {x₁ x₂ : ℝ}
  theorem problem_3 (hx₁ : 0 < x₁) (hx₂ : 0 < x₂) : (Real.log (x₁ * x₂) = Real.log x₁ + Real.log x₂) :=
  by
    sorry

  theorem problem_4 (hx₁ : 0 < x₁) (hx₂ : 0 < x₂) (hlt : x₁ < x₂) : ((Real.log x₁ - Real.log x₂) / (x₁ - x₂) > 0) :=
  by
    sorry
end

end problem_3_problem_4_l116_116251


namespace cricket_runs_l116_116003

theorem cricket_runs (A B C : ℕ) (h1 : A / B = 1 / 3) (h2 : B / C = 1 / 5) (h3 : A + B + C = 95) : C = 75 :=
by
  -- Skipping proof details
  sorry

end cricket_runs_l116_116003


namespace accurate_mass_l116_116613

variable (m1 m2 a b x : Real) -- Declare the variables

theorem accurate_mass (h1 : a * x = b * m1) (h2 : b * x = a * m2) : x = Real.sqrt (m1 * m2) := by
  -- We will prove the statement later
  sorry

end accurate_mass_l116_116613


namespace largest_tile_size_l116_116027

theorem largest_tile_size
  (length width : ℕ)
  (H1 : length = 378)
  (H2 : width = 595) :
  Nat.gcd length width = 7 :=
by
  sorry

end largest_tile_size_l116_116027


namespace inequality_proof_l116_116670

theorem inequality_proof (a b : ℝ) (h : a + b > 0) : 
  a / b^2 + b / a^2 ≥ 1 / a + 1 / b := 
sorry

end inequality_proof_l116_116670


namespace train_length_proof_l116_116927

def speed_kmph : ℝ := 54
def time_seconds : ℝ := 54.995600351971845
def bridge_length_m : ℝ := 660
def train_length_approx : ℝ := 164.93

noncomputable def speed_m_s : ℝ := speed_kmph * (1000 / 3600)
noncomputable def total_distance : ℝ := speed_m_s * time_seconds
noncomputable def train_length : ℝ := total_distance - bridge_length_m

theorem train_length_proof :
  abs (train_length - train_length_approx) < 0.01 :=
by
  sorry

end train_length_proof_l116_116927


namespace ms_hatcher_students_l116_116209

-- Define the number of third-graders
def third_graders : ℕ := 20

-- Condition: The number of fourth-graders is twice the number of third-graders
def fourth_graders : ℕ := 2 * third_graders

-- Condition: The number of fifth-graders is half the number of third-graders
def fifth_graders : ℕ := third_graders / 2

-- The total number of students Ms. Hatcher teaches in a day
def total_students : ℕ := third_graders + fourth_graders + fifth_graders

-- The statement to prove
theorem ms_hatcher_students : total_students = 70 := by
  sorry

end ms_hatcher_students_l116_116209


namespace y_intercept_probability_l116_116935

theorem y_intercept_probability (b : ℝ) (hb : b ∈ Set.Icc (-2 : ℝ) 3 ) :
  (∃ P : ℚ, P = (2 / 5)) := 
by 
  sorry

end y_intercept_probability_l116_116935


namespace tan_beta_eq_minus_one_seventh_l116_116222

theorem tan_beta_eq_minus_one_seventh {α β : ℝ} 
  (h1 : Real.tan α = 3) 
  (h2 : Real.tan (α + β) = 2) : 
  Real.tan β = -1/7 := 
by
  sorry

end tan_beta_eq_minus_one_seventh_l116_116222


namespace hall_area_l116_116212

theorem hall_area {L W : ℝ} (h₁ : W = 0.5 * L) (h₂ : L - W = 20) : L * W = 800 := by
  sorry

end hall_area_l116_116212


namespace parabola_above_line_l116_116406

variable (a b c : ℝ) (h : (b - c)^2 - 4 * a * c < 0)

theorem parabola_above_line : (b - c)^2 - 4 * a * c < 0 → (b - c)^2 - 4 * c * (a + b) < 0 :=
by sorry

end parabola_above_line_l116_116406


namespace meaning_of_poverty_l116_116612

theorem meaning_of_poverty (s : String) : s = "poverty" ↔ s = "poverty" := sorry

end meaning_of_poverty_l116_116612


namespace arithmetic_sequence_term_count_l116_116693

def first_term : ℕ := 5
def common_difference : ℕ := 3
def last_term : ℕ := 203

theorem arithmetic_sequence_term_count :
  ∃ n : ℕ, last_term = first_term + (n - 1) * common_difference ∧ n = 67 :=
by
  sorry

end arithmetic_sequence_term_count_l116_116693


namespace range_of_f_l116_116200

-- Define the function f(x) = 4 sin^3(x) + sin^2(x) - 4 sin(x) + 8
noncomputable def f (x : ℝ) : ℝ :=
  4 * (Real.sin x) ^ 3 + (Real.sin x) ^ 2 - 4 * (Real.sin x) + 8

-- Statement to prove the range of f(x)
theorem range_of_f :
  ∀ x : ℝ, 6 + 3 / 4 ≤ f x ∧ f x ≤ 9 + 25 / 27 :=
sorry

end range_of_f_l116_116200


namespace ac_bd_leq_8_l116_116881

theorem ac_bd_leq_8 (a b c d : ℝ) (h1 : a^2 + b^2 = 4) (h2 : c^2 + d^2 = 16) : ac + bd ≤ 8 :=
sorry

end ac_bd_leq_8_l116_116881


namespace remainder_3_pow_1000_mod_7_l116_116397

theorem remainder_3_pow_1000_mod_7 : 3 ^ 1000 % 7 = 4 := by
  sorry

end remainder_3_pow_1000_mod_7_l116_116397


namespace june_biking_time_l116_116836

theorem june_biking_time :
  ∀ (d_jj d_jb : ℕ) (t_jj : ℕ), (d_jj = 2) → (t_jj = 8) → (d_jb = 6) →
  (t_jb : ℕ) → t_jb = (d_jb * t_jj) / d_jj → t_jb = 24 :=
by
  intros d_jj d_jb t_jj h_djj h_tjj h_djb t_jb h_eq
  rw [h_djj, h_tjj, h_djb] at h_eq
  simp at h_eq
  exact h_eq

end june_biking_time_l116_116836


namespace range_of_m_l116_116468

noncomputable def f (x : ℝ) : ℝ := (1 / 2) * x^2 - ((Real.exp x - 1) / (Real.exp x + 1))

theorem range_of_m (m : ℝ) (h : f (4 - m) - f m ≥ 8 - 4 * m) : 2 ≤ m := by
  sorry

end range_of_m_l116_116468


namespace range_of_a_l116_116184

noncomputable def f (x : ℝ) : ℝ := 
  if x ≥ 0 then x^2 else -x^2

theorem range_of_a (a : ℝ) : (∀ x : ℝ, a ≤ x ∧ x ≤ a + 2 → f (x + a) ≥ 2 * f x) → a ≥ Real.sqrt 2 :=
by
  -- provided condition
  intros h
  sorry

end range_of_a_l116_116184


namespace area_between_hexagon_and_square_l116_116895

noncomputable def circleRadius : ℝ := 6

noncomputable def centralAngleSquare : ℝ := Real.pi / 2

noncomputable def centralAngleHexagon : ℝ := Real.pi / 3

noncomputable def areaSegment (r α : ℝ) : ℝ :=
  0.5 * r^2 * (α - Real.sin α)

noncomputable def areaBetweenArcs : ℝ :=
  let r := circleRadius
  let T_AB := areaSegment r centralAngleSquare
  let T_CD := areaSegment r centralAngleHexagon
  2 * (T_AB - T_CD)

theorem area_between_hexagon_and_square :
  abs (areaBetweenArcs - 14.03) < 0.01 :=
by
  sorry

end area_between_hexagon_and_square_l116_116895


namespace vector_BC_correct_l116_116261

-- Define the conditions
def vector_AB : ℝ × ℝ := (-3, 2)
def vector_AC : ℝ × ℝ := (1, -2)

-- Define the problem to be proved
theorem vector_BC_correct :
  let vector_BC := (vector_AC.1 - vector_AB.1, vector_AC.2 - vector_AB.2)
  vector_BC = (4, -4) :=
by
  sorry -- The proof is not required, but the structure indicates where it would go

end vector_BC_correct_l116_116261


namespace prime_factorization_675_l116_116583

theorem prime_factorization_675 :
  ∃ (n h : ℕ), n > 1 ∧ n = 3 ∧ h = 225 ∧ 675 = (3^3) * (5^2) :=
by
  sorry

end prime_factorization_675_l116_116583


namespace smallest_solution_x4_minus_40x2_plus_400_eq_zero_l116_116443

theorem smallest_solution_x4_minus_40x2_plus_400_eq_zero :
  ∃ x : ℝ, (x^4 - 40 * x^2 + 400 = 0) ∧ (∀ y : ℝ, (y^4 - 40 * y^2 + 400 = 0) → x ≤ y) :=
sorry

end smallest_solution_x4_minus_40x2_plus_400_eq_zero_l116_116443


namespace mao_li_total_cards_l116_116179

theorem mao_li_total_cards : (23 : ℕ) + (20 : ℕ) = 43 := by
  sorry

end mao_li_total_cards_l116_116179


namespace problem_1_solution_problem_2_solution_l116_116782

variables (total_balls : ℕ) (red_balls : ℕ) (black_balls : ℕ) (white_balls : ℕ) (green_ball : ℕ)

def probability_of_red_or_black_ball (total_balls red_balls black_balls white_balls green_ball : ℕ) : ℚ :=
  (red_balls + black_balls : ℚ) / total_balls

def probability_of_at_least_one_red_ball (total_balls red_balls black_balls white_balls green_ball : ℕ) : ℚ :=
  (((red_balls * (total_balls - red_balls)) + ((red_balls * (red_balls - 1)) / 2)) : ℚ)
  / ((total_balls * (total_balls - 1) / 2) : ℚ)

theorem problem_1_solution :
  probability_of_red_or_black_ball 12 5 4 2 1 = 3 / 4 :=
by
  sorry

theorem problem_2_solution :
  probability_of_at_least_one_red_ball 12 5 4 2 1 = 15 / 22 :=
by
  sorry

end problem_1_solution_problem_2_solution_l116_116782


namespace juanita_sunscreen_cost_l116_116542

theorem juanita_sunscreen_cost:
  let bottles_per_month := 1
  let months_in_year := 12
  let cost_per_bottle := 30.0
  let discount_rate := 0.30
  let total_bottles := bottles_per_month * months_in_year
  let total_cost_before_discount := total_bottles * cost_per_bottle
  let discount_amount := discount_rate * total_cost_before_discount
  let total_cost_after_discount := total_cost_before_discount - discount_amount
  total_cost_after_discount = 252.00 := 
by
  sorry

end juanita_sunscreen_cost_l116_116542


namespace subtract_eq_l116_116232

theorem subtract_eq (x y : ℝ) (h1 : 4 * x - 3 * y = 2) (h2 : 4 * x + y = 10) : 4 * y = 8 :=
by
  sorry

end subtract_eq_l116_116232


namespace inequality_f_x_f_a_l116_116463

noncomputable def f (x : ℝ) : ℝ := x * x + x + 13

theorem inequality_f_x_f_a (a x : ℝ) (h : |x - a| < 1) : |f x * f a| < 2 * (|a| + 1) := 
sorry

end inequality_f_x_f_a_l116_116463


namespace percentage_increase_l116_116404

theorem percentage_increase (P : ℝ) (x : ℝ) 
(h1 : 1.17 * P = 0.90 * P * (1 + x / 100)) : x = 33.33 :=
by sorry

end percentage_increase_l116_116404


namespace find_largest_number_l116_116891

noncomputable def largest_of_three_numbers (x y z : ℝ) : ℝ :=
  if x ≥ y ∧ x ≥ z then x
  else if y ≥ x ∧ y ≥ z then y
  else z

theorem find_largest_number (x y z : ℝ) (h1 : x + y + z = 3) (h2 : xy + xz + yz = -11) (h3 : xyz = 15) :
  largest_of_three_numbers x y z = Real.sqrt 5 := by
  sorry

end find_largest_number_l116_116891


namespace union_of_M_N_l116_116913

-- Definitions of sets M and N
def M : Set ℕ := {0, 1}
def N : Set ℕ := {1, 2}

-- The theorem to prove
theorem union_of_M_N : M ∪ N = {0, 1, 2} :=
  by sorry

end union_of_M_N_l116_116913


namespace num_factors_36_l116_116597

theorem num_factors_36 : ∀ (n : ℕ), n = 36 → (∃ (a b : ℕ), 36 = 2^a * 3^b ∧ a = 2 ∧ b = 2 ∧ (a + 1) * (b + 1) = 9) :=
by
  sorry

end num_factors_36_l116_116597


namespace kanul_total_amount_l116_116833

theorem kanul_total_amount (T : ℝ) (h1 : 500 + 400 + 0.10 * T = T) : T = 1000 :=
  sorry

end kanul_total_amount_l116_116833


namespace fraction_addition_l116_116896

variable {w x y : ℝ}

theorem fraction_addition (h1 : w / x = 1 / 3) (h2 : w / y = 3 / 4) : (x + y) / y = 13 / 4 := by
  sorry

end fraction_addition_l116_116896


namespace percentage_william_land_l116_116874

-- Definitions of the given conditions
def total_tax_collected : ℝ := 3840
def william_tax : ℝ := 480

-- Proof statement
theorem percentage_william_land :
  ((william_tax / total_tax_collected) * 100) = 12.5 :=
by
  sorry

end percentage_william_land_l116_116874


namespace probability_of_success_l116_116543

theorem probability_of_success 
  (pA : ℚ) (pB : ℚ) 
  (hA : pA = 2 / 3) 
  (hB : pB = 3 / 5) :
  1 - ((1 - pA) * (1 - pB)) = 13 / 15 :=
by
  sorry

end probability_of_success_l116_116543


namespace correct_equation_l116_116484

variables (x y : ℕ)

-- Conditions
def condition1 : Prop := y = 8 * x - 3
def condition2 : Prop := y = 7 * x + 4

-- Theorem to prove
theorem correct_equation (h1 : condition1 x y) (h2 : condition2 x y) : (y + 3) / 8 = (y - 4) / 7 :=
sorry

end correct_equation_l116_116484


namespace fraction_white_surface_area_l116_116655

theorem fraction_white_surface_area : 
  let total_surface_area := 96
  let black_faces_corners := 6
  let black_faces_centers := 6
  let black_faces_total := 12
  let white_faces_total := total_surface_area - black_faces_total
  white_faces_total / total_surface_area = 7 / 8 :=
by
  sorry

end fraction_white_surface_area_l116_116655


namespace infinite_triples_exists_l116_116575

/-- There are infinitely many ordered triples (a, b, c) of positive integers such that 
the greatest common divisor of a, b, and c is 1, and the sum a^2b^2 + b^2c^2 + c^2a^2 
is the square of an integer. -/
theorem infinite_triples_exists : ∃ (a b c : ℕ), (∀ p q : ℕ, p ≠ q ∧ p % 2 = 1 ∧ q % 2 = 1 ∧ 2 < p ∧ 2 < q →
  let a := p * q
  let b := 2 * p^2
  let c := q^2
  gcd (gcd a b) c = 1 ∧
  ∃ k : ℕ, a^2 * b^2 + b^2 * c^2 + c^2 * a^2 = k^2) :=
sorry

end infinite_triples_exists_l116_116575


namespace square_side_length_l116_116928

/-- If the area of a square is 9m^2 + 24mn + 16n^2, then the length of the side of the square is |3m + 4n|. -/
theorem square_side_length (m n : ℝ) (a : ℝ) (h : a^2 = 9 * m^2 + 24 * m * n + 16 * n^2) : a = |3 * m + 4 * n| :=
sorry

end square_side_length_l116_116928


namespace rebecca_income_l116_116286

variable (R : ℝ) -- Rebecca's current yearly income (denoted as R)
variable (increase : ℝ := 7000) -- The increase in Rebecca's income
variable (jimmy_income : ℝ := 18000) -- Jimmy's yearly income
variable (combined_income : ℝ := (R + increase) + jimmy_income) -- Combined income after increase
variable (new_income_ratio : ℝ := 0.55) -- Proportion of total income that is Rebecca's new income

theorem rebecca_income : (R + increase) = new_income_ratio * combined_income → R = 15000 :=
by
  sorry

end rebecca_income_l116_116286


namespace ice_cream_cost_l116_116440

variable {x F M : ℤ}

theorem ice_cream_cost (h1 : F = x - 7) (h2 : M = x - 1) (h3 : F + M < x) : x = 7 :=
by
  sorry

end ice_cream_cost_l116_116440


namespace total_amount_spent_l116_116642

-- Definitions for the conditions
def cost_magazine : ℝ := 0.85
def cost_pencil : ℝ := 0.50
def coupon_discount : ℝ := 0.35

-- The main theorem to prove
theorem total_amount_spent : cost_magazine + cost_pencil - coupon_discount = 1.00 := by
  sorry

end total_amount_spent_l116_116642


namespace delaney_travel_time_l116_116580

def bus_leaves_at := 8 * 60
def delaney_left_at := 7 * 60 + 50
def missed_by := 20

theorem delaney_travel_time
  (bus_leaves_at : ℕ) (delaney_left_at : ℕ) (missed_by : ℕ) :
  delaney_left_at + (bus_leaves_at + missed_by - bus_leaves_at) - delaney_left_at = 30 :=
by
  exact sorry

end delaney_travel_time_l116_116580


namespace smallest_value_of_a_minus_b_l116_116224

theorem smallest_value_of_a_minus_b (a b : ℤ) (h1 : a + b < 11) (h2 : a > 6) : a - b = 4 :=
by
  sorry

end smallest_value_of_a_minus_b_l116_116224


namespace eval_expression_l116_116399

theorem eval_expression : 15 * 30 + 45 * 15 - 15 * 10 = 975 :=
by 
  sorry

end eval_expression_l116_116399


namespace arithmetic_progression_sum_l116_116358

noncomputable def a (n : ℕ) (a1 d : ℤ) : ℤ := a1 + (n - 1) * d
noncomputable def S (n : ℕ) (a1 d : ℤ) : ℤ := n * (2 * a1 + (n - 1) * d) / 2

theorem arithmetic_progression_sum
    (a1 d : ℤ)
    (h : a 9 a1 d = a 12 a1 d / 2 + 3) :
  S 11 a1 d = 66 := 
by 
  sorry

end arithmetic_progression_sum_l116_116358


namespace estimate_blue_balls_l116_116281

theorem estimate_blue_balls (total_balls : ℕ) (prob_yellow : ℚ)
  (h_total : total_balls = 80)
  (h_prob_yellow : prob_yellow = 0.25) :
  total_balls * (1 - prob_yellow) = 60 :=
by
  -- proof
  sorry

end estimate_blue_balls_l116_116281


namespace min_n_satisfies_inequality_l116_116042

theorem min_n_satisfies_inequality :
  ∃ (n : ℕ), (∀ (x y z : ℝ), (x^2 + y^2 + z^2) ≤ n * (x^4 + y^4 + z^4)) ∧ (n = 3) :=
by
  sorry

end min_n_satisfies_inequality_l116_116042


namespace find_B_inter_complement_U_A_l116_116749

-- Define Universal set U
def U : Set ℤ := {-1, 0, 1, 2, 3, 4}

-- Define Set A
def A : Set ℤ := {2, 3}

-- Define complement of A relative to U
def complement_U_A : Set ℤ := U \ A

-- Define set B
def B : Set ℤ := {1, 4}

-- The goal to prove
theorem find_B_inter_complement_U_A : B ∩ complement_U_A = {1, 4} :=
by 
  have h1 : A = {2, 3} := rfl
  have h2 : U = {-1, 0, 1, 2, 3, 4} := rfl
  have h3 : B = {1, 4} := rfl
  sorry

end find_B_inter_complement_U_A_l116_116749


namespace min_value_is_neg_500000_l116_116851

noncomputable def min_expression_value (a b : ℝ) : ℝ :=
  let term1 := a + 1/b
  let term2 := b + 1/a
  (term1 * (term1 - 1000) + term2 * (term2 - 1000))

theorem min_value_is_neg_500000 (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  min_expression_value a b ≥ -500000 :=
sorry

end min_value_is_neg_500000_l116_116851


namespace parallelogram_area_l116_116860

theorem parallelogram_area
  (a b : ℕ)
  (h1 : a + b = 15)
  (h2 : 2 * a = 3 * b) :
  2 * a = 18 :=
by
  -- Proof is omitted; the statement shows what needs to be proven
  sorry

end parallelogram_area_l116_116860


namespace fraction_of_shaded_circle_l116_116419

theorem fraction_of_shaded_circle (total_regions shaded_regions : ℕ) (h1 : total_regions = 4) (h2 : shaded_regions = 1) :
  shaded_regions / total_regions = 1 / 4 := by
  sorry

end fraction_of_shaded_circle_l116_116419


namespace positive_difference_x_coordinates_lines_l116_116021

theorem positive_difference_x_coordinates_lines :
  let l := fun x : ℝ => -2 * x + 4
  let m := fun x : ℝ => - (1 / 5) * x + 1
  let x_l := (- (10 - 4) / 2)
  let x_m := (- (10 - 1) * 5)
  abs (x_l - x_m) = 42 := by
  sorry

end positive_difference_x_coordinates_lines_l116_116021


namespace trig_identity_l116_116871

variable {α : ℝ}

theorem trig_identity (h : Real.sin α = 2 * Real.cos α) :
  (Real.sin α - Real.cos α) / (Real.sin α + Real.cos α) = 1 / 3 := by
  sorry

end trig_identity_l116_116871


namespace simplify_expression_l116_116667

theorem simplify_expression : 
  2 + (1 / (2 + Real.sqrt 5)) - (1 / (2 - Real.sqrt 5)) = 2 - 2 * Real.sqrt 5 := 
by 
  sorry

end simplify_expression_l116_116667


namespace last_digit_of_7_to_the_7_l116_116967

theorem last_digit_of_7_to_the_7 :
  (7 ^ 7) % 10 = 3 :=
by
  sorry

end last_digit_of_7_to_the_7_l116_116967


namespace input_statement_is_INPUT_l116_116045

-- Define the type for statements
inductive Statement
| PRINT
| INPUT
| IF
| END

-- Define roles for the types of statements
def isOutput (s : Statement) : Prop := s = Statement.PRINT
def isInput (s : Statement) : Prop := s = Statement.INPUT
def isConditional (s : Statement) : Prop := s = Statement.IF
def isTermination (s : Statement) : Prop := s = Statement.END

-- Theorem to prove INPUT is the input statement
theorem input_statement_is_INPUT :
  isInput Statement.INPUT := by
  -- Proof to be provided
  sorry

end input_statement_is_INPUT_l116_116045


namespace find_third_side_of_triangle_l116_116988

theorem find_third_side_of_triangle (a b : ℝ) (A : ℝ) (h1 : a = 6) (h2 : b = 10) (h3 : A = 18) (h4 : ∃ C, 0 < C ∧ C < π / 2 ∧ A = 0.5 * a * b * Real.sin C) : 
  ∃ c : ℝ, c = 2 * Real.sqrt 22 :=
by
  sorry

end find_third_side_of_triangle_l116_116988


namespace not_possible_coloring_possible_coloring_l116_116090

-- Problem (a): For n = 2001 and k = 4001, prove that such coloring is not possible.
theorem not_possible_coloring (n : ℕ) (k : ℕ) (h_n : n = 2001) (h_k : k = 4001) :
  ¬ ∃ (color : ℕ × ℕ → ℕ), (∀ i j : ℕ, (1 ≤ i ∧ i ≤ n) ∧ (1 ≤ j ∧ j ≤ n) → 1 ≤ color (i, j) ∧ color (i, j) ≤ k)
  ∧ (∀ i, 1 ≤ i ∧ i ≤ n → ∀ j1 j2, (1 ≤ j1 ∧ j1 ≤ n) ∧ (1 ≤ j2 ∧ j2 ≤ n) → j1 ≠ j2 → color (i, j1) ≠ color (i, j2))
  ∧ (∀ j, 1 ≤ j ∧ j ≤ n → ∀ i1 i2, (1 ≤ i1 ∧ i1 ≤ n) ∧ (1 ≤ i2 ∧ i2 ≤ n) → i1 ≠ i2 → color (i1, j) ≠ color (i2, j)) := 
sorry

-- Problem (b): For n = 2^m - 1 and k = 2^(m+1) - 1, prove that such coloring is possible.
theorem possible_coloring (m : ℕ) (n k : ℕ) (h_n : n = 2^m - 1) (h_k : k = 2^(m+1) - 1) :
  ∃ (color : ℕ × ℕ → ℕ), (∀ i j : ℕ, (1 ≤ i ∧ i ≤ n) ∧ (1 ≤ j ∧ j ≤ n) → 1 ≤ color (i, j) ∧ color (i, j) ≤ k)
  ∧ (∀ i, 1 ≤ i ∧ i ≤ n → ∀ j1 j2, (1 ≤ j1 ∧ j1 ≤ n) ∧ (1 ≤ j2 ∧ j2 ≤ n) → j1 ≠ j2 → color (i, j1) ≠ color (i, j2))
  ∧ (∀ j, 1 ≤ j ∧ j ≤ n → ∀ i1 i2, (1 ≤ i1 ∧ i1 ≤ n) ∧ (1 ≤ i2 ∧ i2 ≤ n) → i1 ≠ i2 → color (i1, j) ≠ color (i2, j)) := 
sorry

end not_possible_coloring_possible_coloring_l116_116090


namespace initial_number_of_men_l116_116075

theorem initial_number_of_men (M A : ℕ) : 
  (∀ (M A : ℕ), ((M * A) - 40 + 61) / M = (A + 3)) ∧ (30.5 = 30.5) → 
  M = 7 :=
by
  sorry

end initial_number_of_men_l116_116075


namespace gcd_bn_bn1_l116_116293

def b (n : ℕ) : ℤ := (7^n - 1) / 6
def e (n : ℕ) : ℤ := Int.gcd (b n) (b (n + 1))

theorem gcd_bn_bn1 (n : ℕ) : e n = 1 := by
  sorry

end gcd_bn_bn1_l116_116293


namespace spend_money_l116_116890

theorem spend_money (n : ℕ) (h : n > 7) : ∃ a b : ℕ, 3 * a + 5 * b = n :=
by
  sorry

end spend_money_l116_116890


namespace vanya_correct_answers_l116_116410

theorem vanya_correct_answers (x : ℕ) : 
  (7 * x = 3 * (50 - x)) → x = 15 := by
sorry

end vanya_correct_answers_l116_116410


namespace ratio_a_c_l116_116702

theorem ratio_a_c (a b c : ℕ) (h1 : a / b = 8 / 3) (h2 : b / c = 1 / 5) : a / c = 8 / 15 := 
by
  sorry

end ratio_a_c_l116_116702


namespace group_total_people_l116_116548

theorem group_total_people (k : ℕ) (h1 : k = 7) (h2 : ((n - k) / n : ℝ) - (k / n : ℝ) = 0.30000000000000004) : n = 20 :=
  sorry

end group_total_people_l116_116548


namespace certain_number_is_correct_l116_116064

def m : ℕ := 72483

theorem certain_number_is_correct : 9999 * m = 724827405 := by
  sorry

end certain_number_is_correct_l116_116064


namespace smallest_five_digit_divisible_by_53_and_3_l116_116437

/-- The smallest five-digit positive integer divisible by 53 and 3 is 10062 -/
theorem smallest_five_digit_divisible_by_53_and_3 : ∃ n : ℕ, 10000 ≤ n ∧ n < 100000 ∧ n % 53 = 0 ∧ n % 3 = 0 ∧ ∀ m : ℕ, 10000 ≤ m ∧ m < 100000 ∧ m % 53 = 0 ∧ m % 3 = 0 → n ≤ m ∧ n = 10062 :=
by
  sorry

end smallest_five_digit_divisible_by_53_and_3_l116_116437


namespace savannah_wrapped_gifts_with_second_roll_l116_116859

theorem savannah_wrapped_gifts_with_second_roll (total_gifts rolls_used roll_1_gifts roll_3_gifts roll_2_gifts : ℕ) 
  (h1 : total_gifts = 12) 
  (h2 : rolls_used = 3) 
  (h3 : roll_1_gifts = 3) 
  (h4 : roll_3_gifts = 4)
  (h5 : total_gifts - roll_1_gifts - roll_3_gifts = roll_2_gifts) :
  roll_2_gifts = 5 := 
by
  sorry

end savannah_wrapped_gifts_with_second_roll_l116_116859


namespace minimum_value_of_f_l116_116223

def f (x : ℝ) : ℝ := |x - 4| + |x + 6| + |x - 5|

theorem minimum_value_of_f :
  ∃ x : ℝ, (x = -6 ∧ f (-6) = 1) ∧ ∀ y : ℝ, f y ≥ 1 :=
by
  sorry

end minimum_value_of_f_l116_116223


namespace simplify_expression1_simplify_expression2_l116_116148

variables {V : Type*} [AddCommGroup V] [Module ℝ V]
variables (A B C D F : V)

-- Problem 1:
theorem simplify_expression1 : 
  (D - C) + (C - B) + (B - A) = D - A := 
sorry

-- Problem 2:
theorem simplify_expression2 : 
  (B - A) + (F - D) + (D - C) + (C - B) + (A - F) = 0 := 
sorry

end simplify_expression1_simplify_expression2_l116_116148


namespace marseille_hairs_l116_116973

theorem marseille_hairs (N : ℕ) (M : ℕ) (hN : N = 2000000) (hM : M = 300001) :
  ∃ k, k ≥ 7 ∧ ∃ b : ℕ, b ≤ M ∧ b > 0 ∧ ∀ i ≤ M, ∃ l : ℕ, l ≥ k → l ≤ (N / M + 1) :=
by
  sorry

end marseille_hairs_l116_116973


namespace evaluate_expression_l116_116473

theorem evaluate_expression : 2 - 1 / (2 + 1 / (2 - 1 / 3)) = 21 / 13 := by
  sorry

end evaluate_expression_l116_116473


namespace sarahs_packages_l116_116453

def num_cupcakes_before : ℕ := 60
def num_cupcakes_ate : ℕ := 22
def cupcakes_per_package : ℕ := 10

theorem sarahs_packages : (num_cupcakes_before - num_cupcakes_ate) / cupcakes_per_package = 3 :=
by
  sorry

end sarahs_packages_l116_116453


namespace cosine_of_negative_135_l116_116875

theorem cosine_of_negative_135 : Real.cos (-(135 * Real.pi / 180)) = -Real.sqrt 2 / 2 :=
by
  sorry

end cosine_of_negative_135_l116_116875


namespace number_of_teams_l116_116330

theorem number_of_teams (n : ℕ) (h : n * (n - 1) / 2 = 28) : n = 8 :=
sorry

end number_of_teams_l116_116330


namespace carol_tom_combined_weight_mildred_heavier_than_carol_tom_combined_l116_116687

def mildred_weight : ℕ := 59
def carol_weight : ℕ := 9
def tom_weight : ℕ := 20

theorem carol_tom_combined_weight :
  carol_weight + tom_weight = 29 := by
  sorry

theorem mildred_heavier_than_carol_tom_combined :
  mildred_weight - (carol_weight + tom_weight) = 30 := by
  sorry

end carol_tom_combined_weight_mildred_heavier_than_carol_tom_combined_l116_116687


namespace plan_b_rate_l116_116585

noncomputable def cost_plan_a (duration : ℕ) : ℝ :=
  if duration ≤ 4 then 0.60
  else 0.60 + 0.06 * (duration - 4)

def cost_plan_b (duration : ℕ) (rate : ℝ) : ℝ :=
  rate * duration

theorem plan_b_rate (rate : ℝ) : 
  cost_plan_a 18 = cost_plan_b 18 rate → rate = 0.08 := 
by
  -- proof goes here
  sorry

end plan_b_rate_l116_116585


namespace expand_binomials_l116_116934

theorem expand_binomials (x : ℝ) : (3 * x + 4) * (2 * x + 7) = 6 * x^2 + 29 * x + 28 := 
by 
  sorry

end expand_binomials_l116_116934


namespace largest_of_five_consecutive_integers_l116_116639

   theorem largest_of_five_consecutive_integers (n1 n2 n3 n4 n5 : ℕ) 
     (h1: 0 < n1) (h2: n1 + 1 = n2) (h3: n2 + 1 = n3) (h4: n3 + 1 = n4)
     (h5: n4 + 1 = n5) (h6: n1 * n2 * n3 * n4 * n5 = 15120) : n5 = 10 :=
   sorry
   
end largest_of_five_consecutive_integers_l116_116639


namespace percentage_students_with_same_grade_l116_116435

def total_students : ℕ := 50
def students_with_same_grade : ℕ := 3 + 6 + 8 + 2 + 1

theorem percentage_students_with_same_grade :
  (students_with_same_grade / total_students : ℚ) * 100 = 40 :=
by
  sorry

end percentage_students_with_same_grade_l116_116435


namespace find_polynomial_l116_116058

noncomputable def polynomial_p (n : ℕ) (P : ℝ → ℝ → ℝ) :=
  ∀ t x y a b c : ℝ,
    (P (t * x) (t * y) = t ^ n * P x y) ∧
    (P (a + b) c + P (b + c) a + P (c + a) b = 0) ∧
    (P 1 0 = 1)

theorem find_polynomial (n : ℕ) (P : ℝ → ℝ → ℝ) (h : polynomial_p n P) :
  ∀ x y : ℝ, P x y = x^n - y^n :=
sorry

end find_polynomial_l116_116058


namespace find_possible_first_term_l116_116678

noncomputable def geometric_sequence_first_term (a r : ℝ) : Prop :=
  (a * r^2 = 3) ∧ (a * r^4 = 27)

theorem find_possible_first_term (a r : ℝ) (h : geometric_sequence_first_term a r) :
    a = 1 / 3 :=
by
  sorry

end find_possible_first_term_l116_116678


namespace contractor_absent_days_l116_116046

theorem contractor_absent_days (x y : ℝ) 
  (h1 : x + y = 30) 
  (h2 : 25 * x - 7.5 * y = 685) : 
  y = 2 :=
by
  sorry

end contractor_absent_days_l116_116046


namespace two_trucks_carry_2_tons_l116_116103

theorem two_trucks_carry_2_tons :
  ∀ (truck_capacity : ℕ), truck_capacity = 999 →
  (truck_capacity * 2) / 1000 = 2 :=
by
  intros truck_capacity h_capacity
  rw [h_capacity]
  exact sorry

end two_trucks_carry_2_tons_l116_116103


namespace fractions_order_l116_116018

theorem fractions_order :
  (25 / 21 < 23 / 19) ∧ (23 / 19 < 21 / 17) :=
by {
  sorry
}

end fractions_order_l116_116018


namespace circle_points_l116_116129

noncomputable def proof_problem (x1 y1 x2 y2: ℝ) : Prop :=
  (x1^2 + y1^2 = 4) ∧ (x2^2 + y2^2 = 4) ∧ ((x1 - x2)^2 + (y1 - y2)^2 = 12) →
    (x1 * x2 + y1 * y2 = -2)

theorem circle_points (x1 y1 x2 y2 : ℝ) : proof_problem x1 y1 x2 y2 := 
by
  sorry

end circle_points_l116_116129


namespace yogurt_combinations_l116_116704

theorem yogurt_combinations (flavors toppings : ℕ) (hflavors : flavors = 5) (htoppings : toppings = 8) :
  (flavors * Nat.choose toppings 3 = 280) :=
by
  rw [hflavors, htoppings]
  sorry

end yogurt_combinations_l116_116704


namespace trigonometric_identity_l116_116621

theorem trigonometric_identity (α : ℝ) (h : Real.tan α = -2) :
  Real.sin (π / 2 + α) * Real.cos (π + α) = -1 / 5 :=
by
  -- The proof will be skipped but the statement should be correct.
  sorry

end trigonometric_identity_l116_116621


namespace range_of_a_l116_116876

theorem range_of_a (a : ℝ) : 
  (∃ x : ℝ, (a+2)/(x+1) = 1 ∧ x ≤ 0) ↔ (a ≤ -1 ∧ a ≠ -2) :=
by
  sorry

end range_of_a_l116_116876


namespace cakes_served_for_lunch_l116_116282

theorem cakes_served_for_lunch (total_cakes: ℕ) (dinner_cakes: ℕ) (lunch_cakes: ℕ) 
  (h1: total_cakes = 15) 
  (h2: dinner_cakes = 9) 
  (h3: total_cakes = lunch_cakes + dinner_cakes) : 
  lunch_cakes = 6 := 
by 
  sorry

end cakes_served_for_lunch_l116_116282


namespace population_ratios_l116_116633

variable (P_X P_Y P_Z : Nat)

theorem population_ratios
  (h1 : P_Y = 2 * P_Z)
  (h2 : P_X = 10 * P_Z) : P_X / P_Y = 5 := by
  sorry

end population_ratios_l116_116633


namespace chocolate_bar_cost_l116_116937

-- Define the quantities Jessica bought
def chocolate_bars := 10
def gummy_bears_packs := 10
def chocolate_chips_bags := 20

-- Define the costs
def total_cost := 150
def gummy_bears_pack_cost := 2
def chocolate_chips_bag_cost := 5

-- Define what we want to prove (the cost of one chocolate bar)
theorem chocolate_bar_cost : 
  ∃ chocolate_bar_cost, 
    chocolate_bars * chocolate_bar_cost + 
    gummy_bears_packs * gummy_bears_pack_cost + 
    chocolate_chips_bags * chocolate_chips_bag_cost = total_cost ∧
    chocolate_bar_cost = 3 :=
by
  -- Proof goes here
  sorry

end chocolate_bar_cost_l116_116937


namespace min_sum_of_dimensions_l116_116104

theorem min_sum_of_dimensions (a b c : ℕ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_vol : a * b * c = 3003) : 
  a + b + c ≥ 57 := sorry

end min_sum_of_dimensions_l116_116104


namespace sequence_relation_l116_116676

theorem sequence_relation
  (a : ℕ → ℚ) (b : ℕ → ℚ)
  (h1 : ∀ n, b (n + 1) * a n + b n * a (n + 1) = (-2)^n + 1)
  (h2 : ∀ n, b n = (3 + (-1 : ℚ)^(n-1)) / 2)
  (h3 : a 1 = 2) :
  ∀ n, a (2 * n) = (1 - 4^n) / 2 :=
by
  intro n
  sorry

end sequence_relation_l116_116676


namespace number_of_ways_to_divide_l116_116508

-- Define the given shape
structure Shape :=
  (sides : Nat) -- Number of 3x1 stripes along the sides
  (centre : Nat) -- Size of the central square (3x3)

-- Define the specific problem shape
def problem_shape : Shape :=
  { sides := 4, centre := 9 } -- 3x1 stripes on all sides and a 3x3 centre

-- Theorem stating the number of ways to divide the shape into 1x3 rectangles
theorem number_of_ways_to_divide (s : Shape) (h1 : s.sides = 4) (h2 : s.centre = 9) : 
  ∃ ways, ways = 2 :=
by
  -- The proof is skipped
  sorry

end number_of_ways_to_divide_l116_116508


namespace bald_eagle_pairs_l116_116454

theorem bald_eagle_pairs (n_1963 : ℕ) (increase : ℕ) (h1 : n_1963 = 417) (h2 : increase = 6649) :
  (n_1963 + increase = 7066) :=
by
  sorry

end bald_eagle_pairs_l116_116454


namespace factorial_sum_perfect_square_iff_l116_116918

def is_perfect_square (n : Nat) : Prop := ∃ m : Nat, m * m = n

def sum_of_factorials (n : Nat) : Nat :=
  (List.range (n + 1)).map Nat.factorial |>.sum

theorem factorial_sum_perfect_square_iff (n : Nat) :
  n = 1 ∨ n = 3 ↔ is_perfect_square (sum_of_factorials n) := by {
  sorry
}

end factorial_sum_perfect_square_iff_l116_116918


namespace student_score_l116_116838

theorem student_score (c w : ℕ) (h1 : c + w = 60) (h2 : 4 * c - w = 150) : c = 42 :=
by
-- Proof steps here, we skip by using sorry for now
sorry

end student_score_l116_116838


namespace max_n_for_factored_poly_l116_116788

theorem max_n_for_factored_poly : 
  ∃ (n : ℤ), (∀ (A B : ℤ), 2 * B + A = n → A * B = 50) ∧ 
            (∀ (m : ℤ), (∀ (A B : ℤ), 2 * B + A = m → A * B = 50) → m ≤ 101) ∧ 
            n = 101 :=
by
  sorry

end max_n_for_factored_poly_l116_116788


namespace tens_digit_of_13_pow_2023_l116_116073

theorem tens_digit_of_13_pow_2023 :
  ∀ (n : ℕ), (13 ^ (2023 % 20) ≡ 13 ^ n [MOD 100]) ∧ (13 ^ n ≡ 97 [MOD 100]) → (13 ^ 2023) % 100 / 10 % 10 = 9 :=
by
sorry

end tens_digit_of_13_pow_2023_l116_116073


namespace starting_number_l116_116966

theorem starting_number (n : ℤ) : 
  (∃ n, (200 - n) / 3 = 33 ∧ (200 % 3 ≠ 0) ∧ (n % 3 = 0 ∧ n ≤ 200)) → n = 102 :=
by
  sorry

end starting_number_l116_116966


namespace jasmine_average_pace_l116_116098

-- Define the conditions given in the problem
def totalDistance : ℝ := 45
def totalTime : ℝ := 9

-- Define the assertion that needs to be proved
theorem jasmine_average_pace : totalDistance / totalTime = 5 :=
by sorry

end jasmine_average_pace_l116_116098


namespace product_of_xy_l116_116196

-- Define the problem conditions
variables (x y : ℝ)
-- Define the condition that |x-3| and |y+1| are opposite numbers
def opposite_abs_values := |x - 3| = - |y + 1|

-- State the theorem
theorem product_of_xy (h : opposite_abs_values x y) : x * y = -3 :=
sorry -- Proof is omitted

end product_of_xy_l116_116196


namespace sequence_formula_l116_116370

-- Definitions of the sequence and conditions
def S (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  Finset.sum (Finset.range n) a

def satisfies_condition (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, S a n + a n = 2 * n + 1

-- Proposition to prove
theorem sequence_formula (a : ℕ → ℝ) (h : satisfies_condition a) : 
  ∀ n : ℕ, a n = 2 - 1 / 2^n := sorry

end sequence_formula_l116_116370


namespace prob_defective_first_draw_prob_defective_both_draws_prob_defective_second_given_first_l116_116193

-- Definitions
def total_products := 20
def defective_products := 5

-- Probability of drawing a defective product on the first draw
theorem prob_defective_first_draw : (defective_products / total_products : ℚ) = 1 / 4 :=
sorry

-- Probability of drawing defective products on both the first and the second draws
theorem prob_defective_both_draws : (defective_products / total_products * (defective_products - 1) / (total_products - 1) : ℚ) = 1 / 19 :=
sorry

-- Probability of drawing a defective product on the second draw given that the first was defective
theorem prob_defective_second_given_first : ((defective_products - 1) / (total_products - 1) / (defective_products / total_products) : ℚ) = 4 / 19 :=
sorry

end prob_defective_first_draw_prob_defective_both_draws_prob_defective_second_given_first_l116_116193


namespace triangle_shape_l116_116386

theorem triangle_shape (a b c : ℝ) (h : a^4 - b^4 + (b^2 * c^2 - a^2 * c^2) = 0) :
  (a = b) ∨ (a^2 + b^2 = c^2) :=
sorry

end triangle_shape_l116_116386


namespace Patricia_money_l116_116808

theorem Patricia_money 
(P L C : ℝ)
(h1 : L = 5 * P)
(h2 : L = 2 * C)
(h3 : P + L + C = 51) :
P = 6.8 := 
by 
  sorry

end Patricia_money_l116_116808


namespace certain_amount_l116_116411

theorem certain_amount (n : ℤ) (x : ℤ) : n = 5 ∧ 7 * n - 15 = 2 * n + x → x = 10 :=
by
  sorry

end certain_amount_l116_116411


namespace largest_digit_divisible_by_6_l116_116925

theorem largest_digit_divisible_by_6 : ∃ N : ℕ, N ≤ 9 ∧ (∃ d : ℕ, 3456 * 10 + N = 6 * d) ∧ (∀ M : ℕ, M ≤ 9 ∧ (∃ d : ℕ, 3456 * 10 + M = 6 * d) → M ≤ N) :=
sorry

end largest_digit_divisible_by_6_l116_116925


namespace total_revenue_l116_116823

theorem total_revenue (chips_sold : ℕ) (chips_price : ℝ) (hotdogs_sold : ℕ) (hotdogs_price : ℝ)
(drinks_sold : ℕ) (drinks_price : ℝ) (sodas_sold : ℕ) (lemonades_sold : ℕ) (sodas_ratio : ℕ)
(lemonades_ratio : ℕ) (h1 : chips_sold = 27) (h2 : chips_price = 1.50) (h3 : hotdogs_sold = chips_sold - 8)
(h4 : hotdogs_price = 3.00) (h5 : drinks_sold = hotdogs_sold + 12) (h6 : drinks_price = 2.00)
(h7 : sodas_ratio = 2) (h8 : lemonades_ratio = 3) (h9 : sodas_sold = (sodas_ratio * drinks_sold) / (sodas_ratio + lemonades_ratio))
(h10 : lemonades_sold = drinks_sold - sodas_sold) :
chips_sold * chips_price + hotdogs_sold * hotdogs_price + drinks_sold * drinks_price = 159.50 := 
by
  -- Proof is left as an exercise for the reader
  sorry

end total_revenue_l116_116823


namespace func_translation_right_symm_yaxis_l116_116745

def f (x : ℝ) : ℝ := sorry

theorem func_translation_right_symm_yaxis (f : ℝ → ℝ) 
  (h1 : ∀ x, f (x - 1) = e ^ (-x)) :
  ∀ x, f x = e ^ (-x - 1) := sorry

end func_translation_right_symm_yaxis_l116_116745


namespace proof_problem_l116_116880

noncomputable def real_numbers (a x y : ℝ) (h₁ : 0 < a ∧ a < 1) (h₂ : a^x < a^y) : Prop :=
  x^3 > y^3

-- The theorem statement
theorem proof_problem (a x y : ℝ) (h₁ : 0 < a) (h₂ : a < 1) (h₃ : a^x < a^y) : x^3 > y^3 :=
by
  sorry

end proof_problem_l116_116880


namespace inequality_abc_l116_116632

theorem inequality_abc (a b c : ℝ) (h : a * b * c = 1) :
  1 / (2 * a^2 + b^2 + 3) + 1 / (2 * b^2 + c^2 + 3) + 1 / (2 * c^2 + a^2 + 3) ≤ 1 / 2 :=
by
  -- Sorry to skip the proof
  sorry

end inequality_abc_l116_116632


namespace distance_between_foci_l116_116666

-- Define the given ellipse equation
def ellipse_eq (x y : ℝ) : Prop := 25 * x^2 - 100 * x + 4 * y^2 + 8 * y + 36 = 0

-- Define the distance between the foci of the ellipse
theorem distance_between_foci (x y : ℝ) (h : ellipse_eq x y) : 2 * Real.sqrt 14.28 = 2 * Real.sqrt 14.28 :=
by sorry

end distance_between_foci_l116_116666


namespace man_l116_116036

theorem man's_salary 
  (food_fraction : ℚ := 1/5) 
  (rent_fraction : ℚ := 1/10) 
  (clothes_fraction : ℚ := 3/5) 
  (remaining_money : ℚ := 15000) 
  (S : ℚ) :
  (S * (1 - (food_fraction + rent_fraction + clothes_fraction)) = remaining_money) →
  S = 150000 := 
by
  intros h1
  sorry

end man_l116_116036


namespace modulo_4_equiv_2_l116_116746

open Nat

noncomputable def f (n : ℕ) [Fintype (ZMod n)] : ZMod n → ZMod n := sorry

theorem modulo_4_equiv_2 (n : ℕ) [hn : Fact (n > 0)] 
  (f : ZMod n → ZMod n)
  (h1 : ∀ x, f x ≠ x)
  (h2 : ∀ x, f (f x) = x)
  (h3 : ∀ x, f (f (f (x + 1) + 1) + 1) = x) : 
  n % 4 = 2 := 
sorry

end modulo_4_equiv_2_l116_116746


namespace time_to_eliminate_mice_l116_116957

def total_work : ℝ := 1
def work_done_by_2_cats_in_5_days : ℝ := 0.5
def initial_2_cats : ℕ := 2
def additional_cats : ℕ := 3
def total_initial_days : ℝ := 5
def total_cats : ℕ := initial_2_cats + additional_cats

theorem time_to_eliminate_mice (h : total_initial_days * (work_done_by_2_cats_in_5_days / total_initial_days) = work_done_by_2_cats_in_5_days) : 
  total_initial_days + (total_work - work_done_by_2_cats_in_5_days) / (total_cats * (work_done_by_2_cats_in_5_days / total_initial_days / initial_2_cats)) = 7 := 
by
  sorry

end time_to_eliminate_mice_l116_116957


namespace remainder_t4_mod7_l116_116327

def T : ℕ → ℕ
| 0 => 0 -- Not used
| 1 => 6
| n+1 => 6 ^ (T n)

theorem remainder_t4_mod7 : (T 4 % 7) = 6 := by
  sorry

end remainder_t4_mod7_l116_116327


namespace max_n_for_Sn_neg_l116_116413

noncomputable def Sn (n : ℕ) (a : ℕ → ℝ) : ℝ :=
  (n * (a 1 + a n)) / 2

theorem max_n_for_Sn_neg (a : ℕ → ℝ) (h1 : ∀ n : ℕ, (n + 1) * Sn n a < n * Sn (n + 1) a)
  (h2 : a 8 / a 7 < -1) :
  ∀ n : ℕ, S_13 < 0 ∧ S_14 > 0 →
  ∀ m : ℕ, m > 13 → Sn m a ≥ 0 :=
sorry

end max_n_for_Sn_neg_l116_116413


namespace compare_log_exp_powers_l116_116478

variable (a b c : ℝ)

theorem compare_log_exp_powers (h1 : a = Real.log 0.3 / Real.log 2)
                               (h2 : b = Real.exp (Real.log 2 * 0.1))
                               (h3 : c = Real.exp (Real.log 0.2 * 1.3)) :
  a < c ∧ c < b :=
by
  sorry

end compare_log_exp_powers_l116_116478


namespace first_discount_percentage_l116_116155

theorem first_discount_percentage :
  ∃ x : ℝ, (9649.12 * (1 - x / 100) * 0.9 * 0.95 = 6600) ∧ (19.64 ≤ x ∧ x ≤ 19.66) :=
sorry

end first_discount_percentage_l116_116155


namespace solve_for_a_l116_116587

theorem solve_for_a : ∀ (a : ℝ), (2 * a - 16 = 9) → (a = 12.5) :=
by
  intro a h
  sorry

end solve_for_a_l116_116587


namespace major_axis_length_l116_116084

theorem major_axis_length (x y : ℝ) (h : 16 * x^2 + 9 * y^2 = 144) : 8 = 8 :=
by
  sorry

end major_axis_length_l116_116084


namespace travel_time_to_Virgo_island_l116_116167

theorem travel_time_to_Virgo_island (boat_time : ℝ) (plane_time : ℝ) (total_time : ℝ) 
  (h1 : boat_time ≤ 2) (h2 : plane_time = 4 * boat_time) (h3 : total_time = plane_time + boat_time) : 
  total_time = 10 :=
by
  sorry

end travel_time_to_Virgo_island_l116_116167


namespace train_a_speed_54_l116_116053

noncomputable def speed_of_train_A (length_A length_B : ℕ) (speed_B : ℕ) (time_to_cross : ℕ) : ℕ :=
  let total_distance := length_A + length_B
  let relative_speed := total_distance / time_to_cross
  let relative_speed_km_per_hr := relative_speed * 36 / 10
  let speed_A := relative_speed_km_per_hr - speed_B
  speed_A

theorem train_a_speed_54 
  (length_A length_B : ℕ)
  (speed_B : ℕ)
  (time_to_cross : ℕ)
  (h_length_A : length_A = 150)
  (h_length_B : length_B = 150)
  (h_speed_B : speed_B = 36)
  (h_time_to_cross : time_to_cross = 12) :
  speed_of_train_A length_A length_B speed_B time_to_cross = 54 := by
  sorry

end train_a_speed_54_l116_116053


namespace john_has_48_l116_116217

variable (Ali Nada John : ℕ)

theorem john_has_48 
  (h1 : Ali + Nada + John = 67)
  (h2 : Ali = Nada - 5)
  (h3 : John = 4 * Nada) : 
  John = 48 := 
by 
  sorry

end john_has_48_l116_116217


namespace inhabitants_reach_ball_on_time_l116_116556

theorem inhabitants_reach_ball_on_time
  (kingdom_side_length : ℝ)
  (messenger_sent_at : ℕ)
  (ball_begins_at : ℕ)
  (inhabitant_speed : ℝ)
  (time_available : ℝ)
  (max_distance_within_square : ℝ)
  (H_side_length : kingdom_side_length = 2)
  (H_messenger_time : messenger_sent_at = 12)
  (H_ball_time : ball_begins_at = 19)
  (H_speed : inhabitant_speed = 3)
  (H_time_avail : time_available = 7)
  (H_max_distance : max_distance_within_square = 2 * Real.sqrt 2) :
  ∃ t : ℝ, t ≤ time_available ∧ max_distance_within_square / inhabitant_speed ≤ t :=
by
  -- You would write the proof here.
  sorry

end inhabitants_reach_ball_on_time_l116_116556


namespace compare_numbers_l116_116724

theorem compare_numbers : 222^2 < 22^22 ∧ 22^22 < 2^222 :=
by {
  sorry
}

end compare_numbers_l116_116724


namespace total_fruit_weight_l116_116034

-- Definitions for the conditions
def mario_ounces : ℕ := 8
def lydia_ounces : ℕ := 24
def nicolai_pounds : ℕ := 6
def ounces_per_pound : ℕ := 16

-- Theorem statement
theorem total_fruit_weight : 
  ((mario_ounces / ounces_per_pound : ℚ) + 
   (lydia_ounces / ounces_per_pound : ℚ) + 
   (nicolai_pounds : ℚ)) = 8 := 
sorry

end total_fruit_weight_l116_116034


namespace number_of_pentagonal_faces_is_12_more_than_heptagonal_faces_l116_116586

theorem number_of_pentagonal_faces_is_12_more_than_heptagonal_faces
  (convex : Prop)
  (trihedral : Prop)
  (faces_have_5_6_or_7_sides : Prop)
  (V E F : ℕ)
  (a b c : ℕ)
  (euler : V - E + F = 2)
  (edges_def : E = (5 * a + 6 * b + 7 * c) / 2)
  (vertices_def : V = (5 * a + 6 * b + 7 * c) / 3) :
  a = c + 12 :=
  sorry

end number_of_pentagonal_faces_is_12_more_than_heptagonal_faces_l116_116586


namespace selling_price_ratio_l116_116237

theorem selling_price_ratio (C : ℝ) (hC : C > 0) :
  let S₁ := 1.60 * C
  let S₂ := 4.20 * C
  S₂ / S₁ = 21 / 8 :=
by
  let S₁ := 1.60 * C
  let S₂ := 4.20 * C
  sorry

end selling_price_ratio_l116_116237


namespace plants_per_row_l116_116987

theorem plants_per_row (P : ℕ) (rows : ℕ) (yield_per_plant : ℕ) (total_yield : ℕ) 
  (h1 : rows = 30)
  (h2 : yield_per_plant = 20)
  (h3 : total_yield = 6000)
  (h4 : rows * yield_per_plant * P = total_yield) : 
  P = 10 :=
by 
  sorry

end plants_per_row_l116_116987


namespace maximize_NPM_l116_116241

theorem maximize_NPM :
  ∃ (M N P : ℕ), 
    (∀ M, M < 10 → (11 * M * M) = N * 100 + P * 10 + M) →
    N * 100 + P * 10 + M = 396 :=
by
  sorry

end maximize_NPM_l116_116241


namespace hexagon_same_length_probability_l116_116728

noncomputable def hexagon_probability_same_length : ℚ :=
  let sides := 6
  let diagonals := 9
  let total_segments := sides + diagonals
  let probability_side_first := (sides : ℚ) / total_segments
  let probability_diagonal_first := (diagonals : ℚ) / total_segments
  let probability_second_side := (sides - 1 : ℚ) / (total_segments - 1)
  let probability_second_diagonal_same_length := 2 / (total_segments - 1)
  probability_side_first * probability_second_side + 
  probability_diagonal_first * probability_second_diagonal_same_length

theorem hexagon_same_length_probability : hexagon_probability_same_length = 11 / 35 := 
  sorry

end hexagon_same_length_probability_l116_116728


namespace prob_two_red_balls_l116_116367

-- Define the initial conditions for the balls in the bag
def red_balls : ℕ := 5
def blue_balls : ℕ := 6
def green_balls : ℕ := 2
def total_balls : ℕ := red_balls + blue_balls + green_balls

-- Define the probability of picking a red ball first
def prob_red1 : ℚ := red_balls / total_balls

-- Define the remaining number of balls and the probability of picking a red ball second
def remaining_red_balls : ℕ := red_balls - 1
def remaining_total_balls : ℕ := total_balls - 1
def prob_red2 : ℚ := remaining_red_balls / remaining_total_balls

-- Define the combined probability of both events
def prob_both_red : ℚ := prob_red1 * prob_red2

-- Statement of the theorem to be proved
theorem prob_two_red_balls : prob_both_red = 5 / 39 := by
  sorry

end prob_two_red_balls_l116_116367


namespace sugar_snap_peas_l116_116265

theorem sugar_snap_peas (P : ℕ) (h1 : P / 7 = 72 / 9) : P = 56 := 
sorry

end sugar_snap_peas_l116_116265


namespace grid_square_division_l116_116796

theorem grid_square_division (m n k : ℕ) (h : m * m = n * k) : ℕ := sorry

end grid_square_division_l116_116796


namespace LeanProof_l116_116477

noncomputable def ProblemStatement : Prop :=
  let AB_parallel_YZ := True -- given condition that AB is parallel to YZ
  let AZ := 36 
  let BQ := 15
  let QY := 20
  let similarity_ratio := BQ / QY = 3 / 4
  ∃ QZ : ℝ, AZ = (3 / 4) * QZ + QZ ∧ QZ = 144 / 7

theorem LeanProof : ProblemStatement :=
sorry

end LeanProof_l116_116477


namespace find_S10_l116_116033

def sequence_sums (S : ℕ → ℚ) (a : ℕ → ℚ) : Prop :=
  a 1 = 1 ∧ (∀ n : ℕ, n > 0 → a (n + 1) = 3 * S n - S (n + 1) - 1)

theorem find_S10 (S a : ℕ → ℚ) (h : sequence_sums S a) : S 10 = 513 / 2 :=
  sorry

end find_S10_l116_116033


namespace quadratic_solution_property_l116_116433

theorem quadratic_solution_property (p q : ℝ)
  (h : ∀ x, 2 * x^2 + 8 * x - 42 = 0 → x = p ∨ x = q) :
  (p - q + 2) ^ 2 = 144 :=
sorry

end quadratic_solution_property_l116_116433


namespace balance_weights_l116_116545

def pair_sum {α : Type*} (l : List α) [Add α] : List (α × α) :=
  l.zip l.tail

theorem balance_weights (w : Fin 100 → ℝ) (h : ∀ i j, |w i - w j| ≤ 20) :
  ∃ (l r : Finset (Fin 100)), l.card = 50 ∧ r.card = 50 ∧
  |(l.sum w - r.sum w)| ≤ 20 :=
sorry

end balance_weights_l116_116545


namespace sum_ages_in_five_years_l116_116019

theorem sum_ages_in_five_years (L J : ℕ) (hL : L = 13) (h_relation : L = 2 * J + 3) : 
  (L + 5) + (J + 5) = 28 := 
by 
  sorry

end sum_ages_in_five_years_l116_116019


namespace unique_function_eq_id_l116_116290

theorem unique_function_eq_id (f : ℝ → ℝ) :
  (∀ x : ℝ, x ≠ 0 → f x = x^2 * f (1 / x)) →
  (∀ x y : ℝ, f (x + y) = f x + f y) →
  (f 1 = 1) →
  (∀ x : ℝ, f x = x) :=
by
  intro h1 h2 h3
  sorry

end unique_function_eq_id_l116_116290


namespace vanya_first_place_l116_116713

theorem vanya_first_place {n : ℕ} {E A : Finset ℕ} (e_v : ℕ) (a_v : ℕ)
  (he_v : e_v = n)
  (h_distinct_places : E.card = (E ∪ A).card)
  (h_all_worse : ∀ e_i ∈ E, e_i ≠ e_v → ∃ a_i ∈ A, a_i > e_i)
  : a_v = 1 := 
sorry

end vanya_first_place_l116_116713


namespace solve_for_y_l116_116910

theorem solve_for_y (y : ℕ) : 8^4 = 2^y → y = 12 :=
by
  sorry

end solve_for_y_l116_116910


namespace problem_equiv_conditions_l116_116285

theorem problem_equiv_conditions (n : ℕ) :
  (∀ a : ℕ, n ∣ a^n - a) ↔ (∀ p : ℕ, p ∣ n → Prime p → ¬ p^2 ∣ n ∧ (p - 1) ∣ (n - 1)) :=
sorry

end problem_equiv_conditions_l116_116285


namespace theta_in_fourth_quadrant_l116_116778

theorem theta_in_fourth_quadrant (θ : ℝ) (h1 : Real.cos θ > 0) (h2 : Real.tan θ < 0) : 
  (π < θ ∧ θ < 2 * π) :=
by
  sorry

end theta_in_fourth_quadrant_l116_116778


namespace distance_apart_after_3_hours_l116_116812

-- Definitions derived from conditions
def Ann_speed (hour : ℕ) : ℕ :=
  if hour = 1 then 6 else if hour = 2 then 8 else 4

def Glenda_speed (hour : ℕ) : ℕ :=
  if hour = 1 then 8 else if hour = 2 then 5 else 9

-- The total distance function for a given skater
def total_distance (speed : ℕ → ℕ) : ℕ :=
  speed 1 + speed 2 + speed 3

-- Ann's total distance skated
def Ann_total_distance : ℕ := total_distance Ann_speed

-- Glenda's total distance skated
def Glenda_total_distance : ℕ := total_distance Glenda_speed

-- The total distance between Ann and Glenda after 3 hours
def total_distance_apart : ℕ := Ann_total_distance + Glenda_total_distance

-- Proof statement (without the proof itself; just the goal declaration)
theorem distance_apart_after_3_hours : total_distance_apart = 40 := by
  sorry

end distance_apart_after_3_hours_l116_116812


namespace cost_of_coffee_A_per_kg_l116_116550

theorem cost_of_coffee_A_per_kg (x : ℝ) :
  (240 * x + 240 * 12 = 480 * 11) → x = 10 :=
by
  intros h
  sorry

end cost_of_coffee_A_per_kg_l116_116550


namespace principal_amount_l116_116392

theorem principal_amount (P R : ℝ) : 
  (P + P * R * 2 / 100 = 850) ∧ (P + P * R * 7 / 100 = 1020) → P = 782 :=
by
  sorry

end principal_amount_l116_116392


namespace find_ab_bc_value_l116_116230

theorem find_ab_bc_value
  (a b c : ℝ)
  (h : a / 3 = b / 4 ∧ b / 4 = c / 5) :
  (a + b) / (b - c) = -7 := by
sorry

end find_ab_bc_value_l116_116230


namespace numPythagoreanTriples_l116_116421

def isPythagoreanTriple (x y z : ℕ) : Prop :=
  x < y ∧ y < z ∧ x^2 + y^2 = z^2

theorem numPythagoreanTriples (n : ℕ) : ∃! T : (ℕ × ℕ × ℕ) → Prop, 
  ∀ k : ℕ, 1 ≤ k ∧ k ≤ n → (T (2^(n+1))) :=
sorry

end numPythagoreanTriples_l116_116421


namespace square_difference_l116_116297

theorem square_difference (x : ℤ) (h : x^2 = 9801) : (x + 1) * (x - 1) = 9800 :=
by {
  sorry
}

end square_difference_l116_116297


namespace fraction_of_married_men_l116_116135

/-- At a social gathering, there are only single women and married men with their wives.
     The probability that a randomly selected woman is single is 3/7.
     The fraction of the people in the gathering that are married men is 4/11. -/
theorem fraction_of_married_men (women : ℕ) (single_women : ℕ) (married_men : ℕ) (total_people : ℕ) 
  (h_women_total : women = 7)
  (h_single_women_probability : single_women = women * 3 / 7)
  (h_married_women : women - single_women = married_men)
  (h_total_people : total_people = women + married_men) :
  married_men / total_people = 4 / 11 := 
by sorry

end fraction_of_married_men_l116_116135


namespace books_sold_online_l116_116739

theorem books_sold_online (X : ℤ) 
  (h1: 743 = 502 + (37 + X) + (74 + X + 34) - 160) : 
  X = 128 := 
by sorry

end books_sold_online_l116_116739


namespace alpha_minus_beta_eq_pi_div_4_l116_116595

open Real

theorem alpha_minus_beta_eq_pi_div_4 (α β : ℝ) (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 4) 
(h : tan α = (cos β + sin β) / (cos β - sin β)) : α - β = π / 4 :=
sorry

end alpha_minus_beta_eq_pi_div_4_l116_116595


namespace pure_imaginary_solution_l116_116352

theorem pure_imaginary_solution (m : ℝ) (z : ℂ)
  (h1 : z = (m^2 - 1) + (m - 1) * I)
  (h2 : z.re = 0) : m = -1 :=
sorry

end pure_imaginary_solution_l116_116352


namespace composite_exists_for_x_64_l116_116939

-- Define the conditions
def is_composite (n : ℕ) : Prop := ∃ m k : ℕ, m > 1 ∧ k > 1 ∧ n = m * k

-- Main statement
theorem composite_exists_for_x_64 :
  ∃ n : ℕ, is_composite (n^4 + 64) :=
sorry

end composite_exists_for_x_64_l116_116939


namespace birthday_pizza_problem_l116_116161

theorem birthday_pizza_problem (m : ℕ) (h1 : m > 11) (h2 : 55 % m = 0) : 10 + 55 / m = 13 := by
  sorry

end birthday_pizza_problem_l116_116161


namespace andrew_stickers_now_l116_116412

-- Defining the conditions
def total_stickers : Nat := 1500
def ratio_susan : Nat := 1
def ratio_andrew : Nat := 1
def ratio_sam : Nat := 3
def total_ratio : Nat := ratio_susan + ratio_andrew + ratio_sam
def part : Nat := total_stickers / total_ratio
def susan_share : Nat := ratio_susan * part
def andrew_share_initial : Nat := ratio_andrew * part
def sam_share : Nat := ratio_sam * part
def sam_to_andrew : Nat := (2 * sam_share) / 3

-- Andrew's final stickers count
def andrew_share_final : Nat :=
  andrew_share_initial + sam_to_andrew

-- The theorem to prove
theorem andrew_stickers_now : andrew_share_final = 900 :=
by
  -- Proof would go here
  sorry

end andrew_stickers_now_l116_116412


namespace darnel_lap_difference_l116_116634

theorem darnel_lap_difference (sprint jog : ℝ) (h_sprint : sprint = 0.88) (h_jog : jog = 0.75) : sprint - jog = 0.13 := 
by 
  rw [h_sprint, h_jog] 
  norm_num

end darnel_lap_difference_l116_116634


namespace alcohol_percentage_new_mixture_l116_116509

namespace AlcoholMixtureProblem

def original_volume : ℝ := 3
def alcohol_percentage : ℝ := 0.33
def additional_water_volume : ℝ := 1
def new_volume : ℝ := original_volume + additional_water_volume
def alcohol_amount : ℝ := original_volume * alcohol_percentage

theorem alcohol_percentage_new_mixture : (alcohol_amount / new_volume) * 100 = 24.75 := by
  sorry

end AlcoholMixtureProblem

end alcohol_percentage_new_mixture_l116_116509


namespace open_box_volume_l116_116344

-- Define the initial conditions
def length_of_sheet := 100
def width_of_sheet := 50
def height_of_parallelogram := 10
def base_of_parallelogram := 10

-- Define the expected dimensions of the box after cutting
def length_of_box := length_of_sheet - 2 * base_of_parallelogram
def width_of_box := width_of_sheet - 2 * base_of_parallelogram
def height_of_box := height_of_parallelogram

-- Define the expected volume of the box
def volume_of_box := length_of_box * width_of_box * height_of_box

-- Theorem to prove the correct volume of the box based on the given dimensions
theorem open_box_volume : volume_of_box = 24000 := by
  -- The proof will be included here
  sorry

end open_box_volume_l116_116344


namespace sum_of_x_and_reciprocal_eq_3_5_l116_116444

theorem sum_of_x_and_reciprocal_eq_3_5
    (x : ℝ)
    (h : x^2 + (1 / x^2) = 10.25) :
    x + (1 / x) = 3.5 := 
by
  sorry

end sum_of_x_and_reciprocal_eq_3_5_l116_116444


namespace sufficient_but_not_necessary_for_reciprocal_l116_116716

theorem sufficient_but_not_necessary_for_reciprocal (x : ℝ) : (x > 1 → 1/x < 1) ∧ (¬ (1/x < 1 → x > 1)) :=
by
  sorry

end sufficient_but_not_necessary_for_reciprocal_l116_116716


namespace parallelogram_point_D_l116_116568

/-- Given points A, B, and C, the coordinates of point D in parallelogram ABCD -/
theorem parallelogram_point_D (A B C D : (ℝ × ℝ))
  (hA : A = (1, 1))
  (hB : B = (3, 2))
  (hC : C = (6, 3))
  (hMid : (2 * (A.1 + C.1), 2 * (A.2 + C.2)) = (2 * (B.1 + D.1), 2 * (B.2 + D.2))) :
  D = (4, 2) :=
sorry

end parallelogram_point_D_l116_116568


namespace div_pow_eq_l116_116982

theorem div_pow_eq : 23^11 / 23^5 = 148035889 := by
  sorry

end div_pow_eq_l116_116982


namespace compare_fractions_l116_116233

theorem compare_fractions : (-1 / 3) > (-1 / 2) := sorry

end compare_fractions_l116_116233


namespace product_of_integers_l116_116361

theorem product_of_integers
  (A B C D : ℕ)
  (hA : A > 0)
  (hB : B > 0)
  (hC : C > 0)
  (hD : D > 0)
  (h_sum : A + B + C + D = 72)
  (h_eq : A + 3 = B - 3 ∧ B - 3 = C * 3 ∧ C * 3 = D / 2) :
  A * B * C * D = 68040 := 
by
  sorry

end product_of_integers_l116_116361


namespace roberta_started_with_8_records_l116_116533

variable (R : ℕ)

def received_records := 12
def bought_records := 30
def total_received_and_bought := received_records + bought_records

theorem roberta_started_with_8_records (h : R + total_received_and_bought = 50) : R = 8 :=
by
  sorry

end roberta_started_with_8_records_l116_116533


namespace waiter_earnings_l116_116731

theorem waiter_earnings (total_customers tipping_customers no_tip_customers tips_each : ℕ) (h1 : total_customers = 7) (h2 : no_tip_customers = 4) (h3 : tips_each = 9) (h4 : tipping_customers = total_customers - no_tip_customers) :
  tipping_customers * tips_each = 27 :=
by sorry

end waiter_earnings_l116_116731


namespace largest_sum_l116_116303

theorem largest_sum :
  max (max (max (max (1/4 + 1/9) (1/4 + 1/10)) (1/4 + 1/11)) (1/4 + 1/12)) (1/4 + 1/13) = 13/36 := 
sorry

end largest_sum_l116_116303


namespace triangle_area_is_2_l116_116588

noncomputable def area_of_triangle_OAB {x₀ : ℝ} (h₀ : 0 < x₀) : ℝ :=
  let y₀ := 1 / x₀
  let slope := -1 / x₀^2
  let tangent_line (x : ℝ) := y₀ + slope * (x - x₀)
  let A : ℝ × ℝ := (2 * x₀, 0) -- Intersection with x-axis
  let B : ℝ × ℝ := (0, 2 * y₀) -- Intersection with y-axis
  1 / 2 * abs (2 * y₀ * 2 * x₀)

theorem triangle_area_is_2 (x₀ : ℝ) (h₀ : 0 < x₀) : area_of_triangle_OAB h₀ = 2 :=
by
  sorry

end triangle_area_is_2_l116_116588


namespace triangle_type_l116_116692

theorem triangle_type (a b c : ℝ) (A B C : ℝ) (h1 : A = 30) (h2 : a = 2 * b ∨ b = 2 * c ∨ c = 2 * a) :
  (C > 90 ∨ B > 90) ∨ C = 90 :=
sorry

end triangle_type_l116_116692


namespace harry_terry_difference_l116_116082

-- Define Harry's answer
def H : ℤ := 8 - (2 + 5)

-- Define Terry's answer
def T : ℤ := 8 - 2 + 5

-- State the theorem to prove H - T = -10
theorem harry_terry_difference : H - T = -10 := by
  sorry

end harry_terry_difference_l116_116082


namespace prove_union_l116_116949

variable (M N : Set ℕ)
variable (x : ℕ)

def M_definition := (0 ∈ M) ∧ (x ∈ M) ∧ (M = {0, x})
def N_definition := (N = {1, 2})
def intersection_condition := (M ∩ N = {2})
def union_result := (M ∪ N = {0, 1, 2})

theorem prove_union (M : Set ℕ) (N : Set ℕ) (x : ℕ) :
  M_definition M x → N_definition N → intersection_condition M N → union_result M N :=
by
  sorry

end prove_union_l116_116949


namespace calculate_cakes_left_l116_116849

-- Define the conditions
def b_lunch : ℕ := 5
def s_dinner : ℕ := 6
def b_yesterday : ℕ := 3

-- Define the calculation of the total cakes baked and cakes left
def total_baked : ℕ := b_lunch + b_yesterday
def cakes_left : ℕ := total_baked - s_dinner

-- The theorem we want to prove
theorem calculate_cakes_left : cakes_left = 2 := 
by
  sorry

end calculate_cakes_left_l116_116849


namespace conference_handshakes_l116_116482

theorem conference_handshakes (n : ℕ) (h : n = 10) :
  (n * (n - 1)) / 2 = 45 :=
by
  sorry

end conference_handshakes_l116_116482


namespace remainder_when_divided_by_x_minus_2_l116_116428

def p (x : ℤ) : ℤ := x^5 + x^3 + x + 3

theorem remainder_when_divided_by_x_minus_2 :
  p 2 = 45 :=
by
  sorry

end remainder_when_divided_by_x_minus_2_l116_116428


namespace exists_even_in_sequence_l116_116192

theorem exists_even_in_sequence 
  (a : ℕ → ℕ)
  (h₀ : ∀ n : ℕ, a (n+1) = a n + (a n % 10)) :
  ∃ n : ℕ, a n % 2 = 0 :=
sorry

end exists_even_in_sequence_l116_116192


namespace fraction_red_knights_magical_l116_116226

theorem fraction_red_knights_magical (total_knights red_knights blue_knights magical_knights : ℕ)
  (fraction_red fraction_magical : ℚ)
  (frac_red_mag : ℚ) :
  (red_knights = total_knights * fraction_red) →
  (fraction_red = 3 / 8) →
  (magical_knights = total_knights * fraction_magical) →
  (fraction_magical = 1 / 4) →
  (frac_red_mag * red_knights + (frac_red_mag / 3) * blue_knights = magical_knights) →
  (frac_red_mag = 3 / 7) :=
by
  -- Skipping proof
  sorry

end fraction_red_knights_magical_l116_116226


namespace principal_amount_unique_l116_116188

theorem principal_amount_unique (SI R T : ℝ) (P : ℝ) : 
  SI = 4016.25 → R = 14 → T = 5 → SI = (P * R * T) / 100 → P = 5737.5 :=
by
  intro h₁ h₂ h₃ h₄
  rw [h₁, h₂, h₃] at h₄
  sorry

end principal_amount_unique_l116_116188


namespace stream_current_l116_116051

noncomputable def solve_stream_current : Prop :=
  ∃ (r w : ℝ), (24 / (r + w) + 6 = 24 / (r - w)) ∧ (24 / (3 * r + w) + 2 = 24 / (3 * r - w)) ∧ (w = 2)

theorem stream_current : solve_stream_current :=
  sorry

end stream_current_l116_116051


namespace negate_universal_proposition_l116_116099

theorem negate_universal_proposition : 
  (¬ (∀ x : ℝ, x^2 - 2 * x + 1 > 0)) ↔ (∃ x : ℝ, x^2 - 2 * x + 1 ≤ 0) :=
by sorry

end negate_universal_proposition_l116_116099


namespace Cheerful_snakes_not_Green_l116_116353

variables {Snake : Type} (snakes : Finset Snake)
variable (Cheerful Green CanSing CanMultiply : Snake → Prop)

-- Conditions
axiom Cheerful_impl_CanSing : ∀ s, Cheerful s → CanSing s
axiom Green_impl_not_CanMultiply : ∀ s, Green s → ¬ CanMultiply s
axiom not_CanMultiply_impl_not_CanSing : ∀ s, ¬ CanMultiply s → ¬ CanSing s

-- Question
theorem Cheerful_snakes_not_Green : ∀ s, Cheerful s → ¬ Green s :=
by sorry

end Cheerful_snakes_not_Green_l116_116353


namespace a2_a8_sum_l116_116124

variable {a : ℕ → ℝ}  -- Define the arithmetic sequence a

-- Conditions:
axiom arithmetic_sequence (n : ℕ) : a (n + 1) - a n = a 1 - a 0
axiom a1_a9_sum : a 1 + a 9 = 8

-- Theorem stating the question and the answer
theorem a2_a8_sum : a 2 + a 8 = 8 :=
by
  sorry

end a2_a8_sum_l116_116124


namespace original_rectangle_area_at_least_90_l116_116102

variable (a b c x y z : ℝ)
variable (hx1 : a * x = 1)
variable (hx2 : c * x = 3)
variable (hy : b * y = 10)
variable (hz : a * z = 9)
variable (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
variable (hx : 0 < x) (hy' : 0 < y) (hz' : 0 < z)

theorem original_rectangle_area_at_least_90 : ∀ {a b c x y z : ℝ},
  (a * x = 1) →
  (c * x = 3) →
  (b * y = 10) →
  (a * z = 9) →
  (0 < a) →
  (0 < b) →
  (0 < c) →
  (0 < x) →
  (0 < y) →
  (0 < z) →
  (a + b + c) * (x + y + z) ≥ 90 :=
sorry

end original_rectangle_area_at_least_90_l116_116102


namespace B_work_days_l116_116850

theorem B_work_days (x : ℝ) :
  (1 / 3 + 1 / x = 1 / 2) → x = 6 := by
  sorry

end B_work_days_l116_116850


namespace power_of_two_l116_116014

theorem power_of_two (Number : ℕ) (h1 : Number = 128) (h2 : Number * (1/4 : ℝ) = 2^5) :
  ∃ power : ℕ, 2^power = 128 := 
by
  use 7
  sorry

end power_of_two_l116_116014


namespace square_side_length_l116_116512

theorem square_side_length (d : ℝ) (s : ℝ) (h : d = Real.sqrt 2) (h2 : d = Real.sqrt 2 * s) : s = 1 :=
by
  sorry

end square_side_length_l116_116512


namespace certain_number_l116_116985

theorem certain_number (G : ℕ) (N : ℕ) (H1 : G = 129) 
  (H2 : N % G = 9) (H3 : 2206 % G = 13) : N = 2202 :=
by
  sorry

end certain_number_l116_116985


namespace alice_has_ball_after_three_turns_l116_116101

def probability_Alice_has_ball (turns: ℕ) : ℚ :=
  match turns with
  | 0 => 1 -- Alice starts with the ball
  | _ => sorry -- We would typically calculate this by recursion or another approach.

theorem alice_has_ball_after_three_turns :
  probability_Alice_has_ball 3 = 11 / 27 :=
by
  sorry

end alice_has_ball_after_three_turns_l116_116101


namespace jacket_cost_is_30_l116_116539

-- Let's define the given conditions
def num_dresses := 5
def cost_per_dress := 20 -- dollars
def num_pants := 3
def cost_per_pant := 12 -- dollars
def num_jackets := 4
def transport_cost := 5 -- dollars
def initial_amount := 400 -- dollars
def remaining_amount := 139 -- dollars

-- Define the cost per jacket
def cost_per_jacket := 30 -- dollars

-- Final theorem statement to be proved
theorem jacket_cost_is_30:
  num_dresses * cost_per_dress + num_pants * cost_per_pant + num_jackets * cost_per_jacket + transport_cost = initial_amount - remaining_amount :=
sorry

end jacket_cost_is_30_l116_116539


namespace tank_filling_time_l116_116771

theorem tank_filling_time (p q r s : ℝ) (leakage : ℝ) :
  (p = 1 / 6) →
  (q = 1 / 12) →
  (r = 1 / 24) →
  (s = 1 / 18) →
  (leakage = -1 / 48) →
  (1 / (p + q + r + s + leakage) = 48 / 15.67) :=
by
  intros hp hq hr hs hleak
  rw [hp, hq, hr, hs, hleak]
  norm_num
  sorry

end tank_filling_time_l116_116771


namespace hancho_tape_length_l116_116888

noncomputable def tape_length (x : ℝ) : Prop :=
  (1 / 4) * (4 / 5) * x = 1.5

theorem hancho_tape_length : ∃ x : ℝ, tape_length x ∧ x = 7.5 :=
by sorry

end hancho_tape_length_l116_116888


namespace find_circle_equation_l116_116180

-- Define the intersection point of the lines x + y + 1 = 0 and x - y - 1 = 0
def center : ℝ × ℝ := (0, -1)

-- Define the chord length AB
def chord_length : ℝ := 6

-- Line equation that intersects the circle
def line_eq (x y : ℝ) : Prop := 3 * x + 4 * y - 11 = 0

-- Circle equation to be proven
def circle_eq (x y : ℝ) : Prop := x^2 + (y + 1)^2 = 18

-- Main theorem: Prove that the given circle equation is correct under the conditions
theorem find_circle_equation (x y : ℝ) (hc : x + y + 1 = 0) (hc' : x - y - 1 = 0) 
  (hl : line_eq x y) : circle_eq x y :=
sorry

end find_circle_equation_l116_116180


namespace range_of_m_l116_116710

theorem range_of_m (m : ℝ) :
  (∃ (x : ℤ), (x > 2 * m ∧ x ≥ m - 3) ∧ x = 1) ↔ 0 ≤ m ∧ m < 0.5 :=
by
  sorry

end range_of_m_l116_116710


namespace collinear_points_cube_l116_116867

-- Define a function that counts the sets of three collinear points in the described structure.
def count_collinear_points : Nat :=
  -- Placeholders for the points (vertices, edge midpoints, face centers, center of the cube) and the count logic
  -- The calculation logic will be implemented as the proof
  49

theorem collinear_points_cube : count_collinear_points = 49 :=
  sorry

end collinear_points_cube_l116_116867


namespace sin_cos_expression_l116_116652

noncomputable def sin_45 := Real.sin (Real.pi / 4)
noncomputable def cos_15 := Real.cos (Real.pi / 12)
noncomputable def cos_225 := Real.cos (5 * Real.pi / 4)
noncomputable def sin_15 := Real.sin (Real.pi / 12)

theorem sin_cos_expression :
  sin_45 * cos_15 + cos_225 * sin_15 = 1 / 2 :=
by
  sorry

end sin_cos_expression_l116_116652


namespace factorization_identity_l116_116069

theorem factorization_identity (a b : ℝ) : 
  -a^3 + 12 * a^2 * b - 36 * a * b^2 = -a * (a - 6 * b)^2 :=
by 
  sorry

end factorization_identity_l116_116069


namespace maria_fraction_of_remaining_distance_l116_116516

theorem maria_fraction_of_remaining_distance (total_distance remaining_distance distance_travelled : ℕ) 
(h_total : total_distance = 480) 
(h_first_stop : distance_travelled = total_distance / 2) 
(h_remaining : remaining_distance = total_distance - distance_travelled)
(h_final_leg : remaining_distance - distance_travelled = 180) : 
(distance_travelled / remaining_distance) = (1 / 4) := 
by
  sorry

end maria_fraction_of_remaining_distance_l116_116516


namespace jamal_books_remaining_l116_116408

variable (initial_books : ℕ := 51)
variable (history_books : ℕ := 12)
variable (fiction_books : ℕ := 19)
variable (children_books : ℕ := 8)
variable (misplaced_books : ℕ := 4)

theorem jamal_books_remaining : 
  initial_books - history_books - fiction_books - children_books + misplaced_books = 16 := by
  sorry

end jamal_books_remaining_l116_116408


namespace book_length_l116_116083

variable (length width perimeter : ℕ)

theorem book_length
  (h1 : perimeter = 100)
  (h2 : width = 20)
  (h3 : perimeter = 2 * (length + width)) :
  length = 30 :=
by sorry

end book_length_l116_116083


namespace number_of_girls_l116_116630

/-- In a school with 632 students, the average age of the boys is 12 years
and that of the girls is 11 years. The average age of the school is 11.75 years.
How many girls are there in the school? Prove that the number of girls is 108. -/
theorem number_of_girls (B G : ℕ) (h1 : B + G = 632) (h2 : 12 * B + 11 * G = 7428) :
  G = 108 :=
sorry

end number_of_girls_l116_116630


namespace unique_prime_p_l116_116968

theorem unique_prime_p (p : ℕ) (hp : Nat.Prime p) (h : Nat.Prime (p^2 + 2)) : p = 3 := 
by 
  sorry

end unique_prime_p_l116_116968


namespace percentage_subtraction_l116_116462

theorem percentage_subtraction (P : ℝ) : (700 - (P / 100 * 7000) = 700) → P = 0 :=
by
  sorry

end percentage_subtraction_l116_116462


namespace WangLi_final_score_l116_116638

def weightedFinalScore (writtenScore : ℕ) (demoScore : ℕ) (interviewScore : ℕ)
    (writtenWeight : ℕ) (demoWeight : ℕ) (interviewWeight : ℕ) : ℕ :=
  (writtenScore * writtenWeight + demoScore * demoWeight + interviewScore * interviewWeight) /
  (writtenWeight + demoWeight + interviewWeight)

theorem WangLi_final_score :
  weightedFinalScore 96 90 95 5 3 2 = 94 :=
  by
  -- proof goes here
  sorry

end WangLi_final_score_l116_116638


namespace cos_neg_three_pi_over_two_eq_zero_l116_116123

noncomputable def cos_neg_three_pi_over_two : ℝ :=
  Real.cos (-3 * Real.pi / 2)

theorem cos_neg_three_pi_over_two_eq_zero :
  cos_neg_three_pi_over_two = 0 :=
by
  -- Using trigonometric identities and periodicity of cosine function
  sorry

end cos_neg_three_pi_over_two_eq_zero_l116_116123


namespace initial_population_l116_116969

theorem initial_population (P : ℝ)
  (h1 : P * 1.25 * 0.75 = 18750) : P = 20000 :=
sorry

end initial_population_l116_116969


namespace mooney_ate_correct_l116_116540

-- Define initial conditions
def initial_brownies : ℕ := 24
def father_ate : ℕ := 8
def mother_added : ℕ := 24
def final_brownies : ℕ := 36

-- Define Mooney ate some brownies
variable (mooney_ate : ℕ)

-- Prove that Mooney ate 4 brownies
theorem mooney_ate_correct :
  (initial_brownies - father_ate) - mooney_ate + mother_added = final_brownies →
  mooney_ate = 4 :=
by
  sorry

end mooney_ate_correct_l116_116540


namespace smallest_four_digit_number_l116_116302

theorem smallest_four_digit_number :
  ∃ m : ℕ, (1000 ≤ m) ∧ (m < 10000) ∧ (∃ n : ℕ, 21 * m = n^2) ∧ m = 1029 :=
by sorry

end smallest_four_digit_number_l116_116302


namespace variance_of_dataSet_l116_116143

-- Define the given data set
def dataSet : List ℤ := [-2, -1, 0, 1, 2]

-- Define the function to calculate mean
def mean (data : List ℤ) : ℚ :=
  (data.sum : ℚ) / data.length

-- Define the function to calculate variance
def variance (data : List ℤ) : ℚ :=
  let μ := mean data
  (data.map (λ x => (x - μ) ^ 2)).sum / data.length

-- State the theorem: The variance of the given data set is 2
theorem variance_of_dataSet : variance dataSet = 2 := by
  sorry

end variance_of_dataSet_l116_116143


namespace neg_abs_value_eq_neg_three_l116_116252

theorem neg_abs_value_eq_neg_three : -|-3| = -3 := 
by sorry

end neg_abs_value_eq_neg_three_l116_116252


namespace probability_at_least_9_heads_in_12_flips_l116_116017

theorem probability_at_least_9_heads_in_12_flips :
  let total_outcomes := 2^12
  let favorable_outcomes := (Nat.choose 12 9) + (Nat.choose 12 10) + (Nat.choose 12 11) + (Nat.choose 12 12)
  let probability := favorable_outcomes / total_outcomes
  probability = 299 / 4096 := 
by
  sorry

end probability_at_least_9_heads_in_12_flips_l116_116017


namespace zoey_holidays_in_a_year_l116_116451

-- Definitions based on the conditions
def holidays_per_month := 2
def months_in_year := 12

-- Lean statement representing the proof problem
theorem zoey_holidays_in_a_year : (holidays_per_month * months_in_year) = 24 :=
by sorry

end zoey_holidays_in_a_year_l116_116451


namespace error_percentage_in_area_l116_116561

theorem error_percentage_in_area
  (L W : ℝ)          -- Actual length and width of the rectangle
  (hL' : ℝ)          -- Measured length with 8% excess
  (hW' : ℝ)          -- Measured width with 5% deficit
  (hL'_def : hL' = 1.08 * L)  -- Condition for length excess
  (hW'_def : hW' = 0.95 * W)  -- Condition for width deficit
  :
  ((hL' * hW' - L * W) / (L * W) * 100 = 2.6) := sorry

end error_percentage_in_area_l116_116561


namespace why_build_offices_l116_116268

structure Company where
  name : String
  hasSkillfulEmployees : Prop
  uniqueComfortableWorkEnvironment : Prop
  integratedWorkLeisureSpaces : Prop
  reducedEmployeeStress : Prop
  flexibleWorkSchedules : Prop
  increasesProfit : Prop

theorem why_build_offices (goog_fb : Company)
  (h1 : goog_fb.hasSkillfulEmployees)
  (h2 : goog_fb.uniqueComfortableWorkEnvironment)
  (h3 : goog_fb.integratedWorkLeisureSpaces)
  (h4 : goog_fb.reducedEmployeeStress)
  (h5 : goog_fb.flexibleWorkSchedules) :
  goog_fb.increasesProfit := 
sorry

end why_build_offices_l116_116268


namespace solve_equation_one_solve_equation_two_l116_116360

theorem solve_equation_one (x : ℝ) : (x - 3) ^ 2 - 4 = 0 ↔ x = 5 ∨ x = 1 := sorry

theorem solve_equation_two (x : ℝ) : (x + 2) ^ 2 - 2 * (x + 2) = 3 ↔ x = 1 ∨ x = -1 := sorry

end solve_equation_one_solve_equation_two_l116_116360


namespace horner_first_calculation_at_3_l116_116182

def f (x : ℝ) : ℝ :=
  0.5 * x ^ 6 + 4 * x ^ 5 - x ^ 4 + 3 * x ^ 3 - 5 * x

def horner_first_step (x : ℝ) : ℝ :=
  0.5 * x + 4

theorem horner_first_calculation_at_3 :
  horner_first_step 3 = 5.5 := by
  sorry

end horner_first_calculation_at_3_l116_116182


namespace domain_of_sqrt_2cosx_plus_1_l116_116828

noncomputable def domain_sqrt_2cosx_plus_1 (x : ℝ) : Prop :=
  ∃ (k : ℤ), (2 * k * Real.pi - 2 * Real.pi / 3) ≤ x ∧ x ≤ (2 * k * Real.pi + 2 * Real.pi / 3)

theorem domain_of_sqrt_2cosx_plus_1 :
  (∀ (x: ℝ), 0 ≤ 2 * Real.cos x + 1 ↔ domain_sqrt_2cosx_plus_1 x) :=
by
  sorry

end domain_of_sqrt_2cosx_plus_1_l116_116828


namespace polygon_length_l116_116037

noncomputable def DE : ℝ := 3
noncomputable def EF : ℝ := 6
noncomputable def DE_plus_EF : ℝ := DE + EF

theorem polygon_length 
  (area_ABCDEF : ℝ)
  (AB BC FA : ℝ)
  (A B C D E F : ℝ × ℝ) :
  area_ABCDEF = 60 →
  AB = 10 →
  BC = 7 →
  FA = 6 →
  A = (0, 10) →
  B = (10, 10) →
  C = (10, 0) →
  D = (6, 0) →
  E = (6, 3) →
  F = (0, 3) →
  DE_plus_EF = 9 :=
by
  intros
  sorry

end polygon_length_l116_116037


namespace sum_integers_neg50_to_60_l116_116007

theorem sum_integers_neg50_to_60 : 
  (Finset.sum (Finset.Icc (-50 : ℤ) 60) id) = 555 := 
by
  -- Placeholder for the actual proof
  sorry

end sum_integers_neg50_to_60_l116_116007


namespace count_integers_six_times_sum_of_digits_l116_116028

theorem count_integers_six_times_sum_of_digits (n : ℕ) (h : n < 1000) 
    (digit_sum : ℕ → ℕ)
    (digit_sum_correct : ∀ (n : ℕ), digit_sum n = (n % 10) + ((n / 10) % 10) + (n / 100)) :
    ∃! n, n < 1000 ∧ n = 6 * digit_sum n :=
sorry

end count_integers_six_times_sum_of_digits_l116_116028


namespace total_surface_area_correct_l116_116100

def surface_area_calculation (height_e height_f height_g : ℚ) : ℚ :=
  let top_bottom_area := 4
  let side_area := (height_e + height_f + height_g) * 2
  let front_back_area := 4
  top_bottom_area + side_area + front_back_area

theorem total_surface_area_correct :
  surface_area_calculation (5 / 8) (1 / 4) (9 / 8) = 12 := 
by
  sorry

end total_surface_area_correct_l116_116100


namespace speed_of_current_l116_116427

theorem speed_of_current (c r : ℝ) 
  (h1 : 12 = (c - r) * 6) 
  (h2 : 12 = (c + r) * 0.75) : 
  r = 7 := 
by
  sorry

end speed_of_current_l116_116427


namespace mark_cans_correct_l116_116507

variable (R : ℕ) -- Rachel's cans
variable (J : ℕ) -- Jaydon's cans
variable (M : ℕ) -- Mark's cans
variable (T : ℕ) -- Total cans 

-- Conditions
def jaydon_cans (R : ℕ) : ℕ := 2 * R + 5
def mark_cans (J : ℕ) : ℕ := 4 * J
def total_cans (R : ℕ) (J : ℕ) (M : ℕ) : ℕ := R + J + M

theorem mark_cans_correct (R : ℕ) (J : ℕ) 
  (h1 : J = jaydon_cans R) 
  (h2 : M = mark_cans J) 
  (h3 : total_cans R J M = 135) : 
  M = 100 := 
sorry

end mark_cans_correct_l116_116507


namespace surveyor_problem_l116_116166

theorem surveyor_problem
  (GF : ℝ) (G4 : ℝ)
  (hGF : GF = 70)
  (hG4 : G4 = 60) :
  (1/2) * GF * G4 = 2100 := 
  by
  sorry

end surveyor_problem_l116_116166


namespace compare_P_Q_l116_116756

noncomputable def P : ℝ := Real.sqrt 7 - 1
noncomputable def Q : ℝ := Real.sqrt 11 - Real.sqrt 5

theorem compare_P_Q : P > Q :=
sorry

end compare_P_Q_l116_116756


namespace measure_of_B_l116_116142

theorem measure_of_B (A B C : ℝ) (h1 : B = A + 20) (h2 : C = 50) (h3 : A + B + C = 180) : B = 75 := by
  sorry

end measure_of_B_l116_116142


namespace proportion_in_triangle_l116_116116

-- Definitions of the variables and conditions
variables {P Q R E : Point}
variables {p q r m n : ℝ}

-- Conditions
def angle_bisector_theorem (h : p = 2 * q) (h1 : m = q + q) (h2 : n = 2 * q) : Prop :=
  ∀ (p q r m n : ℝ), 
  (m / r) = (n / q) ∧ 
  (m + n = p) ∧
  (p = 2 * q)

-- The theorem to be proved
theorem proportion_in_triangle (h : p = 2 * q) (h1 : m / r = n / q) (h2 : m + n = p) : 
  (n / q = 2 * q / (r + q)) :=
by
  sorry

end proportion_in_triangle_l116_116116


namespace original_number_is_106_25_l116_116472

theorem original_number_is_106_25 (x : ℝ) (h : (x + 0.375 * x) - (x - 0.425 * x) = 85) : x = 106.25 := by
  sorry

end original_number_is_106_25_l116_116472


namespace inverse_function_property_l116_116426

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := a ^ x
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

theorem inverse_function_property (a : ℝ) (h : g a 2 = 4) : f a 2 = 1 := by
  have g_inverse_f : g a (f a 2) = 2 := by sorry
  have a_value : a = 2 := by sorry
  rw [a_value]
  sorry

end inverse_function_property_l116_116426


namespace solution_of_inequality_l116_116080

theorem solution_of_inequality (x : ℝ) : -2 * x - 1 < -1 → x > 0 :=
by
  sorry

end solution_of_inequality_l116_116080


namespace kim_saplings_left_l116_116225

def sprouted_pits (total_pits num_sprouted_pits: ℕ) (percent_sprouted: ℝ) : Prop :=
  percent_sprouted * total_pits = num_sprouted_pits

def sold_saplings (total_saplings saplings_sold saplings_left: ℕ) : Prop :=
  total_saplings - saplings_sold = saplings_left

theorem kim_saplings_left
  (total_pits : ℕ) (num_sprouted_pits : ℕ) (percent_sprouted : ℝ)
  (saplings_sold : ℕ) (saplings_left : ℕ) :
  total_pits = 80 →
  percent_sprouted = 0.25 →
  saplings_sold = 6 →
  sprouted_pits total_pits num_sprouted_pits percent_sprouted →
  sold_saplings num_sprouted_pits saplings_sold saplings_left →
  saplings_left = 14 :=
by
  intros
  sorry

end kim_saplings_left_l116_116225


namespace least_value_MX_l116_116464

-- Definitions of points and lines
variables (A B C D M P X : ℝ × ℝ)
variables (y : ℝ)

-- Hypotheses based on the conditions
variables (h1 : A = (0, 0))
variables (h2 : B = (33, 0))
variables (h3 : C = (33, 56))
variables (h4 : D = (0, 56))
variables (h5 : M = (33 / 2, 0)) -- M is midpoint of AB
variables (h6 : P = (33, y)) -- P is on BC
variables (hy_range : 0 ≤ y ∧ y ≤ 56) -- y is within the bounds of BC

-- Additional derived hypotheses needed for the proof
variables (h7 : ∃ x, X = (x, sqrt (816.75))) -- X is intersection point on DA

-- The theorem statement
theorem least_value_MX : ∃ y, 0 ≤ y ∧ y ≤ 56 ∧ MX = 33 :=
by
  use 28
  sorry

end least_value_MX_l116_116464


namespace value_of_expression_l116_116897

theorem value_of_expression (x y : ℚ) (hx : x ≠ 0) (hy : y ≠ 0) :
  (x / |x| + |y| / y = 2) ∨ (x / |x| + |y| / y = 0) ∨ (x / |x| + |y| / y = -2) :=
by
  sorry

end value_of_expression_l116_116897


namespace find_p_value_l116_116004

noncomputable def solve_p (m p : ℕ) :=
  (1^m / 5^m) * (1^16 / 4^16) = 1 / (2 * p^31)

theorem find_p_value (m p : ℕ) (hm : m = 31) :
  solve_p m p ↔ p = 10 :=
by
  sorry

end find_p_value_l116_116004


namespace jane_oldest_child_age_l116_116260

-- Define the conditions
def jane_start_age : ℕ := 20
def jane_current_age : ℕ := 32
def stopped_babysitting_years_ago : ℕ := 10
def baby_sat_condition (jane_age child_age : ℕ) : Prop := child_age ≤ jane_age / 2

-- Define the proof problem
theorem jane_oldest_child_age :
  (∃ age_stopped child_age,
    stopped_babysitting_years_ago = jane_current_age - age_stopped ∧
    baby_sat_condition age_stopped child_age ∧
    (32 - stopped_babysitting_years_ago = 22) ∧ -- Jane's age when she stopped baby-sitting
    child_age = 22 / 2 ∧ -- Oldest child she could have baby-sat at age 22
    child_age + stopped_babysitting_years_ago = 21) --  current age of the oldest person for whom Jane could have baby-sat
:= sorry

end jane_oldest_child_age_l116_116260


namespace shpuntik_can_form_triangle_l116_116269

-- Define lengths of the sticks before swap
variables {a b c d e f : ℝ}

-- Conditions before the swap
-- Both sets of sticks can form a triangle
-- The lengths of Vintik's sticks are a, b, c
-- The lengths of Shpuntik's sticks are d, e, f
axiom triangle_ineq_vintik : a + b > c ∧ b + c > a ∧ c + a > b
axiom triangle_ineq_shpuntik : d + e > f ∧ e + f > d ∧ f + d > e
axiom sum_lengths_vintik : a + b + c = 1
axiom sum_lengths_shpuntik : d + e + f = 1

-- Define lengths of the sticks after swap
-- x1, x2, x3 are Vintik's new sticks; y1, y2, y3 are Shpuntik's new sticks
variables {x1 x2 x3 y1 y2 y3 : ℝ}

-- Neznaika's swap
axiom swap_stick_vintik : x1 = a ∧ x2 = b ∧ x3 = f ∨ x1 = a ∧ x2 = d ∧ x3 = c ∨ x1 = e ∧ x2 = b ∧ x3 = c
axiom swap_stick_shpuntik : y1 = d ∧ y2 = e ∧ y3 = c ∨ y1 = e ∧ y2 = b ∧ y3 = f ∨ y1 = a ∧ y2 = b ∧ y3 = f 

-- Total length after the swap remains unchanged
axiom sum_lengths_after_swap : x1 + x2 + x3 + y1 + y2 + y3 = 2

-- Vintik cannot form a triangle with the current lengths
axiom no_triangle_vintik : x1 >= x2 + x3

-- Prove that Shpuntik can still form a triangle
theorem shpuntik_can_form_triangle : y1 + y2 > y3 ∧ y2 + y3 > y1 ∧ y3 + y1 > y2 := sorry

end shpuntik_can_form_triangle_l116_116269


namespace population_increase_20th_century_l116_116337

theorem population_increase_20th_century (P : ℕ) :
  let population_mid_century := 3 * P
  let population_end_century := 12 * P
  (population_end_century - P) / P * 100 = 1100 :=
by
  sorry

end population_increase_20th_century_l116_116337


namespace line_equation_sum_l116_116250

theorem line_equation_sum (m b x y : ℝ) (hx : x = 4) (hy : y = 2) (hm : m = -5) (hline : y = m * x + b) : m + b = 17 := by
  sorry

end line_equation_sum_l116_116250


namespace total_trees_correct_l116_116097

def apricot_trees : ℕ := 58
def peach_trees : ℕ := 3 * apricot_trees
def total_trees : ℕ := apricot_trees + peach_trees

theorem total_trees_correct : total_trees = 232 :=
by
  sorry

end total_trees_correct_l116_116097


namespace percent_of_g_is_a_l116_116998

theorem percent_of_g_is_a (a b c d e f g : ℤ) (h1 : (a + b + c + d + e + f + g) / 7 = 9)
: (a / g) * 100 = 50 := 
sorry

end percent_of_g_is_a_l116_116998


namespace exp_mul_l116_116160

variable {a : ℝ}

-- Define a theorem stating the problem: proof that a^2 * a^3 = a^5
theorem exp_mul (a : ℝ) : a^2 * a^3 = a^5 := by
  sorry

end exp_mul_l116_116160


namespace Grant_spending_is_200_l116_116162

def Juanita_daily_spending (day: String) : Float :=
  if day = "Sunday" then 2.0 else 0.5

def Juanita_weekly_spending : Float :=
  6 * Juanita_daily_spending "weekday" + Juanita_daily_spending "Sunday"

def Juanita_yearly_spending : Float :=
  52 * Juanita_weekly_spending

def Grant_yearly_spending := Juanita_yearly_spending - 60

theorem Grant_spending_is_200 : Grant_yearly_spending = 200 := by
  sorry

end Grant_spending_is_200_l116_116162


namespace intersection_of_A_and_B_l116_116044

open Set

theorem intersection_of_A_and_B :
  let A := {x : ℝ | x^2 - x ≤ 0}
  let B := ({0, 1, 2} : Set ℝ)
  A ∩ B = ({0, 1} : Set ℝ) :=
by
  sorry

end intersection_of_A_and_B_l116_116044


namespace reciprocal_neg_3_div_4_l116_116111

theorem reciprocal_neg_3_div_4 : (- (3 / 4 : ℚ))⁻¹ = -(4 / 3 : ℚ) :=
by
  sorry

end reciprocal_neg_3_div_4_l116_116111


namespace compare_negatives_l116_116614

theorem compare_negatives : -1 < - (2 / 3) := by
  sorry

end compare_negatives_l116_116614


namespace height_on_fifth_bounce_l116_116065

-- Define initial conditions
def initial_height : ℝ := 96
def initial_efficiency : ℝ := 0.5
def efficiency_decrease : ℝ := 0.05
def air_resistance_loss : ℝ := 0.02

-- Recursive function to compute the height after each bounce
def bounce_height (height : ℝ) (efficiency : ℝ) : ℝ :=
  let height_after_bounce := height * efficiency
  height_after_bounce - (height_after_bounce * air_resistance_loss)

-- Function to compute the bounce efficiency after each bounce
def bounce_efficiency (initial_efficiency : ℝ) (n : ℕ) : ℝ :=
  initial_efficiency - n * efficiency_decrease

-- Function to calculate the height after n-th bounce
def height_after_n_bounces (n : ℕ) : ℝ :=
  match n with
  | 0     => initial_height
  | n + 1 => bounce_height (height_after_n_bounces n) (bounce_efficiency initial_efficiency n)

-- Lean statement to prove the problem
theorem height_on_fifth_bounce :
  height_after_n_bounces 5 = 0.82003694685696 := by
  sorry

end height_on_fifth_bounce_l116_116065


namespace price_of_each_shirt_l116_116822

theorem price_of_each_shirt 
  (toys_cost : ℕ := 3 * 10)
  (cards_cost : ℕ := 2 * 5)
  (total_spent : ℕ := 70)
  (remaining_cost: ℕ := total_spent - (toys_cost + cards_cost))
  (num_shirts : ℕ := 3 + 2) :
  (remaining_cost / num_shirts) = 6 :=
by
  sorry

end price_of_each_shirt_l116_116822


namespace dasha_flags_proof_l116_116113

variable (Tata_flags_right Yasha_flags_right Vera_flags_right Maxim_flags_right : ℕ)
variable (Total_flags : ℕ)

theorem dasha_flags_proof 
  (hTata: Tata_flags_right = 14)
  (hYasha: Yasha_flags_right = 32)
  (hVera: Vera_flags_right = 20)
  (hMaxim: Maxim_flags_right = 8)
  (hTotal: Total_flags = 37) :
  ∃ (Dasha_flags : ℕ), Dasha_flags = 8 :=
by
  sorry

end dasha_flags_proof_l116_116113


namespace ratio_a7_b7_l116_116489

variable (a b : ℕ → ℕ) -- Define sequences a and b
variable (S T : ℕ → ℕ) -- Define sums S and T

-- Define conditions: arithmetic sequences and given ratio
variable (h_arith_a : ∀ n, a (n + 1) - a n = a 1)
variable (h_arith_b : ∀ n, b (n + 1) - b n = b 1)
variable (h_sum_a : ∀ n, S n = (n + 1) * a 1 + n * a n)
variable (h_sum_b : ∀ n, T n = (n + 1) * b 1 + n * b n)
variable (h_ratio : ∀ n, (S n) / (T n) = (3 * n + 2) / (2 * n))

-- Define the problem statement using the given conditions
theorem ratio_a7_b7 : (a 7) / (b 7) = 41 / 26 :=
by
  sorry

end ratio_a7_b7_l116_116489


namespace peanuts_total_correct_l116_116312

def initial_peanuts : ℕ := 4
def added_peanuts : ℕ := 6
def total_peanuts : ℕ := initial_peanuts + added_peanuts

theorem peanuts_total_correct : total_peanuts = 10 := by
  sorry

end peanuts_total_correct_l116_116312


namespace tourists_walking_speed_l116_116262

-- Define the conditions
def tourists_start_time := 3 + 10 / 60 -- 3:10 A.M.
def bus_pickup_time := 5 -- 5:00 A.M.
def bus_speed := 60 -- 60 km/h
def early_arrival := 20 / 60 -- 20 minutes earlier

-- This is the Lean 4 theorem statement
theorem tourists_walking_speed : 
  (bus_speed * (10 / 60) / (100 / 60)) = 6 := 
by
  sorry

end tourists_walking_speed_l116_116262


namespace range_of_k_l116_116691

noncomputable def quadratic_inequality (k x : ℝ) : ℝ :=
  k * x^2 + 2 * k * x - (k + 2)

theorem range_of_k :
  (∀ x : ℝ, quadratic_inequality k x < 0) ↔ -1 < k ∧ k < 0 :=
by
  sorry

end range_of_k_l116_116691


namespace walter_exceptional_days_l116_116668

variable (b w : Nat)

-- Definitions of the conditions
def total_days (b w : Nat) : Prop := b + w = 10
def total_earnings (b w : Nat) : Prop := 3 * b + 6 * w = 42

-- The theorem states that given the conditions, the number of days Walter did his chores exceptionally well is 4
theorem walter_exceptional_days : total_days b w → total_earnings b w → w = 4 := 
  by
    sorry

end walter_exceptional_days_l116_116668


namespace exists_square_in_interval_l116_116134

def x_k (k : ℕ) : ℕ := k * (k + 1) / 2

noncomputable def sum_x (n : ℕ) : ℕ := (List.range n).map x_k |>.sum

theorem exists_square_in_interval (n : ℕ) (hn : n ≥ 10) :
  ∃ m, (sum_x n - x_k n ≤ m^2 ∧ m^2 ≤ sum_x n) :=
by sorry

end exists_square_in_interval_l116_116134


namespace third_side_length_is_six_l116_116153

theorem third_side_length_is_six
  (a b : ℝ) (c : ℤ)
  (h1 : a = 6.31) 
  (h2 : b = 0.82) 
  (h3 : (a + b > c) ∧ ((b : ℝ) + (c : ℝ) > a) ∧ (c + a > b)) 
  (h4 : 5.49 < (c : ℝ)) 
  (h5 : (c : ℝ) < 7.13) : 
  c = 6 :=
by
  -- Proof goes here
  sorry

end third_side_length_is_six_l116_116153


namespace pizzas_ordered_l116_116547

variable (m : ℕ) (x : ℕ)

theorem pizzas_ordered (h1 : m * 2 * x = 14) (h2 : x = 1 / 2 * m) (h3 : m > 13) : 
  14 + 13 * x = 15 := 
sorry

end pizzas_ordered_l116_116547


namespace factor_polynomial_l116_116401

-- Define the polynomial expression
def polynomial (x : ℝ) : ℝ := 60 * x + 45 + 9 * x ^ 2

-- Define the factored form of the polynomial
def factored_form (x : ℝ) : ℝ := 3 * (3 * x + 5) * (x + 3)

-- The statement of the problem to prove equivalence of the forms
theorem factor_polynomial : ∀ x : ℝ, polynomial x = factored_form x :=
by
  -- The actual proof is omitted and replaced by sorry
  sorry

end factor_polynomial_l116_116401


namespace age_ratio_l116_116248

theorem age_ratio (x : ℕ) (h : (5 * x - 4) = (3 * x + 4)) :
    (5 * x + 4) / (3 * x - 4) = 3 :=
by sorry

end age_ratio_l116_116248


namespace value_of_a_minus_b_l116_116403

theorem value_of_a_minus_b (a b : ℝ) (h1 : |a| = 3) (h2 : |b| = 13) (h3 : a * b > 0) : a - b = -10 ∨ a - b = 10 :=
sorry

end value_of_a_minus_b_l116_116403


namespace inverse_function_correct_l116_116629

theorem inverse_function_correct :
  ( ∀ x : ℝ, (x > 1) → (∃ y : ℝ, y = 1 + Real.log (x - 1)) ↔ (∀ y : ℝ, y > 0 → (∃ x : ℝ, x = e^(y + 1) - 1))) :=
by
  sorry

end inverse_function_correct_l116_116629


namespace find_m_l116_116474

def U : Set ℤ := {-1, 2, 3, 6}
def A (m : ℤ) : Set ℤ := {x | x^2 - 5 * x + m = 0}
def complement_U_A (m : ℤ) : Set ℤ := U \ A m

theorem find_m (m : ℤ) (hU : U = {-1, 2, 3, 6}) (hcomp : complement_U_A m = {2, 3}) :
  m = -6 := by
  sorry

end find_m_l116_116474


namespace cashier_adjustment_l116_116436

-- Define the conditions
variables {y : ℝ}

-- Error calculation given the conditions
def half_dollar_error (y : ℝ) : ℝ := 0.50 * y
def five_dollar_error (y : ℝ) : ℝ := 5 * y
def total_error (y : ℝ) : ℝ := half_dollar_error y + five_dollar_error y

-- Theorem statement
theorem cashier_adjustment (y : ℝ) : total_error y = 5.50 * y :=
sorry

end cashier_adjustment_l116_116436


namespace total_amount_shared_l116_116664

-- Given John (J), Jose (Jo), and Binoy (B) and their proportion of money
variables (J Jo B : ℕ)
-- John received 1440 Rs.
variable (John_received : J = 1440)

-- The ratio of their shares is 2:4:6
axiom ratio_condition : J * 2 = Jo * 4 ∧ J * 2 = B * 6

-- The target statement to prove
theorem total_amount_shared : J + Jo + B = 8640 :=
by {
  sorry
}

end total_amount_shared_l116_116664


namespace goods_train_length_l116_116288

theorem goods_train_length (speed_kmph : ℕ) (platform_length_m : ℕ) (time_s : ℕ) 
    (h_speed : speed_kmph = 72) (h_platform : platform_length_m = 250) (h_time : time_s = 24) : 
    ∃ train_length_m : ℕ, train_length_m = 230 := 
by 
  sorry

end goods_train_length_l116_116288


namespace count_three_digit_numbers_divisible_by_seventeen_l116_116087

def is_three_digit_number (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def is_divisible_by_seventeen (n : ℕ) : Prop := n % 17 = 0

theorem count_three_digit_numbers_divisible_by_seventeen : 
  ∃ (count : ℕ), count = 53 ∧ 
    (∀ (n : ℕ), is_three_digit_number n → is_divisible_by_seventeen n → response) := 
sorry

end count_three_digit_numbers_divisible_by_seventeen_l116_116087


namespace find_positive_integer_N_l116_116422

theorem find_positive_integer_N (N : ℕ) (h₁ : 33^2 * 55^2 = 15^2 * N^2) : N = 121 :=
by {
  sorry
}

end find_positive_integer_N_l116_116422


namespace max_value_of_function_l116_116012

theorem max_value_of_function (x : ℝ) (h : x < 1 / 2) : 
  ∃ y, y = 2 * x + 1 / (2 * x - 1) ∧ y ≤ -1 :=
by
  sorry

end max_value_of_function_l116_116012


namespace tip_percentage_l116_116753

/--
A family paid $30 for food, the sales tax rate is 9.5%, and the total amount paid was $35.75. Prove that the tip percentage is 9.67%.
-/
theorem tip_percentage (food_cost : ℝ) (sales_tax_rate : ℝ) (total_paid : ℝ)
  (h1 : food_cost = 30)
  (h2 : sales_tax_rate = 0.095)
  (h3 : total_paid = 35.75) :
  ((total_paid - (food_cost * (1 + sales_tax_rate))) / food_cost) * 100 = 9.67 :=
by
  sorry

end tip_percentage_l116_116753


namespace cars_between_15000_and_20000_l116_116591

theorem cars_between_15000_and_20000 
  (total_cars : ℕ)
  (less_than_15000_ratio : ℝ)
  (more_than_20000_ratio : ℝ)
  : less_than_15000_ratio = 0.15 → 
    more_than_20000_ratio = 0.40 → 
    total_cars = 3000 → 
    ∃ (cars_between : ℕ),
      cars_between = total_cars - (less_than_15000_ratio * total_cars + more_than_20000_ratio * total_cars) ∧ 
      cars_between = 1350 :=
by
  sorry

end cars_between_15000_and_20000_l116_116591


namespace min_f_value_l116_116475

noncomputable def f (x y z : ℝ) : ℝ := x^2 + 4 * x * y + 9 * y^2 + 8 * y * z + 3 * z^2

theorem min_f_value (x y z : ℝ) (hxyz_pos : 0 < x ∧ 0 < y ∧ 0 < z) (hxyz : x * y * z = 1) :
  f x y z ≥ 18 :=
sorry

end min_f_value_l116_116475


namespace same_yield_among_squares_l116_116457

-- Define the conditions
def rectangular_schoolyard (length : ℝ) (width : ℝ) := length = 70 ∧ width = 35

def total_harvest (harvest : ℝ) := harvest = 1470 -- in kilograms (14.7 quintals)

def smaller_square (side : ℝ) := side = 0.7

-- Define the proof problem
theorem same_yield_among_squares :
  ∃ side : ℝ, smaller_square side ∧
  ∃ length width harvest : ℝ, rectangular_schoolyard length width ∧ total_harvest harvest →
  ∃ (yield1 yield2 : ℝ), yield1 = yield2 ∧ yield1 ≠ 0 ∧ yield2 ≠ 0 :=
by sorry

end same_yield_among_squares_l116_116457


namespace polynomial_remainder_l116_116340

theorem polynomial_remainder (P : Polynomial ℝ) (H1 : P.eval 1 = 2) (H2 : P.eval 2 = 1) :
  ∃ Q : Polynomial ℝ, P = Q * (Polynomial.X - 1) * (Polynomial.X - 2) + (3 - Polynomial.X) :=
by
  sorry

end polynomial_remainder_l116_116340


namespace sum_of_rel_prime_greater_than_one_l116_116011

theorem sum_of_rel_prime_greater_than_one (a : ℕ) (h : a > 6) : 
  ∃ b c : ℕ, a = b + c ∧ b > 1 ∧ c > 1 ∧ Nat.gcd b c = 1 :=
sorry

end sum_of_rel_prime_greater_than_one_l116_116011


namespace percentage_runs_by_running_l116_116819

theorem percentage_runs_by_running
  (total_runs : ℕ)
  (boundaries : ℕ)
  (sixes : ℕ)
  (runs_per_boundary : ℕ)
  (runs_per_six : ℕ)
  (eq_total_runs : total_runs = 120)
  (eq_boundaries : boundaries = 3)
  (eq_sixes : sixes = 8)
  (eq_runs_per_boundary : runs_per_boundary = 4)
  (eq_runs_per_six : runs_per_six = 6) :
  ((total_runs - (boundaries * runs_per_boundary + sixes * runs_per_six)) / total_runs * 100) = 50 :=
by
  sorry

end percentage_runs_by_running_l116_116819


namespace negation_equivalence_l116_116961

-- Define the propositions
def proposition (a b : ℝ) : Prop := a > b → a + 1 > b

def negation_proposition (a b : ℝ) : Prop := a ≤ b → a + 1 ≤ b

-- Statement to prove
theorem negation_equivalence (a b : ℝ) : ¬(proposition a b) ↔ negation_proposition a b := 
sorry

end negation_equivalence_l116_116961


namespace team_arrangement_count_l116_116520

-- Definitions of the problem
def veteran_players := 2
def new_players := 3
def total_players := veteran_players + new_players
def team_size := 3

-- Conditions
def condition_veteran : Prop := 
  ∀ (team : Finset ℕ), team.card = team_size → Finset.card (team ∩ (Finset.range veteran_players)) ≥ 1

def condition_new_player : Prop := 
  ∀ (team : Finset ℕ), team.card = team_size → 
    ∃ (p1 p2 : ℕ), p1 ∈ team ∧ p2 ∈ team ∧ 
    p1 ≠ p2 ∧ p1 < team_size ∧ p2 < team_size ∧
    (p1 ∈ (Finset.Ico veteran_players total_players) ∨ p2 ∈ (Finset.Ico veteran_players total_players))

-- Goal
def number_of_arrangements := 48

-- The statement to prove
theorem team_arrangement_count : condition_veteran → condition_new_player → 
  (∃ (arrangements : ℕ), arrangements = number_of_arrangements) :=
by
  sorry

end team_arrangement_count_l116_116520


namespace both_firms_participate_l116_116393

-- Definitions based on the conditions
variable (V IC : ℝ) (α : ℝ)
-- Assumptions
variable (hα : 0 < α ∧ α < 1)
-- Part (a) condition transformation
def participation_condition := α * (1 - α) * V + 0.5 * α^2 * V ≥ IC

-- Given values for part (b)
def V_value : ℝ := 24
def α_value : ℝ := 0.5
def IC_value : ℝ := 7

-- New definitions for given values
def part_b_condition := (α_value * (1 - α_value) * V_value + 0.5 * α_value^2 * V_value) ≥ IC_value

-- Profits for part (c) comparison
def profit_when_both := 2 * (α * (1 - α) * V + 0.5 * α^2 * V - IC)
def profit_when_one := α * V - IC

-- Proof problem statement in Lean 4
theorem both_firms_participate (hV : V = 24) (hα : α = 0.5) (hIC : IC = 7) :
    (α * (1 - α) * V + 0.5 * α^2 * V) ≥ IC ∧ profit_when_both V alpha IC > profit_when_one V α IC := by
  sorry

end both_firms_participate_l116_116393


namespace simplify_fraction_l116_116736

theorem simplify_fraction : 
  (1 / (1 / (Real.sqrt 2 + 1) + 2 / (Real.sqrt 3 - 1))) = Real.sqrt 3 - Real.sqrt 2 :=
by
  sorry

end simplify_fraction_l116_116736


namespace boys_passed_l116_116202

theorem boys_passed (total_boys : ℕ) (avg_marks : ℕ) (avg_passed : ℕ) (avg_failed : ℕ) (P : ℕ) 
    (h1 : total_boys = 120) (h2 : avg_marks = 36) (h3 : avg_passed = 39) (h4 : avg_failed = 15)
    (h5 : P + (total_boys - P) = 120) 
    (h6 : P * avg_passed + (total_boys - P) * avg_failed = total_boys * avg_marks) :
    P = 105 := 
sorry

end boys_passed_l116_116202


namespace chocolate_bars_left_l116_116806

noncomputable def chocolateBarsCount : ℕ :=
  let initial_bars := 800
  let thomas_friends_bars := (3 * initial_bars) / 8
  let adjusted_thomas_friends_bars := thomas_friends_bars + 1  -- Adjust for the extra bar rounding issue
  let piper_bars_taken := initial_bars / 4
  let piper_bars_returned := 8
  let adjusted_piper_bars := piper_bars_taken - piper_bars_returned
  let paul_club_bars := 9
  let polly_club_bars := 7
  let catherine_bars_returned := 15
  
  initial_bars
  - adjusted_thomas_friends_bars
  - adjusted_piper_bars
  - paul_club_bars
  - polly_club_bars
  + catherine_bars_returned

theorem chocolate_bars_left : chocolateBarsCount = 308 := by
  sorry

end chocolate_bars_left_l116_116806


namespace lion_king_cost_l116_116023

theorem lion_king_cost
  (LK_earned : ℕ := 200) -- The Lion King earned 200 million
  (LK_profit : ℕ := 190) -- The Lion King profit calculated from half of Star Wars' profit
  (SW_cost : ℕ := 25)    -- Star Wars cost 25 million
  (SW_earned : ℕ := 405) -- Star Wars earned 405 million
  (SW_profit : SW_earned - SW_cost = 380) -- Star Wars profit
  (LK_profit_from_SW : LK_profit = 1/2 * (SW_earned - SW_cost)) -- The Lion King profit calculation
  (LK_cost : ℕ := LK_earned - LK_profit) -- The Lion King cost calculation
  : LK_cost = 10 := 
sorry

end lion_king_cost_l116_116023


namespace find_radius_of_smaller_circles_l116_116441

noncomputable def smaller_circle_radius (r : ℝ) : Prop :=
  ∃ sin72 : ℝ, sin72 = Real.sin (72 * Real.pi / 180) ∧
  r = (2 * sin72) / (1 - sin72)

theorem find_radius_of_smaller_circles (r : ℝ) :
  (smaller_circle_radius r) ↔
  r = (2 * Real.sin (72 * Real.pi / 180)) / (1 - Real.sin (72 * Real.pi / 180)) :=
by
  sorry

end find_radius_of_smaller_circles_l116_116441


namespace max_difference_in_masses_of_two_flour_bags_l116_116040

theorem max_difference_in_masses_of_two_flour_bags :
  ∀ (x y : ℝ), (24.8 ≤ x ∧ x ≤ 25.2) → (24.8 ≤ y ∧ y ≤ 25.2) → |x - y| ≤ 0.4 :=
by
  sorry

end max_difference_in_masses_of_two_flour_bags_l116_116040


namespace total_pokemon_cards_l116_116892

def pokemon_cards (sam dan tom keith : Nat) : Nat :=
  sam + dan + tom + keith

theorem total_pokemon_cards :
  pokemon_cards 14 14 14 14 = 56 := by
  sorry

end total_pokemon_cards_l116_116892


namespace min_value_of_square_sum_l116_116723

theorem min_value_of_square_sum (x y : ℝ) (h : (x-1)^2 + y^2 = 16) : ∃ (a : ℝ), a = x^2 + y^2 ∧ a = 9 :=
by 
  sorry

end min_value_of_square_sum_l116_116723


namespace area_of_shaded_region_l116_116609

-- Definitions of conditions
def center (O : Type) := O
def radius_large_circle (R : ℝ) := R
def radius_small_circle (r : ℝ) := r
def length_chord_CD (CD : ℝ) := CD = 60
def chord_tangent_to_smaller_circle (r : ℝ) (R : ℝ) := r^2 = R^2 - 900

-- Theorem for the area of the shaded region
theorem area_of_shaded_region 
(O : Type) 
(R r : ℝ) 
(CD : ℝ)
(h1 : length_chord_CD CD)
(h2 : chord_tangent_to_smaller_circle r R) : 
  π * (R^2 - r^2) = 900 * π := by
  sorry

end area_of_shaded_region_l116_116609


namespace number_of_permissible_sandwiches_l116_116997

theorem number_of_permissible_sandwiches (b m c : ℕ) (h : b = 5) (me : m = 7) (ch : c = 6) 
  (no_ham_cheddar : ∀ bread, ¬(bread = ham ∧ cheese = cheddar))
  (no_turkey_swiss : ∀ bread, ¬(bread = turkey ∧ cheese = swiss)) : 
  5 * 7 * 6 - (5 * 1 * 1) - (5 * 1 * 1) = 200 := 
by 
  sorry

end number_of_permissible_sandwiches_l116_116997


namespace ratio_area_rectangle_triangle_l116_116396

noncomputable def area_rectangle (L W : ℝ) : ℝ :=
  L * W

noncomputable def area_triangle (L W : ℝ) : ℝ :=
  (1 / 2) * L * W

theorem ratio_area_rectangle_triangle (L W : ℝ) (hL : L > 0) (hW : W > 0) :
  area_rectangle L W / area_triangle L W = 2 :=
by
  -- sorry will be replaced by the actual proof
  sorry

end ratio_area_rectangle_triangle_l116_116396


namespace rita_months_needed_l116_116679

def total_hours_required : ℕ := 2500
def backstroke_hours : ℕ := 75
def breaststroke_hours : ℕ := 25
def butterfly_hours : ℕ := 200
def hours_per_month : ℕ := 300

def total_completed_hours : ℕ := backstroke_hours + breaststroke_hours + butterfly_hours
def remaining_hours : ℕ := total_hours_required - total_completed_hours
def months_needed (remaining_hours hours_per_month : ℕ) : ℕ := (remaining_hours + hours_per_month - 1) / hours_per_month

theorem rita_months_needed : months_needed remaining_hours hours_per_month = 8 := by
  -- Lean 4 proof goes here
  sorry

end rita_months_needed_l116_116679


namespace find_n_from_degree_l116_116947

theorem find_n_from_degree (n : ℕ) (h : 2 + n = 5) : n = 3 :=
by {
  sorry
}

end find_n_from_degree_l116_116947


namespace proportion_of_boys_geq_35_percent_l116_116837

variables (a b c d n : ℕ)

axiom room_constraint : 2 * (b + d) ≥ n
axiom girl_constraint : 3 * a ≥ 8 * b

theorem proportion_of_boys_geq_35_percent : (3 * c + 4 * d : ℚ) / (3 * a + 4 * b + 3 * c + 4 * d : ℚ) ≥ 0.35 :=
by 
  sorry

end proportion_of_boys_geq_35_percent_l116_116837


namespace ratio_of_second_to_first_show_l116_116791

-- Definitions based on conditions
def first_show_length : ℕ := 30
def total_show_time : ℕ := 150
def second_show_length := total_show_time - first_show_length

-- Proof problem in Lean 4 statement
theorem ratio_of_second_to_first_show : 
  (second_show_length / first_show_length) = 4 := by
  sorry

end ratio_of_second_to_first_show_l116_116791


namespace remainder_of_12111_div_3_l116_116718

theorem remainder_of_12111_div_3 : 12111 % 3 = 0 := by
  sorry

end remainder_of_12111_div_3_l116_116718


namespace cards_distribution_l116_116257

theorem cards_distribution (total_cards people : ℕ) (h1 : total_cards = 48) (h2 : people = 7) :
  (people - (total_cards % people)) = 1 :=
by
  sorry

end cards_distribution_l116_116257


namespace distinct_complex_numbers_count_l116_116283

theorem distinct_complex_numbers_count :
  let real_choices := 10
  let imag_choices := 9
  let distinct_complex_numbers := real_choices * imag_choices
  distinct_complex_numbers = 90 :=
by
  sorry

end distinct_complex_numbers_count_l116_116283


namespace max_a4_l116_116877

variable {a_n : ℕ → ℝ}

-- Assume a_n is a positive geometric sequence
def is_geometric_seq (a_n : ℕ → ℝ) : Prop :=
  ∃ r > 0, ∀ n, a_n (n + 1) = a_n n * r

-- Given conditions
def condition1 (a_n : ℕ → ℝ) : Prop := is_geometric_seq a_n
def condition2 (a_n : ℕ → ℝ) : Prop := a_n 3 + a_n 5 = 4

theorem max_a4 (a_n : ℕ → ℝ) (h1 : condition1 a_n) (h2 : condition2 a_n) :
    ∃ max_a4 : ℝ, max_a4 = 2 :=
  sorry

end max_a4_l116_116877


namespace not_enough_pharmacies_l116_116405

theorem not_enough_pharmacies : 
  ∀ (n m : ℕ), n = 10 ∧ m = 10 →
  ∃ (intersections : ℕ), intersections = n * m ∧ 
  ∀ (d : ℕ), d = 3 →
  ∀ (coverage : ℕ), coverage = (2 * d + 1) * (2 * d + 1) →
  ¬ (coverage * 12 ≥ intersections * 2) :=
by sorry

end not_enough_pharmacies_l116_116405


namespace greatest_int_less_neg_22_3_l116_116835

theorem greatest_int_less_neg_22_3 : ∃ n : ℤ, n = -8 ∧ n < -22 / 3 ∧ ∀ m : ℤ, m < -22 / 3 → m ≤ n :=
by
  sorry

end greatest_int_less_neg_22_3_l116_116835


namespace number_of_samples_from_retired_l116_116164

def ratio_of_forms (retired current students : ℕ) : Prop :=
retired = 3 ∧ current = 7 ∧ students = 40

def total_sampled_forms := 300

theorem number_of_samples_from_retired :
  ∃ (xr : ℕ), ratio_of_forms 3 7 40 → xr = (300 / (3 + 7 + 40)) * 3 :=
sorry

end number_of_samples_from_retired_l116_116164


namespace quadratic_no_real_roots_min_k_l116_116273

theorem quadratic_no_real_roots_min_k :
  ∀ (k : ℤ), 
    (∀ x : ℝ, 3*x*(k*x-5) - 2*x^2 + 8 ≠ 0) ↔ 
    (k ≥ 3) := 
by 
  sorry

end quadratic_no_real_roots_min_k_l116_116273


namespace cat_head_start_15_minutes_l116_116809

theorem cat_head_start_15_minutes :
  ∀ (t : ℕ), (25 : ℝ) = (20 : ℝ) * (1 + (t : ℝ) / 60) → t = 15 := by
  sorry

end cat_head_start_15_minutes_l116_116809


namespace minimum_value_x2_y2_l116_116544

variable {x y : ℝ}

theorem minimum_value_x2_y2 (hx_pos : 0 < x) (hy_pos : 0 < y) (hxy : x * y = 1) : x^2 + y^2 = 2 :=
sorry

end minimum_value_x2_y2_l116_116544


namespace four_digit_number_divisible_by_9_l116_116762

theorem four_digit_number_divisible_by_9
    (a b c d e f g h i j : ℕ)
    (h₀ : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧ a ≠ h ∧ a ≠ i ∧ a ≠ j ∧
               b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧ b ≠ h ∧ b ≠ i ∧ b ≠ j ∧
               c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧ c ≠ h ∧ c ≠ i ∧ c ≠ j ∧
               d ≠ e ∧ d ≠ f ∧ d ≠ g ∧ d ≠ h ∧ d ≠ i ∧ d ≠ j ∧
               e ≠ f ∧ e ≠ g ∧ e ≠ h ∧ e ≠ i ∧ e ≠ j ∧
               f ≠ g ∧ f ≠ h ∧ f ≠ i ∧ f ≠ j ∧
               g ≠ h ∧ g ≠ i ∧ g ≠ j ∧
               h ≠ i ∧ h ≠ j ∧
               i ≠ j )
    (h₁ : a + b + c + d + e + f + g + h + i + j = 45)
    (h₂ : 100 * a + 10 * b + c + 100 * d + 10 * e + f = 1000 * g + 100 * h + 10 * i + j) :
  ((1000 * g + 100 * h + 10 * i + j) % 9 = 0) := sorry

end four_digit_number_divisible_by_9_l116_116762


namespace fraction_spent_at_arcade_l116_116531

theorem fraction_spent_at_arcade :
  ∃ f : ℝ, 
    (2.25 - (2.25 * f) - ((2.25 - (2.25 * f)) / 3) = 0.60) → 
    f = 3 / 5 :=
by
  sorry

end fraction_spent_at_arcade_l116_116531


namespace point_value_of_other_questions_l116_116368

theorem point_value_of_other_questions (x y p : ℕ) 
  (h1 : x = 10) 
  (h2 : x + y = 40) 
  (h3 : 40 + 30 * p = 100) : 
  p = 2 := 
  sorry

end point_value_of_other_questions_l116_116368


namespace trays_needed_l116_116958

theorem trays_needed (cookies_classmates cookies_teachers cookies_per_tray : ℕ) 
  (hc1 : cookies_classmates = 276) 
  (hc2 : cookies_teachers = 92) 
  (hc3 : cookies_per_tray = 12) : 
  (cookies_classmates + cookies_teachers + cookies_per_tray - 1) / cookies_per_tray = 31 :=
by
  sorry

end trays_needed_l116_116958


namespace riddles_ratio_l116_116559

theorem riddles_ratio (Josh_riddles : ℕ) (Ivory_riddles : ℕ) (Taso_riddles : ℕ) 
  (h1 : Josh_riddles = 8) 
  (h2 : Ivory_riddles = Josh_riddles + 4) 
  (h3 : Taso_riddles = 24) : 
  Taso_riddles / Ivory_riddles = 2 := 
by sorry

end riddles_ratio_l116_116559


namespace solution_l116_116070

def money_problem (x y : ℝ) : Prop :=
  (x + y / 2 = 50) ∧ (y + 2 * x / 3 = 50)

theorem solution :
  ∃ x y : ℝ, money_problem x y ∧ x = 37.5 ∧ y = 25 :=
by
  use 37.5, 25
  sorry

end solution_l116_116070


namespace solve_quadratic_equation_l116_116846

theorem solve_quadratic_equation (x : ℝ) : 
  2 * x^2 - 4 * x = 6 - 3 * x ↔ (x = -3/2 ∨ x = 2) :=
by
  sorry

end solve_quadratic_equation_l116_116846


namespace exists_xyz_prime_expression_l116_116517

theorem exists_xyz_prime_expression (a b c p : ℤ) (h_prime : Prime p)
    (h_div : p ∣ (a^2 + b^2 + c^2 - ab - bc - ca))
    (h_gcd : Int.gcd p ((a^2 + b^2 + c^2 - ab - bc - ca) / p) = 1) :
    ∃ x y z : ℤ, p = x^2 + y^2 + z^2 - xy - yz - zx := by
  sorry

end exists_xyz_prime_expression_l116_116517


namespace percentage_subtracted_l116_116712

theorem percentage_subtracted (a : ℝ) (p : ℝ) (h : (1 - p / 100) * a = 0.97 * a) : p = 3 :=
by
  sorry

end percentage_subtracted_l116_116712


namespace find_x_l116_116742

-- Define the custom operation on m and n
def operation (m n : ℤ) : ℤ := 2 * m - 3 * n

-- Lean statement of the problem
theorem find_x (x : ℤ) (h : operation x 7 = operation 7 x) : x = 7 :=
by {
  -- Placeholder for the proof
  sorry
}

end find_x_l116_116742


namespace compare_solutions_l116_116409

variables (p q r s : ℝ)
variables (hp : p ≠ 0) (hr : r ≠ 0)

theorem compare_solutions :
  ((-q / p) > (-s / r)) ↔ (s * r > q * p) :=
by sorry

end compare_solutions_l116_116409


namespace solve_for_x_l116_116864

theorem solve_for_x (x : ℝ) (h : x^4 = (-3)^4) : x = 3 ∨ x = -3 :=
sorry

end solve_for_x_l116_116864


namespace product_of_two_numbers_l116_116900

theorem product_of_two_numbers (x y : ℝ) (h1 : x - y = 12) (h2 : x^2 + y^2 = 340) : x * y = 97.9450625 :=
by
  sorry

end product_of_two_numbers_l116_116900


namespace prove_divisibility_l116_116185

-- Given the conditions:
variables (a b r s : ℕ)
variables (pos_a : a > 0) (pos_b : b > 0) (pos_r : r > 0) (pos_s : s > 0)
variables (a_le_two : a ≤ 2)
variables (no_common_prime_factor : (gcd a b) = 1)
variables (divisibility_condition : (a ^ s + b ^ s) ∣ (a ^ r + b ^ r))

-- We aim to prove that:
theorem prove_divisibility : s ∣ r := 
sorry

end prove_divisibility_l116_116185


namespace difference_of_numbers_l116_116610

theorem difference_of_numbers (a b : ℕ) (h1 : a + b = 20460) (h2 : b % 12 = 0) (h3 : b / 10 = a) : b - a = 17314 :=
by
  sorry

end difference_of_numbers_l116_116610


namespace smallest_three_digit_multiple_of_three_with_odd_hundreds_l116_116186

theorem smallest_three_digit_multiple_of_three_with_odd_hundreds :
  ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ (∃ a b c : ℕ, n = 100 * a + 10 * b + c ∧ a % 2 = 1 ∧ n % 3 = 0 ∧ n = 102) :=
by
  sorry

end smallest_three_digit_multiple_of_three_with_odd_hundreds_l116_116186


namespace green_beads_in_each_necklace_l116_116572

theorem green_beads_in_each_necklace (G : ℕ) :
  (∀ n, (n = 5) → (6 * n ≤ 45) ∧ (3 * n ≤ 45) ∧ (G * n = 45)) → G = 9 :=
by
  intros h
  have hn : 5 = 5 := rfl
  cases h 5 hn
  sorry

end green_beads_in_each_necklace_l116_116572


namespace sum_of_squares_of_sines_l116_116624

theorem sum_of_squares_of_sines (α : ℝ) : 
  (Real.sin α)^2 + (Real.sin (α + 60 * Real.pi / 180))^2 + (Real.sin (α + 120 * Real.pi / 180))^2 = 3 / 2 := 
by
  sorry

end sum_of_squares_of_sines_l116_116624


namespace perimeter_of_square_l116_116738

variable (s : ℝ) (side_length : ℝ)
def is_square_side_length_5 (s : ℝ) : Prop := s = 5
theorem perimeter_of_square (h: is_square_side_length_5 s) : 4 * s = 20 := sorry

end perimeter_of_square_l116_116738


namespace geometric_sequence_common_ratio_l116_116338

theorem geometric_sequence_common_ratio 
  (a : ℕ → ℝ) 
  (q : ℝ)
  (h_geom : ∀ n, a (n+1) = a n * q)
  (h_a1 : a 1 = 1/2) 
  (h_a4 : a 4 = -4) : 
  q = -2 := 
by
  sorry

end geometric_sequence_common_ratio_l116_116338


namespace quad_common_root_l116_116455

theorem quad_common_root (a b c d : ℝ) :
  (∃ α : ℝ, α^2 + a * α + b = 0 ∧ α^2 + c * α + d = 0) ↔ (a * d - b * c) * (c - a) = (b - d)^2 ∧ (a ≠ c) := 
sorry

end quad_common_root_l116_116455


namespace lines_perpendicular_l116_116015

-- Definition of lines and their relationships
def Line : Type := ℝ × ℝ × ℝ → Prop

variables (a b c : Line)

-- Condition 1: a is perpendicular to b
axiom perp (a b : Line) : Prop
-- Condition 2: b is parallel to c
axiom parallel (b c : Line) : Prop

-- Theorem to prove: 
theorem lines_perpendicular (h1 : perp a b) (h2 : parallel b c) : perp a c :=
sorry

end lines_perpendicular_l116_116015


namespace union_P_complement_Q_l116_116236

-- Define sets P and Q
def P : Set ℝ := {x | 1 ≤ x ∧ x ≤ 3}
def Q : Set ℝ := {x | x^2 ≥ 4}

-- Define the complement of Q in ℝ
def C_RQ : Set ℝ := {x | -2 < x ∧ x < 2}

-- State the main theorem
theorem union_P_complement_Q : (P ∪ C_RQ) = {x | -2 < x ∧ x ≤ 3} := 
by
  sorry

end union_P_complement_Q_l116_116236


namespace shortest_distance_parabola_line_l116_116452

theorem shortest_distance_parabola_line :
  ∃ (P Q : ℝ × ℝ), P.2 = P.1^2 - 6 * P.1 + 15 ∧ Q.2 = 2 * Q.1 - 7 ∧
  ∀ (p q : ℝ × ℝ), p.2 = p.1^2 - 6 * p.1 + 15 → q.2 = 2 * q.1 - 7 → 
  dist p q ≥ dist P Q :=
sorry

end shortest_distance_parabola_line_l116_116452


namespace Dave_ticket_count_l116_116500

variable (T C total : ℕ)

theorem Dave_ticket_count
  (hT1 : T = 12)
  (hC1 : C = 7)
  (hT2 : T = C + 5) :
  total = T + C → total = 19 := by
  sorry

end Dave_ticket_count_l116_116500


namespace tennis_ball_price_l116_116669

theorem tennis_ball_price (x y : ℝ) 
  (h₁ : 2 * x + 7 * y = 220)
  (h₂ : x = y + 83) : 
  y = 6 := 
by 
  sorry

end tennis_ball_price_l116_116669


namespace inequality_condition_necessary_not_sufficient_l116_116378

theorem inequality_condition (a b : ℝ) (h1 : 0 < a) (h2 : a < b) : 
  (1 / a > 1 / b) :=
by
  sorry

theorem necessary_not_sufficient (a b : ℝ) :
  (1 / a > 1 / b → 0 < a ∧ a < b) ∧ ¬ (0 < a ∧ a < b → 1 / a > 1 / b) :=
by
  sorry

end inequality_condition_necessary_not_sufficient_l116_116378


namespace correct_answers_count_l116_116345

theorem correct_answers_count (total_questions correct_pts incorrect_pts final_score : ℤ)
  (h1 : total_questions = 26)
  (h2 : correct_pts = 8)
  (h3 : incorrect_pts = -5)
  (h4 : final_score = 0) :
  ∃ c i : ℤ, c + i = total_questions ∧ correct_pts * c + incorrect_pts * i = final_score ∧ c = 10 :=
by
  use 10, (26 - 10)
  simp
  sorry

end correct_answers_count_l116_116345


namespace frustum_volume_and_lateral_surface_area_l116_116725

theorem frustum_volume_and_lateral_surface_area (h : ℝ) 
    (A1 A2 : ℝ) (r R : ℝ) (V S_lateral : ℝ) : 
    A1 = 4 * Real.pi → 
    A2 = 25 * Real.pi → 
    h = 4 → 
    r = 2 → 
    R = 5 → 
    V = (1 / 3) * (A1 + A2 + Real.sqrt (A1 * A2)) * h → 
    S_lateral = Real.pi * r * Real.sqrt (h ^ 2 + (R - r) ^ 2) + Real.pi * R * Real.sqrt (h ^ 2 + (R - r) ^ 2) → 
    V = 42 * Real.pi ∧ S_lateral = 35 * Real.pi := by
  sorry

end frustum_volume_and_lateral_surface_area_l116_116725


namespace contrapositive_statement_l116_116071

def is_even (n : ℕ) : Prop := n % 2 = 0
def is_odd (n : ℕ) : Prop := ¬is_even n

theorem contrapositive_statement (a b : ℕ) :
  (¬(is_odd a ∧ is_odd b) ∧ ¬(is_even a ∧ is_even b)) → ¬is_even (a + b) :=
by
  sorry

end contrapositive_statement_l116_116071


namespace tangent_line_eq_l116_116635

theorem tangent_line_eq
  (x y : ℝ)
  (h : x^2 + y^2 - 4 * x = 0)
  (P : ℝ × ℝ)
  (hP : P = (1, Real.sqrt 3))
  : x - Real.sqrt 3 * y + 2 = 0 :=
sorry

end tangent_line_eq_l116_116635


namespace jose_marks_difference_l116_116923

theorem jose_marks_difference (M J A : ℕ) 
  (h1 : M = J - 20)
  (h2 : J + M + A = 210)
  (h3 : J = 90) : (J - A) = 40 :=
by
  sorry

end jose_marks_difference_l116_116923


namespace difference_longest_shortest_worm_l116_116975

theorem difference_longest_shortest_worm
  (A B C D E : ℝ)
  (hA : A = 0.8)
  (hB : B = 0.1)
  (hC : C = 1.2)
  (hD : D = 0.4)
  (hE : E = 0.7) :
  (max C (max A (max E (max D B))) - min B (min D (min E (min A C)))) = 1.1 :=
by
  sorry

end difference_longest_shortest_worm_l116_116975


namespace min_value_expr_sum_of_squares_inequality_l116_116465

-- Given conditions
variables (a b : ℝ)
variables (ha : a > 0) (hb : b > 0) (hab : a + b = 2)

-- Problem (1): Prove minimum value of (2 / a + 8 / b) is 9
theorem min_value_expr : ∃ a b, a > 0 ∧ b > 0 ∧ a + b = 2 ∧ ((2 / a) + (8 / b) = 9) := sorry

-- Problem (2): Prove a^2 + b^2 ≥ 2
theorem sum_of_squares_inequality : a^2 + b^2 ≥ 2 :=
by { sorry }

end min_value_expr_sum_of_squares_inequality_l116_116465


namespace number_of_nurses_l116_116503

theorem number_of_nurses (total : ℕ) (ratio_d_to_n : ℕ → ℕ) (h1 : total = 250) (h2 : ratio_d_to_n 2 = 3) : ∃ n : ℕ, n = 150 := 
by
  sorry

end number_of_nurses_l116_116503


namespace find_x2_y2_l116_116744

theorem find_x2_y2 (x y : ℕ) (h1 : x > 0) (h2 : y > 0) (h3 : xy + x + y = 35) (h4 : xy * (x + y) = 360) : x^2 + y^2 = 185 := by
  sorry

end find_x2_y2_l116_116744


namespace sequence_a_n_sum_T_n_l116_116076

variable (a : ℕ → ℕ)
variable (S : ℕ → ℕ)
variable (b : ℕ → ℕ)
variable (T : ℕ → ℕ)

theorem sequence_a_n (n : ℕ) (hS : ∀ n, S n = 2 * a n - n) :
  a n = 2 ^ n - 1 :=
sorry

theorem sum_T_n (n : ℕ) (hb : ∀ n, b n = (2 * n + 1) * (a n + 1)) 
  (ha : ∀ n, a n = 2 ^ n - 1) :
  T n = 2 + (2 * n - 1) * 2 ^ (n + 1) :=
sorry

end sequence_a_n_sum_T_n_l116_116076


namespace rectangular_field_length_l116_116926

theorem rectangular_field_length {w l : ℝ} (h1 : l = 2 * w) (h2 : (8 : ℝ) * 8 = 1 / 18 * (l * w)) : l = 48 :=
by sorry

end rectangular_field_length_l116_116926


namespace sin_add_pi_over_2_l116_116334

theorem sin_add_pi_over_2 (θ : ℝ) (h : Real.cos θ = -3 / 5) : Real.sin (θ + π / 2) = -3 / 5 :=
sorry

end sin_add_pi_over_2_l116_116334


namespace required_lemons_for_20_gallons_l116_116640

-- Conditions
def lemons_for_50_gallons : ℕ := 40
def gallons_for_lemons : ℕ := 50
def additional_lemons_per_10_gallons : ℕ := 1
def number_of_gallons : ℕ := 20
def base_lemons (g: ℕ) : ℕ := (lemons_for_50_gallons * g) / gallons_for_lemons
def additional_lemons (g: ℕ) : ℕ := (g / 10) * additional_lemons_per_10_gallons
def total_lemons (g: ℕ) : ℕ := base_lemons g + additional_lemons g

-- Proof statement
theorem required_lemons_for_20_gallons : total_lemons number_of_gallons = 18 :=
by
  sorry

end required_lemons_for_20_gallons_l116_116640


namespace max_b_minus_a_l116_116772

theorem max_b_minus_a (a b : ℝ) (h_a: a < 0) (h_ineq: ∀ x : ℝ, (3 * x^2 + a) * (2 * x + b) ≥ 0) : 
b - a = 1 / 3 := 
sorry

end max_b_minus_a_l116_116772


namespace sum_of_two_numbers_l116_116039

theorem sum_of_two_numbers (x y : ℝ) 
  (h1 : x^2 + y^2 = 220) 
  (h2 : x * y = 52) : 
  x + y = 18 :=
by
  sorry

end sum_of_two_numbers_l116_116039


namespace second_cyclist_speed_l116_116373

-- Definitions of the given conditions
def total_course_length : ℝ := 45
def first_cyclist_speed : ℝ := 14
def meeting_time : ℝ := 1.5

-- Lean 4 statement for the proof problem
theorem second_cyclist_speed : 
  ∃ v : ℝ, first_cyclist_speed * meeting_time + v * meeting_time = total_course_length → v = 16 := 
by 
  sorry

end second_cyclist_speed_l116_116373


namespace expression_evaluation_l116_116570

theorem expression_evaluation : 
  3 / 5 * ((2 / 3 + 3 / 8) / 2) - 1 / 16 = 1 / 4 := 
by
  sorry

end expression_evaluation_l116_116570


namespace milk_distribution_l116_116183

theorem milk_distribution 
  (x y z : ℕ)
  (h_total : x + y + z = 780)
  (h_equiv : 3 * x / 4 = 4 * y / 5 ∧ 3 * x / 4 = 4 * z / 7) :
  x = 240 ∧ y = 225 ∧ z = 315 := 
sorry

end milk_distribution_l116_116183


namespace right_triangle_unique_value_l116_116996

theorem right_triangle_unique_value (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
(h1 : a + b + c = (1/2) * a * b) (h2 : c^2 = a^2 + b^2) : a + b - c = 4 :=
by
  sorry

end right_triangle_unique_value_l116_116996


namespace original_price_of_tennis_racket_l116_116562

theorem original_price_of_tennis_racket
  (sneaker_cost : ℝ) (outfit_cost : ℝ) (discount_rate : ℝ) (total_spent : ℝ)
  (price_of_tennis_racket : ℝ) :
  sneaker_cost = 200 → 
  outfit_cost = 250 → 
  discount_rate = 0.20 → 
  total_spent = 750 → 
  price_of_tennis_racket = 289.77 :=
by
  intros hs ho hd ht
  have ht := ht.symm   -- To rearrange the equation
  sorry

end original_price_of_tennis_racket_l116_116562


namespace three_digit_numbers_subtract_297_l116_116564

theorem three_digit_numbers_subtract_297:
  (∃ (p q r : ℕ), 1 ≤ p ∧ p ≤ 9 ∧ 0 ≤ q ∧ q ≤ 9 ∧ 0 ≤ r ∧ r ≤ 9 ∧ (100 * p + 10 * q + r - 297 = 100 * r + 10 * q + p)) →
  (num_valid_three_digit_numbers = 60) :=
by
  sorry

end three_digit_numbers_subtract_297_l116_116564


namespace units_digit_multiplication_l116_116191

-- Define a function to find the units digit of a number
def units_digit (n : ℕ) : ℕ := n % 10

-- Statement of the problem: Given the product 27 * 36, prove that the units digit is 2.
theorem units_digit_multiplication (a b : ℕ) (h1 : units_digit 27 = 7) (h2 : units_digit 36 = 6) :
  units_digit (27 * 36) = 2 :=
by
  have h3 : units_digit (7 * 6) = 2 := by sorry
  exact h3

end units_digit_multiplication_l116_116191


namespace four_faucets_fill_time_correct_l116_116074

-- Define the parameters given in the conditions
def three_faucets_rate (volume : ℕ) (time : ℕ) := volume / time
def one_faucet_rate (rate : ℕ) := rate / 3
def four_faucets_rate (rate : ℕ) := 4 * rate
def fill_time (volume : ℕ) (rate : ℕ) := volume / rate

-- Given problem parameters
def volume_large_tub : ℕ := 100
def time_large_tub : ℕ := 6
def volume_small_tub : ℕ := 50

-- Theorem to be proven
theorem four_faucets_fill_time_correct :
  fill_time volume_small_tub (four_faucets_rate (one_faucet_rate (three_faucets_rate volume_large_tub time_large_tub))) * 60 = 135 :=
sorry

end four_faucets_fill_time_correct_l116_116074


namespace annulus_area_l116_116247

theorem annulus_area (r R x : ℝ) (hR_gt_r : R > r) (h_tangent : r^2 + x^2 = R^2) : 
  π * x^2 = π * (R^2 - r^2) :=
by
  sorry

end annulus_area_l116_116247


namespace minimum_framing_needed_l116_116137

-- Definitions given the conditions
def original_width := 5
def original_height := 7
def enlargement_factor := 4
def border_width := 3
def inches_per_foot := 12

-- Conditions translated to definitions
def enlarged_width := original_width * enlargement_factor
def enlarged_height := original_height * enlargement_factor
def bordered_width := enlarged_width + 2 * border_width
def bordered_height := enlarged_height + 2 * border_width
def perimeter := 2 * (bordered_width + bordered_height)
def perimeter_in_feet := perimeter / inches_per_foot

-- Prove that the minimum number of linear feet of framing required is 10 feet
theorem minimum_framing_needed : perimeter_in_feet = 10 := 
by 
  sorry

end minimum_framing_needed_l116_116137


namespace sufficient_but_not_necessary_l116_116461

theorem sufficient_but_not_necessary (x : ℝ) :
  (x > 1 → x^2 + 2 * x > 0) ∧ ¬(x^2 + 2 * x > 0 → x > 1) :=
by
  sorry

end sufficient_but_not_necessary_l116_116461


namespace liquidX_percentage_l116_116764

variable (wA wB : ℝ) (pA pB : ℝ) (mA mB : ℝ)

-- Conditions
def weightA : ℝ := 200
def weightB : ℝ := 700
def percentA : ℝ := 0.8
def percentB : ℝ := 1.8

-- The question and answer.
theorem liquidX_percentage :
  (percentA / 100 * weightA + percentB / 100 * weightB) / (weightA + weightB) * 100 = 1.58 := by
  sorry

end liquidX_percentage_l116_116764


namespace share_sheets_equally_l116_116787

theorem share_sheets_equally (sheets friends : ℕ) (h_sheets : sheets = 15) (h_friends : friends = 3) : sheets / friends = 5 := by
  sorry

end share_sheets_equally_l116_116787


namespace find_p_q_d_l116_116066

noncomputable def cubic_polynomial_real_root (p q d : ℕ) (x : ℝ) : Prop :=
  27 * x^3 - 12 * x^2 - 4 * x - 1 = 0 ∧ x = (p^(1/3) + q^(1/3) + 1) / d ∧
  p > 0 ∧ q > 0 ∧ d > 0

theorem find_p_q_d :
  ∃ (p q d : ℕ), cubic_polynomial_real_root p q d 1 ∧ p + q + d = 3 :=
by
  sorry

end find_p_q_d_l116_116066


namespace b_seq_formula_c_seq_arithmetic_c_seq_formula_a_seq_formula_sum_S_5_l116_116865

-- Define the sequences a_n, b_n, and c_n along with their properties

-- Definitions
def a_seq (n : ℕ) : ℕ := sorry            -- Define a_n

def S_seq (n : ℕ) : ℕ := sorry            -- Define S_n

def b_seq (n : ℕ) : ℕ := a_seq (n+1) - 2 * a_seq n

def c_seq (n : ℕ) : ℕ := a_seq n / 2^n

-- Conditions
axiom S_n_condition (n : ℕ) : S_seq (n+1) = 4 * a_seq n + 2
axiom a_1_condition : a_seq 1 = 1

-- Goals
theorem b_seq_formula (n : ℕ) : b_seq n = 3 * 2^(n-1) := sorry

theorem c_seq_arithmetic (n : ℕ) : c_seq (n+1) - c_seq n = 3 / 4 := sorry

theorem c_seq_formula (n : ℕ) : c_seq n = (3 * n - 1) / 4 := sorry

theorem a_seq_formula (n : ℕ) : a_seq n = (3 * n - 1) * 2^(n-2) := sorry

theorem sum_S_5 : S_seq 5 = 178 := sorry

end b_seq_formula_c_seq_arithmetic_c_seq_formula_a_seq_formula_sum_S_5_l116_116865


namespace Ella_food_each_day_l116_116020

variable {E : ℕ} -- Define E as the number of pounds of food Ella eats each day

def food_dog_eats (E : ℕ) : ℕ := 4 * E -- Definition of food the dog eats each day

def total_food_eaten_in_10_days (E : ℕ) : ℕ := 10 * E + 10 * (food_dog_eats E) -- Total food (Ella and dog) in 10 days

theorem Ella_food_each_day : total_food_eaten_in_10_days E = 1000 → E = 20 :=
by
  intros h -- Assume the given condition
  sorry -- Skip the actual proof

end Ella_food_each_day_l116_116020


namespace compute_value_l116_116504

variables {p q r : ℝ}

theorem compute_value (h1 : (p * q) / (p + r) + (q * r) / (q + p) + (r * p) / (r + q) = -7)
                      (h2 : (p * r) / (p + r) + (q * p) / (q + p) + (r * q) / (r + q) = 8) :
  (q / (p + q) + r / (q + r) + p / (r + p)) = 9 :=
sorry

end compute_value_l116_116504


namespace pyramid_edges_l116_116626

-- Define the conditions
def isPyramid (n : ℕ) : Prop :=
  (n + 1) + (n + 1) = 16

-- Statement to be proved
theorem pyramid_edges : ∃ (n : ℕ), isPyramid n ∧ 2 * n = 14 :=
by {
  sorry
}

end pyramid_edges_l116_116626


namespace cos_double_angle_zero_l116_116258

theorem cos_double_angle_zero (α : ℝ) (h : Real.sin (Real.pi / 6 - α) = Real.cos (Real.pi / 6 + α)) : Real.cos (2 * α) = 0 := 
sorry

end cos_double_angle_zero_l116_116258


namespace prime_ge_5_divisible_by_12_l116_116382

theorem prime_ge_5_divisible_by_12 (p : ℕ) (hp1 : p ≥ 5) (hp2 : Nat.Prime p) : 12 ∣ p^2 - 1 :=
by
  sorry

end prime_ge_5_divisible_by_12_l116_116382


namespace ten_percent_of_n_l116_116931

variable (n f : ℝ)

theorem ten_percent_of_n (h : n - (1 / 4 * 2) - (1 / 3 * 3) - f * n = 27) : 
  0.10 * n = 0.10 * (28.5 / (1 - f)) :=
by
  simp only [*, mul_one_div_cancel, mul_sub, sub_eq_add_neg, add_div, div_self, one_div, mul_add]
  sorry

end ten_percent_of_n_l116_116931


namespace simplify_expression_correct_l116_116560

noncomputable def simplify_expression : Prop :=
  (1 / (Real.log 3 / Real.log 6 + 1) + 1 / (Real.log 7 / Real.log 15 + 1) + 1 / (Real.log 4 / Real.log 12 + 1)) = -Real.log 84 / Real.log 10

theorem simplify_expression_correct : simplify_expression :=
  by
    sorry

end simplify_expression_correct_l116_116560


namespace middle_integer_is_six_l116_116577

def valid_even_integer (n : ℕ) : Prop :=
  ∃ (x y z : ℕ), n = x ∧ x = n - 2 ∧ y = n ∧ z = n + 2 ∧ x < y ∧ y < z ∧
  x % 2 = 0 ∧ y % 2 = 0 ∧ z % 2 = 0 ∧
  1 ≤ x ∧ x ≤ 9 ∧ 1 ≤ y ∧ y ≤ 9 ∧ 1 ≤ z ∧ z ≤ 9

theorem middle_integer_is_six (n : ℕ) (h : valid_even_integer n) :
  n = 6 :=
by
  sorry

end middle_integer_is_six_l116_116577


namespace wrap_XL_boxes_per_roll_l116_116000

-- Conditions
def rolls_per_shirt_box : ℕ := 5
def num_shirt_boxes : ℕ := 20
def num_XL_boxes : ℕ := 12
def cost_per_roll : ℕ := 4
def total_cost : ℕ := 32

-- Prove that one roll of wrapping paper can wrap 3 XL boxes
theorem wrap_XL_boxes_per_roll : (num_XL_boxes / ((total_cost / cost_per_roll) - (num_shirt_boxes / rolls_per_shirt_box))) = 3 := 
sorry

end wrap_XL_boxes_per_roll_l116_116000


namespace inequality_AM_GM_l116_116700

variable {a b c d : ℝ}
variable (h₁ : 0 < a)
variable (h₂ : 0 < b)
variable (h₃ : 0 < c)
variable (h₄ : 0 < d)

theorem inequality_AM_GM :
  (c / a * (8 * b + c) + d / b * (8 * c + d) + a / c * (8 * d + a) + b / d * (8 * a + b)) ≥ 9 * (a + b + c + d) :=
sorry

end inequality_AM_GM_l116_116700


namespace number_equation_l116_116856

-- Lean statement equivalent to the mathematical problem
theorem number_equation (x : ℝ) (h : 5 * x - 2 * x = 10) : 5 * x - 2 * x = 10 :=
by exact h

end number_equation_l116_116856


namespace regular_2020_gon_isosceles_probability_l116_116275

theorem regular_2020_gon_isosceles_probability :
  let n := 2020
  let totalTriangles := (n * (n - 1) * (n - 2)) / 6
  let isoscelesTriangles := n * ((n - 2) / 2)
  let probability := isoscelesTriangles * 6 / totalTriangles
  let (a, b) := (1, 673)
  100 * a + b = 773 := by
    sorry

end regular_2020_gon_isosceles_probability_l116_116275


namespace vertical_asymptotes_count_l116_116307

theorem vertical_asymptotes_count : 
  let f (x : ℝ) := (x - 2) / (x^2 + 4*x - 5) 
  ∃! c : ℕ, c = 2 :=
by
  sorry

end vertical_asymptotes_count_l116_116307


namespace arithmetic_sequence_ratio_l116_116647

noncomputable def A_n (a d : ℤ) (n : ℕ) : ℤ :=
  n * (2 * a + (n - 1) * d) / 2

noncomputable def B_n (b e : ℤ) (n : ℕ) : ℤ :=
  n * (2 * b + (n - 1) * e) / 2

theorem arithmetic_sequence_ratio (a d b e : ℤ) :
  (∀ n : ℕ, n ≠ 0 → A_n a d n / B_n b e n = (5 * n - 3) / (n + 9)) →
  (a + 5 * d) / (b + 2 * e) = 26 / 7 :=
by
  sorry

end arithmetic_sequence_ratio_l116_116647


namespace find_dolls_l116_116387

namespace DollsProblem

variables (S D : ℕ) -- Define S and D as natural numbers

-- Conditions as per the problem
def cond1 : Prop := 4 * S + 3 = D
def cond2 : Prop := 5 * S = D + 6

-- Theorem stating the problem
theorem find_dolls (h1 : cond1 S D) (h2 : cond2 S D) : D = 39 :=
by
  sorry

end DollsProblem

end find_dolls_l116_116387


namespace speed_goods_train_l116_116287

def length_train : ℝ := 50
def length_platform : ℝ := 250
def time_crossing : ℝ := 15

/-- The speed of the goods train in km/hr given the length of the train, the length of the platform, and the time to cross the platform. -/
theorem speed_goods_train :
  (length_train + length_platform) / time_crossing * 3.6 = 72 :=
by
  sorry

end speed_goods_train_l116_116287


namespace volume_and_area_of_pyramid_l116_116291

-- Define the base of the pyramid.
def rect (EF FG : ℕ) : Prop := EF = 10 ∧ FG = 6

-- Define the perpendicular relationships and height of the pyramid.
def pyramid (EF FG PE : ℕ) : Prop := 
  rect EF FG ∧
  PE = 10 ∧ 
  (PE > 0) -- Given conditions include perpendicular properties, implying height is positive.

-- Problem translation: Prove the volume and area calculations.
theorem volume_and_area_of_pyramid (EF FG PE : ℕ) 
  (h1 : rect EF FG) 
  (h2 : PE = 10) : 
  (1 / 3 * EF * FG * PE = 200 ∧ 1 / 2 * EF * FG = 30) := 
by
  sorry

end volume_and_area_of_pyramid_l116_116291


namespace bus_passengers_l116_116789

def passengers_after_first_stop := 7

def passengers_after_second_stop := passengers_after_first_stop - 3 + 5

def passengers_after_third_stop := passengers_after_second_stop - 2 + 4

theorem bus_passengers (passengers_after_first_stop passengers_after_second_stop passengers_after_third_stop : ℕ) : passengers_after_third_stop = 11 :=
by
  sorry

end bus_passengers_l116_116789


namespace ellipse_range_of_k_l116_116887

theorem ellipse_range_of_k (k : ℝ) :
  (∃ (eq : ((x y : ℝ) → (x ^ 2 / (3 + k) + y ^ 2 / (2 - k) = 1))),
  ((3 + k > 0) ∧ (2 - k > 0) ∧ (3 + k ≠ 2 - k))) ↔
  (k ∈ Set.Ioo (-3 : ℝ) ((-1) / 2) ∪ Set.Ioo ((-1) / 2) 2) :=
by sorry

end ellipse_range_of_k_l116_116887


namespace minimize_F_l116_116220

theorem minimize_F : ∃ x1 x2 x3 x4 x5 : ℝ, 
  (-2 * x1 + x2 + x3 = 2) ∧ 
  (x1 - 2 * x2 + x4 = 2) ∧ 
  (x1 + x2 + x5 = 5) ∧ 
  (x1 ≥ 0) ∧ 
  (x2 ≥ 0) ∧ 
  (x2 - x1 = -3) :=
by {
  sorry
}

end minimize_F_l116_116220


namespace exists_a_log_eq_l116_116641

theorem exists_a_log_eq (a : ℝ) (h : a = 10 ^ ((Real.log 2 * Real.log 3) / (Real.log 2 + Real.log 3))) :
  ∀ x > 0, Real.log x / Real.log 2 + Real.log x / Real.log 3 = Real.log x / Real.log a :=
by
  sorry

end exists_a_log_eq_l116_116641


namespace half_angle_quadrant_l116_116168

variables {α : ℝ} {k : ℤ} {n : ℤ}

theorem half_angle_quadrant (h : ∃ k : ℤ, k * 360 + 180 < α ∧ α < k * 360 + 270) :
  ∃ (n : ℤ), (n * 360 + 90 < α / 2 ∧ α / 2 < n * 360 + 135) ∨ 
      (n * 360 + 270 < α / 2 ∧ α / 2 < n * 360 + 315) :=
by sorry

end half_angle_quadrant_l116_116168


namespace number_times_frac_eq_cube_l116_116974

theorem number_times_frac_eq_cube (x : ℕ) : x * (1/6)^2 = 6^3 → x = 7776 :=
by
  intro h
  -- skipped proof
  sorry

end number_times_frac_eq_cube_l116_116974


namespace number_of_terms_is_10_l116_116549

noncomputable def arith_seq_number_of_terms (a : ℕ) (n : ℕ) (d : ℕ) : Prop :=
  (n % 2 = 0) ∧ ((n-1)*d = 16) ∧ (n * (2*a + (n-2)*d) = 56) ∧ (n * (2*a + n*d) = 76)

theorem number_of_terms_is_10 (a d n : ℕ) (h : arith_seq_number_of_terms a n d) : n = 10 := by
  sorry

end number_of_terms_is_10_l116_116549


namespace expression_value_l116_116799

theorem expression_value : (8 * 6) - (4 / 2) = 46 :=
by
  sorry

end expression_value_l116_116799


namespace find_pos_ints_l116_116701

theorem find_pos_ints (a m n : ℕ) (ha : 0 < a) (hm : 0 < m) (hn : 0 < n) :
    (((m = 1) ∨ (a = 1) ∨ (a = 2 ∧ m = 3 ∧ 2 ≤ n)) →
    (a^m + 1 ∣ (a + 1)^n)) :=
by
  sorry

end find_pos_ints_l116_116701


namespace line_equation_l116_116195

theorem line_equation
  (P : ℝ × ℝ) (A : ℝ × ℝ) (B : ℝ × ℝ) (hP : P = (-4, 6))
  (hxA : A.2 = 0) (hyB : B.1 = 0)
  (hMidpoint : P = ((A.1 + B.1)/2, (A.2 + B.2)/2)):
  3 * A.1 - 2 * B.2 + 24 = 0 :=
by
  -- Define point P
  let P := (-4, 6)
  -- Define points A and B, knowing P is the midpoint of AB and using conditions from the problem
  let A := (-8, 0)
  let B := (0, 12)
  sorry

end line_equation_l116_116195


namespace complement_intersection_l116_116801

-- Define the universal set U and sets A and B
def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8}
def A : Set ℕ := {3, 4, 5}
def B : Set ℕ := {4, 7, 8}

-- Compute the complements
def complement_U (s : Set ℕ) : Set ℕ := U \ s
def comp_A : Set ℕ := complement_U A
def comp_B : Set ℕ := complement_U B

-- Define the intersection of the complements
def intersection_complements : Set ℕ := comp_A ∩ comp_B

-- The theorem to prove
theorem complement_intersection :
  intersection_complements = {1, 2, 6} :=
by
  sorry

end complement_intersection_l116_116801


namespace steve_speed_l116_116681

theorem steve_speed (v : ℝ) : 
  (John_initial_distance_behind_Steve = 15) ∧ 
  (John_final_distance_ahead_of_Steve = 2) ∧ 
  (John_speed = 4.2) ∧ 
  (final_push_duration = 34) → 
  v * final_push_duration = (John_speed * final_push_duration) - (John_initial_distance_behind_Steve + John_final_distance_ahead_of_Steve) →
  v = 3.7 := 
by
  intros hconds heq
  exact sorry

end steve_speed_l116_116681


namespace find_m_l116_116466

-- Define the pattern of splitting cubes into odd numbers
def split_cubes (m : ℕ) : List ℕ := 
  let rec odd_numbers (n : ℕ) : List ℕ :=
    if n = 0 then []
    else (2 * n - 1) :: odd_numbers (n - 1)
  odd_numbers m

-- Define the condition that 59 is part of the split numbers of m^3
def is_split_number (m : ℕ) (n : ℕ) : Prop :=
  n ∈ (split_cubes m)

-- Prove that if 59 is part of the split numbers of m^3, then m = 8
theorem find_m (m : ℕ) (h : is_split_number m 59) : m = 8 := 
sorry

end find_m_l116_116466


namespace rainwater_cows_l116_116873

theorem rainwater_cows (chickens goats cows : ℕ) 
  (h1 : chickens = 18) 
  (h2 : goats = 2 * chickens) 
  (h3 : goats = 4 * cows) : 
  cows = 9 := 
sorry

end rainwater_cows_l116_116873


namespace cleaning_time_is_correct_l116_116331

-- Define the given conditions
def vacuuming_minutes_per_day : ℕ := 30
def vacuuming_days_per_week : ℕ := 3
def dusting_minutes_per_day : ℕ := 20
def dusting_days_per_week : ℕ := 2

-- Define the total cleaning time per week
def total_cleaning_time_per_week : ℕ :=
  (vacuuming_minutes_per_day * vacuuming_days_per_week) + (dusting_minutes_per_day * dusting_days_per_week)

-- State the theorem we want to prove
theorem cleaning_time_is_correct : total_cleaning_time_per_week = 130 := by
  sorry

end cleaning_time_is_correct_l116_116331


namespace range_of_a_l116_116651

-- Given function
def f (x a : ℝ) : ℝ := x^3 + a*x^2 + (a + 6)*x + 1

-- Derivative of the function
def f' (x a : ℝ) : ℝ := 3*x^2 + 2*a*x + (a + 6)

-- Discriminant of the derivative
def discriminant (a : ℝ) : ℝ := 4*a^2 - 12*(a + 6)

-- Proof that the range of 'a' is 'a < -3 or a > 6' for f(x) to have both maximum and minimum values
theorem range_of_a (a : ℝ) : discriminant a > 0 ↔ (a < -3 ∨ a > 6) :=
by
  sorry

end range_of_a_l116_116651


namespace remaining_volume_of_cube_l116_116429

theorem remaining_volume_of_cube :
  let s := 6
  let r := 3
  let h := 6
  let V_cube := s^3
  let V_cylinder := Real.pi * (r^2) * h
  V_cube - V_cylinder = 216 - 54 * Real.pi :=
by
  sorry

end remaining_volume_of_cube_l116_116429


namespace abc_inequalities_l116_116611

noncomputable def a : ℝ := Real.log 1 / Real.log 2 - Real.log 3 / Real.log 2
noncomputable def b : ℝ := (1 / 2) ^ 3
noncomputable def c : ℝ := Real.sqrt 3

theorem abc_inequalities :
  a < b ∧ b < c :=
by
  -- Proof omitted
  sorry

end abc_inequalities_l116_116611


namespace eq_neg_one_fifth_l116_116775

theorem eq_neg_one_fifth : 
  ((1 : ℝ) / ((-5) ^ 4) ^ 2 * (-5) ^ 7) = -1 / 5 := by
  sorry

end eq_neg_one_fifth_l116_116775


namespace jenna_eel_length_l116_116266

theorem jenna_eel_length (j b : ℕ) (h1 : b = 3 * j) (h2 : b + j = 64) : j = 16 := by 
  sorry

end jenna_eel_length_l116_116266


namespace circle_area_and_circumference_changes_l116_116578

noncomputable section

structure Circle :=
  (r : ℝ)

def area (c : Circle) : ℝ := Real.pi * c.r^2

def circumference (c : Circle) : ℝ := 2 * Real.pi * c.r

def percentage_change (original new : ℝ) : ℝ :=
  ((original - new) / original) * 100

theorem circle_area_and_circumference_changes
  (r1 r2 : ℝ) (c1 : Circle := {r := r1}) (c2 : Circle := {r := r2})
  (h1 : r1 = 5) (h2 : r2 = 4) :
  let original_area := area c1
  let new_area := area c2
  let original_circumference := circumference c1
  let new_circumference := circumference c2
  percentage_change original_area new_area = 36 ∧
  new_circumference = 8 * Real.pi ∧
  percentage_change original_circumference new_circumference = 20 :=
by
  sorry

end circle_area_and_circumference_changes_l116_116578


namespace smallest_whole_number_larger_than_sum_l116_116599

theorem smallest_whole_number_larger_than_sum :
    let sum := 2 + 1 / 2 + 3 + 1 / 3 + 4 + 1 / 4 + 5 + 1 / 5 
    let smallest_whole := 16
    (sum < smallest_whole ∧ smallest_whole - 1 < sum) := 
by
    sorry

end smallest_whole_number_larger_than_sum_l116_116599


namespace student_marks_l116_116008

variable (x : ℕ)
variable (passing_marks : ℕ)
variable (max_marks : ℕ := 400)
variable (fail_by : ℕ := 14)

theorem student_marks :
  (passing_marks = 36 * max_marks / 100) →
  (x + fail_by = passing_marks) →
  x = 130 :=
by sorry

end student_marks_l116_116008


namespace eval_expression_l116_116946

theorem eval_expression : 0.5 * 0.8 - 0.2 = 0.2 := by
  sorry

end eval_expression_l116_116946


namespace edward_original_amount_l116_116945

theorem edward_original_amount (spent left total : ℕ) (h1 : spent = 13) (h2 : left = 6) (h3 : total = spent + left) : total = 19 := by 
  sorry

end edward_original_amount_l116_116945


namespace balls_in_boxes_l116_116199

theorem balls_in_boxes :
  let n := 7
  let k := 3
  (Nat.choose (n + k - 1) (k - 1)) = 36 :=
by
  let n := 7
  let k := 3
  sorry

end balls_in_boxes_l116_116199


namespace total_questions_attempted_l116_116518

theorem total_questions_attempted (C W T : ℕ) 
    (hC : C = 36) 
    (hScore : 120 = (4 * C) - W) 
    (hT : T = C + W) : 
    T = 60 := 
by 
  sorry

end total_questions_attempted_l116_116518


namespace age_ratio_l116_116852
open Nat

theorem age_ratio (B_c : ℕ) (h1 : B_c = 42) (h2 : ∀ A_c, A_c = B_c + 12) : (A_c + 10) / (B_c - 10) = 2 :=
by
  sorry

end age_ratio_l116_116852


namespace variance_of_scores_l116_116355

-- Define the list of scores
def scores : List ℕ := [110, 114, 121, 119, 126]

-- Define the formula for variance calculation
def variance (l : List ℕ) : ℚ :=
  let n := l.length
  let mean := (l.sum : ℚ) / n
  (l.map (λ x => ((x : ℚ) - mean) ^ 2)).sum / n

-- The main theorem to be proved
theorem variance_of_scores :
  variance scores = 30.8 := 
  by
    sorry

end variance_of_scores_l116_116355


namespace sum_of_four_circles_l116_116309

open Real

theorem sum_of_four_circles:
  ∀ (s c : ℝ), 
  (2 * s + 3 * c = 26) → 
  (3 * s + 2 * c = 23) → 
  (4 * c = 128 / 5) :=
by
  intros s c h1 h2
  sorry

end sum_of_four_circles_l116_116309


namespace find_k_l116_116582

-- Definitions of conditions
def equation1 (x k : ℝ) : Prop := x^2 + k*x + 10 = 0
def equation2 (x k : ℝ) : Prop := x^2 - k*x + 10 = 0
def roots_relation (a b k : ℝ) : Prop :=
  equation1 a k ∧ 
  equation1 b k ∧ 
  equation2 (a + 3) k ∧
  equation2 (b + 3) k

-- Statement to be proven
theorem find_k (a b k : ℝ) (h : roots_relation a b k) : k = 3 :=
sorry

end find_k_l116_116582


namespace min_fraction_ineq_l116_116686

theorem min_fraction_ineq (x y : ℝ) (hx : 0.4 ≤ x ∧ x ≤ 0.6) (hy : 0.3 ≤ y ∧ y ≤ 0.5) :
  ∃ z, (z = x * y / (x^2 + 2 * y^2)) ∧ z = 1 / 3 := sorry

end min_fraction_ineq_l116_116686


namespace minimum_value_of_expression_l116_116717

theorem minimum_value_of_expression (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a + b + c = 6) :
  (9 / a + 16 / b + 25 / c) = 24 :=
sorry

end minimum_value_of_expression_l116_116717


namespace solve_for_n_l116_116953

theorem solve_for_n (n : ℕ) (h : 2 * 2 * 3 * 3 * 5 * 6 = 5 * 6 * n * n) : n = 6 := by
  sorry

end solve_for_n_l116_116953


namespace compute_pqr_l116_116342

theorem compute_pqr (p q r : ℤ) (hp : p ≠ 0) (hq : q ≠ 0) (hr : r ≠ 0) 
  (h_sum : p + q + r = 30) 
  (h_equation : 1 / p + 1 / q + 1 / r + 420 / (p * q * r) = 1) : 
  p * q * r = 576 :=
sorry

end compute_pqr_l116_116342


namespace y_intercept_of_tangent_line_l116_116498

def point (x y : ℝ) : Prop := true

def circle_eq (x y : ℝ) : ℝ := x^2 + y^2 + 4*x - 2*y + 3

theorem y_intercept_of_tangent_line :
  ∃ m b : ℝ,
  (∀ x : ℝ, circle_eq x (m*x + b) = 0 → m * m = 1) ∧
  (∃ P: ℝ × ℝ, P = (-1, 0)) ∧
  ∀ b : ℝ, (∃ m : ℝ, m = 1 ∧ (∃ P: ℝ × ℝ, P = (-1, 0)) ∧ b = 1) := 
sorry

end y_intercept_of_tangent_line_l116_116498


namespace find_a_range_l116_116683

-- Definitions as per conditions
def prop_P (a : ℝ) : Prop := ∀ x : ℝ, x^2 + a*x + 4 > 0
def prop_Q (a : ℝ) : Prop := ∃ x : ℝ, x^2 - 2*x + a = 0

-- Given conditions
def P_true (a : ℝ) (h : prop_P a) : Prop :=
  ∀ (a : ℝ), a^2 - 16 < 0

def Q_false (a : ℝ) (h : ¬prop_Q a) : Prop :=
  ∀ (a : ℝ), a > 1

-- Main theorem
theorem find_a_range (a : ℝ) (hP : prop_P a) (hQ : ¬prop_Q a) : 1 < a ∧ a < 4 :=
sorry

end find_a_range_l116_116683


namespace sum_possible_values_of_p_l116_116845

theorem sum_possible_values_of_p (p q : ℤ) (h1 : p + q = 2010)
  (h2 : ∃ (α β : ℕ), (10 * α * β = q) ∧ (10 * (α + β) = -p)) :
  p = -3100 :=
by
  sorry

end sum_possible_values_of_p_l116_116845


namespace area_of_triangle_l116_116922

theorem area_of_triangle (a b c : ℝ) (h₁ : a + b = 14) (h₂ : c = 10) (h₃ : a^2 + b^2 = c^2) :
  (1 / 2) * a * b = 24 :=
  sorry

end area_of_triangle_l116_116922


namespace third_player_game_count_l116_116289

theorem third_player_game_count (fp_games : ℕ) (sp_games : ℕ) (tp_games : ℕ) (total_games : ℕ) 
  (h1 : fp_games = 10) (h2 : sp_games = 21) (h3 : total_games = sp_games) 
  (h4 : total_games = fp_games + tp_games + 1): tp_games = 11 := 
  sorry

end third_player_game_count_l116_116289


namespace b_value_l116_116861

theorem b_value (x y b : ℝ) (h1 : x / (2 * y) = 3 / 2) (h2 : (7 * x + b * y) / (x - 2 * y) = 25) : b = 4 := 
by
  sorry

end b_value_l116_116861


namespace max_value_of_f_l116_116322

noncomputable def f (x : ℝ) : ℝ := 3 * Real.cos (2 * x)

theorem max_value_of_f :
  ∃ x : ℝ, ∃ k : ℤ, f x = 3 ∧ x = k * Real.pi :=
by
  -- The proof is omitted
  sorry

end max_value_of_f_l116_116322


namespace range_of_a_l116_116959

def p (a : ℝ) : Prop := ∀ x : ℝ, x^2 + 2 * a * x + 4 > 0
def q (a : ℝ) : Prop := ∀ x : ℝ, (3 - 2 * a) ^ x > 0 -- using our characterization for 'increasing'

theorem range_of_a (a : ℝ) : (p a ∨ q a) ∧ ¬ (p a ∧ q a) ↔ (a ≤ -2 ∨ (1 ≤ a ∧ a < 2)) :=
by
  sorry

end range_of_a_l116_116959


namespace average_payment_l116_116904

-- Each condition from part a) is used as a definition here
variable (n : Nat) (p1 p2 first_payment remaining_payment : Nat)

-- Conditions given in natural language
def payments_every_year : Prop :=
  n = 52 ∧
  first_payment = 410 ∧
  remaining_payment = first_payment + 65 ∧
  p1 = 8 * first_payment ∧
  p2 = 44 * remaining_payment ∧
  p2 = 44 * (first_payment + 65) ∧
  p1 + p2 = 24180

-- The theorem to prove based on the conditions
theorem average_payment 
  (h : payments_every_year n p1 p2 first_payment remaining_payment) 
  : (p1 + p2) / n = 465 := 
sorry  -- Proof is omitted intentionally

end average_payment_l116_116904


namespace find_a_plus_b_l116_116471

-- Conditions for the lines
def line_l0 (x y : ℝ) : Prop := x - y + 1 = 0
def line_l1 (a : ℝ) (x y : ℝ) : Prop := a * x - 2 * y + 1 = 0
def line_l2 (b : ℝ) (x y : ℝ) : Prop := x + b * y + 3 = 0

-- Perpendicularity condition for l1 to l0
def perpendicular (a : ℝ) : Prop := 1 * a + (-1) * (-2) = 0

-- Parallel condition for l2 to l0
def parallel (b : ℝ) : Prop := 1 * b = (-1) * 1

-- Prove the value of a + b given the conditions
theorem find_a_plus_b (a b : ℝ) 
  (h1 : perpendicular a)
  (h2 : parallel b) : a + b = -3 :=
sorry

end find_a_plus_b_l116_116471


namespace remainder_when_divided_by_30_l116_116381

theorem remainder_when_divided_by_30 (x : ℤ) : 
  (4 + x) % 8 = 9 % 8 ∧
  (6 + x) % 27 = 4 % 27 ∧
  (8 + x) % 125 = 49 % 125 
  → x % 30 = 1 % 30 := by
  sorry

end remainder_when_divided_by_30_l116_116381


namespace projection_matrix_ordered_pair_l116_116842

theorem projection_matrix_ordered_pair (a c : ℚ)
  (P : Matrix (Fin 2) (Fin 2) ℚ) 
  (P := ![![a, 15 / 34], ![c, 25 / 34]]) :
  P * P = P ->
  (a, c) = (9 / 34, 15 / 34) :=
by
  sorry

end projection_matrix_ordered_pair_l116_116842


namespace total_donations_correct_l116_116553

def num_basketball_hoops : Nat := 60

def num_hoops_with_balls : Nat := num_basketball_hoops / 2

def num_pool_floats : Nat := 120
def num_damaged_floats : Nat := num_pool_floats / 4
def num_remaining_floats : Nat := num_pool_floats - num_damaged_floats

def num_footballs : Nat := 50
def num_tennis_balls : Nat := 40

def num_hoops_without_balls : Nat := num_basketball_hoops - num_hoops_with_balls

def total_donations : Nat := 
  num_hoops_without_balls + num_hoops_with_balls + num_remaining_floats + num_footballs + num_tennis_balls

theorem total_donations_correct : total_donations = 240 := by
  sorry

end total_donations_correct_l116_116553


namespace find_A_from_eq_l116_116804

theorem find_A_from_eq (A : ℕ) (h : 10 - A = 6) : A = 4 :=
by
  sorry

end find_A_from_eq_l116_116804


namespace proportional_parts_l116_116156

theorem proportional_parts (A B C D : ℕ) (number : ℕ) (h1 : A = 5 * x) (h2 : B = 7 * x) (h3 : C = 4 * x) (h4 : D = 8 * x) (h5 : C = 60) : number = 360 := by
  sorry

end proportional_parts_l116_116156


namespace truck_distance_in_3_hours_l116_116832

theorem truck_distance_in_3_hours : 
  ∀ (speed_2miles_2_5minutes : ℝ) 
    (time_minutes : ℝ),
    (speed_2miles_2_5minutes = 2 / 2.5) →
    (time_minutes = 180) →
    (speed_2miles_2_5minutes * time_minutes = 144) :=
by
  intros
  sorry

end truck_distance_in_3_hours_l116_116832


namespace arithmetic_sqrt_of_49_l116_116894

theorem arithmetic_sqrt_of_49 : ∃ x : ℕ, x^2 = 49 ∧ x = 7 :=
by
  sorry

end arithmetic_sqrt_of_49_l116_116894


namespace coin_toss_probability_l116_116151

-- Definition of the conditions
def total_outcomes : ℕ := 2 ^ 8
def favorable_outcomes : ℕ := Nat.choose 8 3
def probability : ℚ := favorable_outcomes / total_outcomes

-- The theorem to be proved
theorem coin_toss_probability : probability = 7 / 32 := by
  sorry

end coin_toss_probability_l116_116151


namespace hyperbola_eccentricity_l116_116821

theorem hyperbola_eccentricity (a b c : ℝ) (h₀ : a > 0) (h₁ : b > 0) 
  (h₂ : 4 * c^2 = 25) (h₃ : a = 1/2) : c/a = 5 :=
by
  sorry

end hyperbola_eccentricity_l116_116821


namespace gcd_245_1001_l116_116056

-- Definitions based on the given conditions

def fact245 : ℕ := 5 * 7^2
def fact1001 : ℕ := 7 * 11 * 13

-- Lean 4 statement of the proof problem
theorem gcd_245_1001 : Nat.gcd fact245 fact1001 = 7 :=
by
  -- Add the prime factorizations as assumptions
  have h1: fact245 = 245 := by sorry
  have h2: fact1001 = 1001 := by sorry
  -- The goal is to prove the GCD
  sorry

end gcd_245_1001_l116_116056


namespace desserts_brought_by_mom_l116_116375

-- Definitions for the number of each type of dessert
def num_coconut := 1
def num_meringues := 2
def num_caramel := 7

-- Conditions from the problem as definitions
def total_desserts := num_coconut + num_meringues + num_caramel = 10
def fewer_coconut_than_meringues := num_coconut < num_meringues
def most_caramel := num_caramel > num_meringues
def josef_jakub_condition := (num_coconut + num_meringues + num_caramel) - (4 * 2) = 1

-- We need to prove the answer based on these conditions
theorem desserts_brought_by_mom :
  total_desserts ∧ fewer_coconut_than_meringues ∧ most_caramel ∧ josef_jakub_condition → 
  num_coconut = 1 ∧ num_meringues = 2 ∧ num_caramel = 7 :=
by sorry

end desserts_brought_by_mom_l116_116375


namespace winner_percentage_l116_116467

theorem winner_percentage (W L V : ℕ) 
    (hW : W = 868) 
    (hDiff : W - L = 336)
    (hV : V = W + L) : 
    (W * 100 / V) = 62 := 
by 
    sorry

end winner_percentage_l116_116467


namespace gate_perimeter_l116_116786

theorem gate_perimeter (r : ℝ) (theta : ℝ) (h1 : r = 2) (h2 : theta = π / 2) :
  let arc_length := (3 / 4) * (2 * π * r)
  let radii_length := 2 * r
  arc_length + radii_length = 3 * π + 4 :=
by
  simp [h1, h2]
  sorry

end gate_perimeter_l116_116786


namespace geo_seq_arith_seq_l116_116348

theorem geo_seq_arith_seq (a_n : ℕ → ℝ) (S : ℕ → ℝ) (q : ℝ) (h_gp : ∀ n, a_n (n+1) = a_n n * q)
  (h_pos : ∀ n, a_n n > 0) (h_arith : a_n 4 - a_n 3 = a_n 5 - a_n 4) 
  (hq_pos : q > 0) (hq_neq1 : q ≠ 1) :
  S 6 / S 3 = 2 := by
  sorry

end geo_seq_arith_seq_l116_116348


namespace round_robin_cycles_l116_116979

-- Define the conditions
def teams : ℕ := 28
def wins_per_team : ℕ := 13
def losses_per_team : ℕ := 13
def total_teams_games := teams * (teams - 1) / 2
def sets_of_three_teams := (teams * (teams - 1) * (teams - 2)) / 6

-- Define the problem statement
theorem round_robin_cycles :
  -- We need to show that the number of sets of three teams {A, B, C} where A beats B, B beats C, and C beats A is 1092
  (sets_of_three_teams - (teams * (wins_per_team * (wins_per_team - 1)) / 2)) = 1092 :=
by
  sorry

end round_robin_cycles_l116_116979


namespace opposite_of_2_is_minus_2_l116_116388

-- Define the opposite function
def opposite (x : ℤ) : ℤ := -x

-- Assert the theorem to prove that the opposite of 2 is -2
theorem opposite_of_2_is_minus_2 : opposite 2 = -2 := by
  sorry -- Placeholder for the proof

end opposite_of_2_is_minus_2_l116_116388


namespace smallest_benches_l116_116989

theorem smallest_benches (N : ℕ) (h1 : ∃ n, 8 * n = 40 ∧ 10 * n = 40) : N = 20 :=
sorry

end smallest_benches_l116_116989


namespace acai_juice_cost_l116_116854

noncomputable def cost_per_litre_juice (x : ℝ) : Prop :=
  let total_cost_cocktail := 1399.45 * 53.333333333333332
  let cost_mixed_fruit_juice := 32 * 262.85
  let cost_acai_juice := 21.333333333333332 * x
  total_cost_cocktail = cost_mixed_fruit_juice + cost_acai_juice

/-- The cost per litre of the açaí berry juice is $3105.00 given the specified conditions. -/
theorem acai_juice_cost : cost_per_litre_juice 3105.00 :=
  sorry

end acai_juice_cost_l116_116854


namespace sequence_product_l116_116903

-- Definitions for the arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (a1 d : ℝ), ∀ n, a n = a1 + (n - 1) * d

-- Definitions for the geometric sequence
def is_geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ (b1 r : ℝ), ∀ n, b n = b1 * r ^ (n - 1)

-- Defining the main proposition
theorem sequence_product (a b : ℕ → ℝ) 
  (h_arith : is_arithmetic_sequence a)
  (h_geom  : is_geometric_sequence b)
  (h_eq    : b 7 = a 7)
  (h_cond  : 2 * a 2 - (a 7) ^ 2 + 2 * a 12 = 0) :
  b 3 * b 11 = 16 :=
sorry

end sequence_product_l116_116903


namespace third_smallest_four_digit_in_pascals_triangle_l116_116680

def is_in_pascals_triangle (n : ℕ) : Prop :=
  ∃ (i j : ℕ), j ≤ i ∧ n = Nat.choose i j

def is_four_digit_number (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

theorem third_smallest_four_digit_in_pascals_triangle :
  ∃ n : ℕ, is_in_pascals_triangle n ∧ is_four_digit_number n ∧
  (∀ m : ℕ, is_in_pascals_triangle m ∧ is_four_digit_number m 
   → m = 1000 ∨ m = 1001 ∨ m = n) ∧ n = 1002 := sorry

end third_smallest_four_digit_in_pascals_triangle_l116_116680


namespace simplify_expression_l116_116359

theorem simplify_expression (x y : ℝ) :
  (2 * x^3 * y^2 - 3 * x^2 * y^3) / (1 / 2 * x * y)^2 = 8 * x - 12 * y := by
  sorry

end simplify_expression_l116_116359


namespace system_of_equations_solutions_l116_116158

theorem system_of_equations_solutions (x1 x2 x3 : ℝ) :
  (2 * x1^2 / (1 + x1^2) = x2) ∧ (2 * x2^2 / (1 + x2^2) = x3) ∧ (2 * x3^2 / (1 + x3^2) = x1)
  → (x1 = 0 ∧ x2 = 0 ∧ x3 = 0) ∨ (x1 = 1 ∧ x2 = 1 ∧ x3 = 1) :=
by
  sorry

end system_of_equations_solutions_l116_116158


namespace rect_RS_over_HJ_zero_l116_116234

theorem rect_RS_over_HJ_zero :
  ∃ (A B C D H I J R S: ℝ × ℝ),
    (A = (0, 6)) ∧
    (B = (8, 6)) ∧
    (C = (8, 0)) ∧
    (D = (0, 0)) ∧
    (H = (5, 6)) ∧
    (I = (8, 4)) ∧
    (J = (3, 0)) ∧
    (R = (15 / 13, -12 / 13)) ∧
    (S = (15 / 13, -12 / 13)) ∧
    (RS = dist R S) ∧
    (HJ = dist H J) ∧
    (HJ ≠ 0) ∧
    (RS / HJ = 0) :=
sorry

end rect_RS_over_HJ_zero_l116_116234


namespace percentage_of_people_with_diploma_l116_116456

variable (P : Type) -- P is the type representing people in Country Z.

-- Given Conditions:
def no_diploma_job (population : ℝ) : ℝ := 0.18 * population
def people_with_job (population : ℝ) : ℝ := 0.40 * population
def diploma_no_job (population : ℝ) : ℝ := 0.25 * (0.60 * population)

-- To Prove:
theorem percentage_of_people_with_diploma (population : ℝ) :
  no_diploma_job population + (diploma_no_job population) + (people_with_job population - no_diploma_job population) = 0.37 * population := 
by
  sorry

end percentage_of_people_with_diploma_l116_116456


namespace find_a7_l116_116031

variable {a : ℕ → ℕ}  -- Define the geometric sequence as a function from natural numbers to natural numbers.
variable (h_geo_seq : ∀ (n k : ℕ), a n ^ 2 = a (n - k) * a (n + k)) -- property of geometric sequences
variable (h_a3 : a 3 = 2) -- given a₃ = 2
variable (h_a5 : a 5 = 8) -- given a₅ = 8

theorem find_a7 : a 7 = 32 :=
by
  sorry

end find_a7_l116_116031


namespace each_friend_pays_20_l116_116552

def rent_cottage_cost_per_hour : ℕ := 5
def rent_cottage_hours : ℕ := 8
def total_rent_cost := rent_cottage_cost_per_hour * rent_cottage_hours
def number_of_friends : ℕ := 2
def each_friend_pays := total_rent_cost / number_of_friends

theorem each_friend_pays_20 :
  each_friend_pays = 20 := by
  sorry

end each_friend_pays_20_l116_116552


namespace exists_half_perimeter_area_rectangle_6x1_not_exists_half_perimeter_area_rectangle_2x1_l116_116081

theorem exists_half_perimeter_area_rectangle_6x1 :
  ∃ x₁ x₂ : ℝ, (6 * 1 / 2 = (6 + 1) / 2) ∧
                x₁ * x₂ = 3 ∧
                (x₁ + x₂ = 3.5) ∧
                (x₁ = 2 ∨ x₁ = 1.5) ∧
                (x₂ = 2 ∨ x₂ = 1.5)
:= by
  sorry

theorem not_exists_half_perimeter_area_rectangle_2x1 :
  ¬(∃ x : ℝ, x * (1.5 - x) = 1)
:= by
  sorry

end exists_half_perimeter_area_rectangle_6x1_not_exists_half_perimeter_area_rectangle_2x1_l116_116081


namespace equal_cake_distribution_l116_116818

theorem equal_cake_distribution (total_cakes : ℕ) (total_friends : ℕ) (h_cakes : total_cakes = 150) (h_friends : total_friends = 50) :
  total_cakes / total_friends = 3 := by
  sorry

end equal_cake_distribution_l116_116818


namespace ben_heads_probability_l116_116211

def coin_flip_probability : ℚ :=
  let total_ways := 2^10
  let ways_exactly_five_heads := Nat.choose 10 5
  let probability_exactly_five_heads := ways_exactly_five_heads / total_ways
  let remaining_probability := 1 - probability_exactly_five_heads
  let probability_more_heads := remaining_probability / 2
  probability_more_heads

theorem ben_heads_probability :
  coin_flip_probability = 193 / 512 := by
  sorry

end ben_heads_probability_l116_116211


namespace axis_of_symmetry_shift_l116_116606

-- Define that f is an even function
def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

-- Define the problem statement in Lean
theorem axis_of_symmetry_shift (f : ℝ → ℝ) 
  (h_even : is_even_function f) :
  ∃ x, ∀ y, f (x + y) = f ((x - 1) + y) ∧ x = -1 :=
sorry

end axis_of_symmetry_shift_l116_116606


namespace min_repetitions_2002_div_by_15_l116_116047

-- Define the function that generates the number based on repetitions of "2002" and appending "15"
def generate_number (n : ℕ) : ℕ :=
  let repeated := (List.replicate n 2002).foldl (λ acc x => acc * 10000 + x) 0
  repeated * 100 + 15

-- Define the minimum n for which the generated number is divisible by 15
def min_n_divisible_by_15 : ℕ := 3

-- The theorem stating the problem with its conditions (divisibility by 15)
theorem min_repetitions_2002_div_by_15 :
  ∀ n : ℕ, (generate_number n % 15 = 0) ↔ (n ≥ min_n_divisible_by_15) :=
sorry

end min_repetitions_2002_div_by_15_l116_116047


namespace train_length_proof_l116_116758

noncomputable def length_of_first_train (speed1 speed2 : ℝ) (time : ℝ) (length2 : ℝ) : ℝ :=
  let relative_speed := (speed1 + speed2) * (5 / 18) -- convert to m/s
  let total_distance := relative_speed * time
  total_distance - length2

theorem train_length_proof (speed1 speed2 : ℝ) (time : ℝ) (length2 : ℝ) :
  speed1 = 120 →
  speed2 = 80 →
  time = 9 →
  length2 = 270.04 →
  length_of_first_train speed1 speed2 time length2 = 230 :=
by
  intros h1 h2 h3 h4
  -- Use the defined function and simplify
  rw [h1, h2, h3, h4]
  simp [length_of_first_train]
  sorry

end train_length_proof_l116_116758


namespace least_positive_integer_multiple_of_53_l116_116783

-- Define the problem in a Lean statement.
theorem least_positive_integer_multiple_of_53 :
  ∃ x : ℕ, (3 * x) ^ 2 + 2 * 58 * 3 * x + 58 ^ 2 % 53 = 0 ∧ x = 16 :=
by
  sorry

end least_positive_integer_multiple_of_53_l116_116783


namespace factory_A_higher_output_l116_116218

theorem factory_A_higher_output (a x : ℝ) (a_pos : a > 0) (x_pos : x > 0) 
  (h_eq_march : 1 + 2 * a = (1 + x) ^ 2) : 
  1 + a > 1 + x :=
by
  sorry

end factory_A_higher_output_l116_116218


namespace unique_N_l116_116244

-- Given conditions and question in the problem
variable (N : Matrix (Fin 2) (Fin 2) ℝ)

-- Problem statement: prove that the matrix defined below is the only matrix satisfying the given condition
theorem unique_N 
  (h : ∀ (w : Fin 2 → ℝ), N.mulVec w = -7 • w) 
  : N = ![![-7, 0], ![0, -7]] := 
sorry

end unique_N_l116_116244


namespace distance_from_apex_to_larger_cross_section_l116_116820

namespace PyramidProof

variables (As Al : ℝ) (d h : ℝ)

theorem distance_from_apex_to_larger_cross_section 
  (As_eq : As = 256 * Real.sqrt 2) 
  (Al_eq : Al = 576 * Real.sqrt 2) 
  (d_eq : d = 12) :
  h = 36 := 
sorry

end PyramidProof

end distance_from_apex_to_larger_cross_section_l116_116820


namespace trip_time_l116_116650

open Real

variables (d T : Real)

theorem trip_time :
  (T = d / 30 + (150 - d) / 6) ∧
  (T = 2 * (d / 30) + 1 + (150 - d) / 30) ∧
  (T - 1 = d / 6 + (150 - d) / 30) →
  T = 20 :=
by
  sorry

end trip_time_l116_116650


namespace arithmetic_geometric_mean_inequality_l116_116346

theorem arithmetic_geometric_mean_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) (A : ℝ) (G : ℝ)
  (hA : A = (a + b) / 2) (hG : G = Real.sqrt (a * b)) : A ≥ G :=
by
  sorry

end arithmetic_geometric_mean_inequality_l116_116346


namespace star_value_l116_116263

-- Define the operation a star b
def star (a b : ℕ) : ℕ := a^2 - 2*a*b + b^2

-- We want to prove that 5 star 3 = 4
theorem star_value : star 5 3 = 4 := by
  sorry

end star_value_l116_116263


namespace bank_account_balance_l116_116460

theorem bank_account_balance : 
  ∀ (initial_amount withdraw_amount deposited_amount final_amount : ℕ),
  initial_amount = 230 →
  withdraw_amount = 60 →
  deposited_amount = 2 * withdraw_amount →
  final_amount = initial_amount - withdraw_amount + deposited_amount →
  final_amount = 290 :=
by
  intros
  sorry

end bank_account_balance_l116_116460


namespace determine_machines_in_first_group_l116_116424

noncomputable def machines_in_first_group (x r : ℝ) : Prop :=
  (x * r * 6 = 1) ∧ (12 * r * 4 = 1)

theorem determine_machines_in_first_group (x r : ℝ) (h : machines_in_first_group x r) :
  x = 8 :=
by
  sorry

end determine_machines_in_first_group_l116_116424


namespace distinct_exponentiation_values_l116_116296

theorem distinct_exponentiation_values : 
  ∃ (standard other1 other2 other3 : ℕ), 
    standard ≠ other1 ∧ 
    standard ≠ other2 ∧ 
    standard ≠ other3 ∧ 
    other1 ≠ other2 ∧ 
    other1 ≠ other3 ∧ 
    other2 ≠ other3 := 
sorry

end distinct_exponentiation_values_l116_116296


namespace how_many_more_rolls_needed_l116_116488

variable (total_needed sold_to_grandmother sold_to_uncle sold_to_neighbor : ℕ)

theorem how_many_more_rolls_needed (h1 : total_needed = 45)
  (h2 : sold_to_grandmother = 1)
  (h3 : sold_to_uncle = 10)
  (h4 : sold_to_neighbor = 6) :
  total_needed - (sold_to_grandmother + sold_to_uncle + sold_to_neighbor) = 28 := by
  sorry

end how_many_more_rolls_needed_l116_116488


namespace determine_n_l116_116483

theorem determine_n : ∃ n : ℤ, 0 ≤ n ∧ n < 8 ∧ -2222 % 8 = n := by
  use 2
  sorry

end determine_n_l116_116483


namespace inequality_to_prove_l116_116726

variable (x y z : ℝ)

axiom h1 : 0 ≤ x
axiom h2 : 0 ≤ y
axiom h3 : 0 ≤ z
axiom h4 : y * z + z * x + x * y = 1

theorem inequality_to_prove : x * (1 - y)^2 * (1 - z^2) + y * (1 - z^2) * (1 - x^2) + z * (1 - x^2) * (1 - y^2) ≤ (4 / 9) * Real.sqrt 3 :=
by 
  -- The proof is omitted.
  sorry

end inequality_to_prove_l116_116726


namespace list_price_of_article_l116_116374

theorem list_price_of_article 
(paid_price : ℝ) 
(first_discount second_discount : ℝ)
(list_price : ℝ)
(h_paid_price : paid_price = 59.22)
(h_first_discount : first_discount = 0.10)
(h_second_discount : second_discount = 0.06000000000000002)
(h_final_price : paid_price = (1 - first_discount) * (1 - second_discount) * list_price) :
  list_price = 70 := 
by
  sorry

end list_price_of_article_l116_116374


namespace A_worked_alone_after_B_left_l116_116009

/-- A and B can together finish a work in 40 days. They worked together for 10 days and then B left.
    A alone can finish the job in 80 days. We need to find out how many days did A work alone after B left. -/
theorem A_worked_alone_after_B_left
  (W : ℝ)
  (A_work_rate : ℝ := W / 80)
  (B_work_rate : ℝ := W / 80)
  (AB_work_rate : ℝ := W / 40)
  (work_done_together_in_10_days : ℝ := 10 * (W / 40))
  (remaining_work : ℝ := W - work_done_together_in_10_days)
  (A_rate_alone : ℝ := W / 80) :
  ∃ D : ℝ, D * (W / 80) = remaining_work → D = 60 :=
by
  sorry

end A_worked_alone_after_B_left_l116_116009


namespace kittens_given_to_Jessica_is_3_l116_116493

def kittens_initial := 18
def kittens_given_to_Sara := 6
def kittens_now := 9

def kittens_after_Sara := kittens_initial - kittens_given_to_Sara
def kittens_given_to_Jessica := kittens_after_Sara - kittens_now

theorem kittens_given_to_Jessica_is_3 : kittens_given_to_Jessica = 3 := by
  sorry

end kittens_given_to_Jessica_is_3_l116_116493


namespace find_total_coins_l116_116280

namespace PiratesTreasure

def total_initial_coins (m : ℤ) : Prop :=
  (m / 3 + 1) + (m / 4 + 5) + (m / 5 + 20) = m

theorem find_total_coins (m : ℤ) (h : total_initial_coins m) : m = 120 :=
  sorry

end PiratesTreasure

end find_total_coins_l116_116280


namespace Owen_spent_720_dollars_on_burgers_l116_116955

def days_in_June : ℕ := 30
def burgers_per_day : ℕ := 2
def cost_per_burger : ℕ := 12

def total_burgers (days : ℕ) (burgers_per_day : ℕ) : ℕ :=
  days * burgers_per_day

def total_cost (burgers : ℕ) (cost_per_burger : ℕ) : ℕ :=
  burgers * cost_per_burger

theorem Owen_spent_720_dollars_on_burgers :
  total_cost (total_burgers days_in_June burgers_per_day) cost_per_burger = 720 := by
  sorry

end Owen_spent_720_dollars_on_burgers_l116_116955


namespace hundredth_ring_square_count_l116_116264

-- Conditions
def center_rectangle : ℤ × ℤ := (1, 2)
def first_ring_square_count : ℕ := 10
def square_count_nth_ring (n : ℕ) : ℕ := 8 * n + 2

-- Problem Statement
theorem hundredth_ring_square_count : square_count_nth_ring 100 = 802 := 
  sorry

end hundredth_ring_square_count_l116_116264


namespace find_y_plus_inv_y_l116_116709

theorem find_y_plus_inv_y (y : ℝ) (h : y^3 + 1 / y^3 = 110) : y + 1 / y = 5 :=
sorry

end find_y_plus_inv_y_l116_116709


namespace distribution_y_value_l116_116831

theorem distribution_y_value :
  ∀ (x y : ℝ),
  (x + 0.1 + 0.3 + y = 1) →
  (7 * x + 8 * 0.1 + 9 * 0.3 + 10 * y = 8.9) →
  y = 0.4 :=
by
  intros x y h1 h2
  sorry

end distribution_y_value_l116_116831


namespace math_quiz_scores_stability_l116_116145

theorem math_quiz_scores_stability :
  let avgA := (90 + 82 + 88 + 96 + 94) / 5
  let avgB := (94 + 86 + 88 + 90 + 92) / 5
  let varA := ((90 - avgA) ^ 2 + (82 - avgA) ^ 2 + (88 - avgA) ^ 2 + (96 - avgA) ^ 2 + (94 - avgA) ^ 2) / 5
  let varB := ((94 - avgB) ^ 2 + (86 - avgB) ^ 2 + (88 - avgB) ^ 2 + (90 - avgB) ^ 2 + (92 - avgB) ^ 2) / 5
  avgA = avgB ∧ varB < varA :=
by
  sorry

end math_quiz_scores_stability_l116_116145


namespace opposite_numbers_l116_116301

theorem opposite_numbers (a b : ℤ) (h1 : -5^2 = a) (h2 : (-5)^2 = b) : a = -b :=
by sorry

end opposite_numbers_l116_116301


namespace employees_salaries_l116_116107

theorem employees_salaries (M N P : ℝ)
  (hM : M = 1.20 * N)
  (hN_median : N = N) -- Indicates N is the median
  (hP : P = 0.65 * M)
  (h_total : N + M + P = 3200) :
  M = 1288.58 ∧ N = 1073.82 ∧ P = 837.38 :=
by
  sorry

end employees_salaries_l116_116107


namespace correct_calculation_l116_116594

theorem correct_calculation (a : ℝ) : (3 * a^3)^2 = 9 * a^6 :=
by sorry

end correct_calculation_l116_116594


namespace bead_necklaces_sold_l116_116205

def cost_per_necklace : ℕ := 7
def total_earnings : ℕ := 70
def gemstone_necklaces_sold : ℕ := 7

theorem bead_necklaces_sold (B : ℕ) 
  (h1 : total_earnings = cost_per_necklace * (B + gemstone_necklaces_sold))  :
  B = 3 :=
by {
  sorry
}

end bead_necklaces_sold_l116_116205


namespace tan_difference_l116_116159

theorem tan_difference (α β : ℝ) (hα : Real.tan α = 5) (hβ : Real.tan β = 3) : 
    Real.tan (α - β) = 1 / 8 := 
by 
  sorry

end tan_difference_l116_116159


namespace sum_a_b_l116_116569

theorem sum_a_b (a b : ℚ) (h1 : 3 * a + 5 * b = 47) (h2 : 4 * a + 2 * b = 38) : a + b = 85 / 7 :=
by
  sorry

end sum_a_b_l116_116569


namespace no_such_natural_numbers_l116_116519

theorem no_such_natural_numbers :
  ¬ ∃ (x y : ℕ), (∃ (a b : ℕ), x^2 + y = a^2 ∧ x - y = b^2) := 
sorry

end no_such_natural_numbers_l116_116519


namespace john_newspaper_percentage_less_l116_116231

theorem john_newspaper_percentage_less
  (total_newspapers : ℕ)
  (selling_price : ℝ)
  (percentage_sold : ℝ)
  (profit : ℝ)
  (total_cost : ℝ)
  (cost_per_newspaper : ℝ)
  (percentage_less : ℝ)
  (h1 : total_newspapers = 500)
  (h2 : selling_price = 2)
  (h3 : percentage_sold = 0.80)
  (h4 : profit = 550)
  (h5 : total_cost = 800 - profit)
  (h6 : cost_per_newspaper = total_cost / total_newspapers)
  (h7 : percentage_less = ((selling_price - cost_per_newspaper) / selling_price) * 100) :
  percentage_less = 75 :=
by
  sorry

end john_newspaper_percentage_less_l116_116231


namespace rationalization_sum_l116_116952

noncomputable def rationalize_denominator : ℚ :=
  let A := 5
  let B := 49
  let C := 21
  A + B + C

theorem rationalization_sum : rationalize_denominator = 75 := by
  sorry

end rationalization_sum_l116_116952


namespace calculate_x_l116_116127

theorem calculate_x (a b x : ℕ) (h1 : b = 9) (h2 : b - a = 5) (h3 : a * b = 2 * (a + b) + x) : x = 10 :=
by
  sorry

end calculate_x_l116_116127


namespace area_of_figure_eq_two_l116_116886

theorem area_of_figure_eq_two :
  ∫ x in (1 / Real.exp 1)..(Real.exp 1), 1 / x = 2 :=
by sorry

end area_of_figure_eq_two_l116_116886


namespace coin_difference_l116_116497

theorem coin_difference (h : ∃ x y z : ℕ, 5*x + 10*y + 20*z = 40) : (∃ x : ℕ, 5*x = 40) → (∃ y : ℕ, 20*y = 40) → 8 - 2 = 6 :=
by
  intros h1 h2
  exact rfl

end coin_difference_l116_116497


namespace roots_equal_and_real_l116_116502

theorem roots_equal_and_real:
  (∀ (x : ℝ), x ^ 2 - 3 * x * y + y ^ 2 + 2 * x - 9 * y + 1 = 0 → (y = 0 ∨ y = -24 / 5)) ∧
  (∀ (x : ℝ), x ^ 2 - 3 * x * y + y ^ 2 + 2 * x - 9 * y + 1 = 0 → (y ≥ 0 ∨ y ≤ -24 / 5)) :=
  by sorry

end roots_equal_and_real_l116_116502


namespace stickers_initial_count_l116_116292

variable (initial : ℕ) (lost : ℕ)

theorem stickers_initial_count (lost_stickers : lost = 6) (remaining_stickers : initial - lost = 87) : initial = 93 :=
by {
  sorry
}

end stickers_initial_count_l116_116292


namespace maximize_A_plus_C_l116_116555

theorem maximize_A_plus_C (A B C D : ℕ) (h1 : A ≠ B) (h2 : A ≠ C) (h3 : A ≠ D)
 (h4 : B ≠ C) (h5 : B ≠ D) (h6 : C ≠ D) (hB : B = 2) (h7 : (A + C) % (B + D) = 0) 
 (h8 : A < 10) (h9 : B < 10) (h10 : C < 10) (h11 : D < 10) : 
 A + C ≤ 15 :=
sorry

end maximize_A_plus_C_l116_116555


namespace phone_answer_prob_within_four_rings_l116_116646

def prob_first_ring : ℚ := 1/10
def prob_second_ring : ℚ := 1/5
def prob_third_ring : ℚ := 3/10
def prob_fourth_ring : ℚ := 1/10

theorem phone_answer_prob_within_four_rings :
  prob_first_ring + prob_second_ring + prob_third_ring + prob_fourth_ring = 7/10 :=
by
  sorry

end phone_answer_prob_within_four_rings_l116_116646


namespace triangle_area_of_tangent_circles_l116_116495

/-- 
Given three circles with radii 1, 3, and 5, that are mutually externally tangent and all tangent to 
the same line, the area of the triangle determined by the points where each circle is tangent to the line 
is 6.
-/
theorem triangle_area_of_tangent_circles :
  let r1 := 1
  let r2 := 3
  let r3 := 5
  ∃ (A B C : ℝ × ℝ),
    A = (0, -(r1 : ℝ)) ∧ B = (0, -(r2 : ℝ)) ∧ C = (0, -(r3 : ℝ)) ∧
    (∃ (h : ℝ), ∃ (b : ℝ), h = 4 ∧ b = 3 ∧
    (1 / 2) * h * b = 6) := 
by
  sorry

end triangle_area_of_tangent_circles_l116_116495


namespace possible_measures_A_l116_116279

open Nat

theorem possible_measures_A :
  ∃ (A B : ℕ), A > 0 ∧ B > 0 ∧ (∃ k : ℕ, k > 0 ∧ A = k * B) ∧ A + B = 180 ∧ (∃! n : ℕ, n = 17) :=
by
  sorry

end possible_measures_A_l116_116279


namespace max_xy_l116_116357

theorem max_xy (x y : ℝ) (hxy_pos : x > 0 ∧ y > 0) (h : 5 * x + 8 * y = 65) : 
  xy ≤ 25 :=
by
  sorry

end max_xy_l116_116357


namespace painter_completion_time_l116_116515

def hours_elapsed (start_time end_time : String) : ℕ :=
  match (start_time, end_time) with
  | ("9:00 AM", "12:00 PM") => 3
  | _ => 0

-- The initial conditions, the start time is 9:00 AM, and 3 hours later 1/4th is done
def start_time := "9:00 AM"
def partial_completion_time := "12:00 PM"
def partial_completion_fraction := 1 / 4
def partial_time_hours := hours_elapsed start_time partial_completion_time

-- The painter works consistently, so it would take 4 times the partial time to complete the job
def total_time_hours := 4 * partial_time_hours

-- Calculate the completion time by adding total_time_hours to the start_time
def completion_time : String :=
  match start_time with
  | "9:00 AM" => "9:00 PM"
  | _         => "unknown"

theorem painter_completion_time :
  completion_time = "9:00 PM" :=
by
  -- Definitions and calculations already included in the setup
  sorry

end painter_completion_time_l116_116515


namespace tan_alpha_plus_405_deg_l116_116243

theorem tan_alpha_plus_405_deg (α : ℝ) (h : Real.tan (180 - α) = -4 / 3) : Real.tan (α + 405) = -7 := 
sorry

end tan_alpha_plus_405_deg_l116_116243


namespace total_number_of_red_and_white_jelly_beans_in_fishbowl_l116_116001

def number_of_red_jelly_beans_in_bag := 24
def number_of_white_jelly_beans_in_bag := 18
def number_of_bags := 3

theorem total_number_of_red_and_white_jelly_beans_in_fishbowl :
  number_of_red_jelly_beans_in_bag * number_of_bags + number_of_white_jelly_beans_in_bag * number_of_bags = 126 := by
  sorry

end total_number_of_red_and_white_jelly_beans_in_fishbowl_l116_116001


namespace directrix_of_parabola_l116_116527

theorem directrix_of_parabola (y : ℝ → ℝ) (h : ∀ x, y x = 4 * x^2 - 6) : 
    ∃ d, (∀ x, y x = 4 * x^2 - 6) ∧ d = -97/16 ↔ (y (-6 - d)) = -10 := 
    sorry

end directrix_of_parabola_l116_116527


namespace music_marks_l116_116711

variable (M : ℕ) -- Variable to represent marks in music

/-- Conditions -/
def science_marks : ℕ := 70
def social_studies_marks : ℕ := 85
def total_marks : ℕ := 275
def physics_marks : ℕ := M / 2

theorem music_marks :
  science_marks + M + social_studies_marks + physics_marks M = total_marks → M = 80 :=
by
  sorry

end music_marks_l116_116711


namespace infinite_primes_of_the_year_2022_l116_116715

theorem infinite_primes_of_the_year_2022 :
  ∃ᶠ p in Filter.atTop, ∃ n : ℕ, p % 2 = 1 ∧ p ^ 2022 ∣ n ^ 2022 + 2022 :=
sorry

end infinite_primes_of_the_year_2022_l116_116715


namespace find_multiple_l116_116797

theorem find_multiple :
  ∀ (total_questions correct_answers score : ℕ) (m : ℕ),
  total_questions = 100 →
  correct_answers = 90 →
  score = 70 →
  score = correct_answers - m * (total_questions - correct_answers) →
  m = 2 :=
by
  intros total_questions correct_answers score m h1 h2 h3 h4
  sorry

end find_multiple_l116_116797


namespace ernie_circles_l116_116339

theorem ernie_circles (boxes_per_circle_ali boxes_per_circle_ernie total_boxes ali_circles : ℕ)
  (h1: boxes_per_circle_ali = 8)
  (h2: boxes_per_circle_ernie = 10)
  (h3: total_boxes = 80)
  (h4: ali_circles = 5) : 
  (total_boxes - ali_circles * boxes_per_circle_ali) / boxes_per_circle_ernie = 4 :=
by
  sorry

end ernie_circles_l116_116339


namespace find_d_given_n_eq_cda_div_a_minus_d_l116_116480

theorem find_d_given_n_eq_cda_div_a_minus_d (a c d n : ℝ) (h : n = c * d * a / (a - d)) :
  d = n * a / (c * d + n) := 
by
  sorry

end find_d_given_n_eq_cda_div_a_minus_d_l116_116480


namespace relative_magnitude_of_reciprocal_l116_116496

theorem relative_magnitude_of_reciprocal 
  (a b : ℝ) (hab : a < 1 / b) :
  (a > 0 ∧ b > 0 ∧ 1 / a > b) ∨ (a < 0 ∧ b < 0 ∧ 1 / a > b)
   ∨ (a > 0 ∧ b < 0 ∧ 1 / a < b) ∨ (a < 0 ∧ b > 0 ∧ 1 / a < b) :=
by sorry

end relative_magnitude_of_reciprocal_l116_116496


namespace find_some_number_l116_116766

theorem find_some_number : 
  ∃ x : ℝ, 
  (6 + 9 * 8 / x - 25 = 5) ↔ (x = 3) :=
by 
  sorry

end find_some_number_l116_116766


namespace distance_between_stripes_l116_116737

theorem distance_between_stripes
  (h1 : ∀ (curbs_are_parallel : Prop), curbs_are_parallel → true)
  (h2 : ∀ (distance_between_curbs : ℝ), distance_between_curbs = 60 → true)
  (h3 : ∀ (length_of_curb : ℝ), length_of_curb = 20 → true)
  (h4 : ∀ (stripe_length : ℝ), stripe_length = 75 → true) :
  ∃ (d : ℝ), d = 16 :=
by
  sorry

end distance_between_stripes_l116_116737


namespace quadratic_inequality_l116_116659

theorem quadratic_inequality (m : ℝ) : (∃ x : ℝ, x^2 - 3*x - m = 0 ∧ (∃ y : ℝ, y^2 - 3*y - m = 0 ∧ x ≠ y)) ↔ m > - 9 / 4 := 
by
  sorry

end quadratic_inequality_l116_116659


namespace evaluate_polynomial_at_4_l116_116734

noncomputable def polynomial_horner (x : ℤ) : ℤ :=
  (((((3 * x + 6) * x - 20) * x - 8) * x + 15) * x + 9)

theorem evaluate_polynomial_at_4 :
  polynomial_horner 4 = 3269 :=
by
  sorry

end evaluate_polynomial_at_4_l116_116734


namespace proof_U_eq_A_union_complement_B_l116_116481

noncomputable def U : Set Nat := {1, 2, 3, 4, 5, 7}
noncomputable def A : Set Nat := {1, 3, 5, 7}
noncomputable def B : Set Nat := {3, 5}
noncomputable def complement_U_B := U \ B

theorem proof_U_eq_A_union_complement_B : U = A ∪ complement_U_B := by
  sorry

end proof_U_eq_A_union_complement_B_l116_116481


namespace winner_more_than_third_l116_116770

theorem winner_more_than_third (W S T F : ℕ) (h1 : F = 199) 
(h2 : W = F + 105) (h3 : W = S + 53) (h4 : W + S + T + F = 979) : 
W - T = 79 :=
by
  -- Here, the proof steps would go, but they are not required as per instructions.
  sorry

end winner_more_than_third_l116_116770


namespace largest_prime_mersenne_below_500_l116_116150

def is_mersenne (m : ℕ) (n : ℕ) := m = 2^n - 1
def is_power_of_2 (n : ℕ) := ∃ (k : ℕ), n = 2^k

theorem largest_prime_mersenne_below_500 : ∀ (m : ℕ), 
  m < 500 →
  (∃ n, is_power_of_2 n ∧ is_mersenne m n ∧ Nat.Prime m) →
  m ≤ 3 := 
by
  sorry

end largest_prime_mersenne_below_500_l116_116150


namespace right_triangle_count_l116_116138

theorem right_triangle_count :
  ∃! (a b : ℕ), (a^2 + b^2 = (b + 3)^2) ∧ (b < 50) :=
by
  sorry

end right_triangle_count_l116_116138


namespace percentage_decrease_in_area_l116_116445

noncomputable def original_radius (r : ℝ) : ℝ := r
noncomputable def new_radius (r : ℝ) : ℝ := 0.5 * r
noncomputable def original_area (r : ℝ) : ℝ := Real.pi * r ^ 2
noncomputable def new_area (r : ℝ) : ℝ := Real.pi * (0.5 * r) ^ 2

theorem percentage_decrease_in_area (r : ℝ) (hr : 0 ≤ r) :
  ((original_area r - new_area r) / original_area r) * 100 = 75 :=
by
  sorry

end percentage_decrease_in_area_l116_116445


namespace div_5_implies_one_div_5_l116_116971

theorem div_5_implies_one_div_5 (a b : ℕ) (h : 5 ∣ (a * b)) : (5 ∣ a) ∨ (5 ∣ b) :=
by 
  sorry

end div_5_implies_one_div_5_l116_116971


namespace solve_for_y_l116_116022

theorem solve_for_y : ∃ (y : ℚ), y + 2 - 2 / 3 = 4 * y - (y + 2) ∧ y = 5 / 3 :=
by
  sorry

end solve_for_y_l116_116022


namespace two_pow_gt_twice_n_plus_one_l116_116005

theorem two_pow_gt_twice_n_plus_one (n : ℕ) (h : n ≥ 3) : 2^n > 2 * n + 1 := 
sorry

end two_pow_gt_twice_n_plus_one_l116_116005


namespace smallest_odd_number_divisible_by_3_l116_116227

theorem smallest_odd_number_divisible_by_3 : ∃ n : ℕ, n = 3 ∧ ∀ m : ℕ, (m % 2 = 1 ∧ m % 3 = 0) → m ≥ n := 
by
  sorry

end smallest_odd_number_divisible_by_3_l116_116227


namespace days_per_week_equals_two_l116_116720

-- Definitions based on conditions
def hourly_rate : ℕ := 10
def hours_per_delivery : ℕ := 3
def total_weeks : ℕ := 6
def total_earnings : ℕ := 360

-- Proof statement: determine the number of days per week Jamie delivers flyers is 2
theorem days_per_week_equals_two (d : ℕ) :
  10 * (total_weeks * d * hours_per_delivery) = total_earnings → d = 2 := by
  sorry

end days_per_week_equals_two_l116_116720


namespace painting_time_equation_l116_116210

theorem painting_time_equation (t : ℝ) :
  (1/6 + 1/8) * (t - 2) = 1 :=
sorry

end painting_time_equation_l116_116210


namespace average_page_count_per_essay_l116_116840

-- Conditions
def numberOfStudents := 15
def pagesFirstFive := 5 * 2
def pagesNextFive := 5 * 3
def pagesLastFive := 5 * 1

-- Total pages
def totalPages := pagesFirstFive + pagesNextFive + pagesLastFive

-- Proof problem statement
theorem average_page_count_per_essay : totalPages / numberOfStudents = 2 := by
  sorry

end average_page_count_per_essay_l116_116840


namespace packs_of_red_bouncy_balls_l116_116755

/-- Given the following conditions:
1. Kate bought 6 packs of yellow bouncy balls.
2. Each pack contained 18 bouncy balls.
3. Kate bought 18 more red bouncy balls than yellow bouncy balls.
Prove that the number of packs of red bouncy balls Kate bought is 7. -/
theorem packs_of_red_bouncy_balls (packs_yellow : ℕ) (balls_per_pack : ℕ) (extra_red_balls : ℕ)
  (h1 : packs_yellow = 6)
  (h2 : balls_per_pack = 18)
  (h3 : extra_red_balls = 18)
  : (packs_yellow * balls_per_pack + extra_red_balls) / balls_per_pack = 7 :=
by
  sorry

end packs_of_red_bouncy_balls_l116_116755


namespace total_students_1150_l116_116589

theorem total_students_1150 (T G : ℝ) (h1 : 92 + G = T) (h2 : G = 0.92 * T) : T = 1150 := 
by
  sorry

end total_students_1150_l116_116589


namespace horse_saddle_ratio_l116_116369

variable (H S : ℝ)
variable (m : ℝ)
variable (total_value saddle_value : ℝ)

theorem horse_saddle_ratio :
  total_value = 100 ∧ saddle_value = 12.5 ∧ H = m * saddle_value ∧ H + saddle_value = total_value → m = 7 :=
by
  sorry

end horse_saddle_ratio_l116_116369


namespace abs_ineq_solution_l116_116657

theorem abs_ineq_solution (x : ℝ) : (2 ≤ |x - 5| ∧ |x - 5| ≤ 4) ↔ (1 ≤ x ∧ x ≤ 3) ∨ (7 ≤ x ∧ x ≤ 9) :=
by
  sorry

end abs_ineq_solution_l116_116657


namespace A_half_B_l116_116235

-- Define the arithmetic series sum function
def series_sum (n : ℕ) : ℕ := n * (n + 1) / 2

-- Define A and B according to the problem conditions
def A : ℕ := (Finset.range 2022).sum (λ m => series_sum (m + 1))

def B : ℕ := (Finset.range 2022).sum (λ m => (m + 1) * (m + 2))

-- The proof statement
theorem A_half_B : A = B / 2 :=
by
  sorry

end A_half_B_l116_116235


namespace solve_for_y_l116_116705

def G (a y c d : ℕ) := 3 ^ y + 6 * d

theorem solve_for_y (a c d : ℕ) (h1 : G a 2 c d = 735) : 2 = 2 := 
by
  sorry

end solve_for_y_l116_116705


namespace domain_of_h_l116_116449

noncomputable def h (x : ℝ) : ℝ := (x^4 - 5 * x + 6) / (|x - 4| + |x + 2| - 1)

theorem domain_of_h : ∀ x : ℝ, |x - 4| + |x + 2| - 1 ≠ 0 := by
  intro x
  sorry

end domain_of_h_l116_116449


namespace ram_ravi_selected_probability_l116_116759

noncomputable def probability_both_selected : ℝ := 
  let probability_ram_80 := (1 : ℝ) / 7
  let probability_ravi_80 := (1 : ℝ) / 5
  let probability_both_80 := probability_ram_80 * probability_ravi_80
  let num_applicants := 200
  let num_spots := 4
  let probability_single_selection := (num_spots : ℝ) / (num_applicants : ℝ)
  let probability_both_selected_given_80 := probability_single_selection * probability_single_selection
  probability_both_80 * probability_both_selected_given_80

theorem ram_ravi_selected_probability :
  probability_both_selected = 1 / 87500 := 
by
  sorry

end ram_ravi_selected_probability_l116_116759


namespace find_f1_find_f3_range_of_x_l116_116571

-- Define f as described
axiom f : ℝ → ℝ
axiom f_domain : ∀ (x : ℝ), x > 0 → ∃ (y : ℝ), f y = f x

-- Given conditions
axiom condition1 : ∀ (x₁ x₂ : ℝ), 0 < x₁ ∧ 0 < x₂ ∧ x₁ ≠ x₂ → (x₁ - x₂) * (f x₁ - f x₂) < 0
axiom condition2 : ∀ (x y : ℝ), 0 < x ∧ 0 < y → f (x * y) = f x + f y
axiom condition3 : f (1 / 3) = 1

-- Prove f(1) = 0
theorem find_f1 : f 1 = 0 := by sorry

-- Prove f(3) = -1
theorem find_f3 : f 3 = -1 := by sorry

-- Given inequality condition
axiom condition4 : ∀ x : ℝ, 0 < x → f x < 2 + f (2 - x)

-- Prove range of x for given inequality
theorem range_of_x : ∀ x, x > 1 / 5 ∧ x < 2 ↔ f x < 2 + f (2 - x) := by sorry

end find_f1_find_f3_range_of_x_l116_116571


namespace abs_sqrt2_sub_2_l116_116092

theorem abs_sqrt2_sub_2 (h : 1 < Real.sqrt 2 ∧ Real.sqrt 2 < 2) : |Real.sqrt 2 - 2| = 2 - Real.sqrt 2 :=
by
  sorry

end abs_sqrt2_sub_2_l116_116092


namespace triangle_inequality_l116_116776

theorem triangle_inequality (a b c : ℝ) (h1 : a + b > c) (h2 : b + c > a) (h3 : c + a > b) :
  (a^2 + 2 * b * c) / (b^2 + c^2) + (b^2 + 2 * a * c) / (c^2 + a^2) + (c^2 + 2 * a * b) / (a^2 + b^2) > 3 :=
by
  sorry

end triangle_inequality_l116_116776


namespace additional_people_needed_l116_116398

theorem additional_people_needed
  (initial_people : ℕ) (initial_time : ℕ) (new_time : ℕ)
  (h_initial : initial_people * initial_time = 24)
  (h_time : new_time = 2)
  (h_initial_people : initial_people = 8)
  (h_initial_time : initial_time = 3) :
  (24 / new_time) - initial_people = 4 :=
by
  sorry

end additional_people_needed_l116_116398


namespace minimum_toothpicks_removal_l116_116311

theorem minimum_toothpicks_removal
    (num_toothpicks : ℕ) 
    (num_triangles : ℕ) 
    (h1 : num_toothpicks = 40) 
    (h2 : num_triangles > 35) :
    ∃ (min_removal : ℕ), min_removal = 15 
    := 
    sorry

end minimum_toothpicks_removal_l116_116311


namespace most_lines_of_symmetry_circle_l116_116546

-- Define the figures and their lines of symmetry
def regular_pentagon_lines_of_symmetry : ℕ := 5
def isosceles_triangle_lines_of_symmetry : ℕ := 1
def circle_lines_of_symmetry : ℕ := 0  -- Representing infinite lines of symmetry in Lean is unconventional; we'll use a special case.
def regular_hexagon_lines_of_symmetry : ℕ := 6
def ellipse_lines_of_symmetry : ℕ := 2

-- Define a predicate to check if one figure has more lines of symmetry than all others
def most_lines_of_symmetry {α : Type} [LinearOrder α] (f : α) (others : List α) : Prop :=
  ∀ x ∈ others, f ≥ x

-- Define the problem statement in Lean
theorem most_lines_of_symmetry_circle :
  most_lines_of_symmetry circle_lines_of_symmetry [
    regular_pentagon_lines_of_symmetry,
    isosceles_triangle_lines_of_symmetry,
    regular_hexagon_lines_of_symmetry,
    ellipse_lines_of_symmetry ] :=
by {
  -- To represent infinite lines, we consider 0 as a larger "dummy" number in this context,
  -- since in Lean we don't have a built-in representation for infinity in finite ordering.
  -- Replace with a suitable model if necessary.
  sorry
}

end most_lines_of_symmetry_circle_l116_116546


namespace apples_harvested_from_garden_l116_116869

def number_of_pies : ℕ := 10
def apples_per_pie : ℕ := 8
def apples_to_buy : ℕ := 30

def total_apples_needed : ℕ := number_of_pies * apples_per_pie

theorem apples_harvested_from_garden : total_apples_needed - apples_to_buy = 50 :=
by
  sorry

end apples_harvested_from_garden_l116_116869


namespace largest_three_digit_number_l116_116684

theorem largest_three_digit_number :
  ∃ n k m : ℤ, 100 ≤ n ∧ n < 1000 ∧ n = 7 * k + 2 ∧ n = 4 * m + 1 ∧ n = 989 :=
by
  sorry

end largest_three_digit_number_l116_116684


namespace avg_weight_a_b_l116_116438

theorem avg_weight_a_b (A B C : ℝ) 
  (h1 : (A + B + C) / 3 = 60)
  (h2 : (B + C) / 2 = 50)
  (h3 : B = 60) :
  (A + B) / 2 = 70 := 
sorry

end avg_weight_a_b_l116_116438


namespace brick_length_proof_l116_116364

-- Defining relevant parameters and conditions
def width_of_brick : ℝ := 10 -- width in cm
def height_of_brick : ℝ := 7.5 -- height in cm
def wall_length : ℝ := 26 -- length in m
def wall_width : ℝ := 2 -- width in m
def wall_height : ℝ := 0.75 -- height in m
def num_bricks : ℝ := 26000 

-- Defining known volumes for conversion
def volume_of_wall_m3 : ℝ := wall_length * wall_width * wall_height
def volume_of_wall_cm3 : ℝ := volume_of_wall_m3 * 1000000 -- converting m³ to cm³

-- Volume of one brick given the unknown length L
def volume_of_one_brick (L : ℝ) : ℝ := L * width_of_brick * height_of_brick

-- Total volume of bricks is the volume of one brick times the number of bricks
def total_volume_of_bricks (L : ℝ) : ℝ := volume_of_one_brick L * num_bricks

-- The length of the brick is found by equating the total volume of bricks to the volume of the wall
theorem brick_length_proof : ∃ L : ℝ, total_volume_of_bricks L = volume_of_wall_cm3 ∧ L = 20 :=
by
  existsi 20
  sorry

end brick_length_proof_l116_116364


namespace optometrist_sales_l116_116221

noncomputable def total_pairs_optometrist_sold (H S : ℕ) (total_sales: ℝ) : Prop :=
  (S = H + 7) ∧ 
  (total_sales = 0.9 * (95 * ↑H + 175 * ↑S)) ∧ 
  (total_sales = 2469)

theorem optometrist_sales :
  ∃ H S : ℕ, total_pairs_optometrist_sold H S 2469 ∧ H + S = 17 :=
by 
  sorry

end optometrist_sales_l116_116221


namespace find_a_conditions_l116_116170

theorem find_a_conditions (a : ℝ) : 
    (∃ m : ℤ, a = m + 1/2) ∨ (∃ m : ℤ, a = m + 1/3) ∨ (∃ m : ℤ, a = m - 1/3) ↔ 
    (∃ n : ℤ, a = n + 1/2 ∨ a = n + 1/3 ∨ a = n - 1/3) :=
by
  sorry

end find_a_conditions_l116_116170


namespace find_f_zero_forall_x_f_pos_solve_inequality_l116_116349

variable {f : ℝ → ℝ}

-- Conditions
axiom condition_1 : ∀ x, x > 0 → f x > 1
axiom condition_2 : ∀ x y, f (x + y) = f x * f y
axiom condition_3 : f 2 = 3

-- Questions rewritten as Lean theorems

theorem find_f_zero : f 0 = 1 := sorry

theorem forall_x_f_pos : ∀ x, f x > 0 := sorry

theorem solve_inequality : ∀ x, f (7 + 2 * x) > 9 ↔ x > -3 / 2 := sorry

end find_f_zero_forall_x_f_pos_solve_inequality_l116_116349


namespace arithmetic_problem_l116_116172

theorem arithmetic_problem :
  12.1212 + 17.0005 - 9.1103 = 20.0114 :=
sorry

end arithmetic_problem_l116_116172


namespace tank_capacity_l116_116916

theorem tank_capacity (C : ℝ) :
  (C / 10 - 960 = C / 18) → C = 21600 := by
  intro h
  sorry

end tank_capacity_l116_116916


namespace sum_of_two_pos_implies_one_pos_l116_116603

theorem sum_of_two_pos_implies_one_pos (x y : ℝ) (h : x + y > 0) : x > 0 ∨ y > 0 :=
  sorry

end sum_of_two_pos_implies_one_pos_l116_116603


namespace shopper_saved_percentage_l116_116215

theorem shopper_saved_percentage (amount_paid : ℝ) (amount_saved : ℝ) (original_price : ℝ)
  (h1 : amount_paid = 45) (h2 : amount_saved = 5) (h3 : original_price = amount_paid + amount_saved) :
  (amount_saved / original_price) * 100 = 10 :=
by
  -- The proof is omitted
  sorry

end shopper_saved_percentage_l116_116215


namespace negation_of_forall_ge_implies_exists_lt_l116_116843

theorem negation_of_forall_ge_implies_exists_lt :
  ¬(∀ x : ℝ, x^2 + 1 ≥ 2 * x) ↔ ∃ x : ℝ, x^2 + 1 < 2 * x := by
  sorry

end negation_of_forall_ge_implies_exists_lt_l116_116843


namespace gcd_ab_eq_one_l116_116122

def a : ℕ := 97^10 + 1
def b : ℕ := 97^10 + 97^3 + 1

theorem gcd_ab_eq_one : Nat.gcd a b = 1 :=
by
  sorry

end gcd_ab_eq_one_l116_116122


namespace intersection_P_Q_l116_116278

def P : Set ℝ := {-1, 0, 1}
def Q : Set ℝ := {y | ∃ x : ℝ, y = Real.cos x}

theorem intersection_P_Q :
  P ∩ Q = {-1, 0, 1} :=
sorry

end intersection_P_Q_l116_116278


namespace cat_food_weight_l116_116983

theorem cat_food_weight (x : ℝ) :
  let bags_of_cat_food := 2
  let bags_of_dog_food := 2
  let ounces_per_pound := 16
  let total_ounces_of_pet_food := 256
  let dog_food_extra_weight := 2
  (ounces_per_pound * (bags_of_cat_food * x + bags_of_dog_food * (x + dog_food_extra_weight))) = total_ounces_of_pet_food
  → x = 3 :=
by
  sorry

end cat_food_weight_l116_116983


namespace total_bananas_bought_l116_116131

-- Define the conditions
def went_to_store_times : ℕ := 2
def bananas_per_trip : ℕ := 10

-- State the theorem/question and provide the answer
theorem total_bananas_bought : (went_to_store_times * bananas_per_trip) = 20 :=
by
  -- Proof here
  sorry

end total_bananas_bought_l116_116131


namespace correct_conclusions_l116_116981

theorem correct_conclusions :
  (∀ n : ℤ, n < -1 -> n < -1) ∧
  (¬ ∀ a : ℤ, abs (a + 2022) > 0) ∧
  (∀ a b : ℤ, a + b = 0 -> a * b < 0) ∧
  (∀ n : ℤ, abs n = n -> n ≥ 0) :=
sorry

end correct_conclusions_l116_116981


namespace bronchitis_option_D_correct_l116_116898

noncomputable def smoking_related_to_bronchitis : Prop :=
  -- Conclusion that "smoking is related to chronic bronchitis"
sorry

noncomputable def confidence_level : ℝ :=
  -- Confidence level in the conclusion
  0.99

theorem bronchitis_option_D_correct :
  smoking_related_to_bronchitis →
  (confidence_level > 0.99) →
  -- Option D is correct: "Among 100 smokers, it is possible that not a single person has chronic bronchitis"
  ∃ (P : ℕ → Prop), (∀ n : ℕ, n ≤ 100 → P n = False) :=
by sorry

end bronchitis_option_D_correct_l116_116898


namespace sum_of_consecutive_integers_of_sqrt3_l116_116086

theorem sum_of_consecutive_integers_of_sqrt3 {a b : ℤ} (h1 : a + 1 = b) (h2 : (a : ℝ) < Real.sqrt 3) (h3 : Real.sqrt 3 < (b : ℝ)) :
  a + b = 3 := by
  sorry

end sum_of_consecutive_integers_of_sqrt3_l116_116086


namespace find_xyz_l116_116305

theorem find_xyz (x y z : ℝ) 
  (h1: 3 * x - y + z = 8)
  (h2: x + 3 * y - z = 2) 
  (h3: x - y + 3 * z = 6) :
  x = 1 ∧ y = 3 ∧ z = 8 := by
  sorry

end find_xyz_l116_116305


namespace f_of_5_eq_1_l116_116708

noncomputable def f : ℝ → ℝ := sorry

theorem f_of_5_eq_1
    (h1 : ∀ x : ℝ, f (-x) = -f x)
    (h2 : ∀ x : ℝ, f (-x) + f (x + 3) = 0)
    (h3 : f (-1) = 1) :
    f 5 = 1 :=
sorry

end f_of_5_eq_1_l116_116708


namespace trigonometric_identity_proof_l116_116486

theorem trigonometric_identity_proof
  (α : Real)
  (h1 : Real.sin (Real.pi + α) = -Real.sin α)
  (h2 : Real.cos (Real.pi + α) = -Real.cos α)
  (h3 : Real.cos (-α) = Real.cos α)
  (h4 : Real.sin α ^ 2 + Real.cos α ^ 2 = 1) :
  Real.sin (Real.pi + α) ^ 2 - Real.cos (Real.pi + α) * Real.cos (-α) + 1 = 2 := 
by
  sorry

end trigonometric_identity_proof_l116_116486


namespace money_made_march_to_august_l116_116810

section
variable (H : ℕ)

-- Given conditions
def hoursMarchToAugust : ℕ := 23
def hoursSeptToFeb : ℕ := 8
def additionalHours : ℕ := 16
def totalCost : ℕ := 600 + 340
def totalHours : ℕ := hoursMarchToAugust + hoursSeptToFeb + additionalHours

-- Total money equation
def totalMoney : ℕ := totalHours * H

-- Theorem to prove the money made from March to August
theorem money_made_march_to_august : totalMoney = totalCost → hoursMarchToAugust * H = 460 :=
by
  intro h
  have hH : H = 20 := by
    sorry
  rw [hH]
  sorry
end

end money_made_march_to_august_l116_116810


namespace blackjack_payment_l116_116319

def casino_payout (b: ℤ) (r: ℤ): ℤ := b + r
def blackjack_payout (bet: ℤ) (ratio_numerator: ℤ) (ratio_denominator: ℤ): ℤ :=
  (ratio_numerator * bet) / ratio_denominator

theorem blackjack_payment (bet: ℤ) (ratio_numerator: ℤ) (ratio_denominator: ℤ) (payout: ℤ):
  ratio_numerator = 3 → 
  ratio_denominator = 2 → 
  bet = 40 →
  payout = blackjack_payout bet ratio_numerator ratio_denominator → 
  casino_payout bet payout = 100 :=
by
  sorry

end blackjack_payment_l116_116319


namespace difference_in_probabilities_is_twenty_percent_l116_116529

-- Definition of the problem conditions
def prob_win_first_lawsuit : ℝ := 0.30
def prob_lose_first_lawsuit : ℝ := 0.70
def prob_win_second_lawsuit : ℝ := 0.50
def prob_lose_second_lawsuit : ℝ := 0.50

-- We need to prove that the difference in probability of losing both lawsuits and winning both lawsuits is 20%
theorem difference_in_probabilities_is_twenty_percent :
  (prob_lose_first_lawsuit * prob_lose_second_lawsuit) -
  (prob_win_first_lawsuit * prob_win_second_lawsuit) = 0.20 := 
by
  sorry

end difference_in_probabilities_is_twenty_percent_l116_116529


namespace principal_amount_l116_116677

theorem principal_amount (P : ℝ) (R : ℝ) (T : ℝ) (SI : ℝ) 
  (h1 : R = 4) 
  (h2 : T = 5) 
  (h3 : SI = P - 1920) 
  (h4 : SI = (P * R * T) / 100) : 
  P = 2400 := 
by 
  sorry

end principal_amount_l116_116677


namespace side_length_of_base_l116_116826

-- Given conditions
def lateral_face_area := 90 -- Area of one lateral face in square meters
def slant_height := 20 -- Slant height in meters

-- The theorem statement
theorem side_length_of_base 
  (s : ℝ)
  (h : ℝ := slant_height)
  (a : ℝ := lateral_face_area)
  (h_area : 2 * a = s * h) :
  s = 9 := 
sorry

end side_length_of_base_l116_116826


namespace train_cross_time_l116_116695

def length_of_train : ℕ := 120 -- the train is 120 m long
def speed_of_train_km_hr : ℕ := 45 -- the train's speed in km/hr
def length_of_bridge : ℕ := 255 -- the bridge is 255 m long

def train_speed_m_s : ℕ := speed_of_train_km_hr * (1000 / 3600)

def total_distance : ℕ := length_of_train + length_of_bridge

def time_to_cross_bridge (distance : ℕ) (speed : ℕ) : ℕ :=
  distance / speed

theorem train_cross_time :
  time_to_cross_bridge total_distance train_speed_m_s = 30 :=
by
  sorry

end train_cross_time_l116_116695


namespace car_speed_l116_116581

theorem car_speed (rev_per_min : ℕ) (circ : ℝ) (h_rev : rev_per_min = 400) (h_circ : circ = 5) : 
  (rev_per_min * circ) * 60 / 1000 = 120 :=
by
  sorry

end car_speed_l116_116581


namespace islander_C_response_l116_116351

-- Define the types and assumptions
variables {Person : Type} (is_knight : Person → Prop) (is_liar : Person → Prop)
variables (A B C : Person)

-- Conditions from the problem
axiom A_statement : (is_liar A) ↔ (is_knight B = false ∧ is_knight C = false)
axiom B_statement : (is_knight B) ↔ (is_knight A ↔ ¬ is_knight C)

-- Conclusion we want to prove
theorem islander_C_response : is_knight C → (is_knight A ↔ ¬ is_knight C) := sorry

end islander_C_response_l116_116351


namespace possible_roots_l116_116110

theorem possible_roots (a b p q : ℤ)
  (h1 : a ≠ 0)
  (h2 : b ≠ 0)
  (h3 : a ≠ b)
  (h4 : p = -(a + b))
  (h5 : q = ab)
  (h6 : (a + p) % (q - 2 * b) = 0) :
  a = 1 ∨ a = 3 :=
  sorry

end possible_roots_l116_116110


namespace attendees_proportion_l116_116242

def attendees (t k : ℕ) := k / t

theorem attendees_proportion (n t new_t : ℕ) (h1 : n * t = 15000) (h2 : t = 50) (h3 : new_t = 75) : attendees new_t 15000 = 200 :=
by
  -- Proof omitted, main goal is to assert equivalency
  sorry

end attendees_proportion_l116_116242


namespace ab_value_l116_116420

theorem ab_value (a b : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b = 30) (h4 : 2 * a * b + 12 * a = 3 * b + 240) :
  a * b = 255 :=
sorry

end ab_value_l116_116420


namespace inequality_solution_set_l116_116328

def is_even (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def is_monotonically_decreasing_on_nonneg (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 ≤ x → 0 ≤ y → x ≤ y → f y ≤ f x

theorem inequality_solution_set
  (f : ℝ → ℝ)
  (h_even : is_even f)
  (h_mono_dec : is_monotonically_decreasing_on_nonneg f) :
  { x : ℝ | f 1 - f (1 / x) < 0 } = { x : ℝ | x < -1 ∨ x > 1 } :=
by
  sorry

end inequality_solution_set_l116_116328


namespace intersection_complement_l116_116813

open Set

def UniversalSet := ℝ
def M : Set ℝ := {x | x^2 > 4}
def N : Set ℝ := {x | 1 < x ∧ x ≤ 3}
def CU_M : Set ℝ := compl M

theorem intersection_complement :
  N ∩ CU_M = {x | 1 < x ∧ x ≤ 2} :=
by sorry

end intersection_complement_l116_116813


namespace doubled_dimensions_new_volume_l116_116616

-- Define the original volume condition
def original_volume_condition (π r h : ℝ) : Prop := π * r^2 * h = 5

-- Define the new volume function after dimensions are doubled
def new_volume (π r h : ℝ) : ℝ := π * (2 * r)^2 * (2 * h)

-- The Lean statement for the proof problem 
theorem doubled_dimensions_new_volume (π r h : ℝ) (h_orig : original_volume_condition π r h) : 
  new_volume π r h = 40 :=
by 
  sorry

end doubled_dimensions_new_volume_l116_116616


namespace star_7_3_l116_116817

def star (a b : ℤ) : ℤ := 4 * a + 3 * b - a * b

theorem star_7_3 : star 7 3 = 16 := 
by 
  sorry

end star_7_3_l116_116817


namespace sum_arithmetic_sequence_l116_116933

def first_term (k : ℕ) : ℕ := k^2 - k + 1

def sum_of_first_k_plus_3_terms (k : ℕ) : ℕ := (k + 3) * (k^2 + (k / 2) + 2)

theorem sum_arithmetic_sequence (k : ℕ) (k_pos : 0 < k) : 
    sum_of_first_k_plus_3_terms k = k^3 + (7 * k^2) / 2 + (15 * k) / 2 + 6 := 
by
  sorry

end sum_arithmetic_sequence_l116_116933


namespace temperature_difference_l116_116834

theorem temperature_difference (initial_temp rise fall : ℤ) (h1 : initial_temp = 25)
    (h2 : rise = 3) (h3 : fall = 15) : initial_temp + rise - fall = 13 := by
  rw [h1, h2, h3]
  norm_num

end temperature_difference_l116_116834


namespace number_of_girls_l116_116784

variable (boys : ℕ) (total_children : ℕ)

theorem number_of_girls (h1 : boys = 40) (h2 : total_children = 117) : total_children - boys = 77 :=
by
  sorry

end number_of_girls_l116_116784


namespace p_is_sufficient_but_not_necessary_for_q_l116_116885

variable (x : ℝ)

def p := x > 1
def q := x > 0

theorem p_is_sufficient_but_not_necessary_for_q : (p x → q x) ∧ ¬(q x → p x) := by
  sorry

end p_is_sufficient_but_not_necessary_for_q_l116_116885


namespace hyperbola_range_m_l116_116505

theorem hyperbola_range_m (m : ℝ) : 
  (∃ x y : ℝ, (m - 2) ≠ 0 ∧ (m + 3) ≠ 0 ∧ (x^2 / (m - 2) + y^2 / (m + 3) = 1)) ↔ (-3 < m ∧ m < 2) :=
by
  sorry

end hyperbola_range_m_l116_116505


namespace marbles_lost_l116_116565

theorem marbles_lost (initial_marbs remaining_marbs marbles_lost : ℕ)
  (h1 : initial_marbs = 38)
  (h2 : remaining_marbs = 23)
  : marbles_lost = initial_marbs - remaining_marbs :=
by
  sorry

end marbles_lost_l116_116565


namespace correct_calculation_l116_116442

theorem correct_calculation (x : ℤ) (h1 : x + 65 = 125) : x + 95 = 155 :=
by sorry

end correct_calculation_l116_116442


namespace student_calculation_no_error_l116_116254

theorem student_calculation_no_error :
  let correct_result : ℚ := (7 * 4) / (5 / 3)
  let student_result : ℚ := (7 * 4) * (3 / 5)
  correct_result = student_result → 0 = 0 := 
by
  intros correct_result student_result h
  sorry

end student_calculation_no_error_l116_116254


namespace arithmetic_seq_a11_l116_116919

variable (a : ℕ → ℤ)
variable (d : ℕ → ℤ)

-- Conditions
def arithmetic_sequence : Prop := ∀ n, a (n + 2) - a n = 6
def a1 : Prop := a 1 = 1

-- Statement of the problem
theorem arithmetic_seq_a11 : arithmetic_sequence a ∧ a1 a → a 11 = 31 :=
by sorry

end arithmetic_seq_a11_l116_116919


namespace charlie_steps_proof_l116_116284

-- Define the conditions
def Steps_Charlie_3km : ℕ := 5350
def Laps : ℚ := 2.5

-- Define the total steps Charlie can make in 2.5 laps
def Steps_Charlie_total : ℕ := 13375

-- The statement to prove
theorem charlie_steps_proof : Laps * Steps_Charlie_3km = Steps_Charlie_total :=
by
  sorry

end charlie_steps_proof_l116_116284


namespace min_value_expression_l116_116006

theorem min_value_expression (α β : ℝ) : 
  (3 * Real.cos α + 4 * Real.sin β - 7) ^ 2 + (3 * Real.sin α + 4 * Real.cos β - 12) ^ 2 ≥ 36 :=
sorry

end min_value_expression_l116_116006


namespace scout_troop_profit_l116_116418

-- Defining the basic conditions as Lean definitions
def num_bars : ℕ := 1500
def cost_rate : ℚ := 3 / 4 -- rate in dollars per bar
def sell_rate : ℚ := 2 / 3 -- rate in dollars per bar

-- Calculate total cost, total revenue, and profit
def total_cost : ℚ := num_bars * cost_rate
def total_revenue : ℚ := num_bars * sell_rate
def profit : ℚ := total_revenue - total_cost

-- The final theorem to be proved
theorem scout_troop_profit : profit = -125 := by
  sorry

end scout_troop_profit_l116_116418


namespace linear_dependency_k_val_l116_116379

theorem linear_dependency_k_val (k : ℝ) :
  (∃ (c1 c2 : ℝ), (c1 ≠ 0 ∨ c2 ≠ 0) ∧ 2 * c1 + 4 * c2 = 0 ∧ 3 * c1 + k * c2 = 0) ↔ k = 6 :=
by sorry

end linear_dependency_k_val_l116_116379


namespace elizabeth_net_profit_l116_116434

theorem elizabeth_net_profit :
  let cost_per_bag := 3.00
  let num_bags := 20
  let price_first_15_bags := 6.00
  let price_last_5_bags := 4.00
  let total_cost := cost_per_bag * num_bags
  let revenue_first_15 := 15 * price_first_15_bags
  let revenue_last_5 := 5 * price_last_5_bags
  let total_revenue := revenue_first_15 + revenue_last_5
  let net_profit := total_revenue - total_cost
  net_profit = 50.00 :=
by
  sorry

end elizabeth_net_profit_l116_116434


namespace sara_total_score_l116_116120

-- Definitions based on the conditions
def correct_points (correct_answers : Nat) : Int := correct_answers * 2
def incorrect_points (incorrect_answers : Nat) : Int := incorrect_answers * (-1)
def unanswered_points (unanswered_questions : Nat) : Int := unanswered_questions * 0

def total_score (correct_answers incorrect_answers unanswered_questions : Nat) : Int :=
  correct_points correct_answers + incorrect_points incorrect_answers + unanswered_points unanswered_questions

-- The main theorem stating the problem requirement
theorem sara_total_score :
  total_score 18 10 2 = 26 :=
by
  sorry

end sara_total_score_l116_116120


namespace ratio_of_numbers_l116_116149

theorem ratio_of_numbers (a b : ℝ) (h1 : 0 < b) (h2 : b < a) 
  (h3 : 2 * ((a + b) / 2) = Real.sqrt (10 * a * b)) : abs (a / b - 8) < 1 :=
by
  sorry

end ratio_of_numbers_l116_116149


namespace water_added_to_mixture_is_11_l116_116175

noncomputable def initial_mixture_volume : ℕ := 45
noncomputable def initial_milk_ratio : ℚ := 4
noncomputable def initial_water_ratio : ℚ := 1
noncomputable def final_milk_ratio : ℚ := 9
noncomputable def final_water_ratio : ℚ := 5

theorem water_added_to_mixture_is_11 :
  ∃ x : ℚ, (initial_milk_ratio * initial_mixture_volume / 
            (initial_water_ratio * initial_mixture_volume + x)) = (final_milk_ratio / final_water_ratio)
  ∧ x = 11 :=
by
  -- Proof here
  sorry

end water_added_to_mixture_is_11_l116_116175


namespace find_m_such_that_no_linear_term_in_expansion_l116_116706

theorem find_m_such_that_no_linear_term_in_expansion :
  ∃ m : ℝ, ∀ x : ℝ, (x^2 - x + m) * (x - 8) = x^3 - 9 * x^2 - 8 * m ∧ ((8 + m) = 0) :=
by
  sorry

end find_m_such_that_no_linear_term_in_expansion_l116_116706


namespace no_exp_function_satisfies_P_d_decreasing_nonnegative_exists_c_infinitely_many_l116_116194

-- Define the context for real numbers and the main property P.
def property_P (f : ℝ → ℝ) : Prop :=
∀ x : ℝ, f x + f (x + 2) ≤ 2 * f (x + 1)

-- For part (1)
theorem no_exp_function_satisfies_P (a : ℝ) (h₁ : 0 < a) (h₂ : a ≠ 1) :
  ¬ ∃ (f : ℝ → ℝ), (∀ x : ℝ, f x = a^x) ∧ property_P f :=
sorry

-- Define the context for natural numbers, d(x), and main properties related to P.
def d (f : ℕ → ℕ) (x : ℕ) : ℕ := f (x + 1) - f x

-- For part (2)(i)
theorem d_decreasing_nonnegative (f : ℕ → ℕ) (P : ∀ x : ℕ, f x + f (x + 2) ≤ 2 * f (x + 1)) :
  ∀ x : ℕ, d f (x + 1) ≤ d f x ∧ d f x ≥ 0 :=
sorry

-- For part (2)(ii)
theorem exists_c_infinitely_many (f : ℕ → ℕ) (P : ∀ x : ℕ, f x + f (x + 2) ≤ 2 * f (x + 1)) :
  ∃ c : ℕ, 0 ≤ c ∧ c ≤ d f 1 ∧ ∀ N : ℕ, ∃ n : ℕ, n > N ∧ d f n = c :=
sorry

end no_exp_function_satisfies_P_d_decreasing_nonnegative_exists_c_infinitely_many_l116_116194


namespace ferry_speed_difference_l116_116294

theorem ferry_speed_difference :
  let V_p := 6
  let Time_P := 3
  let Distance_P := V_p * Time_P
  let Distance_Q := 2 * Distance_P
  let Time_Q := Time_P + 1
  let V_q := Distance_Q / Time_Q
  V_q - V_p = 3 := by
  sorry

end ferry_speed_difference_l116_116294


namespace total_students_l116_116249

theorem total_students (N : ℕ) (num_provincial : ℕ) (sample_provincial : ℕ) 
(sample_experimental : ℕ) (sample_regular : ℕ) (sample_sino_canadian : ℕ) 
(ratio : ℕ) 
(h1 : num_provincial = 96) 
(h2 : sample_provincial = 12) 
(h3 : sample_experimental = 21) 
(h4 : sample_regular = 25) 
(h5 : sample_sino_canadian = 43) 
(h6 : ratio = num_provincial / sample_provincial) 
(h7 : ratio = 8) 
: N = ratio * (sample_provincial + sample_experimental + sample_regular + sample_sino_canadian) := 
by 
  sorry

end total_students_l116_116249


namespace solve_for_star_l116_116752

theorem solve_for_star : ∀ (star : ℝ), (45 - (28 - (37 - (15 - star))) = 54) → star = 15 := by
  intros star h
  sorry

end solve_for_star_l116_116752


namespace cupcakes_sold_l116_116052

theorem cupcakes_sold (initial additional final sold : ℕ) (h1 : initial = 14) (h2 : additional = 17) (h3 : final = 25) :
  initial + additional - final = sold :=
by
  sorry

end cupcakes_sold_l116_116052


namespace base_number_min_sum_l116_116748

theorem base_number_min_sum (a b : ℕ) (h₁ : 5 * a + 2 = 2 * b + 5) : a + b = 9 :=
by {
  -- this proof is skipped with sorry
  sorry
}

end base_number_min_sum_l116_116748


namespace calculate_moment_of_inertia_l116_116510

noncomputable def moment_of_inertia (a ρ₀ k : ℝ) : ℝ :=
  8 * (a ^ (9/2)) * ((ρ₀ / 7) + (k * a / 9))

theorem calculate_moment_of_inertia (a ρ₀ k : ℝ) 
  (h₀ : 0 ≤ a) :
  moment_of_inertia a ρ₀ k = 8 * a ^ (9/2) * ((ρ₀ / 7) + (k * a / 9)) :=
sorry

end calculate_moment_of_inertia_l116_116510


namespace diane_15_cents_arrangement_l116_116030

def stamps : List (ℕ × ℕ) := 
  [(1, 1), 
   (2, 2), 
   (3, 3), 
   (4, 4), 
   (5, 5), 
   (6, 6), 
   (7, 7), 
   (8, 8), 
   (9, 9), 
   (10, 10), 
   (11, 11), 
   (12, 12)]

def number_of_arrangements (value : ℕ) (stamps : List (ℕ × ℕ)) : ℕ := sorry

theorem diane_15_cents_arrangement : number_of_arrangements 15 stamps = 32 := 
sorry

end diane_15_cents_arrangement_l116_116030


namespace possible_values_of_a2b_b2c_c2a_l116_116965

theorem possible_values_of_a2b_b2c_c2a (a b c : ℝ) (h : a + b + c = 1) : ∀ x : ℝ, ∃ a b c : ℝ, a + b + c = 1 ∧ a^2 * b + b^2 * c + c^2 * a = x :=
by
  sorry

end possible_values_of_a2b_b2c_c2a_l116_116965


namespace Isabel_afternoon_runs_l116_116551

theorem Isabel_afternoon_runs (circuit_length morning_runs weekly_distance afternoon_runs : ℕ)
  (h_circuit_length : circuit_length = 365)
  (h_morning_runs : morning_runs = 7)
  (h_weekly_distance : weekly_distance = 25550)
  (h_afternoon_runs : weekly_distance = morning_runs * circuit_length * 7 + afternoon_runs * circuit_length) :
  afternoon_runs = 21 :=
by
  -- The actual proof goes here
  sorry

end Isabel_afternoon_runs_l116_116551


namespace trigonometric_identity_proof_l116_116685

theorem trigonometric_identity_proof (α : ℝ) (h : Real.cos (π / 6 - α) = Real.sqrt 3 / 3) :
  Real.cos (5 * π / 6 + α) - Real.sin (α - π / 6) ^ 2 = - (Real.sqrt 3 + 2) / 3 :=
by
  sorry

end trigonometric_identity_proof_l116_116685


namespace fx_leq_one_l116_116144

noncomputable def f (x : ℝ) : ℝ := (x + 1) / Real.exp x

theorem fx_leq_one : ∀ x : ℝ, f x ≤ 1 := by
  sorry

end fx_leq_one_l116_116144


namespace suitable_for_systematic_sampling_l116_116815

-- Define the given conditions as a structure
structure SamplingProblem where
  option_A : String
  option_B : String
  option_C : String
  option_D : String

-- Define the equivalence theorem to prove Option C is the most suitable
theorem suitable_for_systematic_sampling (p : SamplingProblem) 
(hA: p.option_A = "Randomly selecting 8 students from a class of 48 students to participate in an activity")
(hB: p.option_B = "A city has 210 department stores, including 20 large stores, 40 medium stores, and 150 small stores. To understand the business situation of each store, a sample of 21 stores needs to be drawn")
(hC: p.option_C = "Randomly selecting 100 candidates from 1200 exam participants to analyze the answer situation of the questions")
(hD: p.option_D = "Randomly selecting 10 students from 1200 high school students participating in a mock exam to understand the situation") :
  p.option_C = "Randomly selecting 100 candidates from 1200 exam participants to analyze the answer situation of the questions" := 
sorry

end suitable_for_systematic_sampling_l116_116815


namespace units_digit_of_24_pow_4_add_42_pow_4_l116_116563

theorem units_digit_of_24_pow_4_add_42_pow_4 : 
  (24^4 + 42^4) % 10 = 2 := 
by sorry

end units_digit_of_24_pow_4_add_42_pow_4_l116_116563


namespace toothpicks_15_l116_116109

noncomputable def toothpicks : ℕ → ℕ
| 0       => 0  -- since the stage count n >= 1, stage 0 is not required, default 0.
| 1       => 5
| (n + 1) => 2 * toothpicks n + 2

theorem toothpicks_15 : toothpicks 15 = 32766 := by
  sorry

end toothpicks_15_l116_116109


namespace projectile_reaches_24m_at_12_7_seconds_l116_116267

theorem projectile_reaches_24m_at_12_7_seconds :
  ∃ t : ℝ, (y = -4.9 * t^2 + 25 * t) ∧ y = 24 ∧ t = 12 / 7 :=
by
  use 12 / 7
  sorry

end projectile_reaches_24m_at_12_7_seconds_l116_116267


namespace time_for_B_alone_to_paint_l116_116948

noncomputable def rate_A := 1 / 4
noncomputable def rate_BC := 1 / 3
noncomputable def rate_AC := 1 / 2
noncomputable def rate_DB := 1 / 6

theorem time_for_B_alone_to_paint :
  (1 / (rate_BC - (rate_AC - rate_A))) = 12 := by
  sorry

end time_for_B_alone_to_paint_l116_116948


namespace af_cd_ratio_l116_116994

theorem af_cd_ratio (a b c d e f : ℝ) 
  (h1 : a * b * c = 130) 
  (h2 : b * c * d = 65) 
  (h3 : c * d * e = 750) 
  (h4 : d * e * f = 250) :
  (a * f) / (c * d) = 2 / 3 := 
by
  sorry

end af_cd_ratio_l116_116994


namespace relationship_among_abc_l116_116620

theorem relationship_among_abc (x : ℝ) (e : ℝ) (ln : ℝ → ℝ) (half_pow : ℝ → ℝ) (exp : ℝ → ℝ) 
  (x_in_e_e2 : x > e ∧ x < exp 2) 
  (def_a : ln x = ln x)
  (def_b : half_pow (ln x) = ((1/2)^(ln x)))
  (def_c : exp (ln x) = x):
  (exp (ln x)) > (ln x) ∧ (ln x) > ((1/2)^(ln x)) :=
by 
  sorry

end relationship_among_abc_l116_116620


namespace eggs_sold_l116_116721

/-- Define the notion of trays and eggs in this context -/
def trays_of_eggs : ℤ := 30

/-- Define the initial collection of trays by Haman -/
def initial_trays : ℤ := 10

/-- Define the number of trays dropped by Haman -/
def dropped_trays : ℤ := 2

/-- Define the additional trays that Haman's father told him to collect -/
def additional_trays : ℤ := 7

/-- Define the total eggs sold -/
def total_eggs_sold : ℤ :=
  (initial_trays - dropped_trays) * trays_of_eggs + additional_trays * trays_of_eggs

-- Theorem to prove the total eggs sold
theorem eggs_sold : total_eggs_sold = 450 :=
by 
  -- Insert proof here
  sorry

end eggs_sold_l116_116721


namespace sequence_formula_l116_116673

theorem sequence_formula (a : ℕ → ℕ) (h₁ : a 1 = 33) (h₂ : ∀ n : ℕ, a (n + 1) - a n = 2 * n) : 
  ∀ n : ℕ, a n = n^2 - n + 33 :=
by
  sorry

end sequence_formula_l116_116673


namespace solve_equation_l116_116492

theorem solve_equation (x : ℝ) :
  (3 / x - (1 / x * 6 / x) = -2.5) ↔ (x = (-3 + Real.sqrt 69) / 5 ∨ x = (-3 - Real.sqrt 69) / 5) :=
by {
  sorry
}

end solve_equation_l116_116492


namespace number_of_solutions_l116_116380

theorem number_of_solutions :
  ∃ sols: Finset (ℕ × ℕ), (∀ (x y : ℕ), (x, y) ∈ sols ↔ x^2 + y^2 + 2*x*y - 1988*x - 1988*y = 1989 ∧ x > 0 ∧ y > 0)
  ∧ sols.card = 1988 :=
by
  sorry

end number_of_solutions_l116_116380


namespace largest_integer_solution_of_abs_eq_and_inequality_l116_116213

theorem largest_integer_solution_of_abs_eq_and_inequality : 
  ∃ x : ℤ, |x - 3| = 15 ∧ x ≤ 20 ∧ (∀ y : ℤ, |y - 3| = 15 ∧ y ≤ 20 → y ≤ x) :=
sorry

end largest_integer_solution_of_abs_eq_and_inequality_l116_116213


namespace min_value_expression_l116_116847

theorem min_value_expression (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  ( (x + y) / z + (x + z) / y + (y + z) / x + (x + y + z) / (x + y) ) ≥ 7 :=
sorry

end min_value_expression_l116_116847


namespace area_of_triangle_PQR_l116_116415

-- Define point P
structure Point where
  x : ℝ
  y : ℝ

def P : Point := { x := 2, y := 5 }

-- Define the lines using their slopes and the point P
def line1 (x : ℝ) : ℝ := -x + 7
def line2 (x : ℝ) : ℝ := -2 * x + 9

-- Definitions of points Q and R, which are the x-intercepts
def Q : Point := { x := 7, y := 0 }
def R : Point := { x := 4.5, y := 0 }

-- Theorem statement
theorem area_of_triangle_PQR : 
  let base := 7 - 4.5
  let height := 5
  (1 / 2) * base * height = 6.25 := by
  sorry

end area_of_triangle_PQR_l116_116415


namespace min_value_x_plus_2y_l116_116714

theorem min_value_x_plus_2y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 / x + 1 / y = 4) : x + 2 * y = 2 :=
sorry

end min_value_x_plus_2y_l116_116714


namespace minimize_perimeter_of_sector_l116_116601

theorem minimize_perimeter_of_sector (r θ: ℝ) (h₁: (1 / 2) * θ * r^2 = 16) (h₂: 2 * r + θ * r = 2 * r + 32 / r): θ = 2 :=
by
  sorry

end minimize_perimeter_of_sector_l116_116601


namespace distinct_solutions_abs_eq_l116_116176

theorem distinct_solutions_abs_eq (x : ℝ) : (|x - 5| = |x + 3|) → x = 1 :=
by
  -- The proof is omitted
  sorry

end distinct_solutions_abs_eq_l116_116176


namespace distance_between_cars_l116_116523

-- Definitions representing the initial conditions and distances traveled by the cars
def initial_distance : ℕ := 113
def first_car_distance_on_road : ℕ := 50
def second_car_distance_on_road : ℕ := 35

-- Statement of the theorem to be proved
theorem distance_between_cars : initial_distance - (first_car_distance_on_road + second_car_distance_on_road) = 28 :=
by
  sorry

end distance_between_cars_l116_116523


namespace janice_trash_fraction_l116_116747

noncomputable def janice_fraction : ℚ :=
  let homework := 30
  let cleaning := homework / 2
  let walking_dog := homework + 5
  let total_tasks := homework + cleaning + walking_dog
  let total_time := 120
  let time_left := 35
  let time_spent := total_time - time_left
  let trash_time := time_spent - total_tasks
  trash_time / homework

theorem janice_trash_fraction : janice_fraction = 1 / 6 :=
by
  sorry

end janice_trash_fraction_l116_116747


namespace sum_difference_arithmetic_sequences_l116_116276

open Nat

def arithmetic_sequence_sum (a d n : Nat) : Nat :=
  n * (2 * a + (n - 1) * d) / 2

theorem sum_difference_arithmetic_sequences :
  arithmetic_sequence_sum 2101 1 123 - arithmetic_sequence_sum 401 1 123 = 209100 := by
  sorry

end sum_difference_arithmetic_sequences_l116_116276


namespace cos_4_3pi_add_alpha_l116_116089

theorem cos_4_3pi_add_alpha (α : ℝ) (h : Real.sin (Real.pi / 6 - α) = 1 / 3) :
    Real.cos (4 * Real.pi / 3 + α) = -1 / 3 := 
by sorry

end cos_4_3pi_add_alpha_l116_116089


namespace length_more_than_breadth_by_10_l116_116802

-- Definitions based on conditions
def length : ℕ := 55
def cost_per_meter : ℚ := 26.5
def total_fencing_cost : ℚ := 5300
def perimeter : ℚ := total_fencing_cost / cost_per_meter

-- Calculate breadth (b) and difference (x)
def breadth := 45 -- This is inferred manually from the solution for completeness
def difference (b : ℚ) := length - b

-- The statement we need to prove
theorem length_more_than_breadth_by_10 :
  difference 45 = 10 :=
by
  sorry

end length_more_than_breadth_by_10_l116_116802


namespace smallest_b_in_arithmetic_series_l116_116050

theorem smallest_b_in_arithmetic_series (a b c d : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c)
  (h_arith_series : a = b - d ∧ c = b + d) (h_product : a * b * c = 125) : b ≥ 5 :=
sorry

end smallest_b_in_arithmetic_series_l116_116050


namespace team_leaders_lcm_l116_116760

/-- Amanda, Brian, Carla, and Derek are team leaders rotating every
    5, 8, 10, and 12 weeks respectively. Given that this week they all are leading
    projects together, prove that they will all lead projects together again in 120 weeks. -/
theorem team_leaders_lcm :
  Nat.lcm (Nat.lcm 5 8) (Nat.lcm 10 12) = 120 := 
  by
  sorry

end team_leaders_lcm_l116_116760


namespace mark_owes_820_l116_116665

-- Definitions of the problem conditions
def base_fine : ℕ := 50
def over_speed_fine (mph_over : ℕ) : ℕ := mph_over * 2
def school_zone_multiplier : ℕ := 2
def court_costs : ℕ := 300
def lawyer_cost_per_hour : ℕ := 80
def lawyer_hours : ℕ := 3

-- Calculation of the total fine
def total_fine (speed_limit : ℕ) (actual_speed : ℕ) : ℕ :=
  let mph_over := actual_speed - speed_limit
  let additional_fine := over_speed_fine mph_over
  let fine_before_multipliers := base_fine + additional_fine
  let fine_after_multipliers := fine_before_multipliers * school_zone_multiplier
  fine_after_multipliers

-- Calculation of the total costs
def total_costs (speed_limit : ℕ) (actual_speed : ℕ) : ℕ :=
  let fine := total_fine speed_limit actual_speed
  fine + court_costs + (lawyer_cost_per_hour * lawyer_hours)

theorem mark_owes_820 : total_costs 30 75 = 820 := 
by
  sorry

end mark_owes_820_l116_116665


namespace burger_cost_l116_116534

theorem burger_cost 
    (b s : ℕ) 
    (h1 : 5 * b + 3 * s = 500) 
    (h2 : 3 * b + 2 * s = 310) :
    b = 70 := by
  sorry

end burger_cost_l116_116534


namespace walter_hushpuppies_per_guest_l116_116993

variables (guests hushpuppies_per_batch time_per_batch total_time : ℕ)

def batches (total_time time_per_batch : ℕ) : ℕ :=
  total_time / time_per_batch

def total_hushpuppies (batches hushpuppies_per_batch : ℕ) : ℕ :=
  batches * hushpuppies_per_batch

def hushpuppies_per_guest (total_hushpuppies guests : ℕ) : ℕ :=
  total_hushpuppies / guests

theorem walter_hushpuppies_per_guest :
  ∀ (guests hushpuppies_per_batch time_per_batch total_time : ℕ),
    guests = 20 →
    hushpuppies_per_batch = 10 →
    time_per_batch = 8 →
    total_time = 80 →
    hushpuppies_per_guest (total_hushpuppies (batches total_time time_per_batch) hushpuppies_per_batch) guests = 5 :=
by 
  intros _ _ _ _ h_guests h_hpb h_tpb h_tt
  sorry

end walter_hushpuppies_per_guest_l116_116993


namespace probability_diff_suits_l116_116889

theorem probability_diff_suits (n : ℕ) (h₁ : n = 65) (suits : ℕ) (h₂ : suits = 5) (cards_per_suit : ℕ) (h₃ : cards_per_suit = n / suits) : 
  (52 : ℚ) / (64 : ℚ) = (13 : ℚ) / (16 : ℚ) := 
by 
  sorry

end probability_diff_suits_l116_116889


namespace restaurant_chili_paste_needs_l116_116078

theorem restaurant_chili_paste_needs:
  let large_can_volume := 25
  let small_can_volume := 15
  let large_cans_required := 45
  let total_volume := large_cans_required * large_can_volume
  let small_cans_needed := total_volume / small_can_volume
  small_cans_needed - large_cans_required = 30 :=
by
  sorry

end restaurant_chili_paste_needs_l116_116078


namespace ice_cream_stall_difference_l116_116329

theorem ice_cream_stall_difference (d : ℕ) 
  (h1 : ∃ d, 10 + (10 + d) + (10 + 2*d) + (10 + 3*d) + (10 + 4*d) = 90) : 
  d = 4 :=
by
  sorry

end ice_cream_stall_difference_l116_116329


namespace range_of_a_l116_116855

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
if x ≤ 2 then -x + 5 else a^x + 2 * a + 2

theorem range_of_a (a : ℝ) (h₁ : a > 0) (h₂ : a ≠ 1) :
  (∀ y ∈ Set.range (f a), y ≥ 3) ↔ (a ∈ Set.Ici (1/2) ∪ Set.Ioi 1) :=
sorry

end range_of_a_l116_116855


namespace find_angle_A_l116_116696

theorem find_angle_A
  (A B C : ℝ)
  (h1 : B = A + 10)
  (h2 : C = B + 10)
  (h3 : A + B + C = 180) : A = 50 :=
by 
  sorry

end find_angle_A_l116_116696


namespace parabola_vertex_and_point_l116_116511

/-- The vertex form of the parabola is at (7, -6) and passes through the point (1,0).
    Verify that the equation parameters a, b, c satisfy a + b + c = -43 / 6. -/
theorem parabola_vertex_and_point (a b c : ℚ)
  (h_eq : ∀ y, (a * y^2 + b * y + c) = a * (y + 6)^2 + 7)
  (h_vertex : ∃ x y, x = a * y^2 + b * y + c ∧ y = -6 ∧ x = 7)
  (h_point : ∃ x y, x = a * y^2 + b * y + c ∧ x = 1 ∧ y = 0) :
  a + b + c = -43 / 6 :=
by
  sorry

end parabola_vertex_and_point_l116_116511


namespace rectangle_diagonal_length_l116_116366

theorem rectangle_diagonal_length (L W : ℝ) (h1 : L * W = 20) (h2 : L + W = 9) :
  (L^2 + W^2) = 41 :=
by
  sorry

end rectangle_diagonal_length_l116_116366


namespace power_multiplication_l116_116636

theorem power_multiplication :
  (- (4 / 5 : ℚ)) ^ 2022 * (5 / 4 : ℚ) ^ 2023 = 5 / 4 := 
by {
  sorry
}

end power_multiplication_l116_116636


namespace maximum_volume_regular_triangular_pyramid_l116_116476

-- Given values
def R : ℝ := 1

-- Prove the maximum volume
theorem maximum_volume_regular_triangular_pyramid : 
  ∃ (V_max : ℝ), V_max = (8 * Real.sqrt 3) / 27 := 
by 
  sorry

end maximum_volume_regular_triangular_pyramid_l116_116476


namespace hawks_points_l116_116590

theorem hawks_points (x y : ℕ) (h1 : x + y = 50) (h2 : x + 4 - y = 12) : y = 21 :=
by
  sorry

end hawks_points_l116_116590


namespace min_value_x2_y2_z2_l116_116853

theorem min_value_x2_y2_z2 (x y z : ℝ) (h : 2 * x + 3 * y + 4 * z = 11) : 
  x^2 + y^2 + z^2 ≥ 121 / 29 :=
sorry

end min_value_x2_y2_z2_l116_116853


namespace ratio_time_B_to_A_l116_116917

-- Definitions for the given conditions
def T_A : ℕ := 10
def work_rate_A : ℚ := 1 / T_A
def combined_work_rate : ℚ := 0.3

-- Lean 4 statement for the problem
theorem ratio_time_B_to_A (T_B : ℚ) (h : (work_rate_A + 1 / T_B) = combined_work_rate) :
  (T_B / T_A) = (1 / 2) := by
  sorry

end ratio_time_B_to_A_l116_116917


namespace total_marbles_l116_116425

theorem total_marbles (jars clay_pots total_marbles jars_marbles pots_marbles : ℕ)
  (h1 : jars = 16)
  (h2 : jars = 2 * clay_pots)
  (h3 : jars_marbles = 5)
  (h4 : pots_marbles = 3 * jars_marbles)
  (h5 : total_marbles = jars * jars_marbles + clay_pots * pots_marbles) :
  total_marbles = 200 := by
  sorry

end total_marbles_l116_116425


namespace MMobile_cheaper_l116_116171

-- Define the given conditions
def TMobile_base_cost : ℕ := 50
def TMobile_additional_cost : ℕ := 16
def MMobile_base_cost : ℕ := 45
def MMobile_additional_cost : ℕ := 14
def additional_lines : ℕ := 3

-- Define functions to calculate total costs
def TMobile_total_cost : ℕ := TMobile_base_cost + TMobile_additional_cost * additional_lines
def MMobile_total_cost : ℕ := MMobile_base_cost + MMobile_additional_cost * additional_lines

-- Statement to be proved
theorem MMobile_cheaper : TMobile_total_cost - MMobile_total_cost = 11 := by
  sorry

end MMobile_cheaper_l116_116171


namespace krystiana_monthly_earnings_l116_116324

-- Definitions based on the conditions
def first_floor_cost : ℕ := 15
def second_floor_cost : ℕ := 20
def third_floor_cost : ℕ := 2 * first_floor_cost
def first_floor_rooms : ℕ := 3
def second_floor_rooms : ℕ := 3
def third_floor_rooms_occupied : ℕ := 2

-- Statement to prove Krystiana's total monthly earnings are $165
theorem krystiana_monthly_earnings : 
  first_floor_cost * first_floor_rooms + 
  second_floor_cost * second_floor_rooms + 
  third_floor_cost * third_floor_rooms_occupied = 165 :=
by admit

end krystiana_monthly_earnings_l116_116324


namespace number_of_streams_l116_116025

theorem number_of_streams (S A B C D : Type) (f : S → A) (f1 : A → B) :
  (∀ (x : ℕ), x = 1000 → 
  (x * 375 / 1000 = 375 ∧ x * 625 / 1000 = 625) ∧ 
  (S ≠ C ∧ S ≠ D ∧ C ≠ D)) →
  -- Introduce some conditions to represent the described transition process
  -- Specifically the conditions mentioning the lakes and transitions 
  ∀ (transition_count : ℕ), 
    (transition_count = 4) →
    ∃ (number_of_streams : ℕ), number_of_streams = 3 := 
sorry

end number_of_streams_l116_116025


namespace sin_A_in_right_triangle_l116_116271

theorem sin_A_in_right_triangle (B C A : Real) (hBC: B + C = π / 2) 
(h_sinB: Real.sin B = 3 / 5) (h_sinC: Real.sin C = 4 / 5) : 
Real.sin A = 1 := 
by 
  sorry

end sin_A_in_right_triangle_l116_116271


namespace mary_balloon_count_l116_116940

theorem mary_balloon_count (n m : ℕ) (hn : n = 7) (hm : m = 4 * n) : m = 28 :=
by
  sorry

end mary_balloon_count_l116_116940


namespace abigail_spent_in_store_l116_116526

theorem abigail_spent_in_store (initial_amount : ℕ) (amount_left : ℕ) (amount_lost : ℕ) (spent : ℕ) 
  (h1 : initial_amount = 11) 
  (h2 : amount_left = 3)
  (h3 : amount_lost = 6) :
  spent = initial_amount - (amount_left + amount_lost) :=
by
  sorry

end abigail_spent_in_store_l116_116526


namespace p_2015_coordinates_l116_116535

namespace AaronWalk

def position (n : ℕ) : ℤ × ℤ :=
sorry

theorem p_2015_coordinates : position 2015 = (22, 57) := 
sorry

end AaronWalk

end p_2015_coordinates_l116_116535


namespace polygon_sides_l116_116147

theorem polygon_sides (n : ℕ) (h : 44 = n * (n - 3) / 2) : n = 11 :=
sorry

end polygon_sides_l116_116147


namespace brock_peanuts_ratio_l116_116688

theorem brock_peanuts_ratio (initial : ℕ) (bonita : ℕ) (remaining : ℕ) (brock : ℕ)
  (h1 : initial = 148) (h2 : bonita = 29) (h3 : remaining = 82) (h4 : brock = 37)
  (h5 : initial - remaining = bonita + brock) :
  (brock : ℚ) / initial = 1 / 4 :=
by {
  sorry
}

end brock_peanuts_ratio_l116_116688


namespace sum_of_max_marks_l116_116384

theorem sum_of_max_marks :
  ∀ (M S E : ℝ),
  (30 / 100 * M = 180) ∧
  (50 / 100 * S = 200) ∧
  (40 / 100 * E = 120) →
  M + S + E = 1300 :=
by
  intros M S E h
  sorry

end sum_of_max_marks_l116_116384


namespace arithmetic_seq_problem_l116_116395

theorem arithmetic_seq_problem (a : ℕ → ℝ) (d : ℝ)
  (h_seq : ∀ n, a n = a 1 + (n - 1) * d)
  (h_cond : a 1 + 3 * a 8 + a 15 = 60) :
  2 * a 9 - a 10 = 12 := 
sorry

end arithmetic_seq_problem_l116_116395


namespace tan_of_acute_angle_and_cos_pi_add_alpha_l116_116365

theorem tan_of_acute_angle_and_cos_pi_add_alpha (α : ℝ) (h1 : 0 < α ∧ α < π/2)
  (h2 : Real.cos (π + α) = -Real.sqrt (3) / 2) : 
  Real.tan α = Real.sqrt (3) / 3 :=
by
  sorry

end tan_of_acute_angle_and_cos_pi_add_alpha_l116_116365


namespace Jungkook_red_balls_count_l116_116414

-- Define the conditions
def red_balls_per_box : ℕ := 3
def boxes_Jungkook_has : ℕ := 2

-- Statement to prove
theorem Jungkook_red_balls_count : red_balls_per_box * boxes_Jungkook_has = 6 :=
by sorry

end Jungkook_red_balls_count_l116_116414


namespace solve_for_a_l116_116623

theorem solve_for_a (x y a : ℝ) (h1 : 2 * x + y = 2 * a + 1) 
                    (h2 : x + 2 * y = a - 1) 
                    (h3 : x - y = 4) : a = 2 :=
by
  sorry

end solve_for_a_l116_116623


namespace sin_alpha_l116_116627

variable (α : Real)
variable (hcos : Real.cos α = 3 / 5)
variable (htan : Real.tan α < 0)

theorem sin_alpha (α : Real) (hcos : Real.cos α = 3 / 5) (htan : Real.tan α < 0) :
  Real.sin α = -4 / 5 :=
sorry

end sin_alpha_l116_116627


namespace probability_two_boys_and_three_girls_l116_116041

def binomial_coefficient (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (binomial_coefficient n k) * (p ^ k) * ((1 - p) ^ (n - k))

theorem probability_two_boys_and_three_girls :
  binomial_probability 5 2 0.5 = 0.3125 :=
by
  sorry

end probability_two_boys_and_three_girls_l116_116041


namespace average_speed_of_the_car_l116_116990

noncomputable def averageSpeed (d1 d2 d3 d4 t1 t2 t3 t4 : ℝ) : ℝ :=
  let totalDistance := d1 + d2 + d3 + d4
  let totalTime := t1 + t2 + t3 + t4
  totalDistance / totalTime

theorem average_speed_of_the_car :
  averageSpeed 30 35 65 (40 * 0.5) (30 / 45) (35 / 55) 1 0.5 = 54 := 
  by 
    sorry

end average_speed_of_the_car_l116_116990


namespace oranges_in_bin_after_changes_l116_116394

-- Define the initial number of oranges
def initial_oranges : ℕ := 34

-- Define the number of oranges thrown away
def oranges_thrown_away : ℕ := 20

-- Define the number of new oranges added
def new_oranges_added : ℕ := 13

-- Theorem statement to prove the final number of oranges in the bin
theorem oranges_in_bin_after_changes :
  initial_oranges - oranges_thrown_away + new_oranges_added = 27 := by
  sorry

end oranges_in_bin_after_changes_l116_116394


namespace loan_period_l116_116067

theorem loan_period (principal : ℝ) (rate_A rate_C : ℝ) (gain : ℝ) (years : ℝ) :
  principal = 3500 ∧ rate_A = 0.1 ∧ rate_C = 0.12 ∧ gain = 210 →
  (rate_C * principal * years - rate_A * principal * years) = gain →
  years = 3 :=
by
  sorry

end loan_period_l116_116067


namespace jean_average_mark_l116_116470

/-
  Jean writes five tests and achieves the following marks: 80, 70, 60, 90, and 80.
  Prove that her average mark on these five tests is 76.
-/
theorem jean_average_mark : 
  let marks := [80, 70, 60, 90, 80]
  let total_marks := marks.sum
  let number_of_tests := marks.length
  let average_mark := total_marks / number_of_tests
  average_mark = 76 :=
by 
  let marks := [80, 70, 60, 90, 80]
  let total_marks := marks.sum
  let number_of_tests := marks.length
  let average_mark := total_marks / number_of_tests
  sorry

end jean_average_mark_l116_116470


namespace margie_change_is_6_25_l116_116592

-- The conditions are given as definitions in Lean
def numberOfApples : Nat := 5
def costPerApple : ℝ := 0.75
def amountPaid : ℝ := 10.00

-- The statement to be proved
theorem margie_change_is_6_25 :
  (amountPaid - (numberOfApples * costPerApple)) = 6.25 := 
  sorry

end margie_change_is_6_25_l116_116592


namespace sqrt_k_kn_eq_k_sqrt_kn_l116_116930

theorem sqrt_k_kn_eq_k_sqrt_kn (k n : ℕ) (h : k = Nat.sqrt (n + 1)) : 
  Real.sqrt (k * (k / n)) = k * Real.sqrt (k / n) := 
sorry

end sqrt_k_kn_eq_k_sqrt_kn_l116_116930


namespace efficiency_ratio_l116_116932

theorem efficiency_ratio (r : ℚ) (work_B : ℚ) (work_AB : ℚ) (B_alone : ℚ) (AB_together : ℚ) (efficiency_A : ℚ) (B_efficiency : ℚ) :
  B_alone = 30 ∧ AB_together = 20 ∧ B_efficiency = (1/B_alone) ∧ efficiency_A = (r * B_efficiency) ∧ (efficiency_A + B_efficiency) = (1 / AB_together) → r = 1 / 2 :=
by
  sorry

end efficiency_ratio_l116_116932


namespace van_speed_maintain_l116_116238

theorem van_speed_maintain 
  (D : ℕ) (T T_new : ℝ) 
  (initial_distance : D = 435) 
  (initial_time : T = 5) 
  (new_time : T_new = T / 2) : 
  D / T_new = 174 := 
by 
  sorry

end van_speed_maintain_l116_116238


namespace numbers_at_distance_1_from_neg2_l116_116189

theorem numbers_at_distance_1_from_neg2 : 
  ∃ x : ℤ, (|x + 2| = 1) ∧ (x = -1 ∨ x = -3) :=
by
  sorry

end numbers_at_distance_1_from_neg2_l116_116189


namespace PolygonNumberSides_l116_116068

theorem PolygonNumberSides (n : ℕ) (h : n - (1 / 2 : ℝ) * (n * (n - 3)) / 2 = 0) : n = 7 :=
by
  sorry

end PolygonNumberSides_l116_116068


namespace convex_quad_sum_greater_diff_l116_116240

theorem convex_quad_sum_greater_diff (α β γ δ : ℝ) 
    (h_sum : α + β + γ + δ = 360) 
    (h_convex : α < 180 ∧ β < 180 ∧ γ < 180 ∧ δ < 180) :
    ∀ (x y z w : ℝ), (x = α ∨ x = β ∨ x = γ ∨ x = δ) → (y = α ∨ y = β ∨ y = γ ∨ y = δ) → 
                     (z = α ∨ z = β ∨ z = γ ∨ z = δ) → (w = α ∨ w = β ∨ w = γ ∨ w = δ) 
                     → x + y > |z - w| := 
by
  sorry

end convex_quad_sum_greater_diff_l116_116240


namespace customOp_eval_l116_116048

-- Define the custom operation
def customOp (a b : ℤ) : ℤ := a * (b + 1) + a * b

-- State the theorem we need to prove
theorem customOp_eval : customOp 4 (-1) = -4 :=
  by
    sorry

end customOp_eval_l116_116048


namespace sum_sq_roots_cubic_l116_116730

noncomputable def sum_sq_roots (r s t : ℝ) : ℝ :=
  r^2 + s^2 + t^2

theorem sum_sq_roots_cubic :
  ∀ r s t, (2 * r^3 + 3 * r^2 - 5 * r + 1 = 0) →
           (2 * s^3 + 3 * s^2 - 5 * s + 1 = 0) →
           (2 * t^3 + 3 * t^2 - 5 * t + 1 = 0) →
           (r + s + t = -3 / 2) →
           (r * s + r * t + s * t = 5 / 2) →
           sum_sq_roots r s t = -11 / 4 :=
by 
  intros r s t h₁ h₂ h₃ sum_roots prod_roots
  sorry

end sum_sq_roots_cubic_l116_116730


namespace problem_statement_l116_116848

theorem problem_statement :
  ¬ (3^2 = 6) ∧ 
  ¬ ((-1 / 4) / (-4) = 1) ∧
  ¬ ((-8)^2 = -16) ∧
  (-5 - (-2) = -3) := 
by 
  sorry

end problem_statement_l116_116848


namespace smallest_among_5_8_4_l116_116063

theorem smallest_among_5_8_4 : ∀ (x y z : ℕ), x = 5 → y = 8 → z = 4 → z ≤ x ∧ z ≤ y :=
by
  intros x y z hx hy hz
  rw [hx, hy, hz]
  exact ⟨by norm_num, by norm_num⟩

end smallest_among_5_8_4_l116_116063


namespace largest_integer_value_x_l116_116013

theorem largest_integer_value_x : ∀ (x : ℤ), (5 - 4 * x > 17) → x ≤ -4 := sorry

end largest_integer_value_x_l116_116013


namespace optionD_is_equation_l116_116703

-- Definitions for options
def optionA (x : ℕ) := 2 * x - 3
def optionB := 2 + 4 = 6
def optionC (x : ℕ) := x > 2
def optionD (x : ℕ) := 2 * x - 1 = 3

-- Goal: prove that option D is an equation.
theorem optionD_is_equation (x : ℕ) : (optionD x) = True :=
sorry

end optionD_is_equation_l116_116703


namespace exists_unique_i_l116_116984

theorem exists_unique_i (p : ℕ) (hp : Nat.Prime p) (hp2 : p % 2 = 1) 
  (a : ℤ) (ha1 : 2 ≤ a) (ha2 : a ≤ p - 2) : 
  ∃! (i : ℤ), 2 ≤ i ∧ i ≤ p - 2 ∧ (i * a) % p = 1 ∧ Nat.gcd (i.natAbs) (a.natAbs) = 1 :=
sorry

end exists_unique_i_l116_116984


namespace triangle_converse_inverse_false_l116_116790

variables {T : Type} (p q : T → Prop)

-- Condition: If a triangle is equilateral, then it is isosceles
axiom h : ∀ t, p t → q t

-- Conclusion: Neither the converse nor the inverse is true
theorem triangle_converse_inverse_false : 
  (∃ t, q t ∧ ¬ p t) ∧ (∃ t, ¬ p t ∧ q t) :=
sorry

end triangle_converse_inverse_false_l116_116790


namespace length_of_walls_l116_116573

-- Definitions of the given conditions.
def wall_height : ℝ := 12
def third_wall_length : ℝ := 20
def third_wall_height : ℝ := 12
def total_area : ℝ := 960

-- The area of two walls with length L each and height 12 feet.
def two_walls_area (L : ℝ) : ℝ := 2 * L * wall_height

-- The area of the third wall.
def third_wall_area : ℝ := third_wall_length * third_wall_height

-- The proof statement
theorem length_of_walls (L : ℝ) (h1 : two_walls_area L + third_wall_area = total_area) : L = 30 :=
by
  sorry

end length_of_walls_l116_116573


namespace rotation_90_deg_l116_116332

theorem rotation_90_deg (z : ℂ) (r : ℂ → ℂ) (h : ∀ (x y : ℝ), r (x + y*I) = -y + x*I) :
  r (8 - 5*I) = 5 + 8*I :=
by sorry

end rotation_90_deg_l116_116332


namespace max_minus_min_eq_32_l116_116600

def f (x : ℝ) : ℝ := x^3 - 12*x + 8

theorem max_minus_min_eq_32 : 
  let M := max (f (-3)) (max (f 3) (max (f (-2)) (f 2)))
  let m := min (f (-3)) (min (f 3) (min (f (-2)) (f 2)))
  M - m = 32 :=
by
  sorry

end max_minus_min_eq_32_l116_116600


namespace mass_of_man_l116_116656

-- Definitions based on problem conditions
def boat_length : ℝ := 8
def boat_breadth : ℝ := 3
def sinking_height : ℝ := 0.01
def water_density : ℝ := 1000

-- Mass of the man to be proven
theorem mass_of_man : boat_density * (boat_length * boat_breadth * sinking_height) = 240 :=
by
  sorry

end mass_of_man_l116_116656


namespace johns_total_animals_l116_116602

variable (Snakes Monkeys Lions Pandas Dogs : ℕ)

theorem johns_total_animals :
  Snakes = 15 →
  Monkeys = 2 * Snakes →
  Lions = Monkeys - 5 →
  Pandas = Lions + 8 →
  Dogs = Pandas / 3 →
  Snakes + Monkeys + Lions + Pandas + Dogs = 114 :=
by
  intros hSnakes hMonkeys hLions hPandas hDogs
  rw [hSnakes] at hMonkeys
  rw [hMonkeys] at hLions
  rw [hLions] at hPandas
  rw [hPandas] at hDogs
  sorry

end johns_total_animals_l116_116602


namespace num_undefined_values_l116_116093

theorem num_undefined_values :
  ∃! x : Finset ℝ, (∀ y ∈ x, (y + 5 = 0) ∨ (y - 1 = 0) ∨ (y - 4 = 0)) ∧ (x.card = 3) := sorry

end num_undefined_values_l116_116093


namespace mean_value_of_quadrilateral_angles_l116_116038

theorem mean_value_of_quadrilateral_angles (a b c d : ℝ) (h_angles_sum : a + b + c + d = 360) : (a + b + c + d) / 4 = 90 := 
by
  sorry

end mean_value_of_quadrilateral_angles_l116_116038


namespace base10_to_base4_156_eq_2130_l116_116333

def base10ToBase4 (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec loop (n : ℕ) (acc : List ℕ) : List ℕ :=
      if n = 0 then acc
      else loop (n / 4) ((n % 4) :: acc)
    loop n []

theorem base10_to_base4_156_eq_2130 :
  base10ToBase4 156 = [2, 1, 3, 0] := sorry

end base10_to_base4_156_eq_2130_l116_116333


namespace speed_downstream_is_correct_l116_116805

-- Definitions corresponding to the conditions
def speed_boat_still_water : ℕ := 60
def speed_current : ℕ := 17

-- Definition of speed downstream from the conditions and proving the result
theorem speed_downstream_is_correct :
  speed_boat_still_water + speed_current = 77 :=
by
  -- Proof is omitted
  sorry

end speed_downstream_is_correct_l116_116805


namespace solve_for_z_l116_116839

theorem solve_for_z {x y z : ℝ} (h : (1 / x^2) - (1 / y^2) = 1 / z) :
  z = (x^2 * y^2) / (y^2 - x^2) :=
sorry

end solve_for_z_l116_116839


namespace not_divisible_by_121_l116_116884

theorem not_divisible_by_121 (n : ℤ) : ¬ (121 ∣ (n^2 + 2 * n + 2014)) :=
sorry

end not_divisible_by_121_l116_116884


namespace baker_additional_cakes_l116_116363

theorem baker_additional_cakes (X : ℕ) : 
  (62 + X) - 144 = 67 → X = 149 :=
by
  intro h
  sorry

end baker_additional_cakes_l116_116363


namespace triangle_interior_angles_not_greater_than_60_l116_116750

theorem triangle_interior_angles_not_greater_than_60 (α β γ : ℝ) (h_sum : α + β + γ = 180) 
  (h_pos : α > 0 ∧ β > 0 ∧ γ > 0) :
  α ≤ 60 ∨ β ≤ 60 ∨ γ ≤ 60 :=
by
  sorry

end triangle_interior_angles_not_greater_than_60_l116_116750


namespace households_neither_car_nor_bike_l116_116177

-- Define the given conditions
def total_households : ℕ := 90
def car_and_bike : ℕ := 18
def households_with_car : ℕ := 44
def bike_only : ℕ := 35

-- Prove the number of households with neither car nor bike
theorem households_neither_car_nor_bike :
  (total_households - ((households_with_car + bike_only) - car_and_bike)) = 11 :=
by
  sorry

end households_neither_car_nor_bike_l116_116177


namespace max_values_of_x_max_area_abc_l116_116300

noncomputable def m (x : ℝ) : ℝ × ℝ := ⟨2 * Real.sin x, Real.sin x - Real.cos x⟩
noncomputable def n (x : ℝ) : ℝ × ℝ := ⟨Real.sqrt 3 * Real.cos x, Real.sin x + Real.cos x⟩
noncomputable def f (x : ℝ) : ℝ := Prod.fst (m x) * Prod.fst (n x) + Prod.snd (m x) * Prod.snd (n x)

theorem max_values_of_x
  (k : ℤ) : ∃ x, x = k * Real.pi + Real.pi / 3 ∧ f x = 2 * Real.sin (2 * x - π / 6) :=
sorry

noncomputable def C : ℝ := Real.pi / 3
noncomputable def area_abc (a b c : ℝ) : ℝ := 1 / 2 * a * b * Real.sin C

theorem max_area_abc (a b : ℝ) (h₁ : c = Real.sqrt 3) (h₂ : f C = 2) :
  area_abc a b c ≤ 3 * Real.sqrt 3 / 4 :=
sorry

end max_values_of_x_max_area_abc_l116_116300


namespace total_first_half_points_l116_116944

-- Define the sequences for Tigers and Lions
variables (a ar b d : ℕ)
-- Defining conditions
def tied_first_quarter : Prop := a = b
def geometric_tigers : Prop := ∃ r : ℕ, ar = a * r ∧ ar^2 = a * r^2 ∧ ar^3 = a * r^3
def arithmetic_lions : Prop := b+d = b + d ∧ b+2*d = b + 2*d ∧ b+3*d = b + 3*d
def tigers_win_by_four : Prop := (a + ar + ar^2 + ar^3) = (b + (b + d) + (b + 2*d) + (b + 3*d)) + 4
def score_limit : Prop := (a + ar + ar^2 + ar^3) ≤ 120 ∧ (b + (b + d) + (b + 2*d) + (b + 3*d)) ≤ 120

-- Goal: The total number of points scored by the two teams in the first half is 23
theorem total_first_half_points : tied_first_quarter a b ∧ geometric_tigers a ar ∧ arithmetic_lions b d ∧ tigers_win_by_four a ar b d ∧ score_limit a ar b d → 
(a + ar) + (b + d) = 23 := 
by {
  sorry
}

end total_first_half_points_l116_116944


namespace population_doubles_in_35_years_l116_116936

noncomputable def birth_rate : ℝ := 39.4 / 1000
noncomputable def death_rate : ℝ := 19.4 / 1000
noncomputable def natural_increase_rate : ℝ := birth_rate - death_rate
noncomputable def doubling_time (r: ℝ) : ℝ := 70 / (r * 100)

theorem population_doubles_in_35_years :
  doubling_time natural_increase_rate = 35 := by sorry

end population_doubles_in_35_years_l116_116936


namespace sean_needs_six_packs_l116_116525

/-- 
 Sean needs to replace 2 light bulbs in the bedroom, 
 1 in the bathroom, 1 in the kitchen, and 4 in the basement. 
 He also needs to replace 1/2 of that amount in the garage. 
 The bulbs come 2 per pack. 
 -/
def bedroom_bulbs: ℕ := 2
def bathroom_bulbs: ℕ := 1
def kitchen_bulbs: ℕ := 1
def basement_bulbs: ℕ := 4
def bulbs_per_pack: ℕ := 2

noncomputable def total_bulbs_needed_including_garage: ℕ := 
  let total_rooms_bulbs := bedroom_bulbs + bathroom_bulbs + kitchen_bulbs + basement_bulbs
  let garage_bulbs := total_rooms_bulbs / 2
  total_rooms_bulbs + garage_bulbs

noncomputable def total_packs_needed: ℕ := total_bulbs_needed_including_garage / bulbs_per_pack

theorem sean_needs_six_packs : total_packs_needed = 6 :=
by
  sorry

end sean_needs_six_packs_l116_116525


namespace card_game_final_amounts_l116_116245

theorem card_game_final_amounts
  (T : ℝ)
  (aldo_initial_ratio : ℝ := 7)
  (bernardo_initial_ratio : ℝ := 6)
  (carlos_initial_ratio : ℝ := 5)
  (aldo_final_ratio : ℝ := 6)
  (bernardo_final_ratio : ℝ := 5)
  (carlos_final_ratio : ℝ := 4)
  (aldo_won : ℝ := 1200) :
  aldo_won = (1 / 90) * T →
  T = 108000 →
  (36 / 90) * T = 43200 ∧ (30 / 90) * T = 36000 ∧ (24 / 90) * T = 28800 := sorry

end card_game_final_amounts_l116_116245


namespace distance_between_lamps_l116_116479

/-- 
A rectangular classroom measures 10 meters in length. Two lamps emitting conical light beams with a 90° opening angle 
are installed on the ceiling. The first lamp is located at the center of the ceiling and illuminates a circle on the 
floor with a diameter of 6 meters. The second lamp is adjusted such that the illuminated area along the length 
of the classroom spans a 10-meter section without reaching the opposite walls. Prove that the distance between the 
two lamps is 4 meters.
-/
theorem distance_between_lamps : 
  ∀ (length width height : ℝ) (center_illum_radius illum_length : ℝ) (d_center_to_lamp1 d_center_to_lamp2 dist_lamps : ℝ),
  length = 10 ∧ d_center_to_lamp1 = 3 ∧ d_center_to_lamp2 = 1 ∧ dist_lamps = 4 → d_center_to_lamp1 - d_center_to_lamp2 = dist_lamps :=
by
  intros length width height center_illum_radius illum_length d_center_to_lamp1 d_center_to_lamp2 dist_lamps conditions
  sorry

end distance_between_lamps_l116_116479


namespace find_fraction_l116_116970

noncomputable def distinct_real_numbers (a b : ℝ) : Prop :=
  a ≠ b

noncomputable def equation_condition (a b : ℝ) : Prop :=
  (2 * a / (3 * b)) + ((a + 12 * b) / (3 * b + 12 * a)) = (5 / 3)

theorem find_fraction (a b : ℝ) (h1 : distinct_real_numbers a b) (h2 : equation_condition a b) : a / b = -93 / 49 :=
by
  sorry

end find_fraction_l116_116970


namespace historical_fiction_new_releases_fraction_l116_116494

noncomputable def HF_fraction_total_inventory : ℝ := 0.4
noncomputable def Mystery_fraction_total_inventory : ℝ := 0.3
noncomputable def SF_fraction_total_inventory : ℝ := 0.2
noncomputable def Romance_fraction_total_inventory : ℝ := 0.1

noncomputable def HF_new_release_percentage : ℝ := 0.35
noncomputable def Mystery_new_release_percentage : ℝ := 0.60
noncomputable def SF_new_release_percentage : ℝ := 0.45
noncomputable def Romance_new_release_percentage : ℝ := 0.80

noncomputable def historical_fiction_new_releases : ℝ := HF_fraction_total_inventory * HF_new_release_percentage
noncomputable def mystery_new_releases : ℝ := Mystery_fraction_total_inventory * Mystery_new_release_percentage
noncomputable def sf_new_releases : ℝ := SF_fraction_total_inventory * SF_new_release_percentage
noncomputable def romance_new_releases : ℝ := Romance_fraction_total_inventory * Romance_new_release_percentage

noncomputable def total_new_releases : ℝ :=
  historical_fiction_new_releases + mystery_new_releases + sf_new_releases + romance_new_releases

theorem historical_fiction_new_releases_fraction :
  (historical_fiction_new_releases / total_new_releases) = (2 / 7) :=
by
  sorry

end historical_fiction_new_releases_fraction_l116_116494


namespace side_length_square_l116_116032

theorem side_length_square (A : ℝ) (s : ℝ) (h1 : A = 30) (h2 : A = s^2) : 5 < s ∧ s < 6 :=
by
  -- the proof would go here
  sorry

end side_length_square_l116_116032


namespace cos_seven_pi_six_l116_116491

theorem cos_seven_pi_six : (Real.cos (7 * Real.pi / 6) = - Real.sqrt 3 / 2) :=
sorry

end cos_seven_pi_six_l116_116491


namespace walter_age_in_2001_l116_116763

/-- In 1996, Walter was one-third as old as his grandmother, 
and the sum of the years in which they were born is 3864.
Prove that Walter will be 37 years old at the end of 2001. -/
theorem walter_age_in_2001 (y : ℕ) (H1 : ∃ g, g = 3 * y)
  (H2 : 1996 - y + (1996 - (3 * y)) = 3864) : y + 5 = 37 :=
by sorry

end walter_age_in_2001_l116_116763


namespace smallest_sum_l116_116567

theorem smallest_sum (x y : ℕ) (hx : x > 0) (hy : y > 0) (hxy : x ≠ y) 
  (h : (1/x + 1/y = 1/10)) : x + y = 49 := 
sorry

end smallest_sum_l116_116567


namespace correct_statement_l116_116016

noncomputable def b (n : ℕ) (α : ℕ → ℕ) : ℚ :=
  let rec b_aux (m : ℕ) :=
    match m with
    | 0     => 0
    | m + 1 => 1 + 1 / (α m + b_aux m)
  b_aux n

theorem correct_statement (α : ℕ → ℕ) (h : ∀ k, α k > 0) : b 4 α < b 7 α :=
by sorry

end correct_statement_l116_116016


namespace SumataFamilyTotalMiles_l116_116458

def miles_per_day := 250
def days := 5

theorem SumataFamilyTotalMiles : miles_per_day * days = 1250 :=
by
  sorry

end SumataFamilyTotalMiles_l116_116458


namespace correct_system_of_equations_l116_116743

theorem correct_system_of_equations :
  ∃ (x y : ℝ), (4 * x + y = 5 * y + x) ∧ (5 * x + 6 * y = 16) := sorry

end correct_system_of_equations_l116_116743


namespace grouping_equal_products_l116_116781

def group1 : List Nat := [12, 42, 95, 143]
def group2 : List Nat := [30, 44, 57, 91]

def product (l : List Nat) : Nat :=
  l.foldl (· * ·) 1

theorem grouping_equal_products :
  product group1 = product group2 := by
  sorry

end grouping_equal_products_l116_116781


namespace rotated_angle_l116_116999

theorem rotated_angle (initial_angle : ℝ) (rotation_angle : ℝ) (final_angle : ℝ) :
  initial_angle = 30 ∧ rotation_angle = 450 → final_angle = 60 :=
by
  intro h
  sorry

end rotated_angle_l116_116999


namespace identify_parrots_l116_116178

-- Definitions of parrots
inductive Parrot
| gosha : Parrot
| kesha : Parrot
| roma : Parrot

open Parrot

-- Properties of each parrot
def always_honest (p : Parrot) : Prop :=
  p = gosha

def always_liar (p : Parrot) : Prop :=
  p = kesha

def sometimes_honest (p : Parrot) : Prop :=
  p = roma

-- Statements given by each parrot
def Gosha_statement : Prop :=
  always_liar kesha

def Kesha_statement : Prop :=
  sometimes_honest kesha

def Roma_statement : Prop :=
  always_honest kesha

-- Final statement to prove the identities
theorem identify_parrots (p : Parrot) :
  Gosha_statement ∧ Kesha_statement ∧ Roma_statement → (always_liar Parrot.kesha ∧ sometimes_honest Parrot.roma) :=
by
  intro h
  exact sorry

end identify_parrots_l116_116178


namespace finalCostCalculation_l116_116972

-- Define the inputs
def tireRepairCost : ℝ := 7
def salesTaxPerTire : ℝ := 0.50
def numberOfTires : ℕ := 4

-- The total cost should be $30
theorem finalCostCalculation : 
  let repairTotal := tireRepairCost * numberOfTires
  let salesTaxTotal := salesTaxPerTire * numberOfTires
  repairTotal + salesTaxTotal = 30 := 
by {
  sorry
}

end finalCostCalculation_l116_116972


namespace value_of_k_l116_116883

theorem value_of_k (k x : ℕ) (h1 : 2^x - 2^(x - 2) = k * 2^10) (h2 : x = 12) : k = 3 := by
  sorry

end value_of_k_l116_116883


namespace max_possible_value_l116_116866

theorem max_possible_value (a b : ℝ) (h : ∀ n : ℕ, 1 ≤ n → n ≤ 2008 → a + b = a^n + b^n) :
  ∃ a b, ∀ n : ℕ, 1 ≤ n → n ≤ 2008 → a + b = a^n + b^n → ∃ s : ℝ, (s = 0 ∨ s = 1 ∨ s = 2) →
  max (1 / a^(2009) + 1 / b^(2009)) = 2 :=
sorry

end max_possible_value_l116_116866


namespace prize_winner_is_B_l116_116356

-- Define the possible entries winning the prize
inductive Prize
| A
| B
| C
| D

open Prize

-- Define each student's predictions
def A_pred (prize : Prize) : Prop := prize = C ∨ prize = D
def B_pred (prize : Prize) : Prop := prize = B
def C_pred (prize : Prize) : Prop := prize ≠ A ∧ prize ≠ D
def D_pred (prize : Prize) : Prop := prize = C

-- Define the main theorem to prove
theorem prize_winner_is_B (prize : Prize) :
  (A_pred prize ∧ B_pred prize ∧ ¬C_pred prize ∧ ¬D_pred prize) ∨
  (A_pred prize ∧ ¬B_pred prize ∧ C_pred prize ∧ ¬D_pred prize) ∨
  (¬A_pred prize ∧ B_pred prize ∧ C_pred prize ∧ ¬D_pred prize) ∨
  (¬A_pred prize ∧ ¬B_pred prize ∧ C_pred prize ∧ D_pred prize) →
  prize = B :=
sorry

end prize_winner_is_B_l116_116356


namespace solve_quadratic_completing_square_l116_116682

theorem solve_quadratic_completing_square :
  ∃ (a b c : ℤ), a > 0 ∧ 25 * a * a + 30 * b - 45 = (a * x + b)^2 - c ∧
                 a + b + c = 62 :=
by
  sorry

end solve_quadratic_completing_square_l116_116682


namespace perfect_square_impossible_l116_116690
noncomputable def is_perfect_square (n : ℕ) : Prop :=
∃ m : ℕ, m * m = n

theorem perfect_square_impossible (a b c : ℕ) (a_positive : a > 0) (b_positive : b > 0) (c_positive : c > 0) :
  ¬ (is_perfect_square (a^2 + b + c) ∧ is_perfect_square (b^2 + c + a) ∧ is_perfect_square (c^2 + a + b)) :=
sorry

end perfect_square_impossible_l116_116690


namespace second_divisor_is_340_l116_116943

theorem second_divisor_is_340 
  (n : ℕ)
  (h1 : n = 349)
  (h2 : n % 13 = 11)
  (h3 : n % D = 9) : D = 340 :=
by
  sorry

end second_divisor_is_340_l116_116943


namespace athena_total_spent_l116_116390

noncomputable def cost_sandwiches := 4 * 3.25
noncomputable def cost_fruit_drinks := 3 * 2.75
noncomputable def cost_cookies := 6 * 1.50
noncomputable def cost_chips := 2 * 1.85

noncomputable def total_cost := cost_sandwiches + cost_fruit_drinks + cost_cookies + cost_chips

theorem athena_total_spent : total_cost = 33.95 := 
by 
  simp [cost_sandwiches, cost_fruit_drinks, cost_cookies, cost_chips, total_cost]
  sorry

end athena_total_spent_l116_116390


namespace find_ratio_of_three_numbers_l116_116658

noncomputable def ratio_of_three_numbers (A B C : ℝ) : Prop :=
  (A + B + C) / (A + B - C) = 4 / 3 ∧
  (A + B) / (B + C) = 7 / 6

theorem find_ratio_of_three_numbers (A B C : ℝ) (h₁ : ratio_of_three_numbers A B C) :
  A / C = 2 ∧ B / C = 5 :=
by
  sorry

end find_ratio_of_three_numbers_l116_116658


namespace jessica_cut_21_roses_l116_116608

def initial_roses : ℕ := 2
def thrown_roses : ℕ := 4
def final_roses : ℕ := 23

theorem jessica_cut_21_roses : (final_roses - initial_roses) = 21 :=
by
  sorry

end jessica_cut_21_roses_l116_116608


namespace log_evaluation_l116_116777

theorem log_evaluation
  (x : ℝ)
  (h : x = (Real.log 3 / Real.log 5) ^ (Real.log 5 / Real.log 3)) :
  Real.log x / Real.log 7 = -(Real.log 5 / Real.log 3) * (Real.log (Real.log 5 / Real.log 3) / Real.log 7) :=
by
  sorry

end log_evaluation_l116_116777


namespace total_distance_traveled_l116_116061

theorem total_distance_traveled
  (r1 r2 : ℝ) (h1 : r1 = 15) (h2 : r2 = 25):
  let arc_outer := 1/4 * 2 * Real.pi * r2
  let radial := r2 - r1
  let circ_inner := 2 * Real.pi * r1
  let return_radial := radial
  let total_distance := arc_outer + radial + circ_inner + return_radial
  total_distance = 42.5 * Real.pi + 20 := 
by
  sorry

end total_distance_traveled_l116_116061


namespace tony_slices_remaining_l116_116661

def slices_remaining : ℕ :=
  let total_slices := 22
  let slices_per_sandwich := 2
  let daily_sandwiches := 1
  let extra_sandwiches := 1
  let total_sandwiches := 7 * daily_sandwiches + extra_sandwiches
  total_slices - total_sandwiches * slices_per_sandwich

theorem tony_slices_remaining :
  slices_remaining = 6 :=
by
  -- Here, the theorem proof would go, but it is omitted as per instructions
  sorry

end tony_slices_remaining_l116_116661


namespace q_can_be_true_or_false_l116_116924

theorem q_can_be_true_or_false (p q : Prop) (h1 : ¬(p ∧ q)) (h2 : ¬p) : q ∨ ¬q :=
by
  sorry

end q_can_be_true_or_false_l116_116924


namespace largest_x_l116_116962

def largest_x_with_condition_eq_7_over_8 (x : ℝ) : Prop :=
  ⌊x⌋ / x = 7 / 8

theorem largest_x (x : ℝ) (h : largest_x_with_condition_eq_7_over_8 x) :
  x = 48 / 7 :=
sorry

end largest_x_l116_116962


namespace solve_quadratic_l116_116389

-- Problem Definition
def quadratic_equation (x : ℝ) : Prop :=
  2 * x^2 - 6 * x + 3 = 0

-- Solution Definition
def solution1 (x : ℝ) : Prop :=
  x = (3 + Real.sqrt 3) / 2 ∨ x = (3 - Real.sqrt 3) / 2

-- Lean Theorem Statement
theorem solve_quadratic : ∀ x : ℝ, quadratic_equation x ↔ solution1 x :=
sorry

end solve_quadratic_l116_116389


namespace tom_strokes_over_par_l116_116992

theorem tom_strokes_over_par 
  (rounds : ℕ) 
  (holes_per_round : ℕ) 
  (avg_strokes_per_hole : ℕ) 
  (par_value_per_hole : ℕ) 
  (h1 : rounds = 9) 
  (h2 : holes_per_round = 18) 
  (h3 : avg_strokes_per_hole = 4) 
  (h4 : par_value_per_hole = 3) : 
  (rounds * holes_per_round * avg_strokes_per_hole - rounds * holes_per_round * par_value_per_hole = 162) :=
by { 
  sorry 
}

end tom_strokes_over_par_l116_116992


namespace count_multiples_of_14_between_100_and_400_l116_116807

theorem count_multiples_of_14_between_100_and_400 : 
  ∃ n : ℕ, n = 21 ∧ (∀ k : ℕ, (100 ≤ k ∧ k ≤ 400 ∧ 14 ∣ k) ↔ (∃ i : ℕ, k = 14 * i ∧ 8 ≤ i ∧ i ≤ 28)) :=
sorry

end count_multiples_of_14_between_100_and_400_l116_116807


namespace common_difference_of_arithmetic_seq_l116_116618

-- Definitions based on the conditions
def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
∀ n m : ℕ, (m - n = 1) → (a (m + 1) - a m) = (a (n + 1) - a n)

/-- The common difference of an arithmetic sequence given certain conditions. -/
theorem common_difference_of_arithmetic_seq (a: ℕ → ℤ) (d : ℤ):
    a 1 + a 2 = 4 → 
    a 3 + a 4 = 16 →
    arithmetic_sequence a →
    (a 2 - a 1) = d → d = 3 :=
by
  intros h1 h2 h3 h4
  -- Proof to be filled in here
  sorry

end common_difference_of_arithmetic_seq_l116_116618


namespace prove_statement_II_l116_116423

variable (digit : ℕ)

def statement_I : Prop := (digit = 2)
def statement_II : Prop := (digit ≠ 3)
def statement_III : Prop := (digit = 5)
def statement_IV : Prop := (digit ≠ 6)

/- The main proposition that three statements are true and one is false. -/
def three_true_one_false (s1 s2 s3 s4 : Prop) : Prop :=
  (s1 ∧ s2 ∧ s3 ∧ ¬s4) ∨ (s1 ∧ s2 ∧ ¬s3 ∧ s4) ∨ 
  (s1 ∧ ¬s2 ∧ s3 ∧ s4) ∨ (¬s1 ∧ s2 ∧ s3 ∧ s4)

theorem prove_statement_II : 
  (three_true_one_false (statement_I digit) (statement_II digit) (statement_III digit) (statement_IV digit)) → 
  statement_II digit :=
sorry

end prove_statement_II_l116_116423


namespace candidates_count_l116_116841

theorem candidates_count (n : ℕ) (h : n * (n - 1) = 42) : n = 7 :=
by sorry

end candidates_count_l116_116841


namespace scott_earnings_l116_116229

theorem scott_earnings
  (price_smoothie : ℝ)
  (price_cake : ℝ)
  (cups_sold : ℝ)
  (cakes_sold : ℝ)
  (earnings_smoothies : ℝ := cups_sold * price_smoothie)
  (earnings_cakes : ℝ := cakes_sold * price_cake) :
  price_smoothie = 3 → price_cake = 2 → cups_sold = 40 → cakes_sold = 18 → 
  (earnings_smoothies + earnings_cakes) = 156 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  sorry

end scott_earnings_l116_116229


namespace find_p_q_sum_l116_116694

-- Define the number of trees
def pine_trees := 2
def cedar_trees := 3
def fir_trees := 4

-- Total number of trees
def total_trees := pine_trees + cedar_trees + fir_trees

-- Number of ways to arrange the 9 trees
def total_arrangements := Nat.choose total_trees fir_trees

-- Number of ways to place fir trees so no two are adjacent
def valid_arrangements := Nat.choose (pine_trees + cedar_trees + 1) fir_trees

-- Desired probability in its simplest form
def probability := valid_arrangements / total_arrangements

-- Denominator and numerator of the simplified fraction
def num := 5
def den := 42

-- Statement to prove that the probability is 5/42
theorem find_p_q_sum : (num + den) = 47 := by
  sorry

end find_p_q_sum_l116_116694


namespace extra_pieces_of_gum_l116_116914

theorem extra_pieces_of_gum (total_packages : ℕ) (pieces_per_package : ℕ) (total_pieces : ℕ) : ℕ :=
  if total_packages = 43 ∧ pieces_per_package = 23 ∧ total_pieces = 997 then
    997 - (43 * 23)
  else
    0  -- This is a dummy value for other cases, as they do not satisfy our conditions.

#print extra_pieces_of_gum

end extra_pieces_of_gum_l116_116914


namespace cube_edge_length_l116_116270

theorem cube_edge_length (a : ℝ) (h : 6 * a^2 = 24) : a = 2 :=
by sorry

end cube_edge_length_l116_116270


namespace opera_house_rows_l116_116439

variable (R : ℕ)
variable (SeatsPerRow : ℕ)
variable (TicketPrice : ℕ)
variable (TotalEarnings : ℕ)
variable (SeatsTakenPercent : ℝ)

-- Conditions
axiom num_seats_per_row : SeatsPerRow = 10
axiom ticket_price : TicketPrice = 10
axiom total_earnings : TotalEarnings = 12000
axiom seats_taken_percent : SeatsTakenPercent = 0.8

-- Main theorem statement
theorem opera_house_rows
  (h1 : SeatsPerRow = 10)
  (h2 : TicketPrice = 10)
  (h3 : TotalEarnings = 12000)
  (h4 : SeatsTakenPercent = 0.8) :
  R = 150 :=
sorry

end opera_house_rows_l116_116439


namespace coefficient_of_friction_l116_116628

/-- Assume m, Pi and ΔL are positive real numbers, and g is the acceleration due to gravity. 
We need to prove that the coefficient of friction μ is given by Pi / (m * g * ΔL). --/
theorem coefficient_of_friction (m Pi ΔL g : ℝ) (h_m : 0 < m) (h_Pi : 0 < Pi) (h_ΔL : 0 < ΔL) (h_g : 0 < g) :
  ∃ μ : ℝ, μ = Pi / (m * g * ΔL) :=
sorry

end coefficient_of_friction_l116_116628


namespace Hamilton_marching_band_members_l116_116654

theorem Hamilton_marching_band_members (m : ℤ) (k : ℤ) :
  30 * m ≡ 5 [ZMOD 31] ∧ m = 26 + 31 * k ∧ 30 * m < 1500 → 30 * m = 780 :=
by
  intro h
  have hmod : 30 * m ≡ 5 [ZMOD 31] := h.left
  have m_eq : m = 26 + 31 * k := h.right.left
  have hlt : 30 * m < 1500 := h.right.right
  sorry

end Hamilton_marching_band_members_l116_116654


namespace amount_kept_by_Tim_l116_116793

-- Define the conditions
def totalAmount : ℝ := 100
def percentageGivenAway : ℝ := 0.2

-- Prove the question == answer
theorem amount_kept_by_Tim : totalAmount - totalAmount * percentageGivenAway = 80 :=
by
  -- Here the proof would take place
  sorry

end amount_kept_by_Tim_l116_116793


namespace hendricks_payment_l116_116108

variable (Hendricks Gerald : ℝ)
variable (less_percent : ℝ) (amount_paid : ℝ)

theorem hendricks_payment (h g : ℝ) (h_less_g : h = g * (1 - less_percent)) (g_val : g = amount_paid) (less_percent_val : less_percent = 0.2) (amount_paid_val: amount_paid = 250) :
h = 200 :=
by
  sorry

end hendricks_payment_l116_116108


namespace total_berries_l116_116537

theorem total_berries (S_stacy S_steve S_skylar : ℕ) 
  (h1 : S_stacy = 800)
  (h2 : S_stacy = 4 * S_steve)
  (h3 : S_steve = 2 * S_skylar) :
  S_stacy + S_steve + S_skylar = 1100 :=
by
  sorry

end total_berries_l116_116537


namespace part1_part2_l116_116663

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := (a * x^2 + b * x + 1) / Real.exp x
noncomputable def f' (a b : ℝ) (x : ℝ) : ℝ := (-a * x^2 + (2 * a - b) * x + b - 1) / Real.exp x

theorem part1 (a b : ℝ) (h : f a b (-1) + f' a b (-1) = 0) : b = 2 * a :=
sorry

theorem part2 (a : ℝ) (h : a ≤ 1 / 2) (x : ℝ) : f a (2 * a) (abs x) ≤ 1 :=
sorry

end part1_part2_l116_116663


namespace maddie_spent_in_all_l116_116128

-- Define the given conditions
def white_packs : ℕ := 2
def blue_packs : ℕ := 4
def t_shirts_per_white_pack : ℕ := 5
def t_shirts_per_blue_pack : ℕ := 3
def cost_per_t_shirt : ℕ := 3

-- Define the question as a theorem to be proved
theorem maddie_spent_in_all :
  (white_packs * t_shirts_per_white_pack + blue_packs * t_shirts_per_blue_pack) * cost_per_t_shirt = 66 :=
by 
  -- The proof goes here
  sorry

end maddie_spent_in_all_l116_116128


namespace bulgarian_inequality_l116_116112

theorem bulgarian_inequality (a b c d : ℝ) (a_pos : 0 < a) (b_pos : 0 < b) (c_pos : 0 < c) (d_pos : 0 < d) :
    (a^4 / (a^3 + a^2 * b + a * b^2 + b^3) + 
     b^4 / (b^3 + b^2 * c + b * c^2 + c^3) + 
     c^4 / (c^3 + c^2 * d + c * d^2 + d^3) + 
     d^4 / (d^3 + d^2 * a + d * a^2 + a^3)) 
    ≥ (a + b + c + d) / 4 :=
sorry

end bulgarian_inequality_l116_116112


namespace shift_parabola_l116_116343

theorem shift_parabola (x : ℝ) : 
  let y := -x^2
  let y_shifted_left := -((x + 3)^2)
  let y_shifted := y_shifted_left + 5
  y_shifted = -(x + 3)^2 + 5 := 
by {
  sorry
}

end shift_parabola_l116_116343


namespace fraction_arithmetic_l116_116255

theorem fraction_arithmetic : ((3 / 5 : ℚ) + (4 / 15)) * (2 / 3) = 26 / 45 := 
by
  sorry

end fraction_arithmetic_l116_116255


namespace child_ticket_cost_l116_116335

/-- Defining the conditions and proving the cost of a child's ticket --/
theorem child_ticket_cost:
  (∀ c: ℕ, 
      -- Revenue from Monday
      (7 * c + 5 * 4) + 
      -- Revenue from Tuesday
      (4 * c + 2 * 4) = 
      -- Total revenue for both days
      61 
    ) → 
    -- Proving c
    (c = 3) :=
by
  sorry

end child_ticket_cost_l116_116335


namespace count_sets_B_l116_116741

open Set

def A : Set ℕ := {1, 2}

theorem count_sets_B (B : Set ℕ) (h1 : A ∪ B = {1, 2, 3}) : 
  (∃ Bs : Finset (Set ℕ), ∀ b ∈ Bs, A ∪ b = {1, 2, 3} ∧ Bs.card = 4) := sorry

end count_sets_B_l116_116741


namespace min_value_I_is_3_l116_116259

noncomputable def min_value_I (a b c x y : ℝ) : ℝ :=
  1 / (2 * a^3 * x + b^3 * y^2) + 1 / (2 * b^3 * x + c^3 * y^2) + 1 / (2 * c^3 * x + a^3 * y^2)

theorem min_value_I_is_3 {a b c x y : ℝ} (h1 : a^6 + b^6 + c^6 = 3) (h2 : (x + 1)^2 + y^2 ≤ 2) :
  3 ≤ min_value_I a b c x y :=
sorry

end min_value_I_is_3_l116_116259


namespace sam_more_than_sarah_l116_116256

-- Defining the conditions
def street_width : ℤ := 25
def block_length : ℤ := 450
def block_width : ℤ := 350
def alleyway : ℤ := 25

-- Defining the distances run by Sarah and Sam
def sarah_long_side : ℤ := block_length + alleyway
def sarah_short_side : ℤ := block_width
def sam_long_side : ℤ := block_length + 2 * street_width
def sam_short_side : ℤ := block_width + 2 * street_width

-- Defining the total distance run by Sarah and Sam in one lap
def sarah_total_distance : ℤ := 2 * sarah_long_side + 2 * sarah_short_side
def sam_total_distance : ℤ := 2 * sam_long_side + 2 * sam_short_side

-- Proving the difference between Sam's and Sarah's running distances
theorem sam_more_than_sarah : sam_total_distance - sarah_total_distance = 150 := by
  -- The proof is omitted
  sorry

end sam_more_than_sarah_l116_116256


namespace greatest_sum_first_quadrant_l116_116950

theorem greatest_sum_first_quadrant (x y : ℤ) (hx_pos : 0 < x) (hy_pos : 0 < y) (h_circle : x^2 + y^2 = 49) : x + y ≤ 7 :=
sorry

end greatest_sum_first_quadrant_l116_116950


namespace vacation_expense_sharing_l116_116459

def alice_paid : ℕ := 90
def bob_paid : ℕ := 150
def charlie_paid : ℕ := 120
def donna_paid : ℕ := 240
def total_paid : ℕ := alice_paid + bob_paid + charlie_paid + donna_paid
def individual_share : ℕ := total_paid / 4

def alice_owes : ℕ := individual_share - alice_paid
def charlie_owes : ℕ := individual_share - charlie_paid
def donna_owes : ℕ := donna_paid - individual_share

def a : ℕ := charlie_owes
def b : ℕ := donna_owes - (donna_owes - charlie_owes)

theorem vacation_expense_sharing : a - b = 0 :=
by
  sorry

end vacation_expense_sharing_l116_116459


namespace find_numbers_l116_116029

theorem find_numbers (A B C D : ℚ) 
  (h1 : A + B = 44)
  (h2 : 5 * A = 6 * B)
  (h3 : C = 2 * (A - B))
  (h4 : D = (A + B + C) / 3 + 3) :
  A = 24 ∧ B = 20 ∧ C = 8 ∧ D = 61 / 3 := 
  by 
    sorry

end find_numbers_l116_116029


namespace simplify_expression_l116_116827

theorem simplify_expression : 
  let x := 2
  let y := -1 / 2
  (2 * x^2 + (-x^2 - 2 * x * y + 2 * y^2) - 3 * (x^2 - x * y + 2 * y^2)) = -10 := by
  sorry

end simplify_expression_l116_116827


namespace parallel_lines_l116_116671

theorem parallel_lines (a : ℝ) :
  (∀ x y : ℝ, x + a * y - (2 * a + 2) = 0 ∧ a * x + y - (a + 1) = 0 → (∀ x y : ℝ, (1 / a = a / 1) ∧ (1 / a ≠ (2 * -a - 2) / (1 * -a - 1)))) → a = 1 := by
sorry

end parallel_lines_l116_116671


namespace find_a_l116_116354

theorem find_a (a : ℝ) (x : ℝ) (h : ∀ (x : ℝ), 2 * x - a ≤ -1 ↔ x ≤ 1) : a = 3 :=
sorry

end find_a_l116_116354


namespace integer_pairs_l116_116062

def is_perfect_square (n : ℤ) : Prop :=
  ∃ m : ℤ, m * m = n

theorem integer_pairs (a b : ℤ) :
  (is_perfect_square (a^2 + 4 * b) ∧ is_perfect_square (b^2 + 4 * a)) ↔ 
  (a = 0 ∧ b = 0) ∨ (a = -4 ∧ b = -4) ∨ (a = 4 ∧ b = -4) ∨
  (∃ (k : ℕ), a = k^2 ∧ b = 0) ∨ (∃ (k : ℕ), a = 0 ∧ b = k^2) ∨
  (a = -6 ∧ b = -5) ∨ (a = -5 ∧ b = -6) ∨
  (∃ (t : ℕ), a = t ∧ b = 1 - t) ∨ (∃ (t : ℕ), a = 1 - t ∧ b = t) :=
sorry

end integer_pairs_l116_116062


namespace lucas_should_give_fraction_l116_116698

-- Conditions as Lean definitions
variables (n : ℕ) -- Number of shells Noah has
def Noah_shells := n
def Emma_shells := 2 * n -- Emma has twice as many shells as Noah
def Lucas_shells := 8 * n -- Lucas has four times as many shells as Emma

-- Desired distribution
def Total_shells := Noah_shells n + Emma_shells n + Lucas_shells n
def Each_person_shells := Total_shells n / 3

-- Fraction calculation
def Shells_needed_by_Emma := Each_person_shells n - Emma_shells n
def Fraction_of_Lucas_shells_given_to_Emma := Shells_needed_by_Emma n / Lucas_shells n 

theorem lucas_should_give_fraction :
  Fraction_of_Lucas_shells_given_to_Emma n = 5 / 24 := 
by
  sorry

end lucas_should_give_fraction_l116_116698


namespace arccos_sin_three_pi_over_two_eq_pi_l116_116204

theorem arccos_sin_three_pi_over_two_eq_pi : 
  Real.arccos (Real.sin (3 * Real.pi / 2)) = Real.pi :=
by
  sorry

end arccos_sin_three_pi_over_two_eq_pi_l116_116204


namespace matt_current_age_is_65_l116_116140

variable (matt_age james_age : ℕ)

def james_current_age := 30
def james_age_in_5_years := james_current_age + 5
def matt_age_in_5_years := 2 * james_age_in_5_years
def matt_current_age := matt_age_in_5_years - 5

theorem matt_current_age_is_65 : matt_current_age = 65 := 
by
  -- sorry is here to skip the proof.
  sorry

end matt_current_age_is_65_l116_116140


namespace min_value_inequality_l116_116321

theorem min_value_inequality (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h : x + y + z = 9) :
  (x^2 + y^2) / (x + y) + (x^2 + z^2) / (x + z) + (y^2 + z^2) / (y + z) ≥ 9 :=
sorry

end min_value_inequality_l116_116321


namespace area_of_smallest_square_l116_116530

theorem area_of_smallest_square (radius : ℝ) (h : radius = 6) : 
    ∃ s : ℝ, s = 2 * radius ∧ s^2 = 144 :=
by
  sorry

end area_of_smallest_square_l116_116530


namespace minimum_a_l116_116253

def f (x a : ℝ) : ℝ := x^2 - 2*x - abs (x-1-a) - abs (x-2) + 4

theorem minimum_a (a : ℝ) : (∀ x : ℝ, f x a ≥ 0) ↔ a = -2 :=
sorry

end minimum_a_l116_116253


namespace find_tire_price_l116_116450

def regular_price_of_tire (x : ℝ) : Prop :=
  3 * x + 0.75 * x = 270

theorem find_tire_price (x : ℝ) (h1 : regular_price_of_tire x) : x = 72 :=
by
  sorry

end find_tire_price_l116_116450


namespace factorize_1_factorize_2_factorize_3_l116_116653

theorem factorize_1 (x : ℝ) : x^4 - 9*x^2 = x^2 * (x + 3) * (x - 3) :=
sorry

theorem factorize_2 (x y : ℝ) : 25*x^2*y + 20*x*y^2 + 4*y^3 = y * (5*x + 2*y)^2 :=
sorry

theorem factorize_3 (x y a : ℝ) : x^2 * (a - 1) + y^2 * (1 - a) = (a - 1) * (x + y) * (x - y) :=
sorry

end factorize_1_factorize_2_factorize_3_l116_116653


namespace piecewise_function_not_composed_of_multiple_functions_l116_116557

theorem piecewise_function_not_composed_of_multiple_functions :
  ∀ (f : ℝ → ℝ), (∃ (I : ℝ → Prop) (f₁ f₂ : ℝ → ℝ),
    (∀ x, I x → f x = f₁ x) ∧ (∀ x, ¬I x → f x = f₂ x)) →
    ¬(∃ (g₁ g₂ : ℝ → ℝ), (∀ x, f x = g₁ x ∨ f x = g₂ x)) :=
by
  sorry

end piecewise_function_not_composed_of_multiple_functions_l116_116557


namespace num_people_got_on_bus_l116_116325

-- Definitions based on the conditions
def initialNum : ℕ := 4
def currentNum : ℕ := 17
def peopleGotOn (initial : ℕ) (current : ℕ) : ℕ := current - initial

-- Theorem statement
theorem num_people_got_on_bus : peopleGotOn initialNum currentNum = 13 := 
by {
  sorry -- Placeholder for the proof
}

end num_people_got_on_bus_l116_116325


namespace diorama_factor_l116_116697

theorem diorama_factor (P B factor : ℕ) (h1 : P + B = 67) (h2 : B = P * factor - 5) (h3 : B = 49) : factor = 3 :=
by
  sorry

end diorama_factor_l116_116697


namespace total_cost_of_returned_packets_l116_116506

/--
  Martin bought 10 packets of milk with varying prices.
  The average price (arithmetic mean) of all the packets is 25¢.
  If Martin returned three packets to the retailer, and the average price of the remaining packets was 20¢,
  then the total cost, in cents, of the three returned milk packets is 110¢.
-/
theorem total_cost_of_returned_packets 
  (T10 : ℕ) (T7 : ℕ) (average_price_10 : T10 / 10 = 25)
  (average_price_7 : T7 / 7 = 20) :
  (T10 - T7 = 110) := 
sorry

end total_cost_of_returned_packets_l116_116506


namespace complement_intersection_l116_116197

variable (U : Set ℕ) (A : Set ℕ) (B : Set ℕ)

#check (Set.compl B) ∩ A = {1}

theorem complement_intersection (hU : U = {1, 2, 3, 4, 5}) (hA : A = {1, 5}) (hB : B = {2, 3, 5}) :
   (U \ B) ∩ A = {1} :=
by
  sorry

end complement_intersection_l116_116197


namespace part1_part2_part3_l116_116026

def pointM (m : ℝ) : ℝ × ℝ := (m - 1, 2 * m + 3)

-- Part 1
theorem part1 (m : ℝ) (h : 2 * m + 3 = 0) : pointM m = (-5 / 2, 0) :=
  sorry

-- Part 2
theorem part2 (m : ℝ) (h : 2 * m + 3 = -1) : pointM m = (-3, -1) :=
  sorry

-- Part 3
theorem part3 (m : ℝ) (h1 : |m - 1| = 2) : pointM m = (2, 9) ∨ pointM m = (-2, 1) :=
  sorry

end part1_part2_part3_l116_116026


namespace units_digit_calculation_l116_116622

theorem units_digit_calculation : 
  ((33 * (83 ^ 1001) * (7 ^ 1002) * (13 ^ 1003)) % 10) = 9 :=
by
  sorry

end units_digit_calculation_l116_116622


namespace greatest_of_six_consecutive_mixed_numbers_l116_116995

theorem greatest_of_six_consecutive_mixed_numbers (A : ℚ) :
  let B := A + 1
  let C := A + 2
  let D := A + 3
  let E := A + 4
  let F := A + 5
  (A + B + C + D + E + F = 75.5) →
  F = 15 + 1/12 :=
by {
  sorry
}

end greatest_of_six_consecutive_mixed_numbers_l116_116995


namespace popsicle_stick_count_l116_116043

variable (Sam Sid Steve : ℕ)

def number_of_sticks (Sam Sid Steve : ℕ) : ℕ :=
  Sam + Sid + Steve

theorem popsicle_stick_count 
  (h1 : Sam = 3 * Sid)
  (h2 : Sid = 2 * Steve)
  (h3 : Steve = 12) :
  number_of_sticks Sam Sid Steve = 108 :=
by
  sorry

end popsicle_stick_count_l116_116043


namespace moles_of_Cl2_l116_116986

def chemical_reaction : Prop :=
  ∀ (CH4 Cl2 HCl : ℕ), 
  (CH4 = 1) → 
  (HCl = 4) →
  -- Given the balanced equation: CH4 + 2Cl2 → CHCl3 + 4HCl
  (CH4 + 2 * Cl2 = CH4 + 2 * Cl2) →
  (4 * HCl = 4 * HCl) → -- This asserts the product side according to the balanced equation
  (Cl2 = 2)

theorem moles_of_Cl2 (CH4 Cl2 HCl : ℕ) (hCH4 : CH4 = 1) (hHCl : HCl = 4)
  (h_balanced : CH4 + 2 * Cl2 = CH4 + 2 * Cl2) (h_product : 4 * HCl = 4 * HCl) :
  Cl2 = 2 := by {
    sorry
}

end moles_of_Cl2_l116_116986


namespace final_expression_in_simplest_form_l116_116645

variable (x : ℝ)

theorem final_expression_in_simplest_form : 
  ((3 * x + 6 - 5 * x + 10) / 5) = (-2 / 5) * x + 16 / 5 :=
by
  sorry

end final_expression_in_simplest_form_l116_116645


namespace find_m_value_l116_116059

def power_function_increasing (m : ℝ) : Prop :=
  (m^2 - m - 1 = 1) ∧ (m^2 - 2*m - 1 > 0)

theorem find_m_value (m : ℝ) (h : power_function_increasing m) : m = -1 :=
  sorry

end find_m_value_l116_116059


namespace additional_cost_per_person_l116_116057

-- Define the initial conditions and variables used in the problem
def base_cost := 1700
def discount_per_person := 50
def car_wash_earnings := 500
def initial_friends := 6
def final_friends := initial_friends - 1

-- Calculate initial cost per person with all friends
def discounted_base_cost_initial := base_cost - (initial_friends * discount_per_person)
def total_cost_after_car_wash_initial := discounted_base_cost_initial - car_wash_earnings
def cost_per_person_initial := total_cost_after_car_wash_initial / initial_friends

-- Calculate final cost per person after Brad leaves
def discounted_base_cost_final := base_cost - (final_friends * discount_per_person)
def total_cost_after_car_wash_final := discounted_base_cost_final - car_wash_earnings
def cost_per_person_final := total_cost_after_car_wash_final / final_friends

-- Proving the amount each friend has to pay more after Brad leaves
theorem additional_cost_per_person : cost_per_person_final - cost_per_person_initial = 40 := 
by
  sorry

end additional_cost_per_person_l116_116057


namespace time_to_see_slow_train_l116_116868

noncomputable def time_to_pass (length_fast_train length_slow_train relative_time_fast seconds_observed_by_slow : ℕ) : ℕ := 
  length_slow_train * seconds_observed_by_slow / length_fast_train

theorem time_to_see_slow_train :
  let length_fast_train := 150
  let length_slow_train := 200
  let seconds_observed_by_slow := 6
  let expected_time := 8
  time_to_pass length_fast_train length_slow_train length_fast_train seconds_observed_by_slow = expected_time :=
by sorry

end time_to_see_slow_train_l116_116868


namespace parabola_equation_l116_116163

theorem parabola_equation (P : ℝ × ℝ) (hp : P = (4, -2)) : 
  ∃ m : ℝ, (∀ x y : ℝ, (y^2 = m * x) → (x, y) = P) ∧ (m = 1) :=
by
  have m_val : 1 = 1 := rfl
  sorry

end parabola_equation_l116_116163


namespace carlson_fraction_jam_l116_116794

-- Definitions and conditions.
def total_time (T : ℕ) := T > 0
def time_maloish_cookies (t : ℕ) := t > 0
def equal_cookies (c : ℕ) := c > 0
def carlson_rate := 3

-- Let j_k and j_m be the amounts of jam eaten by Carlson and Maloish respectively.
def fraction_jam_carlson (j_k j_m : ℕ) : ℚ := j_k / (j_k + j_m)

-- The problem statement
theorem carlson_fraction_jam (T t c j_k j_m : ℕ)
  (hT : total_time T)
  (ht : time_maloish_cookies t)
  (hc : equal_cookies c)
  (h_carlson_rate : carlson_rate = 3)
  (h_equal_cookies : c > 0)  -- Both ate equal cookies
  (h_jam : j_k + j_m = j_k * 9 / 10 + j_m / 10) :
  fraction_jam_carlson j_k j_m = 9 / 10 :=
by
  sorry

end carlson_fraction_jam_l116_116794


namespace no_distinct_integers_cycle_l116_116644

theorem no_distinct_integers_cycle (p : ℤ → ℤ) 
  (x : ℕ → ℤ) (h_distinct : ∀ i j, i ≠ j → x i ≠ x j)
  (n : ℕ) (h_n_ge_3 : n ≥ 3)
  (hx_cycle : ∀ i, i < n → p (x i) = x (i + 1) % n) :
  false :=
sorry

end no_distinct_integers_cycle_l116_116644


namespace opposite_of_neg_one_third_l116_116584

theorem opposite_of_neg_one_third : -(-1/3) = 1/3 := 
sorry

end opposite_of_neg_one_third_l116_116584


namespace cary_strips_ivy_l116_116035

variable (strip_per_day : ℕ) (grow_per_night : ℕ) (total_ivy : ℕ)

theorem cary_strips_ivy (h1 : strip_per_day = 6) (h2 : grow_per_night = 2) (h3 : total_ivy = 40) :
  (total_ivy / (strip_per_day - grow_per_night)) = 10 := by
  sorry

end cary_strips_ivy_l116_116035


namespace A_share_correct_l116_116649

noncomputable def investment_shares (x : ℝ) (annual_gain : ℝ) := 
  let A_share := x * 12
  let B_share := (2 * x) * 6
  let C_share := (3 * x) * 4
  let total_share := A_share + B_share + C_share
  let total_ratio := 1 + 1 + 1
  annual_gain / total_ratio

theorem A_share_correct (x : ℝ) (annual_gain : ℝ) (h_gain : annual_gain = 18000) : 
  investment_shares x annual_gain / 3 = 6000 := by
  sorry

end A_share_correct_l116_116649


namespace parabola_y_axis_symmetry_l116_116689

theorem parabola_y_axis_symmetry (a b c d : ℝ) (r : ℝ) :
  (2019^2 + 2019 * a + b = 0) ∧ (2019^2 + 2019 * c + d = 0) ∧
  (a = -(2019 + r)) ∧ (c = -(2019 - r)) →
  b = -d :=
by
  sorry

end parabola_y_axis_symmetry_l116_116689


namespace number_of_math_books_l116_116773

theorem number_of_math_books (M H : ℕ) (h1 : M + H = 90) (h2 : 4 * M + 5 * H = 397) : M = 53 :=
by
  sorry

end number_of_math_books_l116_116773


namespace initial_bags_l116_116318

variable (b : ℕ)

theorem initial_bags (h : 5 * (b - 2) = 45) : b = 11 := 
by 
  sorry

end initial_bags_l116_116318


namespace determine_counterfeit_coin_l116_116761

theorem determine_counterfeit_coin (wt_1 wt_2 wt_3 wt_5 : ℕ) (coin : ℕ) :
  (wt_1 = 1) ∧ (wt_2 = 2) ∧ (wt_3 = 3) ∧ (wt_5 = 5) ∧
  (coin = wt_1 ∨ coin = wt_2 ∨ coin = wt_3 ∨ coin = wt_5) ∧
  (coin ≠ 1 ∨ coin ≠ 2 ∨ coin ≠ 3 ∨ coin ≠ 5) → 
  ∃ (counterfeit : ℕ), (counterfeit = 1 ∨ counterfeit = 2 ∨ counterfeit = 3 ∨ counterfeit = 5) ∧ 
  (counterfeit ≠ 1 ∧ counterfeit ≠ 2 ∧ counterfeit ≠ 3 ∧ counterfeit ≠ 5) :=
by
  sorry

end determine_counterfeit_coin_l116_116761


namespace find_XY_square_l116_116094

noncomputable def triangleABC := Type

variables (A B C T X Y : triangleABC)
variables (ω : Type) (BT CT BC TX TY XY : ℝ)

axiom acute_scalene_triangle (ABC : triangleABC) : Prop
axiom circumcircle (ABC: triangleABC) (ω: Type) : Prop
axiom tangents_intersect (ω: Type) (B C T: triangleABC) (BT CT : ℝ) : Prop
axiom projections (T: triangleABC) (X: triangleABC) (AB: triangleABC) (Y: triangleABC) (AC: triangleABC) : Prop

axiom BT_value : BT = 18
axiom CT_value : CT = 18
axiom BC_value : BC = 24
axiom TX_TY_XY_relation : TX^2 + TY^2 + XY^2 = 1450

theorem find_XY_square : XY^2 = 841 :=
by { sorry }

end find_XY_square_l116_116094


namespace common_difference_of_arithmetic_sequence_l116_116157

variable (a1 d : ℤ)
def S : ℕ → ℤ
| 0     => 0
| (n+1) => S n + (a1 + n * d)

theorem common_difference_of_arithmetic_sequence
  (h : 2 * S a1 d 3 = 3 * S a1 d 2 + 6) :
  d = 2 :=
  sorry

end common_difference_of_arithmetic_sequence_l116_116157


namespace range_of_a_l116_116769

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  a * Real.log x + (1/2) * x^2

theorem range_of_a (a : ℝ) (h : 0 < a) :
  (∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → 0 < x₁ → 0 < x₂ → (f a x₁ - f a x₂) / (x₁ - x₂) ≥ 2) ↔ (1 ≤ a) :=
by
  sorry

end range_of_a_l116_116769


namespace papaya_production_l116_116707

theorem papaya_production (P : ℕ)
  (h1 : 2 * P + 3 * 20 = 80) :
  P = 10 := 
by sorry

end papaya_production_l116_116707


namespace num_cubes_with_more_than_one_blue_face_l116_116980

-- Define the parameters of the problem
def block_length : ℕ := 5
def block_width : ℕ := 3
def block_height : ℕ := 1

def total_cubes : ℕ := 15
def corners : ℕ := 4
def edges : ℕ := 6
def middles : ℕ := 5

-- Define the condition that the total number of cubes painted on more than one face.
def cubes_more_than_one_blue_face : ℕ := corners + edges

-- Prove that the number of cubes painted on more than one face is 10
theorem num_cubes_with_more_than_one_blue_face :
  cubes_more_than_one_blue_face = 10 :=
by
  show (4 + 6) = 10
  sorry

end num_cubes_with_more_than_one_blue_face_l116_116980


namespace proof_part1_proof_part2_l116_116909

variable {a b c : ℝ}

def pos_and_sum_of_powers (a b c : ℝ) : Prop :=
  0 < a ∧ 0 < b ∧ 0 < c ∧ a^(3/2) + b^(3/2) + c^(3/2) = 1

theorem proof_part1 (h : pos_and_sum_of_powers a b c) : abc ≤ 1 / 9 :=
sorry

theorem proof_part2 (h : pos_and_sum_of_powers a b c) : 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (abc)) :=
sorry

end proof_part1_proof_part2_l116_116909


namespace right_triangle_with_a_as_hypotenuse_l116_116141

theorem right_triangle_with_a_as_hypotenuse
  (a b c : ℝ)
  (A B C : ℝ)
  (h1 : a = (b^2 + c^2 - a^2) / (2 * b * c))
  (h2 : b = (a^2 + c^2 - b^2) / (2 * a * c))
  (h3 : c = (a^2 + b^2 - c^2) / (2 * a * b))
  (h4 : a * ((b^2 + c^2 - a^2) / (2 * b * c)) + b * ((a^2 + c^2 - b^2) / (2 * a * c)) = c * ((a^2 + b^2 - c^2) / (2 * a * b))) :
  a^2 = b^2 + c^2 :=
by
  sorry

end right_triangle_with_a_as_hypotenuse_l116_116141


namespace condition1_condition2_condition3_l116_116173

noncomputable def Z (m : ℝ) : ℂ := (m^2 - 4 * m) + (m^2 - m - 6) * Complex.I

-- Condition 1: Point Z is in the third quadrant
theorem condition1 (m : ℝ) (h_quad3 : (m^2 - 4 * m) < 0 ∧ (m^2 - m - 6) < 0) : 0 < m ∧ m < 3 :=
sorry

-- Condition 2: Point Z is on the imaginary axis
theorem condition2 (m : ℝ) (h_imaginary : (m^2 - 4 * m) = 0 ∧ (m^2 - m - 6) ≠ 0) : m = 0 ∨ m = 4 :=
sorry

-- Condition 3: Point Z is on the line x - y + 3 = 0
theorem condition3 (m : ℝ) (h_line : (m^2 - 4 * m) - (m^2 - m - 6) + 3 = 0) : m = 3 :=
sorry

end condition1_condition2_condition3_l116_116173


namespace walking_rate_on_escalator_l116_116767

theorem walking_rate_on_escalator 
  (escalator_speed person_time : ℝ) 
  (escalator_length : ℝ) 
  (h1 : escalator_speed = 12) 
  (h2 : person_time = 15) 
  (h3 : escalator_length = 210) 
  : (∃ v : ℝ, escalator_length = (v + escalator_speed) * person_time ∧ v = 2) :=
by
  use 2
  rw [h1, h2, h3]
  sorry

end walking_rate_on_escalator_l116_116767


namespace sum_of_x_and_y_l116_116096

theorem sum_of_x_and_y (x y : ℤ) (h : 2 * x * y + x + y = 83) : x + y = 83 ∨ x + y = -85 :=
sorry

end sum_of_x_and_y_l116_116096


namespace tv_power_consumption_l116_116735

-- Let's define the problem conditions
def hours_per_day : ℕ := 4
def days_per_week : ℕ := 7
def weekly_cost : ℝ := 49              -- in cents
def cost_per_kwh : ℝ := 14             -- in cents

-- Define the theorem to prove the TV power consumption is 125 watts per hour
theorem tv_power_consumption : (weekly_cost / cost_per_kwh) / (hours_per_day * days_per_week) * 1000 = 125 :=
by
  sorry

end tv_power_consumption_l116_116735


namespace number_problem_l116_116660

theorem number_problem (x : ℤ) (h : (x - 5) / 7 = 7) : (x - 34) / 10 = 2 := by
  sorry

end number_problem_l116_116660


namespace largest_4digit_congruent_17_mod_28_l116_116203

theorem largest_4digit_congruent_17_mod_28 :
  ∃ n, n < 10000 ∧ n % 28 = 17 ∧ ∀ m, m < 10000 ∧ m % 28 = 17 → m ≤ 9982 :=
by
  sorry

end largest_4digit_congruent_17_mod_28_l116_116203
