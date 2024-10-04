import Mathlib

namespace ratio_of_roses_l288_288707

theorem ratio_of_roses (total_flowers tulips carnations roses : ℕ) 
  (h1 : total_flowers = 40) 
  (h2 : tulips = 10) 
  (h3 : carnations = 14) 
  (h4 : roses = total_flowers - (tulips + carnations)) :
  roses / total_flowers = 2 / 5 :=
by
  sorry

end ratio_of_roses_l288_288707


namespace factorize_xy2_minus_x_l288_288871

theorem factorize_xy2_minus_x (x y : ℝ) : xy^2 - x = x * (y - 1) * (y + 1) :=
by
  sorry

end factorize_xy2_minus_x_l288_288871


namespace tablecloth_radius_l288_288131

theorem tablecloth_radius (diameter : ℝ) (h : diameter = 10) : diameter / 2 = 5 :=
by {
  -- Outline the proof structure to ensure the statement is correct
  sorry
}

end tablecloth_radius_l288_288131


namespace john_swimming_improvement_l288_288301

theorem john_swimming_improvement :
  let initial_lap_time := 35 / 15 -- initial lap time in minutes per lap
  let current_lap_time := 33 / 18 -- current lap time in minutes per lap
  initial_lap_time - current_lap_time = 1 / 9 := 
by
  -- Definition of initial and current lap times are implied in Lean.
  sorry

end john_swimming_improvement_l288_288301


namespace distinct_divisors_sum_factorial_l288_288959

theorem distinct_divisors_sum_factorial (n : ℕ) (h : n ≥ 3) :
  ∃ (d : Fin n → ℕ), (∀ i j, i ≠ j → d i ≠ d j) ∧ (∀ i, d i ∣ n!) ∧ (n! = (Finset.univ.sum d)) :=
sorry

end distinct_divisors_sum_factorial_l288_288959


namespace problem_intersection_empty_l288_288133

open Set

noncomputable def A (m : ℝ) : Set ℝ := {x | x^2 + 2*x + m = 0}
def B : Set ℝ := {x | x > 0}

theorem problem_intersection_empty (m : ℝ) : (A m ∩ B = ∅) ↔ (0 ≤ m) :=
sorry

end problem_intersection_empty_l288_288133


namespace youngest_child_age_l288_288402

theorem youngest_child_age
  (ten_years_ago_avg_age : Nat) (family_initial_size : Nat) (present_avg_age : Nat)
  (age_difference : Nat) (age_ten_years_ago_total : Nat)
  (age_increase : Nat) (current_age_total : Nat)
  (current_family_size : Nat) (total_age_increment : Nat) :
  ten_years_ago_avg_age = 24 →
  family_initial_size = 4 →
  present_avg_age = 24 →
  age_difference = 2 →
  age_ten_years_ago_total = family_initial_size * ten_years_ago_avg_age →
  age_increase = family_initial_size * 10 →
  current_age_total = age_ten_years_ago_total + age_increase →
  current_family_size = family_initial_size + 2 →
  total_age_increment = current_family_size * present_avg_age →
  total_age_increment - current_age_total = 8 →
  ∃ (Y : Nat), Y + Y + age_difference = 8 ∧ Y = 3 :=
by
  intros
  sorry

end youngest_child_age_l288_288402


namespace alcohol_percentage_new_mixture_l288_288692

theorem alcohol_percentage_new_mixture (initial_volume new_volume alcohol_initial : ℝ)
  (h1 : initial_volume = 15)
  (h2 : alcohol_initial = 0.20 * initial_volume)
  (h3 : new_volume = initial_volume + 5) :
  (alcohol_initial / new_volume) * 100 = 15 := by
  sorry

end alcohol_percentage_new_mixture_l288_288692


namespace first_term_formula_correct_l288_288689

theorem first_term_formula_correct
  (S n d a : ℝ) 
  (h_sum_formula : S = (n / 2) * (2 * a + (n - 1) * d)) :
  a = (S / n) + (n - 1) * (d / 2) := 
sorry

end first_term_formula_correct_l288_288689


namespace gratuity_calculation_correct_l288_288227

noncomputable def tax_rate (item: String): ℝ :=
  if item = "NY Striploin" then 0.10
  else if item = "Glass of wine" then 0.15
  else if item = "Dessert" then 0.05
  else if item = "Bottle of water" then 0.00
  else 0

noncomputable def base_price (item: String): ℝ :=
  if item = "NY Striploin" then 80
  else if item = "Glass of wine" then 10
  else if item = "Dessert" then 12
  else if item = "Bottle of water" then 3
  else 0

noncomputable def total_price_with_tax (item: String): ℝ :=
  base_price item + base_price item * tax_rate item

noncomputable def gratuity (item: String): ℝ :=
  total_price_with_tax item * 0.20

noncomputable def total_gratuity: ℝ :=
  gratuity "NY Striploin" + gratuity "Glass of wine" + gratuity "Dessert" + gratuity "Bottle of water"

theorem gratuity_calculation_correct :
  total_gratuity = 23.02 :=
by
  sorry

end gratuity_calculation_correct_l288_288227


namespace cube_cross_section_area_l288_288225

def cube_edge_length (a : ℝ) := a > 0

def plane_perpendicular_body_diagonal := 
  ∃ (p : ℝ × ℝ × ℝ), ∀ (x y z : ℝ), 
  p = (x / 2, y / 2, z / 2) ∧ 
  (x + y + z) = (1 : ℝ)

theorem cube_cross_section_area
  (a : ℝ) 
  (h : cube_edge_length a) 
  (plane : plane_perpendicular_body_diagonal) : 
  ∃ (A : ℝ), 
  A = (3 * a^2 * Real.sqrt 3 / 4) := sorry

end cube_cross_section_area_l288_288225


namespace parallel_lines_slope_equal_l288_288005

theorem parallel_lines_slope_equal (k : ℝ) : (∀ x : ℝ, 2 * x = k * x + 3) → k = 2 :=
by
  intros
  sorry

end parallel_lines_slope_equal_l288_288005


namespace quotient_remainder_difference_l288_288091

theorem quotient_remainder_difference (N Q P R k : ℕ) (h1 : N = 75) (h2 : N = 5 * Q) (h3 : N = 34 * P + R) (h4 : Q = R + k) (h5 : k > 0) :
  Q - R = 8 :=
sorry

end quotient_remainder_difference_l288_288091


namespace other_root_of_quadratic_eq_l288_288953

theorem other_root_of_quadratic_eq (m : ℝ) (q : ℝ) :
  (∃ x : ℝ, x ≠ q ∧ 3 * x^2 + m * x - 7 = 0) →
  (3 * q^2 + m * q - 7 = 0) →
  q = -7 / 3 :=
by
  intro h
  sorry

end other_root_of_quadratic_eq_l288_288953


namespace cans_collected_on_first_day_l288_288393

-- Declare the main theorem
theorem cans_collected_on_first_day 
  (x : ℕ) -- Number of cans collected on the first day
  (total_cans : x + (x + 5) + (x + 10) + (x + 15) + (x + 20) = 150) :
  x = 20 :=
sorry

end cans_collected_on_first_day_l288_288393


namespace inscribed_sphere_radius_of_tetrahedron_l288_288897

variables (V S1 S2 S3 S4 R : ℝ)

theorem inscribed_sphere_radius_of_tetrahedron
  (hV_pos : 0 < V)
  (hS_pos : 0 < S1) (hS2_pos : 0 < S2) (hS3_pos : 0 < S3) (hS4_pos : 0 < S4) :
  R = 3 * V / (S1 + S2 + S3 + S4) :=
sorry

end inscribed_sphere_radius_of_tetrahedron_l288_288897


namespace average_computation_l288_288728

variable {a b c X Y Z : ℝ}

theorem average_computation 
  (h1 : a + b + c = 15)
  (h2 : X + Y + Z = 21) :
  ((2 * a + 3 * X) + (2 * b + 3 * Y) + (2 * c + 3 * Z)) / 3 = 31 :=
by
  sorry

end average_computation_l288_288728


namespace factorize_xy_squared_minus_x_l288_288880

theorem factorize_xy_squared_minus_x (x y : ℝ) : xy^2 - x = x * (y - 1) * (y + 1) :=
  sorry

end factorize_xy_squared_minus_x_l288_288880


namespace period_of_f_max_value_of_f_and_values_l288_288146

noncomputable def f (x : ℝ) : ℝ := (1 - Real.sin (2 * x)) / (Real.sin x - Real.cos x)

-- Statement 1: The period of f(x) is 2π
theorem period_of_f : ∀ x, f (x + 2 * Real.pi) = f x := by
  sorry

-- Statement 2: The maximum value of f(x) is √2 and it is attained at x = 2kπ + 3π/4, k ∈ ℤ
theorem max_value_of_f_and_values :
  (∀ x, f x ≤ Real.sqrt 2) ∧
  (∃ k : ℤ, f (2 * k * Real.pi + 3 * Real.pi / 4) = Real.sqrt 2) := by
  sorry

end period_of_f_max_value_of_f_and_values_l288_288146


namespace fg_of_5_eq_140_l288_288013

def g (x : ℝ) : ℝ := 4 * x + 5
def f (x : ℝ) : ℝ := 6 * x - 10

theorem fg_of_5_eq_140 : f (g 5) = 140 := by
  sorry

end fg_of_5_eq_140_l288_288013


namespace div_by_13_l288_288806

theorem div_by_13 (n : ℕ) (h : 0 < n) : 13 ∣ (4^(2*n - 1) + 3^(n + 1)) :=
by 
  sorry

end div_by_13_l288_288806


namespace system_equations_solution_exists_l288_288129

theorem system_equations_solution_exists (m : ℝ) :
  (∃ x y : ℝ, y = 3 * m * x + 6 ∧ y = (4 * m - 3) * x + 9) ↔ m ≠ 3 :=
by
  sorry

end system_equations_solution_exists_l288_288129


namespace geometric_sequence_common_ratio_l288_288621

theorem geometric_sequence_common_ratio (a_n : ℕ → ℝ) (q : ℝ) 
  (h1 : a_n 3 = a_n 2 * q) 
  (h2 : a_n 2 * q - 3 * a_n 2 = 2) 
  (h3 : 5 * a_n 4 = (12 * a_n 3 + 2 * a_n 5) / 2) : 
  q = 3 := 
by
  sorry

end geometric_sequence_common_ratio_l288_288621


namespace average_age_of_women_l288_288332

-- Defining the conditions
def average_age_of_men : ℝ := 40
def number_of_men : ℕ := 15
def increase_in_average : ℝ := 2.9
def ages_of_replaced_men : List ℝ := [26, 32, 41, 39]
def number_of_women : ℕ := 4

-- Stating the proof problem
theorem average_age_of_women :
  let total_age_of_men := average_age_of_men * number_of_men
  let total_age_of_replaced_men := ages_of_replaced_men.sum
  let new_average_age := average_age_of_men + increase_in_average
  let new_total_age_of_group := new_average_age * number_of_men
  let total_age_of_women := new_total_age_of_group - (total_age_of_men - total_age_of_replaced_men)
  let average_age_of_women := total_age_of_women / number_of_women
  average_age_of_women = 45.375 :=
sorry

end average_age_of_women_l288_288332


namespace min_value_z_l288_288259

variable (x y : ℝ)

theorem min_value_z : ∃ (x y : ℝ), 2 * x + 3 * y = 9 :=
sorry

end min_value_z_l288_288259


namespace total_growing_space_correct_l288_288847

-- Define the dimensions of the garden beds
def length_bed1 : ℕ := 3
def width_bed1 : ℕ := 3
def num_bed1 : ℕ := 2

def length_bed2 : ℕ := 4
def width_bed2 : ℕ := 3
def num_bed2 : ℕ := 2

-- Define the areas of the individual beds and total growing space
def area_bed1 : ℕ := length_bed1 * width_bed1
def total_area_bed1 : ℕ := area_bed1 * num_bed1

def area_bed2 : ℕ := length_bed2 * width_bed2
def total_area_bed2 : ℕ := area_bed2 * num_bed2

def total_growing_space : ℕ := total_area_bed1 + total_area_bed2

-- The theorem proving the total growing space
theorem total_growing_space_correct : total_growing_space = 42 := by
  sorry

end total_growing_space_correct_l288_288847


namespace find_line_equation_l288_288584

-- Define the first line equation
def line1 (x y : ℝ) : Prop := 2 * x - y - 5 = 0

-- Define the second line equation
def line2 (x y : ℝ) : Prop := x + y + 2 = 0

-- Define the parallel line equation with a variable constant term
def line_parallel (x y m : ℝ) : Prop := 3 * x + y + m = 0

-- State the intersection point
def intersect_point (x y : ℝ) : Prop := line1 x y ∧ line2 x y

-- The desired equation of the line passing through the intersection point
theorem find_line_equation (x y : ℝ) (h : intersect_point x y) : ∃ m, line_parallel x y m := by
  sorry

end find_line_equation_l288_288584


namespace deposit_increases_l288_288945

theorem deposit_increases (X r s : ℝ) (hX : 0 < X) (hr : 0 ≤ r) (hs : s < 20) : 
  r > 100 * s / (100 - s) :=
by sorry

end deposit_increases_l288_288945


namespace booksReadPerDay_l288_288040

-- Mrs. Hilt read 14 books in a week.
def totalBooksReadInWeek : ℕ := 14

-- There are 7 days in a week.
def daysInWeek : ℕ := 7

-- We need to prove that the number of books read per day is 2.
theorem booksReadPerDay :
  totalBooksReadInWeek / daysInWeek = 2 :=
by
  sorry

end booksReadPerDay_l288_288040


namespace problem_solution_l288_288903

noncomputable def f (x : ℝ) : ℝ := Real.exp (-(x - 1)^2)

def a : ℝ := f (Real.sqrt 2 / 2)
def b : ℝ := f (Real.sqrt 3 / 2)
def c : ℝ := f (Real.sqrt 6 / 2)

theorem problem_solution : b > c ∧ c > a := 
by 
  sorry

end problem_solution_l288_288903


namespace percentage_of_girls_l288_288976

theorem percentage_of_girls (B G : ℕ) (h1 : B + G = 900) (h2 : B = 90) :
  (G / (B + G) : ℚ) * 100 = 90 :=
  by
  sorry

end percentage_of_girls_l288_288976


namespace maximum_value_of_f_in_interval_l288_288789

noncomputable def f (x : ℝ) := (Real.sin x)^2 + (Real.sqrt 3) * Real.cos x - (3 / 4)

theorem maximum_value_of_f_in_interval : 
  ∃ x ∈ Set.Icc (0 : ℝ) (Real.pi / 2), f x = 1 := 
  sorry

end maximum_value_of_f_in_interval_l288_288789


namespace mark_score_is_46_l288_288292

theorem mark_score_is_46 (highest_score : ℕ) (range: ℕ) (mark_score : ℕ) :
  highest_score = 98 →
  range = 75 →
  (mark_score = 2 * (highest_score - range)) →
  mark_score = 46 := by
  intros
  sorry

end mark_score_is_46_l288_288292


namespace total_cost_l288_288708

variable (a b : ℝ)

def tomato_cost (a : ℝ) := 30 * a
def cabbage_cost (b : ℝ) := 50 * b

theorem total_cost (a b : ℝ) : 
  tomato_cost a + cabbage_cost b = 30 * a + 50 * b := 
by 
  unfold tomato_cost cabbage_cost
  sorry

end total_cost_l288_288708


namespace sequence_problem_l288_288436

theorem sequence_problem (S : ℕ → ℚ) (a : ℕ → ℚ) (h : ∀ n, S n + a n = 2 * n) :
  a 1 = 1 ∧ a 2 = 3 / 2 ∧ a 3 = 7 / 4 ∧ a 4 = 15 / 8 ∧ 
  (∀ n : ℕ, n > 0 → a n = (2^n - 1) / 2^(n-1)) :=
by
  sorry

end sequence_problem_l288_288436


namespace real_solutions_count_l288_288604

theorem real_solutions_count :
  ∃ n : ℕ, n = 2 ∧ ∀ x : ℝ, |x + 1| = |x - 3| + |x - 4| → x = 2 ∨ x = 8 :=
by
  sorry

end real_solutions_count_l288_288604


namespace select_2n_comparable_rectangles_l288_288458

def comparable (A B : Rectangle) : Prop :=
  -- A can be placed into B by translation and rotation
  exists f : Rectangle → Rectangle, f A = B

theorem select_2n_comparable_rectangles (n : ℕ) (h : n > 1) :
  ∃ (rectangles : List Rectangle), rectangles.length = 2 * n ∧
  ∀ (a b : Rectangle), a ∈ rectangles → b ∈ rectangles → comparable a b :=
sorry

end select_2n_comparable_rectangles_l288_288458


namespace kostyas_table_prime_l288_288765

theorem kostyas_table_prime (n : ℕ) (h₁ : n > 3) 
    (h₂ : ¬ ∃ r s : ℕ, r ≥ 3 ∧ s ≥ 3 ∧ n = r * s - (r + s)) : 
    Prime (n + 1) := 
sorry

end kostyas_table_prime_l288_288765


namespace continuous_stripe_probability_l288_288244

noncomputable def probability_continuous_stripe : ℚ :=
  let total_configurations := 4^6
  let favorable_configurations := 48
  favorable_configurations / total_configurations

theorem continuous_stripe_probability : probability_continuous_stripe = 3 / 256 :=
  by
  sorry

end continuous_stripe_probability_l288_288244


namespace factorize_xy_squared_minus_x_l288_288879

theorem factorize_xy_squared_minus_x (x y : ℝ) : xy^2 - x = x * (y - 1) * (y + 1) :=
  sorry

end factorize_xy_squared_minus_x_l288_288879


namespace range_of_a_l288_288595

theorem range_of_a {x y : ℝ} (hx : 0 < x) (hy : 0 < y) (h1 : x + y + 4 = 2 * x * y) (h2 : ∀ (x y : ℝ), x^2 + 2*x*y + y^2 - a*x - a*y + 1 ≥ 0) :
  a ≤ 17/4 := sorry

end range_of_a_l288_288595


namespace necessary_but_not_sufficient_condition_l288_288597

variables (p q : Prop)

theorem necessary_but_not_sufficient_condition
  (h : ¬p → q) (hn : ¬q → p) : 
  (p → ¬q) ∧ ¬(¬q → p) :=
sorry

end necessary_but_not_sufficient_condition_l288_288597


namespace set_star_result_l288_288242

-- Define the sets A and B
def A : Set ℕ := {1, 2}
def B : Set ℕ := {1, 2, 3}

-- Define the operation ∗ between sets A and B
def set_star (A B : Set ℕ) : Set ℕ := {x | ∃ x1 ∈ A, ∃ x2 ∈ B, x = x1 + x2}

-- Rewrite the main theorem to be proven
theorem set_star_result : set_star A B = {2, 3, 4, 5} :=
  sorry

end set_star_result_l288_288242


namespace rational_solutions_k_l288_288893

theorem rational_solutions_k (k : ℕ) (hpos : k > 0) : (∃ x : ℚ, k * x^2 + 22 * x + k = 0) ↔ k = 11 :=
by
  sorry

end rational_solutions_k_l288_288893


namespace triangles_in_divided_square_l288_288691

theorem triangles_in_divided_square (V E F : ℕ) 
  (hV : V = 24) 
  (h1 : 3 * F + 1 = 2 * E) 
  (h2 : V - E + F = 2) : F = 43 ∧ (F - 1 = 42) := 
by 
  have hF : F = 43 := sorry
  have hTriangles : F - 1 = 42 := sorry
  exact ⟨hF, hTriangles⟩

end triangles_in_divided_square_l288_288691


namespace uranus_appears_7_minutes_after_6AM_l288_288800

def mars_last_seen := 0 * 60 + 10 -- 12:10 AM in minutes after midnight
def jupiter_after_mars := 2 * 60 + 41 -- 2 hours and 41 minutes in minutes
def uranus_after_jupiter := 3 * 60 + 16 -- 3 hours and 16 minutes in minutes
def uranus_appearance := mars_last_seen + jupiter_after_mars + uranus_after_jupiter

theorem uranus_appears_7_minutes_after_6AM : uranus_appearance - (6 * 60) = 7 := by
  sorry

end uranus_appears_7_minutes_after_6AM_l288_288800


namespace calculate_mod_l288_288737

theorem calculate_mod
  (x : ℤ)
  (h : 4 * x + 9 ≡ 3 [ZMOD 19]) :
  3 * x + 8 ≡ 13 [ZMOD 19] :=
sorry

end calculate_mod_l288_288737


namespace finiteness_of_triples_l288_288639

theorem finiteness_of_triples (x : ℚ) : ∃! (a b c : ℤ), a < 0 ∧ b^2 - 4*a*c = 5 ∧ (a*x^2 + b*x + c > 0) := sorry

end finiteness_of_triples_l288_288639


namespace range_of_m_l288_288139

-- Definitions based on the conditions
def p (x : ℝ) : Prop := (x - 3) * (x + 1) > 0
def q (x : ℝ) (m : ℝ) : Prop := x^2 - 2 * x + 1 - m^2 > 0

theorem range_of_m (m : ℝ) (hm : m > 0) :
  (∀ x : ℝ, p x → q x m) ∧ (∃ x : ℝ, ¬(p x) ∧ q x m) ↔ 0 < m ∧ m ≤ 2 := sorry

end range_of_m_l288_288139


namespace eval_expression_at_values_l288_288715

theorem eval_expression_at_values : 
  ∀ x y : ℕ, x = 3 ∧ y = 4 → 
  5 * (x^(y+1)) + 6 * (y^(x+1)) + 2 * x * y = 2775 :=
by
  intros x y hxy
  cases hxy
  sorry

end eval_expression_at_values_l288_288715


namespace probability_of_pairing_with_friends_l288_288923

theorem probability_of_pairing_with_friends (n : ℕ) (f : ℕ) (h1 : n = 32) (h2 : f = 2):
  (f / (n - 1) : ℚ) = 2 / 31 :=
by
  rw [h1, h2]
  norm_num

end probability_of_pairing_with_friends_l288_288923


namespace find_x_l288_288085

theorem find_x (x : ℝ) (h : 45 * x = 0.60 * 900) : x = 12 :=
by
  sorry

end find_x_l288_288085


namespace average_abcd_l288_288057

-- Define the average condition of the numbers 4, 6, 9, a, b, c, d given as 20
def average_condition (a b c d : ℝ) : Prop :=
  (4 + 6 + 9 + a + b + c + d) / 7 = 20

-- Prove that the average of a, b, c, and d is 30.25 given the above condition
theorem average_abcd (a b c d : ℝ) (h : average_condition a b c d) : 
  (a + b + c + d) / 4 = 30.25 :=
by
  sorry

end average_abcd_l288_288057


namespace hotel_R_greater_than_G_l288_288654

variables (R G P : ℝ)

def hotel_charges_conditions :=
  P = 0.50 * R ∧ P = 0.80 * G

theorem hotel_R_greater_than_G :
  hotel_charges_conditions R G P → R = 1.60 * G :=
by
  sorry

end hotel_R_greater_than_G_l288_288654


namespace smallest_positive_perfect_square_divisible_by_2_3_5_l288_288385

theorem smallest_positive_perfect_square_divisible_by_2_3_5 : 
  ∃ (n : ℕ), (∃ k : ℕ, n = k^2) ∧ n % 2 = 0 ∧ n % 3 = 0 ∧ n % 5 = 0 ∧ 
  ∀ m : ℕ, (∃ k : ℕ, m = k^2) ∧ m % 2 = 0 ∧ m % 3 = 0 ∧ m % 5 = 0 → n ≤ m :=
sorry

end smallest_positive_perfect_square_divisible_by_2_3_5_l288_288385


namespace circles_intersect_twice_l288_288575

noncomputable def circle1 (x y : ℝ) : Prop :=
  (x - 3)^2 + y^2 = 9

noncomputable def circle2 (x y : ℝ) : Prop :=
  x^2 + (y - 1.5)^2 = 9 / 4

theorem circles_intersect_twice : 
  (∃ (p : ℝ × ℝ), circle1 p.1 p.2 ∧ circle2 p.1 p.2) ∧ 
  (∀ (p q : ℝ × ℝ), circle1 p.1 p.2 ∧ circle2 p.1 p.2 ∧ circle1 q.1 q.2 ∧ circle2 q.1 q.2 → (p = q ∨ p ≠ q)) →
  ∃ (p1 p2 : ℝ × ℝ), 
    p1 ≠ p2 ∧
    circle1 p1.1 p1.2 ∧ circle2 p1.1 p1.2 ∧
    circle1 p2.1 p2.2 ∧ circle2 p2.1 p2.2 := 
by {
  sorry
}

end circles_intersect_twice_l288_288575


namespace sqrt_fraction_l288_288809

theorem sqrt_fraction {a b c : ℝ}
  (h1 : a = Real.sqrt 27)
  (h2 : b = Real.sqrt 243)
  (h3 : c = Real.sqrt 48) :
  (a + b) / c = 3 := by
  sorry

end sqrt_fraction_l288_288809


namespace intersection_of_A_and_B_l288_288630

-- Define the sets A and B
def A := {x : ℝ | x ≥ 1}
def B := {x : ℝ | -1 < x ∧ x < 2}

-- Define the expected intersection
def expected_intersection := {x : ℝ | 1 ≤ x ∧ x < 2}

-- The proof problem statement
theorem intersection_of_A_and_B :
  A ∩ B = expected_intersection := by
  sorry

end intersection_of_A_and_B_l288_288630


namespace smallest_perfect_square_divisible_by_2_3_5_l288_288371

theorem smallest_perfect_square_divisible_by_2_3_5 :
  ∃ n : ℕ, (n > 0) ∧ (nat.sqrt n * nat.sqrt n = n) ∧ (2 ∣ n) ∧ (3 ∣ n) ∧ (5 ∣ n) ∧ n = 900 := sorry

end smallest_perfect_square_divisible_by_2_3_5_l288_288371


namespace flour_per_new_base_is_one_fifth_l288_288711

def total_flour : ℚ := 40 * (1 / 8)

def flour_per_new_base (p : ℚ) (total_flour : ℚ) : ℚ := total_flour / p

theorem flour_per_new_base_is_one_fifth :
  flour_per_new_base 25 total_flour = 1 / 5 :=
by
  sorry

end flour_per_new_base_is_one_fifth_l288_288711


namespace sum_of_primes_20_to_40_l288_288814

def is_prime (n : ℕ) : Prop :=
  n = 2 ∨ (2 < n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n)

def sum_of_primes_between (a b : ℕ) : ℕ :=
  (list.range' a (b - a)).filter is_prime |>.sum

theorem sum_of_primes_20_to_40 : sum_of_primes_between 20 40 = 120 := by
  sorry

end sum_of_primes_20_to_40_l288_288814


namespace sum_place_values_of_7s_l288_288538

theorem sum_place_values_of_7s (n : ℝ) (h : n = 87953.0727) : 
  let a := 7000
  let b := 0.07
  let c := 0.0007
  a + b + c = 7000.0707 :=
by
  sorry

end sum_place_values_of_7s_l288_288538


namespace exist_divisible_n_and_n1_l288_288317

theorem exist_divisible_n_and_n1 (d : ℕ) (hd : 0 < d) :
  ∃ (n n1 : ℕ), n % d = 0 ∧ n1 % d = 0 ∧ n ≠ n1 ∧
  (∃ (k a b c : ℕ), b ≠ 0 ∧ n = 10^k * (10 * a + b) + c ∧ n1 = 10^k * a + c) :=
by
  sorry

end exist_divisible_n_and_n1_l288_288317


namespace real_roots_condition_l288_288892

theorem real_roots_condition (k m : ℝ) (h : m ≠ 0) : (∃ x : ℝ, x^2 + k * x + m = 0) ↔ (m ≤ k^2 / 4) :=
by
  sorry

end real_roots_condition_l288_288892


namespace total_students_in_class_l288_288616

def students_chorus := 18
def students_band := 26
def students_both := 2
def students_neither := 8

theorem total_students_in_class : 
  (students_chorus + students_band - students_both) + students_neither = 50 := by
  sorry

end total_students_in_class_l288_288616


namespace simplify_fraction_l288_288648

theorem simplify_fraction :
  (18 / 462) + (35 / 77) = 38 / 77 := 
by sorry

end simplify_fraction_l288_288648


namespace probability_gather_info_both_workshops_l288_288754

theorem probability_gather_info_both_workshops :
  ∃ (p : ℚ), p = 56 / 62 :=
by
  sorry

end probability_gather_info_both_workshops_l288_288754


namespace trigonometric_identity_l288_288721

open Real

theorem trigonometric_identity (α β : ℝ) (h : 2 * cos (2 * α + β) - 3 * cos β = 0) :
  tan α * tan (α + β) = -1 / 5 := 
by {
  sorry
}

end trigonometric_identity_l288_288721


namespace combined_population_port_perry_lazy_harbor_l288_288969

theorem combined_population_port_perry_lazy_harbor 
  (PP LH W : ℕ)
  (h1 : PP = 7 * W)
  (h2 : PP = LH + 800)
  (h3 : W = 900) :
  PP + LH = 11800 :=
by
  sorry

end combined_population_port_perry_lazy_harbor_l288_288969


namespace find_s_l288_288032

theorem find_s (n r s c d : ℝ) (h1 : c^2 - n * c + 3 = 0) (h2 : d^2 - n * d + 3 = 0) 
  (h3 : (c + 1/d)^2 - r * (c + 1/d) + s = 0) (h4 : (d + 1/c)^2 - r * (d + 1/c) + s = 0) 
  (h5 : c * d = 3) : s = 16 / 3 := 
by
  sorry

end find_s_l288_288032


namespace intersection_of_sets_l288_288341

def A : Set ℝ := { x | 1 ≤ x ∧ x ≤ 3 }
def B : Set ℝ := { x | 2 < x ∧ x < 4 }

theorem intersection_of_sets : A ∩ B = { x | 2 < x ∧ x ≤ 3 } := 
by 
  sorry

end intersection_of_sets_l288_288341


namespace find_value_l288_288134

noncomputable def a : ℝ := 5 - 2 * Real.sqrt 6

theorem find_value :
  a^2 - 10 * a + 1 = 0 :=
by
  -- Since we are only required to write the statement, add sorry to skip the proof.
  sorry

end find_value_l288_288134


namespace primes_between_2_and_100_l288_288174

open Nat

theorem primes_between_2_and_100 :
  { p : ℕ | 2 ≤ p ∧ p ≤ 100 ∧ Nat.Prime p } = 
  {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97} :=
by
  sorry

end primes_between_2_and_100_l288_288174


namespace quadratic_square_binomial_l288_288206

theorem quadratic_square_binomial (k : ℝ) : (∃ b : ℝ, x^2 - 18 * x + k = (x + b) ^ 2) ↔ k = 81 := by
  sorry

end quadratic_square_binomial_l288_288206


namespace cost_per_pack_is_correct_l288_288086

def total_amount_spent : ℝ := 120
def num_packs_bought : ℕ := 6
def expected_cost_per_pack : ℝ := 20

theorem cost_per_pack_is_correct :
  total_amount_spent / num_packs_bought = expected_cost_per_pack :=
  by 
    -- here would be the proof
    sorry

end cost_per_pack_is_correct_l288_288086


namespace divisors_not_multiples_of_14_l288_288035

def is_perfect_square (x : ℕ) : Prop := ∃ k : ℕ, x = k ^ 2
def is_perfect_cube (x : ℕ) : Prop := ∃ k : ℕ, x = k ^ 3
def is_perfect_fifth (x : ℕ) : Prop := ∃ k : ℕ, x = k ^ 5
def is_perfect_seventh (x : ℕ) : Prop := ∃ k : ℕ, x = k ^ 7

def n : ℕ := 2^2 * 3^3 * 5^5 * 7^7

theorem divisors_not_multiples_of_14 :
  is_perfect_square (n / 2) →
  is_perfect_cube (n / 3) →
  is_perfect_fifth (n / 5) →
  is_perfect_seventh (n / 7) →
  (∃ d : ℕ, d = 240) :=
by
  sorry

end divisors_not_multiples_of_14_l288_288035


namespace even_function_a_value_l288_288285

def f (x a : ℝ) : ℝ := (x + 1) * (x - a)

theorem even_function_a_value (a : ℝ) (h : ∀ x : ℝ, f x a = f (-x) a) : a = 1 := by
  sorry

end even_function_a_value_l288_288285


namespace total_mustard_bottles_l288_288568

theorem total_mustard_bottles : 
  let table1 : ℝ := 0.25
  let table2 : ℝ := 0.25
  let table3 : ℝ := 0.38
  table1 + table2 + table3 = 0.88 :=
by
  sorry

end total_mustard_bottles_l288_288568


namespace three_divides_two_pow_n_plus_one_l288_288431

theorem three_divides_two_pow_n_plus_one (n : ℕ) (hn : n > 0) : 
  (3 ∣ 2^n + 1) ↔ Odd n := 
sorry

end three_divides_two_pow_n_plus_one_l288_288431


namespace jennifer_initial_money_eq_120_l288_288024

variable (X : ℚ) -- Define X as a rational number

-- Declare the conditions as variables.
variable (sandwich_expense museum_expense book_expense leftover_money : ℚ)

-- Set definitions based on the problem conditions.
def sandwich_expense := (1 / 5) * X
def museum_expense := (1 / 6) * X
def book_expense := (1 / 2) * X
def leftover_money := 16

-- Define the main theorem to prove.
theorem jennifer_initial_money_eq_120 
  (h: X - (sandwich_expense + museum_expense + book_expense) = leftover_money) : 
  X = 120 :=
by
  -- Proof omitted, add "sorry" to indicate the proof is required but not provided.
  sorry

end jennifer_initial_money_eq_120_l288_288024


namespace solution_of_abs_square_eq_zero_l288_288278

-- Define the given conditions as hypotheses
variables {x y : ℝ}
theorem solution_of_abs_square_eq_zero (h : |x + 2| + (y - 1)^2 = 0) : x = -2 ∧ y = 1 :=
sorry

end solution_of_abs_square_eq_zero_l288_288278


namespace max_volume_of_cylinder_max_volume_is_max_l288_288495

open Real

noncomputable def max_volume_cylinder (h : ℝ) : ℝ :=
  if h ≤ 3 / 2 then (π * (sqrt 3 / 4) ^ 2 * h) else (π / 12 * h * (2 - h / 3) ^ 2)

theorem max_volume_of_cylinder (h : ℝ) (h_pos : 0 < h) (h_le_3 : h ≤ 3) :
  max_volume_cylinder h = if h ≤ 3 / 2 then (π * (sqrt 3 / 4) ^ 2 * h) else (π / 12 * h * (2 - h / 3) ^ 2) :=
by
  sorry

noncomputable def max_volume_among_all : ℝ := π * 8 / 27

theorem max_volume_is_max (V : ℝ) : V = max_volume_among_all ↔ V = π * 8 / 27 :=
by
  sorry

end max_volume_of_cylinder_max_volume_is_max_l288_288495


namespace ones_digit_9_pow_53_l288_288522

theorem ones_digit_9_pow_53 :
  (9 ^ 53) % 10 = 9 :=
by
  sorry

end ones_digit_9_pow_53_l288_288522


namespace probability_either_condition_1_or_2_eq_43_over_128_l288_288529

noncomputable def prob_event_condition_1_2 (balls : Fin 8 → Bool) : Prop :=
  let condition_1 := (∑ i, if balls i then 1 else 0) = 4
  let condition_2 := (∑ i, if balls i then 1 else 0) = 1 ∨ (∑ i, if balls i then 1 else 0) = 7
  condition_1 ∨ condition_2

theorem probability_either_condition_1_or_2_eq_43_over_128 :
  let prob : Probability (Fin 8 → Bool) := classical.some (Probability.uniform (Univ : set (Fin 8 → Bool))) in
  Probability.prob_event prob prob_event_condition_1_2 = 43 / 128 := 
sorry

end probability_either_condition_1_or_2_eq_43_over_128_l288_288529


namespace diving_competition_score_l288_288534

theorem diving_competition_score 
  (scores : List ℝ)
  (h : scores = [7.5, 8.0, 9.0, 6.0, 8.8])
  (degree_of_difficulty : ℝ)
  (hd : degree_of_difficulty = 3.2) :
  let sorted_scores := scores.erase 9.0 |>.erase 6.0
  let remaining_sum := sorted_scores.sum
  remaining_sum * degree_of_difficulty = 77.76 :=
by
  sorry

end diving_competition_score_l288_288534


namespace probability_two_dice_sum_gt_8_l288_288199

def num_ways_to_get_sum_at_most_8 := 
  1 + 2 + 3 + 4 + 5 + 6 + 5

def total_outcomes := 36

def probability_sum_greater_than_8 : ℚ := 1 - (num_ways_to_get_sum_at_most_8 / total_outcomes)

theorem probability_two_dice_sum_gt_8 :
  probability_sum_greater_than_8 = 5 / 18 :=
by
  sorry

end probability_two_dice_sum_gt_8_l288_288199


namespace part1_part2_part3_l288_288143

-- Definitions for conditions used in the proof problems
def eq1 (a b : ℝ) : Prop := 2 * a + b = 0
def eq2 (a x : ℝ) : Prop := x = a ^ 2

-- Part 1: Prove b = 4 and x = 4 given a = -2
theorem part1 (a b x : ℝ) (h1 : a = -2) (h2 : eq1 a b) (h3 : eq2 a x) : b = 4 ∧ x = 4 :=
by sorry

-- Part 2: Prove a = -3 and x = 9 given b = 6
theorem part2 (a b x : ℝ) (h1 : b = 6) (h2 : eq1 a b) (h3 : eq2 a x) : a = -3 ∧ x = 9 :=
by sorry

-- Part 3: Prove x = 2 given a^2*x + (a + b)^2*x = 8
theorem part3 (a b x : ℝ) (h : a^2 * x + (a + b)^2 * x = 8) : x = 2 :=
by sorry

end part1_part2_part3_l288_288143


namespace necessary_but_not_sufficient_l288_288305

def simple_prop (p q : Prop) :=
  (¬ (p ∧ q)) → (¬ (p ∨ q))

theorem necessary_but_not_sufficient (p q : Prop) (h : simple_prop p q) :
  ((¬ (p ∧ q)) → (¬ (p ∨ q))) ∧ ¬ ((¬ (p ∨ q)) → (¬ (p ∧ q))) := by
sorry

end necessary_but_not_sufficient_l288_288305


namespace sin_a6_plus_b6_eq_neg_half_l288_288933

open Real

theorem sin_a6_plus_b6_eq_neg_half {a : ℕ → ℝ} {b : ℕ → ℝ}
  (hS11 : S 11 = (11 / 3) * π)
  (hS11_def : S 11 = 11 * a 6)
  (hb4b8_eq : ∀ x, 4 * x ^ 2 + 100 * x + π ^ 2 = 0 → x = b 4 ∨ x = b 8)
  (hb4b8_sum : b 4 + b 8 = -25)
  (hb4b8_prod : b 4 * b 8 = (π ^ 2) / 4)
  (hb4_neg : b 4 < 0)
  (hb8_neg : b 8 < 0) :
  sin (a 6 + b 6) = -1 / 2 :=
by
  sorry

end sin_a6_plus_b6_eq_neg_half_l288_288933


namespace smallest_positive_debt_resolvable_l288_288521

theorem smallest_positive_debt_resolvable :
  ∃ D : ℤ, D > 0 ∧ (D = 250 * p + 175 * g + 125 * s ∧ 
  (∀ (D' : ℤ), D' > 0 → (∃ p g s : ℤ, D' = 250 * p + 175 * g + 125 * s) → D' ≥ D)) := 
sorry

end smallest_positive_debt_resolvable_l288_288521


namespace smallest_positive_perfect_square_divisible_by_2_3_5_l288_288379

/-- Problem statement: 
    The smallest positive perfect square that is divisible by 2, 3, and 5 is 900.
-/
theorem smallest_positive_perfect_square_divisible_by_2_3_5 : 
  ∃ n : ℕ, 0 < n ∧ n * n % 2 = 0 ∧ n * n % 3 = 0 ∧ n * n % 5 = 0 ∧ n * n = 900 := 
by
  sorry

end smallest_positive_perfect_square_divisible_by_2_3_5_l288_288379


namespace find_f_7_over_2_l288_288262

section
variable {f : ℝ → ℝ}

-- Conditions
axiom odd_fn : ∀ x : ℝ, f (-x) = -f (x)
axiom even_shift_fn : ∀ x : ℝ, f (x + 1) = f (1 - x)
axiom range_x : Π x : ℝ, -1 ≤ x ∧ x ≤ 0 → f (x) = 2 * x^2

-- Prove that f(7/2) = 1/2
theorem find_f_7_over_2 : f (7 / 2) = 1 / 2 :=
sorry
end

end find_f_7_over_2_l288_288262


namespace right_triangle_area_l288_288092

theorem right_triangle_area (a b c : ℝ) (hypotenuse : ℝ) 
  (h_angle_sum : a = 45) (h_other_angle : b = 45) (h_right_angle : c = 90)
  (h_altitude : ∃ height : ℝ, height = 4) :
  ∃ area : ℝ, area = 8 := 
by
  sorry

end right_triangle_area_l288_288092


namespace average_class_weight_l288_288192

theorem average_class_weight
  (n_boys n_girls n_total : ℕ)
  (avg_weight_boys avg_weight_girls total_students : ℕ)
  (h1 : n_boys = 15)
  (h2 : n_girls = 10)
  (h3 : n_total = 25)
  (h4 : avg_weight_boys = 48)
  (h5 : avg_weight_girls = 405 / 10) 
  (h6 : total_students = 25) :
  (48 * 15 + 40.5 * 10) / 25 = 45 := 
sorry

end average_class_weight_l288_288192


namespace not_perfect_square_l288_288045

theorem not_perfect_square (n : ℕ) : ¬ ∃ k : ℕ, k^2 = 3^n + 2 * 17^n := sorry

end not_perfect_square_l288_288045


namespace Louie_monthly_payment_l288_288946

noncomputable def monthly_payment (P : ℕ) (r : ℚ) (n t : ℕ) : ℚ :=
  (P : ℚ) * (1 + r / n)^(n * t) / t

theorem Louie_monthly_payment : 
  monthly_payment 2000 0.10 1 3 = 887 := 
by
  sorry

end Louie_monthly_payment_l288_288946


namespace check_interval_of_quadratic_l288_288255

theorem check_interval_of_quadratic (z : ℝ) : (z^2 - 40 * z + 344 ≤ 0) ↔ (20 - 2 * Real.sqrt 14 ≤ z ∧ z ≤ 20 + 2 * Real.sqrt 14) :=
sorry

end check_interval_of_quadratic_l288_288255


namespace gcd_polynomials_l288_288439

theorem gcd_polynomials (b : ℕ) (hb : ∃ k : ℕ, b = 2 * 7771 * k) :
  Int.gcd (8 * b ^ 2 + 55 * b + 144) (4 * b + 19) = 8 :=
by sorry

end gcd_polynomials_l288_288439


namespace ratio_of_allergic_to_peanut_to_total_l288_288463

def total_children : ℕ := 34
def children_not_allergic_to_cashew : ℕ := 10
def children_allergic_to_both : ℕ := 10
def children_allergic_to_cashew : ℕ := 18
def children_not_allergic_to_any : ℕ := 6
def children_allergic_to_peanut : ℕ := 20

theorem ratio_of_allergic_to_peanut_to_total :
  (children_allergic_to_peanut : ℚ) / (total_children : ℚ) = 10 / 17 :=
by
  sorry

end ratio_of_allergic_to_peanut_to_total_l288_288463


namespace range_of_m_l288_288144

theorem range_of_m (x y m : ℝ) (h1 : x > 0) (h2 : y > 0)
  (h3 : (2 / x) + (3 / y) = 1)
  (h4 : 3 * x + 2 * y > m^2 + 2 * m) :
  -6 < m ∧ m < 4 :=
sorry

end range_of_m_l288_288144


namespace acid_solution_l288_288519

theorem acid_solution (m x : ℝ) (h1 : 0 < m) (h2 : m > 50)
  (h3 : (m / 100) * m = (m - 20) / 100 * (m + x)) : x = 20 * m / (m + 20) := 
sorry

end acid_solution_l288_288519


namespace remaining_standby_time_l288_288855

variable (fully_charged_standby : ℝ) (fully_charged_gaming : ℝ)
variable (standby_time : ℝ) (gaming_time : ℝ)

theorem remaining_standby_time
  (h1 : fully_charged_standby = 10)
  (h2 : fully_charged_gaming = 2)
  (h3 : standby_time = 4)
  (h4 : gaming_time = 1.5) :
  (10 - ((standby_time * (1 / fully_charged_standby)) + (gaming_time * (1 / fully_charged_gaming)))) * 10 = 1 :=
by
  sorry

end remaining_standby_time_l288_288855


namespace remainder_of_12345678910_div_101_l288_288995

theorem remainder_of_12345678910_div_101 :
  12345678910 % 101 = 31 :=
sorry

end remainder_of_12345678910_div_101_l288_288995


namespace gcd_fib_2017_99_101_plus_1_eq_1_l288_288444

def fib : ℕ → ℕ
| 0     := 0
| 1     := 1
| (n+2) := fib (n+1) + fib n

theorem gcd_fib_2017_99_101_plus_1_eq_1 :
  gcd (fib 2017) (fib (99) * fib (101) + 1) = 1 :=
sorry

end gcd_fib_2017_99_101_plus_1_eq_1_l288_288444


namespace min_value_of_2x_plus_y_l288_288592

theorem min_value_of_2x_plus_y {x y : ℝ} (hx : x > 0) (hy : y > 0)
  (h : 1 / (x + 1) + 8 / y = 2) : 2 * x + y ≥ 7 :=
sorry

end min_value_of_2x_plus_y_l288_288592


namespace integer_roots_polynomial_l288_288888

theorem integer_roots_polynomial (a : ℤ) :
  (∃ x : ℤ, x^3 + 3 * x^2 + a * x + 9 = 0) ↔ 
  (a = -109 ∨ a = -21 ∨ a = -13 ∨ a = 3 ∨ a = 11 ∨ a = 53) :=
by
  sorry

end integer_roots_polynomial_l288_288888


namespace fifty_third_card_is_A_s_l288_288120

def sequence_position (n : ℕ) : String :=
  let cycle_length := 26
  let pos_in_cycle := (n - 1) % cycle_length + 1
  if pos_in_cycle <= 13 then
    "A_s"
  else
    "A_h"

theorem fifty_third_card_is_A_s : sequence_position 53 = "A_s" := by
  sorry  -- proof placeholder

end fifty_third_card_is_A_s_l288_288120


namespace number_of_children_l288_288515

-- Definitions based on conditions from the problem
def total_spectators := 10000
def men_spectators := 7000
def spectators_other_than_men := total_spectators - men_spectators
def women_and_children_ratio := 5

-- Prove there are 2500 children
theorem number_of_children : 
  ∃ (women children : ℕ), 
    spectators_other_than_men = women + women_and_children_ratio * women ∧ 
    children = women_and_children_ratio * women ∧
    children = 2500 :=
by
  sorry

end number_of_children_l288_288515


namespace emma_time_l288_288864

theorem emma_time (E : ℝ) (h1 : 2 * E + E = 60) : E = 20 :=
sorry

end emma_time_l288_288864


namespace print_output_l288_288390

-- Conditions
def a : Nat := 10

/-- The print statement with the given conditions should output "a=10" -/
theorem print_output : "a=" ++ toString a = "a=10" :=
sorry

end print_output_l288_288390


namespace point_of_tangency_l288_288701

theorem point_of_tangency : 
    ∃ (m n : ℝ), 
    (∀ x : ℝ, x ≠ 0 → n = 1 / m ∧ (-1 / m^2) = (n - 2) / (m - 0)) ∧ 
    m = 1 ∧ n = 1 :=
by
  sorry

end point_of_tangency_l288_288701


namespace sphere_views_identical_l288_288574

-- Define the geometric shape as a type
inductive GeometricShape
| sphere
| cube
| other (name : String)

-- Define a function to get the view of a sphere
def view (s : GeometricShape) (direction : String) : String :=
  match s with
  | GeometricShape.sphere => "circle"
  | GeometricShape.cube => "square"
  | GeometricShape.other _ => "unknown"

-- The theorem to prove that a sphere has identical front, top, and side views
theorem sphere_views_identical :
  ∀ (direction1 direction2 : String), view GeometricShape.sphere direction1 = view GeometricShape.sphere direction2 :=
by
  intros direction1 direction2
  sorry

end sphere_views_identical_l288_288574


namespace slope_of_line_AB_l288_288667

-- Define the points A and B
def A : ℝ × ℝ := (0, -1)
def B : ℝ × ℝ := (2, 4)

-- State the proposition that we need to prove
theorem slope_of_line_AB :
  let slope := (B.2 - A.2) / (B.1 - A.1)
  slope = 5 / 2 := by
  sorry

end slope_of_line_AB_l288_288667


namespace product_simplifies_l288_288570

theorem product_simplifies :
  (6 * (1/2) * (3/4) * (1/5) = (9/20)) :=
by
  sorry

end product_simplifies_l288_288570


namespace tan_square_B_eq_tan_A_tan_C_range_l288_288599

theorem tan_square_B_eq_tan_A_tan_C_range (A B C : ℝ) (h_triangle : 0 < A ∧ A < π / 2 ∧ 0 < B ∧ B < π / 2 ∧ 0 < C ∧ C < π / 2 ∧ A + B + C = π) 
  (h_tan : Real.tan B * Real.tan B = Real.tan A * Real.tan C) : (π / 3) ≤ B ∧ B < (π / 2) :=
by
  sorry

end tan_square_B_eq_tan_A_tan_C_range_l288_288599


namespace range_of_m_l288_288895

-- Define sets A and B
def A := {x : ℝ | x ≤ 1}
def B (m : ℝ) := {x : ℝ | x ≤ m}

-- Statement: Prove the range of m such that B ⊆ A
theorem range_of_m (m : ℝ) : (∀ x, x ∈ B m → x ∈ A) ↔ (m ≤ 1) :=
by sorry

end range_of_m_l288_288895


namespace exists_set_B_l288_288475

open Finset

theorem exists_set_B {n : ℕ} (hn : n ≥ 2) (A : Finset ℕ) (S := Finset.Icc 2 n) (k := S.filter Nat.Prime).card
  (hA_sub_S : A ⊆ S) (hA_card : A.card ≤ k) (hA_no_div : ∀ {x y}, x ∈ A → y ∈ A → x ≠ y → ¬(x ∣ y) ∧ ¬(y ∣ x)) :
  ∃ B, B.card = k ∧ A ⊆ B ∧ B ⊆ S ∧ ∀ {x y}, x ∈ B → y ∈ B → x ≠ y → ¬(x ∣ y) ∧ ¬(y ∣ x) :=
  sorry

end exists_set_B_l288_288475


namespace unique_prime_solution_l288_288428

-- Define the variables and properties
def is_prime (n : ℕ) := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Define the proof goal
theorem unique_prime_solution (p q : ℕ) (hp : is_prime p) (hq : is_prime q) (hp_pos : 0 < p) (hq_pos : 0 < q) :
  p^2 - q^3 = 1 → (p = 3 ∧ q = 2) :=
by sorry

end unique_prime_solution_l288_288428


namespace find_solutions_l288_288891

noncomputable def solution_exists (x y z p : ℝ) : Prop :=
  (x^2 - 1 = p * (y + z)) ∧
  (y^2 - 1 = p * (z + x)) ∧
  (z^2 - 1 = p * (x + y))

theorem find_solutions (x y z p : ℝ) :
  solution_exists x y z p ↔
  (x = (p + Real.sqrt (p^2 + 1)) ∧ y = (p + Real.sqrt (p^2 + 1)) ∧ z = (p + Real.sqrt (p^2 + 1)) ∨
   x = (p - Real.sqrt (p^2 + 1)) ∧ y = (p - Real.sqrt (p^2 + 1)) ∧ z = (p - Real.sqrt (p^2 + 1))) ∨
  (x = (Real.sqrt (1 - p^2)) ∧ y = (Real.sqrt (1 - p^2)) ∧ z = (-p - Real.sqrt (1 - p^2)) ∨
   x = (-Real.sqrt (1 - p^2)) ∧ y = (-Real.sqrt (1 - p^2)) ∧ z = (-p + Real.sqrt (1 - p^2))) :=
by
  -- Proof starts here
  sorry

end find_solutions_l288_288891


namespace angle_OQP_is_right_angle_l288_288768

open EuclideanGeometry

variables {Ω : Type*} [metric_space Ω] [normed_group Ω] [normed_space ℝ Ω] [inner_product_space ℝ Ω]

def circle (O : Ω) (r : ℝ) : set Ω := {P | dist P O = r}

def cyclic_quad (A B C D : Ω) : Prop :=
∃ O : Ω, ∃ r : ℝ, circle O r = {P | ∃ (θ : ℝ), unit_vector (rotate θ (landmark_basis O r)) ∈ {A, B, C, D}}

def intersection_of_diagonals (P A C B D : Ω) : Prop :=
∃ A' C' B' D' : Ω, segment A C ∩ segment B D = {P}

noncomputable def circumcircle_intersection (P Q A B C D : Ω) : Prop :=
∃ Ω₁ Ω₂ : set Ω, Ω₁ = circumcircle A B P ∧ Ω₂ = circumcircle C D P ∧ Q ∈ Ω₁ ∩ Ω₂

theorem angle_OQP_is_right_angle (O A B C D P Q : Ω) 
    (H1 : circle O 1 ⊆ {A, B, C, D})
    (H2 : cyclic_quad A B C D)
    (H3 : intersection_of_diagonals P A C B D)
    (H4 : circumcircle_intersection P Q A B C D) :
      angle O Q P = π / 2 :=
by
  sorry

end angle_OQP_is_right_angle_l288_288768


namespace suitcase_problem_l288_288505

noncomputable def weight_of_electronics (k : ℝ) : ℝ :=
  2 * k

theorem suitcase_problem (k : ℝ) (B C E T : ℝ) (hc1 : B = 5 * k) (hc2 : C = 4 * k) (hc3 : E = 2 * k) (hc4 : T = 3 * k) (new_ratio : 5 * k / (4 * k - 7) = 3) :
  E = 6 :=
by
  sorry

end suitcase_problem_l288_288505


namespace valid_duty_schedules_l288_288838

noncomputable def validSchedules : ℕ := 
  let A_schedule := Nat.choose 7 4  -- \binom{7}{4} for A
  let B_schedule := Nat.choose 4 4  -- \binom{4}{4} for B
  let C_schedule := Nat.choose 6 3  -- \binom{6}{3} for C
  let D_schedule := Nat.choose 5 5  -- \binom{5}{5} for D
  A_schedule * B_schedule * C_schedule * D_schedule

theorem valid_duty_schedules : validSchedules = 700 := by
  -- proof steps will go here
  sorry

end valid_duty_schedules_l288_288838


namespace smallest_perfect_square_divisible_by_2_3_5_l288_288360

theorem smallest_perfect_square_divisible_by_2_3_5 : 
  ∃ n : ℕ, n > 0 ∧ (∃ m : ℕ, n = m * m) ∧ (2 ∣ n) ∧ (3 ∣ n) ∧ (5 ∣ n) ∧
         (∀ k : ℕ, 
           (k > 0 ∧ (∃ l : ℕ, k = l * l) ∧ (2 ∣ k) ∧ (3 ∣ k) ∧ (5 ∣ k)) → n ≤ k) :=
sorry

end smallest_perfect_square_divisible_by_2_3_5_l288_288360


namespace lower_side_length_is_correct_l288_288012

noncomputable def length_of_lower_side
  (a b h : ℝ) (A : ℝ) 
  (cond1 : a = b + 3.4)
  (cond2 : h = 5.2)
  (cond3 : A = 100.62) : ℝ :=
b

theorem lower_side_length_is_correct
  (a b h : ℝ) (A : ℝ)
  (cond1 : a = b + 3.4)
  (cond2 : h = 5.2)
  (cond3 : A = 100.62)
  (ha : A = (1/2) * (a + b) * h) : b = 17.65 :=
by
  sorry

end lower_side_length_is_correct_l288_288012


namespace yuan_older_than_david_l288_288082

theorem yuan_older_than_david (David_age : ℕ) (Yuan_age : ℕ) 
  (h1 : Yuan_age = 2 * David_age) 
  (h2 : David_age = 7) : 
  Yuan_age - David_age = 7 := by
  sorry

end yuan_older_than_david_l288_288082


namespace quadratic_has_real_roots_l288_288902

theorem quadratic_has_real_roots (k : ℝ) : (∃ x : ℝ, x^2 - 4 * x - 2 * k + 8 = 0) ->
  k ≥ 2 :=
by
  sorry

end quadratic_has_real_roots_l288_288902


namespace conservation_of_mass_l288_288713

def molecular_weight_C := 12.01
def molecular_weight_H := 1.008
def molecular_weight_O := 16.00
def molecular_weight_Na := 22.99

def molecular_weight_C9H8O4 := (9 * molecular_weight_C) + (8 * molecular_weight_H) + (4 * molecular_weight_O)
def molecular_weight_NaOH := molecular_weight_Na + molecular_weight_O + molecular_weight_H
def molecular_weight_C7H6O3 := (7 * molecular_weight_C) + (6 * molecular_weight_H) + (3 * molecular_weight_O)
def molecular_weight_CH3COONa := (2 * molecular_weight_C) + (3 * molecular_weight_H) + (2 * molecular_weight_O) + molecular_weight_Na

theorem conservation_of_mass :
  (molecular_weight_C9H8O4 + molecular_weight_NaOH) = (molecular_weight_C7H6O3 + molecular_weight_CH3COONa) := by
  sorry

end conservation_of_mass_l288_288713


namespace impossible_event_l288_288824

noncomputable def EventA := ∃ (ω : ℕ), ω = 0 ∨ ω = 1
noncomputable def EventB := ∃ (t : ℤ), t >= 0
noncomputable def Bag := {b : String // b = "White"}
noncomputable def EventC := ∀ (x : Bag), x.val ≠ "Red"
noncomputable def EventD := ∀ (a b : ℤ), (a > 0 ∧ b < 0) → a > b

theorem impossible_event:
  (EventA ∧ EventB ∧ EventD) →
  EventC :=
by
  sorry

end impossible_event_l288_288824


namespace simplify_problem_l288_288322

noncomputable def simplify_expression : ℝ :=
  let numer := (Real.sqrt 3 - 1) ^ (1 - Real.sqrt 2)
  let denom := (Real.sqrt 3 + 1) ^ (1 + Real.sqrt 2)
  numer / denom

theorem simplify_problem :
  simplify_expression = 2 ^ (1 - Real.sqrt 2) * (4 - 2 * Real.sqrt 3) :=
by
  sorry

end simplify_problem_l288_288322


namespace average_temps_l288_288236

-- Define the temperature lists
def temps_C : List ℚ := [
  37.3, 37.2, 36.9, -- Sunday
  36.6, 36.9, 37.1, -- Monday
  37.1, 37.3, 37.2, -- Tuesday
  36.8, 37.3, 37.5, -- Wednesday
  37.1, 37.7, 37.3, -- Thursday
  37.5, 37.4, 36.9, -- Friday
  36.9, 37.0, 37.1  -- Saturday
]

def temps_K : List ℚ := [
  310.4, 310.3, 310.0, -- Sunday
  309.8, 310.0, 310.2, -- Monday
  310.2, 310.4, 310.3, -- Tuesday
  309.9, 310.4, 310.6, -- Wednesday
  310.2, 310.8, 310.4, -- Thursday
  310.6, 310.5, 310.0, -- Friday
  310.0, 310.1, 310.2  -- Saturday
]

def temps_R : List ℚ := [
  558.7, 558.6, 558.1, -- Sunday
  557.7, 558.1, 558.3, -- Monday
  558.3, 558.7, 558.6, -- Tuesday
  558.0, 558.7, 559.1, -- Wednesday
  558.3, 559.4, 558.7, -- Thursday
  559.1, 558.9, 558.1, -- Friday
  558.1, 558.2, 558.3  -- Saturday
]

-- Calculate the average of a list of temperatures
def average (temps : List ℚ) : ℚ :=
  temps.sum / temps.length

-- Define the average temperatures
def avg_C := average temps_C
def avg_K := average temps_K
def avg_R := average temps_R

-- State that the computed averages are equal to the provided values
theorem average_temps :
  avg_C = 37.1143 ∧
  avg_K = 310.1619 ∧
  avg_R = 558.2524 :=
by
  -- Proof can be completed here
  sorry

end average_temps_l288_288236


namespace gcd_g10_g13_l288_288774

-- Define the polynomial function g
def g (x : ℤ) : ℤ := x^3 - 3 * x^2 + x + 2050

-- State the theorem to prove that gcd(g(10), g(13)) is 1
theorem gcd_g10_g13 : Int.gcd (g 10) (g 13) = 1 := by
  sorry

end gcd_g10_g13_l288_288774


namespace cherry_trees_leaves_l288_288556

-- Define the original number of trees
def original_num_trees : ℕ := 7

-- Define the number of trees actually planted
def actual_num_trees : ℕ := 2 * original_num_trees

-- Define the number of leaves each tree drops
def leaves_per_tree : ℕ := 100

-- Define the total number of leaves that fall
def total_leaves : ℕ := actual_num_trees * leaves_per_tree

-- Theorem statement for the problem
theorem cherry_trees_leaves : total_leaves = 1400 := by
  sorry

end cherry_trees_leaves_l288_288556


namespace total_shaded_area_l288_288841

theorem total_shaded_area 
  (side': ℝ) (d: ℝ) (s: ℝ)
  (h1: 12 / d = 4)
  (h2: d / s = 4) : 
  d = 3 →
  s = 3 / 4 →
  (π * (d / 2) ^ 2 + 8 * s ^ 2) = 9 * π / 4 + 9 / 2 :=
by
  intro h3 h4
  have h5 : d = 3 := h3
  have h6 : s = 3 / 4 := h4
  rw [h5, h6]
  sorry

end total_shaded_area_l288_288841


namespace fraction_increases_l288_288479

theorem fraction_increases (a : ℝ) (h : ℝ) (ha : a > -1) (hh : h > 0) : 
  (a + h) / (a + h + 1) > a / (a + 1) := 
by 
  sorry

end fraction_increases_l288_288479


namespace other_root_of_quadratic_eq_l288_288952

theorem other_root_of_quadratic_eq (m : ℝ) (q : ℝ) :
  (∃ x : ℝ, x ≠ q ∧ 3 * x^2 + m * x - 7 = 0) →
  (3 * q^2 + m * q - 7 = 0) →
  q = -7 / 3 :=
by
  intro h
  sorry

end other_root_of_quadratic_eq_l288_288952


namespace trapezoid_area_is_correct_l288_288432

def square_side_lengths : List ℕ := [1, 3, 5, 7]
def total_base_length : ℕ := square_side_lengths.sum
def tallest_square_height : ℕ := 7

noncomputable def trapezoid_area_between_segment_and_base : ℚ :=
  let height_at_x (x : ℚ) : ℚ := x * (7/16)
  let base_1 := 4
  let base_2 := 9
  let height_1 := height_at_x base_1
  let height_2 := height_at_x base_2
  ((height_1 + height_2) * (base_2 - base_1) / 2)

theorem trapezoid_area_is_correct :
  trapezoid_area_between_segment_and_base = 14.21875 :=
sorry

end trapezoid_area_is_correct_l288_288432


namespace sin_add_double_alpha_l288_288896

open Real

theorem sin_add_double_alpha (alpha : ℝ) (h : sin (π / 6 - alpha) = 3 / 5) :
  sin (π / 6 + 2 * alpha) = 7 / 25 :=
by
  sorry

end sin_add_double_alpha_l288_288896


namespace luke_fish_fillets_l288_288577

def fish_per_day : ℕ := 2
def days : ℕ := 30
def fillets_per_fish : ℕ := 2

theorem luke_fish_fillets : fish_per_day * days * fillets_per_fish = 120 := 
by
  sorry

end luke_fish_fillets_l288_288577


namespace sum_greater_than_two_l288_288501

variables {x y : ℝ}

theorem sum_greater_than_two (hx : x^7 > y^6) (hy : y^7 > x^6) : x + y > 2 :=
sorry

end sum_greater_than_two_l288_288501


namespace Matilda_age_is_35_l288_288927

-- Definitions based on conditions
def Louis_age : ℕ := 14
def Jerica_age : ℕ := 2 * Louis_age
def Matilda_age : ℕ := Jerica_age + 7

-- Theorem to prove the question's answer is correct
theorem Matilda_age_is_35 : Matilda_age = 35 :=
by
  -- Adding proof steps
  sorry

end Matilda_age_is_35_l288_288927


namespace probability_blue_face_facing_up_l288_288354

-- Define the context
def octahedron_faces : ℕ := 8
def blue_faces : ℕ := 5
def red_faces : ℕ := 3
def total_faces : ℕ := blue_faces + red_faces

-- The probability calculation theorem
theorem probability_blue_face_facing_up (h : total_faces = octahedron_faces) :
  (blue_faces : ℝ) / (octahedron_faces : ℝ) = 5 / 8 :=
by
  -- Placeholder for proof
  sorry

end probability_blue_face_facing_up_l288_288354


namespace factorize_expression_l288_288883

variable {x y : ℝ}

theorem factorize_expression : xy^2 - x = x * (y - 1) * (y + 1) := 
by
  -- Define the left-hand side of the equation
  let lhs := x * y^2 - x
  -- Define the right-hand side of the equation
  let rhs := x * (y - 1) * (y + 1)
  -- Provide the goal to prove
  show lhs = rhs
  sorry

end factorize_expression_l288_288883


namespace value_of_a3_minus_a2_l288_288001

theorem value_of_a3_minus_a2 : 
  (∃ S : ℕ → ℕ, (∀ n : ℕ, S n = n^2) ∧ (S 3 - S 2 - (S 2 - S 1)) = 2) :=
sorry

end value_of_a3_minus_a2_l288_288001


namespace barbara_wins_gameA_l288_288399

noncomputable def gameA_winning_strategy : Prop :=
∃ (has_winning_strategy : (ℤ → ℝ) → Prop),
  has_winning_strategy (fun n => n : ℤ → ℝ)

theorem barbara_wins_gameA :
  gameA_winning_strategy := sorry

end barbara_wins_gameA_l288_288399


namespace base_r_correct_l288_288016

theorem base_r_correct (r : ℕ) :
  (5 * r ^ 2 + 6 * r) + (4 * r ^ 2 + 2 * r) = r ^ 3 + r ^ 2 → r = 8 := 
by 
  sorry

end base_r_correct_l288_288016


namespace skye_race_l288_288922

noncomputable def first_part_length := 3

theorem skye_race 
  (total_track_length : ℕ := 6)
  (speed_first_part : ℕ := 150)
  (distance_second_part : ℕ := 2)
  (speed_second_part : ℕ := 200)
  (distance_third_part : ℕ := 1)
  (speed_third_part : ℕ := 300)
  (avg_speed : ℕ := 180) :
  first_part_length = 3 :=
  sorry

end skye_race_l288_288922


namespace greatest_four_digit_multiple_of_17_l288_288990

theorem greatest_four_digit_multiple_of_17 : ∃ n, (1000 ≤ n) ∧ (n ≤ 9999) ∧ (17 ∣ n) ∧ ∀ m, (1000 ≤ m) ∧ (m ≤ 9999) ∧ (17 ∣ m) → m ≤ n :=
begin
  sorry
end

end greatest_four_digit_multiple_of_17_l288_288990


namespace product_of_fractions_is_27_l288_288240

theorem product_of_fractions_is_27 :
  (1/3) * (9/1) * (1/27) * (81/1) * (1/243) * (729/1) = 27 :=
by
  sorry

end product_of_fractions_is_27_l288_288240


namespace part1_part2_part3_l288_288435

noncomputable def seq (a : ℝ) (n : ℕ) : ℝ := if n = 0 then 0 else (1 - a) / n

theorem part1 (a : ℝ) (h_pos : ∀ n : ℕ, n > 0 → seq a n > 0)
  (a1_eq : seq a 1 = 1 / 2) (a2_eq : seq a 2 = 1 / 4) : true :=
by trivial

theorem part2 (a : ℝ) (h_pos : ∀ n : ℕ, n > 0 → seq a n > 0)
  (n : ℕ) (hn : n > 0) : 0 < seq a n ∧ seq a n < 1 :=
sorry

theorem part3 (a : ℝ) (h_pos : ∀ n : ℕ, n > 0 → seq a n > 0)
  (n : ℕ) (hn : n > 0) : seq a n > seq a (n + 1) :=
sorry

end part1_part2_part3_l288_288435


namespace range_of_m_l288_288908

noncomputable def f (x m : ℝ) : ℝ := (1 / 4) * x^4 - (2 / 3) * x^3 + m

theorem range_of_m (m : ℝ) : (∀ x : ℝ, f x m + (1 / 3) ≥ 0) ↔ m ≥ 1 := 
sorry

end range_of_m_l288_288908


namespace area_relation_l288_288021

-- Define the areas of the triangles
variables (a b c : ℝ)

-- Define the condition that triangles T_a and T_c are similar (i.e., homothetic)
-- which implies the relationship between their areas.
theorem area_relation (ha : 0 < a) (hc : 0 < c) (habc : b = Real.sqrt (a * c)) : b = Real.sqrt (a * c) := by
  sorry

end area_relation_l288_288021


namespace total_new_cans_l288_288130

-- Define the condition
def initial_cans : ℕ := 256
def first_term : ℕ := 64
def ratio : ℚ := 1 / 4
def terms : ℕ := 4

-- Define the sum of the geometric series
noncomputable def geometric_series_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * ((1 - r ^ n) / (1 - r))

-- Problem statement in Lean 4
theorem total_new_cans : geometric_series_sum first_term ratio terms = 85 := by
  sorry

end total_new_cans_l288_288130


namespace jackson_sandwiches_l288_288624

noncomputable def total_sandwiches (weeks : ℕ) (miss_wed : ℕ) (miss_fri : ℕ) : ℕ :=
  let total_wednesdays := weeks - miss_wed
  let total_fridays := weeks - miss_fri
  total_wednesdays + total_fridays

theorem jackson_sandwiches : total_sandwiches 36 1 2 = 69 := by
  sorry

end jackson_sandwiches_l288_288624


namespace simplify_fraction_l288_288321

theorem simplify_fraction :
  (1 / (1 / (Real.sqrt 3 + 1) + 3 / (Real.sqrt 5 - 2))) = 2 / (Real.sqrt 3 + 6 * Real.sqrt 5 + 11) :=
by
  sorry

end simplify_fraction_l288_288321


namespace x_equals_neg_one_l288_288155

theorem x_equals_neg_one
  (a b c : ℝ)
  (h1 : a ≠ 0)
  (h2 : b ≠ 0)
  (h3 : c ≠ 0)
  (h4 : (a + b - c) / c = (a - b + c) / b ∧ (a + b - c) / c = (-a + b + c) / a)
  (x : ℝ)
  (h5 : x = (a + b) * (b + c) * (c + a) / (a * b * c))
  (h6 : x < 0) :
  x = -1 := 
sorry

end x_equals_neg_one_l288_288155


namespace cos_double_angle_l288_288261

theorem cos_double_angle (a : ℝ) (h : Real.sin a = 1 / 3) : Real.cos (2 * a) = 7 / 9 :=
by
  sorry

end cos_double_angle_l288_288261


namespace best_regression_effect_l288_288751

theorem best_regression_effect (R2_1 R2_2 R2_3 R2_4 : ℝ)
  (h1 : R2_1 = 0.36)
  (h2 : R2_2 = 0.95)
  (h3 : R2_3 = 0.74)
  (h4 : R2_4 = 0.81):
  max (max (max R2_1 R2_2) R2_3) R2_4 = 0.95 := by
  sorry

end best_regression_effect_l288_288751


namespace probability_of_ab_divisible_by_4_l288_288528

noncomputable def probability_divisible_by_4 : ℚ :=
  let outcomes := (1, 2, 3, 4, 5, 6, 7, 8, 9)
  let favorable_outcomes := Set.filter (λ x => x % 4 = 0) outcomes
  let prob_die := favorable_outcomes.card.to_rat / outcomes.card.to_rat
  prob_die * prob_die

theorem probability_of_ab_divisible_by_4 :
  probability_divisible_by_4 = 4 / 81 := 
sorry

end probability_of_ab_divisible_by_4_l288_288528


namespace packs_sold_in_other_villages_l288_288079

theorem packs_sold_in_other_villages
  (packs_v1 : ℕ) (packs_v2 : ℕ) (h1 : packs_v1 = 23) (h2 : packs_v2 = 28) :
  packs_v1 + packs_v2 = 51 := 
by {
  sorry
}

end packs_sold_in_other_villages_l288_288079


namespace fraction_power_computation_l288_288677

theorem fraction_power_computation : (5 / 6) ^ 4 = 625 / 1296 :=
by
  -- Normally we'd provide the proof here, but it's omitted as per instructions
  sorry

end fraction_power_computation_l288_288677


namespace rationalize_denominator_l288_288046

theorem rationalize_denominator : (35 : ℝ) / Real.sqrt 15 = (7 / 3 : ℝ) * Real.sqrt 15 :=
by
  sorry

end rationalize_denominator_l288_288046


namespace car_speed_l288_288829

theorem car_speed (v : ℝ) (h : (1/v) * 3600 = ((1/48) * 3600) + 15) : v = 40 := 
by 
  sorry

end car_speed_l288_288829


namespace uranus_appearance_minutes_after_6AM_l288_288798

-- Definitions of the given times and intervals
def mars_last_seen : Int := 0 -- 12:10 AM in minutes after midnight
def jupiter_after_mars : Int := 2 * 60 + 41 -- 2 hours and 41 minutes in minutes
def uranus_after_jupiter : Int := 3 * 60 + 16 -- 3 hours and 16 minutes in minutes
def reference_time : Int := 6 * 60 -- 6:00 AM in minutes after midnight

-- Statement of the problem
theorem uranus_appearance_minutes_after_6AM :
  let jupiter_first_appearance := mars_last_seen + jupiter_after_mars
  let uranus_first_appearance := jupiter_first_appearance + uranus_after_jupiter
  (uranus_first_appearance - reference_time) = 7 := by
  sorry

end uranus_appearance_minutes_after_6AM_l288_288798


namespace b_cong_zero_l288_288028

theorem b_cong_zero (a b c m : ℤ) (h₀ : 1 < m) (h : ∀ (n : ℕ), (a ^ n + b * n + c) % m = 0) : b % m = 0 :=
  sorry

end b_cong_zero_l288_288028


namespace find_integer_pairs_l288_288243

theorem find_integer_pairs :
  ∀ (a b : ℕ), 0 < a → 0 < b → a * b + 2 = a^3 + 2 * b →
  (a = 1 ∧ b = 1) ∨ (a = 3 ∧ b = 25) ∨ (a = 4 ∧ b = 31) ∨ (a = 5 ∧ b = 41) ∨ (a = 8 ∧ b = 85) :=
by
  intros a b ha hb hab_eq
  -- Proof goes here
  sorry

end find_integer_pairs_l288_288243


namespace average_speed_l288_288546

theorem average_speed (speed1 speed2 time1 time2: ℝ) (h1 : speed1 = 60) (h2 : time1 = 3) (h3 : speed2 = 85) (h4 : time2 = 2) : 
  (speed1 * time1 + speed2 * time2) / (time1 + time2) = 70 :=
by
  -- Definitions
  have distance1 := speed1 * time1
  have distance2 := speed2 * time2
  have total_distance := distance1 + distance2
  have total_time := time1 + time2
  -- Proof skeleton
  sorry

end average_speed_l288_288546


namespace cos_double_angle_l288_288153

theorem cos_double_angle (α : ℝ) (h : Real.tan α = 2) : Real.cos (2 * α) = -3 / 5 := by
  sorry

end cos_double_angle_l288_288153


namespace find_number_l288_288652

theorem find_number (x : ℕ) (h : x + 20 + x + 30 + x + 40 + x + 10 = 4100) : x = 1000 := 
by
  sorry

end find_number_l288_288652


namespace product_is_correct_l288_288179

noncomputable def IKS := 521
noncomputable def KSI := 215
def product := 112015

theorem product_is_correct : IKS * KSI = product :=
by
  -- Proof yet to be constructed
  sorry

end product_is_correct_l288_288179


namespace sequence_count_415800_l288_288862

open Equiv.Perm

/-- Define the transformations A, B, C, D as permutations -/
def A : equiv.perm (fin 4) := @equiv.perm.transposition _ _ 0 1
def B : equiv.perm (fin 4) := (A)⁻¹
def C : equiv.perm (fin 4) := @equiv.perm.transposition _ _ 0 2
def D : equiv.perm (fin 4) := @equiv.perm.transposition _ _ 1 3

/-- Define the dihedral group D4 -/
inductive D4 : Type
| id | A | A2 | A3 | A4 | C | D | AC | BD

/-- Define the identity element -/
def id : D4 := D4.id

/-- Define a proof that there are 415800 sequences of 12 transformations
    using {A, B, C, D} that restore square WXYZ to its original positions -/
theorem sequence_count_415800 :
  ∃ (seq : list (D4)), seq.length = 12 ∧
  (list.foldl (•) id seq = id) ∧ (list.permutations seq).length = 415800 :=
sorry

end sequence_count_415800_l288_288862


namespace minimum_cards_to_draw_to_ensure_2_of_each_suit_l288_288090

noncomputable def min_cards_to_draw {total_cards : ℕ} {suit_count : ℕ} {cards_per_suit : ℕ} {joker_count : ℕ}
  (h_total : total_cards = 54)
  (h_suits : suit_count = 4)
  (h_cards_per_suit : cards_per_suit = 13)
  (h_jokers : joker_count = 2) : ℕ :=
  43

theorem minimum_cards_to_draw_to_ensure_2_of_each_suit 
  (total_cards suit_count cards_per_suit joker_count : ℕ)
  (h_total : total_cards = 54) 
  (h_suits : suit_count = 4)
  (h_cards_per_suit : cards_per_suit = 13)
  (h_jokers : joker_count = 2) : 
  min_cards_to_draw h_total h_suits h_cards_per_suit h_jokers = 43 :=
  by
  sorry

end minimum_cards_to_draw_to_ensure_2_of_each_suit_l288_288090


namespace coffee_shop_cups_l288_288641

variables (A B X Y : ℕ) (Z : ℕ)

theorem coffee_shop_cups (h1 : Z = (A * B * X) + (A * (7 - B) * Y)) : 
  Z = (A * B * X) + (A * (7 - B) * Y) := 
by
  sorry

end coffee_shop_cups_l288_288641


namespace no_such_function_l288_288169

theorem no_such_function : ¬ ∃ f : ℝ → ℝ, ∀ x y : ℝ, |f (x + y) + Real.sin x + Real.sin y| < 2 := 
sorry

end no_such_function_l288_288169


namespace base6_addition_correct_l288_288303

theorem base6_addition_correct (S H E : ℕ) (h1 : S < 6) (h2 : H < 6) (h3 : E < 6) 
  (distinct : S ≠ H ∧ H ≠ E ∧ S ≠ E) 
  (h4: S + H * 6 + E * 6^2 +  H * 6 = H + E * 6 + H * 6^2 + E * 6^1) :
  S + H + E = 12 :=
by sorry

end base6_addition_correct_l288_288303


namespace Nicole_cards_l288_288176

variables (N : ℕ)

-- Conditions from step A
def Cindy_collected (N : ℕ) : ℕ := 2 * N
def Nicole_and_Cindy_combined (N : ℕ) : ℕ := N + Cindy_collected N
def Rex_collected (N : ℕ) : ℕ := (Nicole_and_Cindy_combined N) / 2
def Rex_cards_each (N : ℕ) : ℕ := Rex_collected N / 4

-- Question: How many cards did Nicole collect? Answer: N = 400
theorem Nicole_cards (N : ℕ) (h : Rex_cards_each N = 150) : N = 400 :=
sorry

end Nicole_cards_l288_288176


namespace least_value_r_minus_p_l288_288022

theorem least_value_r_minus_p (x : ℝ) (h1 : 1 / 2 < x) (h2 : x < 5) :
  ∃ r p, r = 5 ∧ p = 1/2 ∧ r - p = 9 / 2 :=
by
  sorry

end least_value_r_minus_p_l288_288022


namespace smallest_perfect_square_divisible_by_2_3_5_l288_288368

theorem smallest_perfect_square_divisible_by_2_3_5 :
  ∃ n : ℕ, (n > 0) ∧ (nat.sqrt n * nat.sqrt n = n) ∧ (2 ∣ n) ∧ (3 ∣ n) ∧ (5 ∣ n) ∧ n = 900 := sorry

end smallest_perfect_square_divisible_by_2_3_5_l288_288368


namespace math_problem_l288_288502

theorem math_problem (x y : ℝ) (h1 : x^7 > y^6) (h2 : y^7 > x^6) : x + y > 2 :=
sorry

end math_problem_l288_288502


namespace nine_pow_1000_mod_13_l288_288060

theorem nine_pow_1000_mod_13 :
  (9^1000) % 13 = 9 :=
by
  have h1 : 9^1 % 13 = 9 := by sorry
  have h2 : 9^2 % 13 = 3 := by sorry
  have h3 : 9^3 % 13 = 1 := by sorry
  have cycle : ∀ n, 9^(3 * n + 1) % 13 = 9 := by sorry
  exact (cycle 333)

end nine_pow_1000_mod_13_l288_288060


namespace train_stop_time_l288_288717

theorem train_stop_time (speed_no_stops speed_with_stops : ℕ) (time_per_hour : ℕ) (stoppage_time_per_hour : ℕ) :
  speed_no_stops = 45 →
  speed_with_stops = 30 →
  time_per_hour = 60 →
  stoppage_time_per_hour = 20 :=
by
  intros h1 h2 h3
  sorry

end train_stop_time_l288_288717


namespace quadratic_expression_value_l288_288335

theorem quadratic_expression_value (x1 x2 : ℝ)
    (h1: x1^2 + 5 * x1 + 1 = 0)
    (h2: x2^2 + 5 * x2 + 1 = 0) :
    ( (x1 * Real.sqrt 6 / (1 + x2))^2 + (x2 * Real.sqrt 6 / (1 + x1))^2 ) = 220 := 
sorry

end quadratic_expression_value_l288_288335


namespace parametric_curve_intersects_itself_l288_288710

-- Given parametric equations
def param_x (t : ℝ) : ℝ := t^2 + 3
def param_y (t : ℝ) : ℝ := t^3 - 6 * t + 4

-- Existential statement for self-intersection
theorem parametric_curve_intersects_itself :
  ∃ t1 t2 : ℝ, t1 ≠ t2 ∧ param_x t1 = param_x t2 ∧ param_y t1 = param_y t2 ∧ param_x t1 = 9 ∧ param_y t1 = 4 :=
sorry

end parametric_curve_intersects_itself_l288_288710


namespace smallest_positive_period_of_f_symmetry_center_of_f_range_of_f_in_interval_l288_288011

open Real

noncomputable def a (x : ℝ) : ℝ × ℝ := (5 * sqrt 3 * cos x, cos x)
noncomputable def b (x : ℝ) : ℝ × ℝ := (sin x, 2 * cos x)

noncomputable def f (x : ℝ) : ℝ := 
  let a_dot_b := (a x).1 * (b x).1 + (a x).2 * (b x).2
  let b_norm_sq := (b x).1 ^ 2 + (b x).2 ^ 2
  a_dot_b + b_norm_sq + 3 / 2

theorem smallest_positive_period_of_f :
  ∀ x, f (x + π) = f x := sorry

theorem symmetry_center_of_f :
  ∃ k : ℤ, ∀ x, f x = 5 ↔ x = (-π / 12 + k * (π / 2) : ℝ) := sorry

theorem range_of_f_in_interval :
  ∀ x, (π / 6 ≤ x ∧ x ≤ π / 2) → (5 / 2 ≤ f x ∧ f x ≤ 10) := sorry

end smallest_positive_period_of_f_symmetry_center_of_f_range_of_f_in_interval_l288_288011


namespace at_least_2020_distinct_n_l288_288184

theorem at_least_2020_distinct_n : 
  ∃ (N : Nat), N ≥ 2020 ∧ ∃ (a : Fin N → ℕ), 
  Function.Injective a ∧ ∀ i, ∃ k : ℚ, (a i : ℚ) + 0.25 = (k + 1/2)^2 := 
sorry

end at_least_2020_distinct_n_l288_288184


namespace range_of_a_l288_288657

variable {f : ℝ → ℝ}

def even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def increasing_on_non_neg (f : ℝ → ℝ) : Prop :=
  ∀ (x y : ℝ), (0 ≤ x ∧ x ≤ y) → f x ≤ f y

theorem range_of_a (f : ℝ → ℝ) (a : ℝ) :
  even_function f → increasing_on_non_neg f → f a ≤ f 2 → -2 ≤ a ∧ a ≤ 2 :=
by
  intro h_even h_increasing h_le
  sorry

end range_of_a_l288_288657


namespace total_cost_correct_l288_288996

noncomputable def total_cost (sandwiches: ℕ) (price_per_sandwich: ℝ) (sodas: ℕ) (price_per_soda: ℝ) (discount: ℝ) (tax: ℝ) : ℝ :=
  let total_sandwich_cost := sandwiches * price_per_sandwich
  let total_soda_cost := sodas * price_per_soda
  let discounted_sandwich_cost := total_sandwich_cost * (1 - discount)
  let total_before_tax := discounted_sandwich_cost + total_soda_cost
  let total_with_tax := total_before_tax * (1 + tax)
  total_with_tax

theorem total_cost_correct : 
  total_cost 2 3.49 4 0.87 0.10 0.05 = 10.25 :=
by
  sorry

end total_cost_correct_l288_288996


namespace line_parallelism_theorem_l288_288911

-- Definitions of the relevant geometric conditions
variables {Line Plane : Type}
variables (m n : Line) (α β : Plane)

-- Conditions as hypotheses
axiom line_parallel_plane (m : Line) (α : Plane) : Prop
axiom line_in_plane (n : Line) (α : Plane) : Prop
axiom plane_intersection_line (α β : Plane) : Line
axiom line_parallel (m n : Line) : Prop

-- The problem statement in Lean 4
theorem line_parallelism_theorem 
  (h1 : line_parallel_plane m α) 
  (h2 : line_in_plane n β) 
  (h3 : plane_intersection_line α β = n) 
  (h4 : line_parallel_plane m β) : line_parallel m n :=
sorry

end line_parallelism_theorem_l288_288911


namespace number_of_children_l288_288513

-- Definitions based on conditions from the problem
def total_spectators := 10000
def men_spectators := 7000
def spectators_other_than_men := total_spectators - men_spectators
def women_and_children_ratio := 5

-- Prove there are 2500 children
theorem number_of_children : 
  ∃ (women children : ℕ), 
    spectators_other_than_men = women + women_and_children_ratio * women ∧ 
    children = women_and_children_ratio * women ∧
    children = 2500 :=
by
  sorry

end number_of_children_l288_288513


namespace youngest_child_age_is_3_l288_288400

noncomputable def family_age_problem : Prop :=
  ∃ (age_diff_2 : ℕ) (age_10_years_ago : ℕ) (new_family_members : ℕ) (same_present_avg_age : ℕ) (youngest_child_age : ℕ),
    age_diff_2 = 2 ∧
    age_10_years_ago = 4 * 24 ∧
    new_family_members = 2 ∧
    same_present_avg_age = 24 ∧
    youngest_child_age = 3 ∧
    (96 + 4 * 10 + (youngest_child_age + (youngest_child_age + age_diff_2)) = 6 * same_present_avg_age)

theorem youngest_child_age_is_3 : family_age_problem := sorry

end youngest_child_age_is_3_l288_288400


namespace sum_primes_between_20_and_40_l288_288819

theorem sum_primes_between_20_and_40 :
  23 + 29 + 31 + 37 = 120 :=
by
  sorry

end sum_primes_between_20_and_40_l288_288819


namespace sum_of_pqrstu_l288_288771

theorem sum_of_pqrstu (p q r s t : ℤ) (h : (8 - p) * (8 - q) * (8 - r) * (8 - s) * (8 - t) = -72) 
  (hpqrs : p ≠ q) (hnpr : p ≠ r) (hnps : p ≠ s) (hnpt : p ≠ t) (hnqr : q ≠ r) 
  (hnqs : q ≠ s) (hnqt : q ≠ t) (hnrs : r ≠ s) (hnrt : r ≠ t) (hnst : s ≠ t) : 
  p + q + r + s + t = 25 := 
by
  sorry

end sum_of_pqrstu_l288_288771


namespace find_nm_l288_288253

theorem find_nm :
  ∃ n m : Int, (-120 : Int) ≤ n ∧ n ≤ 120 ∧ (-120 : Int) ≤ m ∧ m ≤ 120 ∧ 
  (Real.sin (n * Real.pi / 180) = Real.sin (580 * Real.pi / 180)) ∧ 
  (Real.cos (m * Real.pi / 180) = Real.cos (300 * Real.pi / 180)) ∧ 
  n = -40 ∧ m = -60 := by
  sorry

end find_nm_l288_288253


namespace stamps_sum_to_n_l288_288827

noncomputable def selectStamps : Prop :=
  ∀ (n : ℕ) (k : ℕ), n > 0 → 
                      ∃ stamps : List ℕ, 
                      stamps.length = k ∧ 
                      n ≤ stamps.sum ∧ stamps.sum < 2 * k → 
                      ∃ (subset : List ℕ), 
                      subset.sum = n

theorem stamps_sum_to_n : selectStamps := sorry

end stamps_sum_to_n_l288_288827


namespace scale_model_height_l288_288409

/-- 
Given a scale model ratio and the actual height of the skyscraper in feet,
we can deduce the height of the model in inches.
-/
theorem scale_model_height
  (scale_ratio : ℕ := 25)
  (actual_height_feet : ℕ := 1250) :
  (actual_height_feet / scale_ratio) * 12 = 600 :=
by 
  sorry

end scale_model_height_l288_288409


namespace baking_completion_time_l288_288693

theorem baking_completion_time (start_time : ℕ) (partial_bake_time : ℕ) (fraction_baked : ℕ) :
  start_time = 9 → partial_bake_time = 3 → fraction_baked = 4 →
  (start_time + (partial_bake_time * fraction_baked)) = 21 :=
by
  intros h_start h_partial h_fraction
  sorry

end baking_completion_time_l288_288693


namespace marble_weight_l288_288956

theorem marble_weight (W : ℝ) (h : 2 * W + 0.08333333333333333 = 0.75) : 
  W = 0.33333333333333335 := 
by 
  -- Skipping the proof as specified
  sorry

end marble_weight_l288_288956


namespace quadratic_inequality_solution_l288_288962

theorem quadratic_inequality_solution :
  ∀ x : ℝ, -9 * x^2 + 6 * x - 8 < 0 :=
by {
  sorry
}

end quadratic_inequality_solution_l288_288962


namespace arithmetic_sequence_properties_geometric_sequence_properties_l288_288725

-- Define the arithmetic sequence {a_n}
def a (n : ℕ) : ℕ :=
  2 * n - 1

-- Define the sum of the first n terms of {a_n}
def S (n : ℕ) : ℕ :=
  n ^ 2

-- Prove the nth term and the sum of the first n terms of {a_n}
theorem arithmetic_sequence_properties (n : ℕ) :
  a n = 2 * n - 1 ∧ S n = n ^ 2 :=
by sorry

-- Define the geometric sequence {b_n}
def b (n : ℕ) : ℕ :=
  2 ^ (2 * n - 1)

-- Define the sum of the first n terms of {b_n}
def T (n : ℕ) : ℕ :=
  (2 ^ n * (4 ^ n - 1)) / 3

-- Prove the nth term and the sum of the first n terms of {b_n}
theorem geometric_sequence_properties (n : ℕ) (a4 S4 : ℕ) (q : ℕ)
  (h_a4 : a4 = a 4)
  (h_S4 : S4 = S 4)
  (h_q : q ^ 2 - (a4 + 1) * q + S4 = 0) :
  b n = 2 ^ (2 * n - 1) ∧ T n = (2 ^ n * (4 ^ n - 1)) / 3 :=
by sorry

end arithmetic_sequence_properties_geometric_sequence_properties_l288_288725


namespace quadratic_inequality_solution_l288_288328

theorem quadratic_inequality_solution (x m : ℝ) :
  (x^2 + (2*m + 1)*x + m^2 + m > 0) ↔ (x > -m ∨ x < -m - 1) :=
by
  sorry

end quadratic_inequality_solution_l288_288328


namespace factorization_correct_l288_288098

theorem factorization_correct: 
  (a : ℝ) → a^2 - 9 = (a - 3) * (a + 3) :=
by
  intro a
  sorry

end factorization_correct_l288_288098


namespace factorization_l288_288875

theorem factorization (x y : ℝ) : (x * y^2 - x = x * (y - 1) * (y + 1)) :=
begin
  sorry
end

end factorization_l288_288875


namespace no_n_ge_1_such_that_sum_is_perfect_square_l288_288121

theorem no_n_ge_1_such_that_sum_is_perfect_square :
  ¬ ∃ n : ℕ, n ≥ 1 ∧ ∃ k : ℕ, 2^n + 12^n + 2014^n = k^2 :=
by
  sorry

end no_n_ge_1_such_that_sum_is_perfect_square_l288_288121


namespace divides_iff_l288_288635

open Int

theorem divides_iff (n m : ℤ) : (9 ∣ (2 * n + 5 * m)) ↔ (9 ∣ (5 * n + 8 * m)) := 
sorry

end divides_iff_l288_288635


namespace system_solution_5_3_l288_288392

variables (x y : ℤ)

theorem system_solution_5_3 :
  (x = 5) ∧ (y = 3) → (2 * x - 3 * y = 1) :=
by intros; sorry

end system_solution_5_3_l288_288392


namespace eighty_five_squared_l288_288109

theorem eighty_five_squared :
  (85:ℕ)^2 = 7225 := 
by
  let a := 80
  let b := 5
  have h1 : (a + b) = 85 := rfl
  have h2 : (a^2 + 2 * a * b + b^2) = 7225 := by norm_num
  rw [←h1, ←h1]
  rw [ sq (a + b)]
  rw [ mul_add, add_mul, add_mul, mul_comm 2 b]
  rw [←mul_assoc, ←mul_assoc, add_assoc, add_assoc, nat.add_right_comm ]
  exact h2

end eighty_five_squared_l288_288109


namespace town_population_original_l288_288703

noncomputable def original_population (n : ℕ) : Prop :=
  let increased_population := n + 1500
  let decreased_population := (85 / 100 : ℚ) * increased_population
  decreased_population = n + 1455

theorem town_population_original : ∃ n : ℕ, original_population n ∧ n = 1200 :=
by
  sorry

end town_population_original_l288_288703


namespace onion_pieces_per_student_l288_288795

theorem onion_pieces_per_student (total_pizzas : ℕ) (slices_per_pizza : ℕ)
  (cheese_pieces_leftover : ℕ) (onion_pieces_leftover : ℕ) (students : ℕ) (cheese_per_student : ℕ)
  (h1 : total_pizzas = 6) (h2 : slices_per_pizza = 18) (h3 : cheese_pieces_leftover = 8) (h4 : onion_pieces_leftover = 4)
  (h5 : students = 32) (h6 : cheese_per_student = 2) :
  ((total_pizzas * slices_per_pizza) - cheese_pieces_leftover - onion_pieces_leftover - (students * cheese_per_student)) / students = 1 := 
by
  sorry

end onion_pieces_per_student_l288_288795


namespace inequality_of_factorials_and_polynomials_l288_288307

open Nat

theorem inequality_of_factorials_and_polynomials (m n : ℕ) (hm : m ≥ n) :
  2^n * n! ≤ (m+n)! / (m-n)! ∧ (m+n)! / (m-n)! ≤ (m^2 + m)^n :=
by
  sorry

end inequality_of_factorials_and_polynomials_l288_288307


namespace second_quadrant_distance_l288_288920

theorem second_quadrant_distance 
    (m : ℝ) 
    (P : ℝ × ℝ)
    (hP1 : P = (m - 3, m + 2))
    (hP2 : (m + 2) > 0)
    (hP3 : (m - 3) < 0)
    (hDist : |(m + 2)| = 4) : P = (-1, 4) := 
by
  have h1 : m + 2 = 4 := sorry
  have h2 : m = 2 := sorry
  have h3 : P = (2 - 3, 2 + 2) := sorry
  have h4 : P = (-1, 4) := sorry
  exact h4

end second_quadrant_distance_l288_288920


namespace find_larger_number_l288_288672

theorem find_larger_number (x y : ℝ) (h1 : x - y = 5) (h2 : 2 * (x + y) = 40) : x = 12.5 :=
by 
  sorry

end find_larger_number_l288_288672


namespace day_197_of_2005_is_tuesday_l288_288284

-- Definitions based on conditions and question
def is_day_of_week (n : ℕ) (day : ℕ) : Prop := 
  (day % 7) = n

theorem day_197_of_2005_is_tuesday (h : is_day_of_week 2 15) : is_day_of_week 2 197 := 
by
  -- Here, we will state the equivalence under Lean definition (saying remainder modulo 7)
  sorry

end day_197_of_2005_is_tuesday_l288_288284


namespace MarksScore_l288_288294

theorem MarksScore (h_highest : ℕ) (h_range : ℕ) (h_relation : h_highest - h_least = h_range) (h_mark_twice : Mark = 2 * h_least) : Mark = 46 :=
by
    let h_highest := 98
    let h_range := 75
    let h_least := h_highest - h_range
    let Mark := 2 * h_least
    have := h_relation
    have := h_mark_twice
    sorry

end MarksScore_l288_288294


namespace expected_sides_general_expected_sides_rectangle_l288_288808

-- General Problem
theorem expected_sides_general (n k : ℕ) : 
  (∀ n k : ℕ, n ≥ 3 → k ≥ 0 → (1:ℝ) / ((k + 1:ℝ)) * (n + 4 * k:ℝ) ≤ (n + 4 * k) / (k + 1)) := 
begin
  sorry
end

-- Specific Problem for Rectangle
theorem expected_sides_rectangle (k : ℕ) :
  (∀ k : ℕ, k ≥ 0 → 4 = (4 + 4 * k) / (k + 1)) := 
begin
  sorry
end

end expected_sides_general_expected_sides_rectangle_l288_288808


namespace min_a4_b2_l288_288939

open Real

noncomputable def min_value (t : ℝ) : ℝ :=
  (t ^ 4) / 16 + (t ^ 2) / 4

theorem min_a4_b2 (a b t : ℝ) (h : a + b = t) :
  ∃ a b, a + b = t ∧ a ^ 4 + b ^ 2 = min_value t :=
begin
  use [t / 2, t / 2],
  split,
  { simp, },
  { simp [min_value],
    ring, sorry, }
end

end min_a4_b2_l288_288939


namespace addition_example_l288_288532

theorem addition_example : 248 + 64 = 312 := by
  sorry

end addition_example_l288_288532


namespace right_triangle_integral_sides_parity_l288_288960

theorem right_triangle_integral_sides_parity 
  (a b c : ℕ) 
  (h : a^2 + b^2 = c^2) 
  (ha : a % 2 = 1 ∨ a % 2 = 0) 
  (hb : b % 2 = 1 ∨ b % 2 = 0) 
  (hc : c % 2 = 1 ∨ c % 2 = 0) : 
  (a % 2 = 0 ∨ b % 2 = 0 ∨ (a % 2 = 0 ∧ b % 2 = 0 ∧ c % 2 = 0)) := 
sorry

end right_triangle_integral_sides_parity_l288_288960


namespace probability_diff_colors_l288_288746

theorem probability_diff_colors (total_balls : ℕ) (white_balls : ℕ) (black_balls : ℕ) 
  (drawn_balls : ℕ) (total_outcomes : ℕ) (favorable_outcomes : ℕ) :
  total_balls = 4 ∧ white_balls = 3 ∧ black_balls = 1 ∧ drawn_balls = 2 ∧ 
  total_outcomes = Nat.choose 4 2 ∧ favorable_outcomes = Nat.choose 3 1 * Nat.choose 1 1
  → favorable_outcomes / total_outcomes = 1 / 2 :=
by
  sorry

end probability_diff_colors_l288_288746


namespace speed_of_first_train_l288_288792

-- Definitions of the conditions
def ratio_speed (speed1 speed2 : ℝ) := speed1 / speed2 = 7 / 8
def speed_of_second_train := 400 / 4

-- The theorem we want to prove
theorem speed_of_first_train (speed2 := speed_of_second_train) (h : ratio_speed S1 speed2) :
  S1 = 87.5 :=
by 
  sorry

end speed_of_first_train_l288_288792


namespace largest_AB_under_conditions_l288_288484

def is_digit (n : ℕ) : Prop :=
  n ≥ 0 ∧ n ≤ 9

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 ∧ m < n → n % m ≠ 0

theorem largest_AB_under_conditions :
  ∃ A B C D : ℕ, is_digit A ∧ is_digit B ∧ is_digit C ∧ is_digit D ∧
    A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D ∧
    (A + B) % (C + D) = 0 ∧
    is_prime (A + B) ∧ is_prime (C + D) ∧
    (A + B) = 11 :=
sorry

end largest_AB_under_conditions_l288_288484


namespace remainder_hx10_div_hx_l288_288936

noncomputable def h (x : ℕ) := x^6 - x^5 + x^4 - x^3 + x^2 - x + 1

theorem remainder_hx10_div_hx (x : ℕ) : (h x ^ 10) % (h x) = 7 := by
  sorry

end remainder_hx10_div_hx_l288_288936


namespace diamond_associative_l288_288308

def diamond (a b : ℕ) : ℕ := a ^ (b / a)

theorem diamond_associative (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c):
  diamond a (diamond b c) = diamond (diamond a b) c :=
sorry

end diamond_associative_l288_288308


namespace find_a_l288_288287

theorem find_a (a : ℝ) :
  (∀ (x y : ℝ), (a * x + 2 * y + 3 * a = 0) → (3 * x + (a - 1) * y = a - 7)) → 
  a = 3 :=
by
  sorry

end find_a_l288_288287


namespace range_of_a_l288_288732

theorem range_of_a (a : ℝ) : (∀ x : ℤ, x > 2 * a - 3 ∧ 2 * (x : ℝ) ≥ 3 * ((x : ℝ) - 2) + 5) ↔ (1 / 2 ≤ a ∧ a < 1) :=
sorry

end range_of_a_l288_288732


namespace not_and_implies_at_most_one_true_l288_288288

def at_most_one_true (p q : Prop) : Prop := (p → ¬ q) ∧ (q → ¬ p)

theorem not_and_implies_at_most_one_true (p q : Prop) (h : ¬ (p ∧ q)) : at_most_one_true p q :=
begin
  sorry
end

end not_and_implies_at_most_one_true_l288_288288


namespace no_appearance_1234_or_3269_no_reappearance_1975_from_2nd_time_l288_288019

def seq (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | 1 => 9
  | 2 => 7
  | 3 => 5
  | n + 4 => (seq n + seq (n + 1) + seq (n + 2) + seq (n + 3)) % 10

theorem no_appearance_1234_or_3269 : 
  ¬∃ n, seq n = 1 ∧ seq (n + 1) = 2 ∧ seq (n + 2) = 3 ∧ seq (n + 3) = 4 ∨
  seq n = 3 ∧ seq (n + 1) = 2 ∧ seq (n + 2) = 6 ∧ seq (n + 3) = 9 := 
sorry

theorem no_reappearance_1975_from_2nd_time : 
  ¬∃ n > 0, seq n = 1 ∧ seq (n + 1) = 9 ∧ seq (n + 2) = 7 ∧ seq (n + 3) = 5 :=
sorry

end no_appearance_1234_or_3269_no_reappearance_1975_from_2nd_time_l288_288019


namespace solution_set_of_inequality_l288_288342

theorem solution_set_of_inequality :
  {x : ℝ | (x - 1) / (x - 3) ≤ 0} = {x : ℝ | 1 ≤ x ∧ x < 3} := 
by
  sorry

end solution_set_of_inequality_l288_288342


namespace translation_coordinates_l288_288163

theorem translation_coordinates :
  ∀ (x y : ℤ) (a : ℤ), 
  (x, y) = (3, -4) → a = 5 → (x - a, y) = (-2, -4) :=
by
  sorry

end translation_coordinates_l288_288163


namespace smallest_period_cos_l288_288668

def smallest_positive_period (f : ℝ → ℝ) (T : ℝ) :=
  T > 0 ∧ ∀ x : ℝ, f (x + T) = f x

theorem smallest_period_cos (x : ℝ) : 
  smallest_positive_period (λ x => 2 * (Real.cos x)^2 + 1) Real.pi := 
by 
  sorry

end smallest_period_cos_l288_288668


namespace instantaneous_velocity_at_t_3_l288_288849

variable (t : ℝ)
def s (t : ℝ) : ℝ := 1 - t + t^2

theorem instantaneous_velocity_at_t_3 : 
  ∃ v, v = -1 + 2 * 3 ∧ v = 5 :=
by
  sorry

end instantaneous_velocity_at_t_3_l288_288849


namespace manny_original_marbles_l288_288087

/-- 
Let total marbles be 120, and the marbles are divided between Mario, Manny, and Mike in the ratio 4:5:6. 
Let x be the number of marbles Manny is left with after giving some marbles to his brother.
Prove that Manny originally had 40 marbles. 
-/
theorem manny_original_marbles (total_marbles : ℕ) (ratio_mario ratio_manny ratio_mike : ℕ)
    (present_marbles : ℕ) (total_parts : ℕ)
    (h_marbles : total_marbles = 120) 
    (h_ratio : ratio_mario = 4 ∧ ratio_manny = 5 ∧ ratio_mike = 6) 
    (h_total_parts : total_parts = ratio_mario + ratio_manny + ratio_mike)
    (h_manny_parts : total_marbles/total_parts * ratio_manny = 40) : 
  present_marbles = 40 := 
sorry

end manny_original_marbles_l288_288087


namespace student_first_subject_percentage_l288_288553

variable (P : ℝ)

theorem student_first_subject_percentage 
  (H1 : 80 = 80)
  (H2 : 75 = 75)
  (H3 : (P + 80 + 75) / 3 = 75) :
  P = 70 :=
by
  sorry

end student_first_subject_percentage_l288_288553


namespace algebraic_expression_value_l288_288433

theorem algebraic_expression_value (x : ℝ) (h : x^2 + 3*x - 1 = 0) :
  (x - 3)^2 - (2*x + 1)*(2*x - 1) - 3*x = 7 :=
sorry

end algebraic_expression_value_l288_288433


namespace corrected_mean_l288_288213

/-- The original mean of 20 observations is 36, an observation of 25 was wrongly recorded as 40.
    The correct mean is 35.25. -/
theorem corrected_mean 
  (Mean : ℝ)
  (Observations : ℕ)
  (IncorrectObservation : ℝ)
  (CorrectObservation : ℝ)
  (h1 : Mean = 36)
  (h2 : Observations = 20)
  (h3 : IncorrectObservation = 40)
  (h4 : CorrectObservation = 25) :
  (Mean * Observations - (IncorrectObservation - CorrectObservation)) / Observations = 35.25 :=
sorry

end corrected_mean_l288_288213


namespace find_k_l288_288136

   theorem find_k (m n : ℝ) (k : ℝ) (hm : m > 0) (hn : n > 0)
     (h1 : k = Real.log m / Real.log 2)
     (h2 : k = Real.log n / (Real.log 4))
     (h3 : k = Real.log (4 * m + 3 * n) / (Real.log 8)) :
     k = 2 :=
   by
     sorry
   
end find_k_l288_288136


namespace man_walking_rate_is_12_l288_288698

theorem man_walking_rate_is_12 (M : ℝ) (woman_speed : ℝ) (time_waiting : ℝ) (catch_up_time : ℝ) 
  (woman_speed_eq : woman_speed = 12) (time_waiting_eq : time_waiting = 1 / 6) 
  (catch_up_time_eq : catch_up_time = 1 / 6): 
  (M * catch_up_time = woman_speed * time_waiting) → M = 12 := by
  intro h
  rw [woman_speed_eq, time_waiting_eq, catch_up_time_eq] at h
  sorry

end man_walking_rate_is_12_l288_288698


namespace solution_set_of_inequality_l288_288793

theorem solution_set_of_inequality (x : ℝ) : x^2 - |x| - 2 < 0 ↔ -2 < x ∧ x < 2 := 
by
  sorry

end solution_set_of_inequality_l288_288793


namespace complete_square_b_l288_288680

theorem complete_square_b (a b x : ℝ) (h : x^2 + 6 * x - 3 = 0) : (x + a)^2 = b → b = 12 := by
  sorry

end complete_square_b_l288_288680


namespace neg_p_l288_288149

variable (x : ℝ)

def p : Prop := ∃ x : ℝ, Real.exp x - x - 1 ≤ 0

theorem neg_p : ¬p ↔ (∀ x : ℝ, Real.exp x - x - 1 > 0) :=
by
  sorry

end neg_p_l288_288149


namespace angles_on_axes_correct_l288_288468

-- Definitions for angles whose terminal sides lie on x-axis and y-axis.
def angles_on_x_axis (α : ℝ) : Prop := ∃ k : ℤ, α = k * Real.pi
def angles_on_y_axis (α : ℝ) : Prop := ∃ k : ℤ, α = k * Real.pi + Real.pi / 2

-- Combined definition for angles on the coordinate axes using Lean notation
def angles_on_axes (α : ℝ) : Prop := ∃ n : ℤ, α = n * (Real.pi / 2)

-- Theorem stating that angles on the coordinate axes are of the form nπ/2.
theorem angles_on_axes_correct : ∀ α : ℝ, (angles_on_x_axis α ∨ angles_on_y_axis α) ↔ angles_on_axes α := 
sorry -- Proof is omitted.

end angles_on_axes_correct_l288_288468


namespace angle_complement_30_l288_288898

def complement_angle (x : ℝ) : ℝ := 90 - x

theorem angle_complement_30 (x : ℝ) (h : x = complement_angle x - 30) : x = 30 :=
by
  sorry

end angle_complement_30_l288_288898


namespace pythagorean_theorem_mod_3_l288_288041

theorem pythagorean_theorem_mod_3 {x y z : ℕ} (h : x^2 + y^2 = z^2) : x % 3 = 0 ∨ y % 3 = 0 ∨ z % 3 = 0 :=
by 
  sorry

end pythagorean_theorem_mod_3_l288_288041


namespace sum_primes_between_20_and_40_l288_288812

open Nat

def primesBetween20And40 : List Nat := [23, 29, 31, 37]

theorem sum_primes_between_20_and_40 :
  (primesBetween20And40.sum = 120) :=
by
  sorry

end sum_primes_between_20_and_40_l288_288812


namespace find_principal_and_rate_l288_288684

variables (P R : ℝ)

theorem find_principal_and_rate
  (h1 : 20 = P * R * 2 / 100)
  (h2 : 22 = P * ((1 + R / 100) ^ 2 - 1)) :
  P = 50 ∧ R = 20 :=
by
  sorry

end find_principal_and_rate_l288_288684


namespace theater_ticket_difference_l288_288702

theorem theater_ticket_difference
  (O B : ℕ)
  (h1 : O + B = 355)
  (h2 : 12 * O + 8 * B = 3320) :
  B - O = 115 :=
sorry

end theater_ticket_difference_l288_288702


namespace triangle_altitudes_perfect_square_l288_288572

theorem triangle_altitudes_perfect_square
  (a b c : ℤ)
  (h : (2 * (↑a * ↑b * ↑c )) = (2 * (↑a * ↑c ) + 2 * (↑a * ↑b))) :
  ∃ k : ℤ, a^2 + b^2 + c^2 = k^2 :=
by
  sorry

end triangle_altitudes_perfect_square_l288_288572


namespace smallest_n_watches_l288_288221

variable {n d : ℕ}

theorem smallest_n_watches (h1 : d > 0)
  (h2 : 10 * n - 30 = 100) : n = 13 :=
by
  sorry

end smallest_n_watches_l288_288221


namespace total_growing_space_l288_288848

theorem total_growing_space : 
  ∀ (beds1 beds2 : ℕ) (l1 w1 l2 w2 : ℕ), 
  beds1 = 2 → l1 = 3 → w1 = 3 → 
  beds2 = 2 → l2 = 4 → w2 = 3 → 
  (beds1 * (l1 * w1) + beds2 * (l2 * w2) = 42) :=
by
  intros beds1 beds2 l1 w1 l2 w2 h_beds1 h_l1 h_w1 h_beds2 h_l2 h_w2
  rw [h_beds1, h_l1, h_w1, h_beds2, h_l2, h_w2]
  norm_num
  sorry

end total_growing_space_l288_288848


namespace cost_effectiveness_order_l288_288837

variables {cS cM cL qS qM qL : ℝ}
variables (h1 : cM = 2 * cS)
variables (h2 : qM = 0.7 * qL)
variables (h3 : qL = 3 * qS)
variables (h4 : cL = 1.2 * cM)

theorem cost_effectiveness_order :
  (cL / qL <= cM / qM) ∧ (cM / qM <= cS / qS) :=
by
  sorry

end cost_effectiveness_order_l288_288837


namespace square_of_85_l288_288110

-- Define the given variables and values
def a := 80
def b := 5
def c := a + b

theorem square_of_85:
  c = 85 → (c * c) = 7225 :=
by
  intros h
  rw h
  sorry

end square_of_85_l288_288110


namespace number_of_people_quit_l288_288309

-- Define the conditions as constants.
def initial_team_size : ℕ := 25
def new_members : ℕ := 13
def final_team_size : ℕ := 30

-- Define the question as a function.
def people_quit (Q : ℕ) : Prop :=
  initial_team_size - Q + new_members = final_team_size

-- Prove the main statement assuming the conditions.
theorem number_of_people_quit (Q : ℕ) (h : people_quit Q) : Q = 8 :=
by
  sorry -- Proof is not required, so we use sorry to skip it.

end number_of_people_quit_l288_288309


namespace domain_change_l288_288215

theorem domain_change (f : ℝ → ℝ) :
  (∀ x : ℝ, -2 ≤ x + 1 ∧ x + 1 ≤ 3) →
  (∀ x : ℝ, -2 ≤ 1 - 2 * x ∧ 1 - 2 * x ≤ 3) →
  ∀ x : ℝ, -3 / 2 ≤ x ∧ x ≤ 1 :=
by {
  sorry
}

end domain_change_l288_288215


namespace simplify_fraction_l288_288490

theorem simplify_fraction (a : ℕ) (h : a = 5) : (15 * a^4) / (75 * a^3) = 1 := 
by
  sorry

end simplify_fraction_l288_288490


namespace race_length_l288_288617

noncomputable def solve_race_length (a b c d : ℝ) : Prop :=
  (d > 0) →
  (d / a = (d - 40) / b) →
  (d / b = (d - 30) / c) →
  (d / a = (d - 65) / c) →
  d = 240

theorem race_length : ∃ (d : ℝ), solve_race_length a b c d :=
by
  use 240
  sorry

end race_length_l288_288617


namespace one_over_a_plus_one_over_b_eq_neg_one_l288_288126

theorem one_over_a_plus_one_over_b_eq_neg_one
  (a b : ℝ) (h_distinct : a ≠ b)
  (h_eq : a / b + a = b / a + b) :
  1 / a + 1 / b = -1 :=
by
  sorry

end one_over_a_plus_one_over_b_eq_neg_one_l288_288126


namespace rectangle_area_l288_288251

-- Definitions
variables {height length : ℝ} (h : height = length / 2)
variables {area perimeter : ℝ} (a : area = perimeter)

-- Problem statement
theorem rectangle_area : ∃ h : ℝ, ∃ l : ℝ, ∃ area : ℝ, 
  (l = 2 * h) ∧ (area = l * h) ∧ (area = 2 * (l + h)) ∧ (area = 18) :=
sorry

end rectangle_area_l288_288251


namespace weekly_exercise_time_l288_288351

def milesWalked := 3
def walkingSpeed := 3 -- in miles per hour
def milesRan := 10
def runningSpeed := 5 -- in miles per hour
def daysInWeek := 7

theorem weekly_exercise_time : (milesWalked / walkingSpeed + milesRan / runningSpeed) * daysInWeek = 21 := 
by
  -- The actual proof part is intentionally omitted as per the instruction
  sorry

end weekly_exercise_time_l288_288351


namespace largest_sum_of_distinct_factors_of_1764_l288_288298

theorem largest_sum_of_distinct_factors_of_1764 :
  ∃ (A B C : ℕ), A * B * C = 1764 ∧ A ≠ B ∧ B ≠ C ∧ A ≠ C ∧ A + B + C = 33 :=
by
  sorry

end largest_sum_of_distinct_factors_of_1764_l288_288298


namespace ratio_AD_DC_in_ABC_l288_288014

theorem ratio_AD_DC_in_ABC 
  (A B C D : Type)
  [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D]
  (AB BC AC : Real) 
  (hAB : AB = 6) (hBC : BC = 8) (hAC : AC = 10) 
  (BD : Real) 
  (hBD : BD = 8) 
  (AD DC : Real)
  (hAD : AD = 2 * Real.sqrt 7)
  (hDC : DC = 10 - 2 * Real.sqrt 7) :
  AD / DC = (10 * Real.sqrt 7 + 14) / 36 :=
sorry

end ratio_AD_DC_in_ABC_l288_288014


namespace charity_years_l288_288408

theorem charity_years :
  ∃! pairs : List (ℕ × ℕ), 
    (∀ (w m : ℕ), (w, m) ∈ pairs → 18 * w + 30 * m = 55 * 12) ∧
    pairs.length = 6 :=
by
  sorry

end charity_years_l288_288408


namespace Matilda_age_is_35_l288_288928

-- Definitions based on conditions
def Louis_age : ℕ := 14
def Jerica_age : ℕ := 2 * Louis_age
def Matilda_age : ℕ := Jerica_age + 7

-- Theorem to prove the question's answer is correct
theorem Matilda_age_is_35 : Matilda_age = 35 :=
by
  -- Adding proof steps
  sorry

end Matilda_age_is_35_l288_288928


namespace largest_lcm_l288_288994

def lcm_list : List ℕ := [
  Nat.lcm 15 3,
  Nat.lcm 15 5,
  Nat.lcm 15 9,
  Nat.lcm 15 10,
  Nat.lcm 15 12,
  Nat.lcm 15 15
]

theorem largest_lcm : List.maximum lcm_list = 60 := by
  sorry

end largest_lcm_l288_288994


namespace domain_of_h_l288_288863

theorem domain_of_h (x : ℝ) : |x - 5| + |x + 3| ≠ 0 := by
  sorry

end domain_of_h_l288_288863


namespace bank_card_payment_technology_order_l288_288982

-- Conditions as definitions
def action_tap := 1
def action_pay_online := 2
def action_swipe := 3
def action_insert_into_terminal := 4

-- Corresponding proof problem statement
theorem bank_card_payment_technology_order :
  [action_insert_into_terminal, action_swipe, action_tap, action_pay_online] = [4, 3, 1, 2] := by
  sorry

end bank_card_payment_technology_order_l288_288982


namespace factor_polynomial_l288_288866

theorem factor_polynomial (x : ℝ) : 54*x^3 - 135*x^5 = 27*x^3*(2 - 5*x^2) := 
by
  sorry

end factor_polynomial_l288_288866


namespace point_in_quadrant_l288_288258

theorem point_in_quadrant (a b : ℝ) (h1 : a - b > 0) (h2 : a * b < 0) : 
  (a > 0 ∧ b < 0) ∧ ¬(a > 0 ∧ b > 0) ∧ ¬(a < 0 ∧ b > 0) ∧ ¬(a < 0 ∧ b < 0) := 
by 
  sorry

end point_in_quadrant_l288_288258


namespace linear_eq_conditions_l288_288912

theorem linear_eq_conditions (m : ℤ) (h : abs m = 1) (h₂ : m + 1 ≠ 0) : m = 1 :=
by
  sorry

end linear_eq_conditions_l288_288912


namespace union_A_B_l288_288320

def set_A : Set ℝ := { x | 1 / x ≤ 0 }
def set_B : Set ℝ := { x | x^2 - 1 < 0 }

theorem union_A_B : set_A ∪ set_B = { x | x < 1 } :=
by
  sorry

end union_A_B_l288_288320


namespace find_integer_n_l288_288122

theorem find_integer_n :
  ∃ (n : ℤ), -90 ≤ n ∧ n ≤ 90 ∧ Real.sin (n * Real.pi / 180) = Real.cos (675 * Real.pi / 180) ∧ n = 45 :=
sorry

end find_integer_n_l288_288122


namespace find_solns_to_eqn_l288_288248

theorem find_solns_to_eqn (x y z w : ℕ) :
  2^x * 3^y - 5^z * 7^w = 1 ↔ (x, y, z, w) = (1, 0, 0, 0) ∨ 
                                        (x, y, z, w) = (3, 0, 0, 1) ∨ 
                                        (x, y, z, w) = (1, 1, 1, 0) ∨ 
                                        (x, y, z, w) = (2, 2, 1, 1) := 
sorry -- Placeholder for the actual proof

end find_solns_to_eqn_l288_288248


namespace youngest_child_age_is_3_l288_288401

noncomputable def family_age_problem : Prop :=
  ∃ (age_diff_2 : ℕ) (age_10_years_ago : ℕ) (new_family_members : ℕ) (same_present_avg_age : ℕ) (youngest_child_age : ℕ),
    age_diff_2 = 2 ∧
    age_10_years_ago = 4 * 24 ∧
    new_family_members = 2 ∧
    same_present_avg_age = 24 ∧
    youngest_child_age = 3 ∧
    (96 + 4 * 10 + (youngest_child_age + (youngest_child_age + age_diff_2)) = 6 * same_present_avg_age)

theorem youngest_child_age_is_3 : family_age_problem := sorry

end youngest_child_age_is_3_l288_288401


namespace max_log_value_l288_288135

noncomputable def max_log_product (a b : ℝ) : ℝ :=
  if h : a > 0 ∧ b > 0 ∧ a * b = 8 then (Real.logb 2 a) * (Real.logb 2 (2 * b)) else 0

theorem max_log_value (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : a * b = 8) :
  max_log_product a b ≤ 4 :=
sorry

end max_log_value_l288_288135


namespace distinct_solutions_subtraction_eq_two_l288_288034

theorem distinct_solutions_subtraction_eq_two :
  ∃ p q : ℝ, (p ≠ q) ∧ (p > q) ∧ ((6 * p - 18) / (p^2 + 4 * p - 21) = p + 3) ∧ ((6 * q - 18) / (q^2 + 4 * q - 21) = q + 3) ∧ (p - q = 2) :=
by
  have p := -3
  have q := -5
  exists p, q
  sorry

end distinct_solutions_subtraction_eq_two_l288_288034


namespace present_age_of_A_l288_288536

theorem present_age_of_A (A B C : ℕ) 
  (h1 : A + B + C = 57)
  (h2 : B - 3 = 2 * (A - 3))
  (h3 : C - 3 = 3 * (A - 3)) :
  A = 11 :=
sorry

end present_age_of_A_l288_288536


namespace smallest_positive_perfect_square_divisible_by_2_3_5_l288_288386

theorem smallest_positive_perfect_square_divisible_by_2_3_5 : 
  ∃ (n : ℕ), (∃ k : ℕ, n = k^2) ∧ n % 2 = 0 ∧ n % 3 = 0 ∧ n % 5 = 0 ∧ 
  ∀ m : ℕ, (∃ k : ℕ, m = k^2) ∧ m % 2 = 0 ∧ m % 3 = 0 ∧ m % 5 = 0 → n ≤ m :=
sorry

end smallest_positive_perfect_square_divisible_by_2_3_5_l288_288386


namespace josie_animal_counts_l288_288026

/-- Josie counted 80 antelopes, 34 more rabbits than antelopes, 42 fewer hyenas than 
the total number of antelopes and rabbits combined, some more wild dogs than hyenas, 
and the number of leopards was half the number of rabbits. The total number of animals 
Josie counted was 605. Prove that the difference between the number of wild dogs 
and hyenas Josie counted is 50. -/
theorem josie_animal_counts :
  ∃ (antelopes rabbits hyenas wild_dogs leopards : ℕ),
    antelopes = 80 ∧
    rabbits = antelopes + 34 ∧
    hyenas = (antelopes + rabbits) - 42 ∧
    leopards = rabbits / 2 ∧
    (antelopes + rabbits + hyenas + wild_dogs + leopards = 605) ∧
    wild_dogs - hyenas = 50 := 
by
  sorry

end josie_animal_counts_l288_288026


namespace sum_of_squares_l288_288154

variable {x y z a b c : Real}
variable (h₁ : x * y = a) (h₂ : x * z = b) (h₃ : y * z = c)
variable (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)

theorem sum_of_squares : x^2 + y^2 + z^2 = (a * b)^2 / (a * b * c) + (a * c)^2 / (a * b * c) + (b * c)^2 / (a * b * c) := 
sorry

end sum_of_squares_l288_288154


namespace g_at_neg10_l288_288480

def g (x : ℤ) : ℤ := 
  if x < -3 then 3 * x + 7 else 4 - x

theorem g_at_neg10 : g (-10) = -23 := by
  -- The proof goes here
  sorry

end g_at_neg10_l288_288480


namespace eval_expression_l288_288865

-- Define the given expression
def given_expression : ℤ := -( (16 / 2) * 12 - 75 + 4 * (2 * 5) + 25 )

-- State the desired result in a theorem
theorem eval_expression : given_expression = -86 := by
  -- Skipping the proof as per instructions
  sorry

end eval_expression_l288_288865


namespace simplify_expr_l288_288325

/-- Theorem: Simplify the expression -/
theorem simplify_expr
  (x y z w : ℝ)
  (hx : x = sqrt 3 - 1)
  (hy : y = sqrt 3 + 1)
  (hz : z = 1 - sqrt 2)
  (hw : w = 1 + sqrt 2) :
  (x ^ z / y ^ w) = 2 ^ (1 - sqrt 2) * (4 - 2 * sqrt 3) :=
by
  sorry

end simplify_expr_l288_288325


namespace compare_y_values_l288_288283

theorem compare_y_values :
  let y₁ := 2 / (-2)
  let y₂ := 2 / (-1)
  y₁ > y₂ := by sorry

end compare_y_values_l288_288283


namespace train_truck_load_l288_288804

variables (x y : ℕ)

def transport_equations (x y : ℕ) : Prop :=
  (2 * x + 5 * y = 120) ∧ (8 * x + 10 * y = 440)

def tonnage (x y : ℕ) : ℕ :=
  5 * x + 8 * y

theorem train_truck_load
  (x y : ℕ)
  (h : transport_equations x y) :
  tonnage x y = 282 :=
sorry

end train_truck_load_l288_288804


namespace unique_solution_of_log_equation_l288_288669

open Real

noncomputable def specific_log_equation (x : ℝ) : Prop := log (2 * x + 1) + log x = 1

theorem unique_solution_of_log_equation :
  ∀ x : ℝ, (x > 0) → (2 * x + 1 > 0) → specific_log_equation x → x = 2 := by
  sorry

end unique_solution_of_log_equation_l288_288669


namespace terminal_side_of_minus_330_in_first_quadrant_l288_288682

def angle_quadrant (angle : ℤ) : ℕ :=
  let reduced_angle := ((angle % 360) + 360) % 360
  if reduced_angle < 90 then 1
  else if reduced_angle < 180 then 2
  else if reduced_angle < 270 then 3
  else 4

theorem terminal_side_of_minus_330_in_first_quadrant :
  angle_quadrant (-330) = 1 :=
by
  -- We need a proof to justify the theorem, so we leave it with 'sorry' as instructed.
  sorry

end terminal_side_of_minus_330_in_first_quadrant_l288_288682


namespace quilt_patch_cost_l288_288762

-- Definitions of the conditions
def length : ℕ := 16
def width : ℕ := 20
def patch_area : ℕ := 4
def cost_first_10 : ℕ := 10
def cost_after_10 : ℕ := 5
def num_first_patches : ℕ := 10

-- Define the calculations based on the problem conditions
def quilt_area : ℕ := length * width
def total_patches : ℕ := quilt_area / patch_area
def cost_first : ℕ := num_first_patches * cost_first_10
def remaining_patches : ℕ := total_patches - num_first_patches
def cost_remaining : ℕ := remaining_patches * cost_after_10
def total_cost : ℕ := cost_first + cost_remaining

-- Statement of the proof problem
theorem quilt_patch_cost : total_cost = 450 := by
  -- Placeholder for the proof
  sorry

end quilt_patch_cost_l288_288762


namespace pencils_left_l288_288103

def total_pencils (boxes : ℕ) (pencils_per_box : ℕ) : ℕ :=
  boxes * pencils_per_box

def remaining_pencils (initial_pencils : ℕ) (pencils_given : ℕ) : ℕ :=
  initial_pencils - pencils_given

theorem pencils_left (boxes : ℕ) (pencils_per_box : ℕ) (pencils_given : ℕ)
  (h_boxes : boxes = 2) (h_pencils_per_box : pencils_per_box = 14) (h_pencils_given : pencils_given = 6) :
  remaining_pencils (total_pencils boxes pencils_per_box) pencils_given = 22 :=
by
  rw [h_boxes, h_pencils_per_box, h_pencils_given]
  norm_num
  sorry

end pencils_left_l288_288103


namespace problem_statement_b_problem_statement_c_l288_288114

def clubsuit (x y : ℝ) : ℝ := |x - y + 3|

theorem problem_statement_b :
  ∃ x y : ℝ, 3 * (clubsuit x y) ≠ clubsuit (3 * x + 3) (3 * y + 3) := by
  sorry

theorem problem_statement_c :
  ∃ x : ℝ, clubsuit x (-3) ≠ x := by
  sorry

end problem_statement_b_problem_statement_c_l288_288114


namespace find_b_l288_288772

noncomputable def p (x : ℝ) : ℝ := 3 * x - 8
noncomputable def q (x : ℝ) (b : ℝ) : ℝ := 4 * x - b

theorem find_b (b : ℝ) : p (q 3 b) = 10 → b = 6 :=
by
  unfold p q
  intro h
  sorry

end find_b_l288_288772


namespace pradeep_max_marks_l288_288181

theorem pradeep_max_marks (M : ℝ) 
  (pass_condition : 0.35 * M = 210) : M = 600 :=
sorry

end pradeep_max_marks_l288_288181


namespace sufficient_but_not_necessary_l288_288398

theorem sufficient_but_not_necessary (x : ℝ) :
  (x > 1 → x^2 + 2 * x > 0) ∧ ¬(x^2 + 2 * x > 0 → x > 1) :=
by
  sorry

end sufficient_but_not_necessary_l288_288398


namespace arrangement_of_digits_11250_l288_288159

def factorial (n : ℕ) : ℕ :=
  if h : n = 0 then
    1
  else
    n * factorial (n - 1)

def number_of_arrangements (digits : List ℕ) : ℕ :=
  let number_ends_in_0 := factorial 4 / factorial 2
  let number_ends_in_5 := 3 * (factorial 3 / factorial 2)
  number_ends_in_0 + number_ends_in_5

theorem arrangement_of_digits_11250 :
  number_of_arrangements [1, 1, 2, 5, 0] = 21 :=
by
  sorry

end arrangement_of_digits_11250_l288_288159


namespace exponential_graph_passes_through_point_l288_288659

variable (a : ℝ) (hx1 : a > 0) (hx2 : a ≠ 1)

theorem exponential_graph_passes_through_point :
  ∃ y : ℝ, (y = a^0 + 1) ∧ (y = 2) :=
sorry

end exponential_graph_passes_through_point_l288_288659


namespace probability_equivalence_l288_288601

-- Definitions for the conditions:
def total_products : ℕ := 7
def genuine_products : ℕ := 4
def defective_products : ℕ := 3

-- Function to return the probability of selecting a genuine product on the second draw, given first is defective
def probability_genuine_given_defective : ℚ := 
  (defective_products / total_products) * (genuine_products / (total_products - 1))

-- The theorem we need to prove:
theorem probability_equivalence :
  probability_genuine_given_defective = 2 / 3 :=
by
  sorry -- Proof placeholder

end probability_equivalence_l288_288601


namespace legs_heads_difference_l288_288750

variables (D C L H : ℕ)

theorem legs_heads_difference
    (hC : C = 18)
    (hL : L = 2 * D + 4 * C)
    (hH : H = D + C) :
    L - 2 * H = 36 :=
by
  have h1 : C = 18 := hC
  have h2 : L = 2 * D + 4 * C := hL
  have h3 : H = D + C := hH
  sorry

end legs_heads_difference_l288_288750


namespace factorize_xy2_minus_x_l288_288867

theorem factorize_xy2_minus_x (x y : ℝ) : xy^2 - x = x * (y - 1) * (y + 1) :=
by
  sorry

end factorize_xy2_minus_x_l288_288867


namespace no_integer_roots_of_quadratic_l288_288044

theorem no_integer_roots_of_quadratic (n : ℤ) : 
  ¬ ∃ (x : ℤ), x^2 - 16 * n * x + 7^5 = 0 := by
  sorry

end no_integer_roots_of_quadratic_l288_288044


namespace mn_eq_one_l288_288267

noncomputable def f (x : ℝ) : ℝ := |Real.log x / Real.log 2|

variables (m n : ℝ) (hmn : m < n) (hm_pos : 0 < m) (hn_pos : 0 < n) (hmn_equal : f m = f n)

theorem mn_eq_one : m * n = 1 := by
  sorry

end mn_eq_one_l288_288267


namespace rectangular_plot_width_l288_288550

theorem rectangular_plot_width :
  ∀ (length width : ℕ), 
    length = 60 → 
    ∀ (poles spacing : ℕ), 
      poles = 44 → 
      spacing = 5 → 
      2 * length + 2 * width = poles * spacing →
      width = 50 :=
by
  intros length width h_length poles spacing h_poles h_spacing h_perimeter
  rw [h_length, h_poles, h_spacing] at h_perimeter
  linarith

end rectangular_plot_width_l288_288550


namespace arithmetic_sequence_theorem_l288_288729

noncomputable def S (a : ℕ → ℝ) (n : ℕ) : ℝ := (n * (a 1 + a n)) / 2

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_theorem (a : ℕ → ℝ) (S : ℕ → ℝ)
  (h_arith_seq : arithmetic_sequence a)
  (h_sum : ∀ n, S n = (n * (a 1 + a n)) / 2)
  (h_a1_pos : a 1 > 0)
  (h_condition : -1 < a 7 / a 6 ∧ a 7 / a 6 < 0) :
  (∃ d, d < 0) ∧ (∀ n, S n > 0 → n ≤ 12) :=
sorry

end arithmetic_sequence_theorem_l288_288729


namespace product_positions_8_2_100_100_l288_288757

def num_at_position : ℕ → ℕ → ℤ
| 0, _ => 0
| n, k => 
  let remainder := k % 3
  if remainder = 1 then 1 
  else if remainder = 2 then 2
  else -3

theorem product_positions_8_2_100_100 : 
  num_at_position 8 2 * num_at_position 100 100 = -3 :=
by
  unfold num_at_position
  -- unfold necessary definition steps
  sorry

end product_positions_8_2_100_100_l288_288757


namespace estimated_percentage_negative_attitude_l288_288162

-- Define the conditions
def total_parents := 2500
def sample_size := 400
def negative_attitude := 360

-- Prove the estimated percentage of parents with a negative attitude is 90%
theorem estimated_percentage_negative_attitude : 
  (negative_attitude: ℝ) / (sample_size: ℝ) * 100 = 90 := by
  sorry

end estimated_percentage_negative_attitude_l288_288162


namespace water_added_l288_288349

theorem water_added (initial_volume : ℕ) (initial_sugar_percentage : ℝ) (final_sugar_percentage : ℝ) (V : ℝ) : 
  initial_volume = 3 →
  initial_sugar_percentage = 0.4 →
  final_sugar_percentage = 0.3 →
  V = 1 :=
by
  sorry

end water_added_l288_288349


namespace ball_more_expensive_l288_288531

theorem ball_more_expensive (B L : ℝ) (h1 : 2 * B + 3 * L = 1300) (h2 : 3 * B + 2 * L = 1200) : 
  L - B = 100 := 
sorry

end ball_more_expensive_l288_288531


namespace sequence_general_formula_l288_288257

theorem sequence_general_formula (a : ℕ → ℝ) (S : ℕ → ℝ) (n : ℕ) 
  (hS : ∀ n, S n = 3 / 2 * a n - 3) : 
  (∀ n, a n = 2 * 3 ^ n) :=
by 
  sorry

end sequence_general_formula_l288_288257


namespace maximize_revenue_l288_288840

-- Defining the revenue function
def revenue (p : ℝ) : ℝ := 200 * p - 4 * p^2

-- Defining the maximum price constraint
def price_constraint (p : ℝ) : Prop := p ≤ 40

-- Statement to be proven
theorem maximize_revenue : ∃ (p : ℝ), price_constraint p ∧ revenue p = 2500 ∧ (∀ q : ℝ, price_constraint q → revenue q ≤ revenue p) :=
sorry

end maximize_revenue_l288_288840


namespace marc_journey_fraction_l288_288486

-- Defining the problem based on identified conditions
def total_cycling_time (k : ℝ) : ℝ := 20 * k
def total_walking_time (k : ℝ) : ℝ := 60 * (1 - k)
def total_travel_time (k : ℝ) : ℝ := total_cycling_time k + total_walking_time k

theorem marc_journey_fraction:
  ∀ (k : ℝ), total_travel_time k = 52 → k = 1 / 5 :=
by
  sorry

end marc_journey_fraction_l288_288486


namespace jackson_sandwiches_l288_288623

noncomputable def total_sandwiches (weeks : ℕ) (miss_wed : ℕ) (miss_fri : ℕ) : ℕ :=
  let total_wednesdays := weeks - miss_wed
  let total_fridays := weeks - miss_fri
  total_wednesdays + total_fridays

theorem jackson_sandwiches : total_sandwiches 36 1 2 = 69 := by
  sorry

end jackson_sandwiches_l288_288623


namespace find_b_l288_288573

def f (x : ℝ) : ℝ := 5 * x - 7

theorem find_b : ∃ (b : ℝ), f b = 3 :=
by
  use 2
  show f 2 = 3
  sorry

end find_b_l288_288573


namespace sum_simplest_form_probability_eq_7068_l288_288696

/-- A jar has 15 red candies and 20 blue candies. Terry picks three candies at random,
    then Mary picks three of the remaining candies at random.
    Given that the probability that they get the same color combination (all reds or all blues, irrespective of order),
    find this probability in the simplest form. The sum of the numerator and denominator in simplest form is: 7068. -/
noncomputable def problem_statement : Nat :=
  let total_candies := 15 + 20;
  let terry_red_prob := (15 * 14 * 13) / (total_candies * (total_candies - 1) * (total_candies - 2));
  let mary_red_prob := (12 * 11 * 10) / ((total_candies - 3) * (total_candies - 4) * (total_candies - 5));
  let both_red := terry_red_prob * mary_red_prob;

  let terry_blue_prob := (20 * 19 * 18) / (total_candies * (total_candies - 1) * (total_candies - 2));
  let mary_blue_prob := (17 * 16 * 15) / ((total_candies - 3) * (total_candies - 4) * (total_candies - 5));
  let both_blue := terry_blue_prob * mary_blue_prob;

  let total_probability := both_red + both_blue;
  let simplest := 243 / 6825; -- This should be simplified form
  243 + 6825 -- Sum of numerator and denominator

theorem sum_simplest_form_probability_eq_7068 : problem_statement = 7068 :=
by sorry

end sum_simplest_form_probability_eq_7068_l288_288696


namespace evaluate_fg_of_8_l288_288478

def g (x : ℕ) : ℕ := 4 * x + 5
def f (x : ℕ) : ℕ := 6 * x - 11

theorem evaluate_fg_of_8 : f (g 8) = 211 :=
by
  sorry

end evaluate_fg_of_8_l288_288478


namespace sum_factorial_eq_l288_288316

-- Define the sum S_n for natural n: 1*1! + 2*2! + ... + n*n!
def S (n : ℕ) : ℕ := ∑ i in Finset.range (n + 1), (i + 1) * Nat.factorial (i + 1)

-- The proposition to prove
theorem sum_factorial_eq (n : ℕ) : S n = Nat.factorial (n + 1) - 1 := 
sorry

end sum_factorial_eq_l288_288316


namespace area_of_rectangle_is_432_l288_288788

/-- Define the width of the rectangle --/
def width : ℕ := 12

/-- Define the length of the rectangle, which is three times the width --/
def length : ℕ := 3 * width

/-- The area of the rectangle is length multiplied by width --/
def area : ℕ := length * width

/-- Proof problem: the area of the rectangle is 432 square meters --/
theorem area_of_rectangle_is_432 :
  area = 432 :=
sorry

end area_of_rectangle_is_432_l288_288788


namespace condition_neither_sufficient_nor_necessary_l288_288598

theorem condition_neither_sufficient_nor_necessary (p q : Prop) :
  (¬ (p ∧ q)) → (p ∨ q) → False :=
by sorry

end condition_neither_sufficient_nor_necessary_l288_288598


namespace ashton_pencils_left_l288_288104

def pencils_left (boxes : Nat) (pencils_per_box : Nat) (pencils_given : Nat) : Nat :=
  boxes * pencils_per_box - pencils_given

theorem ashton_pencils_left :
  pencils_left 2 14 6 = 22 :=
by
  sorry

end ashton_pencils_left_l288_288104


namespace investment_doubling_time_l288_288919

theorem investment_doubling_time :
  ∀ (r : ℝ) (initial_investment future_investment : ℝ),
  r = 8 →
  initial_investment = 5000 →
  future_investment = 20000 →
  (future_investment = initial_investment * 2 ^ (70 / r * 2)) →
  70 / r * 2 = 17.5 :=
by
  intros r initial_investment future_investment h_r h_initial h_future h_double
  sorry

end investment_doubling_time_l288_288919


namespace quadratic_inequality_solution_set_l288_288290

theorem quadratic_inequality_solution_set
  (a b : ℝ)
  (h1 : 2 + 3 = -a)
  (h2 : 2 * 3 = b) :
  ∀ x : ℝ, 6 * x^2 - 5 * x + 1 > 0 ↔ x < (1 / 3) ∨ x > (1 / 2) := by
  sorry

end quadratic_inequality_solution_set_l288_288290


namespace boat_speed_in_still_water_l288_288404

theorem boat_speed_in_still_water (b : ℝ) (h : (36 / (b - 2)) - (36 / (b + 2)) = 1.5) : b = 10 :=
by
  sorry

end boat_speed_in_still_water_l288_288404


namespace omega_sum_equals_one_l288_288934

variables (ω : ℂ) (h₀ : ω^5 = 1) (h₁ : ω ≠ 1)

theorem omega_sum_equals_one :
  (ω^15 + ω^18 + ω^21 + ω^24 + ω^27 + ω^30 + ω^33 + ω^36 + ω^39 + ω^42 + ω^45) = 1 :=
begin
  sorry
end

end omega_sum_equals_one_l288_288934


namespace arrangement_count_of_multiple_of_5_l288_288158

-- Define the digits and the condition that the number must be a five-digit multiple of 5
def digits := [1, 1, 2, 5, 0]
def is_multiple_of_5 (n : Nat) : Prop := n % 5 = 0

theorem arrangement_count_of_multiple_of_5 :
  ∃ (count : Nat), count = 21 ∧
  (∀ (num : List Nat), num.perm digits → is_multiple_of_5 (Nat.of_digits 10 num) → true) :=
begin
  use 21,
  split,
  { refl },
  { intros num h_perm h_multiple_of_5,
    sorry
  }
end

end arrangement_count_of_multiple_of_5_l288_288158


namespace factorize_expression_l288_288884

variable {x y : ℝ}

theorem factorize_expression : xy^2 - x = x * (y - 1) * (y + 1) := 
by
  -- Define the left-hand side of the equation
  let lhs := x * y^2 - x
  -- Define the right-hand side of the equation
  let rhs := x * (y - 1) * (y + 1)
  -- Provide the goal to prove
  show lhs = rhs
  sorry

end factorize_expression_l288_288884


namespace fixed_amount_at_least_190_l288_288520

variable (F S : ℝ)

theorem fixed_amount_at_least_190
  (h1 : S = 7750)
  (h2 : F + 0.04 * S ≥ 500) :
  F ≥ 190 := by
  sorry

end fixed_amount_at_least_190_l288_288520


namespace cos_equivalent_l288_288141

open Real

theorem cos_equivalent (alpha : ℝ) (h : sin (π / 3 + alpha) = 1 / 3) : 
  cos (5 * π / 6 + alpha) = -1 / 3 :=
sorry

end cos_equivalent_l288_288141


namespace order_wxyz_l288_288917

def w : ℕ := 2^129 * 3^81 * 5^128
def x : ℕ := 2^127 * 3^81 * 5^128
def y : ℕ := 2^126 * 3^82 * 5^128
def z : ℕ := 2^125 * 3^82 * 5^129

theorem order_wxyz : x < y ∧ y < z ∧ z < w := by
  sorry

end order_wxyz_l288_288917


namespace infinite_geometric_series_sum_l288_288853

theorem infinite_geometric_series_sum
  (a : ℚ) (r : ℚ) (h_a : a = 1) (h_r : r = 2 / 3) (h_r_abs_lt_one : |r| < 1) :
  ∑' (n : ℕ), a * r^n = 3 :=
by
  -- Import necessary lemmas and properties for infinite series
  sorry -- Proof is omitted.

end infinite_geometric_series_sum_l288_288853


namespace transform_identity_l288_288803

theorem transform_identity (a b c d : ℝ) : 
  (a^2 + b^2) * (c^2 + d^2) = (a * c + b * d)^2 + (a * d - b * c)^2 := 
sorry

end transform_identity_l288_288803


namespace angle_C_eq_pi_div_3_find_ab_values_l288_288447

noncomputable def find_angle_C (A B C : ℝ) (a b c : ℝ) : ℝ :=
  if c * Real.cos B + b * Real.cos C = 2 * a * Real.cos C then C else 0

noncomputable def find_sides_ab (A B C : ℝ) (c S : ℝ) : Set (ℝ × ℝ) :=
  if C = Real.pi / 3 ∧ c = 2 * Real.sqrt 3 ∧ S = 2 * Real.sqrt 3 then
    { (a, b) | a^4 - 20 * a^2 + 64 = 0 ∧ b = 8 / a } else
    ∅

theorem angle_C_eq_pi_div_3 (A B C : ℝ) (a b c : ℝ) :
  (c * Real.cos B + b * Real.cos C = 2 * a * Real.cos C)
  ↔ (C = Real.pi / 3) :=
sorry

theorem find_ab_values (A B C : ℝ) (c S a b : ℝ) :
  (C = Real.pi / 3) ∧ (c = 2 * Real.sqrt 3) ∧ (S = 2 * Real.sqrt 3) ∧ (a^4 - 20 * a^2 + 64 = 0) ∧ (b = 8 / a)
  ↔ ((a, b) = (2, 4) ∨ (a, b) = (4, 2)) :=
sorry

end angle_C_eq_pi_div_3_find_ab_values_l288_288447


namespace sum_of_primes_between_20_and_40_l288_288822

theorem sum_of_primes_between_20_and_40 : 
  (23 + 29 + 31 + 37) = 120 := 
by
  -- Proof goes here
sorry

end sum_of_primes_between_20_and_40_l288_288822


namespace regular_n_gon_center_inside_circle_l288_288831

-- Define a regular n-gon
structure RegularNGon (n : ℕ) :=
(center : ℝ × ℝ)
(vertices : Fin n → (ℝ × ℝ))

-- Define the condition to be able to roll and reflect the n-gon over any of its sides
def canReflectSymmetrically (n : ℕ) (g : RegularNGon n) : Prop := sorry

-- Definition of a circle with a given center and radius
structure Circle :=
(center : ℝ × ℝ)
(radius : ℝ)

-- Define the problem for determining if reflection can bring the center of n-gon inside any circle
def canCenterBeInsideCircle (n : ℕ) (g : RegularNGon n) (c : Circle) : Prop :=
  ∃ (f : ℝ × ℝ → ℝ × ℝ), -- Some function representing the reflections
    canReflectSymmetrically n g ∧ f g.center = c.center

-- State the main theorem determining for which n-gons the assertion is true
theorem regular_n_gon_center_inside_circle (n : ℕ) 
  (h : n ≠ 3 ∧ n ≠ 4 ∧ n ≠ 6) : 
  ∀ (g : RegularNGon n) (c : Circle), canCenterBeInsideCircle n g c :=
sorry

end regular_n_gon_center_inside_circle_l288_288831


namespace system_soln_l288_288733

theorem system_soln (a1 b1 a2 b2 : ℚ)
  (h1 : a1 * 3 + b1 * 6 = 21)
  (h2 : a2 * 3 + b2 * 6 = 12) :
  (3 = 3 ∧ -3 = -3) ∧ (a1 * (2 * 3 + -3) + b1 * (3 - -3) = 21) ∧ (a2 * (2 * 3 + -3) + b2 * (3 - -3) = 12) :=
by
  sorry

end system_soln_l288_288733


namespace alice_reeboks_sold_l288_288846

theorem alice_reeboks_sold
  (quota : ℝ)
  (price_adidas : ℝ)
  (price_nike : ℝ)
  (price_reeboks : ℝ)
  (num_nike : ℕ)
  (num_adidas : ℕ)
  (excess : ℝ)
  (total_sales_goal : ℝ)
  (total_sales : ℝ)
  (sales_nikes_adidas : ℝ)
  (sales_reeboks : ℝ)
  (num_reeboks : ℕ) :
  quota = 1000 →
  price_adidas = 45 →
  price_nike = 60 →
  price_reeboks = 35 →
  num_nike = 8 →
  num_adidas = 6 →
  excess = 65 →
  total_sales_goal = quota + excess →
  total_sales = 1065 →
  sales_nikes_adidas = price_nike * num_nike + price_adidas * num_adidas →
  sales_reeboks = total_sales - sales_nikes_adidas →
  num_reeboks = sales_reeboks / price_reeboks →
  num_reeboks = 9 :=
by
  intros
  sorry

end alice_reeboks_sold_l288_288846


namespace area_of_square_inscribed_in_circle_l288_288194

theorem area_of_square_inscribed_in_circle (a : ℝ) :
  ∃ S : ℝ, S = (2 * a^2) / 3 :=
sorry

end area_of_square_inscribed_in_circle_l288_288194


namespace largest_n_for_inequality_l288_288993

theorem largest_n_for_inequality :
  ∃ n : ℕ, 3 * n^2007 < 3^4015 ∧ ∀ m : ℕ, 3 * m^2007 < 3^4015 → m ≤ 8 ∧ n = 8 :=
by
  sorry

end largest_n_for_inequality_l288_288993


namespace tony_exercise_hours_per_week_l288_288352

variable (dist_walk dist_run speed_walk speed_run days_per_week : ℕ)

#eval dist_walk : ℕ := 3
#eval dist_run : ℕ := 10
#eval speed_walk : ℕ := 3
#eval speed_run : ℕ := 5
#eval days_per_week : ℕ := 7

theorem tony_exercise_hours_per_week :
  (dist_walk / speed_walk + dist_run / speed_run) * days_per_week = 21 := by
  sorry

end tony_exercise_hours_per_week_l288_288352


namespace earnings_per_hour_l288_288828

-- Define the conditions
def widgetsProduced : Nat := 750
def hoursWorked : Nat := 40
def totalEarnings : ℝ := 620
def earningsPerWidget : ℝ := 0.16

-- Define the proof goal
theorem earnings_per_hour :
  ∃ H : ℝ, (hoursWorked * H + widgetsProduced * earningsPerWidget = totalEarnings) ∧ H = 12.5 :=
by
  sorry

end earnings_per_hour_l288_288828


namespace quilt_patch_cost_is_correct_l288_288758

noncomputable def quilt_area : ℕ := 16 * 20

def patch_area : ℕ := 4

def first_10_patch_cost : ℕ := 10

def discount_patch_cost : ℕ := 5

def total_patches (quilt_area patch_area : ℕ) : ℕ := quilt_area / patch_area

def cost_for_first_10 (first_10_patch_cost : ℕ) : ℕ := 10 * first_10_patch_cost

def cost_for_discounted (total_patches first_10_patch_cost discount_patch_cost : ℕ) : ℕ :=
  (total_patches - 10) * discount_patch_cost

def total_cost (cost_for_first_10 cost_for_discounted : ℕ) : ℕ :=
  cost_for_first_10 + cost_for_discounted

theorem quilt_patch_cost_is_correct :
  total_cost (cost_for_first_10 first_10_patch_cost)
             (cost_for_discounted (total_patches quilt_area patch_area) first_10_patch_cost discount_patch_cost) = 450 :=
by
  sorry

end quilt_patch_cost_is_correct_l288_288758


namespace industrial_lubricants_percentage_l288_288694

noncomputable def percentage_microphotonics : ℕ := 14
noncomputable def percentage_home_electronics : ℕ := 19
noncomputable def percentage_food_additives : ℕ := 10
noncomputable def percentage_gmo : ℕ := 24
noncomputable def total_percentage : ℕ := 100
noncomputable def percentage_basic_astrophysics : ℕ := 25

theorem industrial_lubricants_percentage :
  total_percentage - (percentage_microphotonics + percentage_home_electronics + 
  percentage_food_additives + percentage_gmo + percentage_basic_astrophysics) = 8 := 
sorry

end industrial_lubricants_percentage_l288_288694


namespace min_value_expr_min_value_achieved_l288_288123

theorem min_value_expr (x : ℝ) (hx : x > 0) : 4*x + 1/x^4 ≥ 5 :=
by
  sorry

theorem min_value_achieved (x : ℝ) : x = 1 → 4*x + 1/x^4 = 5 :=
by
  sorry

end min_value_expr_min_value_achieved_l288_288123


namespace part_a_part_b_l288_288852

noncomputable def curl (A : Π (i : ℕ), ℝ^3 → ℝ) : ℝ^3 → ℝ^3 :=
  λ x, ⟨(∂ (A 2 x) / ∂ x.2 - ∂ (A 1 x) / ∂ x.3,
         ∂ (A 2 x) / ∂ x.1 - ∂ (A 0 x) / ∂ x.3,
         ∂ (A 1 x) / ∂ x.1 - ∂ (A 0 x) / ∂ x.2)⟩

theorem part_a (x y z : ℝ) :
  curl (λ i, match i with
                 | 0 := λ_, x
                 | 1 := λ_, -z^2
                 | _ := λ_, y^2) = (λ x, ⟨2 * x.2 + 2 * x.3, 0, 0⟩) :=
by
  ext1
  simp [curl, Function.uncurry]
  repeat { sorry }

theorem part_b (x y z : ℝ) :
  curl (λ i, match i with
                 | 0 := λ_, y * z
                 | 1 := λ_, x * z
                 | _ := λ_, x * y) = 0 :=
by
  ext1
  simp [curl, Function.uncurry]
  repeat { sorry }

end part_a_part_b_l288_288852


namespace smallest_three_digit_solution_l288_288388

theorem smallest_three_digit_solution :
  ∃ n : ℕ, 70 * n ≡ 210 [MOD 350] ∧ 100 ≤ n ∧ n = 103 :=
by
  sorry

end smallest_three_digit_solution_l288_288388


namespace ratio_proof_l288_288182

theorem ratio_proof (a b x : ℝ) (h : a > b) (h_b_pos : b > 0)
  (h_x : x = 0.5 * Real.sqrt (a / b) + 0.5 * Real.sqrt (b / a)) :
  2 * b * Real.sqrt (x^2 - 1) / (x - Real.sqrt (x^2 - 1)) = a - b := 
sorry

end ratio_proof_l288_288182


namespace trick_deck_cost_l288_288353

theorem trick_deck_cost (x : ℝ) (h1 : 6 * x + 2 * x = 64) : x = 8 :=
  sorry

end trick_deck_cost_l288_288353


namespace intersection_A_complement_is_2_4_l288_288151

-- Declare the universal set U, set A, and set B
def U : Set ℕ := { 1, 2, 3, 4, 5, 6, 7 }
def A : Set ℕ := { 2, 4, 5 }
def B : Set ℕ := { 1, 3, 5, 7 }

-- Define the complement of set B with respect to U
def complement_U_B : Set ℕ := { x ∈ U | x ∉ B }

-- Define the intersection of set A and the complement of set B
def intersection_A_complement_U_B : Set ℕ := { x ∈ A | x ∈ complement_U_B }

-- State the theorem
theorem intersection_A_complement_is_2_4 : 
  intersection_A_complement_U_B = { 2, 4 } := 
by
  sorry

end intersection_A_complement_is_2_4_l288_288151


namespace modulus_of_complex_l288_288440

noncomputable def modulus (z : Complex) : Real :=
  Complex.abs z

theorem modulus_of_complex :
  ∀ (i : Complex) (z : Complex), i = Complex.I → z = i * (2 - i) → modulus z = Real.sqrt 5 :=
by
  intros i z hi hz
  -- Proof omitted
  sorry

end modulus_of_complex_l288_288440


namespace A_and_D_independent_l288_288508

open ProbabilityTheory

-- Definition of the events
def event_A : set (ℕ × ℕ) := {p | p.1 = 1}
def event_B : set (ℕ × ℕ) := {p | p.2 = 2}
def event_C : set (ℕ × ℕ) := {p | p.1 + p.2 = 8}
def event_D : set (ℕ × ℕ) := {p | p.1 + p.2 = 7}

-- Assumption of uniform probability over the sample space
def sample_space : Finset (ℕ × ℕ) := 
  Finset.univ.pi Finset.univ

noncomputable def prob : Measure (ℕ × ℕ) :=
  uniform sample_space

-- Defining independence
def independent (P : Measure (ℕ × ℕ)) (A B : set (ℕ × ℕ)) : Prop :=
  P (A ∩ B) = P A * P B

-- Desired statement
theorem A_and_D_independent :
  independent prob event_A event_D :=
by sorry

end A_and_D_independent_l288_288508


namespace arrangement_of_11250_l288_288160

theorem arrangement_of_11250 : 
  let digits := [1, 1, 2, 5, 0]
  let total_count := 21
  let valid_arrangement (num : ℕ) : Prop := ∃ (perm : List ℕ), List.perm perm digits ∧ (num % 5 = 0) ∧ (num / 10000 ≥ 1)
  ∃ (count : ℕ), count = total_count ∧ 
  count = Nat.card {n // valid_arrangement n} := 
by 
  sorry

end arrangement_of_11250_l288_288160


namespace alice_password_prob_correct_l288_288563

noncomputable def password_probability : ℚ :=
  let even_digit_prob := 5 / 10
  let valid_symbol_prob := 3 / 5
  let non_zero_digit_prob := 9 / 10
  even_digit_prob * valid_symbol_prob * non_zero_digit_prob

theorem alice_password_prob_correct :
  password_probability = 27 / 100 := by
  rfl

end alice_password_prob_correct_l288_288563


namespace perp_bisector_chord_l288_288265

theorem perp_bisector_chord (x y : ℝ) :
  (2 * x + 3 * y + 1 = 0) ∧ (x^2 + y^2 - 2 * x + 4 * y = 0) → 
  ∃ k l m : ℝ, (3 * x - 2 * y - 7 = 0) :=
by
  sorry

end perp_bisector_chord_l288_288265


namespace length_of_hypotenuse_l288_288551

theorem length_of_hypotenuse (a b : ℝ) (h1 : a = 15) (h2 : b = 21) : 
hypotenuse_length = Real.sqrt (a^2 + b^2) :=
by
  rw [h1, h2]
  sorry

end length_of_hypotenuse_l288_288551


namespace overall_gain_percent_l288_288743

variables (C_A S_A C_B S_B : ℝ)

def cost_price_A (n : ℝ) : ℝ := n * C_A
def selling_price_A (n : ℝ) : ℝ := n * S_A

def cost_price_B (n : ℝ) : ℝ := n * C_B
def selling_price_B (n : ℝ) : ℝ := n * S_B

theorem overall_gain_percent :
  (selling_price_A 25 = cost_price_A 50) →
  (selling_price_B 30 = cost_price_B 60) →
  ((S_A - C_A) / C_A * 100 = 100) ∧ ((S_B - C_B) / C_B * 100 = 100) :=
by
  sorry

end overall_gain_percent_l288_288743


namespace greatest_integer_value_of_x_l288_288430

theorem greatest_integer_value_of_x :
  ∃ x : ℤ, (3 * |2 * x + 1| + 10 > 28) ∧ (∀ y : ℤ, 3 * |2 * y + 1| + 10 > 28 → y ≤ x) :=
sorry

end greatest_integer_value_of_x_l288_288430


namespace daniel_age_is_correct_l288_288673

open Nat

-- Define Uncle Ben's age
def uncleBenAge : ℕ := 50

-- Define Edward's age as two-thirds of Uncle Ben's age
def edwardAge : ℚ := (2 / 3) * uncleBenAge

-- Define that Daniel is 7 years younger than Edward
def danielAge : ℚ := edwardAge - 7

-- Assert that Daniel's age is 79/3 years old
theorem daniel_age_is_correct : danielAge = 79 / 3 := by
  sorry

end daniel_age_is_correct_l288_288673


namespace trig_identity_example_l288_288450

open Real

noncomputable def tan_alpha_eq_two_tan_pi_fifths (α : ℝ) :=
  tan α = 2 * tan (π / 5)

theorem trig_identity_example (α : ℝ) (h : tan_alpha_eq_two_tan_pi_fifths α) :
  (cos (α - 3 * π / 10) / sin (α - π / 5)) = 3 :=
sorry

end trig_identity_example_l288_288450


namespace ways_to_reach_5_5_l288_288699

def moves_to_destination : ℕ → ℕ → ℕ
| 0, 0     => 1
| 0, j+1   => moves_to_destination 0 j
| i+1, 0   => moves_to_destination i 0
| i+1, j+1 => moves_to_destination i (j+1) + moves_to_destination (i+1) j + moves_to_destination i j

theorem ways_to_reach_5_5 : moves_to_destination 5 5 = 1573 := by
  sorry

end ways_to_reach_5_5_l288_288699


namespace number_of_children_l288_288514

-- Definitions based on conditions from the problem
def total_spectators := 10000
def men_spectators := 7000
def spectators_other_than_men := total_spectators - men_spectators
def women_and_children_ratio := 5

-- Prove there are 2500 children
theorem number_of_children : 
  ∃ (women children : ℕ), 
    spectators_other_than_men = women + women_and_children_ratio * women ∧ 
    children = women_and_children_ratio * women ∧
    children = 2500 :=
by
  sorry

end number_of_children_l288_288514


namespace total_weight_AlF3_10_moles_l288_288356

noncomputable def molecular_weight_AlF3 (atomic_weight_Al: ℝ) (atomic_weight_F: ℝ) : ℝ :=
  atomic_weight_Al + 3 * atomic_weight_F

theorem total_weight_AlF3_10_moles :
  let atomic_weight_Al := 26.98
  let atomic_weight_F := 19.00
  let num_moles := 10
  molecular_weight_AlF3 atomic_weight_Al atomic_weight_F * num_moles = 839.8 :=
by
  sorry

end total_weight_AlF3_10_moles_l288_288356


namespace paul_money_duration_l288_288644

theorem paul_money_duration (mowing_income weed_eating_income weekly_spending money_last: ℕ) 
    (h1: mowing_income = 44) 
    (h2: weed_eating_income = 28) 
    (h3: weekly_spending = 9) 
    (h4: money_last = 8) 
    : (mowing_income + weed_eating_income) / weekly_spending = money_last := 
by
  sorry

end paul_money_duration_l288_288644


namespace segment_halving_1M_l288_288567

noncomputable def segment_halving_sum (k : ℕ) : ℕ :=
  3^k + 1

theorem segment_halving_1M : segment_halving_sum 1000000 = 3^1000000 + 1 :=
by
  sorry

end segment_halving_1M_l288_288567


namespace circle_intersection_l288_288974

noncomputable def distance (p1 p2 : ℝ × ℝ) := Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem circle_intersection (m : ℝ) :
  (∃ x y : ℝ, x^2 + y^2 = m ∧ (∃ x y : ℝ, x^2 + y^2 - 6*x + 8*y - 24 = 0)) ↔ 4 < m ∧ m < 144 :=
by
  have h1 : distance (0, 0) (3, -4) = 5 := by sorry
  have h2 : ∀ m, |7 - Real.sqrt m| < 5 ↔ 4 < m ∧ m < 144 := by sorry
  exact sorry

end circle_intersection_l288_288974


namespace product_of_integers_l288_288787

theorem product_of_integers (a b : ℤ) (h1 : Int.gcd a b = 12) (h2 : Int.lcm a b = 60) : a * b = 720 :=
sorry

end product_of_integers_l288_288787


namespace geometric_sequence_b_mn_theorem_l288_288442

noncomputable def geometric_sequence_b_mn (b : ℕ → ℝ) (c d : ℝ) (m n : ℕ) 
  (h_b : ∀ (k : ℕ), b k > 0)
  (h_seq : ∃ q : ℝ, (q ≠ 0) ∧ ∀ k : ℕ, b k = b 1 * q ^ (k - 1))
  (h_m : b m = c)
  (h_n : b n = d)
  (h_cond : n - m ≥ 2) 
  (h_nm_pos : m > 0 ∧ n > 0): Prop :=
  b (m + n) = (d ^ n / c ^ m) ^ (1 / (n - m))

-- We skip the proof using sorry.
theorem geometric_sequence_b_mn_theorem 
  (b : ℕ → ℝ) (c d : ℝ) (m n : ℕ)
  (h_b : ∀ (k : ℕ), b k > 0)
  (h_seq : ∃ q : ℝ, (q ≠ 0) ∧ ∀ k : ℕ, b k = b 1 * q ^ (k - 1))
  (h_m : b m = c)
  (h_n : b n = d)
  (h_cond : n - m ≥ 2)
  (h_nm_pos : m > 0 ∧ n > 0) : 
  b (m + n) = (d ^ n / c ^ m) ^ (1 / (n - m)) :=
sorry

end geometric_sequence_b_mn_theorem_l288_288442


namespace seven_digit_numbers_count_l288_288132

/-- Given a six-digit phone number represented by six digits A, B, C, D, E, F:
- There are 7 positions where a new digit can be inserted: before A, between each pair of consecutive digits, and after F.
- Each of these positions can be occupied by any of the 10 digits (0 through 9).
The number of seven-digit numbers that can be formed by adding one digit to the six-digit phone number is 70. -/
theorem seven_digit_numbers_count (A B C D E F : ℕ) (hA : 0 ≤ A ∧ A < 10) (hB : 0 ≤ B ∧ B < 10) 
  (hC : 0 ≤ C ∧ C < 10) (hD : 0 ≤ D ∧ D < 10) (hE : 0 ≤ E ∧ E < 10) (hF : 0 ≤ F ∧ F < 10) : 
  ∃ n : ℕ, n = 70 :=
sorry

end seven_digit_numbers_count_l288_288132


namespace trigonometric_expression_value_l288_288722

theorem trigonometric_expression_value 
  (h1 : cos (π / 2 + x) = 4 / 5) 
  (h2 : x ∈ Ioo (-π / 2) 0) :
  (sin (2 * x) - 2 * (sin x)^2) / (1 + tan x) = -168 / 25 := 
by
  -- Proof omitted; corresponds to the steps mentioned above
  sorry

end trigonometric_expression_value_l288_288722


namespace proof_equiv_l288_288650

def f (x : ℝ) : ℝ := 3 * x ^ 2 - 6 * x + 1
def g (x : ℝ) : ℝ := 2 * x - 1

theorem proof_equiv (x : ℝ) : f (g x) - g (f x) = 6 * x ^ 2 - 12 * x + 9 := by
  sorry

end proof_equiv_l288_288650


namespace positive_y_percent_y_eq_16_l288_288411

theorem positive_y_percent_y_eq_16 (y : ℝ) (hy : 0 < y) (h : 0.01 * y * y = 16) : y = 40 :=
by
  sorry

end positive_y_percent_y_eq_16_l288_288411


namespace factorization_l288_288876

theorem factorization (x y : ℝ) : (x * y^2 - x = x * (y - 1) * (y + 1)) :=
begin
  sorry
end

end factorization_l288_288876


namespace smallest_positive_perfect_square_divisible_by_2_3_5_l288_288387

theorem smallest_positive_perfect_square_divisible_by_2_3_5 : 
  ∃ (n : ℕ), (∃ k : ℕ, n = k^2) ∧ n % 2 = 0 ∧ n % 3 = 0 ∧ n % 5 = 0 ∧ 
  ∀ m : ℕ, (∃ k : ℕ, m = k^2) ∧ m % 2 = 0 ∧ m % 3 = 0 ∧ m % 5 = 0 → n ≤ m :=
sorry

end smallest_positive_perfect_square_divisible_by_2_3_5_l288_288387


namespace instantaneous_velocity_at_3s_l288_288456

theorem instantaneous_velocity_at_3s (t s v : ℝ) (hs : s = t^3) (hts : t = 3*s) : v = 27 :=
by
  sorry

end instantaneous_velocity_at_3s_l288_288456


namespace lisa_takes_72_more_minutes_than_ken_l288_288171

theorem lisa_takes_72_more_minutes_than_ken
  (ken_speed : ℕ) (lisa_speed : ℕ) (book_pages : ℕ)
  (h_ken_speed: ken_speed = 75)
  (h_lisa_speed: lisa_speed = 60)
  (h_book_pages: book_pages = 360) :
  ((book_pages / lisa_speed:ℚ) - (book_pages / ken_speed:ℚ)) * 60 = 72 :=
by
  sorry

end lisa_takes_72_more_minutes_than_ken_l288_288171


namespace sum_even_deg_coeff_l288_288062

theorem sum_even_deg_coeff (x : ℕ) : 
  (3 - 2*x)^3 * (2*x + 1)^4 = (3 - 2*x)^3 * (2*x + 1)^4 →
  (∀ (x : ℕ), (3 - 2*x)^3 * (2*1 + 1)^4 =  81 ∧ 
  (3 - 2*(-1))^3 * (2*(-1) + 1)^4 = 125 → 
  (81 + 125) / 2 = 103) :=
by
  sorry

end sum_even_deg_coeff_l288_288062


namespace x_y_sum_cube_proof_l288_288773

noncomputable def x_y_sum_cube (x y : ℝ) : ℝ := x^3 + y^3

theorem x_y_sum_cube_proof (x y : ℝ) (hx : 1 < x) (hy : 1 < y)
  (h_eq : (Real.log x / Real.log 2)^3 + (Real.log y / Real.log 3)^3 = 3 * (Real.log x / Real.log 2) * (Real.log y / Real.log 3)) :
  x_y_sum_cube x y = 307 :=
sorry

end x_y_sum_cube_proof_l288_288773


namespace cube_face_problem_l288_288561

theorem cube_face_problem (n : ℕ) (h : 0 < n) :
  ((6 * n^2) : ℚ) / (6 * n^3) = 1 / 3 → n = 3 :=
by
  sorry

end cube_face_problem_l288_288561


namespace inequality_preservation_l288_288915

theorem inequality_preservation (x y : ℝ) (h : x < y) : 2 * x < 2 * y :=
sorry

end inequality_preservation_l288_288915


namespace youngest_child_age_l288_288403

theorem youngest_child_age
  (ten_years_ago_avg_age : Nat) (family_initial_size : Nat) (present_avg_age : Nat)
  (age_difference : Nat) (age_ten_years_ago_total : Nat)
  (age_increase : Nat) (current_age_total : Nat)
  (current_family_size : Nat) (total_age_increment : Nat) :
  ten_years_ago_avg_age = 24 →
  family_initial_size = 4 →
  present_avg_age = 24 →
  age_difference = 2 →
  age_ten_years_ago_total = family_initial_size * ten_years_ago_avg_age →
  age_increase = family_initial_size * 10 →
  current_age_total = age_ten_years_ago_total + age_increase →
  current_family_size = family_initial_size + 2 →
  total_age_increment = current_family_size * present_avg_age →
  total_age_increment - current_age_total = 8 →
  ∃ (Y : Nat), Y + Y + age_difference = 8 ∧ Y = 3 :=
by
  intros
  sorry

end youngest_child_age_l288_288403


namespace total_numbers_l288_288051

-- Setting up constants and conditions
variables (n : ℕ)
variables (s1 s2 s3 : ℕ → ℝ)

-- Conditions
axiom avg_all : (s1 n + s2 n + s3 n) / n = 2.5
axiom avg_2_1 : s1 2 / 2 = 1.1
axiom avg_2_2 : s2 2 / 2 = 1.4
axiom avg_2_3 : s3 2 / 2 = 5.0

-- Proposed theorem to prove
theorem total_numbers : n = 6 :=
by
  sorry

end total_numbers_l288_288051


namespace evaluate_polynomial_103_l288_288997

theorem evaluate_polynomial_103 :
  103 ^ 4 - 4 * 103 ^ 3 + 6 * 103 ^ 2 - 4 * 103 + 1 = 108243216 :=
by
  sorry

end evaluate_polynomial_103_l288_288997


namespace prop1_prop3_l288_288172

def custom_op (x y : ℝ) : ℝ := (x + 1) * (y + 1) - 1

theorem prop1 (x y : ℝ) : custom_op x y = custom_op y x :=
by sorry

theorem prop3 (x : ℝ) : custom_op (x + 1) (x - 1) = custom_op x x - 1 :=
by sorry

end prop1_prop3_l288_288172


namespace find_xyz_l288_288611

theorem find_xyz (x y z : ℝ) 
  (h1 : x * (y + z) = 180) 
  (h2 : y * (z + x) = 192) 
  (h3 : z * (x + y) = 204) 
  (hx : 0 < x)
  (hy : 0 < y)
  (hz : 0 < z) : 
  x * y * z = 168 * Real.sqrt 6 :=
sorry

end find_xyz_l288_288611


namespace tan_sub_pi_over_four_l288_288007

theorem tan_sub_pi_over_four (θ : ℝ) (h : Real.tan θ = 2) : Real.tan (θ - Real.pi / 4) = 1 / 3 :=
by
  sorry

end tan_sub_pi_over_four_l288_288007


namespace purchase_price_of_article_l288_288973

theorem purchase_price_of_article (P M : ℝ) (h1 : M = 55) (h2 : M = 0.30 * P + 12) : P = 143.33 :=
  sorry

end purchase_price_of_article_l288_288973


namespace symmetry_center_l288_288147

theorem symmetry_center {φ : ℝ} (hφ : |φ| < Real.pi / 2) (h : 2 * Real.sin φ = Real.sqrt 3) : 
  ∃ x : ℝ, 2 * Real.sin (2 * x + φ) = 2 * Real.sin (- (2 * x + φ)) ∧ x = -Real.pi / 6 :=
by
  sorry

end symmetry_center_l288_288147


namespace sum_of_odd_coefficients_in_binomial_expansion_l288_288297

theorem sum_of_odd_coefficients_in_binomial_expansion :
  let a_0 := 1
  let a_1 := 10
  let a_2 := 45
  let a_3 := 120
  let a_4 := 210
  let a_5 := 252
  let a_6 := 210
  let a_7 := 120
  let a_8 := 45
  let a_9 := 10
  let a_10 := 1
  (a_1 + a_3 + a_5 + a_7 + a_9) = 512 := by
  sorry

end sum_of_odd_coefficients_in_binomial_expansion_l288_288297


namespace inequality_min_value_l288_288588

theorem inequality_min_value (x y : ℝ) (hx : x > 0) (hy : y > 0) : 
  ∃ m : ℝ, (x + 2 * y) * (2 / x + 1 / y) ≥ m ∧ m ≤ 8 :=
by
  sorry

end inequality_min_value_l288_288588


namespace stratified_sampling_group_l288_288544

-- Definitions of conditions
def female_students : ℕ := 24
def male_students : ℕ := 36
def selected_females : ℕ := 8
def selected_males : ℕ := 12

-- Total number of ways to select the group
def total_combinations : ℕ := Nat.choose female_students selected_females * Nat.choose male_students selected_males

-- Proof of the problem
theorem stratified_sampling_group :
  (total_combinations = Nat.choose 24 8 * Nat.choose 36 12) :=
by
  sorry

end stratified_sampling_group_l288_288544


namespace alloy_chromium_l288_288161

variable (x : ℝ)

theorem alloy_chromium (h : 0.15 * 15 + 0.08 * x = 0.101 * (15 + x)) : x = 35 := by
  sorry

end alloy_chromium_l288_288161


namespace garden_perimeter_l288_288018

theorem garden_perimeter (A : ℝ) (P : ℝ) : 
  (A = 97) → (P = 40) :=
by
  sorry

end garden_perimeter_l288_288018


namespace find_m_from_equation_l288_288145

theorem find_m_from_equation :
  ∀ (x m : ℝ), (x^2 + 2 * x - 1 = 0) → ((x + m)^2 = 2) → m = 1 :=
by
  intros x m h1 h2
  sorry

end find_m_from_equation_l288_288145


namespace men_in_second_group_l288_288539

theorem men_in_second_group (W : ℝ)
  (h1 : W = 18 * 20)
  (h2 : W = M * 30) :
  M = 12 :=
by
  sorry

end men_in_second_group_l288_288539


namespace union_of_A_and_B_l288_288613

open Set

def A : Set ℝ := {x | x > 2}
def B : Set ℝ := {x | -1 < x ∧ x < 4}

theorem union_of_A_and_B : A ∪ B = {x | x > -1} :=
by sorry

end union_of_A_and_B_l288_288613


namespace ratio_a_to_c_l288_288769

variables {a b c d : ℚ}

theorem ratio_a_to_c
  (h1 : a / b = 5 / 2)
  (h2 : c / d = 4 / 1)
  (h3 : d / b = 3 / 10) :
  a / c = 25 / 12 :=
sorry

end ratio_a_to_c_l288_288769


namespace A_intersection_B_eq_intersection_set_l288_288009

def A : Set ℝ := {x : ℝ | x * (x - 2) < 0}
def B : Set ℝ := {x : ℝ | x > 1}
def intersection_set := {x : ℝ | 1 < x ∧ x < 2}

theorem A_intersection_B_eq_intersection_set : A ∩ B = intersection_set := by
  sorry

end A_intersection_B_eq_intersection_set_l288_288009


namespace factorize_expression_l288_288886

variable {x y : ℝ}

theorem factorize_expression : xy^2 - x = x * (y - 1) * (y + 1) := 
by
  -- Define the left-hand side of the equation
  let lhs := x * y^2 - x
  -- Define the right-hand side of the equation
  let rhs := x * (y - 1) * (y + 1)
  -- Provide the goal to prove
  show lhs = rhs
  sorry

end factorize_expression_l288_288886


namespace value_of_a_l288_288112

def f (x : ℝ) : ℝ := x^2 + 2
def g (x : ℝ) : ℝ := x^2 + 2

theorem value_of_a (a : ℝ) (ha : a > 1) (h : f (g a) = 12) : 
  a = Real.sqrt (Real.sqrt 10 - 2) :=
by sorry

end value_of_a_l288_288112


namespace clock_angle_at_3_15_l288_288077

-- Conditions
def full_circle_degrees : ℕ := 360
def hour_degree : ℕ := full_circle_degrees / 12
def minute_degree : ℕ := full_circle_degrees / 60
def minute_position (m : ℕ) : ℕ := m * minute_degree
def hour_position (h m : ℕ) : ℕ := h * hour_degree + m * (hour_degree / 60)

-- Theorem to prove
theorem clock_angle_at_3_15 : (|minute_position 15 - hour_position 3 15| : ℚ) = 7.5 := by
  sorry

end clock_angle_at_3_15_l288_288077


namespace diana_can_paint_statues_l288_288083

theorem diana_can_paint_statues : (3 / 6) / (1 / 6) = 3 := 
by 
  sorry

end diana_can_paint_statues_l288_288083


namespace sum_and_product_of_reciprocals_l288_288977

theorem sum_and_product_of_reciprocals (x y : ℝ) (h_sum : x + y = 12) (h_prod : x * y = 32) :
  (1/x + 1/y = 3/8) ∧ (1/x * 1/y = 1/32) :=
by
  sorry

end sum_and_product_of_reciprocals_l288_288977


namespace incorrect_permutations_hello_l288_288569

theorem incorrect_permutations_hello :
  ∃ n, n = 5 ∧ 
       (∃ m, m = 2 ∧
        (∃ t, t = (Nat.factorial 5) / (Nat.factorial 2) - 1 ∧ t = 59)) :=
by
  exists 5
  exists 2
  exists ((Nat.factorial 5) / (Nat.factorial 2) - 1)
  sorry

end incorrect_permutations_hello_l288_288569


namespace math_problem_l288_288503

theorem math_problem (x y : ℝ) (h1 : x^7 > y^6) (h2 : y^7 > x^6) : x + y > 2 :=
sorry

end math_problem_l288_288503


namespace other_root_l288_288951

theorem other_root (m : ℝ) (x : ℝ) (hx : 3 * x ^ 2 + m * x - 7 = 0) (root1 : x = 1) :
  ∃ y : ℝ, 3 * y ^ 2 + m * y - 7 = 0 ∧ y = -7 / 3 :=
by
  sorry

end other_root_l288_288951


namespace smallest_positive_perfect_square_div_by_2_3_5_l288_288373

-- Definition capturing the conditions of the problem
def is_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

def divisible_by (n m : ℕ) : Prop :=
  m % n = 0

-- The theorem statement combining all conditions and requiring the proof of the correct answer
theorem smallest_positive_perfect_square_div_by_2_3_5 : 
  ∃ n : ℕ, 0 < n ∧ is_perfect_square n ∧ divisible_by 2 n ∧ divisible_by 3 n ∧ divisible_by 5 n ∧ n = 900 :=
sorry

end smallest_positive_perfect_square_div_by_2_3_5_l288_288373


namespace other_root_l288_288955

theorem other_root (m : ℝ) :
  ∃ r, (r = -7 / 3) ∧ (3 * 1 ^ 2 + m * 1 - 7 = 0) := 
begin
  use -7 / 3,
  split,
  { refl },
  { linarith }
end

end other_root_l288_288955


namespace gross_profit_percentage_l288_288975

theorem gross_profit_percentage (sales_price gross_profit : ℝ) (h_sales_price : sales_price = 91) (h_gross_profit : gross_profit = 56) :
  (gross_profit / (sales_price - gross_profit)) * 100 = 160 :=
by
  sorry

end gross_profit_percentage_l288_288975


namespace sum_of_primes_between_20_and_40_l288_288821

theorem sum_of_primes_between_20_and_40 : 
  (23 + 29 + 31 + 37) = 120 := 
by
  -- Proof goes here
sorry

end sum_of_primes_between_20_and_40_l288_288821


namespace hexagon_area_l288_288472

noncomputable def area_of_hexagon (P Q R P' Q' R' : Point) (radius : ℝ) : ℝ :=
  -- a placeholder for the actual area calculation
  sorry 

theorem hexagon_area (P Q R P' Q' R' : Point) 
  (radius : ℝ) (perimeter : ℝ) :
  radius = 9 → perimeter = 42 →
  area_of_hexagon P Q R P' Q' R' radius = 189 := by
  intros h1 h2
  sorry

end hexagon_area_l288_288472


namespace average_percentage_decrease_l288_288219

theorem average_percentage_decrease (p1 p2 : ℝ) (n : ℕ) (h₀ : p1 = 2000) (h₁ : p2 = 1280) (h₂ : n = 2) :
  ((p1 - p2) / p1 * 100) / n = 18 := 
by
  sorry

end average_percentage_decrease_l288_288219


namespace smallest_z_l288_288925

-- Given conditions
def distinct_consecutive_even_positive_perfect_cubes (w x y z : ℕ) : Prop :=
  w^3 + x^3 + y^3 = z^3 ∧
  ∃ a b c d : ℕ, 
    a < b ∧ b < c ∧ c < d ∧
    2 * a = w ∧ 2 * b = x ∧ 2 * c = y ∧ 2 * d = z

-- The smallest value of z proving the equation holds
theorem smallest_z (w x y z : ℕ) (h : distinct_consecutive_even_positive_perfect_cubes w x y z) : z = 12 :=
  sorry

end smallest_z_l288_288925


namespace point_in_first_quadrant_l288_288164

/-- In the Cartesian coordinate system, if a point P has x-coordinate 2 and y-coordinate 4, it lies in the first quadrant. -/
theorem point_in_first_quadrant (x y : ℝ) (h1 : x = 2) (h2 : y = 4) : 
  x > 0 ∧ y > 0 → 
  (x, y).1 = 2 ∧ (x, y).2 = 4 → 
  (x > 0 ∧ y > 0) := 
by
  intros
  sorry

end point_in_first_quadrant_l288_288164


namespace inequality_solution_l288_288963

theorem inequality_solution (x : ℝ) :
  (x - 3) / (x^2 + 4 * x + 13) ≥ 0 ↔ x ∈ Set.Ici 3 :=
by
  sorry

end inequality_solution_l288_288963


namespace num_three_digit_powers_of_three_l288_288605

theorem num_three_digit_powers_of_three : 
  ∃ n1 n2 : ℕ, 100 ≤ 3^n1 ∧ 3^n1 ≤ 999 ∧ 100 ≤ 3^n2 ∧ 3^n2 ≤ 999 ∧ n1 ≠ n2 ∧ 
  (∀ n : ℕ, 100 ≤ 3^n ∧ 3^n ≤ 999 → n = n1 ∨ n = n2) :=
sorry

end num_three_digit_powers_of_three_l288_288605


namespace remainder_of_4000th_term_l288_288421

def sequence_term_position (n : ℕ) : ℕ :=
  n^2

def sum_of_squares_up_to (n : ℕ) : ℕ :=
  (n * (n + 1) * (2 * n + 1)) / 6

theorem remainder_of_4000th_term : 
  ∃ n : ℕ, sum_of_squares_up_to n ≥ 4000 ∧ (n-1) * n * (2 * (n-1) + 1) / 6 < 4000 ∧ (n % 7) = 1 :=
by 
  sorry

end remainder_of_4000th_term_l288_288421


namespace probability_exactly_three_germinate_probability_at_least_three_germinate_l288_288498

noncomputable def binom (n k : ℕ) : ℚ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

theorem probability_exactly_three_germinate :
  let n := 4
  let k := 3
  let p := 0.9
  let q := 0.1
  (binom n k) * (p^k) * (q^(n - k)) = 0.2916 := 
by
  sorry

theorem probability_at_least_three_germinate :
  let n := 4
  let p := 0.9
  let q := 0.1
  let p_3 := (binom n 3) * (p^3) * (q^(n - 3))
  let p_4 := (binom n 4) * (p^4) * (q^(n - 4))
  p_3 + p_4 = 0.9477 :=
by
  sorry

end probability_exactly_three_germinate_probability_at_least_three_germinate_l288_288498


namespace studentC_spending_l288_288348

-- Definitions based on the problem conditions

-- Prices of Type A and Type B notebooks, respectively
variables (x y : ℝ)

-- Number of each type of notebook bought by Student A
def studentA : Prop := x + y = 3

-- Number of Type A notebooks bought by Student B
variables (a : ℕ)

-- Total cost and number of notebooks bought by Student B
def studentB : Prop := (x * a + y * (8 - a) = 11)

-- Constraints on the number of Type A and B notebooks bought by Student C
def studentC_notebooks : Prop := ∃ b : ℕ, b = 8 - a ∧ b = a

-- The total amount spent by Student C
def studentC_cost : ℝ := (8 - a) * x + a * y

-- The statement asserting the cost is 13 yuan
theorem studentC_spending (x y : ℝ) (a : ℕ) (hA : studentA x y) (hB : studentB x y a) (hC : studentC_notebooks a) : studentC_cost x y a = 13 := sorry

end studentC_spending_l288_288348


namespace roots_of_quadratic_solve_inequality_l288_288270

theorem roots_of_quadratic (a b : ℝ) (h1 : ∀ x : ℝ, a * x^2 - 3 * x + 2 > 0 ↔ x < 1 ∨ x > b) :
  a = 1 ∧ b = 2 :=
by
  sorry

theorem solve_inequality (c : ℝ) :
  let a := 1
  let b := 2
  ∀ x : ℝ, a * x^2 - (a * c + b) * x + b * x < 0 ↔
    (c > 0 → (0 < x ∧ x < c)) ∧
    (c = 0 → false) ∧
    (c < 0 → (c < x ∧ x < 0)) :=
by
  sorry

end roots_of_quadratic_solve_inequality_l288_288270


namespace shifted_parabola_relationship_l288_288658

-- Step a) and conditions
def original_function (x : ℝ) : ℝ := -2 * x ^ 2 + 4

def shift_left (f : ℝ → ℝ) (a : ℝ) : ℝ → ℝ := fun x => f (x + a)
def shift_up (f : ℝ → ℝ) (b : ℝ) : ℝ → ℝ := fun x => f x + b

-- Step c) encoding the proof problem
theorem shifted_parabola_relationship :
  (shift_up (shift_left original_function 2) 3 = fun x => -2 * (x + 2) ^ 2 + 7) :=
by
  sorry

end shifted_parabola_relationship_l288_288658


namespace area_of_ABCD_l288_288618

theorem area_of_ABCD 
  (AB CD DA: ℝ) (angle_CDA: ℝ) (a b c: ℕ) 
  (H1: AB = 10) 
  (H2: BC = 6) 
  (H3: CD = 13) 
  (H4: DA = 13) 
  (H5: angle_CDA = 45) 
  (H_area: a = 8 ∧ b = 30 ∧ c = 2) :

  ∃ (a b c : ℝ), a + b + c = 40 := 
by
  sorry

end area_of_ABCD_l288_288618


namespace sum_even_integers_12_to_46_l288_288527

theorem sum_even_integers_12_to_46 : 
  let a1 := 12
  let d := 2
  let an := 46
  let n := (an - a1) / d + 1
  let Sn := n * (a1 + an) / 2
  Sn = 522 := 
by
  let a1 := 12 
  let d := 2 
  let an := 46
  let n := (an - a1) / d + 1 
  let Sn := n * (a1 + an) / 2
  sorry

end sum_even_integers_12_to_46_l288_288527


namespace probability_satisfies_condition_l288_288173

-- Define the sets for x and y
def X : Finset ℤ := { -1, 1 }
def Y : Finset ℤ := { -2, 0, 2 }

-- Define the condition to check
def satisfies_condition (x y : ℤ) : Prop :=
  x + 2 * y ≥ 1

-- Calculate the probability
theorem probability_satisfies_condition : (Finset.card ((X.product Y).filter (λ p, satisfies_condition p.1 p.2))).toRat / (X.card * Y.card) = 1 / 2 :=
by
  sorry

end probability_satisfies_condition_l288_288173


namespace min_bdf_proof_exists_l288_288214

noncomputable def minBDF (a b c d e f : ℕ) (A : ℕ) :=
  (A = 3 * a ∧ A = 4 * c ∧ A = 5 * e) →
  (a / b * c / d * e / f = A) →
  b * d * f = 60

theorem min_bdf_proof_exists :
  ∃ (a b c d e f A : ℕ), minBDF a b c d e f A :=
by
  sorry

end min_bdf_proof_exists_l288_288214


namespace smallest_perfect_square_divisible_by_2_3_5_l288_288362

theorem smallest_perfect_square_divisible_by_2_3_5 : 
  ∃ n : ℕ, n > 0 ∧ (∃ m : ℕ, n = m * m) ∧ (2 ∣ n) ∧ (3 ∣ n) ∧ (5 ∣ n) ∧
         (∀ k : ℕ, 
           (k > 0 ∧ (∃ l : ℕ, k = l * l) ∧ (2 ∣ k) ∧ (3 ∣ k) ∧ (5 ∣ k)) → n ≤ k) :=
sorry

end smallest_perfect_square_divisible_by_2_3_5_l288_288362


namespace jake_present_weight_l288_288211

variables (J S : ℕ)

theorem jake_present_weight :
  (J - 33 = 2 * S) ∧ (J + S = 153) → J = 113 :=
by
  sorry

end jake_present_weight_l288_288211


namespace xy_squared_l288_288455

theorem xy_squared (x y : ℚ) (h1 : x + y = 9 / 20) (h2 : x - y = 1 / 20) :
  x^2 - y^2 = 9 / 400 :=
by
  sorry

end xy_squared_l288_288455


namespace cos_2alpha_plus_pi_over_3_l288_288451

open Real

theorem cos_2alpha_plus_pi_over_3 
  (alpha : ℝ) 
  (h1 : cos (alpha - π / 12) = 3 / 5) 
  (h2 : 0 < alpha ∧ alpha < π / 2) : 
  cos (2 * alpha + π / 3) = -24 / 25 := 
sorry

end cos_2alpha_plus_pi_over_3_l288_288451


namespace intersection_of_A_and_B_l288_288632

-- Definitions of sets A and B
def A : Set ℝ := { x | x ≥ 1 }
def B : Set ℝ := { x | -1 < x ∧ x < 2 }

-- The theorem we want to prove
theorem intersection_of_A_and_B : A ∩ B = { x | 1 ≤ x ∧ x < 2 } :=
by
  sorry

end intersection_of_A_and_B_l288_288632


namespace problem_l288_288275

variable (a b : ℝ)

theorem problem (h₁ : a + b = 2) (h₂ : a - b = 3) : a^2 - b^2 = 6 := by
  sorry

end problem_l288_288275


namespace compute_cubic_sum_l288_288140

theorem compute_cubic_sum (x y : ℝ) (h1 : 1 / x + 1 / y = 4) (h2 : x * y + x ^ 2 + y ^ 2 = 17) : x ^ 3 + y ^ 3 = 52 :=
sorry

end compute_cubic_sum_l288_288140


namespace find_m_value_l288_288448

theorem find_m_value (f g : ℝ → ℝ) (m : ℝ)
  (h1 : ∀ (x : ℝ), f x = 4 * x^2 - 3 * x + 5)
  (h2 : ∀ (x : ℝ), g x = 2 * x^2 - m * x + 8)
  (h3 : f 5 - g 5 = 15) :
  m = -17 / 5 :=
by
  sorry

end find_m_value_l288_288448


namespace proof_inequality_l288_288596

noncomputable def proof_problem (a b c d : ℝ) (h_ab : a * b + b * c + c * d + d * a = 1) : Prop :=
  (a^3 / (b + c + d)) + (b^3 / (a + c + d)) + (c^3 / (a + b + d)) + (d^3 / (a + b + c)) ≥ 1 / 3

theorem proof_inequality (a b c d : ℝ) (h_nonneg : 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ 0 ≤ d) 
  (h_ab : a * b + b * c + c * d + d * a = 1) : 
  proof_problem a b c d h_ab := 
by
  sorry

end proof_inequality_l288_288596


namespace ratio_S15_S5_l288_288482

-- Definition of a geometric sequence sum and the given ratio S10/S5 = 1/2
noncomputable def geom_sum : ℕ → ℕ := sorry
axiom ratio_S10_S5 : geom_sum 10 / geom_sum 5 = 1 / 2

-- The goal is to prove that the ratio S15/S5 = 3/4
theorem ratio_S15_S5 : geom_sum 15 / geom_sum 5 = 3 / 4 :=
by sorry

end ratio_S15_S5_l288_288482


namespace find_number_l288_288216

theorem find_number (a b some_number : ℕ) (h1 : a = 69842) (h2 : b = 30158) (h3 : (a^2 - b^2) / some_number = 100000) : some_number = 39684 :=
by {
  -- Proof skipped
  sorry
}

end find_number_l288_288216


namespace jills_uncles_medicine_last_time_l288_288764

theorem jills_uncles_medicine_last_time :
  let pills := 90
  let third_of_pill_days := 3
  let days_per_full_pill := 9
  let days_per_month := 30
  let total_days := pills * days_per_full_pill
  let total_months := total_days / days_per_month
  total_months = 27 :=
by {
  sorry
}

end jills_uncles_medicine_last_time_l288_288764


namespace university_admissions_l288_288118

def students : Fin 5 := ⟦0, 1, 2, 3, 4⟧ -- Representing students A, B, and others via indices
def universities : Fin 3 := ⟦0, 1, 2⟧ -- Representing universities via indices (0, 1, 2)

theorem university_admissions :
  (number_of_distributions : 
    ∃ f : Fin 5 → Fin 3,
      (∀ u : Fin 3, ∃ s : Finset (Fin 5), ∃ x : Fin 5, x ∈ s ∧ f x = u)
  ) = 150 := 
sorry

end university_admissions_l288_288118


namespace product_of_consecutive_integers_l288_288938

theorem product_of_consecutive_integers (a b : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_less : a < b) :
  ∃ (x y : ℕ), x ≠ y ∧ x * y % (a * b) = 0 :=
by
  sorry

end product_of_consecutive_integers_l288_288938


namespace leaves_fall_l288_288557

theorem leaves_fall (planned_trees : ℕ) (tree_multiplier : ℕ) (leaves_per_tree : ℕ) (h1 : planned_trees = 7) (h2 : tree_multiplier = 2) (h3 : leaves_per_tree = 100) :
  (planned_trees * tree_multiplier) * leaves_per_tree = 1400 :=
by
  rw [h1, h2, h3]
  -- Additional step suggestions for interactive proof environments, e.g.,
  -- Have: 7 * 2 = 14
  -- Goal: 14 * 100 = 1400
  sorry

end leaves_fall_l288_288557


namespace sixth_power_sum_l288_288196

theorem sixth_power_sum (a b c d e f : ℤ) :
  a^6 + b^6 + c^6 + d^6 + e^6 + f^6 = 6 * a * b * c * d * e * f + 1 → 
  (a = 1 ∨ a = -1 ∨ b = 1 ∨ b = -1 ∨ c = 1 ∨ c = -1 ∨ 
   d = 1 ∨ d = -1 ∨ e = 1 ∨ e = -1 ∨ f = 1 ∨ f = -1) ∧
  ((a = 1 ∨ a = -1 ∨ a = 0) ∧ 
   (b = 1 ∨ b = -1 ∨ b = 0) ∧ 
   (c = 1 ∨ c = -1 ∨ c = 0) ∧ 
   (d = 1 ∨ d = -1 ∨ d = 0) ∧ 
   (e = 1 ∨ e = -1 ∨ e = 0) ∧ 
   (f = 1 ∨ f = -1 ∨ f = 0)) ∧
  (a ≠ 0 ∨ b ≠ 0 ∨ c ≠ 0 ∨ d ≠ 0 ∨ e ≠ 0 ∨ f ≠ 0) ∧
  (a = 0 ∨ b = 0 ∨ c = 0 ∨ d = 0 ∨ e = 0 ∨ f = 0) := 
sorry

end sixth_power_sum_l288_288196


namespace simplify_expression_l288_288188

theorem simplify_expression {x a : ℝ} (h1 : x > a) (h2 : x ≠ 0) (h3 : a ≠ 0) :
  (x * (x^2 - a^2)⁻¹ + 1) / (a * (x - a)⁻¹ + (x - a)^(1 / 2))
  / ((a^2 * (x + a)^(1 / 2)) / (x - (x^2 - a^2)^(1 / 2)) + 1 / (x^2 - a * x))
  = 2 / (x^2 - a^2) :=
by sorry

end simplify_expression_l288_288188


namespace shortest_distance_between_circles_l288_288525

theorem shortest_distance_between_circles :
  let c1 := (1, -3)
  let r1 := 2 * Real.sqrt 2
  let c2 := (-3, 1)
  let r2 := 1
  let distance_centers := Real.sqrt ((1 - -3)^2 + (-3 - 1)^2)
  let shortest_distance := distance_centers - (r1 + r2)
  shortest_distance = 2 * Real.sqrt 2 - 1 :=
by
  sorry

end shortest_distance_between_circles_l288_288525


namespace ab_value_l288_288741

/-- 
  Given the conditions:
  - a - b = 10
  - a^2 + b^2 = 210
  Prove that ab = 55.
-/
theorem ab_value (a b : ℝ) (h1 : a - b = 10) (h2 : a^2 + b^2 = 210) : a * b = 55 :=
by
  sorry

end ab_value_l288_288741


namespace cyclic_sum_inequality_l288_288942

theorem cyclic_sum_inequality (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) :
  ( (b + c - a)^2 / (a^2 + (b + c)^2) +
    (c + a - b)^2 / (b^2 + (c + a)^2) +
    (a + b - c)^2 / (c^2 + (a + b)^2) ) ≥ 3 / 5 :=
  sorry

end cyclic_sum_inequality_l288_288942


namespace smallest_positive_perfect_square_divisible_by_2_3_5_l288_288383

theorem smallest_positive_perfect_square_divisible_by_2_3_5 : 
  ∃ (n : ℕ), (∃ k : ℕ, n = k^2) ∧ n % 2 = 0 ∧ n % 3 = 0 ∧ n % 5 = 0 ∧ 
  ∀ m : ℕ, (∃ k : ℕ, m = k^2) ∧ m % 2 = 0 ∧ m % 3 = 0 ∧ m % 5 = 0 → n ≤ m :=
sorry

end smallest_positive_perfect_square_divisible_by_2_3_5_l288_288383


namespace N_is_even_l288_288926

def sum_of_digits : ℕ → ℕ := sorry

theorem N_is_even 
  (N : ℕ)
  (h1 : sum_of_digits N = 100)
  (h2 : sum_of_digits (5 * N) = 50) : 
  Even N :=
sorry

end N_is_even_l288_288926


namespace max_sum_x_y_min_diff_x_y_l288_288980

def circle_points (x y : ℤ) : Prop := (x - 1)^2 + (y + 2)^2 = 36

theorem max_sum_x_y : ∃ (x y : ℤ), circle_points x y ∧ (∀ (x' y' : ℤ), circle_points x' y' → x + y ≥ x' + y') :=
  by sorry

theorem min_diff_x_y : ∃ (x y : ℤ), circle_points x y ∧ (∀ (x' y' : ℤ), circle_points x' y' → x - y ≤ x' - y') :=
  by sorry

end max_sum_x_y_min_diff_x_y_l288_288980


namespace sequence_sum_l288_288424

theorem sequence_sum (r x y : ℝ) (h1 : r = 1/4) 
  (h2 : x = 256 * r)
  (h3 : y = x * r) : x + y = 80 :=
by
  sorry

end sequence_sum_l288_288424


namespace village_household_count_l288_288747

theorem village_household_count
  (H : ℕ)
  (water_per_household_per_month : ℕ := 20)
  (total_water : ℕ := 2000)
  (duration_months : ℕ := 10)
  (total_consumption_condition : water_per_household_per_month * H * duration_months = total_water) :
  H = 10 :=
by
  sorry

end village_household_count_l288_288747


namespace smallest_positive_perfect_square_divisible_by_2_3_5_l288_288378

/-- Problem statement: 
    The smallest positive perfect square that is divisible by 2, 3, and 5 is 900.
-/
theorem smallest_positive_perfect_square_divisible_by_2_3_5 : 
  ∃ n : ℕ, 0 < n ∧ n * n % 2 = 0 ∧ n * n % 3 = 0 ∧ n * n % 5 = 0 ∧ n * n = 900 := 
by
  sorry

end smallest_positive_perfect_square_divisible_by_2_3_5_l288_288378


namespace additional_slow_workers_needed_l288_288749

-- Definitions based on conditions
def production_per_worker_fast (m : ℕ) (n : ℕ) (a : ℕ) : ℚ := m / (n * a)
def production_per_worker_slow (m : ℕ) (n : ℕ) (b : ℕ) : ℚ := m / (n * b)

def required_daily_production (p : ℕ) (q : ℕ) : ℚ := p / q

def contribution_fast_workers (m : ℕ) (n : ℕ) (a : ℕ) (c : ℕ) : ℚ :=
  (m * c) / (n * a)

def remaining_production (p : ℕ) (q : ℕ) (m : ℕ) (n : ℕ) (a : ℕ) (c : ℕ) : ℚ :=
  (p / q) - ((m * c) / (n * a))

def required_slow_workers (p : ℕ) (q : ℕ) (m : ℕ) (n : ℕ) (a : ℕ) (b : ℕ) (c : ℕ) : ℚ :=
  ((p * n * a - q * m * c) * b) / (q * m * a)

theorem additional_slow_workers_needed (m n a b p q c : ℕ) :
  required_slow_workers p q m n a b c = ((p * n * a - q * m * c) * b) / (q * m * a) := by
  sorry

end additional_slow_workers_needed_l288_288749


namespace unique_solutions_of_system_l288_288241

def system_of_equations (x y : ℝ) : Prop :=
  (x - 1) * (x - 2) * (x - 3) = 0 ∧
  (|x - 1| + |y - 1|) * (|x - 2| + |y - 2|) * (|x - 3| + |y - 4|) = 0

theorem unique_solutions_of_system :
  ∀ (x y : ℝ), system_of_equations x y ↔ (x = 1 ∧ y = 1) ∨ (x = 2 ∧ y = 2) ∨ (x = 3 ∧ y = 4) :=
by sorry

end unique_solutions_of_system_l288_288241


namespace math_problem_l288_288823

theorem math_problem 
  (num := 1 * 2 * 3 * 4 * 5 * 6 * 7)
  (den := 1 + 2 + 3 + 4 + 5 + 6 + 7) :
  (num / den) = 180 :=
by
  sorry

end math_problem_l288_288823


namespace problem_c_d_sum_l288_288476

theorem problem_c_d_sum (C D : ℝ) (h : ∀ x : ℝ, x ≠ 3 → (C / (x - 3) + D * (x - 2) = (5 * x ^ 2 - 8 * x - 6) / (x - 3))) : C + D = 20 :=
sorry

end problem_c_d_sum_l288_288476


namespace find_coefficients_l288_288010

variables {V : Type*} [AddCommGroup V] [Module ℝ V]

-- Definitions based on conditions
def A' (A B : V) : V := (3 : ℝ) • (B - A) + A
def B' (B C : V) : V := (3 : ℝ) • (C - B) + C

-- The problem statement
theorem find_coefficients (A A' B B' : V) (p q r : ℝ) 
  (hB : B = (1/4 : ℝ) • A + (3/4 : ℝ) • A') 
  (hC : C = (1/4 : ℝ) • B + (3/4 : ℝ) • B') : 
  ∃ (p q r : ℝ), A = p • A' + q • B + r • B' ∧ p = 4/13 ∧ q = 12/13 ∧ r = 48/13 :=
sorry

end find_coefficients_l288_288010


namespace quadratics_roots_l288_288636

theorem quadratics_roots (m n : ℝ) (r₁ r₂ : ℝ) 
  (h₁ : r₁^2 - m * r₁ + n = 0) (h₂ : r₂^2 - m * r₂ + n = 0) 
  (p q : ℝ) (h₃ : (r₁^2 - r₂^2)^2 + p * (r₁^2 - r₂^2) + q = 0) :
  p = 0 ∧ q = -m^4 + 4 * m^2 * n := 
sorry

end quadratics_roots_l288_288636


namespace percentage_apples_basket_l288_288069

theorem percentage_apples_basket :
  let initial_apples := 10
  let initial_oranges := 5
  let added_oranges := 5
  let total_apples := initial_apples
  let total_oranges := initial_oranges + added_oranges
  let total_fruits := total_apples + total_oranges
  (total_apples / total_fruits) * 100 = 50 :=
by
  sorry

end percentage_apples_basket_l288_288069


namespace count_two_digit_even_congruent_to_1_mod_4_l288_288734

theorem count_two_digit_even_congruent_to_1_mod_4 : 
  ∃ (S : Finset ℕ), (∀ n ∈ S, n % 4 = 1 ∧ 10 ≤ n ∧ n ≤ 98 ∧ n % 2 = 0) ∧ S.card = 23 := 
sorry

end count_two_digit_even_congruent_to_1_mod_4_l288_288734


namespace smallest_positive_perfect_square_divisible_by_2_3_5_l288_288380

/-- Problem statement: 
    The smallest positive perfect square that is divisible by 2, 3, and 5 is 900.
-/
theorem smallest_positive_perfect_square_divisible_by_2_3_5 : 
  ∃ n : ℕ, 0 < n ∧ n * n % 2 = 0 ∧ n * n % 3 = 0 ∧ n * n % 5 = 0 ∧ n * n = 900 := 
by
  sorry

end smallest_positive_perfect_square_divisible_by_2_3_5_l288_288380


namespace smallest_positive_perfect_square_div_by_2_3_5_l288_288376

-- Definition capturing the conditions of the problem
def is_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

def divisible_by (n m : ℕ) : Prop :=
  m % n = 0

-- The theorem statement combining all conditions and requiring the proof of the correct answer
theorem smallest_positive_perfect_square_div_by_2_3_5 : 
  ∃ n : ℕ, 0 < n ∧ is_perfect_square n ∧ divisible_by 2 n ∧ divisible_by 3 n ∧ divisible_by 5 n ∧ n = 900 :=
sorry

end smallest_positive_perfect_square_div_by_2_3_5_l288_288376


namespace peggy_dolls_after_all_events_l288_288957

def initial_dolls : Nat := 6
def grandmother_gift : Nat := 28
def birthday_gift : Nat := grandmother_gift / 2
def lost_dolls (total : Nat) : Nat := (10 * total + 9) / 100  -- using integer division for rounding 10% up
def easter_gift : Nat := (birthday_gift + 2) / 3  -- using integer division for rounding one-third up
def friend_exchange_gain : Int := -1  -- gaining 1 doll but losing 2
def christmas_gift (easter_dolls : Nat) : Nat := (20 * easter_dolls) / 100 + easter_dolls  -- 20% more dolls
def ruined_dolls : Nat := 3

theorem peggy_dolls_after_all_events : initial_dolls + grandmother_gift + birthday_gift - lost_dolls (initial_dolls + grandmother_gift + birthday_gift) + easter_gift + friend_exchange_gain.toNat + christmas_gift easter_gift - ruined_dolls = 50 :=
by
  sorry

end peggy_dolls_after_all_events_l288_288957


namespace joan_needs_more_flour_l288_288300

-- Definitions for the conditions
def total_flour : ℕ := 7
def flour_added : ℕ := 3

-- The theorem stating the proof problem
theorem joan_needs_more_flour : total_flour - flour_added = 4 :=
by
  sorry

end joan_needs_more_flour_l288_288300


namespace students_suggesting_bacon_l288_288851

theorem students_suggesting_bacon (S : ℕ) (M : ℕ) (h1: S = 310) (h2: M = 185) : S - M = 125 := 
by
  -- proof here
  sorry

end students_suggesting_bacon_l288_288851


namespace greatest_possible_n_l288_288457

theorem greatest_possible_n (n : ℤ) (h : 93 * n^3 ≤ 145800) : n ≤ 11 :=
by {
  sorry
}

end greatest_possible_n_l288_288457


namespace problem_lean_l288_288964

theorem problem_lean :
  (86 * 95 * 107) % 20 = 10 :=
sorry

end problem_lean_l288_288964


namespace smallest_four_digit_multiple_of_8_with_digit_sum_20_l288_288679

def sum_of_digits (n : Nat) : Nat :=
  Nat.digits 10 n |>.foldl (· + ·) 0

theorem smallest_four_digit_multiple_of_8_with_digit_sum_20:
  ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 8 = 0 ∧ sum_of_digits n = 20 ∧ 
  ∀ m : ℕ, 1000 ≤ m ∧ m < 10000 ∧ m % 8 = 0 ∧ sum_of_digits m = 20 → n ≤ m :=
by { sorry }

end smallest_four_digit_multiple_of_8_with_digit_sum_20_l288_288679


namespace smallest_number_l288_288492

theorem smallest_number (A B C : ℕ) 
  (h1 : A / 3 = B / 5) 
  (h2 : B / 5 = C / 7) 
  (h3 : C = 56) 
  (h4 : C - A = 32) : 
  A = 24 := 
sorry

end smallest_number_l288_288492


namespace tan_pi_add_theta_l288_288913

theorem tan_pi_add_theta (θ : ℝ) (h : Real.tan (Real.pi + θ) = 2) : 
  (2 * Real.sin θ - Real.cos θ) / (Real.sin θ + 2 * Real.cos θ) = 3 / 4 :=
by
  sorry

end tan_pi_add_theta_l288_288913


namespace set_equality_example_l288_288426

theorem set_equality_example : {x : ℕ | 2 * x + 3 ≥ 3 * x} = {0, 1, 2, 3} := by
  sorry

end set_equality_example_l288_288426


namespace p_necessary_not_sufficient_for_p_and_q_l288_288061

-- Define statements p and q as propositions
variables (p q : Prop)

-- Prove that "p is true" is a necessary but not sufficient condition for "p ∧ q is true"
theorem p_necessary_not_sufficient_for_p_and_q : (p ∧ q → p) ∧ (p → ¬ (p ∧ q)) :=
by sorry

end p_necessary_not_sufficient_for_p_and_q_l288_288061


namespace stream_speed_fraction_l288_288697

theorem stream_speed_fraction (B S : ℝ) (h1 : B = 3 * S) 
  (h2 : (1 / (B - S)) = 2 * (1 / (B + S))) : (S / B) = 1 / 3 :=
sorry

end stream_speed_fraction_l288_288697


namespace total_vegetables_l288_288094

-- Define the initial conditions
def potatoes : Nat := 560
def cucumbers : Nat := potatoes - 132
def tomatoes : Nat := 3 * cucumbers
def peppers : Nat := tomatoes / 2
def carrots : Nat := cucumbers + tomatoes

-- State the theorem to prove the total number of vegetables
theorem total_vegetables :
  560 + (560 - 132) + (3 * (560 - 132)) + ((3 * (560 - 132)) / 2) + ((560 - 132) + (3 * (560 - 132))) = 4626 := by
  sorry

end total_vegetables_l288_288094


namespace sum_primes_between_20_and_40_l288_288817

theorem sum_primes_between_20_and_40 :
  23 + 29 + 31 + 37 = 120 :=
by
  sorry

end sum_primes_between_20_and_40_l288_288817


namespace jason_books_l288_288627

theorem jason_books (books_per_shelf : ℕ) (num_shelves : ℕ) (total_books : ℕ) :
  books_per_shelf = 45 → num_shelves = 7 → total_books = books_per_shelf * num_shelves → total_books = 315 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end jason_books_l288_288627


namespace p_6_eq_163_l288_288937

noncomputable def p (x : ℕ) : ℕ :=
  (x - 1) * (x - 2) * (x - 3) * (x - 4) * (x - 5) + x^2 + x + 1

theorem p_6_eq_163 : p 6 = 163 :=
by
  sorry

end p_6_eq_163_l288_288937


namespace five_digit_sine_rule_count_l288_288222

theorem five_digit_sine_rule_count :
    ∃ (count : ℕ), 
        (∀ (a b c d e : ℕ), 
          (a <  b) ∧
          (b >  c) ∧
          (c >  d) ∧
          (d <  e) ∧
          (a >  d) ∧
          (b >  e) ∧
          (∃ (num : ℕ), num = 10000 * a + 1000 * b + 100 * c + 10 * d + e))
        →
        count = 2892 :=
sorry

end five_digit_sine_rule_count_l288_288222


namespace smallest_number_of_three_l288_288071

theorem smallest_number_of_three (a b c : ℕ) (h1 : a + b + c = 78) (h2 : b = 27) (h3 : c = b + 5) :
  a = 19 :=
by
  sorry

end smallest_number_of_three_l288_288071


namespace no_solution_ineq_l288_288459

theorem no_solution_ineq (m : ℝ) :
  (¬ ∃ (x : ℝ), x - 1 > 1 ∧ x < m) → m ≤ 2 :=
by
  sorry

end no_solution_ineq_l288_288459


namespace neg_p_true_l288_288008

theorem neg_p_true :
  (∃ x : ℝ, x^2 ≤ 0) :=
sorry

end neg_p_true_l288_288008


namespace find_t_l288_288931

-- Given a quadratic equation
def quadratic_eq (x : ℝ) := 4 * x ^ 2 - 16 * x - 200

-- Completing the square to find t
theorem find_t : ∃ q t : ℝ, (x : ℝ) → (quadratic_eq x = 0) → (x + q) ^ 2 = t ∧ t = 54 :=
by
  sorry

end find_t_l288_288931


namespace intervals_of_monotonicity_m_in_terms_of_x0_at_least_two_tangents_l288_288268

noncomputable def h (a x : ℝ) : ℝ := a * x^3 - 1
noncomputable def g (x : ℝ) : ℝ := Real.log x

noncomputable def f (a x : ℝ) : ℝ := h a x + 3 * x * g x
noncomputable def F (a x : ℝ) : ℝ := (a - (1/3)) * x^3 + (1/2) * x^2 * g a - h a x - 1

theorem intervals_of_monotonicity (a : ℝ) (ha : f a 1 = -1) :
  ((a = 0) → (∀ x : ℝ, (0 < x ∧ x < Real.exp (-1) → f 0 x < f 0 x + 3 * x * g x)) ∧
    (Real.exp (-1) < x ∧ 0 < x → f 0 x + 3 * x * g x > f 0 x)) := sorry

theorem m_in_terms_of_x0 (a x0 m : ℝ) (ha : a > Real.exp (10 / 3))
  (tangent_line : ∀ y, y - ( -(1 / 3) * x0^3 + (1 / 2) * x0^2 * g a) = 
    (-(x0^2) + x0 * g a) * (x - x0)) :
  m = (2 / 3) * x0^3 - (1 + (1 / 2) * g a) * x0^2 + x0 * g a := sorry

theorem at_least_two_tangents (a m : ℝ) (ha : a > Real.exp (10 / 3))
  (at_least_two : ∃ x0 y, x0 ≠ y ∧ F a x0 = m ∧ F a y = m) :
  m = 4 / 3 := sorry

end intervals_of_monotonicity_m_in_terms_of_x0_at_least_two_tangents_l288_288268


namespace find_solutions_l288_288427

-- Definitions
def is_solution (x y z n : ℕ) : Prop :=
  x^3 + y^3 + z^3 = n * (x^2) * (y^2) * (z^2)

-- Theorem statement
theorem find_solutions :
  {sol : ℕ × ℕ × ℕ × ℕ | is_solution sol.1 sol.2.1 sol.2.2.1 sol.2.2.2} =
  {(1, 1, 1, 3), (1, 2, 3, 1), (2, 1, 3, 1)} :=
by sorry

end find_solutions_l288_288427


namespace ashton_sheets_l288_288072
-- Import the entire Mathlib to bring in the necessary library

-- Defining the conditions and proving the statement
theorem ashton_sheets (t j a : ℕ) (h1 : t = j + 10) (h2 : j = 32) (h3 : j + a = t + 30) : a = 40 := by
  -- Sorry placeholder for the proof
  sorry

end ashton_sheets_l288_288072


namespace how_many_children_l288_288517

-- Definitions based on conditions
def total_spectators : ℕ := 10000
def men : ℕ := 7000
def others : ℕ := total_spectators - men -- women + children
def children_per_woman : ℕ := 5

-- Variables
variable (W C : ℕ)

-- Conditions as Lean equalities
def condition1 : W + C = others := by sorry
def condition2 : C = children_per_woman * W := by sorry

-- Theorem statement to prove the number of children
theorem how_many_children (h1 : W + C = others) (h2 : C = children_per_woman * W) : C = 2500 :=
by sorry

end how_many_children_l288_288517


namespace value_of_k_l288_288504

-- Define the conditions of the quartic equation and the product of two roots
variable (a b c d k : ℝ)
variable (hx : (Polynomial.X ^ 4 - 18 * Polynomial.X ^ 3 + k * Polynomial.X ^ 2 + 200 * Polynomial.X - 1984).rootSet ℝ = {a, b, c, d})
variable (hprod_ab : a * b = -32)

-- The statement to prove: the value of k is 86
theorem value_of_k :
  k = 86 :=
by sorry

end value_of_k_l288_288504


namespace malachi_additional_photos_l288_288467

-- Definition of the conditions
def total_photos : ℕ := 2430
def ratio_last_year : ℕ := 10
def ratio_this_year : ℕ := 17
def total_ratio_units : ℕ := ratio_last_year + ratio_this_year
def diff_ratio_units : ℕ := ratio_this_year - ratio_last_year
def photos_per_unit : ℕ := total_photos / total_ratio_units
def additional_photos : ℕ := diff_ratio_units * photos_per_unit

-- The theorem proving how many more photos Malachi took this year than last year
theorem malachi_additional_photos : additional_photos = 630 := by
  sorry

end malachi_additional_photos_l288_288467


namespace evaluate_polynomial_103_l288_288998

theorem evaluate_polynomial_103 :
  103 ^ 4 - 4 * 103 ^ 3 + 6 * 103 ^ 2 - 4 * 103 + 1 = 108243216 :=
by
  sorry

end evaluate_polynomial_103_l288_288998


namespace min_value_expression_l288_288590

theorem min_value_expression : ∃ x : ℝ, x = 300 ∧ ∀ y : ℝ, (y^2 - 600*y + 369) ≥ (300^2 - 600*300 + 369) := by
  use 300
  sorry

end min_value_expression_l288_288590


namespace even_not_divisible_by_4_not_sum_of_two_consecutive_odds_l288_288043

theorem even_not_divisible_by_4_not_sum_of_two_consecutive_odds (x n : ℕ) (h₁ : Even x) (h₂ : ¬ ∃ k, x = 4 * k) : x ≠ (2 * n + 1) + (2 * n + 3) := by
  sorry

end even_not_divisible_by_4_not_sum_of_two_consecutive_odds_l288_288043


namespace numerical_value_expression_l288_288738

theorem numerical_value_expression (x y z : ℚ) (h1 : x - 4 * y - 2 * z = 0) (h2 : 3 * x + 2 * y - z = 0) (h3 : z ≠ 0) : 
  (x^2 - 5 * x * y) / (2 * y^2 + z^2) = 164 / 147 :=
by sorry

end numerical_value_expression_l288_288738


namespace calc_op_l288_288058

def op (a b : ℕ) := (a + b) * (a - b)

theorem calc_op : (op 5 2)^2 = 441 := 
by 
  sorry

end calc_op_l288_288058


namespace cookie_distribution_l288_288826

theorem cookie_distribution : 
  ∀ (n c T : ℕ), n = 6 → c = 4 → T = n * c → T = 24 :=
by 
  intros n c T h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end cookie_distribution_l288_288826


namespace trivia_team_students_l288_288198

def total_students (not_picked groups students_per_group: ℕ) :=
  not_picked + groups * students_per_group

theorem trivia_team_students (not_picked groups students_per_group: ℕ) (h_not_picked: not_picked = 10) (h_groups: groups = 8) (h_students_per_group: students_per_group = 6) :
  total_students not_picked groups students_per_group = 58 :=
by
  sorry

end trivia_team_students_l288_288198


namespace line_intersects_parabola_at_one_point_l288_288968

theorem line_intersects_parabola_at_one_point (k : ℝ) :
    (∃ y : ℝ, x = -3 * y^2 - 4 * y + 7) ↔ (x = k) := by
  sorry

end line_intersects_parabola_at_one_point_l288_288968


namespace solve_xyz_l288_288355

theorem solve_xyz (x y z : ℕ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_pos_z : 0 < z) :
  (x / 21) * (y / 189) + z = 1 ↔ x = 21 ∧ y = 567 ∧ z = 0 :=
sorry

end solve_xyz_l288_288355


namespace exists_multiple_of_n_with_ones_l288_288304

theorem exists_multiple_of_n_with_ones (n : ℤ) (hn1 : n ≥ 1) (hn2 : Int.gcd n 10 = 1) :
  ∃ k : ℕ, n ∣ (10^k - 1) / 9 :=
by sorry

end exists_multiple_of_n_with_ones_l288_288304


namespace smallest_x_solution_l288_288587

theorem smallest_x_solution :
  (∃ x : ℚ, abs (4 * x + 3) = 30 ∧ ∀ y : ℚ, abs (4 * y + 3) = 30 → x ≤ y) ↔ x = -33 / 4 := by
  sorry

end smallest_x_solution_l288_288587


namespace oranges_apples_ratio_l288_288228

variable (A O P : ℕ)
variable (n : ℚ)
variable (h1 : O = n * A)
variable (h2 : P = 4 * O)
variable (h3 : A = (0.08333333333333333 : ℚ) * P)

theorem oranges_apples_ratio (A O P : ℕ) (n : ℚ) 
  (h1 : O = n * A) (h2 : P = 4 * O) (h3 : A = (0.08333333333333333 : ℚ) * P) : n = 3 := 
by
  sorry

end oranges_apples_ratio_l288_288228


namespace conclusion_l288_288921

def balls : Finset ℕ := (Finset.range 6).map (Function.Embedding.coeFn (Finset.range 1)) ∪ (Finset.range 7 10).map (Function.Embedding.coeFn (Finset.range 1))

def n_black : ℕ := 6
def n_white : ℕ := 4
def n_total : ℕ := n_black + n_white
def num_selected : ℕ := 4

/-
The main conditions are:
1. There are 6 black balls and 4 white balls, making 10 balls total.
2. 4 balls are randomly selected.
-/

theorem conclusion (X : ℕ) (Y : ℕ) (P : ℕ → ℚ) : 
  (¬ (X = hypergeometric_dist)) ∧ 
  (Y = hypergeometric_dist) ∧ 
  (P 2 ≠ 1 / 14) ∧ 
  (prob_score_high_score 2 1 2 = 1 / 14) :=
sorry

end conclusion_l288_288921


namespace uranus_appears_7_minutes_after_6AM_l288_288801

def mars_last_seen := 0 * 60 + 10 -- 12:10 AM in minutes after midnight
def jupiter_after_mars := 2 * 60 + 41 -- 2 hours and 41 minutes in minutes
def uranus_after_jupiter := 3 * 60 + 16 -- 3 hours and 16 minutes in minutes
def uranus_appearance := mars_last_seen + jupiter_after_mars + uranus_after_jupiter

theorem uranus_appears_7_minutes_after_6AM : uranus_appearance - (6 * 60) = 7 := by
  sorry

end uranus_appears_7_minutes_after_6AM_l288_288801


namespace binom_divisible_by_prime_l288_288781

theorem binom_divisible_by_prime {p k : ℕ} (hp : Nat.Prime p) (hk1 : 1 ≤ k) (hk2 : k ≤ p - 1) : Nat.choose p k % p = 0 := 
  sorry

end binom_divisible_by_prime_l288_288781


namespace value_of_k_for_binomial_square_l288_288204

theorem value_of_k_for_binomial_square (k : ℝ) : (∃ (b : ℝ), x^2 - 18 * x + k = (x + b)^2) → k = 81 :=
by
  intro h
  cases h with b hb
  -- We will use these to directly infer things without needing the proof here
  sorry

end value_of_k_for_binomial_square_l288_288204


namespace proof_problem_l288_288266

-- Define the sum of the first n terms of the arithmetic sequence
def S (n : ℕ) : ℕ := (n^2 + n) / 2

-- Define the arithmetic sequence a_n based on S_n
def a (n : ℕ) : ℕ := if n = 1 then 1 else S n - S (n - 1)

-- Define the geometric sequence b_n with initial conditions
def b (n : ℕ) : ℕ :=
  if n = 1 then a 1 + 1
  else if n = 2 then a 2 + 2
  else 2^n

-- Define the sum of the first n terms of the geometric sequence b_n
def T (n : ℕ) : ℕ := 2 * (2^n - 1)

-- Main theorem to prove
theorem proof_problem :
  (∀ n, a n = n) ∧
  (∀ n, n ≥ 1 → b n = 2^n) ∧
  (∃ n, T n + a n > 300 ∧ ∀ m < n, T m + a m ≤ 300) :=
by {
  sorry
}

end proof_problem_l288_288266


namespace exists_square_all_invisible_l288_288410

open Nat

theorem exists_square_all_invisible (n : ℕ) : 
  ∃ a b : ℤ, ∀ i j : ℕ, i < n → j < n → gcd (a + i) (b + j) > 1 := 
sorry

end exists_square_all_invisible_l288_288410


namespace harrys_mothers_age_l288_288272

theorem harrys_mothers_age 
  (h : ℕ)  -- Harry's age
  (f : ℕ)  -- Father's age
  (m : ℕ)  -- Mother's age
  (h_age : h = 50)
  (f_age : f = h + 24)
  (m_age : m = f - h / 25) 
  : (m - h = 22) := 
by
  sorry

end harrys_mothers_age_l288_288272


namespace fraction_power_computation_l288_288676

theorem fraction_power_computation : (5 / 6) ^ 4 = 625 / 1296 :=
by
  -- Normally we'd provide the proof here, but it's omitted as per instructions
  sorry

end fraction_power_computation_l288_288676


namespace parallel_vectors_x_value_l288_288260

/-
Given that \(\overrightarrow{a} = (1,2)\) and \(\overrightarrow{b} = (2x, -3)\) are parallel vectors, prove that \(x = -\frac{3}{4}\).
-/
theorem parallel_vectors_x_value (x : ℝ) 
  (a : ℝ × ℝ := (1, 2)) 
  (b : ℝ × ℝ := (2 * x, -3)) 
  (h_parallel : (a.1 * b.2) - (a.2 * b.1) = 0) : 
  x = -3 / 4 := by
  sorry

end parallel_vectors_x_value_l288_288260


namespace root_of_quadratic_l288_288099

theorem root_of_quadratic :
  (∀ x : ℝ, 2 * x^2 + 3 * x - 65 = 0 → x = 5 ∨ x = -6.5) :=
sorry

end root_of_quadratic_l288_288099


namespace min_value_expression_l288_288634

theorem min_value_expression (a b c d : ℝ) 
  (h1 : 2 ≤ a) (h2 : a ≤ b) (h3 : b ≤ c) (h4 : c ≤ d) (h5 : d ≤ 5) :
  (a - 2)^2 + (b / a - 1)^2 + (c / b - 1)^2 + (d / c - 1)^2 + (5 / d - 1)^2 
  = 5^(5/4) - 10 * Real.sqrt (5^(1/4)) + 5 := 
sorry

end min_value_expression_l288_288634


namespace mowing_time_l288_288778

theorem mowing_time (length width: ℝ) (swath_width_overlap_rate: ℝ)
                    (walking_speed: ℝ) (ft_per_inch: ℝ)
                    (length_eq: length = 100)
                    (width_eq: width = 120)
                    (swath_eq: swath_width_overlap_rate = 24)
                    (walking_eq: walking_speed = 4500)
                    (conversion_eq: ft_per_inch = 1/12) :
                    (length / walking_speed) * (width / (swath_width_overlap_rate * ft_per_inch)) = 1.33 :=
by
    rw [length_eq, width_eq, swath_eq, walking_eq, conversion_eq]
    exact sorry

end mowing_time_l288_288778


namespace range_of_m_l288_288461

theorem range_of_m (m : ℝ) :
  (¬ ∃ x : ℝ, 0 < x ∧ mx^2 + 2 * x + m > 0) →
  m ≤ -1 := by
  sorry

end range_of_m_l288_288461


namespace cake_remaining_portion_l288_288845

theorem cake_remaining_portion (initial_cake : ℝ) (alex_share_percentage : ℝ) (jordan_share_fraction : ℝ) :
  initial_cake = 1 ∧ alex_share_percentage = 0.4 ∧ jordan_share_fraction = 0.5 →
  (initial_cake - alex_share_percentage * initial_cake) * (1 - jordan_share_fraction) = 0.3 :=
by
  sorry

end cake_remaining_portion_l288_288845


namespace mondays_in_first_70_days_l288_288678

theorem mondays_in_first_70_days (days : ℕ) (h1 : days = 70) (mondays_per_week : ℕ) (h2 : mondays_per_week = 1) : 
  ∃ (mondays : ℕ), mondays = 10 := 
by
  sorry

end mondays_in_first_70_days_l288_288678


namespace repeating_decimal_to_fraction_l288_288718

theorem repeating_decimal_to_fraction : (6 + 81 / 99) = 75 / 11 := 
by 
  sorry

end repeating_decimal_to_fraction_l288_288718


namespace smallest_positive_perfect_square_div_by_2_3_5_l288_288374

-- Definition capturing the conditions of the problem
def is_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

def divisible_by (n m : ℕ) : Prop :=
  m % n = 0

-- The theorem statement combining all conditions and requiring the proof of the correct answer
theorem smallest_positive_perfect_square_div_by_2_3_5 : 
  ∃ n : ℕ, 0 < n ∧ is_perfect_square n ∧ divisible_by 2 n ∧ divisible_by 3 n ∧ divisible_by 5 n ∧ n = 900 :=
sorry

end smallest_positive_perfect_square_div_by_2_3_5_l288_288374


namespace ratio_female_to_male_l288_288653

theorem ratio_female_to_male
  (a b c : ℕ)
  (ha : a = 60)
  (hb : b = 80)
  (hc : c = 65) :
  f / m = 1 / 3 := 
by
  sorry

end ratio_female_to_male_l288_288653


namespace evaluate_expression_l288_288247

theorem evaluate_expression : 2 + 1 / (2 + 1 / (2 + 1 / 3)) = 41 / 17 := by
  sorry

end evaluate_expression_l288_288247


namespace divisible_by_55_l288_288128

theorem divisible_by_55 (n : ℤ) : 
  (55 ∣ (n^2 + 3 * n + 1)) ↔ (n % 55 = 46 ∨ n % 55 = 6) := 
by 
  sorry

end divisible_by_55_l288_288128


namespace smaller_angle_3_15_l288_288076

theorem smaller_angle_3_15 :
  let minute_hand_degrees := 15 * (360 / 60)
  let hour_hand_degrees := 90 + (30 / 60) * 15
  abs (hour_hand_degrees - minute_hand_degrees) = 7.5 :=
by
  let minute_hand_degrees := 15 * (360 / 60)
  let hour_hand_degrees := 90 + (30 / 60) * 15
  have h1 : minute_hand_degrees = 90 := by sorry
  have h2 : hour_hand_degrees = 97.5 := by sorry
  have h3 : abs (hour_hand_degrees - minute_hand_degrees) = 7.5 := by sorry
  exact h3

end smaller_angle_3_15_l288_288076


namespace greatest_four_digit_multiple_of_17_l288_288988

theorem greatest_four_digit_multiple_of_17 :
  ∃ n, 1000 ≤ n ∧ n ≤ 9999 ∧ n % 17 = 0 ∧ ∀ m, (1000 ≤ m ∧ m ≤ 9999 ∧ m % 17 = 0) → m ≤ n :=
  ⟨9996, by {
        split,
        { linarith },
        { split,
            { linarith },
            { split,
                { exact ModEq.rfl },
                { intros m hm hle,
                  have h : m ≤ 9999 := hm.2.2,
                  have : m = 17 * (m / 17) := (Nat.div_mul_cancel hm.2.1).symm,
                  have : 17 * (m / 17) ≤ 17 * 588 := Nat.mul_le_mul_left 17 (Nat.div_le_of_le_mul (by linarith)),
                  linarith,
                },
            },
        },
    },
  ⟩ sorry

end greatest_four_digit_multiple_of_17_l288_288988


namespace carrie_harvests_9000_l288_288571

noncomputable def garden_area (length width : ℕ) := length * width
noncomputable def total_plants (plants_per_sqft sqft : ℕ) := plants_per_sqft * sqft
noncomputable def total_cucumbers (yield_plants plants : ℕ) := yield_plants * plants

theorem carrie_harvests_9000 :
  garden_area 10 12 = 120 →
  total_plants 5 120 = 600 →
  total_cucumbers 15 600 = 9000 :=
by sorry

end carrie_harvests_9000_l288_288571


namespace distance_against_stream_l288_288296

variable (vs : ℝ) -- speed of the stream

-- condition: in one hour, the boat goes 9 km along the stream
def cond1 (vs : ℝ) := 7 + vs = 9

-- condition: the speed of the boat in still water (7 km/hr)
def speed_still_water := 7

-- theorem to prove: the distance the boat goes against the stream in one hour
theorem distance_against_stream (vs : ℝ) (h : cond1 vs) : 
  (speed_still_water - vs) * 1 = 5 :=
by
  rw [speed_still_water, mul_one]
  sorry

end distance_against_stream_l288_288296


namespace smallest_perfect_square_divisible_by_2_3_5_l288_288370

theorem smallest_perfect_square_divisible_by_2_3_5 :
  ∃ n : ℕ, (n > 0) ∧ (nat.sqrt n * nat.sqrt n = n) ∧ (2 ∣ n) ∧ (3 ∣ n) ∧ (5 ∣ n) ∧ n = 900 := sorry

end smallest_perfect_square_divisible_by_2_3_5_l288_288370


namespace factorize_xy2_minus_x_l288_288870

theorem factorize_xy2_minus_x (x y : ℝ) : xy^2 - x = x * (y - 1) * (y + 1) :=
by
  sorry

end factorize_xy2_minus_x_l288_288870


namespace compare_values_l288_288905

noncomputable def f (x : ℝ) : ℝ := Real.exp (-(x - 1)^2)

def a : ℝ := f (Real.sqrt 2 / 2)
def b : ℝ := f (Real.sqrt 3 / 2)
def c : ℝ := f (Real.sqrt 6 / 2)

theorem compare_values : b > c ∧ c > a := by
  sorry

end compare_values_l288_288905


namespace Elise_savings_l288_288246

theorem Elise_savings :
  let initial_dollars := 8
  let saved_euros := 11
  let euro_to_dollar := 1.18
  let comic_cost := 2
  let puzzle_pounds := 13
  let pound_to_dollar := 1.38
  let euros_to_dollars := saved_euros * euro_to_dollar
  let total_after_saving := initial_dollars + euros_to_dollars
  let after_comic := total_after_saving - comic_cost
  let pounds_to_dollars := puzzle_pounds * pound_to_dollar
  let final_amount := after_comic - pounds_to_dollars
  final_amount = 1.04 :=
by
  sorry

end Elise_savings_l288_288246


namespace rearrange_numbers_diff_3_or_5_l288_288168

noncomputable def is_valid_permutation (n : ℕ) (σ : list ℕ) : Prop :=
  (σ.nodup ∧ ∀ i < σ.length - 1, |σ.nth_le i sorry - σ.nth_le (i + 1) sorry| = 3 ∨ |σ.nth_le i sorry - σ.nth_le (i + 1) sorry| = 5)

theorem rearrange_numbers_diff_3_or_5 (n : ℕ) :
  (n = 25 ∨ n = 1000) → ∃ σ : list ℕ, (σ = (list.range n).map (+1)) ∧ is_valid_permutation n σ :=
by
  sorry

end rearrange_numbers_diff_3_or_5_l288_288168


namespace greatest_four_digit_multiple_of_17_l288_288992

theorem greatest_four_digit_multiple_of_17 : ∃ (n : ℕ), 1000 ≤ n ∧ n < 10000 ∧ 17 ∣ n ∧ n = 9996 :=
by
  use 9996
  split
  { exact by norm_num }
  split
  { exact by norm_num [lt.sub (ofNat 10000) (ofNat 4)] }
  split
  { exact by norm_num }
  { exact by norm_num [dvd_of_mod_eq_zero (by norm_num : 10000 % 17 = 4)] }

end greatest_four_digit_multiple_of_17_l288_288992


namespace correct_action_order_l288_288983

inductive Action
| tap : Action
| pay_online : Action
| swipe : Action
| insert_into_terminal : Action
deriving DecidableEq, Repr, Inhabited

inductive Technology
| chip : Technology
| magnetic_stripe : Technology
| paypass : Technology
| cvc : Technology
deriving DecidableEq, Repr, Inhabited

def action_for_technology : Technology → Action
| Technology.chip := Action.insert_into_terminal
| Technology.magnetic_stripe := Action.swipe
| Technology.paypass := Action.tap
| Technology.cvc := Action.pay_online

theorem correct_action_order :
  [action_for_technology Technology.chip, action_for_technology Technology.magnetic_stripe,
   action_for_technology Technology.paypass, action_for_technology Technology.cvc] = 
  [Action.insert_into_terminal, Action.swipe, Action.tap, Action.pay_online] := 
sorry

end correct_action_order_l288_288983


namespace polynomial_value_l288_288999

theorem polynomial_value : 103^4 - 4 * 103^3 + 6 * 103^2 - 4 * 103 + 1 = 108243216 :=
by
  sorry

end polynomial_value_l288_288999


namespace find_three_tuple_solutions_l288_288889

open Real

theorem find_three_tuple_solutions :
  (x y z : ℝ) → (x^2 + y^2 + 25 * z^2 = 6 * x * z + 8 * y * z)
  → (3 * x^2 + 2 * y^2 + z^2 = 240)
  → (x = 6 ∧ y = 8 ∧ z = 2) ∨ (x = -6 ∧ y = -8 ∧ z = -2) :=
by
  intro x y z
  intro h1 h2
  sorry

end find_three_tuple_solutions_l288_288889


namespace sum_of_values_l288_288943

namespace ProofProblem

noncomputable def f (x : ℝ) : ℝ :=
  if x < 1 then 5 * x - 3 else x^2 - 4 * x + 3

theorem sum_of_values (s : Finset ℝ) : 
  (∀ x ∈ s, f x = 2) → s.sum id = 4 :=
by 
  sorry

end ProofProblem

end sum_of_values_l288_288943


namespace nine_point_circle_equation_l288_288429

theorem nine_point_circle_equation 
  (α β γ : ℝ) 
  (x y z : ℝ) :
  (x^2 * (Real.sin α) * (Real.cos α) + y^2 * (Real.sin β) * (Real.cos β) + z^2 * (Real.sin γ) * (Real.cos γ) = 
  y * z * (Real.sin α) + x * z * (Real.sin β) + x * y * (Real.sin γ))
:= sorry

end nine_point_circle_equation_l288_288429


namespace cakes_served_during_lunch_l288_288226

theorem cakes_served_during_lunch (T D L : ℕ) (h1 : T = 15) (h2 : D = 9) : L = T - D → L = 6 :=
by
  intros h
  rw [h1, h2] at h
  exact h

end cakes_served_during_lunch_l288_288226


namespace smallest_non_factor_product_of_factors_of_60_l288_288805

theorem smallest_non_factor_product_of_factors_of_60 :
  ∃ x y : ℕ, x ≠ y ∧ x ∣ 60 ∧ y ∣ 60 ∧ ¬ (x * y ∣ 60) ∧ ∀ x' y' : ℕ, x' ≠ y' → x' ∣ 60 → y' ∣ 60 → ¬(x' * y' ∣ 60) → x * y ≤ x' * y' := 
sorry

end smallest_non_factor_product_of_factors_of_60_l288_288805


namespace percentage_apples_is_50_percent_l288_288067

-- Definitions for the given conditions
def initial_apples : ℕ := 10
def initial_oranges : ℕ := 5
def added_oranges : ℕ := 5

-- Defining the proof problem
theorem percentage_apples_is_50_percent :
  let total_fruits := initial_apples + initial_oranges + added_oranges in
  let apples_percentage := (initial_apples * 100) / total_fruits in
  apples_percentage = 50 :=
by
  sorry

end percentage_apples_is_50_percent_l288_288067


namespace incorrect_conclusion_l288_288494

-- Define the linear regression model
def model (x : ℝ) : ℝ := 0.85 * x - 85.71

-- Define the conditions
axiom linear_correlation : ∀ (x y : ℝ), ∃ (x_i y_i : ℝ) (i : ℕ), model x = y

-- The theorem to prove the statement for x = 170 is false
theorem incorrect_conclusion (x : ℝ) (h : x = 170) : ¬ (model x = 58.79) :=
  by sorry

end incorrect_conclusion_l288_288494


namespace factorize_xy2_minus_x_l288_288868

theorem factorize_xy2_minus_x (x y : ℝ) : xy^2 - x = x * (y - 1) * (y + 1) :=
by
  sorry

end factorize_xy2_minus_x_l288_288868


namespace negative_integer_solution_l288_288794

theorem negative_integer_solution (M : ℤ) (h1 : 2 * M^2 + M = 12) (h2 : M < 0) : M = -4 :=
sorry

end negative_integer_solution_l288_288794


namespace boxes_of_nerds_l288_288106

def totalCandies (kitKatBars hersheyKisses lollipops babyRuths reeseCups nerds : Nat) : Nat := 
  kitKatBars + hersheyKisses + lollipops + babyRuths + reeseCups + nerds

def adjustForGivenLollipops (total lollipopsGiven : Nat) : Nat :=
  total - lollipopsGiven

theorem boxes_of_nerds :
  ∀ (kitKatBars hersheyKisses lollipops babyRuths reeseCups lollipopsGiven totalAfterGiving nerds : Nat),
  kitKatBars = 5 →
  hersheyKisses = 3 * kitKatBars →
  lollipops = 11 →
  babyRuths = 10 →
  reeseCups = babyRuths / 2 →
  lollipopsGiven = 5 →
  totalAfterGiving = 49 →
  totalCandies kitKatBars hersheyKisses lollipops babyRuths reeseCups 0 - lollipopsGiven + nerds = totalAfterGiving →
  nerds = 8 :=
by
  intros
  sorry

end boxes_of_nerds_l288_288106


namespace probability_top_card_is_king_correct_l288_288412

noncomputable def probability_top_card_is_king (total_cards kings : ℕ) : ℚ :=
kings / total_cards

theorem probability_top_card_is_king_correct :
  probability_top_card_is_king 52 4 = 1 / 13 :=
by
  sorry

end probability_top_card_is_king_correct_l288_288412


namespace factorize_xy_squared_minus_x_l288_288881

theorem factorize_xy_squared_minus_x (x y : ℝ) : xy^2 - x = x * (y - 1) * (y + 1) :=
  sorry

end factorize_xy_squared_minus_x_l288_288881


namespace total_intersections_l288_288777

def north_south_streets : ℕ := 10
def east_west_streets : ℕ := 10

theorem total_intersections :
  (north_south_streets * east_west_streets = 100) :=
by
  sorry

end total_intersections_l288_288777


namespace probability_to_pass_the_test_l288_288753

noncomputable def probability_passes_test : ℝ :=
  (finset.card (@set.to_finset {s : finset (fin 3) // s.card = 2})) * 
    (0.6 ^ 2) * (0.4) + 
  (finset.card (@set.to_finset {s : finset (fin 3) // s.card = 3})) * (0.6 ^ 3)

theorem probability_to_pass_the_test : 
  probability_passes_test = 0.648 :=
by 
  sorry

end probability_to_pass_the_test_l288_288753


namespace even_function_f_l288_288191

noncomputable def f (x : ℝ) : ℝ := if 0 < x ∧ x < 10 then Real.log x else 0

theorem even_function_f (x : ℝ) (h : f (-x) = f x) (h1 : ∀ x, 0 < x ∧ x < 10 → f x = Real.log x) :
  f (-Real.exp 1) + f (Real.exp 2) = 3 := by
  sorry

end even_function_f_l288_288191


namespace alex_has_more_pens_than_jane_l288_288844

-- Definitions based on the conditions
def starting_pens_alex : ℕ := 4
def pens_jane_after_month : ℕ := 16

-- Alex's pen count after each week
def pens_alex_after_week (w : ℕ) : ℕ :=
  starting_pens_alex * 2 ^ w

-- Proof statement
theorem alex_has_more_pens_than_jane :
  pens_alex_after_week 4 - pens_jane_after_month = 16 := by
  sorry

end alex_has_more_pens_than_jane_l288_288844


namespace crackers_per_friend_l288_288038

theorem crackers_per_friend (total_crackers : ℕ) (num_friends : ℕ) (n : ℕ) 
  (h1 : total_crackers = 8) 
  (h2 : num_friends = 4)
  (h3 : total_crackers / num_friends = n) : n = 2 :=
by
  sorry

end crackers_per_friend_l288_288038


namespace algebraic_expression_value_l288_288207

theorem algebraic_expression_value (p q : ℤ) 
  (h : 8 * p + 2 * q = -2023) : 
  (p * (-2) ^ 3 + q * (-2) + 1 = 2024) :=
by
  sorry

end algebraic_expression_value_l288_288207


namespace spent_on_music_l288_288736

variable (total_allowance : ℝ) (fraction_music : ℝ)

-- Assuming the conditions
def conditions : Prop :=
  total_allowance = 50 ∧ fraction_music = 3 / 10

-- The proof problem
theorem spent_on_music (h : conditions total_allowance fraction_music) : 
  total_allowance * fraction_music = 15 := by
  cases h with
  | intro h_total h_fraction =>
  sorry

end spent_on_music_l288_288736


namespace find_a_if_f_is_even_l288_288286

-- Defining f as given in the problem conditions
noncomputable def f (x a : ℝ) : ℝ := (x + a) * 3 ^ (x - 2 + a ^ 2) - (x - a) * 3 ^ (8 - x - 3 * a)

-- Statement of the proof problem with the conditions
theorem find_a_if_f_is_even (a : ℝ) :
  (∀ x : ℝ, f x a = f (-x) a) → (a = -5 ∨ a = 2) :=
by
  sorry

end find_a_if_f_is_even_l288_288286


namespace greatest_four_digit_multiple_of_17_l288_288991

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999

def is_multiple_of (n d : ℕ) : Prop :=
  ∃ k : ℕ, n = k * d

theorem greatest_four_digit_multiple_of_17 : ∃ n, is_four_digit n ∧ is_multiple_of n 17 ∧
  ∀ m, is_four_digit m → is_multiple_of m 17 → m ≤ n :=
  by
  existsi 9996
  sorry

end greatest_four_digit_multiple_of_17_l288_288991


namespace unique_solution_l288_288719

theorem unique_solution (k n : ℕ) (hk : k > 0) (hn : n > 0) (h : (7^k - 3^n) ∣ (k^4 + n^2)) : (k = 2 ∧ n = 4) :=
by
  sorry

end unique_solution_l288_288719


namespace prob_divisible_by_3_of_three_digits_l288_288065

-- Define the set of digits available
def digits : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9}

-- Three digits are to be chosen from this set
def choose_three_digits (s : Finset ℕ) : Finset (Finset ℕ) := s.powerset.filter (λ x, x.card = 3)

-- Define the property of the sum of digits being divisible by 3
def divisible_by_3 (s : Finset ℕ) : Prop := s.sum id % 3 = 0

-- Total combinations of choosing 3 out of 9 digits
def total_combinations : ℕ := (digits.card.choose 3)

-- Valid combinations where sum of digits is divisible by 3
def valid_combinations : Finset (Finset ℕ) := (choose_three_digits digits).filter divisible_by_3

-- Finally, the probability of a three-digit number being divisible by 3
def probability : ℕ × ℕ := (valid_combinations.card, total_combinations)

theorem prob_divisible_by_3_of_three_digits :
  probability = (5, 14) :=
by
  -- Proof to be filled
  sorry

end prob_divisible_by_3_of_three_digits_l288_288065


namespace factorize_xy_squared_minus_x_l288_288878

theorem factorize_xy_squared_minus_x (x y : ℝ) : xy^2 - x = x * (y - 1) * (y + 1) :=
  sorry

end factorize_xy_squared_minus_x_l288_288878


namespace find_n_l288_288690

theorem find_n
  (n : ℕ)
  (h1 : 2287 % n = r)
  (h2 : 2028 % n = r)
  (h3 : 1806 % n = r)
  (h_r_non_zero : r ≠ 0) : 
  n = 37 :=
by
  sorry

end find_n_l288_288690


namespace product_probability_probability_one_l288_288185

def S : Set Int := {13, 57}

theorem product_probability (a b : Int) (h₁ : a ∈ S) (h₂ : b ∈ S) (h₃ : a ≠ b) : 
  (a * b > 15) := 
by 
  sorry

theorem probability_one : 
  (∃ a b : Int, a ∈ S ∧ b ∈ S ∧ a ≠ b ∧ a * b > 15) ∧ 
  (∀ a b : Int, a ∈ S ∧ b ∈ S ∧ a ≠ b → a * b > 15) :=
by 
  sorry

end product_probability_probability_one_l288_288185


namespace combined_population_l288_288971

theorem combined_population (W PP LH : ℕ) 
  (hW : W = 900)
  (hPP : PP = 7 * W)
  (hLH : PP = LH + 800) : 
  (PP + LH) = 11800 :=
by
  sorry

end combined_population_l288_288971


namespace sum_of_primes_20_to_40_l288_288815

def is_prime (n : ℕ) : Prop :=
  n = 2 ∨ (2 < n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n)

def sum_of_primes_between (a b : ℕ) : ℕ :=
  (list.range' a (b - a)).filter is_prime |>.sum

theorem sum_of_primes_20_to_40 : sum_of_primes_between 20 40 = 120 := by
  sorry

end sum_of_primes_20_to_40_l288_288815


namespace factorization_l288_288873

theorem factorization (x y : ℝ) : (x * y^2 - x = x * (y - 1) * (y + 1)) :=
begin
  sorry
end

end factorization_l288_288873


namespace random_events_l288_288564

/-- Definition of what constitutes a random event --/
def is_random_event (e : String) : Prop :=
  e = "Drawing 3 first-quality glasses out of 10 glasses (8 first-quality, 2 substandard)" ∨
  e = "Forgetting the last digit of a phone number, randomly pressing and it is correct" ∨
  e = "Winning the first prize in a sports lottery"

/-- Define the specific events --/
def event_1 := "Drawing 3 first-quality glasses out of 10 glasses (8 first-quality, 2 substandard)"
def event_2 := "Forgetting the last digit of a phone number, randomly pressing and it is correct"
def event_3 := "Opposite electric charges attract each other"
def event_4 := "Winning the first prize in a sports lottery"

/-- Lean 4 statement for the proof problem --/
theorem random_events :
  (is_random_event event_1) ∧
  (is_random_event event_2) ∧
  ¬(is_random_event event_3) ∧
  (is_random_event event_4) :=
by 
  sorry

end random_events_l288_288564


namespace max_non_multiples_of_3_l288_288562

theorem max_non_multiples_of_3 (a b c d e f : ℕ) (h1 : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧ f > 0) (h2 : a * b * c * d * e * f % 3 = 0) : 
  ¬ ∃ (count : ℕ), count > 5 ∧ (∀ x ∈ [a, b, c, d, e, f], x % 3 ≠ 0) :=
by
  sorry

end max_non_multiples_of_3_l288_288562


namespace standard_equation_of_parabola_l288_288506

theorem standard_equation_of_parabola (x : ℝ) (y : ℝ) (directrix : ℝ) (eq_directrix : directrix = 1) :
  y^2 = -4 * x :=
sorry

end standard_equation_of_parabola_l288_288506


namespace difference_students_pets_in_all_classrooms_l288_288119

-- Definitions of the conditions
def students_per_classroom : ℕ := 24
def rabbits_per_classroom : ℕ := 3
def guinea_pigs_per_classroom : ℕ := 2
def number_of_classrooms : ℕ := 5

-- Proof problem statement
theorem difference_students_pets_in_all_classrooms :
  (students_per_classroom * number_of_classrooms) - 
  ((rabbits_per_classroom + guinea_pigs_per_classroom) * number_of_classrooms) = 95 := by
  sorry

end difference_students_pets_in_all_classrooms_l288_288119


namespace geometric_series_sum_l288_288709

theorem geometric_series_sum :
  let a := -3
  let r := -2
  let n := 9
  let term := a * r^(n-1)
  let Sn := (a * (r^n - 1)) / (r - 1)
  term = -768 → Sn = 514 := by
  intros a r n term Sn h_term
  sorry

end geometric_series_sum_l288_288709


namespace min_value_frac_ineq_l288_288002

theorem min_value_frac_ineq (a b : ℝ) (h1 : a > 1) (h2 : b > 2) (h3 : a + b = 5) : 
  (1 / (a - 1) + 9 / (b - 2)) = 8 :=
sorry

end min_value_frac_ineq_l288_288002


namespace inequality_relationship_l288_288906

noncomputable def f (x : ℝ) : ℝ := Real.exp (-(x - 1)^2)

def a : ℝ := f (Real.sqrt 2 / 2)
def b : ℝ := f (Real.sqrt 3 / 2)
def c : ℝ := f (Real.sqrt 6 / 2)

theorem inequality_relationship : b > c ∧ c > a :=
by
  sorry

end inequality_relationship_l288_288906


namespace smallest_perfect_square_divisible_by_2_3_5_l288_288361

theorem smallest_perfect_square_divisible_by_2_3_5 : 
  ∃ n : ℕ, n > 0 ∧ (∃ m : ℕ, n = m * m) ∧ (2 ∣ n) ∧ (3 ∣ n) ∧ (5 ∣ n) ∧
         (∀ k : ℕ, 
           (k > 0 ∧ (∃ l : ℕ, k = l * l) ∧ (2 ∣ k) ∧ (3 ∣ k) ∧ (5 ∣ k)) → n ≤ k) :=
sorry

end smallest_perfect_square_divisible_by_2_3_5_l288_288361


namespace rearrange_numbers_l288_288166

theorem rearrange_numbers (n : ℕ) (h : n = 25 ∨ n = 1000) :
  ∃ (l : List ℕ), l.nodup ∧ l.perm (List.range (n + 1))
  ∧ (∀ (i : ℕ), i < n → ((l.get? i.succ = some (l.get! i + 3)) ∨ (l.get? i.succ = some (l.get! i + 5)) ∨ (l.get? i.succ = some (l.get! i - 3)) ∨ (l.get? i.succ = some (l.get! i - 5)))) :=
by
  sorry

end rearrange_numbers_l288_288166


namespace mark_score_is_46_l288_288293

theorem mark_score_is_46 (highest_score : ℕ) (range: ℕ) (mark_score : ℕ) :
  highest_score = 98 →
  range = 75 →
  (mark_score = 2 * (highest_score - range)) →
  mark_score = 46 := by
  intros
  sorry

end mark_score_is_46_l288_288293


namespace feet_perpendiculars_concyclic_l288_288437

variables {S A B C D O M N P Q : Type} 

-- Given conditions
variables (is_convex_quadrilateral : convex_quadrilateral A B C D)
variables (diagonals_perpendicular : ∀ (AC BD : Line), perpendicular AC BD)
variables (foot_perpendicular : ∀ (O : Point), intersection_point O = foot (perpendicular_from S (base_quadrilateral A B C D)))

-- Define the proof statement
theorem feet_perpendiculars_concyclic
  (h1 : is_convex_quadrilateral A B C D)
  (h2 : diagonals_perpendicular AC BD)
  (h3 : foot_perpendicular O) :
  concyclic (feet_perpendicular_pts O (face S A B)) (feet_perpendicular_pts O (face S B C)) 
            (feet_perpendicular_pts O (face S C D)) (feet_perpendicular_pts O (face S D A)) := sorry

end feet_perpendiculars_concyclic_l288_288437


namespace geometric_series_sum_l288_288054

theorem geometric_series_sum (a r : ℝ)
  (h₁ : a / (1 - r) = 15)
  (h₂ : a / (1 - r^4) = 9) :
  r = 1 / 3 :=
sorry

end geometric_series_sum_l288_288054


namespace train_speed_l288_288093

theorem train_speed (length : ℝ) (time : ℝ) (length_eq : length = 375.03) (time_eq : time = 5) :
  let speed_kmph := (length / 1000) / (time / 3600)
  speed_kmph = 270.02 :=
by
  sorry

end train_speed_l288_288093


namespace isosceles_triangle_smallest_angle_l288_288565

def is_isosceles (angle_A angle_B angle_C : ℝ) : Prop := 
(angle_A = angle_B) ∨ (angle_B = angle_C) ∨ (angle_C = angle_A)

theorem isosceles_triangle_smallest_angle
  (angle_A angle_B angle_C : ℝ)
  (h_isosceles : is_isosceles angle_A angle_B angle_C)
  (h_angle_162 : angle_A = 162) :
  angle_B = 9 ∧ angle_C = 9 ∨ angle_A = 9 ∧ (angle_B = 9 ∨ angle_C = 9) :=
by
  sorry

end isosceles_triangle_smallest_angle_l288_288565


namespace how_many_children_l288_288516

-- Definitions based on conditions
def total_spectators : ℕ := 10000
def men : ℕ := 7000
def others : ℕ := total_spectators - men -- women + children
def children_per_woman : ℕ := 5

-- Variables
variable (W C : ℕ)

-- Conditions as Lean equalities
def condition1 : W + C = others := by sorry
def condition2 : C = children_per_woman * W := by sorry

-- Theorem statement to prove the number of children
theorem how_many_children (h1 : W + C = others) (h2 : C = children_per_woman * W) : C = 2500 :=
by sorry

end how_many_children_l288_288516


namespace equation_solution_l288_288491

theorem equation_solution (x : ℝ) (h₁ : 2 * x - 5 ≠ 0) (h₂ : 5 - 2 * x ≠ 0) :
  (x / (2 * x - 5) + 5 / (5 - 2 * x) = 1) ↔ (x = 0) :=
by
  sorry

end equation_solution_l288_288491


namespace g_difference_l288_288280

def g (n : ℕ) : ℚ := (1 / 4) * n * (n + 1) * (n + 2) * (n + 3)

theorem g_difference (s : ℕ) : g s - g (s - 1) = s * (s + 1) * (s + 2) := 
by sorry

end g_difference_l288_288280


namespace wooden_block_length_is_correct_l288_288339

noncomputable def length_of_block : ℝ :=
  let initial_length := 31
  let reduction := 30 / 100
  initial_length - reduction

theorem wooden_block_length_is_correct :
  length_of_block = 30.7 :=
by
  sorry

end wooden_block_length_is_correct_l288_288339


namespace price_of_brand_X_pen_l288_288245

variable (P : ℝ)

theorem price_of_brand_X_pen :
  (∀ (n : ℕ), n = 12 → 6 * P + 6 * 2.20 = 42 - 13.20) →
  P = 4.80 :=
by
  intro h₁
  have h₂ := h₁ 12 rfl
  sorry

end price_of_brand_X_pen_l288_288245


namespace min_value_l288_288593

theorem min_value (x : ℝ) (h : 0 < x) : x + 2 / (2 * x + 1) - 3 / 2 ≥ 0 :=
sorry

end min_value_l288_288593


namespace combined_population_port_perry_lazy_harbor_l288_288970

theorem combined_population_port_perry_lazy_harbor 
  (PP LH W : ℕ)
  (h1 : PP = 7 * W)
  (h2 : PP = LH + 800)
  (h3 : W = 900) :
  PP + LH = 11800 :=
by
  sorry

end combined_population_port_perry_lazy_harbor_l288_288970


namespace smallest_positive_perfect_square_divisible_by_2_3_5_l288_288384

theorem smallest_positive_perfect_square_divisible_by_2_3_5 : 
  ∃ (n : ℕ), (∃ k : ℕ, n = k^2) ∧ n % 2 = 0 ∧ n % 3 = 0 ∧ n % 5 = 0 ∧ 
  ∀ m : ℕ, (∃ k : ℕ, m = k^2) ∧ m % 2 = 0 ∧ m % 3 = 0 ∧ m % 5 = 0 → n ≤ m :=
sorry

end smallest_positive_perfect_square_divisible_by_2_3_5_l288_288384


namespace sum_binomial_2k_eq_2_2n_l288_288830

open scoped BigOperators

noncomputable def binomial_coeff (n k : ℕ) : ℕ :=
  Nat.choose n k

theorem sum_binomial_2k_eq_2_2n (n : ℕ) :
  ∑ k in Finset.range (n + 1), 2^k * binomial_coeff (2*n - k) n = 2^(2*n) := 
by
  sorry

end sum_binomial_2k_eq_2_2n_l288_288830


namespace find_ordered_pair_l288_288890

theorem find_ordered_pair : ∃ (x y : ℚ), 
  3 * x - 4 * y = -7 ∧ 4 * x + 5 * y = 23 ∧ 
  x = 57 / 31 ∧ y = 195 / 62 :=
by {
  sorry
}

end find_ordered_pair_l288_288890


namespace circle_radius_five_c_value_l288_288423

theorem circle_radius_five_c_value {c : ℝ} :
  (∀ x y : ℝ, x^2 + 8 * x + y^2 + 2 * y + c = 0) → 
  (∃ x y : ℝ, (x + 4)^2 + (y + 1)^2 = 25) → 
  c = 42 :=
by
  sorry

end circle_radius_five_c_value_l288_288423


namespace find_x_l288_288796

noncomputable def x_half_y (x y : ℚ) : Prop := x = (1 / 2) * y
noncomputable def y_third_z (y z : ℚ) : Prop := y = (1 / 3) * z

theorem find_x (x y z : ℚ) (h₁ : x_half_y x y) (h₂ : y_third_z y z) (h₃ : z = 100) :
  x = 16 + (2 / 3 : ℚ) :=
by
  sorry

end find_x_l288_288796


namespace initial_people_in_elevator_l288_288985

theorem initial_people_in_elevator (W n : ℕ) (avg_initial_weight avg_new_weight new_person_weight : ℚ)
  (h1 : avg_initial_weight = 152)
  (h2 : avg_new_weight = 151)
  (h3 : new_person_weight = 145)
  (h4 : W = n * avg_initial_weight)
  (h5 : W + new_person_weight = (n + 1) * avg_new_weight) :
  n = 6 :=
by
  sorry

end initial_people_in_elevator_l288_288985


namespace smallest_positive_perfect_square_divisible_by_2_3_5_l288_288366

theorem smallest_positive_perfect_square_divisible_by_2_3_5 : 
  ∃ n : ℕ, (n > 0 ∧ n % 2 = 0 ∧ n % 3 = 0 ∧ n % 5 = 0) ∧ ∃ k : ℕ, n = k^2 ∧ n = 225 := 
begin
  sorry
end

end smallest_positive_perfect_square_divisible_by_2_3_5_l288_288366


namespace number_of_nurses_l288_288537

theorem number_of_nurses (total : ℕ) (ratio_d_to_n : ℕ → ℕ) (h1 : total = 250) (h2 : ratio_d_to_n 2 = 3) : ∃ n : ℕ, n = 150 := 
by
  sorry

end number_of_nurses_l288_288537


namespace find_p_over_q_at_neg1_l288_288786

noncomputable def p (x : ℝ) : ℝ := (-27 / 8) * x
noncomputable def q (x : ℝ) : ℝ := (x + 5) * (x - 1)

theorem find_p_over_q_at_neg1 : p (-1) / q (-1) = 27 / 64 := by
  -- Skipping the proof
  sorry

end find_p_over_q_at_neg1_l288_288786


namespace smallest_number_of_locks_and_keys_l288_288197

open Finset Nat

-- Definitions based on conditions
def committee : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}

def can_open_safe (members : Finset ℕ) : Prop :=
  ∀ (subset : Finset ℕ), subset.card = 6 → members ⊆ subset

def cannot_open_safe (members : Finset ℕ) : Prop :=
  ∃ (subset : Finset ℕ), subset.card = 5 ∧ members ⊆ subset

-- Proof statement
theorem smallest_number_of_locks_and_keys :
  ∃ (locks keys : ℕ), locks = 462 ∧ keys = 2772 ∧
  (∀ (subset : Finset ℕ), subset.card = 6 → can_open_safe subset) ∧
  (∀ (subset : Finset ℕ), subset.card = 5 → ¬can_open_safe subset) :=
sorry

end smallest_number_of_locks_and_keys_l288_288197


namespace larger_sphere_radius_l288_288200

theorem larger_sphere_radius (r : ℝ) (π : ℝ) (h : r^3 = 2) :
  r = 2^(1/3) :=
by
  sorry

end larger_sphere_radius_l288_288200


namespace coloring_satisfies_conditions_l288_288466

-- Define lattice points as points with integer coordinates
structure LatticePoint :=
  (x : ℤ)
  (y : ℤ)

-- Define the color function
def color (p : LatticePoint) : ℕ :=
  if (p.x % 2 = 0) ∧ (p.y % 2 = 1) then 0 -- Black
  else if (p.x % 2 = 1) ∧ (p.y % 2 = 0) then 1 -- White
  else 2 -- Red

-- Define condition (1)
def infinite_lines_with_color (c : ℕ) : Prop :=
  ∀ k : ℤ, ∃ p : LatticePoint, color p = c ∧ p.x = k

-- Define condition (2)
def parallelogram_exists (A B C : LatticePoint) (wc rc bc : ℕ) : Prop :=
  (color A = wc) ∧ (color B = rc) ∧ (color C = bc) →
  ∃ D : LatticePoint, color D = rc ∧ D.x = C.x + (A.x - B.x) ∧ D.y = C.y + (A.y - B.y)

-- Main theorem
theorem coloring_satisfies_conditions :
  (∀ c : ℕ, ∃ p : LatticePoint, infinite_lines_with_color c) ∧
  (∀ A B C : LatticePoint, ∃ wc rc bc : ℕ, parallelogram_exists A B C wc rc bc) :=
sorry

end coloring_satisfies_conditions_l288_288466


namespace constant_term_binomial_expansion_l288_288619

theorem constant_term_binomial_expansion :
  ∃ (r : ℕ), (8 - 2 * r = 0) ∧ Nat.choose 8 r = 70 := by
  sorry

end constant_term_binomial_expansion_l288_288619


namespace dihedral_angle_sum_bounds_l288_288645

variable (α β γ : ℝ)

/-- The sum of the internal dihedral angles of a trihedral angle is greater than 180 degrees and less than 540 degrees. -/
theorem dihedral_angle_sum_bounds (hα: α < 180) (hβ: β < 180) (hγ: γ < 180) : 180 < α + β + γ ∧ α + β + γ < 540 :=
by
  sorry

end dihedral_angle_sum_bounds_l288_288645


namespace area_of_black_region_l288_288229

-- Definitions for the side lengths of the smaller and larger squares
def s₁ : ℕ := 4
def s₂ : ℕ := 8

-- The mathematical problem statement in Lean 4
theorem area_of_black_region : (s₂ * s₂) - (s₁ * s₁) = 48 := by
  sorry

end area_of_black_region_l288_288229


namespace inequality_abc_l288_288904

def f (x : ℝ) : ℝ := Real.exp (-(x - 1)^2)

def a : ℝ := f (Real.sqrt 2 / 2)
def b : ℝ := f (Real.sqrt 3 / 2)
def c : ℝ := f (Real.sqrt 6 / 2)

theorem inequality_abc : b > c ∧ c > a := sorry

end inequality_abc_l288_288904


namespace greatest_four_digit_multiple_of_17_l288_288989

theorem greatest_four_digit_multiple_of_17 :
  ∃ n, (n % 17 = 0) ∧ (1000 ≤ n ∧ n ≤ 9999) ∧ ∀ m, (m % 17 = 0) ∧ (1000 ≤ m ∧ m ≤ 9999) → n ≥ m :=
begin
  use 9996,
  split,
  { norm_num, },
  split,
  { norm_num, },
  { intros m hm,
    cases hm with hm1 hm2,
    cases hm2 with hm2a hm2b,
    exact nat.mul_le_mul_right _ (nat.div_le_of_le_mul (nat.le_sub_one_of_lt (lt_of_le_of_lt (nat.mul_le_mul_right _ (nat.le_of_dvd hm1)) (by norm_num)))),
  },
  sorry
end

end greatest_four_digit_multiple_of_17_l288_288989


namespace ashton_pencils_left_l288_288101

theorem ashton_pencils_left (boxes : ℕ) (pencils_per_box : ℕ) (pencils_given : ℕ) :
  boxes = 2 → pencils_per_box = 14 → pencils_given = 6 → (boxes * pencils_per_box) - pencils_given = 22 :=
by
  intros boxes_eq pencils_per_box_eq pencils_given_eq
  rw [boxes_eq, pencils_per_box_eq, pencils_given_eq]
  norm_num
  sorry

end ashton_pencils_left_l288_288101


namespace circumscribed_sphere_surface_area_l288_288445

theorem circumscribed_sphere_surface_area 
    (x y z : ℝ) 
    (h1 : x * y = Real.sqrt 6) 
    (h2 : y * z = Real.sqrt 2) 
    (h3 : z * x = Real.sqrt 3) : 
    4 * Real.pi * ((Real.sqrt (x^2 + y^2 + z^2)) / 2)^2 = 6 * Real.pi := 
by
  sorry

end circumscribed_sphere_surface_area_l288_288445


namespace min_ab_given_parallel_l288_288899

-- Define the conditions
def parallel_vectors (a b : ℝ) : Prop :=
  4 * b - a * (b - 1) = 0 ∧ b > 1

-- Prove the main statement
theorem min_ab_given_parallel (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h_parallel : parallel_vectors a b) :
  a + b = 9 :=
sorry  -- Proof is omitted

end min_ab_given_parallel_l288_288899


namespace total_length_of_pencil_l288_288454

def purple := 3
def black := 2
def blue := 1
def total_length := purple + black + blue

theorem total_length_of_pencil : total_length = 6 := 
by 
  sorry -- proof not needed

end total_length_of_pencil_l288_288454


namespace rectangle_area_l288_288338

variable (l w : ℕ)

def length_is_three_times_width := l = 3 * w

def perimeter_is_160 := 2 * l + 2 * w = 160

theorem rectangle_area : 
  length_is_three_times_width l w → 
  perimeter_is_160 l w → 
  l * w = 1200 :=
by
  intros h₁ h₂
  sorry

end rectangle_area_l288_288338


namespace cherry_tree_leaves_l288_288559

theorem cherry_tree_leaves (original_plan : ℕ) (multiplier : ℕ) (leaves_per_tree : ℕ) 
  (h1 : original_plan = 7) (h2 : multiplier = 2) (h3 : leaves_per_tree = 100) : 
  (original_plan * multiplier * leaves_per_tree = 1400) :=
by
  sorry

end cherry_tree_leaves_l288_288559


namespace price_per_strawberry_basket_is_9_l288_288948

-- Define the conditions
def strawberry_plants := 5
def tomato_plants := 7
def strawberries_per_plant := 14
def tomatoes_per_plant := 16
def items_per_basket := 7
def price_per_tomato_basket := 6
def total_revenue := 186

-- Define the total number of strawberries and tomatoes harvested
def total_strawberries := strawberry_plants * strawberries_per_plant
def total_tomatoes := tomato_plants * tomatoes_per_plant

-- Define the number of baskets of strawberries and tomatoes
def strawberry_baskets := total_strawberries / items_per_basket
def tomato_baskets := total_tomatoes / items_per_basket

-- Define the revenue from tomato baskets
def revenue_tomatoes := tomato_baskets * price_per_tomato_basket

-- Define the revenue from strawberry baskets
def revenue_strawberries := total_revenue - revenue_tomatoes

-- Calculate the price per basket of strawberries (which should be $9)
def price_per_strawberry_basket := revenue_strawberries / strawberry_baskets

theorem price_per_strawberry_basket_is_9 : 
  price_per_strawberry_basket = 9 := by
    sorry

end price_per_strawberry_basket_is_9_l288_288948


namespace fewest_seats_to_be_occupied_l288_288344

theorem fewest_seats_to_be_occupied (n : ℕ) (h : n = 120) : ∃ m, m = 40 ∧
  ∀ a b, a + b = n → a ≥ m → ∀ x, (x > 0 ∧ x ≤ n) → (x > 1 → a = m → a + (b / 2) ≥ n / 3) :=
sorry

end fewest_seats_to_be_occupied_l288_288344


namespace sum_of_primes_20_to_40_l288_288816

def is_prime (n : ℕ) : Prop :=
  n = 2 ∨ (2 < n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n)

def sum_of_primes_between (a b : ℕ) : ℕ :=
  (list.range' a (b - a)).filter is_prime |>.sum

theorem sum_of_primes_20_to_40 : sum_of_primes_between 20 40 = 120 := by
  sorry

end sum_of_primes_20_to_40_l288_288816


namespace surface_area_sphere_l288_288023

-- Definitions based on conditions
def SA : ℝ := 3
def SB : ℝ := 4
def SC : ℝ := 5
def vertices_perpendicular : Prop := ∀ (a b c : ℝ), (a = SA ∧ b = SB ∧ c = SC) → (a * b * c = SA * SB * SC)

-- Definition of the theorem based on problem and correct answer
theorem surface_area_sphere (h1 : vertices_perpendicular) : 
  4 * Real.pi * ((Real.sqrt (SA^2 + SB^2 + SC^2)) / 2)^2 = 50 * Real.pi :=
by
  -- skip the proof
  sorry

end surface_area_sphere_l288_288023


namespace basketball_children_l288_288510

/-- Given:
  1. total spectators is 10,000
  2. 7,000 of them were men
  3. Of the remaining spectators, there were 5 times as many children as women

Prove that the number of children was 2,500. -/
theorem basketball_children (total_spectators : ℕ) (men : ℕ) (women_children : ℕ) (women children : ℕ) 
  (h1 : total_spectators = 10000) 
  (h2 : men = 7000) 
  (h3 : women_children = total_spectators - men) 
  (h4 : women + 5 * women = women_children) 
  : children = 5 * 500 := 
  by 
  sorry

end basketball_children_l288_288510


namespace kerosene_price_increase_l288_288622

theorem kerosene_price_increase (P C : ℝ) (x : ℝ)
  (h1 : 1 = (1 + x / 100) * 0.8) :
  x = 25 := by
  sorry

end kerosene_price_increase_l288_288622


namespace power_expression_evaluation_l288_288833

theorem power_expression_evaluation :
  (1 / 2) ^ 2016 * (-2) ^ 2017 * (-1) ^ 2017 = 2 := 
by
  sorry

end power_expression_evaluation_l288_288833


namespace flat_fee_is_65_l288_288548

-- Define the problem constants
def George_nights : ℕ := 3
def Noah_nights : ℕ := 6
def George_cost : ℤ := 155
def Noah_cost : ℤ := 290

-- Prove that the flat fee for the first night is 65, given the costs and number of nights stayed.
theorem flat_fee_is_65 
  (f n : ℤ)
  (h1 : f + (George_nights - 1) * n = George_cost)
  (h2 : f + (Noah_nights - 1) * n = Noah_cost) :
  f = 65 := 
sorry

end flat_fee_is_65_l288_288548


namespace add_solution_y_to_solution_x_l288_288326

theorem add_solution_y_to_solution_x
  (x_volume : ℝ) (x_percent : ℝ) (y_percent : ℝ) (desired_percent : ℝ) (final_volume : ℝ)
  (x_alcohol : ℝ := x_volume * x_percent / 100) (y : ℝ := final_volume - x_volume) :
  (x_percent = 10) → (y_percent = 30) → (desired_percent = 15) → (x_volume = 300) →
  (final_volume = 300 + y) →
  ((x_alcohol + y * y_percent / 100) / final_volume = desired_percent / 100) →
  y = 100 := by
    intros h1 h2 h3 h4 h5 h6
    sorry

end add_solution_y_to_solution_x_l288_288326


namespace Matilda_correct_age_l288_288929

def Louis_age : ℕ := 14
def Jerica_age : ℕ := 2 * Louis_age
def Matilda_age : ℕ := Jerica_age + 7

theorem Matilda_correct_age : Matilda_age = 35 :=
by
  -- Proof needs to be filled here
  sorry

end Matilda_correct_age_l288_288929


namespace cos_inequality_l288_288615

open Real

-- Given angles of a triangle A, B, C

theorem cos_inequality (A B C : ℝ) (hA : 0 < A) (hB : 0 < B) (hC : 0 < C) (hTriangle : A + B + C = π) :
  1 / (1 + cos B ^ 2 + cos C ^ 2) + 1 / (1 + cos C ^ 2 + cos A ^ 2) + 1 / (1 + cos A ^ 2 + cos B ^ 2) ≤ 2 :=
by
  sorry

end cos_inequality_l288_288615


namespace range_of_b_l288_288591

noncomputable def f (x a b : ℝ) := (x - a)^2 * (x + b) * Real.exp x

theorem range_of_b (a b : ℝ) (h_max : ∃ δ > 0, ∀ x, |x - a| < δ → f x a b ≤ f a a b) : b < -a := sorry

end range_of_b_l288_288591


namespace uranus_appearance_minutes_after_6AM_l288_288799

-- Definitions of the given times and intervals
def mars_last_seen : Int := 0 -- 12:10 AM in minutes after midnight
def jupiter_after_mars : Int := 2 * 60 + 41 -- 2 hours and 41 minutes in minutes
def uranus_after_jupiter : Int := 3 * 60 + 16 -- 3 hours and 16 minutes in minutes
def reference_time : Int := 6 * 60 -- 6:00 AM in minutes after midnight

-- Statement of the problem
theorem uranus_appearance_minutes_after_6AM :
  let jupiter_first_appearance := mars_last_seen + jupiter_after_mars
  let uranus_first_appearance := jupiter_first_appearance + uranus_after_jupiter
  (uranus_first_appearance - reference_time) = 7 := by
  sorry

end uranus_appearance_minutes_after_6AM_l288_288799


namespace second_round_score_l288_288310

/-- 
  Given the scores in three rounds of darts, where the second round score is twice the
  first round score, and the third round score is 1.5 times the second round score,
  prove that the score in the second round is 48, given that the maximum score in the 
  third round is 72.
-/
theorem second_round_score (x y z : ℝ) (h1 : y = 2 * x) (h2 : z = 1.5 * y) (h3 : z = 72) : y = 48 :=
sorry

end second_round_score_l288_288310


namespace true_propositions_count_l288_288665

theorem true_propositions_count
  (a b c : ℝ)
  (h : a > b) :
  ( (a > b → a * c^2 > b * c^2) ∧
    (a * c^2 > b * c^2 → a > b) ∧
    (a ≤ b → a * c^2 ≤ b * c^2) ∧
    (a * c^2 ≤ b * c^2 → a ≤ b) 
  ) ∧ 
  (¬(a > b → a * c^2 > b * c^2) ∧
   ¬(a * c^2 ≤ b * c^2 → a ≤ b)) →
  (a * c^2 > b * c^2 → a > b) ∧
  (a ≤ b → a * c^2 ≤ b * c^2) ∨
  (a > b → a * c^2 > b * c^2) ∨
  (a * c^2 ≤ b * c^2 → a ≤ b) :=
sorry

end true_propositions_count_l288_288665


namespace probability_of_three_digit_number_div_by_three_l288_288066

noncomputable def probability_three_digit_div_by_three : ℚ :=
  let digit_mod3_groups := 
    {rem0 := {3, 6, 9}, rem1 := {1, 4, 7}, rem2 := {2, 5, 8}} in
  let valid_combinations :=
    (finset.card (finset.powersetLen 3 digit_mod3_groups.rem0) +
     (finset.card digit_mod3_groups.rem0 * finset.card digit_mod3_groups.rem1 * finset.card digit_mod3_groups.rem2) +
     finset.card (finset.powersetLen 3 digit_mod3_groups.rem1) +
     finset.card (finset.powersetLen 3 digit_mod3_groups.rem2))
  in
  let total_combinations := finset.card (finset.powersetLen 3 (finset.univ : finset (fin 9))) in
  (valid_combinations : ℚ) / total_combinations

theorem probability_of_three_digit_number_div_by_three :
  probability_three_digit_div_by_three = 5 / 14 := by
  -- provide proof here
  sorry

end probability_of_three_digit_number_div_by_three_l288_288066


namespace female_students_count_l288_288050

theorem female_students_count 
  (total_average : ℝ)
  (male_count : ℕ)
  (male_average : ℝ)
  (female_average : ℝ)
  (female_count : ℕ) 
  (correct_female_count : female_count = 12)
  (h1 : total_average = 90)
  (h2 : male_count = 8)
  (h3 : male_average = 87)
  (h4 : female_average = 92) :
  total_average * (male_count + female_count) = male_count * male_average + female_count * female_average :=
by sorry

end female_students_count_l288_288050


namespace find_eg_dot_fh_l288_288137

variables {V : Type*} [inner_product_space ℝ V]

-- Definitions and conditions given in the problem
variable P A B C E F G H : V

noncomputable def edge_length := 1
def tetrahedron_regular : Prop := 
  dist P A = edge_length ∧
  dist P B = edge_length ∧
  dist P C = edge_length ∧
  dist A B = edge_length ∧
  dist A C = edge_length ∧
  dist B C = edge_length

def is_edge_point (u v : V) (t : ℝ) : Prop := 
  0 ≤ t ∧ t ≤ 1 ∧ E = t • u + (1 - t) • v

def points_on_edges : Prop := 
  is_edge_point P A E ∧
  is_edge_point P B F ∧
  is_edge_point C A G ∧
  is_edge_point C B H

def vector_condition1 : Prop := 
  E + F = B

def vector_condition2 : Prop := 
  inner_product (E - H) (F - G) = 1 / 18

theorem find_eg_dot_fh :
  tetrahedron_regular →
  points_on_edges →
  vector_condition1 →
  vector_condition2 →
  inner_product (E - G) (F - H) = 5 / 18 :=
sorry

end find_eg_dot_fh_l288_288137


namespace jackson_sandwiches_l288_288625

theorem jackson_sandwiches (weeks : ℕ) (missed_wednesdays : ℕ) (missed_fridays : ℕ)
    (h_weeks : weeks = 36) (h_missed_wednesdays : missed_wednesdays = 1) (h_missed_fridays : missed_fridays = 2) :
    let total_days := weeks * 2
    let missed_days := missed_wednesdays + missed_fridays
    total_days - missed_days = 69 :=
by
    sorry

end jackson_sandwiches_l288_288625


namespace number_of_tables_l288_288748

theorem number_of_tables (c t : ℕ) (h1 : c = 8 * t) (h2 : 4 * c + 3 * t = 759) : t = 22 := by
  sorry

end number_of_tables_l288_288748


namespace consecutive_diff_possible_l288_288446

variable (a b c : ℝ)

def greater_than_2022 :=
  a > 2022 ∨ b > 2022 ∨ c > 2022

def distinct_numbers :=
  a ≠ b ∧ b ≠ c ∧ a ≠ c

theorem consecutive_diff_possible :
  greater_than_2022 a b c → distinct_numbers a b c → 
  ∃ (x y z : ℤ), x + 1 = y ∧ y + 1 = z ∧ 
  (a^2 - b^2 = ↑x) ∧ (b^2 - c^2 = ↑y) ∧ (c^2 - a^2 = ↑z) :=
by
  intros h1 h2
  -- proof goes here
  sorry

end consecutive_diff_possible_l288_288446


namespace roots_of_modified_quadratic_l288_288111

theorem roots_of_modified_quadratic 
  (k : ℝ) (hk : 0 < k) :
  (∃ z₁ z₂ : ℂ, (12 * z₁^2 - 4 * I * z₁ - k = 0) ∧ (12 * z₂^2 - 4 * I * z₂ - k = 0) ∧ (z₁ ≠ z₂) ∧ (z₁.im = 0) ∧ (z₂.im ≠ 0)) ↔ (k = 1/4) :=
by
  sorry

end roots_of_modified_quadratic_l288_288111


namespace simplify_expression_l288_288965

theorem simplify_expression (x y : ℕ) (hx : x ≠ 0) (hy : y ≠ 0) :
  (2 * x * (x^2 * y - x * y^2) + x * y * (2 * x * y - x^2)) / (x^2 * y) = x := 
by
  sorry

end simplify_expression_l288_288965


namespace all_equal_l288_288438

theorem all_equal (n : ℕ) (a : ℕ → ℝ) (h1 : 3 < n)
  (h2 : ∀ k : ℕ, k < n -> (a k)^3 = (a (k + 1 % n))^2 + (a (k + 2 % n))^2 + (a (k + 3 % n))^2) : 
  ∀ i j : ℕ, i < n -> j < n -> a i = a j :=
by
  sorry

end all_equal_l288_288438


namespace average_of_three_numbers_is_165_l288_288345

variable (x y z : ℕ)
variable (hy : y = 90)
variable (h1 : z = 4 * y)
variable (h2 : y = 2 * x)

theorem average_of_three_numbers_is_165 : (x + y + z) / 3 = 165 := by
  sorry

end average_of_three_numbers_is_165_l288_288345


namespace geometric_series_common_ratio_l288_288006

theorem geometric_series_common_ratio (a₁ q : ℝ) (S₃ : ℝ)
  (h1 : S₃ = 7 * a₁)
  (h2 : S₃ = a₁ + a₁ * q + a₁ * q^2) :
  q = 2 ∨ q = -3 :=
by
  sorry

end geometric_series_common_ratio_l288_288006


namespace red_balls_number_l288_288756

namespace BallDrawing

variable (x : ℕ) -- define x as the number of red balls

noncomputable def total_balls : ℕ := x + 4
noncomputable def yellow_ball_probability : ℚ := 4 / total_balls x

theorem red_balls_number : yellow_ball_probability x = 0.2 → x = 16 :=
by
  unfold yellow_ball_probability
  sorry

end BallDrawing

end red_balls_number_l288_288756


namespace parallelogram_base_l288_288252

theorem parallelogram_base (height area : ℕ) (h_height : height = 18) (h_area : area = 612) : ∃ base, base = 34 :=
by
  -- The proof would go here
  sorry

end parallelogram_base_l288_288252


namespace post_height_l288_288230

theorem post_height 
  (circumference : ℕ) 
  (rise_per_circuit : ℕ) 
  (travel_distance : ℕ)
  (circuits : ℕ := travel_distance / circumference) 
  (total_rise : ℕ := circuits * rise_per_circuit) 
  (c : circumference = 3)
  (r : rise_per_circuit = 4)
  (t : travel_distance = 9) :
  total_rise = 12 := by
  sorry

end post_height_l288_288230


namespace base_conversion_403_base_6_eq_223_base_8_l288_288235

theorem base_conversion_403_base_6_eq_223_base_8 :
  (6^2 * 4 + 6^1 * 0 + 6^0 * 3 : ℕ) = (8^2 * 2 + 8^1 * 2 + 8^0 * 3 : ℕ) :=
by
  sorry

end base_conversion_403_base_6_eq_223_base_8_l288_288235


namespace smallest_perfect_square_divisible_by_2_3_5_l288_288358

theorem smallest_perfect_square_divisible_by_2_3_5 : 
  ∃ n : ℕ, n > 0 ∧ (∃ m : ℕ, n = m * m) ∧ (2 ∣ n) ∧ (3 ∣ n) ∧ (5 ∣ n) ∧
         (∀ k : ℕ, 
           (k > 0 ∧ (∃ l : ℕ, k = l * l) ∧ (2 ∣ k) ∧ (3 ∣ k) ∧ (5 ∣ k)) → n ≤ k) :=
sorry

end smallest_perfect_square_divisible_by_2_3_5_l288_288358


namespace files_remaining_l288_288602

def initial_music_files : ℕ := 27
def initial_video_files : ℕ := 42
def initial_doc_files : ℕ := 12
def compression_ratio_music : ℕ := 2
def compression_ratio_video : ℕ := 3
def files_deleted : ℕ := 11

def compressed_music_files : ℕ := initial_music_files * compression_ratio_music
def compressed_video_files : ℕ := initial_video_files * compression_ratio_video
def total_compressed_files : ℕ := compressed_music_files + compressed_video_files + initial_doc_files

theorem files_remaining : total_compressed_files - files_deleted = 181 := by
  -- we skip the proof for now
  sorry

end files_remaining_l288_288602


namespace find_s_l288_288029

theorem find_s (c d n r s : ℝ) 
(h1 : c * d = 3)
(h2 : ∃ p q : ℝ, (p + q = r) ∧ (p * q = s) ∧ (p = c + 1/d ∧ q = d + 1/c)) :
s = 16 / 3 :=
by
  sorry

end find_s_l288_288029


namespace rahul_batting_average_before_match_l288_288783

open Nat

theorem rahul_batting_average_before_match (R : ℕ) (A : ℕ) :
  (R + 69 = 6 * 54) ∧ (A = R / 5) → (A = 51) :=
by
  sorry

end rahul_batting_average_before_match_l288_288783


namespace triangle_problem_proof_l288_288594

-- Given conditions
variables {a b c : ℝ}
variables {A B C : ℝ}
variables (h1 : a * (Real.sin A - Real.sin B) = (c - b) * (Real.sin C + Real.sin B))
variables (h2 : c = Real.sqrt 7)
variables (area : ℝ := 3 * Real.sqrt 3 / 2)

-- Prove angle C = π / 3 and perimeter of triangle
theorem triangle_problem_proof 
(h1 : a * (Real.sin A - Real.sin B) = (c - b) * (Real.sin C + Real.sin B))
(h2 : c = Real.sqrt 7)
(area_condition : (1 / 2) * a * b * (Real.sin C) = area) :
  (C = Real.pi / 3) ∧ (a + b + c = 5 + Real.sqrt 7) := 
by
  sorry

end triangle_problem_proof_l288_288594


namespace roots_of_quadratic_are_integers_l288_288961

theorem roots_of_quadratic_are_integers
  (b c : ℤ)
  (Δ : ℤ)
  (h_discriminant: Δ = b^2 - 4 * c)
  (h_perfect_square: ∃ k : ℤ, k^2 = Δ)
  : (∃ x1 x2 : ℤ, x1 * x2 = c ∧ x1 + x2 = -b) :=
by
  sorry

end roots_of_quadratic_are_integers_l288_288961


namespace quilt_patch_cost_l288_288760

-- conditions
def quilt_length : ℕ := 16
def quilt_width : ℕ := 20
def patch_area : ℕ := 4
def first_ten_patch_cost : ℕ := 10
def subsequent_patch_cost : ℕ := 5

theorem quilt_patch_cost :
  let total_quilt_area := quilt_length * quilt_width,
      total_patches := total_quilt_area / patch_area,
      cost_first_ten := 10 * first_ten_patch_cost,
      remaining_patches := total_patches - 10,
      cost_remaining := remaining_patches * subsequent_patch_cost,
      total_cost := cost_first_ten + cost_remaining
  in total_cost = 450 :=
by
  sorry

end quilt_patch_cost_l288_288760


namespace fraction_Cal_to_Anthony_l288_288642

-- definitions for Mabel, Anthony, Cal, and Jade's transactions
def Mabel_transactions : ℕ := 90
def Anthony_transactions : ℕ := Mabel_transactions + (Mabel_transactions / 10)
def Jade_transactions : ℕ := 85
def Cal_transactions : ℕ := Jade_transactions - 19

-- goal: prove the fraction Cal handled compared to Anthony is 2/3
theorem fraction_Cal_to_Anthony : (Cal_transactions : ℚ) / (Anthony_transactions : ℚ) = 2 / 3 :=
by
  sorry

end fraction_Cal_to_Anthony_l288_288642


namespace min_value_xy_l288_288460

theorem min_value_xy (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 2 * x + y + 6 = x * y) : x * y ≥ 18 := 
sorry

end min_value_xy_l288_288460


namespace cars_needed_to_double_earnings_l288_288395

-- Define the conditions
def baseSalary : Int := 1000
def commissionPerCar : Int := 200
def januaryEarnings : Int := 1800

-- The proof goal
theorem cars_needed_to_double_earnings : 
  ∃ (carsSoldInFeb : Int), 
    1000 + commissionPerCar * carsSoldInFeb = 2 * januaryEarnings :=
by
  sorry

end cars_needed_to_double_earnings_l288_288395


namespace acute_triangle_inequality_l288_288033

variable (f : ℝ → ℝ)
variable {A B : ℝ}
variable (h₁ : ∀ x : ℝ, x * (f'' x) - 2 * (f x) > 0)
variable (h₂ : A + B < Real.pi / 2 ∧ 0 < A ∧ 0 < B)

theorem acute_triangle_inequality :
  f (Real.cos A) * (Real.sin B) ^ 2 < f (Real.sin B) * (Real.cos A) ^ 2 := 
  sorry

end acute_triangle_inequality_l288_288033


namespace negation_proposition_l288_288499

variable (n : ℕ)
variable (n_positive : n > 0)
variable (f : ℕ → ℕ)
variable (H1 : ∀ n, n > 0 → (f n) > 0 ∧ (f n) ≤ n)

theorem negation_proposition :
  (∃ n_0, n_0 > 0 ∧ ((f n_0) ≤ 0 ∨ (f n_0) > n_0)) ↔ ¬(∀ n, n > 0 → (f n) >0 ∧ (f n) ≤ n) :=
by 
  sorry

end negation_proposition_l288_288499


namespace lawn_mowing_rate_l288_288113

-- Definitions based on conditions
def total_hours_mowed : ℕ := 2 * 7
def money_left_after_expenses (R : ℕ) : ℕ := (14 * R) / 4

-- The problem statement
theorem lawn_mowing_rate (h : money_left_after_expenses R = 49) : R = 14 := 
sorry

end lawn_mowing_rate_l288_288113


namespace maximize_daily_profit_l288_288695

noncomputable def daily_profit : ℝ → ℝ → ℝ
| x, c => if h : 0 < x ∧ x ≤ c then (3 * (9 * x - 2 * x^2)) / (2 * (6 - x)) else 0

theorem maximize_daily_profit (c : ℝ) (x : ℝ) (h1 : 0 < c) (h2 : c < 6) :
  (y = daily_profit x c) ∧
  (if 0 < c ∧ c < 3 then x = c else if 3 ≤ c ∧ c < 6 then x = 3 else False) :=
by
  sorry

end maximize_daily_profit_l288_288695


namespace find_x_squared_inv_x_squared_l288_288900

theorem find_x_squared_inv_x_squared (x : ℝ) (h : x^3 + 1/x^3 = 110) : x^2 + 1/x^2 = 23 :=
sorry

end find_x_squared_inv_x_squared_l288_288900


namespace rectangular_prism_sum_of_dimensions_l288_288978

theorem rectangular_prism_sum_of_dimensions (a b c : ℕ) (h_volume : a * b * c = 21) 
(h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) (h_positive : 0 < a ∧ 0 < b ∧ 0 < c) : 
a + b + c = 11 :=
sorry

end rectangular_prism_sum_of_dimensions_l288_288978


namespace relatively_prime_sums_l288_288302

theorem relatively_prime_sums (x y : ℤ) (h : Int.gcd x y = 1) 
  : Int.gcd (x^2 + x * y + y^2) (x^2 + 3 * x * y + y^2) = 1 :=
by
  sorry

end relatively_prime_sums_l288_288302


namespace at_least_two_fail_l288_288664

theorem at_least_two_fail (p q : ℝ) (n : ℕ) (h_p : p = 0.2) (h_q : q = 1 - p) :
  n ≥ 18 → (1 - ((q^n) * (1 + n * p / 4))) ≥ 0.9 :=
by
  sorry

end at_least_two_fail_l288_288664


namespace probability_acute_angle_AMB_l288_288755

noncomputable def square_side_length := 2

structure Square :=
(A B C D : ℝ × ℝ)
(side_length : ℝ)
(valid_square : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D)

structure Point :=
(x y : ℝ)

def inside_square (P : Point) (S : Square) : Prop :=
S.A.1 ≤ P.x ∧ P.x ≤ S.C.1 ∧ S.A.2 ≤ P.y ∧ P.y ≤ S.C.2

def acute_angle_condition (P : Point) (S : Square) (θ : ℝ) : Prop :=
θ < π / 2

theorem probability_acute_angle_AMB (S : Square) (M : Point) (h_in_square : inside_square M S) :
  (1 - π / 8) = sorry :=
sorry

end probability_acute_angle_AMB_l288_288755


namespace how_many_children_l288_288518

-- Definitions based on conditions
def total_spectators : ℕ := 10000
def men : ℕ := 7000
def others : ℕ := total_spectators - men -- women + children
def children_per_woman : ℕ := 5

-- Variables
variable (W C : ℕ)

-- Conditions as Lean equalities
def condition1 : W + C = others := by sorry
def condition2 : C = children_per_woman * W := by sorry

-- Theorem statement to prove the number of children
theorem how_many_children (h1 : W + C = others) (h2 : C = children_per_woman * W) : C = 2500 :=
by sorry

end how_many_children_l288_288518


namespace all_div_by_6_upto_88_l288_288723

def numbers_divisible_by_6_upto_88 := [6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84]

theorem all_div_by_6_upto_88 :
  ∀ n : ℕ, 1 < n ∧ n ≤ 88 ∧ n % 6 = 0 → n ∈ numbers_divisible_by_6_upto_88 :=
by
  intro n
  intro h
  have h1 := h.1
  have h2 := h.2.1
  have h3 := h.2.2
  sorry

end all_div_by_6_upto_88_l288_288723


namespace sum_primes_between_20_and_40_l288_288813

open Nat

def primesBetween20And40 : List Nat := [23, 29, 31, 37]

theorem sum_primes_between_20_and_40 :
  (primesBetween20And40.sum = 120) :=
by
  sorry

end sum_primes_between_20_and_40_l288_288813


namespace solution_to_problem_l288_288249

theorem solution_to_problem (x y : ℕ) : 
  (x.gcd y + x.lcm y = x + y) ↔ 
  ∃ (d k : ℕ), (x = d ∧ y = d * k) ∨ (x = d * k ∧ y = d) :=
by sorry

end solution_to_problem_l288_288249


namespace converse_l288_288655

theorem converse (x y : ℝ) (h : x + y ≥ 5) : x ≥ 2 ∧ y ≥ 3 := 
sorry

end converse_l288_288655


namespace train_crossing_time_is_correct_l288_288449

noncomputable def train_crossing_time (train_length bridge_length : ℝ) (train_speed_kmph : ℝ) : ℝ :=
  let total_distance := train_length + bridge_length
  let train_speed_mps := train_speed_kmph * (1000 / 3600)
  total_distance / train_speed_mps

theorem train_crossing_time_is_correct :
  train_crossing_time 250 180 120 = 12.9 :=
by
  sorry

end train_crossing_time_is_correct_l288_288449


namespace simplify_expression_l288_288048

-- Define the given expression
def given_expression (x : ℝ) : ℝ := 5 * x + 9 * x^2 + 8 - (6 - 5 * x - 3 * x^2)

-- Define the expected simplified form
def expected_expression (x : ℝ) : ℝ := 12 * x^2 + 10 * x + 2

-- The theorem we want to prove
theorem simplify_expression (x : ℝ) : given_expression x = expected_expression x := by
  sorry

end simplify_expression_l288_288048


namespace compute_expression_l288_288860

theorem compute_expression : 2 + 7 * 3 - 4 + 8 / 2 = 23 := by
  sorry

end compute_expression_l288_288860


namespace ticket_number_l288_288493

-- Define the conditions and the problem
theorem ticket_number (x y z N : ℕ) (hx : 0 ≤ x ∧ x ≤ 9) (hy: 0 ≤ y ∧ y ≤ 9) (hz: 0 ≤ z ∧ z ≤ 9) 
(hN1: N = 100 * x + 10 * y + z) (hN2: N = 11 * (x + y + z)) : 
N = 198 :=
sorry

end ticket_number_l288_288493


namespace luke_bike_vs_bus_slowness_l288_288037

theorem luke_bike_vs_bus_slowness
  (luke_bus_time : ℕ)
  (paula_ratio : ℚ)
  (total_travel_time : ℕ)
  (paula_total_bus_time : ℕ)
  (luke_total_travel_time_lhs : ℕ)
  (luke_total_travel_time_rhs : ℕ)
  (bike_time : ℕ)
  (ratio : ℚ) :
  luke_bus_time = 70 ∧
  paula_ratio = 3 / 5 ∧
  total_travel_time = 504 ∧
  paula_total_bus_time = 2 * (paula_ratio * luke_bus_time) ∧
  luke_total_travel_time_lhs = luke_bus_time + bike_time ∧
  luke_total_travel_time_rhs + paula_total_bus_time = total_travel_time ∧
  bike_time = ratio * luke_bus_time ∧
  ratio = bike_time / luke_bus_time →
  ratio = 5 :=
sorry

end luke_bike_vs_bus_slowness_l288_288037


namespace quilt_patch_cost_l288_288761

-- conditions
def quilt_length : ℕ := 16
def quilt_width : ℕ := 20
def patch_area : ℕ := 4
def first_ten_patch_cost : ℕ := 10
def subsequent_patch_cost : ℕ := 5

theorem quilt_patch_cost :
  let total_quilt_area := quilt_length * quilt_width,
      total_patches := total_quilt_area / patch_area,
      cost_first_ten := 10 * first_ten_patch_cost,
      remaining_patches := total_patches - 10,
      cost_remaining := remaining_patches * subsequent_patch_cost,
      total_cost := cost_first_ten + cost_remaining
  in total_cost = 450 :=
by
  sorry

end quilt_patch_cost_l288_288761


namespace blake_initial_money_l288_288237

theorem blake_initial_money (amount_spent_oranges amount_spent_apples amount_spent_mangoes change_received initial_amount : ℕ)
  (h1 : amount_spent_oranges = 40)
  (h2 : amount_spent_apples = 50)
  (h3 : amount_spent_mangoes = 60)
  (h4 : change_received = 150)
  (h5 : initial_amount = (amount_spent_oranges + amount_spent_apples + amount_spent_mangoes) + change_received) :
  initial_amount = 300 :=
by
  sorry

end blake_initial_money_l288_288237


namespace solve_quadratic_expr_l288_288152

theorem solve_quadratic_expr (x : ℝ) (h : 2 * x^2 - 5 = 11) : 
  4 * x^2 + 4 * x + 1 = 33 + 8 * Real.sqrt 2 ∨ 4 * x^2 + 4 * x + 1 = 33 - 8 * Real.sqrt 2 := 
by 
  sorry

end solve_quadratic_expr_l288_288152


namespace solution_inequality_l288_288291

-- Conditions
variables {a b x : ℝ}
theorem solution_inequality (h1 : a < 0) (h2 : b = a) :
  {x : ℝ | (ax + b) ≤ 0} = {x : ℝ | x ≥ -1} →
  {x : ℝ | (ax + b) / (x - 2) > 0} = {x : ℝ | -1 < x ∧ x < 2} :=
by
  sorry

end solution_inequality_l288_288291


namespace sum_greater_than_two_l288_288500

variables {x y : ℝ}

theorem sum_greater_than_two (hx : x^7 > y^6) (hy : y^7 > x^6) : x + y > 2 :=
sorry

end sum_greater_than_two_l288_288500


namespace find_initial_apples_l288_288346

def initial_apples (a b c : ℕ) : Prop :=
  b + c = a

theorem find_initial_apples (a b initial_apples : ℕ) (h : b + initial_apples = a) : initial_apples = 8 :=
by
  sorry

end find_initial_apples_l288_288346


namespace max_happy_times_l288_288175

theorem max_happy_times (weights : Fin 2021 → ℝ) (unique_mass : Function.Injective weights) : 
  ∃ max_happy : Nat, max_happy = 673 :=
by
  sorry

end max_happy_times_l288_288175


namespace hyperbola_range_m_l288_288523

theorem hyperbola_range_m (m : ℝ) :
  (∃ x y : ℝ, (x^2 / (|m| - 1)) - (y^2 / (m - 2)) = 1) ↔ (m < -1) ∨ (m > 2) := 
by
  sorry

end hyperbola_range_m_l288_288523


namespace fraction_of_girls_in_debate_l288_288850

theorem fraction_of_girls_in_debate (g b : ℕ) (h : g = b) :
  ((2 / 3) * g) / ((2 / 3) * g + (3 / 5) * b) = 30 / 57 :=
by
  sorry

end fraction_of_girls_in_debate_l288_288850


namespace range_of_a_l288_288656

noncomputable def f (x : ℝ) : ℝ := x^2 * Real.exp x

theorem range_of_a (a : ℝ) : 
  (∃ x ∈ Set.Ioo a (a + 1), ∃ f' : ℝ → ℝ, ∀ x, f' x = (x * Real.exp x) * (x + 2) ∧ f' x = 0) ↔ 
  a ∈ Set.Ioo (-3 : ℝ) (-2) ∪ Set.Ioo (-1) (0) := 
sorry

end range_of_a_l288_288656


namespace group9_40_41_right_angled_l288_288097

theorem group9_40_41_right_angled :
  ¬ (∃ a b c : ℝ, a = 3 ∧ b = 4 ∧ c = 7 ∧ a^2 + b^2 = c^2) ∧
  ¬ (∃ a b c : ℝ, a = 1/3 ∧ b = 1/4 ∧ c = 1/5 ∧ a^2 + b^2 = c^2) ∧
  ¬ (∃ a b c : ℝ, a = 4 ∧ b = 6 ∧ c = 8 ∧ a^2 + b^2 = c^2) ∧
  (∃ a b c : ℝ, a = 9 ∧ b = 40 ∧ c = 41 ∧ a^2 + b^2 = c^2) :=
by
  sorry

end group9_40_41_right_angled_l288_288097


namespace range_of_m_l288_288331

variable (f : Real → Real)

-- Conditions
axiom odd_function : ∀ x, f (-x) = -f x
axiom decreasing_function : ∀ x y, x < y → -1 < x ∧ y < 1 → f x > f y
axiom domain : ∀ x, -1 < x ∧ x < 1 → true

-- The statement to be proved
theorem range_of_m (m : Real) : 
  f (1 - m) + f (1 - m^2) < 0 → 0 < m → m < 1 :=
by
  sorry

end range_of_m_l288_288331


namespace find_a_l288_288744

theorem find_a (S_n : ℕ → ℝ) (a_n : ℕ → ℝ) (a : ℝ) (h1 : ∀ n, S_n n = 3^(n+1) + a)
  (h2 : ∀ n, a_n (n+1) = S_n (n+1) - S_n n)
  (h3 : ∀ n m k, a_n m * a_n k = (a_n n)^2 → n = m + k) : 
  a = -3 := 
sorry

end find_a_l288_288744


namespace average_infect_influence_l288_288233

theorem average_infect_influence
  (x : ℝ)
  (h : (1 + x)^2 = 100) :
  x = 9 :=
sorry

end average_infect_influence_l288_288233


namespace speed_with_current_l288_288547

theorem speed_with_current (v : ℝ) (current_speed : ℝ) (against_current_speed : ℝ) 
  (h1 : current_speed = 2.8) 
  (h2 : against_current_speed = 9.4) 
  (h3 : against_current_speed = v - current_speed) 
  : (v + current_speed) = 15 := by
  sorry

end speed_with_current_l288_288547


namespace max_gcd_bn_bnp1_l288_288770

def b_n (n : ℕ) : ℤ := (7 ^ n - 4) / 3
def b_n_plus_1 (n : ℕ) : ℤ := (7 ^ (n + 1) - 4) / 3

theorem max_gcd_bn_bnp1 (n : ℕ) : ∃ d_max : ℕ, (∀ d : ℕ, (gcd (b_n n) (b_n_plus_1 n) ≤ d) → d ≤ d_max) ∧ d_max = 3 :=
sorry

end max_gcd_bn_bnp1_l288_288770


namespace smallest_positive_perfect_square_divisible_by_2_3_5_l288_288363

theorem smallest_positive_perfect_square_divisible_by_2_3_5 : 
  ∃ n : ℕ, (n > 0 ∧ n % 2 = 0 ∧ n % 3 = 0 ∧ n % 5 = 0) ∧ ∃ k : ℕ, n = k^2 ∧ n = 225 := 
begin
  sorry
end

end smallest_positive_perfect_square_divisible_by_2_3_5_l288_288363


namespace probability_is_one_third_l288_288189

noncomputable def probability_four_of_a_kind_or_full_house : ℚ :=
  let total_outcomes := 6
  let probability_triplet_match := 1 / total_outcomes
  let probability_pair_match := 1 / total_outcomes
  probability_triplet_match + probability_pair_match

theorem probability_is_one_third :
  probability_four_of_a_kind_or_full_house = 1 / 3 :=
by
  -- sorry
  trivial

end probability_is_one_third_l288_288189


namespace travel_time_third_to_first_l288_288405

variable (boat_speed current_speed : ℝ) -- speeds of the boat and current
variable (d1 d2 d3 : ℝ) -- distances between the docks

-- Conditions
variable (h1 : 30 / 60 = d1 / (boat_speed - current_speed)) -- 30 minutes from one dock to another against current
variable (h2 : 18 / 60 = d2 / (boat_speed + current_speed)) -- 18 minutes from another dock to the third with current
variable (h3 : d1 + d2 = d3) -- Total distance is sum of d1 and d2

theorem travel_time_third_to_first : (d3 / (boat_speed - current_speed)) * 60 = 72 := 
by 
  -- here goes the proof which is omitted
  sorry

end travel_time_third_to_first_l288_288405


namespace difference_in_combined_area_l288_288735

-- Define the dimensions of the two rectangular sheets of paper
def paper1_length : ℝ := 11
def paper1_width : ℝ := 17
def paper2_length : ℝ := 8.5
def paper2_width : ℝ := 11

-- Define the areas of one side of each sheet
def area1 : ℝ := paper1_length * paper1_width -- 187
def area2 : ℝ := paper2_length * paper2_width -- 93.5

-- Define the combined areas of front and back of each sheet
def combined_area1 : ℝ := 2 * area1 -- 374
def combined_area2 : ℝ := 2 * area2 -- 187

-- Prove that the difference in combined area is 187
theorem difference_in_combined_area : combined_area1 - combined_area2 = 187 :=
by 
  -- Using the definitions above to simplify the goal
  sorry

end difference_in_combined_area_l288_288735


namespace total_animals_l288_288670

theorem total_animals (total_legs : ℕ) (number_of_sheep : ℕ)
  (legs_per_chicken : ℕ) (legs_per_sheep : ℕ)
  (H1 : total_legs = 60) 
  (H2 : number_of_sheep = 10)
  (H3 : legs_per_chicken = 2)
  (H4 : legs_per_sheep = 4) : 
  number_of_sheep + (total_legs - number_of_sheep * legs_per_sheep) / legs_per_chicken = 20 :=
by {
  sorry
}

end total_animals_l288_288670


namespace classrooms_students_guinea_pigs_difference_l288_288581

theorem classrooms_students_guinea_pigs_difference :
  let students_per_classroom := 22
  let guinea_pigs_per_classroom := 3
  let number_of_classrooms := 5
  let total_students := students_per_classroom * number_of_classrooms
  let total_guinea_pigs := guinea_pigs_per_classroom * number_of_classrooms
  total_students - total_guinea_pigs = 95 :=
  by
    sorry

end classrooms_students_guinea_pigs_difference_l288_288581


namespace simplify_problem_l288_288323

noncomputable def simplify_expression : ℝ :=
  let numer := (Real.sqrt 3 - 1) ^ (1 - Real.sqrt 2)
  let denom := (Real.sqrt 3 + 1) ^ (1 + Real.sqrt 2)
  numer / denom

theorem simplify_problem :
  simplify_expression = 2 ^ (1 - Real.sqrt 2) * (4 - 2 * Real.sqrt 3) :=
by
  sorry

end simplify_problem_l288_288323


namespace solve_for_x_l288_288327

theorem solve_for_x (x : ℝ) (h : 12 - 2 * x = 6) : x = 3 :=
sorry

end solve_for_x_l288_288327


namespace find_rate_of_interest_l288_288052

noncomputable def rate_of_interest (P : ℝ) (r : ℝ) : Prop :=
  let CI2 := P * (1 + r)^2 - P
  let CI3 := P * (1 + r)^3 - P
  CI2 = 1200 ∧ CI3 = 1272 → r = 0.06

theorem find_rate_of_interest (P : ℝ) (r : ℝ) : rate_of_interest P r :=
by sorry

end find_rate_of_interest_l288_288052


namespace an_values_and_formula_is_geometric_sequence_l288_288483

-- Definitions based on the conditions
def Sn (n : ℕ) : ℝ := sorry  -- S_n to be defined in the context or problem details
def a (n : ℕ) : ℝ := 2 - Sn n

-- Prove the specific values and general formula given the condition a_n = 2 - S_n
theorem an_values_and_formula (Sn : ℕ → ℝ) :
  a 1 = 1 ∧ a 2 = 1 / 2 ∧ a 3 = 1 / 4 ∧ a 4 = 1 / 8 ∧ (∀ n, a n = (1 / 2)^(n-1)) :=
sorry

-- Prove the sequence is geometric
theorem is_geometric_sequence (Sn : ℕ → ℝ) :
  (∀ n, a n = (1 / 2)^(n-1)) → ∀ n, a (n + 1) / a n = 1 / 2 :=
sorry

end an_values_and_formula_is_geometric_sequence_l288_288483


namespace largest_root_ratio_l288_288941

-- Define the polynomials f(x) and g(x)
def f (x : ℝ) : ℝ := 1 - x - 4 * x^2 + x^4
def g (x : ℝ) : ℝ := 16 - 8 * x - 16 * x^2 + x^4

-- Define the property that x1 is the largest root of f(x) and x2 is the largest root of g(x)
def is_largest_root (p : ℝ → ℝ) (r : ℝ) : Prop := 
  p r = 0 ∧ ∀ x : ℝ, p x = 0 → x ≤ r

-- The main theorem
theorem largest_root_ratio (x1 x2 : ℝ) 
  (hx1 : is_largest_root f x1) 
  (hx2 : is_largest_root g x2) : x2 = 2 * x1 :=
sorry

end largest_root_ratio_l288_288941


namespace strictly_decreasing_interval_l288_288586

noncomputable def f (x : ℝ) : ℝ := x^3 - 3 * x^2 + 1

theorem strictly_decreasing_interval :
  ∀ x, (0 < x) ∧ (x < 2) → (deriv f x < 0) := by
sorry

end strictly_decreasing_interval_l288_288586


namespace triangular_number_difference_l288_288419

-- Definition of the nth triangular number
def triangular_number (n : ℕ) : ℕ :=
  n * (n + 1) / 2

-- Theorem stating the problem
theorem triangular_number_difference :
  triangular_number 2010 - triangular_number 2008 = 4019 :=
by
  sorry

end triangular_number_difference_l288_288419


namespace points_for_correct_answer_l288_288081

theorem points_for_correct_answer
  (x y a b : ℕ)
  (hx : x - y = 7)
  (hsum : a + b = 43)
  (hw_score : a * x - b * (20 - x) = 328)
  (hz_score : a * y - b * (20 - y) = 27) :
  a = 25 := 
sorry

end points_for_correct_answer_l288_288081


namespace factorize_expression_l288_288882

variable {x y : ℝ}

theorem factorize_expression : xy^2 - x = x * (y - 1) * (y + 1) := 
by
  -- Define the left-hand side of the equation
  let lhs := x * y^2 - x
  -- Define the right-hand side of the equation
  let rhs := x * (y - 1) * (y + 1)
  -- Provide the goal to prove
  show lhs = rhs
  sorry

end factorize_expression_l288_288882


namespace parabola_focus_coordinates_l288_288116

theorem parabola_focus_coordinates :
  ∃ h k : ℝ, (y = -1/8 * x^2 + 2 * x - 1) ∧ (h = 8 ∧ k = 5) :=
sorry

end parabola_focus_coordinates_l288_288116


namespace x_varies_as_z_l288_288612

variable {x y z : ℝ}
variable (k j : ℝ)
variable (h1 : x = k * y^3)
variable (h2 : y = j * z^(1/3))

theorem x_varies_as_z (m : ℝ) (h3 : m = k * j^3) : x = m * z := by
  sorry

end x_varies_as_z_l288_288612


namespace probability_different_numbers_probability_sum_six_probability_three_odds_in_five_throws_l288_288986

-- Probability for different numbers facing up when die is thrown twice
theorem probability_different_numbers :
  let n_faces := 6
  let total_outcomes := n_faces * n_faces
  let favorable_outcomes := n_faces * (n_faces - 1)
  let probability := (favorable_outcomes : ℚ) / total_outcomes
  probability = 5 / 6 :=
by
  sorry -- Proof to be filled

-- Probability for sum of numbers being 6 when die is thrown twice
theorem probability_sum_six :
  let n_faces := 6
  let total_outcomes := n_faces * n_faces
  let favorable_outcomes := 5
  let probability := (favorable_outcomes : ℚ) / total_outcomes
  probability = 5 / 36 :=
by
  sorry -- Proof to be filled

-- Probability for exactly three outcomes being odd when die is thrown five times
theorem probability_three_odds_in_five_throws :
  let n_faces := 6
  let n_throws := 5
  let p_odd := 3 / n_faces
  let p_even := 1 - p_odd
  let binomial_coeff := Nat.choose n_throws 3
  let p_three_odds := (binomial_coeff : ℚ) * (p_odd ^ 3) * (p_even ^ 2)
  p_three_odds = 5 / 16 :=
by
  sorry -- Proof to be filled

end probability_different_numbers_probability_sum_six_probability_three_odds_in_five_throws_l288_288986


namespace sum_of_triangle_angles_is_540_l288_288509

theorem sum_of_triangle_angles_is_540
  (A1 A3 A5 B2 B4 B6 C7 C8 C9 : ℝ)
  (H1 : A1 + A3 + A5 = 180)
  (H2 : B2 + B4 + B6 = 180)
  (H3 : C7 + C8 + C9 = 180) :
  A1 + A3 + A5 + B2 + B4 + B6 + C7 + C8 + C9 = 540 :=
by
  sorry

end sum_of_triangle_angles_is_540_l288_288509


namespace final_number_is_correct_l288_288224

-- Define the problem conditions as Lean definitions/statements
def original_number : ℤ := 4
def doubled_number (x : ℤ) : ℤ := 2 * x
def resultant_number (x : ℤ) : ℤ := doubled_number x + 9
def final_number (x : ℤ) : ℤ := 3 * resultant_number x

-- Formulate the theorem using the conditions
theorem final_number_is_correct :
  final_number original_number = 51 :=
by
  sorry

end final_number_is_correct_l288_288224


namespace simplify_fraction_expression_l288_288107

theorem simplify_fraction_expression : 
  (18 / 42 - 3 / 8 - 1 / 12 : ℚ) = -5 / 168 :=
by
  sorry

end simplify_fraction_expression_l288_288107


namespace simplify_and_sum_coefficients_l288_288576

theorem simplify_and_sum_coefficients :
  (∃ A B C D : ℤ, (∀ x : ℝ, x ≠ D → (x^3 + 6 * x^2 + 11 * x + 6) / (x + 1) = A * x^2 + B * x + C) ∧ A + B + C + D = 11) :=
sorry

end simplify_and_sum_coefficients_l288_288576


namespace M_gt_N_l288_288638

variable (x y : ℝ)

def M := x^2 + y^2 + 1
def N := 2 * (x + y - 1)

theorem M_gt_N : M x y > N x y := sorry

end M_gt_N_l288_288638


namespace hyperbola_eccentricity_l288_288932

-- Definition of the parabola C1: y^2 = 2px with p > 0.
def parabola (p : ℝ) (p_pos : 0 < p) (x y : ℝ) : Prop := y^2 = 2 * p * x

-- Definition of the hyperbola C2: x^2 / a^2 - y^2 / b^2 = 1 with a > 0 and b > 0.
def hyperbola (a b : ℝ) (a_pos : 0 < a) (b_pos : 0 < b) (x y : ℝ) : Prop := 
  x^2 / a^2 - y^2 / b^2 = 1

-- Definition of having a common focus F at (p / 2, 0).
def common_focus (p a b c : ℝ) (p_pos : 0 < p) (a_pos : 0 < a) (b_pos : 0 < b) : Prop := 
  c = p / 2 ∧ c^2 = a^2 + b^2

-- Definition for points A and B on parabola C1 and point M on hyperbola C2.
def points_A_B_M (c a b : ℝ) (x1 y1 x2 y2 yM : ℝ) : Prop := 
  x1 = c ∧ y1 = 2 * c ∧ x2 = c ∧ y2 = -2 * c ∧ yM = b^2 / a

-- Condition for OM, OA, and OB relation and mn = 1/8.
def OM_OA_OB_relation (m n : ℝ) : Prop := 
  m * n = 1 / 8

-- Theorem statement: Given the conditions, the eccentricity of hyperbola C2 is √6 + √2 / 2.
theorem hyperbola_eccentricity (p a b c m n : ℝ) (p_pos : 0 < p) (a_pos : 0 < a) (b_pos : 0 < b) :
  parabola p p_pos c (2 * c) → 
  hyperbola a b a_pos b_pos c (b^2 / a) → 
  common_focus p a b c p_pos a_pos b_pos →
  points_A_B_M c a b c (2 * c) c (-2 * c) (b^2 / a) →
  OM_OA_OB_relation m n → 
  m * n = 1 / 8 →
  ∃ e : ℝ, e = (Real.sqrt 6 + Real.sqrt 2) / 2 :=
sorry

end hyperbola_eccentricity_l288_288932


namespace smaller_number_is_25_l288_288064

theorem smaller_number_is_25 (x y : ℕ) (h1 : x + y = 62) (h2 : y = x + 12) : x = 25 :=
by sorry

end smaller_number_is_25_l288_288064


namespace fifth_inequality_nth_inequality_solve_given_inequality_l288_288487

theorem fifth_inequality :
  ∀ x, 1 < x ∧ x < 2 → (x + 2 / x < 3) →
  ∀ x, 3 < x ∧ x < 4 → (x + 12 / x < 7) →
  ∀ x, 5 < x ∧ x < 6 → (x + 30 / x < 11) →
  (x + 90 / x < 19) := by
  sorry

theorem nth_inequality (n : ℕ) :
  ∀ x, (2 * n - 1 < x ∧ x < 2 * n) →
  (x + 2 * n * (2 * n - 1) / x < 4 * n - 1) := by
  sorry

theorem solve_given_inequality (a : ℕ) (x : ℝ) (h_a_pos: 0 < a) :
  x + 12 * a / (x + 1) < 4 * a + 2 →
  (2 < x ∧ x < 4 * a - 1) := by
  sorry

end fifth_inequality_nth_inequality_solve_given_inequality_l288_288487


namespace transform_polynomial_l288_288281

open Real

variable {x y : ℝ}

theorem transform_polynomial 
  (h1 : y = x + 1 / x) 
  (h2 : x^4 + x^3 - 4 * x^2 + x + 1 = 0) : 
  x^2 * (y^2 + y - 6) = 0 := 
sorry

end transform_polynomial_l288_288281


namespace city_cleaning_total_l288_288485

variable (A B C D : ℕ)

theorem city_cleaning_total : 
  A = 54 →
  A = B + 17 →
  C = 2 * B →
  D = A / 3 →
  A + B + C + D = 183 := 
by 
  intros hA hAB hC hD
  sorry

end city_cleaning_total_l288_288485


namespace complex_division_l288_288263

noncomputable def imagine_unit : ℂ := Complex.I

theorem complex_division :
  (Complex.mk (-3) 1) / (Complex.mk 1 (-1)) = (Complex.mk (-2) 1) :=
by
sorry

end complex_division_l288_288263


namespace final_number_after_operations_l288_288643

theorem final_number_after_operations:
  (∀ d_k : ℕ, 1 ≤ d_k → ∃ n : ℕ, n = 1 ∧
  (S = (∑ k in (range 1988), k) - ∑ k in (range 1987), (1989 - k) * d_k)
  ∧ ∀ k, (S ≥ 0)) :=
sorry

end final_number_after_operations_l288_288643


namespace eccentricity_of_hyperbola_l288_288333

theorem eccentricity_of_hyperbola :
  let a := Real.sqrt 5
  let b := 2
  let c := Real.sqrt (a^2 + b^2)
  let e := c / a
  (∃ (x y : ℝ), (x^2 / 5) - (y^2 / 4) = 1 ∧ e = (3 * Real.sqrt 5) / 5) := sorry

end eccentricity_of_hyperbola_l288_288333


namespace quadratic_transformation_l288_288791

theorem quadratic_transformation (a b c : ℝ) (h : a * x^2 + b * x + c = 5 * (x + 2)^2 - 7) :
  ∃ (n m g : ℝ), 2 * a * x^2 + 2 * b * x + 2 * c = n * (x - g)^2 + m ∧ g = -2 :=
by
  sorry

end quadratic_transformation_l288_288791


namespace find_f_m_eq_neg_one_l288_288731

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := x^(2 - m)

theorem find_f_m_eq_neg_one (m : ℝ)
  (h1 : ∀ x : ℝ, f x m = - f (-x) m) (h2 : m^2 - m = 3 + m) :
  f m m = -1 :=
by
  sorry

end find_f_m_eq_neg_one_l288_288731


namespace max_grandchildren_l288_288947

theorem max_grandchildren (children_count : ℕ) (common_gc : ℕ) (special_gc_count : ℕ) : 
  children_count = 8 ∧ common_gc = 8 ∧ special_gc_count = 5 →
  (6 * common_gc + 2 * special_gc_count) = 58 := by
  sorry

end max_grandchildren_l288_288947


namespace second_quarter_profit_l288_288413

theorem second_quarter_profit (q1 q3 q4 annual : ℕ) (h1 : q1 = 1500) (h2 : q3 = 3000) (h3 : q4 = 2000) (h4 : annual = 8000) :
  annual - (q1 + q3 + q4) = 1500 :=
by
  sorry

end second_quarter_profit_l288_288413


namespace Matilda_correct_age_l288_288930

def Louis_age : ℕ := 14
def Jerica_age : ℕ := 2 * Louis_age
def Matilda_age : ℕ := Jerica_age + 7

theorem Matilda_correct_age : Matilda_age = 35 :=
by
  -- Proof needs to be filled here
  sorry

end Matilda_correct_age_l288_288930


namespace basketball_children_l288_288512

/-- Given:
  1. total spectators is 10,000
  2. 7,000 of them were men
  3. Of the remaining spectators, there were 5 times as many children as women

Prove that the number of children was 2,500. -/
theorem basketball_children (total_spectators : ℕ) (men : ℕ) (women_children : ℕ) (women children : ℕ) 
  (h1 : total_spectators = 10000) 
  (h2 : men = 7000) 
  (h3 : women_children = total_spectators - men) 
  (h4 : women + 5 * women = women_children) 
  : children = 5 * 500 := 
  by 
  sorry

end basketball_children_l288_288512


namespace probability_truth_or_lie_l288_288832

axiom probability_truth : ℝ := 0.30
axiom probability_lie : ℝ := 0.20
axiom probability_both : ℝ := 0.10

theorem probability_truth_or_lie :
  probability_truth + probability_lie - probability_both = 0.40 :=
by sorry

end probability_truth_or_lie_l288_288832


namespace find_m_l288_288610

theorem find_m (x y m : ℝ) (h1 : x = 2) (h2 : y = 1) (h3 : x + m * y = 5) : m = 3 := 
by
  sorry

end find_m_l288_288610


namespace eighty_five_squared_l288_288108

theorem eighty_five_squared : 85^2 = 7225 := by
  sorry

end eighty_five_squared_l288_288108


namespace buyers_muffin_mix_l288_288835

variable (P C M CM: ℕ)

theorem buyers_muffin_mix
    (h_total: P = 100)
    (h_cake: C = 50)
    (h_both: CM = 17)
    (h_neither: P - (C + M - CM) = 27)
    : M = 73 :=
by sorry

end buyers_muffin_mix_l288_288835


namespace number_of_terms_in_arithmetic_sequence_l288_288745

noncomputable def arithmetic_sequence_terms (a d n : ℕ) : Prop :=
  let sum_first_three := 3 * a + 3 * d = 34
  let sum_last_three := 3 * a + 3 * (n - 1) * d = 146
  let sum_all := n * (2 * a + (n - 1) * d) / 2 = 390
  (sum_first_three ∧ sum_last_three ∧ sum_all) → n = 13

theorem number_of_terms_in_arithmetic_sequence (a d n : ℕ) : arithmetic_sequence_terms a d n → n = 13 := 
by
  sorry

end number_of_terms_in_arithmetic_sequence_l288_288745


namespace product_y_coordinates_l288_288042

theorem product_y_coordinates : 
  ∀ y : ℝ, (∀ P : ℝ × ℝ, P.1 = -1 ∧ (P.1 - 4)^2 + (P.2 - 3)^2 = 64 → P = (-1, y)) →
  ((3 + Real.sqrt 39) * (3 - Real.sqrt 39) = -30) :=
by
  intros y h
  sorry

end product_y_coordinates_l288_288042


namespace exchange_silver_cards_l288_288218

theorem exchange_silver_cards : 
  (∃ red gold silver : ℕ,
    (∀ (r g s : ℕ), ((2 * g = 5 * r) ∧ (g = r + s) ∧ (r = 3) ∧ (g = 3) → s = 7))) :=
by
  sorry

end exchange_silver_cards_l288_288218


namespace percentage_apples_basket_l288_288070

theorem percentage_apples_basket :
  let initial_apples := 10
  let initial_oranges := 5
  let added_oranges := 5
  let total_apples := initial_apples
  let total_oranges := initial_oranges + added_oranges
  let total_fruits := total_apples + total_oranges
  (total_apples / total_fruits) * 100 = 50 :=
by
  sorry

end percentage_apples_basket_l288_288070


namespace victoria_should_return_22_l288_288987

theorem victoria_should_return_22 :
  let initial_money := 50
  let pizza_cost_per_box := 12
  let pizzas_bought := 2
  let juice_cost_per_pack := 2
  let juices_bought := 2
  let total_spent := (pizza_cost_per_box * pizzas_bought) + (juice_cost_per_pack * juices_bought)
  let money_returned := initial_money - total_spent
  money_returned = 22 :=
by
  sorry

end victoria_should_return_22_l288_288987


namespace mean_transformation_l288_288724

variable {x1 x2 x3 : ℝ}
variable (s : ℝ)
variable (h_var : s^2 = (1 / 3) * (x1^2 + x2^2 + x3^2 - 12))

theorem mean_transformation :
  (x1 + 1 + x2 + 1 + x3 + 1) / 3 = 3 :=
by
  sorry

end mean_transformation_l288_288724


namespace zero_is_smallest_natural_number_l288_288340

theorem zero_is_smallest_natural_number : ∀ n : ℕ, 0 ≤ n :=
by
  intro n
  exact Nat.zero_le n

#check zero_is_smallest_natural_number  -- confirming the theorem check

end zero_is_smallest_natural_number_l288_288340


namespace lemonade_price_fraction_l288_288223

theorem lemonade_price_fraction :
  (2 / 5) * (L / S) = 0.35714285714285715 → L / S = 0.8928571428571429 :=
by
  intro h
  sorry

end lemonade_price_fraction_l288_288223


namespace cylinder_from_sector_l288_288681

noncomputable def circle_radius : ℝ := 12
noncomputable def sector_angle : ℝ := 300
noncomputable def arc_length (r : ℝ) (θ : ℝ) : ℝ := (θ / 360) * 2 * Real.pi * r

noncomputable def is_valid_cylinder (base_radius height : ℝ) : Prop :=
  2 * Real.pi * base_radius = arc_length circle_radius sector_angle ∧ height = circle_radius

theorem cylinder_from_sector :
  is_valid_cylinder 10 12 :=
by
  -- here, the proof will be provided
  sorry

end cylinder_from_sector_l288_288681


namespace system_of_equations_solution_l288_288329

theorem system_of_equations_solution (x y : ℝ) :
  (2 * x + y + 2 * x * y = 11 ∧ 2 * x^2 * y + x * y^2 = 15) ↔
  ((x = 1/2 ∧ y = 5) ∨ (x = 1 ∧ y = 3) ∨ (x = 3/2 ∧ y = 2) ∨ (x = 5/2 ∧ y = 1)) :=
by 
  sorry

end system_of_equations_solution_l288_288329


namespace statues_added_in_third_year_l288_288271

/-
Definition of the turtle statues problem:

1. Initially, there are 4 statues in the first year.
2. In the second year, the number of statues quadruples.
3. In the third year, x statues are added, and then 3 statues are broken.
4. In the fourth year, 2 * 3 new statues are added.
5. In total, at the end of the fourth year, there are 31 statues.
-/

def year1_statues : ℕ := 4
def year2_statues : ℕ := 4 * year1_statues
def before_hailstorm_year3_statues (x : ℕ) : ℕ := year2_statues + x
def after_hailstorm_year3_statues (x : ℕ) : ℕ := before_hailstorm_year3_statues x - 3
def total_year4_statues (x : ℕ) : ℕ := after_hailstorm_year3_statues x + 2 * 3

theorem statues_added_in_third_year (x : ℕ) (h : total_year4_statues x = 31) : x = 12 :=
by
  sorry

end statues_added_in_third_year_l288_288271


namespace smallest_a_b_sum_l288_288117

theorem smallest_a_b_sum :
∀ (a b : ℕ), 
  (5 * a + 6 = 6 * b + 5) ∧ 
  (∀ d : ℕ, d < 10 → d < a) ∧ 
  (∀ d : ℕ, d < 10 → d < b) ∧ 
  (0 < a) ∧ 
  (0 < b) 
  → a + b = 13 :=
by
  sorry

end smallest_a_b_sum_l288_288117


namespace intersection_of_A_and_B_l288_288629

-- Define the sets A and B
def A := {x : ℝ | x ≥ 1}
def B := {x : ℝ | -1 < x ∧ x < 2}

-- Define the expected intersection
def expected_intersection := {x : ℝ | 1 ≤ x ∧ x < 2}

-- The proof problem statement
theorem intersection_of_A_and_B :
  A ∩ B = expected_intersection := by
  sorry

end intersection_of_A_and_B_l288_288629


namespace cherry_tree_leaves_l288_288560

theorem cherry_tree_leaves (original_plan : ℕ) (multiplier : ℕ) (leaves_per_tree : ℕ) 
  (h1 : original_plan = 7) (h2 : multiplier = 2) (h3 : leaves_per_tree = 100) : 
  (original_plan * multiplier * leaves_per_tree = 1400) :=
by
  sorry

end cherry_tree_leaves_l288_288560


namespace parabola_properties_l288_288434

-- Definitions of the conditions
def parabola (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c
def point_A (a b c : ℝ) : Prop := parabola a b c (-1) = 0
def point_B (a b c m : ℝ) : Prop := parabola a b c m = 0
def opens_downwards (a : ℝ) : Prop := a < 0
def valid_m (m : ℝ) : Prop := 1 < m ∧ m < 2

-- Conclusion ①
def conclusion_1 (a b : ℝ) : Prop := b > 0

-- Conclusion ②
def conclusion_2 (a c : ℝ) : Prop := 3 * a + 2 * c < 0

-- Conclusion ③
def conclusion_3 (a b c x1 x2 y1 y2 : ℝ) : Prop :=
  x1 < x2 ∧ x1 + x2 > 1 ∧ parabola a b c x1 = y1 ∧ parabola a b c x2 = y2 → y1 > y2

-- Conclusion ④
def conclusion_4 (a b c : ℝ) : Prop :=
  a ≤ -1 → ∃ x1 x2 : ℝ, (a * x1^2 + b * x1 + c = 1) ∧ (a * x2^2 + b * x2 + c = 1) ∧ (x1 ≠ x2)

-- The theorem to prove
theorem parabola_properties (a b c m : ℝ) :
  (opens_downwards a) →
  (point_A a b c) →
  (point_B a b c m) →
  (valid_m m) →
  (conclusion_1 a b) ∧ (conclusion_2 a c → false) ∧ (∀ x1 x2 y1 y2, conclusion_3 a b c x1 x2 y1 y2) ∧ (conclusion_4 a b c) :=
by
  sorry

end parabola_properties_l288_288434


namespace haley_money_difference_l288_288603

def initial_amount : ℕ := 2
def chores : ℕ := 5
def birthday : ℕ := 10
def neighbor : ℕ := 7
def candy : ℕ := 3
def lost : ℕ := 2

theorem haley_money_difference : (initial_amount + chores + birthday + neighbor - candy - lost) - initial_amount = 17 := by
  sorry

end haley_money_difference_l288_288603


namespace valid_third_side_length_l288_288142

theorem valid_third_side_length (x : ℝ) : 4 < x ∧ x < 14 ↔ (((5 : ℝ) + 9 > x) ∧ (x + 5 > 9) ∧ (x + 9 > 5)) :=
by 
  sorry

end valid_third_side_length_l288_288142


namespace swimming_club_total_members_l288_288554

def valid_total_members (total : ℕ) : Prop :=
  ∃ (J S V : ℕ),
    3 * S = 2 * J ∧
    5 * V = 2 * S ∧
    total = J + S + V

theorem swimming_club_total_members :
  valid_total_members 58 := by
  sorry

end swimming_club_total_members_l288_288554


namespace fraction_simplification_l288_288854

theorem fraction_simplification (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ -1) : 
  (2 * x - 5) / (x ^ 2 - 1) + 3 / (1 - x) = - (x + 8) / (x ^ 2 - 1) :=
  sorry

end fraction_simplification_l288_288854


namespace volume_inequality_find_min_k_l288_288861

noncomputable def cone_volume (R h : ℝ) : ℝ := (1 / 3) * Real.pi * R^2 * h

noncomputable def cylinder_volume (R h : ℝ) : ℝ :=
    let r := (R * h) / Real.sqrt (R^2 + h^2)
    Real.pi * r^2 * h

noncomputable def k_value (R h : ℝ) : ℝ := (R^2 + h^2) / (3 * h^2)

theorem volume_inequality (R h : ℝ) (h_pos : R > 0 ∧ h > 0) : 
    cone_volume R h ≠ cylinder_volume R h := by sorry

theorem find_min_k (R h : ℝ) (h_pos : R > 0 ∧ h > 0) (k : ℝ) :
    cone_volume R h = k * cylinder_volume R h → k = (R^2 + h^2) / (3 * h^2) := by sorry

end volume_inequality_find_min_k_l288_288861


namespace book_transaction_difference_l288_288311

def number_of_books : ℕ := 15
def cost_per_book : ℕ := 11
def selling_price_per_book : ℕ := 25

theorem book_transaction_difference :
  number_of_books * selling_price_per_book - number_of_books * cost_per_book = 210 :=
by
  sorry

end book_transaction_difference_l288_288311


namespace price_of_cashew_nuts_l288_288545

theorem price_of_cashew_nuts 
  (C : ℝ)  -- price per kilo of cashew nuts
  (P_p : ℝ := 130)  -- price per kilo of peanuts
  (cashew_kilos : ℝ := 3)  -- kilos of cashew nuts bought
  (peanut_kilos : ℝ := 2)  -- kilos of peanuts bought
  (total_kilos : ℝ := 5)  -- total kilos of nuts bought
  (total_price_per_kilo : ℝ := 178)  -- total price per kilo of all nuts
  (h_total_cost : cashew_kilos * C + peanut_kilos * P_p = total_kilos * total_price_per_kilo) :
  C = 210 :=
sorry

end price_of_cashew_nuts_l288_288545


namespace f_neg_l288_288727

variable (f : ℝ → ℝ)

-- Given condition that f is an odd function
def odd_function (f : ℝ → ℝ) := ∀ x, f (-x) = -f x

-- The form of f for x ≥ 0
def f_pos (x : ℝ) (h : 0 ≤ x) : f x = -x^2 + 2 * x := sorry

-- Objective to prove f(x) for x < 0
theorem f_neg {x : ℝ} (h : x < 0) (hf_odd : odd_function f) (hf_pos : ∀ x, 0 ≤ x → f x = -x^2 + 2 * x) : f x = x^2 + 2 * x := 
by 
  sorry

end f_neg_l288_288727


namespace number_of_pupils_wrong_entry_l288_288700

theorem number_of_pupils_wrong_entry 
  (n : ℕ) (A : ℝ) 
  (h_wrong_entry : ∀ m, (m = 85 → n * (A + 1 / 2) = n * A + 52))
  (h_increase : ∀ m, (m = 33 → n * (A + 1 / 2) = n * A + 52)) 
  : n = 104 := 
sorry

end number_of_pupils_wrong_entry_l288_288700


namespace remainder_of_polynomial_l288_288524

theorem remainder_of_polynomial (x : ℕ) :
  (x + 1) ^ 2021 % (x ^ 2 + x + 1) = 1 + x ^ 2 := 
by
  sorry

end remainder_of_polynomial_l288_288524


namespace minute_hand_distance_l288_288055

noncomputable def distance_traveled (length_of_minute_hand : ℝ) (time_duration : ℝ) : ℝ :=
  let circumference := 2 * Real.pi * length_of_minute_hand
  let revolutions := time_duration / 60
  circumference * revolutions

theorem minute_hand_distance :
  distance_traveled 8 45 = 12 * Real.pi :=
by
  sorry

end minute_hand_distance_l288_288055


namespace min_value_l288_288901

theorem min_value (x y z : ℝ) (h : 2*x + 3*y + 4*z = 1) : 
  x^2 + y^2 + z^2 ≥ 1/29 :=
sorry

end min_value_l288_288901


namespace breadth_of_hall_l288_288407

/-- Given a hall of length 20 meters and a uniform verandah width of 2.5 meters,
    with a cost of Rs. 700 for flooring the verandah at Rs. 3.50 per square meter,
    prove that the breadth of the hall is 15 meters. -/
theorem breadth_of_hall (h_length : ℝ) (v_width : ℝ) (cost : ℝ) (rate : ℝ) (b : ℝ) :
  h_length = 20 ∧ v_width = 2.5 ∧ cost = 700 ∧ rate = 3.50 →
  25 * (b + 5) - 20 * b = 200 →
  b = 15 :=
by
  intros hc ha
  sorry

end breadth_of_hall_l288_288407


namespace angle_between_hands_at_seven_l288_288394

-- Define the conditions
def clock_parts := 12 -- The clock is divided into 12 parts
def degrees_per_part := 30 -- Each part is 30 degrees

-- Define the position of the hour and minute hands at 7:00 AM
def hour_position_at_seven := 7 -- Hour hand points to 7
def minute_position_at_seven := 0 -- Minute hand points to 12

-- Calculate the number of parts between the two positions
def parts_between_hands := if minute_position_at_seven = 0 then hour_position_at_seven else 12 - hour_position_at_seven

-- Calculate the angle between the hour hand and the minute hand at 7:00 AM
def angle_at_seven := degrees_per_part * parts_between_hands

-- State the theorem
theorem angle_between_hands_at_seven : angle_at_seven = 150 :=
by
  sorry

end angle_between_hands_at_seven_l288_288394


namespace solve_equation_1_solve_equation_2_l288_288649

theorem solve_equation_1 (x : ℝ) : x^2 - 3 * x = 4 ↔ x = 4 ∨ x = -1 :=
by
  sorry

theorem solve_equation_2 (x : ℝ) : x * (x - 2) + x - 2 = 0 ↔ x = 2 ∨ x = -1 :=
by
  sorry

end solve_equation_1_solve_equation_2_l288_288649


namespace proof_problem_l288_288443

def sum_of_sequence (S : ℕ → ℕ) (a : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, 0 < n → S n = n * (n + 1) + 2 ∧ S 1 = a 1 ∧ (∀ n, 1 < n → a n = S n - S (n - 1))

def general_term_a (a : ℕ → ℕ) : Prop :=
  a 1 = 4 ∧ (∀ n, 1 < n → a n = 2 * n)

def geometric_sequence (a : ℕ → ℕ) (b : ℕ → ℕ) : Prop :=
  ∀ k : ℕ, 0 < k → 
  a 2 = 4 ∧ a (k+2) = 2 * (k + 2) ∧ a (3 * k + 2) = 2 * (3 * k + 2) →
  b 1 = a 2 ∧ b 2 = a (k + 2) ∧ b 3 = a (3 * k + 2) ∧ 
  (∀ n, b n = 2^(n + 1))

theorem proof_problem :
  ∃ (a b S : ℕ → ℕ),
  sum_of_sequence S a ∧ general_term_a a ∧ geometric_sequence a b :=
sorry

end proof_problem_l288_288443


namespace intersection_of_A_and_B_l288_288631

-- Definitions of sets A and B
def A : Set ℝ := { x | x ≥ 1 }
def B : Set ℝ := { x | -1 < x ∧ x < 2 }

-- The theorem we want to prove
theorem intersection_of_A_and_B : A ∩ B = { x | 1 ≤ x ∧ x < 2 } :=
by
  sorry

end intersection_of_A_and_B_l288_288631


namespace factorize_xy_squared_minus_x_l288_288877

theorem factorize_xy_squared_minus_x (x y : ℝ) : xy^2 - x = x * (y - 1) * (y + 1) :=
  sorry

end factorize_xy_squared_minus_x_l288_288877


namespace sum_of_coordinates_D_l288_288958

theorem sum_of_coordinates_D (x y : ℝ) 
  (M_midpoint : (4, 10) = ((8 + x) / 2, (6 + y) / 2)) : 
  x + y = 14 := 
by 
  sorry

end sum_of_coordinates_D_l288_288958


namespace average_mileage_first_car_l288_288464

theorem average_mileage_first_car (X Y : ℝ) 
  (h1 : X + Y = 75) 
  (h2 : 25 * X + 35 * Y = 2275) : 
  X = 35 :=
by 
  sorry

end average_mileage_first_car_l288_288464


namespace travel_distance_of_wheel_l288_288231

theorem travel_distance_of_wheel (r : ℝ) (revolutions : ℕ) (h_r : r = 2) (h_revolutions : revolutions = 2) : 
    ∃ d : ℝ, d = 8 * Real.pi :=
by
  sorry

end travel_distance_of_wheel_l288_288231


namespace combined_population_l288_288972

theorem combined_population (W PP LH : ℕ) 
  (hW : W = 900)
  (hPP : PP = 7 * W)
  (hLH : PP = LH + 800) : 
  (PP + LH) = 11800 :=
by
  sorry

end combined_population_l288_288972


namespace smallest_positive_perfect_square_divisible_by_2_3_5_l288_288367

theorem smallest_positive_perfect_square_divisible_by_2_3_5 : 
  ∃ n : ℕ, (n > 0 ∧ n % 2 = 0 ∧ n % 3 = 0 ∧ n % 5 = 0) ∧ ∃ k : ℕ, n = k^2 ∧ n = 225 := 
begin
  sorry
end

end smallest_positive_perfect_square_divisible_by_2_3_5_l288_288367


namespace ashton_pencils_left_l288_288105

def pencils_left (boxes : Nat) (pencils_per_box : Nat) (pencils_given : Nat) : Nat :=
  boxes * pencils_per_box - pencils_given

theorem ashton_pencils_left :
  pencils_left 2 14 6 = 22 :=
by
  sorry

end ashton_pencils_left_l288_288105


namespace polynomial_irreducible_if_not_divisible_by_5_l288_288481

theorem polynomial_irreducible_if_not_divisible_by_5 (k : ℤ) (h1 : ¬ ∃ m : ℤ, k = 5 * m) :
    ¬ ∃ (f g : Polynomial ℤ), (f.degree < 5) ∧ (f * g = x^5 - x + Polynomial.C k) :=
  sorry

end polynomial_irreducible_if_not_divisible_by_5_l288_288481


namespace smallest_perfect_square_divisible_by_2_3_5_l288_288372

theorem smallest_perfect_square_divisible_by_2_3_5 :
  ∃ n : ℕ, (n > 0) ∧ (nat.sqrt n * nat.sqrt n = n) ∧ (2 ∣ n) ∧ (3 ∣ n) ∧ (5 ∣ n) ∧ n = 900 := sorry

end smallest_perfect_square_divisible_by_2_3_5_l288_288372


namespace extreme_value_and_inequality_l288_288730

theorem extreme_value_and_inequality
  (f : ℝ → ℝ)
  (a c : ℝ)
  (h_odd : ∀ x : ℝ, f (-x) = -f x)
  (h_extreme : f 1 = -2)
  (h_f_def : ∀ x : ℝ, f x = a * x^3 + c * x)
  (h_a_c : a = 1 ∧ c = -3) :
  (∀ x : ℝ, x < -1 → deriv f x > 0) ∧
  (∀ x : ℝ, -1 < x ∧ x < 1 → deriv f x < 0) ∧
  (∀ x : ℝ, 1 < x → deriv f x > 0) ∧
  f (-1) = 2 ∧
  (∀ x₁ x₂ : ℝ, -1 < x₁ ∧ x₁ < 1 ∧ -1 < x₂ ∧ x₂ < 1 → |f x₁ - f x₂| < 4) :=
by sorry

end extreme_value_and_inequality_l288_288730


namespace smallest_perfect_square_divisible_by_2_3_5_l288_288359

theorem smallest_perfect_square_divisible_by_2_3_5 : 
  ∃ n : ℕ, n > 0 ∧ (∃ m : ℕ, n = m * m) ∧ (2 ∣ n) ∧ (3 ∣ n) ∧ (5 ∣ n) ∧
         (∀ k : ℕ, 
           (k > 0 ∧ (∃ l : ℕ, k = l * l) ∧ (2 ∣ k) ∧ (3 ∣ k) ∧ (5 ∣ k)) → n ≤ k) :=
sorry

end smallest_perfect_square_divisible_by_2_3_5_l288_288359


namespace daughter_current_age_l288_288414

-- Define the conditions
def mother_current_age := 42
def years_later := 9
def mother_age_in_9_years := mother_current_age + years_later
def daughter_age_in_9_years (D : ℕ) := D + years_later

-- Define the statement we need to prove
theorem daughter_current_age : ∃ D : ℕ, mother_age_in_9_years = 3 * daughter_age_in_9_years D ∧ D = 8 :=
by {
  sorry
}

end daughter_current_age_l288_288414


namespace Nina_can_buy_8_widgets_at_reduced_cost_l288_288780

def money_Nina_has : ℕ := 48
def widgets_she_can_buy_initially : ℕ := 6
def reduction_per_widget : ℕ := 2

theorem Nina_can_buy_8_widgets_at_reduced_cost :
  let initial_cost_per_widget := money_Nina_has / widgets_she_can_buy_initially
  let reduced_cost_per_widget := initial_cost_per_widget - reduction_per_widget
  money_Nina_has / reduced_cost_per_widget = 8 :=
by
  sorry

end Nina_can_buy_8_widgets_at_reduced_cost_l288_288780


namespace find_third_side_of_triangle_l288_288202

theorem find_third_side_of_triangle (a b : ℝ) (A : ℝ) (h1 : a = 6) (h2 : b = 10) (h3 : A = 18) (h4 : ∃ C, 0 < C ∧ C < π / 2 ∧ A = 0.5 * a * b * Real.sin C) : 
  ∃ c : ℝ, c = 2 * Real.sqrt 22 :=
by
  sorry

end find_third_side_of_triangle_l288_288202


namespace Marty_combinations_l288_288776

def unique_combinations (colors techniques : ℕ) : ℕ :=
  colors * techniques

theorem Marty_combinations :
  unique_combinations 6 5 = 30 := by
  sorry

end Marty_combinations_l288_288776


namespace correct_operation_l288_288209

variable (a b : ℝ)

theorem correct_operation :
  -a^6 / a^3 = -a^3 := by
  sorry

end correct_operation_l288_288209


namespace triangle_angles_l288_288924

theorem triangle_angles (second_angle first_angle third_angle : ℝ) 
  (h1 : first_angle = 2 * second_angle)
  (h2 : third_angle = second_angle + 30)
  (h3 : second_angle + first_angle + third_angle = 180) :
  second_angle = 37.5 ∧ first_angle = 75 ∧ third_angle = 67.5 :=
sorry

end triangle_angles_l288_288924


namespace intersection_is_correct_l288_288269

def A : Set ℤ := {0, 3, 4}
def B : Set ℤ := {-1, 0, 2, 3}

theorem intersection_is_correct : A ∩ B = {0, 3} := by
  sorry

end intersection_is_correct_l288_288269


namespace reading_comprehension_application_method_1_application_method_2_l288_288180

-- Reading Comprehension Problem in Lean 4
theorem reading_comprehension (x : ℝ) (h : x^2 + x + 5 = 8) : 2 * x^2 + 2 * x - 4 = 2 :=
by sorry

-- Application of Methods Problem (1) in Lean 4
theorem application_method_1 (x : ℝ) (h : x^2 + x + 2 = 9) : -2 * x^2 - 2 * x + 3 = -11 :=
by sorry

-- Application of Methods Problem (2) in Lean 4
theorem application_method_2 (a b : ℝ) (h : 8 * a + 2 * b = 5) : a * (-2)^3 + b * (-2) + 3 = -2 :=
by sorry

end reading_comprehension_application_method_1_application_method_2_l288_288180


namespace vertical_lines_count_l288_288752

theorem vertical_lines_count (n : ℕ) 
  (h_intersections : (18 * n * (n - 1)) = 756) : 
  n = 7 :=
by 
  sorry

end vertical_lines_count_l288_288752


namespace luke_fish_fillets_l288_288579

theorem luke_fish_fillets : 
  (∃ (catch_rate : ℕ) (days : ℕ) (fillets_per_fish : ℕ), catch_rate = 2 ∧ days = 30 ∧ fillets_per_fish = 2 → 
  (catch_rate * days * fillets_per_fish = 120)) :=
by
  sorry

end luke_fish_fillets_l288_288579


namespace simplify_expression_l288_288047

noncomputable def term1 : ℝ := 3 / (Real.sqrt 2 + 2)
noncomputable def term2 : ℝ := 4 / (Real.sqrt 5 - 2)
noncomputable def simplifiedExpression : ℝ := 1 / (term1 + term2)
noncomputable def finalExpression : ℝ := 1 / (11 + 4 * Real.sqrt 5 - 3 * Real.sqrt 2 / 2)

theorem simplify_expression : simplifiedExpression = finalExpression := by
  sorry

end simplify_expression_l288_288047


namespace inscribed_triangle_perimeter_geq_half_l288_288318

theorem inscribed_triangle_perimeter_geq_half (a : ℝ) (s' : ℝ) (h_a_pos : a > 0) 
  (h_equilateral : ∀ (A B C : Type) (a b c : A), a = b ∧ b = c ∧ c = a) :
  2 * s' >= (3 * a) / 2 :=
by
  sorry

end inscribed_triangle_perimeter_geq_half_l288_288318


namespace electrical_appliance_supermarket_l288_288542

-- Define the known quantities and conditions
def purchase_price_A : ℝ := 140
def purchase_price_B : ℝ := 100
def week1_sales_A : ℕ := 4
def week1_sales_B : ℕ := 3
def week1_revenue : ℝ := 1250
def week2_sales_A : ℕ := 5
def week2_sales_B : ℕ := 5
def week2_revenue : ℝ := 1750
def total_units : ℕ := 50
def budget : ℝ := 6500
def profit_goal : ℝ := 2850

-- Define the unknown selling prices
noncomputable def selling_price_A : ℝ := 200
noncomputable def selling_price_B : ℝ := 150

-- Define the constraints
def cost_constraint (m : ℕ) : Prop := 140 * m + 100 * (50 - m) ≤ 6500
def profit_exceeds_goal (m : ℕ) : Prop := (200 - 140) * m + (150 - 100) * (50 - m) > 2850

-- The main theorem stating the results
theorem electrical_appliance_supermarket :
  (4 * selling_price_A + 3 * selling_price_B = week1_revenue)
  ∧ (5 * selling_price_A + 5 * selling_price_B = week2_revenue)
  ∧ (∃ m : ℕ, m ≤ 37 ∧ cost_constraint m)
  ∧ (∃ m : ℕ, m > 35 ∧ m ≤ 37 ∧ profit_exceeds_goal m) :=
sorry

end electrical_appliance_supermarket_l288_288542


namespace notable_features_points_l288_288315

namespace Points3D

def is_first_octant (x y z : ℝ) : Prop := (x > 0) ∧ (y > 0) ∧ (z > 0)
def is_second_octant (x y z : ℝ) : Prop := (x < 0) ∧ (y > 0) ∧ (z > 0)
def is_eighth_octant (x y z : ℝ) : Prop := (x > 0) ∧ (y < 0) ∧ (z < 0)
def lies_in_YOZ_plane (x y z : ℝ) : Prop := (x = 0) ∧ (y ≠ 0) ∧ (z ≠ 0)
def lies_on_OY_axis (x y z : ℝ) : Prop := (x = 0) ∧ (y ≠ 0) ∧ (z = 0)
def is_origin (x y z : ℝ) : Prop := (x = 0) ∧ (y = 0) ∧ (z = 0)

theorem notable_features_points :
  is_first_octant 3 2 6 ∧
  is_second_octant (-2) 3 1 ∧
  is_eighth_octant 1 (-4) (-2) ∧
  is_eighth_octant 1 (-2) (-1) ∧
  lies_in_YOZ_plane 0 4 1 ∧
  lies_on_OY_axis 0 2 0 ∧
  is_origin 0 0 0 :=
by
  sorry

end Points3D

end notable_features_points_l288_288315


namespace initial_books_eq_41_l288_288541

-- Definitions and conditions
def books_sold : ℕ := 33
def books_added : ℕ := 2
def books_remaining : ℕ := 10

-- Proof problem
theorem initial_books_eq_41 (B : ℕ) (h : B - books_sold + books_added = books_remaining) : B = 41 :=
by
  sorry

end initial_books_eq_41_l288_288541


namespace no_prime_satisfies_polynomial_l288_288651

theorem no_prime_satisfies_polynomial :
  ∀ p : ℕ, p.Prime → p^3 - 6*p^2 - 3*p + 14 ≠ 0 := by
  sorry

end no_prime_satisfies_polynomial_l288_288651


namespace molecular_weight_of_NH4Cl_l288_288357

theorem molecular_weight_of_NH4Cl (weight_8_moles : ℕ) (weight_per_mole : ℕ) :
  weight_8_moles = 424 →
  weight_per_mole = 53 →
  weight_8_moles / 8 = weight_per_mole :=
by
  intro h1 h2
  sorry

end molecular_weight_of_NH4Cl_l288_288357


namespace candies_count_l288_288049

theorem candies_count :
  ∃ n, (n = 35 ∧ ∃ x, x ≥ 11 ∧ n = 3 * (x - 1) + 2) ∧ ∃ y, y ≤ 9 ∧ n = 4 * (y - 1) + 3 :=
  by {
    sorry
  }

end candies_count_l288_288049


namespace boat_speed_in_still_water_l288_288540

theorem boat_speed_in_still_water (V_b : ℝ) : 
    (∀ (stream_speed : ℝ) (travel_time : ℝ) (distance : ℝ), 
        stream_speed = 5 ∧ 
        travel_time = 5 ∧ 
        distance = 105 →
        distance = (V_b + stream_speed) * travel_time) → 
    V_b = 16 := 
by 
    intro h
    specialize h 5 5 105 
    have h1 : 105 = (V_b + 5) * 5 := h ⟨rfl, ⟨rfl, rfl⟩⟩
    sorry

end boat_speed_in_still_water_l288_288540


namespace product_divisible_by_4_l288_288661

theorem product_divisible_by_4 (a b c d : ℤ) 
    (h : a^2 + b^2 + c^2 = d^2) : 4 ∣ (a * b * c) :=
sorry

end product_divisible_by_4_l288_288661


namespace factorization_l288_288874

theorem factorization (x y : ℝ) : (x * y^2 - x = x * (y - 1) * (y + 1)) :=
begin
  sorry
end

end factorization_l288_288874


namespace kerosene_cost_l288_288212

/-- In a market, a dozen eggs cost as much as a pound of rice, and a half-liter of kerosene 
costs as much as 8 eggs. If the cost of each pound of rice is $0.33, then a liter of kerosene costs 44 cents. --/
theorem kerosene_cost : 
  let egg_cost := 0.33 / 12
  let rice_cost := 0.33
  let half_liter_kerosene_cost := 8 * egg_cost
  rice_cost = 0.33 → 1 * ((2 * half_liter_kerosene_cost) * 100) = 44 := 
by
  intros egg_cost rice_cost half_liter_kerosene_cost h_rice_cost
  sorry

end kerosene_cost_l288_288212


namespace problem_statement_l288_288334

noncomputable def roots (a b c : ℝ) : ℝ × ℝ := 
let Δ := b^2 - 4*a*c in
if Δ < 0 then (0, 0) else ((-b + Real.sqrt Δ) / (2*a), (-b - Real.sqrt Δ) / (2*a))

theorem problem_statement : 
  let (x1, x2) := roots 1 5 1 in
  (x1^2 + 5*x1 + 1 = 0 ∧ x2^2 + 5*x2 + 1 = 0) →
  (let expr := (x1 * Real.sqrt 6 / (1 + x2))^2 + (x2 * Real.sqrt 6 / (1 + x1))^2 
  in expr = 220) := 
by {
  sorry
}

end problem_statement_l288_288334


namespace new_number_formed_l288_288614

variable (a b : ℕ)

theorem new_number_formed (ha : a < 10) (hb : b < 10) : 
  ((10 * a + b) * 10 + 2) = 100 * a + 10 * b + 2 := 
by
  sorry

end new_number_formed_l288_288614


namespace g_triple_composition_l288_288115

def g (n : ℕ) : ℕ :=
if n < 5 then n^2 + 1 else 2 * n + 3

theorem g_triple_composition : g (g (g 3)) = 49 :=
by
  sorry

end g_triple_composition_l288_288115


namespace smallest_positive_perfect_square_divisible_by_2_3_5_l288_288382

/-- Problem statement: 
    The smallest positive perfect square that is divisible by 2, 3, and 5 is 900.
-/
theorem smallest_positive_perfect_square_divisible_by_2_3_5 : 
  ∃ n : ℕ, 0 < n ∧ n * n % 2 = 0 ∧ n * n % 3 = 0 ∧ n * n % 5 = 0 ∧ n * n = 900 := 
by
  sorry

end smallest_positive_perfect_square_divisible_by_2_3_5_l288_288382


namespace smallest_int_x_l288_288810

theorem smallest_int_x (x : ℤ) (h : 2 * x + 5 < 3 * x - 10) : x = 16 :=
sorry

end smallest_int_x_l288_288810


namespace tickets_distribution_correct_l288_288418

def tickets_distribution (tickets programs : nat) (A_tickets_min : nat) : nat :=
sorry

theorem tickets_distribution_correct :
  tickets_distribution 6 4 3 = 17 :=
by
  sorry

end tickets_distribution_correct_l288_288418


namespace other_root_l288_288954

theorem other_root (m : ℝ) :
  ∃ r, (r = -7 / 3) ∧ (3 * 1 ^ 2 + m * 1 - 7 = 0) := 
begin
  use -7 / 3,
  split,
  { refl },
  { linarith }
end

end other_root_l288_288954


namespace luke_fish_fillets_l288_288580

theorem luke_fish_fillets : 
  (∃ (catch_rate : ℕ) (days : ℕ) (fillets_per_fish : ℕ), catch_rate = 2 ∧ days = 30 ∧ fillets_per_fish = 2 → 
  (catch_rate * days * fillets_per_fish = 120)) :=
by
  sorry

end luke_fish_fillets_l288_288580


namespace find_s_l288_288031

theorem find_s (n r s c d : ℝ) (h1 : c^2 - n * c + 3 = 0) (h2 : d^2 - n * d + 3 = 0) 
  (h3 : (c + 1/d)^2 - r * (c + 1/d) + s = 0) (h4 : (d + 1/c)^2 - r * (d + 1/c) + s = 0) 
  (h5 : c * d = 3) : s = 16 / 3 := 
by
  sorry

end find_s_l288_288031


namespace non_empty_subsets_count_l288_288273

open Finset

theorem non_empty_subsets_count :
  let S := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
  (∃ k, 1 ≤ k ∧
    ∀ (x : ℕ), x ∈ S → ¬(x + 1 ∈ S)) ∧
    ∀ (S_sub : Finset ℕ), S_sub.card = k →
    ∀ x ∈ S_sub, k ≤ x →
  S.count = 143 := by
  sorry

end non_empty_subsets_count_l288_288273


namespace modulo_remainder_l288_288124

theorem modulo_remainder :
  (7 * 10^24 + 2^24) % 13 = 8 := 
by
  sorry

end modulo_remainder_l288_288124


namespace calculation_result_l288_288203

theorem calculation_result :
  let a := 0.0088
  let b := 4.5
  let c := 0.05
  let d := 0.1
  let e := 0.008
  (a * b) / (c * d * e) = 990 :=
by
  sorry

end calculation_result_l288_288203


namespace drone_height_l288_288465

theorem drone_height (TR TS TU : ℝ) (UR : TU^2 + TR^2 = 180^2) (US : TU^2 + TS^2 = 150^2) (RS : TR^2 + TS^2 = 160^2) : 
  TU = Real.sqrt 14650 :=
by
  sorry

end drone_height_l288_288465


namespace find_m_l288_288767

def U : Set ℕ := {0, 1, 2, 3}

def A (m : ℝ) : Set ℕ := {x ∈ U | x^2 + m * x = 0}

def C_UA : Set ℕ := {1, 2}

theorem find_m (m : ℝ) (hA : A m = {0, 3}) (hCUA : U \ A m = C_UA) : m = -3 := 
  sorry

end find_m_l288_288767


namespace Kristyna_number_l288_288027

theorem Kristyna_number (k n : ℕ) (h1 : k = 6 * n + 3) (h2 : 3 * n + 1 + 2 * n = 1681) : k = 2019 := 
by
  -- Proof goes here
  sorry

end Kristyna_number_l288_288027


namespace find_X_l288_288157

variable (X : ℝ)  -- Threshold income level for the lower tax rate
variable (I : ℝ)  -- Income of the citizen
variable (T : ℝ)  -- Total tax amount

-- Conditions
def income : Prop := I = 50000
def tax_amount : Prop := T = 8000
def tax_formula : Prop := T = 0.15 * X + 0.20 * (I - X)

theorem find_X (h1 : income I) (h2 : tax_amount T) (h3 : tax_formula T I X) : X = 40000 :=
by
  sorry

end find_X_l288_288157


namespace inequality_l288_288907

noncomputable def f (x : ℝ) : ℝ := real.exp (-(x - 1)^2)

def a : ℝ := f (real.sqrt 2 / 2)
def b : ℝ := f (real.sqrt 3 / 2)
def c : ℝ := f (real.sqrt 6 / 2)

theorem inequality : b > c ∧ c > a := by
  sorry

end inequality_l288_288907


namespace evaluate_absolute_value_l288_288716

theorem evaluate_absolute_value (π : ℝ) (h : π < 5.5) : |5.5 - π| = 5.5 - π :=
by
  sorry

end evaluate_absolute_value_l288_288716


namespace larger_model_ratio_smaller_model_ratio_l288_288195

-- Definitions for conditions
def statue_height := 305 -- The height of the actual statue in feet
def larger_model_height := 10 -- The height of the larger model in inches
def smaller_model_height := 5 -- The height of the smaller model in inches

-- The ratio calculation for larger model
theorem larger_model_ratio : 
  (statue_height : ℝ) / (larger_model_height : ℝ) = 30.5 := by
  sorry

-- The ratio calculation for smaller model
theorem smaller_model_ratio : 
  (statue_height : ℝ) / (smaller_model_height : ℝ) = 61 := by
  sorry

end larger_model_ratio_smaller_model_ratio_l288_288195


namespace more_pie_eaten_l288_288566

theorem more_pie_eaten (erik_pie : ℝ) (frank_pie : ℝ)
  (h_erik : erik_pie = 0.6666666666666666)
  (h_frank : frank_pie = 0.3333333333333333) :
  erik_pie - frank_pie = 0.3333333333333333 :=
by
  sorry

end more_pie_eaten_l288_288566


namespace rectangle_perimeter_l288_288533

theorem rectangle_perimeter (L W : ℝ) 
  (h1 : L - 4 = W + 3) 
  (h2 : (L - 4) * (W + 3) = L * W) : 
  2 * L + 2 * W = 50 :=
by
  -- Proving the theorem here
  sorry

end rectangle_perimeter_l288_288533


namespace consecutive_integers_average_and_product_l288_288671

theorem consecutive_integers_average_and_product (n m : ℤ) (hnm : n ≤ m) 
  (h1 : (n + m) / 2 = 20) 
  (h2 : n * m = 391) :  m - n + 1 = 7 :=
  sorry

end consecutive_integers_average_and_product_l288_288671


namespace f_96_value_l288_288589

noncomputable def f : ℕ → ℕ :=
sorry

axiom condition_1 (a b : ℕ) : 
  f (a * b) = f a + f b

axiom condition_2 (n : ℕ) (hp : Nat.Prime n) (hlt : 10 < n) : 
  f n = 0

axiom condition_3 : 
  f 1 < f 243 ∧ f 243 < f 2 ∧ f 2 < 11

axiom condition_4 : 
  f 2106 < 11

theorem f_96_value :
  f 96 = 31 :=
sorry

end f_96_value_l288_288589


namespace correct_quotient_division_l288_288274

variable (k : Nat) -- the unknown original number

def mistaken_division := k = 7 * 12 + 4

theorem correct_quotient_division (h : mistaken_division k) : 
  (k / 3) = 29 :=
by
  sorry

end correct_quotient_division_l288_288274


namespace smallest_perfect_square_divisible_by_2_3_5_l288_288369

theorem smallest_perfect_square_divisible_by_2_3_5 :
  ∃ n : ℕ, (n > 0) ∧ (nat.sqrt n * nat.sqrt n = n) ∧ (2 ∣ n) ∧ (3 ∣ n) ∧ (5 ∣ n) ∧ n = 900 := sorry

end smallest_perfect_square_divisible_by_2_3_5_l288_288369


namespace team_points_difference_l288_288020

   -- Definitions for points of each member
   def Max_points : ℝ := 7
   def Dulce_points : ℝ := 5
   def Val_points : ℝ := 4 * (Max_points + Dulce_points)
   def Sarah_points : ℝ := 2 * Dulce_points
   def Steve_points : ℝ := 2.5 * (Max_points + Val_points)

   -- Definition for total points of their team
   def their_team_points : ℝ := Max_points + Dulce_points + Val_points + Sarah_points + Steve_points

   -- Definition for total points of the opponents' team
   def opponents_team_points : ℝ := 200

   -- The main theorem to prove
   theorem team_points_difference : their_team_points - opponents_team_points = 7.5 := by
     sorry
   
end team_points_difference_l288_288020


namespace leaves_fall_l288_288558

theorem leaves_fall (planned_trees : ℕ) (tree_multiplier : ℕ) (leaves_per_tree : ℕ) (h1 : planned_trees = 7) (h2 : tree_multiplier = 2) (h3 : leaves_per_tree = 100) :
  (planned_trees * tree_multiplier) * leaves_per_tree = 1400 :=
by
  rw [h1, h2, h3]
  -- Additional step suggestions for interactive proof environments, e.g.,
  -- Have: 7 * 2 = 14
  -- Goal: 14 * 100 = 1400
  sorry

end leaves_fall_l288_288558


namespace expected_sides_rectangle_expected_sides_polygon_l288_288807

-- Part (a)
theorem expected_sides_rectangle (k : ℕ) (h : k > 0) : (4 + 4 * k) / (k + 1) → 4 :=
by sorry

-- Part (b)
theorem expected_sides_polygon (n k : ℕ) (h : n > 2) (h_k : k ≥ 0) : (n + 4 * k) / (k + 1) = (n + 4 * k) / (k + 1) :=
by sorry

end expected_sides_rectangle_expected_sides_polygon_l288_288807


namespace number_of_men_l288_288347

theorem number_of_men (M W : ℕ) (h1 : W = 2) (h2 : ∃k, k = 4) : M = 4 :=
by
  sorry

end number_of_men_l288_288347


namespace tangent_circle_l288_288000

noncomputable theory

-- Definitions for the geometric setup
variables {A B C D E F G H : Point}

-- Conditions from the problem
def cyclic_quadrilateral (A B C D : Point) : Prop := ∃ c : Circle, A ∈ c ∧ B ∈ c ∧ C ∈ c ∧ D ∈ c
def intersection_of_diagonals (AC BD E : Point) : Prop := line_through A C ∩ line_through B D = {E}
def intersection_of_lines (AD BC F : Point) : Prop := line_through A D ∩ line_through B C = {F}
def is_midpoint (G A B : Point) : Prop := dist A G = dist G B ∧ collinear A G B
def midpoint_AB (G : Point) : Prop := is_midpoint G A B
def midpoint_CD (H : Point) : Prop := is_midpoint H C D

-- Final goal to prove in Lean
theorem tangent_circle (cyclic_quad : cyclic_quadrilateral A B C D)
  (inter_diag : intersection_of_diagonals A C E)
  (inter_lines : intersection_of_lines A D F)
  (mid_ab : midpoint_AB G)
  (mid_cd : midpoint_CD H) :
  tangent_line_at EF E (circumcircle E G H) :=
begin
  sorry -- proof goes here
end

end tangent_circle_l288_288000


namespace Ed_cats_l288_288582

variable (C F : ℕ)

theorem Ed_cats 
  (h1 : F = 2 * (C + 2))
  (h2 : 2 + C + F = 15) : 
  C = 3 := by 
  sorry

end Ed_cats_l288_288582


namespace fraction_doubled_l288_288462

theorem fraction_doubled (x y : ℝ) (h_nonzero : x + y ≠ 0) : (4 * x^2) / (2 * (x + y)) = 2 * (x^2 / (x + y)) :=
by
  sorry

end fraction_doubled_l288_288462


namespace rainfall_second_week_l288_288683

theorem rainfall_second_week (x : ℝ) 
  (h1 : x + 1.5 * x = 25) :
  1.5 * x = 15 :=
by
  sorry

end rainfall_second_week_l288_288683


namespace complex_sum_zero_l288_288935

noncomputable def complexSum {ω : ℂ} (h1 : ω^5 = 1) (h2 : ω ≠ 1) : ℂ :=
  ω^(15) + ω^(18) + ω^(21) + ω^(24) + ω^(27) + ω^(30) +
  ω^(33) + ω^(36) + ω^(39) + ω^(42) + ω^(45)

theorem complex_sum_zero {ω : ℂ} (h1 : ω^5 = 1) (h2 : ω ≠ 1) : complexSum h1 h2 = 0 :=
by
  sorry

end complex_sum_zero_l288_288935


namespace probability_none_needs_attention_probability_at_least_one_needs_attention_l288_288415

open Probability

-- Define the probabilities of needing attention
def P_A : ℝ := 0.9
def P_B : ℝ := 0.8
def P_C : ℝ := 0.85

-- Given these probabilities, we aim to prove the following:
theorem probability_none_needs_attention :
  let P_not_A := 1 - P_A in
  let P_not_B := 1 - P_B in
  let P_not_C := 1 - P_C in
  P_not_A * P_not_B * P_not_C = 0.003 :=
  by
    sorry

theorem probability_at_least_one_needs_attention :
  let P_not_A := 1 - P_A in
  let P_not_B := 1 - P_B in
  let P_not_C := 1 - P_C in
  1 - (P_not_A * P_not_B * P_not_C) = 0.997 :=
  by
    sorry

end probability_none_needs_attention_probability_at_least_one_needs_attention_l288_288415


namespace largest_value_of_c_l288_288452

noncomputable def g (x : ℝ) : ℝ := x^2 + 3 * x + 1

theorem largest_value_of_c :
  ∃ (c : ℝ), (∀ (x d : ℝ), |x - 1| ≤ d ∧ 0 < d ∧ 0 < c → |g x - 1| ≤ c) ∧ (∀ (c' : ℝ), (∀ (x d : ℝ), |x - 1| ≤ d ∧ 0 < d ∧ 0 < c' → |g x - 1| ≤ c') → c' ≤ c) :=
sorry

end largest_value_of_c_l288_288452


namespace volume_correctness_l288_288125

noncomputable def volume_of_regular_triangular_pyramid (d : ℝ) : ℝ :=
  1/3 * d^2 * d * Real.sqrt 2

theorem volume_correctness (d : ℝ) : 
  volume_of_regular_triangular_pyramid d = 1/3 * d^3 * Real.sqrt 2 :=
by
  sorry

end volume_correctness_l288_288125


namespace remainder_zero_by_68_l288_288312

theorem remainder_zero_by_68 (N R1 Q2 : ℕ) (h1 : N = 68 * 269 + R1) (h2 : N % 67 = 1) : R1 = 0 := by
  sorry

end remainder_zero_by_68_l288_288312


namespace shells_picked_in_morning_l288_288640

-- Definitions based on conditions
def total_shells : ℕ := 616
def afternoon_shells : ℕ := 324

-- The goal is to prove that morning_shells = 292
theorem shells_picked_in_morning (morning_shells : ℕ) (h : total_shells = morning_shells + afternoon_shells) : morning_shells = 292 := 
by
  sorry

end shells_picked_in_morning_l288_288640


namespace rearrange_possible_l288_288167

theorem rearrange_possible (n : ℕ) (h : n = 25 ∨ n = 1000) :
  ∃ (f : ℕ → ℕ), (∀ i < n, f i + 1 < n → (f (i + 1) - f i = 3 ∨ f (i + 1) - f i = 5)) :=
  sorry

end rearrange_possible_l288_288167


namespace weight_of_second_new_player_l288_288981

theorem weight_of_second_new_player
  (number_of_original_players : ℕ)
  (average_weight_of_original_players : ℝ)
  (weight_of_first_new_player : ℝ)
  (new_average_weight : ℝ)
  (total_number_of_players : ℕ)
  (total_weight_of_9_players : ℝ)
  (combined_weight_of_original_and_first_new : ℝ)
  (weight_of_second_new_player : ℝ)
  (h1 : number_of_original_players = 7)
  (h2 : average_weight_of_original_players = 103)
  (h3 : weight_of_first_new_player = 110)
  (h4 : new_average_weight = 99)
  (h5 : total_number_of_players = 9)
  (h6 : total_weight_of_9_players = total_number_of_players * new_average_weight)
  (h7 : combined_weight_of_original_and_first_new = number_of_original_players * average_weight_of_original_players + weight_of_first_new_player)
  (h8 : total_weight_of_9_players - combined_weight_of_original_and_first_new = weight_of_second_new_player) :
  weight_of_second_new_player = 60 :=
by
  sorry

end weight_of_second_new_player_l288_288981


namespace largest_value_l288_288078

-- Define the five expressions as given in the conditions
def exprA : ℕ := 3 + 1 + 2 + 8
def exprB : ℕ := 3 * 1 + 2 + 8
def exprC : ℕ := 3 + 1 * 2 + 8
def exprD : ℕ := 3 + 1 + 2 * 8
def exprE : ℕ := 3 * 1 * 2 * 8

-- Define the theorem stating that exprE is the largest value
theorem largest_value : exprE = 48 ∧ exprE > exprA ∧ exprE > exprB ∧ exprE > exprC ∧ exprE > exprD := by
  sorry

end largest_value_l288_288078


namespace bricks_per_course_l288_288628

theorem bricks_per_course : 
  ∃ B : ℕ, (let initial_courses := 3
            let additional_courses := 2
            let total_courses := initial_courses + additional_courses
            let last_course_half_removed := B / 2
            let total_bricks := B * total_courses - last_course_half_removed
            total_bricks = 1800) ↔ B = 400 :=
by {sorry}

end bricks_per_course_l288_288628


namespace solution_of_linear_system_l288_288004

theorem solution_of_linear_system (a b : ℚ) :
  ∃ x y : ℚ, (a - b) * x - (a + b) * y = a + b ∧ x = 0 ∧ y = -1 :=
by
  use 0
  use -1
  sorry

end solution_of_linear_system_l288_288004


namespace chairs_to_remove_l288_288220

/-- Given conditions:
1. Each row holds 13 chairs.
2. There are 169 chairs initially.
3. There are 95 expected attendees.

Task: 
Prove that the number of chairs to be removed to ensure complete rows and minimize empty seats is 65. -/
theorem chairs_to_remove (chairs_per_row total_chairs expected_attendees : ℕ)
  (h1 : chairs_per_row = 13)
  (h2 : total_chairs = 169)
  (h3 : expected_attendees = 95) :
  ∃ chairs_to_remove : ℕ, chairs_to_remove = 65 :=
by
  sorry -- proof omitted

end chairs_to_remove_l288_288220


namespace cherry_trees_leaves_l288_288555

-- Define the original number of trees
def original_num_trees : ℕ := 7

-- Define the number of trees actually planted
def actual_num_trees : ℕ := 2 * original_num_trees

-- Define the number of leaves each tree drops
def leaves_per_tree : ℕ := 100

-- Define the total number of leaves that fall
def total_leaves : ℕ := actual_num_trees * leaves_per_tree

-- Theorem statement for the problem
theorem cherry_trees_leaves : total_leaves = 1400 := by
  sorry

end cherry_trees_leaves_l288_288555


namespace Linda_original_savings_l288_288775

variable (TV_cost : ℝ := 200) -- TV cost
variable (savings : ℝ) -- Linda's original savings

-- Prices, Discounts, Taxes
variable (sofa_price : ℝ := 600)
variable (sofa_discount : ℝ := 0.20)
variable (sofa_tax : ℝ := 0.05)

variable (dining_table_price : ℝ := 400)
variable (dining_table_discount : ℝ := 0.15)
variable (dining_table_tax : ℝ := 0.06)

variable (chair_set_price : ℝ := 300)
variable (chair_set_discount : ℝ := 0.25)
variable (chair_set_tax : ℝ := 0.04)

variable (coffee_table_price : ℝ := 100)
variable (coffee_table_discount : ℝ := 0.10)
variable (coffee_table_tax : ℝ := 0.03)

variable (service_charge_rate : ℝ := 0.02) -- Service charge rate

noncomputable def discounted_price_with_tax (price discount tax : ℝ) : ℝ :=
  let discounted_price := price * (1 - discount)
  let taxed_price := discounted_price * (1 + tax)
  taxed_price

noncomputable def total_furniture_cost : ℝ :=
  let sofa_cost := discounted_price_with_tax sofa_price sofa_discount sofa_tax
  let dining_table_cost := discounted_price_with_tax dining_table_price dining_table_discount dining_table_tax
  let chair_set_cost := discounted_price_with_tax chair_set_price chair_set_discount chair_set_tax
  let coffee_table_cost := discounted_price_with_tax coffee_table_price coffee_table_discount coffee_table_tax
  let combined_cost := sofa_cost + dining_table_cost + chair_set_cost + coffee_table_cost
  combined_cost * (1 + service_charge_rate)

theorem Linda_original_savings : savings = 4 * TV_cost ∧ savings / 4 * 3 = total_furniture_cost :=
by
  sorry -- Proof skipped

end Linda_original_savings_l288_288775


namespace fraction_tips_l288_288705

theorem fraction_tips {S : ℝ} (H1 : S > 0) (H2 : tips = (7 / 3 : ℝ) * S) (H3 : bonuses = (2 / 5 : ℝ) * S) :
  (tips / (S + tips + bonuses)) = (5 / 8 : ℝ) :=
by
  sorry

end fraction_tips_l288_288705


namespace final_value_exceeds_N_iff_k_gt_100r_over_100_minus_r_l288_288276

variable (k r s N : ℝ)
variable (h_pos_k : 0 < k)
variable (h_pos_r : 0 < r)
variable (h_pos_s : 0 < s)
variable (h_pos_N : 0 < N)
variable (h_r_lt_80 : r < 80)

theorem final_value_exceeds_N_iff_k_gt_100r_over_100_minus_r :
  N * (1 + k / 100) * (1 - r / 100) + 10 * s > N ↔ k > 100 * r / (100 - r) :=
sorry

end final_value_exceeds_N_iff_k_gt_100r_over_100_minus_r_l288_288276


namespace jackson_sandwiches_l288_288626

theorem jackson_sandwiches (weeks : ℕ) (missed_wednesdays : ℕ) (missed_fridays : ℕ)
    (h_weeks : weeks = 36) (h_missed_wednesdays : missed_wednesdays = 1) (h_missed_fridays : missed_fridays = 2) :
    let total_days := weeks * 2
    let missed_days := missed_wednesdays + missed_fridays
    total_days - missed_days = 69 :=
by
    sorry

end jackson_sandwiches_l288_288626


namespace simplify_expr_l288_288324

/-- Theorem: Simplify the expression -/
theorem simplify_expr
  (x y z w : ℝ)
  (hx : x = sqrt 3 - 1)
  (hy : y = sqrt 3 + 1)
  (hz : z = 1 - sqrt 2)
  (hw : w = 1 + sqrt 2) :
  (x ^ z / y ^ w) = 2 ^ (1 - sqrt 2) * (4 - 2 * sqrt 3) :=
by
  sorry

end simplify_expr_l288_288324


namespace time_for_b_l288_288742

theorem time_for_b (A B C : ℚ) (H1 : A + B + C = 1/4) (H2 : A = 1/12) (H3 : C = 1/18) : B = 1/9 :=
by {
  sorry
}

end time_for_b_l288_288742


namespace arithmetic_sequence_function_positive_l288_288600

def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = - (f x)

def is_monotonically_increasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d, ∀ n, a (n + 1) = a n + d

theorem arithmetic_sequence_function_positive
  {f : ℝ → ℝ} {a : ℕ → ℝ}
  (hf_odd : is_odd f)
  (hf_mono : is_monotonically_increasing f)
  (ha_arith : is_arithmetic_sequence a)
  (ha3_pos : a 3 > 0) : 
  f (a 1) + f (a 3) + f (a 5) > 0 := 
sorry

end arithmetic_sequence_function_positive_l288_288600


namespace archer_expected_hits_l288_288232

noncomputable def binomial_expected_value (n : ℕ) (p : ℝ) : ℝ :=
  n * p

theorem archer_expected_hits :
  binomial_expected_value 10 0.9 = 9 :=
by
  sorry

end archer_expected_hits_l288_288232


namespace factorization_l288_288872

theorem factorization (x y : ℝ) : (x * y^2 - x = x * (y - 1) * (y + 1)) :=
begin
  sorry
end

end factorization_l288_288872


namespace ashton_pencils_left_l288_288100

theorem ashton_pencils_left (boxes : ℕ) (pencils_per_box : ℕ) (pencils_given : ℕ) :
  boxes = 2 → pencils_per_box = 14 → pencils_given = 6 → (boxes * pencils_per_box) - pencils_given = 22 :=
by
  intros boxes_eq pencils_per_box_eq pencils_given_eq
  rw [boxes_eq, pencils_per_box_eq, pencils_given_eq]
  norm_num
  sorry

end ashton_pencils_left_l288_288100


namespace minute_hand_distance_traveled_in_45_minutes_l288_288056

-- Definitions for the problem
def minute_hand_length : ℝ := 8 -- The length of the minute hand
def time_in_minutes : ℝ := 45 -- Time in minutes
def revolutions_per_hour : ℝ := 1 -- The number of revolutions per hour for the minute hand (60 minutes)

-- Circumference of the circle
def circumference (radius : ℝ) : ℝ := 2 * Real.pi * radius

-- Number of revolutions the minute hand makes in the given time
def number_of_revolutions (time: ℝ) (total_minutes: ℝ) : ℝ :=
  time / total_minutes * revolutions_per_hour

-- Total distance traveled by the tip of the minute hand
def total_distance_traveled (radius : ℝ) (time: ℝ) : ℝ :=
  circumference(radius) * number_of_revolutions(time, 60)

-- Theorem to prove
theorem minute_hand_distance_traveled_in_45_minutes :
  total_distance_traveled(minute_hand_length, time_in_minutes) = 12 * Real.pi :=
by
  -- Placeholder for the proof
  sorry

end minute_hand_distance_traveled_in_45_minutes_l288_288056


namespace factor_expression_l288_288647

theorem factor_expression (x a b c : ℝ) :
  (x - a) ^ 2 * (b - c) + (x - b) ^ 2 * (c - a) + (x - c) ^ 2 * (a - b) = -(a - b) * (b - c) * (c - a) :=
by
  sorry

end factor_expression_l288_288647


namespace problem_solution_l288_288530

theorem problem_solution (a b c d : ℝ) 
  (h1 : 3 * a + 2 * b + 4 * c + 8 * d = 40)
  (h2 : 4 * (d + c) = b)
  (h3 : 2 * b + 2 * c = a)
  (h4 : c + 1 = d) :
  a * b * c * d = 0 :=
sorry

end problem_solution_l288_288530


namespace five_fourths_of_x_over_3_l288_288250

theorem five_fourths_of_x_over_3 (x : ℚ) : (5/4) * (x/3) = 5 * x / 12 :=
by
  sorry

end five_fourths_of_x_over_3_l288_288250


namespace sum_of_integers_l288_288785

theorem sum_of_integers (x y : ℕ) (h1 : x - y = 10) (h2 : x * y = 80) (hx_pos : 0 < x) (hy_pos : 0 < y) : x + y = 20 := by
  sorry

end sum_of_integers_l288_288785


namespace mrs_awesome_class_l288_288039

def num_students (b g : ℕ) : ℕ := b + g

theorem mrs_awesome_class (b g : ℕ) (h1 : b = g + 3) (h2 : 480 - (b * b + g * g) = 5) : num_students b g = 31 :=
by
  sorry

end mrs_awesome_class_l288_288039


namespace smallest_n_exceeds_15_l288_288453

noncomputable def g (n : ℕ) : ℕ :=
  sorry  -- Define the sum of the digits of 1 / 3^n to the right of the decimal point

theorem smallest_n_exceeds_15 : ∃ n : ℕ, n > 0 ∧ g n > 15 ∧ ∀ m : ℕ, m > 0 ∧ g m > 15 → n ≤ m :=
  sorry  -- Prove the smallest n such that g(n) > 15

end smallest_n_exceeds_15_l288_288453


namespace sequence_a2017_l288_288138

theorem sequence_a2017 (a : ℕ → ℤ)
  (h1 : a 1 = 2)
  (h2 : a 2 = 3)
  (h_rec : ∀ n, a (n + 2) = a (n + 1) - a n) :
  a 2017 = 2 :=
sorry

end sequence_a2017_l288_288138


namespace count_n_condition_l288_288127

open Real

theorem count_n_condition : (Finset.card {n ∈ Finset.range 1001 | ∀ t : ℝ, (sin t + complex.i * cos t)^n = sin (n * t) + complex.i * cos (n * t) }) = 250 := 
sorry

end count_n_condition_l288_288127


namespace shorten_other_side_area_l288_288646

-- Assuming initial dimensions and given conditions
variable (length1 length2 : ℕ)
variable (new_length : ℕ)
variable (area1 area2 : ℕ)

-- Initial dimensions of the index card
def initial_dimensions (length1 length2 : ℕ) : Prop :=
  length1 = 3 ∧ length2 = 7

-- Area when one side is shortened to a specific new length
def shortened_area (length1 length2 new_length : ℕ) : ℕ :=
  if new_length = length1 - 1 then new_length * length2 else length1 * (length2 - 1)

-- Condition that the area is 15 square inches when one side is shortened
def condition_area_15 (length1 length2 : ℕ) : Prop :=
  (shortened_area length1 length2 (length1 - 1) = 15 ∨
   shortened_area length1 length2 (length2 - 1) = 15)

-- Area when the other side is shortened by 1 inch
def new_area (length1 new_length : ℕ) : ℕ :=
  new_length * (length1 - 1)

-- Proving the final area when the other side is shortened
theorem shorten_other_side_area :
  initial_dimensions length1 length2 →
  condition_area_15 length1 length2 →
  new_area length2 (length2 - 1) = 10 :=
by
  intros hdim hc15
  have hlength1 : length1 = 3 := hdim.1
  have hlength2 : length2 = 7 := hdim.2
  sorry

end shorten_other_side_area_l288_288646


namespace jake_not_drop_coffee_l288_288299

theorem jake_not_drop_coffee :
  let p_trip := 0.40
  let p_drop_trip := 0.25
  let p_step := 0.30
  let p_drop_step := 0.20
  let p_no_drop_trip := 1 - (p_trip * p_drop_trip)
  let p_no_drop_step := 1 - (p_step * p_drop_step)
  (p_no_drop_trip * p_no_drop_step) = 0.846 :=
by
  sorry

end jake_not_drop_coffee_l288_288299


namespace find_digits_l288_288256

def five_digit_subtraction (a b c d e : ℕ) : Prop :=
    let n1 := 10000 * a + 1000 * b + 100 * c + 10 * d + e
    let n2 := 10000 * e + 1000 * d + 100 * c + 10 * b + a
    (n1 - n2) % 10 = 2 ∧ (((n1 - n2) / 10) % 10) = 7 ∧ a > e ∧ a - e = 2 ∧ b - a = 7

theorem find_digits 
    (a b c d e : ℕ) 
    (h : five_digit_subtraction a b c d e) :
    a = 9 ∧ e = 7 :=
by 
    sorry

end find_digits_l288_288256


namespace find_value_of_a_l288_288620

-- Let a, b, and c be different numbers from {1, 2, 4}
def a_b_c_valid (a b c : ℕ) : Prop := 
  (a ≠ b ∧ a ≠ c ∧ b ≠ c) ∧ 
  (a = 1 ∨ a = 2 ∨ a = 4) ∧ 
  (b = 1 ∨ b = 2 ∨ b = 4) ∧ 
  (c = 1 ∨ c = 2 ∨ c = 4)

-- The condition that (a / 2) / (b / c) equals 4 when evaluated
def expr_eq_four (a b c : ℕ) : Prop :=
  (a / 2 : ℚ) / (b / c : ℚ) = 4

-- Given the above conditions, prove that the value of 'a' is 4
theorem find_value_of_a (a b c : ℕ) (h_valid : a_b_c_valid a b c) (h_expr : expr_eq_four a b c) : a = 4 := 
  sorry

end find_value_of_a_l288_288620


namespace product_is_112015_l288_288178

-- Definitions and conditions
def is_valid_digit (d : ℕ) : Prop := d ≠ 0 ∧ d < 10

def valid_product (a b : ℕ) (p : ℕ) : Prop :=
  p = a * b ∧
  (∀ d, d ∈ [a / 100, (a % 100) / 10, a % 10, b / 100, (b % 100) / 10, b % 10] → is_valid_digit d) ∧
  (∃ C I K S, C ≠ I ∧ C ≠ K ∧ C ≠ S ∧ I ≠ K ∧ I ≠ S ∧ K ≠ S ∧ is_valid_digit C ∧ is_valid_digit I ∧ is_valid_digit K ∧ is_valid_digit S ∧
    (C ∈ [p / 100000, (p % 100000) / 10000, (p % 10000) / 1000, (p % 1000) / 100, (p % 100) / 10, p % 10]) ∧
    I ∈ [p / 100000, (p % 100000) / 10000, (p % 10000) / 1000, (p % 1000) / 100, (p % 100) / 10, p % 10] ∧
    K ∈ [p / 100000, (p % 100000) / 10000, (p % 10000) / 1000, (p % 1000) / 100, (p % 100) / 10, p %10] ∧
    S ∈ [p / 100000, (p % 100000) / 10000, (p % 10000) / 1000, (p % 1000) / 100, (p % 100) / 10, p % 10]) ∧
  p / 100000 = C

-- Theorem statement
theorem product_is_112015 (a b : ℕ) (h1: a = 521) (h2: b = 215) : valid_product a b 112015 := 
by sorry

end product_is_112015_l288_288178


namespace triangle_area_is_3_max_f_l288_288165

noncomputable def triangle_area :=
  let a : ℝ := 2
  let b : ℝ := 2 * Real.sqrt 3
  let c : ℝ := 2
  let A : ℝ := Real.pi / 3
  (1 / 2) * b * c * Real.sin A

theorem triangle_area_is_3 :
  triangle_area = 3 := by
  sorry

noncomputable def f (x : ℝ) : ℝ :=
  4 * Real.cos x * (Real.sin x * Real.cos (Real.pi / 3) + Real.cos x * Real.sin (Real.pi / 3))

theorem max_f :
  ∃ x ∈ Set.Icc 0 (Real.pi / 3), f x = 2 + Real.sqrt 3 ∧ x = Real.pi / 12 := by
  sorry

end triangle_area_is_3_max_f_l288_288165


namespace simplify_expression_as_single_fraction_l288_288080

variable (d : ℚ)

theorem simplify_expression_as_single_fraction :
  (5 + 4*d)/9 + 3 = (32 + 4*d)/9 := 
by
  sorry

end simplify_expression_as_single_fraction_l288_288080


namespace tangent_line_at_1_l288_288585

noncomputable def f : ℝ → ℝ := λ x, x * Real.log x

theorem tangent_line_at_1 : tangent_line f 1 = λ x, x - 1 := by
  sorry

end tangent_line_at_1_l288_288585


namespace richard_older_than_david_by_l288_288406

-- Definitions based on given conditions

def richard : ℕ := sorry
def david : ℕ := 14 -- David is 14 years old.
def scott : ℕ := david - 8 -- Scott is 8 years younger than David.

-- In 8 years, Richard will be twice as old as Scott
axiom richard_in_8_years : richard + 8 = 2 * (scott + 8)

-- To prove: How many years older is Richard than David?
theorem richard_older_than_david_by : richard - david = 6 := sorry

end richard_older_than_david_by_l288_288406


namespace ratio_of_x_to_y_l288_288156

variable {x y : ℝ}

theorem ratio_of_x_to_y (h1 : (3 * x - 2 * y) / (2 * x + 3 * y) = 5 / 4) (h2 : x + y = 5) : x / y = 23 / 2 := 
by {
  sorry
}

end ratio_of_x_to_y_l288_288156


namespace neg_and_implication_l288_288289

variable (p q : Prop)

theorem neg_and_implication : ¬ (p ∧ q) → ¬ p ∨ ¬ q := by
  sorry

end neg_and_implication_l288_288289


namespace jill_spent_on_other_items_l288_288488

theorem jill_spent_on_other_items {T : ℝ} (h₁ : T > 0)
    (h₁ : 0.5 * T + 0.2 * T + O * T / 100 = T)
    (h₂ : 0.04 * 0.5 * T = 0.02 * T)
    (h₃ : 0 * 0.2 * T = 0)
    (h₄ : 0.08 * O * T / 100 = 0.0008 * O * T)
    (h₅ : 0.044 * T = 0.02 * T + 0 + 0.0008 * O * T) :
  O = 30 := 
sorry

end jill_spent_on_other_items_l288_288488


namespace darks_washing_time_l288_288096

theorem darks_washing_time (x : ℕ) :
  (72 + x + 45) + (50 + 65 + 54) = 344 → x = 58 :=
by
  sorry

end darks_washing_time_l288_288096


namespace average_price_per_bottle_l288_288687

/-
  Given:
  * Number of large bottles: 1300
  * Price per large bottle: 1.89
  * Number of small bottles: 750
  * Price per small bottle: 1.38
  
  Prove:
  The approximate average price per bottle is 1.70
-/
theorem average_price_per_bottle : 
  let num_large_bottles := 1300
  let price_per_large_bottle := 1.89
  let num_small_bottles := 750
  let price_per_small_bottle := 1.38
  let total_cost_large_bottles := num_large_bottles * price_per_large_bottle
  let total_cost_small_bottles := num_small_bottles * price_per_small_bottle
  let total_number_bottles := num_large_bottles + num_small_bottles
  let overall_total_cost := total_cost_large_bottles + total_cost_small_bottles
  let average_price := overall_total_cost / total_number_bottles
  average_price = 1.70 :=
by
  sorry

end average_price_per_bottle_l288_288687


namespace extra_birds_l288_288979

def num_sparrows : ℕ := 10
def num_robins : ℕ := 5
def num_bluebirds : ℕ := 3
def nests_for_sparrows : ℕ := 4
def nests_for_robins : ℕ := 2
def nests_for_bluebirds : ℕ := 2

theorem extra_birds (num_sparrows : ℕ)
                    (num_robins : ℕ)
                    (num_bluebirds : ℕ)
                    (nests_for_sparrows : ℕ)
                    (nests_for_robins : ℕ)
                    (nests_for_bluebirds : ℕ) :
    num_sparrows = 10 ∧ 
    num_robins = 5 ∧ 
    num_bluebirds = 3 ∧ 
    nests_for_sparrows = 4 ∧ 
    nests_for_robins = 2 ∧ 
    nests_for_bluebirds = 2 ->
    num_sparrows - nests_for_sparrows = 6 ∧ 
    num_robins - nests_for_robins = 3 ∧ 
    num_bluebirds - nests_for_bluebirds = 1 :=
by sorry

end extra_birds_l288_288979


namespace sum_of_roots_l288_288637

theorem sum_of_roots (k p x1 x2 : ℝ) (h1 : (4 * x1 ^ 2 - k * x1 - p = 0))
  (h2 : (4 * x2 ^ 2 - k * x2 - p = 0))
  (h3 : x1 ≠ x2) : 
  x1 + x2 = k / 4 :=
sorry

end sum_of_roots_l288_288637


namespace real_part_largest_modulus_root_l288_288053

theorem real_part_largest_modulus_root 
  (z : ℂ) 
  (h : 5 * z^4 + 10 * z^3 + 10 * z^2 + 5 * z + 1 = 0) 
  (h_modulus : ∀ w : ℂ, (5 * w^4 + 10 * w^3 + 10 * w^2 + 5 * w + 1 = 0) → |z| ≥ |w|) : 
  z.re = -1/2 :=
sorry

end real_part_largest_modulus_root_l288_288053


namespace train_overtake_l288_288073

theorem train_overtake :
  let speedA := 30 -- speed of Train A in miles per hour
  let speedB := 38 -- speed of Train B in miles per hour
  let lead_timeA := 2 -- lead time of Train A in hours
  let distanceA := speedA * lead_timeA -- distance traveled by Train A in the lead time
  let t := 7.5 -- time in hours Train B travels to catch up Train A
  let total_distanceB := speedB * t -- total distance traveled by Train B in time t
  total_distanceB = 285 := 
by
  sorry

end train_overtake_l288_288073


namespace floor_x_plus_x_eq_13_div_3_l288_288887

-- Statement representing the mathematical problem
theorem floor_x_plus_x_eq_13_div_3 (x : ℚ) (h : ⌊x⌋ + x = 13/3) : x = 7/3 := 
sorry

end floor_x_plus_x_eq_13_div_3_l288_288887


namespace difference_in_girls_and_boys_l288_288666

theorem difference_in_girls_and_boys (x : ℕ) (h1 : 3 + 4 = 7) (h2 : 7 * x = 49) : 4 * x - 3 * x = 7 := by
  sorry

end difference_in_girls_and_boys_l288_288666


namespace surface_area_of_rectangular_prism_l288_288389

def SurfaceArea (length : ℕ) (width : ℕ) (height : ℕ) : ℕ :=
  2 * ((length * width) + (width * height) + (height * length))

theorem surface_area_of_rectangular_prism 
  (l w h : ℕ) 
  (hl : l = 1) 
  (hw : w = 2) 
  (hh : h = 2) : 
  SurfaceArea l w h = 16 := by
  sorry

end surface_area_of_rectangular_prism_l288_288389


namespace repeated_three_digit_divisible_l288_288918

theorem repeated_three_digit_divisible (μ : ℕ) (h : 100 ≤ μ ∧ μ < 1000) :
  ∃ k : ℕ, (1000 * μ + μ) = k * 7 * 11 * 13 := by
sorry

end repeated_three_digit_divisible_l288_288918


namespace population_meets_capacity_l288_288177

-- Define the initial conditions and parameters
def initial_year : ℕ := 1998
def initial_population : ℕ := 100
def population_growth_rate : ℕ := 4  -- quadruples every 20 years
def years_per_growth_period : ℕ := 20
def land_area_hectares : ℕ := 15000
def hectares_per_person : ℕ := 2
def maximum_capacity : ℕ := land_area_hectares / hectares_per_person

-- Define the statement
theorem population_meets_capacity :
  ∃ (years_from_initial : ℕ), years_from_initial = 60 ∧
  initial_population * population_growth_rate ^ (years_from_initial / years_per_growth_period) ≥ maximum_capacity :=
by
  sorry

end population_meets_capacity_l288_288177


namespace find_numbers_l288_288836

theorem find_numbers (a b c : ℕ) (h₁ : 10 ≤ b ∧ b < 100) (h₂ : 10 ≤ c ∧ c < 100)
    (h₃ : 10^4 * a + 100 * b + c = (a + b + c)^3) : (a = 9 ∧ b = 11 ∧ c = 25) :=
by
  sorry

end find_numbers_l288_288836


namespace system_of_equations_solution_l288_288190

theorem system_of_equations_solution (x y : ℤ) (h1 : 2 * x + 5 * y = 26) (h2 : 4 * x - 2 * y = 4) : 
    x = 3 ∧ y = 4 :=
by
  sorry

end system_of_equations_solution_l288_288190


namespace Bridget_weight_is_correct_l288_288239

-- Definitions based on conditions
def Martha_weight : ℕ := 2
def weight_difference : ℕ := 37

-- Bridget's weight based on the conditions
def Bridget_weight : ℕ := Martha_weight + weight_difference

-- Proof problem: Prove that Bridget's weight is 39
theorem Bridget_weight_is_correct : Bridget_weight = 39 := by
  -- Proof goes here
  sorry

end Bridget_weight_is_correct_l288_288239


namespace find_roots_l288_288148

-- Given the conditions:
def quadratic_function (a b c : ℝ) (x : ℝ) : ℝ := a * x ^ 2 + b * x + c

-- Given points (x, y)
def points := [(-5, 6), (-4, 0), (-2, -6), (0, -4), (2, 6)] 

-- Prove that the roots of the quadratic equation are -4 and 1
theorem find_roots (a b c : ℝ) (h₀ : a ≠ 0)
  (h₁ : quadratic_function a b c (-5) = 6)
  (h₂ : quadratic_function a b c (-4) = 0)
  (h₃ : quadratic_function a b c (-2) = -6)
  (h₄ : quadratic_function a b c (0) = -4)
  (h₅ : quadratic_function a b c (2) = 6) :
  ∃ x₁ x₂ : ℝ, quadratic_function a b c x₁ = 0 ∧ quadratic_function a b c x₂ = 0 ∧ x₁ = -4 ∧ x₂ = 1 := 
sorry

end find_roots_l288_288148


namespace fg_of_2_eq_225_l288_288726

def f (x : ℝ) : ℝ := x^2
def g (x : ℝ) : ℝ := 2 * x^2 + 3 * x + 1

theorem fg_of_2_eq_225 : f (g 2) = 225 := by
  sorry

end fg_of_2_eq_225_l288_288726


namespace fraction_to_decimal_l288_288425

theorem fraction_to_decimal : (47 : ℝ) / 160 = 0.29375 :=
by
  sorry

end fraction_to_decimal_l288_288425


namespace rectangular_prism_sum_l288_288858

theorem rectangular_prism_sum : 
  let edges := 12
  let vertices := 8
  let faces := 6
  edges + vertices + faces = 26 := by
sorry

end rectangular_prism_sum_l288_288858


namespace price_of_other_frisbees_l288_288552

theorem price_of_other_frisbees :
  ∃ F3 Fx Px : ℕ, F3 + Fx = 60 ∧ 3 * F3 + Px * Fx = 204 ∧ Fx ≥ 24 ∧ Px = 4 := 
by
  sorry

end price_of_other_frisbees_l288_288552


namespace sum_of_cubes_eq_96_over_7_l288_288063

-- Define the conditions from the problem
variables (a r : ℝ)
axiom condition_sum : a / (1 - r) = 2
axiom condition_sum_squares : a^2 / (1 - r^2) = 6

-- Define the correct answer that we expect to prove
theorem sum_of_cubes_eq_96_over_7 :
  a^3 / (1 - r^3) = 96 / 7 :=
sorry

end sum_of_cubes_eq_96_over_7_l288_288063


namespace luke_fish_fillets_l288_288578

def fish_per_day : ℕ := 2
def days : ℕ := 30
def fillets_per_fish : ℕ := 2

theorem luke_fish_fillets : fish_per_day * days * fillets_per_fish = 120 := 
by
  sorry

end luke_fish_fillets_l288_288578


namespace greatest_integer_sum_l288_288633

def floor (x : ℚ) : ℤ := ⌊x⌋

theorem greatest_integer_sum :
  floor (2017 * 3 / 11) + 
  floor (2017 * 4 / 11) + 
  floor (2017 * 5 / 11) + 
  floor (2017 * 6 / 11) + 
  floor (2017 * 7 / 11) + 
  floor (2017 * 8 / 11) = 6048 :=
  by sorry

end greatest_integer_sum_l288_288633


namespace fraction_power_rule_example_l288_288674

theorem fraction_power_rule_example : (5 / 6)^4 = 625 / 1296 :=
by
  sorry

end fraction_power_rule_example_l288_288674


namespace correct_order_of_actions_l288_288984

-- Definitions based on the conditions
def actions : ℕ → String
| 1 => "tap"
| 2 => "pay online"
| 3 => "swipe"
| 4 => "insert into terminal"
| _ => "undefined"

def paymentTechnology : ℕ → String
| 1 => "PayPass"
| 2 => "CVC"
| 3 => "magnetic stripe"
| 4 => "chip"
| _ => "undefined"

-- Proof problem statement
theorem correct_order_of_actions :
  (actions 4 = "insert into terminal") ∧
  (actions 3 = "swipe") ∧
  (actions 1 = "tap") ∧
  (actions 2 = "pay online") →
  [4, 3, 1, 2] corresponds to ["chip", "magnetic stripe", "PayPass", "CVC"]
:=
by
  sorry

end correct_order_of_actions_l288_288984


namespace complement_intersection_l288_288910

-- Define the universal set U
def U : Set ℕ := {1, 2, 3, 4, 5}

-- Define set A
def A : Set ℕ := {1, 3, 4}

-- Define set B
def B : Set ℕ := {2, 3}

-- Define the complement of A with respect to U
def complement_U (s : Set ℕ) : Set ℕ := {x ∈ U | x ∉ s}

-- Define the statement to be proven
theorem complement_intersection :
  (complement_U A ∩ B) = {2} :=
by
  sorry

end complement_intersection_l288_288910


namespace smallest_positive_perfect_square_divisible_by_2_3_5_l288_288364

theorem smallest_positive_perfect_square_divisible_by_2_3_5 : 
  ∃ n : ℕ, (n > 0 ∧ n % 2 = 0 ∧ n % 3 = 0 ∧ n % 5 = 0) ∧ ∃ k : ℕ, n = k^2 ∧ n = 225 := 
begin
  sorry
end

end smallest_positive_perfect_square_divisible_by_2_3_5_l288_288364


namespace spiral_grid_last_column_sum_l288_288779

-- Define properties of the grid and the spiral fill
def in_spiral_grid (n: ℕ) (pos: ℕ × ℕ) : ℕ :=
  sorry -- A function that would return the number at position pos in an n x n spiral grid

-- Numerical grid parameters
def grid_size := 15
def center_row := 8
def center_col := 8

-- Positions in the last column
def bottom_right : ℕ × ℕ := (grid_size, grid_size)
def top_right : ℕ × ℕ := (1, grid_size)

-- The theorem statement
theorem spiral_grid_last_column_sum :
  in_spiral_grid grid_size bottom_right + in_spiral_grid grid_size top_right = 436 :=
  sorry

end spiral_grid_last_column_sum_l288_288779


namespace maize_donation_amount_l288_288416

-- Definitions and Conditions
def monthly_storage : ℕ := 1
def months_in_year : ℕ := 12
def years : ℕ := 2
def stolen_tonnes : ℕ := 5
def total_tonnes_at_end : ℕ := 27

-- Theorem statement
theorem maize_donation_amount :
  let total_stored := monthly_storage * (months_in_year * years)
  let remaining_after_theft := total_stored - stolen_tonnes
  total_tonnes_at_end - remaining_after_theft = 8 :=
by
  -- This part is just the statement, hence we use sorry to omit the proof.
  sorry

end maize_donation_amount_l288_288416


namespace digging_project_length_l288_288543

theorem digging_project_length (L : ℝ) (V1 V2 : ℝ) (depth1 length1 depth2 breadth1 breadth2 : ℝ) 
  (h1 : depth1 = 100) (h2 : length1 = 25) (h3 : breadth1 = 30) (h4 : V1 = depth1 * length1 * breadth1)
  (h5 : depth2 = 75) (h6 : breadth2 = 50) (h7 : V2 = depth2 * L * breadth2) (h8 : V1 / V2 = 1) :
  L = 20 :=
by
  sorry

end digging_project_length_l288_288543


namespace triangle_proportion_l288_288015

theorem triangle_proportion (p q r x y : ℝ)
  (h1 : x / q = y / r)
  (h2 : x + y = p) :
  y / r = p / (q + r) := sorry

end triangle_proportion_l288_288015


namespace ratio_of_adults_to_children_is_24_over_25_l288_288966

theorem ratio_of_adults_to_children_is_24_over_25
  (a c : ℕ) (h₁ : a ≥ 1) (h₂ : c ≥ 1) 
  (h₃ : 30 * a + 18 * c = 2340) 
  (h₄ : c % 5 = 0) :
  a = 48 ∧ c = 50 ∧ (a / c : ℚ) = 24 / 25 :=
sorry

end ratio_of_adults_to_children_is_24_over_25_l288_288966


namespace f_strictly_decreasing_intervals_f_max_min_on_interval_l288_288909

-- Define the function f
def f (x : ℝ) : ℝ := -x^3 - 6 * x^2 - 9 * x + 3

-- Define the derivative of f
def f_deriv (x : ℝ) : ℝ := -3 * x^2 - 12 * x - 9

-- Statement for part (I)
theorem f_strictly_decreasing_intervals :
  (∀ x : ℝ, x < -3 → f_deriv x < 0) ∧ (∀ x : ℝ, x > -1 → f_deriv x < 0) := by
  sorry

-- Statement for part (II)
theorem f_max_min_on_interval :
  (∀ x ∈ Set.Icc (-4 : ℝ) (2 : ℝ), f x ≤ 7) ∧
  (∀ x ∈ Set.Icc (-4 : ℝ) (2 : ℝ), f x ≥ -47) :=
  sorry

end f_strictly_decreasing_intervals_f_max_min_on_interval_l288_288909


namespace solve_ineq_system_l288_288330

theorem solve_ineq_system (x : ℝ) :
  (x - 1) / (x + 2) ≤ 0 ∧ x^2 - 2 * x - 3 < 0 ↔ -1 < x ∧ x ≤ 1 :=
by sorry

end solve_ineq_system_l288_288330


namespace distance_from_D_to_plane_B1EF_l288_288471

theorem distance_from_D_to_plane_B1EF :
  let D := (0, 0, 0)
  let B₁ := (1, 1, 1)
  let E := (1, 1/2, 0)
  let F := (1/2, 1, 0)
  ∃ (d : ℝ), d = 1 := by
  sorry

end distance_from_D_to_plane_B1EF_l288_288471


namespace calculate_a_plus_b_l288_288739

theorem calculate_a_plus_b (a b : ℝ) (h1 : 3 = a + b / 2) (h2 : 2 = a + b / 4) : a + b = 5 :=
by
  sorry

end calculate_a_plus_b_l288_288739


namespace product_a3_a10_a17_l288_288470

-- Let's define the problem setup
variable {α : Type*} [LinearOrderedField α]

def geometric_sequence (a r : α) (n : ℕ) : α := a * r ^ (n - 1)

theorem product_a3_a10_a17 
  (a r : α)
  (h1 : geometric_sequence a r 2 + geometric_sequence a r 18 = -15) 
  (h2 : geometric_sequence a r 2 * geometric_sequence a r 18 = 16) 
  (ha2pos : geometric_sequence a r 18 ≠ 0) 
  (h3 : r < 0) :
  geometric_sequence a r 3 * geometric_sequence a r 10 * geometric_sequence a r 17 = -64 :=
sorry

end product_a3_a10_a17_l288_288470


namespace jessica_has_100_dollars_l288_288319

-- Define the variables for Rodney, Ian, and Jessica
variables (R I J : ℝ)

-- Given conditions
axiom rodney_more_than_ian : R = I + 35
axiom ian_half_of_jessica : I = J / 2
axiom jessica_more_than_rodney : J = R + 15

-- The statement to prove
theorem jessica_has_100_dollars : J = 100 :=
by
  -- Proof will be completed here
  sorry

end jessica_has_100_dollars_l288_288319


namespace rectangle_area_l288_288089

noncomputable def radius : ℝ := 7
noncomputable def width : ℝ := 2 * radius
noncomputable def length : ℝ := 3 * width
noncomputable def area (l w : ℝ) : ℝ := l * w

theorem rectangle_area : area length width = 588 := sorry

end rectangle_area_l288_288089


namespace repeatable_transformation_l288_288842

theorem repeatable_transformation (a b c : ℝ) (h₁ : a + b > c) (h₂ : b + c > a) (h₃ : c + a > b) :
  (2 * c > a + b) ∧ (2 * a > b + c) ∧ (2 * b > c + a) := 
sorry

end repeatable_transformation_l288_288842


namespace factorize_xy2_minus_x_l288_288869

theorem factorize_xy2_minus_x (x y : ℝ) : xy^2 - x = x * (y - 1) * (y + 1) :=
by
  sorry

end factorize_xy2_minus_x_l288_288869


namespace find_y_l288_288391

theorem find_y (y : ℝ) (h : 2 * y / 3 = 30) : y = 45 :=
by
  sorry

end find_y_l288_288391


namespace quilt_patch_cost_l288_288763

-- Definitions of the conditions
def length : ℕ := 16
def width : ℕ := 20
def patch_area : ℕ := 4
def cost_first_10 : ℕ := 10
def cost_after_10 : ℕ := 5
def num_first_patches : ℕ := 10

-- Define the calculations based on the problem conditions
def quilt_area : ℕ := length * width
def total_patches : ℕ := quilt_area / patch_area
def cost_first : ℕ := num_first_patches * cost_first_10
def remaining_patches : ℕ := total_patches - num_first_patches
def cost_remaining : ℕ := remaining_patches * cost_after_10
def total_cost : ℕ := cost_first + cost_remaining

-- Statement of the proof problem
theorem quilt_patch_cost : total_cost = 450 := by
  -- Placeholder for the proof
  sorry

end quilt_patch_cost_l288_288763


namespace sequence_third_term_l288_288150

theorem sequence_third_term (a m : ℤ) (h_a_neg : a < 0) (h_a1 : a + m = 2) (h_a2 : a^2 + m = 4) :
  (a^3 + m = 2) :=
by
  sorry

end sequence_third_term_l288_288150


namespace john_daily_reading_hours_l288_288474

-- Definitions from the conditions
def reading_rate := 50  -- pages per hour
def total_pages := 2800  -- pages
def weeks := 4
def days_per_week := 7

-- Hypotheses derived from the conditions
def total_hours := total_pages / reading_rate  -- 2800 / 50 = 56 hours
def total_days := weeks * days_per_week  -- 4 * 7 = 28 days

-- Theorem to prove 
theorem john_daily_reading_hours : (total_hours / total_days) = 2 := by
  sorry

end john_daily_reading_hours_l288_288474


namespace quadratic_eq_mn_sum_l288_288254

theorem quadratic_eq_mn_sum (m n : ℤ) 
  (h1 : m - 1 = 2) 
  (h2 : 16 + 4 * n = 0) 
  : m + n = -1 :=
by
  sorry

end quadratic_eq_mn_sum_l288_288254


namespace arithmetic_mean_34_58_l288_288075

theorem arithmetic_mean_34_58 :
  (3 / 4 : ℚ) + (5 / 8 : ℚ) / 2 = 11 / 16 := sorry

end arithmetic_mean_34_58_l288_288075


namespace a_minus_b_is_perfect_square_l288_288441
-- Import necessary libraries

-- Define the problem in Lean
theorem a_minus_b_is_perfect_square (a b c : ℕ) (h1: Nat.gcd a (Nat.gcd b c) = 1) 
    (h2: (ab : ℚ) / (a - b) = c) : ∃ k : ℕ, a - b = k * k :=
by
  sorry

end a_minus_b_is_perfect_square_l288_288441


namespace tony_exercises_hours_per_week_l288_288350

theorem tony_exercises_hours_per_week
  (distance_walked : ℝ)
  (speed_walking : ℝ)
  (distance_ran : ℝ)
  (speed_running : ℝ)
  (days_per_week : ℕ)
  (distance_walked = 3)
  (speed_walking = 3)
  (distance_ran = 10)
  (speed_running = 5)
  (days_per_week = 7) :
  let time_walking := distance_walked / speed_walking,
      time_running := distance_ran / speed_running,
      total_time_per_day := time_walking + time_running
  in total_time_per_day * days_per_week = 21 :=
by
  -- Proof goes here
  sorry

end tony_exercises_hours_per_week_l288_288350


namespace alpha_beta_working_together_time_l288_288802

theorem alpha_beta_working_together_time
  (A B C : ℝ)
  (h : ℝ)
  (hA : A = B + 5)
  (work_together_A : A > 0)
  (work_together_B : B > 0)
  (work_together_C : C > 0)
  (combined_work : 1/A + 1/B + 1/C = 1/(A - 6))
  (combined_work2 : 1/A + 1/B + 1/C = 1/(B - 1))
  (time_gamma : 1/A + 1/B + 1/C = 2/C) :
  h = 4/3 :=
sorry

end alpha_beta_working_together_time_l288_288802


namespace distance_between_A_and_B_l288_288208

theorem distance_between_A_and_B 
  (d : ℝ)
  (h1 : ∀ (t : ℝ), (t = 2 * (t / 2)) → t = 200) 
  (h2 : ∀ (t : ℝ), 100 = d - (t / 2 + 50))
  (h3 : ∀ (t : ℝ), d = 2 * (d - 60)): 
  d = 300 :=
sorry

end distance_between_A_and_B_l288_288208


namespace no_valid_digit_replacement_l288_288397

theorem no_valid_digit_replacement :
  ¬ ∃ (A B C D E M X : ℕ),
    (A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧ A ≠ M ∧ A ≠ X ∧
     B ≠ C ∧ B ≠ D ∧ B ≠ E ∧ B ≠ M ∧ B ≠ X ∧
     C ≠ D ∧ C ≠ E ∧ C ≠ M ∧ C ≠ X ∧
     D ≠ E ∧ D ≠ M ∧ D ≠ X ∧
     E ≠ M ∧ E ≠ X ∧
     M ≠ X ∧
     0 ≤ A ∧ A < 10 ∧
     0 ≤ B ∧ B < 10 ∧
     0 ≤ C ∧ C < 10 ∧
     0 ≤ D ∧ D < 10 ∧
     0 ≤ E ∧ E < 10 ∧
     0 ≤ M ∧ M < 10 ∧
     0 ≤ X ∧ X < 10 ∧
     A * B * C * D + 1 = C * E * M * X) :=
sorry

end no_valid_digit_replacement_l288_288397


namespace calculate_base_length_l288_288967

variable (A b h : ℝ)

def is_parallelogram_base_length (A : ℝ) (b : ℝ) (h : ℝ) : Prop :=
  (A = b * h) ∧ (h = 2 * b)

theorem calculate_base_length (H : is_parallelogram_base_length A b h) : b = 15 := by
  -- H gives us the hypothesis that (A = b * h) and (h = 2 * b)
  have H1 : A = b * h := H.1
  have H2 : h = 2 * b := H.2
  -- Use substitution and algebra to solve for b
  sorry

end calculate_base_length_l288_288967


namespace max_books_borrowed_l288_288084

theorem max_books_borrowed (students : ℕ) (no_books : ℕ) (one_book : ℕ) (two_books : ℕ) (more_books : ℕ)
  (h_students : students = 30)
  (h_no_books : no_books = 5)
  (h_one_book : one_book = 12)
  (h_two_books : two_books = 8)
  (h_more_books : more_books = students - no_books - one_book - two_books)
  (avg_books : ℕ)
  (h_avg_books : avg_books = 2) :
  ∃ max_books : ℕ, max_books = 20 := 
by 
  sorry

end max_books_borrowed_l288_288084


namespace no_representation_of_form_eight_k_plus_3_or_5_l288_288782

theorem no_representation_of_form_eight_k_plus_3_or_5 (k : ℤ) :
  ∀ x y : ℤ, (8 * k + 3 ≠ x^2 - 2 * y^2) ∧ (8 * k + 5 ≠ x^2 - 2 * y^2) :=
by sorry

end no_representation_of_form_eight_k_plus_3_or_5_l288_288782


namespace exists_n_for_binomial_congruence_l288_288766

theorem exists_n_for_binomial_congruence 
  (p : ℕ) (a k : ℕ) (prime_p : Nat.Prime p) 
  (positive_a : a > 0) (positive_k : k > 0)
  (h1 : p^a < k) (h2 : k < 2 * p^a) : 
  ∃ n : ℕ, n < p^(2 * a) ∧ (Nat.choose n k) % p^a = n % p^a ∧ n % p^a = k % p^a :=
by
  sorry

end exists_n_for_binomial_congruence_l288_288766


namespace john_total_distance_l288_288396

theorem john_total_distance (speed1 time1 speed2 time2 : ℕ) (distance1 distance2 : ℕ) :
  speed1 = 35 →
  time1 = 2 →
  speed2 = 55 →
  time2 = 3 →
  distance1 = speed1 * time1 →
  distance2 = speed2 * time2 →
  distance1 + distance2 = 235 := by
  intros
  sorry

end john_total_distance_l288_288396


namespace percent_exceed_l288_288685

theorem percent_exceed (x y : ℝ) (h : x = 0.75 * y) : ((y - x) / x) * 100 = 33.33 :=
by
  sorry

end percent_exceed_l288_288685


namespace time_taken_by_A_l288_288686

theorem time_taken_by_A (v_A v_B D t_A t_B : ℚ) (h1 : v_A / v_B = 3 / 4) 
  (h2 : t_A = t_B + 30) (h3 : t_A = D / v_A) (h4 : t_B = D / v_B) 
  : t_A = 120 := 
by 
  sorry

end time_taken_by_A_l288_288686


namespace min_vertical_segment_length_l288_288337

def f (x : ℝ) : ℝ := |x - 1|
def g (x : ℝ) : ℝ := -x^2 - 4 * x - 3
def L (x : ℝ) : ℝ := f x - g x

theorem min_vertical_segment_length : ∃ (x : ℝ), L x = 10 :=
by
  sorry

end min_vertical_segment_length_l288_288337


namespace total_animals_in_farm_l288_288234

theorem total_animals_in_farm (C B : ℕ) (h1 : C = 5) (h2 : 2 * C + 4 * B = 26) : C + B = 9 :=
by
  sorry

end total_animals_in_farm_l288_288234


namespace intersection_solution_l288_288583

theorem intersection_solution (x : ℝ) (y : ℝ) (h₁ : y = 12 / (x^2 + 6)) (h₂ : x + y = 4) : x = 2 :=
by
  sorry

end intersection_solution_l288_288583


namespace find_s_l288_288030

theorem find_s (c d n r s : ℝ) 
(h1 : c * d = 3)
(h2 : ∃ p q : ℝ, (p + q = r) ∧ (p * q = s) ∧ (p = c + 1/d ∧ q = d + 1/c)) :
s = 16 / 3 :=
by
  sorry

end find_s_l288_288030


namespace basketball_children_l288_288511

/-- Given:
  1. total spectators is 10,000
  2. 7,000 of them were men
  3. Of the remaining spectators, there were 5 times as many children as women

Prove that the number of children was 2,500. -/
theorem basketball_children (total_spectators : ℕ) (men : ℕ) (women_children : ℕ) (women children : ℕ) 
  (h1 : total_spectators = 10000) 
  (h2 : men = 7000) 
  (h3 : women_children = total_spectators - men) 
  (h4 : women + 5 * women = women_children) 
  : children = 5 * 500 := 
  by 
  sorry

end basketball_children_l288_288511


namespace muffs_bought_before_december_correct_l288_288420

/-- Total ear muffs bought by customers in December. -/
def muffs_bought_in_december := 6444

/-- Total ear muffs bought by customers in all. -/
def total_muffs_bought := 7790

/-- Ear muffs bought before December. -/
def muffs_bought_before_december : Nat :=
  total_muffs_bought - muffs_bought_in_december

/-- Theorem stating the number of ear muffs bought before December. -/
theorem muffs_bought_before_december_correct :
  muffs_bought_before_december = 1346 :=
by
  unfold muffs_bought_before_december
  unfold total_muffs_bought
  unfold muffs_bought_in_december
  sorry

end muffs_bought_before_december_correct_l288_288420


namespace pencils_left_l288_288102

def total_pencils (boxes : ℕ) (pencils_per_box : ℕ) : ℕ :=
  boxes * pencils_per_box

def remaining_pencils (initial_pencils : ℕ) (pencils_given : ℕ) : ℕ :=
  initial_pencils - pencils_given

theorem pencils_left (boxes : ℕ) (pencils_per_box : ℕ) (pencils_given : ℕ)
  (h_boxes : boxes = 2) (h_pencils_per_box : pencils_per_box = 14) (h_pencils_given : pencils_given = 6) :
  remaining_pencils (total_pencils boxes pencils_per_box) pencils_given = 22 :=
by
  rw [h_boxes, h_pencils_per_box, h_pencils_given]
  norm_num
  sorry

end pencils_left_l288_288102


namespace sum_of_a_values_l288_288282

theorem sum_of_a_values : 
  (∀ (a x : ℝ), (a + x) / 2 ≥ x - 2 ∧ x / 3 - (x - 2) > 2 / 3 ∧ 
  (x - 1) / (4 - x) + (a + 5) / (x - 4) = -4 ∧ x < 2 ∧ (∃ n : ℤ, x = n ∧ 0 < n)) →
  ∃ I : ℤ, I = 12 :=
by
  sorry

end sum_of_a_values_l288_288282


namespace number_at_two_units_right_of_origin_l288_288313

theorem number_at_two_units_right_of_origin : 
  ∀ (n : ℝ), (n = 0) →
  ∀ (x : ℝ), (x = n + 2) →
  x = 2 := 
by
  sorry

end number_at_two_units_right_of_origin_l288_288313


namespace erased_number_is_six_l288_288949

theorem erased_number_is_six (n x : ℕ) (h1 : (n * (n + 1)) / 2 - x = 45 * (n - 1) / 4):
  x = 6 :=
by
  sorry

end erased_number_is_six_l288_288949


namespace paul_books_left_l288_288489
-- Add the necessary imports

-- Define the initial conditions
def initial_books : ℕ := 115
def books_sold : ℕ := 78

-- Statement of the problem as a theorem
theorem paul_books_left : (initial_books - books_sold) = 37 := by
  -- Proof omitted
  sorry

end paul_books_left_l288_288489


namespace orange_juice_fraction_in_mixture_l288_288201

theorem orange_juice_fraction_in_mixture :
  let capacity1 := 800
  let capacity2 := 700
  let fraction1 := (1 : ℚ) / 4
  let fraction2 := (3 : ℚ) / 7
  let orange_juice1 := capacity1 * fraction1
  let orange_juice2 := capacity2 * fraction2
  let total_orange_juice := orange_juice1 + orange_juice2
  let total_volume := capacity1 + capacity2
  let fraction := total_orange_juice / total_volume
  fraction = (1 : ℚ) / 3 := by
  sorry

end orange_juice_fraction_in_mixture_l288_288201


namespace quilt_patch_cost_is_correct_l288_288759

noncomputable def quilt_area : ℕ := 16 * 20

def patch_area : ℕ := 4

def first_10_patch_cost : ℕ := 10

def discount_patch_cost : ℕ := 5

def total_patches (quilt_area patch_area : ℕ) : ℕ := quilt_area / patch_area

def cost_for_first_10 (first_10_patch_cost : ℕ) : ℕ := 10 * first_10_patch_cost

def cost_for_discounted (total_patches first_10_patch_cost discount_patch_cost : ℕ) : ℕ :=
  (total_patches - 10) * discount_patch_cost

def total_cost (cost_for_first_10 cost_for_discounted : ℕ) : ℕ :=
  cost_for_first_10 + cost_for_discounted

theorem quilt_patch_cost_is_correct :
  total_cost (cost_for_first_10 first_10_patch_cost)
             (cost_for_discounted (total_patches quilt_area patch_area) first_10_patch_cost discount_patch_cost) = 450 :=
by
  sorry

end quilt_patch_cost_is_correct_l288_288759


namespace one_point_shots_count_l288_288074

-- Define the given conditions
def three_point_shots : Nat := 15
def two_point_shots : Nat := 12
def total_points : Nat := 75
def points_per_three_shot : Nat := 3
def points_per_two_shot : Nat := 2

-- Define the total points contributed by three-point and two-point shots
def three_point_total : Nat := three_point_shots * points_per_three_shot
def two_point_total : Nat := two_point_shots * points_per_two_shot
def combined_point_total : Nat := three_point_total + two_point_total

-- Formulate the theorem to prove the number of one-point shots Tyson made
theorem one_point_shots_count : combined_point_total <= total_points →
  (total_points - combined_point_total = 6) :=
by 
  -- Skip the proof
  sorry

end one_point_shots_count_l288_288074


namespace sum_primes_between_20_and_40_l288_288811

open Nat

def primesBetween20And40 : List Nat := [23, 29, 31, 37]

theorem sum_primes_between_20_and_40 :
  (primesBetween20And40.sum = 120) :=
by
  sorry

end sum_primes_between_20_and_40_l288_288811


namespace smallest_number_diminished_by_2_divisible_12_16_18_21_28_l288_288526

def conditions_holds (n : ℕ) : Prop :=
  (n - 2) % 12 = 0 ∧ (n - 2) % 16 = 0 ∧ (n - 2) % 18 = 0 ∧ (n - 2) % 21 = 0 ∧ (n - 2) % 28 = 0

theorem smallest_number_diminished_by_2_divisible_12_16_18_21_28 :
  ∃ (n : ℕ), conditions_holds n ∧ (∀ m, conditions_holds m → n ≤ m) ∧ n = 1009 :=
by
  sorry

end smallest_number_diminished_by_2_divisible_12_16_18_21_28_l288_288526


namespace find_line_equation_through_two_points_find_circle_equation_tangent_to_x_axis_l288_288839

open Real

-- Given conditions
def line_passes_through (x1 y1 x2 y2 : ℝ) (l : ℝ → ℝ → Prop) : Prop :=
  l x1 y1 ∧ l x2 y2

def circle_tangent_to_x_axis (center_x center_y : ℝ) (r : ℝ) (C : ℝ → ℝ → Prop) : Prop :=
  C center_x center_y ∧ center_y = r

-- We want to prove:
-- 1. The equation of line l is x - 2y = 0
theorem find_line_equation_through_two_points:
  ∃ l : ℝ → ℝ → Prop, line_passes_through 2 1 6 3 l ∧ (∀ x y, l x y ↔ x - 2 * y = 0) :=
  sorry

-- 2. The equation of circle C is (x - 2)^2 + (y - 1)^2 = 1
theorem find_circle_equation_tangent_to_x_axis:
  ∃ C : ℝ → ℝ → Prop, circle_tangent_to_x_axis 2 1 1 C ∧ (∀ x y, C x y ↔ (x - 2)^2 + (y - 1)^2 = 1) :=
  sorry

end find_line_equation_through_two_points_find_circle_equation_tangent_to_x_axis_l288_288839


namespace initial_population_l288_288663

/-- The population of a town decreases annually at the rate of 20% p.a.
    Given that the population of the town after 2 years is 19200,
    prove that the initial population of the town was 30,000. -/
theorem initial_population (P : ℝ) (h : 0.64 * P = 19200) : P = 30000 :=
sorry

end initial_population_l288_288663


namespace balloon_difference_l288_288210

def num_balloons_you := 7
def num_balloons_friend := 5

theorem balloon_difference : (num_balloons_you - num_balloons_friend) = 2 := by
  sorry

end balloon_difference_l288_288210


namespace find_k_value_l288_288343

theorem find_k_value : 
  let a := 3 ^ 1001
  let b := 4 ^ 1002
  (a + b) ^ 2 - (a - b) ^ 2 = 16 * 12 ^ 1001 :=
by
  let a := 3 ^ 1001
  let b := 4 ^ 1002
  sorry

end find_k_value_l288_288343


namespace find_k_l288_288003

variable (a b : ℝ → ℝ → ℝ)
variable {k : ℝ}

-- Defining conditions
axiom a_perpendicular_b : ∀ x y, a x y = 0
axiom a_unit_vector : a 1 0 = 1
axiom b_unit_vector : b 0 1 = 1
axiom sum_perpendicular_to_k_diff : ∀ x y, (a x y + b x y) * (k * a x y - b x y) = 0

theorem find_k : k = 1 :=
sorry

end find_k_l288_288003


namespace probability_of_all_red_is_correct_l288_288088

noncomputable def probability_of_all_red_drawn : ℚ :=
  let total_ways := (Nat.choose 10 5)   -- Total ways to choose 5 balls from 10
  let red_ways := (Nat.choose 5 5)      -- Ways to choose all 5 red balls
  red_ways / total_ways

theorem probability_of_all_red_is_correct :
  probability_of_all_red_drawn = 1 / 252 := by
  sorry

end probability_of_all_red_is_correct_l288_288088


namespace problem_sum_150_consecutive_integers_l288_288825

theorem problem_sum_150_consecutive_integers : 
  ∃ k : ℕ, 150 * k + 11325 = 5310375 :=
sorry

end problem_sum_150_consecutive_integers_l288_288825


namespace tetrahedron_colorings_l288_288714

-- Define the problem conditions
def tetrahedron_faces : ℕ := 4
def colors : List String := ["red", "white", "blue", "yellow"]

-- The theorem statement
theorem tetrahedron_colorings :
  ∃ n : ℕ, n = 35 ∧ ∀ (c : List String), c.length = tetrahedron_faces → c ⊆ colors →
  (true) := -- Placeholder (you can replace this condition with the appropriate condition)
by
  -- The proof is omitted with 'sorry' as instructed
  sorry

end tetrahedron_colorings_l288_288714


namespace reduced_travel_time_l288_288704

-- Definition of conditions as given in part a)
def initial_speed := 48 -- km/h
def initial_time := 50/60 -- hours (50 minutes)
def required_speed := 60 -- km/h
def reduced_time := 40/60 -- hours (40 minutes)

-- Problem statement
theorem reduced_travel_time :
  ∃ t2, (initial_speed * initial_time = required_speed * t2) ∧ (t2 = reduced_time) :=
by
  sorry

end reduced_travel_time_l288_288704


namespace jett_profit_l288_288170

def initial_cost : ℕ := 600
def vaccination_cost : ℕ := 500
def daily_food_cost : ℕ := 20
def number_of_days : ℕ := 40
def selling_price : ℕ := 2500

def total_expenses : ℕ := initial_cost + vaccination_cost + daily_food_cost * number_of_days
def profit : ℕ := selling_price - total_expenses

theorem jett_profit : profit = 600 :=
by
  -- Completed proof steps
  sorry

end jett_profit_l288_288170


namespace min_vertical_distance_l288_288336

theorem min_vertical_distance :
  ∃ (d : ℝ), ∀ (x : ℝ),
    (y1 x = |x - 1| ∧ y2 x = -x^2 - 4*x - 3) ∧ 
    (d = infi (λ x, abs (y1 x - y2 x))) ∧
    (d = 7 / 4) :=
by
  let y1 := λ x : ℝ, abs (x - 1)
  let y2 := λ x : ℝ, -x^2 - 4 * x - 3
  exists 7 / 4
  simp
  sorry

end min_vertical_distance_l288_288336


namespace landmark_distance_l288_288095

theorem landmark_distance (d : ℝ) : 
  (d >= 7 → d < 7) ∨ (d <= 8 → d > 8) ∨ (d <= 10 → d > 10) → d > 10 :=
by
  sorry

end landmark_distance_l288_288095


namespace trigonometric_identity_l288_288859

theorem trigonometric_identity :
  (1 / Real.cos (70 * Real.pi / 180) + Real.sqrt 3 / Real.sin (70 * Real.pi / 180)) = 4 * Real.tan (10 * Real.pi / 180) :=
by
  sorry

end trigonometric_identity_l288_288859


namespace three_digit_numbers_form_3_pow_l288_288608

theorem three_digit_numbers_form_3_pow (n : ℤ) : 
  ∃! (n : ℤ), 100 ≤ 3^n ∧ 3^n ≤ 999 :=
by {
  use [5, 6],
  sorry
}

end three_digit_numbers_form_3_pow_l288_288608


namespace exists_y_lt_p_div2_py_plus1_not_product_of_greater_y_l288_288940

theorem exists_y_lt_p_div2_py_plus1_not_product_of_greater_y (p : ℕ) [hp : Fact (Nat.Prime p)] (h3 : 3 < p) :
  ∃ y : ℕ, y < p / 2 ∧ ∀ a b : ℕ, py + 1 ≠ a * b ∨ a ≤ y ∨ b ≤ y :=
by
  sorry

end exists_y_lt_p_div2_py_plus1_not_product_of_greater_y_l288_288940


namespace smallest_positive_perfect_square_divisible_by_2_3_5_l288_288381

/-- Problem statement: 
    The smallest positive perfect square that is divisible by 2, 3, and 5 is 900.
-/
theorem smallest_positive_perfect_square_divisible_by_2_3_5 : 
  ∃ n : ℕ, 0 < n ∧ n * n % 2 = 0 ∧ n * n % 3 = 0 ∧ n * n % 5 = 0 ∧ n * n = 900 := 
by
  sorry

end smallest_positive_perfect_square_divisible_by_2_3_5_l288_288381


namespace simplify_expression_l288_288187

theorem simplify_expression (a b c d x y : ℝ) (h : cx ≠ -dy) :
  (cx * (b^2 * x^2 + 3 * b^2 * y^2 + a^2 * y^2) + dy * (b^2 * x^2 + 3 * a^2 * x^2 + a^2 * y^2)) / (cx + dy)
  = (b^2 + 3 * a^2) * x^2 + (a^2 + 3 * b^2) * y^2 := by
  sorry

end simplify_expression_l288_288187


namespace fraction_power_rule_example_l288_288675

theorem fraction_power_rule_example : (5 / 6)^4 = 625 / 1296 :=
by
  sorry

end fraction_power_rule_example_l288_288675


namespace fraction_operation_correct_l288_288417

theorem fraction_operation_correct 
  (a b : ℝ) : 
  (0.2 * (3 * a + 10 * b) = 6 * a + 20 * b) → 
  (0.1 * (2 * a + 5 * b) = 2 * a + 5 * b) →
  (∀ c : ℝ, c ≠ 0 → (a / b = (a * c) / (b * c))) ∨
  (∀ x y : ℝ, ((x - y) / (x + y) ≠ (y - x) / (x - y))) ∨
  (∀ x : ℝ, (x + x * x * x + x * y ≠ 1 / x * x)) →
  ((0.3 * a + b) / (0.2 * a + 0.5 * b) = (3 * a + 10 * b) / (2 * a + 5 * b)) :=
sorry

end fraction_operation_correct_l288_288417


namespace original_amount_l288_288834

theorem original_amount (X : ℝ) (h : 0.05 * X = 25) : X = 500 :=
sorry

end original_amount_l288_288834


namespace hyperbola_eccentricity_l288_288790

theorem hyperbola_eccentricity (m : ℝ) (h : 0 < m) :
  ∃ e, e = Real.sqrt (1 + m) ∧ e > Real.sqrt 2 → m > 1 :=
by
  sorry

end hyperbola_eccentricity_l288_288790


namespace piggy_bank_donation_l288_288706

theorem piggy_bank_donation (total_earnings : ℕ) (cost_of_ingredients : ℕ) 
  (total_donation_homeless_shelter : ℕ) : 
  (total_earnings = 400) → (cost_of_ingredients = 100) → (total_donation_homeless_shelter = 160) → 
  (total_donation_homeless_shelter - (total_earnings - cost_of_ingredients) / 2 = 10) :=
by
  intros h1 h2 h3
  sorry

end piggy_bank_donation_l288_288706


namespace other_root_l288_288950

theorem other_root (m : ℝ) (x : ℝ) (hx : 3 * x ^ 2 + m * x - 7 = 0) (root1 : x = 1) :
  ∃ y : ℝ, 3 * y ^ 2 + m * y - 7 = 0 ∧ y = -7 / 3 :=
by
  sorry

end other_root_l288_288950


namespace problem_statement_l288_288277

theorem problem_statement (x : ℝ) (h : x + 1/x = 3) : (x - 3)^4 + 81 / (x - 3)^4 = 63 :=
by
  sorry

end problem_statement_l288_288277


namespace samantha_spends_on_dog_toys_l288_288496

theorem samantha_spends_on_dog_toys:
  let toy_price := 12.00
  let discount := 0.5
  let num_toys := 4
  let tax_rate := 0.08
  let full_price_toys := num_toys / 2
  let half_price_toys := num_toys / 2
  let total_cost_before_tax := full_price_toys * toy_price + half_price_toys * (toy_price * discount)
  let sales_tax := total_cost_before_tax * tax_rate
  let total_cost_after_tax := total_cost_before_tax + sales_tax
  total_cost_after_tax = 38.88 :=
by {
  sorry
}

end samantha_spends_on_dog_toys_l288_288496


namespace min_value_expression_l288_288477

noncomputable def min_expression (a b c : ℝ) : ℝ :=
  (a + b) / c + (b + c) / a + (c + a) / b + 3

theorem min_value_expression {a b c : ℝ} (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : a ≠ b) (h5 : b ≠ c) (h6 : c ≠ a) : 
  ∃ x, min_expression a b c = x ∧ x ≥ 9 := 
sorry

end min_value_expression_l288_288477


namespace bobby_pancakes_left_l288_288238

theorem bobby_pancakes_left (initial_pancakes : ℕ) (bobby_ate : ℕ) (dog_ate : ℕ) :
  initial_pancakes = 21 → bobby_ate = 5 → dog_ate = 7 → initial_pancakes - (bobby_ate + dog_ate) = 9 :=
by
  intros h1 h2 h3
  sorry

end bobby_pancakes_left_l288_288238


namespace geometric_power_inequality_l288_288306

theorem geometric_power_inequality {a : ℝ} {n k : ℕ} (h₀ : 1 < a) (h₁ : 0 < n) (h₂ : n < k) :
  (a^n - 1) / n < (a^k - 1) / k :=
sorry

end geometric_power_inequality_l288_288306


namespace expenditure_representation_l288_288857

theorem expenditure_representation (income expenditure : ℤ)
  (h_income : income = 60)
  (h_expenditure : expenditure = 40) :
  -expenditure = -40 :=
by {
  sorry
}

end expenditure_representation_l288_288857


namespace quadratic_binomial_square_l288_288205

theorem quadratic_binomial_square (k : ℝ) : (∃ b : ℝ, x^2 - 18 * x + k = (x - b)^2) → k = 81 :=
begin
  sorry
end

end quadratic_binomial_square_l288_288205


namespace seohyun_initial_marbles_l288_288025

variable (M : ℤ)

theorem seohyun_initial_marbles (h1 : (2 / 3) * M = 12) (h2 : (1 / 2) * M + 12 = M) : M = 36 :=
sorry

end seohyun_initial_marbles_l288_288025


namespace least_area_of_square_l288_288688

theorem least_area_of_square :
  ∀ (s : ℝ), (3.5 ≤ s ∧ s < 4.5) → (s * s ≥ 12.25) :=
by
  intro s
  intro hs
  sorry

end least_area_of_square_l288_288688


namespace shaded_area_l288_288017

-- Let A be the length of the side of the smaller square
def A : ℝ := 4

-- Let B be the length of the side of the larger square
def B : ℝ := 12

-- The problem is to prove that the area of the shaded region is 10 square inches
theorem shaded_area (A B : ℝ) (hA : A = 4) (hB : B = 12) :
  (A * A) - (1/2 * (B / (B + A)) * A * B) = 10 := by
  sorry

end shaded_area_l288_288017


namespace calc_result_l288_288720

-- Define the operation and conditions
def my_op (a b c : ℝ) : ℝ :=
  3 * (a - b - c)^2

theorem calc_result (x y z : ℝ) : 
  my_op ((x - y - z)^2) ((y - x - z)^2) ((z - x - y)^2) = 0 :=
by
  sorry

end calc_result_l288_288720


namespace shaded_solid_volume_l288_288469

noncomputable def volume_rectangular_prism (length width height : ℕ) : ℕ :=
  length * width * height

theorem shaded_solid_volume :
  volume_rectangular_prism 4 5 6 - volume_rectangular_prism 1 2 4 = 112 :=
by
  sorry

end shaded_solid_volume_l288_288469


namespace MarksScore_l288_288295

theorem MarksScore (h_highest : ℕ) (h_range : ℕ) (h_relation : h_highest - h_least = h_range) (h_mark_twice : Mark = 2 * h_least) : Mark = 46 :=
by
    let h_highest := 98
    let h_range := 75
    let h_least := h_highest - h_range
    let Mark := 2 * h_least
    have := h_relation
    have := h_mark_twice
    sorry

end MarksScore_l288_288295


namespace inequality_preservation_l288_288916

theorem inequality_preservation (x y : ℝ) (h : x < y) : 2 * x < 2 * y :=
sorry

end inequality_preservation_l288_288916


namespace simplification_evaluation_l288_288186

-- Define the variables x and y
def x : ℕ := 2
def y : ℕ := 3

-- Define the expression
def expr := 5 * (3 * x^2 * y - x * y^2) - (x * y^2 + 3 * x^2 * y)

-- Lean 4 statement to prove the equivalence
theorem simplification_evaluation : expr = 36 :=
by
  -- Place the proof here when needed
  sorry

end simplification_evaluation_l288_288186


namespace solve_problem_l288_288036

noncomputable def x_star (x : ℝ) : ℝ :=
  if h : x ≥ 2 then 2 * ((x : ℕ) / 2) else 0

noncomputable def f (x : ℝ) : ℝ :=
  x^2 * (x_star x)^2

def problem_statement : Prop :=
  let y := Real.log (f 7.2 / f 5.4) / Real.log 3 in
  abs (y - 1.261) < 0.001   

theorem solve_problem : problem_statement :=
  by sorry

end solve_problem_l288_288036


namespace polynomial_sum_squares_l288_288894

theorem polynomial_sum_squares (a0 a1 a2 a3 a4 a5 a6 a7 : ℤ)
  (h₁ : (1 - 2) ^ 7 = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7)
  (h₂ : (1 + -2) ^ 7 = a0 - a1 + a2 - a3 + a4 - a5 + a6 - a7) :
  (a0 + a2 + a4 + a6) ^ 2 - (a1 + a3 + a5 + a7) ^ 2 = -2187 := 
  sorry

end polynomial_sum_squares_l288_288894


namespace count_three_digit_numbers_power_of_three_l288_288607

theorem count_three_digit_numbers_power_of_three :
  { n : ℕ | 100 ≤ 3^n ∧ 3^n ≤ 999 }.toFinset.card = 2 := by
  sorry

end count_three_digit_numbers_power_of_three_l288_288607


namespace proper_subsets_of_A_l288_288944

open Set

namespace ProofProblem

def universal_set : Set ℕ := {0, 1, 2, 3}
def complement_of_A : Set ℕ := {2}
def A : Set ℕ := universal_set \ complement_of_A

theorem proper_subsets_of_A :
  (A = {0, 1, 3}) →
  (complement_of_A = universal_set \ A) →
  ∃ n, n = 2^Fintype.card A - 1 ∧ n = 7 :=
by {
  intros hA hComplementA,
  existsi (2 ^ Fintype.card A - 1),
  split,
  {
    rw [←hA, Fintype.card, Fintype.card, Finset.card],
    simp,
    norm_num,
  },
  { 
    norm_num,
  }
}

end ProofProblem

end proper_subsets_of_A_l288_288944


namespace tetrahedron_volume_lower_bound_l288_288183

noncomputable def volume_tetrahedron (d1 d2 d3 : ℝ) : ℝ := sorry

theorem tetrahedron_volume_lower_bound {d1 d2 d3 : ℝ} (h1 : d1 > 0) (h2 : d2 > 0) (h3 : d3 > 0) :
  volume_tetrahedron d1 d2 d3 ≥ (1 / 3) * d1 * d2 * d3 :=
sorry

end tetrahedron_volume_lower_bound_l288_288183


namespace ainsley_wins_100a_plus_b_eq_109_l288_288843

theorem ainsley_wins_100a_plus_b_eq_109 : 
  let a b : ℕ := 1, 9,
  100 * a + b = 109 :=
by
  sorry

end ainsley_wins_100a_plus_b_eq_109_l288_288843


namespace compare_08_and_one_eighth_l288_288535

theorem compare_08_and_one_eighth :
  0.8 - (1 / 8 : ℝ) = 0.675 := 
sorry

end compare_08_and_one_eighth_l288_288535


namespace inequality_preservation_l288_288914

theorem inequality_preservation (x y : ℝ) (h : x < y) : 2 * x < 2 * y :=
sorry

end inequality_preservation_l288_288914


namespace correct_formula_l288_288497

theorem correct_formula {x y : ℕ} : 
  (x = 0 ∧ y = 100) ∨
  (x = 1 ∧ y = 90) ∨
  (x = 2 ∧ y = 70) ∨
  (x = 3 ∧ y = 40) ∨
  (x = 4 ∧ y = 0) →
  y = 100 - 5 * x - 5 * x^2 :=
by
  sorry

end correct_formula_l288_288497


namespace natural_number_195_is_solution_l288_288549

-- Define the conditions
def is_odd_digit (n : ℕ) : Prop :=
  n > 0 ∧ n % 2 = 1

def all_digits_odd (n : ℕ) : Prop :=
  ∀ d : ℕ, n / 10 ^ d % 10 < 10 → is_odd_digit (n / 10 ^ d % 10)

-- Define the proof problem
theorem natural_number_195_is_solution :
  195 < 200 ∧ all_digits_odd 195 ∧ (∃ (a b : ℕ), 10 ≤ a ∧ a < 100 ∧ 10 ≤ b ∧ b < 100 ∧ a * b = 195) :=
by
  sorry

end natural_number_195_is_solution_l288_288549


namespace percentage_apples_is_50_percent_l288_288068

-- Definitions for the given conditions
def initial_apples : ℕ := 10
def initial_oranges : ℕ := 5
def added_oranges : ℕ := 5

-- Defining the proof problem
theorem percentage_apples_is_50_percent :
  let total_fruits := initial_apples + initial_oranges + added_oranges in
  let apples_percentage := (initial_apples * 100) / total_fruits in
  apples_percentage = 50 :=
by
  sorry

end percentage_apples_is_50_percent_l288_288068


namespace group_formations_at_fair_l288_288473

theorem group_formations_at_fair : 
  (Nat.choose 7 3) * (Nat.choose 4 4) = 35 := by
  sorry

end group_formations_at_fair_l288_288473


namespace problem_inequality_l288_288059

theorem problem_inequality (a b c d : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : d > 0) (h5 : a * b * c * d = 1) : 
  a^2 + b^2 + c^2 + d^2 + a * b + a * c + a * d + b * c + b * d + c * d ≥ 10 := 
by 
  sorry

end problem_inequality_l288_288059


namespace simplify_polynomial_l288_288784

theorem simplify_polynomial (x : ℝ) :
  (2 * x^6 + x^5 + 3 * x^4 + x^3 + 5) - (x^6 + 2 * x^5 + x^4 - x^3 + 7) = 
  x^6 - x^5 + 2 * x^4 + 2 * x^3 - 2 :=
by
  sorry

end simplify_polynomial_l288_288784


namespace three_digit_powers_of_three_l288_288606

theorem three_digit_powers_of_three : 
  {n : ℤ | 100 ≤ 3^n ∧ 3^n ≤ 999}.finset.card = 2 :=
by
  sorry

end three_digit_powers_of_three_l288_288606


namespace find_focus_with_larger_x_l288_288422

def hyperbola_foci_coordinates : Prop :=
  let center := (5, 10)
  let a := 7
  let b := 3
  let c := Real.sqrt (a^2 + b^2)
  let focus1 := (5 + c, 10)
  let focus2 := (5 - c, 10)
  focus1 = (5 + Real.sqrt 58, 10)
  
theorem find_focus_with_larger_x : hyperbola_foci_coordinates := 
  by
    sorry

end find_focus_with_larger_x_l288_288422


namespace sum_primes_between_20_and_40_l288_288818

theorem sum_primes_between_20_and_40 :
  23 + 29 + 31 + 37 = 120 :=
by
  sorry

end sum_primes_between_20_and_40_l288_288818


namespace sum_of_primes_between_20_and_40_l288_288820

theorem sum_of_primes_between_20_and_40 : 
  (23 + 29 + 31 + 37) = 120 := 
by
  -- Proof goes here
sorry

end sum_of_primes_between_20_and_40_l288_288820


namespace max_students_equal_distribution_l288_288797

-- Define the number of pens and pencils
def pens : ℕ := 1008
def pencils : ℕ := 928

-- Define the problem statement which asks for the GCD of the given numbers
theorem max_students_equal_distribution : Nat.gcd pens pencils = 16 :=
by 
  -- Lean's gcd computation can be used to confirm the result
  sorry

end max_students_equal_distribution_l288_288797


namespace glycerin_solution_l288_288217

theorem glycerin_solution (x : ℝ) :
    let total_volume := 100
    let final_glycerin_percentage := 0.75
    let volume_first_solution := 75
    let volume_second_solution := 75
    let second_solution_percentage := 0.90
    let final_glycerin_volume := final_glycerin_percentage * total_volume
    let glycerin_second_solution := second_solution_percentage * volume_second_solution
    let glycerin_first_solution := x * volume_first_solution / 100
    glycerin_first_solution + glycerin_second_solution = final_glycerin_volume →
    x = 10 :=
by
    sorry

end glycerin_solution_l288_288217


namespace cost_of_acai_berry_juice_l288_288662

theorem cost_of_acai_berry_juice (cost_per_litre_cocktail : ℝ)
                                 (cost_per_litre_fruit_juice : ℝ)
                                 (litres_fruit_juice : ℝ)
                                 (litres_acai_juice : ℝ)
                                 (total_cost_cocktail : ℝ)
                                 (cost_per_litre_acai : ℝ) :
  cost_per_litre_cocktail = 1399.45 →
  cost_per_litre_fruit_juice = 262.85 →
  litres_fruit_juice = 34 →
  litres_acai_juice = 22.666666666666668 →
  total_cost_cocktail = (34 + 22.666666666666668) * 1399.45 →
  (litres_fruit_juice * cost_per_litre_fruit_juice + litres_acai_juice * cost_per_litre_acai) = total_cost_cocktail →
  cost_per_litre_acai = 3106.66666666666666 :=
by
  intros
  sorry

end cost_of_acai_berry_juice_l288_288662


namespace smallest_positive_perfect_square_div_by_2_3_5_l288_288377

-- Definition capturing the conditions of the problem
def is_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

def divisible_by (n m : ℕ) : Prop :=
  m % n = 0

-- The theorem statement combining all conditions and requiring the proof of the correct answer
theorem smallest_positive_perfect_square_div_by_2_3_5 : 
  ∃ n : ℕ, 0 < n ∧ is_perfect_square n ∧ divisible_by 2 n ∧ divisible_by 3 n ∧ divisible_by 5 n ∧ n = 900 :=
sorry

end smallest_positive_perfect_square_div_by_2_3_5_l288_288377


namespace factorize_expression_l288_288885

variable {x y : ℝ}

theorem factorize_expression : xy^2 - x = x * (y - 1) * (y + 1) := 
by
  -- Define the left-hand side of the equation
  let lhs := x * y^2 - x
  -- Define the right-hand side of the equation
  let rhs := x * (y - 1) * (y + 1)
  -- Provide the goal to prove
  show lhs = rhs
  sorry

end factorize_expression_l288_288885


namespace hot_dog_remainder_l288_288740

theorem hot_dog_remainder : 35252983 % 6 = 1 :=
by
  sorry

end hot_dog_remainder_l288_288740


namespace eval_f_neg_2_l288_288279

def f (x : ℝ) : ℝ := 2 * x^2 - 4 * x + 3

theorem eval_f_neg_2 : f (-2) = 19 :=
by
  sorry

end eval_f_neg_2_l288_288279


namespace paul_spent_252_dollars_l288_288314

noncomputable def total_cost_before_discounts : ℝ :=
  let dress_shirts := 4 * 15
  let pants := 2 * 40
  let suit := 150
  let sweaters := 2 * 30
  dress_shirts + pants + suit + sweaters

noncomputable def store_discount : ℝ := 0.20

noncomputable def coupon_discount : ℝ := 0.10

noncomputable def total_cost_after_store_discount : ℝ :=
  let initial_total := total_cost_before_discounts
  initial_total - store_discount * initial_total

noncomputable def final_total : ℝ :=
  let intermediate_total := total_cost_after_store_discount
  intermediate_total - coupon_discount * intermediate_total

theorem paul_spent_252_dollars :
  final_total = 252 := by
  sorry

end paul_spent_252_dollars_l288_288314


namespace smallest_positive_perfect_square_divisible_by_2_3_5_l288_288365

theorem smallest_positive_perfect_square_divisible_by_2_3_5 : 
  ∃ n : ℕ, (n > 0 ∧ n % 2 = 0 ∧ n % 3 = 0 ∧ n % 5 = 0) ∧ ∃ k : ℕ, n = k^2 ∧ n = 225 := 
begin
  sorry
end

end smallest_positive_perfect_square_divisible_by_2_3_5_l288_288365


namespace expenditure_representation_l288_288856

theorem expenditure_representation
    (income_representation : ℤ)
    (income_is_positive : income_representation = 60)
    (use_of_opposite_signs : ∀ (n : ℤ), n >= 0 ↔ n = 60)
    (representation_criteria : ∀ (x : ℤ), x <= 0 → x = -40) :
  ∀ (expenditure : ℤ), expenditure = 40 → -expenditure = -40 :=
by
  intro expenditure
  intro expenditure_criteria
  rw [neg_eq_neg_one_mul]
  change (-1) * expenditure = (-40)
  apply representation_criteria
  exact le_of_eq rfl
  sorry

end expenditure_representation_l288_288856


namespace difference_between_m_and_n_l288_288193

theorem difference_between_m_and_n (m n : ℕ) (hm : m > 0) (hn : n > 0) (h : 10 * 2^m = 2^n + 2^(n + 2)) :
  n - m = 1 :=
sorry

end difference_between_m_and_n_l288_288193


namespace imaginary_condition_l288_288264

noncomputable def is_imaginary (z : ℂ) : Prop := z.im ≠ 0

theorem imaginary_condition (z1 z2 : ℂ) :
  ( ∃ (z1 : ℂ), is_imaginary z1 ∨ is_imaginary z2 ∨ (is_imaginary (z1 - z2))) ↔
  ∃ (z1 z2 : ℂ), is_imaginary z1 ∨ is_imaginary z2 ∧ ¬ (is_imaginary (z1 - z2)) :=
sorry

end imaginary_condition_l288_288264


namespace lcm_factor_of_hcf_and_larger_number_l288_288660

theorem lcm_factor_of_hcf_and_larger_number (A B : ℕ) (hcf : ℕ) (hlarger : A = 450) (hhcf : hcf = 30) (hwrel : A % hcf = 0) : ∃ x y, x = 15 ∧ (A * B = hcf * x * y) :=
by
  sorry

end lcm_factor_of_hcf_and_larger_number_l288_288660


namespace smallest_positive_perfect_square_div_by_2_3_5_l288_288375

-- Definition capturing the conditions of the problem
def is_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

def divisible_by (n m : ℕ) : Prop :=
  m % n = 0

-- The theorem statement combining all conditions and requiring the proof of the correct answer
theorem smallest_positive_perfect_square_div_by_2_3_5 : 
  ∃ n : ℕ, 0 < n ∧ is_perfect_square n ∧ divisible_by 2 n ∧ divisible_by 3 n ∧ divisible_by 5 n ∧ n = 900 :=
sorry

end smallest_positive_perfect_square_div_by_2_3_5_l288_288375


namespace pencil_count_l288_288507

def total_pencils (drawer : Nat) (desk_0 : Nat) (add_dan : Nat) (remove_sarah : Nat) : Nat :=
  let desk_1 := desk_0 + add_dan
  let desk_2 := desk_1 - remove_sarah
  drawer + desk_2

theorem pencil_count :
  total_pencils 43 19 16 7 = 71 :=
by
  sorry

end pencil_count_l288_288507


namespace quadratic_has_real_roots_l288_288712

open Real

theorem quadratic_has_real_roots (k : ℝ) (h : k ≠ 0) :
    ∃ x : ℝ, x^2 + k * x + k^2 - 1 = 0 ↔
    -2 / sqrt 3 ≤ k ∧ k ≤ 2 / sqrt 3 :=
by
  sorry

end quadratic_has_real_roots_l288_288712


namespace relationship_xyz_w_l288_288609

theorem relationship_xyz_w (x y z w : ℝ) (h : (x + y) / (y + z) = (2 * z + w) / (w + x)) :
  x = 2 * z - w := 
sorry

end relationship_xyz_w_l288_288609
