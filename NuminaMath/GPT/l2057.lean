import Mathlib

namespace NUMINAMATH_GPT_solution_set_ineq_l2057_205768

theorem solution_set_ineq (x : ℝ) : x^2 - 2 * abs x - 15 > 0 ↔ x < -5 ∨ x > 5 :=
sorry

end NUMINAMATH_GPT_solution_set_ineq_l2057_205768


namespace NUMINAMATH_GPT_least_positive_base_ten_seven_binary_digits_l2057_205725

theorem least_positive_base_ten_seven_binary_digits : ∃ n : ℕ, n = 64 ∧ (n >= 2^6 ∧ n < 2^7) := 
by
  sorry

end NUMINAMATH_GPT_least_positive_base_ten_seven_binary_digits_l2057_205725


namespace NUMINAMATH_GPT_added_water_correct_l2057_205751

theorem added_water_correct (initial_fullness : ℝ) (final_fullness : ℝ) (capacity : ℝ) (initial_amount : ℝ) (final_amount : ℝ) (added_water : ℝ) :
    initial_fullness = 0.30 →
    final_fullness = 3/4 →
    capacity = 60 →
    initial_amount = initial_fullness * capacity →
    final_amount = final_fullness * capacity →
    added_water = final_amount - initial_amount →
    added_water = 27 :=
by
  intros
  -- Insert the proof here
  sorry

end NUMINAMATH_GPT_added_water_correct_l2057_205751


namespace NUMINAMATH_GPT_value_of_x_l2057_205797

theorem value_of_x : ∃ (x : ℚ), (10 - 2 * x) ^ 2 = 4 * x ^ 2 + 20 * x ∧ x = 5 / 3 :=
by
  sorry

end NUMINAMATH_GPT_value_of_x_l2057_205797


namespace NUMINAMATH_GPT_book_profit_percentage_l2057_205745

noncomputable def profit_percentage (cost_price marked_price : ℝ) (discount_rate : ℝ) : ℝ :=
  let discount := discount_rate / 100 * marked_price
  let selling_price := marked_price - discount
  let profit := selling_price - cost_price
  (profit / cost_price) * 100

theorem book_profit_percentage :
  profit_percentage 47.50 69.85 15 = 24.994736842105263 :=
by
  sorry

end NUMINAMATH_GPT_book_profit_percentage_l2057_205745


namespace NUMINAMATH_GPT_intersection_count_sum_l2057_205716

theorem intersection_count_sum : 
  let m := 252
  let n := 252
  m + n = 504 := 
by {
  let m := 252 
  let n := 252 
  exact Eq.refl 504
}

end NUMINAMATH_GPT_intersection_count_sum_l2057_205716


namespace NUMINAMATH_GPT_problem_statement_l2057_205762

noncomputable def calculateValue (n : ℕ) : ℕ :=
  Nat.choose (3 * n) (38 - n) + Nat.choose (n + 21) (3 * n)

theorem problem_statement : calculateValue 10 = 466 := by
  sorry

end NUMINAMATH_GPT_problem_statement_l2057_205762


namespace NUMINAMATH_GPT_composite_shape_perimeter_l2057_205704

theorem composite_shape_perimeter :
  let r1 := 2.1
  let r2 := 3.6
  let π_approx := 3.14159
  let total_perimeter := π_approx * (r1 + r2)
  total_perimeter = 18.31 :=
by
  let radius1 := 2.1
  let radius2 := 3.6
  let total_radius := radius1 + radius2
  let pi_value := 3.14159
  let perimeter := pi_value * total_radius
  have calculation : perimeter = 18.31 := sorry
  exact calculation

end NUMINAMATH_GPT_composite_shape_perimeter_l2057_205704


namespace NUMINAMATH_GPT_ellipse_product_l2057_205722

noncomputable def AB_CD_product (a b c : ℝ) (h1 : a^2 - b^2 = c^2) (h2 : a + b = 14) : ℝ :=
  2 * a * 2 * b

-- The main statement
theorem ellipse_product (c : ℝ) (h_c : c = 8) (h_diameter : 6 = 6)
  (a b : ℝ) (h1 : a^2 - b^2 = c^2) (h2 : a + b = 14) :
  AB_CD_product a b c h1 h2 = 175 := sorry

end NUMINAMATH_GPT_ellipse_product_l2057_205722


namespace NUMINAMATH_GPT_mike_pull_ups_per_week_l2057_205757

theorem mike_pull_ups_per_week (pull_ups_per_entry entries_per_day days_per_week : ℕ)
  (h1 : pull_ups_per_entry = 2)
  (h2 : entries_per_day = 5)
  (h3 : days_per_week = 7)
  : pull_ups_per_entry * entries_per_day * days_per_week = 70 := 
by
  sorry

end NUMINAMATH_GPT_mike_pull_ups_per_week_l2057_205757


namespace NUMINAMATH_GPT_maximize_f_at_1_5_l2057_205718

noncomputable def f (x: ℝ) : ℝ := -3 * x^2 + 9 * x + 5

theorem maximize_f_at_1_5 : ∀ x: ℝ, f 1.5 ≥ f x := by
  sorry

end NUMINAMATH_GPT_maximize_f_at_1_5_l2057_205718


namespace NUMINAMATH_GPT_greatest_whole_number_satisfies_inequality_l2057_205785

theorem greatest_whole_number_satisfies_inequality : 
  ∃ (x : ℕ), (∀ (y : ℕ), (6 * y - 4 < 5 - 3 * y) → y ≤ x) ∧ x = 0 := 
sorry

end NUMINAMATH_GPT_greatest_whole_number_satisfies_inequality_l2057_205785


namespace NUMINAMATH_GPT_friendships_structure_count_l2057_205714

/-- In a group of 8 individuals, where each person has exactly 3 friends within the group,
there are 420 different ways to structure these friendships. -/
theorem friendships_structure_count : 
  ∃ (structure_count : ℕ), 
    structure_count = 420 ∧ 
    (∀ (G : Fin 8 → Fin 8 → Prop), 
      (∀ i, ∃! (j₁ j₂ j₃ : Fin 8), G i j₁ ∧ G i j₂ ∧ G i j₃) ∧ 
      (∀ i j, G i j → G j i) ∧ 
      (structure_count = 420)) := 
by
  sorry

end NUMINAMATH_GPT_friendships_structure_count_l2057_205714


namespace NUMINAMATH_GPT_find_B_l2057_205767

theorem find_B (A B : ℝ) : (1 / 4 * 1 / 8 = 1 / (4 * A) ∧ 1 / 32 = 1 / B) → B = 32 := by
  intros h
  sorry

end NUMINAMATH_GPT_find_B_l2057_205767


namespace NUMINAMATH_GPT_prime_in_A_l2057_205769

open Nat

def is_in_A (x : ℕ) : Prop :=
  ∃ (a b : ℤ), x = a^2 + 2*b^2 ∧ a * b ≠ 0

theorem prime_in_A (p : ℕ) [Fact (Nat.Prime p)] (h : is_in_A (p^2)) : is_in_A p :=
  sorry

end NUMINAMATH_GPT_prime_in_A_l2057_205769


namespace NUMINAMATH_GPT_evaluate_expression_l2057_205781

theorem evaluate_expression :
  (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l2057_205781


namespace NUMINAMATH_GPT_interest_rate_of_first_account_l2057_205792

theorem interest_rate_of_first_account (r : ℝ) 
  (h1 : 7200 = 4000 + 4000)
  (h2 : 4000 * r = 4000 * 0.10) : 
  r = 0.10 :=
sorry

end NUMINAMATH_GPT_interest_rate_of_first_account_l2057_205792


namespace NUMINAMATH_GPT_minimum_value_of_polynomial_l2057_205732

-- Define the polynomial expression
def polynomial_expr (x : ℝ) : ℝ := (8 - x) * (6 - x) * (8 + x) * (6 + x)

-- State the theorem with the minimum value
theorem minimum_value_of_polynomial : ∃ x : ℝ, polynomial_expr x = -196 := by
  sorry

end NUMINAMATH_GPT_minimum_value_of_polynomial_l2057_205732


namespace NUMINAMATH_GPT_ajay_income_l2057_205737

theorem ajay_income
  (I : ℝ)
  (h₁ : I * 0.45 + I * 0.25 + I * 0.075 + 9000 = I) :
  I = 40000 :=
by
  sorry

end NUMINAMATH_GPT_ajay_income_l2057_205737


namespace NUMINAMATH_GPT_students_with_screws_neq_bolts_l2057_205750

-- Let's define the main entities
def total_students : ℕ := 40
def nails_neq_bolts : ℕ := 15
def screws_eq_nails : ℕ := 10

-- Main theorem statement
theorem students_with_screws_neq_bolts (total : ℕ) (neq_nails_bolts : ℕ) (eq_screws_nails : ℕ) :
  total = 40 → neq_nails_bolts = 15 → eq_screws_nails = 10 → ∃ k, k ≥ 15 ∧ k ≤ 40 - eq_screws_nails - neq_nails_bolts := 
by
  intros
  sorry

end NUMINAMATH_GPT_students_with_screws_neq_bolts_l2057_205750


namespace NUMINAMATH_GPT_find_a_c_l2057_205758

theorem find_a_c (a c : ℝ) (h_discriminant : ∀ x : ℝ, a * x^2 + 10 * x + c = 0 → ∃ k : ℝ, a * k^2 + 10 * k + c = 0 ∧ (a * x^2 + 10 * k + c = 0 → x = k))
  (h_sum : a + c = 12) (h_lt : a < c) : (a, c) = (6 - Real.sqrt 11, 6 + Real.sqrt 11) :=
sorry

end NUMINAMATH_GPT_find_a_c_l2057_205758


namespace NUMINAMATH_GPT_min_value_of_expression_l2057_205741

theorem min_value_of_expression (x y z : ℝ) (h : x > 0 ∧ y > 0 ∧ z > 0) (hxyz : x * y * z = 27) :
  x + 3 * y + 6 * z >= 27 :=
by
  sorry

end NUMINAMATH_GPT_min_value_of_expression_l2057_205741


namespace NUMINAMATH_GPT_tedra_tomato_harvest_l2057_205765

theorem tedra_tomato_harvest (W T F : ℝ) 
    (h1 : T = W / 2) 
    (h2 : W + T + F = 2000) 
    (h3 : F - 700 = 700) : 
    W = 400 := 
sorry

end NUMINAMATH_GPT_tedra_tomato_harvest_l2057_205765


namespace NUMINAMATH_GPT_probability_of_answering_phone_in_4_rings_l2057_205782

/-- A proof statement that asserts the probability of answering the phone within the first four rings is equal to 9/10. -/
theorem probability_of_answering_phone_in_4_rings :
  (1/10) + (3/10) + (2/5) + (1/10) = 9/10 :=
by
  sorry

end NUMINAMATH_GPT_probability_of_answering_phone_in_4_rings_l2057_205782


namespace NUMINAMATH_GPT_volume_of_pyramid_l2057_205799

noncomputable def volume_of_pyramid_QEFGH : ℝ := 
  let EF := 10
  let FG := 3
  let base_area := EF * FG
  let height := 9
  (1/3) * base_area * height

theorem volume_of_pyramid {EF FG : ℝ} (hEF : EF = 10) (hFG : FG = 3)
  (QE_perpendicular_EF : true) (QE_perpendicular_EH : true) (QE_height : QE = 9) :
  volume_of_pyramid_QEFGH = 90 := by
  sorry

end NUMINAMATH_GPT_volume_of_pyramid_l2057_205799


namespace NUMINAMATH_GPT_systematic_sampling_example_l2057_205742

theorem systematic_sampling_example (rows seats : ℕ) (all_seats_filled : Prop) (chosen_seat : ℕ):
  rows = 50 ∧ seats = 60 ∧ all_seats_filled ∧ chosen_seat = 18 → sampling_method = "systematic_sampling" :=
by
  sorry

end NUMINAMATH_GPT_systematic_sampling_example_l2057_205742


namespace NUMINAMATH_GPT_toothpaste_usage_l2057_205763

-- Define the variables involved
variables (t : ℕ) -- total toothpaste in grams
variables (d : ℕ) -- grams used by dad per brushing
variables (m : ℕ) -- grams used by mom per brushing
variables (b : ℕ) -- grams used by Anne + brother per brushing
variables (r : ℕ) -- brushing rate per day
variables (days : ℕ) -- days for toothpaste to run out
variables (N : ℕ) -- family members

-- Given conditions
variables (ht : t = 105)         -- Total toothpaste is 105 grams
variables (hd : d = 3)           -- Dad uses 3 grams per brushing
variables (hm : m = 2)           -- Mom uses 2 grams per brushing
variables (hr : r = 3)           -- Each member brushes three times a day
variables (hdays : days = 5)     -- Toothpaste runs out in 5 days

-- Additional calculations
variable (total_brushing : ℕ)
variable (total_usage_d: ℕ)
variable (total_usage_m: ℕ)
variable (total_usage_parents: ℕ)
variable (total_usage_family: ℕ)

-- Helper expressions
def total_brushing_expr := days * r * 2
def total_usage_d_expr := d * r
def total_usage_m_expr := m * r
def total_usage_parents_expr := (total_usage_d_expr + total_usage_m_expr) * days
def total_usage_family_expr := t - total_usage_parents_expr

-- Assume calculations
variables (h1: total_usage_d = total_usage_d_expr)  
variables (h2: total_usage_m = total_usage_m_expr)
variables (h3: total_usage_parents = total_usage_parents_expr)
variables (h4: total_usage_family = total_usage_family_expr)
variables (h5 : total_brushing = total_brushing_expr)

-- Define the proof
theorem toothpaste_usage : 
  b = total_usage_family / total_brushing := 
  sorry

end NUMINAMATH_GPT_toothpaste_usage_l2057_205763


namespace NUMINAMATH_GPT_find_minimum_value_l2057_205794

theorem find_minimum_value (a : ℝ) (x : ℝ) (h1 : a > 0) (h2 : x > 0): 
  (∃ m : ℝ, ∀ x > 0, (a^2 + x^2) / x ≥ m ∧ ∃ x₀ > 0, (a^2 + x₀^2) / x₀ = m) :=
sorry

end NUMINAMATH_GPT_find_minimum_value_l2057_205794


namespace NUMINAMATH_GPT_range_of_f_l2057_205731

noncomputable def f (x : ℝ) : ℝ := 4^x - 2^(x + 1) + 2

theorem range_of_f (h : ∀ x : ℝ, x ≤ 1) : (f '' {x : ℝ | x ≤ 1}) = {y : ℝ | 1 ≤ y ∧ y ≤ 2} :=
by
  sorry

end NUMINAMATH_GPT_range_of_f_l2057_205731


namespace NUMINAMATH_GPT_alice_speed_is_6_5_l2057_205795

-- Definitions based on the conditions.
def a : ℝ := sorry -- Alice's speed
def b : ℝ := a + 3 -- Bob's speed

-- Alice cycles towards the park 80 miles away and Bob meets her 15 miles away from the park
def d_alice : ℝ := 65 -- Alice's distance traveled (80 - 15)
def d_bob : ℝ := 95 -- Bob's distance traveled (80 + 15)

-- Equating the times
def time_eqn := d_alice / a = d_bob / b

-- Alice's speed is 6.5 mph
theorem alice_speed_is_6_5 : a = 6.5 :=
by
  have h1 : b = a + 3 := sorry
  have h2 : a * 65 = (a + 3) * 95 := sorry
  have h3 : 30 * a = 195 := sorry
  have h4 : a = 6.5 := sorry
  exact h4

end NUMINAMATH_GPT_alice_speed_is_6_5_l2057_205795


namespace NUMINAMATH_GPT_average_is_207_l2057_205793

variable (x : ℕ)

theorem average_is_207 (h : (201 + 202 + 204 + 205 + 206 + 209 + 209 + 210 + 212 + x) / 10 = 207) :
  x = 212 :=
sorry

end NUMINAMATH_GPT_average_is_207_l2057_205793


namespace NUMINAMATH_GPT_find_k_l2057_205744

theorem find_k (x y k : ℝ) (h₁ : 3 * x + y = k) (h₂ : -1.2 * x + y = -20) (hx : x = 7) : k = 9.4 :=
by
  sorry

end NUMINAMATH_GPT_find_k_l2057_205744


namespace NUMINAMATH_GPT_sequence_divisible_by_13_l2057_205738

theorem sequence_divisible_by_13 (n : ℕ) (h : n ≤ 1000) : 
  ∃ m, m = 165 ∧ ∀ k, 1 ≤ k ∧ k ≤ m → (10^(6*k) + 1) % 13 = 0 := 
sorry

end NUMINAMATH_GPT_sequence_divisible_by_13_l2057_205738


namespace NUMINAMATH_GPT_find_m_plus_n_l2057_205717

-- Definitions
structure Triangle (A B C P M N : Type) :=
  (midpoint_AD_P : P)
  (intersection_M_AB : M)
  (intersection_N_AC : N)
  (vec_AB : ℝ)
  (vec_AM : ℝ)
  (vec_AC : ℝ)
  (vec_AN : ℝ)
  (m : ℝ)
  (n : ℝ)
  (AB_eq_AM_mul_m : vec_AB = m * vec_AM)
  (AC_eq_AN_mul_n : vec_AC = n * vec_AN)

-- The theorem to prove
theorem find_m_plus_n (A B C P M N : Type)
  (t : Triangle A B C P M N) :
  t.m + t.n = 4 :=
sorry

end NUMINAMATH_GPT_find_m_plus_n_l2057_205717


namespace NUMINAMATH_GPT_find_a_l2057_205798

theorem find_a (r s : ℚ) (a : ℚ) :
  (∀ x : ℚ, (ax^2 + 18 * x + 16 = (r * x + s)^2)) → 
  s = 4 ∨ s = -4 →
  a = (9 / 4) * (9 / 4)
:= sorry

end NUMINAMATH_GPT_find_a_l2057_205798


namespace NUMINAMATH_GPT_right_angled_triangle_count_in_pyramid_l2057_205747

-- Define the cuboid and the triangular pyramid within it
variables (A B C D A₁ B₁ C₁ D₁ : Type)

-- Assume there exists a cuboid ABCD-A₁B₁C₁D₁
axiom cuboid : Prop

-- Define the triangular pyramid A₁-ABC
structure triangular_pyramid (A₁ A B C : Type) : Type :=
  (vertex₁ : A₁)
  (vertex₂ : A)
  (vertex₃ : B)
  (vertex4 : C)
  
-- The mathematical statement to prove: the number of right-angled triangles in A₁-ABC is 4
theorem right_angled_triangle_count_in_pyramid (A : Type) (B : Type) (C : Type) (A₁ : Type)
  (h_pyramid : triangular_pyramid A₁ A B C) (h_cuboid : cuboid) :
  ∃ n : ℕ, n = 4 :=
by
  sorry

end NUMINAMATH_GPT_right_angled_triangle_count_in_pyramid_l2057_205747


namespace NUMINAMATH_GPT_juice_difference_is_eight_l2057_205723

-- Defining the initial conditions
def initial_large_barrel : ℕ := 10
def initial_small_barrel : ℕ := 8
def poured_juice : ℕ := 3

-- Defining the final amounts
def final_large_barrel : ℕ := initial_large_barrel + poured_juice
def final_small_barrel : ℕ := initial_small_barrel - poured_juice

-- The statement we need to prove
theorem juice_difference_is_eight :
  final_large_barrel - final_small_barrel = 8 :=
by
  -- Skipping the proof
  sorry

end NUMINAMATH_GPT_juice_difference_is_eight_l2057_205723


namespace NUMINAMATH_GPT_carol_betty_age_ratio_l2057_205706

theorem carol_betty_age_ratio:
  ∀ (C A B : ℕ), 
    C = 5 * A → 
    A = C - 12 → 
    B = 6 → 
    C / B = 5 / 2 :=
by
  intros C A B h1 h2 h3
  sorry

end NUMINAMATH_GPT_carol_betty_age_ratio_l2057_205706


namespace NUMINAMATH_GPT_find_number_of_terms_l2057_205777

variable {n : ℕ} {a : ℕ → ℤ}
variable (a_seq : ℕ → ℤ)

def sum_first_three_terms (a : ℕ → ℤ) : ℤ :=
  a 1 + a 2 + a 3

def sum_last_three_terms (n : ℕ) (a : ℕ → ℤ) : ℤ :=
  a (n-2) + a (n-1) + a n

def sum_all_terms (n : ℕ) (a : ℕ → ℤ) : ℤ :=
  (Finset.range n).sum a

theorem find_number_of_terms (h1 : sum_first_three_terms a_seq = 20)
    (h2 : sum_last_three_terms n a_seq = 130)
    (h3 : sum_all_terms n a_seq = 200) : n = 8 :=
sorry

end NUMINAMATH_GPT_find_number_of_terms_l2057_205777


namespace NUMINAMATH_GPT_inradius_of_triangle_l2057_205790

theorem inradius_of_triangle (A p s r : ℝ) (h1 : A = 2 * p) (h2 : A = r * s) (h3 : p = 2 * s) : r = 4 :=
by
  sorry

end NUMINAMATH_GPT_inradius_of_triangle_l2057_205790


namespace NUMINAMATH_GPT_d_minus_r_eq_15_l2057_205760

theorem d_minus_r_eq_15 (d r : ℤ) (h_d_gt_1 : d > 1)
  (h1 : 1059 % d = r)
  (h2 : 1417 % d = r)
  (h3 : 2312 % d = r) :
  d - r = 15 :=
sorry

end NUMINAMATH_GPT_d_minus_r_eq_15_l2057_205760


namespace NUMINAMATH_GPT_noncongruent_triangles_count_l2057_205700

/-- Prove the number of noncongruent integer-sided triangles with positive area,
    perimeter less than 20, that are neither equilateral, isosceles, nor right triangles
    is 17 -/
theorem noncongruent_triangles_count:
  ∃ (s : Finset (ℕ × ℕ × ℕ)), 
    (∀ (a b c : ℕ), (a, b, c) ∈ s → a + b + c < 20 ∧ a + b > c ∧ a < b ∧ b < c ∧ 
         ¬(a = b ∨ b = c ∨ a = c) ∧ ¬(a * a + b * b = c * c)) ∧ 
    s.card = 17 := 
sorry

end NUMINAMATH_GPT_noncongruent_triangles_count_l2057_205700


namespace NUMINAMATH_GPT_find_m_for_asymptotes_l2057_205708

-- Definition of the hyperbola
def hyperbola (x y : ℝ) : Prop :=
  (y - 1)^2 / 16 - (x + 2)^2 / 9 = 1

-- Definition of the asymptotes form
def asymptote_form (m : ℝ) (x y : ℝ) : Prop :=
  y - 1 = m * x + 2 * m ∨ y - 1 = -m * x - 2 * m

-- The main theorem to prove
theorem find_m_for_asymptotes :
  (∀ x y : ℝ, hyperbola x y → asymptote_form (4 / 3) x y) :=
sorry

end NUMINAMATH_GPT_find_m_for_asymptotes_l2057_205708


namespace NUMINAMATH_GPT_beach_ball_problem_l2057_205749

noncomputable def change_in_radius (C₁ C₂ : ℝ) : ℝ := (C₂ - C₁) / (2 * Real.pi)

noncomputable def volume (r : ℝ) : ℝ := (4 / 3) * Real.pi * r ^ 3

noncomputable def percentage_increase_in_volume (V₁ V₂ : ℝ) : ℝ := (V₂ - V₁) / V₁ * 100

theorem beach_ball_problem (C₁ C₂ : ℝ) (hC₁ : C₁ = 30) (hC₂ : C₂ = 36) :
  change_in_radius C₁ C₂ = 3 / Real.pi ∧
  percentage_increase_in_volume (volume (C₁ / (2 * Real.pi))) (volume (C₂ / (2 * Real.pi))) = 72.78 :=
by
  sorry

end NUMINAMATH_GPT_beach_ball_problem_l2057_205749


namespace NUMINAMATH_GPT_tunnel_length_scale_l2057_205733

theorem tunnel_length_scale (map_length_cm : ℝ) (scale_ratio : ℝ) (convert_factor : ℝ) : 
  map_length_cm = 7 → scale_ratio = 38000 → convert_factor = 100000 →
  (map_length_cm * scale_ratio / convert_factor) = 2.66 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end NUMINAMATH_GPT_tunnel_length_scale_l2057_205733


namespace NUMINAMATH_GPT_trigonometric_expression_l2057_205764

open Real

theorem trigonometric_expression (α β : ℝ) (h : cos α ^ 2 = cos β ^ 2) :
  (sin β ^ 2 / sin α + cos β ^ 2 / cos α = sin α + cos α ∨ sin β ^ 2 / sin α + cos β ^ 2 / cos α = -sin α + cos α) :=
sorry

end NUMINAMATH_GPT_trigonometric_expression_l2057_205764


namespace NUMINAMATH_GPT_greatest_non_sum_complex_l2057_205726

def is_complex (n : ℕ) : Prop :=
  ∃ p q : ℕ, p ≠ q ∧ Nat.Prime p ∧ Nat.Prime q ∧ p ∣ n ∧ q ∣ n

theorem greatest_non_sum_complex : ∀ n : ℕ, (¬ ∃ a b : ℕ, is_complex a ∧ is_complex b ∧ a + b = n) → n ≤ 23 :=
by {
  sorry
}

end NUMINAMATH_GPT_greatest_non_sum_complex_l2057_205726


namespace NUMINAMATH_GPT_larger_integer_value_l2057_205743

theorem larger_integer_value
  (a b : ℕ)
  (h1 : a ≥ b)
  (h2 : ↑a / ↑b = 7 / 3)
  (h3 : a * b = 294) :
  a = 7 * Int.sqrt 14 := 
sorry

end NUMINAMATH_GPT_larger_integer_value_l2057_205743


namespace NUMINAMATH_GPT_find_BC_l2057_205754

variable (A B C : Type)
variables (a b : ℝ) -- Angles
variables (AB BC CA : ℝ) -- Sides of the triangle

-- Given conditions:
-- 1: Triangle ABC
-- 2: cos(a - b) + sin(a + b) = 2
-- 3: AB = 4

theorem find_BC (hAB : AB = 4) (hTrig : Real.cos (a - b) + Real.sin (a + b) = 2) :
  BC = 2 * Real.sqrt 2 := 
sorry

end NUMINAMATH_GPT_find_BC_l2057_205754


namespace NUMINAMATH_GPT_isosceles_triangle_vertex_angle_l2057_205783

theorem isosceles_triangle_vertex_angle (exterior_angle : ℝ) (h1 : exterior_angle = 40) : 
  ∃ vertex_angle : ℝ, vertex_angle = 140 :=
by
  sorry

end NUMINAMATH_GPT_isosceles_triangle_vertex_angle_l2057_205783


namespace NUMINAMATH_GPT_time_saved_is_35_minutes_l2057_205779

-- Define the speed and distances for each day
def monday_distance := 3
def wednesday_distance := 3
def friday_distance := 3
def sunday_distance := 4
def speed_monday := 6
def speed_wednesday := 4
def speed_friday := 5
def speed_sunday := 3
def speed_uniform := 5

-- Calculate the total time spent on the treadmill originally
def time_monday := monday_distance / speed_monday
def time_wednesday := wednesday_distance / speed_wednesday
def time_friday := friday_distance / speed_friday
def time_sunday := sunday_distance / speed_sunday
def total_time := time_monday + time_wednesday + time_friday + time_sunday

-- Calculate the total time if speed was uniformly 5 mph 
def total_distance := monday_distance + wednesday_distance + friday_distance + sunday_distance
def total_time_uniform := total_distance / speed_uniform

-- Time saved if walking at 5 mph every day
def time_saved := total_time - total_time_uniform

-- Convert time saved to minutes
def minutes_saved := time_saved * 60

theorem time_saved_is_35_minutes : minutes_saved = 35 := by
  sorry

end NUMINAMATH_GPT_time_saved_is_35_minutes_l2057_205779


namespace NUMINAMATH_GPT_intersection_of_complements_l2057_205791

open Set

theorem intersection_of_complements (U : Set ℕ) (A B : Set ℕ)
  (hU : U = {1,2,3,4,5,6,7,8})
  (hA : A = {3,4,5})
  (hB : B = {1,3,6}) :
  (U \ A) ∩ (U \ B) = {2,7,8} := by
  rw [hU, hA, hB]
  sorry

end NUMINAMATH_GPT_intersection_of_complements_l2057_205791


namespace NUMINAMATH_GPT_twelve_people_pairing_l2057_205780

noncomputable def num_ways_to_pair : ℕ := sorry

theorem twelve_people_pairing :
  (∀ (n : ℕ), n = 12 → (∃ f : ℕ → ℕ, ∀ i, f i = 2 ∨ f i = 12 ∨ f i = 7) → num_ways_to_pair = 3) := 
sorry

end NUMINAMATH_GPT_twelve_people_pairing_l2057_205780


namespace NUMINAMATH_GPT_max_imaginary_part_of_roots_l2057_205734

noncomputable def find_phi : Prop :=
  ∃ z : ℂ, z^6 - z^4 + z^2 - 1 = 0 ∧ (∀ w : ℂ, w^6 - w^4 + w^2 - 1 = 0 → z.im ≤ w.im) ∧ z.im = Real.sin (Real.pi / 4)

theorem max_imaginary_part_of_roots : find_phi :=
sorry

end NUMINAMATH_GPT_max_imaginary_part_of_roots_l2057_205734


namespace NUMINAMATH_GPT_sum_of_integers_sqrt_485_l2057_205753

theorem sum_of_integers_sqrt_485 (x y : ℕ) (h1 : x^2 + y^2 = 245) (h2 : x * y = 120) : x + y = Real.sqrt 485 :=
sorry

end NUMINAMATH_GPT_sum_of_integers_sqrt_485_l2057_205753


namespace NUMINAMATH_GPT_positive_difference_of_two_numbers_l2057_205759

theorem positive_difference_of_two_numbers (x y : ℝ) (h1 : x + y = 10) (h2 : x^2 - y^2 = 40) : |x - y| = 4 :=
sorry

end NUMINAMATH_GPT_positive_difference_of_two_numbers_l2057_205759


namespace NUMINAMATH_GPT_number_of_short_trees_to_plant_l2057_205775

-- Definitions of the conditions
def current_short_trees : ℕ := 41
def current_tall_trees : ℕ := 44
def total_short_trees_after_planting : ℕ := 98

-- The statement to be proved
theorem number_of_short_trees_to_plant :
  total_short_trees_after_planting - current_short_trees = 57 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_number_of_short_trees_to_plant_l2057_205775


namespace NUMINAMATH_GPT_tetrahedron_volume_l2057_205766

theorem tetrahedron_volume (R S1 S2 S3 S4 : ℝ) : 
    V = (1 / 3) * R * (S1 + S2 + S3 + S4) :=
sorry

end NUMINAMATH_GPT_tetrahedron_volume_l2057_205766


namespace NUMINAMATH_GPT_tan_add_formula_l2057_205735

noncomputable def tan_subtract (a b : ℝ) : ℝ := (Real.tan a - Real.tan b) / (1 + Real.tan a * Real.tan b)
noncomputable def tan_add (a b : ℝ) : ℝ := (Real.tan a + Real.tan b) / (1 - Real.tan a * Real.tan b)

theorem tan_add_formula (α : ℝ) (hf : tan_subtract α (Real.pi / 4) = 1 / 4) :
  tan_add α (Real.pi / 4) = -4 :=
by
  sorry

end NUMINAMATH_GPT_tan_add_formula_l2057_205735


namespace NUMINAMATH_GPT_second_differences_of_cubes_l2057_205787

-- Define the first difference for cubes of consecutive natural numbers
def first_difference (n : ℕ) : ℕ :=
  ((n + 1) ^ 3) - (n ^ 3)

-- Define the second difference for the first differences
def second_difference (n : ℕ) : ℕ :=
  first_difference (n + 1) - first_difference n

-- Proof statement: Prove that second differences are equal to 6n + 6
theorem second_differences_of_cubes (n : ℕ) : second_difference n = 6 * n + 6 :=
  sorry

end NUMINAMATH_GPT_second_differences_of_cubes_l2057_205787


namespace NUMINAMATH_GPT_geometric_sequence_a6_l2057_205771

theorem geometric_sequence_a6 (a : ℕ → ℝ) 
  (h1 : a 4 * a 8 = 9) 
  (h2 : a 4 + a 8 = 8) 
  (geom_seq : ∀ n m, a (n + m) = a n * a m): 
  a 6 = 3 :=
by
  -- skipped proof
  sorry

end NUMINAMATH_GPT_geometric_sequence_a6_l2057_205771


namespace NUMINAMATH_GPT_minimum_employees_needed_l2057_205796

noncomputable def employees_needed (total_days : ℕ) (work_days : ℕ) (rest_days : ℕ) (min_on_duty : ℕ) : ℕ :=
  let comb := (total_days.choose rest_days)
  min_on_duty * comb / work_days

theorem minimum_employees_needed {total_days work_days rest_days min_on_duty : ℕ} (h_total_days: total_days = 7) (h_work_days: work_days = 5) (h_rest_days: rest_days = 2) (h_min_on_duty: min_on_duty = 45) : 
  employees_needed total_days work_days rest_days min_on_duty = 63 := by
  rw [h_total_days, h_work_days, h_rest_days, h_min_on_duty]
  -- detailed computation and proofs steps omitted
  -- the critical part is to ensure 63 is derived correctly based on provided values
  sorry

end NUMINAMATH_GPT_minimum_employees_needed_l2057_205796


namespace NUMINAMATH_GPT_fencing_required_l2057_205755

theorem fencing_required
  (L : ℝ) (W : ℝ) (A : ℝ) (F : ℝ)
  (hL : L = 25)
  (hA : A = 880)
  (hArea : A = L * W)
  (hF : F = L + 2 * W) :
  F = 95.4 :=
by
  sorry

end NUMINAMATH_GPT_fencing_required_l2057_205755


namespace NUMINAMATH_GPT_biggest_number_in_ratio_l2057_205770

theorem biggest_number_in_ratio (A B C D : ℕ) (h1 : 2 * D = 5 * A) (h2 : 3 * D = 5 * B) (h3 : 4 * D = 5 * C) (h_sum : A + B + C + D = 1344) : D = 480 := 
by
  sorry

end NUMINAMATH_GPT_biggest_number_in_ratio_l2057_205770


namespace NUMINAMATH_GPT_race_result_130m_l2057_205724

theorem race_result_130m (d : ℕ) (t_a t_b: ℕ) (a_speed b_speed : ℚ) (d_a_t : ℚ) (d_b_t : ℚ) (distance_covered_by_B_in_20_secs : ℚ) :
  d = 130 →
  t_a = 20 →
  t_b = 25 →
  a_speed = (↑d) / t_a →
  b_speed = (↑d) / t_b →
  d_a_t = a_speed * t_a →
  d_b_t = b_speed * t_b →
  distance_covered_by_B_in_20_secs = b_speed * 20 →
  (d - distance_covered_by_B_in_20_secs = 26) :=
by
  sorry

end NUMINAMATH_GPT_race_result_130m_l2057_205724


namespace NUMINAMATH_GPT_cyclist_north_speed_l2057_205707

variable {v : ℝ} -- Speed of the cyclist going north.

-- Conditions: 
def speed_south := 15 -- Speed of the cyclist going south (15 kmph).
def time := 2 -- The time after which they are 50 km apart (2 hours).
def distance := 50 -- The distance they are apart after 2 hours (50 km).

-- Theorem statement:
theorem cyclist_north_speed :
    (v + speed_south) * time = distance → v = 10 := by
  intro h
  sorry

end NUMINAMATH_GPT_cyclist_north_speed_l2057_205707


namespace NUMINAMATH_GPT_point_in_second_quadrant_coordinates_l2057_205773

theorem point_in_second_quadrant_coordinates (a : ℤ) (h1 : a + 1 < 0) (h2 : 2 * a + 6 > 0) :
  (a + 1, 2 * a + 6) = (-1, 2) :=
sorry

end NUMINAMATH_GPT_point_in_second_quadrant_coordinates_l2057_205773


namespace NUMINAMATH_GPT_initial_total_packs_l2057_205772

def initial_packs (total_packs : ℕ) (regular_packs : ℕ) (unusual_packs : ℕ) (excellent_packs : ℕ) : Prop :=
  total_packs = regular_packs + unusual_packs + excellent_packs

def ratio_packs (regular_packs : ℕ) (unusual_packs : ℕ) (excellent_packs : ℕ) : Prop :=
  3 * (regular_packs + unusual_packs + excellent_packs) = 3 * regular_packs + 4 * unusual_packs + 6 * excellent_packs

def new_ratios (regular_packs : ℕ) (unusual_packs : ℕ) (excellent_packs : ℕ) (new_regular_packs : ℕ) (new_unusual_packs : ℕ) (new_excellent_packs : ℕ) : Prop :=
  2 * (new_regular_packs) + 5 * (new_unusual_packs) + 8 * (new_excellent_packs) = regular_packs + unusual_packs + excellent_packs + 8 * (regular_packs)

def pack_changes (initial_regular_packs : ℕ) (initial_unusual_packs : ℕ) (initial_excellent_packs : ℕ) (new_regular_packs : ℕ) (new_unusual_packs : ℕ) (new_excellent_packs : ℕ) : Prop :=
  initial_excellent_packs <= new_excellent_packs + 80 ∧ initial_regular_packs - new_regular_packs ≤ 10

theorem initial_total_packs (total_packs : ℕ) (regular_packs : ℕ) (unusual_packs : ℕ) (excellent_packs : ℕ) 
(new_regular_packs : ℕ) (new_unusual_packs : ℕ) (new_excellent_packs : ℕ) :
  initial_packs total_packs regular_packs unusual_packs excellent_packs ∧
  ratio_packs regular_packs unusual_packs excellent_packs ∧ 
  new_ratios regular_packs unusual_packs excellent_packs new_regular_packs new_unusual_packs new_excellent_packs ∧ 
  pack_changes regular_packs unusual_packs excellent_packs new_regular_packs new_unusual_packs new_excellent_packs 
  → total_packs = 260 := 
sorry

end NUMINAMATH_GPT_initial_total_packs_l2057_205772


namespace NUMINAMATH_GPT_kenneth_earnings_l2057_205788

theorem kenneth_earnings (E : ℝ) (h1 : E - 0.1 * E = 405) : E = 450 :=
sorry

end NUMINAMATH_GPT_kenneth_earnings_l2057_205788


namespace NUMINAMATH_GPT_cone_apex_angle_l2057_205702

theorem cone_apex_angle (R : ℝ) 
  (h1 : ∀ (θ : ℝ), (∃ (r : ℝ), r = R / 2 ∧ 2 * π * r = π * R)) :
  ∀ (θ : ℝ), θ = π / 3 :=
by
  sorry

end NUMINAMATH_GPT_cone_apex_angle_l2057_205702


namespace NUMINAMATH_GPT_sequence_an_correct_l2057_205728

noncomputable def seq_an (n : ℕ) : ℚ :=
if h : n = 1 then 1 else (1 / (2 * n - 1) - 1 / (2 * n - 3))

theorem sequence_an_correct (n : ℕ) (S : ℕ → ℚ)
  (h1 : S 1 = 1)
  (h2 : ∀ n ≥ 2, S n ^ 2 = seq_an n * (S n - 0.5)) :
  seq_an n = if n = 1 then 1 else (1 / (2 * n - 1) - 1 / (2 * n - 3)) :=
sorry

end NUMINAMATH_GPT_sequence_an_correct_l2057_205728


namespace NUMINAMATH_GPT_caps_percentage_l2057_205786

open Real

-- Define the conditions as given in part (a)
def total_caps : ℝ := 575
def red_caps : ℝ := 150
def green_caps : ℝ := 120
def blue_caps : ℝ := 175
def yellow_caps : ℝ := total_caps - (red_caps + green_caps + blue_caps)

-- Define the problem asking for the percentages of each color and proving the answer
theorem caps_percentage :
  (red_caps / total_caps) * 100 = 26.09 ∧
  (green_caps / total_caps) * 100 = 20.87 ∧
  (blue_caps / total_caps) * 100 = 30.43 ∧
  (yellow_caps / total_caps) * 100 = 22.61 :=
by
  -- proof steps would go here
  sorry

end NUMINAMATH_GPT_caps_percentage_l2057_205786


namespace NUMINAMATH_GPT_count_N_less_than_2000_l2057_205736

noncomputable def count_valid_N (N : ℕ) : Prop :=
  ∃ x : ℝ, x > 0 ∧ x < 2000 ∧ N < 2000 ∧ x ^ (⌊x⌋₊) = N

theorem count_N_less_than_2000 : 
  ∃ (count : ℕ), count = 412 ∧ 
  (∀ (N : ℕ), N < 2000 → (∃ x : ℝ, x > 0 ∧ x < 2000 ∧ x ^ (⌊x⌋₊) = N) ↔ N < count) :=
sorry

end NUMINAMATH_GPT_count_N_less_than_2000_l2057_205736


namespace NUMINAMATH_GPT_incorrect_inequality_transformation_l2057_205784

theorem incorrect_inequality_transformation 
    (a b : ℝ) 
    (h : a > b) 
    : ¬(1 - a > 1 - b) := 
by {
  sorry 
}

end NUMINAMATH_GPT_incorrect_inequality_transformation_l2057_205784


namespace NUMINAMATH_GPT_additional_savings_zero_l2057_205748

noncomputable def windows_savings (purchase_price : ℕ) (free_windows : ℕ) (paid_windows : ℕ)
  (dave_needs : ℕ) (doug_needs : ℕ) : ℕ := sorry

theorem additional_savings_zero :
  windows_savings 100 2 5 12 10 = 0 := sorry

end NUMINAMATH_GPT_additional_savings_zero_l2057_205748


namespace NUMINAMATH_GPT_smallest_repunit_divisible_by_97_l2057_205761

theorem smallest_repunit_divisible_by_97 :
  ∃ n : ℕ, (∃ d : ℤ, 10^n - 1 = 97 * 9 * d) ∧ (∀ m : ℕ, (∃ d : ℤ, 10^m - 1 = 97 * 9 * d) → n ≤ m) :=
by
  sorry

end NUMINAMATH_GPT_smallest_repunit_divisible_by_97_l2057_205761


namespace NUMINAMATH_GPT_question1_solution_question2_solution_l2057_205729

-- Definitions of the problem conditions
def f (x a : ℝ) : ℝ := abs (x - a)

-- First proof problem (Question 1)
theorem question1_solution (x : ℝ) : (f x 2) ≥ (4 - abs (x - 4)) ↔ (x ≥ 5 ∨ x ≤ 1) :=
by sorry

-- Second proof problem (Question 2)
theorem question2_solution (x : ℝ) (a : ℝ) (h_sol : 1 ≤ x ∧ x ≤ 2) 
  (h_ineq : abs (f (2 * x + a) a - 2 * f x a) ≤ 2) : a = 3 :=
by sorry

end NUMINAMATH_GPT_question1_solution_question2_solution_l2057_205729


namespace NUMINAMATH_GPT_jump_length_third_frog_l2057_205740

theorem jump_length_third_frog (A B C : ℝ) 
  (h1 : A = (B + C) / 2) 
  (h2 : B = (A + C) / 2) 
  (h3 : |B - A| + |(B - C) / 2| = 60) : 
  |C - (A + B) / 2| = 30 :=
sorry

end NUMINAMATH_GPT_jump_length_third_frog_l2057_205740


namespace NUMINAMATH_GPT_find_sum_uv_l2057_205776

theorem find_sum_uv (u v : ℝ) (h1 : 3 * u - 7 * v = 29) (h2 : 5 * u + 3 * v = -9) : u + v = -3.363 := 
sorry

end NUMINAMATH_GPT_find_sum_uv_l2057_205776


namespace NUMINAMATH_GPT_triangle_angle_y_l2057_205720

theorem triangle_angle_y (y : ℝ) (h : y + 3 * y + 45 = 180) : y = 33.75 :=
by
  have h1 : 4 * y + 45 = 180 := by sorry
  have h2 : 4 * y = 135 := by sorry
  have h3 : y = 33.75 := by sorry
  exact h3

end NUMINAMATH_GPT_triangle_angle_y_l2057_205720


namespace NUMINAMATH_GPT_inequality_proof_l2057_205715

theorem inequality_proof (a b c : ℝ) (h1 : a ≥ b) (h2 : b ≥ c) (h3 : c > 0) :
  (a - b + c) * (1 / a - 1 / b + 1 / c) ≥ 1 :=
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l2057_205715


namespace NUMINAMATH_GPT_Jim_weekly_savings_l2057_205778

-- Define the given conditions
def Sara_initial_savings : ℕ := 4100
def Sara_weekly_savings : ℕ := 10
def weeks : ℕ := 820

-- Define the proof goal based on the conditions
theorem Jim_weekly_savings :
  let Sara_total_savings := Sara_initial_savings + (Sara_weekly_savings * weeks)
  let Jim_weekly_savings := Sara_total_savings / weeks
  Jim_weekly_savings = 15 := 
by 
  sorry

end NUMINAMATH_GPT_Jim_weekly_savings_l2057_205778


namespace NUMINAMATH_GPT_fish_weight_l2057_205752

-- Definitions of weights
variable (T B H : ℝ)

-- Given conditions
def cond1 : Prop := T = 9
def cond2 : Prop := H = T + (1/2) * B
def cond3 : Prop := B = H + T

-- Theorem to prove
theorem fish_weight (h1 : cond1 T) (h2 : cond2 T B H) (h3 : cond3 T B H) :
  T + B + H = 72 :=
by
  sorry

end NUMINAMATH_GPT_fish_weight_l2057_205752


namespace NUMINAMATH_GPT_square_side_length_l2057_205701

-- Variables for the conditions
variables (totalWire triangleWire : ℕ)
-- Definitions of the conditions
def totalLengthCondition := totalWire = 78
def triangleLengthCondition := triangleWire = 46

-- Goal is to prove the side length of the square
theorem square_side_length
  (h1 : totalLengthCondition totalWire)
  (h2 : triangleLengthCondition triangleWire)
  : (totalWire - triangleWire) / 4 = 8 := 
by
  rw [totalLengthCondition, triangleLengthCondition] at *
  sorry

end NUMINAMATH_GPT_square_side_length_l2057_205701


namespace NUMINAMATH_GPT_original_number_is_509_l2057_205713

theorem original_number_is_509 (n : ℕ) (h : n - 5 = 504) : n = 509 :=
by {
    sorry
}

end NUMINAMATH_GPT_original_number_is_509_l2057_205713


namespace NUMINAMATH_GPT_intersection_eq_T_l2057_205703

noncomputable def S : Set ℝ := { y : ℝ | ∃ x : ℝ, y = 3 * x + 2 }
noncomputable def T : Set ℝ := { y : ℝ | ∃ x : ℝ, y = x ^ 2 - 1 }

theorem intersection_eq_T : S ∩ T = T := 
by 
  sorry

end NUMINAMATH_GPT_intersection_eq_T_l2057_205703


namespace NUMINAMATH_GPT_bracket_mul_l2057_205709

def bracket (x : ℕ) : ℕ :=
  if x % 2 = 0 then x / 2 + 1 else 2 * x + 1

theorem bracket_mul : bracket 6 * bracket 3 = 28 := by
  sorry

end NUMINAMATH_GPT_bracket_mul_l2057_205709


namespace NUMINAMATH_GPT_number_of_sets_of_positive_integers_l2057_205789

theorem number_of_sets_of_positive_integers : 
  ∃ n : ℕ, n = 3333 ∧ ∀ x y z : ℕ, x > 0 → y > 0 → z > 0 → x < y → y < z → x + y + z = 203 → n = 3333 :=
by
  sorry

end NUMINAMATH_GPT_number_of_sets_of_positive_integers_l2057_205789


namespace NUMINAMATH_GPT_gary_has_left_amount_l2057_205719

def initial_amount : ℝ := 100
def cost_pet_snake : ℝ := 55
def cost_toy_car : ℝ := 12
def cost_novel : ℝ := 7.5
def cost_pack_stickers : ℝ := 3.25
def number_packs_stickers : ℕ := 3

theorem gary_has_left_amount : initial_amount - (cost_pet_snake + cost_toy_car + cost_novel + number_packs_stickers * cost_pack_stickers) = 15.75 :=
by
  sorry

end NUMINAMATH_GPT_gary_has_left_amount_l2057_205719


namespace NUMINAMATH_GPT_bob_daily_work_hours_l2057_205727

theorem bob_daily_work_hours
  (total_hours_in_month : ℕ)
  (days_per_week : ℕ)
  (weeks_per_month : ℕ)
  (total_working_days : ℕ)
  (daily_working_hours : ℕ)
  (h1 : total_hours_in_month = 200)
  (h2 : days_per_week = 5)
  (h3 : weeks_per_month = 4)
  (h4 : total_working_days = days_per_week * weeks_per_month)
  (h5 : daily_working_hours = total_hours_in_month / total_working_days) :
  daily_working_hours = 10 := 
sorry

end NUMINAMATH_GPT_bob_daily_work_hours_l2057_205727


namespace NUMINAMATH_GPT_ratio_of_girls_who_like_bb_to_boys_dont_like_bb_l2057_205739

-- Definitions based on the given conditions.
def total_students : ℕ := 25
def percent_girls : ℕ := 60
def percent_boys_like_bb : ℕ := 40
def percent_girls_like_bb : ℕ := 80

-- Results from those conditions.
def num_girls : ℕ := percent_girls * total_students / 100
def num_boys : ℕ := total_students - num_girls
def num_boys_like_bb : ℕ := percent_boys_like_bb * num_boys / 100
def num_boys_dont_like_bb : ℕ := num_boys - num_boys_like_bb
def num_girls_like_bb : ℕ := percent_girls_like_bb * num_girls / 100

-- Proof Problem Statement
theorem ratio_of_girls_who_like_bb_to_boys_dont_like_bb :
  (num_girls_like_bb : ℕ) / num_boys_dont_like_bb = 2 / 1 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_girls_who_like_bb_to_boys_dont_like_bb_l2057_205739


namespace NUMINAMATH_GPT_area_of_lune_l2057_205721

theorem area_of_lune :
  ∃ (A L : ℝ), A = (3/2) ∧ L = 2 ∧
  (Lune_area : ℝ) = (9 * Real.sqrt 3 / 4) - (55 * π / 24) →
  Lune_area = (9 * Real.sqrt 3 / 4) - (55 * π / 24) :=
by
  sorry

end NUMINAMATH_GPT_area_of_lune_l2057_205721


namespace NUMINAMATH_GPT_conditions_for_a_and_b_l2057_205774

variables (a b x y : ℝ)

theorem conditions_for_a_and_b (h1 : x^2 + x * y + y^2 - y = 0) (h2 : a * x^2 + b * x * y + x = 0) :
  (a + 1)^2 = 4 * (b + 1) ∧ b ≠ -1 :=
sorry

end NUMINAMATH_GPT_conditions_for_a_and_b_l2057_205774


namespace NUMINAMATH_GPT_common_difference_of_AP_l2057_205711

theorem common_difference_of_AP (a T_12 : ℝ) (d : ℝ) (n : ℕ) (h1 : a = 2) (h2 : T_12 = 90) (h3 : n = 12) 
(h4 : T_12 = a + (n - 1) * d) : d = 8 := 
by sorry

end NUMINAMATH_GPT_common_difference_of_AP_l2057_205711


namespace NUMINAMATH_GPT_tim_weekly_earnings_l2057_205746

-- Define the constants based on the conditions
def tasks_per_day : ℕ := 100
def pay_per_task : ℝ := 1.2
def days_per_week : ℕ := 6

-- Statement to prove Tim's weekly earnings
theorem tim_weekly_earnings : tasks_per_day * pay_per_task * days_per_week = 720 := by
  sorry

end NUMINAMATH_GPT_tim_weekly_earnings_l2057_205746


namespace NUMINAMATH_GPT_probability_of_drawing_three_white_marbles_l2057_205730

noncomputable def probability_of_three_white_marbles : ℚ :=
  let total_marbles := 5 + 7 + 15
  let prob_first_white := 15 / total_marbles
  let prob_second_white := 14 / (total_marbles - 1)
  let prob_third_white := 13 / (total_marbles - 2)
  prob_first_white * prob_second_white * prob_third_white

theorem probability_of_drawing_three_white_marbles :
  probability_of_three_white_marbles = 2 / 13 := 
by 
  sorry

end NUMINAMATH_GPT_probability_of_drawing_three_white_marbles_l2057_205730


namespace NUMINAMATH_GPT_geometric_sequence_sum_condition_l2057_205710

theorem geometric_sequence_sum_condition
  (a_1 r : ℝ) 
  (S₄ : ℝ := a_1 * (1 + r + r^2 + r^3)) 
  (S₈ : ℝ := S₄ + a_1 * (r^4 + r^5 + r^6 + r^7)) 
  (h₁ : S₄ = 1) 
  (h₂ : S₈ = 3) :
  a_1 * r^16 * (1 + r + r^2 + r^3) = 8 := 
sorry

end NUMINAMATH_GPT_geometric_sequence_sum_condition_l2057_205710


namespace NUMINAMATH_GPT_volume_of_normal_block_is_3_l2057_205712

variable (w d l : ℝ)
def V_normal : ℝ := w * d * l
def V_large : ℝ := (2 * w) * (2 * d) * (3 * l)

theorem volume_of_normal_block_is_3 (h : V_large w d l = 36) : V_normal w d l = 3 :=
by sorry

end NUMINAMATH_GPT_volume_of_normal_block_is_3_l2057_205712


namespace NUMINAMATH_GPT_maryann_rescue_time_l2057_205705

def time_to_free_cheaph (minutes : ℕ) : ℕ := 6
def time_to_free_expenh (minutes : ℕ) : ℕ := 8
def num_friends : ℕ := 3

theorem maryann_rescue_time : (time_to_free_cheaph 6 + time_to_free_expenh 8) * num_friends = 42 := 
by
  sorry

end NUMINAMATH_GPT_maryann_rescue_time_l2057_205705


namespace NUMINAMATH_GPT_math_problem_l2057_205756

theorem math_problem (x y : ℝ) (h1 : x - 2 * y = 4) (h2 : x * y = 8) :
  x^2 + 4 * y^2 = 48 :=
sorry

end NUMINAMATH_GPT_math_problem_l2057_205756
