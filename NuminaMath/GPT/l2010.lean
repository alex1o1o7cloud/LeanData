import Mathlib

namespace NUMINAMATH_GPT_number_of_persons_in_group_l2010_201034

theorem number_of_persons_in_group 
    (n : ℕ)
    (h1 : average_age_before - average_age_after = 3)
    (h2 : person_replaced_age = 40)
    (h3 : new_person_age = 10)
    (h4 : total_age_decrease = 3 * n):
  n = 10 := 
sorry

end NUMINAMATH_GPT_number_of_persons_in_group_l2010_201034


namespace NUMINAMATH_GPT_no_real_roots_equationD_l2010_201008

def discriminant (a b c : ℕ) : ℤ := b^2 - 4 * a * c

def equationA := (1, -2, -4)
def equationB := (1, -4, 4)
def equationC := (1, -2, -5)
def equationD := (1, 3, 5)

theorem no_real_roots_equationD :
  discriminant (1 : ℕ) 3 5 < 0 :=
by
  show discriminant 1 3 5 < 0
  sorry

end NUMINAMATH_GPT_no_real_roots_equationD_l2010_201008


namespace NUMINAMATH_GPT_find_number_l2010_201056

theorem find_number (n : ℕ) (h : Nat.factorial 4 / Nat.factorial (4 - n) = 24) : n = 3 :=
by
  sorry

end NUMINAMATH_GPT_find_number_l2010_201056


namespace NUMINAMATH_GPT_part_a_part_b_l2010_201014

variable {A B C A₁ B₁ C₁ : Prop}
variables {a b c a₁ b₁ c₁ S S₁ : ℝ}

-- Assume basic conditions of triangles
variable (h1 : IsTriangle A B C)
variable (h2 : IsTriangleWithCentersAndSquares A B C A₁ B₁ C₁ a b c a₁ b₁ c₁ S S₁)
variable (h3 : IsExternalSquaresConstructed A B C A₁ B₁ C₁)

-- Part (a)
theorem part_a : a₁^2 + b₁^2 + c₁^2 = a^2 + b^2 + c^2 + 6 * S := 
sorry

-- Part (b)
theorem part_b : S₁ - S = (a^2 + b^2 + c^2) / 8 := 
sorry

end NUMINAMATH_GPT_part_a_part_b_l2010_201014


namespace NUMINAMATH_GPT_tangent_line_eq_l2010_201079

theorem tangent_line_eq : 
  ∀ (x y: ℝ), y = x^3 - x + 3 → (x = 1 ∧ y = 3) → (2 * x - y - 1 = 0) :=
by
  intros x y h_curve h_point
  sorry

end NUMINAMATH_GPT_tangent_line_eq_l2010_201079


namespace NUMINAMATH_GPT_negation_of_universal_proposition_l2010_201045

theorem negation_of_universal_proposition :
  ¬ (∀ x : ℝ, 1 < x ∧ x < 2 → x^2 > 1) ↔ ∃ x : ℝ, 1 < x ∧ x < 2 ∧ x^2 ≤ 1 := 
sorry

end NUMINAMATH_GPT_negation_of_universal_proposition_l2010_201045


namespace NUMINAMATH_GPT_brandon_cards_l2010_201018

theorem brandon_cards (b m : ℕ) 
  (h1 : m = b + 8) 
  (h2 : 14 = m / 2) : 
  b = 20 := by
  sorry

end NUMINAMATH_GPT_brandon_cards_l2010_201018


namespace NUMINAMATH_GPT_max_eq_zero_max_two_solutions_l2010_201084

theorem max_eq_zero_max_two_solutions {a b : Fin 10 → ℝ}
  (h : ∀ i, a i ≠ 0) : 
  ∃ (solution_count : ℕ), solution_count <= 2 ∧
  ∃ (solutions : Fin solution_count → ℝ), 
    ∀ (x : ℝ), (∀ i, max (a i * x + b i) = 0) ↔ ∃ j, x = solutions j := sorry

end NUMINAMATH_GPT_max_eq_zero_max_two_solutions_l2010_201084


namespace NUMINAMATH_GPT_units_digit_47_pow_47_l2010_201041

theorem units_digit_47_pow_47 : (47 ^ 47) % 10 = 3 :=
by sorry

end NUMINAMATH_GPT_units_digit_47_pow_47_l2010_201041


namespace NUMINAMATH_GPT_right_triangle_side_lengths_l2010_201021

theorem right_triangle_side_lengths (a b c : ℝ) (varrho r : ℝ) (h_varrho : varrho = 8) (h_r : r = 41) : 
  (a = 80 ∧ b = 18 ∧ c = 82) ∨ (a = 18 ∧ b = 80 ∧ c = 82) :=
by
  sorry

end NUMINAMATH_GPT_right_triangle_side_lengths_l2010_201021


namespace NUMINAMATH_GPT_smallest_w_correct_l2010_201057

-- Define the conditions
def is_factor (a b : ℕ) : Prop := ∃ k, a = b * k

-- Given conditions
def cond1 (w : ℕ) : Prop := is_factor (2^6) (1152 * w)
def cond2 (w : ℕ) : Prop := is_factor (3^4) (1152 * w)
def cond3 (w : ℕ) : Prop := is_factor (5^3) (1152 * w)
def cond4 (w : ℕ) : Prop := is_factor (7^2) (1152 * w)
def cond5 (w : ℕ) : Prop := is_factor (11) (1152 * w)
def is_positive (w : ℕ) : Prop := w > 0

-- The smallest possible value of w given all conditions
def smallest_w : ℕ := 16275

-- Proof statement
theorem smallest_w_correct : 
  ∀ (w : ℕ), cond1 w ∧ cond2 w ∧ cond3 w ∧ cond4 w ∧ cond5 w ∧ is_positive w ↔ w = smallest_w := sorry

end NUMINAMATH_GPT_smallest_w_correct_l2010_201057


namespace NUMINAMATH_GPT_percentage_vanaspati_after_adding_ghee_l2010_201039

theorem percentage_vanaspati_after_adding_ghee :
  ∀ (original_quantity new_pure_ghee percentage_ghee percentage_vanaspati : ℝ),
    original_quantity = 30 →
    percentage_ghee = 0.5 →
    percentage_vanaspati = 0.5 →
    new_pure_ghee = 20 →
    (percentage_vanaspati * original_quantity) /
    (original_quantity + new_pure_ghee) * 100 = 30 :=
by
  intros original_quantity new_pure_ghee percentage_ghee percentage_vanaspati
  sorry

end NUMINAMATH_GPT_percentage_vanaspati_after_adding_ghee_l2010_201039


namespace NUMINAMATH_GPT_number_of_red_squares_in_19th_row_l2010_201031

-- Define the number of squares in the n-th row
def number_of_squares (n : ℕ) : ℕ := 3 * n - 1

-- Define the number of red squares in the n-th row
def red_squares (n : ℕ) : ℕ := (number_of_squares n) / 2

-- The theorem stating the problem
theorem number_of_red_squares_in_19th_row : red_squares 19 = 28 := by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_number_of_red_squares_in_19th_row_l2010_201031


namespace NUMINAMATH_GPT_number_of_yellow_marbles_l2010_201091

/-- 
 In a jar with blue, red, and yellow marbles:
  - there are 7 blue marbles
  - there are 11 red marbles
  - the probability of picking a yellow marble is 1/4
 Show that the number of yellow marbles is 6.
-/
theorem number_of_yellow_marbles 
  (blue red y : ℕ) 
  (h_blue : blue = 7) 
  (h_red : red = 11) 
  (h_prob : y / (18 + y) = 1 / 4) : 
  y = 6 := 
sorry

end NUMINAMATH_GPT_number_of_yellow_marbles_l2010_201091


namespace NUMINAMATH_GPT_arithmetic_contains_geometric_progression_l2010_201011

theorem arithmetic_contains_geometric_progression (a d : ℕ) (h_pos : d > 0) :
  ∃ (a' : ℕ) (r : ℕ), a' = a ∧ r = 1 + d ∧ (∀ k : ℕ, ∃ n : ℕ, a' * r^k = a + (n-1)*d) :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_contains_geometric_progression_l2010_201011


namespace NUMINAMATH_GPT_Brian_age_in_eight_years_l2010_201089

-- Definitions based on conditions
variable {Christian Brian : ℕ}
variable (h1 : Christian = 2 * Brian)
variable (h2 : Christian + 8 = 72)

-- Target statement to prove Brian's age in eight years
theorem Brian_age_in_eight_years : (Brian + 8) = 40 :=
by 
  sorry

end NUMINAMATH_GPT_Brian_age_in_eight_years_l2010_201089


namespace NUMINAMATH_GPT_exists_odd_k_l2010_201064

noncomputable def f (n : ℕ) : ℕ :=
sorry

theorem exists_odd_k : 
  (∀ m n : ℕ, f (m * n) = f m * f n) → 
  (∀ m n : ℕ, (m + n) ∣ (f m + f n)) → 
  ∃ k : ℕ, (k % 2 = 1) ∧ (∀ n : ℕ, f n = n ^ k) :=
sorry

end NUMINAMATH_GPT_exists_odd_k_l2010_201064


namespace NUMINAMATH_GPT_semicircle_radius_l2010_201033

theorem semicircle_radius (P : ℝ) (r : ℝ) (h₁ : P = π * r + 2 * r) (h₂ : P = 198) :
  r = 198 / (π + 2) :=
sorry

end NUMINAMATH_GPT_semicircle_radius_l2010_201033


namespace NUMINAMATH_GPT_equidistant_point_l2010_201007

/-- 
  Find the point in the xz-plane that is equidistant from the points (1, 0, 0), 
  (0, -2, 3), and (4, 2, -2). The point in question is \left( \frac{41}{7}, 0, -\frac{19}{14} \right).
-/
theorem equidistant_point :
  ∃ (x z : ℚ), 
    (x - 1)^2 + z^2 = x^2 + 4 + (z - 3)^2 ∧
    (x - 1)^2 + z^2 = (x - 4)^2 + 4 + (z + 2)^2 ∧
    x = 41 / 7 ∧ z = -19 / 14 :=
by
  sorry

end NUMINAMATH_GPT_equidistant_point_l2010_201007


namespace NUMINAMATH_GPT_sums_same_remainder_exists_l2010_201015

theorem sums_same_remainder_exists (n : ℕ) (h : n > 0) (a : Fin (2 * n) → Fin (2 * n)) (ha_permutation : Function.Bijective a) :
  ∃ (i j : Fin (2 * n)), i ≠ j ∧ ((a i + i) % (2 * n) = (a j + j) % (2 * n)) :=
by sorry

end NUMINAMATH_GPT_sums_same_remainder_exists_l2010_201015


namespace NUMINAMATH_GPT_number_of_books_in_box_l2010_201049

theorem number_of_books_in_box (total_weight : ℕ) (weight_per_book : ℕ) 
  (h1 : total_weight = 42) (h2 : weight_per_book = 3) : total_weight / weight_per_book = 14 :=
by sorry

end NUMINAMATH_GPT_number_of_books_in_box_l2010_201049


namespace NUMINAMATH_GPT_find_c_of_parabola_l2010_201017

theorem find_c_of_parabola 
  (a b c : ℝ)
  (h_eq : ∀ y, -3 = a * (y - 1)^2 + b * (y - 1) - 3)
  (h1 : -1 = a * (3 - 1)^2 + b * (3 - 1) - 3) :
  c = -5/2 := by
  sorry

end NUMINAMATH_GPT_find_c_of_parabola_l2010_201017


namespace NUMINAMATH_GPT_sum_of_decimals_as_fraction_l2010_201000

/-- Define the problem inputs as constants -/
def d1 : ℚ := 2 / 10
def d2 : ℚ := 4 / 100
def d3 : ℚ := 6 / 1000
def d4 : ℚ := 8 / 10000
def d5 : ℚ := 1 / 100000

/-- The main theorem statement -/
theorem sum_of_decimals_as_fraction : 
  d1 + d2 + d3 + d4 + d5 = 24681 / 100000 := 
by 
  sorry

end NUMINAMATH_GPT_sum_of_decimals_as_fraction_l2010_201000


namespace NUMINAMATH_GPT_application_methods_count_l2010_201051

theorem application_methods_count (n_graduates m_universities : ℕ) (h_graduates : n_graduates = 5) (h_universities : m_universities = 3) :
  (m_universities ^ n_graduates) = 243 :=
by
  rw [h_graduates, h_universities]
  show 3 ^ 5 = 243
  sorry

end NUMINAMATH_GPT_application_methods_count_l2010_201051


namespace NUMINAMATH_GPT_max_value_of_expression_l2010_201062

theorem max_value_of_expression (A M C : ℕ) (h : A + M + C = 15) :
  A * M * C + A * M + M * C + C * A ≤ 200 :=
sorry

end NUMINAMATH_GPT_max_value_of_expression_l2010_201062


namespace NUMINAMATH_GPT_find_x_for_opposite_expressions_l2010_201004

theorem find_x_for_opposite_expressions :
  ∃ x : ℝ, (x + 1) + (3 * x - 5) = 0 ↔ x = 1 :=
by
  sorry

end NUMINAMATH_GPT_find_x_for_opposite_expressions_l2010_201004


namespace NUMINAMATH_GPT_negation_example_l2010_201069

theorem negation_example :
  (¬ ∀ x : ℝ, x^2 ≥ 0) ↔ (∃ x : ℝ, x^2 < 0) := sorry

end NUMINAMATH_GPT_negation_example_l2010_201069


namespace NUMINAMATH_GPT_average_age_when_youngest_born_l2010_201026

theorem average_age_when_youngest_born (n : ℕ) (avg_age current_youngest_age total_age_when_youngest_born : ℝ) 
  (h1 : n = 7) (h2 : avg_age = 30) (h3 : current_youngest_age = 8) (h4 : total_age_when_youngest_born = (n * avg_age - n * current_youngest_age)) : 
  total_age_when_youngest_born / n = 22 :=
by
  sorry

end NUMINAMATH_GPT_average_age_when_youngest_born_l2010_201026


namespace NUMINAMATH_GPT_number_of_correct_statements_l2010_201061

noncomputable def f (x : ℝ) : ℝ := 1 / 2 * Real.sin (2 * x)

def statement_1 : Prop := ∃ T > 0, ∀ x, f (x + T) = f x ∧ T = 2 * Real.pi
def statement_2 : Prop := ∀ x y, -Real.pi / 4 ≤ x ∧ x ≤ y ∧ y ≤ Real.pi / 4 → f x ≤ f y
def statement_3 : Prop := ∀ y, -Real.pi / 6 ≤ y ∧ y ≤ Real.pi / 3 → -Real.sqrt 3 / 4 ≤ f y ∧ f y ≤ Real.sqrt 3 / 4
def statement_4 : Prop := ∀ x, f x = (1 / 2 * Real.sin (2 * x + Real.pi / 4) - Real.pi / 8)

theorem number_of_correct_statements : 
  (¬ statement_1 ∧ statement_2 ∧ ¬ statement_3 ∧ ¬ statement_4) = true :=
sorry

end NUMINAMATH_GPT_number_of_correct_statements_l2010_201061


namespace NUMINAMATH_GPT_number_of_tons_is_3_l2010_201006

noncomputable def calculate_tons_of_mulch {total_cost price_per_pound pounds_per_ton : ℝ} 
  (h_total_cost : total_cost = 15000) 
  (h_price_per_pound : price_per_pound = 2.5) 
  (h_pounds_per_ton : pounds_per_ton = 2000) : ℝ := 
  total_cost / price_per_pound / pounds_per_ton

theorem number_of_tons_is_3 
  (total_cost price_per_pound pounds_per_ton : ℝ) 
  (h_total_cost : total_cost = 15000) 
  (h_price_per_pound : price_per_pound = 2.5) 
  (h_pounds_per_ton : pounds_per_ton = 2000) : 
  calculate_tons_of_mulch h_total_cost h_price_per_pound h_pounds_per_ton = 3 := 
by
  sorry

end NUMINAMATH_GPT_number_of_tons_is_3_l2010_201006


namespace NUMINAMATH_GPT_max_a_condition_l2010_201063

theorem max_a_condition (a : ℝ) :
  (∀ x : ℝ, x < a → |x| > 2) ∧ (∃ x : ℝ, |x| > 2 ∧ ¬ (x < a)) →
  a ≤ -2 :=
by 
  sorry

end NUMINAMATH_GPT_max_a_condition_l2010_201063


namespace NUMINAMATH_GPT_calc_fraction_l2010_201044

variable {x y : ℝ}

theorem calc_fraction (h : x + y = x * y - 1) (hx : x ≠ 0) (hy : y ≠ 0) : 
  (1 / x) + (1 / y) = 1 - 1 / (x * y) := 
by 
  sorry

end NUMINAMATH_GPT_calc_fraction_l2010_201044


namespace NUMINAMATH_GPT_func_positive_range_l2010_201087

theorem func_positive_range (a : ℝ) : 
  (∀ x : ℝ, (5 - a) * x^2 - 6 * x + a + 5 > 0) → (-4 < a ∧ a < 4) := 
by 
  sorry

end NUMINAMATH_GPT_func_positive_range_l2010_201087


namespace NUMINAMATH_GPT_value_after_increase_l2010_201043

def original_number : ℝ := 400
def percentage_increase : ℝ := 0.20

theorem value_after_increase : original_number * (1 + percentage_increase) = 480 := by
  sorry

end NUMINAMATH_GPT_value_after_increase_l2010_201043


namespace NUMINAMATH_GPT_smallest_x_y_sum_l2010_201042

theorem smallest_x_y_sum (x y : ℕ) (h1 : x ≠ y) (h2 : x > 0) (h3 : y > 0) (h4 : 1 / (x : ℚ) + 1 / (y : ℚ) = 1 / 10) : x + y = 45 :=
by
  sorry

end NUMINAMATH_GPT_smallest_x_y_sum_l2010_201042


namespace NUMINAMATH_GPT_plane_equation_l2010_201085

theorem plane_equation :
  ∃ (A B C D : ℤ), (A > 0) ∧ (Int.gcd (Int.gcd A B) (Int.gcd C D) = 1) ∧
  (∀ x y z : ℤ, 
    (A * x + B * y + C * z + D = 0) ↔
      (x = 1 ∧ y = 6 ∧ z = -8 ∨ (∃ t : ℤ, 
        x = 2 + 4 * t ∧ y = 4 - t ∧ z = -3 + 5 * t))) ∧
  (A = 5 ∧ B = 15 ∧ C = -7 ∧ D = -151) :=
sorry

end NUMINAMATH_GPT_plane_equation_l2010_201085


namespace NUMINAMATH_GPT_probability_no_rain_five_days_probability_drought_alert_approx_l2010_201022

theorem probability_no_rain_five_days (p : ℚ) (h : p = 1/3) :
  (p ^ 5) = 1 / 243 :=
by
  -- Add assumptions and proceed
  sorry

theorem probability_drought_alert_approx (p : ℚ) (h : p = 1/3) :
  4 * (p ^ 2) = 4 / 9 :=
by
  -- Add assumptions and proceed
  sorry

end NUMINAMATH_GPT_probability_no_rain_five_days_probability_drought_alert_approx_l2010_201022


namespace NUMINAMATH_GPT_difference_of_sum_l2010_201086

theorem difference_of_sum (a b c : ℤ) (h1 : a = 11) (h2 : b = 13) (h3 : c = 15) :
  (b + c) - a = 17 := by
  sorry

end NUMINAMATH_GPT_difference_of_sum_l2010_201086


namespace NUMINAMATH_GPT_dorothy_total_sea_glass_l2010_201030

def Blanche_red : ℕ := 3
def Rose_red : ℕ := 9
def Rose_blue : ℕ := 11

def Dorothy_red : ℕ := 2 * (Blanche_red + Rose_red)
def Dorothy_blue : ℕ := 3 * Rose_blue

theorem dorothy_total_sea_glass : Dorothy_red + Dorothy_blue = 57 :=
by
  sorry

end NUMINAMATH_GPT_dorothy_total_sea_glass_l2010_201030


namespace NUMINAMATH_GPT_problem_1_solution_set_problem_2_range_of_T_l2010_201059

noncomputable def f (x : ℝ) : ℝ := |2 * x + 1| - |x - 2|

theorem problem_1_solution_set :
  {x : ℝ | f x > 2} = {x | x < -5 ∨ 1 < x} :=
by 
  -- to be proven
  sorry

theorem problem_2_range_of_T (T : ℝ) :
  (∀ x : ℝ, f x ≥ -T^2 - 2.5 * T - 1) →
  (T ≤ -3 ∨ T ≥ 0.5) :=
by
  -- to be proven
  sorry

end NUMINAMATH_GPT_problem_1_solution_set_problem_2_range_of_T_l2010_201059


namespace NUMINAMATH_GPT_gem_stone_necklaces_sold_l2010_201020

-- Definitions and conditions
def bead_necklaces : ℕ := 7
def total_earnings : ℝ := 90
def price_per_necklace : ℝ := 9

-- Theorem to prove the number of gem stone necklaces sold
theorem gem_stone_necklaces_sold : 
  ∃ (G : ℕ), G * price_per_necklace = total_earnings - (bead_necklaces * price_per_necklace) ∧ G = 3 :=
by
  sorry

end NUMINAMATH_GPT_gem_stone_necklaces_sold_l2010_201020


namespace NUMINAMATH_GPT_modulus_of_z_l2010_201088

noncomputable def i : ℂ := Complex.I

noncomputable def z : ℂ := sorry

theorem modulus_of_z 
  (hz : i * z = (1 - 2 * i)^2) : 
  Complex.abs z = 5 := by
  sorry

end NUMINAMATH_GPT_modulus_of_z_l2010_201088


namespace NUMINAMATH_GPT_brick_length_correct_l2010_201047

-- Define the constants
def courtyard_length_meters : ℝ := 25
def courtyard_width_meters : ℝ := 18
def courtyard_area_meters : ℝ := courtyard_length_meters * courtyard_width_meters
def bricks_number : ℕ := 22500
def brick_width_cm : ℕ := 10

-- We want to prove the length of each brick
def brick_length_cm : ℕ := 20

-- Convert courtyard area to square centimeters
def courtyard_area_cm : ℝ := courtyard_area_meters * 10000

-- Define the proof statement
theorem brick_length_correct :
  courtyard_area_cm = (brick_length_cm * brick_width_cm) * bricks_number :=
by
  sorry

end NUMINAMATH_GPT_brick_length_correct_l2010_201047


namespace NUMINAMATH_GPT_smallest_composite_no_prime_factors_less_than_20_is_529_l2010_201029

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_composite (n : ℕ) : Prop :=
  n > 1 ∧ ¬is_prime n

def smallest_prime_factor_greater_than_20 (n : ℕ) : Prop :=
  is_prime n ∧ n > 20 ∧ ∀ p : ℕ, is_prime p ∧ p > 20 → p >= n

def smallest_composite_with_no_prime_factors_less_than_20 (n : ℕ) : Prop :=
  is_composite n ∧ ∀ p : ℕ, is_prime p ∧ p < 20 → ¬ p ∣ n

theorem smallest_composite_no_prime_factors_less_than_20_is_529 :
  smallest_composite_with_no_prime_factors_less_than_20 529 :=
by
  sorry

end NUMINAMATH_GPT_smallest_composite_no_prime_factors_less_than_20_is_529_l2010_201029


namespace NUMINAMATH_GPT_team_a_wins_at_least_2_l2010_201055

def team_a_wins_at_least (total_games lost_games : ℕ) (points : ℕ) (won_points draw_points lost_points : ℕ) : Prop :=
  ∃ (won_games : ℕ), 
    total_games = won_games + (total_games - lost_games - won_games) + lost_games ∧
    won_games * won_points + (total_games - lost_games - won_games) * draw_points > points ∧
    won_games ≥ 2

theorem team_a_wins_at_least_2 :
  team_a_wins_at_least 5 1 7 3 1 0 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_team_a_wins_at_least_2_l2010_201055


namespace NUMINAMATH_GPT_hyperbola_center_l2010_201027

theorem hyperbola_center :
  (∃ h k : ℝ,
    (∀ x y : ℝ, ((4 * x - 8) / 9)^2 - ((5 * y + 5) / 7)^2 = 1 ↔ (x - h)^2 / (81 / 16) - (y - k)^2 / (49 / 25) = 1) ∧
    (h = 2) ∧ (k = -1)) :=
sorry

end NUMINAMATH_GPT_hyperbola_center_l2010_201027


namespace NUMINAMATH_GPT_min_route_length_5x5_l2010_201082

-- Definition of the grid and its properties
def grid : Type := Fin 5 × Fin 5

-- Define a function to calculate the minimum route length
noncomputable def min_route_length (grid_size : ℕ) : ℕ :=
  if h : grid_size = 5 then 68 else 0

-- The proof problem statement
theorem min_route_length_5x5 : min_route_length 5 = 68 :=
by
  -- Skipping the actual proof
  sorry

end NUMINAMATH_GPT_min_route_length_5x5_l2010_201082


namespace NUMINAMATH_GPT_num_factors_of_M_l2010_201053

def M : ℕ := 2^4 * 3^3 * 7^2

theorem num_factors_of_M : ∃ n, n = 60 ∧ (∀ d e f : ℕ, 0 ≤ d ∧ d ≤ 4 ∧ 0 ≤ e ∧ e ≤ 3 ∧ 0 ≤ f ∧ f ≤ 2 → (2^d * 3^e * 7^f ∣ M) ∧ ∃ k, k = 5 * 4 * 3 ∧ k = n) :=
by
  sorry

end NUMINAMATH_GPT_num_factors_of_M_l2010_201053


namespace NUMINAMATH_GPT_dot_product_eq_eight_l2010_201081

def vec_a : ℝ × ℝ := (0, 4)
def vec_b : ℝ × ℝ := (2, 2)

theorem dot_product_eq_eight : (vec_a.1 * vec_b.1 + vec_a.2 * vec_b.2) = 8 := by
  sorry

end NUMINAMATH_GPT_dot_product_eq_eight_l2010_201081


namespace NUMINAMATH_GPT_math_expr_evaluation_l2010_201025

theorem math_expr_evaluation :
  3 + 15 / 3 - 2^2 + 1 = 5 :=
by
  -- The proof will be filled here
  sorry

end NUMINAMATH_GPT_math_expr_evaluation_l2010_201025


namespace NUMINAMATH_GPT_pointC_on_same_side_as_point1_l2010_201070

-- Definitions of points and the line equation
def is_on_same_side (p1 p2 : ℝ × ℝ) (line : ℝ × ℝ → ℝ) : Prop :=
  (line p1 > 0) ↔ (line p2 > 0)

def line_eq (p : ℝ × ℝ) : ℝ := p.1 + p.2 - 1

def point1 : ℝ × ℝ := (1, 2)
def pointC : ℝ × ℝ := (-1, 3)

-- Theorem to prove the equivalence
theorem pointC_on_same_side_as_point1 :
  is_on_same_side point1 pointC line_eq :=
sorry

end NUMINAMATH_GPT_pointC_on_same_side_as_point1_l2010_201070


namespace NUMINAMATH_GPT_volume_conversion_l2010_201024

theorem volume_conversion (v_feet : ℕ) (h : v_feet = 250) : (v_feet / 27 : ℚ) = 250 / 27 := by
  sorry

end NUMINAMATH_GPT_volume_conversion_l2010_201024


namespace NUMINAMATH_GPT_average_of_remaining_two_numbers_l2010_201028

theorem average_of_remaining_two_numbers 
  (a b c d e f : ℝ)
  (h1 : (a + b + c + d + e + f) / 6 = 4.60)
  (h2 : (a + b) / 2 = 3.4)
  (h3 : (c + d) / 2 = 3.8) :
  ((e + f) / 2) = 6.6 :=
sorry

end NUMINAMATH_GPT_average_of_remaining_two_numbers_l2010_201028


namespace NUMINAMATH_GPT_arithmetic_sequence_smallest_value_l2010_201076

theorem arithmetic_sequence_smallest_value:
  ∃ a : ℕ, (7 * a + 63) % 11 = 0 ∧ (a - 9) % 11 = 4 := sorry

end NUMINAMATH_GPT_arithmetic_sequence_smallest_value_l2010_201076


namespace NUMINAMATH_GPT_ratio_of_areas_l2010_201092

theorem ratio_of_areas (s : ℝ) (h_s_pos : 0 < s) :
  let small_triangle_area := (s^2 * Real.sqrt 3) / 4
  let total_small_triangles_area := 6 * small_triangle_area
  let large_triangle_side := 6 * s
  let large_triangle_area := (large_triangle_side^2 * Real.sqrt 3) / 4
  total_small_triangles_area / large_triangle_area = 1 / 6 :=
by
  let small_triangle_area := (s^2 * Real.sqrt 3) / 4
  let total_small_triangles_area := 6 * small_triangle_area
  let large_triangle_side := 6 * s
  let large_triangle_area := (large_triangle_side^2 * Real.sqrt 3) / 4
  sorry
 
end NUMINAMATH_GPT_ratio_of_areas_l2010_201092


namespace NUMINAMATH_GPT_ferry_max_weight_capacity_l2010_201077

def automobile_max_weight : ℝ := 3200
def automobile_count : ℝ := 62.5
def pounds_to_tons : ℝ := 2000

theorem ferry_max_weight_capacity : 
  (automobile_max_weight * automobile_count) / pounds_to_tons = 100 := 
by 
  sorry

end NUMINAMATH_GPT_ferry_max_weight_capacity_l2010_201077


namespace NUMINAMATH_GPT_prob_both_standard_prob_only_one_standard_l2010_201078

-- Given conditions
axiom prob_A1 : ℝ
axiom prob_A2 : ℝ
axiom prob_A1_std : prob_A1 = 0.95
axiom prob_A2_std : prob_A2 = 0.95
axiom prob_not_A1 : ℝ
axiom prob_not_A2 : ℝ
axiom prob_not_A1_std : prob_not_A1 = 0.05
axiom prob_not_A2_std : prob_not_A2 = 0.05
axiom independent_A1_A2 : prob_A1 * prob_A2 = prob_A1 * prob_A2

-- Definitions of events
def event_A1 := true -- Event that the first product is standard
def event_A2 := true -- Event that the second product is standard
def event_not_A1 := not event_A1
def event_not_A2 := not event_A2

-- Proof problems
theorem prob_both_standard :
  prob_A1 * prob_A2 = 0.9025 := by sorry

theorem prob_only_one_standard :
  (prob_A1 * prob_not_A2) + (prob_not_A1 * prob_A2) = 0.095 := by sorry

end NUMINAMATH_GPT_prob_both_standard_prob_only_one_standard_l2010_201078


namespace NUMINAMATH_GPT_periodic_sum_constant_l2010_201002

noncomputable def is_periodic (f : ℝ → ℝ) (a : ℝ) : Prop :=
a ≠ 0 ∧ ∀ x : ℝ, f (a + x) = f x

theorem periodic_sum_constant (f g : ℝ → ℝ) (a b : ℝ)
  (ha : a ≠ 0) (hb : b ≠ 0) (hfa : is_periodic f a) (hgb : is_periodic g b)
  (harational : ∃ m n : ℤ, (a : ℝ) = m / n) (hbirrational : ¬ ∃ m n : ℤ, (b : ℝ) = m / n) :
  (∃ c : ℝ, c ≠ 0 ∧ ∀ x : ℝ, (f + g) (c + x) = (f + g) x) →
  (∀ x : ℝ, f x = f 0) ∨ (∀ x : ℝ, g x = g 0) :=
sorry

end NUMINAMATH_GPT_periodic_sum_constant_l2010_201002


namespace NUMINAMATH_GPT_find_f_7_l2010_201054

noncomputable def f (x : ℝ) : ℝ := 2*x^4 - 17*x^3 + 26*x^2 - 24*x - 60

theorem find_f_7 : f 7 = 17 :=
  by
  -- The proof steps will go here
  sorry

end NUMINAMATH_GPT_find_f_7_l2010_201054


namespace NUMINAMATH_GPT_circle_range_of_a_l2010_201099

theorem circle_range_of_a (a : ℝ) :
  (∀ x y : ℝ, x^2 + y^2 + 2 * a * x - 4 * y + (a^2 + a) = 0 → (x - h)^2 + (y - k)^2 = r^2) ↔ (a < 4) :=
sorry

end NUMINAMATH_GPT_circle_range_of_a_l2010_201099


namespace NUMINAMATH_GPT_golden_section_length_l2010_201037

noncomputable def golden_section_point (a b : ℝ) := a / (a + b) = b / a

theorem golden_section_length (A B P : ℝ) (h : golden_section_point A P) (hAP_gt_PB : A > P) (hAB : A + P = 2) : 
  A = Real.sqrt 5 - 1 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_golden_section_length_l2010_201037


namespace NUMINAMATH_GPT_no_positive_integer_satisfies_inequality_l2010_201023

theorem no_positive_integer_satisfies_inequality :
  ∀ x : ℕ, 0 < x → ¬ (15 < -3 * (x : ℤ) + 18) := by
  sorry

end NUMINAMATH_GPT_no_positive_integer_satisfies_inequality_l2010_201023


namespace NUMINAMATH_GPT_girls_attending_picnic_l2010_201074

theorem girls_attending_picnic (g b : ℕ) (h1 : g + b = 1200) (h2 : (2 * g) / 3 + b / 2 = 730) : (2 * g) / 3 = 520 :=
by
  -- The proof steps would go here.
  sorry

end NUMINAMATH_GPT_girls_attending_picnic_l2010_201074


namespace NUMINAMATH_GPT_lowest_fraction_done_in_an_hour_by_two_people_l2010_201065

def a_rate : ℚ := 1 / 4
def b_rate : ℚ := 1 / 5
def c_rate : ℚ := 1 / 6

theorem lowest_fraction_done_in_an_hour_by_two_people : 
  min (min (a_rate + b_rate) (a_rate + c_rate)) (b_rate + c_rate) = 11 / 30 := 
by
  sorry

end NUMINAMATH_GPT_lowest_fraction_done_in_an_hour_by_two_people_l2010_201065


namespace NUMINAMATH_GPT_min_value_z_l2010_201048

variable (x y : ℝ)

theorem min_value_z : ∃ (x y : ℝ), 2 * x + 3 * y = 9 :=
sorry

end NUMINAMATH_GPT_min_value_z_l2010_201048


namespace NUMINAMATH_GPT_find_sum_of_a_b_c_l2010_201080

def a := 8
def b := 2
def c := 2

theorem find_sum_of_a_b_c : a + b + c = 12 :=
by
  have ha : a = 8 := rfl
  have hb : b = 2 := rfl
  have hc : c = 2 := rfl
  sorry

end NUMINAMATH_GPT_find_sum_of_a_b_c_l2010_201080


namespace NUMINAMATH_GPT_ten_thousand_times_ten_thousand_l2010_201066

theorem ten_thousand_times_ten_thousand :
  10000 * 10000 = 100000000 :=
by
  sorry

end NUMINAMATH_GPT_ten_thousand_times_ten_thousand_l2010_201066


namespace NUMINAMATH_GPT_benny_seashells_l2010_201016

-- Define the initial number of seashells Benny found
def seashells_found : ℝ := 66.5

-- Define the percentage of seashells Benny gave away
def percentage_given_away : ℝ := 0.75

-- Calculate the number of seashells Benny gave away
def seashells_given_away : ℝ := percentage_given_away * seashells_found

-- Calculate the number of seashells Benny now has
def seashells_left : ℝ := seashells_found - seashells_given_away

-- Prove that Benny now has 16.625 seashells
theorem benny_seashells : seashells_left = 16.625 :=
by
  sorry

end NUMINAMATH_GPT_benny_seashells_l2010_201016


namespace NUMINAMATH_GPT_number_of_jerseys_bought_l2010_201096

-- Define the given constants
def initial_money : ℕ := 50
def cost_per_jersey : ℕ := 2
def cost_basketball : ℕ := 18
def cost_shorts : ℕ := 8
def money_left : ℕ := 14

-- Define the theorem to prove the number of jerseys Jeremy bought.
theorem number_of_jerseys_bought :
  (initial_money - money_left) = (cost_basketball + cost_shorts + 5 * cost_per_jersey) :=
by
  sorry

end NUMINAMATH_GPT_number_of_jerseys_bought_l2010_201096


namespace NUMINAMATH_GPT_solve_inequality_l2010_201040

theorem solve_inequality (x : ℝ) : -4 * x - 8 > 0 → x < -2 := sorry

end NUMINAMATH_GPT_solve_inequality_l2010_201040


namespace NUMINAMATH_GPT_circle_standard_form1_circle_standard_form2_l2010_201095

theorem circle_standard_form1 (x y : ℝ) :
  x^2 + y^2 - 4 * x + 6 * y - 3 = 0 ↔ (x - 2)^2 + (y + 3)^2 = 16 :=
by
  sorry

theorem circle_standard_form2 (x y : ℝ) :
  4 * x^2 + 4 * y^2 - 8 * x + 4 * y - 11 = 0 ↔ (x - 1)^2 + (y + 1 / 2)^2 = 4 :=
by
  sorry

end NUMINAMATH_GPT_circle_standard_form1_circle_standard_form2_l2010_201095


namespace NUMINAMATH_GPT_geometric_sequence_sum_l2010_201097

noncomputable def seq (a : ℕ → ℝ) : Prop :=
∀ n ≥ 2, a n ^ 2 = a (n - 1) * a (n + 1)

theorem geometric_sequence_sum (a : ℕ → ℝ) 
  (h_seq : seq a)
  (h_a2 : a 2 = 3)
  (h_sum : a 2 + a 4 + a 6 = 21) :
  (a 4 + a 6 + a 8) = 42 :=
sorry

end NUMINAMATH_GPT_geometric_sequence_sum_l2010_201097


namespace NUMINAMATH_GPT_expression_equality_l2010_201072

theorem expression_equality : (2 + Real.sqrt 2 + 1 / (2 + Real.sqrt 2) + 1 / (Real.sqrt 2 - 2) = 2) :=
sorry

end NUMINAMATH_GPT_expression_equality_l2010_201072


namespace NUMINAMATH_GPT_shoes_remaining_l2010_201060

theorem shoes_remaining (monthly_goal : ℕ) (sold_last_week : ℕ) (sold_this_week : ℕ) (remaining_shoes : ℕ) :
  monthly_goal = 80 →
  sold_last_week = 27 →
  sold_this_week = 12 →
  remaining_shoes = monthly_goal - sold_last_week - sold_this_week →
  remaining_shoes = 41 := by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  exact h4

end NUMINAMATH_GPT_shoes_remaining_l2010_201060


namespace NUMINAMATH_GPT_range_of_m_l2010_201093

theorem range_of_m (m : ℝ) : (∀ x : ℝ, |x + 1| + |x - 3| ≥ |m - 1|) → -3 ≤ m ∧ m ≤ 5 :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l2010_201093


namespace NUMINAMATH_GPT_prime_sum_square_mod_3_l2010_201036

theorem prime_sum_square_mod_3 (p : Fin 100 → ℕ) (h_prime : ∀ i, Nat.Prime (p i)) (h_distinct : Function.Injective p) :
  let N := (Finset.univ : Finset (Fin 100)).sum (λ i => (p i)^2)
  N % 3 = 1 := by
  sorry

end NUMINAMATH_GPT_prime_sum_square_mod_3_l2010_201036


namespace NUMINAMATH_GPT_find_ck_l2010_201058

-- Definitions based on the conditions
def arithmetic_sequence (d : ℕ) (n : ℕ) : ℕ :=
  1 + (n - 1) * d

def geometric_sequence (r : ℕ) (n : ℕ) : ℕ :=
  r^(n - 1)

def combined_sequence (d r : ℕ) (n : ℕ) : ℕ :=
  arithmetic_sequence d n + geometric_sequence r n

-- Given conditions
variable {d r k : ℕ}
variable (hd : combined_sequence d r (k-1) = 250)
variable (hk : combined_sequence d r (k+1) = 1250)

-- The theorem statement
theorem find_ck : combined_sequence d r k = 502 :=
  sorry

end NUMINAMATH_GPT_find_ck_l2010_201058


namespace NUMINAMATH_GPT_cost_of_cherries_l2010_201067

theorem cost_of_cherries (total_spent amount_for_grapes amount_for_cherries : ℝ)
  (h1 : total_spent = 21.93)
  (h2 : amount_for_grapes = 12.08)
  (h3 : amount_for_cherries = total_spent - amount_for_grapes) :
  amount_for_cherries = 9.85 :=
sorry

end NUMINAMATH_GPT_cost_of_cherries_l2010_201067


namespace NUMINAMATH_GPT_probability_of_2_gold_no_danger_l2010_201071

variable (caves : Finset Nat) (n : Nat)

-- Probability definitions
def P_gold_no_danger : ℚ := 1 / 5
def P_danger_no_gold : ℚ := 1 / 10
def P_neither : ℚ := 4 / 5

-- Probability calculation
def P_exactly_2_gold_none_danger : ℚ :=
  10 * (P_gold_no_danger) ^ 2 * (P_neither) ^ 3

theorem probability_of_2_gold_no_danger :
  (P_exactly_2_gold_none_danger) = 128 / 625 :=
sorry

end NUMINAMATH_GPT_probability_of_2_gold_no_danger_l2010_201071


namespace NUMINAMATH_GPT_valid_differences_of_squares_l2010_201012

theorem valid_differences_of_squares (n : ℕ) (h : 2 * n + 1 < 150) :
    (2 * n + 1 = 129 ∨ 2 * n +1 = 147) :=
by
  sorry

end NUMINAMATH_GPT_valid_differences_of_squares_l2010_201012


namespace NUMINAMATH_GPT_value_of_a_l2010_201009

theorem value_of_a (x a : ℤ) (h : x = 4) (h_eq : 5 * (x - 1) - 3 * a = -3) : a = 6 :=
by {
  sorry
}

end NUMINAMATH_GPT_value_of_a_l2010_201009


namespace NUMINAMATH_GPT_solve_arithmetic_sequence_l2010_201052

variable {a : ℕ → ℝ}
variable {d a1 a2 a3 a10 a11 a6 a7 : ℝ}

axiom arithmetic_seq (n : ℕ) : a (n + 1) = a1 + n * d

def arithmetic_condition (h : a 2 + a 3 + a 10 + a 11 = 32) : Prop :=
  a 6 + a 7 = 16

theorem solve_arithmetic_sequence (h : a 2 + a 3 + a 10 + a 11 = 32) : a 6 + a 7 = 16 :=
  by
    -- Proof will go here
    sorry

end NUMINAMATH_GPT_solve_arithmetic_sequence_l2010_201052


namespace NUMINAMATH_GPT_proof_problem1_proof_problem2_l2010_201083

noncomputable def problem1_lhs : ℝ := 
  1 / (Real.sqrt 3 + 1) - Real.sin (Real.pi / 3) + Real.sqrt 32 * Real.sqrt (1 / 8)

noncomputable def problem1_rhs : ℝ := 3 / 2

theorem proof_problem1 : problem1_lhs = problem1_rhs :=
by 
  sorry

noncomputable def problem2_lhs : ℝ := 
  2^(-2 : ℤ) - Real.sqrt ((-2)^2) + 6 * Real.sin (Real.pi / 4) - Real.sqrt 18

noncomputable def problem2_rhs : ℝ := -7 / 4

theorem proof_problem2 : problem2_lhs = problem2_rhs :=
by 
  sorry

end NUMINAMATH_GPT_proof_problem1_proof_problem2_l2010_201083


namespace NUMINAMATH_GPT_value_of_m_minus_n_l2010_201068

theorem value_of_m_minus_n (m n : ℝ) (i : ℂ) (h1 : i * i = -1) (h2 : (m : ℂ) / (1 + i) = 1 - n * i) : m - n = 1 :=
sorry

end NUMINAMATH_GPT_value_of_m_minus_n_l2010_201068


namespace NUMINAMATH_GPT_circle_sector_cones_sum_radii_l2010_201010

theorem circle_sector_cones_sum_radii :
  let r := 5
  let a₁ := 1
  let a₂ := 2
  let a₃ := 3
  let total_area := π * r * r
  let θ₁ := (a₁ / (a₁ + a₂ + a₃)) * 2 * π
  let θ₂ := (a₂ / (a₁ + a₂ + a₃)) * 2 * π
  let θ₃ := (a₃ / (a₁ + a₂ + a₃)) * 2 * π
  let r₁ := (a₁ / (a₁ + a₂ + a₃)) * r
  let r₂ := (a₂ / (a₁ + a₂ + a₃)) * r
  let r₃ := (a₃ / (a₁ + a₂ + a₃)) * r
  r₁ + r₂ + r₃ = 5 :=
by {
  sorry
}

end NUMINAMATH_GPT_circle_sector_cones_sum_radii_l2010_201010


namespace NUMINAMATH_GPT_min_expression_l2010_201098

theorem min_expression (a b c d e f : ℝ) 
  (h_pos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ 0 < e ∧ 0 < f)
  (h_sum : a + b + c + d + e + f = 10) : 
  (1 / a + 9 / b + 16 / c + 25 / d + 36 / e + 49 / f) ≥ 67.6 :=
sorry

end NUMINAMATH_GPT_min_expression_l2010_201098


namespace NUMINAMATH_GPT_range_of_a_l2010_201094

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, a * x^2 - 2 * a * x + 3 > 0) ↔ (0 ≤ a ∧ a < 3) :=
sorry

end NUMINAMATH_GPT_range_of_a_l2010_201094


namespace NUMINAMATH_GPT_eleven_y_minus_x_eq_one_l2010_201090

theorem eleven_y_minus_x_eq_one 
  (x y : ℤ) 
  (hx_pos : x > 0)
  (h1 : x = 7 * y + 3)
  (h2 : 2 * x = 6 * (3 * y) + 2) : 
  11 * y - x = 1 := 
by 
  sorry

end NUMINAMATH_GPT_eleven_y_minus_x_eq_one_l2010_201090


namespace NUMINAMATH_GPT_slope_of_intersection_points_l2010_201001

theorem slope_of_intersection_points :
  ∀ s : ℝ, ∃ k b : ℝ, (∀ (x y : ℝ), (2 * x - 3 * y = 4 * s + 6) ∧ (2 * x + y = 3 * s + 1) → y = k * x + b) ∧ k = -2/13 := 
by
  intros s
  -- Proof to be provided here
  sorry

end NUMINAMATH_GPT_slope_of_intersection_points_l2010_201001


namespace NUMINAMATH_GPT_max_volume_pyramid_l2010_201050

theorem max_volume_pyramid 
  (AB AC : ℝ)
  (sin_BAC : ℝ)
  (angle_cond : ∀ (SA SB SC : ℝ), SA = SB ∧ SB = SC ∧ SC = SA → ∀ θ, θ ≤ 60 → true)
  (h : ℝ)
  (V : ℝ)
  (AB_eq : AB = 3)
  (AC_eq : AC = 5)
  (sin_BAC_eq : sin_BAC = 3/5)
  (height_cond : h = (5 * Real.sqrt 3) / 2)
  (volume_cond : V = (1/3) * (1/2 * 3 * 5 * (3/5)) * h) :
  V = (5 * Real.sqrt 174) / 4 := sorry

end NUMINAMATH_GPT_max_volume_pyramid_l2010_201050


namespace NUMINAMATH_GPT_water_tank_capacity_l2010_201075

-- Define the variables and conditions
variables (T : ℝ) (h : 0.35 * T = 36)

-- State the theorem
theorem water_tank_capacity : T = 103 :=
by
  -- Placeholder for proof
  sorry

end NUMINAMATH_GPT_water_tank_capacity_l2010_201075


namespace NUMINAMATH_GPT_sum_of_reciprocals_eq_six_l2010_201003

theorem sum_of_reciprocals_eq_six (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0)
  (h1 : x + y = 6 * x * y) (h2 : y = 2 * x) :
  (1 / x) + (1 / y) = 6 := by
  sorry

end NUMINAMATH_GPT_sum_of_reciprocals_eq_six_l2010_201003


namespace NUMINAMATH_GPT_find_m_l2010_201038

theorem find_m (y x m : ℝ) (h1 : 2 - 3 * (1 - y) = 2 * y) (h2 : y = x) (h3 : m * (x - 3) - 2 = -8) : m = 3 :=
sorry

end NUMINAMATH_GPT_find_m_l2010_201038


namespace NUMINAMATH_GPT_sphere_diagonal_property_l2010_201013

variable {A B C D : ℝ}

-- conditions provided
variable (radius : ℝ) (x y z : ℝ)
variable (h_radius : radius = 1)
variable (h_non_coplanar : ¬(is_coplanar A B C D))
variable (h_AB_CD : dist A B = x ∧ dist C D = x)
variable (h_BC_DA : dist B C = y ∧ dist D A = y)
variable (h_CA_BD : dist C A = z ∧ dist B D = z)

theorem sphere_diagonal_property :
  x^2 + y^2 + z^2 = 8 := 
sorry

end NUMINAMATH_GPT_sphere_diagonal_property_l2010_201013


namespace NUMINAMATH_GPT_total_spent_on_concert_tickets_l2010_201019

theorem total_spent_on_concert_tickets : 
  let price_per_ticket := 4
  let number_of_tickets := 3 + 5
  let discount_threshold := 5
  let discount_rate := 0.10
  let service_fee_per_ticket := 2
  let initial_cost := number_of_tickets * price_per_ticket
  let discount := if number_of_tickets > discount_threshold then discount_rate * initial_cost else 0
  let discounted_cost := initial_cost - discount
  let service_fee := number_of_tickets * service_fee_per_ticket
  let total_cost := discounted_cost + service_fee
  total_cost = 44.8 :=
by
  sorry

end NUMINAMATH_GPT_total_spent_on_concert_tickets_l2010_201019


namespace NUMINAMATH_GPT_injured_player_age_l2010_201046

noncomputable def average_age_full_team := 22
noncomputable def number_of_players := 11
noncomputable def average_age_remaining_players := 21
noncomputable def number_of_remaining_players := 10
noncomputable def total_age_full_team := number_of_players * average_age_full_team
noncomputable def total_age_remaining_players := number_of_remaining_players * average_age_remaining_players

theorem injured_player_age :
  (number_of_players * average_age_full_team) -
  (number_of_remaining_players * average_age_remaining_players) = 32 :=
by
  sorry

end NUMINAMATH_GPT_injured_player_age_l2010_201046


namespace NUMINAMATH_GPT_largest_initial_number_l2010_201073

theorem largest_initial_number (n : ℕ) (h : (∃ a b c d e : ℕ, n ≠ 0 ∧ n + a + b + c + d + e = 200 
                                              ∧ n % a ≠ 0 ∧ n % b ≠ 0 ∧ n % c ≠ 0 ∧ n % d ≠ 0 ∧ n % e ≠ 0)) 
: n ≤ 189 :=
sorry

end NUMINAMATH_GPT_largest_initial_number_l2010_201073


namespace NUMINAMATH_GPT_simplify_expression_l2010_201032

theorem simplify_expression : 0.72 * 0.43 + 0.12 * 0.34 = 0.3504 := by
  sorry

end NUMINAMATH_GPT_simplify_expression_l2010_201032


namespace NUMINAMATH_GPT_unit_price_first_purchase_l2010_201035

theorem unit_price_first_purchase (x y : ℝ) (h1 : x * y = 500000) 
    (h2 : 1.4 * x * (y + 10000) = 770000) : x = 5 :=
by
  -- Proof details here
  sorry

end NUMINAMATH_GPT_unit_price_first_purchase_l2010_201035


namespace NUMINAMATH_GPT_has_exactly_one_zero_interval_l2010_201005

noncomputable def f (a x : ℝ) : ℝ := x^2 - a*x + 1

theorem has_exactly_one_zero_interval (a : ℝ) (h : a > 3) : ∃! x, 0 < x ∧ x < 2 ∧ f a x = 0 :=
sorry

end NUMINAMATH_GPT_has_exactly_one_zero_interval_l2010_201005
