import Mathlib

namespace smallest_n_for_gcd_l5_5519

theorem smallest_n_for_gcd (n : ℕ) :
  (∃ n > 0, gcd (11 * n - 3) (8 * n + 4) > 1) ∧ (∀ m > 0, gcd (11 * m - 3) (8 * m + 4) > 1 → n ≤ m) → n = 38 :=
by
  sorry

end smallest_n_for_gcd_l5_5519


namespace modulus_of_z_is_five_l5_5747

def z : Complex := 3 + 4 * Complex.I

theorem modulus_of_z_is_five : Complex.abs z = 5 := by
  sorry

end modulus_of_z_is_five_l5_5747


namespace product_of_first_two_terms_l5_5485

theorem product_of_first_two_terms (a_7 : ℕ) (d : ℕ) (a_7_eq : a_7 = 17) (d_eq : d = 2) :
  let a_1 := a_7 - 6 * d
  let a_2 := a_1 + d
  a_1 * a_2 = 35 :=
by
  sorry

end product_of_first_two_terms_l5_5485


namespace distance_of_coming_down_stairs_l5_5084

noncomputable def totalTimeAscendingDescending (D : ℝ) : ℝ :=
  (D / 2) + ((D + 2) / 3)

theorem distance_of_coming_down_stairs : ∃ D : ℝ, totalTimeAscendingDescending D = 4 ∧ (D + 2) = 6 :=
by
  sorry

end distance_of_coming_down_stairs_l5_5084


namespace parabola_intercepts_l5_5227

noncomputable def question (y : ℝ) := 3 * y ^ 2 - 9 * y + 4

theorem parabola_intercepts (a b c : ℝ) (h_a : a = question 0) (h_b : 3 * b ^ 2 - 9 * b + 4 = 0) (h_c : 3 * c ^ 2 - 9 * c + 4 = 0) :
  a + b + c = 7 :=
by
  sorry

end parabola_intercepts_l5_5227


namespace necessary_but_not_sufficient_l5_5192

variable (k : ℝ)

def is_ellipse : Prop := 
  (k > 1) ∧ (k < 5) ∧ (k ≠ 3)

theorem necessary_but_not_sufficient :
  (1 < k) ∧ (k < 5) → is_ellipse k :=
by sorry

end necessary_but_not_sufficient_l5_5192


namespace Mark_speeding_ticket_owed_amount_l5_5246

theorem Mark_speeding_ticket_owed_amount :
  let base_fine := 50
  let additional_penalty_per_mph := 2
  let mph_over_limit := 45
  let school_zone_multiplier := 2
  let court_costs := 300
  let lawyer_fee_per_hour := 80
  let lawyer_hours := 3
  let additional_penalty := additional_penalty_per_mph * mph_over_limit
  let pre_school_zone_fine := base_fine + additional_penalty
  let doubled_fine := pre_school_zone_fine * school_zone_multiplier
  let total_fine_with_court_costs := doubled_fine + court_costs
  let lawyer_total_fee := lawyer_fee_per_hour * lawyer_hours
  let total_owed := total_fine_with_court_costs + lawyer_total_fee
  total_owed = 820 :=
by
  sorry

end Mark_speeding_ticket_owed_amount_l5_5246


namespace exists_multiple_representations_l5_5626

def V (n : ℕ) : Set ℕ := {m : ℕ | ∃ k : ℕ, m = 1 + k * n}

def indecomposable (n : ℕ) (m : ℕ) : Prop :=
  m ∈ V n ∧ ¬∃ (p q : ℕ), p ∈ V n ∧ q ∈ V n ∧ p * q = m

theorem exists_multiple_representations (n : ℕ) (h : 2 < n) :
  ∃ r ∈ V n, ∃ s t u v : ℕ, 
    indecomposable n s ∧ indecomposable n t ∧ indecomposable n u ∧ indecomposable n v ∧ 
    r = s * t ∧ r = u * v ∧ (s ≠ u ∨ t ≠ v) :=
sorry

end exists_multiple_representations_l5_5626


namespace part1_area_quadrilateral_part2_maximized_line_equation_l5_5738

noncomputable def area_MA_NB (α : ℝ) : ℝ :=
  (352 * Real.sqrt 33) / 9 * (abs (Real.sin α - Real.cos α)) / (16 - 5 * Real.cos α ^ 2)

theorem part1_area_quadrilateral (α : ℝ) :
  area_MA_NB α = (352 * Real.sqrt 33) / 9 * (abs (Real.sin α - Real.cos α)) / (16 - 5 * Real.cos α ^ 2) :=
by sorry

theorem part2_maximized_line_equation :
  ∃ α : ℝ, area_MA_NB α = (352 * Real.sqrt 33) / 9 * (abs (Real.sin α - Real.cos α)) / (16 - 5 * Real.cos α ^ 2)
    ∧ (Real.tan α = -1 / 2) ∧ (∀ x : ℝ, x = -1 / 2 * y + Real.sqrt 5 / 2) :=
by sorry

end part1_area_quadrilateral_part2_maximized_line_equation_l5_5738


namespace min_dot_product_l5_5238

theorem min_dot_product (m n : ℝ) (x1 x2 : ℝ)
    (h1 : m ≠ 0) 
    (h2 : n ≠ 0)
    (h3 : (x1 + 2) * (x2 - x1) + m * x1 * (n - m * x1) = 0) :
    ∃ (x1 : ℝ), (x1 = -2 / (m^2 + 1)) → 
    (x1 + 2) * (x2 + 2) + m * n * x1 = 4 * m^2 / (m^2 + 1) := 
sorry

end min_dot_product_l5_5238


namespace three_digit_square_ends_with_self_l5_5161

theorem three_digit_square_ends_with_self (A : ℕ) (hA1 : 100 ≤ A) (hA2 : A ≤ 999) (hA3 : A^2 % 1000 = A) : 
  A = 376 ∨ A = 625 :=
sorry

end three_digit_square_ends_with_self_l5_5161


namespace isosceles_triangle_area_of_triangle_l5_5037

variables {A B C : ℝ} {a b c : ℝ}

-- Conditions
axiom triangle_sides (a b c : ℝ) (A B C : ℝ) : c = 2
axiom cosine_condition (a b c : ℝ) (A B C : ℝ) : b^2 - 2 * b * c * Real.cos A = a^2 - 2 * a * c * Real.cos B

-- Questions
theorem isosceles_triangle (a b c : ℝ) (A B C : ℝ)
  (h1 : c = 2) 
  (h2 : b^2 - 2 * b * c * Real.cos A = a^2 - 2 * a * c * Real.cos B) :
  a = b :=
sorry

theorem area_of_triangle (a b c : ℝ) (A B C : ℝ) 
  (h1 : c = 2) 
  (h2 : b^2 - 2 * b * c * Real.cos A = a^2 - 2 * a * c * Real.cos B)
  (h3 : 7 * Real.cos B = 2 * Real.cos C) 
  (h4 : a = b) :
  ∃ S : ℝ, S = Real.sqrt 15 :=
sorry

end isosceles_triangle_area_of_triangle_l5_5037


namespace a_n_is_perfect_square_l5_5370

def sequence_c (n : ℕ) : ℤ :=
  if n = 0 then 1
  else if n = 1 then 0
  else if n = 2 then 2005
  else -3 * sequence_c (n - 2) - 4 * sequence_c (n - 3) + 2008

def sequence_a (n : ℕ) :=
  if n < 2 then 0
  else 5 * (sequence_c (n + 2) - sequence_c n) * (502 - sequence_c (n - 1) - sequence_c (n - 2)) + (4 ^ n) * 2004 * 501

theorem a_n_is_perfect_square (n : ℕ) (h : n > 2) : ∃ k : ℤ, sequence_a n = k^2 :=
sorry

end a_n_is_perfect_square_l5_5370


namespace bicyclist_speed_remainder_l5_5351

noncomputable def speed_of_bicyclist (total_distance first_distance remaining_distance time_for_first_distance total_time : ℝ) : ℝ :=
  remaining_distance / (total_time - time_for_first_distance)

theorem bicyclist_speed_remainder 
  (total_distance : ℝ)
  (first_distance : ℝ)
  (remaining_distance : ℝ)
  (first_speed : ℝ)
  (average_speed : ℝ)
  (correct_speed : ℝ) :
  total_distance = 250 → 
  first_distance = 100 →
  remaining_distance = total_distance - first_distance →
  first_speed = 20 →
  average_speed = 16.67 →
  correct_speed = 15 →
  speed_of_bicyclist total_distance first_distance remaining_distance (first_distance / first_speed) (total_distance / average_speed) = correct_speed :=
by
  sorry

end bicyclist_speed_remainder_l5_5351


namespace ratio_of_40_to_8_l5_5578

theorem ratio_of_40_to_8 : 40 / 8 = 5 := 
by
  sorry

end ratio_of_40_to_8_l5_5578


namespace find_f_prime_zero_l5_5484

noncomputable def f (a : ℝ) (fd0 : ℝ) (x : ℝ) : ℝ :=
  (a * x^2 + x - 1) * Real.exp x + fd0

theorem find_f_prime_zero (a fd0 : ℝ) : (deriv (f a fd0) 0 = 0) :=
by
  -- the proof would go here
  sorry

end find_f_prime_zero_l5_5484


namespace measure_8_cm_measure_5_cm_1_measure_5_cm_2_l5_5661

theorem measure_8_cm:
  ∃ n : ℕ, n * (11 - 7) = 8 := by
  sorry

theorem measure_5_cm_1:
  ∃ x : ℕ, ∃ y : ℕ, x * ((11 - 7) * 2) - y * 7 = 5 := by
  sorry

theorem measure_5_cm_2:
  3 * 11 - 4 * 7 = 5 := by
  sorry

end measure_8_cm_measure_5_cm_1_measure_5_cm_2_l5_5661


namespace triangle_perimeter_from_medians_l5_5730

theorem triangle_perimeter_from_medians (m1 m2 m3 : ℕ) (h1 : m1 = 3) (h2 : m2 = 4) (h3 : m3 = 6) :
  ∃ (p : ℕ), p = 26 :=
by sorry

end triangle_perimeter_from_medians_l5_5730


namespace relationship_between_a_and_b_l5_5907

theorem relationship_between_a_and_b : 
  ∀ (a b : ℝ), (∀ x y : ℝ, (x-a)^2 + (y-b)^2 = b^2 + 1 → (x+1)^2 + (y+1)^2 = 4 → (2 + 2*a)*x + (2 + 2*b)*y - a^2 - 1 = 0) → a^2 + 2*a + 2*b + 5 = 0 :=
by
  intros a b hyp
  sorry

end relationship_between_a_and_b_l5_5907


namespace value_of_sum_l5_5428

theorem value_of_sum (x y z : ℝ) 
    (h1 : x + 2*y + 3*z = 10) 
    (h2 : 4*x + 3*y + 2*z = 15) : 
    x + y + z = 5 :=
by
    sorry

end value_of_sum_l5_5428


namespace slope_of_AB_l5_5792

theorem slope_of_AB (A B : (ℕ × ℕ)) (hA : A = (3, 4)) (hB : B = (2, 3)) : 
  (B.2 - A.2) / (B.1 - A.1) = 1 := 
by 
  sorry

end slope_of_AB_l5_5792


namespace remainder_18_pow_63_mod_5_l5_5615

theorem remainder_18_pow_63_mod_5 :
  (18:ℤ) ^ 63 % 5 = 2 :=
by
  -- Given conditions
  have h1 : (18:ℤ) % 5 = 3 := by norm_num
  have h2 : (3:ℤ) ^ 4 % 5 = 1 := by norm_num
  sorry

end remainder_18_pow_63_mod_5_l5_5615


namespace value_of_x_l5_5624

theorem value_of_x (x : ℝ) (h : x = 52 * (1 + 20 / 100)) : x = 62.4 :=
by sorry

end value_of_x_l5_5624


namespace thirteen_pow_2011_mod_100_l5_5446

theorem thirteen_pow_2011_mod_100 : (13^2011) % 100 = 37 := by
  sorry

end thirteen_pow_2011_mod_100_l5_5446


namespace intersection_of_A_and_B_l5_5333

def A : Set ℚ := { x | x^2 - 4*x + 3 < 0 }
def B : Set ℚ := { x | 2 < x ∧ x < 4 }

theorem intersection_of_A_and_B : A ∩ B = { x | 2 < x ∧ x < 3 } := by
  sorry

end intersection_of_A_and_B_l5_5333


namespace sum_of_coordinates_l5_5271

theorem sum_of_coordinates (f : ℝ → ℝ) (h : f 2 = 4) :
  let x := 4
  let y := (f⁻¹ x) / 4
  x + y = 9 / 2 :=
by
  sorry

end sum_of_coordinates_l5_5271


namespace quadratic_distinct_roots_example_l5_5664

theorem quadratic_distinct_roots_example {b c : ℝ} (hb : b = 1) (hc : c = 0) :
    (b^2 - 4 * c) > 0 := by
  sorry

end quadratic_distinct_roots_example_l5_5664


namespace largest_of_consecutive_odds_l5_5577

-- Defining the six consecutive odd numbers
def consecutive_odd_numbers (a b c d e f : ℕ) : Prop :=
  (a = b + 2) ∧ (b = c + 2) ∧ (c = d + 2) ∧ (d = e + 2) ∧ (e = f + 2)

-- Defining the product condition
def product_of_odds (a b c d e f : ℕ) : Prop :=
  a * b * c * d * e * f = 135135

-- Defining the odd numbers greater than zero
def positive_odds (a b c d e f : ℕ) : Prop :=
  (a > 0) ∧ (b > 0) ∧ (c > 0) ∧ (d > 0) ∧ (e > 0) ∧ (f > 0) ∧
  (a % 2 = 1) ∧ (b % 2 = 1) ∧ (c % 2 = 1) ∧ (d % 2 = 1) ∧ (e % 2 = 1) ∧ (f % 2 = 1)

-- Theorem
theorem largest_of_consecutive_odds (a b c d e f : ℕ) 
  (h1 : consecutive_odd_numbers a b c d e f)
  (h2 : product_of_odds a b c d e f)
  (h3 : positive_odds a b c d e f) : 
  a = 13 :=
sorry

end largest_of_consecutive_odds_l5_5577


namespace possible_numbers_erased_one_digit_reduce_sixfold_l5_5513

theorem possible_numbers_erased_one_digit_reduce_sixfold (N : ℕ) :
  (∃ N' : ℕ, N = 6 * N' ∧ N % 10 ≠ 0 ∧ ¬N = N') ↔
  N = 12 ∨ N = 24 ∨ N = 36 ∨ N = 48 ∨ N = 108 :=
by {
  sorry
}

end possible_numbers_erased_one_digit_reduce_sixfold_l5_5513


namespace max_books_borrowed_l5_5986

theorem max_books_borrowed (total_students : ℕ) (no_books : ℕ) (one_book : ℕ)
  (two_books : ℕ) (at_least_three_books : ℕ) (avg_books_per_student : ℕ) :
  total_students = 35 →
  no_books = 2 →
  one_book = 12 →
  two_books = 10 →
  avg_books_per_student = 2 →
  total_students - (no_books + one_book + two_books) = at_least_three_books →
  ∃ max_books_borrowed_by_individual, max_books_borrowed_by_individual = 8 :=
by
  intros h_total_students h_no_books h_one_book h_two_books h_avg_books_per_student h_remaining_students
  -- Skipping the proof steps
  sorry

end max_books_borrowed_l5_5986


namespace equation_three_no_real_roots_l5_5430

theorem equation_three_no_real_roots
  (a₁ a₂ a₃ : ℝ)
  (h₁ : a₁^2 - 4 ≥ 0)
  (h₂ : a₂^2 - 8 < 0)
  (h₃ : a₂^2 = a₁ * a₃) :
  a₃^2 - 16 < 0 :=
sorry

end equation_three_no_real_roots_l5_5430


namespace grid_area_l5_5952

theorem grid_area :
  let B := 10   -- Number of boundary points
  let I := 12   -- Number of interior points
  I + B / 2 - 1 = 16 :=
by
  sorry

end grid_area_l5_5952


namespace average_hamburgers_per_day_l5_5964

def total_hamburgers : ℕ := 63
def days_in_week : ℕ := 7
def average_per_day : ℕ := total_hamburgers / days_in_week

theorem average_hamburgers_per_day : average_per_day = 9 := by
  sorry

end average_hamburgers_per_day_l5_5964


namespace map_distance_representation_l5_5339

-- Define the conditions and the question as a Lean statement
theorem map_distance_representation :
  (∀ (length_cm : ℕ), (length_cm : ℕ) = 23 → (length_cm * 50 / 10 : ℕ) = 115) :=
by
  sorry

end map_distance_representation_l5_5339


namespace brad_weighs_more_l5_5634

theorem brad_weighs_more :
  ∀ (Billy Brad Carl : ℕ), 
    (Billy = Brad + 9) → 
    (Carl = 145) → 
    (Billy = 159) → 
    (Brad - Carl = 5) :=
by
  intros Billy Brad Carl h1 h2 h3
  sorry

end brad_weighs_more_l5_5634


namespace find_a_max_min_f_l5_5335

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x * Real.exp x

theorem find_a (a : ℝ) (h : (deriv (f a) 0 = 1)) : a = 1 :=
by sorry

noncomputable def f_one (x : ℝ) : ℝ := f 1 x

theorem max_min_f (h : ∀ x, 0 ≤ x → x ≤ 2 → deriv (f_one) x > 0) :
  (f_one 0 = 0) ∧ (f_one 2 = 2 * Real.exp 2) :=
by sorry

end find_a_max_min_f_l5_5335


namespace anna_baked_60_cupcakes_l5_5911

variable (C : ℕ)
variable (h1 : (1/5 : ℚ) * C - 3 = 9)

theorem anna_baked_60_cupcakes (h1 : (1/5 : ℚ) * C - 3 = 9) : C = 60 :=
sorry

end anna_baked_60_cupcakes_l5_5911


namespace Ivan_increases_share_more_than_six_times_l5_5089

theorem Ivan_increases_share_more_than_six_times
  (p v s i : ℝ)
  (hp : p / (v + s + i) = 3 / 7)
  (hv : v / (p + s + i) = 1 / 3)
  (hs : s / (p + v + i) = 1 / 3) :
  ∃ k : ℝ, k > 6 ∧ i * k > 0.6 * (p + v + s + i * k) :=
by
  sorry

end Ivan_increases_share_more_than_six_times_l5_5089


namespace factorization_identity_l5_5095

theorem factorization_identity (m : ℝ) : 
  -4 * m^3 + 4 * m^2 - m = -m * (2 * m - 1)^2 :=
sorry

end factorization_identity_l5_5095


namespace sum_of_four_powers_l5_5502

theorem sum_of_four_powers (a : ℕ) : 4 * a^3 = 500 :=
by
  rw [Nat.pow_succ, Nat.pow_succ]
  sorry

end sum_of_four_powers_l5_5502


namespace A_alone_finishes_in_27_days_l5_5300

noncomputable def work (B : ℝ) : ℝ := 54 * B  -- amount of work W
noncomputable def days_to_finish_alone (B : ℝ) : ℝ := (work B) / (2 * B)

theorem A_alone_finishes_in_27_days (B : ℝ) (h : (work B) / (2 * B + B) = 18) : 
  days_to_finish_alone B = 27 :=
by
  sorry

end A_alone_finishes_in_27_days_l5_5300


namespace smallest_integer_x_l5_5273

-- Conditions
def condition1 (x : ℤ) : Prop := 7 - 5 * x < 25
def condition2 (x : ℤ) : Prop := ∃ y : ℤ, y = 10 ∧ y - 3 * x > 6

-- Statement
theorem smallest_integer_x : ∃ x : ℤ, condition1 x ∧ condition2 x ∧ ∀ z : ℤ, condition1 z ∧ condition2 z → x ≤ z :=
  sorry

end smallest_integer_x_l5_5273


namespace problem_l5_5767

theorem problem (a b : ℝ) (n : ℕ) (ha : a > 0) (hb : b > 0) 
  (h1 : 1 / a + 1 / b = 1) : 
  (a + b)^n - a^n - b^n ≥ 2^(2*n) - 2^(n + 1) := 
by
  sorry

end problem_l5_5767


namespace polynomial_remainder_is_zero_l5_5399

theorem polynomial_remainder_is_zero :
  ∀ (x : ℤ), ((x^5 - 1) * (x^3 - 1)) % (x^2 + x + 1) = 0 := 
by
  sorry

end polynomial_remainder_is_zero_l5_5399


namespace puzzles_sold_eq_36_l5_5105

def n_science_kits : ℕ := 45
def n_puzzles : ℕ := n_science_kits - 9

theorem puzzles_sold_eq_36 : n_puzzles = 36 := by
  sorry

end puzzles_sold_eq_36_l5_5105


namespace find_range_m_l5_5463

noncomputable def p (m : ℝ) : Prop :=
  ∃ x₀ : ℝ, m * x₀^2 + 1 < 1

def q (m : ℝ) : Prop :=
  ∀ x : ℝ, x^2 + m * x + 1 ≥ 0

theorem find_range_m (m : ℝ) : ¬ (p m ∨ ¬ q m) ↔ -2 ≤ m ∧ m ≤ 2 :=
  sorry

end find_range_m_l5_5463


namespace radius_B_l5_5042

noncomputable def radius_A := 2
noncomputable def radius_D := 4

theorem radius_B (r_B : ℝ) (x y : ℝ) 
  (h1 : (2 : ℝ) + y = x + (x^2 / 4)) 
  (h2 : y = 2 - (x^2 / 8)) 
  (h3 : x = (4: ℝ) / 3) 
  (h4 : y = x + (x^2 / 4)) : r_B = 20 / 9 :=
sorry

end radius_B_l5_5042


namespace numeral_is_1_11_l5_5763

-- Define the numeral question and condition
def place_value_difference (a b : ℝ) : Prop :=
  10 * b - b = 99.99

-- Now we define the problem statement in Lean
theorem numeral_is_1_11 (a b : ℝ) (h : place_value_difference a b) : 
  a = 100 ∧ b = 11.11 ∧ (a - b = 99.99) :=
  sorry

end numeral_is_1_11_l5_5763


namespace shaded_area_proof_l5_5680

-- Given Definitions
def rectangle_area (length : ℕ) (width : ℕ) : ℕ := length * width
def triangle_area (base : ℕ) (height : ℕ) : ℕ := (base * height) / 2

-- Conditions
def grid_area : ℕ :=
  rectangle_area 2 3 + rectangle_area 3 4 + rectangle_area 4 5

def unshaded_triangle_area : ℕ := triangle_area 12 4

-- Question
def shaded_area : ℕ := grid_area - unshaded_triangle_area

-- Proof statement
theorem shaded_area_proof : shaded_area = 14 := by
  sorry

end shaded_area_proof_l5_5680


namespace best_fit_model_l5_5564

theorem best_fit_model
  (R2_M1 R2_M2 R2_M3 R2_M4 : ℝ)
  (h1 : R2_M1 = 0.78)
  (h2 : R2_M2 = 0.85)
  (h3 : R2_M3 = 0.61)
  (h4 : R2_M4 = 0.31) :
  ∀ i, (i = 2 ∧ R2_M2 ≥ R2_M1 ∧ R2_M2 ≥ R2_M3 ∧ R2_M2 ≥ R2_M4) := 
sorry

end best_fit_model_l5_5564


namespace product_of_roots_eq_neg_125_over_4_l5_5511

theorem product_of_roots_eq_neg_125_over_4 :
  (∀ x y : ℝ, (24 * x^2 + 60 * x - 750 = 0 ∧ 24 * y^2 + 60 * y - 750 = 0 ∧ x ≠ y) → x * y = -125 / 4) :=
by
  intro x y h
  sorry

end product_of_roots_eq_neg_125_over_4_l5_5511


namespace new_avg_weight_l5_5492

theorem new_avg_weight 
  (initial_avg_weight : ℝ)
  (initial_num_members : ℕ)
  (new_person1_weight : ℝ)
  (new_person2_weight : ℝ)
  (new_num_members : ℕ)
  (final_total_weight : ℝ)
  (final_avg_weight : ℝ) :
  initial_avg_weight = 48 →
  initial_num_members = 23 →
  new_person1_weight = 78 →
  new_person2_weight = 93 →
  new_num_members = initial_num_members + 2 →
  final_total_weight = (initial_avg_weight * initial_num_members) + new_person1_weight + new_person2_weight →
  final_avg_weight = final_total_weight / new_num_members →
  final_avg_weight = 51 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end new_avg_weight_l5_5492


namespace multiplication_of_exponents_l5_5439

theorem multiplication_of_exponents (x : ℝ) : (x ^ 4) * (x ^ 2) = x ^ 6 := 
by
  sorry

end multiplication_of_exponents_l5_5439


namespace mario_total_flowers_l5_5739

-- Define the number of flowers on the first plant
def F1 : ℕ := 2

-- Define the number of flowers on the second plant as twice the first
def F2 : ℕ := 2 * F1

-- Define the number of flowers on the third plant as four times the second
def F3 : ℕ := 4 * F2

-- Prove that total number of flowers is 22
theorem mario_total_flowers : F1 + F2 + F3 = 22 := by
  -- Proof is to be filled here
  sorry

end mario_total_flowers_l5_5739


namespace max_f_of_sin_bounded_l5_5849

theorem max_f_of_sin_bounded (x : ℝ) : (∀ y, -1 ≤ Real.sin y ∧ Real.sin y ≤ 1) → ∃ m, (∀ z, (1 + 2 * Real.sin z) ≤ m) ∧ (∀ n, (∀ z, (1 + 2 * Real.sin z) ≤ n) → m ≤ n) :=
by
  sorry

end max_f_of_sin_bounded_l5_5849


namespace cost_of_fencing_l5_5209

-- Definitions of ratio and area conditions
def sides_ratio (length width : ℕ) : Prop := length / width = 3 / 2
def area (length width : ℕ) : Prop := length * width = 3750

-- Define the cost per meter in paise
def cost_per_meter : ℕ := 70

-- Convert paise to rupees
def paise_to_rupees (paise : ℕ) : ℕ := paise / 100

-- The main statement we want to prove
theorem cost_of_fencing (length width perimeter : ℕ)
  (H1 : sides_ratio length width)
  (H2 : area length width)
  (H3 : perimeter = 2 * length + 2 * width) :
  paise_to_rupees (perimeter * cost_per_meter) = 175 := by
  sorry

end cost_of_fencing_l5_5209


namespace max_possible_n_l5_5514

theorem max_possible_n (n : ℤ) (h : 101 * n ^ 2 ≤ 6400) : n ≤ 7 :=
by {
  sorry
}

end max_possible_n_l5_5514


namespace equation_solution_1_equation_solution_2_equation_solution_3_l5_5557

def system_of_equations (x y : ℝ) : Prop :=
  (x * (x^2 - 3 * y^2) = 16) ∧ (y * (3 * x^2 - y^2) = 88)

theorem equation_solution_1 :
  system_of_equations 4 2 :=
by
  -- The proof is skipped.
  sorry

theorem equation_solution_2 :
  system_of_equations (-3.7) 2.5 :=
by
  -- The proof is skipped.
  sorry

theorem equation_solution_3 :
  system_of_equations (-0.3) (-4.5) :=
by
  -- The proof is skipped.
  sorry

end equation_solution_1_equation_solution_2_equation_solution_3_l5_5557


namespace complementSetM_l5_5135

open Set Real

-- The universal set U is the set of all real numbers
def universalSet : Set ℝ := univ

-- The set M is defined as {x | |x - 1| ≤ 2}
def setM : Set ℝ := {x : ℝ | |x - 1| ≤ 2}

-- We need to prove that the complement of M with respect to U is {x | x < -1 ∨ x > 3}
theorem complementSetM :
  (universalSet \ setM) = {x : ℝ | x < -1 ∨ x > 3} :=
by
  sorry

end complementSetM_l5_5135


namespace determine_a_b_l5_5718

theorem determine_a_b (a b : ℝ) :
  (∀ x, y = x^2 + a * x + b) ∧ (∀ t, t = 0 → 3 * t - (t^2 + a * t + b) + 1 = 0) →
  a = 3 ∧ b = 1 :=
by
  sorry

end determine_a_b_l5_5718


namespace point_in_fourth_quadrant_l5_5172

def point : ℝ × ℝ := (4, -3)

def is_fourth_quadrant (p : ℝ × ℝ) : Prop :=
  p.1 > 0 ∧ p.2 < 0

theorem point_in_fourth_quadrant : is_fourth_quadrant point :=
by
  sorry

end point_in_fourth_quadrant_l5_5172


namespace nublian_total_words_l5_5894

-- Define the problem's constants and conditions
def nublian_alphabet_size := 6
def word_length_one := nublian_alphabet_size
def word_length_two := nublian_alphabet_size * nublian_alphabet_size
def word_length_three := nublian_alphabet_size * nublian_alphabet_size * nublian_alphabet_size

-- Define the total number of words
def total_words := word_length_one + word_length_two + word_length_three

-- Main theorem statement
theorem nublian_total_words : total_words = 258 := by
  sorry

end nublian_total_words_l5_5894


namespace percent_first_question_l5_5799

variable (A B : ℝ) (A_inter_B : ℝ) (A_union_B : ℝ)

-- Given conditions
def condition1 : B = 0.49 := sorry
def condition2 : A_inter_B = 0.32 := sorry
def condition3 : A_union_B = 0.80 := sorry
def union_formula : A_union_B = A + B - A_inter_B := 
by sorry

-- Prove that A = 0.63
theorem percent_first_question (h1 : B = 0.49) 
                               (h2 : A_inter_B = 0.32) 
                               (h3 : A_union_B = 0.80) 
                               (h4 : A_union_B = A + B - A_inter_B) : 
                               A = 0.63 :=
by sorry

end percent_first_question_l5_5799


namespace find_line_l_l5_5017

def line_equation (x y: ℤ) : Prop := x - 2 * y = 2

def scaling_transform_x (x: ℤ) : ℤ := x
def scaling_transform_y (y: ℤ) : ℤ := 2 * y

theorem find_line_l :
  ∀ (x y x' y': ℤ),
  x' = scaling_transform_x x →
  y' = scaling_transform_y y →
  line_equation x y →
  x' - y' = 2 := by
  sorry

end find_line_l_l5_5017


namespace cube_eq_minus_one_l5_5158

theorem cube_eq_minus_one (x : ℝ) (h : x = -2) : (x + 1) ^ 3 = -1 :=
by
  sorry

end cube_eq_minus_one_l5_5158


namespace balls_distribution_l5_5544

theorem balls_distribution : 
  ∃ (n : ℕ), 
    (∀ (b1 b2 : ℕ), ∀ (h : b1 + b2 = 4), b1 ≥ 1 ∧ b2 ≥ 2 → n = 10) :=
sorry

end balls_distribution_l5_5544


namespace largest_value_l5_5619

def expr_A : ℕ := 3 + 1 + 0 + 5
def expr_B : ℕ := 3 * 1 + 0 + 5
def expr_C : ℕ := 3 + 1 * 0 + 5
def expr_D : ℕ := 3 * 1 + 0 * 5
def expr_E : ℕ := 3 * 1 + 0 * 5 * 3

theorem largest_value :
  expr_A > expr_B ∧
  expr_A > expr_C ∧
  expr_A > expr_D ∧
  expr_A > expr_E :=
by
  sorry

end largest_value_l5_5619


namespace customer_total_payment_l5_5251

def Riqing_Beef_Noodles_quantity : ℕ := 24
def Riqing_Beef_Noodles_price_per_bag : ℝ := 1.80
def Riqing_Beef_Noodles_discount : ℝ := 0.8

def Kang_Shifu_Ice_Red_Tea_quantity : ℕ := 6
def Kang_Shifu_Ice_Red_Tea_price_per_box : ℝ := 1.70
def Kang_Shifu_Ice_Red_Tea_discount : ℝ := 0.8

def Shanlin_Purple_Cabbage_Soup_quantity : ℕ := 5
def Shanlin_Purple_Cabbage_Soup_price_per_bag : ℝ := 3.40

def Shuanghui_Ham_Sausage_quantity : ℕ := 3
def Shuanghui_Ham_Sausage_price_per_bag : ℝ := 11.20
def Shuanghui_Ham_Sausage_discount : ℝ := 0.9

def total_price : ℝ :=
  (Riqing_Beef_Noodles_quantity * Riqing_Beef_Noodles_price_per_bag * Riqing_Beef_Noodles_discount) +
  (Kang_Shifu_Ice_Red_Tea_quantity * Kang_Shifu_Ice_Red_Tea_price_per_box * Kang_Shifu_Ice_Red_Tea_discount) +
  (Shanlin_Purple_Cabbage_Soup_quantity * Shanlin_Purple_Cabbage_Soup_price_per_bag) +
  (Shuanghui_Ham_Sausage_quantity * Shuanghui_Ham_Sausage_price_per_bag * Shuanghui_Ham_Sausage_discount)

theorem customer_total_payment :
  total_price = 89.96 :=
by
  unfold total_price
  sorry

end customer_total_payment_l5_5251


namespace total_students_l5_5147

theorem total_students (n1 n2 : ℕ) (h1 : (158 - 140)/(n1 + 1) = 2) (h2 : (158 - 140)/(n2 + 1) = 3) :
  n1 + n2 + 2 = 15 :=
sorry

end total_students_l5_5147


namespace tetrahedron_edge_length_correct_l5_5988

noncomputable def radius := Real.sqrt 2
noncomputable def center_to_center_distance := 2 * radius
noncomputable def tetrahedron_edge_length := center_to_center_distance

theorem tetrahedron_edge_length_correct :
  tetrahedron_edge_length = 2 * Real.sqrt 2 := by
  sorry

end tetrahedron_edge_length_correct_l5_5988


namespace value_of_ab_l5_5258

theorem value_of_ab (a b c : ℝ) (C : ℝ) (h1 : (a + b) ^ 2 - c ^ 2 = 4) (h2 : C = Real.pi / 3) : 
  a * b = 4 / 3 :=
by
  sorry

end value_of_ab_l5_5258


namespace find_divisor_l5_5687

noncomputable def divisor_of_nearest_divisible (a b : ℕ) (d : ℕ) : ℕ :=
  if h : b % d = 0 ∧ (b - a < d) then d else 0

theorem find_divisor (a b : ℕ) (d : ℕ) (h1 : b = 462) (h2 : a = 457)
  (h3 : b % d = 0) (h4 : b - a < d) :
  d = 5 :=
sorry

end find_divisor_l5_5687


namespace area_quadrilateral_ABCDE_correct_l5_5171

noncomputable def area_quadrilateral_ABCDE (AM NM AN BN BO OC CP CD EP DE : ℝ) : ℝ :=
  (0.5 * AM * NM * Real.sqrt 2) + (0.5 * BN * BO) + (0.5 * OC * CP * Real.sqrt 2) - (0.5 * DE * EP)

theorem area_quadrilateral_ABCDE_correct :
  ∀ (AM NM AN BN BO OC CP CD EP DE : ℝ),
    DE = 12 ∧ 
    AM = 36 ∧ 
    NM = 36 ∧ 
    AN = 36 * Real.sqrt 2 ∧
    BN = 36 * Real.sqrt 2 - 36 ∧
    BO = 36 ∧
    OC = 36 ∧
    CP = 36 * Real.sqrt 2 ∧
    CD = 24 ∧
    EP = 24
    → area_quadrilateral_ABCDE AM NM AN BN BO OC CP CD EP DE = 2311.2 * Real.sqrt 2 + 504 :=
by intro AM NM AN BN BO OC CP CD EP DE h;
   cases h;
   sorry

end area_quadrilateral_ABCDE_correct_l5_5171


namespace list_price_is_35_l5_5284

-- Define the conditions in Lean
variable (x : ℝ)

def alice_selling_price (x : ℝ) : ℝ := x - 15
def alice_commission (x : ℝ) : ℝ := 0.15 * (alice_selling_price x)

def bob_selling_price (x : ℝ) : ℝ := x - 20
def bob_commission (x : ℝ) : ℝ := 0.20 * (bob_selling_price x)

-- Define the theorem to be proven
theorem list_price_is_35 (x : ℝ) 
  (h : alice_commission x = bob_commission x) : x = 35 :=
by sorry

end list_price_is_35_l5_5284


namespace sophie_germain_identity_l5_5488

theorem sophie_germain_identity (a b : ℝ) : 
  a^4 + 4 * b^4 = (a^2 + 2 * a * b + 2 * b^2) * (a^2 - 2 * a * b + 2 * b^2) :=
by sorry

end sophie_germain_identity_l5_5488


namespace rectangle_area_l5_5900

def length : ℝ := 2
def width : ℝ := 4
def area := length * width

theorem rectangle_area : area = 8 := 
by
  -- Proof can be written here
  sorry

end rectangle_area_l5_5900


namespace bronze_needed_l5_5929

/-- 
The total amount of bronze Martin needs for three bells in pounds.
-/
theorem bronze_needed (w1 w2 w3 : ℕ) 
  (h1 : w1 = 50) 
  (h2 : w2 = 2 * w1) 
  (h3 : w3 = 4 * w2) 
  : (w1 + w2 + w3 = 550) := 
by { 
  sorry 
}

end bronze_needed_l5_5929


namespace ex1_simplified_ex2_simplified_l5_5325

-- Definitions and problem setup
def ex1 (a : ℝ) : ℝ := ((-a^3)^2 * a^3 - 4 * a^2 * a^7)
def ex2 (a : ℝ) : ℝ := (2 * a + 1) * (-2 * a + 1)

-- Proof goals
theorem ex1_simplified (a : ℝ) : ex1 a = -3 * a^9 :=
by sorry

theorem ex2_simplified (a : ℝ) : ex2 a = 4 * a^2 - 1 :=
by sorry

end ex1_simplified_ex2_simplified_l5_5325


namespace students_with_uncool_parents_l5_5790

theorem students_with_uncool_parents (class_size : ℕ) (cool_dads : ℕ) (cool_moms : ℕ) (both_cool_parents : ℕ) : 
  class_size = 40 → cool_dads = 18 → cool_moms = 20 → both_cool_parents = 10 → 
  (class_size - (cool_dads - both_cool_parents + cool_moms - both_cool_parents + both_cool_parents) = 12) :=
by
  sorry

end students_with_uncool_parents_l5_5790


namespace problem_x_y_z_l5_5183

theorem problem_x_y_z (x y z : ℕ) (h1 : xy + z = 47) (h2 : yz + x = 47) (h3 : xz + y = 47) : x + y + z = 48 :=
sorry

end problem_x_y_z_l5_5183


namespace number_of_possible_heights_is_680_l5_5016

noncomputable def total_possible_heights : Nat :=
  let base_height := 200 * 3
  let max_additional_height := 200 * (20 - 3)
  let min_height := base_height
  let max_height := base_height + max_additional_height
  let number_of_possible_heights := (max_height - min_height) / 5 + 1
  number_of_possible_heights

theorem number_of_possible_heights_is_680 : total_possible_heights = 680 := by
  sorry

end number_of_possible_heights_is_680_l5_5016


namespace brother_and_sister_ages_l5_5137

theorem brother_and_sister_ages :
  ∃ (b s : ℕ), (b - 3 = 7 * (s - 3)) ∧ (b - 2 = 4 * (s - 2)) ∧ (b - 1 = 3 * (s - 1)) ∧ (b = 5 / 2 * s) ∧ b = 10 ∧ s = 4 :=
by 
  sorry

end brother_and_sister_ages_l5_5137


namespace condition_eq_l5_5895

-- We are given a triangle ABC with sides opposite angles A, B, and C being a, b, and c respectively.
variable (A B C a b c : ℝ)

-- Conditions for the problem
def sin_eq (A B : ℝ) := Real.sin A = Real.sin B
def cos_eq (A B : ℝ) := Real.cos A = Real.cos B
def sin2_eq (A B : ℝ) := Real.sin (2 * A) = Real.sin (2 * B)
def cos2_eq (A B : ℝ) := Real.cos (2 * A) = Real.cos (2 * B)

-- The main statement we need to prove
theorem condition_eq (h1 : sin_eq A B) (h2 : cos_eq A B) (h4 : cos2_eq A B) : a = b :=
sorry

end condition_eq_l5_5895


namespace total_expenses_l5_5092

def tulips : ℕ := 250
def carnations : ℕ := 375
def roses : ℕ := 320
def cost_per_flower : ℕ := 2

theorem total_expenses :
  tulips + carnations + roses * cost_per_flower = 1890 := 
sorry

end total_expenses_l5_5092


namespace find_missing_dimension_l5_5080

def carton_volume (l w h : ℕ) : ℕ := l * w * h

def soapbox_base_area (l w : ℕ) : ℕ := l * w

def total_base_area (n l w : ℕ) : ℕ := n * soapbox_base_area l w

def missing_dimension (carton_volume total_base_area : ℕ) : ℕ := carton_volume / total_base_area

theorem find_missing_dimension 
  (carton_l carton_w carton_h : ℕ) 
  (soapbox_l soapbox_w : ℕ) 
  (n : ℕ) 
  (h_carton_l : carton_l = 25)
  (h_carton_w : carton_w = 48)
  (h_carton_h : carton_h = 60)
  (h_soapbox_l : soapbox_l = 8)
  (h_soapbox_w : soapbox_w = 6)
  (h_n : n = 300) :
  missing_dimension (carton_volume carton_l carton_w carton_h) (total_base_area n soapbox_l soapbox_w) = 5 := 
by 
  sorry

end find_missing_dimension_l5_5080


namespace simplify_fraction_l5_5734

theorem simplify_fraction :
  (4 / (Real.sqrt 108 + 2 * Real.sqrt 12 + 2 * Real.sqrt 27)) = (Real.sqrt 3 / 12) := 
by
  -- Proof goes here
  sorry

end simplify_fraction_l5_5734


namespace solve_equation_l5_5281

theorem solve_equation (x : ℝ) : 
  x^2 + 2 * x + 4 * Real.sqrt (x^2 + 2 * x) - 5 = 0 →
  (x = Real.sqrt 2 - 1) ∨ (x = - (Real.sqrt 2 + 1)) :=
by 
  sorry

end solve_equation_l5_5281


namespace shaniqua_earnings_l5_5355

noncomputable def shaniqua_total_earnings : ℕ :=
  let haircut_rate := 12
  let style_rate := 25
  let coloring_rate := 35
  let treatment_rate := 50
  let haircuts := 8
  let styles := 5
  let colorings := 10
  let treatments := 6
  (haircuts * haircut_rate) +
  (styles * style_rate) +
  (colorings * coloring_rate) +
  (treatments * treatment_rate)

theorem shaniqua_earnings : shaniqua_total_earnings = 871 := by
  sorry

end shaniqua_earnings_l5_5355


namespace avg_of_all_5_is_8_l5_5440

-- Let a1, a2, a3 be three quantities such that their average is 4.
def is_avg_4 (a1 a2 a3 : ℝ) : Prop :=
  (a1 + a2 + a3) / 3 = 4

-- Let a4, a5 be the remaining two quantities such that their average is 14.
def is_avg_14 (a4 a5 : ℝ) : Prop :=
  (a4 + a5) / 2 = 14

-- Prove that the average of all 5 quantities is 8.
theorem avg_of_all_5_is_8 (a1 a2 a3 a4 a5 : ℝ) :
  is_avg_4 a1 a2 a3 ∧ is_avg_14 a4 a5 → 
  ((a1 + a2 + a3 + a4 + a5) / 5 = 8) :=
by
  intro h
  sorry

end avg_of_all_5_is_8_l5_5440


namespace find_a_from_perpendicular_lines_l5_5991

theorem find_a_from_perpendicular_lines (a : ℝ) :
  (a * (a + 2) = -1) → a = -1 := 
by 
  sorry

end find_a_from_perpendicular_lines_l5_5991


namespace problem_inequality_l5_5875

theorem problem_inequality 
  (a b c d : ℝ) 
  (h_pos_a : 0 < a) (h_le_a : a ≤ 1)
  (h_pos_b : 0 < b) (h_le_b : b ≤ 1)
  (h_pos_c : 0 < c) (h_le_c : c ≤ 1)
  (h_pos_d : 0 < d) (h_le_d : d ≤ 1) :
  (1 / (a^2 + b^2 + c^2 + d^2)) ≥ (1 / 4) + (1 - a) * (1 - b) * (1 - c) * (1 - d) :=
by
  sorry

end problem_inequality_l5_5875


namespace fraction_product_l5_5306

theorem fraction_product (a b : ℕ) 
  (h1 : 1/5 < a / b)
  (h2 : a / b < 1/4)
  (h3 : b ≤ 19) :
  ∃ a1 a2 b1 b2, 4 * a2 < b1 ∧ b1 < 5 * a2 ∧ b2 ≤ 19 ∧ 4 * a2 < b2 ∧ b2 < 20 ∧ a = 4 ∧ b = 19 ∧ a1 = 2 ∧ b1 = 9 ∧ 
  (a + b = 23 ∨ a + b = 11) ∧ (23 * 11 = 253) := by
  sorry

end fraction_product_l5_5306


namespace greyhound_catches_hare_l5_5549

theorem greyhound_catches_hare {a b : ℝ} (h_speed : b < a) : ∃ t : ℝ, ∀ s : ℝ, ∃ n : ℕ, (n * t * (a - b)) > s + t * (a + b) :=
by
  sorry

end greyhound_catches_hare_l5_5549


namespace find_c_l5_5896

theorem find_c (a b c : ℝ) (h : 1/a + 1/b = 1/c) : c = (a * b) / (a + b) := 
by
  sorry

end find_c_l5_5896


namespace photograph_area_l5_5055

def dimensions_are_valid (a b : ℕ) : Prop :=
a > 0 ∧ b > 0 ∧ (a + 4) * (b + 5) = 77

theorem photograph_area (a b : ℕ) (h : dimensions_are_valid a b) : (a * b = 18 ∨ a * b = 14) :=
by 
  sorry

end photograph_area_l5_5055


namespace percentage_saving_l5_5712

theorem percentage_saving 
  (p_coat p_pants : ℝ)
  (d_coat d_pants : ℝ)
  (h_coat : p_coat = 100)
  (h_pants : p_pants = 50)
  (h_d_coat : d_coat = 0.30)
  (h_d_pants : d_pants = 0.40) :
  (p_coat * d_coat + p_pants * d_pants) / (p_coat + p_pants) = 0.333 :=
by
  sorry

end percentage_saving_l5_5712


namespace perfect_squares_example_l5_5207

def isPerfectSquare (n: ℕ) : Prop := ∃ m: ℕ, m * m = n

theorem perfect_squares_example :
  let a := 10430
  let b := 3970
  let c := 2114
  let d := 386
  isPerfectSquare (a + b) ∧
  isPerfectSquare (a + c) ∧
  isPerfectSquare (a + d) ∧
  isPerfectSquare (b + c) ∧
  isPerfectSquare (b + d) ∧
  isPerfectSquare (c + d) ∧
  isPerfectSquare (a + b + c + d) :=
by
  -- Proof steps go here
  sorry

end perfect_squares_example_l5_5207


namespace simplify_expression_l5_5665

variable {a b : ℝ}

theorem simplify_expression {a b : ℝ} (h : |2 - a + b| + (ab + 1)^2 = 0) :
  (4 * a - 5 * b - a * b) - (2 * a - 3 * b + 5 * a * b) = 10 := by
  sorry

end simplify_expression_l5_5665


namespace smallest_b_l5_5282

theorem smallest_b (a b : ℝ) (h1 : 2 < a) (h2 : a < b) 
(h3 : 2 + a ≤ b) (h4 : 1 / a + 1 / b ≤ 2) : b = 2 :=
sorry

end smallest_b_l5_5282


namespace find_n_divisible_by_highest_power_of_2_l5_5605

def a_n (n : ℕ) : ℕ :=
  10^n * 999 + 488

theorem find_n_divisible_by_highest_power_of_2:
  ∀ n : ℕ, (n > 0) → (a_n n = 10^n * 999 + 488) → (∃ k : ℕ, 2^(k + 9) ∣ a_n 6) := sorry

end find_n_divisible_by_highest_power_of_2_l5_5605


namespace least_number_of_coins_l5_5931

theorem least_number_of_coins : ∃ (n : ℕ), 
  (n % 6 = 3) ∧ 
  (n % 4 = 1) ∧ 
  (n % 7 = 2) ∧ 
  (∀ m : ℕ, (m % 6 = 3) ∧ (m % 4 = 1) ∧ (m % 7 = 2) → n ≤ m) :=
by
  exists 9
  simp
  sorry

end least_number_of_coins_l5_5931


namespace combination_identity_l5_5379

theorem combination_identity : (Nat.choose 5 3 + Nat.choose 5 4 = Nat.choose 6 4) := 
by 
  sorry

end combination_identity_l5_5379


namespace correct_calculation_l5_5800

theorem correct_calculation :
  (∀ x : ℤ, x^5 + x^3 ≠ x^8) ∧
  (∀ x : ℤ, x^5 - x^3 ≠ x^2) ∧
  (∀ x : ℤ, x^5 * x^3 = x^8) ∧
  (∀ x : ℤ, (-3 * x)^3 ≠ -9 * x^3) :=
by
  sorry

end correct_calculation_l5_5800


namespace tangent_line_is_correct_l5_5647

-- Define the curve
def curve (x : ℝ) : ℝ := x^3 - 3 * x^2 + 1

-- Define the point of tangency
def point_of_tangency : ℝ × ℝ := (1, -1)

-- Define the equation of the tangent line
def tangent_line (x : ℝ) : ℝ := -3 * x + 2

-- Statement of the problem (to prove)
theorem tangent_line_is_correct :
  curve point_of_tangency.1 = point_of_tangency.2 ∧
  ∃ m b, (∀ x, (tangent_line x) = m * x + b) ∧
         tangent_line point_of_tangency.1 = point_of_tangency.2 ∧
         (∀ x, deriv (curve) x = -3 ↔ deriv (tangent_line) point_of_tangency.1 = -3) :=
by
  sorry

end tangent_line_is_correct_l5_5647


namespace transylvanian_human_truth_transylvanian_vampire_lie_l5_5521

-- Definitions of predicates for human and vampire behavior
def is_human (A : Type) : Prop := ∀ (X : Prop), (A → X) → X
def is_vampire (A : Type) : Prop := ∀ (X : Prop), (A → X) → ¬X

-- Lean definitions for the problem
theorem transylvanian_human_truth (A : Type) (X : Prop) (h_human : is_human A) (h_says_true : A → X) :
  X :=
by sorry

theorem transylvanian_vampire_lie (A : Type) (X : Prop) (h_vampire : is_vampire A) (h_says_true : A → X) :
  ¬X :=
by sorry

end transylvanian_human_truth_transylvanian_vampire_lie_l5_5521


namespace least_number_to_add_l5_5644

theorem least_number_to_add (x : ℕ) : (1021 + x) % 25 = 0 ↔ x = 4 := 
by 
  sorry

end least_number_to_add_l5_5644


namespace integer_solutions_count_for_equation_l5_5861

theorem integer_solutions_count_for_equation :
  (∃ n : ℕ, (∀ x y : ℤ, (1/x + 1/y = 1/7) → (x ≠ 0) → (y ≠ 0) → n = 5 )) :=
sorry

end integer_solutions_count_for_equation_l5_5861


namespace y_work_duration_l5_5449

theorem y_work_duration (x_rate y_rate : ℝ) (d : ℝ) :
  -- 1. x and y together can do the work in 20 days.
  (x_rate + y_rate = 1/20) →
  -- 2. x started the work alone and after 4 days y joined him till the work completed.
  -- 3. The total work lasted 10 days.
  (4 * x_rate + 6 * (x_rate + y_rate) = 1) →
  -- Prove: y can do the work alone in 12 days.
  y_rate = 1/12 :=
by {
  sorry
}

end y_work_duration_l5_5449


namespace complex_eq_solution_l5_5506

theorem complex_eq_solution (x y : ℝ) (i : ℂ) (h : (2 * x - 1) + i = y - (3 - y) * i) : 
  x = 5 / 2 ∧ y = 4 :=
  sorry

end complex_eq_solution_l5_5506


namespace rate_of_interest_per_annum_l5_5720

def simple_interest (P T R : ℕ) : ℕ :=
  (P * T * R) / 100

theorem rate_of_interest_per_annum :
  let P_B := 5000
  let T_B := 2
  let P_C := 3000
  let T_C := 4
  let total_interest := 1980
  ∃ R : ℕ, 
      simple_interest P_B T_B R + simple_interest P_C T_C R = total_interest ∧
      R = 9 :=
by
  sorry

end rate_of_interest_per_annum_l5_5720


namespace find_fg3_l5_5987

def f (x : ℝ) : ℝ := x^2 + 1
def g (x : ℝ) : ℝ := x^2 - 4*x + 4

theorem find_fg3 : f (g 3) = 2 := by
  sorry

end find_fg3_l5_5987


namespace sum_of_missing_angles_l5_5087

theorem sum_of_missing_angles (angle_sum_known : ℕ) (divisor : ℕ) (total_sides : ℕ) (missing_angles_sum : ℕ)
  (h1 : angle_sum_known = 1620)
  (h2 : divisor = 180)
  (h3 : total_sides = 12)
  (h4 : angle_sum_known + missing_angles_sum = divisor * (total_sides - 2)) :
  missing_angles_sum = 180 :=
by
  -- Skipping the proof for this theorem
  sorry

end sum_of_missing_angles_l5_5087


namespace total_balloons_l5_5224

theorem total_balloons:
  ∀ (R1 R2 G1 G2 B1 B2 Y1 Y2 O1 O2: ℕ),
    R1 = 31 →
    R2 = 24 →
    G1 = 15 →
    G2 = 7 →
    B1 = 12 →
    B2 = 14 →
    Y1 = 18 →
    Y2 = 20 →
    O1 = 10 →
    O2 = 16 →
    (R1 + R2 = 55) ∧
    (G1 + G2 = 22) ∧
    (B1 + B2 = 26) ∧
    (Y1 + Y2 = 38) ∧
    (O1 + O2 = 26) :=
by
  intros
  sorry

end total_balloons_l5_5224


namespace root_in_interval_l5_5173

def polynomial (x : ℝ) := x^3 + 3 * x^2 - x + 1

noncomputable def A : ℤ := -4
noncomputable def B : ℤ := -3

theorem root_in_interval : (∃ x : ℝ, polynomial x = 0 ∧ (A : ℝ) < x ∧ x < (B : ℝ)) :=
sorry

end root_in_interval_l5_5173


namespace production_equation_l5_5483

-- Definitions based on the problem conditions
def original_production_rate (x : ℕ) := x
def additional_parts_per_day := 4
def original_days := 20
def actual_days := 15
def extra_parts := 10

-- Prove the equation
theorem production_equation (x : ℕ) :
  original_days * original_production_rate x = actual_days * (original_production_rate x + additional_parts_per_day) - extra_parts :=
by
  simp [original_production_rate, additional_parts_per_day, original_days, actual_days, extra_parts]
  sorry

end production_equation_l5_5483


namespace calc_expression_l5_5278

theorem calc_expression : 
  abs (Real.sqrt 3 - 2) + (8:ℝ)^(1/3) - Real.sqrt 16 + (-1)^(2023:ℝ) = -(Real.sqrt 3) - 1 :=
by
  sorry

end calc_expression_l5_5278


namespace common_root_conds_l5_5844

theorem common_root_conds (α a b c d : ℝ) (h₁ : a ≠ c)
  (h₂ : α^2 + a * α + b = 0)
  (h₃ : α^2 + c * α + d = 0) :
  α = (d - b) / (a - c) :=
by 
  sorry

end common_root_conds_l5_5844


namespace book_costs_and_scenarios_l5_5637

theorem book_costs_and_scenarios :
  (∃ (x y : ℕ), x + 3 * y = 180 ∧ 3 * x + y = 140 ∧ 
    (x = 30) ∧ (y = 50)) ∧ 
  (∀ (m : ℕ), (30 * m + 75 * m) ≤ 700 → (∃ (m_values : Finset ℕ), 
    m_values = {2, 4, 6} ∧ (m ∈ m_values))) :=
  sorry

end book_costs_and_scenarios_l5_5637


namespace distance_between_foci_is_six_l5_5810

-- Lean 4 Statement
noncomputable def distance_between_foci (p1 p2 p3 : ℝ × ℝ) : ℝ :=
  if (p1 = (1, 3) ∧ p2 = (6, -1) ∧ p3 = (11, 3)) then 6 else 0

theorem distance_between_foci_is_six : distance_between_foci (1, 3) (6, -1) (11, 3) = 6 :=
by
  sorry

end distance_between_foci_is_six_l5_5810


namespace probability_multiple_of_3_or_4_l5_5716

theorem probability_multiple_of_3_or_4 : ((15 : ℚ) / 30) = (1 / 2) := by
  sorry

end probability_multiple_of_3_or_4_l5_5716


namespace length_to_width_ratio_l5_5780

/-- Let the perimeter of the rectangular sandbox be 30 feet,
    the width be 5 feet, and the length be some multiple of the width.
    Prove that the ratio of the length to the width is 2:1. -/
theorem length_to_width_ratio (P w : ℕ) (h1 : P = 30) (h2 : w = 5) (h3 : ∃ k, l = k * w) : 
  ∃ l, (P = 2 * (l + w)) ∧ (l / w = 2) := 
sorry

end length_to_width_ratio_l5_5780


namespace solve_for_m_l5_5504

theorem solve_for_m : ∃ m : ℝ, ((∀ x : ℝ, (x + 5) * (x + 2) = m + 3 * x) → (m = 6)) :=
by
  sorry

end solve_for_m_l5_5504


namespace lattice_intersections_l5_5889

theorem lattice_intersections (squares : ℕ) (circles : ℕ) 
        (line_segment : ℤ × ℤ → ℤ × ℤ) 
        (radius : ℚ) (side_length : ℚ) : 
        line_segment (0, 0) = (1009, 437) → 
        radius = 1/8 → side_length = 1/4 → 
        (squares + circles = 430) :=
by
  sorry

end lattice_intersections_l5_5889


namespace no_positive_integer_solutions_l5_5363

theorem no_positive_integer_solutions : ¬∃ (x y z : ℕ), x > 0 ∧ y > 0 ∧ z > 0 ∧ x ^ 4004 + y ^ 4004 = z ^ 2002 :=
by
  sorry

end no_positive_integer_solutions_l5_5363


namespace gcd_times_xyz_is_square_l5_5169

theorem gcd_times_xyz_is_square (x y z : ℕ) (h : 1 / (x : ℚ) - 1 / (y : ℚ) = 1 / (z : ℚ)) : 
  ∃ k : ℕ, (Nat.gcd x (Nat.gcd y z) * x * y * z) = k ^ 2 :=
sorry

end gcd_times_xyz_is_square_l5_5169


namespace compute_expression_l5_5989

theorem compute_expression : 1004^2 - 996^2 - 1000^2 + 1000^2 = 16000 := 
by sorry

end compute_expression_l5_5989


namespace digit_matching_equalities_l5_5905

theorem digit_matching_equalities :
  ∀ (a b : ℕ), 0 ≤ a ∧ a ≤ 99 → 0 ≤ b ∧ b ≤ 99 →
    ((a = 98 ∧ b = 1 ∧ (98 + 1)^2 = 100*98 + 1) ∨
     (a = 20 ∧ b = 25 ∧ (20 + 25)^2 = 100*20 + 25)) :=
by
  intros a b ha hb
  sorry

end digit_matching_equalities_l5_5905


namespace arithmetic_mean_solution_l5_5150

theorem arithmetic_mean_solution (x : ℚ) :
  (x + 10 + 20 + 3*x + 18 + 3*x + 6) / 5 = 30 → x = 96 / 7 :=
by
  intros h
  sorry

end arithmetic_mean_solution_l5_5150


namespace correct_operation_l5_5819

theorem correct_operation (a : ℝ) : (-a^3)^4 = a^12 :=
by sorry

end correct_operation_l5_5819


namespace min_baseball_cards_divisible_by_15_l5_5835

theorem min_baseball_cards_divisible_by_15 :
  ∀ (j m c e t : ℕ),
    j = m →
    m = c - 6 →
    c = 20 →
    e = 2 * (j + m) →
    t = c + m + j + e →
    t ≥ 104 →
    ∃ k : ℕ, t = 15 * k ∧ t = 105 :=
by
  intros j m c e t h1 h2 h3 h4 h5 h6
  sorry

end min_baseball_cards_divisible_by_15_l5_5835


namespace part1_part2_l5_5625

noncomputable def f (x a : ℝ) : ℝ := x^2 + a * x + 6

-- Part (I)
theorem part1 (a : ℝ) (h : a = 5) : ∀ x : ℝ, f x 5 < 0 ↔ -3 < x ∧ x < -2 := by
  sorry

-- Part (II)
theorem part2 (a : ℝ) : (∀ x : ℝ, f x a > 0) ↔ -2 * Real.sqrt 6 < a ∧ a < 2 * Real.sqrt 6 := by
  sorry

end part1_part2_l5_5625


namespace percentage_sum_l5_5951

noncomputable def womenWithRedHairBelow30 : ℝ := 0.07
noncomputable def menWithDarkHair30OrOlder : ℝ := 0.13

theorem percentage_sum :
  womenWithRedHairBelow30 + menWithDarkHair30OrOlder = 0.20 := by
  sorry -- Proof is omitted

end percentage_sum_l5_5951


namespace sum_of_roots_l5_5406

noncomputable def f (x : ℝ) : ℝ := Real.sin (x + Real.pi / 4)

theorem sum_of_roots (m : ℝ) (x1 x2 : ℝ) (h1 : 0 ≤ x1) (h2 : x1 < 2 * Real.pi)
  (h3 : 0 ≤ x2) (h4 : x2 < 2 * Real.pi) (h_distinct : x1 ≠ x2)
  (h_eq1 : f x1 = m) (h_eq2 : f x2 = m) : x1 + x2 = Real.pi / 2 ∨ x1 + x2 = 5 * Real.pi / 2 :=
by
  sorry

end sum_of_roots_l5_5406


namespace largest_divisor_360_450_l5_5897

theorem largest_divisor_360_450 : ∃ d, (d ∣ 360 ∧ d ∣ 450) ∧ (∀ e, (e ∣ 360 ∧ e ∣ 450) → e ≤ d) ∧ d = 90 :=
by
  sorry

end largest_divisor_360_450_l5_5897


namespace find_h_l5_5075

theorem find_h {a b c n k : ℝ} (x : ℝ) (h_val : ℝ) 
  (h_quad : a * x^2 + b * x + c = 3 * (x - 5)^2 + 15) :
  (4 * a) * x^2 + (4 * b) * x + (4 * c) = n * (x - h_val)^2 + k → h_val = 5 :=
sorry

end find_h_l5_5075


namespace pyramid_dihedral_angle_l5_5248

theorem pyramid_dihedral_angle 
  (k : ℝ) 
  (h_k_pos : 0 < k) :
  ∃ α : ℝ, α = 2 * Real.arccos (1 / Real.sqrt (Real.sqrt (4 * k))) :=
sorry

end pyramid_dihedral_angle_l5_5248


namespace number_of_A_items_number_of_A_proof_l5_5436

def total_items : ℕ := 600
def ratio_A_B_C := (1, 2, 3)
def selected_items : ℕ := 120

theorem number_of_A_items (total_items : ℕ) (selected_items : ℕ) (rA rB rC : ℕ) (ratio_proof : rA + rB + rC = 6) : ℕ :=
  let total_ratio := rA + rB + rC
  let A_ratio := rA
  (selected_items * A_ratio) / total_ratio

theorem number_of_A_proof : number_of_A_items total_items selected_items 1 2 3 (rfl) = 20 := by
  sorry

end number_of_A_items_number_of_A_proof_l5_5436


namespace ravenswood_forest_percentage_l5_5114

def ravenswood_gnomes (westerville_gnomes : ℕ) : ℕ := 4 * westerville_gnomes
def remaining_gnomes (total_gnomes taken_percentage: ℕ) : ℕ := (total_gnomes * (100 - taken_percentage)) / 100

theorem ravenswood_forest_percentage:
  ∀ (westerville_gnomes : ℕ) (remaining : ℕ) (total_gnomes : ℕ),
  westerville_gnomes = 20 →
  total_gnomes = ravenswood_gnomes westerville_gnomes →
  remaining = 48 →
  remaining_gnomes total_gnomes 40 = remaining :=
by
  sorry

end ravenswood_forest_percentage_l5_5114


namespace polygon_sides_16_l5_5342

def sum_of_interior_angles (n : ℕ) : ℝ := 180 * (n - 2)

noncomputable def arithmetic_sequence_sum (a1 an : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  n * (a1 + an) / 2

theorem polygon_sides_16 (n : ℕ) (a1 an : ℝ) (d : ℝ) 
  (h1 : d = 5) (h2 : an = 160) (h3 : a1 = 160 - 5 * (n - 1))
  (h4 : arithmetic_sequence_sum a1 an d n = sum_of_interior_angles n)
  : n = 16 :=
sorry

end polygon_sides_16_l5_5342


namespace reflected_ray_equation_l5_5508

-- Definitions for the given conditions
def incident_line (x : ℝ) : ℝ := 2 * x + 1
def reflection_line (x : ℝ) : ℝ := x

-- Problem statement: proving equation of the reflected ray
theorem reflected_ray_equation : 
  ∀ x y : ℝ, incident_line x = y ∧ reflection_line x = y → x - 2*y - 1 = 0 :=
by
  sorry

end reflected_ray_equation_l5_5508


namespace find_pqr_l5_5770

variable (p q r : ℚ)

theorem find_pqr (h1 : ∃ a : ℚ, ∀ x : ℚ, (p = a) ∧ (q = -2 * a * 3) ∧ (r = a * 3 * 3 + 7) ∧ (r = 10 + 7)) :
  p + q + r = 8 + 1/3 := by
  sorry

end find_pqr_l5_5770


namespace find_y_l5_5445

theorem find_y (x y : ℝ) (h1 : 9823 + x = 13200) (h2 : x = y / 3 + 37.5) : y = 10018.5 :=
by
  sorry

end find_y_l5_5445


namespace area_of_red_region_on_larger_sphere_l5_5766

/-- 
A smooth ball with a radius of 1 cm was dipped in red paint and placed between two 
absolutely smooth concentric spheres with radii of 4 cm and 6 cm, respectively
(the ball is outside the smaller sphere but inside the larger sphere).
As the ball moves and touches both spheres, it leaves a red mark. 
After traveling a closed path, a region outlined in red with an area of 37 square centimeters is formed on the smaller sphere. 
Find the area of the region outlined in red on the larger sphere. 
The answer should be 55.5 square centimeters.
-/
theorem area_of_red_region_on_larger_sphere
  (r1 r2 r3 : ℝ)
  (A_small : ℝ)
  (h_red_small_sphere : 37 = 2 * π * r2 * (A_small / (2 * π * r2)))
  (h_red_large_sphere : 55.5 = 2 * π * r3 * (A_small / (2 * π * r2))) :
  ∃ A_large : ℝ, A_large = 55.5 :=
by
  -- Definitions and conditions
  let r1 := 1  -- radius of small ball (1 cm)
  let r2 := 4  -- radius of smaller sphere (4 cm)
  let r3 := 6  -- radius of larger sphere (6 cm)

  -- Given: A small red area is 37 cm^2 on the smaller sphere.
  let A_small := 37

  -- Proof of the relationship of the spherical caps
  sorry

end area_of_red_region_on_larger_sphere_l5_5766


namespace find_k_l5_5888

-- Definitions based on the conditions
def number := 24
def bigPart := 13
  
theorem find_k (x y k : ℕ) 
  (original_number : x + y = 24)
  (big_part : x = 13 ∨ y = 13)
  (equation : k * x + 5 * y = 146) : k = 7 := 
  sorry

end find_k_l5_5888


namespace total_rainfall_2010_to_2012_l5_5336

noncomputable def average_rainfall (year : ℕ) : ℕ :=
  if year = 2010 then 35
  else if year = 2011 then 38
  else if year = 2012 then 41
  else 0

theorem total_rainfall_2010_to_2012 :
  (12 * average_rainfall 2010) + 
  (12 * average_rainfall 2011) + 
  (12 * average_rainfall 2012) = 1368 :=
by
  sorry

end total_rainfall_2010_to_2012_l5_5336


namespace correct_distribution_l5_5432

-- Define the conditions
def num_students : ℕ := 40
def ratio_A_to_B : ℚ := 0.8
def ratio_C_to_B : ℚ := 1.2

-- Definitions for the number of students earning each grade
def num_B (x : ℕ) : ℕ := x
def num_A (x : ℕ) : ℕ := Nat.floor (ratio_A_to_B * x)
def num_C (x : ℕ) : ℕ := Nat.ceil (ratio_C_to_B * x)

-- Prove the distribution is correct
theorem correct_distribution :
  ∃ x : ℕ, num_A x + num_B x + num_C x = num_students ∧ 
           num_A x = 10 ∧ num_B x = 14 ∧ num_C x = 16 :=
by
  sorry

end correct_distribution_l5_5432


namespace ice_cream_maker_completion_time_l5_5642

def start_time := 9
def time_to_half := 3
def end_time := start_time + 2 * time_to_half

theorem ice_cream_maker_completion_time :
  end_time = 15 :=
by
  -- Definitions: 9:00 AM -> 9, 12:00 PM -> 12, 3:00 PM -> 15
  -- Calculation: end_time = 9 + 2 * 3 = 15
  sorry

end ice_cream_maker_completion_time_l5_5642


namespace star_three_five_l5_5044

def star (x y : ℕ) := x^2 + 2 * x * y + y^2

theorem star_three_five : star 3 5 = 64 :=
by
  sorry

end star_three_five_l5_5044


namespace modified_goldbach_2024_l5_5947

def is_prime (p : ℕ) : Prop := ∀ n : ℕ, n > 1 → n < p → ¬ (p % n = 0)

theorem modified_goldbach_2024 :
  ∃ (p1 p2 : ℕ), p1 ≠ p2 ∧ is_prime p1 ∧ is_prime p2 ∧ p1 + p2 = 2024 := 
sorry

end modified_goldbach_2024_l5_5947


namespace square_perimeter_l5_5291

theorem square_perimeter (s : ℝ)
  (h1 : ∃ (s : ℝ), 4 * s = s * 1 + s / 4 * 1 + s * 1 + s / 4 * 1)
  (h2 : ∃ (P : ℝ), P = 4 * s)
  : (5/2) * s = 40 → 4 * s = 64 :=
by
  intro h
  sorry

end square_perimeter_l5_5291


namespace averages_correct_l5_5034

variables (marksEnglish totalEnglish marksMath totalMath marksPhysics totalPhysics 
           marksChemistry totalChemistry marksBiology totalBiology 
           marksHistory totalHistory marksGeography totalGeography : ℕ)

variables (avgEnglish avgMath avgPhysics avgChemistry avgBiology avgHistory avgGeography : ℚ)

def Kamal_average_english : Prop :=
  marksEnglish = 76 ∧ totalEnglish = 120 ∧ avgEnglish = (marksEnglish / totalEnglish) * 100

def Kamal_average_math : Prop :=
  marksMath = 65 ∧ totalMath = 150 ∧ avgMath = (marksMath / totalMath) * 100

def Kamal_average_physics : Prop :=
  marksPhysics = 82 ∧ totalPhysics = 100 ∧ avgPhysics = (marksPhysics / totalPhysics) * 100

def Kamal_average_chemistry : Prop :=
  marksChemistry = 67 ∧ totalChemistry = 80 ∧ avgChemistry = (marksChemistry / totalChemistry) * 100

def Kamal_average_biology : Prop :=
  marksBiology = 85 ∧ totalBiology = 100 ∧ avgBiology = (marksBiology / totalBiology) * 100

def Kamal_average_history : Prop :=
  marksHistory = 92 ∧ totalHistory = 150 ∧ avgHistory = (marksHistory / totalHistory) * 100

def Kamal_average_geography : Prop :=
  marksGeography = 58 ∧ totalGeography = 75 ∧ avgGeography = (marksGeography / totalGeography) * 100

theorem averages_correct :
  ∀ (marksEnglish totalEnglish marksMath totalMath marksPhysics totalPhysics 
      marksChemistry totalChemistry marksBiology totalBiology 
      marksHistory totalHistory marksGeography totalGeography : ℕ),
  ∀ (avgEnglish avgMath avgPhysics avgChemistry avgBiology avgHistory avgGeography : ℚ),
  Kamal_average_english marksEnglish totalEnglish avgEnglish →
  Kamal_average_math marksMath totalMath avgMath →
  Kamal_average_physics marksPhysics totalPhysics avgPhysics →
  Kamal_average_chemistry marksChemistry totalChemistry avgChemistry →
  Kamal_average_biology marksBiology totalBiology avgBiology →
  Kamal_average_history marksHistory totalHistory avgHistory →
  Kamal_average_geography marksGeography totalGeography avgGeography →
  avgEnglish = 63.33 ∧ avgMath = 43.33 ∧ avgPhysics = 82 ∧
  avgChemistry = 83.75 ∧ avgBiology = 85 ∧ avgHistory = 61.33 ∧ avgGeography = 77.33 :=
by
  sorry

end averages_correct_l5_5034


namespace Megan_popsicles_l5_5270

def minutes_in_hour : ℕ := 60

def total_minutes (hours : ℕ) (minutes : ℕ) : ℕ :=
  hours * minutes_in_hour + minutes

def popsicle_time : ℕ := 18

def popsicles_consumed (total_minutes : ℕ) (popsicle_time : ℕ) : ℕ :=
  total_minutes / popsicle_time

theorem Megan_popsicles (hours : ℕ) (minutes : ℕ) (popsicle_time : ℕ)
  (total_minutes : ℕ) (h_hours : hours = 5) (h_minutes : minutes = 36) (h_popsicle_time : popsicle_time = 18)
  (h_total_minutes : total_minutes = (5 * 60 + 36)) :
  popsicles_consumed 336 popsicle_time = 18 :=
by 
  sorry

end Megan_popsicles_l5_5270


namespace expected_adjacent_black_pairs_proof_l5_5217

-- Define the modified deck conditions.
def modified_deck (n : ℕ) := n = 60
def black_cards (b : ℕ) := b = 30
def red_cards (r : ℕ) := r = 30

-- Define the expected value of pairs of adjacent black cards.
def expected_adjacent_black_pairs (n b : ℕ) : ℚ :=
  b * (b - 1) / (n - 1)

theorem expected_adjacent_black_pairs_proof :
  modified_deck 60 →
  black_cards 30 →
  red_cards 30 →
  expected_adjacent_black_pairs 60 30 = 870 / 59 :=
by intros; sorry

end expected_adjacent_black_pairs_proof_l5_5217


namespace correct_tourism_model_l5_5715

noncomputable def tourism_model (x : ℕ) : ℝ :=
  80 * (Real.cos ((Real.pi / 6) * x + (2 * Real.pi / 3))) + 120

theorem correct_tourism_model :
  (∀ n : ℕ, tourism_model (n + 12) = tourism_model n) ∧
  (tourism_model 8 - tourism_model 2 = 160) ∧
  (tourism_model 2 = 40) :=
by
  sorry

end correct_tourism_model_l5_5715


namespace minimum_n_minus_m_l5_5850

noncomputable def f (x : Real) : Real :=
    (Real.sin x) * (Real.sin (x + Real.pi / 3)) - 1 / 4

theorem minimum_n_minus_m (m n : Real) (h : m < n) 
  (h_domain : ∀ x, m ≤ x ∧ x ≤ n → -1 / 2 ≤ f x ∧ f x ≤ 1 / 4) :
  n - m = 2 * Real.pi / 3 :=
by
  sorry

end minimum_n_minus_m_l5_5850


namespace polar_coordinates_equivalence_l5_5593

theorem polar_coordinates_equivalence :
  ∀ (ρ θ1 θ2 : ℝ), θ1 = π / 3 ∧ θ2 = -5 * π / 3 →
  (ρ = 5) → 
  (ρ * Real.cos θ1 = ρ * Real.cos θ2 ∧ ρ * Real.sin θ1 = ρ * Real.sin θ2) :=
by
  sorry

end polar_coordinates_equivalence_l5_5593


namespace maximize_perimeter_OIH_l5_5005

/-- In triangle ABC, given certain angles and side lengths, prove that
    angle ABC = 70° maximizes the perimeter of triangle OIH, where O, I,
    and H are the circumcenter, incenter, and orthocenter of triangle ABC. -/
theorem maximize_perimeter_OIH 
  (A : ℝ) (B : ℝ) (C : ℝ)
  (BC : ℝ) (AB : ℝ) (AC : ℝ)
  (BOC : ℝ) (BIC : ℝ) (BHC : ℝ) :
  A = 75 ∧ BC = 2 ∧ AB ≥ AC ∧
  BOC = 150 ∧ BIC = 127.5 ∧ BHC = 105 → 
  B = 70 :=
by
  sorry

end maximize_perimeter_OIH_l5_5005


namespace Sandy_total_marks_l5_5972

theorem Sandy_total_marks
  (correct_marks_per_sum : ℤ)
  (incorrect_marks_per_sum : ℤ)
  (total_sums : ℕ)
  (correct_sums : ℕ)
  (incorrect_sums : ℕ)
  (total_marks : ℤ) :
  correct_marks_per_sum = 3 →
  incorrect_marks_per_sum = -2 →
  total_sums = 30 →
  correct_sums = 24 →
  incorrect_sums = total_sums - correct_sums →
  total_marks = correct_marks_per_sum * correct_sums + incorrect_marks_per_sum * incorrect_sums →
  total_marks = 60 :=
by
  sorry

end Sandy_total_marks_l5_5972


namespace min_max_value_of_expr_l5_5030

theorem min_max_value_of_expr (p q r s : ℝ)
  (h1 : p + q + r + s = 10)
  (h2 : p^2 + q^2 + r^2 + s^2 = 20) :
  ∃ m M : ℝ, m = 2 ∧ M = 0 ∧ ∀ x, (x = 3 * (p^3 + q^3 + r^3 + s^3) - 2 * (p^4 + q^4 + r^4 + s^4)) → m ≤ x ∧ x ≤ M :=
sorry

end min_max_value_of_expr_l5_5030


namespace countNegativeValues_l5_5010

-- Define the condition that sqrt(x + 122) is a positive integer
noncomputable def isPositiveInteger (n : ℤ) (x : ℤ) : Prop :=
  ∃ n : ℤ, (n > 0) ∧ (x + 122 = n * n)

-- Define the condition that x is negative
def isNegative (x : ℤ) : Prop :=
  x < 0

-- Prove the number of different negative values of x such that sqrt(x + 122) is a positive integer is 11
theorem countNegativeValues :
  ∃ x_set : Finset ℤ, (∀ x ∈ x_set, isNegative x ∧ isPositiveInteger x (x + 122)) ∧ x_set.card = 11 :=
sorry

end countNegativeValues_l5_5010


namespace correct_inequality_l5_5847

variables {a b c : ℝ}
def quadratic_function (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem correct_inequality (h_a_pos : a > 0) (h_discriminant_pos : b^2 - 4 * a * c > 0) (h_c_neg : c < 0) (h_b_neg : b < 0) :
  a * b * c > 0 :=
sorry

end correct_inequality_l5_5847


namespace miniature_model_to_actual_statue_scale_l5_5390

theorem miniature_model_to_actual_statue_scale (height_actual : ℝ) (height_model : ℝ) : 
  height_actual = 90 → height_model = 6 → 
  (height_actual / height_model = 15) := 
by
  intros h_actual h_model
  rw [h_actual, h_model]
  sorry

end miniature_model_to_actual_statue_scale_l5_5390


namespace solution_set_l5_5199

noncomputable def system_of_equations (x y z : ℝ) : Prop :=
  6 * (x^2 * y^2 + y^2 * z^2 + z^2 * x^2) - 49 * x * y * z = 0 ∧
  6 * y * (x^2 - z^2) + 5 * x * z = 0 ∧
  2 * z * (x^2 - y^2) - 9 * x * y = 0

theorem solution_set :
  ∀ x y z : ℝ, system_of_equations x y z ↔ (x = 0 ∧ y = 0 ∧ z = 0) ∨
  (x = 2 ∧ y = 1 ∧ z = 3) ∨ (x = 2 ∧ y = -1 ∧ z = -3) ∨ 
  (x = -2 ∧ y = 1 ∧ z = -3) ∨ (x = -2 ∧ y = -1 ∧ z = 3) :=
by
  sorry

end solution_set_l5_5199


namespace aladdin_can_find_heavy_coins_l5_5447

theorem aladdin_can_find_heavy_coins :
  ∃ (x y : ℕ), 1 ≤ x ∧ x ≤ 20 ∧ 1 ≤ y ∧ y ≤ 20 ∧ x ≠ y ∧ (x + y ≥ 28) :=
by
  sorry

end aladdin_can_find_heavy_coins_l5_5447


namespace part1_find_a_b_part2_inequality_l5_5936

theorem part1_find_a_b (f : ℝ → ℝ) (a b : ℝ) (h_f : ∀ x, f x = |2 * x + 1| + |x + a|) 
  (h_sol : ∀ x, f x ≤ 3 ↔ b ≤ x ∧ x ≤ 1) : 
  a = -1 ∧ b = -1 :=
sorry

theorem part2_inequality (m n : ℝ) (a : ℝ) (h_m : 0 < m) (h_n : 0 < n) 
  (h_eq : (1 / (2 * m)) + (2 / n) + 2 * a = 0) (h_a : a = -1) : 
  4 * m^2 + n^2 ≥ 4 :=
sorry

end part1_find_a_b_part2_inequality_l5_5936


namespace length_of_bridge_l5_5696

theorem length_of_bridge (train_length : ℕ) (train_speed_kmph : ℕ) (cross_time_sec : ℕ) (bridge_length: ℕ):
  train_length = 110 →
  train_speed_kmph = 45 →
  cross_time_sec = 30 →
  bridge_length = 265 :=
by
  intros h1 h2 h3
  sorry

end length_of_bridge_l5_5696


namespace ratio_a_to_c_l5_5772

theorem ratio_a_to_c (a b c d : ℚ)
  (h1 : a / b = 5 / 2)
  (h2 : c / d = 4 / 1)
  (h3 : d / b = 1 / 3) :
  a / c = 15 / 8 :=
by {
  sorry
}

end ratio_a_to_c_l5_5772


namespace solve_linear_system_l5_5969

theorem solve_linear_system :
  ∃ (x y : ℝ), (x + 3 * y = -1) ∧ (2 * x + y = 3) ∧ (x = 2) ∧ (y = -1) :=
  sorry

end solve_linear_system_l5_5969


namespace decompose_x_l5_5939

def x : ℝ × ℝ × ℝ := (6, 12, -1)
def p : ℝ × ℝ × ℝ := (1, 3, 0)
def q : ℝ × ℝ × ℝ := (2, -1, 1)
def r : ℝ × ℝ × ℝ := (0, -1, 2)

theorem decompose_x :
  x = (4 : ℝ) • p + q - r :=
sorry

end decompose_x_l5_5939


namespace solve_quadratic_complete_square_l5_5775

theorem solve_quadratic_complete_square :
  ∃ b c : ℤ, (∀ x : ℝ, (x + b)^2 = c ↔ x^2 + 6 * x - 9 = 0) ∧ b + c = 21 := by
  sorry

end solve_quadratic_complete_square_l5_5775


namespace find_constants_l5_5528

theorem find_constants : 
  ∃ (a b : ℝ), a • (⟨1, 4⟩ : ℝ × ℝ) + b • (⟨3, -2⟩ : ℝ × ℝ) = (⟨5, 6⟩ : ℝ × ℝ) ∧ a = 2 ∧ b = 1 :=
by 
  sorry

end find_constants_l5_5528


namespace range_of_m_l5_5576

theorem range_of_m (α β m : ℝ)
  (h1 : 0 < α ∧ α < 1)
  (h2 : 1 < β ∧ β < 2)
  (h3 : ∀ x, x^2 - m * x + 1 = 0 ↔ (x = α ∨ x = β)) :
  2 < m ∧ m < 5 / 2 :=
sorry

end range_of_m_l5_5576


namespace inequality_proof_l5_5451

theorem inequality_proof (a b : ℝ) (h : a + b ≠ 0) :
  (a + b) / (a^2 - a * b + b^2) ≤ 4 / |a + b| ∧
  ((a + b) / (a^2 - a * b + b^2) = 4 / |a + b| ↔ a = b) :=
by
  sorry

end inequality_proof_l5_5451


namespace mean_of_jane_scores_l5_5225

theorem mean_of_jane_scores :
  let scores := [96, 95, 90, 87, 91, 75]
  let n := 6
  let sum_scores := 96 + 95 + 90 + 87 + 91 + 75
  let mean := sum_scores / n
  mean = 89 := by
    sorry

end mean_of_jane_scores_l5_5225


namespace anne_total_bottle_caps_l5_5124

/-- 
Anne initially has 10 bottle caps 
and then finds another 5 bottle caps.
-/
def anne_initial_bottle_caps : ℕ := 10
def anne_found_bottle_caps : ℕ := 5

/-- 
Prove that the total number of bottle caps
Anne ends with is equal to 15.
-/
theorem anne_total_bottle_caps : 
  anne_initial_bottle_caps + anne_found_bottle_caps = 15 :=
by 
  sorry

end anne_total_bottle_caps_l5_5124


namespace inscribed_circle_radius_l5_5425

theorem inscribed_circle_radius (R r x : ℝ) (hR : R = 18) (hr : r = 9) :
  x = 8 :=
sorry

end inscribed_circle_radius_l5_5425


namespace arithmetic_sequence_sum_l5_5663

theorem arithmetic_sequence_sum (a b : ℤ) (h1 : 10 - 3 = 7)
  (h2 : a = 10 + 7) (h3 : b = 24 + 7) : a + b = 48 :=
by
  sorry

end arithmetic_sequence_sum_l5_5663


namespace largest_divisor_of_product_of_five_consecutive_integers_l5_5094

theorem largest_divisor_of_product_of_five_consecutive_integers :
  ∃ n, (∀ k : ℤ, n ∣ (k * (k + 1) * (k + 2) * (k + 3) * (k + 4))) ∧ n = 60 :=
by
  sorry

end largest_divisor_of_product_of_five_consecutive_integers_l5_5094


namespace max_profit_price_range_for_minimum_profit_l5_5806

noncomputable def functional_relationship (x : ℝ) : ℝ :=
-10 * x^2 + 2000 * x - 84000

theorem max_profit :
  ∃ x, (∀ x₀, x₀ ≠ x → functional_relationship x₀ < functional_relationship x) ∧
  functional_relationship x = 16000 := 
sorry

theorem price_range_for_minimum_profit :
  ∀ (x : ℝ), 
  -10 * (x - 100)^2 + 16000 - 1750 ≥ 12000 → 
  85 ≤ x ∧ x ≤ 115 :=
sorry

end max_profit_price_range_for_minimum_profit_l5_5806


namespace points_on_line_l5_5090

-- Define the points
def P1 : (ℝ × ℝ) := (8, 16)
def P2 : (ℝ × ℝ) := (2, 4)

-- Define the line equation as a predicate
def on_line (m b : ℝ) (p : ℝ × ℝ) : Prop := p.2 = m * p.1 + b

-- Define the given points to be checked
def P3 : (ℝ × ℝ) := (5, 10)
def P4 : (ℝ × ℝ) := (7, 14)
def P5 : (ℝ × ℝ) := (4, 7)
def P6 : (ℝ × ℝ) := (10, 20)
def P7 : (ℝ × ℝ) := (3, 6)

theorem points_on_line :
  let m := 2
  let b := 0
  on_line m b P3 ∧
  on_line m b P4 ∧
  ¬ on_line m b P5 ∧
  on_line m b P6 ∧
  on_line m b P7 :=
by
  sorry

end points_on_line_l5_5090


namespace multiply_469111111_by_99999999_l5_5051

theorem multiply_469111111_by_99999999 :
  469111111 * 99999999 = 46911111053088889 :=
sorry

end multiply_469111111_by_99999999_l5_5051


namespace simplify_cube_root_l5_5706

theorem simplify_cube_root (a : ℝ) (h : 0 ≤ a) : (a * a^(1/2))^(1/3) = a^(1/2) :=
sorry

end simplify_cube_root_l5_5706


namespace annual_income_is_32000_l5_5601

noncomputable def compute_tax (p A: ℝ) : ℝ := 
  0.01 * p * 28000 + 0.01 * (p + 2) * (A - 28000)

noncomputable def stated_tax (p A: ℝ) : ℝ := 
  0.01 * (p + 0.25) * A

theorem annual_income_is_32000 (p : ℝ) (A : ℝ) :
  compute_tax p A = stated_tax p A → A = 32000 :=
by
  intros h
  have : 0.01 * p * 28000 + 0.01 * (p + 2) * (A - 28000) = 0.01 * (p + 0.25) * A := h
  sorry

end annual_income_is_32000_l5_5601


namespace vann_teeth_cleaning_l5_5211

def numDogsCleaned (D : Nat) : Prop :=
  let dogTeethCount := 42
  let catTeethCount := 30
  let pigTeethCount := 28
  let numCats := 10
  let numPigs := 7
  let totalTeeth := 706
  dogTeethCount * D + catTeethCount * numCats + pigTeethCount * numPigs = totalTeeth

theorem vann_teeth_cleaning : numDogsCleaned 5 :=
by
  sorry

end vann_teeth_cleaning_l5_5211


namespace transform_quadratic_l5_5779

theorem transform_quadratic (x m n : ℝ) 
  (h : x^2 - 6 * x - 1 = 0) : 
  (x + m)^2 = n ↔ (m = 3 ∧ n = 10) :=
by sorry

end transform_quadratic_l5_5779


namespace solve_equation_5x_plus1_div_2x_sq_plus_5x_minus3_eq_2x_div_2x_minus1_l5_5865

theorem solve_equation_5x_plus1_div_2x_sq_plus_5x_minus3_eq_2x_div_2x_minus1 :
  ∀ x : ℝ, 2 * x ^ 2 + 5 * x - 3 ≠ 0 ∧ 2 * x - 1 ≠ 0 → 
  (5 * x + 1) / (2 * x ^ 2 + 5 * x - 3) = (2 * x) / (2 * x - 1) → 
  x = -1 :=
by
  intro x h_cond h_eq
  sorry

end solve_equation_5x_plus1_div_2x_sq_plus_5x_minus3_eq_2x_div_2x_minus1_l5_5865


namespace negation_exists_l5_5755

-- Definitions used in the conditions
def prop1 (x : ℝ) : Prop := x^2 ≥ 1
def neg_prop1 : Prop := ∃ x : ℝ, x^2 < 1

-- Statement to be proved
theorem negation_exists (h : ∀ x : ℝ, prop1 x) : neg_prop1 :=
by
  sorry

end negation_exists_l5_5755


namespace problem_solve_l5_5992

theorem problem_solve (x y : ℝ) (h1 : x ≠ y) (h2 : x / y + (x + 6 * y) / (y + 6 * x) = 3) : 
    x / y = (8 + Real.sqrt 46) / 6 := 
  sorry

end problem_solve_l5_5992


namespace pyramid_edges_sum_l5_5705

noncomputable def sum_of_pyramid_edges (s : ℝ) (h : ℝ) : ℝ :=
  let diagonal := s * Real.sqrt 2
  let half_diagonal := diagonal / 2
  let slant_height := Real.sqrt (half_diagonal^2 + h^2)
  4 * s + 4 * slant_height

theorem pyramid_edges_sum
  (s : ℝ) (h : ℝ)
  (hs : s = 15)
  (hh : h = 15) :
  sum_of_pyramid_edges s h = 135 :=
sorry

end pyramid_edges_sum_l5_5705


namespace hexagon_pillar_height_l5_5000

noncomputable def height_of_pillar_at_vertex_F (s : ℝ) (hA hB hC : ℝ) (A : ℝ × ℝ) : ℝ :=
  10

theorem hexagon_pillar_height :
  ∀ (s hA hB hC : ℝ) (A : ℝ × ℝ),
  s = 8 ∧ hA = 15 ∧ hB = 10 ∧ hC = 12 ∧ A = (3, 3 * Real.sqrt 3) →
  height_of_pillar_at_vertex_F s hA hB hC A = 10 := by
  sorry

end hexagon_pillar_height_l5_5000


namespace condition_iff_absolute_value_l5_5151

theorem condition_iff_absolute_value (a b : ℝ) : (a > b) ↔ (a * |a| > b * |b|) :=
sorry

end condition_iff_absolute_value_l5_5151


namespace round_robin_teams_l5_5877

theorem round_robin_teams (x : ℕ) (h : (x * (x - 1)) / 2 = 45) : x = 10 := 
by
  sorry

end round_robin_teams_l5_5877


namespace max_mn_on_parabola_l5_5468

theorem max_mn_on_parabola :
  ∀ m n : ℝ, (n = -m^2 + 3) → (m + n ≤ 13 / 4) :=
by
  sorry

end max_mn_on_parabola_l5_5468


namespace exists_non_decreasing_subsequences_l5_5327

theorem exists_non_decreasing_subsequences {a b c : ℕ → ℕ} : 
  ∃ p q : ℕ, a p ≥ a q ∧ b p ≥ b q ∧ c p ≥ c q :=
sorry

end exists_non_decreasing_subsequences_l5_5327


namespace smallest_n_for_divisibility_problem_l5_5221

theorem smallest_n_for_divisibility_problem :
  ∃ n : ℕ, n > 0 ∧ (∀ k : ℕ, 1 ≤ k ∧ k ≤ n → n * (n + 1) ≠ 0 ∧
  (∃ k : ℕ, 1 ≤ k ∧ k ≤ n ∧ ¬ (n * (n + 1)) % k = 0) ∧
  ∀ m : ℕ, m > 0 ∧ m < n → (∀ k : ℕ, 1 ≤ k ∧ k ≤ m → (m * (m + 1)) % k ≠ 0)) → n = 4 := sorry

end smallest_n_for_divisibility_problem_l5_5221


namespace find_b_l5_5230

theorem find_b (b : ℚ) : (-4 : ℚ) * (45 / 4) = -45 → (-4 + 45 / 4) = -b → b = -29 / 4 := by
  intros h1 h2
  sorry

end find_b_l5_5230


namespace expression_for_f_l5_5710

theorem expression_for_f (f : ℤ → ℤ) (h : ∀ x : ℤ, f (x + 1) = x^2 - x - 2) : ∀ x : ℤ, f x = x^2 - 3 * x := 
by
  sorry

end expression_for_f_l5_5710


namespace female_salmon_returned_l5_5334

/-- The number of female salmon that returned to their rivers is 259378,
    given that the total number of salmon that made the trip is 971639 and
    the number of male salmon that returned is 712261. -/
theorem female_salmon_returned :
  let n := 971639
  let m := 712261
  let f := n - m
  f = 259378 :=
by
  rfl

end female_salmon_returned_l5_5334


namespace inequality_ab_l5_5717

theorem inequality_ab (a b : ℝ) (h : a * b < 0) : |a + b| < |a - b| := 
sorry

end inequality_ab_l5_5717


namespace smallest_possible_value_l5_5341

-- Definitions of the digits
def P := 1
def A := 9
def B := 2
def H := 8
def O := 3

-- Expression for continued fraction T
noncomputable def T : ℚ :=
  P + 1 / (A + 1 / (B + 1 / (H + 1 / O)))

-- The goal is to prove that T is the smallest possible value given the conditions
theorem smallest_possible_value : T = 555 / 502 :=
by
  -- The detailed proof would be done here, but for now we use sorry because we only need the statement
  sorry

end smallest_possible_value_l5_5341


namespace value_of_expression_l5_5571

theorem value_of_expression (x y : ℤ) (h1 : x = 3) (h2 : y = 2) : 3 * x - 4 * y = 1 := by
  sorry

end value_of_expression_l5_5571


namespace max_elem_one_correct_max_elem_two_correct_min_range_x_correct_average_min_eq_x_correct_l5_5591

def max_elem_one (c : ℝ) : Prop :=
  max (-2) (max 3 c) = max 3 c

def max_elem_two (m n : ℝ) (h1 : m < 0) (h2 : n > 0) : Prop :=
  max (3 * m) (max ((n + 3) * m) (-m * n)) = - m * n

def min_range_x (x : ℝ) : Prop :=
  min 2 (min (2 * x + 2) (4 - 2 * x)) = 2 → 0 ≤ x ∧ x ≤ 1

def average_min_eq_x : Prop :=
  ∀ (x : ℝ), (2 + (x + 1) + 2 * x) / 3 = min 2 (min (x + 1) (2 * x)) → x = 1

-- Lean 4 statements
theorem max_elem_one_correct (c : ℝ) : max_elem_one c := 
  sorry

theorem max_elem_two_correct {m n : ℝ} (h1 : m < 0) (h2 : n > 0) : max_elem_two m n h1 h2 :=
  sorry

theorem min_range_x_correct (h : min 2 (min (2 * x + 2) (4 - 2 * x)) = 2) : min_range_x x :=
  sorry

theorem average_min_eq_x_correct : average_min_eq_x :=
  sorry

end max_elem_one_correct_max_elem_two_correct_min_range_x_correct_average_min_eq_x_correct_l5_5591


namespace math_problem_proof_l5_5422

-- Define the problem statement
def problem_expr : ℕ :=
  28 * 7 * 25 + 12 * 7 * 25 + 7 * 11 * 3 + 44

-- Prove the problem statement equals to the correct answer
theorem math_problem_proof : problem_expr = 7275 := by
  sorry

end math_problem_proof_l5_5422


namespace unique_solution_for_equation_l5_5693

theorem unique_solution_for_equation (a b c d : ℝ) 
  (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : 0 < c) (h₄ : 0 < d)
  (h : ∀ x : ℝ, (a * x + b) ^ 2016 + (x ^ 2 + c * x + d) ^ 1008 = 8 * (x - 2) ^ 2016) :
  a = 2 ^ (1 / 672) ∧ b = -2 * 2 ^ (1 / 672) ∧ c = -4 ∧ d = 4 :=
by
  sorry

end unique_solution_for_equation_l5_5693


namespace coeff_x2_in_x_minus_1_pow_4_l5_5517

theorem coeff_x2_in_x_minus_1_pow_4 :
  ∀ (x : ℝ), (∃ (p : ℕ), (x - 1) ^ 4 = p * x^2 + (other_terms) ∧ p = 6) :=
by sorry

end coeff_x2_in_x_minus_1_pow_4_l5_5517


namespace last_integer_in_geometric_sequence_l5_5671

theorem last_integer_in_geometric_sequence (a : ℕ) (r : ℚ) (h_a : a = 2048000) (h_r : r = 1/2) : 
  ∃ n : ℕ, (a : ℚ) * (r^n : ℚ) = 125 := 
by
  sorry

end last_integer_in_geometric_sequence_l5_5671


namespace gopi_servant_salary_l5_5393

theorem gopi_servant_salary (S : ℕ) (turban_price : ℕ) (cash_received : ℕ) (months_worked : ℕ) (total_months : ℕ) :
  turban_price = 70 →
  cash_received = 50 →
  months_worked = 9 →
  total_months = 12 →
  S = 160 :=
by
  sorry

end gopi_servant_salary_l5_5393


namespace simplify_and_evaluate_expression_l5_5633

-- Define a and b with given values
def a := 1 / 2
def b := 1 / 3

-- Define the expression
def expr := 5 * (3 * a ^ 2 * b - a * b ^ 2) - (a * b ^ 2 + 3 * a ^ 2 * b)

-- State the theorem
theorem simplify_and_evaluate_expression : expr = 2 / 3 := 
by
  -- Proof can be inserted here
  sorry

end simplify_and_evaluate_expression_l5_5633


namespace sequence_general_formula_l5_5815

theorem sequence_general_formula (n : ℕ) (h : n ≥ 1) :
  ∃ a : ℕ → ℝ, a 1 = 1 ∧ (∀ n ≥ 1, a (n + 1) = a n / (1 + a n)) ∧ a n = (1 : ℝ) / n :=
by
  sorry

end sequence_general_formula_l5_5815


namespace min_trips_correct_l5_5188

-- Define the masses of the individuals and the elevator capacity as constants
def masses : List ℕ := [150, 62, 63, 66, 70, 75, 79, 84, 95, 96, 99]
def elevator_capacity : ℕ := 190

-- Define a function that computes the minimum number of trips required to transport all individuals
noncomputable def min_trips (masses : List ℕ) (capacity : ℕ) : ℕ := sorry

-- State the theorem to be proven
theorem min_trips_correct :
  min_trips masses elevator_capacity = 6 := sorry

end min_trips_correct_l5_5188


namespace inscribed_circle_radius_isosceles_triangle_l5_5825

noncomputable def isosceles_triangle_base : ℝ := 30 -- base AC
noncomputable def isosceles_triangle_equal_side : ℝ := 39 -- equal sides AB and BC

theorem inscribed_circle_radius_isosceles_triangle :
  ∀ (AC AB BC: ℝ), 
  AC = isosceles_triangle_base → 
  AB = isosceles_triangle_equal_side →
  BC = isosceles_triangle_equal_side →
  ∃ r : ℝ, r = 10 := 
by
  intros AC AB BC hAC hAB hBC
  sorry

end inscribed_circle_radius_isosceles_triangle_l5_5825


namespace length_of_platform_l5_5535

theorem length_of_platform 
  (speed_train_kmph : ℝ)
  (time_cross_platform : ℝ)
  (time_cross_man : ℝ)
  (conversion_factor : ℝ)
  (speed_train_mps : ℝ)
  (length_train : ℝ)
  (total_distance : ℝ)
  (length_platform : ℝ) :
  speed_train_kmph = 150 →
  time_cross_platform = 45 →
  time_cross_man = 20 →
  conversion_factor = (1000 / 3600) →
  speed_train_mps = speed_train_kmph * conversion_factor →
  length_train = speed_train_mps * time_cross_man →
  total_distance = speed_train_mps * time_cross_platform →
  length_platform = total_distance - length_train →
  length_platform = 1041.75 :=
by sorry

end length_of_platform_l5_5535


namespace floor_sqrt_80_eq_8_l5_5144

theorem floor_sqrt_80_eq_8
  (h1 : 8^2 = 64)
  (h2 : 9^2 = 81)
  (h3 : 64 < 80 ∧ 80 < 81)
  (h4 : 8 < Real.sqrt 80 ∧ Real.sqrt 80 < 9) : 
  Int.floor (Real.sqrt 80) = 8 := by
  sorry

end floor_sqrt_80_eq_8_l5_5144


namespace miles_per_dollar_l5_5121

def car_mpg : ℝ := 32
def gas_cost_per_gallon : ℝ := 4

theorem miles_per_dollar (X : ℝ) : 
  (X / gas_cost_per_gallon) * car_mpg = 8 * X :=
by
  sorry

end miles_per_dollar_l5_5121


namespace GCF_seven_eight_factorial_l5_5874

-- Given conditions
def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

-- Calculating 7! and 8!
def seven_factorial := factorial 7
def eight_factorial := factorial 8

-- Proof statement
theorem GCF_seven_eight_factorial : ∃ g, g = seven_factorial ∧ g = Nat.gcd seven_factorial eight_factorial ∧ g = 5040 :=
by sorry

end GCF_seven_eight_factorial_l5_5874


namespace inequality_in_triangle_l5_5106

variables {a b c : ℝ}

namespace InequalityInTriangle

-- Define the condition that a, b, c are sides of a triangle
def is_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

theorem inequality_in_triangle (a b c : ℝ) (h : is_triangle a b c) :
  1 / (b + c - a) + 1 / (c + a - b) + 1 / (a + b - c) > 9 / (a + b + c) :=
sorry

end InequalityInTriangle

end inequality_in_triangle_l5_5106


namespace missing_number_is_6630_l5_5321

theorem missing_number_is_6630 (x : ℕ) (h : 815472 / x = 123) : x = 6630 :=
by {
  sorry
}

end missing_number_is_6630_l5_5321


namespace correct_calculation_l5_5669

variable (a : ℝ)

theorem correct_calculation : (2 * a ^ 3) ^ 3 = 8 * a ^ 9 :=
by sorry

end correct_calculation_l5_5669


namespace even_function_f_l5_5142

noncomputable def f (x : ℝ) : ℝ :=
if x ≤ 0 then x^3 - x^2 else -(-x)^3 - (-x)^2

theorem even_function_f (x : ℝ) (h : ∀ x ≤ 0, f x = x^3 - x^2) :
  (∀ x, f x = f (-x)) ∧ (∀ x > 0, f x = -x^3 - x^2) :=
by
  sorry

end even_function_f_l5_5142


namespace probability_two_queens_or_at_least_one_jack_l5_5901

-- Definitions
def num_jacks : ℕ := 4
def num_queens : ℕ := 4
def total_cards : ℕ := 52

-- Probability calculation for drawing either two Queens or at least one Jack
theorem probability_two_queens_or_at_least_one_jack :
  (4 / 52) * (3 / (52 - 1)) + ((4 / 52) * (48 / (52 - 1)) + (48 / 52) * (4 / (52 - 1)) + (4 / 52) * (3 / (52 - 1))) = 2 / 13 :=
by
  sorry

end probability_two_queens_or_at_least_one_jack_l5_5901


namespace intersection_point_of_lines_l5_5417

theorem intersection_point_of_lines (n : ℕ) (x y : ℤ) :
  15 * x + 18 * y = 1005 ∧ y = n * x + 2 → n = 2 :=
by
  sorry

end intersection_point_of_lines_l5_5417


namespace happy_children_count_l5_5259

-- Definitions of the conditions
def total_children : ℕ := 60
def sad_children : ℕ := 10
def neither_happy_nor_sad_children : ℕ := 20
def boys : ℕ := 22
def girls : ℕ := 38
def happy_boys : ℕ := 6
def sad_girls : ℕ := 4
def boys_neither_happy_nor_sad : ℕ := 10

-- The theorem we wish to prove
theorem happy_children_count :
  total_children - sad_children - neither_happy_nor_sad_children = 30 :=
by 
  -- Placeholder for the proof
  sorry

end happy_children_count_l5_5259


namespace identify_set_A_l5_5283

open Set

def A : Set ℕ := {x | 0 ≤ x ∧ x < 3}

theorem identify_set_A : A = {0, 1, 2} := 
by
  sorry

end identify_set_A_l5_5283


namespace area_of_rectangle_A_is_88_l5_5387

theorem area_of_rectangle_A_is_88 
  (lA lB lC w wC : ℝ)
  (h1 : lB = lA + 2)
  (h2 : lB * w = lA * w + 22)
  (h3 : wC = w - 4)
  (AreaB : ℝ := lB * w)
  (AreaC : ℝ := lB * wC)
  (h4 : AreaC = AreaB - 40) : 
  (lA * w = 88) :=
sorry

end area_of_rectangle_A_is_88_l5_5387


namespace snow_fall_time_l5_5189

theorem snow_fall_time :
  (∀ rate_per_six_minutes : ℕ, rate_per_six_minutes = 1 →
    (∀ minute : ℕ, minute = 6 →
      (∀ height_in_m : ℕ, height_in_m = 1 →
        ∃ time_in_hours : ℕ, time_in_hours = 100 ))) :=
sorry

end snow_fall_time_l5_5189


namespace complement_A_in_U_l5_5769

open Set

variable {𝕜 : Type*} [LinearOrderedField 𝕜]

def A (x : 𝕜) : Prop := |x - (1 : 𝕜)| > 2
def U : Set 𝕜 := univ

theorem complement_A_in_U : (U \ {x : 𝕜 | A x}) = {x : 𝕜 | -1 ≤ x ∧ x ≤ 3} := by
  sorry

end complement_A_in_U_l5_5769


namespace molecular_weight_C4H10_l5_5498

theorem molecular_weight_C4H10
  (atomic_weight_C : ℝ)
  (atomic_weight_H : ℝ)
  (C4H10_C_atoms : ℕ)
  (C4H10_H_atoms : ℕ)
  (moles : ℝ) : 
  atomic_weight_C = 12.01 →
  atomic_weight_H = 1.008 →
  C4H10_C_atoms = 4 →
  C4H10_H_atoms = 10 →
  moles = 6 →
  (C4H10_C_atoms * atomic_weight_C + C4H10_H_atoms * atomic_weight_H) * moles = 348.72 :=
by
  sorry

end molecular_weight_C4H10_l5_5498


namespace min_value_expr_l5_5480

theorem min_value_expr (x y : ℝ) (h1 : x^2 + y^2 = 2) (h2 : |x| ≠ |y|) : 
  (∃ m : ℝ, (∀ x y : ℝ, x^2 + y^2 = 2 ∧ |x| ≠ |y| → m ≤ (1 / (x + y)^2 + 1 / (x - y)^2)) ∧ m = 1) :=
by
  sorry

end min_value_expr_l5_5480


namespace length_more_than_breadth_l5_5820

theorem length_more_than_breadth (b x : ℕ) 
  (h1 : 60 = b + x) 
  (h2 : 4 * b + 2 * x = 200) : x = 20 :=
by {
  sorry
}

end length_more_than_breadth_l5_5820


namespace Noah_age_in_10_years_is_22_l5_5077

def Joe_age : Nat := 6
def Noah_age := 2 * Joe_age
def Noah_age_after_10_years := Noah_age + 10

theorem Noah_age_in_10_years_is_22 : Noah_age_after_10_years = 22 := by
  sorry

end Noah_age_in_10_years_is_22_l5_5077


namespace problem_statement_l5_5532

variable {f : ℝ → ℝ}

-- Condition 1: f(x) has domain ℝ (implicitly given by the type signature ωf)
-- Condition 2: f is decreasing on the interval (6, +∞)
def is_decreasing_on_6_infty (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, 6 < x → x < y → f x > f y

-- Condition 3: y = f(x + 6) is an even function
def is_even_shifted (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (x + 6) = f (-x - 6)

-- The statement to prove
theorem problem_statement (h_decrease : is_decreasing_on_6_infty f) (h_even_shift : is_even_shifted f) : f 5 > f 8 :=
sorry

end problem_statement_l5_5532


namespace triangle_b_value_triangle_area_value_l5_5967

noncomputable def triangle_b (a : ℝ) (cosA : ℝ) : ℝ :=
  let sinA := Real.sqrt (1 - cosA^2)
  let sinB := cosA
  (a * sinB) / sinA

noncomputable def triangle_area (a b c : ℝ) (sinC : ℝ) : ℝ :=
  0.5 * a * b * sinC

-- Given conditions
variable (A B : ℝ) (a : ℝ := 3) (cosA : ℝ := Real.sqrt 6 / 3) (B := A + Real.pi / 2)

-- The assertions to prove
theorem triangle_b_value :
  triangle_b a cosA = 3 * Real.sqrt 2 :=
sorry

theorem triangle_area_value :
  triangle_area 3 (3 * Real.sqrt 2) 1 (1 / 3) = (3 * Real.sqrt 2) / 2 :=
sorry

end triangle_b_value_triangle_area_value_l5_5967


namespace original_stickers_l5_5011

theorem original_stickers (x : ℕ) (h₁ : x * 3 / 4 * 4 / 5 = 45) : x = 75 :=
by
  sorry

end original_stickers_l5_5011


namespace find_A_for_diamondsuit_l5_5777

-- Define the operation
def diamondsuit (A B : ℝ) : ℝ := 4 * A - 3 * B + 7

-- Define the specific instance of the operation equated to 57
theorem find_A_for_diamondsuit :
  ∃ A : ℝ, diamondsuit A 10 = 57 ↔ A = 20 := by
  sorry

end find_A_for_diamondsuit_l5_5777


namespace range_of_a_l5_5242

variable {α : Type*} [LinearOrderedField α]

def setA (a : α) : Set α := {x | abs (x - a) < 1}
def setB : Set α := {x | 1 < x ∧ x < 5}

theorem range_of_a (a : α) (h : setA a ∩ setB = ∅) : a ≤ 0 ∨ a ≥ 6 :=
sorry

end range_of_a_l5_5242


namespace john_spent_fraction_l5_5267

theorem john_spent_fraction (initial_money snacks_left necessities_left snacks_fraction : ℝ)
  (h1 : initial_money = 20)
  (h2 : snacks_fraction = 1/5)
  (h3 : snacks_left = initial_money * snacks_fraction)
  (h4 : necessities_left = 4)
  (remaining_money : ℝ) (h5 : remaining_money = initial_money - snacks_left)
  (spent_on_necessities : ℝ) (h6 : spent_on_necessities = remaining_money - necessities_left) 
  (fraction_spent : ℝ) (h7 : fraction_spent = spent_on_necessities / remaining_money) : 
  fraction_spent = 3/4 := 
sorry

end john_spent_fraction_l5_5267


namespace balloon_count_l5_5243

theorem balloon_count (total_balloons red_balloons blue_balloons black_balloons : ℕ) 
  (h_total : total_balloons = 180)
  (h_red : red_balloons = 3 * blue_balloons)
  (h_black : black_balloons = 2 * blue_balloons) :
  red_balloons = 90 ∧ blue_balloons = 30 ∧ black_balloons = 60 :=
by
  sorry

end balloon_count_l5_5243


namespace algebraic_expression_value_l5_5205

variables (a b c d m : ℤ)

def opposite (a b : ℤ) : Prop := a + b = 0
def reciprocal (c d : ℤ) : Prop := c * d = 1
def abs_eq_2 (m : ℤ) : Prop := |m| = 2

theorem algebraic_expression_value {a b c d m : ℤ} 
  (h1 : opposite a b) 
  (h2 : reciprocal c d) 
  (h3 : abs_eq_2 m) :
  (2 * m - (a + b - 1) + 3 * c * d = 8 ∨ 2 * m - (a + b - 1) + 3 * c * d = 0) :=
by
  sorry

end algebraic_expression_value_l5_5205


namespace product_of_integers_l5_5060

theorem product_of_integers (x y : ℕ) (h1 : x + y = 26) (h2 : x^2 - y^2 = 52) : x * y = 168 := by
  sorry

end product_of_integers_l5_5060


namespace speed_of_water_l5_5297

-- Definitions based on conditions
def swim_speed_in_still_water : ℝ := 4
def distance_against_current : ℝ := 6
def time_against_current : ℝ := 3
def effective_speed (v : ℝ) : ℝ := swim_speed_in_still_water - v

-- Theorem to prove the speed of the water
theorem speed_of_water (v : ℝ) : 
  effective_speed v * time_against_current = distance_against_current → 
  v = 2 :=
by
  sorry

end speed_of_water_l5_5297


namespace combined_ticket_cost_l5_5324

variables (S K : ℕ)

theorem combined_ticket_cost (total_budget : ℕ) (samuel_food_drink : ℕ) (kevin_food : ℕ) (kevin_drink : ℕ) :
  total_budget = 20 →
  samuel_food_drink = 6 →
  kevin_food = 4 →
  kevin_drink = 2 →
  S + samuel_food_drink + K + kevin_food + kevin_drink = total_budget →
  S + K = 8 :=
by
  intros h_total_budget h_samuel_food_drink h_kevin_food h_kevin_drink h_total_spent
  /-
  We have the following conditions:
  1. total_budget = 20
  2. samuel_food_drink = 6
  3. kevin_food = 4
  4. kevin_drink = 2
  5. S + samuel_food_drink + K + kevin_food + kevin_drink = total_budget

  We need to prove that S + K = 8. We can use the conditions to derive this.
  -/
  rw [h_total_budget, h_samuel_food_drink, h_kevin_food, h_kevin_drink] at h_total_spent
  exact sorry

end combined_ticket_cost_l5_5324


namespace eval_expression_l5_5830

theorem eval_expression : 7^3 + 3 * 7^2 + 3 * 7 + 1 = 512 := 
by 
  sorry

end eval_expression_l5_5830


namespace total_cost_of_tickets_l5_5136

-- Conditions
def normal_price : ℝ := 50
def website_tickets_cost : ℝ := 2 * normal_price
def scalper_tickets_cost : ℝ := 2 * (2.4 * normal_price) - 10
def discounted_ticket_cost : ℝ := 0.6 * normal_price

-- Proof Statement
theorem total_cost_of_tickets :
  website_tickets_cost + scalper_tickets_cost + discounted_ticket_cost = 360 :=
by
  sorry

end total_cost_of_tickets_l5_5136


namespace find_a_for_quadratic_max_l5_5925

theorem find_a_for_quadratic_max :
  ∃ a : ℝ, (∀ x : ℝ, a ≤ x ∧ x ≤ 1/2 → (x^2 + 2 * x - 2 ≤ 1)) ∧
           (∃ x : ℝ, a ≤ x ∧ x ≤ 1/2 ∧ (x^2 + 2 * x - 2 = 1)) ∧ 
           a = -3 :=
sorry

end find_a_for_quadratic_max_l5_5925


namespace solution_set_of_inequality_l5_5580

noncomputable def f : ℝ → ℝ := sorry

axiom f_domain : ∀ x : ℝ, true
axiom f_zero_eq : f 0 = 2
axiom f_derivative_ineq : ∀ x : ℝ, f x + (deriv f x) > 1

theorem solution_set_of_inequality : { x : ℝ | e^x * f x > e^x + 1 } = { x | x > 0 } :=
by
  sorry

end solution_set_of_inequality_l5_5580


namespace countEquilateralTriangles_l5_5640

-- Define the problem conditions
def numSmallTriangles := 18  -- The number of small equilateral triangles
def includesMarkedTriangle: Prop := True  -- All counted triangles include the marked triangle "**"

-- Define the main question as a proposition
def totalEquilateralTriangles : Prop :=
  (numSmallTriangles = 18 ∧ includesMarkedTriangle) → (1 + 4 + 1 = 6)

-- The theorem stating the number of equilateral triangles containing the marked triangle
theorem countEquilateralTriangles : totalEquilateralTriangles :=
  by
    sorry

end countEquilateralTriangles_l5_5640


namespace cone_angle_l5_5551

theorem cone_angle (r l : ℝ) (α : ℝ)
  (h1 : 2 * Real.pi * r = Real.pi * l) 
  (h2 : Real.cos α = r / l) : α = Real.pi / 3 :=
by
  sorry

end cone_angle_l5_5551


namespace total_bananas_in_collection_l5_5352

-- Definitions based on the conditions
def group_size : ℕ := 18
def number_of_groups : ℕ := 10

-- The proof problem statement
theorem total_bananas_in_collection : group_size * number_of_groups = 180 := by
  sorry

end total_bananas_in_collection_l5_5352


namespace x_plus_y_value_l5_5962

theorem x_plus_y_value (x y : ℕ) (h1 : 2^x = 8^(y + 1)) (h2 : 9^y = 3^(x - 9)) : x + y = 27 :=
by
  sorry

end x_plus_y_value_l5_5962


namespace complement_of_angle_l5_5997

theorem complement_of_angle (x : ℝ) (h : 90 - x = 3 * x + 10) : x = 20 := by
  sorry

end complement_of_angle_l5_5997


namespace truncated_cone_resistance_l5_5275

theorem truncated_cone_resistance (a b h : ℝ) (ρ : ℝ) (a_pos : 0 < a) (b_pos : 0 < b) (h_pos : 0 < h) :
  (∫ x in (0:ℝ)..h, ρ / (π * ((a + x * (b - a) / h) / 2) ^ 2)) = 4 * ρ * h / (π * a * b) := 
sorry

end truncated_cone_resistance_l5_5275


namespace evaluate_expression_l5_5107

theorem evaluate_expression : (502 * 502) - (501 * 503) = 1 := sorry

end evaluate_expression_l5_5107


namespace range_of_a_l5_5826

theorem range_of_a (a : ℝ) 
  (h : ¬ ∃ x : ℝ, Real.exp x ≤ 2 * x + a) : a < 2 - 2 * Real.log 2 := 
  sorry

end range_of_a_l5_5826


namespace sequence_problems_l5_5985
open Nat

-- Define the arithmetic sequence conditions
def arith_seq_condition_1 (a : ℕ → ℤ) : Prop :=
  a 2 + a 7 = -23

def arith_seq_condition_2 (a : ℕ → ℤ) : Prop :=
  a 3 + a 8 = -29

-- Define the geometric sequence condition
def geom_seq_condition (a b : ℕ → ℤ) (c : ℤ) : Prop :=
  ∀ n, a n + b n = c^(n - 1)

-- Define the arithmetic sequence formula
def arith_seq_formula (a : ℕ → ℤ) : Prop :=
  ∀ n, a n = -3 * n + 2

-- Define the sum of the first n terms of the sequence b_n
def sum_b_n (b : ℕ → ℤ) (S_n : ℕ → ℤ) (c : ℤ) : Prop :=
  (c = 1 → ∀ n, S_n n = (3 * n^2 + n) / 2) ∧
  (c ≠ 1 → ∀ n, S_n n = (n * (3 * n - 1)) / 2 + ((1 - c^n) / (1 - c)))

-- Define the main theorem
theorem sequence_problems (a b : ℕ → ℤ) (c : ℤ) (S_n : ℕ → ℤ) :
  arith_seq_condition_1 a →
  arith_seq_condition_2 a →
  geom_seq_condition a b c →
  arith_seq_formula a ∧ sum_b_n b S_n c :=
by
  -- Proofs for the conditions to the formula
  sorry

end sequence_problems_l5_5985


namespace geometric_sequence_S9_l5_5784

theorem geometric_sequence_S9 (S : ℕ → ℝ) (S3_eq : S 3 = 2) (S6_eq : S 6 = 6) : S 9 = 14 :=
by
  sorry

end geometric_sequence_S9_l5_5784


namespace find_number_l5_5349

theorem find_number (x : ℝ) (h : (x / 6) * 12 = 11) : x = 5.5 :=
by
  sorry

end find_number_l5_5349


namespace total_profit_l5_5323

-- Definitions
def investment_a : ℝ := 45000
def investment_b : ℝ := 63000
def investment_c : ℝ := 72000
def c_share : ℝ := 24000

-- Theorem statement
theorem total_profit : (investment_a + investment_b + investment_c) * (c_share / investment_c) = 60000 := by
  sorry

end total_profit_l5_5323


namespace divide_composite_products_l5_5149

def first_eight_composites : List ℕ := [4, 6, 8, 9, 10, 12, 14, 15]
def next_eight_composites : List ℕ := [16, 18, 20, 21, 22, 24, 25, 26]

def product (l : List ℕ) : ℕ := l.foldr (· * ·) 1

theorem divide_composite_products :
  product first_eight_composites * 3120 = product next_eight_composites :=
by
  -- This would be the place for the proof solution
  sorry

end divide_composite_products_l5_5149


namespace lines_parallel_or_coincident_l5_5714

/-- Given lines l₁ and l₂ with certain properties,
    prove that they are either parallel or coincident. -/
theorem lines_parallel_or_coincident
  (P Q : ℝ × ℝ)
  (hP : P = (-2, -1))
  (hQ : Q = (3, -6))
  (h_slope1 : ∀ θ, θ = 135 → Real.tan (θ * (Real.pi / 180)) = -1)
  (h_slope2 : (Q.2 - P.2) / (Q.1 - P.1) = -1) : 
  true :=
by sorry

end lines_parallel_or_coincident_l5_5714


namespace more_boys_than_girls_l5_5742

theorem more_boys_than_girls : 
  let girls := 28.0
  let boys := 35.0
  boys - girls = 7.0 :=
by
  sorry

end more_boys_than_girls_l5_5742


namespace compute_a1d1_a2d2_a3d3_eq_1_l5_5660

theorem compute_a1d1_a2d2_a3d3_eq_1 {a1 a2 a3 d1 d2 d3 : ℝ}
  (h : ∀ x : ℝ, x^6 + x^5 + x^4 + x^3 + x^2 + x + 1 = (x^2 + a1 * x + d1) * (x^2 + a2 * x + d2) * (x^2 + a3 * x + d3)) :
  a1 * d1 + a2 * d2 + a3 * d3 = 1 := by
  sorry

end compute_a1d1_a2d2_a3d3_eq_1_l5_5660


namespace hyperbola_eccentricity_l5_5594

noncomputable def point_on_hyperbola (x y a b : ℝ) (h_pos_a : a > 0) (h_pos_b : b > 0) : Prop :=
  (x^2 / a^2) - (y^2 / b^2) = 1

noncomputable def focal_length (a b c : ℝ) : Prop :=
  2 * c = 4

noncomputable def eccentricity (e c a : ℝ) : Prop :=
  e = c / a

theorem hyperbola_eccentricity 
  (a b c e : ℝ)
  (h_pos_a : a > 0)
  (h_pos_b : b > 0)
  (h_point_on_hyperbola : point_on_hyperbola 2 3 a b h_pos_a h_pos_b)
  (h_focal_length : focal_length a b c)
  : eccentricity e c a :=
sorry -- proof omitted

end hyperbola_eccentricity_l5_5594


namespace balance_rearrangement_vowels_at_end_l5_5670

theorem balance_rearrangement_vowels_at_end : 
  let vowels := ['A', 'A', 'E'];
  let consonants := ['B', 'L', 'N', 'C'];
  (Nat.factorial 3 / Nat.factorial 2) * Nat.factorial 4 = 72 :=
by
  sorry

end balance_rearrangement_vowels_at_end_l5_5670


namespace symmetric_axis_parabola_l5_5679

theorem symmetric_axis_parabola (h k : ℝ) (x : ℝ) :
  (∀ x, y = (x - h)^2 + k) → h = 2 → (x = 2) :=
by
  sorry

end symmetric_axis_parabola_l5_5679


namespace sally_seashells_l5_5678

variable (M : ℝ)

theorem sally_seashells : 
  (1.20 * (M + M / 2) = 54) → M = 30 := 
by
  sorry

end sally_seashells_l5_5678


namespace value_of_c_l5_5523

theorem value_of_c (b c : ℝ) (h1 : (x : ℝ) → (x + 4) * (x + b) = x^2 + c * x + 12) : c = 7 :=
by
  have h2 : 4 * b = 12 := by sorry
  have h3 : b = 3 := by sorry
  have h4 : c = b + 4 := by sorry
  rw [h3] at h4
  rw [h4]
  exact by norm_num

end value_of_c_l5_5523


namespace base_number_l5_5881

theorem base_number (a x : ℕ) (h1 : a ^ x - a ^ (x - 2) = 3 * 2 ^ 11) (h2 : x = 13) : a = 2 :=
by
  sorry

end base_number_l5_5881


namespace max_length_is_3sqrt2_l5_5464

noncomputable def max_vector_length (θ : ℝ) (h : 0 ≤ θ ∧ θ < 2 * Real.pi) : ℝ :=
  let OP₁ := (Real.cos θ, Real.sin θ)
  let OP₂ := (2 + Real.sin θ, 2 - Real.cos θ)
  let P₁P₂ := (OP₂.1 - OP₁.1, OP₂.2 - OP₁.2)
  Real.sqrt ((P₁P₂.1)^2 + (P₁P₂.2)^2)

theorem max_length_is_3sqrt2 : ∀ θ : ℝ, 0 ≤ θ ∧ θ < 2 * Real.pi → max_vector_length θ sorry = 3 * Real.sqrt 2 := 
sorry

end max_length_is_3sqrt2_l5_5464


namespace kendra_change_is_correct_l5_5398

-- Define the initial conditions
def price_wooden_toy : ℕ := 20
def price_hat : ℕ := 10
def kendra_initial_money : ℕ := 100
def num_wooden_toys : ℕ := 2
def num_hats : ℕ := 3

-- Calculate the total costs
def total_wooden_toys_cost : ℕ := price_wooden_toy * num_wooden_toys
def total_hats_cost : ℕ := price_hat * num_hats
def total_cost : ℕ := total_wooden_toys_cost + total_hats_cost

-- Calculate the change Kendra received
def kendra_change : ℕ := kendra_initial_money - total_cost

theorem kendra_change_is_correct : kendra_change = 30 := by
  sorry

end kendra_change_is_correct_l5_5398


namespace emily_subtracts_99_l5_5617

theorem emily_subtracts_99 (a b : ℕ) : (a = 50) → (b = 1) → (49^2 = 50^2 - 99) :=
by
  sorry

end emily_subtracts_99_l5_5617


namespace mary_total_baseball_cards_l5_5723

noncomputable def mary_initial_baseball_cards : ℕ := 18
noncomputable def torn_baseball_cards : ℕ := 8
noncomputable def fred_given_baseball_cards : ℕ := 26
noncomputable def mary_bought_baseball_cards : ℕ := 40

theorem mary_total_baseball_cards :
  mary_initial_baseball_cards - torn_baseball_cards + fred_given_baseball_cards + mary_bought_baseball_cards = 76 :=
by
  sorry

end mary_total_baseball_cards_l5_5723


namespace minimal_pieces_required_for_cubes_l5_5748

theorem minimal_pieces_required_for_cubes 
  (e₁ e₂ n₁ n₂ n₃ : ℕ)
  (h₁ : e₁ = 14)
  (h₂ : e₂ = 10)
  (h₃ : n₁ = 13)
  (h₄ : n₂ = 11)
  (h₅ : n₃ = 6)
  (disassembly_possible : ∀ {x y z : ℕ}, x^3 + y^3 = z^3 → n₁^3 + n₂^3 + n₃^3 = 14^3 + 10^3)
  (cutting_constraints : ∀ d : ℕ, (d > 0) → (d ≤ e₁ ∨ d ≤ e₂) → (d ≤ n₁ ∨ d ≤ n₂ ∨ d ≤ n₃) → (d ≤ 6))
  : ∃ minimal_pieces : ℕ, minimal_pieces = 11 := 
sorry

end minimal_pieces_required_for_cubes_l5_5748


namespace sequence_term_is_correct_l5_5499

theorem sequence_term_is_correct : ∀ (n : ℕ), (n = 7) → (2 * Real.sqrt 5 = Real.sqrt (3 * n - 1)) :=
by
  sorry

end sequence_term_is_correct_l5_5499


namespace three_digit_divisible_by_11_l5_5858

theorem three_digit_divisible_by_11 {x y z : ℕ} 
  (h1 : 0 ≤ x ∧ x < 10) 
  (h2 : 0 ≤ y ∧ y < 10) 
  (h3 : 0 ≤ z ∧ z < 10) 
  (h4 : x + z = y) : 
  (100 * x + 10 * y + z) % 11 = 0 := 
by 
  sorry

end three_digit_divisible_by_11_l5_5858


namespace predicted_height_at_age_10_l5_5070

-- Define the regression model as a function
def regression_model (x : ℝ) : ℝ := 7.19 * x + 73.93

-- Assert the predicted height at age 10
theorem predicted_height_at_age_10 : abs (regression_model 10 - 145.83) < 0.01 := 
by
  -- Here, we would prove the calculation steps
  sorry

end predicted_height_at_age_10_l5_5070


namespace smallest_integer_in_set_l5_5746

theorem smallest_integer_in_set (median : ℤ) (greatest : ℤ) (h1 : median = 157) (h2 : greatest = 169) :
  ∃ (smallest : ℤ), smallest = 145 :=
by
  -- Setup the conditions
  have set_cons_odd : True := trivial
  -- Known facts
  have h_median : median = 157 := by exact h1
  have h_greatest : greatest = 169 := by exact h2
  -- We must prove
  existsi 145
  sorry

end smallest_integer_in_set_l5_5746


namespace find_c_of_binomial_square_l5_5674

theorem find_c_of_binomial_square (c : ℝ) (h : ∃ d : ℝ, (9*x^2 - 24*x + c = (3*x + d)^2)) : c = 16 := sorry

end find_c_of_binomial_square_l5_5674


namespace sin_45_degrees_l5_5675

noncomputable def Q := (Real.sqrt 2 / 2, Real.sqrt 2 / 2)

theorem sin_45_degrees : Real.sin (Real.pi / 4) = Real.sqrt 2 / 2 := 
by sorry

end sin_45_degrees_l5_5675


namespace expression_equals_thirteen_l5_5456

-- Define the expression
def expression : ℤ :=
    8 + 15 / 3 - 4 * 2 + Nat.pow 2 3

-- State the theorem that proves the value of the expression
theorem expression_equals_thirteen : expression = 13 :=
by
  sorry

end expression_equals_thirteen_l5_5456


namespace mod_pow_solution_l5_5344

def m (x : ℕ) := x

theorem mod_pow_solution :
  ∃ (m : ℕ), 0 ≤ m ∧ m < 8 ∧ 13^6 % 8 = m ∧ m = 1 :=
by
  use 1
  sorry

end mod_pow_solution_l5_5344


namespace max_product_of_roots_of_quadratic_l5_5280

theorem max_product_of_roots_of_quadratic :
  ∃ k : ℚ, 6 * k^2 - 8 * k + (4 / 3) = 0 ∧ (64 - 48 * k) ≥ 0 ∧ (∀ k' : ℚ, (64 - 48 * k') ≥ 0 → (k'/3) ≤ (4/9)) :=
by
  sorry

end max_product_of_roots_of_quadratic_l5_5280


namespace goldfish_equal_in_seven_months_l5_5091

/-- Define the growth of Alice's goldfish: they triple every month. -/
def alice_goldfish (n : ℕ) : ℕ := 3 * 3 ^ n

/-- Define the growth of Bob's goldfish: they quadruple every month. -/
def bob_goldfish (n : ℕ) : ℕ := 256 * 4 ^ n

/-- The main theorem we want to prove: For Alice and Bob's goldfish count to be equal,
    it takes 7 months. -/
theorem goldfish_equal_in_seven_months : ∃ n : ℕ, alice_goldfish n = bob_goldfish n ∧ n = 7 := 
by
  sorry

end goldfish_equal_in_seven_months_l5_5091


namespace not_characteristic_of_algorithm_l5_5631

def characteristic_of_algorithm (c : String) : Prop :=
  c = "Abstraction" ∨ c = "Precision" ∨ c = "Finiteness"

theorem not_characteristic_of_algorithm : 
  ¬ characteristic_of_algorithm "Uniqueness" :=
by
  sorry

end not_characteristic_of_algorithm_l5_5631


namespace perimeter_of_triangle_l5_5898

theorem perimeter_of_triangle
  (P : ℝ)
  (r : ℝ := 1.5)
  (A : ℝ := 29.25)
  (h : A = r * (P / 2)) :
  P = 39 :=
by
  sorry

end perimeter_of_triangle_l5_5898


namespace brick_height_correct_l5_5160

-- Definitions
def wall_length : ℝ := 8
def wall_height : ℝ := 6
def wall_thickness : ℝ := 0.02 -- converted from 2 cm to meters
def brick_length : ℝ := 0.05 -- converted from 5 cm to meters
def brick_width : ℝ := 0.11 -- converted from 11 cm to meters
def brick_height : ℝ := 0.06 -- converted from 6 cm to meters
def number_of_bricks : ℝ := 2909.090909090909

-- Statement to prove
theorem brick_height_correct : brick_height = 0.06 := by
  sorry

end brick_height_correct_l5_5160


namespace matching_charge_and_minutes_l5_5384

def charge_at_time (x : ℕ) : ℕ :=
  100 - x / 6

def minutes_past_midnight (x : ℕ) : ℕ :=
  x % 60

theorem matching_charge_and_minutes :
  ∃ x, (x = 292 ∨ x = 343 ∨ x = 395 ∨ x = 446 ∨ x = 549) ∧ 
       charge_at_time x = minutes_past_midnight x :=
by {
  sorry
}

end matching_charge_and_minutes_l5_5384


namespace race_car_cost_l5_5145

variable (R : ℝ)
variable (Mater_cost SallyMcQueen_cost : ℝ)

-- Conditions
def Mater_cost_def : Mater_cost = 0.10 * R := by sorry
def SallyMcQueen_cost_def : SallyMcQueen_cost = 3 * Mater_cost := by sorry
def SallyMcQueen_cost_val : SallyMcQueen_cost = 42000 := by sorry

-- Theorem to prove the race car cost
theorem race_car_cost : R = 140000 :=
  by
    -- Use the conditions to prove
    sorry

end race_car_cost_l5_5145


namespace positive_integer_divisors_of_sum_l5_5545

theorem positive_integer_divisors_of_sum (n : ℕ) :
  (∃ n_values : Finset ℕ, 
    (∀ n ∈ n_values, n > 0 
      ∧ (n * (n + 1)) ∣ (2 * 10 * n)) 
      ∧ n_values.card = 5) :=
by
  sorry

end positive_integer_divisors_of_sum_l5_5545


namespace main_theorem_l5_5022

-- The condition
def condition (f : ℝ → ℝ) : Prop :=
∀ x : ℝ, f (2^x - 1) = 4^x - 1

-- The property we need to prove
def proves (f : ℝ → ℝ) : Prop :=
∀ x : ℝ, -1 ≤ x → f x = x^2 + 2*x

-- The main theorem connecting the condition to the desired property
theorem main_theorem (f : ℝ → ℝ) (h : condition f) : proves f :=
sorry

end main_theorem_l5_5022


namespace arithmetic_mean_calc_l5_5385

theorem arithmetic_mean_calc (x a : ℝ) (hx : x ≠ 0) (ha : a ≠ 0) :
  ( ( (x + a)^2 / x ) + ( (x - a)^2 / x ) ) / 2 = x + (a^2 / x) :=
sorry

end arithmetic_mean_calc_l5_5385


namespace poly_sequence_correct_l5_5357

-- Sequence of polynomials defined recursively
def f : ℕ → ℕ → ℕ 
| 0, x => 1
| 1, x => 1 + x 
| (k + 1), x => ((x + 1) * f (k) (x) - (x - k) * f (k - 1) (x)) / (k + 1)

-- Prove f(k, k) = 2^k for all k ≥ 0
theorem poly_sequence_correct (k : ℕ) : f k k = 2 ^ k := by
  sorry

end poly_sequence_correct_l5_5357


namespace average_age_of_girls_l5_5065

variable (B G : ℝ)
variable (age_students age_boys age_girls : ℝ)
variable (ratio_boys_girls : ℝ)

theorem average_age_of_girls :
  age_students = 15.8 ∧ age_boys = 16.2 ∧ ratio_boys_girls = 1.0000000000000044 ∧ B / G = ratio_boys_girls →
  (B * age_boys + G * age_girls) / (B + G) = age_students →
  age_girls = 15.4 :=
by
  intros hconds haverage
  sorry

end average_age_of_girls_l5_5065


namespace cake_cubes_with_exactly_two_faces_iced_l5_5958

theorem cake_cubes_with_exactly_two_faces_iced :
  let cake : ℕ := 3 -- cake dimension
  let total_cubes : ℕ := cake ^ 3 -- number of smaller cubes (total 27)
  let cubes_with_two_faces_icing := 4
  (∀ cake icing (smaller_cubes : ℕ), icing ≠ 0 → smaller_cubes = cake ^ 3 → 
    let top_iced := cake - 2 -- cubes with icing on top only
    let front_iced := cake - 2 -- cubes with icing on front only
    let back_iced := cake - 2 -- cubes with icing on back only
    ((top_iced * 2) = cubes_with_two_faces_icing)) :=
  sorry

end cake_cubes_with_exactly_two_faces_iced_l5_5958


namespace simplify_expression_l5_5838

theorem simplify_expression (x : ℝ) : (3 * x + 15) + (100 * x + 15) + (10 * x - 5) = 113 * x + 25 :=
by
  sorry

end simplify_expression_l5_5838


namespace solve_for_a_l5_5241

def op (a b : ℝ) : ℝ := 3 * a - 2 * b ^ 2

theorem solve_for_a (a : ℝ) : op a 3 = 15 → a = 11 :=
by
  intro h
  rw [op] at h
  sorry

end solve_for_a_l5_5241


namespace f_2015_2016_l5_5318

theorem f_2015_2016 (f : ℤ → ℤ) 
  (h_odd : ∀ x, f (-x) = -f x)
  (h_periodic : ∀ x, f (x + 2) = -f x)
  (h_f1 : f 1 = 2) :
  f 2015 + f 2016 = -2 :=
sorry

end f_2015_2016_l5_5318


namespace range_of_m_l5_5040

variables {m x : ℝ}

def p (m : ℝ) : Prop := (16 * (m - 2)^2 - 16 > 0) ∧ (m - 2 < 0)
def q (m : ℝ) : Prop := (9 * m^2 - 4 < 0)
def pq (m : ℝ) : Prop := (p m ∨ q m) ∧ ¬(q m)

theorem range_of_m (h : pq m) : m ≤ -2/3 ∨ (2/3 ≤ m ∧ m < 1) :=
sorry

end range_of_m_l5_5040


namespace base_comparison_l5_5735

theorem base_comparison : (1 * 6^1 + 2 * 6^0) > (1 * 2^2 + 0 * 2^1 + 1 * 2^0) := by
  sorry

end base_comparison_l5_5735


namespace gcd_of_45_and_75_l5_5703

def gcd_problem : Prop :=
  gcd 45 75 = 15

theorem gcd_of_45_and_75 : gcd_problem :=
by {
  sorry
}

end gcd_of_45_and_75_l5_5703


namespace largest_int_less_than_100_remainder_4_l5_5935

theorem largest_int_less_than_100_remainder_4 : ∃ x : ℤ, x < 100 ∧ (∃ m : ℤ, x = 6 * m + 4) ∧ x = 94 := 
by
  sorry

end largest_int_less_than_100_remainder_4_l5_5935


namespace distinct_four_digit_integers_l5_5516

open Nat

theorem distinct_four_digit_integers (count_digs_18 : ℕ) :
  (∀ n : ℕ, 1000 ≤ n ∧ n < 10000 → (∃ d1 d2 d3 d4 : ℕ,
      d1 * d2 * d3 * d4 = 18 ∧
      d1 > 0 ∧ d1 < 10 ∧
      d2 > 0 ∧ d2 < 10 ∧
      d3 > 0 ∧ d3 < 10 ∧
      d4 > 0 ∧ d4 < 10 ∧
      n = d1 * 1000 + d2 * 100 + d3 * 10 + d4)) →
  count_digs_18 = 24 :=
sorry

end distinct_four_digit_integers_l5_5516


namespace four_point_questions_l5_5602

theorem four_point_questions (x y : ℕ) (h1 : x + y = 40) (h2 : 2 * x + 4 * y = 100) : y = 10 :=
sorry

end four_point_questions_l5_5602


namespace smallest_value_of_sum_l5_5396

theorem smallest_value_of_sum (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : 3 * a = 4 * b ∧ 4 * b = 7 * c) : a + b + c = 61 :=
sorry

end smallest_value_of_sum_l5_5396


namespace greatest_possible_sum_of_two_consecutive_integers_lt_500_l5_5415

theorem greatest_possible_sum_of_two_consecutive_integers_lt_500 (n : ℕ) (h : n * (n + 1) < 500) : n + (n + 1) ≤ 43 := by
  sorry

end greatest_possible_sum_of_two_consecutive_integers_lt_500_l5_5415


namespace greatest_possible_int_diff_l5_5757

theorem greatest_possible_int_diff (x a y b : ℝ) 
    (hx : 3 < x ∧ x < 4) 
    (ha : 4 < a ∧ a < x) 
    (hy : 6 < y ∧ y < 8) 
    (hb : 8 < b ∧ b < y) 
    (h_ineq : a^2 + b^2 > x^2 + y^2) : 
    abs (⌊x⌋ - ⌈y⌉) = 2 :=
sorry

end greatest_possible_int_diff_l5_5757


namespace least_possible_integer_discussed_l5_5596
open Nat

theorem least_possible_integer_discussed (N : ℕ) (H : ∀ k : ℕ, 1 ≤ k ∧ k ≤ 30 → k ≠ 8 ∧ k ≠ 9 → k ∣ N) : N = 2329089562800 :=
sorry

end least_possible_integer_discussed_l5_5596


namespace find_expression_l5_5509

def even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def symmetric_about_x2 (f : ℝ → ℝ) : Prop :=
  ∀ x, f (2 + x) = f (2 - x)

theorem find_expression (f : ℝ → ℝ)
  (h1 : even_function f)
  (h2 : symmetric_about_x2 f)
  (h3 : ∀ x, -2 < x ∧ x ≤ 2 → f x = -x^2 + 1) :
  ∀ x, -6 < x ∧ x < -2 → f x = -(x + 4)^2 + 1 :=
by
  sorry

end find_expression_l5_5509


namespace Robie_boxes_with_him_l5_5585

-- Definition of the given conditions
def total_cards : Nat := 75
def cards_per_box : Nat := 10
def cards_not_placed : Nat := 5
def boxes_given_away : Nat := 2

-- Definition of the proof that Robie has 5 boxes with him
theorem Robie_boxes_with_him : ((total_cards - cards_not_placed) / cards_per_box) - boxes_given_away = 5 := by
  sorry

end Robie_boxes_with_him_l5_5585


namespace probability_of_B_given_A_l5_5410

noncomputable def balls_in_box : Prop :=
  let total_balls := 12
  let yellow_balls := 5
  let blue_balls := 4
  let green_balls := 3
  let event_A := (yellow_balls * green_balls + yellow_balls * blue_balls + green_balls * blue_balls) / (total_balls * (total_balls - 1) / 2)
  let event_B := (yellow_balls * blue_balls) / (total_balls * (total_balls - 1) / 2)
  (event_B / event_A) = 20 / 47

theorem probability_of_B_given_A : balls_in_box := sorry

end probability_of_B_given_A_l5_5410


namespace traffic_light_probability_l5_5918

theorem traffic_light_probability :
  let total_cycle_time := 63
  let green_time := 30
  let yellow_time := 3
  let red_time := 30
  let observation_window := 3
  let change_intervals := 3 * 3
  ∃ (P : ℚ), P = change_intervals / total_cycle_time ∧ P = 1 / 7 := 
by
  sorry

end traffic_light_probability_l5_5918


namespace star_difference_l5_5701

def star (x y : ℤ) : ℤ := x * y + 3 * x - y

theorem star_difference : (star 7 4) - (star 4 7) = 12 := by
  sorry

end star_difference_l5_5701


namespace smallest_EF_minus_DE_l5_5400

theorem smallest_EF_minus_DE (x y z : ℕ) (h1 : x < y) (h2 : y ≤ z) (h3 : x + y + z = 2050)
  (h4 : x + y > z) (h5 : y + z > x) (h6 : z + x > y) : y - x = 1 :=
by
  sorry

end smallest_EF_minus_DE_l5_5400


namespace beta_gt_half_alpha_l5_5968

theorem beta_gt_half_alpha (alpha beta : ℝ) (h1 : Real.sin beta = (3/4) * Real.sin alpha) (h2 : 0 < alpha ∧ alpha ≤ 90) : beta > alpha / 2 :=
by
  sorry

end beta_gt_half_alpha_l5_5968


namespace boat_travel_distance_per_day_l5_5971

-- Definitions from conditions
def men : ℕ := 25
def water_daily_per_man : ℚ := 1/2
def travel_distance : ℕ := 4000
def total_water : ℕ := 250

-- Main theorem
theorem boat_travel_distance_per_day : 
  ∀ (men : ℕ) (water_daily_per_man : ℚ) (travel_distance : ℕ) (total_water : ℕ), 
  men = 25 ∧ water_daily_per_man = 1/2 ∧ travel_distance = 4000 ∧ total_water = 250 ->
  travel_distance / (total_water / (men * water_daily_per_man)) = 200 :=
by
  sorry

end boat_travel_distance_per_day_l5_5971


namespace zero_intersections_l5_5711

noncomputable def Line : Type := sorry  -- Define Line as a type
noncomputable def is_skew (a b : Line) : Prop := sorry  -- Predicate for skew lines
noncomputable def is_common_perpendicular (EF a b : Line) : Prop := sorry  -- Predicate for common perpendicular
noncomputable def is_parallel (l EF : Line) : Prop := sorry  -- Predicate for parallel lines
noncomputable def count_intersections (l a b : Line) : ℕ := sorry  -- Function to count intersections

theorem zero_intersections (EF a b l : Line) 
  (h_skew : is_skew a b) 
  (h_common_perpendicular : is_common_perpendicular EF a b)
  (h_parallel : is_parallel l EF) : 
  count_intersections l a b = 0 := 
sorry

end zero_intersections_l5_5711


namespace part_a_part_b_l5_5727

noncomputable def f (g n : ℕ) : ℕ := g^n + 1

theorem part_a (g : ℕ) (h_even : g % 2 = 0) (h_pos : 0 < g) :
  ∀ n : ℕ, 0 < n → f g n ∣ f g (3*n) ∧ f g n ∣ f g (5*n) ∧ f g n ∣ f g (7*n) :=
sorry

theorem part_b (g : ℕ) (h_even : g % 2 = 0) (h_pos : 0 < g) :
  ∀ n : ℕ, 0 < n → ∀ k : ℕ, 1 ≤ k → gcd (f g n) (f g (2*k*n)) = 1 :=
sorry

end part_a_part_b_l5_5727


namespace total_number_of_posters_l5_5407

theorem total_number_of_posters : 
  ∀ (P : ℕ), 
  (2 / 5 : ℚ) * P + (1 / 2 : ℚ) * P + 5 = P → 
  P = 50 :=
by
  intro P
  intro h
  sorry

end total_number_of_posters_l5_5407


namespace part1_part2_l5_5654

noncomputable def f (x a : ℝ) : ℝ := Real.log x - a * x

theorem part1 (a : ℝ) (h : ∀ x > 0, f x a ≤ 0) : a ≥ 1 / Real.exp 1 :=
  sorry

noncomputable def g (x b : ℝ) : ℝ := Real.log x + 1/2 * x^2 - (b + 1) * x

theorem part2 (b : ℝ) (x1 x2 : ℝ) (h1 : b ≥ 3/2) (h2 : x1 < x2) (hx3 : g x1 b - g x2 b ≥ k) : k ≤ 15/8 - 2 * Real.log 2 :=
  sorry

end part1_part2_l5_5654


namespace range_of_a_fall_within_D_l5_5581

-- Define the conditions
variable (a : ℝ) (c : ℝ)
axiom A_through : c = 9
axiom D_through : a < 0 ∧ (6, 7) ∈ { (x, y) | y = a * x ^ 2 + c }

-- Prove the range of a given the conditions
theorem range_of_a : -1/4 < a ∧ a < -1/18 := sorry

-- Define the additional condition for point P
axiom P_through : (2, 8.1) ∈ { (x, y) | y = a * x ^ 2 + c }

-- Prove that the object can fall within interval D when passing through point P
theorem fall_within_D : a = -9/40 ∧ -1/4 < a ∧ a < -1/18 := sorry

end range_of_a_fall_within_D_l5_5581


namespace sum_of_altitudes_of_triangle_l5_5416

theorem sum_of_altitudes_of_triangle (a b c : ℝ) (h_line : ∀ x y, 8 * x + 10 * y = 80 → x = 10 ∨ y = 8) :
  (8 + 10 + 40/Real.sqrt 41) = 18 + 40/Real.sqrt 41 :=
by
  sorry

end sum_of_altitudes_of_triangle_l5_5416


namespace job_completion_time_l5_5574

theorem job_completion_time
  (A C : ℝ)
  (A_rate : A = 1 / 6)
  (C_rate : C = 1 / 12)
  (B_share : 390 / 1170 = 1 / 3) :
  ∃ B : ℝ, B = 1 / 8 ∧ (B * 8 = 1) :=
by
  -- Proof omitted
  sorry

end job_completion_time_l5_5574


namespace circumradius_relation_l5_5401

-- Definitions of the geometric constructs from the problem
open EuclideanGeometry

noncomputable def circumradius (A B C : Point) : Real := sorry

-- Given conditions
def angle_bisectors_intersect_at_point (A B C B1 C1 I : Point) : Prop := sorry
def line_intersects_circumcircle_at_points (B1 C1 : Point) (circumcircle : Circle) (M N : Point) : Prop := sorry

-- Main statement to prove
theorem circumradius_relation
  (A B C B1 C1 I M N : Point)
  (circumcircle : Circle)
  (h1 : angle_bisectors_intersect_at_point A B C B1 C1 I)
  (h2 : line_intersects_circumcircle_at_points B1 C1 circumcircle M N) :
  circumradius M I N = 2 * circumradius A B C :=
sorry

end circumradius_relation_l5_5401


namespace jaden_toy_cars_problem_l5_5655

theorem jaden_toy_cars_problem :
  let initial := 14
  let bought := 28
  let birthday := 12
  let to_vinnie := 3
  let left := 43
  let total := initial + bought + birthday
  let after_vinnie := total - to_vinnie
  (after_vinnie - left = 8) :=
by
  sorry

end jaden_toy_cars_problem_l5_5655


namespace yella_computer_usage_difference_l5_5391

-- Define the given conditions
def last_week_usage : ℕ := 91
def this_week_daily_usage : ℕ := 8
def days_in_week : ℕ := 7

-- Compute this week's total usage
def this_week_total_usage := this_week_daily_usage * days_in_week

-- Statement to prove
theorem yella_computer_usage_difference :
  last_week_usage - this_week_total_usage = 35 := 
by
  -- The proof will be filled in here
  sorry

end yella_computer_usage_difference_l5_5391


namespace no_afg_fourth_place_l5_5946

theorem no_afg_fourth_place
  (A B C D E F G : ℕ)
  (h1 : A < B)
  (h2 : A < C)
  (h3 : B < D)
  (h4 : C < E)
  (h5 : A < F ∧ F < B)
  (h6 : B < G ∧ G < C) :
  ¬ (A = 4 ∨ F = 4 ∨ G = 4) :=
by
  sorry

end no_afg_fourth_place_l5_5946


namespace simplify_to_quadratic_l5_5919

noncomputable def simplify_expression (a b c x : ℝ) : ℝ := 
  (x + a)^2 / ((a - b) * (a - c)) + 
  (x + b)^2 / ((b - a) * (b - c + 2)) + 
  (x + c)^2 / ((c - a) * (c - b))

theorem simplify_to_quadratic {a b c x : ℝ} (ha : a ≠ b) (hb : b ≠ c) (hc : c ≠ a) :
  simplify_expression a b c x = x^2 - (a + b + c) * x + sorry :=
sorry

end simplify_to_quadratic_l5_5919


namespace pollywogs_disappear_in_44_days_l5_5272

theorem pollywogs_disappear_in_44_days :
  ∀ (initial_count rate_mature rate_caught first_period_days : ℕ),
  initial_count = 2400 →
  rate_mature = 50 →
  rate_caught = 10 →
  first_period_days = 20 →
  (initial_count - first_period_days * (rate_mature + rate_caught)) / rate_mature + first_period_days = 44 := 
by
  intros initial_count rate_mature rate_caught first_period_days h1 h2 h3 h4
  sorry

end pollywogs_disappear_in_44_days_l5_5272


namespace calculate_total_area_l5_5356

theorem calculate_total_area :
  let height1 := 7
  let width1 := 6
  let width2 := 4
  let height2 := 5
  let height3 := 1
  let width3 := 2
  let width4 := 5
  let height4 := 6
  let area1 := width1 * height1
  let area2 := width2 * height2
  let area3 := height3 * width3
  let area4 := width4 * height4
  area1 + area2 + area3 + area4 = 94 := by
  sorry

end calculate_total_area_l5_5356


namespace johns_current_income_l5_5733

theorem johns_current_income
  (prev_income : ℝ := 1000000)
  (prev_tax_rate : ℝ := 0.20)
  (new_tax_rate : ℝ := 0.30)
  (extra_taxes_paid : ℝ := 250000) :
  ∃ (X : ℝ), 0.30 * X - 0.20 * prev_income = extra_taxes_paid ∧ X = 1500000 :=
by
  use 1500000
  -- Proof would come here
  sorry

end johns_current_income_l5_5733


namespace julia_money_given_l5_5086

-- Define the conditions
def num_snickers : ℕ := 2
def num_mms : ℕ := 3
def cost_snickers : ℚ := 1.5
def cost_mms : ℚ := 2 * cost_snickers
def change_received : ℚ := 8

-- The total cost Julia had to pay
def total_cost : ℚ := (num_snickers * cost_snickers) + (num_mms * cost_mms)

-- Julia gave this amount of money to the cashier
def money_given : ℚ := total_cost + change_received

-- The problem to prove
theorem julia_money_given : money_given = 20 := by
  sorry

end julia_money_given_l5_5086


namespace max_pieces_l5_5699

theorem max_pieces (plywood_width plywood_height piece_width piece_height : ℕ)
  (h_plywood : plywood_width = 22) (h_plywood_height : plywood_height = 15)
  (h_piece : piece_width = 3) (h_piece_height : piece_height = 5) :
  (plywood_width * plywood_height) / (piece_width * piece_height) = 22 := by
  sorry

end max_pieces_l5_5699


namespace max_number_of_girls_l5_5322

theorem max_number_of_girls (students : ℕ)
  (num_friends : ℕ → ℕ)
  (h_students : students = 25)
  (h_distinct_friends : ∀ (i j : ℕ), i ≠ j → num_friends i ≠ num_friends j)
  (h_girls_boys : ∃ (G B : ℕ), G + B = students) :
  ∃ G : ℕ, G = 13 := 
sorry

end max_number_of_girls_l5_5322


namespace fraction_simplify_l5_5924

theorem fraction_simplify:
  (1/5 + 1/7) / (3/8 - 1/9) = 864 / 665 :=
by
  sorry

end fraction_simplify_l5_5924


namespace minimum_value_of_f_l5_5566

noncomputable def f (x y z : ℝ) := (x^2) / (1 + x) + (y^2) / (1 + y) + (z^2) / (1 + z)

theorem minimum_value_of_f (a b c x y z : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : x > 0) (h5 : y > 0) (h6 : z > 0) 
  (h7 : b * z + c * y = a) (h8 : a * z + c * x = b) (h9 : a * y + b * x = c) : 
  f x y z ≥ 1 / 2 :=
sorry

end minimum_value_of_f_l5_5566


namespace mean_value_of_interior_angles_of_quadrilateral_l5_5215

theorem mean_value_of_interior_angles_of_quadrilateral :
  (360 / 4) = 90 := 
by
  sorry

end mean_value_of_interior_angles_of_quadrilateral_l5_5215


namespace work_problem_l5_5315

theorem work_problem 
  (A_real : ℝ)
  (B_days : ℝ := 16)
  (C_days : ℝ := 16)
  (ABC_days : ℝ := 4)
  (H_b : (1 / B_days) = 1 / 16)
  (H_c : (1 / C_days) = 1 / 16)
  (H_abc : (1 / A_real + 1 / B_days + 1 / C_days) = 1 / ABC_days) : 
  A_real = 8 := 
sorry

end work_problem_l5_5315


namespace coin_arrangements_l5_5059

theorem coin_arrangements (n m : ℕ) (hp_pos : n = 5) (hq_pos : m = 5) :
  ∃ (num_arrangements : ℕ), num_arrangements = 8568 :=
by
  -- Note: 'sorry' is used to indicate here that the proof is omitted.
  sorry

end coin_arrangements_l5_5059


namespace fraction_identity_l5_5558

theorem fraction_identity (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h1 : a^2 = b^2 + b * c) (h2 : b^2 = c^2 + a * c) : 
  (1 / c) = (1 / a) + (1 / b) :=
by 
  sorry

end fraction_identity_l5_5558


namespace cube_volume_l5_5252

theorem cube_volume (A : ℝ) (V : ℝ) (h : A = 64) : V = 512 :=
by
  sorry

end cube_volume_l5_5252


namespace proof_problem_l5_5795

-- Definitions of the conditions
def cond1 (r : ℕ) : Prop := 2^r = 16
def cond2 (s : ℕ) : Prop := 5^s = 25

-- Statement of the problem
theorem proof_problem (r s : ℕ) (h₁ : cond1 r) (h₂ : cond2 s) : r + s = 6 :=
by
  sorry

end proof_problem_l5_5795


namespace coupon1_greater_l5_5870

variable (x : ℝ)

def coupon1_discount (x : ℝ) : ℝ := 0.15 * x
def coupon2_discount : ℝ := 50
def coupon3_discount (x : ℝ) : ℝ := 0.25 * x - 62.5

theorem coupon1_greater (x : ℝ) (hx1 : 333.33 < x ∧ x < 625) : 
  coupon1_discount x > coupon2_discount ∧ coupon1_discount x > coupon3_discount x := by
  sorry

end coupon1_greater_l5_5870


namespace problem1_problem2_problem3_problem4_l5_5467

-- Problem 1
theorem problem1 (a : ℝ) : -2 * a^3 * 3 * a^2 = -6 * a^5 := 
by
  sorry

-- Problem 2
theorem problem2 (m : ℝ) : m^4 * (m^2)^3 / m^8 = m^2 := 
by
  sorry

-- Problem 3
theorem problem3 (x : ℝ) : (-2 * x - 1) * (2 * x - 1) = 1 - 4 * x^2 := 
by
  sorry

-- Problem 4
theorem problem4 (x : ℝ) : (-3 * x + 2)^2 = 9 * x^2 - 12 * x + 4 := 
by
  sorry

end problem1_problem2_problem3_problem4_l5_5467


namespace solve_equation_l5_5856

theorem solve_equation (a : ℝ) (x : ℝ) : (2 * a * x + 3) / (a - x) = 3 / 4 → x = 1 → a = -3 :=
by
  intros h h1
  rw [h1] at h
  sorry

end solve_equation_l5_5856


namespace repaved_today_l5_5185

theorem repaved_today (total before : ℕ) (h_total : total = 4938) (h_before : before = 4133) : total - before = 805 := by
  sorry

end repaved_today_l5_5185


namespace ratio_of_ripe_mangoes_l5_5707

theorem ratio_of_ripe_mangoes (total_mangoes : ℕ) (unripe_two_thirds : ℚ)
  (kept_unripe_mangoes : ℕ) (mangoes_per_jar : ℕ) (jars_made : ℕ)
  (h1 : total_mangoes = 54)
  (h2 : unripe_two_thirds = 2 / 3)
  (h3 : kept_unripe_mangoes = 16)
  (h4 : mangoes_per_jar = 4)
  (h5 : jars_made = 5) :
  1 / 3 = 18 / 54 :=
sorry

end ratio_of_ripe_mangoes_l5_5707


namespace arith_geo_mean_extended_arith_geo_mean_l5_5507
noncomputable section

open Real

-- Definition for Problem 1
def arith_geo_mean_inequality (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) : Prop :=
  (a + b) / 2 ≥ Real.sqrt (a * b)

-- Theorem for Problem 1
theorem arith_geo_mean (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) : arith_geo_mean_inequality a b h1 h2 :=
  sorry

-- Definition for Problem 2
def extended_arith_geo_mean_inequality (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) : Prop :=
  (a + b) * (b + c) * (c + a) ≥ 8 * a * b * c

-- Theorem for Problem 2
theorem extended_arith_geo_mean (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) : extended_arith_geo_mean_inequality a b c h1 h2 h3 :=
  sorry

end arith_geo_mean_extended_arith_geo_mean_l5_5507


namespace problem_statement_l5_5184

noncomputable def g (x : ℝ) : ℝ := x^2 - 2 * Real.sqrt x

theorem problem_statement : 3 * g 3 - g 9 = -48 - 6 * Real.sqrt 3 := by
  sorry

end problem_statement_l5_5184


namespace band_formation_l5_5752

theorem band_formation (r x m : ℕ) (h1 : r * x + 3 = m) (h2 : (r - 1) * (x + 2) = m) (h3 : m < 100) : m = 69 :=
by
  sorry

end band_formation_l5_5752


namespace tea_mixture_ratio_l5_5533

theorem tea_mixture_ratio
    (x y : ℝ)
    (h₁ : 62 * x + 72 * y = 64.5 * (x + y)) :
    x / y = 3 := by
  sorry

end tea_mixture_ratio_l5_5533


namespace leaves_falling_every_day_l5_5559

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

end leaves_falling_every_day_l5_5559


namespace line_equation_passes_through_l5_5583

theorem line_equation_passes_through (a b : ℝ) (x y : ℝ) 
  (h_intercept : b = a + 1)
  (h_point : (6 * b) + (-2 * a) = a * b) :
  (x + 2 * y - 2 = 0 ∨ 2 * x + 3 * y - 6 = 0) := 
sorry

end line_equation_passes_through_l5_5583


namespace sin_difference_identity_l5_5501

theorem sin_difference_identity 
  (α : ℝ) (hα : 0 < α ∧ α < π / 2) (hcos : Real.cos α = 1 / 3) : 
  Real.sin (π / 4 - α) = (Real.sqrt 2 - 4) / 6 := 
  sorry

end sin_difference_identity_l5_5501


namespace ratio_of_areas_l5_5814

theorem ratio_of_areas (len_rect width_rect area_tri : ℝ) (h1 : len_rect = 6) (h2 : width_rect = 4) (h3 : area_tri = 60) :
    (len_rect * width_rect) / area_tri = 2 / 5 :=
by
  rw [h1, h2, h3]
  norm_num

end ratio_of_areas_l5_5814


namespace solve_triangle_l5_5598

open Real

noncomputable def triangle_sides_angles (a b c A B C : ℝ) : Prop :=
  b^2 - (2 * (sqrt 3 / 3) * b * c * sin A) + c^2 = a^2

theorem solve_triangle 
  (b c : ℝ) (hb : b = 2) (hc : c = 3)
  (h : triangle_sides_angles a b c A B C) : 
  (A = π / 3) ∧ 
  (a = sqrt 7) ∧ 
  (sin (2 * B - A) = 3 * sqrt 3 / 14) := 
by
  sorry

end solve_triangle_l5_5598


namespace Carla_final_position_l5_5021

-- Carla's initial position
def Carla_initial_position : ℤ × ℤ := (10, -10)

-- Function to calculate Carla's new position after each move
def Carla_move (pos : ℤ × ℤ) (direction : ℕ) (distance : ℤ) : ℤ × ℤ :=
  match direction % 4 with
  | 0 => (pos.1, pos.2 + distance)   -- North
  | 1 => (pos.1 + distance, pos.2)   -- East
  | 2 => (pos.1, pos.2 - distance)   -- South
  | 3 => (pos.1 - distance, pos.2)   -- West
  | _ => pos  -- This case will never happen due to the modulo operation

-- Recursive function to simulate Carla's journey
def Carla_journey : ℕ → ℤ × ℤ → ℤ × ℤ 
  | 0, pos => pos
  | n + 1, pos => 
    let next_pos := Carla_move pos n (2 + n / 2 * 2)
    Carla_journey n next_pos

-- Prove that after 100 moves, Carla's position is (-191, -10)
theorem Carla_final_position : Carla_journey 100 Carla_initial_position = (-191, -10) :=
sorry

end Carla_final_position_l5_5021


namespace probability_more_than_60000_l5_5372

def boxes : List ℕ := [8, 800, 8000, 40000, 80000]

def probability_keys (keys : ℕ) : ℚ :=
  1 / keys

def probability_winning (n : ℕ) : ℚ :=
  if n = 4 then probability_keys 5 + probability_keys 5 * probability_keys 4 else 0

theorem probability_more_than_60000 : 
  probability_winning 4 = 1/4 := sorry

end probability_more_than_60000_l5_5372


namespace find_first_term_geometric_series_l5_5588

variables {a r : ℝ}

theorem find_first_term_geometric_series
  (h1 : a / (1 - r) = 20)
  (h2 : a^2 / (1 - r^2) = 80) :
  a = 20 / 3 :=
sorry

end find_first_term_geometric_series_l5_5588


namespace sin_of_angle_F_l5_5138

theorem sin_of_angle_F 
  (DE EF DF : ℝ) 
  (h : DE = 12) 
  (h0 : EF = 20) 
  (h1 : DF = Real.sqrt (DE^2 + EF^2)) : 
  Real.sin (Real.arctan (DF / EF)) = 12 / Real.sqrt (DE^2 + EF^2) := 
by 
  sorry

end sin_of_angle_F_l5_5138


namespace find_digits_of_abc_l5_5305

theorem find_digits_of_abc (a b c : ℕ) (h1 : a ≠ c) (h2 : c - a = 3) (h3 : (100 * a + 10 * b + c) - (100 * c + 10 * a + b) = 100 * (a - (c - 1)) + 0 + (b - b)) : 
  100 * a + 10 * b + c = 619 :=
by
  sorry

end find_digits_of_abc_l5_5305


namespace relationship_between_x_y_l5_5041

theorem relationship_between_x_y (x y m : ℝ) (h₁ : x + m = 4) (h₂ : y - 5 = m) : x + y = 9 := 
sorry

end relationship_between_x_y_l5_5041


namespace students_calculation_l5_5666

variable (students_boys students_playing_soccer students_not_playing_soccer girls_not_playing_soccer : ℕ)
variable (percentage_boys_play_soccer : ℚ)

def students_not_playing_sum (students_boys_not_playing : ℕ) : ℕ :=
  students_boys_not_playing + girls_not_playing_soccer

def total_students (students_not_playing_sum students_playing_soccer : ℕ) : ℕ :=
  students_not_playing_sum + students_playing_soccer

theorem students_calculation 
  (H1 : students_boys = 312)
  (H2 : students_playing_soccer = 250)
  (H3 : percentage_boys_play_soccer = 0.86)
  (H4 : girls_not_playing_soccer = 73)
  (H5 : percentage_boys_play_soccer * students_playing_soccer = 215)
  (H6 : students_boys - 215 = 97)
  (H7 : students_not_playing_sum 97 = 170)
  (H8 : total_students 170 250 = 420) : ∃ total, total = 420 :=
by 
  existsi total_students 170 250
  exact H8

end students_calculation_l5_5666


namespace identical_lines_unique_pair_l5_5965

theorem identical_lines_unique_pair :
  ∃! (a b : ℚ), 2 * (0 : ℚ) + a * (0 : ℚ) + 10 = 0 ∧ b * (0 : ℚ) - 3 * (0 : ℚ) - 15 = 0 ∧ 
  (-2 / a = b / 3) ∧ (-10 / a = 5) :=
by {
  -- Given equations in slope-intercept form:
  -- y = -2 / a * x - 10 / a
  -- y = b / 3 * x + 5
  -- Slope and intercept comparison leads to equations:
  -- -2 / a = b / 3
  -- -10 / a = 5
  sorry
}

end identical_lines_unique_pair_l5_5965


namespace romance_movie_tickets_l5_5469

-- Define the given conditions.
def horror_movie_tickets := 93
def relationship (R : ℕ) := 3 * R + 18 = horror_movie_tickets

-- The theorem we need to prove
theorem romance_movie_tickets (R : ℕ) (h : relationship R) : R = 25 :=
by sorry

end romance_movie_tickets_l5_5469


namespace point_in_plane_region_l5_5419

theorem point_in_plane_region :
  (2 * 0 + 1 - 6 < 0) ∧ ¬(2 * 5 + 0 - 6 < 0) ∧ ¬(2 * 0 + 7 - 6 < 0) ∧ ¬(2 * 2 + 3 - 6 < 0) :=
by
  -- Proof detail goes here.
  sorry

end point_in_plane_region_l5_5419


namespace sphere_surface_area_of_circumscribing_cuboid_l5_5369

theorem sphere_surface_area_of_circumscribing_cuboid :
  ∀ (a b c : ℝ), a = 5 ∧ b = 4 ∧ c = 3 → 4 * Real.pi * ((Real.sqrt ((a^2 + b^2 + c^2)) / 2) ^ 2) = 50 * Real.pi :=
by
  -- introduction of variables and conditions
  intros a b c h
  obtain ⟨_, _, _⟩ := h -- decomposing the conditions
  -- the proof is skipped
  sorry

end sphere_surface_area_of_circumscribing_cuboid_l5_5369


namespace oranges_per_tree_correct_l5_5234

-- Definitions for the conditions
def betty_oranges : ℕ := 15
def bill_oranges : ℕ := 12
def total_oranges := betty_oranges + bill_oranges
def frank_oranges := 3 * total_oranges
def seeds_planted := 2 * frank_oranges
def total_trees := seeds_planted
def total_oranges_picked := 810
def oranges_per_tree := total_oranges_picked / total_trees

-- Theorem statement
theorem oranges_per_tree_correct : oranges_per_tree = 5 :=
by
  -- Proof steps would go here
  sorry

end oranges_per_tree_correct_l5_5234


namespace shelves_used_l5_5808

-- Define the initial conditions
def initial_stock : Float := 40.0
def additional_stock : Float := 20.0
def books_per_shelf : Float := 4.0

-- Define the total number of books
def total_books : Float := initial_stock + additional_stock

-- Define the number of shelves
def number_of_shelves : Float := total_books / books_per_shelf

-- The proof statement that needs to be proven
theorem shelves_used : number_of_shelves = 15.0 :=
by
  -- The proof will go here
  sorry

end shelves_used_l5_5808


namespace temperature_difference_l5_5990

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

end temperature_difference_l5_5990


namespace x_squared_plus_y_squared_l5_5479

theorem x_squared_plus_y_squared (x y : ℝ) (h1 : (x + y)^2 = 1) (h2 : x * y = -4) : x^2 + y^2 = 9 :=
sorry

end x_squared_plus_y_squared_l5_5479


namespace roots_sum_of_squares_l5_5093

theorem roots_sum_of_squares {r s : ℝ} (h : Polynomial.roots (X^2 - 3*X + 1) = {r, s}) : r^2 + s^2 = 7 :=
by
  sorry

end roots_sum_of_squares_l5_5093


namespace largest_divisor_l5_5362

theorem largest_divisor (n : ℕ) (h1 : 0 < n) (h2 : 450 ∣ n ^ 2) : 30 ∣ n :=
sorry

end largest_divisor_l5_5362


namespace domain_myFunction_l5_5489

noncomputable def myFunction (x : ℝ) : ℝ :=
  (x^3 - 125) / (x + 125)

theorem domain_myFunction :
  {x : ℝ | ∀ y, y = myFunction x → x ≠ -125} = { x : ℝ | x ≠ -125 } := 
by
  sorry

end domain_myFunction_l5_5489


namespace eight_points_on_circle_l5_5540

theorem eight_points_on_circle
  (R : ℝ) (hR : R > 0)
  (points : Fin 8 → (ℝ × ℝ))
  (hpoints : ∀ i : Fin 8, (points i).1 ^ 2 + (points i).2 ^ 2 ≤ R ^ 2) :
  ∃ (i j : Fin 8), i ≠ j ∧ (dist (points i) (points j) < R) :=
sorry

end eight_points_on_circle_l5_5540


namespace total_dresses_l5_5388

theorem total_dresses (D M E : ℕ) (h1 : E = 16) (h2 : M = E / 2) (h3 : D = M + 12) : D + M + E = 44 :=
by
  sorry

end total_dresses_l5_5388


namespace four_numbers_divisible_by_2310_in_4_digit_range_l5_5981

/--
There exist exactly 4 numbers within the range of 1000 to 9999 that are divisible by 2310.
-/
theorem four_numbers_divisible_by_2310_in_4_digit_range :
  ∃ n₁ n₂ n₃ n₄,
    1000 ≤ n₁ ∧ n₁ ≤ 9999 ∧ n₁ % 2310 = 0 ∧
    1000 ≤ n₂ ∧ n₂ ≤ 9999 ∧ n₂ % 2310 = 0 ∧ n₁ < n₂ ∧
    1000 ≤ n₃ ∧ n₃ ≤ 9999 ∧ n₃ % 2310 = 0 ∧ n₂ < n₃ ∧
    1000 ≤ n₄ ∧ n₄ ≤ 9999 ∧ n₄ % 2310 = 0 ∧ n₃ < n₄ ∧
    ∀ n, 1000 ≤ n ∧ n ≤ 9999 ∧ n % 2310 = 0 → (n = n₁ ∨ n = n₂ ∨ n = n₃ ∨ n = n₄) :=
by
  sorry

end four_numbers_divisible_by_2310_in_4_digit_range_l5_5981


namespace car_travel_distance_l5_5443

-- Definitions of conditions
def speed_kmph : ℝ := 27 -- 27 kilometers per hour
def time_sec : ℝ := 50 -- 50 seconds

-- Equivalent in Lean 4 for car moving distance in meters
theorem car_travel_distance : (speed_kmph * 1000 / 3600) * time_sec = 375 := by
  sorry

end car_travel_distance_l5_5443


namespace tan_alpha_add_pi_over_4_l5_5279

open Real

theorem tan_alpha_add_pi_over_4 
  (α : ℝ)
  (h1 : tan α = sqrt 3) : 
  tan (α + π / 4) = -2 - sqrt 3 :=
by
  sorry

end tan_alpha_add_pi_over_4_l5_5279


namespace initial_walnut_trees_l5_5512

theorem initial_walnut_trees (total_trees_after_planting : ℕ) (trees_planted_today : ℕ) (initial_trees : ℕ) : 
  (total_trees_after_planting = 55) → (trees_planted_today = 33) → (initial_trees + trees_planted_today = total_trees_after_planting) → (initial_trees = 22) :=
by
  sorry

end initial_walnut_trees_l5_5512


namespace no_prime_divisible_by_56_l5_5495

-- Define what it means to be a prime number
def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Define what it means for a number to be divisible by another number
def divisible_by (a b : ℕ) : Prop :=
  b ≠ 0 ∧ ∃ k : ℕ, a = b * k

-- The main theorem stating the problem
theorem no_prime_divisible_by_56 : ¬ ∃ p : ℕ, is_prime p ∧ divisible_by p 56 :=
  sorry

end no_prime_divisible_by_56_l5_5495


namespace max_M_min_N_l5_5097

noncomputable def M (x y : ℝ) : ℝ := x / (2 * x + y) + y / (x + 2 * y)
noncomputable def N (x y : ℝ) : ℝ := x / (x + 2 * y) + y / (2 * x + y)

theorem max_M_min_N (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
  (∃ t : ℝ, (∀ x y, 0 < x → 0 < y → M x y ≤ t) ∧ (∀ x y, 0 < x → 0 < y → N x y ≥ t) ∧ t = 2 / 3) :=
sorry

end max_M_min_N_l5_5097


namespace arithmetic_sequence_sum_l5_5015

theorem arithmetic_sequence_sum
  (a_n : ℕ → ℤ)
  (S_n : ℕ → ℤ)
  (n : ℕ)
  (a1 : ℤ)
  (d : ℤ)
  (h1 : a1 = 2)
  (h2 : a_n 5 = a_n 1 + 4 * d)
  (h3 : a_n 3 = a_n 1 + 2 * d)
  (h4 : a_n 5 = 3 * a_n 3) :
  S_n 9 = -54 := 
by  
  sorry

end arithmetic_sequence_sum_l5_5015


namespace retail_price_l5_5216

theorem retail_price (R : ℝ) (wholesale_price : ℝ)
  (discount_rate : ℝ) (profit_rate : ℝ)
  (selling_price : ℝ) :
  wholesale_price = 81 →
  discount_rate = 0.10 →
  profit_rate = 0.20 →
  selling_price = wholesale_price * (1 + profit_rate) →
  selling_price = R * (1 - discount_rate) →
  R = 108 := 
by 
  intros h_wholesale h_discount h_profit h_selling_price h_discounted_selling_price
  sorry

end retail_price_l5_5216


namespace median_product_sum_l5_5736

-- Let's define the lengths of medians and distances from a point P to these medians
variables {s1 s2 s3 d1 d2 d3 : ℝ}

-- Define the conditions
def is_median_lengths (s1 s2 s3 : ℝ) : Prop := 
  ∃ (A B C : ℝ × ℝ), -- vertices of the triangle
    (s1 = ((B.1 - A.1)^2 + (B.2 - A.2)^2) / 2) ∧
    (s2 = ((C.1 - B.1)^2 + (C.2 - B.2)^2) / 2) ∧
    (s3 = ((A.1 - C.1)^2 + (A.2 - C.2)^2) / 2)

def distances_to_medians (d1 d2 d3 : ℝ) : Prop :=
  ∃ (P A B C : ℝ × ℝ), -- point P and vertices of the triangle
    (d1 = dist P ((B.1 + C.1) / 2, (B.2 + C.2) / 2)) ∧
    (d2 = dist P ((A.1 + C.1) / 2, (A.2 + C.2) / 2)) ∧
    (d3 = dist P ((A.1 + B.1) / 2, (A.2 + B.2) / 2))

-- The theorem which we need to prove
theorem median_product_sum (h_medians : is_median_lengths s1 s2 s3) 
  (h_distances : distances_to_medians d1 d2 d3) :
  s1 * d1 + s2 * d2 + s3 * d3 = 0 := sorry

end median_product_sum_l5_5736


namespace A_inter_B_eq_A_union_C_U_B_eq_l5_5140

section
  -- Define the universal set U
  def U : Set ℝ := { x | x^2 - (5 / 2) * x + 1 ≥ 0 }

  -- Define set A
  def A : Set ℝ := { x | |x - 1| > 1 }

  -- Define set B
  def B : Set ℝ := { x | (x + 1) / (x - 2) ≥ 0 }

  -- Define the complement of B in U
  def C_U_B : Set ℝ := U \ B

  -- Theorem for A ∩ B
  theorem A_inter_B_eq : A ∩ B = { x | x ≤ -1 ∨ x > 2 } := sorry

  -- Theorem for A ∪ (C_U_B)
  theorem A_union_C_U_B_eq : A ∪ C_U_B = U := sorry
end

end A_inter_B_eq_A_union_C_U_B_eq_l5_5140


namespace sum_abs_coeffs_expansion_l5_5108

theorem sum_abs_coeffs_expansion (x : ℝ) :
  (|1 - 0 * x| + |1 - 3 * x| + |1 - 3^2 * x^2| + |1 - 3^3 * x^3| + |1 - 3^4 * x^4| + |1 - 3^5 * x^5| = 1024) :=
sorry

end sum_abs_coeffs_expansion_l5_5108


namespace original_number_abc_l5_5761

theorem original_number_abc (a b c : ℕ)
  (h : 100 * a + 10 * b + c = 528)
  (N : ℕ)
  (h1 : N + (100 * a + 10 * b + c) = 222 * (a + b + c))
  (hN : N = 2670) :
  100 * a + 10 * b + c = 528 := by
  sorry

end original_number_abc_l5_5761


namespace juan_speed_l5_5247

-- Statement of given distances and time
def distance : ℕ := 80
def time : ℕ := 8

-- Desired speed in miles per hour
def expected_speed : ℕ := 10

-- Theorem statement: Speed is distance divided by time and should equal 10 miles per hour
theorem juan_speed : distance / time = expected_speed :=
  by
  sorry

end juan_speed_l5_5247


namespace a7_of_expansion_x10_l5_5223

theorem a7_of_expansion_x10 : 
  (∃ (a : ℕ) (a1 : ℕ) (a2 : ℕ) (a3 : ℕ) 
     (a4 : ℕ) (a5 : ℕ) (a6 : ℕ) 
     (a8 : ℕ) (a9 : ℕ) (a10 : ℕ),
     ((x : ℕ) → x^10 = a + a1*(x-1) + a2*(x-1)^2 + a3*(x-1)^3 + 
                      a4*(x-1)^4 + a5*(x-1)^5 + a6*(x-1)^6 + 
                      120*(x-1)^7 + a8*(x-1)^8 + a9*(x-1)^9 + a10*(x-1)^10)) :=
  sorry

end a7_of_expansion_x10_l5_5223


namespace remainder_division_l5_5304

theorem remainder_division (N : ℤ) (R1 : ℤ) (Q2 : ℤ) 
  (h1 : N = 44 * 432 + R1)
  (h2 : N = 38 * Q2 + 8) : 
  R1 = 0 := by
  sorry

end remainder_division_l5_5304


namespace triangle_side_ratio_l5_5776

theorem triangle_side_ratio
  (α β γ : Real)
  (a b c p q r : Real)
  (h1 : (Real.tan α) / (Real.tan β) = p / q)
  (h2 : (Real.tan β) / (Real.tan γ) = q / r)
  (h3 : (Real.tan γ) / (Real.tan α) = r / p) :
  a^2 / b^2 / c^2 = (1/q + 1/r) / (1/r + 1/p) / (1/p + 1/q) := 
sorry

end triangle_side_ratio_l5_5776


namespace ratio_of_larger_to_smaller_l5_5565

theorem ratio_of_larger_to_smaller (S L k : ℕ) 
  (hS : S = 32)
  (h_sum : S + L = 96)
  (h_multiple : L = k * S) : L / S = 2 :=
by
  sorry

end ratio_of_larger_to_smaller_l5_5565


namespace Mark_bill_total_l5_5071

theorem Mark_bill_total
  (original_bill : ℝ)
  (first_late_charge_rate : ℝ)
  (second_late_charge_rate : ℝ)
  (after_first_late_charge : ℝ)
  (final_total : ℝ) :
  original_bill = 500 ∧
  first_late_charge_rate = 0.02 ∧
  second_late_charge_rate = 0.02 ∧
  after_first_late_charge = original_bill * (1 + first_late_charge_rate) ∧
  final_total = after_first_late_charge * (1 + second_late_charge_rate) →
  final_total = 520.20 := by
  sorry

end Mark_bill_total_l5_5071


namespace general_term_formula_no_pos_int_for_S_n_gt_40n_plus_600_exists_pos_int_for_S_n_gt_40n_plus_600_l5_5099

noncomputable def arith_seq (n : ℕ) (d : ℝ) :=
  2 + (n - 1) * d

theorem general_term_formula :
  ∃ d, ∀ n, arith_seq n d = 2 ∨ arith_seq n d = 4 * n - 2 :=
by sorry

theorem no_pos_int_for_S_n_gt_40n_plus_600 :
  ∀ n, (arith_seq n 0) * n ≤ 40 * n + 600 :=
by sorry

theorem exists_pos_int_for_S_n_gt_40n_plus_600 :
  ∃ n, (arith_seq n 4) * n > 40 * n + 600 ∧ n = 31 :=
by sorry

end general_term_formula_no_pos_int_for_S_n_gt_40n_plus_600_exists_pos_int_for_S_n_gt_40n_plus_600_l5_5099


namespace probability_red_or_white_ball_l5_5805

theorem probability_red_or_white_ball :
  let red_balls := 3
  let yellow_balls := 2
  let white_balls := 1
  let total_balls := red_balls + yellow_balls + white_balls
  let favorable_outcomes := red_balls + white_balls
  (favorable_outcomes / total_balls : ℚ) = 2 / 3 := by
  sorry

end probability_red_or_white_ball_l5_5805


namespace necessary_but_not_sufficient_l5_5265

theorem necessary_but_not_sufficient (a : ℝ) : 
  (∃ x : ℝ, x^2 - 2*x + a < 0) → (a < 11) ∧ ¬((a < 11) → (∃ x : ℝ, x^2 - 2*x + a < 0)) :=
by
  -- Sorry to bypass proof below, which is correct as per the problem statement requirements.
  sorry

end necessary_but_not_sufficient_l5_5265


namespace least_positive_integer_greater_than_100_l5_5821

theorem least_positive_integer_greater_than_100 : ∃ n : ℕ, n > 100 ∧ (∀ k ∈ [2, 3, 4, 5, 6, 7, 8, 9, 10], n % k = 1) ∧ n = 2521 :=
by
  sorry

end least_positive_integer_greater_than_100_l5_5821


namespace period_of_repeating_decimal_l5_5628

def is_100_digit_number_with_98_sevens (a : ℕ) : Prop :=
  ∃ (n : ℕ), n = 10^98 ∧ a = 1776 + 1777 * n

theorem period_of_repeating_decimal (a : ℕ) (h : is_100_digit_number_with_98_sevens a) : 
  (1:ℚ) / a == 1 / 99 := 
  sorry

end period_of_repeating_decimal_l5_5628


namespace find_a_plus_b_plus_c_l5_5308

-- Definitions of conditions
def is_vertex (a b c : ℝ) (vertex_x vertex_y : ℝ) := 
  ∀ x : ℝ, vertex_y = (a * (vertex_x ^ 2)) + (b * vertex_x) + c

def contains_point (a b c : ℝ) (x y : ℝ) := 
  y = (a * (x ^ 2)) + (b * x) + c

theorem find_a_plus_b_plus_c
  (a b c : ℝ)
  (h_vertex : is_vertex a b c 3 4)
  (h_symmetry : ∃ h : ℝ, ∀ x : ℝ, a * (x - h) ^ 2 = a * (h - x) ^ 2)
  (h_contains : contains_point a b c 1 0)
  : a + b + c = 0 := 
sorry

end find_a_plus_b_plus_c_l5_5308


namespace safe_dishes_count_l5_5993

theorem safe_dishes_count (total_dishes vegan_dishes vegan_with_nuts : ℕ) 
  (h1 : vegan_dishes = total_dishes / 3) 
  (h2 : vegan_with_nuts = 4) 
  (h3 : vegan_dishes = 6) : vegan_dishes - vegan_with_nuts = 2 :=
by
  sorry

end safe_dishes_count_l5_5993


namespace sales_tax_difference_l5_5377

theorem sales_tax_difference :
  let price : ℝ := 50
  let tax_rate1 : ℝ := 0.075
  let tax_rate2 : ℝ := 0.07
  (price * tax_rate1) - (price * tax_rate2) = 0.25 := by
  sorry

end sales_tax_difference_l5_5377


namespace smallest_N_exists_l5_5198

def find_smallest_N (N : ℕ) : Prop :=
  ∃ (c1 c2 c3 c4 c5 c6 : ℕ),
  (N ≠ 0) ∧ 
  (c1 = 6 * c2 - 1) ∧ 
  (N + c2 = 6 * c3 - 2) ∧ 
  (2 * N + c3 = 6 * c4 - 3) ∧ 
  (3 * N + c4 = 6 * c5 - 4) ∧ 
  (4 * N + c5 = 6 * c6 - 5) ∧ 
  (5 * N + c6 = 6 * c1)

theorem smallest_N_exists : ∃ (N : ℕ), find_smallest_N N :=
sorry

end smallest_N_exists_l5_5198


namespace total_football_games_l5_5487

theorem total_football_games (months : ℕ) (games_per_month : ℕ) (season_length : months = 17 ∧ games_per_month = 19) :
  (months * games_per_month) = 323 :=
by
  sorry

end total_football_games_l5_5487


namespace total_bottles_ordered_in_april_and_may_is_1000_l5_5120

-- Define the conditions
def casesInApril : Nat := 20
def casesInMay : Nat := 30
def bottlesPerCase : Nat := 20

-- The total number of bottles ordered in April and May
def totalBottlesOrdered : Nat := (casesInApril + casesInMay) * bottlesPerCase

-- The main statement to be proved
theorem total_bottles_ordered_in_april_and_may_is_1000 :
  totalBottlesOrdered = 1000 :=
sorry

end total_bottles_ordered_in_april_and_may_is_1000_l5_5120


namespace hyunwoo_family_saving_l5_5534

def daily_water_usage : ℝ := 215
def saving_factor : ℝ := 0.32

theorem hyunwoo_family_saving:
  daily_water_usage * saving_factor = 68.8 := by
  sorry

end hyunwoo_family_saving_l5_5534


namespace carpet_interior_length_l5_5365

/--
A carpet is designed using three different colors, forming three nested rectangles with different areas in an arithmetic progression. 
The innermost rectangle has a width of two feet. Each of the two colored borders is 2 feet wide on all sides.
Determine the length in feet of the innermost rectangle. 
-/
theorem carpet_interior_length 
  (x : ℕ) -- length of the innermost rectangle
  (hp : ∀ (a b c : ℕ), a = 2 * x ∧ b = (4 * x + 24) ∧ c = (4 * x + 56) → (b - a) = (c - b)) 
  : x = 4 :=
by
  sorry

end carpet_interior_length_l5_5365


namespace team_card_sending_l5_5721

theorem team_card_sending (x : ℕ) (h : x * (x - 1) = 56) : x * (x - 1) = 56 := 
by 
  sorry

end team_card_sending_l5_5721


namespace number_of_two_point_safeties_l5_5394

variables (f g s : ℕ)

theorem number_of_two_point_safeties (h1 : 4 * f = 6 * g) 
                                    (h2 : s = g + 2) 
                                    (h3 : 4 * f + 3 * g + 2 * s = 50) : 
                                    s = 6 := 
by sorry

end number_of_two_point_safeties_l5_5394


namespace cube_root_of_8_is_2_l5_5916

theorem cube_root_of_8_is_2 : ∃ x : ℝ, x ^ 3 = 8 ∧ x = 2 :=
by
  have h : (2 : ℝ) ^ 3 = 8 := by norm_num
  exact ⟨2, h, rfl⟩

end cube_root_of_8_is_2_l5_5916


namespace expected_value_proof_l5_5930

noncomputable def expected_value_xi : ℚ :=
  let p_xi_2 : ℚ := 3/5
  let p_xi_3 : ℚ := 3/10
  let p_xi_4 : ℚ := 1/10
  2 * p_xi_2 + 3 * p_xi_3 + 4 * p_xi_4

theorem expected_value_proof :
  expected_value_xi = 5/2 :=
by
  sorry

end expected_value_proof_l5_5930


namespace find_angle_degree_l5_5832

-- Define the angle
variable {x : ℝ}

-- Define the conditions
def complement (x : ℝ) : ℝ := 90 - x
def supplement (x : ℝ) : ℝ := 180 - x

-- Define the given condition
def condition (x : ℝ) : Prop := complement x = (1/3) * (supplement x)

-- The theorem statement
theorem find_angle_degree (x : ℝ) (h : condition x) : x = 45 :=
by
  sorry

end find_angle_degree_l5_5832


namespace range_of_a_for_zero_l5_5538

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x - 2 * x + a

theorem range_of_a_for_zero (a : ℝ) : a ≤ 2 * Real.log 2 - 2 → ∃ x : ℝ, f a x = 0 := by
  sorry

end range_of_a_for_zero_l5_5538


namespace percentage_of_8thgraders_correct_l5_5461

def total_students_oakwood : ℕ := 150
def total_students_pinecrest : ℕ := 250

def percent_8thgraders_oakwood : ℕ := 60
def percent_8thgraders_pinecrest : ℕ := 55

def number_of_8thgraders_oakwood : ℚ := (percent_8thgraders_oakwood * total_students_oakwood) / 100
def number_of_8thgraders_pinecrest : ℚ := (percent_8thgraders_pinecrest * total_students_pinecrest) / 100

def total_number_of_8thgraders : ℚ := number_of_8thgraders_oakwood + number_of_8thgraders_pinecrest
def total_number_of_students : ℕ := total_students_oakwood + total_students_pinecrest

def percent_8thgraders_combined : ℚ := (total_number_of_8thgraders / total_number_of_students) * 100

theorem percentage_of_8thgraders_correct : percent_8thgraders_combined = 57 := 
by
  sorry

end percentage_of_8thgraders_correct_l5_5461


namespace sin_equations_solution_l5_5933

theorem sin_equations_solution {k : ℤ} (hk : k ≤ 1 ∨ k ≥ 5) : 
  (∃ x : ℝ, 2 * x = π * k ∧ x = (π * k) / 2) ∨ x = 7 * π / 4 :=
by
  sorry

end sin_equations_solution_l5_5933


namespace johann_ate_ten_oranges_l5_5681

variable (x : ℕ)
variable (y : ℕ)

def johann_initial_oranges := 60

def johann_remaining_after_eating := johann_initial_oranges - x

def johann_remaining_after_theft := (johann_remaining_after_eating / 2)

def johann_remaining_after_return := johann_remaining_after_theft + 5

theorem johann_ate_ten_oranges (h : johann_remaining_after_return = 30) : x = 10 :=
by
  sorry

end johann_ate_ten_oranges_l5_5681


namespace fraction_of_income_from_tips_l5_5756

theorem fraction_of_income_from_tips 
  (salary tips : ℝ)
  (h1 : tips = (7/4) * salary) 
  (total_income : ℝ)
  (h2 : total_income = salary + tips) :
  (tips / total_income) = (7 / 11) :=
by
  sorry

end fraction_of_income_from_tips_l5_5756


namespace cost_of_monogramming_each_backpack_l5_5176

def number_of_backpacks : ℕ := 5
def original_price_per_backpack : ℝ := 20.00
def discount_rate : ℝ := 0.20
def total_cost : ℝ := 140.00

theorem cost_of_monogramming_each_backpack : 
  (total_cost - (number_of_backpacks * (original_price_per_backpack * (1 - discount_rate)))) / number_of_backpacks = 12.00 :=
by
  sorry 

end cost_of_monogramming_each_backpack_l5_5176


namespace savings_proof_l5_5774

variable (income expenditure savings : ℕ)

def ratio_income_expenditure (i e : ℕ) := i / 10 = e / 7

theorem savings_proof (h : ratio_income_expenditure income expenditure) (hincome : income = 10000) :
  savings = income - expenditure → savings = 3000 :=
by
  sorry

end savings_proof_l5_5774


namespace find_x_positive_integers_l5_5828

theorem find_x_positive_integers (a b c x : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a + b + c = x * a * b * c) → (x = 1 ∧ a = 1 ∧ b = 2 ∧ c = 3) ∨
  (x = 2 ∧ a = 1 ∧ b = 1 ∧ c = 2) ∨
  (x = 3 ∧ a = 1 ∧ b = 1 ∧ c = 1) :=
  sorry

end find_x_positive_integers_l5_5828


namespace soda_preference_count_eq_243_l5_5117

def total_respondents : ℕ := 540
def soda_angle : ℕ := 162
def total_circle_angle : ℕ := 360

theorem soda_preference_count_eq_243 :
  (total_respondents * soda_angle / total_circle_angle) = 243 := 
by 
  sorry

end soda_preference_count_eq_243_l5_5117


namespace domain_f_a_5_abs_inequality_ab_l5_5639

-- Definition for the domain of f(x) when a=5
def domain_of_f_a_5 (x : ℝ) : Prop := |x + 1| + |x + 2| - 5 ≥ 0

-- The theorem to find the domain A of the function f(x) when a=5.
theorem domain_f_a_5 (x : ℝ) : domain_of_f_a_5 x ↔ (x ≤ -4 ∨ x ≥ 1) :=
by
  sorry

-- Theorem to prove the inequality for a, b ∈ (-1, 1)
theorem abs_inequality_ab (a b : ℝ) (ha : -1 < a ∧ a < 1) (hb : -1 < b ∧ b < 1) :
  |a + b| / 2 < |1 + a * b / 4| :=
by
  sorry

end domain_f_a_5_abs_inequality_ab_l5_5639


namespace max_integer_is_110003_l5_5866

def greatest_integer : Prop :=
  let a := 100004
  let b := 110003
  let c := 102002
  let d := 100301
  let e := 100041
  b > a ∧ b > c ∧ b > d ∧ b > e

theorem max_integer_is_110003 : greatest_integer :=
by
  sorry

end max_integer_is_110003_l5_5866


namespace liam_annual_income_l5_5260

theorem liam_annual_income (q : ℝ) (I : ℝ) (T : ℝ) 
  (h1 : T = (q + 0.5) * 0.01 * I) 
  (h2 : I > 50000) 
  (h3 : T = 0.01 * q * 30000 + 0.01 * (q + 3) * 20000 + 0.01 * (q + 5) * (I - 50000)) : 
  I = 56000 :=
by
  sorry

end liam_annual_income_l5_5260


namespace polygon_perimeter_l5_5367

-- Define a regular polygon with side length 7 units
def side_length : ℝ := 7

-- Define the exterior angle of the polygon in degrees
def exterior_angle : ℝ := 90

-- The statement to prove that the perimeter of the polygon is 28 units
theorem polygon_perimeter : ∃ (P : ℝ), P = 28 ∧ 
  (∃ n : ℕ, n = (360 / exterior_angle) ∧ P = n * side_length) := 
sorry

end polygon_perimeter_l5_5367


namespace sufficient_remedy_l5_5579

-- Definitions based on conditions
def aspirin_relieves_headache : Prop := true
def aspirin_relieves_knee_rheumatism : Prop := true
def aspirin_causes_heart_pain : Prop := true
def aspirin_causes_stomach_pain : Prop := true

def homeopathic_relieves_heart_issues : Prop := true
def homeopathic_relieves_stomach_issues : Prop := true
def homeopathic_causes_hip_rheumatism : Prop := true

def antibiotics_cure_migraines : Prop := true
def antibiotics_cure_heart_pain : Prop := true
def antibiotics_cause_stomach_pain : Prop := true
def antibiotics_cause_knee_pain : Prop := true
def antibiotics_cause_itching : Prop := true

def cortisone_relieves_itching : Prop := true
def cortisone_relieves_knee_rheumatism : Prop := true
def cortisone_exacerbates_hip_rheumatism : Prop := true

def warm_compress_relieves_itching : Prop := true
def warm_compress_relieves_stomach_pain : Prop := true

def severe_headache_morning : Prop := true
def impaired_ability_to_think : Prop := severe_headache_morning

-- Statement of the proof problem
theorem sufficient_remedy :
  (aspirin_relieves_headache ∧ antibiotics_cure_heart_pain ∧ warm_compress_relieves_itching ∧ warm_compress_relieves_stomach_pain) →
  (impaired_ability_to_think → true) :=
by
  sorry

end sufficient_remedy_l5_5579


namespace f_neg2_eq_neg1_l5_5995

noncomputable def f (x : ℝ) : ℝ := x^2 + 2*x - 1

theorem f_neg2_eq_neg1 : f (-2) = -1 := by
  sorry

end f_neg2_eq_neg1_l5_5995


namespace find_k_l5_5641

theorem find_k :
  ∀ (k : ℤ),
    (∃ a1 a2 a3 : ℤ,
        a1 = 49 + k ∧
        a2 = 225 + k ∧
        a3 = 484 + k ∧
        2 * a2 = a1 + a3) →
    k = 324 :=
by
  sorry

end find_k_l5_5641


namespace incorrect_ac_bc_impl_a_b_l5_5412

theorem incorrect_ac_bc_impl_a_b : ∀ (a b c : ℝ), (ac = bc → a = b) ↔ c ≠ 0 :=
by sorry

end incorrect_ac_bc_impl_a_b_l5_5412


namespace evenness_oddness_of_f_min_value_of_f_l5_5307

noncomputable def f (a x : ℝ) : ℝ :=
  x^2 + |x - a| + 1

def is_even (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = -f (-x)

theorem evenness_oddness_of_f (a : ℝ) :
  (is_even (f a) ↔ a = 0) ∧ (a ≠ 0 → ¬ is_even (f a) ∧ ¬ is_odd (f a)) :=
by
  sorry

theorem min_value_of_f (a x : ℝ) (h : x ≥ a) :
  (a ≤ -1 / 2 → f a x = 3 / 4 - a) ∧ (a > -1 / 2 → f a x = a^2 + 1) :=
by
  sorry

end evenness_oddness_of_f_min_value_of_f_l5_5307


namespace cards_value_1_count_l5_5418

/-- There are 4 different suits in a deck of cards containing a total of 52 cards.
  Each suit has 13 cards numbered from 1 to 13.
  Feifei draws 2 hearts, 3 spades, 4 diamonds, and 5 clubs.
  The sum of the face values of these 14 cards is exactly 35.
  Prove that 4 of these cards have a face value of 1. -/
theorem cards_value_1_count :
  ∃ (hearts spades diamonds clubs : List ℕ),
  hearts.length = 2 ∧ spades.length = 3 ∧ diamonds.length = 4 ∧ clubs.length = 5 ∧
  (∀ v, v ∈ hearts → v ∈ List.range 13) ∧ 
  (∀ v, v ∈ spades → v ∈ List.range 13) ∧
  (∀ v, v ∈ diamonds → v ∈ List.range 13) ∧
  (∀ v, v ∈ clubs → v ∈ List.range 13) ∧
  (hearts.sum + spades.sum + diamonds.sum + clubs.sum = 35) ∧
  ((hearts ++ spades ++ diamonds ++ clubs).count 1 = 4) := sorry

end cards_value_1_count_l5_5418


namespace sequence_general_formula_l5_5646

theorem sequence_general_formula (n : ℕ) (h : n > 0) :
  ∃ (a : ℕ → ℚ), a 1 = 1 ∧ (∀ n, a (n + 1) = a n / (3 * a n + 1)) ∧ a n = 1 / (3 * n - 2) :=
by sorry

end sequence_general_formula_l5_5646


namespace no_geometric_progression_11_12_13_l5_5510

theorem no_geometric_progression_11_12_13 :
  ∀ (b1 : ℝ) (q : ℝ) (k l n : ℕ), 
  (b1 * q ^ (k - 1) = 11) → 
  (b1 * q ^ (l - 1) = 12) → 
  (b1 * q ^ (n - 1) = 13) → 
  False :=
by
  intros b1 q k l n hk hl hn
  sorry

end no_geometric_progression_11_12_13_l5_5510


namespace solve_inequality_l5_5811

theorem solve_inequality (x : ℝ) :
  (4 * x^4 + x^2 + 4 * x - 5 * x^2 * |x + 2| + 4) ≥ 0 ↔ 
  x ∈ Set.Iic (-1) ∪ Set.Icc ((1 - Real.sqrt 33) / 8) ((1 + Real.sqrt 33) / 8) ∪ Set.Ici 2 :=
by
  sorry

end solve_inequality_l5_5811


namespace zero_in_interval_l5_5801

noncomputable def f (x : ℝ) : ℝ := 2 * x - 8 + Real.logb 3 x

theorem zero_in_interval : 
  (0 < 3) ∧ (3 < 4) → (f 3 < 0) ∧ (f 4 > 0) → ∃ x, 3 < x ∧ x < 4 ∧ f x = 0 :=
by
  intro h1 h2
  obtain ⟨h3, h4⟩ := h2
  sorry

end zero_in_interval_l5_5801


namespace GMAT_scores_ratio_l5_5750

variables (u v w : ℝ)

theorem GMAT_scores_ratio
  (h1 : u - w = (u + v + w) / 3)
  (h2 : u - v = 2 * (v - w))
  : v / u = 4 / 7 :=
sorry

end GMAT_scores_ratio_l5_5750


namespace number_of_m_l5_5162

theorem number_of_m (k : ℕ) : 
  (∀ m a b : ℤ, 
      (a ≠ 0 ∧ b ≠ 0) ∧ 
      (a + b = m) ∧ 
      (a * b = m + 2006) → k = 5) :=
sorry

end number_of_m_l5_5162


namespace hyperbola_focal_distance_and_asymptotes_l5_5945

-- Define the hyperbola
def hyperbola (y x : ℝ) : Prop := (y^2 / 4) - (x^2 / 3) = 1

-- Prove the properties
theorem hyperbola_focal_distance_and_asymptotes :
  (∀ y x : ℝ, hyperbola y x → ∃ c : ℝ, c = 2 * Real.sqrt 7)
  ∧
  (∀ y x : ℝ, hyperbola y x → (y = (2 * Real.sqrt 3 / 3) * x ∨ y = -(2 * Real.sqrt 3 / 3) * x)) :=
by
  sorry

end hyperbola_focal_distance_and_asymptotes_l5_5945


namespace intersection_of_lines_l5_5442

theorem intersection_of_lines : 
  (∃ x y : ℚ, y = -3 * x + 1 ∧ y = 5 * x + 4) ↔ 
  (∃ x y : ℚ, x = -3 / 8 ∧ y = 17 / 8) :=
by
  sorry

end intersection_of_lines_l5_5442


namespace abs_eq_five_l5_5518

theorem abs_eq_five (x : ℝ) : |x| = 5 → (x = 5 ∨ x = -5) :=
by
  intro h
  sorry

end abs_eq_five_l5_5518


namespace car_trip_eq_560_miles_l5_5618

noncomputable def car_trip_length (v L : ℝ) :=
  -- Conditions from the problem
  -- 1. Car travels for 2 hours before the delay
  let pre_delay_time := 2
  -- 2. Delay time is 1 hour
  let delay_time := 1
  -- 3. Post-delay speed is 2/3 of the initial speed
  let post_delay_speed := (2 / 3) * v
  -- 4. Car arrives 4 hours late under initial scenario:
  let late_4_hours_time := 2 + 1 + (3 * (L - 2 * v)) / (2 * v)
  -- Expected travel time without any delays is 2 + (L / v)
  -- Difference indicates delay of 4 hours
  let without_delay_time := (L / v)
  let time_diff_late_4 := (late_4_hours_time - without_delay_time = 4)
  -- 5. Delay 120 miles farther, car arrives 3 hours late
  let delay_120_miles_farther := 120
  let late_3_hours_time := 2 + delay_120_miles_farther / v + 1 + (3 * (L - 2 * v - 120)) / (2 * v)
  let time_diff_late_3 := (late_3_hours_time - without_delay_time = 3)

  -- Combining conditions to solve for L
  -- Goal: Prove L = 560
  L = 560 -> time_diff_late_4 ∧ time_diff_late_3

theorem car_trip_eq_560_miles (v : ℝ) : ∃ (L : ℝ), car_trip_length v L := 
by 
  sorry

end car_trip_eq_560_miles_l5_5618


namespace total_teachers_correct_l5_5240

noncomputable def total_teachers (x : ℕ) : ℕ := 26 + 104 + x

theorem total_teachers_correct
    (x : ℕ)
    (h : (x : ℝ) / (26 + 104 + x) = 16 / 56) :
  total_teachers x = 182 :=
sorry

end total_teachers_correct_l5_5240


namespace number_of_smaller_cubes_l5_5156

theorem number_of_smaller_cubes 
  (volume_large_cube : ℝ)
  (volume_small_cube : ℝ)
  (surface_area_difference : ℝ)
  (h1 : volume_large_cube = 216)
  (h2 : volume_small_cube = 1)
  (h3 : surface_area_difference = 1080) :
  ∃ n : ℕ, n * 6 - 6 * (volume_large_cube^(1/3))^2 = surface_area_difference ∧ n = 216 :=
by
  sorry

end number_of_smaller_cubes_l5_5156


namespace total_spent_after_discount_and_tax_l5_5085

-- Define prices for each item
def price_bracelet := 4
def price_keychain := 5
def price_coloring_book := 3
def price_sticker := 1
def price_toy_car := 6

-- Define discounts and tax rates
def discount_bracelet := 0.10
def sales_tax := 0.05

-- Define the quantity of each item purchased by Paula, Olive, and Nathan
def quantity_paula_bracelets := 3
def quantity_paula_keychains := 2
def quantity_paula_coloring_books := 1
def quantity_paula_stickers := 4

def quantity_olive_coloring_books := 1
def quantity_olive_bracelets := 2
def quantity_olive_toy_cars := 1
def quantity_olive_stickers := 3

def quantity_nathan_toy_cars := 4
def quantity_nathan_stickers := 5
def quantity_nathan_keychains := 1

-- Function to calculate total cost before discount and tax
def total_cost_before_discount_and_tax (bracelets keychains coloring_books stickers toy_cars : Nat) : Float :=
  Float.ofNat (bracelets * price_bracelet) +
  Float.ofNat (keychains * price_keychain) +
  Float.ofNat (coloring_books * price_coloring_book) +
  Float.ofNat (stickers * price_sticker) +
  Float.ofNat (toy_cars * price_toy_car)

-- Function to calculate discount on bracelets
def bracelet_discount (bracelets : Nat) : Float :=
  Float.ofNat (bracelets * price_bracelet) * discount_bracelet

-- Function to calculate total cost after discount and before tax
def total_cost_after_discount (total_cost discount : Float) : Float :=
  total_cost - discount

-- Function to calculate total cost after tax
def total_cost_after_tax (total_cost : Float) (tax_rate : Float) : Float :=
  total_cost * (1 + tax_rate)

-- Proof statement (no proof provided, only the statement)
theorem total_spent_after_discount_and_tax : 
  total_cost_after_tax (
    total_cost_after_discount
      (total_cost_before_discount_and_tax quantity_paula_bracelets quantity_paula_keychains quantity_paula_coloring_books quantity_paula_stickers 0)
      (bracelet_discount quantity_paula_bracelets)
    +
    total_cost_after_discount
      (total_cost_before_discount_and_tax quantity_olive_bracelets 0 quantity_olive_coloring_books quantity_olive_stickers quantity_olive_toy_cars)
      (bracelet_discount quantity_olive_bracelets)
    +
    total_cost_before_discount_and_tax 0 quantity_nathan_keychains 0 quantity_nathan_stickers quantity_nathan_toy_cars
  ) sales_tax = 85.05 := 
sorry

end total_spent_after_discount_and_tax_l5_5085


namespace probability_of_region_l5_5943

theorem probability_of_region :
  let area_rect := (1000: ℝ) * 1500
  let area_polygon := 500000
  let prob := area_polygon / area_rect
  prob = (1 / 3) := sorry

end probability_of_region_l5_5943


namespace value_of_m_l5_5166

theorem value_of_m (m : ℝ) : (3 = 2 * m + 1) → m = 1 :=
by
  intro h
  -- skipped proof due to requirement
  sorry

end value_of_m_l5_5166


namespace profit_share_difference_correct_l5_5613

noncomputable def profit_share_difference (a_capital b_capital c_capital b_profit : ℕ) : ℕ :=
  let total_parts := 4 + 5 + 6
  let part_size := b_profit / 5
  let a_profit := 4 * part_size
  let c_profit := 6 * part_size
  c_profit - a_profit

theorem profit_share_difference_correct :
  profit_share_difference 8000 10000 12000 1600 = 640 :=
by
  sorry

end profit_share_difference_correct_l5_5613


namespace total_cost_correct_l5_5062

-- Define the conditions
def uber_cost : ℤ := 22
def lyft_additional_cost : ℤ := 3
def taxi_additional_cost : ℤ := 4
def tip_percentage : ℚ := 0.20

-- Define the variables for cost of Lyft and Taxi based on the problem
def lyft_cost : ℤ := uber_cost - lyft_additional_cost
def taxi_cost : ℤ := lyft_cost - taxi_additional_cost

-- Calculate the tip
def tip : ℚ := taxi_cost * tip_percentage

-- Final total cost including the tip
def total_cost : ℚ := taxi_cost + tip

-- The theorem to prove
theorem total_cost_correct :
  total_cost = 18 := by
  sorry

end total_cost_correct_l5_5062


namespace solve_problem_1_solve_problem_2_l5_5912

-- Problem statement 1: Prove that the solutions to x(x-2) = x-2 are x = 1 and x = 2.
theorem solve_problem_1 (x : ℝ) : (x * (x - 2) = x - 2) ↔ (x = 1 ∨ x = 2) :=
  sorry

-- Problem statement 2: Prove that the solutions to 2x^2 + 3x - 5 = 0 are x = 1 and x = -5/2.
theorem solve_problem_2 (x : ℝ) : (2 * x^2 + 3 * x - 5 = 0) ↔ (x = 1 ∨ x = -5 / 2) :=
  sorry

end solve_problem_1_solve_problem_2_l5_5912


namespace lucas_investment_l5_5904

noncomputable def investment_amount (y : ℝ) : ℝ := 1500 - y

theorem lucas_investment :
  ∃ y : ℝ, (y * 1.04 + (investment_amount y) * 1.06 = 1584.50) ∧ y = 275 :=
by
  sorry

end lucas_investment_l5_5904


namespace intersection_M_N_l5_5157

def M : Set ℝ := {x | (x - 1) * (x - 4) = 0}
def N : Set ℝ := {x | (x + 1) * (x - 3) < 0}

theorem intersection_M_N :
  M ∩ N = {1} :=
sorry

end intersection_M_N_l5_5157


namespace total_presents_l5_5047

variables (ChristmasPresents BirthdayPresents EasterPresents HalloweenPresents : ℕ)

-- Given conditions
def condition1 : ChristmasPresents = 60 := sorry
def condition2 : BirthdayPresents = 3 * EasterPresents := sorry
def condition3 : EasterPresents = (ChristmasPresents / 2) - 10 := sorry
def condition4 : HalloweenPresents = BirthdayPresents - EasterPresents := sorry

-- Proof statement
theorem total_presents (h1 : ChristmasPresents = 60)
    (h2 : BirthdayPresents = 3 * EasterPresents)
    (h3 : EasterPresents = (ChristmasPresents / 2) - 10)
    (h4 : HalloweenPresents = BirthdayPresents - EasterPresents) :
    ChristmasPresents + BirthdayPresents + EasterPresents + HalloweenPresents = 180 :=
sorry

end total_presents_l5_5047


namespace sum_of_powers_eight_l5_5614

variable {a b : ℝ}

theorem sum_of_powers_eight :
  a + b = 1 → 
  a^2 + b^2 = 3 → 
  a^3 + b^3 = 4 → 
  a^4 + b^4 = 7 → 
  a^5 + b^5 = 11 → 
  a^8 + b^8 = 47 := 
by
  intros h₁ h₂ h₃ h₄ h₅
  -- Proof to be filled in
  sorry

end sum_of_powers_eight_l5_5614


namespace tangent_slope_at_pi_over_four_l5_5638

theorem tangent_slope_at_pi_over_four :
  deriv (fun x => Real.tan x) (Real.pi / 4) = 2 :=
sorry

end tangent_slope_at_pi_over_four_l5_5638


namespace sinA_value_triangle_area_l5_5131

-- Definitions of the given variables
variables (A B C : ℝ)
variables (a b c : ℝ)
variables (sinA sinC cosC : ℝ)

-- Given conditions
axiom h_c : c = Real.sqrt 2
axiom h_a : a = 1
axiom h_cosC : cosC = 3 / 4
axiom h_sinC : sinC = Real.sqrt 7 / 4
axiom h_b : b = 2

-- Question 1: Prove sin A = sqrt 14 / 8
theorem sinA_value : sinA = Real.sqrt 14 / 8 :=
sorry

-- Question 2: Prove the area of triangle ABC is sqrt 7 / 4
theorem triangle_area : 1/2 * a * b * sinC = Real.sqrt 7 / 4 :=
sorry

end sinA_value_triangle_area_l5_5131


namespace floor_x_floor_x_eq_20_l5_5809

theorem floor_x_floor_x_eq_20 (x : ℝ) : ⌊x * ⌊x⌋⌋ = 20 ↔ 5 ≤ x ∧ x < 5.25 := 
sorry

end floor_x_floor_x_eq_20_l5_5809


namespace rocket_travel_time_l5_5203

/-- The rocket's distance formula as an arithmetic series sum.
    We need to prove that the rocket reaches 240 km after 15 seconds
    given the conditions in the problem. -/
theorem rocket_travel_time :
  ∃ n : ℕ, (2 * n + (n * (n - 1))) / 2 = 240 ∧ n = 15 :=
by
  sorry

end rocket_travel_time_l5_5203


namespace greater_number_is_33_l5_5035

theorem greater_number_is_33 (A B : ℕ) (hcf_11 : Nat.gcd A B = 11) (product_363 : A * B = 363) :
  max A B = 33 :=
by
  sorry

end greater_number_is_33_l5_5035


namespace ratio_of_ages_l5_5673

theorem ratio_of_ages (S : ℕ) (M : ℕ) (h1 : S = 18) (h2 : M = S + 20) :
  (M + 2) / (S + 2) = 2 :=
by
  sorry

end ratio_of_ages_l5_5673


namespace integer_div_product_l5_5096

theorem integer_div_product (n : ℤ) : ∃ (k : ℤ), n * (n + 1) * (n + 2) = 6 * k := by
  sorry

end integer_div_product_l5_5096


namespace temperature_difference_l5_5573

-- Define variables for the highest and lowest temperatures.
def highest_temp : ℤ := 18
def lowest_temp : ℤ := -2

-- Define the statement for the maximum temperature difference.
theorem temperature_difference : 
  highest_temp - lowest_temp = 20 := 
by 
  sorry

end temperature_difference_l5_5573


namespace density_change_l5_5083

theorem density_change (V : ℝ) (Δa : ℝ) (decrease_percent : ℝ) (initial_volume : V = 27) (edge_increase : Δa = 0.9) : 
    decrease_percent = 8 := 
by 
  sorry

end density_change_l5_5083


namespace inequality_proof_l5_5026

theorem inequality_proof
  (x y z : ℝ)
  (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (h : Real.sqrt x + Real.sqrt y + Real.sqrt z = 1) :
  (x^2 + y * z) / (Real.sqrt (2 * x^2 * (y + z))) + 
  (y^2 + z * x) / (Real.sqrt (2 * y^2 * (z + x))) + 
  (z^2 + x * y) / (Real.sqrt (2 * z^2 * (x + y))) ≥ 1 := 
sorry

end inequality_proof_l5_5026


namespace green_disks_more_than_blue_l5_5636

theorem green_disks_more_than_blue (total_disks : ℕ) (b y g : ℕ) (h1 : total_disks = 108)
  (h2 : b / y = 3 / 7) (h3 : b / g = 3 / 8) : g - b = 30 :=
by
  sorry

end green_disks_more_than_blue_l5_5636


namespace coopers_age_l5_5413

theorem coopers_age (C D M E : ℝ) 
  (h1 : D = 2 * C) 
  (h2 : M = 2 * C + 1) 
  (h3 : E = 3 * C)
  (h4 : C + D + M + E = 62) : 
  C = 61 / 8 := 
by 
  sorry

end coopers_age_l5_5413


namespace hyperbola_eccentricity_asymptotic_lines_l5_5072

-- Define the conditions and the proof goal:

theorem hyperbola_eccentricity_asymptotic_lines {a b c e : ℝ} 
  (h_asym : ∀ x y : ℝ, (y = x ∨ y = -x) ↔ (a = b)) 
  (h_c : c = Real.sqrt (a ^ 2 + b ^ 2))
  (h_e : e = c / a) : e = Real.sqrt 2 := sorry

end hyperbola_eccentricity_asymptotic_lines_l5_5072


namespace simplify_expr1_simplify_expr2_l5_5326

-- Expression simplification proof statement 1
theorem simplify_expr1 (m n : ℤ) : 
  (5 * m + 3 * n - 7 * m - n) = (-2 * m + 2 * n) :=
sorry

-- Expression simplification proof statement 2
theorem simplify_expr2 (x : ℤ) : 
  (2 * x^2 - (3 * x - 2 * (x^2 - x + 3) + 2 * x^2)) = (2 * x^2 - 5 * x + 6) :=
sorry

end simplify_expr1_simplify_expr2_l5_5326


namespace line_perp_to_plane_imp_perp_to_line_l5_5444

def Line := Type
def Plane := Type

variables (m n : Line) (α : Plane)

def is_parallel (l : Line) (p : Plane) : Prop := sorry
def is_perpendicular (l1 l2 : Line) : Prop := sorry
def is_contained (l : Line) (p : Plane) : Prop := sorry

theorem line_perp_to_plane_imp_perp_to_line :
  (is_perpendicular m α) ∧ (is_contained n α) → (is_perpendicular m n) :=
sorry

end line_perp_to_plane_imp_perp_to_line_l5_5444


namespace train_length_l5_5454

theorem train_length (V L : ℝ) 
  (h1 : L = V * 18) 
  (h2 : L + 600.0000000000001 = V * 54) : 
  L = 300.00000000000005 :=
by 
  sorry

end train_length_l5_5454


namespace points_A_B_D_collinear_l5_5360

variable (a b : ℝ)

theorem points_A_B_D_collinear
  (AB : ℝ × ℝ := (a, 5 * b))
  (BC : ℝ × ℝ := (-2 * a, 8 * b))
  (CD : ℝ × ℝ := (3 * a, -3 * b)) :
  AB = (BC.1 + CD.1, BC.2 + CD.2) := 
by
  sorry

end points_A_B_D_collinear_l5_5360


namespace maximum_distance_point_to_line_l5_5012

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x + 1)^2 + (y - 2)^2 = 1

-- Define the line l
def line_l (m x y : ℝ) : Prop := (m - 1) * x + m * y + 2 = 0

-- Statement of the problem to prove
theorem maximum_distance_point_to_line :
  ∀ (x y m : ℝ), circle_C x y → ∃ P : ℝ, line_l m x y → P = 6 :=
by 
  sorry

end maximum_distance_point_to_line_l5_5012


namespace find_fractions_l5_5125

-- Define the numerators and denominators
def p1 := 75
def p2 := 70
def q1 := 34
def q2 := 51

-- Define the fractions
def frac1 := p1 / q1
def frac2 := p1 / q2

-- Define the greatest common divisor (gcd) condition
def gcd_condition := Nat.gcd p1 p2 = p1 - p2

-- Define the least common multiple (lcm) condition
def lcm_condition := Nat.lcm p1 p2 = 1050

-- Define the difference condition
def difference_condition := (frac1 - frac2) = (5 / 6)

-- Lean proof statement
theorem find_fractions :
  gcd_condition ∧ lcm_condition ∧ difference_condition :=
by
  sorry

end find_fractions_l5_5125


namespace part_one_part_two_l5_5482

def f (x a : ℝ) : ℝ :=
  x^2 + a * (abs x) + x 

theorem part_one (x1 x2 a : ℝ) (h1 : 0 < x1) (h2 : 0 < x2) :
  (1 / 2) * (f x1 a + f x2 a) ≥ f ((x1 + x2) / 2) a :=
sorry

theorem part_two (a : ℝ) (ha : 0 ≤ a) (x1 x2 : ℝ) :
  (1 / 2) * (f x1 a + f x2 a) ≥ f ((x1 + x2) / 2) a :=
sorry

end part_one_part_two_l5_5482


namespace trailing_zeros_in_15_factorial_base_15_are_3_l5_5134

/--
Compute the number of trailing zeros in \( 15! \) when expressed in base 15.
-/
def compute_trailing_zeros_in_factorial_base_15 : ℕ :=
  let num_factors_3 := (15 / 3) + (15 / 9)
  let num_factors_5 := (15 / 5)
  min num_factors_3 num_factors_5

theorem trailing_zeros_in_15_factorial_base_15_are_3 :
  compute_trailing_zeros_in_factorial_base_15 = 3 :=
sorry

end trailing_zeros_in_15_factorial_base_15_are_3_l5_5134


namespace hostel_cost_for_23_days_l5_5797

theorem hostel_cost_for_23_days :
  let first_week_days := 7
  let additional_days := 23 - first_week_days
  let cost_first_week := 18 * first_week_days
  let cost_additional_weeks := 11 * additional_days
  23 * ((cost_first_week + cost_additional_weeks) / 23) = 302 :=
by sorry

end hostel_cost_for_23_days_l5_5797


namespace sum_geq_4k_l5_5862

theorem sum_geq_4k (a b k : ℕ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_k : k > 1)
  (h_lcm_gcd : Nat.lcm a b + Nat.gcd a b = k * (a + b)) : a + b ≥ 4 * k := 
by 
  sorry

end sum_geq_4k_l5_5862


namespace lin_reg_proof_l5_5822

variable (x y : List ℝ)
variable (n : ℝ := 10)
variable (sum_x : ℝ := 80)
variable (sum_y : ℝ := 20)
variable (sum_xy : ℝ := 184)
variable (sum_x2 : ℝ := 720)

noncomputable def mean (lst: List ℝ) (n: ℝ) : ℝ := (List.sum lst) / n

noncomputable def lin_reg_slope (n sum_x sum_y sum_xy sum_x2 : ℝ) : ℝ :=
  (sum_xy - n * (sum_x / n) * (sum_y / n)) / (sum_x2 - n * (sum_x / n) ^ 2)

noncomputable def lin_reg_intercept (sum_x sum_y : ℝ) (slope : ℝ) (n : ℝ) : ℝ :=
  (sum_y / n) - slope * (sum_x / n)

theorem lin_reg_proof :
  lin_reg_slope n sum_x sum_y sum_xy sum_x2 = 0.3 ∧ 
  lin_reg_intercept sum_x sum_y 0.3 n = -0.4 ∧ 
  (0.3 * 7 - 0.4 = 1.7) :=
by
  sorry

end lin_reg_proof_l5_5822


namespace koala_food_consumed_l5_5081

theorem koala_food_consumed (x y : ℝ) (h1 : 0.40 * x = 12) (h2 : 0.20 * y = 2) : 
  x = 30 ∧ y = 10 := 
by
  sorry

end koala_food_consumed_l5_5081


namespace f_four_l5_5069

noncomputable def f : ℝ → ℝ := sorry

axiom functional_eq (a b : ℝ) : f (a + b) + f (a - b) = 2 * f a + 2 * f b
axiom f_two : f 2 = 9 
axiom not_identically_zero : ¬ ∀ x : ℝ, f x = 0

theorem f_four : f 4 = 36 :=
by sorry

end f_four_l5_5069


namespace avg_price_per_book_l5_5941

theorem avg_price_per_book (n1 n2 p1 p2 : ℕ) (h1 : n1 = 65) (h2 : n2 = 55) (h3 : p1 = 1380) (h4 : p2 = 900) :
    (p1 + p2) / (n1 + n2) = 19 := by
  sorry

end avg_price_per_book_l5_5941


namespace supercomputer_transformation_stops_l5_5569

def transformation_rule (n : ℕ) : ℕ :=
  let A : ℕ := n / 100
  let B : ℕ := n % 100
  2 * A + 8 * B

theorem supercomputer_transformation_stops (n : ℕ) :
  let start := (10^900 - 1) / 9 -- 111...111 with 900 ones
  (n = start) → (∀ m, transformation_rule m < 100 → false) :=
by
  sorry

end supercomputer_transformation_stops_l5_5569


namespace point_on_line_l5_5213

theorem point_on_line : ∀ (t : ℤ), 
  (∃ m : ℤ, (6 - 2) * m = 20 - 8 ∧ (10 - 6) * m = 32 - 20) →
  (∃ b : ℤ, 8 - 2 * m = b) →
  t = m * 35 + b → t = 107 :=
by
  sorry

end point_on_line_l5_5213


namespace find_t_l5_5759

-- Define the vectors
def a : ℝ × ℝ := (1, 2)
def b : ℝ × ℝ := (4, 3)

-- Define the perpendicular condition and solve for t
theorem find_t (t : ℝ) : a.1 * (t * a.1 + b.1) + a.2 * (t * a.2 + b.2) = 0 → t = -2 :=
by
  sorry

end find_t_l5_5759


namespace tan_45_deg_eq_one_l5_5960

/-- The tangent of 45 degrees is equal to 1. -/
theorem tan_45_deg_eq_one : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_deg_eq_one_l5_5960


namespace median_score_interval_l5_5729

def intervals : List (Nat × Nat × Nat) :=
  [(80, 84, 20), (75, 79, 18), (70, 74, 15), (65, 69, 22), (60, 64, 14), (55, 59, 11)]

def total_students : Nat := 100

def median_interval : Nat × Nat :=
  (70, 74)

theorem median_score_interval :
  ∃ l u n, intervals = [(80, 84, 20), (75, 79, 18), (70, 74, 15), (65, 69, 22), (60, 64, 14), (55, 59, 11)]
  ∧ total_students = 100
  ∧ median_interval = (70, 74)
  ∧ ((l, u, n) ∈ intervals ∧ l ≤ 50 ∧ 50 ≤ u) :=
by
  sorry

end median_score_interval_l5_5729


namespace ratio_of_ages_three_years_from_now_l5_5466

theorem ratio_of_ages_three_years_from_now :
  ∃ L B : ℕ,
  (L + B = 6) ∧ 
  (L = (1/2 : ℝ) * B) ∧ 
  (L + 3 = 5) ∧ 
  (B + 3 = 7) → 
  (L + 3) / (B + 3) = (5/7 : ℝ) :=
by
  sorry

end ratio_of_ages_three_years_from_now_l5_5466


namespace maximum_area_of_right_angled_triangle_l5_5048

noncomputable def max_area_right_angled_triangle (a b c : ℕ) (h1 : a^2 + b^2 = c^2) (h2 : a + b + c = 48) : ℕ := 
  max (a * b / 2) 288

theorem maximum_area_of_right_angled_triangle (a b c : ℕ) 
  (h1 : a^2 + b^2 = c^2)    -- Pythagorean theorem
  (h2 : a + b + c = 48)     -- Perimeter condition
  (h3 : 0 < a)              -- Positive integer side length condition
  (h4 : 0 < b)              -- Positive integer side length condition
  (h5 : 0 < c)              -- Positive integer side length condition
  : max_area_right_angled_triangle a b c h1 h2 = 288 := 
sorry

end maximum_area_of_right_angled_triangle_l5_5048


namespace parabola_translation_shift_downwards_l5_5740

theorem parabola_translation_shift_downwards :
  ∀ (x y : ℝ), (y = x^2 - 5) ↔ ((∃ (k : ℝ), k = -5 ∧ y = x^2 + k)) :=
by
  sorry

end parabola_translation_shift_downwards_l5_5740


namespace degree_greater_than_2_l5_5493

variable (P Q : ℤ[X]) -- P and Q are polynomials with integer coefficients

theorem degree_greater_than_2 (P_nonconstant : ¬(P.degree = 0))
  (Q_nonconstant : ¬(Q.degree = 0))
  (h : ∃ S : Finset ℤ, S.card ≥ 25 ∧ ∀ x ∈ S, (P.eval x) * (Q.eval x) = 2009) :
  P.degree > 2 ∧ Q.degree > 2 :=
by
  sorry

end degree_greater_than_2_l5_5493


namespace garden_perimeter_equals_104_l5_5963

theorem garden_perimeter_equals_104 :
  let playground_length := 16
  let playground_width := 12
  let playground_area := playground_length * playground_width
  let garden_width := 4
  let garden_length := playground_area / garden_width
  let garden_perimeter := 2 * garden_length + 2 * garden_width
  playground_area = 192 ∧ garden_perimeter = 104 :=
by {
  -- Declarations
  let playground_length := 16
  let playground_width := 12
  let playground_area := playground_length * playground_width
  let garden_width := 4
  let garden_length := playground_area / garden_width
  let garden_perimeter := 2 * garden_length + 2 * garden_width

  -- Assertions
  have area_playground : playground_area = 192 := by sorry
  have perimeter_garden : garden_perimeter = 104 := by sorry

  -- Conclusion
  exact ⟨area_playground, perimeter_garden⟩
}

end garden_perimeter_equals_104_l5_5963


namespace log_sum_equality_l5_5695

noncomputable def log_base_5 (x : ℝ) := Real.log x / Real.log 5

theorem log_sum_equality :
  2 * log_base_5 10 + log_base_5 0.25 = 2 :=
by
  sorry -- proof goes here

end log_sum_equality_l5_5695


namespace total_cost_full_units_l5_5890

def total_units : Nat := 12
def cost_1_bedroom : Nat := 360
def cost_2_bedroom : Nat := 450
def num_2_bedroom : Nat := 7
def num_1_bedroom : Nat := total_units - num_2_bedroom

def total_cost : Nat := (num_1_bedroom * cost_1_bedroom) + (num_2_bedroom * cost_2_bedroom)

theorem total_cost_full_units : total_cost = 4950 := by
  -- proof would go here
  sorry

end total_cost_full_units_l5_5890


namespace derivative_at_one_l5_5130

noncomputable def f (x : ℝ) : ℝ := 1 / (1 + x)

theorem derivative_at_one :
  deriv f 1 = -1 / 4 :=
by
  sorry

end derivative_at_one_l5_5130


namespace expression_evaluates_to_3_l5_5177

theorem expression_evaluates_to_3 :
  (3 + Real.sqrt 3 + 1 / (3 + Real.sqrt 3) + 1 / (Real.sqrt 3 - 3)) = 3 :=
sorry

end expression_evaluates_to_3_l5_5177


namespace students_errors_proof_l5_5102

noncomputable def students (x y0 y1 y2 y3 y4 y5 : ℕ): ℕ :=
  x + y5 + y4 + y3 + y2 + y1 + y0

noncomputable def errors (x y1 y2 y3 y4 y5 : ℕ): ℕ :=
  6 * x + 5 * y5 + 4 * y4 + 3 * y3 + 2 * y2 + y1

theorem students_errors_proof
  (x y0 y1 y2 y3 y4 y5 : ℕ)
  (h1 : students x y0 y1 y2 y3 y4 y5 = 333)
  (h2 : errors x y1 y2 y3 y4 y5 ≤ 1000) :
  x ≤ y3 + y2 + y1 + y0 :=
by
  sorry

end students_errors_proof_l5_5102


namespace relationship_between_b_and_c_l5_5973

-- Definitions based on the given conditions
def y1 (x a b : ℝ) : ℝ := (x + 2 * a) * (x - 2 * b)
def y2 (x b : ℝ) : ℝ := -x + 2 * b
def y (x a b : ℝ) : ℝ := y1 x a b + y2 x b

-- Lean theorem for the proof problem
theorem relationship_between_b_and_c
  (a b c : ℝ)
  (h : a + 2 = b)
  (h_y : y c a b = 0) :
  c = 5 - 2 * b ∨ c = 2 * b :=
by
  -- The proof will go here, currently omitted
  sorry

end relationship_between_b_and_c_l5_5973


namespace leaves_dropped_on_fifth_day_l5_5194

theorem leaves_dropped_on_fifth_day 
  (initial_leaves : ℕ)
  (days : ℕ)
  (drops_per_day : ℕ)
  (total_dropped_four_days : ℕ)
  (leaves_dropped_fifth_day : ℕ)
  (h1 : initial_leaves = 340)
  (h2 : days = 4)
  (h3 : drops_per_day = initial_leaves / 10)
  (h4 : total_dropped_four_days = drops_per_day * days)
  (h5 : leaves_dropped_fifth_day = initial_leaves - total_dropped_four_days) :
  leaves_dropped_fifth_day = 204 :=
by
  sorry

end leaves_dropped_on_fifth_day_l5_5194


namespace simplify_fraction_l5_5708

theorem simplify_fraction :
  (1 / (1 / (Real.sqrt 2 + 1) + 1 / (Real.sqrt 5 - 2))) =
  ((Real.sqrt 2 + Real.sqrt 5 - 1) / (6 + 2 * Real.sqrt 10)) :=
by
  sorry

end simplify_fraction_l5_5708


namespace sufficient_condition_l5_5974

variable {α : Type*} (A B : Set α)

theorem sufficient_condition (h : A ⊆ B) (x : α) : x ∈ A → x ∈ B :=
by
  sorry

end sufficient_condition_l5_5974


namespace right_triangle_hypotenuse_l5_5298

theorem right_triangle_hypotenuse :
  ∃ b a : ℕ, a^2 + 1994^2 = b^2 ∧ b = 994010 :=
by
  sorry

end right_triangle_hypotenuse_l5_5298


namespace initial_legos_l5_5032

-- Definitions and conditions
def legos_won : ℝ := 17.0
def legos_now : ℝ := 2097.0

-- The statement to prove
theorem initial_legos : (legos_now - legos_won) = 2080 :=
by sorry

end initial_legos_l5_5032


namespace ellipse_focus_m_eq_3_l5_5208

theorem ellipse_focus_m_eq_3 (m : ℝ) (h : m > 0) : 
  (∃ a c : ℝ, a = 5 ∧ c = 4 ∧ c^2 = a^2 - m^2)
  → m = 3 :=
by
  sorry

end ellipse_focus_m_eq_3_l5_5208


namespace find_function_α_l5_5731

theorem find_function_α (α : ℝ) (hα : 0 < α) 
  (f : ℕ+ → ℝ) (h : ∀ k m : ℕ+, α * m ≤ k ∧ k < (α + 1) * m → f (k + m) = f k + f m) :
  ∃ b : ℝ, ∀ n : ℕ+, f n = b * n :=
sorry

end find_function_α_l5_5731


namespace alcohol_added_amount_l5_5261

theorem alcohol_added_amount :
  ∀ (x : ℝ), (40 * 0.05 + x) = 0.15 * (40 + x + 4.5) -> x = 5.5 :=
by
  intro x
  sorry

end alcohol_added_amount_l5_5261


namespace impossible_to_empty_pile_l5_5783

theorem impossible_to_empty_pile (a b c : ℕ) (h : a = 1993 ∧ b = 199 ∧ c = 19) : 
  ¬ (∃ x y z : ℕ, (x + y + z = 0) ∧ (x = a ∨ x = b ∨ x = c ∧ y = a ∨ y = b ∨ y = c ∧ z = a ∨ z = b ∨ z = c)) := 
sorry

end impossible_to_empty_pile_l5_5783


namespace K_3_15_10_eq_151_30_l5_5653

def K (a b c : ℕ) : ℚ := (a : ℚ) / b + (b : ℚ) / c + (c : ℚ) / a

theorem K_3_15_10_eq_151_30 : K 3 15 10 = 151 / 30 := 
by
  sorry

end K_3_15_10_eq_151_30_l5_5653


namespace factorize_quadratic_l5_5331

theorem factorize_quadratic (x : ℝ) : x^2 - 2 * x = x * (x - 2) :=
sorry

end factorize_quadratic_l5_5331


namespace sum_of_highest_powers_of_10_and_6_dividing_20_factorial_l5_5465

def legendre (n p : Nat) : Nat :=
  if p > 1 then (Nat.div n p + Nat.div n (p * p) + Nat.div n (p * p * p) + Nat.div n (p * p * p * p)) else 0

theorem sum_of_highest_powers_of_10_and_6_dividing_20_factorial :
  let highest_power_5 := legendre 20 5
  let highest_power_2 := legendre 20 2
  let highest_power_3 := legendre 20 3
  let highest_power_10 := min highest_power_2 highest_power_5
  let highest_power_6 := min highest_power_2 highest_power_3
  highest_power_10 + highest_power_6 = 12 :=
by
  sorry

end sum_of_highest_powers_of_10_and_6_dividing_20_factorial_l5_5465


namespace johns_avg_speed_l5_5652

/-
John cycled 40 miles at 8 miles per hour and 20 miles at 40 miles per hour.
We want to prove that his average speed for the entire trip is 10.91 miles per hour.
-/

theorem johns_avg_speed :
  let distance1 := 40
  let speed1 := 8
  let distance2 := 20
  let speed2 := 40
  let total_distance := distance1 + distance2
  let time1 := distance1 / speed1
  let time2 := distance2 / speed2
  let total_time := time1 + time2
  let avg_speed := total_distance / total_time
  avg_speed = 10.91 :=
by
  sorry

end johns_avg_speed_l5_5652


namespace pairs_m_n_l5_5592

theorem pairs_m_n (m n : ℤ) : n ^ 2 - 3 * m * n + m - n = 0 ↔ (m = 0 ∧ n = 0) ∨ (m = 0 ∧ n = 1) :=
by sorry

end pairs_m_n_l5_5592


namespace gunny_bag_capacity_l5_5608

def pounds_per_ton : ℝ := 2500
def ounces_per_pound : ℝ := 16
def packets : ℝ := 2000
def packet_weight_pounds : ℝ := 16
def packet_weight_ounces : ℝ := 4

theorem gunny_bag_capacity :
  (packets * (packet_weight_pounds + packet_weight_ounces / ounces_per_pound) / pounds_per_ton) = 13 := 
by
  sorry

end gunny_bag_capacity_l5_5608


namespace mini_bottles_needed_to_fill_jumbo_l5_5672

def mini_bottle_capacity : ℕ := 45
def jumbo_bottle_capacity : ℕ := 600

-- The problem statement expressed as a Lean theorem.
theorem mini_bottles_needed_to_fill_jumbo :
  (jumbo_bottle_capacity + mini_bottle_capacity - 1) / mini_bottle_capacity = 14 :=
by
  sorry

end mini_bottles_needed_to_fill_jumbo_l5_5672


namespace find_additional_payment_l5_5024

-- Definitions used from the conditions
def total_payments : ℕ := 52
def first_partial_payments : ℕ := 25
def second_partial_payments : ℕ := total_payments - first_partial_payments
def first_payment_amount : ℝ := 500
def average_payment : ℝ := 551.9230769230769

-- Condition in Lean
theorem find_additional_payment :
  let total_amount := average_payment * total_payments
  let first_payment_total := first_partial_payments * first_payment_amount
  ∃ x : ℝ, total_amount = first_payment_total + second_partial_payments * (first_payment_amount + x) → x = 100 :=
by
  sorry

end find_additional_payment_l5_5024


namespace area_of_shape_is_correct_l5_5893

noncomputable def square_side_length : ℝ := 2 * Real.pi

noncomputable def semicircle_radius : ℝ := square_side_length / 2

noncomputable def area_of_resulting_shape : ℝ :=
  let area_square := square_side_length^2
  let area_semicircle := (1/2) * Real.pi * semicircle_radius^2
  let total_area := area_square + 4 * area_semicircle
  total_area

theorem area_of_shape_is_correct :
  area_of_resulting_shape = 2 * Real.pi^2 * (Real.pi + 2) :=
sorry

end area_of_shape_is_correct_l5_5893


namespace valid_four_digit_number_count_l5_5337

theorem valid_four_digit_number_count : 
  let first_digit_choices := 6 
  let last_digit_choices := 10 
  let middle_digits_valid_pairs := 9 * 9 - 18
  (first_digit_choices * middle_digits_valid_pairs * last_digit_choices = 3780) := by
  sorry

end valid_four_digit_number_count_l5_5337


namespace negation_example_l5_5778

open Classical
variable (x : ℝ)

theorem negation_example :
  (¬ (∀ x : ℝ, 2 * x - 1 > 0)) ↔ (∃ x : ℝ, 2 * x - 1 ≤ 0) :=
by
  sorry

end negation_example_l5_5778


namespace trucks_transportation_l5_5611

theorem trucks_transportation (k : ℕ) (H : ℝ) : 
  (∃ (A B C : ℕ), 
     A + B + C = k ∧ 
     A ≤ k / 2 ∧ B ≤ k / 2 ∧ C ≤ k / 2 ∧ 
     (0 ≤ (k - 2*A)) ∧ (0 ≤ (k - 2*B)) ∧ (0 ≤ (k - 2*C))) 
  →  (k = 7 → (2 : ℕ) = 2) :=
sorry

end trucks_transportation_l5_5611


namespace range_of_m_l5_5433

noncomputable def f (x m : ℝ) : ℝ := x^2 - x + m * (2 * x + 1)

theorem range_of_m (m : ℝ) : (∀ x > 1, 0 < 2 * x + (2 * m - 1)) ↔ (m ≥ -1/2) := by
  sorry

end range_of_m_l5_5433


namespace mark_cans_l5_5222

variable (r j m : ℕ) -- r for Rachel, j for Jaydon, m for Mark

theorem mark_cans (r j m : ℕ) 
  (h1 : j = 5 + 2 * r)
  (h2 : m = 4 * j)
  (h3 : r + j + m = 135) : 
  m = 100 :=
by
  sorry

end mark_cans_l5_5222


namespace number_of_round_trips_each_bird_made_l5_5285

theorem number_of_round_trips_each_bird_made
  (distance_to_materials : ℕ)
  (total_distance_covered : ℕ)
  (distance_one_round_trip : ℕ)
  (total_number_of_trips : ℕ)
  (individual_bird_trips : ℕ) :
  distance_to_materials = 200 →
  total_distance_covered = 8000 →
  distance_one_round_trip = 2 * distance_to_materials →
  total_number_of_trips = total_distance_covered / distance_one_round_trip →
  individual_bird_trips = total_number_of_trips / 2 →
  individual_bird_trips = 10 :=
by
  intros
  sorry

end number_of_round_trips_each_bird_made_l5_5285


namespace hyperbola_sufficient_condition_l5_5474

-- Define the condition for the equation to represent a hyperbola
def represents_hyperbola (k : ℝ) : Prop :=
  (3 - k) * (k - 1) < 0

-- Lean 4 statement to prove that k > 3 is a sufficient condition for the given equation
theorem hyperbola_sufficient_condition (k : ℝ) (h : k > 3) :
  represents_hyperbola k :=
sorry

end hyperbola_sufficient_condition_l5_5474


namespace value_of_f_eval_at_pi_over_12_l5_5529

noncomputable def f (x : ℝ) : ℝ := Real.sin x + Real.cos x

theorem value_of_f_eval_at_pi_over_12 : f (Real.pi / 12) = (Real.sqrt 6) / 2 :=
by
  sorry

end value_of_f_eval_at_pi_over_12_l5_5529


namespace sum_digits_base8_to_base4_l5_5210

theorem sum_digits_base8_to_base4 :
  ∀ n : ℕ, (n ≥ 512 ∧ n ≤ 4095) →
  (∃ d : ℕ, (4^d > n ∧ n ≥ 4^(d-1))) →
  (d = 6) :=
by {
  sorry
}

end sum_digits_base8_to_base4_l5_5210


namespace costume_total_cost_l5_5685

variable (friends : ℕ) (cost_per_costume : ℕ) 

theorem costume_total_cost (h1 : friends = 8) (h2 : cost_per_costume = 5) : friends * cost_per_costume = 40 :=
by {
  sorry -- We omit the proof, as instructed.
}

end costume_total_cost_l5_5685


namespace total_diagonals_in_rectangular_prism_l5_5190

-- We define the rectangular prism with its properties
structure RectangularPrism :=
  (vertices : ℕ)
  (edges : ℕ)
  (distinct_dimensions : ℕ)

-- We specify the conditions for the rectangular prism
def givenPrism : RectangularPrism :=
{
  vertices := 8,
  edges := 12,
  distinct_dimensions := 3
}

-- We assert the total number of diagonals in the rectangular prism
theorem total_diagonals_in_rectangular_prism (P : RectangularPrism) : P = givenPrism → ∃ diag, diag = 16 :=
by
  intro h
  have diag := 16
  use diag
  sorry

end total_diagonals_in_rectangular_prism_l5_5190


namespace range_of_m_l5_5597

theorem range_of_m (m : ℝ) :
  (∀ x : ℝ, (x + m) * (2 - x) < 1) ↔ (-4 < m ∧ m < 0) :=
sorry

end range_of_m_l5_5597


namespace lattice_points_in_region_l5_5629

theorem lattice_points_in_region : ∃ n : ℕ, n = 1 ∧ ∀ p : ℤ × ℤ, 
  (p.snd = abs p.fst ∨ p.snd = -(p.fst ^ 3) + 6 * (p.fst)) → n = 1 :=
by
  sorry

end lattice_points_in_region_l5_5629


namespace children_got_on_the_bus_l5_5033

-- Definitions
def original_children : ℕ := 26
def current_children : ℕ := 64

-- Theorem stating the problem
theorem children_got_on_the_bus : (current_children - original_children = 38) :=
by {
  sorry
}

end children_got_on_the_bus_l5_5033


namespace part_1_part_2_1_part_2_2_l5_5382

variable {k x : ℝ}
def y (k : ℝ) (x : ℝ) := k * x^2 - 2 * k * x + 2 * k - 1

theorem part_1 (k : ℝ) : (∀ x, y k x ≥ 4 * k - 2) ↔ (0 ≤ k ∧ k ≤ 1 / 3) := by
  sorry

theorem part_2_1 (k : ℝ) : ¬∃ x1 x2 : ℝ, y k x = 0 ∧ y k x = 0 ∧ x1^2 + x2^2 = 3 * x1 * x2 - 4 := by
  sorry

theorem part_2_2 (k : ℝ) : (∀ x1 x2 : ℝ, y k x = 0 ∧ y k x = 0 ∧ x1 > 0 ∧ x2 > 0) ↔ (1 / 2 < k ∧ k < 1) := by
  sorry

end part_1_part_2_1_part_2_2_l5_5382


namespace smallest_b_for_45_b_square_l5_5760

theorem smallest_b_for_45_b_square :
  ∃ b : ℕ, b > 5 ∧ ∃ n : ℕ, 4 * b + 5 = n^2 ∧ b = 11 :=
by
  sorry

end smallest_b_for_45_b_square_l5_5760


namespace find_prob_A_l5_5906

variable (P : String → ℝ)
variable (A B : String)

-- Conditions
axiom prob_complement_twice : P B = 2 * P A
axiom prob_sum_to_one : P A + P B = 1

-- Statement to be proved
theorem find_prob_A : P A = 1 / 3 :=
by
  -- Proof to be filled in
  sorry

end find_prob_A_l5_5906


namespace integral_solution_l5_5966

noncomputable def integral_expression : Real → Real :=
  fun x => (1 + (x ^ (3 / 4))) ^ (4 / 5) / (x ^ (47 / 20))

theorem integral_solution :
  ∫ (x : Real), integral_expression x = - (20 / 27) * ((1 + (x ^ (3 / 4)) / (x ^ (3 / 4))) ^ (9 / 5)) + C := 
by 
  sorry

end integral_solution_l5_5966


namespace cos_negative_570_equals_negative_sqrt3_div_2_l5_5253

theorem cos_negative_570_equals_negative_sqrt3_div_2 : Real.cos (-570 * Real.pi / 180) = - (Real.sqrt 3 / 2) :=
by
  sorry

end cos_negative_570_equals_negative_sqrt3_div_2_l5_5253


namespace vasya_numbers_l5_5079

theorem vasya_numbers (x y : ℝ) (h1 : x + y = x * y) (h2 : x * y = x / y) : 
  (x = 1 / 2 ∧ y = -1) ∨ (x = -1 ∧ y = 1 / 2) ∨ (x = y ∧ x = 0) :=
by 
  -- Placeholder to show where the proof would go
  sorry

end vasya_numbers_l5_5079


namespace south_walk_correct_representation_l5_5472

theorem south_walk_correct_representation {north south : ℤ} (h_north : north = 3) (h_representation : south = -north) : south = -5 :=
by
  have h1 : -north = -3 := by rw [h_north]
  have h2 : -3 = -5 := by sorry
  rw [h_representation, h1]
  exact h2

end south_walk_correct_representation_l5_5472


namespace quadratic_roots_squared_sum_l5_5526

theorem quadratic_roots_squared_sum (m n : ℝ) (h1 : m^2 - 2 * m - 1 = 0) (h2 : n^2 - 2 * n - 1 = 0) : m^2 + n^2 = 6 :=
sorry

end quadratic_roots_squared_sum_l5_5526


namespace allowance_amount_l5_5635

variable (initial_money spent_money final_money : ℕ)

theorem allowance_amount (initial_money : ℕ) (spent_money : ℕ) (final_money : ℕ) (h1: initial_money = 5) (h2: spent_money = 2) (h3: final_money = 8) : (final_money - (initial_money - spent_money)) = 5 := 
by 
  sorry

end allowance_amount_l5_5635


namespace pipe_B_fill_time_l5_5153

theorem pipe_B_fill_time
  (rate_A : ℝ)
  (rate_B : ℝ)
  (t : ℝ)
  (h_rate_A : rate_A = 2 / 75)
  (h_rate_B : rate_B = 1 / t)
  (h_fill_total : 9 * (rate_A + rate_B) + 21 * rate_A = 1) :
  t = 45 := 
sorry

end pipe_B_fill_time_l5_5153


namespace geometric_seq_a4_a7_l5_5751

variable {a : ℕ → ℝ}

def is_geometric (a : ℕ → ℝ) : Prop :=
∀ n m : ℕ, ∃ r : ℝ, a (n + 1) = r * a n

theorem geometric_seq_a4_a7
  (h_geom : is_geometric a)
  (h_roots : ∃ a_1 a_10 : ℝ, (a 1 = a_1 ∧ a 10 = a_10) ∧ (2 * a_1 ^ 2 + 5 * a_1 + 1 = 0) ∧ (2 * a_10 ^ 2 + 5 * a_10 + 1 = 0)):
  a 4 * a 7 = 1 / 2 :=
by
  sorry

end geometric_seq_a4_a7_l5_5751


namespace total_pens_is_50_l5_5197

theorem total_pens_is_50
  (red : ℕ) (black : ℕ) (blue : ℕ) (green : ℕ) (purple : ℕ) (total : ℕ)
  (h1 : red = 8)
  (h2 : black = 3 / 2 * red)
  (h3 : blue = black + 5 ∧ blue = 1 / 5 * total)
  (h4 : green = blue / 2)
  (h5 : purple = 5)
  : total = red + black + blue + green + purple := sorry

end total_pens_is_50_l5_5197


namespace find_x_in_triangle_XYZ_l5_5942

theorem find_x_in_triangle_XYZ (y : ℝ) (z : ℝ) (cos_Y_minus_Z : ℝ) (hx : y = 7) (hz : z = 6) (hcos : cos_Y_minus_Z = 47 / 64) : 
    ∃ x : ℝ, x = Real.sqrt 63.75 :=
by
  -- The proof will go here, but it is skipped for now.
  sorry

end find_x_in_triangle_XYZ_l5_5942


namespace value_of_polynomial_l5_5477

theorem value_of_polynomial :
  98^3 + 3 * (98^2) + 3 * 98 + 1 = 970299 :=
by sorry

end value_of_polynomial_l5_5477


namespace smallest_sector_angle_division_is_10_l5_5917

/-
  Prove that the smallest possible sector angle in a 15-sector division of a circle,
  where the central angles form an arithmetic sequence with integer values and the
  total sum of angles is 360 degrees, is 10 degrees.
-/
theorem smallest_sector_angle_division_is_10 :
  ∃ (a1 d : ℕ), (∀ i, i ∈ (List.range 15) → a1 + i * d > 0) ∧ (List.sum (List.map (fun i => a1 + i * d) (List.range 15)) = 360) ∧
  a1 = 10 := by
  sorry

end smallest_sector_angle_division_is_10_l5_5917


namespace algebra_simplification_l5_5709

theorem algebra_simplification (a b : ℤ) (h : ∀ x : ℤ, x^2 - 6 * x + b = (x - a)^2 - 1) : b - a = 5 := by
  sorry

end algebra_simplification_l5_5709


namespace common_chord_eq_l5_5319

theorem common_chord_eq : 
  (∀ x y : ℝ, x^2 + y^2 + 2*x + 8*y - 8 = 0) ∧ 
  (∀ x y : ℝ, x^2 + y^2 - 4*x - 4*y - 2 = 0) → 
  (∀ x y : ℝ, x + 2*y - 1 = 0) :=
by 
  sorry

end common_chord_eq_l5_5319


namespace qinJiushao_value_l5_5119

/-- A specific function f(x) with given a and b -/
def f (x : ℤ) : ℤ :=
  x^5 + 47 * x^4 - 37 * x^2 + 1

/-- Qin Jiushao algorithm to find V3 at x = -1 -/
def qinJiushao (x : ℤ) : ℤ :=
  let V0 := 1
  let V1 := V0 * x + 47
  let V2 := V1 * x + 0
  let V3 := V2 * x - 37
  V3

theorem qinJiushao_value :
  qinJiushao (-1) = 9 :=
by
  sorry

end qinJiushao_value_l5_5119


namespace exponentiation_rule_proof_l5_5543

-- Definitions based on conditions
def x : ℕ := 3
def a : ℕ := 4
def b : ℕ := 2

-- The rule that relates the exponents
def rule (x a b : ℕ) : ℕ := x^(a * b)

-- Proposition that we need to prove
theorem exponentiation_rule_proof : rule x a b = 6561 :=
by
  -- sorry is used to indicate the proof is omitted
  sorry

end exponentiation_rule_proof_l5_5543


namespace length_segment_AB_l5_5276

theorem length_segment_AB (A B : ℝ) (hA : A = -5) (hB : B = 2) : |A - B| = 7 :=
by
  sorry

end length_segment_AB_l5_5276


namespace a2_plus_b2_minus_abc_is_perfect_square_l5_5020

noncomputable def is_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

theorem a2_plus_b2_minus_abc_is_perfect_square {a b c : ℕ} (h : 0 < a^2 + b^2 - a * b * c ∧ a^2 + b^2 - a * b * c ≤ c) :
  is_perfect_square (a^2 + b^2 - a * b * c) :=
by
  sorry

end a2_plus_b2_minus_abc_is_perfect_square_l5_5020


namespace sector_angle_l5_5887

theorem sector_angle (l S : ℝ) (r α : ℝ) 
  (h_arc_length : l = 6)
  (h_area : S = 6)
  (h_area_formula : S = 1/2 * l * r)
  (h_arc_formula : l = r * α) : 
  α = 3 :=
by
  sorry

end sector_angle_l5_5887


namespace sum_of_areas_of_triangles_l5_5353

theorem sum_of_areas_of_triangles 
  (AB BG GE DE : ℕ) 
  (A₁ A₂ : ℕ)
  (H1 : AB = 2) 
  (H2 : BG = 3) 
  (H3 : GE = 4) 
  (H4 : DE = 5) 
  (H5 : 3 * A₁ + 4 * A₂ = 48)
  (H6 : 9 * A₁ + 5 * A₂ = 102) : 
  1 * AB * A₁ / 2 + 1 * DE * A₂ / 2 = 23 :=
by
  sorry

end sum_of_areas_of_triangles_l5_5353


namespace grunters_win_all_6_games_l5_5648

noncomputable def prob_no_overtime_win : ℚ := 0.54
noncomputable def prob_overtime_win : ℚ := 0.05
noncomputable def prob_win_any_game : ℚ := prob_no_overtime_win + prob_overtime_win
noncomputable def prob_win_all_6_games : ℚ := prob_win_any_game ^ 6

theorem grunters_win_all_6_games :
  prob_win_all_6_games = (823543 / 10000000) :=
by sorry

end grunters_win_all_6_games_l5_5648


namespace length_of_each_stone_l5_5232

-- Define the dimensions of the hall in decimeters
def hall_length_dm : ℕ := 36 * 10
def hall_breadth_dm : ℕ := 15 * 10

-- Define the width of each stone in decimeters
def stone_width_dm : ℕ := 5

-- Define the number of stones
def number_of_stones : ℕ := 1350

-- Define the total area of the hall
def hall_area : ℕ := hall_length_dm * hall_breadth_dm

-- Define the area of one stone
def stone_area : ℕ := hall_area / number_of_stones

-- Define the length of each stone and state the theorem
theorem length_of_each_stone : (stone_area / stone_width_dm) = 8 :=
by
  sorry

end length_of_each_stone_l5_5232


namespace sasha_remainder_is_20_l5_5878

theorem sasha_remainder_is_20 (n a b c d : ℤ) (h1 : n = 102 * a + b) (h2 : n = 103 * c + d) (h3 : d = 20 - a) : b = 20 :=
by
  sorry

end sasha_remainder_is_20_l5_5878


namespace simplify_expression_l5_5277

variable (x y : ℕ)
variable (h_x : x = 5)
variable (h_y : y = 2)

theorem simplify_expression : (10 * x^2 * y^3) / (15 * x * y^2) = 20 / 3 := by
  sorry

end simplify_expression_l5_5277


namespace lateral_surface_area_of_cylinder_l5_5595

theorem lateral_surface_area_of_cylinder :
  (∀ (side_length : ℕ), side_length = 10 → 
  ∃ (lateral_surface_area : ℝ), lateral_surface_area = 100 * Real.pi) :=
by
  sorry

end lateral_surface_area_of_cylinder_l5_5595


namespace oliver_final_amount_is_54_04_l5_5536

noncomputable def final_amount : ℝ :=
  let initial := 33
  let feb_spent := 0.15 * initial
  let after_feb := initial - feb_spent
  let march_added := 32
  let after_march := after_feb + march_added
  let march_spent := 0.10 * after_march
  after_march - march_spent

theorem oliver_final_amount_is_54_04 : final_amount = 54.04 := by
  sorry

end oliver_final_amount_is_54_04_l5_5536


namespace limit_tanxy_over_y_l5_5074

theorem limit_tanxy_over_y (f : ℝ×ℝ → ℝ) :
  (∀ ε > 0, ∃ δ > 0, ∀ x y, abs (x - 3) < δ ∧ abs y < δ → abs (f (x, y) - 3) < ε) :=
sorry

end limit_tanxy_over_y_l5_5074


namespace multiplicative_inverse_l5_5688

def A : ℕ := 123456
def B : ℕ := 171428
def mod_val : ℕ := 1000000
def sum_A_B : ℕ := A + B
def N : ℕ := 863347

theorem multiplicative_inverse : (sum_A_B * N) % mod_val = 1 :=
by
  -- diverting proof with sorry since proof steps aren't the focus
  sorry

end multiplicative_inverse_l5_5688


namespace train_times_valid_l5_5302

-- Define the parameters and conditions
def trainA_usual_time : ℝ := 180 -- minutes
def trainB_travel_time : ℝ := 810 -- minutes

theorem train_times_valid (t : ℝ) (T_B : ℝ) 
  (cond1 : (7 / 6) * t = t + 30)
  (cond2 : T_B = 4.5 * t) : 
  t = trainA_usual_time ∧ T_B = trainB_travel_time :=
by
  sorry

end train_times_valid_l5_5302


namespace john_books_per_day_l5_5088

theorem john_books_per_day (books_total : ℕ) (total_weeks : ℕ) (days_per_week : ℕ) (total_days : ℕ)
  (read_days_eq : total_days = total_weeks * days_per_week)
  (books_per_day_eq : books_total = total_days * 4) : (books_total / total_days = 4) :=
by
  -- The conditions state the following:
  -- books_total = 48 (total books read)
  -- total_weeks = 6 (total number of weeks)
  -- days_per_week = 2 (number of days John reads per week)
  -- total_days = 12 (total number of days in which John reads books)
  -- read_days_eq :- total_days = total_weeks * days_per_week
  -- books_per_day_eq :- books_total = total_days * 4
  sorry

end john_books_per_day_l5_5088


namespace final_output_value_of_m_l5_5563

variables (a b m : ℕ)

theorem final_output_value_of_m (h₁ : a = 2) (h₂ : b = 3) (program_logic : (a > b → m = a) ∧ (a ≤ b → m = b)) :
  m = 3 :=
by
  have h₃ : a ≤ b := by
    rw [h₁, h₂]
    exact le_of_lt (by norm_num)
  exact (program_logic.right h₃).trans h₂

end final_output_value_of_m_l5_5563


namespace tiffany_total_bags_l5_5023

-- Define the initial and additional bags correctly
def bags_on_monday : ℕ := 10
def bags_next_day : ℕ := 3
def bags_day_after : ℕ := 7

-- Define the total bags calculation
def total_bags (initial : ℕ) (next : ℕ) (after : ℕ) : ℕ :=
  initial + next + after

-- Prove that the total bags collected is 20
theorem tiffany_total_bags : total_bags bags_on_monday bags_next_day bags_day_after = 20 :=
by
  sorry

end tiffany_total_bags_l5_5023


namespace tan_150_degrees_l5_5920

theorem tan_150_degrees : Real.tan (150 * Real.pi / 180) = -Real.sqrt 3 / 3 := by
  sorry

end tan_150_degrees_l5_5920


namespace lioness_hyena_age_ratio_l5_5698

variables {k H : ℕ}

-- Conditions
def lioness_age (lioness_age hyena_age : ℕ) : Prop := ∃ k, lioness_age = k * hyena_age
def lioness_is_12 (lioness_age : ℕ) : Prop := lioness_age = 12
def baby_age (mother_age baby_age : ℕ) : Prop := baby_age = mother_age / 2
def baby_ages_sum_in_5_years (baby_l_age baby_h_age sum : ℕ) : Prop := 
  (baby_l_age + 5) + (baby_h_age + 5) = sum

-- The statement to be proved
theorem lioness_hyena_age_ratio (H : ℕ)
  (h1 : lioness_age 12 H) 
  (h2 : baby_age 12 6) 
  (h3 : baby_age H (H / 2)) 
  (h4 : baby_ages_sum_in_5_years 6 (H / 2) 19) : 12 / H = 2 := 
sorry

end lioness_hyena_age_ratio_l5_5698


namespace value_of_a_l5_5204

-- Definition of the function and the point
def graph_function (x : ℝ) : ℝ := -x^2
def point_lies_on_graph (a : ℝ) : Prop := (a, -9) ∈ {p : ℝ × ℝ | p.2 = graph_function p.1}

-- The theorem stating that if the point (a, -9) lies on the graph of y = -x^2, then a = ±3
theorem value_of_a (a : ℝ) (h : point_lies_on_graph a) : a = 3 ∨ a = -3 :=
by 
  sorry

end value_of_a_l5_5204


namespace div_by_6_l5_5837

theorem div_by_6 (m : ℕ) : 6 ∣ (m^3 + 11 * m) :=
sorry

end div_by_6_l5_5837


namespace selected_female_athletes_l5_5845

-- Definitions based on conditions
def total_male_athletes := 56
def total_female_athletes := 42
def selected_male_athletes := 8
def male_to_female_ratio := 4 / 3

-- Problem statement: Prove that the number of selected female athletes is 6
theorem selected_female_athletes :
  selected_male_athletes * (3 / 4) = 6 :=
by 
  -- Placeholder for the proof
  sorry

end selected_female_athletes_l5_5845


namespace math_problem_l5_5768

theorem math_problem
  (a b c : ℝ)
  (h : a / (30 - a) + b / (70 - b) + c / (80 - c) = 8) :
  6 / (30 - a) + 14 / (70 - b) + 16 / (80 - c) = 5 :=
sorry

end math_problem_l5_5768


namespace largest_divisible_by_88_l5_5869

theorem largest_divisible_by_88 (n : ℕ) (h₁ : n = 9999) (h₂ : n % 88 = 55) : n - 55 = 9944 := by
  sorry

end largest_divisible_by_88_l5_5869


namespace path_area_and_cost_correct_l5_5490

def length_field : ℝ := 75
def width_field : ℝ := 55
def path_width : ℝ := 2.8
def area_of_path : ℝ := 759.36
def cost_per_sqm : ℝ := 2
def total_cost : ℝ := 1518.72

theorem path_area_and_cost_correct :
    let length_with_path := length_field + 2 * path_width
    let width_with_path := width_field + 2 * path_width
    let area_with_path := length_with_path * width_with_path
    let area_field := length_field * width_field
    let calculated_area_of_path := area_with_path - area_field
    let calculated_total_cost := calculated_area_of_path * cost_per_sqm
    calculated_area_of_path = area_of_path ∧ calculated_total_cost = total_cost :=
by
    sorry

end path_area_and_cost_correct_l5_5490


namespace pencil_cost_l5_5539

theorem pencil_cost 
  (x y : ℚ)
  (h1 : 3 * x + 2 * y = 165)
  (h2 : 4 * x + 7 * y = 303) :
  y = 19.155 := 
by
  sorry

end pencil_cost_l5_5539


namespace shortest_altitude_of_right_triangle_l5_5214

theorem shortest_altitude_of_right_triangle
  (a b c : ℝ)
  (ha : a = 9) 
  (hb : b = 12) 
  (hc : c = 15)
  (ht : a^2 + b^2 = c^2) :
  ∃ h : ℝ, (1 / 2) * c * h = (1 / 2) * a * b ∧ h = 7.2 := by
  sorry

end shortest_altitude_of_right_triangle_l5_5214


namespace function_decreasing_in_interval_l5_5006

theorem function_decreasing_in_interval :
  ∀ (x1 x2 : ℝ), (0 < x1 ∧ 0 < x2 ∧ x1 ≠ x2) → 
  (x1 - x2) * ((1 / x1 - x1) - (1 / x2 - x2)) < 0 :=
by
  intros x1 x2 hx
  sorry

end function_decreasing_in_interval_l5_5006


namespace parallel_lines_l5_5980

open Real -- Open the real number namespace

/-- Definition of line l1 --/
def line_l1 (a : ℝ) (x y : ℝ) := a * x + 2 * y - 1 = 0

/-- Definition of line l2 --/
def line_l2 (a : ℝ) (x y : ℝ) := x + (a + 1) * y + 4 = 0

/-- The proof statement --/
theorem parallel_lines (a : ℝ) : (a = 1) → (line_l1 a x y) → (line_l2 a x y) := 
sorry

end parallel_lines_l5_5980


namespace total_weight_is_correct_l5_5266

noncomputable def A (B : ℝ) : ℝ := 12 + (1/2) * B
noncomputable def B (C : ℝ) : ℝ := 8 + (1/3) * C
noncomputable def C (A : ℝ) : ℝ := 20 + 2 * A
noncomputable def NewWeightB (A B : ℝ) : ℝ := B + 0.15 * A
noncomputable def NewWeightA (A C : ℝ) : ℝ := A - 0.10 * C

theorem total_weight_is_correct (B C : ℝ) (h1 : A B = (C - 20) / 2)
  (h2 : B = 8 + (1/3) * C) 
  (h3 : C = 20 + 2 * A B) 
  (h4 : NewWeightB (A B) B = 38.35) 
  (h5 : NewWeightA (A B) C = 21.2) :
  NewWeightA (A B) C + NewWeightB (A B) B + C = 139.55 :=
sorry

end total_weight_is_correct_l5_5266


namespace convert_to_scientific_notation_l5_5863

theorem convert_to_scientific_notation :
  (448000 : ℝ) = 4.48 * 10^5 :=
by
  sorry

end convert_to_scientific_notation_l5_5863


namespace intersection_counts_l5_5537

theorem intersection_counts (f g h : ℝ → ℝ)
  (hf : ∀ x, f x = -x^2 + 4 * x - 3)
  (hg : ∀ x, g x = -f x)
  (hh : ∀ x, h x = f (-x))
  (c : ℕ) (hc : c = 2)
  (d : ℕ) (hd : d = 1):
  10 * c + d = 21 :=
by
  sorry

end intersection_counts_l5_5537


namespace find_n_cosine_l5_5940

theorem find_n_cosine : ∃ (n : ℤ), -180 ≤ n ∧ n ≤ 180 ∧ (∃ m : ℤ, n = 25 + 360 * m ∨ n = -25 + 360 * m) :=
by
  sorry

end find_n_cosine_l5_5940


namespace find_radius_l5_5884

-- Define the given conditions as variables
variables (l A r : ℝ)

-- Conditions from the problem
-- 1. The arc length of the sector is 2 cm
def arc_length_eq : Prop := l = 2

-- 2. The area of the sector is 2 cm²
def area_eq : Prop := A = 2

-- Formula for the area of the sector
def sector_area (l r : ℝ) : ℝ := 0.5 * l * r

-- Define the goal to prove the radius is 2 cm
theorem find_radius (h₁ : arc_length_eq l) (h₂ : area_eq A) : r = 2 :=
by {
  sorry -- proof omitted
}

end find_radius_l5_5884


namespace find_c_l5_5603

theorem find_c (c : ℝ) : (∀ x : ℝ, -2 < x ∧ x < 1 → x^2 + x - c < 0) → c = 2 :=
by
  intros h
  -- Sorry to skip the proof
  sorry

end find_c_l5_5603


namespace quadrilateral_area_l5_5001

theorem quadrilateral_area 
  (AB BC DC : ℝ)
  (hAB_perp_BC : true)
  (hDC_perp_BC : true)
  (hAB_eq : AB = 8)
  (hDC_eq : DC = 3)
  (hBC_eq : BC = 10) : 
  (1 / 2 * (AB + DC) * BC = 55) :=
by 
  sorry

end quadrilateral_area_l5_5001


namespace problem_statement_l5_5254

theorem problem_statement (a b c : ℝ) (h1 : 0 < a) (h2 : a < 1) (h3 : 0 < b) (h4 : b < 1) (h5 : 0 < c) (h6 : c < 1) :
  ¬ ((1 - a) * b > 1/4 ∧ (1 - b) * c > 1/4 ∧ (1 - c) * a > 1/4) :=
sorry

end problem_statement_l5_5254


namespace original_faculty_members_l5_5702

theorem original_faculty_members (X : ℝ) (H0 : X > 0) 
  (H1 : 0.75 * X ≤ X)
  (H2 : ((0.75 * X + 35) * 1.10 * 0.80 = 195)) :
  X = 253 :=
by {
  sorry
}

end original_faculty_members_l5_5702


namespace model_N_completion_time_l5_5902

variable (T : ℕ)

def model_M_time : ℕ := 36
def number_of_M_computers : ℕ := 12
def number_of_N_computers := number_of_M_computers -- given that they are the same.

-- Statement of the problem: Given the conditions, prove T = 18
theorem model_N_completion_time :
  (number_of_M_computers : ℝ) * (1 / model_M_time) + (number_of_N_computers : ℝ) * (1 / T) = 1 →
  T = 18 :=
by
  sorry

end model_N_completion_time_l5_5902


namespace scientific_notation_of_2200_l5_5178

-- Define scientific notation criteria
def is_scientific_notation (a : ℝ) (n : ℤ) (x : ℝ) : Prop :=
  x = a * 10^n ∧ 1 ≤ a ∧ a < 10

-- Problem statement
theorem scientific_notation_of_2200 : ∃ (a : ℝ) (n : ℤ), is_scientific_notation a n 2200 ∧ a = 2.2 ∧ n = 3 :=
by {
  -- Proof can be added here.
  sorry
}

end scientific_notation_of_2200_l5_5178


namespace total_ideal_matching_sets_l5_5728

-- Definitions based on the provided problem statement
def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def is_ideal_matching_set (A B : Set ℕ) : Prop := A ∩ B = {1, 3, 5}

-- Theorem statement for the total number of ideal matching sets
theorem total_ideal_matching_sets : ∃ n, n = 27 ∧ ∀ (A B : Set ℕ), A ⊆ U ∧ B ⊆ U ∧ is_ideal_matching_set A B → n = 27 := 
sorry

end total_ideal_matching_sets_l5_5728


namespace ratio_of_segments_l5_5076

theorem ratio_of_segments (a b c r s : ℝ) (h : a / b = 1 / 4)
  (h₁ : c ^ 2 = a ^ 2 + b ^ 2)
  (h₂ : r = a ^ 2 / c)
  (h₃ : s = b ^ 2 / c) :
  r / s = 1 / 16 :=
by
  sorry

end ratio_of_segments_l5_5076


namespace exercise_l5_5403

open Set

theorem exercise (U A B : Set ℕ) (hU : U = {0, 1, 2, 3, 4, 5, 6}) (hA : A = {1, 3, 5}) (hB : B = {2, 4, 5}) :
  A ∩ (U \ B) = {1, 3} := by
  sorry

end exercise_l5_5403


namespace seeds_per_flowerbed_l5_5932

theorem seeds_per_flowerbed (total_seeds : ℕ) (flowerbeds : ℕ) (seeds_per_bed : ℕ) 
  (h1 : total_seeds = 45) (h2 : flowerbeds = 9) 
  (h3 : total_seeds = flowerbeds * seeds_per_bed) : seeds_per_bed = 5 :=
by sorry

end seeds_per_flowerbed_l5_5932


namespace triangle_ABC_area_l5_5132

-- definition of points A, B, and C
def A : (ℝ × ℝ) := (0, 2)
def B : (ℝ × ℝ) := (6, 0)
def C : (ℝ × ℝ) := (3, 7)

-- helper function to calculate area of triangle given vertices
def triangle_area (A B C : ℝ × ℝ) : ℝ :=
  0.5 * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))

theorem triangle_ABC_area :
  triangle_area A B C = 18 := by
  sorry

end triangle_ABC_area_l5_5132


namespace domain_ln_x_plus_one_l5_5174

theorem domain_ln_x_plus_one :
  ∀ (x : ℝ), ∃ (y : ℝ), y = Real.log (x + 1) ↔ x > -1 :=
by sorry

end domain_ln_x_plus_one_l5_5174


namespace initial_horses_to_cows_ratio_l5_5008

theorem initial_horses_to_cows_ratio (H C : ℕ) (h₁ : (H - 15) / (C + 15) = 13 / 7) (h₂ : H - 15 = C + 45) :
  H / C = 4 / 1 := 
sorry

end initial_horses_to_cows_ratio_l5_5008


namespace find_a_solution_l5_5411

open Complex

noncomputable def find_a : Prop := 
  ∃ a : ℂ, ((1 + a * I) / (2 + I) = 1 + 2 * I) ∧ (a = 5 + I)

theorem find_a_solution : find_a := 
  by
    sorry

end find_a_solution_l5_5411


namespace number_subtracted_l5_5386

theorem number_subtracted (x : ℝ) : 3 + 2 * (8 - x) = 24.16 → x = -2.58 :=
by
  intro h
  sorry

end number_subtracted_l5_5386


namespace probability_C_l5_5956

-- Variables representing the probabilities of each region
variables (P_A P_B P_C P_D P_E : ℚ)

-- Given conditions
def conditions := P_A = 3/10 ∧ P_B = 1/4 ∧ P_D = 1/5 ∧ P_E = 1/10 ∧ P_A + P_B + P_C + P_D + P_E = 1

-- The statement to prove
theorem probability_C (h : conditions P_A P_B P_C P_D P_E) : P_C = 3/20 := 
by
  sorry

end probability_C_l5_5956


namespace correct_option_l5_5371

-- Definitions of the options as Lean statements
def optionA : Prop := (-1 : ℝ) / 6 > (-1 : ℝ) / 7
def optionB : Prop := (-4 : ℝ) / 3 < (-3 : ℝ) / 2
def optionC : Prop := (-2 : ℝ)^3 = -2^3
def optionD : Prop := -(-4.5 : ℝ) > abs (-4.6 : ℝ)

-- Theorem stating that optionC is the correct statement among the provided options
theorem correct_option : optionC :=
by
  unfold optionC
  rw [neg_pow, neg_pow, pow_succ, pow_succ]
  sorry  -- The proof is omitted as per instructions

end correct_option_l5_5371


namespace correspond_half_l5_5840

theorem correspond_half (m n : ℕ) 
  (H : ∀ h : Fin m, ∃ g_set : Finset (Fin n), (g_set.card = n / 2) ∧ (∀ g : Fin n, g ∈ g_set))
  (G : ∀ g : Fin n, ∃ h_set : Finset (Fin m), (h_set.card ≤ m / 2) ∧ (∀ h : Fin m, h ∈ h_set)) :
  (∀ h : Fin m, ∀ g_set : Finset (Fin n), g_set.card = n / 2) ∧ (∀ g : Fin n, ∀ h_set : Finset (Fin m), h_set.card = m / 2) :=
by
  sorry

end correspond_half_l5_5840


namespace minimum_distance_l5_5899

theorem minimum_distance (x y : ℝ) (h : x - y - 1 = 0) : (x - 2)^2 + (y - 2)^2 ≥ 1 / 2 :=
sorry

end minimum_distance_l5_5899


namespace exists_unique_solution_l5_5753

theorem exists_unique_solution : ∀ a b : ℝ, 2 * (a ^ 2 + 1) * (b ^ 2 + 1) = (a + 1) * (b + 1) * (a * b + 1) ↔ (a, b) = (1, 1) := by
  sorry

end exists_unique_solution_l5_5753


namespace value_of_y_l5_5458

theorem value_of_y : exists y : ℝ, (∀ k : ℝ, (∀ x y : ℝ, x = k / y^2 → (x = 1 → y = 2 → k = 4)) ∧ (x = 0.1111111111111111 → k = 4 → y = 6)) := by
  sorry

end value_of_y_l5_5458


namespace student_made_mistake_l5_5582

theorem student_made_mistake (AB CD MLNKT : ℕ) (h1 : 10 ≤ AB ∧ AB ≤ 99) (h2 : 10 ≤ CD ∧ CD ≤ 99) (h3 : 10000 ≤ MLNKT ∧ MLNKT < 100000) : AB * CD ≠ MLNKT :=
by {
  sorry
}

end student_made_mistake_l5_5582


namespace find_t_l5_5500

theorem find_t (k m r s t : ℕ) (h1 : k < m) (h2 : m < r) (h3 : r < s) (h4 : s < t)
    (havg : (k + m + r + s + t) / 5 = 18)
    (hmed : r = 23) 
    (hpos_k : 0 < k)
    (hpos_m : 0 < m)
    (hpos_r : 0 < r)
    (hpos_s : 0 < s)
    (hpos_t : 0 < t) :
  t = 40 := sorry

end find_t_l5_5500


namespace product_of_first_two_terms_l5_5542

-- Given parameters
variables (a d : ℤ) -- a is the first term, d is the common difference

-- Conditions
def fifth_term_condition (a d : ℤ) : Prop := a + 4 * d = 11
def common_difference_condition (d : ℤ) : Prop := d = 1

-- Main statement to prove
theorem product_of_first_two_terms (a d : ℤ) (h1 : fifth_term_condition a d) (h2 : common_difference_condition d) :
  a * (a + d) = 56 :=
by
  sorry

end product_of_first_two_terms_l5_5542


namespace relationship_between_number_and_square_l5_5311

theorem relationship_between_number_and_square (n : ℕ) (h : n = 9) :
  (n + n^2) / 2 = 5 * n := by
    sorry

end relationship_between_number_and_square_l5_5311


namespace bookstore_floor_l5_5921

theorem bookstore_floor
  (academy_floor : ℤ)
  (reading_room_floor : ℤ)
  (bookstore_floor : ℤ)
  (h1 : academy_floor = 7)
  (h2 : reading_room_floor = academy_floor + 4)
  (h3 : bookstore_floor = reading_room_floor - 9) :
  bookstore_floor = 2 :=
by
  sorry

end bookstore_floor_l5_5921


namespace find_original_price_l5_5350

variable (original_price : ℝ)
variable (final_price : ℝ) (first_reduction_rate : ℝ) (second_reduction_rate : ℝ)

theorem find_original_price :
  final_price = 15000 →
  first_reduction_rate = 0.30 →
  second_reduction_rate = 0.40 →
  0.42 * original_price = final_price →
  original_price = 35714 := by
  intros h1 h2 h3 h4
  sorry

end find_original_price_l5_5350


namespace average_speed_of_car_l5_5312

-- Definitions of the given conditions
def uphill_speed : ℝ := 30  -- km/hr
def downhill_speed : ℝ := 70  -- km/hr
def uphill_distance : ℝ := 100  -- km
def downhill_distance : ℝ := 50  -- km

-- Required proof statement (with the correct answer derived from the conditions)
theorem average_speed_of_car :
  (uphill_distance + downhill_distance) / 
  ((uphill_distance / uphill_speed) + (downhill_distance / downhill_speed)) = 37.04 := by
  sorry

end average_speed_of_car_l5_5312


namespace geometric_sequence_sum_l5_5589

theorem geometric_sequence_sum (a : ℕ → ℝ) (S_n : ℕ → ℝ) (q : ℝ) :
  (∀ n, a (n+1) = a n * q) → -- geometric sequence condition
  a 2 = 6 → -- first condition
  6 * a 1 + a 3 = 30 → -- second condition
  (∀ n, S_n n = (if q = 2 then 3*(2^n - 1) else if q = 3 then 3^n - 1 else 0)) :=
by intros
   sorry

end geometric_sequence_sum_l5_5589


namespace gcd_98_63_l5_5560

theorem gcd_98_63 : Nat.gcd 98 63 = 7 :=
by
  sorry

end gcd_98_63_l5_5560


namespace intersection_of_sets_l5_5813

def setM : Set ℝ := { x | x^2 - 3 * x - 4 ≤ 0 }
def setN : Set ℝ := { x | Real.log x ≥ 0 }

theorem intersection_of_sets : (setM ∩ setN) = { x | 1 ≤ x ∧ x ≤ 4 } := 
by {
  sorry
}

end intersection_of_sets_l5_5813


namespace raghu_investment_l5_5883

theorem raghu_investment
  (R trishul vishal : ℝ)
  (h1 : trishul = 0.90 * R)
  (h2 : vishal = 0.99 * R)
  (h3 : R + trishul + vishal = 6647) :
  R = 2299.65 :=
by
  sorry

end raghu_investment_l5_5883


namespace range_of_a_l5_5786

theorem range_of_a (a : ℝ) (h : (∀ x1 x2 : ℝ, x1 < x2 → (2 * a - 1) ^ x1 > (2 * a - 1) ^ x2)) :
  1 / 2 < a ∧ a < 1 :=
by
  sorry

end range_of_a_l5_5786


namespace lavinias_son_older_than_daughter_l5_5694

def katies_daughter_age := 12
def lavinias_daughter_age := katies_daughter_age - 10
def lavinias_son_age := 2 * katies_daughter_age

theorem lavinias_son_older_than_daughter :
  lavinias_son_age - lavinias_daughter_age = 22 :=
by
  sorry

end lavinias_son_older_than_daughter_l5_5694


namespace cylinder_twice_volume_l5_5245

theorem cylinder_twice_volume :
  let r := 8
  let h1 := 10
  let h2 := 20
  let V := (pi * r^2 * h1)
  let V_desired := 2 * V
  V_desired = pi * r^2 * h2 :=
by
  let r := 8
  let h1 := 10
  let h2 := 20
  let V := (pi * r^2 * h1)
  let V_desired := 2 * V
  show V_desired = pi * r^2 * h2
  sorry

end cylinder_twice_volume_l5_5245


namespace range_of_q_eq_eight_inf_l5_5525

noncomputable def q (x : ℝ) : ℝ := (x^2 + 2)^3

theorem range_of_q_eq_eight_inf (x : ℝ) : 0 ≤ x → ∃ y, y = q x ∧ 8 ≤ y := sorry

end range_of_q_eq_eight_inf_l5_5525


namespace evaluate_fraction_l5_5420

theorem evaluate_fraction (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a - b * (1 / a) ≠ 0) :
  (a^2 - 1 / b^2) / (b^2 - 1 / a^2) = a^2 / b^2 :=
by
  sorry

end evaluate_fraction_l5_5420


namespace highlighter_difference_l5_5937

theorem highlighter_difference :
  ∀ (yellow pink blue : ℕ),
    yellow = 7 →
    pink = yellow + 7 →
    yellow + pink + blue = 40 →
    blue - pink = 5 :=
by
  intros yellow pink blue h_yellow h_pink h_total
  rw [h_yellow, h_pink] at h_total
  sorry

end highlighter_difference_l5_5937


namespace simplify_cosine_expression_l5_5950

theorem simplify_cosine_expression :
  ∀ (θ : ℝ), θ = 30 * Real.pi / 180 → (1 - Real.cos θ) * (1 + Real.cos θ) = 1 / 4 :=
by
  intro θ hθ
  have cos_30 := Real.cos θ
  rewrite [hθ]
  sorry

end simplify_cosine_expression_l5_5950


namespace area_of_square_with_perimeter_32_l5_5520

theorem area_of_square_with_perimeter_32 :
  ∀ (s : ℝ), 4 * s = 32 → s * s = 64 :=
by
  intros s h
  sorry

end area_of_square_with_perimeter_32_l5_5520


namespace minimum_value_of_expression_l5_5548

theorem minimum_value_of_expression 
  (a b c : ℝ) 
  (ha : 0 < a) 
  (hb : 0 < b) 
  (hc : 0 < c) 
  (h : 3 * a + 4 * b + 2 * c = 3) : 
  (1 / (2 * a + b) + 1 / (a + 3 * c) + 1 / (4 * b + c)) = 1.5 :=
sorry

end minimum_value_of_expression_l5_5548


namespace coin_flip_sequences_l5_5049

theorem coin_flip_sequences : 2^10 = 1024 := by
  sorry

end coin_flip_sequences_l5_5049


namespace find_general_students_l5_5343

-- Define the conditions and the question
structure Halls :=
  (general : ℕ)
  (biology : ℕ)
  (math : ℕ)
  (total : ℕ)

def conditions_met (h : Halls) : Prop :=
  h.biology = 2 * h.general ∧
  h.math = (3 / 5 : ℚ) * (h.general + h.biology) ∧
  h.total = h.general + h.biology + h.math ∧
  h.total = 144

-- The proof problem statement
theorem find_general_students (h : Halls) (h_cond : conditions_met h) : h.general = 30 :=
sorry

end find_general_students_l5_5343


namespace elementary_school_coats_correct_l5_5316

def total_coats : ℕ := 9437
def high_school_coats : ℕ := (3 * total_coats) / 5
def elementary_school_coats := total_coats - high_school_coats

theorem elementary_school_coats_correct : 
  elementary_school_coats = 3775 :=
by
  sorry

end elementary_school_coats_correct_l5_5316


namespace cylinder_volume_multiplication_factor_l5_5193

theorem cylinder_volume_multiplication_factor (r h : ℝ) (h_r_positive : r > 0) (h_h_positive : h > 0) :
  let V := π * r^2 * h
  let V' := π * (2.5 * r)^2 * (3 * h)
  let X := V' / V
  X = 18.75 :=
by
  -- Proceed with the proof here
  sorry

end cylinder_volume_multiplication_factor_l5_5193


namespace root_poly_ratio_c_d_l5_5039

theorem root_poly_ratio_c_d (a b c d : ℝ)
  (h₁ : 1 + (-2) + 3 = 2)
  (h₂ : 1 * (-2) + (-2) * 3 + 3 * 1 = -5)
  (h₃ : 1 * (-2) * 3 = -6)
  (h_sum : -b / a = 2)
  (h_pair_prod : c / a = -5)
  (h_prod : -d / a = -6) :
  c / d = 5 / 6 := by
  sorry

end root_poly_ratio_c_d_l5_5039


namespace no_both_squares_l5_5829

theorem no_both_squares {x y : ℕ} (hx : x > 0) (hy : y > 0) : ¬ (∃ a b : ℕ, a^2 = x^2 + 2 * y ∧ b^2 = y^2 + 2 * x) :=
by
  sorry

end no_both_squares_l5_5829


namespace sum_of_edges_of_geometric_progression_solid_l5_5435

theorem sum_of_edges_of_geometric_progression_solid
  (a : ℝ)
  (r : ℝ)
  (volume_eq : a^3 = 512)
  (surface_eq : 2 * (64 / r + 64 * r + 64) = 352)
  (r_value : r = 1.25 ∨ r = 0.8) :
  4 * (8 / r + 8 + 8 * r) = 97.6 := by
  sorry

end sum_of_edges_of_geometric_progression_solid_l5_5435


namespace solve_system_of_equations_l5_5143

theorem solve_system_of_equations :
  ∃ (x y : ℕ), (x + 2 * y = 5) ∧ (3 * x + y = 5) ∧ (x = 1) ∧ (y = 2) :=
by {
  sorry
}

end solve_system_of_equations_l5_5143


namespace sum_of_primes_l5_5928

theorem sum_of_primes (p1 p2 p3 : ℕ) (hp1 : Nat.Prime p1) (hp2 : Nat.Prime p2) (hp3 : Nat.Prime p3) 
    (h : p1 * p2 * p3 = 31 * (p1 + p2 + p3)) :
    p1 + p2 + p3 = 51 := by
  sorry

end sum_of_primes_l5_5928


namespace solve_for_x_l5_5864

theorem solve_for_x (x : ℂ) (i : ℂ) (h : i ^ 2 = -1) (eqn : 3 + i * x = 5 - 2 * i * x) : x = i / 3 :=
sorry

end solve_for_x_l5_5864


namespace trader_gain_percentage_l5_5200

theorem trader_gain_percentage 
  (C : ℝ) -- cost of each pen
  (h1 : 250 * C ≠ 0) -- ensure the cost of 250 pens is non-zero
  (h2 : 65 * C > 0) -- ensure the gain is positive
  (h3 : 250 * C + 65 * C > 0) -- ensure the selling price is positive
  : (65 / 250) * 100 = 26 := 
sorry

end trader_gain_percentage_l5_5200


namespace smallest_sum_divisible_by_3_l5_5212

def is_prime (n : ℕ) : Prop := ∀ m, m ∣ n → m = 1 ∨ m = n

def is_consecutive_prime (p1 p2 p3 p4 : ℕ) : Prop :=
  is_prime p1 ∧ is_prime p2 ∧ is_prime p3 ∧ is_prime p4 ∧
  (p2 = p1 + 4 ∨ p2 = p1 + 6 ∨ p2 = p1 + 2) ∧
  (p3 = p2 + 2 ∨ p3 = p2 + 4) ∧
  (p4 = p3 + 2 ∨ p4 = p3 + 4)

def greater_than_5 (p : ℕ) : Prop := p > 5

theorem smallest_sum_divisible_by_3 :
  ∃ (p1 p2 p3 p4 : ℕ), is_consecutive_prime p1 p2 p3 p4 ∧
                      greater_than_5 p1 ∧
                      (p1 + p2 + p3 + p4) % 3 = 0 ∧
                      (p1 + p2 + p3 + p4) = 48 :=
by sorry

end smallest_sum_divisible_by_3_l5_5212


namespace males_only_in_band_l5_5054

theorem males_only_in_band
  (females_in_band : ℕ)
  (males_in_band : ℕ)
  (females_in_orchestra : ℕ)
  (males_in_orchestra : ℕ)
  (females_in_both : ℕ)
  (total_students : ℕ)
  (total_students_in_either : ℕ)
  (hf_in_band : females_in_band = 120)
  (hm_in_band : males_in_band = 90)
  (hf_in_orchestra : females_in_orchestra = 100)
  (hm_in_orchestra : males_in_orchestra = 130)
  (hf_in_both : females_in_both = 80)
  (h_total_students : total_students = 260) :
  total_students_in_either = 260 → 
  (males_in_band - (90 + 130 + 80 - 260 - 120)) = 30 :=
by
  intros h_total_students_in_either
  sorry

end males_only_in_band_l5_5054


namespace find_b_value_l5_5046

noncomputable def find_b (p q : ℕ) : ℕ := p^2 + q^2

theorem find_b_value
  (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q)
  (h_distinct : p ≠ q) (h_roots : p + q = 13 ∧ p * q = 22) :
  find_b p q = 125 :=
by
  sorry

end find_b_value_l5_5046


namespace smallest_percentage_all_correct_l5_5807

theorem smallest_percentage_all_correct (p1 p2 p3 : ℝ) 
  (h1 : p1 = 0.9) 
  (h2 : p2 = 0.8)
  (h3 : p3 = 0.7) :
  ∃ x, x = 0.4 ∧ (x ≤ 1 - ((1 - p1) + (1 - p2) + (1 - p3))) :=
by 
  sorry

end smallest_percentage_all_correct_l5_5807


namespace train_crossing_time_is_correct_l5_5527

-- Define the constant values
def train_length : ℝ := 350        -- Train length in meters
def train_speed : ℝ := 20          -- Train speed in m/s
def crossing_time : ℝ := 17.5      -- Time to cross the signal post in seconds

-- Proving the relationship that the time taken for the train to cross the signal post is as calculated
theorem train_crossing_time_is_correct : (train_length / train_speed) = crossing_time :=
by
  sorry

end train_crossing_time_is_correct_l5_5527


namespace find_c_l5_5804

def p (x : ℝ) := 4 * x - 9
def q (x : ℝ) (c : ℝ) := 5 * x - c

theorem find_c : ∃ (c : ℝ), p (q 3 c) = 14 ∧ c = 9.25 :=
by
  sorry

end find_c_l5_5804


namespace scaled_det_l5_5892

variable (x y z a b c p q r : ℝ)
variable (det_orig : ℝ)
variable (h : Matrix.det ![![x, y, z], ![a, b, c], ![p, q, r]] = 2)

theorem scaled_det (h : Matrix.det ![![x, y, z], ![a, b, c], ![p, q, r]] = 2) :
  Matrix.det ![![3*x, 3*y, 3*z], ![3*a, 3*b, 3*c], ![3*p, 3*q, 3*r]] = 54 :=
by
  sorry

end scaled_det_l5_5892


namespace water_added_l5_5293

theorem water_added (W X : ℝ) 
  (h1 : 45 / W = 2 / 1)
  (h2 : 45 / (W + X) = 6 / 5) : 
  X = 15 := 
by
  sorry

end water_added_l5_5293


namespace inequality_solution_l5_5903

theorem inequality_solution (x : ℝ) (h : x * (x^2 + 1) > (x + 1) * (x^2 - x + 1)) : x > 1 := 
sorry

end inequality_solution_l5_5903


namespace find_k_l5_5999

theorem find_k (a : ℕ → ℕ) (h₀ : a 1 = 2) (h₁ : ∀ m n, a (m + n) = a m * a n) (hk : a (k + 1) = 1024) : k = 9 := 
sorry

end find_k_l5_5999


namespace martha_cakes_l5_5264

theorem martha_cakes :
  ∀ (n : ℕ), (∀ (c : ℕ), c = 3 → (∀ (k : ℕ), k = 6 → n = c * k)) → n = 18 :=
by
  intros n h
  specialize h 3 rfl 6 rfl
  exact h

end martha_cakes_l5_5264


namespace original_number_of_turtles_l5_5148

-- Define the problem
theorem original_number_of_turtles (T : ℕ) (h1 : 17 = (T + 3 * T - 2) / 2) : T = 9 := by
  sorry

end original_number_of_turtles_l5_5148


namespace incorrect_statements_l5_5206

def even_function (f : ℝ → ℝ) : Prop :=
∀ x, f x = f (-x)

def monotonically_decreasing_in_pos (f : ℝ → ℝ) : Prop :=
∀ x y, 0 < x ∧ x < y → f y ≤ f x

theorem incorrect_statements
  (f : ℝ → ℝ)
  (hf_even : even_function f)
  (hf_decreasing : monotonically_decreasing_in_pos f) :
  ¬ (∀ a, f (2 * a) < f (-a)) ∧ ¬ (f π > f (-3)) ∧ ¬ (∀ a, f (a^2 + 1) < f 1) :=
by sorry

end incorrect_statements_l5_5206


namespace integral_sign_l5_5427

noncomputable def I : ℝ := ∫ x in -Real.pi..0, Real.sin x

theorem integral_sign : I < 0 := sorry

end integral_sign_l5_5427


namespace fraction_sum_squares_eq_sixteen_l5_5568

variables (x a y b z c : ℝ)

theorem fraction_sum_squares_eq_sixteen
  (h1 : x / a + y / b + z / c = 4)
  (h2 : a / x + b / y + c / z = 0) :
  (x^2 / a^2 + y^2 / b^2 + z^2 / c^2) = 16 := 
sorry

end fraction_sum_squares_eq_sixteen_l5_5568


namespace relatively_prime_dates_in_september_l5_5854

-- Define a condition to check if two numbers are relatively prime
def relatively_prime (a b : ℕ) : Prop := Nat.gcd a b = 1

-- Define the number of days in September
def days_in_september := 30

-- Define the month of September as the 9th month
def month_of_september := 9

-- Define the proposition that the number of relatively prime dates in September is 20
theorem relatively_prime_dates_in_september : 
  ∃ count, (count = 20 ∧ ∀ day, day ∈ Finset.range (days_in_september + 1) → relatively_prime month_of_september day → count = 20) := sorry

end relatively_prime_dates_in_september_l5_5854


namespace find_line_equation_l5_5112

noncomputable def line_equation (x y : ℝ) : Prop :=
  y = (Real.sqrt 3 / 3) * x - 4

theorem find_line_equation :
  ∃ (x₁ y₁ : ℝ), x₁ = Real.sqrt 3 ∧ y₁ = -3 ∧ ∀ x y, (line_equation x y ↔ 
  (y + 3 = (Real.sqrt 3 / 3) * (x - Real.sqrt 3))) :=
sorry

end find_line_equation_l5_5112


namespace oatmeal_cookies_divisible_by_6_l5_5627

theorem oatmeal_cookies_divisible_by_6 (O : ℕ) (h1 : 48 % 6 = 0) (h2 : O % 6 = 0) :
    ∃ x : ℕ, O = 6 * x :=
by sorry

end oatmeal_cookies_divisible_by_6_l5_5627


namespace return_trip_time_l5_5604

variable {d p w : ℝ} -- Distance, plane's speed in calm air, wind speed

theorem return_trip_time (h1 : d = 75 * (p - w)) 
                         (h2 : d / (p + w) = d / p - 10) :
                         (d / (p + w) = 15 ∨ d / (p + w) = 50) :=
sorry

end return_trip_time_l5_5604


namespace no_preimage_implies_p_gt_1_l5_5561

   noncomputable def f (x : ℝ) : ℝ :=
     -x^2 + 2 * x

   theorem no_preimage_implies_p_gt_1 (p : ℝ) (hp : ∀ x : ℝ, f x ≠ p) : p > 1 :=
   sorry
   
end no_preimage_implies_p_gt_1_l5_5561


namespace reading_order_l5_5762

theorem reading_order (a b c d : ℝ) 
  (h1 : a + c = b + d) 
  (h2 : a + b > c + d)
  (h3 : d > b + c) :
  a > d ∧ d > b ∧ b > c :=
by sorry

end reading_order_l5_5762


namespace monthly_income_of_labourer_l5_5129

variable (I : ℕ) -- Monthly income

-- Conditions: 
def condition1 := (85 * 6) - (6 * I) -- A boolean expression depicting the labourer fell into debt
def condition2 := (60 * 4) + (85 * 6 - 6 * I) + 30 -- Total income covers debt and saving 30

-- Statement to be proven
theorem monthly_income_of_labourer : 
  ∃ I : ℕ, condition1 I = 0 ∧ condition2 I = 4 * I → I = 78 :=
by
  sorry

end monthly_income_of_labourer_l5_5129


namespace range_of_expression_positive_range_of_expression_negative_l5_5450

theorem range_of_expression_positive (x : ℝ) : 
  (2 * x ^ 2 - 5 * x - 12 > 0) ↔ (x < -3/2 ∨ x > 4) :=
sorry

theorem range_of_expression_negative (x : ℝ) : 
  (2 * x ^ 2 - 5 * x - 12 < 0) ↔ ( -3/2 < x ∧ x < 4) :=
sorry

end range_of_expression_positive_range_of_expression_negative_l5_5450


namespace alice_score_l5_5462

variables (correct_answers wrong_answers unanswered_questions : ℕ)
variables (points_correct points_incorrect : ℚ)

def compute_score (correct_answers wrong_answers : ℕ) (points_correct points_incorrect : ℚ) : ℚ :=
    (correct_answers : ℚ) * points_correct + (wrong_answers : ℚ) * points_incorrect

theorem alice_score : 
    correct_answers = 15 → 
    wrong_answers = 5 → 
    unanswered_questions = 10 → 
    points_correct = 1 → 
    points_incorrect = -0.25 → 
    compute_score 15 5 1 (-0.25) = 13.75 := 
by intros; sorry

end alice_score_l5_5462


namespace greatest_radius_l5_5860

theorem greatest_radius (A : ℝ) (hA : A < 60 * Real.pi) : ∃ r : ℕ, r = 7 ∧ (r : ℝ) * (r : ℝ) < 60 :=
by
  sorry

end greatest_radius_l5_5860


namespace evaluate_product_eq_l5_5191

noncomputable def w : ℂ := Complex.exp (2 * Real.pi * Complex.I / 13)

theorem evaluate_product_eq : 
  (3 - w) * (3 - w^2) * (3 - w^3) * (3 - w^4) * (3 - w^5) * (3 - w^6) *
  (3 - w^7) * (3 - w^8) * (3 - w^9) * (3 - w^10) * (3 - w^11) * (3 - w^12) = 885735 := 
sorry

end evaluate_product_eq_l5_5191


namespace find_k_l5_5229

noncomputable def expr_to_complete_square (x : ℝ) : ℝ :=
  x^2 - 6 * x

theorem find_k (x : ℝ) : ∃ a h k, expr_to_complete_square x = a * (x - h)^2 + k ∧ k = -9 :=
by
  use 1, 3, -9
  -- detailed steps of the proof would go here
  sorry

end find_k_l5_5229


namespace slices_per_person_is_correct_l5_5876

-- Conditions
def slices_per_tomato : Nat := 8
def total_tomatoes : Nat := 20
def people_for_meal : Nat := 8

-- Calculate number of slices for a single person
def slices_needed_for_single_person (slices_per_tomato : Nat) (total_tomatoes : Nat) (people_for_meal : Nat) : Nat :=
  (slices_per_tomato * total_tomatoes) / people_for_meal

-- The statement to be proved
theorem slices_per_person_is_correct : slices_needed_for_single_person slices_per_tomato total_tomatoes people_for_meal = 20 :=
by
  sorry

end slices_per_person_is_correct_l5_5876


namespace solve_asterisk_l5_5226

theorem solve_asterisk (x : ℝ) (h : (x / 21) * (x / 84) = 1) : x = 42 :=
sorry

end solve_asterisk_l5_5226


namespace trigonometric_identity_l5_5494

theorem trigonometric_identity (α : ℝ) (h : Real.tan α = 2) : 
  (Real.sin α + 2 * Real.cos α) / (Real.sin α - Real.cos α) = 4 :=
by 
  sorry

end trigonometric_identity_l5_5494


namespace rational_solutions_quadratic_eq_l5_5541

theorem rational_solutions_quadratic_eq (k : ℕ) (h_pos : k > 0) :
  (∃ x : ℚ, k * x^2 + 24 * x + k = 0) ↔ (k = 8 ∨ k = 12) :=
by sorry

end rational_solutions_quadratic_eq_l5_5541


namespace fraction_of_number_l5_5152

theorem fraction_of_number : (7 / 8) * 64 = 56 := by
  sorry

end fraction_of_number_l5_5152


namespace people_on_bus_before_stop_l5_5713

variable (P_before P_after P_got_on : ℕ)
variable (h1 : P_got_on = 13)
variable (h2 : P_after = 17)

theorem people_on_bus_before_stop : P_before = 4 :=
by
  -- Given that P_after = 17 and P_got_on = 13
  -- We need to prove P_before = P_after - P_got_on = 4
  sorry

end people_on_bus_before_stop_l5_5713


namespace scientific_notation_of_203000_l5_5567

-- Define the number
def n : ℝ := 203000

-- Define the representation of the number in scientific notation
def scientific_notation (a b : ℝ) : Prop := n = a * 10^b ∧ 1 ≤ a ∧ a < 10

-- The theorem to state 
theorem scientific_notation_of_203000 : ∃ a b : ℝ, scientific_notation a b ∧ a = 2.03 ∧ b = 5 :=
by
  use 2.03
  use 5
  sorry

end scientific_notation_of_203000_l5_5567


namespace remainder_eq_27_l5_5771

def p (x : ℝ) : ℝ := x^4 + 2 * x^2 + 3
def a : ℝ := -2
def remainder := p (-2)
theorem remainder_eq_27 : remainder = 27 :=
by
  sorry

end remainder_eq_27_l5_5771


namespace solve_for_q_l5_5886

theorem solve_for_q (m n q : ℕ) (h1 : 7/8 = m/96) (h2 : 7/8 = (n + m)/112) (h3 : 7/8 = (q - m)/144) :
  q = 210 :=
sorry

end solve_for_q_l5_5886


namespace arithmetic_sequence_middle_term_l5_5692

theorem arithmetic_sequence_middle_term :
  let a1 := 3^2
  let a3 := 3^4
  let y := (a1 + a3) / 2
  y = 45 :=
by
  let a1 := (3:ℕ)^2
  let a3 := (3:ℕ)^4
  let y := (a1 + a3) / 2
  have : a1 = 9 := by norm_num
  have : a3 = 81 := by norm_num
  have : y = 45 := by norm_num
  exact this

end arithmetic_sequence_middle_term_l5_5692


namespace sufficient_but_not_necessary_condition_l5_5019

theorem sufficient_but_not_necessary_condition 
(a b : ℝ) : (b ≥ 0) → ((a + 1)^2 + b ≥ 0) ∧ (¬ (∀ a b, ((a + 1)^2 + b ≥ 0) → b ≥ 0)) :=
by sorry

end sufficient_but_not_necessary_condition_l5_5019


namespace product_divisible_by_14_l5_5842

theorem product_divisible_by_14 (a b c d : ℤ) (h : 7 * a + 8 * b = 14 * c + 28 * d) : 14 ∣ a * b := 
sorry

end product_divisible_by_14_l5_5842


namespace number_of_letters_l5_5459

-- Definitions and Conditions, based on the given problem
variables (n : ℕ) -- n is the number of different letters in the local language

-- Given: The people have lost 129 words due to the prohibition of the seventh letter
def words_lost_due_to_prohibition (n : ℕ) : ℕ := 2 * n

-- The main theorem to prove
theorem number_of_letters (h : 129 = words_lost_due_to_prohibition n) : n = 65 :=
by sorry

end number_of_letters_l5_5459


namespace radius_of_smaller_circle_l5_5460

theorem radius_of_smaller_circle (R r : ℝ) (h1 : R = 6)
  (h2 : 2 * R = 3 * 2 * r) : r = 2 :=
by
  sorry

end radius_of_smaller_circle_l5_5460


namespace sqrt_value_l5_5113

theorem sqrt_value {A B C : ℝ} (x y : ℝ) 
  (h1 : A = 5 * Real.sqrt (2 * x + 1)) 
  (h2 : B = 3 * Real.sqrt (x + 3)) 
  (h3 : C = Real.sqrt (10 * x + 3 * y)) 
  (h4 : A + B = C) 
  (h5 : 2 * x + 1 = x + 3) : 
  Real.sqrt (2 * y - x^2) = 14 :=
by
  sorry

end sqrt_value_l5_5113


namespace fraction_of_remaining_prize_money_each_winner_receives_l5_5031

-- Definitions based on conditions
def total_prize_money : ℕ := 2400
def first_winner_fraction : ℚ := 1 / 3
def each_following_winner_prize : ℕ := 160

-- Calculate the first winner's prize
def first_winner_prize : ℚ := first_winner_fraction * total_prize_money

-- Calculate the remaining prize money after the first winner
def remaining_prize_money : ℚ := total_prize_money - first_winner_prize

-- Calculate the fraction of the remaining prize money that each of the next ten winners will receive
def following_winner_fraction : ℚ := each_following_winner_prize / remaining_prize_money

-- Theorem statement
theorem fraction_of_remaining_prize_money_each_winner_receives :
  following_winner_fraction = 1 / 10 :=
sorry

end fraction_of_remaining_prize_money_each_winner_receives_l5_5031


namespace front_view_l5_5228

def first_column_heights := [3, 2]
def middle_column_heights := [1, 4, 2]
def third_column_heights := [5]

theorem front_view (h1 : first_column_heights = [3, 2])
                   (h2 : middle_column_heights = [1, 4, 2])
                   (h3 : third_column_heights = [5]) :
    [3, 4, 5] = [
        first_column_heights.foldr max 0,
        middle_column_heights.foldr max 0,
        third_column_heights.foldr max 0
    ] :=
    sorry

end front_view_l5_5228


namespace smallest_number_of_students_l5_5235

theorem smallest_number_of_students 
  (ninth_to_seventh : ℕ → ℕ → Prop)
  (ninth_to_sixth : ℕ → ℕ → Prop) 
  (r1 : ninth_to_seventh 3 2) 
  (r2 : ninth_to_sixth 7 4) : 
  ∃ n7 n6 n9, 
    ninth_to_seventh n9 n7 ∧ 
    ninth_to_sixth n9 n6 ∧ 
    n9 + n7 + n6 = 47 :=
sorry

end smallest_number_of_students_l5_5235


namespace ticket_value_unique_l5_5373

theorem ticket_value_unique (x : ℕ) (h₁ : ∃ n, n > 0 ∧ x * n = 60)
  (h₂ : ∃ m, m > 0 ∧ x * m = 90)
  (h₃ : ∃ p, p > 0 ∧ x * p = 49) : 
  ∃! x, x = 1 :=
by
  sorry

end ticket_value_unique_l5_5373


namespace find_n_for_sine_equality_l5_5650

theorem find_n_for_sine_equality : 
  ∃ (n: ℤ), -90 ≤ n ∧ n ≤ 90 ∧ Real.sin (n * Real.pi / 180) = Real.sin (670 * Real.pi / 180) ∧ n = -50 := by
  sorry

end find_n_for_sine_equality_l5_5650


namespace ab_greater_than_a_plus_b_l5_5787

theorem ab_greater_than_a_plus_b (a b : ℝ) (ha : a ≥ 2) (hb : b > 2) : a * b > a + b :=
by
  sorry

end ab_greater_than_a_plus_b_l5_5787


namespace fixed_point_of_parabolas_l5_5531

theorem fixed_point_of_parabolas 
  (t : ℝ) 
  (fixed_x fixed_y : ℝ) 
  (hx : fixed_x = 2) 
  (hy : fixed_y = 12) 
  (H : ∀ t : ℝ, ∃ y : ℝ, y = 3 * fixed_x^2 + t * fixed_x - 2 * t) : 
  ∃ y : ℝ, y = fixed_y :=
by
  sorry

end fixed_point_of_parabolas_l5_5531


namespace bids_per_person_l5_5773

theorem bids_per_person (initial_price final_price price_increase_per_bid : ℕ) (num_people : ℕ)
  (h1 : initial_price = 15) (h2 : final_price = 65) (h3 : price_increase_per_bid = 5) (h4 : num_people = 2) :
  (final_price - initial_price) / price_increase_per_bid / num_people = 5 :=
  sorry

end bids_per_person_l5_5773


namespace num_balls_in_box_l5_5256

theorem num_balls_in_box (n : ℕ) (h1: 9 <= n) (h2: (9 : ℝ) / n = 0.30) : n = 30 :=
sorry

end num_balls_in_box_l5_5256


namespace balloons_lost_l5_5855

-- Definitions corresponding to the conditions
def initial_balloons : ℕ := 7
def current_balloons : ℕ := 4

-- The mathematically equivalent proof problem
theorem balloons_lost : initial_balloons - current_balloons = 3 := by
  -- proof steps would go here, but we use sorry to skip them 
  sorry

end balloons_lost_l5_5855


namespace gamma_max_success_ratio_l5_5879

theorem gamma_max_success_ratio (x y z w : ℕ) (h_yw : y + w = 500)
    (h_gamma_first_day : 0 < x ∧ x < 170 * y / 280)
    (h_gamma_second_day : 0 < z ∧ z < 150 * w / 220)
    (h_less_than_500 : (28 * x + 22 * z) / 17 < 500) :
    (x + z) ≤ 170 := 
sorry

end gamma_max_success_ratio_l5_5879


namespace find_a_l5_5785

theorem find_a (a r s : ℝ) (h1 : r^2 = a) (h2 : 2 * r * s = 24) (h3 : s^2 = 9) : a = 16 :=
sorry

end find_a_l5_5785


namespace average_of_values_l5_5003

theorem average_of_values (z : ℝ) : 
  (0 + 1 + 2 + 4 + 8 + 32 : ℝ) * z / (6 : ℝ) = 47 * z / 6 :=
by
  sorry

end average_of_values_l5_5003


namespace find_a_plus_b_l5_5286

theorem find_a_plus_b (a b : ℝ) :
  (∀ x : ℝ, x^2 + (a+1)*x + ab = 0 → (x = -1 ∨ x = 4)) → a + b = -3 :=
by
  sorry

end find_a_plus_b_l5_5286


namespace total_sticks_needed_l5_5781

theorem total_sticks_needed :
  let simon_sticks := 36
  let gerry_sticks := 2 * (simon_sticks / 3)
  let total_simon_and_gerry := simon_sticks + gerry_sticks
  let micky_sticks := total_simon_and_gerry + 9
  total_simon_and_gerry + micky_sticks = 129 :=
by
  sorry

end total_sticks_needed_l5_5781


namespace decimal_zeros_l5_5606

theorem decimal_zeros (h : 2520 = 2^3 * 3^2 * 5 * 7) : 
  ∃ (n : ℕ), n = 2 ∧ (∃ d : ℚ, d = 5 / 2520 ∧ ↑d = 0.004) :=
by
  -- We assume the factorization of 2520 is correct
  have h_fact := h
  -- We need to prove there are exactly 2 zeros between the decimal point and the first non-zero digit
  sorry

end decimal_zeros_l5_5606


namespace simplify_expression_l5_5381

theorem simplify_expression (x y : ℝ) : 3 * y - 5 * x + 2 * y + 4 * x = 5 * y - x :=
by
  sorry

end simplify_expression_l5_5381


namespace tangerines_times_persimmons_l5_5910

-- Definitions from the problem conditions
def apples : ℕ := 24
def tangerines : ℕ := 6 * apples
def persimmons : ℕ := 8

-- Statement to be proved
theorem tangerines_times_persimmons :
  tangerines / persimmons = 18 := by
  sorry

end tangerines_times_persimmons_l5_5910


namespace expected_value_die_l5_5236

noncomputable def expected_value (P_Star P_Moon : ℚ) (win_Star lose_Moon : ℚ) : ℚ :=
  P_Star * win_Star + P_Moon * lose_Moon

theorem expected_value_die :
  expected_value (2/5) (3/5) 4 (-3) = -1/5 := by
  sorry

end expected_value_die_l5_5236


namespace problem1_problem2_problem3_l5_5556

-- Problem Conditions
def inductive_reasoning (s: Sort _) (g: Sort _) : Prop := 
  ∀ (x: s → g), true 

def probabilistic_conclusion : Prop :=
  ∀ (x : Prop), true

def analogical_reasoning (a: Sort _) : Prop := 
  ∀ (x: a), true 

-- The Statements to be Proved
theorem problem1 : ¬ inductive_reasoning Prop Prop = true := 
sorry

theorem problem2 : probabilistic_conclusion = true :=
sorry 

theorem problem3 : ¬ analogical_reasoning Prop = true :=
sorry 

end problem1_problem2_problem3_l5_5556


namespace minimum_score_4th_quarter_l5_5296

theorem minimum_score_4th_quarter (q1 q2 q3 : ℕ) (q4 : ℕ) :
  q1 = 85 → q2 = 80 → q3 = 90 →
  (q1 + q2 + q3 + q4) / 4 ≥ 85 →
  q4 ≥ 85 :=
by intros hq1 hq2 hq3 h_avg
   sorry

end minimum_score_4th_quarter_l5_5296


namespace reflection_across_x_axis_l5_5983

theorem reflection_across_x_axis (x y : ℝ) : 
  (x, -y) = (-2, -4) ↔ (x, y) = (-2, 4) :=
by
  sorry

end reflection_across_x_axis_l5_5983


namespace train_speed_is_252_144_l5_5473

/-- Train and pedestrian problem setup -/
noncomputable def train_speed (train_length : ℕ) (cross_time : ℕ) (man_speed_kmph : ℕ) : ℝ :=
  let man_speed_mps := (man_speed_kmph : ℝ) * 1000 / 3600
  let relative_speed_mps := (train_length : ℝ) / (cross_time : ℝ)
  let train_speed_mps := relative_speed_mps - man_speed_mps
  train_speed_mps * 3600 / 1000

theorem train_speed_is_252_144 :
  train_speed 500 7 5 = 252.144 := by
  sorry

end train_speed_is_252_144_l5_5473


namespace number_of_polynomials_is_seven_l5_5052

-- Definitions of what constitutes a polynomial
def is_polynomial (expr : String) : Bool :=
  match expr with
  | "3/4*x^2" => true
  | "3ab" => true
  | "x+5" => true
  | "y/5x" => false
  | "-1" => true
  | "y/3" => true
  | "a^2-b^2" => true
  | "a" => true
  | _ => false

-- Given set of algebraic expressions
def expressions : List String := 
  ["3/4*x^2", "3ab", "x+5", "y/5x", "-1", "y/3", "a^2-b^2", "a"]

-- Count the number of polynomials in the given expressions
def count_polynomials (exprs : List String) : Nat :=
  exprs.foldr (fun expr count => if is_polynomial expr then count + 1 else count) 0

theorem number_of_polynomials_is_seven : count_polynomials expressions = 7 :=
  by
    sorry

end number_of_polynomials_is_seven_l5_5052


namespace rattlesnakes_count_l5_5002

-- Definitions
def total_snakes : ℕ := 200
def boa_constrictors : ℕ := 40
def pythons : ℕ := 3 * boa_constrictors
def rattlesnakes : ℕ := total_snakes - (boa_constrictors + pythons)

-- Theorem to prove
theorem rattlesnakes_count : rattlesnakes = 40 := by
  -- provide proof here
  sorry

end rattlesnakes_count_l5_5002


namespace lucille_paint_cans_needed_l5_5154

theorem lucille_paint_cans_needed :
  let wall1_area := 3 * 2
  let wall2_area := 3 * 2
  let wall3_area := 5 * 2
  let wall4_area := 4 * 2
  let total_area := wall1_area + wall2_area + wall3_area + wall4_area
  let coverage_per_can := 2
  let cans_needed := total_area / coverage_per_can
  cans_needed = 15 := 
by 
  sorry

end lucille_paint_cans_needed_l5_5154


namespace problem1_line_equation_problem2_circle_equation_l5_5802

-- Problem 1: Equation of a specific line
def line_intersection (x y : ℝ) : Prop := 
  2 * x + y - 8 = 0 ∧ x - 2 * y + 1 = 0

def line_perpendicular (x y : ℝ) : Prop :=
  6 * x - 8 * y + 3 = 0

noncomputable def find_line (x y : ℝ) : Prop :=
  ∃ (l : ℝ), (8 * x + 6 * y + l = 0) ∧ 
  line_intersection x y ∧ line_perpendicular x y

theorem problem1_line_equation : ∃ (x y : ℝ), find_line x y :=
sorry

-- Problem 2: Equation of a specific circle
def point_A (x y : ℝ) : Prop := 
  x = 5 ∧ y = 2

def point_B (x y : ℝ) : Prop := 
  x = 3 ∧ y = -2

def center_on_line (x y : ℝ) : Prop :=
  2 * x - y = 3

noncomputable def find_circle (x y r : ℝ) : Prop :=
  ((x - 2)^2 + (y - 1)^2 = r) ∧
  ∃ x1 y1 x2 y2, point_A x1 y1 ∧ point_B x2 y2 ∧ center_on_line x y ∧ ((x1 - x)^2 + (y1 - y)^2 = r)

theorem problem2_circle_equation : ∃ (x y r : ℝ), find_circle x y 10 :=
sorry

end problem1_line_equation_problem2_circle_equation_l5_5802


namespace probability_each_university_at_least_one_admission_l5_5101

def total_students := 4
def total_universities := 3

theorem probability_each_university_at_least_one_admission :
  ∃ (p : ℚ), p = 4 / 9 :=
by
  sorry

end probability_each_university_at_least_one_admission_l5_5101


namespace weight_of_new_person_l5_5632

theorem weight_of_new_person (W : ℝ) (N : ℝ) (h1 : (W + (8 * 2.5)) = (W - 20 + N)) : N = 40 :=
by
  sorry

end weight_of_new_person_l5_5632


namespace intersecting_lines_l5_5546

theorem intersecting_lines (m n : ℝ) : 
  (∀ x y : ℝ, y = x / 2 + n → y = mx - 1 → (x = 1 ∧ y = -2)) → 
  m = -1 ∧ n = -5 / 2 :=
by
  sorry

end intersecting_lines_l5_5546


namespace cosine_ab_ac_l5_5122

noncomputable def vector_a := (-2, 4, -6)
noncomputable def vector_b := (0, 2, -4)
noncomputable def vector_c := (-6, 8, -10)

noncomputable def a_b : ℝ × ℝ × ℝ := (2, -2, 2)
noncomputable def a_c : ℝ × ℝ × ℝ := (-4, 4, -4)

noncomputable def ab_dot_ac : ℝ := -24

noncomputable def mag_a_b : ℝ := 2 * Real.sqrt 3
noncomputable def mag_a_c : ℝ := 4 * Real.sqrt 3

theorem cosine_ab_ac :
  (ab_dot_ac / (mag_a_b * mag_a_c) = -1) :=
sorry

end cosine_ab_ac_l5_5122


namespace minimum_votes_for_tall_to_win_l5_5269

-- Definitions based on the conditions
def num_voters := 135
def num_districts := 5
def num_precincts_per_district := 9
def num_voters_per_precinct := 3

-- Tall won the contest
def tall_won := True

-- Winning conditions
def majority_precinct_vote (votes_for_tall : ℕ) : Prop :=
  votes_for_tall >= 2

def majority_district_win (precincts_won_by_tall : ℕ) : Prop :=
  precincts_won_by_tall >= 5

def majority_contest_win (districts_won_by_tall : ℕ) : Prop :=
  districts_won_by_tall >= 3

-- Prove the minimum number of voters who could have voted for Tall
theorem minimum_votes_for_tall_to_win : 
  ∃ (votes : ℕ), votes = 30 ∧ majority_contest_win 3 ∧ 
  (∀ d, d < 3 → majority_district_win 5) ∧ 
  (∀ p, p < 5 → majority_precinct_vote 2) :=
by
  sorry

end minimum_votes_for_tall_to_win_l5_5269


namespace cost_per_adult_meal_l5_5927

theorem cost_per_adult_meal (total_people : ℕ) (num_kids : ℕ) (total_cost : ℕ) (cost_per_kid : ℕ) :
  total_people = 12 →
  num_kids = 7 →
  cost_per_kid = 0 →
  total_cost = 15 →
  (total_cost / (total_people - num_kids)) = 3 :=
by
  intros
  sorry

end cost_per_adult_meal_l5_5927


namespace geometric_sequence_sum_of_first_four_terms_l5_5139

theorem geometric_sequence_sum_of_first_four_terms (a r : ℝ) 
  (h1 : a + a * r = 7) 
  (h2 : a * (1 + r + r^2 + r^3 + r^4 + r^5) = 91) : 
  a * (1 + r + r^2 + r^3) = 32 :=
by
  sorry

end geometric_sequence_sum_of_first_four_terms_l5_5139


namespace proof_problem_l5_5791

theorem proof_problem :
  ∀ (X : ℝ), 213 * 16 = 3408 → (213 * 16) + (1.6 * 2.13) = X → X - (5 / 2) * 1.25 = 3408.283 :=
by
  intros X h1 h2
  sorry

end proof_problem_l5_5791


namespace age_difference_l5_5852

theorem age_difference (a b c : ℕ) (h : a + b = b + c + 10) : c = a - 10 :=
by
  sorry

end age_difference_l5_5852


namespace average_of_175_results_l5_5109

theorem average_of_175_results (x y : ℕ) (hx : x = 100) (hy : y = 75) 
(a b : ℚ) (ha : a = 45) (hb : b = 65) :
  ((x * a + y * b) / (x + y) = 53.57) :=
sorry

end average_of_175_results_l5_5109


namespace line_passing_quadrants_l5_5923

-- Definition of the line
def line (x : ℝ) : ℝ := 5 * x - 2

-- Prove that the line passes through Quadrants I, III, and IV
theorem line_passing_quadrants :
  ∃ x_1 > 0, line x_1 > 0 ∧      -- Quadrant I
  ∃ x_3 < 0, line x_3 < 0 ∧      -- Quadrant III
  ∃ x_4 > 0, line x_4 < 0 :=     -- Quadrant IV
sorry

end line_passing_quadrants_l5_5923


namespace ten_yuan_notes_count_l5_5007

theorem ten_yuan_notes_count (total_notes : ℕ) (total_change : ℕ) (item_cost : ℕ) (change_given : ℕ → ℕ → ℕ) (is_ten_yuan_notes : ℕ → Prop) :
    total_notes = 16 →
    total_change = 95 →
    item_cost = 5 →
    change_given 10 5 = total_change →
    (∃ x y : ℕ, x + y = total_notes ∧ 10 * x + 5 * y = total_change ∧ is_ten_yuan_notes x) → is_ten_yuan_notes 3 :=
by
  sorry

end ten_yuan_notes_count_l5_5007


namespace even_function_value_sum_l5_5429

noncomputable def g (x : ℝ) (d e f : ℝ) : ℝ :=
  d * x^8 - e * x^6 + f * x^2 + 5

theorem even_function_value_sum (d e f : ℝ) (h : g 15 d e f = 7) :
  g 15 d e f + g (-15) d e f = 14 := by
  sorry

end even_function_value_sum_l5_5429


namespace evaluate_s_squared_plus_c_squared_l5_5383

variable {x y : ℝ}

theorem evaluate_s_squared_plus_c_squared (r : ℝ) (h_r_def : r = Real.sqrt (x^2 + y^2))
                                          (s : ℝ) (h_s_def : s = y / r)
                                          (c : ℝ) (h_c_def : c = x / r) :
  s^2 + c^2 = 1 :=
sorry

end evaluate_s_squared_plus_c_squared_l5_5383


namespace quadratic_inequality_solution_l5_5364

theorem quadratic_inequality_solution (x : ℝ) : (-x^2 + 5 * x - 4 < 0) ↔ (1 < x ∧ x < 4) :=
sorry

end quadratic_inequality_solution_l5_5364


namespace find_subtracted_number_l5_5340

theorem find_subtracted_number (t k x : ℝ) (h1 : t = 20) (h2 : k = 68) (h3 : t = 5/9 * (k - x)) :
  x = 32 :=
by
  sorry

end find_subtracted_number_l5_5340


namespace striped_nails_painted_l5_5029

theorem striped_nails_painted (total_nails purple_nails blue_nails : ℕ) (h_total : total_nails = 20)
    (h_purple : purple_nails = 6) (h_blue : blue_nails = 8)
    (h_diff_percent : |(blue_nails:ℚ) / total_nails * 100 - 
    ((total_nails - purple_nails - blue_nails):ℚ) / total_nails * 100| = 10) :
    (total_nails - purple_nails - blue_nails) = 6 := 
by 
  sorry

end striped_nails_painted_l5_5029


namespace rectangle_area_l5_5330

/-- 
In the rectangle \(ABCD\), \(AD - AB = 9\) cm. The area of trapezoid \(ABCE\) is 5 times 
the area of triangle \(ADE\). The perimeter of triangle \(ADE\) is 68 cm less than the 
perimeter of trapezoid \(ABCE\). Prove that the area of the rectangle \(ABCD\) 
is 3060 square centimeters.
-/
theorem rectangle_area (AB AD : ℝ) (S_ABC : ℝ) (S_ADE : ℝ) (P_ADE : ℝ) (P_ABC : ℝ) :
  AD - AB = 9 →
  S_ABC = 5 * S_ADE →
  P_ADE = P_ABC - 68 →
  (AB * AD = 3060) :=
by
  sorry

end rectangle_area_l5_5330


namespace box_height_l5_5289

theorem box_height (x h : ℕ) 
  (h1 : h = x + 5) 
  (h2 : 6 * x^2 + 20 * x ≥ 150) 
  (h3 : 5 * x + 5 ≥ 25) 
  : h = 9 :=
by 
  sorry

end box_height_l5_5289


namespace check_inequality_l5_5913

theorem check_inequality : 1.7^0.3 > 0.9^3.1 :=
sorry

end check_inequality_l5_5913


namespace graph_symmetry_l5_5659

variable (f : ℝ → ℝ)

theorem graph_symmetry :
  (∀ x y, y = f (x - 1) ↔ ∃ x', x' = 2 - x ∧ y = f (1 - x'))
  ∧ (∀ x' y', y' = f (1 - x') ↔ ∃ x, x = 2 - x' ∧ y' = f (x - 1)) :=
sorry

end graph_symmetry_l5_5659


namespace number_of_sodas_l5_5299

theorem number_of_sodas (cost_sandwich : ℝ) (num_sandwiches : ℕ) (cost_soda : ℝ) (total_cost : ℝ):
  cost_sandwich = 2.45 → 
  num_sandwiches = 2 → 
  cost_soda = 0.87 → 
  total_cost = 8.38 → 
  (total_cost - num_sandwiches * cost_sandwich) / cost_soda = 4 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  sorry

end number_of_sodas_l5_5299


namespace minimum_value_of_2x_plus_y_l5_5948

theorem minimum_value_of_2x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 * x + y + 6 = x * y) : 2 * x + y ≥ 12 :=
  sorry

end minimum_value_of_2x_plus_y_l5_5948


namespace solution_correct_l5_5846

def mascot_options := ["A Xiang", "A He", "A Ru", "A Yi", "Le Yangyang"]

def volunteer_options := ["A", "B", "C", "D", "E"]

noncomputable def count_valid_assignments (mascots : List String) (volunteers : List String) : Nat :=
  let all_assignments := mascots.permutations
  let valid_assignments := all_assignments.filter (λ p =>
    (p.get! 0 = "A Xiang" ∨ p.get! 1 = "A Xiang") ∧ p.get! 2 ≠ "Le Yangyang")
  valid_assignments.length

theorem solution_correct :
  count_valid_assignments mascot_options volunteer_options = 36 :=
by
  sorry

end solution_correct_l5_5846


namespace gage_skating_time_l5_5345

theorem gage_skating_time :
  let minutes_skated_first_5_days := 75 * 5
  let minutes_skated_next_3_days := 90 * 3
  let total_minutes_skated_8_days := minutes_skated_first_5_days + minutes_skated_next_3_days
  let total_minutes_required := 85 * 9
  let minutes_needed_ninth_day := total_minutes_required - total_minutes_skated_8_days
  minutes_needed_ninth_day = 120 :=
by
  let minutes_skated_first_5_days := 75 * 5
  let minutes_skated_next_3_days := 90 * 3
  let total_minutes_skated_8_days := minutes_skated_first_5_days + minutes_skated_next_3_days
  let total_minutes_required := 85 * 9
  let minutes_needed_ninth_day := total_minutes_required - total_minutes_skated_8_days
  sorry

end gage_skating_time_l5_5345


namespace Ravi_Prakash_finish_together_l5_5610

-- Definitions based on conditions
def Ravi_time := 24
def Prakash_time := 40

-- Main theorem statement
theorem Ravi_Prakash_finish_together :
  (1 / Ravi_time + 1 / Prakash_time) = 1 / 15 :=
by
  sorry

end Ravi_Prakash_finish_together_l5_5610


namespace cannot_form_optionE_l5_5165

-- Define the 4x4 tile
structure Tile4x4 :=
(matrix : Fin 4 → Fin 4 → Bool) -- Boolean to represent black or white

-- Define the condition of alternating rows and columns
def alternating_pattern (tile : Tile4x4) : Prop :=
  (∀ i, tile.matrix i 0 ≠ tile.matrix i 1 ∧
         tile.matrix i 2 ≠ tile.matrix i 3) ∧
  (∀ j, tile.matrix 0 j ≠ tile.matrix 1 j ∧
         tile.matrix 2 j ≠ tile.matrix 3 j)

-- Example tiles for options A, B, C, D, E
def optionA : Tile4x4 := sorry
def optionB : Tile4x4 := sorry
def optionC : Tile4x4 := sorry
def optionD : Tile4x4 := sorry
def optionE : Tile4x4 := sorry

-- Given pieces that can form a 4x4 alternating tile
axiom given_piece1 : Tile4x4
axiom given_piece2 : Tile4x4

-- Combining given pieces to form a 4x4 tile
def combine_pieces (p1 p2 : Tile4x4) : Tile4x4 := sorry -- Combination logic here

-- Proposition stating the problem
theorem cannot_form_optionE :
  (∀ tile, tile = optionA ∨ tile = optionB ∨ tile = optionC ∨ tile = optionD ∨ tile = optionE →
    (tile = optionA ∨ tile = optionB ∨ tile = optionC ∨ tile = optionD → alternating_pattern tile) ∧
    tile = optionE → ¬alternating_pattern tile) :=
sorry

end cannot_form_optionE_l5_5165


namespace function_quadrants_l5_5196

theorem function_quadrants (n : ℝ) (h: ∀ x : ℝ, x ≠ 0 → ((n-1)*x * x > 0)) : n > 1 :=
sorry

end function_quadrants_l5_5196


namespace find_extrema_of_f_l5_5880

noncomputable def f (x : ℝ) : ℝ := (x^4 + x^2 + 5) / (x^2 + 1)^2

theorem find_extrema_of_f :
  (∀ x : ℝ, f x ≤ 5) ∧ (∃ x : ℝ, f x = 5) ∧ (∀ x : ℝ, f x ≥ 0.95) ∧ (∃ x : ℝ, f x = 0.95) :=
by {
  sorry
}

end find_extrema_of_f_l5_5880


namespace area_difference_l5_5676

-- Define the original and new rectangle dimensions
def original_rect_area (length width : ℕ) : ℕ := length * width
def new_rect_area (length width : ℕ) : ℕ := (length - 2) * (width + 2)

-- Define the problem statement
theorem area_difference (a : ℕ) : new_rect_area a 5 - original_rect_area a 5 = 2 * a - 14 :=
by
  -- Insert proof here
  sorry

end area_difference_l5_5676


namespace correct_equation_l5_5061

noncomputable def team_a_initial := 96
noncomputable def team_b_initial := 72
noncomputable def team_b_final (x : ℕ) := team_b_initial - x
noncomputable def team_a_final (x : ℕ) := team_a_initial + x

theorem correct_equation (x : ℕ) : 
  (1 / 3 : ℚ) * (team_a_final x) = (team_b_final x) := 
  sorry

end correct_equation_l5_5061


namespace range_of_a_l5_5290

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, 0 < x ∧ x < 3 ↔ x > Real.log a / Real.log 2) → 0 < a ∧ a ≤ 1 := 
by 
  sorry

end range_of_a_l5_5290


namespace not_solution_B_l5_5503

theorem not_solution_B : ¬ (1 + 6 = 5) := by
  sorry

end not_solution_B_l5_5503


namespace average_percentage_of_10_students_l5_5067

theorem average_percentage_of_10_students 
  (avg_15_students : ℕ := 80)
  (n_15_students : ℕ := 15)
  (total_students : ℕ := 25)
  (overall_avg : ℕ := 84) : 
  ∃ (x : ℕ), ((n_15_students * avg_15_students + 10 * x) / total_students = overall_avg) → x = 90 := 
sorry

end average_percentage_of_10_students_l5_5067


namespace sqrt_9_eq_pm3_l5_5656

theorem sqrt_9_eq_pm3 : ∃ x : ℝ, x^2 = 9 ∧ (x = 3 ∨ x = -3) :=
by
  sorry

end sqrt_9_eq_pm3_l5_5656


namespace third_chapter_pages_l5_5857

theorem third_chapter_pages (x : ℕ) (h : 18 = x + 15) : x = 3 :=
by
  sorry

end third_chapter_pages_l5_5857


namespace alyssa_cookie_count_l5_5882

variable (Aiyanna_cookies Alyssa_cookies : ℕ)
variable (h1 : Aiyanna_cookies = 140)
variable (h2 : Aiyanna_cookies = Alyssa_cookies + 11)

theorem alyssa_cookie_count : Alyssa_cookies = 129 := by
  -- We can use the given conditions to prove the theorem
  sorry

end alyssa_cookie_count_l5_5882


namespace solve_fraction_eq_for_x_l5_5164

theorem solve_fraction_eq_for_x (x : ℝ) (hx : (x + 6) / (x - 3) = 4) : x = 6 :=
by sorry

end solve_fraction_eq_for_x_l5_5164


namespace three_city_population_l5_5530

noncomputable def totalPopulation (boise seattle lakeView: ℕ) : ℕ :=
  boise + seattle + lakeView

theorem three_city_population (pBoise pSeattle pLakeView : ℕ)
  (h1 : pBoise = 3 * pSeattle / 5)
  (h2 : pLakeView = pSeattle + 4000)
  (h3 : pLakeView = 24000) :
  totalPopulation pBoise pSeattle pLakeView = 56000 := by
  sorry

end three_city_population_l5_5530


namespace find_angle_l5_5609

theorem find_angle (x : Real) : 
  (x - (1 / 2) * (180 - x) = -18 - 24/60 - 36/3600) -> 
  x = 47 + 43/60 + 36/3600 :=
by
  sorry

end find_angle_l5_5609


namespace sin_alpha_minus_beta_l5_5550

theorem sin_alpha_minus_beta (α β : Real) 
  (h1 : Real.sin α = 12 / 13) 
  (h2 : Real.cos β = 4 / 5)
  (hα : π / 2 ≤ α ∧ α ≤ π)
  (hβ : -π / 2 ≤ β ∧ β ≤ 0) :
  Real.sin (α - β) = 33 / 65 := 
sorry

end sin_alpha_minus_beta_l5_5550


namespace annie_crayons_l5_5126

def initial_crayons : ℕ := 4
def additional_crayons : ℕ := 36
def total_crayons : ℕ := initial_crayons + additional_crayons

theorem annie_crayons : total_crayons = 40 :=
by
  sorry

end annie_crayons_l5_5126


namespace min_pairs_opponents_statement_l5_5045

-- Problem statement definitions
variables (h p : ℕ) (h_ge_1 : h ≥ 1) (p_ge_2 : p ≥ 2)

-- Required minimum number of pairs of opponents in a parliament
def min_pairs_opponents (h p : ℕ) : ℕ :=
  min ((h - 1) * p + 1) (Nat.choose (h + 1) 2)

-- Proof statement
theorem min_pairs_opponents_statement (h p : ℕ) (h_ge_1 : h ≥ 1) (p_ge_2 : p ≥ 2) :
  ∀ (hp : ℕ), ∃ (pairs : ℕ), 
    pairs = min_pairs_opponents h p :=
  sorry

end min_pairs_opponents_statement_l5_5045


namespace percent_increase_in_area_l5_5823

theorem percent_increase_in_area (s : ℝ) (h_s : s > 0) :
  let medium_area := s^2
  let large_length := 1.20 * s
  let large_width := 1.25 * s
  let large_area := large_length * large_width 
  let percent_increase := ((large_area - medium_area) / medium_area) * 100
  percent_increase = 50 := by
    sorry

end percent_increase_in_area_l5_5823


namespace hexagon_ratio_l5_5812

noncomputable def ratio_of_hexagon_areas (s : ℝ) : ℝ :=
  let area_ABCDEF := (3 * Real.sqrt 3 / 2) * s^2
  let side_smaller := (3 * s) / 2
  let area_smaller := (3 * Real.sqrt 3 / 2) * side_smaller^2
  area_smaller / area_ABCDEF

theorem hexagon_ratio (s : ℝ) : ratio_of_hexagon_areas s = 9 / 4 :=
by
  sorry

end hexagon_ratio_l5_5812


namespace total_earnings_l5_5476

def oil_change_cost : ℕ := 20
def repair_cost : ℕ := 30
def car_wash_cost : ℕ := 5

def num_oil_changes : ℕ := 5
def num_repairs : ℕ := 10
def num_car_washes : ℕ := 15

theorem total_earnings :
  (num_oil_changes * oil_change_cost) +
  (num_repairs * repair_cost) +
  (num_car_washes * car_wash_cost) = 475 :=
by
  sorry

end total_earnings_l5_5476


namespace Reeta_pencils_l5_5244

-- Let R be the number of pencils Reeta has
variable (R : ℕ)

-- Condition 1: Anika has 4 more than twice the number of pencils as Reeta
def Anika_pencils := 2 * R + 4

-- Condition 2: Together, Anika and Reeta have 64 pencils
def combined_pencils := R + Anika_pencils R

theorem Reeta_pencils (h : combined_pencils R = 64) : R = 20 :=
by
  sorry

end Reeta_pencils_l5_5244


namespace pooh_piglet_cake_sharing_l5_5478

theorem pooh_piglet_cake_sharing (a b : ℚ) (h1 : a + b = 1) (h2 : b + a/3 = 3*b) : 
  a = 6/7 ∧ b = 1/7 :=
by
  sorry

end pooh_piglet_cake_sharing_l5_5478


namespace min_employees_needed_l5_5868

-- Define the conditions
variable (W A : Finset ℕ)
variable (n_W n_A n_WA : ℕ)

-- Assume the given condition values
def sizeW := 95
def sizeA := 80
def sizeWA := 30

-- Define the proof problem
theorem min_employees_needed :
  (sizeW + sizeA - sizeWA) = 145 :=
by sorry

end min_employees_needed_l5_5868


namespace reflect_over_x_axis_reflect_over_y_axis_l5_5452

-- Mathematical Definitions
def Point := (ℝ × ℝ)

-- Reflect a point over the x-axis
def reflectOverX (M : Point) : Point :=
  (M.1, -M.2)

-- Reflect a point over the y-axis
def reflectOverY (M : Point) : Point :=
  (-M.1, M.2)

-- Theorem statements
theorem reflect_over_x_axis (M : Point) : reflectOverX M = (M.1, -M.2) :=
by
  sorry

theorem reflect_over_y_axis (M : Point) : reflectOverY M = (-M.1, M.2) :=
by
  sorry

end reflect_over_x_axis_reflect_over_y_axis_l5_5452


namespace solve_m_n_l5_5471

theorem solve_m_n (m n : ℝ) (h : m^2 + 2 * m + n^2 - 6 * n + 10 = 0) :
  m = -1 ∧ n = 3 :=
sorry

end solve_m_n_l5_5471


namespace smallest_pos_n_l5_5891

theorem smallest_pos_n (n : ℕ) (h : 435 * n % 30 = 867 * n % 30) : n = 5 :=
by
  sorry

end smallest_pos_n_l5_5891


namespace trigonometric_identity_l5_5737

theorem trigonometric_identity
  (α : ℝ)
  (h : Real.sin (π / 6 - α) = 1 / 3) :
  2 * Real.cos (π / 6 + α / 2) ^ 2 - 1 = 1 / 3 := by
  sorry

end trigonometric_identity_l5_5737


namespace count_two_digit_primes_with_given_conditions_l5_5818

def is_two_digit_prime (n : ℕ) : Prop :=
  n ≥ 10 ∧ n < 100 ∧ Prime n

def sum_of_digits_is_nine (n : ℕ) : Prop :=
  let tens := n / 10
  let units := n % 10
  tens + units = 9

def tens_greater_than_units (n : ℕ) : Prop :=
  let tens := n / 10
  let units := n % 10
  tens > units

theorem count_two_digit_primes_with_given_conditions :
  ∃ count : ℕ, count = 0 ∧ ∀ n, is_two_digit_prime n ∧ sum_of_digits_is_nine n ∧ tens_greater_than_units n → false :=
by
  -- proof goes here
  sorry

end count_two_digit_primes_with_given_conditions_l5_5818


namespace Jayden_less_Coraline_l5_5421

variables (M J : ℕ)
def Coraline_number := 80
def total_sum := 180

theorem Jayden_less_Coraline
  (h1 : M = J + 20)
  (h2 : J < Coraline_number)
  (h3 : M + J + Coraline_number = total_sum) :
  Coraline_number - J = 40 := by
  sorry

end Jayden_less_Coraline_l5_5421


namespace arc_length_of_sector_l5_5255

theorem arc_length_of_sector (theta : ℝ) (r : ℝ) (h_theta : theta = 90) (h_r : r = 6) : 
  (theta / 360) * 2 * Real.pi * r = 3 * Real.pi :=
by
  sorry

end arc_length_of_sector_l5_5255


namespace piles_can_be_combined_l5_5612

-- Define a predicate indicating that two integers x and y are similar sizes
def similar_sizes (x y : ℕ) : Prop :=
  x ≤ y ∧ y ≤ 2 * x

-- Define a function stating that we can combine piles while maintaining the similar sizes property
noncomputable def combine_piles (piles : List ℕ) : ℕ :=
  sorry

-- State the theorem where we prove that any initial configuration of piles can be combined into a single pile
theorem piles_can_be_combined (piles : List ℕ) :
  ∃ n : ℕ, combine_piles piles = n :=
by sorry

end piles_can_be_combined_l5_5612


namespace janet_time_per_post_l5_5682

/-- Janet gets paid $0.25 per post she checks. She earns $90 per hour. 
    Prove that it takes her 10 seconds to check a post. -/
theorem janet_time_per_post
  (payment_per_post : ℕ → ℝ)
  (hourly_pay : ℝ)
  (posts_checked_hourly : ℕ)
  (secs_per_post : ℝ) :
  payment_per_post 1 = 0.25 →
  hourly_pay = 90 →
  hourly_pay = payment_per_post (posts_checked_hourly) →
  secs_per_post = 10 :=
sorry

end janet_time_per_post_l5_5682


namespace find_angle_C_l5_5274

variable {A B C a b c : ℝ}
variable (hAcute : 0 < A ∧ A < π / 2 ∧ 0 < B ∧ B < π / 2 ∧ 0 < C ∧ C < π / 2)
variable (hTriangle : A + B + C = π)
variable (hSides : a > 0 ∧ b > 0 ∧ c > 0)
variable (hCondition : Real.sqrt 3 * a = 2 * c * Real.sin A)

theorem find_angle_C (hA_pos : A ≠ 0) : C = π / 3 :=
  sorry

end find_angle_C_l5_5274


namespace A_time_to_complete_work_l5_5570

-- Definitions of work rates for A, B, and C.
variables (A_work B_work C_work : ℚ)

-- Conditions
axiom cond1 : A_work = 3 * B_work
axiom cond2 : B_work = 2 * C_work
axiom cond3 : A_work + B_work + C_work = 1 / 15

-- Proof statement: The time taken by A alone to do the work is 22.5 days.
theorem A_time_to_complete_work : 1 / A_work = 22.5 :=
by {
  sorry
}

end A_time_to_complete_work_l5_5570


namespace ninth_term_arithmetic_sequence_l5_5831

variable (a d : ℕ)

def arithmetic_sequence_sum (a d : ℕ) : ℕ :=
  a + (a + d) + (a + 2 * d) + (a + 3 * d) + (a + 4 * d) + (a + 5 * d)

theorem ninth_term_arithmetic_sequence (h1 : arithmetic_sequence_sum a d = 21) (h2 : a + 6 * d = 7) : a + 8 * d = 9 :=
by
  sorry

end ninth_term_arithmetic_sequence_l5_5831


namespace num_integer_ks_l5_5555

theorem num_integer_ks (k : Int) :
  (∃ a b c d : Int, (2*x + a) * (x + b) = 2*x^2 - k*x + 6 ∨
                   (2*x + c) * (x + d) = 2*x^2 - k*x + 6) →
  ∃ ks : Finset Int, ks.card = 6 ∧ k ∈ ks :=
sorry

end num_integer_ks_l5_5555


namespace total_distance_traveled_l5_5470

/--
A spider is on the edge of a ceiling of a circular room with a radius of 65 feet. 
The spider walks straight across the ceiling to the opposite edge, passing through 
the center. It then walks straight to another point on the edge of the circle but 
not back through the center. The third part of the journey is straight back to the 
original starting point. If the third part of the journey was 90 feet long, then 
the total distance traveled by the spider is 313.81 feet.
-/
theorem total_distance_traveled (r : ℝ) (d1 d2 d3 : ℝ) (h1 : r = 65) (h2 : d1 = 2 * r) (h3 : d3 = 90) :
  d1 + d2 + d3 = 313.81 :=
by
  sorry

end total_distance_traveled_l5_5470


namespace fraction_eq_zero_iff_x_eq_6_l5_5872

theorem fraction_eq_zero_iff_x_eq_6 (x : ℝ) : (x - 6) / (5 * x) = 0 ↔ x = 6 :=
by
  sorry

end fraction_eq_zero_iff_x_eq_6_l5_5872


namespace Elise_paid_23_dollars_l5_5163

-- Definitions and conditions
def base_price := 3
def cost_per_mile := 4
def distance := 5

-- Desired conclusion (total cost)
def total_cost := base_price + cost_per_mile * distance

-- Theorem statement
theorem Elise_paid_23_dollars : total_cost = 23 := by
  sorry

end Elise_paid_23_dollars_l5_5163


namespace sin_alpha_neg_point_two_l5_5218

theorem sin_alpha_neg_point_two (a : ℝ) (h : Real.sin (Real.pi + a) = 0.2) : Real.sin a = -0.2 := 
by
  sorry

end sin_alpha_neg_point_two_l5_5218


namespace initial_tiger_sharks_l5_5441

open Nat

theorem initial_tiger_sharks (initial_guppies : ℕ) (initial_angelfish : ℕ) (initial_oscar_fish : ℕ)
  (sold_guppies : ℕ) (sold_angelfish : ℕ) (sold_tiger_sharks : ℕ) (sold_oscar_fish : ℕ)
  (remaining_fish : ℕ) (initial_total_fish : ℕ) (total_guppies_angelfish_oscar : ℕ) (initial_tiger_sharks : ℕ) :
  initial_guppies = 94 → initial_angelfish = 76 → initial_oscar_fish = 58 →
  sold_guppies = 30 → sold_angelfish = 48 → sold_tiger_sharks = 17 → sold_oscar_fish = 24 →
  remaining_fish = 198 →
  initial_total_fish = (sold_guppies + sold_angelfish + sold_tiger_sharks + sold_oscar_fish + remaining_fish) →
  total_guppies_angelfish_oscar = (initial_guppies + initial_angelfish + initial_oscar_fish) →
  initial_tiger_sharks = (initial_total_fish - total_guppies_angelfish_oscar) →
  initial_tiger_sharks = 89 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8 h9 h10 h11
  sorry

end initial_tiger_sharks_l5_5441


namespace circle_area_irrational_of_rational_radius_l5_5839

theorem circle_area_irrational_of_rational_radius (r : ℚ) : ¬ ∃ A : ℚ, A = π * (r:ℝ) * (r:ℝ) :=
by sorry

end circle_area_irrational_of_rational_radius_l5_5839


namespace meet_at_starting_point_second_time_in_minutes_l5_5725

theorem meet_at_starting_point_second_time_in_minutes :
  let racing_magic_time := 60 -- in seconds
  let charging_bull_time := 3600 / 40 -- in seconds
  let lcm_time := Nat.lcm racing_magic_time charging_bull_time -- LCM of the round times in seconds
  let answer := lcm_time / 60 -- convert seconds to minutes
  answer = 3 :=
by
  sorry

end meet_at_starting_point_second_time_in_minutes_l5_5725


namespace quadratic_factor_n_l5_5616

theorem quadratic_factor_n (n : ℤ) (h : ∃ m : ℤ, (x + 5) * (x + m) = x^2 + 7 * x + n) : n = 10 :=
sorry

end quadratic_factor_n_l5_5616


namespace solve_natural_a_l5_5677

theorem solve_natural_a (a : ℕ) : 
  (∃ n : ℕ, a^2 + a + 1589 = n^2) ↔ (a = 43 ∨ a = 28 ∨ a = 316 ∨ a = 1588) :=
sorry

end solve_natural_a_l5_5677


namespace fraction_difference_l5_5908

theorem fraction_difference (a b : ℝ) (h : a - b = 2 * a * b) : (1 / a - 1 / b) = -2 := 
by
  sorry

end fraction_difference_l5_5908


namespace find_S6_l5_5758

noncomputable def geometric_series_nth_term (a1 q : ℝ) (n : ℕ) : ℝ :=
  a1 * q^(n - 1)

noncomputable def geometric_series_sum (a1 q : ℝ) (n : ℕ) : ℝ :=
  if q = 1 then a1 * n else a1 * (1 - q^n) / (1 - q)

variables (a2 q : ℝ)

-- Conditions
axiom a_n_pos : ∀ n, n > 0 → geometric_series_nth_term a2 q n > 0
axiom q_gt_one : q > 1
axiom condition1 : geometric_series_nth_term a2 q 3 + geometric_series_nth_term a2 q 5 = 20
axiom condition2 : geometric_series_nth_term a2 q 2 * geometric_series_nth_term a2 q 6 = 64

-- Question/statement of the theorem
theorem find_S6 : geometric_series_sum 1 q 6 = 63 :=
  sorry

end find_S6_l5_5758


namespace cost_formula_l5_5313

-- Definitions based on conditions
def base_cost : ℕ := 15
def additional_cost_per_pound : ℕ := 5
def environmental_fee : ℕ := 2

-- Definition of cost function
def cost (P : ℕ) : ℕ := base_cost + additional_cost_per_pound * (P - 1) + environmental_fee

-- Theorem stating the formula for the cost C
theorem cost_formula (P : ℕ) (h : 1 ≤ P) : cost P = 12 + 5 * P :=
by
  -- Proof would go here
  sorry

end cost_formula_l5_5313


namespace mass_percent_O_CaOH2_is_correct_mass_percent_O_Na2CO3_is_correct_mass_percent_O_K2SO4_is_correct_l5_5620

-- Definitions for molar masses used in calculations
def molar_mass_Ca := 40.08
def molar_mass_O := 16.00
def molar_mass_H := 1.01
def molar_mass_Na := 22.99
def molar_mass_C := 12.01
def molar_mass_K := 39.10
def molar_mass_S := 32.07

-- Molar masses of the compounds
def molar_mass_CaOH2 := molar_mass_Ca + 2 * molar_mass_O + 2 * molar_mass_H
def molar_mass_Na2CO3 := 2 * molar_mass_Na + molar_mass_C + 3 * molar_mass_O
def molar_mass_K2SO4 := 2 * molar_mass_K + molar_mass_S + 4 * molar_mass_O

-- Mass of O in each compound
def mass_O_CaOH2 := 2 * molar_mass_O
def mass_O_Na2CO3 := 3 * molar_mass_O
def mass_O_K2SO4 := 4 * molar_mass_O

-- Mass percentages of O in each compound
def mass_percent_O_CaOH2 := (mass_O_CaOH2 / molar_mass_CaOH2) * 100
def mass_percent_O_Na2CO3 := (mass_O_Na2CO3 / molar_mass_Na2CO3) * 100
def mass_percent_O_K2SO4 := (mass_O_K2SO4 / molar_mass_K2SO4) * 100

theorem mass_percent_O_CaOH2_is_correct :
  mass_percent_O_CaOH2 = 43.19 := by sorry

theorem mass_percent_O_Na2CO3_is_correct :
  mass_percent_O_Na2CO3 = 45.29 := by sorry

theorem mass_percent_O_K2SO4_is_correct :
  mass_percent_O_K2SO4 = 36.73 := by sorry

end mass_percent_O_CaOH2_is_correct_mass_percent_O_Na2CO3_is_correct_mass_percent_O_K2SO4_is_correct_l5_5620


namespace inverse_proportion_function_point_l5_5955

theorem inverse_proportion_function_point (k x y : ℝ) (h₁ : 1 = k / (-6)) (h₂ : y = k / x) :
  k = -6 ∧ (x = 2 ∧ y = -3 ↔ y = -k / x) :=
by
  sorry

end inverse_proportion_function_point_l5_5955


namespace mil_equals_one_fortieth_mm_l5_5743

-- The condition that one mil is equal to one thousandth of an inch
def mil_in_inch := 1 / 1000

-- The condition that an inch is about 2.5 cm
def inch_in_mm := 25

-- The problem statement in Lean 4 form
theorem mil_equals_one_fortieth_mm : (mil_in_inch * inch_in_mm = 1 / 40) :=
by
  sorry

end mil_equals_one_fortieth_mm_l5_5743


namespace who_plays_chess_l5_5726

def person_plays_chess (A B C : Prop) : Prop := 
  (A ∧ ¬ B ∧ ¬ C) ∨ (¬ A ∧ B ∧ ¬ C) ∨ (¬ A ∧ ¬ B ∧ C)

axiom statement_A : Prop
axiom statement_B : Prop
axiom statement_C : Prop
axiom one_statement_true : (statement_A ∧ ¬ statement_B ∧ ¬ statement_C) ∨ (¬ statement_A ∧ statement_B ∧ ¬ statement_C) ∨ (¬ statement_A ∧ ¬ statement_B ∧ statement_C)

-- Definition translating the statements made by A, B, and C
def A_plays := true
def B_not_plays := true
def A_not_plays := ¬ A_plays

-- Axiom stating that only one of A's, B's, or C's statements are true
axiom only_one_true : (statement_A ∧ ¬ statement_B ∧ ¬ statement_C) ∨ (¬ statement_A ∧ statement_B ∧ ¬ statement_C) ∨ (¬ statement_A ∧ ¬ statement_B ∧ statement_C)

-- Prove that B is the one who knows how to play Chinese chess
theorem who_plays_chess : B_plays :=
by
  -- Insert proof steps here
  sorry

end who_plays_chess_l5_5726


namespace power_of_two_square_l5_5348

theorem power_of_two_square (n : ℕ) : ∃ k : ℕ, 2^6 + 2^9 + 2^n = k^2 ↔ n = 10 :=
by
  sorry

end power_of_two_square_l5_5348


namespace base_of_numbering_system_l5_5314

-- Definitions based on conditions
def num_children := 100
def num_boys := 24
def num_girls := 32

-- Problem statement: Prove the base of numbering system used is 6
theorem base_of_numbering_system (n: ℕ) (h: n ≠ 0):
    n^2 = (2 * n + 4) + (3 * n + 2) → n = 6 := 
  by
    sorry

end base_of_numbering_system_l5_5314


namespace b_2056_l5_5843

noncomputable def b (n : ℕ) : ℝ := sorry

-- Conditions
axiom h1 : b 1 = 2 + Real.sqrt 8
axiom h2 : b 2023 = 15 + Real.sqrt 8
axiom recurrence : ∀ n, n ≥ 2 → b n = b (n - 1) * b (n + 1)

-- Problem statement to prove
theorem b_2056 : b 2056 = (2 + Real.sqrt 8)^2 / (15 + Real.sqrt 8) :=
sorry

end b_2056_l5_5843


namespace expenditure_ratio_l5_5977

def ratio_of_incomes (I1 I2 : ℕ) : Prop := I1 / I2 = 5 / 4
def savings (I E : ℕ) : ℕ := I - E
def ratio_of_expenditures (E1 E2 : ℕ) : Prop := E1 / E2 = 3 / 2

theorem expenditure_ratio (I1 I2 E1 E2 : ℕ) 
  (I1_income : I1 = 5500)
  (income_ratio : ratio_of_incomes I1 I2)
  (savings_equal : savings I1 E1 = 2200 ∧ savings I2 E2 = 2200)
  : ratio_of_expenditures E1 E2 :=
by 
  sorry

end expenditure_ratio_l5_5977


namespace rounding_to_one_decimal_place_l5_5505

def number_to_round : Float := 5.049

def rounded_value : Float := 5.0

theorem rounding_to_one_decimal_place :
  (Float.round (number_to_round * 10) / 10) = rounded_value :=
by
  sorry

end rounding_to_one_decimal_place_l5_5505


namespace bob_calories_consumed_l5_5587

theorem bob_calories_consumed 
  (total_slices : ℕ)
  (half_slices : ℕ)
  (calories_per_slice : ℕ) 
  (H1 : total_slices = 8) 
  (H2 : half_slices = total_slices / 2) 
  (H3 : calories_per_slice = 300) : 
  half_slices * calories_per_slice = 1200 := 
by 
  sorry

end bob_calories_consumed_l5_5587


namespace right_triangle_shorter_leg_l5_5366

theorem right_triangle_shorter_leg (a b c : ℕ) (h1 : a^2 + b^2 = c^2) (h2 : c = 65) : a = 25 ∨ b = 25 := 
by
  sorry

end right_triangle_shorter_leg_l5_5366


namespace min_value_a_is_1_or_100_l5_5793

noncomputable def f (x : ℝ) : ℝ := x + 100 / x

theorem min_value_a_is_1_or_100 (a : ℝ) (m1 m2 : ℝ) 
  (h1 : a > 0) 
  (h_m1 : ∀ x, 0 < x ∧ x ≤ a → f x ≥ m1)
  (h_m1_min : ∃ x, 0 < x ∧ x ≤ a ∧ f x = m1)
  (h_m2 : ∀ x, a ≤ x → f x ≥ m2)
  (h_m2_min : ∃ x, a ≤ x ∧ f x = m2)
  (h_prod : m1 * m2 = 2020) : 
  a = 1 ∨ a = 100 :=
sorry

end min_value_a_is_1_or_100_l5_5793


namespace sqrt_54_sub_sqrt_6_l5_5794

theorem sqrt_54_sub_sqrt_6 : Real.sqrt 54 - Real.sqrt 6 = 2 * Real.sqrt 6 := by
  sorry

end sqrt_54_sub_sqrt_6_l5_5794


namespace average_salary_of_all_workers_l5_5013

def totalTechnicians : Nat := 6
def avgSalaryTechnician : Nat := 12000
def restWorkers : Nat := 6
def avgSalaryRest : Nat := 6000
def totalWorkers : Nat := 12
def totalSalary := (totalTechnicians * avgSalaryTechnician) + (restWorkers * avgSalaryRest)

theorem average_salary_of_all_workers : totalSalary / totalWorkers = 9000 := 
by
    -- replace with mathematical proof once available
    sorry

end average_salary_of_all_workers_l5_5013


namespace multiples_of_6_or_8_under_201_not_both_l5_5684

theorem multiples_of_6_or_8_under_201_not_both : 
  ∃ (n : ℕ), n = 42 ∧ 
    (∀ x : ℕ, x < 201 → ((x % 6 = 0 ∨ x % 8 = 0) ∧ x % 24 ≠ 0) → x ∈ Finset.range 201) :=
by
  sorry

end multiples_of_6_or_8_under_201_not_both_l5_5684


namespace price_of_cheaper_feed_l5_5562

theorem price_of_cheaper_feed 
  (W_total : ℝ) (P_total : ℝ) (E : ℝ) (W_C : ℝ) 
  (H1 : W_total = 27) 
  (H2 : P_total = 0.26)
  (H3 : E = 0.36)
  (H4 : W_C = 14.2105263158) 
  : (W_total * P_total = W_C * C + (W_total - W_C) * E) → 
    (C = 0.17) :=
by {
  sorry
}

end price_of_cheaper_feed_l5_5562


namespace eight_painters_finish_in_required_days_l5_5303

/- Conditions setup -/
def initial_painters : ℕ := 6
def initial_days : ℕ := 2
def job_constant := initial_painters * initial_days

def new_painters : ℕ := 8
def required_days := 3 / 2

/- Theorem statement -/
theorem eight_painters_finish_in_required_days : new_painters * required_days = job_constant :=
sorry

end eight_painters_finish_in_required_days_l5_5303


namespace trader_sold_95_pens_l5_5764

theorem trader_sold_95_pens
  (C : ℝ)   -- cost price of one pen
  (N : ℝ)   -- number of pens sold
  (h1 : 19 * C = 0.20 * N * C):  -- condition: profit from selling N pens is equal to the cost of 19 pens, with 20% gain percentage
  N = 95 := by
-- You would place the proof here.
  sorry

end trader_sold_95_pens_l5_5764


namespace lucas_change_l5_5584

def initialAmount : ℕ := 20
def costPerAvocado : ℕ := 2
def numberOfAvocados : ℕ := 3

def totalCost : ℕ := numberOfAvocados * costPerAvocado
def change : ℕ := initialAmount - totalCost

theorem lucas_change : change = 14 := by
  sorry

end lucas_change_l5_5584


namespace maximum_positive_numbers_l5_5475

theorem maximum_positive_numbers (a : ℕ → ℝ) (n : ℕ) (h₀ : n = 100)
  (h₁ : ∀ i : ℕ, 0 < a i) 
  (h₂ : ∀ i : ℕ, a i > a ((i + 1) % n) * a ((i + 2) % n)) : 
  ∃ m : ℕ, m ≤ 50 ∧ (∀ k : ℕ, k < m → (a k) > 0) :=
by sorry

end maximum_positive_numbers_l5_5475


namespace unit_price_solution_purchase_plan_and_costs_l5_5949

-- Definitions based on the conditions (Note: numbers and relationships purely)
def unit_prices (x y : ℕ) : Prop :=
  3 * x + 2 * y = 60 ∧ x + 3 * y = 55

def prize_purchase_conditions (m n : ℕ) : Prop :=
  m + n = 100 ∧ 10 * m + 15 * n ≤ 1160 ∧ m ≤ 3 * n

-- Proving that the unit prices found match the given constraints
theorem unit_price_solution : ∃ x y : ℕ, unit_prices x y := by
  sorry

-- Proving the number of purchasing plans and minimum cost
theorem purchase_plan_and_costs : 
  (∃ (num_plans : ℕ) (min_cost : ℕ), 
    num_plans = 8 ∧ min_cost = 1125 ∧ 
    ∀ m n : ℕ, prize_purchase_conditions m n → 
      ((68 ≤ m ∧ m ≤ 75) →
      10 * m + 15 * (100 - m) = min_cost)) := by
  sorry

end unit_price_solution_purchase_plan_and_costs_l5_5949


namespace problem_f_of_3_l5_5522

noncomputable def f (x : ℝ) : ℝ :=
if x ≤ 1 then x^2 + 1 else -2 * x + 3

theorem problem_f_of_3 : f (f 3) = 10 := by
  sorry

end problem_f_of_3_l5_5522


namespace min_value_of_fractions_l5_5376

theorem min_value_of_fractions (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
    (a+b)/(c+d) + (a+c)/(b+d) + (a+d)/(b+c) + (b+c)/(a+d) + (b+d)/(a+c) + (c+d)/(a+b) ≥ 6 :=
by
  sorry

end min_value_of_fractions_l5_5376


namespace binomial_7_4_eq_35_l5_5268

-- Define the binomial coefficient using the binomial coefficient formula.
def binomial_coefficient (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- State the theorem to prove.
theorem binomial_7_4_eq_35 : binomial_coefficient 7 4 = 35 :=
sorry

end binomial_7_4_eq_35_l5_5268


namespace find_triplet_l5_5957

theorem find_triplet (x y z : ℕ) :
  (1 + (x : ℚ) / (y + z))^2 + (1 + (y : ℚ) / (z + x))^2 + (1 + (z : ℚ) / (x + y))^2 = 27 / 4 ↔ (x, y, z) = (1, 1, 1) :=
by
  sorry

end find_triplet_l5_5957


namespace computer_additions_per_hour_l5_5553

theorem computer_additions_per_hour : 
  ∀ (initial_rate : ℕ) (increase_rate: ℚ) (intervals_per_hour : ℕ),
  initial_rate = 12000 → 
  increase_rate = 0.05 → 
  intervals_per_hour = 4 → 
  (12000 * 900) + (12000 * 1.05 * 900) + (12000 * 1.05^2 * 900) + (12000 * 1.05^3 * 900) = 46549350 := 
by
  intros initial_rate increase_rate intervals_per_hour h1 h2 h3
  have h4 : initial_rate = 12000 := h1
  have h5 : increase_rate = 0.05 := h2
  have h6 : intervals_per_hour = 4 := h3
  sorry

end computer_additions_per_hour_l5_5553


namespace audrey_dreaming_fraction_l5_5953

theorem audrey_dreaming_fraction
  (total_asleep_time : ℕ) 
  (not_dreaming_time : ℕ)
  (dreaming_time : ℕ)
  (fraction_dreaming : ℚ)
  (h_total_asleep : total_asleep_time = 10)
  (h_not_dreaming : not_dreaming_time = 6)
  (h_dreaming : dreaming_time = total_asleep_time - not_dreaming_time)
  (h_fraction : fraction_dreaming = dreaming_time / total_asleep_time) :
  fraction_dreaming = 2 / 5 := 
by {
  sorry
}

end audrey_dreaming_fraction_l5_5953


namespace fraction_equality_l5_5683

variable (a_n b_n : ℕ → ℝ)
variable (S_n T_n : ℕ → ℝ)

-- Conditions
axiom S_T_ratio (n : ℕ) : T_n n ≠ 0 → S_n n / T_n n = (2 * n + 1) / (4 * n - 2)
axiom Sn_def (n : ℕ) : S_n n = n / 2 * (2 * a_n 0 + (n - 1) * (a_n 1 - a_n 0))
axiom Tn_def (n : ℕ) : T_n n = n / 2 * (2 * b_n 0 + (n - 1) * (b_n 1 - b_n 0))
axiom an_def (n : ℕ) : a_n n = a_n 0 + n * (a_n 1 - a_n 0)
axiom bn_def (n : ℕ) : b_n n = b_n 0 + n * (b_n 1 - b_n 0)

-- Proof statement
theorem fraction_equality :
  (b_n 3 + b_n 18) ≠ 0 → (b_n 6 + b_n 15) ≠ 0 →
  (a_n 10 / (b_n 3 + b_n 18) + a_n 11 / (b_n 6 + b_n 15)) = (41 / 78) :=
by
  sorry

end fraction_equality_l5_5683


namespace oranges_kilos_bought_l5_5028

-- Definitions based on the given conditions
variable (O A x : ℝ)

-- Definitions from conditions
def A_value : Prop := A = 29
def equation1 : Prop := x * O + 5 * A = 419
def equation2 : Prop := 5 * O + 7 * A = 488

-- The theorem we want to prove
theorem oranges_kilos_bought {O A x : ℝ} (A_value: A = 29) (h1: x * O + 5 * A = 419) (h2: 5 * O + 7 * A = 488) : x = 5 :=
by
  -- start of proof
  sorry  -- proof omitted

end oranges_kilos_bought_l5_5028


namespace least_n_l5_5368

theorem least_n (n : ℕ) (h_pos : n > 0) (h_ineq : 1 / n - 1 / (n + 1) < 1 / 15) : n = 4 :=
sorry

end least_n_l5_5368


namespace selling_price_correct_l5_5954

noncomputable def selling_price (purchase_price : ℝ) (overhead_expenses : ℝ) (profit_percent : ℝ) : ℝ :=
  let total_cost_price := purchase_price + overhead_expenses
  let profit := (profit_percent / 100) * total_cost_price
  total_cost_price + profit

theorem selling_price_correct :
    selling_price 225 28 18.577075098814234 = 300 := by
  sorry

end selling_price_correct_l5_5954


namespace number_of_solutions_l5_5168

-- Define the relevant trigonometric equation
def trig_equation (x : ℝ) : Prop := (Real.cos x)^2 + 3 * (Real.sin x)^2 = 1

-- Define the range for x
def in_range (x : ℝ) : Prop := -20 < x ∧ x < 100

-- Define the predicate that x satisfies both the trig equation and the range condition
def satisfies_conditions (x : ℝ) : Prop := trig_equation x ∧ in_range x

-- The final theorem statement (proof is omitted)
theorem number_of_solutions : 
  ∃ (count : ℕ), count = 38 ∧ ∀ (x : ℝ), satisfies_conditions x ↔ x = k * Real.pi ∧ -20 < k * Real.pi ∧ k * Real.pi < 100 := sorry

end number_of_solutions_l5_5168


namespace Ian_hours_worked_l5_5404

theorem Ian_hours_worked (money_left: ℝ) (hourly_rate: ℝ) (spent: ℝ) (earned: ℝ) (hours: ℝ) :
  money_left = 72 → hourly_rate = 18 → spent = earned / 2 → earned = money_left * 2 → 
  earned = hourly_rate * hours → hours = 8 :=
by
  intros h1 h2 h3 h4 h5
  -- Begin mathematical validation process here
  sorry

end Ian_hours_worked_l5_5404


namespace min_value_of_x_sq_plus_6x_l5_5554

theorem min_value_of_x_sq_plus_6x : ∃ x : ℝ, ∀ y : ℝ, y^2 + 6*y ≥ -9 :=
by
  sorry

end min_value_of_x_sq_plus_6x_l5_5554


namespace equivalent_terminal_side_l5_5871

theorem equivalent_terminal_side (k : ℤ) : 
    (∃ k : ℤ, (5 * π / 3 = -π / 3 + 2 * π * k)) :=
sorry

end equivalent_terminal_side_l5_5871


namespace find_m_n_condition_l5_5788

theorem find_m_n_condition (m n : ℕ) :
  m ≥ 1 ∧ n > m ∧ (42 ^ n ≡ 42 ^ m [MOD 100]) ∧ m + n = 24 :=
sorry

end find_m_n_condition_l5_5788


namespace zach_saved_money_l5_5115

-- Definitions of known quantities
def cost_of_bike : ℝ := 100
def weekly_allowance : ℝ := 5
def mowing_earnings : ℝ := 10
def babysitting_rate : ℝ := 7
def babysitting_hours : ℝ := 2
def additional_earnings_needed : ℝ := 6

-- Calculate total earnings for this week
def total_earnings_this_week : ℝ := weekly_allowance + mowing_earnings + (babysitting_rate * babysitting_hours)

-- Prove that Zach has already saved $65
theorem zach_saved_money : (cost_of_bike - total_earnings_this_week - additional_earnings_needed) = 65 :=
by
  -- Sorry used as placeholder to skip the proof
  sorry

end zach_saved_money_l5_5115


namespace packages_per_box_l5_5586

theorem packages_per_box (P : ℕ) 
  (h1 : 100 * 25 = 2500) 
  (h2 : 2 * P * 250 = 2500) : 
  P = 5 := 
sorry

end packages_per_box_l5_5586


namespace two_a_minus_b_l5_5926

variables (a b : ℝ × ℝ)
variables (m : ℝ)

def parallel (u v : ℝ × ℝ) : Prop := ∃ k : ℝ, u = k • v

theorem two_a_minus_b 
  (ha : a = (1, -2))
  (hb : b = (m, 4))
  (h_parallel : parallel a b) :
  2 • a - b = (4, -8) :=
sorry

end two_a_minus_b_l5_5926


namespace polynomial_remainder_l5_5292

theorem polynomial_remainder (y : ℂ) (h1 : y^5 + y^4 + y^3 + y^2 + y + 1 = 0) (h2 : y^6 = 1) :
  (y^55 + y^40 + y^25 + y^10 + 1) % (y^5 + y^4 + y^3 + y^2 + y + 1) = 2 * y + 3 :=
sorry

end polynomial_remainder_l5_5292


namespace minimum_value_expression_l5_5359

theorem minimum_value_expression (a b c : ℤ) (h : a ≠ b ∧ b ≠ c ∧ a ≠ c) :
    3 * a^2 + 2 * b^2 + 4 * c^2 - a * b - 3 * b * c - 5 * c * a ≥ 6 :=
sorry

end minimum_value_expression_l5_5359


namespace two_digit_number_satisfying_conditions_l5_5025

theorem two_digit_number_satisfying_conditions :
  ∃ (s : Finset (ℕ × ℕ)), s.card = 8 ∧
  ∀ p ∈ s, ∃ (a b : ℕ), p = (a, b) ∧
    (10 * a + b < 100) ∧
    (a ≥ 2) ∧
    (10 * a + b + 10 * b + a = 110) :=
by
  sorry

end two_digit_number_satisfying_conditions_l5_5025


namespace sufficient_but_not_necessary_condition_subset_condition_l5_5873

open Set

variable (a : ℝ)
def U : Set ℝ := univ
def A : Set ℝ := {x : ℝ | 1 ≤ x ∧ x ≤ 5}
def B (a : ℝ) : Set ℝ := {x : ℝ | -1-2*a ≤ x ∧ x ≤ a-2}

theorem sufficient_but_not_necessary_condition (H : ∃ x ∈ A, x ∉ B a) : a ≥ 7 := sorry

theorem subset_condition (H : B a ⊆ A) : a < 1/3 := sorry

end sufficient_but_not_necessary_condition_subset_condition_l5_5873


namespace inscribed_circle_radius_right_triangle_l5_5745

theorem inscribed_circle_radius_right_triangle : 
  ∀ (DE EF DF : ℝ), 
    DE = 6 →
    EF = 8 →
    DF = 10 →
    ∃ (r : ℝ), r = 2 :=
by
  intros DE EF DF hDE hEF hDF
  sorry

end inscribed_circle_radius_right_triangle_l5_5745


namespace average_percent_increase_in_profit_per_car_l5_5424

theorem average_percent_increase_in_profit_per_car
  (N P : ℝ) -- N: Number of cars sold last year, P: Profit per car last year
  (HP1 : N > 0) -- Non-zero number of cars
  (HP2 : P > 0) -- Non-zero profit
  (HProfitIncrease : 1.3 * (N * P) = 1.3 * N * P) -- Total profit increased by 30%
  (HCarDecrease : 0.7 * N = 0.7 * N) -- Number of cars decreased by 30%
  : ((1.3 / 0.7) - 1) * 100 = 85.7 := sorry

end average_percent_increase_in_profit_per_car_l5_5424


namespace work_duration_l5_5182

variable (a b c : ℕ)
variable (daysTogether daysA daysB daysC : ℕ)

theorem work_duration (H1 : daysTogether = 4)
                      (H2 : daysA = 12)
                      (H3 : daysB = 18)
                      (H4: a = 1 / 12)
                      (H5: b = 1 / 18)
                      (H6: 1 / daysTogether = 1 / daysA + 1 / daysB + 1 / daysC) :
                      daysC = 9 :=
sorry

end work_duration_l5_5182


namespace point_in_second_quadrant_l5_5220

def in_second_quadrant (z : Complex) : Prop := 
  z.re < 0 ∧ z.im > 0

theorem point_in_second_quadrant : in_second_quadrant (Complex.ofReal (1) + 2 * Complex.I / (Complex.ofReal (1) - Complex.I)) :=
by sorry

end point_in_second_quadrant_l5_5220


namespace bike_distance_from_rest_l5_5600

variable (u : ℝ) (a : ℝ) (t : ℝ)

theorem bike_distance_from_rest (h1 : u = 0) (h2 : a = 0.5) (h3 : t = 8) : 
  (1 / 2 * a * t^2 = 16) :=
by
  sorry

end bike_distance_from_rest_l5_5600


namespace athlete_difference_is_30_l5_5068

def initial_athletes : ℕ := 600
def leaving_rate : ℕ := 35
def leaving_duration : ℕ := 6
def arrival_rate : ℕ := 20
def arrival_duration : ℕ := 9

def athletes_left : ℕ := leaving_rate * leaving_duration
def new_athletes : ℕ := arrival_rate * arrival_duration
def remaining_athletes : ℕ := initial_athletes - athletes_left
def final_athletes : ℕ := remaining_athletes + new_athletes
def athlete_difference : ℕ := initial_athletes - final_athletes

theorem athlete_difference_is_30 : athlete_difference = 30 :=
by
  show athlete_difference = 30
  -- Proof goes here
  sorry

end athlete_difference_is_30_l5_5068


namespace tom_paid_amount_correct_l5_5332

def kg (n : Nat) : Nat := n -- Just a type alias clarification

theorem tom_paid_amount_correct :
  ∀ (quantity_apples : Nat) (rate_apples : Nat) (quantity_mangoes : Nat) (rate_mangoes : Nat),
  quantity_apples = kg 8 →
  rate_apples = 70 →
  quantity_mangoes = kg 9 →
  rate_mangoes = 55 →
  (quantity_apples * rate_apples) + (quantity_mangoes * rate_mangoes) = 1055 :=
by
  intros quantity_apples rate_apples quantity_mangoes rate_mangoes
  intro h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  norm_num
  sorry

end tom_paid_amount_correct_l5_5332


namespace larry_gave_52_apples_l5_5262

-- Define the initial and final count of Joyce's apples
def initial_apples : ℝ := 75.0
def final_apples : ℝ := 127.0

-- Define the number of apples Larry gave Joyce
def apples_given : ℝ := final_apples - initial_apples

-- The theorem stating that Larry gave Joyce 52 apples
theorem larry_gave_52_apples : apples_given = 52 := by
  sorry

end larry_gave_52_apples_l5_5262


namespace contrapositive_proof_l5_5066

theorem contrapositive_proof (x m : ℝ) :
  (m < 0 → (∃ r : ℝ, r * r + 3 * r + m = 0)) ↔
  (¬ (∃ r : ℝ, r * r + 3 * r + m = 0) → m ≥ 0) :=
by
  sorry

end contrapositive_proof_l5_5066


namespace team_score_is_correct_l5_5623

-- Definitions based on given conditions
def connor_score : ℕ := 2
def amy_score : ℕ := connor_score + 4
def jason_score : ℕ := 2 * amy_score
def combined_score : ℕ := connor_score + amy_score + jason_score
def emily_score : ℕ := 3 * combined_score
def team_score : ℕ := connor_score + amy_score + jason_score + emily_score

-- Theorem stating team_score should be 80
theorem team_score_is_correct : team_score = 80 := by
  sorry

end team_score_is_correct_l5_5623


namespace compare_abc_l5_5320

theorem compare_abc (a b c : ℝ)
  (h1 : a = Real.log 0.9 / Real.log 2)
  (h2 : b = 3 ^ (-1 / 3 : ℝ))
  (h3 : c = (1 / 3 : ℝ) ^ (1 / 2 : ℝ)) :
  a < c ∧ c < b := by
  sorry

end compare_abc_l5_5320


namespace hex_B1C_base10_l5_5686

theorem hex_B1C_base10 : (11 * 16^2 + 1 * 16^1 + 12 * 16^0) = 2844 :=
by
  sorry

end hex_B1C_base10_l5_5686


namespace equivalent_proof_problem_l5_5515

variable (a b d e c f g h : ℚ)

def condition1 : Prop := 8 = (6 / 100) * a
def condition2 : Prop := 6 = (8 / 100) * b
def condition3 : Prop := 9 = (5 / 100) * d
def condition4 : Prop := 7 = (3 / 100) * e
def condition5 : Prop := c = b / a
def condition6 : Prop := f = d / a
def condition7 : Prop := g = e / b

theorem equivalent_proof_problem (hac1 : condition1 a)
                                 (hac2 : condition2 b)
                                 (hac3 : condition3 d)
                                 (hac4 : condition4 e)
                                 (hac5 : condition5 a b c)
                                 (hac6 : condition6 a d f)
                                 (hac7 : condition7 b e g) :
    h = f + g ↔ h = (803 / 20) * c := 
by sorry

end equivalent_proof_problem_l5_5515


namespace sales_growth_correct_equation_l5_5181

theorem sales_growth_correct_equation (x : ℝ) 
(sales_24th : ℝ) (total_sales_25th_26th : ℝ) 
(h_initial : sales_24th = 5000) (h_total : total_sales_25th_26th = 30000) :
  (5000 * (1 + x)) + (5000 * (1 + x)^2) = 30000 :=
sorry

end sales_growth_correct_equation_l5_5181


namespace students_in_line_l5_5961

theorem students_in_line (T N : ℕ) (hT : T = 1) (h_btw : N = T + 4) (h_behind: ∃ k, k = 8) : T + (N - T) + 1 + 8 = 13 :=
by
  sorry

end students_in_line_l5_5961


namespace percentage_of_invalid_votes_calculation_l5_5431

theorem percentage_of_invalid_votes_calculation
  (total_votes_poled : ℕ)
  (valid_votes_B : ℕ)
  (additional_percent_votes_A : ℝ)
  (Vb : ℝ)
  (total_valid_votes : ℝ)
  (P : ℝ) :
  total_votes_poled = 8720 →
  valid_votes_B = 2834 →
  additional_percent_votes_A = 0.15 →
  Vb = valid_votes_B →
  total_valid_votes = (2 * Vb) + (additional_percent_votes_A * total_votes_poled) →
  total_valid_votes / total_votes_poled = 1 - P/100 →
  P = 20 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end percentage_of_invalid_votes_calculation_l5_5431


namespace gloria_initial_dimes_l5_5934

variable (Q D : ℕ)

theorem gloria_initial_dimes (h1 : D = 5 * Q) 
                             (h2 : (3 * Q) / 5 + D = 392) : 
                             D = 350 := 
by {
  sorry
}

end gloria_initial_dimes_l5_5934


namespace ratio_expression_value_l5_5722

theorem ratio_expression_value (A B C : ℚ) (h_ratio : A / B = 3 / 2 ∧ B / C = 2 / 5) :
  (4 * A + 3 * B) / (5 * C - 2 * A) = 18 / 19 :=
by
  sorry

end ratio_expression_value_l5_5722


namespace problem1_problem2_l5_5649

noncomputable def p (x a : ℝ) : Prop := x^2 + 4 * a * x + 3 * a^2 < 0
noncomputable def q (x : ℝ) : Prop := (x^2 - 6 * x - 72 ≤ 0) ∧ (x^2 + x - 6 > 0)
noncomputable def condition1 (a : ℝ) : Prop := 
  a = -1 ∧ (∃ x, p x a ∨ q x)

noncomputable def condition2 (a : ℝ) : Prop :=
  ∀ x, ¬ p x a → ¬ q x

theorem problem1 (x : ℝ) (a : ℝ) (h₁ : condition1 a) : -6 ≤ x ∧ x < -3 ∨ 1 < x ∧ x ≤ 12 := 
sorry

theorem problem2 (a : ℝ) (h₂ : condition2 a) : -4 ≤ a ∧ a ≤ -2 :=
sorry

end problem1_problem2_l5_5649


namespace point_M_coordinates_l5_5915

theorem point_M_coordinates (a : ℤ) (h : a + 3 = 0) : (a + 3, 2 * a - 2) = (0, -8) :=
by
  sorry

end point_M_coordinates_l5_5915


namespace cylinder_volume_eq_l5_5457

variable (α β l : ℝ)

theorem cylinder_volume_eq (hα_pos : 0 < α ∧ α < π/2) (hβ_pos : 0 < β ∧ β < π/2) (hl_pos : 0 < l) :
  let V := (π * l^3 * Real.sin (2 * β) * Real.cos β) / (8 * (Real.cos α)^2)
  V = (π * l^3 * Real.sin (2 * β) * Real.cos β) / (8 * (Real.cos α)^2) :=
by 
  sorry

end cylinder_volume_eq_l5_5457


namespace min_value_of_sum_range_of_x_l5_5998

noncomputable def ab_condition (a b : ℝ) : Prop := a + b = 1
noncomputable def ra_positive (a b : ℝ) : Prop := a > 0 ∧ b > 0

-- Problem 1: Minimum value of (1/a + 4/b)

theorem min_value_of_sum (a b : ℝ) (h_ab : ab_condition a b) (h_pos : ra_positive a b) : 
    ∃ m : ℝ, m = 9 ∧ ∀ a b, ab_condition a b → ra_positive a b → 
    (1 / a + 4 / b) ≥ m :=
by sorry

-- Problem 2: Range of x for which the inequality holds

theorem range_of_x (a b x : ℝ) (h_ab : ab_condition a b) (h_pos : ra_positive a b) : 
    (1 / a + 4 / b) ≥ |2 * x - 1| - |x + 1| → x ∈ Set.Icc (-7 : ℝ) 11 :=
by sorry

end min_value_of_sum_range_of_x_l5_5998


namespace length_of_bridge_correct_l5_5938

noncomputable def length_of_bridge (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time_seconds : ℝ) : ℝ :=
  let train_speed_ms : ℝ := (train_speed_kmh * 1000) / 3600
  let total_distance : ℝ := train_speed_ms * crossing_time_seconds
  total_distance - train_length

theorem length_of_bridge_correct :
  length_of_bridge 500 42 60 = 200.2 :=
by
  sorry -- Proof of the theorem

end length_of_bridge_correct_l5_5938


namespace not_or_false_implies_or_true_l5_5116

variable (p q : Prop)

theorem not_or_false_implies_or_true (h : ¬(p ∨ q) = False) : p ∨ q :=
by
  sorry

end not_or_false_implies_or_true_l5_5116


namespace bricks_in_chimney_900_l5_5851

theorem bricks_in_chimney_900 (h : ℕ) :
  let Brenda_rate := h / 9
  let Brandon_rate := h / 10
  let combined_rate := (Brenda_rate + Brandon_rate) - 10
  5 * combined_rate = h → h = 900 :=
by
  intros Brenda_rate Brandon_rate combined_rate
  sorry

end bricks_in_chimney_900_l5_5851


namespace quadratic_eq_solutions_l5_5123

theorem quadratic_eq_solutions : ∃ x1 x2 : ℝ, (x^2 = x) ∨ (x = 0 ∧ x = 1) := by
  sorry

end quadratic_eq_solutions_l5_5123


namespace remaining_money_after_expenditures_l5_5732

def initial_amount : ℝ := 200.50
def spent_on_sweets : ℝ := 35.25
def given_to_each_friend : ℝ := 25.20

theorem remaining_money_after_expenditures :
  ((initial_amount - spent_on_sweets) - 2 * given_to_each_friend) = 114.85 :=
by
  sorry

end remaining_money_after_expenditures_l5_5732


namespace divisible_by_17_l5_5658

theorem divisible_by_17 (n : ℕ) : 17 ∣ (2 ^ (5 * n + 3) + 5 ^ n * 3 ^ (n + 2)) := 
by {
  sorry
}

end divisible_by_17_l5_5658


namespace trapezoid_median_properties_l5_5018

-- Define the variables
variables (a b x : ℝ)

-- State the conditions and the theorem
theorem trapezoid_median_properties (h1 : x = (2 * a) / 3) (h2 : x = b + 3) (h3 : x = (a + b) / 2) : x = 6 :=
by
  sorry

end trapezoid_median_properties_l5_5018


namespace insurance_compensation_l5_5202

/-- Given the actual damage amount and the deductible percentage, 
we can compute the amount of insurance compensation. -/
theorem insurance_compensation : 
  ∀ (damage_amount : ℕ) (deductible_percent : ℕ), 
  damage_amount = 300000 → 
  deductible_percent = 1 →
  (damage_amount - (damage_amount * deductible_percent / 100)) = 297000 :=
by
  intros damage_amount deductible_percent h_damage h_deductible
  sorry

end insurance_compensation_l5_5202


namespace compute_g_f_1_l5_5834

def f (x : ℝ) : ℝ := x^3 - 2 * x + 3
def g (x : ℝ) : ℝ := 2 * x^2 - 3 * x + 1

theorem compute_g_f_1 : g (f 1) = 3 :=
by
  sorry

end compute_g_f_1_l5_5834


namespace complex_neither_sufficient_nor_necessary_real_l5_5100

noncomputable def quadratic_equation_real_roots (a : ℝ) : Prop := 
  (a^2 - 4 * a ≥ 0)

noncomputable def quadratic_equation_complex_roots (a : ℝ) : Prop := 
  (a^2 - 4 * (-a) < 0)

theorem complex_neither_sufficient_nor_necessary_real (a : ℝ) :
  (quadratic_equation_complex_roots a ↔ quadratic_equation_real_roots a) = false := 
sorry

end complex_neither_sufficient_nor_necessary_real_l5_5100


namespace initial_number_of_persons_l5_5127

theorem initial_number_of_persons (n : ℕ) (avg_increase : ℝ) (old_weight : ℝ) (new_weight : ℝ) (weight_diff : ℝ)
  (h1 : avg_increase = 2.5) 
  (h2 : old_weight = 75) 
  (h3 : new_weight = 95)
  (h4 : weight_diff = new_weight - old_weight)
  (h5 : weight_diff = avg_increase * n) : n = 8 := 
sorry

end initial_number_of_persons_l5_5127


namespace max_square_test_plots_l5_5630

theorem max_square_test_plots (h_field_dims : (24 : ℝ) = 24 ∧ (52 : ℝ) = 52)
    (h_total_fencing : 1994 = 1994)
    (h_partitioning : ∀ (n : ℤ), n % 6 = 0 → n ≤ 19 → 
      (104 * n - 76 ≤ 1994) → (n / 6 * 13)^2 = 702) :
    ∃ n : ℤ, (n / 6 * 13)^2 = 702 := sorry

end max_square_test_plots_l5_5630


namespace proposition_A_iff_proposition_B_l5_5397

-- Define propositions
def Proposition_A (A B C : ℕ) : Prop := (A = 60 ∨ B = 60 ∨ C = 60)
def Proposition_B (A B C : ℕ) : Prop :=
  (A + B + C = 180) ∧ 
  (2 * B = A + C)

-- The theorem stating the relationship between Proposition_A and Proposition_B
theorem proposition_A_iff_proposition_B (A B C : ℕ) :
  Proposition_A A B C ↔ Proposition_B A B C :=
sorry

end proposition_A_iff_proposition_B_l5_5397


namespace not_divisible_l5_5944

theorem not_divisible (a b : ℕ) (h1 : 1 ≤ a) (h2 : a ≤ 12) : ¬∃ k : ℕ, 120 * a + 2 * b = k * (100 * a + b) := 
sorry

end not_divisible_l5_5944


namespace max_value_expression_l5_5354

theorem max_value_expression : ∃ s_max : ℝ, 
  (∀ s : ℝ, -3 * s^2 + 24 * s - 7 ≤ -3 * s_max^2 + 24 * s_max - 7) ∧
  (-3 * s_max^2 + 24 * s_max - 7 = 41) :=
sorry

end max_value_expression_l5_5354


namespace least_value_of_x_l5_5448

theorem least_value_of_x 
  (x : ℕ) 
  (p : ℕ) 
  (hx : 0 < x) 
  (hp : Prime p) 
  (h : x = 2 * 11 * p) : x = 44 := 
by
  sorry

end least_value_of_x_l5_5448


namespace sum_of_roots_eq_3n_l5_5481

variable {n : ℝ} 

-- Define the conditions
def quadratic_eq (x : ℝ) (m : ℝ) (n : ℝ) : Prop :=
  x^2 - (m + n) * x + m * n = 0

theorem sum_of_roots_eq_3n (m : ℝ) (n : ℝ) 
  (hm : m = 2 * n)
  (hroot_m : quadratic_eq m m n)
  (hroot_n : quadratic_eq n m n) :
  m + n = 3 * n :=
by sorry

end sum_of_roots_eq_3n_l5_5481


namespace find_a99_l5_5233

def seq (a : ℕ → ℕ) :=
  a 1 = 2 ∧ ∀ n ≥ 2, a n - a (n-1) = n + 1

theorem find_a99 (a : ℕ → ℕ) (h : seq a) : a 99 = 5049 :=
by
  have : seq a := h
  sorry

end find_a99_l5_5233


namespace isosceles_trapezoid_sides_length_l5_5486

theorem isosceles_trapezoid_sides_length (b1 b2 A : ℝ) (h s : ℝ) 
  (hb1 : b1 = 11) (hb2 : b2 = 17) (hA : A = 56) :
  (A = 1/2 * (b1 + b2) * h) →
  (s ^ 2 = h ^ 2 + (b2 - b1) ^ 2 / 4) →
  s = 5 :=
by
  intro
  sorry

end isosceles_trapezoid_sides_length_l5_5486


namespace daria_still_owes_l5_5128

-- Definitions of the given conditions
def saved_amount : ℝ := 500
def couch_cost : ℝ := 750
def table_cost : ℝ := 100
def lamp_cost : ℝ := 50

-- Calculation of total cost of the furniture
def total_cost : ℝ := couch_cost + table_cost + lamp_cost

-- Calculation of the remaining amount owed
def remaining_owed : ℝ := total_cost - saved_amount

-- Proof statement that Daria still owes $400 before interest
theorem daria_still_owes : remaining_owed = 400 := by
  -- Skipping the proof
  sorry

end daria_still_owes_l5_5128


namespace john_bought_two_dozens_l5_5975

theorem john_bought_two_dozens (x : ℕ) (h₁ : 21 + 3 = x * 12) : x = 2 :=
by {
    -- Placeholder for skipping the proof since it's not required.
    sorry
}

end john_bought_two_dozens_l5_5975


namespace perimeter_of_figure_is_correct_l5_5741

-- Define the conditions as Lean variables and constants
def area_of_figure : ℝ := 144
def number_of_squares : ℕ := 4

-- Define the question as a theorem to be proven in Lean
theorem perimeter_of_figure_is_correct :
  let area_of_square := area_of_figure / number_of_squares
  let side_length := Real.sqrt area_of_square
  let perimeter := 9 * side_length
  perimeter = 54 :=
by
  intro area_of_square
  intro side_length
  intro perimeter
  sorry

end perimeter_of_figure_is_correct_l5_5741


namespace AlissaMorePresents_l5_5111

/-- Ethan has 31 presents -/
def EthanPresents : ℕ := 31

/-- Alissa has 53 presents -/
def AlissaPresents : ℕ := 53

/-- How many more presents does Alissa have than Ethan? -/
theorem AlissaMorePresents : AlissaPresents - EthanPresents = 22 := by
  -- Place the proof here
  sorry

end AlissaMorePresents_l5_5111


namespace pinwheel_area_eq_six_l5_5050

open Set

/-- Define the pinwheel in a 6x6 grid -/
def is_midpoint (x y : ℤ) : Prop :=
  (x = 3 ∧ (y = 1 ∨ y = 5)) ∨ (y = 3 ∧ (x = 1 ∨ x = 5))

def is_center (x y : ℤ) : Prop :=
  x = 3 ∧ y = 3

def is_triangle_vertex (x y : ℤ) : Prop :=
  is_center x y ∨ is_midpoint x y

-- Main theorem statement
theorem pinwheel_area_eq_six :
  let pinwheel : Set (ℤ × ℤ) := {p | is_triangle_vertex p.1 p.2}
  ∀ A : ℝ, A = 6 :=
by sorry

end pinwheel_area_eq_six_l5_5050


namespace base5_minus_base8_to_base10_l5_5118

def base5_to_base10 (n : Nat) : Nat :=
  5 * 5^5 + 4 * 5^4 + 3 * 5^3 + 2 * 5^2 + 1 * 5^1 + 0 * 5^0

def base8_to_base10 (n : Nat) : Nat :=
  4 * 8^4 + 3 * 8^3 + 2 * 8^2 + 1 * 8^1 + 0 * 8^0

theorem base5_minus_base8_to_base10 :
  (base5_to_base10 543210 - base8_to_base10 43210) = 499 :=
by
  sorry

end base5_minus_base8_to_base10_l5_5118


namespace regression_analysis_incorrect_statement_l5_5437

theorem regression_analysis_incorrect_statement
  (y : ℕ → ℝ) (x : ℕ → ℝ) (b a : ℝ)
  (r : ℝ) (l : ℝ → ℝ) (P : ℝ × ℝ)
  (H1 : ∀ i, y i = b * x i + a)
  (H2 : abs r = 1 → ∀ x1 x2, l x1 = l x2 → x1 = x2)
  (H3 : ∃ m k, ∀ x, l x = m * x + k)
  (H4 : P.1 = b → l P.1 = P.2)
  (cond_A : ∀ i, y i ≠ b * x i + a) : false := 
sorry

end regression_analysis_incorrect_statement_l5_5437


namespace monotonically_decreasing_interval_range_of_f_l5_5697

noncomputable def f (x : ℝ) : ℝ := (1/2) ^ (abs (x - 1))

theorem monotonically_decreasing_interval :
  ∀ x₁ x₂ : ℝ, 1 < x₁ → x₁ < x₂ → f x₁ > f x₂ := by sorry

theorem range_of_f :
  Set.range f = {y : ℝ | 0 < y ∧ y ≤ 1 } := by sorry

end monotonically_decreasing_interval_range_of_f_l5_5697


namespace relationship_among_a_b_c_l5_5146

noncomputable def a : ℝ := (0.8 : ℝ)^(5.2 : ℝ)
noncomputable def b : ℝ := (0.8 : ℝ)^(5.5 : ℝ)
noncomputable def c : ℝ := (5.2 : ℝ)^(0.1 : ℝ)

theorem relationship_among_a_b_c : b < a ∧ a < c := by
  sorry

end relationship_among_a_b_c_l5_5146


namespace find_second_number_l5_5301

theorem find_second_number (x y z : ℝ) 
  (h1 : x + y + z = 120) 
  (h2 : x = (3/4) * y) 
  (h3 : z = (9/7) * y) 
  : y = 40 :=
sorry

end find_second_number_l5_5301


namespace tv_station_ads_l5_5043

theorem tv_station_ads (n m : ℕ) :
  n > 1 → 
  ∃ (an : ℕ → ℕ), 
  (an 0 = m) ∧ 
  (∀ k, 1 ≤ k ∧ k < n → an k = an (k - 1) - (k + (1 / 8) * (an (k - 1) - k))) ∧
  an n = 0 →
  (n = 7 ∧ m = 49) :=
by
  intro h
  exists sorry
  sorry

-- The proof steps are omitted

end tv_station_ads_l5_5043


namespace rhind_papyrus_max_bread_l5_5237

theorem rhind_papyrus_max_bread
  (a1 a2 a3 a4 a5 : ℕ) (d : ℕ)
  (h1 : a1 + a2 + a3 + a4 + a5 = 100)
  (h2 : a1 < a2 ∧ a2 < a3 ∧ a3 < a4 ∧ a4 < a5)
  (h3 : a2 = a1 + d)
  (h4 : a3 = a1 + 2 * d)
  (h5 : a4 = a1 + 3 * d)
  (h6 : a5 = a1 + 4 * d)
  (h7 : a3 + a4 + a5 = 3 * (a1 + a2)) :
  a5 = 30 :=
by {
  sorry
}

end rhind_papyrus_max_bread_l5_5237


namespace correct_proposition_is_B_l5_5250

variables {m n : Type} {α β : Type}

-- Define parallel and perpendicular relationships
def parallel (l₁ l₂ : Type) : Prop := sorry
def perpendicular (l₁ l₂ : Type) : Prop := sorry

def lies_in (l : Type) (p : Type) : Prop := sorry

-- The problem statement
theorem correct_proposition_is_B
  (H1 : perpendicular m α)
  (H2 : perpendicular n β)
  (H3 : perpendicular α β) :
  perpendicular m n :=
sorry

end correct_proposition_is_B_l5_5250


namespace train_pass_bridge_in_50_seconds_l5_5984

noncomputable def time_to_pass_bridge (length_train length_bridge : ℕ) (speed_kmh : ℕ) : ℕ :=
  let total_distance := length_train + length_bridge
  let speed_ms := (speed_kmh * 1000) / 3600
  total_distance / speed_ms

theorem train_pass_bridge_in_50_seconds :
  time_to_pass_bridge 485 140 45 = 50 :=
by
  sorry

end train_pass_bridge_in_50_seconds_l5_5984


namespace sum_of_cubes_1998_l5_5978

theorem sum_of_cubes_1998 : 1998 = 334^3 + 332^3 + (-333)^3 + (-333)^3 := by
  sorry

end sum_of_cubes_1998_l5_5978


namespace number_of_real_z5_is_10_l5_5014

theorem number_of_real_z5_is_10 :
  ∃ S : Finset ℂ, (∀ z ∈ S, z ^ 30 = 1 ∧ (z ^ 5).im = 0) ∧ S.card = 10 :=
sorry

end number_of_real_z5_is_10_l5_5014


namespace convert_base_5_to_decimal_l5_5827

-- Define the base-5 number 44 and its decimal equivalent
def base_5_number : ℕ := 4 * 5^1 + 4 * 5^0

-- Prove that the base-5 number 44 equals 24 in decimal
theorem convert_base_5_to_decimal : base_5_number = 24 := by
  sorry

end convert_base_5_to_decimal_l5_5827


namespace find_m_l5_5058

noncomputable def m_solution (m : ℝ) : ℂ := (m - 3 * Complex.I) / (2 + Complex.I)

theorem find_m :
  ∀ (m : ℝ), Complex.im (m_solution m) ≠ 0 → Complex.re (m_solution m) = 0 → m = 3 / 2 :=
by
  intro m h_im h_re
  sorry

end find_m_l5_5058


namespace calculate_flat_rate_shipping_l5_5036

noncomputable def flat_rate_shipping : ℝ :=
  17.00

theorem calculate_flat_rate_shipping
  (price_per_shirt : ℝ)
  (num_shirts : ℤ)
  (price_pack_socks : ℝ)
  (num_packs_socks : ℤ)
  (price_per_short : ℝ)
  (num_shorts : ℤ)
  (price_swim_trunks : ℝ)
  (num_swim_trunks : ℤ)
  (total_bill : ℝ)
  (total_items_cost : ℝ)
  (shipping_cost : ℝ) :
  price_per_shirt * num_shirts + 
  price_pack_socks * num_packs_socks + 
  price_per_short * num_shorts +
  price_swim_trunks * num_swim_trunks = total_items_cost →
  total_bill - total_items_cost = shipping_cost →
  total_items_cost > 50 → 
  0.20 * total_items_cost ≠ shipping_cost →
  flat_rate_shipping = 17.00 := 
sorry

end calculate_flat_rate_shipping_l5_5036


namespace sum_of_numbers_eq_answer_l5_5996

open Real

noncomputable def sum_of_numbers (x y : ℝ) : ℝ := x + y

theorem sum_of_numbers_eq_answer (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x * y = 16) (h4 : (1 / x) = 3 * (1 / y)) :
  sum_of_numbers x y = 16 * Real.sqrt 3 / 3 := 
sorry

end sum_of_numbers_eq_answer_l5_5996


namespace solve_quadratic_equation_l5_5103

theorem solve_quadratic_equation :
  ∃ x₁ x₂ : ℝ, x₁ = 1 + Real.sqrt 2 ∧ x₂ = 1 - Real.sqrt 2 ∧ ∀ x : ℝ, (x^2 - 2*x - 1 = 0) ↔ (x = x₁ ∨ x = x₂) :=
by
  sorry

end solve_quadratic_equation_l5_5103


namespace polygon_sides_l5_5389

theorem polygon_sides (n : ℕ) (h1 : (n - 2) * 180 = 3 * 360) (h2 : n > 2) : n = 8 := by
  -- Conditions given:
  -- h1: (n - 2) * 180 = 3 * 360
  -- h2: n > 2
  sorry

end polygon_sides_l5_5389


namespace maximum_height_of_projectile_l5_5195

def h (t : ℝ) : ℝ := -20 * t^2 + 80 * t + 36

theorem maximum_height_of_projectile : ∀ t : ℝ, (h t ≤ 116) :=
by sorry

end maximum_height_of_projectile_l5_5195


namespace relationship_y_values_l5_5328

theorem relationship_y_values (x1 x2 y1 y2 : ℝ) (h1 : x1 > x2) (h2 : 0 < x2) (h3 : y1 = - (3 / x1)) (h4 : y2 = - (3 / x2)) : y1 > y2 :=
by
  sorry

end relationship_y_values_l5_5328


namespace min_value_expression_l5_5575

theorem min_value_expression (x y k : ℝ) (hk : 1 < k) (hx : k < x) (hy : k < y) : 
  (∀ x y, x > k → y > k → (∃ m, (m ≤ (x^2 / (y - k) + y^2 / (x - k)))) ∧ (m = 8 * k)) := sorry

end min_value_expression_l5_5575


namespace problem_condition_l5_5179

noncomputable def m : ℤ := sorry
noncomputable def n : ℤ := sorry
noncomputable def x : ℤ := sorry
noncomputable def a : ℤ := 0
noncomputable def b : ℤ := -m + n

theorem problem_condition 
  (h1 : m ≠ 0)
  (h2 : n ≠ 0)
  (h3 : m ≠ n)
  (h4 : (x + m)^2 - (x^2 + n^2) = (m - n)^2) :
  x = a * m + b * n :=
sorry

end problem_condition_l5_5179


namespace circumcircle_eq_of_triangle_ABC_l5_5201

noncomputable def circumcircle_equation (A B C : ℝ × ℝ) : String := sorry

theorem circumcircle_eq_of_triangle_ABC :
  circumcircle_equation (4, 1) (-6, 3) (3, 0) = "x^2 + y^2 + x - 9y - 12 = 0" :=
sorry

end circumcircle_eq_of_triangle_ABC_l5_5201


namespace carl_typing_speed_l5_5853

theorem carl_typing_speed (words_per_day: ℕ) (minutes_per_day: ℕ) (total_words: ℕ) (days: ℕ) : 
  words_per_day = total_words / days ∧ 
  minutes_per_day = 4 * 60 ∧ 
  (words_per_day / minutes_per_day) = 50 :=
by 
  sorry

end carl_typing_speed_l5_5853


namespace point_C_velocity_l5_5552

theorem point_C_velocity (a T R L x : ℝ) (h : a * T / (a * T - R) = (L + x) / x) :
  x = L * (a * T / R - 1) → 
  (L * (a * T / R - 1)) / T = a * L / R :=
by
  sorry

end point_C_velocity_l5_5552


namespace five_a_plus_five_b_eq_neg_twenty_five_thirds_l5_5848

variable (g f : ℝ → ℝ)
variable (a b : ℝ)
axiom g_def : ∀ x, g x = 3 * x + 5
axiom g_inv_rel : ∀ x, g x = (f⁻¹ x) - 1
axiom f_def : ∀ x, f x = a * x + b
axiom f_inv_def : ∀ x, f⁻¹ (f x) = x

theorem five_a_plus_five_b_eq_neg_twenty_five_thirds :
    5 * a + 5 * b = -25 / 3 :=
sorry

end five_a_plus_five_b_eq_neg_twenty_five_thirds_l5_5848


namespace solve_fraction_equation_l5_5976

theorem solve_fraction_equation (x : ℝ) :
  (1 / (x + 10) + 1 / (x + 8) = 1 / (x + 11) + 1 / (x + 7)) ↔ x = -9 :=
by {
  sorry
}

end solve_fraction_equation_l5_5976


namespace average_tickets_per_day_l5_5263

def total_revenue : ℕ := 960
def price_per_ticket : ℕ := 4
def number_of_days : ℕ := 3

theorem average_tickets_per_day :
  (total_revenue / price_per_ticket) / number_of_days = 80 := 
sorry

end average_tickets_per_day_l5_5263


namespace Teena_speed_is_55_l5_5833

def Teena_speed (Roe_speed T : ℝ) (initial_gap final_gap time : ℝ) : Prop :=
  Roe_speed * time + initial_gap + final_gap = T * time

theorem Teena_speed_is_55 :
  Teena_speed 40 55 7.5 15 1.5 :=
by 
  sorry

end Teena_speed_is_55_l5_5833


namespace simplify_complex_expression_l5_5979

theorem simplify_complex_expression (i : ℂ) (h_i : i * i = -1) : 
  (11 - 3 * i) / (1 + 2 * i) = 3 - 5 * i :=
sorry

end simplify_complex_expression_l5_5979


namespace triangle_area_correct_l5_5700

noncomputable def triangle_area_given_conditions (a b c : ℝ) (A : ℝ) : ℝ :=
  if h : a = c + 4 ∧ b = c + 2 ∧ Real.cos A = -1/2 then
  1/2 * b * c * Real.sin A
  else 0

theorem triangle_area_correct :
  ∀ (a b c : ℝ), ∀ A : ℝ, a = c + 4 → b = c + 2 → Real.cos A = -1/2 → 
  triangle_area_given_conditions a b c A = 15 * Real.sqrt 3 / 4 :=
by
  intros a b c A ha hb hc
  simp [triangle_area_given_conditions, ha, hb, hc]
  sorry

end triangle_area_correct_l5_5700


namespace largest_sphere_surface_area_in_cone_l5_5645

theorem largest_sphere_surface_area_in_cone :
  (∀ (r : ℝ), (∃ (r : ℝ), r > 0 ∧ (1^2 + (3^2 - r^2) = 3^2)) →
    4 * π * r^2 ≤ 2 * π) :=
by
  sorry

end largest_sphere_surface_area_in_cone_l5_5645


namespace water_to_concentrate_ratio_l5_5078

theorem water_to_concentrate_ratio (servings : ℕ) (serving_size_oz concentrate_size_oz : ℕ)
                                (cans_of_concentrate required_juice_oz : ℕ)
                                (h_servings : servings = 280)
                                (h_serving_size : serving_size_oz = 6)
                                (h_concentrate_size : concentrate_size_oz = 12)
                                (h_cans_of_concentrate : cans_of_concentrate = 35)
                                (h_required_juice : required_juice_oz = servings * serving_size_oz)
                                (h_made_juice : required_juice_oz = 1680)
                                (h_concentrate_volume : cans_of_concentrate * concentrate_size_oz = 420)
                                (h_water_volume : required_juice_oz - (cans_of_concentrate * concentrate_size_oz) = 1260)
                                (h_water_cans : 1260 / concentrate_size_oz = 105) :
                                105 / 35 = 3 :=
by
  sorry

end water_to_concentrate_ratio_l5_5078


namespace uncle_wang_withdraw_amount_l5_5375

noncomputable def total_amount (principal : ℕ) (rate : ℚ) (time : ℕ) : ℚ :=
  principal + principal * rate * time

theorem uncle_wang_withdraw_amount :
  total_amount 100000 (315/10000) 2 = 106300 := by
  sorry

end uncle_wang_withdraw_amount_l5_5375


namespace perpendicular_slope_l5_5885

theorem perpendicular_slope (k : ℝ) : (∀ x, y = k*x) ∧ (∀ x, y = 2*x + 1) → k = -1 / 2 :=
by
  intro h
  sorry

end perpendicular_slope_l5_5885


namespace apples_count_l5_5867

def total_apples (mike_apples nancy_apples keith_apples : Nat) : Nat :=
  mike_apples + nancy_apples + keith_apples

theorem apples_count :
  total_apples 7 3 6 = 16 :=
by
  rfl

end apples_count_l5_5867


namespace age_ratio_l5_5064

variables (A B : ℕ)
def present_age_of_A : ℕ := 15
def future_ratio (A B : ℕ) : Prop := (A + 6) / (B + 6) = 7 / 5

theorem age_ratio (A_eq : A = present_age_of_A) (future_ratio_cond : future_ratio A B) : A / B = 5 / 3 :=
sorry

end age_ratio_l5_5064


namespace tim_more_points_than_joe_l5_5402

variable (J K T : ℕ)

theorem tim_more_points_than_joe (h1 : T = 30) (h2 : T = K / 2) (h3 : J + T + K = 100) : T - J = 20 :=
by
  sorry

end tim_more_points_than_joe_l5_5402


namespace a4_is_5_l5_5798

-- Define the condition x^5 = a_n + a_1(x-1) + a_2(x-1)^2 + a_3(x-1)^3 + a_4(x-1)^4 + a_5(x-1)^5
noncomputable def polynomial_identity (x a_n a_1 a_2 a_3 a_4 a_5 : ℝ) : Prop :=
  x^5 = a_n + a_1 * (x - 1) + a_2 * (x - 1)^2 + a_3 * (x - 1)^3 + a_4 * (x - 1)^4 + a_5 * (x - 1)^5

-- Define the theorem statement
theorem a4_is_5 (x a_n a_1 a_2 a_3 a_5 : ℝ) (h : polynomial_identity x a_n a_1 a_2 a_3 5 a_5) : a_4 = 5 :=
 by
 sorry

end a4_is_5_l5_5798


namespace three_digit_numbers_l5_5621

theorem three_digit_numbers (N : ℕ) (a b c : ℕ) 
  (h1 : N = 100 * a + 10 * b + c)
  (h2 : 1 ≤ a ∧ a ≤ 9)
  (h3 : b ≤ 9 ∧ c ≤ 9)
  (h4 : a - b + c % 11 = 0)
  (h5 : N % 11 = 0)
  (h6 : N = 11 * (a^2 + b^2 + c^2)) :
  N = 550 ∨ N = 803 :=
  sorry

end three_digit_numbers_l5_5621


namespace officeEmployees_l5_5668

noncomputable def totalEmployees 
  (averageSalaryAll : ℝ) 
  (averageSalaryOfficers : ℝ) 
  (averageSalaryManagers : ℝ) 
  (averageSalaryWorkers : ℝ) 
  (numOfficers : ℕ) 
  (numManagers : ℕ) 
  (numWorkers : ℕ) : ℕ := 
  if (numOfficers * averageSalaryOfficers + numManagers * averageSalaryManagers + numWorkers * averageSalaryWorkers) 
      = (numOfficers + numManagers + numWorkers) * averageSalaryAll 
  then numOfficers + numManagers + numWorkers 
  else 0

theorem officeEmployees
  (averageSalaryAll : ℝ)
  (averageSalaryOfficers : ℝ)
  (averageSalaryManagers : ℝ)
  (averageSalaryWorkers : ℝ)
  (numOfficers : ℕ)
  (numManagers : ℕ)
  (numWorkers : ℕ) :
  averageSalaryAll = 720 →
  averageSalaryOfficers = 1320 →
  averageSalaryManagers = 840 →
  averageSalaryWorkers = 600 →
  numOfficers = 10 →
  numManagers = 20 →
  (numOfficers * averageSalaryOfficers + numManagers * averageSalaryManagers + numWorkers * averageSalaryWorkers) 
    = (numOfficers + numManagers + numWorkers) * averageSalaryAll →
  totalEmployees averageSalaryAll averageSalaryOfficers averageSalaryManagers averageSalaryWorkers numOfficers numManagers numWorkers = 100 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  rw [h2, h3, h4, h5, h6] at h7
  rw [h1]
  simp [totalEmployees, h7]
  sorry

end officeEmployees_l5_5668


namespace tan_195_l5_5098

theorem tan_195 (a : ℝ) (h : Real.cos 165 = a) : Real.tan 195 = - (Real.sqrt (1 - a^2)) / a := 
sorry

end tan_195_l5_5098


namespace problem_solution_l5_5765

/-- Define proposition p: ∀α∈ℝ, sin(π-α) ≠ -sin(α) -/
def p := ∀ α : ℝ, Real.sin (Real.pi - α) ≠ -Real.sin α

/-- Define proposition q: ∃x∈[0,+∞), sin(x) > x -/
def q := ∃ x : ℝ, 0 ≤ x ∧ Real.sin x > x

/-- Prove that ¬p ∨ q is a true proposition -/
theorem problem_solution : ¬p ∨ q :=
by
  sorry

end problem_solution_l5_5765


namespace jaeho_got_most_notebooks_l5_5690

-- Define the number of notebooks each friend received
def notebooks_jaehyuk : ℕ := 12
def notebooks_kyunghwan : ℕ := 3
def notebooks_jaeho : ℕ := 15

-- Define the statement proving that Jaeho received the most notebooks
theorem jaeho_got_most_notebooks : notebooks_jaeho > notebooks_jaehyuk ∧ notebooks_jaeho > notebooks_kyunghwan :=
by {
  sorry -- this is where the proof would go
}

end jaeho_got_most_notebooks_l5_5690


namespace youngest_child_age_is_3_l5_5219

noncomputable def family_age_problem : Prop :=
  ∃ (age_diff_2 : ℕ) (age_10_years_ago : ℕ) (new_family_members : ℕ) (same_present_avg_age : ℕ) (youngest_child_age : ℕ),
    age_diff_2 = 2 ∧
    age_10_years_ago = 4 * 24 ∧
    new_family_members = 2 ∧
    same_present_avg_age = 24 ∧
    youngest_child_age = 3 ∧
    (96 + 4 * 10 + (youngest_child_age + (youngest_child_age + age_diff_2)) = 6 * same_present_avg_age)

theorem youngest_child_age_is_3 : family_age_problem := sorry

end youngest_child_age_is_3_l5_5219


namespace incorrect_conclusion_C_l5_5155

noncomputable def f (x : ℝ) := (x - 1)^2 * Real.exp x

theorem incorrect_conclusion_C : 
  ¬(∀ x, ∀ ε > 0, ∃ δ > 0, ∀ y, abs (y - x) < δ → abs (f y - f x) ≥ ε) :=
by
  sorry

end incorrect_conclusion_C_l5_5155


namespace pradeep_maximum_marks_l5_5491

theorem pradeep_maximum_marks (M : ℝ) (h1 : 0.35 * M = 175) :
  M = 500 :=
by
  sorry

end pradeep_maximum_marks_l5_5491


namespace prices_correct_minimum_cost_correct_l5_5141

-- Define the prices of the mustard brands
variables (x y m : ℝ)

def brandACost : ℝ := 9 * x + 6 * y
def brandBCost : ℝ := 5 * x + 8 * y

-- Conditions for prices
axiom cost_condition1 : brandACost x y = 390
axiom cost_condition2 : brandBCost x y = 310

-- Solution for prices
def priceA : ℝ := 30
def priceB : ℝ := 20

theorem prices_correct : x = priceA ∧ y = priceB :=
sorry

-- Conditions for minimizing cost
def totalCost (m : ℝ) : ℝ := 30 * m + 20 * (30 - m)
def totalPacks : ℝ := 30

-- Constraints
def constraint1 (m : ℝ) : Prop := m ≥ 5 + (30 - m)
def constraint2 (m : ℝ) : Prop := m ≤ 2 * (30 - m)

-- Minimum cost condition
def min_cost : ℝ := 780
def optimal_m : ℝ := 18

theorem minimum_cost_correct : constraint1 optimal_m ∧ constraint2 optimal_m ∧ totalCost optimal_m = min_cost :=
sorry

end prices_correct_minimum_cost_correct_l5_5141


namespace solve_cubic_eq_solve_quadratic_eq_l5_5329

-- Define the first equation and prove its solution
theorem solve_cubic_eq (x : ℝ) (h : x^3 + 64 = 0) : x = -4 :=
by
  -- skipped proof
  sorry

-- Define the second equation and prove its solutions
theorem solve_quadratic_eq (x : ℝ) (h : (x - 2)^2 = 81) : x = 11 ∨ x = -7 :=
by
  -- skipped proof
  sorry

end solve_cubic_eq_solve_quadratic_eq_l5_5329


namespace val_of_7c_plus_7d_l5_5434

noncomputable def h (x : ℝ) : ℝ := 7 * x - 6

noncomputable def f_inv (x : ℝ) : ℝ := 7 * x - 4

noncomputable def f (c d x : ℝ) : ℝ := c * x + d

theorem val_of_7c_plus_7d (c d : ℝ) (h_eq : ∀ x, h x = f_inv x - 2) 
  (inv_prop : ∀ x, f c d (f_inv x) = x) : 7 * c + 7 * d = 5 :=
by
  sorry

end val_of_7c_plus_7d_l5_5434


namespace handshake_count_250_l5_5816

theorem handshake_count_250 (n m : ℕ) (h1 : n = 5) (h2 : m = 5) :
  (n * m * (n * m - 1 - (n - 1))) / 2 = 250 :=
by
  -- Traditionally the theorem proof part goes here but it is omitted
  sorry

end handshake_count_250_l5_5816


namespace equalize_champagne_futile_l5_5287

/-- Stepashka cannot distribute champagne into 2018 glasses in such a way 
that Kryusha's attempts to equalize the amount in all glasses become futile. -/
theorem equalize_champagne_futile (n : ℕ) (h : n = 2018) : 
∃ (a : ℕ), (∀ (A B : ℕ), A ≠ B ∧ A + B = 2019 → (A + B) % 2 = 1) := 
sorry

end equalize_champagne_futile_l5_5287


namespace find_divisor_l5_5453

theorem find_divisor (n k : ℤ) (h1 : n % 30 = 16) : (2 * n) % 30 = 2 :=
by
  sorry

end find_divisor_l5_5453


namespace sherman_weekend_driving_time_l5_5347

def total_driving_time_per_week : ℕ := 9
def commute_time_per_day : ℕ := 1
def work_days_per_week : ℕ := 5
def weekend_days : ℕ := 2

theorem sherman_weekend_driving_time :
  (total_driving_time_per_week - commute_time_per_day * work_days_per_week) / weekend_days = 2 :=
sorry

end sherman_weekend_driving_time_l5_5347


namespace maximum_items_6_yuan_l5_5180

theorem maximum_items_6_yuan :
  ∃ (x : ℕ), (∀ (x' : ℕ), (∃ (y z : ℕ), 6 * x' + 4 * y + 2 * z = 60 ∧ x' + y + z = 16) →
    x' ≤ 7) → x = 7 :=
by
  sorry

end maximum_items_6_yuan_l5_5180


namespace problem_l5_5110

def f (x : ℤ) := 3 * x + 2

theorem problem : f (f (f 3)) = 107 := by
  sorry

end problem_l5_5110


namespace solve_for_x_l5_5789

theorem solve_for_x (x : ℝ) : (x - 55) / 3 = (2 - 3*x + x^2) / 4 → (x = 20 / 3 ∨ x = -11) :=
by
  intro h
  sorry

end solve_for_x_l5_5789


namespace sharks_problem_l5_5317

variable (F : ℝ)
variable (S : ℝ := 0.25 * (F + 3 * F))
variable (total_sharks : ℝ := 15)

theorem sharks_problem : 
  (0.25 * (F + 3 * F) = 15) ↔ (F = 15) :=
by 
  sorry

end sharks_problem_l5_5317


namespace common_ratio_of_geometric_series_l5_5378

theorem common_ratio_of_geometric_series :
  let a := (8:ℚ) / 10
  let second_term := (-6:ℚ) / 15 
  let r := second_term / a
  r = -1 / 2 :=
by
  let a := (8:ℚ) / 10
  let second_term := (-6:ℚ) / 15 
  let r := second_term / a
  have : r = -1 / 2 := sorry
  exact this

end common_ratio_of_geometric_series_l5_5378


namespace thirtieth_entry_satisfies_l5_5455

def r_9 (n : ℕ) : ℕ := n % 9

theorem thirtieth_entry_satisfies (n : ℕ) (h : ∃ k : ℕ, k < 30 ∧ ∀ m < 30, k ≠ m → 
    (r_9 (7 * n + 3) ≤ 4) ∧ 
    ((r_9 (7 * n + 3) ≤ 4) ↔ 
    (r_9 (7 * m + 3) > 4))) :
  n = 37 :=
sorry

end thirtieth_entry_satisfies_l5_5455


namespace no_three_positive_reals_l5_5346

noncomputable def S (a : ℝ) : Set ℕ := { n | ∃ (k : ℕ), n = ⌊(k : ℝ) * a⌋ }

theorem no_three_positive_reals (a b c : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : 0 < c) :
  (S a ∩ S b = ∅) ∧ (S b ∩ S c = ∅) ∧ (S c ∩ S a = ∅) ∧ (S a ∪ S b ∪ S c = Set.univ) → false :=
sorry

end no_three_positive_reals_l5_5346


namespace perimeter_of_triangle_hyperbola_l5_5599

theorem perimeter_of_triangle_hyperbola (x y : ℝ) (F1 F2 A B : ℝ) :
  (x^2 / 16) - (y^2 / 9) = 1 →
  |A - F2| - |A - F1| = 8 →
  |B - F2| - |B - F1| = 8 →
  |B - A| = 5 →
  |A - F2| + |B - F2| + |B - A| = 26 :=
by
  sorry

end perimeter_of_triangle_hyperbola_l5_5599


namespace f_not_surjective_l5_5133

def f : ℝ → ℕ → Prop := sorry

theorem f_not_surjective (f : ℝ → ℕ) 
  (h : ∀ x y : ℝ, f (x + (1 / f y)) = f (y + (1 / f x))) : 
  ¬ (∀ n : ℕ, ∃ x : ℝ, f x = n) :=
sorry

end f_not_surjective_l5_5133


namespace total_winter_clothing_l5_5392

def first_box_items : Nat := 3 + 5 + 2
def second_box_items : Nat := 4 + 3 + 1
def third_box_items : Nat := 2 + 6 + 3
def fourth_box_items : Nat := 1 + 7 + 2

theorem total_winter_clothing : first_box_items + second_box_items + third_box_items + fourth_box_items = 39 := by
  sorry

end total_winter_clothing_l5_5392


namespace solution_set_abs_inequality_l5_5704

theorem solution_set_abs_inequality :
  {x : ℝ | |x + 1| > 1} = {x : ℝ | x < -2} ∪ {x : ℝ | x > 0} :=
sorry

end solution_set_abs_inequality_l5_5704


namespace painting_time_l5_5497

theorem painting_time (rate_taylor rate_jennifer rate_alex : ℚ) 
  (h_taylor : rate_taylor = 1 / 12) 
  (h_jennifer : rate_jennifer = 1 / 10) 
  (h_alex : rate_alex = 1 / 15) : 
  ∃ t : ℚ, t = 4 ∧ (1 / t) = rate_taylor + rate_jennifer + rate_alex :=
by
  sorry

end painting_time_l5_5497


namespace coin_flip_probability_l5_5426

def total_outcomes := 2^6
def favorable_outcomes := 2^3
def probability := favorable_outcomes / total_outcomes

theorem coin_flip_probability :
  probability = 1 / 8 :=
by
  unfold probability total_outcomes favorable_outcomes
  sorry

end coin_flip_probability_l5_5426


namespace linear_equation_a_ne_1_l5_5395

theorem linear_equation_a_ne_1 (a : ℝ) : (∀ x : ℝ, (a - 1) * x - 6 = 0 → a ≠ 1) :=
sorry

end linear_equation_a_ne_1_l5_5395


namespace determine_n_l5_5524

noncomputable def P : ℤ → ℤ := sorry

theorem determine_n (n : ℕ) (P : ℤ → ℤ)
  (h_deg : ∀ x : ℤ, P x = 2 ∨ P x = 1 ∨ P x = 0)
  (h0 : ∀ k : ℕ, k ≤ n → P (3 * k) = 2)
  (h1 : ∀ k : ℕ, k < n → P (3 * k + 1) = 1)
  (h2 : ∀ k : ℕ, k < n → P (3 * k + 2) = 0)
  (h_f : P (3 * n + 1) = 730) :
  n = 4 := 
sorry

end determine_n_l5_5524


namespace negate_universal_statement_l5_5982

theorem negate_universal_statement :
  (¬ ∀ x : ℝ, x^2 ≥ 0) ↔ (∃ x : ℝ, x^2 < 0) :=
by
  sorry

end negate_universal_statement_l5_5982


namespace math_problem_l5_5744

theorem math_problem
  (a b c : ℚ)
  (h1 : a ≠ 0)
  (h2 : b ≠ 0)
  (h3 : c ≠ 0)
  (h4 : a * b^2 = c / a - b) :
  ( (a^2 * b^2 / c^2 - 2 / c + 1 / (a^2 * b^2) + 2 * a * b / c^2 - 2 / (a * b * c))
    / (2 / (a * b) - 2 * a * b / c)
    / (101 / c)
  ) = -1 / 202 := 
sorry

end math_problem_l5_5744


namespace max_value_of_m_l5_5231

noncomputable def f (x m n : ℝ) : ℝ := x^2 + m*x + n^2
noncomputable def g (x m n : ℝ) : ℝ := x^2 + (m+2)*x + n^2 + m + 1

theorem max_value_of_m (m n t : ℝ) :
  (∀(t : ℝ), f t m n ≥ 0 ∨ g t m n ≥ 0) → m ≤ 1 :=
by
  intro h
  sorry

end max_value_of_m_l5_5231


namespace distance_between_sasha_and_kolya_is_19_meters_l5_5358

theorem distance_between_sasha_and_kolya_is_19_meters
  (v_S v_L v_K : ℝ)
  (h1 : v_L = 0.9 * v_S)
  (h2 : v_K = 0.81 * v_S)
  (h3 : ∀ t_S : ℝ, t_S = 100 / v_S) :
  (∀ t_S : ℝ, 100 - v_K * t_S = 19) :=
by
  intros t_S
  have vL_defined : v_L = 0.9 * v_S := h1
  have vK_defined : v_K = 0.81 * v_S := h2
  have time_S : t_S = 100 / v_S := h3 t_S
  sorry

end distance_between_sasha_and_kolya_is_19_meters_l5_5358


namespace Ramesh_paid_l5_5749

theorem Ramesh_paid (P : ℝ) (h1 : 1.10 * P = 21725) : 0.80 * P + 125 + 250 = 16175 :=
by
  sorry

end Ramesh_paid_l5_5749


namespace last_three_digits_l5_5438

theorem last_three_digits (n : ℕ) : 7^106 % 1000 = 321 :=
by
  sorry

end last_three_digits_l5_5438


namespace candy_bar_cost_correct_l5_5689

-- Definitions based on conditions
def candy_bar_cost := 3
def chocolate_cost := candy_bar_cost + 5
def total_cost := chocolate_cost + candy_bar_cost

-- Assertion to be proved
theorem candy_bar_cost_correct :
  total_cost = 11 → candy_bar_cost = 3 :=
by
  intro h
  simp [total_cost, chocolate_cost, candy_bar_cost] at h
  sorry

end candy_bar_cost_correct_l5_5689


namespace find_positive_integers_l5_5959

theorem find_positive_integers (n : ℕ) (h_pos : n > 0) : 
  (∃ d : ℕ, ∀ k : ℕ, 6^n + 1 = d * (10^k - 1) / 9 → d = 7) → 
  n = 1 ∨ n = 5 :=
sorry

end find_positive_integers_l5_5959


namespace arith_seq_ninth_term_value_l5_5038

variable {a : Nat -> ℤ}
variable {S : Nat -> ℤ}

def arith_seq (a : Nat -> ℤ) : Prop :=
  ∀ n, a (n + 1) = a n + a 1^2

def arith_sum (S : Nat -> ℤ) (a : Nat -> ℤ) : Prop :=
  ∀ n, S n = n * (a 1 + a n) / 2

theorem arith_seq_ninth_term_value
  (h_seq : arith_seq a)
  (h_sum : arith_sum S a)
  (h_cond1 : a 1 + a 2^2 = -3)
  (h_cond2 : S 5 = 10) :
  a 9 = 20 :=
by
  sorry

end arith_seq_ninth_term_value_l5_5038


namespace afternoon_to_morning_ratio_l5_5909

theorem afternoon_to_morning_ratio (total_kg : ℕ) (afternoon_kg : ℕ) (morning_kg : ℕ) 
  (h1 : total_kg = 390) (h2 : afternoon_kg = 260) (h3 : morning_kg = total_kg - afternoon_kg) :
  afternoon_kg / morning_kg = 2 :=
sorry

end afternoon_to_morning_ratio_l5_5909


namespace emily_original_salary_l5_5691

def original_salary_emily (num_employees : ℕ) (original_employee_salary new_employee_salary new_salary_emily : ℕ) : ℕ :=
  new_salary_emily + (new_employee_salary - original_employee_salary) * num_employees

theorem emily_original_salary :
  original_salary_emily 10 20000 35000 850000 = 1000000 :=
by
  sorry

end emily_original_salary_l5_5691


namespace recliner_price_drop_l5_5414

theorem recliner_price_drop
  (P : ℝ) (N : ℝ)
  (N' : ℝ := 1.8 * N)
  (G : ℝ := P * N)
  (G' : ℝ := 1.44 * G) :
  (P' : ℝ) → P' = 0.8 * P → (P - P') / P * 100 = 20 :=
by
  intros
  sorry

end recliner_price_drop_l5_5414


namespace parking_space_area_l5_5053

theorem parking_space_area
  (L : ℕ) (W : ℕ)
  (hL : L = 9)
  (hSum : 2 * W + L = 37) : L * W = 126 := 
by
  sorry

end parking_space_area_l5_5053


namespace Linda_has_24_classmates_l5_5796

theorem Linda_has_24_classmates 
  (cookies_per_student : ℕ := 10)
  (cookies_per_batch : ℕ := 48)
  (chocolate_chip_batches : ℕ := 2)
  (oatmeal_raisin_batches : ℕ := 1)
  (additional_batches : ℕ := 2) : 
  (chocolate_chip_batches * cookies_per_batch + oatmeal_raisin_batches * cookies_per_batch + additional_batches * cookies_per_batch) / cookies_per_student = 24 := 
by 
  sorry

end Linda_has_24_classmates_l5_5796


namespace quadratic_roots_ratio_l5_5803

theorem quadratic_roots_ratio (m n p : ℝ) (h₁ : m ≠ 0) (h₂ : n ≠ 0) (h₃ : p ≠ 0)
    (h₄ : ∀ (s₁ s₂ : ℝ), s₁ + s₂ = -p ∧ s₁ * s₂ = m ∧ 3 * s₁ + 3 * s₂ = -m ∧ 9 * s₁ * s₂ = n) :
    n / p = 27 :=
sorry

end quadratic_roots_ratio_l5_5803


namespace robin_candy_consumption_l5_5409

theorem robin_candy_consumption (x : ℕ) : 23 - x + 21 = 37 → x = 7 :=
by
  intros h
  sorry

end robin_candy_consumption_l5_5409


namespace total_votes_is_120_l5_5309

-- Define the conditions
def Fiona_votes : ℕ := 48
def fraction_of_votes : ℚ := 2 / 5

-- The proof goal
theorem total_votes_is_120 (V : ℕ) (h : Fiona_votes = fraction_of_votes * V) : V = 120 :=
by
  sorry

end total_votes_is_120_l5_5309


namespace total_books_read_l5_5496

-- Given conditions
variables (c s : ℕ) -- variable c represents the number of classes, s represents the number of students per class

-- Main statement to prove
theorem total_books_read (h1 : ∀ a, a = 7) (h2 : ∀ b, b = 12) :
  84 * c * s = 84 * c * s :=
by
  sorry

end total_books_read_l5_5496


namespace quadratic_two_distinct_roots_l5_5257

theorem quadratic_two_distinct_roots :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ 2 * x1^2 - 3 = 0 ∧ 2 * x2^2 - 3 = 0) :=
by
  sorry

end quadratic_two_distinct_roots_l5_5257


namespace total_cans_l5_5724

def bag1 := 5
def bag2 := 7
def bag3 := 12
def bag4 := 4
def bag5 := 8
def bag6 := 10

theorem total_cans : bag1 + bag2 + bag3 + bag4 + bag5 + bag6 = 46 := by
  sorry

end total_cans_l5_5724


namespace remainder_2n_div_14_l5_5249

theorem remainder_2n_div_14 (n : ℤ) (h : n % 28 = 15) : (2 * n) % 14 = 2 :=
sorry

end remainder_2n_div_14_l5_5249


namespace yoojung_namjoon_total_flowers_l5_5239

theorem yoojung_namjoon_total_flowers
  (yoojung_flowers : ℕ)
  (namjoon_flowers : ℕ)
  (yoojung_condition : yoojung_flowers = 4 * namjoon_flowers)
  (yoojung_count : yoojung_flowers = 32) :
  yoojung_flowers + namjoon_flowers = 40 :=
by
  sorry

end yoojung_namjoon_total_flowers_l5_5239


namespace value_expression_eq_zero_l5_5423

theorem value_expression_eq_zero (a b c : ℝ) (h_distinct : a ≠ b ∧ b ≠ c ∧ c ≠ a)
    (h_condition : a / (b - c) + b / (c - a) + c / (a - b) = 1) :
    a / (b - c)^2 + b / (c - a)^2 + c / (a - b)^2 = 0 :=
by
  sorry

end value_expression_eq_zero_l5_5423


namespace polynomial_divisible_by_x_minus_2_l5_5824

theorem polynomial_divisible_by_x_minus_2 (k : ℝ) :
  (2 * (2 : ℝ)^3 - 8 * (2 : ℝ)^2 + k * (2 : ℝ) - 10 = 0) → 
  k = 13 :=
by 
  intro h
  sorry

end polynomial_divisible_by_x_minus_2_l5_5824


namespace range_contains_pi_div_4_l5_5082

noncomputable def f (x : ℝ) : ℝ :=
  Real.arctan x + Real.arctan ((2 - x) / (2 + x))

theorem range_contains_pi_div_4 : ∃ x : ℝ, f x = (Real.pi / 4) := by
  sorry

end range_contains_pi_div_4_l5_5082


namespace natalie_blueberry_bushes_l5_5970

-- Definitions of the conditions
def bushes_yield_containers (bushes containers : ℕ) : Prop :=
  containers = bushes * 7

def containers_exchange_zucchinis (containers zucchinis : ℕ) : Prop :=
  zucchinis = containers * 3 / 7

-- Theorem statement
theorem natalie_blueberry_bushes (zucchinis_needed : ℕ) (zucchinis_per_trade containers_per_trade bushes_per_container : ℕ) 
  (h1 : zucchinis_per_trade = 3) (h2 : containers_per_trade = 7) (h3 : bushes_per_container = 7) 
  (h4 : zucchinis_needed = 63) : 
  ∃ bushes_needed : ℕ, bushes_needed = 21 := 
by
  sorry

end natalie_blueberry_bushes_l5_5970


namespace total_distance_collinear_centers_l5_5643

theorem total_distance_collinear_centers (r1 r2 r3 : ℝ) (d12 d13 d23 : ℝ) 
  (h1 : r1 = 6) 
  (h2 : r2 = 14) 
  (h3 : d12 = r1 + r2) 
  (h4 : d13 = r3 - r1) 
  (h5 : d23 = r3 - r2) :
  d13 = d12 + r1 := by
  -- proof follows here
  sorry

end total_distance_collinear_centers_l5_5643


namespace marbles_left_l5_5310

def initial_marbles : ℕ := 64
def marbles_given : ℕ := 14

theorem marbles_left : (initial_marbles - marbles_given) = 50 := by
  sorry

end marbles_left_l5_5310


namespace total_spending_is_140_l5_5622

-- Define definitions for each day's spending based on the conditions.
def monday_spending : ℕ := 6
def tuesday_spending : ℕ := 2 * monday_spending
def wednesday_spending : ℕ := 2 * (monday_spending + tuesday_spending)
def thursday_spending : ℕ := (monday_spending + tuesday_spending + wednesday_spending) / 3
def friday_spending : ℕ := thursday_spending - 4
def saturday_spending : ℕ := friday_spending + (friday_spending / 2)
def sunday_spending : ℕ := tuesday_spending + saturday_spending

-- The total spending for the week.
def total_spending : ℕ := 
  monday_spending + 
  tuesday_spending + 
  wednesday_spending + 
  thursday_spending + 
  friday_spending + 
  saturday_spending + 
  sunday_spending

-- The theorem to prove that the total spending is $140.
theorem total_spending_is_140 : total_spending = 140 := 
  by {
    -- Due to the problem's requirement, we skip the proof steps.
    sorry
  }

end total_spending_is_140_l5_5622


namespace fraction_sum_l5_5836

theorem fraction_sum :
  (3 / 30 : ℝ) + (5 / 300) + (7 / 3000) = 0.119 := by
  sorry

end fraction_sum_l5_5836


namespace normal_CDF_is_correct_l5_5057

noncomputable def normal_cdf (a σ : ℝ) (x : ℝ) : ℝ :=
  0.5 + (1 / Real.sqrt (2 * Real.pi)) * ∫ t in (0)..(x - a) / σ, Real.exp (-t^2 / 2)

theorem normal_CDF_is_correct (a σ : ℝ) (ha : σ > 0) (x : ℝ) :
  (normal_cdf a σ x) = 0.5 + (1 / Real.sqrt (2 * Real.pi)) * ∫ t in (0)..(x - a) / σ, Real.exp (-t^2 / 2) :=
by
  sorry

end normal_CDF_is_correct_l5_5057


namespace time_for_C_alone_to_finish_the_job_l5_5027

variable {A B C : ℝ} -- Declare work rates as real numbers

-- Define the conditions
axiom h1 : A + B = 1/15
axiom h2 : A + B + C = 1/10

-- Define the theorem to prove
theorem time_for_C_alone_to_finish_the_job : C = 1/30 :=
by
  apply sorry

end time_for_C_alone_to_finish_the_job_l5_5027


namespace total_bedrooms_is_correct_l5_5073

def bedrooms_second_floor : Nat := 2
def bedrooms_first_floor : Nat := 8
def total_bedrooms (b1 b2 : Nat) : Nat := b1 + b2

theorem total_bedrooms_is_correct : total_bedrooms bedrooms_second_floor bedrooms_first_floor = 10 := 
by
  sorry

end total_bedrooms_is_correct_l5_5073


namespace compare_f_ln_l5_5547

variable {f : ℝ → ℝ}

theorem compare_f_ln (h : ∀ x : ℝ, deriv f x > f x) : 3 * f (Real.log 2) < 2 * f (Real.log 3) :=
by
  sorry

end compare_f_ln_l5_5547


namespace operation_result_l5_5288

def a : ℝ := 0.8
def b : ℝ := 0.5
def c : ℝ := 0.40

theorem operation_result :
  (a ^ 3 - b ^ 3 / a ^ 2 + c + b ^ 2) = 0.9666875 := by
  sorry

end operation_result_l5_5288


namespace closest_ratio_adults_children_l5_5572

theorem closest_ratio_adults_children (a c : ℕ) (h1 : 30 * a + 15 * c = 2250) (h2 : a ≥ 2) (h3 : c ≥ 2) : 
  (a : ℚ) / (c : ℚ) = 1 :=
  sorry

end closest_ratio_adults_children_l5_5572


namespace sum_of_ages_l5_5295

theorem sum_of_ages (a b c : ℕ) (h1 : a * b * c = 72) (h2 : b = c) (h3 : a < b) : a + b + c = 14 :=
sorry

end sum_of_ages_l5_5295


namespace set_properties_proof_l5_5994

open Set

variable (U : Set ℝ := univ)
variable (M : Set ℝ := Icc (-2 : ℝ) 2)
variable (N : Set ℝ := Iic (1 : ℝ))

theorem set_properties_proof :
  (M ∪ N = Iic (2 : ℝ)) ∧
  (M ∩ N = Icc (-2 : ℝ) 1) ∧
  (U \ N = Ioi (1 : ℝ)) := by
  sorry

end set_properties_proof_l5_5994


namespace marbles_count_l5_5104

variables {g y : ℕ}

theorem marbles_count (h1 : (g - 1)/(g + y - 1) = 1/8)
                      (h2 : g/(g + y - 3) = 1/6) :
                      g + y = 9 :=
by
-- This is just setting up the statements we need to prove the theorem. The actual proof is to be completed.
sorry

end marbles_count_l5_5104


namespace similar_triangles_perimeter_l5_5914

theorem similar_triangles_perimeter (P_small P_large : ℝ) 
  (h_ratio : P_small / P_large = 2 / 3) 
  (h_sum : P_small + P_large = 20) : 
  P_small = 8 := 
sorry

end similar_triangles_perimeter_l5_5914


namespace cos_of_tan_l5_5056

/-- Given a triangle ABC with angle A such that tan(A) = -5/12, prove cos(A) = -12/13. -/
theorem cos_of_tan (A : ℝ) (h : Real.tan A = -5 / 12) : Real.cos A = -12 / 13 := by
  sorry

end cos_of_tan_l5_5056


namespace sphere_radius_l5_5187

theorem sphere_radius 
  (r h1 h2 : ℝ)
  (A1_eq : 5 * π = π * (r^2 - h1^2))
  (A2_eq : 8 * π = π * (r^2 - h2^2))
  (h1_h2_eq : h1 - h2 = 1) : r = 3 :=
by
  sorry

end sphere_radius_l5_5187


namespace orange_gumdrops_after_replacement_l5_5380

noncomputable def total_gumdrops : ℕ :=
  100

noncomputable def initial_orange_gumdrops : ℕ :=
  10

noncomputable def initial_blue_gumdrops : ℕ :=
  40

noncomputable def replaced_blue_gumdrops : ℕ :=
  initial_blue_gumdrops / 3

theorem orange_gumdrops_after_replacement : 
  (initial_orange_gumdrops + replaced_blue_gumdrops) = 23 :=
by
  sorry

end orange_gumdrops_after_replacement_l5_5380


namespace geometric_sequence_x_value_l5_5374

theorem geometric_sequence_x_value (x : ℝ) (r : ℝ) 
  (h1 : 12 * r = x) 
  (h2 : x * r = 2 / 3) 
  (h3 : 0 < x) :
  x = 2 * Real.sqrt 2 :=
by
  sorry

end geometric_sequence_x_value_l5_5374


namespace similar_polygons_area_sum_l5_5754

theorem similar_polygons_area_sum (a b c k : ℝ) (t' t'' T : ℝ)
    (h₁ : t' = k * a^2)
    (h₂ : t'' = k * b^2)
    (h₃ : T = t' + t''):
    c^2 = a^2 + b^2 := 
by 
  sorry

end similar_polygons_area_sum_l5_5754


namespace complete_square_identity_l5_5063

theorem complete_square_identity (x : ℝ) : ∃ (d e : ℤ), (x^2 - 10 * x + 13 = 0 → (x + d)^2 = e ∧ d + e = 7) :=
sorry

end complete_square_identity_l5_5063


namespace maggie_earnings_l5_5662

def subscriptions_to_parents := 4
def subscriptions_to_grandfather := 1
def subscriptions_to_next_door_neighbor := 2
def subscriptions_to_another_neighbor := 2 * subscriptions_to_next_door_neighbor
def subscription_rate := 5

theorem maggie_earnings : 
  (subscriptions_to_parents + subscriptions_to_grandfather + subscriptions_to_next_door_neighbor + subscriptions_to_another_neighbor) * subscription_rate = 55 := 
by
  sorry

end maggie_earnings_l5_5662


namespace average_marks_of_passed_l5_5817

theorem average_marks_of_passed
  (total_boys : ℕ)
  (average_all : ℕ)
  (average_failed : ℕ)
  (passed_boys : ℕ)
  (num_boys := 120)
  (avg_all := 37)
  (avg_failed := 15)
  (passed := 110)
  (failed_boys := total_boys - passed_boys)
  (total_marks_all := average_all * total_boys)
  (total_marks_failed := average_failed * failed_boys)
  (total_marks_passed := total_marks_all - total_marks_failed)
  (average_passed := total_marks_passed / passed_boys) :
  average_passed = 39 :=
by
  -- start of proof
  sorry

end average_marks_of_passed_l5_5817


namespace domain_of_tan_function_l5_5167

theorem domain_of_tan_function :
  (∀ x : ℝ, ∀ k : ℤ, 2 * x - π / 4 ≠ k * π + π / 2 ↔ x ≠ (k * π) / 2 + 3 * π / 8) :=
sorry

end domain_of_tan_function_l5_5167


namespace remainder_when_sum_divided_by_5_l5_5175

/-- Reinterpreting the same conditions and question: -/
theorem remainder_when_sum_divided_by_5 (a b c : ℕ) 
    (ha : a < 5) (hb : b < 5) (hc : c < 5) 
    (h1 : a * b * c % 5 = 1) 
    (h2 : 3 * c % 5 = 2)
    (h3 : 4 * b % 5 = (3 + b) % 5): 
    (a + b + c) % 5 = 4 := 
sorry

end remainder_when_sum_divided_by_5_l5_5175


namespace max_n_base_10_l5_5651

theorem max_n_base_10:
  ∃ (A B C n: ℕ), (A < 5 ∧ B < 5 ∧ C < 5) ∧
                 (n = 25 * A + 5 * B + C) ∧ (n = 81 * C + 9 * B + A) ∧ 
                 (∀ (A' B' C' n': ℕ), 
                 (A' < 5 ∧ B' < 5 ∧ C' < 5) ∧ (n' = 25 * A' + 5 * B' + C') ∧ 
                 (n' = 81 * C' + 9 * B' + A') → n' ≤ n) →
  n = 111 :=
by {
    sorry
}

end max_n_base_10_l5_5651


namespace Lizzy_money_after_loan_l5_5186

theorem Lizzy_money_after_loan :
  let initial_savings := 30
  let loaned_amount := 15
  let interest_rate := 0.20
  let interest := loaned_amount * interest_rate
  let total_amount_returned := loaned_amount + interest
  let remaining_money := initial_savings - loaned_amount
  let total_money := remaining_money + total_amount_returned
  total_money = 33 :=
by
  sorry

end Lizzy_money_after_loan_l5_5186


namespace quadratic_no_real_roots_l5_5004

open Real

theorem quadratic_no_real_roots
  (p q a b c : ℝ)
  (hpq : p ≠ q)
  (hpositive_p : 0 < p)
  (hpositive_q : 0 < q)
  (hpositive_a : 0 < a)
  (hpositive_b : 0 < b)
  (hpositive_c : 0 < c)
  (h_geo_sequence : a^2 = p * q)
  (h_ari_sequence : b + c = p + q) :
  (a^2 - b * c) < 0 :=
by
  sorry

end quadratic_no_real_roots_l5_5004


namespace part1_solution_part2_solution_l5_5159

-- Definitions for costs
variables (x y : ℝ)
variables (cost_A cost_B : ℝ)

-- Conditions
def condition1 : 80 * x + 35 * y = 2250 :=
  sorry

def condition2 : x = y - 15 :=
  sorry

-- Part 1: Cost of one bottle of each disinfectant
theorem part1_solution : x = cost_A ∧ y = cost_B :=
  sorry

-- Additional conditions for part 2
variables (m : ℕ)
variables (total_bottles : ℕ := 50)
variables (budget : ℝ := 1200)

-- Conditions for part 2
def condition3 : m + (total_bottles - m) = total_bottles :=
  sorry

def condition4 : 15 * m + 30 * (total_bottles - m) ≤ budget :=
  sorry

-- Part 2: Minimum number of bottles of Class A disinfectant
theorem part2_solution : m ≥ 20 :=
  sorry

end part1_solution_part2_solution_l5_5159


namespace min_cuts_for_100_quadrilaterals_l5_5782

theorem min_cuts_for_100_quadrilaterals : ∃ n : ℕ, (∃ q : ℕ, q = 100 ∧ n + 1 = q + 99) ∧ n = 1699 :=
sorry

end min_cuts_for_100_quadrilaterals_l5_5782


namespace six_digit_square_number_cases_l5_5922

theorem six_digit_square_number_cases :
  ∃ n : ℕ, 316 ≤ n ∧ n < 1000 ∧ (n^2 = 232324 ∨ n^2 = 595984 ∨ n^2 = 929296) :=
by {
  sorry
}

end six_digit_square_number_cases_l5_5922


namespace total_wood_needed_l5_5607

theorem total_wood_needed : 
      (4 * 4 + 4 * (4 * 5)) + 
      (10 * 6 + 10 * (6 - 3)) + 
      (8 * 5.5) + 
      (6 * (5.5 * 2) + 6 * (5.5 * 1.5)) = 345.5 := 
by 
  sorry

end total_wood_needed_l5_5607


namespace unique_c1_c2_exists_l5_5859

theorem unique_c1_c2_exists (a_0 a_1 x_1 x_2 : ℝ) (h_distinct : x_1 ≠ x_2) : 
  ∃! (c_1 c_2 : ℝ), ∀ n : ℕ, a_n = c_1 * x_1^n + c_2 * x_2^n :=
sorry

end unique_c1_c2_exists_l5_5859


namespace plant_branches_l5_5590

theorem plant_branches (x : ℕ) (h : 1 + x + x^2 = 91) : 1 + x + x^2 = 91 :=
by sorry

end plant_branches_l5_5590


namespace flowers_in_each_row_l5_5294

theorem flowers_in_each_row (rows : ℕ) (total_remaining_flowers : ℕ) 
  (percentage_remaining : ℚ) (correct_rows : rows = 50) 
  (correct_remaining : total_remaining_flowers = 8000) 
  (correct_percentage : percentage_remaining = 0.40) :
  (total_remaining_flowers : ℚ) / percentage_remaining / (rows : ℚ) = 400 := 
by {
 sorry
}

end flowers_in_each_row_l5_5294


namespace walking_area_calculation_l5_5841

noncomputable def walking_area_of_park (park_length park_width fountain_radius : ℝ) : ℝ :=
  let park_area := park_length * park_width
  let fountain_area := Real.pi * fountain_radius^2
  park_area - fountain_area

theorem walking_area_calculation :
  walking_area_of_park 50 30 5 = 1500 - 25 * Real.pi :=
by
  sorry

end walking_area_calculation_l5_5841


namespace rahul_matches_l5_5170

theorem rahul_matches
  (initial_avg : ℕ)
  (runs_today : ℕ)
  (final_avg : ℕ)
  (n : ℕ)
  (H1 : initial_avg = 50)
  (H2 : runs_today = 78)
  (H3 : final_avg = 54)
  (H4 : (initial_avg * n + runs_today) = final_avg * (n + 1)) :
  n = 6 :=
by
  sorry

end rahul_matches_l5_5170


namespace ff_of_10_eq_2_l5_5657

noncomputable def f : ℝ → ℝ
| x => if x ≤ 1 then x^2 + 1 else Real.log x

theorem ff_of_10_eq_2 : f (f 10) = 2 :=
by
  sorry

end ff_of_10_eq_2_l5_5657


namespace percentage_palm_oil_in_cheese_l5_5408

theorem percentage_palm_oil_in_cheese
  (initial_cheese_price: ℝ := 100)
  (cheese_price_increase: ℝ := 3)
  (palm_oil_price_increase_percentage: ℝ := 0.10)
  (expected_palm_oil_percentage : ℝ := 30):
  ∃ (palm_oil_initial_price: ℝ),
  cheese_price_increase = palm_oil_initial_price * palm_oil_price_increase_percentage ∧
  expected_palm_oil_percentage = 100 * (palm_oil_initial_price / initial_cheese_price) := by
  sorry

end percentage_palm_oil_in_cheese_l5_5408


namespace remainder_when_divided_l5_5338

open Polynomial

noncomputable def poly : Polynomial ℚ := X^6 + X^5 + 2*X^3 - X^2 + 3
noncomputable def divisor : Polynomial ℚ := (X + 2) * (X - 1)
noncomputable def remainder : Polynomial ℚ := -X + 5

theorem remainder_when_divided :
  ∃ q : Polynomial ℚ, poly = divisor * q + remainder :=
sorry

end remainder_when_divided_l5_5338


namespace primes_sum_eq_2001_l5_5719

/-- If a and b are prime numbers such that a^2 + b = 2003, then a + b = 2001. -/
theorem primes_sum_eq_2001 (a b : ℕ) (ha : Nat.Prime a) (hb : Nat.Prime b) (h : a^2 + b = 2003) :
    a + b = 2001 := 
  sorry

end primes_sum_eq_2001_l5_5719


namespace max_y_value_l5_5361

theorem max_y_value (x : ℝ) : ∃ y : ℝ, y = -x^2 + 4 * x + 3 ∧ y ≤ 7 :=
by
  sorry

end max_y_value_l5_5361


namespace percent_of_total_is_correct_l5_5667

theorem percent_of_total_is_correct :
  (6.620000000000001 / 100 * 1000 = 66.2) :=
by
  sorry

end percent_of_total_is_correct_l5_5667


namespace B_pow_5_eq_r_B_add_s_I_l5_5405

def B : Matrix (Fin 2) (Fin 2) ℤ := ![![ -2,  3 ], 
                                      ![  4,  5 ]]

noncomputable def I : Matrix (Fin 2) (Fin 2) ℤ := 1

theorem B_pow_5_eq_r_B_add_s_I :
  ∃ r s : ℤ, (r = 425) ∧ (s = 780) ∧ (B^5 = r • B + s • I) :=
by
  sorry

end B_pow_5_eq_r_B_add_s_I_l5_5405


namespace fraction_of_meat_used_for_meatballs_l5_5009

theorem fraction_of_meat_used_for_meatballs
    (initial_meat : ℕ)
    (spring_rolls_meat : ℕ)
    (remaining_meat : ℕ)
    (total_meat_used : ℕ)
    (meatballs_meat : ℕ)
    (h_initial : initial_meat = 20)
    (h_spring_rolls : spring_rolls_meat = 3)
    (h_remaining : remaining_meat = 12) :
    (initial_meat - remaining_meat) = total_meat_used ∧
    (total_meat_used - spring_rolls_meat) = meatballs_meat ∧
    (meatballs_meat / initial_meat) = (1/4 : ℝ) :=
by
  sorry

end fraction_of_meat_used_for_meatballs_l5_5009
