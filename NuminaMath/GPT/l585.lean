import Mathlib

namespace sum_of_possible_lengths_l585_585120

theorem sum_of_possible_lengths
  (m : ℕ) 
  (h1 : m < 18)
  (h2 : m > 4) : ∑ i in (Finset.range 13).map (λ x, x + 5) = 143 := by
sorry

end sum_of_possible_lengths_l585_585120


namespace two_digit_remainder_one_when_divided_by_4_and_17_l585_585502

-- Given the conditions
def is_two_digit (n : ℕ) : Prop := n ≥ 10 ∧ n < 100
def yields_remainder (n d r : ℕ) : Prop := n % d = r

-- Define the main problem that checks if there is only one such number
theorem two_digit_remainder_one_when_divided_by_4_and_17 :
  ∃! n : ℕ, is_two_digit n ∧ yields_remainder n 4 1 ∧ yields_remainder n 17 1 :=
sorry

end two_digit_remainder_one_when_divided_by_4_and_17_l585_585502


namespace tetrahedron_a_exists_tetrahedron_b_not_exists_l585_585343

/-- Part (a): There exists a tetrahedron with two edges shorter than 1 cm,
    and the other four edges longer than 1 km. -/
theorem tetrahedron_a_exists : 
  ∃ (a b c d : ℝ), 
    ((a < 1 ∧ b < 1 ∧ 1000 < c ∧ 1000 < d ∧ 1000 < (a + c) ∧ 1000 < (b + d)) ∧ 
     a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ d ≠ a) := 
sorry

/-- Part (b): There does not exist a tetrahedron with four edges shorter than 1 cm,
    and the other two edges longer than 1 km. -/
theorem tetrahedron_b_not_exists : 
  ¬ ∃ (a b c d : ℝ), 
    ((a < 1 ∧ b < 1 ∧ c < 1 ∧ d < 1 ∧ 1000 < (a + c) ∧ 1000 < (b + d)) ∧ 
     a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ d ≠ a ) := 
sorry

end tetrahedron_a_exists_tetrahedron_b_not_exists_l585_585343


namespace bakery_batches_per_day_l585_585422

-- Definitions for the given problem's conditions
def baguettes_per_batch := 48
def baguettes_sold_batch1 := 37
def baguettes_sold_batch2 := 52
def baguettes_sold_batch3 := 49
def baguettes_left := 6

-- Theorem stating the number of batches made
theorem bakery_batches_per_day : 
  (baguettes_sold_batch1 + baguettes_sold_batch2 + baguettes_sold_batch3 + baguettes_left) / baguettes_per_batch = 3 :=
by 
  sorry

end bakery_batches_per_day_l585_585422


namespace correct_product_l585_585935

theorem correct_product (a b : ℕ) (h1 : 9 < a ∧ a < 100) (h2 : Prime b)
  (a' : ℕ) (h3 : a' = nat.reverse_digits 10 a) (h4 : a' * b = 280) : a * b = 28 :=
sorry

end correct_product_l585_585935


namespace no_distinct_ending_digits_for_all_sums_of_five_numbers_l585_585225

theorem no_distinct_ending_digits_for_all_sums_of_five_numbers :
  ¬ ∃ (a b c d e : ℕ),
      let S := [a + b, a + c, a + d, a + e,
                b + c, b + d, b + e,
                c + d, c + e,
                d + e] in
      (∀ i j, i ≠ j → (S[i] % 10) ≠ (S[j] % 10)) :=
by
  sorry

end no_distinct_ending_digits_for_all_sums_of_five_numbers_l585_585225


namespace range_of_m_l585_585237

noncomputable def f (x m : ℝ) : ℝ :=
  Real.exp x * (Real.log x + 0.5 * x^2 - m * x)

noncomputable def f' (x m : ℝ) : ℝ :=
  Real.exp x * (Real.log x + 0.5 * x^2 - m * x) + Real.exp x * (1 / x + x - m)

theorem range_of_m (m : ℝ) : (∀ x : ℝ, 0 < x → f'(x, m) - f(x, m) > 0) → m < 2 :=
begin
  sorry
end

end range_of_m_l585_585237


namespace probability_four_even_draws_l585_585071

noncomputable def even_numbers := {0, 2, 4, 6, 8}
noncomputable def total_numbers := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

theorem probability_four_even_draws : 
  let event := (even_numbers.to_finset.filter (λ n, n % 2 = 0)),
      total := total_numbers.to_finset in
  (event.card * (event.card - 1) * (event.card - 2) * (event.card - 3)) / 
    (total.card * (total.card - 1) * (total.card - 2) * (total.card - 3)) = 1 / 42 := 
  by sorry

end probability_four_even_draws_l585_585071


namespace turtles_on_sand_l585_585537

variable (total_turtles : ℕ)
variable (first_wave_fraction second_wave_fraction : ℚ)

theorem turtles_on_sand (h_total : total_turtles = 240)
  (h_first_wave : first_wave_fraction = 1/3)
  (h_second_wave : second_wave_fraction = 1/5) :
  let turtles_after_first_wave := total_turtles - (total_turtles * first_wave_fraction).nat_abs in
  let turtles_after_second_wave := turtles_after_first_wave - (turtles_after_first_wave * second_wave_fraction).nat_abs in
  turtles_after_second_wave = 128 := 
by
  sorry

end turtles_on_sand_l585_585537


namespace possible_values_ceil_x_squared_l585_585669

theorem possible_values_ceil_x_squared (x : ℝ) (h : ⌈x⌉ = 9) : (finset.Icc 65 81).card = 17 := by
  sorry

end possible_values_ceil_x_squared_l585_585669


namespace area_enclosed_by_curve_and_line_l585_585784

noncomputable def enclosed_area : ℝ :=
  ∫ x in -3..1, (3 - x^2 - 2*x)

theorem area_enclosed_by_curve_and_line : enclosed_area = 32/3 := by
  sorry

end area_enclosed_by_curve_and_line_l585_585784


namespace max_ones_in_table_l585_585685

theorem max_ones_in_table (M : Matrix (Fin 3) (Fin 3) ℕ)
  (h_distinct_products : (Π i j, list.prod (M i) ≠ list.prod (M j)) ∧ (Π i j, list.prod (λ k => M k i) ≠ list.prod (λ k => M k j))) :
  ∃ k, (finset.filter (= 1) (Fin 3.univ.bUnion (λ i => Fin 3.univ.map (M i))).card = k) ∧ k = 5 :=
sorry

end max_ones_in_table_l585_585685


namespace bond_interest_percentage_l585_585353

open Real

theorem bond_interest_percentage (face_value selling_price: ℝ) (H1 : face_value = 5000) (H2 : selling_price ≈ 7692.307692307692) (yield: ℝ) (H3 : yield = 0.10 * face_value) :
  ((yield / selling_price) * 100) ≈ 6.5 :=
by
  sorry

end bond_interest_percentage_l585_585353


namespace circle_form_eq_standard_form_l585_585786

theorem circle_form_eq_standard_form :
  ∀ (x y : ℝ), x^2 + y^2 + 2*x - 4*y - 6 = 0 ↔ (x + 1)^2 + (y - 2)^2 = 11 := 
by
  intro x y
  sorry

end circle_form_eq_standard_form_l585_585786


namespace find_square_l585_585019

theorem find_square (s : ℕ) : 
    (7863 / 13 = 604 + (s / 13)) → s = 11 :=
by
  sorry

end find_square_l585_585019


namespace sequence_nth_term_mod_2500_l585_585920

def sequence_nth_term (n : ℕ) : ℕ :=
  -- this is a placeholder function definition; the actual implementation to locate the nth term is skipped
  sorry

theorem sequence_nth_term_mod_2500 : (sequence_nth_term 2500) % 7 = 1 := 
sorry

end sequence_nth_term_mod_2500_l585_585920


namespace sin_theta_intrinsic_pattern_l585_585194

theorem sin_theta_intrinsic_pattern (r : ℝ → ℝ) (t : ℝ) :
  (∀ θ, 0 ≤ θ ∧ θ ≤ t → r θ = Real.sin θ) → t = 2 * Real.pi :=
begin
  sorry,
end

end sin_theta_intrinsic_pattern_l585_585194


namespace remainder_of_division_l585_585591

-- Definitions of the polynomials
def f (x : ℚ) : ℚ := 3 * x^3 + (254/25) * x^2 + 5 * x - 15
def g (x : ℚ) : ℚ := 3 * x + 5
def r : ℚ := -9

-- The main theorem stating the remainder result
theorem remainder_of_division : ∃ Q : polynomial ℚ, f - Q * g = polynomial.C r :=
by sorry

end remainder_of_division_l585_585591


namespace product_of_tangents_l585_585610

-- Definitions based on the given conditions
def tangent_to_unit_circle (l : Line) (S : Circle) (P : Point) : Prop :=
  is_tangent l S P

def on_same_side (A : Point) (l : Line) (S : Circle) : Prop :=
  same_side A l S

def distance_from_point_to_line (A : Point) (l : Line) (h : Real) : Prop :=
  dist_point_line A l = h ∧ h > 2

def tangents_from_point_to_circle (A : Point) (S : Circle) (B C : Point) : Prop :=
  are_tangents_from_point A S [B, C]

-- Main theorem statement
theorem product_of_tangents 
  (l : Line) (S : Circle) (P A B C : Point) (h : Real)
  (hl_tangent : tangent_to_unit_circle l S P)
  (h_A_side : on_same_side A l S)
  (h_dist : distance_from_point_to_line A l h)
  (h_tangents : tangents_from_point_to_circle A S B C) : 
  (pb_pc_product : Real) := 
sorry

end product_of_tangents_l585_585610


namespace complex_eq_l585_585308

theorem complex_eq : ∀ (z : ℂ), (i * z = i + z) → (z = (1 - i) / 2) :=
by
  intros z h
  sorry

end complex_eq_l585_585308


namespace angle_between_generatrix_and_base_of_cone_l585_585101

theorem angle_between_generatrix_and_base_of_cone (r R H : ℝ) (α : ℝ)
  (h_cylinder_height : H = 2 * R)
  (h_total_surface_area : 2 * Real.pi * r * H + 2 * Real.pi * r^2 = Real.pi * R^2) :
  α = Real.arctan (2 * (4 + Real.sqrt 6) / 5) :=
sorry

end angle_between_generatrix_and_base_of_cone_l585_585101


namespace sum_last_three_coefficients_l585_585062

theorem sum_last_three_coefficients (a : ℝ) (h : a ≠ 0) : 
  let expr := (1 - 1 / a)^7 in
  let coeffs := (List.range 8).reverse.map (λ k, (Nat.choose 7 k) * (-1) ^ k) in
  let last_three := coeffs.take 3 in
  last_three.sum = 15 := 
by sorry

end sum_last_three_coefficients_l585_585062


namespace ordered_triple_count_l585_585895

theorem ordered_triple_count :
  ∃ n: ℕ, 
    n = 4 ∧ 
    ∃ (a b c: ℕ), 
      1 ≤ a ∧ a ≤ b ∧ b ≤ c ∧ c ≤ 10 ∧ 
      3 * (a * b + b * c + c * a) = a * b * c :=
begin
  sorry
end

end ordered_triple_count_l585_585895


namespace focal_length_of_ellipse_l585_585788

theorem focal_length_of_ellipse (c : ℝ) : 
  (∀ x y : ℝ, 2 * x^2 + 3 * y^2 = 6 → c = real.sqrt 1) → (2 * c = 2) :=
by
sorry

end focal_length_of_ellipse_l585_585788


namespace minimum_zeros_in_2011_numbers_l585_585809

theorem minimum_zeros_in_2011_numbers (s : Finset ℤ) :
  s.card = 2011 →
  (∀ a b c ∈ s, a + b + c ∈ s) →
  ∃ zero_count : ℕ, zero_count ≥ 2009 ∧
    ∀ t ⊆ s, t.card = 2011 → 
    ∀ a b c ∈ t, a + b + c ∈ t → 
    (∃ z : Finset ℤ, z.card = zero_count ∧ ∀ x ∈ z, x = 0) :=
by sorry

end minimum_zeros_in_2011_numbers_l585_585809


namespace count_even_positive_4_digit_integers_no_5_l585_585665

-- Conditions definitions
def is_4_digit_number (n : ℕ) : Prop := 1000 ≤ n ∧ n < 10000

def is_even (n : ℕ) : Prop := n % 2 = 0

def is_divisible_by_5 (n : ℕ) : Prop := n % 5 = 0

def does_not_contain_digit_5 (n : ℕ) : Prop :=
  ∀ d ∈ n.digits 10, d ≠ 5

-- Lean statement for the proof problem
theorem count_even_positive_4_digit_integers_no_5 :
  { n : ℕ | is_4_digit_number n ∧ is_even n ∧ is_divisible_by_5 n ∧ does_not_contain_digit_5 n }.card = 648 :=
sorry

end count_even_positive_4_digit_integers_no_5_l585_585665


namespace brother_reading_time_l585_585754

variable (my_time_in_hours : ℕ)
variable (speed_ratio : ℕ)

theorem brother_reading_time
  (h1 : my_time_in_hours = 3)
  (h2 : speed_ratio = 4) :
  my_time_in_hours * 60 / speed_ratio = 45 := 
by
  sorry

end brother_reading_time_l585_585754


namespace smallest_period_and_min_value_l585_585598

def f (x : ℝ) : ℝ := (real.sqrt 3) * real.sin x * real.cos x

theorem smallest_period_and_min_value :
  (∃ T > 0, ∀ x, f (x + T) = f x ∧ T = π) ∧ (∃ x, f x = - (real.sqrt 3) / 2) :=
sorry

end smallest_period_and_min_value_l585_585598


namespace triangle_properties_l585_585693

noncomputable def triangle_side_lengths (m1 m2 m3 : ℝ) : Prop :=
  ∃ a b c s,
    m1 = 20 ∧
    m2 = 24 ∧
    m3 = 30 ∧
    a = 36.28 ∧
    b = 30.24 ∧
    c = 24.19 ∧
    s = 362.84

theorem triangle_properties :
  triangle_side_lengths 20 24 30 :=
by
  sorry

end triangle_properties_l585_585693


namespace linda_babysitting_hours_l585_585742

-- Define constants
def hourly_wage : ℝ := 10.0
def application_fee : ℝ := 25.0
def number_of_colleges : ℝ := 6.0

-- Theorem statement
theorem linda_babysitting_hours : 
    (application_fee * number_of_colleges) / hourly_wage = 15 := 
by
  -- Here the proof would go, but we'll use sorry as per instructions
  sorry

end linda_babysitting_hours_l585_585742


namespace exists_F_squared_l585_585764

theorem exists_F_squared (n : ℕ) : ∃ F : ℕ → ℕ, ∀ n : ℕ, (F (F n)) = n^2 := 
sorry

end exists_F_squared_l585_585764


namespace solve_for_x_l585_585417

theorem solve_for_x : 
  ∀ (x : ℝ), (∀ (a b : ℝ), a * b = 4 * a - 2 * b) → (3 * (6 * x) = -2) → (x = 17 / 2) :=
by
  sorry

end solve_for_x_l585_585417


namespace candies_eaten_l585_585166

-- Definitions

def Andrey_rate_eq_Boris_rate (candies_eaten_by_Andrey candies_eaten_by_Boris : ℕ) : Prop :=
  candies_eaten_by_Andrey / 4 = candies_eaten_by_Boris / 3

def Denis_rate_eq_Andrey_rate (candies_eaten_by_Denis candies_eaten_by_Andrey : ℕ) : Prop :=
  candies_eaten_by_Denis / 7 = candies_eaten_by_Andrey / 6

def total_candies (candies_eaten_by_Andrey candies_eaten_by_Boris candies_eaten_by_Denis : ℕ) : Prop :=
  candies_eaten_by_Andrey + candies_eaten_by_Boris + candies_eaten_by_Denis = 70

-- Theorem to prove the candies eaten by Andrey, Boris, and Denis
theorem candies_eaten (candies_eaten_by_Andrey candies_eaten_by_Boris candies_eaten_by_Denis : ℕ) :
  Andrey_rate_eq_Boris_rate candies_eaten_by_Andrey candies_eaten_by_Boris →
  Denis_rate_eq_Andrey_rate candies_eaten_by_Denis candies_eaten_by_Andrey →
  total_candies candies_eaten_by_Andrey candies_eaten_by_Boris candies_eaten_by_Denis →
  candies_eaten_by_Andrey = 24 ∧ candies_eaten_by_Boris = 18 ∧ candies_eaten_by_Denis = 28 :=
  by sorry

end candies_eaten_l585_585166


namespace smallest_positive_period_and_monotonicity_value_of_b_in_triangle_l585_585640

noncomputable def f (x : ℝ) : ℝ := (sqrt 3) * sin (2 * x) + cos (2 * x) - 1

theorem smallest_positive_period_and_monotonicity :
  (∃ T > 0, ∀ x : ℝ, f (x + T) = f x ∧ T = π) ∧
  (∃ k : ℤ, ∀ x : ℝ, -π / 3 + k * π ≤ x ∧ x ≤ π / 6 + k * π) :=
sorry

theorem value_of_b_in_triangle
  {A B C : ℝ}
  {a b c : ℝ}
  (hA : ∠A = ∠B + ∠C)
  (hB : ∠B = π / 3)
  (h_sum : a + c = 4)
  (h_prod : a * c = 3)
  (h_dot : (a * c) * cos B = 3 / 2) : b = sqrt 7 :=
sorry

end smallest_positive_period_and_monotonicity_value_of_b_in_triangle_l585_585640


namespace transfer_students_increase_average_l585_585470

theorem transfer_students_increase_average
    (avgA : ℚ) (numA : ℕ) (avgB : ℚ) (numB : ℕ)
    (kalinina_grade : ℚ) (sidorova_grade : ℚ)
    (avgA_init : avgA = 44.2) (numA_init : numA = 10)
    (avgB_init : avgB = 38.8) (numB_init : numB = 10)
    (kalinina_init : kalinina_grade = 41)
    (sidorova_init : sidorova_grade = 44) : 
    let new_avg_B_k := (avgB * numB + kalinina_grade) / (numB + 1) in
    let new_avg_A_s := (avgA * numA - sidorova_grade) / (numA - 1) in
    let new_avg_A_both := (avgA * numA - kalinina_grade - sidorova_grade) / (numA - 2) in
    let new_avg_B_both := (avgB * numB + kalinina_grade + sidorova_grade) / (numB + 2) in
    (new_avg_B_k <= avgB) ∧ (new_avg_A_s <= avgA) ∧ (new_avg_A_both > avgA) ∧ (new_avg_B_both > avgB) :=
by
  sorry

end transfer_students_increase_average_l585_585470


namespace g_zero_value_l585_585366

variables {R : Type*} [Ring R]

def polynomial_h (f g h : Polynomial R) : Prop :=
  h = f * g

def constant_term (p : Polynomial R) : R :=
  p.coeff 0

variables {f g h : Polynomial R}

theorem g_zero_value
  (Hf : constant_term f = 6)
  (Hh : constant_term h = -18)
  (H : polynomial_h f g h) :
  g.coeff 0 = -3 :=
by
  sorry

end g_zero_value_l585_585366


namespace bumper_cars_initial_count_l585_585562

variable {X : ℕ}

theorem bumper_cars_initial_count (h : (X - 6) + 3 = 6) : X = 9 := 
by
  sorry

end bumper_cars_initial_count_l585_585562


namespace value_of_expression_l585_585668

theorem value_of_expression (m n : ℝ) (h : m + n = -2) : 5 * m^2 + 5 * n^2 + 10 * m * n = 20 := 
by
  sorry

end value_of_expression_l585_585668


namespace noncongruent_triangles_l585_585324

noncomputable def isosceles_triangle (A B C : Type*) : Prop :=
∀ (P Q R : Prop), is_midpoint P A B ∧ is_midpoint Q B C ∧ is_midpoint R C A ∧ is_centroid S A B C

def is_midpoint {V : Type*} [Affine V (EuclideanSpace V)] (P : Point V) (A B : Point V) : Prop :=
Dist P A = Dist P B ∧ LineThrough P B ∧ LineThrough P A 

def is_centroid {V : Type*} [Affine V (EuclideanSpace V)] (S : Point V) (A B C : Point V) : Prop :=
Barycenter S [A, B, C] ∧ Dist S B₁ = ⅓ * sum (dist B S)

theorem noncongruent_triangles (A B C P Q R S : Type*) [Isosceles_triangle A B C] :
   num_noncongruent_triangles [A, B, C, P, Q, R, S] = 5 := sorry

end noncongruent_triangles_l585_585324


namespace shapeB_is_symmetric_to_original_l585_585494

-- Assume a simple type to represent our shapes
inductive Shape
| shapeA
| shapeB
| shapeC
| shapeD
| shapeE
| originalShape

-- Define the symmetry condition
def is_symmetric (s1 s2 : Shape) : Prop := sorry  -- this would be the condition to check symmetry

-- The theorem to prove that shapeB is symmetric to the original shape
theorem shapeB_is_symmetric_to_original :
  is_symmetric Shape.shapeB Shape.originalShape :=
sorry

end shapeB_is_symmetric_to_original_l585_585494


namespace total_kids_attended_camp_l585_585580

theorem total_kids_attended_camp :
  let n1 := 34044
  let n2 := 424944
  n1 + n2 = 458988 := 
by {
  sorry
}

end total_kids_attended_camp_l585_585580


namespace seq_a_n_sum_b_n_no_arith_seq_l585_585281

variable {n : ℕ} (a b : ℕ → ℕ)

-- Define the sequence a_n and its properties
def a (n : ℕ) : ℕ := (2 ^ n - 1) * 3

def S (n : ℕ) : ℕ := 
if n = 0 then 0 else (a n) + S (n - 1) 

-- Proving explicit formula for the sequence a_n
theorem seq_a_n (n : ℕ) : a n = 3 * (2^n - 1) := sorry

-- Define sequence b_n
def b (n : ℕ) : ℕ := (n * a n) / 3

def sum_b (n : ℕ) : ℕ :=
if n = 0 then 0 else b n + sum_b (n - 1)

-- Proving the sum of the first n terms of b_n sequence
theorem sum_b_n (n : ℕ) : sum_b n = 2 + (n - 1) * 2^(n + 1) - n * (n + 1) / 2 := sorry

-- Proving nonexistence of four terms in a_n forming an arithmetic sequence
theorem no_arith_seq (m n p q : ℕ) (h₁ : m < n) (h₂ : n < p) (h₃ : p < q) : 
    3 * (2^m - 1) + 3 * (2^q - 1) ≠ 3 * (2^n - 1) + 3 * (2^p - 1) := sorry

end seq_a_n_sum_b_n_no_arith_seq_l585_585281


namespace line_inclination_and_intersection_length_l585_585260

theorem line_inclination_and_intersection_length :
  let l_x (t : ℝ) := (1 / 2) * t
      l_y (t : ℝ) := (sqrt 2 / 2) + (sqrt 3 / 2) * t
      rho (θ : ℝ) := 2 * cos (θ - π / 4)
  in
  let inclination := 60
      A B : ℝ
  in
    (exists θ, l_y A = l_y B ∧ l_x A = l_x B ∧
               let d := abs (sqrt (A^2 + B^2))
               in d = sqrt 10 / 2) :=
  sorry

end line_inclination_and_intersection_length_l585_585260


namespace common_ratio_3_l585_585323

variable {a : ℕ → ℝ}
variable {d : ℝ}
variable {a2 a3 a6 : ℝ}

-- Definitions from conditions
def arithmetic_seq (a : ℕ → ℝ) := ∀ n, a (n + 1) = a n + d

axiom h1 : arithmetic_seq a
axiom h2 : d ≠ 0
axiom h3 : (a 2 + d)^2 = a 2 * (a 2 + 4 * d)

-- The Lean statement of the proof problem
theorem common_ratio_3 : ∃ q, q = 3 :=
by
  let q := (a 3) / (a 2)
  have q_def : q = 3 := sorry
  exact ⟨q, q_def⟩

end common_ratio_3_l585_585323


namespace family_member_bites_count_l585_585497

-- Definitions based on the given conditions
def cyrus_bites_arms_and_legs : Nat := 14
def cyrus_bites_body : Nat := 10
def family_size : Nat := 6
def total_bites_cyrus : Nat := cyrus_bites_arms_and_legs + cyrus_bites_body
def total_bites_family : Nat := total_bites_cyrus / 2

-- Translation of the question to a theorem statement
theorem family_member_bites_count : (total_bites_family / family_size) = 2 := by
  -- use sorry to indicate the proof is skipped
  sorry

end family_member_bites_count_l585_585497


namespace largest_subset_nat_1_to_3000_not_diff_1_4_5_l585_585054

/-- 
The largest set of integers that can be selected from {1, ..., 3000}
such that the difference between any two of them is not 1, 4, or 5
has 1000 elements.
-/
theorem largest_subset_nat_1_to_3000_not_diff_1_4_5 : 
  ∃ (s : Finset ℕ), s.card = 1000 ∧ (∀ x y ∈ s, x ≠ y → (x - y).natAbs ≠ 1 ∧ (x - y).natAbs ≠ 4 ∧ (x - y).natAbs ≠ 5) :=
sorry

end largest_subset_nat_1_to_3000_not_diff_1_4_5_l585_585054


namespace total_files_deleted_l585_585066

variable (deleted_pictures : ℕ := 5)
variable (deleted_songs : ℕ := 12)
variable (deleted_text_files : ℕ := 10)
variable (deleted_video_files : ℕ := 6)

theorem total_files_deleted (deleted_pictures deleted_songs deleted_text_files deleted_video_files : ℕ)
  (h1 : deleted_pictures = 5)
  (h2 : deleted_songs = 12)
  (h3 : deleted_text_files = 10)
  (h4 : deleted_video_files = 6) :
  deleted_pictures + deleted_songs + deleted_text_files + deleted_video_files = 33 :=
by
  rw [h1, h2, h3, h4]
  norm_num

end sorry

end total_files_deleted_l585_585066


namespace solve_matrix_system_l585_585411

theorem solve_matrix_system :
  let A := !![ [ 1, 2, 4 ], [ 2, 1, 5 ], [ 3, 2, 6 ] ]
      B := !![ [ 5 ], [ 7 ], [ 9 ] ]
      X := !![ [ 1 ], [ 0 ], [ 1 ] ]
      A_inv := !![ [-2/3, -2/3, 1], [1/2, -1, 1/2], [1/6, 2/3, -1/2] ]
  in A_inv * B = X := 
  sorry

end solve_matrix_system_l585_585411


namespace find_x0_l585_585739

noncomputable def f : ℝ → ℝ :=
λ x, if x < 1 then x^2 - 2 * x - 2 else 2 * x - 3

theorem find_x0 (x₀ : ℝ) (h : f x₀ = 1) : x₀ = -1 ∨ x₀ = 2 :=
sorry

end find_x0_l585_585739


namespace largest_subset_count_l585_585052

theorem largest_subset_count :
  ∃ (A : Set ℕ), A ⊆ Finset.range 3000 ∧ 
  (∀ a b ∈ A, a ≠ b → |a - b| ≠ 1 ∧ |a - b| ≠ 4 ∧ |a - b| ≠ 5) ∧ 
  A.card = 1000 := sorry

end largest_subset_count_l585_585052


namespace largest_common_multiple_3_5_l585_585947

theorem largest_common_multiple_3_5 (n : ℕ) :
  (n < 10000) ∧ (n ≥ 1000) ∧ (n % 3 = 0) ∧ (n % 5 = 0) → n ≤ 9990 :=
sorry

end largest_common_multiple_3_5_l585_585947


namespace divide_equal_parts_l585_585823

theorem divide_equal_parts (m n: ℕ) (h₁: (m + n) % 2 = 0) (h₂: gcd m n ∣ ((m + n) / 2)) : ∃ a b: ℕ, a = b ∧ a + b = m + n ∧ a ≤ m + n ∧ b ≤ m + n :=
sorry

end divide_equal_parts_l585_585823


namespace repeating_decimal_sum_l585_585942

noncomputable def repeating_decimal_6 : ℚ := 2 / 3
noncomputable def repeating_decimal_2 : ℚ := 2 / 9
noncomputable def repeating_decimal_4 : ℚ := 4 / 9
noncomputable def repeating_decimal_7 : ℚ := 7 / 9

theorem repeating_decimal_sum : 
  repeating_decimal_6 + repeating_decimal_2 - repeating_decimal_4 - repeating_decimal_7 = -1 / 3 :=
by {
  sorry
}

end repeating_decimal_sum_l585_585942


namespace interval_of_monotonic_increase_and_min_value_l585_585792

-- Define the function f(x)
def f (x : ℝ) : ℝ := log 2 (x^2 - 2*x + 3)

-- Function to calculate the derivative
noncomputable def f_deriv := deriv f

-- The main theorem asserting the interval of monotonic increase and the minimum value
theorem interval_of_monotonic_increase_and_min_value : 
  (∀ x > 1, f' x > 0) ∧ f 1 = 1 :=
  sorry

end interval_of_monotonic_increase_and_min_value_l585_585792


namespace factor_ax2_minus_ay2_l585_585801

variable (a x y : ℝ)

theorem factor_ax2_minus_ay2 : a * x^2 - a * y^2 = a * (x + y) * (x - y) := 
sorry

end factor_ax2_minus_ay2_l585_585801


namespace OP_perpendicular_AB_l585_585356

-- Define prerequisites for the problem
variables {A B C D P X O : Type}

-- Explicitly state the geometric relationships
def quadrilateral_contains_point (A B C D P : Type) : Prop := sorry
def circumcircles_congruent (P A B C D : Type) : Prop := sorry
def lines_intersect (A D B C : Type) (X : Type) : Prop := sorry
def is_circumcenter_of_triangle (O : Type) (X C D : Type) : Prop := sorry

-- Main statement of the problem
theorem OP_perpendicular_AB (A B C D P X O : Type) 
  (h1 : quadrilateral_contains_point A B C D P)
  (h2 : circumcircles_congruent P A B C D)
  (h3 : lines_intersect A D B C X)
  (h4 : is_circumcenter_of_triangle O X C D) : 
  OP ⊥ AB :=
sorry

end OP_perpendicular_AB_l585_585356


namespace find_quadratic_function_l585_585214

def quadratic_function (c d : ℝ) (x : ℝ) : ℝ :=
  x^2 + c * x + d

theorem find_quadratic_function :
  ∃ c d, (∀ x, 
    (quadratic_function c d (quadratic_function c d x + 2 * x)) / (quadratic_function c d x) = 2 * x^2 + 1984 * x + 2024) ∧ 
    quadratic_function c d x = x^2 + 1982 * x + 21 :=
by
  sorry

end find_quadratic_function_l585_585214


namespace derivative_f1_derivative_f2_l585_585928

noncomputable def f1 (x : ℝ) : ℝ := sin x / (1 + sin x)
noncomputable def f2 (x : ℝ) : ℝ := x * tan x

theorem derivative_f1 (x : ℝ) : deriv (λ x, f1 x) x = cos x / (1 + sin x)^2 :=
by sorry

theorem derivative_f2 (x : ℝ) : deriv (λ x, f2 x) x = sin x / cos x + x / cos x^2 :=
by sorry

end derivative_f1_derivative_f2_l585_585928


namespace smallest_four_digit_multiple_of_18_l585_585958

theorem smallest_four_digit_multiple_of_18 : ∃ n: ℕ, (1000 ≤ n ∧ n < 10000 ∧ 18 ∣ n) ∧ ∀ m: ℕ, (1000 ≤ m ∧ m < n ∧ 18 ∣ m) → false :=
begin
  use 1008,
  split,
  { split,
    { -- prove 1000 ≤ 1008
      linarith,
    },
    { split,
      { -- prove 1008 < 10000
        linarith,
      },
      { -- prove 18 ∣ 1008
        norm_num,
      }
    }
  },
  { -- prove there is no four-digit multiple of 18 less than 1008
    intros m,
    intro h,
    cases h with h1 h2,
    cases h2 with h3 h4,
    linarith,
  }
end

end smallest_four_digit_multiple_of_18_l585_585958


namespace smallest_degree_of_f_l585_585863

theorem smallest_degree_of_f (p : Polynomial ℂ) (hp_deg : p.degree < 1992)
  (hp0 : p.eval 0 ≠ 0) (hp1 : p.eval 1 ≠ 0) (hp_1 : p.eval (-1) ≠ 0) :
  ∃ f g : Polynomial ℂ, 
    (Polynomial.derivative^[1992] (p / (X^3 - X))) = f / g ∧ f.degree = 3984 := 
sorry

end smallest_degree_of_f_l585_585863


namespace sin_double_angle_value_l585_585293

theorem sin_double_angle_value
  (α : ℝ)
  (h1 : α ∈ set.Ioo (π / 2) π)
  (h2 : 3 * Real.cos (2 * α) = Real.cos (π / 4 + α)) :
  Real.sin (2 * α) = -17 / 18 :=
sorry

end sin_double_angle_value_l585_585293


namespace problem1_problem2_l585_585605

theorem problem1 (α : ℝ) (h1 : Real.sin (α + π / 4) = sqrt 2 / 10) (h2 : α ∈ Ioo (π / 2) π) :
    Real.cos α = -3 / 5 :=
sorry

theorem problem2 (α : ℝ) (h1 : Real.sin (α + π / 4) = sqrt 2 / 10) (h2 : α ∈ Ioo (π / 2) π) :
    Real.sin (2 * α - π / 4) = -17 * sqrt 2 / 50 :=
sorry

end problem1_problem2_l585_585605


namespace max_largest_integer_of_five_l585_585350

theorem max_largest_integer_of_five (a b c d e : ℕ) (h1 : (a + b + c + d + e) = 500)
    (h2 : e > c ∧ c > d ∧ d > b ∧ b > a)
    (h3 : (a + b + d + e) / 4 = 105)
    (h4 : b + e = 150) : d ≤ 269 := 
sorry

end max_largest_integer_of_five_l585_585350


namespace tobias_time_spent_at_pool_l585_585039

-- Define the conditions
def distance_per_interval : ℕ := 100
def time_per_interval : ℕ := 5
def pause_interval : ℕ := 25
def pause_time : ℕ := 5
def total_distance : ℕ := 3000
def total_time_in_hours : ℕ := 3

-- Hypotheses based on the problem conditions
def swimming_time_without_pauses := (total_distance / distance_per_interval) * time_per_interval
def number_of_pauses := (swimming_time_without_pauses / pause_interval)
def total_pause_time := number_of_pauses * pause_time
def total_time := swimming_time_without_pauses + total_pause_time

-- Proof statement
theorem tobias_time_spent_at_pool : total_time / 60 = total_time_in_hours :=
by 
  -- Put proof here
  sorry

end tobias_time_spent_at_pool_l585_585039


namespace jack_loss_l585_585345

theorem jack_loss :
  let monthly_expense := 3 * 20
  let annual_expense := monthly_expense * 12
  let sell_price := 500
  annual_expense - sell_price = 220 :=
by {
  let monthly_expense := 3 * 20
  let annual_expense := monthly_expense * 12
  let sell_price := 500
  calc
    annual_expense - sell_price = (3 * 20 * 12) - 500 : by rfl
  ... = 720 - 500 : by rfl
  ... = 220 : by rfl
}

end jack_loss_l585_585345


namespace ellipse_and_circle_intersection_l585_585245

/-- Given an ellipse E with the equation x^2/a^2 + y^2/b^2 = 1 (a > b > 0) 
    and eccentricity sqrt(2)/2, and a circle C with the equation 
    (x-2)^2+(y-1)^2=20/3. If E and C intersect at points A and B, and the 
    line segment AB is exactly the diameter of circle C, then
    (1) The equation of the line AB is x + y - 3 = 0.
    (2) The standard equation of the ellipse E is x^2/16 + y^2/8 = 1. -/
theorem ellipse_and_circle_intersection (a b c : ℝ) (h1 : a > b) (h2 : b > 0)
  (h3 : c = a * (sqrt (2) / 2)) (h4 : (x y : ℝ) -> E = (x^2 / a^2) + (y^2 / b^2) = 1)
  (h5 : (x y : ℝ) -> C = (x - 2)^2 + (y - 1)^2 = 20 / 3)
  (h6 : (A B : ℝ × ℝ) -> E.intersect C A B)
  (h7 : (A B : ℝ × ℝ) -> ((fst A + fst B = 4) ∧ (snd A + snd B = 2)))
  (h8 : (A B : ℝ × ℝ) -> AB.is_diameter C A B) :
  ∃ k, (k = x + y - 3 = 0) ∧ ∀ x y, E = (x^2 / 16) + (y^2 / 8) = 1 :=
by
  sorry

end ellipse_and_circle_intersection_l585_585245


namespace polycarp_kolka_number_l585_585415

-- defining the largest five-digit number composed of distinct odd digits
def largest_distinct_odd_five_digit_number : ℕ :=
  97531

-- defining the largest five-digit number using the highest odd digit repeatedly
def largest_repeating_odd_five_digit_number : ℕ :=
  99999

-- Stating the proof problem
theorem polycarp_kolka_number :
  ∃ (polycarp_number kolka_number : ℕ),
    polycarp_number = largest_distinct_odd_five_digit_number ∧
    kolka_number = largest_repeating_odd_five_digit_number :=
by {
  use (largest_distinct_odd_five_digit_number, largest_repeating_odd_five_digit_number),
  split,
  { refl },   -- we prove that Polycarp's number is 97531
  { refl }    -- we prove that Kolka's number is 99999
}

end polycarp_kolka_number_l585_585415


namespace find_principal_sum_l585_585899

def SI (P R T : ℝ) : ℝ := (P * R * T) / 100

theorem find_principal_sum :
  ∃ P : ℝ, SI P 3 5 = 4016.25 ∧ P = 26775 :=
by
  use 26775
  split
  . simp [SI]
  . rfl
  . sorry

end find_principal_sum_l585_585899


namespace domain_of_f_x_minus_1_l585_585428

noncomputable def domain_f_x_squared : Set ℝ := Set.Ioc (-3:ℝ) (1:ℝ)

noncomputable def domain_f : Set ℝ := Set.Ico (0:ℝ) (9:ℝ)

-- Proof statement
theorem domain_of_f_x_minus_1 :
  ∀ f : ℝ → ℝ, 
    (∀ x : ℝ, x^2 ∈ domain_f_x_squared → f x ∈ domain_f) →
    (∀ x : ℝ, x-1 ∈ domain_f → f x ∈ domain_f) →
    ∀ (x: ℝ), (x-1) ∈ domain_f → x ∈ Set.Ico (1:ℝ) (10:ℝ) :=
begin
  intros f h₁ h₂ x hx,
  let domain_f_x_minus_1 : Set ℝ := Set.Ico (1:ℝ) (10:ℝ),
  sorry -- Proof omitted
end

end domain_of_f_x_minus_1_l585_585428


namespace area_of_inscribed_hexagon_l585_585832

theorem area_of_inscribed_hexagon (r : ℝ) (h : π * r^2 = 100 * π) : 
  ∃ (A : ℝ), A = 150 * real.sqrt 3 :=
by 
  -- Definitions of necessary geometric entities and properties would be here
  -- Proof would be provided here
  sorry

end area_of_inscribed_hexagon_l585_585832


namespace solution_set_quadratic_inequality_l585_585444

def quadratic_inequality_solution (x : ℝ) : Prop := x^2 + x - 2 > 0

theorem solution_set_quadratic_inequality :
  {x : ℝ | quadratic_inequality_solution x} = {x : ℝ | x < -2 ∨ x > 1} :=
by
  sorry

end solution_set_quadratic_inequality_l585_585444


namespace smallest_four_digit_multiple_of_18_l585_585973

theorem smallest_four_digit_multiple_of_18 : ∃ n : ℕ, 1000 ≤ n ∧ n ≤ 9999 ∧ n % 18 = 0 ∧ 
  ∀ m : ℕ, 1000 ≤ m ∧ m ≤ 9999 ∧ m % 18 = 0 → n ≤ m :=
begin
  use 1008,
  split,
  { exact nat.le_refl 1008, },
  split,
  { exact nat.le_succ 9999, },
  split,
  { exact nat.mod_eq_zero_of_divisible 1008 18 sorry, },
  { intros m h1 h2 h3,
    apply nat.le_of_sub_nonneg,
    sorry, }
end

end smallest_four_digit_multiple_of_18_l585_585973


namespace percent_of_x_l585_585503

theorem percent_of_x (x : ℝ) (h : x > 0) : (x / 50 + x / 25 - x / 10 + x / 5) = (16 / 100) * x := by
  sorry

end percent_of_x_l585_585503


namespace cost_price_computer_table_l585_585020

theorem cost_price_computer_table (C S : ℝ) (hS1 : S = 1.25 * C) (hS2 : S = 1000) : C = 800 :=
by
  sorry

end cost_price_computer_table_l585_585020


namespace find_q_l585_585867

theorem find_q (q : ℤ) (x : ℤ) (y : ℤ) (h1 : x = 55 + 2 * q) (h2 : y = 4 * q + 41) (h3 : x = y) : q = 7 :=
by
  sorry

end find_q_l585_585867


namespace sum_possible_m_values_l585_585133

theorem sum_possible_m_values : 
  let m_values := {m : ℕ | 4 < m ∧ m < 18}
  let sum_m := ∑ m in m_values, m
  sum_m = 143 :=
by
  sorry

end sum_possible_m_values_l585_585133


namespace coeff_x4_binom_expansion_l585_585425

theorem coeff_x4_binom_expansion : 
  (∃ (c: ℤ), (∀ (x: ℤ), 
  let term := (x ^ 2 - 1 / x) ^ 5 in
  c = coeff term 4) → 
  c = 10) :=
begin
  sorry,
end
end coeff_x4_binom_expansion_l585_585425


namespace aubriella_poured_18_gallons_l585_585190

theorem aubriella_poured_18_gallons 
  (total_capacity : ℕ := 50)
  (pour_time_minutes : ℕ := 6)
  (remaining_gallons : ℕ := 32)
  (seconds_per_gallon : ℕ := 20)
  (minutes_per_hour : ℕ := 60) : 
  (total_capacity - remaining_gallons) = 18 := 
begin
  -- Definitions from conditions
  let pouring_rate := (minutes_per_hour / seconds_per_gallon),
  let poured_gallons := (pour_time_minutes * pouring_rate),
  have h : (total_capacity - remaining_gallons) = poured_gallons,
    from calc
      total_capacity - remaining_gallons = 50 - 32 : by rfl
      ... = 18 : by norm_num,
  show (total_capacity - remaining_gallons) = 18, from h
end

end aubriella_poured_18_gallons_l585_585190


namespace isosceles_triangle_leg_length_l585_585615

theorem isosceles_triangle_leg_length
  (P : ℝ) (base : ℝ) (L : ℝ)
  (h_isosceles : true)
  (h_perimeter : P = 24)
  (h_base : base = 10)
  (h_perimeter_formula : P = base + 2 * L) :
  L = 7 := 
by
  sorry

end isosceles_triangle_leg_length_l585_585615


namespace d1_mul_d2_divisible_by_5_count_a_values_divisible_by_5_l585_585729

theorem d1_mul_d2_divisible_by_5 (a : ℕ) (h1 : 1 ≤ a) (h2 : a ≤ 251) :
  let d1 := a^2 + 2^a + a * 2^((a + 1) / 2)
  let d2 := a^2 + 2^a - a * 2^((a + 1) / 2) in
  (d1 * d2) % 5 = 0 ↔ d1 * d2 = 5 * (d1 * d2 / 5) := sorry

theorem count_a_values_divisible_by_5 : 
  ∃ (n : ℕ), n = 101 ∧
  ∀ (a : ℕ), (1 ≤ a) ∧ (a ≤ 251) ∧ ((a^2 + 2^a + a * 2^((a + 1) / 2)) * (a^2 + 2^a - a * 2^((a + 1) / 2))) % 5 = 0 ↔
    (a ∉ {2, 5}): sorry

end d1_mul_d2_divisible_by_5_count_a_values_divisible_by_5_l585_585729


namespace total_unique_items_l585_585152

-- Define the conditions
def shared_albums : ℕ := 12
def total_andrew_albums : ℕ := 23
def exclusive_andrew_memorabilia : ℕ := 5
def exclusive_john_albums : ℕ := 8

-- Define the number of unique items in Andrew's and John's collection 
def unique_andrew_albums : ℕ := total_andrew_albums - shared_albums
def unique_total_items : ℕ := unique_andrew_albums + exclusive_john_albums + exclusive_andrew_memorabilia

-- The proof goal
theorem total_unique_items : unique_total_items = 24 := by
  -- Proof steps would go here
  sorry

end total_unique_items_l585_585152


namespace fraction_classification_l585_585855

theorem fraction_classification (x y : ℤ) :
  (∃ a b : ℤ, a/b = x/(x+1)) ∧ ¬(∃ a b : ℤ, a/b = x/2 + 1) ∧ ¬(∃ a b : ℤ, a/b = x/2) ∧ ¬(∃ a b : ℤ, a/b = xy/3) :=
by sorry

end fraction_classification_l585_585855


namespace distinct_midpoints_at_least_2n_minus_3_l585_585607

open Set

theorem distinct_midpoints_at_least_2n_minus_3 
  (n : ℕ) 
  (points : Finset (ℝ × ℝ)) 
  (h_points_card : points.card = n) :
  ∃ (midpoints : Finset (ℝ × ℝ)), 
    midpoints.card ≥ 2 * n - 3 := 
sorry

end distinct_midpoints_at_least_2n_minus_3_l585_585607


namespace divide_square_into_smaller_squares_l585_585770

def P (n : Nat) : Prop :=
  ∃ f : ℕ → ℕ, (∀ m, m < n → f m ≠ 0) ∧ (∀ s, s ∈ finset.range n → s = n)

theorem divide_square_into_smaller_squares (n : Nat) (h : n > 5) : P n := sorry

end divide_square_into_smaller_squares_l585_585770


namespace shoppers_in_store_l585_585512

variable (S : ℕ)
variable (a : ℕ)
variable (b : ℕ)

-- Given conditions
def prefers_express (total_shoppers : ℕ) : Prop := 
  total_shoppers * 5 / 8 = a

def pays_at_checkout (total_shoppers : ℕ) : Prop := 
  total_shoppers * 3 / 8 = 180

-- The theorem to be proved
theorem shoppers_in_store : prefers_express S → pays_at_checkout S → S = 480 :=
by
  sorry

end shoppers_in_store_l585_585512


namespace product_of_y_values_l585_585931

theorem product_of_y_values : 
  (∀ y : ℝ, |5 * y| + 7 = 47 → y = 8 ∨ y = -8) →
  (8 * -8 = -64) := 
by
  intros h _,
  have hy8 : 8 * -8 = -64 := by norm_num,
  exact hy8,
  sorry

end product_of_y_values_l585_585931


namespace downstream_distance_l585_585446

-- Define the speeds and distances as constants or variables
def speed_boat := 30 -- speed in kmph
def speed_stream := 10 -- speed in kmph
def distance_upstream := 40 -- distance in km
def time_upstream := distance_upstream / (speed_boat - speed_stream) -- time in hours

-- Define the variable for the downstream distance
variable {D : ℝ}

-- The Lean 4 statement to prove that the downstream distance is the specified value
theorem downstream_distance : 
  (time_upstream = D / (speed_boat + speed_stream)) → D = 80 :=
by
  sorry

end downstream_distance_l585_585446


namespace count_three_digit_integers_with_7_without_4_6_l585_585666

theorem count_three_digit_integers_with_7_without_4_6 : 
  let is_valid_digit (d : ℕ) := d ≠ 4 ∧ d ≠ 6,
      is_valid_three_digit (n : ℕ) := 100 ≤ n ∧ n ≤ 999 ∧ 
                                      (n / 100 ≠ 4 ∧ n / 100 ≠ 6) ∧ 
                                      (n / 10 % 10 ≠ 4 ∧ n / 10 % 10 ≠ 6) ∧ 
                                      (n % 10 ≠ 4 ∧ n % 10 ≠ 6) ∧
                                      ((n / 100 = 7) ∨ (n / 10 % 10 = 7) ∨ (n % 10 = 7)),
      count_valid_numbers := ∑ n in Finset.range 900, if is_valid_three_digit (n + 100) then 1 else 0
  in count_valid_numbers = 154 :=
by
  sorry

end count_three_digit_integers_with_7_without_4_6_l585_585666


namespace calculate_expression_l585_585484

theorem calculate_expression : (50 - (5020 - 520) + (5020 - (520 - 50))) = 100 := 
by
  sorry

end calculate_expression_l585_585484


namespace train_length_given_speed_time_l585_585144

open Real

def kmph_to_mps (speed_kmph : ℝ) : ℝ :=
  speed_kmph * (1000 / 3600)

def train_length (speed_kmph : ℝ) (time_sec : ℝ) : ℝ :=
  kmph_to_mps(speed_kmph) * time_sec

theorem train_length_given_speed_time :
  (train_length 68 9) ≈ 170 :=
by
  -- Skipping the proof steps by adding sorry
  sorry

end train_length_given_speed_time_l585_585144


namespace smallest_four_digit_multiple_of_18_l585_585956

theorem smallest_four_digit_multiple_of_18 : ∃ n: ℕ, (1000 ≤ n ∧ n < 10000 ∧ 18 ∣ n) ∧ ∀ m: ℕ, (1000 ≤ m ∧ m < n ∧ 18 ∣ m) → false :=
begin
  use 1008,
  split,
  { split,
    { -- prove 1000 ≤ 1008
      linarith,
    },
    { split,
      { -- prove 1008 < 10000
        linarith,
      },
      { -- prove 18 ∣ 1008
        norm_num,
      }
    }
  },
  { -- prove there is no four-digit multiple of 18 less than 1008
    intros m,
    intro h,
    cases h with h1 h2,
    cases h2 with h3 h4,
    linarith,
  }
end

end smallest_four_digit_multiple_of_18_l585_585956


namespace principal_amount_l585_585596

theorem principal_amount (A R T : ℝ) (A_eq : A = 2120) (R_eq : R = 0.05) (T_eq : T = 2.4) : 
  let P := 1892.86 in A = P * (1 + R * T) := 
by {
  sorry
}

end principal_amount_l585_585596


namespace total_employees_l585_585559

theorem total_employees (female_employees managers male_associates female_managers : ℕ)
  (h_female_employees : female_employees = 90)
  (h_managers : managers = 40)
  (h_male_associates : male_associates = 160)
  (h_female_managers : female_managers = 40) :
  female_employees - female_managers + male_associates + managers = 250 :=
by {
  sorry
}

end total_employees_l585_585559


namespace shift_down_two_units_l585_585437

theorem shift_down_two_units (x : ℝ) : 
  (y = 2 * x) → (y - 2 = 2 * x - 2) := by
sorry

end shift_down_two_units_l585_585437


namespace f_2011_l585_585267

noncomputable def f (x : ℝ) : ℝ := sin x + exp x + x ^ 2010
def f_seq (n : ℕ) (x : ℝ) : ℝ := Nat.recOn n 
  (f x)
  (λ n fn_x, (deriv fn_x))

theorem f_2011 (x : ℝ) : f_seq 2011 x = - cos x + exp x :=
sorry

end f_2011_l585_585267


namespace cyclic_quadrilateral_bxmy_l585_585759

-- Definitions and conditions given in the problem
variables (A B C : Type) [incidence_geometry A]
variables (triangle_ABC : triangle A B C)
variables (M P Q X Y : A)
variables (circumcircle_ABQ : circle A B Q)
variables (circumcircle_BCP : circle B C P)

-- M is the midpoint of AC
def is_midpoint (M A C : A) [metric_space A] :=
  dist A M = dist M C

-- P and Q are on AM and CM respectively such that PQ = AC / 2
def on_segment (P A M : A) : Prop := distance P A + distance P M = distance A M
def on_segment (Q C M : A) : Prop := distance Q C + distance Q M = distance C M
def pq_half_ac (P Q A C : A) [metric_space A] :=
  dist P Q = dist A C / 2

-- Circumcircle of ABQ intersects BC at X, circumcircle of BCP intersects AB at Y
def intersects_bc (X : A) : Prop := X ∈ circumcircle_ABQ ∧ (X ≠ B ∧ X ≠ C)
def intersects_ab (Y : A) : Prop := Y ∈ circumcircle_BCP ∧ (Y ≠ A ∧ Y ≠ B)

-- Prove that quadrilateral BXMY is cyclic
theorem cyclic_quadrilateral_bxmy 
  (hM: is_midpoint M A C)
  (hP: on_segment P A M) 
  (hQ: on_segment Q C M) 
  (hPQ: pq_half_ac P Q A C)
  (hX: intersects_bc X)
  (hY: intersects_ab Y):
  cyclic_quad B X M Y :=
by sorry

end cyclic_quadrilateral_bxmy_l585_585759


namespace gcd_m_n_l585_585372

noncomputable def m := 55555555
noncomputable def n := 111111111

theorem gcd_m_n : Int.gcd m n = 1 := by
  sorry

end gcd_m_n_l585_585372


namespace trader_profit_l585_585538

noncomputable def original_price (P : ℝ) : ℝ := P
noncomputable def discount_price (P : ℝ) : ℝ := 0.95 * P
noncomputable def selling_price (P : ℝ) : ℝ := 1.52 * P
noncomputable def profit (P : ℝ) : ℝ := selling_price P - original_price P
noncomputable def percent_profit (P : ℝ) : ℝ := (profit P / original_price P) * 100

theorem trader_profit (P : ℝ) (hP : 0 < P) : percent_profit P = 52 := by 
  sorry

end trader_profit_l585_585538


namespace integral_sin_pi_neg_pi_zero_l585_585508

theorem integral_sin_pi_neg_pi_zero : ∫ x in -real.pi..real.pi, sin x = 0 :=
by
  sorry

end integral_sin_pi_neg_pi_zero_l585_585508


namespace max_edges_Kk_free_graph_l585_585674

noncomputable def binom : ℕ → ℕ → ℕ
| n 0 := 1
| 0 k := 0
| n (k+1) := binom (n-1) k + binom (n-1) (k+1)

theorem max_edges_Kk_free_graph (n k : ℕ) (r : ℕ) 
  (hr : r ≡ n % (k-1))
  (hr_bound : 0 ≤ r ∧ r ≤ k-2) :
  let M := (k-2) / (k-1) * ((n^2 - r^2) / 2) + binom r 2 in
  ∀ G : SimpleGraph (Fin n), ¬G.Clique k → G.edgeCount ≤ M := 
sorry

end max_edges_Kk_free_graph_l585_585674


namespace divide_square_into_smaller_squares_l585_585772

-- Definition of the property P(n)
def P (n : ℕ) : Prop := ∃ (f : ℕ → ℕ), ∀ i, i < n → (f i > 0)

-- Proposition for the problem
theorem divide_square_into_smaller_squares (n : ℕ) (h : n > 5) : P n :=
sorry

end divide_square_into_smaller_squares_l585_585772


namespace radius_range_l585_585262

theorem radius_range
  (r : ℝ)
  (h_circle : ∀ (x y : ℝ), (x - 1)^2 + (y - 2)^2 = r^2 → (4 * x + 3 * y - 35 = 0 → (4 < r ∧ r < 6))) :
  ∀ (x y : ℝ), (x - 1)^2 + (y - 2)^2 = r^2 → dist (x, y) (4, 3, -35) = 1 → 4 < r ∧ r < 6 := by
  sorry

end radius_range_l585_585262


namespace minimum_total_distance_walked_l585_585215

theorem minimum_total_distance_walked (n : ℕ) (d : ℝ) (h_n : n = 20) (h_d : d = 10) : 
  let P = (n / 2 : ℝ) * d in
  let distance_walked (k : ℕ) := k * d in
  let total_distance_walked :=
    2 * (∑ i in (finset.range (n / 2)), distance_walked (i + 1)) + 2 * distance_walked (n / 2 - 1) in
  total_distance_walked = 1280 :=
by {
  sorry
}

end minimum_total_distance_walked_l585_585215


namespace totalMountainNumbers_l585_585196

-- Define a 4-digit mountain number based on the given conditions.
def isMountainNumber (n : ℕ) : Prop :=
  ∃ (a b c d : ℕ), n = a * 1000 + b * 100 + c * 10 + d ∧
    1 ≤ a ∧ a ≤ 9 ∧
    1 ≤ b ∧ b ≤ 9 ∧
    1 ≤ c ∧ c ≤ 9 ∧
    1 ≤ d ∧ d ≤ 9 ∧
    b > a ∧ b > d ∧ c > a ∧ c > d ∧
    a ≠ d

-- Define the main theorem stating that the total number of 4-digit mountain numbers is 1512.
theorem totalMountainNumbers : 
  ∃ n, (∀ m, isMountainNumber m → ∃ l, l = 1 ∧ 4 ≤ m ∧ m ≤ 9999) ∧ n = 1512 := sorry

end totalMountainNumbers_l585_585196


namespace total_equals_70_l585_585681

-- Definitions based on the conditions
variables (M D Total : ℕ)
def mother_age : M = 40 := sorry
def condition1 : M + 2 * D = Total := sorry
def condition2 : D + 2 * M = 95 := sorry

-- The main statement to be proved
theorem total_equals_70 (M = 40) (M + 2 * D = Total) (D + 2 * M = 95) : Total = 70 := 
by sorry

end total_equals_70_l585_585681


namespace recycling_team_points_l585_585500

theorem recycling_team_points (z_paper z_plastic z_aluminum : ℕ)
    (f1_paper f1_plastic f1_aluminum : ℕ)
    (f2_paper f2_plastic f2_aluminum : ℕ)
    (f3_paper f3_plastic f3_aluminum : ℕ)
    (f4_paper f4_plastic f4_aluminum : ℕ) :
  z_paper = 35 → z_plastic = 15 → z_aluminum = 5 →
  f1_paper = 28 → f1_plastic = 18 → f1_aluminum = 8 →
  f2_paper = 22 → f2_plastic = 10 → f2_aluminum = 6 →
  f3_paper = 40 → f3_plastic = 20 → f3_aluminum = 10 →
  f4_paper = 18 → f4_plastic = 12 → f4_aluminum = 8 →
  (z_paper / 12 + z_plastic / 6 + z_aluminum / 4 +
  f1_paper / 12 + f1_plastic / 6 + f1_aluminum / 4 +
  f2_paper / 12 + f2_plastic / 6 + f2_aluminum / 4 +
  f3_paper / 12 + f3_plastic / 6 + f3_aluminum / 4 +
  f4_paper / 12 + f4_plastic / 6 + f4_aluminum / 4) = 28 :=
by
  intros z_paper_cond z_plastic_cond z_aluminum_cond
         f1_paper_cond f1_plastic_cond f1_aluminum_cond
         f2_paper_cond f2_plastic_cond f2_aluminum_cond
         f3_paper_cond f3_plastic_cond f3_aluminum_cond
         f4_paper_cond f4_plastic_cond f4_aluminum_cond
  rw [z_paper_cond, z_plastic_cond, z_aluminum_cond,
      f1_paper_cond, f1_plastic_cond, f1_aluminum_cond,
      f2_paper_cond, f2_plastic_cond, f2_aluminum_cond,
      f3_paper_cond, f3_plastic_cond, f3_aluminum_cond,
      f4_paper_cond, f4_plastic_cond, f4_aluminum_cond]
  simp
  norm_num
  sorry

end recycling_team_points_l585_585500


namespace months_after_withdrawal_and_advance_eq_eight_l585_585871

-- Define initial conditions
def initial_investment_A : ℝ := 3000
def initial_investment_B : ℝ := 4000
def withdrawal_A : ℝ := 1000
def advancement_B : ℝ := 1000
def total_profit : ℝ := 630
def share_A : ℝ := 240
def share_B : ℝ := total_profit - share_A

-- Define the main proof problem
theorem months_after_withdrawal_and_advance_eq_eight
  (initial_investment_A : ℝ) (initial_investment_B : ℝ)
  (withdrawal_A : ℝ) (advancement_B : ℝ)
  (total_profit : ℝ) (share_A : ℝ) (share_B : ℝ) : 
  ∃ x : ℝ, 
  (3000 * x + 2000 * (12 - x)) / (4000 * x + 5000 * (12 - x)) = 240 / 390 ∧
  x = 8 :=
sorry

end months_after_withdrawal_and_advance_eq_eight_l585_585871


namespace remaining_coal_burn_days_l585_585785

theorem remaining_coal_burn_days (burned_fraction : ℚ) (days : ℕ) : 
  burned_fraction = 2 / 9 → days = 6 →
  let burned_per_day := burned_fraction / days in
  let remaining_coal := 1 - burned_fraction in
  let remaining_days := remaining_coal / burned_per_day in
  remaining_days = 21 :=
by
  intros h_fraction h_days
  let burned_per_day := (2 / 9) / 6
  let remaining_coal := 1 - (2 / 9)
  let remaining_days := remaining_coal / burned_per_day
  have : remaining_days = 21, sorry
  exact this

end remaining_coal_burn_days_l585_585785


namespace tangent_line_at_e_l585_585219

noncomputable def f (x : ℝ) : ℝ := x * Real.log x

theorem tangent_line_at_e : 
  (∀ x : ℝ, y = f x) → (∀ x = e, y = 2 * x - e) := 
by
  sorry

end tangent_line_at_e_l585_585219


namespace smallest_four_digit_multiple_of_18_l585_585975

theorem smallest_four_digit_multiple_of_18 : 
  ∃ (n : ℕ), 999 < n ∧ n < 10000 ∧ 18 ∣ n ∧ ∀ m : ℕ, 999 < m ∧ m < 10000 ∧ 18 ∣ m → n ≤ m ∧ n = 1008 := 
sorry

end smallest_four_digit_multiple_of_18_l585_585975


namespace smallest_four_digit_multiple_of_18_l585_585989

theorem smallest_four_digit_multiple_of_18 (n : ℕ) (h1 : 1000 ≤ n) (h2 : n < 10000) (h3 : n % 18 = 0) : n = 1008 :=
by
  have h4 : n ≥ 1008, sorry
  have h5 : n ≤ 1008, sorry
  exact eq_of_le_of_ge h4 h5

end smallest_four_digit_multiple_of_18_l585_585989


namespace candies_eaten_l585_585174

theorem candies_eaten (A B D : ℕ) 
                      (h1 : 4 * B = 3 * A) 
                      (h2 : 7 * A = 6 * D) 
                      (h3 : A + B + D = 70) :
  A = 24 ∧ B = 18 ∧ D = 28 := 
by
  sorry

end candies_eaten_l585_585174


namespace train_length_l585_585900

def length_of_train (speed_km_hr : ℕ) (time_s : ℕ) (platform_length_m : ℕ) : ℕ :=
  let speed_m_s := (speed_km_hr * 1000) / 3600
  let total_distance := speed_m_s * time_s
  total_distance - platform_length_m

theorem train_length (speed_km_hr : ℕ) (time_s : ℕ) (platform_length_m : ℕ) (L : ℕ) 
  (h1 : speed_km_hr = 45)
  (h2 : time_s = 48)
  (h3 : platform_length_m = 240) :
  L = 360 :=
by
  have h_speed_m_s : speed_m_s = (45 * 1000) / 3600 := by
    rw [h1]
    norm_num
  have h_total_distance : total_distance = 12 * 48 := by
    rw [h_speed_m_s, h2]
    norm_num
  have h4 : total_distance = L + platform_length_m := by
    rw [h3, h_total_distance, h2]
    norm_num
  rw [h4, sub_eq_iff_eq_add.mpr (eq.symm h3)]
  norm_num
  sorry

end train_length_l585_585900


namespace smallest_angle_in_15_sided_polygon_arithmetic_sequence_l585_585004

theorem smallest_angle_in_15_sided_polygon_arithmetic_sequence
  (a d : ℕ) 
  (angles : Fin 15 → ℕ)
  (h_seq : ∀ i : Fin 15, angles i = a + i * d)
  (h_convex : ∀ i : Fin 15, angles i < 180)
  (h_sum : ∑ i, angles i = 2340) : 
  a = 135 := 
sorry

end smallest_angle_in_15_sided_polygon_arithmetic_sequence_l585_585004


namespace tangent_line_equation_at_M_minus_pi_l585_585430

noncomputable def f (x : ℝ) : ℝ := 
  if x = 0 then 1 else (sin x) / x

def tangent_line_slope (x0 : ℝ) : ℝ := 
  if x0 = 0 then 0 else (x0 * cos x0 - sin x0) / (x0 ^ 2)

theorem tangent_line_equation_at_M_minus_pi :
  let M := (-π, 0) in
  let slope := tangent_line_slope (-π) in
  slope = 1 / π →
  ∀ (x y : ℝ), y = f x → y = (1 / π) * (x + π) → x - π * y + π = 0
:= by
  intro M slope h_slope x y fx hx
  rw [hx, h_slope]
  sorry

end tangent_line_equation_at_M_minus_pi_l585_585430


namespace minimum_magnitude_of_diff_l585_585663

noncomputable def a (t : ℝ) : ℝ × ℝ × ℝ := (1 - t, 2 * t - 1, 0)
noncomputable def b (t : ℝ) : ℝ × ℝ × ℝ := (2, t, t)
noncomputable def diff (t : ℝ) : ℝ × ℝ × ℝ := (1 + t, 1 - t, t)

noncomputable def dot_product (u v : ℝ × ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2 + u.3 * v.3

noncomputable def magnitude_squared (u : ℝ × ℝ × ℝ) : ℝ :=
  dot_product u u

theorem minimum_magnitude_of_diff :
  (∃ t : ℝ, magnitude_squared (diff t) = 2) ∧ ∀ t : ℝ, magnitude_squared (diff t) ≥ 2 :=
begin
  split,
  { use 0, 
    simp [diff, magnitude_squared, dot_product],
    norm_num },
  { intro t,
    simp [diff, magnitude_squared, dot_product, add_mul, mul_add, mul_comm],
    linarith }
end

end minimum_magnitude_of_diff_l585_585663


namespace one_inch_cubes_with_red_paint_at_least_two_faces_l585_585028

theorem one_inch_cubes_with_red_paint_at_least_two_faces
  (number_of_one_inch_cubes : ℕ)
  (cubes_with_three_faces : ℕ)
  (cubes_with_two_faces : ℕ)
  (total_cubes_with_at_least_two_faces : ℕ) :
  number_of_one_inch_cubes = 64 →
  cubes_with_three_faces = 8 →
  cubes_with_two_faces = 24 →
  total_cubes_with_at_least_two_faces = cubes_with_three_faces + cubes_with_two_faces →
  total_cubes_with_at_least_two_faces = 32 :=
by
  sorry

end one_inch_cubes_with_red_paint_at_least_two_faces_l585_585028


namespace candy_eating_l585_585182

-- Definitions based on the conditions
def candies (andrey_rate boris_rate denis_rate : ℕ) (andrey_candies boris_candies denis_candies total_candies : ℕ) : Prop :=
  andrey_rate = 4 ∧ boris_rate = 3 ∧ denis_rate = 7 ∧ andrey_candies = 24 ∧ boris_candies = 18 ∧ denis_candies = 28 ∧
  total_candies = andrey_candies + boris_candies + denis_candies

-- Problem statement
theorem candy_eating :
  ∃ (a b d : ℕ), 
    candies 4 3 7 a b d 70 :=
sorry

end candy_eating_l585_585182


namespace transfer_students_increase_average_l585_585468

theorem transfer_students_increase_average
    (avgA : ℚ) (numA : ℕ) (avgB : ℚ) (numB : ℕ)
    (kalinina_grade : ℚ) (sidorova_grade : ℚ)
    (avgA_init : avgA = 44.2) (numA_init : numA = 10)
    (avgB_init : avgB = 38.8) (numB_init : numB = 10)
    (kalinina_init : kalinina_grade = 41)
    (sidorova_init : sidorova_grade = 44) : 
    let new_avg_B_k := (avgB * numB + kalinina_grade) / (numB + 1) in
    let new_avg_A_s := (avgA * numA - sidorova_grade) / (numA - 1) in
    let new_avg_A_both := (avgA * numA - kalinina_grade - sidorova_grade) / (numA - 2) in
    let new_avg_B_both := (avgB * numB + kalinina_grade + sidorova_grade) / (numB + 2) in
    (new_avg_B_k <= avgB) ∧ (new_avg_A_s <= avgA) ∧ (new_avg_A_both > avgA) ∧ (new_avg_B_both > avgB) :=
by
  sorry

end transfer_students_increase_average_l585_585468


namespace heptagon_vertex_coloring_l585_585200

theorem heptagon_vertex_coloring :
  let vertices := {1, 2, 3, 4, 5, 6, 7}
      colors := {red, green, blue}
      valid_coloring (c : 1 → {red, green, blue}) : Prop :=
        ∀ (i j k : 1), i ≠ j ∧ j ≠ k ∧ k ≠ i -> ¬ ((i, j, k) ∈ {{x | x.1 = x.2 ∧ x.2 = x.3}}))
  in ∃ n : ℕ, n = 294 ∧ 
  (∃ (f : 1 → colors), valid_coloring f) = n := 
sorry

end heptagon_vertex_coloring_l585_585200


namespace time_to_cover_remaining_distance_l585_585351

theorem time_to_cover_remaining_distance :
  let rate := 45 / 15 in  -- Kit's movement rate in feet per minute
  let remaining_distance_yards := 50 in  -- Remaining distance in yards
  let conversion_factor := 3 in  -- Feet per yard
  let remaining_distance_feet := remaining_distance_yards * conversion_factor in
  let time := remaining_distance_feet / rate in
  time = 50 := -- Time in minutes
by
  -- We skip the proof as per the instruction
  sorry

end time_to_cover_remaining_distance_l585_585351


namespace sum_of_possible_m_values_l585_585125

theorem sum_of_possible_m_values : 
  sum (setOf m in {m | 4 < m ∧ m < 18}) = 132 :=
by
  sorry

end sum_of_possible_m_values_l585_585125


namespace domain_f_f_positive_l585_585383

noncomputable def f (b a x : ℝ) : ℝ :=
  log b ((x^2 - 2*x + 2) / (1 + 2*a*x))

theorem domain_f (b a : ℝ) (hb1 : b > 0) (hb2 : b ≠ 1) :
  (a > 0 → ∀ x, x > -1/(2*a) → f(b, a, x)) ∧
  (a = 0 → ∀ x, true) ∧
  (a < 0 → ∀ x, x < -1/(2*a) → f(b, a, x)) :=
sorry

theorem f_positive (b a : ℝ) (hb1 : b > 1) :
  (a < -2 → ∀ x, (x < (1 + a - sqrt(a^2 + 2*a)) ∨
                   ((1 + a + sqrt(a^2 + 2*a)) < x ∧ x < -1/(2*a))) → f(b, a, x) > 0) ∧
  (a = -2 → ∀ x, (x < 1/4 ∧ x ≠ -1) → f(b, a, x) > 0) ∧
  (-2 < a ∧ a < 0 → ∀ x, x < -1/(2*a) → f(b, a, x) > 0) ∧
  (a = 0 → ∀ x, x ≠ 1 → f(b, a, x) > 0) ∧
  (a > 0 → ∀ x, ((-1/(2*a) < x ∧ x < 1 + a - sqrt(a^2 + 2*a)) ∨
                 (x > 1 + a + sqrt(a^2 + 2*a))) → f(b, a, x) > 0) :=
sorry

end domain_f_f_positive_l585_585383


namespace temperature_difference_l585_585794

theorem temperature_difference 
  (lowest: ℤ) (highest: ℤ) 
  (h_lowest : lowest = -4)
  (h_highest : highest = 5) :
  highest - lowest = 9 := 
by
  --relies on the correctness of problem and given simplyifying
  sorry

end temperature_difference_l585_585794


namespace sum_of_possible_lengths_l585_585119

theorem sum_of_possible_lengths
  (m : ℕ) 
  (h1 : m < 18)
  (h2 : m > 4) : ∑ i in (Finset.range 13).map (λ x, x + 5) = 143 := by
sorry

end sum_of_possible_lengths_l585_585119


namespace part1_monotonic_increasing_part2_inequality_l585_585078

-- Define the function f
def f (x : ℝ) : ℝ := 2 * sin^2(π / 4 + x) - sqrt 3 * cos (2 * x) - 1

-- Part 1: Prove that f(x) is monotonically increasing in [π/4, 5π/12]
theorem part1_monotonic_increasing (x : ℝ) (hx : π / 4 ≤ x ∧ x ≤ 5 * π / 12) :
  ∀ x1 x2, (π / 4 ≤ x1 ∧ x1 ≤ x2 ∧ x2 ≤ 5 * π / 12) → f x1 ≤ f x2 := 
sorry

-- Part 2: Prove that the inequality |f(x) - m| < 2 holds for 0 < m < 3
theorem part2_inequality (m : ℝ) (hm : 0 < m ∧ m < 3) (x : ℝ) (hx : π / 4 ≤ x ∧ x ≤ π / 2) :
  |f x - m| < 2 :=
sorry

end part1_monotonic_increasing_part2_inequality_l585_585078


namespace cars_needed_to_double_earnings_l585_585097

theorem cars_needed_to_double_earnings
  (base_salary : ℕ)
  (commission_per_car : ℕ)
  (january_earnings : ℕ)
  (february_target : ℕ)
  (january_cars_sold : ℕ) :
  base_salary = 1000 →
  commission_per_car = 200 →
  january_earnings = 1800 →
  february_target = 2 * january_earnings →
  january_earnings = base_salary + commission_per_car * january_cars_sold →
  february_target - base_salary = 200 * 13 :=
begin
  intros,
  sorry
end

end cars_needed_to_double_earnings_l585_585097


namespace sum_triangle_inequality_solutions_l585_585139

theorem sum_triangle_inequality_solutions :
  (∑ m in (Finset.Icc 5 17), m) = 143 :=
by
  sorry

end sum_triangle_inequality_solutions_l585_585139


namespace diagonal_intersection_unit_squares_l585_585092

theorem diagonal_intersection_unit_squares (m n : ℕ) (h0 : m = 20) (h1 : n = 19) :
  let rect := (0, 0, 20, 19) in
  let gcd_mn := Nat.gcd 20 19 in
  let vert_crossings := m - 1 in
  let horiz_crossings := n - 1 in
  let initial_square := 1 in
  let total_single_diagonal :(vert_crossings + horiz_crossings + initial_square) := 38
  let total_with_double_count := 2 * total_single_diagonal in
  let overlap_points := 2 in
  total_with_double_count - overlap_points = 74 :=
by
  sorry

end diagonal_intersection_unit_squares_l585_585092


namespace cubic_sequence_problem_l585_585212

def cubic_sequence (b c d : ℤ) (n : ℤ) : ℤ :=
  n^3 + b * n^2 + c * n + d

theorem cubic_sequence_problem 
  (b c d : ℤ) 
  (a : list ℤ)
  (h1 : a = list.map (cubic_sequence b c d) [2015, 2016])
  (square_2015 : ∃ m:ℤ, a.head = m^2)
  (square_2016 : ∃ n:ℤ, a.tail.head = n^2)
  : a.head * a.tail.head = 0 := 
sorry

end cubic_sequence_problem_l585_585212


namespace density_is_not_vector_l585_585546

/-- Conditions definition -/
def is_vector (quantity : String) : Prop :=
quantity = "Buoyancy" ∨ quantity = "Wind speed" ∨ quantity = "Displacement"

/-- Problem statement -/
theorem density_is_not_vector : ¬ is_vector "Density" := 
by 
sorry

end density_is_not_vector_l585_585546


namespace Janet_fewer_siblings_l585_585715

def Masud_siblings : ℕ := 40
def Janet_siblings : ℕ := 4 * Masud_siblings - 60
def Carlos_siblings : ℕ := 64 -- Corrected based on problem inconsistency
def Stella_siblings : ℕ := 2 * Carlos_siblings - 8
def Combined_siblings : ℕ := Carlos_siblings + Stella_siblings

theorem Janet_fewer_siblings : Janet_siblings < Combined_siblings :=
by {
  have h1 : Janet_siblings = 100, by 
    simp [Janet_siblings, Masud_siblings],
  have h2 : Combined_siblings = 116, by 
    simp [Combined_siblings, Carlos_siblings, Stella_siblings],
  rw [h1, h2],
  norm_num,
}

end Janet_fewer_siblings_l585_585715


namespace zookeeper_feeding_problem_l585_585147

noncomputable def feeding_ways : ℕ :=
  sorry

theorem zookeeper_feeding_problem :
  feeding_ways = 2880 := 
sorry

end zookeeper_feeding_problem_l585_585147


namespace cylinder_new_volume_l585_585440

noncomputable def original_volume : ℝ := 15
noncomputable def new_volume (r h : ℝ) : ℝ := 
  let V := π * r^2 * h
  let new_r := 3 * r
  let new_h := h / 2
  π * new_r^2 * new_h

theorem cylinder_new_volume (r h : ℝ) (hV : π * r^2 * h = original_volume) : 
  new_volume r h = 67.5 :=
by
  sorry

end cylinder_new_volume_l585_585440


namespace maximize_GDP_investment_l585_585476

def invest_A_B_max_GDP : Prop :=
  ∃ (A B : ℝ), 
  A + B ≤ 30 ∧
  20000 * A + 40000 * B ≤ 1000000 ∧
  24 * A + 32 * B ≥ 800 ∧
  A = 20 ∧ B = 10

theorem maximize_GDP_investment : invest_A_B_max_GDP :=
by
  sorry

end maximize_GDP_investment_l585_585476


namespace no_irrational_root_of_cubic_in_quadratic_l585_585247

theorem no_irrational_root_of_cubic_in_quadratic
  (P : Polynomial ℤ)
  (hP_deg : P.degree = 3)
  (hP_roots : ∀ x, x ∈ P.roots → irrational x)
  (Q : Polynomial ℤ)
  (hQ_deg : Q.degree = 2) :
  ∀ x, x ∈ P.roots → x ∉ Q.roots := by
  sorry

end no_irrational_root_of_cubic_in_quadratic_l585_585247


namespace arithmetic_seq_20th_term_l585_585207

theorem arithmetic_seq_20th_term {a1 d : ℕ} (h_a1 : a1 = 2) (h_d : d = 3) : 
  let a (n : ℕ) := a1 + (n - 1) * d in a 20 = 59 :=
by
  sorry

end arithmetic_seq_20th_term_l585_585207


namespace sum_triangle_inequality_solutions_l585_585137

theorem sum_triangle_inequality_solutions :
  (∑ m in (Finset.Icc 5 17), m) = 143 :=
by
  sorry

end sum_triangle_inequality_solutions_l585_585137


namespace regular_hexagon_area_inscribed_in_circle_l585_585836

theorem regular_hexagon_area_inscribed_in_circle
  (h : Real.pi * r^2 = 100 * Real.pi) :
  6 * (r^2 * Real.sqrt 3 / 4) = 150 * Real.sqrt 3 :=
by {
  sorry
}

end regular_hexagon_area_inscribed_in_circle_l585_585836


namespace total_pencils_bought_l585_585861

theorem total_pencils_bought (x y : ℕ) (y_pos : 0 < y) (initial_cost : y * (x + 10) = 5 * x) (later_cost : (4 * y) * (x + 10) = 20 * x) :
    x = 15 → (40 = x + x + 10) ∨ x = 40 → (90 = x + (x + 10)) :=
by
  sorry

end total_pencils_bought_l585_585861


namespace total_problems_math_contest_l585_585389

variable {Neznayka DoctorPilyulkin Knopochka Vintik Znayka : ℕ}
variable (n : ℕ)

def conditions :=
  Znayka = 10 ∧
  Neznayka = 6 ∧
  DoctorPilyulkin ∈ {7, 8, 9} ∧
  Knopochka ∈ {7, 8, 9} ∧
  Vintik ∈ {7, 8, 9} ∧
  Neznayka + DoctorPilyulkin + Knopochka + Vintik + Znayka = 4 * n

theorem total_problems_math_contest (n : ℕ) :
  conditions →
  n = 10 :=
by
  intros h
  sorry

end total_problems_math_contest_l585_585389


namespace trigonometric_expression_evaluation_l585_585778

noncomputable def c := (2 * Real.pi) / 13

theorem trigonometric_expression_evaluation :
  (sin (4 * c) * sin (8 * c) * sin (12 * c) * sin (16 * c) * sin (20 * c)) /
  (sin c * sin (2 * c) * sin (3 * c) * sin (4 * c) * sin (5 * c))
  =
  1 / (sin c) :=
by
  sorry

end trigonometric_expression_evaluation_l585_585778


namespace largest_subset_count_l585_585053

theorem largest_subset_count :
  ∃ (A : Set ℕ), A ⊆ Finset.range 3000 ∧ 
  (∀ a b ∈ A, a ≠ b → |a - b| ≠ 1 ∧ |a - b| ≠ 4 ∧ |a - b| ≠ 5) ∧ 
  A.card = 1000 := sorry

end largest_subset_count_l585_585053


namespace contrapositive_inequality_l585_585297

theorem contrapositive_inequality {x y : ℝ} (h : x^2 ≤ y^2) : x ≤ y :=
  sorry

end contrapositive_inequality_l585_585297


namespace find_a_l585_585617

noncomputable theory

open Real

def line1 (x y : ℝ) : Prop := 2 * x - y + 1 = 0
def line2 (a x y : ℝ) : Prop := 4 * x - 2 * y + a = 0
def circle (x y : ℝ) : Prop := x^2 + y^2 - 2 * x = 0
def distance (x₁ y₁ a b c : ℝ) : ℝ := abs (a * x₁ + b * y₁ + c) / sqrt (a^2 + b^2)

theorem find_a (a : ℝ) :
  (∀ (x y : ℝ), circle x y → (distance 1 0 2 (-1) 1 + distance 1 0 4 (-2) a = 2 * sqrt 5)) →
  a = 10 ∨ a = -18 :=
sorry

end find_a_l585_585617


namespace lcm_even_numbers_between_14_and_21_l585_585948

-- Define the even numbers between 14 and 21
def evenNumbers := [14, 16, 18, 20]

-- Define a function to compute the LCM of a list of integers
def lcm_list (l : List ℕ) : ℕ :=
  l.foldr Nat.lcm 1

-- Theorem statement: the LCM of the even numbers between 14 and 21 equals 5040
theorem lcm_even_numbers_between_14_and_21 :
  lcm_list evenNumbers = 5040 :=
by
  sorry

end lcm_even_numbers_between_14_and_21_l585_585948


namespace functions_with_inverses_l585_585856

-- Definitions of each function as per conditions.
def a (x : ℝ) : ℝ := x^2
def b (x : ℝ) : ℝ := x^3 - 3 * x
def c (x : ℝ) : ℝ := Real.sin x + Real.cos x
def d (x : ℝ) : ℝ := Real.exp x - 2
def e (x : ℝ) : ℝ := |x| - 3
def f (x : ℝ) : ℝ := Real.log x + Real.log (2 * x)
def g (x : ℝ) : ℝ := Real.tan x - x
def h (x : ℝ) : ℝ := x / 3

-- Defining the domains
def domain_a : Set ℝ := { x | 0 ≤ x ∧ x ≤ 2 }
def domain_b : Set ℝ := Set.univ
def domain_c : Set ℝ := { x | 0 ≤ x ∧ x ≤ Real.pi }
def domain_d : Set ℝ := { x | 0 ≤ x ∧ x ≤ 1 }
def domain_e : Set ℝ := { x | -5 ≤ x ∧ x ≤ 5 }
def domain_f : Set ℝ := { x | 0 < x }
def domain_g : Set ℝ := { x | - (Real.pi / 2) < x ∧ x < Real.pi / 2 }
def domain_h : Set ℝ := { x | -9 ≤ x ∧ x ≤ 21 }

-- Statement of the problem
theorem functions_with_inverses :
  (∃ (a_inv : ℝ → ℝ), ∀ x ∈ domain_a, a (a_inv x) = x ∧ a_inv (a x) = x) ∧
  (¬ ∃ (b_inv : ℝ → ℝ), ∀ x ∈ domain_b, b (b_inv x) = x ∧ b_inv (b x) = x) ∧
  (∃ (c_inv : ℝ → ℝ), ∀ x ∈ domain_c, c (c_inv x) = x ∧ c_inv (c x) = x) ∧
  (∃ (d_inv : ℝ → ℝ), ∀ x ∈ domain_d, d (d_inv x) = x ∧ d_inv (d x) = x) ∧
  (¬ ∃ (e_inv : ℝ → ℝ), ∀ x ∈ domain_e, e (e_inv x) = x ∧ e_inv (e x) = x) ∧
  (∃ (f_inv : ℝ → ℝ), ∀ x ∈ domain_f, f (f_inv x) = x ∧ f_inv (f x) = x) ∧
  (¬ ∃ (g_inv : ℝ → ℝ), ∀ x ∈ domain_g, g (g_inv x) = x ∧ g_inv (g x) = x) ∧
  (∃ (h_inv : ℝ → ℝ), ∀ x ∈ domain_h, h (h_inv x) = x ∧ h_inv (h x) = x) :=
sorry

end functions_with_inverses_l585_585856


namespace perimeter_relationship_l585_585185

theorem perimeter_relationship (n : ℕ) (AC l p : ℝ) 
  (m : ℕ → ℝ) 
  (h1 : (finset.range n).sum m = AC)
  (p_k : ℕ → ℝ)
  (hk : ∀ k, m k = (real.sqrt 2) * (p_k k / 4))
  (p_sum : (finset.range n).sum p_k = p)
  (a : ℝ) :
  l = 4 * a → 
  AC = a * (real.sqrt 2) →
  p = l :=
by
  intros h2 h3
  sorry

end perimeter_relationship_l585_585185


namespace cannot_determine_plane_l585_585433

-- Definitions of the conditions
def condition1 (P1 P2 P3 : Point) : Prop := ¬ collinear P1 P2 P3 
def condition2 (L : Line) (P : Point) : Prop := P ∉ L
def condition3 (a : Line) (L1 L2 : Line) : Prop := intersect L1 a ∧ intersect L2 a
def condition4 (L1 L2 L3 : Line) : Prop := pairwise_intersect L1 L2 L3

-- The theorem stating that none of these conditions can uniquely determine a plane.
theorem cannot_determine_plane (P1 P2 P3 : Point) (L : Line) (P : Point) (a L1 L2 L3 : Line) :
  ¬ condition1 P1 P2 P3 ∧ ¬ condition2 L P ∧ ¬ condition3 a L1 L2 ∧ ¬ condition4 L1 L2 L3 := 
sorry

end cannot_determine_plane_l585_585433


namespace range_of_function_is_correct_l585_585597

def range_of_quadratic_function : Set ℝ :=
  {y | ∃ x : ℝ, y = -x^2 - 6 * x - 5}

theorem range_of_function_is_correct :
  range_of_quadratic_function = {y | y ≤ 4} :=
by
  -- sorry allows skipping the actual proof step
  sorry

end range_of_function_is_correct_l585_585597


namespace find_BP_CN_l585_585394

/- Definitions of the initial conditions as given in the problem -/
def AK_over_BK := 3 / 2
def BM_over_MC := 3
def AC := a : Real

/- Define points K, M, P, and N such that ratios and intersections hold as given in the problem -/

noncomputable def BP (a : Real) : Real := 6 * a / 11
noncomputable def CN (a : Real) : Real := 2 * a / 11

theorem find_BP_CN (AC a : Real) (h₁ : AK_over_BK = 3 / 2) (h₂ : BM_over_MC = 3) (h₃ : AC = a) : 
  BP a = 6 * a / 11 ∧ CN a = 2 * a / 11 := by
  sorry

end find_BP_CN_l585_585394


namespace trapezoid_base_ratio_l585_585550

-- Define the problem as a Lean statement
theorem trapezoid_base_ratio
  (a b : ℝ) 
  (h₀ : a > b) 
  (area_trapezoid : ℝ) 
  (area_quadrilateral : ℝ)
  (h₁ : area_quadrilateral = (Real.cbrt 3) / 3 * area_trapezoid) :
  a / b = 3 := 
sorry

end trapezoid_base_ratio_l585_585550


namespace num_intersection_points_tan_sin_l585_585334

theorem num_intersection_points_tan_sin : 
    (set.count (set_of (λ x : ℝ, tan x = sin x)) (-2 * Real.pi).le (2 * Real.pi).le = 5 := 
by
  sorry

end num_intersection_points_tan_sin_l585_585334


namespace constant_term_in_expansion_l585_585485

theorem constant_term_in_expansion : 
  ∃ (c : ℕ), c = 26730 ∧ 
  (∀ (k : ℕ), (k ≤ 11 → 
  (∃ (m : ℤ), 
  (m = (11 - k) * -1 + k * (1/2) ∧ m = 0))) implies c = (nat.choose 11 4 * 3^4)) :=
begin
  sorry
end

end constant_term_in_expansion_l585_585485


namespace sin_square_sum_le_one_l585_585312

theorem sin_square_sum_le_one (α : Type*) [plane α] (θ₁ θ₂ : ℝ) (h₁ : θ₁ + θ₂ ≤ π / 2) :
  sin(θ₁)^2 + sin(θ₂)^2 ≤ 1 :=
sorry

end sin_square_sum_le_one_l585_585312


namespace same_terminal_side_angle_l585_585829

theorem same_terminal_side_angle (k : ℤ) :
  let α := (Real.pi / 12) + 2 * k * Real.pi
  in α % (2 * Real.pi) = (375 * Real.pi / 180) % (2 * Real.pi) :=
by
  intros
  -- We assume α is given as (Real.pi / 12) + 2 * k * Real.pi
  let α := (Real.pi / 12) + 2 * k * Real.pi
  -- We need to prove α mod (2 * Real.pi) = (375 * Real.pi / 180) mod (2 * Real.pi)
  have h : α % (2 * Real.pi) = (375 * Real.pi / 180) % (2 * Real.pi) := sorry
  exact h

end same_terminal_side_angle_l585_585829


namespace min_f_on_interval_range_a_l585_585648

noncomputable def f (x : ℝ) : ℝ := x * Real.log x
noncomputable def g (x a : ℝ) : ℝ := -x ^ 2 + a * x - 3

theorem min_f_on_interval (t : ℝ) (ht : 0 < t) :
  let min_val := if t < 1 / Real.exp 1 then -1 / Real.exp 1 else t * Real.log t in
  min f (t + 2) = min_val := sorry

theorem range_a (a : ℝ) :
  (∃ x0 ∈ Icc (1 / Real.exp 1) (Real.exp 1), 2 * f x0 ≥ g x0 a) →
  a ≤ -2 + 1 / Real.exp 1 + 3 * Real.exp 1 := sorry

end min_f_on_interval_range_a_l585_585648


namespace smallest_four_digit_multiple_of_18_l585_585971

theorem smallest_four_digit_multiple_of_18 : ∃ n : ℕ, 1000 ≤ n ∧ n ≤ 9999 ∧ n % 18 = 0 ∧ 
  ∀ m : ℕ, 1000 ≤ m ∧ m ≤ 9999 ∧ m % 18 = 0 → n ≤ m :=
begin
  use 1008,
  split,
  { exact nat.le_refl 1008, },
  split,
  { exact nat.le_succ 9999, },
  split,
  { exact nat.mod_eq_zero_of_divisible 1008 18 sorry, },
  { intros m h1 h2 h3,
    apply nat.le_of_sub_nonneg,
    sorry, }
end

end smallest_four_digit_multiple_of_18_l585_585971


namespace distinguishable_large_triangles_l585_585581

/-- Conditions for the proof problem -/
def large_triangle_configuration (colors : Finset (Fin 8)) (corners center_base center_overlay : Fin 8) : Prop :=
  (colors.card = 8) ∧
  (corners ∈ colors) ∧
  (center_base ∈ colors) ∧
  (center_overlay ∈ colors) ∧
  (center_base ≠ center_overlay)

/-- Proof problem: The total number of distinguishably different large triangles -/
theorem distinguishable_large_triangles (colors : Finset (Fin 8)) :
  (∃ corners center_base center_overlay, 
    large_triangle_configuration colors corners center_base center_overlay ∧
    card(colors ∪ {corners, center_base, center_overlay}) = 6720) :=
sorry

end distinguishable_large_triangles_l585_585581


namespace candies_eaten_l585_585172

theorem candies_eaten (A B D : ℕ) 
                      (h1 : 4 * B = 3 * A) 
                      (h2 : 7 * A = 6 * D) 
                      (h3 : A + B + D = 70) :
  A = 24 ∧ B = 18 ∧ D = 28 := 
by
  sorry

end candies_eaten_l585_585172


namespace market_value_of_share_l585_585069

-- Definitions from the conditions
def nominal_value : ℝ := 48
def dividend_rate : ℝ := 0.09
def desired_interest_rate : ℝ := 0.12

-- The proof problem (theorem statement) in Lean 4
theorem market_value_of_share : (nominal_value * dividend_rate / desired_interest_rate * 100) = 36 := 
by
  sorry

end market_value_of_share_l585_585069


namespace plane_not_unique_l585_585307

variables {a b : Line}

-- Assume lines a and b are skew lines.
axiom skew_lines (a b : Line) : ¬ (∃ p, (p ∈ a) ∧ (p ∈ b)) ∧ ¬(parallel a b)

-- Definition of a plane passing through line a and perpendicular to line b.
def plane_passing_through_and_perpendicular (a b : Line) : Prop :=
  ∃ (plane : Plane), (∀ p ∈ a, p ∈ plane) ∧ (∀ q ∈ b, plane.perpendicular q)

theorem plane_not_unique (a b : Line) (h : skew_lines a b) : ¬ (∃! (plane : Plane), 
  (∀ p ∈ a, p ∈ plane) ∧ (∀ q ∈ b, plane.perpendicular q)) := sorry

end plane_not_unique_l585_585307


namespace find_b_l585_585022

theorem find_b (b : ℝ) :
  let v1 := ![-6, b]
  let v2 := ![3, 2]
  let projection := (v1⬝v2) / (v2⬝v2) • v2
  projection = (-17/13 : ℝ) • v2 → b = 1/2 :=
by
  sorry

end find_b_l585_585022


namespace regular_hexagon_area_l585_585839

theorem regular_hexagon_area (A : ℝ) (r : ℝ) (hex_area : ℝ) :
  A = 100 * Real.pi → r = Real.sqrt 100 → 
  hex_area = 150 * Real.sqrt 3 → 
  150 * Real.sqrt 3 = 150 * Real.sqrt 3 :=
by
  sorry

end regular_hexagon_area_l585_585839


namespace max_min_func1_max_min_func2_l585_585221

noncomputable def func1 (x : ℝ) := x^4 - 8 * x^2 + 3
noncomputable def func2 (x : ℝ) := tan x - x

theorem max_min_func1 :
  (∀ x ∈ set.Icc (-2:ℝ) 2, func1 x ≤ 3) ∧ 
  (∃ x ∈ set.Icc (-2:ℝ) 2, func1 x = 3) ∧
  (∀ x ∈ set.Icc (-2:ℝ) 2, func1 x ≥ -13) ∧ 
  (∃ x ∈ set.Icc (-2:ℝ) 2, func1 x = -13) := 
sorry

theorem max_min_func2 :
  (∀ x ∈ set.Icc (-π/4) (π/4), func2 x ≤ (1 - π/4)) ∧ 
  (∃ x ∈ set.Icc (-π/4) (π/4), func2 x = (1 - π/4)) ∧
  (∀ x ∈ set.Icc (-π/4) (π/4), func2 x ≥ (π/4 - 1)) ∧ 
  (∃ x ∈ set.Icc (-π/4) (π/4), func2 x = (π/4 - 1)) := 
sorry

end max_min_func1_max_min_func2_l585_585221


namespace find_x_l585_585866

theorem find_x (h : 0.60 / x = 6 / 2) : x = 0.20 :=
by
  sorry

end find_x_l585_585866


namespace coefficient_without_x_l585_585426

theorem coefficient_without_x (y : ℝ) :
  let f (x : ℝ) := (2 * x + y / x) ^ 6 in
  (∃ c : ℝ, ∀ x : ℝ, 6 - 2 * 3 = 0 → c = 2 ^ 3 * fintype.card (finset (range (6.choose 3)))) :=
by
  sorry

end coefficient_without_x_l585_585426


namespace equilateral_triangle_segments_sum_l585_585918

-- We define the problem conditions
def is_equilateral_triangle (ABC : Triangle) : Prop :=
  ABC.side_length AB = 2011 ∧ ABC.side_length BC = 2011 ∧ ABC.side_length CA = 2011
  
def parallel_segments (P : Point) (ABC : Triangle) (DE FG HI : Segment) : Prop :=
  DE ∥ BC ∧ FG ∥ CA ∧ HI ∥ AB

def segments_ratio (DE FG HI : Segment) : Prop :=
  DE.length / FG.length = 8/7 ∧ FG.length / HI.length = 7/10

-- The theorem statement
theorem equilateral_triangle_segments_sum (ABC : Triangle) (P : Point)
  (DE FG HI : Segment) :
  is_equilateral_triangle ABC →
  parallel_segments P ABC DE FG HI →
  segments_ratio DE FG HI →
  DE.length + FG.length + HI.length = 4022 :=
by
  sorry

end equilateral_triangle_segments_sum_l585_585918


namespace time_to_meet_approx_l585_585583

-- Defining the given conditions
def distance : ℝ := 200
def enrique_speed : ℝ := 16
def jamal_speed : ℝ := 23
def combined_speed := enrique_speed + jamal_speed

-- Time to meet
def time_to_meet := distance / combined_speed

-- The goal is to prove that the time to meet is approximately 5.1282 hours
theorem time_to_meet_approx : abs(time_to_meet - 5.1282) < 0.0001 :=
by
  sorry

end time_to_meet_approx_l585_585583


namespace count_good_numbers_l585_585302

def is_good_number (n : ℕ) : Prop :=
  (1000 ≤ n) ∧ (n < 10000) ∧ (n % 10 = 1) ∧ (
    let digits := [n / 1000 % 10, n / 100 % 10 % 10, n / 10 % 10, n % 10];
    ([1, 2, 3, 4]).any (λ d, digits.count d = 3)
  )

theorem count_good_numbers : 
  {n : ℕ | is_good_number n}.finite.to_finset.card = 12 := 
sorry

end count_good_numbers_l585_585302


namespace num_proper_subsets_set_123_l585_585949

theorem num_proper_subsets_set_123 : 
  ∃ s : set ℕ, s = {1, 2, 3} ∧ (s.subsets \ {s}).card = 7 :=
by
  sorry

end num_proper_subsets_set_123_l585_585949


namespace a2_greater_than_floor_2n_over_3_l585_585204

theorem a2_greater_than_floor_2n_over_3 (n : ℕ) (a : Fin n → ℕ)
  (h1 : ∀ i, i < n → a i ≤ 2 * n)
  (h2 : ∀ i j, i ≠ j → i < n → j < n → Nat.lcm (a i) (a j) > 2 * n) :
  a 1 > Nat.floor (2 * n / 3) := sorry

end a2_greater_than_floor_2n_over_3_l585_585204


namespace optimal_firefighters_l585_585878

noncomputable def forest_fire_loss (n : ℕ) : ℝ :=
  let fire_spread_rate := 100
  let time_to_arrive := 5
  let firefighter_extinguish_rate := 50
  let cost_per_firefighter_minute := 125
  let additional_cost_per_firefighter := 100
  let cost_of_burned_forest := 60
  let initial_fire_area := fire_spread_rate * time_to_arrive  -- 500m^2
  let t := -500 / (100 - 50 * n)
  60 * (initial_fire_area + (100 - firefighter_extinguish_rate * n) * t) + 125 * n * t + 100 * n

theorem optimal_firefighters : (n : ℕ) := 
  argmin forest_fire_loss 27 := sorry

end optimal_firefighters_l585_585878


namespace residues_not_permutation_l585_585722

theorem residues_not_permutation (p : ℕ) (hp : nat.prime p) (h_odd : p % 2 = 1) :
  ∀ (a b : fin p → fin p), 
  (∀ i j, a i ≠ a j → i ≠ j) → 
  (∀ i j, b i ≠ b j → i ≠ j) → 
  (¬ ∀ i, ∃ j, ∃ k, (a j * b k) % p = i) := 
begin
  intro a,
  intro b,
  intros ha hb,
  intro h,
  have hp_fact := nat.modeq_of_mul_modeq_prod (nat.prime.pred_mul p hp) (by norm_num),
  sorry
end

end residues_not_permutation_l585_585722


namespace shaded_region_area_l585_585443

noncomputable theory

open Real

-- Definitions based on the conditions extracted
def side_length := 10 -- cm
def base := side_length
def height := 10 * √3 -- height calculated based on 30-60-90 triangle properties

-- Statement of the proof
theorem shaded_region_area : base * height = 100 * √3 :=
by sorry

end shaded_region_area_l585_585443


namespace find_k_l585_585624

theorem find_k (a b k : ℝ) (h1 : 2^a = k) (h2 : 3^b = k) (h3 : k ≠ 1) (h4 : 1/a + 2/b = 1) : k = 18 := by
  sorry

end find_k_l585_585624


namespace shaded_area_of_circles_problem_l585_585705

theorem shaded_area_of_circles_problem :
  let R := 8
  let r := R / 2
  let large_circle_area := Real.pi * R ^ 2
  let small_circle_area := Real.pi * r ^ 2
  let total_small_circles_area := 2 * small_circle_area
  large_circle_area - total_small_circles_area = 32 * Real.pi :=
by
  -- Definitions based on conditions
  let R := 8
  let r := R / 2
  let large_circle_area := Real.pi * R ^ 2
  let small_circle_area := Real.pi * r ^ 2
  let total_small_circles_area := 2 * small_circle_area
  -- Proof obligation
  show large_circle_area - total_small_circles_area = 32 * Real.pi
  sorry

end shaded_area_of_circles_problem_l585_585705


namespace condition_neither_sufficient_nor_necessary_l585_585332

-- Definitions used in the conditions
def geom_seq (a : ℕ → ℝ) : Prop := ∀ n : ℕ, a (n + 1) = q * a n

-- Statement of the problem
theorem condition_neither_sufficient_nor_necessary
  (a : ℕ → ℝ) (q : ℝ)
  (h_geom : geom_seq a)
  (h_condition : 8 * a 2 - a 5 = 0) :
  ¬(∀ (a : ℕ → ℝ) (q : ℝ), h_geom → (8 * a 2 - a 5 = 0) → (a 1 < a 2 ∧ a 2 < a 3)) ∧
  ¬(∀ (a : ℕ → ℝ) (q : ℝ), h_geom → (a 1 < a 2 ∧ a 2 < a 3) → (8 * a 2 - a 5 = 0)) :=
sorry

end condition_neither_sufficient_nor_necessary_l585_585332


namespace batsman_average_increase_l585_585514

-- Definitions to capture the initial conditions
def runs_scored_in_17th_inning : ℕ := 74
def average_after_17_innings : ℕ := 26

-- Statement to prove the increment in average is 3 runs per inning
theorem batsman_average_increase (A : ℕ) (initial_avg : ℕ)
  (h_initial_runs : 16 * initial_avg + 74 = 17 * 26) :
  26 - initial_avg = 3 :=
by
  sorry

end batsman_average_increase_l585_585514


namespace fraction_of_juices_consumed_l585_585816

def guests := 90
def soda_cans := 50
def plastic_bottles := 50
def glass_bottles := 50
def half_guests_drink_soda := guests / 2
def third_guests_drink_water := guests / 3
def total_recyclable_cans_bottles := 115

theorem fraction_of_juices_consumed : 
  let soda_consumed := half_guests_drink_soda
      sparkling_water_consumed := third_guests_drink_water
      total_soda_sparkling := soda_consumed + sparkling_water_consumed
      juice_consumed := total_recyclable_cans_bottles - total_soda_sparkling
      fraction_juices_consumed := juice_consumed / glass_bottles
  in fraction_juices_consumed = 4 / 5 :=
by sorry

end fraction_of_juices_consumed_l585_585816


namespace extremum_necessary_not_sufficient_l585_585373

def differentiable_on_R (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, ∃ (f' : ℝ), has_deriv_at f f' x

theorem extremum_necessary_not_sufficient (f : ℝ → ℝ) (x : ℝ) :
  differentiable_on_R f →
  (∀ (x : ℝ), (differentiable_on_R f) → (has_deriv_at f 0 x → (∃ t : ℝ, (t ≠ x) ∧ (has_deriv_at f f' t) ∧ ∀ (h : 0 < |t - x|, (has_deriv_at f f' x) ∧ (¬ is_extremum f x)))))
    → (¬ (∀ t, t ≠ x → (has_deriv_at f f' t) ∧ (is_extremum f x)))) :=
begin
  sorry
end

end extremum_necessary_not_sufficient_l585_585373


namespace female_democrats_l585_585507

theorem female_democrats (F M : ℕ) 
    (h₁ : F + M = 990)
    (h₂ : F / 2 + M / 4 = 330) : F / 2 = 275 := 
by sorry

end female_democrats_l585_585507


namespace second_player_wins_l585_585088

-- Definitions and conditions
def card_color := Prop  -- representing the color side up as a proposition
def blue (c: card_color) := c = true
def yellow (c: card_color) := c = false

def initial_setup : List card_color := List.replicate 2009 true

def valid_move (block: List card_color) : Prop :=
  block.length = 50 ∧ blue (List.headI block)

def flip_block (block: List card_color) : List card_color :=
  block.map Bnot

def next_state (state: List card_color) (move: List.card_color → List.card_color) (start: ℕ) : List card_color :=
  let (prefix, suffix) := state.splitAt start
  prefix ++ (move (suffix.take 50)) ++ (suffix.drop 50)

-- Theorem to prove
theorem second_player_wins (initial_state: List card_color) 
  (H_initial: initial_state = initial_setup): 
  ∃ n : ℕ, n > 0 ∧ ∀ m : ℕ, (m < n) → ¬ (∃ move : List card_color → List card_color, 
     ∃ start : ℕ, valid_move (initial_state.drop start.take 50) ∧ 
     (initial_state = next_state initial_state move start)) :=
sorry

end second_player_wins_l585_585088


namespace bug_returns_to_starting_vertex_after_ten_moves_l585_585095

noncomputable def P : ℕ → ℚ
| 0     := 1
| (n+1) := (2 / 3) * P n

theorem bug_returns_to_starting_vertex_after_ten_moves :
  let p := 1024
  let q := 59049 in
  P 10 = p / q ∧ Nat.gcd p q = 1 ∧ p + q = 59573 :=
by
  sorry

end bug_returns_to_starting_vertex_after_ten_moves_l585_585095


namespace candy_eating_l585_585178

-- Definitions based on the conditions
def candies (andrey_rate boris_rate denis_rate : ℕ) (andrey_candies boris_candies denis_candies total_candies : ℕ) : Prop :=
  andrey_rate = 4 ∧ boris_rate = 3 ∧ denis_rate = 7 ∧ andrey_candies = 24 ∧ boris_candies = 18 ∧ denis_candies = 28 ∧
  total_candies = andrey_candies + boris_candies + denis_candies

-- Problem statement
theorem candy_eating :
  ∃ (a b d : ℕ), 
    candies 4 3 7 a b d 70 :=
sorry

end candy_eating_l585_585178


namespace bubble_pass_result_l585_585416

theorem bubble_pass_result :
  let n := 50
  let distinct_reals := list.range 1 (n + 1)  -- let it simulate distinct real numbers since the problem just needs a distinct sequence
  let r := distinct_reals ++ distinct_reals -- just to simulate r_1 to r_50 by doubling range
  let r25 := r.nth_le 24 sorry
  let r_sorted := r.qsort (<=)
  let after_bubble_pass := r_sorted.init.concat (r25, r_sorted.tail)
  let p := 1
  let q := 1640
  p + q = 1641
by
  sorry

end bubble_pass_result_l585_585416


namespace number_of_terms_in_arithmetic_sequence_l585_585941

theorem number_of_terms_in_arithmetic_sequence 
  (a d n l : ℤ) (h1 : a = 7) (h2 : d = 2) (h3 : l = 145) 
  (h4 : l = a + (n - 1) * d) : n = 70 := 
by sorry

end number_of_terms_in_arithmetic_sequence_l585_585941


namespace find_lambda_l585_585287

variable (λ : ℝ)
variables m n : ℝ × ℝ

def m : ℝ × ℝ := (1, 1)
def n : ℝ × ℝ := (λ, 2)

def is_perpendicular (a b : ℝ × ℝ) : Prop := a.1 * b.1 + a.2 * b.2 = 0

theorem find_lambda
  (h_perpendicular : is_perpendicular (m + n) (m - n)) :
  λ = -1 :=
by
  sorry

end find_lambda_l585_585287


namespace housing_price_growth_l585_585683

theorem housing_price_growth (x : ℝ) (h₁ : (5500 : ℝ) > 0) (h₂ : (7000 : ℝ) > 0) :
  5500 * (1 + x) ^ 2 = 7000 := 
sorry

end housing_price_growth_l585_585683


namespace smallest_four_digit_multiple_of_18_l585_585974

theorem smallest_four_digit_multiple_of_18 : 
  ∃ (n : ℕ), 999 < n ∧ n < 10000 ∧ 18 ∣ n ∧ ∀ m : ℕ, 999 < m ∧ m < 10000 ∧ 18 ∣ m → n ≤ m ∧ n = 1008 := 
sorry

end smallest_four_digit_multiple_of_18_l585_585974


namespace regular_hexagon_area_l585_585842

theorem regular_hexagon_area (A : ℝ) (r : ℝ) (hex_area : ℝ) :
  A = 100 * Real.pi → r = Real.sqrt 100 → 
  hex_area = 150 * Real.sqrt 3 → 
  150 * Real.sqrt 3 = 150 * Real.sqrt 3 :=
by
  sorry

end regular_hexagon_area_l585_585842


namespace calc_expression_l585_585565

theorem calc_expression : 2012 * 2016 - 2014^2 = -4 := by
  sorry

end calc_expression_l585_585565


namespace gcd_55555555_111111111_l585_585369

/-- Let \( m = 55555555 \) and \( n = 111111111 \).
We want to prove that the greatest common divisor (gcd) of \( m \) and \( n \) is 1. -/
theorem gcd_55555555_111111111 :
  let m := 55555555
  let n := 111111111
  Nat.gcd m n = 1 :=
by
  sorry

end gcd_55555555_111111111_l585_585369


namespace fraction_area_enclosed_by_nested_hexagon_l585_585360

theorem fraction_area_enclosed_by_nested_hexagon (s : ℝ) :
  let original_hexagon_area := (3 * Real.sqrt 3 / 2) * s^2
  let scaled_hexagon_area := (3 * Real.sqrt 3 / 8) * s^2
  scaled_hexagon_area / original_hexagon_area = 1 / 4 :=
begin
  sorry
end

end fraction_area_enclosed_by_nested_hexagon_l585_585360


namespace choose_two_from_four_l585_585327

theorem choose_two_from_four : nat.choose 4 2 = 6 := by
  sorry

end choose_two_from_four_l585_585327


namespace angle_A_eq_pi_over_3_perimeter_eq_24_l585_585339

namespace TriangleProof

-- We introduce the basic setup for the triangle
variables {A B C : ℝ} {a b c : ℝ}

-- Given condition
axiom condition : 2 * b = 2 * a * Real.cos C + c

-- Part 1: Prove angle A is π/3
theorem angle_A_eq_pi_over_3 (h : 2 * b = 2 * a * Real.cos C + c) :
  A = Real.pi / 3 :=
sorry

-- Part 2: Given a = 10 and the area is 8√3, prove perimeter is 24
theorem perimeter_eq_24 (a_eq_10 : a = 10) (area_eq_8sqrt3 : 8 * Real.sqrt 3 = (1 / 2) * b * c * Real.sin A) :
  a + b + c = 24 :=
sorry

end TriangleProof

end angle_A_eq_pi_over_3_perimeter_eq_24_l585_585339


namespace candies_eaten_l585_585154

variables (A B D : ℕ)

-- Conditions:
def condition1 : Prop := ∃ k1 k2 k3 : ℕ, k1 * 4 + k2 * 3 + k3 * 7 = 70
def condition2 : Prop := (B * 3 = A * 4) ∧ (D * 7 = A * 6)
def condition3 : Prop := A + B + D = 70

-- Theorem statement:
theorem candies_eaten (h1 : condition1) (h2 : condition2) (h3 : condition3) :
    A = 24 ∧ B = 18 ∧ D = 28 := sorry

end candies_eaten_l585_585154


namespace find_n_for_series_sum_l585_585992

theorem find_n_for_series_sum :
  ∃ n : ℕ, (∑ k in finset.range n, 1 / (Real.sqrt k + Real.sqrt (k + 1))) = 2019 ↔ n = 4080399 :=
sorry

end find_n_for_series_sum_l585_585992


namespace increase_average_grades_l585_585466

theorem increase_average_grades (XA XB : ℝ) (nA nB : ℕ) (k s : ℝ) 
    (hXA : XA = 44.2) (hXB : XB = 38.8) 
    (hnA : nA = 10) (hnB : nB = 10) 
    (hk : k = 41) (hs : s = 44) :
    let new_XA := (XA * nA - k - s) / (nA - 2)
    let new_XB := (XB * nB + k + s) / (nB + 2) in
    (new_XA > XA) ∧ (new_XB > XB) := by
  sorry

end increase_average_grades_l585_585466


namespace simplify_tan_expression_l585_585773

theorem simplify_tan_expression :
  (1 + Real.tan (Real.pi / 12)) * (1 + Real.tan (Real.pi / 6)) = 2 :=
by
  have h1 : Real.tan (Real.pi / 4) = 1 := by
    rw Real.tan_pi_div_four
  have h2 : Real.tan (Real.pi / 4) = (Real.tan (Real.pi / 12) + Real.tan (Real.pi / 6)) / 
                                     (1 - Real.tan (Real.pi / 12) * Real.tan (Real.pi / 6)) := by
    rw [Real.tan_add, Real.tan_pi_div_four]
  have h_eq : Real.tan (Real.pi / 12) + Real.tan (Real.pi / 6) = 
              1 - Real.tan (Real.pi / 12) * Real.tan (Real.pi / 6) := by
    rw ← h1, ← h2, ← Real.tan_add
  calc
    (1 + Real.tan (Real.pi / 12)) * 
    (1 + Real.tan (Real.pi / 6)) =
    1 + Real.tan (Real.pi / 12) + Real.tan (Real.pi / 6) + Real.tan (Real.pi / 12) * Real.tan (Real.pi / 6) : by rw add_mul
    ... = 1 + (1 - Real.tan (Real.pi / 12) * Real.tan (Real.pi / 6)) + Real.tan (Real.pi / 12) * Real.tan (Real.pi / 6) : by rw h_eq
    ... = 2 : by ring

end simplify_tan_expression_l585_585773


namespace complement_A_in_U_l585_585284

noncomputable def U : Set ℝ := {x | x > -Real.sqrt 3}
noncomputable def A : Set ℝ := {x | 1 < 4 - x^2 ∧ 4 - x^2 ≤ 2}

theorem complement_A_in_U :
  (U \ A) = {x | -Real.sqrt 3 < x ∧ x ≤ -Real.sqrt 2} ∪ {x | Real.sqrt 2 ≤ x ∧ x < (Real.sqrt 3) ∨ Real.sqrt 3 ≤ x} :=
by
  sorry

end complement_A_in_U_l585_585284


namespace faster_train_passes_slower_train_in_54_seconds_l585_585824

section train_problem

-- Define the speeds of the faster and slower trains in km/hr
def speed_faster_train_kmph : ℝ := 46
def speed_slower_train_kmph : ℝ := 36

-- Define the length of each train in meters
def length_of_train_m : ℝ := 75

-- Convert km/hr to m/s
def kmph_to_mps (speed_kmph : ℝ) : ℝ := speed_kmph * (5 / 18)

-- Calculate the relative speed in m/s
def relative_speed_mps : ℝ := kmph_to_mps (speed_faster_train_kmph - speed_slower_train_kmph)

-- Calculate the total distance to cover for the faster train to pass the slower train
def total_distance_m : ℝ := 2 * length_of_train_m

-- Calculate the time taken to pass the slower train in seconds
def time_to_pass_slower_train_s : ℝ := total_distance_m / relative_speed_mps

theorem faster_train_passes_slower_train_in_54_seconds :
  time_to_pass_slower_train_s = 54 := by
  -- Here would be the proof that time_to_pass_slower_train_s equals 54 seconds
  sorry

end train_problem

end faster_train_passes_slower_train_in_54_seconds_l585_585824


namespace bus_initial_speed_is_60_l585_585872

noncomputable def initial_average_speed (v : ℝ) : Prop :=
  ∃ D : ℝ, D = v * (65 / 60) ∧ D = (v + 5) * 1

theorem bus_initial_speed_is_60 : initial_average_speed 60 :=
by
  use 65 * 60 / 60
  split
  {
    calc
    60 * (65 / 60) = (60 * 65) / 60 : by ring
    ... = 65 : by norm_num
  }
  {
    calc
    ((60 + 5) * 1) = 65 : by norm_num
  }

end bus_initial_speed_is_60_l585_585872


namespace initial_milk_quantity_l585_585817

theorem initial_milk_quantity 
  (milk_left_in_tank : ℕ) -- the remaining milk in the tank
  (pumping_rate : ℕ) -- the rate at which milk was pumped out
  (pumping_hours : ℕ) -- hours during which milk was pumped out
  (adding_rate : ℕ) -- the rate at which milk was added
  (adding_hours : ℕ) -- hours during which milk was added 
  (initial_milk : ℕ) -- initial milk collected
  (h1 : milk_left_in_tank = 28980) -- condition 3
  (h2 : pumping_rate = 2880) -- condition 1 (rate)
  (h3 : pumping_hours = 4) -- condition 1 (hours)
  (h4 : adding_rate = 1500) -- condition 2 (rate)
  (h5 : adding_hours = 7) -- condition 2 (hours)
  : initial_milk = 30000 :=
by
  sorry

end initial_milk_quantity_l585_585817


namespace harriett_found_nickels_l585_585852

theorem harriett_found_nickels (quarters dimes nickels pennies : ℕ) (total_value : ℝ) 
  (h1 : quarters = 10) (h2 : dimes = 3) (h3 : pennies = 5) (h4 : total_value = 3) :
  nickels = 3 :=
by 
  have h_quarters_value : ℝ := quarters * 0.25,
  have h_dimes_value : ℝ := dimes * 0.10,
  have h_pennies_value : ℝ := pennies * 0.01,
  let sum_quarters_dimes_pennies := h_quarters_value + h_dimes_value + h_pennies_value,
  have missing_value_nickels := total_value - sum_quarters_dimes_pennies,
  have nickels_value := missing_value_nickels / 0.05,
  have nickels_correct := nat.of_real nickels_value,
  -- Prove that 'nickels' is indeed equal to 3
  sorry

end harriett_found_nickels_l585_585852


namespace find_a_value_l585_585012

noncomputable def parabola_vertex_form (a x : ℝ) : ℝ :=
  a * (x - 3)^2 + 2

theorem find_a_value : 
  (∃ a : ℝ, parabola_vertex_form a (-2) = -43) → 
  ∃ a : ℝ, a = -1.8 :=
by 
  intro h
  obtain ⟨a, h1⟩ := h
  have h2 : parabola_vertex_form a (-2) = a * 25 + 2 := by
    simp [parabola_vertex_form]
    ring
  rw h1 at h2
  sorry

end find_a_value_l585_585012


namespace triangle_is_either_isosceles_or_right_angled_l585_585682

theorem triangle_is_either_isosceles_or_right_angled
  (A B : Real)
  (a b c : Real)
  (h : (a^2 + b^2) * Real.sin (A - B) = (a^2 - b^2) * Real.sin (A + B))
  : a = b ∨ a^2 + b^2 = c^2 :=
sorry

end triangle_is_either_isosceles_or_right_angled_l585_585682


namespace fraction_product_sum_l585_585488

theorem fraction_product_sum :
  (1/3) * (5/6) * (3/7) + (1/4) * (1/8) = 101/672 :=
by
  sorry

end fraction_product_sum_l585_585488


namespace saree_final_sale_price_in_inr_l585_585535

noncomputable def finalSalePrice (initialPrice: ℝ) (discounts: List ℝ) (conversionRate: ℝ) : ℝ :=
  let finalUSDPrice := discounts.foldl (fun acc discount => acc * (1 - discount)) initialPrice
  finalUSDPrice * conversionRate

theorem saree_final_sale_price_in_inr
  (initialPrice : ℝ := 150)
  (discounts : List ℝ := [0.20, 0.15, 0.05])
  (conversionRate : ℝ := 75)
  : finalSalePrice initialPrice discounts conversionRate = 7267.5 :=
by
  sorry

end saree_final_sale_price_in_inr_l585_585535


namespace brownies_in_pan_l585_585518

-- Given conditions
def slices_cake : Nat := 8
def calories_per_slice_cake : Nat := 347
def calories_per_brownie : Nat := 375
def extra_calories : Int := 526

-- Formulate the goal
theorem brownies_in_pan : 
  let total_cake_calories := slices_cake * calories_per_slice_cake
  let total_brownie_calories := total_cake_calories - extra_calories
  total_brownie_calories / calories_per_brownie = 6 :=
by
  let total_cake_calories := slices_cake * calories_per_slice_cake
  let total_brownie_calories := total_cake_calories - extra_calories
  have h : total_brownie_calories / calories_per_brownie = 6
  sorry

end brownies_in_pan_l585_585518


namespace particular_solution_satisfies_l585_585595

noncomputable def particular_solution (x : ℝ) : ℝ :=
  (1/3) * Real.exp (-4 * x) - (1/3) * Real.exp (2 * x) + (x ^ 2 + 3 * x) * Real.exp (2 * x)

def initial_conditions (f df : ℝ → ℝ) : Prop :=
  f 0 = 0 ∧ df 0 = 1

def differential_equation (f df ddf : ℝ → ℝ) : Prop :=
  ∀ x, ddf x + 2 * df x - 8 * f x = (12 * x + 20) * Real.exp (2 * x)

theorem particular_solution_satisfies :
  ∃ C1 C2 : ℝ, initial_conditions (λ x => C1 * Real.exp (-4 * x) + C2 * Real.exp (2 * x) + particular_solution x) 
              (λ x => -4 * C1 * Real.exp (-4 * x) + 2 * C2 * Real.exp (2 * x) + (2 * x^2 + 8 * x + 3) * Real.exp (2 * x)) ∧ 
              differential_equation (λ x => C1 * Real.exp (-4 * x) + C2 * Real.exp (2 * x) + particular_solution x) 
                                  (λ x => -4 * C1 * Real.exp (-4 * x) + 2 * C2 * Real.exp (2 * x) + (2 * x^2 + 8 * x + 3) * Real.exp (2 * x)) 
                                  (λ x => 16 * C1 * Real.exp (-4 * x) + 4 * C2 * Real.exp (2 * x) + (4 * x^2 + 12 * x + 1) * Real.exp (2 * x)) :=
sorry

end particular_solution_satisfies_l585_585595


namespace increase_avg_grade_transfer_l585_585475

-- Definitions for initial conditions
def avg_grade_A_initial := 44.2
def avg_grade_B_initial := 38.8
def num_students_A_initial := 10
def num_students_B_initial := 10

def grade_kalinina := 41
def grade_sidorov := 44

-- Definitions for expected conditions after transfer
def sum_grades_A_initial := avg_grade_A_initial * num_students_A_initial
def sum_grades_B_initial := avg_grade_B_initial * num_students_B_initial

-- Verify the transfer condition will meet the requirements
theorem increase_avg_grade_transfer : 
  let sum_grades_A_after := sum_grades_A_initial - grade_kalinina - grade_sidorov in
  let sum_grades_B_after := sum_grades_B_initial + grade_kalinina + grade_sidorov in
  let num_students_A_after := num_students_A_initial - 2 in
  let num_students_B_after := num_students_B_initial + 2 in
  let avg_grade_A_after := sum_grades_A_after / num_students_A_after in
  let avg_grade_B_after := sum_grades_B_after / num_students_B_after in

  avg_grade_A_after > avg_grade_A_initial ∧ avg_grade_B_after > avg_grade_B_initial :=
by
  sorry

end increase_avg_grade_transfer_l585_585475


namespace solve_sqrt_equation_l585_585209

theorem solve_sqrt_equation :
  (∃ x : ℝ, (real.sqrt (x^2 + 4 * x) = 9) → (x = -2 + real.sqrt 85)) :=
begin
  sorry
end

end solve_sqrt_equation_l585_585209


namespace find_angle_C_find_area_l585_585322

open Real

def is_acute_triangle (A B C a b c : ℝ) : Prop :=
  0 < A ∧ A < π / 2 ∧ 0 < B ∧ B < π / 2 ∧ 0 < C ∧ C < π / 2 ∧ A + B + C = π

theorem find_angle_C (A B C a b c : ℝ) 
    (h_acute : is_acute_triangle A B C a b c)
    (h_sides : 2 * c * sin A = sqrt 3 * a) :
  C = π / 3 :=
sorry

theorem find_area (A B C a b c : ℝ) 
    (h_acute : is_acute_triangle A B C a b c)
    (h_C : C = π / 3) (h_b : b = 2) (h_c : c = sqrt 7) (h_a : a = 3) :
  let S := 1 / 2 * a * b * sin C in
  S = (3 * sqrt 3) / 2 :=
sorry

end find_angle_C_find_area_l585_585322


namespace original_amount_of_money_l585_585923

-- Define the conditions
variables (x : ℕ) -- daily allowance

-- Spending details
def spend_10_days := 6 * 10 - 6 * x
def spend_15_days := 15 * 3 - 3 * x

-- Lean proof statement
theorem original_amount_of_money (h : spend_10_days = spend_15_days) : (6 * 10 - 6 * x) = 30 :=
by
  sorry

end original_amount_of_money_l585_585923


namespace sum_of_numerator_and_denominator_of_fraction_equiv_2_52_in_lowest_terms_l585_585795

theorem sum_of_numerator_and_denominator_of_fraction_equiv_2_52_in_lowest_terms :
    let a := 63
    let b := 25
    a + b = 88 := by
  sorry

end sum_of_numerator_and_denominator_of_fraction_equiv_2_52_in_lowest_terms_l585_585795


namespace run_of_heads_before_tails_l585_585379

theorem run_of_heads_before_tails (q : ℚ) (m n : ℕ) :
  (q = 1) → (nat.gcd m n = 1) → (n > 0) → (m > 0) → (q = (m / n)) → (m + n = 2) :=
by
  intro hq hrel_prime hn_positive hm_positive h_fraction_form
  sorry

end run_of_heads_before_tails_l585_585379


namespace find_z_l585_585629

def z_value (i : ℂ) (z : ℂ) : Prop := z * (1 - (2 * i)) = 2 + (4 * i)

theorem find_z (i z : ℂ) (hi : i^2 = -1) (h : z_value i z) : z = - (2 / 5) + (8 / 5) * i := by
  sorry

end find_z_l585_585629


namespace probability_a_squared_plus_b_divisible_by_3_l585_585529

theorem probability_a_squared_plus_b_divisible_by_3 : 
  let a_vals := {n ∈ (Finset.range 11).erase 0 | n ∈ ∅ ∪ (Finset.range 10).map Nat.negSuccPnatCoe}
  let b_vals := {(m : ℤ) | m ∈ (Finset.range 11).erase 0}.image (λ x, - (x : ℤ))
  ∃ a ∈ a_vals, ∃ b ∈ b_vals, ↑((Finset.filter (λ x, (fst x) ^ 2 + snd x ≡ 0 [MOD 3]) ((a_vals ×ˢ b_vals)).card) / (a_vals.card * b_vals.card)) = 0.37 :=
by
  sorry

end probability_a_squared_plus_b_divisible_by_3_l585_585529


namespace pizza_slices_all_toppings_l585_585892

theorem pizza_slices_all_toppings (x : ℕ) :
  (16 = (8 - x) + (12 - x) + (6 - x) + x) → x = 5 := by
  sorry

end pizza_slices_all_toppings_l585_585892


namespace increase_average_grades_l585_585465

theorem increase_average_grades (XA XB : ℝ) (nA nB : ℕ) (k s : ℝ) 
    (hXA : XA = 44.2) (hXB : XB = 38.8) 
    (hnA : nA = 10) (hnB : nB = 10) 
    (hk : k = 41) (hs : s = 44) :
    let new_XA := (XA * nA - k - s) / (nA - 2)
    let new_XB := (XB * nB + k + s) / (nB + 2) in
    (new_XA > XA) ∧ (new_XB > XB) := by
  sorry

end increase_average_grades_l585_585465


namespace problem_statement_l585_585292

theorem problem_statement (x y : ℤ) (k : ℤ) (h : 4 * x - y = 3 * k) : 9 ∣ 4 * x^2 + 7 * x * y - 2 * y^2 :=
by
  sorry

end problem_statement_l585_585292


namespace smallest_difference_l585_585601

theorem smallest_difference (w x y z : ℕ) (hw : 0 < w) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (h1 : w * x * y * z = 8!) (h2 : w < x) (h3 : x < y) (h4 : y < z) : z - w = 16 := by
  sorry

end smallest_difference_l585_585601


namespace candies_eaten_l585_585157

variables (A B D : ℕ)

-- Conditions:
def condition1 : Prop := ∃ k1 k2 k3 : ℕ, k1 * 4 + k2 * 3 + k3 * 7 = 70
def condition2 : Prop := (B * 3 = A * 4) ∧ (D * 7 = A * 6)
def condition3 : Prop := A + B + D = 70

-- Theorem statement:
theorem candies_eaten (h1 : condition1) (h2 : condition2) (h3 : condition3) :
    A = 24 ∧ B = 18 ∧ D = 28 := sorry

end candies_eaten_l585_585157


namespace length_of_AD_l585_585697

theorem length_of_AD (AB BC CD DE : ℝ) (right_angle_B right_angle_C : Prop) :
  AB = 6 → BC = 7 → CD = 25 → DE = 15 → AD = Real.sqrt 274 :=
by
  intros
  sorry

end length_of_AD_l585_585697


namespace volume_ratio_truncated_pyramid_l585_585320

noncomputable def volume_ratio (AB A1B1 : ℝ) : ℝ :=
  (3 : ℝ) ^ 3 - 1

theorem volume_ratio_truncated_pyramid (AB A1B1 : ℝ) (h : AB / A1B1 = 3) :
  volume_ratio AB A1B1 ≈ 4.2 :=
by
  sorry

end volume_ratio_truncated_pyramid_l585_585320


namespace smallest_four_digit_multiple_of_18_l585_585967

theorem smallest_four_digit_multiple_of_18 : 
  ∃ (n : ℤ), 1000 ≤ n ∧ n < 10000 ∧ 18 ∣ n ∧ 
  ∀ (m : ℤ), (1000 ≤ m ∧ m < 10000 ∧ 18 ∣ m) → n ≤ m :=
sorry

end smallest_four_digit_multiple_of_18_l585_585967


namespace open_box_volume_l585_585888

theorem open_box_volume (l w s : ℝ) (hl : l = 48) (hw : w = 36) (hs : s = 8) :
  (l - 2 * s) * (w - 2 * s) * s = 5120 :=
by
  sorry

end open_box_volume_l585_585888


namespace circumcircle_tangent_to_omega_l585_585783

-- Definitions for the geometric problem
variables {A B C H B₁ C₁ : Type*}
variables [euclidean_geometry A B C H B₁ C₁]

-- Conditions as assumptions
variable (acute_triangle : acute_triangle A B C)
variable (H_is_orthocenter : orthocenter A B C H)
variable (B₁_on_BH : on_segment B H B₁)
variable (C₁_on_CH : on_segment C H C₁)
variable (B₁C₁_parallel_BC : parallel B₁ C₁ B C)
variable (circumcenter_omega_on_BC : on_line (circumcenter (triangle B₁ H C₁)) B C)

-- Proof goal (tangency of the circumcircle of triangle ABC and circle omega)
theorem circumcircle_tangent_to_omega :
  tangent (circumcircle (triangle A B C)) (circumcircle (triangle B₁ H C₁)) :=
sorry

end circumcircle_tangent_to_omega_l585_585783


namespace range_of_a_l585_585804

theorem range_of_a (a : ℝ) : 
  (∃ x : ℝ, (x^2 + a*x + 4 < 0)) ↔ (a < -4 ∨ a > 4) :=
by
  sorry

end range_of_a_l585_585804


namespace smallest_four_digit_multiple_of_18_l585_585977

theorem smallest_four_digit_multiple_of_18 : 
  ∃ (n : ℕ), 999 < n ∧ n < 10000 ∧ 18 ∣ n ∧ ∀ m : ℕ, 999 < m ∧ m < 10000 ∧ 18 ∣ m → n ≤ m ∧ n = 1008 := 
sorry

end smallest_four_digit_multiple_of_18_l585_585977


namespace shaded_region_area_is_700_l585_585222

noncomputable def large_square_area : ℝ :=
  40 * 40

noncomputable def triangle_area (base height : ℝ) : ℝ :=
  1 / 2 * base * height

def area_of_unshaded_triangles : ℝ :=
  triangle_area 30 30 + triangle_area 30 30

def area_of_shaded_region : ℝ :=
  large_square_area - area_of_unshaded_triangles

theorem shaded_region_area_is_700 :
  area_of_shaded_region = 700 :=
by
  sorry

end shaded_region_area_is_700_l585_585222


namespace arithmetic_seq_20th_term_l585_585208

theorem arithmetic_seq_20th_term {a1 d : ℕ} (h_a1 : a1 = 2) (h_d : d = 3) : 
  let a (n : ℕ) := a1 + (n - 1) * d in a 20 = 59 :=
by
  sorry

end arithmetic_seq_20th_term_l585_585208


namespace sum_S5_l585_585905

-- The definition of an arithmetic progression
def is_arithmetic_progression (a : ℕ → ℝ) : Prop :=
  ∃ d, ∀ n, a (n + 1) = a n + d

-- The given conditions in the problem
variables (a : ℕ → ℝ) (S : ℕ → ℝ)

axiom sum_of_first_n_terms : ∀ n, S n = (n / 2.0) * (2 * a 1 + (n - 1) * d)
axiom condition1 : a 2 * a 5 = 3 * a 3
axiom condition2 : a 4 - 9 * a 7 = 2

-- The statement to be proved
theorem sum_S5 : is_arithmetic_progression a → S 5 = 121 :=
by sorry

end sum_S5_l585_585905


namespace train_speed_l585_585539

theorem train_speed
  (length_train1 length_train2 : ℝ)
  (speed_train2 : ℝ)
  (time_to_cross : ℝ)
  (relative_speed_in_m_s : ℝ)
  (speed_train1 : ℝ) :
  length_train1 = 200 →
  length_train2 = 300 →
  speed_train2 = 36 →
  time_to_cross = 49.9960003199744 →
  relative_speed_in_m_s = 500 / time_to_cross →
  speed_train1 = relative_speed_in_m_s * 18 / 5 + speed_train2 →
  speed_train1 ≈ 72 :=
by
  intro h1 h2 h3 h4 h5 h6
  sorry

end train_speed_l585_585539


namespace smallest_four_digit_multiple_of_18_l585_585990

theorem smallest_four_digit_multiple_of_18 (n : ℕ) (h1 : 1000 ≤ n) (h2 : n < 10000) (h3 : n % 18 = 0) : n = 1008 :=
by
  have h4 : n ≥ 1008, sorry
  have h5 : n ≤ 1008, sorry
  exact eq_of_le_of_ge h4 h5

end smallest_four_digit_multiple_of_18_l585_585990


namespace family_member_bites_count_l585_585498

-- Definitions based on the given conditions
def cyrus_bites_arms_and_legs : Nat := 14
def cyrus_bites_body : Nat := 10
def family_size : Nat := 6
def total_bites_cyrus : Nat := cyrus_bites_arms_and_legs + cyrus_bites_body
def total_bites_family : Nat := total_bites_cyrus / 2

-- Translation of the question to a theorem statement
theorem family_member_bites_count : (total_bites_family / family_size) = 2 := by
  -- use sorry to indicate the proof is skipped
  sorry

end family_member_bites_count_l585_585498


namespace ineq_one_of_two_sqrt_amgm_l585_585511

-- Lean 4 statement for Question 1
theorem ineq_one_of_two (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x + y > 2) :
  (1 + x) / y < 2 ∨ (1 + y) / x < 2 :=
sorry

-- Lean 4 statement for Question 2
theorem sqrt_amgm (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  Real.sqrt ((a^2 + b^2 + c^2) / 3) ≥ (a + b + c) / 3 :=
sorry

end ineq_one_of_two_sqrt_amgm_l585_585511


namespace gcd_55555555_111111111_l585_585370

/-- Let \( m = 55555555 \) and \( n = 111111111 \).
We want to prove that the greatest common divisor (gcd) of \( m \) and \( n \) is 1. -/
theorem gcd_55555555_111111111 :
  let m := 55555555
  let n := 111111111
  Nat.gcd m n = 1 :=
by
  sorry

end gcd_55555555_111111111_l585_585370


namespace original_square_area_l585_585447

theorem original_square_area {x y : ℕ} (h1 : y ≠ 1)
  (h2 : x^2 = 24 + y^2) : x^2 = 49 :=
sorry

end original_square_area_l585_585447


namespace vector_sum_zero_l585_585631

/-- Proof problem statement: For any point P inside triangle ABC,
the sum of the dot products of the vectors AP, BP, and CP with the opposite sides BC, CA, and AB equals zero. -/
theorem vector_sum_zero (A B C P : ℝ^3) (h : P ∈ ↑(triangle A B C)) :
  (P - A) • (C - B) + (P - B) • (A - C) + (P - C) • (B - A) = 0 :=
sorry

-- Definitions and assumptions
definition triangle (A B C : ℝ^3) := conv ⟨A, B, C⟩

end vector_sum_zero_l585_585631


namespace can_construct_prism_with_fewer_than_20_shapes_l585_585482

/-
  We have 5 congruent unit cubes glued together to form complex shapes.
  4 of these cubes form a 4-unit high prism, and the fifth is attached to one of the inner cubes with a full face.
  Prove that we can construct a solid rectangular prism using fewer than 20 of these shapes.
-/

theorem can_construct_prism_with_fewer_than_20_shapes :
  ∃ (n : ℕ), n < 20 ∧ (∃ (length width height : ℕ), length * width * height = 5 * n) :=
sorry

end can_construct_prism_with_fewer_than_20_shapes_l585_585482


namespace triangle_is_right_angle_l585_585255

theorem triangle_is_right_angle (a b c : ℝ) 
  (h : a^2 + b^2 + c^2 - 12*a - 16*b - 20*c + 200 = 0) : 
  a^2 + b^2 = c^2 :=
by 
  sorry

end triangle_is_right_angle_l585_585255


namespace range_of_t_l585_585653

noncomputable def a_n (t : ℝ) (n : ℕ) : ℝ :=
  if n > 8 then ((1 / 3) - t) * (n:ℝ) + 2 else t ^ (n - 7)

theorem range_of_t (t : ℝ) :
  (∀ (n : ℕ), n ≠ 0 → a_n t n > a_n t (n + 1)) →
  (1/2 < t ∧ t < 1) :=
by
  intros h
  -- The proof would go here.
  sorry

end range_of_t_l585_585653


namespace smallest_four_digit_multiple_of_18_l585_585991

theorem smallest_four_digit_multiple_of_18 (n : ℕ) (h1 : 1000 ≤ n) (h2 : n < 10000) (h3 : n % 18 = 0) : n = 1008 :=
by
  have h4 : n ≥ 1008, sorry
  have h5 : n ≤ 1008, sorry
  exact eq_of_le_of_ge h4 h5

end smallest_four_digit_multiple_of_18_l585_585991


namespace sum_even_digits_101_to_200_l585_585195

noncomputable def E (n : ℕ) : ℕ :=
  (n.digits 10).filter (λ x, x % 2 = 0).sum

theorem sum_even_digits_101_to_200 :
  (∑ n in (Finset.range 100).map (λ x, x + 101), E n) + E 200 = 402 :=
by
  have h1 : (∑ n in (Finset.range 101).erase 0, E n) = 400 := sorry,
  have h2 : (∀ n, E (n + 101) = E n) := sorry,
  calc
    (∑ n in (Finset.range 100).map (λ x, x + 101), E n) + E 200
      = (∑ n in (Finset.range 99), E n + 1) + E 200 : by sorry
    ... = 400 + 2 : by rw [h1, E 200, add_comm]
    ... = 402 : rfl

end sum_even_digits_101_to_200_l585_585195


namespace foci_equality_ellipse_hyperbola_l585_585311

theorem foci_equality_ellipse_hyperbola (m : ℝ) (h : m > 0) 
  (hl: ∀ x y : ℝ, x^2 / 4 + y^2 / m^2 = 1 → 
     ∃ c : ℝ, c = Real.sqrt (4 - m^2)) 
  (hh: ∀ x y : ℝ, x^2 / m^2 - y^2 / 2 = 1 → 
     ∃ c : ℝ, c = Real.sqrt (m^2 + 2)) : 
  m = 1 :=
by {
  sorry
}

end foci_equality_ellipse_hyperbola_l585_585311


namespace max_ways_to_ascend_descend_l585_585454

theorem max_ways_to_ascend_descend :
  let east_paths := 2
  let west_paths := 1
  let south_paths := 3
  let north_paths := 4

  let descend_from_east := west_paths + south_paths + north_paths
  let descend_from_west := east_paths + south_paths + north_paths
  let descend_from_south := east_paths + west_paths + north_paths
  let descend_from_north := east_paths + west_paths + south_paths

  let ways_from_east := east_paths * descend_from_east
  let ways_from_west := west_paths * descend_from_west
  let ways_from_south := south_paths * descend_from_south
  let ways_from_north := north_paths * descend_from_north

  max ways_from_east (max ways_from_west (max ways_from_south ways_from_north)) = 24 := 
by
  -- Insert the proof here
  sorry

end max_ways_to_ascend_descend_l585_585454


namespace area_triangle_QPO_l585_585698

-- Given conditions as definitions
variables {A B C D P Q O M N : Type}
variables {rectangle : A × B × C × D}
variables {trisect_BC : ∃ N, sorry} -- Trisects BC at N
variables {meet_AB_DP : ∃ P,  sorry} -- DP meets AB at P
variables {trisect_AD : ∃ M,  sorry} -- Trisects AD at M
variables {meet_AB_CQ : ∃ Q,  sorry} -- CQ meets AB at Q
variables {intersection : ∃ O,  sorry} -- DP and CQ intersect at O
variables {area_rectangle : nat := 360} -- Area of the rectangle

-- Statement to prove in Lean
theorem area_triangle_QPO : sorry  := 140 := sorry

end area_triangle_QPO_l585_585698


namespace regular_polygons_that_tile_plane_l585_585602

def internal_angle (n : ℕ) : ℝ := (1 - 2 / n) * Real.pi

def can_tile_plane (n : ℕ) : Prop :=
  (∃ k : ℕ, k * internal_angle n = 2 * Real.pi ∧ 4 % (n - 2) = 0) ∨
  (∃ k : ℕ, (k - 1) * (internal_angle n) + internal_angle n / n = 2 * Real.pi ∧ 2 % (n - 2) = 0)

theorem regular_polygons_that_tile_plane :
  ∀ n : ℕ, can_tile_plane n ↔ n = 3 ∨ n = 4 ∨ n = 6 :=
by sorry

end regular_polygons_that_tile_plane_l585_585602


namespace snap_population_extinction_l585_585418

def initial_population := (12 * (71 ^ 12) + 12 * (71 ^ 11) + 10 * (71 ^ 10) + 22 * (71 ^ 9) +
                           10 * (71 ^ 8) + 29 * (71 ^ 7) + 17 * (71 ^ 6) + 11 * (71 ^ 5) +
                           24 * (71 ^ 4) + 23 * (71 ^ 3) + 10 * (71 ^ 2) + 23 * (71 ^ 1) +
                           35 * (71 ^ 0)) ^ 100

noncomputable def log_estimation : ℕ := 
  let log2_12 := Real.log 12 / Real.log 2
  let log2_71 := Real.log 71 / Real.log 2
  in floor (100 * log2_12 + 1300 * log2_71)

theorem snap_population_extinction : log_estimation = 8326 :=
by
  sorry

end snap_population_extinction_l585_585418


namespace handshake_count_l585_585560

/-- At a networking event, 15 women are divided into three age groups: under 25, 25 to 40, and over 40.
Each woman decides to only shake hands with women from other age groups or with women from the same
age group who have not yet shaken hands with each other. There are 5 women in each group. -/
theorem handshake_count :
  let inter_group := 3 * (5 * 5) in
  let intra_group := 3 * (nat.choose 5 2) in
  inter_group + intra_group = 105 :=
by
  sorry

end handshake_count_l585_585560


namespace arithmetic_sequence_fraction_l585_585940

theorem arithmetic_sequence_fraction :
  let numerator_seq := List.iota 50 |>.map (λ n, 3 + 3 * n)
  let denominator_seq := List.iota 49 |>.map (λ n, 3 + 2 * n)
  (numerator_seq.sum : ℚ) / (denominator_seq.sum) = 153 / 100 :=
by
  sorry

end arithmetic_sequence_fraction_l585_585940


namespace A_wins_with_25_marked_squares_l585_585509

-- Defining the game conditions
def game_condition (grid_size : ℕ) (A_marked_squares : ℕ) : Prop :=
  grid_size = 25 ∧ A_marked_squares = 25

-- The proof problem statement
theorem A_wins_with_25_marked_squares (H : game_condition 25 25) : 
  ∀ grid_size A_marked_squares, 
    grid_size = 25 → A_marked_squares = 25 → 
    (∀ B_placement_strategy, ∃ A_winning_strategy, A_wins A_winning_strategy B_placement_strategy) 
:=
sorry

end A_wins_with_25_marked_squares_l585_585509


namespace storm_deposited_water_l585_585542

-- Definitions of initial conditions
def reservoir_capacity := 400 * 10^9 -- in gallons
def before_storm_content := 200 * 10^9 -- in gallons
def after_storm_content := 0.80 * reservoir_capacity

-- Main theorem statement
theorem storm_deposited_water:
  after_storm_content - before_storm_content = 120 * 10^9 :=
by
  sorry

end storm_deposited_water_l585_585542


namespace july_percentage_is_correct_l585_585013

def total_scientists : ℕ := 120
def july_scientists : ℕ := 16
def july_percentage : ℚ := (july_scientists : ℚ) / (total_scientists : ℚ) * 100

theorem july_percentage_is_correct : july_percentage = 13.33 := 
by 
  -- Provides the proof directly as a statement
  sorry

end july_percentage_is_correct_l585_585013


namespace parallelogram_with_equal_diagonals_is_rectangle_l585_585523

open Set

theorem parallelogram_with_equal_diagonals_is_rectangle (P : Type) [EuclideanGeometry P] 
  (parallelogram : Prop)
  (equal_diagonals : parallelogram → Prop) :
  (parallelogram ∧ equal_diagonals) → is_rectangle parallelogram := 
by
  intros h
  sorry

end parallelogram_with_equal_diagonals_is_rectangle_l585_585523


namespace candies_eaten_l585_585165

-- Definitions

def Andrey_rate_eq_Boris_rate (candies_eaten_by_Andrey candies_eaten_by_Boris : ℕ) : Prop :=
  candies_eaten_by_Andrey / 4 = candies_eaten_by_Boris / 3

def Denis_rate_eq_Andrey_rate (candies_eaten_by_Denis candies_eaten_by_Andrey : ℕ) : Prop :=
  candies_eaten_by_Denis / 7 = candies_eaten_by_Andrey / 6

def total_candies (candies_eaten_by_Andrey candies_eaten_by_Boris candies_eaten_by_Denis : ℕ) : Prop :=
  candies_eaten_by_Andrey + candies_eaten_by_Boris + candies_eaten_by_Denis = 70

-- Theorem to prove the candies eaten by Andrey, Boris, and Denis
theorem candies_eaten (candies_eaten_by_Andrey candies_eaten_by_Boris candies_eaten_by_Denis : ℕ) :
  Andrey_rate_eq_Boris_rate candies_eaten_by_Andrey candies_eaten_by_Boris →
  Denis_rate_eq_Andrey_rate candies_eaten_by_Denis candies_eaten_by_Andrey →
  total_candies candies_eaten_by_Andrey candies_eaten_by_Boris candies_eaten_by_Denis →
  candies_eaten_by_Andrey = 24 ∧ candies_eaten_by_Boris = 18 ∧ candies_eaten_by_Denis = 28 :=
  by sorry

end candies_eaten_l585_585165


namespace ratio_avg_eq_42_l585_585023

theorem ratio_avg_eq_42 (a b c d : ℕ)
  (h1 : ∃ k : ℕ, a = 2 * k ∧ b = 3 * k ∧ c = 4 * k ∧ d = 5 * k)
  (h2 : (a + b + c + d) / 4 = 42) : a = 24 :=
by sorry

end ratio_avg_eq_42_l585_585023


namespace price_per_foot_l585_585522

theorem price_per_foot (area : ℝ) (cost : ℝ) (side_length : ℝ) (perimeter : ℝ) 
  (h1 : area = 289) (h2 : cost = 3740) 
  (h3 : side_length^2 = area) (h4 : perimeter = 4 * side_length) : 
  (cost / perimeter = 55) :=
by
  sorry

end price_per_foot_l585_585522


namespace total_clowns_in_acts_l585_585318

theorem total_clowns_in_acts :
  let act1_mobiles := 8
  let act1_clowns_per_mobile := 35
  let act2_mobiles := 10
  let act2_clowns_per_mobile := 40
  let act3_mobiles := 12
  let act3_clowns_per_mobile := 45
  let total_clowns := act1_mobiles * act1_clowns_per_mobile 
                      + act2_mobiles * act2_clowns_per_mobile 
                      + act3_mobiles * act3_clowns_per_mobile
  total_clowns = 1220 := by
  -- Act 1 clowns
  have act1_clowns : act1_mobiles * act1_clowns_per_mobile = 280 := by rfl
  -- Act 2 clowns
  have act2_clowns : act2_mobiles * act2_clowns_per_mobile = 400 := by rfl
  -- Act 3 clowns
  have act3_clowns : act3_mobiles * act3_clowns_per_mobile = 540 := by rfl
  -- Total clowns
  have total_clowns_calc : total_clowns = act1_clowns + act2_clowns + act3_clowns := by rfl
  rw [act1_clowns, act2_clowns, act3_clowns] at total_clowns_calc
  exact total_clowns_calc

end total_clowns_in_acts_l585_585318


namespace triangle_third_side_one_third_perimeter_l585_585081

theorem triangle_third_side_one_third_perimeter
  (a b x y p c : ℝ)
  (h1 : x^2 - y^2 = a^2 - b^2)
  (h2 : p = (a + b + c) / 2)
  (h3 : x - y = 2 * (a - b)) :
  c = (a + b + c) / 3 := by
  sorry

end triangle_third_side_one_third_perimeter_l585_585081


namespace rachel_homework_l585_585400

theorem rachel_homework : 5 + 2 = 7 := by
  sorry

end rachel_homework_l585_585400


namespace calculation_correct_l585_585495

theorem calculation_correct (a b : ℝ) : 4 * a^2 * b - 3 * b * a^2 = a^2 * b :=
by sorry

end calculation_correct_l585_585495


namespace geometric_region_intersection_l585_585734

theorem geometric_region_intersection (x y : ℝ) :
    x^2 + y^2 - 4 ≥ 0 ∧ x^2 - 1 ≥ 0 ∧ y^2 - 1 ≥ 0 ↔
    (x^2 + y^2 ≥ 4 ∧ (x ≤ -1 ∨ x ≥ 1) ∧ (y ≤ -1 ∨ y ≥ 1)) :=
begin
    sorry
end

end geometric_region_intersection_l585_585734


namespace range_of_a_l585_585737

noncomputable def f : ℝ → ℝ := sorry  -- function f, assuming it is defined
axiom monotone_f : ∀ x y : ℝ, (0 < x ∧ 0 < y ∧ x < y) → f x ≤ f y
axiom main_condition : ∀ x : ℝ, (0 < x) → f (f x - Real.exp x + x) = Real.exp 1
axiom inequality_condition : ∀ x : ℝ, (0 < x) → f x + (derivative f) x ≥ a * x

theorem range_of_a (a : ℝ) : a ≤ 2 * Real.exp 1 - 1 :=
by
  sorry

end range_of_a_l585_585737


namespace base_4_calculation_l585_585566

theorem base_4_calculation :
  (231_4 * 21_4 + 32_4 / 2_4) = 6130_4 :=
by sorry

end base_4_calculation_l585_585566


namespace smallest_four_digit_multiple_of_18_l585_585960

theorem smallest_four_digit_multiple_of_18 : ∃ n: ℕ, (1000 ≤ n ∧ n < 10000 ∧ 18 ∣ n) ∧ ∀ m: ℕ, (1000 ≤ m ∧ m < n ∧ 18 ∣ m) → false :=
begin
  use 1008,
  split,
  { split,
    { -- prove 1000 ≤ 1008
      linarith,
    },
    { split,
      { -- prove 1008 < 10000
        linarith,
      },
      { -- prove 18 ∣ 1008
        norm_num,
      }
    }
  },
  { -- prove there is no four-digit multiple of 18 less than 1008
    intros m,
    intro h,
    cases h with h1 h2,
    cases h2 with h3 h4,
    linarith,
  }
end

end smallest_four_digit_multiple_of_18_l585_585960


namespace altitudes_squared_equality_l585_585083

theorem altitudes_squared_equality (A B C A1 B1 C1 : Point) 
(altitude_A : is_altitude A A1)
(altitude_B : is_altitude B B1)
(altitude_C : is_altitude C C1) :
  dist A B1 ^ 2 + dist B C1 ^ 2 + dist C A1 ^ 2 = dist A C1 ^ 2 + dist B A1 ^ 2 + dist C B1 ^ 2 := sorry

end altitudes_squared_equality_l585_585083


namespace hexagon_area_of_circle_l585_585845

noncomputable def radius (area : ℝ) : ℝ :=
  Real.sqrt (area / Real.pi)

noncomputable def area_equilateral_triangle (s : ℝ) : ℝ :=
  (s * s * Real.sqrt 3) / 4

theorem hexagon_area_of_circle {r : ℝ} (h : π * r^2 = 100 * π) :
  6 * area_equilateral_triangle r = 150 * Real.sqrt 3 :=
by
  sorry

end hexagon_area_of_circle_l585_585845


namespace sum_triangle_inequality_solutions_l585_585138

theorem sum_triangle_inequality_solutions :
  (∑ m in (Finset.Icc 5 17), m) = 143 :=
by
  sorry

end sum_triangle_inequality_solutions_l585_585138


namespace problem_statement_l585_585233

noncomputable def coeff_sum (a : Fin 2017 → ℚ) : ℚ :=
  ∑ i in Finset.range 2017, (a i) / (2 ^ i)

theorem problem_statement
  (a : Fin 2017 → ℚ)
  (h : ∀ x : ℚ, (2 * x - 1) ^ 2016 = ∑ i in Finset.range 2017, (a i) * (x ^ i))
  (h₀ : a 0 = 1) :
  coeff_sum (function.update a 0 0) = -1 :=
by {
  sorry
}

end problem_statement_l585_585233


namespace smallest_period_sum_l585_585079

noncomputable def smallest_positive_period (f : ℝ → ℝ) (g : ℝ → ℝ): ℝ → ℝ :=
λ x => f x + g x

theorem smallest_period_sum
  (f g : ℝ → ℝ)
  (m n : ℕ)
  (hf : ∀ x, f (x + m) = f x)
  (hg : ∀ x, g (x + n) = g x)
  (hm : m > 1)
  (hn : n > 1)
  (hgcd : Nat.gcd m n = 1)
  : ∃ T, T > 0 ∧ (∀ x, smallest_positive_period f g (x + T) = smallest_positive_period f g x) ∧ T = m * n := by
  sorry

end smallest_period_sum_l585_585079


namespace function_passes_through_vertex_l585_585452

theorem function_passes_through_vertex (a : ℝ) (h₁ : 0 < a) (h₂ : a ≠ 1) : a^(2 - 2) + 1 = 2 :=
by
  sorry

end function_passes_through_vertex_l585_585452


namespace initial_men_employed_l585_585907

theorem initial_men_employed (M : ℕ) 
  (h1 : ∀ m d, m * d = 2 * 10)
  (h2 : ∀ m t, (m + 30) * t = 10 * 30) : 
  M = 75 :=
by
  sorry

end initial_men_employed_l585_585907


namespace distinct_fixed_points_count_l585_585367

def g (x : ℝ) : ℝ := (x + 8) / x

def sequence_of_functions (n : ℕ) : (ℝ → ℝ) :=
  if n = 1 then g else g ∘ sequence_of_functions (n - 1)

theorem distinct_fixed_points_count :
  ∃! (S : Finset ℝ), S.card = 2 ∧ (∀ (x : ℝ), x ∈ S ↔ ∃ n : ℕ, n > 0 ∧ sequence_of_functions n x = x) :=
sorry

end distinct_fixed_points_count_l585_585367


namespace domain_of_f_l585_585008

noncomputable def f (x : ℝ) : ℝ := (Real.sqrt x) / (x - 1)

theorem domain_of_f :
  { x : ℝ | x ≥ 0 ∧ x ≠ 1 } = set.Icc 0 1 ∪ set.Ioo 1 ∞ :=
by
  sorry

end domain_of_f_l585_585008


namespace series_convergent_l585_585229

-- Define the closest integer function ⟨n⟩ to √n.
def closest_int_to_sqrt (n : ℕ) : ℕ :=
  if 2 * n + 1 < (2 * sqrt n + 1) ^ 2 then sqrt n else sqrt n + 1

-- Define the hyperbolic sine function.
def sinh (x : ℝ) : ℝ := (Real.exp x - Real.exp (-x)) / 2

-- Define the infinite series we are evaluating.
noncomputable def series_sum : ℝ :=
  ∑' n : ℕ, sinh (closest_int_to_sqrt (n + 1)) / (3 ^ (n + 1))

-- Theorem stating the series is convergent.
theorem series_convergent : Summable (λ n : ℕ, sinh (closest_int_to_sqrt (n + 1)) / (3 ^ (n + 1))) :=
  sorry

end series_convergent_l585_585229


namespace range_of_EG_dihedral_angle_volume_of_tetrahedron_l585_585634

-- Definitions of given conditions
def Rectangle (A B C D : Type) (f_ab : A → B → ℕ) (f_ac : A → C → ℝ) := 
  ∃ AB CD : ℕ, ∃ AC : ℝ, AB = 5 ∧ AC = Real.sqrt 34

def Folding (e f a b d : Type) (AB : a → b → ℕ) (AD : a → d → ℕ) := 
  ∃ EF : d → e → Type, parallel EF AD

-- Proof of range for EG
theorem range_of_EG (A B E F G : Type) (g_ab : A → B → ℕ) (g_ac : A → B → ℝ) :
  Rectangle A B C D g_ab g_ac ∧ Folding E F A B D g_ab g_ab → ∀ EG : G → ℝ, 0 ≤ EG ∧ EG < 2.5 :=
sorry

-- Proof of dihedral angle given EG is the largest integer and AB is minimized
theorem dihedral_angle (A B D E F G : Type) (g_ac : A → B → ℝ) :
  Rectangle A B C D g_ab g_ac ∧ Folding E F A B D g_ab g_ab → ∀ EG : G → ℝ, 
  round EG = 2 → ∀ angle : ℝ, angle = Real.pi - Real.arccos (8 / 25) := 
sorry

-- Proof of volume of tetrahedron A - BFG
theorem volume_of_tetrahedron (A B F G : Type) (g_ab : A → B → ℕ) (EG : G → ℝ) :
  Rectangle A B C D g_ab g_ac ∧ Folding F G A B D g_ab g_ab → 
  volume (tetrahedron A B F G) = (3 * Real.sqrt 561 - 82) / 24 :=
sorry

end range_of_EG_dihedral_angle_volume_of_tetrahedron_l585_585634


namespace second_certificate_interest_rate_l585_585906

-- Define the conditions
def initial_investment : ℝ := 15000
def first_interest_rate : ℝ := 0.08
def first_period_growth_rate : ℝ := 1 + first_interest_rate / 4
def value_after_first_period : ℝ := initial_investment * first_period_growth_rate
def final_value : ℝ := 15612

-- Let s represent the annual interest rate (in percentage) of the second certificate
def s : ℝ := 8

-- The proof problem statement
theorem second_certificate_interest_rate
  (initial_investment : ℝ)
  (first_interest_rate : ℝ)
  (first_period_growth_rate : ℝ)
  (value_after_first_period : ℝ)
  (final_value : ℝ)
  (h1 : initial_investment = 15000)
  (h2 : first_interest_rate = 0.08)
  (h3 : first_period_growth_rate = 1 + first_interest_rate / 4)
  (h4 : value_after_first_period = initial_investment * first_period_growth_rate)
  (h5 : final_value = 15612)
  (h6 : value_after_first_period = 15300) :
  s = 8 := by
  sorry

end second_certificate_interest_rate_l585_585906


namespace transfer_both_increases_average_l585_585462

noncomputable def initial_average_A : ℚ := 44.2
noncomputable def initial_students_A : ℕ := 10
noncomputable def initial_sum_A : ℚ := initial_average_A * initial_students_A

noncomputable def initial_average_B : ℚ := 38.8
noncomputable def initial_students_B : ℕ := 10
noncomputable def initial_sum_B : ℚ := initial_average_B * initial_students_B

noncomputable def score_Kalinina : ℚ := 41
noncomputable def score_Sidorov : ℚ := 44

theorem transfer_both_increases_average :
  let new_sum_A := initial_sum_A - score_Kalinina - score_Sidorov,
      new_students_A := initial_students_A - 2,
      new_average_A := new_sum_A / new_students_A,
      new_sum_B := initial_sum_B + score_Kalinina + score_Sidorov,
      new_students_B := initial_students_B + 2,
      new_average_B := new_sum_B / new_students_B in
        new_average_A > initial_average_A ∧ new_average_B > initial_average_B :=
by
  sorry

end transfer_both_increases_average_l585_585462


namespace regular_hexagon_area_l585_585840

theorem regular_hexagon_area (A : ℝ) (r : ℝ) (hex_area : ℝ) :
  A = 100 * Real.pi → r = Real.sqrt 100 → 
  hex_area = 150 * Real.sqrt 3 → 
  150 * Real.sqrt 3 = 150 * Real.sqrt 3 :=
by
  sorry

end regular_hexagon_area_l585_585840


namespace sin_A_and_side_b_of_triangle_l585_585286

noncomputable def triangle_sides (a b c : ℝ) (A B C : ℝ) : Prop :=
3 * a = 5 * c * sin A ∧ cos B = -5 / 13

noncomputable def triangle_area (a b c : ℝ) (A B C : ℝ) (area : ℝ) : Prop :=
1 / 2 * a * c * sin B = area

theorem sin_A_and_side_b_of_triangle
  (a b c A B C : ℝ) (h1 : triangle_sides a b c A B C)
  (h2 : triangle_area a b c A B C 33 / 2) :
  sin A = 33 / 65 ∧ b = 10 :=
sorry

end sin_A_and_side_b_of_triangle_l585_585286


namespace paint_red_and_cut_then_count_l585_585025

def initial_cube_side_length : ℕ := 4

def cube_painted_faces (side_length : ℕ) : Prop :=
∀ (f : Fin 6 → set (Fin₃ × Fin₃ × Fin₃)), 
  (∀ p : Fin₃ × Fin₃ × Fin₃, ∃ f₁, f p) → 
  side_length = initial_cube_side_length

def cut_into_small_cubes (side_length small_cube_size : ℕ) : Prop :=
side_length % small_cube_size = 0

noncomputable def number_of_cubes_with_at_least_two_red_faces : ℕ :=
56

theorem paint_red_and_cut_then_count :
  ∀ (n : ℕ), cube_painted_faces n ∧ cut_into_small_cubes n 1 → n = initial_cube_side_length →
  number_of_cubes_with_at_least_two_red_faces = 56 :=
sorry

end paint_red_and_cut_then_count_l585_585025


namespace min_value_f_max_value_y_l585_585870

-- For problem 1
theorem min_value_f (x : ℝ) (h : x > 0) : 
  (∃ x_min : ℝ, x_min = 1 ∧ ∀ y > 0, f(y) = (2 / y + 2 * y) ≥ 4) ∧ f(1) = 4 :=
sorry

-- For problem 2
theorem max_value_y (x : ℝ) (h : 0 < x ∧ x < (1/3)) : 
  (∃ x_max : ℝ, x_max = 1/6 ∧ ∀ y > 0, y(y(1 - 3 * y)) ≤ 1/12) ∧ y(1/6) = 1/12 :=
sorry

end min_value_f_max_value_y_l585_585870


namespace sum_of_possible_m_values_l585_585127

theorem sum_of_possible_m_values :
  let m_range := Finset.Icc 5 17 in
  m_range.sum id = 143 := by
  sorry

end sum_of_possible_m_values_l585_585127


namespace intersection_trajectory_polar_radius_l585_585336

theorem intersection_trajectory :
  ∀ (k t m : ℝ) (P : ℝ × ℝ),
    (P = (2 + t, k * t)) ∧ (P = (-2 + m, m / k)) →
    (P.1^2 - P.2^2 = 4) ∧ (P.2 ≠ 0) :=
by
  intros k t m P H
  sorry

theorem polar_radius :
  ∀ (x y : ℝ), 
    (x + y = √2) ∧ (x^2 - y^2 = 4) →
    (√(x^2 + y^2) = √5) :=
by
  intros x y H
  sorry

end intersection_trajectory_polar_radius_l585_585336


namespace smallest_multiple_of_3_l585_585828

open Finset

theorem smallest_multiple_of_3 (cards : Finset ℕ) (cond1 : {1, 2, 6} ⊆ cards) :
  ∃ x, x ∈ {12, 16, 21, 26, 61, 62} ∧ (x % 3 = 0) ∧ (∀ y, y ∈ {12, 16, 21, 26, 61, 62} ∧ (y % 3 = 0) → x ≤ y) :=
by
  sorry

end smallest_multiple_of_3_l585_585828


namespace general_formula_a_n_sum_first_n_terms_b_n_l585_585694

variable (a n : ℕ) (a_n b_n S_n : ℕ → ℤ)

-- Define arithmetic sequence conditions
axiom a2_plus_a7 : a_n 2 + a_n 7 = -23
axiom a3_plus_a8 : a_n 3 + a_n 8 = -29

-- Define the general form of the geometric sequence
axiom geom_seq : ∀ n, a_n n + b_n n = 2 ^ (n - 1)

-- Prove the general formula for the arithmetic sequence a_n
theorem general_formula_a_n : (∀ n, a_n n = -3 * n + 2) := sorry

-- Prove the sum of the first n terms of b_n is as given
theorem sum_first_n_terms_b_n :
  (∀ n, (∑ k in finset.range n, b_n k) = ((3 * n^2 - n) / 2) + 2^n - 1) := sorry

end general_formula_a_n_sum_first_n_terms_b_n_l585_585694


namespace candies_eaten_l585_585168

-- Definitions

def Andrey_rate_eq_Boris_rate (candies_eaten_by_Andrey candies_eaten_by_Boris : ℕ) : Prop :=
  candies_eaten_by_Andrey / 4 = candies_eaten_by_Boris / 3

def Denis_rate_eq_Andrey_rate (candies_eaten_by_Denis candies_eaten_by_Andrey : ℕ) : Prop :=
  candies_eaten_by_Denis / 7 = candies_eaten_by_Andrey / 6

def total_candies (candies_eaten_by_Andrey candies_eaten_by_Boris candies_eaten_by_Denis : ℕ) : Prop :=
  candies_eaten_by_Andrey + candies_eaten_by_Boris + candies_eaten_by_Denis = 70

-- Theorem to prove the candies eaten by Andrey, Boris, and Denis
theorem candies_eaten (candies_eaten_by_Andrey candies_eaten_by_Boris candies_eaten_by_Denis : ℕ) :
  Andrey_rate_eq_Boris_rate candies_eaten_by_Andrey candies_eaten_by_Boris →
  Denis_rate_eq_Andrey_rate candies_eaten_by_Denis candies_eaten_by_Andrey →
  total_candies candies_eaten_by_Andrey candies_eaten_by_Boris candies_eaten_by_Denis →
  candies_eaten_by_Andrey = 24 ∧ candies_eaten_by_Boris = 18 ∧ candies_eaten_by_Denis = 28 :=
  by sorry

end candies_eaten_l585_585168


namespace part_a_part_b_l585_585362

noncomputable def varphi_bounded_density (varphi : ℝ → ℂ) : Prop :=
∃ f : ℝ → ℂ, (∀ x, bounded (f x)) ∧
(∫ t : ℝ, abs (real.part (varphi t)) < ∞) →
true

noncomputable def varphi_L2_density (varphi : ℝ → ℂ) : Prop :=
∃ f : ℝ → ℂ, (∫ x : ℝ, norm (f x) ^ 2 < ∞) ∧
(∫ t : ℝ, (abs (real.part (varphi t))) ^ 2 < ∞) →
true

theorem part_a : varphi_bounded_density varphi := 
sorry

theorem part_b : varphi_L2_density varphi :=
sorry

end part_a_part_b_l585_585362


namespace series_tangent_sum_l585_585762

variable {α : Type*} [Real α] (n : ℕ) (a : α)

theorem series_tangent_sum (h1: α ≠ 0) (h2: n > 0) :
  (∑ k in Finset.range n, (1 / (Real.cos (k * a) * Real.cos ((k + 1) * a))))
  = (Real.tan (n * a) - Real.tan a) / (Real.sin a) :=
by 
  sorry

end series_tangent_sum_l585_585762


namespace andrey_boris_denis_eat_candies_l585_585161

def andrey_boris_condition (a b : ℕ) : Prop :=
  a = 4 ∧ b = 3

def andrey_denis_condition (a d : ℕ) : Prop :=
  a = 6 ∧ d = 7

def total_candies_condition (total : ℕ) : Prop :=
  total = 70

theorem andrey_boris_denis_eat_candies :
  ∃ (a b d : ℕ), andrey_boris_condition a b ∧ andrey_denis_condition a d ∧ 
                  (total_candies_condition (2 * (12 + 9 + 14)) ∧ 
                   2 * 12 = 24 ∧ 2 * 9 = 18 ∧ 2 * 14 = 28) →
                  (a = 24 ∧ b = 18 ∧ d = 28) :=
by
  sorry

end andrey_boris_denis_eat_candies_l585_585161


namespace find_A_l585_585084

theorem find_A (A B C D: ℕ) (h1: A ≠ B) (h2: A ≠ C) (h3: A ≠ D) (h4: B ≠ C) (h5: B ≠ D) (h6: C ≠ D)
  (hAB: A * B = 72) (hCD: C * D = 72) (hDiff: A - B = C + D + 2) : A = 6 :=
sorry

end find_A_l585_585084


namespace tangent_line_at_point_P_fx_gt_2x_minus_ln_x_l585_585270

noncomputable def f (x : ℝ) : ℝ := Real.exp x / x
noncomputable def g (x : ℝ) := f x - 2 * (x - Real.log x)

theorem tangent_line_at_point_P :
  let P := (2 : ℝ, Real.exp 2 / 2)
  let tangent_eq := (fun (x y : ℝ) => Real.exp 2 * x - 4 * y = 0)
  tangent_eq (P.1) (P.2) :=
sorry

theorem fx_gt_2x_minus_ln_x (x : ℝ) (hx : 0 < x) : 
  f x > 2 * (x - Real.log x) :=
sorry

end tangent_line_at_point_P_fx_gt_2x_minus_ln_x_l585_585270


namespace volume_of_box_l585_585879

theorem volume_of_box (L W S : ℕ) (hL : L = 48) (hW : W = 36) (hS : S = 8) : 
  let new_length := L - 2 * S,
      new_width := W - 2 * S,
      height := S
  in new_length * new_width * height = 5120 := by
sorry

end volume_of_box_l585_585879


namespace shaded_area_is_correct_l585_585799

def area_of_rectangle (l w : ℕ) : ℕ := l * w

def area_of_triangle (b h : ℕ) : ℕ := (b * h) / 2

def area_of_shaded_region : ℕ :=
  let length := 8
  let width := 4
  let area_rectangle := area_of_rectangle length width
  let area_triangle := area_of_triangle length width
  area_rectangle - area_triangle

theorem shaded_area_is_correct : area_of_shaded_region = 16 :=
by
  sorry

end shaded_area_is_correct_l585_585799


namespace angle_ADC_145_l585_585325

theorem angle_ADC_145 (AB BC BD : ℝ) (angle_ABC : ℝ) (h1 : AB = BC) (h2 : BC = BD) (h3 : angle_ABC = 70) : 
  ∠ADC = 145 :=
by
  sorry

end angle_ADC_145_l585_585325


namespace ruth_ate_one_sandwich_l585_585766

theorem ruth_ate_one_sandwich :
  ∀ (prepared brother first_cousin each_of_two_cousins left : ℕ),
  prepared = 10 →
  brother = 2 →
  first_cousin = 2 →
  each_of_two_cousins = 1 →
  left = 3 →
  let total_eaten_by_others := brother + first_cousin + 2 * each_of_two_cousins in
  let total_eaten := prepared - left in
  total_eaten - total_eaten_by_others = 1 :=
by
  intros prepared brother first_cousin each_of_two_cousins left
  intros h1 h2 h3 h4 h5
  let total_eaten_by_others := brother + first_cousin + 2 * each_of_two_cousins
  let total_eaten := prepared - left
  have h : total_eaten - total_eaten_by_others = 1 := sorry
  exact h

end ruth_ate_one_sandwich_l585_585766


namespace biologist_mixed_solution_l585_585515

noncomputable def mixed_solution_volume (v1 : ℕ) (c1 c2 cm : ℝ) (x : ℕ) : ℕ :=
  v1 + x

theorem biologist_mixed_solution
  (v1 : ℕ) (c1 c2 cm : ℝ)
  (h1 : c1 = 0.03)
  (h2 : c2 = 0.12)
  (h3 : cm = 0.084)
  (h4 : v1 = 600)
  (x : ℕ)
  (hx : 0.03 * 600 + 0.12 * x = 0.084 * (600 + x)) :
  mixed_solution_volume v1 c1 c2 cm x = 1500 :=
begin
  sorry -- Proof goes here
end

end biologist_mixed_solution_l585_585515


namespace ant_reaches_bottom_vertex_l585_585102

-- Definitions based on the problem conditions
structure Dodecahedron :=
  (top_vertex : Type)
  (bottom_vertex : Type)
  (adjacent_vertices : Type → List Type)
  (num_adjacent_vertices : ∀ v, adjacent_vertices v = 5)
  (pyramid_structure : ∀ v, v ≠ bottom_vertex → ∃ adjacent_v, adjacent_v ≠ top_vertex ∧ adjacent_v = bottom_vertex)

def probability_reach_bottom_vertex (ant_path : Dodecahedron) : ℚ := 1 / 5

theorem ant_reaches_bottom_vertex (ant_path : Dodecahedron) :
  probability_reach_bottom_vertex ant_path = 1 / 5 :=
by
  -- Placeholder for the proof
  sorry

end ant_reaches_bottom_vertex_l585_585102


namespace find_lambda_l585_585662

def vector (α : Type*) := (α × α × α)

def a : vector ℝ := (2, -1, 1)
def b : vector ℝ := (-1, 4, -2)
def c (λ : ℝ) : vector ℝ := (11, 5, λ)

def coplanar (a b c : vector ℝ) : Prop :=
  ∃ m n : ℝ, (λ : vector ℝ) := m • a + n • b

theorem find_lambda : ∀ λ : ℝ, coplanar a b (c λ) → λ = 1 :=
by
  sorry

end find_lambda_l585_585662


namespace leading_coefficient_of_f_l585_585021

noncomputable def polynomial : Type := ℕ → ℝ

def satisfies_condition (f : polynomial) : Prop :=
  ∀ (x : ℕ), f (x + 1) - f x = 6 * x + 4

theorem leading_coefficient_of_f (f : polynomial) (h : satisfies_condition f) : 
  ∃ a b c : ℝ, (∀ (x : ℕ), f x = a * (x^2) + b * x + c) ∧ a = 3 := 
by
  sorry

end leading_coefficient_of_f_l585_585021


namespace initial_gasohol_amount_l585_585521

variable (x : ℝ)

def gasohol_ethanol_percentage (initial_gasohol : ℝ) := 0.05 * initial_gasohol
def mixture_ethanol_percentage (initial_gasohol : ℝ) := gasohol_ethanol_percentage initial_gasohol + 3

def optimal_mixture (total_volume : ℝ) := 0.10 * total_volume

theorem initial_gasohol_amount :
  ∀ (initial_gasohol : ℝ), 
  mixture_ethanol_percentage initial_gasohol = optimal_mixture (initial_gasohol + 3) →
  initial_gasohol = 54 :=
by
  intros
  sorry

end initial_gasohol_amount_l585_585521


namespace triangle_inequality_for_sum_l585_585080

theorem triangle_inequality_for_sum (n : ℕ) (t : ℕ → ℝ) (ht : ∀ i, 1 ≤ t i) :
  (∀ i j k, 1 ≤ i ∧ i < j ∧ j < k ∧ k ≤ n → 
    n^2 + 1 > (∑ k in Finset.range n, t k) * (∑ k in Finset.range n, 1 / t k)) :=
by
  sorry

end triangle_inequality_for_sum_l585_585080


namespace solution_set_ineq_l585_585803

theorem solution_set_ineq (x : ℝ) : (1 < x ∧ x ≤ 3) ↔ (x - 3) / (x - 1) ≤ 0 := sorry

end solution_set_ineq_l585_585803


namespace tournament_rounds_l585_585037

theorem tournament_rounds (A_played : ℕ) (B_played : ℕ) (C_referee : ℕ) 
  (hA : A_played = 5) (hB : B_played = 6) (hC : C_referee = 2) : 
  A_played - C_referee + B_played - C_referee + C_referee = 9 := 
by
  simp [hA, hB, hC]
  exact sorry

end tournament_rounds_l585_585037


namespace sylvia_buttons_l585_585780

theorem sylvia_buttons (n : ℕ) (h₁: n % 10 = 0) (h₂: n ≥ 80):
  (∃ w : ℕ, w = (n - (n / 2) - (n / 5) - 8)) ∧ (n - (n / 2) - (n / 5) - 8 = 1) :=
by
  sorry

end sylvia_buttons_l585_585780


namespace f_neg_l585_585635

-- Define the function f(x) and its properties
def f (x : ℝ) : ℝ := if x >= 0 then x^3 - 2*x^2 - x else -(x^3 + 2*x^2 - x)

theorem f_neg (x : ℝ) (h : x < 0) : f x = x^3 + 2*x^2 - x :=
by {
  have h1 : -x >= 0 := by {linarith},
  have h2 : f(-x) = -(-x^3 - 2*x^2 + x) := by {rw f, simpa [h1] using f(-x)},
  simpa [f, h] using h2,
}

#check f_neg

end f_neg_l585_585635


namespace construct_x45_l585_585541

-- Definitions for General Position and Axis, and Profile Line
constant Line : Type
constant Axis : Type
constant general_position_line : Line → Prop
constant arbitrary_axis : Axis → Prop
constant perpendicular : Axis → Line → Prop
constant profile_line_in_system : Line → Axis → Axis → Prop

-- Given conditions
variables (g : Line) (x_14 : Axis)
hypotheses (H1 : general_position_line g)
           (H2 : arbitrary_axis x_14)

-- The statement to prove
theorem construct_x45 (g : Line) (x_14 : Axis) 
  (H1 : general_position_line g) 
  (H2 : arbitrary_axis x_14) :
  ∃ (x_45 : Axis), ∃ (g_IV : Line), perpendicular x_45 g_IV ∧ profile_line_in_system g x_14 x_45 :=
sorry

end construct_x45_l585_585541


namespace draw_13_cards_no_straight_flush_l585_585520

theorem draw_13_cards_no_straight_flush :
  let deck_size := 52
  let suit_count := 4
  let rank_count := 13
  let non_straight_flush_draws (n : ℕ) := 3^n - 3
  n = rank_count →
  ∀ (draw : ℕ), draw = non_straight_flush_draws n :=
by
-- Proof would be here
sorry

end draw_13_cards_no_straight_flush_l585_585520


namespace g_function_satisfies_l585_585436

theorem g_function_satisfies (g : ℝ → ℝ) 
    (h : ∀ a c : ℝ, c^3 * g(a) = a^3 * g(c)) 
    (h_nonzero : g 3 ≠ 0) :
  (g 6 - g 2) / g 3 = 208 / 27 := sorry

end g_function_satisfies_l585_585436


namespace ratio_of_milk_to_thermos_capacity_l585_585571

def thermos_capacity : ℕ := 20
def fills_per_day : ℕ := 2
def days_per_week : ℕ := 5
def coffee_per_week_now : ℕ := 40
def coffee_fraction : ℚ := 1 / 4

theorem ratio_of_milk_to_thermos_capacity :
  let coffee_per_week : ℕ := coffee_per_week_now / coffee_fraction
  let total_capacity_per_week : ℕ := thermos_capacity * fills_per_day * days_per_week
  let milk_per_week : ℕ := total_capacity_per_week - coffee_per_week
  let milk_per_filling : ℚ := milk_per_week / (fills_per_day * days_per_week)
  (milk_per_filling / thermos_capacity) = (1 / 5) := by
  sorry

end ratio_of_milk_to_thermos_capacity_l585_585571


namespace vityaCanDetermineFirstDigit_l585_585757

noncomputable def canVityaDetermineFirstDigitWithIn4Moves (petyaNum : String) : Prop :=
  ∀ (vityaNum1 vityaNum2 vityaNum3 vityaNum4: String), 
  petyaNum.length = 9 ∧ 
  (∀ digit, digit ∈ "123456789" ∧ ∀ i in [0..8], petyaNum[i] = digit) →
  (∃ firstDigit : Char, 
    (firstDigit ∈ "123456789") ∧
    (vityaNum1.length = 9 ∧ vityaNum2.length = 9 ∧ vityaNum3.length = 9 ∧ vityaNum4.length = 9) ∧
    (vityaNum1[0] = firstDigit ∨ vityaNum2[0] = firstDigit ∨ vityaNum3[0] = firstDigit ∨ vityaNum4[0] = firstDigit) ∧
    (petyaNum[0] = vityaNum1[0] ∨ petyaNum[0] = vityaNum2[0] ∨ petyaNum[0] = vityaNum3[0] ∨ petyaNum[0] = vityaNum4[0])
  )

theorem vityaCanDetermineFirstDigit : 
  ∀ (petyaNum : String), canVityaDetermineFirstDigitWithIn4Moves petyaNum :=
by
  sorry

end vityaCanDetermineFirstDigit_l585_585757


namespace trail_mix_max_portions_l585_585720

theorem trail_mix_max_portions (n d c f : ℕ) (hn : n = 16) (hd : d = 6) (hc : c = 8) (hf : f = 4) :
  let x := min (min (n / 4) (d / 3)) (min (c / 2) (f / 1)) in
  x = 2 :=
by
  sorry

end trail_mix_max_portions_l585_585720


namespace final_answer_l585_585891

theorem final_answer : (848 / 8) - 100 = 6 := 
by
  sorry

end final_answer_l585_585891


namespace replaced_crew_member_weight_l585_585421

def W := 53
def avg_increase := 1.8
def new_man_weight := 71
def num_oarsmen := 10
def total_weight_increase := num_oarsmen * avg_increase

theorem replaced_crew_member_weight : 
  new_man_weight - W = total_weight_increase →
  W = 53 :=
by
  intro h
  unfold W at h
  unfold total_weight_increase at h
  unfold num_oarsmen at h
  unfold avg_increase at h
  linarith

end replaced_crew_member_weight_l585_585421


namespace sum_of_radii_of_c1_and_c2_is_ten_l585_585478

noncomputable def circle1_center : (ℝ × ℝ) := (3, 4)
noncomputable def circle2_center : (ℝ × ℝ) := (3, 4)
noncomputable def circle3_center : (ℝ × ℝ) := (0, 0)
noncomputable def circle3_radius : ℝ := 2

theorem sum_of_radii_of_c1_and_c2_is_ten :
  ∃ r1 r2 : ℝ, 
  sqrt ((circle1_center.1 - circle3_center.1)^2 + (circle1_center.2 - circle3_center.2)^2) = circle3_radius + r1 ∧
  sqrt ((circle2_center.1 - circle3_center.1)^2 + (circle2_center.2 - circle3_center.2)^2) = circle3_radius + r2 ∧
  r1 + r2 = 10 :=
by
  sorry

end sum_of_radii_of_c1_and_c2_is_ten_l585_585478


namespace clay_pot_cost_difference_l585_585787

def flower_cost : ℕ := 9
def soil_cost : ℕ := flower_cost - 2
def total_cost : ℕ := 45

theorem clay_pot_cost_difference :
  ∃ (clay_pot_cost : ℕ), 
  flower_cost + clay_pot_cost + soil_cost = total_cost ∧
  (clay_pot_cost - flower_cost = 20) :=
by
  -- Definitions from the problem:
  let clay_pot_cost := 29 in
  have h1 : flower_cost + clay_pot_cost + soil_cost = total_cost := by
    simp [flower_cost, soil_cost], norm_num,
  have h2 : clay_pot_cost - flower_cost = 20 := by
    simp [flower_cost], norm_num,
  use clay_pot_cost,
  exact ⟨h1, h2⟩

end clay_pot_cost_difference_l585_585787


namespace smallest_angle_in_convex_15sided_polygon_l585_585000

def isConvexPolygon (n : ℕ) (angles : Fin n → ℚ) : Prop :=
  ∑ i, angles i = (n - 2) * 180 ∧ ∀ i,  angles i < 180

def arithmeticSequence (angles : Fin 15 → ℚ) : Prop :=
  ∃ a d : ℚ, ∀ i : Fin 15, angles i = a + i * d

def increasingSequence (angles : Fin 15 → ℚ) : Prop :=
  ∀ i j : Fin 15, i < j → angles i < angles j

def integerSequence (angles : Fin 15 → ℚ) : Prop :=
  ∀ i : Fin 15, (angles i : ℚ) = angles i

theorem smallest_angle_in_convex_15sided_polygon :
  ∃ (angles : Fin 15 → ℚ),
    isConvexPolygon 15 angles ∧
    arithmeticSequence angles ∧
    increasingSequence angles ∧
    integerSequence angles ∧
    angles 0 = 135 :=
by
  sorry

end smallest_angle_in_convex_15sided_polygon_l585_585000


namespace hexagon_area_of_circle_l585_585846

noncomputable def radius (area : ℝ) : ℝ :=
  Real.sqrt (area / Real.pi)

noncomputable def area_equilateral_triangle (s : ℝ) : ℝ :=
  (s * s * Real.sqrt 3) / 4

theorem hexagon_area_of_circle {r : ℝ} (h : π * r^2 = 100 * π) :
  6 * area_equilateral_triangle r = 150 * Real.sqrt 3 :=
by
  sorry

end hexagon_area_of_circle_l585_585846


namespace linear_correlation_l585_585309

variable (r : ℝ) (r_critical : ℝ)

theorem linear_correlation (h1 : r = -0.9362) (h2 : r_critical = 0.8013) :
  |r| > r_critical :=
by
  sorry

end linear_correlation_l585_585309


namespace sum_of_possible_m_values_l585_585124

theorem sum_of_possible_m_values : 
  sum (setOf m in {m | 4 < m ∧ m < 18}) = 132 :=
by
  sorry

end sum_of_possible_m_values_l585_585124


namespace minimum_value_a_l585_585628

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := if x > 0 then exp x + a else -f a (-x)

theorem minimum_value_a (a : ℝ) :
  (∀ x : ℝ, f a x = -f a (-x)) ∧
  (∀ x : ℝ, x > 0 → f a x = exp x + a) ∧
  (∀ x y : ℝ, x ≤ y → f a x ≤ f a y) →
  a ≥ -1 :=
by sorry

end minimum_value_a_l585_585628


namespace find_s_when_t_is_64_l585_585265

theorem find_s_when_t_is_64 (s : ℝ) (t : ℝ) (h1 : t = 8 * s^3) (h2 : t = 64) : s = 2 :=
by
  -- Proof will be written here
  sorry

end find_s_when_t_is_64_l585_585265


namespace sum_of_all_possible_values_of_abs_b_l585_585765

theorem sum_of_all_possible_values_of_abs_b {a b : ℝ}
  {r s : ℝ} (hr : r^3 + a * r + b = 0) (hs : s^3 + a * s + b = 0)
  (hr4 : (r + 4)^3 + a * (r + 4) + b + 240 = 0) (hs3 : (s - 3)^3 + a * (s - 3) + b + 240 = 0) :
  |b| = 20 ∨ |b| = 42 →
  20 + 42 = 62 :=
by
  sorry

end sum_of_all_possible_values_of_abs_b_l585_585765


namespace possible_values_m_l585_585269

def f (x : ℝ) : ℝ := -x^3 + 2*x^2 - 3*x

def g (x : ℝ) : ℝ := 2*x^3 + x^2 - 4*x + 3

def g' (x : ℝ) : ℝ := 6*x^2 + 2*x - 4

theorem possible_values_m (m : ℤ) (h : m ∈ Set.range (λ x, g x)) : m = 4 ∨ m = 5 :=
by
  sorry

end possible_values_m_l585_585269


namespace open_box_volume_l585_585889

theorem open_box_volume (l w s : ℝ) (hl : l = 48) (hw : w = 36) (hs : s = 8) :
  (l - 2 * s) * (w - 2 * s) * s = 5120 :=
by
  sorry

end open_box_volume_l585_585889


namespace smallest_digit_divisible_by_9_l585_585953

theorem smallest_digit_divisible_by_9 : 
  ∃ d : ℕ, d ∈ finset.range(10) ∧ 9 ∣ (22 + d) ∧ 
    ∀ k : ℕ, k ∈ finset.range(10) ∧ 9 ∣ (22 + k) → d ≤ k :=
sorry

end smallest_digit_divisible_by_9_l585_585953


namespace linda_babysitting_hours_l585_585743

-- Define constants
def hourly_wage : ℝ := 10.0
def application_fee : ℝ := 25.0
def number_of_colleges : ℝ := 6.0

-- Theorem statement
theorem linda_babysitting_hours : 
    (application_fee * number_of_colleges) / hourly_wage = 15 := 
by
  -- Here the proof would go, but we'll use sorry as per instructions
  sorry

end linda_babysitting_hours_l585_585743


namespace trisect_ln_points_l585_585822

theorem trisect_ln_points
  (x1 x2 : ℝ) (h1 : 0 < x1) (h2 : x1 < x2)
  (x1_eq : x1 = 1) (x2_eq : x2 = 8) :
  let f (x : ℝ) := Real.log (x^2),
      A : ℝ × ℝ := (x1, f x1),
      B : ℝ × ℝ := (x2, f x2),
      yC : ℝ := (2 / 3) * (f x1) + (1 / 3) * (f x2),
      yD : ℝ := (1 / 3) * (f x1) + (2 / 3) * (f x2),
      x3 := Real.exp ((1 / 3) * Real.log (64)),
      x4 := Real.exp ((2 / 3) * Real.log (64))
  in x3 = 4 ∧ x4 = 16 :=
by
  let f (x : ℝ) := Real.log (x^2)
  let A : ℝ × ℝ := (x1, f x1)
  let B : ℝ × ℝ := (x2, f x2)
  let yC : ℝ := (2 / 3) * (f x1) + (1 / 3) * (f x2)
  let yD : ℝ := (1 / 3) * (f x1) + (2 / 3) * (f x2)
  let x3 := Real.exp ((1 / 3) * Real.log (64))
  let x4 := Real.exp ((2 / 3) * Real.log (64))
  sorry

end trisect_ln_points_l585_585822


namespace candy_eating_l585_585181

-- Definitions based on the conditions
def candies (andrey_rate boris_rate denis_rate : ℕ) (andrey_candies boris_candies denis_candies total_candies : ℕ) : Prop :=
  andrey_rate = 4 ∧ boris_rate = 3 ∧ denis_rate = 7 ∧ andrey_candies = 24 ∧ boris_candies = 18 ∧ denis_candies = 28 ∧
  total_candies = andrey_candies + boris_candies + denis_candies

-- Problem statement
theorem candy_eating :
  ∃ (a b d : ℕ), 
    candies 4 3 7 a b d 70 :=
sorry

end candy_eating_l585_585181


namespace regular_hexagon_area_inscribed_in_circle_l585_585837

theorem regular_hexagon_area_inscribed_in_circle
  (h : Real.pi * r^2 = 100 * Real.pi) :
  6 * (r^2 * Real.sqrt 3 / 4) = 150 * Real.sqrt 3 :=
by {
  sorry
}

end regular_hexagon_area_inscribed_in_circle_l585_585837


namespace necessary_but_not_sufficient_condition_l585_585857

theorem necessary_but_not_sufficient_condition (m : ℝ) :
  (m < 1) → (∀ x y : ℝ, (x - m) ^ 2 + y ^ 2 = m ^ 2 → (x, y) ≠ (1, 1)) :=
sorry

end necessary_but_not_sufficient_condition_l585_585857


namespace gcd_m_n_eq_one_l585_585593

/-- Mathematical definitions of m and n. --/
def m : ℕ := 123^2 + 235^2 + 347^2
def n : ℕ := 122^2 + 234^2 + 348^2

/-- Listing the conditions and deriving the result that gcd(m, n) = 1. --/
theorem gcd_m_n_eq_one : gcd m n = 1 :=
by sorry

end gcd_m_n_eq_one_l585_585593


namespace rectangle_difference_length_width_l585_585677

theorem rectangle_difference_length_width (x y p d : ℝ) (h1 : x + y = p / 2) (h2 : x^2 + y^2 = d^2) (h3 : x > y) : 
  x - y = (Real.sqrt (8 * d^2 - p^2)) / 2 := sorry

end rectangle_difference_length_width_l585_585677


namespace common_factor_of_polynomials_l585_585732

theorem common_factor_of_polynomials (P Q R S : Polynomial ℂ)
  (h : ∀ x : ℂ, P (x^5) + x * Q (x^5) + x^2 * R (x^5) = (x^4 + x^3 + x^2 + x + 1) * S x) :
  (x-1) ∣ P(x) ∧ (x-1) ∣ Q(x) ∧ (x-1) ∣ R(x) ∧ (x-1) ∣ S(x) := sorry

end common_factor_of_polynomials_l585_585732


namespace inequality_and_monotonicity_problems_l585_585381

variables {a b c : ℝ}

theorem inequality_and_monotonicity_problems (h1: a > b) (h2: b > 1) (h3: c < 0) :
  (c / a > c / b) ∧ (a^c < b^c) ∧ (Real.log (a - c) / Real.log b > Real.log (b - c) / Real.log a) :=
by
  -- Condition translating c to valid logarithmic base
  have log_base_a_gt_1: a - c > 1 := by
    have ac_pos: a - c > 0 := by linarith
    have b_pos: b > 1 := by linarith
    exact ac_pos -- Use simplified case to fit into lean's base log rule
  -- Providing proofs using provided conditions
  sorry

end inequality_and_monotonicity_problems_l585_585381


namespace isosceles_triangle_area_theorem_l585_585477

noncomputable def isosceles_triangle_area : ℝ :=
  let P := (0 : ℝ, 0 : ℝ)
  let Q := (a : ℝ, b : ℝ)
  let R := (a : ℝ, -b : ℝ)
  let S := ((2 * a) /3, 0 : ℝ)
  let T := (a / 2, b / 2 : ℝ)
  have h1 : (P  , S) = (P.x , S.y) := rfl
  have h2 : (Q , S) = (Q.x , S.y) := rfl
  have h3 : is_perpendicular (P , S) (Q , S) : by simp [ h1 , h2 ]
  let PS := (10 : ℝ)
  let QR := (15 : ℝ)
  let PQ := PS + QR
  let area := (b - a) * PQ
  have h4 : area = 300
  exact h4

theorem isosceles_triangle_area_theorem :
  isosceles_triangle_area = 300 := sorry

end isosceles_triangle_area_theorem_l585_585477


namespace min_r_minus_q_l585_585038
open Nat

-- Define the conditions
def pqr_satisfy_8_factorial (p q r : ℕ) : Prop :=
  p * q * r = fact 8 ∧ p < q ∧ q < r

-- The theorem statement
theorem min_r_minus_q (p q r : ℕ) (h : pqr_satisfy_8_factorial p q r) : r - q = 6 :=
sorry

end min_r_minus_q_l585_585038


namespace selection_of_hexagonal_shape_l585_585402

-- Lean 4 Statement: Prove that there are 78 distinct ways to select diagram b from the hexagonal grid of diagram a, considering rotations.

theorem selection_of_hexagonal_shape :
  let center_positions := 1
  let first_ring_positions := 6
  let second_ring_positions := 12
  let third_ring_positions := 6
  let fourth_ring_positions := 1
  let total_positions := center_positions + first_ring_positions + second_ring_positions + third_ring_positions + fourth_ring_positions
  let rotations := 3
  total_positions * rotations = 78 := by
  -- You can skip the explicit proof body here, replace with sorry
  sorry

end selection_of_hexagonal_shape_l585_585402


namespace area_under_curve_tangent_lines_l585_585644

noncomputable def f (x : ℝ) := x^2 - 3
noncomputable def g (x : ℝ) := x * f(x)

theorem area_under_curve :
  let S := 2 * ∫ x in 0..(Real.sqrt 3), f x in
  S = 4 * Real.sqrt 3 :=
by 
  sorry

theorem tangent_lines :
  (∃ m b, (1, -2) ∈ {p : ℝ × ℝ | p.2 = m * p.1 + b} ∧ ∃ x0 ≠ 1, g x0 = m * x0 + b ∧ g'(x0) = m)
  ∧ (1, -2) ∈ {p : ℝ × ℝ | p.2 = -2} ∧ tangent? :
by 
  sorry

end area_under_curve_tangent_lines_l585_585644


namespace integer_condition_l585_585616

theorem integer_condition (n k : ℕ) (h1 : 1 ≤ k) (h2 : k < n) :
  ∃ m : ℤ, (n - 2 * k - 1) * nat.comb n k = m * (k + 1) := by
sorry

end integer_condition_l585_585616


namespace segment_length_l585_585057

theorem segment_length (x : ℝ) (y : ℝ) (u : ℝ) (v : ℝ) :
  (|x - u| = 5 ∧ |y - u| = 5 ∧ u = √[3]{27} ∧ v = √[3]{27}) → (|x - y| = 10) :=
by
  sorry

end segment_length_l585_585057


namespace proof_problem_l585_585703

noncomputable def problem_statement : Prop :=
  ∀ (OA OB OC : EuclideanSpace ℝ (Fin 3)),
    ∥OA∥ = 2 →
    ∥OB∥ = 3 →
    ∥OC∥ = 2 →
    let angleAOC := real.arctan 3 in
    let angleBOC := π/2 in
    ∃ (p q : ℝ),
      OC = p • OA + q • OB ∧
      (p, q) = (real.sqrt 10 / 5, 0)

theorem proof_problem : problem_statement :=
by {
  intros OA OB OC h₁ h₂ h₃ h₄ h₅,
  use real.sqrt 10 / 5,
  use 0,
  split,
  { sorry },
  { exact ⟨rfl, rfl⟩ }
}

end proof_problem_l585_585703


namespace cuboid_surface_area_l585_585072

def length : ℕ := 8
def breadth : ℕ := 10
def height : ℕ := 12

def surface_area (l b h : ℕ) : ℕ := 2 * (l * b + b * h + h * l)

theorem cuboid_surface_area :
  surface_area length breadth height = 592 := 
by
  sorry

end cuboid_surface_area_l585_585072


namespace find_ratio_c_a_find_value_of_b_l585_585248

theorem find_ratio_c_a 
  (A B C a b c : ℝ) 
  (h1 : c = a) 
  (h2 : sin (A + B) = sin (C + B)) 
  (h3 : A + B + C = Real.pi) : 
  c / a = 1 :=
by
  sorry

theorem find_value_of_b
  (A B C a b c : ℝ)
  (h_cos_B : cos B = 2 / 3)
  (h_area : a * b * sin C / 2 = sqrt 5 / 6)
  (h1 : c = a) :
  b = (sqrt 6) / 3 :=
by
  sorry

end find_ratio_c_a_find_value_of_b_l585_585248


namespace discount_difference_l585_585536

noncomputable def true_discount (p : ℝ) : ℝ :=
let first_discount := 0.75 * p in
let second_discount := 0.9 * first_discount in
p - second_discount

def claimed_discount (p : ℝ) : ℝ :=
p * 0.35

theorem discount_difference (p : ℝ) :
  claimed_discount p - true_discount p = p * 0.025 :=
by
  sorry

end discount_difference_l585_585536


namespace distance_between_skew_lines_dihedral_angle_between_planes_l585_585186

variables (a : ℝ)
variables (ABC A1B1C1 B1C1CB AB AC1) : Prop

theorem distance_between_skew_lines 
  (h1 : ∀ D, midpoint(B,C,D) → perpendicular(AD,BC) )
  (h2 : AA1 ∥ CC1)
  (h3 : plane(B1C1CB) ⊥ plane(ABC))
  (h4 : AC1 ⊥ BC) :
  distance(AA1, B1C1) = (√3/2) * a := sorry

theorem dihedral_angle_between_planes 
  (h1 : ∀ O, ∀ E, B1O ⊥ BC → OE ⊥ AB → OB = a/2 → OE = (√3/4) * a )
  (h2 : angle(B1EO) = arctan 2) :
  dihedral_angle(A1B1BA, ABC) = π - arctan 2 := sorry

end distance_between_skew_lines_dihedral_angle_between_planes_l585_585186


namespace period_sum_sin_cos_l585_585487

noncomputable def period_sin_5x : ℝ := 2 * Real.pi / 5
noncomputable def period_cos_3x : ℝ := 2 * Real.pi / 3

theorem period_sum_sin_cos : 
  ∀ x : ℝ, period_sin_5x = 2 * Real.pi / 5 ∧ period_cos_3x = 2 * Real.pi / 3 → 
    (∀ y, y = sin (5 * x) + cos (3 * x) → (∃ p : ℝ, p = 2 * Real.pi)) :=
by
  intros x cond y h
  use 2 * Real.pi
  sorry

end period_sum_sin_cos_l585_585487


namespace gcd_m_n_l585_585371

noncomputable def m := 55555555
noncomputable def n := 111111111

theorem gcd_m_n : Int.gcd m n = 1 := by
  sorry

end gcd_m_n_l585_585371


namespace solve_for_x_l585_585704

theorem solve_for_x (h_perimeter_square : ∀(s : ℝ), 4 * s = 64)
  (h_height_triangle : ∀(h : ℝ), h = 48)
  (h_area_equal : ∀(s h x : ℝ), s * s = 1/2 * h * x) : 
  x = 32 / 3 := by
  sorry

end solve_for_x_l585_585704


namespace paco_salty_cookies_left_l585_585756

-- Define the initial number of salty cookies Paco had
def initial_salty_cookies : ℕ := 26

-- Define the number of salty cookies Paco ate
def eaten_salty_cookies : ℕ := 9

-- The theorem statement that Paco had 17 salty cookies left
theorem paco_salty_cookies_left : initial_salty_cookies - eaten_salty_cookies = 17 := 
 by
  -- Here we skip the proof by adding sorry
  sorry

end paco_salty_cookies_left_l585_585756


namespace constants_sum_l585_585779

theorem constants_sum (A B C D : ℕ) 
  (h : ∀ n : ℕ, n ≥ 4 → n^4 = A * (n.choose 4) + B * (n.choose 3) + C * (n.choose 2) + D * (n.choose 1)) 
  : A + B + C + D = 75 :=
by
  sorry

end constants_sum_l585_585779


namespace jake_fewer_peaches_l585_585414

theorem jake_fewer_peaches (steven_peaches : ℕ) (jake_peaches : ℕ) (h1 : steven_peaches = 19) (h2 : jake_peaches = 7) : steven_peaches - jake_peaches = 12 :=
sorry

end jake_fewer_peaches_l585_585414


namespace problem_solution_l585_585199

def neg_div_positive_result (a b : Int) (h : a > 0 ∧ b > 0) : (-a) / (-b) = a / b := by sorry

theorem problem_solution : (-180) / (-45) + (-9) = -5 := by
  have h1 : 180 > 0 := by norm_num
  have h2 : 45 > 0 := by norm_num
  have eq1 : (-180) / (-45) = 180 / 45 := neg_div_positive_result 180 45 ⟨h1, h2⟩
  rw [eq1]
  norm_num
  sorry

end problem_solution_l585_585199


namespace find_x_exp_4374_l585_585238

theorem find_x_exp_4374 (x : ℂ) (h : x - 1/x = complex.I * real.sqrt 3) :
  x^4374 - 1/x^4374 = -complex.I :=
sorry

end find_x_exp_4374_l585_585238


namespace problem1_problem2_l585_585266

-- Definition of the function
def f (x : ℝ) (k : ℝ) : ℝ := 2 ^ x + k * (2 ^ (-x))

-- Problem statement 1:
-- Prove that if f(x) is odd, then k = -1
theorem problem1 (k : ℝ) (h : ∀ x : ℝ, f (-x) k = -f x k) : k = -1 :=
  sorry

-- Problem statement 2:
-- Prove that if ∀ x ∈ [0, +∞), f(x) > 2^(-x), then k > 0
theorem problem2 (k : ℝ) (h : ∀ x : ℝ, 0 ≤ x → f x k > 2 ^ (-x)) : 0 < k :=
  sorry

end problem1_problem2_l585_585266


namespace tammy_speed_second_day_l585_585781

-- Definitions based on given conditions
def total_hours_climbed := 14
def total_distance := 52
def distance_day1 := 0.6 * total_distance
def distance_day2 := 0.4 * total_distance
def hours_day2 (hours_day1 : ℝ) := hours_day1 - 2
def speed_day2 (speed_day1 : ℝ) := speed_day1 + 0.5

-- The mathematical equivalent proof statement in Lean 4
theorem tammy_speed_second_day : 
  ∃ (hours_day1 : ℝ) (hours_day2 : ℝ) (speed_day1 : ℝ) (speed_day2 : ℝ),
  hours_day1 + hours_day2 = total_hours_climbed ∧
  hours_day2 = hours_day1 - 2 ∧
  speed_day2 = speed_day1 + 0.5 ∧
  distance_day1 = speed_day1 * hours_day1 ∧
  distance_day2 = speed_day2 * hours_day2 ∧
  speed_day2 = 4.4 :=
by
  sorry

end tammy_speed_second_day_l585_585781


namespace mixture_concentration_l585_585459

-- Definitions reflecting the given conditions
def sol1_concentration : ℝ := 0.30
def sol1_volume : ℝ := 8

def sol2_concentration : ℝ := 0.50
def sol2_volume : ℝ := 5

def sol3_concentration : ℝ := 0.70
def sol3_volume : ℝ := 7

-- The proof problem stating that the resulting concentration is 49%
theorem mixture_concentration :
  (sol1_concentration * sol1_volume + sol2_concentration * sol2_volume + sol3_concentration * sol3_volume) /
  (sol1_volume + sol2_volume + sol3_volume) * 100 = 49 :=
by
  sorry

end mixture_concentration_l585_585459


namespace binary_representation_413_l585_585077

theorem binary_representation_413 : nat.binary_repr 413 = "110011101" := by
  sorry

end binary_representation_413_l585_585077


namespace students_study_math_or_science_l585_585114

theorem students_study_math_or_science (A B m M : ℕ) 
  (h_total_students : 3000 = A + B - (A ∩ B))
  (h_math_students : 2100 ≤ A ∧ A ≤ 2250)
  (h_science_students : 1200 ≤ B ∧ B ≤ 1500) 
  (h_min_intersection : m = 300)
  (h_max_intersection : M = 750) :
  M - m = 450 :=
by sorry

end students_study_math_or_science_l585_585114


namespace james_profit_l585_585346

theorem james_profit :
  let toys_initial := 200
  let percent_sold := 0.80
  let buy_price := 20
  let sell_price := 30
  let toys_sold := percent_sold * toys_initial
  let total_revenue := toys_sold * sell_price
  let total_cost := toys_sold * buy_price
  let profit := total_revenue - total_cost
  in profit = 1600 :=
by
  let toys_initial := 200
  let percent_sold := 0.80
  let buy_price := 20
  let sell_price := 30
  let toys_sold := percent_sold * toys_initial
  let total_revenue := toys_sold * sell_price
  let total_cost := toys_sold * buy_price
  let profit := total_revenue - total_cost
  show profit = 1600
  sorry

end james_profit_l585_585346


namespace jack_loss_l585_585344

theorem jack_loss :
  let monthly_expense := 3 * 20
  let annual_expense := monthly_expense * 12
  let sell_price := 500
  annual_expense - sell_price = 220 :=
by {
  let monthly_expense := 3 * 20
  let annual_expense := monthly_expense * 12
  let sell_price := 500
  calc
    annual_expense - sell_price = (3 * 20 * 12) - 500 : by rfl
  ... = 720 - 500 : by rfl
  ... = 220 : by rfl
}

end jack_loss_l585_585344


namespace li_family_cinema_cost_l585_585702

theorem li_family_cinema_cost :
  let standard_ticket_price := 10
  let child_discount := 0.4
  let senior_discount := 0.3
  let handling_fee := 5
  let num_adults := 2
  let num_children := 1
  let num_seniors := 1
  let child_ticket_price := (1 - child_discount) * standard_ticket_price
  let senior_ticket_price := (1 - senior_discount) * standard_ticket_price
  let total_ticket_cost := num_adults * standard_ticket_price + num_children * child_ticket_price + num_seniors * senior_ticket_price
  let final_cost := total_ticket_cost + handling_fee
  final_cost = 38 :=
by
  sorry

end li_family_cinema_cost_l585_585702


namespace quadrilateral_properties_l585_585545

theorem quadrilateral_properties :
  ( (∀ (Q : Type) [quadrilateral Q], 
      (∀ a b c d : Q, (opposite_angles Q a b c d) → is_parallelogram Q a b c d)) ∧
    (∀ (Q : Type) [quadrilateral Q], 
      (∀ a b c d : Q, (perpendicular_diagonals Q a b c d) ↔ is_rhombus Q a b c d)) ∧
    (∀ (Q : Type) [square Q], 
      (is_rectangle Q ∧ is_rhombus Q)) ∧
    (∀ (Q : Type) [trapezoid Q], 
      (diagonals_bisect Q))) → 
  correct_option = "B" := 
sorry

end quadrilateral_properties_l585_585545


namespace problem_proof_l585_585082

theorem problem_proof (n : ℕ) 
  (h : ∃ k, 2 * k = n) :
  4 ∣ n :=
sorry

end problem_proof_l585_585082


namespace tan_22_5_decomposition_l585_585798

theorem tan_22_5_decomposition :
  ∃ (x y z w : ℕ), (tan (22.5 : ℝ) = real.sqrt x - real.sqrt y + real.sqrt z - w) ∧ x + y + z + w = 3 ∧ x ≥ y ∧ y ≥ z ∧ z ≥ w := 
by
  use [2, 0, 0, 1]
  split -- splitting to apply each conjunct
  sorry -- the proof would go here

end tan_22_5_decomposition_l585_585798


namespace curves_intersect_at_l585_585043

noncomputable def f (x : ℝ) : ℝ := 3 * x^2 - 4 * x + 2
noncomputable def g (x : ℝ) : ℝ := -x^3 + 9 * x^2 - 4 * x + 2

theorem curves_intersect_at :
  (∃ x : ℝ, f x = g x) ↔ ([(0, 2), (6, 86)] = [(0, 2), (6, 86)]) :=
by
  sorry

end curves_intersect_at_l585_585043


namespace rational_xyz_squared_l585_585619

theorem rational_xyz_squared
  (x y z : ℝ)
  (hx : ∃ r1 : ℚ, x + y * z = r1)
  (hy : ∃ r2 : ℚ, y + z * x = r2)
  (hz : ∃ r3 : ℚ, z + x * y = r3)
  (hxy : x^2 + y^2 = 1) :
  ∃ r4 : ℚ, x * y * z^2 = r4 := 
sorry

end rational_xyz_squared_l585_585619


namespace lottery_common_numbers_l585_585775

theorem lottery_common_numbers :
  ∃ (s : Finset ℕ), s.card = 4 ∧ ∀ (ticket : Finset ℕ), ticket ∈ tickets → ∃ (n : ℕ), n ∈ s ∧ n ∈ ticket :=
by
  -- ticket set and common number conditions
  let tickets : Finset (Finset ℕ) := sorry -- Assume tickets is defined properly as per conditions
  have common : ∀ t1 t2 ∈ tickets, t1 ≠ t2 → (t1 ∩ t2).nonempty := sorry -- Condition interpretation
  -- main proof statement: existence of 4 common numbers in each ticket
  sorry

end lottery_common_numbers_l585_585775


namespace increase_avg_grade_transfer_l585_585473

-- Definitions for initial conditions
def avg_grade_A_initial := 44.2
def avg_grade_B_initial := 38.8
def num_students_A_initial := 10
def num_students_B_initial := 10

def grade_kalinina := 41
def grade_sidorov := 44

-- Definitions for expected conditions after transfer
def sum_grades_A_initial := avg_grade_A_initial * num_students_A_initial
def sum_grades_B_initial := avg_grade_B_initial * num_students_B_initial

-- Verify the transfer condition will meet the requirements
theorem increase_avg_grade_transfer : 
  let sum_grades_A_after := sum_grades_A_initial - grade_kalinina - grade_sidorov in
  let sum_grades_B_after := sum_grades_B_initial + grade_kalinina + grade_sidorov in
  let num_students_A_after := num_students_A_initial - 2 in
  let num_students_B_after := num_students_B_initial + 2 in
  let avg_grade_A_after := sum_grades_A_after / num_students_A_after in
  let avg_grade_B_after := sum_grades_B_after / num_students_B_after in

  avg_grade_A_after > avg_grade_A_initial ∧ avg_grade_B_after > avg_grade_B_initial :=
by
  sorry

end increase_avg_grade_transfer_l585_585473


namespace find_constant_l585_585015

theorem find_constant (x1 x2 : ℝ) (C : ℝ) :
  x1 - x2 = 5.5 ∧
  x1 + x2 = -5 / 2 ∧
  x1 * x2 = C / 2 →
  C = -12 :=
by
  -- proof goes here
  sorry

end find_constant_l585_585015


namespace space_correct_propositions_l585_585328

-- Definition of each individual proposition as provided in the conditions
def prop1 : Prop := ∀ {l₁ l₂ l₃ : ℚ × ℚ × ℚ}, (l₁ ∥ l₃ ∧ l₂ ∥ l₃) → l₁ ∥ l₂
def prop2 : Prop := ∀ {l₁ l₂ l₃ : ℚ × ℚ × ℚ}, (l₁ ⊥ l₃ ∧ l₂ ⊥ l₃) → l₁ ∥ l₂
def prop3 : Prop := ∀ {l₁ l₂ : ℚ × ℚ × ℚ}, {P : set (ℚ × ℚ × ℚ)}, (l₁ ∥ P ∧ l₂ ∥ P) → l₁ ∥ l₂
def prop4 : Prop := ∀ {l₁ l₂ : ℚ × ℚ × ℚ}, {P : set (ℚ × ℚ × ℚ)}, (l₁ ⊥ P ∧ l₂ ⊥ P) → l₁ ∥ l₂

-- The mathematical proof problem in Lean 4 statement
theorem space_correct_propositions :
  (prop1 ∧ prop2 ∧ ¬prop3 ∧ ¬prop4) ∨ (prop1 ∧ ¬prop2 ∧ prop3 ∧ ¬prop4) ∨
  (prop1 ∧ ¬prop2 ∧ ¬prop3 ∧ prop4) ∨ (¬prop1 ∧ prop2 ∧ prop3 ∧ ¬prop4) ∨ 
  (¬prop1 ∧ prop2 ∧ ¬prop3 ∧ prop4) ∨ (¬prop1 ∧ ¬prop2 ∧ prop3 ∧ prop4) ∨
  (¬prop1 ∧ ¬prop2 ∧ ¬prop3 ∧ ¬prop4) :=
sorry

end space_correct_propositions_l585_585328


namespace decreasing_f_on_domain_max_min_values_of_f_l585_585236

open Function

def f (x : ℝ) : ℝ := 1 / (x - 1)

theorem decreasing_f_on_domain : ∀ (x₁ x₂ : ℝ), 2 ≤ x₁ → x₁ < x₂ → x₂ ≤ 6 → f x₁ > f x₂ :=
by
sorry

theorem max_min_values_of_f : 
  (∃ x₁ x₂, f x₁ = 1 ∧ 2 ≤ x₁ ∧ x₁ ≤ 6 ∧ f x₂ = 1 / 5 ∧ 2 ≤ x₂ ∧ x₂ ≤ 6) :=
by
sorry

end decreasing_f_on_domain_max_min_values_of_f_l585_585236


namespace find_p_q_sum_l585_585203

theorem find_p_q_sum (p q : ℕ) (a : ℚ) :
  a = p / q ∧ p > 0 ∧ q > 0 ∧ Nat.gcd p q = 1 ∧ 
  (Σ' x : ℝ, (⌊x⌋ * (x - ⌊x⌋) = a * x^2) : ℝ) = 272 → 
  p + q = 136 :=
sorry

end find_p_q_sum_l585_585203


namespace one_inch_cubes_with_red_paint_at_least_two_faces_l585_585027

theorem one_inch_cubes_with_red_paint_at_least_two_faces
  (number_of_one_inch_cubes : ℕ)
  (cubes_with_three_faces : ℕ)
  (cubes_with_two_faces : ℕ)
  (total_cubes_with_at_least_two_faces : ℕ) :
  number_of_one_inch_cubes = 64 →
  cubes_with_three_faces = 8 →
  cubes_with_two_faces = 24 →
  total_cubes_with_at_least_two_faces = cubes_with_three_faces + cubes_with_two_faces →
  total_cubes_with_at_least_two_faces = 32 :=
by
  sorry

end one_inch_cubes_with_red_paint_at_least_two_faces_l585_585027


namespace tetrahedron_prob_max_min_tetrahedron_prob_t_ge_4_l585_585146

theorem tetrahedron_prob_max_min :
  let x : Finset ℕ := {1, 2, 3, 4}
  let t (x1 x2 : ℕ) : ℕ := (x1 - 3)^2 + (x2 - 3)^2
  let max_t := 8
  let min_t := 0
  (x.card * x.card = 16) →
  (∃ a b, t a b = max_t ∧ (1 / 16 : ℚ)) ∧ (∃ a b, t a b = min_t ∧ (1 / 16 : ℚ))
:= sorry

theorem tetrahedron_prob_t_ge_4 :
  let x : Finset ℕ := {1, 2, 3, 4}
  let t (x1 x2 : ℕ) : ℕ := (x1 - 3)^2 + (x2 - 3)^2
  (x.card * x.card = 16) →
  (x1 x2 a b,
    t x1 x2 ≥ 4 → 
    ((t x1 x2 = 5 → (1 / 4 : ℚ)) ∧ (t x1 x2 = 8 → (1 / 16 : ℚ)) → ∑ 2 ≥ 4 → 
    ((1 / 4) + (1 / 16)) = (5 / 16))
:= sorry

end tetrahedron_prob_max_min_tetrahedron_prob_t_ge_4_l585_585146


namespace five_digit_numbers_divisible_by_3_and_5_l585_585922

theorem five_digit_numbers_divisible_by_3_and_5 : 
  ∀ digits : Finset ℕ, 
    digits = {0, 1, 2, 3, 4, 5} →
    ∃ ways : ℕ, 
      (ways = 48) ∧ 
      (∀ n, n ∈ ways →
        (n ≥ 10000 ∧ n < 100000) ∧ 
        (∃ d, d ∈ digits ∧ d > 0 ∧ NineDigits.indexOf d = d) ∧ 
        ((n % 3 = 0) ∧ (n % 5 = 0)) ∧ 
        (∀ d ∈ digits, NineDigits.indexOf d < 6)) :=
by 
  sorry

end five_digit_numbers_divisible_by_3_and_5_l585_585922


namespace candy_eating_l585_585177

-- Definitions based on the conditions
def candies (andrey_rate boris_rate denis_rate : ℕ) (andrey_candies boris_candies denis_candies total_candies : ℕ) : Prop :=
  andrey_rate = 4 ∧ boris_rate = 3 ∧ denis_rate = 7 ∧ andrey_candies = 24 ∧ boris_candies = 18 ∧ denis_candies = 28 ∧
  total_candies = andrey_candies + boris_candies + denis_candies

-- Problem statement
theorem candy_eating :
  ∃ (a b d : ℕ), 
    candies 4 3 7 a b d 70 :=
sorry

end candy_eating_l585_585177


namespace internet_bill_amount_l585_585776

def electricity_bill := 60
def gas_bill := 40
def gas_paid_fraction := 3/4
def additional_gas_payment := 5
def water_bill := 40
def water_paid_fraction := 1/2
def internet_payment_per_time := 5
def internet_payment_times := 4
def total_remaining_payment := 30

theorem internet_bill_amount:
  let gas_remaining := gas_bill - (gas_bill * gas_paid_fraction) - additional_gas_payment in
  let water_remaining := water_bill - (water_bill * water_paid_fraction) in
  let already_paid_internet := internet_payment_per_time * internet_payment_times in
  let remaining_excluding_internet := gas_remaining + water_remaining in
  let remaining_internet := total_remaining_payment - remaining_excluding_internet in
  let total_internet := already_paid_internet + remaining_internet in
  total_internet = 25 :=
by
  let gas_remaining := gas_bill - (gas_bill * gas_paid_fraction) - additional_gas_payment
  let water_remaining := water_bill - (water_bill * water_paid_fraction)
  let already_paid_internet := internet_payment_per_time * internet_payment_times
  let remaining_excluding_internet := gas_remaining + water_remaining
  let remaining_internet := total_remaining_payment - remaining_excluding_internet
  let total_internet := already_paid_internet + remaining_internet
  show total_internet = 25 from sorry

end internet_bill_amount_l585_585776


namespace jon_and_mary_frosting_l585_585719

-- Jon frosts a cupcake every 40 seconds
def jon_frost_rate : ℚ := 1 / 40

-- Mary frosts a cupcake every 24 seconds
def mary_frost_rate : ℚ := 1 / 24

-- Combined frosting rate of Jon and Mary
def combined_frost_rate : ℚ := jon_frost_rate + mary_frost_rate

-- Total time in seconds for 12 minutes
def total_time_seconds : ℕ := 12 * 60

-- Calculate the total number of cupcakes frosted in 12 minutes
def total_cupcakes_frosted (time_seconds : ℕ) (rate : ℚ) : ℚ :=
  time_seconds * rate

theorem jon_and_mary_frosting : total_cupcakes_frosted total_time_seconds combined_frost_rate = 48 := by
  sorry

end jon_and_mary_frosting_l585_585719


namespace find_number_l585_585301

theorem find_number : ∃ x : ℝ, 0.35 * x = 0.15 * 40 ∧ x = 120 / 7 :=
by
  sorry

end find_number_l585_585301


namespace find_coordinates_of_P_l585_585335

-- Define the parametric equation of circle C1
def circle_C1 (ϕ : ℝ) : ℝ × ℝ := (1 + cos ϕ, 2 + sin ϕ)

-- Define the polar coordinate equation of line C2
def line_C2 (ρ θ : ℝ) : Prop := ρ * cos θ + 2 = 0

-- Define the polar coordinate equation of line C3
def line_C3 (ρ θ : ℝ) : Prop := θ = π / 4

-- Define the rectangular equation of C1
def rect_circle_C1 : Prop := ∀ x y : ℝ, (x = 1 + cos ϕ) ∧ (y = 2 + sin ϕ) → (x - 1)^2 + (y - 2)^2 = 1

-- Define the polar equation of C1
def polar_circle_C1 : Prop := ∀ (ρ θ : ℝ), ρ^2 - 2*ρ*cos θ - 4*ρ*sin θ + 4 = 0

-- Define the rectangular equation of C2
def rect_line_C2 : Prop := ∀ x y : ℝ, line_C2 x y → x = -2

-- Define the conditions for area of triangle PMN
def triangle_area_condition : Prop :=
  let M : ℝ × ℝ := (some_value_for_M) in -- Need proper values for M
  let N : ℝ × ℝ := (some_value_for_N) in -- Need proper values for N
  let P : ℝ × ℝ := (some_value_for_P) in -- Need proper values for P
  1/2 * abs (some_value_for_area_PMN) = 1

-- Main proof structure
theorem find_coordinates_of_P :
  (∃ P : ℝ × ℝ, 
    (line_C2 P.1 P.2) ∧ 
    triangle_area_condition P = 1 ∧
    (P = (-2, 0) ∨ P = (-2, -4))) :=
sorry

end find_coordinates_of_P_l585_585335


namespace geometry_problem_l585_585555

open EuclideanGeometry

-- Define the main geometry problem
universe u
variable {α : Type u} [euclidean_geometry α]

variables {A B C D P Q R K E L F : α}

-- Specify the given conditions
variable (h1 : triangle A B C)
variable (h2 : dist B A > dist C A)
variable (h3 : ∃ D, is_angle_bisector A B C D)
variable (h4 : line_collinear P D A)
variable (h5 : circle (circumcircle A B D) Q ∧ tangent_to_circle Q P (circumcircle A B D) ∧ same_side P A B)
variable (h6 : circle (circumcircle A C D) R ∧ tangent_to_circle R P (circumcircle A C D) ∧ same_side P A C)
variable (h7 : point K ∧ line_intersects K B R ∧ line_intersects K C Q)
variable (h8 : line_parallel K E Q ∧ line_parallel K L A ∧ line_parallel K F R)

-- The final theorem to prove
theorem geometry_problem : dist E L = dist F K :=
sorry

end geometry_problem_l585_585555


namespace sum_of_y_for_f_l585_585365

noncomputable def f (x : ℝ) : ℝ := x^3 - 2 * x + 5

theorem sum_of_y_for_f (y1 y2 y3 : ℝ) :
  (∀ y, 64 * y^3 - 8 * y + 5 = 7) →
  y1 + y2 + y3 = 0 :=
by
  -- placeholder for actual proof
  sorry

end sum_of_y_for_f_l585_585365


namespace func1_satisfies_equation_func2_satisfies_equation_find_functions_l585_585589

/-- A function that satisfies the given functional equation -/
def candidate_function (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f(x) * f(y) + f(x + y) = x * y

/-- A function that equals x - 1 -/
def func1 (x : ℝ) : ℝ := x - 1

/-- A function that equals -x - 1 -/
def func2 (x : ℝ) : ℝ := -x - 1

/-- Proof that x - 1 satisfies the functional equation -/
theorem func1_satisfies_equation : candidate_function func1 := by
  intros x y
  unfold func1
  sorry

/-- Proof that -x - 1 satisfies the functional equation -/
theorem func2_satisfies_equation : candidate_function func2 := by
  intros x y
  unfold func2
  sorry

/-- Theorems are proved showing that the only functions satisfying the condition are x - 1 and -x - 1 -/
theorem find_functions : ∀ f : ℝ → ℝ, candidate_function f ↔ f = func1 ∨ f = func2 := by
  intro f
  split
  {
    intro h
    sorry -- proof that candidate_function f implies f = func1 or f = func2
  }
  {
    intro h
    cases h
    {
      rw h
      exact func1_satisfies_equation
    }
    {
      rw h
      exact func2_satisfies_equation
    }
  }

end func1_satisfies_equation_func2_satisfies_equation_find_functions_l585_585589


namespace prove_ellipse_equation_l585_585085

variables {a b c : ℝ}
variables {E : ℝ → ℝ → Prop} -- Define the ellipse

-- Conditions
def ellipse_condition_1 : Prop := a > b ∧ b > 0
def ellipse_condition_2 : Prop := ∀ F1 F2 : ℝ, |F1 - F2| = 4
def point_M_condition_1 (M F2 : ℝ) : Prop := (M - F2).orthogonal (0, 1)
def point_M_condition_2 (M F1 F2 : ℝ) : Prop := dist M F1 = 3 * dist M F2

-- The equation of the ellipse to be proven
def ellipse_equation (x y : ℝ) : Prop := x^2 / 8 + y^2 / 4 = 1

-- The full problem statement
theorem prove_ellipse_equation (F1 F2 M : ℝ) :
  ellipse_condition_1 →
  ellipse_condition_2 F1 F2 →
  point_M_condition_1 M F2 →
  point_M_condition_2 M F1 F2 →
  (∀ x y, E x y ↔ ellipse_equation x y) :=
by
  intros
  sorry

end prove_ellipse_equation_l585_585085


namespace find_a_6_l585_585241

noncomputable def seq : ℕ → ℝ
| 1     := 7
| (n+1) := 1/2 * seq n + 3

theorem find_a_6 : seq 6 = 193 / 32 :=
by {
  sorry
}

end find_a_6_l585_585241


namespace greatest_integer_value_of_third_side_l585_585046

def sides_of_triangle (a b : ℕ) (c : ℝ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

theorem greatest_integer_value_of_third_side :
  let s := 15 in
  sides_of_triangle 6 10 s ∧ (4 < s ∧ s < 16) :=
by
  sorry

end greatest_integer_value_of_third_side_l585_585046


namespace minimum_flips_l585_585774

-- Definitions for the problem
def consecutive_adjacent (grid : matrix (fin 4) (fin 4) ℕ) : Prop :=
  ∀ i j, match grid i j with
  | n =>
    (∃ i' j', (abs (i - i') + abs (j - j') = 1) ∧ (grid i' j' = n + 1)) ∧
    (∃ i' j', (abs (i - i') + abs (j - j') = 1) ∧ (grid i' j' = n - 1))

-- The theorem statement to prove
theorem minimum_flips (grid : matrix (fin 4) (fin 4) ℕ) :
  consecutive_adjacent grid → true := -- here true is placeholder for the actual implication we need to formalize.
  sorry

end minimum_flips_l585_585774


namespace total_value_of_alice_money_l585_585149

theorem total_value_of_alice_money (quarters nickels iron_nickels regular_nickels : ℕ)
  (value_quarter : ℕ) (value_nickel_cents : ℕ) (value_iron_nickel_dollars : ℝ)
  (value_reg_nickel_dollars : ℝ) (percentage_iron_nickels : ℝ)
  (hvq : value_quarter = 25)
  (hvn : value_nickel_cents = 5)
  (hvind : value_iron_nickel_dollars = 3)
  (hvnd : value_reg_nickel_dollars = 5 / 100)
  (hpio : percentage_iron_nickels = 0.2)
  (hq : quarters = 20)
  (hn : nickels = (quarters * value_quarter) / value_nickel_cents)
  (hion : iron_nickels = ℕ.floor (percentage_iron_nickels * nickels))
  (hrn : regular_nickels = nickels - iron_nickels) :
  (iron_nickels * value_iron_nickel_dollars + regular_nickels * value_reg_nickel_dollars) = 64 := sorry

end total_value_of_alice_money_l585_585149


namespace sum_of_possible_lengths_l585_585117

theorem sum_of_possible_lengths
  (m : ℕ) 
  (h1 : m < 18)
  (h2 : m > 4) : ∑ i in (Finset.range 13).map (λ x, x + 5) = 143 := by
sorry

end sum_of_possible_lengths_l585_585117


namespace spent_on_video_game_l585_585718

def saved_September : ℕ := 30
def saved_October : ℕ := 49
def saved_November : ℕ := 46
def money_left : ℕ := 67
def total_saved := saved_September + saved_October + saved_November

theorem spent_on_video_game : total_saved - money_left = 58 := by
  -- proof steps go here
  sorry

end spent_on_video_game_l585_585718


namespace angle_sum_around_point_l585_585850

theorem angle_sum_around_point (x : ℝ) (h : 2 * x + 140 = 360) : x = 110 := 
  sorry

end angle_sum_around_point_l585_585850


namespace new_average_of_subtracted_elements_l585_585419

theorem new_average_of_subtracted_elements (a b c d e : ℝ) 
  (h_average : (a + b + c + d + e) / 5 = 5) 
  (new_a : ℝ := a - 2) 
  (new_b : ℝ := b - 2) 
  (new_c : ℝ := c - 2) 
  (new_d : ℝ := d - 2) :
  (new_a + new_b + new_c + new_d + e) / 5 = 3.4 := 
by 
  sorry

end new_average_of_subtracted_elements_l585_585419


namespace cos_30_eq_l585_585869

theorem cos_30_eq : Real.cos (30 * Real.pi / 180) = sqrt 3 / 2 := by
  sorry

end cos_30_eq_l585_585869


namespace rolling_semicircle_path_length_l585_585183

theorem rolling_semicircle_path_length (BD : ℝ) (hBD : BD = 4 / π) : 
  let total_path_length := 8 in total_path_length = 8 :=
by
  sorry

end rolling_semicircle_path_length_l585_585183


namespace angle_opposite_side_c_l585_585692

theorem angle_opposite_side_c (a b c : ℕ) (h : (a + b + c) * (a + b - c) = 3 * a * b) : 
  ∃ θ : ℝ, θ = 60 ∧ cos θ = 1/2 := 
by
  sorry

end angle_opposite_side_c_l585_585692


namespace planting_methods_with_conditions_l585_585401

noncomputable def number_of_planting_methods : ℕ := 34

theorem planting_methods_with_conditions :
  ∃ (count : ℕ), count = number_of_planting_methods ∧
  (∀ (lst : list bool), lst.length = 7 → 
    (∀ i, i < 6 → lst[i] = ff ∨ lst[i + 1] = ff) → 
    list.count lst tt <= 4 ∧ list.count lst ff <= 7) :=
by
  sorry

end planting_methods_with_conditions_l585_585401


namespace increase_avg_grade_transfer_l585_585472

-- Definitions for initial conditions
def avg_grade_A_initial := 44.2
def avg_grade_B_initial := 38.8
def num_students_A_initial := 10
def num_students_B_initial := 10

def grade_kalinina := 41
def grade_sidorov := 44

-- Definitions for expected conditions after transfer
def sum_grades_A_initial := avg_grade_A_initial * num_students_A_initial
def sum_grades_B_initial := avg_grade_B_initial * num_students_B_initial

-- Verify the transfer condition will meet the requirements
theorem increase_avg_grade_transfer : 
  let sum_grades_A_after := sum_grades_A_initial - grade_kalinina - grade_sidorov in
  let sum_grades_B_after := sum_grades_B_initial + grade_kalinina + grade_sidorov in
  let num_students_A_after := num_students_A_initial - 2 in
  let num_students_B_after := num_students_B_initial + 2 in
  let avg_grade_A_after := sum_grades_A_after / num_students_A_after in
  let avg_grade_B_after := sum_grades_B_after / num_students_B_after in

  avg_grade_A_after > avg_grade_A_initial ∧ avg_grade_B_after > avg_grade_B_initial :=
by
  sorry

end increase_avg_grade_transfer_l585_585472


namespace find_values_of_x_and_y_l585_585063

variables (a b c d e f : ℚ)

-- Assumptions
axiom h1 : a ≠ b
axiom h2 : b ≠ 0
axiom h3 : (a + x) / (b + y) = c / d
axiom h4 : (a + 2 * x) / (b + 2 * y) = e / f

noncomputable def find_x : ℚ := (a * d - b * c) / (d - c)
noncomputable def find_y : ℚ := (a * f - b * e) / (f - e)

theorem find_values_of_x_and_y : x = find_x a b c d ∧ y = find_y a b e f :=
sorry

end find_values_of_x_and_y_l585_585063


namespace volume_of_box_l585_585880

theorem volume_of_box (L W S : ℕ) (hL : L = 48) (hW : W = 36) (hS : S = 8) : 
  let new_length := L - 2 * S,
      new_width := W - 2 * S,
      height := S
  in new_length * new_width * height = 5120 := by
sorry

end volume_of_box_l585_585880


namespace smallest_four_digit_multiple_of_18_l585_585970

theorem smallest_four_digit_multiple_of_18 : ∃ n : ℕ, 1000 ≤ n ∧ n ≤ 9999 ∧ n % 18 = 0 ∧ 
  ∀ m : ℕ, 1000 ≤ m ∧ m ≤ 9999 ∧ m % 18 = 0 → n ≤ m :=
begin
  use 1008,
  split,
  { exact nat.le_refl 1008, },
  split,
  { exact nat.le_succ 9999, },
  split,
  { exact nat.mod_eq_zero_of_divisible 1008 18 sorry, },
  { intros m h1 h2 h3,
    apply nat.le_of_sub_nonneg,
    sorry, }
end

end smallest_four_digit_multiple_of_18_l585_585970


namespace sum_of_possible_m_values_l585_585128

theorem sum_of_possible_m_values :
  let m_range := Finset.Icc 5 17 in
  m_range.sum id = 143 := by
  sorry

end sum_of_possible_m_values_l585_585128


namespace ratio_of_perimeters_l585_585047

theorem ratio_of_perimeters (s S : ℝ) 
  (h1 : S = 3 * s) : 
  (4 * S) / (4 * s) = 3 :=
by
  sorry

end ratio_of_perimeters_l585_585047


namespace dart_prob_center_square_l585_585875

noncomputable def hexagon_prob (s : ℝ) : ℝ :=
  let square_area := s^2
  let hexagon_area := (3 * Real.sqrt 3 / 2) * s^2
  square_area / hexagon_area

theorem dart_prob_center_square (s : ℝ) : hexagon_prob s = 2 * Real.sqrt 3 / 9 :=
by
  -- Proof omitted
  sorry

end dart_prob_center_square_l585_585875


namespace trigonometric_inequality_l585_585398

theorem trigonometric_inequality (x : Real) (h1 : 0 < x) (h2 : x < (3 * Real.pi) / 8) :
  (1 / Real.sin (x / 3) + 1 / Real.sin (8 * x / 3) > (Real.sin (3 * x / 2)) / (Real.sin (x / 2) * Real.sin (2 * x))) :=
  by
  sorry

end trigonometric_inequality_l585_585398


namespace candies_eaten_l585_585176

theorem candies_eaten (A B D : ℕ) 
                      (h1 : 4 * B = 3 * A) 
                      (h2 : 7 * A = 6 * D) 
                      (h3 : A + B + D = 70) :
  A = 24 ∧ B = 18 ∧ D = 28 := 
by
  sorry

end candies_eaten_l585_585176


namespace total_surfers_calculation_l585_585814

def surfers_on_malibu_beach (m_sm : ℕ) (s_sm : ℕ) : ℕ := 2 * s_sm

def total_surfers (m_sm s_sm : ℕ) : ℕ := m_sm + s_sm

theorem total_surfers_calculation : total_surfers (surfers_on_malibu_beach 20 20) 20 = 60 := by
  sorry

end total_surfers_calculation_l585_585814


namespace pages_left_l585_585898

variable (a b : ℕ)

theorem pages_left (a b : ℕ) : a - 8 * b = a - 8 * b :=
by
  sorry

end pages_left_l585_585898


namespace roots_of_quadratic_eq_l585_585252

theorem roots_of_quadratic_eq:
  (8 * γ^3 + 15 * δ^2 = 179) ↔ (γ^2 - 3 * γ + 1 = 0 ∧ δ^2 - 3 * δ + 1 = 0) :=
sorry

end roots_of_quadratic_eq_l585_585252


namespace candies_eaten_l585_585153

variables (A B D : ℕ)

-- Conditions:
def condition1 : Prop := ∃ k1 k2 k3 : ℕ, k1 * 4 + k2 * 3 + k3 * 7 = 70
def condition2 : Prop := (B * 3 = A * 4) ∧ (D * 7 = A * 6)
def condition3 : Prop := A + B + D = 70

-- Theorem statement:
theorem candies_eaten (h1 : condition1) (h2 : condition2) (h3 : condition3) :
    A = 24 ∧ B = 18 ∧ D = 28 := sorry

end candies_eaten_l585_585153


namespace exchange_rate_l585_585091

theorem exchange_rate :
  (4500 / 3900 : ℝ) * 3000 ≈ 3461.54 :=
by sorry

end exchange_rate_l585_585091


namespace airplane_seat_count_l585_585547

noncomputable def total_seats_in_airplane (n : ℝ) : ℝ :=
  36 + 0.30 * n + 0.60 * n

theorem airplane_seat_count : ∃ n : ℝ, total_seats_in_airplane n = n ∧ n = 360 :=
by
  use 360
  split
  . sorry
  . sorry

end airplane_seat_count_l585_585547


namespace polynomial_divisibility_l585_585230

theorem polynomial_divisibility (r s : ℝ) :
  (∀ x, 10 * x^4 - 15 * x^3 - 55 * x^2 + 85 * x - 51 = 10 * (x - r)^2 * (x - s)) →
  r = 3 / 2 ∧ s = -5 / 2 :=
by
  intros h
  sorry

end polynomial_divisibility_l585_585230


namespace transfer_students_increase_average_l585_585471

theorem transfer_students_increase_average
    (avgA : ℚ) (numA : ℕ) (avgB : ℚ) (numB : ℕ)
    (kalinina_grade : ℚ) (sidorova_grade : ℚ)
    (avgA_init : avgA = 44.2) (numA_init : numA = 10)
    (avgB_init : avgB = 38.8) (numB_init : numB = 10)
    (kalinina_init : kalinina_grade = 41)
    (sidorova_init : sidorova_grade = 44) : 
    let new_avg_B_k := (avgB * numB + kalinina_grade) / (numB + 1) in
    let new_avg_A_s := (avgA * numA - sidorova_grade) / (numA - 1) in
    let new_avg_A_both := (avgA * numA - kalinina_grade - sidorova_grade) / (numA - 2) in
    let new_avg_B_both := (avgB * numB + kalinina_grade + sidorova_grade) / (numB + 2) in
    (new_avg_B_k <= avgB) ∧ (new_avg_A_s <= avgA) ∧ (new_avg_A_both > avgA) ∧ (new_avg_B_both > avgB) :=
by
  sorry

end transfer_students_increase_average_l585_585471


namespace find_angle_AO₂B_l585_585423

-- Given conditions as definitions
variables {R r : ℝ} (O₁ O₂ A B : Type) [MetricSpace O₁] [MetricSpace O₂] [MetricSpace A] [MetricSpace B]
variable (ratio : R / r = Real.sqrt 2)
variable (angle_AO₁B : ∠ A O₁ B = 60)
variable (chord_AB : Chord O₁ O₂ A B)

-- Required statement
theorem find_angle_AO₂B : ∠ A O₂ B = 90 :=
by 
  sorry

end find_angle_AO₂B_l585_585423


namespace fifteenth_prime_number_l585_585586

-- Define the 7th prime number
def seventh_prime : ℕ := 17

-- State that the 15th prime number is 47 given that the 7th prime number is 17
theorem fifteenth_prime_number (h : seventh_prime = 17) : Nat.prime_number 15 = 47 := 
by
  sorry

end fifteenth_prime_number_l585_585586


namespace tan_pi9_2pi9_4pi9_l585_585068

theorem tan_pi9_2pi9_4pi9 :
  tan (Real.pi / 9) * tan (2 * Real.pi / 9) * tan (4 * Real.pi / 9) = Real.sqrt 3 :=
by
  sorry

end tan_pi9_2pi9_4pi9_l585_585068


namespace relationship_ab_c_l585_585235

noncomputable def a : ℝ := (3 / 5) ^ (1 / 3)
noncomputable def b : ℝ := (3 / 5) ^ (-1 / 3)
noncomputable def c : ℝ := (2 / 5) ^ (1 / 3)

theorem relationship_ab_c : b > a ∧ a > c :=
by
  sorry

end relationship_ab_c_l585_585235


namespace log_identity_l585_585993

theorem log_identity (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : log y (x ^ 4) = 6) : log x (y ^ 3) = 1 / 2 := 
by
  sorry

end log_identity_l585_585993


namespace max_ones_in_distinct_product_table_l585_585687

theorem max_ones_in_distinct_product_table :
  ∀ (table : Matrix (Fin 3) (Fin 3) ℕ), 
    (∀ i j k, table i j ≠ table i k) ∧
    (∀ i j k, table j i ≠ table k i) →
    (card (set_of (λ x, x = 1) (table.map id) : set ℕ) ≤ 5) :=
by sorry

end max_ones_in_distinct_product_table_l585_585687


namespace find_f_l585_585377

noncomputable def f : ℝ → ℝ := sorry

theorem find_f (f : ℝ → ℝ) 
  (h1 : continuous f) 
  (h2 : monotone f) 
  (h3 : f 0 = 1) 
  (h4 : ∀ x y : ℝ, f (x + y) ≥ f x * f y - f (x * y) + 1) : 
  (∀ x : ℝ, f x = x + 1) := 
sorry

end find_f_l585_585377


namespace fifteenth_entry_condition_satisfied_l585_585228

def r_7 (n : ℕ) : ℕ := n % 7

theorem fifteenth_entry_condition_satisfied (n : ℕ) : (∃ k : ℕ, 
    List.nthLe (List.filter (λ n, r_7 (3 * n) ≤ 3) (List.range (k + 1))) 14  sorry = 22) :=
sorry

end fifteenth_entry_condition_satisfied_l585_585228


namespace smallest_four_digit_multiple_of_18_l585_585978

theorem smallest_four_digit_multiple_of_18 : 
  ∃ (n : ℕ), 999 < n ∧ n < 10000 ∧ 18 ∣ n ∧ ∀ m : ℕ, 999 < m ∧ m < 10000 ∧ 18 ∣ m → n ≤ m ∧ n = 1008 := 
sorry

end smallest_four_digit_multiple_of_18_l585_585978


namespace root5_inequality_l585_585481

theorem root5_inequality (x y : ℝ) (h : x < y) : (sqrt x ^ (1 / 5)) > (sqrt y ^ (1 / 5)) :=
by
  sorry

end root5_inequality_l585_585481


namespace divide_into_groups_l585_585910

def sequence := list bool
def is_valid_sequence (s : sequence) : Prop := s.count (= true) = 1011 ∧ s.count (= false) = 1011
def compatible (s1 s2 : sequence) : Prop := list.zip_with (λ b1 b2 => b1 = b2) s1 s2 |>.count true = 4

theorem divide_into_groups (L : list sequence) (h : ∀ s ∈ L, is_valid_sequence s) :
  ∃ (G : fin 20 → list sequence), 
  (∀ i : fin 20, ∀ s1 s2 ∈ G i, ¬ compatible s1 s2) ∧ 
  (∀ s ∈ L, ∃ i : fin 20, s ∈ G i) :=
sorry

end divide_into_groups_l585_585910


namespace frog_probability_l585_585104

-- Define the boundary conditions
def boundary_conditions (x y : ℕ) : ℚ :=
  if (y = 0 ∨ y = 5) ∧ 0 ≤ x ∧ x ≤ 5 then 1 -- Horizontal edges
  else if x = 0 ∨ x = 5 then 0 -- Vertical edges
  else P (x, y)

-- Recursive definition for P (probability of ending up on a horizontal edge)
noncomputable def P : ℕ × ℕ → ℚ
| (2, 3) := 1/4 * P (1, 3) + 1/4 * P (3, 3) + 1/4 * P (2, 2) + 1/4 * P (2, 4)
| (2, 2) := 1/4 * P (1, 2) + 1/4 * P (3, 2) + 1/4 * P (2, 1) + 1/4 * P (2, 3)
| p := boundary_conditions p.1 p.2

-- Formal statement of the problem
theorem frog_probability :
  P (2, 3) = 1/2 :=
sorry

end frog_probability_l585_585104


namespace construct_triangle_with_altitudes_l585_585660

-- Define the given line segments m_a and m_b
variables (m_a m_b : ℝ) (h_mab : m_a <= m_b)

-- Define a function that checks the existence of a triangle with given altitudes
def exists_triangle_with_altitudes (h : ∃ (t : ℝ) (a b c : ℝ),
    t > 0 ∧
    a = 2 * t / m_a ∧
    b = 2 * t / m_b ∧
    c = 2 * t / (m_a + m_b)) : Prop := 
    ∃ (a b c : ℝ), a * b * c = t^2 / 4 ∧ 
    Real.dist a b < c ∧ Real.dist a c < b ∧ Real.dist b c < a

-- Main theorem statement
theorem construct_triangle_with_altitudes :
  ∃ (a b c : ℝ),
    ∃ (t : ℝ), t > 0 ∧
    a = 2 * t / m_a ∧
    b = 2 * t / m_b ∧
    c = 2 * t / (m_a + m_b) ∧
    a * b * c = t^2 / 4 ∧
    Real.dist a b < c ∧ Real.dist a c < b ∧ Real.dist b c < a := 
sorry

end construct_triangle_with_altitudes_l585_585660


namespace fourth_vertex_of_tetrahedron_l585_585896

-- Define the given vertices
def A : ℝ × ℝ × ℝ := (0, 1, 2)
def B : ℝ × ℝ × ℝ := (4, 2, 1)
def C : ℝ × ℝ × ℝ := (3, 1, 4)

-- Define the distance squared
def dist_squared (P Q : ℝ × ℝ × ℝ) : ℝ :=
  (P.1 - Q.1) ^ 2 + (P.2 - Q.2) ^ 2 + (P.3 - Q.3) ^ 2

-- Define the problem to prove:
theorem fourth_vertex_of_tetrahedron :
  ∃ (x y z : ℝ), x ∈ ℤ ∧ y ∈ ℤ ∧ z ∈ ℤ ∧ 
    dist_squared (x, y, z) A = 18 ∧ 
    dist_squared (x, y, z) B = 18 ∧ 
    dist_squared (x, y, z) C = 18 ∧ 
    (x, y, z) = (3, -2, 2) :=
sorry

end fourth_vertex_of_tetrahedron_l585_585896


namespace hexagon_area_of_circle_l585_585844

noncomputable def radius (area : ℝ) : ℝ :=
  Real.sqrt (area / Real.pi)

noncomputable def area_equilateral_triangle (s : ℝ) : ℝ :=
  (s * s * Real.sqrt 3) / 4

theorem hexagon_area_of_circle {r : ℝ} (h : π * r^2 = 100 * π) :
  6 * area_equilateral_triangle r = 150 * Real.sqrt 3 :=
by
  sorry

end hexagon_area_of_circle_l585_585844


namespace arithmetic_sequence_20th_term_l585_585205

theorem arithmetic_sequence_20th_term (a₁ : ℕ) (d : ℕ) (n : ℕ) (h1 : a₁ = 2) (h2 : d = 3) (h3 : n = 20) :
  a₁ + (n - 1) * d = 59 :=
by
  rw [h1, h2, h3]
  sorry

end arithmetic_sequence_20th_term_l585_585205


namespace correct_propositions_2_and_3_l585_585700

open Real

-- Define the companion point function
def companion_point (x y : ℝ) : ℝ × ℝ :=
  if x = 0 ∧ y = 0 then (0, 0) else (y / (x^2 + y^2), -x / (x^2 + y^2))

-- Propositions related to the companion point
def proposition_2 := ∀ (x y : ℝ), x^2 + y^2 = 1 → let (y_p, x_p) := companion_point x y in y_p^2 + x_p^2 = 1
def proposition_3 := ∀ (x y : ℝ), let (y_p, x_p) := companion_point x y in symmetric_about_x x y → let (y_q, x_q) := companion_point x (-y) in symmetric_about_y x_q y_q

-- Symmetric about x-axis definition
def symmetric_about_x (x y : ℝ) : Prop := x ≠ 0 → y ≠ 0

-- Symmetric about y-axis definition
def symmetric_about_y (x y : ℝ) : Prop := symmetric_about_x y x

-- Theorem stating that propositions ② and ③ are true
theorem correct_propositions_2_and_3 : proposition_2 ∧ proposition_3 :=
by trivial

end correct_propositions_2_and_3_l585_585700


namespace scientific_notation_l585_585067

-- defining the main result to be proved
theorem scientific_notation (h: 0.000014 = (1.4 * 10 ^ (-5))) : true :=
begin
  sorry
end

end scientific_notation_l585_585067


namespace derivative_at_one_l585_585258

theorem derivative_at_one
  (f : ℝ → ℝ)
  (f_diff : Differentiable ℝ f)
  (h : ∀ x, f x = 2 * x * (f' 1) + Real.log x) :
  (f' 1) = -1 :=
by
  sorry

end derivative_at_one_l585_585258


namespace irreducibility_polynomial_irreducibility_polynomial_l585_585723

variable (n : ℕ) (hn : n > 0)
variable (K : Type) [Field K] [Fintype K] (hK : Fintype.card K = 2^n)
variable (f : Polynomial K) (hf : f = Polynomial.C 1 + Polynomial.X + Polynomial.X ^ 4)

theorem irreducibility_polynomial (hn_even : Even n) : f.IsReducible := by
  sorry

theorem irreducibility_polynomial (hn_odd : Odd n) : f.IsIrreducible := by
  sorry

end irreducibility_polynomial_irreducibility_polynomial_l585_585723


namespace y_coordinate_of_A_l585_585620

-- Define the parabola and points P, Q
def parabola (x y : ℝ) : Prop := x^2 = 2 * y

def point_P : ℝ × ℝ := (4, 8) -- P is at (4, 8)
def point_Q : ℝ × ℝ := (-2, 2) -- Q is at (-2, 2)

-- Define the tangents at points P and Q
def tangent_P (x y : ℝ) : Prop := y = 4 * x - 8
def tangent_Q (x y : ℝ) : Prop := y = -2 * x - 2

-- Define the intersection point A of the tangents
def point_A (x y : ℝ) : Prop := tangent_P x y ∧ tangent_Q x y

-- Prove that the y-coordinate of point A is -4
theorem y_coordinate_of_A : ∃ y : ℝ, point_A 1 y ∧ y = -4 := 
by
  -- x-coordinate at the intersection point is found to be 1
  use -4
  split
  . sorry -- tangent_P (1, -4)
  . sorry -- tangent_Q (1, -4)

end y_coordinate_of_A_l585_585620


namespace smallest_n_for_real_root_gt_1999_l585_585954

def P (x : ℝ) (n : ℕ) : ℝ := x^n - x^{n-1} - x^{n-2} - ... - x - 1 -- Full polynomial definition required

theorem smallest_n_for_real_root_gt_1999 :
  ∃ (n : ℕ), (P (1.999) n = 0) ∧ (∀ (k : ℕ), k < n → ¬(P (1.999) k = 0)) := sorry

end smallest_n_for_real_root_gt_1999_l585_585954


namespace smallest_angle_in_15_sided_polygon_arithmetic_sequence_l585_585007

theorem smallest_angle_in_15_sided_polygon_arithmetic_sequence
  (a d : ℕ) 
  (angles : Fin 15 → ℕ)
  (h_seq : ∀ i : Fin 15, angles i = a + i * d)
  (h_convex : ∀ i : Fin 15, angles i < 180)
  (h_sum : ∑ i, angles i = 2340) : 
  a = 135 := 
sorry

end smallest_angle_in_15_sided_polygon_arithmetic_sequence_l585_585007


namespace area_of_inscribed_hexagon_l585_585834

theorem area_of_inscribed_hexagon (r : ℝ) (h : π * r^2 = 100 * π) : 
  ∃ (A : ℝ), A = 150 * real.sqrt 3 :=
by 
  -- Definitions of necessary geometric entities and properties would be here
  -- Proof would be provided here
  sorry

end area_of_inscribed_hexagon_l585_585834


namespace faster_cow_days_to_eat_one_bag_l585_585874

-- Conditions as assumptions
def num_cows : ℕ := 60
def num_husks : ℕ := 150
def num_days : ℕ := 80
def faster_cows : ℕ := 20
def normal_cows : ℕ := num_cows - faster_cows
def faster_rate : ℝ := 1.3

-- The question translated to Lean 4 statement
theorem faster_cow_days_to_eat_one_bag :
  (faster_cows * faster_rate + normal_cows) / num_cows * (num_husks / num_days) = 1 / 27.08 :=
sorry

end faster_cow_days_to_eat_one_bag_l585_585874


namespace no_real_solution_for_inequality_l585_585926

theorem no_real_solution_for_inequality :
  ¬ ∃ a : ℝ, ∃ x : ℝ, ∀ b : ℝ, |x^2 + 4*a*x + 5*a| ≤ 3 :=
by
  sorry

end no_real_solution_for_inequality_l585_585926


namespace arrangements_with_gap_l585_585810

theorem arrangements_with_gap (A B : Type) (row : list Type) (h : row.length = 6) :
  let total := (6!).toInt
  let adjacent := (2 * (5!).toInt)
  total - adjacent = 480 :=
by sorry

end arrangements_with_gap_l585_585810


namespace correct_statements_l585_585496

theorem correct_statements :
  (let variance := 4,
       transformed_sd := @Real.sqrt (4 * 4),
       normal_prob := 0.5 - 0.2
   in transformed_sd = 4 ∧ normal_prob = 0.3) :=
by
  let variance := 4
  let transformed_sd := @Real.sqrt (4 * 4)
  let normal_prob := 0.5 - 0.2
  have h1 : transformed_sd = 4 := sorry
  have h2 : normal_prob = 0.3 := sorry
  show transformed_sd = 4 ∧ normal_prob = 0.3 from ⟨h1, h2⟩

end correct_statements_l585_585496


namespace volume_of_box_l585_585882

theorem volume_of_box (L W S : ℕ) (hL : L = 48) (hW : W = 36) (hS : S = 8) : 
  let new_length := L - 2 * S,
      new_width := W - 2 * S,
      height := S
  in new_length * new_width * height = 5120 := by
sorry

end volume_of_box_l585_585882


namespace smallest_cube_ends_in_584_l585_585955

theorem smallest_cube_ends_in_584 (n : ℕ) : n^3 ≡ 584 [MOD 1000] → n = 34 := by
  sorry

end smallest_cube_ends_in_584_l585_585955


namespace factor_problem_l585_585431

theorem factor_problem (C D : ℤ) (h1 : 16 * x^2 - 88 * x + 63 = (C * x - 21) * (D * x - 3)) (h2 : C * D + C = 21) : C = 7 ∧ D = 2 :=
by 
  sorry

end factor_problem_l585_585431


namespace arrange_books_l585_585760

theorem arrange_books (A F E : ℕ) (hA : A = 2) (hF : F = 3) (hE : E = 4) : 
  let units := 3 -- 1 Arabic unit + 1 English unit + 3 French books
  let total_books := A + F + E
  (units + 2)! * (A)! * (E)! = 5760 :=
by
  let units := 3
  let total_books := hA + hF + hE
  have hA_fact : A! = 2! := by sorry
  have hE_fact : E! = 4! := by sorry
  have combined_fact : (units + 2)! = 5! := by sorry
  rw [combined_fact, hA_fact, hE_fact]
  norm_num
  exact sorry

end arrange_books_l585_585760


namespace solve_for_x_l585_585405

theorem solve_for_x (x : ℝ) (h : 64^(3 * x) = 16^(4 * x - 5)) : x = -10 := 
by
  sorry

end solve_for_x_l585_585405


namespace product_of_slopes_eq_neg_one_fourth_max_area_triangle_AMN_eq_six_sqrt_three_l585_585244

-- Definitions of conditions
def is_outside_ellipse (x₀ y₀ : ℝ) : Prop :=
  (x₀^2 / 16 + y₀^2 / 4 > 1)

def is_point_on_ellipse (x y : ℝ) : Prop :=
  (x^2 / 16 + y^2 / 4 = 1)

def slope (x1 y1 x2 y2 : ℝ) : ℝ :=
  (y2 - y1) / (x2 - x1)

def product_of_slopes (x₀ y₀ : ℝ) :=
  slope 0 0 x₀ y₀ * slope x₀ y₀ 16 4 = -1 / 4

def max_area_triangle_AMN :=
  6 * real.sqrt 3

-- Theorem statements
theorem product_of_slopes_eq_neg_one_fourth : ∀ (x₀ y₀ : ℝ),
  is_outside_ellipse x₀ y₀ →
  product_of_slopes x₀ y₀ := sorry

theorem max_area_triangle_AMN_eq_six_sqrt_three : ∀ (x₀ y₀ : ℝ),
  is_outside_ellipse x₀ y₀ →
  x₀^2 + 4*y₀^2 = 64 →
  max_area_triangle_AMN = 6 * real.sqrt 3 :=
  sorry

end product_of_slopes_eq_neg_one_fourth_max_area_triangle_AMN_eq_six_sqrt_three_l585_585244


namespace find_g3_l585_585434

noncomputable def g : ℝ → ℝ := sorry

theorem find_g3 (h : ∀ x : ℝ, x ≠ 1/4 → g(x) + g((x + 1) / (1 - 4 * x)) = x) : g 3 = 50 / 27 :=
sorry

end find_g3_l585_585434


namespace minimum_value_of_xy_l585_585676

theorem minimum_value_of_xy (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 2 * x + y + 6 = x * y) : 
  18 ≤ x * y :=
sorry

end minimum_value_of_xy_l585_585676


namespace largest_k_value_l585_585594

theorem largest_k_value (a b c d : ℕ) (k : ℝ)
  (h1 : a + b = c + d)
  (h2 : 2 * (a * b) = c * d)
  (h3 : a ≥ b) :
  (∀ k', (∀ a b (h1_b : a + b = c + d)
              (h2_b : 2 * a * b = c * d)
              (h3_b : a ≥ b), (a : ℝ) / (b : ℝ) ≥ k') → k' ≤ k) → k = 3 + 2 * Real.sqrt 2 :=
sorry

end largest_k_value_l585_585594


namespace smallest_m_for_R_m_eq_l_l585_585746

noncomputable def smallest_positive_m (α β : ℝ) (l : ℝ → ℝ) : ℕ :=
  let angle_change := 2 * β - 2 * α
  let k := (-1 : ℤ)
  let m := (-1 : ℤ) * (75 : ℤ)
  m.toNat

theorem smallest_m_for_R_m_eq_l :
  let l_1_angle := Real.pi / 50
  let l_2_angle := Real.pi / 75
  let l := λ x : ℝ, (7 / 25) * x
  smallest_positive_m l_1_angle l_2_angle l = 75 :=
by
  -- The proof needs to establish that m = 75 is the correct answer.
  -- The proof steps would go here.
  sorry

end smallest_m_for_R_m_eq_l_l585_585746


namespace speed_of_current_l585_585445

def rowing_speed_still_water_kmph : ℝ := 20  -- km/h
def distance_downstream_meters : ℝ := 60  -- meters
def time_downstream_seconds : ℝ := 9.390553103577801  -- seconds

theorem speed_of_current :
  let rowing_speed_still_water_mps := (rowing_speed_still_water_kmph * 1000) / 3600 in  -- convert to m/s
  let downstream_speed := distance_downstream_meters / time_downstream_seconds in
  let current_speed_mps := downstream_speed - rowing_speed_still_water_mps in
  let current_speed_kmph := current_speed_mps * 3600 / 1000 in
  current_speed_kmph ≈ 3.0125 :=
by
  sorry

end speed_of_current_l585_585445


namespace T_15_equals_2555_l585_585361

noncomputable def T (n : ℕ) : ℕ :=
if n = 1 then 2
else if n = 2 then 3
else if n = 3 then 7
else if n = 4 then 15
else T (n - 1) + T (n - 2) + T (n - 3) + T (n - 4)

theorem T_15_equals_2555 : T 15 = 2555 := 
by
  sorry

end T_15_equals_2555_l585_585361


namespace area_of_triangle_ABC_unit_circle_l585_585375

variables {A B C D E O I : Type} [InnerProductSpace ℝ A]
variables (triangle_ABC : Triangle A B C) 
variables (unit_circle : circle A O 1)
variables (incenter_I : incenter triangle_ABC I)
variables (angle_bisector_AD : ∀ {BC}, AD = angle_bisector ∠BAC)
variables (intersection_D : D = intersection_line_angle_bisector BC AD)
variables (circumcircle_triangle_ADO : ∀ {BC}, circumcircle (triangle A D O) BC E ∧ E ∈ IO)

def area_triangle (cos_A : ℝ) [hcos : cos_A = 12 / 13] : ℝ :=
  (3 * (1 - cos_A) * (5 / 13))

theorem area_of_triangle_ABC_unit_circle
  (triangle_inscribed_unit : unit_circle.triangle_ABC)
  (cos_A : cos A = 12 / 13) :
  area_triangle cos_A = 15 / 169 :=
sorry

end area_of_triangle_ABC_unit_circle_l585_585375


namespace andrey_boris_denis_eat_candies_l585_585164

def andrey_boris_condition (a b : ℕ) : Prop :=
  a = 4 ∧ b = 3

def andrey_denis_condition (a d : ℕ) : Prop :=
  a = 6 ∧ d = 7

def total_candies_condition (total : ℕ) : Prop :=
  total = 70

theorem andrey_boris_denis_eat_candies :
  ∃ (a b d : ℕ), andrey_boris_condition a b ∧ andrey_denis_condition a d ∧ 
                  (total_candies_condition (2 * (12 + 9 + 14)) ∧ 
                   2 * 12 = 24 ∧ 2 * 9 = 18 ∧ 2 * 14 = 28) →
                  (a = 24 ∧ b = 18 ∧ d = 28) :=
by
  sorry

end andrey_boris_denis_eat_candies_l585_585164


namespace tan_beta_is_six_over_seventeen_l585_585613
-- Import the Mathlib library

-- Define the problem in Lean
theorem tan_beta_is_six_over_seventeen
  (α β : ℝ) (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2)
  (h1 : Real.sin α = 4 / 5)
  (h2 : Real.tan (α - β) = 2 / 3) :
  Real.tan β = 6 / 17 := 
by
  sorry

end tan_beta_is_six_over_seventeen_l585_585613


namespace measure_17_kg_of_cranberries_l585_585316

theorem measure_17_kg_of_cranberries :
  ∃ w1 w2 : ℝ, 0 < w1 ∧ 0 < w2 ∧ w1 + w2 = 17 ∧ 
  (1.surjective f → ∃ x ∈ s, f x = y)/w1 + w2 = 22 :=
sorry

end measure_17_kg_of_cranberries_l585_585316


namespace max_almost_centers_of_symmetry_l585_585501

-- Define the concept of an "almost center of symmetry"
structure FiniteSet (α : Type) :=
(points : Set α)
(finite_points : points.finite)

noncomputable def is_almost_center_of_symmetry {α : Type} [MetricSpace α] 
  (M : FiniteSet α) (O : α) : Prop :=
  ∃ P : α, P ∈ M.points ∧ ∀ Q ∈ (M.points \ {P}), 
  ∃ R ∈ (M.points \ {P}), dist O Q = dist O R

-- The main theorem statement
theorem max_almost_centers_of_symmetry {α : Type} [MetricSpace α] 
  (M : FiniteSet α) : 
  (∃ O : α, is_almost_center_of_symmetry M O) → 
  (Set.card {O : α | is_almost_center_of_symmetry M O} ≤ 3) := 
by  sorry

end max_almost_centers_of_symmetry_l585_585501


namespace min_box_height_l585_585386

noncomputable def minimum_height (x : ℝ) : ℝ := x + 5

theorem min_box_height :
  ∃ (x : ℝ), x ≥ 0 ∧ (6 * x^2 + 20 * x ≥ 150) ∧ minimum_height x = 10 :=
by
  use 5
  simp [minimum_height]
  split
  { linarith }
  split
  { linarith }
  { linarith }
  sorry

end min_box_height_l585_585386


namespace smallest_four_digit_multiple_of_18_l585_585981

theorem smallest_four_digit_multiple_of_18 : ∃ n : ℕ, (1000 ≤ n) ∧ (n < 10000) ∧ (n % 18 = 0) ∧ (∀ m : ℕ, (1000 ≤ m) ∧ (m < 10000) ∧ (m % 18 = 0) → n ≤ m) ∧ n = 1008 :=
by
  sorry

end smallest_four_digit_multiple_of_18_l585_585981


namespace candies_eaten_l585_585170

-- Definitions

def Andrey_rate_eq_Boris_rate (candies_eaten_by_Andrey candies_eaten_by_Boris : ℕ) : Prop :=
  candies_eaten_by_Andrey / 4 = candies_eaten_by_Boris / 3

def Denis_rate_eq_Andrey_rate (candies_eaten_by_Denis candies_eaten_by_Andrey : ℕ) : Prop :=
  candies_eaten_by_Denis / 7 = candies_eaten_by_Andrey / 6

def total_candies (candies_eaten_by_Andrey candies_eaten_by_Boris candies_eaten_by_Denis : ℕ) : Prop :=
  candies_eaten_by_Andrey + candies_eaten_by_Boris + candies_eaten_by_Denis = 70

-- Theorem to prove the candies eaten by Andrey, Boris, and Denis
theorem candies_eaten (candies_eaten_by_Andrey candies_eaten_by_Boris candies_eaten_by_Denis : ℕ) :
  Andrey_rate_eq_Boris_rate candies_eaten_by_Andrey candies_eaten_by_Boris →
  Denis_rate_eq_Andrey_rate candies_eaten_by_Denis candies_eaten_by_Andrey →
  total_candies candies_eaten_by_Andrey candies_eaten_by_Boris candies_eaten_by_Denis →
  candies_eaten_by_Andrey = 24 ∧ candies_eaten_by_Boris = 18 ∧ candies_eaten_by_Denis = 28 :=
  by sorry

end candies_eaten_l585_585170


namespace tax_free_amount_l585_585142

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

end tax_free_amount_l585_585142


namespace andrey_boris_denis_eat_candies_l585_585162

def andrey_boris_condition (a b : ℕ) : Prop :=
  a = 4 ∧ b = 3

def andrey_denis_condition (a d : ℕ) : Prop :=
  a = 6 ∧ d = 7

def total_candies_condition (total : ℕ) : Prop :=
  total = 70

theorem andrey_boris_denis_eat_candies :
  ∃ (a b d : ℕ), andrey_boris_condition a b ∧ andrey_denis_condition a d ∧ 
                  (total_candies_condition (2 * (12 + 9 + 14)) ∧ 
                   2 * 12 = 24 ∧ 2 * 9 = 18 ∧ 2 * 14 = 28) →
                  (a = 24 ∧ b = 18 ∧ d = 28) :=
by
  sorry

end andrey_boris_denis_eat_candies_l585_585162


namespace cos_60_eq_half_l585_585450

theorem cos_60_eq_half : Real.cos (60 * Real.pi / 180) = 1 / 2 :=
by
  sorry

end cos_60_eq_half_l585_585450


namespace every_nonnegative_rational_appears_exactly_once_l585_585599

def f (n : ℕ) : ℕ := sorry -- Placeholder for the function f

noncomputable def x : ℕ → ℚ
| 0       := 0
| (n + 1) := 1 / (1 + 2 * (f (n + 1)) - (x n))

theorem every_nonnegative_rational_appears_exactly_once :
  ∀ (q : ℚ), 0 ≤ q → Nat.Den q > 0 → ∃! n : ℕ, x n = q :=
sorry

end every_nonnegative_rational_appears_exactly_once_l585_585599


namespace inverse_mod_187_l585_585943

theorem inverse_mod_187 : ∃ (x : ℤ), 0 ≤ x ∧ x ≤ 186 ∧ (2 * x) % 187 = 1 :=
by
  use 94
  sorry

end inverse_mod_187_l585_585943


namespace basketball_team_combinations_l585_585024

theorem basketball_team_combinations : 
  let total_players := 16
  let twins := 2
  let players_without_twins := total_players - twins
  let choose (n k : ℕ) := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

  let no_twins := choose players_without_twins 5
  let one_twin := 2 * choose players_without_twins 4
  let both_twins := choose players_without_twins 3

in no_twins + one_twin + both_twins = 4368 := 
by 
  let total_players := 16
  let twins := 2
  let players_without_twins := total_players - twins
  let choose (n k : ℕ) := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

  let no_twins := choose players_without_twins 5
  let one_twin := 2 * choose players_without_twins 4
  let both_twins := choose players_without_twins 3

  show no_twins + one_twin + both_twins = 4368 from sorry

end basketball_team_combinations_l585_585024


namespace total_subset_sum_l585_585611

-- Define the set and the function to calculate the sum given the conditions
def S (n : ℕ) : Finset ℕ := Finset.range (n + 1) \ {0}

def subset_sum (n : ℕ) (A : Finset ℕ) : ℤ :=
  A.sum (λ k => (-1 : ℤ)^(k : ℤ) * (k : ℤ))

-- Define the statement about the total sum of subset sums
theorem total_subset_sum (n : ℕ) :
  ∑ A in (S n).powerset \ {∅},
    subset_sum n A =
  (-1 : ℤ)^n * (n + ((1 : ℤ) - (-1 : ℤ)^n) / 2) * 2^(n - 2) :=
by
  sorry

end total_subset_sum_l585_585611


namespace staff_discount_l585_585877

theorem staff_discount (d : ℝ) (h1 : 0 < d) : 
  ∃ (x : ℝ), 0.85 * d * (1 - x / 100) = 0.765 * d ∧ x = 10 :=
by
  use 10
  split
  sorry

end staff_discount_l585_585877


namespace triangle_two_acute_angles_l585_585049

theorem triangle_two_acute_angles (A B C : ℝ) (h_triangle : A + B + C = 180) (h_pos : A > 0 ∧ B > 0 ∧ C > 0)
  (h_acute_triangle: A < 90 ∨ B < 90 ∨ C < 90): A < 90 ∧ B < 90 ∨ A < 90 ∧ C < 90 ∨ B < 90 ∧ C < 90 :=
by
  sorry

end triangle_two_acute_angles_l585_585049


namespace factorization_of_expression_l585_585585

theorem factorization_of_expression (a : ℝ) : a^3 - 4 * a = a * (a + 2) * (a - 2) := by
  sorry

end factorization_of_expression_l585_585585


namespace animals_per_aquarium_l585_585825

theorem animals_per_aquarium
  (saltwater_aquariums : ℕ)
  (saltwater_animals : ℕ)
  (h1 : saltwater_aquariums = 56)
  (h2 : saltwater_animals = 2184)
  : saltwater_animals / saltwater_aquariums = 39 := by
  sorry

end animals_per_aquarium_l585_585825


namespace range_of_a_l585_585647

theorem range_of_a (a : ℝ) :
  (∀ x ≥ 0, real.log (1 + x) ≥ a * x / (1 + x)) ↔ a ≤ 1 :=
sorry

end range_of_a_l585_585647


namespace simplify1_simplify2_l585_585404

-- Proof for the first expression
theorem simplify1 : 
  sin (π / 6 - 2 * π) * cos (π / 4 + π) = - sqrt 2 / 4 := 
by
  sorry

-- Proof for the second expression
theorem simplify2 : 
  sin (π / 4 + 5 * π / 2) = sqrt 2 / 2 := 
by
  sorry

end simplify1_simplify2_l585_585404


namespace max_depth_of_copper_cube_in_ice_l585_585519

-- Define the constants given in the problem
def l : ℝ := 0.05 -- Edge length of the copper cube in meters
def t1 : ℝ := 100 -- Initial temperature of the copper cube in degrees Celsius
def t2 : ℝ := 0 -- Temperature of the ice in degrees Celsius
def c_Cu : ℝ := 400 -- Specific heat capacity of copper in J/(kg °C)
def λ : ℝ := 3.3 * 10^5 -- Latent heat of fusion of ice in J/kg
def ρ_Cu : ℝ := 8900 -- Density of copper in kg/m^3
def ρ_ice : ℝ := 900 -- Density of ice in kg/m^3

-- Prove the maximum depth the copper cube can sink into the ice
theorem max_depth_of_copper_cube_in_ice : 
  ∀ (h : ℝ), 
    h = ((c_Cu * ρ_Cu * l^3 * (t1 - t2)) / (λ * ρ_ice)) / l^2 → 
    h = 0.06 := 
by
  sorry

end max_depth_of_copper_cube_in_ice_l585_585519


namespace isosceles_right_triangle_hypotenuse_length_l585_585041

theorem isosceles_right_triangle_hypotenuse_length
  (DE DF : ℝ)
  (DP PE DQ QF QE PF : ℝ)
  (h_isosceles : DE = DF)
  (h_ratios : DP / PE = 2 / 3 ∧ DQ / QF = 2 / 3)
  (h_len_QE : QE = 25)
  (h_len_PF : PF = 36)
  (h_triangle_QE : (2 * DE / 5) ^ 2 + DE ^ 2 = QM ^ 2)
  (h_triangle_PF : DE ^ 2 + (2 * DF / 5) ^ 2 = PF ^ 2) :
  let s := DE in hypotenuse_length s = 54.88 :=
by
  sorry

noncomputable def hypotenuse_length (s : ℝ) : ℝ := s * real.sqrt 2

end isosceles_right_triangle_hypotenuse_length_l585_585041


namespace sum_of_possible_lengths_l585_585118

theorem sum_of_possible_lengths
  (m : ℕ) 
  (h1 : m < 18)
  (h2 : m > 4) : ∑ i in (Finset.range 13).map (λ x, x + 5) = 143 := by
sorry

end sum_of_possible_lengths_l585_585118


namespace steak_amount_per_member_l585_585820

theorem steak_amount_per_member : 
  ∀ (num_members steaks_needed ounces_per_steak total_ounces each_amount : ℕ),
    num_members = 5 →
    steaks_needed = 4 →
    ounces_per_steak = 20 →
    total_ounces = steaks_needed * ounces_per_steak →
    each_amount = total_ounces / num_members →
    each_amount = 16 :=
by
  intros num_members steaks_needed ounces_per_steak total_ounces each_amount
  intro h_members h_steaks h_ounces_per_steak h_total_ounces h_each_amount
  sorry

end steak_amount_per_member_l585_585820


namespace darcy_walking_speed_proof_l585_585211

noncomputable def darcy_walking_speed (d : ℝ) (v_train : ℝ) (t_extra_train : ℝ) (t_extra_walk : ℝ) : ℝ :=
  let t_train := d / v_train + t_extra_train in
  let t_walk := t_train + t_extra_walk in
  d / t_walk

theorem darcy_walking_speed_proof :
  darcy_walking_speed 1.5 20 (0.5 / 60) (25 / 60) = 3 := 
by 
  -- proof will go here 
  sorry

end darcy_walking_speed_proof_l585_585211


namespace place_value_differences_sum_l585_585819

theorem place_value_differences_sum :
  let numeral := 58219435
  let place_value_8 := 80000000
  let place_value_first_5 := 500000000
  let place_value_second_5 := 50000
  let place_value_1 := 1000000
  let difference_first_5_8 := place_value_first_5 - place_value_8
  let difference_second_5_1 := abs (place_value_second_5 - place_value_1)
  let sum_of_differences := difference_first_5_8 + difference_second_5_1
  let num_of_2s := 1
  sum_of_differences ^ num_of_2s = 420950000 :=
by 
  let numeral := 58219435
  let place_value_8 := 80000000
  let place_value_first_5 := 500000000
  let place_value_second_5 := 50000
  let place_value_1 := 1000000
  let difference_first_5_8 := place_value_first_5 - place_value_8
  let difference_second_5_1 := abs (place_value_second_5 - place_value_1)
  let sum_of_differences := difference_first_5_8 + difference_second_5_1
  let num_of_2s := 1
  exact rfl

end place_value_differences_sum_l585_585819


namespace sum_of_15_smallest_elements_of_S_l585_585725

-- Conditions in a formal manner
variable {x : Fin 1369 → ℂ}

-- Define d(m) and r(m)
def d (m : ℕ) : ℕ := m / 37
def r (m : ℕ) : ℕ := m % 37

-- Define the matrix A
def a (i j : Fin 1369) : ℂ :=
  if r i = r j ∧ i ≠ j then x (⟨37 * d j + d i, sorry⟩)
  else if d i = d j ∧ i ≠ j then -x (⟨37 * r i + r j, sorry⟩)
  else if i = j then x (⟨38 * d i, sorry⟩) - x (⟨38 * r i, sorry⟩)
  else 0

-- Define A as a matrix
def A : Matrix (Fin 1369) (Fin 1369) ℂ := fun i j => a i j

-- Define r-murine
def r_murine (A : Matrix (Fin 1369) (Fin 1369) ℂ) (r : ℕ) :=
  ∃ M : Matrix (Fin 1369) (Fin 1369) ℂ, (A * M - 1).cols.take r = 0

-- Define the rank of A
def rk (A : Matrix (Fin 1369) (Fin 1369) ℂ) :=
  Nat.find (λ r, r_murine A r)

-- Define the set S of possible ranks
def S : Set ℕ := {r | ∃ x : Fin 1369 → ℂ, rk (matrix_from_fn x) = r}

-- Main theorem statement
theorem sum_of_15_smallest_elements_of_S : 
  (∑ k in (Finset.range 15), (1369 - (k * (k - 1) / 2 + 703))) = 9745 :=
  sorry

end sum_of_15_smallest_elements_of_S_l585_585725


namespace divide_square_into_smaller_squares_l585_585769

def P (n : Nat) : Prop :=
  ∃ f : ℕ → ℕ, (∀ m, m < n → f m ≠ 0) ∧ (∀ s, s ∈ finset.range n → s = n)

theorem divide_square_into_smaller_squares (n : Nat) (h : n > 5) : P n := sorry

end divide_square_into_smaller_squares_l585_585769


namespace smallest_angle_in_15_sided_polygon_arithmetic_sequence_l585_585005

theorem smallest_angle_in_15_sided_polygon_arithmetic_sequence
  (a d : ℕ) 
  (angles : Fin 15 → ℕ)
  (h_seq : ∀ i : Fin 15, angles i = a + i * d)
  (h_convex : ∀ i : Fin 15, angles i < 180)
  (h_sum : ∑ i, angles i = 2340) : 
  a = 135 := 
sorry

end smallest_angle_in_15_sided_polygon_arithmetic_sequence_l585_585005


namespace vectors_not_coplanar_l585_585553

def vector_a : ℝ × ℝ × ℝ := (3, 7, 2)
def vector_b : ℝ × ℝ × ℝ := (-2, 0, -1)
def vector_c : ℝ × ℝ × ℝ := (2, 2, 1)

def scalarTripleProduct (a b c : ℝ × ℝ × ℝ) : ℝ :=
  a.1 * (b.2 * c.3 - b.3 * c.2) - 
  a.2 * (b.1 * c.3 - b.3 * c.1) + 
  a.3 * (b.1 * c.2 - b.2 * c.1)

theorem vectors_not_coplanar (a b c : ℝ × ℝ × ℝ) (h1 : a = vector_a) (h2 : b = vector_b) (h3 : c = vector_c) :
  scalarTripleProduct a b c ≠ 0 :=
by
  simp [h1, h2, h3, scalarTripleProduct]
  norm_num
  sorry

end vectors_not_coplanar_l585_585553


namespace num_ways_for_volunteers_l585_585456

theorem num_ways_for_volunteers:
  let pavilions := 4
  let volunteers := 5
  let ways_to_choose_A := 4
  let ways_to_choose_B_after_A := 3
  let total_distributions := 
    let case_1 := 2
    let case_2 := (2^3) - 2
    case_1 + case_2
  ways_to_choose_A * ways_to_choose_B_after_A * total_distributions = 72 := 
by
  sorry

end num_ways_for_volunteers_l585_585456


namespace rectangle_area_from_circle_l585_585540

-- Circle radius
def radius : ℝ := 3.5

-- Circumference of the circle
def circumference : ℝ := 2 * Real.pi * radius

-- Ratio of length to breadth
def length_ratio : ℝ := 6
def breadth_ratio : ℝ := 5

-- Length and breadth of the rectangle
def length (C : ℝ) (L_ratio B_ratio : ℝ) : ℝ :=
  (C * L_ratio) / (2 * (L_ratio + B_ratio))

def breadth (C : ℝ) (L_ratio B_ratio : ℝ) : ℝ :=
  (C * B_ratio) / (2 * (L_ratio + B_ratio))

-- Area of the rectangle
def rectangle_area (L B : ℝ) : ℝ := 
  L * B

theorem rectangle_area_from_circle (h_circumference : circumference = 7 * Real.pi) :
  rectangle_area (length circumference length_ratio breadth_ratio) (breadth circumference length_ratio breadth_ratio) = (735 * Real.pi^2) / 242 := by
  sorry -- proof to be filled in

end rectangle_area_from_circle_l585_585540


namespace number_of_possible_monograms_l585_585753

-- Define the set of letters before 'M'
def letters_before_M : Finset Char := {'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L'}

-- Define the set of letters after 'M'
def letters_after_M : Finset Char := {'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'}

-- State the theorem 
theorem number_of_possible_monograms : 
  (letters_before_M.card * letters_after_M.card) = 156 :=
by
  sorry

end number_of_possible_monograms_l585_585753


namespace possible_values_of_ab_plus_ac_plus_bc_l585_585364

theorem possible_values_of_ab_plus_ac_plus_bc (a b c : ℝ) (h : a + b + 2 * c = 0) :
  ∃ x ∈ Set.Iic 0, ab + ac + bc = x :=
sorry

end possible_values_of_ab_plus_ac_plus_bc_l585_585364


namespace daily_charge_second_agency_l585_585009

theorem daily_charge_second_agency (x : ℝ) 
  (h_first_agency_total : 20.25 + 0.14 * 25 < x + 0.22 * 25) :
  x = 18.25 := 
by
  have h : 20.25 + 3.5 = x + 5.5, by calc
    20.25 + (0.14 * 25) = 20.25 + 3.5 : by norm_num
    ... = x + (0.22 * 25) : h_first_agency_total,
  linarith


end daily_charge_second_agency_l585_585009


namespace max_value_of_A_l585_585280

noncomputable def A (x y : ℝ) : ℝ :=
  (Real.sqrt (Real.cos x * Real.cos y)) / (Real.sqrt (Real.cot x) + Real.sqrt (Real.cot y))

theorem max_value_of_A (x y : ℝ) (h1 : 0 < x) (h2 : x < (π / 2)) (h3 : 0 < y) (h4 : y < (π / 2)) :
  A x y ≤ (Real.sqrt 2) / 4 :=
sorry

end max_value_of_A_l585_585280


namespace det_eq_of_inv_eq_l585_585600

open Matrix Real

theorem det_eq_of_inv_eq (n : ℕ) (A B : Matrix ℤ n n) (h : n ≥ 2) 
  (inv_eq : (A⁻¹ + B⁻¹) = (A + B)⁻¹) : det A = det B := 
  sorry

end det_eq_of_inv_eq_l585_585600


namespace ratio_of_areas_l585_585413

-- Define the side lengths of Squared B and Square C
variables (y : ℝ)

-- Define the areas of Square B and C
def area_B := (2 * y) * (2 * y)
def area_C := (8 * y) * (8 * y)

-- The theorem statement proving the ratio of the areas
theorem ratio_of_areas : area_B y / area_C y = 1 / 16 := 
by sorry

end ratio_of_areas_l585_585413


namespace Donovan_Mitchell_current_average_l585_585579

theorem Donovan_Mitchell_current_average 
    (points_per_game_goal : ℕ) 
    (games_played : ℕ) 
    (total_games_goal : ℕ) 
    (average_needed_remaining_games : ℕ)
    (points_needed : ℕ) 
    (remaining_games : ℕ) 
    (x : ℕ) 
    (h₁ : games_played = 15) 
    (h₂ : total_games_goal = 20) 
    (h₃ : points_per_game_goal = 30) 
    (h₄ : remaining_games = total_games_goal - games_played)
    (h₅ : average_needed_remaining_games = 42) 
    (h₆ : points_needed = remaining_games * average_needed_remaining_games) 
    (h₇ : points_needed = 210)  
    (h₈ : points_per_game_goal * total_games_goal = 600) 
    (h₉ : games_played * x + points_needed = 600) : 
    x = 26 :=
by {
  sorry
}

end Donovan_Mitchell_current_average_l585_585579


namespace sin_neg_330_eq_one_half_l585_585572

theorem sin_neg_330_eq_one_half :
  ∀ (sin : ℝ → ℝ), (∀ x, sin (x + 360) = sin x) → sin 30 = 1/2 → sin (-330) = 1/2 :=
by
  intros sin periodicity sin_30_eq_half
  have h1 : sin (-330 + 360) = sin (-330) := by rw [periodicity]
  rw [neg_add] at h1
  rw [← h1] at sin_30_eq_half
  exact sin_30_eq_half

#check sin_neg_330_eq_one_half

end sin_neg_330_eq_one_half_l585_585572


namespace inverse_function_l585_585438

noncomputable def f (x : ℝ) : ℝ := x^2

theorem inverse_function (x : ℝ) (h₁ : x ≤ -1) (h₂ : x ≥ 1) : f(-real.sqrt x) = x :=
by
  sorry

end inverse_function_l585_585438


namespace carnival_friends_l585_585561

theorem carnival_friends (F : ℕ) (h1 : 865 % F ≠ 0) (h2 : 873 % F = 0) : F = 3 :=
by
  -- proof is not required
  sorry

end carnival_friends_l585_585561


namespace finite_questions_solution_l585_585480

theorem finite_questions_solution (m n : ℕ) (hm : 0 < m) (hn : 0 < n) (T S : ℕ) :
  (T = m + n ∨ T ≠ m + n) → 
  (∃ k : ℕ, (2 * k - 1 = 2 * k - 1 → (hm : m > S - n / k)) ∨ 
  (2 * k = 2 * k → (hn : n > -(T - S)))) :=
by
  sorry

end finite_questions_solution_l585_585480


namespace find_point_on_quadrilateral_l585_585659

namespace InscribedQuadrilateral

variables {α : Type*}
variables [euclidean_space α] -- Assuming α represents a Euclidean space

noncomputable def is_circle (c : Set α) := ∃ (O : α) (r : ℝ), ∀ (P : α), P ∈ c ↔ dist O P = r

theorem find_point_on_quadrilateral
  (A B C : α) (hABC_on_circle : ∃ (c : Set α), is_circle c ∧ A ∈ c ∧ B ∈ c ∧ C ∈ c)
  (h_inscribed : ∃ D : α, ∀ (O : α), is_circle {A, B, C, D} ∧ 
    ∃ (r : ℝ), ∀ (P : α), P ∈ {A, B, C, D} ↔ dist O P = r) :
  ∃ (O : α), ∠ A O C = 270 - ∠ A B C :=
sorry

end InscribedQuadrilateral

end find_point_on_quadrilateral_l585_585659


namespace solve_for_x_l585_585409

theorem solve_for_x (x : ℝ) : 64^(3 * x) = 16^(4 * x - 5) → x = -10 := 
by 
  sorry

end solve_for_x_l585_585409


namespace function_not_monotonic_a_l585_585558

open Real

theorem function_not_monotonic_a (a : ℝ) (h : a > 0) : ¬(∀ x y ∈ Ioo 0 3, (f x ≤ f y ↔ x ≤ y)) → a > 2/3 :=
by
  let f : ℝ → ℝ := λ x, (1/3) * a * x^3 - x^2
  intro non_monotonic
  sorry

end function_not_monotonic_a_l585_585558


namespace three_consecutive_integers_prime_factors_l585_585396

theorem three_consecutive_integers_prime_factors (n : ℕ) (h : n > 7) :
  ∃ i, i ∈ ({n, n+1, n+2} : set ℕ) ∧
  ∃ p q : ℕ, (p.prime ∧ q.prime ∧ p ≠ q ∧ p ∣ i ∧ q ∣ i) :=
by
  sorry

end three_consecutive_integers_prime_factors_l585_585396


namespace find_x_l585_585298

theorem find_x (x : ℝ) (h1 : 0 < x) (h2 : real.sqrt (12 * x) * real.sqrt (18 * x) * real.sqrt (6 * x) * real.sqrt (9 * x) = 27) : x = 1 / 2 :=
by 
  sorry

end find_x_l585_585298


namespace blood_drug_concentration_at_13_hours_l585_585193

theorem blood_drug_concentration_at_13_hours :
  let peak_time := 3
  let test_interval := 2
  let decrease_rate := 0.4
  let target_rate := 0.01024
  let time_to_reach_target := (fun n => (2 * n + 1))
  peak_time + test_interval * 5 = 13 :=
sorry

end blood_drug_concentration_at_13_hours_l585_585193


namespace sum_max_min_f_l585_585807

noncomputable def f (x : ℝ) : ℝ := (sqrt 2 * sin (x + π / 4) + 2 * x^2 + x) / (2 * x^2 + cos x)

theorem sum_max_min_f : 
  let max_f := ... -- define the maximum value of f
  let min_f := ... -- define the minimum value of f
  (max_f + min_f = 2) :=
sorry

end sum_max_min_f_l585_585807


namespace roots_reciprocal_l585_585680

theorem roots_reciprocal (x1 x2 : ℝ) (h1 : x1^2 - 3 * x1 - 1 = 0) (h2 : x2^2 - 3 * x2 - 1 = 0) 
                         (h_sum : x1 + x2 = 3) (h_prod : x1 * x2 = -1) :
  (1 / x1) + (1 / x2) = -3 :=
by
  sorry

end roots_reciprocal_l585_585680


namespace horners_rule_example_l585_585048

noncomputable def f (x : ℝ) : ℝ :=
  1 + x + 0.5*x^2 + 0.16667*x^3 + 0.04167*x^4 + 0.00833*x^5

theorem horners_rule_example : f (-0.2) = 0.81873 :=
by
  have V₀ := 0.00833
  have V₁ := V₀ * -0.2 + 0.04167
  have V₂ := V₁ * -0.2 + 0.16667
  have V₃ := V₂ * -0.2 + 0.5
  have V₄ := V₃ * -0.2 + 1
  have V₅ := V₄ * -0.2 + 1
  show f (-0.2) = V₅ by
    rw [f]
    dsimp
    rw [V₀, V₁, V₂, V₃, V₄, V₅]
    sorry

end horners_rule_example_l585_585048


namespace g_one_minus_g_four_l585_585790

theorem g_one_minus_g_four (g : ℝ → ℝ)
  (h_linear : ∀ x y : ℝ, g (x + y) = g x + g y)
  (h_diff : ∀ x : ℝ, g (x + 1) - g x = 5) :
  g 1 - g 4 = -15 :=
sorry

end g_one_minus_g_four_l585_585790


namespace min_distance_l585_585672

theorem min_distance : ∃ (x y : ℝ), 5 * x + 12 * y = 60 ∧ sqrt (x^2 + y^2) = 60 / 13 :=
by
  -- This is where you would normally write the proof
  sorry

end min_distance_l585_585672


namespace total_students_above_8_l585_585317

theorem total_students_above_8 (T E B A : ℕ) (hT : T = 150) (hE : E = 72) 
  (hB : B = 20 * T / 100) (hA : A = 2 * E / 3) : A = 48 :=
begin
  sorry
end

end total_students_above_8_l585_585317


namespace eccentricity_of_ellipse_slope_of_ab_l585_585264

noncomputable def ellipse_equation (a : ℝ) (x y : ℝ) : Prop :=
  (x^2 / a^2) + y^2 = 1

noncomputable def point_on_ellipse (a : ℝ) : Prop :=
  ellipse_equation a 1 (real.sqrt 2 / 2)

theorem eccentricity_of_ellipse : ∃ (a c e : ℝ), 
  (point_on_ellipse a ∧ a^2 = 2 ∧ c^2 = a^2 - 1 ∧ e = c / a ∧ e = 1 / real.sqrt 2) :=
by {
  use [real.sqrt 2, 1, 1 / real.sqrt 2],
  sorry
}

noncomputable def slope (x1 y1 x2 y2 : ℝ) : ℝ :=
  (y2 - y1) / (x2 - x1)

noncomputable def line_through_focus (k : ℝ) (x y : ℝ) : Prop :=
  y = k * (x - 1)

theorem slope_of_ab (x1 y1 x2 y2 : ℝ) :
  (x1 + x2 = 4 * (real.sqrt 2 / 2)^2 / (1 + 2 * (real.sqrt 2 / 2)^2) ∧
   x1 * x2 = 2 * ((real.sqrt 2 / 2)^2 - 1) / (1 + 2 * (real.sqrt 2 / 2)^2) ∧
   slope 1 (real.sqrt 2 / 2) x1 y1 = real.sqrt 2 / 2 + (real.sqrt 2 / 2) ∧
   slope 1 (real.sqrt 2 / 2) x2 y2 = real.sqrt 2 / 2 + (real.sqrt 2 / 2) ∧
   slope 1 (real.sqrt 2 / 2) x1 y1 + slope 1 (real.sqrt 2 / 2) x2 y2 = 0 ∧
   slope (x1 - 1) y1 (x2 - 1) y2 = (real.sqrt 2 / 2)) :=
by {
  sorry
}

end eccentricity_of_ellipse_slope_of_ab_l585_585264


namespace sum_of_possible_m_values_l585_585126

theorem sum_of_possible_m_values : 
  sum (setOf m in {m | 4 < m ∧ m < 18}) = 132 :=
by
  sorry

end sum_of_possible_m_values_l585_585126


namespace candy_eating_l585_585179

-- Definitions based on the conditions
def candies (andrey_rate boris_rate denis_rate : ℕ) (andrey_candies boris_candies denis_candies total_candies : ℕ) : Prop :=
  andrey_rate = 4 ∧ boris_rate = 3 ∧ denis_rate = 7 ∧ andrey_candies = 24 ∧ boris_candies = 18 ∧ denis_candies = 28 ∧
  total_candies = andrey_candies + boris_candies + denis_candies

-- Problem statement
theorem candy_eating :
  ∃ (a b d : ℕ), 
    candies 4 3 7 a b d 70 :=
sorry

end candy_eating_l585_585179


namespace day_of_week_150th_day_previous_year_l585_585712

theorem day_of_week_150th_day_previous_year (N : ℕ) 
  (h1 : (275 % 7 = 4))  -- Thursday is 4th day of the week if starting from Sunday as 0
  (h2 : (215 % 7 = 4))  -- Similarly, Thursday is 4th day of the week
  : (150 % 7 = 6) :=     -- Proving the 150th day of year N-1 is a Saturday (Saturday as 6th day of the week)
sorry

end day_of_week_150th_day_previous_year_l585_585712


namespace correct_sqrt_operation_l585_585858

theorem correct_sqrt_operation :
  ∃ k : ℝ, (k = 1 ∧ ¬ (sqrt 5 - sqrt 3 = sqrt 2)) ∧
  (k = 2 ∧ ¬ (3 + sqrt 2 = 3 * sqrt 2)) ∧
  (k = 3 ∧ sqrt 6 * sqrt 2 = 2 * sqrt 3) ∧
  (k = 4 ∧ ¬ (sqrt 6 / 2 = 3)) :=
begin
  use 3,
  split,
  { use 1, exact ⟨by norm_num, by norm_num⟩ },
  split,
  { use 2, exact ⟨by norm_num, by norm_num⟩ },
  split,
  { use 3, simp [sqrt_mul] },
  split,
  { use 4, exact ⟨by norm_num, by norm_num⟩ }
end

end correct_sqrt_operation_l585_585858


namespace continuous_stripe_encircling_cube_l585_585938
noncomputable def probability_continuous_stripe (cube_faces : ℕ) : ℚ := 
  if cube_faces = 6 then 
    3 / 16777216 
  else 
    0 

theorem continuous_stripe_encircling_cube :
  ∀ (cube_faces : ℕ), 
    cube_faces = 6 → 
    probability_continuous_stripe cube_faces = 3 / 16777216 := 
by 
  intros 
  unfold probability_continuous_stripe 
  split_ifs 
  case h_1 { 
    refl 
  } 
  case h_2 { 
    sorry 
  }

end continuous_stripe_encircling_cube_l585_585938


namespace smallest_four_digit_multiple_of_18_l585_585961

theorem smallest_four_digit_multiple_of_18 : ∃ n: ℕ, (1000 ≤ n ∧ n < 10000 ∧ 18 ∣ n) ∧ ∀ m: ℕ, (1000 ≤ m ∧ m < n ∧ 18 ∣ m) → false :=
begin
  use 1008,
  split,
  { split,
    { -- prove 1000 ≤ 1008
      linarith,
    },
    { split,
      { -- prove 1008 < 10000
        linarith,
      },
      { -- prove 18 ∣ 1008
        norm_num,
      }
    }
  },
  { -- prove there is no four-digit multiple of 18 less than 1008
    intros m,
    intro h,
    cases h with h1 h2,
    cases h2 with h3 h4,
    linarith,
  }
end

end smallest_four_digit_multiple_of_18_l585_585961


namespace max_value_A_l585_585277

noncomputable theory
open Real

/-- Given x and y in the interval (0, π/2), the maximum value of the 
    expression A = sqrt(cos x * cos y) / (sqrt(cot x) + sqrt(cot y)) is sqrt(2) / 4. -/
theorem max_value_A (x y : ℝ) (hx : 0 < x ∧ x < π / 2) (hy : 0 < y ∧ y < π / 2) :
    (∃ x y ∈ Ioo 0 (π / 2), 
      ∀ x y, (0 < x ∧ x < π / 2 ∧ 0 < y ∧ y < π / 2) →
      (A = sqrt(cos x * cos y) / (sqrt (cos x / sin x) + sqrt (cos y / sin y))) ≤ sqrt 2 / 4) :=
sorry

end max_value_A_l585_585277


namespace probability_first_grade_probability_at_least_one_second_grade_l585_585690

-- Define conditions
def total_products : ℕ := 10
def first_grade_products : ℕ := 8
def second_grade_products : ℕ := 2
def inspected_products : ℕ := 2
def total_combinations : ℕ := Nat.choose total_products inspected_products
def first_grade_combinations : ℕ := Nat.choose first_grade_products inspected_products
def mixed_combinations : ℕ := first_grade_products * second_grade_products
def second_grade_combinations : ℕ := Nat.choose second_grade_products inspected_products

-- Define probabilities
def P_A : ℚ := first_grade_combinations / total_combinations
def P_B1 : ℚ := mixed_combinations / total_combinations
def P_B2 : ℚ := second_grade_combinations / total_combinations
def P_B : ℚ := P_B1 + P_B2

-- Statements
theorem probability_first_grade : P_A = 28 / 45 := sorry
theorem probability_at_least_one_second_grade : P_B = 17 / 45 := sorry

end probability_first_grade_probability_at_least_one_second_grade_l585_585690


namespace no_such_polynomials_f_g_l585_585554

theorem no_such_polynomials_f_g (f g : Polynomial ℝ) :
  ¬(∃ a b : ℝ, f^3 - g^2 = a * X + b) :=
sorry

end no_such_polynomials_f_g_l585_585554


namespace work_hours_l585_585749

-- Let h be the number of hours worked
def hours_worked (total_paid part_cost hourly_rate : ℕ) : ℕ :=
  (total_paid - part_cost) / hourly_rate

-- Given conditions
def total_paid : ℕ := 300
def part_cost : ℕ := 150
def hourly_rate : ℕ := 75

-- The statement to be proved
theorem work_hours :
  hours_worked total_paid part_cost hourly_rate = 2 :=
by
  -- Provide the proof here
  sorry

end work_hours_l585_585749


namespace remaining_money_after_expenditures_l585_585504

def initial_amount : ℝ := 200.50
def spent_on_sweets : ℝ := 35.25
def given_to_each_friend : ℝ := 25.20

theorem remaining_money_after_expenditures :
  ((initial_amount - spent_on_sweets) - 2 * given_to_each_friend) = 114.85 :=
by
  sorry

end remaining_money_after_expenditures_l585_585504


namespace simplex_labeling_min_n_l585_585359

-- Define the main theorem
theorem simplex_labeling_min_n (n : ℕ) (h_n : n > 0) (S : Finset (Fin 4066273)) 
  (label : Fin 4066273 → ℝ) :
  (∃ x ∈ S, label x ≠ 0) →           -- Condition (a)
  (∀ ℓ, is_parallel_line ℓ → sum (label '' (S ∩ ℓ.points)) = 0) →  -- Condition (b)
  (∀ ℓ, is_parallel_line ℓ → symmetric_along ℓ label) →            -- Condition (c)
  n = 4066273 := 
sorry

-- Supporting definitions
noncomputable def is_parallel_line (ℓ : Finset (Fin 4066273)) : Prop := sorry
noncomputable def sum (s : Finset ℝ) : ℝ := sorry
noncomputable def symmetric_along (ℓ : Finset (Fin 4066273)) (label : Fin 4066273 → ℝ) : Prop := sorry

end simplex_labeling_min_n_l585_585359


namespace high_linear_correlation_abs_r_close_to_one_l585_585310

theorem high_linear_correlation_abs_r_close_to_one
  (r : ℝ)
  (hr_range : -1 ≤ r ∧ r ≤ 1)
  (high_correlation : abs r > 0.5) :
  abs r ≈ 1 := 
sorry

end high_linear_correlation_abs_r_close_to_one_l585_585310


namespace find_n_of_extreme_value_at_neg1_l585_585639

noncomputable def f (x : ℝ) (n : ℝ) : ℝ := x^3 + 6 * x^2 + n * x + 4

theorem find_n_of_extreme_value_at_neg1 :
  ∃ n : ℝ, (∀ x : ℝ, deriv (λ x, f x n) (-1) = 0) → n = 9 :=
by
  intro n h
  have h1 : deriv (λ x, f x n) = λ x, 3 * x^2 + 12 * x + n := by
    funext
    simp [f]
    ring
  rw [h1] at h
  specialize h (-1)
  linarith

#check find_n_of_extreme_value_at_neg1

end find_n_of_extreme_value_at_neg1_l585_585639


namespace smallest_angle_in_convex_15sided_polygon_l585_585002

def isConvexPolygon (n : ℕ) (angles : Fin n → ℚ) : Prop :=
  ∑ i, angles i = (n - 2) * 180 ∧ ∀ i,  angles i < 180

def arithmeticSequence (angles : Fin 15 → ℚ) : Prop :=
  ∃ a d : ℚ, ∀ i : Fin 15, angles i = a + i * d

def increasingSequence (angles : Fin 15 → ℚ) : Prop :=
  ∀ i j : Fin 15, i < j → angles i < angles j

def integerSequence (angles : Fin 15 → ℚ) : Prop :=
  ∀ i : Fin 15, (angles i : ℚ) = angles i

theorem smallest_angle_in_convex_15sided_polygon :
  ∃ (angles : Fin 15 → ℚ),
    isConvexPolygon 15 angles ∧
    arithmeticSequence angles ∧
    increasingSequence angles ∧
    integerSequence angles ∧
    angles 0 = 135 :=
by
  sorry

end smallest_angle_in_convex_15sided_polygon_l585_585002


namespace revenue_increase_l585_585505

theorem revenue_increase (P Q : ℝ) : 
  let P_new := P * 1.30,
      Q_new := Q * 0.80,
      R := P * Q,
      R_new := P_new * Q_new
  in R_new = R * 1.04 :=
by
  let P_new := P * 1.30
  let Q_new := Q * 0.80
  let R := P * Q
  let R_new := P_new * Q_new
  sorry

end revenue_increase_l585_585505


namespace cartesian_eq_C2_length_AB_l585_585708

noncomputable def curve_C1 : Set (ℝ × ℝ) := 
  {p | ∃ α : ℝ, p.1 = 2 * Real.cos α ∧ p.2 = 2 + 2 * Real.sin α}

def point_P (M : ℝ × ℝ) : ℝ × ℝ := (2 * M.1, 2 * M.2)

noncomputable def curve_C2 : Set (ℝ × ℝ) := 
  {p | ∃ α : ℝ, p.1 = 4 * Real.cos α ∧ p.2 = 4 + 4 * Real.sin α}

theorem cartesian_eq_C2 :
  ∀ (P : ℝ × ℝ), P ∈ curve_C2 ↔ (P.1)^2 + (P.2 - 4)^2 = 16 :=
sorry

noncomputable def polar_eq_C1 : ℝ → ℝ := λ θ, 4 * Real.sin θ
noncomputable def polar_eq_C2 : ℝ → ℝ := λ θ, 8 * Real.sin θ

theorem length_AB :
  let θ := Real.pi / 3 in 
  let rho1 := polar_eq_C1 θ in
  let rho2 := polar_eq_C2 θ in
  Real.sqrt (rho2 - rho1)^2 = 2 * Real.sqrt 3 :=
sorry

end cartesian_eq_C2_length_AB_l585_585708


namespace max_abs_z_l585_585728

theorem max_abs_z 
  (a b c z : ℂ)
  (h1 : |a| = 2 * |b|)
  (h2 : |b| = |c|)
  (h3 : a + b + c ≠ 0)
  (h4 : 2 * a * z^2 + 3 * b * z + 4 * c = 0) : |z| ≤ 1 :=
sorry

end max_abs_z_l585_585728


namespace ratio_proof_l585_585873

theorem ratio_proof (X: ℕ) (h: 150 * 2 = 300 * X) : X = 1 := by
  sorry

end ratio_proof_l585_585873


namespace min_value_of_a_plus_2b_l585_585626

theorem min_value_of_a_plus_2b (a b : ℝ) (h_pos : a > 0 ∧ b > 0) (h_eq : 1 / a + 2 / b = 4) : a + 2 * b = 9 / 4 :=
by
  sorry

end min_value_of_a_plus_2b_l585_585626


namespace monotonic_intervals_f_inequality_holds_l585_585642

noncomputable def f (x a : ℝ) : ℝ := Real.log (x + a) - x

theorem monotonic_intervals_f (a : ℝ) :
  ∀ x : ℝ, 1 < x ∧ x < 2 → (f x a = Real.log (x - 1) - x ∧ f' x > 0)
  ↔ (f x a = Real.log (x - 1) - x ∧ f' x < 0) :=
sorry

theorem inequality_holds (a : ℝ) :
  (∀ x : ℝ, x ≥ 1 → e^(f x a) + (a / 2) * x^2 > 1) ↔ a > (2 * (Real.exp 1 - 1)) / ((Real.exp 1) + 2) :=
sorry

end monotonic_intervals_f_inequality_holds_l585_585642


namespace normal_dist_probability_l585_585018

-- Definitions
def xi : Type := ℝ -- The type for our random variable, which is real numbers
def ξ_pdf (x : ℝ) : ℝ := 
  1 / (5 * (real.pi ^ (1 / 2))) * real.exp (-((x - 100) ^ 2) / (2 * 5 ^ 2))

-- Given conditions
axiom H_normal : ∀ x : ℝ, pdf xi x = ξ_pdf x
axiom P_xi_110 : P(λ x, x < 110) xi = 0.98

-- The statement we need to prove
theorem normal_dist_probability :
  P (λ x, 90 < x ∧ x < 100) xi = 0.48 :=
sorry

end normal_dist_probability_l585_585018


namespace squares_lines_concurrent_l585_585805

theorem squares_lines_concurrent
  {O A B C A₁ B₁ C₁ : Type*}
  (h1 : is_square O A B C)
  (h2 : is_square O A₁ B₁ C₁)
  (h3 : same_plane O A B C O A₁ B₁ C₁)
  (h4 : directly_oriented O A B C)
  (h5 : directly_oriented O A₁ B₁ C₁) :
  concurrent (line_through A A₁) (line_through B B₁) (line_through C C₁) := 
sorry

end squares_lines_concurrent_l585_585805


namespace box_volume_l585_585884

theorem box_volume (length width side_len : ℕ) (h1 : length = 48) (h2 : width = 36) (h3 : side_len = 8) :
  (length - 2 * side_len) * (width - 2 * side_len) * side_len = 5120 := 
by
  rw [h1, h2, h3]
  norm_num
  sorry

end box_volume_l585_585884


namespace total_balloons_l585_585040

def tom_balloons : Nat := 9
def sara_balloons : Nat := 8

theorem total_balloons : tom_balloons + sara_balloons = 17 := 
by
  sorry

end total_balloons_l585_585040


namespace pentagon_largest_angle_l585_585821

theorem pentagon_largest_angle
    (P Q : ℝ)
    (hP : P = 55)
    (hQ : Q = 120)
    (R S T : ℝ)
    (hR_eq_S : R = S)
    (hT : T = 2 * R + 20):
    R + S + T + P + Q = 540 → T = 192.5 :=
by
    sorry

end pentagon_largest_angle_l585_585821


namespace counting_numbers_remainder_11_l585_585664

-- Define the conditions and the statement.

def divisors_50 : List Nat := [1, 2, 5, 10, 25, 50]

def is_valid_divisor (d : Nat) : Prop := d > 11

theorem counting_numbers_remainder_11 :
  (divisors_50.filter is_valid_divisor).length = 2 :=
by {
  sorry -- Proof is omitted
}

end counting_numbers_remainder_11_l585_585664


namespace possible_values_ceil_x_squared_l585_585670

theorem possible_values_ceil_x_squared (x : ℝ) (h : ⌈x⌉ = 9) : (finset.Icc 65 81).card = 17 := by
  sorry

end possible_values_ceil_x_squared_l585_585670


namespace probability_normal_interval_l585_585678

noncomputable def probability_of_interval (xi : ℝ → ℝ) (μ σ : ℝ) : ℝ :=
  let Z := λ x, (xi x - μ) / σ
  (cdf stdNormal (-2)) - (cdf stdNormal (-4))

theorem probability_normal_interval :
  ∀ (ξ : ℝ → ℝ) (μ σ : ℝ),
    (∀ x, ξ x ∼ Normal μ σ^2) →
    μ = 3 →
    σ^2 = 1 →
    P(-1 < ξ ≤ 1) = (cdf stdNormal (-4)) - (cdf stdNormal (-2)) :=
by
  intros ξ μ σ h_normal h_mean h_variance
  unfold P interval
  sorry

end probability_normal_interval_l585_585678


namespace tiling_impossible_l585_585050

theorem tiling_impossible :
  let board_size : ℕ := 8
  let tile1_coverage : ℕ := 4
  let tile2_coverage : ℕ := 4
  let num_tile1 : ℕ := 15
  let num_tile2 : ℕ := 1
  let board_area : ℕ := board_size * board_size
  let total_coverage : ℕ := num_tile1 * tile1_coverage + num_tile2 * tile2_coverage
  board_area = total_coverage ∧ ∃ (c : board_size → board_size → Prop), (∃ i j, ¬c i j) → False :=
by
  let board_size := 8
  let tile1_coverage := 4
  let tile2_coverage := 4
  let num_tile1 := 15
  let num_tile2 := 1
  let board_area := board_size * board_size
  let total_coverage := num_tile1 * tile1_coverage + num_tile2 * tile2_coverage
  have h1 : board_area = 64 := rfl
  have h2 : total_coverage = 64 := rfl
  have h3 : total_coverage ≤ board_area := by simp [board_area, total_coverage]
  have h4 : board_area = total_coverage := by simp [board_area, total_coverage]
  have impossible_cover : ∃ (c : board_size → board_size → Prop), (∃ i j, ¬c i j) → False := sorry
  exact impossible_cover

end tiling_impossible_l585_585050


namespace profit_without_discount_l585_585115

noncomputable def cost_price : ℝ := 100
noncomputable def profit_percentage_with_discount : ℝ := 44
noncomputable def discount : ℝ := 4

theorem profit_without_discount (CP MP SP : ℝ) (h_CP : CP = cost_price) (h_pwpd : profit_percentage_with_discount = 44) (h_discount : discount = 4) (h_SP : SP = CP * (1 + profit_percentage_with_discount / 100)) (h_MP : SP = MP * (1 - discount / 100)) :
  ((MP - CP) / CP * 100) = 50 :=
by
  sorry

end profit_without_discount_l585_585115


namespace smallest_four_digit_multiple_of_18_l585_585966

theorem smallest_four_digit_multiple_of_18 : 
  ∃ (n : ℤ), 1000 ≤ n ∧ n < 10000 ∧ 18 ∣ n ∧ 
  ∀ (m : ℤ), (1000 ≤ m ∧ m < 10000 ∧ 18 ∣ m) → n ≤ m :=
sorry

end smallest_four_digit_multiple_of_18_l585_585966


namespace regular_hexagon_area_l585_585841

theorem regular_hexagon_area (A : ℝ) (r : ℝ) (hex_area : ℝ) :
  A = 100 * Real.pi → r = Real.sqrt 100 → 
  hex_area = 150 * Real.sqrt 3 → 
  150 * Real.sqrt 3 = 150 * Real.sqrt 3 :=
by
  sorry

end regular_hexagon_area_l585_585841


namespace age_proof_l585_585782

theorem age_proof (A B C D : ℕ) 
  (h1 : A = D + 16)
  (h2 : B = D + 8)
  (h3 : C = D + 4)
  (h4 : A - 6 = 3 * (D - 6))
  (h5 : A - 6 = 2 * (B - 6))
  (h6 : A - 6 = (C - 6) + 4) 
  : A = 30 ∧ B = 22 ∧ C = 18 ∧ D = 14 :=
sorry

end age_proof_l585_585782


namespace at_least_500_friendly_integers_in_2012_a_eq_2_not_friendly_l585_585151

/-- Part (a) -/
theorem at_least_500_friendly_integers_in_2012 :
  ∃ (a_set : Set ℕ), a_set ⊆ { a | 1 ≤ a ∧ a ≤ 2012 } ∧ 
  (∀ a ∈ a_set, ∃ m n : ℕ, 0 < m ∧ 0 < n ∧ (m^2 + n) * (n^2 + m) = a * (m - n)^3) ∧ 
  Set.card a_set ≥ 500 :=
sorry

/-- Part (b) -/
theorem a_eq_2_not_friendly :
  ¬ ∃ m n : ℕ, 0 < m ∧ 0 < n ∧ (m^2 + n) * (n^2 + m) = 2 * (m - n)^3 :=
sorry

end at_least_500_friendly_integers_in_2012_a_eq_2_not_friendly_l585_585151


namespace limit_expression_l585_585679

noncomputable theory

def a_n (n : ℕ) : ℝ := 3^n
def b_n (n : ℕ) : ℝ := 2^n

theorem limit_expression : 
  filter.tendsto (λ n, (b_n (n + 1) - a_n n) / (a_n (n + 1) + b_n n)) filter.at_top (nhds (-1 / 3)) :=
by sorry

end limit_expression_l585_585679


namespace candies_eaten_l585_585173

theorem candies_eaten (A B D : ℕ) 
                      (h1 : 4 * B = 3 * A) 
                      (h2 : 7 * A = 6 * D) 
                      (h3 : A + B + D = 70) :
  A = 24 ∧ B = 18 ∧ D = 28 := 
by
  sorry

end candies_eaten_l585_585173


namespace log_equation_solution_l585_585588

theorem log_equation_solution (x : ℝ) (h : log x 8 = 2 / 3) : x = 8 :=
sorry

end log_equation_solution_l585_585588


namespace class_average_score_l585_585570

theorem class_average_score :
  let total_students := 40
  let absent_students := 2
  let present_students := total_students - absent_students
  let initial_avg := 92
  let absent_scores := [100, 100]
  let initial_total_score := initial_avg * present_students
  let total_final_score := initial_total_score + absent_scores.sum
  let final_avg := total_final_score / total_students
  final_avg = 92.4 := by
  sorry

end class_average_score_l585_585570


namespace increase_avg_grade_transfer_l585_585474

-- Definitions for initial conditions
def avg_grade_A_initial := 44.2
def avg_grade_B_initial := 38.8
def num_students_A_initial := 10
def num_students_B_initial := 10

def grade_kalinina := 41
def grade_sidorov := 44

-- Definitions for expected conditions after transfer
def sum_grades_A_initial := avg_grade_A_initial * num_students_A_initial
def sum_grades_B_initial := avg_grade_B_initial * num_students_B_initial

-- Verify the transfer condition will meet the requirements
theorem increase_avg_grade_transfer : 
  let sum_grades_A_after := sum_grades_A_initial - grade_kalinina - grade_sidorov in
  let sum_grades_B_after := sum_grades_B_initial + grade_kalinina + grade_sidorov in
  let num_students_A_after := num_students_A_initial - 2 in
  let num_students_B_after := num_students_B_initial + 2 in
  let avg_grade_A_after := sum_grades_A_after / num_students_A_after in
  let avg_grade_B_after := sum_grades_B_after / num_students_B_after in

  avg_grade_A_after > avg_grade_A_initial ∧ avg_grade_B_after > avg_grade_B_initial :=
by
  sorry

end increase_avg_grade_transfer_l585_585474


namespace total_rainfall_in_Springdale_l585_585937

theorem total_rainfall_in_Springdale
    (rainfall_first_week rainfall_second_week : ℝ)
    (h1 : rainfall_second_week = 1.5 * rainfall_first_week)
    (h2 : rainfall_second_week = 12) :
    (rainfall_first_week + rainfall_second_week = 20) :=
by
  sorry

end total_rainfall_in_Springdale_l585_585937


namespace monotone_increasing_implies_range_a_l585_585645

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^2 + a / x

theorem monotone_increasing_implies_range_a (a : ℝ) :
  (∀ x : ℝ, 2 ≤ x → (2 * x^3 - a) / x^2 ≥ 0) →
  a ≤ 16 :=
by
  intro h
  specialize h 2
  linarith
  sorry

end monotone_increasing_implies_range_a_l585_585645


namespace y_intercept_is_neg2_l585_585618

noncomputable def find_y_intercept_of_perpendicular_line : ℝ :=
  let l := {a : ℝ × ℝ // a.1 - 2 * a.2 + 1 = 0} in
  let point := (-1, 0 : ℝ × ℝ) in
  let m_slope := -2 in
  let m := {b : ℝ × ℝ // b.2 = m_slope * (b.1 + 1) - 2} in
  match (0 : ℝ × ℝ) with
  | (x, y) := y + 2

theorem y_intercept_is_neg2 :
  find_y_intercept_of_perpendicular_line = -2 :=
sorry

end y_intercept_is_neg2_l585_585618


namespace students_after_joining_l585_585420

theorem students_after_joining (N : ℕ) (T : ℕ)
  (h1 : T = 48 * N)
  (h2 : 120 * 32 / (N + 120) + (T / (N + 120)) = 44)
  : N + 120 = 480 :=
by
  sorry

end students_after_joining_l585_585420


namespace cyclist_wait_20_minutes_l585_585791

noncomputable def cyclist_wait_time 
  (hiker_speed : ℝ) (cyclist_speed : ℝ) (time_passed_minutes : ℝ) : ℝ :=
  let time_passed_hours := time_passed_minutes / 60
  let distance := cyclist_speed * time_passed_hours
  let hiker_catch_up_time := distance / hiker_speed
  hiker_catch_up_time * 60

theorem cyclist_wait_20_minutes :
  cyclist_wait_time 5 20 5 = 20 :=
by
  -- Definitions according to given conditions
  let hiker_speed := 5 -- miles per hour
  let cyclist_speed := 20 -- miles per hour
  let time_passed_minutes := 5
  -- Required result
  let result_needed := 20
  -- Using the cyclist_wait_time function
  show cyclist_wait_time hiker_speed cyclist_speed time_passed_minutes = result_needed
  sorry

end cyclist_wait_20_minutes_l585_585791


namespace even_three_digit_numbers_less_than_600_l585_585827

def count_even_three_digit_numbers : ℕ :=
  let hundreds_choices := 5
  let tens_choices := 6
  let units_choices := 3
  hundreds_choices * tens_choices * units_choices

theorem even_three_digit_numbers_less_than_600 : count_even_three_digit_numbers = 90 := by
  -- sorry ensures that the statement type checks even without the proof.
  sorry

end even_three_digit_numbers_less_than_600_l585_585827


namespace existence_of_three_disjoint_tilings_l585_585908

open Classical

noncomputable section

variables {B W : Type*} [Infinite B] [Infinite W] (G : Graph B W)

-- Define that G is 4-regular and bipartite
def four_regular : Prop :=
  ∀ v ∈ B ∪ W, G.degree v = 4

variables (M M_1 M_2 M_3 : Set (B × W))

-- M is a perfect matching
def perfect_matching (M : Set (B × W)) : Prop :=
  ∀ v ∈ B ∪ W, ∃! w, (v, w) ∈ M ∨ (w, v) ∈ M

-- Partitions of M, M_1, M_2, M_3 do not overlap with M or each other
def disjoint_matchings : Prop :=
  (M ∩ M_1 = ∅) ∧ (M ∩ M_2 = ∅) ∧ (M ∩ M_3 = ∅) ∧
  (M_1 ∩ M_2 = ∅) ∧ (M_1 ∩ M_3 = ∅) ∧ (M_2 ∩ M_3 = ∅)

-- Problem statement
theorem existence_of_three_disjoint_tilings (hG : four_regular G) (hM : perfect_matching G M) :
  ∃ M_1 M_2 M_3, perfect_matching G M_1 ∧ perfect_matching G M_2 ∧ perfect_matching G M_3 ∧ disjoint_matchings M M_1 M_2 M_3 := 
sorry

end existence_of_three_disjoint_tilings_l585_585908


namespace open_box_volume_l585_585887

theorem open_box_volume (l w s : ℝ) (hl : l = 48) (hw : w = 36) (hs : s = 8) :
  (l - 2 * s) * (w - 2 * s) * s = 5120 :=
by
  sorry

end open_box_volume_l585_585887


namespace smallest_n_l585_585849

def is_divisible (a b : ℕ) : Prop := ∃ k : ℕ, b * k = a

def meets_condition (n : ℕ) : Prop :=
  n > 0 ∧
  ∃ k : ℕ, 1 ≤ k ∧ k ≤ n + 1 ∧ is_divisible (n^2 - n + 1) k ∧
  ∃ l : ℕ, 1 ≤ l ∧ l ≤ n + 1 ∧ ¬ is_divisible (n^2 - n + 1) l

theorem smallest_n : ∃ n : ℕ, meets_condition n ∧ n = 5 :=
by
  sorry

end smallest_n_l585_585849


namespace closest_total_population_of_cities_l585_585701

theorem closest_total_population_of_cities 
    (n_cities : ℕ) (avg_population_lower avg_population_upper : ℕ)
    (h_lower : avg_population_lower = 3800) (h_upper : avg_population_upper = 4200) :
  (25:ℕ) * (4000:ℕ) = 100000 :=
by
  sorry

end closest_total_population_of_cities_l585_585701


namespace solve_for_x_l585_585407

theorem solve_for_x (x : ℝ) (h : 64^(3 * x) = 16^(4 * x - 5)) : x = -10 := 
by
  sorry

end solve_for_x_l585_585407


namespace problem_1_unit_vector_problem_2_acute_angle_l585_585254

variables (a b : ℝ × ℝ) (λ : ℝ)

def unit_vector_of_a (a : ℝ × ℝ) : Prop :=
  a = (real.sqrt 3, 1) → (∃ a0 : ℝ × ℝ, 
    a0 = (real.sqrt 3 / 2, 1 / 2) ∨ a0 = (- real.sqrt 3 / 2, - 1 / 2))

def acute_angle_range (a b : ℝ × ℝ) (λ : ℝ) : Prop :=
  a = (real.sqrt 3, 1) → b = (0, 1) →
  ((a.1 + λ * b.1, a.2 + λ * b.2) ∈ set.Ioi (0 : ℝ) →
   (a.1 - λ * b.1, a.2 - λ * b.2) ∈ set.Ioi (0 : ℝ) →
   ∀ λ, λ ^ 2 < 4 ∧ λ ≠ 0 → λ ∈ set.Ioo (-2) 0 ∪ set.Ioo 0 2 )

theorem problem_1_unit_vector : unit_vector_of_a (real.sqrt 3, 1) :=
by
  intro ha
  use ((real.sqrt 3 / 2), (1 / 2))
  -- ∃ a0 : ℝ × ℝ, a0 = (real.sqrt 3 / 2, 1 / 2) ∨ a0 = (- real.sqrt 3 / 2, -1 / 2)
  constructor
  · left
    rfl
  
  -- sorry (omitted steps)
  sorry

theorem problem_2_acute_angle (λ : ℝ) : acute_angle_range (real.sqrt 3, 1) (0, 1) λ :=
by
  intros ha hb h1 h2
  -- ((√3)+λ(0), (1)+λ(1)) ∈ (0,∞) → ((√3)-λ(0), (1)-λ(1)) ∈ (0,∞) → λ²<4 ∧ λ≠0 → λ∈ (-2,0)∪(0,2)
  constructor
  · intro h
    exact λ ^ 2 < 4
    -- sorry (omitted steps)
    sorry

  -- sorry (omitted steps)
  sorry

end problem_1_unit_vector_problem_2_acute_angle_l585_585254


namespace olivia_paper_count_l585_585391

-- State the problem conditions and the final proof statement.
theorem olivia_paper_count :
  let math_initial := 220
  let science_initial := 150
  let math_used := 95
  let science_used := 68
  let math_received := 30
  let science_given := 15
  let math_remaining := math_initial - math_used + math_received
  let science_remaining := science_initial - science_used - science_given
  let total_pieces := math_remaining + science_remaining
  total_pieces = 222 :=
by
  -- Placeholder for the proof
  sorry

end olivia_paper_count_l585_585391


namespace arrangement_exists_no_circular_arrangement_l585_585399
open Nat List

def isPerfectSquare (n : ℕ) : Prop := ∃ m, m * m = n

noncomputable def check_adjacent_sum (l : List ℕ) (condition : (ℕ → ℕ → Prop)) : Prop :=
  ∀ i, i < l.length - 1 → condition (l.get ⟨i, Nat.lt_pred_of_lt (Nat.lt_pred_self (l.length_pos_of_ne_nil _))⟩) (l.get ⟨i + 1, Nat.lt_of_succ_lt (Nat.lt_pred_self (l.length_pos_of_ne_nil _))⟩)

theorem arrangement_exists :
  ∃ (l : List ℕ), l = [16, 9, 7, 2, 14, 11, 5, 4, 12, 13, 3, 6, 10, 15, 1, 8] ∧ check_adjacent_sum l (λ x y => isPerfectSquare (x + y)) :=
sorry

theorem no_circular_arrangement :
  ¬ ∃ (l : List ℕ), l.length = 16 ∧ {l.head} = {l.last} ∧ check_adjacent_sum l (λ x y => isPerfectSquare (x + y)) :=
sorry

end arrangement_exists_no_circular_arrangement_l585_585399


namespace increase_average_grades_l585_585467

theorem increase_average_grades (XA XB : ℝ) (nA nB : ℕ) (k s : ℝ) 
    (hXA : XA = 44.2) (hXB : XB = 38.8) 
    (hnA : nA = 10) (hnB : nB = 10) 
    (hk : k = 41) (hs : s = 44) :
    let new_XA := (XA * nA - k - s) / (nA - 2)
    let new_XB := (XB * nB + k + s) / (nB + 2) in
    (new_XA > XA) ∧ (new_XB > XB) := by
  sorry

end increase_average_grades_l585_585467


namespace double_sum_equals_factorial_l585_585761

theorem double_sum_equals_factorial (n : ℕ) :
  (∑ i in Finset.range (n + 1), ∑ j in Finset.range (n + 1), (-1 : ℤ)^(i + j) * Nat.choose n i * Nat.choose n j * n^(i - j) * (j : ℤ)^i) = n! :=
by sorry

end double_sum_equals_factorial_l585_585761


namespace not_algorithm_l585_585150

theorem not_algorithm (A: string) (B: string) (D: string) (C: string) (hA: A = "To travel from Jinan to Beijing, first take a train, then take a plane to arrive") 
  (hB: B = "The steps to solve a linear equation with one variable are to eliminate the denominator, remove brackets, transpose terms, combine like terms, and make the coefficient 1")
  (hD: D = "To find the value of 1 + 2 + 3 + 4 + 5, first calculate 1 + 2 = 3, then 3 + 3 = 6, 6 + 4 = 10, 10 + 5 = 15, resulting in a final value of 15")
  (hC: C = "The equation x^2 - 1 = 0 has two real roots") : 
  C ≠ "An algorithm for solving a problem" := by sorry

end not_algorithm_l585_585150


namespace smallest_four_digit_multiple_of_18_l585_585968

theorem smallest_four_digit_multiple_of_18 : ∃ n : ℕ, 1000 ≤ n ∧ n ≤ 9999 ∧ n % 18 = 0 ∧ 
  ∀ m : ℕ, 1000 ≤ m ∧ m ≤ 9999 ∧ m % 18 = 0 → n ≤ m :=
begin
  use 1008,
  split,
  { exact nat.le_refl 1008, },
  split,
  { exact nat.le_succ 9999, },
  split,
  { exact nat.mod_eq_zero_of_divisible 1008 18 sorry, },
  { intros m h1 h2 h3,
    apply nat.le_of_sub_nonneg,
    sorry, }
end

end smallest_four_digit_multiple_of_18_l585_585968


namespace find_m_n_sum_l585_585656

theorem find_m_n_sum (m n : ℝ) :
  ( ∀ x, -3 < x ∧ x < 6 → x^2 - m * x - 6 * n < 0 ) →
  m + n = 6 :=
by
  sorry

end find_m_n_sum_l585_585656


namespace problem1_problem2_l585_585275

open Real

def f (x a : ℝ) : ℝ := abs (x + a) + abs (x + 2)

theorem problem1 (a : ℝ) (h : a = -1) : { x | f x a ≥ x + 5 } = { x | x ≤ -2 ∨ x ≥ 4 } :=
by
  sorry

theorem problem2 (a : ℝ) (h : a < 2) : (∀ x, x ∈ Ioo (-5 : ℝ) (-3 : ℝ) → f x a > x^2 + 2*x - 5) ↔ a ≤ -2 :=
by
  sorry

end problem1_problem2_l585_585275


namespace simplify_fractions_l585_585403

theorem simplify_fractions :
  (36 / 51) * (35 / 24) * (68 / 49) = (20 / 7) :=
by
  have h1 : 36 = 2^2 * 3^2 := by norm_num
  have h2 : 51 = 3 * 17 := by norm_num
  have h3 : 35 = 5 * 7 := by norm_num
  have h4 : 24 = 2^3 * 3 := by norm_num
  have h5 : 68 = 2^2 * 17 := by norm_num
  have h6 : 49 = 7^2 := by norm_num
  sorry

end simplify_fractions_l585_585403


namespace Linda_needs_15_hours_to_cover_fees_l585_585744

def wage : ℝ := 10
def fee_per_college : ℝ := 25
def number_of_colleges : ℝ := 6

theorem Linda_needs_15_hours_to_cover_fees :
  (number_of_colleges * fee_per_college) / wage = 15 := by
  sorry

end Linda_needs_15_hours_to_cover_fees_l585_585744


namespace exists_convex_polyhedron_with_short_diagonals_l585_585578

theorem exists_convex_polyhedron_with_short_diagonals :
  ∃ (P : Type) [convex_polyhedron P],
    ∀ (d : diagonal P) (e : edge P), length d < length e :=
by
  let ABC : triangle {a : ℝ // 0 ≤ a ∧ a < 1}
  let S1 S2 : point {a : ℝ // 0 ≤ a ∧ a < 1}
  -- Define symmetric construction and properties of points S1 and S2
  -- Further details would be in a proper proof
  have h_S1A_lt_1 : ∀ p ∈ vertices ABC, dist S1 p < 1 := sorry
  have h_S2A_lt_1 : ∀ p ∈ vertices ABC, dist S2 p < 1 := sorry
  let P : convex_polyhedron := convex_hull {ABC, S1, S2} -- Simplified
  exists P
  intros d e
  sorry

end exists_convex_polyhedron_with_short_diagonals_l585_585578


namespace stuart_segments_return_l585_585777

theorem stuart_segments_return (r1 r2 : ℝ) (tangent_chord : ℝ)
  (angle_ABC : ℝ) (h1 : r1 < r2) (h2 : tangent_chord = r1 * 2)
  (h3 : angle_ABC = 75) :
  ∃ (n : ℕ), n = 24 ∧ tangent_chord * n = 360 * (n / 24) :=
by {
  sorry
}

end stuart_segments_return_l585_585777


namespace johns_bakery_fraction_l585_585189

theorem johns_bakery_fraction :
  ∀ (M : ℝ), 
  (M / 4 + M / 3 + 6 + (24 - (M / 4 + M / 3 + 6)) = 24) →
  (24 : ℝ) = M →
  (4 + 8 + 6 = 18) →
  (24 - 18 = 6) →
  (6 / 24 = (1 / 6 : ℝ)) :=
by
  intros M h1 h2 h3 h4
  sorry

end johns_bakery_fraction_l585_585189


namespace sam_distance_l585_585748

theorem sam_distance (miles_marguerite : ℕ) (hours_marguerite : ℕ) (hours_sam : ℕ) 
  (speed_increase : ℚ) (avg_speed_marguerite : ℚ) (speed_sam : ℚ) (distance_sam : ℚ) :
  miles_marguerite = 120 ∧ hours_marguerite = 3 ∧ hours_sam = 4 ∧ speed_increase = 1.20 ∧
  avg_speed_marguerite = miles_marguerite / hours_marguerite ∧ 
  speed_sam = avg_speed_marguerite * speed_increase ∧
  distance_sam = speed_sam * hours_sam →
  distance_sam = 192 :=
by
  intros h
  sorry

end sam_distance_l585_585748


namespace measurable_right_continuous_and_linear_l585_585051

-- Define the theorem
theorem measurable_right_continuous_and_linear (f : ℝ → ℝ) :
  (measurable f) →
  (∀ (x h : ℝ), 0 ≤ x ∧ 0 ≤ h → x + h ≤ 1 → f(x + h) = f(x) + f(h)) →
  (continuous f at_right 0) ∧ (∀ x, 0 ≤ x → x ≤ 1 → f(x) = f(1) * x)
  :=
begin
  -- Proof steps go here
  sorry
end

end measurable_right_continuous_and_linear_l585_585051


namespace children_too_heavy_l585_585897

def Kelly_weight : ℝ := 34
def Sam_weight : ℝ := 40
def Daisy_weight : ℝ := 28
def Megan_weight := 1.1 * Kelly_weight
def Mike_weight := Megan_weight + 5

def Total_weight := Kelly_weight + Sam_weight + Daisy_weight + Megan_weight + Mike_weight
def Bridge_limit : ℝ := 130

theorem children_too_heavy :
  Total_weight - Bridge_limit = 51.8 :=
by
  sorry

end children_too_heavy_l585_585897


namespace second_highest_coefficient_l585_585797

def g (x : ℝ) : ℝ := sorry

theorem second_highest_coefficient :
  (∀ x : ℝ, g (x + 1) - g x = 6 * x^2 + 4 * x + 2) →
  (∃ a b c d : ℝ, g x = (a / 3) * x^3 + (b / 3) * x^2 + (c / 2) * x + d ∧ b = 2) :=
begin
  sorry,
end

end second_highest_coefficient_l585_585797


namespace sum_possible_m_values_l585_585136

theorem sum_possible_m_values : 
  let m_values := {m : ℕ | 4 < m ∧ m < 18}
  let sum_m := ∑ m in m_values, m
  sum_m = 143 :=
by
  sorry

end sum_possible_m_values_l585_585136


namespace max_value_magnitude_l585_585296
noncomputable theory

variables (a b c : ℝ^3)
variable [inner_product_space ℝ ℝ^3]

-- Conditions
def unit_vector (v : ℝ^3) : Prop := ‖v‖ = 1
def orthogonal (x y : ℝ^3) : Prop := ⟪x, y⟫ = 0
def condition3 (a b c : ℝ^3) : Prop := ⟪a - c, b - c⟫ ≤ 0

-- Proof statement
theorem max_value_magnitude (h1 : unit_vector a) (h2 : unit_vector b) (h3 : unit_vector c)
  (h4 : orthogonal a b) (h5 : condition3 a b c) : 
  ∃ M, ∀ v, v = a + b - c → ‖v‖ ≤ M ∧ M = 1 :=
  sorry

end max_value_magnitude_l585_585296


namespace min_diameter_of_covering_circles_correct_l585_585306

noncomputable def min_diameter_of_covering_circles (side_length : ℝ := 1) : ℝ := 
  let radius := (Real.sqrt 65) / 8 in
  radius * 2

theorem min_diameter_of_covering_circles_correct :
  min_diameter_of_covering_circles 1 = (Real.sqrt 65) / 4 :=
by
  sorry

end min_diameter_of_covering_circles_correct_l585_585306


namespace digits_ways_to_form_number_l585_585326

open List

def is_divisible_by_4 (n : Nat) : Bool :=
  n % 4 = 0

def valid_seven_digit_numbers (digits : List Nat) : List (List Nat) :=
  digits.permutations.filter (λ l, head l ≠ 0 ∧ is_divisible_by_4 ((l.reverse.headD 0) * 10 + (l.reverse.tail.headD 0)))

theorem digits_ways_to_form_number :
  let digits := [0, 1, 2, 3, 4, 5, 6]
  length (valid_seven_digit_numbers digits) = 1248 :=
sorry

end digits_ways_to_form_number_l585_585326


namespace average_speed_correct_l585_585014

-- Define the problem conditions
def total_distance : ℝ := 160
def total_time : ℝ := 6

-- Define the expected average speed
def expected_average_speed : ℝ := 80 / 3

-- The main statement to be proven
theorem average_speed_correct : (total_distance / total_time) = expected_average_speed := by 
  sorry

end average_speed_correct_l585_585014


namespace initials_count_l585_585289

theorem initials_count : 
  let letters := finset.range 10 in 
  let num_first := letters.card in
  let num_second := (letters.erase 0).card in
  let num_third := (letters.erase 0).erase 1).card in
  num_first * num_second * num_third = 720 :=
begin
  sorry
end

end initials_count_l585_585289


namespace find_f_of_5_l585_585256

noncomputable def f : ℝ → ℝ := λ x, if x ∈ (Set.Icc (-2 : ℝ) 0) then -2 ^ x else sorry

def even_function (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)
def periodic_function (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

theorem find_f_of_5 :
  even_function f → periodic_function f 4 → (∀ x ∈ (Set.Icc (-2 : ℝ) 0), f x = -2 ^ x) → f 5 = - 1 / 2 :=
by
  intros h_even h_period h_def
  sorry

end find_f_of_5_l585_585256


namespace lacustrine_glacial_monoliths_count_l585_585216

-- Definitions based on the problem conditions
def total_volume : ℕ := 98
def sand_probability : ℚ := 1 / 7
def marine_loam_probability : ℚ := 9 / 14

-- Statement: Prove number of lacustrine-glacial genesis monoliths is 44
theorem lacustrine_glacial_monoliths_count :
  ∀ (S L_m L_lg : ℕ), 
    S = (sand_probability * total_volume).natCeil →
    L_m = (marine_loam_probability * (total_volume - S)).natFloor →
    L_lg = (total_volume - S - L_m) →
    S + L_lg = 44 :=
by
  -- This is just the statement, so we put sorry to complete it
  sorry

end lacustrine_glacial_monoliths_count_l585_585216


namespace time_to_travel_is_correct_l585_585513
noncomputable def time_to_travel : ℝ :=
  let radius : ℝ := 25
  let period : ℝ := 80
  let height : ℝ := 15
  let amplitude := radius
  let vertical_shift := radius
  let B := (2 * Real.pi) / period
  let f := λ x => amplitude * Real.cos(B * x) + vertical_shift
  let x := Real.acos(-2/5) / B
  x

-- The theorem to prove that the time taken is 80/3 seconds.
theorem time_to_travel_is_correct :
  time_to_travel = 80 / 3 :=
begin
  -- skipping the proof
  sorry
end

end time_to_travel_is_correct_l585_585513


namespace integer_values_of_f_l585_585727

theorem integer_values_of_f (a b : ℕ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_neq : a * b ≠ 1) : 
  ∃ k ∈ ({4, 7} : Finset ℕ), 
    (a^2 + b^2 + a * b) / (a * b - 1) = k := 
by
  sorry

end integer_values_of_f_l585_585727


namespace fewer_apples_per_day_l585_585747

noncomputable def apples_per_box := 40
noncomputable def boxes_per_day := 50
noncomputable def normal_days_per_week := 7
noncomputable def total_apples_two_weeks := 24500

def normal_apples_per_day : ℕ := apples_per_box * boxes_per_day
def normal_week_apples : ℕ := normal_apples_per_day * normal_days_per_week
def second_week_apples : ℕ := total_apples_two_weeks - normal_week_apples
def second_week_apples_per_day : ℕ := second_week_apples / normal_days_per_week

theorem fewer_apples_per_day :
  normal_apples_per_day - second_week_apples_per_day = 500 :=
by
  sorry

end fewer_apples_per_day_l585_585747


namespace solve_for_x_l585_585410

theorem solve_for_x (x : ℝ) : 64^(3 * x) = 16^(4 * x - 5) → x = -10 := 
by 
  sorry

end solve_for_x_l585_585410


namespace smallest_four_digit_multiple_of_18_l585_585972

theorem smallest_four_digit_multiple_of_18 : ∃ n : ℕ, 1000 ≤ n ∧ n ≤ 9999 ∧ n % 18 = 0 ∧ 
  ∀ m : ℕ, 1000 ≤ m ∧ m ≤ 9999 ∧ m % 18 = 0 → n ≤ m :=
begin
  use 1008,
  split,
  { exact nat.le_refl 1008, },
  split,
  { exact nat.le_succ 9999, },
  split,
  { exact nat.mod_eq_zero_of_divisible 1008 18 sorry, },
  { intros m h1 h2 h3,
    apply nat.le_of_sub_nonneg,
    sorry, }
end

end smallest_four_digit_multiple_of_18_l585_585972


namespace problem_1_problem_2_l585_585282

-- Definitions and conditions for the problems
def A : Set ℝ := { x | abs (x - 2) < 3 }
def B (m : ℝ) : Set ℝ := { x | x^2 - 2 * x - m < 0 }

-- Problem (I)
theorem problem_1 : (A ∩ (Set.univ \ B 3)) = { x | 3 ≤ x ∧ x < 5 } :=
sorry

-- Problem (II)
theorem problem_2 (m : ℝ) : (A ∩ B m = { x | -1 < x ∧ x < 4 }) → m = 8 :=
sorry

end problem_1_problem_2_l585_585282


namespace train_length_l585_585526

theorem train_length : 
    ∀ (jogger_speed_kmh train_speed_kmh : ℝ) (initial_gap time_in_seconds : ℕ), 
    jogger_speed_kmh = 9 → 
    train_speed_kmh = 45 → 
    initial_gap = 120 → 
    time_in_seconds = 24 → 
    let jogger_speed := jogger_speed_kmh * (1000 / 3600) in
    let train_speed := train_speed_kmh * (1000 / 3600) in
    let relative_speed := train_speed - jogger_speed in
    let total_distance := relative_speed * ↑time_in_seconds in
    let train_length := total_distance - initial_gap in
    train_length = 120 := 
by
    intros jogger_speed_kmh train_speed_kmh initial_gap time_in_seconds
    intros h1 h2 h3 h4
    dsimp only
    rw [h1, h2, h3, h4]
    norm_num
    sorry

end train_length_l585_585526


namespace pens_purchased_is_30_l585_585098

def num_pens_purchased (cost_total: ℕ) 
                       (num_pencils: ℕ) 
                       (price_per_pencil: ℚ) 
                       (price_per_pen: ℚ)
                       (expected_pens: ℕ): Prop :=
   let cost_pencils := num_pencils * price_per_pencil
   let cost_pens := cost_total - cost_pencils
   let num_pens := cost_pens / price_per_pen
   num_pens = expected_pens

theorem pens_purchased_is_30 : num_pens_purchased 630 75 2.00 16 30 :=
by
  -- Unfold the definition manually if needed
  sorry

end pens_purchased_is_30_l585_585098


namespace candies_eaten_l585_585171

theorem candies_eaten (A B D : ℕ) 
                      (h1 : 4 * B = 3 * A) 
                      (h2 : 7 * A = 6 * D) 
                      (h3 : A + B + D = 70) :
  A = 24 ∧ B = 18 ∧ D = 28 := 
by
  sorry

end candies_eaten_l585_585171


namespace deductive_reasoning_example_l585_585868

-- Definitions for the conditions
def Metal (x : Type) : Prop := sorry
def ConductsElectricity (x : Type) : Prop := sorry
def Iron : Type := sorry

-- The problem statement
theorem deductive_reasoning_example (H1 : ∀ x, Metal x → ConductsElectricity x) (H2 : Metal Iron) : ConductsElectricity Iron :=
by sorry

end deductive_reasoning_example_l585_585868


namespace Issac_pen_pencil_multiplicity_l585_585713

theorem Issac_pen_pencil_multiplicity :
  ∃ M : ℕ,
    let P := 16 in
    let L := 108 - P in
    L = M * P + 12 ∧ M = 5 :=
by
  sorry

end Issac_pen_pencil_multiplicity_l585_585713


namespace a_2023_eq_2_l585_585457

-- Define the sequence according to the problem statement
def a_seq : ℕ → ℚ
| 0 := 2                       -- Here a_0 corresponds to a_1 in the problem since Lean sequences are 0-indexed
| (n+1) := if (n+1) % 2 = 0 then 1 / a_seq n else 1 / (1 - a_seq n)

theorem a_2023_eq_2 : a_seq 2022 = 2 :=   -- Lean sequences are 0-indexed hence a_2023 corresponds to a_seq (2023-1)
by 
  sorry

end a_2023_eq_2_l585_585457


namespace max_n_color_grid_l585_585524

theorem max_n_color_grid (n : ℕ) : (∀ (grid : ℕ → ℕ → bool), (∀ (a b c d : ℕ), a < b → c < d → ¬ (grid a c = grid a d ∧ grid a c = grid b c ∧ grid a c = grid b d)) → n ≤ 5) := 
sorry

end max_n_color_grid_l585_585524


namespace square_side_length_l585_585489

-- Definition of the problem (statements)
theorem square_side_length (A : ℝ) (s : ℝ) (h : A = s * s) (hA : A = 49) : s = 7 := 
by 
  sorry

end square_side_length_l585_585489


namespace find_a_l585_585661

open Real

def are_perpendicular (l1 l2 : Real × Real × Real) : Prop :=
  let (a1, b1, c1) := l1
  let (a2, b2, c2) := l2
  a1 * a2 + b1 * b2 = 0

theorem find_a (a : Real) :
  let l1 := (a + 2, 1 - a, -1)
  let l2 := (a - 1, 2 * a + 3, 2)
  are_perpendicular l1 l2 → a = 1 ∨ a = -1 :=
by
  intro h
  sorry

end find_a_l585_585661


namespace segment_length_l585_585058

theorem segment_length (x : ℝ) (y : ℝ) (u : ℝ) (v : ℝ) :
  (|x - u| = 5 ∧ |y - u| = 5 ∧ u = √[3]{27} ∧ v = √[3]{27}) → (|x - y| = 10) :=
by
  sorry

end segment_length_l585_585058


namespace find_ellipse_equation_find_line_equation_l585_585919

noncomputable def ellipse_equation (a b : ℝ) : Prop :=
  (a > b) ∧ (b > 0) ∧ (ab = 2 * Real.sqrt 3) ∧ (Real.sqrt (1 - (b^2 / a^2)) = Real.sqrt 6 / 3)

theorem find_ellipse_equation {a b : ℝ} (h : ellipse_equation a b) :
  (a = Real.sqrt 6) ∧ (b = Real.sqrt 2) ∧ (∀ x y, (x^2 / 6) + (y^2 / 2) = 1) :=
  sorry

noncomputable def ellipse_focus_and_line (a b k : ℝ) : Prop :=
  ellipse_equation a b ∧ (∀ P : ℝ, P = 3) ∧ 
  (∀ y, y = k * (x - 2)) ∧ 
  (x = 2) ∧ (∀ x y, (x^2 / 6) + ((k * (x - 2))^2 / 2) = 1)

theorem find_line_equation {a b k : ℝ} (h : ellipse_focus_and_line a b k) :
  (k = 1 ∨ k = -1) ∧ (∀ x y, (x - y - 2 = 0) ∨ (x + y - 2 = 0)) :=
  sorry

end find_ellipse_equation_find_line_equation_l585_585919


namespace log_linear_intersect_unique_l585_585210

theorem log_linear_intersect_unique :
  ∃! x : ℝ, x > 0 ∧  3 * log x = 3 * x - 9 :=
by
  sorry

end log_linear_intersect_unique_l585_585210


namespace sum_set_cardinality_l585_585086

-- Define sets A and B
def A : Set Int := {-1, 1}
def B : Set Int := {0, 2}

-- Define the set of all unique sums of elements from A and B
def sum_set : Set Int := {z | ∃ x ∈ A, ∃ y ∈ B, z = x + y}

-- State the theorem we want to prove
theorem sum_set_cardinality : sum_set.card = 3 := by
  -- proof goes here, but we'll skip it with 'sorry'
  sorry

end sum_set_cardinality_l585_585086


namespace g_func_eq_l585_585435

theorem g_func_eq (g : ℝ → ℝ) 
  (h1 : ∀ x y : ℝ, 0 < x → 0 < y → g (x / y) = y * g x)
  (h2 : g 50 = 10) :
  g 25 = 20 :=
sorry

end g_func_eq_l585_585435


namespace problem_statement_l585_585606

theorem problem_statement (a b : ℝ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_sum : a + b = 2) :
(a + b = 2) ∧ ¬( (a^2 + a > 2) ∧ (b^2 + b > 2) ) := by
  sorry

end problem_statement_l585_585606


namespace friend_pays_correct_percentage_l585_585347

theorem friend_pays_correct_percentage (adoption_fee : ℝ) (james_payment : ℝ) (friend_payment : ℝ) 
  (h1 : adoption_fee = 200) 
  (h2 : james_payment = 150)
  (h3 : friend_payment = adoption_fee - james_payment) : 
  (friend_payment / adoption_fee) * 100 = 25 :=
by
  sorry

end friend_pays_correct_percentage_l585_585347


namespace sum_triangle_inequality_solutions_l585_585140

theorem sum_triangle_inequality_solutions :
  (∑ m in (Finset.Icc 5 17), m) = 143 :=
by
  sorry

end sum_triangle_inequality_solutions_l585_585140


namespace star_angle_of_regular_polygon_l585_585112

theorem star_angle_of_regular_polygon (n : ℕ) (h : n > 6) : 
  let internal_angle := (n - 2) * 180 / n in
  let external_angle := 180 - internal_angle in
  let angle_at_star_point := 360 - 3 * external_angle in
  angle_at_star_point = (n - 3) * 360 / n :=
by
  sorry

end star_angle_of_regular_polygon_l585_585112


namespace box_volume_l585_585886

theorem box_volume (length width side_len : ℕ) (h1 : length = 48) (h2 : width = 36) (h3 : side_len = 8) :
  (length - 2 * side_len) * (width - 2 * side_len) * side_len = 5120 := 
by
  rw [h1, h2, h3]
  norm_num
  sorry

end box_volume_l585_585886


namespace divide_square_into_smaller_squares_l585_585771

-- Definition of the property P(n)
def P (n : ℕ) : Prop := ∃ (f : ℕ → ℕ), ∀ i, i < n → (f i > 0)

-- Proposition for the problem
theorem divide_square_into_smaller_squares (n : ℕ) (h : n > 5) : P n :=
sorry

end divide_square_into_smaller_squares_l585_585771


namespace correct_distribution_l585_585563

noncomputable def ways_to_distribute_items (items : ℕ) (bags : ℕ) : ℕ :=
  if items = 5 ∧ bags = 4 then 41 else 0

theorem correct_distribution :
  ways_to_distribute_items 5 4 = 41 :=
  by simp [ways_to_distribute_items]; sorry

end correct_distribution_l585_585563


namespace daisies_bought_l585_585552

-- Definitions from the given conditions
def cost_per_flower : ℕ := 6
def num_roses : ℕ := 7
def total_spent : ℕ := 60

-- Proving the number of daisies Maria bought
theorem daisies_bought : ∃ (D : ℕ), D = 3 ∧ total_spent = num_roses * cost_per_flower + D * cost_per_flower :=
by
  sorry

end daisies_bought_l585_585552


namespace intersection_P_Q_l585_585655

def P := {x : ℝ | x^2 - 9 < 0}
def Q := {y : ℤ | ∃ x : ℤ, y = 2*x}

theorem intersection_P_Q :
  {x : ℝ | x ∈ P ∧ (∃ n : ℤ, x = 2*n)} = {-2, 0, 2} :=
by
  sorry

end intersection_P_Q_l585_585655


namespace angle_abc_equilateral_triangle_l585_585103

/-- In a regular hexagon with an equilateral triangle sharing a common side, the angle ABC is 60 degrees. -/
theorem angle_abc_equilateral_triangle {A B C : Point} (hexagon : regular_hexagon A)
  (triangle : equilateral_triangle B C) (share_side : share_side hexagon triangle) : 
  ∠ A B C = 60 :=
sorry

end angle_abc_equilateral_triangle_l585_585103


namespace range_of_b_l585_585329

theorem range_of_b (b : ℝ) (h : (1 + b*complex.I) * (2 + complex.I)).im < 0 ∧ (1 + b*complex.I) * (2 + complex.I).re > 0 : b < -1 / 2 :=
by
  sorry

end range_of_b_l585_585329


namespace coefficient_x2_in_expansion_l585_585927

theorem coefficient_x2_in_expansion :
  ∀ (x : ℝ), ∑ k in finset.range 8, (nat.choose 7 k) * ((1:ℝ) ^ (7 - k)) * ((-x) ^ k) = ∑ k in finset.range 8, (nat.choose 7 k) * (1 ^ (7 - k)) * ((-x) ^ k) :=
by
  sorry

end coefficient_x2_in_expansion_l585_585927


namespace max_value_of_largest_integer_l585_585073

theorem max_value_of_largest_integer (a1 a2 a3 a4 a5 a6 a7 : ℕ) (h1 : a1 + a2 + a3 + a4 + a5 + a6 + a7 = 560) (h2 : a7 - a1 = 20) : a7 ≤ 21 :=
sorry

end max_value_of_largest_integer_l585_585073


namespace major_axis_length_of_ellipse_l585_585548

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2)

def f1 : ℝ × ℝ := (15, -5)
def f2 : ℝ × ℝ := (15, 45)
def f1_reflected : ℝ × ℝ := (-15, -5)

def midpoint (p1 p2 : ℝ × ℝ) : ℝ × ℝ :=
  ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

def major_axis_length (f1 f2 : ℝ × ℝ) : ℝ :=
  distance f1.f2

theorem major_axis_length_of_ellipse :
  major_axis_length f1 f1_reflected = 10 * Real.sqrt 34 :=
sorry

end major_axis_length_of_ellipse_l585_585548


namespace total_surfers_is_60_l585_585812

-- Define the number of surfers in Santa Monica beach
def surfers_santa_monica : ℕ := 20

-- Define the number of surfers in Malibu beach as twice the number of surfers in Santa Monica beach
def surfers_malibu : ℕ := 2 * surfers_santa_monica

-- Define the total number of surfers on both beaches
def total_surfers : ℕ := surfers_santa_monica + surfers_malibu

-- Prove that the total number of surfers is 60
theorem total_surfers_is_60 : total_surfers = 60 := by
  sorry

end total_surfers_is_60_l585_585812


namespace correct_values_of_a_l585_585627

noncomputable def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f (x)

noncomputable def is_monotonically_increasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 < x → x < y → f (x) < f (y)

def valid_a_values (a : ℝ) : Prop :=
  a ∈ {(-1 : ℝ), 2, (1 / 2 : ℝ), 3, (1 / 3 : ℝ)} ∧
  is_odd (λ x, x ^ a) ∧
  is_monotonically_increasing (λ x, x ^ a)

theorem correct_values_of_a :
  {a : ℝ | valid_a_values a} = {3, 1 / 3} :=
by {
  sorry
}

end correct_values_of_a_l585_585627


namespace paint_red_and_cut_then_count_l585_585026

def initial_cube_side_length : ℕ := 4

def cube_painted_faces (side_length : ℕ) : Prop :=
∀ (f : Fin 6 → set (Fin₃ × Fin₃ × Fin₃)), 
  (∀ p : Fin₃ × Fin₃ × Fin₃, ∃ f₁, f p) → 
  side_length = initial_cube_side_length

def cut_into_small_cubes (side_length small_cube_size : ℕ) : Prop :=
side_length % small_cube_size = 0

noncomputable def number_of_cubes_with_at_least_two_red_faces : ℕ :=
56

theorem paint_red_and_cut_then_count :
  ∀ (n : ℕ), cube_painted_faces n ∧ cut_into_small_cubes n 1 → n = initial_cube_side_length →
  number_of_cubes_with_at_least_two_red_faces = 56 :=
sorry

end paint_red_and_cut_then_count_l585_585026


namespace tangent_line_at_one_critical_points_inequality_l585_585268

noncomputable def f (x m : ℝ) := real.exp (x + m) + (m + 1) * x - x * real.log x

theorem tangent_line_at_one (m : ℝ) (h: m = 0) :
  ∃ (l : ℝ → ℝ), (∀ x, l x = f 1 0 + (real.exp 1) * (x - 1)) ∧
                  (l = λ x, real.exp 1 * x - y + 1) :=
sorry

theorem critical_points_inequality {m x1 x2 : ℝ} (h1 : x1 ≠ x2) (h2 : 0 < x1) (h3 : 0 < x2)
  (h4 : f x1 m = 0) (h5 : f x2 m = 0) : x1 * x2 < 1 :=
sorry

end tangent_line_at_one_critical_points_inequality_l585_585268


namespace compute_expression_l585_585201

-- Define the conditions as specific values and operations within the theorem itself
theorem compute_expression : 5 + 7 * (2 - 9)^2 = 348 := 
  by
  sorry

end compute_expression_l585_585201


namespace maximize_l_for_a_l585_585273

noncomputable def l (a : ℝ) : ℝ := (Real.sqrt (5) + 1) / 2

theorem maximize_l_for_a (a : ℝ) (f : ℝ → ℝ) (hfa : ∀ x ∈ set.Icc 0 (l a), |f x| ≤ 5) (ha : a < 0) : 
  (∃ (a_max : ℝ), a_max = -8 ∧ l a_max = (Real.sqrt (5) + 1) / 2) :=
  sorry

end maximize_l_for_a_l585_585273


namespace non_adjacent_girls_arrangements_l585_585226

-- Definitions corresponding to conditions in the problem

def boys : ℕ := 3
def girls : ℕ := 2
def total_students : ℕ := boys + girls

theorem non_adjacent_girls_arrangements : 
  ∀ (boys girls : ℕ), 
  boys = 3 → 
  girls = 2 → 
  ∑ i in finset.range (boys!), 
  ∑ j in finset.range ((boys + 1) - girls)! = 72 :=
begin
  -- sorry is used to indicate that the proof is omitted
  sorry
end

end non_adjacent_girls_arrangements_l585_585226


namespace arithmetic_seq_sixth_term_l585_585432

theorem arithmetic_seq_sixth_term (a₁ a₁₁ : ℚ) (h₁ : a₁ = 3 / 8) (h₁₁ : a₁₁ = 5 / 6) : 
  let a₆ := (a₁ + a₁₁) / 2 
  in a₆ = 29 / 48 :=
by
  sorry

end arithmetic_seq_sixth_term_l585_585432


namespace store_profit_l585_585099

theorem store_profit (m n : ℝ) (hmn : m > n) : 
  let selling_price := (m + n) / 2
  let profit_a := 40 * (selling_price - m)
  let profit_b := 60 * (selling_price - n)
  let total_profit := profit_a + profit_b
  total_profit > 0 :=
by sorry

end store_profit_l585_585099


namespace segment_length_l585_585060

def cbrt (x : ℝ) : ℝ := x^(1/3)

theorem segment_length (x : ℝ) 
  (h : |x - cbrt 27| = 5) : (abs ((cbrt 27 + 5) - (cbrt 27 - 5)) = 10) :=
by
  sorry

end segment_length_l585_585060


namespace minimize_area_of_triangle_l585_585355

variables {A B C D O X : Type*} [Nonempty X] [Nonempty A] [Nonempty B] [Nonempty C] [Nonempty D] [Nonempty O]

-- Assume the geometric configuration
variables (ABC : Triangle A B C) (ω : Circle X) (O : Point O) (minor_arc_AB : Arc ω A B)
variable (X : Point X)
variables (CX : Line C X) (AB : Line A B) (D : Point D)
variables (O1 : Point O) (ADX : Triangle A D X) (circumcenter_ADX : circumcenter ADX O1)
variables (O2 : Point O) (BDX : Triangle B D X) (circumcenter_BDX : circumcenter BDX O2)

-- Define the area function of triangle
noncomputable def area (Δ : Triangle X O O1 O2) : ℝ := sorry -- Assuming we have a way to calculate the area

-- Hypotheses
variables (h1 : inscribed_in_circle ABC ω)
variables (h2 : center ω O)
variables (h3 : X ∈ minor_arc_AB)
variables (h4 : D ∈ (CX ∩ AB))
variables (h5 : circumcenter (ADX) O1)
variables (h6 : circumcenter (BDX) O2)

-- The statement to prove
theorem minimize_area_of_triangle (Area : ℝ) :
  (∀ X, area (Triangle O O1 O2) ≥ Area) ↔ (XC.perpendicular AB) :=
by sorry

end minimize_area_of_triangle_l585_585355


namespace sum_of_possible_m_values_l585_585131

theorem sum_of_possible_m_values :
  let m_range := Finset.Icc 5 17 in
  m_range.sum id = 143 := by
  sorry

end sum_of_possible_m_values_l585_585131


namespace sum_triangle_inequality_solutions_l585_585141

theorem sum_triangle_inequality_solutions :
  (∑ m in (Finset.Icc 5 17), m) = 143 :=
by
  sorry

end sum_triangle_inequality_solutions_l585_585141


namespace ratio_of_seashells_sold_l585_585543

theorem ratio_of_seashells_sold:
  ∀ (initial given_friends given_brothers left_after_selling : ℕ),
  initial = 180 →
  given_friends = 40 →
  given_brothers = 30 →
  left_after_selling = 55 →
  (initial - given_friends - given_brothers - left_after_selling) * 2 = left_after_selling :=
by
  intros initial given_friends given_brothers left_after_selling
  assume h_initial h_given_friends h_given_brothers h_left_after_selling
  sorry

end ratio_of_seashells_sold_l585_585543


namespace find_x_for_g_statement_l585_585667

noncomputable def g (x : ℝ) : ℝ := (x + 4) ^ (1/3) / 5 ^ (1/3)

theorem find_x_for_g_statement (x : ℝ) : g (3 * x) = 3 * g x ↔ x = -13 / 3 := by
  sorry

end find_x_for_g_statement_l585_585667


namespace balloons_in_package_initially_l585_585034

-- Definition of conditions
def friends : ℕ := 5
def balloons_given_back : ℕ := 11
def balloons_after_giving_back : ℕ := 39

-- Calculation for original balloons each friend had
def original_balloons_each_friend := balloons_after_giving_back + balloons_given_back

-- Theorem: Number of balloons in the package initially
theorem balloons_in_package_initially : 
  (original_balloons_each_friend * friends) = 250 :=
by
  sorry

end balloons_in_package_initially_l585_585034


namespace highland_baseball_club_members_l585_585750

-- Define the given costs and expenditures.
def socks_cost : ℕ := 6
def tshirt_cost : ℕ := socks_cost + 7
def cap_cost : ℕ := socks_cost
def total_expenditure : ℕ := 5112
def home_game_cost : ℕ := socks_cost + tshirt_cost
def away_game_cost : ℕ := socks_cost + tshirt_cost + cap_cost
def cost_per_member : ℕ := home_game_cost + away_game_cost

theorem highland_baseball_club_members :
  total_expenditure / cost_per_member = 116 :=
by
  sorry

end highland_baseball_club_members_l585_585750


namespace smallest_four_digit_multiple_of_18_l585_585987

theorem smallest_four_digit_multiple_of_18 (n : ℕ) (h1 : 1000 ≤ n) (h2 : n < 10000) (h3 : n % 18 = 0) : n = 1008 :=
by
  have h4 : n ≥ 1008, sorry
  have h5 : n ≤ 1008, sorry
  exact eq_of_le_of_ge h4 h5

end smallest_four_digit_multiple_of_18_l585_585987


namespace log_inequality_solution_l585_585253

noncomputable def log_a (a x : ℝ) : ℝ := Real.log x / Real.log a

theorem log_inequality_solution (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (log_a a (3 / 5) < 1) ↔ (a ∈ Set.Ioo 0 (3 / 5) ∪ Set.Ioi 1) := 
by
  sorry

end log_inequality_solution_l585_585253


namespace max_ones_in_distinct_product_table_l585_585686

theorem max_ones_in_distinct_product_table :
  ∀ (table : Matrix (Fin 3) (Fin 3) ℕ), 
    (∀ i j k, table i j ≠ table i k) ∧
    (∀ i j k, table j i ≠ table k i) →
    (card (set_of (λ x, x = 1) (table.map id) : set ℕ) ≤ 5) :=
by sorry

end max_ones_in_distinct_product_table_l585_585686


namespace number_of_seats_on_each_bus_l585_585802

theorem number_of_seats_on_each_bus (total_students : ℕ) (number_of_buses : ℕ) (h1 : total_students = 60) (h2 : number_of_buses = 6) : total_students / number_of_buses = 10 :=
by
  rw [h1, h2]
  norm_num
  .
  .

end number_of_seats_on_each_bus_l585_585802


namespace arithmetic_sequence_20th_term_l585_585206

theorem arithmetic_sequence_20th_term (a₁ : ℕ) (d : ℕ) (n : ℕ) (h1 : a₁ = 2) (h2 : d = 3) (h3 : n = 20) :
  a₁ + (n - 1) * d = 59 :=
by
  rw [h1, h2, h3]
  sorry

end arithmetic_sequence_20th_term_l585_585206


namespace mom_prepared_pieces_l585_585996

-- Define the conditions
def jane_pieces : ℕ := 4
def total_eaters : ℕ := 3

-- Define the hypothesis that each of the eaters ate an equal number of pieces
def each_ate_equal (pieces : ℕ) : Prop := pieces = jane_pieces

-- The number of pieces Jane's mom prepared
theorem mom_prepared_pieces : total_eaters * jane_pieces = 12 :=
by
  -- Placeholder for actual proof
  sorry

end mom_prepared_pieces_l585_585996


namespace smallest_four_digit_multiple_of_18_l585_585980

theorem smallest_four_digit_multiple_of_18 : ∃ n : ℕ, (1000 ≤ n) ∧ (n < 10000) ∧ (n % 18 = 0) ∧ (∀ m : ℕ, (1000 ≤ m) ∧ (m < 10000) ∧ (m % 18 = 0) → n ≤ m) ∧ n = 1008 :=
by
  sorry

end smallest_four_digit_multiple_of_18_l585_585980


namespace sum_possible_m_values_l585_585134

theorem sum_possible_m_values : 
  let m_values := {m : ℕ | 4 < m ∧ m < 18}
  let sum_m := ∑ m in m_values, m
  sum_m = 143 :=
by
  sorry

end sum_possible_m_values_l585_585134


namespace area_of_S_l585_585374

-- Define the properties of the sequence
def seq (x y : ℝ) : ℕ → ℝ
| 0       := x
| (n + 1) := (seq n) ^ 2 / 2 + y^2 / 2

-- Define the set S
def S : set (ℝ × ℝ) := {p : ℝ × ℝ | (∃ x y, p = (x, y) ∧ ∃ l, is_seq_limit (seq x y) l)}

-- Prove that the area of S is 4 + π
theorem area_of_S : (set.area S) = 4 + real.pi :=
sorry

end area_of_S_l585_585374


namespace both_white_balls_probability_l585_585516

noncomputable def probability_two_white_balls
  (white_balls : ℕ) (black_balls : ℕ) (draws : ℕ) : ℚ :=
  if h : (white_balls + black_balls) = 15 ∧ white_balls = 7 ∧ black_balls = 8 ∧ draws = 2 then
    have total_balls : ℚ := white_balls + black_balls, from 15,
    have first_draw_white : ℚ := white_balls / total_balls, from 7 / 15,
    have second_draw_white : ℚ := (white_balls - 1) / (total_balls - 1), from 6 / 14,
    first_draw_white * second_draw_white
  else 0

theorem both_white_balls_probability : probability_two_white_balls 7 8 2 = 1 / 5 :=
by {
  dsimp [probability_two_white_balls],
  split_ifs,
  sorry
}

end both_white_balls_probability_l585_585516


namespace correct_conditional_statement_description_l585_585064

theorem correct_conditional_statement_description :
  (exists (P Q : Prop), 
    (¬ ∃ (R : Prop), if R then P else Q) ∧ 
    (∃ S T : Prop, if S ∧ T then P else Q) ∧ 
    (∃ U V : Prop, if U then P) ∧
    (¬ ∃ W X : Prop, if W then P else if X then Q)) :=
begin
  sorry
end

end correct_conditional_statement_description_l585_585064


namespace min_value_cos_function_l585_585271

theorem min_value_cos_function (φ : ℝ) (hφ1 : 0 < φ) (hφ2 : φ < π) :
  let f := λ (x : ℝ), sin (2 * x + φ) + sqrt 3 * cos (2 * x + φ)
  let f' := λ (x : ℝ), 2 * sin (2 * (x + π / 4) + φ + π / 3)
  let g := λ (x : ℝ), cos (x + φ)
  (∀ x, f' (x + π / 4) = f x) →
  (∀ x, f' (x) = 2 * cos (2 * x + φ + π / 3)) →
  (∀ x, f' (π / 2 - x) = - f' (π / 2 + x)) →
  (∃ φ, f' π / 2 = 0) →
  (∀ x ∈ Icc (-π / 2) (π / 6), g x ≥ - 1/2 ∧ g x ≤ 1/2) → 
  ∃ x ∈ Icc (-π / 2) (π / 6), g x = 1/2 :=
sorry

end min_value_cos_function_l585_585271


namespace andrey_boris_denis_eat_candies_l585_585163

def andrey_boris_condition (a b : ℕ) : Prop :=
  a = 4 ∧ b = 3

def andrey_denis_condition (a d : ℕ) : Prop :=
  a = 6 ∧ d = 7

def total_candies_condition (total : ℕ) : Prop :=
  total = 70

theorem andrey_boris_denis_eat_candies :
  ∃ (a b d : ℕ), andrey_boris_condition a b ∧ andrey_denis_condition a d ∧ 
                  (total_candies_condition (2 * (12 + 9 + 14)) ∧ 
                   2 * 12 = 24 ∧ 2 * 9 = 18 ∧ 2 * 14 = 28) →
                  (a = 24 ∧ b = 18 ∧ d = 28) :=
by
  sorry

end andrey_boris_denis_eat_candies_l585_585163


namespace find_ellipse_equation_l585_585740

noncomputable def ellipse_equation (x y a b : ℝ) : Prop :=
  (x^2 / a^2 + y^2 / b^2 = 1)

theorem find_ellipse_equation (M N : ℝ × ℝ)
  (a b : ℝ)
  (h1 : a > 0)
  (h2 : b > 0)
  (h3 : a > b)
  (h4 : let y := (λ x : ℝ, x + Real.sqrt 2) in
        let line := (λ x : ℝ, (x, y x)) in
        ∃ xM yM xN yNN M = line xM ∧ M = line xN ∧ 
        ellipse_equation xM yM a b ∧ ellipse_equation xN yN a b)
  (h5 : let o := (0, 0) in
        let ⟨xM, yM⟩ := M in
        let ⟨xN, yN⟩ := N in
        (xM * xN + yM * yN = 0))
  (h6 : let ⟨xM, yM⟩ := M in
        let ⟨xN, yN⟩ := N in
        Real.sqrt ((xM - xN)^2 + (yM - yN)^2) = Real.sqrt 6) :
  ellipse_equation x y (Real.sqrt (4 + 2 * Real.sqrt 2)) (Real.sqrt (4 - 2 * Real.sqrt 2)) :=
by
  sorry

end find_ellipse_equation_l585_585740


namespace geom_seq_q_eq_l585_585333

theorem geom_seq_q_eq (a1 : ℕ := 2) (S3 : ℕ := 26) 
  (h1 : a1 = 2) 
  (h2 : S3 = 26) : 
  ∃ q : ℝ, (q = 3 ∨ q = -4) := by
  sorry

end geom_seq_q_eq_l585_585333


namespace cost_of_paving_l585_585017

def length_of_room : ℝ := 5.5
def width_of_room : ℝ := 3.75
def rate_per_sq_meter : ℝ := 1000

theorem cost_of_paving :
  let area := length_of_room * width_of_room in
  let cost := area * rate_per_sq_meter in
  cost = 20625 :=
by
  sorry

end cost_of_paving_l585_585017


namespace problem_A_problem_B_problem_C_problem_D_l585_585493

-- Define conditions
def isValidOutcome (p q : ℕ) : Prop :=
  p ∈ {1, 2, 3, 4, 5, 6} ∧ q ∈ {1, 2, 3, 4, 5, 6}

def A (p q : ℕ) : ℕ :=
  (p / q : ℤ).toNat

-- Define the problems
theorem problem_A (p q : ℕ) (h : isValidOutcome p q) : 
  Probability (p + q = 5) ≠ 1 / 4 := 
by
  sorry

theorem problem_B (p q : ℕ) (h : isValidOutcome p q) :
  (p = 6 ∧ A p q = 0) → False := 
by
  sorry

theorem problem_C (p q : ℕ) (h : isValidOutcome p q) :
  Probability (p > q) = 5 / 12 := 
by
  sorry

theorem problem_D (p q : ℕ) (h : isValidOutcome p q) :
  (q = 1 ∧ A p q = 0) → False := 
by
  sorry

end problem_A_problem_B_problem_C_problem_D_l585_585493


namespace min_degree_g_l585_585483

theorem min_degree_g (f g h : Polynomial ℝ) (hf : f.degree = 8) (hh : h.degree = 9) (h_eq : 3 * f + 4 * g = h) : g.degree ≥ 9 :=
sorry

end min_degree_g_l585_585483


namespace value_range_f_l585_585032

noncomputable def f (x: ℝ) : ℝ := (1/2) * Real.exp x * (Real.sin x + Real.cos x)

theorem value_range_f : 
  set.range (λ x, f x) = set.Icc (1/2) ((1/2) * Real.exp (Real.pi / 2)) :=
begin
  sorry
end

end value_range_f_l585_585032


namespace liars_in_square_l585_585089

-- Define the initial problem setup
def num_people := 2019

-- Define properties of individuals in the line
inductive PersonType
| knight : PersonType
| liar : PersonType

-- There is one king who is a knight
def is_king (p : ℕ) : Prop :=
  p < num_people ∧ PersonType.knight

-- Define the conditions for the problem 
def condition (p k : ℕ) : Prop :=
  if p = k then
    true
  else if p < k then
    (p + 7 = k ∨ (num_people - (k - (p + 7)))) ∧ (PersonType.liar)
  else 
    (p - 7 = k ∨ (num_people - (p - 7))) ∧ (PersonType.liar)

-- Mathematical statement summarizing the required proof
theorem liars_in_square :
  ∃ (liars : ℕ), 7 ≤ liars ∧ liars ≤ 14 ∧ 
  (∀ p : ℕ, p < num_people → ∃ k : ℕ, k < num_people ∧ is_king k ∧ condition p k) :=
by
  sorry

end liars_in_square_l585_585089


namespace penultimate_digit_odd_of_square_last_digit_six_l585_585231

theorem penultimate_digit_odd_of_square_last_digit_six 
  (n : ℕ) 
  (h : (n * n) % 10 = 6) : 
  ((n * n) / 10) % 2 = 1 :=
sorry

end penultimate_digit_odd_of_square_last_digit_six_l585_585231


namespace max_area_of_rectangular_frame_l585_585111

theorem max_area_of_rectangular_frame (x y : ℕ) (h : 3 * x + 5 * y ≤ 50) : 
  ∃ A : ℕ, A = 40 ∧ A = x * y :=
begin
  sorry
end

end max_area_of_rectangular_frame_l585_585111


namespace largest_possible_sum_l585_585510

def max_sum_pair_mult_48 : Prop :=
  ∃ (heartsuit clubsuit : ℕ), (heartsuit * clubsuit = 48) ∧ (heartsuit + clubsuit = 49) ∧ 
  (∀ (h c : ℕ), (h * c = 48) → (h + c ≤ 49))

theorem largest_possible_sum : max_sum_pair_mult_48 :=
  sorry

end largest_possible_sum_l585_585510


namespace pens_sold_during_promotion_l585_585534

theorem pens_sold_during_promotion (x y n : ℕ) 
  (h_profit: 12 * x + 7 * y = 2011)
  (h_n: n = 2 * x + y) : 
  n = 335 := by
  sorry

end pens_sold_during_promotion_l585_585534


namespace triangle_angle_bisectors_ratio_l585_585710

/-- In triangle ABC, AD bisects angle BAC and BE bisects angle ABC, and they intersect at point P. We need to show that the ratio AP/PD equals AC/(BC - AC). -/
theorem triangle_angle_bisectors_ratio
  (A B C D E P : Type)
  (h_triangle : ∃ (Δ : Triangle A B C), Δ AD BE ∧ D ∈ interior Δ) :
  let AP := segment_pos_length A P,
      PD := segment_pos_length P D,
      AC := segment_pos_length A C,
      BC := segment_pos_length B C 
  in
  AP / PD = AC / (BC - AC) :=
sorry

end triangle_angle_bisectors_ratio_l585_585710


namespace find_angle_C_l585_585998

-- Definitions based on conditions
variables (α β γ : ℝ) -- Angles of the triangle

-- Condition: Angles between the altitude and the angle bisector at vertices A and B are equal
-- This implies α = β
def angles_equal (α β : ℝ) : Prop :=
  α = β

-- Condition: Sum of the angles in a triangle is 180 degrees
def angles_sum_to_180 (α β γ : ℝ) : Prop :=
  α + β + γ = 180

-- Condition: Angle at vertex C is greater than angles at vertices A and B
def c_greater_than_a_and_b (α γ : ℝ) : Prop :=
  γ > α

-- The proof problem: Prove γ = 120 degrees given the conditions
theorem find_angle_C (α β γ : ℝ) (h1 : angles_equal α β) (h2 : angles_sum_to_180 α β γ) (h3 : c_greater_than_a_and_b α γ) : γ = 120 :=
by
  sorry

end find_angle_C_l585_585998


namespace molecular_weight_constant_l585_585848

-- Given the molecular weight of a compound
def molecular_weight (w : ℕ) := w = 1188

-- Statement about molecular weight of n moles
def weight_of_n_moles (n : ℕ) := n * 1188

theorem molecular_weight_constant (moles : ℕ) : 
  ∀ (w : ℕ), molecular_weight w → ∀ (n : ℕ), weight_of_n_moles n = n * w :=
by
  intro w h n
  sorry

end molecular_weight_constant_l585_585848


namespace locus_of_K_l585_585010

variables {A B C D : ℝ³} 
variable (d : set ℝ³) -- line in space
variable (S : ℝ³) -- centroid of triangle ABC

def is_midpoint (M X Y : ℝ³) : Prop :=
  M = (X + Y) / 2

def is_parallelogram_midpoint (K P Q R S : ℝ³) : Prop :=
  K = (P + R) / 2 ∧ K = (Q + S) / 2

theorem locus_of_K (h_non_collinear: A ≠ B ∧ B ≠ C ∧ A ≠ C ∧ 
                                    ∀ α β γ, α • (B - A) + β • (C - A) ≠ γ • (C - A))
  (h_on_line_d: ∀ t: ℝ, ∃ D, D ∈ d ∧ D = t • S)
  (h_centroid: S = (A + B + C) / 3):
  ∃ k, ∀ D ∈ d, ∃ K, K ∈ k ∧ (is_midpoint K (1 / 4 • D + 3 / 4 • S) S) :=
sorry

end locus_of_K_l585_585010


namespace number_of_hyperbolas_l585_585232

theorem number_of_hyperbolas : 
  let set := {1, 2, 3, 5, 7, 9, 0, -4, -6, -8}
  ∃ A B C ∈ set, A ≠ B ∧ B ≠ C ∧ C ≠ A ∧ 
  (A > 0 ∧ B < 0 ∨ A < 0 ∧ B > 0) →
  (nOfHyperbolas : ℕ) → nOfHyperbolas = 252 := 
begin
  sorry
end

end number_of_hyperbolas_l585_585232


namespace negative_only_option_B_l585_585065

theorem negative_only_option_B :
  (0 > -3) ∧ 
  (|-3| = 3) ∧ 
  (0 < 3) ∧
  (0 < (1/3)) ∧
  ∀ x, x = -3 → x < 0 :=
by
  sorry

end negative_only_option_B_l585_585065


namespace trapezoid_area_l585_585902

theorem trapezoid_area (y : ℝ) : 
  let base1 := 3 * y,
      base2 := 4 * y in
  let area := y * (base1 + base2) / 2 in
  area = (7 * y^2) / 2 :=
by
  sorry

end trapezoid_area_l585_585902


namespace original_decimal_number_l585_585876

theorem original_decimal_number (I : ℤ) (d : ℝ) (h1 : 0 ≤ d) (h2 : d < 1) (h3 : I + 4 * (I + d) = 21.2) : I + d = 4.3 :=
by
  sorry

end original_decimal_number_l585_585876


namespace transformed_set_sum_l585_585533

-- Defining the sum before transformation and the conditions of transformation.
def original_sum (n : ℕ) (x : Fin n → ℝ) : ℝ := ∑ i, x i
def transformed_sum (n : ℕ) (x : Fin n → ℝ) : ℝ := ∑ i, 3 * x i + 20

-- Stating the theorem based on the identified conditions and the correct answer.
theorem transformed_set_sum (n : ℕ) (x : Fin n → ℝ)
  (s : ℝ) (h : original_sum n x = s) :
  transformed_sum n x = 3 * s + 20 * n :=
  sorry

end transformed_set_sum_l585_585533


namespace AE_mul_EB_l585_585341

-- Define the basic geometry setup
variables {A B C P E : Type}
variables [metric_space A] [metric_space B] [metric_space C]
variables [right_triangle A B C] [angle_bisectors_intersect A B C P]
variables [perpendicular PE A B E]

-- Given lengths
variables (BC AC : ℝ)
variables (h_bc : BC = 2) (h_ac : AC = 3)

-- The goal is to prove AE * EB = 3
theorem AE_mul_EB (AE EB : ℝ) (h_AE_perp_EB : AE * EB = 3) : 
  AE * EB = 3 :=
sorry

end AE_mul_EB_l585_585341


namespace area_of_BGFC_l585_585313

theorem area_of_BGFC (A B C D E F G : Point)
  (h1 : collinear A D E)
  (h2 : on_segment B A D)
  (h3 : on_segment C A D)
  (h4 : collinear A E F)
  (h5 : on_segment G A E)
  (h6 : BG ∥ CF)
  (h7 : CF ∥ DE)
  (h8 : BG ∥ DE)
  (h9 : area_triangle A B G = 36)
  (h10 : area_trapezoid C F E D = 144)
  (h11 : length_segment A B = length_segment C D) :
  area_trapezoid B G F C = 45 := sorry

end area_of_BGFC_l585_585313


namespace mean_exercise_days_correct_l585_585315

-- Conditions
def students_exercise_days : List (ℕ × ℕ) :=
[
  (2, 0), -- 2 students exercised for 0 days
  (4, 1), -- 4 students exercised for 1 day
  (5, 3), -- 5 students exercised for 3 days
  (7, 4), -- 7 students exercised for 4 days
  (3, 5), -- 3 students exercised for 5 days
  (2, 6)  -- 2 students exercised for 6 days
]

-- Total number of days exercised
def total_days : ℕ :=
students_exercise_days.foldl (λ acc (n, d) => acc + n * d) 0 

-- Total number of students
def total_students : ℕ :=
students_exercise_days.foldl (λ acc (n, _) => acc + n) 0

-- Mean number of days
noncomputable def mean : ℚ := total_days / total_students

-- Theorem to prove
theorem mean_exercise_days_correct : Float.round (mean.toRat.toReal * 100) / 100 = 3.22 := by
  sorry

end mean_exercise_days_correct_l585_585315


namespace open_box_volume_l585_585890

theorem open_box_volume (l w s : ℝ) (hl : l = 48) (hw : w = 36) (hs : s = 8) :
  (l - 2 * s) * (w - 2 * s) * s = 5120 :=
by
  sorry

end open_box_volume_l585_585890


namespace smallest_four_digit_multiple_of_18_l585_585979

theorem smallest_four_digit_multiple_of_18 : 
  ∃ (n : ℕ), 999 < n ∧ n < 10000 ∧ 18 ∣ n ∧ ∀ m : ℕ, 999 < m ∧ m < 10000 ∧ 18 ∣ m → n ≤ m ∧ n = 1008 := 
sorry

end smallest_four_digit_multiple_of_18_l585_585979


namespace solution_correct_l585_585649

noncomputable def is_periodic (f : ℝ → ℝ) (T : ℝ) : Prop :=
∀ x, f (x + T) = f x

noncomputable def is_increasing_on (f : ℝ → ℝ) (I : set ℝ) : Prop :=
∀ ⦃a b⦄, a ∈ I → b ∈ I → a < b → f a < f b

noncomputable def prob : Prop :=
  (is_periodic (fun x => Real.tan x) Real.pi ∧ is_increasing_on (fun x => Real.tan x) (set.Ioo 0 (Real.pi / 2))) ∧
  ¬(is_periodic (fun x => Real.sin (Real.abs x)) Real.pi ∧ is_increasing_on (fun x => Real.sin (Real.abs x)) (set.Ioo 0 (Real.pi / 2))) ∧
  (is_periodic (fun x => Real.abs (Real.sin x)) Real.pi ∧ is_increasing_on (fun x => Real.abs (Real.sin x)) (set.Ioo 0 (Real.pi / 2))) ∧
  ¬(is_periodic (fun x => Real.abs (Real.cos x)) Real.pi ∧ is_increasing_on (fun x => Real.abs (Real.cos x)) (set.Ioo 0 (Real.pi / 2)))

theorem solution_correct : prob :=
  sorry

end solution_correct_l585_585649


namespace general_formula_find_m_l585_585714

noncomputable def a_n : ℕ → ℕ
| 0 := 0 -- Indexed from 1 for simplicity in Lean
| (n+1) := if (n + 1) % 2 = 1 then n + 1 else 2 ^ ((n + 1) / 2)

theorem general_formula (n : ℕ) :
  a_n (n + 1) = if (n + 1) % 2 = 1 then n + 1 else 2 ^ ((n + 1) / 2) := by
  sorry

theorem find_m (m : ℕ) :
  (a_n m) * (a_n (m + 1)) * (a_n (m + 2)) = (a_n m) + (a_n (m + 1)) + (a_n (m + 2)) ↔ m = 1 := by
  sorry

end general_formula_find_m_l585_585714


namespace candies_eaten_l585_585155

variables (A B D : ℕ)

-- Conditions:
def condition1 : Prop := ∃ k1 k2 k3 : ℕ, k1 * 4 + k2 * 3 + k3 * 7 = 70
def condition2 : Prop := (B * 3 = A * 4) ∧ (D * 7 = A * 6)
def condition3 : Prop := A + B + D = 70

-- Theorem statement:
theorem candies_eaten (h1 : condition1) (h2 : condition2) (h3 : condition3) :
    A = 24 ∧ B = 18 ∧ D = 28 := sorry

end candies_eaten_l585_585155


namespace solution_set_l585_585646

noncomputable def f (x : ℝ) : ℝ := 2017^x + real.log 2017 (real.sqrt (x^2 + 1) + x) - 2017^(-x) + 2

theorem solution_set (x : ℝ) : f (3 * x + 1) + f x > 4 ↔ x > -1/4 := 
sorry

end solution_set_l585_585646


namespace number_of_tower_heights_l585_585939

theorem number_of_tower_heights (h1 : ∀ (heights : list ℕ), ∀ (x ∈ heights, x = 3 ∨ x = 8 ∨ x = 18)) 
                                (h2 : heights.length = 80) 
                                (h3 : ∀ (height_of_towers : list ℕ), 
                                  height_of_towers = [240, 240 + 5, 240 + 10, ..., 1440]) :
  (height_of_towers.length = 241) :=
sorry

end number_of_tower_heights_l585_585939


namespace boy_speed_second_day_l585_585517

noncomputable def hours (minutes : ℕ) : ℝ := minutes / 60.0

def distance : ℝ := 2.0  -- distance between home and school in km
def speed_first_day : ℝ := 4.0  -- speed on the first day in km/hr
def time_standard : ℝ := distance / speed_first_day  -- standard time to travel in hr
def time_late : ℝ := time_standard + hours 7  -- time taken on the first day (7 minutes late)
def time_early : ℝ := time_standard - hours 8  -- time taken on the second day (8 minutes early)

def speed_second_day : ℝ := distance / time_early  -- speed on the second day

theorem boy_speed_second_day : speed_second_day ≈ 5.45 := 
by
  sorry

end boy_speed_second_day_l585_585517


namespace union_A_B_inter_complB_A_l585_585657

-- Define the universal set U
def U : Set ℝ := Set.univ

-- Define the set A
def A : Set ℝ := {x | -3 < x ∧ x ≤ 6}

-- Define the set B
def B : Set ℝ := {x | x^2 - 5*x - 6 < 0}

-- Define the complement of B with respect to U
def compl_B : Set ℝ := {x | x ≤ -1 ∨ x ≥ 6}

-- Problem (1): Prove that A ∪ B = {x | -3 < x ∧ x ≤ 6}
theorem union_A_B : A ∪ B = {x | -3 < x ∧ x ≤ 6} := by
  sorry

-- Problem (2): Prove that (compl_B) ∩ A = {x | (-3 < x ∧ x ≤ -1) ∨ x = 6}
theorem inter_complB_A : compl_B ∩ A = {x | (-3 < x ∧ x ≤ -1) ∨ x = 6} := by 
  sorry

end union_A_B_inter_complB_A_l585_585657


namespace John_lost_3_ebook_readers_l585_585349

-- Definitions based on the conditions
def A : Nat := 50  -- Anna bought 50 eBook readers
def J : Nat := A - 15  -- John bought 15 less than Anna
def total : Nat := 82  -- Total eBook readers now

-- The number of eBook readers John has after the loss:
def J_after_loss : Nat := total - A

-- The number of eBook readers John lost:
def John_loss : Nat := J - J_after_loss

theorem John_lost_3_ebook_readers : John_loss = 3 :=
by
  sorry

end John_lost_3_ebook_readers_l585_585349


namespace probability_A_wins_match_against_B_optimal_strategy_A_second_match_l585_585758

noncomputable def prob_win_A_game_B : ℚ := 2 / 3

noncomputable def prob_win_match_A_B : ℚ := 
  prob_win_A_game_B * prob_win_A_game_B + 
  (1 - prob_win_A_game_B) * prob_win_A_game_B * prob_win_A_game_B + 
  prob_win_A_game_B * (1 - prob_win_A_game_B) * prob_win_A_game_B

theorem probability_A_wins_match_against_B : prob_win_match_A_B = 20 / 27 :=
by
  have h1 : prob_win_A_game_B * prob_win_A_game_B = 4 / 9,
  { sorry, }

  have h2 : (1 - prob_win_A_game_B) * prob_win_A_game_B * prob_win_A_game_B = 4 / 27,
  { sorry, }

  have h3 : prob_win_A_game_B * (1 - prob_win_A_game_B) * prob_win_A_game_B = 4 / 27,
  { sorry, }

  calc
    prob_win_match_A_B = 4 / 9 + 4 / 27 + 4 / 27 : by rw [h1, h2, h3]
    ... = 12 / 27 + 4 / 27 + 4 / 27 : by norm_num
    ... = 20 / 27 : by norm_num

noncomputable def prob_win_A_game (opponent_prob : ℚ) : ℚ :=
opponent_prob

def prob_win_two_consecutive (p1 p2 : ℚ) : ℚ :=
p1 * ((1 - p2) * 2)

theorem optimal_strategy_A_second_match :
  ∀ (pB pC pD : ℚ), pB = 1/2 → pC = 2/3 → pD = 3/4 → 
  prob_win_two_consecutive pC pD = 3/4 :=
by
  intros pB pC pD hB hC hD,
  have hB_expr : prob_win_two_consecutive pB pD = 5 / 12,
  { sorry, }

  have hC_expr : prob_win_two_consecutive pC pD = 2 / 3,
  { sorry, }

  have hD_expr : prob_win_two_consecutive pD pD = 3 / 4,
  { sorry, }

  rw hC at hC_expr,
  rw hD at hD_expr,
  exact hD_expr

end probability_A_wins_match_against_B_optimal_strategy_A_second_match_l585_585758


namespace ducklings_snails_l585_585107

theorem ducklings_snails :
  ∃ n : ℕ,
    let snails_one := 3 * 5,
        snails_two := 3 * 9,
        total_snails_groups := snails_one + snails_two,
        snails_mother := 3 * total_snails_groups,
        snails_per_remaining_duckling := snails_mother / 2,
        snails_total := 294,
        snails_first_two_and_mother := total_snails_groups + snails_mother,
        snails_remaining := snails_total - snails_first_two_and_mother,
        remaining_ducklings := snails_remaining / snails_per_remaining_duckling,
        total_ducklings := 3 + 3 + remaining_ducklings
    in total_ducklings = 8 := 
sorry

end ducklings_snails_l585_585107


namespace find_b_of_triangle_ABC_l585_585385

theorem find_b_of_triangle_ABC (a b c : ℝ) (cos_A : ℝ) 
  (h1 : a = 2) 
  (h2 : c = 2 * Real.sqrt 3) 
  (h3 : cos_A = Real.sqrt 3 / 2) 
  (h4 : b < c) : 
  b = 2 := 
by
  sorry

end find_b_of_triangle_ABC_l585_585385


namespace students_who_chose_water_l585_585188

noncomputable def num_students_chose_water (total_students : ℕ) (juice_percentage water_percentage : ℚ) (students_chose_juice : ℕ) : ℕ :=
(students_chose_juice * water_percentage / juice_percentage).nat_abs

theorem students_who_chose_water :
  num_students_chose_water 100 0.4 0.3 100 = 75 :=
by
  sorry

end students_who_chose_water_l585_585188


namespace min_value_m2n_mn_l585_585239

theorem min_value_m2n_mn (m n : ℝ) 
  (h1 : (x - m)^2 + (y - n)^2 = 9)
  (h2 : x + 2 * y + 2 = 0)
  (h3 : 0 < m)
  (h4 : 0 < n)
  (h5 : m + 2 * n + 2 = 5)
  (h6 : ∃ l : ℝ, l = 4 ): (m + 2 * n) / (m * n) = 8/3 :=
by
  sorry

end min_value_m2n_mn_l585_585239


namespace angle_AOD_twice_angle_CAD_l585_585106

noncomputable def angle (A B C : Point) : ℝ := sorry

noncomputable def circle (O : Point) (r : ℝ) : Set Point := sorry

noncomputable def is_tangent (line : Set Point) (circle : Set Point) (A : Point) : Prop := sorry

noncomputable def is_on_line (C : Point) (line : Set Point) : Prop := sorry

noncomputable def is_on_circle (D : Point) (circle : Set Point) : Prop := sorry

noncomputable def same_side (C D : Point) (line : Set Point) : Prop := sorry

theorem angle_AOD_twice_angle_CAD
  (O A C D : Point)
  (r : ℝ)
  (line : Set Point)
  (circ := circle O r)
  (tangent := is_tangent line circ A)
  (C_on_line := is_on_line C line)
  (D_on_circle := is_on_circle D circ)
  (C_D_same_side := same_side C D line)
  : angle C A D = 1/2 * angle A O D := sorry

end angle_AOD_twice_angle_CAD_l585_585106


namespace Mrs_Lara_Late_l585_585388

noncomputable def required_speed (d t : ℝ) : ℝ := d / t

theorem Mrs_Lara_Late (d t : ℝ) (h1 : d = 50 * (t + 7 / 60)) (h2 : d = 70 * (t - 5 / 60)) :
  required_speed d t = 70 := by
  sorry

end Mrs_Lara_Late_l585_585388


namespace neg_pi_lt_neg_314_l585_585198

theorem neg_pi_lt_neg_314 (h : Real.pi > 3.14) : -Real.pi < -3.14 :=
sorry

end neg_pi_lt_neg_314_l585_585198


namespace trapezoid_area_l585_585921

-- Definitions based on given problem conditions
def P : ℝ × ℝ := (-3, -5)
def Q : ℝ × ℝ := (-3, 3)
def R : ℝ × ℝ := (4, 10)
def S : ℝ × ℝ := (4, -1)

-- Formal statement of the problem
theorem trapezoid_area : 
  let h := 4 - (-3) in 
  let b1 := 3 - (-5) in 
  let b2 := 10 - (-1) in 
  (h = 7) ∧ (b1 = 8) ∧ (b2 = 11) → 
  (1 / 2 : ℝ) * ((b1 + b2) * h) = 66.5 :=
by
  sorry

end trapezoid_area_l585_585921


namespace prime_composite_lcm_l585_585378

theorem prime_composite_lcm (p c : ℕ) (prime_p : p.prime) (composite_c : 2 ≤ c) :
  ∃ m n : ℕ, m = p^(2*c-1) ∧ n = p^(2*c-1) - p^(c-1) ∧ 0 < m - n ∧
  (m - n < p^c) ∧
  (nat.lcm (list.range (m + 1)).drop (n + 1)) / nat.lcm (list.range m).drop n = p^c := 
sorry

end prime_composite_lcm_l585_585378


namespace problem_proof_periodic_function_l585_585259

-- Define the piecewise function f with the given properties
def f (x : ℝ) : ℝ :=
  if 0 < x ∧ x <= 2 then cos (π * x / 2)
  else if 2 < x ∧ x <= 4 then log x - log (1.5)
  else f (x + 4)

-- Prove that f(f(-1/2)) = 0
theorem problem_proof_periodic_function : f (f (-1 / 2)) = 0 :=
by
  sorry -- Proof needed here

end problem_proof_periodic_function_l585_585259


namespace sqrt_ratio_div_l585_585584

variable (x y : ℝ)

theorem sqrt_ratio_div (h :  (1/2)^2 + (1/3)^2 / ( (1/4)^2 + (1/5)^2 ) = 13 * x / (41 * y)) :
    (sqrt x / sqrt y) = 10/3 := by
  sorry

end sqrt_ratio_div_l585_585584


namespace andrey_boris_denis_eat_candies_l585_585160

def andrey_boris_condition (a b : ℕ) : Prop :=
  a = 4 ∧ b = 3

def andrey_denis_condition (a d : ℕ) : Prop :=
  a = 6 ∧ d = 7

def total_candies_condition (total : ℕ) : Prop :=
  total = 70

theorem andrey_boris_denis_eat_candies :
  ∃ (a b d : ℕ), andrey_boris_condition a b ∧ andrey_denis_condition a d ∧ 
                  (total_candies_condition (2 * (12 + 9 + 14)) ∧ 
                   2 * 12 = 24 ∧ 2 * 9 = 18 ∧ 2 * 14 = 28) →
                  (a = 24 ∧ b = 18 ∧ d = 28) :=
by
  sorry

end andrey_boris_denis_eat_candies_l585_585160


namespace proof_statement_l585_585567

def convert_base_9_to_10 (n : Nat) : Nat :=
  3 * 9^2 + 2 * 9^1 + 4 * 9^0

def convert_base_6_to_10 (n : Nat) : Nat :=
  2 * 6^2 + 2 * 6^1 + 1 * 6^0

def problem_statement : Prop :=
  convert_base_9_to_10 324 - convert_base_6_to_10 221 = 180

theorem proof_statement : problem_statement := 
  by
    sorry

end proof_statement_l585_585567


namespace max_students_l585_585439

theorem max_students (n : ℕ) (h1 : 640 % n = 0) (h2 : 520 % n = 0) : n ≤ 40 :=
begin
  sorry,
end

end max_students_l585_585439


namespace exponential_inequality_l585_585272

noncomputable def f (x : ℝ) : ℝ := (x - 3) ^ 3 + 2 * x - 6

theorem exponential_inequality (a b : ℝ) (h : f (2 * a - b) + f (6 - b) > 0) : exp a > exp b :=
by
  -- Preliminary conditions are included in the theorem's assumptions.
  -- The full proof is omitted here.
  sorry

end exponential_inequality_l585_585272


namespace midpoint_centroid_trace_lines_cyclic_quadrilateral_APMQ_l585_585392

-- Part 1, Problem 1
theorem midpoint_centroid_trace_lines
  (ABC : Triangle) (M : Point) (hM : M ∈ BC)
  (P : Point) (hP : P ∈ LineParallelToACThroughM)
  (Q : Point) (hQ : Q ∈ LineParallelToBCThroughM) :
  let O := Midpoint(APMQ)
  let S := Centroid(APQ)
  (traceO : LineParallelToBC) (traceS : LineParallelToBC) :=
sorry

-- Part 2, Problem 2
theorem cyclic_quadrilateral_APMQ
  (ABC : Triangle) 
  (hCyclicAPMQ : CyclicQuadrilateralAPMQ)
  (hAngleBAC : ∠BAC = 90°) 
  (AMMinRadius : ∀ x, circumcircleRadius (APMQ) ≤ circumcircleRadius (APMxQ)) :
    IsRightTriangleAtA(ABC) ∧
    MInAltitudeFromAToBC :=
sorry

end midpoint_centroid_trace_lines_cyclic_quadrilateral_APMQ_l585_585392


namespace intersection_A_B_eq_23_l585_585623

-- Define the sets A and B according to the problem's conditions
def A : Set ℝ := {x | |x| ≥ 2}
def B : Set ℝ := {x | x^2 - 2x - 3 < 0}

-- Prove that the intersection of A and B equals [2, 3)
theorem intersection_A_B_eq_23 : A ∩ B = {x | 2 ≤ x ∧ x < 3} :=
by
  sorry

end intersection_A_B_eq_23_l585_585623


namespace total_surfers_calculation_l585_585813

def surfers_on_malibu_beach (m_sm : ℕ) (s_sm : ℕ) : ℕ := 2 * s_sm

def total_surfers (m_sm s_sm : ℕ) : ℕ := m_sm + s_sm

theorem total_surfers_calculation : total_surfers (surfers_on_malibu_beach 20 20) 20 = 60 := by
  sorry

end total_surfers_calculation_l585_585813


namespace volume_of_regular_tetrahedron_l585_585691

variables (α : ℝ) -- The angle parameter

/- Define the conditions / parameters -/
def dihedral_angle (α : ℝ) := 2 * α
def distance_from_center_to_edge := 1

/- Define the volume of the regular tetrahedron based on the given conditions -/
def volume_tetrahedron (α : ℝ) := 9 * (Real.tan α)^3 / (4 * Real.sqrt (3 * (Real.tan α)^2 - 1))

/-- The main theorem stating the volume -/
theorem volume_of_regular_tetrahedron 
  (α : ℝ) 
  (d : ℝ)
  (h_d : d = distance_from_center_to_edge) 
  : volume_tetrahedron α = 9 * (Real.tan α)^3 / (4 * Real.sqrt (3 * (Real.tan α)^2 - 1)) := 
by 
  sorry

end volume_of_regular_tetrahedron_l585_585691


namespace mode_median_mean_l585_585826

def data := [4, 1, 2, 2, 4, 1, 1]

def mean (l : List ℕ) : ℚ := (l.sum : ℚ) / l.length

def median (l : List ℕ) : ℚ := 
  let sorted := l.qsort (· ≤ ·) 
  if sorted.length % 2 = 0 then 
    (sorted.get! (sorted.length / 2 - 1) + sorted.get! (sorted.length / 2)) / 2
  else 
    sorted.get! (sorted.length / 2)

def mode (l : List ℕ) : ℕ := 
  l.foldl (λ acc x, if l.count x > l.count acc then x else acc) l.head!

theorem mode_median_mean (l : List ℕ) (h : l = data) : 
  mode l < median l ∧ median l < mean l := 
by 
  have hl : l = data := h
  have m := mean l
  have md := median l
  have mo := mode l
  have : mo = 1 := by sorry
  have : md = 2 := by sorry
  have : m = 15 / 7 := by sorry
  sorry

end mode_median_mean_l585_585826


namespace largest_divisor_of_Pn_for_even_n_l585_585357

def P (n : ℕ) : ℕ := 
  (n + 1) * (n + 3) * (n + 5) * (n + 7) * (n + 9)

theorem largest_divisor_of_Pn_for_even_n : 
  ∀ (n : ℕ), (0 < n ∧ n % 2 = 0) → ∃ d, d = 15 ∧ d ∣ P n :=
by
  intro n h
  sorry

end largest_divisor_of_Pn_for_even_n_l585_585357


namespace bridget_apples_l585_585564

theorem bridget_apples (x : ℕ) (h1 : x - 2 ≥ 0) (h2 : (x - 2) / 3 = 0 → false)
    (h3 : (2 * (x - 2) / 3) - 5 = 6) : x = 20 :=
by
  sorry

end bridget_apples_l585_585564


namespace billy_cherries_left_l585_585192

def initial_cherries : Nat := 2450
def eaten_cherries : Nat := 1625
def given_away_cherries (remaining_cherries : Nat) : Nat := remaining_cherries / 2

theorem billy_cherries_left :
  let remaining_cherries := initial_cherries - eaten_cherries
  ∧ let gave_away := 412
  in remaining_cherries - gave_away = 413 := by
  sorry

end billy_cherries_left_l585_585192


namespace sum_of_m_max_area_l585_585650

def area_of_triangle (x1 y1 x2 y2 x3 y3 : ℚ) : ℚ :=
  (1 / 2 : ℚ) * abs (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))

theorem sum_of_m_max_area :
  let p1 := (2 : ℚ, 3 : ℚ)
  let p2 := (10 : ℚ, 9 : ℚ)
  let line_eq (x : ℚ) := (3 / 4) * x + 3 / 2
  let possible_m : set ℚ := {m | m ≠ line_eq 6}
  let m1 := 5
  let m2 := 7
  m1 ∈ possible_m ∧ m2 ∈ possible_m →
  area_of_triangle 2 3 10 9 6 m1 = area_of_triangle 2 3 10 9 6 m2 →
  (m1 + m2 = 12) :=
by
  sorry

end sum_of_m_max_area_l585_585650


namespace find_length_bisector_AD_l585_585340

noncomputable def length_of_angle_bisector_AD (A B C : ℝ × ℝ) : ℝ :=
  let |AB| := Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)
  let |AC| := Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2)
  let ratio := |AB| / |AC|
  let D := ((B.1 * 1 + C.1 * ratio) / (1 + ratio), (B.2 * 1 + C.2 * ratio) / (1 + ratio))
  Real.sqrt ((A.1 - D.1)^2 + (A.2 - D.2)^2)

theorem find_length_bisector_AD (A B C : ℝ × ℝ) (hA : A = (5, -1)) (hB : B = (-1, 7)) (hC : C = (1, 2)) :
  length_of_angle_bisector_AD A B C = (14 * Real.sqrt 2) / 3 :=
by
  rw [hA, hB, hC]
  sorry

end find_length_bisector_AD_l585_585340


namespace smallest_four_digit_multiple_of_18_l585_585959

theorem smallest_four_digit_multiple_of_18 : ∃ n: ℕ, (1000 ≤ n ∧ n < 10000 ∧ 18 ∣ n) ∧ ∀ m: ℕ, (1000 ≤ m ∧ m < n ∧ 18 ∣ m) → false :=
begin
  use 1008,
  split,
  { split,
    { -- prove 1000 ≤ 1008
      linarith,
    },
    { split,
      { -- prove 1008 < 10000
        linarith,
      },
      { -- prove 18 ∣ 1008
        norm_num,
      }
    }
  },
  { -- prove there is no four-digit multiple of 18 less than 1008
    intros m,
    intro h,
    cases h with h1 h2,
    cases h2 with h3 h4,
    linarith,
  }
end

end smallest_four_digit_multiple_of_18_l585_585959


namespace angle_ratio_l585_585636

theorem angle_ratio (x y α β : ℝ)
  (h1 : y = x + β)
  (h2 : 2 * y = 2 * x + α) :
  α / β = 2 :=
by
  sorry

end angle_ratio_l585_585636


namespace max_S_n_value_l585_585363

theorem max_S_n_value (a : ℕ → ℤ) (d : ℤ)
  (h_arithmetic : ∀ n, a (n + 1) - a n = d)
  (h_d_neg : d < 0)
  (h_S8_S12 : (∑ i in Finset.range 8, a i) = (∑ i in Finset.range 12, a i)) :
  ∃ n : ℕ, n = 10 :=
by
  sorry

end max_S_n_value_l585_585363


namespace width_of_rectangle_l585_585612

theorem width_of_rectangle (s : ℝ) : 
  let area_square := s^2,
      diagonal := s * sqrt 2,
      area_rectangle := diagonal * (area_square / diagonal)
  in 
  diagonal * (area_square / diagonal) = area_square → 
  (area_square / diagonal) = s / sqrt 2 := 
by sorry

end width_of_rectangle_l585_585612


namespace sum_of_possible_m_values_l585_585123

theorem sum_of_possible_m_values : 
  sum (setOf m in {m | 4 < m ∧ m < 18}) = 132 :=
by
  sorry

end sum_of_possible_m_values_l585_585123


namespace sum_of_all_possible_m_values_l585_585380

def g (x : ℝ) (m : ℝ) : ℝ :=
if x < m then x^2 - 3 else 3 * x + 4

theorem sum_of_all_possible_m_values :
  (∃ m1 m2 : ℝ, m1^2 - 3 = 3 * m1 + 4 ∧ m2^2 - 3 = 3 * m2 + 4 ∧ m1 + m2 = 3) :=
sorry

end sum_of_all_possible_m_values_l585_585380


namespace candies_eaten_l585_585156

variables (A B D : ℕ)

-- Conditions:
def condition1 : Prop := ∃ k1 k2 k3 : ℕ, k1 * 4 + k2 * 3 + k3 * 7 = 70
def condition2 : Prop := (B * 3 = A * 4) ∧ (D * 7 = A * 6)
def condition3 : Prop := A + B + D = 70

-- Theorem statement:
theorem candies_eaten (h1 : condition1) (h2 : condition2) (h3 : condition3) :
    A = 24 ∧ B = 18 ∧ D = 28 := sorry

end candies_eaten_l585_585156


namespace main_theorem_l585_585243

-- Definitions based on given conditions
variables (a b : ℝ)
noncomputable def C := {p : ℝ × ℝ | (p.1^2) / (a^2) + (p.2^2) / (b^2) = 1}
noncomputable def e : ℝ := (sqrt 2) / 2
noncomputable def minor_axis_length : ℝ := 2 * (sqrt 2)
noncomputable def a' : ℝ := 2
noncomputable def b' : ℝ := sqrt 2
noncomputable def ellipse_eq : Prop := a = a' ∧ b = b'

-- Given points
variables (A B M N : ℝ × ℝ)
noncomputable def A := (-2, 0)
noncomputable def B := (2, 0)
noncomputable def on_ellipse (p : ℝ × ℝ) : Prop :=
  (p.1^2) / (4) + (p.2^2) / (2) = 1

-- Slopes definitions
noncomputable def k_MA : ℝ := (M.2 - A.2) / (M.1 - A.1)
noncomputable def k_MB : ℝ := (M.2 - B.2) / (M.1 - B.1)
noncomputable def k_NB : ℝ := (N.2 - B.2) / (N.1 - B.1)

-- Slope condition
noncomputable def slope_condition : Prop := k_NB = 2 * k_MA

-- Fixed point
noncomputable def fixed_point : ℝ × ℝ := (2 / 3, 0)

-- Proving conditions
theorem main_theorem :
  ellipse_eq ∧
  ∀ M, on_ellipse M → M ≠ A ∧ M ≠ B →
  k_MA * k_MB = -1 / 2 ∧
  (∀ N, on_ellipse N → N ≠ A ∧ N ≠ B ∧ N ≠ M →
  slope_condition →
  ∃ P, P = fixed_point) :=
by sorry

end main_theorem_l585_585243


namespace greatest_drop_in_price_l585_585035

theorem greatest_drop_in_price (jan feb mar apr may jun : ℝ)
  (h_jan : jan = -0.50)
  (h_feb : feb = 2.00)
  (h_mar : mar = -2.50)
  (h_apr : apr = 3.00)
  (h_may : may = -0.50)
  (h_jun : jun = -2.00) :
  mar = -2.50 ∧ (mar ≤ jan ∧ mar ≤ may ∧ mar ≤ jun) :=
by
  sorry

end greatest_drop_in_price_l585_585035


namespace plumber_total_cost_l585_585893

variable (copperLength : ℕ) (plasticLength : ℕ) (costPerMeter : ℕ)
variable (condition1 : copperLength = 10)
variable (condition2 : plasticLength = copperLength + 5)
variable (condition3 : costPerMeter = 4)

theorem plumber_total_cost (copperLength plasticLength costPerMeter : ℕ)
  (condition1 : copperLength = 10)
  (condition2 : plasticLength = copperLength + 5)
  (condition3 : costPerMeter = 4) :
  copperLength * costPerMeter + plasticLength * costPerMeter = 100 := by
  sorry

end plumber_total_cost_l585_585893


namespace find_x0_l585_585638

-- Define the given function
def f (x : ℝ) : ℝ := Real.log 2 (x + 2)

-- Define the condition f(x_0) = 2
theorem find_x0 (x0 : ℝ) (h : f x0 = 2) : x0 = 2 :=
sorry

end find_x0_l585_585638


namespace hyperbola_focus_coordinates_l585_585574

theorem hyperbola_focus_coordinates :
  ∀ (x y : ℝ), (x = 5) ∧ (y = -8 + real.sqrt 58) ↔
    ((x - 5) ^ 2) / (7 ^ 2) - ((y + 8) ^ 2) / (3 ^ 2) = 1 :=
by
  sorry

end hyperbola_focus_coordinates_l585_585574


namespace candy_eating_l585_585180

-- Definitions based on the conditions
def candies (andrey_rate boris_rate denis_rate : ℕ) (andrey_candies boris_candies denis_candies total_candies : ℕ) : Prop :=
  andrey_rate = 4 ∧ boris_rate = 3 ∧ denis_rate = 7 ∧ andrey_candies = 24 ∧ boris_candies = 18 ∧ denis_candies = 28 ∧
  total_candies = andrey_candies + boris_candies + denis_candies

-- Problem statement
theorem candy_eating :
  ∃ (a b d : ℕ), 
    candies 4 3 7 a b d 70 :=
sorry

end candy_eating_l585_585180


namespace seq_satisfies_recurrence_sq_seq_satisfies_recurrence_cube_l585_585851

-- Define the sequences
def a_sq (n : ℕ) : ℕ := n ^ 2
def a_cube (n : ℕ) : ℕ := n ^ 3

-- First proof problem statement
theorem seq_satisfies_recurrence_sq :
  (a_sq 0 = 0) ∧ (a_sq 1 = 1) ∧ (a_sq 2 = 4) ∧ (a_sq 3 = 9) ∧ (a_sq 4 = 16) →
  (∀ n : ℕ, n ≥ 3 → a_sq n = 3 * a_sq (n - 1) - 3 * a_sq (n - 2) + a_sq (n - 3)) :=
by
  sorry

-- Second proof problem statement
theorem seq_satisfies_recurrence_cube :
  (a_cube 0 = 0) ∧ (a_cube 1 = 1) ∧ (a_cube 2 = 8) ∧ (a_cube 3 = 27) ∧ (a_cube 4 = 64) →
  (∀ n : ℕ, n ≥ 4 → a_cube n = 4 * a_cube (n - 1) - 6 * a_cube (n - 2) + 4 * a_cube (n - 3) - a_cube (n - 4)) :=
by
  sorry

end seq_satisfies_recurrence_sq_seq_satisfies_recurrence_cube_l585_585851


namespace max_sum_of_exponents_l585_585250

theorem max_sum_of_exponents (x y : ℝ) (h : 2^x + 2^y = 1) : 
  x + y ≤ -2 ∧ (x + y = -2 → x = -1 ∧ y = -1) :=
by 
  sorry

end max_sum_of_exponents_l585_585250


namespace candies_eaten_l585_585169

-- Definitions

def Andrey_rate_eq_Boris_rate (candies_eaten_by_Andrey candies_eaten_by_Boris : ℕ) : Prop :=
  candies_eaten_by_Andrey / 4 = candies_eaten_by_Boris / 3

def Denis_rate_eq_Andrey_rate (candies_eaten_by_Denis candies_eaten_by_Andrey : ℕ) : Prop :=
  candies_eaten_by_Denis / 7 = candies_eaten_by_Andrey / 6

def total_candies (candies_eaten_by_Andrey candies_eaten_by_Boris candies_eaten_by_Denis : ℕ) : Prop :=
  candies_eaten_by_Andrey + candies_eaten_by_Boris + candies_eaten_by_Denis = 70

-- Theorem to prove the candies eaten by Andrey, Boris, and Denis
theorem candies_eaten (candies_eaten_by_Andrey candies_eaten_by_Boris candies_eaten_by_Denis : ℕ) :
  Andrey_rate_eq_Boris_rate candies_eaten_by_Andrey candies_eaten_by_Boris →
  Denis_rate_eq_Andrey_rate candies_eaten_by_Denis candies_eaten_by_Andrey →
  total_candies candies_eaten_by_Andrey candies_eaten_by_Boris candies_eaten_by_Denis →
  candies_eaten_by_Andrey = 24 ∧ candies_eaten_by_Boris = 18 ∧ candies_eaten_by_Denis = 28 :=
  by sorry

end candies_eaten_l585_585169


namespace smallest_four_digit_multiple_of_18_l585_585965

theorem smallest_four_digit_multiple_of_18 : 
  ∃ (n : ℤ), 1000 ≤ n ∧ n < 10000 ∧ 18 ∣ n ∧ 
  ∀ (m : ℤ), (1000 ≤ m ∧ m < 10000 ∧ 18 ∣ m) → n ≤ m :=
sorry

end smallest_four_digit_multiple_of_18_l585_585965


namespace value_of_x_l585_585076

-- Define variables and conditions
def consecutive (x y z : ℤ) : Prop := x = z + 2 ∧ y = z + 1

-- Main proposition
theorem value_of_x (x y z : ℤ) (h1 : consecutive x y z) (h2 : z = 2) (h3 : 2 * x + 3 * y + 3 * z = 5 * y + 8) : x = 4 :=
by
  sorry

end value_of_x_l585_585076


namespace network_connections_l585_585042

theorem network_connections (n k : ℕ) (h_n : n = 25) (h_k : k = 4) :
  (n * k) / 2 = 50 :=
by
  rw [h_n, h_k]
  norm_num
  sorry

end network_connections_l585_585042


namespace angela_age_in_ten_years_l585_585551

variables (A B C : ℕ)

-- Conditions
def condition1 : Prop := A = 3 * B
def condition2 : Prop := (A - 10) + (B - 10) = 2 * (C - 10)
def condition3 : Prop := (A + 5) - (B + 5) = C

-- The proof problem
theorem angela_age_in_ten_years (h1 : condition1 A B)
                                (h2 : condition2 A B C)
                                (h3 : condition3 A B C) :
    A + 10 = 25 :=
begin
  sorry
end

end angela_age_in_ten_years_l585_585551


namespace find_N_between_9_5_and_10_5_l585_585944

theorem find_N_between_9_5_and_10_5 :
  ∃ N : ℕ, 9.5 < N / 4 ∧ N / 4 < 10.5 ∧ (N % 2 = 1) ∧ (N = 39 ∨ N = 41) :=
by {
  -- This will be the proof section
  sorry
}

end find_N_between_9_5_and_10_5_l585_585944


namespace ratio_ZQ_QX_l585_585338

-- Definitions based on given conditions
variable {X Y Z E N Q : Type}
variable [TriangleXYZ : Triangle X Y Z]
variable (XY : ℝ) (XZ : ℝ)
variable [AngleBisector : AngleBisector X Y Z E]
variable [Midpoint : Midpoint X E N]
variable [Intersection : Intersects Y N X Z Q]

-- Setting the specific lengths as given in the problem
def XY_length : ℝ := 15
def XZ_length : ℝ := 9

-- The ratio that needs to be proven
theorem ratio_ZQ_QX : (ZQ / QX) = (1 / 2) :=
  sorry

end ratio_ZQ_QX_l585_585338


namespace MK_perp_AD_l585_585912

-- Define the geometric setup and conditions
variables (A B C D K E F M : Type) [Point A] [Point B] [Point C] [Point D] [Point K] [Point E] [Point F] [Point M]

-- Assume ABCD is a rectangle
variable (ABCD_is_rectangle : Rectangle A B C D)

-- Points K are connected to vertices A and D
variable (AK : Line A K)
variable (DK : Line D K)

-- Define the conditions of the perpendiculars and their intersections
variable (BE : Perpendicular B E DK)
variable (CF : Perpendicular C F AK)
variable (M_intersection : Intersection BE CF M)

-- The statement we want to prove
theorem MK_perp_AD (M_ne_K : M ≠ K) : MK ⊥ AD := by 
  sorry

end MK_perp_AD_l585_585912


namespace sum_possible_m_values_l585_585135

theorem sum_possible_m_values : 
  let m_values := {m : ℕ | 4 < m ∧ m < 18}
  let sum_m := ∑ m in m_values, m
  sum_m = 143 :=
by
  sorry

end sum_possible_m_values_l585_585135


namespace product_of_primitive_polynomials_is_primitive_l585_585304

-- Define that a polynomial is primitive
def is_primitive (f : Polynomial ℤ) : Prop :=
  Int.gcd f.coeff.range.toList = 1

-- Main theorem to be proved: the product of two primitive polynomials is primitive
theorem product_of_primitive_polynomials_is_primitive
  (f g : Polynomial ℤ) (hf : is_primitive f) (hg : is_primitive g) : is_primitive (f * g) :=
sorry

end product_of_primitive_polynomials_is_primitive_l585_585304


namespace each_squirrel_needs_more_acorns_l585_585087

noncomputable def acorns_needed : ℕ := 300
noncomputable def total_acorns_collected : ℕ := 4500
noncomputable def number_of_squirrels : ℕ := 20

theorem each_squirrel_needs_more_acorns : 
  (acorns_needed - total_acorns_collected / number_of_squirrels) = 75 :=
by
  sorry

end each_squirrel_needs_more_acorns_l585_585087


namespace coefficient_x5_expansion_l585_585424

theorem coefficient_x5_expansion : 
  (λ x : ℝ, (1 + x) * (1 + x)^6).coeff 5 = 21 := by
  sorry

end coefficient_x5_expansion_l585_585424


namespace transfer_both_increases_average_l585_585461

noncomputable def initial_average_A : ℚ := 44.2
noncomputable def initial_students_A : ℕ := 10
noncomputable def initial_sum_A : ℚ := initial_average_A * initial_students_A

noncomputable def initial_average_B : ℚ := 38.8
noncomputable def initial_students_B : ℕ := 10
noncomputable def initial_sum_B : ℚ := initial_average_B * initial_students_B

noncomputable def score_Kalinina : ℚ := 41
noncomputable def score_Sidorov : ℚ := 44

theorem transfer_both_increases_average :
  let new_sum_A := initial_sum_A - score_Kalinina - score_Sidorov,
      new_students_A := initial_students_A - 2,
      new_average_A := new_sum_A / new_students_A,
      new_sum_B := initial_sum_B + score_Kalinina + score_Sidorov,
      new_students_B := initial_students_B + 2,
      new_average_B := new_sum_B / new_students_B in
        new_average_A > initial_average_A ∧ new_average_B > initial_average_B :=
by
  sorry

end transfer_both_increases_average_l585_585461


namespace largest_subset_nat_1_to_3000_not_diff_1_4_5_l585_585055

/-- 
The largest set of integers that can be selected from {1, ..., 3000}
such that the difference between any two of them is not 1, 4, or 5
has 1000 elements.
-/
theorem largest_subset_nat_1_to_3000_not_diff_1_4_5 : 
  ∃ (s : Finset ℕ), s.card = 1000 ∧ (∀ x y ∈ s, x ≠ y → (x - y).natAbs ≠ 1 ∧ (x - y).natAbs ≠ 4 ∧ (x - y).natAbs ≠ 5) :=
sorry

end largest_subset_nat_1_to_3000_not_diff_1_4_5_l585_585055


namespace max_kn_l585_585358

theorem max_kn (k n : ℕ) (x : Fin k → ℤ) (y : Fin n → ℤ) (P : ℤ → ℤ) (hP : ∀ i, i ∈ Fin k → P (x i) = 54) 
  (hQ : ∀ j, j ∈ Fin n → P (y j) = 2013) : k * n ≤ 6 :=
sorry

end max_kn_l585_585358


namespace volume_ratio_sphere_cylinder_l585_585116

theorem volume_ratio_sphere_cylinder
  (a : ℝ)
  (h : ∃ (O : point ℝ 3), is_inscribed_in_sphere O (right_cylinder (equilateral_triangle a))
  (r : ℝ)) :
  r = (a * real.sqrt 3) / 6 ∧
  volume (sphere_with_radius r) / volume (cylinder_with_radius_height r ((2 * r)) (equilateral_triangle a.radius)) = (2 * real.sqrt 3) * π / 27 :=
sorry

end volume_ratio_sphere_cylinder_l585_585116


namespace sum_of_squares_l585_585673

-- Define conditions
def condition1 (a b : ℝ) : Prop := a - b = 6
def condition2 (a b : ℝ) : Prop := a * b = 7

-- Define what we want to prove
def target (a b : ℝ) : Prop := a^2 + b^2 = 50

-- Main theorem stating the required proof
theorem sum_of_squares (a b : ℝ) (h1 : condition1 a b) (h2 : condition2 a b) : target a b :=
by sorry

end sum_of_squares_l585_585673


namespace ratio_of_medians_to_sides_l585_585951

theorem ratio_of_medians_to_sides (a b c : ℝ) (m_a m_b m_c : ℝ) 
  (h1: m_a = 1/2 * (2 * b^2 + 2 * c^2 - a^2)^(1/2))
  (h2: m_b = 1/2 * (2 * a^2 + 2 * c^2 - b^2)^(1/2))
  (h3: m_c = 1/2 * (2 * a^2 + 2 * b^2 - c^2)^(1/2)) :
  (m_a*m_a + m_b*m_b + m_c*m_c) / (a*a + b*b + c*c) = 3/4 := 
by 
  sorry

end ratio_of_medians_to_sides_l585_585951


namespace anna_walk_distance_l585_585911

theorem anna_walk_distance (d: ℚ) 
  (hd: 22 * 1.25 - 4 * 1.25 = d)
  (d2: d = 3.7): d = 3.7 :=
by 
  sorry

end anna_walk_distance_l585_585911


namespace largest_of_four_numbers_l585_585903

variables {x y z w : ℕ}

theorem largest_of_four_numbers
  (h1 : x + y + z = 180)
  (h2 : x + y + w = 197)
  (h3 : x + z + w = 208)
  (h4 : y + z + w = 222) :
  max x (max y (max z w)) = 89 :=
sorry

end largest_of_four_numbers_l585_585903


namespace find_m_and_n_l585_585641

def f (x : ℝ) : ℝ := -x^2 / 2 + x

theorem find_m_and_n (m n : ℝ) (h_min : ∀ x ∈ Set.Icc m n, f x ≥ f m) (h_max : ∀ x ∈ Set.Icc m n, f x ≤ f n) :
  m = -2 ∧ n = 0 :=
by
  sorry

end find_m_and_n_l585_585641


namespace cos_theta_range_l585_585108

-- Mathematical definitions directly derived from the conditions
noncomputable def circle1 : set (ℝ × ℝ) := {p | p.1^2 + p.2^2 - 6*p.1 - 8*p.2 + 21 = 0}
noncomputable def circle2 : set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 4}
def is_on_circle1 (P : ℝ × ℝ) : Prop := P ∈ circle1
def intersects_on_circle2 (P A B : ℝ × ℝ) : Prop := A ∈ circle2 ∧ B ∈ circle2

-- The theorem statement
theorem cos_theta_range (P A B : ℝ × ℝ) (θ : ℝ) 
  (hP : is_on_circle1 P) 
  (hIntersects : intersects_on_circle2 P A B) :
  ∃ PO, 3 ≤ PO ∧ PO ≤ 7 ∧ θ = angle Ɛ ↼PA Ɛ ↼PB ∧ 
  (PO = dist P (0,0)) ∧ 
  (1 / 9 : ℝ) ≤ cos θ ∧ cos θ ≤ (41 / 49 : ℝ) :=
sorry

end cos_theta_range_l585_585108


namespace num_two_digit_numbers_with_digit_sum_10_l585_585290

theorem num_two_digit_numbers_with_digit_sum_10 : 
  ∃ n, n = 9 ∧ ∀ a b, 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ a + b = 10 → ∃ m, 10 * a + b = m :=
sorry

end num_two_digit_numbers_with_digit_sum_10_l585_585290


namespace find_f_4_l585_585789

noncomputable def f : ℝ → ℝ := sorry

axiom functional_equation : ∀ x : ℝ, f x + 2 * f (1 - x) = 3 * x^2

theorem find_f_4 : f 4 = 2 := 
by {
    -- The proof is omitted as per the task.
    sorry
}

end find_f_4_l585_585789


namespace final_temperature_l585_585716

variable (initial_temp : ℝ := 40)
variable (double_temp : ℝ := initial_temp * 2)
variable (reduce_by_dad : ℝ := double_temp - 30)
variable (reduce_by_mother : ℝ := reduce_by_dad * 0.70)
variable (increase_by_sister : ℝ := reduce_by_mother + 24)

theorem final_temperature : increase_by_sister = 59 := by
  sorry

end final_temperature_l585_585716


namespace smallest_angle_in_15_sided_polygon_arithmetic_sequence_l585_585006

theorem smallest_angle_in_15_sided_polygon_arithmetic_sequence
  (a d : ℕ) 
  (angles : Fin 15 → ℕ)
  (h_seq : ∀ i : Fin 15, angles i = a + i * d)
  (h_convex : ∀ i : Fin 15, angles i < 180)
  (h_sum : ∑ i, angles i = 2340) : 
  a = 135 := 
sorry

end smallest_angle_in_15_sided_polygon_arithmetic_sequence_l585_585006


namespace plumber_total_cost_l585_585894

variable (copperLength : ℕ) (plasticLength : ℕ) (costPerMeter : ℕ)
variable (condition1 : copperLength = 10)
variable (condition2 : plasticLength = copperLength + 5)
variable (condition3 : costPerMeter = 4)

theorem plumber_total_cost (copperLength plasticLength costPerMeter : ℕ)
  (condition1 : copperLength = 10)
  (condition2 : plasticLength = copperLength + 5)
  (condition3 : costPerMeter = 4) :
  copperLength * costPerMeter + plasticLength * costPerMeter = 100 := by
  sorry

end plumber_total_cost_l585_585894


namespace solve_for_x_l585_585406

theorem solve_for_x (x : ℝ) (h : 64^(3 * x) = 16^(4 * x - 5)) : x = -10 := 
by
  sorry

end solve_for_x_l585_585406


namespace kiril_age_problem_l585_585352

theorem kiril_age_problem (x : ℕ) (h1 : x % 5 = 0) (h2 : (x - 1) % 7 = 0) : 26 - x = 11 :=
by
  sorry

end kiril_age_problem_l585_585352


namespace b_a_range_l585_585321
open Real

-- Definitions of angles A, B, and sides a, b in an acute triangle ABC we assume that these are given.
variables {A B C a b c : ℝ}
variable {ABC_acute : A + B + C = π}
variable {angle_condition : B = 2 * A}
variable {sides : a = b * (sin A / sin B)}

theorem b_a_range (h₁ : 0 < A) (h₂ : A < π/2) (h₃ : 0 < C) (h₄ : C < π/2) :
  (∃ A, 30 * (π/180) < A ∧ A < 45 * (π/180)) → 
  (∃ b a, b / a = 2 * cos A) → 
  (∃ x : ℝ, x = b / a ∧ sqrt 2 < x ∧ x < sqrt 3) :=
sorry

end b_a_range_l585_585321


namespace determine_x_l585_585932

theorem determine_x : ∀ (x y z w : ℕ),
  w = 90 →
  z = w + 25 →
  y = z + 12 →
  x = y + 7 →
  x = 134 :=
by
  intros x y z w hw hz hy hx
  rw [hw, hz, hy, hx]
  sorry

end determine_x_l585_585932


namespace max_coefficient_terms_l585_585330

theorem max_coefficient_terms (x : ℝ) :
  let n := 8
  let T_3 := 7 * x^2
  let T_4 := 7 * x
  true := by
  sorry

end max_coefficient_terms_l585_585330


namespace sum_of_possible_lengths_l585_585121

theorem sum_of_possible_lengths
  (m : ℕ) 
  (h1 : m < 18)
  (h2 : m > 4) : ∑ i in (Finset.range 13).map (λ x, x + 5) = 143 := by
sorry

end sum_of_possible_lengths_l585_585121


namespace Linda_needs_15_hours_to_cover_fees_l585_585745

def wage : ℝ := 10
def fee_per_college : ℝ := 25
def number_of_colleges : ℝ := 6

theorem Linda_needs_15_hours_to_cover_fees :
  (number_of_colleges * fee_per_college) / wage = 15 := by
  sorry

end Linda_needs_15_hours_to_cover_fees_l585_585745


namespace andrey_boris_denis_eat_candies_l585_585159

def andrey_boris_condition (a b : ℕ) : Prop :=
  a = 4 ∧ b = 3

def andrey_denis_condition (a d : ℕ) : Prop :=
  a = 6 ∧ d = 7

def total_candies_condition (total : ℕ) : Prop :=
  total = 70

theorem andrey_boris_denis_eat_candies :
  ∃ (a b d : ℕ), andrey_boris_condition a b ∧ andrey_denis_condition a d ∧ 
                  (total_candies_condition (2 * (12 + 9 + 14)) ∧ 
                   2 * 12 = 24 ∧ 2 * 9 = 18 ∧ 2 * 14 = 28) →
                  (a = 24 ∧ b = 18 ∧ d = 28) :=
by
  sorry

end andrey_boris_denis_eat_candies_l585_585159


namespace grape_juice_percentage_after_addition_l585_585300

def initial_mixture_volume : ℝ := 40
def initial_grape_juice_percentage : ℝ := 0.10
def added_grape_juice_volume : ℝ := 10

theorem grape_juice_percentage_after_addition :
  ((initial_mixture_volume * initial_grape_juice_percentage + added_grape_juice_volume) /
  (initial_mixture_volume + added_grape_juice_volume)) * 100 = 28 :=
by 
  sorry

end grape_juice_percentage_after_addition_l585_585300


namespace average_monthly_balance_correct_l585_585575

def january_balance : ℕ := 100
def february_balance : ℕ := 200
def march_balance : ℕ := 250
def april_balance : ℕ := 250
def may_balance : ℕ := 150
def june_balance : ℕ := 100

def total_balance : ℕ :=
  january_balance + february_balance + march_balance + april_balance + may_balance + june_balance

def number_of_months : ℕ := 6

def average_monthly_balance : ℕ :=
  total_balance / number_of_months

theorem average_monthly_balance_correct :
  average_monthly_balance = 175 := by
  sorry

end average_monthly_balance_correct_l585_585575


namespace triangle_folding_eq_l585_585530

-- Definition of given conditions
def is_equilateral_triangle (A B C : Type) [metric_space A] (side_length : ℝ) :=
  dist A B = side_length ∧ dist B C = side_length ∧ dist C A = side_length

def touches_on_side (B P : Type) [metric_space B] (dist_B_P : ℝ) :=
  dist B P = dist_B_P

-- The problem statement in Lean 4
theorem triangle_folding_eq (A B C P Q : Type) [metric_space A] 
  (h_eq_triangle : is_equilateral_triangle A B C 10) 
  (h_BP : touches_on_side B P 7)
  (h_P_on_AB : true)  -- Adding trivial condition for P being on AB (to keep the condition check)
  :
  (dist P Q) ^ 2 = 9 :=
sorry

end triangle_folding_eq_l585_585530


namespace fixed_point_Q_l585_585393

theorem fixed_point_Q (M : ℝ × ℝ) (hM : M.snd^2 = 4 * M.fst) :
  ∃ Q : ℝ × ℝ, Q = (1, 0) ∧ (let r := abs (M.fst + 1) in
    (M.fst - 1)^2 + M.snd^2 = r^2) := 
sorry

end fixed_point_Q_l585_585393


namespace sum_of_possible_m_values_l585_585129

theorem sum_of_possible_m_values :
  let m_range := Finset.Icc 5 17 in
  m_range.sum id = 143 := by
  sorry

end sum_of_possible_m_values_l585_585129


namespace log_sum_identity_l585_585451

-- Prove that: lg 8 + 3 * lg 5 = 3

noncomputable def common_logarithm (x : ℝ) : ℝ := Real.log x / Real.log 10

theorem log_sum_identity : 
    common_logarithm 8 + 3 * common_logarithm 5 = 3 := 
by
  sorry

end log_sum_identity_l585_585451


namespace equation_of_line_min_chord_l585_585528

noncomputable def line_through_point_intersects_circle_min_chord_length 
    (l : ℝ → ℝ)
    (cond : ∀ x, l x = x + 1)
    (p : ℝ × ℝ := (0, 1))
    (circle_eq : ∀ x y, (x - 1)^2 + y^2 = 4) :
    Prop :=
  ∀ x y, (l x - y = 1) ↔ (x - y + 1 = 0)

theorem equation_of_line_min_chord :
  ∀ l cond p circle_eq, 
    @line_through_point_intersects_circle_min_chord_length l cond p circle_eq :=
begin
  sorry
end

end equation_of_line_min_chord_l585_585528


namespace NumberOfStudentsEnrolledOnlyInEnglish_l585_585688

-- Definition of the problem's variables and conditions
variables (TotalStudents BothEnglishAndGerman TotalGerman OnlyEnglish OnlyGerman : ℕ)
variables (h1 : TotalStudents = 52)
variables (h2 : BothEnglishAndGerman = 12)
variables (h3 : TotalGerman = 22)
variables (h4 : TotalStudents = OnlyEnglish + OnlyGerman + BothEnglishAndGerman)
variables (h5 : OnlyGerman = TotalGerman - BothEnglishAndGerman)

-- Theorem to prove the number of students enrolled only in English
theorem NumberOfStudentsEnrolledOnlyInEnglish : OnlyEnglish = 30 :=
by
  -- Insert the necessary proof steps here to derive the number of students enrolled only in English from the given conditions
  sorry

end NumberOfStudentsEnrolledOnlyInEnglish_l585_585688


namespace line_position_relative_to_parallel_planes_l585_585303

-- Definitions to represent the conditions
variables {P1 P2 : Plane} {l : Line}

-- Premises
def parallel_planes (P1 P2 : Plane) : Prop :=
  P1.parallel P2

def line_parallel_to_plane (l : Line) (P : Plane) : Prop :=
  l.parallel P

-- Lean 4 statement reflecting the problem
theorem line_position_relative_to_parallel_planes (h1 : parallel_planes P1 P2) (h2 : line_parallel_to_plane l P1) :
  line_parallel_to_plane l P2 ∨ (l ⊆ P2) :=
sorry

end line_position_relative_to_parallel_planes_l585_585303


namespace question_a_question_b_l585_585999

-- Definition of polynomial P(x) in Lean.
def P (x : ℤ) : ℤ := x^4 - 2*x^3 + 3*x^2 - 2*x - 5

-- Question a): Prove that the set of primes for which P(x) is factorable modulo p is exactly {5}.
theorem question_a (p : ℕ) (hp : Nat.Prime p) :
  ¬ ¬ ∃ f g : ℤ[x], (f.degree < P.degree) ∧ (g.degree < P.degree) ∧ (∀ n : ℕ, (P n ≡ (f * g) n [ZMOD p])) ↔ p = 5 :=
sorry

-- Definition of irreducible polynomial Q(x) in Lean.
def Q (x : ℤ) : ℤ := x^6 + 3

-- Question b): Prove that Q(x) is irreducible over ℤ[x] and factorable modulo every prime p.
theorem question_b (hp : Nat.Prime p) :
  irreducible Q ∧ ∀ p : ℕ, Nat.Prime p → ¬ ∀ f g : ℤ[x], (f.degree < Q.degree) ∧ (g.degree < Q.degree) ∧ (∀ n : ℕ, (Q n ≡ (f * g) n [ZMOD p])) :=
sorry

end question_a_question_b_l585_585999


namespace max_modulus_pure_imaginary_l585_585263

open Complex

theorem max_modulus_pure_imaginary (z : ℂ) (h : Im ((z - I) / (z - 1)) = 0) : |z| <= sqrt 2 := sorry

end max_modulus_pure_imaginary_l585_585263


namespace find_larger_number_l585_585675

variable (x y : ℕ)

theorem find_larger_number (h1 : 4 * y = 5 * x) (h2 : y - x = 10) : y = 50 := 
by 
  sorry

end find_larger_number_l585_585675


namespace smallest_four_digit_multiple_of_18_l585_585982

theorem smallest_four_digit_multiple_of_18 : ∃ n : ℕ, (1000 ≤ n) ∧ (n < 10000) ∧ (n % 18 = 0) ∧ (∀ m : ℕ, (1000 ≤ m) ∧ (m < 10000) ∧ (m % 18 = 0) → n ≤ m) ∧ n = 1008 :=
by
  sorry

end smallest_four_digit_multiple_of_18_l585_585982


namespace smallest_four_digit_multiple_of_18_l585_585986

theorem smallest_four_digit_multiple_of_18 (n : ℕ) (h1 : 1000 ≤ n) (h2 : n < 10000) (h3 : n % 18 = 0) : n = 1008 :=
by
  have h4 : n ≥ 1008, sorry
  have h5 : n ≤ 1008, sorry
  exact eq_of_le_of_ge h4 h5

end smallest_four_digit_multiple_of_18_l585_585986


namespace simplest_quadratic_radical_l585_585904

-- Define each of the options as provided in the conditions
def optionA := Real.sqrt 8
def optionB := Real.sqrt 26
def optionC := Real.sqrt (1 / 3)
def optionD := 2 / Real.sqrt 6

-- Theorem to state that among the options, optionB (√26) is the simplest form
theorem simplest_quadratic_radical :
  optionB = Real.sqrt 26 ∧
  optionA = 2 * Real.sqrt 2 ∧
  optionC = Real.sqrt 3 / 3 ∧
  optionD = Real.sqrt 6 / 3 :=
by
  sorry

end simplest_quadratic_radical_l585_585904


namespace sum_possible_m_values_l585_585132

theorem sum_possible_m_values : 
  let m_values := {m : ℕ | 4 < m ∧ m < 18}
  let sum_m := ∑ m in m_values, m
  sum_m = 143 :=
by
  sorry

end sum_possible_m_values_l585_585132


namespace equilateral_triangle_l585_585731

theorem equilateral_triangle (A B C I X Y Z : Type)
  (is_incenter : ∀ {X Y Z}, ∃ (I : Type), 
                   (is_incenter A B C I) ∧
                   (is_incenter B I C X) ∧
                   (is_incenter C I A Y) ∧
                   (is_incenter A I B Z))
  (is_equilateral_triangle_XYZ : ∀ {X Y Z}, is_equilateral X Y Z) :
  is_equilateral A B C := 
sorry

end equilateral_triangle_l585_585731


namespace joe_more_than_double_sara_l585_585767

def sara_height := ∃ S : ℕ, (∃ X : ℕ, 2 * S + X = 82) ∧ S + (2 * S + X) = 120
def joe_height := 82

theorem joe_more_than_double_sara : ∃ X : ℕ, (∃ S : ℕ, 2 * S + X = 82 ∧ S + (2 * S + X) = 120) → X = 6 :=
by
  sorry

end joe_more_than_double_sara_l585_585767


namespace functional_eq_solution_l585_585213

theorem functional_eq_solution (f : ℝ → ℝ) 
  (H : ∀ x y : ℝ, f (2 * x + f y) = x + y + f x) : 
  ∀ x : ℝ, f x = x := 
by 
  sorry

end functional_eq_solution_l585_585213


namespace fencing_cost_proof_l585_585016

noncomputable def totalCostOfFencing (length : ℕ) (breadth : ℕ) (costPerMeter : ℚ) : ℚ :=
  2 * (length + breadth) * costPerMeter

theorem fencing_cost_proof : totalCostOfFencing 56 (56 - 12) 26.50 = 5300 := by
  sorry

end fencing_cost_proof_l585_585016


namespace log_sqrt2_minus_1_eq_neg1_l585_585295

theorem log_sqrt2_minus_1_eq_neg1 : ∀ (x : ℝ), log x (sqrt 2 - 1) = -1 → x = sqrt 2 + 1 :=
by
  intros x h
  sorry

end log_sqrt2_minus_1_eq_neg1_l585_585295


namespace coin_pile_weights_l585_585815

def weights_non_increasing (w : list ℝ) : Prop :=
  ∀ i j, i < j → (w.nth i).get_or_else 0 ≥ (w.nth j).get_or_else 0

theorem coin_pile_weights (x : ℝ) (n m : ℕ) (xs ys : list ℝ) 
  (h_length_xs : xs.length = n) 
  (h_length_ys : ys.length = m) 
  (h_non_increasing_xs : weights_non_increasing xs) 
  (h_non_increasing_ys : weights_non_increasing ys) 
  (h_total_weight : xs.sum = ys.sum)
  (h_heaviest_property : ∀ k : ℕ, 1 ≤ k → k ≤ min xs.length ys.length → xs.take k.sum ≤ ys.take k.sum)
  (hx_pos : 0 < x) : 
  (xs.map (λ z, min z x)).sum ≥ (ys.map (λ z, min z x)).sum :=
sorry

end coin_pile_weights_l585_585815


namespace greatest_x_for_lcm_l585_585793

theorem greatest_x_for_lcm (x : ℕ) (h_lcm : Nat.lcm (Nat.lcm x 15) 21 = 105) : x ≤ 105 :=
by
  sorry

end greatest_x_for_lcm_l585_585793


namespace daily_wage_of_c_correct_l585_585865

-- Define the conditions from the problem
variables (x : ℕ) -- common wage multiplier
variables (a_days b_days c_days : ℕ)
variables (ratio_a ratio_b ratio_c : ℕ)
variables (total_earnings : ℕ)

-- Set values based on the conditions
def a_days := 6
def b_days := 9
def c_days := 4

def ratio_a := 3
def ratio_b := 4
def ratio_c := 5

def total_earnings := 1702

-- Define the earnings
def a_earnings := a_days * ratio_a * x
def b_earnings := b_days * ratio_b * x
def c_earnings := c_days * ratio_c * x

-- Define the total earnings equation
def total := a_earnings + b_earnings + c_earnings

-- Apply the total earnings equation to find x
def find_x (total_earnings_eq : total = total_earnings) : x := total_earnings / (a_days * ratio_a + b_days * ratio_b + c_days * ratio_c)

-- Using the value of x, determine the daily wage of c
def daily_wage_c (value_of_x : x = total_earnings / 74) : ℕ := ratio_c * (total_earnings / 74)

-- The goal: Prove that the daily wage of c is 115
theorem daily_wage_of_c_correct : daily_wage_c (find_x sorry) = 115 := sorry

end daily_wage_of_c_correct_l585_585865


namespace find_k_l585_585630

theorem find_k : ∀ (x y k : ℤ), (x = -y) → (2 * x + 5 * y = k) → (x - 3 * y = 16) → (k = -12) :=
by
  intros x y k h1 h2 h3
  sorry

end find_k_l585_585630


namespace find_value_of_expression_l585_585249

theorem find_value_of_expression (x y z : ℕ) (h1 : x^2 + 12^2 = y^2) (h2 : x^2 + 40^2 = z^2) : x^2 + y^2 - z^2 = -1375 := 
by {
  sorry
}

end find_value_of_expression_l585_585249


namespace tens_digit_of_3_pow_2023_l585_585486

theorem tens_digit_of_3_pow_2023 : (3 ^ 2023 % 100) / 10 = 2 := 
sorry

end tens_digit_of_3_pow_2023_l585_585486


namespace percentage_calculation_l585_585492

theorem percentage_calculation (Part Whole : ℕ) (h1 : Part = 90) (h2 : Whole = 270) : 
  ((Part : ℝ) / (Whole : ℝ) * 100) = 33.33 :=
by
  sorry

end percentage_calculation_l585_585492


namespace probability_two_diamonds_then_ace_l585_585036

variable {deck : Finset (Fin 52)} (cards : list deck) 
  (first_two_diamonds : cards.head ∈ {i | i % 4 = 0} ∧ cards.nth 1 ∈ {i | i % 4 = 0})
  (third_card_ace : cards.nth 2 ∈ {i | i % 13 = 0})

noncomputable def prob_two_diamonds_then_ace : ℚ :=
  if length cards = 3 then (1 / 17) * (29 / 650) else 0

theorem probability_two_diamonds_then_ace :
  prob_two_diamonds_then_ace cards = 29 / 11050 :=
sorry

end probability_two_diamonds_then_ace_l585_585036


namespace geometric_sequence_a3_l585_585707

variable {a : ℕ → ℝ} (h1 : a 1 > 0) (h2 : a 2 * a 4 = 25)
def geometric_sequence (a : ℕ → ℝ) := ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_a3 (h_geom : geometric_sequence a) : 
  a 3 = 5 := 
by
  sorry

end geometric_sequence_a3_l585_585707


namespace largest_value_f12_l585_585730

theorem largest_value_f12 (f : ℝ → ℝ) (hf_poly : ∀ x, f x ≥ 0) 
  (hf_6 : f 6 = 24) (hf_24 : f 24 = 1536) :
  f 12 ≤ 192 :=
sorry

end largest_value_f12_l585_585730


namespace count_elements_matching_conditions_l585_585337

theorem count_elements_matching_conditions : 
  let S := (Finset.range 10000).filter 
    (λ n, (n % 3 = 2) ∧ (n % 5 = 3) ∧ (n % 7 = 4))
  in S.card = 95 := 
by
  let S := (Finset.range 10000).filter 
    (λ n, (n % 3 = 2) ∧ (n % 5 = 3) ∧ (n % 7 = 4))
  have h_S_card : S.card = 95 := sorry
  exact h_S_card

end count_elements_matching_conditions_l585_585337


namespace triangle_cos_A_and_circumradius_l585_585242

theorem triangle_cos_A_and_circumradius (a b c A B C : ℝ)
  (h1 : a / b = 7 / 5)
  (h2 : b / c = 5 / 3)
  (h_area : 0.5 * b * c * Real.sin A = 45 * Real.sqrt 3)
  (h_cos : ∀ a b c, Real.cos A = (b^2 + c^2 - a^2) / (2 * b * c))
  (h_law_of_sines : ∀ a A R, a = 2 * R * Real.sin A) :
  Real.cos A = -1 / 2 ∧ 
  ∃ R, R = 14 := 
sorry

end triangle_cos_A_and_circumradius_l585_585242


namespace xiaopang_initial_salary_l585_585499

theorem xiaopang_initial_salary :
  ∀ (P : ℝ),
  (P * 11 / 10) * 10 / 11 * 13 / 12 * 12 / 13 = 5000 →
  P = 5000 :=
begin
  intros P h,
  calc
    P
    = (P * (11 / 10) * (10 / 11) * (13 / 12) * (12 / 13)) : by { rw mul_assoc, rw mul_assoc, rw mul_assoc, rw mul_assoc }
    ... = 5000 : by { rw ←h },
end

end xiaopang_initial_salary_l585_585499


namespace solve_for_x_l585_585577

noncomputable def solution_x : ℝ := -1011.5

theorem solve_for_x (x : ℝ) (h : (2023 + x)^2 = x^2) : x = solution_x :=
by sorry

end solve_for_x_l585_585577


namespace polynomial_characterization_l585_585945

def is_solution (P : ℝ[X]) : Prop :=
  ∀ a b c : ℝ, ab:=(a*b), bc:=(b*c), ca:=(c*a), ab + bc + ca = 0 →
  (P.eval (a - b) + P.eval (b - c) + P.eval (c - a) = P.eval (a + b + c))

theorem polynomial_characterization :
  ∀ P : ℝ[X], is_solution P →
  ∃ (a b : ℝ), P = Polynomial.C a * Polynomial.X ^ 4 + Polynomial.C b * Polynomial.X ^ 2 :=
by
  intros P hP
  sorry

end polynomial_characterization_l585_585945


namespace max_value_of_A_l585_585279

noncomputable def A (x y : ℝ) : ℝ :=
  (Real.sqrt (Real.cos x * Real.cos y)) / (Real.sqrt (Real.cot x) + Real.sqrt (Real.cot y))

theorem max_value_of_A (x y : ℝ) (h1 : 0 < x) (h2 : x < (π / 2)) (h3 : 0 < y) (h4 : y < (π / 2)) :
  A x y ≤ (Real.sqrt 2) / 4 :=
sorry

end max_value_of_A_l585_585279


namespace problem_ω_problem_a_l585_585625

noncomputable def f (ω x : ℝ) : ℝ :=
  2 * (Real.cos (ω * x / 2))^2 + Real.cos (ω * x + π / 3) - 1

theorem problem_ω (A B : ℝ) (ω : ℝ) (h_f_A : f ω A = 0) (h_f_B : f ω B = 0) (h_AB : abs (A - B) = π / 2)
  (h_ω : ω > 0) : ω = 2 :=
sorry

theorem problem_a (A B C a b c : ℝ) (h_f_A : f 2 A = -3 / 2) (h_c : c = 3) (h_area : (1/2) * b * c * Real.sin A = 3 * Real.sqrt 3)
  (h_acute : A = π / 3) : a = Real.sqrt 13 :=
sorry

end problem_ω_problem_a_l585_585625


namespace vector_addition_l585_585604

def vector := (ℕ × ℕ)

def a : vector := (2, 1)
def b : vector := (-1, 2)

theorem vector_addition :
  2 • a + 3 • b = (1, 8) :=
sorry

end vector_addition_l585_585604


namespace volume_in_cubic_yards_l585_585110

-- Definition: A box with a specific volume in cubic feet.
def volume_in_cubic_feet (v : ℝ) : Prop :=
  v = 200

-- Definition: Conversion factor from cubic feet to cubic yards.
def cubic_feet_per_cubic_yard : ℝ := 27

-- Theorem: The volume of the box in cubic yards given the volume in cubic feet.
theorem volume_in_cubic_yards (v_cubic_feet : ℝ) 
    (h : volume_in_cubic_feet v_cubic_feet) : 
    v_cubic_feet / cubic_feet_per_cubic_yard = 200 / 27 :=
  by
    rw [h]
    sorry

end volume_in_cubic_yards_l585_585110


namespace probability_of_edge_endpoints_l585_585044

-- Define a regular octahedron with vertices and edges
structure RegularOctahedron :=
  (vertices : Finset ℕ) -- Assuming vertices are labeled as natural numbers.
  (edges : Finset (ℕ × ℕ)) -- Edges represented as pairs of vertices.

-- Regular octahedron property
def is_regular_octahedron (O : RegularOctahedron) : Prop :=
  O.vertices.card = 6 ∧ ∀ v ∈ O.vertices, (Finset.filter (λ e, e.fst = v ∨ e.snd = v) O.edges).card = 4

-- Define the problem statement
theorem probability_of_edge_endpoints (O : RegularOctahedron) (h : is_regular_octahedron O) :
  let total_ways_to_choose_vertices := Nat.choose 6 2,
      favorable_outcomes := 12
  in favorable_outcomes / total_ways_to_choose_vertices = 4 / 5 := 
by
  sorry

end probability_of_edge_endpoints_l585_585044


namespace smallest_n_for_c_10e100_l585_585382

def sequence_c : ℕ → ℕ
| 0 := 0  -- c_0 won't be used but defined for completeness
| 1 := 3
| 2 := 6
| (n+3) := sequence_c (n+2) * sequence_c (n+1)

def fib : ℕ → ℕ
| 0 := 0
| 1 := 1
| (n+2) := fib (n+1) + fib n

theorem smallest_n_for_c_10e100 : ∃ n, sequence_c n > 10^100 ∧ n = 45 :=
by
  have h : ∀ n, sequence_c n = 3 ^ (fib n),
    sorry,
  -- Based on the found sequence value and definitions
  use 45,
  -- Prove the conditions
  split,
  { calc
      sequence_c 45
      = 3 ^ (fib 45) : by rw h
      ... > 10 ^ 100 : by norm_num },
  { refl }

end smallest_n_for_c_10e100_l585_585382


namespace area_triangle_ABC_l585_585711

-- Define the given conditions only
theorem area_triangle_ABC:
  ∀ (a b c : ℝ) (A B C : ℝ),
    c = 1 → 
    cos A = -3/5 → 
    sin C = 1/2 →
    (a = 2 * 1 * sqrt (1 - (cos A)^2)) →
    let sin_A := sqrt (1 - (cos A)^2) in
    let sin_B := sin_A * (sqrt 3 / 2) + (-3 / 5) * (1 / 2) in
    let area := 1/2 * a * c * sin_B in
    area = (8 * sqrt 3 - 6) / 25 :=
by
  -- skipping the proof
  intros a b c A B C hc hcosA hsinC ha sin_A sin_B area,
  sorry

end area_triangle_ABC_l585_585711


namespace parabola_y_axis_intersection_l585_585796

theorem parabola_y_axis_intersection:
  (∀ x y : ℝ, y = -2 * (x - 1)^2 - 3 → x = 0 → y = -5) :=
by
  intros x y h_eq h_x
  sorry

end parabola_y_axis_intersection_l585_585796


namespace highest_daily_profit_and_total_profit_l585_585862

def cost_price : ℕ := 6
def standard_price : ℕ := 10

def price_relative (day : ℕ) : ℤ := 
  match day with
  | 1 => 3
  | 2 => 2
  | 3 => 1
  | 4 => -1
  | 5 => -2
  | _ => 0

def quantity_sold (day : ℕ) : ℕ :=
  match day with
  | 1 => 7
  | 2 => 12
  | 3 => 15
  | 4 => 32
  | 5 => 34
  | _ => 0

noncomputable def selling_price (day : ℕ) : ℤ := standard_price + price_relative day

noncomputable def profit_per_pen (day : ℕ) : ℤ := (selling_price day) - cost_price

noncomputable def daily_profit (day : ℕ) : ℤ := (profit_per_pen day) * (quantity_sold day)

theorem highest_daily_profit_and_total_profit 
  (h_highest_profit: daily_profit 4 = 96) 
  (h_total_profit: daily_profit 1 + daily_profit 2 + daily_profit 3 + daily_profit 4 + daily_profit 5 = 360) : 
  True :=
by
  sorry

end highest_daily_profit_and_total_profit_l585_585862


namespace golf_balls_count_l585_585818

theorem golf_balls_count (dozen_count : ℕ) (balls_per_dozen : ℕ) (total_balls : ℕ) 
  (h1 : dozen_count = 13) 
  (h2 : balls_per_dozen = 12) 
  (h3 : total_balls = dozen_count * balls_per_dozen) : 
  total_balls = 156 := 
sorry

end golf_balls_count_l585_585818


namespace smallest_four_digit_multiple_of_18_l585_585964

theorem smallest_four_digit_multiple_of_18 : 
  ∃ (n : ℤ), 1000 ≤ n ∧ n < 10000 ∧ 18 ∣ n ∧ 
  ∀ (m : ℤ), (1000 ≤ m ∧ m < 10000 ∧ 18 ∣ m) → n ≤ m :=
sorry

end smallest_four_digit_multiple_of_18_l585_585964


namespace find_m_value_l585_585632

-- Conditions: Given the quadratic equation and root
def quadratic_equation (m : ℝ) : Prop := ∀ x : ℝ, x^2 + m * x + 3 = 0

-- To show: The value of m is -4
theorem find_m_value (m : ℝ) (h : quadratic_equation m (1 : ℝ)) : m = -4 :=
by
  -- Placeholder for the proof
  sorry

end find_m_value_l585_585632


namespace smallest_multiple_greater_than_50_l585_585061

def lcm (m n : Nat) : Nat := Nat.lcm m n
noncomputable def lcm_9_15 := lcm 9 15

theorem smallest_multiple_greater_than_50 (m n k : Nat) (h1 : m = 9) (h2 : n = 15) (h3 : lcm m n = 45) (h4 : k = 2) : 45 * k > 50 ∧ 45 * k = 90 := by
  simp [h1, h2, h3, h4]
  split
  · exact Nat.mul_pos (by decide) (by decide) -- 45 * 2 > 50
  · rw [Nat.mul_comm, mul_comm]; exact rfl    -- 45 * 2 = 90

#check smallest_multiple_greater_than_50

end smallest_multiple_greater_than_50_l585_585061


namespace joan_apples_l585_585348

def initial_apples : ℕ := 43
def additional_apples : ℕ := 27
def total_apples (initial additional: ℕ) := initial + additional

theorem joan_apples : total_apples initial_apples additional_apples = 70 := by
  sorry

end joan_apples_l585_585348


namespace option_D_forms_triangle_l585_585859

theorem option_D_forms_triangle (a b c : ℝ) (ha : a = 6) (hb : b = 8) (hc : c = 9) : 
  a + b > c ∧ a + c > b ∧ b + c > a := by
  sorry

end option_D_forms_triangle_l585_585859


namespace regular_hexagon_area_inscribed_in_circle_l585_585835

theorem regular_hexagon_area_inscribed_in_circle
  (h : Real.pi * r^2 = 100 * Real.pi) :
  6 * (r^2 * Real.sqrt 3 / 4) = 150 * Real.sqrt 3 :=
by {
  sorry
}

end regular_hexagon_area_inscribed_in_circle_l585_585835


namespace binomials_product_evaluation_l585_585224

-- Define the binomials and the resulting polynomial
def binomial_one (x : ℝ) := 4 * x + 3
def binomial_two (x : ℝ) := 2 * x - 6
def resulting_polynomial (x : ℝ) := 8 * x^2 - 18 * x - 18

-- Define the proof problem
theorem binomials_product_evaluation :
  ∀ (x : ℝ), (binomial_one x) * (binomial_two x) = resulting_polynomial x ∧ 
  resulting_polynomial (-1) = 8 := 
by 
  intro x
  have h1 : (4 * x + 3) * (2 * x - 6) = 8 * x^2 - 18 * x - 18 := sorry
  have h2 : resulting_polynomial (-1) = 8 := sorry
  exact ⟨h1, h2⟩

end binomials_product_evaluation_l585_585224


namespace train_stoppage_time_l585_585217

theorem train_stoppage_time
    (speed_without_stoppages : ℕ)
    (speed_with_stoppages : ℕ)
    (time_unit : ℕ)
    (h1 : speed_without_stoppages = 50)
    (h2 : speed_with_stoppages = 30)
    (h3 : time_unit = 60) :
    (time_unit * (speed_without_stoppages - speed_with_stoppages) / speed_without_stoppages) = 24 :=
by
  sorry

end train_stoppage_time_l585_585217


namespace box_volume_l585_585885

theorem box_volume (length width side_len : ℕ) (h1 : length = 48) (h2 : width = 36) (h3 : side_len = 8) :
  (length - 2 * side_len) * (width - 2 * side_len) * side_len = 5120 := 
by
  rw [h1, h2, h3]
  norm_num
  sorry

end box_volume_l585_585885


namespace partial_fraction_identity_l585_585590

theorem partial_fraction_identity
  (P Q R : ℝ)
  (h1 : -2 = P + Q)
  (h2 : 1 = Q + R)
  (h3 : -1 = P + R) :
  (P, Q, R) = (-2, 0, 1) :=
by
  sorry

end partial_fraction_identity_l585_585590


namespace smallest_four_digit_multiple_of_18_l585_585988

theorem smallest_four_digit_multiple_of_18 (n : ℕ) (h1 : 1000 ≤ n) (h2 : n < 10000) (h3 : n % 18 = 0) : n = 1008 :=
by
  have h4 : n ≥ 1008, sorry
  have h5 : n ≤ 1008, sorry
  exact eq_of_le_of_ge h4 h5

end smallest_four_digit_multiple_of_18_l585_585988


namespace max_sequence_length_l585_585319

theorem max_sequence_length (a : ℕ → ℝ) (n : ℕ) 
  (h7 : ∀ i, 1 ≤ i ∧ i + 6 ≤ n → ∑ k in i..i+6, a k < 0) 
  (h11 : ∀ i, 1 ≤ i ∧ i + 10 ≤ n → ∑ k in i..i+10, a k > 0) : 
  n ≤ 16 := 
sorry

end max_sequence_length_l585_585319


namespace sum_of_special_primes_l585_585576

theorem sum_of_special_primes : 
  (∑ p in finset.filter (λ p, p.prime ∧ p < 20 ∧ ¬ (∃ x : ℤ, (18 * x + 2) % p = (5 % p)))
    (finset.range 20)) = 5 := sorry

end sum_of_special_primes_l585_585576


namespace trajectory_equation_l585_585637

noncomputable def f (x : ℝ) : ℝ := (x - 2) / (x - 1)
noncomputable def g (m x : ℝ) : ℝ := m * x + 1 - m

theorem trajectory_equation 
  (A B P : ℝ × ℝ) 
  (m : ℝ) 
  (hA : ∃ x y, x = 1 - real.sqrt(-1 / m) ∧ y = 1 - m * real.sqrt(-1 / m) ∧ A = (x, y))
  (hB : ∃ x y, x = 1 + real.sqrt(-1 / m) ∧ y = 1 + m * real.sqrt(-1 / m) ∧ B = (x, y))
  (hP: ∃ x y, P = (x, y))
  (h_eq: abs ((A.1 - P.1) + (B.1 - P.1) + (A.2 - P.2) + (B.2 - P.2)) = 2):
  (P.1 - 1) ^ 2 + (P.2 - 1) ^ 2 = 1 := 
sorry

end trajectory_equation_l585_585637


namespace determinant_A_l585_585573

def matrix_A : Matrix (Fin 2) (Fin 2) ℤ :=
  ![-7, 2; 6, -3]

theorem determinant_A : Matrix.det matrix_A = 9 :=
by
  -- skipping the actual proof
  sorry

end determinant_A_l585_585573


namespace calc_result_l585_585568

theorem calc_result (a : ℤ) : 3 * a - 5 * a + a = -a := by
  sorry

end calc_result_l585_585568


namespace solve_for_x_l585_585408

theorem solve_for_x (x : ℝ) : 64^(3 * x) = 16^(4 * x - 5) → x = -10 := 
by 
  sorry

end solve_for_x_l585_585408


namespace remaining_area_exclude_smaller_rectangles_l585_585527

-- Conditions from part a)
variables (x : ℕ)
def large_rectangle_area := (x + 8) * (x + 6)
def small1_rectangle_area := (2 * x - 1) * (x - 1)
def small2_rectangle_area := (x - 3) * (x - 5)

-- Proof statement from part c)
theorem remaining_area_exclude_smaller_rectangles :
  large_rectangle_area x - (small1_rectangle_area x - small2_rectangle_area x) = 25 * x + 62 :=
by
  sorry

end remaining_area_exclude_smaller_rectangles_l585_585527


namespace range_of_x_minus_y_l585_585603

variable (x y : ℝ)
variable (h1 : 2 < x) (h2 : x < 4) (h3 : -1 < y) (h4 : y < 3)

theorem range_of_x_minus_y : -1 < x - y ∧ x - y < 5 := 
by {
  sorry
}

end range_of_x_minus_y_l585_585603


namespace total_songs_megan_bought_l585_585860

-- Definitions for the problem conditions
def country_albums : ℕ := 2
def pop_albums : ℕ := 8
def songs_per_album : ℕ := 7
def total_albums : ℕ := country_albums + pop_albums

-- Theorem stating the conclusion we need to prove
theorem total_songs_megan_bought : total_albums * songs_per_album = 70 :=
by
  sorry

end total_songs_megan_bought_l585_585860


namespace ineq_real_inequality_l585_585735

open Real

theorem ineq_real_inequality 
  (x y z : ℝ) 
  (λ μ ν : ℝ) 
  (h1 : λ ≥ 0)
  (h2 : μ ≥ 0)
  (h3 : ν ≥ 0)
  (h4 : λ ≠ 0 ∨ μ ≠ 0 ∨ ν ≠ 0):
  (x^2 / (λ * x + μ * y + ν * z) +
   y^2 / (λ * y + μ * z + ν * x) +
   z^2 / (λ * z + μ * x + ν * y)) ≥ 
  ((x + y + z) / (λ + μ + ν)) := 
by 
  sorry

end ineq_real_inequality_l585_585735


namespace domain_of_f_l585_585929

noncomputable def f (x : ℝ) : ℝ := (1 / x) - Real.sqrt (x - 1)

theorem domain_of_f : ∀ x, (1 ≤ x) → (∃ y, f x = y) :=
by
  intro x hx
  have h1 : 1 / x ≠ 0 := by sorry
  have h2 : 0 ≤ Real.sqrt (x - 1) := by sorry
  use f x
  sorry

end domain_of_f_l585_585929


namespace percentage_of_votes_won_in_county_x_l585_585695

noncomputable def percentage_win_in_county_x (V : ℕ) : ℕ :=
let total_votes := 3 * V in
let votes_in_county_y := 0.38 * V in
let total_won_votes := 0.54 * total_votes in
let votes_in_county_x := total_won_votes - votes_in_county_y in
let P := (votes_in_county_x / (2 * V)) * 100 in
P

theorem percentage_of_votes_won_in_county_x (V : ℕ) :
  (percentage_win_in_county_x V = 62) :=
by
  unfold percentage_win_in_county_x
  sorry

end percentage_of_votes_won_in_county_x_l585_585695


namespace sqrt_sub_eq_sqrt_l585_585853

theorem sqrt_sub_eq_sqrt (h₁: real.sqrt 8 = real.sqrt (4 * 2))
  (h₂: real.sqrt (4 * 2) = real.sqrt 4 * real.sqrt 2)
  (h₃: real.sqrt 4 = 2)
  (h₄: 2 * real.sqrt 2 - real.sqrt 2 = (2 - 1) * real.sqrt 2) :
  real.sqrt 8 - real.sqrt 2 = real.sqrt 2 :=
by
  have h5 : real.sqrt 8 = 2 * real.sqrt 2, from (Eq.trans h₁ h₂).trans (congr_arg _ h₃),
  calc
    real.sqrt 8 - real.sqrt 2
        = 2 * real.sqrt 2 - real.sqrt 2 : by rw [h5]
    ... = (2 - 1) * real.sqrt 2         : by rwa [mul_sub, mul_one]
    ... = real.sqrt 2                   : by ring

end sqrt_sub_eq_sqrt_l585_585853


namespace time_to_cross_platform_l585_585090

open Real

-- Definitions based on conditions
def train_length := 420 -- meters
def platform_length := 420 -- meters, same as train length
def crossing_pole_time := 30 -- seconds

-- Let speed of the train be V
def train_speed : ℝ := train_length / crossing_pole_time

-- To prove: Time taken to cross the platform
theorem time_to_cross_platform :
  let total_crossing_distance := train_length + platform_length in
  let time_to_cross_platform := total_crossing_distance / train_speed in
  time_to_cross_platform = 60 := by
  sorry

end time_to_cross_platform_l585_585090


namespace max_value_A_l585_585278

noncomputable theory
open Real

/-- Given x and y in the interval (0, π/2), the maximum value of the 
    expression A = sqrt(cos x * cos y) / (sqrt(cot x) + sqrt(cot y)) is sqrt(2) / 4. -/
theorem max_value_A (x y : ℝ) (hx : 0 < x ∧ x < π / 2) (hy : 0 < y ∧ y < π / 2) :
    (∃ x y ∈ Ioo 0 (π / 2), 
      ∀ x y, (0 < x ∧ x < π / 2 ∧ 0 < y ∧ y < π / 2) →
      (A = sqrt(cos x * cos y) / (sqrt (cos x / sin x) + sqrt (cos y / sin y))) ≤ sqrt 2 / 4) :=
sorry

end max_value_A_l585_585278


namespace cupcakes_purchased_l585_585412

-- Definitions based on conditions
def cupcake_price : ℝ := 2
def doughnut_price : ℝ := 1
def apple_pie_price : ℝ := 2
def cookie_price : ℝ := 0.6
def total_spent : ℝ := 33
def num_doughnuts : ℝ := 6
def num_apple_pies : ℝ := 4
def num_cookies : ℝ := 15

-- Calculate the total cost of other items
def total_doughnuts_cost : ℝ := num_doughnuts * doughnut_price
def total_apple_pies_cost : ℝ := num_apple_pies * apple_pie_price
def total_cookies_cost : ℝ := num_cookies * cookie_price

-- Define the equation from the problem
def equation (c : ℝ) : Prop :=
  (cupcake_price * c) + total_doughnuts_cost + total_apple_pies_cost + total_cookies_cost = total_spent

-- Specify the proof objective
theorem cupcakes_purchased : ∃ c : ℝ, equation c ∧ c = 5 :=
by
  use 5
  split
  sorry
  sorry

end cupcakes_purchased_l585_585412


namespace valid_pairs_iff_l585_585925

noncomputable def valid_pairs (a b : ℝ) : Prop :=
  ∃ n : ℕ, n > 0 ∧ a * (⌊ b * n ⌋ : ℝ) = b * (⌊ a * n ⌋ : ℝ)

theorem valid_pairs_iff (a b : ℝ) : valid_pairs a b ↔
  (a = 0 ∨ b = 0 ∨ a = b ∨ (∃ (m n : ℤ), a = m ∧ b = n)) :=
by sorry

end valid_pairs_iff_l585_585925


namespace pool_radius_l585_585100

theorem pool_radius :
  ∃ (r : ℝ), 
    let pool_area := Real.pi * r^2 in 
    let outer_radius := r + 4 in
    let combined_area := Real.pi * outer_radius^2 in
    let wall_area := combined_area - pool_area in
    wall_area = (11 / 25) * pool_area ∧ r = 20 :=
begin
  sorry,
end

end pool_radius_l585_585100


namespace unit_vectors_l585_585854

noncomputable def conditionB : Prop := 
  ∀ a : ℝ × ℝ, a = (-1, 0) → ∥a∥ = 1

noncomputable def conditionC : Prop := 
  ∀ a : ℝ × ℝ, a = (Real.cos (Real.toRadians 38), Real.sin (Real.toRadians 52)) → ∥a∥ = 1

noncomputable def conditionD : Prop := 
  ∀ (a m : ℝ × ℝ), m ≠ (0, 0) → a = (m.1 / ∥m∥, m.2 / ∥m∥) → ∥a∥ = 1

theorem unit_vectors : Prop := 
  conditionB ∧ conditionC ∧ conditionD

-- sorry is used since no proof is required in the statement.
example : unit_vectors := sorry

end unit_vectors_l585_585854


namespace find_coordinates_l585_585218

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^3 + x - 2

-- Define the derivative of f(x)
def f_prime (x : ℝ) : ℝ := 3 * x^2 + 1

-- Define the condition under which the tangent line's slope equals 4 (parallel to y = 4x - 1)
def tangent_parallel (a : ℝ) : Prop := f_prime(a) = 4

-- The main theorem stating the coordinates of point P where the tangent line is parallel to y = 4x - 1
theorem find_coordinates : 
  (∃ a : ℝ, tangent_parallel a ∧ (f a = 0 ∨ f a = -4)) → 
  (∃ P : ℝ × ℝ, (P = (1, 0)) ∨ (P = (-1, -4))) :=
begin
  intro h,
  rcases h with ⟨a, ha1, ha2⟩,
  use if f a = 0 then (1,0) else (-1, -4),
  -- Proof is omitted
  sorry
end

end find_coordinates_l585_585218


namespace son_work_rate_l585_585864

theorem son_work_rate (M S : ℝ) (hM : M = 1 / 5) (hMS : M + S = 1 / 4) : 1 / S = 20 :=
by
  sorry

end son_work_rate_l585_585864


namespace vector_a_properties_l585_585285

-- Definitions of the points in space
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Definition of vector subtraction to find the vector between two points
def vector_sub (p1 p2 : Point3D) : Point3D :=
  { x := p2.x - p1.x, y := p2.y - p1.y, z := p2.z - p1.z }

-- Definition of dot product for vectors
def dot_product (v1 v2 : Point3D) : ℝ :=
  v1.x * v2.x + v1.y * v2.y + v1.z * v2.z

-- Definition of vector magnitude squared for vectors
def magnitude_squared (v : Point3D) : ℝ :=
  v.x * v.x + v.y * v.y + v.z * v.z

-- Main theorem statement
theorem vector_a_properties :
  let A := {x := 0, y := 2, z := 3}
  let B := {x := -2, y := 1, z := 6}
  let C := {x := 1, y := -1, z := 5}
  let AB := vector_sub A B
  let AC := vector_sub A C
  ∀ (a : Point3D), 
    (magnitude_squared a = 3) → 
    (dot_product a AB = 0) → 
    (dot_product a AC = 0) → 
    (a = {x := 1, y := 1, z := 1} ∨ a = {x := -1, y := -1, z := -1}) := 
by
  intros A B C AB AC a ha_magnitude ha_perpendicular_AB ha_perpendicular_AC
  sorry

end vector_a_properties_l585_585285


namespace parallel_lines_slope_l585_585933

theorem parallel_lines_slope (a : ℝ) :
  (∃ (a : ℝ), ∀ x y, (3 * y - a = 9 * x + 1) ∧ (y - 2 = (2 * a - 3) * x)) → a = 3 :=
by
  sorry

end parallel_lines_slope_l585_585933


namespace total_number_of_flowers_l585_585113

theorem total_number_of_flowers
  (rose_row_from_front : ℕ)
  (rose_row_from_back : ℕ)
  (rose_col_from_right : ℕ)
  (rose_col_from_left : ℕ)
  (same_number_of_flowers : Bool)
  (h1 : rose_row_from_front = 7)
  (h2 : rose_row_from_back = 16)
  (h3 : rose_col_from_right = 13)
  (h4 : rose_col_from_left = 9)
  (h5 : same_number_of_flowers = true) :
  (6 + 1 + 15) * (12 + 1 + 8) = 462 := by
  calc
    (6 + 1 + 15) * (12 + 1 + 8) = 22 * 21 := by sorry
    ... = 462 := by sorry

end total_number_of_flowers_l585_585113


namespace probability_of_observing_color_change_l585_585143

def cycle_duration := 100
def observation_interval := 4
def change_times := [45, 50, 100]

def probability_of_change : ℚ :=
  (observation_interval * change_times.length : ℚ) / cycle_duration

theorem probability_of_observing_color_change :
  probability_of_change = 0.12 := by
  -- Proof goes here
  sorry

end probability_of_observing_color_change_l585_585143


namespace distinct_sum_values_l585_585654

open Finset

noncomputable def is_arithmetic_seq (B : ℕ → ℝ) := ∃ d : ℝ, ∀ i : ℕ, B (i+1) = B i + d

theorem distinct_sum_values (n : ℕ) (B : ℕ → ℝ) (h1 : is_arithmetic_seq B) :
  (card (image (λ (p : ℕ × ℕ), B p.1 + B p.2)
     (filter (λ (p : ℕ × ℕ), p.1 < p.2) (range n).product (range n)))) = 2 * n - 3 := sorry

end distinct_sum_values_l585_585654


namespace different_sum_values_count_l585_585191

theorem different_sum_values_count :
  let BagA := {1, 4, 5}
  let BagB := {2, 4, 6}
  let possible_sums := {a + b | a ∈ BagA, b ∈ BagB}
  possible_sums.count = 8 :=
by
  let BagA := {1, 4, 5}
  let BagB := {2, 4, 6}
  let possible_sums := {a + b | a ∈ BagA, b ∈ BagB}
  sorry

end different_sum_values_count_l585_585191


namespace compute_expression_l585_585202

-- Define the conditions as specific values and operations within the theorem itself
theorem compute_expression : 5 + 7 * (2 - 9)^2 = 348 := 
  by
  sorry

end compute_expression_l585_585202


namespace sum_digits_0_to_99_l585_585506

theorem sum_digits_0_to_99 : 
  (∑ i in List.range 18.append (List.range 22.drop 18), (i.to_digits.sum)) = 24 → 
  (∑ i in List.range (100), (i.to_digits.sum)) = 900 :=
by sorry

end sum_digits_0_to_99_l585_585506


namespace segment_length_l585_585059

def cbrt (x : ℝ) : ℝ := x^(1/3)

theorem segment_length (x : ℝ) 
  (h : |x - cbrt 27| = 5) : (abs ((cbrt 27 + 5) - (cbrt 27 - 5)) = 10) :=
by
  sorry

end segment_length_l585_585059


namespace find_S5_l585_585608

variable {a : ℕ → ℝ}
variable {S : ℕ → ℝ}

axiom arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : ∀ n : ℕ, a (n + 1) = a 1 + n * d
axiom sum_first_n_terms (S : ℕ → ℝ) (a : ℕ → ℝ) : ∀ n : ℕ, S n = n / 2 * (a 1 + a n)

theorem find_S5 (h : a 1 + a 3 + a 5 = 3) : S 5 = 5 :=
by
  sorry

end find_S5_l585_585608


namespace area_of_inscribed_hexagon_l585_585831

theorem area_of_inscribed_hexagon (r : ℝ) (h : π * r^2 = 100 * π) : 
  ∃ (A : ℝ), A = 150 * real.sqrt 3 :=
by 
  -- Definitions of necessary geometric entities and properties would be here
  -- Proof would be provided here
  sorry

end area_of_inscribed_hexagon_l585_585831


namespace candies_eaten_l585_585158

variables (A B D : ℕ)

-- Conditions:
def condition1 : Prop := ∃ k1 k2 k3 : ℕ, k1 * 4 + k2 * 3 + k3 * 7 = 70
def condition2 : Prop := (B * 3 = A * 4) ∧ (D * 7 = A * 6)
def condition3 : Prop := A + B + D = 70

-- Theorem statement:
theorem candies_eaten (h1 : condition1) (h2 : condition2) (h3 : condition3) :
    A = 24 ∧ B = 18 ∧ D = 28 := sorry

end candies_eaten_l585_585158


namespace smallest_four_digit_multiple_of_18_l585_585984

theorem smallest_four_digit_multiple_of_18 : ∃ n : ℕ, (1000 ≤ n) ∧ (n < 10000) ∧ (n % 18 = 0) ∧ (∀ m : ℕ, (1000 ≤ m) ∧ (m < 10000) ∧ (m % 18 = 0) → n ≤ m) ∧ n = 1008 :=
by
  sorry

end smallest_four_digit_multiple_of_18_l585_585984


namespace largest_x_l585_585056

-- Define the condition of the problem.
def equation_holds (x : ℝ) : Prop :=
  (5 * x - 20) / (4 * x - 5) ^ 2 + (5 * x - 20) / (4 * x - 5) = 20

-- State the theorem to prove the largest value of x is 9/5.
theorem largest_x : ∃ x : ℝ, equation_holds x ∧ ∀ y : ℝ, equation_holds y → y ≤ 9 / 5 :=
by
  sorry

end largest_x_l585_585056


namespace ball_bounce_height_l585_585093

theorem ball_bounce_height (initial_height : ℝ) (bounce_ratio : ℝ) (threshold_height : ℝ) :
  initial_height = 320 → bounce_ratio = 0.75 → threshold_height = 40 →
  ∃ b : ℕ, (initial_height * (bounce_ratio ^ b) < threshold_height ∧ initial_height * (bounce_ratio ^ (b - 1)) ≥ threshold_height) ∧ b = 6 :=
by
  intros h_initial h_ratio h_threshold
  use 6
  split
  { split
    { rw [h_initial, h_ratio, h_threshold]
      norm_num
      ring },
    { rw [h_initial, h_ratio, h_threshold]
      norm_num
      ring }
  sorry

end ball_bounce_height_l585_585093


namespace find_dividend_l585_585847

-- Definitions from conditions
def divisor : ℕ := 14
def quotient : ℕ := 12
def remainder : ℕ := 8

-- The problem statement to prove
theorem find_dividend : (divisor * quotient + remainder) = 176 := by
  sorry

end find_dividend_l585_585847


namespace min_value_55_l585_585622

noncomputable def min_value (x y : ℝ) (h : x * y + 2 * x + 3 * y = 42) : ℝ :=
  xy + 5x + 4y

theorem min_value_55 (x y : ℝ) (h : x * y + 2 * x + 3 * y = 42) (hx : 0 < x) (hy : 0 < y) : 
  min_value x y h = 55 :=
sorry

end min_value_55_l585_585622


namespace apply_f_2019_times_l585_585274

noncomputable def f (x : ℝ) : ℝ := (1 - x^3) ^ (-1/3 : ℝ)

theorem apply_f_2019_times (x : ℝ) (n : ℕ) (h : n = 2019) (hx : x = 2018) : 
  (f^[n]) x = 2018 :=
by
  sorry

end apply_f_2019_times_l585_585274


namespace trigonometric_inequality_l585_585397

theorem trigonometric_inequality (x : ℝ) (h1 : 0 < x) (h2 : x < π / 2) :
  0 < (1 / (Real.sin x)^2) - (1 / x^2) ∧ (1 / (Real.sin x)^2) - (1 / x^2) < 1 := 
sorry

end trigonometric_inequality_l585_585397


namespace largest_value_of_number_l585_585448

theorem largest_value_of_number 
  (v w x y z : ℝ)
  (h1 : v + w + x + y + z = 8)
  (h2 : v^2 + w^2 + x^2 + y^2 + z^2 = 16) :
  ∃ (m : ℝ), m = 2.4 ∧ (m = v ∨ m = w ∨ m = x ∨ m = y ∨ m = z) :=
sorry

end largest_value_of_number_l585_585448


namespace line_through_parallel_tangent_l585_585592

theorem line_through_parallel_tangent 
  (P : ℝ × ℝ) (M : ℝ × ℝ) 
  (hP : P = (-1, 2)) 
  (hM : M = (1, 1))
  (f : ℝ → ℝ) 
  (hf : f = λ x, 3 * x ^ 2 - 4 * x + 2) 
  (df : ℝ → ℝ)
  (hdf : df = λ x, 6 * x - 4) :
  ∃ (a b c : ℝ), a = 2 ∧ b = -1 ∧ c = 4 ∧ ∀ (x y : ℝ), y = a * x + b → a * x - y + c = 0 :=
by
  sorry

end line_through_parallel_tangent_l585_585592


namespace var_of_scores_l585_585197

def scores : List ℝ := [10.3, 10.3, 10.4, 10.4, 10.8, 10.8, 10.5, 10.4, 10.7, 10.5, 10.7, 10.7, 10.3, 10.6]

def mean (l : List ℝ) : ℝ := l.sum / l.length

def variance (l : List ℝ) : ℝ := 
  let μ := mean l
  (l.map (λ x => (x - μ) ^ 2)).sum / l.length

theorem var_of_scores : abs (variance scores - 0.032) < 0.001 := by
  sorry

end var_of_scores_l585_585197


namespace scooter_travel_time_l585_585094

variable (x : ℝ)
variable (h_speed : x > 0)
variable (h_travel_time : (50 / (x - 1/2)) - (50 / x) = 3/4)

theorem scooter_travel_time : 50 / x = 50 / x := 
  sorry

end scooter_travel_time_l585_585094


namespace smallest_four_digit_multiple_of_18_l585_585969

theorem smallest_four_digit_multiple_of_18 : ∃ n : ℕ, 1000 ≤ n ∧ n ≤ 9999 ∧ n % 18 = 0 ∧ 
  ∀ m : ℕ, 1000 ≤ m ∧ m ≤ 9999 ∧ m % 18 = 0 → n ≤ m :=
begin
  use 1008,
  split,
  { exact nat.le_refl 1008, },
  split,
  { exact nat.le_succ 9999, },
  split,
  { exact nat.mod_eq_zero_of_divisible 1008 18 sorry, },
  { intros m h1 h2 h3,
    apply nat.le_of_sub_nonneg,
    sorry, }
end

end smallest_four_digit_multiple_of_18_l585_585969


namespace sum_of_arith_geo_progression_l585_585479

noncomputable def sum_two_numbers (a b : ℝ) : ℝ :=
  a + b

theorem sum_of_arith_geo_progression : 
  ∃ (a b : ℝ), (∃ d : ℝ, a = 4 + d ∧ b = 4 + 2 * d) ∧ 
  (∃ r : ℝ, a * r = b ∧ b * r = 16) ∧ 
  sum_two_numbers a b = 8 + 6 * Real.sqrt 3 :=
by
  sorry

end sum_of_arith_geo_progression_l585_585479


namespace square_diff_theorem_l585_585395

theorem square_diff_theorem
  (a b c p x : ℝ)
  (h1 : a + b + c = 2 * p)
  (h2 : x = (b^2 + c^2 - a^2) / (2 * c))
  (h3 : c ≠ 0) :
  b^2 - x^2 = 4 / c^2 * (p * (p - a) * (p - b) * (p - c)) := by
  sorry

end square_diff_theorem_l585_585395


namespace find_number_l585_585305

theorem find_number (x a_3 a_4 : ℕ) (h1 : x + a_4 = 5574) (h2 : x + a_3 = 557) : x = 5567 :=
  sorry

end find_number_l585_585305


namespace equilateral_triangle_of_altitudes_intersection_l585_585763

theorem equilateral_triangle_of_altitudes_intersection
  (ABC : Type)
  (A B C H A₁ B₁: ABC)
  (altitude_A : Line ABC)
  (altitude_B : Line ABC)
  (altitude_C : Line ABC)
  (ha : altitude_A ∋ A ∧ altitude_A ∋ A₁ ∧ altitude_A ∋ H)
  (hb : altitude_B ∋ B ∧ altitude_B ∋ B₁ ∧ altitude_B ∋ H)
  (hc : altitude_C ∋ C ∧ altitude_C ∋ H)
  (acute : is_acute_triangle A B C)
  (H_divides_altitudes : (dist A₁ H) * (dist B H) = (dist B₁ H) * (dist A H)) :
  is_equilateral_triangle A B C :=
by
  sorry

end equilateral_triangle_of_altitudes_intersection_l585_585763


namespace curve_is_line_l585_585427

open Real

theorem curve_is_line (t : ℝ) : ∃ m b, ∀ t, y = 5 * t - 7 ↔ y = (5 / 3) * (3 * t + 6 - 6) - 17 := 
sorry

end curve_is_line_l585_585427


namespace box_volume_l585_585883

theorem box_volume (length width side_len : ℕ) (h1 : length = 48) (h2 : width = 36) (h3 : side_len = 8) :
  (length - 2 * side_len) * (width - 2 * side_len) * side_len = 5120 := 
by
  rw [h1, h2, h3]
  norm_num
  sorry

end box_volume_l585_585883


namespace graph_symmetry_translation_l585_585011

theorem graph_symmetry_translation :
  ∃ t : ℝ, t = \(\frac{\pi}{12} \) ∧
  ∀ x : ℝ, sin(2 * (x - t) + \(\frac{\pi}{3})\) = sin(2 * x + \(\frac{\pi}{6})\) ∧
  (sin(2 * x + \(\frac{\pi}{6})\) = 0 → x = -\(\frac{\pi}{12}) ) :=
by
  sorry

end graph_symmetry_translation_l585_585011


namespace integral_inequality_l585_585354

variable {a b : ℝ}
variables (ha : a > 0) (hb : b > 0)

noncomputable def integral_bound (a b : ℝ) : ℝ := 
  ∫ x in (a - 2 * b)..(2 * a - b), 
    abs (sqrt (3 * b * (2 * a - b) + 2 * (a - 2 * b) * x - x^2) - 
          sqrt (3 * a * (2 * b - a) + 2 * (2 * a - b) * x - x^2))

theorem integral_inequality : 
  integral_bound a b ≤ (Real.pi / 3) * (a ^ 2 + b ^ 2) :=
sorry

end integral_inequality_l585_585354


namespace candies_eaten_l585_585167

-- Definitions

def Andrey_rate_eq_Boris_rate (candies_eaten_by_Andrey candies_eaten_by_Boris : ℕ) : Prop :=
  candies_eaten_by_Andrey / 4 = candies_eaten_by_Boris / 3

def Denis_rate_eq_Andrey_rate (candies_eaten_by_Denis candies_eaten_by_Andrey : ℕ) : Prop :=
  candies_eaten_by_Denis / 7 = candies_eaten_by_Andrey / 6

def total_candies (candies_eaten_by_Andrey candies_eaten_by_Boris candies_eaten_by_Denis : ℕ) : Prop :=
  candies_eaten_by_Andrey + candies_eaten_by_Boris + candies_eaten_by_Denis = 70

-- Theorem to prove the candies eaten by Andrey, Boris, and Denis
theorem candies_eaten (candies_eaten_by_Andrey candies_eaten_by_Boris candies_eaten_by_Denis : ℕ) :
  Andrey_rate_eq_Boris_rate candies_eaten_by_Andrey candies_eaten_by_Boris →
  Denis_rate_eq_Andrey_rate candies_eaten_by_Denis candies_eaten_by_Andrey →
  total_candies candies_eaten_by_Andrey candies_eaten_by_Boris candies_eaten_by_Denis →
  candies_eaten_by_Andrey = 24 ∧ candies_eaten_by_Boris = 18 ∧ candies_eaten_by_Denis = 28 :=
  by sorry

end candies_eaten_l585_585167


namespace right_handed_players_count_l585_585390

theorem right_handed_players_count (total_players : ℕ) (throwers : ℕ) (left_handed_percentage : ℚ) 
  (total_players_eq : total_players = 100) 
  (throwers_eq : throwers = 60)
  (left_handed_percentage_eq : left_handed_percentage = 0.40) 
  (throwers_right_handed : ∀ t, t < throwers → true) :
  let non_throwers := total_players - throwers,
      left_handed_non_throwers := (left_handed_percentage * non_throwers).to_nat,
      right_handed_non_throwers := non_throwers - left_handed_non_throwers,
      total_right_handed := throwers + right_handed_non_throwers
  in total_right_handed = 84 :=
by 
  sorry

end right_handed_players_count_l585_585390


namespace k_range_l585_585257

noncomputable def valid_k (k : ℝ) : Prop :=
  ∀ x : ℝ, 0 < x ∧ x < 2 → x / Real.exp x < 1 / (k + 2 * x - x^2)

theorem k_range : {k : ℝ | valid_k k} = {k : ℝ | 0 ≤ k ∧ k < Real.exp 1 - 1} :=
by sorry

end k_range_l585_585257


namespace find_value_of_y_l585_585671

theorem find_value_of_y (x y : ℤ) (h1 : x^2 = y - 3) (h2 : x = 7) : y = 52 :=
by
  sorry

end find_value_of_y_l585_585671


namespace increase_average_grades_l585_585464

theorem increase_average_grades (XA XB : ℝ) (nA nB : ℕ) (k s : ℝ) 
    (hXA : XA = 44.2) (hXB : XB = 38.8) 
    (hnA : nA = 10) (hnB : nB = 10) 
    (hk : k = 41) (hs : s = 44) :
    let new_XA := (XA * nA - k - s) / (nA - 2)
    let new_XB := (XB * nB + k + s) / (nB + 2) in
    (new_XA > XA) ∧ (new_XB > XB) := by
  sorry

end increase_average_grades_l585_585464


namespace minimum_triangle_area_l585_585342

theorem minimum_triangle_area {A P Q K M N : Type}
  (h_angle : ∠ PAQ = 30)
  (h_km : KM = 1)
  (h_kn : KN = 2)
  (h_midpoint : is_midpoint K P Q) :
  triangle_area A P Q ≥ 8 :=
begin
  sorry  -- proof to be provided
end

end minimum_triangle_area_l585_585342


namespace work_completion_days_l585_585096

theorem work_completion_days (A B C : ℕ) (A_rate B_rate C_rate : ℚ) :
  A_rate = 1 / 30 → B_rate = 1 / 55 → C_rate = 1 / 45 →
  1 / (A_rate + B_rate + C_rate) = 55 / 4 :=
by
  intro hA hB hC
  rw [hA, hB, hC]
  sorry

end work_completion_days_l585_585096


namespace notebook_area_l585_585033

variable (w h : ℝ)

def width_to_height_ratio (w h : ℝ) : Prop := w / h = 7 / 5
def perimeter (w h : ℝ) : Prop := 2 * w + 2 * h = 48
def area (w h : ℝ) : ℝ := w * h

theorem notebook_area (w h : ℝ) (ratio : width_to_height_ratio w h) (peri : perimeter w h) :
  area w h = 140 :=
by
  sorry

end notebook_area_l585_585033


namespace distance_per_interval_l585_585291

-- Definitions for the conditions
def total_distance : ℕ := 3  -- miles
def total_time : ℕ := 45  -- minutes
def interval_time : ℕ := 15  -- minutes per interval

-- Mathematical problem statement
theorem distance_per_interval :
  (total_distance / (total_time / interval_time) = 1) :=
by 
  sorry

end distance_per_interval_l585_585291


namespace work_completion_days_l585_585441

-- Conditions
def capacity_A (x : ℝ) := 3 * x
def capacity_B (x : ℝ) := 2 * x
def days_A := 45
def total_work (x : ℝ) := capacity_A x * days_A

-- Definition to prove the combined days
def combined_capacity (x : ℝ) := capacity_A x + capacity_B x
def combined_days (x : ℝ) := total_work x / combined_capacity x

-- Theorem statement to prove
theorem work_completion_days (x : ℝ) : combined_days x = 27 := by
  sorry

end work_completion_days_l585_585441


namespace min_value_Sn_an_l585_585261

-- Definitions as per conditions in part a)
variable {a : ℕ → ℝ} -- a_n is a sequence of positive terms
variable {S : ℕ → ℝ} -- S_n is the sum of the first n terms
variable (d : ℝ) -- common difference in the arithmetic sequence
variable (a_1 : ℝ) -- first term of the arithmetic sequence

-- Assuming the basic definitions and their properties
axiom pos_sequence : ∀ n, a n > 0
axiom sum_sequence : ∀ n, S n = ∑ i in range (n + 1), a i
axiom arithmetic_sequence : ∀ n, a (n + 1) = a_1 + n * d
axiom sqrt_arithmetic_sequence : ∀ n, sqrt (S (n + 1)) - sqrt (S n) = sqrt (S 1) - sqrt (S 0) -- assuming based on arithmetic difference

-- The statement to be proved
theorem min_value_Sn_an (n : ℕ) : ∃ N : ℕ, n = 11 → (S (n + 10) / a n) = 21 := sorry

end min_value_Sn_an_l585_585261


namespace third_number_in_tenth_row_l585_585145

theorem third_number_in_tenth_row : 
  let nth_row_start (n : ℕ) : ℕ := 2^( (n - 1) * n / 2 )
  in let tenth_row_start := nth_row_start 10
  in (tenth_row_start * 2^2) = 2^47 :=
by
  -- nth_row_start defines the first number in the nth row
  -- nth_row_start 10 is the first number in the tenth row, which is 2^45
  -- the third number in the tenth row is hence 2^(45 + 2) = 2^47
  -- Proof is left as an exercise
  sorry

end third_number_in_tenth_row_l585_585145


namespace find_x_l585_585952

theorem find_x : (∀ x: ℝ, (32^(x-2) / 8^(x-2))^2 = 1024^(2*x-1) → x = 1/8) :=
by
  sorry

end find_x_l585_585952


namespace beach_weather_condition_l585_585187

theorem beach_weather_condition
  (T : ℝ) -- Temperature in degrees Fahrenheit
  (sunny : Prop) -- Whether it is sunny
  (crowded : Prop) -- Whether the beach is crowded
  (H1 : ∀ (T : ℝ) (sunny : Prop), (T ≥ 80) ∧ sunny → crowded) -- Condition 1
  (H2 : ¬ crowded) -- Condition 2
  : T < 80 ∨ ¬ sunny := sorry

end beach_weather_condition_l585_585187


namespace minimum_sum_of_distances_l585_585276

theorem minimum_sum_of_distances :
  let l1 := {P : ℝ × ℝ | P.1 = 2}
  let l2 := {Q : ℝ × ℝ | 3 * Q.1 + 5 * Q.2 - 30 = 0}
  let parabola := {R : ℝ × ℝ | (R.2)^2 = -8 * R.1}
  let focus := (-2, 0)
  ∃ P ∈ parabola, ∃ d1 ∈ l1, ∃ d2 ∈ l2,
    (P.1, P.2) = (focus.1, focus.2) ∧ 
    d2 = (abs (3 * focus.1 + 5 * focus.2 - 30) / real.sqrt (3^2 + 5^2)) :=
    abs ((3 * (-2) + 5 * 0 - 30) / real.sqrt (3^2 + 5^2)) = (18 / 17) * real.sqrt 34 :=
by
  sorry

end minimum_sum_of_distances_l585_585276


namespace y_intercept_of_tangent_line_l585_585808

-- Definitions of the conditions
def curve (x : ℝ) : ℝ := x + Real.log x
def point_of_tangency : ℝ × ℝ := (Real.exp 2, Real.exp 2 + 2)

-- Lean 4 statement for the proof problem
theorem y_intercept_of_tangent_line :
  ∃ b : ℝ, (∃ k : ℝ, tangent_line_at_point (curve) point_of_tangency k b ∧ k = 1 + (1 / (Real.exp 2))) ∧ 
  b = 1 :=
  sorry

-- tangent_line_at_point is a placeholder function that encapsulates the usual 
-- notion of a tangent line. It states that a line with slope k and y-intercept b 
-- is tangent to the curve at the given point.
def tangent_line_at_point (f : ℝ → ℝ) (p : ℝ × ℝ) (k b : ℝ) : Prop :=
  ∀ x : ℝ, f x = k * (x - p.1) + p.2

end y_intercept_of_tangent_line_l585_585808


namespace total_surfers_is_60_l585_585811

-- Define the number of surfers in Santa Monica beach
def surfers_santa_monica : ℕ := 20

-- Define the number of surfers in Malibu beach as twice the number of surfers in Santa Monica beach
def surfers_malibu : ℕ := 2 * surfers_santa_monica

-- Define the total number of surfers on both beaches
def total_surfers : ℕ := surfers_santa_monica + surfers_malibu

-- Prove that the total number of surfers is 60
theorem total_surfers_is_60 : total_surfers = 60 := by
  sorry

end total_surfers_is_60_l585_585811


namespace intersection_of_M_and_N_l585_585251

theorem intersection_of_M_and_N :
  let M := {x : ℝ | 2^x < 1}
  let N := {x : ℝ | x^2 - x - 2 < 0}
  M ∩ N = {x : ℝ | -1 < x ∧ x < 0} :=
by
  let M := {x : ℝ | 2^x < 1}
  let N := {x : ℝ | x^2 - x - 2 < 0}
  sorry

end intersection_of_M_and_N_l585_585251


namespace equilateral_triangle_min_xyz_l585_585614

/--
Given an equilateral triangle ABC with side length 4, points D, E, and F are on BC,
CA, and AB respectively such that |AE| = |BF| = |CD| = 1. The lines AD, BE, and CF
intersect pairwise forming the triangle RQS. Point P moves inside triangle PQR and 
along its boundary. Let x, y, and z be the distances from P to the three sides of 
triangle ABC.

1. Prove that when P is at one of the vertices of triangle RQS, the product xyz reaches
   a minimum value.
2. Determine the minimum value of xyz.
-/
theorem equilateral_triangle_min_xyz
  (A B C D E F R Q S P : Type*)
  [Triangle A B C]
  (h1 : length (side A B) = 4)
  (h2 : length (segment A E) = 1)
  (h3 : length (segment B F) = 1)
  (h4 : length (segment C D) = 1)
  (h5 : intersection R A D B E)
  (h6 : intersection Q B E C F)
  (h7 : intersection S A D C F)
  (h8 : inside_triangle P R Q S) :
  ∃ x y z : ℝ, 
    (x = distance P (side A B)) ∧ 
    (y = distance P (side B C)) ∧ 
    (z = distance P (side C A)) ∧ 
    ((P = R ∨ P = Q ∨ P = S) → xyz = (min_value := (648 / 2197) * (2 * sqrt 3))) := sorry

end equilateral_triangle_min_xyz_l585_585614


namespace number_of_total_flowers_l585_585455

theorem number_of_total_flowers :
  let n_pots := 141
  let flowers_per_pot := 71
  n_pots * flowers_per_pot = 10011 :=
by
  sorry

end number_of_total_flowers_l585_585455


namespace smallest_angle_in_convex_15sided_polygon_l585_585001

def isConvexPolygon (n : ℕ) (angles : Fin n → ℚ) : Prop :=
  ∑ i, angles i = (n - 2) * 180 ∧ ∀ i,  angles i < 180

def arithmeticSequence (angles : Fin 15 → ℚ) : Prop :=
  ∃ a d : ℚ, ∀ i : Fin 15, angles i = a + i * d

def increasingSequence (angles : Fin 15 → ℚ) : Prop :=
  ∀ i j : Fin 15, i < j → angles i < angles j

def integerSequence (angles : Fin 15 → ℚ) : Prop :=
  ∀ i : Fin 15, (angles i : ℚ) = angles i

theorem smallest_angle_in_convex_15sided_polygon :
  ∃ (angles : Fin 15 → ℚ),
    isConvexPolygon 15 angles ∧
    arithmeticSequence angles ∧
    increasingSequence angles ∧
    integerSequence angles ∧
    angles 0 = 135 :=
by
  sorry

end smallest_angle_in_convex_15sided_polygon_l585_585001


namespace polar_to_cartesian_l585_585651

theorem polar_to_cartesian (θ ρ : ℝ) (h : ρ = 2 * Real.cos θ) :
  ∃ x y : ℝ, (x=ρ*Real.cos θ ∧ y=ρ*Real.sin θ) ∧ (x-1)^2 + y^2 = 1 :=
by
  sorry

end polar_to_cartesian_l585_585651


namespace conjugate_of_z_l585_585738

theorem conjugate_of_z (z : ℂ) (h : (1 - z) / (1 + z) = complex.I) : conj z = complex.I :=
sorry

end conjugate_of_z_l585_585738


namespace max_ones_in_table_l585_585684

theorem max_ones_in_table (M : Matrix (Fin 3) (Fin 3) ℕ)
  (h_distinct_products : (Π i j, list.prod (M i) ≠ list.prod (M j)) ∧ (Π i j, list.prod (λ k => M k i) ≠ list.prod (λ k => M k j))) :
  ∃ k, (finset.filter (= 1) (Fin 3.univ.bUnion (λ i => Fin 3.univ.map (M i))).card = k) ∧ k = 5 :=
sorry

end max_ones_in_table_l585_585684


namespace number_of_divisors_of_16m_cubed_l585_585368

theorem number_of_divisors_of_16m_cubed (m : ℕ) (h1 : 2 ∣ m) (h2 : (Nat.divisors m).length = 9) : 
  (Nat.divisors (16 * m^3)).length = 29 :=
by
  sorry

end number_of_divisors_of_16m_cubed_l585_585368


namespace exists_point_on_line_with_distance_ratio_l585_585609

structure Line :=
  (x y z : ℝ → ℝ)  -- Parametric form of the line

structure Plane :=
  (a b c d : ℝ)  -- Coefficients of the plane equation ax + by + cz + d = 0

def distance_to_plane (P : Plane) (Q : ℝ × ℝ × ℝ) : ℝ :=
  let (x, y, z) := Q in
  abs (P.a * x + P.b * y + P.c * z + P.d) / sqrt (P.a^2 + P.b^2 + P.c^2)

noncomputable def find_point_on_line_with_ratio 
  (a : Line) (P1 P2 : Plane) (r1 r2 : ℝ) : ℝ × ℝ × ℝ :=
  sorry

theorem exists_point_on_line_with_distance_ratio
  (a : Line) (P1 P2 : Plane) :
  ∃ (Q : ℝ × ℝ × ℝ), 
    (∃ t : ℝ, Q = (a.x t, a.y t, a.z t)) ∧ 
    distance_to_plane P1 Q / distance_to_plane P2 Q = 2 / 3 :=
sorry

end exists_point_on_line_with_distance_ratio_l585_585609


namespace rectangle_distance_sum_l585_585917

noncomputable def distance (p q : (ℝ × ℝ)) : ℝ :=
  real.sqrt ((p.1 - q.1) ^ 2 + (p.2 - q.2) ^ 2)

theorem rectangle_distance_sum :
  let A := (0, 0)
  let M := (1.5, 0)
  let N := (3, 2)
  let O := (1.5, 4)
  let P := (0, 2)
  distance A M + distance A N + distance A O + distance A P = 7.77 + real.sqrt 13 :=
by
  sorry

end rectangle_distance_sum_l585_585917


namespace cost_difference_for_80_copies_l585_585997

def cost_per_copy_X := 1.25
def cost_per_copy_Y := 2.75
def discount_X50_79 := 0.05
def discount_X80_plus := 0.10
def tiered_pricing_Y40_59 := 2.50
def tiered_pricing_Y60_79 := 2.25
def tiered_pricing_Y80_plus := 2.00

def cost_for_X (n : ℕ) : ℝ :=
  if n >= 80 then n * cost_per_copy_X * (1 - discount_X80_plus)
  else if n >= 50 then n * cost_per_copy_X * (1 - discount_X50_79)
  else n * cost_per_copy_X

def cost_for_Y (n : ℕ) : ℝ :=
  if n >= 80 then n * tiered_pricing_Y80_plus
  else if n >= 60 then n * tiered_pricing_Y60_79
  else if n >= 40 then n * tiered_pricing_Y40_59
  else n * cost_per_copy_Y

def diff_in_cost (n : ℕ) : ℝ :=
  cost_for_Y n - cost_for_X n

theorem cost_difference_for_80_copies :
  diff_in_cost 80 = 70 :=
by
  -- todo: Provide proof
  sorry

end cost_difference_for_80_copies_l585_585997


namespace part_a_l585_585070

theorem part_a :
  ∃ (g h : Polynomial ℤ), (∃ c₁, c₁ ≠ 0 ∧ abs(c₁) > 1387 ∧ c₁ ∈ g.coeffs ∨ c₁ ∈ h.coeffs)
  ∧ ∀ c ∈ (Polynomial.coeffs (g * h)), c ∈ {-1, 0, 1} :=
by
  sorry

end part_a_l585_585070


namespace folded_square_length_l585_585706

theorem folded_square_length
  (A B C D E F G : Point)
  (side_length : ℝ)
  (H1 : A ≠ B ∧ B ≠ C ∧ C ≠ D ∧ D ≠ A)
  (H2 : distance A B = side_length ∧ distance B C = side_length ∧ distance C D = side_length ∧ distance D A = side_length)
  (H3 : E = midpoint A D)
  (H4 : F ∈ line_segment C D)
  (H5 : G ∈ line_segment A B)
  (H6 : distance F G = distance F E)
  (side_length_eq_6 : side_length = 6) :
  distance F D = 9 / 4 :=
by
  sorry

end folded_square_length_l585_585706


namespace find_ellipse_equation_find_maximum_area_AOB_l585_585246

noncomputable section

open Real

variables (a b c : ℝ) (x y : ℝ)
variable (A : Point) where
  A_mem_ellipse : A.1^2 / a^2 + A.2^2 / b^2 = 1
  A_eccentricity : (sqrt 2) / 2
  AF2_perpendicular_x_axis : A.1 = 0
  AF2_length : dist A F2 = sqrt 2

variables (O B : Point) where
  O := Point.mk 0 0
  B := Point.mk x (sqrt 2 / 2 * x + 2 * sqrt 2)

variable (triangle_area : ℝ) where
  triangle_area := (1 / 2) * dist O A * dist (line_mk A B) O

theorem find_ellipse_equation : 
  a = 2 * sqrt 2 ∧ b = 2 ∧ c = 2 → 
  (x^2 / 8 + y^2 / 4 = 1) := by sorry

theorem find_maximum_area_AOB : 
  ∃ (A : Point) (B : Point),
  A_mem_ellipse ∧ 
  A_eccentricity ∧
  AF2_perpendicular_x_axis ∧
  AF2_length ∧
  A.1 > 0 ∧ A.2 > 0 ∧
  l_exists_perpendicular_OA ∧
  intersect_ellipse :=
  (triangle_area = 2 * sqrt 2) := by sorry

end find_ellipse_equation_find_maximum_area_AOB_l585_585246


namespace min_arithmetic_series_sum_l585_585806

-- Definitions from the conditions
def arithmetic_sequence (a1 : ℤ) (d : ℤ) (n : ℕ) : ℤ := a1 + (n - 1) * d
def arithmetic_series_sum (a1 : ℤ) (d : ℤ) (n : ℕ) : ℤ := n * (a1 + (n-1) * d / 2)

-- Theorem statement
theorem min_arithmetic_series_sum (a2 a7 : ℤ) (h1 : a2 = -7) (h2 : a7 = 3) :
  ∃ n, (n * (a2 + (n - 1) * 2 / 2) = (n * n) - 10 * n) ∧
  (∀ m, n* (a2 + (m - 1) * 2 / 2) ≥ n * (n * n - 10 * n)) :=
sorry

end min_arithmetic_series_sum_l585_585806


namespace g_has_at_most_1991_roots_l585_585733

noncomputable def g (f : Polynomial ℤ) : Polynomial ℤ :=
  f ^ 2 - Polynomial.C 9

theorem g_has_at_most_1991_roots (f : Polynomial ℤ) (hf : f.degree = 1991) :
  (g f).roots.count (fun x => x ∈ ℤ) ≤ 1991 :=
sorry

end g_has_at_most_1991_roots_l585_585733


namespace Michael_and_truck_meet_exactly_once_l585_585387

-- Define the time variable t
variable (t : ℕ)

-- Michael's position function M(t)
def M (t : ℕ) : ℕ := 4 * t

-- Truck's position function T(t) with its cycle considered
def T (t : ℕ) : ℕ :=
  let cycle_time := 32.5
  let cycles := t / cycle_time
  let residual_time := t % cycle_time
  let distance_in_cycles := 150 * cycles
  let residual_distance :=
    if residual_time <= 12.5 then
      12 * residual_time
    else
      150
  distance_in_cycles + residual_distance

-- Distance function D(t) = T(t) - M(t)
def D (t : ℕ) : ℕ := T t - M t

-- Prove that D(t) = 0 causes exactly one meeting
theorem Michael_and_truck_meet_exactly_once :
  ∃ t : ℕ, D t = 0 := sorry

end Michael_and_truck_meet_exactly_once_l585_585387


namespace find_m_l585_585699

-- Problem statement and conditions
def vect_sum : ℝ × ℝ → ℝ × ℝ → ℝ × ℝ := λ u v, (u.1 + v.1, u.2 + v.2)
def dot_product : ℝ × ℝ → ℝ × ℝ → ℝ := λ u v, u.1 * v.1 + u.2 * v.2

-- Given vectors
def AB := (1, 1)
def AC (m : ℝ) := (2, m)
def BC (m : ℝ) := vect_sum AC m (-AB)

-- Proof that if triangle ABC is a right triangle, then m = -2 or m = 0.
theorem find_m : ∃ m : ℝ, 
  (dot_product AB (BC m) = 0 ∨
  dot_product AB (AC m) = 0 ∨
  dot_product (AC m) (BC m) = 0) ↔ (m = -2 ∨ m = 0) :=
sorry

end find_m_l585_585699


namespace mapril_relatively_prime_days_l585_585184

theorem mapril_relatively_prime_days : 
  ∀ (days : ℕ), days = 28 → (month_number : ℕ), month_number = 8 → 
  (∃ (prime_days : ℕ), prime_days = 14 ∧ 
   ∀ d, 1 ≤ d ∧ d ≤ 28 → gcd d 8 = 1 ↔ d ∉ {2,4,6,8,10,12,14,16,18,20,22,24,26,28}) :=
by
  intro days h_days month_number h_number
  existsi 14
  split
  · rfl -- Proof steps for the correct number of relatively prime days
  · intro d -- Proof steps for relatively prime conditions
    sorry

end mapril_relatively_prime_days_l585_585184


namespace complement_intersection_l585_585283

open Set

def setA : Set ℝ := { x | ∃ y, y = Real.sqrt (x - 4) }
def setB : Set ℝ := { x | -1 ≤ 2 * x - 1 ∧ 2 * x - 1 ≤ 0 }

theorem complement_intersection :
  (compl setA ∩ setB) = { x | 0 ≤ x ∧ x ≤ 1 / 2 } :=
by
  -- Proof goes here
  sorry

end complement_intersection_l585_585283


namespace no_primes_in_factorial_range_l585_585995

theorem no_primes_in_factorial_range (n : ℕ) (hn : n > 1) : 
  ∀ k, n! - (n - 1) ≤ k ∧ k < n! → ¬prime k := 
by
  sorry

end no_primes_in_factorial_range_l585_585995


namespace sphere_radius_eq_3_l585_585453

theorem sphere_radius_eq_3 (r : ℝ) (h : (4/3) * Real.pi * r^3 = 4 * Real.pi * r^2) : r = 3 :=
sorry

end sphere_radius_eq_3_l585_585453


namespace find_x_value_l585_585294

theorem find_x_value (x : ℝ) (log_2 : ℝ) (h1 : log_2 = 0.3010) (h2 : 2^(x+4) = 256) : x = 4 :=
by
  sorry

end find_x_value_l585_585294


namespace range_of_k_l585_585621

theorem range_of_k
  (a b c : ℝ)
  (ha : 0 < a)
  (hb : 0 < b)
  (hc : 0 < c)
  (h1 : a^2 + c^2 = 16)
  (h2 : b^2 + c^2 = 25) : 
  9 < a^2 + b^2 ∧ a^2 + b^2 < 41 :=
by
  sorry

end range_of_k_l585_585621


namespace sum_gcd_lcm_l585_585490

theorem sum_gcd_lcm (a b c d : ℕ) (ha : a = 15) (hb : b = 45) (hc : c = 30) :
  Int.gcd a b + Nat.lcm a c = 45 := 
by
  sorry

end sum_gcd_lcm_l585_585490


namespace transfer_both_increases_average_l585_585460

noncomputable def initial_average_A : ℚ := 44.2
noncomputable def initial_students_A : ℕ := 10
noncomputable def initial_sum_A : ℚ := initial_average_A * initial_students_A

noncomputable def initial_average_B : ℚ := 38.8
noncomputable def initial_students_B : ℕ := 10
noncomputable def initial_sum_B : ℚ := initial_average_B * initial_students_B

noncomputable def score_Kalinina : ℚ := 41
noncomputable def score_Sidorov : ℚ := 44

theorem transfer_both_increases_average :
  let new_sum_A := initial_sum_A - score_Kalinina - score_Sidorov,
      new_students_A := initial_students_A - 2,
      new_average_A := new_sum_A / new_students_A,
      new_sum_B := initial_sum_B + score_Kalinina + score_Sidorov,
      new_students_B := initial_students_B + 2,
      new_average_B := new_sum_B / new_students_B in
        new_average_A > initial_average_A ∧ new_average_B > initial_average_B :=
by
  sorry

end transfer_both_increases_average_l585_585460


namespace complement_of_A_in_U_l585_585658

-- Conditions definitions
def U : Set ℕ := {x | x ≤ 5}
def A : Set ℕ := {x | 2 * x - 5 < 0}

-- Theorem stating the question and the correct answer
theorem complement_of_A_in_U :
  U \ A = {x | 3 ≤ x ∧ x ≤ 5} :=
by
  -- The proof will go here
  sorry

end complement_of_A_in_U_l585_585658


namespace flammable_ice_storage_capacity_l585_585029

theorem flammable_ice_storage_capacity (billion : ℕ) (h : billion = 10^9) : (800 * billion = 8 * 10^11) :=
by
  sorry

end flammable_ice_storage_capacity_l585_585029


namespace developer_profit_decision_l585_585531

-- Define the profit function
def profit (n : ℕ) : ℝ :=
  300000 * n - (810000 + 10000 * n + 20_000 * (n * (n - 1) / 2))

-- Axiom: Establish bounds for net profit starting year
axiom net_profit_starts (n : ℕ) : profit n > 0 → n ≥ 4

-- Define profitability options' evaluation
def total_profit (n : ℕ) (k : ℕ) : ℝ :=
  if k = 1 then profit n + 500000 else profit n + 100000

-- Axiom: Option 2 gives a higher profit
axiom option_comparison (n k : ℕ) : 
  total_profit n k > total_profit n (if k = 1 then 2 else 1) → k = 2

-- Define the final statement and prove by contradiction if necessary
theorem developer_profit_decision (n : ℕ) (k : ℕ) : 
  profit n > 0 → n ≥ 4 ∧ option_comparison n k := by
  intros
  constructor
  apply net_profit_starts
  apply option_comparison
  sorry -- Proof to be filled

end developer_profit_decision_l585_585531


namespace find_A_l585_585914

theorem find_A (A B : ℚ) (h1 : B - A = 211.5) (h2 : B = 10 * A) : A = 23.5 :=
by sorry

end find_A_l585_585914


namespace phase_shift_cosine_l585_585950

theorem phase_shift_cosine (x : ℝ) : 2 * x + (Real.pi / 2) = 0 → x = - (Real.pi / 4) :=
by
  intro h
  sorry

end phase_shift_cosine_l585_585950


namespace rectangle_area_relation_l585_585442

theorem rectangle_area_relation (x : ℝ) (h : x > 0) : ∃ y : ℝ, 2 * (x + (12 - x)) = 24 ∧ y = x * (12 - x) :=
by 
  use x * (12 - x)
  split
  {
    calc 2 * (x + (12 - x)) = 2 * 12 : by rw add_comm ; rw [add_sub_cancel, mul_comm]
    ... = 24 : by norm_num
  }
  {
    refl
  }

end rectangle_area_relation_l585_585442


namespace sum_tens_ones_digit_of_seven_pow_21_l585_585569

theorem sum_tens_ones_digit_of_seven_pow_21 :
  let n := 21 in
  let base := 7 in
  let tens_digit := (base^n / 10) % 10 in
  let ones_digit := base^n % 10 in
  let digit_sum := tens_digit + ones_digit in
  digit_sum = 7 := by
  sorry

end sum_tens_ones_digit_of_seven_pow_21_l585_585569


namespace smallest_four_digit_multiple_of_18_l585_585962

theorem smallest_four_digit_multiple_of_18 : 
  ∃ (n : ℤ), 1000 ≤ n ∧ n < 10000 ∧ 18 ∣ n ∧ 
  ∀ (m : ℤ), (1000 ≤ m ∧ m < 10000 ∧ 18 ∣ m) → n ≤ m :=
sorry

end smallest_four_digit_multiple_of_18_l585_585962


namespace calculate_area_of_shaded_region_l585_585830

namespace Proof

noncomputable def AreaOfShadedRegion (width height : ℝ) (divisions : ℕ) : ℝ :=
  let small_width := width
  let small_height := height / divisions
  let area_of_small := small_width * small_height
  let shaded_in_small := area_of_small / 2
  let total_shaded := divisions * shaded_in_small
  total_shaded

theorem calculate_area_of_shaded_region :
  AreaOfShadedRegion 3 14 4 = 21 := by
  sorry

end Proof

end calculate_area_of_shaded_region_l585_585830


namespace square_of_integer_l585_585724

theorem square_of_integer (n : ℕ) (h : ∃ l : ℤ, l^2 = 1 + 12 * (n^2 : ℤ)) :
  ∃ m : ℤ, 2 + 2 * Int.sqrt (1 + 12 * (n^2 : ℤ)) = m^2 := by
  sorry

end square_of_integer_l585_585724


namespace sum_binom_equals_220_l585_585915

/-- The binomial coefficient C(n, k) -/
def binom (n k : ℕ) : ℕ :=
  if h : k ≤ n then (Nat.choose n k) else 0

/-- Prove that the sum C(2, 2) + C(3, 2) + C(4, 2) + ... + C(11, 2) equals 220 using the 
    binomial coefficient property C(n, r+1) + C(n, r) = C(n+1, r+1) -/
theorem sum_binom_equals_220 :
  binom 2 2 + binom 3 2 + binom 4 2 + binom 5 2 + binom 6 2 + binom 7 2 + 
  binom 8 2 + binom 9 2 + binom 10 2 + binom 11 2 = 220 := by
sorry

end sum_binom_equals_220_l585_585915


namespace puzzle_pieces_l585_585717

theorem puzzle_pieces (x : ℝ) (h : x + 2 * 1.5 * x = 4000) : x = 1000 :=
  sorry

end puzzle_pieces_l585_585717


namespace sum_of_squares_permutations_l585_585652

open List

theorem sum_of_squares_permutations {n : ℕ} (x y z : Fin n → ℝ)
  (hx : Sorted (· ≥ ·) (List.ofFn x))
  (hy : Sorted (· ≥ ·) (List.ofFn y))
  (hz : Perm (List.ofFn y) (List.ofFn z)) :
  ((Finset.univ : Finset (Fin n)).sum (λ i, (x i - y i) ^ 2)) ≤
  ((Finset.univ : Finset (Fin n)).sum (λ i, (x i - z i) ^ 2)) :=
sorry

end sum_of_squares_permutations_l585_585652


namespace minimum_marked_cells_l585_585109

/--
  Given a positive integer k, this statement asserts that the smallest possible number of initially 
  marked cells N needed to ensure every cell on an infinite checkered plane can eventually
  be marked is equal to the sum from i = 1 to k of the ceiling of i/2.
-/
theorem minimum_marked_cells (k : ℕ) (hk : k > 0) : 
  ∃ (N : ℕ), N = (finset.range k).sum (λ i, (i + 1) / 2) := 
sorry

end minimum_marked_cells_l585_585109


namespace shaded_area_percentage_is_correct_l585_585491

noncomputable def total_area_of_square : ℕ := 49

noncomputable def area_of_first_shaded_region : ℕ := 2^2

noncomputable def area_of_second_shaded_region : ℕ := 25 - 9

noncomputable def area_of_third_shaded_region : ℕ := 49 - 36

noncomputable def total_shaded_area : ℕ :=
  area_of_first_shaded_region + area_of_second_shaded_region + area_of_third_shaded_region

noncomputable def percent_shaded_area : ℚ :=
  (total_shaded_area : ℚ) / total_area_of_square * 100

theorem shaded_area_percentage_is_correct :
  percent_shaded_area = 67.35 := by
sorry

end shaded_area_percentage_is_correct_l585_585491


namespace students_at_table_l585_585768

def numStudents (candies : ℕ) (first_last : ℕ) (st_len : ℕ) : Prop :=
  candies - 1 = st_len * first_last

theorem students_at_table 
  (candies : ℕ)
  (first_last : ℕ)
  (st_len : ℕ)
  (h1 : candies = 120) 
  (h2 : first_last = 1) :
  (st_len = 7 ∨ st_len = 17) :=
by
  sorry

end students_at_table_l585_585768


namespace regular_hexagon_area_inscribed_in_circle_l585_585838

theorem regular_hexagon_area_inscribed_in_circle
  (h : Real.pi * r^2 = 100 * Real.pi) :
  6 * (r^2 * Real.sqrt 3 / 4) = 150 * Real.sqrt 3 :=
by {
  sorry
}

end regular_hexagon_area_inscribed_in_circle_l585_585838


namespace sum_of_possible_m_values_l585_585130

theorem sum_of_possible_m_values :
  let m_range := Finset.Icc 5 17 in
  m_range.sum id = 143 := by
  sorry

end sum_of_possible_m_values_l585_585130


namespace rachel_age_when_emily_half_her_age_l585_585582

theorem rachel_age_when_emily_half_her_age (emily_current_age rachel_current_age : ℕ) 
  (h1 : emily_current_age = 20) 
  (h2 : rachel_current_age = 24) 
  (age_difference : ℕ) 
  (h3 : rachel_current_age - emily_current_age = age_difference) 
  (emily_age_when_half : ℕ) 
  (rachel_age_when_half : ℕ) 
  (h4 : emily_age_when_half = rachel_age_when_half / 2)
  (h5 : rachel_age_when_half = emily_age_when_half + age_difference) :
  rachel_age_when_half = 8 :=
by
  sorry

end rachel_age_when_emily_half_her_age_l585_585582


namespace num_members_in_league_l585_585751

-- Definitions based on conditions
def sock_cost : ℕ := 6
def tshirt_cost : ℕ := sock_cost + 7
def shorts_cost : ℕ := tshirt_cost
def total_cost_per_member : ℕ := 2 * (sock_cost + tshirt_cost + shorts_cost)
def total_league_cost : ℕ := 4719

-- Theorem statement
theorem num_members_in_league : (total_league_cost / total_cost_per_member) = 74 :=
by
  sorry

end num_members_in_league_l585_585751


namespace ratio_xz_y2_l585_585994

-- Define the system of equations
def system (k x y z : ℝ) : Prop := 
  x + k * y + 4 * z = 0 ∧ 
  4 * x + k * y - 3 * z = 0 ∧ 
  3 * x + 5 * y - 4 * z = 0

-- Our main theorem to prove the value of xz / y^2 given the system with k = 7.923
theorem ratio_xz_y2 (x y z : ℝ) (h : system 7.923 x y z) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) : 
  ∃ r : ℝ, r = (x * z) / (y ^ 2) :=
sorry

end ratio_xz_y2_l585_585994


namespace smallest_angle_in_15_sided_polygon_arithmetic_sequence_l585_585003

theorem smallest_angle_in_15_sided_polygon_arithmetic_sequence
  (a d : ℕ) 
  (angles : Fin 15 → ℕ)
  (h_seq : ∀ i : Fin 15, angles i = a + i * d)
  (h_convex : ∀ i : Fin 15, angles i < 180)
  (h_sum : ∑ i, angles i = 2340) : 
  a = 135 := 
sorry

end smallest_angle_in_15_sided_polygon_arithmetic_sequence_l585_585003


namespace transfer_both_increases_average_l585_585463

noncomputable def initial_average_A : ℚ := 44.2
noncomputable def initial_students_A : ℕ := 10
noncomputable def initial_sum_A : ℚ := initial_average_A * initial_students_A

noncomputable def initial_average_B : ℚ := 38.8
noncomputable def initial_students_B : ℕ := 10
noncomputable def initial_sum_B : ℚ := initial_average_B * initial_students_B

noncomputable def score_Kalinina : ℚ := 41
noncomputable def score_Sidorov : ℚ := 44

theorem transfer_both_increases_average :
  let new_sum_A := initial_sum_A - score_Kalinina - score_Sidorov,
      new_students_A := initial_students_A - 2,
      new_average_A := new_sum_A / new_students_A,
      new_sum_B := initial_sum_B + score_Kalinina + score_Sidorov,
      new_students_B := initial_students_B + 2,
      new_average_B := new_sum_B / new_students_B in
        new_average_A > initial_average_A ∧ new_average_B > initial_average_B :=
by
  sorry

end transfer_both_increases_average_l585_585463


namespace distinct_values_of_f_l585_585376

noncomputable def f (x : ℝ) := ∑ k in (Finset.range 11).map (λ i, i+2), (floor (k*x) - k * floor x)

theorem distinct_values_of_f : 
  ∀ x, x ≥ 0 → (∑ k in (Finset.range 11).map (λ i, i+2), floor (k*x) - k * floor x).card = 46 :=
sorry

end distinct_values_of_f_l585_585376


namespace number_of_solutions_l585_585223

noncomputable def sign (a : ℝ) : ℝ :=
  if a > 0 then 1 else if a = 0 then 0 else -1

theorem number_of_solutions :
  let N := 3 in
  { (x : ℝ, y : ℝ, z : ℝ) |
    x = 3000 - 3001 * sign (y + z + 3) ∧
    y = 3000 - 3001 * sign (x + z + 3) ∧
    z = 3000 - 3001 * sign (x + y + 3) }.card = N := 
sorry

end number_of_solutions_l585_585223


namespace find_d_from_roots_l585_585240

theorem find_d_from_roots (d : ℝ) (h : ∀ x, x^2 - 3*x + d = 0 ↔ x = (3 + sqrt d) / 2 ∨ x = (3 - sqrt d) / 2) : d = 9 / 5 :=
begin
  sorry
end

end find_d_from_roots_l585_585240


namespace olivia_wallet_after_shopping_l585_585755

variable (initial_wallet : ℝ := 200) 
variable (groceries : ℝ := 65)
variable (shoes_original_price : ℝ := 75)
variable (shoes_discount_rate : ℝ := 0.15)
variable (belt : ℝ := 25)

theorem olivia_wallet_after_shopping :
  initial_wallet - (groceries + (shoes_original_price - shoes_original_price * shoes_discount_rate) + belt) = 46.25 := by
  sorry

end olivia_wallet_after_shopping_l585_585755


namespace num_interesting_quadruples_l585_585924

theorem num_interesting_quadruples : 
  ∑ (a : ℕ) in Finset.Icc 1 15, 
  ∑ (b : ℕ) in Finset.Icc (a + 1) 15, 
  ∑ (c : ℕ) in Finset.Icc (b + 1) 15, 
  ∑ (d : ℕ) in Finset.Icc (c + 1) 15, 
  if a + d > 2 * (b + c) then 1 else 0 = 500 :=
by
  sorry

end num_interesting_quadruples_l585_585924


namespace projectile_first_reaches_35_meters_l585_585429

noncomputable def quadratic_solve (a b c : ℝ) : ℝ × ℝ :=
let discriminant := b^2 - 4 * a * c in
(((-b) + real.sqrt discriminant) / (2 * a), ((-b) - real.sqrt discriminant) / (2 * a))

theorem projectile_first_reaches_35_meters :
  ∃ t > 0, (-6.1 * t^2 + 36.6 * t = 35) ∧
  ∀ t' > 0, (-6.1 * t'^2 + 36.6 * t' = 35) → t' ≥ t :=
begin
  let equation := λ t, -6.1 * t^2 + 36.6 * t - 35,
  have solutions : ℝ × ℝ := quadratic_solve (-6.1) 36.6 (-35),
  cases solutions with t1 t2,
  let valid_solution := if t1 > 0 then t1 else t2,
  use valid_solution,
  split,
  -- Valid time condition and equation check
  { sorry },
  -- First time condition check
  { sorry },
end

end projectile_first_reaches_35_meters_l585_585429


namespace exists_strictly_increasing_function_l585_585934

def is_strictly_increasing {α : Type*} [Preorder α] (f : α → α) : Prop :=
  ∀ ⦃x y⦄, x < y → f x < f y

noncomputable def f : ℕ+ → ℕ+ := sorry

theorem exists_strictly_increasing_function :
  ∃ (f : ℕ+ → ℕ+), is_strictly_increasing f ∧ f 1 = 2 ∧ ∀ n : ℕ+, f (f n) = f n + n :=
begin
  sorry
end

end exists_strictly_increasing_function_l585_585934


namespace solid_is_sphere_l585_585544

-- Define the solid and the property that all its cross-sections with planes are circles.
structure Solid where
  is_circle_section : ∀ (P : Plane), Circle (P.intersection_with_solid)

-- Define that a sphere is a specific type of solid
def is_sphere (s : Solid) : Prop :=
  ∀ (P : Plane), Circle (P.intersection_with_solid s)

-- Prove that a solid with all circular cross-sections is a sphere.
theorem solid_is_sphere (s : Solid) (h: ∀ (P : Plane), Circle (P.intersection_with_solid s)) : is_sphere s :=
by
  sorry

end solid_is_sphere_l585_585544


namespace smallest_four_digit_multiple_of_18_l585_585983

theorem smallest_four_digit_multiple_of_18 : ∃ n : ℕ, (1000 ≤ n) ∧ (n < 10000) ∧ (n % 18 = 0) ∧ (∀ m : ℕ, (1000 ≤ m) ∧ (m < 10000) ∧ (m % 18 = 0) → n ≤ m) ∧ n = 1008 :=
by
  sorry

end smallest_four_digit_multiple_of_18_l585_585983


namespace range_of_m_l585_585736

theorem range_of_m (a b c m : ℝ) (h1 : a > b) (h2 : b > c) 
  (h3 : (1 / (a - b)) + (1 / (b - c)) ≥ m / (a - c)) :
  m ≤ 4 :=
sorry

end range_of_m_l585_585736


namespace number_of_attendees_choosing_water_l585_585936

variables {total_attendees : ℕ} (juice_percent water_percent : ℚ)

-- Conditions
def attendees_juice (total_attendees : ℕ) : ℚ := 0.7 * total_attendees
def attendees_water (total_attendees : ℕ) : ℚ := 0.3 * total_attendees
def attendees_juice_given := (attendees_juice total_attendees) = 140

-- Theorem statement
theorem number_of_attendees_choosing_water 
  (h1 : juice_percent = 0.7) 
  (h2 : water_percent = 0.3) 
  (h3 : attendees_juice total_attendees = 140) : 
  attendees_water total_attendees = 60 :=
sorry

end number_of_attendees_choosing_water_l585_585936


namespace right_triangle_area_floor_l585_585532

theorem right_triangle_area_floor 
  (perimeter : ℝ)
  (perimeter_eq: perimeter = 2008)
  (inscribed_circle_area : ℝ)
  (inscribed_circle_area_eq : inscribed_circle_area = 100 * π ^ 3) : 
  ⌊let s := perimeter / 2
    let r := sqrt (inscribed_circle_area / π)
    let A := r * s
    A⌋ = 31541 := 
by 
  have r_def : r = 10 * π := by 
    rw [inscribed_circle_area_eq, div_eq_iff_eq_mul π_ne_zero]
    rw [sqrt_eq_rpow, (r ^ 2) = (10 * π) ^ 2]
  have s_def : s = 1004 := by 
    rw [perimeter_eq]
    exact 2008 / 2
  have A_def : A = r * s := by 
    rw [r_def, s_def]
  have A_floor : ⌊A⌋ = 31541 := by 
    rw [A_def, pi_approx_eq, floor_approx_10040_pi_eq_31541]
  exact A_floor

end right_triangle_area_floor_l585_585532


namespace points_on_same_line_a_eq_one_fourth_l585_585458

theorem points_on_same_line_a_eq_one_fourth
  (a : ℝ)
  (h : collinear { 
      (1 : ℝ, -3 : ℝ),
      (4 * a + 1, 5 : ℝ),
      (-3 * a + 2, -1 : ℝ) }
  ) : 
  a = 1 / 4 :=
sorry

end points_on_same_line_a_eq_one_fourth_l585_585458


namespace find_smallest_value_l585_585227

theorem find_smallest_value (y : ℕ) (h : y = 9) : 
  min (min (div 5 y) (min (div 5 (y+2)) (min (div 5 (y-2)) (div y 5)))) (div (y+2) 5) = (div 5 (y+2)) := 
by
  sorry

end find_smallest_value_l585_585227


namespace monroe_collection_legs_l585_585752

theorem monroe_collection_legs : 
  let ants := 12 
  let spiders := 8 
  let beetles := 15 
  let centipedes := 5 
  let legs_ants := 6 
  let legs_spiders := 8 
  let legs_beetles := 6 
  let legs_centipedes := 100
  (ants * legs_ants + spiders * legs_spiders + beetles * legs_beetles + centipedes * legs_centipedes = 726) := 
by 
  sorry

end monroe_collection_legs_l585_585752


namespace class1_qualified_l585_585525

variables (Tardiness : ℕ → ℕ) -- Tardiness function mapping days to number of tardy students

def classQualified (mean variance median mode : ℕ) : Prop :=
  (mean = 2 ∧ variance = 2) ∨
  (mean = 3 ∧ median = 3) ∨
  (mean = 2 ∧ variance > 0) ∨
  (median = 2 ∧ mode = 2)

def eligible (Tardiness : ℕ → ℕ) : Prop :=
  ∀ i, i < 5 → Tardiness i ≤ 5

theorem class1_qualified : 
  (∀ Tardiness, (∃ mean variance median mode,
    classQualified mean variance median mode 
    ∧ mean = 2 ∧ variance = 2 
    ∧ eligible Tardiness)) → 
  (∀ Tardiness, eligible Tardiness) :=
by
  sorry

end class1_qualified_l585_585525


namespace totalPeoplePresent_l585_585913

-- Defining the constants based on the problem conditions
def associateProfessors := 2
def assistantProfessors := 7

def totalPencils := 11
def totalCharts := 16

-- The main proof statement
theorem totalPeoplePresent :
  (∃ (A B : ℕ), (2 * A + B = totalPencils) ∧ (A + 2 * B = totalCharts)) →
  (associateProfessors + assistantProfessors = 9) :=
  by
  sorry

end totalPeoplePresent_l585_585913


namespace angle_between_skew_lines_PC_and_AB_l585_585709

-- Given points P, A, B, C and the following conditions:
variables (P A B C : Type*)
variables [metric_space P] [metric_space A] [metric_space B] [metric_space C]
variables (P₁ : P) (A₁ : A) (B₁ : B) (C₁ : C)

-- Conditions:
variables (h1 : dist P₁ A₁ = dist A₁ B₁)
variables (h2 : dist A₁ B₁ = dist B₁ C₁)
variables (h3 : dist P₁ A₁ = dist A₁ C₁)
variables (h4 : ∠ P₁ A₁ B₁ = π/2)
variables (h5 : ∠ A₁ C₁ B₁ = π/2)

-- Objective:
def angle_between_PC_AB : ℝ :=
  45

theorem angle_between_skew_lines_PC_and_AB :
  ∃ θ : ℝ, θ = angle_between_PC_AB ∧ θ = 45 := 
by
  sorry

end angle_between_skew_lines_PC_and_AB_l585_585709


namespace intervals_of_monotonicity_l585_585384

open Real

noncomputable def f (a x : ℝ) : ℝ := a * x^2 - log x + 1

theorem intervals_of_monotonicity (a : ℝ) :
  (∀ x > 0, a ≤ 0 → deriv (λ x, f a x) x < 0) ∧
  (∀ x > 0, 0 < a → 
    (∀ x ∈ Ioi (sqrt (1 / (2 * a))), deriv (λ x, f a x) x > 0) ∧ 
    (∀ x ∈ Iio (sqrt (1 / (2 * a))), deriv (λ x, f a x) x < 0)) := 
sorry

end intervals_of_monotonicity_l585_585384


namespace cubic_sum_l585_585557

-- Define the variables x, y, z as real numbers
variables (x y z : ℝ)

-- Define the conditions
def conditions : Prop := x + y + z = 2 ∧ xy + yz + zx = -6 ∧ xyz = -6

-- Define the Lean 4 statement for the proof problem
theorem cubic_sum (h : conditions x y z) : x^3 + y^3 + z^3 = 25 :=
by
  -- Skipping the proof
  sorry

end cubic_sum_l585_585557


namespace interval_of_monotonicity_range_of_a_l585_585643

def f (x a : ℝ) := x * Real.log x - a * x^2 + (2 * a - 1) * x

-- Statement 1: prove interval of monotonicity when a = 1/2
theorem interval_of_monotonicity (x : ℝ) (h : 0 < x) : 
  (f x (1/2)).diff ≤ 0 := sorry

-- Statement 2: prove the range of a satisfying the inequality condition for f
theorem range_of_a (a : ℝ) :
  (∀ x, 1 ≤ x → f x a ≤ a - 1) → 1 ≤ a := sorry

end interval_of_monotonicity_range_of_a_l585_585643


namespace smallest_four_digit_multiple_of_18_l585_585976

theorem smallest_four_digit_multiple_of_18 : 
  ∃ (n : ℕ), 999 < n ∧ n < 10000 ∧ 18 ∣ n ∧ ∀ m : ℕ, 999 < m ∧ m < 10000 ∧ 18 ∣ m → n ≤ m ∧ n = 1008 := 
sorry

end smallest_four_digit_multiple_of_18_l585_585976


namespace A_share_calculation_l585_585148

-- Variables for investments
variables (x : ℝ)  -- The amount A invested initially

-- Conditions
def A_investment := x * 12
def B_investment := 2 * x * 6
def C_investment := 3 * x * 4

-- Total annual gain
def total_annual_gain := 19200

-- A's share of the annual gain
def A_share := (A_investment x) / (A_investment x + B_investment x + C_investment x) * total_annual_gain

theorem A_share_calculation : A_share x = 6400 := by
  sorry

end A_share_calculation_l585_585148


namespace smallest_four_digit_multiple_of_18_l585_585963

theorem smallest_four_digit_multiple_of_18 : 
  ∃ (n : ℤ), 1000 ≤ n ∧ n < 10000 ∧ 18 ∣ n ∧ 
  ∀ (m : ℤ), (1000 ≤ m ∧ m < 10000 ∧ 18 ∣ m) → n ≤ m :=
sorry

end smallest_four_digit_multiple_of_18_l585_585963


namespace sum_of_possible_m_values_l585_585122

theorem sum_of_possible_m_values : 
  sum (setOf m in {m | 4 < m ∧ m < 18}) = 132 :=
by
  sorry

end sum_of_possible_m_values_l585_585122


namespace quadrilateral_inequality_l585_585633

variables {A B C D : Point}
variables {O P Q R S : Point}

-- Define the conditions
def quadrilateral_has_circumscribed_circle (ABCD : quadrilateral) : Prop := 
  circumscribed_circle ABCD

def diagonals_intersect_at (ABCD : quadrilateral) (O : Point) : Prop :=
  let (AC, BD) := diagonals ABCD
  AC ∩ BD = {O}

def foot_perpendicular (O A B : Point) : Point := P -- Assuming definition available
def feet_perpendiculars (O A B C D : Point) [circumscribed_circle ABCD] : 
  (Point, Point, Point, Point) :=
  (foot_perpendicular O A B, foot_perpendicular O B C,
   foot_perpendicular O C D, foot_perpendicular O D A)

-- Main theorem statement
theorem quadrilateral_inequality
  {A B C D O : Point} 
  (h1 : quadrilateral_has_circumscribed_circle (quadrilateral A B C D))
  (h2 : diagonals_intersect_at (quadrilateral A B C D) O) 
  (h3 : feet_perpendiculars O A B C D = (P, Q, R, S)) :
  (BD_length (A, B, C, D)) ≥ length (S, P) + length (Q, R) :=
sorry

end quadrilateral_inequality_l585_585633


namespace tournament_start_count_l585_585689

theorem tournament_start_count (x : ℝ) (h1 : (0.1 * x = 30)) : x = 300 :=
by
  sorry

end tournament_start_count_l585_585689


namespace find_first_term_l585_585549

-- Definitions and conditions
def is_geometric_series (a : ℝ) (r : ℝ) (S : ℝ) : Prop :=
  S = a / (1 - r)

-- Given conditions
def common_ratio : ℝ := 1 / 3
def sum_of_series : ℝ := 18

-- Main theorem statement: Prove the first term of the series
theorem find_first_term (a : ℝ) : is_geometric_series a common_ratio sum_of_series → a = 12 := by
  sorry

end find_first_term_l585_585549


namespace smallest_four_digit_multiple_of_18_l585_585957

theorem smallest_four_digit_multiple_of_18 : ∃ n: ℕ, (1000 ≤ n ∧ n < 10000 ∧ 18 ∣ n) ∧ ∀ m: ℕ, (1000 ≤ m ∧ m < n ∧ 18 ∣ m) → false :=
begin
  use 1008,
  split,
  { split,
    { -- prove 1000 ≤ 1008
      linarith,
    },
    { split,
      { -- prove 1008 < 10000
        linarith,
      },
      { -- prove 18 ∣ 1008
        norm_num,
      }
    }
  },
  { -- prove there is no four-digit multiple of 18 less than 1008
    intros m,
    intro h,
    cases h with h1 h2,
    cases h2 with h3 h4,
    linarith,
  }
end

end smallest_four_digit_multiple_of_18_l585_585957


namespace probability_product_pos_eq_half_l585_585045

-- Defines the interval for sampling
def interval := set.Icc (-20 : ℝ) 20

-- Defines a uniform probability distribution over the interval
def uniform_dist (a b : ℝ) : ProbabilityMassFunction ℝ :=
  ProbabilityMassFunction.uniform (set.Icc a b)

-- Defines the event that the product is greater than zero
def product_pos (x y : ℝ) : Prop := x * y > 0

-- The problem statement
theorem probability_product_pos_eq_half : 
  Probability (uniform_dist (-20) 20) *
  Probability (uniform_dist (-20) 20) {x | ∃ (y : ℝ), y ∈ interval ∧ product_pos x y} = 1 / 2 :=
begin
  sorry
end

end probability_product_pos_eq_half_l585_585045


namespace heather_oranges_l585_585288

theorem heather_oranges (heather_initial_oranges russell_takes : ℕ) (h1 : heather_initial_oranges = 60) (h2 : russell_takes = 35) :
  heather_initial_oranges - russell_takes = 25 :=
by
  rw [h1, h2]
  exact rfl

end heather_oranges_l585_585288


namespace difference_of_squares_l585_585031

theorem difference_of_squares {a b : ℝ} (h1 : a + b = 75) (h2 : a - b = 15) : a^2 - b^2 = 1125 :=
by
  sorry

end difference_of_squares_l585_585031


namespace max_ratio_solution_l585_585800

noncomputable def max_ratio_spheres_in_cones
  (R h r x : ℝ) -- R: radius of the base, h: height of the cone, r: radius of the first two identical spheres, x: radius of the third sphere
  (cond1 : h = R / 2) -- condition on cone height
  (cond2 : ∃ (O1 O2 O3 : ℝ), O1 = r ∧ O2 = r ∧ O3 = x ∧ (O1 + O2 + O3 ≤ R)) -- condition on spheres
  (cond3 : 3r = R) -- derived geometric relationship
  : ℝ := 
  let t : ℝ := x / r 
  in (7 - real.sqrt 22) / 3 -- maximum ratio
  
theorem max_ratio_solution (R r x : ℝ) 
  (h : ℝ)
  (cond1 : h = R / 2)
  (cond2 : ∃(O1 O2 O3 : ℝ), O1 = r ∧ O2 = r ∧ O3 = x ∧ O1 + O2 + O3 ≤ R)
  (cond3 : 3r = R)
  : max_ratio_spheres_in_cones R h r x cond1 cond2 cond3 = (7 - real.sqrt 22) / 3 := sorry

end max_ratio_solution_l585_585800


namespace angle_equality_in_convex_quadrilateral_l585_585556

structure ConvexQuadrilateral (V : Type) [EuclideanSpace V] :=
(A B C D E F P O : V)
(AB_CD_intersect : ∃ E, LineThrough A B ∩ LineThrough C D = {E})
(BC_AD_intersect : ∃ F, LineThrough B C ∩ LineThrough A D = {F})
(diagonals_intersect : ∃ P, LineThrough A C ∩ LineThrough B D = {P})
(perpendicular_from_P_to_EF : ∃ O, PerpendicularLineFromPoint P (LineThrough E F) = {O})

theorem angle_equality_in_convex_quadrilateral
  {V : Type} [EuclideanSpace V] (quad : ConvexQuadrilateral V) :
  ∠ quad.B quad.O quad.C = ∠ quad.A quad.O quad.D := 
sorry

end angle_equality_in_convex_quadrilateral_l585_585556


namespace isosceles_triangle_area_l585_585909

theorem isosceles_triangle_area (s b : ℝ) (h₁ : s + b = 20) (h₂ : b^2 + 10^2 = s^2) : 
  1/2 * 2 * b * 10 = 75 :=
by sorry

end isosceles_triangle_area_l585_585909


namespace value_of_x_when_y_is_six_l585_585075

theorem value_of_x_when_y_is_six 
  (k : ℝ) -- The constant of variation
  (h1 : ∀ y : ℝ, x = k / y^2) -- The inverse relationship
  (h2 : y = 2)
  (h3 : x = 1)
  : x = 1 / 9 :=
by
  sorry

end value_of_x_when_y_is_six_l585_585075


namespace candies_eaten_l585_585175

theorem candies_eaten (A B D : ℕ) 
                      (h1 : 4 * B = 3 * A) 
                      (h2 : 7 * A = 6 * D) 
                      (h3 : A + B + D = 70) :
  A = 24 ∧ B = 18 ∧ D = 28 := 
by
  sorry

end candies_eaten_l585_585175


namespace triangle_area_calculation_l585_585314

open Nat

-- Defining the points and conditions as assumptions
variable {A B C D E F : Type}
variable [Triangle ABC]
variable [Midpoint D B C]
variable [OnLineSegment E A C]
variable [OnLineSegment F A D]
variable (ratio_AE_EC : ℕ) (ratio_AF_FD : ℕ)
variable (area_DEF : ℕ)

-- Main statement
theorem triangle_area_calculation :
  ratio_AE_EC = 3 ∧ ratio_AF_FD = 2 ∧ area_DEF = 10 → triangle_area ABC = 80 :=
by
  -- Adding assumptions as definitions
  sorry

end triangle_area_calculation_l585_585314


namespace area_of_inscribed_hexagon_l585_585833

theorem area_of_inscribed_hexagon (r : ℝ) (h : π * r^2 = 100 * π) : 
  ∃ (A : ℝ), A = 150 * real.sqrt 3 :=
by 
  -- Definitions of necessary geometric entities and properties would be here
  -- Proof would be provided here
  sorry

end area_of_inscribed_hexagon_l585_585833


namespace train_cross_pole_time_l585_585901

-- Definitions based on the conditions
def train_speed_kmh := 54
def train_length_m := 105
def train_speed_ms := (train_speed_kmh * 1000) / 3600
def expected_time := train_length_m / train_speed_ms

-- Theorem statement, encapsulating the problem
theorem train_cross_pole_time : expected_time = 7 := by
  sorry

end train_cross_pole_time_l585_585901


namespace highest_power_of_2_dividing_2n_factorial_l585_585220

def highest_power_of_2_dividing_factorial (n : ℕ) : ℕ :=
  ∑ k in finset.range (nat.bit_length (2 * n)), (2 * n) / 2 ^ k

theorem highest_power_of_2_dividing_2n_factorial (n : ℕ) : 
  (2 : ℕ) ^ (highest_power_of_2_dividing_factorial n) = 2 ^ (2 * n - 1) :=
by
  sorry

end highest_power_of_2_dividing_2n_factorial_l585_585220


namespace transfer_students_increase_average_l585_585469

theorem transfer_students_increase_average
    (avgA : ℚ) (numA : ℕ) (avgB : ℚ) (numB : ℕ)
    (kalinina_grade : ℚ) (sidorova_grade : ℚ)
    (avgA_init : avgA = 44.2) (numA_init : numA = 10)
    (avgB_init : avgB = 38.8) (numB_init : numB = 10)
    (kalinina_init : kalinina_grade = 41)
    (sidorova_init : sidorova_grade = 44) : 
    let new_avg_B_k := (avgB * numB + kalinina_grade) / (numB + 1) in
    let new_avg_A_s := (avgA * numA - sidorova_grade) / (numA - 1) in
    let new_avg_A_both := (avgA * numA - kalinina_grade - sidorova_grade) / (numA - 2) in
    let new_avg_B_both := (avgB * numB + kalinina_grade + sidorova_grade) / (numB + 2) in
    (new_avg_B_k <= avgB) ∧ (new_avg_A_s <= avgA) ∧ (new_avg_A_both > avgA) ∧ (new_avg_B_both > avgB) :=
by
  sorry

end transfer_students_increase_average_l585_585469


namespace polar_coordinate_standard_equivalent_l585_585696

theorem polar_coordinate_standard_equivalent (r θ : ℝ) (h_r : r = -3) (h_θ : θ = π / 6) :
  ∃ r' θ', r' = 3 ∧ θ' = 7 * π / 6 ∧ r' > 0 ∧ 0 ≤ θ' ∧ θ' < 2 * π :=
by
  have h_r' : r' = 3, from sorry
  have h_θ' : θ' = 7 * π / 6, from sorry
  have hr_pos : r' > 0, from sorry
  have hθ_range : 0 ≤ θ' ∧ θ' < 2 * π, from sorry
  exact ⟨3, 7 * π / 6, h_r', h_θ', hr_pos, hθ_range⟩

end polar_coordinate_standard_equivalent_l585_585696


namespace volume_of_box_l585_585881

theorem volume_of_box (L W S : ℕ) (hL : L = 48) (hW : W = 36) (hS : S = 8) : 
  let new_length := L - 2 * S,
      new_width := W - 2 * S,
      height := S
  in new_length * new_width * height = 5120 := by
sorry

end volume_of_box_l585_585881


namespace sum_arithmetic_sequence_l585_585449

variable {α : Type*} [LinearOrderedField α]

def sum_first_n (a d : α) (n : ℕ) : α := 
  n * (2 * a + (n - 1) * d) / 2

theorem sum_arithmetic_sequence
    (a d : α) 
    (S_5 S_10 S_15 : α)
    (h1 : S_5 = 5 * (2 * a + 4 * d) / 2)
    (h2 : S_10 = 10 * (2 * a + 9 * d) / 2)
    (h3 : S_5 = 10)
    (h4 : S_10 = 50) : 
  S_15 = 15 * (2 * a + 14 * d) / 2 := 
sorry

end sum_arithmetic_sequence_l585_585449


namespace smallest_four_digit_multiple_of_18_l585_585985

theorem smallest_four_digit_multiple_of_18 : ∃ n : ℕ, (1000 ≤ n) ∧ (n < 10000) ∧ (n % 18 = 0) ∧ (∀ m : ℕ, (1000 ≤ m) ∧ (m < 10000) ∧ (m % 18 = 0) → n ≤ m) ∧ n = 1008 :=
by
  sorry

end smallest_four_digit_multiple_of_18_l585_585985


namespace coeff_x2_in_expansion_l585_585331

open BigOperators

-- Given condition
def exp := (√x - 2)^5

-- Main goal: prove the coefficient of x^2 in the expansion of exp equals -10 
theorem coeff_x2_in_expansion : coefficient x^2 (expansion_exp exp) = -10 := by
  sorry

end coeff_x2_in_expansion_l585_585331


namespace number_of_teachers_l585_585721

-- Definitions from the problem conditions
def num_students : Nat := 1500
def classes_per_student : Nat := 6
def classes_per_teacher : Nat := 5
def students_per_class : Nat := 25

-- The proof problem statement
theorem number_of_teachers : 
  (num_students * classes_per_student / students_per_class) / classes_per_teacher = 72 := by
  sorry

end number_of_teachers_l585_585721


namespace find_multiple_l585_585587

theorem find_multiple :
    ∃ m : ℝ, 
      (let x := 25/3 in
       15 + m * x = 6 * x - 10) ∧ m = 3 :=
begin
  sorry
end

end find_multiple_l585_585587


namespace integral_cos_squared_l585_585916

theorem integral_cos_squared:
  ∫ x in 0..π, (cos x)^2 = π / 2 :=
by
  sorry

end integral_cos_squared_l585_585916


namespace log_computation_l585_585726

theorem log_computation (a b : ℝ) (h1 : a = Real.log 8) (h2 : b = Real.log 27) : 2 ^ (a / b) + 3 ^ (b / a) = 5 :=
by
  -- proofs to be filled
  sorry

end log_computation_l585_585726


namespace time_to_complete_work_together_l585_585074

variables 
  (p_efficiency q_efficiency r_efficiency : ℝ)
  (work : ℝ)

-- Given conditions
def efficiency_of_q := 1 -- q's efficiency is 1 unit of work per day
def p_efficiency := 1.4 * efficiency_of_q
def r_efficiency := 0.7 * efficiency_of_q
def work := p_efficiency * 24 -- total work done by p in 24 days

-- Prove that when p, q, and r work together, it will take approximately 10.84 days
theorem time_to_complete_work_together 
  (combined_efficiency := p_efficiency + q_efficiency + r_efficiency) :
  (work / combined_efficiency) ≈ 10.84 := by
  sorry

end time_to_complete_work_together_l585_585074


namespace hexagon_area_of_circle_l585_585843

noncomputable def radius (area : ℝ) : ℝ :=
  Real.sqrt (area / Real.pi)

noncomputable def area_equilateral_triangle (s : ℝ) : ℝ :=
  (s * s * Real.sqrt 3) / 4

theorem hexagon_area_of_circle {r : ℝ} (h : π * r^2 = 100 * π) :
  6 * area_equilateral_triangle r = 150 * Real.sqrt 3 :=
by
  sorry

end hexagon_area_of_circle_l585_585843


namespace general_term_sum_T_l585_585741

-- Geometric sequence as described in the problem
def a : ℕ → ℝ
| 0 => 0 -- a_0 does not exist but initializing it for convenience
| n+1 => (1 / 2) ^ (n + 1)

-- Sum of the first n terms
def S : ℕ → ℝ
| 0 => 0 -- sum of first 0 terms is 0
| n => (1 / 2) * ((1 - (1 / 2) ^ n) / (1 - 1 / 2))

-- Sum of the first n terms of {nS_n}
noncomputable def T (n : ℕ) : ℝ := 
∑ i in Finset.range n, (i + 1) * (1 - (1 / 2) ^ (i + 1))

-- Prove the general term is of the form (1/2)^n
theorem general_term (n : ℕ) : a (n + 1) = (1 / 2) ^ (n + 1) :=
sorry

-- Prove the sum T_n
theorem sum_T (n : ℕ) : 
  T n = (n * (n + 1) / 2) + (1 / 2^{n-1}) + (n / 2^n) - 2 :=
sorry

end general_term_sum_T_l585_585741


namespace sum_of_coefficients_of_terms_without_x_l585_585030

-- Define the binomial coefficient function.
def binom (n k : ℕ) : ℕ := Nat.choose n k

-- Define the expression.
def expr (x y : ℝ) : ℝ := (x + 1 / (3 * x) - 4 * y)^7

-- The proof statement.
theorem sum_of_coefficients_of_terms_without_x (x y : ℝ) :
  let term_without_x := -binom 7 3 * binom 4 3 * (4:ℝ)^3 - (4:ℝ)^7
  (∑ coeff in (expr x y).coefficients {z : ℝ | z ≠ x}, coeff) = term_without_x :=
sorry

end sum_of_coefficients_of_terms_without_x_l585_585030


namespace midpoint_equation_of_line_vector_equation_of_line_l585_585105

variable {A B P : ℝ}

-- Part I
-- Midpoint of AB is P
def isMidpoint (A B P : ℝ) :=
  P = ((A + B) / 2)

-- Equation of line l part I
theorem midpoint_equation_of_line (x y : ℝ) (P : ℝ) :
  isMidpoint x y (-3, 1) → P = (-3, 1) → (l : ℝ) := sorry

-- Part II
-- AP = 2PB
def equalVectors (A B P : ℝ) :=
  A - P = 2 * (B - P)

-- Equation of line l part II
theorem vector_equation_of_line (x y : ℝ) (P : ℝ) :
  equalVectors x y (-3, 1) → P = (-3, 1) → (l : ℝ) := sorry

end midpoint_equation_of_line_vector_equation_of_line_l585_585105


namespace compare_values_l585_585234

-- Definitions of the given values
def a : ℝ := Real.sin (80 * Real.pi / 180)  -- sin 80 degrees
def b : ℝ := 2                             -- (1/2)^(-1)
def c : ℝ := -Real.log 3 / Real.log 2      -- -log_2 3

-- Statement to prove
theorem compare_values : b > a ∧ a > c := by
  sorry

end compare_values_l585_585234


namespace solve_for_z_l585_585946

theorem solve_for_z (z : ℤ) (h : sqrt (9 + 3 * z) = 12) : z = 45 := by
  sorry

end solve_for_z_l585_585946


namespace units_digit_x4_invx4_l585_585299

theorem units_digit_x4_invx4 (x : ℝ) (h : x^2 - 12 * x + 1 = 0) : 
  (x^4 + (1 / x)^4) % 10 = 2 := 
by
  sorry

end units_digit_x4_invx4_l585_585299


namespace domain_of_function_l585_585930

theorem domain_of_function : 
  (∀ x : ℝ, (x + 1 > 0) ∧ (1 - log 10 (x + 1) ≥ 0) ↔ (-1 < x ∧ x ≤ 9)) := 
by
  sorry

end domain_of_function_l585_585930
