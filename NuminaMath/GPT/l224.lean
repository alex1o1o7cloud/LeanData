import Mathlib

namespace balloon_permutations_l224_224947

theorem balloon_permutations : 
  let total_letters := 7
  let repetitions_L := 2
  let repetitions_O := 2
  (Nat.factorial total_letters) / ((Nat.factorial repetitions_L) * (Nat.factorial repetitions_O)) = 1260 := 
by
  sorry

end balloon_permutations_l224_224947


namespace solve_inequality_l224_224304

noncomputable def inequality_holds (x : ℝ) : Prop :=
  |4 * x^2 - 32 / x| + |x^2 + 5 / (x^2 - 6)| ≤ |3 * x^2 - 5 / (x^2 - 6) - 32 / x|

theorem solve_inequality (x : ℝ) : 
  inequality_holds x ↔ 
  (-√6 < x ∧ x ≤ -√5) ∨ (-1 ≤ x ∧ x < 0) ∨ (1 ≤ x ∧ x ≤ 2) ∨ (√5 ≤ x ∧ x < √6) :=
sorry

end solve_inequality_l224_224304


namespace sum_of_factors_l224_224377

theorem sum_of_factors (W F c : ℕ) (hW_gt_20: W > 20) (hF_gt_20: F > 20) (product_eq : W * F = 770) (sum_eq : W + F = c) :
  c = 57 :=
by sorry

end sum_of_factors_l224_224377


namespace min_value_is_4_l224_224627

noncomputable def min_value (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : 1 / x + 1 / y = 1) : ℝ :=
  infi (λ (z : ℝ), ∃ (x y : ℝ), z = 1 / (x - 1) + 4 / (y - 1) ∧ x > 0 ∧ y > 0 ∧ 1 / x + 1 / y = 1)

theorem min_value_is_4 (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : 1 / x + 1 / y = 1) :
  min_value x y h1 h2 h3 = 4 :=
by
  sorry

end min_value_is_4_l224_224627


namespace vector_magnitude_solution_l224_224576

noncomputable def angle_between_vectors_is_pi_over_3 (a b : ℝ) : Prop :=
  ∃ (θ : ℝ), θ = π / 3 ∧ ‖a‖ = 4 ∧ ‖b‖ = 1 ∧
    ‖a - 4 * b‖ = 4

theorem vector_magnitude_solution (a b : ℝ) (h : angle_between_vectors_is_pi_over_3 a b) :
    |a - 4 * b| = 4 :=
sorry

end vector_magnitude_solution_l224_224576


namespace find_e_l224_224796

variables (j p t b a : ℝ) (e : ℝ)

theorem find_e
  (h1 : j = 0.75 * p)
  (h2 : j = 0.80 * t)
  (h3 : t = p - (e / 100) * p)
  (h4 : b = 1.40 * j)
  (h5 : a = 0.85 * b)
  (h6 : e = 2 * ((p - a) / p) * 100) :
  e = 21.5 := by
  sorry

end find_e_l224_224796


namespace Zlatoust_to_Miass_distance_l224_224355

theorem Zlatoust_to_Miass_distance
  (x g k m : ℝ)
  (H1 : (x + 18) / k = (x - 18) / m)
  (H2 : (x + 25) / k = (x - 25) / g)
  (H3 : (x + 8) / m = (x - 8) / g) :
  x = 60 :=
sorry

end Zlatoust_to_Miass_distance_l224_224355


namespace factorize_expression_l224_224996

theorem factorize_expression (x y : ℝ) : x^2 - 1 + 2 * x * y + y^2 = (x + y + 1) * (x + y - 1) :=
by sorry

end factorize_expression_l224_224996


namespace collinear_vectors_m_n_sum_l224_224521

theorem collinear_vectors_m_n_sum (m n : ℕ)
  (h1 : (2, 3, m) = (2 * n, 6, 8)) :
  m + n = 6 :=
sorry

end collinear_vectors_m_n_sum_l224_224521


namespace log3_20_approx_l224_224520

theorem log3_20_approx :
  (log 3 20) ≈ (33 / 12) ∧
  ∀ (log10_2 log10_5 : ℝ), log10_2 ≈ 0.301 ∧ log10_5 ≈ 0.699 -> 
  log 3 20 ≈ (33 / 12) :=
by
  sorry

end log3_20_approx_l224_224520


namespace greatest_possible_remainder_l224_224643

/-- Lucas distributed y stickers evenly among 12 children and we want to
    find the greatest possible number of stickers that could have been left over.
    -/
theorem greatest_possible_remainder (y : ℕ) :
  ∃ r : ℕ, r < 12 ∧ ∀ r' : ℕ, r' < 12 → (∃ q : ℕ, y = 12 * q + r') → r' ≤ r :=
begin
  sorry
end

end greatest_possible_remainder_l224_224643


namespace solutions_of_equations_l224_224456

noncomputable def is_solution (x y z : ℂ) : Prop :=
x^2 = y + z ∧ y^2 = x + z ∧ z^2 = x + y

theorem solutions_of_equations :
  { (x, y, z) : ℂ × ℂ × ℂ | is_solution x y z } =
  { (0, 0, 0), (2, 2, 2), 
    (-1+complex.i, -complex.i, -complex.i), 
    (-complex.i, -1+complex.i, -complex.i), 
    (-complex.i, -complex.i, -1+complex.i), 
    (-1-complex.i, complex.i, complex.i), 
    (complex.i, -1-complex.i, complex.i), 
    (complex.i, complex.i, -1-complex.i) } :=
by 
  sorry

end solutions_of_equations_l224_224456


namespace balloon_permutations_l224_224958

theorem balloon_permutations : (∀ (arrangements : ℕ),
  arrangements = nat.factorial 7 / (nat.factorial 2 * nat.factorial 2) → arrangements = 1260) :=
begin
  intros arrangements h,
  rw [nat.factorial_succ, nat.factorial],
  sorry,
end

end balloon_permutations_l224_224958


namespace range_of_a_l224_224139

theorem range_of_a (a : ℝ) (h : ∀ x ∈ set.Icc 1 4, |x + 4/x - a| + a ≤ 5) : a ≤ 9/2 :=
sorry

end range_of_a_l224_224139


namespace log_eq_solve_l224_224297

theorem log_eq_solve (x : Real) (hx1 : log 9 x + log 3 (x ^ 3) = 7) : x = 9 :=
by
  sorry

end log_eq_solve_l224_224297


namespace no_distinct_ordered_pairs_l224_224121

theorem no_distinct_ordered_pairs (x y : ℕ) (h₁ : 0 < x) (h₂ : 0 < y) :
  (x^2 * y^2)^2 - 14 * x^2 * y^2 + 49 ≠ 0 :=
by
  sorry

end no_distinct_ordered_pairs_l224_224121


namespace triangle_area_from_line_and_axes_l224_224082

theorem triangle_area_from_line_and_axes : 
  let line := (λ x y : ℝ, x + y - 2 = 0) in
  let point_on_x_axis := (2, 0) in
  let point_on_y_axis := (0, 2) in
  let origin := (0, 0) in
  let base := 2 in
  let height := 2 in
  1 / 2 * base * height = 2 := 
by
  sorry

end triangle_area_from_line_and_axes_l224_224082


namespace missy_tv_watching_time_l224_224267

def reality_show_count : Nat := 5
def reality_show_duration : Nat := 28
def cartoon_duration : Nat := 10

theorem missy_tv_watching_time :
  reality_show_count * reality_show_duration + cartoon_duration = 150 := by
  sorry

end missy_tv_watching_time_l224_224267


namespace find_e_l224_224797

variables (j p t b a : ℝ) (e : ℝ)

theorem find_e
  (h1 : j = 0.75 * p)
  (h2 : j = 0.80 * t)
  (h3 : t = p - (e / 100) * p)
  (h4 : b = 1.40 * j)
  (h5 : a = 0.85 * b)
  (h6 : e = 2 * ((p - a) / p) * 100) :
  e = 21.5 := by
  sorry

end find_e_l224_224797


namespace fill_tank_time_l224_224378

theorem fill_tank_time :
  ∀ (rate_fill rate_empty : ℝ), 
    rate_fill = 1 / 25 → 
    rate_empty = 1 / 50 → 
    (1/2) / (rate_fill - rate_empty) = 25 :=
by
  intros rate_fill rate_empty h_fill h_empty
  sorry

end fill_tank_time_l224_224378


namespace balloon_arrangement_count_l224_224967

-- Definitions of letter frequencies for the word BALLOON
def letter_frequencies : List (Char × Nat) := [('B', 1), ('A', 1), ('L', 2), ('O', 2), ('N', 1)]

-- Total number of letters
def total_letters := 7

-- The formula for the number of unique arrangements of the letters
noncomputable def arrangements :=
  (Nat.factorial total_letters) / 
  (Nat.factorial 1 * Nat.factorial 1 * Nat.factorial 2 * Nat.factorial 2 * Nat.factorial 1)

-- The theorem to prove the number of ways to arrange the letters in "BALLOON"
theorem balloon_arrangement_count : arrangements = 1260 :=
  sorry

end balloon_arrangement_count_l224_224967


namespace rectangle_length_fraction_of_radius_l224_224330

theorem rectangle_length_fraction_of_radius
  (r : ℝ) (b : ℝ) (A_square : ℝ)
  (h1 : A_square = 2025)
  (h2 : b = 10)
  (A_rectangle : ℝ)
  (h3 : A_rectangle = 180) :
  (18 : ℝ) / r = 2 / 5 :=
by
  have r : r = real.sqrt A_square, by sorry
  have r : r = 45, by sorry
  have l : (18 : ℝ), by sorry
  exact sorry

end rectangle_length_fraction_of_radius_l224_224330


namespace chess_tournament_l224_224191

theorem chess_tournament (n m : ℕ) (h1 : 10 * n = 10 * n)
                          (h2 : 4.5 * m = 4.5 * m)
                          (h3 : n + 10 * n = 11 * n)
                          (h4 : m + 4.5 * m = 5.5 * m)
                          (h5 : 5.5 * m = (11 * n * (11 * n - 1)) / 2) :
  n = 1 ∧ m = 10 :=
by
  sorry

end chess_tournament_l224_224191


namespace triangle_area_l224_224058

-- Defining the rectangle dimensions
def length : ℝ := 35
def width : ℝ := 48

-- Defining the area of the right triangle formed by the diagonal of the rectangle
theorem triangle_area : (1 / 2) * length * width = 840 := by
  sorry

end triangle_area_l224_224058


namespace part1_odd_function_part2_monotonic_increasing_part3_max_min_values_l224_224538

def f (x : ℝ) : ℝ := x - 4 / x

theorem part1_odd_function :
  ∀ (x : ℝ), f (-x) = -f (x) := sorry

theorem part2_monotonic_increasing :
  ∀ (x1 x2 : ℝ), (0 < x1) → (x1 < x2) → f (x1) < f (x2) := sorry

theorem part3_max_min_values :
  f (4) = 3 ∧ f (1) = -3 := sorry

end part1_odd_function_part2_monotonic_increasing_part3_max_min_values_l224_224538


namespace smallest_percentage_increase_l224_224855

theorem smallest_percentage_increase :
  let n2005 := 75
  let n2006 := 85
  let n2007 := 88
  let n2008 := 94
  let n2009 := 96
  let n2010 := 102
  let perc_increase (a b : ℕ) := ((b - a) : ℚ) / a * 100
  perc_increase n2008 n2009 < perc_increase n2006 n2007 ∧
  perc_increase n2008 n2009 < perc_increase n2007 n2008 ∧
  perc_increase n2008 n2009 < perc_increase n2009 n2010 ∧
  perc_increase n2008 n2009 < perc_increase n2005 n2006
:= sorry

end smallest_percentage_increase_l224_224855


namespace unique_solution_3_pow_m_eq_2_pow_k_add_7_pow_n_and_m_pow_k_eq_series_l224_224474

theorem unique_solution_3_pow_m_eq_2_pow_k_add_7_pow_n_and_m_pow_k_eq_series (m n k l : ℕ) (hm: m > 0) (hn: n > 0) (hk: k > 0) (hl: l > 0) :
  (3 ^ m = 2 ^ k + 7 ^ n ∧ m ^ k = (Finset.range (l + 1)).sum (λ i, k ^ i)) ↔ (m, n, k, l) = (2, 1, 1, 1) := 
begin
  sorry
end

end unique_solution_3_pow_m_eq_2_pow_k_add_7_pow_n_and_m_pow_k_eq_series_l224_224474


namespace neighbor_distances_l224_224105

theorem neighbor_distances (A B C D E F G H : Point)
  (dist_A_D : distance A D = 28)
  (dist_A_F : distance A F = 43)
  (dist_D_G : distance D G = 22)
  (dist_F_C : distance F C = 25)
  (dist_C_H : distance C H = 38)
  (dist_H_E : distance H E = 24)
  (dist_E_B : distance E B = 27) :
  distance A B = 5 ∧
  distance B C = 13 ∧
  distance C D = 10 ∧
  distance D E = 4 ∧
  distance E F = 11 ∧
  distance F G = 7 ∧
  distance G H = 6 :=
by
  sorry

end neighbor_distances_l224_224105


namespace parabola_directrix_l224_224320

theorem parabola_directrix (x y : ℝ) : (y^2 = 2 * x) → (x = -(1 / 2)) := by
  sorry

end parabola_directrix_l224_224320


namespace derivative_at_2_l224_224162

def f (x : ℝ) : ℝ := x^3 + 2

theorem derivative_at_2 : deriv f 2 = 12 := by
  sorry

end derivative_at_2_l224_224162


namespace a_11_is_12_l224_224156

-- Definitions and conditions
def arithmetic_sequence (a : ℕ → ℝ) := ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d
def a_2 (a : ℕ → ℝ) := a 2 = 3
def a_6 (a : ℕ → ℝ) := a 6 = 7

-- The statement to prove
theorem a_11_is_12 (a : ℕ → ℝ) (h_arith : arithmetic_sequence a) (h_a2 : a_2 a) (h_a6 : a_6 a) : a 11 = 12 :=
  sorry

end a_11_is_12_l224_224156


namespace quadrilateral_is_square_l224_224412

-- Definitions of the conditions
structure Quadrilateral (α : Type) :=
(sides_are_equal : ∀ a b : α, a = b)
(diagonals_are_perpendicular : ∀ d1 d2 : α, d1 = d2)
(diagonals_are_equal : ∀ d1 d2 : α, d1 = d2)

-- The proof problem
theorem quadrilateral_is_square {α : Type} [Quadrilateral α] : ∀ q : Quadrilateral α, q = square := 
sorry

end quadrilateral_is_square_l224_224412


namespace find_AC_l224_224198

-- Definitions for the problem
variables (A B C M N P : Type*)
variables [EuclideanSpace ℝ] 
variables (A B C M N P : EuclideanSpace ℝ)
variables (hM : midpoint A B M)
variables (hN : midpoint B C N)
variables (hTangents : tangents_circumcircle_BMN M N = P)
variables (hParallel : line_parallel AP BC)
variables (hAP_len : distance A P = 9)
variables (hPN_len : distance P N = 15)

-- The condition we need to prove
theorem find_AC : distance A C = 20 * sqrt 2 := 
sorry

end find_AC_l224_224198


namespace repeating_decimal_as_fraction_l224_224746

theorem repeating_decimal_as_fraction :
  let x := 56 / 99
  in x = 0.56 + 0.0056 + 0.000056 + (0.00000056) : ℚ :=
by
  sorry

end repeating_decimal_as_fraction_l224_224746


namespace balloon_permutations_l224_224943

theorem balloon_permutations : 
  let balloons := "BALLOON".toList.length in
  let total_permutations := Nat.factorial balloons in
  let repetitive_L := Nat.factorial 2 in
  let repetitive_O := Nat.factorial 2 in
  (total_permutations / (repetitive_L * repetitive_O)) = 1260 := sorry

end balloon_permutations_l224_224943


namespace reach_any_natural_number_l224_224438

theorem reach_any_natural_number (n : ℕ) : ∃ (f : ℕ → ℕ), f 0 = 1 ∧ (∀ k, f (k + 1) = 3 * f k + 1 ∨ f (k + 1) = f k / 2) ∧ (∃ m, f m = n) := by
  sorry

end reach_any_natural_number_l224_224438


namespace lune_area_with_triangle_l224_224415

theorem lune_area_with_triangle (d1 d2 : ℝ) (d1_eq : d1 = 3) (d2_eq : d2 = 4) : 
  let r1 := d1 / 2,
      r2 := d2 / 2,
      area_small_semicircle := (1 / 2) * π * r1^2,
      area_large_semicircle := (1 / 2) * π * r2^2,
      equilateral_triangle_area := (sqrt (3) / 4) * d1^2 in
  (area_small_semicircle - area_large_semicircle - equilateral_triangle_area) = - (7 / 8) * π - (9 / 4) * sqrt 3 :=
by
  -- Here we rely on the given diameters to reduce areas
  -- Simplify the expression based on provided values
  sorry

end lune_area_with_triangle_l224_224415


namespace water_distribution_l224_224171

theorem water_distribution :
  ∀ (bottles: ℕ) (liters_per_bottle: ℕ) (liters_per_student: ℚ),
  bottles = 12 →
  liters_per_bottle = 3 →
  liters_per_student = 3/4 →
  (bottles * liters_per_bottle) / liters_per_student = 48 :=
by
  intros bottles liters_per_bottle liters_per_student h_bottles h_liters_per_bottle h_liters_per_student
  rw [h_bottles, h_liters_per_bottle, h_liters_per_student]
  norm_num
  exact eq.refl 48

end water_distribution_l224_224171


namespace balloon_arrangements_l224_224979

def factorial (n : Nat) : Nat :=
  if n = 0 then 1 else n * factorial (n - 1)

def arrangements (n : Nat) (ks : List Nat) :=
  factorial n / (ks.map factorial).foldr (*) 1

theorem balloon_arrangements : arrangements 7 [2, 2] = 1260 :=
by
  sorry

end balloon_arrangements_l224_224979


namespace centroid_distance_relation_l224_224237

variable (A B C G : Point)

def is_centroid (G : Point) (A B C : Point) : Prop :=
  G = (A + B + C) / 3

def GA_squared (G A : Point) : ℝ :=
  normSq (G - A)

def GB_squared (G B : Point) : ℝ :=
  normSq (G - B)

def GC_squared (G C : Point) : ℝ :=
  normSq (G - C)

def AB_squared (A B : Point) : ℝ :=
  normSq (A - B)

def AC_squared (A C : Point) : ℝ :=
  normSq (A - C)

def BC_squared (B C : Point) : ℝ :=
  normSq (B - C)

theorem centroid_distance_relation
  (hG: is_centroid G A B C)
  (h_eq: GA_squared G A + 2 * GB_squared G B + 3 * GC_squared G C = 123) :
  AB_squared A B + AC_squared A C + BC_squared B C = 246 :=
sorry

end centroid_distance_relation_l224_224237


namespace repeating_decimal_fraction_l224_224781

theorem repeating_decimal_fraction : ∀ x : ℚ, (x = 0.5656565656565656) → 100 * x = 56.5656565656565656 → 100 * x - x = 56.5656565656565656 - 0.5656565656565656
  → 99 * x = 56 → x = 56 / 99 :=
begin
  intros x h1 h2 h3 h4,
  sorry,
end

end repeating_decimal_fraction_l224_224781


namespace balloon_permutation_count_l224_224926

-- Define the given conditions:
def balloon_letters : List Char := ['B', 'A', 'L', 'L', 'O', 'O', 'N']
def total_letters : Nat := 7
def count_L : Nat := 2
def count_O : Nat := 2

-- Statement to prove:
theorem balloon_permutation_count :
  (nat.factorial total_letters) / ((nat.factorial count_L) * (nat.factorial count_O)) = 1260 := by
  sorry

end balloon_permutation_count_l224_224926


namespace rectangle_circle_ratio_l224_224821

theorem rectangle_circle_ratio (r s : ℝ) (h : ∀ r s : ℝ, 2 * r * s - π * r^2 = π * r^2) : s / (2 * r) = π / 2 :=
by
  sorry

end rectangle_circle_ratio_l224_224821


namespace triangle_lattice_points_l224_224288

theorem triangle_lattice_points :
  let S := 300
  let L := 60
  let N := 271 in
  S = N + (1/2 : ℚ) * L - 1 :=
by
  -- Calculate the number of lattice points on OA; 31
  -- Calculate the number of lattice points on OB; 10
  -- Calculate the number of lattice points on AB; 19
  -- Sum these to get L: 31 + 10 + 19 = 60
  -- Calculate the area of triangle ABO: 300
  -- Use Pick's theorem to solve for N: 271
  sorry

end triangle_lattice_points_l224_224288


namespace kolya_is_collection_agency_l224_224210

structure Condition :=
  (katya_lent_books_to_vasya : ∀ books : Type, katya_lent books vasya)
  (vasya_failed_to_return_books : ∀ (books : Type) (month_later : Time), failed_return books vasya month_later)
  (katya_asked_kolya_to_retrieve : ∀ books : Type, katya_asks books katya kolya vaaya)
  (kolya_agrees_for_reward : ∀ book : Type, kolya_agrees book katya_retrieve books)

theorem kolya_is_collection_agency {books : Type} (h : Condition books) : 
  KolyaRole = CollectionAgency :=
sorry

end kolya_is_collection_agency_l224_224210


namespace mixed_number_sum_l224_224713

theorem mixed_number_sum : (2 + (1 / 10 : ℝ)) + (3 + (11 / 100 : ℝ)) = 5.21 := by
  sorry

end mixed_number_sum_l224_224713


namespace distance_probability_l224_224704

def city_pairs : List Nat := [6200, 6750, 5800, 11700, 6000, 7400]

theorem distance_probability : 
  let count := city_pairs.countp (λ x => x < 6500) in
  let total := city_pairs.length in
  (count.toRational / total.toRational) = 1 / 2 :=
by
  sorry

end distance_probability_l224_224704


namespace quadratic_trinomials_have_two_roots_l224_224516

theorem quadratic_trinomials_have_two_roots
  (a1 a2 a3 b1 b2 b3 : ℝ)
  (h1 : a1 * a2 * a3 = b1 * b2 * b3)
  (h2 : a1 * a2 * a3 > 1) :
  ∃ k ∈ {1, 2, 3}, (let a := [a1, a2, a3].nth k sorry; let b := [b1, b2, b3].nth k sorry in ∃ r s : ℝ, r ≠ s ∧ r * s = b ∧ r + s = -2 * a) :=
sorry

end quadratic_trinomials_have_two_roots_l224_224516


namespace liquid_level_ratio_l224_224361

theorem liquid_level_ratio
  (h_1 h_2 : ℝ) 
  (hc : (1/3) * π * (4 ^ 2) * h_1 = (1/3) * π * (8 ^ 2) * h_2)
  (r_marble : ℝ := 2) 
  (v_marble : ℝ := (4/3) * π * (r_marble ^ 3)) :
  ((4 * (∛3 - 1)) / ((∛(3 / 2)) - 1)) = 29 := sorry

end liquid_level_ratio_l224_224361


namespace Integers_and_fractions_are_rational_numbers_l224_224023

-- Definitions from conditions
def is_fraction (x : ℚ) : Prop :=
  ∃a b : ℤ, b ≠ 0 ∧ x = (a : ℚ) / (b : ℚ)

def is_integer (x : ℤ) : Prop := 
  ∃n : ℤ, x = n

def is_rational (x : ℚ) : Prop := 
  ∃a b : ℤ, b ≠ 0 ∧ x = (a : ℚ) / (b : ℚ)

-- The statement to be proven
theorem Integers_and_fractions_are_rational_numbers (x : ℚ) : 
  (∃n : ℤ, x = (n : ℚ)) ∨ is_fraction x ↔ is_rational x :=
by sorry

end Integers_and_fractions_are_rational_numbers_l224_224023


namespace train_speed_approx_900072_kmph_l224_224840

noncomputable def speed_of_train (train_length platform_length time_seconds : ℝ) : ℝ :=
  let total_distance := train_length + platform_length
  let speed_m_s := total_distance / time_seconds
  speed_m_s * 3.6

theorem train_speed_approx_900072_kmph :
  abs (speed_of_train 225 400.05 25 - 90.0072) < 0.001 :=
by
  sorry

end train_speed_approx_900072_kmph_l224_224840


namespace arithmetic_sequence_linear_polynomial_l224_224805

variable {ℕ : Type*} [Nat ℕ]

theorem arithmetic_sequence_linear_polynomial 
  (a : ℕ → ℕ)
  (h0 : a 0 ≠ a 1)
  (h1 : ∀ i : ℕ, i ≥ 1 → a (i - 1) + a (i + 1) = 2 * a i) :
  ∀ n x : ℕ,
  let p := ∑ k in finRange (n + 1), a k * nat.choose n k * x ^ k
  in p = a 0 + (a 1 - a 0) * n * x :=
by
  sorry

end arithmetic_sequence_linear_polynomial_l224_224805


namespace balloon_permutation_count_l224_224913

-- Define the necessary factorials and the formula for permutations of multiset
def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem balloon_permutation_count : 
  let total_letters := 7,
      repeat_L := 2,
      repeat_O := 2,
      unique_arrangements := factorial total_letters / (factorial repeat_L * factorial repeat_O) 
  in unique_arrangements = 1260 :=
by
  sorry

end balloon_permutation_count_l224_224913


namespace sherman_drives_nine_hours_a_week_l224_224668

-- Define the daily commute time in minutes.
def daily_commute_time := 30 + 30

-- Define the number of weekdays Sherman commutes.
def weekdays := 5

-- Define the weekly commute time in minutes.
def weekly_commute_time := weekdays * daily_commute_time

-- Define the conversion from minutes to hours.
def minutes_to_hours (m : ℕ) : ℕ := m / 60

-- Define the weekend driving time in hours.
def weekend_driving_time := 2 * 2

-- Define the total weekly driving time in hours.
def total_weekly_driving_time := minutes_to_hours weekly_commute_time + weekend_driving_time

-- The theorem we need to prove
theorem sherman_drives_nine_hours_a_week :
  total_weekly_driving_time = 9 :=
by
  sorry

end sherman_drives_nine_hours_a_week_l224_224668


namespace line_general_eq_curve_rect_eq_dist_product_PA_PB_l224_224208

-- Definitions
def line_l (t : ℝ) : ℝ × ℝ := (-1 - t, 2 + t)
def curve_C_polar (θ : ℝ) : ℝ := real.sqrt (2 / (1 + (real.sin θ)^2))
def point_P := (real.sqrt(2) / 2, real.pi / 4)

-- Statements
theorem line_general_eq (t : ℝ) : ∃ a b c, a = 1 ∧ b = 1 ∧ c = -1 ∧ ∀ x y, ((x, y) = line_l t → a * x + b * y + c = 0) :=
sorry

theorem curve_rect_eq (x y : ℝ) : (x^2 / 2) + y^2 = 1 ↔ ∃ θ, (x, y) = (curve_C_polar θ * real.cos θ, curve_C_polar θ * real.sin θ) :=
sorry

theorem dist_product_PA_PB : let P := (1/2, 1/2), A := (0, 1), B := (4/3, -1/3) in
    real.sqrt ((P.1 - A.1)^2 + (P.2 - A.2)^2) * real.sqrt ((P.1 - B.1)^2 + (P.2 - B.2)^2) = 5/6 :=
sorry

end line_general_eq_curve_rect_eq_dist_product_PA_PB_l224_224208


namespace inversely_proportional_x_y_l224_224677

-- Statement of the problem
theorem inversely_proportional_x_y :
  ∃ k : ℝ, (∀ (x y : ℝ), (x * y = k) ∧ (x = 4) ∧ (y = 2) → x * (-5) = -8 / 5) :=
by
  sorry

end inversely_proportional_x_y_l224_224677


namespace relationship_inequality_l224_224254

theorem relationship_inequality (a b : ℝ) (h1 : a > 0) (h2 : (-1 < b) ∧ (b < 0)) : ab < ab^2 < a :=
by
  sorry

end relationship_inequality_l224_224254


namespace ratio_of_typing_speeds_l224_224387

-- Defining Tim's and Tom's typing speeds
variables (T M : ℝ)

-- Conditions given in the problem
def condition1 : Prop := T + M = 15
def condition2 : Prop := T + 1.6 * M = 18

-- Conclusion to be proved: the ratio of M to T is 1:2
theorem ratio_of_typing_speeds (h1 : condition1 T M) (h2 : condition2 T M) :
  M / T = 1 / 2 :=
by
  -- skip the proof
  sorry

end ratio_of_typing_speeds_l224_224387


namespace count_valid_c_values_l224_224133

-- Define the interval and the equation
def interval := Set.Icc 0 2000
def equation (x : ℝ) (c : ℝ) : Prop := 10 * (Real.floor x) + 3 * (Real.ceil x) = c

-- Define the proof problem
theorem count_valid_c_values : 
  (∃ c ∈ interval, ∃ x : ℝ, equation x c) → 
  Finset.card (Finset.filter (λ (c : ℝ), ∃ x : ℝ, equation x c) (Finset.Icc 0 2000)) = 308 := 
sorry

end count_valid_c_values_l224_224133


namespace minimum_value_2a_plus_3b_is_25_l224_224524

noncomputable def minimum_value_2a_plus_3b (a b : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : (2 / a) + (3 / b) = 1) : ℝ :=
2 * a + 3 * b

theorem minimum_value_2a_plus_3b_is_25
  (a b : ℝ)
  (h₁ : 0 < a)
  (h₂ : 0 < b)
  (h₃ : (2 / a) + (3 / b) = 1) :
  minimum_value_2a_plus_3b a b h₁ h₂ h₃ = 25 :=
sorry

end minimum_value_2a_plus_3b_is_25_l224_224524


namespace lengths_C_can_form_triangle_l224_224374

-- Definition of sets of lengths
def lengths_A := (3, 6, 9)
def lengths_B := (3, 5, 9)
def lengths_C := (4, 6, 9)
def lengths_D := (2, 6, 4)

-- Triangle condition for a given set of lengths
def can_form_triangle (a b c : Nat) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

-- Proof problem statement 
theorem lengths_C_can_form_triangle : can_form_triangle 4 6 9 :=
by
  sorry

end lengths_C_can_form_triangle_l224_224374


namespace balloon_permutation_count_l224_224916

-- Define the necessary factorials and the formula for permutations of multiset
def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem balloon_permutation_count : 
  let total_letters := 7,
      repeat_L := 2,
      repeat_O := 2,
      unique_arrangements := factorial total_letters / (factorial repeat_L * factorial repeat_O) 
  in unique_arrangements = 1260 :=
by
  sorry

end balloon_permutation_count_l224_224916


namespace no_statements_are_correct_l224_224228

-- Conditions
variables 
  (A B C M K : Type)
  [inhabited A] [inhabited B] [inhabited C] [inhabited M] [inhabited K]
  (AM BK : ℝ)
  (h_AM : AM = 3)
  (h_BK : BK = 5)

-- Statement
theorem no_statements_are_correct
  (h_AM_is_3 : AM = 3)
  (h_BK_is_5 : BK = 5)
  : ¬(AB = 6 ∨ perimeter_ABC = 22 ∨ ¬(can_estimate_perimeter ∨ can_estimate_AB)) :=
sorry

end no_statements_are_correct_l224_224228


namespace recurring_decimal_sum_l224_224468

theorem recurring_decimal_sum (x y : ℚ) (hx : x = 4/9) (hy : y = 7/9) :
  x + y = 11/9 :=
by
  rw [hx, hy]
  exact sorry

end recurring_decimal_sum_l224_224468


namespace max_pyramid_edges_intersected_by_plane_l224_224366

-- Our goal is to show the maximum number of edges of an n-sided pyramid intersected by a plane.
-- Define the problem according to the given conditions.

-- We define the pyramid and the plane intersection condition.
theorem max_pyramid_edges_intersected_by_plane (n : ℕ) : 
  ∃ k : ℕ, k = (3 * n) / 2 ∧ (plane_max_intersect n = k) :=
sorry

end max_pyramid_edges_intersected_by_plane_l224_224366


namespace card_symm_prob_l224_224345

def shape : Type := ℕ
def circle : shape := 1
def parallelogram : shape := 2
def isosceles_triangle : shape := 3
def rectangle : shape := 4
def square : shape := 5

def symm_about_axis (s : shape) : Prop :=
  s = circle ∨ s = rectangle ∨ s = square

def symm_about_center (s : shape) : Prop :=
  s = circle ∨ s = rectangle ∨ s = square

def symmetric_shape (s : shape) : Prop :=
  symm_about_axis s ∧ symm_about_center s

def all_shapes : list shape := [circle, parallelogram, isosceles_triangle, rectangle, square]

def shapes_symmetric_about_axis_and_center : list shape :=
  list.filter symmetric_shape all_shapes

theorem card_symm_prob : (list.length shapes_symmetric_about_axis_and_center : ℚ) / (list.length all_shapes) = 3 / 5 :=
by
  sorry

end card_symm_prob_l224_224345


namespace balloon_permutation_count_l224_224928

-- Define the given conditions:
def balloon_letters : List Char := ['B', 'A', 'L', 'L', 'O', 'O', 'N']
def total_letters : Nat := 7
def count_L : Nat := 2
def count_O : Nat := 2

-- Statement to prove:
theorem balloon_permutation_count :
  (nat.factorial total_letters) / ((nat.factorial count_L) * (nat.factorial count_O)) = 1260 := by
  sorry

end balloon_permutation_count_l224_224928


namespace all_jobs_filled_after_switch_l224_224811

-- Definition: Given conditions
variables (n : ℕ) (hn : n = 100)
variables (graduates : fin n → ℕ) (offer1 offer2 : fin n → ℕ)

-- Assumption: Unique proposals and all jobs filled initially
axiom proposals_unique_1 : ∀ (i j : fin n), (offer1 i = offer1 j) → i = j
axiom proposals_unique_2 : ∀ (i j : fin n), (offer2 i = offer2 j) → i = j
axiom all_jobs_filled_initially : (graduated : fin n) → offer1 graduated ≠ offer2 graduated 

-- Theorem: All jobs will still be filled after switching proposals
theorem all_jobs_filled_after_switch : 
  (∀ (graduated : fin n), (offer2 graduated = offer1 graduated) ∨ (offer1 graduated = offer1 graduated)) →
  (∀ (i : fin n), ∃ (graduated : fin n), offer2 (graduates graduated) = i) :=
by
sintro h
apply all_jobs_filled_initially 
sorry

end all_jobs_filled_after_switch_l224_224811


namespace constant_term_of_expansion_l224_224364

noncomputable def constant_term := 
  (20: ℕ) * (216: ℕ) * (1/27: ℚ) = (160: ℕ)

theorem constant_term_of_expansion : constant_term :=
  by sorry

end constant_term_of_expansion_l224_224364


namespace option_B_same_function_l224_224429

-- Definition of functions for Option B
def f_B (x : ℝ) := 2 * abs x
def g_B (x : ℝ) := sqrt (4 * x ^ 2)

-- Lean 4 statement to check if f_B and g_B represent the same function
theorem option_B_same_function : ∀ x : ℝ, f_B x = g_B x :=
by
  intros x
  sorry

end option_B_same_function_l224_224429


namespace digit_in_ten_thousandths_place_l224_224017

theorem digit_in_ten_thousandths_place : 
  (let decimal_eq := 7 / 32 in 
  let ten_thousandths_place := (decimal_eq * 100000) % 10 in 
  ten_thousandths_place = 5) := 
begin
  sorry -- not providing the proof here
end

end digit_in_ten_thousandths_place_l224_224017


namespace lattice_points_in_triangle_271_l224_224284

def Pick_theorem (N L S : ℝ) : Prop :=
  S = N + (1/2) * L - 1

noncomputable def lattice_triangle_inside_points (A B O : ℕ × ℕ) : ℕ :=
  271

theorem lattice_points_in_triangle_271 :
  ∃ N L : ℝ, 
  let S := 300 in
  let L := 60 in
  A = (0, 30) ∧ B = (20, 10) ∧
  O = (0, 0) ∧ Pick_theorem N L S :=
sorry

end lattice_points_in_triangle_271_l224_224284


namespace orthocenter_in_XaXbXc_l224_224072

noncomputable def orthocenter (A B C : Point) : Point := sorry
noncomputable def reflect_across_line (X : Point) (l : Line) : Point := sorry
noncomputable def triangle (A B C : Point) : Set Point := sorry
noncomputable def is_interior_point (P : Point) (ABC : Set Point) : Prop := sorry

theorem orthocenter_in_XaXbXc 
  (A B C H X : Point) 
  (hA : is_interior_point A (triangle A B C))
  (hB : is_interior_point B (triangle A B C))
  (hC : is_interior_point C (triangle A B C))
  (hH : H = orthocenter A B C)
  (hX : is_interior_point X (triangle A B C)) :
  let X_a := reflect_across_line X (line B C)
  let X_b := reflect_across_line X (line C A)
  let X_c := reflect_across_line X (line A B)
  is_interior_point H (triangle X_a X_b X_c) :=
sorry

end orthocenter_in_XaXbXc_l224_224072


namespace nasobek_children_ages_l224_224339

noncomputable def product_of_ages (ages : List ℕ) : ℕ :=
ages.foldr (λ x acc => x * acc) 1

theorem nasobek_children_ages :
  ∃ (ages : List ℕ), product_of_ages ages = 1408 ∧
  List.length ages = 3 ∧
  ages.nth 0 = some 8 ∧
  ages.nth 1 = some 11 ∧
  ages.nth 2 = some 16 ∧
  ages.nth 0 = some (ages.last.getD 0 / 2) :=
sorry

end nasobek_children_ages_l224_224339


namespace triangle_lattice_points_l224_224287

theorem triangle_lattice_points :
  let S := 300
  let L := 60
  let N := 271 in
  S = N + (1/2 : ℚ) * L - 1 :=
by
  -- Calculate the number of lattice points on OA; 31
  -- Calculate the number of lattice points on OB; 10
  -- Calculate the number of lattice points on AB; 19
  -- Sum these to get L: 31 + 10 + 19 = 60
  -- Calculate the area of triangle ABO: 300
  -- Use Pick's theorem to solve for N: 271
  sorry

end triangle_lattice_points_l224_224287


namespace balloon_permutation_count_l224_224920

-- Define the necessary factorials and the formula for permutations of multiset
def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem balloon_permutation_count : 
  let total_letters := 7,
      repeat_L := 2,
      repeat_O := 2,
      unique_arrangements := factorial total_letters / (factorial repeat_L * factorial repeat_O) 
  in unique_arrangements = 1260 :=
by
  sorry

end balloon_permutation_count_l224_224920


namespace peter_food_necessity_l224_224656

/-- Discuss the conditions  -/
def peter_horses (num_horses num_days : ℕ) (oats_per_meal grain_per_day : ℕ) (meals_per_day : ℕ) : ℕ :=
  let daily_oats := oats_per_meal * meals_per_day in
  let total_oats := daily_oats * num_days * num_horses in
  let total_grain := grain_per_day * num_days * num_horses in
  total_oats + total_grain

/-- Prove that Peter needs 132 pounds of food to feed his horses for 3 days -/
theorem peter_food_necessity : peter_horses 4 3 4 3 2 = 132 :=
  sorry

end peter_food_necessity_l224_224656


namespace num_times_greater_l224_224829

theorem num_times_greater (x y : ℝ) (h₁ : x = 18 * y) (h₂ : y = 0.0555555555555556 * x) : x = 18 * y := by
  /* proof omitted */
  sorry

end num_times_greater_l224_224829


namespace MishaTotalMoney_l224_224264

-- Define Misha's initial amount of money
def initialMoney : ℕ := 34

-- Define the amount of money Misha earns
def earnedMoney : ℕ := 13

-- Define the total amount of money Misha will have
def totalMoney : ℕ := initialMoney + earnedMoney

-- Statement to prove
theorem MishaTotalMoney : totalMoney = 47 := by
  sorry

end MishaTotalMoney_l224_224264


namespace positive_t_value_l224_224098

theorem positive_t_value (t : ℝ) (h : abs (-5 + t * complex.i) = 3 * real.sqrt 13) :
  t = 2 * real.sqrt 23 :=
sorry

end positive_t_value_l224_224098


namespace hyperbola_triangle_area_l224_224328

noncomputable theory
open_locale real

def hyperbola (x y : ℝ) := (x^2 / 9) - (y^2 / 16) = 1

def inclination_angle_difference (P F1 F2 : ℝ × ℝ) (θ : ℝ) := θ = π / 3

def area_of_triangle (P F1 F2 : ℝ × ℝ) : ℝ := 
  let b := 4 in -- 2b is 16, based on the hyperbola parameters
  b^2 * real.cot (π / 6)

theorem hyperbola_triangle_area (P F1 F2 : ℝ × ℝ)
  (hp : hyperbola P.1 P.2)
  (hf1 : hyperbola F1.1 F1.2)
  (hf2 : hyperbola F2.1 F2.2)
  (hθ : inclination_angle_difference P F1 F2 (π / 3)) :
  area_of_triangle P F1 F2 = 16 * real.sqrt 3 :=
sorry

end hyperbola_triangle_area_l224_224328


namespace units_digit_27_64_l224_224489

/-- 
  Given that the units digit of 27 is 7, 
  and the units digit of 64 is 4, 
  prove that the units digit of 27 * 64 is 8.
-/
theorem units_digit_27_64 : 
  ∀ (n m : ℕ), 
  (n % 10 = 7) → 
  (m % 10 = 4) → 
  ((n * m) % 10 = 8) :=
by
  intros n m h1 h2
  -- Utilize modular arithmetic properties
  sorry

end units_digit_27_64_l224_224489


namespace point_on_parabola_l224_224738

def parabola (x : ℝ) : ℝ := 2 * x^2 - 3 * x + 1

theorem point_on_parabola : parabola (1/2) = 0 := 
by sorry

end point_on_parabola_l224_224738


namespace sum_of_digits_A_squared_l224_224806

def A : ℕ := 10 ^ 221 - 1

theorem sum_of_digits_A_squared : (sum_of_digits (A ^ 2)) = 1989 :=
sorry

end sum_of_digits_A_squared_l224_224806


namespace angle_EKM_l224_224591

noncomputable def triangle_acute (α β γ : ℝ) : Prop :=
α + β + γ = 180 ∧ α < 90 ∧ β < 90 ∧ γ < 90

theorem angle_EKM (α β γ δ : ℝ) :
  triangle_acute α β γ ∧
  α = 53 ∧
  β = 81 ∧
  δ = 180 - α - β ∧
  δ = 180 - 53 - 81 ∧
  δ = 46 ∧
  true -- for simplicity, we add some axiomatic condition for perpendicular to triangle segment
  → (90 - δ = 44) :=
by
  intros,
  sorry

end angle_EKM_l224_224591


namespace number_of_male_employees_l224_224382

theorem number_of_male_employees (num_female : ℕ) (x y : ℕ) 
  (h1 : 7 * x = y) 
  (h2 : 8 * x = num_female) 
  (h3 : 9 * (7 * x + 3) = 8 * num_female) :
  y = 189 := by
  sorry

end number_of_male_employees_l224_224382


namespace four_digit_integers_divisible_by_nine_count_l224_224174

theorem four_digit_integers_divisible_by_nine_count : 
  let smallest := 1008,
      largest := 9999,
      diff := 9 in
  ((largest - smallest) / diff + 1) = 1000 :=
by -- We skip the proof, only the statement is present
  sorry

end four_digit_integers_divisible_by_nine_count_l224_224174


namespace bus_ride_cost_l224_224841

theorem bus_ride_cost (B T : ℝ) 
  (h1 : T = B + 6.85)
  (h2 : T + B = 9.65)
  (h3 : ∃ n : ℤ, B = 0.35 * n ∧ ∃ m : ℤ, T = 0.35 * m) : 
  B = 1.40 := 
by
  sorry

end bus_ride_cost_l224_224841


namespace factorable_polynomial_l224_224321

theorem factorable_polynomial (d f e g b : ℤ) (h1 : d * f = 28) (h2 : e * g = 14)
  (h3 : d * g + e * f = b) : b = 42 :=
by sorry

end factorable_polynomial_l224_224321


namespace repeating_decimal_eq_l224_224767

noncomputable def repeating_decimal : ℚ := 56 / 99

theorem repeating_decimal_eq : ∃ x : ℚ, x = repeating_decimal ∧ x = 56 / 99 :=
by
  use 56 / 99
  split
  all_goals { sorry }

end repeating_decimal_eq_l224_224767


namespace cut_rectangle_to_square_l224_224089

theorem cut_rectangle_to_square (a b : ℕ) (h₁ : a = 16) (h₂ : b = 9) :
  ∃ (s : ℕ), s * s = a * b ∧ s = 12 :=
by {
  sorry
}

end cut_rectangle_to_square_l224_224089


namespace sum_of_visible_numbers_not_80_l224_224808

theorem sum_of_visible_numbers_not_80 : 
  ∀ (cubes : Fin 8 → Fin 6 → ℕ),
  (∀ (i : Fin 8), cubes i = λ j, j.val + 1) →
  ∀ arrangement : Fin 8 → Fin 3 → Fin 3 → Fin 3,
  (∑ i, cubes (arrangement i) 0 + cubes (arrangement i) 1 + cubes (arrangement i) 2 + cubes (arrangement i) 3 + cubes (arrangement i) 4 + cubes (arrangement i) 5) ≠ 80 :=
sorry

end sum_of_visible_numbers_not_80_l224_224808


namespace peanuts_in_box_l224_224381

theorem peanuts_in_box (initial_peanuts : ℕ) (added_peanuts : ℕ) (total_peanuts : ℕ) : 
  initial_peanuts = 4 → added_peanuts = 2 → total_peanuts = 6 → initial_peanuts + added_peanuts = total_peanuts :=
by
  intros
  rw [h_fst, h_snd, h_trd]
  rfl

end peanuts_in_box_l224_224381


namespace johns_total_money_made_l224_224234

-- Defining the conditions
def total_caterpillars := 4 * 10
def failure_rate := 0.4
def success_rate := 1 - failure_rate
def price_per_butterfly := 3

-- Derived values
def caterpillars_failed := total_caterpillars * failure_rate
def caterpillars_successful := total_caterpillars - caterpillars_failed
def total_money_made := caterpillars_successful * price_per_butterfly

-- Statement to be proven
theorem johns_total_money_made : total_money_made = 72 := by
  sorry

end johns_total_money_made_l224_224234


namespace part_1_intersection_part_1_union_complement_part_2_no_intersection_l224_224620

def A := {x : ℝ | -2 ≤ x ∧ x ≤ 5}
def B (m : ℝ) := {x : ℝ | m-1 ≤ x ∧ x ≤ 2*m+1}

theorem part_1_intersection (m : ℝ) (h : m = 3) : 
  A ∩ B 3 = {x : ℝ | 2 ≤ x ∧ x ≤ 5} :=
by {
  sorry
}

theorem part_1_union_complement (m : ℝ) (h : m = 3) : 
  (set.compl A) ∪ B 3 = {x : ℝ | x < -2 ∨ x ≥ 2} :=
by {
  sorry
}

theorem part_2_no_intersection (m : ℝ) : 
  A ∩ B m = ∅ ↔ (m < -(3/2) ∨ 6 < m) :=
by {
  sorry
}

end part_1_intersection_part_1_union_complement_part_2_no_intersection_l224_224620


namespace recurring_decimal_to_fraction_correct_l224_224763

noncomputable def recurring_decimal_to_fraction (b : ℚ) : Prop :=
  b = 0.\overline{56} ↔ b = 56/99

theorem recurring_decimal_to_fraction_correct : recurring_decimal_to_fraction 0.\overline{56} :=
  sorry

end recurring_decimal_to_fraction_correct_l224_224763


namespace supervisor_quality_related_l224_224647

noncomputable def K_squared (a b c d : ℕ) : ℚ :=
  let n := a + b + c + d
  (n * (a * d - b * c)^2) / ((a + b) * (c + d) * (a + c) * (b + d))

theorem supervisor_quality_related : 
  let a := 982 in
  let b := 8 in
  let c := 493 in
  let d := 17 in
  let k_critical := 10.828 in
  K_squared a b c d > k_critical :=
by
  sorry

end supervisor_quality_related_l224_224647


namespace total_food_needed_l224_224655

-- Definitions for the conditions
def horses : ℕ := 4
def oats_per_meal : ℕ := 4
def oats_meals_per_day : ℕ := 2
def grain_per_day : ℕ := 3
def days : ℕ := 3

-- Theorem stating the problem
theorem total_food_needed :
  (horses * (days * (oats_per_meal * oats_meals_per_day) + days * grain_per_day)) = 132 :=
by sorry

end total_food_needed_l224_224655


namespace best_discount_sequence_l224_224698

/-- 
The initial price of the book is 30.
Stay focused on two sequences of discounts.
Sequence 1: $5 off, then 10% off, then $2 off if applicable.
Sequence 2: 10% off, then $5 off, then $2 off if applicable.
Compare the final prices obtained from applying these sequences.
-/
noncomputable def initial_price : ℝ := 30
noncomputable def five_off (price : ℝ) : ℝ := price - 5
noncomputable def ten_percent_off (price : ℝ) : ℝ := 0.9 * price
noncomputable def additional_two_off_if_applicable (price : ℝ) : ℝ := 
  if price > 20 then price - 2 else price

noncomputable def sequence1_final_price : ℝ := 
  additional_two_off_if_applicable (ten_percent_off (five_off initial_price))

noncomputable def sequence2_final_price : ℝ := 
  additional_two_off_if_applicable (five_off (ten_percent_off initial_price))

theorem best_discount_sequence : 
  sequence2_final_price = 20 ∧ 
  sequence2_final_price < sequence1_final_price ∧ 
  sequence1_final_price - sequence2_final_price = 0.5 :=
by
  sorry

end best_discount_sequence_l224_224698


namespace minimal_abs_difference_l224_224569

theorem minimal_abs_difference (a b : ℕ) (h : a > 0 ∧ b > 0) : 
  (a * b - 4 * a + 3 * b = 128) → ∃ m : ℕ, m = 9 ∧ ∀ a' b' : ℕ, (a' > 0 ∧ b' > 0 ∧ a' * b' - 4 * a' + 3 * b' = 128) → |a' - b'| ≥ m :=
by
  sorry

end minimal_abs_difference_l224_224569


namespace train_crossing_time_l224_224602

-- Given conditions
def train_length : ℝ := 100
def train_speed_kmh : ℝ := 72

-- Conversion factor
def kmh_to_ms (kmh : ℝ) : ℝ := kmh * (1000 / 3600)

-- Theorem to be proven
theorem train_crossing_time : 
  let speed_ms := kmh_to_ms train_speed_kmh in
  let time := train_length / speed_ms in
  time = 5 :=
by 
  -- Compute the speed in m/s
  have speed_ms : ℝ := kmh_to_ms train_speed_kmh,
  -- Compute the time to cross the pole
  have time : ℝ := train_length / speed_ms,
  -- Assert the correct time
  show time = 5,
  sorry

end train_crossing_time_l224_224602


namespace min_dist_squared_l224_224183

theorem min_dist_squared 
    (a b : ℝ)
    (h : b = √3 * a - √3) :
    (a + 1)^2 + b^2 = 3 :=
sorry

end min_dist_squared_l224_224183


namespace perfect_square_factors_32400_l224_224557

theorem perfect_square_factors_32400 : 
  let n := 32400
  let factors := λ (a b c : ℕ),  (a ≤ 3) ∧ (b ≤ 4) ∧ (c ≤ 2)
  let perfect_square := λ (a b c : ℕ), (a % 2 = 0) ∧ (b % 2 = 0) ∧ (c % 2 = 0)
  ∃ (count : ℕ), count = 12 ∧ 
     count = (∑ a in Finset.range 4, ∑ b in Finset.range 5, ∑ c in Finset.range 3,
     if (perfect_square a b c) then 1 else 0) := 
by 
  sorry

end perfect_square_factors_32400_l224_224557


namespace room_wall_count_l224_224722

theorem room_wall_count :
  let wall_area := 2 * 3 -- Each wall is 2 meters by 3 meters
  let paint_rate := 1 / 10 -- John can paint 1 square meter every 10 minutes
  let total_time := 10 * 60 -- John has 10 hours to paint, converting to minutes
  let spare_time := 5 * 60 -- John has 5 hours to spare, converting to minutes
  let available_time := total_time - spare_time -- Time available for painting in minutes
  let total_area := available_time / paint_rate -- Total area John can paint
  let walls_to_paint := total_area / wall_area -- Number of walls that can be painted
  in walls_to_paint = 5 := sorry

end room_wall_count_l224_224722


namespace convex_quadrilateral_area_gt_21_l224_224587

variable {A B C D : ℝ × ℝ}

open real geometry

theorem convex_quadrilateral_area_gt_21
  (h_square : ∀ (P : ℝ × ℝ), P = A ∨ P = B ∨ P = C ∨ P = D → P.1 ≥ 0 ∧ P.1 ≤ 6 ∧ P.2 ≥ 0 ∧ P.2 ≤ 6)
  (h_dist : ∀ (P Q : ℝ × ℝ), (P = A ∨ P = B ∨ P = C ∨ P = D) ∧ (Q = A ∨ Q = B ∨ Q = C ∨ Q = D) ∧ P ≠ Q → dist P Q ≥ 5) :
  convex_hull {A, B, C, D}.convex ∧
  area_of_convex_quadrilateral {A, B, C, D} > 21 :=
by
  sorry

end convex_quadrilateral_area_gt_21_l224_224587


namespace problem_l224_224509

theorem problem (x y z : ℝ) (h1 : x = y + z) (h2 : x = 2) : 
  x^3 + 3 * y^2 + 3 * z^2 + 3 * x * y * z = 20 := by
sorry

end problem_l224_224509


namespace rectangular_bed_ratio_l224_224417

theorem rectangular_bed_ratio (s y x : ℝ) (h1 : 0 < s) (h2 : 0 < y) (h3 : 0 < x)
  (bed_area_condition : (s + 2 * y) ^ 2 = 3 * (s ^ 2)) 
  (bed_dim_condition : x + y = s * sqrt 3) :
  (x / y) = 2 + sqrt 3 :=
by
  sorry

end rectangular_bed_ratio_l224_224417


namespace sherman_drives_nine_hours_a_week_l224_224670

-- Define the daily commute time in minutes.
def daily_commute_time := 30 + 30

-- Define the number of weekdays Sherman commutes.
def weekdays := 5

-- Define the weekly commute time in minutes.
def weekly_commute_time := weekdays * daily_commute_time

-- Define the conversion from minutes to hours.
def minutes_to_hours (m : ℕ) : ℕ := m / 60

-- Define the weekend driving time in hours.
def weekend_driving_time := 2 * 2

-- Define the total weekly driving time in hours.
def total_weekly_driving_time := minutes_to_hours weekly_commute_time + weekend_driving_time

-- The theorem we need to prove
theorem sherman_drives_nine_hours_a_week :
  total_weekly_driving_time = 9 :=
by
  sorry

end sherman_drives_nine_hours_a_week_l224_224670


namespace anna_should_plant_8_lettuce_plants_l224_224074

/-- Anna wants to grow some lettuce in the garden and would like to grow enough to have at least
    12 large salads.
- Conditions:
  1. Half of the lettuce will be lost to insects and rabbits.
  2. Each lettuce plant is estimated to provide 3 large salads.
  
  Proof that Anna should plant 8 lettuce plants in the garden. --/
theorem anna_should_plant_8_lettuce_plants 
    (desired_salads: ℕ)
    (salads_per_plant: ℕ)
    (loss_fraction: ℚ) :
    desired_salads = 12 →
    salads_per_plant = 3 →
    loss_fraction = 1 / 2 →
    ∃ plants: ℕ, plants = 8 :=
by
  intros h1 h2 h3
  sorry

end anna_should_plant_8_lettuce_plants_l224_224074


namespace product_arrangements_l224_224853

def is_left_of (A B : Nat) : Prop := A < B

theorem product_arrangements : 
  ∀ (A B C D : Nat), 
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D ∧
  is_left_of A C ∧ is_left_of B C →
  fintype.card {arrangement : List Nat // arrangement.perm [A, B, C, D] ∧ is_left_of (arrangement.inth 0) C ∧ is_left_of (arrangement.inth 1) C } = 8 :=
by
  intros A B C D h,
  -- Proof would go here
  sorry

end product_arrangements_l224_224853


namespace flow_velocity_range_maximum_flow_rate_l224_224327

noncomputable def v (x : ℝ) : ℝ :=
  if h : 0 < x ∧ x ≤ 10 then 100
  else if h : 10 < x ∧ x ≤ 110 then 150 - 7500 / (160 - x)
  else 0 -- Not defined outside (0, 110]

theorem flow_velocity_range (x : ℝ) : v x ≥ 90 → 0 < x ∧ x ≤ 35 :=
begin
  sorry,
end

noncomputable def y (x : ℝ) : ℝ :=
  x * v x

theorem maximum_flow_rate : 
  ∃ (x y : ℝ), x = 71 ∧ y = 4668 ∧ ∀ t ≤ 71, y t ≤ y :=
begin
  sorry,
end

end flow_velocity_range_maximum_flow_rate_l224_224327


namespace fraction_equivalence_l224_224427

theorem fraction_equivalence (x y : ℝ) (h : x ≠ y) :
  (x - y)^2 / (x^2 - y^2) = (x - y) / (x + y) :=
by
  sorry

end fraction_equivalence_l224_224427


namespace find_a2_and_sum_l224_224175

theorem find_a2_and_sum (a a1 a2 a3 a4 : ℝ) (x : ℝ) (h1 : (1 + 2 * x)^4 = a + a1 * x + a2 * x^2 + a3 * x^3 + a4 * x^4) :
  a2 = 24 ∧ a + a1 + a2 + a3 + a4 = 81 :=
by
  sorry

end find_a2_and_sum_l224_224175


namespace estimate_total_number_of_fish_l224_224049

-- Define the conditions
variables (totalMarked : ℕ) (secondSample : ℕ) (markedInSecondSample : ℕ) (N : ℕ)

-- Assume the conditions
axiom condition1 : totalMarked = 60
axiom condition2 : secondSample = 80
axiom condition3 : markedInSecondSample = 5

-- Lean theorem statement proving N = 960 given the conditions
theorem estimate_total_number_of_fish (totalMarked secondSample markedInSecondSample N : ℕ)
  (h1 : totalMarked = 60)
  (h2 : secondSample = 80)
  (h3 : markedInSecondSample = 5) :
  N = 960 :=
sorry

end estimate_total_number_of_fish_l224_224049


namespace sequence_periodic_a_24_value_l224_224544

noncomputable def sequence (n : ℕ) : ℚ :=
  if h : n > 0 then
    Nat.recOn (n - 1)
      (6 / 7 : ℚ)
      (fun k a_k => if a_k < 1/2 then 2 * a_k else 2 * a_k - 1)
  else 0

theorem sequence_periodic :
  ∀ n : ℕ, sequence (n + 3) = sequence n := by
  sorry

theorem a_24_value : sequence 24 = 3 / 7 := by
  have periodicity := sequence_periodic 21
  show sequence 24 = 3 / 7, because sequence 24 = sequence (21 + 3) = sequence 21 = 3 / 7
  sorry

end sequence_periodic_a_24_value_l224_224544


namespace minimum_strikes_l224_224604

structure Dragon where
  heads : ℕ
  tails : ℕ

def strike (d : Dragon) (n : ℕ) : Dragon :=
  match n with
  | 0 => { heads := d.heads + 1, tails := d.tails }       -- chop one head
  | 1 => { heads := d.heads, tails := d.tails + 1 }       -- chop one tail
  | 2 => { heads := d.heads - 2, tails := d.tails }       -- chop two heads
  | _ => { heads := d.heads + 1, tails := d.tails - 2 }   -- chop two tails

theorem minimum_strikes (d : Dragon) : d.heads = 3 ∧ d.tails = 3 → ∃ s : list ℕ, s.length = 9 ∧ (s.foldl strike d) = {heads := 0, tails := 0} := by
  sorry

end minimum_strikes_l224_224604


namespace students_class_division_l224_224819

theorem students_class_division (n : ℕ) (h1 : n % 15 = 0) (h2 : n % 24 = 0) : n = 120 :=
sorry

end students_class_division_l224_224819


namespace parallelogram_area_l224_224410

theorem parallelogram_area 
  (b : ℕ) (h : ℕ) (A : ℕ) 
  (h_eq : h = 2 * b) (b_eq : b = 11) : 
  A = b * h :=
by {
  have h_value : h = 22 := by rw [h_eq, b_eq]; linarith,
  have A_value : A = 11 * 22 := by rw [←b_eq, h_value],
  rw A_value,
  norm_num,
}

end parallelogram_area_l224_224410


namespace condition_I_condition_II_l224_224164

/-- Definition of the function f(x). -/
def f (x : ℝ) (a : ℝ) : ℝ := x * Real.log x + a * x

/-- Condition I: f(x) is decreasing on [e, e^2]. -/
def is_decreasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x ∈ set.Icc a b, f' x ≤ 0

/-- Helper lemma to find the derivative of f. -/
lemma f_derivative (x a : ℝ) : deriv (λ x, f x a) x = Real.log x + a + 1 := by
  sorry

/-- To prove: If f(x) is decreasing on [e, e^2], then a ≤ -3. -/
theorem condition_I (a : ℝ) :
  is_decreasing_on (λ x, f x a) Real.exp (Real.exp 2) → a ≤ -3 :=
by
  sorry

/-- Condition II: For any x > 1, f(x) > k(x - 1) + ax - x. -/
def holds_for_any_x (f : ℝ → ℝ) (k a : ℝ) : Prop :=
  ∀ x : ℝ, 1 < x → f x > k * (x - 1) + a * x - x

/-- To prove: If for any x > 1, f(x) > k(x - 1) + ax - x, then k ≤ 3. -/
theorem condition_II (a k : ℝ) :
  holds_for_any_x (λ x, f x a) k a → k ≤ 3 :=
by
  sorry

end condition_I_condition_II_l224_224164


namespace count_four_digit_integers_with_1_or_7_l224_224559

/-- 
The total number of four-digit integers with at least one digit being 1 or 7 is 5416.
-/
theorem count_four_digit_integers_with_1_or_7 : 
  let all_four_digit_integers := 9000
  let without_1_or_7 := 7 * 8 * 8 * 8
  let with_1_or_7 := all_four_digit_integers - without_1_or_7
  with_1_or_7 = 5416
:= by
  let all_four_digit_integers := 9000
  let without_1_or_7 := 7 * 8 * 8 * 8
  let with_1_or_7 := all_four_digit_integers - without_1_or_7
  show with_1_or_7 = 5416
  sorry

end count_four_digit_integers_with_1_or_7_l224_224559


namespace tetrahedron_circumcenter_unique_sphere_through_two_circles_l224_224027

-- Problem (a): Tetrahedron circumcenter intersection
theorem tetrahedron_circumcenter {A1 A2 A3 A4 : Point} (cA1A2A3 cA1A2A4 cA1A3A4 cA2A3A4 : Circle)
    (P1 P2 P3 P4 : Point)
    (hA1A2A3 : center cA1A2A3 = P1 ∧ ∀ P ∈ plane A1 A2 A3, dist P1 P = dist P A1)
    (hA1A2A4 : center cA1A2A4 = P2 ∧ ∀ P ∈ plane A1 A2 A4, dist P2 P = dist P A1)
    (hA1A3A4 : center cA1A3A4 = P3 ∧ ∀ P ∈ plane A1 A3 A4, dist P3 P = dist P A1)
    (hA2A3A4 : center cA2A3A4 = P4 ∧ ∀ P ∈ plane A2 A3 A4, dist P4 P = dist P A2) :
    ∃ O : Point, ∀ i ∈ {P1, P2, P3, P4}, on_perpendicular O i := sorry

-- Problem (b): Draw sphere through two intersecting circles
theorem unique_sphere_through_two_circles (circle1 circle2 : Circle)
    (A B : Point)
    (h_intersect : intersects circle1 circle2 A ∧ intersects circle1 circle2 B ∧ A ≠ B) :
    ∃! sphere, (∀ P ∈ circle1, on_sphere sphere P) ∧ (∀ Q ∈ circle2, on_sphere sphere Q) := sorry

end tetrahedron_circumcenter_unique_sphere_through_two_circles_l224_224027


namespace balloon_permutations_l224_224938

theorem balloon_permutations : 
  let balloons := "BALLOON".toList.length in
  let total_permutations := Nat.factorial balloons in
  let repetitive_L := Nat.factorial 2 in
  let repetitive_O := Nat.factorial 2 in
  (total_permutations / (repetitive_L * repetitive_O)) = 1260 := sorry

end balloon_permutations_l224_224938


namespace number_of_triangles_with_conditions_l224_224454

theorem number_of_triangles_with_conditions :
  let valid_x_coords := { x : ℕ | 0 ≤ x ∧ 39 * x ≤ 2070 }
  let even_count := 27
  let odd_count := 26 
    ∑ (x1 : ℕ) in valid_x_coords, ∑ (x2 : ℕ) in valid_x_coords,
      39 * x1 + 3 * y1 = 2070 ∧ 39 * x2 + 3 * y2 = 2070 ∧ 
      3 ∣ (x1 - x2) ∧ (x1 ≠ x2)  = 676 :=
  by sorry

end number_of_triangles_with_conditions_l224_224454


namespace repeating_decimal_as_fraction_l224_224744

theorem repeating_decimal_as_fraction :
  let x := 56 / 99
  in x = 0.56 + 0.0056 + 0.000056 + (0.00000056) : ℚ :=
by
  sorry

end repeating_decimal_as_fraction_l224_224744


namespace log_lt_square_lt_exp_l224_224989

theorem log_lt_square_lt_exp (a b c : ℝ) (h₀ : a = 0.3) (h₁ : b = log 0.3 / log 2) (h₂ : c = 0.3^2) (h₃ : d = 2^0.3) :
  b < c ∧ c < d := by
  sorry

end log_lt_square_lt_exp_l224_224989


namespace pencils_per_row_l224_224108

theorem pencils_per_row (P : ℕ) (rows : ℕ) (crayons_per_row : ℕ) (total : ℕ) :
  rows = 11 → crayons_per_row = 27 → total = 638 → 11 * P + 11 * crayons_per_row = total → P = 31 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2] at h4
  simp at h4
  exact (nat.mul_right_eq (11:ℕ)).1 h4

end pencils_per_row_l224_224108


namespace unique_number_with_units_digit_2_in_range_l224_224486

theorem unique_number_with_units_digit_2_in_range :
  ∃! n : ℕ, (30 < n) ∧ (n < 40) ∧ (n % 10 = 2) := 
begin
  sorry
end

end unique_number_with_units_digit_2_in_range_l224_224486


namespace fraction_subtraction_simplify_l224_224861

theorem fraction_subtraction_simplify :
  (9 / 19 - 3 / 57 - 1 / 3) = 5 / 57 :=
by
  sorry

end fraction_subtraction_simplify_l224_224861


namespace problem_solution_l224_224177

theorem problem_solution (a b c d : ℝ) (h1 : a = 5 * b) (h2 : b = 3 * c) (h3 : c = 6 * d) :
  (a + b * c) / (c + d * b) = (3 * (5 + 6 * d)) / (1 + 3 * d) :=
by
  sorry

end problem_solution_l224_224177


namespace sqrt_cbrt_of_0_001024_eq_l224_224446

theorem sqrt_cbrt_of_0_001024_eq : (Real.sqrt (Real.cbrt 0.001024)).round = 1.1 :=
sorry

end sqrt_cbrt_of_0_001024_eq_l224_224446


namespace symmetric_point_x_axis_l224_224313

theorem symmetric_point_x_axis (P : ℝ × ℝ) (hx : P = (2, 3)) : P.1 = 2 ∧ P.2 = -3 :=
by
  -- The proof is omitted
  sorry

end symmetric_point_x_axis_l224_224313


namespace variance_of_transformed_variable_l224_224258

-- Define the binomial random variable
variable (Ω : Type) [ProbabilitySpace Ω]

def X : Ω → ℕ := sorry 
axiom binom_X : Binomial 10 (0.8) X

-- Define the transformed random variable
def Y (ω : Ω) := 2 * X ω + 1

-- Prove the variance of 2X + 1
theorem variance_of_transformed_variable : (Var[Y]) = 6.4 :=
by {
  -- Expected proof goes here
  sorry
}

end variance_of_transformed_variable_l224_224258


namespace sequence_sum_identity_l224_224209

noncomputable def a : ℕ → ℕ
| 0       := 0
| (n + 1) := match n with
             | 0        := 1
             | (m + 1)  := a (m + 1) + a 1 + (m + 1)

theorem sequence_sum_identity :
  (a 1 = 1) →
  (∀ m n : ℕ, 0 < m → 0 < n → a (m + n) = a m + a n + m * n) →
  (∑ t in finset.range 2019, (1:ℚ) / a (t + 1)) = 2019 / 1010 :=
  by sorry

end sequence_sum_identity_l224_224209


namespace sum_of_third_row_l224_224689

-- Define the grid consisting of integers 1 to 9
def grid : Type := matrix (fin 3) (fin 3) ℕ

-- Define the conditions
def distinct_integers (g : grid) : Prop :=
  ∀ i j, i ≠ j -> g i ≠ g j

def row_product (g : grid) (r : fin 3) (product : ℕ) : Prop :=
  g (r, 0) * g (r, 1) * g (r, 2) = product

-- Express the theorem
theorem sum_of_third_row (g : grid) 
  (h_distinct : distinct_integers g)
  (h_product_row1 : row_product g ⟨0⟩ 60)
  (h_product_row2 : row_product g ⟨1⟩ 96) :
  (g ⟨2⟩ ⟨0⟩ + g ⟨2⟩ ⟨1⟩ + g ⟨2⟩ ⟨2⟩) = 17 := 
sorry

end sum_of_third_row_l224_224689


namespace triangle_statements_correct_l224_224433

theorem triangle_statements_correct :
  let statement_1 := ∀ (T : Triangle), 
                     (T.acute → (∀ (A : Altitude T), A.inside)) ∧ 
                     (T.right → (∃ (A : Altitude T), A.inside ∧ 
                      ∀ (A' ∈ Set.remove (Altitudes T) A, A'.on_side_of T))) ∧ 
                     (T.obtuse → (∃ (A : Altitude T), A.inside ∧ 
                      ∀ (A' ∈ Set.remove (Altitudes T) A, A'.outside T)))
    statement_2 := ∀ (a b c : ℝ), (a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c) → Triangle.formable a b c
    statement_3 := ∃ (T : Triangle), (T.angle_ratio 3 2 1 ∧ T.right)
    statement_4 := ∀ (T₁ T₂ : Triangle), (T₁.angle_eq T₂ ∧ T₁.side_eq T₂) → T₁ ≅ T₂
  in
  statement_1 ∧ ¬statement_2 ∧ statement_3 ∧ statement_4 → 
  (correct_statements = [1, 3, 4] → correct_answer = 3) :=
by
  sorry

end triangle_statements_correct_l224_224433


namespace grocery_store_total_bottles_l224_224824

def total_bottles (regular_soda : Nat) (diet_soda : Nat) : Nat :=
  regular_soda + diet_soda

theorem grocery_store_total_bottles :
 (total_bottles 9 8 = 17) :=
 by
   sorry

end grocery_store_total_bottles_l224_224824


namespace probability_of_log2N_is_integer_and_N_is_even_l224_224057

-- Defining the range of N as a four-digit number in base four
def is_base4_four_digit (N : ℕ) : Prop := 64 ≤ N ∧ N ≤ 255

-- Defining the condition that log_2 N is an integer
def is_power_of_two (N : ℕ) : Prop := ∃ k : ℕ, N = 2^k

-- Defining the condition that N is even
def is_even (N : ℕ) : Prop := N % 2 = 0

-- Combining all conditions
def meets_conditions (N : ℕ) : Prop := is_base4_four_digit N ∧ is_power_of_two N ∧ is_even N

-- Total number of four-digit numbers in base four
def total_base4_four_digits : ℕ := 192

-- Set of N values that meet the conditions
def valid_N_values : Finset ℕ := {64, 128}

-- The probability calculation
def calculated_probability : ℚ := valid_N_values.card / total_base4_four_digits

-- The final proof statement
theorem probability_of_log2N_is_integer_and_N_is_even : calculated_probability = 1 / 96 :=
by
  -- Prove the equality here (matching the solution given)
  sorry

end probability_of_log2N_is_integer_and_N_is_even_l224_224057


namespace balloon_arrangements_l224_224881

noncomputable def factorial (n : ℕ) : ℕ :=
if h : 0 < n then n * factorial (n - 1) else 1

theorem balloon_arrangements : 
  ∃ (n : ℕ), n = (factorial 7) / (factorial 2 * factorial 2) ∧ n = 1260 := by
  have fact_7 := factorial 7
  have fact_2 := factorial 2
  exists (fact_7 / (fact_2 * fact_2))
  split
  · rw [← int.coe_nat_eq_coe_nat_iff]
    norm_cast
    sorry -- Here we do the exact calculations
  · exact rfl

end balloon_arrangements_l224_224881


namespace units_digit_seven_pow_ten_l224_224368

theorem units_digit_seven_pow_ten : ∃ u : ℕ, (7^10) % 10 = u ∧ u = 9 :=
by
  use 9
  sorry

end units_digit_seven_pow_ten_l224_224368


namespace jean_total_cost_l224_224607

theorem jean_total_cost 
  (num_pants : ℕ)
  (original_price_per_pant : ℝ)
  (discount_rate : ℝ)
  (tax_rate : ℝ)
  (num_pants_eq : num_pants = 10)
  (original_price_per_pant_eq : original_price_per_pant = 45)
  (discount_rate_eq : discount_rate = 0.2)
  (tax_rate_eq : tax_rate = 0.1) : 
  ∃ total_cost : ℝ, total_cost = 396 :=
by
  sorry

end jean_total_cost_l224_224607


namespace domain_of_f_2x_minus_2_l224_224185

theorem domain_of_f_2x_minus_2
  (f : ℝ → ℝ)
  (h : ∀ x : ℝ, 0 ≤ x + 1 ∧ x + 1 ≤ 1 → f (x + 1) = f (real.exp (real.log 2 * x) - 2))
  : ∀ x : ℝ, real.log 2 3 ≤ x ∧ x ≤ 2 → f ((2 : ℝ) ^ x - 2) = f (x + 1) :=
by
  sorry

end domain_of_f_2x_minus_2_l224_224185


namespace set_equality_proof_l224_224548

open Set

variable (U P Q : Set ℕ) (x : ℕ)

def U := {1, 2, 3, 4, 5, 6, 7, 8}
def P := {3, 4, 5}
def Q := {1, 3, 6}

theorem set_equality_proof : (\{2, 7, 8\} : Set ℕ) = (U \ P) ∩ (U \ Q) := by
  sorry

end set_equality_proof_l224_224548


namespace three_digit_numbers_l224_224877

theorem three_digit_numbers (N : ℕ) (a b c : ℕ) 
  (h1 : N = 100 * a + 10 * b + c)
  (h2 : 1 ≤ a ∧ a ≤ 9)
  (h3 : b ≤ 9 ∧ c ≤ 9)
  (h4 : a - b + c % 11 = 0)
  (h5 : N % 11 = 0)
  (h6 : N = 11 * (a^2 + b^2 + c^2)) :
  N = 550 ∨ N = 803 :=
  sorry

end three_digit_numbers_l224_224877


namespace permutations_exclude_substrings_l224_224731

theorem permutations_exclude_substrings :
  let n_perm := 26!.natFactorial
  let sub_perms := 3 * 24!.natFactorial - 3 * 20!.natFactorial - 2 * 19!.natFactorial + 17!.natFactorial
  (n_perm - sub_perms) = 26! - 3 * 24! + 3 * 20! + 2 * 19! - 17! :=
by sorry

end permutations_exclude_substrings_l224_224731


namespace number_of_rational_terms_in_expansion_l224_224205

def binomial_expansion_rational_terms : Prop :=
  let rational_terms := finset.card (finset.filter (λ r, r % 3 = 0 ∧ (100 - r) % 2 = 0) (finset.range 101))
  rational_terms = 17

theorem number_of_rational_terms_in_expansion : binomial_expansion_rational_terms :=
by
  sorry

end number_of_rational_terms_in_expansion_l224_224205


namespace infinite_squares_of_consecutive_integers_non_perfect_square_has_infinite_solutions_l224_224296

theorem infinite_squares_of_consecutive_integers :
  ∃ (infinitely_many n : ℕ), ∃ (x k : ℤ),
    (∑ i in Finset.range n, (x + i)^2) = k^2 := sorry

theorem non_perfect_square_has_infinite_solutions (n : ℕ) (hn : ¬ ∃ m : ℕ, m * m = n)
  (x : ℤ) (hx : ∃ k : ℤ, (∑ i in Finset.range n, (x + i)^2) = k^2) :
  ∃ (infinitely_many y : ℕ), ∃ k' : ℤ,
    (∑ i in Finset.range n, (y + i)^2) = k'^2 := sorry

end infinite_squares_of_consecutive_integers_non_perfect_square_has_infinite_solutions_l224_224296


namespace find_extrema_on_interval_l224_224482

noncomputable def y (x : ℝ) := (10 * x + 10) / (x^2 + 2 * x + 2)

theorem find_extrema_on_interval :
  ∃ (min_val max_val : ℝ) (min_x max_x : ℝ), 
    min_val = 0 ∧ min_x = -1 ∧ max_val = 5 ∧ max_x = 0 ∧ 
    (∀ x ∈ Set.Icc (-1 : ℝ) 2, y x ≥ min_val) ∧
    (∀ x ∈ Set.Icc (-1 : ℝ) 2, y x ≤ max_val) :=
by
  sorry

end find_extrema_on_interval_l224_224482


namespace recurring_decimal_to_fraction_correct_l224_224760

noncomputable def recurring_decimal_to_fraction (b : ℚ) : Prop :=
  b = 0.\overline{56} ↔ b = 56/99

theorem recurring_decimal_to_fraction_correct : recurring_decimal_to_fraction 0.\overline{56} :=
  sorry

end recurring_decimal_to_fraction_correct_l224_224760


namespace equilateral_triangles_count_l224_224825

-- Define the hexagonal lattice and the condition that each point is one unit away from its closest neighbors.
def hexagonal_lattice : Type := sorry -- This would be a formal definition of the hexagonal lattice

-- Define what it means for a triangle to be equilateral in this lattice
def is_equilateral_triangle (A B C : hexagonal_lattice) : Prop :=
  (dist A B = dist B C ∧ dist B C = dist C A)

-- Specific distance definitions within the hexagonal lattice
def dist (A B : hexagonal_lattice) : ℝ := sorry -- Distance function for points in the lattice

-- Given conditions as specific hypotheses
def side_length_one_unit (A B : hexagonal_lattice) : Prop :=
  dist A B = 1

def side_length_two_units (A B : hexagonal_lattice) : Prop :=
  dist A B = 2

-- The main proof statement that we need to show
theorem equilateral_triangles_count : ∃ (n : ℕ), n = 6 :=
begin
  -- let the number of triangles with side length of 1 unit be counted
  sorry
end

end equilateral_triangles_count_l224_224825


namespace find_angle_C_l224_224154

noncomputable def angle_C_obtuse_triangle : Prop :=
  ∃ (A B C : Prop) (a b : ℝ),
    a = 4 ∧
    b = 4 * real.sqrt 3 ∧
    A = 30 ∧
    (B = 60 ∨ B = 120) ∧
    ∃ (t : Tri ABC),
      C = 180 - A - B ∧
      (B = 120 → C = 30)

theorem find_angle_C :
  ∀ (ABC : Triangle) (a b : ℝ) (A : ℝ),
    (a = 4) → 
    (b = 4 * real.sqrt 3) →
    (A = 30) → ∃ (C : ℝ), C = 30 :=
begin
  intros ABC a b A h1 h2 h3,
  sorry
end

end find_angle_C_l224_224154


namespace min_distance_to_curve_l224_224146

noncomputable def curve (x : ℝ) : ℝ := x^2 - real.log x
noncomputable def line (x : ℝ) : ℝ := x - 4
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ := real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

theorem min_distance_to_curve : ∀ P : ℝ × ℝ, P.2 = curve P.1 →
  let d := (abs (P.2 - line P.1)) / (real.sqrt (1^2 + (-1)^2)) in d = 2 * real.sqrt 2 :=
begin
  intro P,
  intro h,
  let d := abs (P.2 - line P.1) / real.sqrt (1^2 + 1^2),
  have d_eq : d = 2 * real.sqrt 2,
  { sorry },
  exact d_eq,
end

end min_distance_to_curve_l224_224146


namespace stuffed_animal_cost_l224_224443

variables 
  (M S A A_single C : ℝ)
  (Coupon_discount : ℝ)
  (Maximum_budget : ℝ)

noncomputable def conditions : Prop :=
  M = 6 ∧
  M = 3 * S ∧
  M = A / 4 ∧
  A_single = A / 2 ∧
  C = A_single / 2 ∧
  C = 2 * S ∧
  Coupon_discount = 0.10 ∧
  Maximum_budget = 30

theorem stuffed_animal_cost (h : conditions M S A A_single C Coupon_discount Maximum_budget) :
  A_single = 12 :=
sorry

end stuffed_animal_cost_l224_224443


namespace pairwise_rel_prime_subset_sum_composite_main_theorem_l224_224473

open Nat

-- Define the set of numbers
def my_set := {121, 241, 361, 481, 601}

-- Prove that every two distinct elements of the set are relatively prime
theorem pairwise_rel_prime (x y : ℕ) (hx : x ∈ my_set) (hy : y ∈ my_set) (hxy : x ≠ y) : gcd x y = 1 := sorry

-- Prove that the sum of any nonempty subset of the set is composite
theorem subset_sum_composite (s : Finset ℕ) (hs : s ⊆ my_set) (h : s.nonempty) : ¬ is_prime (s.sum id) := sorry

-- Combine the conditions to state the main theorem
theorem main_theorem : 
  let my_set := {121, 241, 361, 481, 601} in
  (∀ x y, x ∈ my_set → y ∈ my_set → x ≠ y → gcd x y = 1) ∧
  (∀ s : Finset ℕ, s ⊆ my_set → s.nonempty → ¬ is_prime (s.sum id)) := 
by
  split
  { exact pairwise_rel_prime }
  { exact subset_sum_composite }

end pairwise_rel_prime_subset_sum_composite_main_theorem_l224_224473


namespace balloon_permutation_count_l224_224917

-- Define the necessary factorials and the formula for permutations of multiset
def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem balloon_permutation_count : 
  let total_letters := 7,
      repeat_L := 2,
      repeat_O := 2,
      unique_arrangements := factorial total_letters / (factorial repeat_L * factorial repeat_O) 
  in unique_arrangements = 1260 :=
by
  sorry

end balloon_permutation_count_l224_224917


namespace sachin_borrowed_amount_l224_224294

variable (P : ℝ) (gain : ℝ)
variable (interest_rate_borrow : ℝ := 4 / 100)
variable (interest_rate_lend : ℝ := 25 / 4 / 100)
variable (time_period : ℝ := 2)
variable (gain_provided : ℝ := 112.5)

theorem sachin_borrowed_amount (h : gain = 0.0225 * P) : P = 5000 :=
by sorry

end sachin_borrowed_amount_l224_224294


namespace increasing_sequence_condition_l224_224150

theorem increasing_sequence_condition (a : ℕ → ℝ) (λ : ℝ) (h : ∀ n : ℕ, n > 0 → a n = n^2 + λ * n) :
  (∀ n : ℕ, n > 0 → a n < a (n + 1)) ↔ λ > -3 :=
by
  intros
  sorry

end increasing_sequence_condition_l224_224150


namespace find_x12_value_l224_224086

/-- Given the equation log_{10x^2} 10 + log_{1000x^4} 10 = -3, 
    prove that the largest solution x to the equation satisfies 1 / x^12 = 10^7.3. -/
theorem find_x12_value :
  ∃ x : ℝ, (real.log 10 / real.log (10 * x^2) + real.log 10 / real.log (1000 * x^4) = -3) ∧
    (1 / x^12 = 10^7.3) :=
begin
  sorry
end

end find_x12_value_l224_224086


namespace repeating_decimal_to_fraction_l224_224775

theorem repeating_decimal_to_fraction : (0.5656565656 : ℚ) = 56 / 99 :=
  by
    sorry

end repeating_decimal_to_fraction_l224_224775


namespace arithmetic_sequence_solution_l224_224578

-- Definitions based on the given conditions
def is_arithmetic_sequence (a A b : ℝ) : Prop := 2 * A = a + b
def frac_part (x : ℝ) : ℝ := x - floor x
def floor_part (x : ℝ) : ℝ := floor x

-- Theorem to be proved
theorem arithmetic_sequence_solution (x : ℝ) (h1 : 0 ≤ frac_part x) (h2 : frac_part x < 1) (h3 : 0 < x) :
  (∃ (_, _, z : ℝ), is_arithmetic_sequence (frac_part x) (floor_part x) x ∧ z = x) → x = 3 / 2 :=
by 
  sorry

end arithmetic_sequence_solution_l224_224578


namespace eight_digit_palindromes_count_l224_224010

theorem eight_digit_palindromes_count : 
  ∃ (count : ℕ), count = 81 ∧ 
    (∀ (d1 d2 d3 d4 : ℕ), 
      d1 ∈ {1, 2, 3} ∧ 
      d2 ∈ {1, 2, 3} ∧ 
      d3 ∈ {1, 2, 3} ∧ 
      d4 ∈ {1, 2, 3} → 
      (d1 ≠ 0) ∧ (d2 ≠ 0) ∧ (d3 ≠ 0) ∧ (d4 ≠ 0) →
      (3^4 = count)) :=
by
  use 81
  sorry

end eight_digit_palindromes_count_l224_224010


namespace anglet_angle_measurement_l224_224435

-- Definitions based on conditions
def anglet_measurement := 1
def sixth_circle_degrees := 360 / 6
def anglets_in_sixth_circle := 6000

-- Lean theorem statement proving the implied angle measurement
theorem anglet_angle_measurement (one_percent : Real := 0.01) :
  (anglets_in_sixth_circle * one_percent * sixth_circle_degrees) = anglet_measurement * 60 := 
  sorry

end anglet_angle_measurement_l224_224435


namespace concurrency_of_lines_l224_224140

-- Definitions of the theorem based on conditions
variables {A₁ A₂ A₃ : Type} [EuclideanGeometry A₁] [EuclideanGeometry A₂] [EuclideanGeometry A₃]
  {a₁ a₂ a₃ : Real} -- side lengths of the triangle
  {M₁ M₂ M₃ T₁ T₂ T₃ I : Type}
  [incircle I A₁ A₂ A₃]
  [midpoint M₁ A₂ A₃]
  [midpoint M₂ A₁ A₃]
  [midpoint M₃ A₁ A₂]
  [tangency_point T₁ I A₂ A₃]
  [tangency_point T₂ I A₁ A₃]
  [tangency_point T₃ I A₁ A₂]
  [reflection S₁ T₁ A₁]
  [reflection S₂ T₂ A₂]
  [reflection S₃ T₃ A₃]

-- The theorem to prove based on the proof problem
theorem concurrency_of_lines :
  are_concurrent M₁ S₁ M₂ S₂ M₃ S₃ :=
sorry

end concurrency_of_lines_l224_224140


namespace product_of_first_16_not_multiple_of_51_l224_224575

theorem product_of_first_16_not_multiple_of_51 : 
  ¬ (∃ k : ℕ, (∏ i in Finset.range 17, (i + 1)) = 51 * k) :=
by
  sorry

end product_of_first_16_not_multiple_of_51_l224_224575


namespace complex_fraction_expression_equals_half_l224_224787

theorem complex_fraction_expression_equals_half :
  ((2 / (3 + 1/5)) + (((3 + 1/4) / 13) / (2 / 3)) + (((2 + 5/18) - (17/36)) * (18 / 65))) * (1 / 3) = 0.5 :=
by
  sorry

end complex_fraction_expression_equals_half_l224_224787


namespace tire_usage_correct_l224_224044

-- Given conditions
def car : Type := { tires : Nat // tires = 5 }
def in_use_tires : Nat := 4
def total_distance : Nat := 50000
def tire_usage : Nat := 40000

-- Question: Prove that each tire was used for 40000 miles given the conditions
theorem tire_usage_correct (c : car):
  (total_distance * in_use_tires) / c.tires.val = tire_usage :=
by {
  -- Here, the proof would go
  sorry
}

end tire_usage_correct_l224_224044


namespace balloon_arrangements_l224_224898

theorem balloon_arrangements :
  let n := 7
  let k1 := 2
  let k2 := 2
  (Nat.factorial n) / ((Nat.factorial k1) * (Nat.factorial k2)) = 1260 :=
by
  -- Use the conditions directly from the problem
  let n := 7
  let k1 := 2
  let k2 := 2
  -- Mathematically, the number of arrangements of the multiset
  have h : (Nat.factorial n) / ((Nat.factorial k1) * (Nat.factorial k2)) = 1260 := sorry
  exact h

end balloon_arrangements_l224_224898


namespace hyperbola_standard_eq_l224_224153

theorem hyperbola_standard_eq (a b : ℝ) :
  (∀ x y : ℝ, ((x = 4) ∧ (y = sqrt 3) → (x^2 / a^2 - y^2 / b^2 = 1))) ∧
  (b = a / 2) ↔ (a = 2 ∧ b = 1) :=
by
  sorry

end hyperbola_standard_eq_l224_224153


namespace who_spoke_truth_l224_224308

namespace ExamTruthTellers

-- Define the boolean variables for each student's truthfulness
variables (A_truth B_truth C_truth D_truth : Prop)

-- Conditions based on what each student said
def A_statement : Prop := ¬∃ x, [A_truth, B_truth, C_truth, D_truth].nth x = some true
def B_statement : Prop := ∃ x, [A_truth, B_truth, C_truth, D_truth].nth x = some true
def C_statement : Prop := ¬(B_truth ∧ D_truth)
def D_statement : Prop := ¬D_truth

-- Given conditions
axiom A_condition : A_truth ↔ A_statement
axiom B_condition : B_truth ↔ B_statement
axiom C_condition : C_truth ↔ C_statement
axiom D_condition : D_truth ↔ D_statement
axiom exactly_two_spoke_truth : (A_truth ↔ true) + (B_truth ↔ true) + (C_truth ↔ true) + (D_truth ↔ true) = 2

theorem who_spoke_truth : (B_truth = true) ∧ (C_truth = true) :=
by sorry

end ExamTruthTellers

end who_spoke_truth_l224_224308


namespace sqrt_expression_eq_twelve_l224_224863

theorem sqrt_expression_eq_twelve : Real.sqrt 3 * (Real.sqrt 3 + Real.sqrt 27) = 12 := 
sorry

end sqrt_expression_eq_twelve_l224_224863


namespace triangle_angles_l224_224094

noncomputable def cos_60 : Float := Math.cos (Float.pi / 3)
noncomputable def cos_75 : Float := Math.cos (5 * Float.pi / 12)
noncomputable def cos_45 : Float := Math.cos (Float.pi / 4)

theorem triangle_angles (a b c : ℝ) (a_pos : 0 < a) (b_pos : 0 < b) (c_pos : 0 < c)
  (triangle_ineq1 : a + b > c) (triangle_ineq2 : a + c > b) (triangle_ineq3 : b + c > a) :
  ∃ (angle_A angle_B angle_C : ℝ),
    (angle_A = 60 ∨ angle_A = 75 ∨ angle_A = 45) ∧
    (angle_B = 60 ∨ angle_B = 75 ∨ angle_B = 45) ∧
    (angle_C = 60 ∨ angle_C = 75 ∨ angle_C = 45) ∧
    angle_A + angle_B + angle_C = 180 :=
begin
  let a := 3,
  let b := real.sqrt 11,
  let c := 2 + real.sqrt 5,
  have a_pos : 0 < a := by norm_num,
  have b_pos : 0 < b := by { unfold real.sqrt, exact real.sqrt_pos.2 (by norm_num) },
  have c_pos : 0 < c := by { linarith [real.sqrt_nonneg 5] },
  have triangle_ineq1 : a + b > c, by linarith [sqrt_positivity _],
  have triangle_ineq2 : a + c > b, by linarith,
  have triangle_ineq3 : b + c > a, by linarith,
  existsi [60, 75, 45],
  split, { norm_num, },
  split, { norm_num, },
  split, { norm_num, },
  norm_num,
  sorry
end

end triangle_angles_l224_224094


namespace Kolya_is_acting_as_collection_agency_l224_224214

-- Definitions for the conditions given in the problem
def Katya_lent_books_to_Vasya : Prop := ∃ books : Type, ∀ b : books, ¬ b ∈ Katya's_collection
def Vasya_promised_to_return_books_in_a_month_but_failed : Prop := ∀ t : Time, t ≥ 1 month → ¬returned books by Vasya
def Katya_asked_Kolya_to_get_books_back : Prop := ∀ k : Kolya, ∀ v : Vasya, asked Katya (k to get books back from v)
def Kolya_agrees_but_wants_a_reward : Prop := ∃ reward : Book, Kolya_gets reward

-- Define the property of Kolya being a collection agency
def Kolya_is_collection_agency : Prop :=
  Katya_lent_books_to_Vasya ∧
  Vasya_promised_to_return_books_in_a_month_but_failed ∧
  Katya_asked_Kolya_to_get_books_back ∧
  Kolya_agrees_but_wants_a_reward

-- The theorem to prove
theorem Kolya_is_acting_as_collection_agency :
  Kolya_is_collection_agency :=
sorry

end Kolya_is_acting_as_collection_agency_l224_224214


namespace power_identity_l224_224176

variable (a : ℝ) (m n : ℝ)

-- Given conditions
def log_a_2 (a : ℝ) (m : ℝ) := log a 2 = m
def log_3_a (a : ℝ) (n : ℝ) := log 3 a = 1 / n

theorem power_identity (h1 : log_a_2 a m) (h2 : log_3_a a n) : a^(m + 2 * n) = 18 :=
by
  sorry

end power_identity_l224_224176


namespace purchasing_methods_count_l224_224414

def material_cost : ℕ := 40
def instrument_cost : ℕ := 60
def budget : ℕ := 400
def min_materials : ℕ := 4
def min_instruments : ℕ := 2

theorem purchasing_methods_count : 
  (∃ (n_m m : ℕ), 
    n_m ≥ min_materials ∧ m ≥ min_instruments ∧ 
    n_m * material_cost + m * instrument_cost ≤ budget) → 
  (∃ (count : ℕ), count = 7) :=
by 
  sorry

end purchasing_methods_count_l224_224414


namespace angle_in_third_quadrant_l224_224567

theorem angle_in_third_quadrant (α : ℝ) (k : ℤ) :
  (2 * ↑k * Real.pi + Real.pi < α ∧ α < 2 * ↑k * Real.pi + 3 * Real.pi / 2) →
  (∃ (m : ℤ), (0 < α / 3 + m * 2 * Real.pi ∧ α / 3 + m * 2 * Real.pi < Real.pi ∨
                π < α / 3 + m * 2 * Real.pi ∧ α / 3 + m * 2 * Real.pi < 3 * Real.pi / 2 ∨ 
                -π < α / 3 + m * 2 * Real.pi ∧ α / 3 + m * 2 * Real.pi < 0)) :=
by
  sorry

end angle_in_third_quadrant_l224_224567


namespace repeating_decimal_eq_l224_224771

noncomputable def repeating_decimal : ℚ := 56 / 99

theorem repeating_decimal_eq : ∃ x : ℚ, x = repeating_decimal ∧ x = 56 / 99 :=
by
  use 56 / 99
  split
  all_goals { sorry }

end repeating_decimal_eq_l224_224771


namespace triangle_a_eq_2_l224_224523

variable {α β γ : ℝ}
variable {a b c : ℝ}
variable {S : ℝ}

-- Conditions
def is_acute_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 > c^2 ∧ a^2 + c^2 > b^2 ∧ b^2 + c^2 > a^2

def cosine_rule (a b c : ℝ) : Prop :=
  b^2 = a^2 + c^2 - a * c

-- Proof problem for part 1:
theorem triangle_a_eq_2
  (h1 : ∠A = 60 * (Math.pi / 180)) -- angle A = 60 degrees
  (h2 : c = 2)
  (h3 : cosine_rule a b c) :
  a = 2 :=
sorry

-- Here ∠ represents the angle function which in practice, you would define according to your specific setup

end triangle_a_eq_2_l224_224523


namespace certain_fraction_ratio_l224_224479

theorem certain_fraction_ratio :
  ∃ x : ℚ,
    (2 / 5 : ℚ) / x = (0.46666666666666673 : ℚ) / (1 / 2) ∧ x = 3 / 7 :=
by sorry

end certain_fraction_ratio_l224_224479


namespace repeating_decimal_eq_l224_224768

noncomputable def repeating_decimal : ℚ := 56 / 99

theorem repeating_decimal_eq : ∃ x : ℚ, x = repeating_decimal ∧ x = 56 / 99 :=
by
  use 56 / 99
  split
  all_goals { sorry }

end repeating_decimal_eq_l224_224768


namespace balloon_arrangements_l224_224887

noncomputable def factorial (n : ℕ) : ℕ :=
if h : 0 < n then n * factorial (n - 1) else 1

theorem balloon_arrangements : 
  ∃ (n : ℕ), n = (factorial 7) / (factorial 2 * factorial 2) ∧ n = 1260 := by
  have fact_7 := factorial 7
  have fact_2 := factorial 2
  exists (fact_7 / (fact_2 * fact_2))
  split
  · rw [← int.coe_nat_eq_coe_nat_iff]
    norm_cast
    sorry -- Here we do the exact calculations
  · exact rfl

end balloon_arrangements_l224_224887


namespace weight_of_7th_person_correct_l224_224344

noncomputable def weight_of_6_people (avg_weight : ℕ) (n : ℕ) : ℕ := avg_weight * n

noncomputable def weight_of_7th_person 
  (total_weight_with_6 : ℕ) (new_avg_weight : ℕ) (n : ℕ) : ℕ := 
  (new_avg_weight * (n + 1)) - total_weight_with_6

theorem weight_of_7th_person_correct :
  ∀ (avg_weight_6 : ℕ) (n : ℕ) (new_avg_weight : ℕ),
    avg_weight_6 = 156 →
    n = 6 →
    new_avg_weight = 151 →
    weight_of_7th_person (weight_of_6_people avg_weight_6 n) new_avg_weight n = 121 :=
by {
  intros,
  sorry
}

end weight_of_7th_person_correct_l224_224344


namespace sqrt_rational_rational_l224_224229

theorem sqrt_rational_rational 
  (a b : ℚ) 
  (h : ∃ r : ℚ, r = (a : ℝ).sqrt + (b : ℝ).sqrt) : 
  (∃ p : ℚ, p = (a : ℝ).sqrt) ∧ (∃ q : ℚ, q = (b : ℝ).sqrt) := 
sorry

end sqrt_rational_rational_l224_224229


namespace triangle_solution_l224_224222

theorem triangle_solution (A B C M : Type) (AB AC AM BC : ℝ) (x : ℝ)
  (h1 : AB = 5)
  (h2 : AC = 7)
  (h3 : AM = 4)
  (h4 : M = (B + C) / 2)
  (h5 : BC = 2 * x)
  (h6 : 2 * x * (x * x - 21) = 0):
  BC = 2 * Real.sqrt 21 := sorry

end triangle_solution_l224_224222


namespace balloon_arrangements_l224_224987

def factorial (n : Nat) : Nat :=
  if n = 0 then 1 else n * factorial (n - 1)

def arrangements (n : Nat) (ks : List Nat) :=
  factorial n / (ks.map factorial).foldr (*) 1

theorem balloon_arrangements : arrangements 7 [2, 2] = 1260 :=
by
  sorry

end balloon_arrangements_l224_224987


namespace evaluate_expression_l224_224994

theorem evaluate_expression :
  (⌈(23 / 9) - ⌈35 / 23⌉⌉ / ⌈(35 / 9) + ⌈(9 * 23) / 35⌉⌉) = 1 / 10 :=
by sorry

end evaluate_expression_l224_224994


namespace balloon_permutation_count_l224_224932

-- Define the given conditions:
def balloon_letters : List Char := ['B', 'A', 'L', 'L', 'O', 'O', 'N']
def total_letters : Nat := 7
def count_L : Nat := 2
def count_O : Nat := 2

-- Statement to prove:
theorem balloon_permutation_count :
  (nat.factorial total_letters) / ((nat.factorial count_L) * (nat.factorial count_O)) = 1260 := by
  sorry

end balloon_permutation_count_l224_224932


namespace freshmen_sophomores_without_pets_l224_224343

theorem freshmen_sophomores_without_pets : 
  let total_students := 400
  let percentage_freshmen_sophomores := 0.50
  let percentage_with_pets := 1/5
  let freshmen_sophomores := percentage_freshmen_sophomores * total_students
  160 = (freshmen_sophomores - (percentage_with_pets * freshmen_sophomores)) :=
by
  sorry

end freshmen_sophomores_without_pets_l224_224343


namespace simplify_trig_expression_l224_224672

theorem simplify_trig_expression :
  (∀ θ : ℝ, ∀ φ : ℝ, tan θ - real.sqrt 3 = (sin θ - real.sqrt 3 * cos θ) / cos θ) →
  (∀ θ : ℝ, ∀ φ : ℝ, (sin θ - real.sqrt 3 * cos θ) / cos θ = 2 * sin (θ - 60°) / cos θ) →
  (∀ θ : ℝ, ∀ φ : ℝ, 2 * sin (θ - 60°) / cos θ = -2 * sin 48° / cos θ) →
  (∀ θ : ℝ, ∀ φ : ℝ, -2 * sin 48° / cos θ = -8 * sin θ * cos 24°) →
  (∀ θ : ℝ, (tan 12° - real.sqrt 3) / (sin 12° * cos 24°) = -8) :=
by
  intros h1 h2 h3 h4 h5
  sorry

end simplify_trig_expression_l224_224672


namespace tangency_splits_segments_l224_224078

def pentagon_lengths (a b c d e : ℕ) (h₁ : a = 1) (h₃ : c = 1) (x1 x2 : ℝ) :=
x1 + x2 = b ∧ x1 = 1/2 ∧ x2 = 1/2

theorem tangency_splits_segments {a b c d e : ℕ} (h₁ : a = 1) (h₃ : c = 1) :
    ∃ x1 x2 : ℝ, pentagon_lengths a b c d e h₁ h₃ x1 x2 :=
    by 
    sorry

end tangency_splits_segments_l224_224078


namespace imaginary_part_of_ratio_l224_224532

noncomputable def z1 := 1 - (2 : ℂ) * complex.I
noncomputable def z2 := -1 - (2 : ℂ) * complex.I

theorem imaginary_part_of_ratio :
  complex.im (z2 / z1) = -(4 / 5 : ℝ) :=
by sorry

end imaginary_part_of_ratio_l224_224532


namespace plot_length_l224_224331

variables (b l : ℝ)
constants (c_fence_metal : ℝ) (c_fence_wood : ℝ) (c_gate : ℝ) (cost_total : ℝ)
constants (extra_length : ℝ) (gate_length : ℝ)

def plot_conditions :=
  extra_length = 14 ∧
  c_fence_metal = 26.5 ∧
  c_fence_wood = 22 ∧
  c_gate = 240 ∧
  cost_total = 5600 ∧
  gate_length = 2 ∧
  l = b + extra_length ∧
  ((2 * l + b) * c_fence_metal + b * c_fence_wood + c_gate = cost_total)

theorem plot_length :
  plot_conditions b l →
  l = 59.5 :=
begin
  sorry
end

end plot_length_l224_224331


namespace distance_between_points_l224_224114

open Real

theorem distance_between_points :
  ∀ (x1 y1 x2 y2 : ℝ),
  (x1, y1) = (-3, 1) →
  (x2, y2) = (5, -5) →
  sqrt ((x2 - x1) ^ 2 + (y2 - y1) ^ 2) = 10 :=
by
  intros x1 y1 x2 y2 h1 h2
  sorry

end distance_between_points_l224_224114


namespace decreasing_interval_l224_224684

theorem decreasing_interval (x : ℝ) (h1 : 0 < x) : 
  2x^2 - log x ∈ set.Ioc 0 (1 / 2) := sorry

end decreasing_interval_l224_224684


namespace find_x_when_y_neg_five_l224_224676

-- Definitions based on the conditions provided
variable (x y : ℝ)
def inversely_proportional (x y : ℝ) := ∃ (k : ℝ), x * y = k

-- Proving the main result
theorem find_x_when_y_neg_five (h_prop : inversely_proportional x y) (hx4 : x = 4) (hy2 : y = 2) :
    (y = -5) → x = - 8 / 5 := by
  sorry

end find_x_when_y_neg_five_l224_224676


namespace dandelions_surviving_to_flower_l224_224102

/-- 
Each dandelion produces 300 seeds. 
1/3rd of the seeds land in water and die. 
1/6 of the starting number are eaten by insects. 
Half the remainder sprout and are immediately eaten.
-/
def starting_seeds : ℕ := 300

def seeds_lost_to_water : ℕ := starting_seeds / 3
def seeds_after_water : ℕ := starting_seeds - seeds_lost_to_water

def seeds_eaten_by_insects : ℕ := starting_seeds / 6
def seeds_after_insects : ℕ := seeds_after_water - seeds_eaten_by_insects

def seeds_eaten_after_sprouting : ℕ := seeds_after_insects / 2
def seeds_surviving : ℕ := seeds_after_insects - seeds_eaten_after_sprouting

theorem dandelions_surviving_to_flower 
  (starting_seeds = 300) 
  (seeds_lost_to_water = starting_seeds / 3) 
  (seeds_after_water = starting_seeds - seeds_lost_to_water) 
  (seeds_eaten_by_insects = starting_seeds / 6) 
  (seeds_after_insects = seeds_after_water - seeds_eaten_by_insects) 
  (seeds_eaten_after_sprouting = seeds_after_insects / 2) 
  (seeds_surviving = seeds_after_insects - seeds_eaten_after_sprouting) : 
  seeds_surviving = 75 := 
sorry

end dandelions_surviving_to_flower_l224_224102


namespace minimum_h18_l224_224625

noncomputable def h := ℕ → ℕ

def tenuous (h : h) : Prop :=
  ∀ (x y : ℕ), x > 0 ∧ y > 0 → h x + h y > x^2

def minimized_sum (h : h) : Prop :=
  h 1 + h 2 + ... + h 30 = 15355

-- The proposition we need to prove
theorem minimum_h18 (h : h) (tenuous_h : tenuous h) (min_sum_h: minimized_sum h) :
  h 18 = 196 :=
sorry

end minimum_h18_l224_224625


namespace rate_is_900_l224_224692

noncomputable def rate_per_square_meter (L W : ℝ) (total_cost : ℝ) : ℝ :=
  total_cost / (L * W)

theorem rate_is_900 :
  rate_per_square_meter 5 4.75 21375 = 900 := by
  sorry

end rate_is_900_l224_224692


namespace product_of_seven_consecutive_integers_not_ends_with_exactly_one_zero_factorial_57_not_ends_with_12_zeros_l224_224389

-- Part (a)
theorem product_of_seven_consecutive_integers_not_ends_with_exactly_one_zero : 
  ∀ a : ℤ, let s := list.range 7 |>.map (λ i, a + i) in 
  (list.foldr (*) 1 s % 10 ≠ 0) :=
by
  sorry

-- Part (b)
theorem factorial_57_not_ends_with_12_zeros :
  let n := 57 in
  (list.foldr (*) 1 (list.range' 1 n) % 10 ^ 12 ≠ 0) :=
by
  sorry

end product_of_seven_consecutive_integers_not_ends_with_exactly_one_zero_factorial_57_not_ends_with_12_zeros_l224_224389


namespace balloon_permutations_l224_224949

theorem balloon_permutations : 
  let total_letters := 7
  let repetitions_L := 2
  let repetitions_O := 2
  (Nat.factorial total_letters) / ((Nat.factorial repetitions_L) * (Nat.factorial repetitions_O)) = 1260 := 
by
  sorry

end balloon_permutations_l224_224949


namespace region_eq_condition_l224_224876

theorem region_eq_condition (a : ℝ) (h_a : a > 0) :
  (∃ f : ℝ → ℝ, continuous_on f (set.Icc 0 a) ∧ (∀ x : ℝ, 0 ≤ x → x ≤ a → 0 ≤ f x) ∧ 
              ((∃ k : ℝ, k > 0 ∧ 
            (∫ x in 0..a, f x) = k ∧ 
            (a + 2 * f 0 + 2 * f a + ∫ x in 0..a, sqrt (1 + (deriv f x)^2)) = k))) ↔ 
  a > 2 :=
begin
  sorry
end

end region_eq_condition_l224_224876


namespace angle_bisector_perpendicular_to_MR_l224_224631

variables {A B C P M R E : Type}

-- Definitions of points and conditions
def is_midpoint (M : Type) (X Y : Type) : Prop := sorry
def is_inside_triangle (P : Type) (A B C : Type) : Prop := sorry
def line_intersects (line1 line2 : Type) (E : Type) : Prop := sorry
def equal_segments (seg1 seg2 : Type) : Prop := sorry
def perpendicular (line1 line2 : Type) : Prop := sorry
def angle_bisector (angle : Type) (line : Type) : Prop := sorry

-- Problem statement based on given conditions
theorem angle_bisector_perpendicular_to_MR
  (P_inside_triangle : is_inside_triangle P A B C)
  (BP_eq_AC : equal_segments (B, P) (A, C))
  (M_is_midpoint : is_midpoint M A P)
  (R_is_midpoint : is_midpoint R B C)
  (BP_intersects_AC_at_E : line_intersects (B, P) (A, C) E) :
  perpendicular (angle_bisector (B, E, A) (E)) (M, R) :=
sorry

end angle_bisector_perpendicular_to_MR_l224_224631


namespace black_tiles_needed_l224_224730

noncomputable def num_black_tiles (total_white: ℕ) : ℕ :=
  let n := total_white / 4 + 1 in
  (n - 2) * (n - 2)

theorem black_tiles_needed (total_white : ℕ) (htotal : total_white = 80) : 
  num_black_tiles total_white = 361 := 
by 
  rw [htotal]
  unfold num_black_tiles
  sorry

end black_tiles_needed_l224_224730


namespace desired_ratio_of_zinc_to_copper_l224_224849

noncomputable def zinc_copper_ratio (initial_zinc_ratio : ℚ) (initial_copper_ratio : ℚ)
  (total_weight : ℚ) (added_zinc : ℚ) : (ℚ × ℚ) :=
  let initial_zinc_weight := (initial_zinc_ratio / (initial_zinc_ratio + initial_copper_ratio)) * total_weight in
  let initial_copper_weight := (initial_copper_ratio / (initial_zinc_ratio + initial_copper_ratio)) * total_weight in
  let new_zinc_weight := initial_zinc_weight + added_zinc in
  let new_copper_weight := initial_copper_weight in
  let ratio := new_zinc_weight / new_copper_weight in
  (47, 9) -- simplified to whole numbers

theorem desired_ratio_of_zinc_to_copper :
  zinc_copper_ratio 5 3 6 8 = (47, 9) :=
by
  unfold zinc_copper_ratio
  sorry

end desired_ratio_of_zinc_to_copper_l224_224849


namespace repeating_decimal_eq_l224_224770

noncomputable def repeating_decimal : ℚ := 56 / 99

theorem repeating_decimal_eq : ∃ x : ℚ, x = repeating_decimal ∧ x = 56 / 99 :=
by
  use 56 / 99
  split
  all_goals { sorry }

end repeating_decimal_eq_l224_224770


namespace average_price_of_towels_l224_224064

-- Definitions based on the conditions
def cost_of_three_towels := 3 * 100
def cost_of_five_towels := 5 * 150
def cost_of_two_towels := 550
def total_cost := cost_of_three_towels + cost_of_five_towels + cost_of_two_towels
def total_number_of_towels := 3 + 5 + 2
def average_price := total_cost / total_number_of_towels

-- The theorem statement
theorem average_price_of_towels :
  average_price = 160 :=
by
  sorry

end average_price_of_towels_l224_224064


namespace correct_system_equations_l224_224035

theorem correct_system_equations (x y : ℤ) : 
  (8 * x - y = 3) ∧ (y - 7 * x = 4) ↔ 
    (8 * x - y = 3) ∧ (y - 7 * x = 4) := by
  sorry

end correct_system_equations_l224_224035


namespace pentagonal_pyramid_probability_l224_224411

def vertices : Finset ℕ := {1, 2, 3, 4, 5, 6}

theorem pentagonal_pyramid_probability :
  (let TP := (vertices.card.choose 3) in
   let NP := ((Finset.card (vertices.erase 6)).choose 3) + 5 in
   let intersecting_case_prob := 1 - (NP / TP : ℚ) in
   intersecting_case_prob = 1 / 4
  ) :=
begin
  sorry
end

end pentagonal_pyramid_probability_l224_224411


namespace quarters_sister_gave_l224_224608

theorem quarters_sister_gave (original_quarters total_quarters_now quarters_given: ℕ) 
  (h₀ : original_quarters = 8)
  (h₁ : total_quarters_now = 11)
  (h₂ : total_quarters_now = original_quarters + quarters_given) :
  quarters_given = 3 :=
by
  rw [h₀, h₁, ←nat.add_sub_assoc (by decide)],  -- using Lean's nat.add_sub_assoc for natural numbers
  assumption

end quarters_sister_gave_l224_224608


namespace find_pairs_l224_224545

noncomputable def sequence_a (n : ℕ) : ℤ :=
if n = 1 then -2 else (-2)^n

def sum_seq (n : ℕ) : ℤ :=
(n + 1) * sequence_a 1

axiom condition_a : ∀ (n : ℕ), sequence_a (n + 1) + 3 * sum_seq n + 2 = 0

lemma general_term :
  ∀ (n : ℕ), sequence_a n = (-2)^n :=
begin
  sorry -- proof should go here
end

theorem find_pairs :
  ∃ (m n : ℤ), sequence_a n ^ 2 - m * sequence_a n - 4 * m - 8 = 0 :=
begin
  use [(-2, 1)],
  use [(1, 2)],
  use [(-14, 3)],
  repeat { split },
  all_goals { sorry }, -- proof should go here
end

end find_pairs_l224_224545


namespace find_t_u_l224_224551

noncomputable theory
open_locale big_operators

variables {𝕜 : Type*} [normed_field 𝕜] [normed_space ℝ 𝕜]
variables (a b : 𝕜) (p : 𝕜)
variables (t u : ℝ)

theorem find_t_u
  (h : ∥p - b∥ = 3 * ∥p - a∥) :
  (t = 9/8) ∧ (u = -1/8) :=
sorry

end find_t_u_l224_224551


namespace percent_cities_less_than_50000_l224_224315

-- Definitions of the conditions
def percent_cities_50000_to_149999 := 40
def percent_cities_less_than_10000 := 35
def percent_cities_10000_to_49999 := 10
def percent_cities_150000_or_more := 15

-- Prove that the total percentage of cities with fewer than 50,000 residents is 45%
theorem percent_cities_less_than_50000 :
  percent_cities_less_than_10000 + percent_cities_10000_to_49999 = 45 :=
by
  sorry

end percent_cities_less_than_50000_l224_224315


namespace maximum_k_inequality_l224_224483

open Real

noncomputable def inequality_problem (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) : Prop :=
  (x / sqrt (y + z)) + (y / sqrt (z + x)) + (z / sqrt (x + y)) ≥ sqrt (3 / 2) * sqrt (x + y + z)
 
-- This is the theorem statement
theorem maximum_k_inequality (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) : 
  inequality_problem x y z h1 h2 h3 :=
  sorry

end maximum_k_inequality_l224_224483


namespace num_distinct_integers_expressed_l224_224561

def f (x : ℝ) : ℤ :=
  Int.floor (3 * x) + Int.floor (5 * x) + Int.floor (7 * x) + Int.floor (9 * x)

theorem num_distinct_integers_expressed (n : ℕ) (h : 1 ≤ n ∧ n ≤ 500) :
  {i : ℤ | ∃ (x : ℝ), 0 < x ∧ x ≤ (500 : ℝ) ∧ f x = i}.finite :=
begin
  -- Proof to be filled in
  sorry
end

end num_distinct_integers_expressed_l224_224561


namespace cone_volume_in_cylinder_l224_224393

theorem cone_volume_in_cylinder (R : ℝ) (hR : 0 < R) :
  let P := 3 / 4 * R,
      α := { p : ℝ × ℝ × ℝ | p.2 = P },
      V := 128 / 375 * π * R^3 in
  ∃ (cylinder : ℝ) (cone : ℝ × ℝ × ℝ × ℝ),
    cylinder = R ∧
    ∃ (inscribed_sphere : ℝ),
      inscribed_sphere = R ∧
      cone = (R, P, P, P) ∧ -- simplified representation of the cone's geometric description
      V = 128 / 375 * π * R^3 :=
  sorry

end cone_volume_in_cylinder_l224_224393


namespace couple_arrangement_is_48_l224_224351

-- Define the number of ways to arrange three couples standing next to each other in a row
def couple_arrangements : ℕ := 3! * 2 * 2 * 2

-- State the theorem proving the number of arrangements is 48
theorem couple_arrangement_is_48 : couple_arrangements = 48 := by
  sorry

end couple_arrangement_is_48_l224_224351


namespace problem_1_problem_2_problem_3_l224_224810

theorem problem_1 (a x : ℝ) (h : -2 ≤ a ∧ a ≤ 1) : |x - 1| + |x - 3| ≥ a^2 + a :=
sorry

theorem problem_2 (a b : ℝ) (h1 : a + b = 1) (h2 : ∀ b > 0, min (a / (4 * |b|) + (b / a) + (1 / 4)) (a = 2) = 1) :
  min (a / (4 * |b|) + (b / a) - (1 / 4)) = 3 / 4 :=
sorry

theorem problem_3 (a : ℝ) (h : 2 ≤ a) : (0 < (2 * a) / (a^2 + 1) ∧ (2 * a) / (a^2 + 1) ≤ 4 / 5) :=
sorry

end problem_1_problem_2_problem_3_l224_224810


namespace total_students_is_46_l224_224192

-- Define the constants for the problem
def students_in_history : ℕ := 19
def students_in_math : ℕ := 14
def students_in_english : ℕ := 26
def students_in_all_three : ℕ := 3
def students_in_exactly_two : ℕ := 7

-- The total number of students as per the inclusion-exclusion principle
def total_students : ℕ :=
  students_in_history + students_in_math + students_in_english
  - students_in_exactly_two - 2 * students_in_all_three + students_in_all_three

theorem total_students_is_46 : total_students = 46 :=
  sorry

end total_students_is_46_l224_224192


namespace box_upper_surface_area_l224_224680

theorem box_upper_surface_area (L W H : ℕ) 
    (h1 : L * W = 120) 
    (h2 : L * H = 72) 
    (h3 : L * W * H = 720) : 
    L * W = 120 := 
by 
  sorry

end box_upper_surface_area_l224_224680


namespace interest_ratio_l224_224709

noncomputable def simple_interest (P r t : ℝ) : ℝ := (P * r * t) / 100
noncomputable def compound_interest (P r t : ℝ) : ℝ := let A := P * (1 + r / 100) ^ t in A - P

theorem interest_ratio :
  simple_interest 1750 8 3 / compound_interest 4000 10 2 = 1 / 2 :=
by
  sorry

end interest_ratio_l224_224709


namespace semicircle_radius_in_trapezoid_l224_224590

theorem semicircle_radius_in_trapezoid 
  (AB CD : ℝ) (AD BC : ℝ) (r : ℝ)
  (h1 : AB = 27) 
  (h2 : CD = 45) 
  (h3 : AD = 13) 
  (h4 : BC = 15) 
  (h5 : r = 13.5) :
  r = 13.5 :=
by
  sorry  -- Detailed proof steps will go here

end semicircle_radius_in_trapezoid_l224_224590


namespace total_number_of_guests_l224_224820

theorem total_number_of_guests (A C S : ℕ) (hA : A = 58) (hC : C = A - 35) (hS : S = 2 * C) : 
  A + C + S = 127 := 
by
  sorry

end total_number_of_guests_l224_224820


namespace belle_stickers_l224_224079

theorem belle_stickers (c_stickers : ℕ) (diff : ℕ) (b_stickers : ℕ) (h1 : c_stickers = 79) (h2 : diff = 18) (h3 : c_stickers = b_stickers - diff) : b_stickers = 97 := 
by
  sorry

end belle_stickers_l224_224079


namespace construct_triangle_l224_224870

theorem construct_triangle (h_b h_c m_a : ℝ) : 
  ∃ (A B C A1 : ℝ × ℝ), 
    ((A1.1 - B.1) * (A1.1 - C.1) + (A1.2 - B.2) * (A1.2 - C.2) = 0) ∧
    ((A1.1 - C.1) * (A1.1 - A.1) + (A1.2 - C.2) * (A1.2 - A.2) = 0) ∧
    ((A1.1 - B.1) * (A1.1 - A.1) + (A1.2 - B.2) * (A1.2 - A.2) = 0) ∧
    (dist (A1.1, A1.2) (A2.1, A2.2) = √((A1.1 - A2.1) ^ 2 + (A1.2 - A2.2) ^ 2)) ∧
    (abs ((A.1 - B.1)^2 + (A.2 - B.2)^2 - m_a^2) = 0) ∧
    (2 * dist (A1.1, A1.2) (B.1, B.2) = h_b) ∧
    (2 * dist (A1.1, A1.2) (C.1, C.2) = h_c) :=
sorry

end construct_triangle_l224_224870


namespace cos_six_arccos_one_fourth_l224_224081

theorem cos_six_arccos_one_fourth : 
  cos (6 * arccos (1 / 4)) = - (7 / 128) :=
by
  sorry

end cos_six_arccos_one_fourth_l224_224081


namespace problem_statement_l224_224493

def y_and (y : ℤ) : ℤ := 9 - y
def and_y (y : ℤ) : ℤ := y - 9

theorem problem_statement : and_y (y_and 15) = -15 := 
by
  sorry

end problem_statement_l224_224493


namespace particle_acceleration_reaches_4_units_at_some_time_l224_224388

theorem particle_acceleration_reaches_4_units_at_some_time
  (v : ℝ → ℝ) (a : ℝ → ℝ)
  (h_v : Continuous v)
  (h_a : Continuous a)
  (h_v_start : v 0 = 0)
  (h_v_end : v 1 = 0)
  (h_distance : ∫ (t : ℝ) in 0..1, v t = 1) :
  ∃ t ∈ Set.Icc (0 : ℝ) (1 : ℝ), |a t| ≥ 4 := sorry

end particle_acceleration_reaches_4_units_at_some_time_l224_224388


namespace equivalent_standard_equation_l224_224118

noncomputable def standard_equation (x y : ℝ) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ x = real.sqrt t ∧ y = 2 * real.sqrt (1 - t)

theorem equivalent_standard_equation (x y : ℝ) (h : 0 ≤ x ∧ x ≤ 1 ∧ 0 ≤ y ∧ y ≤ 2) :
  standard_equation x y ↔ x^2 + (y^2 / 4) = 1 :=
by
  sorry

end equivalent_standard_equation_l224_224118


namespace percentage_books_not_sold_l224_224798

theorem percentage_books_not_sold :
    let initial_stock := 700
    let books_sold_mon := 50
    let books_sold_tue := 82
    let books_sold_wed := 60
    let books_sold_thu := 48
    let books_sold_fri := 40
    let total_books_sold := books_sold_mon + books_sold_tue + books_sold_wed + books_sold_thu + books_sold_fri 
    let books_not_sold := initial_stock - total_books_sold
    let percentage_not_sold := (books_not_sold * 100) / initial_stock
    percentage_not_sold = 60 :=
by
  -- definitions
  let initial_stock := 700
  let books_sold_mon := 50
  let books_sold_tue := 82
  let books_sold_wed := 60
  let books_sold_thu := 48
  let books_sold_fri := 40
  let total_books_sold := books_sold_mon + books_sold_tue + books_sold_wed + books_sold_thu + books_sold_fri
  let books_not_sold := initial_stock - total_books_sold
  let percentage_not_sold := (books_not_sold * 100) / initial_stock
  have : percentage_not_sold = 60 := sorry
  exact this

end percentage_books_not_sold_l224_224798


namespace calc_probability_l224_224178

variable (ξ : ℝ → Prop)
variable (N : ℝ → ℝ → Prop)

axiom normal_distribution (xi : ℝ → Prop) (mu sigma : ℝ) : Prop :=
  ∃ N, xi = N(mu, sigma)

def probability (A : ℝ → Prop) (a b : ℝ) : ℝ :=
sorry

theorem calc_probability :
  normal_distribution ξ (-1) 36 →
  probability ξ (-3) (-1) = 0.4 →
  probability ξ 1 ∞ = 0.1 :=
sorry

end calc_probability_l224_224178


namespace perpendicular_vectors_l224_224642

noncomputable def a : EuclideanSpace ℝ (Fin 2) := ![1, -2]
noncomputable def b (m : ℝ) : EuclideanSpace ℝ (Fin 2) := ![6, m]

theorem perpendicular_vectors (m : ℝ) : (a ⬝ (b m) = 0) → m = 3 := by
  sorry

end perpendicular_vectors_l224_224642


namespace perimeter_of_square_l224_224702

theorem perimeter_of_square (a : Real) (h_a : a ^ 2 = 144) : 4 * a = 48 :=
by
  sorry

end perimeter_of_square_l224_224702


namespace A_investment_is_correct_l224_224065

-- Definitions based on the given conditions
def B_investment : ℝ := 8000
def C_investment : ℝ := 10000
def P_B : ℝ := 1000
def diff_P_A_P_C : ℝ := 500

-- Main statement we need to prove
theorem A_investment_is_correct (A_investment : ℝ) 
  (h1 : B_investment = 8000) 
  (h2 : C_investment = 10000)
  (h3 : P_B = 1000)
  (h4 : diff_P_A_P_C = 500)
  (h5 : A_investment = B_investment * (P_B / 1000) * 1.5) :
  A_investment = 12000 :=
sorry

end A_investment_is_correct_l224_224065


namespace beetle_total_distance_l224_224043

theorem beetle_total_distance (r : ℝ) (r_eq : r = 75) : (2 * r + r + r) = 300 := 
by
  sorry

end beetle_total_distance_l224_224043


namespace perimeter_square_C_l224_224306

theorem perimeter_square_C 
  (a b c : ℝ) 
  (ha : 4 * a = 16) 
  (hb : 4 * b = 28) 
  (hc : c = |a - b|) : 
  4 * c = 12 := 
sorry

end perimeter_square_C_l224_224306


namespace green_ball_count_l224_224276

theorem green_ball_count 
  (total_balls : ℕ)
  (n_red n_blue n_green : ℕ)
  (h_total : n_red + n_blue + n_green = 50)
  (h_red : ∀ (A : Finset ℕ), A.card = 34 -> ∃ a ∈ A, a < n_red)
  (h_blue : ∀ (A : Finset ℕ), A.card = 35 -> ∃ a ∈ A, a < n_blue)
  (h_green : ∀ (A : Finset ℕ), A.card = 36 -> ∃ a ∈ A, a < n_green)
  : n_green = 15 ∨ n_green = 16 ∨ n_green = 17 :=
by
  sorry

end green_ball_count_l224_224276


namespace repeating_decimal_to_fraction_l224_224776

theorem repeating_decimal_to_fraction : (0.5656565656 : ℚ) = 56 / 99 :=
  by
    sorry

end repeating_decimal_to_fraction_l224_224776


namespace functions_are_equal_l224_224432

def f (x : ℝ) : ℝ := 2 * |x|
def g (x : ℝ) : ℝ := Real.sqrt (4 * x^2)

theorem functions_are_equal (x : ℝ) : f x = g x :=
by
  sorry

end functions_are_equal_l224_224432


namespace balloon_permutations_l224_224944

theorem balloon_permutations : 
  let balloons := "BALLOON".toList.length in
  let total_permutations := Nat.factorial balloons in
  let repetitive_L := Nat.factorial 2 in
  let repetitive_O := Nat.factorial 2 in
  (total_permutations / (repetitive_L * repetitive_O)) = 1260 := sorry

end balloon_permutations_l224_224944


namespace minor_arc_length_l224_224619

/-- 
Given three points A, B, and C on a circle with radius 15 and an inscribed angle ACB of 50 degrees, 
the length of the minor arc AB is 25/3 * π.
-/
theorem minor_arc_length {A B C : Type} (r : ℝ) (angle_ACB : ℝ) (circle_radius : r = 15) (angle_ACB_measure : angle_ACB = 50) :
  (100 / 360) * (2 * real.pi * r) = (25 / 3) * real.pi :=
by
  have h_radius : r = 15 := circle_radius
  have h_angle : angle_ACB = 50 := angle_ACB_measure
  calc
    (100 / 360) * (2 * real.pi * 15)
    = (100 / 360) * 30 * real.pi : by rw [←h_radius, mul_assoc, ←mul_assoc (100 / 360)]
    ... = (25 / 3) * real.pi     : by norm_num

end minor_arc_length_l224_224619


namespace sherman_drives_nine_hours_a_week_l224_224669

-- Define the daily commute time in minutes.
def daily_commute_time := 30 + 30

-- Define the number of weekdays Sherman commutes.
def weekdays := 5

-- Define the weekly commute time in minutes.
def weekly_commute_time := weekdays * daily_commute_time

-- Define the conversion from minutes to hours.
def minutes_to_hours (m : ℕ) : ℕ := m / 60

-- Define the weekend driving time in hours.
def weekend_driving_time := 2 * 2

-- Define the total weekly driving time in hours.
def total_weekly_driving_time := minutes_to_hours weekly_commute_time + weekend_driving_time

-- The theorem we need to prove
theorem sherman_drives_nine_hours_a_week :
  total_weekly_driving_time = 9 :=
by
  sorry

end sherman_drives_nine_hours_a_week_l224_224669


namespace petya_green_balls_l224_224278

theorem petya_green_balls (total_balls : ℕ) (red_balls blue_balls green_balls : ℕ)
  (h1 : total_balls = 50)
  (h2 : ∀ s, s.card ≥ 34 → ∃ r, r ∈ s ∧ r = red_balls)
  (h3 : ∀ s, s.card ≥ 35 → ∃ b, b ∈ s ∧ b = blue_balls)
  (h4 : ∀ s, s.card ≥ 36 → ∃ g, g ∈ s ∧ g = green_balls) :
  green_balls = 15 ∨ green_balls = 16 ∨ green_balls = 17 :=
sorry

end petya_green_balls_l224_224278


namespace find_shirts_yesterday_l224_224436

def shirts_per_minute : ℕ := 8
def total_minutes : ℕ := 2
def shirts_today : ℕ := 3

def total_shirts : ℕ := shirts_per_minute * total_minutes
def shirts_yesterday : ℕ := total_shirts - shirts_today

theorem find_shirts_yesterday : shirts_yesterday = 13 := by
  sorry

end find_shirts_yesterday_l224_224436


namespace intersection_S_T_l224_224547

def S := {x : ℝ | abs x < 5}
def T := {x : ℝ | (x + 7) * (x - 3) < 0}

theorem intersection_S_T : S ∩ T = {x : ℝ | -5 < x ∧ x < 3} :=
by
  sorry

end intersection_S_T_l224_224547


namespace repeating_decimal_as_fraction_l224_224745

theorem repeating_decimal_as_fraction :
  let x := 56 / 99
  in x = 0.56 + 0.0056 + 0.000056 + (0.00000056) : ℚ :=
by
  sorry

end repeating_decimal_as_fraction_l224_224745


namespace parabola_directrix_l224_224319

theorem parabola_directrix (x y : ℝ) : (y^2 = 2 * x) → (x = -(1 / 2)) := by
  sorry

end parabola_directrix_l224_224319


namespace tangent_line_with_min_slope_eq_l224_224458

noncomputable def f (x : ℝ) := x^3 - 6 * x^2 - x + 6

theorem tangent_line_with_min_slope_eq : 
  ∃ (m b : ℝ), (∀ (x : ℝ), 
    (∃ y : ℝ, f x = y ∧ y = m * x + b) ∧ 
    (∀ (x₀ x₁ : ℝ), f' x₀ = m ∧ f' x₁ ≤ f' x₀) ∧ 
    (b = -14 ∧ m = 13)
  ) :=
sorry

end tangent_line_with_min_slope_eq_l224_224458


namespace recurring_to_fraction_l224_224751

theorem recurring_to_fraction : ∀ (x : ℚ), x = 0.5656 ∧ 100 * x = 56.5656 → x = 56 / 99 :=
by
  intros x hx
  sorry

end recurring_to_fraction_l224_224751


namespace find_m_div_n_l224_224850

noncomputable def ellipse_line_intersection (m n : ℝ) (x1 y1 x2 y2 x0 y0 : ℝ) : Prop :=
  let ellipse := m * x1^2 + n * y1^2 = 1 ∧ m * x2^2 + n * y2^2 = 1
  let line := y1 = 1 - 4*x1 ∧ y2 = 1 - 4*x2
  let midpoint := (x1 + x2) / 2 = x0 ∧ (y1 + y2) / 2 = y0
  let origin_slope := y0 / x0 = sqrt 2 / 2
ellipse_line_intersection :=
  ellipse ∧ line ∧ midpoint ∧ origin_slope

theorem find_m_div_n (m n x1 y1 x2 y2 x0 y0 : ℝ) : ellipse_line_intersection m n x1 y1 x2 y2 x0 y0 →
  m / n = 2 * sqrt 2 :=
sorry

end find_m_div_n_l224_224850


namespace foot_of_altitude_l224_224842

variables {A B C L M N P : Type} [euclidean_geometry ABC]

-- Assume the following conditions:
-- 1. \( ABC \) is a triangle
-- 2. \( L, M, \) and \( N \) are the midpoints of \( BC, CA, \) and \( AB \), respectively.
-- 3. Point \( P \) is on \( BC \).
-- 4. \(\angle CPM = \angle LNM\).

def midpoint (A B C L M N : Type) : Prop := 
  is_midpoint L B C ∧ is_midpoint M C A ∧ is_midpoint N A B

def point_on_line (P B C : Type) : Prop := 
  is_point_on_line P B C

def angle_eq (M N C P : Type) : Prop := 
  angle_eq ∠CPM ∠LNM

theorem foot_of_altitude {A B C L M N P : Type} [euclidean_geometry ABC]
  (h1 : midpoint A B C L M N)
  (h2 : point_on_line P B C)
  (h3 : angle_eq M N C P) :
  is_foot_of_altitude_from A P :=
sorry

end foot_of_altitude_l224_224842


namespace no_integer_pairs_satisfy_equation_l224_224485

theorem no_integer_pairs_satisfy_equation :
  ¬ ∃ m n : ℤ, m^3 + 8 * m^2 + 17 * m = 8 * n^3 + 12 * n^2 + 6 * n + 1 :=
sorry

end no_integer_pairs_satisfy_equation_l224_224485


namespace minimum_value_l224_224247

theorem minimum_value (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + y = 20) :
  (∃ (m : ℝ), m = (1 / x ^ 2 + 1 / y ^ 2) ∧ m ≥ 2 / 25) :=
by
  sorry

end minimum_value_l224_224247


namespace largest_exponent_l224_224741

theorem largest_exponent : 
  ∀ (a b c d e : ℕ), a = 2^5000 → b = 3^4000 → c = 4^3000 → d = 5^2000 → e = 6^1000 → b > a ∧ b > c ∧ b > d ∧ b > e :=
by
  sorry

end largest_exponent_l224_224741


namespace green_ball_count_l224_224277

theorem green_ball_count 
  (total_balls : ℕ)
  (n_red n_blue n_green : ℕ)
  (h_total : n_red + n_blue + n_green = 50)
  (h_red : ∀ (A : Finset ℕ), A.card = 34 -> ∃ a ∈ A, a < n_red)
  (h_blue : ∀ (A : Finset ℕ), A.card = 35 -> ∃ a ∈ A, a < n_blue)
  (h_green : ∀ (A : Finset ℕ), A.card = 36 -> ∃ a ∈ A, a < n_green)
  : n_green = 15 ∨ n_green = 16 ∨ n_green = 17 :=
by
  sorry

end green_ball_count_l224_224277


namespace value_of_f_30_l224_224246

theorem value_of_f_30 (f : ℕ → ℕ) 
  (h1 : ∀ n : ℕ, 0 < n → f(n + 1) > f(n))
  (h2 : ∀ (m n : ℕ), 0 < m → 0 < n → f(m * n) = f(m) * f(n))
  (h3 : ∀ (m n : ℕ), 0 < m → 0 < n → (m ≠ n → m ^ m = n ^ n → f(m) = n ∨ f(n) = m)) :
  f(30) = 900 :=
sorry

end value_of_f_30_l224_224246


namespace number_of_factors_l224_224459

theorem number_of_factors : 
  ∃ (count : ℕ), count = 45 ∧
    (∀ n : ℕ, (1 ≤ n ∧ n ≤ 500) → 
      ∃ a b : ℤ, (x - a) * (x - b) = x^2 + 2 * x - n) :=
by
  sorry

end number_of_factors_l224_224459


namespace prove_inequality_cos_l224_224033

noncomputable def inequality_cos : Prop :=
  ∀ x : ℝ, 
  (0 ≤ x^4 - 5 * x^2 + 4) →
  cos (2 * x^3 - x^2 - 5 * x - 2) +
  cos (2 * x^3 + 3 * x^2 - 3 * x - 2) -
  cos ((2 * x + 1) * sqrt (x^4 - 5 * x^2 + 4)) < 3

theorem prove_inequality_cos : inequality_cos :=
by
  sorry

end prove_inequality_cos_l224_224033


namespace equilateral_triangle_inequality_equilateral_triangle_equality_condition_l224_224791

noncomputable theory

variables {A B C M : Point}
variable (ABC : Triangle)
variable (is_equilateral_ABC : EquilateralTriangle ABC)
variable {distance : Point → Point → ℝ}

theorem equilateral_triangle_inequality (hM : ∀ P, distance M P ∈ plane) :
  distance M A ≤ distance M B + distance M C :=
sorry

theorem equilateral_triangle_equality_condition (hM : ∀ P, distance M P ∈ plane) :
  (distance M A = distance M B + distance M C) ↔ M ∈ circumcircle ABC :=
sorry

end equilateral_triangle_inequality_equilateral_triangle_equality_condition_l224_224791


namespace lcm_18_28_45_65_eq_16380_l224_224481

theorem lcm_18_28_45_65_eq_16380 : Nat.lcm 18 (Nat.lcm 28 (Nat.lcm 45 65)) = 16380 :=
sorry

end lcm_18_28_45_65_eq_16380_l224_224481


namespace inversely_proportional_x_y_l224_224678

-- Statement of the problem
theorem inversely_proportional_x_y :
  ∃ k : ℝ, (∀ (x y : ℝ), (x * y = k) ∧ (x = 4) ∧ (y = 2) → x * (-5) = -8 / 5) :=
by
  sorry

end inversely_proportional_x_y_l224_224678


namespace sum_of_solutions_of_equation_is_zero_l224_224367

theorem sum_of_solutions_of_equation_is_zero :
  let f (x : ℝ) := (6 * x) / 30 - 8 / x in
  (∃ x1 x2 : ℝ, f x1 = 0 ∧ f x2 = 0 ∧ x1 + x2 = 0) :=
by
  sorry

end sum_of_solutions_of_equation_is_zero_l224_224367


namespace value_of_f_at_6_l224_224514

-- The condition that f is an odd function
def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = -f (x)

-- The condition that f(x + 2) = -f(x)
def periodic_sign_flip (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (x + 2) = -f (x)

-- The theorem statement
theorem value_of_f_at_6 (f : ℝ → ℝ) (h1 : is_odd_function f) (h2 : periodic_sign_flip f) : f 6 = 0 :=
sorry

end value_of_f_at_6_l224_224514


namespace percentage_increase_correct_l224_224231

noncomputable def Avg_before_John : ℝ := 75
noncomputable def Num_contributions_before_John : ℕ := 6
noncomputable def John_donation : ℝ := 225

def Total_before_John (Avg_before_John : ℝ) (Num_contributions_before_John : ℕ) : ℝ :=
  Avg_before_John * Num_contributions_before_John

def Total_after_John (Total_before_John : ℝ) (John_donation : ℝ) : ℝ :=
  Total_before_John + John_donation 

def Avg_after_John (Total_after_John : ℝ) (Num_contributions_after_John : ℕ) : ℝ :=
  Total_after_John / Num_contributions_after_John

def Percentage_increase (Avg_before_John Avg_after_John : ℝ) : ℝ :=
  ((Avg_after_John - Avg_before_John) / Avg_before_John) * 100

theorem percentage_increase_correct :
  let Total_before_John := Total_before_John Avg_before_John Num_contributions_before_John
  let Total_after_John := Total_after_John Total_before_John John_donation
  let Num_contributions_after_John := Num_contributions_before_John + 1
  let Avg_after_John := Avg_after_John Total_after_John Num_contributions_after_John
  Percentage_increase Avg_before_John Avg_after_John ≈ 28.57 :=
by
  sorry

end percentage_increase_correct_l224_224231


namespace cos_largest_angle_value_l224_224224

noncomputable def cos_largest_angle (a b c : ℝ) (h1 : a = 2) (h2 : b = 3) (h3 : c = 4) : ℝ :=
  (a * a + b * b - c * c) / (2 * a * b)

theorem cos_largest_angle_value : cos_largest_angle 2 3 4 (by rfl) (by rfl) (by rfl) = -1 / 4 := 
sorry

end cos_largest_angle_value_l224_224224


namespace integer_points_in_intersection_l224_224869

theorem integer_points_in_intersection :
  let sphere1 (x y z : ℤ) := (x - 3)^2 + y^2 + (z - 10)^2 ≤ 64
  let sphere2 (x y z : ℤ) := x^2 + y^2 + (z - 12)^2 ≤ 9
  (Finset.filter
    (λ (p : ℤ × ℤ × ℤ), sphere1 p.1 p.2.1 p.2.2 ∧ sphere2 p.1 p.2.1 p.2.2)
    ((Finset.Icc (-8) 18).product ((Finset.Icc (-8) 8).product (Finset.Icc (2) 18)))
  ).card = 7 := 
sorry

end integer_points_in_intersection_l224_224869


namespace triangle_vector_sum_l224_224221

variable {V : Type} [AddCommGroup V] [Module ℝ V] 
variables {A B C D : V}

theorem triangle_vector_sum (hD : D ∈ line_segment ℝ A B) 
  (hAD : (2 : ℝ) • (B - D) = A - D) : 
  (C - D) = (1/3 : ℝ) • (C - A) + (2/3 : ℝ) • (C - B) :=
sorry

end triangle_vector_sum_l224_224221


namespace magnitude_of_T_l224_224241

theorem magnitude_of_T :
  let i : ℂ := complex.I in
  let T : ℂ := (1 + i) ^ 19 - (1 - i) ^ 19 in
  complex.abs T = 512 * real.sqrt 2 :=
by
  sorry

end magnitude_of_T_l224_224241


namespace tangent_alignment_l224_224252

namespace TeubnerTheorem

variable {AM BC AD ω Γ Δ : Type} -- Defining variables for lines, circles and triangles

def incircle (t : Δ) : Type := sorry -- placeholder definition for incircle
def circumcircle (t : Δ) : Type := sorry -- placeholder definition for circumcircle
def tangent (c : incircle Δ) (p : Type) : Type := sorry -- placeholder definition for tangent

variables {t : Δ} {p S U_d intersect_point_E : Type}

axiom formed_triangle : t = Δ -- axiom stating triangle formation
axiom intersect_circumcircle : tangent (incircle t) S = circumcircle t -- tangent intersects circumcircle
axiom arbitrary_point_D : ∃ (d : Type), d ∈ BC -- D lies on BC
axiom teubner_touch : ∀ (d : Type), ∃ (teubner_1 teubner_2 : Type), (teubner_1 ∈ tangent (circumcircle t) d) ∧ (teubner_2 ∈ tangent (circumcircle t) d) -- definition of Teubner circles

axiom common_external_tangent : ∀ (teubner_1 teubner_2 : Type), ∃ (tangent_common : Type), tangent_common ∈ tangent (incircle t) ω -- common external tangent
axiom tangent_point_E : intersect_point_E ∈ tangent (incircle t) ω -- intersection point E on tangent to ω

-- Statement of the theorem to be proved
theorem tangent_alignment
  (tangent_intersection : ∀ (tangent_common : Type), ∃ (intersection_segment : Type), intersection_segment = AD) :
  Proof :=
sorry

end TeubnerTheorem

end tangent_alignment_l224_224252


namespace correct_option_is_B_l224_224736

theorem correct_option_is_B (a : ℝ) : 
  (¬ (-2 * a^2 * b)^3 = -6 * a^6 * b^3) ∧
  (a^7 / a = a^6) ∧
  (¬ (a + 1)^2 = a^2 + 1) ∧
  (¬ 2 * a + 3 * b = 5 * a * b) :=
by
  sorry

end correct_option_is_B_l224_224736


namespace find_a_plus_b_l224_224571

theorem find_a_plus_b (a b : ℕ) (ha : 0 < a) (hb : 0 < b) (h : a^2 - b^4 = 2009) : a + b = 47 := 
by 
  sorry

end find_a_plus_b_l224_224571


namespace simplest_quadratic_radical_l224_224022

theorem simplest_quadratic_radical :
  ∀ (x : ℝ), (x = sqrt 7) ↔ (sqrt 12 ≠ x ∧ sqrt (2 / 3) ≠ x ∧ sqrt 0.2 ≠ x) := 
sorry

end simplest_quadratic_radical_l224_224022


namespace balloon_permutation_count_l224_224929

-- Define the given conditions:
def balloon_letters : List Char := ['B', 'A', 'L', 'L', 'O', 'O', 'N']
def total_letters : Nat := 7
def count_L : Nat := 2
def count_O : Nat := 2

-- Statement to prove:
theorem balloon_permutation_count :
  (nat.factorial total_letters) / ((nat.factorial count_L) * (nat.factorial count_O)) = 1260 := by
  sorry

end balloon_permutation_count_l224_224929


namespace max_value_of_f_l224_224096

def f (x : ℝ) : ℝ := 12 * x - 4 * x^2

theorem max_value_of_f : ∀ x : ℝ, f x ≤ 9 :=
by
  have h₁ : ∀ x : ℝ, 12 * x - 4 * x^2 ≤ 9
  { sorry }
  exact h₁

end max_value_of_f_l224_224096


namespace firetruck_reachable_area_l224_224197

theorem firetruck_reachable_area (r1 r2 : ℝ) (t : ℝ) (h1 : r1 = 60)
  (h2 : r2 = 18) (h3 : t = 0.25) :
  let d1 := r1 * t in
  let d2 := r2 * t in
  let r := d1 + d2 in
  let area := π * r^2 in
  ∃ m n : ℤ, m / ∈ rel_prime n ∧ area = m / n ∧ m + n = 119575 :=
by
  sorry

end firetruck_reachable_area_l224_224197


namespace total_expenditure_l224_224403

-- Definitions of costs and purchases
def bracelet_cost : ℕ := 4
def keychain_cost : ℕ := 5
def coloring_book_cost : ℕ := 3

def paula_bracelets : ℕ := 2
def paula_keychains : ℕ := 1

def olive_coloring_books : ℕ := 1
def olive_bracelets : ℕ := 1

-- Hypothesis stating the total expenditure for Paula and Olive
theorem total_expenditure
  (bracelet_cost keychain_cost coloring_book_cost : ℕ)
  (paula_bracelets paula_keychains olive_coloring_books olive_bracelets : ℕ) :
  paula_bracelets * bracelet_cost + paula_keychains * keychain_cost + olive_coloring_books * coloring_book_cost + olive_bracelets * bracelet_cost = 20 := 
  by
  -- Applying the given costs
  let bracelet_cost := 4
  let keychain_cost := 5
  let coloring_book_cost := 3 

  -- Applying the purchases made by Paula and Olive
  let paula_bracelets := 2
  let paula_keychains := 1
  let olive_coloring_books := 1
  let olive_bracelets := 1

  sorry

end total_expenditure_l224_224403


namespace balloon_permutations_count_l224_224910

-- Definitions of the conditions
def total_letters_count : ℕ := 7
def l_count : ℕ := 2
def o_count : ℕ := 2

-- Now the mathematical problem as a Lean statement
theorem balloon_permutations_count : 
  (Nat.factorial total_letters_count) / ((Nat.factorial l_count) * (Nat.factorial o_count)) = 1260 := 
by
  sorry

end balloon_permutations_count_l224_224910


namespace balloon_arrangements_l224_224895

theorem balloon_arrangements :
  let n := 7
  let k1 := 2
  let k2 := 2
  (Nat.factorial n) / ((Nat.factorial k1) * (Nat.factorial k2)) = 1260 :=
by
  -- Use the conditions directly from the problem
  let n := 7
  let k1 := 2
  let k2 := 2
  -- Mathematically, the number of arrangements of the multiset
  have h : (Nat.factorial n) / ((Nat.factorial k1) * (Nat.factorial k2)) = 1260 := sorry
  exact h

end balloon_arrangements_l224_224895


namespace recurring_decimal_to_fraction_correct_l224_224764

noncomputable def recurring_decimal_to_fraction (b : ℚ) : Prop :=
  b = 0.\overline{56} ↔ b = 56/99

theorem recurring_decimal_to_fraction_correct : recurring_decimal_to_fraction 0.\overline{56} :=
  sorry

end recurring_decimal_to_fraction_correct_l224_224764


namespace total_exclusive_l224_224852

-- Conditions
variables (A J : ℕ) (shared : ℕ) (exclusive_john : ℕ)
variables (total_andrew : ℕ) (collection_total : ℕ)

-- Given conditions
def shared_albums := (shared = 12)
def andrew_total := (total_andrew = 20)
def john_exclusive := (exclusive_john = 8)

-- Calculate exclusive to Andrew
def exclusive_andrew := (total_andrew - shared)

-- Question: Calculate exclusive albums and show result is 16
theorem total_exclusive (h1 : shared_albums) (h2 : andrew_total) (h3 : john_exclusive) :
  exclusive_andrew + exclusive_john = 16 :=
by
  sorry

end total_exclusive_l224_224852


namespace average_people_per_row_l224_224812

theorem average_people_per_row (boys girls rows : ℕ) (h_boys : boys = 24) (h_girls : girls = 24) (h_rows : rows = 6) : 
  (boys + girls) / rows = 8 :=
by
  sorry

end average_people_per_row_l224_224812


namespace function_bounds_l224_224187

theorem function_bounds {a : ℝ} :
  (∀ x : ℝ, x > 0 → 4 - x^2 + a * Real.log x ≤ 3) → a = 2 :=
by
  sorry

end function_bounds_l224_224187


namespace divide_24kg_into_parts_l224_224555

theorem divide_24kg_into_parts (W : ℕ) (part1 part2 : ℕ) (h_sum : part1 + part2 = 24) :
  (part1 = 9 ∧ part2 = 15) ∨ (part1 = 15 ∧ part2 = 9) :=
by
  sorry

end divide_24kg_into_parts_l224_224555


namespace ratio_AE_BE_l224_224220

-- Define the known values
variables (A B C D E : Type)
variables {AB BC AC : ℝ}
variables [midpoint : Midpoint B C D]
variables [tangent : TangentToCircleAtPointAndLine E D A C]
variables [diameter : DiameterOfCircle D E]

-- Assumptions based on the problem conditions
axiom AB_eq_sqrt14 : AB = Real.sqrt 14
axiom BC_eq_2 : BC = 2

-- Stating the target proposition
theorem ratio_AE_BE : (AE / BE) = (4 / 3) :=
by 
  sorry

end ratio_AE_BE_l224_224220


namespace balloon_permutations_l224_224964

theorem balloon_permutations : (∀ (arrangements : ℕ),
  arrangements = nat.factorial 7 / (nat.factorial 2 * nat.factorial 2) → arrangements = 1260) :=
begin
  intros arrangements h,
  rw [nat.factorial_succ, nat.factorial],
  sorry,
end

end balloon_permutations_l224_224964


namespace expression_simplification_l224_224322

theorem expression_simplification (a : ℝ) (h : a ≠ 0) :
  (a ^ (-2) / a ^ 5) * (4 * a / (2 ^ (-1) * a) ^ (-3)) = 1 / (2 * a ^ 3) :=
by sorry

end expression_simplification_l224_224322


namespace peter_horses_food_requirement_l224_224650

theorem peter_horses_food_requirement :
  let daily_oats_per_horse := 4 * 2 in
  let daily_grain_per_horse := 3 in
  let daily_food_per_horse := daily_oats_per_horse + daily_grain_per_horse in
  let number_of_horses := 4 in
  let daily_food_all_horses := daily_food_per_horse * number_of_horses in
  let days := 3 in
  daily_food_all_horses * days = 132 :=
by
  sorry

end peter_horses_food_requirement_l224_224650


namespace inequality_proof_l224_224289

theorem inequality_proof (a b c : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : c > 0) :
  a^3 * b + b^3 * c + c^3 * a ≥ a^2 * b * c + b^2 * c * a + c^2 * a * b :=
by {
  sorry
}

end inequality_proof_l224_224289


namespace balloon_permutations_l224_224953

theorem balloon_permutations : 
  let total_letters := 7
  let repetitions_L := 2
  let repetitions_O := 2
  (Nat.factorial total_letters) / ((Nat.factorial repetitions_L) * (Nat.factorial repetitions_O)) = 1260 := 
by
  sorry

end balloon_permutations_l224_224953


namespace total_bags_proof_l224_224715

def pounds_oranges : ℝ := 156.0
def pounds_apples : ℝ := 89.0
def pounds_bananas : ℝ := 54.0

def bag_capacity_oranges : ℝ := 23.0
def bag_capacity_apples : ℝ := 15.0
def bag_capacity_bananas : ℝ := 8.0

noncomputable def calculate_bags (total_pounds : ℝ) (bag_capacity : ℝ) : ℕ :=
  (total_pounds / bag_capacity).ceil.toNat

noncomputable def total_bags_needed :=
  calculate_bags pounds_oranges bag_capacity_oranges +
  calculate_bags pounds_apples bag_capacity_apples +
  calculate_bags pounds_bananas bag_capacity_bananas

theorem total_bags_proof : total_bags_needed = 20 :=
  by
  sorry

end total_bags_proof_l224_224715


namespace min_sum_abc_l224_224337

theorem min_sum_abc (a b c : ℕ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0) (h_prod : a * b * c = 2450) : a + b + c ≥ 82 :=
sorry

end min_sum_abc_l224_224337


namespace Sherman_weekly_driving_time_l224_224665

theorem Sherman_weekly_driving_time (daily_commute : ℕ := 30) (weekend_drive : ℕ := 2) : 
  (5 * (2 * daily_commute) / 60 + 2 * weekend_drive) = 9 := 
by
  sorry

end Sherman_weekly_driving_time_l224_224665


namespace sum_of_roots_of_quadratic_l224_224488

theorem sum_of_roots_of_quadratic :
  let equation := λ x : ℝ, x^2 + 1992 * x - 1993 = 0 in
  ∀ x1 x2 : ℝ, equation x1 → equation x2 → x1 + x2 = -1992 := by
  assume equation x1 x2,
  sorry

end sum_of_roots_of_quadratic_l224_224488


namespace shorter_diagonal_length_l224_224357

-- Definition of the trapezoid with given side lengths
structure Trapezoid (EF GH EG FH : ℝ) :=
(parallel_sides : EF ∥ GH)
(side1 : EG = 13)
(side2 : FH = 15)
(parallel1 : EF = 39)
(parallel2 : GH = 27)
(acute_angleE : is_acute ∠E)
(acute_angleF : is_acute ∠F)

-- The theorem to state the length of the shorter diagonal
theorem shorter_diagonal_length {EF GH EG FH : ℝ}
  (T : Trapezoid EF GH EG FH) : 
  min_diagonal_length T = 25 :=
sorry

end shorter_diagonal_length_l224_224357


namespace find_c_for_quadratic_inequality_l224_224369

def quadratic_inequality_condition (c : ℝ) (x : ℝ) : Prop :=
  -3 * x^2 + c * x - 8 < 0

theorem find_c_for_quadratic_inequality :
  (∀ x : ℝ, quadratic_inequality_condition 18 x ↔ (x ∈ set.Iio 2 ∪ set.Ioi 4)) → 
  18 = 18 :=
by
  intro h
  rfl

end find_c_for_quadratic_inequality_l224_224369


namespace fishermen_total_catch_l224_224800

noncomputable def m : ℕ := 30  -- Mike can catch 30 fish per hour
noncomputable def j : ℕ := 2 * m  -- Jim can catch twice as much as Mike
noncomputable def b : ℕ := j + (j / 2)  -- Bob can catch 50% more than Jim

noncomputable def fish_caught_in_40_minutes : ℕ := (2 * m) / 3 -- Fishermen fish together for 40 minutes (2/3 hour)
noncomputable def fish_caught_by_jim_in_remaining_time : ℕ := j / 3 -- Jim fishes alone for the remaining 20 minutes (1/3 hour)

noncomputable def total_fish_caught : ℕ :=
  fish_caught_in_40_minutes * 3 + fish_caught_by_jim_in_remaining_time

theorem fishermen_total_catch : total_fish_caught = 140 := by
  sorry

end fishermen_total_catch_l224_224800


namespace trajectory_of_Q_is_parabola_l224_224511

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

end trajectory_of_Q_is_parabola_l224_224511


namespace total_vases_l224_224606

theorem total_vases (vases_per_day : ℕ) (days : ℕ) (total_vases : ℕ) 
  (h1 : vases_per_day = 16) 
  (h2 : days = 16) 
  (h3 : total_vases = vases_per_day * days) : 
  total_vases = 256 := 
by 
  sorry

end total_vases_l224_224606


namespace ratio_of_areas_l224_224054

noncomputable def parabola : Set (ℝ × ℝ) := {p | p.2^2 = 2 * p.1}
noncomputable def disk : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 ≤ 8}

theorem ratio_of_areas : 
  let S1 := ∫ y in -2..2, (sqrt (8 - y^2) - y^2 / 2) 
  let S2 := 8 * π - S1 
  S1 / S2 = (3 * π + 2) / (9 * π - 2) :=
by
  sorry

end ratio_of_areas_l224_224054


namespace distinct_gamma_count_l224_224040

-- Define what it means for a triple to be γ-special
def gamma_special (γ a b c : ℕ) : Prop := 
  a ≤ γ * (b + c) ∧ 
  b ≤ γ * (c + a) ∧ 
  c ≤ γ * (a + b)

-- Define the range condition
def in_range (a b c : ℕ) : Prop := 
  1 ≤ a ∧ a ≤ 20 ∧ 
  1 ≤ b ∧ b ≤ 20 ∧ 
  1 ≤ c ∧ c ≤ 20

-- Define the problem statement
theorem distinct_gamma_count : 
  (∑ a in Finset.range 20 + 1, (2 * Nat.totient a - 1)) + 1 = 
  ∑ a in Finset.range 20 + 1, (2 * Nat.totient a - 1) :=
sorry

end distinct_gamma_count_l224_224040


namespace options_evaluation_l224_224071

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f(x)

def is_monotonically_increasing (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f(x) < f(y)

def y_A (x : ℝ) : ℝ := x^3
def y_B (x : ℝ) : ℝ := 2^x
def y_C (x : ℝ) : ℝ := x - 1/x
def y_D (x : ℝ) : ℝ := Real.sin (2 * x)

theorem options_evaluation :
  (is_odd_function y_A ∧ is_monotonically_increasing y_A) ∧
  ¬(is_odd_function y_B ∧ is_monotonically_increasing y_B) ∧
  ¬(is_odd_function y_C ∧ is_monotonically_increasing y_C) ∧
  ¬(is_odd_function y_D ∧ is_monotonically_increasing y_D) :=
by
  sorry

end options_evaluation_l224_224071


namespace no_photo_overlap_l224_224851

def lap_time_andrew : ℕ := 75 -- seconds
def lap_time_bella : ℕ := 120 -- seconds
def track_fraction_camera : ℚ := 1 / 4
def photographer_position_fraction : ℚ := 1 / 3
def photo_start_time : ℕ := 540 -- 9 minutes in seconds
def photo_end_time : ℕ := 600 -- 10 minutes in seconds

def andrew_position (time : ℕ) : ℚ := (time % lap_time_andrew : ℕ) / lap_time_andrew
def bella_position (time : ℕ) : ℚ := (time % lap_time_bella : ℕ) / lap_time_bella

def in_photo_range (position : ℚ) : Prop :=
  photographer_position_fraction - track_fraction_camera / 2 ≤ position ∧ position ≤ photographer_position_fraction + track_fraction_camera / 2

theorem no_photo_overlap :
  ∀ t in photo_start_time..photo_end_time,
    ¬ (in_photo_range (andrew_position t) ∧ in_photo_range (bella_position t)) :=
  by
  sorry

end no_photo_overlap_l224_224851


namespace cistern_empty_time_without_tap_l224_224045

noncomputable def leak_rate (L : ℕ) : Prop :=
  let tap_rate := 4
  let cistern_volume := 480
  let empty_time_with_tap := 24
  let empty_rate_net := cistern_volume / empty_time_with_tap
  L - tap_rate = empty_rate_net

theorem cistern_empty_time_without_tap (L : ℕ) (h : leak_rate L) :
  480 / L = 20 := by
  -- placeholder for the proof
  sorry

end cistern_empty_time_without_tap_l224_224045


namespace fx_alpha_value_l224_224528

noncomputable def fx (x : ℝ) (ω θ : ℝ) : ℝ := cos (ω * x + θ)

theorem fx_alpha_value (ω θ α : ℝ) (hω : ω > 0) (hθ1 : 0 < θ) (hθ2 : θ < π) 
  (hT : 2 * π / ω = π) (hodd : ∀ x : ℝ, fx x ω θ + fx (-x) ω θ = 0) 
  (htanα : tan α = 2) : fx α ω θ = -4 / 5 :=
by
  -- Proof omitted
  sorry

end fx_alpha_value_l224_224528


namespace term_2089_is_16_l224_224390

noncomputable def next_term (n : ℕ) := (nat.digits 10 n).map (λ x, x ^ 2).sum

def sequence : ℕ → ℕ
| 0       := 2089
| (n + 1) := next_term (sequence n)

theorem term_2089_is_16 : sequence 2088 = 16 := 
by 
  -- This is where the formal proof would go
  sorry

end term_2089_is_16_l224_224390


namespace height_difference_l224_224727

noncomputable def rod_diameter : ℝ := 8
noncomputable def num_rods : ℕ := 150
noncomputable def height_vertical (d : ℝ) (n : ℕ) : ℝ := n * d
noncomputable def height_hexagonal_close_packing (d : ℝ) (n : ℕ) : ℝ := (n-1) * (d * (Real.sqrt 3 / 2)) + d

theorem height_difference (rod_diameter : ℝ) (num_rods : ℕ) :
  rod_diameter = 8 ∧ num_rods = 150 →
  ∣ height_vertical rod_diameter num_rods - height_hexagonal_close_packing rod_diameter num_rods ∣ = 1192 - 596 * Real.sqrt 3 :=
by
  sorry

end height_difference_l224_224727


namespace arithmetic_sequence_sum_zero_l224_224530

theorem arithmetic_sequence_sum_zero {a1 d n : ℤ} 
(h1 : a1 = 35) 
(h2 : d = -2) 
(h3 : (n * (2 * a1 + (n - 1) * d)) / 2 = 0) : 
n = 36 :=
by sorry

end arithmetic_sequence_sum_zero_l224_224530


namespace balloon_arrangements_l224_224885

noncomputable def factorial (n : ℕ) : ℕ :=
if h : 0 < n then n * factorial (n - 1) else 1

theorem balloon_arrangements : 
  ∃ (n : ℕ), n = (factorial 7) / (factorial 2 * factorial 2) ∧ n = 1260 := by
  have fact_7 := factorial 7
  have fact_2 := factorial 2
  exists (fact_7 / (fact_2 * fact_2))
  split
  · rw [← int.coe_nat_eq_coe_nat_iff]
    norm_cast
    sorry -- Here we do the exact calculations
  · exact rfl

end balloon_arrangements_l224_224885


namespace directrix_of_parabola_l224_224686

theorem directrix_of_parabola (x y : ℝ) (h : y = (1/4) * x^2) : y = -1 :=
sorry

end directrix_of_parabola_l224_224686


namespace count_even_distinct_digits_200_to_500_l224_224172

/-- Given a number is between 200 and 500, is even, and has distinct digits, 
prove that the total number of such three-digit numbers is 120. -/
theorem count_even_distinct_digits_200_to_500 : 
  ∃ n, n = 120 ∧ 
  ∀ (x : ℕ), 200 ≤ x ∧ x < 500 ∧ even x ∧ distinct_digits x → valid_number x :=
sorry

end count_even_distinct_digits_200_to_500_l224_224172


namespace balloon_arrangements_l224_224882

noncomputable def factorial (n : ℕ) : ℕ :=
if h : 0 < n then n * factorial (n - 1) else 1

theorem balloon_arrangements : 
  ∃ (n : ℕ), n = (factorial 7) / (factorial 2 * factorial 2) ∧ n = 1260 := by
  have fact_7 := factorial 7
  have fact_2 := factorial 2
  exists (fact_7 / (fact_2 * fact_2))
  split
  · rw [← int.coe_nat_eq_coe_nat_iff]
    norm_cast
    sorry -- Here we do the exact calculations
  · exact rfl

end balloon_arrangements_l224_224882


namespace sheep_count_l224_224347

theorem sheep_count (cows sheep shepherds : ℕ) 
  (h_cows : cows = 12) 
  (h_ears : 2 * cows < sheep) 
  (h_legs : sheep < 4 * cows) 
  (h_shepherds : sheep = 12 * shepherds) :
  sheep = 36 :=
by {
  sorry
}

end sheep_count_l224_224347


namespace number_of_ks_l224_224122

theorem number_of_ks (M N : ℕ) (hM : N = 267000) (h_div : ∀ k, k ≤ N → (k^2 - 1) % 267 = 0 ↔ ∃ m, k = 267 * m + 1 ∨ k = 267 * m + 88 ∨ k = 267 * m + 177 ∨ k = 267 * m + 266) : 
  (finset.range (N + 1)).filter (λ k, (k^2 - 1) % 267 = 0).card = 4000 := 
by
  sorry

end number_of_ks_l224_224122


namespace license_plate_count_l224_224560

/-- Number of valid license plates given specific conditions. -/
theorem license_plate_count :
  let letters := 26 in
  let prime_digits := 4 in
  let non_prime_digits := 6 in
  (letters * letters * (prime_digits * non_prime_digits) = 16224) :=
by
  let letters := 26
  let prime_digits := 4
  let non_prime_digits := 6
  have total_combinations : letters * letters * (prime_digits * non_prime_digits) = 16224 := by
    sorry
  exact total_combinations

end license_plate_count_l224_224560


namespace find_gain_approx_30_l224_224860

-- Define the cost price (C) of one pencil.
def cost_price (C : ℝ) : Prop :=
  1 = 0.7 * (20 * C)

-- Define gain (G) when selling 10.77 pencils for a rupee.
def gain (C G : ℝ) : Prop :=
  1 = (10.77 * C) * (1 + G / 100)

-- The theorem to prove that the gain is approximately 30% given the conditions.
theorem find_gain_approx_30 (C G : ℝ) (hC : cost_price C) (hG : gain C G) : G ≈ 30 :=
by
  -- Here, we assume the necessary structure for ≈ (approximation) is defined
  sorry

end find_gain_approx_30_l224_224860


namespace triangle_not_necessarily_isosceles_l224_224309

theorem triangle_not_necessarily_isosceles
  (A B C A1 B1 I A2 B2 : Type)
  [triangle : Triangle.contains A B C]
  (A_A1 : Line.contains A A1)
  (B_B1 : Line.contains B B1)
  (iso_A2 : Isosceles △(A1 I) A2)
  (iso_B2 : Isosceles △(B1 I) B2)
  (lineCI_bisects : Bisects (Line.contains C I) (Segment.contains A2 B2)) :
  ¬Isosceles △(A B C) := sorry

end triangle_not_necessarily_isosceles_l224_224309


namespace digit_in_ten_thousandths_place_l224_224016

theorem digit_in_ten_thousandths_place : 
  (let decimal_eq := 7 / 32 in 
  let ten_thousandths_place := (decimal_eq * 100000) % 10 in 
  ten_thousandths_place = 5) := 
begin
  sorry -- not providing the proof here
end

end digit_in_ten_thousandths_place_l224_224016


namespace work_lifting_satellite_l224_224992

noncomputable def work_done (m H R g : ℝ) : ℝ :=
  let u := 1 + H / R
  in -m * g * R^2 * (1 / u - 1)

theorem work_lifting_satellite :
  work_done 7000 (250 * 1000) (6380 * 1000) 10 = 1.72 * 10^10 := by
  sorry

end work_lifting_satellite_l224_224992


namespace smallest_repeating_block_3_over_8_l224_224556

noncomputable def smallest_repeating_block_length : ℕ :=
  let decExpand := 0.375 in
  if ∃ n m : ℕ, n ≠ m ∧ sublist (take n (nat.digits 10 375)) (drop m (nat.digits 10 375)) then n else 0

theorem smallest_repeating_block_3_over_8 :
  smallest_repeating_block_length = 0 := 
sorry

end smallest_repeating_block_3_over_8_l224_224556


namespace balloon_permutations_l224_224935

theorem balloon_permutations : 
  let balloons := "BALLOON".toList.length in
  let total_permutations := Nat.factorial balloons in
  let repetitive_L := Nat.factorial 2 in
  let repetitive_O := Nat.factorial 2 in
  (total_permutations / (repetitive_L * repetitive_O)) = 1260 := sorry

end balloon_permutations_l224_224935


namespace calculate_E_l224_224795

theorem calculate_E (P J T B A E : ℝ) 
  (h1 : J = 0.75 * P)
  (h2 : J = 0.80 * T)
  (h3 : B = 1.40 * J)
  (h4 : A = 0.85 * B)
  (h5 : T = P - (E / 100) * P)
  (h6 : E = 2 * ((P - A) / P) * 100) : 
  E = 21.5 := 
sorry

end calculate_E_l224_224795


namespace neznaika_statement_false_l224_224426

-- Define points for victory, draw, and defeat
def victory_points := 1
def draw_points := 0.5
def defeat_points := 0

-- Define a function to calculate the point difference after a list of outcomes
def point_difference (outcomes : List ℕ) : ℤ :=
  outcomes.foldl (λ diff outcome,
    match outcome with
    | 0 => diff   -- defeat contributes 0 points, difference decreases by 1
    | 1 => diff + 1  -- victory contributes 1 point, difference increases by 1
    | 2 => diff    -- draw contributes 0.5 points gained and 0.5 points "lost", no effect on difference
    | _ => diff   -- invalid outcome does nothing
  ) 0

-- Define the theorem stating that the point difference cannot be 3.5
theorem neznaika_statement_false :
  ¬ (∃ outcomes : List ℕ, point_difference outcomes = 3.5) := by
  sorry

end neznaika_statement_false_l224_224426


namespace t_range_l224_224398

noncomputable def f (x : ℝ) : ℝ :=
if h₀ : x ∈ [0, 1) then x^2 - x
else if h₁ : x ∈ [1, 2) then -((1 / 2) ^ |x - 1.5|)
else if h₂ : x ∈ [-2, 0) then f (x+2) / 2
else if h₃ : x ∈ [-4, -2) then f (x+4) / 4
else 0

theorem t_range (t : ℝ) :
(∀ x ∈ Ico (-4:ℝ) (-2), f x ≥ (t^2 / 4 - t + 1/2)) ↔ (1 ≤ t ∧ t ≤ 3) :=
sorry

end t_range_l224_224398


namespace number_of_factors_l224_224793

-- Definitions and assumptions given in the problem
def positive_integer (a : ℕ) : Prop := a > 0
def greater_than_one (a : ℕ) : Prop := a > 1
def factors_product_eq_cube (a : ℕ) (f : ℕ) : Prop :=
  ∃ (d : Fin f → ℕ), (∏ i, d i) = a^3 ∧ (∀ i, a % d i = 0)

-- Main theorem stating the proof problem
theorem number_of_factors (a : ℕ) (f : ℕ) 
  (h1 : positive_integer a) 
  (h2 : greater_than_one a) 
  (h3 : factors_product_eq_cube a f) : 
  f = 6 :=
sorry

end number_of_factors_l224_224793


namespace proof_l224_224549

open Set

variable (U M P : Set ℕ)

noncomputable def prob_statement : Prop :=
  let C_U (A : Set ℕ) : Set ℕ := {x ∈ U | x ∉ A}
  U = {1,2,3,4,5,6,7,8} ∧ M = {2,3,4} ∧ P = {1,3,6} ∧ C_U (M ∪ P) = {5,7,8}

theorem proof : prob_statement {1,2,3,4,5,6,7,8} {2,3,4} {1,3,6} :=
by
  sorry

end proof_l224_224549


namespace sin_sum_to_product_identity_l224_224732

theorem sin_sum_to_product_identity (α β : ℝ) (h : α + β ≤ 180) : 
sin α + sin β = 2 * sin ((α + β) / 2) * cos ((α - β) / 2) :=
sorry

end sin_sum_to_product_identity_l224_224732


namespace max_angle_A_and_area_l224_224581

-- Definitions for the problem conditions
variables (A B C : ℝ) (a b c : ℝ) (m n : ℝ × ℝ)

-- Condition: angles sum up to π
axiom angle_sum : A + B + C = Real.pi

-- Condition: side opposite to angle A is 2
axiom side_a : a = 2

-- Definitions for vectors m and n
noncomputable def vec_m : ℝ × ℝ :=
  (2, 2 * Real.cos (B + C) / 2 ^ 2 - 1)

noncomputable def vec_n : ℝ × ℝ :=
  (Real.sin (A / 2), -1)

-- Dot product of m and n
noncomputable def dot_product (v1 v2 : ℝ × ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

-- Given the conditions, we want to prove the following:
theorem max_angle_A_and_area :
  (dot_product vec_m vec_n A B C) (a = 2) ∧ 
  ((A = Real.pi / 3) ∧ (∀ (b c : ℝ),
    b = 2 → c = 2 →
    (A = Real.pi / 3 ∧ (1 / 2 * b * c * Real.sin(A) = Real.sqrt 3)) )) :=
sorry

end max_angle_A_and_area_l224_224581


namespace f_five_eq_zero_l224_224324

noncomputable def f : ℝ → ℝ := sorry

axiom f_mul (x y : ℝ) : f(x * y) = f(x) * f(y)
axiom f_zero_ne_zero : f 0 ≠ 0
axiom f_one : f 1 = 2

theorem f_five_eq_zero : f 5 = 0 :=
by
  sorry

end f_five_eq_zero_l224_224324


namespace system1_solution_l224_224674

theorem system1_solution (x y : ℝ) (h₁ : x = 2 * y) (h₂ : 3 * x - 2 * y = 8) : x = 4 ∧ y = 2 := 
by admit

end system1_solution_l224_224674


namespace range_of_a_l224_224579

theorem range_of_a (a : ℝ) : 
  (2 * (-1) + 0 + a) * (2 * 2 + (-1) + a) < 0 ↔ -3 < a ∧ a < 2 := 
by 
  sorry

end range_of_a_l224_224579


namespace cubes_sum_eq_ten_squared_l224_224370

theorem cubes_sum_eq_ten_squared : 1^3 + 2^3 + 3^3 + 4^3 = 10^2 := by
  sorry

end cubes_sum_eq_ten_squared_l224_224370


namespace age_difference_l224_224661

theorem age_difference (p f : ℕ) (hp : p = 11) (hf : f = 42) : f - p = 31 :=
by
  sorry

end age_difference_l224_224661


namespace polar_eq_C1_distance_AB_l224_224207

theorem polar_eq_C1 (θ : ℝ) : 
  (∃ (ρ : ℝ), ρ = 6 * Real.cos θ ∧ (3 + 3 * Real.cos θ)^2 + (3 * Real.sin θ)^2 = ρ^2) :=
sorry

theorem distance_AB : 
  let ρ1 := 6 * Real.cos (Real.pi / 3) 
  let ρ2 := (Real.sqrt 3) * Real.sin (Real.pi / 3) + Real.cos (Real.pi / 3)
  in abs (ρ1 - ρ2) = 1 :=
sorry

end polar_eq_C1_distance_AB_l224_224207


namespace option_B_same_function_l224_224430

-- Definition of functions for Option B
def f_B (x : ℝ) := 2 * abs x
def g_B (x : ℝ) := sqrt (4 * x ^ 2)

-- Lean 4 statement to check if f_B and g_B represent the same function
theorem option_B_same_function : ∀ x : ℝ, f_B x = g_B x :=
by
  intros x
  sorry

end option_B_same_function_l224_224430


namespace explain_judge_statement_with_twins_l224_224728

-- Defining the conditions
def two_people_on_trial_murder : Prop := true
def one_found_guilty : Prop := true
def one_found_innocent : Prop := true
def judge_statement : Prop :=
  ∀ (guilty : Prop) (innocent : Prop), guilty ∧ ¬innocent → must_set_free(guilty)

-- The unexpected statement must hold under the condition that the guilty and innocent are Siamese twins
def must_set_free (guilty: Prop) : Prop := guilty → false

-- The proof problem
theorem explain_judge_statement_with_twins :
  two_people_on_trial_murder →
  one_found_guilty →
  one_found_innocent →
  judge_statement →
  (∃ (twin1 twin2 : Prop), twin1 = twin2) :=
by
  intros h1 h2 h3 h4
  -- Assuming the defendants are Siamese twins
  exact ⟨one_found_guilty, one_found_innocent, sorry⟩

end explain_judge_statement_with_twins_l224_224728


namespace balloon_permutation_count_l224_224923

-- Define the given conditions:
def balloon_letters : List Char := ['B', 'A', 'L', 'L', 'O', 'O', 'N']
def total_letters : Nat := 7
def count_L : Nat := 2
def count_O : Nat := 2

-- Statement to prove:
theorem balloon_permutation_count :
  (nat.factorial total_letters) / ((nat.factorial count_L) * (nat.factorial count_O)) = 1260 := by
  sorry

end balloon_permutation_count_l224_224923


namespace grasshopper_visit_all_points_min_jumps_l224_224341

noncomputable def grasshopper_min_jumps : ℕ := 18

theorem grasshopper_visit_all_points_min_jumps (n m : ℕ) (h₁ : n = 2014) (h₂ : m = 18) :
  ∃ k : ℕ, k ≤ m ∧ (∀ i : ℤ, 0 ≤ i → i < n → ∃ j : ℕ, j < k ∧ (j * 57 + i * 10) % n = i) :=
sorry

end grasshopper_visit_all_points_min_jumps_l224_224341


namespace total_number_of_triangles_is_twenty_l224_224867

-- Define the essential elements: a rectangle with midpoints and diagonals
variables (A B C D M N P Q O : Type) [AffineSpace ℝ A]
variables [AffineSpace ℝ B] [AffineSpace ℝ C] [AffineSpace ℝ D]
variables [AffineSpace ℝ M] [AffineSpace ℝ N] [AffineSpace ℝ P] [AffineSpace ℝ Q] [AffineSpace ℝ O]

-- Conditions
def is_rectangle (A B C D : Type) : Prop := True -- Assuming pre-existing definitions or properties
def has_midpoints (M N P Q : Type) (A B C D : Type) : Prop := True -- Assuming pre-existing definitions or properties
def diagonals_intersect_at_center (O : Type) (A B C D : Type) : Prop := True -- Assuming pre-existing definitions or properties

-- Proof Problem
theorem total_number_of_triangles_is_twenty
  (h1 : is_rectangle A B C D)
  (h2 : has_midpoints M N P Q A B C D)
  (h3 : diagonals_intersect_at_center O A B C D) :
  ∑ (t : Triangle), 1 = 20 := 
  sorry  -- Proof not provided, just the statement

end total_number_of_triangles_is_twenty_l224_224867


namespace exponential_function_base_l224_224577

theorem exponential_function_base (a b : ℝ) (h1 : 0 < a ∧ a ≠ 1)
  (h2 : ∀ x ∈ (set.Icc (-2:ℝ) 1), f x = a^x)
  (h3 : (∀ x : ℝ, g x = (2 - 7 * b) * x) ∧ ∀ x, g x ≤ g 0) 
  (h4 : f(-2) = b ∧ f(1) = 4) :
  a = 1 / 2 := 
sorry

end exponential_function_base_l224_224577


namespace probability_not_yellow_l224_224814

-- Define the conditions
def red_jelly_beans : Nat := 4
def green_jelly_beans : Nat := 7
def yellow_jelly_beans : Nat := 9
def blue_jelly_beans : Nat := 10

-- Definitions used in the proof problem
def total_jelly_beans : Nat := red_jelly_beans + green_jelly_beans + yellow_jelly_beans + blue_jelly_beans
def non_yellow_jelly_beans : Nat := total_jelly_beans - yellow_jelly_beans

-- Lean statement of the probability problem
theorem probability_not_yellow : 
  (non_yellow_jelly_beans : ℚ) / (total_jelly_beans : ℚ) = 7 / 10 := 
by 
  sorry

end probability_not_yellow_l224_224814


namespace balloon_arrangements_l224_224894

theorem balloon_arrangements :
  let n := 7
  let k1 := 2
  let k2 := 2
  (Nat.factorial n) / ((Nat.factorial k1) * (Nat.factorial k2)) = 1260 :=
by
  -- Use the conditions directly from the problem
  let n := 7
  let k1 := 2
  let k2 := 2
  -- Mathematically, the number of arrangements of the multiset
  have h : (Nat.factorial n) / ((Nat.factorial k1) * (Nat.factorial k2)) = 1260 := sorry
  exact h

end balloon_arrangements_l224_224894


namespace smallest_n_condition_l224_224126

open Nat

-- Define the sum of squares formula
noncomputable def sum_of_squares (n : ℕ) : ℕ :=
  (n * (n + 1) * (2 * n + 1)) / 6

-- Define the condition for being a square number
def is_square (m : ℕ) : Prop :=
  ∃ k : ℕ, k * k = m

-- The proof problem statement
theorem smallest_n_condition : 
  ∃ n : ℕ, n > 1 ∧ is_square (sum_of_squares n / n) ∧ (∀ m : ℕ, m > 1 ∧ is_square (sum_of_squares m / m) → n ≤ m) :=
sorry

end smallest_n_condition_l224_224126


namespace part1_simplified_part2_value_part3_independent_l224_224505

-- Definitions of A and B
def A (x y : ℝ) : ℝ := 3 * x^2 - x + 2 * y - 4 * x * y
def B (x y : ℝ) : ℝ := 2 * x^2 - 3 * x - y + x * y

-- Proof statement for part 1
theorem part1_simplified (x y : ℝ) :
  2 * A x y - 3 * B x y = 7*x + 7*y - 11*x*y :=
by sorry

-- Proof statement for part 2
theorem part2_value (x y : ℝ) (hxy : x + y = 6/7) (hprod : x * y = -1) :
  2 * A x y - 3 * B x y = 17 :=
by sorry

-- Proof statement for part 3
theorem part3_independent (y : ℝ) :
  2 * A (7/11) y - 3 * B (7/11) y = 49/11 :=
by sorry

end part1_simplified_part2_value_part3_independent_l224_224505


namespace functions_of_same_family_count_l224_224182

theorem functions_of_same_family_count : 
  (∃ (y : ℝ → ℝ), ∀ x, y x = x^2) ∧ 
  (∃ (range_set : Set ℝ), range_set = {1, 2}) → 
  ∃ n, n = 9 :=
by
  sorry

end functions_of_same_family_count_l224_224182


namespace percentage_lee_june_approx_60_l224_224582

noncomputable def percentage_june (income_lee_may total_income_may earnings_mr_may earnings_jack_may : ℝ) : ℝ :=
  let earnings_lee_june := 1.2 * income_lee_may
  let total_income_june := earnings_lee_june + 1.1 * earnings_mr_may + 0.85 * earnings_jack_may + (total_income_may - income_lee_may - earnings_mr_may - earnings_jack_may)
  100 * earnings_lee_june / total_income_june

theorem percentage_lee_june_approx_60
  (T L M J : ℝ)
  (hL : L = 0.5 * T)
  (hJ : 0 < M ∧ 0 < J) : percentage_june L T M J ≈ 60 :=
by
  have hL_june : 1.2 * L = 0.6 * T, from mul_eq_mul_left_of_ne_zero (by linarith [hL]) (by norm_num)
  have total_income_june := 1.1 * T + 0.1 * M - 0.15 * J
  have : percentage_june L T M J =  60 :=
    sorry
  exact this

end percentage_lee_june_approx_60_l224_224582


namespace volume_of_cylinder_cut_l224_224490

open Real

noncomputable def cylinder_cut_volume (R α : ℝ) : ℝ :=
  (2 / 3) * R^3 * tan α

theorem volume_of_cylinder_cut (R α : ℝ) :
  cylinder_cut_volume R α = (2 / 3) * R^3 * tan α :=
by
  sorry

end volume_of_cylinder_cut_l224_224490


namespace interest_credited_is_63_cents_l224_224434

theorem interest_credited_is_63_cents :
  ∀ (P : ℝ) (r : ℝ) (t : ℝ) (additional_deposit : ℝ) (final_amount : ℝ),
  P = 500 → r = 0.03 → t = 3/12 → additional_deposit = 15 → final_amount = 515.63 →
  let interest := 100 * (final_amount - (P + additional_deposit)) in
  interest = 63 :=
by
  intros P r t additional_deposit final_amount hP hr ht hadd hfinal
  have h1 : P + (P * r * t) + additional_deposit = 500 + (500 * 0.03 * (3/12)) + 15 := by
    rw [hP, hr, ht, hadd]
  have h2 : final_amount = 515.63 := hfinal
  have hinterest : 100 * (final_amount - (P + additional_deposit)) = 63 := 
    sorry -- Proof omitted
  exact hinterest

end interest_credited_is_63_cents_l224_224434


namespace exponent_equation_l224_224566

theorem exponent_equation (m n : ℝ) (h1 : 2^m = 3) (h2 : 2^n = 2) : 2^(2*m + 2*n) = 36 := 
by 
  sorry

end exponent_equation_l224_224566


namespace AB_parallel_to_plane_l224_224629

open Point Geometry Affine

-- Lean definition for non-coplanar points and midpoints
structure NotCoplanar (A B C D : Point) : Prop :=
(non_collinear : ¬Collinear ℝ (Set.insert A {B, C, D}))

structure Midpoint (M A D : Point) : Prop :=
(is_midpoint : 2 • M = A + D)

noncomputable def line_parallel_to_plane (A B C D M N K : Point)
  (h_non_coplanar : NotCoplanar A B C D)
  (hM : Midpoint M A D)
  (hN : Midpoint N B D)
  (hK : Midpoint K C D) : Prop :=
Parallel (line_through A B) (plane_through M N K)

-- Statement of the problem as a theorem in Lean 4
theorem AB_parallel_to_plane (A B C D M N K : Point)
  (h_non_coplanar : NotCoplanar A B C D)
  (hM : Midpoint M A D)
  (hN : Midpoint N B D)
  (hK : Midpoint K C D) :
  line_parallel_to_plane A B C D M N K h_non_coplanar hM hN hK :=
sorry

end AB_parallel_to_plane_l224_224629


namespace paul_needs_score_to_achieve_mean_l224_224275

theorem paul_needs_score_to_achieve_mean (x : ℤ) :
  (78 + 84 + 76 + 82 + 88 + x) / 6 = 85 → x = 102 :=
by 
  sorry

end paul_needs_score_to_achieve_mean_l224_224275


namespace balloon_permutations_l224_224936

theorem balloon_permutations : 
  let balloons := "BALLOON".toList.length in
  let total_permutations := Nat.factorial balloons in
  let repetitive_L := Nat.factorial 2 in
  let repetitive_O := Nat.factorial 2 in
  (total_permutations / (repetitive_L * repetitive_O)) = 1260 := sorry

end balloon_permutations_l224_224936


namespace total_spent_l224_224405

theorem total_spent (bracelet_price keychain_price coloring_book_price : ℕ)
  (paula_bracelets paula_keychains olive_coloring_books olive_bracelets : ℕ)
  (total : ℕ) :
  bracelet_price = 4 →
  keychain_price = 5 →
  coloring_book_price = 3 →
  paula_bracelets = 2 →
  paula_keychains = 1 →
  olive_coloring_books = 1 →
  olive_bracelets = 1 →
  total = paula_bracelets * bracelet_price + paula_keychains * keychain_price +
          olive_coloring_books * coloring_book_price + olive_bracelets * bracelet_price →
  total = 20 :=
by sorry

end total_spent_l224_224405


namespace contradiction_example_l224_224729

theorem contradiction_example (a b c d : ℝ) 
  (h1 : a + b = 1) 
  (h2 : c + d = 1) 
  (h3 : ac + bd > 1) : 
  ¬ (a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0) → 
  a < 0 ∨ b < 0 ∨ c < 0 ∨ d < 0 :=
by
  sorry

end contradiction_example_l224_224729


namespace balloon_arrangement_count_l224_224973

-- Definitions of letter frequencies for the word BALLOON
def letter_frequencies : List (Char × Nat) := [('B', 1), ('A', 1), ('L', 2), ('O', 2), ('N', 1)]

-- Total number of letters
def total_letters := 7

-- The formula for the number of unique arrangements of the letters
noncomputable def arrangements :=
  (Nat.factorial total_letters) / 
  (Nat.factorial 1 * Nat.factorial 1 * Nat.factorial 2 * Nat.factorial 2 * Nat.factorial 1)

-- The theorem to prove the number of ways to arrange the letters in "BALLOON"
theorem balloon_arrangement_count : arrangements = 1260 :=
  sorry

end balloon_arrangement_count_l224_224973


namespace probability_of_drawing_white_l224_224062

-- Definitions of the conditions
def initial_urn : Type := { a // 0 ≤ a ∧ a ≤ 2 }
def urn_after_adding_white (b : initial_urn) := b.val + 1
def possible_initial_states := [0, 1, 2]

-- Probabilities tied to initial hypotheses
def prob_b1 : ℝ := 1 / 3
def prob_b2 : ℝ := 1 / 3
def prob_b3 : ℝ := 1 / 3

-- Conditional probabilities given each hypothesis
def P_A_given_B1 : ℝ := 1 / 3
def P_A_given_B2 : ℝ := 2 / 3
def P_A_given_B3 : ℝ := 1

-- Law of Total Probability
def P_A : ℝ := 
  prob_b1 * P_A_given_B1 + 
  prob_b2 * P_A_given_B2 + 
  prob_b3 * P_A_given_B3

theorem probability_of_drawing_white :
  P_A = 2 / 3 :=
sorry

end probability_of_drawing_white_l224_224062


namespace repeating_decimal_eq_l224_224765

noncomputable def repeating_decimal : ℚ := 56 / 99

theorem repeating_decimal_eq : ∃ x : ℚ, x = repeating_decimal ∧ x = 56 / 99 :=
by
  use 56 / 99
  split
  all_goals { sorry }

end repeating_decimal_eq_l224_224765


namespace sin_of_angle_given_point_l224_224188

theorem sin_of_angle_given_point :
  let x := 1
  let y := -Real.sqrt 3
  let r := Real.sqrt (x ^ 2 + y ^ 2)
  r = 2
  ∃ α : ℝ, sin α = y / r := sorry

end sin_of_angle_given_point_l224_224188


namespace proof_statements_BCD_l224_224568

variable (a b : ℝ)

theorem proof_statements_BCD (h1 : a > b) (h2 : b > 0) :
  (-1 / b < -1 / a) ∧ (a^2 * b > a * b^2) ∧ (a / b > b / a) :=
by
  sorry

end proof_statements_BCD_l224_224568


namespace balloon_permutations_l224_224956

theorem balloon_permutations : (∀ (arrangements : ℕ),
  arrangements = nat.factorial 7 / (nat.factorial 2 * nat.factorial 2) → arrangements = 1260) :=
begin
  intros arrangements h,
  rw [nat.factorial_succ, nat.factorial],
  sorry,
end

end balloon_permutations_l224_224956


namespace slope_product_l224_224868

theorem slope_product (p q : Real) 
  (h1 : ∀ θ1 θ2, tan θ1 = p → tan θ2 = q → θ1 = 3 * θ2)
  (h2 : p = -q / 2) 
  (h3 : q > 0) 
  (h4 : p ≠ 0 ∧ q ≠ 0) : 
  p * q = -7 / 10 :=
by
  sorry

end slope_product_l224_224868


namespace domain_of_function_l224_224116

theorem domain_of_function :
  {x : ℝ | x < -1 ∨ 4 ≤ x} = {x : ℝ | (x^2 - 7*x + 12) / (x^2 - 2*x - 3) ≥ 0} \ {3} :=
by
  sorry

end domain_of_function_l224_224116


namespace calculate_total_cost_l224_224705

variables (p n : ℝ)

def cost_condition_1 : Prop := 10 * p + 6 * n = 3.50
def cost_condition_2 : Prop := 4 * p + 9 * n = 2.70
def unit_condition : Prop := 24 + 15 > 15

theorem calculate_total_cost : 
  cost_condition_1 p n → 
  cost_condition_2 p n → 
  unit_condition →
  24 * p + 15 * n + 0.50 = 9.02 :=
by
  intros,
  sorry

end calculate_total_cost_l224_224705


namespace repeating_decimal_fraction_l224_224785

theorem repeating_decimal_fraction : ∀ x : ℚ, (x = 0.5656565656565656) → 100 * x = 56.5656565656565656 → 100 * x - x = 56.5656565656565656 - 0.5656565656565656
  → 99 * x = 56 → x = 56 / 99 :=
begin
  intros x h1 h2 h3 h4,
  sorry,
end

end repeating_decimal_fraction_l224_224785


namespace area_of_triangle_l224_224693

noncomputable def hyperbola := {a b : ℝ // a ≠ 0 ∧ b ≠ 0 ∧ (b^2 = (2 : ℝ)^2 - a^2)}

noncomputable def parabola := {x y : ℝ // y^2 = 2 * x}

theorem area_of_triangle
  (x y : ℝ)
  (h1 : y = x^2 - 6*x + 8)
  (h2 : 2 * 1 = 2)
  (h3 : 2 * 2 = 4)
  (h4 : ∃ ℝ h1 h2 h3, hyperbola)
  (h5 : ∃ (x1 y1 x2 y2 : ℝ), (parabola ∧ (y1 = sqrt 3 * x1) ∧ (x1 = 2 / 3) ∧ (y1 = 2 * sqrt 3 / 3) ∧
                             (y2 = -sqrt 3 * x2) ∧ (x2 = 2 / 3) ∧ (y2 = -2 * sqrt 3 / 3)))
  : area_of_triangle = 4 * sqrt 3 / 9 := 
sorry

end area_of_triangle_l224_224693


namespace Sherman_weekly_driving_time_l224_224667

theorem Sherman_weekly_driving_time (daily_commute : ℕ := 30) (weekend_drive : ℕ := 2) : 
  (5 * (2 * daily_commute) / 60 + 2 * weekend_drive) = 9 := 
by
  sorry

end Sherman_weekly_driving_time_l224_224667


namespace kolya_is_collection_agency_l224_224212

structure Condition :=
  (katya_lent_books_to_vasya : ∀ books : Type, katya_lent books vasya)
  (vasya_failed_to_return_books : ∀ (books : Type) (month_later : Time), failed_return books vasya month_later)
  (katya_asked_kolya_to_retrieve : ∀ books : Type, katya_asks books katya kolya vaaya)
  (kolya_agrees_for_reward : ∀ book : Type, kolya_agrees book katya_retrieve books)

theorem kolya_is_collection_agency {books : Type} (h : Condition books) : 
  KolyaRole = CollectionAgency :=
sorry

end kolya_is_collection_agency_l224_224212


namespace predicted_population_increase_is_40_percent_l224_224334

noncomputable def mojave_population_increase := 
  let initial_population := 4000
  let current_population := initial_population * 3
  let future_population := 16800
  let percentage_increase := ((future_population - current_population).toReal / current_population.toReal) * 100
  percentage_increase

theorem predicted_population_increase_is_40_percent : mojave_population_increase = 40 := by
  sorry

end predicted_population_increase_is_40_percent_l224_224334


namespace speedPossibilities_l224_224008

noncomputable def speedOfSecondPedestrian (S_0 v_1 t S: ℝ) : Set ℝ :=
  {v_2 | v_2 = (v_1 * t - S_0 - S) / t ∨ v_2 = (v_1 * t - S_0 + S) / t}

theorem speedPossibilities 
  (S_0 : ℝ) 
  (v_1 : ℝ) 
  (t : ℝ) 
  (S : ℝ) 
  (v2_1 v2_2 : ℝ) :
  S_0 = 200 ∧ 
  v_1 = 7 ∧ 
  t = 5 * 60 ∧ 
  S = 100 ∧ 
  v2_1 = 6 ∧ 
  v2_2 ≈ 6.67 →
  speedOfSecondPedestrian S_0 v_1 t S = {v2_1, v2_2} :=
begin
  sorry
end

end speedPossibilities_l224_224008


namespace misread_number_is_correct_l224_224310

-- Definitions for the given conditions
def avg_incorrect : ℕ := 19
def incorrect_number : ℕ := 26
def avg_correct : ℕ := 24

-- Statement to prove the actual number that was misread
theorem misread_number_is_correct (x : ℕ) (h : 10 * avg_correct - 10 * avg_incorrect = x - incorrect_number) : x = 76 :=
by {
  sorry
}

end misread_number_is_correct_l224_224310


namespace vector_dot_product_l224_224543

-- Define point P
def P : (ℝ × ℝ) := (2, 1)

-- Define the function f
def f (x : ℝ) : ℝ := (2 * x + 2) / (2 * x - 4)

-- Define the vectors OA and OB assuming their properties from intersections
def OA (x1 y1 : ℝ) : (ℝ × ℝ) := (x1, y1)
def OB (x2 y2 : ℝ) : (ℝ × ℝ) := (x2, y2)

-- Define the vector OP
def OP : (ℝ × ℝ) := (2, 1)

-- The dot product function
def dotProduct (v1 v2 : (ℝ × ℝ)) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

-- Main theorem statement:
theorem vector_dot_product : ∃ (x1 x2 y1 y2 : ℝ), 
  ((y1 = f x1) ∧ (y2 = f x2) ∧ 
  ((x1 + x2) = 4) ∧ 
  ((y1 + y2) = 2)) ∧ 
  dotProduct (OA x1 y1 + OB x2 y2) OP = 10 :=
by 
  sorry

end vector_dot_product_l224_224543


namespace ferris_wheel_seat_calculation_l224_224391

theorem ferris_wheel_seat_calculation (n k : ℕ) (h1 : n = 4) (h2 : k = 2) : n / k = 2 := 
by
  sorry

end ferris_wheel_seat_calculation_l224_224391


namespace num_positive_integers_satisfying_inequality_l224_224496

theorem num_positive_integers_satisfying_inequality :
  {x : ℕ | 0 < x ∧ 30 < x^2 + 10 * x + 25 ∧ x^2 + 10 * x + 25 < 60}.to_finset.card = 2 :=
by sorry

end num_positive_integers_satisfying_inequality_l224_224496


namespace average_postcards_collected_per_day_l224_224866

theorem average_postcards_collected_per_day 
    (a : ℕ) (d : ℕ) (n : ℕ) 
    (h_a : a = 10)
    (h_d : d = 12)
    (h_n : n = 7) :
    (a + (a + (n - 1) * d)) / 2 = 46 := by
  sorry

end average_postcards_collected_per_day_l224_224866


namespace unique_functional_equation_l224_224998

theorem unique_functional_equation :
  ∀ (f : ℚ⁺ → ℚ⁺), 
  (∀ x y : ℚ⁺, f (x^2 * (f y) ^ 2) = (f x) ^ 2 * f y) → 
  (∀ x : ℚ⁺, f x = 1) :=
by
  intros f h
  sorry

end unique_functional_equation_l224_224998


namespace part_a_part_b_l224_224068

variables {V : Type} [Fintype V] (T : SimpleGraph V)

structure BipartiteCovering :=
(V1 V2 : Finset V)
(h_partition : ∀ v, v ∈ V1 ∨ v ∈ V2)
(h_disjoint : ∀ v, v ∈ V1 → v ∉ V2)
(h_covering : ∀ {v w}, T.adj v w → (v ∈ V1 ∧ w ∈ V2) ∨ (v ∈ V2 ∧ w ∈ V1))

noncomputable def minimum_moves 
(V1 V2 : Finset V)
(d : V → V → ℕ)
: ℕ :=
  @inf ℕ _ _ (finset.univ.perm V).val (λ σ, (finset.range (Fintype.card V1)).sum (λ i, d (V1.nth_le i sorry) (V2.nth_le (σ i) sorry)))

theorem part_a (h_tree : T.is_tree) (V1 V2 : Finset V)
(h_partition : ∀ v, v ∈ V1 ∨ v ∈ V2)
(h_disjoint : ∀ v, v ∈ V1 → v ∉ V2)
(h_covering : ∀ {v w}, T.adj v w → (v ∈ V1 ∧ w ∈ V2) ∨ (v ∈ V2 ∧ w ∈ V1))
(h_size : V1.card = V2.card):
  true :=
sorry

theorem part_b (h_tree : T.is_tree) (h_Bipartite : BipartiteCovering T)
  (d : V → V → ℕ)
  (h_winning : ∃ n, ∀ P : Finset (V × V), P ⊆ T.edgeSet → P.card = n → ∀ v1 v2 ∈ V, T.connecting_path v1 v2 → (d v1 v2 + d v2 v1) % 2 = 0): 
  true :=
sorry

end part_a_part_b_l224_224068


namespace perfect_square_poly_divides_b2_minus_4ac_l224_224614

theorem perfect_square_poly_divides_b2_minus_4ac
  (a b c : ℤ) (p : ℕ) [hp_prime : Fact (Nat.Prime p)] (hp_odd : p % 2 = 1)
  (h_perfect_square : ∀ x : ℤ, ∃ k : ℤ, f x (x + (2*p - 2)) = k^2 ) :
  p ∣ (b^2 - 4*a*c) := sorry

end perfect_square_poly_divides_b2_minus_4ac_l224_224614


namespace values_of_a_l224_224564

theorem values_of_a (a : ℝ) : 
  ∃a1 a2 : ℝ, 
  (∀ x y : ℝ, (y = 3 * x + a) ∧ (y = x^3 + 3 * a^2) → (x = 0) → (y = 3 * a^2)) →
  ((a = 0) ∨ (a = 1/3)) ∧ 
  ((a1 = 0) ∨ (a1 = 1/3)) ∧
  ((a2 = 0) ∨ (a2 = 1/3)) ∧ 
  (a ≠ a1 ∨ a ≠ a2) ∧ 
  (∃ n : ℤ, n = 2) :=
by sorry

end values_of_a_l224_224564


namespace repeating_decimal_fraction_l224_224780

theorem repeating_decimal_fraction : ∀ x : ℚ, (x = 0.5656565656565656) → 100 * x = 56.5656565656565656 → 100 * x - x = 56.5656565656565656 - 0.5656565656565656
  → 99 * x = 56 → x = 56 / 99 :=
begin
  intros x h1 h2 h3 h4,
  sorry,
end

end repeating_decimal_fraction_l224_224780


namespace directrix_of_parabola_l224_224317

theorem directrix_of_parabola (p : ℝ) : by { assume h : y² = 4 * p * x, sorry } :=
assume h₁ : y² = 2 * x,
have hp : p = 1 / 2 , from sorry,
have directrix_eq : x = -p, from sorry,
show x = -1 / 2, from sorry

end directrix_of_parabola_l224_224317


namespace xy_power_l224_224508

def x : ℚ := 3/4
def y : ℚ := 4/3

theorem xy_power : x^7 * y^8 = 4/3 := by
  sorry

end xy_power_l224_224508


namespace range_of_a_l224_224152

variable (f : ℝ → ℝ)
variable (a : ℝ)

-- Conditions
def decreasing_on (f : ℝ → ℝ) (s : Set ℝ) : Prop := 
  ∀ ⦃x y⦄, x ∈ s → y ∈ s → x < y → f x > f y

def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def condition (f : ℝ → ℝ) (a : ℝ) : Prop :=
  f (1 - a) + f (1 - 2 * a) < 0

-- Theorem statement
theorem range_of_a (h_decreasing : decreasing_on f (Set.Ioo (-1) 1))
                   (h_odd : odd_function f)
                   (h_condition : condition f a) :
  0 < a ∧ a < 2 / 3 :=
sorry

end range_of_a_l224_224152


namespace balloon_permutations_l224_224960

theorem balloon_permutations : (∀ (arrangements : ℕ),
  arrangements = nat.factorial 7 / (nat.factorial 2 * nat.factorial 2) → arrangements = 1260) :=
begin
  intros arrangements h,
  rw [nat.factorial_succ, nat.factorial],
  sorry,
end

end balloon_permutations_l224_224960


namespace last_stage_erased_numbers_l224_224271

theorem last_stage_erased_numbers :
  ∀ (numbers : Set ℕ), (∀ n ∈ numbers, 1 ≤ n ∧ n ≤ 100) →
  (∀ stage remaining,
    (∀ n ∈ remaining, ¬ ∃ d ∈ remaining, d ≠ n ∧ d ∣ n) →
    (∀ next_remaining, (next_remaining = remaining \ {n | ¬ ∃ d ∈ remaining, d ≠ n ∧ d ∣ n})) →
      final_stage remaining = {64, 96}) := sorry

end last_stage_erased_numbers_l224_224271


namespace problem_1_problem_2_problem_3_problem_4_l224_224865

theorem problem_1 : 12 - (-18) + (-7) - 15 = 8 := sorry

theorem problem_2 : -0.5 + (- (3 + 1/4)) + (-2.75) + (7 + 1/2) = 1 := sorry

theorem problem_3 : -2^2 + 3 * (-1)^(2023) - abs (-4) * 5 = -27 := sorry

theorem problem_4 : -3 - (-5 + (1 - 2 * (3 / 5)) / (-2)) = 19 / 10 := sorry

end problem_1_problem_2_problem_3_problem_4_l224_224865


namespace missy_total_watching_time_l224_224266

def num_reality_shows := 5
def length_reality_show := 28
def num_cartoons := 1
def length_cartoon := 10

theorem missy_total_watching_time : 
  (num_reality_shows * length_reality_show + num_cartoons * length_cartoon) = 150 := 
by 
  sorry

end missy_total_watching_time_l224_224266


namespace max_of_a_l224_224710

theorem max_of_a (a b c d : ℝ) (h1 : a ≥ b) (h2 : b ≥ c) (h3 : c ≥ d) (h4 : d > 0)
  (h5 : a + b + c + d = 4) (h6 : a^2 + b^2 + c^2 + d^2 = 8) : a ≤ 1 + Real.sqrt 3 :=
sorry

end max_of_a_l224_224710


namespace exists_small_triangle_l224_224585

noncomputable def area (A B C : ℝ × ℝ) : ℝ :=
  0.5 * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))

theorem exists_small_triangle :
  ∀ (P : Fin 6 → ℝ × ℝ),
  (∀ i, abs (P i).1 ≤ 2 ∧ abs (P i).2 ≤ 2) →
  (∀ i j k, (i ≠ j → j ≠ k → i ≠ k → det ![(P i).1, (P i).2, 1; (P j).1, (P j).2, 1; (P k).1, (P k).2, 1] ≠ 0)) →
  ∃ i j k, i ≠ j ∧ j ≠ k ∧ i ≠ k ∧ area (P i) (P j) (P k) ≤ 2 :=
by
  sorry

end exists_small_triangle_l224_224585


namespace sum_of_primes_p_minus_1_power_of_2_below_100000_l224_224618

theorem sum_of_primes_p_minus_1_power_of_2_below_100000 : 
  ∑ p in (Finset.filter (λ p, p < 100000 ∧ (p % 2 = 1) ∧ ((p - 1).natAbs.isPowerOfTwo)) Finset.range(100000)), p = 65819 := 
  sorry

end sum_of_primes_p_minus_1_power_of_2_below_100000_l224_224618


namespace average_cost_across_all_products_sold_is_670_l224_224441

-- Definitions based on conditions
def iphones_sold : ℕ := 100
def ipad_sold : ℕ := 20
def appletv_sold : ℕ := 80

def cost_iphone : ℕ := 1000
def cost_ipad : ℕ := 900
def cost_appletv : ℕ := 200

-- Calculations based on conditions
def revenue_iphone : ℕ := iphones_sold * cost_iphone
def revenue_ipad : ℕ := ipad_sold * cost_ipad
def revenue_appletv : ℕ := appletv_sold * cost_appletv

def total_revenue : ℕ := revenue_iphone + revenue_ipad + revenue_appletv
def total_products_sold : ℕ := iphones_sold + ipad_sold + appletv_sold

def average_cost := total_revenue / total_products_sold

-- Theorem to be proved
theorem average_cost_across_all_products_sold_is_670 :
  average_cost = 670 :=
by
  sorry

end average_cost_across_all_products_sold_is_670_l224_224441


namespace weight_of_daughter_l224_224828

variable (M D G S : ℝ)

theorem weight_of_daughter :
  M + D + G + S = 200 →
  D + G = 60 →
  G = M / 5 →
  S = 2 * D →
  D = 800 / 15 :=
by
  intros h1 h2 h3 h4
  sorry

end weight_of_daughter_l224_224828


namespace balloon_arrangement_count_l224_224976

-- Definitions of letter frequencies for the word BALLOON
def letter_frequencies : List (Char × Nat) := [('B', 1), ('A', 1), ('L', 2), ('O', 2), ('N', 1)]

-- Total number of letters
def total_letters := 7

-- The formula for the number of unique arrangements of the letters
noncomputable def arrangements :=
  (Nat.factorial total_letters) / 
  (Nat.factorial 1 * Nat.factorial 1 * Nat.factorial 2 * Nat.factorial 2 * Nat.factorial 1)

-- The theorem to prove the number of ways to arrange the letters in "BALLOON"
theorem balloon_arrangement_count : arrangements = 1260 :=
  sorry

end balloon_arrangement_count_l224_224976


namespace num_roots_f_eq_zero_l224_224257

theorem num_roots_f_eq_zero
  (f : ℝ → ℝ)
  (h_symm1 : ∀ x : ℝ, f (2 - x) = f (2 + x))
  (h_symm2 : ∀ x : ℝ, f (7 - x) = f (7 + x))
  (h_zero_1 : f 1 = 0)
  (h_zero_3 : f 3 = 0) :
  (filter (λ x, f x = 0) (Icc (-2005 : ℝ) 2005)).card = 802 :=
sorry

end num_roots_f_eq_zero_l224_224257


namespace fencing_rate_l224_224478

noncomputable def rate_per_meter (diameter : ℝ) (total_cost : ℝ) : ℝ :=
  total_cost / (Real.pi * diameter)

theorem fencing_rate (d : ℝ) (tc : ℝ) (h_d : d = 22) (h_tc : tc = 207.34511513692632) :
  rate_per_meter d tc ≈ 3 := 
by 
  sorry

end fencing_rate_l224_224478


namespace newly_grown_uneaten_potatoes_l224_224263

variable (u : ℕ)

def initially_planted : ℕ := 8
def total_now : ℕ := 11

theorem newly_grown_uneaten_potatoes : u = total_now - initially_planted := by
  sorry

end newly_grown_uneaten_potatoes_l224_224263


namespace petya_eight_squares_l224_224803

-- Definitions
variable (circle : Type) [MetricSpace circle] [MetricCircle circle] -- circle as a type with standard metric properties

-- Assumption about the placement of squares
variable (inscribed_square : circle → circle → Prop) -- predicate to check if a square is inscribed in the circle

-- Main theorem statement
theorem petya_eight_squares (c : circle) (r : ℝ) 
  (h1 : ∀ (A B : circle), inscribed_square A B → ∃ (C : circle), inscribed_square B C ∧ A ≠ B ∧ B ≠ C)
  (h2 : r = 5) :
  (∃ (n : ℕ), n = 8) ∧ 
  (∀ (A₁ A₈ : circle), inscribed_square A₁ A₈ → inscribed_square A₈ A₁) :=
sorry

end petya_eight_squares_l224_224803


namespace min_value_of_quadratic_l224_224018

-- Define the function
def f (x : ℝ) : ℝ := x^2 - 8 * x + 15

-- State the hypothesis and the target to prove
theorem min_value_of_quadratic : ∀ x : ℝ, f(x) ≥ f(4) :=
by
  sorry -- proof omitted

end min_value_of_quadratic_l224_224018


namespace sample_size_correct_l224_224066

theorem sample_size_correct :
  (total_students : ℕ) → (sample_students : ℕ) →
  total_students = 2000 →
  sample_students = 200 →
  sample_students = 200 :=
by
  intros total_students sample_students 
  assume h1 : total_students = 2000
  assume h2 : sample_students = 200
  exact h2

end sample_size_correct_l224_224066


namespace angle_of_inclination_l224_224165

-- Given condition as definitions
def line_eq (x y : ℝ) : Prop := x - real.sqrt 3 * y - 2 = 0
def slope : ℝ := real.sqrt 3 / 3
def inclination : ℝ := real.arctan slope

-- The Lean statement that represents the proof problem
theorem angle_of_inclination (line_eq : ∀ x y, x - real.sqrt 3 * y - 2 = 0) : inclination = real.pi / 6 :=
  sorry

end angle_of_inclination_l224_224165


namespace ratio_equality_l224_224573

variable (a b : ℝ)

theorem ratio_equality (h : a / b = 4 / 3) : (3 * a + 2 * b) / (3 * a - 2 * b) = 3 :=
by
sorry

end ratio_equality_l224_224573


namespace balloon_arrangements_l224_224978

def factorial (n : Nat) : Nat :=
  if n = 0 then 1 else n * factorial (n - 1)

def arrangements (n : Nat) (ks : List Nat) :=
  factorial n / (ks.map factorial).foldr (*) 1

theorem balloon_arrangements : arrangements 7 [2, 2] = 1260 :=
by
  sorry

end balloon_arrangements_l224_224978


namespace divisors_of_8820_multiple_of_5_l224_224173

theorem divisors_of_8820_multiple_of_5 : 
  let a_choices := {0, 1, 2}
  let b_choices := {0, 1, 2}
  let d_choices := {0, 1, 2}
  let c_choices := {1}
  let number_of_divisors := a_choices.card * b_choices.card * c_choices.card * d_choices.card
  (2^2 * 3^2 * 5^1 * 7^2 = 8820) ->
  number_of_divisors = 27 :=
by
  sorry

end divisors_of_8820_multiple_of_5_l224_224173


namespace additional_time_required_l224_224804

-- Definitions based on conditions
def time_to_clean_three_sections : ℕ := 24
def total_sections : ℕ := 27

-- Rate of cleaning
def cleaning_rate_per_section (t : ℕ) (n : ℕ) : ℕ := t / n

-- Total time required to clean all sections
def total_cleaning_time (n : ℕ) (r : ℕ) : ℕ := n * r

-- Additional time required to clean the remaining sections
def additional_cleaning_time (t_total : ℕ) (t_spent : ℕ) : ℕ := t_total - t_spent

-- Theorem statement
theorem additional_time_required 
  (t3 : ℕ) (n : ℕ) (t_spent : ℕ) 
  (h₁ : t3 = time_to_clean_three_sections)
  (h₂ : n = total_sections)
  (h₃ : t_spent = time_to_clean_three_sections)
  : additional_cleaning_time (total_cleaning_time n (cleaning_rate_per_section t3 3)) t_spent = 192 :=
by
  sorry

end additional_time_required_l224_224804


namespace triangle_area_heron_l224_224143

-- Define the side lengths of the triangle
def a : ℝ := 31.5
def b : ℝ := 27.8
def c : ℝ := 10.3

-- Define the semi-perimeter
def s : ℝ := (a + b + c) / 2

-- Using Heron's formula to define the area of the triangle
def area : ℝ := Real.sqrt (s * (s - a) * (s - b) * (s - c))

-- The goal is to prove the area is approximately 141.65
theorem triangle_area_heron : abs (area - 141.65) < 1e-2 := by
  sorry

end triangle_area_heron_l224_224143


namespace balloon_permutations_l224_224934

theorem balloon_permutations : 
  let balloons := "BALLOON".toList.length in
  let total_permutations := Nat.factorial balloons in
  let repetitive_L := Nat.factorial 2 in
  let repetitive_O := Nat.factorial 2 in
  (total_permutations / (repetitive_L * repetitive_O)) = 1260 := sorry

end balloon_permutations_l224_224934


namespace calculate_E_l224_224794

theorem calculate_E (P J T B A E : ℝ) 
  (h1 : J = 0.75 * P)
  (h2 : J = 0.80 * T)
  (h3 : B = 1.40 * J)
  (h4 : A = 0.85 * B)
  (h5 : T = P - (E / 100) * P)
  (h6 : E = 2 * ((P - A) / P) * 100) : 
  E = 21.5 := 
sorry

end calculate_E_l224_224794


namespace larry_whales_erased_first_l224_224465
noncomputable theory

def evan_whales : ℕ := 10
def larry_whales : ℕ := 15
def alex_whales : ℕ := 20
def total_whales : ℕ := evan_whales + larry_whales + alex_whales

theorem larry_whales_erased_first :
  ∃ p q : ℕ, (p + q = 143) ∧ (∃ h1 : q ≠ 0, 38 / 105 = p / q) :=
by
  sorry

end larry_whales_erased_first_l224_224465


namespace balloon_permutations_l224_224962

theorem balloon_permutations : (∀ (arrangements : ℕ),
  arrangements = nat.factorial 7 / (nat.factorial 2 * nat.factorial 2) → arrangements = 1260) :=
begin
  intros arrangements h,
  rw [nat.factorial_succ, nat.factorial],
  sorry,
end

end balloon_permutations_l224_224962


namespace max_odd_row_sums_l224_224047

theorem max_odd_row_sums (numbers : Fin 27 → Fin 28) : 
  ∃ O, (O ≤ 24 ∧ 
        (∀ i j k : Fin 3, 
          let sum_rows_x := (numbers ⟨i*9 + j*3 + 0, by simp⟩) + (numbers ⟨i*9 + j*3 + 1, by simp⟩) + (numbers ⟨i*9 + j*3 + 2, by simp⟩) in
          let sum_rows_y := (numbers ⟨i*9 + 0*3 + k, by simp⟩) + (numbers ⟨i*9 + 1*3 + k, by simp⟩) + (numbers ⟨i*9 + 2*3 + k, by simp⟩) in
          let sum_rows_z := (numbers ⟨0*9 + j*3 + k, by simp⟩) + (numbers ⟨1*9 + j*3 + k, by simp⟩) + (numbers ⟨2*9 + j*3 + k, by simp⟩) in
          (sum_rows_x % 2 = 1 → O ≥ 1) ∧
          (sum_rows_y % 2 = 1 → O ≥ 1) ∧ 
          (sum_rows_z % 2 = 1 → O ≥ 1)) ∧
        (27 - O) % 2 = 0) :=
sorry

end max_odd_row_sums_l224_224047


namespace number_of_good_subsets_l224_224092

def is_good_subset (s : Finset ℤ) : Prop := 
  ∃ (a c ∈ s) (b ∉ s), a < b ∧ b < c

def count_good_subsets : ℤ :=
  2^2019 - 2041211

theorem number_of_good_subsets (T : Finset ℤ) (hT : T = Finset.range 2020) :
  (Finset.filter is_good_subset (Finset.powerset T)).card = count_good_subsets := 
sorry

end number_of_good_subsets_l224_224092


namespace woman_year_of_birth_l224_224063

def year_of_birth (x : ℕ) : ℕ := x^2 - x

theorem woman_year_of_birth : ∃ (x : ℕ), 1850 ≤ year_of_birth x ∧ year_of_birth x < 1900 ∧ year_of_birth x = 1892 :=
by
  sorry

end woman_year_of_birth_l224_224063


namespace distinct_numbers_lcm_sum_l224_224502

theorem distinct_numbers_lcm_sum (n : ℕ) (h : n > 2) :
  ∃ (a : Fin n → ℕ), (∀ i j : Fin n, i ≠ j → a i ≠ a j) ∧ (Nat.lcm_list (List.ofFn a) = List.sum (List.ofFn a)) :=
sorry

end distinct_numbers_lcm_sum_l224_224502


namespace largest_prime_factor_expression_l224_224743

open Nat

/-- The largest prime factor of the given expression is 547. -/
theorem largest_prime_factor_expression :
  ∀ (a b c d : ℕ), a = 16 ∧ b = 3 ∧ c = 2 ∧ d = 17 →
  prime 547 ∧
  (a^4 + b*a^2 + c - d^4) = -31 * 547 → 547 = 547 :=
by
  intro a b c d
  intro habcd h
  sorry

end largest_prime_factor_expression_l224_224743


namespace polygons_intersection_l224_224588

/-- In a square with an area of 5, nine polygons, each with an area of 1, are placed. 
    Prove that some two of them must have an intersection area of at least 1 / 9. -/
theorem polygons_intersection 
  (S : ℝ) (hS : S = 5)
  (n : ℕ) (hn : n = 9)
  (polygons : Fin n → ℝ) (hpolys : ∀ i, polygons i = 1)
  (intersection : Fin n → Fin n → ℝ) : 
  ∃ i j : Fin n, i ≠ j ∧ intersection i j ≥ 1 / 9 := 
sorry

end polygons_intersection_l224_224588


namespace element_B_weight_mg_l224_224452

theorem element_B_weight_mg (total_weight_g : ℝ) (ratio_A : ℕ) (ratio_B : ℕ) (ratio_C : ℕ)
  (total_mg_per_g : ℕ) (total_ratio : ℕ) : 
  ratio_A = 2 → ratio_B = 10 → ratio_C = 3 →
  total_weight_g = 330 → total_mg_per_g = 1000 →
  total_ratio = ratio_A + ratio_B + ratio_C →
  let weight_B_g := (total_weight_g * ratio_B) / total_ratio in
  let weight_B_mg := weight_B_g * total_mg_per_g in
  weight_B_mg = 220000 :=
by
  intros h₁ h₂ h₃ h₄ h₅ h₆
  let weight_B_g := (330 * 10) / (2 + 10 + 3)
  let weight_B_mg := weight_B_g * 1000
  have h₇ : weight_B_g = 220 := by sorry
  have h₈ : weight_B_mg = weight_B_g * 1000 := rfl
  rw [h₇, mul_comm 220 1000]
  exact rfl

end element_B_weight_mg_l224_224452


namespace balloon_permutations_l224_224942

theorem balloon_permutations : 
  let balloons := "BALLOON".toList.length in
  let total_permutations := Nat.factorial balloons in
  let repetitive_L := Nat.factorial 2 in
  let repetitive_O := Nat.factorial 2 in
  (total_permutations / (repetitive_L * repetitive_O)) = 1260 := sorry

end balloon_permutations_l224_224942


namespace arithmetic_sequence_term_l224_224599

def a : ℕ → ℕ
| 1       := 1
| (n + 1) := a n + 4

theorem arithmetic_sequence_term : a 20 = 77 := by
  sorry

end arithmetic_sequence_term_l224_224599


namespace peter_horses_food_requirement_l224_224652

theorem peter_horses_food_requirement :
  let daily_oats_per_horse := 4 * 2 in
  let daily_grain_per_horse := 3 in
  let daily_food_per_horse := daily_oats_per_horse + daily_grain_per_horse in
  let number_of_horses := 4 in
  let daily_food_all_horses := daily_food_per_horse * number_of_horses in
  let days := 3 in
  daily_food_all_horses * days = 132 :=
by
  sorry

end peter_horses_food_requirement_l224_224652


namespace kolya_is_collection_agency_l224_224211

structure Condition :=
  (katya_lent_books_to_vasya : ∀ books : Type, katya_lent books vasya)
  (vasya_failed_to_return_books : ∀ (books : Type) (month_later : Time), failed_return books vasya month_later)
  (katya_asked_kolya_to_retrieve : ∀ books : Type, katya_asks books katya kolya vaaya)
  (kolya_agrees_for_reward : ∀ book : Type, kolya_agrees book katya_retrieve books)

theorem kolya_is_collection_agency {books : Type} (h : Condition books) : 
  KolyaRole = CollectionAgency :=
sorry

end kolya_is_collection_agency_l224_224211


namespace sum_binom_149_l224_224371

theorem sum_binom_149 :
  let S := ∑ k in Finset.range 75, (-1 : ℂ) ^ k * nat.choose 149 (2 * k)
  S = -2 ^ 74 :=
by sorry

end sum_binom_149_l224_224371


namespace arithmetic_sequence_example_l224_224594

noncomputable def is_arithmetic_sequence {α : Type*} [has_add α] [has_mul α] (a : ℕ → α) : Prop :=
∃ d, ∀ n, a (n + 1) = a n + d

theorem arithmetic_sequence_example (a : ℕ → ℤ) (h1 : a 3 = -5) (h2 : a 7 = -1) (h3 : is_arithmetic_sequence a) : a 5 = -3 :=
by
  sorry

end arithmetic_sequence_example_l224_224594


namespace repeating_decimal_as_fraction_l224_224750

theorem repeating_decimal_as_fraction :
  let x := 56 / 99
  in x = 0.56 + 0.0056 + 0.000056 + (0.00000056) : ℚ :=
by
  sorry

end repeating_decimal_as_fraction_l224_224750


namespace convert_to_cloaks_l224_224584

theorem convert_to_cloaks : 
  ∀ (hůlky plášťů klobouky : ℕ), 
  (4 * hůlky = 6 * plášťů) → 
  (5 * hůlky = 5 * klobouky) → 
  (5 * hůlky + 1 * klobouky = 9 * plášťů) :=
by
  intro hůlky plášťů klobouky
  assume hůlky_to_plášťů : 4 * hůlky = 6 * plášťů
  assume hůlky_to_klobouky : 5 * hůlky = 5 * klobouky
  sorry

end convert_to_cloaks_l224_224584


namespace quadratic_function_vertex_l224_224714

def vertex_quadratic_function (a x h k : ℝ) : ℝ := a * (x - h)^2 + k

theorem quadratic_function_vertex :
  ∃ a k : ℝ, (vertex_quadratic_function a 1 1 (-2) = k ∧ a = -3 ∧ k = -2) :=
by
  use [-3, -2]
  split
  · simp only [vertex_quadratic_function]
    ring
  · split
    · refl
    · refl

end quadratic_function_vertex_l224_224714


namespace parameter_values_l224_224112

def system_equation_1 (x y : ℝ) : Prop :=
  (|y + 9| + |x + 2| - 2) * (x^2 + y^2 - 3) = 0

def system_equation_2 (x y a : ℝ) : Prop :=
  (x + 2)^2 + (y + 4)^2 = a

theorem parameter_values (a : ℝ) :
  (∃ x y : ℝ, system_equation_1 x y ∧ system_equation_2 x y a ∧ 
    -- counting the number of solutions to the system of equations that total exactly three,
    -- meaning the system has exactly three solutions
    -- Placeholder for counting solutions
    sorry) ↔ (a = 9 ∨ a = 23 + 4 * Real.sqrt 15) := 
sorry

end parameter_values_l224_224112


namespace marriage_year_proof_l224_224046

-- Definitions based on conditions
def marriage_year : ℕ := sorry
def child1_birth_year : ℕ := 1982
def child2_birth_year : ℕ := 1984
def reference_year : ℕ := 1986

-- Age calculations based on reference year
def age_in_1986 (birth_year : ℕ) : ℕ := reference_year - birth_year

-- Combined ages in the reference year
def combined_ages_in_1986 : ℕ := age_in_1986 child1_birth_year + age_in_1986 child2_birth_year

-- The main theorem to prove
theorem marriage_year_proof :
  combined_ages_in_1986 = reference_year - marriage_year →
  marriage_year = 1980 := by
  sorry

end marriage_year_proof_l224_224046


namespace balloon_arrangements_l224_224983

def factorial (n : Nat) : Nat :=
  if n = 0 then 1 else n * factorial (n - 1)

def arrangements (n : Nat) (ks : List Nat) :=
  factorial n / (ks.map factorial).foldr (*) 1

theorem balloon_arrangements : arrangements 7 [2, 2] = 1260 :=
by
  sorry

end balloon_arrangements_l224_224983


namespace limit_calculation_l224_224447

noncomputable def limit_frac : ℕ → ℝ := λ n, (2 * n - 5) / (n + 1)

theorem limit_calculation : filter.tendsto limit_frac filter.at_top (nhds 2) := 
sorry

end limit_calculation_l224_224447


namespace recurring_to_fraction_l224_224755

theorem recurring_to_fraction : ∀ (x : ℚ), x = 0.5656 ∧ 100 * x = 56.5656 → x = 56 / 99 :=
by
  intros x hx
  sorry

end recurring_to_fraction_l224_224755


namespace part1_part2_part3_l224_224537

-- Definitions given in the conditions.
def characteristic_coeff_pair (a b c : ℝ) : (ℝ × ℝ × ℝ) :=
  (a, b, c)

def characteristic_poly (p : ℝ × ℝ × ℝ) : ℝ → ℝ :=
  λ x => p.1 * x ^ 2 + p.2 * x + p.3

-- Part (1)
theorem part1 : characteristic_coeff_pair 3 4 1 = (3, 4, 1) :=
by
  sorry

-- Part (2)
theorem part2 : characteristic_poly (2, 1, 2) + characteristic_poly (2, -1, 2) =
  characteristic_poly (4, 0, 4) :=
by
  sorry

-- Part (3)
theorem part3 (m n : ℝ) (h : characteristic_poly (1, 2, m) - characteristic_poly (2, n, 3) = λ x => - x^2 + x - 1) :
  m * n = 2 :=
by
  sorry

end part1_part2_part3_l224_224537


namespace percent_change_area_decrease_l224_224013

theorem percent_change_area_decrease (L W : ℝ) (hL : L > 0) (hW : W > 0) :
    let A_initial := L * W
    let L_new := 1.60 * L
    let W_new := 0.40 * W
    let A_new := L_new * W_new
    let percent_change := (A_new - A_initial) / A_initial * 100
    percent_change = -36 :=
by
  sorry

end percent_change_area_decrease_l224_224013


namespace random_interval_probability_l224_224195

theorem random_interval_probability (x : ℝ) : 
  (x ∈ set.Icc (-2 : ℝ) 3) → 
  (∀ (y : ℝ), y ∈ set.Icc (-2 : ℝ) 3 → (∃ p : ℝ, 0 ≤ p ∧ p ≤ 1 ∧ (set.Icc (-2 : ℝ) (1 : ℝ)).indicator 1 y = p)) →
  (measure_theory.measure_space.measure (set.Icc (-2 : ℝ) 1) / measure_theory.measure_space.measure (set.Icc (-2 : ℝ) 3)) = (3 / 5 : ℝ) :=
begin
  sorry
end

end random_interval_probability_l224_224195


namespace balloon_arrangement_count_l224_224977

-- Definitions of letter frequencies for the word BALLOON
def letter_frequencies : List (Char × Nat) := [('B', 1), ('A', 1), ('L', 2), ('O', 2), ('N', 1)]

-- Total number of letters
def total_letters := 7

-- The formula for the number of unique arrangements of the letters
noncomputable def arrangements :=
  (Nat.factorial total_letters) / 
  (Nat.factorial 1 * Nat.factorial 1 * Nat.factorial 2 * Nat.factorial 2 * Nat.factorial 1)

-- The theorem to prove the number of ways to arrange the letters in "BALLOON"
theorem balloon_arrangement_count : arrangements = 1260 :=
  sorry

end balloon_arrangement_count_l224_224977


namespace recurring_decimal_to_fraction_correct_l224_224761

noncomputable def recurring_decimal_to_fraction (b : ℚ) : Prop :=
  b = 0.\overline{56} ↔ b = 56/99

theorem recurring_decimal_to_fraction_correct : recurring_decimal_to_fraction 0.\overline{56} :=
  sorry

end recurring_decimal_to_fraction_correct_l224_224761


namespace exist_triangle_l224_224455

-- Definitions of points and properties required in the conditions
structure Point :=
(x : ℝ) (y : ℝ)

def orthocenter (M : Point) := M 
def centroid (S : Point) := S 
def vertex (C : Point) := C 

-- The problem statement that needs to be proven
theorem exist_triangle (M S C : Point) 
    (h_orthocenter : orthocenter M = M)
    (h_centroid : centroid S = S)
    (h_vertex : vertex C = C) : 
    ∃ (A B : Point), 
        -- A, B, and C form a triangle ABC
        -- S is the centroid of this triangle
        -- M is the orthocenter of this triangle
        -- C is one of the vertices
        true := 
sorry

end exist_triangle_l224_224455


namespace quadratic_function_properties_l224_224504

def quadratic_function (x : ℝ) : ℝ :=
  -6 * x^2 + 36 * x - 48

theorem quadratic_function_properties :
  quadratic_function 2 = 0 ∧ quadratic_function 4 = 0 ∧ quadratic_function 3 = 6 :=
by
  -- The proof is omitted
  -- Placeholder for the proof
  sorry

end quadratic_function_properties_l224_224504


namespace problem_l224_224685

theorem problem
  (x y : ℝ)
  (h1 : x - y = 12)
  (h2 : x^2 + y^2 = 320) :
  x * y = 64 ∧ x^3 + y^3 = 4160 :=
by
  sorry

end problem_l224_224685


namespace count_valid_x_l224_224499

def is_valid_x (x : ℕ) : Prop := 30 < (x + 5) * (x + 5) ∧ (x + 5) * (x + 5) < 60

theorem count_valid_x :
  {x : ℕ // is_valid_x x}.to_finset.card = 2 := by
  -- Proof omitted
  sorry

end count_valid_x_l224_224499


namespace triangle_value_l224_224522

theorem triangle_value (A B C : ℝ) (a b c : ℝ) 
  (h₁ : b = 5) 
  (h₂ : real.cos B = 4 / 5) 
  (h₃ : 1 / 2 * a * c * real.sin B = 12) 
  : (a + c) / (real.sin A + real.sin C) = 25 / 3 :=
sorry

end triangle_value_l224_224522


namespace time_to_pass_platform_l224_224379

-- Define the known values from the conditions
def lengthOfTrain : ℝ := 120 -- in meters
def speedOfTrain : ℝ := 60 -- in kmph
def lengthOfPlatform : ℝ := 240 -- in meters

-- Conversion factor from kmph to m/s
def kmph_to_mps (speed: ℝ) : ℝ := speed * (10 / 36)

-- Calculate the total distance
def totalDistance : ℝ := lengthOfTrain + lengthOfPlatform

-- Convert the speed from kmph to m/s
def speedInMps : ℝ := kmph_to_mps speedOfTrain

-- Define the target time to be proven
def targetTime : ℝ := 21.6 -- in seconds

-- Theorem: The time it takes for the train to pass the platform is approximately 21.6 seconds.
theorem time_to_pass_platform : 
  abs ((totalDistance / speedInMps) - targetTime) < 0.1 := by
  sorry

end time_to_pass_platform_l224_224379


namespace balloon_permutation_count_l224_224914

-- Define the necessary factorials and the formula for permutations of multiset
def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem balloon_permutation_count : 
  let total_letters := 7,
      repeat_L := 2,
      repeat_O := 2,
      unique_arrangements := factorial total_letters / (factorial repeat_L * factorial repeat_O) 
  in unique_arrangements = 1260 :=
by
  sorry

end balloon_permutation_count_l224_224914


namespace balloon_permutations_count_l224_224904

-- Definitions of the conditions
def total_letters_count : ℕ := 7
def l_count : ℕ := 2
def o_count : ℕ := 2

-- Now the mathematical problem as a Lean statement
theorem balloon_permutations_count : 
  (Nat.factorial total_letters_count) / ((Nat.factorial l_count) * (Nat.factorial o_count)) = 1260 := 
by
  sorry

end balloon_permutations_count_l224_224904


namespace michael_peach_pies_l224_224645

/--
Michael ran a bakeshop and had to fill an order for some peach pies, 4 apple pies and 3 blueberry pies.
Each pie recipe called for 3 pounds of fruit each. At the market, produce was on sale for $1.00 per pound for both blueberries and apples.
The peaches each cost $2.00 per pound. Michael spent $51 at the market buying the fruit for his pie order.
Prove that Michael had to make 5 peach pies.
-/
theorem michael_peach_pies :
  let apple_pies := 4
  let blueberry_pies := 3
  let peach_pie_cost_per_pound := 2
  let apple_blueberry_cost_per_pound := 1
  let pounds_per_pie := 3
  let total_spent := 51
  (total_spent - ((apple_pies + blueberry_pies) * pounds_per_pie * apple_blueberry_cost_per_pound)) 
  / (pounds_per_pie * peach_pie_cost_per_pound) = 5 :=
by
  let apple_pies := 4
  let blueberry_pies := 3
  let peach_pie_cost_per_pound := 2
  let apple_blueberry_cost_per_pound := 1
  let pounds_per_pie := 3
  let total_spent := 51
  have H1 : (total_spent - ((apple_pies + blueberry_pies) * pounds_per_pie * apple_blueberry_cost_per_pound)) 
             / (pounds_per_pie * peach_pie_cost_per_pound) = 5 := sorry
  exact H1

end michael_peach_pies_l224_224645


namespace real_solutions_count_l224_224563

theorem real_solutions_count :
  (∃ x : ℝ, |x - 2| - 4 = 1 / |x - 3|) ∧
  (∃ y : ℝ, |y - 2| - 4 = 1 / |y - 3| ∧ x ≠ y) :=
sorry

end real_solutions_count_l224_224563


namespace sufficient_not_necessary_l224_224137

theorem sufficient_not_necessary (a b : ℝ) : (a^2 + b^2 ≤ 2) → (-1 ≤ a * b ∧ a * b ≤ 1) ∧ ¬((-1 ≤ a * b ∧ a * b ≤ 1) → a^2 + b^2 ≤ 2) := 
by
  sorry

end sufficient_not_necessary_l224_224137


namespace binary_to_decimal_10101_l224_224088

theorem binary_to_decimal_10101:
  ∑ i in Finset.range 5, (bit_of_bin_10101 i) * (2 ^ i) = 21 := by
sorry

-- Helper function to get the bit at position i for the binary number 10101
def bit_of_bin_10101 (i : ℕ) : ℕ :=
  if i = 0 then 1
  else if i = 1 then 0
  else if i = 2 then 1
  else if i = 3 then 0
  else if i = 4 then 1
  else 0

end binary_to_decimal_10101_l224_224088


namespace smallest_n_divisible_by_15_exists_subset_sum_divisible_by_15_l224_224127

theorem smallest_n_divisible_by_15 (n : ℕ) (h : ∀ (A : finset ℕ), A.card = n → 
  ∃ S : finset ℕ, S ⊆ A ∧ S.card = 15 ∧ (S.sum id) % 15 = 0) : n ≥ 29 :=
sorry

theorem exists_subset_sum_divisible_by_15 (A : finset ℕ) (hA : A.card = 29) :
  ∃ S : finset ℕ, S ⊆ A ∧ S.card = 15 ∧ (S.sum id) % 15 = 0 :=
sorry

end smallest_n_divisible_by_15_exists_subset_sum_divisible_by_15_l224_224127


namespace rabbit_is_hit_l224_224353

noncomputable def P_A : ℝ := 0.6
noncomputable def P_B : ℝ := 0.5
noncomputable def P_C : ℝ := 0.4

noncomputable def P_none_hit : ℝ := (1 - P_A) * (1 - P_B) * (1 - P_C)
noncomputable def P_rabbit_hit : ℝ := 1 - P_none_hit

theorem rabbit_is_hit :
  P_rabbit_hit = 0.88 :=
by
  -- Proof is omitted
  sorry

end rabbit_is_hit_l224_224353


namespace find_y_l224_224862

noncomputable def a := (3/5) * 2500
noncomputable def b := (2/7) * ((5/8) * 4000 + (1/4) * 3600 - (11/20) * 7200)
noncomputable def c (y : ℚ) := (3/10) * y
def result (a b c : ℚ) := a * b / c

theorem find_y : ∃ y : ℚ, result a b (c y) = 25000 ∧ y = -4/21 := 
by
  sorry

end find_y_l224_224862


namespace rhombus_fourth_vertex_l224_224503

noncomputable def fourth_vertex_of_rhombus (z1 z2 z3 : Complex) : Complex :=
  if (z1 = 2 + 3 * Complex.i ∧ z2 = -3 + 2 * Complex.i ∧ z3 = -2 - 3 * Complex.i) then
    3 - 2 * Complex.i
  else
    sorry

theorem rhombus_fourth_vertex :
  fourth_vertex_of_rhombus (2 + 3 * Complex.i) (-3 + 2 * Complex.i) (-2 - 3 * Complex.i) = 3 - 2 * Complex.i :=
begin
  -- Proof would be provided here if needed
  sorry
end

end rhombus_fourth_vertex_l224_224503


namespace probability_non_yellow_l224_224816

def num_red := 4
def num_green := 7
def num_yellow := 9
def num_blue := 10

def total_jelly_beans := num_red + num_green + num_yellow + num_blue
def num_non_yellow := num_red + num_green + num_blue

theorem probability_non_yellow : (num_non_yellow : ℚ) / total_jelly_beans = 7 / 10 :=
by
  have h1: total_jelly_beans = 30 := by norm_num
  have h2: num_non_yellow = 21 := by norm_num
  rw [h1, h2]
  norm_num
  sorry

end probability_non_yellow_l224_224816


namespace balloon_arrangements_l224_224981

def factorial (n : Nat) : Nat :=
  if n = 0 then 1 else n * factorial (n - 1)

def arrangements (n : Nat) (ks : List Nat) :=
  factorial n / (ks.map factorial).foldr (*) 1

theorem balloon_arrangements : arrangements 7 [2, 2] = 1260 :=
by
  sorry

end balloon_arrangements_l224_224981


namespace problem1_problem2_problem3_problem4_l224_224864

-- Problem 1
theorem problem1 : (- (3 : ℝ) / 7) + (1 / 5) + (2 / 7) + (- (6 / 5)) = - (8 / 7) :=
by
  sorry

-- Problem 2
theorem problem2 : -(-1) + 3^2 / (1 - 4) * 2 = -5 :=
by
  sorry

-- Problem 3
theorem problem3 :  (-(1 / 6))^2 / ((1 / 2 - 1 / 3)^2) / (abs (-6))^2 = 1 / 36 :=
by
  sorry

-- Problem 4
theorem problem4 : (-1) ^ 1000 - 2.45 * 8 + 2.55 * (-8) = -39 :=
by
  sorry

end problem1_problem2_problem3_problem4_l224_224864


namespace cowboy_jimmy_wins_bet_l224_224873

-- Definitions based on problem conditions
def revolution_rate : ℝ := 50  -- revolutions per second
def blades_count : ℕ := 4       -- total number of blades
def angle_separation : ℝ := 90  -- degrees separation between blades

-- Main theorem statement
theorem cowboy_jimmy_wins_bet :
  ∃ (t : ℝ), ∀ (i : ℕ), (i < blades_count) → (blade_position t i) = true :=
begin
  -- Assume blade_position is a function that checks if Jimmy's bullet intersects the i-th blade at time t
  -- The implementation of blade_position is omitted here, would depend on details of system setup
  sorry,
end

end cowboy_jimmy_wins_bet_l224_224873


namespace solve_inequality_l224_224163

def f (x : ℝ) : ℝ :=
if x ≥ 0 then x^2 + 2 * x else -x^2 + 2 * x

theorem solve_inequality (x : ℝ) : f x > 3 ↔ x > 1 := by
  sorry

end solve_inequality_l224_224163


namespace zero_count_iff_a_range_l224_224245

noncomputable def f (a x : ℝ) : ℝ :=
if x < a then sin (3 * π * x - 3 * π * a)
else -x^2 + 2 * (a + 1) * x - a^2 - 5

theorem zero_count_iff_a_range (a : ℝ) :
  (∃ count : ℕ, count = 9 ∧ (∀ x, f a x = 0 → 0 < x → x ∈ set.Ioi 0)) ↔ 
  a ∈ set.Icc (7 / 3) (5 / 2) ∪ set.Icc (8 / 3) 3 := 
sorry

end zero_count_iff_a_range_l224_224245


namespace part_a_part_b_part_c_l224_224616

-- Part (a)
theorem part_a (n : ℕ) (hn : n ≥ 1) : 
  let a_n := n^2
  let a_n_plu1 := (n+1)^2
  gcd a_n a_n_plu1 = 1 := sorry

-- Part (b)
theorem part_b (n : ℕ) (hn : n ≥ 1) : 
  let a_n := n^2 + 1
  let a_n_plu1 := (n+1)^2 + 1
  gcd a_n a_n_plu1 ∈ {1, 5} := sorry

-- Part (c)
theorem part_c (n : ℕ) (hn : n ≥ 1) (c : ℕ) :
  let a_n := n^2 + c
  let a_n_plu1 := (n+1)^2 + c
  gcd a_n a_n_plu1 ≤ 4*c + 1 := sorry

end part_a_part_b_part_c_l224_224616


namespace total_volume_of_water_polluted_l224_224091

theorem total_volume_of_water_polluted :
  let number_of_students_who_discard_batteries := 2200 / 2,
      pollution_per_battery := 600000,
      total_pollution := number_of_students_who_discard_batteries * pollution_per_battery in
  total_pollution = 6.6 * 10^8 :=
by
  let number_of_students_who_discard_batteries := 2200 / 2,
      pollution_per_battery := 600000,
      total_pollution := number_of_students_who_discard_batteries * pollution_per_battery
  show total_pollution = 6.6 * 10^8 by sorry

end total_volume_of_water_polluted_l224_224091


namespace original_six_digit_number_is_105262_l224_224060

def is_valid_number (N : ℕ) : Prop :=
  ∃ A : ℕ, A < 100000 ∧ (N = 10 * A + 2) ∧ (200000 + A = 2 * N + 2)

theorem original_six_digit_number_is_105262 :
  ∃ N : ℕ, is_valid_number N ∧ N = 105262 :=
by
  sorry

end original_six_digit_number_is_105262_l224_224060


namespace triangle_angle_measure_l224_224601

theorem triangle_angle_measure {D E F : ℝ} (hD : D = 90) (hE : E = 2 * F + 15) : 
  D + E + F = 180 → F = 25 :=
by
  intro h_sum
  sorry

end triangle_angle_measure_l224_224601


namespace balloon_permutations_count_l224_224908

-- Definitions of the conditions
def total_letters_count : ℕ := 7
def l_count : ℕ := 2
def o_count : ℕ := 2

-- Now the mathematical problem as a Lean statement
theorem balloon_permutations_count : 
  (Nat.factorial total_letters_count) / ((Nat.factorial l_count) * (Nat.factorial o_count)) = 1260 := 
by
  sorry

end balloon_permutations_count_l224_224908


namespace balloon_permutations_l224_224957

theorem balloon_permutations : (∀ (arrangements : ℕ),
  arrangements = nat.factorial 7 / (nat.factorial 2 * nat.factorial 2) → arrangements = 1260) :=
begin
  intros arrangements h,
  rw [nat.factorial_succ, nat.factorial],
  sorry,
end

end balloon_permutations_l224_224957


namespace fewest_number_of_gymnasts_l224_224424

theorem fewest_number_of_gymnasts (n : ℕ) (h : n % 2 = 0)
  (handshakes : ∀ (n : ℕ), (n * (n - 1) / 2) + n = 465) : 
  n = 30 :=
by
  sorry

end fewest_number_of_gymnasts_l224_224424


namespace log_comparison_l224_224138

theorem log_comparison (a b c : ℝ) (h₁ : a = Real.log 6 / Real.log 4) (h₂ : b = Real.log 3 / Real.log 2) (h₃ : c = 3/2) : b > c ∧ c > a := 
by 
  sorry

end log_comparison_l224_224138


namespace female_officers_count_l224_224029

theorem female_officers_count
  (total_on_duty : ℕ)
  (on_duty_females : ℕ)
  (total_female_officers : ℕ)
  (h1 : total_on_duty = 240)
  (h2 : on_duty_females = total_on_duty / 2)
  (h3 : on_duty_females = (40 * total_female_officers) / 100) : 
  total_female_officers = 300 := 
by
  sorry

end female_officers_count_l224_224029


namespace solve_for_x_l224_224990

theorem solve_for_x (x : ℝ) 
  (h : (2 / (x + 3)) + (3 * x / (x + 3)) - (5 / (x + 3)) = 2) : 
  x = 9 := 
by
  sorry

end solve_for_x_l224_224990


namespace second_hose_correct_l224_224554

/-- Define the problem parameters -/
def first_hose_rate : ℕ := 50
def initial_hours : ℕ := 3
def additional_hours : ℕ := 2
def total_capacity : ℕ := 390

/-- Define the total hours the first hose was used -/
def total_hours (initial_hours additional_hours : ℕ) : ℕ := initial_hours + additional_hours

/-- Define the amount of water sprayed by the first hose -/
def first_hose_total (first_hose_rate initial_hours additional_hours : ℕ) : ℕ :=
  first_hose_rate * (initial_hours + additional_hours)

/-- Define the remaining water needed to fill the pool -/
def remaining_water (total_capacity first_hose_total : ℕ) : ℕ :=
  total_capacity - first_hose_total

/-- Define the additional water sprayed by the first hose during the last 2 hours -/
def additional_first_hose (first_hose_rate additional_hours : ℕ) : ℕ :=
  first_hose_rate * additional_hours

/-- Define the water sprayed by the second hose -/
def second_hose_total (remaining_water additional_first_hose : ℕ) : ℕ :=
  remaining_water - additional_first_hose

/-- Define the rate of the second hose (output) -/
def second_hose_rate (second_hose_total additional_hours : ℕ) : ℕ :=
  second_hose_total / additional_hours

/-- Define the theorem we want to prove -/
theorem second_hose_correct :
  second_hose_rate
    (second_hose_total
        (remaining_water total_capacity (first_hose_total first_hose_rate initial_hours additional_hours))
        (additional_first_hose first_hose_rate additional_hours))
    additional_hours = 20 := by
  sorry

end second_hose_correct_l224_224554


namespace inequality_proof_l224_224526

theorem inequality_proof (n : ℕ)
  (a b : Fin n → ℝ)
  (h_pos : ∀ i, 0 < a i ∧ 0 < b i)
  (h_condition : a 0 ^ 2 > ∑ i in Finset.range n \ {0}, a i ^ 2) :
  (a 0 ^ 2 - ∑ i in Finset.range n \ {0}, a i ^ 2) * (b 0 ^ 2 - ∑ i in Finset.range n \ {0}, b i ^ 2) 
  ≤ (a 0 * b 0 - ∑ i in Finset.range n \ {0}, a i * b i) ^ 2 :=
by
  sorry

end inequality_proof_l224_224526


namespace second_job_pay_rate_l224_224610

-- Definitions of the conditions
def h1 : ℕ := 3 -- hours for the first job
def r1 : ℕ := 7 -- rate for the first job
def h2 : ℕ := 2 -- hours for the second job
def h3 : ℕ := 4 -- hours for the third job
def r3 : ℕ := 12 -- rate for the third job
def d : ℕ := 5   -- number of days
def T : ℕ := 445 -- total earnings

-- The proof statement
theorem second_job_pay_rate (x : ℕ) : 
  d * (h1 * r1 + 2 * x + h3 * r3) = T ↔ x = 10 := 
by 
  -- Implement the necessary proof steps here
  sorry

end second_job_pay_rate_l224_224610


namespace find_positive_integer_solutions_l224_224123

theorem find_positive_integer_solutions (x y z : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  2^x + 3^y = z^2 ↔ (x = 0 ∧ y = 1 ∧ z = 2) ∨ (x = 3 ∧ y = 0 ∧ z = 3) ∨ (x = 4 ∧ y = 2 ∧ z = 5) := 
sorry

end find_positive_integer_solutions_l224_224123


namespace wheel_distance_approx_l224_224421

noncomputable def pi := Real.pi
def diameter := 14
def revolutions := 16.014558689717926

theorem wheel_distance_approx :
  |(π * diameter * revolutions) - 705.878| < 0.001 := by
sorry

end wheel_distance_approx_l224_224421


namespace total_cans_collected_l224_224687

theorem total_cans_collected (students_perez : ℕ) (half_perez_collected_20 : ℕ) (two_perez_collected_0 : ℕ) (remaining_perez_collected_8 : ℕ)
                             (students_johnson : ℕ) (third_johnson_collected_25 : ℕ) (three_johnson_collected_0 : ℕ) (remaining_johnson_collected_10 : ℕ)
                             (hp : students_perez = 28) (hc1 : half_perez_collected_20 = 28 / 2) (hc2 : two_perez_collected_0 = 2) (hc3 : remaining_perez_collected_8 = 12)
                             (hj : students_johnson = 30) (jc1 : third_johnson_collected_25 = 30 / 3) (jc2 : three_johnson_collected_0 = 3) (jc3 : remaining_johnson_collected_10 = 18) :
    (half_perez_collected_20 * 20 + two_perez_collected_0 * 0 + remaining_perez_collected_8 * 8
    + third_johnson_collected_25 * 25 + three_johnson_collected_0 * 0 + remaining_johnson_collected_10 * 10) = 806 :=
by
  sorry

end total_cans_collected_l224_224687


namespace maximum_determinant_l224_224244

/-
Conditions:
1. v = (3, -2, 2)
2. w = (-1, 4, 1)
3. u is a unit vector.
-/

open Matrix

def vec_v : Fin 3 → ℝ := ![3, -2, 2]
def vec_w : Fin 3 → ℝ := ![-1, 4, 1]
def is_unit_vector (u : Fin 3 → ℝ) : Prop := ‖u‖ = 1

theorem maximum_determinant (u : Fin 3 → ℝ) (hu : is_unit_vector u) :
    let det := (u ⬝ (vec_v ⬝ vec_w))
    det = Real.sqrt 345 :=
sorry

end maximum_determinant_l224_224244


namespace complex_numbers_power_sum_magnitude_l224_224533

theorem complex_numbers_power_sum_magnitude 
  (a b c : ℂ) 
  (h₁ : complex.abs a = 1) 
  (h₂ : complex.abs b = 1) 
  (h₃ : complex.abs c = 1) 
  (h₄ : a^2 + b^2 + c^2 = 1) : 
  complex.abs (a^2020 + b^2020 + c^2020) = 1 := 
sorry

end complex_numbers_power_sum_magnitude_l224_224533


namespace triangle_area_ratio_l224_224519

-- Define the conditions of the problem as variables and hypotheses
variables {A B C D E F K M N : Type*}
variable (ABC : Triangle A B C)
variable (DEF : Triangle D E F)
variable (KMN : Triangle K M N)
variable (r R : ℝ)

-- Assume D, E, F are points of tangency of the incircle of triangle ABC
variable (tangency_points : PointsOfTangency ABC D E F)

-- Assume r is the radius of the incircle of triangle DEF
variable (incircle_def : Incircle DEF r)

-- Assume R is the radius of the circumcircle of triangle ABC
variable (circumcircle_abc : Circumcircle ABC R)

-- Assume KMN is the orthic triangle of DEF
variable (orthic_triangle : OrthicTriangle DEF KMN)

-- Define triangles are defined and have areas
axiom area_triangle_ABC : Area ABC ≠ 0
axiom area_triangle_DEF : Area DEF ≠ 0
axiom area_triangle_KMN : Area KMN ≠ 0

-- State the theorem to be proven in terms of given conditions and correct answers
theorem triangle_area_ratio (h : tangency_points ∧ incircle_def ∧ circumcircle_abc ∧ orthic_triangle) :
  Area KMN / Area ABC = r^2 / (4 * R^2) :=
sorry

end triangle_area_ratio_l224_224519


namespace peter_food_necessity_l224_224658

/-- Discuss the conditions  -/
def peter_horses (num_horses num_days : ℕ) (oats_per_meal grain_per_day : ℕ) (meals_per_day : ℕ) : ℕ :=
  let daily_oats := oats_per_meal * meals_per_day in
  let total_oats := daily_oats * num_days * num_horses in
  let total_grain := grain_per_day * num_days * num_horses in
  total_oats + total_grain

/-- Prove that Peter needs 132 pounds of food to feed his horses for 3 days -/
theorem peter_food_necessity : peter_horses 4 3 4 3 2 = 132 :=
  sorry

end peter_food_necessity_l224_224658


namespace balloon_permutation_count_l224_224930

-- Define the given conditions:
def balloon_letters : List Char := ['B', 'A', 'L', 'L', 'O', 'O', 'N']
def total_letters : Nat := 7
def count_L : Nat := 2
def count_O : Nat := 2

-- Statement to prove:
theorem balloon_permutation_count :
  (nat.factorial total_letters) / ((nat.factorial count_L) * (nat.factorial count_O)) = 1260 := by
  sorry

end balloon_permutation_count_l224_224930


namespace correct_incorrect_difference_multiple_of_9_l224_224019

theorem correct_incorrect_difference_multiple_of_9 (a b : ℕ) (h : a ≠ b) :
  (∃ x ∈ ({45, 46, 47, 48, 49} : Finset ℕ), 9 * (a - b).natAbs = x) ↔ x = 45 :=
by
  sorry

end correct_incorrect_difference_multiple_of_9_l224_224019


namespace repeating_decimal_eq_l224_224769

noncomputable def repeating_decimal : ℚ := 56 / 99

theorem repeating_decimal_eq : ∃ x : ℚ, x = repeating_decimal ∧ x = 56 / 99 :=
by
  use 56 / 99
  split
  all_goals { sorry }

end repeating_decimal_eq_l224_224769


namespace swimmer_speed_in_still_water_l224_224061

/-- Conditions of the problem. -/
variables (distance : ℝ) (time : ℝ) (current_speed : ℝ) (swimmer_speed : ℝ)
variables (h1 : distance = 12)
variables (h2 : time = 3.1578947368421053)
variables (h3 : current_speed = 1.2)

/-- The goal is to prove the swimmer's speed in still water. -/
theorem swimmer_speed_in_still_water : swimmer_speed = 5 :=
by
  -- Additional necessary calculations can be inserted here
  -- with appropriate expressions.
  sorry

end swimmer_speed_in_still_water_l224_224061


namespace lean4_proof_statement_l224_224857

-- Definitions of the conditions
def first_flip_count := 10
def total_outcomes := 2 ^ first_flip_count
def equal_heads_tails_count := Nat.choose first_flip_count (first_flip_count / 2)
def prob_equal_heads_tails : ℚ := equal_heads_tails_count / total_outcomes
def prob_diff_increases_when_equal : ℚ := 1
def prob_diff_increases_when_not_equal : ℚ := 1 / 2
def prob_not_equal_heads_tails : ℚ := 1 - prob_equal_heads_tails

-- Combining the probabilities
def prob_diff_increases : ℚ :=
  prob_equal_heads_tails * prob_diff_increases_when_equal +
  prob_not_equal_heads_tails * prob_diff_increases_when_not_equal

-- Given this probability, determine the reduced form and sum a + b
def probability_fraction := prob_diff_increases
def a := 319
def b := 512

def is_reduced_form (a b : ℕ) : Prop := Nat.gcd a b = 1

def problem_statement : Prop :=
  a + b = 831 ∧ probability_fraction = (a : ℚ) / b ∧ is_reduced_form a b

-- The statement we aim to prove
theorem lean4_proof_statement : problem_statement :=
by
  sorry

end lean4_proof_statement_l224_224857


namespace correct_propositions_l224_224688

def vertical_angles (α β : ℝ) [angle α] [angle β] : Prop :=
α = β

def complementary_angles_equal (α β : ℝ) [angle α] [angle β] : Prop :=
α = β → (90 - α) = (90 - β)

def parallel_transitive (a b c : ℝ) : Prop :=
a = b ∧ c = a → b = c 

def corresponding_angles (α β : ℝ) : Prop :=
α = β  -- This statement has issues as it should depend on some transversal

theorem correct_propositions :
  (vertical_angles α β) ∧
  (complementary_angles_equal α β) ∧
  (parallel_transitive a b c) :=
by
  sorry

end correct_propositions_l224_224688


namespace abhinav_bhupathi_total_l224_224843
noncomputable def total_amount (A B : ℝ) : ℝ := A + B

theorem abhinav_bhupathi_total :
  ∀ (A B : ℝ),
    (B = 484) →
    ((4 / 15) * A = (2 / 5) * B) →
    total_amount A B = 1210 :=
begin
  intros A B hB hEq,
  sorry
end

end abhinav_bhupathi_total_l224_224843


namespace avg_of_8_numbers_l224_224717

theorem avg_of_8_numbers
  (n : ℕ)
  (h₁ : n = 8)
  (sum_first_half : ℝ)
  (h₂ : sum_first_half = 158.4)
  (avg_second_half : ℝ)
  (h₃ : avg_second_half = 46.6) :
  ((sum_first_half + avg_second_half * (n / 2)) / n) = 43.1 :=
by
  sorry

end avg_of_8_numbers_l224_224717


namespace statement_A_is_correct_statement_B_is_incorrect_statement_C_is_correct_statement_D_is_incorrect_l224_224375

theorem statement_A_is_correct : 
  ∀ (x y : ℝ), (√3) * x + y + 1 = 0 → ∃ θ, Real.arctan (- √3) = Real.arctan (Real.tan 120) :=
sorry

theorem statement_B_is_incorrect : 
  ∀ (P : ℝ × ℝ), P = (2, 1) → ¬(∀ a b : ℝ, (∃ x y, y = -a/x ∧ x = b ∧ P = (x, y)) ∧ (x - y - 1 = 0)) :=
sorry

theorem statement_C_is_correct : 
  ∀ (x y : ℝ) (m : ℝ), (m * x + y + 2 - m = 0) → (1, -2) ∈ (x, y) :=
sorry

theorem statement_D_is_incorrect : 
  ∀ (a : ℝ), (a = -3 ∨ a = 0) → ax + 2ay + 1 = 0 ∧ (a - 1)x - (a + 1)y - 4 = 0 → ¬ perpendicular :=
sorry

end statement_A_is_correct_statement_B_is_incorrect_statement_C_is_correct_statement_D_is_incorrect_l224_224375


namespace range_of_a_l224_224539

noncomputable def f (a : ℝ) (x : ℝ) := 2 * a * x^2 + 4 * (a - 3) * x + 5

theorem range_of_a (a : ℝ) :
  (∀ x < 3, f a x ≤ f a (3 : ℝ)) ↔ 0 ≤ a ∧ a ≤ 3 / 4 :=
by
  sorry

end range_of_a_l224_224539


namespace probability_not_yellow_l224_224813

-- Define the conditions
def red_jelly_beans : Nat := 4
def green_jelly_beans : Nat := 7
def yellow_jelly_beans : Nat := 9
def blue_jelly_beans : Nat := 10

-- Definitions used in the proof problem
def total_jelly_beans : Nat := red_jelly_beans + green_jelly_beans + yellow_jelly_beans + blue_jelly_beans
def non_yellow_jelly_beans : Nat := total_jelly_beans - yellow_jelly_beans

-- Lean statement of the probability problem
theorem probability_not_yellow : 
  (non_yellow_jelly_beans : ℚ) / (total_jelly_beans : ℚ) = 7 / 10 := 
by 
  sorry

end probability_not_yellow_l224_224813


namespace hexagonal_prism_cross_section_l224_224020

theorem hexagonal_prism_cross_section (n : ℕ) (h₁: n ≥ 3) (h₂: n ≤ 8) : ¬ (n = 9):=
sorry

end hexagonal_prism_cross_section_l224_224020


namespace polynomial_first_degree_l224_224056

theorem polynomial_first_degree (P : ℕ → ℤ) (h1 : ∀ n : ℕ, 0 ≤ n → P(n) = P(n))
  (h2 : ∀ m : ℕ, ∃ x : ℕ, P(x) = 2^m) (h3 : leading_coefficient P = 1) : 
  ∃ c : ℕ, P(x) = x + c :=
sorry

end polynomial_first_degree_l224_224056


namespace tetrachloromethane_produced_l224_224484

def methane := "CH4"
def chlorine := "Cl2"
def tetrachloromethane := "CCl4"
def hydrogen_chloride := "HCl"

def balanced_reaction (moles_of_methane : ℕ) (moles_of_chlorine : ℕ) : Prop :=
  moles_of_methane = 1 ∧ moles_of_chlorine = 4

theorem tetrachloromethane_produced (moles_of_methane : ℕ) (moles_of_chlorine : ℕ) :
  balanced_reaction moles_of_methane moles_of_chlorine → ∃ moles_of_CCl4 : ℕ, moles_of_CCl4 = 1 :=
by
  intro h
  use 1
  sorry

end tetrachloromethane_produced_l224_224484


namespace locus_of_center_l224_224346

-- Definitions and conditions
variables {O O' k : Type} {A A' P : O} {C D C' D' : O}

-- Assume points lie on corresponding circles
variable (circle_O : ∀ (x : O), x = A → x ∈ O)
variable (circle_O' : ∀ (x : O'), x = A' → x ∈ O')
variable (circle_k : ∀ (x : O), x = P → x ∈ k)

-- Assume chords pass through A and A'
variable (chord_CD : ∀ (x : O), x ∈ CD ↔ x = A)
variable (chord_C'D' : ∀ (x : O), x ∈ C'D' ↔ x = A')

variable (points_on_k : ∀ (x : O), x ∈ k ↔ x = C ∨ x = D ∨ x = C' ∨ x = D')

-- Definition: radical axis of circles O and O'
def radical_axis (O O' : O) := ∀ (x : O), x ∈ O ∩ O' → ∃ P: O, P ∈ O ∧ P ∈ O'

-- Prove the geometric locus of center of k is a straight line perpendicular to AA'
theorem locus_of_center
  (O O' k : Type) (A A' P : O) (C D C' D' : O)
  (circle_O : ∀ (x : O), x = A → x ∈ O) (circle_O' : ∀ (x : O'), x = A' → x ∈ O')
  (circle_k : ∀ (x : O), x = P → x ∈ k)
  (chord_CD : ∀ (x : O), x ∈ CD ↔ x = A) (chord_C'D' : ∀ (x : O), x ∈ C'D' ↔ x = A')
  (points_on_k : ∀ (x : O), x ∈ k ↔ x = C ∨ x = D ∨ x = C' ∨ x = D')
  (radical_axis_O_O' : radical_axis O O')
  : ∀ ω : O, (ω ∈ k → ω ∉ A ∧ ω ∉ D ∧ ω ∉ C' ∧ ω ∉ D')
    → (∃ g: O, (g ⊥ AA') ∧ (ω ∈ g)) :=
sorry

end locus_of_center_l224_224346


namespace min_guests_l224_224030

theorem min_guests (total_food : ℕ) (max_food_per_guest : ℕ) 
  (h_total : total_food = 327) (h_max : max_food_per_guest = 2) : 
  total_food / max_food_per_guest = 163 ∧ total_food % max_food_per_guest ≠ 0 → 
  (total_food / max_food_per_guest) + 1 = 164 :=
by
  intros h_div h_mod
  rw [h_total, h_max] at h_div h_mod
  have h1 : 327 / 2 = 163 := by norm_num
  have h2 : 327 % 2 ≠ 0 := by norm_num
  exact eq.trans (by linarith) h_mod
  sorry

end min_guests_l224_224030


namespace birds_sold_ratio_l224_224830

-- Define the initial conditions
def initial_birds : Nat := 12
def initial_puppies : Nat := 9
def initial_cats : Nat := 5
def initial_spiders : Nat := 15
def adopted_puppies : Nat := 3
def loose_spiders : Nat := 7
def remaining_animals : Nat := 25

-- The question we need to prove, ensuring the correct ratio
theorem birds_sold_ratio : 
  ∃ birds_sold : Nat, birds_sold = 6 ∧ birds_sold.to_rat / initial_birds.to_rat = (1 : ℚ) / 2 :=
by
  sorry

end birds_sold_ratio_l224_224830


namespace player_Y_winning_strategy_l224_224513

theorem player_Y_winning_strategy (n : ℕ) (h : n ≥ 2) :
  (∃ N: ℕ, ∃ circles: finset (circle ℕ), 
   ∀ (c1 c2 : circle ℕ), c1 ≠ c2 → (intersections c1 c2).card = 2 ∧
   ∀ (c1 c2 c3 : circle ℕ), c1 ≠ c2 → c2 ≠ c3 → c1 ≠ c3 → (intersections c1 c2 ∩ intersections c2 c3 ∩ intersections c1 c3).card = 0 ∧ 
   ∀ (X Y : player), has_winning_strategy Y n) :=
by sorry

end player_Y_winning_strategy_l224_224513


namespace sum_binomial_minus_one_pow_sum_binomial_minus_one_pow_k_l224_224662

theorem sum_binomial_minus_one_pow (x : ℝ) (n : ℕ) :
  ∑ r in Finset.range (n + 1), (-1) ^ r * Nat.choose n r * (x - r) ^ n = n! :=
sorry

theorem sum_binomial_minus_one_pow_k (x : ℝ) (n k : ℕ)
  (h : k ≤ n) :
  ∑ r in Finset.range (n + 2), (-1) ^ r * Nat.choose (n + 1) r * (x - r) ^ k = 0 :=
sorry

end sum_binomial_minus_one_pow_sum_binomial_minus_one_pow_k_l224_224662


namespace balloon_arrangements_l224_224899

theorem balloon_arrangements :
  let n := 7
  let k1 := 2
  let k2 := 2
  (Nat.factorial n) / ((Nat.factorial k1) * (Nat.factorial k2)) = 1260 :=
by
  -- Use the conditions directly from the problem
  let n := 7
  let k1 := 2
  let k2 := 2
  -- Mathematically, the number of arrangements of the multiset
  have h : (Nat.factorial n) / ((Nat.factorial k1) * (Nat.factorial k2)) = 1260 := sorry
  exact h

end balloon_arrangements_l224_224899


namespace balloon_permutations_l224_224937

theorem balloon_permutations : 
  let balloons := "BALLOON".toList.length in
  let total_permutations := Nat.factorial balloons in
  let repetitive_L := Nat.factorial 2 in
  let repetitive_O := Nat.factorial 2 in
  (total_permutations / (repetitive_L * repetitive_O)) = 1260 := sorry

end balloon_permutations_l224_224937


namespace ball_radius_l224_224042

noncomputable def radius_of_ball (d h : ℝ) : ℝ :=
  let r := d / 2
  (325 / 20 : ℝ)

theorem ball_radius (d h : ℝ) (hd : d = 30) (hh : h = 10) :
  radius_of_ball d h = 16.25 := by
  sorry

end ball_radius_l224_224042


namespace sum_of_cubes_l224_224128

theorem sum_of_cubes (x y : ℝ) (h₁ : x + y = -1) (h₂ : x * y = -1) : x^3 + y^3 = -4 := by
  sorry

end sum_of_cubes_l224_224128


namespace track_width_l224_224059

theorem track_width (r_1 r_2 : ℝ) (h1 : r_2 = 20) (h2 : 2 * Real.pi * r_1 - 2 * Real.pi * r_2 = 20 * Real.pi) : r_1 - r_2 = 10 :=
sorry

end track_width_l224_224059


namespace balloon_arrangements_l224_224891

theorem balloon_arrangements :
  let n := 7
  let k1 := 2
  let k2 := 2
  (Nat.factorial n) / ((Nat.factorial k1) * (Nat.factorial k2)) = 1260 :=
by
  -- Use the conditions directly from the problem
  let n := 7
  let k1 := 2
  let k2 := 2
  -- Mathematically, the number of arrangements of the multiset
  have h : (Nat.factorial n) / ((Nat.factorial k1) * (Nat.factorial k2)) = 1260 := sorry
  exact h

end balloon_arrangements_l224_224891


namespace tangent_line_parabola_l224_224129

theorem tangent_line_parabola (a : ℝ) :
  (∀ x y : ℝ, y^2 = 32 * x → 4 * x + 3 * y + a = 0) → a = 18 :=
by
  sorry

end tangent_line_parabola_l224_224129


namespace problem_statement_l224_224515

section
variables (A B : ℝ × ℝ) (L_1 : ℝ → ℝ)

def midpoint (P Q : ℝ × ℝ) : ℝ × ℝ :=
  ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)

def line_passes_through (line : ℝ → ℝ) (P : ℝ × ℝ) : Prop :=
  P.2 = line P.1

def area_of_triangle (A B C : ℝ × ℝ) : ℝ :=
  abs((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2)) / 2

noncomputable def line_L1 (m : ℝ) : ℝ → ℝ :=
  λ x, m * x + 2

theorem problem_statement :
  A = (3, 2) →
  B = (-1, 5) →
  line_passes_through (line_L1 (5/2)) (midpoint A B) →
  (∃ C : ℝ × ℝ, line_passes_through (line_L1 (5/2)) C ∧ area_of_triangle A B C = 10 →
    (C = (29/9, 41/6) ∨ C = (-11/9, 1/6))) :=
by
  sorry
end

end problem_statement_l224_224515


namespace smallest_X_exists_l224_224621

def is_composed_of_0_and_1 (n : ℕ) : Prop := 
  ∀ d, digit d n → d = 0 ∨ d = 1

noncomputable def T : ℕ := 11111111100
noncomputable def X : ℕ := 308642525

theorem smallest_X_exists : (is_composed_of_0_and_1 T) ∧ (X = T / 36) ∧ (T % 36 = 0) → X = 308642525 :=
sorry

end smallest_X_exists_l224_224621


namespace distance_between_vertices_of_hyperbola_l224_224115

theorem distance_between_vertices_of_hyperbola : 
  let a := Real.sqrt 27
  in 2 * a = 6 * Real.sqrt 3 :=
by 
  sorry

end distance_between_vertices_of_hyperbola_l224_224115


namespace gemstones_needed_for_sets_l224_224664

-- Define the number of magnets per earring
def magnets_per_earring : ℕ := 2

-- Define the number of buttons per earring as half the number of magnets
def buttons_per_earring (magnets : ℕ) : ℕ := magnets / 2

-- Define the number of gemstones per earring as three times the number of buttons
def gemstones_per_earring (buttons : ℕ) : ℕ := 3 * buttons

-- Define the number of earrings per set
def earrings_per_set : ℕ := 2

-- Define the number of sets
def sets : ℕ := 4

-- Prove that Rebecca needs 24 gemstones for 4 sets of earrings given the conditions
theorem gemstones_needed_for_sets :
  gemstones_per_earring (buttons_per_earring magnets_per_earring) * earrings_per_set * sets = 24 :=
by
  sorry

end gemstones_needed_for_sets_l224_224664


namespace triangle_medians_l224_224225

theorem triangle_medians
  (A B C M K O : Type)
  (med_AM : A M ∈ (segment A (midpoint (B, C))))
  (med_BK : B K ∈ (segment B (midpoint (A, C))))
  (AM_eq : dist A M = 3)
  (BK_eq : dist B K = 5) :
  (∀ AB perimeter, AB ≤ 6 ∧ perimeter ≤ 22 → false) ∧
  (∀ AB perimeter, AB > 6 ∨ perimeter > 22 → false) ∧
  (∀ AB perimeter, (AB = dist A B) ∧ (perimeter = dist A B + dist B C + dist C A) → false) :=
by sorry

end triangle_medians_l224_224225


namespace obtain_any_natural_l224_224439

-- Define the operations
def op1 (x : ℕ) : ℕ := 3 * x + 1
def op2 (x : ℤ) : ℕ := (x / 2).to_nat

-- Theorem statement: Starting from 1, any natural number n can be obtained using op1 and op2
theorem obtain_any_natural (n : ℕ) : ∃ (f : ℕ → ℕ), f 1 = n ∧ ∀ i, f (i + 1) = op1 (f i) ∨ f (i + 1) = op2 (f i) := sorry

end obtain_any_natural_l224_224439


namespace perpendicular_vectors_l224_224506

/-- Given vectors a and b, prove that m = 6 if a is perpendicular to b -/
theorem perpendicular_vectors {m : ℝ} (h₁ : (1, 5, -2) = (1, 5, -2)) (h₂ : ∃ m : ℝ, (m, 2, m+2) = (m, 2, m+2)) (h₃ : (1 * m + 5 * 2 + (-2) * (m + 2) = 0)) :
  m = 6 :=
sorry

end perpendicular_vectors_l224_224506


namespace rationalize_denominator_l224_224291

theorem rationalize_denominator :
  ∃ (A B C D E F : ℤ),
    (A = 5 ∧ B = 4 ∧ C = -1 ∧ D = 1 ∧ E = 70 ∧ F = 20 ∧ 
    A + B + C + D + E + F = 99 ∧ 
    (1:ℚ) / (Real.sqrt 2 + Real.sqrt 5 + Real.sqrt 7) = 
    (A * Real.sqrt 2 + B * Real.sqrt 5 + C * Real.sqrt 7 + D * Real.sqrt E) / F) :=
begin
  use [5, 4, -1, 1, 70, 20],
  split,
  { exact rfl }, split,
  { exact rfl }, split,
  { exact rfl }, split,
  { exact rfl }, split,
  { exact rfl }, split,
  { exact rfl }, split,
  { norm_num }, 
  { field_simp [Real.sqrt_nonneg, Real.sqrt_sq], 
    -- Applying steps to rationalize denominator and verifying the result is required.
    -- Leaving as sorry as no proof is required per instructions.
    sorry
  }
end

end rationalize_denominator_l224_224291


namespace ball_hits_ground_at_l224_224316

-- Define the quadratic equation for the height of the ball.
def height (t : ℝ) : ℝ := -16 * t^2 + 22 * t + 45

-- Prove that the ball hits the ground at t = 5 / 2
theorem ball_hits_ground_at {
  t : ℝ
} : height(t) = 0 → t = 5 / 2 := by
  sorry

end ball_hits_ground_at_l224_224316


namespace find_C_l224_224600

-- Defining the angles and sides
variables (A C : ℝ) (a b c : ℝ)

-- Given conditions
def condition1 : Prop := A - C = π / 2
def condition2 : Prop := a + c = (sqrt 2) * b

-- The proof goal
theorem find_C (h1 : condition1 A C) (h2 : condition2 a b c) : 
  C = π / 12 :=
sorry

end find_C_l224_224600


namespace increasing_or_decreasing_subsequence_l224_224135

theorem increasing_or_decreasing_subsequence {α : Type*} [linear_order α]
  (a : ℕ → α) (m n : ℕ) :
  ∃ s : finset (fin (m * n + 1)), (s.card ≥ m + 1 ∧ ∀ i j ∈ s, i < j → a i ≤ a j) ∨ (s.card ≥ n + 1 ∧ ∀ i j ∈ s, i < j → a i > a j) :=
sorry

end increasing_or_decreasing_subsequence_l224_224135


namespace eq_triangle_property_l224_224603

-- Definitions of equilateral triangle and angles
structure EquilateralTriangle (A B C : Point) : Prop :=
  (AB_eq_BC : dist A B = dist B C)
  (BC_eq_CA : dist B C = dist C A)

def Point := ℝ × ℝ

noncomputable def dist (p1 p2 : Point) : ℝ :=
  ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2).sqrt

noncomputable def angle (p1 p2 p3 : Point) : ℝ := sorry

theorem eq_triangle_property (A B C M : Point) 
  (h1 : EquilateralTriangle A B C)
  (h2 : angle A M C = 150) :
  dist A M ^ 2 + dist C M ^ 2 = dist B M ^ 2 := 
sorry

end eq_triangle_property_l224_224603


namespace problem_solution_l224_224605

-- Definitions of the probabilities of getting heads for each coin type
def fair_coin_prob : ℚ := 1 / 2
def biased_coin_prob : ℚ := 2 / 5

-- Number of fair coins and the number of biased coins
def num_fair_coins : ℕ := 2
def num_biased_coins : ℕ := 1

-- Function to calculate the probability of getting same number of heads for Jackie and Alex
noncomputable def probability_same_heads : ℚ :=
  let fair_coin_gf := (1 + x) ^ num_fair_coins
  let biased_coin_gf := 3 + 2 * x
  let combined_gf := fair_coin_gf * biased_coin_gf
  let coefficients := [3, 8, 7, 2] -- extracted from the combined generating function
  let total_ways := (coefficients.sum)^2
  let matching_ways := (3^2 + 8^2 + 7^2 + 2^2)
  matching_ways / total_ways

theorem problem_solution :
  let p : ℕ := 63
  let q : ℕ := 200
  p + q = 263 :=
by 
  sorry

end problem_solution_l224_224605


namespace intersection_M_N_l224_224147

open Set

-- Define the sets M and N
def M : Set ℝ := {-2, 0, 1}
def N : Set ℝ := {x | -1 < x ∧ x < 2}

-- State the theorem
theorem intersection_M_N :
  M ∩ N = {0, 1} := 
sorry

end intersection_M_N_l224_224147


namespace propositions_correct_l224_224259

-- Problem statement
theorem propositions_correct (S : ℕ → ℝ) (a : ℕ → ℝ) (b : ℕ → ℝ) (h1 : ∀ n, S(n + 1) = S n + a(n + 1)) (h2 : ∀ n, S n = a n * a n + b n) :
  (¬ ∀ n, a (n + 1) = a n) ∧
  (¬ (∀ n, S n = 1 - (-1)^n → ∀ n, S(n + 1) - S n = 2 * (-1)^n → ∀ n, a(n + 1) / a n = -1)) ∧
  (∀ n, S n = 1 - (-1)^n → ∀ n, S(n + 1) - S n = 2 * (-1)^n → ∀ n, a(n + 1) / a n = -1) ∧
  (∀ n, ∃ d ∈ ℝ, S(n + 1) - S n = d ∧ S(2 * n) - S n = d ∧ S(3 * n) - S(2 * n) = d) :=
begin
  -- Proof not included as per instructions
  sorry
end

end propositions_correct_l224_224259


namespace spring_excursion_participants_l224_224348

theorem spring_excursion_participants (water fruit neither both total : ℕ) 
  (h_water : water = 80) 
  (h_fruit : fruit = 70) 
  (h_neither : neither = 6) 
  (h_both : both = total / 2) 
  (h_total_eq : total = water + fruit - both + neither) : 
  total = 104 := 
  sorry

end spring_excursion_participants_l224_224348


namespace triangle_medians_l224_224226

theorem triangle_medians
  (A B C M K O : Type)
  (med_AM : A M ∈ (segment A (midpoint (B, C))))
  (med_BK : B K ∈ (segment B (midpoint (A, C))))
  (AM_eq : dist A M = 3)
  (BK_eq : dist B K = 5) :
  (∀ AB perimeter, AB ≤ 6 ∧ perimeter ≤ 22 → false) ∧
  (∀ AB perimeter, AB > 6 ∨ perimeter > 22 → false) ∧
  (∀ AB perimeter, (AB = dist A B) ∧ (perimeter = dist A B + dist B C + dist C A) → false) :=
by sorry

end triangle_medians_l224_224226


namespace Isla_investment_change_l224_224189

theorem Isla_investment_change :
  ∀ (initial amount loss_percent gain_percent : ℝ),
    initial = 150 → 
    loss_percent = 0.10 → 
    gain_percent = 0.25 → 
    let after_loss := initial * (1 - loss_percent) in
    let after_gain := after_loss * (1 + gain_percent) in
    (after_gain - initial) / initial * 100 = 12.5 :=
by
  intros initial amount loss_percent gain_percent
  assume h_initial h_loss h_gain
  let after_loss := initial * (1 - loss_percent)
  let after_gain := after_loss * (1 + gain_percent)
  show (after_gain - initial) / initial * 100 = 12.5
  sorry

end Isla_investment_change_l224_224189


namespace irrational_pi_l224_224737

def is_irrational (x : ℝ) : Prop := ¬ ∃ (a b : ℤ), b ≠ 0 ∧ x = a / b

theorem irrational_pi : 
  (is_irrational (3.1415926 : ℝ) = false) ∧
  (is_irrational ((1/3 : ℝ)) = false)  ∧
  (is_irrational (sqrt 16) = false)  ∧
  (is_irrational (π) = true) :=
by {
  sorry
}

end irrational_pi_l224_224737


namespace Kolya_is_acting_as_collection_agency_l224_224213

-- Definitions for the conditions given in the problem
def Katya_lent_books_to_Vasya : Prop := ∃ books : Type, ∀ b : books, ¬ b ∈ Katya's_collection
def Vasya_promised_to_return_books_in_a_month_but_failed : Prop := ∀ t : Time, t ≥ 1 month → ¬returned books by Vasya
def Katya_asked_Kolya_to_get_books_back : Prop := ∀ k : Kolya, ∀ v : Vasya, asked Katya (k to get books back from v)
def Kolya_agrees_but_wants_a_reward : Prop := ∃ reward : Book, Kolya_gets reward

-- Define the property of Kolya being a collection agency
def Kolya_is_collection_agency : Prop :=
  Katya_lent_books_to_Vasya ∧
  Vasya_promised_to_return_books_in_a_month_but_failed ∧
  Katya_asked_Kolya_to_get_books_back ∧
  Kolya_agrees_but_wants_a_reward

-- The theorem to prove
theorem Kolya_is_acting_as_collection_agency :
  Kolya_is_collection_agency :=
sorry

end Kolya_is_acting_as_collection_agency_l224_224213


namespace cards_received_at_home_l224_224644

-- Definitions based on the conditions
def initial_cards := 403
def total_cards := 690

-- The theorem to prove the number of cards received at home
theorem cards_received_at_home : total_cards - initial_cards = 287 :=
by
  -- Proof goes here, but we use sorry as a placeholder.
  sorry

end cards_received_at_home_l224_224644


namespace paving_cost_l224_224386

def length : ℝ := 5.5
def width : ℝ := 3.75
def rate : ℝ := 1000
def area : ℝ := length * width
def cost : ℝ := area * rate

theorem paving_cost :
  cost = 20625 := by sorry

end paving_cost_l224_224386


namespace all_positive_integers_are_clever_l224_224734

theorem all_positive_integers_are_clever : ∀ n : ℕ, 0 < n → ∃ a b c d : ℕ, 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ n = (a^2 - b^2) / (c^2 + d^2) := 
by
  intros n h_pos
  sorry

end all_positive_integers_are_clever_l224_224734


namespace average_balance_correct_l224_224844

-- Define the monthly balances
def january_balance : ℕ := 120
def february_balance : ℕ := 240
def march_balance : ℕ := 180
def april_balance : ℕ := 180
def may_balance : ℕ := 210
def june_balance : ℕ := 300

-- List of all balances
def balances : List ℕ := [january_balance, february_balance, march_balance, april_balance, may_balance, june_balance]

-- Define the function to calculate the average balance
def average_balance (balances : List ℕ) : ℕ :=
  (balances.sum / balances.length)

-- Define the target average balance
def target_average_balance : ℕ := 205

-- The theorem we need to prove
theorem average_balance_correct :
  average_balance balances = target_average_balance :=
by
  sorry

end average_balance_correct_l224_224844


namespace enlarged_circles_cover_triangle_l224_224350

theorem enlarged_circles_cover_triangle
  (O1 O2 O3 X Y Z : Point) 
  (r1 r2 r3 : ℝ) 
  (h1 : distance O1 O2 = r1 + r2)
  (h2 : distance O2 O3 = r2 + r3)
  (h3 : distance O3 O1 = r3 + r1)
  (k : ℝ)
  (hk : k > 2 / Real.sqrt 3)
  (r1_new : ℝ := k * r1) 
  (r2_new : ℝ := k * r2) 
  (r3_new : ℝ := k * r3) :
  ∀ (P : Point), (inside_triangle P X Y Z) → (distance P O1 ≤ r1_new) ∨ (distance P O2 ≤ r2_new) ∨ (distance P O3 ≤ r3_new) :=
by
  sorry

end enlarged_circles_cover_triangle_l224_224350


namespace sum_of_angles_equal_partition_l224_224141

theorem sum_of_angles_equal_partition (n : ℕ) (hn : n > 1) (pts : Fin (2 * n) → Real × Real) 
  (hcollinear : ∀ (i j k : Fin (2 * n)), i ≠ j → j ≠ k → k ≠ i → 
  ¬collinear ({pts i, pts j, pts k} : Set (Real × Real))) :
  ∃ (σ : Equiv.Perm (Fin (2 * n))),
    ∃ (angles : Fin (2 * n) → ℝ), (∀ i, 0 < angles i ∧ angles i < 180) ∧ 
    ((∑ i in Finset.filter (λ j, j < n) (Finset.univ : Finset (Fin (2 * n))), angles (σ i)) = 
    ∑ i in Finset.filter (λ j, n ≤ j) (Finset.univ : Finset (Fin (2 * n))), angles (σ i)) := 
sorry

end sum_of_angles_equal_partition_l224_224141


namespace balloon_permutations_l224_224955

theorem balloon_permutations : 
  let total_letters := 7
  let repetitions_L := 2
  let repetitions_O := 2
  (Nat.factorial total_letters) / ((Nat.factorial repetitions_L) * (Nat.factorial repetitions_O)) = 1260 := 
by
  sorry

end balloon_permutations_l224_224955


namespace balloon_permutations_l224_224966

theorem balloon_permutations : (∀ (arrangements : ℕ),
  arrangements = nat.factorial 7 / (nat.factorial 2 * nat.factorial 2) → arrangements = 1260) :=
begin
  intros arrangements h,
  rw [nat.factorial_succ, nat.factorial],
  sorry,
end

end balloon_permutations_l224_224966


namespace number_of_lattice_points_in_triangle_l224_224281

theorem number_of_lattice_points_in_triangle (N L S : ℕ) (A B O : (ℕ × ℕ)) :
  (A = (0, 30)) →
  (B = (20, 10)) →
  (O = (0, 0)) →
  (S = 300) →
  (L = 60) →
  S = N + L / 2 - 1 →
  N = 271 :=
by
  intros hA hB hO hS hL hPick
  sorry

end number_of_lattice_points_in_triangle_l224_224281


namespace coefficient_of_x_squared_term_l224_224457

noncomputable def x := polynomial.X

theorem coefficient_of_x_squared_term :
  polynomial.coeff (x^2 * (1 + x + x^2) * (x - 1 / x)^6) 2 = -5 :=
by {
  -- The proof steps would be inserted here
  sorry
}

end coefficient_of_x_squared_term_l224_224457


namespace bee_directions_at_distance_14_l224_224005

def bee_pattern_position (A B : ℕ → ℤ × ℤ × ℤ) (n : ℕ) : Prop := 
  A n = (2 * n, 2 * n, n) ∧ B n = (n, n, -2 * n)

theorem bee_directions_at_distance_14 :
  ∃ n : ℕ, 
  bee_pattern_position (λ n, (2 * n, 2 * n, n)) (λ n, (n, n, -2 * n)) n ∧ 
  sqrt ((2 * n - n)^2 + (2 * n - n)^2 + (n - (-2 * n))^2) = 14 ∧ 
  ((n * 3 + 1) % 3 = 1 ∨ (n * 3 + 1) % 3 = 2) ∧ 
  ((n * 3 + 2) % 3 = 2 ∨ (n * 3 + 2) % 3 = 0) 
  :=
by {
  sorry
}

end bee_directions_at_distance_14_l224_224005


namespace toothpaste_runs_out_in_two_days_l224_224711

noncomputable def toothpaste_capacity := 90
noncomputable def dad_usage_per_brushing := 4
noncomputable def mom_usage_per_brushing := 3
noncomputable def anne_usage_per_brushing := 2
noncomputable def brother_usage_per_brushing := 1
noncomputable def sister_usage_per_brushing := 1

noncomputable def dad_brushes_per_day := 4
noncomputable def mom_brushes_per_day := 4
noncomputable def anne_brushes_per_day := 4
noncomputable def brother_brushes_per_day := 4
noncomputable def sister_brushes_per_day := 2

noncomputable def total_daily_usage :=
  dad_usage_per_brushing * dad_brushes_per_day + 
  mom_usage_per_brushing * mom_brushes_per_day + 
  anne_usage_per_brushing * anne_brushes_per_day + 
  brother_usage_per_brushing * brother_brushes_per_day + 
  sister_usage_per_brushing * sister_brushes_per_day

theorem toothpaste_runs_out_in_two_days :
  toothpaste_capacity / total_daily_usage = 2 := by
  -- Proof omitted
  sorry

end toothpaste_runs_out_in_two_days_l224_224711


namespace base_b_representation_l224_224180

theorem base_b_representation (b : ℕ) (h₁ : 1 * b + 5 = n) (h₂ : n^2 = 4 * b^2 + 3 * b + 3) : b = 7 :=
by {
  sorry
}

end base_b_representation_l224_224180


namespace proof_geometric_sequence_l224_224512

variable {d : ℚ} (h : d ≠ 0)

def arithmetic_sequence (n : ℕ) : ℚ := 
  if n = 0 then 0 
  else match n with
       | 1 => -4 * d
       | _ => -4 * d + (n - 1) * d

def sum_first_n_terms (n : ℕ) : ℚ :=
  n * (arithmetic_sequence d 1 + arithmetic_sequence d n) / 2

theorem proof_geometric_sequence (h : d ≠ 0) : 
  ({a_1 := arithmetic_sequence d 1, a_3 := arithmetic_sequence d 3, a_4 := arithmetic_sequence d 4}) → 
  ((arithmetic_sequence d 3)^2 = (arithmetic_sequence d 1) * (arithmetic_sequence d 4)) → 
  (sum_first_n_terms d 3 - sum_first_n_terms d 2) / (sum_first_n_terms d 5 - sum_first_n_terms d 3) = 2 := 
by {
  -- proof here
  sorry
}

end proof_geometric_sequence_l224_224512


namespace team_selection_count_l224_224051

theorem team_selection_count (n k : ℕ) (h_n : n = 22) (h_k : k = 8) : nat.choose n k = 319770 :=
by
  rw [h_n, h_k]
  exact nat.choose_eq_factorial_div_factorial (by norm_num) (by norm_num)
  sorry

end team_selection_count_l224_224051


namespace total_payment_correct_l224_224006

def payment_y : ℝ := 318.1818181818182
def payment_ratio : ℝ := 1.2
def payment_x : ℝ := payment_ratio * payment_y
def total_payment : ℝ := payment_x + payment_y

theorem total_payment_correct :
  total_payment = 700.00 :=
sorry

end total_payment_correct_l224_224006


namespace number_smaller_than_neg_one_l224_224070

theorem number_smaller_than_neg_one :
  ∃ (x : ℝ), (x = -3) ∧ (x < -1) :=
begin
  sorry
end

end number_smaller_than_neg_one_l224_224070


namespace total_walnut_trees_l224_224342

variable (t1 t2 t : ℕ)
variable (h1 : t1 = 33)
variable (h2 : t2 = 44)

theorem total_walnut_trees : t1 + t2 = 77 :=
by
  rw [h1, h2]
  exact rfl

end total_walnut_trees_l224_224342


namespace equilateral_triangles_on_surface_count_l224_224453

def point := ℝ × ℝ × ℝ

def cube_points : set point := 
  { (x, y, z) | x ∈ {0, 2, 4} ∧ y ∈ {0, 2, 4} ∧ z ∈ {0, 2, 4} }

def is_on_surface (p : point) : Prop :=
  let (x, y, z) := p in x = 0 ∨ x = 2 ∨ x = 4 ∨ y = 0 ∨ y = 2 ∨ y = 4 ∨ z = 0 ∨ z = 2 ∨ z = 4

def equilateral_triangle (a b c : point) : Prop :=
  let dist := λ (u v : point), real.sqrt ((u.1 - v.1)^2 + (u.2 - v.2)^2 + (u.3 - v.3)^2) in
  dist a b = dist b c ∧ dist b c = dist a c ∧ dist a b ≠ 0

noncomputable def count_equilateral_triangles_on_surface : ℕ :=
  { t : set point | ∃ a b c, t = {a, b, c} ∧ 
    a ∈ cube_points ∧ b ∈ cube_points ∧ c ∈ cube_points ∧ 
    is_on_surface a ∧ is_on_surface b ∧ is_on_surface c ∧
    equilateral_triangle a b c }.card

theorem equilateral_triangles_on_surface_count :
  count_equilateral_triangles_on_surface = 12 := 
by
  sorry

end equilateral_triangles_on_surface_count_l224_224453


namespace circles_orthogonality_l224_224726

open EuclideanGeometry

theorem circles_orthogonality (C1 C2 : Circle) (A B P Q M N C D E : Point) 
  (intersect : A ∈ C1 ∧ A ∈ C2 ∧ B ∈ C1 ∧ B ∈ C2) 
  (P_on_C1 : P ∈ C1) (Q_on_C2 : Q ∈ C2) 
  (AP_AQ_eq : dist A P = dist A Q) 
  (PQ_intersect_MN : LineSegment.mk P Q ∩ C1 = {M} ∧ LineSegment.mk P Q ∩ C2 = {N})
  (C_center_BP : C.center = arc C1 B P ∧ arc C1 B P = {C}) 
  (D_center_BQ : D.center = arc C2 B Q ∧ arc C2 B Q = {D})
  (E_intersection : E = Line.mk C M ∩ Line.mk D N) : 
  perp (Line.mk A E) (Line.mk C D) := 
sorry

end circles_orthogonality_l224_224726


namespace problem1_problem2_l224_224553

noncomputable theory -- Enable non-computable definitions

variables (x y z : ℝ)

-- The first proof problem
theorem problem1 (h1 : x + y + z = 1) (h2 : x^2 + y^2 + z^2 = 2) (h3 : x^3 + y^3 + z^3 = 3) : 
  x * y * z = 1 / 6 :=
sorry

-- The second proof problem
theorem problem2 (h4 : x + y + z = 1) (h5 : x^2 + y^2 + z^2 = 2) (h6 : x^3 + y^3 + z^3 = 3) (h7 : x * y * z = 1 / 6) : 
  x^4 + y^4 + z^4 = 25 / 6 :=
sorry

end problem1_problem2_l224_224553


namespace max_possible_salary_l224_224407

-- Define the given conditions as parameters
variable (num_players : ℕ := 25)
variable (min_salary : ℕ := 12000)
variable (total_salary_limit : ℕ := 800000)

-- The statement we need to prove
theorem max_possible_salary : 
  ∃ max_salary, max_salary = 512000 ∧ 
  (∀ (salaries : Fin num_players → ℕ), 
   (∀ i, salaries i ≥ min_salary) ∧ 
   (salaries.sum ≤ total_salary_limit) → 
   ∃ i, salaries i = max_salary) := 
sorry

end max_possible_salary_l224_224407


namespace tangent_line_at_one_extreme_values_l224_224160

noncomputable def f (x : ℝ) : ℝ := -Real.log x + (1 / (2 * x)) + (3 / 2) * x + 1

theorem tangent_line_at_one :
  let f' (x : ℝ) : ℝ := -1 / x - 1 / (2 * x^2) + 3 / 2,
      tangent_line (x : ℝ) := 3 in
  (f' 1 = 0 ∧ f 1 = 3 ∧ tangent_line 1 = f 1) :=
begin
  intros,
  split,
  { simp [f'], linarith },
  split,
  { simp [f], ring },
  { simp },
end

theorem extreme_values :
  ∃ x ∈ (0 : ℝ, ∞), (∀ y ∈ (0 : ℝ, x), f y > f x) ∧ (∀ y ∈ (x : ℝ, ∞), f y > f x) :=
begin
  use 1,
  simp [f],
  sorry
end

end tangent_line_at_one_extreme_values_l224_224160


namespace area_of_trapezoid_ABCD_l224_224589

-- Definitions of points, lines, and areas in the trapezoid
variables {A B C D E : Type} 
variables {area_ABE area_ADE area_BCE area_CDE : ℝ}
variables {trapezoid : Prop}

-- Given conditions
def AB_parallel_CD (AB CD : A) : Prop := ∀ (x : ℝ), true -- AB is parallel to CD
def diagonals_intersect_at_E (AC BD E : A) : Prop := ∀ (y : ℝ), true -- diagonals AC and BD intersect at E
def area_of_triangle_ABE (area_ABE : ℝ) : Prop := area_ABE = 72 -- Area of triangle ABE is 72 square units
def area_of_triangle_ADE (area_ADE : ℝ) : Prop := area_ADE = 32 -- Area of triangle ADE is 32 square units

-- Problem: Prove that the area of trapezoid ABCD is 168 square units
theorem area_of_trapezoid_ABCD 
  (h1 : AB_parallel_CD A B)
  (h2 : diagonals_intersect_at_E C D E)
  (h3 : area_of_triangle_ABE area_ABE)
  (h4 : area_of_triangle_ADE area_ADE)
: ∃ (area_ABCD : ℝ), area_ABCD = 168 :=
sorry

end area_of_trapezoid_ABCD_l224_224589


namespace repeating_decimal_to_fraction_l224_224773

theorem repeating_decimal_to_fraction : (0.5656565656 : ℚ) = 56 / 99 :=
  by
    sorry

end repeating_decimal_to_fraction_l224_224773


namespace probability_first_heart_second_spades_or_clubs_l224_224358

-- Defining probabilities and conditions in the problem
variable (deck : Finset ℕ)
variable (n_cards : ℕ)
variable (hearts : Finset ℕ)
variable (spades : Finset ℕ)
variable (clubs : Finset ℕ)

-- Definitions of card sets
def is_standard_deck := deck.card = 52
def is_hearts_set := hearts.card = 13
def is_spades_set := spades.card = 13
def is_clubs_set := clubs.card = 13

-- Defining events
def first_card_heart : Prop := (13 : ℝ) / (52 : ℝ) = 1 / 4
def second_card_spades_or_clubs : Prop := (26 : ℝ) / (51 : ℝ) = 26 / 51

-- Proof goal
theorem probability_first_heart_second_spades_or_clubs :
  is_standard_deck deck →
  is_hearts_set hearts →
  is_spades_set spades →
  is_clubs_set clubs →
  first_card_heart →
  second_card_spades_or_clubs →
  (1 / 4 * 26 / 51 = 13 / 102) :=
by
  intro h1 h2 h3 h4 h5 h6
  sorry

end probability_first_heart_second_spades_or_clubs_l224_224358


namespace lucy_last_10_shots_l224_224423

variable (shots_30 : ℕ) (percentage_30 : ℚ) (total_shots : ℕ) (percentage_40 : ℚ)
variable (shots_made_30 : ℕ) (shots_made_40 : ℕ) (shots_made_last_10 : ℕ)

theorem lucy_last_10_shots 
    (h1 : shots_30 = 30) 
    (h2 : percentage_30 = 0.60) 
    (h3 : total_shots = 40) 
    (h4 : percentage_40 = 0.62 )
    (h5 : shots_made_30 = Nat.floor (percentage_30 * shots_30)) 
    (h6 : shots_made_40 = Nat.floor (percentage_40 * total_shots))
    (h7 : shots_made_last_10 = shots_made_40 - shots_made_30) 
    : shots_made_last_10 = 7 := sorry

end lucy_last_10_shots_l224_224423


namespace max_f_value_range_b_plus_c_l224_224169

open Real

-- Definitions for problem 1
def m (x : ℝ) : ℝ × ℝ := (sin x, 1)
def n (x : ℝ) : ℝ × ℝ := (√3 * cos x, 1 / 2 * cos (2 * x))
def f (x : ℝ) := (m x).fst * (n x).fst + (m x).snd * (n x).snd

-- Definitions for problem 2
noncomputable def A := π / 3
def side_a : ℝ := 2
def f_A := (m A).fst * (n A).fst + (m A).snd * (n A).snd = 1 / 2

-- Theorem for problem 1
theorem max_f_value : ∃ (x : ℝ), ∀ (k : ℤ), f x = 1 := sorry

-- Theorem for problem 2
theorem range_b_plus_c (b c : ℝ) : f_A → 2 < b + c ∧ b + c ≤ 4 := sorry

end max_f_value_range_b_plus_c_l224_224169


namespace f_n_pos_l224_224638

noncomputable def f_n (n : ℕ) (x : ℝ) : ℝ :=
  ∑ k in finset.range (n - 1), nat.choose n (k + 2) * x^(k)

theorem f_n_pos (n : ℕ) (x : ℝ) (hn : n ≥ 2) (hx_pos : x > -1) (hx_ne_zero : x ≠ 0) : f_n n x > 0 := by
  sorry

end f_n_pos_l224_224638


namespace tangent_line_curve_l224_224694

theorem tangent_line_curve {k b a : ℝ}
  (h1 : 3 = 2 * k + b) 
  (h2 : 3 = 9 + 2 * a) 
  (h3 : k = 3 * 2^2 + a) :
  b = -15 :=
begin
  sorry
end

end tangent_line_curve_l224_224694


namespace projection_and_symmetric_point_l224_224203

def point (x y z : ℝ) : ℝ × ℝ × ℝ := (x, y, z)

def projection_onto_xOy_plane (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (p.1, p.2, 0)

def symmetric_with_respect_to_xOy_plane (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (p.1, p.2, -p.3)

theorem projection_and_symmetric_point : 
  let P := point 2 3 4 in
  projection_onto_xOy_plane P = (2, 3, 0) ∧
  symmetric_with_respect_to_xOy_plane P = (2, 3, -4) := 
by
  let P := point 2 3 4
  show projection_onto_xOy_plane P = (2, 3, 0) ∧ symmetric_with_respect_to_xOy_plane P = (2, 3, -4)
  sorry

end projection_and_symmetric_point_l224_224203


namespace a2b_etc_ge_9a2b2c2_l224_224492

theorem a2b_etc_ge_9a2b2c2 (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a^2 * b + b^2 * c + c^2 * a) * (a * b^2 + b * c^2 + c * a^2) ≥ 9 * a^2 * b^2 * c^2 :=
by
  sorry

end a2b_etc_ge_9a2b2c2_l224_224492


namespace percent_of_rectangle_area_inside_square_l224_224834

theorem percent_of_rectangle_area_inside_square
  (s : ℝ)  -- Let the side length of the square be \( s \).
  (width : ℝ) (length: ℝ)
  (h1 : width = 3 * s)  -- The width of the rectangle is \( 3s \).
  (h2 : length = 2 * width) :  -- The length of the rectangle is \( 2 * width \).
  (s^2 / (length * width)) * 100 = 5.56 :=
by
  sorry

end percent_of_rectangle_area_inside_square_l224_224834


namespace repeating_decimal_to_fraction_l224_224774

theorem repeating_decimal_to_fraction : (0.5656565656 : ℚ) = 56 / 99 :=
  by
    sorry

end repeating_decimal_to_fraction_l224_224774


namespace balloon_arrangement_count_l224_224970

-- Definitions of letter frequencies for the word BALLOON
def letter_frequencies : List (Char × Nat) := [('B', 1), ('A', 1), ('L', 2), ('O', 2), ('N', 1)]

-- Total number of letters
def total_letters := 7

-- The formula for the number of unique arrangements of the letters
noncomputable def arrangements :=
  (Nat.factorial total_letters) / 
  (Nat.factorial 1 * Nat.factorial 1 * Nat.factorial 2 * Nat.factorial 2 * Nat.factorial 1)

-- The theorem to prove the number of ways to arrange the letters in "BALLOON"
theorem balloon_arrangement_count : arrangements = 1260 :=
  sorry

end balloon_arrangement_count_l224_224970


namespace fraction_of_repeating_decimal_l224_224119

theorem fraction_of_repeating_decimal :
  ∃ (f : ℚ), f = 0.73 ∧ f = 73 / 99 := by
  sorry

end fraction_of_repeating_decimal_l224_224119


namespace exists_subsequence_converging_to_sqrt2_l224_224448

noncomputable def converges_to_sqrt2_subsequence (s : ℕ × ℕ → ℝ) :=
  ∃ f : ℕ → ℕ × ℕ, filter.tendsto (λ n, s (f n)) filter.at_top (𝓝 (Real.sqrt 2))

theorem exists_subsequence_converging_to_sqrt2 :
  converges_to_sqrt2_subsequence (λ p, (p.1 : ℝ)^(1/3) - (p.2 : ℝ)^(1/3)) :=
sorry

end exists_subsequence_converging_to_sqrt2_l224_224448


namespace find_n_l224_224360

variable (x n : ℕ)
variable (y : ℕ) {h1 : y = 24}

theorem find_n
  (h1 : y = 24) 
  (h2 : x / y = 1 / 4) 
  (h3 : (x + n) / y = 1 / 2) : 
  n = 6 := 
sorry

end find_n_l224_224360


namespace find_quadratic_polynomial_l224_224487

theorem find_quadratic_polynomial :
  ∃ (p : polynomial ℝ), (∀ x : ℂ, (x = -1 - 4 * complex.I ∨ x = -1 + 4 * complex.I) → p.eval x.re = 0) ∧ p.coeff 1 = 6 ∧ p = -3 * polynomial.X^2 - 6 * polynomial.X - 51 :=
by
  sorry

end find_quadratic_polynomial_l224_224487


namespace cut_into_two_pieces_is_possible_cut_into_three_pieces_is_impossible_cut_into_four_pieces_is_possible_cut_into_five_pieces_is_impossible_l224_224835

-- Definitions based on the conditions:
-- 1. Folded napkin structure
structure Napkin where
  folded_in_two: Bool -- A napkin folded in half once along one axis 
  folded_in_four: Bool -- A napkin folded in half twice to form a smaller square

-- 2. Cutting through a folded napkin
def single_cut_through_folded_napkin (n: Nat) (napkin: Napkin) : Bool :=
  if (n = 2 ∨ n = 4) then
    true
  else
    false

-- Main theorem statements 
-- If the napkin can be cut into 2 pieces
theorem cut_into_two_pieces_is_possible (napkin: Napkin) : single_cut_through_folded_napkin 2 napkin = true := by
  sorry

-- If the napkin can be cut into 3 pieces
theorem cut_into_three_pieces_is_impossible (napkin: Napkin) : single_cut_through_folded_napkin 3 napkin = false := by
  sorry

-- If the napkin can be cut into 4 pieces
theorem cut_into_four_pieces_is_possible (napkin: Napkin) : single_cut_through_folded_napkin 4 napkin = true := by
  sorry

-- If the napkin can be cut into 5 pieces
theorem cut_into_five_pieces_is_impossible (napkin: Napkin) : single_cut_through_folded_napkin 5 napkin = false := by
  sorry

end cut_into_two_pieces_is_possible_cut_into_three_pieces_is_impossible_cut_into_four_pieces_is_possible_cut_into_five_pieces_is_impossible_l224_224835


namespace balloon_permutations_count_l224_224909

-- Definitions of the conditions
def total_letters_count : ℕ := 7
def l_count : ℕ := 2
def o_count : ℕ := 2

-- Now the mathematical problem as a Lean statement
theorem balloon_permutations_count : 
  (Nat.factorial total_letters_count) / ((Nat.factorial l_count) * (Nat.factorial o_count)) = 1260 := 
by
  sorry

end balloon_permutations_count_l224_224909


namespace kolya_is_collection_agency_l224_224217

-- Defining the conditions
def lent_books (Katya Vasya : Type) : Prop := sorry
def missed_return_date (Vasya : Type) : Prop := sorry
def agreed_to_help_retrieve (Katya Kolya Vasya : Type) : Prop := sorry
def received_reward (Kolya : Type) : Prop := sorry

-- Problem statement
theorem kolya_is_collection_agency (Katya Kolya Vasya : Type)
  (h1 : lent_books Katya Vasya) 
  (h2 : missed_return_date Vasya) 
  (h3 : agreed_to_help_retrieve Katya Kolya Vasya)
  (h4 : received_reward Kolya) : 
  ⟦Kolya's Role⟧ = "collection agency" := 
sorry

end kolya_is_collection_agency_l224_224217


namespace minimum_children_l224_224100

-- Definitions based on conditions
def boys (m : ℕ) := fin m
def girls (d : ℕ) := fin d

variable (m d : ℕ)

-- Conditions
axiom boys_friends_five_girls : ∀ b : boys m, ∃ (g_set : finset (girls d)), g_set.card = 5
axiom girls_different_friends_count : ∀ (g1 g2 : girls d), g1 ≠ g2 → count_friends g1 ≠ count_friends g2

-- Auxiliary function to count friends of a girl
noncomputable def count_friends (g : girls d) : ℕ :=
  (finset.univ.filter (λ b : boys m, ∃ S, S.card = 5 ∧ g ∈ S)).card

-- Proof statement
theorem minimum_children
  (boys_friends_five_girls : ∀ b : boys m, ∃ (g_set : finset (girls d)), g_set.card = 5)
  (girls_different_friends_count : ∀ (g1 g2 : girls d), g1 ≠ g2 → count_friends g1 ≠ count_friends g2) :
  m + d ≥ 18 :=
sorry

end minimum_children_l224_224100


namespace football_daily_practice_hours_l224_224396

-- Define the total practice hours and the days missed.
def total_hours := 30
def days_missed := 1
def days_in_week := 7

-- Calculate the number of days practiced.
def days_practiced := days_in_week - days_missed

-- Define the daily practice hours.
def daily_practice_hours := total_hours / days_practiced

-- State the proposition.
theorem football_daily_practice_hours :
  daily_practice_hours = 5 := sorry

end football_daily_practice_hours_l224_224396


namespace parts_processed_per_hour_l224_224024

theorem parts_processed_per_hour (x : ℕ) (y : ℕ) (h1 : y = x + 10) (h2 : 150 / y = 120 / x) :
  x = 40 ∧ y = 50 :=
by {
  sorry
}

end parts_processed_per_hour_l224_224024


namespace cube_root_of_unity_identity_l224_224149

theorem cube_root_of_unity_identity (ω : ℂ) (hω3: ω^3 = 1) (hω_ne_1 : ω ≠ 1) (hunit : ω^2 + ω + 1 = 0) :
  (1 - ω) * (1 - ω^2) * (1 - ω^4) * (1 - ω^8) = 9 :=
by
  sorry

end cube_root_of_unity_identity_l224_224149


namespace vanya_meets_mother_opposite_dir_every_4_minutes_l224_224408

-- Define the parameters
def lake_perimeter : ℝ := sorry  -- Length of the lake's perimeter, denoted as l
def mother_time_lap : ℝ := 12    -- Time taken by the mother to complete one lap (in minutes)
def vanya_time_overtake : ℝ := 12 -- Time taken by Vanya to overtake the mother (in minutes)

-- Define speeds
noncomputable def mother_speed : ℝ := lake_perimeter / mother_time_lap
noncomputable def vanya_speed : ℝ := 2 * lake_perimeter / vanya_time_overtake

-- Define their relative speed when moving in opposite directions
noncomputable def relative_speed : ℝ := mother_speed + vanya_speed

-- Prove that the meeting interval is 4 minutes
theorem vanya_meets_mother_opposite_dir_every_4_minutes :
  (lake_perimeter / relative_speed) = 4 := by
  sorry

end vanya_meets_mother_opposite_dir_every_4_minutes_l224_224408


namespace balloon_permutations_count_l224_224907

-- Definitions of the conditions
def total_letters_count : ℕ := 7
def l_count : ℕ := 2
def o_count : ℕ := 2

-- Now the mathematical problem as a Lean statement
theorem balloon_permutations_count : 
  (Nat.factorial total_letters_count) / ((Nat.factorial l_count) * (Nat.factorial o_count)) = 1260 := 
by
  sorry

end balloon_permutations_count_l224_224907


namespace factorize_expression_l224_224471

theorem factorize_expression (a : ℝ) : 3 * a^2 + 6 * a + 3 = 3 * (a + 1)^2 := 
by sorry

end factorize_expression_l224_224471


namespace max_dominoes_after_removal_l224_224136

theorem max_dominoes_after_removal (removed_squares : Finset (Fin 8 × Fin 8)) (h_card : removed_squares.card = 10) : 
  let max_dominoes := (64 - 10) / 2
  in max_dominoes ≤ 23 := by
  sorry

end max_dominoes_after_removal_l224_224136


namespace Rich_walk_distance_l224_224293

def total_distance (initialdist sidewalkdist doubledist rightturndist parkstrolldist tripledist halfdist roundtripdist: ℕ) 
: Prop := initialdist + sidewalkdist + doubledist + rightturndist + parkstrolldist + tripledist + halfdist = roundtripdist 

theorem Rich_walk_distance : 
  total_distance  20 200 (2 * (20 + 200)) 500 300 (3 * (20 + 200 + 2 * (20 + 200) + 500 + 300)) ((20 + 200 + 2 * (20 + 200) + 500 + 300 + 3 * (20 + 200 + 2 * (20 + 200) + 500 + 300)) / 2) (17520 / 2):
   20 + 200 + (2 * (20 + 200)) + 500 + 300 + (3 * (20 + 200 + (2 * (20 + 200)) + 500 + 300)) + ((20 + 200 + (2 * (20 + 200)) + 500 + 300 + (3 * (20 + 200 + (2 * (20 + 200)) + 500 + 300))) / 2) * 2 = 17520 :=
sorry

end Rich_walk_distance_l224_224293


namespace balloon_permutations_count_l224_224901

-- Definitions of the conditions
def total_letters_count : ℕ := 7
def l_count : ℕ := 2
def o_count : ℕ := 2

-- Now the mathematical problem as a Lean statement
theorem balloon_permutations_count : 
  (Nat.factorial total_letters_count) / ((Nat.factorial l_count) * (Nat.factorial o_count)) = 1260 := 
by
  sorry

end balloon_permutations_count_l224_224901


namespace find_general_term_and_sum_l224_224144

noncomputable def arithmetic_sequence_properties (a : ℕ → ℝ) : Prop :=
(all (λ n, a n > 0) ∧
 a 1 + a 2 = 12 ∧
 9 * (a 3) ^ 2 = a 2 * a 6)

theorem find_general_term_and_sum (a : ℕ → ℝ)
  (h_arith_seq : arithmetic_sequence_properties a) :
  (∀ n, a n = 3^n) ∧
  (∀ b : ℕ → ℝ, 
    (∀ n, b n = ∑ i in finset.range n, log 3 (a i)) → 
    ∀ n, (∑ i in finset.range n (1 / b (i + 1))) = 2 * n / (n + 1)) :=
by {
  sorry
}

end find_general_term_and_sum_l224_224144


namespace rhombus_area_l224_224004

-- Definitions based on the conditions of the problem.
def side_length : ℝ := 4
def angle_between_sides_deg : ℝ := 45
def area_of_rhombus : ℝ := 8 * Real.sqrt 2

-- The theorem statement.
theorem rhombus_area :
  ∀ (side_length : ℝ) (angle_between_sides_deg : ℝ) 
  (h_side_length : side_length = 4) 
  (h_angle_45 : angle_between_sides_deg = 45),
  let height := side_length * Real.sqrt 2 in
  let base := side_length in
  (base * height) = area_of_rhombus :=
by
  sorry

end rhombus_area_l224_224004


namespace sum_g_terms_l224_224501

def g (x : ℝ) : ℝ := (1 / 3) * x^3 - (1 / 2) * x^2 + 3 * x - (5 / 12)

theorem sum_g_terms : 
  (∑ k in finset.range(2017), g (((k + 1) : ℝ) / 2018)) = 2017 := 
by
  sorry

end sum_g_terms_l224_224501


namespace fraction_powers_sum_l224_224363

theorem fraction_powers_sum : 
  ( (5:ℚ) / (3:ℚ) )^6 + ( (2:ℚ) / (3:ℚ) )^6 = (15689:ℚ) / (729:ℚ) :=
by
  sorry

end fraction_powers_sum_l224_224363


namespace brenda_total_distance_l224_224858

def distance (p1 p2 : (ℤ × ℤ)) : ℝ :=
  real.sqrt (real.to_real ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2))

def P0 : ℤ × ℤ := (-4, 6)
def P1 : ℤ × ℤ := (0, 0)
def P2 : ℤ × ℤ := (2, -3)
def P3 : ℤ × ℤ := (6, -1)

noncomputable def total_distance : ℝ :=
  distance P0 P1 + distance P1 P2 + distance P2 P3

theorem brenda_total_distance :
  total_distance = real.sqrt 52 + real.sqrt 13 + real.sqrt 20 :=
by sorry

end brenda_total_distance_l224_224858


namespace recurring_to_fraction_l224_224753

theorem recurring_to_fraction : ∀ (x : ℚ), x = 0.5656 ∧ 100 * x = 56.5656 → x = 56 / 99 :=
by
  intros x hx
  sorry

end recurring_to_fraction_l224_224753


namespace problem_solution_l224_224626

variables (l m n : Type)
variables (α β γ : Type)
variables [line l] [line m] [line n]
variables [plane α] [plane β] [plane γ]

def false_propositions (l m n : Type) (α β γ : Type) [line l] [line m] [line n] [plane α] [plane β] [plane γ] : Prop :=
  (l ∥ β ∧ α ∥ β → l ∥ α) ∨ 
  (l ∥ n ∧ m ∥ n → l ∥ m) ∨ 
  (α ⟂ β ∧ l ∥ α → l ⟂ β) ∨ 
  (l ⟂ α ∧ m ⟂ β ∧ α ⟂ β → l ⟂ m)

theorem problem_solution : false_propositions l m n α β γ :=
begin
  sorry
end

end problem_solution_l224_224626


namespace point_P_moves_correctly_l224_224048

noncomputable def center_of_dilation : Real × Real := (-5, -9)

noncomputable def dilation_factor : Real := 1.5

def initial_distance (P : Real × Real) : Real :=
  let C := center_of_dilation in
  Real.sqrt ((P.1 + 5)^2 + (P.2 + 9)^2)

def moved_distance (P : Real × Real) : Real :=
  dilation_factor * initial_distance P

def distance_moved_by_P (P : Real × Real) : Real :=
  moved_distance P - initial_distance P

theorem point_P_moves_correctly : distance_moved_by_P (3, 1) = Real.sqrt 41 :=
by
  sorry

end point_P_moves_correctly_l224_224048


namespace tangent_expression_value_l224_224535

theorem tangent_expression_value :
  (∃ (tan_10 tan_20 : ℝ), (tan 30 = (sqrt 3) / 3) ∧ (tan 30 = (tan_10 + tan_20) / (1 - tan_10 * tan_20)) → 
  ((sqrt 3) / 3 * tan_10 * tan_20 + tan_10 + tan_20 = (sqrt 3) / 3)) :=
sorry

end tangent_expression_value_l224_224535


namespace speed_of_current_l224_224399

-- Definitions coming from the given problem conditions
def Vs : ℝ := 30 -- Speed of the boat in still water (kmph)
def distance : ℝ := 240 / 1000 -- Distance covered downstream in km
def time : ℝ := 24 / 3600 -- Time taken to cover the distance in hours

-- Definition of the speed downstream
def Vd : ℝ := distance / time

-- Statement to prove: The speed of the current is 6 kmph
theorem speed_of_current (Vs_eq : Vs = 30) (distance_eq : distance = 240 / 1000) (time_eq : time = 24 / 3600) :
  Vd - Vs = 6 := by
  sorry

end speed_of_current_l224_224399


namespace part1_f2_f_half_part1_f3_f_third_part2_conjecture_part3_summation_l224_224540

-- Define the function f
def f (x : ℝ) : ℝ := x^2 / (1 + x^2)

-- Part 1: Specific value calculations
theorem part1_f2_f_half :
  f(2) + f(1 / 2) = 1 :=
sorry

theorem part1_f3_f_third :
  f(3) + f(1 / 3) = 1 :=
sorry

-- Part 2: Conjecture
theorem part2_conjecture (x : ℝ) (hx : x ≠ 0) :
  f(x) + f(1 / x) = 1 :=
sorry

-- Part 3: Summation
theorem part3_summation :
  ∑ k in (Finset.range 2015).map (λ n, n + 2) ∪ (Finset.range 2015).map (λ n, (1 / (n + 2) : ℝ)), f k = 2015 :=
sorry

end part1_f2_f_half_part1_f3_f_third_part2_conjecture_part3_summation_l224_224540


namespace elvis_writing_time_l224_224464

-- Define the conditions
def total_studio_time := 9 * 60
def recording_time_per_song := 18
def extra_time_per_collaborated_song := 10
def num_collaborated_songs := 4
def editing_time := 45
def mixing_and_mastering_time := 60
def num_songs := 15

-- The time taken to write each song
def writing_time_per_song : ℝ :=
  (total_studio_time - ((num_songs * recording_time_per_song) + (num_collaborated_songs * extra_time_per_collaborated_song) + editing_time + mixing_and_mastering_time)) / num_songs

-- Statement to be proved
theorem elvis_writing_time : writing_time_per_song = 8.33 :=
  sorry

end elvis_writing_time_l224_224464


namespace balloon_arrangement_count_l224_224972

-- Definitions of letter frequencies for the word BALLOON
def letter_frequencies : List (Char × Nat) := [('B', 1), ('A', 1), ('L', 2), ('O', 2), ('N', 1)]

-- Total number of letters
def total_letters := 7

-- The formula for the number of unique arrangements of the letters
noncomputable def arrangements :=
  (Nat.factorial total_letters) / 
  (Nat.factorial 1 * Nat.factorial 1 * Nat.factorial 2 * Nat.factorial 2 * Nat.factorial 1)

-- The theorem to prove the number of ways to arrange the letters in "BALLOON"
theorem balloon_arrangement_count : arrangements = 1260 :=
  sorry

end balloon_arrangement_count_l224_224972


namespace find_f_prime_neg2_l224_224541

-- Define the function f
def f (x : ℝ) (f'neg2 : ℝ) := f'neg2 * Real.exp x - x^2

-- State the main theorem
theorem find_f_prime_neg2 (f'neg2 : ℝ) :
  (∃ (f : ℝ → ℝ), (∀ x, deriv (f x) = f'(-2) * Real.exp x - 2 * x) 
  ∧ f'(-2) = 4 * (Real.exp 2) / (Real.exp 2 - 1)) :=
sorry

end find_f_prime_neg2_l224_224541


namespace find_CD_l224_224155

theorem find_CD (a b c d : ℝ) (angle_ACB : ℝ) (AD BC AC CD : ℝ) :
  (1 / 6 = volume_of_DABC) ∧ (angle_ACB = 45) ∧ (AD + BC + AC / (sqrt 2) = 3) →
  CD = sqrt 3 :=
by
  sorry

end find_CD_l224_224155


namespace balloon_permutations_l224_224965

theorem balloon_permutations : (∀ (arrangements : ℕ),
  arrangements = nat.factorial 7 / (nat.factorial 2 * nat.factorial 2) → arrangements = 1260) :=
begin
  intros arrangements h,
  rw [nat.factorial_succ, nat.factorial],
  sorry,
end

end balloon_permutations_l224_224965


namespace parallelogram_vertices_l224_224028

theorem parallelogram_vertices (x y : ℝ) :
  ((∃ P : ℝ × ℝ,
    P = (x, y) ∧ 
    ((0, 0), (1, 1), (1, 0), (x, y)).sorted_and_nodup_finset ↔ 
    P ∈ {(2, 1), (0, 1), (0, -1)})) := 
sorry

end parallelogram_vertices_l224_224028


namespace orchestra_seat_cost_l224_224839

theorem orchestra_seat_cost (y x : ℕ) (y_eq : 2 * y + 190 = 370) (total_cost_eq : 90 * x + 280 * 8 = 3320) : x = 12 :=
by
  have h1 : 2 * y = 180,
  { rw [←y_eq, sub_add_cancel] },
  have y_val : y = 90,
  { linarith },
  sorry

end orchestra_seat_cost_l224_224839


namespace measure_angle_BAC_l224_224204

theorem measure_angle_BAC (ABC : Triangle) (AB BC : Real) (t : Real) (h1 : IsIsosceles ABC AB BC) (h2 : Angle_ABC ABC = t) :
  Angle_BAC ABC = (180 - t) / 2 :=
by
  sorry

end measure_angle_BAC_l224_224204


namespace eight_digit_palindrome_count_l224_224011

def is_palindrome (n : ℕ) : Prop :=
  let digits := n.digits 10 in
  digits = digits.reverse

def is_eight_digit (n : ℕ) : Prop :=
  n >= 10^7 ∧ n < 10^8

def valid_digits (n : ℕ) : Prop :=
  ∀ d ∈ n.digits 10, d = 1 ∨ d = 2 ∨ d = 3

theorem eight_digit_palindrome_count : 
  ∃ count : ℕ, count = 81 ∧ (∀ n : ℕ, is_palindrome n ∧ is_eight_digit n ∧ valid_digits n ↔ n ∈ { n | is_palindrome n ∧ is_eight_digit n ∧ valid_digits n } ∧ |{ n | is_palindrome n ∧ is_eight_digit n ∧ valid_digits n }| = 81) :=
by
  sorry

end eight_digit_palindrome_count_l224_224011


namespace no_arithmetic_progression_trigonometric_l224_224462

theorem no_arithmetic_progression_trigonometric (α : ℝ) (hα : 0 < α ∧ α < Real.pi / 2) :
  ¬∃ (f g h i : ℝ), 
    (f = Real.sin α ∨ f = Real.cos α ∨ f = Real.tan α ∨ f = Real.cot α) ∧
    (g = Real.sin α ∨ g = Real.cos α ∨ g = Real.tan α ∨ g = Real.cot α) ∧
    (h = Real.sin α ∨ h = Real.cos α ∨ h = Real.tan α ∨ h = Real.cot α) ∧
    (i = Real.sin α ∨ i = Real.cos α ∨ i = Real.tan α ∨ i = Real.cot α) ∧
    f ≠ g ∧ f ≠ h ∧ f ≠ i ∧ g ≠ h ∧ g ≠ i ∧ h ≠ i ∧
    g - f = h - g ∧ h - g = i - h :=
sorry

end no_arithmetic_progression_trigonometric_l224_224462


namespace balloon_permutations_l224_224951

theorem balloon_permutations : 
  let total_letters := 7
  let repetitions_L := 2
  let repetitions_O := 2
  (Nat.factorial total_letters) / ((Nat.factorial repetitions_L) * (Nat.factorial repetitions_O)) = 1260 := 
by
  sorry

end balloon_permutations_l224_224951


namespace range_of_f_l224_224125

noncomputable def f (x : ℝ) : ℝ :=
  2 * cos ( (π / 4) * sin ( (x + 1)^2 + 1 + cos x ) )

theorem range_of_f :
  set.range f = set.Icc (Real.sqrt 2) 2 :=
sorry

end range_of_f_l224_224125


namespace find_common_difference_max_sum_first_n_terms_max_n_Sn_positive_l224_224706

-- Define the arithmetic sequence
def a (n : ℕ) (d : ℤ) := 23 + n * d

-- Define the sum of the first n terms of the sequence
def S (n : ℕ) (d : ℤ) := n * 23 + (n * (n - 1) / 2) * d

-- Prove the common difference is -4
theorem find_common_difference (d : ℤ) :
  a 5 d > 0 ∧ a 6 d < 0 → d = -4 := sorry

-- Prove the maximum value of the sum S_n of the first n terms
theorem max_sum_first_n_terms (S_n : ℕ) :
  S 6 -4 = 78 := sorry

-- Prove the maximum value of n such that S_n > 0
theorem max_n_Sn_positive (n : ℕ) :
  S n -4 > 0 → n ≤ 12 := sorry

end find_common_difference_max_sum_first_n_terms_max_n_Sn_positive_l224_224706


namespace repeating_decimal_as_fraction_l224_224748

theorem repeating_decimal_as_fraction :
  let x := 56 / 99
  in x = 0.56 + 0.0056 + 0.000056 + (0.00000056) : ℚ :=
by
  sorry

end repeating_decimal_as_fraction_l224_224748


namespace simplify_polynomial_l224_224673

theorem simplify_polynomial :
  (2 * x * (4 * x ^ 3 - 3 * x + 1) - 4 * (2 * x ^ 3 - x ^ 2 + 3 * x - 5)) =
  8 * x ^ 4 - 8 * x ^ 3 - 2 * x ^ 2 - 10 * x + 20 :=
by
  sorry

end simplify_polynomial_l224_224673


namespace find_other_number_l224_224690

theorem find_other_number (LCM : ℕ) (HCF : ℕ) (n1 : ℕ) (n2 : ℕ) 
  (h_lcm : LCM = 2310) (h_hcf : HCF = 26) (h_n1 : n1 = 210) :
  n2 = 286 :=
by
  sorry

end find_other_number_l224_224690


namespace max_value_g_l224_224131

-- Definitions of the functions
def f1 (x : ℝ) := 3 * x + 3
def f2 (x : ℝ) := (2 / 3) * x + 2
def f3 (x : ℝ) := - (1 / 2) * x + 8

-- Definition of g(x)
def g (x : ℝ) := min (f1 x) (min (f2 x) (f3 x))

theorem max_value_g : ∃ (x : ℝ), g x = 78 / 21 :=
sorry

end max_value_g_l224_224131


namespace net_rate_of_pay_l224_224823

theorem net_rate_of_pay :
  ∀ (duration_travel : ℝ) (speed : ℝ) (fuel_efficiency : ℝ) (earnings_rate : ℝ) (gas_cost : ℝ),
  duration_travel = 3 → speed = 50 → fuel_efficiency = 30 → earnings_rate = 0.75 → gas_cost = 2.50 →
  (earnings_rate * speed * duration_travel - (speed * duration_travel / fuel_efficiency) * gas_cost) / duration_travel = 33.33 :=
by
  intros duration_travel speed fuel_efficiency earnings_rate gas_cost
  intros h1 h2 h3 h4 h5
  sorry

end net_rate_of_pay_l224_224823


namespace largest_number_is_l224_224801

def hcf : ℕ := 31
def lcm_factors : List ℕ := [13, 14, 17]

theorem largest_number_is :
  (lcm_factors.product * hcf = 95914) := by
  sorry

end largest_number_is_l224_224801


namespace balloon_permutation_count_l224_224924

-- Define the given conditions:
def balloon_letters : List Char := ['B', 'A', 'L', 'L', 'O', 'O', 'N']
def total_letters : Nat := 7
def count_L : Nat := 2
def count_O : Nat := 2

-- Statement to prove:
theorem balloon_permutation_count :
  (nat.factorial total_letters) / ((nat.factorial count_L) * (nat.factorial count_O)) = 1260 := by
  sorry

end balloon_permutation_count_l224_224924


namespace proof_A_union_B_eq_R_l224_224641

def A : Set ℝ := { x | x^2 - 5 * x - 6 > 0 }
def B (a : ℝ) : Set ℝ := { x | abs (x - 5) < a }

theorem proof_A_union_B_eq_R (a : ℝ) (h : a > 6) : 
  A ∪ B a = Set.univ :=
by {
  sorry
}

end proof_A_union_B_eq_R_l224_224641


namespace magnitude_of_T_l224_224240

theorem magnitude_of_T :
  let i : ℂ := complex.I in
  let T : ℂ := (1 + i) ^ 19 - (1 - i) ^ 19 in
  complex.abs T = 512 * real.sqrt 2 :=
by
  sorry

end magnitude_of_T_l224_224240


namespace balloon_permutation_count_l224_224931

-- Define the given conditions:
def balloon_letters : List Char := ['B', 'A', 'L', 'L', 'O', 'O', 'N']
def total_letters : Nat := 7
def count_L : Nat := 2
def count_O : Nat := 2

-- Statement to prove:
theorem balloon_permutation_count :
  (nat.factorial total_letters) / ((nat.factorial count_L) * (nat.factorial count_O)) = 1260 := by
  sorry

end balloon_permutation_count_l224_224931


namespace add_multiply_round_l224_224422

theorem add_multiply_round :
  let a := 73.5891
  let b := 24.376
  let c := (a + b) * 2
  (Float.round (c * 100) / 100) = 195.93 :=
by
  sorry

end add_multiply_round_l224_224422


namespace balloon_permutation_count_l224_224933

-- Define the given conditions:
def balloon_letters : List Char := ['B', 'A', 'L', 'L', 'O', 'O', 'N']
def total_letters : Nat := 7
def count_L : Nat := 2
def count_O : Nat := 2

-- Statement to prove:
theorem balloon_permutation_count :
  (nat.factorial total_letters) / ((nat.factorial count_L) * (nat.factorial count_O)) = 1260 := by
  sorry

end balloon_permutation_count_l224_224933


namespace process_box_usage_l224_224586

theorem process_box_usage (output : Prop) (assignment : Prop) (decision : Prop) (end_algo : Prop)
  (h : ∀ (symbol : Prop), symbol = assignment) : 
  ∀ (symbol : Prop), symbol = assignment :=
by
  -- Given conditions definition
  have h_output : ¬symbol = output, from sorry,
  have h_decision : ¬symbol = decision, from sorry,
  have h_end_algo : ¬symbol = end_algo, from sorry,
  -- Prove symbol equals to assignment
  exact h symbol

end process_box_usage_l224_224586


namespace group_left_to_clean_is_third_group_l224_224392

-- Definition of group sizes
def group1 := 7
def group2 := 10
def group3 := 16
def group4 := 18

-- Definitions and conditions
def total_students := group1 + group2 + group3 + group4
def lecture_factor := 4
def english_students := 7  -- From solution: must be 7 students attending the English lecture
def math_students := lecture_factor * english_students

-- Hypothesis of the students allocating to lectures
def students_attending_lectures := english_students + math_students
def students_left_to_clean := total_students - students_attending_lectures

-- The statement to be proved in Lean
theorem group_left_to_clean_is_third_group
  (h : students_left_to_clean = group3) :
  students_left_to_clean = 16 :=
sorry

end group_left_to_clean_is_third_group_l224_224392


namespace sum_of_all_possible_values_f1_l224_224236

noncomputable def f (x : ℝ) : ℝ := sorry

lemma f_polynomial_nonconstant (x : ℝ) (hx : x ≠ 0) :
  f(x - 1) + f(x) + f(x + 1) = f(x)^2 / (2013 * x) := sorry

theorem sum_of_all_possible_values_f1 :
  f(1) = 6039 := sorry

end sum_of_all_possible_values_f1_l224_224236


namespace cuboid_breadth_l224_224476

theorem cuboid_breadth (l h A : ℝ) (w : ℝ) :
  l = 8 ∧ h = 12 ∧ A = 960 → 2 * (l * w + l * h + w * h) = A → w = 19.2 :=
by
  intros h1 h2
  sorry

end cuboid_breadth_l224_224476


namespace analytic_expression_of_f_comparison_fba_fca_l224_224050

noncomputable def f (a b c : ℝ) (x : ℝ) : ℝ := a * Real.sin x + b * Real.cos x + c

theorem analytic_expression_of_f :
  ∀ (a b c : ℝ),
  (f a b c (0) = 0) →
  (∀ x : ℝ, f a b c x ≤ f a b c (Real.pi/3)) →
  f a b c (Real.pi/3) = 1 →
  f' a b c (Real.pi/3) = 0 →
  f a b c x = (\sqrt{3}) * Real.sin x + Real.cos x - 1 :=
by
  sorry

theorem comparison_fba_fca :
  ∀ (a b c : ℝ),
  f a b c x = (\sqrt{3}) * Real.sin x + Real.cos x - 1 →
  f (b / a) > f (c / a) :=
by 
  sorry

end analytic_expression_of_f_comparison_fba_fca_l224_224050


namespace repeating_decimal_fraction_l224_224784

theorem repeating_decimal_fraction : ∀ x : ℚ, (x = 0.5656565656565656) → 100 * x = 56.5656565656565656 → 100 * x - x = 56.5656565656565656 - 0.5656565656565656
  → 99 * x = 56 → x = 56 / 99 :=
begin
  intros x h1 h2 h3 h4,
  sorry,
end

end repeating_decimal_fraction_l224_224784


namespace functions_are_equal_l224_224431

def f (x : ℝ) : ℝ := 2 * |x|
def g (x : ℝ) : ℝ := Real.sqrt (4 * x^2)

theorem functions_are_equal (x : ℝ) : f x = g x :=
by
  sorry

end functions_are_equal_l224_224431


namespace three_points_in_circle_of_radius_one_seventh_l224_224039

-- Define the problem
theorem three_points_in_circle_of_radius_one_seventh (P : Fin 51 → ℝ × ℝ) :
  (∀ i, 0 ≤ (P i).1 ∧ (P i).1 ≤ 1 ∧ 0 ≤ (P i).2 ∧ (P i).2 ≤ 1) →
  ∃ (i j k : Fin 51), i ≠ j ∧ j ≠ k ∧ i ≠ k ∧
    dist (P i) (P j) ≤ 2/7 ∧ dist (P j) (P k) ≤ 2/7 ∧ dist (P k) (P i) ≤ 2/7 :=
by
  sorry

end three_points_in_circle_of_radius_one_seventh_l224_224039


namespace energy_inverse_wavelength_l224_224742

theorem energy_inverse_wavelength (h : Real) (c : Real) (λ : Real) (E : Real) 
  (eq1 : ∀ f, E = h * f) (eq2 : ∀ λ, f = c / λ) :
  E = h * c / λ := by
  sorry

end energy_inverse_wavelength_l224_224742


namespace lattice_points_in_triangle_271_l224_224283

def Pick_theorem (N L S : ℝ) : Prop :=
  S = N + (1/2) * L - 1

noncomputable def lattice_triangle_inside_points (A B O : ℕ × ℕ) : ℕ :=
  271

theorem lattice_points_in_triangle_271 :
  ∃ N L : ℝ, 
  let S := 300 in
  let L := 60 in
  A = (0, 30) ∧ B = (20, 10) ∧
  O = (0, 0) ∧ Pick_theorem N L S :=
sorry

end lattice_points_in_triangle_271_l224_224283


namespace triangle_height_relationship_l224_224354

theorem triangle_height_relationship
  (b : ℝ) (h1 h2 h3 : ℝ)
  (area1 area2 area3 : ℝ)
  (h_equal_angle : area1 / area2 = 16 / 25)
  (h_diff_angle : area1 / area3 = 4 / 9) :
  4 * h2 = 5 * h1 ∧ 6 * h2 = 5 * h3 := by
    sorry

end triangle_height_relationship_l224_224354


namespace valid_coloring_of_n_polygon_l224_224847

def coloring_ways (n : ℕ) : ℤ :=
  2^n + 2 * (-1)^n

theorem valid_coloring_of_n_polygon :
  ∀ n : ℕ, n ≥ 3 →
  let S := (λ (n : ℕ), 2^n + 2 * (-1)^n)
  ∃ S, S = coloring_ways n :=
by
  intro n hn
  let S := (λ (n : ℕ), 2^n + 2 * (-1)^n)
  existsi S
  sorry

end valid_coloring_of_n_polygon_l224_224847


namespace triangle_lattice_points_l224_224286

theorem triangle_lattice_points :
  let S := 300
  let L := 60
  let N := 271 in
  S = N + (1/2 : ℚ) * L - 1 :=
by
  -- Calculate the number of lattice points on OA; 31
  -- Calculate the number of lattice points on OB; 10
  -- Calculate the number of lattice points on AB; 19
  -- Sum these to get L: 31 + 10 + 19 = 60
  -- Calculate the area of triangle ABO: 300
  -- Use Pick's theorem to solve for N: 271
  sorry

end triangle_lattice_points_l224_224286


namespace ten_thousandths_place_of_fraction_l224_224015

-- Noncomputable theory to use real numbers and their properties
noncomputable theory

open Real BigOperators 

def digit_in_ten_thousandths_place (n d : ℕ) (den_lt_0: d > 0) :=
  ∃ k : ℕ, (10^4 * k = (n / d - (real.floor (n / d))) * 10^5) ∧ k % 10 = 8

theorem ten_thousandths_place_of_fraction :
  digit_in_ten_thousandths_place 7 32 (by decide) :=
begin
  sorry,
end

end ten_thousandths_place_of_fraction_l224_224015


namespace log_order_l224_224460

theorem log_order : 
  let x := 0.45
  let y := 50.4
  let z := Real.log x in
  z < x ∧ x < y :=
by
  sorry

end log_order_l224_224460


namespace balloon_arrangements_l224_224896

theorem balloon_arrangements :
  let n := 7
  let k1 := 2
  let k2 := 2
  (Nat.factorial n) / ((Nat.factorial k1) * (Nat.factorial k2)) = 1260 :=
by
  -- Use the conditions directly from the problem
  let n := 7
  let k1 := 2
  let k2 := 2
  -- Mathematically, the number of arrangements of the multiset
  have h : (Nat.factorial n) / ((Nat.factorial k1) * (Nat.factorial k2)) = 1260 := sorry
  exact h

end balloon_arrangements_l224_224896


namespace balloon_arrangements_l224_224893

theorem balloon_arrangements :
  let n := 7
  let k1 := 2
  let k2 := 2
  (Nat.factorial n) / ((Nat.factorial k1) * (Nat.factorial k2)) = 1260 :=
by
  -- Use the conditions directly from the problem
  let n := 7
  let k1 := 2
  let k2 := 2
  -- Mathematically, the number of arrangements of the multiset
  have h : (Nat.factorial n) / ((Nat.factorial k1) * (Nat.factorial k2)) = 1260 := sorry
  exact h

end balloon_arrangements_l224_224893


namespace find_min_value_exists_x0_k_range_l224_224256

-- Definition of functions
def f (x : ℝ) := Real.exp x
def g (x : ℝ) (k : ℝ) := k * x + 1

-- Proof problem (I)
theorem find_min_value : ∃ x, x = 0 ∧ (f x - (x + 1) = 0) := sorry

-- Proof problem (II)
theorem exists_x0 (k : ℝ) (h : 1 < k) : ∃ x0 : ℝ, 0 < x0 ∧ ∀ x : ℝ, 0 < x ∧ x < x0 → f x < g x k := sorry

-- Proof problem (III)
theorem k_range (m : ℝ) (k : ℝ) 
  (h : ∀ x : ℝ, 0 < x ∧ x < m → abs (f x - g x k) > x) : k ≤ 0 ∨ k > 2 := sorry

end find_min_value_exists_x0_k_range_l224_224256


namespace percent_of_z_equals_120_percent_of_y_l224_224574

variable {x y z : ℝ}
variable {p : ℝ}

theorem percent_of_z_equals_120_percent_of_y
  (h1 : (p / 100) * z = 1.2 * y)
  (h2 : y = 0.75 * x)
  (h3 : z = 2 * x) :
  p = 45 :=
by sorry

end percent_of_z_equals_120_percent_of_y_l224_224574


namespace brainiacs_like_both_l224_224838

theorem brainiacs_like_both
  (R M B : ℕ)
  (h1 : R = 2 * M)
  (h2 : R + M - B = 96)
  (h3 : M - B = 20) : B = 18 := by
  sorry

end brainiacs_like_both_l224_224838


namespace smallest_N_to_form_rectangle_l224_224419

theorem smallest_N_to_form_rectangle (N : ℕ) (h₁ : N ≥ 102) 
  (lengths : fin N → ℕ) (h₂ : (∑ i, lengths i) = 200)
  (h₃ : ∀ i, 1 ≤ lengths i ∧ lengths i ≤ 200) :
  ∃ pairs (P : fin N → fin N → Prop), 
    (∀ i j, P i j → lengths i + lengths j ≤ 200) ∧
    (∑ i, ∑ j, if P i j then 1 else 0) = 2 * (sum (λ i, lengths i)) :=
sorry

end smallest_N_to_form_rectangle_l224_224419


namespace find_a_l224_224161

noncomputable def f (a x : ℝ) : ℝ := a * Real.exp x + x^2 - 8 * x

theorem find_a (a : ℝ) (h : (∀ x : ℝ, deriv (f a) 0 = -5)) : a = 3 :=
by
  have hd : deriv (f a) 0 = a * Real.exp 0 + 2 * 0 - 8 := sorry -- Simplification step
  rw [Real.exp_zero, mul_one, add_zero] at hd
  rw h at hd
  sorry -- Finish the proof: show that a - 8 = -5 implies a = 3

end find_a_l224_224161


namespace balloon_arrangements_l224_224892

theorem balloon_arrangements :
  let n := 7
  let k1 := 2
  let k2 := 2
  (Nat.factorial n) / ((Nat.factorial k1) * (Nat.factorial k2)) = 1260 :=
by
  -- Use the conditions directly from the problem
  let n := 7
  let k1 := 2
  let k2 := 2
  -- Mathematically, the number of arrangements of the multiset
  have h : (Nat.factorial n) / ((Nat.factorial k1) * (Nat.factorial k2)) = 1260 := sorry
  exact h

end balloon_arrangements_l224_224892


namespace g_of_zero_l224_224507

theorem g_of_zero (f g : ℤ → ℤ) (h₁ : ∀ x, f x = 2 * x + 3) (h₂ : ∀ x, g (x + 2) = f x) : 
  g 0 = -1 :=
by
  sorry

end g_of_zero_l224_224507


namespace peter_horses_food_requirement_l224_224651

theorem peter_horses_food_requirement :
  let daily_oats_per_horse := 4 * 2 in
  let daily_grain_per_horse := 3 in
  let daily_food_per_horse := daily_oats_per_horse + daily_grain_per_horse in
  let number_of_horses := 4 in
  let daily_food_all_horses := daily_food_per_horse * number_of_horses in
  let days := 3 in
  daily_food_all_horses * days = 132 :=
by
  sorry

end peter_horses_food_requirement_l224_224651


namespace find_magnitude_l224_224242

namespace ProofProblem

noncomputable def i : ℂ := complex.I
noncomputable def T : ℂ := (1 + i)^19 - (1 - i)^19

theorem find_magnitude : complex.abs T = 1024 := 
sorry

end ProofProblem

end find_magnitude_l224_224242


namespace distance_to_incenter_l224_224223

variable (D E F J : Type)  -- Define points D, E, F, and incenter J
variable [EuclideanGeometry D E F] -- Assume geometry context for Euclidean space
variable (DE DF EF DJ : ℝ)  -- Define the distances

-- Given conditions: DE = 6, DF = 6, and ∠D = 90°
axiom DE_eq : DE = 6
axiom DF_eq : DF = 6
axiom angle_D : angle D = 90 -- Hypothetical rep. of right angle at D

-- Given values for the purposes of the proof translated in Lean
axiom EF_eq : EF = 6 * Real.sqrt 2
axiom DJ_eq : DJ = 6 * Real.sqrt 2 - 6

theorem distance_to_incenter :
  DJ = 6 * Real.sqrt 2 - 6 := sorry

end distance_to_incenter_l224_224223


namespace dandelions_surviving_to_flower_l224_224101

/-- 
Each dandelion produces 300 seeds. 
1/3rd of the seeds land in water and die. 
1/6 of the starting number are eaten by insects. 
Half the remainder sprout and are immediately eaten.
-/
def starting_seeds : ℕ := 300

def seeds_lost_to_water : ℕ := starting_seeds / 3
def seeds_after_water : ℕ := starting_seeds - seeds_lost_to_water

def seeds_eaten_by_insects : ℕ := starting_seeds / 6
def seeds_after_insects : ℕ := seeds_after_water - seeds_eaten_by_insects

def seeds_eaten_after_sprouting : ℕ := seeds_after_insects / 2
def seeds_surviving : ℕ := seeds_after_insects - seeds_eaten_after_sprouting

theorem dandelions_surviving_to_flower 
  (starting_seeds = 300) 
  (seeds_lost_to_water = starting_seeds / 3) 
  (seeds_after_water = starting_seeds - seeds_lost_to_water) 
  (seeds_eaten_by_insects = starting_seeds / 6) 
  (seeds_after_insects = seeds_after_water - seeds_eaten_by_insects) 
  (seeds_eaten_after_sprouting = seeds_after_insects / 2) 
  (seeds_surviving = seeds_after_insects - seeds_eaten_after_sprouting) : 
  seeds_surviving = 75 := 
sorry

end dandelions_surviving_to_flower_l224_224101


namespace correct_answer_l224_224083

-- Definitions for the relations
variable {a b : Type} -- lines
variable {α β : Type} -- planes

-- Predicate definitions
variables (Parallel : ∀ {x y : Type}, Prop)
variables (Perpendicular : ∀ {x y : Type}, Prop)

-- Propositions
def prop1 (a : Type) (α : Type) (b : Type) (β : Type) := 
  Parallel a α ∧ Parallel b β ∧ Parallel α β → Parallel a b

def prop2 (a : Type) (α : Type) (b : Type) (β : Type) := 
  Perpendicular a α ∧ Perpendicular b β ∧ Perpendicular α β → Perpendicular α b

def prop3 (a : Type) (α : Type) (b : Type) (β : Type) := 
  Perpendicular a α ∧ Parallel b β ∧ Parallel α β → Perpendicular a b

def prop4 (a : Type) (α : Type) (b : Type) (β : Type) := 
  Parallel a α ∧ Perpendicular b β ∧ Perpendicular α β → Parallel a b

-- Theorem to be proved
theorem correct_answer :
  ¬prop1 a α b β ∧ prop2 a α b β ∧ prop3 a α b β ∧ ¬prop4 a α b β := 
by
  sorry

end correct_answer_l224_224083


namespace m_range_satisfies_inequality_l224_224186

open Real

noncomputable def f (x : ℝ) : ℝ := -2 * x + sin x

theorem m_range_satisfies_inequality :
  ∀ (m : ℝ), f (2 * m ^ 2 - m + π - 1) ≥ -2 * π ↔ -1 / 2 ≤ m ∧ m ≤ 1 := 
by
  sorry

end m_range_satisfies_inequality_l224_224186


namespace factor_expression_l224_224107

theorem factor_expression (b : ℤ) : 53 * b^2 + 159 * b = 53 * b * (b + 3) :=
by
  sorry

end factor_expression_l224_224107


namespace petya_green_balls_l224_224279

theorem petya_green_balls (total_balls : ℕ) (red_balls blue_balls green_balls : ℕ)
  (h1 : total_balls = 50)
  (h2 : ∀ s, s.card ≥ 34 → ∃ r, r ∈ s ∧ r = red_balls)
  (h3 : ∀ s, s.card ≥ 35 → ∃ b, b ∈ s ∧ b = blue_balls)
  (h4 : ∀ s, s.card ≥ 36 → ∃ g, g ∈ s ∧ g = green_balls) :
  green_balls = 15 ∨ green_balls = 16 ∨ green_balls = 17 :=
sorry

end petya_green_balls_l224_224279


namespace largest_prime_factor_cyclic_sum_l224_224085

def cyclic_increment_sequence :=
  ∃ f : ℕ → ℕ, ∀ n, f (n + 1) % 1000 = (10 * (f n / 100) % 10 + 1) * 100 + 
                   (10 * ((f n % 100) / 10) % 10 + 1) * 10 + 
                   (10 * ((f n % 10) + 1) % 10)

theorem largest_prime_factor_cyclic_sum (S_seq : ℕ → ℕ) (h : cyclic_increment_sequence S_seq):
  ∃ p, nat.prime p ∧ p = 37 ∧ ∀ n, 37 ∣ S_seq n :=
sorry

end largest_prime_factor_cyclic_sum_l224_224085


namespace repeating_decimal_as_fraction_l224_224747

theorem repeating_decimal_as_fraction :
  let x := 56 / 99
  in x = 0.56 + 0.0056 + 0.000056 + (0.00000056) : ℚ :=
by
  sorry

end repeating_decimal_as_fraction_l224_224747


namespace find_number_l224_224026

theorem find_number (x : ℝ) (h : 3 * (2 * x + 5) = 105) : x = 15 :=
by
  sorry

end find_number_l224_224026


namespace noah_total_earnings_l224_224206

theorem noah_total_earnings : 
  ∃ x : ℝ, 
    let x := 54 / 7 in
    let total_hours := 18 + 25 in
    let total_earnings := total_hours * x in
    total_earnings = 331.71 :=
by {
  sorry
}

end noah_total_earnings_l224_224206


namespace hyperbola_equilateral_triangle_proof_l224_224639

theorem hyperbola_equilateral_triangle_proof :
  (∀ P Q R : ℝ × ℝ, P.1 * P.2 = 1 → Q.1 * Q.2 = 1 → R.1 * R.2 = 1 → 
    (∀ P_x P_y Q_x Q_y R_x R_y : ℝ, P = (P_x, P_y) ∧ Q = (Q_x, Q_y) ∧ R = (R_x, R_y) →
      ¬ ((P_x = Q_x ∨ Q_x = R_x ∨ P_x = R_x) ∧ (P_y = Q_y ∨ Q_y = R_y ∨ P_y = R_y)) →
      ¬ ((P_x = P_x ∧ P_y = P_y) ∧ (Q_x = Q_x ∧ Q_y = Q_y) ∧ (R_x = R_x ∧ R_y = R_y)))) →
  (∀ Q R : ℝ × ℝ, 
    P : ℝ × ℝ := (-1, -1), 
    P.1 * P.2 = 1 → 
    Q.1 * Q.2 = 1 → 
    R.1 * R.2 = 1 → 
    (∃ Q_x R_x : ℝ, 
      Q = (Q_x, Q_x⁻¹) ∧ 
      R = (R_x, R_x⁻¹) ∧ 
      (Q_x = 2 - Real.sqrt 3 ∨ Q_x = 2 + Real.sqrt 3) ∧
      (R_x = 2 + Real.sqrt 3 ∨ R_x = 2 - Real.sqrt 3))) :=
sorry

end hyperbola_equilateral_triangle_proof_l224_224639


namespace solve_for_x_l224_224298

theorem solve_for_x : ∀ (x : ℝ), (x ≠ 3) → ((x - 3) / (x + 2) + (3 * x - 6) / (x - 3) = 2) → x = 1 / 2 := 
by
  intros x hx h
  sorry

end solve_for_x_l224_224298


namespace maximum_area_PMN_l224_224993

noncomputable def problem_statement :=
  let C := {p : ℝ × ℝ | p.1^2 - 2*p.1 + p.2^2 = 0}
  let l := {p : ℝ × ℝ | p.2 = p.1}
  let ellipse := {p : ℝ × ℝ | p.1^2 / 3 + p.2^2 = 1}
  let M : ℝ × ℝ := (0, 0)
  let N : ℝ × ℝ := (1, 1)
  let polar_M := (0, 0)  -- coordinates in polar form
  let polar_N := (real.sqrt 2, real.pi / 4)
  ∀ P ∈ ellipse,
    let d := (abs (real.sqrt 3 * real.cos P.1 - real.sin P.2)) / real.sqrt 2
    S := (real.sqrt 2 / 2) * d
    S ≤ 1

theorem maximum_area_PMN : ∀ P ∈ ellipse, 
  let d := (abs (real.sqrt 3 * real.cos P.1 - real.sin P.2)) / real.sqrt 2
  let S := (real.sqrt 2 / 2) * d
  S ≤ 1 :=
by sorry

end maximum_area_PMN_l224_224993


namespace interest_rate_B_to_C_l224_224052

-- Definitions based on the given conditions
def principal : ℝ := 1000
def rate_A : ℝ := 0.1  -- 10% per annum in decimal form
def duration : ℝ := 3  -- years
def gain_B : ℝ := 45

-- Calculate total interest paid by B to A
def interest_paid_by_B_to_A := duration * rate_A * principal

-- Calculate total interest received by B from C
def interest_received_by_B_from_C := interest_paid_by_B_to_A + gain_B

-- Calculate annual interest received by B from C
def annual_interest_received_by_B_from_C := interest_received_by_B_from_C / duration

-- Define the proof for the interest rate R at which B lent the money to C
theorem interest_rate_B_to_C : 
  (annual_interest_received_by_B_from_C = (11.5 / 100) * principal) :=
sorry

end interest_rate_B_to_C_l224_224052


namespace polynomial_roots_to_determinant_l224_224633

noncomputable def determinant_eq (a b c m p q : ℂ) : Prop :=
  (Matrix.det ![
    ![a, 1, 1],
    ![1, b, 1],
    ![1, 1, c]
  ] = 2 - m - q)

theorem polynomial_roots_to_determinant (a b c m p q : ℂ) 
  (h1 : Polynomial.eval a (Polynomial.C q + Polynomial.monomial 1 p + Polynomial.monomial 2 m + Polynomial.monomial 3 1) = 0)
  (h2 : Polynomial.eval b (Polynomial.C q + Polynomial.monomial 1 p + Polynomial.monomial 2 m + Polynomial.monomial 3 1) = 0)
  (h3 : Polynomial.eval c (Polynomial.C q + Polynomial.monomial 1 p + Polynomial.monomial 2 m + Polynomial.monomial 3 1) = 0)
  : determinant_eq a b c m p q :=
by sorry

end polynomial_roots_to_determinant_l224_224633


namespace jordan_time_to_run_7_miles_l224_224611

def time_taken (distance time_per_unit : ℝ) : ℝ :=
  distance * time_per_unit

theorem jordan_time_to_run_7_miles :
  ∀ (t_S d_S d_J : ℝ), t_S = 36 → d_S = 6 → d_J = 4 → time_taken 7 ((t_S / 2) / d_J) = 31.5 :=
by
  intros t_S d_S d_J h_t_S h_d_S h_d_J
  -- skipping the proof
  sorry

end jordan_time_to_run_7_miles_l224_224611


namespace find_constants_l224_224113

theorem find_constants (a b c d : ℚ) :
  (6 * x^3 - 4 * x + 2) * (a * x^3 + b * x^2 + c * x + d) =
  18 * x^6 - 2 * x^5 + 16 * x^4 - (28 / 3) * x^3 + (8 / 3) * x^2 - 4 * x + 2 →
  a = 3 ∧ b = -1 / 3 ∧ c = 14 / 9 :=
by
  sorry

end find_constants_l224_224113


namespace balloon_permutations_l224_224941

theorem balloon_permutations : 
  let balloons := "BALLOON".toList.length in
  let total_permutations := Nat.factorial balloons in
  let repetitive_L := Nat.factorial 2 in
  let repetitive_O := Nat.factorial 2 in
  (total_permutations / (repetitive_L * repetitive_O)) = 1260 := sorry

end balloon_permutations_l224_224941


namespace claire_photos_l224_224799

theorem claire_photos (L R C : ℕ) (h1 : L = R) (h2 : L = 3 * C) (h3 : R = C + 28) : C = 14 := by
  sorry

end claire_photos_l224_224799


namespace union_of_sets_l224_224148

theorem union_of_sets :
  let A := {1, 2, 3}
  let B := {-1, 3}
  (A ∪ B) = {-1, 1, 2, 3} :=
by
  sorry

end union_of_sets_l224_224148


namespace balloon_arrangements_l224_224889

noncomputable def factorial (n : ℕ) : ℕ :=
if h : 0 < n then n * factorial (n - 1) else 1

theorem balloon_arrangements : 
  ∃ (n : ℕ), n = (factorial 7) / (factorial 2 * factorial 2) ∧ n = 1260 := by
  have fact_7 := factorial 7
  have fact_2 := factorial 2
  exists (fact_7 / (fact_2 * fact_2))
  split
  · rw [← int.coe_nat_eq_coe_nat_iff]
    norm_cast
    sorry -- Here we do the exact calculations
  · exact rfl

end balloon_arrangements_l224_224889


namespace standard_equation_of_parabola_l224_224701

noncomputable def parabola_equation (a : ℝ) : Prop :=
  (∃ (x0 : ℝ), (x0, 2) on the parabola ∧ |F(0, a / 4) - (x0,2)| = 3) → a = 4

-- Theorem statement
theorem standard_equation_of_parabola :
  parabola_equation 4 :=
by
  sorry

end standard_equation_of_parabola_l224_224701


namespace length_of_AD_l224_224036

-- Define the basic setup
variables {Point : Type} [metric_space Point]
variables (A D B C M : Point)
variables (AD MC : ℝ)

-- Conditions
def is_midpoint (M A D : Point) := dist A M = dist M D
def trisect (B C A D : Point) := dist A B = dist B C ∧ dist B C = dist C D
def midpoint_condition := is_midpoint M A D
def trisect_condition := trisect B C A D
def MC_condition := dist M C = 8

-- Main statement to prove
theorem length_of_AD 
  (H1 : midpoint_condition) 
  (H2 : trisect_condition) 
  (H3 : MC_condition) : 
  dist A D = 48 :=
sorry

end length_of_AD_l224_224036


namespace order_the_numbers_l224_224878

noncomputable def e := 2.718281828459045
noncomputable def π := Real.pi

theorem order_the_numbers :
  3 * e < 3 ∧
  3 < e * π ∧
  e * π < π * e ∧
  π * e < π ^ 3 ∧
  π ^ 3 < 3 * π :=
  sorry

end order_the_numbers_l224_224878


namespace decimal_to_base5_l224_224871

theorem decimal_to_base5 : ∀ (n : ℕ), n = 453 → n = 3 * 5^3 + 3 * 5^2 + 0 * 5^1 + 3 * 5^0 :=
by
  intro n
  assume h : n = 453
  rw [h]
  norm_num
  sorry

end decimal_to_base5_l224_224871


namespace balloon_arrangement_count_l224_224968

-- Definitions of letter frequencies for the word BALLOON
def letter_frequencies : List (Char × Nat) := [('B', 1), ('A', 1), ('L', 2), ('O', 2), ('N', 1)]

-- Total number of letters
def total_letters := 7

-- The formula for the number of unique arrangements of the letters
noncomputable def arrangements :=
  (Nat.factorial total_letters) / 
  (Nat.factorial 1 * Nat.factorial 1 * Nat.factorial 2 * Nat.factorial 2 * Nat.factorial 1)

-- The theorem to prove the number of ways to arrange the letters in "BALLOON"
theorem balloon_arrangement_count : arrangements = 1260 :=
  sorry

end balloon_arrangement_count_l224_224968


namespace cannot_form_right_triangle_l224_224739

theorem cannot_form_right_triangle :
  ¬ (6^2 + 7^2 = 8^2) :=
by
  sorry

end cannot_form_right_triangle_l224_224739


namespace sqrt8_incorrect_statement_l224_224740

theorem sqrt8_incorrect_statement :
  (∀ x, Real.sqrt 8 = x ↔ (x = 2 * Real.sqrt 2)) →
  (Real.sqrt 8 ≠ (2 * Real.sqrt 2)) →
  (2 < Real.sqrt 8 ∧ Real.sqrt 8 < 3) →
  ¬(Real.sqrt 8 = ±(2 * Real.sqrt 2)) :=
by
  sorry

end sqrt8_incorrect_statement_l224_224740


namespace balloon_arrangements_l224_224884

noncomputable def factorial (n : ℕ) : ℕ :=
if h : 0 < n then n * factorial (n - 1) else 1

theorem balloon_arrangements : 
  ∃ (n : ℕ), n = (factorial 7) / (factorial 2 * factorial 2) ∧ n = 1260 := by
  have fact_7 := factorial 7
  have fact_2 := factorial 2
  exists (fact_7 / (fact_2 * fact_2))
  split
  · rw [← int.coe_nat_eq_coe_nat_iff]
    norm_cast
    sorry -- Here we do the exact calculations
  · exact rfl

end balloon_arrangements_l224_224884


namespace solution_pairs_correct_l224_224110

theorem solution_pairs_correct:
  { (n, m) : ℕ × ℕ | m^2 + 2 * 3^n = m * (2^(n+1) - 1) }
  = {(3, 6), (3, 9), (6, 54), (6, 27)} :=
by
  sorry -- no proof is required as per the instruction

end solution_pairs_correct_l224_224110


namespace week_profit_promotional_method_choice_l224_224397

-- Define the initial conditions
def cost_price : ℝ := 10
def standard_price : ℝ := 15
def price_changes : List ℝ := [+1, -3, +2, -1, +3, +4, -9]
def quantities_sold : List ℝ := [20, 35, 10, 30, 15, 5, 50]

-- Define the selling prices for each day based on the standard price and price changes
def selling_prices : List ℝ :=
price_changes.map (λ change, standard_price + change)

-- Unit selling price on October 3rd
def unit_price_oct3 := selling_prices.nth_le 2 (by decide)

-- Proof that total profit for the first week is as given
theorem week_profit : 
  let costs := selling_prices.map (λ sp, sp - cost_price)
  let profits := costs.zip_with (*) quantities_sold
  (17 = unit_price_oct3) ∧ (profits.sum = 245) :=
by
  -- Define the necessary intermediate calculations
  let unit_price_oct3 := 15 + 2
  let total_profit := (16-10) * 20 + (12-10) * 35 + (17-10) * 10 + 
                      (14-10) * 30 + (18-10) * 15 + (19-10) * 5 + 
                      (6-10) * 50 
  have unit_price_correct : unit_price_oct3 = 17 := rfl
  have profit_correct : total_profit = 245 :=
    by norm_num; rfl 
  exact ⟨unit_price_correct, profit_correct⟩

-- Proof about the cost-effectiveness of the promotional methods
theorem promotional_method_choice : 
  let cost_method_one := 5 * 20 + (20 - 5) * 16
  let cost_method_two := 20 * 17
  (cost_method_one ≤ cost_method_two) :=
by
  have h_one := by norm_num; rfl
  have h_two := by norm_num; rfl
  rw [h_one, h_two]
  exact le_refl _

end week_profit_promotional_method_choice_l224_224397


namespace no_statements_are_correct_l224_224227

-- Conditions
variables 
  (A B C M K : Type)
  [inhabited A] [inhabited B] [inhabited C] [inhabited M] [inhabited K]
  (AM BK : ℝ)
  (h_AM : AM = 3)
  (h_BK : BK = 5)

-- Statement
theorem no_statements_are_correct
  (h_AM_is_3 : AM = 3)
  (h_BK_is_5 : BK = 5)
  : ¬(AB = 6 ∨ perimeter_ABC = 22 ∨ ¬(can_estimate_perimeter ∨ can_estimate_AB)) :=
sorry

end no_statements_are_correct_l224_224227


namespace amount_is_15_l224_224570

theorem amount_is_15 (x : ℝ) (A : ℝ) (h1 : x = 1050) (h2 : 0.2 * x = 225 - A) : A = 15 :=
by
  rw [← h1] at h2
  sorry

end amount_is_15_l224_224570


namespace garden_roller_length_l224_224314

/-- The length of a garden roller with diameter 1.4m,
covering 52.8m² in 6 revolutions, and using π = 22/7,
is 2 meters. -/
theorem garden_roller_length
  (diameter : ℝ)
  (total_area_covered : ℝ)
  (revolutions : ℕ)
  (approx_pi : ℝ)
  (circumference : ℝ := approx_pi * diameter)
  (area_per_revolution : ℝ := total_area_covered / (revolutions : ℝ))
  (length : ℝ := area_per_revolution / circumference) :
  diameter = 1.4 ∧ total_area_covered = 52.8 ∧ revolutions = 6 ∧ approx_pi = (22 / 7) → length = 2 :=
by
  sorry

end garden_roller_length_l224_224314


namespace probability_wind_given_haze_l224_224856

variable (A B : Prop)

-- Probabilities given in the problem
variable (P_A : ℝ) (P_B : ℝ) (P_AB : ℝ)

-- Definition of the conditional probability
def conditional_probability (P_AB P_A : ℝ) : ℝ := P_AB / P_A

-- Conditions in the given problem
axiom h1 : P_A = 0.25
axiom h2 : P_B = 0.4
axiom h3 : P_AB = 0.02

-- Target statement to prove
theorem probability_wind_given_haze :
  conditional_probability P_AB P_A = 0.08 := by
  sorry

end probability_wind_given_haze_l224_224856


namespace max_distinct_residues_mod_11_l224_224809

theorem max_distinct_residues_mod_11 (a : ℕ → ℤ) (h : ∀ n, a (n + 1) = a n ^ 3 + a n ^ 2) :
  ∃ k : ℕ, k ≤ 3 ∧ ∀ i, (∃ j < k, a i ≡ a j [MOD 11]) :=
sorry

end max_distinct_residues_mod_11_l224_224809


namespace all_primes_in_sequence_l224_224875

def largest_prime_divisor (n : ℕ) : ℕ := sorry

def sequence (a : ℕ → ℕ) (n : ℕ) : Prop := 
  a 0 = 2 ∧ ∀ n, a (n + 1) = a n + largest_prime_divisor (a n)

theorem all_primes_in_sequence (m : ℕ) : 
  (∃ i : ℕ, sequence (λ n, a n) i ∧ a i = m^2) ↔ m.prime :=
sorry

end all_primes_in_sequence_l224_224875


namespace first_year_exceeds_200_million_l224_224822

noncomputable def exceeds_200_million (n : ℕ) : Prop :=
  let initial_investment := 130000000
  let growth_rate := 1.12
  initial_investment * growth_rate^(n - 2015) > 200000000

theorem first_year_exceeds_200_million : {n : ℕ // exceeds_200_million n ∧ ∀ m < n, ¬ exceeds_200_million m} :=
  ⟨2019, by
    { have H1 : (2019 - 2015) * 0.05 > 0.30 - 0.11 := by
      { rw [nat.sub_sub_self (dec_trivial : 2015 ≤ 2019)],
        norm_num [sub_lt_sub_iff_right, sub_pos],
        norm_num [lt_trans] },
      exact H1 }, sorry⟩

end first_year_exceeds_200_million_l224_224822


namespace remainder_of_f_l224_224372

theorem remainder_of_f (f y : ℤ) 
  (hy : y % 5 = 4)
  (hfy : (f + y) % 5 = 2) : f % 5 = 3 :=
by
  sorry

end remainder_of_f_l224_224372


namespace modulus_of_complex_calc_l224_224333

def modulus (z : ℂ) : ℝ :=
  complex.abs z

theorem modulus_of_complex_calc : 
  modulus ((1 + complex.I) / (1 - complex.I)) ^ 3 = 1 :=
by sorry

end modulus_of_complex_calc_l224_224333


namespace ratio_of_areas_l224_224250

open Real

-- Define points A, B, C, and P in a 2-dimensional vector space over the reals.
variables {V : Type*} [inner_product_space ℝ V] (A B C P : V)

-- Conditions: P inside triangle ABC satisfying the vector equation.
def point_condition (A B C P : V) : Prop :=
  2 • (A - P) + 3 • (B - P) + (C - P) = 0

-- Function to calculate the area of triangle given three points.
def area (A B C : V) : ℝ :=
  0.5 * abs (det ![B - A, C - A])

-- The main statement: the ratio of the area of triangles ABC and APB.
theorem ratio_of_areas (h : point_condition A B C P) :
  area A B C / area A P B = 3 / 2 :=
sorry

end ratio_of_areas_l224_224250


namespace total_fencing_correct_l224_224831

-- Definitions based on conditions
constant width : ℝ := 25
constant area : ℝ := 880
constant radius : ℝ := 5

noncomputable def length : ℝ := area / width

noncomputable def fencing_field : ℝ := 2 * length + width

noncomputable def circumference_garden : ℝ := 2 * Real.pi * radius

noncomputable def total_fencing : ℝ := fencing_field + circumference_garden

-- The theorem to prove
theorem total_fencing_correct : total_fencing = 126.82 := by
  sorry

end total_fencing_correct_l224_224831


namespace determine_b_l224_224991

theorem determine_b (a b : ℤ) (h1 : 3 * a + 4 = 1) (h2 : b - 2 * a = 5) : b = 3 :=
by
  sorry

end determine_b_l224_224991


namespace repeating_decimal_to_fraction_l224_224772

theorem repeating_decimal_to_fraction : (0.5656565656 : ℚ) = 56 / 99 :=
  by
    sorry

end repeating_decimal_to_fraction_l224_224772


namespace recurring_decimal_to_fraction_correct_l224_224759

noncomputable def recurring_decimal_to_fraction (b : ℚ) : Prop :=
  b = 0.\overline{56} ↔ b = 56/99

theorem recurring_decimal_to_fraction_correct : recurring_decimal_to_fraction 0.\overline{56} :=
  sorry

end recurring_decimal_to_fraction_correct_l224_224759


namespace median_is_177_l224_224595

def cumulative_count (n : ℕ) : ℕ := n * (n + 1) / 2

theorem median_is_177 : let total_elements := cumulative_count 250 in
  let median_position := (total_elements + 1) / 2 in
  (∃ n : ℕ, cumulative_count (n - 1) < median_position ∧ median_position ≤ cumulative_count n) → 
  (∃ n : ℕ, n = 177) :=
by
  sorry

end median_is_177_l224_224595


namespace finite_good_tuples_l224_224617

-- Define the conditions of the problem
variables {n : ℕ} (a : fin n → ℕ)
variable (Hn : n > 1)
variable (Hgt1 : ∀ i, 1 < a i)
variable (Hgood : ∀ i, a i ∣ ((finset.univ.prod a) / (a i) - 1))

-- Statement to prove that there are finitely many good n-tuples
theorem finite_good_tuples (n : ℕ) (Hn : n > 1) :
  {a : fin n → ℕ | (∀ i, 1 < a i) ∧ (∀ i, a i ∣ ((finset.univ.prod a) / (a i) - 1))}.finite :=
by sorry

end finite_good_tuples_l224_224617


namespace john_makes_money_l224_224232

variables 
  (jars : ℕ) 
  (caterpillars_per_jar : ℕ) 
  (failure_rate : ℝ) 
  (price_per_butterfly : ℝ)

noncomputable def totalCaterpillars := jars * caterpillars_per_jar
noncomputable def failedCaterpillars := failure_rate * totalCaterpillars
noncomputable def successfulButterflies := totalCaterpillars - failedCaterpillars
noncomputable def totalMoney := successfulButterflies * price_per_butterfly

theorem john_makes_money (h1 : jars = 4) (h2 : caterpillars_per_jar = 10) (h3 : failure_rate = 0.40) (h4 : price_per_butterfly = 3) :
  totalMoney jars caterpillars_per_jar failure_rate price_per_butterfly = 72 :=
by
  sorry

end john_makes_money_l224_224232


namespace equal_probability_l224_224000

-- Define the probability space
noncomputable theory
open Classical

def coin_toss : Type := bool × bool

def HH (x : coin_toss) : Prop := x = (true, true)
def HT (x : coin_toss) : Prop := x = (true, false)
def TH (x : coin_toss) : Prop := x = (false, true)
def TT (x : coin_toss) : Prop := x = (false, false)

def probability (s : set coin_toss) : ℝ :=
  (finset.card {x | s x}.to_finset).to_real / 4

theorem equal_probability :
  probability {x | HH x} = probability {x | HT x} ∧
  probability {x | HT x} = probability {x | TH x} ∧
  probability {x | TH x} = (1/3 : ℝ) :=
by
  sorry

end equal_probability_l224_224000


namespace circle_area_intersection_problem_l224_224451

noncomputable def radius_A : ℝ := 1
noncomputable def radius_B : ℝ := 1
noncomputable def radius_C : ℝ := 2

noncomputable def area_C (r : ℝ) : ℝ := π * r^2
noncomputable def equilateral_triangle_area (s : ℝ) : ℝ := (sqrt 3 / 4) * s^2

theorem circle_area_intersection_problem :
  Circle_A.radius = radius_A ∧ Circle_B.radius = radius_B ∧ Circle_C.radius = radius_C ∧
  Circle_A.is_tangent_to Circle_B ∧ Circle_C.is_tangent_to_midpoint_of_AB →
  area_C radius_C - (2 * (area_C radius_C / 6 - equilateral_triangle_area (2 * radius_A))) = (10 * π / 3) + 2 * sqrt 3 :=
by
  -- Assuming Circle_A and Circle_B have radius 1:
  let Circle_A : Circle := ⟨radius_A⟩
  let Circle_B : Circle := ⟨radius_B⟩
  let Circle_C : Circle := ⟨radius_C⟩
  
  -- Proof goes here, if required
  sorry

end circle_area_intersection_problem_l224_224451


namespace paths_to_spell_ABC12_l224_224194

theorem paths_to_spell_ABC12 : ∃ (n : ℕ), n = 12 := 
begin
    let a_to_b : ℕ := 2,
    let b_to_c : ℕ := 3,
    let c_to_1 : ℕ := 2,
    let one_to_2 : ℕ := 1,
    existsi (a_to_b * b_to_c * c_to_1 * one_to_2),
    repeat {trivial},
end

end paths_to_spell_ABC12_l224_224194


namespace find_P_l224_224632

variable (P : ℝ)
theorem find_P (h : sqrt (3 - 2 * P) + sqrt (1 - 2 * P) = 2) : P = 3 / 8 :=
by
  sorry

end find_P_l224_224632


namespace inequality_N_value_l224_224461

theorem inequality_N_value (a c : ℝ) (ha : 0 < a) (hc : 0 < c) (b : ℝ) (hb : b = 2 * a) : 
  (a^2 + b^2) / c^2 > 5 / 9 := 
by sorry

end inequality_N_value_l224_224461


namespace product_never_zero_l224_224132

open Complex

theorem product_never_zero (n : ℕ) (h : 1 ≤ n ∧ n ≤ 3000) :
  (∏ k in Finset.range n, ((2 + exp (2 * π * I * k / n)) ^ n - 1)) ≠ 0 :=
by
  sorry

end product_never_zero_l224_224132


namespace min_distance_C1_C2_l224_224596

noncomputable def C1_param_x (θ : ℝ) : ℝ := sin θ + cos θ
noncomputable def C1_param_y (θ : ℝ) : ℝ := sin (2 * θ)

def C1_rect_eq (x y : ℝ) : Prop := y = x^2 - 1 ∧ x ∈ Set.Icc (-Real.sqrt 2) (Real.sqrt 2)

noncomputable def C2_polar_eq (ρ θ : ℝ) : ℝ := ρ * sin (θ + Real.pi / 4) + Real.sqrt 2

def C2_rect_eq (x y : ℝ) : Prop := x + y + 2 = 0

theorem min_distance_C1_C2 :
  (∀ θ : ℝ, C1_param_y θ = (C1_param_x θ)^2 - 1) →
  (∀ ρ θ : ℝ, C2_polar_eq ρ θ = 0 → x + y + 2 = 0) →
  (∃ d : ℝ, d = Real.sqrt 2 / Real.abs (C1_param_x θ + 1/2)^2 + 3/4) :=
sorry

end min_distance_C1_C2_l224_224596


namespace value_of_expression_l224_224170

theorem value_of_expression (m a b c d : ℚ) 
  (hm : |m + 1| = 4)
  (hab : a = -b) 
  (hcd : c * d = 1) :
  a + b + 3 * c * d - m = 0 ∨ a + b + 3 * c * d - m = 8 :=
by
  sorry

end value_of_expression_l224_224170


namespace sequence_inequality_l224_224707

open Nat

def sequence_satisfy_conditions (a : ℕ → ℕ) (n : ℕ) :=
  a 0 = 0 ∧ ∀ (k : ℕ), k < n → a(k + 1) ≥ a k + 1

theorem sequence_inequality (a : ℕ → ℕ) (n : ℕ) 
  (h : sequence_satisfy_conditions a n) :
  ∑ k in Finset.range n.succ, (a k)^3 ≥ (∑ k in Finset.range n.succ, a k)^2 :=
sorry

end sequence_inequality_l224_224707


namespace balloon_arrangements_l224_224988

def factorial (n : Nat) : Nat :=
  if n = 0 then 1 else n * factorial (n - 1)

def arrangements (n : Nat) (ks : List Nat) :=
  factorial n / (ks.map factorial).foldr (*) 1

theorem balloon_arrangements : arrangements 7 [2, 2] = 1260 :=
by
  sorry

end balloon_arrangements_l224_224988


namespace missy_tv_watching_time_l224_224268

def reality_show_count : Nat := 5
def reality_show_duration : Nat := 28
def cartoon_duration : Nat := 10

theorem missy_tv_watching_time :
  reality_show_count * reality_show_duration + cartoon_duration = 150 := by
  sorry

end missy_tv_watching_time_l224_224268


namespace positive_difference_a_l224_224636

def f (n : ℤ) : ℤ :=
if n < 0 then n^2 - 2 else 2 * n - 20

theorem positive_difference_a :
  let a1 := if 14 < 0 then (√14 : ℤ) else (-√14 : ℤ)
  let a2 := if 14 >= 0 then (34/2 : ℤ) else (-34/2 : ℤ)
  abs (a1 - a2) = 21 :=
by
  let f_neg2 := if (-2 : ℤ) < 0 then (-2 : ℤ)^2 - 2 else 2 * (-2 : ℤ) - 20
  let f_2 := if (2 : ℤ) < 0 then 2^2 - 2 else 2 * 2 - 20
  let f_a := if (14: ℤ) < 0 then (√14 : ℤ) else (-√14: ℤ)
  let a1 := (f_a - f_2 - f_neg2)
  let a1 := if 14 < 0 then (√14 : ℕ) else (-√14 : ℕ)
  let a2 := if 14 >= 0 then (34/2 : ℤ) else (-34/2 : ℤ)
  sorry

end positive_difference_a_l224_224636


namespace max_value_y_l224_224572

variable (x : ℝ)
def y : ℝ := -3 * x^2 + 6

theorem max_value_y : ∃ M, ∀ x : ℝ, y x ≤ M ∧ (∀ x : ℝ, y x = M → x = 0) :=
by
  use 6
  sorry

end max_value_y_l224_224572


namespace min_sum_of_factors_l224_224336

theorem min_sum_of_factors 
  (a b c: ℕ)
  (h1: a > 0)
  (h2: b > 0)
  (h3: c > 0)
  (h4: a * b * c = 2450) :
  a + b + c ≥ 76 :=
sorry

end min_sum_of_factors_l224_224336


namespace anna_plants_needed_l224_224077

def required_salads : ℕ := 12
def salads_per_plant : ℕ := 3
def loss_fraction : ℚ := 1 / 2

theorem anna_plants_needed : 
  ∀ (plants_needed : ℕ), 
  plants_needed = Nat.ceil (required_salads / salads_per_plant * (1 / (1 - (loss_fraction : ℚ)))) :=
by
  sorry

end anna_plants_needed_l224_224077


namespace duodecimal_divisibility_l224_224699

-- Define the specific divisibility rule for duodecimal system
def last_digit (n : Nat) : Fin 12 := ⟨n % 12, by apply Nat.mod_lt; exact dec_trivial⟩

-- Predicate for checking if a last digit is divisible by a given number
def is_divisible_by (a b : Nat) : Prop := a % b = 0

-- Main theorem
theorem duodecimal_divisibility (n : Nat) : 
  (is_divisible_by n 2 → is_divisible_by (last_digit n).val 2) ∧
  (is_divisible_by n 3 → is_divisible_by (last_digit n).val 3) ∧
  (is_divisible_by n 4 → is_divisible_by (last_digit n).val 4) :=
by
  sorry

end duodecimal_divisibility_l224_224699


namespace student_A_claps_l224_224491

def fibonacci : ℕ → ℕ
| 0 => 0
| 1 => 1
| (n + 2) => fibonacci n + fibonacci (n + 1)

def claps_by_student_A (n : ℕ) : ℕ :=
  (List.range n).count (fun i => i % 15 = 14)

theorem student_A_claps :
  claps_by_student_A 100 = 6 := by
  sorry

end student_A_claps_l224_224491


namespace number_of_stars_yesterday_l224_224295

-- Definitions
variable (t y e : ℕ)

-- Conditions
def total_stars (t y e : ℕ) : Prop := t = y + e
def total_is_7 : Prop := t = 7
def earned_today_is_3 : Prop := e = 3

-- Theorem
theorem number_of_stars_yesterday (h1 : total_stars t y e) (h2 : total_is_7) (h3 : earned_today_is_3) : y = 4 := by
  sorry

end number_of_stars_yesterday_l224_224295


namespace irrational_existence_l224_224290

noncomputable def irrational_solution (a : ℝ) : Prop :=
a ≠ 0 → (¬(∃ q : ℚ, a = q)) →
  ∃ (b b' : ℝ), (¬(∃ p : ℚ, b = p)) ∧ (¬(∃ r : ℚ, b' = r)) ∧
    (∃ q : ℚ, a + b = q) ∧
    (¬(∃ q' : ℚ, a * b = q')) ∧
    (∃ q'' : ℚ, a * b' = q'') ∧
    (¬(∃ q''' : ℚ, a + b' = q''')

theorem irrational_existence :
  ∀ a : ℝ, irrational_solution a := sorry

end irrational_existence_l224_224290


namespace least_number_of_marbles_divisible_l224_224416

theorem least_number_of_marbles_divisible (n : ℕ) : 
  (∀ k ∈ [2, 3, 4, 5, 6, 7, 8], n % k = 0) -> n >= 840 :=
by sorry

end least_number_of_marbles_divisible_l224_224416


namespace isosceles_triangle_to_two_rhombuses_l224_224874

-- Define the vertices and basic conditions for the isosceles triangle
variables {A B C M N : Type}
variables [IsoscelesTriangle A B C] (h1 : AB = AC) (h2 : Midpoint M BC)

-- Define the cut and rearrangement steps
variable (cut1 : LineSegment A M)
variable (cut2 : LineSegment B N)
variable (cut3 : LineSegment C N)
variable [Midpoint N AM]

-- Main statement: isosceles triangle can be rearranged into two rhombuses
theorem isosceles_triangle_to_two_rhombuses : ∃ p1 p2 : Polygon, IsRhombus p1 ∧ IsRhombus p2 :=
by
  sorry

end isosceles_triangle_to_two_rhombuses_l224_224874


namespace largest_product_of_three_numbers_l224_224021

open Finset

theorem largest_product_of_three_numbers : 
  ∀ (s : Finset ℤ), s = {-3, -2, -1, 4, 5} → 
  (∀ a b c, a ∈ s → b ∈ s → c ∈ s → a ≠ b → a ≠ c → b ≠ c → True) →
  ∃ a b c, a ∈ s ∧ b ∈ s ∧ c ∈ s ∧ a ≠ b ∧ a ≠ c ∧ b ≠ c ∧ a * b * c = 30 :=
by
  -- This is the problem statement.
  sorry

end largest_product_of_three_numbers_l224_224021


namespace balloon_permutations_count_l224_224905

-- Definitions of the conditions
def total_letters_count : ℕ := 7
def l_count : ℕ := 2
def o_count : ℕ := 2

-- Now the mathematical problem as a Lean statement
theorem balloon_permutations_count : 
  (Nat.factorial total_letters_count) / ((Nat.factorial l_count) * (Nat.factorial o_count)) = 1260 := 
by
  sorry

end balloon_permutations_count_l224_224905


namespace kittens_initial_l224_224356

variable (initial_kittens : ℕ) (kittens_given_to_jessica : ℕ) (kittens_given_to_sara : ℕ) (kittens_left : ℕ)

def total_kittens_given : ℕ := kittens_given_to_jessica + kittens_given_to_sara

def initial_kittens_calc : ℕ := total_kittens_given + kittens_left

theorem kittens_initial (hJ : kittens_given_to_jessica = 3) 
                        (hS : kittens_given_to_sara = 6) 
                        (hL : kittens_left = 9) :
  initial_kittens = 18 := 
by
  simp [total_kittens_given, initial_kittens_calc, hJ, hS, hL]
  sorry

end kittens_initial_l224_224356


namespace find_m_l224_224510

def f (x : ℝ) : ℝ :=
  if x ≤ -1 then x + 2 else -x^2 + 4 * x

theorem find_m (m : ℝ) (h : f m = -5) : m = -7 ∨ m = 5 := by
  -- Proof to be filled in
  sorry

end find_m_l224_224510


namespace num_positive_integers_satisfying_inequality_l224_224497

theorem num_positive_integers_satisfying_inequality :
  {x : ℕ | 0 < x ∧ 30 < x^2 + 10 * x + 25 ∧ x^2 + 10 * x + 25 < 60}.to_finset.card = 2 :=
by sorry

end num_positive_integers_satisfying_inequality_l224_224497


namespace balloon_arrangements_l224_224880

noncomputable def factorial (n : ℕ) : ℕ :=
if h : 0 < n then n * factorial (n - 1) else 1

theorem balloon_arrangements : 
  ∃ (n : ℕ), n = (factorial 7) / (factorial 2 * factorial 2) ∧ n = 1260 := by
  have fact_7 := factorial 7
  have fact_2 := factorial 2
  exists (fact_7 / (fact_2 * fact_2))
  split
  · rw [← int.coe_nat_eq_coe_nat_iff]
    norm_cast
    sorry -- Here we do the exact calculations
  · exact rfl

end balloon_arrangements_l224_224880


namespace selling_price_per_machine_l224_224090

theorem selling_price_per_machine (parts_cost patent_cost : ℕ) (num_machines : ℕ) 
  (hc1 : parts_cost = 3600) (hc2 : patent_cost = 4500) (hc3 : num_machines = 45) :
  (parts_cost + patent_cost) / num_machines = 180 :=
by
  sorry

end selling_price_per_machine_l224_224090


namespace balloon_permutation_count_l224_224919

-- Define the necessary factorials and the formula for permutations of multiset
def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem balloon_permutation_count : 
  let total_letters := 7,
      repeat_L := 2,
      repeat_O := 2,
      unique_arrangements := factorial total_letters / (factorial repeat_L * factorial repeat_O) 
  in unique_arrangements = 1260 :=
by
  sorry

end balloon_permutation_count_l224_224919


namespace sparrow_flight_recurrence_l224_224106

-- Let there be eight poles standing along the road.
-- Sparow starts at the first pole, can fly to a neighboring pole once a minute.
-- a(n) is the number of ways to reach the last pole in 2n + 1 flights.
-- a(m) = 0 for m < 3
noncomputable def a : ℕ → ℕ
| 0       := 0
| 1       := 0
| 2       := 0
| (n + 3) := sorry -- Definition of a(n) for n >= 4 goes here, which 
                   -- we assume to be given explicitly or defined iteratively.

theorem sparrow_flight_recurrence (n : ℕ) (h : n ≥ 4) :
  a(n) - 7 * a(n - 1) + 15 * a(n - 2) - 10 * a(n - 3) + a(n - 4) = 0 := 
sorry

end sparrow_flight_recurrence_l224_224106


namespace recurring_decimal_sum_l224_224469

theorem recurring_decimal_sum (x y : ℚ) (hx : x = 4/9) (hy : y = 7/9) :
  x + y = 11/9 :=
by
  rw [hx, hy]
  exact sorry

end recurring_decimal_sum_l224_224469


namespace min_sum_of_factors_l224_224335

theorem min_sum_of_factors 
  (a b c: ℕ)
  (h1: a > 0)
  (h2: b > 0)
  (h3: c > 0)
  (h4: a * b * c = 2450) :
  a + b + c ≥ 76 :=
sorry

end min_sum_of_factors_l224_224335


namespace diagonal_is_5_l224_224649

def diagonal_length_of_rectangle (w l d : ℝ) : Prop :=
  w = 4 ∧ (w * l = 12) ∧ (d^2 = w^2 + l^2)

theorem diagonal_is_5 : ∃ d, ∃ l, diagonal_length_of_rectangle 4 l d ∧ d = 5 :=
by
  use 3   -- use 3 for the length l
  use 5   -- use 5 for the diagonal d
  simp [diagonal_length_of_rectangle]  -- simplify and check the conditions
  split
  repeat { sorry }  -- placeholders for the actual proof

end diagonal_is_5_l224_224649


namespace kolya_is_collection_agency_l224_224216

-- Defining the conditions
def lent_books (Katya Vasya : Type) : Prop := sorry
def missed_return_date (Vasya : Type) : Prop := sorry
def agreed_to_help_retrieve (Katya Kolya Vasya : Type) : Prop := sorry
def received_reward (Kolya : Type) : Prop := sorry

-- Problem statement
theorem kolya_is_collection_agency (Katya Kolya Vasya : Type)
  (h1 : lent_books Katya Vasya) 
  (h2 : missed_return_date Vasya) 
  (h3 : agreed_to_help_retrieve Katya Kolya Vasya)
  (h4 : received_reward Kolya) : 
  ⟦Kolya's Role⟧ = "collection agency" := 
sorry

end kolya_is_collection_agency_l224_224216


namespace total_cats_and_kittens_received_l224_224719

theorem total_cats_and_kittens_received (total_adult_cats : ℕ) (percentage_female : ℕ) (fraction_with_kittens : ℚ) (kittens_per_litter : ℕ) 
  (h1 : total_adult_cats = 100) (h2 : percentage_female = 40) (h3 : fraction_with_kittens = 2 / 3) (h4 : kittens_per_litter = 3) :
  total_adult_cats + ((percentage_female * total_adult_cats / 100) * (fraction_with_kittens * total_adult_cats * kittens_per_litter) / 100) = 181 := by
  sorry

end total_cats_and_kittens_received_l224_224719


namespace balloon_permutation_count_l224_224925

-- Define the given conditions:
def balloon_letters : List Char := ['B', 'A', 'L', 'L', 'O', 'O', 'N']
def total_letters : Nat := 7
def count_L : Nat := 2
def count_O : Nat := 2

-- Statement to prove:
theorem balloon_permutation_count :
  (nat.factorial total_letters) / ((nat.factorial count_L) * (nat.factorial count_O)) = 1260 := by
  sorry

end balloon_permutation_count_l224_224925


namespace balloon_permutations_l224_224963

theorem balloon_permutations : (∀ (arrangements : ℕ),
  arrangements = nat.factorial 7 / (nat.factorial 2 * nat.factorial 2) → arrangements = 1260) :=
begin
  intros arrangements h,
  rw [nat.factorial_succ, nat.factorial],
  sorry,
end

end balloon_permutations_l224_224963


namespace balloon_arrangements_l224_224986

def factorial (n : Nat) : Nat :=
  if n = 0 then 1 else n * factorial (n - 1)

def arrangements (n : Nat) (ks : List Nat) :=
  factorial n / (ks.map factorial).foldr (*) 1

theorem balloon_arrangements : arrangements 7 [2, 2] = 1260 :=
by
  sorry

end balloon_arrangements_l224_224986


namespace min_sum_abc_l224_224338

theorem min_sum_abc (a b c : ℕ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0) (h_prod : a * b * c = 2450) : a + b + c ≥ 82 :=
sorry

end min_sum_abc_l224_224338


namespace balloon_permutations_l224_224961

theorem balloon_permutations : (∀ (arrangements : ℕ),
  arrangements = nat.factorial 7 / (nat.factorial 2 * nat.factorial 2) → arrangements = 1260) :=
begin
  intros arrangements h,
  rw [nat.factorial_succ, nat.factorial],
  sorry,
end

end balloon_permutations_l224_224961


namespace ratio_fraction_l224_224480

theorem ratio_fraction (x : ℚ) : x = 2 / 9 ↔ (2 / 6) / x = (3 / 4) / (1 / 2) := by
  sorry

end ratio_fraction_l224_224480


namespace harvest_apples_every_3_months_l224_224612

def harvest_period (total_yearly_earnings oranges_per_year apple_harvest_income orange_harvest_income months_in_year orange_harvest_frequency) : ℕ := (months_in_year / ((total_yearly_earnings - (oranges_per_year * orange_harvest_income)) / apple_harvest_income))

theorem harvest_apples_every_3_months :
  ∀ (total_yearly_earnings oranges_per_year apple_harvest_income orange_harvest_income months_in_year orange_harvest_frequency), 
      total_yearly_earnings = 420 → 
      orange_harvest_frequency = 2 → 
      orange_harvest_income = 50 → 
      apple_harvest_income = 30 →
      months_in_year = 12 → 
      oranges_per_year = (months_in_year / orange_harvest_frequency) →
      harvest_period total_yearly_earnings oranges_per_year apple_harvest_income orange_harvest_income months_in_year orange_harvest_frequency = 3 := 
by 
  intros total_yearly_earnings oranges_per_year apple_harvest_income orange_harvest_income months_in_year orange_harvest_frequency 
  intros H1 H2 H3 H4 H5 H6
  rw [H1, H2, H3, H4, H5, H6]
  unfold harvest_period
  sorry

end harvest_apples_every_3_months_l224_224612


namespace min_y_value_l224_224525

noncomputable def f (x : ℝ) : ℝ := real.sqrt (x^2 - 2 * x + 2) + real.sqrt (x^2 - 10 * x + 34)

theorem min_y_value :
  ∃ x : ℝ, (∀ y : ℝ, f y ≥ f x) ∧ f x = 4 * real.sqrt 2 :=
sorry

end min_y_value_l224_224525


namespace at_least_two_balls_same_color_l224_224583

theorem at_least_two_balls_same_color (R W : ℕ) (total_balls : ℕ) (drawn_balls : ℕ) 
  (hR : R = 13) (hW : W = 7) (htotal : total_balls = R + W) (hdrawn : drawn_balls = 3) :
  ∃ (colors : set (set ℕ)), 
  colors = {{r, r, w}, {w, w, r}, {r, r, r}, {w, w, w}, {r, w, r}, {w, r, w}, {r, w, w}, {w, r, r}} ∧ 
  (∀ three_drawn ∈ colors, (∃ (same_color : ℕ → Prop), (∃ x ∈ three_drawn, same_color x) ∧ (∃ y ∈ three_drawn, same_color y) ∧ same_color x ∧ same_color y)) :=
by {
  sorry -- Proof goes here
}

end at_least_two_balls_same_color_l224_224583


namespace integer_solution_for_binom_expression_l224_224500

noncomputable def binom (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem integer_solution_for_binom_expression (k n : ℕ) (hk1 : 1 ≤ k) (hk2 : k < n) (hn3 : 3 ∣ n) : 
  ∃ m : ℤ, (2 * ↑n - 3 * ↑k - 2) * binom n k / (↑k + 2) = ↑m :=
by
  sorry

end integer_solution_for_binom_expression_l224_224500


namespace correct_statements_l224_224130

variables (a : Nat → ℤ) (d : ℤ)

-- Suppose {a_n} is an arithmetic sequence with common difference d
def S (n : ℕ) : ℤ := (n * (2 * a 1 + (n - 1) * d)) / 2

-- Conditions: S_11 > 0 and S_12 < 0
axiom S11_pos : S a d 11 > 0
axiom S12_neg : S a d 12 < 0

-- The goal is to determine which statements are correct
theorem correct_statements : (d < 0) ∧ (∀ n, 1 ≤ n → n ≤ 12 → S a d 6 ≥ S a d n ∧ S a d 6 ≠ S a d 11 ) := 
sorry

end correct_statements_l224_224130


namespace problem1_problem2_l224_224038

theorem problem1 :
  (27 : ℝ)^(2/3) - 2^(Real.log2 3) * Real.log2 (1/8) + Real.log2 3 * (Real.logb 3 4) = 20 :=
by
  sorry 

theorem problem2 (α : ℝ) :
  (sin (α - π/2) * cos (3*π/2 + α) * tan (π - α)) / (tan (-α - π) * sin (-α - π)) = -cos α :=
by
  sorry

end problem1_problem2_l224_224038


namespace robot_glove_ring_permutations_l224_224413

theorem robot_glove_ring_permutations :
  let n := 6 in
  let total_items := 2 * n in
  let factorial := Nat.factorial in
  n * 2 = total_items →
  (factorial total_items) / (2^n) = (factorial 12) / (2^6) :=
by {
    intros n total_items h,
    rw [h, Nat.mul_comm 2 n, Nat.mul_assoc, Nat.factorial],
    exact eq.refl (factorial 12 / 2^6)
}

end robot_glove_ring_permutations_l224_224413


namespace dandelions_survive_to_flower_l224_224104

def seeds_initial : ℕ := 300
def seeds_in_water : ℕ := seeds_initial / 3
def seeds_eaten_by_insects : ℕ := seeds_initial / 6
def seeds_remaining : ℕ := seeds_initial - seeds_in_water - seeds_eaten_by_insects
def seeds_to_flower : ℕ := seeds_remaining / 2

theorem dandelions_survive_to_flower : seeds_to_flower = 75 := by
  sorry

end dandelions_survive_to_flower_l224_224104


namespace revenue_increase_consistent_l224_224703

noncomputable theory
open_locale classical

structure RevenueData :=
(revenues : ℕ → ℝ)

def monthly_revenues (year1 : List ℝ) (year2 : List ℝ) : RevenueData :=
{ revenues := λ n, if n < 12 then year1.nth_le n (by decide) else year2.nth_le (n - 12) (by decide)}

def monthly_increase (data : RevenueData) (n : ℕ) : ℝ :=
data.revenues (n + 1) - data.revenues n

theorem revenue_increase_consistent :
  let year1 := [$150000, $180000, $210000, $240000, $270000, $300000,
                 $330000, $300000, $270000, $300000, $330000, $360000] in
  let year2 := [$390000, $420000] in
  let data := monthly_revenues year1 year2 in
  ∀ n, (0 ≤ n ∧ n < 12 ∨ 13 ≤ n ∧ n < 13 + 12) → 
  (monthly_increase data n) = 30000 ∨ (n = 7 ∨ n = 8) → (monthly_increase data n) = -30000 := 
begin
  sorry,
end

end revenue_increase_consistent_l224_224703


namespace balloon_permutations_l224_224954

theorem balloon_permutations : 
  let total_letters := 7
  let repetitions_L := 2
  let repetitions_O := 2
  (Nat.factorial total_letters) / ((Nat.factorial repetitions_L) * (Nat.factorial repetitions_O)) = 1260 := 
by
  sorry

end balloon_permutations_l224_224954


namespace proof_problem_l224_224292

variables {l m : Line} {α β : Plane}

-- Definitions for parallel and perpendicular
def parallel (x y : Set) : Prop := ∃ u, u ∈ x ∧ u ∈ y
def perpendicular (x y : Set) : Prop := ∀ u v, u ∈ x ∧ v ∈ y → ∃ w, w ∈ x ∧ w ∈ y ∧ (u ⬝ v = 0)

theorem proof_problem (h1 : perpendicular l α) (h2 : parallel l β) : perpendicular α β := 
sorry

end proof_problem_l224_224292


namespace natural_pair_prime_ratio_l224_224111

noncomputable def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem natural_pair_prime_ratio :
  ∃ (x y : ℕ), (x = 14 ∧ y = 2) ∧ is_prime (x * y^3 / (x + y)) :=
by
  use 14
  use 2
  sorry

end natural_pair_prime_ratio_l224_224111


namespace dandelions_survive_to_flower_l224_224103

def seeds_initial : ℕ := 300
def seeds_in_water : ℕ := seeds_initial / 3
def seeds_eaten_by_insects : ℕ := seeds_initial / 6
def seeds_remaining : ℕ := seeds_initial - seeds_in_water - seeds_eaten_by_insects
def seeds_to_flower : ℕ := seeds_remaining / 2

theorem dandelions_survive_to_flower : seeds_to_flower = 75 := by
  sorry

end dandelions_survive_to_flower_l224_224103


namespace balloon_permutations_l224_224940

theorem balloon_permutations : 
  let balloons := "BALLOON".toList.length in
  let total_permutations := Nat.factorial balloons in
  let repetitive_L := Nat.factorial 2 in
  let repetitive_O := Nat.factorial 2 in
  (total_permutations / (repetitive_L * repetitive_O)) = 1260 := sorry

end balloon_permutations_l224_224940


namespace additional_knight_l224_224807

open ChessBoard Knight

theorem additional_knight (board : ChessBoard) (knights : Finset (Fin 64)) (hknights : knights.card = 11)
  (no_attack : ∀ k1 k2 ∈ knights, k1 ≠ k2 → ¬ attack k1 k2) :
  ∃ k', k' ∉ knights ∧ ∀ k ∈ knights, ¬ attack k k' :=
sorry

end additional_knight_l224_224807


namespace sum_of_elements_in_M_l224_224167

theorem sum_of_elements_in_M (m : ℝ) (h : ∃ x : ℝ, x^2 - 2 * x + m = 0) :
  (∀ x : ℝ, x ∈ {x | x^2 - 2 * x + m = 0} → x = 1) ∧ m = 1 ∨
  (∃ x1 x2 : ℝ, x1 ∈ {x | x^2 - 2 * x + m = 0} ∧ x2 ∈ {x | x^2 - 2 * x + m = 0} ∧ x1 ≠ x2 ∧
   x1 + x2 = 2 ∧ m < 1) :=
sorry

end sum_of_elements_in_M_l224_224167


namespace blue_marbles_difference_l224_224007

-- Definitions of the conditions
def total_green_marbles := 95

-- Ratios for Jar 1 and Jar 2
def ratio_blue_green_jar1 := (9, 1)
def ratio_blue_green_jar2 := (8, 1)

-- Total number of green marbles in each jar
def green_marbles_jar1 (a : ℕ) := a
def green_marbles_jar2 (b : ℕ) := b

-- Total number of marbles in each jar
def total_marbles_jar1 (a : ℕ) := 10 * a
def total_marbles_jar2 (b : ℕ) := 9 * b

-- Number of blue marbles in each jar
def blue_marbles_jar1 (a : ℕ) := 9 * a
def blue_marbles_jar2 (b : ℕ) := 8 * b

-- Conditions in terms of Lean definitions
theorem blue_marbles_difference:
  ∀ (a b : ℕ), green_marbles_jar1 a + green_marbles_jar2 b = total_green_marbles →
  total_marbles_jar1 a = total_marbles_jar2 b →
  blue_marbles_jar1 a - blue_marbles_jar2 b = 5 :=
by sorry

end blue_marbles_difference_l224_224007


namespace gravitational_force_at_distance_320000_miles_l224_224326

theorem gravitational_force_at_distance_320000_miles:
  ∀ (d₁ d₂ : ℝ) (f₁ : ℝ),
    (f₂ : ℝ) → 
    (h₁ : d₁ = 8000) 
    (h₂ : f₁ = 150)
    (h₃ : d₂ = 320000),
    (f₁ * d₁^2 = f₂ * d₂^2) →
    (f₂ = 3 / 32):=
by
  intros d₁ d₂ f₁ f₂ h₁ h₂ h₃ h_proportional
  sorry

end gravitational_force_at_distance_320000_miles_l224_224326


namespace trapezoid_parallel_line_intersections_l224_224142

/-- Given a trapezoid ABCD with parallel bases BC and AD such that BC = a and AD = b,
a line parallel to the bases intersects AB at P and CD at Q,
and intersects diagonals AC and BD at L and R respectively,
with diagonals AC and BD intersecting at O and the areas of triangles BOC and LOR equal,
then PQ is given by (a * (3 * b - a)) / (b + a). -/
theorem trapezoid_parallel_line_intersections
  (ABCD : Trapezoid)
  (BC AD : ℝ)
  (P Q L R O : Point)
  (hBC : BC = a)
  (hAD : AD = b)
  (h1 : LineIntersABCD_parallel_legs ABCD P Q L R)
  (h2 : TrianglesEqualArea (BOC B O C) (LOR L O R))
  (h3 : Point_between L A O) :
  Distance (P, Q) = (a * (3 * b - a)) / (b + a) :=
by
  sorry

end trapezoid_parallel_line_intersections_l224_224142


namespace initial_speeds_l224_224646

/-- Motorcyclists Vasya and Petya ride at constant speeds around a circular track 1 km long.
    Petya overtakes Vasya every 2 minutes. Then Vasya doubles his speed and now he himself 
    overtakes Petya every 2 minutes. What were the initial speeds of Vasya and Petya? 
    Answer: 1000 and 1500 meters per minute.
-/

theorem initial_speeds (V_v V_p : ℕ) (track_length : ℕ) (time_interval : ℕ) 
  (h1 : track_length = 1000)
  (h2 : time_interval = 2)
  (h3 : V_p - V_v = track_length / time_interval)
  (h4 : 2 * V_v - V_p = track_length / time_interval):
  V_v = 1000 ∧ V_p = 1500 :=
by
  sorry

end initial_speeds_l224_224646


namespace find_value_l224_224546

noncomputable def real_numbers (x : ℝ) := true

def condition_A (x : ℝ) : Prop := x^2 - 2*x - 3 > 0

def condition_B (x : ℝ) (a b c : ℝ) : Prop := a * x^2 + b * x + c ≤ 0 ∧ a ≠ 0 ∧ c ≠ 0

def interval_I := set.Ioc 3 5

def union_real := @set.univ ℝ

theorem find_value (a b c : ℝ) (h₁ : ∀ x, condition_A x ∨ condition_B x a b c) 
    (h₂ : ∀ x, interval_I x ⟷ (condition_A x ∧ condition_B x a b c)) 
    (h₃ : a * 1 * 1 - b + c = 0) 
    (h₄ : a * 5^2 + b * 5 + c = 0) :
    (b / a + a^2 / c^2 = - (74 / 25)) :=
begin
    sorry
end

end find_value_l224_224546


namespace odds_of_picking_blue_marble_l224_224449

theorem odds_of_picking_blue_marble :
  ∀ (total_marbles yellow_marbles : ℕ)
  (h1 : total_marbles = 60)
  (h2 : yellow_marbles = 20)
  (green_marbles : ℕ)
  (h3 : green_marbles = yellow_marbles / 2)
  (remaining_marbles : ℕ)
  (h4 : remaining_marbles = total_marbles - yellow_marbles - green_marbles)
  (blue_marbles : ℕ)
  (h5 : blue_marbles = remaining_marbles / 2),
  (blue_marbles / total_marbles : ℚ) * 100 = 25 :=
by
  intros total_marbles yellow_marbles h1 h2 green_marbles h3 remaining_marbles h4 blue_marbles h5
  sorry

end odds_of_picking_blue_marble_l224_224449


namespace average_marks_of_all_students_l224_224385

/-
Consider two classes:
- The first class has 12 students with an average mark of 40.
- The second class has 28 students with an average mark of 60.

We are to prove that the average marks of all students from both classes combined is 54.
-/

theorem average_marks_of_all_students (s1 s2 : ℕ) (m1 m2 : ℤ)
  (h1 : s1 = 12) (h2 : m1 = 40) (h3 : s2 = 28) (h4 : m2 = 60) :
  (s1 * m1 + s2 * m2) / (s1 + s2) = 54 :=
by
  rw [h1, h2, h3, h4]
  sorry

end average_marks_of_all_students_l224_224385


namespace anna_plants_needed_l224_224076

def required_salads : ℕ := 12
def salads_per_plant : ℕ := 3
def loss_fraction : ℚ := 1 / 2

theorem anna_plants_needed : 
  ∀ (plants_needed : ℕ), 
  plants_needed = Nat.ceil (required_salads / salads_per_plant * (1 / (1 - (loss_fraction : ℚ)))) :=
by
  sorry

end anna_plants_needed_l224_224076


namespace limit_of_R_l224_224238

noncomputable def R (m b : ℝ) : ℝ :=
  let x := ((-b) + Real.sqrt (b^2 + 4 * m)) / 2
  m * x + 3 

theorem limit_of_R (b : ℝ) (hb : b ≠ 0) : 
  (∀ m : ℝ, m < 3) → 
  (∀ ε > 0, ∃ δ > 0, ∀ x, abs (x - 0) < δ → abs ((R x (-b) - R x b) / x - b) < ε) :=
by
  sorry

end limit_of_R_l224_224238


namespace number_of_lattice_points_in_triangle_l224_224282

theorem number_of_lattice_points_in_triangle (N L S : ℕ) (A B O : (ℕ × ℕ)) :
  (A = (0, 30)) →
  (B = (20, 10)) →
  (O = (0, 0)) →
  (S = 300) →
  (L = 60) →
  S = N + L / 2 - 1 →
  N = 271 :=
by
  intros hA hB hO hS hL hPick
  sorry

end number_of_lattice_points_in_triangle_l224_224282


namespace determine_a_from_quad_function_l224_224166

theorem determine_a_from_quad_function :
  ∃ a : ℝ, (∀ x in Set.Icc (-3 : ℝ) (2 : ℝ), f x = a * x^2 + 2 * a * x + 1) ∧
            (∀ x in Set.Icc (-3 : ℝ) (2 : ℝ), f x ≤ 4) ∧
            (∃ x in Set.Icc (-3 : ℝ) (2 : ℝ), f x = 4) ∧
            (a = 3/8 ∨ a = -3) :=
by
  -- Define the function f
  let f : ℝ → ℝ := λ x, a * x^2 + 2 * a * x + 1

  -- Conditions based on the given problem
  have f_defined : ∀ x ∈ Set.Icc (-3 : ℝ) (2 : ℝ), f x = a * x^2 + 2 * a * x + 1 :=
    by sorry

  have f_bounded : ∀ x ∈ Set.Icc (-3 : ℝ) (2 : ℝ), f x ≤ 4 :=
    by sorry

  have f_max_value : ∃ x ∈ Set.Icc (-3 : ℝ) (2 : ℝ), f x = 4 :=
    by sorry

  -- Combine the conditions and conclude the possible values for a
  existsi (a : ℝ)
  split
  exact f_defined
  split
  exact f_bounded
  split
  exact f_max_value
  sorry

end determine_a_from_quad_function_l224_224166


namespace ratio_of_areas_l224_224251

open Real

-- Define points A, B, C, and P in a 2-dimensional vector space over the reals.
variables {V : Type*} [inner_product_space ℝ V] (A B C P : V)

-- Conditions: P inside triangle ABC satisfying the vector equation.
def point_condition (A B C P : V) : Prop :=
  2 • (A - P) + 3 • (B - P) + (C - P) = 0

-- Function to calculate the area of triangle given three points.
def area (A B C : V) : ℝ :=
  0.5 * abs (det ![B - A, C - A])

-- The main statement: the ratio of the area of triangles ABC and APB.
theorem ratio_of_areas (h : point_condition A B C P) :
  area A B C / area A P B = 3 / 2 :=
sorry

end ratio_of_areas_l224_224251


namespace polygon_divided_into_7_triangles_l224_224307

theorem polygon_divided_into_7_triangles (n : ℕ) (h : n - 2 = 7) : n = 9 :=
by
  sorry

end polygon_divided_into_7_triangles_l224_224307


namespace garden_area_increase_l224_224041

theorem garden_area_increase :
    let length := 60
    let width := 20
    let perimeter := 2 * (length + width)
    let side_of_square := perimeter / 4
    let area_rectangular := length * width
    let area_square := side_of_square * side_of_square
    area_square - area_rectangular = 400 :=
by
  sorry

end garden_area_increase_l224_224041


namespace find_sum_of_perimeters_l224_224682

variables (x y : ℝ)
noncomputable def sum_of_perimeters := 4 * x + 4 * y

theorem find_sum_of_perimeters (h1 : x^2 + y^2 = 65) (h2 : x^2 - y^2 = 33) :
  sum_of_perimeters x y = 44 :=
sorry

end find_sum_of_perimeters_l224_224682


namespace unknown_support_preferences_l224_224193

theorem unknown_support_preferences:
  let total_audience := 1500
  let team_a_percent := 0.40
  let team_b_percent := 0.30
  let team_c_percent := 0.20
  let team_d_percent := 0.10
  let team_e_percent := 0.05
  let overlap_ab_percent := 0.12
  let overlap_bc_percent := 0.15
  let overlap_cd_percent := 0.10
  let overlap_de_percent := 0.05
  let non_supporter_percent := 0.04
  
  let team_a_supporters := team_a_percent * total_audience
  let team_b_supporters := team_b_percent * total_audience
  let team_c_supporters := team_c_percent * total_audience
  let team_d_supporters := team_d_percent * total_audience
  let team_e_supporters := team_e_percent * total_audience
  
  let overlap_ab := overlap_ab_percent * team_a_supporters
  let overlap_bc := overlap_bc_percent * team_b_supporters
  let overlap_cd := overlap_cd_percent * team_c_supporters
  let overlap_de := overlap_de_percent * team_d_supporters
  
  let non_supporters := non_supporter_percent * total_audience
  
  let total_supporters := team_a_supporters + team_b_supporters + team_c_supporters + team_d_supporters + team_e_supporters
  let total_supporters_no_overlap := total_supporters - (overlap_ab + overlap_bc + overlap_cd + overlap_de)
  let total_accounted_for := total_supporters_no_overlap + non_supporters
  let unknown_preferences := total_audience - total_accounted_for
  
  in unknown_preferences = 43 := 
begin
  sorry
end

end unknown_support_preferences_l224_224193


namespace line_through_point_parallel_to_given_line_line_through_point_sum_intercepts_is_minus_four_l224_224536

theorem line_through_point_parallel_to_given_line :
  ∃ c : ℤ, (∀ x y : ℤ, 2 * x + 3 * y + c = 0 ↔ (x, y) = (2, 1)) ∧ c = -7 :=
sorry

theorem line_through_point_sum_intercepts_is_minus_four :
  ∃ (a b : ℤ), (∀ x y : ℤ, (x / a) + (y / b) = 1 ↔ (x, y) = (-3, 1)) ∧ (a + b = -4) ∧ 
  ((a = -6 ∧ b = 2) ∨ (a = -2 ∧ b = -2)) ∧ 
  ((∀ x y : ℤ, x - 3 * y + 6 = 0 ↔ (x, y) = (-3, 1)) ∨ 
  (∀ x y : ℤ, x + y + 2 = 0 ↔ (x, y) = (-3, 1))) :=
sorry

end line_through_point_parallel_to_given_line_line_through_point_sum_intercepts_is_minus_four_l224_224536


namespace balloon_permutation_count_l224_224922

-- Define the necessary factorials and the formula for permutations of multiset
def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem balloon_permutation_count : 
  let total_letters := 7,
      repeat_L := 2,
      repeat_O := 2,
      unique_arrangements := factorial total_letters / (factorial repeat_L * factorial repeat_O) 
  in unique_arrangements = 1260 :=
by
  sorry

end balloon_permutation_count_l224_224922


namespace periodic_property_l224_224788

-- Define the necessary functions and properties
def is_non_decreasing (f : ℝ → ℝ) := ∀ x y : ℝ, x ≤ y → f x ≤ f y

def satisfies_functional_equation (f : ℝ → ℝ) := ∀ x : ℝ, f (x + 1) = f (x) + 1

-- Define the n-th iterate of function f
def iterate (f : ℝ → ℝ) : ℕ → (ℝ → ℝ)
| 0       := id
| (n + 1) := f ∘ iterate n

-- Define φ(x)
def φ (f : ℝ → ℝ) (n : ℕ) (x : ℝ) := iterate f n x - x

-- Define the main theorem to be proved
theorem periodic_property (f : ℝ → ℝ) (n : ℕ) (h1 : is_non_decreasing f) (h2 : satisfies_functional_equation f) :
  ∀ x y : ℝ, |φ f n x - φ f n y| < 1 :=
by
  -- Insert the proof here
  sorry

end periodic_property_l224_224788


namespace cot_triang_min_value_l224_224219

open Real Topology

theorem cot_triang_min_value (A B C : ℝ) (h : A + B + C = π) :
  |((cot A) + (cot B)) * ((cot B) + (cot C)) * ((cot C) + (cot A))| = 8 * sqrt 3 / 9 :=
sorry

end cot_triang_min_value_l224_224219


namespace complement_of_A_in_U_l224_224550

open Set

def U : Set ℕ := {1, 2, 3, 4}
def A : Set ℕ := {1, 3}
def complementA : Set ℕ := {2, 4}

theorem complement_of_A_in_U :
  (U \ A) = complementA :=
by
  sorry

end complement_of_A_in_U_l224_224550


namespace ao_parallel_hk_l224_224196

open Triangle Geometry

variables {A B C H I O K : Point}
variables {BC : Line}

-- Define the conditions on the variables
def conditions (T : Triangle A B C) (H : Point) (I : Point) (O : Point) (K : Point) (BC : Line) : Prop :=
  is_orthocenter T H ∧ is_incenter T I ∧ is_circumcenter T O ∧
  tangency_point_with_incircle T BC K ∧ parallel (line_through_points I O) BC

-- Define the proof statement
theorem ao_parallel_hk (T : Triangle A B C) (H : Point) (I : Point) (O : Point) (K : Point) (BC : Line)
  (h_conditions : conditions T H I O K BC) : 
  parallel (line_through_points A O) (line_through_points H K) :=
sorry

end ao_parallel_hk_l224_224196


namespace repeating_decimal_fraction_l224_224783

theorem repeating_decimal_fraction : ∀ x : ℚ, (x = 0.5656565656565656) → 100 * x = 56.5656565656565656 → 100 * x - x = 56.5656565656565656 - 0.5656565656565656
  → 99 * x = 56 → x = 56 / 99 :=
begin
  intros x h1 h2 h3 h4,
  sorry,
end

end repeating_decimal_fraction_l224_224783


namespace power_of_point_l224_224034

theorem power_of_point (S1 S2 : Circle) (A B P : Point) (p : ℝ) (h : ℝ) (d : ℝ)
  (hS1 : S1 ∩ S2 = {A, B}) (h_power : power_of_point P S2 = p)
  (h_dist : dist P (line_through A B) = h)
  (h_center_dist : dist S1.center S2.center = d) :
  |p| = 2 * d * h := sorry

end power_of_point_l224_224034


namespace recurring_to_fraction_l224_224757

theorem recurring_to_fraction : ∀ (x : ℚ), x = 0.5656 ∧ 100 * x = 56.5656 → x = 56 / 99 :=
by
  intros x hx
  sorry

end recurring_to_fraction_l224_224757


namespace total_expenditure_l224_224402

-- Definitions of costs and purchases
def bracelet_cost : ℕ := 4
def keychain_cost : ℕ := 5
def coloring_book_cost : ℕ := 3

def paula_bracelets : ℕ := 2
def paula_keychains : ℕ := 1

def olive_coloring_books : ℕ := 1
def olive_bracelets : ℕ := 1

-- Hypothesis stating the total expenditure for Paula and Olive
theorem total_expenditure
  (bracelet_cost keychain_cost coloring_book_cost : ℕ)
  (paula_bracelets paula_keychains olive_coloring_books olive_bracelets : ℕ) :
  paula_bracelets * bracelet_cost + paula_keychains * keychain_cost + olive_coloring_books * coloring_book_cost + olive_bracelets * bracelet_cost = 20 := 
  by
  -- Applying the given costs
  let bracelet_cost := 4
  let keychain_cost := 5
  let coloring_book_cost := 3 

  -- Applying the purchases made by Paula and Olive
  let paula_bracelets := 2
  let paula_keychains := 1
  let olive_coloring_books := 1
  let olive_bracelets := 1

  sorry

end total_expenditure_l224_224402


namespace candidateA_votes_l224_224200

theorem candidateA_votes (total_votes : ℕ) (invalid_percentage : ℕ) (candidateA_percentage : ℕ)
  (h1 : total_votes = 560000)
  (h2 : invalid_percentage = 15)
  (h3 : candidateA_percentage = 75) :
  let valid_votes := (85 * total_votes) / 100 in
  let candidateA_votes := (candidateA_percentage * valid_votes) / 100 in
  candidateA_votes = 357000 :=
by
  -- Proof omitted
  sorry

end candidateA_votes_l224_224200


namespace balloon_arrangements_l224_224982

def factorial (n : Nat) : Nat :=
  if n = 0 then 1 else n * factorial (n - 1)

def arrangements (n : Nat) (ks : List Nat) :=
  factorial n / (ks.map factorial).foldr (*) 1

theorem balloon_arrangements : arrangements 7 [2, 2] = 1260 :=
by
  sorry

end balloon_arrangements_l224_224982


namespace compute_nested_star_l224_224700

def star (a b : ℕ) : ℝ := (a - b) / (1 - a * b)

theorem compute_nested_star : 
  star 2 (star 4 (star 6 (star 8 (star 10 (star 12 (star 14 (star 16 (star 18 (star 20 (star 22 (star 24 (star 26 (star 28 (star 30 (star 32 (star 34 (star 36 (star 38 (star 40 (star 42 (star 44 (star 46 (star 48 (star 50 (star 52 (star 54 (star 56 (star 58 (star 60 (star 62 (star 64 (star 66 (star 68 (star 70 (star 72 (star 74 (star 76 (star 78 (star 80 (star 82 (star 84 (star 86 (star 88 (star 90 (star 92 (star 94 (star 96 (star 98 (star 100 (star 102 (star 104 (star 106 (star 108 (star 110 (star 112 (star 114 (star 116 (star 118 (star 120 (star 122 (star 124 (star 126 (star 128 (star 130 (star 132 (star 134 (star 136 (star 138 (star 140 (star 142 (star 144 (star 146 (star 148 (star 150 (star 152 (star 154 (star 156 (star 158 (star 160 (star 162 (star 164 (star 166 (star 168 (star 170 (star 172 (star 174 (star 176 (star 178 (star 180 (star 182 (star 184 (star 186 (star 188 (star 190 (star 192 (star 194 (star 196 (star 198 (star 200 (star 202 (star 204 (star 206 (star 208 (star 210 (star 212 (star 214 (star 216 (star 218 (star 220 (star 222 (star 224 (star 226 (star 228 (star 230 (star 232 (star 234 (star 236 (star 238 (star 240 (star 242 (star 244 (star 246 (star 248 (star 250 (star 252 (star 254 (star 256 (star 258 (star 260 (star 262 (star 264 (star 266 (star 268 (star 270 (star 272 (star 274 (star 276 (star 278 (star 280 (star 282 (star 284 (star 286 (star 288 (star 290 (star 292 (star 294 (star 296 (star 298 (star 300 (star 302 (star 304 (star 306 (star 308 (star 310 (star 312 (star 314 (star 316 (star 318 (star 320 (star 322 (star 324 (star 326 (star 328 (star 330 (star 332 (star 334 (star 336 (star 338 (star 340 (star 342 (star 344 (star 346 (star 348 (star 350 (star 352 (star 354 (star 356 (star 358 (star 360 (star 362 (star 364 (star 366 (star 368 (star 370 (star 372 (star 374 (star 376 (star 378 (star 380 (star 382 (star 384 (star 386 (star 388 (star 390 (star 392 (star 394 (star 396 (star 398 (star 400 (star 402 (star 404 (star 406 (star 408 (star 410 (star 412 (star 414 (star 416 (star 418 (star 420 (star 422 (star 424 (star 426 (star 428 (star 430 (star 432 (star 434 (star 436 (star 438 (star 440 (star 442 (star 444 (star 446 (star 448 (star 450 (star 452 (star 454 (star 456 (star 458 (star 460 (star 462 (star 464 (star 466 (star 468 (star 470 (star 472 (star 474 (star 476 (star 478 (star 480 (star 482 (star 484 (star 486 (star 488 (star 490 (star 492 (star 494 (star 496 (star 498 (star 500 (star 502 (star 504 (star 506 (star 508 (star 510 (star 512 (star 514 (star 516 (star 518 (star 520 (star 522 (star 524 (star 526 (star 528 (star 530 (star 532 (star 534 (star 536 (star 538 (star 540 (star 542 (star 544 (star 546 (star 548 (star 550 (star 552 (star 554 (star 556 (star 558 (star 560 (star 562 (star 564 (star 566 (star 568 (star 570 (star 572 (star 574 (star 576 (star 578 (star 580 (star 582 (star 584 (star 586 (star 588 (star 590 (star 592 (star 594 (star 596 (star 598 (star 600 (star 602 (star 604 (star 606 (star 608 (star 610 (star 612 (star 614 (star 616 (star 618 (star 620 (star 622 (star 624 (star 626 (star 628 (star 630 (star 632 (star 634 (star 636 (star 638 (star 640 (star 642 (star 644 (star 646 (star 648 (star 650 (star 652 (star 654 (star 656 (star 658 (star 660 (star 662 (star 664 (star 666 (star 668 (star 670 (star 672 (star 674 (star 676 (star 678 (star 680 (star 682 (star 684 (star 686 (star 688 (star 690 (star 692 (star 694 (star 696 (star 698 (star 700 (star 702 (star 704 (star 706 (star 708 (star 710 (star 712 (star 714 (star 716 (star 718 (star 720 (star 722 (star 724 (star 726 (star 728 (star  730 (star 732 (star 734 (star 736 (star 738 (star 740 (star 742 (star 744 (star 746 (star 748 (star 750 (star 752 (star 754 (star 756 (star 758 (star 760 (star 762 (star 764 (star 766 (star 768 (star 770 (star 772 (star 774 (star 776 (star 778 (star 780 (star 782 (star 784 (star 786 (star 788 (star 790 (star 792 (star 794 (star 796 (star 798 (star 800 (star 802 (star 804 (star 806 (star 808 (star 810 (star 812 (star 814 (star 816 (star 818 (star 820 (star 822 (star 824 (star 826 (star 828 (star 830 (star 832 (star 834 (star 836 (star 838 (star 840 (star 842 (star 844 (star 846 (star 848 (star 850 (star 852 (star 854 (star 856 (star 858 (star 860 (star 862 (star 864 (star 866 (star 868 (star 870 (star 872 (star 874 (star 876 (star 878 (star 880 (star 882 (star 884 (star 886 (star 888 (star 890 (star 892 (star 894 (star 896 (star 898 (star 900 (star 902 (star 904 (star 906 (star 908 (star 910 (star 912 (star 914 (star 916 (star 918 (star 920 (star 922 (star 924 (star 926 (star 928 (star 930 (star 932 (star 934 (star 936 (star 938 (star 940 (star 942 (star 944 (star 946 (star 948 (star 950 (star 952 (star 954 (star 956 (star 958 (star 960 (star 962 (star 964 (star 966 (star 968 (star 970 (star 972 (star 974 (star 976 (star 978 (star 980 (star 982 (star 984 (star 986 (star 988 (star 990 (star 992 (star 994 (star 996 (star 998 1000)))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))) = 2 := sorry

end compute_nested_star_l224_224700


namespace balloon_arrangements_l224_224879

noncomputable def factorial (n : ℕ) : ℕ :=
if h : 0 < n then n * factorial (n - 1) else 1

theorem balloon_arrangements : 
  ∃ (n : ℕ), n = (factorial 7) / (factorial 2 * factorial 2) ∧ n = 1260 := by
  have fact_7 := factorial 7
  have fact_2 := factorial 2
  exists (fact_7 / (fact_2 * fact_2))
  split
  · rw [← int.coe_nat_eq_coe_nat_iff]
    norm_cast
    sorry -- Here we do the exact calculations
  · exact rfl

end balloon_arrangements_l224_224879


namespace balloon_permutations_l224_224959

theorem balloon_permutations : (∀ (arrangements : ℕ),
  arrangements = nat.factorial 7 / (nat.factorial 2 * nat.factorial 2) → arrangements = 1260) :=
begin
  intros arrangements h,
  rw [nat.factorial_succ, nat.factorial],
  sorry,
end

end balloon_permutations_l224_224959


namespace domain_of_g_l224_224095

noncomputable def g (x : ℝ) : ℝ := (x + 2) / Real.sqrt (x^2 - 5 * x + 6)

theorem domain_of_g :
  {x : ℝ | x^2 - 5 * x + 6 ≥ 0} = {x : ℝ | x ≤ 2 ∨ x ≥ 3} :=
by {
  sorry
}

end domain_of_g_l224_224095


namespace engineer_last_name_is_smith_l224_224349

/-- Given these conditions:
 1. Businessman Robinson and a conductor live in Sheffield.
 2. Businessman Jones and a stoker live in Leeds.
 3. Businessman Smith and the railroad engineer live halfway between Leeds and Sheffield.
 4. The conductor’s namesake earns $10,000 a year.
 5. The engineer earns exactly 1/3 of what the businessman who lives closest to him earns.
 6. Railroad worker Smith beats the stoker at billiards.
 
We need to prove that the last name of the engineer is Smith. -/
theorem engineer_last_name_is_smith
  (lives_in_Sheffield_Robinson : Prop)
  (lives_in_Sheffield_conductor : Prop)
  (lives_in_Leeds_Jones : Prop)
  (lives_in_Leeds_stoker : Prop)
  (lives_in_halfway_Smith : Prop)
  (lives_in_halfway_engineer : Prop)
  (conductor_namesake_earns_10000 : Prop)
  (engineer_earns_one_third_closest_bizman : Prop)
  (railway_worker_Smith_beats_stoker_at_billiards : Prop) :
  (engineer_last_name = "Smith") :=
by
  -- Proof will go here
  sorry

end engineer_last_name_is_smith_l224_224349


namespace handshakes_in_tournament_l224_224444

theorem handshakes_in_tournament : 
  ∀ (teams : fin 4 → fin 2 → Type), 
  (∀ (team1 team2 : fin 4) (a b : fin 2), 
    team1 ≠ team2 → 
    ∃ (handshake : teams team1 a → teams team2 b → Prop), 
    ∀ w1 w2, handshake w1 w2 → handshake w2 w1) → 
  ∃ H : ℕ, H = 24 :=
by 
  intros teams handshake_condition,
  sorry

end handshakes_in_tournament_l224_224444


namespace kolya_is_collection_agency_l224_224218

-- Defining the conditions
def lent_books (Katya Vasya : Type) : Prop := sorry
def missed_return_date (Vasya : Type) : Prop := sorry
def agreed_to_help_retrieve (Katya Kolya Vasya : Type) : Prop := sorry
def received_reward (Kolya : Type) : Prop := sorry

-- Problem statement
theorem kolya_is_collection_agency (Katya Kolya Vasya : Type)
  (h1 : lent_books Katya Vasya) 
  (h2 : missed_return_date Vasya) 
  (h3 : agreed_to_help_retrieve Katya Kolya Vasya)
  (h4 : received_reward Kolya) : 
  ⟦Kolya's Role⟧ = "collection agency" := 
sorry

end kolya_is_collection_agency_l224_224218


namespace train_length_l224_224420

theorem train_length (t : ℝ) (s : ℝ) (L : ℝ) : 
  t = 6 ∧ s = 72 ∧ (s * 1000 / 3600) * t = L → L = 120 :=
by
  intro h
  cases h with ht hsL
  cases hsL with hs hL
  sorry

end train_length_l224_224420


namespace hyperbola_eccentricity_l224_224542

theorem hyperbola_eccentricity {a b c : ℝ} (ha : 0 < a) (hb : 0 < b) (hc : c = a * sqrt (1 + (b / a) ^ 2))
  (M_on_hyperbola : ∃ x y : ℝ, x ^ 2 / a ^ 2 - y ^ 2 / b ^ 2 = 1 ) 
  (isosceles_ABM : ∀ A B M : (ℝ × ℝ), ∠AMB = 120) :
  let e := c / a in e = ((sqrt 3 + 1) / 2) := 
by 
  sorry

end hyperbola_eccentricity_l224_224542


namespace percentage_ryegrass_X_l224_224832

/-
The goal is to prove that the percentage of ryegrass in seed mixture X is 40%, given the conditions about the ratios of bluegrass, ryegrass, and mixture contents.
-/

variable (R : ℝ) (B : ℝ) (Y_ryegrass : ℝ)
variable (mix_ryegrass : ℝ) (X_weight : ℝ)

-- Conditions identified from the problem
def conditions : Prop :=
  B = 60 / 100 ∧ Y_ryegrass = 0.25 ∧ mix_ryegrass = 0.38 ∧ X_weight = 0.8667

-- The percentage of ryegrass in seed mixture X
theorem percentage_ryegrass_X (h : conditions R B Y_ryegrass mix_ryegrass X_weight) : R = 0.40 := by
  sorry

end percentage_ryegrass_X_l224_224832


namespace unit_price_quantity_inverse_proportion_map_distance_actual_distance_direct_proportion_l224_224373

-- Definitions based on conditions
variable (unit_price quantity total_price : ℕ)
variable (map_distance actual_distance scale : ℕ)

-- Given conditions
def total_price_fixed := unit_price * quantity = total_price
def scale_fixed := map_distance * scale = actual_distance

-- Proof problem statements
theorem unit_price_quantity_inverse_proportion (h : total_price_fixed unit_price quantity total_price) :
  ∃ k : ℕ, unit_price = k / quantity := sorry

theorem map_distance_actual_distance_direct_proportion (h : scale_fixed map_distance actual_distance scale) :
  ∃ k : ℕ, map_distance * scale = k * actual_distance := sorry

end unit_price_quantity_inverse_proportion_map_distance_actual_distance_direct_proportion_l224_224373


namespace count_valid_x_l224_224498

def is_valid_x (x : ℕ) : Prop := 30 < (x + 5) * (x + 5) ∧ (x + 5) * (x + 5) < 60

theorem count_valid_x :
  {x : ℕ // is_valid_x x}.to_finset.card = 2 := by
  -- Proof omitted
  sorry

end count_valid_x_l224_224498


namespace solve_inequality_l224_224303

noncomputable def inequality_holds (x : ℝ) : Prop :=
  |4 * x^2 - 32 / x| + |x^2 + 5 / (x^2 - 6)| ≤ |3 * x^2 - 5 / (x^2 - 6) - 32 / x|

theorem solve_inequality (x : ℝ) : 
  inequality_holds x ↔ 
  (-√6 < x ∧ x ≤ -√5) ∨ (-1 ≤ x ∧ x < 0) ∨ (1 ≤ x ∧ x ≤ 2) ∨ (√5 ≤ x ∧ x < √6) :=
sorry

end solve_inequality_l224_224303


namespace Sherman_weekly_driving_time_l224_224666

theorem Sherman_weekly_driving_time (daily_commute : ℕ := 30) (weekend_drive : ℕ := 2) : 
  (5 * (2 * daily_commute) / 60 + 2 * weekend_drive) = 9 := 
by
  sorry

end Sherman_weekly_driving_time_l224_224666


namespace input_values_for_y_eq_3_l224_224323

def program_output (x : ℝ) : ℝ :=
  if x ≤ 0 then -x
  else if 0 < x ∧ x ≤ 1 then 0
  else x - 1

theorem input_values_for_y_eq_3 (x : ℝ) :
  program_output x = 3 ↔ x = -3 ∨ x = 4 := sorry

end input_values_for_y_eq_3_l224_224323


namespace probability_consecutive_computer_scientists_l224_224724

theorem probability_consecutive_computer_scientists :
  let n := 12
  let k := 5
  let total_permutations := Nat.factorial (n - 1)
  let consecutive_permutations := Nat.factorial (7) * Nat.factorial (5)
  let probability := consecutive_permutations / total_permutations
  probability = (1 / 66) :=
by
  sorry

end probability_consecutive_computer_scientists_l224_224724


namespace angle_AKC_is_90_l224_224660

-- Define the structures and assumptions
variables (M N K A B C : Type)
variables [add_comm_group A] [add_comm_group B] [add_comm_group C]
variables (m : is_midpoint M A C)
variables (n : is_symmetric N M BC)
variables (k_parallel : is_parallel (line_through ↥N) ↥A ↥C)

-- Define the proof target
theorem angle_AKC_is_90 ( 
    m : is_midpoint M A C, 
    n : is_symmetric N M BC,
    k_parallel : is_parallel (line_through N) A C,
    k_intersects : exists (K : Type), point_on_line K AB 
): angle_AKC = 90 :=
sorry

end angle_AKC_is_90_l224_224660


namespace circumscribed_sphere_radius_l224_224124

open Real -- Use the Real number library for calculations

theorem circumscribed_sphere_radius (n : ℕ) (h a : ℝ) (hn_pos : 0 < n) (ha_pos : 0 < a) (hh_pos : 0 < h) :
  let R := 1 / 2 * sqrt ((a^2 / (sin (180 / n * π / 180))^2) + h^2) in
  R = 1 / 2 * sqrt ((a^2 / (sin (180 / n * π / 180))^2) + h^2) :=
by
  sorry

end circumscribed_sphere_radius_l224_224124


namespace perpendicular_distance_is_8_cm_l224_224836

theorem perpendicular_distance_is_8_cm :
  ∀ (side_length distance_from_corner cut_angle : ℝ),
    side_length = 100 →
    distance_from_corner = 8 →
    cut_angle = 45 →
    (∃ h : ℝ, h = 8) :=
by
  intros side_length distance_from_corner cut_angle hms d8 a45
  sorry

end perpendicular_distance_is_8_cm_l224_224836


namespace balloon_permutation_count_l224_224918

-- Define the necessary factorials and the formula for permutations of multiset
def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem balloon_permutation_count : 
  let total_letters := 7,
      repeat_L := 2,
      repeat_O := 2,
      unique_arrangements := factorial total_letters / (factorial repeat_L * factorial repeat_O) 
  in unique_arrangements = 1260 :=
by
  sorry

end balloon_permutation_count_l224_224918


namespace repeating_decimal_as_fraction_l224_224749

theorem repeating_decimal_as_fraction :
  let x := 56 / 99
  in x = 0.56 + 0.0056 + 0.000056 + (0.00000056) : ℚ :=
by
  sorry

end repeating_decimal_as_fraction_l224_224749


namespace savings_increase_100_percent_l224_224827

variables {I : ℝ} (S : ℝ := 0.30 * I) (E1 : ℝ := I - S) (I2 : ℝ := 1.30 * I) (E2 : ℝ)
          {S2 : ℝ}  {percentage_increase : ℝ}

-- Conditions
axiom savings_first_year : S = 0.30 * I
axiom expenditure_first_year : E1 = 0.70 * I
axiom income_second_year : I2 = 1.30 * I
axiom total_expenditure : E1 + E2 = 2 * E1

-- Defining the second year's savings and expenditure based on the given conditions
axiom expenditure_second_year : E2 = 0.70 * I
axiom savings_second_year : S2 = I2 - E2
axiom percentage_increase_def : percentage_increase = (S2 - S) / S * 100

-- Theorem: The man's savings increase by 100% in the second year
theorem savings_increase_100_percent : percentage_increase = 100 := by
  rw [savings_first_year, expenditure_first_year, income_second_year, expenditure_second_year,
      savings_second_year, percentage_increase_def]
  sorry

end savings_increase_100_percent_l224_224827


namespace balloon_permutation_count_l224_224912

-- Define the necessary factorials and the formula for permutations of multiset
def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem balloon_permutation_count : 
  let total_letters := 7,
      repeat_L := 2,
      repeat_O := 2,
      unique_arrangements := factorial total_letters / (factorial repeat_L * factorial repeat_O) 
  in unique_arrangements = 1260 :=
by
  sorry

end balloon_permutation_count_l224_224912


namespace area_of_triangle_ABC_l224_224721

-- Given definitions
def radius : ℝ := 4
def diameter : ℝ := 2 * radius
def AM_ratio : ℝ := 2
def MB_ratio : ℝ := 3
def θ : ℝ := 30  -- Angle formed between AB and the diameter, in degrees
def angle_BMC : ℝ := 90 -- Angle formed as BC is perpendicular to diameter at B

-- Lean statement to be proved
theorem area_of_triangle_ABC :
∃ (A B C M : Point) (circle : Circle) (AB BC : LineSegment),
    circle.radius = radius
    ∧ (A, M, B are_collinear) 
    ∧ (M is mid_point_of_the_diameter_of circle)
    ∧ (meets_at_angle (A, B, M) θ) 
    ∧ (meets_at_right_angle (B, C, M))
    ∧ (ratio_of_segments (AM, MB) = AM_ratio / MB_ratio)
    → area ⟨A, B, C⟩ = \frac{180 \sqrt{3}}{19} :=
sorry

end area_of_triangle_ABC_l224_224721


namespace find_speed_of_car_A_find_distance_between_A_and_B_l224_224359

-- Definition for the first proof problem
def speed_of_car_A (t1 t2 t total_distance speed_C : ℝ) : ℝ :=
  let time_difference := t + t2 in
  (total_distance / time_difference)

-- Statement for the first proof problem
theorem find_speed_of_car_A (speed_C : ℝ) (t1 : ℝ) (t2 : ℝ) (total_distance : ℝ) :
  speed_of_car_A t1 t2 t2 total_distance speed_C = 40 :=
by sorry

-- Definition for the second proof problem
def distance_between_A_and_B (speed_A speed_B : ℝ) (time_interval : ℝ) (additional_distance : ℝ) : ℝ :=
  (speed_A + speed_B) * time_interval + additional_distance

-- Statement for the second proof problem
theorem find_distance_between_A_and_B (speed_A : ℝ) (speed_B : ℝ) (time_interval : ℝ) (additional_distance : ℝ):
  distance_between_A_and_B speed_A speed_B time_interval additional_distance = 40 :=
by sorry

end find_speed_of_car_A_find_distance_between_A_and_B_l224_224359


namespace recurring_decimal_to_fraction_correct_l224_224758

noncomputable def recurring_decimal_to_fraction (b : ℚ) : Prop :=
  b = 0.\overline{56} ↔ b = 56/99

theorem recurring_decimal_to_fraction_correct : recurring_decimal_to_fraction 0.\overline{56} :=
  sorry

end recurring_decimal_to_fraction_correct_l224_224758


namespace area_ratio_of_triangles_l224_224723

theorem area_ratio_of_triangles
  (ABCD_is_isosceles : is_isosceles_trapezoid ABCD)
  (AD_parallel_BC : AD ∥ BC)
  (EAD_is_equilateral : is_equilateral_triangle E A D)
  (EF_parallel_AD : EF ∥ AD) 
  (FG_parallel_AD : FG ∥ AD)
  (GH_parallel_AD : GH ∥ AD)
  (equal_segments : AB = BE ∧ BE = EF ∧ EF = FG ∧ FG = GH) :
  (area_of_triangle F G H) / (area_of_triangle E A D) = 1 / 4 := sorry

end area_ratio_of_triangles_l224_224723


namespace john_makes_money_l224_224233

variables 
  (jars : ℕ) 
  (caterpillars_per_jar : ℕ) 
  (failure_rate : ℝ) 
  (price_per_butterfly : ℝ)

noncomputable def totalCaterpillars := jars * caterpillars_per_jar
noncomputable def failedCaterpillars := failure_rate * totalCaterpillars
noncomputable def successfulButterflies := totalCaterpillars - failedCaterpillars
noncomputable def totalMoney := successfulButterflies * price_per_butterfly

theorem john_makes_money (h1 : jars = 4) (h2 : caterpillars_per_jar = 10) (h3 : failure_rate = 0.40) (h4 : price_per_butterfly = 3) :
  totalMoney jars caterpillars_per_jar failure_rate price_per_butterfly = 72 :=
by
  sorry

end john_makes_money_l224_224233


namespace first_nonzero_digit_one_div_139_l224_224365

theorem first_nonzero_digit_one_div_139 :
  ∀ n : ℕ, (n > 0 → (∀ m : ℕ, (m > 0 → (m * 10^n) ∣ (10^n * 1 - 1) ∧ n ∣ (139 * 10 ^ (n + 1)) ∧ 10^(n+1 - 1) * 1 - 1 < 10^n))) :=
sorry

end first_nonzero_digit_one_div_139_l224_224365


namespace repeating_decimal_sum_l224_224466

noncomputable def repeating_decimal_four : ℚ := 0.44444 -- 0.\overline{4}
noncomputable def repeating_decimal_seven : ℚ := 0.77777 -- 0.\overline{7}

-- Proving that the sum of these repeating decimals is equivalent to the fraction 11/9.
theorem repeating_decimal_sum : repeating_decimal_four + repeating_decimal_seven = 11/9 := by
  -- Placeholder to skip the actual proof
  sorry

end repeating_decimal_sum_l224_224466


namespace find_k_l224_224477

theorem find_k (k : ℚ) : 
  (∃ q r : ℚ[X], (3 * X + 4) * q + r = 3 * X^3 + k * X^2 + 5 * X - 8 ∧ degree r < degree (3 * X + 4)) → r = 10 := 
sorry

end find_k_l224_224477


namespace point_on_x_axis_l224_224659

theorem point_on_x_axis (m : ℝ) (h : m - 2 = 0) :
  (m + 3, m - 2) = (5, 0) :=
by
  sorry

end point_on_x_axis_l224_224659


namespace mouse_jump_l224_224325

theorem mouse_jump : ∃ (F : ℤ), 
  let G := F + 19 in 
  let M := F - 12 in 
  G = 39 ∧ M = 8 :=
by
  sorry

end mouse_jump_l224_224325


namespace complement_union_l224_224260

open Set

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def A : Set ℕ := {4, 5}
def B : Set ℕ := {3, 4}

theorem complement_union : (U \ (A ∪ B)) = {1, 2, 6} :=
by simp only [U, A, B, Set.mem_union, Set.mem_compl, Set.mem_diff];
   sorry

end complement_union_l224_224260


namespace exists_points_with_irrational_distances_and_rational_areas_l224_224635

theorem exists_points_with_irrational_distances_and_rational_areas (n : ℕ) (hn : n ≥ 3) :
  ∃ S : Fin n → (ℚ × ℚ), 
    (∀ i j : Fin n, i ≠ j → irrational (real.sqrt (((S i).fst - (S j).fst)^2 + ((S i).snd - (S j).snd)^2))) ∧
    (∀ i j k : Fin n, 
     i ≠ j ∧ j ≠ k ∧ i ≠ k →
     ∃ T : ℚ, 2 * T = abs ((S i).fst * ((S j).snd - (S k).snd) + (S j).fst * ((S k).snd - (S i).snd) + (S k).fst * ((S i).snd - (S j).snd))) :=
sorry

end exists_points_with_irrational_distances_and_rational_areas_l224_224635


namespace intersection_unique_ratio_S1_S2_l224_224442

variable {a : ℝ} (h_a_pos : a > 0)
def point_A := (-2 * a^2, 0)
def point_A' := (2 * a^2, 0)
def parabola (y : ℝ) : set (ℝ × ℝ) := {p | p.2^2 = 2 * p.1}

def point_B := (2 * a^2, 2 * a)
def point_C := (2 * a^2, -2 * a)

structure Point_on_AB (B : ℝ × ℝ) (A : ℝ × ℝ) :=
  (x : ℝ)
  (y : ℝ)
  (on_AB : ∃ λ : ℝ, 0 ≤ λ ∧ λ ≤ 1 ∧ (x, y) = (A.1 + λ * (B.1 - A.1), A.2 + λ * (B.2 - A.2)))

structure Point_on_AC (C : ℝ × ℝ) (A : ℝ × ℝ) :=
  (x : ℝ)
  (y : ℝ)
  (on_AC : ∃ λ : ℝ, 0 ≤ λ ∧ λ ≤ 1 ∧ (x, y) = (A.1 + λ * (C.1 - A.1), A.2 + λ * (C.2 - A.2)))

variable {D : Point_on_AB point_B point_A}
variable {E : Point_on_AC point_C point_A}
variable (ratio_CE_CA_AD_AB : |E.y - point_C.snd| / |point_A.snd - point_C.snd| = |D.y - point_A.snd| / |point_B.snd - point_A.snd|)

theorem intersection_unique : ∃ F ∈ (parabola F.2), ((E.y - D.y) / (E.x - D.x)) = ((F.2 - D.y) / (F.1 - D.x)) := sorry

theorem ratio_S1_S2 (S1 S2 : ℝ) :
  S1 = |(point_B.1 - point_C.1) * 2 * a * (a - a)| / 2 ∧
  S2 = 2 * (a^2.2 - (2*a*E.1 / D.y) * (2*a*point_A.1.2)) / 2  →
  S1 / S2 = 2 := sorry

end intersection_unique_ratio_S1_S2_l224_224442


namespace balloon_arrangements_l224_224984

def factorial (n : Nat) : Nat :=
  if n = 0 then 1 else n * factorial (n - 1)

def arrangements (n : Nat) (ks : List Nat) :=
  factorial n / (ks.map factorial).foldr (*) 1

theorem balloon_arrangements : arrangements 7 [2, 2] = 1260 :=
by
  sorry

end balloon_arrangements_l224_224984


namespace two_tetrahedrons_exist_l224_224001

-- Given segments denoted as AA1, BB1, and CC1
variables {A A1 B B1 C C1 O : Type} [point A] [point A1] [point B] [point B1] [point C] [point C1] [point O]
variables (AA1 BB1 CC1 : Segment A A1 × Segment B B1 × Segment C C1)
variables (halved_at_O : ∀ (seg : Segment), midpoint seg = O)
variables (not_in_same_plane : ¬ coplanar A A1 B B1 C C1)

-- Define the condition where three segments intersect at one point O
def segments_intersect_at_O := 
  (intersect AA1 BB1 = O) ∧ (intersect BB1 CC1 = O) ∧ (intersect CC1 AA1 = O)

-- Theorem statement
theorem two_tetrahedrons_exist (h_intersects : segments_intersect_at_O AA1 BB1 CC1)
                              (h_halved : halved_at_O AA1 ∧ halved_at_O BB1 ∧ halved_at_O CC1)
                              (h_non_coplanar : not_in_same_plane):
  ∃ (T1 T2 : Tetrahedron), connects_midpoints_of_opposite_edges T1 AA1 BB1 CC1 ∧ connects_midpoints_of_opposite_edges T2 AA1 BB1 CC1 ∧ T1 ≠ T2 :=
sorry

end two_tetrahedrons_exist_l224_224001


namespace odds_of_picking_blue_marble_l224_224450

theorem odds_of_picking_blue_marble :
  ∀ (total_marbles yellow_marbles : ℕ)
  (h1 : total_marbles = 60)
  (h2 : yellow_marbles = 20)
  (green_marbles : ℕ)
  (h3 : green_marbles = yellow_marbles / 2)
  (remaining_marbles : ℕ)
  (h4 : remaining_marbles = total_marbles - yellow_marbles - green_marbles)
  (blue_marbles : ℕ)
  (h5 : blue_marbles = remaining_marbles / 2),
  (blue_marbles / total_marbles : ℚ) * 100 = 25 :=
by
  intros total_marbles yellow_marbles h1 h2 green_marbles h3 remaining_marbles h4 blue_marbles h5
  sorry

end odds_of_picking_blue_marble_l224_224450


namespace find_a_of_imaginary_division_l224_224637

theorem find_a_of_imaginary_division (a : ℝ) (ha : (a + 2 * complex.I) / (1 + complex.I) = (a + 2 - a * complex.I) / 2) : a = 2 :=
sorry

end find_a_of_imaginary_division_l224_224637


namespace candy_eaten_total_l224_224190

noncomputable def totalCandyEaten (initial: Nat) (firstDayPercent: Float) (secondDay: Nat) (thirdDayFraction: Float) : Nat :=
  let firstDay := (firstDayPercent * initial.to_float).to_nat
  let remainingAfterFirstDay := initial - firstDay
  let remainingAfterSecondDay := remainingAfterFirstDay - secondDay
  let thirdDay := (thirdDayFraction * remainingAfterSecondDay.to_float).to_nat
  firstDay + secondDay + thirdDay

theorem candy_eaten_total : totalCandyEaten 1200 0.25 300 0.25 = 750 := 
by
  sorry

end candy_eaten_total_l224_224190


namespace transformed_page_units_digit_count_l224_224084

theorem transformed_page_units_digit_count : 
  (finset.filter (λ x : ℕ, (x % 10 = (68 - x) % 10)) (finset.range 66)).card = 13 := 
by
  sorry

end transformed_page_units_digit_count_l224_224084


namespace total_spent_l224_224404

theorem total_spent (bracelet_price keychain_price coloring_book_price : ℕ)
  (paula_bracelets paula_keychains olive_coloring_books olive_bracelets : ℕ)
  (total : ℕ) :
  bracelet_price = 4 →
  keychain_price = 5 →
  coloring_book_price = 3 →
  paula_bracelets = 2 →
  paula_keychains = 1 →
  olive_coloring_books = 1 →
  olive_bracelets = 1 →
  total = paula_bracelets * bracelet_price + paula_keychains * keychain_price +
          olive_coloring_books * coloring_book_price + olive_bracelets * bracelet_price →
  total = 20 :=
by sorry

end total_spent_l224_224404


namespace intersection_empty_l224_224255

open Set

-- Definition of set A
def A : Set (ℝ × ℝ) := { p | ∃ x : ℝ, p = (x, 2 * x + 3) }

-- Definition of set B
def B : Set (ℝ × ℝ) := { p | ∃ x : ℝ, p = (x, 4 * x + 1) }

-- The proof problem statement in Lean
theorem intersection_empty : A ∩ B = ∅ := sorry

end intersection_empty_l224_224255


namespace missy_total_watching_time_l224_224265

def num_reality_shows := 5
def length_reality_show := 28
def num_cartoons := 1
def length_cartoon := 10

theorem missy_total_watching_time : 
  (num_reality_shows * length_reality_show + num_cartoons * length_cartoon) = 150 := 
by 
  sorry

end missy_total_watching_time_l224_224265


namespace find_t_l224_224383

-- Given conditions 
variables (p j t : ℝ)

-- Condition 1: j is 25% less than p
def condition1 : Prop := j = 0.75 * p

-- Condition 2: j is 20% less than t
def condition2 : Prop := j = 0.80 * t

-- Condition 3: t is t% less than p
def condition3 : Prop := t = p * (1 - t / 100)

-- Final proof statement
theorem find_t (h1 : condition1 p j) (h2 : condition2 j t) (h3 : condition3 p t) : t = 6.25 :=
sorry

end find_t_l224_224383


namespace find_unknown_number_l224_224384

theorem find_unknown_number (x : ℕ) (h₁ : (20 + 40 + 60) / 3 = 5 + (10 + 50 + x) / 3) : x = 45 :=
by sorry

end find_unknown_number_l224_224384


namespace repeating_decimal_eq_l224_224766

noncomputable def repeating_decimal : ℚ := 56 / 99

theorem repeating_decimal_eq : ∃ x : ℚ, x = repeating_decimal ∧ x = 56 / 99 :=
by
  use 56 / 99
  split
  all_goals { sorry }

end repeating_decimal_eq_l224_224766


namespace only_valid_polynomials_l224_224999

-- Define the given conditions
def P_satisfies_condition (P : ℝ → ℝ) : Prop :=
  ∃ (infinitely_many_pairs : ∃ᶠ (m n : ℕ) in (atTop × atTop), Nat.coprime m n ∧ P (m / n) = 1 / n)

-- Define the polynomials of the form x / k where k is a positive integer
def is_valid_polynomial (P : ℝ → ℝ) : Prop :=
  ∃ (k : ℕ), k > 0 ∧ ∀ x : ℝ, P x = x / k

theorem only_valid_polynomials (P : ℝ → ℝ) :
  P_satisfies_condition P ↔ is_valid_polynomial P :=
sorry

end only_valid_polynomials_l224_224999


namespace balloon_arrangements_l224_224886

noncomputable def factorial (n : ℕ) : ℕ :=
if h : 0 < n then n * factorial (n - 1) else 1

theorem balloon_arrangements : 
  ∃ (n : ℕ), n = (factorial 7) / (factorial 2 * factorial 2) ∧ n = 1260 := by
  have fact_7 := factorial 7
  have fact_2 := factorial 2
  exists (fact_7 / (fact_2 * fact_2))
  split
  · rw [← int.coe_nat_eq_coe_nat_iff]
    norm_cast
    sorry -- Here we do the exact calculations
  · exact rfl

end balloon_arrangements_l224_224886


namespace maximum_b_value_l224_224826

def not_lattice_point (m : ℚ) (x : ℤ) : Prop :=
  ∀ (y : ℤ), y ≠ m * x + 3

def no_lattice_points (m : ℚ) : Prop :=
  ∀ x : ℤ, 1 < x ∧ x ≤ 150 → not_lattice_point m x

theorem maximum_b_value :
  ∃ b : ℚ, (∀ m : ℚ, (1/3 : ℚ) < m ∧ m < b → no_lattice_points m) ∧ b = 52/151 := 
sorry

end maximum_b_value_l224_224826


namespace number_of_lattice_points_in_triangle_l224_224280

theorem number_of_lattice_points_in_triangle (N L S : ℕ) (A B O : (ℕ × ℕ)) :
  (A = (0, 30)) →
  (B = (20, 10)) →
  (O = (0, 0)) →
  (S = 300) →
  (L = 60) →
  S = N + L / 2 - 1 →
  N = 271 :=
by
  intros hA hB hO hS hL hPick
  sorry

end number_of_lattice_points_in_triangle_l224_224280


namespace tank_leak_time_l224_224790

/--
The rate at which the tank is filled without a leak is R = 1/5 tank per hour.
The effective rate with the leak is 1/6 tank per hour.
Prove that the time it takes for the leak to empty the full tank is 30 hours.
-/
theorem tank_leak_time (R : ℝ) (L : ℝ) (h1 : R = 1 / 5) (h2 : R - L = 1 / 6) :
  1 / L = 30 :=
by
  sorry

end tank_leak_time_l224_224790


namespace nh_angle_bisector_of_mnc_l224_224032

structure Triangle (P Q R : Type) :=
(eq_ab_bc : P = Q)
(eq_ac_cm : Q = R)

variable {P Q R M N H : Type}

def altitude_intercepts (triangle : Triangle P Q R) : Prop :=
  exists (altitude : P → Q → R → Type) (intersection : Type) 
  (h_property : altitude P Q (triangle.eq_ab_bc) = intersection),
  M = altitude P Q (triangle.eq_ab_bc)

def bisector_of_angle (triangle : Triangle P Q R) : Prop :=
  ∀ {bisector : P → Q → R → Type},
  N = bisector P (triangle.eq_ac_cm) R → true 

theorem nh_angle_bisector_of_mnc 
(triangle : Triangle P Q R)
(altitude_condition : altitude_intercepts triangle)
: bisector_of_angle triangle :=
by
  sorry

end nh_angle_bisector_of_mnc_l224_224032


namespace abs_value_inequality_solutions_l224_224301

noncomputable def a (x : ℝ) : ℝ := 4 * x^2 - 32 / x
noncomputable def b (x : ℝ) : ℝ := x^2 + 5 / (x^2 - 6)
noncomputable def c (x : ℝ) : ℝ := 3 * x^2 - 32 / x - 5 / (x^2 - 6)

theorem abs_value_inequality_solutions :
  { x : ℝ | ∥a x - b x∥ ≤ ∥a x∥ + ∥b x∥ → (a x) * (b x) ≤ 0 } ∩
  { x : ℝ | (x > -sqrt (6) ∧ x <= -sqrt (5)) ∨
            (x >= -1 ∧ x < 0) ∨
            (x >= 1 ∧ x <= 2) ∨
            (x >= sqrt(5) ∧ x < sqrt(6))} :=
begin
  sorry
end

end abs_value_inequality_solutions_l224_224301


namespace repeating_decimal_fraction_l224_224779

theorem repeating_decimal_fraction : ∀ x : ℚ, (x = 0.5656565656565656) → 100 * x = 56.5656565656565656 → 100 * x - x = 56.5656565656565656 - 0.5656565656565656
  → 99 * x = 56 → x = 56 / 99 :=
begin
  intros x h1 h2 h3 h4,
  sorry,
end

end repeating_decimal_fraction_l224_224779


namespace factor_expression_l224_224470

theorem factor_expression (x : ℕ) : 63 * x + 54 = 9 * (7 * x + 6) :=
by
  sorry

end factor_expression_l224_224470


namespace circle_coordinates_correct_l224_224845

theorem circle_coordinates_correct :
  ∀ (r : List ℝ), r = [1.5, 2, 3.5, 4.5, 5.5] →
  let points := List.map (λ r, (2 * real.pi * r, real.pi * r^2)) r in
  points = [(3 * real.pi, 2.25 * real.pi), (4 * real.pi, 4 * real.pi), (7 * real.pi, 12.25 * real.pi), (9 * real.pi, 20.25 * real.pi), (11 * real.pi, 30.25 * real.pi)] :=
by
  intro r hr
  rw hr
  let points := List.map (λ r, (2 * real.pi * r, real.pi * r^2)) [1.5, 2, 3.5, 4.5, 5.5]
  have h : points = [(3 * real.pi, 2.25 * real.pi), (4 * real.pi, 4 * real.pi), (7 * real.pi, 12.25 * real.pi), (9 * real.pi, 20.25 * real.pi), (11 * real.pi, 30.25 * real.pi)] := sorry
  exact h

end circle_coordinates_correct_l224_224845


namespace flour_per_batch_l224_224025

def initial_flour := 1.5 -- kg
def water := 5.0 -- kg
def required_ratio := 3.0 / 2.0

theorem flour_per_batch (added_flour_per_batch: ℝ):
  let total_flour := initial_flour + 3 * added_flour_per_batch in
  let ratio := total_flour / water in
  ratio = required_ratio → added_flour_per_batch = 2 :=
sorry

end flour_per_batch_l224_224025


namespace find_x_l224_224179

theorem find_x (x : ℚ) (h : (3 * x - 6 + 4) / 7 = 15) : x = 107 / 3 :=
by
  sorry

end find_x_l224_224179


namespace ways_to_distribute_teachers_l224_224854

-- Definitions and conditions directly from the problem.
def Teacher := ℕ -- Using natural numbers to index the teachers (A, B, C, and D).
def School := ℕ  -- Using natural numbers to index the schools.

def distribution_valid (dist : Teacher → School) : Prop :=
  (∀ t1 t2, t1 ≠ t2 → dist t1 ≠ dist t2) ∧
  ((∃ t1, dist t1 = 0) ∧ (∃ t2, dist t2 = 1) ∧ (∃ t3, dist t3 = 2)) ∧
  dist 0 ≠ dist 1 -- A ≠ B

theorem ways_to_distribute_teachers : ∃ (dist : (Teacher → School)), distribution_valid dist → 30 :=
sorry

end ways_to_distribute_teachers_l224_224854


namespace salary_proof_l224_224352

-- Defining the monthly salaries of the officials
def D_Dupon : ℕ := 6000
def D_Duran : ℕ := 8000
def D_Marten : ℕ := 5000

-- Defining the statements made by each official
def Dupon_statement1 : Prop := D_Dupon = 6000
def Dupon_statement2 : Prop := D_Duran = D_Dupon + 2000
def Dupon_statement3 : Prop := D_Marten = D_Dupon - 1000

def Duran_statement1 : Prop := D_Duran > D_Marten
def Duran_statement2 : Prop := D_Duran - D_Marten = 3000
def Duran_statement3 : Prop := D_Marten = 9000

def Marten_statement1 : Prop := D_Marten < D_Dupon
def Marten_statement2 : Prop := D_Dupon = 7000
def Marten_statement3 : Prop := D_Duran = D_Dupon + 3000

-- Defining the constraints about the number of truth and lies
def Told_the_truth_twice_and_lied_once : Prop :=
  (Dupon_statement1 ∧ Dupon_statement2 ∧ ¬Dupon_statement3) ∨
  (Dupon_statement1 ∧ ¬Dupon_statement2 ∧ Dupon_statement3) ∨
  (¬Dupon_statement1 ∧ Dupon_statement2 ∧ Dupon_statement3) ∨
  (Duran_statement1 ∧ Duran_statement2 ∧ ¬Duran_statement3) ∨
  (Duran_statement1 ∧ ¬Duran_statement2 ∧ Duran_statement3) ∨
  (¬Duran_statement1 ∧ Duran_statement2 ∧ Duran_statement3) ∨
  (Marten_statement1 ∧ Marten_statement2 ∧ ¬Marten_statement3) ∨
  (Marten_statement1 ∧ ¬Marten_statement2 ∧ Marten_statement3) ∨
  (¬Marten_statement1 ∧ Marten_statement2 ∧ Marten_statement3)

-- The final proof goal
theorem salary_proof : Told_the_truth_twice_and_lied_once →
  D_Dupon = 6000 ∧ D_Duran = 8000 ∧ D_Marten = 5000 := by 
  sorry

end salary_proof_l224_224352


namespace probability_matching_shoes_l224_224789

theorem probability_matching_shoes :
  let total_shoes := 24;
  let total_pairs := 12;
  let total_combinations := (total_shoes * (total_shoes - 1)) / 2;
  let matching_pairs := total_pairs;
  let probability := matching_pairs / total_combinations;
  probability = 1 / 23 :=
by
  let total_shoes := 24
  let total_pairs := 12
  let total_combinations := (total_shoes * (total_shoes - 1)) / 2
  let matching_pairs := total_pairs
  let probability := matching_pairs / total_combinations
  have : total_combinations = 276 := by norm_num
  have : matching_pairs = 12 := by norm_num
  have : probability = 1 / 23 := by norm_num
  exact this

end probability_matching_shoes_l224_224789


namespace contrapositive_even_contrapositive_not_even_l224_224312

theorem contrapositive_even (x y : ℤ) : 
  (∃ a b : ℤ, x = 2*a ∧ y = 2*b)  → (∃ c : ℤ, x + y = 2*c) :=
sorry

theorem contrapositive_not_even (x y : ℤ) :
  (¬ ∃ c : ℤ, x + y = 2*c) → (¬ ∃ a b : ℤ, x = 2*a ∧ y = 2*b) :=
sorry

end contrapositive_even_contrapositive_not_even_l224_224312


namespace non_congruent_triangles_l224_224593

-- Definition of points and isosceles property
variable (A B C P Q R : Type)
variable [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited P] [Inhabited Q] [Inhabited R]

-- Conditions of the problem
def is_isosceles (A B C : Type) : Prop := (A = B) ∧ (A = C)
def is_midpoint (P Q R : Type) (A B C : Type) : Prop := sorry -- precise formal definition omitted for brevity

-- Theorem stating the final result
theorem non_congruent_triangles (A B C P Q R : Type)
  (h_iso : is_isosceles A B C)
  (h_midpoints : is_midpoint P Q R A B C) :
  ∃ (n : ℕ), n = 4 := 
  by 
    -- proof abbreviated
    sorry

end non_congruent_triangles_l224_224593


namespace balloon_permutations_count_l224_224911

-- Definitions of the conditions
def total_letters_count : ℕ := 7
def l_count : ℕ := 2
def o_count : ℕ := 2

-- Now the mathematical problem as a Lean statement
theorem balloon_permutations_count : 
  (Nat.factorial total_letters_count) / ((Nat.factorial l_count) * (Nat.factorial o_count)) = 1260 := 
by
  sorry

end balloon_permutations_count_l224_224911


namespace correct_subsidy_equation_l224_224679

-- Define the necessary variables and conditions
def sales_price (x : ℝ) := x  -- sales price of the mobile phone in yuan
def subsidy_rate : ℝ := 0.13  -- 13% subsidy rate
def number_of_phones : ℝ := 20  -- 20 units sold
def total_subsidy : ℝ := 2340  -- total subsidy provided

-- Lean theorem statement to prove the correct equation
theorem correct_subsidy_equation (x : ℝ) :
  number_of_phones * x * subsidy_rate = total_subsidy :=
by
  sorry -- proof to be completed

end correct_subsidy_equation_l224_224679


namespace determine_parallel_planes_l224_224037

def Plane : Type := sorry
def Line : Type := sorry
def Parallel (x y : Line) : Prop := sorry
def Skew (x y : Line) : Prop := sorry
def PlaneParallel (α β : Plane) : Prop := sorry

variables (α β : Plane) (a b : Line)
variable (hSkew : Skew a b)
variable (hαa : Parallel a α) 
variable (hαb : Parallel b α)
variable (hβa : Parallel a β)
variable (hβb : Parallel b β)

theorem determine_parallel_planes : PlaneParallel α β := sorry

end determine_parallel_planes_l224_224037


namespace positive_integer_solutions_of_inequality_system_l224_224300

theorem positive_integer_solutions_of_inequality_system :
  {x : ℤ | 2 * (x - 1) < x + 1 ∧ 1 - (2 * x + 5) / 3 ≤ x ∧ x > 0} = {1, 2} :=
by
  sorry

end positive_integer_solutions_of_inequality_system_l224_224300


namespace hashtag_3_8_l224_224181

-- Define the hashtag operation
def hashtag (a b : ℤ) : ℤ := a * b - b + b ^ 2

-- Prove that 3 # 8 equals 80
theorem hashtag_3_8 : hashtag 3 8 = 80 := by
  sorry

end hashtag_3_8_l224_224181


namespace moment_of_inertia_equals_sum_of_squares_l224_224362

structure Tetrahedron :=
  (A B C D : ℝ × ℝ × ℝ)  -- Coordinates of the vertices

def center_of_mass (T : Tetrahedron) : ℝ × ℝ × ℝ :=
  let (A, B, C, D) := (T.A, T.B, T.C, T.D)
  ((A.1 + B.1 + C.1 + D.1) / 4, 
   (A.2 + B.2 + C.2 + D.2) / 4, 
   (A.3 + B.3 + C.3 + D.3) / 4)

def distance (p1 p2 : ℝ × ℝ × ℝ) : ℝ :=
  real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2 + (p1.3 - p2.3)^2)

noncomputable def moment_of_inertia (T : Tetrahedron) : ℝ :=
  let O := center_of_mass T
  let m := 1 -- unit mass
  m * ((distance O T.A)^2 + (distance O T.B)^2 + (distance O T.C)^2 + (distance O T.D)^2)

def midpoint (p1 p2 : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2, (p1.3 + p2.3) / 2)

def sum_of_squares_of_midpoints_distances (T : Tetrahedron) : ℝ :=
  let (A, B, C, D) := (T.A, T.B, T.C, T.D)
  (distance (midpoint A B) (midpoint C D))^2 +
  (distance (midpoint A C) (midpoint B D))^2 +
  (distance (midpoint A D) (midpoint B C))^2

theorem moment_of_inertia_equals_sum_of_squares (T : Tetrahedron) :
  moment_of_inertia T = sum_of_squares_of_midpoints_distances T :=
sorry

end moment_of_inertia_equals_sum_of_squares_l224_224362


namespace unique_solution_for_4_circ_20_l224_224093

def operation (x y : ℝ) : ℝ := 3 * x - 2 * y + 2 * x * y

theorem unique_solution_for_4_circ_20 : ∃! y : ℝ, operation 4 y = 20 :=
by 
  sorry

end unique_solution_for_4_circ_20_l224_224093


namespace balloon_arrangement_count_l224_224969

-- Definitions of letter frequencies for the word BALLOON
def letter_frequencies : List (Char × Nat) := [('B', 1), ('A', 1), ('L', 2), ('O', 2), ('N', 1)]

-- Total number of letters
def total_letters := 7

-- The formula for the number of unique arrangements of the letters
noncomputable def arrangements :=
  (Nat.factorial total_letters) / 
  (Nat.factorial 1 * Nat.factorial 1 * Nat.factorial 2 * Nat.factorial 2 * Nat.factorial 1)

-- The theorem to prove the number of ways to arrange the letters in "BALLOON"
theorem balloon_arrangement_count : arrangements = 1260 :=
  sorry

end balloon_arrangement_count_l224_224969


namespace prob_exactly_two_dice_show_four_l224_224109

-- Given condition: We roll fifteen 6-sided dice.
-- Question: Prove the probability that exactly two of the dice show a 4 is approximately 0.231.

theorem prob_exactly_two_dice_show_four : 
  let n := 15
  let p := (1 / 6 : ℚ)
  let q := (5 / 6 : ℚ)
  let probability := (nat.choose n 2) * p^2 * q^13
  probability ≈ 0.231 :=
by 
  --The exact calculations leading to probability ≈ 0.231
  sorry

end prob_exactly_two_dice_show_four_l224_224109


namespace triangle_sides_and_angles_l224_224580

noncomputable def side_b (a c : ℝ) (cos_B : ℝ) : ℝ :=
  real.sqrt (a^2 + c^2 - 2 * a * c * cos_B)

noncomputable def sin_C (a b c : ℝ) (cos_B : ℝ) : ℝ :=
  let sin_B := real.sqrt (1 - cos_B^2) in
  let sin_C := c * sin_B / b in
  sin_C

theorem triangle_sides_and_angles (a b c : ℝ) (cos_B : ℝ)
  (ha : a = 2) (hc : c = 5) (hcosB : cos_B = 3 / 5) :
  b = real.sqrt 17 ∧ sin_C a b c cos_B = 4 * real.sqrt 17 / 17 :=
by
  rw [ha, hc, hcosB]
  have h1: b = side_b 2 5 (3 / 5),
  { sorry }
  rw h1,
  have h2: sin_C 2 (real.sqrt 17) 5 (3 / 5) = 4 * real.sqrt 17 / 17,
  { sorry }
  rw h2,
  exact ⟨h1, h2⟩

end triangle_sides_and_angles_l224_224580


namespace sin_theta_eq_2_over_7_l224_224261

variables (a b c : ℝ^3) (θ : ℝ)
hypotheses 
  (h_norm_a : ‖a‖ = 2)
  (h_norm_b : ‖b‖ = 7)
  (h_norm_c : ‖c‖ = 4)
  (h_triple_product : a × (a × b) = c)
  (h_angle : θ = real.angleBetween a b)

theorem sin_theta_eq_2_over_7 : real.sin θ = 2/7 :=
by
  sorry

end sin_theta_eq_2_over_7_l224_224261


namespace tennis_tournament_cycle_l224_224445

noncomputable def exists_cycle_of_three_players (P : Type) [Fintype P] (G : P → P → Bool) : Prop :=
  (∃ (a b c : P), a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ G a b ∧ G b c ∧ G c a)

theorem tennis_tournament_cycle (P : Type) [Fintype P] (n : ℕ) (hp : 3 ≤ n) 
  (G : P → P → Bool) (H : ∀ a b : P, a ≠ b → (G a b ∨ G b a))
  (Hw : ∀ a : P, ∃ b : P, a ≠ b ∧ G a b) : exists_cycle_of_three_players P G :=
by 
  sorry

end tennis_tournament_cycle_l224_224445


namespace no_calls_days_l224_224269

/-- Mrs. Johnson has three grandchildren who call her regularly. 
    One calls her every two days, another every three days, and the third every four days. 
    All three called her on January 1, 2022. 
    Given this: How many days during 2022 did she not receive a phone call from any of her grandchildren? -/
theorem no_calls_days : 
  let days_in_year := 365
  let calls_every (n : ℕ) := λ d : ℕ, d % n = 0
  let grandchild_1_calls := calls_every 2
  let grandchild_2_calls := calls_every 3
  let grandchild_3_calls := calls_every 4
  let days_with_calls := finset.filter (λ d, grandchild_1_calls d ∨ grandchild_2_calls d ∨ grandchild_3_calls d) (finset.range days_in_year).card
  days_in_year - days_with_calls = 122 := 
by sorry

end no_calls_days_l224_224269


namespace Kolya_is_acting_as_collection_agency_l224_224215

-- Definitions for the conditions given in the problem
def Katya_lent_books_to_Vasya : Prop := ∃ books : Type, ∀ b : books, ¬ b ∈ Katya's_collection
def Vasya_promised_to_return_books_in_a_month_but_failed : Prop := ∀ t : Time, t ≥ 1 month → ¬returned books by Vasya
def Katya_asked_Kolya_to_get_books_back : Prop := ∀ k : Kolya, ∀ v : Vasya, asked Katya (k to get books back from v)
def Kolya_agrees_but_wants_a_reward : Prop := ∃ reward : Book, Kolya_gets reward

-- Define the property of Kolya being a collection agency
def Kolya_is_collection_agency : Prop :=
  Katya_lent_books_to_Vasya ∧
  Vasya_promised_to_return_books_in_a_month_but_failed ∧
  Katya_asked_Kolya_to_get_books_back ∧
  Kolya_agrees_but_wants_a_reward

-- The theorem to prove
theorem Kolya_is_acting_as_collection_agency :
  Kolya_is_collection_agency :=
sorry

end Kolya_is_acting_as_collection_agency_l224_224215


namespace balloon_permutation_count_l224_224927

-- Define the given conditions:
def balloon_letters : List Char := ['B', 'A', 'L', 'L', 'O', 'O', 'N']
def total_letters : Nat := 7
def count_L : Nat := 2
def count_O : Nat := 2

-- Statement to prove:
theorem balloon_permutation_count :
  (nat.factorial total_letters) / ((nat.factorial count_L) * (nat.factorial count_O)) = 1260 := by
  sorry

end balloon_permutation_count_l224_224927


namespace balloon_permutations_l224_224948

theorem balloon_permutations : 
  let total_letters := 7
  let repetitions_L := 2
  let repetitions_O := 2
  (Nat.factorial total_letters) / ((Nat.factorial repetitions_L) * (Nat.factorial repetitions_O)) = 1260 := 
by
  sorry

end balloon_permutations_l224_224948


namespace balloon_arrangement_count_l224_224971

-- Definitions of letter frequencies for the word BALLOON
def letter_frequencies : List (Char × Nat) := [('B', 1), ('A', 1), ('L', 2), ('O', 2), ('N', 1)]

-- Total number of letters
def total_letters := 7

-- The formula for the number of unique arrangements of the letters
noncomputable def arrangements :=
  (Nat.factorial total_letters) / 
  (Nat.factorial 1 * Nat.factorial 1 * Nat.factorial 2 * Nat.factorial 2 * Nat.factorial 1)

-- The theorem to prove the number of ways to arrange the letters in "BALLOON"
theorem balloon_arrangement_count : arrangements = 1260 :=
  sorry

end balloon_arrangement_count_l224_224971


namespace measure_angle_Q_l224_224671

-- Define the structure for a regular decagon
structure Decagon (A B C D E F G H I J : Type) where
  regular : ∀ (i j : ℕ), (1 ≤ i ∧ i ≤ 10) → (1 ≤ j ∧ j ≤ 10) → (angle A B C = angle C D E)

-- Define the geometry and the extended lines meeting at point Q
noncomputable def angle_measure_Q (A B C D E F G H I J Q : Type) [Decagon A B C D E F G H I J] : ℝ :=
      let interior_angle := 144 in
      let reflex_E := 360 - interior_angle in
      360 - 36 - reflex_E - 36

theorem measure_angle_Q (A B C D E F G H I J Q : Type) [Decagon A B C D E F G H I J] :
    angle_measure_Q A B C D E F G H I J Q = 72 :=
  sorry

end measure_angle_Q_l224_224671


namespace spaceship_age_base10_l224_224848

def octalToDecimal (n : ℕ) : ℕ :=
  let digits := [n % 10, (n / 10) % 10, (n / 100) % 10]
  digits[0] * 8^0 + digits[1] * 8^1 + digits[2] * 8^2

theorem spaceship_age_base10 (h : octalToDecimal 367 = 247) : True :=
begin
  sorry
end

end spaceship_age_base10_l224_224848


namespace eight_digit_palindromes_count_l224_224009

theorem eight_digit_palindromes_count : 
  ∃ (count : ℕ), count = 81 ∧ 
    (∀ (d1 d2 d3 d4 : ℕ), 
      d1 ∈ {1, 2, 3} ∧ 
      d2 ∈ {1, 2, 3} ∧ 
      d3 ∈ {1, 2, 3} ∧ 
      d4 ∈ {1, 2, 3} → 
      (d1 ≠ 0) ∧ (d2 ≠ 0) ∧ (d3 ≠ 0) ∧ (d4 ≠ 0) →
      (3^4 = count)) :=
by
  use 81
  sorry

end eight_digit_palindromes_count_l224_224009


namespace balloon_permutations_count_l224_224906

-- Definitions of the conditions
def total_letters_count : ℕ := 7
def l_count : ℕ := 2
def o_count : ℕ := 2

-- Now the mathematical problem as a Lean statement
theorem balloon_permutations_count : 
  (Nat.factorial total_letters_count) / ((Nat.factorial l_count) * (Nat.factorial o_count)) = 1260 := 
by
  sorry

end balloon_permutations_count_l224_224906


namespace discount_calculation_l224_224708

noncomputable def cost_price : ℝ := 180
noncomputable def markup_percentage : ℝ := 0.4778
noncomputable def profit_percentage : ℝ := 0.20

noncomputable def marked_price (CP : ℝ) (MP_percent : ℝ) : ℝ := CP + (MP_percent * CP)
noncomputable def selling_price (CP : ℝ) (PP_percent : ℝ) : ℝ := CP + (PP_percent * CP)
noncomputable def discount (MP : ℝ) (SP : ℝ) : ℝ := MP - SP

theorem discount_calculation :
  discount (marked_price cost_price markup_percentage) (selling_price cost_price profit_percentage) = 50.004 :=
by
  sorry

end discount_calculation_l224_224708


namespace intersection_lies_on_circumcircle_l224_224002

variables (A B C O M N A' B' : Point)
variables (circumcircle : Circle)
variables [incircle_of_triangle A B C O]
variables [line_through_perpendicular_to O C M N AC BC]
variables [line_intersects_circumcircle AO circumcircle A']
variables [line_intersects_circumcircle BO circumcircle B']

theorem intersection_lies_on_circumcircle :
  lies_on_circumcircle (intersection (line A' N) (line B' M)) circumcircle :=
sorry

end intersection_lies_on_circumcircle_l224_224002


namespace evan_can_write_one_on_board_l224_224634

theorem evan_can_write_one_on_board (m n : ℕ) (h_rel_prime : Nat.gcd m n = 1) : 
  (∃ steps : ℕ, evan_can_write_one (m, n) steps) ↔ Nat.is_power_of_two (m + n) := sorry

end evan_can_write_one_on_board_l224_224634


namespace recurring_to_fraction_l224_224756

theorem recurring_to_fraction : ∀ (x : ℚ), x = 0.5656 ∧ 100 * x = 56.5656 → x = 56 / 99 :=
by
  intros x hx
  sorry

end recurring_to_fraction_l224_224756


namespace find_x_when_y_neg_five_l224_224675

-- Definitions based on the conditions provided
variable (x y : ℝ)
def inversely_proportional (x y : ℝ) := ∃ (k : ℝ), x * y = k

-- Proving the main result
theorem find_x_when_y_neg_five (h_prop : inversely_proportional x y) (hx4 : x = 4) (hy2 : y = 2) :
    (y = -5) → x = - 8 / 5 := by
  sorry

end find_x_when_y_neg_five_l224_224675


namespace johns_total_money_made_l224_224235

-- Defining the conditions
def total_caterpillars := 4 * 10
def failure_rate := 0.4
def success_rate := 1 - failure_rate
def price_per_butterfly := 3

-- Derived values
def caterpillars_failed := total_caterpillars * failure_rate
def caterpillars_successful := total_caterpillars - caterpillars_failed
def total_money_made := caterpillars_successful * price_per_butterfly

-- Statement to be proven
theorem johns_total_money_made : total_money_made = 72 := by
  sorry

end johns_total_money_made_l224_224235


namespace obtain_any_natural_l224_224440

-- Define the operations
def op1 (x : ℕ) : ℕ := 3 * x + 1
def op2 (x : ℤ) : ℕ := (x / 2).to_nat

-- Theorem statement: Starting from 1, any natural number n can be obtained using op1 and op2
theorem obtain_any_natural (n : ℕ) : ∃ (f : ℕ → ℕ), f 1 = n ∧ ∀ i, f (i + 1) = op1 (f i) ∨ f (i + 1) = op2 (f i) := sorry

end obtain_any_natural_l224_224440


namespace total_food_needed_l224_224653

-- Definitions for the conditions
def horses : ℕ := 4
def oats_per_meal : ℕ := 4
def oats_meals_per_day : ℕ := 2
def grain_per_day : ℕ := 3
def days : ℕ := 3

-- Theorem stating the problem
theorem total_food_needed :
  (horses * (days * (oats_per_meal * oats_meals_per_day) + days * grain_per_day)) = 132 :=
by sorry

end total_food_needed_l224_224653


namespace curve_equation_l224_224534

theorem curve_equation {a b : ℝ} (h1 : a + b = 2) (h2 : a*0^2 + b*(5/3)^2 = 2) (h3 : a*1^2 + b*1^1 = 2) :
  (a = 32/25) ∧ (b = 18/25) → ∀ x y : ℝ, (16/25)*x^2 + (9/25)*y^2 = 1 :=
by
  intro hab eq1 eq2,
  sorry

end curve_equation_l224_224534


namespace sum_of_digits_of_d_l224_224463

noncomputable def rate : ℚ := 8 / 5
noncomputable def spent_amount : ℤ := 80
noncomputable def initial_dollars (d : ℤ) : ℚ := (rate * d : ℚ)
noncomputable def remaining_euros (initial_d : ℤ) : ℤ := (initial_d - spent_amount : ℤ)

theorem sum_of_digits_of_d (d : ℤ) (h : remaining_euros (initial_dollars d) = d) : d.digits.sum = 7 :=
by {
  sorry
}

end sum_of_digits_of_d_l224_224463


namespace product_of_x_y_l224_224380

theorem product_of_x_y (x y : ℝ) (h1 : -3 * x + 4 * y = 28) (h2 : 3 * x - 2 * y = 8) : x * y = 264 :=
by
  sorry

end product_of_x_y_l224_224380


namespace balloon_permutations_l224_224950

theorem balloon_permutations : 
  let total_letters := 7
  let repetitions_L := 2
  let repetitions_O := 2
  (Nat.factorial total_letters) / ((Nat.factorial repetitions_L) * (Nat.factorial repetitions_O)) = 1260 := 
by
  sorry

end balloon_permutations_l224_224950


namespace triangle_area_ratio_proof_l224_224248

section
variables {V : Type*} [inner_product_space ℝ V]

noncomputable def ratio_area_ABC_APB (A B C P : V) 
  (h : 2 • (A - P) + 3 • (B - P) + (C - P) = 0) : ℝ :=
5

theorem triangle_area_ratio_proof 
  (A B C P : V) 
  (h : 2 • (A - P) + 3 • (B - P) + (C - P) = 0) : 
  ratio_area_ABC_APB A B C P h = 5 :=
by
  sorry

end triangle_area_ratio_proof_l224_224248


namespace functions_equal_l224_224428

theorem functions_equal
  (f : ℝ → ℝ) (g : ℝ → ℝ)
  (hf : ∀ x, f x = x)
  (hg : ∀ x, g x = log10 (10 ^ x)) :
  f = g :=
sorry

end functions_equal_l224_224428


namespace probability_product_less_than_30_l224_224274

-- Define the spinner outcomes
def PacoSpinner := {n : ℕ // 1 ≤ n ∧ n ≤ 5}
def ManuSpinner := {n : ℕ // 1 ≤ n ∧ n ≤ 12}

-- Define the probability space for Paco and Manu's spinners
noncomputable def uniform_prob {α : Type} [Fintype α] : ProbabilityMeasure α :=
Fintype.uniformProbability α

-- Event definition
def eventProductLessThanThirty (p : PacoSpinner) (m : ManuSpinner) : Prop :=
p.val * m.val < 30

-- Main theorem statement
theorem probability_product_less_than_30 :
  ProbabilityMeasure.toMeasure (uniform_prob : ProbabilityMeasure (PacoSpinner × ManuSpinner))
    (λ pm : PacoSpinner × ManuSpinner, eventProductLessThanThirty pm.1 pm.2) =
  51 / 60 := 
sorry

end probability_product_less_than_30_l224_224274


namespace triangle_area_ratio_proof_l224_224249

section
variables {V : Type*} [inner_product_space ℝ V]

noncomputable def ratio_area_ABC_APB (A B C P : V) 
  (h : 2 • (A - P) + 3 • (B - P) + (C - P) = 0) : ℝ :=
5

theorem triangle_area_ratio_proof 
  (A B C P : V) 
  (h : 2 • (A - P) + 3 • (B - P) + (C - P) = 0) : 
  ratio_area_ABC_APB A B C P h = 5 :=
by
  sorry

end triangle_area_ratio_proof_l224_224249


namespace find_magnitude_l224_224243

namespace ProofProblem

noncomputable def i : ℂ := complex.I
noncomputable def T : ℂ := (1 + i)^19 - (1 - i)^19

theorem find_magnitude : complex.abs T = 1024 := 
sorry

end ProofProblem

end find_magnitude_l224_224243


namespace ellipse_equation_l224_224311

theorem ellipse_equation (P Q : ℝ × ℝ)
  (foci_axis : ℝ) 
  (line_intersect : ∀ (x : ℝ), (x, x + 1) = P ∨ (x, x + 1) = Q) 
  (dot_product_zero : P.1 * Q.1 + P.2 * Q.2 = 0) 
  (distance_PQ : real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) = real.sqrt(10) / 2) :
  (∃ m n : ℝ, [m, n] = [3/2, 1/2] ∨ [m, n] = [1/2, 3/2]
       ∧ (∀ (x y : ℝ), m * x^2 + n * y^2 = 1 ↔ 
                        (n = 2/3 ∧ x^2 / 2 + y^2 / (2/3) = 1) ∨ 
                        (m = 2/3 ∧ y^2 / 2 + x^2 / (2/3) = 1))) := 
by sorry

end ellipse_equation_l224_224311


namespace isosceles_trapezoid_area_l224_224592

theorem isosceles_trapezoid_area (a : ℝ) : 
  ∀ (ABCD : Type) (AB BC CD DA : ℝ) (is_isosceles : //isosceles trapezoid condition//)
    (midline_eq_a : // midline condition //) 
    (diagonals_perpendicular : //perpendicular diagonals condition//),
    // area of ABCD equals //
    area = a^2 :=
by
  sorry

end isosceles_trapezoid_area_l224_224592


namespace equation_solutions_l224_224299

theorem equation_solutions : 
  ∀ x : ℝ, (2 * x - 1) - x * (1 - 2 * x) = 0 ↔ (x = 1 / 2 ∨ x = -1) :=
by
  intro x
  sorry

end equation_solutions_l224_224299


namespace area_of_triangle_ABC_l224_224720

structure Circle :=
  (center : ℝ × ℝ)
  (radius : ℝ)

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem area_of_triangle_ABC :
  ∃ (A B C : Circle),
    A.radius = 2 ∧
    B.radius = 4 ∧
    C.radius = 5 ∧
    A.center.2 = 2 ∧
    B.center = (0, 4) ∧
    C.center = (-9, 5) ∧
    distance A.center B.center = 2 ∧
    distance B.center C.center = 9 →
    1/2 * real.abs (2 * (4-5) + 0 * (5-2) + (-9) * (2-4)) = 8 :=
by intros A B C hA hB hC hAcenter hBcenter hCcenter hABdist hBCdist sorry

end area_of_triangle_ABC_l224_224720


namespace octahedron_cube_ratio_l224_224517

theorem octahedron_cube_ratio (O C : Type) [RegularOctahedron O] [CubeWithVerticesOfFaceCentersOfOctahedron C] :
    let V_O := volume_regular_octahedron 1
    let V_C := volume_cube_with_vertices_of_face_centers 1
    let ratio := V_O / V_C
    let m := 9
    let n := 2
    ratio = (m / n) := by
  -- The conditions are represented but the proof steps are skipped.
  sorry

lemma result_value_of_m_plus_n : 9 + 2 = 11 := by
  exact rfl

end octahedron_cube_ratio_l224_224517


namespace minimum_committees_l224_224733

variables {Prof : Type} {Comm : Type}

-- Conditions setup
def distinct (a b : Prof) : Prop := a ≠ b

axiom cond1 : ∀ (a b : Prof) (h : distinct a b), ∃! (C : Comm), a ∈ C ∧ b ∈ C
axiom cond2 : ∀ (C : Comm) (P : Prof), P ∉ C → ∃! (D : Comm), P ∈ D ∧ ∀ (Q : Prof), Q ∈ C → Q ∉ D
axiom cond3 : ∀ (C : Comm), ∃ (a b : Prof), a ∈ C ∧ b ∈ C ∧ a ≠ b
axiom cond4 : ∃ (C1 C2 : Comm), C1 ≠ C2

-- Main theorem
theorem minimum_committees : ∃ (n : ℕ), (∀ C : Comm, cond3 C) ∧ (∀ C1 C2 : Comm, C1 ≠ C2 → ∃ (a : Prof), a ∈ C1 ∧ a ∉ C2) ∧ n = 6 :=
begin
  sorry
end

end minimum_committees_l224_224733


namespace total_distance_covered_l224_224053

theorem total_distance_covered
  (V_up : ℝ) (V_down : ℝ) (T_total : ℝ)
  (h1 : V_up = 50) (h2 : V_down = 100) (h3 : T_total = 16) :
  let D := (T_total * V_up * V_down) / (V_up + V_down) in
  2 * D = 1066.6 := by
  sorry

end total_distance_covered_l224_224053


namespace abs_value_inequality_solutions_l224_224302

noncomputable def a (x : ℝ) : ℝ := 4 * x^2 - 32 / x
noncomputable def b (x : ℝ) : ℝ := x^2 + 5 / (x^2 - 6)
noncomputable def c (x : ℝ) : ℝ := 3 * x^2 - 32 / x - 5 / (x^2 - 6)

theorem abs_value_inequality_solutions :
  { x : ℝ | ∥a x - b x∥ ≤ ∥a x∥ + ∥b x∥ → (a x) * (b x) ≤ 0 } ∩
  { x : ℝ | (x > -sqrt (6) ∧ x <= -sqrt (5)) ∨
            (x >= -1 ∧ x < 0) ∨
            (x >= 1 ∧ x <= 2) ∨
            (x >= sqrt(5) ∧ x < sqrt(6))} :=
begin
  sorry
end

end abs_value_inequality_solutions_l224_224302


namespace cylindrical_to_rectangular_l224_224872

noncomputable def convertToRectangular (r θ z : ℝ) : ℝ × ℝ × ℝ :=
  (r * Real.cos θ, r * Real.sin θ, z)

theorem cylindrical_to_rectangular :
  let r := 10
  let θ := Real.pi / 3
  let z := 2
  let r' := 2 * r
  let z' := z + 1
  convertToRectangular r' θ z' = (10, 10 * Real.sqrt 3, 3) :=
by
  sorry

end cylindrical_to_rectangular_l224_224872


namespace profit_calculation_l224_224270

noncomputable def calculate_profit 
  (cost_per_rug : ℕ)
  (discount_rate : ℚ)
  (selling_price_per_rug : ℕ)
  (tax_rate : ℚ)
  (transportation_fee_per_rug : ℕ)
  (number_of_rugs : ℕ) : ℚ :=
let total_cost_before_discount := cost_per_rug * number_of_rugs,
    discount_amount := total_cost_before_discount * discount_rate,
    total_cost_after_discount := total_cost_before_discount - discount_amount,
    total_selling_price_before_tax := selling_price_per_rug * number_of_rugs,
    total_tax := total_selling_price_before_tax * tax_rate,
    total_selling_price_after_tax := total_selling_price_before_tax + total_tax,
    total_transportation_fee := transportation_fee_per_rug * number_of_rugs in
total_selling_price_after_tax - total_cost_after_discount - total_transportation_fee

theorem profit_calculation : 
  calculate_profit 40 0.05 60 0.10 5 20 = 460 := 
by 
  -- This is where the proof steps will be written, for now we use sorry to skip the proof
  sorry

end profit_calculation_l224_224270


namespace cost_of_individual_roll_is_correct_l224_224818

-- Definitions given in the problem's conditions
def cost_per_case : ℝ := 9
def number_of_rolls : ℝ := 12
def percent_savings : ℝ := 0.25

-- The cost of one roll sold individually
noncomputable def individual_roll_cost : ℝ := 0.9375

-- The theorem to prove
theorem cost_of_individual_roll_is_correct :
  individual_roll_cost = (cost_per_case * (1 + percent_savings)) / number_of_rolls :=
by
  sorry

end cost_of_individual_roll_is_correct_l224_224818


namespace find_b_l224_224472

-- Definitions of polynomials
def P := (8 * X^3 - 9 * X^2 + b * X + 5)
def Q := (3 * X^2 - 2 * X + 1)

-- Condition: The remainder is constant
theorem find_b (b : ℚ) (h : Polynomial.ring_div P Q = C r) : 
  b = 46 / 9 :=
begin
  sorry
end

end find_b_l224_224472


namespace recurring_to_fraction_l224_224754

theorem recurring_to_fraction : ∀ (x : ℚ), x = 0.5656 ∧ 100 * x = 56.5656 → x = 56 / 99 :=
by
  intros x hx
  sorry

end recurring_to_fraction_l224_224754


namespace simplify_expression_l224_224134

def operation (a b : ℚ) : ℚ := 2 * a - b

theorem simplify_expression (x y : ℚ) : 
  operation (operation (x - y) (x + y)) (-3 * y) = 2 * x - 3 * y :=
by
  sorry

end simplify_expression_l224_224134


namespace balloon_arrangements_l224_224900

theorem balloon_arrangements :
  let n := 7
  let k1 := 2
  let k2 := 2
  (Nat.factorial n) / ((Nat.factorial k1) * (Nat.factorial k2)) = 1260 :=
by
  -- Use the conditions directly from the problem
  let n := 7
  let k1 := 2
  let k2 := 2
  -- Mathematically, the number of arrangements of the multiset
  have h : (Nat.factorial n) / ((Nat.factorial k1) * (Nat.factorial k2)) = 1260 := sorry
  exact h

end balloon_arrangements_l224_224900


namespace length_of_goods_train_l224_224400

theorem length_of_goods_train
  (speed_km_per_hr : ℕ)
  (platform_length_m : ℕ)
  (crossing_time_s : ℕ)
  (conversion_factor : ℝ := 1000 / 3600) :
  speed_km_per_hr = 72 → 
  platform_length_m = 150 → 
  crossing_time_s = 26 → 
  let speed_m_per_s := speed_km_per_hr * conversion_factor in
  let distance_covered := speed_m_per_s * crossing_time_s in
  let train_length := distance_covered - platform_length_m in
  train_length = 370 :=
by
  intros
  simp [speed_km_per_hr, platform_length_m, crossing_time_s, conversion_factor]
  let speed_m_per_s := 72 * (1000 / 3600)
  let distance_covered := speed_m_per_s * 26
  let train_length := distance_covered - 150
  show train_length = 370
  sorry

end length_of_goods_train_l224_224400


namespace increasing_interval_l224_224696

noncomputable def f (x k : ℝ) : ℝ := (x^2 / 2) - k * (Real.log x)

theorem increasing_interval (k : ℝ) (h₀ : 0 < k) : 
  ∃ (a : ℝ), (a = Real.sqrt k) ∧ 
  ∀ (x : ℝ), (x > a) → (∃ ε > 0, ∀ y, (x < y) → (f y k > f x k)) :=
sorry

end increasing_interval_l224_224696


namespace total_selling_price_correct_l224_224406

-- Definitions for the problem conditions
def purchase_price_cycle : ℝ := 2000
def purchase_price_scooter : ℝ := 25000
def purchase_price_bike : ℝ := 60000

def loss_percentage_cycle : ℝ := 10 / 100
def loss_percentage_scooter : ℝ := 15 / 100
def loss_percentage_bike : ℝ := 5 / 100

-- Definitions for calculating loss amounts
def loss_amount (purchase_price loss_percentage : ℝ) : ℝ :=
  (loss_percentage) * purchase_price

def selling_price (purchase_price loss_amount : ℝ) : ℝ :=
  purchase_price - loss_amount

-- Calculate the selling prices of each item
def selling_price_cycle : ℝ :=
  selling_price purchase_price_cycle (loss_amount purchase_price_cycle loss_percentage_cycle)

def selling_price_scooter : ℝ :=
  selling_price purchase_price_scooter (loss_amount purchase_price_scooter loss_percentage_scooter)

def selling_price_bike : ℝ :=
  selling_price purchase_price_bike (loss_amount purchase_price_bike loss_percentage_bike)

-- Calculate the total selling price
def total_selling_price : ℝ :=
  selling_price_cycle + selling_price_scooter + selling_price_bike

-- Lean 4 statement for the theorem
theorem total_selling_price_correct : total_selling_price = 80050 := by
  sorry

end total_selling_price_correct_l224_224406


namespace balloon_permutations_l224_224946

theorem balloon_permutations : 
  let total_letters := 7
  let repetitions_L := 2
  let repetitions_O := 2
  (Nat.factorial total_letters) / ((Nat.factorial repetitions_L) * (Nat.factorial repetitions_O)) = 1260 := 
by
  sorry

end balloon_permutations_l224_224946


namespace cos_angle_eq_l224_224120

theorem cos_angle_eq :
  ∃ n : ℕ, 0 ≤ n ∧ n ≤ 180 ∧ real.cos (n * real.pi / 180) = real.cos (259 * real.pi / 180) ∧ n = 101 :=
by
  sorry

end cos_angle_eq_l224_224120


namespace eight_people_lineup_ways_l224_224695

theorem eight_people_lineup_ways : (Nat.factorial 8 = 40320) :=
by
  sorry

end eight_people_lineup_ways_l224_224695


namespace arithmetic_sequence_ratio_l224_224239

variable {a b : ℕ → ℝ}

-- Definitions for the sums of the first n terms of the sequences
def S (n : ℕ) := ∑ i in Finset.range n, a i
def T (n : ℕ) := ∑ i in Finset.range n, b i

-- Given condition
axiom cond (n : ℕ) : S n / T n = 7 * n / (n + 3)

-- The proof goal with the specific case of the problem
theorem arithmetic_sequence_ratio (h : ∀ n, S n / T n = 7 * n / (n + 3)) : a 5 / b 5 = 21 / 4 :=
by
  sorry

end arithmetic_sequence_ratio_l224_224239


namespace repeating_decimal_to_fraction_l224_224777

theorem repeating_decimal_to_fraction : (0.5656565656 : ℚ) = 56 / 99 :=
  by
    sorry

end repeating_decimal_to_fraction_l224_224777


namespace log_expr_result_l224_224995

noncomputable def log_expr: ℝ :=  
  logBase 3 ((27:ℝ)^(1/4) / 3) + log 10 25 + log 10 4 + 7^(log 7 2)

theorem log_expr_result: log_expr = (15 / 4) := by
  sorry

end log_expr_result_l224_224995


namespace angle_bisector_AC_l224_224615

open EuclideanGeometry

section GeometryProblem

variables {P Q A B C O : Point}
variables {H : Circle}
variables {l : Line}

-- Provided conditions
axiom diameter_PQ : diameter H P Q
axiom tangent_C_H_PQ : is_tangent H O ∧ is_tangent O l at C

-- Given an additional conditions on A and B
axiom A_on_H : on_circle H A
axiom B_on_PQ : on_line l B
axiom AB_perp_PQ : perp AB l
axiom AB_tangent_O : is_tangent O AB

-- Result to prove
theorem angle_bisector_AC :
  bisects_angle A C P Q l := sorry

end GeometryProblem

end angle_bisector_AC_l224_224615


namespace ratio_recharged_total_pay_l224_224859

variable (total_pay car_cost remaining_money recharged_amount : ℝ)

axiom h1 : total_pay = 5000
axiom h2 : car_cost = 1500
axiom h3 : remaining_money = 1000
axiom h4 : recharged_amount = total_pay - car_cost - remaining_money

theorem ratio_recharged_total_pay : recharged_amount / total_pay = 1 / 2 :=
  by
    rw [h1, h2, h3, h4]
    sorry

end ratio_recharged_total_pay_l224_224859


namespace probability_non_yellow_l224_224815

def num_red := 4
def num_green := 7
def num_yellow := 9
def num_blue := 10

def total_jelly_beans := num_red + num_green + num_yellow + num_blue
def num_non_yellow := num_red + num_green + num_blue

theorem probability_non_yellow : (num_non_yellow : ℚ) / total_jelly_beans = 7 / 10 :=
by
  have h1: total_jelly_beans = 30 := by norm_num
  have h2: num_non_yellow = 21 := by norm_num
  rw [h1, h2]
  norm_num
  sorry

end probability_non_yellow_l224_224815


namespace billy_ate_9_apples_on_Wednesday_l224_224080

theorem billy_ate_9_apples_on_Wednesday :
  ∀ (total_apples monday_apples tuesday_apples friday_apples thursday_apples wednesday_apples : ℕ),
    total_apples = 20 →
    monday_apples = 2 →
    tuesday_apples = 2 * monday_apples →
    friday_apples = monday_apples / 2 →
    thursday_apples = 4 * friday_apples →
    total_apples = monday_apples + tuesday_apples + thursday_apples + friday_apples + wednesday_apples →
      wednesday_apples = 9 :=
by {
  intros,
  sorry
}

end billy_ate_9_apples_on_Wednesday_l224_224080


namespace eight_digit_palindrome_count_l224_224012

def is_palindrome (n : ℕ) : Prop :=
  let digits := n.digits 10 in
  digits = digits.reverse

def is_eight_digit (n : ℕ) : Prop :=
  n >= 10^7 ∧ n < 10^8

def valid_digits (n : ℕ) : Prop :=
  ∀ d ∈ n.digits 10, d = 1 ∨ d = 2 ∨ d = 3

theorem eight_digit_palindrome_count : 
  ∃ count : ℕ, count = 81 ∧ (∀ n : ℕ, is_palindrome n ∧ is_eight_digit n ∧ valid_digits n ↔ n ∈ { n | is_palindrome n ∧ is_eight_digit n ∧ valid_digits n } ∧ |{ n | is_palindrome n ∧ is_eight_digit n ∧ valid_digits n }| = 81) :=
by
  sorry

end eight_digit_palindrome_count_l224_224012


namespace find_S2010_l224_224145

variables (a : ℕ → ℝ) (S : ℕ → ℝ)

-- Definition of an arithmetic sequence sum of the first n terms
def sum_arithmetic_sequence (n : ℕ) : ℝ := (n / 2) * (a 1 + a n)

-- Conditions
axiom condition1 : a 1005 + a 1006 = 1
axiom condition2 (n : ℕ) : S n = sum_arithmetic_sequence a n

-- The result we want to prove
theorem find_S2010 : S 2010 = 1005 :=
sorry

end find_S2010_l224_224145


namespace fraction_to_decimal_l224_224395

theorem fraction_to_decimal (num : ℕ) (denom : ℕ) (h_fraction : num / denom = 7 / 20) : num / denom = 0.35 := 
by
  sorry

end fraction_to_decimal_l224_224395


namespace simplification_l224_224151

variable (a b : ℝ)

-- Define the conditions
def condition1 := a < 0
def condition2 := ab < 0

-- Define the expression to be simplified
def expr := |a - b - 3| - |4 + b - a|

-- State the theorem
theorem simplification (h1 : condition1 a b) (h2 : condition2 a b) : expr a b = -1 :=
by
  sorry

end simplification_l224_224151


namespace find_sum_of_perimeters_l224_224681

variables (x y : ℝ)
noncomputable def sum_of_perimeters := 4 * x + 4 * y

theorem find_sum_of_perimeters (h1 : x^2 + y^2 = 65) (h2 : x^2 - y^2 = 33) :
  sum_of_perimeters x y = 44 :=
sorry

end find_sum_of_perimeters_l224_224681


namespace balloon_arrangements_l224_224890

theorem balloon_arrangements :
  let n := 7
  let k1 := 2
  let k2 := 2
  (Nat.factorial n) / ((Nat.factorial k1) * (Nat.factorial k2)) = 1260 :=
by
  -- Use the conditions directly from the problem
  let n := 7
  let k1 := 2
  let k2 := 2
  -- Mathematically, the number of arrangements of the multiset
  have h : (Nat.factorial n) / ((Nat.factorial k1) * (Nat.factorial k2)) = 1260 := sorry
  exact h

end balloon_arrangements_l224_224890


namespace total_food_needed_l224_224654

-- Definitions for the conditions
def horses : ℕ := 4
def oats_per_meal : ℕ := 4
def oats_meals_per_day : ℕ := 2
def grain_per_day : ℕ := 3
def days : ℕ := 3

-- Theorem stating the problem
theorem total_food_needed :
  (horses * (days * (oats_per_meal * oats_meals_per_day) + days * grain_per_day)) = 132 :=
by sorry

end total_food_needed_l224_224654


namespace ten_thousandths_place_of_fraction_l224_224014

-- Noncomputable theory to use real numbers and their properties
noncomputable theory

open Real BigOperators 

def digit_in_ten_thousandths_place (n d : ℕ) (den_lt_0: d > 0) :=
  ∃ k : ℕ, (10^4 * k = (n / d - (real.floor (n / d))) * 10^5) ∧ k % 10 = 8

theorem ten_thousandths_place_of_fraction :
  digit_in_ten_thousandths_place 7 32 (by decide) :=
begin
  sorry,
end

end ten_thousandths_place_of_fraction_l224_224014


namespace alice_bob_get_same_heads_mn_sum_correct_l224_224067

def prob_heads_eq (fair_prob : ℚ) (biased_prob : ℚ) (num_heads : ℚ) : ℚ :=
  let fair_gen := 1 + fair_prob in
  let biased_gen := 2 * biased_prob + 3 in
  (fair_gen ^ 2 * biased_gen).num ^ 2 - ((fair_prob ^ 2 + 8 * fair_prob + biased_prob ^ 3) ^ 2 + (9 + 64 + 49 + 4))
  
#check prob_heads_eq (1/2) (3/5) (63/200)

theorem alice_bob_get_same_heads : prob_heads_eq (1/2) (3/5) (63/200) = 63 / 200 :=
by sorry

theorem mn_sum_correct : 63 + 200 = 263 :=
by sorry

end alice_bob_get_same_heads_mn_sum_correct_l224_224067


namespace min_positive_period_f_l224_224332

def f (x : ℝ) : ℝ := 2 * Real.sin (3 * x + Real.pi / 6)

theorem min_positive_period_f :
  (exists T : ℝ, T > 0 ∧ (∀ x : ℝ, f (x + T) = f x)) ∧
  (∀ T' : ℝ, T' > 0 ∧ (∀ x : ℝ, f (x + T') = f x) → T' ≥ 2 * Real.pi / 3) := sorry

end min_positive_period_f_l224_224332


namespace largest_among_abc_is_25_l224_224609

-- Given variables and condition
variables (a b c : ℤ)
variables (h1 : 2 * a + 3 * b + 4 * c = 225)
variables (h2 : a + b + c = 60)
variables (h3 : a = 15 ∨ b = 15 ∨ c = 15)

-- Define the property to prove
def largest_integer := a.max b.max c

-- Goal: Prove the largest integer among a, b, and c is 25
theorem largest_among_abc_is_25 (h1 : 2 * a + 3 * b + 4 * c = 225)
                                (h2 : a + b + c = 60)
                                (h3 : a = 15 ∨ b = 15 ∨ c = 15) :
  largest_integer a b c = 25 :=
sorry

end largest_among_abc_is_25_l224_224609


namespace median_of_ride_times_is_163_l224_224340

def ride_times : List (ℕ × ℕ) :=
  [ (0, 28), (0, 28), (0, 50), (0, 55),
    (1, 0),  (1, 2),  (1, 10),
    (2, 20), (2, 25), (2, 35), (2, 43), (2, 45), (2, 50),
    (3, 0),  (3, 0),  (3, 0),  (3, 5),  (3, 30), (3, 36),
    (4, 0),  (4, 10), (4, 15), (4, 20) ]

def time_to_seconds (t : ℕ × ℕ) : ℕ :=
  t.1 * 60 + t.2

noncomputable def median_time_in_seconds : ℕ :=
  time_to_seconds (ride_times.nth 10).getD (0, 0) -- The 11th element (index 10)

theorem median_of_ride_times_is_163 : median_time_in_seconds = 163 :=
  by
    sorry

end median_of_ride_times_is_163_l224_224340


namespace area_ACD_l224_224712

def base_ABD : ℝ := 8
def height_ABD : ℝ := 4
def base_ABC : ℝ := 4
def height_ABC : ℝ := 4

theorem area_ACD : (1/2 * base_ABD * height_ABD) - (1/2 * base_ABC * height_ABC) = 8 := by
  sorry

end area_ACD_l224_224712


namespace vector_projection_condition_l224_224087

noncomputable def line_l (t : ℝ) : ℝ × ℝ := (2 + 3 * t, 3 + 2 * t)
noncomputable def line_m (s : ℝ) : ℝ × ℝ := (4 + 2 * s, 5 + 3 * s)

def is_perpendicular (v w : ℝ × ℝ) : Prop :=
  v.1 * w.1 + v.2 * w.2 = 0

theorem vector_projection_condition 
  (t s : ℝ)
  (C : ℝ × ℝ := line_l t)
  (D : ℝ × ℝ := line_m s)
  (Q : ℝ × ℝ)
  (hQ : is_perpendicular (Q.1 - C.1, Q.2 - C.2) (2, 3))
  (v1 v2 : ℝ)
  (hv_sum : v1 + v2 = 3)
  (hv_def : ∃ k : ℝ, v1 = 3 * k ∧ v2 = -2 * k)
  : (v1, v2) = (9, -6) := 
sorry

end vector_projection_condition_l224_224087


namespace lattice_points_in_triangle_271_l224_224285

def Pick_theorem (N L S : ℝ) : Prop :=
  S = N + (1/2) * L - 1

noncomputable def lattice_triangle_inside_points (A B O : ℕ × ℕ) : ℕ :=
  271

theorem lattice_points_in_triangle_271 :
  ∃ N L : ℝ, 
  let S := 300 in
  let L := 60 in
  A = (0, 30) ∧ B = (20, 10) ∧
  O = (0, 0) ∧ Pick_theorem N L S :=
sorry

end lattice_points_in_triangle_271_l224_224285


namespace ce_external_angle_bisector_intersect_circumcircle_l224_224598

noncomputable def right_triangle (ABC : Triangle) : Prop :=
  ∃ (A B C : Point) (a b c : ℝ),
    Right_Triangle ABC A B C ∧ Angle.C_right.name A B C

theorem ce_external_angle_bisector_intersect_circumcircle (ABC : Triangle) (A B C D E : Point) :
  right_triangle ABC →
  angle_bisector A B C D →
  midpoint A D E →
  ∃ F : Point, lies_on_circumcircle ABC F ∧ ce_intersects_external_angle_bisector A E F :=
begin
  sorry

end ce_external_angle_bisector_intersect_circumcircle_l224_224598


namespace CE_squared_plus_DE_squared_l224_224622

noncomputable def radius : ℝ := 6 * real.sqrt 3
noncomputable def BE : ℝ := 4 * real.sqrt 6
noncomputable def angle_AEC : ℝ := 60

theorem CE_squared_plus_DE_squared (radius : ℝ) (BE : ℝ) (angle_AEC : ℝ) (h_radius : radius = 6 * real.sqrt 3) (h_BE : BE = 4 * real.sqrt 6) (h_angle : angle_AEC = 60) : CE^2 + DE^2 = 216 :=
by
  sorry

end CE_squared_plus_DE_squared_l224_224622


namespace apple_counting_l224_224069

theorem apple_counting
  (n m : ℕ)
  (vasya_trees_a_b petya_trees_a_b vasya_trees_b_c petya_trees_b_c vasya_trees_c_d petya_trees_c_d vasya_apples_a_b petya_apples_a_b vasya_apples_c_d petya_apples_c_d : ℕ)
  (h1 : petya_trees_a_b = 2 * vasya_trees_a_b)
  (h2 : petya_apples_a_b = 7 * vasya_apples_a_b)
  (h3 : petya_trees_b_c = 2 * vasya_trees_b_c)
  (h4 : petya_trees_c_d = 2 * vasya_trees_c_d)
  (h5 : n = vasya_trees_a_b + petya_trees_a_b)
  (h6 : m = vasya_apples_a_b + petya_apples_a_b)
  (h7 : vasya_trees_c_d = n / 3)
  (h8 : petya_trees_c_d = 2 * (n / 3))
  (h9 : vasya_apples_c_d = 3 * petya_apples_c_d)
  : vasya_apples_c_d = 3 * petya_apples_c_d :=
by 
  sorry

end apple_counting_l224_224069


namespace bases_with_final_digit_one_l224_224494

theorem bases_with_final_digit_one : 
  let bases := [2, 3, 4, 5, 6, 7, 8, 9]
  let divisors_of_624 := [2, 3, 4, 8]
  ∃ (b : List ℕ), b = divisors_of_624 ∧ ∀ b ∈ b, b ∈ bases ∧ 624 % b = 0 :=
sorry

end bases_with_final_digit_one_l224_224494


namespace balloon_arrangement_count_l224_224975

-- Definitions of letter frequencies for the word BALLOON
def letter_frequencies : List (Char × Nat) := [('B', 1), ('A', 1), ('L', 2), ('O', 2), ('N', 1)]

-- Total number of letters
def total_letters := 7

-- The formula for the number of unique arrangements of the letters
noncomputable def arrangements :=
  (Nat.factorial total_letters) / 
  (Nat.factorial 1 * Nat.factorial 1 * Nat.factorial 2 * Nat.factorial 2 * Nat.factorial 1)

-- The theorem to prove the number of ways to arrange the letters in "BALLOON"
theorem balloon_arrangement_count : arrangements = 1260 :=
  sorry

end balloon_arrangement_count_l224_224975


namespace cube_surface_area_l224_224529

theorem cube_surface_area (V : ℝ) (hV : V = 64) : ∃ S : ℝ, S = 96 := 
by
  sorry

end cube_surface_area_l224_224529


namespace find_3x2y2_l224_224997

theorem find_3x2y2 (x y : ℤ) 
  (h1 : y^2 + 3 * x^2 * y^2 = 30 * x^2 + 517) : 
  3 * x^2 * y^2 = 588 := by
  sorry

end find_3x2y2_l224_224997


namespace balloon_permutations_l224_224945

theorem balloon_permutations : 
  let total_letters := 7
  let repetitions_L := 2
  let repetitions_O := 2
  (Nat.factorial total_letters) / ((Nat.factorial repetitions_L) * (Nat.factorial repetitions_O)) = 1260 := 
by
  sorry

end balloon_permutations_l224_224945


namespace odd_integer_95th_l224_224735

theorem odd_integer_95th : (2 * 95 - 1) = 189 := 
by
  -- The proof would go here
  sorry

end odd_integer_95th_l224_224735


namespace balloon_permutations_l224_224939

theorem balloon_permutations : 
  let balloons := "BALLOON".toList.length in
  let total_permutations := Nat.factorial balloons in
  let repetitive_L := Nat.factorial 2 in
  let repetitive_O := Nat.factorial 2 in
  (total_permutations / (repetitive_L * repetitive_O)) = 1260 := sorry

end balloon_permutations_l224_224939


namespace Rachelle_meat_needed_l224_224663

theorem Rachelle_meat_needed (meat_per_hamburger : ℝ)
    (five_pounds : meat_per_hamburger = 5 / 10)
    (hamburgers_needed : ℕ) :
    hamburgers_needed = 30 → 
    meat_per_hamburger * 30 = 15 := by
  intros
  rw [five_pounds]
  linarith
  sorry

end Rachelle_meat_needed_l224_224663


namespace energy_ratio_l224_224648

-- Define the energy release function E
def E (x : ℝ) : ℝ := 10^x

-- Define the frequency of earthquakes function f
variable (f : ℝ → ℝ)

-- Define the given conditions
axiom freq_condition : f 5 = 2 * f 3

-- Theorem that we need to prove
theorem energy_ratio (hf : f 5 = 2 * f 3) : E 5 / E 3 = 200 := 
  by
  -- Since E(x) = 10^x, we can directly compute E(5) / E(3)
  have h1 : E 5 = 10^5 := rfl
  have h2 : E 3 = 10^3 := rfl
  calc
    E 5 / E 3 = (10^5) / (10^3)   : by rw [h1, h2]
            ... = 10^(5 - 3)       : by rw [←div_eq_pow_sub]
            ... = 10^2             : by norm_num
            ... = 100              : by norm_num
            ... = 200              : sorry

end energy_ratio_l224_224648


namespace graph_no_4_cycles_bound_l224_224624

-- Define a graph structure
structure Graph (V : Type) :=
  (edges : V → V → Prop)
  (no4Cycle : ∀ (a b c d : V), (edges a b ∧ edges b c ∧ edges c d ∧ edges d a) → false)

variables {V : Type} [Fintype V]

def vertex_count (G : Graph V) : ℕ := Fintype.card V

def edge_count (G : Graph V) : ℕ := 
  Fintype.card ({e : V × V // G.edges e.fst e.snd})

theorem graph_no_4_cycles_bound (G : Graph V) (n m : ℕ) 
  (hV : vertex_count G = n) 
  (hE : edge_count G = m) :
  m ≤ n / 4 * (1 + real.sqrt (4 * n - 3)) := 
sorry

end graph_no_4_cycles_bound_l224_224624


namespace gravitational_constant_significant_digits_l224_224055

theorem gravitational_constant_significant_digits 
  (G : ℝ) (uncertainty : ℝ)
  (H1 : G = 6.67384)
  (H2 : uncertainty = 0.00021) :
  let G_upper := G + uncertainty,
      G_lower := G - uncertainty in
  (Real.floor (G_upper * 1000) / 1000 = 6.674)
  ∧ (Real.floor (G_lower * 1000) / 1000 = 6.674) :=
by
  sorry

end gravitational_constant_significant_digits_l224_224055


namespace hyperbola_asymptotes_l224_224159

def hyperbola (x y : ℝ) : Prop := (x^2 / 8) - (y^2 / 2) = 1

theorem hyperbola_asymptotes (x y : ℝ) :
  hyperbola x y → (y = (1/2) * x ∨ y = - (1/2) * x) :=
by
  sorry

end hyperbola_asymptotes_l224_224159


namespace repeating_decimal_sum_l224_224467

noncomputable def repeating_decimal_four : ℚ := 0.44444 -- 0.\overline{4}
noncomputable def repeating_decimal_seven : ℚ := 0.77777 -- 0.\overline{7}

-- Proving that the sum of these repeating decimals is equivalent to the fraction 11/9.
theorem repeating_decimal_sum : repeating_decimal_four + repeating_decimal_seven = 11/9 := by
  -- Placeholder to skip the actual proof
  sorry

end repeating_decimal_sum_l224_224467


namespace balloon_arrangements_l224_224985

def factorial (n : Nat) : Nat :=
  if n = 0 then 1 else n * factorial (n - 1)

def arrangements (n : Nat) (ks : List Nat) :=
  factorial n / (ks.map factorial).foldr (*) 1

theorem balloon_arrangements : arrangements 7 [2, 2] = 1260 :=
by
  sorry

end balloon_arrangements_l224_224985


namespace necklace_cost_l224_224273

theorem necklace_cost (N : ℕ) (h1 : N + (N + 5) = 73) : N = 34 := by
  sorry

end necklace_cost_l224_224273


namespace pure_imaginary_solutions_l224_224099

theorem pure_imaginary_solutions:
  ∀ (x : ℂ), (x.im ≠ 0 ∧ x.re = 0) → (x ^ 4 - 5 * x ^ 3 + 10 * x ^ 2 - 50 * x - 75 = 0)
         → (x = Complex.I * Real.sqrt 10 ∨ x = -Complex.I * Real.sqrt 10) :=
by
  sorry

end pure_imaginary_solutions_l224_224099


namespace balloon_arrangement_count_l224_224974

-- Definitions of letter frequencies for the word BALLOON
def letter_frequencies : List (Char × Nat) := [('B', 1), ('A', 1), ('L', 2), ('O', 2), ('N', 1)]

-- Total number of letters
def total_letters := 7

-- The formula for the number of unique arrangements of the letters
noncomputable def arrangements :=
  (Nat.factorial total_letters) / 
  (Nat.factorial 1 * Nat.factorial 1 * Nat.factorial 2 * Nat.factorial 2 * Nat.factorial 1)

-- The theorem to prove the number of ways to arrange the letters in "BALLOON"
theorem balloon_arrangement_count : arrangements = 1260 :=
  sorry

end balloon_arrangement_count_l224_224974


namespace shortest_rope_length_l224_224691

variable {x : ℕ}
variable {l1 l2 l3 : ℕ}

-- Defining the lengths of the ropes based on the given ratio
def length1 := 4 * x
def length2 := 5 * x
def length3 := 6 * x

-- Condition given in the problem
def condition : Prop := length1 + length3 = length2 + 100

-- Theorem stating that if the condition holds, then l1 must be 80
theorem shortest_rope_length (h : condition) : length1 = 80 :=
by sorry

end shortest_rope_length_l224_224691


namespace directrix_of_parabola_l224_224318

theorem directrix_of_parabola (p : ℝ) : by { assume h : y² = 4 * p * x, sorry } :=
assume h₁ : y² = 2 * x,
have hp : p = 1 / 2 , from sorry,
have directrix_eq : x = -p, from sorry,
show x = -1 / 2, from sorry

end directrix_of_parabola_l224_224318


namespace find_a5_in_arithmetic_progression_l224_224199

variable {α : Type*} [AddCommGroup α] [Module ℕ α] [Zero α]

-- Definitions based on the conditions
def is_arithmetic_progression (a : ℕ → α) : Prop :=
  ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)

def sum_in_arithmetic_progression (a : ℕ → α) :=
  a 3 + a 4 + a 5 + a 6 + a 7

-- Statement of the problem
theorem find_a5_in_arithmetic_progression (a : ℕ → α)
  (h_arith : is_arithmetic_progression a)
  (h_sum : sum_in_arithmetic_progression a = 45) :
  a 5 = 9 :=
sorry

end find_a5_in_arithmetic_progression_l224_224199


namespace repeating_decimal_to_fraction_l224_224778

theorem repeating_decimal_to_fraction : (0.5656565656 : ℚ) = 56 / 99 :=
  by
    sorry

end repeating_decimal_to_fraction_l224_224778


namespace f_is_decreasing_on_0_to_pi_l224_224697

-- Define the function
def f (x : ℝ) : ℝ := cos x - x

-- State the condition of function differentiability and the condition on the interval (0, π)
theorem f_is_decreasing_on_0_to_pi : 
  ∀ x ∈ Ioo 0 π, deriv f x < 0 :=
by
  -- Proof omitted
  intro x hx
  sorry

end f_is_decreasing_on_0_to_pi_l224_224697


namespace area_parallelogram_l224_224253

def vec1 : ℝ × ℝ := (6, -4)
def vec2 : ℝ × ℝ := (13, -1)

def parallelogram_area (v w : ℝ × ℝ) : ℝ :=
  Float.abs (v.1 * w.2 - v.2 * w.1)

theorem area_parallelogram : parallelogram_area vec1 vec2 = 46 := by
  sorry

end area_parallelogram_l224_224253


namespace area_of_ABC_is_50_l224_224201

noncomputable def area_of_triangle (a b c : ℝ) : ℝ :=
  (1 / 2) * a * b

theorem area_of_ABC_is_50 (A B C : Type) [metric_space A] [metric_space B] [metric_space C]
  (angle_A : A) (angle_B : B) (angle_C : C)
  (h_right_triangle : angle_C = 90)
  (h_isosceles : angle_A = angle_B)
  (h_AB : dist A B = 10 * real.sqrt 2) :
  area_of_triangle (dist A C) (dist B C) = 50 :=
begin
  sorry
end

end area_of_ABC_is_50_l224_224201


namespace area_ratio_ABC_ABM_cos_angle_AMB_locus_includes_orthocenter_l224_224184

-- Define the vectors MA, MB, and MC
variables {V : Type*} [add_comm_group V] [vector_space ℝ V]
variables (A B C M : V)

-- Condition for option B and C
variables (hB : (A - M) + 2 * (B - M) + 3 * (C - M) = 0)
variables (hC : M is the orthocenter of triangle ABC) 

-- Condition for option D
variables (λ : ℝ)
variables (hD : (A - M) = λ * ( (B - A) / (|| B - A || * cos (angle A B C)) + (C - A) / (|| C - A || * cos (angle A B C)) ))

-- Proof Statements for B, C and D
theorem area_ratio_ABC_ABM (hB : (A - M) + 2 * (B - M) + 3 * (C - M) = 0) : 
  area (triangle A B C) = 2 * area (triangle A B M) := 
sorry

theorem cos_angle_AMB (hB : (A - M) + 2 * (B - M) + 3 * (C - M) = 0) (hC : M is the orthocenter of triangle ABC) :
  cos angle (A M B) = -sqrt(10) / 10 :=
sorry

theorem locus_includes_orthocenter (hD : (A - M) = λ * ( (B - A) / (|| B - A || * cos (angle A B C)) + (C - A) / (|| C - A || * cos (angle A B C)) )) : 
  M passes through the orthocenter of triangle ABC :=
sorry

end area_ratio_ABC_ABM_cos_angle_AMB_locus_includes_orthocenter_l224_224184


namespace perp_tan_alpha_parallel_alpha_l224_224552

open Real

variable {α : ℝ} (h₀ : α ∈ set.Ioo 0 π)
variable (a : Vector ℝ 2) (b : Vector ℝ 2)
def a := ⟨[sin (α + π / 6), 3]⟩
def b := ⟨[1, 4 * cos α]⟩

theorem perp_tan_alpha (h₁ : dot_product a b = 0) : 
  tan α = -25 * sqrt 3 / 3 := 
begin
  sorry
end

theorem parallel_alpha (h₂ : (∃ k : ℝ, b = k • a)) : 
  α = π / 6 := 
begin
  sorry
end

end perp_tan_alpha_parallel_alpha_l224_224552


namespace ensure_one_of_each_color_l224_224376

theorem ensure_one_of_each_color
  (red_socks blue_socks green_socks khaki_socks : ℕ)
  (h_r : red_socks = 10)
  (h_b : blue_socks = 20)
  (h_g : green_socks = 30)
  (h_k : khaki_socks = 40) :
  ∃ n, n = 91 ∧
       ∀ (socks : finset ℕ), (multiset.card socks) = n → 
       (∀ c, c ∈ socks → c = 10 ∨ c = 20 ∨ c = 30 ∨ c = 40) → 
       (∃ (r b g k : ℕ), r ≤ 10 ∧ b ≤ 20 ∧ g ≤ 30 ∧ k ≤ 40 ∧ 
                          r ≥ 1 ∧ b ≥ 1 ∧ g ≥ 1 ∧ k ≥ 1) :=
by
  -- Here we should provide the proof, but we'll place 'sorry' as instructed.
  sorry

end ensure_one_of_each_color_l224_224376


namespace balloon_permutation_count_l224_224921

-- Define the necessary factorials and the formula for permutations of multiset
def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem balloon_permutation_count : 
  let total_letters := 7,
      repeat_L := 2,
      repeat_O := 2,
      unique_arrangements := factorial total_letters / (factorial repeat_L * factorial repeat_O) 
  in unique_arrangements = 1260 :=
by
  sorry

end balloon_permutation_count_l224_224921


namespace balloon_permutations_l224_224952

theorem balloon_permutations : 
  let total_letters := 7
  let repetitions_L := 2
  let repetitions_O := 2
  (Nat.factorial total_letters) / ((Nat.factorial repetitions_L) * (Nat.factorial repetitions_O)) = 1260 := 
by
  sorry

end balloon_permutations_l224_224952


namespace lark_lock_combination_count_l224_224613

-- Definitions for the conditions
def is_odd (n : ℕ) : Prop := n % 2 = 1
def is_even (n : ℕ) : Prop := n % 2 = 0
def is_multiple_of_5 (n : ℕ) : Prop := n % 5 = 0

def lark_lock_combination (a b c : ℕ) : Prop := 
  is_odd a ∧ is_even b ∧ is_multiple_of_5 c ∧ 1 ≤ a ∧ a ≤ 30 ∧ 1 ≤ b ∧ b ≤ 30 ∧ 1 ≤ c ∧ c ≤ 30

-- The core theorem
theorem lark_lock_combination_count : 
  (∃ a b c : ℕ, lark_lock_combination a b c) ↔ (15 * 15 * 6 = 1350) :=
by
  sorry

end lark_lock_combination_count_l224_224613


namespace partition_weights_l224_224716

theorem partition_weights :
  ∃ A B C : Finset ℕ,
    (∀ x ∈ A, x ≤ 552) ∧
    (∀ x ∈ B, x ≤ 552) ∧
    (∀ x ∈ C, x ≤ 552) ∧
    ∀ x, (x ∈ A ∨ x ∈ B ∨ x ∈ C) ↔ 1 ≤ x ∧ x ≤ 552 ∧
    A ∩ B = ∅ ∧ B ∩ C = ∅ ∧ A ∩ C = ∅ ∧
    A.sum id = 50876 ∧ B.sum id = 50876 ∧ C.sum id = 50876 :=
by
  sorry

end partition_weights_l224_224716


namespace hyperbola_area_l224_224640

theorem hyperbola_area {F1 F2 P : Type}
  (hyperbola_eq : ∀ (x y : ℝ), x^2 - y^2 / 8 = 1)
  (on_hyperbola : P)
  (perpendicular : PF₁ ⊥ PF₂) :
  area_triangle PF₁ F₂ = 8 :=
sorry

end hyperbola_area_l224_224640


namespace repeated_term_sequence_l224_224833

theorem repeated_term_sequence (a : ℝ) :
  (∀ n : ℕ, (λ n, a) n - (λ n, a) (n + 1) = 0) ∧ 
  (∀ n : ℕ, a ≠ 0 → (λ n, a) (n + 1) / (λ n, a) n = 1) ∧ 
  (a = 0 → ∀ n : ℕ, (λ n, a) (n + 1) / (λ n, a) n = 1) :=
by
  sorry

end repeated_term_sequence_l224_224833


namespace y_coordinate_of_given_point_l224_224597

noncomputable def line_through_points_and_x_intercept (P Q : Point ℝ) (x_int : ℝ) : RealLinearMap ℝ ℝ :=
let m := (Q.y - P.y) / (Q.x - P.x) in
let b := - m * x_int in
λ x, m * x + b

theorem y_coordinate_of_given_point :
  let P : Point ℝ := (⟨4, 0⟩ : Point ℝ)
  let Q : Point ℝ := (⟨10, 3⟩ : Point ℝ)
  let x_intercept : ℝ := 4
  let line := line_through_points_and_x_intercept P Q x_intercept in
  @line (-8) = -6 := sorry

end y_coordinate_of_given_point_l224_224597


namespace apple_pear_box_difference_l224_224718

theorem apple_pear_box_difference :
  let initial_apples := 25
  let initial_pears := 12
  let additional_apples := 8
  let additional_pears := 8
  let final_apples := initial_apples + additional_apples
  let final_pears := initial_pears + additional_pears
  final_apples - final_pears = 13 :=
by {
  let initial_apples := 25
  let initial_pears := 12
  let additional_apples := 8
  let additional_pears := 8
  let final_apples := initial_apples + additional_apples
  let final_pears := initial_pears + additional_pears
  have h1 : final_apples = 33 := by rfl -- 25 + 8 = 33
  have h2 : final_pears = 20 := by rfl -- 12 + 8 = 20
  calc
    final_apples - final_pears = 33 - 20 : by rw [h1, h2]
                        ... = 13         : by rfl
}

end apple_pear_box_difference_l224_224718


namespace mutually_exclusive_not_opposite_l224_224401

namespace event_theory

-- Definition to represent the student group
structure Group where
  boys : ℕ
  girls : ℕ

def student_group : Group := {boys := 3, girls := 2}

-- Definition of events
inductive Event
| AtLeastOneBoyAndOneGirl
| ExactlyOneBoyExactlyTwoBoys
| AtLeastOneBoyAllGirls
| AtMostOneBoyAllGirls

open Event

-- Conditions provided in the problem
def condition (grp : Group) : Prop :=
  grp.boys = 3 ∧ grp.girls = 2

-- The main statement to prove in Lean
theorem mutually_exclusive_not_opposite :
  condition student_group →
  ∃ e₁ e₂ : Event, e₁ = ExactlyOneBoyExactlyTwoBoys ∧ e₂ = ExactlyOneBoyExactlyTwoBoys ∧ (
    (e₁ ≠ e₂) ∧ (¬ (e₁ = e₂ ∧ e₁ = ExactlyOneBoyExactlyTwoBoys))
  ) :=
by
  sorry

end event_theory

end mutually_exclusive_not_opposite_l224_224401


namespace four_digit_numbers_with_avg_condition_l224_224558

theorem four_digit_numbers_with_avg_condition : 
  ∃ (n : ℕ), n = 3360 ∧ 
  (∀ (a b c d : ℕ), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ 
    0 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 ∧ 0 ≤ d ∧ d ≤ 9 ∧ 
    (b = (a + c)/2 ∨ c = (a + b)/2 ∨ a = (b + c)/2) ∧
    (1000 * a + 100 * b + 10 * c + d ≥ 1000 ∨ 1000 * b + 100 * a + 10 * c + d ≥ 1000 ∨ 
     1000 * c + 100 * b + 10 * a + d ≥ 1000 ∨ 1000 * d + 100 * b + 10 * c + a ≥ 1000)) :=
begin
  sorry
end

end four_digit_numbers_with_avg_condition_l224_224558


namespace true_statements_count_l224_224168

variables {Ω : Type*} [MeasureTheory.ProbabilityMeasure Ω]
variables {A B : Event Ω}

theorem true_statements_count (h : P (A ∩ B) = P A * P B) :
  (P (¬A ∩ B) = P (¬A) * P B) ∧ (P (A ∩ ¬B) = P A * P (¬B)) ∧ (P (¬A ∩ ¬B) = P (¬A) * P (¬B)) :=
by
  sorry

end true_statements_count_l224_224168


namespace range_of_x_l224_224527

-- Define the quadratic function f(x) = x^2 + (a - 4)x + 4 - 2a
def f (x a : ℝ) : ℝ := x^2 + (a - 4) * x + 4 - 2 * a

-- Define the condition for a being in the interval [-1, 1]
def a_in_bounds (a : ℝ) : Prop := a ≥ -1 ∧ a ≤ 1

theorem range_of_x {x a : ℝ} (h : a_in_bounds a) : 
  (∀ a ∈ Icc (-1:ℝ) (1:ℝ), f x a > 0) → (x < 1 ∨ x > 3) :=
sorry

end range_of_x_l224_224527


namespace balloon_permutations_count_l224_224903

-- Definitions of the conditions
def total_letters_count : ℕ := 7
def l_count : ℕ := 2
def o_count : ℕ := 2

-- Now the mathematical problem as a Lean statement
theorem balloon_permutations_count : 
  (Nat.factorial total_letters_count) / ((Nat.factorial l_count) * (Nat.factorial o_count)) = 1260 := 
by
  sorry

end balloon_permutations_count_l224_224903


namespace count_valid_numbers_l224_224562

def is_three_digit (n : ℕ) : Prop := n >= 100 ∧ n <= 999
def contains_digit (n : ℕ) (d : ℕ) : Prop := d ∈ n.digits 10
def contains_7 (n : ℕ) : Prop := contains_digit n 7
def contains_2 (n : ℕ) : Prop := contains_digit n 2
def no_2 (n : ℕ) : Prop := ¬ contains_2 n
def valid_number (n : ℕ) : Prop := is_three_digit n ∧ contains_7 n ∧ no_2 n

theorem count_valid_numbers : Fintype.card {n // valid_number n} = 200 :=
sorry

end count_valid_numbers_l224_224562


namespace total_edges_after_10_cuts_l224_224418

theorem total_edges_after_10_cuts 
  (initial_edges : ℕ) (cuts : ℕ) (edge_increase_per_cut : ℕ) 
  (initial_edges_eq : initial_edges = 4) 
  (cuts_eq : cuts = 10) 
  (edge_increase_per_cut_eq : edge_increase_per_cut = 3) : 
  initial_edges + cuts * edge_increase_per_cut = 34 :=
by 
  rw [initial_edges_eq, cuts_eq, edge_increase_per_cut_eq]
  sorry

end total_edges_after_10_cuts_l224_224418


namespace balloon_arrangements_l224_224980

def factorial (n : Nat) : Nat :=
  if n = 0 then 1 else n * factorial (n - 1)

def arrangements (n : Nat) (ks : List Nat) :=
  factorial n / (ks.map factorial).foldr (*) 1

theorem balloon_arrangements : arrangements 7 [2, 2] = 1260 :=
by
  sorry

end balloon_arrangements_l224_224980


namespace width_of_foil_covered_prism_l224_224802

theorem width_of_foil_covered_prism :
  ∃ (m : ℝ), (∃ (l : ℝ), l^3 = 32 ∧ ∃ (w : ℝ), w = 4 * l ∧ m = w + 2) ∧ m ≈ 7.04 :=
sorry

end width_of_foil_covered_prism_l224_224802


namespace balloon_arrangements_l224_224888

noncomputable def factorial (n : ℕ) : ℕ :=
if h : 0 < n then n * factorial (n - 1) else 1

theorem balloon_arrangements : 
  ∃ (n : ℕ), n = (factorial 7) / (factorial 2 * factorial 2) ∧ n = 1260 := by
  have fact_7 := factorial 7
  have fact_2 := factorial 2
  exists (fact_7 / (fact_2 * fact_2))
  split
  · rw [← int.coe_nat_eq_coe_nat_iff]
    norm_cast
    sorry -- Here we do the exact calculations
  · exact rfl

end balloon_arrangements_l224_224888


namespace prime_numbers_for_1002_n_l224_224495

noncomputable def is_prime(n : ℕ) : Prop := nat.prime n

theorem prime_numbers_for_1002_n : 
    {n : ℕ | n ≥ 2 ∧ is_prime (n^3 + 2)}.finite ∧ 
    ({n : ℕ | n ≥ 2 ∧ is_prime (n^3 + 2)}.to_finset.card = 2) := 
by
  sorry

end prime_numbers_for_1002_n_l224_224495


namespace area_of_shaded_region_l224_224837

theorem area_of_shaded_region (α : ℝ) (hα1 : 0 < α ∧ α < 90) (hα2 : Real.cos α = 3 / 5) : 
  let side_length := 1,
  let sin_α := Real.sqrt (1 - (Real.cos α)^2),
  let tan_half_α := sin_α / (1 + Real.cos α),
  let tan_diff := (1 - tan_half_α) / (1 + tan_half_α),
  let area_triangle := 1 / 2 * tan_diff^2,
  2 * area_triangle = 1 / 9 :=
by
  let side_length := 1
  let sin_α := Real.sqrt (1 - (Real.cos α)^2)
  let tan_half_α := sin_α / (1 + Real.cos α)
  let tan_diff := (1 - tan_half_α) / (1 + tan_half_α)
  let area_triangle := 1 / 2 * tan_diff^2
  sorry

end area_of_shaded_region_l224_224837


namespace triangle_area_ratio_l224_224630

noncomputable def vector₃ := (ℝ × ℝ × ℝ)

noncomputable def area_ratio (A B C P : vector₃) : ℝ :=
  let PA := (fst P - fst A, snd P - snd A, thd P - thd A)
  let PB := (fst P - fst B, snd P - snd B, thd P - thd B)
  let PC := (fst P - fst C, snd P - snd C, thd P - thd C)
  -- Following the condition: PA + 3PB + 4PC = 0
  if PA + (3:ℝ) * PB + (4:ℝ) * PC = (0, 0, 0) then (5/3:ℝ) else 0

variables {A B C P : vector₃}

theorem triangle_area_ratio (h : (fst P - fst A, snd P - snd A, thd P - thd A) +
                                (3:ℝ) * ((fst P - fst B, snd P - snd B, thd P - thd B)) +
                                (4:ℝ) * ((fst P - fst C, snd P - snd C, thd P - thd C)) = (0, 0, 0)) :
  area_ratio A B C P = (5/3:ℝ) :=
sorry

end triangle_area_ratio_l224_224630


namespace recurring_decimal_to_fraction_correct_l224_224762

noncomputable def recurring_decimal_to_fraction (b : ℚ) : Prop :=
  b = 0.\overline{56} ↔ b = 56/99

theorem recurring_decimal_to_fraction_correct : recurring_decimal_to_fraction 0.\overline{56} :=
  sorry

end recurring_decimal_to_fraction_correct_l224_224762


namespace average_of_scores_with_average_twice_l224_224425

variable (scores: List ℝ) (A: ℝ) (A': ℝ)
variable (h1: scores.length = 50)
variable (h2: A = (scores.sum) / 50)
variable (h3: A' = ((scores.sum + 2 * A) / 52))

theorem average_of_scores_with_average_twice (h1: scores.length = 50) (h2: A = (scores.sum) / 50) (h3: A' = ((scores.sum + 2 * A) / 52)) :
  A' = A :=
by
  sorry

end average_of_scores_with_average_twice_l224_224425


namespace recurring_to_fraction_l224_224752

theorem recurring_to_fraction : ∀ (x : ℚ), x = 0.5656 ∧ 100 * x = 56.5656 → x = 56 / 99 :=
by
  intros x hx
  sorry

end recurring_to_fraction_l224_224752


namespace balloon_arrangements_l224_224897

theorem balloon_arrangements :
  let n := 7
  let k1 := 2
  let k2 := 2
  (Nat.factorial n) / ((Nat.factorial k1) * (Nat.factorial k2)) = 1260 :=
by
  -- Use the conditions directly from the problem
  let n := 7
  let k1 := 2
  let k2 := 2
  -- Mathematically, the number of arrangements of the multiset
  have h : (Nat.factorial n) / ((Nat.factorial k1) * (Nat.factorial k2)) = 1260 := sorry
  exact h

end balloon_arrangements_l224_224897


namespace balloon_arrangements_l224_224883

noncomputable def factorial (n : ℕ) : ℕ :=
if h : 0 < n then n * factorial (n - 1) else 1

theorem balloon_arrangements : 
  ∃ (n : ℕ), n = (factorial 7) / (factorial 2 * factorial 2) ∧ n = 1260 := by
  have fact_7 := factorial 7
  have fact_2 := factorial 2
  exists (fact_7 / (fact_2 * fact_2))
  split
  · rw [← int.coe_nat_eq_coe_nat_iff]
    norm_cast
    sorry -- Here we do the exact calculations
  · exact rfl

end balloon_arrangements_l224_224883


namespace peter_food_necessity_l224_224657

/-- Discuss the conditions  -/
def peter_horses (num_horses num_days : ℕ) (oats_per_meal grain_per_day : ℕ) (meals_per_day : ℕ) : ℕ :=
  let daily_oats := oats_per_meal * meals_per_day in
  let total_oats := daily_oats * num_days * num_horses in
  let total_grain := grain_per_day * num_days * num_horses in
  total_oats + total_grain

/-- Prove that Peter needs 132 pounds of food to feed his horses for 3 days -/
theorem peter_food_necessity : peter_horses 4 3 4 3 2 = 132 :=
  sorry

end peter_food_necessity_l224_224657


namespace equilateral_triangle_height_base_2_l224_224683

noncomputable def sqrt3 : ℝ := Real.sqrt 3

def equilateral_triangle_height (side : ℝ) : ℝ := (Real.sqrt 3) / 2 * side

theorem equilateral_triangle_height_base_2 : 
  equilateral_triangle_height 2 = sqrt3 := 
by 
  sorry

end equilateral_triangle_height_base_2_l224_224683


namespace ratio_pow_eq_l224_224792

variable (a b c d e f p q r : ℝ)
variable (n : ℕ)
variable (h : a / b = c / d)
variable (h1 : a / b = e / f)
variable (h2 : p ≠ 0 ∨ q ≠ 0 ∨ r ≠ 0)

theorem ratio_pow_eq
  (h : a / b = c / d)
  (h1 : a / b = e / f)
  (h2 : p ≠ 0 ∨ q ≠ 0 ∨ r ≠ 0)
  (n_ne_zero : n ≠ 0):
  (a / b) ^ n = (p * a ^ n + q * c ^ n + r * e ^ n) / (p * b ^ n + q * d ^ n + r * f ^ n) :=
by
  sorry

end ratio_pow_eq_l224_224792


namespace parallel_line_through_intersection_perpendicular_line_through_intersection_l224_224117

/-- Given two lines l1: x + y - 4 = 0 and l2: x - y + 2 = 0,
the line passing through their intersection point and parallel to the line 2x - y - 1 = 0 
is 2x - y + 1 = 0 --/
theorem parallel_line_through_intersection :
  ∃ (c : ℝ), ∃ (x y : ℝ), (x + y - 4 = 0 ∧ x - y + 2 = 0) ∧ (2 * x - y + c = 0) ∧ c = 1 :=
by
  sorry

/-- Given two lines l1: x + y - 4 = 0 and l2: x - y + 2 = 0,
the line passing through their intersection point and perpendicular to the line 2x - y - 1 = 0
is x + 2y - 7 = 0 --/
theorem perpendicular_line_through_intersection :
  ∃ (d : ℝ), ∃ (x y : ℝ), (x + y - 4 = 0 ∧ x - y + 2 = 0) ∧ (x + 2 * y + d = 0) ∧ d = -7 :=
by
  sorry

end parallel_line_through_intersection_perpendicular_line_through_intersection_l224_224117


namespace angle_C_of_triangle_area_l224_224158

theorem angle_C_of_triangle_area (a b c : ℝ) (S : ℝ) (hS : S = (a^2 + b^2 - c^2) / 4) :
  ∠ of_triangle_area a b c = π / 4 :=
sorry

end angle_C_of_triangle_area_l224_224158


namespace balloon_permutation_count_l224_224915

-- Define the necessary factorials and the formula for permutations of multiset
def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem balloon_permutation_count : 
  let total_letters := 7,
      repeat_L := 2,
      repeat_O := 2,
      unique_arrangements := factorial total_letters / (factorial repeat_L * factorial repeat_O) 
  in unique_arrangements = 1260 :=
by
  sorry

end balloon_permutation_count_l224_224915


namespace arrangement_count_27_arrangement_count_26_l224_224565

open Int

def valid_arrangement_count (n : ℕ) : ℕ :=
  if n = 27 then 14 else if n = 26 then 105 else 0

theorem arrangement_count_27 : valid_arrangement_count 27 = 14 :=
  by
    sorry

theorem arrangement_count_26 : valid_arrangement_count 26 = 105 :=
  by
    sorry

end arrangement_count_27_arrangement_count_26_l224_224565


namespace ball_total_distance_on_10th_touch_l224_224817

theorem ball_total_distance_on_10th_touch :
  let H₀ := 100
  let total_distance (n : ℕ) : ℝ :=
    if n = 0 then H₀ 
    else H₀ + 2 * ∑ i in finset.range n, H₀ * (1 / 2) ^ i
  in total_distance 10 = 100 + 200 * (1 - (1 / 2) ^ 9) :=
by
  sorry

end ball_total_distance_on_10th_touch_l224_224817


namespace probability_of_events_l224_224725

-- Define the sets of tiles in each box
def boxA : Set ℕ := {n | 1 ≤ n ∧ n ≤ 25}
def boxB : Set ℕ := {n | 15 ≤ n ∧ n ≤ 40}

-- Define the specific conditions
def eventA (tile : ℕ) : Prop := tile ≤ 20
def eventB (tile : ℕ) : Prop := (Odd tile ∨ tile > 35)

-- Define the probabilities as calculations
def prob_eventA : ℚ := 20 / 25
def prob_eventB : ℚ := 15 / 26

-- The final probability given independence
def combined_prob : ℚ := prob_eventA * prob_eventB

-- The theorem statement we want to prove
theorem probability_of_events :
  combined_prob = 6 / 13 := 
by 
  -- proof details would go here
  sorry

end probability_of_events_l224_224725


namespace find_x_l224_224031

theorem find_x
  (x : ℝ)
  (h : (x + 1) / (x + 5) = (x + 5) / (x + 13)) :
  x = 3 :=
sorry

end find_x_l224_224031


namespace anna_should_plant_8_lettuce_plants_l224_224075

/-- Anna wants to grow some lettuce in the garden and would like to grow enough to have at least
    12 large salads.
- Conditions:
  1. Half of the lettuce will be lost to insects and rabbits.
  2. Each lettuce plant is estimated to provide 3 large salads.
  
  Proof that Anna should plant 8 lettuce plants in the garden. --/
theorem anna_should_plant_8_lettuce_plants 
    (desired_salads: ℕ)
    (salads_per_plant: ℕ)
    (loss_fraction: ℚ) :
    desired_salads = 12 →
    salads_per_plant = 3 →
    loss_fraction = 1 / 2 →
    ∃ plants: ℕ, plants = 8 :=
by
  intros h1 h2 h3
  sorry

end anna_should_plant_8_lettuce_plants_l224_224075


namespace probability_even_divisible_by_5_distinct_digits_l224_224073

def is_even (n : ℕ) : Prop := n % 2 = 0

def is_divisible_by_5 (n : ℕ) : Prop := n % 5 = 0

def digits_distinct (n : ℕ) : Prop :=
  let d := n.digits 10 in d.nodup

def satisfies_conditions (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999 ∧ is_even n ∧ is_divisible_by_5 n ∧ digits_distinct n

theorem probability_even_divisible_by_5_distinct_digits :
  let total := 9000
  let favourable := 504
  (favourable / total : ℚ) = 7 / 125 :=
by {
  sorry
}

end probability_even_divisible_by_5_distinct_digits_l224_224073


namespace complex_to_polar_l224_224531

open Complex

noncomputable def theta_value (z : ℂ) : ℝ := if z = -1 - sqrt 3 * Complex.i then (4 * Real.pi) / 3 else 0

theorem complex_to_polar (z : ℂ) (h : z = -1 - sqrt 3 * Complex.i) : ∃ r θ, z = r * Complex.exp (θ * Complex.i) ∧ θ = theta_value z :=
by
  sorry

end complex_to_polar_l224_224531


namespace sum_of_digits_of_palindrome_l224_224409

theorem sum_of_digits_of_palindrome (x : ℕ) (h1 : 100 ≤ x ∧ x ≤ 999) (h2 : (x + 27) ≥ 1000 ∧ (x + 27) ≤ 1026) (h3 : 
  let str_x := x.digits 10 in
  let rev_str_x := (x.digits 10).reverse in
  str_x = rev_str_x
) (h4 : 
  let str_x_plus_27 := (x + 27).digits 10 in
  let rev_str_x_plus_27 := ((x + 27).digits 10).reverse in
  str_x_plus_27 = rev_str_x_plus_27
) : 
  (x.digits 10).sum = 20 :=
by 
  sorry

end sum_of_digits_of_palindrome_l224_224409


namespace different_rhetorical_device_in_optionA_l224_224846

def optionA_uses_metaphor : Prop :=
  -- Here, define the condition explaining that Option A uses metaphor
  true -- This will denote that Option A uses metaphor 

def optionsBCD_use_personification : Prop :=
  -- Here, define the condition explaining that Options B, C, and D use personification
  true -- This will denote that Options B, C, and D use personification

theorem different_rhetorical_device_in_optionA :
  optionA_uses_metaphor ∧ optionsBCD_use_personification → 
  (∃ (A P : Prop), A ≠ P) :=
by
  -- No proof is required as per instructions
  intro h
  exact Exists.intro optionA_uses_metaphor (Exists.intro optionsBCD_use_personification sorry)

end different_rhetorical_device_in_optionA_l224_224846


namespace repeating_decimal_fraction_l224_224782

theorem repeating_decimal_fraction : ∀ x : ℚ, (x = 0.5656565656565656) → 100 * x = 56.5656565656565656 → 100 * x - x = 56.5656565656565656 - 0.5656565656565656
  → 99 * x = 56 → x = 56 / 99 :=
begin
  intros x h1 h2 h3 h4,
  sorry,
end

end repeating_decimal_fraction_l224_224782


namespace arctic_circle_spherical_distance_l224_224272

noncomputable def spherical_distance (A B : Point) (R : ℝ) : ℝ := sorry

theorem arctic_circle_spherical_distance (A B : Point) (R : ℝ) 
  (arc_length_AB : arc_length A B = π * R / 2) 
  (latitude : ArcticCircle.latitude = 60) :
  spherical_distance A B R = π * R / 3 := 
sorry

end arctic_circle_spherical_distance_l224_224272


namespace imaginary_part_is_one_l224_224329

def imaginary_part_of_quotient : ℂ :=
  let z := (2 * complex.I) / (1 + complex.I) in
  complex.im z

theorem imaginary_part_is_one : imaginary_part_of_quotient = 1 by
  sorry

end imaginary_part_is_one_l224_224329


namespace incenter_excenter_of_ODM_l224_224786

-- Step by Step Defining All the Conditions

noncomputable def point : Type := ℝ × ℝ

structure Triangle :=
(A B C : point)
(height_from_A_to_BC : point)
(incenter_ABD incenter_ACD : point)
(circumcircle_AI1I2 : set point)
(intersection_AB_AC : point × point)
(intersection_EF_BC : point)

-- Define the given triangle ABC with all the conditions
def ABC : Triangle :=
{ A := (0, 0),
  B := (1, 0),
  C := (0, 1),
  height_from_A_to_BC := (0, 1),
  incenter_ABD := (0.5, 0),
  incenter_ACD := (0, 0.5),
  circumcircle_AI1I2 := { p : point | ((p.1 - 0.5)^2 + (p.2 - 0.5)^2) = (1 / 2)^2 },
  intersection_AB_AC := ((0.5, 0.5), (0, 1)),
  intersection_EF_BC := (0.25, 0.75) }

-- The theorem to prove
theorem incenter_excenter_of_ODM (△ABC : Triangle) : 
  let I1 := △ABC.incenter_ABD,
      I2 := △ABC.incenter_ACD,
      O := (0, 1), -- Circumcircle AI1I2's center (just for defining)
      M := △ABC.intersection_EF_BC,
      D := (0, 1) -- height from A to BC (just for defining) in the right triangle
  in
  (is_incenter I1 (D, O, M) ∧ is_excenter I2 (D, O, M))
:=
sorry

end incenter_excenter_of_ODM_l224_224786


namespace meeting_time_is_10_38_am_l224_224230

-- Definitions for conditions
def jenny_leaves_time : ℕ := 8 * 60  -- 8:00 AM in minutes
def zack_leaves_time : ℕ := 8 * 60 + 30  -- 8:30 AM in minutes
def jenny_speed : ℝ := 15.0  -- Jenny's speed in miles per hour
def zack_speed : ℝ := 18.0  -- Zack's speed in miles per hour
def distance_between_towns : ℝ := 78.0  -- Distance between towns A and B in miles

-- The time in minutes they meet
def meeting_time : ℝ := jenny_leaves_time + 60 * ((87 / 33) : ℝ)

theorem meeting_time_is_10_38_am 
  (jenny_leaves_time zack_leaves_time : ℕ)
  (jenny_speed zack_speed distance_between_towns : ℝ) : 
  meeting_time = 10 * 60 + 38 :=
by
  sorry

end meeting_time_is_10_38_am_l224_224230


namespace find_k_l224_224628

theorem find_k
  (k x1 x2 : ℝ)
  (h1 : x1^2 - 3*x1 + k = 0)
  (h2 : x2^2 - 3*x2 + k = 0)
  (h3 : x1 = 2 * x2) :
  k = 2 :=
sorry

end find_k_l224_224628


namespace new_volume_is_64_l224_224394

variable (r h : ℝ)
variable (V_original : ℝ) (V' : ℝ)

-- Conditions
def cylinder_volume (r h : ℝ) : ℝ := π * r^2 * h
def original_volume_condition : Prop := cylinder_volume r h = 2
def height_double : ℝ := 2 * h
def radius_quadruple : ℝ := 4 * r

-- New volume after dimension changes
def new_volume : ℝ := cylinder_volume radius_quadruple height_double

-- Statement: Prove the new volume is 64 gallons
theorem new_volume_is_64 (h1 : original_volume_condition) : new_volume = 64 := 
sorry

end new_volume_is_64_l224_224394


namespace tom_has_1_dollar_left_l224_224003

/-- Tom has $19 and each folder costs $2. After buying as many folders as possible,
Tom will have $1 left. -/
theorem tom_has_1_dollar_left (initial_money : ℕ) (folder_cost : ℕ) (folders_bought : ℕ) (money_left : ℕ) 
  (h1 : initial_money = 19)
  (h2 : folder_cost = 2)
  (h3 : folders_bought = initial_money / folder_cost)
  (h4 : money_left = initial_money - folders_bought * folder_cost) :
  money_left = 1 :=
by
  -- proof will be provided here
  sorry

end tom_has_1_dollar_left_l224_224003


namespace find_values_of_m_and_n_l224_224518

theorem find_values_of_m_and_n (m n : ℝ) (h : m / (1 + I) = 1 - n * I) : 
  m = 2 ∧ n = 1 :=
sorry

end find_values_of_m_and_n_l224_224518


namespace lucy_average_speed_l224_224262

noncomputable def total_distance : ℝ := 420 + 480
noncomputable def time_first_segment : ℝ := 7 + (15 / 60) + (30 / 3600)
noncomputable def time_second_segment : ℝ := 8 + (20 / 60) + (45 / 3600)
noncomputable def total_time : ℝ := time_first_segment + time_second_segment
noncomputable def average_speed : ℝ := total_distance / total_time

theorem lucy_average_speed :
  average_speed ≈ 57.69 := sorry

end lucy_average_speed_l224_224262


namespace arithmetic_sequence_100_l224_224157

variable {a : ℕ → ℝ}

def is_arithmetic_sequence (a : ℕ → ℝ) :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

variables (S₉ : ℝ) (a₁₀ : ℝ)

theorem arithmetic_sequence_100
  (h1: is_arithmetic_sequence a)
  (h2: S₉ = 27) 
  (h3: a₁₀ = 8): 
  a 100 = 98 := 
sorry

end arithmetic_sequence_100_l224_224157


namespace balloon_permutations_count_l224_224902

-- Definitions of the conditions
def total_letters_count : ℕ := 7
def l_count : ℕ := 2
def o_count : ℕ := 2

-- Now the mathematical problem as a Lean statement
theorem balloon_permutations_count : 
  (Nat.factorial total_letters_count) / ((Nat.factorial l_count) * (Nat.factorial o_count)) = 1260 := 
by
  sorry

end balloon_permutations_count_l224_224902


namespace reach_any_natural_number_l224_224437

theorem reach_any_natural_number (n : ℕ) : ∃ (f : ℕ → ℕ), f 0 = 1 ∧ (∀ k, f (k + 1) = 3 * f k + 1 ∨ f (k + 1) = f k / 2) ∧ (∃ m, f m = n) := by
  sorry

end reach_any_natural_number_l224_224437


namespace another_representation_l224_224475

def positive_int_set : Set ℕ := {x | x > 0}

theorem another_representation :
  {x ∈ positive_int_set | x - 3 < 2} = {1, 2, 3, 4} :=
by
  sorry

end another_representation_l224_224475


namespace area_of_trapezoid_l224_224305

theorem area_of_trapezoid {a b c d e f g h o r : ℝ}
  (square_in_circle : ∀ (A B C D : ℝ), A * A + B * B = r * r ∧ A * B = b * B ∧ b = a)
  (vertices_positions : ((2 * a) = 8) ∧ 4 = a ∧ r = 4 * sqrt 2)
  (trapezoid_conditions : b = 4 ∧ a = 8)
  (right_angle_trapezoid : ∠(e, g, h) = π / 2):
  area e f g h = 24 * sqrt 2 :=
begin
  sorry
end

end area_of_trapezoid_l224_224305


namespace intersection_point_line_ellipse_l224_224202

noncomputable def sqrt_two := Real.sqrt 2

theorem intersection_point_line_ellipse :
  ∃ (x y : ℝ), (x = 2 * y - sqrt_two) ∧ (x^2 + 4 * y^2 = 1) ∧ (x = -sqrt_two / 2) ∧ (y = sqrt_two / 4) :=
begin
  use [-sqrt_two / 2, sqrt_two / 4],
  split,
  { simp [sqrt_two], ring },
  split,
  { simp [sqrt_two], ring_exp },
  split,
  { refl },
  { refl }
end

end intersection_point_line_ellipse_l224_224202


namespace min_value_is_nine_exists_minimum_value_expression_l224_224097

noncomputable def min_expression_value : Real :=
  (λ x : Real, (Real.sin x + Real.csc x + Real.tan x)^2 + (Real.cos x + Real.sec x + Real.cot x)^2)

theorem min_value_is_nine_exists :
  ∃ x : Real, 0 < x ∧ x < Real.pi / 2 ∧ min_expression_value x = 9 :=
by
  sorry

theorem minimum_value_expression :
  ∀ x : Real, 0 < x ∧ x < Real.pi / 2 → min_expression_value x ≥ 9 :=
by
  sorry

end min_value_is_nine_exists_minimum_value_expression_l224_224097


namespace find_g_neg3_l224_224623

def f (x : ℚ) : ℚ := 4 * x - 6
def g (u : ℚ) : ℚ := 3 * (f u)^2 + 4 * (f u) - 2

theorem find_g_neg3 : g (-3) = 43 / 16 := by
  sorry

end find_g_neg3_l224_224623
