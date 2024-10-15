import Mathlib

namespace NUMINAMATH_GPT_ratio_of_m_div_x_l653_65342

theorem ratio_of_m_div_x (a b : ℝ) (h1 : a / b = 4 / 5) (h2 : a > 0) (h3 : b > 0) :
  let x := a + 0.25 * a
  let m := b - 0.60 * b
  (m / x) = 2 / 5 :=
by
  -- Define x and m
  let x := a + 0.25 * a
  let m := b - 0.60 * b
  -- Include the steps or assumptions here if necessary
  sorry

end NUMINAMATH_GPT_ratio_of_m_div_x_l653_65342


namespace NUMINAMATH_GPT_part1_part2_l653_65341

noncomputable section

variables (a x : ℝ)

def P : Prop := x^2 - 4*a*x + 3*a^2 < 0
def Q : Prop := abs (x - 3) ≤ 1

-- Part 1: If a=1 and P ∨ Q, prove the range of x is 1 < x ≤ 4
theorem part1 (h1 : a = 1) (h2 : P a x ∨ Q x) : 1 < x ∧ x ≤ 4 :=
sorry

-- Part 2: If ¬P is necessary but not sufficient for ¬Q, prove the range of a is 4/3 ≤ a ≤ 2
theorem part2 (h : (¬P a x → ¬Q x) ∧ (¬Q x → ¬P a x → False)) : 4/3 ≤ a ∧ a ≤ 2 :=
sorry

end NUMINAMATH_GPT_part1_part2_l653_65341


namespace NUMINAMATH_GPT_compute_g_neg_101_l653_65377

variable (g : ℝ → ℝ)

def functional_eqn := ∀ x y : ℝ, g (x * y) + 2 * x = x * g y + g x
def g_neg_one := g (-1) = 3
def g_one := g (1) = 1

theorem compute_g_neg_101 (g : ℝ → ℝ)
  (H1 : functional_eqn g)
  (H2 : g_neg_one g)
  (H3 : g_one g) :
  g (-101) = 103 := 
by
  sorry

end NUMINAMATH_GPT_compute_g_neg_101_l653_65377


namespace NUMINAMATH_GPT_matrix_identity_l653_65382

noncomputable def B : Matrix (Fin 2) (Fin 2) ℝ := !![2, -1; 4, 3]
noncomputable def I : Matrix (Fin 2) (Fin 2) ℝ := 1

theorem matrix_identity :
  B^4 = -3 • B + 2 • I :=
by
  sorry

end NUMINAMATH_GPT_matrix_identity_l653_65382


namespace NUMINAMATH_GPT_length_of_GH_l653_65358

variable (S_A S_C S_E S_F : ℝ)
variable (AB FE CD GH : ℝ)

-- Given conditions
axiom h1 : AB = 11
axiom h2 : FE = 13
axiom h3 : CD = 5

-- Relationships between the sizes of the squares
axiom h4 : S_A = S_C + AB
axiom h5 : S_C = S_E + CD
axiom h6 : S_E = S_F + FE
axiom h7 : GH = S_A - S_F

theorem length_of_GH : GH = 29 :=
by
  -- This is where the proof would go
  sorry

end NUMINAMATH_GPT_length_of_GH_l653_65358


namespace NUMINAMATH_GPT_total_beds_in_hotel_l653_65381

theorem total_beds_in_hotel (total_rooms : ℕ) (rooms_two_beds rooms_three_beds : ℕ) (beds_two beds_three : ℕ) 
  (h1 : total_rooms = 13) 
  (h2 : rooms_two_beds = 8) 
  (h3 : rooms_three_beds = total_rooms - rooms_two_beds) 
  (h4 : beds_two = 2) 
  (h5 : beds_three = 3) : 
  rooms_two_beds * beds_two + rooms_three_beds * beds_three = 31 :=
by
  sorry

end NUMINAMATH_GPT_total_beds_in_hotel_l653_65381


namespace NUMINAMATH_GPT_find_area_of_triangle_l653_65370

noncomputable def triangle_area (a b: ℝ) (cosC: ℝ) : ℝ :=
  let sinC := Real.sqrt (1 - cosC^2)
  0.5 * a * b * sinC

theorem find_area_of_triangle :
  ∀ (a b cosC : ℝ), a = 3 * Real.sqrt 2 → b = 2 * Real.sqrt 3 → cosC = 1 / 3 →
  triangle_area a b cosC = 4 * Real.sqrt 3 :=
by
  intros a b cosC ha hb hcosC
  rw [ha, hb, hcosC]
  sorry

end NUMINAMATH_GPT_find_area_of_triangle_l653_65370


namespace NUMINAMATH_GPT_brenda_ends_with_15_skittles_l653_65333

def initial_skittles : ℕ := 7
def skittles_bought : ℕ := 8

theorem brenda_ends_with_15_skittles : initial_skittles + skittles_bought = 15 := 
by {
  sorry
}

end NUMINAMATH_GPT_brenda_ends_with_15_skittles_l653_65333


namespace NUMINAMATH_GPT_regular_discount_rate_l653_65331

theorem regular_discount_rate (MSRP : ℝ) (s : ℝ) (sale_price : ℝ) (d : ℝ) :
  MSRP = 35 ∧ s = 0.20 ∧ sale_price = 19.6 → d = 0.3 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_regular_discount_rate_l653_65331


namespace NUMINAMATH_GPT_max_knights_between_knights_l653_65389

def num_knights : ℕ := 40
def num_samurais : ℕ := 10
def total_people : ℕ := 50
def num_knights_with_samurai_right : ℕ := 7

theorem max_knights_between_knights :
  (num_knights - num_knights_with_samurai_right + 1) = 32 :=
sorry

end NUMINAMATH_GPT_max_knights_between_knights_l653_65389


namespace NUMINAMATH_GPT_part1_part2_l653_65330

open Real

variables (α : ℝ) (A : (ℝ × ℝ)) (B : (ℝ × ℝ)) (C : (ℝ × ℝ))

def points_coordinates : Prop :=
A = (3, 0) ∧ B = (0, 3) ∧ C = (cos α, sin α) ∧ π / 2 < α ∧ α < 3 * π / 2

theorem part1 (h : points_coordinates α A B C) (h1 : dist (3, 0) (cos α, sin α) = dist (0, 3) (cos α, sin α)) : 
  α = 5 * π / 4 :=
sorry

theorem part2 (h : points_coordinates α A B C) (h2 : ((cos α - 3) * cos α + (sin α) * (sin α - 3)) = -1) : 
  (2 * sin α * sin α + sin (2 * α)) / (1 + tan α) = -5 / 9 :=
sorry

end NUMINAMATH_GPT_part1_part2_l653_65330


namespace NUMINAMATH_GPT_Carmela_difference_l653_65309

theorem Carmela_difference (Cecil Catherine Carmela : ℤ) (X : ℤ) (h1 : Cecil = 600) 
(h2 : Catherine = 2 * Cecil - 250) (h3 : Carmela = 2 * Cecil + X) 
(h4 : Cecil + Catherine + Carmela = 2800) : X = 50 :=
by { sorry }

end NUMINAMATH_GPT_Carmela_difference_l653_65309


namespace NUMINAMATH_GPT_seq_fixed_point_l653_65384

theorem seq_fixed_point (a_0 b_0 : ℝ) (a b : ℕ → ℝ)
  (h1 : a 0 = a_0)
  (h2 : b 0 = b_0)
  (h3 : ∀ n, a (n + 1) = a n + b n)
  (h4 : ∀ n, b (n + 1) = a n * b n) :
  a 2022 = a_0 ∧ b 2022 = b_0 ↔ b_0 = 0 := sorry

end NUMINAMATH_GPT_seq_fixed_point_l653_65384


namespace NUMINAMATH_GPT_part_a_part_b_l653_65323

-- Part (a) Equivalent Proof Problem
theorem part_a (k : ℤ) : 
  ∃ a b c : ℤ, 3 * k - 2 = a ^ 2 + b ^ 3 + c ^ 3 := 
sorry

-- Part (b) Equivalent Proof Problem
theorem part_b (n : ℤ) : 
  ∃ a b c d : ℤ, n = a ^ 2 + b ^ 3 + c ^ 3 + d ^ 3 := 
sorry

end NUMINAMATH_GPT_part_a_part_b_l653_65323


namespace NUMINAMATH_GPT_ratio_ac_bd_l653_65346

theorem ratio_ac_bd (a b c d : ℝ) (h1 : a = 4 * b) (h2 : b = 2 * c) (h3 : c = 5 * d) : 
  (a * c) / (b * d) = 20 :=
by
  sorry

end NUMINAMATH_GPT_ratio_ac_bd_l653_65346


namespace NUMINAMATH_GPT_function_passes_through_fixed_point_l653_65336

theorem function_passes_through_fixed_point (a : ℝ) (h₀ : 0 < a) (h₁ : a ≠ 1) : (2 - a^(0 : ℝ) = 1) :=
by
  sorry

end NUMINAMATH_GPT_function_passes_through_fixed_point_l653_65336


namespace NUMINAMATH_GPT_find_certain_number_l653_65332

theorem find_certain_number 
  (x : ℝ) 
  (h : ( (x + 2 - 6) * 3 ) / 4 = 3) 
  : x = 8 :=
by
  sorry

end NUMINAMATH_GPT_find_certain_number_l653_65332


namespace NUMINAMATH_GPT_minimum_value_of_polynomial_l653_65307

def polynomial (a b : ℝ) : ℝ := 2 * a^2 - 8 * a * b + 17 * b^2 - 16 * a + 4 * b + 1999

theorem minimum_value_of_polynomial : ∃ (a b : ℝ), polynomial a b = 1947 :=
by
  sorry

end NUMINAMATH_GPT_minimum_value_of_polynomial_l653_65307


namespace NUMINAMATH_GPT_trapezoid_area_l653_65325

theorem trapezoid_area:
  let vert1 := (10, 10)
  let vert2 := (15, 15)
  let vert3 := (0, 15)
  let vert4 := (0, 10)
  let base1 := 10
  let base2 := 15
  let height := 5
  ∃ (area : ℝ), area = 62.5 := by
  sorry

end NUMINAMATH_GPT_trapezoid_area_l653_65325


namespace NUMINAMATH_GPT_complex_equation_square_sum_l653_65316

-- Lean 4 statement of the mathematical proof problem
theorem complex_equation_square_sum (a b : ℝ) (i : ℂ) (h : i^2 = -1) 
    (h1 : (a - 2 * i) * i = b - i) : a^2 + b^2 = 5 := by
  sorry

end NUMINAMATH_GPT_complex_equation_square_sum_l653_65316


namespace NUMINAMATH_GPT_find_original_number_l653_65373

theorem find_original_number
  (x : ℤ)
  (h : 3 * (2 * x + 5) = 123) :
  x = 18 := 
sorry

end NUMINAMATH_GPT_find_original_number_l653_65373


namespace NUMINAMATH_GPT_value_of_y_l653_65314

theorem value_of_y (x y : ℤ) (h1 : x^2 - 3 * x + 6 = y + 2) (h2 : x = -8) : y = 92 :=
by
  sorry

end NUMINAMATH_GPT_value_of_y_l653_65314


namespace NUMINAMATH_GPT_range_of_a_l653_65329

theorem range_of_a (a : ℝ) : (-1/3 ≤ a) ∧ (a ≤ 2/3) ↔ (∀ x y : ℝ, (0 ≤ x ∧ x ≤ 1) → (0 ≤ y ∧ y ≤ 1) → y = a * x + 1/3) :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l653_65329


namespace NUMINAMATH_GPT_embankment_building_l653_65391

theorem embankment_building (days : ℕ) (workers_initial : ℕ) (workers_later : ℕ) (embankments : ℕ) :
  workers_initial = 75 → days = 4 → embankments = 2 →
  (∀ r : ℚ, embankments = workers_initial * r * days →
            embankments = workers_later * r * 5) :=
by
  intros h75 hd4 h2 r hr
  sorry

end NUMINAMATH_GPT_embankment_building_l653_65391


namespace NUMINAMATH_GPT_solve_for_q_l653_65312

theorem solve_for_q (p q : ℚ) (h1 : 5 * p + 3 * q = 7) (h2 : 3 * p + 5 * q = 8) : q = 19 / 16 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_q_l653_65312


namespace NUMINAMATH_GPT_range_f_iff_l653_65363

noncomputable def f (m x : ℝ) : ℝ :=
  Real.log ((m^2 - 3 * m + 2) * x^2 + 2 * (m - 1) * x + 5)

theorem range_f_iff (m : ℝ) :
  (∀ y ∈ Set.univ, ∃ x, f m x = y) ↔ (m = 1 ∨ (2 < m ∧ m ≤ 9/4)) := 
by
  sorry

end NUMINAMATH_GPT_range_f_iff_l653_65363


namespace NUMINAMATH_GPT_nat_pow_eq_sub_two_case_l653_65306

theorem nat_pow_eq_sub_two_case (n : ℕ) : (∃ a k : ℕ, k ≥ 2 ∧ 2^n - 1 = a^k) ↔ (n = 0 ∨ n = 1) :=
by
  sorry

end NUMINAMATH_GPT_nat_pow_eq_sub_two_case_l653_65306


namespace NUMINAMATH_GPT_range_of_a_l653_65356

theorem range_of_a (a : ℝ) :
  (∃ x : ℤ, 2 * (x : ℝ) - 1 > 3 ∧ x ≤ a) ∧ (∀ x : ℤ, 2 * (x : ℝ) - 1 > 3 → x ≤ a) → 5 ≤ a ∧ a < 6 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l653_65356


namespace NUMINAMATH_GPT_quadrilateral_is_trapezoid_l653_65328

variables {V : Type*} [AddCommGroup V] [Module ℝ V] -- Define the type of vectors and vector space over the reals
variables (a b : V) -- Vectors a and b
variables (AB BC CD AD : V) -- Vectors representing sides of quadrilateral

-- Condition: vectors a and b are not collinear
def not_collinear (a b : V) : Prop := ∀ k : ℝ, k ≠ 0 → a ≠ k • b

-- Given Conditions
def conditions (a b AB BC CD : V) : Prop :=
  AB = a + 2 • b ∧
  BC = -4 • a - b ∧
  CD = -5 • a - 3 • b ∧
  not_collinear a b

-- The to-be-proven property
def is_trapezoid (AB BC CD AD : V) : Prop :=
  AD = 2 • BC

theorem quadrilateral_is_trapezoid 
  (a b AB BC CD : V) 
  (h : conditions a b AB BC CD)
  : is_trapezoid AB BC CD (AB + BC + CD) :=
sorry

end NUMINAMATH_GPT_quadrilateral_is_trapezoid_l653_65328


namespace NUMINAMATH_GPT_anna_discontinued_coaching_on_2nd_august_l653_65303

theorem anna_discontinued_coaching_on_2nd_august
  (coaching_days : ℕ) (non_leap_year : ℕ) (first_day : ℕ) 
  (days_in_january : ℕ) (days_in_february : ℕ) (days_in_march : ℕ) 
  (days_in_april : ℕ) (days_in_may : ℕ) (days_in_june : ℕ) 
  (days_in_july : ℕ) (days_in_august : ℕ)
  (not_leap_year : non_leap_year = 365)
  (first_day_of_year : first_day = 1)
  (january_days : days_in_january = 31)
  (february_days : days_in_february = 28)
  (march_days : days_in_march = 31)
  (april_days : days_in_april = 30)
  (may_days : days_in_may = 31)
  (june_days : days_in_june = 30)
  (july_days : days_in_july = 31)
  (august_days : days_in_august = 31)
  (total_coaching_days : coaching_days = 245) :
  ∃ day, day = 2 ∧ month = "August" := 
sorry

end NUMINAMATH_GPT_anna_discontinued_coaching_on_2nd_august_l653_65303


namespace NUMINAMATH_GPT_meaningful_range_l653_65388

   noncomputable def isMeaningful (x : ℝ) : Prop :=
     (3 - x ≥ 0) ∧ (x + 1 ≠ 0)

   theorem meaningful_range :
     ∀ x : ℝ, isMeaningful x ↔ (x ≤ 3 ∧ x ≠ -1) :=
   by
     sorry
   
end NUMINAMATH_GPT_meaningful_range_l653_65388


namespace NUMINAMATH_GPT_cube_painted_surface_l653_65326

theorem cube_painted_surface (n : ℕ) (hn : n > 2) 
: 6 * (n - 2) ^ 2 = (n - 2) ^ 3 → n = 8 :=
by
  sorry

end NUMINAMATH_GPT_cube_painted_surface_l653_65326


namespace NUMINAMATH_GPT_eq_infinite_solutions_pos_int_l653_65359

noncomputable def eq_has_inf_solutions_in_positive_integers (m : ℕ) : Prop :=
    ∀ (a b c : ℕ), 
    a > 0 ∧ b > 0 ∧ c > 0 → 
    ∃ (a' b' c' : ℕ), 
    a' > 0 ∧ b' > 0 ∧ c' > 0 ∧ 
    (1 / a + 1 / b + 1 / c + 1 / (a * b * c) = m / (a + b + c))

theorem eq_infinite_solutions_pos_int (m : ℕ) (hm : m > 0) : eq_has_inf_solutions_in_positive_integers m := 
by 
  sorry

end NUMINAMATH_GPT_eq_infinite_solutions_pos_int_l653_65359


namespace NUMINAMATH_GPT_largest_number_l653_65348

theorem largest_number (a b : ℕ) (hcf_ab : Nat.gcd a b = 42) (h_dvd_a : 42 ∣ a) (h_dvd_b : 42 ∣ b)
  (a_eq : a = 42 * 11) (b_eq : b = 42 * 12) : max a b = 504 := by
  sorry

end NUMINAMATH_GPT_largest_number_l653_65348


namespace NUMINAMATH_GPT_exists_idempotent_l653_65375

-- Definition of the set M as the natural numbers from 1 to 1993
def M : Set ℕ := { n | 1 ≤ n ∧ n ≤ 1993 }

-- Operation * on M
noncomputable def star (a b : ℕ) : ℕ := sorry

-- Hypothesis: * is closed on M and (a * b) * a = b for any a, b in M
axiom star_closed (a b : ℕ) (ha : a ∈ M) (hb : b ∈ M) : star a b ∈ M
axiom star_property (a b : ℕ) (ha : a ∈ M) (hb : b ∈ M) : star (star a b) a = b

-- Goal: Prove that there exists a number a in M such that a * a = a
theorem exists_idempotent : ∃ a ∈ M, star a a = a := by
  sorry

end NUMINAMATH_GPT_exists_idempotent_l653_65375


namespace NUMINAMATH_GPT_gcd_consecutive_terms_l653_65327

theorem gcd_consecutive_terms (n : ℕ) : 
  Nat.gcd (2 * Nat.factorial n + n) (2 * Nat.factorial (n + 1) + (n + 1)) = 1 :=
by
  sorry

end NUMINAMATH_GPT_gcd_consecutive_terms_l653_65327


namespace NUMINAMATH_GPT_real_mul_eq_zero_iff_l653_65352

theorem real_mul_eq_zero_iff (a b : ℝ) (h : a * b = 0) : a = 0 ∨ b = 0 :=
sorry

end NUMINAMATH_GPT_real_mul_eq_zero_iff_l653_65352


namespace NUMINAMATH_GPT_find_ab_l653_65362

theorem find_ab (a b : ℝ) (h1 : a - b = 10) (h2 : a^2 + b^2 = 150) : a * b = 25 :=
by 
  sorry

end NUMINAMATH_GPT_find_ab_l653_65362


namespace NUMINAMATH_GPT_max_popsicles_with_10_dollars_l653_65385

theorem max_popsicles_with_10_dollars :
  (∃ (single_popsicle_cost : ℕ) (four_popsicle_box_cost : ℕ) (six_popsicle_box_cost : ℕ) (budget : ℕ),
    single_popsicle_cost = 1 ∧
    four_popsicle_box_cost = 3 ∧
    six_popsicle_box_cost = 4 ∧
    budget = 10 ∧
    ∃ (max_popsicles : ℕ),
      max_popsicles = 14 ∧
      ∀ (popsicles : ℕ),
        popsicles ≤ 14 →
        ∃ (x y z : ℕ),
          popsicles = x + 4*y + 6*z ∧
          x * single_popsicle_cost + y * four_popsicle_box_cost + z * six_popsicle_box_cost ≤ budget
  ) :=
sorry

end NUMINAMATH_GPT_max_popsicles_with_10_dollars_l653_65385


namespace NUMINAMATH_GPT_books_left_correct_l653_65365

variable (initial_books : ℝ) (sold_books : ℝ)

def number_of_books_left (initial_books sold_books : ℝ) : ℝ :=
  initial_books - sold_books

theorem books_left_correct :
  number_of_books_left 51.5 45.75 = 5.75 :=
by
  sorry

end NUMINAMATH_GPT_books_left_correct_l653_65365


namespace NUMINAMATH_GPT_cost_of_items_l653_65343

theorem cost_of_items (M R F : ℝ)
  (h1 : 10 * M = 24 * R) 
  (h2 : F = 2 * R) 
  (h3 : F = 20.50) : 
  4 * M + 3 * R + 5 * F = 231.65 := 
by
  sorry

end NUMINAMATH_GPT_cost_of_items_l653_65343


namespace NUMINAMATH_GPT_num_arrangements_thm1_num_arrangements_thm2_num_arrangements_thm3_l653_65386

open Nat

def num_arrangements_A_middle (n : ℕ) : ℕ :=
  if n = 4 then factorial 4 else 0

def num_arrangements_A_not_adj_B (n : ℕ) : ℕ :=
  if n = 5 then (factorial 3) * (factorial 4 / factorial 2) else 0

def num_arrangements_A_B_not_ends (n : ℕ) : ℕ :=
  if n = 5 then (factorial 3 / factorial 2) * factorial 3 else 0

theorem num_arrangements_thm1 : num_arrangements_A_middle 4 = 24 := 
  sorry

theorem num_arrangements_thm2 : num_arrangements_A_not_adj_B 5 = 72 := 
  sorry

theorem num_arrangements_thm3 : num_arrangements_A_B_not_ends 5 = 36 := 
  sorry

end NUMINAMATH_GPT_num_arrangements_thm1_num_arrangements_thm2_num_arrangements_thm3_l653_65386


namespace NUMINAMATH_GPT_fraction_meaningful_l653_65310

theorem fraction_meaningful (x : ℝ) : x - 3 ≠ 0 ↔ x ≠ 3 :=
by sorry

end NUMINAMATH_GPT_fraction_meaningful_l653_65310


namespace NUMINAMATH_GPT_ratio_of_combined_area_to_combined_perimeter_l653_65378

noncomputable def equilateral_triangle_area (s : ℝ) : ℝ :=
  (s^2 * Real.sqrt 3) / 4

noncomputable def equilateral_triangle_perimeter (s : ℝ) : ℝ :=
  3 * s

theorem ratio_of_combined_area_to_combined_perimeter :
  (equilateral_triangle_area 6 + equilateral_triangle_area 8) / 
  (equilateral_triangle_perimeter 6 + equilateral_triangle_perimeter 8) = (25 * Real.sqrt 3) / 42 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_combined_area_to_combined_perimeter_l653_65378


namespace NUMINAMATH_GPT_parity_sum_matches_parity_of_M_l653_65396

theorem parity_sum_matches_parity_of_M (N M : ℕ) (even_numbers odd_numbers : ℕ → ℤ)
  (hn : ∀ i, i < N → even_numbers i % 2 = 0)
  (hm : ∀ i, i < M → odd_numbers i % 2 ≠ 0) : 
  (N + M) % 2 = M % 2 := 
sorry

end NUMINAMATH_GPT_parity_sum_matches_parity_of_M_l653_65396


namespace NUMINAMATH_GPT_polynomial_even_iff_exists_Q_l653_65397

open Polynomial

noncomputable def exists_polynomial_Q (P : Polynomial ℂ) : Prop :=
  ∃ Q : Polynomial ℂ, ∀ z : ℂ, P.eval z = (Q.eval z) * (Q.eval (-z))

theorem polynomial_even_iff_exists_Q (P : Polynomial ℂ) :
  (∀ z : ℂ, P.eval z = P.eval (-z)) ↔ exists_polynomial_Q P :=
by 
  sorry

end NUMINAMATH_GPT_polynomial_even_iff_exists_Q_l653_65397


namespace NUMINAMATH_GPT_girls_25_percent_less_false_l653_65308

theorem girls_25_percent_less_false (g b : ℕ) (h : b = g * 125 / 100) : (b - g) / b ≠ 25 / 100 := by
  sorry

end NUMINAMATH_GPT_girls_25_percent_less_false_l653_65308


namespace NUMINAMATH_GPT_second_player_wins_when_2003_candies_l653_65337

def game_winning_strategy (n : ℕ) : ℕ :=
  if n % 2 = 0 then 1 else 2

theorem second_player_wins_when_2003_candies :
  game_winning_strategy 2003 = 2 :=
by 
  sorry

end NUMINAMATH_GPT_second_player_wins_when_2003_candies_l653_65337


namespace NUMINAMATH_GPT_average_height_40_girls_l653_65372

/-- Given conditions for a class of 50 students, where the average height of 40 girls is H,
    the average height of the remaining 10 girls is 167 cm, and the average height of the whole
    class is 168.6 cm, prove that the average height H of the 40 girls is 169 cm. -/
theorem average_height_40_girls (H : ℝ)
  (h1 : 0 < H)
  (h2 : (40 * H + 10 * 167) = 50 * 168.6) :
  H = 169 :=
by
  sorry

end NUMINAMATH_GPT_average_height_40_girls_l653_65372


namespace NUMINAMATH_GPT_range_of_f_l653_65395

noncomputable def f (x : ℝ) : ℝ := x^2 - 2 * x - 1

theorem range_of_f : 
  ∀ x, -1 ≤ x ∧ x ≤ 1 → -2 ≤ f x ∧ f x ≤ 2 :=
by
  intro x Hx
  sorry

end NUMINAMATH_GPT_range_of_f_l653_65395


namespace NUMINAMATH_GPT_sin_half_alpha_l653_65347

theorem sin_half_alpha (α : ℝ) (h_cos : Real.cos α = -2/3) (h_range : π < α ∧ α < 3 * π / 2) :
  Real.sin (α / 2) = Real.sqrt 30 / 6 :=
by
  sorry

end NUMINAMATH_GPT_sin_half_alpha_l653_65347


namespace NUMINAMATH_GPT_find_principal_amount_l653_65394

noncomputable def principal_amount_loan (SI R T : ℝ) : ℝ :=
  SI / (R * T)

theorem find_principal_amount (SI R T : ℝ) (h_SI : SI = 6480) (h_R : R = 0.12) (h_T : T = 3) :
  principal_amount_loan SI R T = 18000 :=
by
  rw [principal_amount_loan, h_SI, h_R, h_T]
  norm_num

#check find_principal_amount

end NUMINAMATH_GPT_find_principal_amount_l653_65394


namespace NUMINAMATH_GPT_students_like_apple_and_chocolate_not_blueberry_l653_65302

variables (n A C B D : ℕ)

theorem students_like_apple_and_chocolate_not_blueberry
  (h1 : n = 50)
  (h2 : A = 25)
  (h3 : C = 20)
  (h4 : B = 5)
  (h5 : D = 15) :
  ∃ (x : ℕ), x = 10 ∧ x = n - D - (A + C - 2 * x) ∧ 0 ≤ 2 * x - A - C + B :=
sorry

end NUMINAMATH_GPT_students_like_apple_and_chocolate_not_blueberry_l653_65302


namespace NUMINAMATH_GPT_mason_father_age_l653_65334

theorem mason_father_age
  (Mason_age : ℕ) 
  (Sydney_age : ℕ) 
  (Father_age : ℕ)
  (h1 : Mason_age = 20)
  (h2 : Sydney_age = 3 * Mason_age)
  (h3 : Father_age = Sydney_age + 6) :
  Father_age = 66 :=
by
  sorry

end NUMINAMATH_GPT_mason_father_age_l653_65334


namespace NUMINAMATH_GPT_find_m_if_polynomial_is_square_l653_65315

theorem find_m_if_polynomial_is_square (m : ℝ) :
  (∀ x, ∃ k : ℝ, x^2 + 2 * (m - 3) * x + 16 = (x + k)^2) ↔ (m = 7 ∨ m = -1) :=
by
  sorry

end NUMINAMATH_GPT_find_m_if_polynomial_is_square_l653_65315


namespace NUMINAMATH_GPT_find_local_value_of_7_in_difference_l653_65317

-- Define the local value of 3 in the number 28943712.
def local_value_of_3_in_28943712 : Nat := 30000

-- Define the property that the local value of 7 in a number Y is 7000.
def local_value_of_7 (Y : Nat) : Prop := (Y / 1000 % 10) = 7

-- Define the unknown number X and its difference with local value of 3 in 28943712.
variable (X : Nat)

-- Assumption: The difference between X and local_value_of_3_in_28943712 results in a number whose local value of 7 is 7000.
axiom difference_condition : local_value_of_7 (X - local_value_of_3_in_28943712)

-- The proof problem statement to be solved.
theorem find_local_value_of_7_in_difference : local_value_of_7 (X - local_value_of_3_in_28943712) = true :=
by
  -- Proof is omitted.
  sorry

end NUMINAMATH_GPT_find_local_value_of_7_in_difference_l653_65317


namespace NUMINAMATH_GPT_max_marks_l653_65380

theorem max_marks (total_marks : ℕ) (obtained_marks : ℕ) (failed_by : ℕ) 
    (passing_percentage : ℝ) (passing_marks : ℝ) (H1 : obtained_marks = 125)
    (H2 : failed_by = 40) (H3 : passing_percentage = 0.33) 
    (H4 : passing_marks = obtained_marks + failed_by) 
    (H5 : passing_marks = passing_percentage * total_marks) : total_marks = 500 := by
  sorry

end NUMINAMATH_GPT_max_marks_l653_65380


namespace NUMINAMATH_GPT_baggies_of_oatmeal_cookies_l653_65390

theorem baggies_of_oatmeal_cookies (total_cookies : ℝ) (chocolate_chip_cookies : ℝ) (cookies_per_baggie : ℝ) 
(h_total : total_cookies = 41)
(h_choc : chocolate_chip_cookies = 13)
(h_baggie : cookies_per_baggie = 9) : 
  ⌊(total_cookies - chocolate_chip_cookies) / cookies_per_baggie⌋ = 3 := 
by 
  sorry

end NUMINAMATH_GPT_baggies_of_oatmeal_cookies_l653_65390


namespace NUMINAMATH_GPT_total_spent_on_clothing_l653_65398

def shorts_cost : ℝ := 15
def jacket_cost : ℝ := 14.82
def shirt_cost : ℝ := 12.51

theorem total_spent_on_clothing : shorts_cost + jacket_cost + shirt_cost = 42.33 := by
  -- Proof goes here.
  sorry

end NUMINAMATH_GPT_total_spent_on_clothing_l653_65398


namespace NUMINAMATH_GPT_tree_circumference_inequality_l653_65355

theorem tree_circumference_inequality (x : ℝ) : 
  (∀ t : ℝ, t = 10 + 3 * x ∧ t > 90 → x > 80 / 3) :=
by
  intro t ht
  obtain ⟨h_t_eq, h_t_gt_90⟩ := ht
  linarith

end NUMINAMATH_GPT_tree_circumference_inequality_l653_65355


namespace NUMINAMATH_GPT_find_y_l653_65374

def vectors_orthogonal_condition (y : ℝ) : Prop :=
  (1 * -2) + (-3 * y) + (-4 * -1) = 0

theorem find_y : vectors_orthogonal_condition (2 / 3) :=
by
  sorry

end NUMINAMATH_GPT_find_y_l653_65374


namespace NUMINAMATH_GPT_smallest_integer_value_of_m_l653_65321

theorem smallest_integer_value_of_m (x y m : ℝ) 
  (h1 : 3*x + y = m + 8) 
  (h2 : 2*x + 2*y = 2*m + 5) 
  (h3 : x - y < 1) : 
  m >= 3 := 
sorry

end NUMINAMATH_GPT_smallest_integer_value_of_m_l653_65321


namespace NUMINAMATH_GPT_geometric_sequence_solution_l653_65322

noncomputable def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ a1 r, ∀ n, a n = a1 * r ^ (n - 1)

theorem geometric_sequence_solution :
  ∀ (a : ℕ → ℝ),
    (geometric_sequence a) →
    (∃ a2 a18, a2 + a18 = -6 ∧ a2 * a18 = 4 ∧ a 2 = a2 ∧ a 18 = a18) →
    a 4 * a 16 + a 10 = 6 :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_solution_l653_65322


namespace NUMINAMATH_GPT_binary_to_decimal_l653_65344

theorem binary_to_decimal (b : ℕ) (h : b = 2^3 + 2^2 + 0 * 2^1 + 2^0) : b = 13 :=
by {
  -- proof is omitted
  sorry
}

end NUMINAMATH_GPT_binary_to_decimal_l653_65344


namespace NUMINAMATH_GPT_problem_a_add_b_eq_five_l653_65361

variable {a b : ℝ}

theorem problem_a_add_b_eq_five
  (h1 : ∀ x, -2 < x ∧ x < 3 → ax^2 + x + b > 0)
  (h2 : a < 0) :
  a + b = 5 :=
sorry

end NUMINAMATH_GPT_problem_a_add_b_eq_five_l653_65361


namespace NUMINAMATH_GPT_intersection_of_sets_l653_65357

def set_a : Set ℝ := { x | -x^2 + 2 * x ≥ 0 }
def set_b : Set ℝ := { x | -1 ≤ x ∧ x ≤ 1 }
def set_intersection : Set ℝ := { x | 0 ≤ x ∧ x ≤ 1 }

theorem intersection_of_sets : (set_a ∩ set_b) = set_intersection := by 
  sorry

end NUMINAMATH_GPT_intersection_of_sets_l653_65357


namespace NUMINAMATH_GPT_numerical_puzzle_unique_solution_l653_65318

theorem numerical_puzzle_unique_solution :
  ∃ (A X Y P : ℕ), 
    A ≠ X ∧ A ≠ Y ∧ A ≠ P ∧ X ≠ Y ∧ X ≠ P ∧ Y ≠ P ∧
    (A * 10 + X) + (Y * 10 + X) = Y * 100 + P * 10 + A ∧
    A = 8 ∧ X = 9 ∧ Y = 1 ∧ P = 0 :=
sorry

end NUMINAMATH_GPT_numerical_puzzle_unique_solution_l653_65318


namespace NUMINAMATH_GPT_clay_boys_proof_l653_65393

variable (total_students : ℕ)
variable (total_boys : ℕ)
variable (total_girls : ℕ)
variable (jonas_students : ℕ)
variable (clay_students : ℕ)
variable (birch_students : ℕ)
variable (jonas_boys : ℕ)
variable (birch_girls : ℕ)

noncomputable def boys_from_clay (total_students total_boys total_girls jonas_students clay_students birch_students jonas_boys birch_girls : ℕ) : ℕ :=
  let birch_boys := birch_students - birch_girls
  let clay_boys := total_boys - (jonas_boys + birch_boys)
  clay_boys

theorem clay_boys_proof (h1 : total_students = 180) (h2 : total_boys = 94) 
    (h3 : total_girls = 86) (h4 : jonas_students = 60) 
    (h5 : clay_students = 80) (h6 : birch_students = 40) 
    (h7 : jonas_boys = 30) (h8 : birch_girls = 24) : 
  boys_from_clay total_students total_boys total_girls jonas_students clay_students birch_students jonas_boys birch_girls = 48 := 
by 
  simp [boys_from_clay] 
  sorry

end NUMINAMATH_GPT_clay_boys_proof_l653_65393


namespace NUMINAMATH_GPT_cats_added_l653_65304

theorem cats_added (siamese_cats house_cats total_cats : ℕ) 
  (h1 : siamese_cats = 13) 
  (h2 : house_cats = 5) 
  (h3 : total_cats = 28) : 
  total_cats - (siamese_cats + house_cats) = 10 := 
by 
  sorry

end NUMINAMATH_GPT_cats_added_l653_65304


namespace NUMINAMATH_GPT_geom_seq_necessity_geom_seq_not_sufficient_l653_65340

theorem geom_seq_necessity (a₁ q : ℝ) (h₁ : 0 < a₁) (h₂ : a₁ < a₁ * q^2) :
    q > 1 ∨ q < -1 :=
  sorry

theorem geom_seq_not_sufficient (a₁ q : ℝ) (h₁ : 0 < a₁) (h₂ : a₁ < a₁ * q^2) :
    ¬ (q > 1 → a₁ < a₁ * q^2) :=
  sorry

end NUMINAMATH_GPT_geom_seq_necessity_geom_seq_not_sufficient_l653_65340


namespace NUMINAMATH_GPT_gun_fan_image_equivalence_l653_65351

def gunPiercingImage : String := "point moving to form a line"
def foldingFanImage : String := "line moving to form a surface"

theorem gun_fan_image_equivalence :
  (gunPiercingImage = "point moving to form a line") ∧ 
  (foldingFanImage = "line moving to form a surface") := by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_gun_fan_image_equivalence_l653_65351


namespace NUMINAMATH_GPT_jesses_room_total_area_l653_65349

-- Define the dimensions of the first rectangular part
def length1 : ℕ := 12
def width1 : ℕ := 8

-- Define the dimensions of the second rectangular part
def length2 : ℕ := 6
def width2 : ℕ := 4

-- Define the areas of both parts
def area1 : ℕ := length1 * width1
def area2 : ℕ := length2 * width2

-- Define the total area
def total_area : ℕ := area1 + area2

-- Statement of the theorem we want to prove
theorem jesses_room_total_area : total_area = 120 :=
by
  -- We would provide the proof here
  sorry

end NUMINAMATH_GPT_jesses_room_total_area_l653_65349


namespace NUMINAMATH_GPT_largest_integral_k_for_real_distinct_roots_l653_65350

theorem largest_integral_k_for_real_distinct_roots :
  ∃ k : ℤ, (k < 9) ∧ (∀ k' : ℤ, k' < 9 → k' ≤ k) :=
sorry

end NUMINAMATH_GPT_largest_integral_k_for_real_distinct_roots_l653_65350


namespace NUMINAMATH_GPT_quadratic_has_single_real_root_l653_65387

theorem quadratic_has_single_real_root (n : ℝ) (h : (6 * n) ^ 2 - 4 * 1 * (2 * n) = 0) : n = 2 / 9 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_has_single_real_root_l653_65387


namespace NUMINAMATH_GPT_solve_for_a_l653_65399

theorem solve_for_a (x a : ℤ) (h : 2 * x - a - 5 = 0) (hx : x = 3) : a = 1 :=
by sorry

end NUMINAMATH_GPT_solve_for_a_l653_65399


namespace NUMINAMATH_GPT_basketball_classes_l653_65301

theorem basketball_classes (x : ℕ) : (x * (x - 1)) / 2 = 10 :=
sorry

end NUMINAMATH_GPT_basketball_classes_l653_65301


namespace NUMINAMATH_GPT_boy_present_age_l653_65360

-- Define the boy's present age
variable (x : ℤ)

-- Conditions from the problem statement
def condition_one : Prop :=
  x + 4 = 2 * (x - 6)

-- Prove that the boy's present age is 16
theorem boy_present_age (h : condition_one x) : x = 16 := 
sorry

end NUMINAMATH_GPT_boy_present_age_l653_65360


namespace NUMINAMATH_GPT_final_lives_equals_20_l653_65319

def initial_lives : ℕ := 30
def lives_lost : ℕ := 12
def bonus_lives : ℕ := 5
def penalty_lives : ℕ := 3

theorem final_lives_equals_20 : (initial_lives - lives_lost + bonus_lives - penalty_lives) = 20 :=
by 
  sorry

end NUMINAMATH_GPT_final_lives_equals_20_l653_65319


namespace NUMINAMATH_GPT_carrots_planted_per_hour_l653_65311

theorem carrots_planted_per_hour (rows plants_per_row hours : ℕ) (h1 : rows = 400) (h2 : plants_per_row = 300) (h3 : hours = 20) :
  (rows * plants_per_row) / hours = 6000 := by
  sorry

end NUMINAMATH_GPT_carrots_planted_per_hour_l653_65311


namespace NUMINAMATH_GPT_compute_2018_square_123_Delta_4_l653_65339

namespace custom_operations

def Delta (a b : ℕ) : ℕ := a * 10 ^ b + b
def Square (a b : ℕ) : ℕ := a * 10 + b

theorem compute_2018_square_123_Delta_4 : Square 2018 (Delta 123 4) = 1250184 :=
by
  sorry

end custom_operations

end NUMINAMATH_GPT_compute_2018_square_123_Delta_4_l653_65339


namespace NUMINAMATH_GPT_ratio_Nikki_to_Michael_l653_65368

theorem ratio_Nikki_to_Michael
  (M Joyce Nikki Ryn : ℕ)
  (h1 : Joyce = M + 2)
  (h2 : Nikki = 30)
  (h3 : Ryn = (4 / 5) * Nikki)
  (h4 : M + Joyce + Nikki + Ryn = 76) :
  Nikki / M = 3 :=
by {
  sorry
}

end NUMINAMATH_GPT_ratio_Nikki_to_Michael_l653_65368


namespace NUMINAMATH_GPT_find_two_digit_ab_l653_65345

def digit_range (n : ℕ) : Prop := 0 ≤ n ∧ n ≤ 9

def different_digits (a b c d : ℕ) : Prop :=
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d

theorem find_two_digit_ab (A B C D : ℕ) (hA : digit_range A) (hB : digit_range B)
                         (hC : digit_range C) (hD : digit_range D)
                         (h_diff : different_digits A B C D)
                         (h_eq : (100 * A + 10 * B + C) * (10 * A + B) + C * D = 2017) :
  10 * A + B = 14 :=
sorry

end NUMINAMATH_GPT_find_two_digit_ab_l653_65345


namespace NUMINAMATH_GPT_range_of_m_in_first_quadrant_l653_65366

theorem range_of_m_in_first_quadrant (m : ℝ) : ((m - 1 > 0) ∧ (m + 2 > 0)) ↔ m > 1 :=
by sorry

end NUMINAMATH_GPT_range_of_m_in_first_quadrant_l653_65366


namespace NUMINAMATH_GPT_amount_left_after_pool_l653_65300

def amount_left (total_earned : ℝ) (cost_per_person : ℝ) (num_people : ℕ) : ℝ :=
  total_earned - (cost_per_person * num_people)

theorem amount_left_after_pool :
  amount_left 30 2.5 10 = 5 :=
by
  sorry

end NUMINAMATH_GPT_amount_left_after_pool_l653_65300


namespace NUMINAMATH_GPT_james_correct_take_home_pay_l653_65371

noncomputable def james_take_home_pay : ℝ :=
  let main_job_hourly_rate := 20
  let second_job_hourly_rate := main_job_hourly_rate * 0.8
  let main_job_hours := 30
  let main_job_overtime_hours := 5
  let second_job_hours := 15
  let side_gig_daily_rate := 100
  let side_gig_days := 2
  let tax_deductions := 200
  let federal_tax_rate := 0.18
  let state_tax_rate := 0.05

  let regular_main_job_hours := main_job_hours - main_job_overtime_hours
  let main_job_regular_pay := regular_main_job_hours * main_job_hourly_rate
  let main_job_overtime_pay := main_job_overtime_hours * main_job_hourly_rate * 1.5
  let total_main_job_pay := main_job_regular_pay + main_job_overtime_pay

  let total_second_job_pay := second_job_hours * second_job_hourly_rate
  let total_side_gig_pay := side_gig_daily_rate * side_gig_days

  let total_earnings := total_main_job_pay + total_second_job_pay + total_side_gig_pay
  let taxable_income := total_earnings - tax_deductions
  let federal_tax := taxable_income * federal_tax_rate
  let state_tax := taxable_income * state_tax_rate
  let total_taxes := federal_tax + state_tax
  total_earnings - total_taxes

theorem james_correct_take_home_pay : james_take_home_pay = 885.30 := by
  sorry

end NUMINAMATH_GPT_james_correct_take_home_pay_l653_65371


namespace NUMINAMATH_GPT_most_likely_composition_l653_65367

def event_a : Prop := (1 / 3) * (1 / 3) * 2 = (2 / 9)
def event_d : Prop := 2 * (1 / 3 * 1 / 3) = (2 / 9)

theorem most_likely_composition :
  event_a ∧ event_d :=
by sorry

end NUMINAMATH_GPT_most_likely_composition_l653_65367


namespace NUMINAMATH_GPT_neither_sufficient_nor_necessary_l653_65376

theorem neither_sufficient_nor_necessary (α β : ℝ) :
  (α + β = 90) ↔ ¬((α + β = 90) ↔ (Real.sin α + Real.sin β > 1)) :=
sorry

end NUMINAMATH_GPT_neither_sufficient_nor_necessary_l653_65376


namespace NUMINAMATH_GPT_cost_of_four_pencils_and_four_pens_l653_65369

def pencil_cost : ℝ := sorry
def pen_cost : ℝ := sorry

axiom h1 : 8 * pencil_cost + 3 * pen_cost = 5.10
axiom h2 : 3 * pencil_cost + 5 * pen_cost = 4.95

theorem cost_of_four_pencils_and_four_pens : 4 * pencil_cost + 4 * pen_cost = 4.488 :=
by
  sorry

end NUMINAMATH_GPT_cost_of_four_pencils_and_four_pens_l653_65369


namespace NUMINAMATH_GPT_repeating_decimal_exceeds_decimal_representation_l653_65320

noncomputable def repeating_decimal : ℚ := 71 / 99
def decimal_representation : ℚ := 71 / 100

theorem repeating_decimal_exceeds_decimal_representation :
  repeating_decimal - decimal_representation = 71 / 9900 := by
  sorry

end NUMINAMATH_GPT_repeating_decimal_exceeds_decimal_representation_l653_65320


namespace NUMINAMATH_GPT_EDTA_Ca2_complex_weight_l653_65354

-- Definitions of atomic weights
def atomic_weight_C : ℝ := 12.01
def atomic_weight_H : ℝ := 1.008
def atomic_weight_N : ℝ := 14.01
def atomic_weight_O : ℝ := 16.00
def atomic_weight_Ca : ℝ := 40.08

-- Number of atoms in EDTA
def num_atoms_C : ℝ := 10
def num_atoms_H : ℝ := 16
def num_atoms_N : ℝ := 2
def num_atoms_O : ℝ := 8

-- Molecular weight of EDTA
def molecular_weight_EDTA : ℝ :=
  num_atoms_C * atomic_weight_C +
  num_atoms_H * atomic_weight_H +
  num_atoms_N * atomic_weight_N +
  num_atoms_O * atomic_weight_O

-- Proof that the molecular weight of the complex is 332.328 g/mol
theorem EDTA_Ca2_complex_weight : molecular_weight_EDTA + atomic_weight_Ca = 332.328 := by
  sorry

end NUMINAMATH_GPT_EDTA_Ca2_complex_weight_l653_65354


namespace NUMINAMATH_GPT_how_many_candies_eaten_l653_65324

variable (candies_tuesday candies_thursday candies_friday candies_left : ℕ)

def total_candies (candies_tuesday candies_thursday candies_friday : ℕ) : ℕ :=
  candies_tuesday + candies_thursday + candies_friday

theorem how_many_candies_eaten (h_tuesday : candies_tuesday = 3)
                               (h_thursday : candies_thursday = 5)
                               (h_friday : candies_friday = 2)
                               (h_left : candies_left = 4) :
  (total_candies candies_tuesday candies_thursday candies_friday) - candies_left = 6 :=
by
  sorry

end NUMINAMATH_GPT_how_many_candies_eaten_l653_65324


namespace NUMINAMATH_GPT_expression_approx_l653_65379

noncomputable def simplified_expression : ℝ :=
  (Real.sqrt 97 + 9 * Real.sqrt 6 + 5 * Real.sqrt 5) / (3 * Real.sqrt 6 + 7)

theorem expression_approx : abs (simplified_expression - 3.002) < 0.001 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_expression_approx_l653_65379


namespace NUMINAMATH_GPT_minimum_value_of_expression_l653_65364

theorem minimum_value_of_expression (x : ℝ) (h : x > 2) : 
  ∃ y, (∀ z, z > 2 → (z^2 - 4 * z + 5) / (z - 2) ≥ y) ∧ 
       y = 2 :=
by
  sorry

end NUMINAMATH_GPT_minimum_value_of_expression_l653_65364


namespace NUMINAMATH_GPT_hat_p_at_1_l653_65305

-- Define the polynomial p(x)
def p (x : ℝ) : ℝ := x^2 - (1 + 1)*x + 1

-- Definition of displeased polynomial
def isDispleased (p : ℝ → ℝ) : Prop :=
  ∃ (x1 x2 x3 x4 : ℝ), p (p x1) = 0 ∧ p (p x2) = 0 ∧ p (p x3) = 0 ∧ p (p x4) = 0

-- Define the specific polynomial hat_p
def hat_p (x : ℝ) : ℝ := p x

-- Theorem statement
theorem hat_p_at_1 : isDispleased hat_p → hat_p 1 = 0 :=
by
  sorry

end NUMINAMATH_GPT_hat_p_at_1_l653_65305


namespace NUMINAMATH_GPT_expression_equals_5_l653_65313

def expression_value : ℤ := 8 + 15 / 3 - 2^3

theorem expression_equals_5 : expression_value = 5 :=
by
  sorry

end NUMINAMATH_GPT_expression_equals_5_l653_65313


namespace NUMINAMATH_GPT_greatest_three_digit_number_divisible_by_3_6_5_l653_65338

theorem greatest_three_digit_number_divisible_by_3_6_5 :
  ∃ n : ℕ, 100 ≤ n ∧ n ≤ 999 ∧ (n % 3 = 0) ∧ (n % 6 = 0) ∧ (n % 5 = 0) ∧ (∀ m : ℕ, 100 ≤ m ∧ m ≤ 999 ∧ (m % 3 = 0) ∧ (m % 6 = 0) ∧ (m % 5 = 0) → m ≤ n) ∧ n = 990 := 
by
  sorry

end NUMINAMATH_GPT_greatest_three_digit_number_divisible_by_3_6_5_l653_65338


namespace NUMINAMATH_GPT_find_a_l653_65353

/-- Given function -/
def f (x: ℝ) : ℝ := (x + 1)^2 - 2 * (x + 1)

/-- Problem statement -/
theorem find_a (a : ℝ) (h : f a = 3) : a = 2 ∨ a = -2 := 
by
  sorry

end NUMINAMATH_GPT_find_a_l653_65353


namespace NUMINAMATH_GPT_find_a_l653_65383

def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, x ≠ 0 → f (-x) = -f x

noncomputable def f (a x : ℝ) : ℝ := (1 / (2^x - 1)) + a

theorem find_a (a : ℝ) : 
  is_odd_function (f a) → a = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_find_a_l653_65383


namespace NUMINAMATH_GPT_student_marks_l653_65392

theorem student_marks (x : ℕ) :
  let total_questions := 60
  let correct_answers := 38
  let wrong_answers := total_questions - correct_answers
  let total_marks := 130
  let marks_from_correct := correct_answers * x
  let marks_lost := wrong_answers * 1
  let net_marks := marks_from_correct - marks_lost
  net_marks = total_marks → x = 4 :=
by
  intros
  sorry

end NUMINAMATH_GPT_student_marks_l653_65392


namespace NUMINAMATH_GPT_vanya_number_l653_65335

def S (n : ℕ) : ℕ := n.digits 10 |>.sum

theorem vanya_number:
  (2014 + S 2014 = 2021) ∧ (1996 + S 1996 = 2021) := by
  sorry

end NUMINAMATH_GPT_vanya_number_l653_65335
