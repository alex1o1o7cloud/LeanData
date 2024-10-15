import Mathlib

namespace NUMINAMATH_GPT_arctan_sum_l1507_150734

theorem arctan_sum (a b : ℝ) (h1 : a = 3) (h2 : b = 7) : 
  Real.arctan (a / b) + Real.arctan (b / a) = Real.pi / 2 := 
by 
  rw [h1, h2]
  sorry

end NUMINAMATH_GPT_arctan_sum_l1507_150734


namespace NUMINAMATH_GPT_problem_1_problem_2_l1507_150702

noncomputable def is_positive_real (x : ℝ) : Prop := x > 0

theorem problem_1 (a b : ℝ) (ha : is_positive_real a) (hb : is_positive_real b)
  (h : 1 / a + 1 / b = 2 * Real.sqrt 2) : 
  a^2 + b^2 ≥ 1 := by
  sorry

theorem problem_2 (a b : ℝ) (ha : is_positive_real a) (hb : is_positive_real b)
  (h : 1 / a + 1 / b = 2 * Real.sqrt 2) (h_extra : (a - b)^2 ≥ 4 * (a * b)^3) : 
  a * b = 1 := by
  sorry

end NUMINAMATH_GPT_problem_1_problem_2_l1507_150702


namespace NUMINAMATH_GPT_rise_in_height_of_field_l1507_150749

theorem rise_in_height_of_field
  (field_length : ℝ)
  (field_width : ℝ)
  (pit_length : ℝ)
  (pit_width : ℝ)
  (pit_depth : ℝ)
  (field_area : ℝ := field_length * field_width)
  (pit_area : ℝ := pit_length * pit_width)
  (remaining_area : ℝ := field_area - pit_area)
  (pit_volume : ℝ := pit_length * pit_width * pit_depth)
  (rise_in_height : ℝ := pit_volume / remaining_area) :
  field_length = 20 →
  field_width = 10 →
  pit_length = 8 →
  pit_width = 5 →
  pit_depth = 2 →
  rise_in_height = 0.5 :=
by
  intros
  sorry

end NUMINAMATH_GPT_rise_in_height_of_field_l1507_150749


namespace NUMINAMATH_GPT_sequence_properties_l1507_150725

theorem sequence_properties :
  ∀ {a : ℕ → ℝ} {b : ℕ → ℝ},
  a 1 = 1 ∧ 
  (∀ n, b n > 4 / 3) ∧ 
  (∀ n, (∀ x, x^2 - b n * x + a n = 0 → (x = a (n + 1) ∨ x = 1 + a n))) →
  (a 2 = 1 / 2 ∧ ∃ n, b n > 4 / 3 ∧ n = 5) := by
  sorry

end NUMINAMATH_GPT_sequence_properties_l1507_150725


namespace NUMINAMATH_GPT_div_poly_iff_l1507_150708

-- Definitions from conditions
def P (x : ℂ) (n : ℕ) := x^(4*n) + x^(3*n) + x^(2*n) + x^n + 1
def Q (x : ℂ) := x^4 + x^3 + x^2 + x + 1

-- The main theorem stating the problem
theorem div_poly_iff (n : ℕ) : 
  ∀ x : ℂ, (P x n) ∣ (Q x) ↔ n % 5 ≠ 0 :=
by sorry

end NUMINAMATH_GPT_div_poly_iff_l1507_150708


namespace NUMINAMATH_GPT_minimum_trucks_required_l1507_150752

-- Definitions for the problem
def total_weight_stones : ℝ := 10
def max_stone_weight : ℝ := 1
def truck_capacity : ℝ := 3

-- The theorem to prove
theorem minimum_trucks_required : ∃ (n : ℕ), n = 5 ∧ (n * truck_capacity) ≥ total_weight_stones := by
  sorry

end NUMINAMATH_GPT_minimum_trucks_required_l1507_150752


namespace NUMINAMATH_GPT_least_positive_integer_y_l1507_150790

theorem least_positive_integer_y (x k y: ℤ) (h1: 24 * x + k * y = 4) (h2: ∃ x: ℤ, ∃ y: ℤ, 24 * x + k * y = 4) : y = 4 :=
sorry

end NUMINAMATH_GPT_least_positive_integer_y_l1507_150790


namespace NUMINAMATH_GPT_trapezoid_leg_length_l1507_150718

theorem trapezoid_leg_length (S : ℝ) (h₁ : S > 0) : 
  ∃ x : ℝ, x = Real.sqrt (2 * S) ∧ x > 0 :=
by
  sorry

end NUMINAMATH_GPT_trapezoid_leg_length_l1507_150718


namespace NUMINAMATH_GPT_grid_path_theorem_l1507_150733

open Nat

variables (m n : ℕ)
variables (A B C : ℕ)

def conditions (m n : ℕ) : Prop := m ≥ 4 ∧ n ≥ 4

noncomputable def grid_path_problem (m n A B C : ℕ) : Prop :=
  conditions m n ∧
  ((m - 1) * (n - 1) = A + (B + C)) ∧
  A = B - C + m + n - 1

theorem grid_path_theorem (m n A B C : ℕ) (h : grid_path_problem m n A B C) : 
  A = B - C + m + n - 1 :=
  sorry

end NUMINAMATH_GPT_grid_path_theorem_l1507_150733


namespace NUMINAMATH_GPT_ratio_of_y_to_x_l1507_150703

theorem ratio_of_y_to_x (c x y : ℝ) (hx : x = 0.90 * c) (hy : y = 1.20 * c) :
  y / x = 4 / 3 := 
sorry

end NUMINAMATH_GPT_ratio_of_y_to_x_l1507_150703


namespace NUMINAMATH_GPT_sum_of_squares_of_roots_eq_l1507_150737

-- Definitions derived directly from conditions
def a := 5
def b := 2
def c := -15

-- Sum of roots
def sum_of_roots : ℚ := (-b : ℚ) / a

-- Product of roots
def product_of_roots : ℚ := (c : ℚ) / a

-- Sum of the squares of the roots
def sum_of_squares_of_roots : ℚ := sum_of_roots^2 - 2 * product_of_roots

-- The statement that needs to be proved
theorem sum_of_squares_of_roots_eq : sum_of_squares_of_roots = 154 / 25 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_squares_of_roots_eq_l1507_150737


namespace NUMINAMATH_GPT_missy_total_watching_time_l1507_150729

def num_reality_shows := 5
def length_reality_show := 28
def num_cartoons := 1
def length_cartoon := 10

theorem missy_total_watching_time : 
  (num_reality_shows * length_reality_show + num_cartoons * length_cartoon) = 150 := 
by 
  sorry

end NUMINAMATH_GPT_missy_total_watching_time_l1507_150729


namespace NUMINAMATH_GPT_product_uvw_l1507_150709

theorem product_uvw (a x y c : ℝ) (u v w : ℤ) :
  (a^u * x - a^v) * (a^w * y - a^3) = a^5 * c^5 → 
  a^8 * x * y - a^7 * y - a^6 * x = a^5 * (c^5 - 1) → 
  u * v * w = 6 :=
by
  intros h1 h2
  -- Proof will go here
  sorry

end NUMINAMATH_GPT_product_uvw_l1507_150709


namespace NUMINAMATH_GPT_final_exam_mean_score_l1507_150780

theorem final_exam_mean_score (μ σ : ℝ) 
  (h1 : 55 = μ - 1.5 * σ)
  (h2 : 75 = μ - 2 * σ)
  (h3 : 85 = μ + 1.5 * σ)
  (h4 : 100 = μ + 3.5 * σ) :
  μ = 115 :=
by
  sorry

end NUMINAMATH_GPT_final_exam_mean_score_l1507_150780


namespace NUMINAMATH_GPT_retailer_mark_up_l1507_150724

theorem retailer_mark_up (R C M S : ℝ) 
  (hC : C = 0.7 * R)
  (hS : S = C / 0.7)
  (hSm : S = 0.9 * M) : 
  M = 1.111 * R :=
by 
  sorry

end NUMINAMATH_GPT_retailer_mark_up_l1507_150724


namespace NUMINAMATH_GPT_proof_problem_l1507_150717

def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

def given_function (f : ℝ → ℝ) : Prop :=
  odd_function f ∧ f (-3) = -2

theorem proof_problem (f : ℝ → ℝ) (h : given_function f) : f 3 + f 0 = -2 :=
by sorry

end NUMINAMATH_GPT_proof_problem_l1507_150717


namespace NUMINAMATH_GPT_integer_solutions_l1507_150759

theorem integer_solutions (m n : ℤ) :
  m^3 - n^3 = 2 * m * n + 8 ↔ (m = 0 ∧ n = -2) ∨ (m = 2 ∧ n = 0) :=
sorry

end NUMINAMATH_GPT_integer_solutions_l1507_150759


namespace NUMINAMATH_GPT_squares_difference_l1507_150757

theorem squares_difference (x y z : ℤ) 
  (h1 : x + y = 10) 
  (h2 : x - y = 8) 
  (h3 : y + z = 15) : 
  x^2 - z^2 = -115 :=
by 
  sorry

end NUMINAMATH_GPT_squares_difference_l1507_150757


namespace NUMINAMATH_GPT_find_angle_CDE_l1507_150791

-- Definition of the angles and their properties
variables {A B C D E : Type}

-- Hypotheses
def angleA_is_right (angleA: ℝ) : Prop := angleA = 90
def angleB_is_right (angleB: ℝ) : Prop := angleB = 90
def angleC_is_right (angleC: ℝ) : Prop := angleC = 90
def angleAEB_value (angleAEB : ℝ) : Prop := angleAEB = 40
def angleBED_eq_angleBDE (angleBED angleBDE : ℝ) : Prop := angleBED = angleBDE

-- The theorem to be proved
theorem find_angle_CDE 
  (angleA : ℝ) (angleB : ℝ) (angleC : ℝ) (angleAEB : ℝ) (angleBED angleBDE : ℝ) (angleCDE : ℝ) :
  angleA_is_right angleA → 
  angleB_is_right angleB → 
  angleC_is_right angleC → 
  angleAEB_value angleAEB → 
  angleBED_eq_angleBDE angleBED angleBDE →
  angleBED = 45 →
  angleCDE = 95 :=
by
  intros
  sorry


end NUMINAMATH_GPT_find_angle_CDE_l1507_150791


namespace NUMINAMATH_GPT_degrees_of_remainder_is_correct_l1507_150761

noncomputable def degrees_of_remainder (P D : Polynomial ℤ) : Finset ℕ :=
  if D.degree = 3 then {0, 1, 2} else ∅

theorem degrees_of_remainder_is_correct
(P : Polynomial ℤ) :
  degrees_of_remainder P (Polynomial.C 3 * Polynomial.X^3 - Polynomial.C 5 * Polynomial.X^2 + Polynomial.C 2 * Polynomial.X - Polynomial.C 4) = {0, 1, 2} :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_degrees_of_remainder_is_correct_l1507_150761


namespace NUMINAMATH_GPT_triangle_area_l1507_150726

theorem triangle_area {r : ℝ} (h_r : r = 6) {x : ℝ} 
  (h1 : 5 * x = 2 * r)
  (h2 : x = 12 / 5) : 
  (1 / 2 * (3 * x) * (4 * x) = 34.56) :=
by
  sorry

end NUMINAMATH_GPT_triangle_area_l1507_150726


namespace NUMINAMATH_GPT_hal_paul_difference_l1507_150770

def halAnswer : Int := 12 - (3 * 2) + 4
def paulAnswer : Int := (12 - 3) * 2 + 4

theorem hal_paul_difference :
  halAnswer - paulAnswer = -12 := by
  sorry

end NUMINAMATH_GPT_hal_paul_difference_l1507_150770


namespace NUMINAMATH_GPT_central_angle_radian_measure_l1507_150750

-- Define the unit circle radius
def unit_circle_radius : ℝ := 1

-- Given an arc of length 1
def arc_length : ℝ := 1

-- Problem Statement: Prove that the radian measure of the central angle α is 1
theorem central_angle_radian_measure :
  ∀ (r : ℝ) (l : ℝ), r = unit_circle_radius → l = arc_length → |l / r| = 1 :=
by
  intros r l hr hl
  rw [hr, hl]
  sorry

end NUMINAMATH_GPT_central_angle_radian_measure_l1507_150750


namespace NUMINAMATH_GPT_equation_of_circle_unique_l1507_150705

noncomputable def equation_of_circle := 
  ∃ (d e f : ℝ), 
    (4 + 4 + 2*d + 2*e + f = 0) ∧ 
    (25 + 9 + 5*d + 3*e + f = 0) ∧ 
    (9 + 1 + 3*d - e + f = 0) ∧ 
    (∀ (x y : ℝ), x^2 + y^2 + d*x + e*y + f = 0 → (x = 2 ∧ y = 2) ∨ (x = 5 ∧ y = 3) ∨ (x = 3 ∧ y = -1))

theorem equation_of_circle_unique :
  equation_of_circle := sorry

end NUMINAMATH_GPT_equation_of_circle_unique_l1507_150705


namespace NUMINAMATH_GPT_units_place_3_pow_34_l1507_150797

theorem units_place_3_pow_34 : (3^34 % 10) = 9 :=
by
  sorry

end NUMINAMATH_GPT_units_place_3_pow_34_l1507_150797


namespace NUMINAMATH_GPT_sum_exists_l1507_150727

theorem sum_exists 
  (n : ℕ) 
  (hn : n ≥ 5) 
  (k : ℕ) 
  (hk : k > (n + 1) / 2) 
  (a : ℕ → ℕ) 
  (ha1 : ∀ i, 1 ≤ a i) 
  (ha2 : ∀ i, a i < n) 
  (h_distinct : ∀ i j, i ≠ j → a i ≠ a j):
  ∃ i j l, i ≠ j ∧ a i + a j = a l := 
by 
  sorry

end NUMINAMATH_GPT_sum_exists_l1507_150727


namespace NUMINAMATH_GPT_books_count_is_8_l1507_150723

theorem books_count_is_8
  (k a p_k p_a : ℕ)
  (h1 : k = a + 6)
  (h2 : k * p_k = 1056)
  (h3 : a * p_a = 56)
  (h4 : p_k > p_a + 100) :
  k = 8 := 
sorry

end NUMINAMATH_GPT_books_count_is_8_l1507_150723


namespace NUMINAMATH_GPT_min_value_of_f_l1507_150741

noncomputable def f (x : ℝ) : ℝ := (Real.log x / Real.log 3 - 1) * (Real.log x / Real.log 3 - 3)

theorem min_value_of_f (x1 x2 : ℝ) (hx1_pos : 0 < x1) (hx1_distinct : x1 ≠ x2) (hx2_pos : 0 < x2)
  (h_f_eq : f x1 = f x2) : (1 / x1 + 9 / x2) = 2 / 3 :=
by
  sorry

end NUMINAMATH_GPT_min_value_of_f_l1507_150741


namespace NUMINAMATH_GPT_integer_k_values_l1507_150706

noncomputable def is_integer_solution (k x : ℤ) : Prop :=
  ((k - 2013) * x = 2015 - 2014 * x)

theorem integer_k_values (k : ℤ) (h : ∃ x : ℤ, is_integer_solution k x) :
  ∃ n : ℕ, n = 16 :=
by
  sorry

end NUMINAMATH_GPT_integer_k_values_l1507_150706


namespace NUMINAMATH_GPT_sum_of_possible_values_of_y_l1507_150794

-- Definitions of the conditions
variables (y : ℝ)
-- Angle measures in degrees
variables (a b c : ℝ)
variables (isosceles : Bool)

-- Given conditions
def is_isosceles_triangle (a b c : ℝ) (isosceles : Bool) : Prop :=
  isosceles = true ∧ (a = b ∨ b = c ∨ c = a)

-- Sum of angles in any triangle
def sum_of_angles_in_triangle (a b c : ℝ) : Prop :=
  a + b + c = 180

-- Main statement to be proven
theorem sum_of_possible_values_of_y (y : ℝ) (a b c : ℝ) (isosceles : Bool) :
  is_isosceles_triangle a b c isosceles →
  sum_of_angles_in_triangle a b c →
  ((y = 60) → (a = y ∨ b = y ∨ c = y)) →
  isosceles = true → a = 60 ∨ b = 60 ∨ c = 60 →
  y + y + y = 180 :=
by
  intros h1 h2 h3 h4 h5
  sorry  -- Proof will be provided here

end NUMINAMATH_GPT_sum_of_possible_values_of_y_l1507_150794


namespace NUMINAMATH_GPT_school_travel_time_is_12_l1507_150754

noncomputable def time_to_school (T : ℕ) : Prop :=
  let extra_time := 6
  let total_distance_covered := 2 * extra_time
  T = total_distance_covered

theorem school_travel_time_is_12 :
  ∃ T : ℕ, time_to_school T ∧ T = 12 :=
by
  sorry

end NUMINAMATH_GPT_school_travel_time_is_12_l1507_150754


namespace NUMINAMATH_GPT_price_of_pants_l1507_150792

-- Given conditions
variables (P B : ℝ)
axiom h1 : P + B = 70.93
axiom h2 : P = B - 2.93

-- Statement to prove
theorem price_of_pants : P = 34.00 :=
by
  sorry

end NUMINAMATH_GPT_price_of_pants_l1507_150792


namespace NUMINAMATH_GPT_intersection_A_B_l1507_150762
open Set

def A := {x : ℝ | 1 ≤ x ∧ x ≤ 3}
def B := {x : ℝ | 2 < x}

theorem intersection_A_B : A ∩ B = {x : ℝ | 2 < x ∧ x ≤ 3} := 
by sorry

end NUMINAMATH_GPT_intersection_A_B_l1507_150762


namespace NUMINAMATH_GPT_alpha_plus_beta_l1507_150777

theorem alpha_plus_beta (α β : ℝ) (h : ∀ x, (x - α) / (x + β) = (x^2 - 116 * x + 2783) / (x^2 + 99 * x - 4080)) 
: α + β = 115 := 
sorry

end NUMINAMATH_GPT_alpha_plus_beta_l1507_150777


namespace NUMINAMATH_GPT_sum_S19_is_190_l1507_150720

-- Define what it means to be an arithmetic sequence
def is_arithmetic_sequence {α : Type*} [AddCommGroup α] (a : ℕ → α) : Prop :=
∀ n m, a n + a m = a (n+1) + a (m-1)

-- Define the sum of the first n terms of the sequence
def sum_of_first_n_terms (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
∀ n, S n = n * (a 1 + a n) / 2

-- Main theorem
theorem sum_S19_is_190 (a : ℕ → ℝ) (S : ℕ → ℝ)
  (h_arith_seq : is_arithmetic_sequence a)
  (h_sum_def : sum_of_first_n_terms a S)
  (h_condition : a 6 + a 14 = 20) :
  S 19 = 190 :=
sorry

end NUMINAMATH_GPT_sum_S19_is_190_l1507_150720


namespace NUMINAMATH_GPT_train_length_l1507_150796

theorem train_length (L : ℝ) (v : ℝ)
  (h1 : v = (L + 130) / 15)
  (h2 : v = (L + 250) / 20) : 
  L = 230 :=
sorry

end NUMINAMATH_GPT_train_length_l1507_150796


namespace NUMINAMATH_GPT_dog_revs_l1507_150758

theorem dog_revs (r₁ r₂ : ℝ) (n₁ : ℕ) (n₂ : ℕ) (h₁ : r₁ = 48) (h₂ : n₁ = 40) (h₃ : r₂ = 12) :
  n₂ = 160 := 
sorry

end NUMINAMATH_GPT_dog_revs_l1507_150758


namespace NUMINAMATH_GPT_min_expression_value_l1507_150798

theorem min_expression_value :
  ∃ x y : ℝ, (9 - x^2 - 8 * x * y - 16 * y^2 > 0) ∧ 
  (∀ x y : ℝ, 9 - x^2 - 8 * x * y - 16 * y^2 > 0 →
  (13 * x^2 + 24 * x * y + 13 * y^2 + 16 * x + 14 * y + 68) / 
  (9 - x^2 - 8 * x * y - 16 * y^2)^(5/2) = (7 / 27)) :=
sorry

end NUMINAMATH_GPT_min_expression_value_l1507_150798


namespace NUMINAMATH_GPT_s_of_4_l1507_150766

noncomputable def t (x : ℚ) : ℚ := 5 * x - 14
noncomputable def s (y : ℚ) : ℚ := 
  let x := (y + 14) / 5
  x^2 + 5 * x - 4

theorem s_of_4 : s (4) = 674 / 25 := by
  sorry

end NUMINAMATH_GPT_s_of_4_l1507_150766


namespace NUMINAMATH_GPT_find_divisor_l1507_150715

theorem find_divisor (dividend quotient remainder divisor : ℕ) 
  (h1 : dividend = 161) 
  (h2 : quotient = 10)
  (h3 : remainder = 1)
  (h4 : dividend = divisor * quotient + remainder) : 
  divisor = 16 :=
by
  sorry

end NUMINAMATH_GPT_find_divisor_l1507_150715


namespace NUMINAMATH_GPT_ram_pairs_sold_correct_l1507_150768

-- Define the costs
def graphics_card_cost := 600
def hard_drive_cost := 80
def cpu_cost := 200
def ram_pair_cost := 60

-- Define the number of items sold
def graphics_cards_sold := 10
def hard_drives_sold := 14
def cpus_sold := 8
def total_earnings := 8960

-- Calculate earnings from individual items
def earnings_graphics_cards := graphics_cards_sold * graphics_card_cost
def earnings_hard_drives := hard_drives_sold * hard_drive_cost
def earnings_cpus := cpus_sold * cpu_cost

-- Calculate total earnings from graphics cards, hard drives, and CPUs
def earnings_other_items := earnings_graphics_cards + earnings_hard_drives + earnings_cpus

-- Calculate earnings from RAM
def earnings_from_ram := total_earnings - earnings_other_items

-- Calculate number of RAM pairs sold
def ram_pairs_sold := earnings_from_ram / ram_pair_cost

-- The theorem to be proven
theorem ram_pairs_sold_correct : ram_pairs_sold = 4 :=
by
  sorry

end NUMINAMATH_GPT_ram_pairs_sold_correct_l1507_150768


namespace NUMINAMATH_GPT_relationship_sides_l1507_150712

-- Definitions for the given condition
variables (a b c : ℝ)

-- Statement of the theorem to prove
theorem relationship_sides (h : a^2 - 16 * b^2 - c^2 + 6 * a * b + 10 * b * c = 0) : a + c = 2 * b :=
sorry

end NUMINAMATH_GPT_relationship_sides_l1507_150712


namespace NUMINAMATH_GPT_remainder_when_7645_divided_by_9_l1507_150707

/--
  Prove that the remainder when 7645 is divided by 9 is 4,
  given that a number is congruent to the sum of its digits modulo 9.
-/
theorem remainder_when_7645_divided_by_9 :
  7645 % 9 = 4 :=
by
  -- Main proof should go here
  sorry

end NUMINAMATH_GPT_remainder_when_7645_divided_by_9_l1507_150707


namespace NUMINAMATH_GPT_bug_paths_l1507_150713

-- Define the problem conditions
structure PathSetup (A B : Type) :=
  (red_arrows : ℕ) -- number of red arrows from point A
  (red_to_blue : ℕ) -- number of blue arrows reachable from each red arrow
  (blue_to_green : ℕ) -- number of green arrows reachable from each blue arrow
  (green_to_orange : ℕ) -- number of orange arrows reachable from each green arrow
  (start_arrows : ℕ) -- starting number of arrows from point A to red arrows
  (orange_arrows : ℕ) -- number of orange arrows equivalent to green arrows

-- Define the conditions for our specific problem setup
def problem_setup : PathSetup Point Point :=
  {
    red_arrows := 3,
    red_to_blue := 2,
    blue_to_green := 2,
    green_to_orange := 1,
    start_arrows := 3,
    orange_arrows := 6 * 2 * 2 -- derived from blue_to_green and red_to_blue steps
  }

-- Prove the number of unique paths from A to B
theorem bug_paths (setup : PathSetup Point Point) : 
  setup.start_arrows * setup.red_to_blue * setup.blue_to_green * setup.green_to_orange * setup.orange_arrows = 1440 :=
by
  -- Calculations are performed; exact values must hold
  sorry

end NUMINAMATH_GPT_bug_paths_l1507_150713


namespace NUMINAMATH_GPT_Chandler_more_rolls_needed_l1507_150710

theorem Chandler_more_rolls_needed :
  let total_goal := 12
  let sold_to_grandmother := 3
  let sold_to_uncle := 4
  let sold_to_neighbor := 3
  let total_sold := sold_to_grandmother + sold_to_uncle + sold_to_neighbor
  total_goal - total_sold = 2 :=
by
  sorry

end NUMINAMATH_GPT_Chandler_more_rolls_needed_l1507_150710


namespace NUMINAMATH_GPT_no_repair_needed_l1507_150771

def nominal_mass : ℝ := 370 -- Assign the nominal mass as determined in the problem solving.

def max_deviation (M : ℝ) : ℝ := 0.1 * M
def preserved_max_deviation : ℝ := 37
def unreadable_max_deviation : ℝ := 37

def within_max_deviation (dev : ℝ) := dev ≤ preserved_max_deviation

noncomputable def standard_deviation : ℝ := preserved_max_deviation

theorem no_repair_needed :
  ∀ (M : ℝ),
  max_deviation M = 0.1 * M →
  preserved_max_deviation ≤ max_deviation M →
  ∀ (dev : ℝ), within_max_deviation dev →
  standard_deviation ≤ preserved_max_deviation →
  preserved_max_deviation = 37 →
  "не требует" = "не требует" :=
by
  intros M h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_no_repair_needed_l1507_150771


namespace NUMINAMATH_GPT_total_wheels_is_90_l1507_150745

-- Defining the conditions
def number_of_bicycles := 20
def number_of_cars := 10
def number_of_motorcycles := 5

-- Calculating the total number of wheels
def total_wheels_in_garage : Nat :=
  (2 * number_of_bicycles) + (4 * number_of_cars) + (2 * number_of_motorcycles)

-- Statement to prove
theorem total_wheels_is_90 : total_wheels_in_garage = 90 := by
  sorry

end NUMINAMATH_GPT_total_wheels_is_90_l1507_150745


namespace NUMINAMATH_GPT_number_of_shirts_is_39_l1507_150764

-- Define the conditions as Lean definitions.
def washing_machine_capacity : ℕ := 8
def number_of_sweaters : ℕ := 33
def number_of_loads : ℕ := 9

-- Define the total number of pieces of clothing based on the conditions.
def total_pieces_of_clothing : ℕ :=
  number_of_loads * washing_machine_capacity

-- Define the number of shirts.
noncomputable def number_of_shirts : ℕ :=
  total_pieces_of_clothing - number_of_sweaters

-- The actual proof problem statement.
theorem number_of_shirts_is_39 :
  number_of_shirts = 39 := by
  sorry

end NUMINAMATH_GPT_number_of_shirts_is_39_l1507_150764


namespace NUMINAMATH_GPT_hyperbola_satisfies_m_l1507_150751

theorem hyperbola_satisfies_m (m : ℝ) 
  (h1 : ∀ x y : ℝ, x^2 - m * y^2 = 1)
  (h2 : ∀ a b : ℝ, (a^2 = 1) ∧ (b^2 = 1/m) ∧ (2*a = 2 * 2*b)) : 
  m = 4 := 
sorry

end NUMINAMATH_GPT_hyperbola_satisfies_m_l1507_150751


namespace NUMINAMATH_GPT_somu_present_age_l1507_150747

def Somu_Age_Problem (S F : ℕ) : Prop := 
  S = F / 3 ∧ S - 6 = (F - 6) / 5

theorem somu_present_age (S F : ℕ) 
  (h : Somu_Age_Problem S F) : S = 12 := 
by
  sorry

end NUMINAMATH_GPT_somu_present_age_l1507_150747


namespace NUMINAMATH_GPT_total_cost_of_one_pencil_and_eraser_l1507_150738

/-- Lila buys 15 pencils and 7 erasers for 170 cents. A pencil costs less than an eraser, 
neither item costs exactly half as much as the other, and both items cost a whole number of cents. 
Prove that the total cost of one pencil and one eraser is 16 cents. -/
theorem total_cost_of_one_pencil_and_eraser (p e : ℕ) (h1 : 15 * p + 7 * e = 170)
  (h2 : p < e) (h3 : p ≠ e / 2) : p + e = 16 :=
sorry

end NUMINAMATH_GPT_total_cost_of_one_pencil_and_eraser_l1507_150738


namespace NUMINAMATH_GPT_div_a2_plus_2_congr_mod8_l1507_150793

variable (a d : ℤ)
variable (h_odd : a % 2 = 1)
variable (h_pos : a > 0)

theorem div_a2_plus_2_congr_mod8 :
  (d ∣ (a ^ 2 + 2)) → (d % 8 = 1 ∨ d % 8 = 3) :=
by
  sorry

end NUMINAMATH_GPT_div_a2_plus_2_congr_mod8_l1507_150793


namespace NUMINAMATH_GPT_cost_of_600_pages_l1507_150779

def cost_per_5_pages := 10 -- 10 cents for 5 pages
def pages_to_copy := 600
def expected_cost := 12 * 100 -- 12 dollars in cents

theorem cost_of_600_pages : pages_to_copy * (cost_per_5_pages / 5) = expected_cost := by
  sorry

end NUMINAMATH_GPT_cost_of_600_pages_l1507_150779


namespace NUMINAMATH_GPT_find_unit_prices_l1507_150744

-- Define the prices of brush and chess set
variables (x y : ℝ)

-- Condition 1: Buying 5 brushes and 12 chess sets costs 315 yuan
def condition1 : Prop := 5 * x + 12 * y = 315

-- Condition 2: Buying 8 brushes and 6 chess sets costs 240 yuan
def condition2 : Prop := 8 * x + 6 * y = 240

-- Prove that the unit price of each brush is 15 yuan and each chess set is 20 yuan
theorem find_unit_prices (hx : condition1 x y) (hy : condition2 x y) :
  x = 15 ∧ y = 20 := 
sorry

end NUMINAMATH_GPT_find_unit_prices_l1507_150744


namespace NUMINAMATH_GPT_ratio_of_numbers_l1507_150700

theorem ratio_of_numbers (A B : ℕ) (hA : A = 45) (hLCM : Nat.lcm A B = 180) : A / Nat.lcm A B = 45 / 4 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_numbers_l1507_150700


namespace NUMINAMATH_GPT_complete_square_l1507_150765

theorem complete_square (x : ℝ) : (x^2 + 4*x - 1 = 0) → ((x + 2)^2 = 5) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_complete_square_l1507_150765


namespace NUMINAMATH_GPT_derivative_exp_l1507_150763

theorem derivative_exp (f : ℝ → ℝ) (h : ∀ x, f x = Real.exp x) : 
    ∀ x, deriv f x = Real.exp x :=
by 
  sorry

end NUMINAMATH_GPT_derivative_exp_l1507_150763


namespace NUMINAMATH_GPT_area_of_fig_between_x1_and_x2_l1507_150788

noncomputable def area_under_curve_x2 (a b : ℝ) : ℝ :=
∫ x in a..b, x^2

theorem area_of_fig_between_x1_and_x2 :
  area_under_curve_x2 1 2 = 7 / 3 := by
  sorry

end NUMINAMATH_GPT_area_of_fig_between_x1_and_x2_l1507_150788


namespace NUMINAMATH_GPT_rth_term_l1507_150772

-- Given arithmetic progression sum formula
def Sn (n : ℕ) : ℕ := 3 * n^2 + 4 * n + 5

-- Prove that the r-th term of the sequence is 6r + 1
theorem rth_term (r : ℕ) : (Sn r) - (Sn (r - 1)) = 6 * r + 1 :=
by
  sorry

end NUMINAMATH_GPT_rth_term_l1507_150772


namespace NUMINAMATH_GPT_count_three_digit_perfect_squares_divisible_by_4_l1507_150755

theorem count_three_digit_perfect_squares_divisible_by_4 :
  ∃ (n : ℕ), n = 11 ∧ ∀ (k : ℕ), 10 ≤ k ∧ k ≤ 31 → (∃ m : ℕ, m^2 = k^2 ∧ 100 ≤ m^2 ∧ m^2 ≤ 999 ∧ m^2 % 4 = 0) := 
sorry

end NUMINAMATH_GPT_count_three_digit_perfect_squares_divisible_by_4_l1507_150755


namespace NUMINAMATH_GPT_root_zero_implies_m_eq_6_l1507_150787

theorem root_zero_implies_m_eq_6 (m : ℝ) (h : ∃ x : ℝ, 3 * (x^2) + m * x + m - 6 = 0) : m = 6 := 
by sorry

end NUMINAMATH_GPT_root_zero_implies_m_eq_6_l1507_150787


namespace NUMINAMATH_GPT_sum_of_digits_0_to_2012_l1507_150785

-- Define the sum of digits function
def sum_of_digits (n : Nat) : Nat := 
  n.digits 10 |>.sum

-- Define the problem to calculate the sum of all digits from 0 to 2012
def sum_digits_up_to (n : Nat) : Nat := 
  (List.range (n + 1)).map sum_of_digits |>.sum

-- Lean theorem statement to prove the sum of digits from 0 to 2012 is 28077
theorem sum_of_digits_0_to_2012 : sum_digits_up_to 2012 = 28077 := 
  sorry

end NUMINAMATH_GPT_sum_of_digits_0_to_2012_l1507_150785


namespace NUMINAMATH_GPT_evaluate_expression_l1507_150739

theorem evaluate_expression :
  (3 * 4 * 5) * ((1 / 3) + (1 / 4) + (1 / 5)) = 47 := 
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l1507_150739


namespace NUMINAMATH_GPT_question_1_question_2_l1507_150778

def f (x a : ℝ) := |x - a|

theorem question_1 :
  (∀ x, f x a ≤ 3 ↔ -1 ≤ x ∧ x ≤ 5) → a = 2 :=
by
  sorry

theorem question_2 (a : ℝ) (h : a = 2) :
  (∀ x, f x a + f (x + 5) a ≥ m) → m ≤ 5 :=
by
  sorry

end NUMINAMATH_GPT_question_1_question_2_l1507_150778


namespace NUMINAMATH_GPT_negation_of_proposition_l1507_150756

theorem negation_of_proposition :
  ¬ (∀ x : ℝ, x^3 - x^2 + 1 ≤ 0) ↔ ∃ x : ℝ, x^3 - x^2 + 1 > 0 := 
by 
  sorry

end NUMINAMATH_GPT_negation_of_proposition_l1507_150756


namespace NUMINAMATH_GPT_Winnie_the_Pooh_stationary_escalator_steps_l1507_150742

theorem Winnie_the_Pooh_stationary_escalator_steps
  (u v L : ℝ)
  (cond1 : L * u / (u + v) = 55)
  (cond2 : L * u / (u - v) = 1155) :
  L = 105 := by
  sorry

end NUMINAMATH_GPT_Winnie_the_Pooh_stationary_escalator_steps_l1507_150742


namespace NUMINAMATH_GPT_students_attending_Harvard_l1507_150760

theorem students_attending_Harvard (total_applicants : ℕ) (perc_accepted : ℝ) (perc_attending : ℝ)
    (h1 : total_applicants = 20000)
    (h2 : perc_accepted = 0.05)
    (h3 : perc_attending = 0.9) :
    total_applicants * perc_accepted * perc_attending = 900 := 
by
    sorry

end NUMINAMATH_GPT_students_attending_Harvard_l1507_150760


namespace NUMINAMATH_GPT_mean_of_solutions_l1507_150748

theorem mean_of_solutions (x : ℝ) (h : x^3 + x^2 - 14 * x = 0) : 
  let a := (0 : ℝ)
  let b := (-1 + Real.sqrt 57) / 2
  let c := (-1 - Real.sqrt 57) / 2
  (a + b + c) / 3 = -2 / 3 :=
sorry

end NUMINAMATH_GPT_mean_of_solutions_l1507_150748


namespace NUMINAMATH_GPT_perp_bisector_eq_parallel_line_eq_reflected_ray_eq_l1507_150740

-- Define points A, B, and P
def A : ℝ × ℝ := (8, -6)
def B : ℝ × ℝ := (2, 2)
def P : ℝ × ℝ := (2, -3)

-- Problem statement for part (I)
theorem perp_bisector_eq : ∃ (k m: ℝ), 3 * k - 4 * m - 23 = 0 :=
sorry

-- Problem statement for part (II)
theorem parallel_line_eq : ∃ (k m: ℝ), 4 * k + 3 * m + 1 = 0 :=
sorry

-- Problem statement for part (III)
theorem reflected_ray_eq : ∃ (k m: ℝ), 11 * k + 27 * m + 74 = 0 :=
sorry

end NUMINAMATH_GPT_perp_bisector_eq_parallel_line_eq_reflected_ray_eq_l1507_150740


namespace NUMINAMATH_GPT_roll_two_dice_prime_sum_l1507_150783

noncomputable def prime_sum_probability : ℚ :=
  let favorable_outcomes := 15
  let total_outcomes := 36
  favorable_outcomes / total_outcomes

theorem roll_two_dice_prime_sum : prime_sum_probability = 5 / 12 :=
  sorry

end NUMINAMATH_GPT_roll_two_dice_prime_sum_l1507_150783


namespace NUMINAMATH_GPT_length_of_train_a_l1507_150773

theorem length_of_train_a
  (speed_train_a : ℝ) (speed_train_b : ℝ) 
  (clearing_time : ℝ) (length_train_b : ℝ)
  (h1 : speed_train_a = 42)
  (h2 : speed_train_b = 30)
  (h3 : clearing_time = 12.998960083193344)
  (h4 : length_train_b = 160) :
  ∃ length_train_a : ℝ, length_train_a = 99.9792016638669 :=
by 
  sorry

end NUMINAMATH_GPT_length_of_train_a_l1507_150773


namespace NUMINAMATH_GPT_op_example_l1507_150701

def myOp (c d : Int) : Int :=
  c * (d + 1) + c * d

theorem op_example : myOp 5 (-2) = -15 := 
  by
    sorry

end NUMINAMATH_GPT_op_example_l1507_150701


namespace NUMINAMATH_GPT_not_all_inequalities_hold_l1507_150732

theorem not_all_inequalities_hold (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  ¬ ((a + b < c + d) ∧ ((a + b) * (c + d) < a * b + c * d) ∧ ((a + b) * c * d < a * b * (c + d))) :=
by
  sorry

end NUMINAMATH_GPT_not_all_inequalities_hold_l1507_150732


namespace NUMINAMATH_GPT_overall_loss_amount_l1507_150769

theorem overall_loss_amount 
    (S : ℝ)
    (hS : S = 12499.99)
    (profit_percent : ℝ)
    (loss_percent : ℝ)
    (sold_at_profit : ℝ)
    (sold_at_loss : ℝ) 
    (condition1 : profit_percent = 0.2)
    (condition2 : loss_percent = -0.1)
    (condition3 : sold_at_profit = 0.2 * S * (1 + profit_percent))
    (condition4 : sold_at_loss = 0.8 * S * (1 + loss_percent))
    :
    S - (sold_at_profit + sold_at_loss) = 500 := 
by 
  sorry

end NUMINAMATH_GPT_overall_loss_amount_l1507_150769


namespace NUMINAMATH_GPT_intersection_A_B_union_A_B_diff_A_B_diff_B_A_l1507_150782

def A : Set Real := {x | -1 < x ∧ x < 2}
def B : Set Real := {x | 0 < x ∧ x < 4}

theorem intersection_A_B :
  A ∩ B = {x | 0 < x ∧ x < 2} :=
sorry

theorem union_A_B :
  A ∪ B = {x | -1 < x ∧ x < 4} :=
sorry

theorem diff_A_B :
  A \ B = {x | -1 < x ∧ x ≤ 0} :=
sorry

theorem diff_B_A :
  B \ A = {x | 2 ≤ x ∧ x < 4} :=
sorry

end NUMINAMATH_GPT_intersection_A_B_union_A_B_diff_A_B_diff_B_A_l1507_150782


namespace NUMINAMATH_GPT_work_done_by_gravity_l1507_150753

noncomputable def work_by_gravity (m g z_A z_B : ℝ) : ℝ :=
  m * g * (z_B - z_A)

theorem work_done_by_gravity (m g z_A z_B : ℝ) :
  work_by_gravity m g z_A z_B = m * g * (z_B - z_A) :=
by
  sorry

end NUMINAMATH_GPT_work_done_by_gravity_l1507_150753


namespace NUMINAMATH_GPT_problem1_problem2_problem3_l1507_150728

-- Definitions and conditions
variable (f : ℝ → ℝ)
variable (h1 : ∀ x y : ℝ, f (x + y) = f x + f y)
variable (h2 : ∀ x : ℝ, x > 0 → f x < 0)

-- Question 1: Prove the function is odd
theorem problem1 : ∀ x : ℝ, f (-x) = -f x := by
  sorry

-- Question 2: Prove the function is monotonically decreasing
theorem problem2 : ∀ x1 x2 : ℝ, x1 < x2 → f x1 < f x2 := by
  sorry

-- Question 3: Solve the inequality given f(2) = 1
theorem problem3 (h3 : f 2 = 1) : ∀ x : ℝ, f (-x^2) + 2*f x + 4 < 0 ↔ -2 < x ∧ x < 4 := by
  sorry

end NUMINAMATH_GPT_problem1_problem2_problem3_l1507_150728


namespace NUMINAMATH_GPT_total_marks_of_all_candidates_l1507_150743

theorem total_marks_of_all_candidates 
  (average_marks : ℕ) 
  (num_candidates : ℕ) 
  (average : average_marks = 35) 
  (candidates : num_candidates = 120) : 
  average_marks * num_candidates = 4200 :=
by
  -- The proof will be written here
  sorry

end NUMINAMATH_GPT_total_marks_of_all_candidates_l1507_150743


namespace NUMINAMATH_GPT_value_of_expr_l1507_150730

theorem value_of_expr (a : Int) (h : a = -2) : a + 1 = -1 := by
  -- Placeholder for the proof, assuming it's correct
  sorry

end NUMINAMATH_GPT_value_of_expr_l1507_150730


namespace NUMINAMATH_GPT_find_a6_l1507_150789

def seq (a : ℕ → ℤ) : Prop :=
  a 1 = 2 ∧ a 2 = 5 ∧ ∀ n : ℕ, a (n + 1) = a (n + 2) + a n

theorem find_a6 (a : ℕ → ℤ) (h : seq a) : a 6 = -3 :=
by
  sorry

end NUMINAMATH_GPT_find_a6_l1507_150789


namespace NUMINAMATH_GPT_hens_to_roosters_multiplier_l1507_150767

def totalChickens : ℕ := 75
def numHens : ℕ := 67

-- Given the total number of chickens and a certain relationship
theorem hens_to_roosters_multiplier
  (numRoosters : ℕ) (multiplier : ℕ)
  (h1 : totalChickens = numHens + numRoosters)
  (h2 : numHens = multiplier * numRoosters - 5) :
  multiplier = 9 :=
by sorry

end NUMINAMATH_GPT_hens_to_roosters_multiplier_l1507_150767


namespace NUMINAMATH_GPT_solve_cryptarithm_l1507_150731

def cryptarithm_puzzle (K I C : ℕ) : Prop :=
  K ≠ I ∧ K ≠ C ∧ I ≠ C ∧
  K + I + C < 30 ∧  -- Ensuring each is a single digit (0-9)
  (10 * K + I + C) + (10 * K + 10 * C + I) = 100 + 10 * I + 10 * C + K

theorem solve_cryptarithm :
  ∃ K I C, cryptarithm_puzzle K I C ∧ K = 4 ∧ I = 9 ∧ C = 5 :=
by
  use 4, 9, 5
  sorry 

end NUMINAMATH_GPT_solve_cryptarithm_l1507_150731


namespace NUMINAMATH_GPT_isosceles_triangle_equal_sides_length_l1507_150704

noncomputable def equal_side_length_isosceles_triangle (base median : ℝ) (vertex_angle_deg : ℝ) : ℝ :=
  if base = 36 ∧ median = 15 ∧ vertex_angle_deg = 60 then 3 * Real.sqrt 191 else 0

theorem isosceles_triangle_equal_sides_length:
  equal_side_length_isosceles_triangle 36 15 60 = 3 * Real.sqrt 191 :=
by
  sorry

end NUMINAMATH_GPT_isosceles_triangle_equal_sides_length_l1507_150704


namespace NUMINAMATH_GPT_no_solution_for_A_to_make_47A8_div_by_5_l1507_150722

def divisible_by_5 (n : ℕ) : Prop :=
  n % 5 = 0

theorem no_solution_for_A_to_make_47A8_div_by_5 (A : ℕ) :
  ¬ (divisible_by_5 (47 * 1000 + A * 100 + 8)) :=
by
  sorry

end NUMINAMATH_GPT_no_solution_for_A_to_make_47A8_div_by_5_l1507_150722


namespace NUMINAMATH_GPT_find_line_equation_l1507_150719

theorem find_line_equation (k m b : ℝ) :
  (∃ k, |(k^2 + 7*k + 10) - (m*k + b)| = 8) ∧ (8 = 2*m + b) ∧ (b ≠ 0) → (m = 5 ∧ b = 3) := 
by
  intro h
  sorry

end NUMINAMATH_GPT_find_line_equation_l1507_150719


namespace NUMINAMATH_GPT_race_length_l1507_150735

theorem race_length (members : ℕ) (member_distance : ℕ) (ralph_multiplier : ℕ) 
    (h1 : members = 4) (h2 : member_distance = 3) (h3 : ralph_multiplier = 2) : 
    members * member_distance + ralph_multiplier * member_distance = 18 :=
by
  -- Start the proof with sorry to denote missing steps.
  sorry

end NUMINAMATH_GPT_race_length_l1507_150735


namespace NUMINAMATH_GPT_train_tunnel_length_l1507_150776

theorem train_tunnel_length 
  (train_length : ℝ) 
  (train_speed : ℝ) 
  (time_for_tail_to_exit : ℝ) 
  (h_train_length : train_length = 2) 
  (h_train_speed : train_speed = 90) 
  (h_time_for_tail_to_exit : time_for_tail_to_exit = 2 / 60) :
  ∃ tunnel_length : ℝ, tunnel_length = 1 := 
by
  sorry

end NUMINAMATH_GPT_train_tunnel_length_l1507_150776


namespace NUMINAMATH_GPT_how_many_more_choc_chip_cookies_l1507_150774

-- Define the given conditions
def choc_chip_cookies_yesterday := 19
def raisin_cookies_this_morning := 231
def choc_chip_cookies_this_morning := 237

-- Define the total chocolate chip cookies
def total_choc_chip_cookies : ℕ := choc_chip_cookies_this_morning + choc_chip_cookies_yesterday

-- Define the proof statement
theorem how_many_more_choc_chip_cookies :
  total_choc_chip_cookies - raisin_cookies_this_morning = 25 :=
by
  -- Proof will go here
  sorry

end NUMINAMATH_GPT_how_many_more_choc_chip_cookies_l1507_150774


namespace NUMINAMATH_GPT_one_eighth_of_power_l1507_150775

theorem one_eighth_of_power (x : ℕ) (h : (1 / 8) * (2 ^ 36) = 2 ^ x) : x = 33 :=
by 
  -- Proof steps are not needed, so we leave it as sorry.
  sorry

end NUMINAMATH_GPT_one_eighth_of_power_l1507_150775


namespace NUMINAMATH_GPT_smallest_multiplier_to_perfect_square_l1507_150746

-- Definitions for the conditions
def y := 2^3 * 3^2 * 4^3 * 5^3 * 6^6 * 7^5 * 8^6 * 9^6

-- The theorem statement itself
theorem smallest_multiplier_to_perfect_square : ∃ k : ℕ, (∀ m : ℕ, (y * m = k) → (∃ n : ℕ, (k * y) = n^2)) :=
by
  let y := 2^3 * 3^2 * 4^3 * 5^3 * 6^6 * 7^5 * 8^6 * 9^6
  let smallest_k := 70
  have h : y = 2^33 * 3^20 * 5^3 * 7^5 := by sorry
  use smallest_k
  intros m hm
  use (2^17 * 3^10 * 5 * 7)
  sorry

end NUMINAMATH_GPT_smallest_multiplier_to_perfect_square_l1507_150746


namespace NUMINAMATH_GPT_quadratic_inequality_solution_l1507_150795

theorem quadratic_inequality_solution {x : ℝ} :
  (x^2 - 6 * x - 16 > 0) ↔ (x < -2 ∨ x > 8) :=
sorry

end NUMINAMATH_GPT_quadratic_inequality_solution_l1507_150795


namespace NUMINAMATH_GPT_beans_in_jar_l1507_150711

theorem beans_in_jar (B : ℕ) 
  (h1 : B / 4 = number_of_red_beans)
  (h2 : number_of_red_beans = B / 4)
  (h3 : number_of_white_beans = (B * 3 / 4) / 3)
  (h4 : number_of_white_beans = B / 4)
  (h5 : number_of_remaining_beans_after_white = B / 2)
  (h6 : 143 = B / 4):
  B = 572 :=
by
  sorry

end NUMINAMATH_GPT_beans_in_jar_l1507_150711


namespace NUMINAMATH_GPT_abs_eq_necessary_but_not_sufficient_l1507_150784

theorem abs_eq_necessary_but_not_sufficient (x y : ℝ) :
  (|x| = |y|) → (¬(x = y) → x = -y) :=
by
  sorry

end NUMINAMATH_GPT_abs_eq_necessary_but_not_sufficient_l1507_150784


namespace NUMINAMATH_GPT_abc_zero_l1507_150781

theorem abc_zero
  (a b c : ℝ)
  (h1 : (a + b) * (b + c) * (c + a) = a * b * c)
  (h2 : (a^3 + b^3) * (b^3 + c^3) * (c^3 + a^3) = (a * b * c)^3) :
  a * b * c = 0 := 
sorry

end NUMINAMATH_GPT_abc_zero_l1507_150781


namespace NUMINAMATH_GPT_find_rate_of_interest_l1507_150799

/-- At what rate percent on simple interest will Rs. 25,000 amount to Rs. 34,500 in 5 years? 
    Given Principal (P) = Rs. 25,000, Amount (A) = Rs. 34,500, Time (T) = 5 years. 
    We need to find the Rate (R). -/
def principal : ℝ := 25000
def amount : ℝ := 34500
def time : ℝ := 5

theorem find_rate_of_interest (P A T : ℝ) : 
  P = principal → 
  A = amount → 
  T = time → 
  ∃ R : ℝ, R = 7.6 :=
by
  intros hP hA hT
  -- proof goes here
  sorry

end NUMINAMATH_GPT_find_rate_of_interest_l1507_150799


namespace NUMINAMATH_GPT_height_of_platform_l1507_150714

variables (l w h : ℕ)

theorem height_of_platform (hl1 : l + h - 2 * w = 36) (hl2 : w + h - l = 30) (hl3 : h = 2 * w) : h = 44 := 
sorry

end NUMINAMATH_GPT_height_of_platform_l1507_150714


namespace NUMINAMATH_GPT_max_homework_ratio_l1507_150716

theorem max_homework_ratio 
  (H : ℕ) -- time spent on history tasks
  (biology_time : ℕ)
  (total_homework_time : ℕ)
  (geography_time : ℕ)
  (history_geography_relation : geography_time = 3 * H)
  (total_time_relation : total_homework_time = 180)
  (biology_time_known : biology_time = 20)
  (sum_time_relation : H + geography_time + biology_time = total_homework_time) :
  H / biology_time = 2 :=
by
  sorry

end NUMINAMATH_GPT_max_homework_ratio_l1507_150716


namespace NUMINAMATH_GPT_arrangement_A_and_B_adjacent_arrangement_A_B_and_C_adjacent_arrangement_A_and_B_adjacent_C_not_ends_arrangement_ABC_and_DEFG_units_l1507_150786

-- Definitions based on conditions in A)
def students : List Char := ['A', 'B', 'C', 'D', 'E', 'F', 'G']
def A : Char := 'A'
def B : Char := 'B'
def C : Char := 'C'
def D : Char := 'D'
def E : Char := 'E'
def F : Char := 'F'
def G : Char := 'G'

-- Holistic theorem statements for each question derived from the correct answers in B)
theorem arrangement_A_and_B_adjacent :
  ∃ (n : ℕ), n = 1440 := sorry

theorem arrangement_A_B_and_C_adjacent :
  ∃ (n : ℕ), n = 720 := sorry

theorem arrangement_A_and_B_adjacent_C_not_ends :
  ∃ (n : ℕ), n = 960 := sorry

theorem arrangement_ABC_and_DEFG_units :
  ∃ (n : ℕ), n = 288 := sorry

end NUMINAMATH_GPT_arrangement_A_and_B_adjacent_arrangement_A_B_and_C_adjacent_arrangement_A_and_B_adjacent_C_not_ends_arrangement_ABC_and_DEFG_units_l1507_150786


namespace NUMINAMATH_GPT_max_area_triangle_l1507_150721

theorem max_area_triangle (a b c S : ℝ) (h₁ : S = a^2 - (b - c)^2) (h₂ : b + c = 8) :
  S ≤ 64 / 17 :=
sorry

end NUMINAMATH_GPT_max_area_triangle_l1507_150721


namespace NUMINAMATH_GPT_difference_students_l1507_150736

variables {A B AB : ℕ}

theorem difference_students (h1 : A + AB + B = 800)
  (h2 : AB = 20 * (A + AB) / 100)
  (h3 : AB = 25 * (B + AB) / 100) :
  A - B = 100 :=
sorry

end NUMINAMATH_GPT_difference_students_l1507_150736
