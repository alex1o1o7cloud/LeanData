import Mathlib

namespace mile_time_sum_is_11_l2217_221732

def mile_time_sum (Tina_time Tony_time Tom_time : ℕ) : ℕ :=
  Tina_time + Tony_time + Tom_time

theorem mile_time_sum_is_11 :
  ∃ (Tina_time Tony_time Tom_time : ℕ),
  (Tina_time = 6 ∧ Tony_time = Tina_time / 2 ∧ Tom_time = Tina_time / 3) →
  mile_time_sum Tina_time Tony_time Tom_time = 11 :=
by
  sorry

end mile_time_sum_is_11_l2217_221732


namespace inequality_proof_l2217_221716

variable (ha la r R : ℝ)
variable (α β γ : ℝ)

-- Conditions
def condition1 : Prop := ha / la = Real.cos ((β - γ) / 2)
def condition2 : Prop := 8 * Real.sin (α / 2) * Real.sin (β / 2) * Real.sin (γ / 2) = 2 * r / R

-- The theorem to be proved
theorem inequality_proof (h1 : condition1 ha la β γ) (h2 : condition2 α β γ r R) :
  Real.cos ((β - γ) / 2) ≥ Real.sqrt (2 * r / R) :=
sorry

end inequality_proof_l2217_221716


namespace inequality_proof_l2217_221763

variable {a b c : ℝ}

theorem inequality_proof (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a / (1 + a + a * b)) + (b / (1 + b + b * c)) + (c / (1 + c + c * a)) ≤ 1 :=
by
  sorry

end inequality_proof_l2217_221763


namespace magnitude_z_is_sqrt_2_l2217_221796

open Complex

noncomputable def z (x y : ℝ) : ℂ := x + y * I

theorem magnitude_z_is_sqrt_2 (x y : ℝ) (h1 : (2 * x) / (1 - I) = 1 + y * I) : abs (z x y) = Real.sqrt 2 :=
by
  -- You would fill in the proof steps here based on the problem's solution.
  sorry

end magnitude_z_is_sqrt_2_l2217_221796


namespace least_positive_integer_l2217_221768

theorem least_positive_integer (a : ℕ) :
  (a % 2 = 0) ∧ (a % 5 = 1) ∧ (a % 4 = 2) → a = 6 :=
by
  sorry

end least_positive_integer_l2217_221768


namespace find_t_l2217_221745

theorem find_t:
  (∃ t, (∀ (x y: ℝ), (x = 2 ∧ y = 8) ∨ (x = 4 ∧ y = 14) ∨ (x = 6 ∧ y = 20) → 
                (∀ (m b: ℝ), y = m * x + b) ∧ 
                (∀ (m b: ℝ), y = 3 * x + b ∧ b = 2 ∧ (t = 3 * 50 + 2) ∧ t = 152))) := by
  sorry

end find_t_l2217_221745


namespace find_value_l2217_221755

theorem find_value (x y z : ℝ) (h₁ : y = 3 * x) (h₂ : z = 3 * y + x) : x + y + z = 14 * x :=
by
  sorry

end find_value_l2217_221755


namespace number_of_ways_to_assign_volunteers_l2217_221797

/-- Theorem: The number of ways to assign 5 volunteers to 3 venues such that each venue has at least one volunteer is 150. -/
theorem number_of_ways_to_assign_volunteers :
  let total_ways := 3^5
  let subtract_one_empty := 3 * 2^5
  let add_back_two_empty := 3 * 1^5
  (total_ways - subtract_one_empty + add_back_two_empty) = 150 :=
by
  sorry

end number_of_ways_to_assign_volunteers_l2217_221797


namespace max_percentage_l2217_221772

def total_students : ℕ := 100
def group_size : ℕ := 66
def min_percentage (scores : Fin 100 → ℝ) : Prop :=
  ∀ (S : Finset (Fin 100)), S.card = 66 → (S.sum scores) / (Finset.univ.sum scores) ≥ 0.5

theorem max_percentage (scores : Fin 100 → ℝ) (h : min_percentage scores) :
  ∃ (x : ℝ), ∀ i : Fin 100, scores i <= x ∧ x <= 0.25 * (Finset.univ.sum scores) := sorry

end max_percentage_l2217_221772


namespace eval_expression_l2217_221741

theorem eval_expression (a : ℕ) (h : a = 2) : a^3 * a^6 = 512 := by
  sorry

end eval_expression_l2217_221741


namespace solve_equation_l2217_221757

theorem solve_equation : ∀ x : ℝ, x * (x + 2) = 3 * x + 6 ↔ (x = -2 ∨ x = 3) := by
  sorry

end solve_equation_l2217_221757


namespace inverse_proportion_point_passes_through_l2217_221758

theorem inverse_proportion_point_passes_through
  (m : ℝ) (h1 : (4, 6) ∈ {p : ℝ × ℝ | p.snd = (m^2 + 2 * m - 1) / p.fst})
  : (-4, -6) ∈ {p : ℝ × ℝ | p.snd = (m^2 + 2 * m - 1) / p.fst} :=
sorry

end inverse_proportion_point_passes_through_l2217_221758


namespace total_surface_area_of_square_pyramid_is_correct_l2217_221734

-- Define the base side length and height from conditions
def a : ℝ := 3
def PD : ℝ := 4

-- Conditions
def square_pyramid : Prop :=
  let AD := a
  let PA := Real.sqrt (PD^2 - a^2)
  let Area_PAD := (1 / 2) * AD * PA
  let Area_PCD := Area_PAD
  let Area_base := a * a
  let Total_surface_area := Area_base + 2 * Area_PAD + 2 * Area_PCD
  Total_surface_area = 9 + 6 * Real.sqrt 7

-- Theorem statement
theorem total_surface_area_of_square_pyramid_is_correct : square_pyramid := sorry

end total_surface_area_of_square_pyramid_is_correct_l2217_221734


namespace equilateral_triangle_area_l2217_221711

theorem equilateral_triangle_area (h : Real) (h_eq : h = Real.sqrt 12):
  ∃ A : Real, A = 12 * Real.sqrt 3 :=
by
  sorry

end equilateral_triangle_area_l2217_221711


namespace solve_equation_l2217_221715

theorem solve_equation : ∃ x : ℝ, 2 * x - 3 = 5 ∧ x = 4 := 
by
  -- Introducing x as a real number and stating the goal
  use 4
  -- Show that 2 * 4 - 3 = 5
  simp
  -- Adding the sorry to skip the proof step
  sorry

end solve_equation_l2217_221715


namespace find_lambda_l2217_221777

variables {a b : ℝ} (lambda : ℝ)

-- Conditions
def orthogonal (x y : ℝ) : Prop := x * y = 0
def magnitude_a : ℝ := 2
def magnitude_b : ℝ := 3
def is_perpendicular (x y : ℝ) : Prop := x * y = 0

-- Proof statement
theorem find_lambda (h₁ : orthogonal a b)
  (h₂ : magnitude_a = 2)
  (h₃ : magnitude_b = 3)
  (h₄ : is_perpendicular (3 * a + 2 * b) (lambda * a - b)) :
  lambda = 3 / 2 :=
sorry

end find_lambda_l2217_221777


namespace rectangle_perimeter_l2217_221759

theorem rectangle_perimeter (a b : ℕ) (h1 : a ≠ b) (h2 : (a * b = 2 * (a + b))) : 2 * (a + b) = 36 :=
by sorry

end rectangle_perimeter_l2217_221759


namespace tens_digit_of_smallest_even_five_digit_number_l2217_221722

def smallest_even_five_digit_number (digits : List ℕ) : ℕ :=
if h : 0 ∈ digits ∧ 3 ∈ digits ∧ 5 ∈ digits ∧ 6 ∈ digits ∧ 8 ∈ digits then
  35086
else
  0  -- this is just a placeholder to make the function total

theorem tens_digit_of_smallest_even_five_digit_number : 
  ∀ digits : List ℕ, 
    0 ∈ digits ∧ 
    3 ∈ digits ∧ 
    5 ∈ digits ∧ 
    6 ∈ digits ∧ 
    8 ∈ digits ∧ 
    digits.length = 5 → 
    (smallest_even_five_digit_number digits) / 10 % 10 = 8 :=
by
  intros digits h
  sorry

end tens_digit_of_smallest_even_five_digit_number_l2217_221722


namespace factor_difference_of_squares_l2217_221727

theorem factor_difference_of_squares (x : ℝ) : 36 - 9 * x^2 = 9 * (2 - x) * (2 + x) :=
by
  sorry

end factor_difference_of_squares_l2217_221727


namespace rain_at_least_once_prob_l2217_221723

theorem rain_at_least_once_prob (p : ℚ) (n : ℕ) (h1 : p = 3/4) (h2 : n = 4) :
  1 - (1 - p)^n = 255/256 :=
by {
  -- Implementation of Lean code is not required as per instructions.
  sorry
}

end rain_at_least_once_prob_l2217_221723


namespace find_expression_value_l2217_221713

variable (a b : ℝ)

theorem find_expression_value (h : a - 2 * b = 7) : 6 - 2 * a + 4 * b = -8 := by
  sorry

end find_expression_value_l2217_221713


namespace no_positive_integer_n_such_that_14n_plus_19_is_prime_l2217_221792

theorem no_positive_integer_n_such_that_14n_plus_19_is_prime :
  ∀ n : Nat, 0 < n → ¬ Nat.Prime (14^n + 19) :=
by
  intro n hn
  sorry

end no_positive_integer_n_such_that_14n_plus_19_is_prime_l2217_221792


namespace solve_for_y_l2217_221778

theorem solve_for_y (y : ℝ) (h : 3 / y + 4 / y / (6 / y) = 1.5) : y = 3.6 :=
sorry

end solve_for_y_l2217_221778


namespace julia_error_approx_97_percent_l2217_221720

theorem julia_error_approx_97_percent (x : ℝ) : 
  abs ((6 * x - x / 6) / (6 * x) * 100 - 97) < 1 :=
by 
  sorry

end julia_error_approx_97_percent_l2217_221720


namespace balls_color_equality_l2217_221736

theorem balls_color_equality (r g b: ℕ) (h1: r + g + b = 20) (h2: b ≥ 7) (h3: r ≥ 4) (h4: b = 2 * g) : 
  r = b ∨ r = g :=
by
  sorry

end balls_color_equality_l2217_221736


namespace number_of_five_digit_numbers_l2217_221750

def count_five_identical_digits: Nat := 9
def count_two_different_digits: Nat := 1215
def count_three_different_digits: Nat := 6480
def count_four_different_digits: Nat := 22680
def count_five_different_digits: Nat := 27216

theorem number_of_five_digit_numbers :
  count_five_identical_digits + count_two_different_digits +
  count_three_different_digits + count_four_different_digits +
  count_five_different_digits = 57600 :=
by
  sorry

end number_of_five_digit_numbers_l2217_221750


namespace num_roots_of_unity_satisfy_cubic_l2217_221766

def root_of_unity (z : ℂ) (n : ℕ) : Prop :=
  z ^ n = 1

def cubic_eqn_root (z : ℂ) (a b c : ℤ) : Prop :=
  z^3 + (a:ℂ) * z^2 + (b:ℂ) * z + (c:ℂ) = 0

theorem num_roots_of_unity_satisfy_cubic (a b c : ℤ) (n : ℕ) 
    (h_n : n ≥ 1) : ∃! z : ℂ, root_of_unity z n ∧ cubic_eqn_root z a b c := sorry

end num_roots_of_unity_satisfy_cubic_l2217_221766


namespace cannot_be_the_lengths_l2217_221774

theorem cannot_be_the_lengths (x y z : ℝ) (h1 : x^2 + y^2 = 16) (h2 : x^2 + z^2 = 25) (h3 : y^2 + z^2 = 49) : false :=
by
  sorry

end cannot_be_the_lengths_l2217_221774


namespace percentage_of_x_l2217_221746

variable (x : ℝ)

theorem percentage_of_x (x : ℝ) : ((40 / 100) * (50 / 100) * x) = (20 / 100) * x := by
  sorry

end percentage_of_x_l2217_221746


namespace range_of_c_l2217_221710

variable (c : ℝ)

def p : Prop := ∀ x : ℝ, x > 0 → c^x = c^(x+1) / c
def q : Prop := ∀ x : ℝ, (1/2 ≤ x ∧ x ≤ 2) → x + 1/x > 1/c

theorem range_of_c (h1 : c > 0) (h2 : p c ∨ q c) (h3 : ¬ (p c ∧ q c)) :
  (0 < c ∧ c ≤ 1/2) ∨ (c ≥ 1) :=
sorry

end range_of_c_l2217_221710


namespace find_number_l2217_221744

theorem find_number 
  (m : ℤ)
  (h13 : m % 13 = 12)
  (h12 : m % 12 = 11)
  (h11 : m % 11 = 10)
  (h10 : m % 10 = 9)
  (h9 : m % 9 = 8)
  (h8 : m % 8 = 7)
  (h7 : m % 7 = 6)
  (h6 : m % 6 = 5)
  (h5 : m % 5 = 4)
  (h4 : m % 4 = 3)
  (h3 : m % 3 = 2) :
  m = 360359 :=
by
  sorry

end find_number_l2217_221744


namespace triangle_with_consecutive_sides_and_angle_property_l2217_221786

theorem triangle_with_consecutive_sides_and_angle_property :
  ∃ (a b c : ℕ), (b = a + 1) ∧ (c = b + 1) ∧
    (∃ (α β γ : ℝ), 2 * α = γ ∧
      (a * a + b * b = c * c + 2 * a * b * α.cos) ∧
      (b * b + c * c = a * a + 2 * b * c * β.cos) ∧
      (c * c + a * a = b * b + 2 * c * a * γ.cos) ∧
      (a = 4) ∧ (b = 5) ∧ (c = 6) ∧
      (γ.cos = 1 / 8)) :=
sorry

end triangle_with_consecutive_sides_and_angle_property_l2217_221786


namespace all_equal_l2217_221781

theorem all_equal (xs xsp : Fin 2011 → ℝ) (h : ∀ i : Fin 2011, xs i + xs ((i + 1) % 2011) = 2 * xsp i) (perm : ∃ σ : Fin 2011 ≃ Fin 2011, ∀ i, xsp i = xs (σ i)) :
  ∀ i j : Fin 2011, xs i = xs j := 
sorry

end all_equal_l2217_221781


namespace simplify_fraction_product_l2217_221780

theorem simplify_fraction_product : 
  (256 / 20 : ℚ) * (10 / 160) * ((16 / 6) ^ 2) = 256 / 45 :=
by norm_num

end simplify_fraction_product_l2217_221780


namespace passing_time_for_platform_l2217_221705

def train_length : ℕ := 1100
def time_to_cross_tree : ℕ := 110
def platform_length : ℕ := 700
def speed := train_length / time_to_cross_tree
def combined_length := train_length + platform_length

theorem passing_time_for_platform : 
  let speed := train_length / time_to_cross_tree
  let combined_length := train_length + platform_length
  combined_length / speed = 180 :=
by
  sorry

end passing_time_for_platform_l2217_221705


namespace find_angle_A_l2217_221708

theorem find_angle_A (a b c : ℝ) (h : a^2 - c^2 = b^2 - b * c) : 
  ∃ (A : ℝ), A = π / 3 :=
by
  sorry

end find_angle_A_l2217_221708


namespace complex_div_symmetry_l2217_221706

open Complex

-- Definitions based on conditions
def z1 : ℂ := 1 + I
def z2 : ℂ := -1 + I

-- Theorem to prove
theorem complex_div_symmetry : z2 / z1 = I := by
  sorry

end complex_div_symmetry_l2217_221706


namespace exists_linear_function_intersecting_negative_axes_l2217_221775

theorem exists_linear_function_intersecting_negative_axes :
  ∃ (k b : ℝ), k < 0 ∧ b < 0 ∧ (∃ x, k * x + b = 0 ∧ x < 0) ∧ (k * 0 + b < 0) :=
by
  sorry

end exists_linear_function_intersecting_negative_axes_l2217_221775


namespace joan_spent_on_trucks_l2217_221767

-- Define constants for the costs
def cost_cars : ℝ := 14.88
def cost_skateboard : ℝ := 4.88
def total_toys : ℝ := 25.62
def cost_trucks : ℝ := 25.62 - (14.88 + 4.88)

-- Statement to prove
theorem joan_spent_on_trucks : cost_trucks = 5.86 := by
  sorry

end joan_spent_on_trucks_l2217_221767


namespace BC_equals_expected_BC_l2217_221730

def point := ℝ × ℝ -- Define a point as a pair of real numbers (coordinates).

def vector_sub (v1 v2 : point) : point := (v1.1 - v2.1, v1.2 - v2.2) -- Define vector subtraction.

-- Definitions of points A and B and vector AC
def A : point := (-1, 1)
def B : point := (0, 2)
def AC : point := (-2, 3)

-- Calculate vector AB
def AB : point := vector_sub B A

-- Calculate vector BC
def BC : point := vector_sub AC AB

-- Expected result
def expected_BC : point := (-3, 2)

-- Proof statement
theorem BC_equals_expected_BC : BC = expected_BC := by
  unfold BC AB AC A B vector_sub
  simp
  sorry

end BC_equals_expected_BC_l2217_221730


namespace range_of_m_when_p_true_range_of_m_when_p_and_q_false_p_or_q_true_l2217_221701

noncomputable def f (x : ℝ) : ℝ := (1 / 2) * x^2 - 9 * Real.log x

def p (m : ℝ) : Prop :=
  ∀ x ∈ (Set.Ioo m (m + 1)), (x - 9 / x) < 0

def q (m : ℝ) : Prop :=
  m > 1 ∧ m < 3

theorem range_of_m_when_p_true :
  ∀ m : ℝ, p m → 0 ≤ m ∧ m ≤ 2 :=
sorry

theorem range_of_m_when_p_and_q_false_p_or_q_true :
  ∀ m : ℝ, (¬(p m ∧ q m) ∧ (p m ∨ q m)) → (0 ≤ m ∧ m ≤ 1) ∨ (2 < m ∧ m < 3) :=
sorry

end range_of_m_when_p_true_range_of_m_when_p_and_q_false_p_or_q_true_l2217_221701


namespace profit_amount_l2217_221700

theorem profit_amount (SP : ℝ) (P : ℝ) (profit : ℝ) : 
  SP = 850 → P = 36 → profit = SP - SP / (1 + P / 100) → profit = 225 :=
by
  intros hSP hP hProfit
  rw [hSP, hP] at *
  simp at *
  sorry

end profit_amount_l2217_221700


namespace adam_apples_count_l2217_221787

variable (Jackie_apples : ℕ)
variable (extra_apples : ℕ)
variable (Adam_apples : ℕ)

theorem adam_apples_count (h1 : Jackie_apples = 9) (h2 : extra_apples = 5) (h3 : Adam_apples = Jackie_apples + extra_apples) :
  Adam_apples = 14 := 
by 
  sorry

end adam_apples_count_l2217_221787


namespace album_cost_l2217_221718

-- Definition of the cost variables
variable (B C A : ℝ)

-- Conditions given in the problem
axiom h1 : B = C + 4
axiom h2 : B = 18
axiom h3 : C = 0.70 * A

-- Theorem to prove the cost of the album
theorem album_cost : A = 20 := sorry

end album_cost_l2217_221718


namespace trapezoid_area_l2217_221737

theorem trapezoid_area (EF GH h : ℕ) (hEF : EF = 60) (hGH : GH = 30) (hh : h = 15) : 
  (EF + GH) * h / 2 = 675 := by 
  sorry

end trapezoid_area_l2217_221737


namespace seq_a3_eq_1_l2217_221776

theorem seq_a3_eq_1 (a : ℕ → ℤ) (h₁ : ∀ n ≥ 1, a (n + 1) = a n - 3) (h₂ : a 1 = 7) : a 3 = 1 :=
by
  sorry

end seq_a3_eq_1_l2217_221776


namespace distance_traveled_l2217_221791

-- Define the variables for speed of slower and faster bike
def slower_speed := 60
def faster_speed := 64

-- Define the condition that slower bike takes 1 hour more than faster bike
def condition (D : ℝ) : Prop := (D / slower_speed) = (D / faster_speed) + 1

-- The theorem we need to prove
theorem distance_traveled : ∃ (D : ℝ), condition D ∧ D = 960 := 
by
  sorry

end distance_traveled_l2217_221791


namespace deepak_age_l2217_221721

variable (R D : ℕ)

theorem deepak_age (h1 : R / D = 4 / 3) (h2 : R + 6 = 26) : D = 15 :=
sorry

end deepak_age_l2217_221721


namespace equal_areas_of_parts_l2217_221779

theorem equal_areas_of_parts :
  ∀ (S1 S2 S3 S4 : ℝ), 
    S1 = S2 → S2 = S3 → 
    (S1 + S2 = S3 + S4) → 
    (S2 + S3 = S1 + S4) → 
    S1 = S2 ∧ S2 = S3 ∧ S3 = S4 :=
by
  intros S1 S2 S3 S4 h1 h2 h3 h4
  sorry

end equal_areas_of_parts_l2217_221779


namespace James_gold_bars_l2217_221794

theorem James_gold_bars (P : ℝ) (h_condition1 : 60 - P / 100 * 60 = 54) : P = 10 := 
  sorry

end James_gold_bars_l2217_221794


namespace min_value_x2_minus_x1_l2217_221748

noncomputable def f (x : ℝ) := 2 * Real.sin (Real.pi / 2 * x + Real.pi / 5)

theorem min_value_x2_minus_x1 :
  (∀ x : ℝ, f x1 ≤ f x ∧ f x ≤ f x2) → |x2 - x1| = 2 :=
sorry

end min_value_x2_minus_x1_l2217_221748


namespace S8_value_l2217_221707

theorem S8_value (x : ℝ) (h : x + 1/x = 4) (S : ℕ → ℝ) (S_def : ∀ m, S m = x^m + 1/x^m) :
  S 8 = 37634 :=
sorry

end S8_value_l2217_221707


namespace prove_f_of_increasing_l2217_221765

noncomputable def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

noncomputable def strictly_increasing_on_positives (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, 0 < x₁ ∧ 0 < x₂ ∧ x₁ ≠ x₂ → (f x₁ - f x₂) / (x₁ - x₂) > 0

theorem prove_f_of_increasing {f : ℝ → ℝ}
  (h_odd : odd_function f)
  (h_incr : strictly_increasing_on_positives f) :
  f (-3) > f (-5) :=
by
  sorry

end prove_f_of_increasing_l2217_221765


namespace coefficients_divisible_by_5_l2217_221735

theorem coefficients_divisible_by_5 
  (a b c d : ℤ) 
  (h : ∀ x : ℤ, 5 ∣ (a * x^3 + b * x^2 + c * x + d)) : 
  5 ∣ a ∧ 5 ∣ b ∧ 5 ∣ c ∧ 5 ∣ d := 
by {
  sorry
}

end coefficients_divisible_by_5_l2217_221735


namespace right_triangle_AB_CA_BC_l2217_221795

namespace TriangleProof

def point := ℝ × ℝ

def dist (p1 p2 : point) : ℝ :=
  (p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2

def A : point := (5, -2)
def B : point := (1, 5)
def C : point := (-1, 2)

def AB2 := dist A B
def BC2 := dist B C
def CA2 := dist C A

theorem right_triangle_AB_CA_BC : CA2 + BC2 = AB2 :=
by 
  -- proof will be filled here
  sorry

end TriangleProof

end right_triangle_AB_CA_BC_l2217_221795


namespace slower_speed_is_l2217_221798

def slower_speed_problem
  (faster_speed : ℝ)
  (additional_distance : ℝ)
  (actual_distance : ℝ)
  (v : ℝ) :
  Prop :=
  actual_distance / v = (actual_distance + additional_distance) / faster_speed

theorem slower_speed_is
  (h1 : faster_speed = 25)
  (h2 : additional_distance = 20)
  (h3 : actual_distance = 13.333333333333332)
  : ∃ v : ℝ,  slower_speed_problem faster_speed additional_distance actual_distance v ∧ v = 10 :=
by {
  sorry
}

end slower_speed_is_l2217_221798


namespace cards_per_set_is_13_l2217_221714

-- Definitions based on the conditions
def total_cards : ℕ := 365
def sets_to_brother : ℕ := 8
def sets_to_sister : ℕ := 5
def sets_to_friend : ℕ := 2
def total_sets_given : ℕ := sets_to_brother + sets_to_sister + sets_to_friend
def total_cards_given : ℕ := 195

-- The problem to prove
theorem cards_per_set_is_13 : total_cards_given / total_sets_given = 13 :=
  by
  -- Here we would provide the proof, but for now, we use sorry
  sorry

end cards_per_set_is_13_l2217_221714


namespace pipes_height_l2217_221789

theorem pipes_height (d : ℝ) (h : ℝ) (r : ℝ) (s : ℝ)
  (hd : d = 12)
  (hs : s = d)
  (hr : r = d / 2)
  (heq : h = 6 * Real.sqrt 3 + r) :
  h = 6 * Real.sqrt 3 + 6 :=
by
  sorry

end pipes_height_l2217_221789


namespace sue_answer_is_106_l2217_221742

-- Definitions based on conditions
def ben_step1 (x : ℕ) : ℕ := x * 3
def ben_step2 (x : ℕ) : ℕ := ben_step1 x + 2
def ben_step3 (x : ℕ) : ℕ := ben_step2 x * 2

def sue_step1 (y : ℕ) : ℕ := y + 3
def sue_step2 (y : ℕ) : ℕ := sue_step1 y - 2
def sue_step3 (y : ℕ) : ℕ := sue_step2 y * 2

-- Ben starts with the number 8
def ben_number : ℕ := 8

-- Ben gives the number to Sue
def given_to_sue : ℕ := ben_step3 ben_number

-- Lean statement to prove
theorem sue_answer_is_106 : sue_step3 given_to_sue = 106 :=
by
  sorry

end sue_answer_is_106_l2217_221742


namespace determinant_expr_l2217_221743

theorem determinant_expr (a b c p q r : ℝ) 
  (h1 : ∀ x, Polynomial.eval x (Polynomial.C a * Polynomial.C b * Polynomial.C c - Polynomial.C p * (Polynomial.C a * Polynomial.C b + Polynomial.C b * Polynomial.C c + Polynomial.C c * Polynomial.C a) + Polynomial.C q * (Polynomial.C a + Polynomial.C b + Polynomial.C c) - Polynomial.C r) = 0) :
  Matrix.det ![
    ![2 + a, 1, 1],
    ![1, 2 + b, 1],
    ![1, 1, 2 + c]
  ] = r + 2*q + 4*p + 4 :=
sorry

end determinant_expr_l2217_221743


namespace sum_of_squares_l2217_221717

def gcd (a b c : Nat) : Nat := (Nat.gcd (Nat.gcd a b) c)

theorem sum_of_squares {a b c : ℕ} (h1 : 3 * a + 2 * b = 4 * c)
                                   (h2 : 3 * c ^ 2 = 4 * a ^ 2 + 2 * b ^ 2)
                                   (h3 : gcd a b c = 1) :
  a^2 + b^2 + c^2 = 45 :=
by
  sorry

end sum_of_squares_l2217_221717


namespace triple_integral_value_l2217_221793

theorem triple_integral_value :
  (∫ x in (-1 : ℝ)..1, ∫ y in (x^2 : ℝ)..1, ∫ z in (0 : ℝ)..y, (4 + z) ) = (16 / 3 : ℝ) :=
by
  sorry

end triple_integral_value_l2217_221793


namespace power_function_value_l2217_221749

theorem power_function_value (α : ℝ) (f : ℝ → ℝ) (h₁ : f x = x ^ α) (h₂ : f (1 / 2) = 4) : f 8 = 1 / 64 := by
  sorry

end power_function_value_l2217_221749


namespace set_equality_l2217_221728

noncomputable def alpha_set : Set ℝ := {α | ∃ k : ℤ, α = k * Real.pi / 2 - Real.pi / 5 ∧ (-Real.pi < α ∧ α < Real.pi)}

theorem set_equality : alpha_set = {-Real.pi / 5, -7 * Real.pi / 10, 3 * Real.pi / 10, 4 * Real.pi / 5} :=
by
  -- proof omitted
  sorry

end set_equality_l2217_221728


namespace range_of_a_l2217_221712

noncomputable def f (x : ℝ) : ℝ :=
if x ≥ 0 then x^2 + x else x^2 + x -- Note: Using the specific definition matches the problem constraints clearly.

theorem range_of_a (a : ℝ) (h_even : ∀ x : ℝ, f x = f (-x)) (h_ineq : f a + f (-a) < 4) : -1 < a ∧ a < 1 := 
by sorry

end range_of_a_l2217_221712


namespace linda_total_distance_l2217_221790

theorem linda_total_distance :
  ∃ x : ℕ, (60 % x = 0) ∧ ((75 % (x + 3)) = 0) ∧ ((90 % (x + 6)) = 0) ∧
  (60 / x + 75 / (x + 3) + 90 / (x + 6) = 15) :=
sorry

end linda_total_distance_l2217_221790


namespace avg_rate_change_l2217_221747

def f (x : ℝ) : ℝ := x^2 + x

theorem avg_rate_change : (f 2 - f 1) / (2 - 1) = 4 := by
  -- here the proof steps should follow
  sorry

end avg_rate_change_l2217_221747


namespace find_c_l2217_221754

-- Given conditions
variables {a b c d e : ℕ} (h1 : 0 < a) (h2 : a < b) (h3 : b < c) (h4 : c < d) (h5 : d < e)
variables (h6 : a + b = e - 1) (h7 : a * b = d + 1)

-- Required to prove
theorem find_c : c = 4 := by
  sorry

end find_c_l2217_221754


namespace sum_as_fraction_l2217_221733

theorem sum_as_fraction :
  (0.1 : ℝ) + (0.03 : ℝ) + (0.004 : ℝ) + (0.0006 : ℝ) + (0.00007 : ℝ) = (13467 / 100000 : ℝ) :=
by
  sorry

end sum_as_fraction_l2217_221733


namespace students_scoring_80_percent_l2217_221762

theorem students_scoring_80_percent
  (x : ℕ)
  (h1 : 10 * 90 + x * 80 = 25 * 84)
  (h2 : x + 10 = 25) : x = 15 := 
by {
  -- Proof goes here
  sorry
}

end students_scoring_80_percent_l2217_221762


namespace total_alligators_seen_l2217_221753

-- Definitions for the conditions
def SamaraSaw : Nat := 35
def NumberOfFriends : Nat := 6
def AverageFriendsSaw : Nat := 15

-- Statement of the proof problem
theorem total_alligators_seen :
  SamaraSaw + NumberOfFriends * AverageFriendsSaw = 125 := by
  -- Skipping the proof
  sorry

end total_alligators_seen_l2217_221753


namespace inequality_must_hold_l2217_221773

variable (a b c : ℝ)

theorem inequality_must_hold (h1 : a > b) (h2 : c < 0) : a * (c - 1) < b * (c - 1) := 
sorry

end inequality_must_hold_l2217_221773


namespace robins_initial_hair_length_l2217_221726

variable (L : ℕ)

def initial_length_after_cutting := L - 11
def length_after_growth := initial_length_after_cutting L + 12
def final_length := 17

theorem robins_initial_hair_length : length_after_growth L = final_length → L = 16 := 
by sorry

end robins_initial_hair_length_l2217_221726


namespace rectangle_ratio_l2217_221783

theorem rectangle_ratio (s : ℝ) (x y : ℝ) 
  (h_outer_area : x * y * 4 + s^2 = 9 * s^2)
  (h_inner_outer_relation : s + 2 * y = 3 * s) :
  x / y = 2 :=
by {
  sorry
}

end rectangle_ratio_l2217_221783


namespace total_books_received_l2217_221756

theorem total_books_received (initial_books additional_books total_books: ℕ)
  (h1 : initial_books = 54)
  (h2 : additional_books = 23) :
  (initial_books + additional_books = 77) := by
  sorry

end total_books_received_l2217_221756


namespace even_function_must_be_two_l2217_221702

def f (m : ℝ) (x : ℝ) : ℝ := (m-1)*x^2 + (m-2)*x + (m^2 - 7*m + 12)

theorem even_function_must_be_two (m : ℝ) :
  (∀ x : ℝ, f m (-x) = f m x) ↔ m = 2 :=
by
  sorry

end even_function_must_be_two_l2217_221702


namespace reggie_games_lost_l2217_221760

-- Define the necessary conditions
def initial_marbles : ℕ := 100
def bet_per_game : ℕ := 10
def marbles_after_games : ℕ := 90
def total_games : ℕ := 9

-- Define the proof problem statement
theorem reggie_games_lost : (initial_marbles - marbles_after_games) / bet_per_game = 1 := by
  sorry

end reggie_games_lost_l2217_221760


namespace taxes_paid_l2217_221729

theorem taxes_paid (gross_pay net_pay : ℤ) (h1 : gross_pay = 450) (h2 : net_pay = 315) :
  gross_pay - net_pay = 135 := 
by 
  rw [h1, h2] 
  norm_num

end taxes_paid_l2217_221729


namespace lunchroom_tables_l2217_221799

/-- Given the total number of students and the number of students per table, 
    prove the number of tables in the lunchroom. -/
theorem lunchroom_tables (total_students : ℕ) (students_per_table : ℕ) 
  (h_total : total_students = 204) (h_per_table : students_per_table = 6) : 
  total_students / students_per_table = 34 := 
by
  sorry

end lunchroom_tables_l2217_221799


namespace two_lines_intersections_with_ellipse_l2217_221740

open Set

def ellipse (x y : ℝ) : Prop := x^2 + y^2 = 1

theorem two_lines_intersections_with_ellipse {L1 L2 : ℝ → ℝ → Prop} :
  (∀ x y, L1 x y → ¬(ellipse x y)) →
  (∀ x y, L2 x y → ¬(ellipse x y)) →
  (∃ x1 y1 x2 y2, x1 ≠ x2 ∧ y1 ≠ y2 ∧ ellipse x1 y1 ∧ ellipse x2 y2 ∧ L1 x1 y1 ∧ L1 x2 y2) →
  (∃ x1 y1 x2 y2, x1 ≠ x2 ∧ y1 ≠ y2 ∧ ellipse x1 y1 ∧ ellipse x2 y2 ∧ L2 x1 y1 ∧ L2 x2 y2) →
  ∃ n, n = 2 ∨ n = 4 :=
by
  sorry

end two_lines_intersections_with_ellipse_l2217_221740


namespace b_2016_result_l2217_221731

theorem b_2016_result (b : ℕ → ℤ) (h₁ : b 1 = 1) (h₂ : b 2 = 5)
  (h₃ : ∀ n : ℕ, b (n + 2) = b (n + 1) - b n) : b 2016 = -4 := sorry

end b_2016_result_l2217_221731


namespace percent_of_y_l2217_221719

theorem percent_of_y (y : ℝ) (h : y > 0) : (2 * y) / 10 + (3 * y) / 10 = (50 / 100) * y :=
by
  sorry

end percent_of_y_l2217_221719


namespace evaluate_f_at_4_l2217_221752

def f (x : ℝ) : ℝ := x^2 - 2*x + 1

theorem evaluate_f_at_4 : f 4 = 9 := by
  sorry

end evaluate_f_at_4_l2217_221752


namespace solve_for_x_l2217_221782

theorem solve_for_x (x : ℝ) (h : 1 = 1 / (4 * x^2 + 2 * x + 1)) : 
  x = 0 ∨ x = -1 / 2 := 
by sorry

end solve_for_x_l2217_221782


namespace intersection_eq_l2217_221761

noncomputable def A := {x : ℝ | x^2 - 4*x + 3 < 0 }
noncomputable def B := {x : ℝ | 2*x - 3 > 0 }

theorem intersection_eq : (A ∩ B) = {x : ℝ | (3 / 2) < x ∧ x < 3} := by
  sorry

end intersection_eq_l2217_221761


namespace calculate_expression_l2217_221770

theorem calculate_expression : 
  let x := 7.5
  let y := 2.5
  (x ^ y + Real.sqrt x + y ^ x) - (x ^ 2 + y ^ y + Real.sqrt y) = 679.2044 :=
by
  sorry

end calculate_expression_l2217_221770


namespace minimize_sum_find_c_l2217_221788

theorem minimize_sum_find_c (a b c d e f : ℕ) (h : a + 2 * b + 6 * c + 30 * d + 210 * e + 2310 * f = 2 ^ 15) 
  (h_min : ∀ a' b' c' d' e' f' : ℕ, a' + 2 * b' + 6 * c' + 30 * d' + 210 * e' + 2310 * f' = 2 ^ 15 → 
  a' + b' + c' + d' + e' + f' ≥ a + b + c + d + e + f) :
  c = 1 :=
sorry

end minimize_sum_find_c_l2217_221788


namespace sum_of_products_leq_one_third_l2217_221703

theorem sum_of_products_leq_one_third (a b c : ℝ) (h : a + b + c = 1) : 
  ab + bc + ca ≤ 1 / 3 :=
sorry

end sum_of_products_leq_one_third_l2217_221703


namespace sum_first_99_terms_l2217_221724

def geom_sum (n : ℕ) : ℕ := (2^n) - 1

def seq_sum (n : ℕ) : ℕ :=
  (Finset.range n).sum geom_sum

theorem sum_first_99_terms :
  seq_sum 99 = 2^100 - 101 := by
  sorry

end sum_first_99_terms_l2217_221724


namespace number_of_possible_IDs_l2217_221725

theorem number_of_possible_IDs : 
  ∃ (n : ℕ), 
  (∀ (a b : Fin 26) (x y : Fin 10),
    a = b ∨ x = y ∨ (a = b ∧ x = y) → 
    n = 9100) :=
sorry

end number_of_possible_IDs_l2217_221725


namespace solution_k_system_eq_l2217_221738

theorem solution_k_system_eq (x y k : ℝ) 
  (h1 : x + y = 5 * k) 
  (h2 : x - y = k) 
  (h3 : 2 * x + 3 * y = 24) : k = 2 :=
by
  sorry

end solution_k_system_eq_l2217_221738


namespace yellow_marbles_l2217_221704

-- Define the conditions from a)
variables (total_marbles red blue green yellow : ℕ)
variables (h1 : total_marbles = 110)
variables (h2 : red = 8)
variables (h3 : blue = 4 * red)
variables (h4 : green = 2 * blue)
variables (h5 : yellow = total_marbles - (red + blue + green))

-- Prove the question in c)
theorem yellow_marbles : yellow = 6 :=
by
  -- Proof will be inserted here
  sorry

end yellow_marbles_l2217_221704


namespace probability_four_of_eight_show_three_l2217_221764

def probability_exactly_four_show_three : ℚ :=
  let num_ways := Nat.choose 8 4
  let prob_four_threes := (1 / 6) ^ 4
  let prob_four_not_threes := (5 / 6) ^ 4
  (num_ways * prob_four_threes * prob_four_not_threes)

theorem probability_four_of_eight_show_three :
  probability_exactly_four_show_three = 43750 / 1679616 :=
by 
  sorry

end probability_four_of_eight_show_three_l2217_221764


namespace find_y_value_l2217_221785

def op (a b : ℤ) : ℤ := 4 * a + 2 * b

theorem find_y_value : ∃ y : ℤ, op 3 (op 4 y) = -14 ∧ y = -29 / 2 := sorry

end find_y_value_l2217_221785


namespace new_total_lines_is_240_l2217_221739

-- Define the original number of lines, the increase, and the percentage increase
variables (L : ℝ) (increase : ℝ := 110) (percentage_increase : ℝ := 84.61538461538461 / 100)

-- The statement to prove
theorem new_total_lines_is_240 (h : increase = percentage_increase * L) : L + increase = 240 := sorry

end new_total_lines_is_240_l2217_221739


namespace smallest_sum_of_xy_l2217_221709

namespace MathProof

theorem smallest_sum_of_xy (x y : ℕ) (hx : x ≠ y) (hxy : (1 : ℝ) / x + (1 : ℝ) / y = 1 / 12) : x + y = 49 :=
sorry

end MathProof

end smallest_sum_of_xy_l2217_221709


namespace maximum_members_in_dance_troupe_l2217_221751

theorem maximum_members_in_dance_troupe (m : ℕ) (h1 : 25 * m % 31 = 7) (h2 : 25 * m < 1300) : 25 * m = 875 :=
by {
  sorry
}

end maximum_members_in_dance_troupe_l2217_221751


namespace rectangle_diagonal_length_l2217_221771

theorem rectangle_diagonal_length :
  ∀ (length width diagonal : ℝ), length = 6 ∧ length * width = 48 ∧ diagonal = Real.sqrt (length^2 + width^2) → diagonal = 10 :=
by
  intro length width diagonal
  rintro ⟨hl, area_eq, diagonal_eq⟩
  sorry

end rectangle_diagonal_length_l2217_221771


namespace sector_area_l2217_221769

theorem sector_area (θ r a : ℝ) (hθ : θ = 2) (haarclength : r * θ = 4) : 
  (1/2) * r * r * θ = 4 :=
by {
  -- Proof goes here
  sorry
}

end sector_area_l2217_221769


namespace loss_percentage_l2217_221784

theorem loss_percentage
  (CP : ℝ := 1166.67)
  (SP : ℝ)
  (H : SP + 140 = CP + 0.02 * CP) :
  ((CP - SP) / CP) * 100 = 10 := 
by 
  sorry

end loss_percentage_l2217_221784
