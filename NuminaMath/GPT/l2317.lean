import Mathlib

namespace NUMINAMATH_GPT_range_of_a_minus_b_l2317_231764

theorem range_of_a_minus_b (a b : ℝ) (ha : 1 < a ∧ a < 4) (hb : -2 < b ∧ b < 4) : 
  -3 < a - b ∧ a - b < 6 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_minus_b_l2317_231764


namespace NUMINAMATH_GPT_benjamin_skating_time_l2317_231700

-- Defining the conditions
def distance : ℕ := 80 -- Distance in kilometers
def speed : ℕ := 10   -- Speed in kilometers per hour

-- The main theorem statement
theorem benjamin_skating_time : ∀ (T : ℕ), T = distance / speed → T = 8 := by
  sorry

end NUMINAMATH_GPT_benjamin_skating_time_l2317_231700


namespace NUMINAMATH_GPT_time_after_1450_minutes_l2317_231781

theorem time_after_1450_minutes (initial_time_in_minutes : ℕ := 360) (minutes_to_add : ℕ := 1450) : 
  (initial_time_in_minutes + minutes_to_add) % (24 * 60) = 370 :=
by
  -- Given (initial_time_in_minutes = 360 which is 6:00 a.m., minutes_to_add = 1450)
  -- Compute the time in minutes after 1450 minutes
  -- 24 hours = 1440 minutes, so (360 + 1450) % 1440 should equal 370
  sorry

end NUMINAMATH_GPT_time_after_1450_minutes_l2317_231781


namespace NUMINAMATH_GPT_integer_sequence_existence_l2317_231767

theorem integer_sequence_existence
  (n : ℕ) (a : ℕ → ℤ) (A B C : ℤ) 
  (h1 : (a 1 < A ∧ A < B ∧ B < a n) ∨ (a 1 > A ∧ A > B ∧ B > a n))
  (h2 : ∀ i, 1 ≤ i ∧ i ≤ n - 1 → (a (i + 1) - a i ≤ 1 ∨ a (i + 1) - a i ≥ -1))
  (h3 : A ≤ C ∧ C ≤ B ∨ A ≥ C ∧ C ≥ B) :
  ∃ i, 1 < i ∧ i < n ∧ a i = C := sorry

end NUMINAMATH_GPT_integer_sequence_existence_l2317_231767


namespace NUMINAMATH_GPT_black_squares_in_35th_row_l2317_231706

-- Define the condition for the starting color based on the row
def starts_with_black (n : ℕ) : Prop := n % 2 = 1
def ends_with_white (n : ℕ) : Prop := true  -- This is trivially true by the problem condition
def total_squares (n : ℕ) : ℕ := 2 * n 
-- Black squares are half of the total squares for rows starting with a black square
def black_squares (n : ℕ) : ℕ := total_squares n / 2

theorem black_squares_in_35th_row : black_squares 35 = 35 :=
sorry

end NUMINAMATH_GPT_black_squares_in_35th_row_l2317_231706


namespace NUMINAMATH_GPT_oranges_per_box_l2317_231756

theorem oranges_per_box
  (total_oranges : ℕ)
  (boxes : ℕ)
  (h1 : total_oranges = 35)
  (h2 : boxes = 7) :
  total_oranges / boxes = 5 := by
  sorry

end NUMINAMATH_GPT_oranges_per_box_l2317_231756


namespace NUMINAMATH_GPT_fraction_value_l2317_231754

-- Define the constants
def eight := 8
def four := 4

-- Statement to prove
theorem fraction_value : (eight + four) / (eight - four) = 3 := 
by
  sorry

end NUMINAMATH_GPT_fraction_value_l2317_231754


namespace NUMINAMATH_GPT_compute_quotient_of_q_and_r_l2317_231702

theorem compute_quotient_of_q_and_r (p q r s t : ℤ) (h_eq_4 : 256 * p + 64 * q + 16 * r + 4 * s + t = 0)
                                     (h_eq_neg3 : -27 * p + 9 * q - 3 * r + s + t = 0)
                                     (h_eq_0 : t = 0)
                                     (h_p_nonzero : p ≠ 0) :
                                     (q + r) / p = -13 :=
by
  have eq1 := h_eq_4
  have eq2 := h_eq_neg3
  rw [h_eq_0] at eq1 eq2
  sorry

end NUMINAMATH_GPT_compute_quotient_of_q_and_r_l2317_231702


namespace NUMINAMATH_GPT_total_population_l2317_231794

theorem total_population (n : ℕ) (avg_population : ℕ) (h1 : n = 20) (h2 : avg_population = 4750) :
  n * avg_population = 95000 := by
  subst_vars
  sorry

end NUMINAMATH_GPT_total_population_l2317_231794


namespace NUMINAMATH_GPT_sum_of_all_possible_values_of_g10_l2317_231770

noncomputable def g : ℕ → ℝ := sorry

axiom h1 : g 1 = 2
axiom h2 : ∀ m n : ℕ, m ≥ n → g (m + n) + g (m - n) = 3 * (g m + g n)
axiom h3 : g 0 = 0

theorem sum_of_all_possible_values_of_g10 : g 10 = 59028 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_all_possible_values_of_g10_l2317_231770


namespace NUMINAMATH_GPT_floor_expression_equals_zero_l2317_231782

theorem floor_expression_equals_zero
  (a b c : ℕ)
  (ha : a = 2010)
  (hb : b = 2007)
  (hc : c = 2008) :
  Int.floor ((a^3 : ℚ) / (b * c^2) - (c^3 : ℚ) / (b^2 * a)) = 0 := 
  sorry

end NUMINAMATH_GPT_floor_expression_equals_zero_l2317_231782


namespace NUMINAMATH_GPT_inequality_true_l2317_231735

variables {a b : ℝ}
variables (c : ℝ)

theorem inequality_true (ha : a ≠ 0) (hb : b ≠ 0) (hab : a < b) : a^3 * b^2 < a^2 * b^3 :=
by sorry

end NUMINAMATH_GPT_inequality_true_l2317_231735


namespace NUMINAMATH_GPT_total_wheels_l2317_231743

-- Definitions of given conditions
def bicycles : ℕ := 50
def tricycles : ℕ := 20
def wheels_per_bicycle : ℕ := 2
def wheels_per_tricycle : ℕ := 3

-- Theorem stating the total number of wheels for bicycles and tricycles combined
theorem total_wheels : bicycles * wheels_per_bicycle + tricycles * wheels_per_tricycle = 160 :=
by
  sorry

end NUMINAMATH_GPT_total_wheels_l2317_231743


namespace NUMINAMATH_GPT_a5_a6_val_l2317_231703

variable (a : ℕ → ℝ)
variable (r : ℝ)

axiom geom_seq (n : ℕ) : a (n + 1) = a n * r
axiom pos_seq (n : ℕ) : a n > 0

axiom a1_a2 : a 1 + a 2 = 1
axiom a3_a4 : a 3 + a 4 = 9

theorem a5_a6_val :
  a 5 + a 6 = 81 :=
by
  sorry

end NUMINAMATH_GPT_a5_a6_val_l2317_231703


namespace NUMINAMATH_GPT_boys_without_calculators_l2317_231728

-- Definitions based on the conditions
def total_boys : Nat := 20
def students_with_calculators : Nat := 26
def girls_with_calculators : Nat := 15

-- We need to prove the number of boys who did not bring their calculators.
theorem boys_without_calculators : (total_boys - (students_with_calculators - girls_with_calculators)) = 9 :=
by {
    -- Proof goes here
    sorry
}

end NUMINAMATH_GPT_boys_without_calculators_l2317_231728


namespace NUMINAMATH_GPT_set_contains_all_nonnegative_integers_l2317_231741

theorem set_contains_all_nonnegative_integers (S : Set ℕ) :
  (∃ a b, a ∈ S ∧ b ∈ S ∧ 1 < a ∧ 1 < b ∧ Nat.gcd a b = 1) →
  (∀ x y, x ∈ S → y ∈ S → y ≠ 0 → (x * y) ∈ S ∧ (x % y) ∈ S) →
  (∀ n, n ∈ S) :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_set_contains_all_nonnegative_integers_l2317_231741


namespace NUMINAMATH_GPT_polynomial_form_l2317_231787

theorem polynomial_form (P : Polynomial ℝ) (hP : P ≠ 0)
    (h : ∀ x : ℝ, P.eval x * P.eval (2 * x^2) = P.eval (2 * x^3 + x)) :
    ∃ k : ℕ, k > 0 ∧ P = (X^2 + 1) ^ k :=
by sorry

end NUMINAMATH_GPT_polynomial_form_l2317_231787


namespace NUMINAMATH_GPT_estimated_probability_is_2_div_9_l2317_231746

def groups : List (List ℕ) :=
  [[3, 4, 3], [4, 3, 2], [3, 4, 1], [3, 4, 2], [2, 3, 4], [1, 4, 2], [2, 4, 3], [3, 3, 1], [1, 1, 2],
   [3, 4, 2], [2, 4, 1], [2, 4, 4], [4, 3, 1], [2, 3, 3], [2, 1, 4], [3, 4, 4], [1, 4, 2], [1, 3, 4]]

def count_desired_groups (gs : List (List ℕ)) : Nat :=
  gs.foldl (fun acc g =>
    if g.contains 1 ∧ g.contains 2 ∧ g.length ≥ 3 then acc + 1 else acc) 0

theorem estimated_probability_is_2_div_9 :
  (count_desired_groups groups) = 4 →
  4 / 18 = 2 / 9 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_estimated_probability_is_2_div_9_l2317_231746


namespace NUMINAMATH_GPT_flower_beds_fraction_correct_l2317_231701

noncomputable def flower_beds_fraction (yard_length : ℝ) (yard_width : ℝ) (trapezoid_parallel_side1 : ℝ) (trapezoid_parallel_side2 : ℝ) : ℝ :=
  let leg_length := (trapezoid_parallel_side2 - trapezoid_parallel_side1) / 2
  let triangle_area := (1 / 2) * leg_length^2
  let total_flower_bed_area := 2 * triangle_area
  let yard_area := yard_length * yard_width
  total_flower_bed_area / yard_area

theorem flower_beds_fraction_correct :
  flower_beds_fraction 30 5 20 30 = 1 / 6 :=
by
  sorry

end NUMINAMATH_GPT_flower_beds_fraction_correct_l2317_231701


namespace NUMINAMATH_GPT_range_of_largest_root_l2317_231748

theorem range_of_largest_root :
  ∀ (a_2 a_1 a_0 : ℝ), 
  (|a_2| ≤ 1 ∧ |a_1| ≤ 1 ∧ |a_0| ≤ 1) ∧ (a_2 + a_1 + a_0 = 0) →
  (∃ s > 1, ∀ x > 0, x^3 + 3*a_2*x^2 + 5*a_1*x + a_0 = 0 → x ≤ s) ∧
  (s < 2) :=
by sorry

end NUMINAMATH_GPT_range_of_largest_root_l2317_231748


namespace NUMINAMATH_GPT_inequality_inverse_l2317_231783

theorem inequality_inverse (a b : ℝ) (h : a > b ∧ b > 0) : (1 / a) < (1 / b) :=
by
  sorry

end NUMINAMATH_GPT_inequality_inverse_l2317_231783


namespace NUMINAMATH_GPT_cost_of_painting_murals_l2317_231710

def first_mural_area : ℕ := 20 * 15
def second_mural_area : ℕ := 25 * 10
def third_mural_area : ℕ := 30 * 8

def first_mural_time : ℕ := first_mural_area * 20
def second_mural_time : ℕ := second_mural_area * 25
def third_mural_time : ℕ := third_mural_area * 30

def total_time : ℚ := (first_mural_time + second_mural_time + third_mural_time) / 60

def total_area : ℕ := first_mural_area + second_mural_area + third_mural_area

def cost (area : ℕ) : ℚ :=
  if area <= 100 then area * 150 else 
  if area <= 300 then 100 * 150 + (area - 100) * 175 
  else 100 * 150 + 200 * 175 + (area - 300) * 200

def total_cost : ℚ := cost total_area

theorem cost_of_painting_murals :
  total_cost = 148000 := by
  sorry

end NUMINAMATH_GPT_cost_of_painting_murals_l2317_231710


namespace NUMINAMATH_GPT_radius_condition_l2317_231765

def X (x y : ℝ) : ℝ := 12 * x
def Y (x y : ℝ) : ℝ := 5 * y

def satisfies_condition (x y : ℝ) : Prop :=
  Real.sin (X x y + Y x y) = Real.sin (X x y) + Real.sin (Y x y)

def no_intersection (R : ℝ) : Prop :=
  ∀ (x y : ℝ), satisfies_condition x y → dist (0, 0) (x, y) ≥ R

theorem radius_condition :
  ∀ R : ℝ, (0 < R ∧ R < Real.pi / 15) →
  no_intersection R :=
sorry

end NUMINAMATH_GPT_radius_condition_l2317_231765


namespace NUMINAMATH_GPT_find_f_13_l2317_231785

noncomputable def f : ℝ → ℝ := sorry

axiom f_property : ∀ x, f (x + f x) = 3 * f x
axiom f_of_1 : f 1 = 3

theorem find_f_13 : f 13 = 27 :=
by
  have hf := f_property
  have hf1 := f_of_1
  sorry

end NUMINAMATH_GPT_find_f_13_l2317_231785


namespace NUMINAMATH_GPT_vector_subtraction_proof_l2317_231769

def v1 : ℝ × ℝ := (3, -4)
def v2 : ℝ × ℝ := (1, -6)
def scalar1 : ℝ := 2
def scalar2 : ℝ := 3

theorem vector_subtraction_proof :
  v1 - (scalar2 • (scalar1 • v2)) = (-3, 32) := by
  sorry

end NUMINAMATH_GPT_vector_subtraction_proof_l2317_231769


namespace NUMINAMATH_GPT_angle_PMN_is_60_l2317_231776

-- Define given variables and their types
variable (P M N R Q : Prop)
variable (angle : Prop → Prop → Prop → ℝ)

-- Given conditions
variables (h1 : angle P Q R = 60)
variables (h2 : PM = MN)

-- The statement of what's to be proven
theorem angle_PMN_is_60 :
  angle P M N = 60 := sorry

end NUMINAMATH_GPT_angle_PMN_is_60_l2317_231776


namespace NUMINAMATH_GPT_part1_infinite_n_part2_no_solutions_l2317_231762

-- Definitions for part (1)
theorem part1_infinite_n (n : ℕ) (x y z t : ℕ) :
  (∃ n, x ^ 2 + y ^ 2 + z ^ 2 + t ^ 2 - n * x * y * z * t - n = 0) :=
  sorry

-- Definitions for part (2)
theorem part2_no_solutions (n k m x y z t : ℕ) :
  n = 4 ^ k * (8 * m + 7) → ¬(x ^ 2 + y ^ 2 + z ^ 2 + t ^ 2 - n * x * y * z * t - n = 0) :=
  sorry

end NUMINAMATH_GPT_part1_infinite_n_part2_no_solutions_l2317_231762


namespace NUMINAMATH_GPT_solve_fractional_eq_l2317_231758

theorem solve_fractional_eq (x : ℝ) (h_non_zero : x ≠ 0) (h_non_neg_one : x ≠ -1) :
  (2 / x = 1 / (x + 1)) → x = -2 :=
by
  intro h_eq
  sorry

end NUMINAMATH_GPT_solve_fractional_eq_l2317_231758


namespace NUMINAMATH_GPT_tangents_intersection_perpendicular_parabola_l2317_231713

theorem tangents_intersection_perpendicular_parabola :
  ∀ (C D : ℝ × ℝ), C.2 = 4 * C.1 ^ 2 → D.2 = 4 * D.1 ^ 2 → 
  (8 * C.1) * (8 * D.1) = -1 → 
  ∃ Q : ℝ × ℝ, Q.2 = -1 / 16 :=
by
  sorry

end NUMINAMATH_GPT_tangents_intersection_perpendicular_parabola_l2317_231713


namespace NUMINAMATH_GPT_problem1_problem2_l2317_231705

-- Define the first problem
theorem problem1 : (Real.cos (25 / 3 * Real.pi) + Real.tan (-15 / 4 * Real.pi)) = 3 / 2 :=
by
  sorry

-- Define vector operations and the problem
variables (a b : ℝ)

theorem problem2 : 2 * (a - b) - (2 * a + b) + 3 * b = 0 :=
by
  sorry

end NUMINAMATH_GPT_problem1_problem2_l2317_231705


namespace NUMINAMATH_GPT_part1_l2317_231744

theorem part1 (a b c : ℝ) (h : 1 / (a + b) + 1 / (b + c) = 2 / (c + a)) : 2 * b^2 = a^2 + c^2 :=
sorry

end NUMINAMATH_GPT_part1_l2317_231744


namespace NUMINAMATH_GPT_sunzi_classic_equation_l2317_231792

theorem sunzi_classic_equation (x : ℕ) : 3 * (x - 2) = 2 * x + 9 :=
  sorry

end NUMINAMATH_GPT_sunzi_classic_equation_l2317_231792


namespace NUMINAMATH_GPT_abs_sub_eq_abs_sub_l2317_231793

theorem abs_sub_eq_abs_sub (a b : ℚ) : |a - b| = |b - a| :=
sorry

end NUMINAMATH_GPT_abs_sub_eq_abs_sub_l2317_231793


namespace NUMINAMATH_GPT_shaded_region_area_l2317_231786

noncomputable def radius1 := 4
noncomputable def radius2 := 5
noncomputable def distance := radius1 + radius2
noncomputable def large_radius := radius2 + distance / 2

theorem shaded_region_area :
  ∃ (A : ℝ), A = (π * large_radius ^ 2) - (π * radius1 ^ 2) - (π * radius2 ^ 2) ∧
  A = 49.25 * π :=
by
  sorry

end NUMINAMATH_GPT_shaded_region_area_l2317_231786


namespace NUMINAMATH_GPT_proposition_4_l2317_231779

variables {Line Plane : Type}
variables {a b : Line} {α β : Plane}

-- Definitions of parallel and perpendicular relationships
class Parallel (l : Line) (p : Plane) : Prop
class Perpendicular (l : Line) (p : Plane) : Prop
class Contains (p : Plane) (l : Line) : Prop

theorem proposition_4
  (h1: Perpendicular a β)
  (h2: Parallel a b)
  (h3: Contains α b) : Perpendicular α β :=
sorry

end NUMINAMATH_GPT_proposition_4_l2317_231779


namespace NUMINAMATH_GPT_unique_solution_for_all_y_l2317_231749

theorem unique_solution_for_all_y (x : ℝ) (h : ∀ y : ℝ, 8 * x * y - 12 * y + 2 * x - 3 = 0) : x = 3 / 2 :=
sorry

end NUMINAMATH_GPT_unique_solution_for_all_y_l2317_231749


namespace NUMINAMATH_GPT_total_presents_l2317_231716

-- Definitions based on the problem conditions
def numChristmasPresents : ℕ := 60
def numBirthdayPresents : ℕ := numChristmasPresents / 2

-- Theorem statement
theorem total_presents : numChristmasPresents + numBirthdayPresents = 90 :=
by
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_total_presents_l2317_231716


namespace NUMINAMATH_GPT_line_equation_correct_l2317_231721

-- Definitions for the conditions
def point := ℝ × ℝ
def vector := ℝ × ℝ

-- Given the line has a direction vector and passes through a point
def line_has_direction_vector (l : point → Prop) (v : vector) : Prop :=
  ∀ p₁ p₂ : point, l p₁ → l p₂ → (p₂.1 - p₁.1, p₂.2 - p₁.2) = v

def line_passes_through_point (l : point → Prop) (p : point) : Prop :=
  l p

-- The line equation in point-direction form
def line_equation (x y : ℝ) : Prop :=
  (x - 1) / 2 = y / -3

-- Main statement
theorem line_equation_correct :
  ∃ l : point → Prop, 
    line_has_direction_vector l (2, -3) ∧
    line_passes_through_point l (1, 0) ∧
    ∀ x y, l (x, y) ↔ line_equation x y := 
sorry

end NUMINAMATH_GPT_line_equation_correct_l2317_231721


namespace NUMINAMATH_GPT_magnitude_of_linear_combination_is_sqrt_65_l2317_231784

noncomputable def vector_a (m : ℝ) : ℝ × ℝ := (2 * m - 1, 2)
noncomputable def vector_b (m : ℝ) : ℝ × ℝ := (-2, 3 * m - 2)
noncomputable def perpendicular (u v : ℝ × ℝ) : Prop := (u.1 * v.1 + u.2 * v.2 = 0)

theorem magnitude_of_linear_combination_is_sqrt_65 (m : ℝ) 
  (h_perpendicular : perpendicular (vector_a m) (vector_b m)) : 
  ‖((2 : ℝ) • (vector_a 1) - (3 : ℝ) • (vector_b 1))‖ = Real.sqrt 65 := 
by
  sorry

end NUMINAMATH_GPT_magnitude_of_linear_combination_is_sqrt_65_l2317_231784


namespace NUMINAMATH_GPT_hexagon_cyclic_identity_l2317_231737

variables (a a' b b' c c' a₁ b₁ c₁ : ℝ)

theorem hexagon_cyclic_identity :
  a₁ * b₁ * c₁ = a * b * c + a' * b' * c' + a * a' * a₁ + b * b' * b₁ + c * c' * c₁ :=
by
  sorry

end NUMINAMATH_GPT_hexagon_cyclic_identity_l2317_231737


namespace NUMINAMATH_GPT_option_C_is_neither_even_nor_odd_l2317_231738

noncomputable def f_A (x : ℝ) : ℝ := x^2 + |x|
noncomputable def f_B (x : ℝ) : ℝ := 2^x - 2^(-x)
noncomputable def f_C (x : ℝ) : ℝ := x^2 - 3^x
noncomputable def f_D (x : ℝ) : ℝ := 1/(x+1) + 1/(x-1)

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x
def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = - (f x)

theorem option_C_is_neither_even_nor_odd : ¬ is_even f_C ∧ ¬ is_odd f_C :=
by
  sorry

end NUMINAMATH_GPT_option_C_is_neither_even_nor_odd_l2317_231738


namespace NUMINAMATH_GPT_A_plus_B_eq_one_fourth_l2317_231725

noncomputable def A : ℚ := 1 / 3
noncomputable def B : ℚ := -1 / 12

theorem A_plus_B_eq_one_fourth :
  A + B = 1 / 4 := by
  sorry

end NUMINAMATH_GPT_A_plus_B_eq_one_fourth_l2317_231725


namespace NUMINAMATH_GPT_divisible_by_9_l2317_231736

theorem divisible_by_9 (k : ℕ) (h : k > 0) : 9 ∣ 3 * (2 + 7^k) :=
sorry

end NUMINAMATH_GPT_divisible_by_9_l2317_231736


namespace NUMINAMATH_GPT_largest_radius_cone_l2317_231717

structure Crate :=
  (width : ℝ)
  (depth : ℝ)
  (height : ℝ)

structure Cone :=
  (radius : ℝ)
  (height : ℝ)

noncomputable def larger_fit_within_crate (c : Crate) (cone : Cone) : Prop :=
  cone.radius = min c.width c.depth / 2 ∧ cone.height = max (max c.width c.depth) c.height

theorem largest_radius_cone (c : Crate) (cone : Cone) : 
  c.width = 5 → c.depth = 8 → c.height = 12 → larger_fit_within_crate c cone → cone.radius = 2.5 :=
by
  sorry

end NUMINAMATH_GPT_largest_radius_cone_l2317_231717


namespace NUMINAMATH_GPT_Jillian_largest_apartment_size_l2317_231774

noncomputable def largest_apartment_size (budget rent_per_sqft: ℝ) : ℝ :=
  budget / rent_per_sqft

theorem Jillian_largest_apartment_size :
  largest_apartment_size 720 1.20 = 600 := 
by
  sorry

end NUMINAMATH_GPT_Jillian_largest_apartment_size_l2317_231774


namespace NUMINAMATH_GPT_gcd_97_power_l2317_231771

theorem gcd_97_power (h : Nat.Prime 97) : 
  Nat.gcd (97^7 + 1) (97^7 + 97^3 + 1) = 1 := 
by 
  sorry

end NUMINAMATH_GPT_gcd_97_power_l2317_231771


namespace NUMINAMATH_GPT_min_adults_at_amusement_park_l2317_231797

def amusement_park_problem : Prop :=
  ∃ (x y z : ℕ), 
    x + y + z = 100 ∧
    3 * x + 2 * y + (3 / 10) * z = 100 ∧
    (∀ (x' : ℕ), x' < 2 → ¬(∃ (y' z' : ℕ), x' + y' + z' = 100 ∧ 3 * x' + 2 * y' + (3 / 10) * z' = 100))

theorem min_adults_at_amusement_park : amusement_park_problem := sorry

end NUMINAMATH_GPT_min_adults_at_amusement_park_l2317_231797


namespace NUMINAMATH_GPT_sum_of_roots_l2317_231722
-- Import Mathlib to cover all necessary functionality.

-- Define the function representing the given equation.
def equation (x : ℝ) : ℝ :=
  (3 * x + 4) * (x - 5) + (3 * x + 4) * (x - 7)

-- State the theorem to be proved.
theorem sum_of_roots : (6 : ℝ) + (-4 / 3) = 14 / 3 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_roots_l2317_231722


namespace NUMINAMATH_GPT_bun_eating_problem_l2317_231733

theorem bun_eating_problem
  (n k : ℕ)
  (H1 : 5 * n / 10 + 3 * k / 10 = 180) -- This corresponds to the condition that Zhenya eats 5 buns in 10 minutes, and Sasha eats 3 buns in 10 minutes, for a total of 180 minutes.
  (H2 : n + k = 70) -- This corresponds to the total number of buns eaten.
  : n = 40 ∧ k = 30 :=
by
  sorry

end NUMINAMATH_GPT_bun_eating_problem_l2317_231733


namespace NUMINAMATH_GPT_seating_arrangements_l2317_231712

theorem seating_arrangements (n m k : Nat) (couples : Fin n -> Fin m -> Prop):
  let pairs : Nat := k
  let adjusted_pairs : Nat := pairs / 24
  adjusted_pairs = 5760 := by
  sorry

end NUMINAMATH_GPT_seating_arrangements_l2317_231712


namespace NUMINAMATH_GPT_value_of_business_l2317_231731

-- Defining the conditions
def owns_shares : ℚ := 2/3
def sold_fraction : ℚ := 3/4 
def sold_amount : ℝ := 75000 

-- The final proof statement
theorem value_of_business : 
  (owns_shares * sold_fraction) * value = sold_amount →
  value = 150000 :=
by
  sorry

end NUMINAMATH_GPT_value_of_business_l2317_231731


namespace NUMINAMATH_GPT_p_and_q_together_complete_in_10_days_l2317_231753

noncomputable def p_time := 50 / 3
noncomputable def q_time := 25
noncomputable def r_time := 50

theorem p_and_q_together_complete_in_10_days 
  (h1 : 1 / p_time = 1 / q_time + 1 / r_time)
  (h2 : r_time = 50)
  (h3 : q_time = 25) :
  (p_time * q_time) / (p_time + q_time) = 10 :=
by
  sorry

end NUMINAMATH_GPT_p_and_q_together_complete_in_10_days_l2317_231753


namespace NUMINAMATH_GPT_solve_equation_correctly_l2317_231726

theorem solve_equation_correctly : 
  ∀ x : ℝ, (x - 1) / 2 - 1 = (2 * x + 1) / 3 → x = -11 :=
by
  intro x h
  sorry

end NUMINAMATH_GPT_solve_equation_correctly_l2317_231726


namespace NUMINAMATH_GPT_water_tank_full_capacity_l2317_231791

theorem water_tank_full_capacity (x : ℝ) (h1 : x * (3/4) - x * (1/3) = 15) : x = 36 := 
by
  sorry

end NUMINAMATH_GPT_water_tank_full_capacity_l2317_231791


namespace NUMINAMATH_GPT_scientific_notation_of_87000000_l2317_231759

theorem scientific_notation_of_87000000 :
  87000000 = 8.7 * 10^7 := 
sorry

end NUMINAMATH_GPT_scientific_notation_of_87000000_l2317_231759


namespace NUMINAMATH_GPT_find_rate_of_interest_l2317_231755

def simple_interest (principal : ℕ) (rate : ℕ) (time : ℕ) : ℕ :=
  principal * rate * time / 100

theorem find_rate_of_interest :
  ∀ (R : ℕ),
  simple_interest 5000 R 2 + simple_interest 3000 R 4 = 2640 → R = 12 :=
by
  intros R h
  sorry

end NUMINAMATH_GPT_find_rate_of_interest_l2317_231755


namespace NUMINAMATH_GPT_gretchen_rachelle_ratio_l2317_231798

-- Definitions of the conditions
def rachelle_pennies : ℕ := 180
def total_pennies : ℕ := 300
def rocky_pennies (gretchen_pennies : ℕ) : ℕ := gretchen_pennies / 3

-- The Lean 4 theorem statement
theorem gretchen_rachelle_ratio (gretchen_pennies : ℕ) 
    (h_total : rachelle_pennies + gretchen_pennies + rocky_pennies gretchen_pennies = total_pennies) :
    (gretchen_pennies : ℚ) / rachelle_pennies = 1 / 2 :=
sorry

end NUMINAMATH_GPT_gretchen_rachelle_ratio_l2317_231798


namespace NUMINAMATH_GPT_election_valid_vote_counts_l2317_231760

noncomputable def totalVotes : ℕ := 900000
noncomputable def invalidPercentage : ℝ := 0.25
noncomputable def validVotes : ℝ := totalVotes * (1.0 - invalidPercentage)
noncomputable def fractionA : ℝ := 7 / 15
noncomputable def fractionB : ℝ := 5 / 15
noncomputable def fractionC : ℝ := 3 / 15
noncomputable def validVotesA : ℝ := fractionA * validVotes
noncomputable def validVotesB : ℝ := fractionB * validVotes
noncomputable def validVotesC : ℝ := fractionC * validVotes

theorem election_valid_vote_counts :
  validVotesA = 315000 ∧ validVotesB = 225000 ∧ validVotesC = 135000 := by
  sorry

end NUMINAMATH_GPT_election_valid_vote_counts_l2317_231760


namespace NUMINAMATH_GPT_min_segments_required_l2317_231790

noncomputable def min_segments (n : ℕ) : ℕ := (3 * n - 2 + 1) / 2

theorem min_segments_required (n : ℕ) (h : ∀ (A B : ℕ) (hA : A < n) (hB : B < n) (hAB : A ≠ B), 
  ∃ (C : ℕ), C < n ∧ (C ≠ A) ∧ (C ≠ B)) : 
  min_segments n = ⌈ (3 * n - 2 : ℝ) / 2 ⌉ := 
sorry

end NUMINAMATH_GPT_min_segments_required_l2317_231790


namespace NUMINAMATH_GPT_pq_difference_l2317_231752

theorem pq_difference (p q : ℚ) (h₁ : 3 / p = 8) (h₂ : 3 / q = 18) : p - q = 5 / 24 := by
  sorry

end NUMINAMATH_GPT_pq_difference_l2317_231752


namespace NUMINAMATH_GPT_max_f_on_interval_l2317_231742

noncomputable def f (x : ℝ) : ℝ := Real.cos x ^ 2 + Real.sin x

theorem max_f_on_interval : 
  ∃ x ∈ Set.Icc (2 * Real.pi / 5) (3 * Real.pi / 4), f x = (1 + Real.sqrt 2) / 2 :=
by
  sorry

end NUMINAMATH_GPT_max_f_on_interval_l2317_231742


namespace NUMINAMATH_GPT_evaluate_expression_l2317_231789

noncomputable def log_4_8 : ℝ := Real.log 8 / Real.log 4
noncomputable def log_8_16 : ℝ := Real.log 16 / Real.log 8

theorem evaluate_expression : Real.sqrt (log_4_8 * log_8_16) = Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l2317_231789


namespace NUMINAMATH_GPT_greatest_divisor_450_90_l2317_231795

open Nat

-- Define a condition for the set of divisors of given numbers which are less than a certain number.
def is_divisor (a : ℕ) (b : ℕ) : Prop := b % a = 0

def is_greatest_divisor (d : ℕ) (n : ℕ) (m : ℕ) (k : ℕ) : Prop :=
  is_divisor m d ∧ d < k ∧ ∀ (x : ℕ), x < k → is_divisor m x → x ≤ d

-- Define the proof problem.
theorem greatest_divisor_450_90 : is_greatest_divisor 18 450 90 30 := 
by
  sorry

end NUMINAMATH_GPT_greatest_divisor_450_90_l2317_231795


namespace NUMINAMATH_GPT_problem_statement_l2317_231747

noncomputable def a : ℝ := -0.5
noncomputable def b : ℝ := (1 + Real.sqrt 3) / 2

theorem problem_statement
  (h1 : a^2 = 9 / 36)
  (h2 : b^2 = (1 + Real.sqrt 3)^2 / 8)
  (h3 : a < 0)
  (h4 : b > 0) :
  ∃ (x y z : ℤ), (a - b)^2 = x * Real.sqrt y / z ∧ (x + y + z = 6) :=
sorry

end NUMINAMATH_GPT_problem_statement_l2317_231747


namespace NUMINAMATH_GPT_b_geometric_l2317_231707

def a (n : ℕ) : ℝ := sorry
def b (n : ℕ) : ℝ := sorry

axiom a1 : a 1 = 1
axiom a_n_recurrence (n : ℕ) : a n + a (n + 1) = 1 / (3^n)
axiom b_def (n : ℕ) : b n = 3^(n - 1) * a n - 1/4

theorem b_geometric (n : ℕ) : b (n + 1) = -3 * b n := sorry

end NUMINAMATH_GPT_b_geometric_l2317_231707


namespace NUMINAMATH_GPT_cos_triple_angle_l2317_231763

theorem cos_triple_angle (θ : ℝ) (h : Real.cos θ = 3 / 5) : Real.cos (3 * θ) = -117 / 125 := 
by
  sorry

end NUMINAMATH_GPT_cos_triple_angle_l2317_231763


namespace NUMINAMATH_GPT_problem_a2_sub_b2_problem_a_mul_b_l2317_231727

theorem problem_a2_sub_b2 {a b : ℝ} (h1 : a + b = 8) (h2 : a - b = 4) : a^2 - b^2 = 32 :=
sorry

theorem problem_a_mul_b {a b : ℝ} (h1 : a + b = 8) (h2 : a - b = 4) : a * b = 12 :=
sorry

end NUMINAMATH_GPT_problem_a2_sub_b2_problem_a_mul_b_l2317_231727


namespace NUMINAMATH_GPT_sector_area_l2317_231709

theorem sector_area (arc_length radius : ℝ) (h1 : arc_length = 2) (h2 : radius = 2) : 
  (1/2) * arc_length * radius = 2 :=
by
  -- sorry placeholder for proof
  sorry

end NUMINAMATH_GPT_sector_area_l2317_231709


namespace NUMINAMATH_GPT_marker_cost_is_13_l2317_231718

theorem marker_cost_is_13 :
  ∃ s m c : ℕ, (s > 20) ∧ (m ≥ 4) ∧ (c > m) ∧ (s * c * m = 3185) ∧ (c = 13) :=
by
  sorry

end NUMINAMATH_GPT_marker_cost_is_13_l2317_231718


namespace NUMINAMATH_GPT_minimize_material_use_l2317_231768

theorem minimize_material_use 
  (x y : ℝ) 
  (hx : 0 < x) 
  (hy : 0 < y)
  (total_area : x * y + (x^2 / 4) = 8) :
  (abs (x - 2.343) ≤ 0.001) ∧ (abs (y - 2.828) ≤ 0.001) :=
sorry

end NUMINAMATH_GPT_minimize_material_use_l2317_231768


namespace NUMINAMATH_GPT_min_value_hyperbola_l2317_231704

open Real 

theorem min_value_hyperbola :
  ∀ (x y : ℝ), (x^2 / 4 - y^2 = 1) → (3 * x^2 - 2 * y ≥ 143 / 12) ∧ 
                                          (∃ (y' : ℝ), y = y' ∧  3 * (2 + 2*y'^2)^2 - 2 * y' = 143 / 12) := 
by
  sorry

end NUMINAMATH_GPT_min_value_hyperbola_l2317_231704


namespace NUMINAMATH_GPT_interest_rate_calculation_l2317_231729

theorem interest_rate_calculation
  (SI : ℕ) (P : ℕ) (T : ℕ) (R : ℕ)
  (h1 : SI = 2100) (h2 : P = 875) (h3 : T = 20) :
  (SI * 100 = P * R * T) → R = 12 :=
by
  sorry

end NUMINAMATH_GPT_interest_rate_calculation_l2317_231729


namespace NUMINAMATH_GPT_Q3_x_coords_sum_eq_Q1_x_coords_sum_l2317_231799

-- Define a 40-gon and its x-coordinates sum
def Q1_x_coords_sum : ℝ := 120

-- Statement to prove
theorem Q3_x_coords_sum_eq_Q1_x_coords_sum (Q1_x_coords_sum: ℝ) (h: Q1_x_coords_sum = 120) : 
  (Q3_x_coords_sum: ℝ) = Q1_x_coords_sum :=
sorry

end NUMINAMATH_GPT_Q3_x_coords_sum_eq_Q1_x_coords_sum_l2317_231799


namespace NUMINAMATH_GPT_find_common_ratio_l2317_231751

def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q

variable {a : ℕ → ℝ} {q : ℝ}

theorem find_common_ratio
  (h1 : is_geometric_sequence a q)
  (h2 : a 2 + a 4 = 20)
  (h3 : a 3 + a 5 = 40) : q = 2 :=
by
  sorry

end NUMINAMATH_GPT_find_common_ratio_l2317_231751


namespace NUMINAMATH_GPT_radius_of_circle_l2317_231740

theorem radius_of_circle
  (r : ℝ)
  (h1 : ∀ x : ℝ, (x^2 + r = x) → (x^2 - x + r = 0) → ((-1)^2 - 4 * 1 * r = 0)) :
  r = 1/4 :=
sorry

end NUMINAMATH_GPT_radius_of_circle_l2317_231740


namespace NUMINAMATH_GPT_sum_between_52_and_53_l2317_231714

theorem sum_between_52_and_53 (x y : ℝ) (h1 : y = 4 * (⌊x⌋ : ℝ) + 2) (h2 : y = 5 * (⌊x - 3⌋ : ℝ) + 7) (h3 : ∀ n : ℤ, x ≠ n) :
  52 < x + y ∧ x + y < 53 := 
sorry

end NUMINAMATH_GPT_sum_between_52_and_53_l2317_231714


namespace NUMINAMATH_GPT_find_p_l2317_231719

variable (f w : ℂ) (p : ℂ)
variable (h1 : f = 4)
variable (h2 : w = 10 + 200 * Complex.I)
variable (h3 : f * p - w = 20000)

theorem find_p : p = 5002.5 + 50 * Complex.I := by
  sorry

end NUMINAMATH_GPT_find_p_l2317_231719


namespace NUMINAMATH_GPT_lines_are_parallel_and_not_coincident_l2317_231739

theorem lines_are_parallel_and_not_coincident (a : ℝ) :
  (a * (a - 1) - 3 * 2 = 0) ∧ (3 * (a - 7) - a * 3 * a ≠ 0) ↔ a = 3 :=
by
  sorry

end NUMINAMATH_GPT_lines_are_parallel_and_not_coincident_l2317_231739


namespace NUMINAMATH_GPT_book_pages_l2317_231761

-- Define the conditions
def pages_per_night : ℕ := 12
def nights : ℕ := 10

-- Define the total pages in the book
def total_pages : ℕ := pages_per_night * nights

-- Prove that the total number of pages is 120
theorem book_pages : total_pages = 120 := by
  sorry

end NUMINAMATH_GPT_book_pages_l2317_231761


namespace NUMINAMATH_GPT_math_problem_l2317_231788

noncomputable def condition1 (a b : ℤ) : Prop :=
  |2 + a| + |b - 3| = 0

noncomputable def condition2 (c d : ℝ) : Prop :=
  1 / c = -d

noncomputable def condition3 (e : ℤ) : Prop :=
  e = -5

theorem math_problem (a b e : ℤ) (c d : ℝ) 
  (h1 : condition1 a b) 
  (h2 : condition2 c d) 
  (h3 : condition3 e) : 
  -a^b + 1 / c - e + d = 13 :=
by
  sorry

end NUMINAMATH_GPT_math_problem_l2317_231788


namespace NUMINAMATH_GPT_initial_bananas_each_child_l2317_231778

-- Define the variables and conditions.
def total_children : ℕ := 320
def absent_children : ℕ := 160
def present_children := total_children - absent_children
def extra_bananas : ℕ := 2

-- We are to prove the initial number of bananas each child was supposed to get.
theorem initial_bananas_each_child (B : ℕ) (x : ℕ) :
  B = total_children * x ∧ B = present_children * (x + extra_bananas) → x = 2 :=
by
  sorry

end NUMINAMATH_GPT_initial_bananas_each_child_l2317_231778


namespace NUMINAMATH_GPT_find_f_2011_l2317_231732

theorem find_f_2011 (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = f (-x)) 
  (h2 : ∀ x, f (x + 1) * f (x - 1) = 1) 
  (h3 : ∀ x, f x > 0) : 
  f 2011 = 1 := 
sorry

end NUMINAMATH_GPT_find_f_2011_l2317_231732


namespace NUMINAMATH_GPT_problem1_monotonic_and_extremum_problem2_range_of_k_problem3_inequality_l2317_231708

noncomputable def f (x : ℝ) (k : ℝ) := (Real.log x - k - 1) * x

-- Problem 1: Interval of monotonicity and extremum.
theorem problem1_monotonic_and_extremum (k : ℝ):
  (k ≤ 0 → ∀ x, 1 < x → f x k = (Real.log x - k - 1) * x) ∧
  (k > 0 → (∀ x, 1 < x ∧ x < Real.exp k → f x k = (Real.log x - k - 1) * x) ∧
           (∀ x, Real.exp k < x → f x k = (Real.log x - k - 1) * x) ∧
           f (Real.exp k) k = -Real.exp k) := sorry

-- Problem 2: Range of k.
theorem problem2_range_of_k (k : ℝ):
  (∀ x, Real.exp 1 ≤ x ∧ x ≤ Real.exp 2 → f x k < 4 * Real.log x) ↔
  k > 1 - (8 / Real.exp 2) := sorry

-- Problem 3: Inequality involving product of x1 and x2.
theorem problem3_inequality (x1 x2 : ℝ) (k : ℝ):
  x1 ≠ x2 ∧ f x1 k = f x2 k → x1 * x2 < Real.exp (2 * k) := sorry

end NUMINAMATH_GPT_problem1_monotonic_and_extremum_problem2_range_of_k_problem3_inequality_l2317_231708


namespace NUMINAMATH_GPT_union_of_subsets_l2317_231796

open Set

variable (A B : Set ℕ)

theorem union_of_subsets (m : ℕ) (hA : A = {1, 3}) (hB : B = {1, 2, m}) (hSubset : A ⊆ B) :
    A ∪ B = {1, 2, 3} :=
  sorry

end NUMINAMATH_GPT_union_of_subsets_l2317_231796


namespace NUMINAMATH_GPT_sequence_term_2010_l2317_231711

theorem sequence_term_2010 :
  ∀ (a : ℕ → ℤ), a 1 = 1 → a 2 = 2 → 
    (∀ n : ℕ, n ≥ 3 → a n = a (n - 1) - a (n - 2)) → 
    a 2010 = -1 :=
by
  sorry

end NUMINAMATH_GPT_sequence_term_2010_l2317_231711


namespace NUMINAMATH_GPT_intersection_of_M_and_N_l2317_231773

open Set

-- Conditions
def M : Set ℕ := {1, 2, 3}
def N : Set ℕ := {2, 3, 4}

-- Proof statement
theorem intersection_of_M_and_N : M ∩ N = {2, 3} :=
by {
  sorry
}

end NUMINAMATH_GPT_intersection_of_M_and_N_l2317_231773


namespace NUMINAMATH_GPT_units_digit_product_l2317_231757

theorem units_digit_product : 
  (2^2010 * 5^2011 * 11^2012) % 10 = 0 := 
by
  sorry

end NUMINAMATH_GPT_units_digit_product_l2317_231757


namespace NUMINAMATH_GPT_fixed_point_for_line_l2317_231780

theorem fixed_point_for_line (m : ℝ) : (m * (1 - 1) + (1 - 1) = 0) :=
by
  sorry

end NUMINAMATH_GPT_fixed_point_for_line_l2317_231780


namespace NUMINAMATH_GPT_ruffy_age_difference_l2317_231715

theorem ruffy_age_difference (R O : ℕ) (hR : R = 9) (hRO : R = (3/4 : ℚ) * O) :
  (R - 4) - (1 / 2 : ℚ) * (O - 4) = 1 :=
by 
  sorry

end NUMINAMATH_GPT_ruffy_age_difference_l2317_231715


namespace NUMINAMATH_GPT_dark_lord_squads_l2317_231775

def total_weight : ℕ := 1200
def orcs_per_squad : ℕ := 8
def capacity_per_orc : ℕ := 15
def squads_needed (w n c : ℕ) : ℕ := w / (n * c)

theorem dark_lord_squads :
  squads_needed total_weight orcs_per_squad capacity_per_orc = 10 :=
by sorry

end NUMINAMATH_GPT_dark_lord_squads_l2317_231775


namespace NUMINAMATH_GPT_markup_percentage_l2317_231750

variable (W R : ℝ) -- W for Wholesale Cost, R for Retail Cost

-- Conditions:
-- 1. The sweater is sold at a 40% discount.
-- 2. When sold at a 40% discount, the merchant nets a 30% profit on the wholesale cost.
def discount_price (R : ℝ) : ℝ := 0.6 * R
def profit_price (W : ℝ) : ℝ := 1.3 * W

-- Hypotheses
axiom wholesale_cost_is_positive : W > 0
axiom discount_condition : discount_price R = profit_price W

-- Question: Prove that the percentage markup from wholesale to retail price is 116.67%.
theorem markup_percentage (W R : ℝ) 
  (wholesale_cost_is_positive : W > 0)
  (discount_condition : discount_price R = profit_price W) :
  ((R - W) / W * 100) = 116.67 := by
  sorry

end NUMINAMATH_GPT_markup_percentage_l2317_231750


namespace NUMINAMATH_GPT_triangle_side_lengths_l2317_231730

theorem triangle_side_lengths (r : ℝ) (AC BC AB : ℝ) (y : ℝ) 
  (h1 : r = 3 * Real.sqrt 2)
  (h2 : AC = 5 * Real.sqrt y) 
  (h3 : BC = 13 * Real.sqrt y) 
  (h4 : AB = 10 * Real.sqrt y) : 
  r = 3 * Real.sqrt 2 → 
  (∃ (AC BC AB : ℝ), 
     AC = 5 * Real.sqrt (7) ∧ 
     BC = 13 * Real.sqrt (7) ∧ 
     AB = 10 * Real.sqrt (7)) :=
by
  sorry

end NUMINAMATH_GPT_triangle_side_lengths_l2317_231730


namespace NUMINAMATH_GPT_A_takes_200_seconds_l2317_231745

/-- 
  A can give B a start of 50 meters or 10 seconds in a kilometer race.
  How long does A take to complete the race?
-/
theorem A_takes_200_seconds (v_A : ℝ) (distance : ℝ) (start_meters : ℝ) (start_seconds : ℝ) :
  (start_meters = 50) ∧ (start_seconds = 10) ∧ (distance = 1000) ∧ 
  (v_A = start_meters / start_seconds) → distance / v_A = 200 :=
by
  sorry

end NUMINAMATH_GPT_A_takes_200_seconds_l2317_231745


namespace NUMINAMATH_GPT_total_amount_Rs20_l2317_231720

theorem total_amount_Rs20 (x y z : ℕ) 
(h1 : x + y + z = 130) 
(h2 : 95 * x + 45 * y + 20 * z = 7000) : 
∃ z : ℕ, (20 * z) = (7000 - 95 * x - 45 * y) / 20 := sorry

end NUMINAMATH_GPT_total_amount_Rs20_l2317_231720


namespace NUMINAMATH_GPT_value_of_g_neg3_l2317_231777

def g (x : ℝ) : ℝ := x^3 - 2 * x

theorem value_of_g_neg3 : g (-3) = -21 := by
  sorry

end NUMINAMATH_GPT_value_of_g_neg3_l2317_231777


namespace NUMINAMATH_GPT_smallest_multiple_of_84_with_6_and_7_l2317_231734

variable (N : Nat)

def is_multiple_of_84 (N : Nat) : Prop :=
  N % 84 = 0

def consists_of_6_and_7 (N : Nat) : Prop :=
  ∀ d ∈ N.digits 10, d = 6 ∨ d = 7

theorem smallest_multiple_of_84_with_6_and_7 :
  ∃ N, is_multiple_of_84 N ∧ consists_of_6_and_7 N ∧ ∀ M, is_multiple_of_84 M ∧ consists_of_6_and_7 M → N ≤ M := 
sorry

end NUMINAMATH_GPT_smallest_multiple_of_84_with_6_and_7_l2317_231734


namespace NUMINAMATH_GPT_remaining_card_number_l2317_231772

theorem remaining_card_number (A B C D E F G H : ℕ) (cards : Finset ℕ) 
  (hA : A + B = 10) 
  (hB : C - D = 1) 
  (hC : E * F = 24) 
  (hD : G / H = 3) 
  (hCards : cards = {1, 2, 3, 4, 5, 6, 7, 8, 9})
  (hDistinct : A ∉ cards ∧ B ∉ cards ∧ C ∉ cards ∧ D ∉ cards ∧ E ∉ cards ∧ F ∉ cards ∧ G ∉ cards ∧ H ∉ cards) :
  7 ∈ cards := 
by
  sorry

end NUMINAMATH_GPT_remaining_card_number_l2317_231772


namespace NUMINAMATH_GPT_stream_current_speed_l2317_231766

theorem stream_current_speed (r w : ℝ) (h1 : 18 / (r + w) + 4 = 18 / (r - w))
  (h2 : 18 / (1.5 * r + w) + 2 = 18 / (1.5 * r - w)) : w = 2.5 :=
by
  -- Translate the equations from the problem conditions directly.
  sorry

end NUMINAMATH_GPT_stream_current_speed_l2317_231766


namespace NUMINAMATH_GPT_find_p_q_r_l2317_231724

theorem find_p_q_r : 
  ∃ (p q r : ℕ), 
  p > 0 ∧ q > 0 ∧ r > 0 ∧ 
  4 * (Real.sqrt (Real.sqrt 7) - Real.sqrt (Real.sqrt 6)) 
  = Real.sqrt (Real.sqrt p) + Real.sqrt (Real.sqrt q) - Real.sqrt (Real.sqrt r) 
  ∧ p + q + r = 99 := 
sorry

end NUMINAMATH_GPT_find_p_q_r_l2317_231724


namespace NUMINAMATH_GPT_problem_inequality_l2317_231723

variables {a b c x1 x2 x3 x4 x5 : ℝ} 

theorem problem_inequality
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_pos_x1: 0 < x1) (h_pos_x2: 0 < x2) (h_pos_x3: 0 < x3) (h_pos_x4: 0 < x4) (h_pos_x5: 0 < x5)
  (h_sum_abc : a + b + c = 1) (h_prod_x : x1 * x2 * x3 * x4 * x5 = 1) :
  (a * x1^2 + b * x1 + c) * (a * x2^2 + b * x2 + c) * (a * x3^2 + b * x3 + c) * 
  (a * x4^2 + b * x4 + c) * (a * x5^2 + b * x5 + c) ≥ 1 :=
sorry

end NUMINAMATH_GPT_problem_inequality_l2317_231723
