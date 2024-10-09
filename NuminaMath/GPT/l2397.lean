import Mathlib

namespace value_of_x_plus_y_l2397_239712

noncomputable def x : ℝ := 1 / 2
noncomputable def y : ℝ := 3

theorem value_of_x_plus_y
  (hx : 1 / x = 2)
  (hy : 1 / x + 3 / y = 3) :
  x + y = 7 / 2 :=
  sorry

end value_of_x_plus_y_l2397_239712


namespace worker_times_l2397_239737

-- Define the problem
theorem worker_times (x y : ℝ) (h1 : (1 / x + 1 / y = 1 / 8)) (h2 : x = y - 12) :
    x = 24 ∧ y = 12 :=
by
  sorry

end worker_times_l2397_239737


namespace perimeter_of_square_l2397_239716

theorem perimeter_of_square (s : ℝ) (area : s^2 = 468) : 4 * s = 24 * Real.sqrt 13 := 
by
  sorry

end perimeter_of_square_l2397_239716


namespace range_of_x_l2397_239794

variable (x : ℝ)

theorem range_of_x (h1 : 2 - x > 0) (h2 : x - 1 ≥ 0) : 1 ≤ x ∧ x < 2 := by
  sorry

end range_of_x_l2397_239794


namespace trig_eq_solution_l2397_239729

open Real

theorem trig_eq_solution (x : ℝ) :
    (∃ k : ℤ, x = -arccos ((sqrt 13 - 1) / 4) + 2 * k * π) ∨ 
    (∃ k : ℤ, x = -arccos ((1 - sqrt 13) / 4) + 2 * k * π) ↔ 
    (cos 5 * x - cos 7 * x) / (sin 4 * x + sin 2 * x) = 2 * abs (sin 2 * x) := by
  sorry

end trig_eq_solution_l2397_239729


namespace odd_func_value_l2397_239768

noncomputable def isOddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

noncomputable def f : ℝ → ℝ
| x => if x > 0 then 2 * x - 3 else 0 -- f(x) is initially set to 0 when x ≤ 0, since we will not use this part directly.

theorem odd_func_value (x : ℝ) (h : x < 0) (hf : isOddFunction f) (hfx : ∀ x > 0, f x = 2 * x - 3) :
  f x = 2 * x + 3 :=
by
  sorry

end odd_func_value_l2397_239768


namespace sqrt_of_six_l2397_239742

theorem sqrt_of_six : Real.sqrt 6 = Real.sqrt 6 := by
  sorry

end sqrt_of_six_l2397_239742


namespace simplify_absolute_value_l2397_239763

theorem simplify_absolute_value : abs (-(5^2) + 6 * 2) = 13 := by
  sorry

end simplify_absolute_value_l2397_239763


namespace star_is_addition_l2397_239726

variable {α : Type} [AddCommGroup α]

-- Define the binary operation star
variable (star : α → α → α)

-- Define the condition given in the problem
axiom star_condition : ∀ (a b c : α), star (star a b) c = a + b + c

-- Prove that star is the same as usual addition
theorem star_is_addition : ∀ (a b : α), star a b = a + b :=
  sorry

end star_is_addition_l2397_239726


namespace subcommittee_formation_l2397_239700

/-- A Senate committee consists of 10 Republicans and 7 Democrats.
    The number of ways to form a subcommittee with 4 Republicans and 3 Democrats is 7350. -/
theorem subcommittee_formation :
  (Nat.choose 10 4) * (Nat.choose 7 3) = 7350 :=
by
  sorry

end subcommittee_formation_l2397_239700


namespace count_balanced_integers_l2397_239797

def is_balanced (n : ℕ) : Prop :=
  let d1 := n / 100
  let d2 := (n % 100) / 10
  let d3 := n % 10
  (d1 + d2 + d3) = d1 + (d2 + d3) ∧ (100 ≤ n) ∧ (n ≤ 999)

theorem count_balanced_integers : ∃ c, c = 330 ∧ ∀ n, 100 ≤ n ∧ n ≤ 999 → is_balanced n ↔ c = 330 :=
sorry

end count_balanced_integers_l2397_239797


namespace eight_step_paths_board_l2397_239738

theorem eight_step_paths_board (P Q : ℕ) (hP : P = 0) (hQ : Q = 7) : 
  ∃ (paths : ℕ), paths = 70 :=
by
  sorry

end eight_step_paths_board_l2397_239738


namespace marias_profit_l2397_239704

theorem marias_profit 
  (initial_loaves : ℕ)
  (morning_price : ℝ)
  (afternoon_discount : ℝ)
  (late_afternoon_price : ℝ)
  (cost_per_loaf : ℝ)
  (loaves_sold_morning : ℕ)
  (loaves_sold_afternoon : ℕ)
  (loaves_remaining : ℕ)
  (revenue_morning : ℝ)
  (revenue_afternoon : ℝ)
  (revenue_late_afternoon : ℝ)
  (total_revenue : ℝ)
  (total_cost : ℝ)
  (profit : ℝ) :
  initial_loaves = 60 →
  morning_price = 3.0 →
  afternoon_discount = 0.75 →
  late_afternoon_price = 1.50 →
  cost_per_loaf = 1.0 →
  loaves_sold_morning = initial_loaves / 3 →
  loaves_sold_afternoon = (initial_loaves - loaves_sold_morning) / 2 →
  loaves_remaining = initial_loaves - loaves_sold_morning - loaves_sold_afternoon →
  revenue_morning = loaves_sold_morning * morning_price →
  revenue_afternoon = loaves_sold_afternoon * (afternoon_discount * morning_price) →
  revenue_late_afternoon = loaves_remaining * late_afternoon_price →
  total_revenue = revenue_morning + revenue_afternoon + revenue_late_afternoon →
  total_cost = initial_loaves * cost_per_loaf →
  profit = total_revenue - total_cost →
  profit = 75 := sorry

end marias_profit_l2397_239704


namespace evaluate_expression_l2397_239725

theorem evaluate_expression : -25 - 7 * (4 + 2) = -67 := by
  sorry

end evaluate_expression_l2397_239725


namespace cookie_count_per_box_l2397_239799

theorem cookie_count_per_box (A B C T: ℝ) (H1: A = 2) (H2: B = 0.75) (H3: C = 3) (H4: T = 276) :
  T / (A + B + C) = 48 :=
by
  sorry

end cookie_count_per_box_l2397_239799


namespace find_q_l2397_239759

noncomputable def q_value (m q : ℕ) : Prop := 
  ((1 ^ m) / (5 ^ m)) * ((1 ^ 16) / (4 ^ 16)) = 1 / (q * 10 ^ 31)

theorem find_q (m : ℕ) (q : ℕ) (h1 : m = 31) (h2 : q_value m q) : q = 2 :=
by
  sorry

end find_q_l2397_239759


namespace multiplier_of_product_l2397_239778

variable {a b : ℝ}

theorem multiplier_of_product (h1 : a ≠ 0) (h2 : b ≠ 0)
  (h3 : a + b = k * (a * b))
  (h4 : (1 / a) + (1 / b) = 6) : k = 6 := by
  sorry

end multiplier_of_product_l2397_239778


namespace probability_equal_white_black_probability_white_ge_black_l2397_239727

/-- Part (a) -/
theorem probability_equal_white_black (n m : ℕ) (h : n ≥ m) :
  (∃ p, p = (2 * m) / (n + m)) := 
  sorry

/-- Part (b) -/
theorem probability_white_ge_black (n m : ℕ) (h : n ≥ m) :
  (∃ p, p = (n - m + 1) / (n + 1)) := 
  sorry

end probability_equal_white_black_probability_white_ge_black_l2397_239727


namespace gardener_cabbages_l2397_239752

theorem gardener_cabbages (area_this_year : ℕ) (side_length_this_year : ℕ) (side_length_last_year : ℕ) (area_last_year : ℕ) (additional_cabbages : ℕ) :
  area_this_year = 9801 →
  side_length_this_year = 99 →
  side_length_last_year = side_length_this_year - 1 →
  area_last_year = side_length_last_year * side_length_last_year →
  additional_cabbages = area_this_year - area_last_year →
  additional_cabbages = 197 :=
by
  sorry

end gardener_cabbages_l2397_239752


namespace mass_percentage_Cl_correct_l2397_239762

-- Define the given condition
def mass_percentage_of_Cl := 66.04

-- Statement to prove
theorem mass_percentage_Cl_correct : mass_percentage_of_Cl = 66.04 :=
by
  -- This is where the proof would go, but we use sorry as placeholder.
  sorry

end mass_percentage_Cl_correct_l2397_239762


namespace shaded_area_of_rotated_semicircle_l2397_239760

-- Definitions and conditions from the problem
def radius (R : ℝ) : Prop := R > 0
def central_angle (α : ℝ) : Prop := α = 30 * (Real.pi / 180)

-- Lean theorem statement for the proof problem
theorem shaded_area_of_rotated_semicircle (R : ℝ) (hR : radius R) (hα : central_angle 30) : 
  ∃ (area : ℝ), area = (Real.pi * R^2) / 3 :=
by
  -- using proofs of radius and angle conditions
  sorry

end shaded_area_of_rotated_semicircle_l2397_239760


namespace imaginary_part_div_z1_z2_l2397_239777

noncomputable def z1 := 1 - 3 * Complex.I
noncomputable def z2 := 3 + Complex.I

theorem imaginary_part_div_z1_z2 : 
  Complex.im ((1 + 3 * Complex.I) / (3 + Complex.I)) = 4 / 5 := 
by 
  sorry

end imaginary_part_div_z1_z2_l2397_239777


namespace remaining_black_area_after_five_changes_l2397_239775

-- Define a function that represents the change process
noncomputable def remaining_black_area (iterations : ℕ) : ℚ :=
  (3 / 4) ^ iterations

-- Define the original problem statement as a theorem in Lean
theorem remaining_black_area_after_five_changes :
  remaining_black_area 5 = 243 / 1024 :=
by
  sorry

end remaining_black_area_after_five_changes_l2397_239775


namespace math_problem_A_B_M_l2397_239782

theorem math_problem_A_B_M :
  ∃ M : Set ℝ,
    M = {m | ∃ A B : Set ℝ,
      A = {x | x^2 - 5 * x + 6 = 0} ∧
      B = {x | m * x - 1 = 0} ∧
      A ∩ B = B ∧
      M = {0, (1:ℝ)/2, (1:ℝ)/3}} ∧
    ∃ subsets : Set (Set ℝ),
      subsets = {∅, {0}, {(1:ℝ)/2}, {(1:ℝ)/3}, {0, (1:ℝ)/2}, {(1:ℝ)/2, (1:ℝ)/3}, {0, (1:ℝ)/3}, {0, (1:ℝ)/2, (1:ℝ)/3}} :=
by
  sorry

end math_problem_A_B_M_l2397_239782


namespace mean_median_difference_is_minus_4_l2397_239793

-- Defining the percentages of students scoring specific points
def perc_60 : ℝ := 0.20
def perc_75 : ℝ := 0.55
def perc_95 : ℝ := 0.10
def perc_110 : ℝ := 1 - (perc_60 + perc_75 + perc_95) -- 0.15

-- Defining the scores
def score_60 : ℝ := 60
def score_75 : ℝ := 75
def score_95 : ℝ := 95
def score_110 : ℝ := 110

-- Calculating the mean score
def mean_score : ℝ := (perc_60 * score_60) + (perc_75 * score_75) + (perc_95 * score_95) + (perc_110 * score_110)

-- Given the median score
def median_score : ℝ := score_75

-- Defining the expected difference
def expected_difference : ℝ := mean_score - median_score

theorem mean_median_difference_is_minus_4 :
  expected_difference = -4 := by sorry

end mean_median_difference_is_minus_4_l2397_239793


namespace solve_for_x_l2397_239787

theorem solve_for_x (x : ℝ) : 2 * x + 3 * x + 4 * x = 12 + 9 + 6 → x = 3 :=
by
  sorry

end solve_for_x_l2397_239787


namespace range_of_a_l2397_239767

theorem range_of_a (a : ℝ) : (∀ x : ℝ, (|x + a| < 3 ↔ 2 < x ∧ x < 3)) ↔ -5 ≤ a ∧ a ≤ 0 := by
  sorry

end range_of_a_l2397_239767


namespace quadratic_roots_identity_l2397_239714

noncomputable def sum_of_roots (a b : ℝ) : Prop := a + b = -10
noncomputable def product_of_roots (a b : ℝ) : Prop := a * b = 5

theorem quadratic_roots_identity (a b : ℝ)
  (h₁ : sum_of_roots a b)
  (h₂ : product_of_roots a b) :
  (a / b + b / a) = 18 :=
by sorry

end quadratic_roots_identity_l2397_239714


namespace incorrect_average_initially_calculated_l2397_239734

theorem incorrect_average_initially_calculated :
  ∀ (S' S : ℕ) (n : ℕ) (incorrect_correct_difference : ℕ),
  n = 10 →
  incorrect_correct_difference = 30 →
  S = 200 →
  S' = S - incorrect_correct_difference →
  (S' / n) = 17 :=
by
  intros S' S n incorrect_correct_difference h_n h_diff h_S h_S' 
  sorry

end incorrect_average_initially_calculated_l2397_239734


namespace find_number_l2397_239722

-- Definitions from the conditions
def condition1 (x : ℝ) := 16 * x = 3408
def condition2 (x : ℝ) := 1.6 * x = 340.8

-- The statement to prove
theorem find_number (x : ℝ) (h1 : condition1 x) (h2 : condition2 x) : x = 213 :=
by
  sorry

end find_number_l2397_239722


namespace regions_formed_l2397_239711

theorem regions_formed (radii : ℕ) (concentric_circles : ℕ) (total_regions : ℕ) 
  (h_radii : radii = 16) (h_concentric_circles : concentric_circles = 10) 
  (h_total_regions : total_regions = radii * (concentric_circles + 1)) : 
  total_regions = 176 := 
by
  rw [h_radii, h_concentric_circles] at h_total_regions
  exact h_total_regions

end regions_formed_l2397_239711


namespace doughnut_problem_l2397_239758

theorem doughnut_problem :
  ∀ (total_doughnuts first_two_box_doughnuts boxes : ℕ),
  total_doughnuts = 72 →
  first_two_box_doughnuts = 12 →
  boxes = 4 →
  (total_doughnuts - 2 * first_two_box_doughnuts) / boxes = 12 :=
by
  intros total_doughnuts first_two_box_doughnuts boxes ht12 hb12 b4
  sorry

end doughnut_problem_l2397_239758


namespace correct_email_sequence_l2397_239786

theorem correct_email_sequence :
  let a := "Open the mailbox"
  let b := "Enter the recipient's address"
  let c := "Enter the subject"
  let d := "Enter the content of the email"
  let e := "Click 'Compose'"
  let f := "Click 'Send'"
  (a, e, b, c, d, f) = ("Open the mailbox", "Click 'Compose'", "Enter the recipient's address", "Enter the subject", "Enter the content of the email", "Click 'Send'") := 
sorry

end correct_email_sequence_l2397_239786


namespace seats_taken_l2397_239766

variable (num_rows : ℕ) (chairs_per_row : ℕ) (unoccupied_chairs : ℕ)

theorem seats_taken (h1 : num_rows = 40) (h2 : chairs_per_row = 20) (h3 : unoccupied_chairs = 10) :
  num_rows * chairs_per_row - unoccupied_chairs = 790 :=
sorry

end seats_taken_l2397_239766


namespace ages_correct_l2397_239715

def ages : List ℕ := [5, 8, 13, 15]
def Tanya : ℕ := 13
def Yura : ℕ := 8
def Sveta : ℕ := 5
def Lena : ℕ := 15

theorem ages_correct (h1 : Tanya ∈ ages) 
                     (h2: Yura ∈ ages)
                     (h3: Sveta ∈ ages)
                     (h4: Lena ∈ ages)
                     (h5: Tanya ≠ Yura)
                     (h6: Tanya ≠ Sveta)
                     (h7: Tanya ≠ Lena)
                     (h8: Yura ≠ Sveta)
                     (h9: Yura ≠ Lena)
                     (h10: Sveta ≠ Lena)
                     (h11: Sveta = 5)
                     (h12: Tanya > Yura)
                     (h13: (Tanya + Sveta) % 3 = 0) :
                     Tanya = 13 ∧ Yura = 8 ∧ Sveta = 5 ∧ Lena = 15 := by
  sorry

end ages_correct_l2397_239715


namespace mary_needs_more_apples_l2397_239753

theorem mary_needs_more_apples :
  let pies := 15
  let apples_per_pie := 10
  let harvested_apples := 40
  let total_apples_needed := pies * apples_per_pie
  let more_apples_needed := total_apples_needed - harvested_apples
  more_apples_needed = 110 :=
by
  sorry

end mary_needs_more_apples_l2397_239753


namespace range_of_x_l2397_239789

theorem range_of_x (a : ℝ) (x : ℝ) (h0 : 0 ≤ a) (h1 : a ≤ 2) :
  a * x^2 + (a + 1) * x + 1 - (3 / 2) * a < 0 → -2 < x ∧ x < -1 :=
by
  sorry

end range_of_x_l2397_239789


namespace ratio_increase_productivity_l2397_239774

theorem ratio_increase_productivity (initial current: ℕ) 
  (h_initial: initial = 10) 
  (h_current: current = 25) : 
  (current - initial) / initial = 3 / 2 := 
by
  sorry

end ratio_increase_productivity_l2397_239774


namespace four_digit_property_l2397_239764

-- Define the problem conditions and statement
theorem four_digit_property (x y : ℕ) (hx : 10 ≤ x ∧ x < 100) (hy : 0 ≤ y ∧ y < 100) :
  (100 * x + y = (x + y) ^ 2) ↔ (100 * x + y = 3025 ∨ 100 * x + y = 2025 ∨ 100 * x + y = 9801) := by
sorry

end four_digit_property_l2397_239764


namespace smallest_number_is_16_l2397_239736

theorem smallest_number_is_16 :
  ∃ (a b c : ℕ), a > 0 ∧ b > 0 ∧ c > 0 ∧ (a + b + c) / 3 = 24 ∧ 
  (b = 25) ∧ (c = b + 6) ∧ min a (min b c) = 16 :=
by
  sorry

end smallest_number_is_16_l2397_239736


namespace volume_triangular_pyramid_correctness_l2397_239718

noncomputable def volume_of_regular_triangular_pyramid 
  (a α l : ℝ) : ℝ :=
  (a ^ 3 * Real.sqrt 3 / 8) * Real.tan α

theorem volume_triangular_pyramid_correctness (a α l : ℝ) : volume_of_regular_triangular_pyramid a α l =
  (a ^ 3 * Real.sqrt 3 / 8) * Real.tan α := 
sorry

end volume_triangular_pyramid_correctness_l2397_239718


namespace hall_100_guests_67_friends_find_clique_l2397_239784

theorem hall_100_guests_67_friends_find_clique :
  ∀ (P : Fin 100 → Fin 100 → Prop) (n : Fin 100),
    (∀ i : Fin 100, ∃ S : Finset (Fin 100), (S.card ≥ 67) ∧ (∀ j ∈ S, P i j)) →
    (∃ (A B C D : Fin 100), P A B ∧ P A C ∧ P A D ∧ P B C ∧ P B D ∧ P C D) :=
by
  sorry

end hall_100_guests_67_friends_find_clique_l2397_239784


namespace empty_set_implies_a_range_l2397_239795

theorem empty_set_implies_a_range (a : ℝ) :
  (∀ x : ℝ, ¬(a * x^2 - 2 * a * x + 1 < 0)) ↔ (0 ≤ a ∧ a < 1) := 
sorry

end empty_set_implies_a_range_l2397_239795


namespace value_of_f2008_plus_f2009_l2397_239751

variable {f : ℤ → ℤ}

-- Conditions
axiom h1 : ∀ x : ℤ, f (-(x) + 2) = -f (x + 2)
axiom h2 : ∀ x : ℤ, f (6 - x) = f x
axiom h3 : f 3 = 2

-- The theorem to prove
theorem value_of_f2008_plus_f2009 : f 2008 + f 2009 = -2 :=
  sorry

end value_of_f2008_plus_f2009_l2397_239751


namespace power_mod_l2397_239720

theorem power_mod (x n m : ℕ) : (x^n) % m = x % m := by 
  sorry

example : 5^2023 % 150 = 5 % 150 :=
by exact power_mod 5 2023 150

end power_mod_l2397_239720


namespace bread_count_at_end_of_day_l2397_239730

def initial_loaves : ℕ := 2355
def sold_loaves : ℕ := 629
def delivered_loaves : ℕ := 489

theorem bread_count_at_end_of_day : 
  initial_loaves - sold_loaves + delivered_loaves = 2215 := by
  sorry

end bread_count_at_end_of_day_l2397_239730


namespace remainder_29_times_171997_pow_2000_mod_7_l2397_239798

theorem remainder_29_times_171997_pow_2000_mod_7 :
  (29 * 171997^2000) % 7 = 4 :=
by
  sorry

end remainder_29_times_171997_pow_2000_mod_7_l2397_239798


namespace right_triangle_area_l2397_239739

theorem right_triangle_area (hypotenuse : ℝ)
  (angle_deg : ℝ)
  (h_hyp : hypotenuse = 10 * Real.sqrt 2)
  (h_angle : angle_deg = 45) : 
  (1 / 2) * (hypotenuse / Real.sqrt 2)^2 = 50 := 
by 
  sorry

end right_triangle_area_l2397_239739


namespace cyclic_inequality_l2397_239721

theorem cyclic_inequality (a b c : ℝ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0) :
  (a^3 * b^3 * (a * b - a * c - b * c + c^2) +
   b^3 * c^3 * (b * c - b * a - c * a + a^2) +
   c^3 * a^3 * (c * a - c * b - a * b + b^2)) ≥ 0 :=
sorry

end cyclic_inequality_l2397_239721


namespace melanie_trout_l2397_239770

theorem melanie_trout (M : ℕ) (h1 : 2 * M = 16) : M = 8 :=
by
  sorry

end melanie_trout_l2397_239770


namespace worker_time_proof_l2397_239772

theorem worker_time_proof (x : ℝ) (h1 : x > 2) (h2 : (100 / (x - 2) - 100 / x) = 5 / 2) : 
  (x = 10) ∧ (x - 2 = 8) :=
by
  sorry

end worker_time_proof_l2397_239772


namespace expressway_lengths_l2397_239781

theorem expressway_lengths (x y : ℕ) (h1 : x + y = 519) (h2 : x = 2 * y - 45) : x = 331 ∧ y = 188 :=
by
  -- Proof omitted
  sorry

end expressway_lengths_l2397_239781


namespace true_propositions_l2397_239705

-- Definitions for the propositions
def proposition1 (a b : ℝ) : Prop := a > b → a^2 > b^2
def proposition2 (a b : ℝ) : Prop := a^2 > b^2 → |a| > |b|
def proposition3 (a b c : ℝ) : Prop := (a > b ↔ a + c > b + c)

-- Theorem to state the true propositions
theorem true_propositions (a b c : ℝ) :
  -- Proposition 3 is true
  (proposition3 a b c) →
  -- Assert that the serial number of the true propositions is 3
  {3} = { i | (i = 1 ∧ proposition1 a b) ∨ (i = 2 ∧ proposition2 a b) ∨ (i = 3 ∧ proposition3 a b c)} :=
by
  sorry

end true_propositions_l2397_239705


namespace parameterized_line_segment_problem_l2397_239788

theorem parameterized_line_segment_problem
  (p q r s : ℝ)
  (hq : q = 1)
  (hs : s = 2)
  (hpq : p + q = 6)
  (hrs : r + s = 9) :
  p^2 + q^2 + r^2 + s^2 = 79 := 
sorry

end parameterized_line_segment_problem_l2397_239788


namespace min_radius_circle_condition_l2397_239724

theorem min_radius_circle_condition (r : ℝ) (a b : ℝ) 
    (h_circle : (a - (r + 1))^2 + b^2 = r^2)
    (h_condition : b^2 ≥ 4 * a) :
    r ≥ 4 := 
sorry

end min_radius_circle_condition_l2397_239724


namespace polynomial_division_l2397_239741

-- Define the polynomials P and D
noncomputable def P : Polynomial ℤ := 5 * Polynomial.X ^ 4 - 3 * Polynomial.X ^ 3 + 7 * Polynomial.X ^ 2 - 9 * Polynomial.X + 12
noncomputable def D : Polynomial ℤ := Polynomial.X - 3
noncomputable def Q : Polynomial ℤ := 5 * Polynomial.X ^ 3 + 12 * Polynomial.X ^ 2 + 43 * Polynomial.X + 120
def R : ℤ := 372

-- State the theorem
theorem polynomial_division :
  P = D * Q + Polynomial.C R := 
sorry

end polynomial_division_l2397_239741


namespace vector_satisfy_condition_l2397_239769

structure Point where
  x : ℝ
  y : ℝ

structure Line where
  parametrize : ℝ → Point

def l : Line :=
  { parametrize := λ t => {x := 1 + 4 * t, y := 4 + 3 * t} }

def m : Line :=
  { parametrize := λ s => {x := -5 + 4 * s, y := 6 + 3 * s} }

def A (t : ℝ) : Point := l.parametrize t
def B (s : ℝ) : Point := m.parametrize s

-- The specific point for A and B are not used directly in the further proof statement.

def v : Point := { x := -6, y := 8 }

theorem vector_satisfy_condition :
  ∃ v1 v2 : ℝ, (v1 * -6) + (v2 * 8) = 2 ∧ (v1 = -6 ∧ v2 = 8) :=
sorry

end vector_satisfy_condition_l2397_239769


namespace sum_of_digits_next_l2397_239706

def sum_of_digits (n : ℕ) : ℕ :=
  (Nat.digits 10 n).sum

theorem sum_of_digits_next (n : ℕ) (h : sum_of_digits n = 1399) : 
  sum_of_digits (n + 1) = 1402 :=
sorry

end sum_of_digits_next_l2397_239706


namespace product_of_two_numbers_l2397_239773

theorem product_of_two_numbers (x y : ℕ) (h₁ : x + y = 16) (h₂ : x^2 + y^2 = 200) : x * y = 28 :=
by
  sorry

end product_of_two_numbers_l2397_239773


namespace remaining_pie_portion_l2397_239709

theorem remaining_pie_portion (Carlos_takes: ℝ) (fraction_Maria: ℝ) :
  Carlos_takes = 0.60 →
  fraction_Maria = 0.25 →
  (1 - Carlos_takes) * (1 - fraction_Maria) = 0.30 := by
  intros h1 h2
  rw [h1, h2]
  simp
  sorry

end remaining_pie_portion_l2397_239709


namespace spring_excursion_participants_l2397_239779

theorem spring_excursion_participants (water fruit neither both total : ℕ) 
  (h_water : water = 80) 
  (h_fruit : fruit = 70) 
  (h_neither : neither = 6) 
  (h_both : both = total / 2) 
  (h_total_eq : total = water + fruit - both + neither) : 
  total = 104 := 
  sorry

end spring_excursion_participants_l2397_239779


namespace amount_of_CaO_required_l2397_239733

theorem amount_of_CaO_required (n_H2O : ℝ) (n_CaOH2 : ℝ) (n_CaO : ℝ) 
  (h1 : n_H2O = 2) (h2 : n_CaOH2 = 2) :
  n_CaO = 2 :=
by
  sorry

end amount_of_CaO_required_l2397_239733


namespace sum_of_squares_ineq_l2397_239735

theorem sum_of_squares_ineq (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_sum_sq : a^2 + b^2 + c^2 = 3) :
  a + b + c ≥ a^2 * b^2 + b^2 * c^2 + c^2 * a^2 :=
sorry

end sum_of_squares_ineq_l2397_239735


namespace diagonals_in_decagon_l2397_239710

def number_of_diagonals (n : ℕ) : ℕ :=
  n * (n - 3) / 2

theorem diagonals_in_decagon : number_of_diagonals 10 = 35 := by
  sorry

end diagonals_in_decagon_l2397_239710


namespace original_six_digit_number_is_105262_l2397_239747

def is_valid_number (N : ℕ) : Prop :=
  ∃ A : ℕ, A < 100000 ∧ (N = 10 * A + 2) ∧ (200000 + A = 2 * N + 2)

theorem original_six_digit_number_is_105262 :
  ∃ N : ℕ, is_valid_number N ∧ N = 105262 :=
by
  sorry

end original_six_digit_number_is_105262_l2397_239747


namespace evaluate_expression_l2397_239780

theorem evaluate_expression : 60 + (105 / 15) + (25 * 16) - 250 + (324 / 9) ^ 2 = 1513 := by
  sorry

end evaluate_expression_l2397_239780


namespace knight_will_be_freed_l2397_239723

/-- Define a structure to hold the state of the piles -/
structure PileState where
  pile1_magical : ℕ
  pile1_non_magical : ℕ
  pile2_magical : ℕ
  pile2_non_magical : ℕ
deriving Repr

-- Function to move one coin from pile1 to pile2
def move_coin (state : PileState) : PileState :=
  if state.pile1_magical > 0 then
    { state with
      pile1_magical := state.pile1_magical - 1,
      pile2_magical := state.pile2_magical + 1 }
  else if state.pile1_non_magical > 0 then
    { state with
      pile1_non_magical := state.pile1_non_magical - 1,
      pile2_non_magical := state.pile2_non_magical + 1 }
  else
    state -- If no coins to move, the state remains unchanged

-- The initial state of the piles
def initial_state : PileState :=
  { pile1_magical := 0, pile1_non_magical := 49, pile2_magical := 50, pile2_non_magical := 1 }

-- Check if the knight can be freed (both piles have the same number of magical or non-magical coins)
def knight_free (state : PileState) : Prop :=
  state.pile1_magical = state.pile2_magical ∨ state.pile1_non_magical = state.pile2_non_magical

noncomputable def knight_can_be_freed_by_25th_day : Prop :=
  exists n : ℕ, n ≤ 25 ∧ knight_free (Nat.iterate move_coin n initial_state)

theorem knight_will_be_freed : knight_can_be_freed_by_25th_day :=
  sorry

end knight_will_be_freed_l2397_239723


namespace mean_of_y_l2397_239785

noncomputable def mean (xs : List ℝ) : ℝ :=
  xs.sum / xs.length

def regression_line (x : ℝ) : ℝ :=
  2 * x + 45

theorem mean_of_y (y₁ y₂ y₃ y₄ y₅ : ℝ) :
  mean [regression_line 1, regression_line 5, regression_line 7, regression_line 13, regression_line 19] = 63 := by
  sorry

end mean_of_y_l2397_239785


namespace AM_GM_inequality_example_l2397_239765

theorem AM_GM_inequality_example (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h_sum : a + b + c = 6) :
  1 / (a * (1 + b)) + 1 / (b * (1 + c)) + 1 / (c * (1 + a)) ≥ 1 / 2 :=
by
  sorry

end AM_GM_inequality_example_l2397_239765


namespace minimum_value_of_F_l2397_239713

theorem minimum_value_of_F (f g : ℝ → ℝ) (a b : ℝ) (h_odd_f : ∀ x, f (-x) = -f x) 
  (h_odd_g : ∀ x, g (-x) = -g x) (h_max_F : ∃ x > 0, a * f x + b * g x + 3 = 10) 
  : ∃ x < 0, a * f x + b * g x + 3 = -4 := 
sorry

end minimum_value_of_F_l2397_239713


namespace farm_produce_weeks_l2397_239732

def eggs_needed_per_week (saly_eggs ben_eggs ked_eggs : ℕ) : ℕ :=
  saly_eggs + ben_eggs + ked_eggs

def number_of_weeks (total_eggs : ℕ) (weekly_eggs : ℕ) : ℕ :=
  total_eggs / weekly_eggs

theorem farm_produce_weeks :
  let saly_eggs := 10
  let ben_eggs := 14
  let ked_eggs := 14 / 2
  let total_eggs := 124
  let weekly_eggs := eggs_needed_per_week saly_eggs ben_eggs ked_eggs
  number_of_weeks total_eggs weekly_eggs = 4 :=
by
  sorry 

end farm_produce_weeks_l2397_239732


namespace inverse_function_condition_l2397_239790

noncomputable def f (m x : ℝ) := (3 * x + 4) / (m * x - 5)

theorem inverse_function_condition (m : ℝ) :
  (∀ x : ℝ, f m (f m x) = x) ↔ m = -4 / 5 :=
by
  sorry

end inverse_function_condition_l2397_239790


namespace sum_of_prime_factors_172944_l2397_239728

theorem sum_of_prime_factors_172944 : 
  (∃ (a b c : ℕ), 2^a * 3^b * 1201^c = 172944 ∧ a = 4 ∧ b = 2 ∧ c = 1) → 2 + 3 + 1201 = 1206 := 
by 
  intros h 
  exact sorry

end sum_of_prime_factors_172944_l2397_239728


namespace solve_equation_naturals_l2397_239746

theorem solve_equation_naturals :
  ∀ (X Y Z : ℕ), X^Y + Y^Z = X * Y * Z ↔ 
    (X = 1 ∧ Y = 1 ∧ Z = 2) ∨ 
    (X = 2 ∧ Y = 2 ∧ Z = 2) ∨ 
    (X = 2 ∧ Y = 2 ∧ Z = 3) ∨ 
    (X = 4 ∧ Y = 2 ∧ Z = 3) ∨ 
    (X = 4 ∧ Y = 2 ∧ Z = 4) := 
by
  sorry

end solve_equation_naturals_l2397_239746


namespace right_triangles_product_hypotenuses_square_l2397_239749

/-- 
Given two right triangles T₁ and T₂ with areas 2 and 8 respectively. 
The hypotenuse of T₁ is congruent to one leg of T₂.
The shorter leg of T₁ is congruent to the hypotenuse of T₂.
Prove that the square of the product of the lengths of their hypotenuses is 4624.
-/
theorem right_triangles_product_hypotenuses_square :
  ∃ x y z u : ℝ, 
    (1 / 2) * x * y = 2 ∧
    (1 / 2) * y * u = 8 ∧
    x^2 + y^2 = z^2 ∧
    y^2 + (16 / y)^2 = z^2 ∧ 
    (z^2)^2 = 4624 := 
sorry

end right_triangles_product_hypotenuses_square_l2397_239749


namespace goose_eggs_calculation_l2397_239761

noncomputable def goose_eggs_total (E : ℕ) : Prop :=
  let hatched := (2/3) * E
  let survived_first_month := (3/4) * hatched
  let survived_first_year := (2/5) * survived_first_month
  survived_first_year = 110

theorem goose_eggs_calculation :
  goose_eggs_total 3300 :=
by
  have h1 : (2 : ℝ) / (3 : ℝ) ≠ 0 := by norm_num
  have h2 : (3 : ℝ) / (4 : ℝ) ≠ 0 := by norm_num
  have h3 : (2 : ℝ) / (5 : ℝ) ≠ 0 := by norm_num
  sorry

end goose_eggs_calculation_l2397_239761


namespace max_books_l2397_239707

theorem max_books (price_per_book available_money : ℕ) (h1 : price_per_book = 15) (h2 : available_money = 200) :
  ∃ n : ℕ, n = 13 ∧ n ≤ available_money / price_per_book :=
by {
  sorry
}

end max_books_l2397_239707


namespace solve_system_of_equations_l2397_239755

theorem solve_system_of_equations : 
  ∃ (x y : ℤ), 2 * x + 5 * y = 8 ∧ 3 * x - 5 * y = -13 ∧ x = -1 ∧ y = 2 :=
by
  sorry

end solve_system_of_equations_l2397_239755


namespace integral_solution_l2397_239745

noncomputable def definite_integral : ℝ :=
  ∫ x in (-2 : ℝ)..(0 : ℝ), (x + 2)^2 * (Real.cos (3 * x))

theorem integral_solution :
  definite_integral = (12 - 2 * Real.sin 6) / 27 :=
sorry

end integral_solution_l2397_239745


namespace cyclists_speeds_product_l2397_239776

theorem cyclists_speeds_product (u v : ℝ) (hu : u > 0) (hv : v > 0)
  (h₁ : 6 / u = 6 / v + 1 / 12) 
  (h₂ : v / 3 = u / 3 + 4) : 
  u * v = 864 := 
by
  sorry

end cyclists_speeds_product_l2397_239776


namespace case_a_second_player_wins_case_b_first_player_wins_case_c_winner_based_on_cell_color_case_d_examples_l2397_239756

-- Conditions for Case (a)
def corner_cell (board : Type) (cell : board) : Prop :=
  -- definition to determine if a cell is a corner cell
  sorry

theorem case_a_second_player_wins (board : Type) (starting_cell : board) (player : ℕ) :
  corner_cell board starting_cell → 
  player = 2 :=
by
  sorry
  
-- Conditions for Case (b)
def initial_setup_according_to_figure (board : Type) (starting_cell : board) : Prop :=
  -- definition to determine if a cell setup matches the figure
  sorry

theorem case_b_first_player_wins (board : Type) (starting_cell : board) (player : ℕ) :
  initial_setup_according_to_figure board starting_cell → 
  player = 1 :=
by
  sorry

-- Conditions for Case (c)
def black_cell (board : Type) (cell : board) : Prop :=
  -- definition to determine if a cell is black
  sorry

theorem case_c_winner_based_on_cell_color (board : Type) (starting_cell : board) (player : ℕ) :
  (black_cell board starting_cell → player = 1) ∧ (¬ black_cell board starting_cell → player = 2) :=
by
  sorry
  
-- Conditions for Case (d)
def same_starting_cell_two_games (board : Type) (starting_cell : board) : Prop :=
  -- definition for same starting cell but different outcomes in games
  sorry

theorem case_d_examples (board : Type) (starting_cell : board) (player1 player2 : ℕ) :
  (same_starting_cell_two_games board starting_cell → (player1 = 1 ∧ player2 = 2)) ∨ 
  (same_starting_cell_two_games board starting_cell → (player1 = 2 ∧ player2 = 1)) :=
by
  sorry

end case_a_second_player_wins_case_b_first_player_wins_case_c_winner_based_on_cell_color_case_d_examples_l2397_239756


namespace casey_nail_decorating_time_l2397_239750

/-- Given the conditions:
1. Casey wants to apply three coats: a base coat, a coat of paint, and a coat of glitter.
2. Each coat takes 20 minutes to apply.
3. Each coat requires 20 minutes of drying time before the next one can be applied.

Prove that the total time taken by Casey to finish decorating her fingernails and toenails is 120 minutes.
-/
theorem casey_nail_decorating_time
  (application_time : ℕ)
  (drying_time : ℕ)
  (num_coats : ℕ)
  (total_time : ℕ)
  (h_app_time : application_time = 20) 
  (h_dry_time : drying_time = 20)
  (h_num_coats : num_coats = 3)
  (h_total_time_eq : total_time = num_coats * (application_time + drying_time)) :
  total_time = 120 :=
sorry

end casey_nail_decorating_time_l2397_239750


namespace not_divisible_by_8_l2397_239792

theorem not_divisible_by_8 : ¬ (456294604884 % 8 = 0) := 
by
  have h : 456294604884 % 1000 = 884 := sorry -- This step reflects the conclusion that the last three digits are 884.
  have h_div : ¬ (884 % 8 = 0) := sorry -- This reflects that 884 is not divisible by 8.
  sorry

end not_divisible_by_8_l2397_239792


namespace calculate_total_money_made_l2397_239731

def original_price : ℕ := 51
def discount : ℕ := 8
def num_tshirts_sold : ℕ := 130
def discounted_price : ℕ := original_price - discount
def total_money_made : ℕ := discounted_price * num_tshirts_sold

theorem calculate_total_money_made :
  total_money_made = 5590 := 
sorry

end calculate_total_money_made_l2397_239731


namespace find_polynomial_l2397_239708

-- Define the polynomial conditions
structure CubicPolynomial :=
  (P : ℝ → ℝ)
  (P0 : ℝ)
  (P1 : ℝ)
  (P2 : ℝ)
  (P3 : ℝ)
  (cubic_eq : ∀ x, P x = P0 + P1 * x + P2 * x^2 + P3 * x^3)

theorem find_polynomial (P : CubicPolynomial) (h_neg1 : P.P (-1) = 2) (h0 : P.P 0 = 3) (h1 : P.P 1 = 1) (h2 : P.P 2 = 15) :
  ∀ x, P.P x = 3 + x - 2 * x^2 - x^3 :=
sorry

end find_polynomial_l2397_239708


namespace empty_solution_set_l2397_239796

theorem empty_solution_set 
  (x : ℝ) 
  (h : -2 + 3 * x - 2 * x^2 > 0) : 
  false :=
by
  -- Discriminant calculation to prove empty solution set
  let delta : ℝ := 9 - 4 * 2 * 2
  have h_delta : delta < 0 := by norm_num
  sorry

end empty_solution_set_l2397_239796


namespace problem_1_problem_2_problem_3_l2397_239701

theorem problem_1 : 
  ∀ x : ℝ, x^2 - 2 * x + 5 = (x - 1)^2 + 4 := 
sorry

theorem problem_2 (n : ℝ) (h : ∀ x : ℝ, x^2 + 2 * n * x + 3 = (x + 5)^2 - 25 + 3) : 
  n = -5 := 
sorry

theorem problem_3 (a : ℝ) (h : ∀ x : ℝ, (x^2 + 6 * x + 9) * (x^2 - 4 * x + 4) = ((x + a)^2 + b)^2) : 
  a = -1/2 := 
sorry

end problem_1_problem_2_problem_3_l2397_239701


namespace find_second_number_l2397_239702

-- Define the two numbers A and B
variables (A B : ℝ)

-- Define the conditions
def condition1 := 0.20 * A = 0.30 * B + 80
def condition2 := A = 580

-- Define the goal
theorem find_second_number (h1 : condition1 A B) (h2 : condition2 A) : B = 120 :=
by sorry

end find_second_number_l2397_239702


namespace parallel_lines_m_l2397_239748

theorem parallel_lines_m (m : ℝ) :
  (∀ x y : ℝ, 2 * x + 3 * y + 1 = 0 → 6 ≠ 0) ∧ 
  (∀ x y : ℝ, m * x + 6 * y - 5 = 0 → 6 ≠ 0) → 
  m = 4 :=
by
  intro h
  sorry

end parallel_lines_m_l2397_239748


namespace sin_A_value_l2397_239744

theorem sin_A_value
  (f : ℝ → ℝ)
  (cos_B : ℝ)
  (f_C_div_2 : ℝ)
  (C_acute : Prop) :
  (∀ x, f x = Real.cos (2 * x + Real.pi / 3) + Real.sin x ^ 2) →
  cos_B = 1 / 3 →
  f (C / 2) = -1 / 4 →
  (0 < C ∧ C < Real.pi / 2) →
  Real.sin (Real.arcsin (Real.sqrt 3 / 2) + Real.arcsin (2 * Real.sqrt 2 / 3)) = (2 * Real.sqrt 2 + Real.sqrt 3) / 6 :=
by
  intros
  sorry

end sin_A_value_l2397_239744


namespace general_pattern_specific_computation_l2397_239791

theorem general_pattern (n : ℕ) : n * (n + 1) * (n + 2) * (n + 3) + 1 = (n^2 + 3*n + 1)^2 :=
by
  sorry

theorem specific_computation : 2000 * 2001 * 2002 * 2003 + 1 = 4006001^2 :=
by
  have h := general_pattern 2000
  exact h

end general_pattern_specific_computation_l2397_239791


namespace percent_females_employed_l2397_239717

noncomputable def employed_percent (population: ℕ) : ℚ := 0.60
noncomputable def employed_males_percent (population: ℕ) : ℚ := 0.48

theorem percent_females_employed (population: ℕ) : ((employed_percent population) - (employed_males_percent population)) / (employed_percent population) = 0.20 :=
by
  sorry

end percent_females_employed_l2397_239717


namespace sum_first_n_terms_arithmetic_sequence_eq_l2397_239719

open Nat

noncomputable def sum_arithmetic_sequence (a₁ a₃ a₆ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  if h: n = 0 then 0 else n * a₁ + (n * (n - 1) * d) / 2

theorem sum_first_n_terms_arithmetic_sequence_eq 
  (a₁ a₃ a₆ : ℝ) (d : ℝ) (n : ℕ) 
  (h₀ : d ≠ 0)
  (h₁ : a₁ = 4)
  (h₂ : a₃ = a₁ + 2 * d)
  (h₃ : a₆ = a₁ + 5 * d)
  (h₄ : a₃^2 = a₁ * a₆) :
  sum_arithmetic_sequence a₁ a₃ a₆ d n = (n^2 + 7 * n) / 2 := 
by
  sorry

end sum_first_n_terms_arithmetic_sequence_eq_l2397_239719


namespace find_k_value_l2397_239703

-- Define the lines l1 and l2 with given conditions
def line1 (x y : ℝ) : Prop := x + 3 * y - 7 = 0
def line2 (k x y : ℝ) : Prop := k * x - y - 2 = 0

-- Define the condition for the quadrilateral to be circumscribed by a circle
def is_circumscribed (k : ℝ) : Prop :=
  ∃ (x y : ℝ), line1 x y ∧ line2 k x y ∧ 0 < x ∧ 0 < y

theorem find_k_value (k : ℝ) : is_circumscribed k → k = 3 := 
sorry

end find_k_value_l2397_239703


namespace reaction_rate_reduction_l2397_239783

theorem reaction_rate_reduction (k : ℝ) (NH3 Br2 NH3_new : ℝ) (v1 v2 : ℝ):
  (v1 = k * NH3^8 * Br2) →
  (v2 = k * NH3_new^8 * Br2) →
  (v2 / v1 = 60) →
  NH3_new = 60 ^ (1 / 8) :=
by
  intro hv1 hv2 hratio
  sorry

end reaction_rate_reduction_l2397_239783


namespace percentage_equivalence_l2397_239740

theorem percentage_equivalence (A B C P : ℝ)
  (hA : A = 0.80 * 600)
  (hB : B = 480)
  (hC : C = 960)
  (hP : P = (B / C) * 100) :
  A = P * 10 :=  -- Since P is the percentage, we use it to relate A to C
sorry

end percentage_equivalence_l2397_239740


namespace cube_surface_area_calc_l2397_239757

-- Edge length of the cube
def edge_length : ℝ := 7

-- Definition of the surface area formula for a cube
def surface_area (a : ℝ) : ℝ := 6 * (a ^ 2)

-- The main theorem stating the surface area of the cube with given edge length
theorem cube_surface_area_calc : surface_area edge_length = 294 :=
by
  sorry

end cube_surface_area_calc_l2397_239757


namespace linear_function_quadrants_passing_through_l2397_239743

theorem linear_function_quadrants_passing_through :
  ∀ (x : ℝ) (y : ℝ), (y = 2 * x + 3 → (x > 0 ∧ y > 0) ∨ (x < 0 ∧ y > 0) ∨ (x < 0 ∧ y < 0)) :=
by
  sorry

end linear_function_quadrants_passing_through_l2397_239743


namespace unique_solution_quadratic_l2397_239754

theorem unique_solution_quadratic (q : ℚ) :
  (∃ x : ℚ, q ≠ 0 ∧ q * x^2 - 16 * x + 9 = 0) ∧ (∀ y z : ℚ, (q * y^2 - 16 * y + 9 = 0 ∧ q * z^2 - 16 * z + 9 = 0) → y = z) → q = 64 / 9 :=
by
  sorry

end unique_solution_quadratic_l2397_239754


namespace initial_courses_of_bricks_l2397_239771

theorem initial_courses_of_bricks (x : ℕ) : 
    400 * x + 2 * 400 - 400 / 2 = 1800 → x = 3 :=
by
  sorry

end initial_courses_of_bricks_l2397_239771
