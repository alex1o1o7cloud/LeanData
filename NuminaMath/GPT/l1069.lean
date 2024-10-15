import Mathlib

namespace NUMINAMATH_GPT_average_homework_time_decrease_l1069_106970

theorem average_homework_time_decrease
  (initial_time final_time : ℝ)
  (rate_of_decrease : ℝ)
  (h1 : initial_time = 100)
  (h2 : final_time = 70) :
  initial_time * (1 - rate_of_decrease)^2 = final_time :=
by
  rw [h1, h2]
  sorry

end NUMINAMATH_GPT_average_homework_time_decrease_l1069_106970


namespace NUMINAMATH_GPT_geometric_seq_problem_l1069_106905

variable {a : ℕ → ℝ}

def geometric_sequence (a : ℕ → ℝ) :=
  ∃ r : ℝ, ∀ n : ℕ, a (n+1) = a n * r

theorem geometric_seq_problem (h_geom : geometric_sequence a) 
  (h_cond : a 8 * a 9 * a 10 = -a 13 ^ 2 ∧ -a 13 ^ 2 = -1000) :
  a 10 * a 12 = 100 * Real.sqrt 10 :=
by
  sorry

end NUMINAMATH_GPT_geometric_seq_problem_l1069_106905


namespace NUMINAMATH_GPT_det_scaled_matrix_l1069_106999

variable {R : Type*} [CommRing R]

def det2x2 (a b c d : R) : R := a * d - b * c

theorem det_scaled_matrix 
  (x y z w : R) 
  (h : det2x2 x y z w = 3) : 
  det2x2 (3 * x) (3 * y) (6 * z) (6 * w) = 54 := by
  sorry

end NUMINAMATH_GPT_det_scaled_matrix_l1069_106999


namespace NUMINAMATH_GPT_slices_per_banana_l1069_106954

-- Define conditions
def yogurts : ℕ := 5
def slices_per_yogurt : ℕ := 8
def bananas : ℕ := 4
def total_slices_needed : ℕ := yogurts * slices_per_yogurt

-- Statement to prove
theorem slices_per_banana : total_slices_needed / bananas = 10 := by sorry

end NUMINAMATH_GPT_slices_per_banana_l1069_106954


namespace NUMINAMATH_GPT_count_students_neither_math_physics_chemistry_l1069_106903

def total_students := 150

def students_math := 90
def students_physics := 70
def students_chemistry := 40

def students_math_and_physics := 20
def students_math_and_chemistry := 15
def students_physics_and_chemistry := 10
def students_all_three := 5

theorem count_students_neither_math_physics_chemistry :
  (total_students - 
   (students_math + students_physics + students_chemistry - 
    students_math_and_physics - students_math_and_chemistry - 
    students_physics_and_chemistry + students_all_three)) = 5 := by
  sorry

end NUMINAMATH_GPT_count_students_neither_math_physics_chemistry_l1069_106903


namespace NUMINAMATH_GPT_max_ab_bc_cd_da_l1069_106980

theorem max_ab_bc_cd_da (a b c d : ℝ) (h₀ : 0 ≤ a) (h₁ : 0 ≤ b) (h₂ : 0 ≤ c) (h₃ : 0 ≤ d) (h₄ : a + b + c + d = 200) :
  ab + bc + cd + da ≤ 10000 :=
sorry

end NUMINAMATH_GPT_max_ab_bc_cd_da_l1069_106980


namespace NUMINAMATH_GPT_gcd_16016_20020_l1069_106939

theorem gcd_16016_20020 : Int.gcd 16016 20020 = 4004 :=
by
  sorry

end NUMINAMATH_GPT_gcd_16016_20020_l1069_106939


namespace NUMINAMATH_GPT_bake_sale_money_made_l1069_106934

theorem bake_sale_money_made :
  let clementine_cookies := 72
  let jake_cookies := 2 * clementine_cookies
  let combined_cookies := jake_cookies + clementine_cookies
  let tory_cookies := combined_cookies / 2
  let total_cookies := clementine_cookies + jake_cookies + tory_cookies
  let price_per_cookie := 2
  let total_money := total_cookies * price_per_cookie
  total_money = 648 :=
by
  let clementine_cookies := 72
  let jake_cookies := 2 * clementine_cookies
  let combined_cookies := jake_cookies + clementine_cookies
  let tory_cookies := combined_cookies / 2
  let total_cookies := clementine_cookies + jake_cookies + tory_cookies
  let price_per_cookie := 2
  let total_money := total_cookies * price_per_cookie
  sorry

end NUMINAMATH_GPT_bake_sale_money_made_l1069_106934


namespace NUMINAMATH_GPT_tickets_left_l1069_106956

-- Definitions for the conditions given in the problem
def tickets_whack_a_mole : ℕ := 32
def tickets_skee_ball : ℕ := 25
def tickets_spent_on_hat : ℕ := 7

-- The main proof statement to verify
theorem tickets_left : tickets_whack_a_mole + tickets_skee_ball - tickets_spent_on_hat = 50 :=
by
  sorry

end NUMINAMATH_GPT_tickets_left_l1069_106956


namespace NUMINAMATH_GPT_largest_divisor_same_remainder_l1069_106965

theorem largest_divisor_same_remainder (n : ℕ) (h : 17 % n = 30 % n) : n = 13 :=
sorry

end NUMINAMATH_GPT_largest_divisor_same_remainder_l1069_106965


namespace NUMINAMATH_GPT_frac_eq_three_l1069_106913

theorem frac_eq_three (a b c : ℝ) 
  (h₁ : a / b = 4 / 3) (h₂ : (a + c) / (b - c) = 5 / 2) : 
  (3 * a + 2 * b) / (3 * a - 2 * b) = 3 :=
by
  sorry

end NUMINAMATH_GPT_frac_eq_three_l1069_106913


namespace NUMINAMATH_GPT_find_plaid_shirts_l1069_106910

def total_shirts : ℕ := 5
def total_pants : ℕ := 24
def total_items : ℕ := total_shirts + total_pants
def neither_plaid_nor_purple : ℕ := 21
def total_plaid_or_purple : ℕ := total_items - neither_plaid_nor_purple
def purple_pants : ℕ := 5
def plaid_shirts (p : ℕ) : Prop := total_plaid_or_purple - purple_pants = p

theorem find_plaid_shirts : plaid_shirts 3 := by
  unfold plaid_shirts
  repeat { sorry }

end NUMINAMATH_GPT_find_plaid_shirts_l1069_106910


namespace NUMINAMATH_GPT_common_noninteger_root_eq_coeffs_l1069_106964

theorem common_noninteger_root_eq_coeffs (p1 p2 q1 q2 : ℤ) (α : ℝ) :
  (α^2 + (p1: ℝ) * α + (q1: ℝ) = 0) ∧ (α^2 + (p2: ℝ) * α + (q2: ℝ) = 0) ∧ ¬(∃ (k : ℤ), α = k) → p1 = p2 ∧ q1 = q2 :=
by {
  sorry
}

end NUMINAMATH_GPT_common_noninteger_root_eq_coeffs_l1069_106964


namespace NUMINAMATH_GPT_sandy_change_correct_l1069_106945

def football_cost : ℚ := 914 / 100
def baseball_cost : ℚ := 681 / 100
def payment : ℚ := 20

def total_cost : ℚ := football_cost + baseball_cost
def change_received : ℚ := payment - total_cost

theorem sandy_change_correct :
  change_received = 405 / 100 :=
by
  -- The proof should go here
  sorry

end NUMINAMATH_GPT_sandy_change_correct_l1069_106945


namespace NUMINAMATH_GPT_cube_difference_l1069_106966

variable (x : ℝ)

theorem cube_difference (h : x - 1/x = 5) : x^3 - 1/x^3 = 120 := by
  sorry

end NUMINAMATH_GPT_cube_difference_l1069_106966


namespace NUMINAMATH_GPT_solid_produces_quadrilateral_l1069_106955

-- Define the solids and their properties
inductive Solid 
| cone 
| cylinder 
| sphere

-- Define the condition for a plane cut resulting in a quadrilateral cross-section
def can_produce_quadrilateral_cross_section (s : Solid) : Prop :=
  match s with
  | Solid.cone => False
  | Solid.cylinder => True
  | Solid.sphere => False

-- Theorem to prove that only a cylinder can produce a quadrilateral cross-section
theorem solid_produces_quadrilateral : 
  ∃ s : Solid, can_produce_quadrilateral_cross_section s :=
by
  existsi Solid.cylinder
  trivial

end NUMINAMATH_GPT_solid_produces_quadrilateral_l1069_106955


namespace NUMINAMATH_GPT_radius_range_l1069_106952

noncomputable def circle_eq (x y r : ℝ) := x^2 + y^2 = r^2

def point_P_on_line_AB (m n : ℝ) := 4 * m + 3 * n - 24 = 0

def point_P_in_interval (m : ℝ) := 0 ≤ m ∧ m ≤ 6

theorem radius_range {r : ℝ} :
  (∀ (m n x y : ℝ), point_P_in_interval m →
     circle_eq x y r →
     circle_eq ((x + m) / 2) ((y + n) / 2) r → 
     point_P_on_line_AB m n ∧
     (4 * r ^ 2 ≤ (25 / 9) * m ^ 2 - (64 / 3) * m + 64 ∧
     (25 / 9) * m ^ 2 - (64 / 3) * m + 64 ≤ 36 * r ^ 2)) →
  (8 / 3 ≤ r ∧ r < 12 / 5) :=
sorry

end NUMINAMATH_GPT_radius_range_l1069_106952


namespace NUMINAMATH_GPT_closest_integer_to_cube_root_of_1728_l1069_106915

theorem closest_integer_to_cube_root_of_1728: 
  ∃ n : ℕ, n^3 = 1728 ∧ (∀ m : ℤ, m^3 < 1728 → m < n) ∧ (∀ p : ℤ, p^3 > 1728 → p > n) :=
by
  sorry

end NUMINAMATH_GPT_closest_integer_to_cube_root_of_1728_l1069_106915


namespace NUMINAMATH_GPT_percentage_of_invalid_votes_l1069_106990

-- Candidate A got 60% of the total valid votes.
-- The total number of votes is 560000.
-- The number of valid votes polled in favor of candidate A is 285600.
variable (total_votes valid_votes_A : ℝ)
variable (percent_A : ℝ := 0.60)
variable (valid_votes total_invalid_votes percent_invalid_votes : ℝ)

axiom h1 : total_votes = 560000
axiom h2 : valid_votes_A = 285600
axiom h3 : valid_votes_A = percent_A * valid_votes
axiom h4 : total_invalid_votes = total_votes - valid_votes
axiom h5 : percent_invalid_votes = (total_invalid_votes / total_votes) * 100

theorem percentage_of_invalid_votes : percent_invalid_votes = 15 := by
  sorry

end NUMINAMATH_GPT_percentage_of_invalid_votes_l1069_106990


namespace NUMINAMATH_GPT_sum_of_x_values_l1069_106981

theorem sum_of_x_values (x : ℂ) (h₁ : x ≠ -3) (h₂ : 3 = (x^3 - 3 * x^2 - 10 * x) / (x + 3)) : x + (5 - x) = 5 :=
sorry

end NUMINAMATH_GPT_sum_of_x_values_l1069_106981


namespace NUMINAMATH_GPT_alpha_arctan_l1069_106998

open Real

theorem alpha_arctan {α : ℝ} (h1 : α ∈ Set.Ioo 0 (π/4)) (h2 : tan (α + (π/4)) = 2 * cos (2 * α)) : 
  α = arctan (2 - sqrt 3) := by
  sorry

end NUMINAMATH_GPT_alpha_arctan_l1069_106998


namespace NUMINAMATH_GPT_total_turnips_grown_l1069_106936

theorem total_turnips_grown 
  (melanie_turnips : ℕ) 
  (benny_turnips : ℕ) 
  (jack_turnips : ℕ) 
  (lynn_turnips : ℕ) : 
  melanie_turnips = 1395 ∧
  benny_turnips = 11380 ∧
  jack_turnips = 15825 ∧
  lynn_turnips = 23500 → 
  melanie_turnips + benny_turnips + jack_turnips + lynn_turnips = 52100 :=
by
  intros h
  rcases h with ⟨hm, hb, hj, hl⟩
  sorry

end NUMINAMATH_GPT_total_turnips_grown_l1069_106936


namespace NUMINAMATH_GPT_no_two_digit_factorization_2023_l1069_106994

theorem no_two_digit_factorization_2023 :
  ¬ ∃ (a b : ℕ), 10 ≤ a ∧ a < 100 ∧ 10 ≤ b ∧ b < 100 ∧ a * b = 2023 := 
by
  sorry

end NUMINAMATH_GPT_no_two_digit_factorization_2023_l1069_106994


namespace NUMINAMATH_GPT_alice_more_than_half_sum_l1069_106969

-- Conditions
def row_of_fifty_coins (denominations : List ℤ) : Prop :=
  denominations.length = 50 ∧ (List.sum denominations) % 2 = 1

def alice_starts (denominations : List ℤ) : Prop := True
def bob_follows (denominations : List ℤ) : Prop := True
def alternating_selection (denominations : List ℤ) : Prop := True

-- Question/Proof Goal
theorem alice_more_than_half_sum (denominations : List ℤ) 
  (h1 : row_of_fifty_coins denominations)
  (h2 : alice_starts denominations)
  (h3 : bob_follows denominations)
  (h4 : alternating_selection denominations) :
  ∃ s_A : ℤ, s_A > (List.sum denominations) / 2 ∧ s_A ≤ List.sum denominations :=
sorry

end NUMINAMATH_GPT_alice_more_than_half_sum_l1069_106969


namespace NUMINAMATH_GPT_conic_section_eccentricity_l1069_106950

theorem conic_section_eccentricity (m : ℝ) (h : 2 * 8 = m^2) :
    (∃ e : ℝ, ((e = (Real.sqrt 2) / 2) ∨ (e = Real.sqrt 3))) :=
by
  sorry

end NUMINAMATH_GPT_conic_section_eccentricity_l1069_106950


namespace NUMINAMATH_GPT_perfect_squares_represented_as_diff_of_consecutive_cubes_l1069_106972

theorem perfect_squares_represented_as_diff_of_consecutive_cubes : ∃ (count : ℕ), 
  count = 40 ∧ 
  ∀ n : ℕ, 
  (∃ a : ℕ, a^2 = ( ( n + 1 )^3 - n^3 ) ∧ a^2 < 20000) → count = 40 := by 
sorry

end NUMINAMATH_GPT_perfect_squares_represented_as_diff_of_consecutive_cubes_l1069_106972


namespace NUMINAMATH_GPT_proof_valid_x_values_l1069_106987

noncomputable def valid_x_values (x : ℝ) : Prop :=
  (x^2 + 2*x^3 - 3*x^4) / (x + 2*x^2 - 3*x^3) ≤ 1

theorem proof_valid_x_values :
  {x : ℝ | valid_x_values x} = {x : ℝ | (x < -1) ∨ (x > -1 ∧ x < 0) ∨ (x > 0 ∧ x < 1)} :=
by {
  sorry
}

end NUMINAMATH_GPT_proof_valid_x_values_l1069_106987


namespace NUMINAMATH_GPT_simplify_expression_l1069_106983

theorem simplify_expression : 
  (1 / (1 / (1 / 3) ^ 1 + 1 / (1 / 3) ^ 2 + 1 / (1 / 3) ^ 3 + 1 / (1 / 3) ^ 4)) = 1 / 120 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1069_106983


namespace NUMINAMATH_GPT_problem_l1069_106978

noncomputable def f : ℝ → ℝ := sorry 

theorem problem
  (h_even : ∀ x : ℝ, f x = f (-x))
  (h_func : ∀ x : ℝ, f (2 + x) = -f (2 - x))
  (h_value : f (-3) = -2) :
  f 2007 = 2 :=
sorry

end NUMINAMATH_GPT_problem_l1069_106978


namespace NUMINAMATH_GPT_range_of_function_l1069_106989

noncomputable def function_range: Set ℝ :=
  { y | ∃ x, y = (1/2)^(x^2 - 2*x + 2) }

theorem range_of_function :
  function_range = {y | 0 < y ∧ y ≤ 1/2} :=
sorry

end NUMINAMATH_GPT_range_of_function_l1069_106989


namespace NUMINAMATH_GPT_product_to_difference_l1069_106919

def x := 88 * 1.25
def y := 150 * 0.60
def z := 60 * 1.15

def product := x * y * z
def difference := x - y

theorem product_to_difference :
  product ^ difference = 683100 ^ 20 := 
sorry

end NUMINAMATH_GPT_product_to_difference_l1069_106919


namespace NUMINAMATH_GPT_no_representation_of_216p3_l1069_106948

theorem no_representation_of_216p3 (p : ℕ) (hp_prime : Nat.Prime p)
  (hp_form : ∃ m : ℤ, p = 4 * m + 1) : ¬ ∃ x y z : ℤ, 216 * (p ^ 3) = x^2 + y^2 + z^9 := by
  sorry

end NUMINAMATH_GPT_no_representation_of_216p3_l1069_106948


namespace NUMINAMATH_GPT_circle_area_percentage_decrease_l1069_106973

theorem circle_area_percentage_decrease (r : ℝ) (A : ℝ := Real.pi * r^2) 
  (r' : ℝ := 0.5 * r) (A' : ℝ := Real.pi * (r')^2) :
  (A - A') / A * 100 = 75 := 
by
  sorry

end NUMINAMATH_GPT_circle_area_percentage_decrease_l1069_106973


namespace NUMINAMATH_GPT_area_increase_percentage_l1069_106958

-- Define the original dimensions l and w as non-zero real numbers
variables (l w : ℝ) (hl : l ≠ 0) (hw : w ≠ 0)

-- Define the new dimensions after increase
def new_length := 1.15 * l
def new_width := 1.25 * w

-- Define the original and new areas
def original_area := l * w
def new_area := new_length l * new_width w

-- The statement to prove
theorem area_increase_percentage :
  ((new_area l w - original_area l w) / original_area l w) * 100 = 43.75 :=
by
  sorry

end NUMINAMATH_GPT_area_increase_percentage_l1069_106958


namespace NUMINAMATH_GPT_students_didnt_make_cut_l1069_106911

theorem students_didnt_make_cut (g b c : ℕ) (hg : g = 15) (hb : b = 25) (hc : c = 7) : g + b - c = 33 := by
  sorry

end NUMINAMATH_GPT_students_didnt_make_cut_l1069_106911


namespace NUMINAMATH_GPT_Sherman_weekly_driving_time_l1069_106974

theorem Sherman_weekly_driving_time (daily_commute : ℕ := 30) (weekend_drive : ℕ := 2) : 
  (5 * (2 * daily_commute) / 60 + 2 * weekend_drive) = 9 := 
by
  sorry

end NUMINAMATH_GPT_Sherman_weekly_driving_time_l1069_106974


namespace NUMINAMATH_GPT_sqrt_31_between_5_and_6_l1069_106901

theorem sqrt_31_between_5_and_6
  (h1 : Real.sqrt 25 = 5)
  (h2 : Real.sqrt 36 = 6)
  (h3 : 25 < 31)
  (h4 : 31 < 36) :
  5 < Real.sqrt 31 ∧ Real.sqrt 31 < 6 :=
sorry

end NUMINAMATH_GPT_sqrt_31_between_5_and_6_l1069_106901


namespace NUMINAMATH_GPT_placements_for_nine_squares_l1069_106967

-- Define the parameters and conditions of the problem
def countPlacements (n : ℕ) : ℕ := sorry

theorem placements_for_nine_squares : countPlacements 9 = 25 := sorry

end NUMINAMATH_GPT_placements_for_nine_squares_l1069_106967


namespace NUMINAMATH_GPT_darnel_difference_l1069_106953

theorem darnel_difference (sprint_1 jog_1 sprint_2 jog_2 sprint_3 jog_3 : ℝ)
  (h_sprint_1 : sprint_1 = 0.8932)
  (h_jog_1 : jog_1 = 0.7683)
  (h_sprint_2 : sprint_2 = 0.9821)
  (h_jog_2 : jog_2 = 0.4356)
  (h_sprint_3 : sprint_3 = 1.2534)
  (h_jog_3 : jog_3 = 0.6549) :
  (sprint_1 + sprint_2 + sprint_3 - (jog_1 + jog_2 + jog_3)) = 1.2699 := by
  sorry

end NUMINAMATH_GPT_darnel_difference_l1069_106953


namespace NUMINAMATH_GPT_square_perimeter_l1069_106928

theorem square_perimeter (s : ℕ) (h : 5 * s / 2 = 40) : 4 * s = 64 := by
  sorry

end NUMINAMATH_GPT_square_perimeter_l1069_106928


namespace NUMINAMATH_GPT_cake_eaten_fraction_l1069_106914

noncomputable def cake_eaten_after_four_trips : ℚ :=
  let consumption_ratio := (1/3 : ℚ)
  let first_trip := consumption_ratio
  let second_trip := consumption_ratio * consumption_ratio
  let third_trip := second_trip * consumption_ratio
  let fourth_trip := third_trip * consumption_ratio
  first_trip + second_trip + third_trip + fourth_trip

theorem cake_eaten_fraction : cake_eaten_after_four_trips = (40 / 81 : ℚ) :=
by
  sorry

end NUMINAMATH_GPT_cake_eaten_fraction_l1069_106914


namespace NUMINAMATH_GPT_find_b_l1069_106937

theorem find_b
  (a b : ℝ)
  (f : ℝ → ℝ)
  (h_f : ∀ x, f x = (1/12) * x^2 + a * x + b)
  (A C: ℝ × ℝ)
  (hA : A = (x1, 0))
  (hC : C = (x2, 0))
  (T : ℝ × ℝ)
  (hT : T = (3, 3))
  (h_TA : dist (3, 3) (x1, 0) = dist (3, 3) (0, b))
  (h_TB : dist (3, 3) (0, b) = dist (3, 3) (x2, 0))
  (vietas : x1 * x2 = 12 * b)
  : b = -6 := 
sorry

end NUMINAMATH_GPT_find_b_l1069_106937


namespace NUMINAMATH_GPT_product_gcd_lcm_l1069_106935

-- Conditions.
def num1 : ℕ := 12
def num2 : ℕ := 9

-- Theorem to prove.
theorem product_gcd_lcm (a b : ℕ) (h1 : a = num1) (h2 : b = num2) :
  (Nat.gcd a b) * (Nat.lcm a b) = 108 :=
by
  sorry

end NUMINAMATH_GPT_product_gcd_lcm_l1069_106935


namespace NUMINAMATH_GPT_correct_transformation_l1069_106968

theorem correct_transformation (a b : ℝ) (hb : b ≠ 0): (a / b = 2 * a / (2 * b)) :=
by
  sorry

end NUMINAMATH_GPT_correct_transformation_l1069_106968


namespace NUMINAMATH_GPT_sin_and_tan_inequality_l1069_106932

theorem sin_and_tan_inequality (n : ℕ) (hn : 0 < n) :
  2 * Real.sin (1 / n) + Real.tan (1 / n) > 3 / n :=
sorry

end NUMINAMATH_GPT_sin_and_tan_inequality_l1069_106932


namespace NUMINAMATH_GPT_percent_increase_expenditure_l1069_106925

theorem percent_increase_expenditure (cost_per_minute_2005 minutes_2005 minutes_2020 total_expenditure_2005 total_expenditure_2020 : ℕ)
  (h1 : cost_per_minute_2005 = 10)
  (h2 : minutes_2005 = 200)
  (h3 : minutes_2020 = 2 * minutes_2005)
  (h4 : total_expenditure_2005 = minutes_2005 * cost_per_minute_2005)
  (h5 : total_expenditure_2020 = minutes_2020 * cost_per_minute_2005) :
  ((total_expenditure_2020 - total_expenditure_2005) * 100 / total_expenditure_2005) = 100 :=
by
  sorry

end NUMINAMATH_GPT_percent_increase_expenditure_l1069_106925


namespace NUMINAMATH_GPT_roots_satisfy_conditions_l1069_106929

variable (a x1 x2 : ℝ)

-- Conditions
def condition1 : Prop := x1 * x2 + x1 + x2 - a = 0
def condition2 : Prop := x1 * x2 - a * (x1 + x2) + 1 = 0

-- Derived quadratic equation
def quadratic_eq : Prop := ∃ x : ℝ, x^2 - x + (a - 1) = 0

theorem roots_satisfy_conditions (h1: condition1 a x1 x2) (h2: condition2 a x1 x2) : quadratic_eq a :=
  sorry

end NUMINAMATH_GPT_roots_satisfy_conditions_l1069_106929


namespace NUMINAMATH_GPT_petunia_fertilizer_problem_l1069_106904

theorem petunia_fertilizer_problem
  (P : ℕ)
  (h1 : 4 * P * 8 + 3 * 6 * 3 + 2 * 2 = 314) :
  P = 8 :=
by
  sorry

end NUMINAMATH_GPT_petunia_fertilizer_problem_l1069_106904


namespace NUMINAMATH_GPT_solution_set_m5_range_m_sufficient_condition_l1069_106922

theorem solution_set_m5 (x : ℝ) : 
  (|x + 1| + |x - 2| > 5) ↔ (x < -2 ∨ x > 3) := 
sorry

theorem range_m_sufficient_condition (x m : ℝ) (h : ∀ x : ℝ, |x + 1| + |x - 2| - m ≥ 2) : 
  m ≤ 1 := 
sorry

end NUMINAMATH_GPT_solution_set_m5_range_m_sufficient_condition_l1069_106922


namespace NUMINAMATH_GPT_range_of_a_l1069_106959

theorem range_of_a (a : ℝ) : (∀ x : ℝ, |x - 3| + |x - a| ≥ 3) ↔ a ≤ 0 ∨ a ≥ 6 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1069_106959


namespace NUMINAMATH_GPT_score_in_first_round_l1069_106906

theorem score_in_first_round (cards : List ℕ) (scores : List ℕ) 
  (total_rounds : ℕ) (last_round_score : ℕ) (total_score : ℕ) : 
  cards = [2, 4, 7, 13] ∧ scores = [16, 17, 21, 24] ∧ total_rounds = 3 ∧ last_round_score = 2 ∧ total_score = 16 →
  ∃ first_round_score, first_round_score = 7 := by
  sorry

end NUMINAMATH_GPT_score_in_first_round_l1069_106906


namespace NUMINAMATH_GPT_total_pieces_of_bread_correct_l1069_106931

-- Define the constants for the number of bread pieces needed per type of sandwich
def pieces_per_regular_sandwich : ℕ := 2
def pieces_per_double_meat_sandwich : ℕ := 3

-- Define the quantities of each type of sandwich
def regular_sandwiches : ℕ := 14
def double_meat_sandwiches : ℕ := 12

-- Define the total pieces of bread calculation
def total_pieces_of_bread : ℕ := pieces_per_regular_sandwich * regular_sandwiches + pieces_per_double_meat_sandwich * double_meat_sandwiches

-- State the theorem
theorem total_pieces_of_bread_correct : total_pieces_of_bread = 64 :=
by
  -- Proof goes here (using sorry for now)
  sorry

end NUMINAMATH_GPT_total_pieces_of_bread_correct_l1069_106931


namespace NUMINAMATH_GPT_complement_of_A_in_U_l1069_106923

def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {2, 4}

theorem complement_of_A_in_U : (U \ A) = {1, 3, 5} := by
  sorry

end NUMINAMATH_GPT_complement_of_A_in_U_l1069_106923


namespace NUMINAMATH_GPT_ball_draw_probability_l1069_106909

/-- 
Four balls labeled with numbers 1, 2, 3, 4 are placed in an urn. 
A ball is drawn, its number is recorded, and then the ball is returned to the urn. 
This process is repeated three times. Each ball is equally likely to be drawn on each occasion. 
Given that the sum of the numbers recorded is 7, the probability that the ball numbered 2 was drawn twice is 1/4. 
-/
theorem ball_draw_probability :
  let draws := [(1, 1, 5),(1, 2, 4),(1, 3, 3),(2, 2, 3)]
  (3 / 12 = 1 / 4) :=
by
  sorry

end NUMINAMATH_GPT_ball_draw_probability_l1069_106909


namespace NUMINAMATH_GPT_probability_of_odd_sum_l1069_106912

theorem probability_of_odd_sum (P : ℝ → Prop) 
    (P_even_sum : ℝ)
    (P_odd_sum : ℝ)
    (h1 : P_even_sum = 2 * P_odd_sum) 
    (h2 : P_even_sum + P_odd_sum = 1) :
    P_odd_sum = 4/9 := 
sorry

end NUMINAMATH_GPT_probability_of_odd_sum_l1069_106912


namespace NUMINAMATH_GPT_x_squared_y_squared_iff_x_squared_y_squared_not_sufficient_x_squared_y_squared_necessary_l1069_106920

theorem x_squared_y_squared_iff (x y : ℝ) : x ^ 2 = y ^ 2 ↔ x = y ∨ x = -y := by
  sorry

theorem x_squared_y_squared_not_sufficient (x y : ℝ) : (x ^ 2 = y ^ 2) → (x = y ∨ x = -y) := by
  sorry

theorem x_squared_y_squared_necessary (x y : ℝ) : (x = y) → (x ^ 2 = y ^ 2) := by
  sorry

end NUMINAMATH_GPT_x_squared_y_squared_iff_x_squared_y_squared_not_sufficient_x_squared_y_squared_necessary_l1069_106920


namespace NUMINAMATH_GPT_perimeter_C_is_74_l1069_106971

/-- Definitions of side lengths based on given perimeters -/
def side_length_A (p_A : ℕ) : ℕ :=
  p_A / 4

def side_length_B (p_B : ℕ) : ℕ :=
  p_B / 4

/-- Definition of side length of C in terms of side lengths of A and B -/
def side_length_C (s_A s_B : ℕ) : ℚ :=
  (s_A : ℚ) / 2 + 2 * (s_B : ℚ)

/-- Definition of perimeter in terms of side length -/
def perimeter (s : ℚ) : ℚ :=
  4 * s

/-- Theorem statement: the perimeter of square C is 74 -/
theorem perimeter_C_is_74 (p_A p_B : ℕ) (h₁ : p_A = 20) (h₂ : p_B = 32) :
  perimeter (side_length_C (side_length_A p_A) (side_length_B p_B)) = 74 := by
  sorry

end NUMINAMATH_GPT_perimeter_C_is_74_l1069_106971


namespace NUMINAMATH_GPT_monthly_earnings_is_correct_l1069_106996

-- Conditions as definitions

def first_floor_cost_per_room : ℕ := 15
def second_floor_cost_per_room : ℕ := 20
def first_floor_rooms : ℕ := 3
def second_floor_rooms : ℕ := 3
def third_floor_rooms : ℕ := 3
def occupied_third_floor_rooms : ℕ := 2

-- Calculated values from conditions
def third_floor_cost_per_room : ℕ := 2 * first_floor_cost_per_room

-- Total earnings on each floor
def first_floor_earnings : ℕ := first_floor_cost_per_room * first_floor_rooms
def second_floor_earnings : ℕ := second_floor_cost_per_room * second_floor_rooms
def third_floor_earnings : ℕ := third_floor_cost_per_room * occupied_third_floor_rooms

-- Total monthly earnings
def total_monthly_earnings : ℕ :=
  first_floor_earnings + second_floor_earnings + third_floor_earnings

theorem monthly_earnings_is_correct : total_monthly_earnings = 165 := by
  -- proof omitted
  sorry

end NUMINAMATH_GPT_monthly_earnings_is_correct_l1069_106996


namespace NUMINAMATH_GPT_arithmetic_square_root_of_16_is_4_l1069_106941

theorem arithmetic_square_root_of_16_is_4 : ∃ x : ℤ, x * x = 16 ∧ x = 4 := 
sorry

end NUMINAMATH_GPT_arithmetic_square_root_of_16_is_4_l1069_106941


namespace NUMINAMATH_GPT_Nicky_wait_time_l1069_106985

theorem Nicky_wait_time (x : ℕ) (h1 : x + (4 * x + 14) = 114) : x = 20 :=
by {
  sorry
}

end NUMINAMATH_GPT_Nicky_wait_time_l1069_106985


namespace NUMINAMATH_GPT_rectangle_length_width_difference_l1069_106916

theorem rectangle_length_width_difference
  (x y : ℝ)
  (h1 : x + y = 40)
  (h2 : x^2 + y^2 = 800) :
  x - y = 0 :=
sorry

end NUMINAMATH_GPT_rectangle_length_width_difference_l1069_106916


namespace NUMINAMATH_GPT_number_of_true_propositions_is_two_l1069_106993

def proposition1 (f : ℝ → ℝ) : Prop :=
∀ x : ℝ, f (1 + x) = f (1 - x)

def proposition2 : Prop :=
∀ x : ℝ, 2 * Real.sin x * Real.cos (abs x) -- minimum period not 1
  -- We need to define proper periodicity which is complex; so here's a simplified representation
  ≠ 2 * Real.sin (x + 1) * Real.cos (abs (x + 1))

def increasing_sequence (a : ℕ → ℝ) : Prop :=
∀ n : ℕ, a (n + 1) > a n

def proposition3 (k : ℝ) : Prop :=
∀ n : ℕ, n > 0 → increasing_sequence (fun n => n^2 + k * n + 2)

def condition (f : ℝ → ℝ) (k : ℝ) : Prop :=
proposition1 f ∧ proposition2 ∧ proposition3 k

theorem number_of_true_propositions_is_two (f : ℝ → ℝ) (k : ℝ) :
  condition f k → 2 = 2 :=
by
  sorry

end NUMINAMATH_GPT_number_of_true_propositions_is_two_l1069_106993


namespace NUMINAMATH_GPT_ellipse_problem_l1069_106933

theorem ellipse_problem
  (a b : ℝ)
  (h₀ : 0 < a)
  (h₁ : 0 < b)
  (h₂ : a > b)
  (P Q : ℝ × ℝ)
  (ellipse_eq : ∀ (x y : ℝ), (x, y) ∈ {p : ℝ × ℝ | (p.1 ^ 2) / (a ^ 2) + (p.2 ^ 2) / (b ^ 2) = 1})
  (A : ℝ × ℝ)
  (hA : A = (a, 0))
  (R : ℝ × ℝ)
  (O : ℝ × ℝ)
  (hO : O = (0, 0))
  (AQ_OP_parallels : ∀ (x y : ℝ) (Qx Qy Px Py : ℝ), 
    x = a ∧ y = 0  ∧ (Qx, Qy) = (x, y) ↔ (O.1, O.2) = (Px, Py)
    ) :
  ∀ (AQ AR OP : ℝ), 
  AQ = dist (a, 0) Q → 
  AR = dist A R → 
  OP = dist O P → 
  |AQ * AR| / (OP ^ 2) = 2 :=
  sorry

end NUMINAMATH_GPT_ellipse_problem_l1069_106933


namespace NUMINAMATH_GPT_remainder_of_3045_div_32_l1069_106921

theorem remainder_of_3045_div_32 : 3045 % 32 = 5 :=
by sorry

end NUMINAMATH_GPT_remainder_of_3045_div_32_l1069_106921


namespace NUMINAMATH_GPT_num_students_above_120_l1069_106986

noncomputable def class_size : ℤ := 60
noncomputable def mean_score : ℝ := 110
noncomputable def std_score : ℝ := sorry  -- We do not know σ explicitly
noncomputable def probability_100_to_110 : ℝ := 0.35

def normal_distribution (x : ℝ) : Prop :=
  sorry -- placeholder for the actual normal distribution formula N(110, σ^2)

theorem num_students_above_120 :
  ∃ (students_above_120 : ℤ),
  (class_size = 60) ∧
  (∀ score, normal_distribution score → (100 ≤ score ∧ score ≤ 110) → probability_100_to_110 = 0.35) →
  students_above_120 = 9 :=
sorry

end NUMINAMATH_GPT_num_students_above_120_l1069_106986


namespace NUMINAMATH_GPT_max_receptivity_compare_receptivity_l1069_106957

-- Define the piecewise function f(x)
noncomputable def f (x : ℝ) : ℝ :=
  if 0 < x ∧ x ≤ 10 then -0.1 * x^2 + 2.6 * x + 43
  else if 10 < x ∧ x <= 16 then 59
  else if 16 < x ∧ x <= 30 then -3 * x + 107
  else 0 -- To cover the case when x is outside the given ranges

-- Problem 1
theorem max_receptivity :
  f 10 = 59 ∧ ∀ x, 10 < x ∧ x ≤ 16 → f x = 59 :=
by
  sorry

-- Problem 2
theorem compare_receptivity :
  f 5 > f 20 :=
by
  sorry

end NUMINAMATH_GPT_max_receptivity_compare_receptivity_l1069_106957


namespace NUMINAMATH_GPT_center_of_circle_l1069_106917

-- Let's define the circle as a set of points satisfying the given condition.
def circle (x y : ℝ) : Prop := (x - 2) ^ 2 + (y + 1) ^ 2 = 4

-- Prove that the point (2, -1) is the center of this circle in ℝ².
theorem center_of_circle : ∀ (x y : ℝ), circle (x - 2) (y + 1) ↔ (x, y) = (2, -1) :=
by
  intros x y
  sorry

end NUMINAMATH_GPT_center_of_circle_l1069_106917


namespace NUMINAMATH_GPT_find_n_l1069_106902

-- Definitions based on conditions
def a := (5 + 10 + 15 + 20 + 25 + 30 + 35) / 7
def b (n : ℕ) := 2 * n

-- Theorem stating the problem
theorem find_n (n : ℕ) (h : a^2 - (b n)^2 = 0) : n = 10 :=
by sorry

end NUMINAMATH_GPT_find_n_l1069_106902


namespace NUMINAMATH_GPT_number_of_terms_in_expanded_polynomial_l1069_106938

theorem number_of_terms_in_expanded_polynomial : 
  ∀ (a : Fin 4 → Type) (b : Fin 2 → Type) (c : Fin 3 → Type), 
  (4 * 2 * 3 = 24) := 
by
  intros a b c
  sorry

end NUMINAMATH_GPT_number_of_terms_in_expanded_polynomial_l1069_106938


namespace NUMINAMATH_GPT_percentage_decrease_is_20_l1069_106949

-- Define the original and new prices in Rs.
def original_price : ℕ := 775
def new_price : ℕ := 620

-- Define the decrease in price
def decrease_in_price : ℕ := original_price - new_price

-- Define the formula to calculate the percentage decrease
def percentage_decrease (orig_price new_price : ℕ) : ℕ :=
  (decrease_in_price * 100) / orig_price

-- Prove that the percentage decrease is 20%
theorem percentage_decrease_is_20 :
  percentage_decrease original_price new_price = 20 :=
by
  sorry

end NUMINAMATH_GPT_percentage_decrease_is_20_l1069_106949


namespace NUMINAMATH_GPT_printing_time_345_l1069_106942

def printing_time (total_pages : ℕ) (rate : ℕ) : ℕ :=
  total_pages / rate

theorem printing_time_345 :
  printing_time 345 23 = 15 :=
by
  sorry

end NUMINAMATH_GPT_printing_time_345_l1069_106942


namespace NUMINAMATH_GPT_shift_sine_graph_l1069_106907

theorem shift_sine_graph (x : ℝ) : 
  (∃ θ : ℝ, θ = (5 * Real.pi) / 4 ∧ 
  y = Real.sin (x - Real.pi / 4) → y = Real.sin (x + θ) 
  ∧ 0 ≤ θ ∧ θ < 2 * Real.pi) := sorry

end NUMINAMATH_GPT_shift_sine_graph_l1069_106907


namespace NUMINAMATH_GPT_corveus_sleep_hours_l1069_106984

-- Definition of the recommended hours of sleep per day
def recommended_sleep_per_day : ℕ := 6

-- Definition of the hours of sleep Corveus lacks per week
def lacking_sleep_per_week : ℕ := 14

-- Definition of days in a week
def days_in_week : ℕ := 7

-- Prove that Corveus sleeps 4 hours per day given the conditions
theorem corveus_sleep_hours :
  (recommended_sleep_per_day * days_in_week - lacking_sleep_per_week) / days_in_week = 4 :=
by
  -- The proof steps would go here
  sorry

end NUMINAMATH_GPT_corveus_sleep_hours_l1069_106984


namespace NUMINAMATH_GPT_faith_overtime_hours_per_day_l1069_106982

noncomputable def normal_pay_per_hour : ℝ := 13.50
noncomputable def normal_daily_hours : ℕ := 8
noncomputable def normal_weekly_days : ℕ := 5
noncomputable def total_weekly_earnings : ℝ := 675
noncomputable def overtime_rate_multiplier : ℝ := 1.5

noncomputable def normal_weekly_hours := normal_daily_hours * normal_weekly_days
noncomputable def normal_weekly_earnings := normal_pay_per_hour * normal_weekly_hours
noncomputable def overtime_earnings := total_weekly_earnings - normal_weekly_earnings
noncomputable def overtime_pay_per_hour := normal_pay_per_hour * overtime_rate_multiplier
noncomputable def total_overtime_hours := overtime_earnings / overtime_pay_per_hour
noncomputable def overtime_hours_per_day := total_overtime_hours / normal_weekly_days

theorem faith_overtime_hours_per_day :
  overtime_hours_per_day = 1.33 := 
by 
  sorry

end NUMINAMATH_GPT_faith_overtime_hours_per_day_l1069_106982


namespace NUMINAMATH_GPT_max_b_value_l1069_106977

theorem max_b_value
  (a b c : ℕ)
  (h1 : 1 < c)
  (h2 : c < b)
  (h3 : b < a)
  (h4 : a * b * c = 240) : b = 10 :=
  sorry

end NUMINAMATH_GPT_max_b_value_l1069_106977


namespace NUMINAMATH_GPT_kitten_length_after_4_months_l1069_106992

theorem kitten_length_after_4_months
  (initial_length : ℕ)
  (doubled_length_2_weeks : ℕ)
  (final_length_4_months : ℕ)
  (h1 : initial_length = 4)
  (h2 : doubled_length_2_weeks = initial_length * 2)
  (h3 : final_length_4_months = doubled_length_2_weeks * 2) :
  final_length_4_months = 16 := 
by
  sorry

end NUMINAMATH_GPT_kitten_length_after_4_months_l1069_106992


namespace NUMINAMATH_GPT_marks_in_math_l1069_106961

theorem marks_in_math (e p c b : ℕ) (avg : ℚ) (n : ℕ) (total_marks_other_subjects : ℚ) :
  e = 45 →
  p = 52 →
  c = 47 →
  b = 55 →
  avg = 46.8 →
  n = 5 →
  total_marks_other_subjects = (e + p + c + b : ℕ) →
  (avg * n) - total_marks_other_subjects = 35 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end NUMINAMATH_GPT_marks_in_math_l1069_106961


namespace NUMINAMATH_GPT_xyz_value_l1069_106918

theorem xyz_value (x y z : ℝ) 
  (h1 : (x + y + z) * (xy + xz + yz) = 40) 
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 10) 
  : x * y * z = 10 :=
sorry

end NUMINAMATH_GPT_xyz_value_l1069_106918


namespace NUMINAMATH_GPT_greatest_non_fiction_books_l1069_106924

def is_prime (p : ℕ) := p > 1 ∧ (∀ d : ℕ, d ∣ p → d = 1 ∨ d = p)

theorem greatest_non_fiction_books (n f k : ℕ) :
  (n + f = 100 ∧ f = n + k ∧ is_prime k) → n ≤ 49 :=
by
  sorry

end NUMINAMATH_GPT_greatest_non_fiction_books_l1069_106924


namespace NUMINAMATH_GPT_total_bill_correct_l1069_106927

def scoop_cost : ℕ := 2
def pierre_scoops : ℕ := 3
def mom_scoops : ℕ := 4

def pierre_total : ℕ := pierre_scoops * scoop_cost
def mom_total : ℕ := mom_scoops * scoop_cost
def total_bill : ℕ := pierre_total + mom_total

theorem total_bill_correct : total_bill = 14 :=
by
  sorry

end NUMINAMATH_GPT_total_bill_correct_l1069_106927


namespace NUMINAMATH_GPT_percentage_difference_l1069_106940

theorem percentage_difference (x : ℝ) : 
  (62 / 100) * 150 - (x / 100) * 250 = 43 → x = 20 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_percentage_difference_l1069_106940


namespace NUMINAMATH_GPT_exists_m_n_l1069_106979

theorem exists_m_n (p : ℕ) (hp : p > 10) [hp_prime : Fact (Nat.Prime p)] :
  ∃ (m n : ℕ), m > 0 ∧ n > 0 ∧ m + n < p ∧ (5^m * 7^n - 1) % p = 0 :=
sorry

end NUMINAMATH_GPT_exists_m_n_l1069_106979


namespace NUMINAMATH_GPT_children_less_than_adults_l1069_106900

theorem children_less_than_adults (total_members : ℕ)
  (percent_adults : ℝ) (percent_teenagers : ℝ) (percent_children : ℝ) :
  total_members = 500 →
  percent_adults = 0.45 →
  percent_teenagers = 0.25 →
  percent_children = 1 - percent_adults - percent_teenagers →
  (percent_children * total_members) - (percent_adults * total_members) = -75 := 
by
  intros h_total h_adults h_teenagers h_children
  sorry

end NUMINAMATH_GPT_children_less_than_adults_l1069_106900


namespace NUMINAMATH_GPT_union_of_A_and_B_intersection_of_complement_A_and_B_l1069_106960

-- Define sets A and B
def A : Set ℝ := {x | 3 ≤ x ∧ x ≤ 7}
def B : Set ℝ := {x | 3 < 2 * x - 1 ∧ 2 * x - 1 < 19}

-- Define the universal set here, which encompass all real numbers
def universal_set : Set ℝ := {x | true}

-- Define the complement of A with respect to the real numbers
def C_R (S : Set ℝ) : Set ℝ := {x | x ∉ S}

-- Prove that A ∪ B is {x | 2 < x < 10}
theorem union_of_A_and_B : A ∪ B = {x | 2 < x ∧ x < 10} := by
  sorry

-- Prove that (C_R A) ∩ B is {x | 2 < x < 3 ∨ 7 < x < 10}
theorem intersection_of_complement_A_and_B : (C_R A) ∪ B = {x | (2 < x ∧ x < 3) ∨ (7 < x ∧ x < 10)} := by
  sorry

end NUMINAMATH_GPT_union_of_A_and_B_intersection_of_complement_A_and_B_l1069_106960


namespace NUMINAMATH_GPT_unique_integer_cube_triple_l1069_106943

theorem unique_integer_cube_triple (x : ℤ) (h : x^3 < 3 * x) : x = 1 := 
sorry

end NUMINAMATH_GPT_unique_integer_cube_triple_l1069_106943


namespace NUMINAMATH_GPT_mixed_fruit_juice_litres_opened_l1069_106944

theorem mixed_fruit_juice_litres_opened (cocktail_cost_per_litre : ℝ)
  (mixed_juice_cost_per_litre : ℝ) (acai_cost_per_litre : ℝ)
  (acai_litres_added : ℝ) (total_mixed_juice_opened : ℝ) :
  cocktail_cost_per_litre = 1399.45 ∧
  mixed_juice_cost_per_litre = 262.85 ∧
  acai_cost_per_litre = 3104.35 ∧
  acai_litres_added = 23.333333333333336 ∧
  (mixed_juice_cost_per_litre * total_mixed_juice_opened + 
  acai_cost_per_litre * acai_litres_added = 
  cocktail_cost_per_litre * (total_mixed_juice_opened + acai_litres_added)) →
  total_mixed_juice_opened = 35 :=
sorry

end NUMINAMATH_GPT_mixed_fruit_juice_litres_opened_l1069_106944


namespace NUMINAMATH_GPT_acute_triangle_side_range_l1069_106908

theorem acute_triangle_side_range {x : ℝ} (h : ∀ a b c : ℝ, a^2 + b^2 > c^2 ∧ a^2 + c^2 > b^2 ∧ b^2 + c^2 > a^2) :
  2 < 4 ∧ 4 < x → (2 * Real.sqrt 3 < x ∧ x < 2 * Real.sqrt 5) :=
  sorry

end NUMINAMATH_GPT_acute_triangle_side_range_l1069_106908


namespace NUMINAMATH_GPT_find_y_l1069_106995

theorem find_y (x y : ℤ) (h1 : x + 2 * y = 100) (h2 : x = 50) : y = 25 :=
by
  sorry

end NUMINAMATH_GPT_find_y_l1069_106995


namespace NUMINAMATH_GPT_correct_option_C_l1069_106947

theorem correct_option_C (a b c : ℝ) : 2 * a^2 * b * c - a^2 * b * c = a^2 * b * c := 
sorry

end NUMINAMATH_GPT_correct_option_C_l1069_106947


namespace NUMINAMATH_GPT_value_of_expression_l1069_106951

theorem value_of_expression
  (a b x y : ℝ)
  (h1 : a + b = 0)
  (h2 : x * y = 1) : 
  2 * (a + b) + (7 / 4) * (x * y) = 7 / 4 := 
sorry

end NUMINAMATH_GPT_value_of_expression_l1069_106951


namespace NUMINAMATH_GPT_ab_value_l1069_106926

noncomputable def func (x : ℝ) (a b : ℝ) : ℝ := 4 * x ^ 3 - a * x ^ 2 - 2 * b * x + 2

theorem ab_value 
  (a b : ℝ)
  (h_max : func 1 a b = -3)
  (h_deriv : (12 - 2 * a - 2 * b) = 0) :
  a * b = 9 :=
by
  sorry

end NUMINAMATH_GPT_ab_value_l1069_106926


namespace NUMINAMATH_GPT_problem_solution_l1069_106946

variables {a b c : ℝ}

theorem problem_solution (h : a + b + c = 0) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  (a^3 * b^3 / ((a^3 - b^2 * c) * (b^3 - a^2 * c)) +
  a^3 * c^3 / ((a^3 - b^2 * c) * (c^3 - a^2 * b)) +
  b^3 * c^3 / ((b^3 - a^2 * c) * (c^3 - a^2 * b))) = 1 :=
sorry

end NUMINAMATH_GPT_problem_solution_l1069_106946


namespace NUMINAMATH_GPT_frank_reading_days_l1069_106962

-- Define the parameters
def pages_weekdays : ℚ := 5.7
def pages_weekends : ℚ := 9.5
def total_pages : ℚ := 576
def pages_per_week : ℚ := (pages_weekdays * 5) + (pages_weekends * 2)

-- Define the property to be proved
theorem frank_reading_days : 
  (total_pages / pages_per_week).floor * 7 + 
  (total_pages - (total_pages / pages_per_week).floor * pages_per_week) / pages_weekdays 
  = 85 := 
  by
    sorry

end NUMINAMATH_GPT_frank_reading_days_l1069_106962


namespace NUMINAMATH_GPT_incorrect_regression_intercept_l1069_106976

theorem incorrect_regression_intercept (points : List (ℕ × ℝ)) (h_points : points = [(1, 0.5), (2, 0.8), (3, 1.0), (4, 1.2), (5, 1.5)]) :
  ¬ (∃ (a : ℝ), a = 0.26 ∧ ∀ x : ℕ, x ∈ ([1, 2, 3, 4, 5] : List ℕ) → (∃ y : ℝ, y = 0.24 * x + a)) := sorry

end NUMINAMATH_GPT_incorrect_regression_intercept_l1069_106976


namespace NUMINAMATH_GPT_polynomial_no_positive_real_roots_l1069_106991

theorem polynomial_no_positive_real_roots : 
  ¬ ∃ x : ℝ, x > 0 ∧ x^3 + 6 * x^2 + 11 * x + 6 = 0 :=
sorry

end NUMINAMATH_GPT_polynomial_no_positive_real_roots_l1069_106991


namespace NUMINAMATH_GPT_find_other_number_l1069_106988

theorem find_other_number (A B : ℕ) (H1 : Nat.lcm A B = 2310) (H2 : Nat.gcd A B = 30) (H3 : A = 770) : B = 90 :=
  by
  sorry

end NUMINAMATH_GPT_find_other_number_l1069_106988


namespace NUMINAMATH_GPT_min_fraction_expression_l1069_106975

theorem min_fraction_expression (a b : ℝ) (ha : 0 < a) (hb : 0 < b) 
  (h : 1 / a + 1 / b = 1) : 
  ∃ a b, ∃ (h : 1 / a + 1 / b = 1), a > 1 ∧ b > 1 ∧ (1 / (a - 1) + 4 / (b - 1)) = 4 := 
by 
  sorry

end NUMINAMATH_GPT_min_fraction_expression_l1069_106975


namespace NUMINAMATH_GPT_difference_between_relations_l1069_106963

-- Definitions based on conditions
def functional_relationship 
  (f : α → β) (x : α) (y : β) : Prop :=
  f x = y

def correlation_relationship (X Y : Type) : Prop :=
  ∃ (X_rand : X → ℝ) (Y_rand : Y → ℝ), 
    ∀ (x : X), ∃ (y : Y), X_rand x ≠ Y_rand y

-- Theorem stating the problem
theorem difference_between_relations :
  (∀ (f : α → β) (x : α) (y : β), functional_relationship f x y) ∧ 
  (∀ (X Y : Type), correlation_relationship X Y) :=
sorry

end NUMINAMATH_GPT_difference_between_relations_l1069_106963


namespace NUMINAMATH_GPT_geometric_sequence_solution_l1069_106997

-- Assume we have a type for real numbers
variable {R : Type} [LinearOrderedField R]

theorem geometric_sequence_solution (a b c : R)
  (h1 : -1 ≠ 0) (h2 : a ≠ 0) (h3 : b ≠ 0) (h4 : c ≠ 0) (h5 : -9 ≠ 0)
  (h : ∃ r : R, r ≠ 0 ∧ (a = r * -1) ∧ (b = r * a) ∧ (c = r * b) ∧ (-9 = r * c)) :
  b = -3 ∧ a * c = 9 := by
  sorry

end NUMINAMATH_GPT_geometric_sequence_solution_l1069_106997


namespace NUMINAMATH_GPT_trigonometric_problem_l1069_106930

theorem trigonometric_problem (θ : ℝ) (h : Real.tan θ = 2) :
  (3 * Real.sin θ - 2 * Real.cos θ) / (Real.sin θ + 3 * Real.cos θ) = 4 / 5 := by
  sorry

end NUMINAMATH_GPT_trigonometric_problem_l1069_106930
