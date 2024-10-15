import Mathlib

namespace NUMINAMATH_GPT_no_primes_divisible_by_60_l1797_179735

theorem no_primes_divisible_by_60 (p : ℕ) (prime_p : Nat.Prime p) : ¬ (60 ∣ p) :=
by
  sorry

end NUMINAMATH_GPT_no_primes_divisible_by_60_l1797_179735


namespace NUMINAMATH_GPT_parabola_focus_distance_l1797_179791

theorem parabola_focus_distance (M : ℝ × ℝ) (h1 : (M.2)^2 = 4 * M.1) (h2 : dist M (1, 0) = 4) : M.1 = 3 :=
sorry

end NUMINAMATH_GPT_parabola_focus_distance_l1797_179791


namespace NUMINAMATH_GPT_parabola_directrix_l1797_179754

theorem parabola_directrix (vertex_origin : ∀ (x y : ℝ), x = 0 ∧ y = 0)
    (directrix : ∀ (y : ℝ), y = 4) : ∃ p, x^2 = -2 * p * y ∧ p = 8 ∧ x^2 = -16 * y := 
sorry

end NUMINAMATH_GPT_parabola_directrix_l1797_179754


namespace NUMINAMATH_GPT_unique_ordered_pair_satisfies_equation_l1797_179798

theorem unique_ordered_pair_satisfies_equation :
  ∃! (m n : ℕ), 0 < m ∧ 0 < n ∧ (6 / m + 3 / n + 1 / (m * n) = 1) :=
by
  sorry

end NUMINAMATH_GPT_unique_ordered_pair_satisfies_equation_l1797_179798


namespace NUMINAMATH_GPT_solveRealInequality_l1797_179782

theorem solveRealInequality (x : ℝ) (hx : 0 < x) : x * Real.sqrt (18 - x) + Real.sqrt (18 * x - x^3) ≥ 18 → x = 3 :=
by
  sorry -- proof to be filled in

end NUMINAMATH_GPT_solveRealInequality_l1797_179782


namespace NUMINAMATH_GPT_square_root_condition_l1797_179786

-- Define the condition under which the square root of an expression is defined
def is_square_root_defined (x : ℝ) : Prop := (x + 3) ≥ 0

-- Prove that the condition for the square root of x + 3 to be defined is x ≥ -3
theorem square_root_condition (x : ℝ) : is_square_root_defined x ↔ x ≥ -3 := 
sorry

end NUMINAMATH_GPT_square_root_condition_l1797_179786


namespace NUMINAMATH_GPT_joey_hourly_wage_l1797_179738

def sneakers_cost : ℕ := 92
def mowing_earnings (lawns : ℕ) (rate : ℕ) : ℕ := lawns * rate
def selling_earnings (figures : ℕ) (rate : ℕ) : ℕ := figures * rate
def total_additional_earnings (mowing : ℕ) (selling : ℕ) : ℕ := mowing + selling
def remaining_amount (total_cost : ℕ) (earned : ℕ) : ℕ := total_cost - earned
def hourly_wage (remaining : ℕ) (hours : ℕ) : ℕ := remaining / hours

theorem joey_hourly_wage :
  let total_mowing := mowing_earnings 3 8
  let total_selling := selling_earnings 2 9
  let total_earned := total_additional_earnings total_mowing total_selling
  let remaining := remaining_amount sneakers_cost total_earned
  hourly_wage remaining 10 = 5 :=
by
  sorry

end NUMINAMATH_GPT_joey_hourly_wage_l1797_179738


namespace NUMINAMATH_GPT_jordan_weight_after_exercise_l1797_179789

theorem jordan_weight_after_exercise :
  let initial_weight := 250
  let weight_loss_4_weeks := 3 * 4
  let weight_loss_8_weeks := 2 * 8
  let total_weight_loss := weight_loss_4_weeks + weight_loss_8_weeks
  let current_weight := initial_weight - total_weight_loss
  current_weight = 222 :=
by
  let initial_weight := 250
  let weight_loss_4_weeks := 3 * 4
  let weight_loss_8_weeks := 2 * 8
  let total_weight_loss := weight_loss_4_weeks + weight_loss_8_weeks
  let current_weight := initial_weight - total_weight_loss
  show current_weight = 222
  sorry

end NUMINAMATH_GPT_jordan_weight_after_exercise_l1797_179789


namespace NUMINAMATH_GPT_cells_after_9_days_l1797_179714

noncomputable def remaining_cells (initial : ℕ) (days : ℕ) : ℕ :=
  let rec divide_and_decay (cells: ℕ) (remaining_days: ℕ) : ℕ :=
    if remaining_days = 0 then cells
    else
      let divided := cells * 2
      let decayed := (divided * 9) / 10
      divide_and_decay decayed (remaining_days - 3)
  divide_and_decay initial days

theorem cells_after_9_days :
  remaining_cells 5 9 = 28 := by
  sorry

end NUMINAMATH_GPT_cells_after_9_days_l1797_179714


namespace NUMINAMATH_GPT_wire_length_approx_is_correct_l1797_179761

noncomputable def S : ℝ := 5.999999999999998
noncomputable def L : ℝ := (5 / 2) * S
noncomputable def W : ℝ := S + L

theorem wire_length_approx_is_correct : abs (W - 21) < 1e-16 := by
  sorry

end NUMINAMATH_GPT_wire_length_approx_is_correct_l1797_179761


namespace NUMINAMATH_GPT_ratio_wx_l1797_179726

theorem ratio_wx (w x y : ℚ) (h1 : w / y = 3 / 4) (h2 : (x + y) / y = 13 / 4) : w / x = 1 / 3 :=
  sorry

end NUMINAMATH_GPT_ratio_wx_l1797_179726


namespace NUMINAMATH_GPT_base8_subtraction_correct_l1797_179711

def base8_sub (a b : Nat) : Nat := sorry  -- function to perform base 8 subtraction

theorem base8_subtraction_correct :
  base8_sub 0o126 0o45 = 0o41 := sorry

end NUMINAMATH_GPT_base8_subtraction_correct_l1797_179711


namespace NUMINAMATH_GPT_circle_equation_tangent_to_line_l1797_179743

def circle_center : (ℝ × ℝ) := (3, -1)
def tangent_line (x y : ℝ) : Prop := 3 * x + 4 * y = 0

/-- The equation of the circle with center at (3, -1) and tangent to the line 3x + 4y = 0 is (x - 3)^2 + (y + 1)^2 = 1 -/
theorem circle_equation_tangent_to_line : 
  ∃ r, ∀ x y: ℝ, ((x - 3)^2 + (y + 1)^2 = r^2) ∧ (∀ (cx cy: ℝ), cx = 3 → cy = -1 → (tangent_line cx cy → r = 1)) :=
by
  sorry

end NUMINAMATH_GPT_circle_equation_tangent_to_line_l1797_179743


namespace NUMINAMATH_GPT_find_x_l1797_179787

theorem find_x (x : ℕ) (h : x * 5^4 = 75625) : x = 121 :=
by
  sorry

end NUMINAMATH_GPT_find_x_l1797_179787


namespace NUMINAMATH_GPT_interest_received_l1797_179793

theorem interest_received
  (total_investment : ℝ)
  (part_invested_6 : ℝ)
  (rate_6 : ℝ)
  (rate_9 : ℝ) :
  part_invested_6 = 7200 →
  rate_6 = 0.06 →
  rate_9 = 0.09 →
  total_investment = 10000 →
  (total_investment - part_invested_6) * rate_9 + part_invested_6 * rate_6 = 684 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  sorry

end NUMINAMATH_GPT_interest_received_l1797_179793


namespace NUMINAMATH_GPT_cos_4_arccos_fraction_l1797_179709

theorem cos_4_arccos_fraction :
  (Real.cos (4 * Real.arccos (2 / 5))) = (-47 / 625) :=
by
  sorry

end NUMINAMATH_GPT_cos_4_arccos_fraction_l1797_179709


namespace NUMINAMATH_GPT_smallest_perimeter_of_scalene_triangle_with_conditions_l1797_179777

def is_odd_prime (n : ℕ) : Prop :=
  Nat.Prime n ∧ n % 2 = 1

-- Define a scalene triangle
structure ScaleneTriangle :=
  (a b c : ℕ)
  (a_ne_b : a ≠ b)
  (a_ne_c : a ≠ c)
  (b_ne_c : b ≠ c)
  (triangle_inequality1 : a + b > c)
  (triangle_inequality2 : a + c > b)
  (triangle_inequality3 : b + c > a)

-- Define the problem conditions
def problem_conditions (a b c : ℕ) : Prop :=
  is_odd_prime a ∧ is_odd_prime b ∧ is_odd_prime c ∧
  a < b ∧ b < c ∧
  Nat.Prime (a + b + c) ∧
  (∃ (t : ScaleneTriangle), t.a = a ∧ t.b = b ∧ t.c = c)

-- Define the proposition
theorem smallest_perimeter_of_scalene_triangle_with_conditions :
  ∃ (a b c : ℕ), problem_conditions a b c ∧ a + b + c = 23 :=
sorry

end NUMINAMATH_GPT_smallest_perimeter_of_scalene_triangle_with_conditions_l1797_179777


namespace NUMINAMATH_GPT_missing_digit_l1797_179705

theorem missing_digit (x : ℕ) (h1 : x ≥ 0) (h2 : x ≤ 9) : 
  (if x ≥ 2 then 9 * 1000 + x * 100 + 2 * 10 + 1 else 9 * 100 + 2 * 10 + x * 1) - (1 * 1000 + 2 * 100 + 9 * 10 + x) = 8262 → x = 5 :=
by 
  sorry

end NUMINAMATH_GPT_missing_digit_l1797_179705


namespace NUMINAMATH_GPT_find_abc_l1797_179774

theorem find_abc (a b c : ℤ) 
  (h₁ : a^4 - 2 * b^2 = a)
  (h₂ : b^4 - 2 * c^2 = b)
  (h₃ : c^4 - 2 * a^2 = c)
  (h₄ : a + b + c = -3) : 
  a = -1 ∧ b = -1 ∧ c = -1 := 
sorry

end NUMINAMATH_GPT_find_abc_l1797_179774


namespace NUMINAMATH_GPT_fraction_a_over_b_l1797_179732

theorem fraction_a_over_b (x y a b : ℝ) (hb : b ≠ 0) (h1 : 4 * x - 2 * y = a) (h2 : 9 * y - 18 * x = b) :
  a / b = -2 / 9 :=
by
  sorry

end NUMINAMATH_GPT_fraction_a_over_b_l1797_179732


namespace NUMINAMATH_GPT_problem1_problem2_problem3_l1797_179746

-- Problem 1
theorem problem1 (A B : Set α) (x : α) : x ∈ A ∪ B → x ∈ A ∨ x ∈ B :=
by sorry

-- Problem 2
theorem problem2 (A B : Set α) (x : α) : x ∈ A ∩ B → x ∈ A ∧ x ∈ B :=
by sorry

-- Problem 3
theorem problem3 (a b : ℝ) : a > 0 ∧ b > 0 → a * b > 0 :=
by sorry

end NUMINAMATH_GPT_problem1_problem2_problem3_l1797_179746


namespace NUMINAMATH_GPT_bob_corn_calc_l1797_179747

noncomputable def bob_corn_left (initial_bushels : ℕ) (ears_per_bushel : ℕ) (bushels_taken_by_terry : ℕ) (bushels_taken_by_jerry : ℕ) (bushels_taken_by_linda : ℕ) (ears_taken_by_stacy : ℕ) : ℕ :=
  let initial_ears := initial_bushels * ears_per_bushel
  let ears_given_away := (bushels_taken_by_terry + bushels_taken_by_jerry + bushels_taken_by_linda) * ears_per_bushel + ears_taken_by_stacy
  initial_ears - ears_given_away

theorem bob_corn_calc :
  bob_corn_left 50 14 8 3 12 21 = 357 :=
by
  sorry

end NUMINAMATH_GPT_bob_corn_calc_l1797_179747


namespace NUMINAMATH_GPT_quadratic_equal_roots_relation_l1797_179752

theorem quadratic_equal_roots_relation (a b c : ℝ) (h₁ : b ≠ c) 
  (h₂ : ∀ x : ℝ, (b - c) * x^2 + (a - b) * x + (c - a) = 0 → 
          (a - b)^2 - 4 * (b - c) * (c - a) = 0) : 
  c = (a + b) / 2 := sorry

end NUMINAMATH_GPT_quadratic_equal_roots_relation_l1797_179752


namespace NUMINAMATH_GPT_total_tissues_brought_l1797_179739

def number_of_students (group1 group2 group3 : Nat) : Nat :=
  group1 + group2 + group3

def number_of_tissues_per_student (tissues_per_box : Nat) (total_students : Nat) : Nat :=
  tissues_per_box * total_students

theorem total_tissues_brought :
  let group1 := 9
  let group2 := 10
  let group3 := 11
  let tissues_per_box := 40
  let total_students := number_of_students group1 group2 group3
  number_of_tissues_per_student tissues_per_box total_students = 1200 :=
by
  sorry

end NUMINAMATH_GPT_total_tissues_brought_l1797_179739


namespace NUMINAMATH_GPT_probability_diff_topics_l1797_179734

theorem probability_diff_topics
  (num_topics : ℕ)
  (num_combinations : ℕ)
  (num_different_combinations : ℕ)
  (h1 : num_topics = 6)
  (h2 : num_combinations = num_topics * num_topics)
  (h3 : num_combinations = 36)
  (h4 : num_different_combinations = num_topics * (num_topics - 1))
  (h5 : num_different_combinations = 30) :
  (num_different_combinations / num_combinations) = 5 / 6 := 
by 
  sorry

end NUMINAMATH_GPT_probability_diff_topics_l1797_179734


namespace NUMINAMATH_GPT_parallel_lines_slope_condition_l1797_179729

-- Define the first line equation and the slope
def line1 (x : ℝ) : ℝ := 6 * x + 5
def slope1 : ℝ := 6

-- Define the second line equation and the slope
def line2 (x c : ℝ) : ℝ := (3 * c) * x - 7
def slope2 (c : ℝ) : ℝ := 3 * c

-- Theorem stating that if the lines are parallel, the value of c is 2
theorem parallel_lines_slope_condition (c : ℝ) : 
  (slope1 = slope2 c) → c = 2 := 
  by
    sorry -- Proof

end NUMINAMATH_GPT_parallel_lines_slope_condition_l1797_179729


namespace NUMINAMATH_GPT_intersection_cardinality_l1797_179731

variable {a b : ℝ}
variable {f : ℝ → ℝ}

theorem intersection_cardinality {a b : ℝ} {f : ℝ → ℝ} :
  (∃! y, (0, y) ∈ ({ (x, y) | y = f x ∧ a ≤ x ∧ x ≤ b } ∩ { (x, y) | x = 0 })) ∨
  ¬ (∃ y, (0, y) ∈ { (x, y) | y = f x ∧ a ≤ x ∧ x ≤ b }) :=
by
  sorry

end NUMINAMATH_GPT_intersection_cardinality_l1797_179731


namespace NUMINAMATH_GPT_arrangement_count_l1797_179763

def numArrangements : Nat := 15000

theorem arrangement_count (students events : ℕ) (nA nB : ℕ) 
  (A_ne_B : nA ≠ nB) 
  (all_students : students = 7) 
  (all_events : events = 5) 
  (one_event_per_student : ∀ (e : ℕ), e < events → ∃ s, s < students ∧ (∀ (s' : ℕ), s' < students → s' ≠ s → e ≠ s')) :
  numArrangements = 15000 := 
sorry

end NUMINAMATH_GPT_arrangement_count_l1797_179763


namespace NUMINAMATH_GPT_solve_remainder_l1797_179785

theorem solve_remainder (y : ℤ) 
  (hc1 : y + 4 ≡ 9 [ZMOD 3^3])
  (hc2 : y + 4 ≡ 16 [ZMOD 5^3])
  (hc3 : y + 4 ≡ 36 [ZMOD 7^3]) : 
  y ≡ 32 [ZMOD 105] :=
by
  sorry

end NUMINAMATH_GPT_solve_remainder_l1797_179785


namespace NUMINAMATH_GPT_weight_of_substance_l1797_179778

variable (k W1 W2 : ℝ)

theorem weight_of_substance (h1 : ∃ (k : ℝ), ∀ (V W : ℝ), V = k * W)
  (h2 : 48 = k * W1) (h3 : 36 = k * 84) : 
  (∃ (W2 : ℝ), 48 = (36 / 84) * W2) → W2 = 112 := 
by
  sorry

end NUMINAMATH_GPT_weight_of_substance_l1797_179778


namespace NUMINAMATH_GPT_closest_fraction_l1797_179733

theorem closest_fraction (won : ℚ) (options : List ℚ) (closest : ℚ) 
  (h_won : won = 25 / 120) 
  (h_options : options = [1 / 4, 1 / 5, 1 / 6, 1 / 7, 1 / 8]) 
  (h_closest : closest = 1 / 5) :
  ∃ x ∈ options, abs (won - x) = abs (won - closest) := 
sorry

end NUMINAMATH_GPT_closest_fraction_l1797_179733


namespace NUMINAMATH_GPT_find_sum_of_a_and_b_l1797_179781

theorem find_sum_of_a_and_b (a b : ℝ) 
  (h1 : ∀ x : ℝ, (abs (x^2 - 2 * a * x + b) = 8) → (x = a ∨ x = a + 4 ∨ x = a - 4))
  (h2 : a^2 + (a - 4)^2 = (a + 4)^2) :
  a + b = 264 :=
by
  sorry

end NUMINAMATH_GPT_find_sum_of_a_and_b_l1797_179781


namespace NUMINAMATH_GPT_scientific_notation_of_360_billion_l1797_179744

def number_in_scientific_notation (n : ℕ) : String :=
  match n with
  | 360000000000 => "3.6 × 10^11"
  | _ => "Unknown"

theorem scientific_notation_of_360_billion : 
  number_in_scientific_notation 360000000000 = "3.6 × 10^11" :=
by
  -- insert proof steps here
  sorry

end NUMINAMATH_GPT_scientific_notation_of_360_billion_l1797_179744


namespace NUMINAMATH_GPT_diagonal_length_l1797_179771

noncomputable def convertHectaresToSquareMeters (hectares : ℝ) : ℝ :=
  hectares * 10000

noncomputable def sideLength (areaSqMeters : ℝ) : ℝ :=
  Real.sqrt areaSqMeters

noncomputable def diagonal (side : ℝ) : ℝ :=
  side * Real.sqrt 2

theorem diagonal_length (area : ℝ) (h : area = 1 / 2) :
  let areaSqMeters := convertHectaresToSquareMeters area
  let side := sideLength areaSqMeters
  let diag := diagonal side
  abs (diag - 100) < 1 :=
by
  sorry

end NUMINAMATH_GPT_diagonal_length_l1797_179771


namespace NUMINAMATH_GPT_intersection_unique_point_x_coordinate_l1797_179716

theorem intersection_unique_point_x_coordinate (a b : ℝ) (h : a ≠ b) : 
  (∃ x y : ℝ, y = x^2 + 2*a*x + 6*b ∧ y = x^2 + 2*b*x + 6*a) → ∃ x : ℝ, x = 3 :=
by
  sorry

end NUMINAMATH_GPT_intersection_unique_point_x_coordinate_l1797_179716


namespace NUMINAMATH_GPT_tens_digit_6_pow_18_l1797_179766

/--
To find the tens digit of \(6^{18}\), we look at the powers of 6 and determine their tens digits. 
We note the pattern in tens digits (3, 1, 9, 7, 6) which repeats every 5 powers. 
Since \(6^{18}\) corresponds to the 3rd position in the repeating cycle, we claim the tens digit is 1.
--/
theorem tens_digit_6_pow_18 : (6^18 / 10) % 10 = 1 :=
by sorry

end NUMINAMATH_GPT_tens_digit_6_pow_18_l1797_179766


namespace NUMINAMATH_GPT_bus_ride_difference_l1797_179753

theorem bus_ride_difference (vince_bus_length zachary_bus_length : Real)
    (h_vince : vince_bus_length = 0.62)
    (h_zachary : zachary_bus_length = 0.5) :
    vince_bus_length - zachary_bus_length = 0.12 :=
by
  sorry

end NUMINAMATH_GPT_bus_ride_difference_l1797_179753


namespace NUMINAMATH_GPT_mingi_math_test_total_pages_l1797_179773

theorem mingi_math_test_total_pages (first_page last_page : Nat) (h_first_page : first_page = 8) (h_last_page : last_page = 21) : first_page <= last_page -> ((last_page - first_page + 1) = 14) :=
by
  sorry

end NUMINAMATH_GPT_mingi_math_test_total_pages_l1797_179773


namespace NUMINAMATH_GPT_insufficient_info_for_pumpkins_l1797_179783

variable (jason_watermelons : ℕ) (sandy_watermelons : ℕ) (total_watermelons : ℕ)

theorem insufficient_info_for_pumpkins (h1 : jason_watermelons = 37)
  (h2 : sandy_watermelons = 11)
  (h3 : jason_watermelons + sandy_watermelons = total_watermelons)
  (h4 : total_watermelons = 48) : 
  ¬∃ (jason_pumpkins : ℕ), true
:= by
  sorry

end NUMINAMATH_GPT_insufficient_info_for_pumpkins_l1797_179783


namespace NUMINAMATH_GPT_weight_difference_l1797_179715

def weight_chemistry : ℝ := 7.12
def weight_geometry : ℝ := 0.62

theorem weight_difference : weight_chemistry - weight_geometry = 6.50 :=
by
  sorry

end NUMINAMATH_GPT_weight_difference_l1797_179715


namespace NUMINAMATH_GPT_final_result_is_8_l1797_179768

theorem final_result_is_8 (n : ℕ) (h1 : n = 2976) (h2 : (n / 12) - 240 = 8) : (n / 12) - 240 = 8 :=
by {
  -- Proof steps would go here
  sorry
}

end NUMINAMATH_GPT_final_result_is_8_l1797_179768


namespace NUMINAMATH_GPT_profit_share_of_B_l1797_179788

theorem profit_share_of_B (P : ℝ) (A_share B_share C_share : ℝ) :
  let A_initial := 8000
  let B_initial := 10000
  let C_initial := 12000
  let total_capital := A_initial + B_initial + C_initial
  let investment_ratio_A := A_initial / total_capital
  let investment_ratio_B := B_initial / total_capital
  let investment_ratio_C := C_initial / total_capital
  let total_profit := 4200
  let diff_AC := 560
  A_share = (investment_ratio_A * total_profit) →
  B_share = (investment_ratio_B * total_profit) →
  C_share = (investment_ratio_C * total_profit) →
  C_share - A_share = diff_AC →
  B_share = 1400 :=
by
  intros
  sorry

end NUMINAMATH_GPT_profit_share_of_B_l1797_179788


namespace NUMINAMATH_GPT_distance_A_beats_B_l1797_179710

theorem distance_A_beats_B 
  (A_time : ℝ) (A_distance : ℝ) (B_time : ℝ) (B_distance : ℝ)
  (hA : A_distance = 128) (hA_time : A_time = 28)
  (hB : B_distance = 128) (hB_time : B_time = 32) :
  (A_distance - (B_distance * (A_time / B_time))) = 16 :=
by
  sorry

end NUMINAMATH_GPT_distance_A_beats_B_l1797_179710


namespace NUMINAMATH_GPT_range_of_c_l1797_179760

def P (c : ℝ) : Prop := ∀ x1 x2 : ℝ, x1 < x2 → (c ^ x1) > (c ^ x2)
def q (c : ℝ) : Prop := ∀ x : ℝ, x > (1 / 2) → (2 * c * x - c) > 0

theorem range_of_c (c : ℝ) (h1 : c > 0) (h2 : c ≠ 1)
  (h3 : ¬ (P c ∧ q c)) (h4 : (P c ∨ q c)) :
  (1 / 2) < c ∧ c < 1 :=
by
  sorry

end NUMINAMATH_GPT_range_of_c_l1797_179760


namespace NUMINAMATH_GPT_shop_owner_percentage_profit_l1797_179794

theorem shop_owner_percentage_profit :
  let cost_price_per_kg := 100
  let buy_cheat_percent := 18.5 / 100
  let sell_cheat_percent := 22.3 / 100
  let amount_bought := 1 / (1 + buy_cheat_percent)
  let amount_sold := 1 - sell_cheat_percent
  let effective_cost_price := cost_price_per_kg * amount_sold / amount_bought
  let selling_price := cost_price_per_kg
  let profit := selling_price - effective_cost_price
  let percentage_profit := (profit / effective_cost_price) * 100
  percentage_profit = 52.52 :=
by
  sorry

end NUMINAMATH_GPT_shop_owner_percentage_profit_l1797_179794


namespace NUMINAMATH_GPT_circle_cartesian_line_circle_intersect_l1797_179769

noncomputable def L_parametric (t : ℝ) : ℝ × ℝ :=
  (t, 1 + 2 * t)

noncomputable def C_polar (θ : ℝ) : ℝ :=
  2 * Real.sqrt 2 * Real.sin (θ + Real.pi / 4)

def L_cartesian (x y : ℝ) : Prop :=
  y = 2 * x + 1

def C_cartesian (x y : ℝ) : Prop :=
  (x - 1)^2 + (y - 1)^2 = 2

theorem circle_cartesian :
  ∀ x y : ℝ, C_polar x = y ↔ C_cartesian x y :=
sorry

theorem line_circle_intersect (x y : ℝ) :
  L_cartesian x y → C_cartesian x y → True :=
sorry

end NUMINAMATH_GPT_circle_cartesian_line_circle_intersect_l1797_179769


namespace NUMINAMATH_GPT_total_weight_correct_total_money_earned_correct_l1797_179775

variable (records : List Int) (std_weight : Int)

-- Conditions
def deviation_sum (records : List Int) : Int := records.foldl (· + ·) 0

def batch_weight (std_weight : Int) (n : Int) (deviation_sum : Int) : Int :=
  deviation_sum + std_weight * n

def first_day_sales (total_weight : Int) (price_per_kg : Int) : Int :=
  price_per_kg * (total_weight / 2)

def second_day_sales (total_weight : Int) (first_day_sales_weight : Int) (discounted_price_per_kg : Int) : Int :=
  discounted_price_per_kg * (total_weight - first_day_sales_weight)

def total_earnings (first_day_sales : Int) (second_day_sales : Int) : Int :=
  first_day_sales + second_day_sales

-- Proof statements
theorem total_weight_correct : 
  deviation_sum records = 4 ∧ std_weight = 30 ∧ records.length = 8 → 
  batch_weight std_weight records.length (deviation_sum records) = 244 :=
by
  intro h
  sorry

theorem total_money_earned_correct :
  first_day_sales (batch_weight std_weight records.length (deviation_sum records)) 10 = 1220 ∧
  second_day_sales (batch_weight std_weight records.length (deviation_sum records)) (batch_weight std_weight records.length (deviation_sum records) / 2) (10 * 9 / 10) = 1098 →
  total_earnings 1220 1098 = 2318 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_total_weight_correct_total_money_earned_correct_l1797_179775


namespace NUMINAMATH_GPT_employee_saves_86_25_l1797_179724

def initial_purchase_price : ℝ := 500
def markup_rate : ℝ := 0.15
def employee_discount_rate : ℝ := 0.15

def retail_price : ℝ := initial_purchase_price * (1 + markup_rate)
def employee_discount_amount : ℝ := retail_price * employee_discount_rate
def employee_savings : ℝ := retail_price - (retail_price - employee_discount_amount)

theorem employee_saves_86_25 :
  employee_savings = 86.25 := 
sorry

end NUMINAMATH_GPT_employee_saves_86_25_l1797_179724


namespace NUMINAMATH_GPT_James_present_age_l1797_179745

variable (D J : ℕ)

theorem James_present_age 
  (h1 : D / J = 6 / 5)
  (h2 : D + 4 = 28) :
  J = 20 := 
by
  sorry

end NUMINAMATH_GPT_James_present_age_l1797_179745


namespace NUMINAMATH_GPT_largest_value_l1797_179703

-- Definition: Given the condition of a quadratic equation
def equation (a : ℚ) : Prop :=
  8 * a^2 + 6 * a + 2 = 0

-- Theorem: Prove the largest value of 3a + 2 is 5/4 given the condition
theorem largest_value (a : ℚ) (h : equation a) : 
  ∃ m, ∀ b, equation b → (3 * b + 2 ≤ m) ∧ (m = 5 / 4) :=
by
  sorry

end NUMINAMATH_GPT_largest_value_l1797_179703


namespace NUMINAMATH_GPT_find_number_l1797_179702

theorem find_number (num : ℝ) (x : ℝ) (h1 : x = 0.08999999999999998) (h2 : num / x = 0.1) : num = 0.008999999999999999 :=
by 
  sorry

end NUMINAMATH_GPT_find_number_l1797_179702


namespace NUMINAMATH_GPT_total_net_loss_l1797_179755

theorem total_net_loss 
  (P_x P_y : ℝ)
  (h1 : 1.2 * P_x = 25000)
  (h2 : 0.8 * P_y = 25000) :
  (25000 - P_x) - (P_y - 25000) = -2083.33 :=
by 
  sorry

end NUMINAMATH_GPT_total_net_loss_l1797_179755


namespace NUMINAMATH_GPT_car_y_speed_l1797_179756

noncomputable def carY_average_speed (vX : ℝ) (tY : ℝ) (d : ℝ) : ℝ :=
  d / tY

theorem car_y_speed (vX : ℝ := 35) (tY_min : ℝ := 72) (dX_after_Y : ℝ := 245) :
  carY_average_speed vX (dX_after_Y / vX) dX_after_Y = 35 := 
by
  sorry

end NUMINAMATH_GPT_car_y_speed_l1797_179756


namespace NUMINAMATH_GPT_power_function_solution_l1797_179770

theorem power_function_solution (f : ℝ → ℝ) (alpha : ℝ)
  (h₀ : ∀ x, f x = x ^ alpha)
  (h₁ : f (1 / 8) = 2) :
  f (-1 / 8) = -2 :=
sorry

end NUMINAMATH_GPT_power_function_solution_l1797_179770


namespace NUMINAMATH_GPT_find_base_k_l1797_179750

theorem find_base_k (k : ℕ) (hk : 0 < k) (h : 7/51 = (2 * k + 3) / (k^2 - 1)) : k = 16 :=
sorry

end NUMINAMATH_GPT_find_base_k_l1797_179750


namespace NUMINAMATH_GPT_triangle_solution_condition_l1797_179728

-- Definitions of segments
variables {A B D E : Type}
variables (c f g : Real)

-- Allow noncomputable definitions for geometric constraints
noncomputable def triangle_construction (c f g : Real) : String :=
  if c > f then "more than one solution"
  else if c = f then "exactly one solution"
  else "no solution"

-- The proof problem statement
theorem triangle_solution_condition (c f g : Real) :
  (c > f → triangle_construction c f g = "more than one solution") ∧
  (c = f → triangle_construction c f g = "exactly one solution") ∧
  (c < f → triangle_construction c f g = "no solution") :=
by
  sorry

end NUMINAMATH_GPT_triangle_solution_condition_l1797_179728


namespace NUMINAMATH_GPT_exists_p_q_for_integer_roots_l1797_179759

theorem exists_p_q_for_integer_roots : 
  ∃ (p q : ℤ), ∀ k (hk : k ∈ (Finset.range 10)), 
    ∃ (r1 r2 : ℤ), (r1 + r2 = -(p + k)) ∧ (r1 * r2 = (q + k)) :=
sorry

end NUMINAMATH_GPT_exists_p_q_for_integer_roots_l1797_179759


namespace NUMINAMATH_GPT_greatest_M_inequality_l1797_179719

theorem greatest_M_inequality :
  ∀ x y z : ℝ, x^4 + y^4 + z^4 + x * y * z * (x + y + z) ≥ (2/3) * (x * y + y * z + z * x)^2 :=
by
  sorry

end NUMINAMATH_GPT_greatest_M_inequality_l1797_179719


namespace NUMINAMATH_GPT_min_frac_a_n_over_n_l1797_179792

open Nat

def a : ℕ → ℕ
| 0     => 60
| (n+1) => a n + 2 * n

theorem min_frac_a_n_over_n : ∃ n : ℕ, n > 0 ∧ (a n / n = (29 / 2) ∧ ∀ m : ℕ, m > 0 → a m / m ≥ (29 / 2)) :=
by
  sorry

end NUMINAMATH_GPT_min_frac_a_n_over_n_l1797_179792


namespace NUMINAMATH_GPT_monthly_earnings_l1797_179740

-- Defining the initial conditions and known information
def current_worth : ℝ := 90
def months : ℕ := 5

-- Let I be the initial investment, and E be the earnings per month.

noncomputable def initial_investment (I : ℝ) := I * 3 = current_worth
noncomputable def earned_twice_initial (E : ℝ) (I : ℝ) := E * months = 2 * I

-- Proving the monthly earnings
theorem monthly_earnings (I E : ℝ) (h1 : initial_investment I) (h2 : earned_twice_initial E I) : E = 12 :=
sorry

end NUMINAMATH_GPT_monthly_earnings_l1797_179740


namespace NUMINAMATH_GPT_largest_number_with_four_digits_divisible_by_72_is_9936_l1797_179717

theorem largest_number_with_four_digits_divisible_by_72_is_9936 :
  ∃ n : ℕ, (n < 10000 ∧ n ≥ 1000) ∧ (72 ∣ n) ∧ (∀ m : ℕ, (m < 10000 ∧ m ≥ 1000) ∧ (72 ∣ m) → m ≤ n) :=
sorry

end NUMINAMATH_GPT_largest_number_with_four_digits_divisible_by_72_is_9936_l1797_179717


namespace NUMINAMATH_GPT_solution_set_l1797_179776

variable (f : ℝ → ℝ)
variable (f' : ℝ → ℝ) -- Function for the derivative of f

axiom f_deriv : ∀ x, f' x = (deriv f) x

axiom f_condition1 : ∀ x, f x > 1 - f' x
axiom f_condition2 : f 0 = 0
  
theorem solution_set (x : ℝ) : (e^x * f x > e^x - 1) ↔ (x > 0) := 
  sorry

end NUMINAMATH_GPT_solution_set_l1797_179776


namespace NUMINAMATH_GPT_baker_usual_pastries_l1797_179795

variable (P : ℕ)

theorem baker_usual_pastries
  (h1 : 2 * 14 + 4 * 25 - (2 * P + 4 * 10) = 48) : P = 20 :=
by
  sorry

end NUMINAMATH_GPT_baker_usual_pastries_l1797_179795


namespace NUMINAMATH_GPT_simplify_expression_l1797_179748

theorem simplify_expression (a : ℤ) (ha : a = -2) : 
  3 * a^2 + (a^2 + (5 * a^2 - 2 * a) - 3 * (a^2 - 3 * a)) = 10 := 
by 
  sorry

end NUMINAMATH_GPT_simplify_expression_l1797_179748


namespace NUMINAMATH_GPT_points_earned_l1797_179737

def each_enemy_points : ℕ := 3
def total_enemies : ℕ := 6
def defeated_enemies : ℕ := total_enemies - 2

theorem points_earned : defeated_enemies * each_enemy_points = 12 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_points_earned_l1797_179737


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_not_necessary_l1797_179712

theorem sufficient_but_not_necessary (x y : ℝ) (h : x < y ∧ y < 0) : x^2 > y^2 :=
by {
  -- a Lean 4 proof can be included here if desired
  sorry
}

theorem not_necessary (x y : ℝ) (h : x^2 > y^2) : ¬ (x < y ∧ y < 0) :=
by {
  -- a Lean 4 proof can be included here if desired
  sorry
}

end NUMINAMATH_GPT_sufficient_but_not_necessary_not_necessary_l1797_179712


namespace NUMINAMATH_GPT_cone_sector_volume_ratio_l1797_179720

theorem cone_sector_volume_ratio 
  (H R : ℝ) 
  (nonneg_H : 0 ≤ H) 
  (nonneg_R : 0 ≤ R) :
  let volume_original := (1/3) * π * R^2 * H
  let volume_sector   := (1/12) * π * R^2 * H
  volume_sector / volume_sector = 1 :=
  by
    sorry

end NUMINAMATH_GPT_cone_sector_volume_ratio_l1797_179720


namespace NUMINAMATH_GPT_percentage_decrease_increase_l1797_179799

theorem percentage_decrease_increase (x : ℝ) : 
  (1 - x / 100) * (1 + x / 100) = 0.75 ↔ x = 50 :=
by
  sorry

end NUMINAMATH_GPT_percentage_decrease_increase_l1797_179799


namespace NUMINAMATH_GPT_selling_price_l1797_179730

/-- 
Prove that the selling price (S) of an article with a cost price (C) of 180 sold at a 15% profit (P) is 207.
-/
theorem selling_price (C P S : ℝ) (hC : C = 180) (hP : P = 15) (hS : S = 207) :
  S = C + (P / 100 * C) :=
by
  -- here we rely on sorry to skip the proof details
  sorry

end NUMINAMATH_GPT_selling_price_l1797_179730


namespace NUMINAMATH_GPT_johns_percentage_increase_l1797_179790

def original_amount : ℕ := 60
def new_amount : ℕ := 84

def percentage_increase (original new : ℕ) := ((new - original : ℕ) * 100) / original 

theorem johns_percentage_increase : percentage_increase original_amount new_amount = 40 :=
by
  sorry

end NUMINAMATH_GPT_johns_percentage_increase_l1797_179790


namespace NUMINAMATH_GPT_max_min_values_l1797_179701

noncomputable def f (x : ℝ) : ℝ := 2 * x^3 - 3 * x^2 - 12 * x

theorem max_min_values : 
  ∃ (max_val min_val : ℝ), 
    max_val = 7 ∧ min_val = -20 ∧ 
    (∀ x ∈ Set.Icc (-2 : ℝ) 3, f x ≤ max_val) ∧ 
    (∀ x ∈ Set.Icc (-2 : ℝ) 3, min_val ≤ f x) := 
by
  sorry

end NUMINAMATH_GPT_max_min_values_l1797_179701


namespace NUMINAMATH_GPT_minimum_value_l1797_179758

noncomputable def function_y (x : ℝ) : ℝ := 3 * x^2 - 12 * x + 1450

theorem minimum_value : ∀ x : ℝ, function_y x ≥ 1438 :=
by 
  intro x
  sorry

end NUMINAMATH_GPT_minimum_value_l1797_179758


namespace NUMINAMATH_GPT_negation_false_l1797_179736

theorem negation_false (a b : ℝ) : ¬ ((a ≤ 1 ∨ b ≤ 1) → a + b ≤ 2) :=
sorry

end NUMINAMATH_GPT_negation_false_l1797_179736


namespace NUMINAMATH_GPT_student_scores_l1797_179797

def weighted_average (math history science geography : ℝ) : ℝ :=
  (math * 0.30) + (history * 0.30) + (science * 0.20) + (geography * 0.20)

theorem student_scores :
  ∀ (math history science geography : ℝ),
    math = 74 →
    history = 81 →
    science = geography + 5 →
    science ≥ 75 →
    weighted_average math history science geography = 80 →
    science = 86.25 ∧ geography = 81.25 :=
by
  intros math history science geography h_math h_history h_science h_min_sci h_avg
  sorry

end NUMINAMATH_GPT_student_scores_l1797_179797


namespace NUMINAMATH_GPT_expression_value_l1797_179713

theorem expression_value :
  (35 + 12) ^ 2 - (12 ^ 2 + 35 ^ 2 - 2 * 12 * 35) = 1680 :=
by
  sorry

end NUMINAMATH_GPT_expression_value_l1797_179713


namespace NUMINAMATH_GPT_find_x_squared_plus_y_squared_l1797_179718

variable (x y : ℝ)

theorem find_x_squared_plus_y_squared (h1 : y + 7 = (x - 3)^2) (h2 : x + 7 = (y - 3)^2) (h3 : x ≠ y) :
  x^2 + y^2 = 17 :=
by
  sorry  -- Proof to be provided

end NUMINAMATH_GPT_find_x_squared_plus_y_squared_l1797_179718


namespace NUMINAMATH_GPT_transformed_expression_value_l1797_179772

theorem transformed_expression_value :
  (240 / 80) * 60 / 40 + 10 = 14.5 :=
by
  sorry

end NUMINAMATH_GPT_transformed_expression_value_l1797_179772


namespace NUMINAMATH_GPT_at_least_one_A_or_B_selected_prob_l1797_179767

theorem at_least_one_A_or_B_selected_prob :
  let students := ['A', 'B', 'C', 'D']
  let total_pairs := 6
  let complementary_event_prob := 1 / total_pairs
  let at_least_one_A_or_B_prob := 1 - complementary_event_prob
  at_least_one_A_or_B_prob = 5 / 6 :=
by
  let students := ['A', 'B', 'C', 'D']
  let total_pairs := 6
  let complementary_event_prob := 1 / total_pairs
  let at_least_one_A_or_B_prob := 1 - complementary_event_prob
  sorry

end NUMINAMATH_GPT_at_least_one_A_or_B_selected_prob_l1797_179767


namespace NUMINAMATH_GPT_absolute_value_simplification_l1797_179751

theorem absolute_value_simplification (x : ℝ) (h : x > 3) : 
  |x - Real.sqrt ((x - 3)^2)| = 3 := 
by 
  sorry

end NUMINAMATH_GPT_absolute_value_simplification_l1797_179751


namespace NUMINAMATH_GPT_find_M_l1797_179706

theorem find_M : 
  ∃ M : ℚ, 
  (5 / 12) * (20 / (20 + M)) + (7 / 12) * (M / (20 + M)) = 0.62 ∧ 
  M = 610 / 1657 :=
by
  sorry

end NUMINAMATH_GPT_find_M_l1797_179706


namespace NUMINAMATH_GPT_eliana_refill_l1797_179742

theorem eliana_refill (total_spent cost_per_refill : ℕ) (h1 : total_spent = 63) (h2 : cost_per_refill = 21) : (total_spent / cost_per_refill) = 3 :=
sorry

end NUMINAMATH_GPT_eliana_refill_l1797_179742


namespace NUMINAMATH_GPT_negation_of_existence_statement_l1797_179707

theorem negation_of_existence_statement :
  (¬ ∃ x : ℝ, x^2 - 3 * x + 2 = 0) = ∀ x : ℝ, x^2 - 3 * x + 2 ≠ 0 :=
by
  sorry

end NUMINAMATH_GPT_negation_of_existence_statement_l1797_179707


namespace NUMINAMATH_GPT_numberOfPairsPaddlesSold_l1797_179708

def totalSalesPaddles : ℝ := 735
def avgPricePerPairPaddles : ℝ := 9.8

theorem numberOfPairsPaddlesSold :
  totalSalesPaddles / avgPricePerPairPaddles = 75 := 
by
  sorry

end NUMINAMATH_GPT_numberOfPairsPaddlesSold_l1797_179708


namespace NUMINAMATH_GPT_num_zeros_in_product_l1797_179700

theorem num_zeros_in_product : ∀ (a b : ℕ), (a = 125) → (b = 960) → (∃ n, a * b = n * 10^4) :=
by
  sorry

end NUMINAMATH_GPT_num_zeros_in_product_l1797_179700


namespace NUMINAMATH_GPT_arithmetic_geom_sequence_ratio_l1797_179764

theorem arithmetic_geom_sequence_ratio (a : ℕ → ℝ) (d a1 : ℝ) (h1 : d ≠ 0) 
(h2 : ∀ n, a (n+1) = a n + d)
(h3 : (a 0 + 2 * d)^2 = a 0 * (a 0 + 8 * d)):
  (a 0 + a 2 + a 8) / (a 1 + a 3 + a 9) = 13 / 16 := 
by sorry

end NUMINAMATH_GPT_arithmetic_geom_sequence_ratio_l1797_179764


namespace NUMINAMATH_GPT_value_of_y_l1797_179784

variable (x y : ℤ)

-- Define the conditions
def condition1 : Prop := 3 * (x^2 + x + 1) = y - 6
def condition2 : Prop := x = -3

-- Theorem to prove
theorem value_of_y (h1 : condition1 x y) (h2 : condition2 x) : y = 27 := by
  sorry

end NUMINAMATH_GPT_value_of_y_l1797_179784


namespace NUMINAMATH_GPT_fewest_tiles_needed_l1797_179779

theorem fewest_tiles_needed 
  (tile_len : ℝ) (tile_wid : ℝ) (region_len : ℝ) (region_wid : ℝ)
  (h_tile_dims : tile_len = 2 ∧ tile_wid = 3)
  (h_region_dims : region_len = 48 ∧ region_wid = 72) :
  (region_len * region_wid) / (tile_len * tile_wid) = 576 :=
by {
  sorry
}

end NUMINAMATH_GPT_fewest_tiles_needed_l1797_179779


namespace NUMINAMATH_GPT_sin_cos_identity_l1797_179796

theorem sin_cos_identity (x : ℝ) (h : Real.sin x = 4 * Real.cos x) : 
  Real.sin x * Real.cos x = 4 / 17 := by
  sorry

end NUMINAMATH_GPT_sin_cos_identity_l1797_179796


namespace NUMINAMATH_GPT_solve_x_l1797_179757

-- Define the structure of the pyramid
def pyramid (x : ℕ) : Prop :=
  let level1 := [x + 4, 12, 15, 18]
  let level2 := [x + 16, 27, 33]
  let level3 := [x + 43, 60]
  let top := x + 103
  top = 120

theorem solve_x : ∃ x : ℕ, pyramid x → x = 17 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_solve_x_l1797_179757


namespace NUMINAMATH_GPT_stratified_sampling_number_of_grade12_students_in_sample_l1797_179722

theorem stratified_sampling_number_of_grade12_students_in_sample 
  (total_students : ℕ)
  (students_grade10 : ℕ)
  (students_grade11_minus_grade12 : ℕ)
  (sampled_students_grade10 : ℕ)
  (total_students_eq : total_students = 1290)
  (students_grade10_eq : students_grade10 = 480)
  (students_grade11_minus_grade12_eq : students_grade11_minus_grade12 = 30)
  (sampled_students_grade10_eq : sampled_students_grade10 = 96) :
  ∃ n : ℕ, n = 78 :=
by
  -- Proof would go here, but we are skipping with "sorry"
  sorry

end NUMINAMATH_GPT_stratified_sampling_number_of_grade12_students_in_sample_l1797_179722


namespace NUMINAMATH_GPT_B_subset_A_A_inter_B_empty_l1797_179721

-- Definitions for the sets A and B
def A : Set ℝ := {x : ℝ | -1 < x ∧ x < 4}
def B (a : ℝ) : Set ℝ := {x : ℝ | 2 * a ≤ x ∧ x ≤ a + 3}

-- Proof statement for Part (1)
theorem B_subset_A (a : ℝ) : (∀ x, x ∈ B a → x ∈ A) ↔ (-1 / 2 < a ∧ a < 1) := sorry

-- Proof statement for Part (2)
theorem A_inter_B_empty (a : ℝ) : (∀ x, ¬(x ∈ A ∧ x ∈ B a)) ↔ (a ≤ -4 ∨ a ≥ 2) := sorry

end NUMINAMATH_GPT_B_subset_A_A_inter_B_empty_l1797_179721


namespace NUMINAMATH_GPT_find_p_l1797_179765

theorem find_p (p : ℕ) : 64^5 = 8^p → p = 10 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_find_p_l1797_179765


namespace NUMINAMATH_GPT_fraction_identity_l1797_179749

-- Definitions for conditions
variables (a b : ℚ)

-- The main statement to prove
theorem fraction_identity (h : a/b = 2/5) : (a + b) / b = 7 / 5 :=
by
  sorry

end NUMINAMATH_GPT_fraction_identity_l1797_179749


namespace NUMINAMATH_GPT_sheila_initial_savings_l1797_179727

noncomputable def initial_savings (monthly_savings : ℕ) (years : ℕ) (family_addition : ℕ) (total_amount : ℕ) : ℕ :=
  total_amount - (monthly_savings * 12 * years + family_addition)

def sheila_initial_savings_proof : Prop :=
  initial_savings 276 4 7000 23248 = 3000

theorem sheila_initial_savings : sheila_initial_savings_proof :=
  by
    -- Proof goes here
    sorry

end NUMINAMATH_GPT_sheila_initial_savings_l1797_179727


namespace NUMINAMATH_GPT_translate_vertex_to_increase_l1797_179723

def quadratic_function (x : ℝ) : ℝ := -x^2 + 1

theorem translate_vertex_to_increase (x : ℝ) :
  ∃ v, v = (2, quadratic_function 2) ∧
    (∀ x < 2, quadratic_function (x + 2) = quadratic_function x + 1 ∧
    ∀ x < 2, quadratic_function x < quadratic_function (x + 1)) :=
sorry

end NUMINAMATH_GPT_translate_vertex_to_increase_l1797_179723


namespace NUMINAMATH_GPT_minimum_s_value_l1797_179780

theorem minimum_s_value (x y z : ℝ) (h_pos : 0 < x ∧ 0 < y ∧ 0 < z) (h_cond : 3 * x^2 + 2 * y^2 + z^2 = 1) :
  ∃ (s : ℝ), s = 8 * Real.sqrt 6 ∧ ∀ (x' y' z' : ℝ), (0 < x' ∧ 0 < y' ∧ 0 < z' ∧ 3 * x'^2 + 2 * y'^2 + z'^2 = 1) → 
      s ≤ (1 + z') / (x' * y' * z') :=
sorry

end NUMINAMATH_GPT_minimum_s_value_l1797_179780


namespace NUMINAMATH_GPT_value_a8_l1797_179704

def sequence_sum (n : ℕ) : ℕ := n^2

def a (n : ℕ) : ℕ := sequence_sum n - sequence_sum (n - 1)

theorem value_a8 : a 8 = 15 :=
by
  sorry

end NUMINAMATH_GPT_value_a8_l1797_179704


namespace NUMINAMATH_GPT_arithmetic_seq_a11_l1797_179725

theorem arithmetic_seq_a11 (a : ℕ → ℕ) (S : ℕ → ℕ)
  (h1 : ∀ n, S n = n * (a 1 + a n) / 2)
  (h2 : S 21 = 105) : a 11 = 5 :=
sorry

end NUMINAMATH_GPT_arithmetic_seq_a11_l1797_179725


namespace NUMINAMATH_GPT_sum_of_two_primes_l1797_179762

theorem sum_of_two_primes (k : ℕ) (n : ℕ) (h : n = 1 + 10 * k) :
  (n = 1 ∨ ∃ p1 p2 : ℕ, Nat.Prime p1 ∧ Nat.Prime p2 ∧ n = p1 + p2) :=
by
  sorry

end NUMINAMATH_GPT_sum_of_two_primes_l1797_179762


namespace NUMINAMATH_GPT_carol_name_tag_l1797_179741

theorem carol_name_tag (a b c : ℕ) (ha : Prime a ∧ a ≥ 10 ∧ a < 100) (hb : Prime b ∧ b ≥ 10 ∧ b < 100) (hc : Prime c ∧ c ≥ 10 ∧ c < 100) 
  (h1 : b + c = 14) (h2 : a + c = 20) (h3 : a + b = 18) : c = 11 := 
by 
  sorry

end NUMINAMATH_GPT_carol_name_tag_l1797_179741
